// This file is part of MIDNIGHT-ZK.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! The Metal device context (macOS only).
//!
//! One process-wide context, made lazily at the first use ([`context`]): the
//! system default `MTLDevice`, one command queue, the shader library (the
//! embedded metallib, else the embedded source compiled at that moment, see
//! `build.rs`) and a cache of compute pipeline states by kernel name.
//!
//! # Use from a kernel module
//!
//! ```ignore
//! let mut s = ctx::lock()?;                  // serialises device work
//! let a = s.ctx().upload(scalars)?;           // a scoped, zeroed-on-drop buffer
//! let tw = s.scratch("ntt-twiddles", bytes)?; // a per-request buffer
//! let ns = s.ctx().run(|c| {
//!     c.dispatch("ntt_radix2", n / 2, &[Arg::Buffer(&a), Arg::value(&n)])?;
//!     Ok(())
//! })?.1;
//! a.read(out);
//! ```
//!
//! [`lock`] returns a [`Session`]: device work, the per-request buffers and
//! [`end_request`] are serialised by it, so [`end_request`] never zeroes a
//! buffer a kernel is using. A [`Ctx::run`] encodes any number of dispatches
//! into one command buffer (one serial compute encoder: each dispatch sees the
//! writes of the previous ones), commits it, waits for it and returns the GPU
//! time. A command buffer that ends in the error state is an
//! [`Error::Device`], never a panic.
//!
//! # Memory on unified-memory Apple GPUs
//!
//! Apple Silicon has one physical memory shared by the CPU and the GPU. Every
//! buffer here is `MTLResourceStorageModeShared`: its `contents()` pointer is
//! host-visible and coherent with the GPU at command-buffer boundaries, so a
//! "transfer" is a plain `memcpy` into or out of the buffer, with no DMA and
//! no staging copy. Reading results after [`Ctx::run`] returns needs no blit
//! and no synchronisation call (`didModifyRange` is a managed-mode concept).
//!
//! Wrapping a caller's slice without a copy
//! (`newBufferWithBytesNoCopy:length:options:deallocator:`) needs a
//! page-aligned start address and a length that is a multiple of the page
//! size (16 KiB on Apple Silicon). Slices from the prover are ordinary heap
//! allocations, so they are generally copied ([`Ctx::upload`],
//! [`Buffer::write`]); at memory bandwidth the copy is small next to the
//! kernels. A copy also keeps the hook contract simple: the caller's slice is
//! written only after the device result is complete, so a decline leaves it
//! unchanged.
//!
//! [`device_bytes`] counts the bytes this process holds in Metal buffers (now
//! and peak since [`begin_request`]), the idle buffers of the pool included;
//! [`in_use_bytes`] leaves them out. Metal's own figure for the device,
//! `currentAllocatedSize`, also includes the library, pipelines and driver
//! allocations.
//!
//! # Buffers
//!
//! - [`Buffer`]: owned by the caller, zeroed when dropped if it may hold a
//!   non-zero byte (written by the host or bound to a kernel since it was
//!   made or zeroed). Large zeroing runs on several threads.
//! - Pooled buffers ([`Ctx::alloc_pooled`]): a zeroed buffer from the pool,
//!   or a new one; when dropped it is zeroed and goes back to the pool
//!   instead of being released. [`release_later`] drops buffers on a
//!   background thread; [`end_request_report`] waits for it.
//! - Per-request buffers ([`Session::scratch`], and [`take_request_buffer`]
//!   for a call that fills its buffer without holding the device): kept by
//!   the context under a key until [`end_request_report`], which zeroes each
//!   one, with its check reads it back from the host mapping, and puts it
//!   back into the pool. A kernel uses them for scratch it wants to reuse
//!   across calls of one request.
//!
//! # Residency and first use
//!
//! The pool and the queue's residency set exist because the start of a
//! command buffer waits while Metal makes its buffers resident: for a new
//! buffer, and for one not used for about two seconds (see [`Ctx`]'s
//! residency set and the pool's text). Measured on an M4 Max, both cost 13
//! to 30 ms for the 448 MiB of the quotient's cosets at k = 20, at every
//! proof, before them.

use std::{
    collections::HashMap,
    ffi::c_void,
    marker::PhantomData,
    ptr::NonNull,
    sync::{
        Mutex, MutexGuard, OnceLock,
        atomic::{AtomicBool, AtomicU64, Ordering::Relaxed},
    },
};

use dispatch2::DispatchData;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSObjectProtocol, NSString};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily, MTLLanguageVersion, MTLLibrary,
    MTLResidencySet, MTLResidencySetDescriptor, MTLResource, MTLResourceOptions, MTLSize,
    MTLStorageMode,
};

use crate::MetalInfo;

// MTLCreateSystemDefaultDevice returns no device in a process that does not
// link CoreGraphics.
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

/// The Metal Shading Language version of the shaders (set by `build.rs`).
pub const MSL_VERSION: &str = env!("MIDNIGHT_METAL_MSL_VERSION");

#[cfg(midnight_metal_metallib)]
static METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/midnight_metal.metallib"));
static SOURCE: &str = include_str!(concat!(env!("OUT_DIR"), "/midnight_metal.metal"));

/// A device failure, mapped to a decline reason by [`Error::reason`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// No usable device, or the context could not be made.
    NoDevice(String),
    /// A buffer could not be allocated (or is above `maxBufferLength`).
    Memory(String),
    /// A pipeline, encoder or command buffer failed.
    Device(String),
}

impl Error {
    /// The decline reason of this error: `no-device`, `memory` or
    /// `device-error`.
    pub fn reason(&self) -> &'static str {
        match self {
            Error::NoDevice(_) => "no-device",
            Error::Memory(_) => "memory",
            Error::Device(_) => "device-error",
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::NoDevice(m) => write!(f, "no device: {m}"),
            Error::Memory(m) => write!(f, "memory: {m}"),
            Error::Device(m) => write!(f, "device error: {m}"),
        }
    }
}

type Device = Retained<ProtocolObject<dyn MTLDevice>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

// ---------------------------------------------------------------------------
// Detection

struct Probe {
    device: Device,
    info: MetalInfo,
}

static PROBE: OnceLock<Option<Probe>> = OnceLock::new();

fn probe_device() -> Option<&'static Probe> {
    PROBE
        .get_or_init(|| {
            let device = MTLCreateSystemDefaultDevice()?;
            let tg = device.maxThreadsPerThreadgroup();
            let apple_family = (1..=10u32)
                .rev()
                .find(|f| device.supportsFamily(MTLGPUFamily(1000 + *f as isize)))
                .unwrap_or(0);
            let info = MetalInfo {
                name: device.name().to_string(),
                registry_id: device.registryID(),
                unified_memory: device.hasUnifiedMemory(),
                recommended_max_working_set: device.recommendedMaxWorkingSetSize(),
                max_buffer_length: device.maxBufferLength() as u64,
                max_threads_per_threadgroup: tg.width as u64,
                apple_family,
                metal3: device.supportsFamily(MTLGPUFamily::Metal3),
                msl_version: MSL_VERSION,
            };
            Some(Probe { device, info })
        })
        .as_ref()
}

/// The device's properties, without making the context (no library is
/// loaded). `None`: no Metal device.
pub fn probe() -> Option<&'static MetalInfo> {
    probe_device().map(|p| &p.info)
}

// ---------------------------------------------------------------------------
// Context

/// The process-wide device context.
pub struct Ctx {
    device: Device,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    library_source: String,
    pipelines: Mutex<HashMap<String, Pipeline>>,
    /// The queue's residency set (see [`Resident`]); `None` where Metal has
    /// none (before macOS 15).
    resident: Option<Mutex<Resident>>,
}

/// Buffers of at least this many bytes are in the queue's residency set
/// while they live.
const RESIDENT_MIN: usize = 1 << 20;

/// The queue's residency set.
///
/// Metal makes a buffer resident for the GPU when a command buffer uses it,
/// and lets it go once it has not been used for about two seconds, even
/// while the GPU works on other buffers. Its next use then waits while it is
/// made resident again: measured on an M4 Max, two 448 MiB buffers unused
/// for 2 to 5 s while the GPU ran other work waited 13 to 19 ms from
/// `commit` to `GPUStartTime` (unused for 1 s: 0.3 ms). A proof takes about
/// 2 s, so the quotient's buffers, the transform buffers and the MSM tables
/// would be made resident again at every proof. A buffer in the queue's
/// residency set stays resident (in the same measurement: no wait). [`Ctx`]
/// adds every buffer of at least [`RESIDENT_MIN`] bytes when it is made and
/// removes it when it is released; an idle buffer of the pool stays in it.
/// The set also makes a new buffer resident when it is added, so its first
/// command buffer no longer waits for that either.
struct Resident {
    set: Retained<ProtocolObject<dyn MTLResidencySet>>,
}

// SAFETY: a residency set may be used from any thread; every change goes
// through the mutex around it.
unsafe impl Send for Resident {}

impl Ctx {
    fn residency(&self) -> Option<MutexGuard<'_, Resident>> {
        self.resident.as_ref().map(|r| r.lock().unwrap_or_else(|e| e.into_inner()))
    }

    /// Adds a new buffer to the residency set (when it is large enough).
    fn keep(&self, b: &ProtocolObject<dyn MTLBuffer>) {
        if b.length() >= RESIDENT_MIN
            && let Some(r) = self.residency()
        {
            r.set.addAllocation(ProtocolObject::from_ref(b));
            r.set.commit();
            r.set.requestResidency();
        }
    }
}

/// Removes a buffer that is being released from the residency set (when it
/// is large enough to be in it). A buffer is released by its last owner,
/// after the GPU work that used it.
fn unkeep(b: &ProtocolObject<dyn MTLBuffer>) {
    if b.length() >= RESIDENT_MIN
        && let Some(Ok(c)) = CTX.get()
        && let Some(r) = c.residency()
    {
        r.set.removeAllocation(ProtocolObject::from_ref(b));
        r.set.commit();
    }
}

impl std::fmt::Debug for Ctx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ctx")
            .field("device", &self.device.name().to_string())
            .field("library_source", &self.library_source)
            .finish_non_exhaustive()
    }
}

static CTX: OnceLock<Result<Ctx, String>> = OnceLock::new();

/// The context, made at the first call. The error of a failed attempt (no
/// device, a library that does not load) is kept: later calls return it
/// without trying again.
pub fn context() -> Result<&'static Ctx, Error> {
    CTX.get_or_init(make_context).as_ref().map_err(|e| Error::NoDevice(e.clone()))
}

/// Why the context could not be made, if it was tried and failed.
pub fn context_error() -> Option<&'static str> {
    match CTX.get() {
        Some(Err(e)) => Some(e),
        _ => None,
    }
}

fn make_context() -> Result<Ctx, String> {
    let probe = probe_device().ok_or("no Metal device")?;
    let device = probe.device.clone();
    let queue = device.newCommandQueue().ok_or("newCommandQueue returned nil")?;
    let (library, library_source) = load_library(&device)?;
    // the residency set of the large buffers (macOS 15+)
    let resident = if device.respondsToSelector(objc2::sel!(newResidencySetWithDescriptor:error:)) {
        let desc = MTLResidencySetDescriptor::new();
        device.newResidencySetWithDescriptor_error(&desc).ok().map(|set| {
            queue.addResidencySet(&set);
            Mutex::new(Resident { set })
        })
    } else {
        None
    };
    Ok(Ctx {
        device,
        queue,
        library,
        library_source,
        pipelines: Mutex::new(HashMap::new()),
        resident,
    })
}

fn load_library(
    device: &Device,
) -> Result<(Retained<ProtocolObject<dyn MTLLibrary>>, String), String> {
    let force_source = std::env::var("MIDNIGHT_METAL_SHADER_SOURCE").as_deref() == Ok("1");
    #[cfg(midnight_metal_metallib)]
    let how = if force_source {
        "source (MIDNIGHT_METAL_SHADER_SOURCE=1)".to_string()
    } else {
        let data = DispatchData::from_static_bytes(METALLIB);
        match device.newLibraryWithData_error(&data) {
            Ok(lib) => return Ok((lib, "metallib".into())),
            Err(e) => {
                let why = e.localizedDescription().to_string();
                eprintln!(
                    "midnight-metal: the embedded metallib did not load ({why}); \
                     compiling the embedded shader source"
                );
                format!("source (metallib refused: {why})")
            }
        }
    };
    #[cfg(not(midnight_metal_metallib))]
    let how = {
        let _ = (force_source, DispatchData::empty);
        "source (no metallib was built)".to_string()
    };
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(language_version()?);
    let lib = device
        .newLibraryWithSource_options_error(&NSString::from_str(SOURCE), Some(&options))
        .map_err(|e| format!("shader compile failed: {}", e.localizedDescription()))?;
    Ok((lib, how))
}

fn language_version() -> Result<MTLLanguageVersion, String> {
    let (major, minor) = MSL_VERSION
        .split_once('.')
        .and_then(|(a, b)| Some((a.parse::<usize>().ok()?, b.parse::<usize>().ok()?)))
        .ok_or_else(|| format!("bad MSL version {MSL_VERSION}"))?;
    Ok(MTLLanguageVersion((major << 16) + minor))
}

// ---------------------------------------------------------------------------
// Byte accounting

static BYTES_NOW: AtomicU64 = AtomicU64::new(0);
static BYTES_PEAK: AtomicU64 = AtomicU64::new(0);
/// Bytes in buffers that are not idle in the pool.
static IN_USE: AtomicU64 = AtomicU64::new(0);

fn count_alloc(bytes: u64) {
    let now = BYTES_NOW.fetch_add(bytes, Relaxed) + bytes;
    BYTES_PEAK.fetch_max(now, Relaxed);
    IN_USE.fetch_add(bytes, Relaxed);
}

/// `(now, peak)` bytes this process holds in Metal buffers of this crate,
/// the idle buffers of the pool included ([`pool_bytes`]). The peak is since
/// process start or the last [`begin_request`].
pub fn device_bytes() -> (u64, u64) {
    (BYTES_NOW.load(Relaxed), BYTES_PEAK.load(Relaxed))
}

/// Bytes in buffers in use now: [`device_bytes`] without the idle buffers of
/// the pool.
pub fn in_use_bytes() -> u64 {
    IN_USE.load(Relaxed)
}

/// Resets the peak of [`device_bytes`] to the current value.
pub fn begin_request() {
    BYTES_PEAK.store(BYTES_NOW.load(Relaxed), Relaxed);
}

// ---------------------------------------------------------------------------
// The pool of zeroed buffers

/// Zeroed buffers kept for reuse ([`Ctx::alloc_pooled`]).
///
/// A new shared buffer costs more than its allocation: the first command
/// buffer that uses it waits while its pages are made resident for the GPU
/// (18 to 22 ms for 448 MiB, measured on an M4 Max as the time from `commit`
/// to `GPUStartTime`, with no residency set; about 4 ms with it, which makes
/// a new buffer resident when it is added). A buffer that is resident starts
/// within 0.1 ms. So the large per-request and per-call buffers come from
/// this pool and go back to it, and stay in the residency set meanwhile.
///
/// Every buffer in the pool reads zero: a buffer goes back only after its
/// zeroing (on the CPU, of its whole length), so a buffer that held a
/// witness value is never handed out again, nor reported erased, before it
/// was zeroed. The pool keeps at most [`pool_limit`] bytes (a buffer that
/// would exceed it is released instead); [`release_pool`] releases every
/// idle buffer.
struct Pool {
    free: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    bytes: u64,
}

// SAFETY: Metal buffers may be used from any thread; the pool's buffers are
// idle (no host reference to their contents, no GPU work).
unsafe impl Send for Pool {}

static POOL: Mutex<Pool> = Mutex::new(Pool {
    free: Vec::new(),
    bytes: 0,
});
/// [`pool_limit`]: `u64::MAX` until read.
static POOL_LIMIT: AtomicU64 = AtomicU64::new(u64::MAX);

fn pool() -> MutexGuard<'static, Pool> {
    POOL.lock().unwrap_or_else(|e| e.into_inner())
}

/// The most bytes the pool keeps idle: [`set_pool_limit`], else
/// `MIDNIGHT_METAL_POOL_MIB`, else a quarter of the device's
/// `recommendedMaxWorkingSetSize` (0 without a device).
pub fn pool_limit() -> u64 {
    let l = POOL_LIMIT.load(Relaxed);
    if l != u64::MAX {
        return l;
    }
    let l = std::env::var("MIDNIGHT_METAL_POOL_MIB")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(|m| m.saturating_mul(1 << 20))
        .unwrap_or_else(|| probe().map_or(0, |i| i.recommended_max_working_set / 4));
    let _ = POOL_LIMIT.compare_exchange(u64::MAX, l, Relaxed, Relaxed);
    POOL_LIMIT.load(Relaxed)
}

/// Sets [`pool_limit`] in bytes (`None`: the default again) and releases
/// idle buffers above it.
pub fn set_pool_limit(limit: Option<u64>) {
    POOL_LIMIT.store(limit.map_or(u64::MAX, |l| l.min(u64::MAX - 1)), Relaxed);
    let limit = pool_limit();
    let freed: Vec<_> = {
        let mut p = pool();
        let mut out = Vec::new();
        while p.bytes > limit {
            let Some(b) = p.free.pop() else { break };
            p.bytes -= b.length() as u64;
            out.push(b);
        }
        out
    };
    release_raw(freed);
}

/// Bytes of the idle (zeroed) buffers in the pool.
pub fn pool_bytes() -> u64 {
    pool().bytes
}

fn release_raw(bufs: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>) -> u64 {
    let bytes: u64 = bufs.iter().map(|b| b.length() as u64).sum();
    for b in &bufs {
        unkeep(b);
    }
    drop(bufs);
    BYTES_NOW.fetch_sub(bytes, Relaxed);
    bytes
}

/// Releases every idle buffer of the pool (after the zeroing still running
/// in the background has ended, see [`release_later`]). Returns the bytes
/// released.
pub fn release_pool() -> u64 {
    settle_releases();
    let bufs = {
        let mut p = pool();
        p.bytes = 0;
        std::mem::take(&mut p.free)
    };
    release_raw(bufs)
}

/// A zeroed buffer of the pool of at least `bytes` bytes and at most a
/// quarter more (the smallest such), if there is one.
fn pool_take(bytes: usize) -> Option<Retained<ProtocolObject<dyn MTLBuffer>>> {
    let mut p = pool();
    let max = bytes.saturating_add(bytes / 4);
    let i = p
        .free
        .iter()
        .enumerate()
        .filter(|(_, b)| (bytes..=max).contains(&b.length()))
        .min_by_key(|(_, b)| b.length())?
        .0;
    let b = p.free.swap_remove(i);
    p.bytes -= b.length() as u64;
    IN_USE.fetch_add(b.length() as u64, Relaxed);
    Some(b)
}

/// Puts a zeroed buffer into the pool, or releases it when the pool would
/// exceed its limit.
fn pool_put(raw: Retained<ProtocolObject<dyn MTLBuffer>>) {
    let len = raw.length() as u64;
    IN_USE.fetch_sub(len, Relaxed);
    let limit = pool_limit();
    let back = {
        let mut p = pool();
        if p.bytes + len <= limit {
            p.bytes += len;
            p.free.push(raw);
            None
        } else {
            Some(raw)
        }
    };
    if let Some(b) = back {
        release_raw(vec![b]);
    }
}

/// Zeroing and release running on background threads ([`release_later`]).
static PENDING: Mutex<Vec<std::thread::JoinHandle<()>>> = Mutex::new(Vec::new());

/// Drops `bufs` (each zeroed, then pooled or released) on a background
/// thread, so that a caller does not wait for the zeroing. [`end_request`],
/// [`end_request_report`] and [`release_pool`] wait for it first: at the end
/// of a request every such buffer has been zeroed.
pub fn release_later(bufs: Vec<Buffer>) {
    if bufs.is_empty() {
        return;
    }
    let h = std::thread::spawn(move || drop(bufs));
    let mut p = PENDING.lock().unwrap_or_else(|e| e.into_inner());
    p.retain(|h| !h.is_finished());
    p.push(h);
}

fn releases_pending() -> bool {
    PENDING
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .any(|h| !h.is_finished())
}

/// Waits for the zeroing of every buffer given to [`release_later`].
pub fn settle_releases() {
    let hs = std::mem::take(&mut *PENDING.lock().unwrap_or_else(|e| e.into_inner()));
    for h in hs {
        let _ = h.join();
    }
}

// ---------------------------------------------------------------------------
// Buffers

/// Plain data that may be copied into and out of a buffer byte for byte:
/// every bit pattern is a valid value and there is no padding.
///
/// # Safety
/// Implement it only for such types.
pub unsafe trait Pod: Copy + 'static {}
// SAFETY: integers and arrays of them have no padding and no invalid values.
unsafe impl Pod for u8 {}
// SAFETY: as above.
unsafe impl Pod for u32 {}
// SAFETY: as above.
unsafe impl Pod for u64 {}
// SAFETY: as above.
unsafe impl<T: Pod, const N: usize> Pod for [T; N] {}

/// A shared-storage Metal buffer, zeroed when dropped.
pub struct Buffer {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    bytes: usize,
    /// Goes back to the pool when dropped (after its zeroing).
    pooled: bool,
    /// May hold a non-zero byte: written by the host or bound to a kernel
    /// since it was made or last zeroed.
    dirty: AtomicBool,
}

// SAFETY: Metal buffers may be used from any thread; host access to the
// contents goes through `&self` / `&mut self`, and GPU access happens only
// inside `Ctx::run`, which borrows the buffer and waits for the GPU.
unsafe impl Send for Buffer {}
// SAFETY: as above.
unsafe impl Sync for Buffer {}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer").field("bytes", &self.bytes).finish()
    }
}

/// Byte runs of at least this size are zeroed, copied or checked on several
/// threads: one core does not reach the memory bandwidth.
const PAR_BYTES: usize = 4 << 20;

/// The threads of a parallel host pass over memory.
fn mem_threads() -> usize {
    std::thread::available_parallelism().map_or(4, |t| t.get()).min(8)
}

/// `f(start, len)` over `[0, len)` in parts of at least [`PAR_BYTES`] (a
/// multiple of 64), on several threads; `true` when every part returned
/// `true`.
fn par_ranges(len: usize, f: impl Fn(usize, usize) -> bool + Sync) -> bool {
    if len < 2 * PAR_BYTES {
        return f(0, len);
    }
    let per = len.div_ceil(mem_threads()).max(PAR_BYTES).next_multiple_of(64);
    std::thread::scope(|s| {
        let hs: Vec<_> = (0..len)
            .step_by(per)
            .map(|a| {
                let f = &f;
                s.spawn(move || f(a, per.min(len - a)))
            })
            .collect();
        let oks: Vec<bool> = hs.into_iter().map(|h| h.join().unwrap_or(false)).collect();
        oks.into_iter().all(|ok| ok)
    })
}

impl Buffer {
    /// The length in bytes.
    pub fn len_bytes(&self) -> usize {
        self.bytes
    }

    fn ptr(&self) -> *mut u8 {
        self.raw.contents().as_ptr().cast()
    }

    /// Records that the buffer may now hold a non-zero byte.
    fn touch(&self) {
        self.dirty.store(true, Relaxed);
    }

    /// The contents as a slice of `T` (`len_bytes / size_of::<T>()`
    /// elements).
    pub fn as_slice<T: Pod>(&self) -> &[T] {
        // SAFETY: `contents` is page aligned, valid for `bytes` bytes, and no
        // GPU work writes it while `&self` is held outside `Ctx::run`.
        unsafe { std::slice::from_raw_parts(self.ptr().cast(), self.bytes / size_of::<T>()) }
    }

    /// The contents as a mutable slice of `T`.
    pub fn as_mut_slice<T: Pod>(&mut self) -> &mut [T] {
        self.touch();
        // SAFETY: as `as_slice`, and `&mut self` is exclusive.
        unsafe { std::slice::from_raw_parts_mut(self.ptr().cast(), self.bytes / size_of::<T>()) }
    }

    /// Copies `data` to the start of the buffer. Panics if it does not fit.
    pub fn write<T: Pod>(&mut self, data: &[T]) {
        self.as_mut_slice::<T>()[..data.len()].copy_from_slice(data);
    }

    /// Copies the start of the buffer into `out`. Panics if the buffer is
    /// shorter.
    pub fn read<T: Pod>(&self, out: &mut [T]) {
        out.copy_from_slice(&self.as_slice::<T>()[..out.len()]);
    }

    /// Writes zero over the whole buffer (its whole allocation, which may be
    /// longer than [`Buffer::len_bytes`]), on several threads when it is
    /// large.
    pub fn zero(&mut self) {
        let base = self.ptr() as usize;
        par_ranges(self.raw.length(), |a, n| {
            // SAFETY: valid for `raw.length()` bytes, exclusive (`&mut self`),
            // and the parts are disjoint.
            unsafe { std::ptr::write_bytes((base + a) as *mut u8, 0, n) };
            true
        });
        std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
        self.dirty.store(false, Relaxed);
    }

    /// The bytes [`Buffer::zero`] writes: the whole allocation.
    pub(crate) fn alloc_bytes(&self) -> usize {
        self.raw.length()
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if self.dirty.load(Relaxed) {
            self.zero();
        }
        if self.pooled {
            pool_put(self.raw.clone());
        } else {
            unkeep(&self.raw);
            IN_USE.fetch_sub(self.raw.length() as u64, Relaxed);
            BYTES_NOW.fetch_sub(self.raw.length() as u64, Relaxed);
        }
    }
}

impl Ctx {
    /// The library's origin: `metallib`, or `source` (with the reason when
    /// the embedded metallib was not used).
    pub fn library_source(&self) -> &str {
        &self.library_source
    }

    fn wrap(
        &self,
        raw: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
        bytes: usize,
        dirty: bool,
    ) -> Result<Buffer, Error> {
        let raw = raw.ok_or_else(|| Error::Memory(format!("buffer of {bytes} bytes")))?;
        count_alloc(raw.length() as u64);
        self.keep(&raw);
        Ok(Buffer {
            raw,
            bytes,
            pooled: false,
            dirty: AtomicBool::new(dirty),
        })
    }

    fn check_len(&self, bytes: usize) -> Result<usize, Error> {
        let max = self.device.maxBufferLength();
        if bytes > max {
            return Err(Error::Memory(format!(
                "{bytes} bytes is above maxBufferLength {max}"
            )));
        }
        // Metal refuses a zero-length buffer
        Ok(bytes.max(16))
    }

    /// A zero-filled shared buffer of `bytes` bytes.
    pub fn alloc(&self, bytes: usize) -> Result<Buffer, Error> {
        let len = self.check_len(bytes)?;
        let raw = self
            .device
            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared);
        self.wrap(raw, bytes, false)
    }

    /// A zero-filled shared buffer of `bytes` bytes from the pool of zeroed
    /// buffers (one at most a quarter longer), else a new one; it goes back
    /// to the pool when dropped, after its zeroing (see the pool's text in
    /// this module). For large buffers made again and again: one that the
    /// GPU used before starts without the mapping delay of a new one.
    pub fn alloc_pooled(&self, bytes: usize) -> Result<Buffer, Error> {
        let len = self.check_len(bytes)?;
        let mut took = pool_take(len);
        if took.is_none() && releases_pending() {
            // a buffer being zeroed in the background may be the one: its
            // zeroing ends sooner than a new buffer is mapped for the GPU
            settle_releases();
            took = pool_take(len);
        }
        let mut b = match took {
            Some(raw) => Buffer {
                raw,
                bytes,
                pooled: true,
                dirty: AtomicBool::new(false),
            },
            None => self.alloc(bytes)?,
        };
        b.pooled = true;
        Ok(b)
    }

    /// A shared buffer holding a copy of `data`.
    pub fn upload<T: Pod>(&self, data: &[T]) -> Result<Buffer, Error> {
        let bytes = size_of_val(data);
        let len = self.check_len(bytes)?;
        if bytes == 0 {
            return self.alloc(0);
        }
        // SAFETY: `data` is valid for `bytes` bytes; Metal copies them.
        let raw = unsafe {
            self.device.newBufferWithBytes_length_options(
                NonNull::new(data.as_ptr() as *mut c_void).unwrap(),
                len,
                MTLResourceOptions::StorageModeShared,
            )
        };
        self.wrap(raw, bytes, true)
    }

    /// The compute pipeline of a kernel, compiled at its first use and
    /// cached.
    pub fn pipeline(&self, kernel: &str) -> Result<Pipeline, Error> {
        let mut cache = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(p) = cache.get(kernel) {
            return Ok(p.clone());
        }
        let function = self
            .library
            .newFunctionWithName(&NSString::from_str(kernel))
            .ok_or_else(|| Error::Device(format!("no kernel `{kernel}` in the library")))?;
        let p = self.device.newComputePipelineStateWithFunction_error(&function).map_err(|e| {
            Error::Device(format!("pipeline `{kernel}`: {}", e.localizedDescription()))
        })?;
        cache.insert(kernel.to_string(), p.clone());
        Ok(p)
    }

    /// Encodes the dispatches of `f` into one command buffer, commits it and
    /// waits for it. Returns `f`'s value and the GPU time in nanoseconds. A
    /// command buffer that ends in error is [`Error::Device`]; an error from
    /// `f` is returned after the encoded work has completed, so no buffer is
    /// still in use when this returns.
    pub fn run<'a, R>(
        &'a self,
        f: impl FnOnce(&mut Command<'a>) -> Result<R, Error>,
    ) -> Result<(R, u64), Error> {
        objc2::rc::autoreleasepool(|_| {
            let cb = self
                .queue
                .commandBuffer()
                .ok_or_else(|| Error::Device("commandBuffer returned nil".into()))?;
            let enc = cb
                .computeCommandEncoder()
                .ok_or_else(|| Error::Device("computeCommandEncoder returned nil".into()))?;
            let mut cmd = Command {
                ctx: self,
                enc,
                _args: PhantomData,
            };
            let value = f(&mut cmd);
            cmd.enc.endEncoding();
            cb.commit();
            cb.waitUntilCompleted();
            let status = cb.status();
            if status != MTLCommandBufferStatus::Completed {
                let why = cb
                    .error()
                    .map(|e| format!("{} (code {})", e.localizedDescription(), e.code()))
                    .unwrap_or_else(|| format!("status {}", status.0));
                return Err(Error::Device(format!("command buffer failed: {why}")));
            }
            let ns = ((cb.GPUEndTime() - cb.GPUStartTime()) * 1e9).max(0.0) as u64;
            Ok((value?, ns))
        })
    }
}

// ---------------------------------------------------------------------------
// Dispatch

/// A kernel argument, bound at the index of its position in the list.
#[derive(Clone, Copy, Debug)]
pub enum Arg<'a> {
    /// A buffer from its start.
    Buffer(&'a Buffer),
    /// A buffer from a byte offset.
    BufferAt(&'a Buffer, usize),
    /// Bytes copied into the command (`setBytes`, at most 4 KiB): a
    /// `constant` value or small struct.
    Bytes(&'a [u8]),
}

impl<'a> Arg<'a> {
    /// A plain value as [`Arg::Bytes`].
    pub fn value<T: Pod>(v: &'a T) -> Self {
        // SAFETY: a Pod value is valid to view as its bytes.
        Arg::Bytes(unsafe { std::slice::from_raw_parts((v as *const T).cast(), size_of::<T>()) })
    }
}

/// The command buffer being encoded inside [`Ctx::run`]: one serial compute
/// encoder, so each dispatch sees the writes of the earlier ones.
pub struct Command<'a> {
    ctx: &'a Ctx,
    enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    _args: PhantomData<&'a Buffer>,
}

impl std::fmt::Debug for Command<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Command").finish_non_exhaustive()
    }
}

impl<'a> Command<'a> {
    fn bind(&self, kernel: &str, args: &[Arg<'a>], tg_mem: &[usize]) -> Result<Pipeline, Error> {
        let p = self.ctx.pipeline(kernel)?;
        self.enc.setComputePipelineState(&p);
        for (i, a) in args.iter().enumerate() {
            match *a {
                Arg::Buffer(b) => {
                    b.touch();
                    // SAFETY: the buffer outlives the command (`'a`), and
                    // `run` waits for the GPU before returning.
                    unsafe { self.enc.setBuffer_offset_atIndex(Some(&b.raw), 0, i) }
                }
                Arg::BufferAt(b, off) => {
                    b.touch();
                    if off > b.bytes {
                        return Err(Error::Device(format!(
                            "argument {i}: offset {off} past the buffer"
                        )));
                    }
                    // SAFETY: as above, and the offset is inside the buffer.
                    unsafe { self.enc.setBuffer_offset_atIndex(Some(&b.raw), off, i) }
                }
                Arg::Bytes(v) => {
                    if v.is_empty() || v.len() > 4096 {
                        return Err(Error::Device(format!(
                            "argument {i}: {} bytes inline",
                            v.len()
                        )));
                    }
                    // SAFETY: valid for `v.len()` bytes; Metal copies them.
                    unsafe {
                        self.enc.setBytes_length_atIndex(
                            NonNull::new(v.as_ptr() as *mut c_void).unwrap(),
                            v.len(),
                            i,
                        )
                    }
                }
            }
        }
        for (i, len) in tg_mem.iter().enumerate() {
            // SAFETY: the length is checked by Metal against the device limit
            // when the pipeline is dispatched.
            unsafe { self.enc.setThreadgroupMemoryLength_atIndex(*len, i) };
        }
        Ok(p)
    }

    /// `threads` threads in one dimension, in threadgroups of the pipeline's
    /// largest size up to 256 (the last group may be partial: Apple GPUs take
    /// non-uniform threadgroups). The kernel still checks its index against
    /// its length.
    pub fn dispatch(
        &mut self,
        kernel: &str,
        threads: usize,
        args: &[Arg<'a>],
    ) -> Result<(), Error> {
        let p = self.bind(kernel, args, &[])?;
        let tg = p.maxTotalThreadsPerThreadgroup().clamp(1, 256);
        self.threads(threads, tg);
        Ok(())
    }

    /// `threads` threads in groups of `group` threads, with threadgroup memory
    /// of `tg_mem[i]` bytes at `[[threadgroup(i)]]`.
    pub fn dispatch_with(
        &mut self,
        kernel: &str,
        threads: usize,
        group: usize,
        args: &[Arg<'a>],
        tg_mem: &[usize],
    ) -> Result<(), Error> {
        let p = self.bind(kernel, args, tg_mem)?;
        if group == 0 || group > p.maxTotalThreadsPerThreadgroup() {
            return Err(Error::Device(format!(
                "`{kernel}`: threadgroup of {group} threads (pipeline max {})",
                p.maxTotalThreadsPerThreadgroup()
            )));
        }
        self.threads(threads, group);
        Ok(())
    }

    /// `groups` whole threadgroups of `group` threads, with threadgroup
    /// memory as [`Command::dispatch_with`].
    pub fn dispatch_groups(
        &mut self,
        kernel: &str,
        groups: usize,
        group: usize,
        args: &[Arg<'a>],
        tg_mem: &[usize],
    ) -> Result<(), Error> {
        let p = self.bind(kernel, args, tg_mem)?;
        if group == 0 || group > p.maxTotalThreadsPerThreadgroup() {
            return Err(Error::Device(format!(
                "`{kernel}`: threadgroup of {group} threads (pipeline max {})",
                p.maxTotalThreadsPerThreadgroup()
            )));
        }
        if groups > 0 {
            self.enc.dispatchThreadgroups_threadsPerThreadgroup(size1(groups), size1(group));
        }
        Ok(())
    }

    fn threads(&self, threads: usize, group: usize) {
        if threads > 0 {
            self.enc.dispatchThreads_threadsPerThreadgroup(size1(threads), size1(group));
        }
    }
}

fn size1(width: usize) -> MTLSize {
    MTLSize {
        width,
        height: 1,
        depth: 1,
    }
}

// ---------------------------------------------------------------------------
// Sessions and per-request buffers

#[derive(Default)]
struct State {
    scratch: HashMap<&'static str, Buffer>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

/// Exclusive use of the device: the context and the per-request buffers.
pub struct Session {
    ctx: &'static Ctx,
    state: MutexGuard<'static, Option<State>>,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session").finish_non_exhaustive()
    }
}

/// Takes the device (waiting for the current holder) after making the
/// context if needed.
pub fn lock() -> Result<Session, Error> {
    let ctx = context()?;
    let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    state.get_or_insert_with(State::default);
    Ok(Session { ctx, state })
}

impl Session {
    /// The context.
    pub fn ctx(&self) -> &'static Ctx {
        self.ctx
    }

    fn st(&mut self) -> &mut State {
        self.state.get_or_insert_with(State::default)
    }

    /// The per-request buffer `key`, of at least `bytes` bytes: the existing
    /// one with its contents if it is long enough, else a zero-filled one
    /// from the pool of zeroed buffers or a new one (the shorter one is
    /// zeroed and goes back to the pool, see [`Ctx::alloc_pooled`]).
    pub fn scratch(&mut self, key: &'static str, bytes: usize) -> Result<&mut Buffer, Error> {
        let ctx = self.ctx;
        let st = self.st();
        if st.scratch.get(key).is_some_and(|b| b.bytes < bytes) {
            st.scratch.remove(key);
        }
        if !st.scratch.contains_key(key) {
            st.scratch.insert(key, ctx.alloc_pooled(bytes)?);
        }
        Ok(st.scratch.get_mut(key).unwrap())
    }

    /// The per-request buffer `key`, if it exists.
    pub fn get_scratch(&self, key: &'static str) -> Option<&Buffer> {
        self.state.as_ref()?.scratch.get(key)
    }

    /// Several per-request buffers at once (for binding them together).
    pub fn scratch_many<const N: usize>(&self, keys: [&'static str; N]) -> Option<[&Buffer; N]> {
        let st = self.state.as_ref()?;
        let mut out = [None; N];
        for (o, k) in out.iter_mut().zip(keys) {
            *o = Some(st.scratch.get(k)?);
        }
        Some(out.map(|b| b.unwrap()))
    }

    /// Bytes held in per-request buffers.
    pub fn scratch_bytes(&self) -> u64 {
        self.state.as_ref().map_or(0, |s| {
            s.scratch.values().map(|b| b.raw.length() as u64).sum()
        })
    }
}

/// Per-request buffers checked out by a call for its own use
/// ([`take_request_buffer`]), by key: the ones given back and idle.
struct Lendable {
    /// The request of the buffers ([`end_request_report`] increments it).
    epoch: u64,
    free: Vec<(&'static str, Buffer)>,
}

static LENDABLE: Mutex<Lendable> = Mutex::new(Lendable {
    epoch: 0,
    free: Vec::new(),
});

fn lendable() -> MutexGuard<'static, Lendable> {
    LENDABLE.lock().unwrap_or_else(|e| e.into_inner())
}

/// A per-request buffer under `key` taken without the device: the call that
/// takes it uses it alone, for example to fill it on the CPU while another
/// call holds the device, and gives it back with [`RequestBuffer::give_back`]
/// (dropped instead, it is zeroed and goes back to the pool). It is the idle
/// one of at least `bytes` bytes that an earlier call of the request gave
/// back, with its contents, else a zero-filled one from the pool or a new
/// one. [`end_request_report`] zeroes, checks and counts the idle ones as it
/// does [`Session::scratch`]; one given back after its request ended is
/// zeroed then.
pub fn take_request_buffer(key: &'static str, bytes: usize) -> Result<RequestBuffer, Error> {
    let (epoch, found) = {
        let mut l = lendable();
        let epoch = l.epoch;
        let found = l
            .free
            .iter()
            .enumerate()
            .filter(|(_, (k, b))| *k == key && b.bytes >= bytes)
            .min_by_key(|(_, (_, b))| b.bytes)
            .map(|(i, _)| i)
            .map(|i| l.free.swap_remove(i).1);
        (epoch, found)
    };
    let buf = match found {
        Some(b) => b,
        None => context()?.alloc_pooled(bytes)?,
    };
    Ok(RequestBuffer {
        key,
        epoch,
        buf: Some(buf),
    })
}

/// A per-request buffer checked out by [`take_request_buffer`].
#[derive(Debug)]
pub struct RequestBuffer {
    key: &'static str,
    epoch: u64,
    buf: Option<Buffer>,
}

impl RequestBuffer {
    /// The buffer.
    pub fn buf(&self) -> &Buffer {
        self.buf.as_ref().expect("a buffer until given back")
    }

    /// The buffer, mutably.
    pub fn buf_mut(&mut self) -> &mut Buffer {
        self.buf.as_mut().expect("a buffer until given back")
    }

    /// Gives the buffer back to its request (its contents stay until
    /// [`end_request_report`] zeroes it); after the end of its request it is
    /// zeroed and goes back to the pool.
    pub fn give_back(mut self) {
        let b = self.buf.take().expect("a buffer until given back");
        let mut l = lendable();
        if l.epoch == self.epoch {
            l.free.push((self.key, b));
        } else {
            drop(l);
            drop(b);
        }
    }
}

/// Test switch of [`end_request_report`]: write one non-zero byte into the
/// first region after its zeroing, so that a test sees a failed check.
static PLANT_NONZERO: AtomicBool = AtomicBool::new(false);

/// Makes every later [`end_request_report`] write a non-zero byte into its
/// first region after the zeroing (`true`), or stops it (`false`): a test
/// switch for the read-back check, not for use outside tests.
#[doc(hidden)]
pub fn set_plant_nonzero(on: bool) {
    PLANT_NONZERO.store(on, Relaxed);
}

/// Whether every byte of `b`'s allocation reads zero, read word by word with
/// volatile loads from the host mapping (on several threads when it is
/// large).
fn reads_zero(b: &Buffer) -> bool {
    let base = b.ptr() as usize;
    par_ranges(b.alloc_bytes(), |a, n| {
        let p = (base + a) as *const u8;
        let words = n / 8;
        // SAFETY: `contents` is valid for the allocation's bytes; `a` is a
        // multiple of 64 and `contents` page aligned, so `p` is 8-byte
        // aligned; the reads stay inside the part.
        unsafe {
            (0..words).all(|i| std::ptr::read_volatile(p.cast::<u64>().add(i)) == 0)
                && (words * 8..n).all(|i| std::ptr::read_volatile(p.add(i)) == 0)
        }
    })
}

/// Zeroes every per-request buffer after any device work in progress (it
/// takes the device) and after the zeroing still running in the background
/// ([`release_later`]), with `check` reads it back from the host mapping and
/// compares it with zero, and puts it back into the pool of zeroed buffers
/// (or releases it, see [`Ctx::alloc_pooled`]). A region is a buffer's
/// whole allocation: its bytes are the bytes zeroed (and read back). On
/// unified memory a shared buffer is the memory the GPU used, so the
/// read-back sees what the device would read. `Err`: a region read back
/// non-zero (it is zeroed again before it goes back) or could not be read
/// from the host. There is no quarantine and no retry: a failed region is
/// still zeroed again and kept, and the failure is reported.
pub fn end_request_report(check: bool) -> Result<crate::WipeStats, crate::WipeStats> {
    let mut w = crate::WipeStats {
        checked: check,
        ..crate::WipeStats::default()
    };
    let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    settle_releases();
    let lent: Vec<Buffer> = {
        let mut l = lendable();
        l.epoch += 1;
        l.free.drain(..).map(|(_, b)| b).collect()
    };
    let scratch: Vec<Buffer> = state
        .as_mut()
        .map(|st| st.scratch.drain().map(|(_, b)| b).collect())
        .unwrap_or_default();
    let plant = PLANT_NONZERO.load(Relaxed);
    for (i, mut b) in scratch.into_iter().chain(lent).enumerate() {
        b.zero();
        w.regions += 1;
        w.bytes += b.alloc_bytes() as u64;
        if plant && i == 0 && b.bytes > 0 {
            let mid = b.bytes / 2;
            b.as_mut_slice::<u8>()[mid] = 0xa5;
        }
        if check {
            w.checked_regions += 1;
            if b.raw.storageMode() != MTLStorageMode::Shared {
                w.check_failures += 1;
            } else if !reads_zero(&b) {
                w.nonzero_regions += 1;
            }
        }
        // zeroed again on drop if a byte was written since
        drop(b);
    }
    if w.nonzero_regions == 0 && w.check_failures == 0 {
        Ok(w)
    } else {
        Err(w)
    }
}

/// [`end_request_report`] without the read-back. `true` when done.
pub fn end_request() -> bool {
    end_request_report(false).is_ok()
}

/// `(free, total)`: `total` is `recommendedMaxWorkingSetSize`, `free` is
/// that minus `currentAllocatedSize`, plus the idle buffers of the pool
/// ([`pool_bytes`]: they are reused before any new allocation, and
/// [`release_pool`] frees them; see [`crate::mem_info`]).
pub fn mem_info() -> Option<(usize, usize)> {
    let ctx = context().ok()?;
    let total = ctx.device.recommendedMaxWorkingSetSize() as usize;
    let used = ctx.device.currentAllocatedSize();
    let idle = pool_bytes() as usize;
    Some(((total + idle).saturating_sub(used).min(total), total))
}
