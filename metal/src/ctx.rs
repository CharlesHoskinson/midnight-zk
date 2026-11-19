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
//! and peak since [`begin_request`]). Metal's own figure for the device,
//! `currentAllocatedSize`, also includes the library, pipelines and driver
//! allocations.
//!
//! # Buffers
//!
//! - [`Buffer`]: owned by the caller, zeroed when dropped.
//! - Per-request buffers ([`Session::scratch`]): kept by the context under a
//!   key until [`end_request_report`], which zeroes each one, with its check
//!   reads it back from the host mapping, and releases it. A kernel uses them
//!   for scratch it wants to reuse across calls of one request.

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
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily, MTLLanguageVersion, MTLLibrary,
    MTLResource, MTLResourceOptions, MTLSize, MTLStorageMode,
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
    Ok(Ctx {
        device,
        queue,
        library,
        library_source,
        pipelines: Mutex::new(HashMap::new()),
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

fn count_alloc(bytes: u64) {
    let now = BYTES_NOW.fetch_add(bytes, Relaxed) + bytes;
    BYTES_PEAK.fetch_max(now, Relaxed);
}

/// `(now, peak)` bytes this process holds in Metal buffers of this crate. The
/// peak is since process start or the last [`begin_request`].
pub fn device_bytes() -> (u64, u64) {
    (BYTES_NOW.load(Relaxed), BYTES_PEAK.load(Relaxed))
}

/// Resets the peak of [`device_bytes`] to the current value.
pub fn begin_request() {
    BYTES_PEAK.store(BYTES_NOW.load(Relaxed), Relaxed);
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

impl Buffer {
    /// The length in bytes.
    pub fn len_bytes(&self) -> usize {
        self.bytes
    }

    fn ptr(&self) -> *mut u8 {
        self.raw.contents().as_ptr().cast()
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

    /// Writes zero over the whole buffer.
    pub fn zero(&mut self) {
        // SAFETY: valid for `bytes` bytes, exclusive.
        unsafe { std::ptr::write_bytes(self.ptr(), 0, self.bytes) };
        std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        self.zero();
        BYTES_NOW.fetch_sub(self.raw.length() as u64, Relaxed);
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
    ) -> Result<Buffer, Error> {
        let raw = raw.ok_or_else(|| Error::Memory(format!("buffer of {bytes} bytes")))?;
        count_alloc(raw.length() as u64);
        Ok(Buffer { raw, bytes })
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
        self.wrap(raw, bytes)
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
        self.wrap(raw, bytes)
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
                // SAFETY: the buffer outlives the command (`'a`), and `run`
                // waits for the GPU before returning.
                Arg::Buffer(b) => unsafe { self.enc.setBuffer_offset_atIndex(Some(&b.raw), 0, i) },
                Arg::BufferAt(b, off) => {
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
    /// one with its contents if it is long enough, else a new zero-filled one
    /// (the shorter one is zeroed and released).
    pub fn scratch(&mut self, key: &'static str, bytes: usize) -> Result<&mut Buffer, Error> {
        let ctx = self.ctx;
        let st = self.st();
        if st.scratch.get(key).is_some_and(|b| b.bytes < bytes) {
            st.scratch.remove(key);
        }
        if !st.scratch.contains_key(key) {
            st.scratch.insert(key, ctx.alloc(bytes)?);
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

/// Whether the first `bytes` bytes of `b` read zero, read word by word with
/// volatile loads from the host mapping.
fn reads_zero(b: &Buffer) -> bool {
    let p = b.ptr();
    let words = b.bytes / 8;
    // SAFETY: `contents` is valid for `bytes` bytes and 8-byte aligned (page
    // aligned); the reads stay inside it.
    unsafe {
        (0..words).all(|i| std::ptr::read_volatile(p.cast::<u64>().add(i)) == 0)
            && (words * 8..b.bytes).all(|i| std::ptr::read_volatile(p.add(i)) == 0)
    }
}

/// Zeroes every per-request buffer after any device work in progress (it
/// takes the device), with `check` reads it back from the host mapping and
/// compares it with zero, and releases it. On unified memory a shared buffer
/// is the memory the GPU used, so the read-back sees what the device would
/// read. `Err`: a region read back non-zero (it is zeroed again before its
/// release) or could not be read from the host. There is no quarantine and
/// no retry: a failed region is still released, and the failure is reported.
pub fn end_request_report(check: bool) -> Result<crate::WipeStats, crate::WipeStats> {
    let mut w = crate::WipeStats {
        checked: check,
        ..crate::WipeStats::default()
    };
    let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(st) = state.as_mut() {
        let plant = PLANT_NONZERO.load(Relaxed);
        for (i, (_, mut b)) in st.scratch.drain().enumerate() {
            b.zero();
            w.regions += 1;
            w.bytes += b.bytes as u64;
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
            // zeroed again on drop
            drop(b);
        }
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
/// that minus `currentAllocatedSize` (see [`crate::mem_info`]).
pub fn mem_info() -> Option<(usize, usize)> {
    let ctx = context().ok()?;
    let total = ctx.device.recommendedMaxWorkingSetSize() as usize;
    let used = ctx.device.currentAllocatedSize();
    Some((total.saturating_sub(used), total))
}
