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

//! The quotient on the device, and the resident proving-key copies: a port of
//! the CUDA crate's `gpu/src/quotient.rs` and `gpu/cuda/quotient.cuh`.
//!
//! [`crate::quotient`] has checked that `job` is not null and that a device
//! is usable before it calls [`quotient`]. Everything else is this module's:
//! the ABI version and layout of `midnight_proofs::gpu::abi::QuotientJobAbi`
//! (a copy in [`abi`]), the extended size against the quotient threshold
//! (`quotient` of [`crate::thresholds`]), the job's shape, the
//! reduction of its scalars and the memory need. It ends with exactly one of
//! `account::served` or `account::decline`, and writes `out` only with the
//! complete result.
//!
//! # Steps
//!
//! 1. Checks on the header and the tables ([`checked`], [`tables`]), in the
//!    order and with the reasons of the CUDA crate.
//! 2. The proving-key polynomials: served from the resident copy of a
//!    registered handle ([`keys`]) or uploaded for the call. They are placed
//!    before their digests are checked ([`keys::plan`]); CPU threads copy
//!    the uploads, refill the empty slots of the resident copy and hash the
//!    registered polynomials to check (every one outside a request, the ones
//!    not yet checked in the request inside one, and the bytes copied into
//!    every slot refilled) while the device runs step 3, and
//!    [`keys::settle`] marks a refilled slot present only when its digest
//!    matched and drops a stale polynomial (uploaded instead) before step 4,
//!    the first kernel that reads a slot.
//! 3. The proof's polynomials (advice, instance, permutation accumulators,
//!    lookup aggregators, multiplicities and helpers, trash) to the extended
//!    coset: each one is copied into a buffer of `2^k` coefficients (checked
//!    reduced on the way; the copy of the next one runs on the CPU while the
//!    device transforms this one) and moved to the coset by
//!    `ntt::coeff_to_extended_dev_at`, which writes it straight into its
//!    place in the flat buffer of the proof's cosets: one command buffer per
//!    polynomial.
//! 4. The numerator kernel (`shaders/quotient.metal`) over row ranges, one
//!    command buffer per range, each about [`TARGET_NANOS`] of GPU time.
//! 5. The division by the vanishing polynomial (`ntt::encode_mul_periodic`
//!    with the job's `t_inv`), the inverse transform and the removal of the
//!    coset factors, in the command buffers of the inverse transform
//!    (`ntt::divide_extended_to_coeff_dev`). Meanwhile CPU threads zero the
//!    cosets, the uploads and the scratch slots and put them back into the
//!    pool (`ctx::release_later`; `end_request` waits for them).
//! 6. The first `len` coefficients are copied into `out`, after every step
//!    that can fail or panic (`out` is written last, by a copy that cannot
//!    fail). No extended-domain vector goes back.
//!
//! Every buffer that holds a witness-dependent value is zeroed before it is
//! used again or reported erased: the transform scratch is per-request
//! scratch (`quotient-tmp`, zeroed by [`crate::end_request`]), every other
//! buffer is owned by the call, comes from the pool of zeroed buffers and is
//! zeroed before it goes back to it (in the background, ended by
//! [`crate::end_request`] at the latest).
//!
//! # Departures from the CUDA backend
//!
//! - A Metal kernel takes at most 31 buffer arguments (Metal feature set
//!   tables, every Apple family) and the device context binds buffers, not GPU
//!   addresses, so the CUDA job's tables of device pointers are flat buffers
//!   with index tables: the proof's cosets one buffer, the proving-key
//!   polynomials two (the lent resident copy, the uploads of the call), the
//!   programs and descriptors a buffer of words, the uniforms a buffer of
//!   field elements (see `shaders/include/quotient.h`). A job whose flat
//!   buffer would exceed `maxBufferLength` declines `memory`.
//! - The CUDA kernel runs one grid-stride launch over the whole domain; here
//!   the domain is split into row ranges of about [`TARGET_NANOS`] of GPU time
//!   each (the first range sized from an estimate, the next ones from the
//!   measured time per row), so no command buffer runs long.
//! - The scratch of the programs' intermediates is the CUDA layout
//!   (`scratch[slot * threads + thread]`) with [`threads`] threads per
//!   dispatch (`MIDNIGHT_METAL_QUOTIENT_THREADS`, the counterpart of
//!   `MIDNIGHT_GPU_QUOTIENT_THREADS`).
//! - The proving-key polynomials are checked reduced when they are copied (the
//!   CUDA crate checks only the proof's polynomials); an unreduced one
//!   declines `unreduced`.
//! - A resident copy is one buffer per [`KeyHandle`] and polynomial length;
//!   a job whose key polynomials are registered under several handles reads
//!   the handle with the most of them from its copy for the job's extended
//!   length and uploads the others for the call.
//! - Outside a request (no [`crate::begin_request`] open) every serve checks
//!   the digests of the registered polynomials it reads; the CUDA crate
//!   checks once per request only.
//! - The memory limit is the device's free working set
//!   (`recommendedMaxWorkingSetSize - currentAllocatedSize`) and
//!   `maxBufferLength` for each buffer, or a test limit.

pub mod abi;
mod keys;
mod tables;

use std::{
    ffi::c_void,
    sync::{
        Mutex,
        atomic::{AtomicU32, AtomicU64, Ordering::Relaxed},
    },
    time::Instant,
};

use blst::{blst_fr, blst_fr_mul, blst_fr_sqr};
pub use keys::{
    KeyHandle, evict_key_copies, keep_resident, key_stale, keys_registered, register_key,
    set_keep_resident,
};
pub(crate) use keys::{begin_request, end_request};

use self::{abi::QuotientJobAbi, keys::Place, tables::Tables};
use crate::{
    MAX_LOG_N, Scalar,
    account::{self, Class, Reason},
    ctx::{self, Arg, Buffer, Ctx, Error, Pod},
    field,
    ntt::{self, Dir},
};

const SCALAR_BYTES: usize = 32;

/// The GPU time one command buffer of the numerator aims at.
pub const TARGET_NANOS: u64 = 250_000_000;

/// The Montgomery form of one (`2^256 mod r`).
const ONE: Scalar = [
    0x0000_0001_ffff_fffe,
    0x5884_b7fa_0003_4802,
    0x998c_4fef_ecbc_4ff5,
    0x1824_b159_acc5_056f,
];

fn mul(a: &Scalar, b: &Scalar) -> Scalar {
    let mut r = blst_fr::default();
    // SAFETY: valid pointers to blst_fr values.
    unsafe { blst_fr_mul(&mut r, &blst_fr { l: *a }, &blst_fr { l: *b }) };
    r.l
}

fn square(a: &Scalar) -> Scalar {
    let mut r = blst_fr::default();
    // SAFETY: as `mul`.
    unsafe { blst_fr_sqr(&mut r, &blst_fr { l: *a }) };
    r.l
}

fn pow(b: &Scalar, mut e: u64) -> Scalar {
    let (mut r, mut b) = (ONE, *b);
    while e != 0 {
        if e & 1 == 1 {
            r = mul(&r, &b);
        }
        b = square(&b);
        e >>= 1;
    }
    r
}

// ---------------------------------------------------------------------------
// Counters and settings

static RUNS: AtomicU64 = AtomicU64::new(0);
static DECLINES: AtomicU64 = AtomicU64::new(0);
static THREADS: AtomicU32 = AtomicU32::new(0);
static MEMORY_LIMIT: AtomicU64 = AtomicU64::new(u64::MAX);
static TARGET: AtomicU64 = AtomicU64::new(TARGET_NANOS);

/// (quotients computed on the device, quotient jobs declined) since process
/// start.
pub fn quotient_counts() -> (u64, u64) {
    (RUNS.load(Relaxed), DECLINES.load(Relaxed))
}

/// The default of [`threads`]. Numerator GPU time at `2^21` rows on an M4
/// Max (40 GPU cores, rc1 chain circuit): 53 ms with 2^12 threads, 18.8 ms
/// with 2^14, 15.5 ms with 2^16, 14.4 ms with 2^18, 17.4 ms with 2^20; 2^16
/// is within 8% of the best and keeps the scratch at `nslots * 2 MiB`.
const DEFAULT_THREADS: u32 = 1 << 16;

/// The threads of one numerator dispatch, the stride of the scratch slots:
/// `MIDNIGHT_METAL_QUOTIENT_THREADS`, else 2^16; a multiple of 256 in
/// `[256, 2^24]`.
pub fn threads() -> u32 {
    let t = THREADS.load(Relaxed);
    if t != 0 {
        return t;
    }
    let t = std::env::var("MIDNIGHT_METAL_QUOTIENT_THREADS")
        .ok()
        .and_then(|e| e.trim().parse::<u64>().ok())
        .map_or(DEFAULT_THREADS, |t| {
            (t / 256 * 256).clamp(256, 1 << 24) as u32
        });
    let _ = THREADS.compare_exchange(0, t, Relaxed, Relaxed);
    THREADS.load(Relaxed)
}

/// The dispatch threads (the scratch stride) for an extended domain of
/// `2^ext_k` rows.
fn stride(ext_k: u32) -> usize {
    (threads() as usize).min(1usize << ext_k)
}

/// Every device allocation one quotient call can make, in bytes ([`run`]
/// allocates exactly these buffers, each at most once; Metal makes a buffer
/// of at least 16 bytes). Their sum is an upper bound of the device bytes
/// the call adds, whatever the pool of zeroed buffers gives back meanwhile.
/// Used by [`run`]'s admission and by [`need_bytes`], so the two cannot
/// drift.
#[derive(Clone, Copy, Debug)]
struct Need {
    /// One extended coset: `2^ext_k * 32`.
    ebytes: usize,
    /// One coefficient buffer of the coset phase: `2^k * 32` (two of them).
    coeff_bytes: usize,
    /// The proof's polynomials (one flat buffer of cosets).
    nproof: usize,
    /// Key polynomials uploaded for the call at the plan.
    ntemp: usize,
    /// The resident copy, when the call allocates it (0: it exists).
    lent_new: usize,
    /// Polynomials of the second upload after a stale key at [`keys::settle`]
    /// (0: no polynomial placed in a slot is checked, so none can move).
    restale: usize,
    /// The programs' scratch slots.
    slots_bytes: usize,
    /// The transform scratch, when the call allocates it.
    tmp_new: usize,
    /// The programs' words and uniforms.
    words_bytes: usize,
    uniforms_bytes: usize,
    /// `t_inv` (the inverse phase only).
    t_inv_bytes: usize,
    /// Whether the call divides and transforms back (not the numerator test
    /// path).
    inverse: bool,
}

impl Need {
    fn buffers(&self) -> [usize; 14] {
        let a = |b: usize| b.max(16);
        let e = self.ebytes;
        let opt = |b: usize| if b > 0 { a(b) } else { 0 };
        [
            opt(self.tmp_new),
            a(self.nproof.max(1) * e),
            a(self.coeff_bytes),
            a(self.coeff_bytes),
            a(self.ntemp * e),
            opt(self.lent_new),
            // the dummy argument
            16,
            opt(self.restale * e),
            // the numerator
            a(e),
            a(self.slots_bytes),
            a(self.words_bytes),
            a(self.uniforms_bytes),
            if self.inverse { a(self.t_inv_bytes) } else { 0 },
            // the scaled twiddle table of the inverse transform
            if self.inverse { ntt::SCALE_TABLE_BYTES } else { 0 },
        ]
    }

    /// The sum of the call's allocations.
    fn total(&self) -> usize {
        self.buffers().iter().sum()
    }

    /// The largest single buffer (against `maxBufferLength`).
    fn biggest(&self) -> usize {
        self.buffers().into_iter().max().unwrap_or(0)
    }
}

/// An upper bound of the device bytes one quotient job allocates: the
/// [`Need`] of [`run`] at its worst over what the shape does not fix (the
/// base size `k <= ext_k`, which key polynomials sit in a resident copy that
/// the call makes, a stale key uploaded again after the check). It holds
/// every buffer of the call: the proof's cosets, the coefficient buffers of
/// the coset phase, the key uploads, the resident copy, the second upload
/// after a stale key, the numerator, the transform scratch, the programs'
/// scratch slots (`nslots * stride * 32`, stride [`threads`] capped at
/// `2^ext_k`), `t_inv` and the scaled twiddle table of the inverse.
///
/// Two terms are outside the shape and assumed: the resident copy holds the
/// job's key polynomials only (a handle registered with the job's
/// `gpu_resident_polys`, as the prover binding does; a handle holding more
/// polynomials of this length lends all of them), and the programs' words
/// and uniforms (a few KiB) are not counted beyond their 16-byte minimum.
/// `test_support::need_bytes_job` gives the exact bound of a job.
///
/// The arguments are the fields of `midnight_proofs::gpu::QuotientShape`.
pub fn need_bytes(ext_k: u32, proof_polys: usize, key_polys: usize, nslots: u32) -> usize {
    let e = SCALAR_BYTES << ext_k;
    let mut worst = 0;
    for k in 1..=ext_k.max(1) {
        for resident in 0..=key_polys {
            let n = Need {
                ebytes: e,
                coeff_bytes: SCALAR_BYTES << k,
                nproof: proof_polys,
                ntemp: key_polys - resident,
                lent_new: resident * e,
                restale: if resident > 0 { key_polys } else { 0 },
                slots_bytes: (nslots as usize).max(1) * stride(ext_k) * SCALAR_BYTES,
                tmp_new: ntt::scratch_bytes(ext_k),
                words_bytes: 0,
                uniforms_bytes: 0,
                t_inv_bytes: SCALAR_BYTES << ext_k.saturating_sub(k),
                inverse: true,
            };
            worst = worst.max(n.total());
        }
    }
    worst
}

// ---------------------------------------------------------------------------
// Measurement and test support

/// Wall and GPU nanoseconds of the stages of the last quotient served.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QuotientStages {
    /// Checks, tables, key placement and allocations (wall).
    pub prepare: u64,
    /// Proving-key uploads and resident copies (wall; on the CPU while the
    /// coset transforms run).
    pub keys: u64,
    /// The proof's polynomials to the extended coset: copies, transforms and
    /// placement (wall).
    pub cosets: u64,
    /// The numerator kernel (wall).
    pub numerator: u64,
    /// The division by the vanishing polynomial (wall): 0, it is encoded in
    /// the first command buffer of the inverse transform.
    pub divide: u64,
    /// The division, the inverse transform and the coset factors (wall).
    pub inverse: u64,
    /// The read-back of `len` coefficients (wall; the buffers are zeroed in
    /// the background, see `ctx::release_later`).
    pub readback: u64,
    /// GPU time of the coset transforms and placement.
    pub gpu_cosets: u64,
    /// GPU time of the numerator.
    pub gpu_numerator: u64,
    /// GPU time of the division: 0, in `gpu_inverse`.
    pub gpu_divide: u64,
    /// GPU time of the division, the inverse transform and the coset factors.
    pub gpu_inverse: u64,
    /// Command buffers of the numerator.
    pub numerator_runs: u32,
    /// Device bytes in use by this call at its peak (new or from the pool).
    pub peak_bytes: u64,
    /// Proving-key polynomials read from a resident copy.
    pub resident_polys: u32,
    /// Proving-key polynomials uploaded for the call.
    pub uploaded_polys: u32,
    /// Slots of the resident copy filled by this call.
    pub key_copies_made: u32,
}

static LAST_STAGES: Mutex<Option<QuotientStages>> = Mutex::new(None);

/// Test and measurement support (not a stable interface; only with the
/// `test-support` feature: [`test_support::numerator`] returns
/// witness-dependent values).
#[cfg(any(test, feature = "test-support"))]
pub mod test_support {
    use super::*;

    /// The exact [`Need`] total of `job` now (the registry and the
    /// per-request scratch as they are): an upper bound of the device bytes
    /// the hook call would add. `Err`: the decline reason of the checks
    /// before the admission.
    ///
    /// # Safety
    /// As [`crate::quotient`].
    pub unsafe fn need_bytes_job(job: *const c_void) -> Result<usize, &'static str> {
        if job.is_null() || !crate::device() {
            return Err("no-device");
        }
        // SAFETY: the caller's contract.
        let j = unsafe { &*(job as *const QuotientJobAbi) };
        checked(j, None).map_err(|r| r.name())?;
        // SAFETY: as above.
        let tab = unsafe { Tables::new(j) }.map_err(|r| r.name())?;
        // SAFETY: as above.
        let t_inv = unsafe { abi::slice(j.t_inv, j.nt_inv) };
        let s = ctx::lock().map_err(|e| err(e).name())?;
        let mut plan = keys::plan(&tab.keys, 1usize << j.ext_k);
        let need = need_of(j, &tab, t_inv, &plan, &s, true);
        drop(s);
        if let Some(l) = plan.lent.take() {
            l.give_back();
        }
        Ok(need.total())
    }

    /// The stages of the last quotient served.
    pub fn last_stages() -> Option<QuotientStages> {
        *LAST_STAGES.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A device-memory limit in bytes for the quotient's admission (`None`:
    /// the device's free working set).
    pub fn set_memory_limit(limit: Option<u64>) {
        MEMORY_LIMIT.store(limit.unwrap_or(u64::MAX), Relaxed);
    }

    /// Sets the GPU time a numerator command buffer aims at (`None`:
    /// [`TARGET_NANOS`]), so that a test can split a small domain.
    pub fn set_target_nanos(ns: Option<u64>) {
        TARGET.store(ns.unwrap_or(TARGET_NANOS).max(1), Relaxed);
    }

    /// Sets the dispatch threads ([`threads`]); `None` restores the default.
    pub fn set_threads(t: Option<u32>) {
        THREADS.store(
            t.map_or(0, |t| (t / 256 * 256).clamp(256, 1 << 24)),
            Relaxed,
        );
    }

    /// The numerator of a job on the device, all `2^ext_k` values read back
    /// (a test of the kernel against `QuotientJob::numerator_reference`; the
    /// hook never reads it back). `Err`: the decline reason.
    ///
    /// # Safety
    /// As [`crate::quotient`].
    pub unsafe fn numerator(job: *const c_void) -> Result<Vec<Scalar>, &'static str> {
        if job.is_null() || !crate::device() {
            return Err("no-device");
        }
        // SAFETY: the caller's contract.
        let j = unsafe { &*(job as *const QuotientJobAbi) };
        let mut nu = Vec::new();
        // SAFETY: as above.
        unsafe { run(j, &mut [], Some(&mut nu)) }.map(|_| nu).map_err(|r| r.name())
    }
}

// ---------------------------------------------------------------------------
// The entry point

/// The first `out.len()` coefficients of the quotient of `job`. `false`:
/// declined, `out` unchanged.
///
/// # Safety
/// `job` points to a valid `QuotientJobAbi` whose pointers live for the call.
pub(crate) unsafe fn quotient(job: *const c_void, out: &mut [Scalar]) -> bool {
    // SAFETY: the caller's contract.
    let j = unsafe { &*(job as *const QuotientJobAbi) };
    // SAFETY: as above.
    match unsafe { run(j, out, None) } {
        Ok(ns) => {
            RUNS.fetch_add(1, Relaxed);
            account::served(Class::Quotient, ns);
            true
        }
        Err(r) => {
            DECLINES.fetch_add(1, Relaxed);
            account::decline(Class::Quotient, r)
        }
    }
}

/// The coset shift is a primitive cube root of unity `z` (`z != 1`,
/// `z^3 = 1`), the job gives `[z, z^2]` into the coset and `[z^2, z]` out of
/// it (the CUDA crate's `cube_root_shift`).
fn cube_root_shift(zeta: [Scalar; 2], zeta_out: [Scalar; 2]) -> bool {
    let z = zeta[0];
    let z2 = square(&z);
    z != ONE && mul(&z2, &z) == ONE && zeta[1] == z2 && zeta_out == [z2, z]
}

/// The header checks of the CUDA crate, in its order: the ABI version and
/// sizes (`shape`), the coset shift (`shape`), the minimum size
/// (`below-min-k`), then the roots, lengths and counts (`shape`) and the
/// reduction of the job's scalars (`unreduced`).
fn checked(j: &QuotientJobAbi, out_len: Option<usize>) -> Result<(), Reason> {
    if j.abi != abi::QUOTIENT_ABI || j.k < 1 || j.k > j.ext_k || j.ext_k > MAX_LOG_N {
        return Err(Reason::Shape);
    }
    let scalars = [
        j.zeta[0],
        j.zeta[1],
        j.y,
        j.beta,
        j.gamma,
        j.delta_start,
        j.delta,
        j.divisor,
        j.zeta_out[0],
        j.zeta_out[1],
    ];
    // the shift check multiplies: only on reduced values
    if !field::all_reduced(&scalars[..2]) || !field::all_reduced(&scalars[8..]) {
        return Err(Reason::Shape);
    }
    if !cube_root_shift(j.zeta, j.zeta_out) {
        return Err(Reason::Shape);
    }
    if j.ext_k < crate::thresholds().quotient {
        return Err(Reason::BelowMinK);
    }
    let en = 1usize << j.ext_k;
    let ok = Some(j.extended_omega) == ntt::std_root(j.ext_k, Dir::Forward)
        && Some(j.omega_inv) == ntt::std_root(j.ext_k, Dir::Inverse)
        && out_len.is_none_or(|l| l == j.len)
        && j.len <= en
        && j.nt_inv.is_power_of_two()
        && j.nt_inv <= en
        && j.gates.noutputs == 1
        && j.perm_nsets <= j.perm_ncolumns.max(1)
        && (j.perm_nsets == 0) == (j.perm_ncolumns == 0)
        && j.perm_chunk_len as usize * j.perm_nsets >= j.perm_ncolumns
        && j.nslots <= abi::INDEX_MASK;
    if !ok {
        return Err(Reason::Shape);
    }
    if !field::all_reduced(&scalars) {
        return Err(Reason::Unreduced);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Host copies

/// Copies `src` into the start of `dst` on several threads; `false` when an
/// element is not reduced (the copy may then be partial).
fn copy_checked(dst: &mut [Scalar], src: &[Scalar]) -> bool {
    const PART: usize = 1 << 16;
    let dst = &mut dst[..src.len()];
    let one = |d: &mut [Scalar], s: &[Scalar]| {
        d.copy_from_slice(s);
        s.iter().all(field::is_reduced)
    };
    if src.len() <= 2 * PART {
        return one(dst, src);
    }
    let threads = std::thread::available_parallelism().map_or(4, |t| t.get()).min(8);
    let per = src.len().div_ceil(threads).max(PART);
    std::thread::scope(|s| {
        let hs: Vec<_> = dst
            .chunks_mut(per)
            .zip(src.chunks(per))
            .map(|(d, c)| s.spawn(move || one(d, c)))
            .collect();
        hs.into_iter().all(|h| h.join().unwrap_or(false))
    })
}

// ---------------------------------------------------------------------------
// The device run

/// One dispatch of the numerator (`quotient::Params` of
/// `shaders/include/quotient.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Params {
    ext_k: u32,
    log_scale: u32,
    mask: u32,
    row0: u32,
    row_end: u32,
    threads: u32,
    gates: u32,
    perm_nsets: u32,
    perm_ncolumns: u32,
    perm_chunk_len: u32,
    perm_last_rotation: i32,
    perm: u32,
    nlookups: u32,
    lookups: u32,
    ntrash: u32,
    trash: u32,
    l0: u32,
    l_last: u32,
    l_active: u32,
    pad: [u32; 5],
    y: Scalar,
    beta: Scalar,
    gamma: Scalar,
    delta_start: Scalar,
    delta: Scalar,
    omega: Scalar,
    omega_step: Scalar,
}
// SAFETY: `repr(C)`, 24 four-byte fields then seven `[u64; 4]` (offset 96, a
// multiple of 8): no padding, every bit pattern valid.
unsafe impl Pod for Params {}
const _: () = assert!(size_of::<Params>() == 96 + 7 * 32);

/// Bytes the device allows this call to allocate: the free working set (the
/// idle buffers of the pool included, see [`ctx::mem_info`]) or the test
/// limit.
fn memory_limit() -> u64 {
    let free = ctx::mem_info().map_or(0, |(free, _)| free as u64);
    free.min(MEMORY_LIMIT.load(Relaxed))
}

fn err(e: Error) -> Reason {
    Reason::of(&e)
}

/// The [`Need`] of a planned call: the proof's cosets, the uploads of the
/// plan, the resident copy when the call makes it, the second upload when a
/// polynomial placed in a slot is checked (it may be stale), the transform
/// scratch unless the request has one long enough, the job's tables.
fn need_of(
    j: &QuotientJobAbi,
    tab: &Tables<'_>,
    t_inv: &[Scalar],
    plan: &keys::Plan,
    s: &ctx::Session,
    inverse: bool,
) -> Need {
    let ebytes = SCALAR_BYTES << j.ext_k;
    let ntemp = plan.places.iter().filter(|p| **p == Place::Temp).count();
    let lent_new = plan.lent.as_ref().map_or(0, |l| if l.buf.is_none() { l.bytes() } else { 0 });
    let resident_checked = plan.resident_checked();
    let tmp_need = ntt::scratch_bytes(j.ext_k);
    let tmp_have = s.get_scratch("quotient-tmp").map_or(0, |b| b.len_bytes());
    Need {
        ebytes,
        coeff_bytes: SCALAR_BYTES << j.k,
        nproof: tab.proof.len(),
        ntemp,
        lent_new,
        restale: if resident_checked > 0 {
            ntemp + resident_checked
        } else {
            0
        },
        slots_bytes: (j.nslots as usize).max(1) * stride(j.ext_k) * SCALAR_BYTES,
        // a shorter one is replaced by a new one
        tmp_new: if tmp_need > 0 && tmp_have < tmp_need {
            tmp_need
        } else {
            0
        },
        words_bytes: tab.words.len() * 4,
        uniforms_bytes: tab.uniforms.len() * SCALAR_BYTES,
        t_inv_bytes: t_inv.len() * SCALAR_BYTES,
        inverse,
    }
}

/// The bank and index of each proving-key polynomial, and the number of
/// uploaded ones.
fn banks(places: &[Place]) -> (Vec<(u32, usize)>, usize) {
    let mut ti = 0;
    let bank_of = places
        .iter()
        .map(|p| match p {
            Place::Resident(slot) => (tables::BANK_KEY_RES, *slot),
            Place::Temp => {
                ti += 1;
                (tables::BANK_KEY_TMP, ti - 1)
            }
        })
        .collect();
    (bank_of, ti)
}

/// The quotient of `j` into `out`, or with `numerator` the whole numerator
/// read back into it (and `out` untouched). Returns the GPU time.
///
/// # Safety
/// The job's contract.
unsafe fn run(
    j: &QuotientJobAbi,
    out: &mut [Scalar],
    mut numerator: Option<&mut Vec<Scalar>>,
) -> Result<u64, Reason> {
    let t_start = Instant::now();
    let mut st = QuotientStages::default();
    checked(j, numerator.is_none().then_some(out.len()))?;
    // SAFETY: the job's contract.
    let mut tab = unsafe { Tables::new(j)? };
    // SAFETY: the job's contract.
    let t_inv = unsafe { abi::slice(j.t_inv, j.nt_inv) };
    if !field::all_reduced(t_inv) {
        return Err(Reason::Unreduced);
    }
    // the scalars passed by value to the transform helpers, whose shader
    // arithmetic needs reduced inputs: `checked` has checked them already
    // (the shift and its inverse, `zeta_out`, the divisor, the roots by
    // equality with the standard roots); checked again here, next to their
    // use, so that a change to `checked` cannot let one through
    let by_value = [
        j.zeta[0],
        j.zeta[1],
        j.zeta_out[0],
        j.zeta_out[1],
        j.divisor,
        j.extended_omega,
        j.omega_inv,
    ];
    if !field::all_reduced(&by_value) {
        return Err(Reason::Unreduced);
    }
    let (k, ext_k) = (j.k, j.ext_k);
    let (n, en) = (1usize << k, 1usize << ext_k);
    let ebytes = SCALAR_BYTES << ext_k;
    let nproof = tab.proof.len();
    let nslots = j.nslots as usize;
    let threads = stride(ext_k);

    let mut s = ctx::lock().map_err(err)?;
    let c: &'static Ctx = s.ctx();
    let info = crate::present().ok_or(Reason::NoDevice)?;

    // the proving-key polynomials: resident or uploaded, placed before their
    // digests are checked (they are checked while the device works on the
    // proof's polynomials, and settled before the numerator reads a slot)
    let mut plan = keys::plan(&tab.keys, en);
    let ntemp = plan.places.iter().filter(|p| **p == Place::Temp).count();

    // memory: every buffer this call can allocate (the same sum as
    // `need_bytes`)
    let need = need_of(j, &tab, t_inv, &plan, &s, numerator.is_none());
    let tmp_need = ntt::scratch_bytes(ext_k);
    let fits = need.total() as u64 <= memory_limit()
        && need.biggest() as u64 <= info.max_buffer_length;
    if !fits {
        if let Some(l) = plan.lent.take() {
            l.give_back();
        }
        return Err(Reason::Memory);
    }
    let bytes0 = ctx::in_use_bytes();
    st.prepare = t_start.elapsed().as_nanos() as u64;

    let r = (|| -> Result<u64, Reason> {
        let mut gpu = 0u64;
        // transform scratch (per request: it holds witness values)
        if tmp_need > 0 {
            s.scratch("quotient-tmp", tmp_need).map_err(err)?;
        }
        let proof_buf = c.alloc_pooled(nproof.max(1) * ebytes).map_err(err)?;
        let mut src_a = c.alloc_pooled(n * SCALAR_BYTES).map_err(err)?;
        let mut src_b = c.alloc_pooled(n * SCALAR_BYTES).map_err(err)?;
        let mut key_tmp = c.alloc_pooled(ntemp * ebytes).map_err(err)?;
        if let Some(l) = plan.lent.as_mut()
            && l.buf.is_none()
        {
            l.buf = Some(c.alloc_pooled(l.npolys * ebytes).map_err(err)?);
        }
        let tmp_ref: Option<&Buffer> = if tmp_need > 0 {
            s.get_scratch("quotient-tmp")
        } else {
            None
        };
        // without transform scratch (one-pass sizes) the scratch argument is
        // not read
        let dummy = c.alloc(0).map_err(err)?;
        let tmp = tmp_ref.unwrap_or(&dummy);

        // keys on CPU threads (digests, uploads, slot refills), cosets on the
        // device; the copy of the first proof polynomial comes first, alone
        // on the CPU, as the device waits for it
        let t_cos = Instant::now();
        if let Some(first) = tab.proof.first()
            && !copy_checked(src_a.as_mut_slice::<Scalar>(), first)
        {
            return Err(Reason::Unreduced);
        }
        let keys_list = &tab.keys;
        let places_ref = &plan.places;
        let check_ref = &plan.check;
        // the polynomials whose slot this call fills: their digests are
        // computed over the bytes copied into the slot, so that a slot is
        // only ever marked present (at `keys::settle`) with verified content
        let fills: Vec<bool> = (0..tab.keys.len()).map(|i| plan.fills(i)).collect();
        let fills_ref = &fills;
        let lent_ref = &mut plan.lent;
        let key_tmp_ref = &mut key_tmp;
        let proof_list = &tab.proof;
        let zeta = j.zeta;
        // (digests in the order of `check`, copies good, slots filled, wall)
        type KeyWork = (Vec<[u8; 32]>, bool, Vec<usize>, u64);
        let (key_work, cos): (KeyWork, Result<u64, Reason>) = std::thread::scope(|sc| {
            let kh = sc.spawn(move || -> KeyWork {
                let t = Instant::now();
                // the registered polynomials read from host memory
                let host_hash: Vec<&[Scalar]> = check_ref
                    .iter()
                    .filter(|&&i| !fills_ref[i])
                    .map(|&i| keys_list[i])
                    .collect();
                let (host_digests, (ok, filled)) = std::thread::scope(|sc2| {
                    let dh = sc2.spawn(|| keys::digest_many(&host_hash));
                    let tmp_slice = key_tmp_ref.as_mut_slice::<Scalar>();
                    let mut ti = 0;
                    let mut ok = true;
                    let mut filled = Vec::new();
                    for (i, (p, pl)) in keys_list.iter().zip(places_ref).enumerate() {
                        match pl {
                            Place::Temp => {
                                ok &= copy_checked(&mut tmp_slice[ti * en..(ti + 1) * en], p);
                                ti += 1;
                            }
                            Place::Resident(slot) => {
                                if fills_ref[i] {
                                    let l = lent_ref.as_mut().expect("a lent copy");
                                    let b =
                                        l.buf.as_mut().expect("allocated").as_mut_slice::<Scalar>();
                                    ok &= copy_checked(&mut b[slot * en..(slot + 1) * en], p);
                                    filled.push(*slot);
                                }
                            }
                        }
                        if !ok {
                            break;
                        }
                    }
                    (dh.join().unwrap_or_default(), (ok, filled))
                });
                if !ok {
                    return (Vec::new(), false, filled, t.elapsed().as_nanos() as u64);
                }
                // the bytes now in the filled slots
                let slot_digests = match lent_ref.as_ref().and_then(|l| l.buf.as_ref()) {
                    Some(b) => {
                        let b = b.as_slice::<Scalar>();
                        let in_slots: Vec<&[Scalar]> =
                            filled.iter().map(|s| &b[s * en..(s + 1) * en]).collect();
                        keys::digest_many(&in_slots)
                    }
                    None => Vec::new(),
                };
                // in the order of `check` (both lists follow it)
                let (mut h, mut f) = (host_digests.into_iter(), slot_digests.into_iter());
                let digests: Option<Vec<[u8; 32]>> = check_ref
                    .iter()
                    .map(|&i| if fills_ref[i] { f.next() } else { h.next() })
                    .collect();
                (digests.unwrap_or_default(), ok, filled, t.elapsed().as_nanos() as u64)
            });
            let cos = (|| -> Result<u64, Reason> {
                let mut ns = 0;
                for i in 0..proof_list.len() {
                    let (cur, next) = if i % 2 == 0 {
                        (&src_a, &mut src_b)
                    } else {
                        (&src_b, &mut src_a)
                    };
                    // the copy of the next polynomial runs on the CPU while
                    // the device transforms this one straight into its place
                    let following = proof_list.get(i + 1).copied();
                    let (r, copied) = std::thread::scope(|sc2| {
                        let h = following.map(|f| {
                            sc2.spawn(move || copy_checked(next.as_mut_slice::<Scalar>(), f))
                        });
                        let r = ntt::coeff_to_extended_dev_at(
                            c,
                            cur,
                            &proof_buf,
                            i * ebytes,
                            tmp,
                            k,
                            ext_k,
                            &zeta,
                        );
                        (r, h.is_none_or(|h| h.join().unwrap_or(false)))
                    });
                    ns += r.map_err(err)?;
                    if !copied {
                        return Err(Reason::Unreduced);
                    }
                }
                Ok(ns)
            })();
            let kw = kh.join().unwrap_or_default();
            (kw, cos)
        });
        let (digests, keys_ok, filled, keys_ns) = key_work;
        st.keys = keys_ns;
        gpu += cos?;
        st.gpu_cosets = gpu;
        if !keys_ok {
            return Err(Reason::Unreduced);
        }
        if digests.len() != plan.check.len() {
            return Err(Reason::DeviceError);
        }
        st.peak_bytes = ctx::in_use_bytes().saturating_sub(bytes0);
        ctx::release_later(vec![src_a, src_b]);
        st.cosets = t_cos.elapsed().as_nanos() as u64;

        // the digests settle the placement: a stale polynomial is uploaded
        let t_num = Instant::now();
        if keys::settle(&mut plan, &tab.keys, &digests) {
            // the uploads again, now with the stale ones (the buffers of the
            // call are allocated already: only this one is new)
            let (_, nt) = banks(&plan.places);
            let more = (nt * ebytes) as u64;
            if more > memory_limit() || more > info.max_buffer_length {
                return Err(Reason::Memory);
            }
            let mut kt = c.alloc_pooled(nt * ebytes).map_err(err)?;
            let ks = kt.as_mut_slice::<Scalar>();
            let mut ti = 0;
            for (p, pl) in tab.keys.iter().zip(&plan.places) {
                if *pl == Place::Temp {
                    if !copy_checked(&mut ks[ti * en..(ti + 1) * en], p) {
                        return Err(Reason::Unreduced);
                    }
                    ti += 1;
                }
            }
            key_tmp = kt;
        }
        let (bank_of, ntemp) = banks(&plan.places);
        let lagrange = tab.place_keys(&bank_of);
        st.uploaded_polys = ntemp as u32;
        st.resident_polys = (plan.places.len() - ntemp) as u32;
        st.key_copies_made = filled
            .iter()
            .filter(|s| plan.lent.as_ref().is_some_and(|l| l.present[**s]))
            .count() as u32;

        // the numerator, into `nu`
        let nu = c.alloc_pooled(ebytes).map_err(err)?;
        let scratch = c.alloc_pooled(nslots.max(1) * threads * SCALAR_BYTES).map_err(err)?;
        let words = c.upload(&tab.words).map_err(err)?;
        let unis = if tab.uniforms.is_empty() {
            c.alloc(0)
        } else {
            c.upload(&tab.uniforms)
        }
        .map_err(err)?;
        let key_res: &Buffer = plan.lent.as_ref().and_then(|l| l.buf.as_ref()).unwrap_or(&dummy);
        let key_tmp_b: &Buffer = if ntemp > 0 { &key_tmp } else { &dummy };
        let omega = j.extended_omega;
        let mut p = Params {
            ext_k,
            log_scale: ext_k - k,
            mask: (en - 1) as u32,
            threads: threads as u32,
            gates: tab.gates,
            perm_nsets: j.perm_nsets as u32,
            perm_ncolumns: j.perm_ncolumns as u32,
            perm_chunk_len: j.perm_chunk_len,
            perm_last_rotation: j.perm_last_rotation,
            perm: tab.perm,
            nlookups: j.nlookups as u32,
            lookups: tab.lookups,
            ntrash: j.ntrash as u32,
            trash: tab.trash,
            l0: lagrange[0],
            l_last: lagrange[1],
            l_active: lagrange[2],
            y: j.y,
            beta: j.beta,
            gamma: j.gamma,
            delta_start: j.delta_start,
            delta: j.delta,
            omega,
            omega_step: pow(&omega, threads as u64),
            ..Default::default()
        };
        // rows per command buffer: a first estimate at about 0.5 ns per
        // field multiplication of a row, then from the measured time per row
        let cost = tab.cost_per_row.max(1) + 8;
        let target = TARGET.load(Relaxed);
        let mut rows = ((target / 2) / cost).max(threads as u64) as usize;
        let mut row0 = 0usize;
        let mut ns_num = 0u64;
        while row0 < en {
            let rows_now = rows.min(en - row0).div_ceil(threads) * threads;
            let end = (row0 + rows_now).min(en);
            p.row0 = row0 as u32;
            p.row_end = end as u32;
            let disp = threads.min(end - row0);
            let pp = p;
            let ns = c
                .run(|cmd| {
                    cmd.dispatch(
                        "quotient_numerator",
                        disp,
                        &[
                            Arg::Buffer(&proof_buf),
                            Arg::Buffer(key_res),
                            Arg::Buffer(key_tmp_b),
                            Arg::Buffer(&words),
                            Arg::Buffer(&unis),
                            Arg::Buffer(&nu),
                            Arg::Buffer(&scratch),
                            Arg::value(&pp),
                        ],
                    )
                })
                .map_err(err)?
                .1;
            ns_num += ns;
            st.numerator_runs += 1;
            let per_row = (ns as f64 / (end - row0) as f64).max(0.01);
            let want = (target as f64 / per_row) as usize;
            rows = want.clamp(threads, rows.saturating_mul(8).max(threads));
            row0 = end;
        }
        gpu += ns_num;
        st.gpu_numerator = ns_num;
        st.peak_bytes = st.peak_bytes.max(ctx::in_use_bytes().saturating_sub(bytes0));
        if let Some(l) = plan.lent.take() {
            l.give_back();
        }
        // the buffers of the cosets are zeroed (on the CPU, in the
        // background) while the device divides and transforms; the zeroing
        // ends before the request ends (`end_request` waits for it)
        ctx::release_later(vec![scratch, proof_buf, key_tmp, words, unis]);
        st.numerator = t_num.elapsed().as_nanos() as u64;

        if let Some(v) = numerator.as_mut() {
            v.clear();
            v.extend_from_slice(&nu.as_slice::<Scalar>()[..en]);
            ctx::release_later(vec![nu]);
            return Ok(gpu);
        }

        // h = nu / t on the coset, then extended_to_coeff, in the command
        // buffers of the inverse transform
        let t_inv_ntt = Instant::now();
        let tinv = c.upload(t_inv).map_err(err)?;
        let log_tn = t_inv.len().ilog2();
        let tmp2 = tmp_ref.unwrap_or(&dummy);
        let ns = ntt::divide_extended_to_coeff_dev(
            c,
            &nu,
            tmp2,
            ext_k,
            &j.divisor,
            &j.zeta_out,
            &tinv,
            log_tn,
        )
        .map_err(err)?;
        gpu += ns;
        st.gpu_inverse = ns;
        st.inverse = t_inv_ntt.elapsed().as_nanos() as u64;

        // the first `len` coefficients. Everything that can fail or panic is
        // done before the first byte of `out` is written (a panic becomes a
        // decline, which must leave `out` unchanged): `tinv` is released, the
        // source is sliced (`checked` has held `out.len()` to `j.len <= en`)
        // and the resident copy was given back after the numerator. What
        // follows the copy cannot fail: `ctx::par_copy` and
        // `ctx::release_later` never panic, and the rest of `run` and of
        // `quotient` stores counters and takes poison-tolerant locks.
        let t_rb = Instant::now();
        drop(tinv);
        debug_assert!(plan.lent.is_none());
        let src = &nu.as_slice::<Scalar>()[..out.len()];
        ctx::par_copy(out, src);
        ctx::release_later(vec![nu]);
        st.readback = t_rb.elapsed().as_nanos() as u64;
        Ok(gpu)
    })();
    if let Some(l) = plan.lent.take() {
        l.give_back();
    }
    drop(s);
    if r.is_ok() && numerator.is_none() {
        *LAST_STAGES.lock().unwrap_or_else(|e| e.into_inner()) = Some(st);
    }
    r
}
