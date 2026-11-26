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

//! The quotient on a GPU backend and the resident proving-key copies; the
//! counterpart of the CUDA crate's `gpu/src/quotient.rs`.
//!
//! [`quotient`] takes the job of the `midnight_proofs::gpu` quotient hook
//! ([`abi`]), moves the proof's polynomials to the extended coset, runs the
//! numerator kernel over the job's programs, divides by the vanishing
//! polynomial, transforms back and writes the first `len` coefficients.
//!
//! Trust boundary: the caller guarantees that the job's pointers are valid
//! for reads of their counts and not written during the call. Everything else
//! (version, sizes, roots, coset shift, table ranges, reduction of every
//! scalar and polynomial, memory need) is checked here before any polynomial
//! is read.
//!
//! Departures from the CUDA backend:
//! - The CUDA tables of device pointers become flat buffers with index
//!   tables ([`tables`]): a kernel takes seven buffers (a Metal kernel takes
//!   at most 31).
//! - The numerator runs over row ranges of about
//!   [`crate::Tuning::quotient_target_nanos`] each, so no command buffer runs
//!   long.
//! - Proving-key polynomials are also checked reduced.
//! - Outside a request every serve checks the digests of the registered
//!   polynomials it reads; the CUDA crate checks once per request.

pub mod abi;
pub mod digest;
mod keys;
pub mod tables;

use std::{
    ffi::c_void,
    sync::{
        Mutex, MutexGuard, OnceLock,
        atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed},
    },
};

use blst::{blst_fr, blst_fr_add, blst_fr_cneg, blst_fr_mul, blst_fr_sqr, blst_fr_sub};
use rayon::prelude::*;
pub(crate) use keys::KeyState;
pub use keys::{
    KeyHandle, begin_request, end_request, evict_key_copies, keep_resident, key_stale,
    keys_registered, register_key, set_keep_resident,
};

use self::{
    abi::QuotientJobAbi,
    keys::{Lent, Place, Plan},
    tables::Tables,
};
use crate::{
    Backend, MAX_LOG_N, Scalar,
    account::{self, Class, Reason},
    ctx::{self, Arg, Buffer, Ctx, Error, Pod, Session},
    field,
    ntt::{self, Dir},
};

const SCALAR_BYTES: usize = 32;

/// The Montgomery form of one (`2^256 mod r`).
const ONE: Scalar = [
    0x0000_0001_ffff_fffe,
    0x5884_b7fa_0003_4802,
    0x998c_4fef_ecbc_4ff5,
    0x1824_b159_acc5_056f,
];

#[inline(always)]
fn fr_add(a: &Scalar, b: &Scalar) -> Scalar {
    let mut r = blst_fr::default();
    // SAFETY: valid pointers to blst_fr values.
    unsafe { blst_fr_add(&mut r, &blst_fr { l: *a }, &blst_fr { l: *b }) };
    r.l
}

#[inline(always)]
fn fr_sub(a: &Scalar, b: &Scalar) -> Scalar {
    let mut r = blst_fr::default();
    // SAFETY: valid pointers to blst_fr values.
    unsafe { blst_fr_sub(&mut r, &blst_fr { l: *a }, &blst_fr { l: *b }) };
    r.l
}

#[inline(always)]
fn fr_neg(a: &Scalar) -> Scalar {
    let mut r = blst_fr::default();
    // SAFETY: valid pointers to blst_fr values.
    unsafe { blst_fr_cneg(&mut r, &blst_fr { l: *a }, true) };
    r.l
}

#[inline(always)]
fn mul(a: &Scalar, b: &Scalar) -> Scalar {
    let mut r = blst_fr::default();
    // SAFETY: valid pointers to blst_fr values.
    unsafe { blst_fr_mul(&mut r, &blst_fr { l: *a }, &blst_fr { l: *b }) };
    r.l
}

#[inline(always)]
fn square(a: &Scalar) -> Scalar {
    let mut r = blst_fr::default();
    // SAFETY: valid pointers to blst_fr values.
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

/// The quotient state of one backend: counters and settings.
pub(crate) struct QuotientState {
    runs: AtomicU64,
    declines: AtomicU64,
    /// [`threads`] once read (0: not yet).
    threads: AtomicU32,
    /// The admission's memory limit in bytes besides the device's (tests).
    memory_limit: AtomicU64,
    /// The GPU time a numerator command buffer aims at (tests lower it; 0:
    /// the backend's tuning).
    target: AtomicU64,
    /// Reusable host staging buffer for CPU-side `coeff_to_extended` cosets.
    /// Invariant: always all zeros whenever unlocked (wiped on every exit).
    cpu_coset_buf: Mutex<Vec<Scalar>>,
    /// The placement counts of the last quotient served.
    #[cfg(any(test, feature = "test-support"))]
    last_stages: std::sync::Mutex<Option<QuotientStages>>,
}

impl QuotientState {
    /// Every counter zero, every setting at its default.
    pub(crate) const fn new() -> Self {
        QuotientState {
            runs: AtomicU64::new(0),
            declines: AtomicU64::new(0),
            threads: AtomicU32::new(0),
            memory_limit: AtomicU64::new(u64::MAX),
            target: AtomicU64::new(0),
            cpu_coset_buf: Mutex::new(Vec::new()),
            #[cfg(any(test, feature = "test-support"))]
            last_stages: std::sync::Mutex::new(None),
        }
    }
}

fn state<B: Backend>() -> &'static QuotientState {
    &B::host().quotient
}

/// The GPU time one command buffer of the numerator aims at.
fn target<B: Backend>() -> u64 {
    match state::<B>().target.load(Relaxed) {
        0 => B::tuning().quotient_target_nanos.max(1),
        t => t,
    }
}

/// (quotients computed on the device, quotient jobs declined) since process
/// start.
pub fn quotient_counts<B: Backend>() -> (u64, u64) {
    let q = state::<B>();
    (q.runs.load(Relaxed), q.declines.load(Relaxed))
}

/// A thread count rounded down to a multiple of 256 in `[256, 2^24]`.
fn normalize_threads(t: u64) -> u32 {
    (t / 256 * 256).clamp(256, 1 << 24) as u32
}

/// The threads of one numerator dispatch, which is also the scratch slot
/// stride: `{ENV_PREFIX}_QUOTIENT_THREADS`, else
/// [`crate::Tuning::quotient_threads`]; a multiple of 256 in `[256, 2^24]`.
pub fn threads<B: Backend>() -> u32 {
    let q = state::<B>();
    let t = q.threads.load(Relaxed);
    if t != 0 {
        return t;
    }
    let t = std::env::var(format!("{}_QUOTIENT_THREADS", B::ENV_PREFIX))
        .ok()
        .and_then(|e| e.trim().parse::<u64>().ok())
        .map_or(B::tuning().quotient_threads, normalize_threads);
    let _ = q.threads.compare_exchange(0, t, Relaxed, Relaxed);
    q.threads.load(Relaxed)
}

/// The dispatch threads (the scratch stride) for an extended domain of
/// `2^ext_k` rows.
fn stride<B: Backend>(ext_k: u32) -> usize {
    (threads::<B>() as usize).min(1usize << ext_k)
}

// ---------------------------------------------------------------------------
// Memory need

/// Every device allocation one quotient call can make, in bytes. [`run`]
/// allocates exactly these, each at most once, so the sum bounds the bytes the
/// call adds. [`run`]'s admission and [`need_bytes`] both sum it, so the two
/// cannot drift.
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
    /// The second upload after a stale key at [`keys::settle`] (0: no slot is
    /// checked, so none can move).
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
            if self.inverse {
                ntt::SCALE_TABLE_BYTES
            } else {
                0
            },
        ]
    }

    /// The sum of the call's allocations.
    fn total(&self) -> usize {
        self.buffers().iter().sum()
    }

    /// The largest single buffer (against the device's largest buffer).
    fn biggest(&self) -> usize {
        self.buffers().into_iter().max().unwrap_or(0)
    }

    /// Whether the call fits the free device memory and the largest buffer.
    fn fits<B: Backend>(&self, c: &Ctx<B>) -> bool {
        self.total() as u64 <= memory_limit::<B>() && self.biggest() as u64 <= c.max_buffer_len()
    }
}

/// An upper bound of the device bytes one quotient job allocates: the `Need`
/// of `run` at its worst over what the shape does not fix. It assumes the
/// resident copy holds only the job's key polynomials and ignores the few KiB
/// of program words and uniforms. The arguments are the fields of
/// `midnight_proofs::gpu::QuotientShape`.
pub fn need_bytes<B: Backend>(
    ext_k: u32,
    proof_polys: usize,
    key_polys: usize,
    nslots: u32,
) -> usize {
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
                slots_bytes: (nslots as usize).max(1) * stride::<B>(ext_k) * SCALAR_BYTES,
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

/// The [`Need`] of a planned call.
fn need_of<B: Backend>(
    j: &QuotientJobAbi,
    tab: &Tables<'_>,
    t_inv: &[Scalar],
    plan: &Plan<B>,
    s: &Session<B>,
    inverse: bool,
) -> Need {
    let ntemp = plan.places.iter().filter(|p| **p == Place::Temp).count();
    let lent_new = plan.lent.as_ref().map_or(0, |l| if l.buf.is_none() { l.bytes() } else { 0 });
    let resident_checked = plan.resident_checked();
    let tmp_need = ntt::scratch_bytes(j.ext_k);
    let tmp_have = s.get_scratch("quotient-tmp").map_or(0, |b| b.len_bytes());
    Need {
        ebytes: SCALAR_BYTES << j.ext_k,
        coeff_bytes: SCALAR_BYTES << j.k,
        nproof: tab.proof.len(),
        ntemp,
        lent_new,
        restale: if resident_checked > 0 {
            ntemp + resident_checked
        } else {
            0
        },
        slots_bytes: (j.nslots as usize).max(1) * stride::<B>(j.ext_k) * SCALAR_BYTES,
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

/// Bytes this call may allocate: the free working set (idle pool buffers
/// included) or the test limit, whichever is smaller.
fn memory_limit<B: Backend>() -> u64 {
    let free = ctx::mem_info::<B>().map_or(0, |(free, _)| free as u64);
    free.min(state::<B>().memory_limit.load(Relaxed))
}

// ---------------------------------------------------------------------------
// Test support

/// The placement counts of the last quotient served.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QuotientStages {
    /// Command buffers of the numerator.
    pub numerator_runs: u32,
    /// Proving-key polynomials read from a resident copy.
    pub resident_polys: u32,
    /// Proving-key polynomials uploaded for the call.
    pub uploaded_polys: u32,
    /// Slots of the resident copy filled by this call.
    pub key_copies_made: u32,
}

/// Test entry points (not a stable interface; [`test_support::numerator`]
/// returns witness-dependent values).
#[cfg(any(test, feature = "test-support"))]
pub mod test_support {
    use super::*;

    /// The exact [`Need`] total of `job` now. `Err`: the decline reason of the
    /// checks before admission.
    ///
    /// # Safety
    /// As [`crate::quotient`].
    pub unsafe fn need_bytes_job<B: Backend>(job: *const c_void) -> Result<usize, &'static str> {
        if job.is_null() || !crate::device::<B>() {
            return Err("no-device");
        }
        // SAFETY: the caller's contract.
        let j = unsafe { &*(job as *const QuotientJobAbi) };
        checked::<B>(j, None).map_err(|r| r.name())?;
        // SAFETY: the caller's contract; `checked` has passed.
        let tab = unsafe { Tables::new(j) }.map_err(|r| r.name())?;
        // SAFETY: the caller's contract.
        let t_inv = unsafe { abi::slice(j.t_inv, j.nt_inv) };
        let s = ctx::lock::<B>().map_err(|e| err(e).name())?;
        let mut plan = keys::plan::<B>(&tab.keys, 1usize << j.ext_k);
        let need = need_of(j, &tab, t_inv, &plan, &s, true);
        drop(s);
        if let Some(l) = plan.lent.take() {
            l.give_back();
        }
        Ok(need.total())
    }

    /// The placement counts of the last quotient served.
    pub fn last_stages<B: Backend>() -> Option<QuotientStages> {
        *state::<B>().last_stages.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A device-memory limit for the admission (`None`: the free working
    /// set).
    pub fn set_memory_limit<B: Backend>(limit: Option<u64>) {
        state::<B>().memory_limit.store(limit.unwrap_or(u64::MAX), Relaxed);
    }

    /// Sets the numerator's target GPU time per command buffer (`None`:
    /// the backend's tuning).
    pub fn set_target_nanos<B: Backend>(ns: Option<u64>) {
        state::<B>().target.store(ns.map_or(0, |n| n.max(1)), Relaxed);
    }

    /// Sets the dispatch threads ([`threads`]); `None` restores the default.
    pub fn set_threads<B: Backend>(t: Option<u32>) {
        state::<B>()
            .threads
            .store(t.map_or(0, |t| normalize_threads(t.into())), Relaxed);
    }

    /// The numerator of a job, all `2^ext_k` values read back, for testing the
    /// kernel. `Err`: the decline reason.
    ///
    /// # Safety
    /// As [`crate::quotient`].
    pub unsafe fn numerator<B: Backend>(job: *const c_void) -> Result<Vec<Scalar>, &'static str> {
        if job.is_null() || !crate::device::<B>() {
            return Err("no-device");
        }
        // SAFETY: the caller's contract.
        let j = unsafe { &*(job as *const QuotientJobAbi) };
        let mut nu = Vec::new();
        // SAFETY: the caller's contract.
        unsafe { run::<B>(j, &mut [], Some(&mut nu)) }.map(|_| nu).map_err(|r| r.name())
    }
}

// ---------------------------------------------------------------------------
// The entry point

/// The first `out.len()` coefficients of the quotient of `job`. `false`:
/// declined, `out` unchanged.
///
/// # Safety
/// `job` is not null and points to a valid `QuotientJobAbi` whose pointers
/// are valid for reads of their counts and not written during the call.
pub unsafe fn quotient<B: Backend>(job: *const c_void, out: &mut [Scalar]) -> bool {
    // SAFETY: the caller's contract.
    let j = unsafe { &*(job as *const QuotientJobAbi) };
    // SAFETY: the caller's contract.
    match unsafe { run::<B>(j, out, None) } {
        Ok((ns, _stages)) => {
            let q = state::<B>();
            #[cfg(any(test, feature = "test-support"))]
            {
                *q.last_stages.lock().unwrap_or_else(|e| e.into_inner()) = Some(_stages);
            }
            q.runs.fetch_add(1, Relaxed);
            account::served::<B>(Class::Quotient, ns);
            true
        }
        Err(r) => {
            state::<B>().declines.fetch_add(1, Relaxed);
            account::decline::<B>(Class::Quotient, r)
        }
    }
}

/// The coset shift is a primitive cube root of unity `z`; the job gives
/// `[z, z^2]` into the coset and `[z^2, z]` out of it.
fn cube_root_shift(zeta: [Scalar; 2], zeta_out: [Scalar; 2]) -> bool {
    let z = zeta[0];
    let z2 = square(&z);
    z != ONE && mul(&z2, &z) == ONE && zeta[1] == z2 && zeta_out == [z2, z]
}

/// The header checks, in the CUDA crate's order and with its reasons.
/// `out_len` must be the job's `len` (`None`: the numerator test path).
fn checked<B: Backend>(j: &QuotientJobAbi, out_len: Option<usize>) -> Result<(), Reason> {
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
    if j.ext_k < crate::thresholds::<B>().quotient {
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

/// Checks that `t_inv` and the by-value transform scalars are reduced. Most
/// were checked by [`checked`]; checking again next to their use keeps a
/// change there from letting an unreduced value reach the device.
fn check_transform_inputs(j: &QuotientJobAbi, t_inv: &[Scalar]) -> Result<(), Reason> {
    if !field::all_reduced(t_inv) {
        return Err(Reason::Unreduced);
    }
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
    Ok(())
}

fn err(e: Error) -> Reason {
    Reason::of(&e)
}

// ---------------------------------------------------------------------------
// Host copies

/// Copies `src` into the start of `dst` on several threads, checking every
/// element reduced; `false` when one is not (the copy may be partial).
fn copy_checked(dst: &mut [Scalar], src: &[Scalar]) -> bool {
    const PART: usize = 1 << 14;
    let dst = &mut dst[..src.len()];
    let one = |d: &mut [Scalar], s: &[Scalar]| {
        d.copy_from_slice(s);
        s.iter().all(field::is_reduced)
    };
    if src.len() <= 2 * PART {
        return one(dst, src);
    }
    let base = dst.as_mut_ptr() as usize;
    crate::msm::par::par_parts(
        src.len(),
        crate::msm::par::host_threads().min(8),
        PART,
        |lo, hi| {
            let s = &src[lo..hi];
            // SAFETY: `dst` has length `src.len()` and `[lo, hi)` partitions
            // are disjoint and in-bounds.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    s.as_ptr(),
                    (base as *mut Scalar).add(lo),
                    hi - lo,
                );
            }
            s.iter().all(field::is_reduced)
        },
    )
    .is_some_and(|ok| ok.into_iter().all(|b| b))
}

/// Cached forward twiddle tables `[1, w, ..., w^(2^(ext_k-1)-1)]` for CPU-side
/// `coeff_to_extended` (public constants, computed once per `ext_k`).
fn cached_fwd_twiddles(ext_k: u32, omega: Scalar) -> &'static [midnight_curves::Fq] {
    static TWIDDLES: [OnceLock<Vec<midnight_curves::Fq>>; MAX_LOG_N as usize + 1] =
        [const { OnceLock::new() }; MAX_LOG_N as usize + 1];
    TWIDDLES[ext_k as usize]
        .get_or_init(|| {
            // SAFETY: `Scalar` is `[u64; 4]` and `midnight_curves::Fq` is a
            // transparent `blst_fr` (`[u64; 4]`), layout asserted in `msm.rs`.
            let w: midnight_curves::Fq = unsafe { std::mem::transmute(omega) };
            midnight_curves::fft::compute_twiddles(&w, ext_k)
        })
        .as_slice()
}

/// Guard over the backend's reusable CPU coset staging buffer. Maintains the
/// invariant that every element is zero whenever unlocked by wiping the used
/// prefix on every exit path (including early returns and panics).
struct WipedBuf<'a> {
    guard: MutexGuard<'a, Vec<Scalar>>,
    used: usize,
    pool: Option<&'static rayon::ThreadPool>,
}

impl<'a> WipedBuf<'a> {
    fn lock<B: Backend>(used: usize) -> Self {
        let guard = state::<B>()
            .cpu_coset_buf
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        Self {
            guard,
            used,
            pool: crate::msm::par::cpu_pool::<B>(),
        }
    }

    fn ensure_len(&mut self, len: usize) {
        if self.guard.len() < len {
            self.guard.resize(len, [0u64; 4]);
        }
        self.used = len;
    }

    fn as_slice(&self) -> &[Scalar] {
        &self.guard[..self.used]
    }

    fn as_mut(&mut self) -> &mut [Scalar] {
        &mut self.guard[..self.used]
    }
}

impl Drop for WipedBuf<'_> {
    fn drop(&mut self) {
        let n = self.used.min(self.guard.len());
        let slice = &mut self.guard[..n];
        if n >= (1 << 16)
            && let Some(pool) = self.pool
        {
            pool.install(|| {
                slice
                    .par_chunks_mut(1 << 14)
                    .for_each(crate::msm::secret::wipe_slice);
            });
        } else {
            crate::msm::secret::wipe_slice(slice);
        }
    }
}

/// Computes `coeff_to_extended` on the CPU rayon pool for `polys` into
/// pre-zeroed `dst` (`polys.len() * (1 << ext_k)` scalars). Checks every input
/// scalar reduced and scrubs pool thread stacks on exit.
fn cpu_coeff_to_extended<B: Backend>(
    polys: &[&[Scalar]],
    dst: &mut [Scalar],
    k: u32,
    ext_k: u32,
    omega: Scalar,
    zeta: [Scalar; 2],
) -> bool {
    let Some(pool) = crate::msm::par::cpu_pool::<B>() else {
        return false;
    };
    let n = 1usize << k;
    let en = 1usize << ext_k;
    let twiddles = cached_fwd_twiddles(ext_k, omega);
    let reduced = AtomicBool::new(true);

    struct ScrubPool(&'static rayon::ThreadPool);
    impl Drop for ScrubPool {
        fn drop(&mut self) {
            self.0.broadcast(|_| crate::msm::secret::scrub_stack());
        }
    }
    let _scrub = ScrubPool(pool);

    pool.install(|| {
        dst.par_chunks_mut(en)
            .zip(polys.par_iter())
            .for_each(|(slot, src)| {
                let mut ok = true;
                let (dst_chunks, dst_rem) = slot[..n].as_chunks_mut::<3>();
                let (src_chunks, src_rem) = src[..n].as_chunks::<3>();
                for (d, s) in dst_chunks.iter_mut().zip(src_chunks.iter()) {
                    ok &= field::is_reduced(&s[0])
                        & field::is_reduced(&s[1])
                        & field::is_reduced(&s[2]);
                    d[0] = s[0];
                    d[1] = mul(&s[1], &zeta[0]);
                    d[2] = mul(&s[2], &zeta[1]);
                }
                for (r, (d, s)) in dst_rem.iter_mut().zip(src_rem.iter()).enumerate() {
                    ok &= field::is_reduced(s);
                    *d = match r {
                        0 => *s,
                        1 => mul(s, &zeta[0]),
                        _ => mul(s, &zeta[1]),
                    };
                }
                if !ok {
                    reduced.store(false, Relaxed);
                    return;
                }
                // SAFETY: `Scalar` (`[u64; 4]`) and `midnight_curves::Fq` have
                // identical size and alignment (asserted in `msm.rs`); every
                // element in `slot[..n]` was verified reduced and `slot[n..en]`
                // is zeroed by `WipedBuf`'s invariant.
                let fq_slot = unsafe {
                    std::slice::from_raw_parts_mut(
                        slot.as_mut_ptr() as *mut midnight_curves::Fq,
                        en,
                    )
                };
                midnight_curves::fft::fft_coeff_to_extended(fq_slot, twiddles, ext_k, n);
            });
    });
    reduced.load(Relaxed)
}

/// Evaluates quotient numerator rows claimed dynamically from the back of
/// `[gpu_end, cpu_start)` on the CPU rayon pool, writing `out[idx - base]`.
/// Returns `true` on success.
#[allow(clippy::too_many_arguments)]
fn cpu_numerator_steal<B: Backend>(
    proof: &[Scalar],
    res_keys: &[&[Scalar]],
    tmp_keys: &[&[Scalar]],
    words: &[u32],
    uniforms: &[Scalar],
    p: &Params,
    nslots: usize,
    base: usize,
    bounds: &Mutex<(usize, usize)>,
    out: &mut [Scalar],
) -> bool {
    let Some(pool) = crate::msm::par::cpu_pool::<B>() else {
        return false;
    };
    const HANDLE_SHIFT: u32 = 30;
    const HANDLE_MASK: u32 = (1 << 30) - 1;
    const OP_ADD: u32 = 0;
    const OP_SUB: u32 = 1;
    const OP_MUL: u32 = 2;
    const OP_SQUARE: u32 = 3;
    const OP_DOUBLE: u32 = 4;
    const OP_NEGATE: u32 = 5;
    const OP_WORDS: usize = 5;
    const PROG_OPS: usize = 0;
    const PROG_NOPS: usize = 1;
    const PROG_READS: usize = 2;
    const PROG_UNIFORMS: usize = 4;
    const PROG_OUTPUTS: usize = 5;
    const PROG_NOUTPUTS: usize = 6;
    const LOOKUP_WORDS: usize = 4;
    const LOOKUP_AGGREGATOR: usize = 0;
    const LOOKUP_MULTIPLICITIES: usize = 1;
    const LOOKUP_NCHUNKS: usize = 2;
    const LOOKUP_CHUNKS: usize = 3;
    const CHUNK_WORDS: usize = 2;
    const CHUNK_HELPER: usize = 0;
    const CHUNK_PROGRAM: usize = 1;
    const TRASH_WORDS: usize = 4;
    const TRASH_POLY: usize = 0;
    const TRASH_SELECTOR: usize = 1;
    const TRASH_PROGRAM: usize = 2;
    const STEAL_CHUNK: usize = 1024;

    struct ScrubPool(&'static rayon::ThreadPool);
    impl Drop for ScrubPool {
        fn drop(&mut self) {
            self.0.broadcast(|_| crate::msm::secret::scrub_stack());
        }
    }
    let _scrub = ScrubPool(pool);

    let ext_k = p.ext_k as usize;
    let log_scale = p.log_scale;
    let mask = p.mask as usize;
    let nsets = p.perm_nsets as usize;
    let perm_ncolumns = p.perm_ncolumns as usize;
    let perm_chunk_len = p.perm_chunk_len as usize;
    let perm_last_rotation = p.perm_last_rotation;
    let y = p.y;
    let beta = p.beta;
    let gamma = p.gamma;
    let delta = p.delta;
    let delta_start = p.delta_start;
    let omega = p.omega;

    let column = |handle: u32, row: usize| -> Scalar {
        let bank = handle >> HANDLE_SHIFT;
        let index = (handle & HANDLE_MASK) as usize;
        match bank {
            tables::BANK_PROOF => proof[(index << ext_k) + row],
            tables::BANK_KEY_RES => res_keys[index][row],
            _ => tmp_keys[index][row],
        }
    };
    let rotate = |idx: usize, rot: i32| -> usize {
        idx.wrapping_add(((rot as u32) << log_scale) as usize) & mask
    };
    let run_program = |pd: usize, idx: usize, slots: &mut [Scalar]| -> [Scalar; 4] {
        let ops = words[pd + PROG_OPS] as usize;
        let nops = words[pd + PROG_NOPS] as usize;
        let reads = words[pd + PROG_READS] as usize;
        let unis = words[pd + PROG_UNIFORMS] as usize;
        let outs = words[pd + PROG_OUTPUTS] as usize;
        let nouts = words[pd + PROG_NOUTPUTS] as usize;
        let operand = |o: u32, slots: &[Scalar]| -> Scalar {
            let i = (o & HANDLE_MASK) as usize;
            match o >> HANDLE_SHIFT {
                abi::TAG_UNIFORM => uniforms[unis + i],
                abi::TAG_READ => {
                    let h = words[reads + 2 * i];
                    let rot = words[reads + 2 * i + 1] as i32;
                    column(h, rotate(idx, rot))
                }
                _ => slots[i],
            }
        };
        for k in 0..nops {
            let o = ops + OP_WORDS * k;
            let kind = words[o];
            let dst = words[o + 1] as usize;
            let a = operand(words[o + 2], slots);
            let v = match kind {
                OP_ADD => fr_add(&a, &operand(words[o + 3], slots)),
                OP_SUB => fr_sub(&a, &operand(words[o + 3], slots)),
                OP_MUL => mul(&a, &operand(words[o + 3], slots)),
                OP_SQUARE => square(&a),
                OP_DOUBLE => fr_add(&a, &a),
                OP_NEGATE => fr_neg(&a),
                _ => {
                    let prod = mul(&a, &operand(words[o + 3], slots));
                    fr_add(&prod, &operand(words[o + 4], slots))
                }
            };
            slots[dst] = v;
        }
        let mut o = [[0u64; 4]; 4];
        for (k, out_slot) in o.iter_mut().enumerate().take(nouts.min(4)) {
            *out_slot = operand(words[outs + k], slots);
        }
        o
    };

    let out_ptr = out.as_mut_ptr() as usize;
    let out_len = out.len();
    let workers = pool.current_num_threads();
    pool.install(|| {
        rayon::in_place_scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|_| {
                    let mut slots = vec![[0u64; 4]; nslots.max(1)];
                    loop {
                        let range = {
                            let mut g = bounds.lock().unwrap_or_else(|e| e.into_inner());
                            let avail = g.1.saturating_sub(g.0);
                            if avail == 0 {
                                None
                            } else {
                                let take = avail.min(STEAL_CHUNK);
                                g.1 -= take;
                                Some((g.1, g.1 + take))
                            }
                        };
                        let Some((start, end)) = range else {
                            break;
                        };
                        debug_assert!(start >= base && end - base <= out_len);
                        // SAFETY: `[start, end)` is uniquely claimed from `bounds`
                        // and disjoint from every other worker's range and from GPU.
                        let dst = unsafe {
                            std::slice::from_raw_parts_mut(
                                (out_ptr as *mut Scalar).add(start - base),
                                end - start,
                            )
                        };
                        let mut omega_i = if nsets != 0 {
                            pow(&omega, start as u64)
                        } else {
                            ONE
                        };
                        for (off, out_val) in dst.iter_mut().enumerate() {
                            let idx = start + off;
                            let o = run_program(p.gates as usize, idx, &mut slots);
                            let mut value = o[0];
                            let l0 = column(p.l0, idx);
                            let l_last = column(p.l_last, idx);
                            let l_active = column(p.l_active, idx);
                            let r_next = rotate(idx, 1);

                            if nsets != 0 {
                                let zs = p.perm as usize;
                                let cols = zs + nsets;
                                let sig = cols + perm_ncolumns;
                                let r_last = rotate(idx, perm_last_rotation);
                                let z_first = column(words[zs], idx);
                                let z_last = column(words[zs + nsets - 1], idx);
                                value =
                                    fr_add(&mul(&value, &y), &mul(&fr_sub(&ONE, &z_first), &l0));
                                value = fr_add(
                                    &mul(&value, &y),
                                    &mul(&fr_sub(&square(&z_last), &z_last), &l_last),
                                );
                                for s in 1..nsets {
                                    let cur = column(words[zs + s], idx);
                                    let prev = column(words[zs + s - 1], r_last);
                                    value =
                                        fr_add(&mul(&value, &y), &mul(&fr_sub(&cur, &prev), &l0));
                                }
                                let mut current_delta = mul(&delta_start, &omega_i);
                                for s in 0..nsets {
                                    let z = words[zs + s];
                                    let c0 = s * perm_chunk_len;
                                    let c1 = (c0 + perm_chunk_len).min(perm_ncolumns);
                                    let mut left = column(z, r_next);
                                    let mut right = column(z, idx);
                                    for k in c0..c1 {
                                        let x_gamma = fr_add(&column(words[cols + k], idx), &gamma);
                                        let sg = column(words[sig + k], idx);
                                        left = mul(&left, &fr_add(&x_gamma, &mul(&beta, &sg)));
                                        right = mul(&right, &fr_add(&x_gamma, &current_delta));
                                        current_delta = mul(&current_delta, &delta);
                                    }
                                    value = fr_add(
                                        &mul(&value, &y),
                                        &mul(&fr_sub(&left, &right), &l_active),
                                    );
                                }
                                omega_i = mul(&omega_i, &omega);
                            }

                            for li in 0..p.nlookups as usize {
                                let l_off = p.lookups as usize + LOOKUP_WORDS * li;
                                let agg = words[l_off + LOOKUP_AGGREGATOR];
                                let mult = words[l_off + LOOKUP_MULTIPLICITIES];
                                let nchunks = words[l_off + LOOKUP_NCHUNKS] as usize;
                                let chunks = words[l_off + LOOKUP_CHUNKS] as usize;
                                let agg_i = column(agg, idx);
                                value =
                                    fr_add(&mul(&value, &y), &mul(&agg_i, &fr_add(&l0, &l_last)));
                                let (mut sum_helpers, mut table, mut selector) =
                                    ([0u64; 4], [0u64; 4], [0u64; 4]);
                                for fi in 0..nchunks {
                                    let ch = chunks + CHUNK_WORDS * fi;
                                    let co = run_program(
                                        words[ch + CHUNK_PROGRAM] as usize,
                                        idx,
                                        &mut slots,
                                    );
                                    if fi == 0 {
                                        table = co[2];
                                        selector = co[3];
                                    }
                                    let h = column(words[ch + CHUNK_HELPER], idx);
                                    value = fr_sub(
                                        &fr_add(&mul(&value, &y), &mul(&h, &co[1])),
                                        &co[0],
                                    );
                                    sum_helpers = fr_add(&sum_helpers, &h);
                                }
                                let agg_next = column(agg, r_next);
                                let m = column(mult, idx);
                                let inner = fr_add(
                                    &mul(
                                        &fr_sub(
                                            &fr_sub(&agg_next, &agg_i),
                                            &mul(&selector, &sum_helpers),
                                        ),
                                        &table,
                                    ),
                                    &m,
                                );
                                value = fr_add(&mul(&value, &y), &mul(&l_active, &inner));
                            }

                            for ti in 0..p.ntrash as usize {
                                let t_off = p.trash as usize + TRASH_WORDS * ti;
                                let to = run_program(
                                    words[t_off + TRASH_PROGRAM] as usize,
                                    idx,
                                    &mut slots,
                                );
                                let q = column(words[t_off + TRASH_SELECTOR], idx);
                                let t = column(words[t_off + TRASH_POLY], idx);
                                value = fr_add(
                                    &mul(&value, &y),
                                    &fr_sub(&to[0], &mul(&fr_sub(&ONE, &q), &t)),
                                );
                            }

                            *out_val = value;
                        }
                    }
                    crate::msm::secret::wipe_slice(&mut slots);
                });
            }
        });
    });
    true
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
    /// The words of the job's tables: no kernel reads past them (Metal's
    /// kernel ignores it).
    nwords: u32,
    pad: [u32; 4],
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

/// The quotient of `j` into `out`, or with `numerator` the whole numerator
/// (and `out` untouched). Returns the GPU time and the placement counts.
///
/// # Safety
/// As [`quotient`].
unsafe fn run<B: Backend>(
    j: &QuotientJobAbi,
    out: &mut [Scalar],
    numerator: Option<&mut Vec<Scalar>>,
) -> Result<(u64, QuotientStages), Reason> {
    checked::<B>(j, numerator.is_none().then_some(out.len()))?;
    // SAFETY: the caller's contract; `checked` has passed.
    let mut tab = unsafe { Tables::new(j)? };
    // SAFETY: the caller's contract.
    let t_inv = unsafe { abi::slice(j.t_inv, j.nt_inv) };
    check_transform_inputs(j, t_inv)?;

    let mut s = ctx::lock::<B>().map_err(err)?;
    let mut plan = keys::plan::<B>(&tab.keys, 1usize << j.ext_k);
    // every buffer this call can allocate (the sum of `need_bytes`)
    let need = need_of(j, &tab, t_inv, &plan, &s, numerator.is_none());
    let r = if need.fits(s.ctx()) {
        device_run(j, &mut tab, t_inv, &mut plan, &mut s, out, numerator)
    } else {
        Err(Reason::Memory)
    };
    // on every path, before the session ends
    if let Some(l) = plan.lent.take() {
        l.give_back();
    }
    drop(s);
    r
}

/// The device steps of an admitted job.
#[allow(clippy::too_many_arguments)]
fn device_run<B: Backend>(
    j: &QuotientJobAbi,
    tab: &mut Tables<'_>,
    t_inv: &[Scalar],
    plan: &mut Plan<B>,
    s: &mut Session<B>,
    out: &mut [Scalar],
    numerator: Option<&mut Vec<Scalar>>,
) -> Result<(u64, QuotientStages), Reason> {
    let (k, ext_k) = (j.k, j.ext_k);
    let (n, en) = (1usize << k, 1usize << ext_k);
    let ebytes = SCALAR_BYTES << ext_k;
    let threads = stride::<B>(ext_k);
    let c: &'static Ctx<B> = s.ctx();
    let mut stages = QuotientStages::default();
    let mut gpu = 0u64;

    // the transform scratch is per-request (it holds witness values; zeroed at
    // the end of the request)
    let tmp_need = ntt::scratch_bytes(ext_k);
    if tmp_need > 0 {
        s.scratch("quotient-tmp", tmp_need).map_err(err)?;
    }
    let ntemp = plan.places.iter().filter(|p| **p == Place::Temp).count();
    let mut proof_buf = c.alloc_pooled(tab.proof.len().max(1) * ebytes).map_err(err)?;
    let mut src_a = c.alloc_pooled(n * SCALAR_BYTES).map_err(err)?;
    let mut src_b = c.alloc_pooled(n * SCALAR_BYTES).map_err(err)?;
    let mut key_tmp = c.alloc_pooled(ntemp * ebytes).map_err(err)?;
    if let Some(l) = plan.lent.as_mut()
        && l.buf.is_none()
    {
        l.buf = Some(c.alloc_pooled(l.npolys * ebytes).map_err(err)?);
    }
    let tmp_ref: Option<&Buffer<B>> = if tmp_need > 0 {
        s.get_scratch("quotient-tmp")
    } else {
        None
    };
    // bound for an argument that is not read: the transform scratch of a
    // one-pass size, an empty key bank
    let dummy = c.alloc(0).map_err(err)?;
    let tmp = tmp_ref.unwrap_or(&dummy);

    // When `ext_k` uses an even pass count (2 passes for 11 <= ext_k <= 20),
    // copy proof polynomials directly into `proof_buf` and run the GPU coset
    // NTTs in a single batched command buffer. On unified memory, split the
    // proof polynomials across GPU and CPU so both compute extended cosets in
    // parallel (after any key copy/hashing completes on the CPU).
    let fills: Vec<bool> = (0..tab.keys.len()).map(|i| plan.fills(i)).collect();
    let batch_cosets =
        c.unified() && !tab.proof.is_empty() && ntt::plan(ext_k).len().is_multiple_of(2);
    let cpu_n = if batch_cosets
        && ext_k >= 16
        && tab.proof.len() >= 4
        && crate::msm::par::cpu_pool::<B>().is_some()
    {
        1usize
    } else {
        0
    };
    let gpu_n = tab.proof.len() - cpu_n;

    if batch_cosets {
        let reduced = proof_buf
            .host_write(|dst: &mut [Scalar]| {
                let base = dst.as_mut_ptr() as usize;
                let total = gpu_n * n;
                const PART: usize = 1 << 14;
                if total <= 2 * PART {
                    return tab.proof[..gpu_n]
                        .iter()
                        .enumerate()
                        .all(|(i, p)| copy_checked(&mut dst[i * en..i * en + n], p));
                }
                crate::msm::par::par_parts(
                    total,
                    crate::msm::par::host_threads().min(8),
                    PART,
                    |lo, hi| {
                        let mut cur = lo;
                        let mut ok = true;
                        while cur < hi {
                            let pi = cur / n;
                            let off = cur % n;
                            let take = (n - off).min(hi - cur).min(1024);
                            let s = &tab.proof[pi][off..off + take];
                            // SAFETY: `dst` holds `npolys * en` elements (`en >= n`) and
                            // `[cur, cur + take)` maps uniquely to disjoint destination
                            // ranges `pi * en + off .. pi * en + off + take`.
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    s.as_ptr(),
                                    (base as *mut Scalar).add(pi * en + off),
                                    take,
                                );
                            }
                            ok &= s.iter().all(field::is_reduced);
                            cur += take;
                        }
                        ok
                    },
                )
                .is_some_and(|ok| ok.into_iter().all(|b| b))
            })
            .map_err(err)?;
        if !reduced {
            return Err(Reason::Unreduced);
        }
    } else if let Some(first) = tab.proof.first()
        && !src_a.host_write(|d: &mut [Scalar]| copy_checked(d, first)).map_err(err)?
    {
        return Err(Reason::Unreduced);
    }
    // Lock and reserve the reusable host staging buffer for CPU cosets (wiped
    // automatically on every exit path, including early returns and panics).
    let mut cpu_guard = if cpu_n > 0 {
        let mut g = WipedBuf::lock::<B>(cpu_n * en);
        g.ensure_len(cpu_n * en);
        Some(g)
    } else {
        None
    };
    let (key_work, cosets, cpu_reduced) = std::thread::scope(|sc| {
        let cpu_polys = &tab.proof[gpu_n..];
        let keys_ref = &tab.keys;
        let places_ref = &plan.places;
        let check_ref = &plan.check;
        let fills_ref = &fills;
        let lent_mut = &mut plan.lent;
        let key_tmp_mut = &mut key_tmp;
        let cpu_dst = cpu_guard.as_mut().map(WipedBuf::as_mut);
        let zeta = j.zeta;
        let omega = j.extended_omega;
        let kh = sc.spawn(move || {
            let kw = copy_and_hash_keys(
                keys_ref,
                places_ref,
                check_ref,
                fills_ref,
                lent_mut,
                key_tmp_mut,
                en,
            );
            let cpu_ok = match cpu_dst {
                Some(dst) => cpu_coeff_to_extended::<B>(cpu_polys, dst, k, ext_k, omega, zeta),
                None => true,
            };
            (kw, cpu_ok)
        });
        let cosets = if batch_cosets {
            ntt::coeff_to_extended_batch_in_place(
                c,
                &proof_buf,
                gpu_n,
                tmp,
                k,
                ext_k,
                &j.zeta,
            )
            .map_err(err)
        } else {
            proof_cosets(c, j, &tab.proof, [&mut src_a, &mut src_b], &proof_buf, tmp)
        };
        // a panicked key/coset thread reads as unreduced copies
        let (kw, cpu_ok) = kh.join().unwrap_or_else(|_| (KeyWork::default(), false));
        (kw, cosets, cpu_ok)
    });
    gpu += cosets?;
    if !cpu_reduced {
        return Err(Reason::Unreduced);
    }
    if let Some(buf) = cpu_guard.take() {
        proof_buf
            .host_write(|dst: &mut [Scalar]| {
                ctx::par_copy(&mut dst[gpu_n * en..tab.proof.len() * en], buf.as_slice())
            })
            .map_err(err)?;
        drop(buf);
    }
    if !key_work.reduced {
        return Err(Reason::Unreduced);
    }
    // a digest missing (a digest thread or a slot that failed): not settled
    if key_work.digests.len() != plan.check.len() {
        return Err(Reason::DeviceError);
    }
    ctx::release_later::<B>(vec![src_a, src_b]);

    // the digests settle the placement: a stale polynomial is uploaded
    if keys::settle(plan, &tab.keys, &key_work.digests) {
        key_tmp = upload_again(c, &tab.keys, &plan.places, en)?;
    }
    let (bank_of, ntemp) = banks(&plan.places);
    let lagrange = tab.place_keys(&bank_of);
    stages.uploaded_polys = ntemp as u32;
    stages.resident_polys = (plan.places.len() - ntemp) as u32;
    stages.key_copies_made = key_work
        .filled
        .iter()
        .filter(|s| plan.lent.as_ref().is_some_and(|l| l.present[**s]))
        .count() as u32;

    // the numerator, into `nu`
    let mut nu = c.alloc_pooled(ebytes).map_err(err)?;
    let scratch = c
        .alloc_pooled((j.nslots as usize).max(1) * threads * SCALAR_BYTES)
        .map_err(err)?;
    let words = c.upload(&tab.words).map_err(err)?;
    let unis = if tab.uniforms.is_empty() {
        c.alloc(0)
    } else {
        c.upload(&tab.uniforms)
    }
    .map_err(err)?;
    let key_res: &Buffer<B> = plan.lent.as_ref().and_then(|l| l.buf.as_ref()).unwrap_or(&dummy);
    let key_tmp_b: &Buffer<B> = if ntemp > 0 { &key_tmp } else { &dummy };
    let params = numerator_params(j, tab, lagrange, threads);
    let use_cpu_num = c.unified()
        && proof_buf.host_cached()
        && ext_k >= 16
        && crate::msm::par::cpu_pool::<B>().is_some();
    let (ns, runs) = if use_cpu_num {
        let mut max_res_slot = 0usize;
        for pl in &plan.places {
            if let Place::Resident(slot) = *pl {
                max_res_slot = max_res_slot.max(slot + 1);
            }
        }
        let mut res_keys: Vec<&[Scalar]> = vec![&[]; max_res_slot];
        let mut tmp_keys: Vec<&[Scalar]> = Vec::with_capacity(ntemp);
        for (kpoly, pl) in tab.keys.iter().zip(&plan.places) {
            match *pl {
                Place::Resident(slot) => res_keys[slot] = *kpoly,
                Place::Temp => tmp_keys.push(*kpoly),
            }
        }
        // Dynamic front/back work-stealing: GPU runs `[0, base)` immediately
        // while CPU steals 1024-row chunks from `en` downward; when GPU
        // finishes Batch 1 it claims all remaining untouched rows so both cool
        // and thermally throttled clocks stay balanced.
        let base = ((en * 2) / 3) & !255usize;
        let bounds = Mutex::new((base, en));
        let mut cpu_nu = WipedBuf::lock::<B>(en - base);
        cpu_nu.ensure_len(en - base);
        let (gpu_res, cpu_ok) = std::thread::scope(|sc| {
            let cpu_dst = cpu_nu.as_mut();
            let proof_ref = &proof_buf;
            let res_ref = &res_keys;
            let tmp_ref = &tmp_keys;
            let words_ref = &tab.words;
            let unis_ref = &tab.uniforms;
            let bounds_ref = &bounds;
            let nslots = j.nslots as usize;
            let ch = sc.spawn(move || {
                proof_ref
                    .host_read(|proof_slice: &[Scalar]| {
                        cpu_numerator_steal::<B>(
                            proof_slice,
                            res_ref,
                            tmp_ref,
                            words_ref,
                            unis_ref,
                            &params,
                            nslots,
                            base,
                            bounds_ref,
                            cpu_dst,
                        )
                    })
                    .unwrap_or(false)
            });
            let bufs = [&proof_buf, key_res, key_tmp_b, &words, &unis, &nu, &scratch];
            let gpu_res = (|| -> Result<(u64, u32), Reason> {
                let (mut total_ns, mut total_runs) =
                    numerator_range_slice(c, bufs, params, tab.cost_per_row, 0, base, threads)?;
                loop {
                    let next = {
                        let mut g = bounds.lock().unwrap_or_else(|e| e.into_inner());
                        let rem = g.1.saturating_sub(g.0);
                        if rem == 0 {
                            None
                        } else {
                            // Claim all remaining untouched rows so CPU finishes
                            // only its tiny in-flight 1024-row chunks (<1 ms).
                            let lo = g.0;
                            g.0 = g.1;
                            Some((lo, g.0))
                        }
                    };
                    let Some((lo, hi)) = next else {
                        break;
                    };
                    let (n_ns, n_runs) =
                        numerator_range_slice(c, bufs, params, tab.cost_per_row, lo, hi, threads)?;
                    total_ns += n_ns;
                    total_runs += n_runs;
                }
                Ok((total_ns, total_runs))
            })();
            (gpu_res, ch.join().unwrap_or(false))
        });
        let (ns, runs) = gpu_res?;
        if !cpu_ok {
            return Err(Reason::DeviceError);
        }
        let cpu_start = bounds.into_inner().unwrap_or_else(|e| e.into_inner()).1;
        if cpu_start < en {
            nu.host_write(|dst: &mut [Scalar]| {
                ctx::par_copy(
                    &mut dst[cpu_start..en],
                    &cpu_nu.as_slice()[cpu_start - base..en - base],
                )
            })
            .map_err(err)?;
        }
        drop(cpu_nu);
        (ns, runs)
    } else {
        numerator_range_slice(
            c,
            [&proof_buf, key_res, key_tmp_b, &words, &unis, &nu, &scratch],
            params,
            tab.cost_per_row,
            0,
            en,
            threads,
        )?
    };
    gpu += ns;
    stages.numerator_runs = runs;
    if let Some(l) = plan.lent.take() {
        l.give_back();
    }
    // zeroed on the CPU in the background while the device divides and
    // transforms; `end_request` waits for it
    ctx::release_later::<B>(vec![scratch, proof_buf, key_tmp, words, unis]);

    if let Some(v) = numerator {
        v.clear();
        nu.host_read(|d: &[Scalar]| v.extend_from_slice(&d[..en])).map_err(err)?;
        ctx::release_later::<B>(vec![nu]);
        return Ok((gpu, stages));
    }

    // h = nu / t on the coset, then extended_to_coeff, in the inverse
    // transform's command buffers
    let tinv = c.upload(t_inv).map_err(err)?;
    gpu += ntt::divide_extended_to_coeff_dev(
        c,
        &nu,
        tmp,
        ext_k,
        &j.divisor,
        &j.zeta_out,
        &tinv,
        t_inv.len().ilog2(),
        true,
    )
    .map_err(err)?;

    // Write `out` last: a panic becomes a decline, which must leave `out`
    // unchanged, so everything that can fail or panic happens first. The host
    // view of `nu` fails, if at all, before the copy. What follows the copy
    // cannot fail: `ctx::par_copy` and `ctx::release_later` never panic, and
    // the rest only stores counters and takes poison-tolerant locks.
    drop(tinv);
    debug_assert!(plan.lent.is_none());
    nu.host_read(|src: &[Scalar]| ctx::par_copy(out, &src[..out.len()]))
        .map_err(err)?;
    ctx::release_later::<B>(vec![nu]);
    Ok((gpu, stages))
}

/// What the key thread of [`device_run`] reports. The default (a panicked
/// thread) reads as unreduced copies.
#[derive(Default)]
struct KeyWork {
    /// The digests of `Plan::check`, in its order; shorter when a digest is
    /// missing.
    digests: Vec<[u8; 32]>,
    /// Whether every polynomial copied was reduced.
    reduced: bool,
    /// The slots of the lent copy filled.
    filled: Vec<usize>,
}

/// The CPU side of the key placement, run while the device builds the proof's
/// cosets: copies `Temp` polynomials into `key_tmp` and `fills` into the lent
/// copy's slots (checking them reduced), and hashes the polynomials of
/// `check`. A filled slot is hashed over the bytes copied into it, so a slot is
/// only marked present at [`keys::settle`] with verified content.
fn copy_and_hash_keys<B: Backend>(
    keys: &[&[Scalar]],
    places: &[Place],
    check: &[usize],
    fills: &[bool],
    lent: &mut Option<Lent<B>>,
    key_tmp: &mut Buffer<B>,
    en: usize,
) -> KeyWork {
    let host: Vec<&[Scalar]> = check.iter().filter(|&&i| !fills[i]).map(|&i| keys[i]).collect();
    // the copies into `tmp`, one coset per `Temp` place, and into the lent
    // copy's `slots`, one per slot (`keys::plan` places only slots below its
    // `npolys`, and the copy is `npolys` cosets long)
    let copy = |tmp: &mut [Scalar], mut slots: Option<&mut [Scalar]>| {
        let (mut ti, mut reduced, mut filled) = (0, true, Vec::new());
        for (i, (p, pl)) in keys.iter().zip(places).enumerate() {
            match pl {
                Place::Temp => {
                    reduced &= copy_checked(&mut tmp[ti * en..(ti + 1) * en], p);
                    ti += 1;
                }
                Place::Resident(slot) => {
                    if fills[i]
                        && let Some(b) = slots.as_deref_mut()
                    {
                        reduced &= copy_checked(&mut b[slot * en..(slot + 1) * en], p);
                        filled.push(*slot);
                    }
                }
            }
            if !reduced {
                break;
            }
        }
        (reduced, filled)
    };
    let (host_digests, copied) = std::thread::scope(|sc| {
        let dh = sc.spawn(|| keys::digest_many(&host));
        // `fills` implies a lent copy, allocated by `device_run`; without one
        // the slot's digest is missing and the call declines
        let copied = key_tmp.host_write(|tmp: &mut [Scalar]| {
            match lent.as_mut().and_then(|l| l.buf.as_mut()) {
                Some(b) => b.host_write(|slots: &mut [Scalar]| copy(tmp, Some(slots))),
                None => Ok(copy(tmp, None)),
            }
        });
        (dh.join().unwrap_or_default(), copied.and_then(|r| r))
    });
    // a host view that could not be made: no digest, the call declines
    let Ok((reduced, filled)) = copied else {
        return KeyWork {
            digests: Vec::new(),
            reduced: true,
            filled: Vec::new(),
        };
    };
    if !reduced {
        return KeyWork {
            digests: Vec::new(),
            reduced,
            filled,
        };
    }
    // the slots are hashed over the bytes the device gets: the buffer, or the
    // shadow the next run copies in
    let slot_digests = match lent.as_ref().and_then(|l| l.buf.as_ref()) {
        Some(b) => b
            .host_read(|b: &[Scalar]| {
                let in_slots: Vec<&[Scalar]> =
                    filled.iter().map(|s| &b[s * en..(s + 1) * en]).collect();
                keys::digest_many(&in_slots)
            })
            .unwrap_or_default(),
        None => Vec::new(),
    };
    // in the order of `check` (both lists follow it)
    let (mut h, mut f) = (host_digests.into_iter(), slot_digests.into_iter());
    let digests: Option<Vec<[u8; 32]>> =
        check.iter().map(|&i| if fills[i] { f.next() } else { h.next() }).collect();
    KeyWork {
        digests: digests.unwrap_or_default(),
        reduced,
        filled,
    }
}

/// Moves the proof's polynomials to the extended coset, polynomial `i` into
/// place `i` of `cosets`. The CPU copies polynomial `i + 1` (checking it
/// reduced) while the device transforms polynomial `i`. Returns the GPU time.
fn proof_cosets<B: Backend>(
    c: &Ctx<B>,
    j: &QuotientJobAbi,
    proof: &[&[Scalar]],
    src: [&mut Buffer<B>; 2],
    cosets: &Buffer<B>,
    tmp: &Buffer<B>,
) -> Result<u64, Reason> {
    let [src_a, src_b] = src;
    let (k, ext_k) = (j.k, j.ext_k);
    let ebytes = SCALAR_BYTES << ext_k;
    let mut ns = 0;
    for i in 0..proof.len() {
        let (cur, next) = if i % 2 == 0 {
            (&*src_a, &mut *src_b)
        } else {
            (&*src_b, &mut *src_a)
        };
        let following = proof.get(i + 1).copied();
        let (r, copied) = std::thread::scope(|sc| {
            let h = following.map(|f| {
                sc.spawn(move || {
                    next.host_write(|d: &mut [Scalar]| copy_checked(d, f)).unwrap_or(false)
                })
            });
            let r = ntt::coeff_to_extended_dev_at(
                c,
                cur,
                cosets,
                i * ebytes,
                tmp,
                k,
                ext_k,
                &j.zeta,
                false,
            );
            (r, h.is_none_or(|h| h.join().unwrap_or(false)))
        });
        ns += r.map_err(err)?;
        if !copied {
            return Err(Reason::Unreduced);
        }
    }
    Ok(ns)
}

/// The uploads again after [`keys::settle`] moved a stale polynomial out of
/// its slot. The call's other buffers already exist, so this one is held to
/// the limits alone.
fn upload_again<B: Backend>(
    c: &Ctx<B>,
    keys: &[&[Scalar]],
    places: &[Place],
    en: usize,
) -> Result<Buffer<B>, Reason> {
    let (_, nt) = banks(places);
    let bytes = nt * en * SCALAR_BYTES;
    if bytes as u64 > memory_limit::<B>() || bytes as u64 > c.max_buffer_len() {
        return Err(Reason::Memory);
    }
    let mut kt = c.alloc_pooled(bytes).map_err(err)?;
    let reduced = kt
        .host_write(|ks: &mut [Scalar]| {
            let temps = keys.iter().zip(places).filter(|(_, pl)| **pl == Place::Temp);
            // `kt` holds one coset per `Temp` place
            temps
                .enumerate()
                .all(|(ti, (p, _))| copy_checked(&mut ks[ti * en..(ti + 1) * en], p))
        })
        .map_err(err)?;
    if !reduced {
        return Err(Reason::Unreduced);
    }
    Ok(kt)
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

/// The dispatch parameters of the numerator, but for the row range.
fn numerator_params(
    j: &QuotientJobAbi,
    tab: &Tables<'_>,
    lagrange: [u32; 3],
    threads: usize,
) -> Params {
    let omega = j.extended_omega;
    Params {
        ext_k: j.ext_k,
        log_scale: j.ext_k - j.k,
        mask: ((1usize << j.ext_k) - 1) as u32,
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
        nwords: tab.words.len() as u32,
        y: j.y,
        beta: j.beta,
        gamma: j.gamma,
        delta_start: j.delta_start,
        delta: j.delta,
        omega,
        omega_step: pow(&omega, threads as u64),
        ..Default::default()
    }
}

/// The numerator over `[start_row, end_row)`, one command buffer per row range.
/// The first range is sized from an estimate, the next ones from the measured
/// time per row. Returns the GPU time and the command buffers.
fn numerator_range_slice<B: Backend>(
    c: &Ctx<B>,
    bufs: [&Buffer<B>; 7],
    mut p: Params,
    cost_per_row: u64,
    start_row: usize,
    end_row: usize,
    threads: usize,
) -> Result<(u64, u32), Reason> {
    let cost = cost_per_row.max(1) + 8;
    let target = target::<B>();
    let mut rows = ((target / 2) / cost).max(threads as u64) as usize;
    let (mut row0, mut total, mut runs) = (start_row, 0u64, 0u32);
    while row0 < end_row {
        let rows_now = rows.min(end_row - row0).div_ceil(threads) * threads;
        let end = (row0 + rows_now).min(end_row);
        p.row0 = row0 as u32;
        p.row_end = end as u32;
        let disp = threads.min(end - row0);
        let ns = c
            .run(|cmd| {
                cmd.dispatch(
                    "quotient_numerator",
                    disp,
                    &[
                        Arg::Buffer(bufs[0]),
                        Arg::Buffer(bufs[1]),
                        Arg::Buffer(bufs[2]),
                        Arg::Buffer(bufs[3]),
                        Arg::Buffer(bufs[4]),
                        Arg::Buffer(bufs[5]),
                        Arg::Buffer(bufs[6]),
                    ],
                    &p,
                )
            })
            .map_err(err)?
            .1;
        total += ns;
        runs += 1;
        let per_row = (ns as f64 / (end - row0) as f64).max(0.01);
        let want = (target as f64 / per_row) as usize;
        rows = want.clamp(threads, rows.saturating_mul(8).max(threads));
        row0 = end;
    }
    Ok((total, runs))
}
