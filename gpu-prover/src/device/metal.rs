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

//! The binding to `midnight-metal` and `midnight_proofs::gpu`: every call into
//! those crates for the Apple GPU is in this module. Compiled only with the
//! cfg `midnight_metal_linked`, which the build script sets with the `metal`
//! feature. It mirrors the CUDA binding (module `cuda` of `device.rs`).
//!
//! The hooks are installed once per process. Each one is a thin wrapper that
//! declines (the prover then runs the stock CPU code) when its phase is not
//! admitted for the current request, so the admission ladder decides per
//! request which phases reach the device.
//!
//! - Resident SRS tables: [`Device::prepare`] registers the four SRS vectors
//!   the KZG prover commits from (`Coeff`, `LagrangeCoeff`,
//!   `LagrangeDeltaCoeff`, `LagrangeDoubleDeltaCoeff`: phase 1 commits the
//!   advice columns from the delta bases, phase 2 the permutation and LogUp
//!   polynomials from the double-delta bases), so that every commitment of a
//!   proof finds its table. The CUDA binding registers the first two only.
//!   Unlike CUDA's, a table stays on the device from its first MSM until its
//!   handle is dropped: [`Device::evict`] with `tables` drops the handles,
//!   and [`Device::held`] reports the bytes each table holds on the device
//!   (`SrsHandle::device_bytes`).
//! - Resident proving-key copies: `prepare` registers the key's resident
//!   polynomials (`ProvingKey::gpu_resident_polys`) with
//!   `midnight_metal::register_key`; [`Device::set_keep`] sets
//!   `set_keep_resident`; `evict` frees the copies (`evict_key_copies`) and
//!   keeps the registration, which holds no device memory (the CUDA binding
//!   drops it and registers the key again at the next `prepare`); `held`
//!   reports `KeyHandle::resident_bytes`.
//! - The pool of zeroed buffers: `midnight-metal` keeps its large buffers
//!   between requests, zeroed (a new buffer delays the first command buffer
//!   that uses it); `evict` releases them (`release_pool`), and the free
//!   memory of `mem` counts them as free (`midnight_metal::mem_info`).
//! - The erasure report: `midnight_metal::end_request_report` zeroes every
//!   per-request region and, with the request's check, reads each one back
//!   from the host mapping (unified memory) and compares it with zero; check
//!   mode also reads back every idle buffer of the pool of zeroed buffers
//!   (counted in the regions). The quarantine and retry of the CUDA wipe
//!   layer have no counterpart: a region that fails the check, or an idle
//!   pool buffer that reads back non-zero (or cannot be read back), is
//!   zeroed again, read back again and released, never pooled; it is
//!   reported (`nonzero_regions` or `check_failures`), and the request's
//!   erasure fails. A zeroing that panicked since the last report (it was
//!   completed on the calling thread and its buffer released) is reported
//!   as `erasure_failures` and fails the erasure too. A hook called outside
//!   a request wipes its scratch before it returns.
//! - Witness independence of the record: the per-proof record reports ONE
//!   MSM total (hook invocations, served on the device and declined to the
//!   CPU together, which depends only on the circuit) and no MSM decline
//!   reason. The MSM declines of `midnight-metal` (`below-min-k` of a sparse
//!   MSM, `unreduced`, `memory`) depend on the scalar values or are not shown
//!   to be independent of them, so they stay out of `DeviceCounters`; the
//!   per-path split is `midnight_metal::stats()` and `declines()`, for local
//!   benchmarks and debugging only. The MSM seconds are the wall time of all
//!   MSM hook invocations, served or declined.
//! - The label of the record: a proof is `gpu` when a call of an admitted
//!   phase ran on the device (`DeviceCounters::device_ran`). The served MSM
//!   count depends on the scalar values, so this binding reports instead
//!   the MSM hook invocations of an admitted MSM phase that the device takes
//!   by their length (`midnight_metal::msm_sized`: the length thresholds and
//!   the table registration, no scalar read) in `msm_sized`. The served
//!   `fft`, `ifft` and `quotient` calls depend on lengths and sizes only.
//! - The device peak of a request: `midnight_metal::request_peak_bytes`, the
//!   peak since `begin_request` of the bytes in use (buffers handed out, the
//!   resident tables and key copies), without the idle pool of zeroed
//!   buffers; the CUDA binding's peak is the same quantity (the crate's
//!   live bytes, which include its tables and key copies, since
//!   `begin_request`).
//! - The quotient need is `midnight_metal::need_bytes` with the stride of
//!   `midnight_metal::quotient_threads`.
//! - The size thresholds are per class (`midnight_metal::thresholds`); the
//!   device info reports them.
//! - `midnight-metal` counts forward and inverse transforms as separate
//!   classes already, so the `ntt` class split of the CUDA binding is absent.

use std::{
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed},
    },
    time::Instant,
};

use midnight_metal::{self as metal, G1AffineLimbs, G1JacobianLimbs, Scalar};
use midnight_proofs::{
    gpu::{self as hooks, GpuHooks, adapter},
    poly::{Coeff, LagrangeCoeff, LagrangeDeltaCoeff, LagrangeDoubleDeltaCoeff},
};

use super::*;

/// The phases admitted for the current request (bits of [`Phases`]).
static ADMITTED: AtomicU32 = AtomicU32::new(0);
/// The decline and success classes of the crate contract.
const CLASSES: [&str; 4] = ["msm", "fft", "ifft", "quotient"];
const PHASES: [Phases; 4] = [Phases::MSM, Phases::FFT, Phases::IFFT, Phases::QUOTIENT];
/// Hook calls declined because their phase was not admitted, by class, since
/// the last reset.
static NOT_ADMITTED: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
/// Wall time in nanoseconds of the calls the device served, by class, since
/// the last reset. For `msm` (class 0) it is the time of every hook
/// invocation, served or declined (see `msm`).
static SERVED_NS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
/// MSM hook invocations since the last reset, served or declined for any
/// reason. Depends only on the circuit, not on the scalar values.
static MSM_CALLS: AtomicU64 = AtomicU64::new(0);
/// MSM hook invocations of an admitted MSM phase that the device takes by
/// their length (`midnight_metal::msm_sized`), served or then declined, since
/// the last reset. Depends only on the circuit and the tables, not on the
/// scalar values.
static MSM_SIZED: AtomicU64 = AtomicU64::new(0);
/// The request's cancel check ([`Device::set_gate`]).
static GATE: std::sync::RwLock<Option<crate::service::Cancel>> = std::sync::RwLock::new(None);
/// The gate withdrew the device phases.
static WITHDRAWN: AtomicBool = AtomicBool::new(false);
/// This binding made the device context (the first device operation).
static CONTEXT: AtomicBool = AtomicBool::new(false);

/// Whether `p` is admitted; counts a `not-admitted` decline of `class` when it
/// is not. Before a device phase launches, the request's gate runs
/// synchronously (lease still held, client still connected); when it fails,
/// every device phase is withdrawn.
fn admitted(p: Phases, class: usize) -> bool {
    let mut ok = Phases::from_bits(ADMITTED.load(Relaxed)).contains(p);
    if ok && GATE.read().unwrap_or_else(|e| e.into_inner()).as_ref().is_some_and(|g| g()) {
        ADMITTED.store(0, Relaxed);
        WITHDRAWN.store(true, Relaxed);
        ok = false;
    }
    if !ok {
        NOT_ADMITTED[class].fetch_add(1, Relaxed);
    }
    ok
}

/// Adds the time since `t` to the served time of `class` when `served`.
fn timed(class: usize, t: Instant, served: bool) {
    if served {
        SERVED_NS[class].fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
    }
}

fn msm(scalars: &[Scalar], bases: &[G1AffineLimbs]) -> Option<G1JacobianLimbs> {
    MSM_CALLS.fetch_add(1, Relaxed);
    if !admitted(Phases::MSM, 0) {
        return None;
    }
    let t = Instant::now();
    if metal::msm_sized(scalars.len(), bases) {
        MSM_SIZED.fetch_add(1, Relaxed);
    }
    let r = metal::msm_g1(scalars, bases);
    // the time of every invocation, so that the figure does not tell a
    // served MSM from a declined one
    timed(0, t, true);
    r
}

fn fft(a: &mut [Scalar], omega: Scalar, log_n: u32) -> bool {
    if !admitted(Phases::FFT, 1) {
        return false;
    }
    let t = Instant::now();
    let ok = metal::fft(a, omega, log_n);
    timed(1, t, ok);
    ok
}

fn ifft(a: &mut [Scalar], omega_inv: Scalar, log_n: u32, divisor: Scalar) -> bool {
    if !admitted(Phases::IFFT, 2) {
        return false;
    }
    let t = Instant::now();
    let ok = metal::ifft(a, omega_inv, log_n, divisor);
    timed(2, t, ok);
    ok
}

/// # Safety
/// As [`midnight_metal::quotient`]: `job` is the job the prover passes.
unsafe fn quotient(job: *const std::ffi::c_void, out: &mut [Scalar]) -> bool {
    if !admitted(Phases::QUOTIENT, 3) {
        return false;
    }
    let t = Instant::now();
    // SAFETY: the prover's job pointer, passed through unchanged
    let ok = unsafe { metal::quotient(job, out) };
    timed(3, t, ok);
    ok
}

/// Installs the hooks once per process. `false`: another component installed
/// hooks first.
fn install() -> bool {
    static INSTALLED: OnceLock<bool> = OnceLock::new();
    *INSTALLED.get_or_init(|| {
        hooks::install(GpuHooks {
            msm_g1: Some(msm),
            fft: Some(fft),
            ifft: Some(ifft),
            quotient: Some(quotient),
        })
    })
}

/// Decline reasons that mean the device was asked and the CPU code ran
/// instead (a fallback). `below-min-k` and `no-device` are by design,
/// `not-admitted` is the admission's decision.
fn is_fallback(reason: &str) -> bool {
    matches!(reason, "memory" | "device-error" | "shape" | "unreduced")
}

/// The calls into `midnight-metal` that it exports on macOS only (the
/// resident tables and key copies, the quotient's figures). On other targets
/// no device is detected, so these are never reached with a device; they
/// hold nothing and report zero.
#[cfg(target_os = "macos")]
mod sys {
    use midnight_metal as metal;

    use super::{G1AffineLimbs, Scalar};

    pub type SrsHandle = metal::SrsHandle;
    pub type KeyHandle = metal::KeyHandle;

    pub fn register_srs(points: &[G1AffineLimbs]) -> SrsHandle {
        metal::register_srs(points)
    }
    pub fn table_bytes(h: &SrsHandle) -> u64 {
        h.device_bytes()
    }
    pub fn register_key(polys: &[&[Scalar]]) -> KeyHandle {
        metal::register_key(polys)
    }
    pub fn key_bytes(h: &KeyHandle) -> u64 {
        h.resident_bytes()
    }
    pub fn set_keep_resident(keep: bool) {
        metal::set_keep_resident(keep)
    }
    pub fn evict_key_copies() -> u64 {
        metal::evict_key_copies()
    }
    pub fn release_pool() -> u64 {
        metal::release_pool()
    }
    pub fn grid_threads(ext_k: u32) -> u64 {
        u64::from(metal::quotient_threads()).min(1u64 << ext_k)
    }
    pub fn need_bytes(ext_k: u32, proof_polys: usize, key_polys: usize, nslots: u32) -> u64 {
        metal::need_bytes(ext_k, proof_polys, key_polys, nslots) as u64
    }
}

#[cfg(not(target_os = "macos"))]
mod sys {
    use super::{G1AffineLimbs, Scalar};

    pub type SrsHandle = ();
    pub type KeyHandle = ();

    pub fn register_srs(_: &[G1AffineLimbs]) -> SrsHandle {}
    pub fn table_bytes(_: &SrsHandle) -> u64 {
        0
    }
    pub fn register_key(_: &[&[Scalar]]) -> KeyHandle {}
    pub fn key_bytes(_: &KeyHandle) -> u64 {
        0
    }
    pub fn set_keep_resident(_: bool) {}
    pub fn evict_key_copies() -> u64 {
        0
    }
    pub fn release_pool() -> u64 {
        0
    }
    pub fn grid_threads(_: u32) -> u64 {
        0
    }
    pub fn need_bytes(_: u32, _: usize, _: usize, _: u32) -> u64 {
        0
    }
}

/// The erasure report of a request from `midnight_metal::end_request_report`;
/// `Err` when it did not confirm the erasure: a region (a per-request buffer
/// or, in check mode, an idle buffer of the pool) read back non-zero or
/// could not be read back, or a zeroing panicked (`erasure_failures`).
fn wipe_result(r: Result<metal::WipeStats, metal::WipeStats>) -> Result<WipeReport, String> {
    let (w, ok) = match r {
        Ok(w) => (w, true),
        Err(w) => (w, false),
    };
    let report = WipeReport {
        regions: w.regions,
        bytes: w.bytes,
        checked: w.checked,
        nonzero_regions: w.nonzero_regions,
        check_failures: w.check_failures,
        erasure_failures: w.erasure_failures,
    };
    if ok
        && report.nonzero_regions == 0
        && report.check_failures == 0
        && report.erasure_failures == 0
    {
        return Ok(report);
    }
    Err(format!(
        "the device erasure was not confirmed ({} non-zero regions, {} failed checks, {} \
         failed erasures, of {} regions, {} of them idle pool buffers)",
        report.nonzero_regions,
        report.check_failures,
        report.erasure_failures,
        report.regions,
        w.pool_regions
    ))
}

#[derive(Default)]
struct State {
    /// The registered SRS tables (the four SRS vectors of the params). The
    /// registration is host work; the first MSM on a slice uploads its table,
    /// which stays on the device until the handle is dropped.
    srs: Vec<sys::SrsHandle>,
    /// The registered proving-key polynomials. The registration holds no
    /// device memory; the first device quotient makes the copies, which stay
    /// between requests only under `keep`.
    key: Option<sys::KeyHandle>,
    admitted: Phases,
    /// The extended domain of the prepared key reaches the quotient's size
    /// threshold (`quotient` of `midnight_metal::thresholds`).
    sized: bool,
}

/// The Metal device of this process (the system default device).
pub struct MetalDevice {
    state: Mutex<State>,
}

impl MetalDevice {
    /// The device, when the system has one the kernels can use and the hooks
    /// are installed. Makes no device context: the first device operation of
    /// a request makes it.
    pub fn detect() -> Option<MetalDevice> {
        if metal::present().is_none() || !install() {
            return None;
        }
        ADMITTED.store(0, Relaxed);
        Some(MetalDevice {
            state: Mutex::new(State::default()),
        })
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Device for MetalDevice {
    fn info(&self) -> Option<DeviceInfo> {
        let i = metal::present()?;
        // Fields with no Metal counterpart stay empty: `runtime_version` and
        // `driver_version` (Metal has no such figures). `arch` is the Apple
        // GPU family; `build_archs` names the shading language version the
        // shaders are built for; `total_bytes` is the recommended working set
        // of the device (unified memory has no dedicated total). The
        // registry id, `unified_memory`, `max_buffer_length`,
        // `max_threads_per_threadgroup` and `metal3` of `MetalInfo` have no
        // field in `DeviceInfo`.
        Some(DeviceInfo {
            name: i.name.clone(),
            arch: format!("apple{}", i.apple_family),
            build_archs: format!("msl-{}", i.msl_version),
            runtime_version: String::new(),
            driver_version: String::new(),
            total_bytes: i.recommended_max_working_set,
            size_thresholds: metal::thresholds()
                .named()
                .into_iter()
                .map(|(c, k)| (c.to_string(), k))
                .collect(),
        })
    }

    fn mem(&self) -> Option<MemInfo> {
        // only once this binding made the context (reading the memory info
        // would make one): before, the free memory is unknown, and none is
        // reported free
        if !CONTEXT.load(Relaxed) {
            let i = metal::present()?;
            return Some(MemInfo {
                free: 0,
                total: i.recommended_max_working_set,
            });
        }
        metal::mem_info().map(|(free, total)| MemInfo {
            free: free as u64,
            total: total as u64,
        })
    }

    fn held(&self) -> Held {
        // the bytes on the device now, as midnight-metal counts them: the key
        // copies of the registered key (made by the first device quotient,
        // kept between requests under `keep`), and each table uploaded by
        // its first MSM
        let s = self.state();
        Held {
            resident: s.key.as_ref().map_or(0, sys::key_bytes),
            tables: s.srs.iter().map(sys::table_bytes).sum(),
        }
    }

    fn shape(&self, pk: &Pk) -> Option<Shape> {
        let q = hooks::quotient_shape(pk)?;
        Some(Shape {
            k: q.k,
            ext_k: q.ext_k,
            proof_cosets: q.proof_polys as u32,
            resident_polys: q.key_polys as u32,
            nslots: q.nslots,
            grid_threads: sys::grid_threads(q.ext_k),
            quotient_bytes: sys::need_bytes(q.ext_k, q.proof_polys, q.key_polys, q.nslots),
        })
    }

    fn set_keep(&self, keep: bool) {
        sys::set_keep_resident(keep);
    }

    fn prepare(&self, params: &ParamsKZG<Bls12>, pk: &Pk, _check: bool) {
        // the read-back check is the argument of `end_request`
        let mut s = self.state();
        s.sized = pk.get_vk().get_domain().extended_k() >= metal::thresholds().quotient;
        if s.srs.is_empty() {
            s.srs = [
                params.bases::<Coeff>(),
                params.bases::<LagrangeCoeff>(),
                params.bases::<LagrangeDeltaCoeff>(),
                params.bases::<LagrangeDoubleDeltaCoeff>(),
            ]
            .into_iter()
            .map(|b| sys::register_srs(adapter::affine(b)))
            .collect();
        }
        if s.key.is_none()
            && let Some(polys) = pk.gpu_resident_polys()
        {
            let limbs: Vec<&[Scalar]> = polys.iter().map(|p| adapter::scalars(p)).collect();
            s.key = Some(sys::register_key(&limbs));
        }
    }

    fn begin_request(&self, phases: Phases) -> Phases {
        // the first device operation: makes the context
        if phases.is_empty() || !metal::device() {
            return Phases::NONE;
        }
        CONTEXT.store(true, Relaxed);
        // resets the request's peak to the bytes in use now
        metal::begin_request();
        self.state().admitted = phases;
        ADMITTED.store(phases.bits(), Relaxed);
        phases
    }

    fn revoke(&self) -> bool {
        ADMITTED.swap(0, Relaxed) != 0
    }

    fn set_gate(&self, gate: Option<crate::service::Cancel>) {
        *GATE.write().unwrap_or_else(|e| e.into_inner()) = gate;
    }

    fn withdrawn(&self) -> bool {
        WITHDRAWN.swap(false, Relaxed)
    }

    fn activate(&self) {
        // the device context (the first device operation)
        if metal::device() {
            CONTEXT.store(true, Relaxed);
        }
    }

    fn evict(&self, tables: bool) -> u64 {
        // the key copies go, the key registration stays (it holds no device
        // memory); a table is freed only by dropping its handle, which also
        // drops the registration, so it goes with `tables` only; the pool of
        // zeroed buffers goes too (after the key copies, which go back to it)
        let mut s = self.state();
        let before = metal::device_bytes().0;
        sys::evict_key_copies();
        sys::release_pool();
        let mut freed = before.saturating_sub(metal::device_bytes().0);
        if tables {
            freed += s.srs.iter().map(sys::table_bytes).sum::<u64>();
            s.srs.clear();
        }
        freed
    }

    fn end_request(&self, check: bool) -> Result<WipeReport, String> {
        ADMITTED.store(0, Relaxed);
        // every per-request region is zeroed (and with `check` read back)
        // before its release; the key copies go unless `keep`
        wipe_result(metal::end_request_report(check))
    }

    fn counters(&self) -> DeviceCounters {
        let s = self.state();
        let st = metal::stats();
        // no `msm` entry: the served count depends on the scalar values (a
        // sparse MSM is declined); the MSM total is `msm_count`
        let calls: BTreeMap<String, u64> = [
            ("fft", st.fft.served),
            ("ifft", st.ifft.served),
            ("quotient", st.quotient.served),
        ]
        .into_iter()
        .filter(|(_, n)| *n > 0)
        .map(|(k, n)| (k.to_string(), n))
        .collect();
        // MSM declines are not broken out (see the module text); the other
        // classes' reasons depend on lengths and sizes only
        let mut declines: Vec<Fallback> = metal::declines()
            .into_iter()
            .filter(|(class, _, _)| *class != "msm")
            .map(|(class, reason, count)| Fallback {
                phase: class.to_string(),
                reason: reason.to_string(),
                count,
            })
            .collect();
        // the hook layer's own declines (before it calls the hook); its
        // `hook` reason mirrors a decline counted above, so it is left out:
        // each declined call is counted once
        // MSM calls that never reached the hook, counted into the MSM total
        let mut msm_count = MSM_CALLS.load(Relaxed);
        for (class, reason, count) in hooks::hook_declines() {
            if class == "msm" {
                if reason != "hook" {
                    msm_count += count;
                }
            } else if reason != "hook" {
                declines.push(Fallback {
                    phase: class.to_string(),
                    reason: reason.to_string(),
                    count,
                });
            }
        }
        for (i, c) in CLASSES.iter().enumerate().skip(1) {
            let n = NOT_ADMITTED[i].load(Relaxed);
            if n > 0 {
                declines.push(Fallback {
                    phase: c.to_string(),
                    reason: "not-admitted".into(),
                    count: n,
                });
            }
        }
        // a fallback: a decline of an admitted phase for a reason other than
        // size or a missing device
        let fallbacks = declines
            .iter()
            .filter(|d| is_fallback(&d.reason))
            .filter(|d| {
                CLASSES.iter().position(|c| *c == d.phase).is_some_and(|i| {
                    let p = PHASES[i];
                    s.admitted.contains(p) && (p != Phases::QUOTIENT || s.sized)
                })
            })
            .cloned()
            .collect();
        let seconds = |i: usize| SERVED_NS[i].load(Relaxed) as f64 * 1e-9;
        let phase_seconds: BTreeMap<String, f64> = CLASSES
            .iter()
            .enumerate()
            .filter(|(i, _)| *i == 0 || SERVED_NS[*i].load(Relaxed) > 0)
            .map(|(i, c)| (c.to_string(), seconds(i)))
            .collect();
        DeviceCounters {
            // every MSM hook invocation, on the device or declined to the CPU
            msm_count,
            msm_seconds: seconds(0),
            msm_sized: MSM_SIZED.load(Relaxed),
            calls,
            fallbacks,
            declines,
            // the device layer resets its peak only when a request starts on
            // the device
            peak_bytes: if s.admitted.is_empty() {
                0
            } else {
                metal::request_peak_bytes()
            },
            phase_seconds,
        }
    }

    fn reset_counters(&self) {
        metal::reset_stats();
        hooks::reset_hook_declines();
        for c in NOT_ADMITTED.iter().chain(SERVED_NS.iter()).chain([&MSM_CALLS, &MSM_SIZED]) {
            c.store(0, Relaxed);
        }
        self.state().admitted = Phases::NONE;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    /// The binding's counters are process-wide: one test at a time.
    static SERIAL: Mutex<()> = Mutex::new(());

    fn device() -> MetalDevice {
        MetalDevice {
            state: Mutex::new(State::default()),
        }
    }

    /// An MSM call is counted once in the total whether the device serves it
    /// or not; the other classes' declines are counted by phase and reason,
    /// and a size decline is not a fallback.
    #[test]
    fn a_declined_msm_is_counted_in_the_total_and_other_declines_by_reason() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let d = device();
        d.reset_counters();
        d.state().admitted = Phases::MSM;
        ADMITTED.store(Phases::MSM.bits(), Relaxed);
        assert!(msm(&[[0u64; 4]], &[[0u64; 12]]).is_none());
        // a phase that was not admitted is a decline of its own reason
        assert!(!fft(&mut [[0u64; 4]; 2], [0u64; 4], 1));
        assert!(!ifft(&mut [[0u64; 4]; 2], [0u64; 4], 1, [0u64; 4]));
        ADMITTED.store(0, Relaxed);
        let c = d.counters();
        let row = |p: &str, r: &str| {
            c.declines
                .iter()
                .filter(|f| f.phase == p && f.reason == r)
                .map(|f| f.count)
                .sum::<u64>()
        };
        // the MSM decline is in the total, not broken out by reason
        assert_eq!(c.msm_count, 1);
        assert!(c.declines.iter().all(|f| f.phase != "msm"), "{:?}", c.declines);
        assert!(!c.calls.contains_key("msm"));
        assert_eq!(row("fft", "not-admitted"), 1, "{:?}", c.declines);
        assert_eq!(row("ifft", "not-admitted"), 1, "{:?}", c.declines);
        assert_eq!(c.fallback_count(), 0);
    }

    /// The gate runs before a device phase launches; a failed gate withdraws
    /// every device phase at once.
    #[test]
    fn a_failed_gate_withdraws_before_the_launch() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let d = device();
        d.reset_counters();
        d.withdrawn();
        ADMITTED.store(Phases::FFT.union(Phases::IFFT).bits(), Relaxed);
        d.set_gate(Some(std::sync::Arc::new(|| true)));
        assert!(!fft(&mut [[0u64; 4]; 2], [0u64; 4], 1));
        assert_eq!(ADMITTED.load(Relaxed), 0);
        assert!(d.withdrawn());
        d.set_gate(None);
        d.reset_counters();
    }

    /// Before this binding made the context no free memory is reported.
    #[test]
    fn a_cold_device_reports_no_free_memory() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        if metal::present().is_none() || CONTEXT.load(Relaxed) {
            return;
        }
        let m = device().mem().unwrap();
        assert!(m.total > 0);
        assert_eq!(m.free, 0, "the total is not free memory");
    }

    /// With no device phase admitted the peak is 0, and nothing is resident.
    #[test]
    fn no_device_phase_reports_no_peak() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let d = device();
        d.reset_counters();
        assert_eq!(d.counters().peak_bytes, 0);
        assert_eq!(d.held(), Held::default());
        // nothing but the idle pool of zeroed buffers
        let pooled = metal::pool_bytes();
        assert_eq!(d.evict(true), pooled);
    }

    /// A failed read-back check fails the request's erasure report and is
    /// counted; a clean one is the crate's counts.
    #[test]
    fn a_failed_check_is_counted_and_fails_the_wipe() {
        let w = metal::WipeStats {
            regions: 3,
            bytes: 300,
            checked: true,
            checked_regions: 3,
            ..Default::default()
        };
        let r = wipe_result(Ok(w)).unwrap();
        assert_eq!((r.regions, r.bytes, r.checked), (3, 300, true));
        assert_eq!((r.nonzero_regions, r.check_failures), (0, 0));
        let e = wipe_result(Err(metal::WipeStats {
            nonzero_regions: 1,
            ..w
        }))
        .unwrap_err();
        assert!(e.contains("1 non-zero regions, 0 failed checks"), "{e}");
        let e = wipe_result(Err(metal::WipeStats {
            check_failures: 2,
            ..w
        }))
        .unwrap_err();
        assert!(e.contains("2 failed checks"), "{e}");
        // a zeroing that panicked is an erasure failure, reported as such
        let e = wipe_result(Err(metal::WipeStats {
            erasure_failures: 1,
            ..w
        }))
        .unwrap_err();
        assert!(
            e.contains("0 non-zero regions, 0 failed checks, 1 failed erasures"),
            "{e}"
        );
        // and is counted even if the crate's result were `Ok`
        let e = wipe_result(Ok(metal::WipeStats {
            erasure_failures: 1,
            ..w
        }))
        .unwrap_err();
        assert!(e.contains("1 failed erasures"), "{e}");
        let r = wipe_result(Ok(metal::WipeStats {
            pool_regions: 2,
            pool_bytes: 200,
            ..w
        }))
        .unwrap();
        assert_eq!(r.erasure_failures, 0);
    }

    /// `keep` turns on the crate's keep setting.
    #[cfg(target_os = "macos")]
    #[test]
    fn keep_sets_the_crate_policy() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let d = device();
        d.set_keep(true);
        assert!(metal::keep_resident());
        d.set_keep(false);
        assert!(!metal::keep_resident());
    }

    /// The registered tables and key hold no device memory until a device
    /// call uploads them; `prepare` registers the four SRS vectors and the
    /// key; the quotient need is the crate's own figure.
    #[cfg(all(target_os = "macos", feature = "test-seed"))]
    #[test]
    fn prepare_registers_and_held_counts_device_bytes_only() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let d = device();
        let c = midnight_proofs::gpu::test_switches::RandomCircuit::new(1, true, 8);
        d.prepare(c.params(), c.pk(), false);
        assert_eq!(d.state().srs.len(), 4, "the four SRS vectors");
        assert!(d.state().key.is_some(), "the key is registered");
        assert_eq!(d.held(), Held::default());
        let q = hooks::quotient_shape(c.pk()).unwrap();
        let shape = d.shape(c.pk()).unwrap();
        assert_eq!(
            shape.quotient_bytes,
            metal::need_bytes(q.ext_k, q.proof_polys, q.key_polys, q.nslots) as u64
        );
        assert_eq!(
            shape.grid_threads,
            u64::from(metal::quotient_threads()).min(1 << q.ext_k)
        );
        let pooled = metal::pool_bytes();
        assert_eq!(d.evict(true), pooled);
        assert!(d.state().srs.is_empty());
    }
}
