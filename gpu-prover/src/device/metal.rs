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
//! Where `midnight-metal` has no counterpart of a `midnight-gpu` function the
//! binding does the neutral thing and names the missing function:
//!
//! - Resident SRS tables: `midnight_metal::register_srs` exists and
//!   [`Device::prepare`] uses it, as the CUDA binding does. Unlike CUDA's, the
//!   device table persists from the first MSM until the handle is dropped:
//!   there is no call that frees an uploaded table and keeps the registration,
//!   and none that reports whether a table is uploaded. [`Device::evict`] with
//!   `tables` drops the handles (the only way to free them) and [`Device::held`]
//!   reports their point bytes once an upload has happened (an upper bound).
//!   Resident key polynomials have no counterpart (`register_key`,
//!   `set_keep_resident`, `evict_key_copies`): [`Device::set_keep`] only
//!   stores the flag and `evict` frees no key bytes.
//! - The erasure report. `midnight_metal::end_request` returns one `bool`
//!   (every per-request buffer zeroed and released); there is no region count,
//!   no quarantine and no read-back check. The report carries what the crate
//!   states: the bytes it held when the request ended, and failure when it
//!   returns `false`.
//! - The quotient memory need. `midnight-metal` has no `quotient::need_bytes`
//!   or `grid_threads`; the need uses the formula of `midnight_gpu::quotient::
//!   need_bytes` with a scratch stride set by [`GRID_THREADS`].
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
    poly::{Coeff, LagrangeCoeff},
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
/// the last reset.
static SERVED_NS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
/// The request's cancel check ([`Device::set_gate`]).
static GATE: std::sync::RwLock<Option<crate::service::Cancel>> = std::sync::RwLock::new(None);
/// The gate withdrew the device phases.
static WITHDRAWN: AtomicBool = AtomicBool::new(false);
/// This binding made the device context (the first device operation).
static CONTEXT: AtomicBool = AtomicBool::new(false);

/// The scratch stride, in rows, that the quotient need assumes. The CUDA
/// binding takes 512 threads per multiprocessor; the Metal quotient kernel
/// does not exist yet and has no such figure. This is the largest stride the
/// formula allows (2^24, the upper clamp of `midnight_gpu::quotient::
/// grid_threads`), which makes the stride the rounded extended domain size and
/// the need an upper bound. Replace it with the kernel's own figure once it
/// has one.
const GRID_THREADS: u64 = 1 << 24;

/// Device bytes of one quotient job: the arithmetic of
/// `midnight_gpu::quotient::need_bytes` (extended cosets of the proof's and of
/// the key's polynomials, the numerator and the scratch slots) with the stride
/// of [`GRID_THREADS`].
fn need_bytes(ext_k: u32, proof_polys: usize, key_polys: usize, nslots: u32) -> usize {
    let en = 1usize << ext_k;
    let stride = GRID_THREADS.min(((en as u64).div_ceil(256) * 256).max(256));
    (proof_polys + 1 + key_polys) * en * 32 + nslots as usize * stride as usize * 32
}

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
    if !admitted(Phases::MSM, 0) {
        return None;
    }
    let t = Instant::now();
    let r = metal::msm_g1(scalars, bases);
    timed(0, t, r.is_some());
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
/// `not-admitted` is the admission's decision. `not-implemented` is a kernel
/// `midnight-metal` does not have yet: the phase was admitted and the CPU ran.
fn is_fallback(reason: &str) -> bool {
    matches!(
        reason,
        "memory" | "device-error" | "shape" | "unreduced" | "not-implemented"
    )
}

/// A registered SRS table. `midnight-metal` exports registration on macOS
/// only, so elsewhere the handle is empty and registration does nothing.
#[cfg(target_os = "macos")]
type SrsHandle = metal::SrsHandle;
#[cfg(not(target_os = "macos"))]
type SrsHandle = ();

#[cfg(target_os = "macos")]
fn register(points: &[G1AffineLimbs]) -> SrsHandle {
    metal::register_srs(points)
}
#[cfg(not(target_os = "macos"))]
fn register(_: &[G1AffineLimbs]) -> SrsHandle {}

#[cfg(target_os = "macos")]
fn handle_len(h: &SrsHandle) -> u64 {
    h.len() as u64
}
#[cfg(not(target_os = "macos"))]
fn handle_len(_: &SrsHandle) -> u64 {
    0
}

/// Table uploads since process start (`msm_counts().2`).
fn uploads() -> u64 {
    #[cfg(target_os = "macos")]
    return metal::msm_counts().2;
    #[cfg(not(target_os = "macos"))]
    0
}

/// Bytes of one affine point on the device (twelve `u64` limbs).
const POINT_BYTES: u64 = 96;

#[derive(Default)]
struct State {
    /// The registered SRS tables (monomial and Lagrange bases), kept for the
    /// life of the params. Registration is host work; the first MSM on a
    /// slice uploads its table, which stays on the device until the handle is
    /// dropped.
    srs: Vec<SrsHandle>,
    /// `uploads()` when the tables were registered: a larger count now means
    /// a table has been uploaded.
    uploads_at_register: u64,
    admitted: Phases,
    /// The extended domain of the prepared key reaches the device size
    /// threshold (`midnight_metal::min_k`).
    sized: bool,
    /// Resident policy `keep`. Recorded only: `midnight-metal` has no
    /// resident copies (`set_keep_resident`).
    keep: bool,
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
        // No key copies are resident. The MSM tables stay on the device after
        // their first MSM; midnight-metal does not say which table was
        // uploaded, so once any upload happened since registration all
        // registered points are counted (an upper bound, exact when both
        // were uploaded in full). `evict(true)` frees exactly this.
        let s = self.state();
        let tables = if !s.srs.is_empty() && uploads() > s.uploads_at_register {
            s.srs.iter().map(handle_len).sum::<u64>() * POINT_BYTES
        } else {
            0
        };
        Held {
            resident: 0,
            tables,
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
            grid_threads: GRID_THREADS,
            quotient_bytes: need_bytes(q.ext_k, q.proof_polys, q.key_polys, q.nslots) as u64,
        })
    }

    fn set_keep(&self, keep: bool) {
        // `midnight_metal::set_keep_resident(keep)` goes here once it exists
        self.state().keep = keep;
    }

    fn prepare(&self, params: &ParamsKZG<Bls12>, pk: &Pk, _check: bool) {
        // `midnight_metal::register_key` (`pk.gpu_resident_polys()`) goes here
        // once it exists. The read-back check flag has no Metal counterpart
        // (`end_request` takes none).
        let mut s = self.state();
        s.sized = pk.get_vk().get_domain().extended_k() >= metal::min_k();
        if s.srs.is_empty() {
            let coeff = params.bases::<Coeff>();
            let lagrange = params.bases::<LagrangeCoeff>();
            s.srs = vec![
                register(adapter::affine(coeff)),
                register(adapter::affine(lagrange)),
            ];
            s.uploads_at_register = uploads();
        }
    }

    fn begin_request(&self, phases: Phases) -> Phases {
        // the first device operation: makes the context
        if phases.is_empty() || !metal::device() {
            return Phases::NONE;
        }
        CONTEXT.store(true, Relaxed);
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
        // No key copies exist (`evict_key_copies` is missing). A table can
        // only be freed by dropping its handle, which also drops the
        // registration, so it is freed with `tables` only.
        if !tables {
            return 0;
        }
        let freed = self.held().tables;
        self.state().srs.clear();
        freed
    }

    fn end_request(&self, check: bool) -> Result<WipeReport, String> {
        let _ = check;
        ADMITTED.store(0, Relaxed);
        // the bytes `midnight-metal` holds in Metal buffers when the request
        // ends: every one of them is zeroed and released by `end_request`
        let bytes = metal::device_bytes().0;
        if metal::end_request() {
            // regions, a read-back check and quarantine have no Metal source:
            // 0 regions, `checked: false`
            return Ok(WipeReport {
                regions: 0,
                bytes,
                checked: false,
                nonzero_regions: 0,
                check_failures: 0,
                erasure_failures: 0,
            });
        }
        Err("the device erasure was not confirmed (midnight-metal could not zero and release \
             every per-request buffer)"
            .to_string())
    }

    fn counters(&self) -> DeviceCounters {
        let s = self.state();
        let st = metal::stats();
        let calls: BTreeMap<String, u64> = [
            ("msm", st.msm.served),
            ("fft", st.fft.served),
            ("ifft", st.ifft.served),
            ("quotient", st.quotient.served),
        ]
        .into_iter()
        .filter(|(_, n)| *n > 0)
        .map(|(k, n)| (k.to_string(), n))
        .collect();
        let mut declines: Vec<Fallback> = metal::declines()
            .into_iter()
            .map(|(class, reason, count)| Fallback {
                phase: class.to_string(),
                reason: reason.to_string(),
                count,
            })
            .collect();
        // the hook layer's own declines (before it calls the hook); its
        // `hook` reason mirrors a decline counted above, so it is left out:
        // each declined call is counted once
        for (class, reason, count) in hooks::hook_declines() {
            if reason != "hook" {
                declines.push(Fallback {
                    phase: class.to_string(),
                    reason: reason.to_string(),
                    count,
                });
            }
        }
        for (i, c) in CLASSES.iter().enumerate() {
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
            msm_count: st.msm.served,
            msm_seconds: seconds(0),
            calls,
            fallbacks,
            declines,
            // the device layer resets its peak only when a request starts on
            // the device
            peak_bytes: if s.admitted.is_empty() {
                0
            } else {
                metal::device_bytes().1
            },
            phase_seconds,
        }
    }

    fn reset_counters(&self) {
        metal::reset_stats();
        hooks::reset_hook_declines();
        for c in NOT_ADMITTED.iter().chain(SERVED_NS.iter()) {
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

    /// A declined call is counted once, by phase and reason, and a size
    /// decline is not a fallback.
    #[test]
    fn a_below_min_k_msm_is_one_decline_by_reason() {
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
        assert_eq!(row("msm", "below-min-k"), 1, "{:?}", c.declines);
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
        assert_eq!(d.evict(true), 0);
    }

    /// The need is the formula of `midnight_gpu::quotient::need_bytes`.
    #[test]
    fn need_is_the_cuda_formula_with_the_assumed_stride() {
        let en = 1usize << 12;
        assert_eq!(need_bytes(12, 5, 7, 3), (5 + 1 + 7) * en * 32 + 3 * en * 32);
    }
}
