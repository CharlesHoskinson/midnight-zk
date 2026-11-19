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

//! # midnight_metal
//!
//! Metal kernels for the accelerator hooks of `midnight-proofs` (feature
//! `gpu`) on Apple GPUs: the counterpart of the CUDA crate `midnight-gpu`,
//! with the same entry points, so a process installs them the same way:
//!
//! ```ignore
//! midnight_proofs::gpu::install(midnight_proofs::gpu::GpuHooks {
//!     msm_g1: Some(midnight_metal::msm_g1),
//!     fft: Some(midnight_metal::fft),
//!     ifft: Some(midnight_metal::ifft),
//!     quotient: Some(midnight_metal::quotient),
//! });
//! ```
//!
//! This crate does not depend on midnight-curves or midnight-proofs.
//!
//! ## The hook contract
//!
//! 1. Every entry point returns the exact value of the stock CPU code on the
//!    same input, or declines (`None` / `false`).
//! 2. A decline leaves the caller's buffer unchanged: a kernel writes the
//!    caller's slice only with the complete device result, after the last
//!    command buffer has completed without error.
//! 3. No entry point draws a random value or touches a transcript. Any
//!    pseudo-random value a kernel needs (for example a load-balancing mask)
//!    is a fixed public function of the input shape.
//! 4. The device is never a correctness dependency: every failure (no device,
//!    an allocation, a pipeline, a command buffer in the error state) is a
//!    counted decline, never a panic.
//!
//! ## Data layout
//!
//! The in-memory form of the `midnight-curves` types (through the zero-copy
//! adapter `midnight_proofs::gpu::adapter`), the same bytes on the device:
//! - [`Scalar`]: `blst_fr`, four little-endian `u64` limbs of the Montgomery
//!   form (`R = 2^256`), reduced modulo `r`. Device type `Fr`
//!   (`shaders/include/fr.h`), eight 32-bit limbs.
//! - [`G1AffineLimbs`]: `blst_p1_affine`, `x` then `y`, six limbs each,
//!   Montgomery (`R = 2^384`); the point at infinity is all zero. Device type
//!   `Fp` (`shaders/include/fp.h`), twelve 32-bit limbs per coordinate.
//! - [`G1JacobianLimbs`]: `blst_p1`, Jacobian `X`, `Y`, `Z`.
//!
//! ## Files
//!
//! - `src/lib.rs`: this API, admission, environment.
//! - `src/account.rs`: decline and service accounting. A kernel ends every
//!   call with exactly one of `account::served(class, gpu_nanos)` or
//!   `account::decline(class, reason)`.
//! - `src/ctx.rs` ([`ctx`]): the device context: device, queue, library,
//!   pipeline cache, shared buffers (scoped, and per-request ones released by
//!   [`end_request`]), command buffers with error reporting, byte accounting,
//!   and what unified memory means for transfers.
//! - `src/field.rs` ([`field`]): moduli and the reduction check.
//! - `src/msm.rs`, `src/ntt.rs`, `src/quotient.rs`: one kernel family each,
//!   compiled on macOS only (on other targets the entry points decline before
//!   reaching them). Each holds one `pub(crate)` function called by the entry
//!   point after admission; today each declines with `not-implemented`.
//! - `build.rs`: shader build (metallib with `xcrun metal`, else embedded
//!   source compiled at first use).
//! - `shaders/include/mont.h`: Montgomery arithmetic over N 32-bit limbs
//!   (add, sub, neg, double, the multiplication strategies, square, the
//!   reduced-ness check); `fr.h`, `fp.h`: the two fields on it (`fr_mul`,
//!   `fp_add`, ...).
//! - `shaders/*.metal`: kernel files. Each file is its own translation unit
//!   when the toolchain is present, and all of them form one translation
//!   unit on the source path, so a new kernel file:
//!   - includes the headers it needs (`#include "fr.h"`, `"fp.h"`); header
//!     guards make repeated includes harmless;
//!   - prefixes its kernel names with its file stem (`ntt_*`, `msm_*`,
//!     `quotient_*`) and keeps its helpers in a namespace of that name, so no
//!     two files clash in the shared library;
//!   - is picked up by `build.rs` without any other change.
//!
//!   `field_test.metal` holds the field test and benchmark kernels,
//!   `diag.metal` the context's diagnostic kernels.
//!
//! The Metal Shading Language version is 3.2 (macOS 15+): 64-bit integer
//! functions (`mulhi` on `ulong`) arrived in 2.3; 3.2 adds device-coherent
//! memory, `atomic_thread_fence` with scopes and barrier memory scopes, which
//! single-pass cross-threadgroup algorithms (scans, sorts) in the MSM may
//! use. Every Apple Silicon Mac runs macOS 15.
//!
//! ## Environment
//!
//! - `MIDNIGHT_METAL=off`: use no device (every call declines `no-device`).
//! - `MIDNIGHT_METAL_MIN_K` (default 14): the smallest log2 size sent to the
//!   device; see [`set_min_k`].
//! - `MIDNIGHT_METAL_SHADER_SOURCE=1` (run time): compile the embedded source
//!   instead of loading the embedded metallib. At build time it skips the
//!   toolchain.
//!
//! ## Requests
//!
//! Call [`begin_request`] and [`end_request`] around each proof:
//! [`end_request`] zeroes and releases the per-request buffers (which can
//! hold witness values). Scoped buffers are zeroed when dropped.

#![deny(missing_docs)]
#![deny(missing_debug_implementations)]
#![cfg_attr(not(target_os = "macos"), allow(dead_code, unused_imports))]

use std::{
    ffi::c_void,
    sync::{
        OnceLock,
        atomic::{AtomicU32, Ordering::Relaxed},
    },
};

mod account;
pub mod field;

#[cfg(target_os = "macos")]
pub mod ctx;
#[cfg(target_os = "macos")]
mod msm;
#[cfg(target_os = "macos")]
pub use msm::{
    MsmStages, SrsHandle, msm_counts, msm_g1_stages, register_srs, srs_handles, srs_stale,
};
#[cfg(target_os = "macos")]
mod ntt;
/// Host-to-host wrappers of the NTT device building blocks, for this crate's
/// integration tests (not a stable interface).
#[cfg(target_os = "macos")]
#[doc(hidden)]
pub use ntt::test_support as ntt_test_support;
#[cfg(target_os = "macos")]
mod quotient;

#[cfg(all(test, target_os = "macos"))]
mod tests;

use account::{Class, Reason};

/// A scalar: `blst_fr` limbs, Montgomery form (see the crate text).
pub type Scalar = [u64; 4];
/// An affine G1 point: `blst_p1_affine` limbs.
pub type G1AffineLimbs = [u64; 12];
/// A Jacobian G1 point: `blst_p1` limbs.
pub type G1JacobianLimbs = [u64; 18];

/// The largest log2 transform size taken.
pub const MAX_LOG_N: u32 = 28;
/// The largest log2 MSM length taken; a longer one declines (`shape`).
pub const MAX_MSM_LOG: u32 = 28;

/// The Metal device as its properties describe it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetalInfo {
    /// Device name (`MTLDevice.name`).
    pub name: String,
    /// `MTLDevice.registryID`.
    pub registry_id: u64,
    /// `hasUnifiedMemory`: the GPU shares the system memory (all Apple
    /// Silicon).
    pub unified_memory: bool,
    /// `recommendedMaxWorkingSetSize`: the bytes the GPU can use without
    /// hurting performance, Metal's guidance for the total of this process.
    pub recommended_max_working_set: u64,
    /// `maxBufferLength`: the largest single buffer in bytes.
    pub max_buffer_length: u64,
    /// `maxThreadsPerThreadgroup.width`.
    pub max_threads_per_threadgroup: u64,
    /// The highest `MTLGPUFamilyAppleN` supported (0: none).
    pub apple_family: u32,
    /// Supports `MTLGPUFamilyMetal3`.
    pub metal3: bool,
    /// The Metal Shading Language version of the shaders.
    pub msl_version: &'static str,
}

/// Whether the Metal code is compiled in (a macOS target).
pub const fn metal_built() -> bool {
    cfg!(target_os = "macos")
}

fn disabled() -> bool {
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| std::env::var("MIDNIGHT_METAL").as_deref() == Ok("off"))
}

/// The Metal device, without making the context (no shader library is
/// loaded): `None` on other targets, with no device, or with
/// `MIDNIGHT_METAL=off`.
pub fn present() -> Option<&'static MetalInfo> {
    if disabled() {
        return None;
    }
    #[cfg(target_os = "macos")]
    {
        ctx::probe()
    }
    #[cfg(not(target_os = "macos"))]
    None
}

/// Whether device calls can run: [`present`], and the context (queue, shader
/// library) is made; the first call makes it.
pub fn device() -> bool {
    if present().is_none() {
        return false;
    }
    #[cfg(target_os = "macos")]
    {
        ctx::context().is_ok()
    }
    #[cfg(not(target_os = "macos"))]
    false
}

/// Where the shader library came from: `metallib` (compiled at build time),
/// or `source ...` (compiled from the embedded source at first use, with the
/// reason). `None` before the context is made or when it failed.
pub fn library_source() -> Option<&'static str> {
    #[cfg(target_os = "macos")]
    {
        ctx::context().ok().map(|c| c.library_source())
    }
    #[cfg(not(target_os = "macos"))]
    None
}

static MIN_K: AtomicU32 = AtomicU32::new(u32::MAX);

/// The smallest log2 size sent to the device: [`set_min_k`], else
/// `MIDNIGHT_METAL_MIN_K` (ignored above [`MAX_LOG_N`]), else 14.
pub fn min_k() -> u32 {
    let k = MIN_K.load(Relaxed);
    if k != u32::MAX {
        return k;
    }
    let k = std::env::var("MIDNIGHT_METAL_MIN_K")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .filter(|k| *k <= MAX_LOG_N)
        .unwrap_or(14);
    let _ = MIN_K.compare_exchange(u32::MAX, k, Relaxed, Relaxed);
    MIN_K.load(Relaxed)
}

/// Sets the smallest log2 size sent to the device. Returns `false` and keeps
/// the old value when `k` is above [`MAX_LOG_N`] (no size would be admitted).
pub fn set_min_k(k: u32) -> bool {
    if k > MAX_LOG_N {
        return false;
    }
    MIN_K.store(k, Relaxed);
    true
}

/// Calls served by the device in one hook class since the last
/// [`reset_stats`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClassStats {
    /// Calls the device computed.
    pub served: u64,
    /// GPU time of those calls, nanoseconds (command-buffer GPU start to end).
    pub device_nanos: u64,
}

/// Per-class service counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// MSMs.
    pub msm: ClassStats,
    /// Forward FFTs.
    pub fft: ClassStats,
    /// Inverse FFTs.
    pub ifft: ClassStats,
    /// Quotients.
    pub quotient: ClassStats,
}

/// The counters.
pub fn stats() -> Stats {
    Stats {
        msm: account::class_stats(Class::Msm),
        fft: account::class_stats(Class::Fft),
        ifft: account::class_stats(Class::Ifft),
        quotient: account::class_stats(Class::Quotient),
    }
}

/// Every decline since the last [`reset_stats`]: `(class, reason, count)`.
/// Classes: `msm`, `fft`, `ifft`, `quotient`. Reasons: `no-device` (another
/// target, no usable device, `MIDNIGHT_METAL=off`), `below-min-k`, `shape`
/// (a length, root, job layout or version the device does not take),
/// `unreduced` (a scalar not below the modulus), `memory` (a buffer above
/// `maxBufferLength`, a failed allocation or a budget), `device-error` (a
/// pipeline, encoder or command buffer failed), `not-implemented`.
pub fn declines() -> Vec<(&'static str, &'static str, u64)> {
    account::declines()
}

/// Resets [`stats`] and [`declines`].
pub fn reset_stats() {
    account::reset();
}

/// The admission of an MSM on its lengths: an empty input, unequal lengths
/// or a length above `2^max_log` is `shape`; below `2^min_k()` is
/// `below-min-k`.
pub fn msm_admission(scalars: usize, bases: usize, max_log: u32) -> Result<(), &'static str> {
    if scalars == 0 || scalars != bases || scalars > (1usize << max_log.min(62)) {
        return Err(Reason::Shape.name());
    }
    if scalars < (1usize << min_k().min(40)) {
        return Err(Reason::BelowMinK.name());
    }
    Ok(())
}

/// G1 MSM `sum scalars[i] * bases[i]` (equal lengths): the hook
/// `GpuHooks::msm_g1`. `None`: declined.
pub fn msm_g1(scalars: &[Scalar], bases: &[G1AffineLimbs]) -> Option<G1JacobianLimbs> {
    if let Err(why) = msm_admission(scalars.len(), bases.len(), MAX_MSM_LOG) {
        let r = if why == Reason::Shape.name() {
            Reason::Shape
        } else {
            Reason::BelowMinK
        };
        account::decline(Class::Msm, r);
        return None;
    }
    if !device() {
        account::decline(Class::Msm, Reason::NoDevice);
        return None;
    }
    if !field::all_reduced(scalars) {
        account::decline(Class::Msm, Reason::Unreduced);
        return None;
    }
    #[cfg(target_os = "macos")]
    {
        msm::msm_g1(scalars, bases)
    }
    #[cfg(not(target_os = "macos"))]
    None
}

/// The admission shared by [`fft`] and [`ifft`].
fn transform_admission(
    class: Class,
    a: &[Scalar],
    omega: &Scalar,
    log_n: u32,
    divisor: Option<&Scalar>,
) -> bool {
    if log_n == 0 || log_n > MAX_LOG_N || a.len() != 1usize << log_n {
        return account::decline(class, Reason::Shape);
    }
    if log_n < min_k() {
        return account::decline(class, Reason::BelowMinK);
    }
    if !device() {
        return account::decline(class, Reason::NoDevice);
    }
    if !field::is_reduced(omega)
        || divisor.is_some_and(|d| !field::is_reduced(d))
        || !field::all_reduced(a)
    {
        return account::decline(class, Reason::Unreduced);
    }
    true
}

/// The FFT of `a` (length `2^log_n`) at `omega`, natural order, in place:
/// the hook `GpuHooks::fft`. `false`: declined, `a` unchanged.
pub fn fft(a: &mut [Scalar], omega: Scalar, log_n: u32) -> bool {
    if !transform_admission(Class::Fft, a, &omega, log_n, None) {
        return false;
    }
    #[cfg(target_os = "macos")]
    {
        ntt::fft(a, omega, log_n)
    }
    #[cfg(not(target_os = "macos"))]
    false
}

/// The FFT of `a` at `omega_inv`, then each element times `divisor`, in
/// place: the hook `GpuHooks::ifft`. `false`: declined, `a` unchanged.
pub fn ifft(a: &mut [Scalar], omega_inv: Scalar, log_n: u32, divisor: Scalar) -> bool {
    if !transform_admission(Class::Ifft, a, &omega_inv, log_n, Some(&divisor)) {
        return false;
    }
    #[cfg(target_os = "macos")]
    {
        ntt::ifft(a, omega_inv, log_n, divisor)
    }
    #[cfg(not(target_os = "macos"))]
    false
}

/// The quotient `h(X)` of a job (`midnight_proofs::gpu::abi::QuotientJobAbi`):
/// the hook `GpuHooks::quotient`, writing the first `out.len()` coefficients.
/// `false`: declined, `out` unchanged.
///
/// # Safety
/// `job` is null or points to a valid `QuotientJobAbi` whose pointers live
/// for the call (the prover passes such a job).
pub unsafe fn quotient(job: *const c_void, out: &mut [Scalar]) -> bool {
    if job.is_null() {
        return account::decline(Class::Quotient, Reason::Shape);
    }
    if !device() {
        return account::decline(Class::Quotient, Reason::NoDevice);
    }
    #[cfg(target_os = "macos")]
    {
        // SAFETY: the caller's contract.
        unsafe { quotient::quotient(job, out) }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = out;
        false
    }
}

/// Starts a request: resets the peak of [`device_bytes`].
pub fn begin_request() {
    #[cfg(target_os = "macos")]
    ctx::begin_request();
}

/// Ends a request (call it between proofs): waits for any device work in
/// progress, then zeroes and releases every per-request buffer. `true` when
/// every buffer was zeroed and released (always, on other targets or before
/// the context exists).
pub fn end_request() -> bool {
    #[cfg(target_os = "macos")]
    {
        ctx::end_request()
    }
    #[cfg(not(target_os = "macos"))]
    true
}

/// `(now, peak)` bytes this process holds in Metal buffers of this crate;
/// the peak is since process start or the last [`begin_request`].
pub fn device_bytes() -> (u64, u64) {
    #[cfg(target_os = "macos")]
    {
        ctx::device_bytes()
    }
    #[cfg(not(target_os = "macos"))]
    (0, 0)
}

/// `(free, total)` bytes for the device, when a device is usable.
///
/// `total` is `MTLDevice.recommendedMaxWorkingSetSize`: Metal's estimate of
/// the memory the GPU can use without hurting performance. `free` is `total`
/// minus `MTLDevice.currentAllocatedSize` (everything this process has
/// allocated on the device: buffers, the library, pipelines, driver
/// structures), saturating at zero.
///
/// On unified memory these are neither physical figures nor reservations:
/// the GPU allocates from the same system memory as the CPU, other processes
/// and this process's own heap draw on it too, and nothing is set aside. A
/// `free` above zero does not promise that an allocation succeeds, and an
/// allocation beyond `total` can succeed (with paging). Use it as a budget
/// hint, not as a guarantee.
pub fn mem_info() -> Option<(usize, usize)> {
    if !device() {
        return None;
    }
    #[cfg(target_os = "macos")]
    {
        ctx::mem_info()
    }
    #[cfg(not(target_os = "macos"))]
    None
}
