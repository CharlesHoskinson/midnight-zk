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

//! The monomorphic entry points of a backend crate.

/// Emits a backend crate's entry points at its root: the hooks, thresholds,
/// counters, registrations, requests and test entry points, each the generic
/// function of this crate at `$backend`. `$env` is the backend's
/// [`Backend::ENV_PREFIX`](crate::Backend::ENV_PREFIX) as a literal, for the
/// docs. The backend crate defines `present`.
#[macro_export]
macro_rules! export_backend {
    ($backend:ident, $env:literal) => {
        /// MSM path and plan of one call (feature `test-support`).
        #[cfg(any(test, feature = "test-support"))]
        #[doc(hidden)]
        pub use $crate::MsmStages;
        /// The placement counts of the last quotient served (feature
        /// `test-support`).
        #[cfg(any(test, feature = "test-support"))]
        #[doc(hidden)]
        pub use $crate::QuotientStages;
        /// The quotient job layout, for integration tests.
        #[doc(hidden)]
        pub use $crate::quotient::abi as quotient_abi;
        pub use $crate::{
            ClassStats, G1AffineLimbs, G1JacobianLimbs, MAX_LOG_N, MAX_MSM_LOG, Scalar, Stats,
            Thresholds, WipeStats, field,
        };

        /// Whether device calls can run: [`present`], and the context is made
        /// (the first call makes it) and usable.
        pub fn device() -> bool {
            $crate::device::<$backend>()
        }

        /// `(free, total)` device bytes: `total` is the device's memory budget,
        /// `free` is `total` minus the memory in use plus the idle pool buffers,
        /// which are released on demand. A budget hint, not a reservation.
        pub fn mem_info() -> Option<(usize, usize)> {
            $crate::mem_info::<$backend>()
        }

        /// The effective threshold of each class.
        pub fn thresholds() -> Thresholds {
            $crate::thresholds::<$backend>()
        }

        /// The explicit threshold, else the smallest per-class default.
        pub fn min_k() -> u32 {
            $crate::min_k::<$backend>()
        }

        /// Sets one threshold for every class and opens the one-shot MSM.
        /// Returns `false` and keeps the old value when `k` is above
        /// [`MAX_LOG_N`].
        pub fn set_min_k(k: u32) -> bool {
            $crate::set_min_k::<$backend>(k)
        }

        /// Undoes [`set_min_k`].
        pub fn reset_min_k() {
            $crate::reset_min_k::<$backend>()
        }

        /// The service counters since the last [`reset_stats`].
        pub fn stats() -> Stats {
            $crate::stats::<$backend>()
        }

        /// Every decline since the last [`reset_stats`]: `(class, reason,
        /// count)`, with the reasons of [`midnight_accel::declines`].
        pub fn declines() -> Vec<(&'static str, &'static str, u64)> {
            $crate::declines::<$backend>()
        }

        /// Resets the counters of [`stats`] and [`declines`].
        pub fn reset_stats() {
            $crate::reset_stats::<$backend>()
        }

        /// The length admission of an MSM: `shape` for an empty input, unequal
        /// lengths or a length above `2^max_log`; `below-min-k` when the size
        /// class is below `msm_table`. [`msm_g1`] may still decline on the
        /// bases.
        pub fn msm_admission(
            scalars: usize,
            bases: usize,
            max_log: u32,
        ) -> Result<(), &'static str> {
            $crate::msm_admission::<$backend>(scalars, bases, max_log)
        }

        /// Whether [`msm_g1`] takes an MSM of `len` scalars over `bases`,
        /// judged on the length and the bases only: it reads no scalar, so the
        /// answer does not depend on the witness. It does not check the device.
        pub fn msm_sized(len: usize, bases: &[G1AffineLimbs]) -> bool {
            $crate::msm_sized::<$backend>(len, bases)
        }

        /// G1 MSM `sum scalars[i] * bases[i]`: the hook `GpuHooks::msm_g1`.
        /// `None`: declined.
        ///
        /// Every check on public facts (lengths, device, registered table,
        /// one-shot admission) runs before any scalar is read.
        pub fn msm_g1(scalars: &[Scalar], bases: &[G1AffineLimbs]) -> Option<G1JacobianLimbs> {
            $crate::msm_g1::<$backend>(scalars, bases)
        }

        /// [`msm_g1`] and [`msm_sized`] of the same call from one registry
        /// lookup. `sized` is decided before any scalar is read and stands even
        /// if the MSM declines or panics.
        pub fn msm_g1_sized(
            scalars: &[Scalar],
            bases: &[G1AffineLimbs],
        ) -> (bool, Option<G1JacobianLimbs>) {
            $crate::msm_g1_sized::<$backend>(scalars, bases)
        }

        /// [`msm_g1`] without the thresholds, reporting the path and plan. Not
        /// counted in [`stats`] or [`declines`].
        #[cfg(any(test, feature = "test-support"))]
        #[doc(hidden)]
        pub fn msm_g1_stages(
            scalars: &[Scalar],
            bases: &[G1AffineLimbs],
        ) -> Option<(G1JacobianLimbs, MsmStages)> {
            $crate::msm_g1_stages::<$backend>(scalars, bases)
        }

        /// Per-call G1 checks started on the one-shot path since process start.
        #[cfg(any(test, feature = "test-support"))]
        #[doc(hidden)]
        pub fn one_shot_g1_checks() -> u64 {
            $crate::one_shot_g1_checks::<$backend>()
        }

        /// The registration of an SRS base slice. While it lives, the
        /// registrant keeps the slice alive and unchanged. Dropping it frees the
        /// table.
        pub type SrsHandle = $crate::SrsHandle<$backend>;

        /// Registers a host copy of an SRS base slice; the first MSM on it
        /// uploads it. Registering the same slice again replaces the table.
        ///
        /// Every point is checked to be in G1 once here. A table with a point
        /// outside G1 is still registered, and MSMs on a prefix that includes it
        /// decline.
        pub fn register_srs(bases: &[G1AffineLimbs]) -> SrsHandle {
            $crate::register_srs::<$backend>(bases)
        }

        /// Live registered tables.
        pub fn srs_handles() -> usize {
            $crate::srs_handles::<$backend>()
        }

        /// Tables dropped at an MSM because the caller's slice no longer
        /// matched.
        pub fn srs_stale() -> u64 {
            $crate::srs_stale::<$backend>()
        }

        /// (MSMs served from a registered table, one-shot MSMs, table uploads)
        /// since process start.
        pub fn msm_counts() -> (u64, u64, u64) {
            $crate::msm_counts::<$backend>()
        }

        /// The FFT of `a` (length `2^log_n`) at `omega`, natural order, in
        /// place: the hook `GpuHooks::fft`. `false`: declined, `a` unchanged.
        pub fn fft(a: &mut [Scalar], omega: Scalar, log_n: u32) -> bool {
            $crate::fft::<$backend>(a, omega, log_n)
        }

        /// The FFT of `a` at `omega_inv`, then each element times `divisor`, in
        /// place: the hook `GpuHooks::ifft`. `false`: declined, `a` unchanged.
        pub fn ifft(a: &mut [Scalar], omega_inv: Scalar, log_n: u32, divisor: Scalar) -> bool {
            $crate::ifft::<$backend>(a, omega_inv, log_n, divisor)
        }

        /// The quotient `h(X)` of a job
        /// (`midnight_proofs::gpu::abi::QuotientJobAbi`): the hook
        /// `GpuHooks::quotient`, writing the first `out.len()` coefficients.
        /// `false`: declined, `out` unchanged.
        ///
        /// # Safety
        /// `job` is null or points to a valid `QuotientJobAbi` whose pointers
        /// live for the call.
        pub unsafe fn quotient(job: *const ::std::ffi::c_void, out: &mut [Scalar]) -> bool {
            // SAFETY: the caller's contract, passed on.
            unsafe { $crate::quotient::<$backend>(job, out) }
        }

        /// The registration of proving-key polynomials. While it lives the
        /// registrant keeps them alive and, within a request, unchanged.
        /// Dropping it frees the device copies.
        pub type KeyHandle = $crate::KeyHandle<$backend>;

        /// Registers proving-key polynomials by their digests. A polynomial
        /// registered again moves to the new handle. Each length has its own
        /// device copy, and a quotient serves from the copy of its extended
        /// length only.
        pub fn register_key(polys: &[&[Scalar]]) -> KeyHandle {
            $crate::register_key::<$backend>(polys)
        }

        /// Frees every device copy of proving-key polynomials (registrations
        /// stay). Returns the bytes freed; a copy lent to a running quotient is
        /// freed when returned and not counted.
        pub fn evict_key_copies() -> u64 {
            $crate::evict_key_copies::<$backend>()
        }

        #[doc = concat!(
                            "Whether the device copies of registered proving-key polynomials ",
                            "stay after a request (default `false`, or `", $env, "_KEEP_KEYS=1`)."
                        )]
        pub fn keep_resident() -> bool {
            $crate::keep_resident::<$backend>()
        }

        /// Sets [`keep_resident`].
        pub fn set_keep_resident(on: bool) {
            $crate::set_keep_resident::<$backend>(on)
        }

        /// Registered polynomials now.
        pub fn keys_registered() -> usize {
            $crate::keys_registered::<$backend>()
        }

        /// Registered polynomials dropped at a quotient because their content
        /// changed, since process start.
        pub fn key_stale() -> u64 {
            $crate::key_stale::<$backend>()
        }

        /// An upper bound of the device bytes one quotient job allocates. The
        /// arguments are the fields of `midnight_proofs::gpu::QuotientShape`.
        pub fn need_bytes(ext_k: u32, proof_polys: usize, key_polys: usize, nslots: u32) -> usize {
            $crate::need_bytes::<$backend>(ext_k, proof_polys, key_polys, nslots)
        }

        /// (quotients computed on the device, quotient jobs declined) since
        /// process start.
        pub fn quotient_counts() -> (u64, u64) {
            $crate::quotient_counts::<$backend>()
        }

        #[doc = concat!(
                            "The threads of one numerator dispatch, which is also the scratch ",
                            "slot stride: `", $env, "_QUOTIENT_THREADS`, else 2^16; a multiple ",
                            "of 256 in `[256, 2^24]`."
                        )]
        pub fn quotient_threads() -> u32 {
            $crate::quotient_threads::<$backend>()
        }

        /// NTT building blocks (not a stable interface).
        #[cfg(any(test, feature = "test-support"))]
        #[doc(hidden)]
        pub mod ntt_test_support {
            pub use $crate::ntt::test_support::{passes, root};

            use super::{Scalar, $backend};

            /// `coeff_to_extended` on the device.
            pub fn coeff_to_extended(
                coeffs: &[Scalar],
                k: u32,
                ext_k: u32,
                zeta: [Scalar; 2],
            ) -> Result<Vec<Scalar>, String> {
                $crate::ntt::test_support::coeff_to_extended::<$backend>(coeffs, k, ext_k, zeta)
            }

            /// `extended_to_coeff` on the device.
            pub fn extended_to_coeff(
                values: &[Scalar],
                ext_k: u32,
                divisor: Scalar,
                zeta_out: [Scalar; 2],
            ) -> Result<Vec<Scalar>, String> {
                $crate::ntt::test_support::extended_to_coeff::<$backend>(
                    values, ext_k, divisor, zeta_out,
                )
            }

            /// The NTT of a copy of `values`.
            pub fn ntt(
                values: &[Scalar],
                log_n: u32,
                inverse: bool,
                scale: Option<Scalar>,
            ) -> Result<Vec<Scalar>, String> {
                $crate::ntt::test_support::ntt::<$backend>(values, log_n, inverse, scale)
            }

            /// The elementwise blocks on a copy of `values`.
            pub fn elementwise(
                values: &[Scalar],
                zeta: [Scalar; 2],
                c: Scalar,
                table: &[Scalar],
            ) -> Result<Vec<Scalar>, String> {
                $crate::ntt::test_support::elementwise::<$backend>(values, zeta, c, table)
            }
        }

        /// Quotient test entry points (not a stable interface).
        #[cfg(any(test, feature = "test-support"))]
        #[doc(hidden)]
        pub mod quotient_test_support {
            use ::std::ffi::c_void;
            use $crate::quotient::test_support as t;

            use super::{QuotientStages, Scalar, $backend};

            /// The exact memory need of `job` now.
            ///
            /// # Safety
            /// As [`super::quotient`].
            pub unsafe fn need_bytes_job(job: *const c_void) -> Result<usize, &'static str> {
                // SAFETY: the caller's contract.
                unsafe { t::need_bytes_job::<$backend>(job) }
            }

            /// The placement counts of the last quotient served.
            pub fn last_stages() -> Option<QuotientStages> {
                t::last_stages::<$backend>()
            }

            /// A device-memory limit for the admission.
            pub fn set_memory_limit(limit: Option<u64>) {
                t::set_memory_limit::<$backend>(limit)
            }

            /// The numerator's target GPU time per command buffer.
            pub fn set_target_nanos(ns: Option<u64>) {
                t::set_target_nanos::<$backend>(ns)
            }

            /// The dispatch threads of the numerator.
            pub fn set_threads(threads: Option<u32>) {
                t::set_threads::<$backend>(threads)
            }

            /// The numerator of a job, read back.
            ///
            /// # Safety
            /// As [`super::quotient`].
            pub unsafe fn numerator(job: *const c_void) -> Result<Vec<Scalar>, &'static str> {
                // SAFETY: the caller's contract.
                unsafe { t::numerator::<$backend>(job) }
            }
        }

        /// Makes the hook `class` panic inside its backend (feature
        /// `test-support`).
        #[cfg(any(test, feature = "test-support"))]
        #[doc(hidden)]
        pub fn set_hook_panic(class: &str, on: bool) {
            $crate::set_hook_panic::<$backend>(class, on)
        }

        /// Hook calls whose backend panicked since process start.
        #[cfg(any(test, feature = "test-support"))]
        #[doc(hidden)]
        pub fn hook_panics() -> u64 {
            $crate::hook_panics::<$backend>()
        }

        /// Starts a request: per-request buffers are kept for reuse until
        /// [`end_request`]. Also resets the peaks of [`device_bytes`] and
        /// [`request_peak_bytes`].
        pub fn begin_request() {
            $crate::begin_request::<$backend>()
        }

        /// Ends a request: waits for device work and background zeroing,
        /// zeroes the per-request buffers and returns them to the pool, then
        /// frees the proving-key copies unless [`keep_resident`]. With `check`,
        /// every zeroed region and every idle pool buffer is read back.
        ///
        /// `Err` when a region read back non-zero or unreadable, or a zeroing
        /// panicked. Such a region is zeroed again and released, never pooled.
        pub fn end_request_report(check: bool) -> Result<WipeStats, WipeStats> {
            $crate::end_request_report::<$backend>(check)
        }

        /// [`end_request_report`] without the read-back; `true` on success.
        pub fn end_request() -> bool {
            $crate::end_request::<$backend>()
        }

        /// Releases the pool of zeroed buffers once background zeroing has
        /// ended. Returns the bytes released.
        pub fn release_pool() -> u64 {
            $crate::release_pool::<$backend>()
        }

        /// Bytes of idle, zeroed buffers in the pool.
        pub fn pool_bytes() -> u64 {
            $crate::pool_bytes::<$backend>()
        }

        /// `(now, peak)` bytes held in this crate's device buffers, idle pool
        /// buffers included. The peak is since process start or the last
        /// [`begin_request`].
        pub fn device_bytes() -> (u64, u64) {
            $crate::device_bytes::<$backend>()
        }

        /// Peak bytes in use since the last [`begin_request`]: buffers handed
        /// out, resident SRS tables and key copies, not idle pool buffers. It
        /// starts from the bytes in use at [`begin_request`], so it includes
        /// what earlier requests left resident.
        pub fn request_peak_bytes() -> u64 {
            $crate::request_peak_bytes::<$backend>()
        }
    };
}

/// Emits a backend's `ctx` module items: the shared context types of
/// [`crate::ctx`] instantiated with `$backend`, and its free functions.
#[macro_export]
macro_rules! export_ctx {
    ($backend:ident, $env:literal) => {
        pub use $crate::ctx::{Error, Pod, ReadBack, bytes_of};

        /// The device context: the backend with the shared buffer management.
        pub type Ctx = $crate::ctx::Ctx<$backend>;
        /// A device buffer, zeroed when dropped.
        pub type Buffer = $crate::ctx::Buffer<$backend>;
        /// Exclusive use of the device: the context and the per-request
        /// buffers.
        pub type Session = $crate::ctx::Session<$backend>;
        /// A per-request buffer checked out by [`take_request_buffer`].
        pub type RequestBuffer = $crate::ctx::RequestBuffer<$backend>;
        /// The command buffer being encoded in [`Ctx::run`].
        pub type Command<'c, 'a> = $crate::ctx::Command<'c, 'a, $backend>;
        /// A buffer argument of a kernel.
        pub type Arg<'a> = $crate::ctx::Arg<'a, $backend>;

        /// The context, made at the first call. A failure is kept: later calls
        /// return it without trying again.
        pub fn context() -> Result<&'static Ctx, Error> {
            $crate::ctx::context::<$backend>()
        }

        /// Takes the device, making the context if needed.
        pub fn lock() -> Result<Session, Error> {
            $crate::ctx::lock::<$backend>()
        }

        /// `(now, peak)` bytes held in this backend's buffers, idle pool
        /// buffers included. The peak is since process start or the last
        /// [`begin_request`].
        pub fn device_bytes() -> (u64, u64) {
            $crate::ctx::device_bytes::<$backend>()
        }

        /// Bytes in buffers in use now: [`device_bytes`] without idle pool
        /// buffers.
        pub fn in_use_bytes() -> u64 {
            $crate::ctx::in_use_bytes::<$backend>()
        }

        /// The peak of [`in_use_bytes`] since the last [`begin_request`], which
        /// resets it to the bytes in use then.
        pub fn request_peak_bytes() -> u64 {
            $crate::ctx::request_peak_bytes::<$backend>()
        }

        /// Opens a request and resets the peaks of [`device_bytes`] and
        /// [`request_peak_bytes`].
        pub fn begin_request() {
            $crate::ctx::begin_request::<$backend>()
        }

        /// Whether a request is open ([`begin_request`] was called and its
        /// [`end_request_report`] has not started).
        pub fn request_open() -> bool {
            $crate::ctx::request_open::<$backend>()
        }

        #[doc = concat!(
                            "The most bytes the pool keeps idle: `", $env, "_POOL_MIB` (read ",
                            "once), else a quarter of the device's memory budget."
                        )]
        pub fn pool_limit() -> u64 {
            $crate::ctx::pool_limit::<$backend>()
        }

        /// Sets [`pool_limit`] in bytes (`None`: the default again) and
        /// releases idle buffers above it.
        #[cfg(any(test, feature = "test-support"))]
        #[doc(hidden)]
        pub fn set_pool_limit(limit: Option<u64>) {
            $crate::ctx::set_pool_limit::<$backend>(limit)
        }

        /// Bytes of the idle (zeroed) buffers in the pool.
        pub fn pool_bytes() -> u64 {
            $crate::ctx::pool_bytes::<$backend>()
        }

        /// Releases every idle buffer of the pool once background zeroing has
        /// ended. Returns the bytes released.
        pub fn release_pool() -> u64 {
            $crate::ctx::release_pool::<$backend>()
        }

        /// Drops `bufs` (each zeroed, then pooled or released) on a background
        /// thread; [`settle_releases`] waits for it.
        pub fn release_later(bufs: Vec<Buffer>) {
            $crate::ctx::release_later(bufs)
        }

        /// Whether a [`release_later`] call has not completed yet.
        #[cfg(any(test, feature = "test-support"))]
        #[doc(hidden)]
        pub fn releases_pending() -> bool {
            $crate::ctx::releases_pending::<$backend>()
        }

        /// Waits until every [`release_later`] call made before this one has
        /// completed.
        pub fn settle_releases() {
            $crate::ctx::settle_releases::<$backend>()
        }

        /// A per-request buffer under `key` taken without the device
        /// ([`midnight_accel::ctx::take_request_buffer`]).
        pub fn take_request_buffer(
            key: &'static str,
            bytes: usize,
        ) -> Result<RequestBuffer, Error> {
            $crate::ctx::take_request_buffer::<$backend>(key, bytes)
        }

        /// Makes every later [`end_request_report`]`(true)` plant a non-zero
        /// byte in its first region after zeroing, so the read-back fails.
        #[cfg(any(test, feature = "test-support"))]
        #[doc(hidden)]
        pub fn set_plant_nonzero(on: bool) {
            $crate::ctx::set_plant_nonzero::<$backend>(on)
        }

        /// The id of the buffer of the last planted byte.
        #[cfg(any(test, feature = "test-support"))]
        #[doc(hidden)]
        pub fn last_planted() -> u64 {
            $crate::ctx::last_planted::<$backend>()
        }

        /// Makes the next zeroing of the dropped buffer `id` panic once; 0
        /// stops it.
        #[cfg(any(test, feature = "test-support"))]
        #[doc(hidden)]
        pub fn set_zeroing_panic(id: u64) {
            $crate::ctx::set_zeroing_panic::<$backend>(id)
        }

        /// Writes a non-zero byte into an idle pool buffer; returns its id.
        #[cfg(any(test, feature = "test-support"))]
        #[doc(hidden)]
        pub fn plant_in_idle_pool() -> Option<u64> {
            $crate::ctx::plant_in_idle_pool::<$backend>()
        }

        /// The ids of the idle buffers of the pool and the staging pool.
        #[cfg(any(test, feature = "test-support"))]
        #[doc(hidden)]
        pub fn idle_pool_ids() -> Vec<u64> {
            $crate::ctx::idle_pool_ids::<$backend>()
        }

        /// Whether every idle pool buffer reads zero.
        #[cfg(any(test, feature = "test-support"))]
        #[doc(hidden)]
        pub fn idle_pool_reads_zero() -> bool {
            $crate::ctx::idle_pool_reads_zero::<$backend>()
        }

        /// The per-request buffers now: `(key, allocation bytes, reads zero,
        /// id)`. Takes the device.
        #[cfg(any(test, feature = "test-support"))]
        #[doc(hidden)]
        pub fn request_buffers() -> Vec<(&'static str, usize, bool, u64)> {
            $crate::ctx::request_buffers::<$backend>()
        }

        /// Zeroes every per-request buffer after any device work and
        /// background zeroing, and closes the request; with `check`, reads
        /// every region back.
        pub fn end_request_report(check: bool) -> Result<$crate::WipeStats, $crate::WipeStats> {
            $crate::ctx::end_request_report::<$backend>(check)
        }

        /// [`end_request_report`] without the read-back. `true` when done.
        pub fn end_request() -> bool {
            $crate::ctx::end_request::<$backend>()
        }

        /// `(free, total)`: `total` is the device's memory budget, `free` is
        /// that minus the memory in use plus the idle pool buffers, which count
        /// as free because an allocation that would not fit releases them
        /// first.
        pub fn mem_info() -> Option<(usize, usize)> {
            $crate::ctx::mem_info::<$backend>()
        }
    };
}
