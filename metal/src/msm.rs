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

//! The G1 MSM on the device: sppark's Pippenger MSM ported to Metal.
//!
//! The host side follows sppark `msm/pippenger.cuh` (`msm_t`: the window
//! size, the batches, `invoke`, `collect`, `integrate_row`; Copyright
//! Supranational LLC, Apache-2.0, <https://github.com/supranational/sppark>,
//! commit 08033103c14ff3c742e710b9463dde42acc0f058, vendored in
//! `gpu/third_party/sppark`). The kernels are in `shaders/msm.metal`, the
//! curve formulas in `shaders/include/g1.h`.
//!
//! [`crate::msm_g1`] admits a call (equal non-zero lengths up to
//! `2^MAX_MSM_LOG`, at least `2^msm_table` of [`crate::thresholds`], a usable
//! device, a path from [`lookup`], the effective size, every scalar reduced)
//! before it calls [`msm_g1`], which ends with exactly one of
//! `account::served` or `account::decline`.
//!
//! # Which MSMs take which path
//!
//! The path is decided before any scalar is read, on public facts only.
//! [`lookup`] finds the registered table by address and length, compares
//! every point of the caller's slice with it (dropping it when one differs)
//! and applies the table's G1 check. A table that serves the bases gives the
//! table path. Without one, the default thresholds (`msm_one_shot` is `None`)
//! decline `unregistered-bases` at once, with no scan of the scalars and no
//! G1 check of the bases, and the stock CPU MSM runs the call. The one-shot
//! path is taken only with an explicit threshold ([`crate::set_min_k`],
//! `MIDNIGHT_METAL_MIN_K`); it checks every base for G1 at every call ("Bases
//! outside G1"), a check that alone costs more than the whole stock CPU MSM
//! (Apple M4 Max: 187 ms for `2^16` points and 593 ms for `2^18`, about 2.1 s
//! for `2^20`, against about 25 ms and 78 ms for the CPU MSM at `2^16` and
//! `2^18`), which is exact for any point on the curve without it.
//!
//! # Effective size
//!
//! The size compared with the thresholds is the MSM's count of non-zero
//! scalars (1 when all are zero), not its length: a zero scalar adds nothing,
//! and the masked device MSM costs the same for 2 non-zero scalars out of
//! `2^20` as for a dense one (about 108 ms on Apple M4 Max), while the stock
//! CPU MSM, whose time follows the sparsity, takes about a millisecond.
//! [`crate::msm_g1`] checks it ([`effective_size_reaches`]) after the length
//! admission and the choice of the path and before any device work, against
//! the need of that path ([`effective_need`]): with the default thresholds
//! `2^12 + n/5` on a registered table, the measured crossover of the stock
//! CPU MSM on sparse inputs. Below it the call declines `below-min-k`, the
//! reason a short dense MSM gives. The count itself is not kept, logged,
//! counted or returned: the decline is its only observable. An explicit
//! threshold replaces the need entirely (`2^min_k`, no fraction of `n`), so
//! `set_min_k(0)` sends every MSM, the all-zero one included, to the device.
//!
//! # Algorithm
//!
//! For `n` points, the window width `wbits` and the chunk count come from a
//! table measured on this hardware (`plan::MEASURED`), with `nwins = ceil(255
//! / wbits)` windows and `2^(wbits-1)` buckets per window (signed digits).
//! The points are split into `batch` chunks of `stride` points; each chunk is
//! one command buffer:
//!
//! 1. `msm_digits` (sppark `breakdown`): each scalar out of Montgomery form,
//!    made "positive" (`abs`), Booth-encoded into `nwins` signed digits.
//! 2. The sort of each window's point indices by bucket (sppark `sort.cuh`, the
//!    two-level counting sort): `msm_sort_count`, `msm_sort_scan`,
//!    `msm_sort_scatter`, `msm_sort_lower`, giving the sorted indices and the
//!    bucket end offsets (`hist`).
//! 3. `msm_bucket_guard`: the hot-bucket bound
//!    ([`HOT_BUCKET_LIMIT`](pipeline::HOT_BUCKET_LIMIT)).
//! 4. `msm_ones` (sppark `batch_addition`): the bucket of digit 1 in window 0,
//!    summed by the whole grid.
//! 5. `msm_accumulate`: one thread per bucket, mixed additions into XYZZ. Each
//!    threadgroup first orders its buckets by length (longest first), so that
//!    the 32 lanes of a SIMD-group sum buckets of similar length.
//! 6. `msm_integrate`: per window, `2 * threads` partial sums of the weighted
//!    bucket sum.
//!
//! The host then combines each window's partial sums and the windows with
//! the doubling ladder (sppark `integrate_row` and `collect`) with `blst`,
//! while the device runs the next chunk.
//!
//! # Skew mask
//!
//! As in the CUDA crate (`gpu/src/msm.rs`): a non-zero scalar repeated many
//! times would send most points of a window to one bucket, whose thread then
//! serialises them. The MSM is computed as `MSM(s + r) - MSM(r)`, with `r`
//! the CUDA crate's public splitmix64 vector (a function of the index only,
//! see `shaders/include/msm.h`). `MSM(r)` is cached per registered table and
//! length; the one-shot path computes it at every call. The result is the
//! same group element. Besides the effective size (above) and the hot-bucket
//! bound, no statistic of the scalars is computed, logged or branched on.
//!
//! # The CPU share
//!
//! While the device runs, the CPU computes part of the MSM: the device takes
//! the points `[0, m)` (masked, with `MSM(r)` cached per table and share
//! length) and the stock `msm_pippenger` of midnight-curves takes `[m, n)`
//! with the plain scalars on another thread at the same time; the two sums
//! are added on the host. `m` is a fixed function of `n` and of the path
//! (`plan::MEASURED_SHARE`); scalar values never move it. The CPU share runs
//! on its own rayon pool, never on the caller's: the prover calls MSMs from
//! rayon workers, which can all be blocked on the device lock while the
//! holder waits for its CPU share, so a share queued on their pool could
//! never start. It reads the caller's slices in place. The environment
//! variable `MIDNIGHT_METAL_MSM_CPU_SHARE=off` gives `m = n` (device only).
//! The device time accounted for the call is the device share's GPU time. A
//! failure of either share declines the whole call.
//!
//! # Registered base tables
//!
//! [`register_srs`] keeps a host copy of an SRS base slice under an
//! [`SrsHandle`]. The first MSM on that slice, or on a prefix of it (same
//! start address, not longer), copies the points into a shared Metal buffer
//! (the host copy is then released: on unified memory the buffer is the
//! host-visible copy) that later MSMs use without a transfer. Before each use
//! every point of the caller's slice is compared with the table, on several
//! threads, before any scalar is read and before any device work. A slice
//! that fails the comparison drops its table, with its cached `MSM(r)`, and
//! the call declines `unregistered-bases` (the stock CPU MSM runs on the
//! caller's current points), or with an explicit threshold takes the
//! one-shot path with them. The table holds public points only; it lives
//! until the handle is dropped ([`SrsHandle::device_bytes`] says how many
//! bytes it holds on the device).
//!
//! A table serves the registered slice and its prefixes, by address: the
//! caller registers the very slice the prover passes. The KZG prover of
//! `midnight-proofs` commits from four SRS vectors of `ParamsKZG`, each used
//! by prefix: `bases::<Coeff>()` (the quotient limbs, the multi-open),
//! `bases::<LagrangeCoeff>()` (committed instances, and the fixed columns at
//! key generation), `bases::<LagrangeDeltaCoeff>()` (the advice columns and
//! LogUp multiplicities) and `bases::<LagrangeDoubleDeltaCoeff>()`
//! (permutation products, LogUp helpers and aggregators, trash); a
//! registrant registers all four. Every other MSM base slice of the prover
//! (the residue bases of an interleaved commitment, a copy) has no table.
//!
//! The registry lock is held only to look a table up and pin it (an `Arc`);
//! the comparison, the upload and the MSM run on the pinned table. A table
//! dropped or replaced while an MSM runs on it stays alive until that MSM
//! ends: its points never change after registration, and the comparison has
//! shown that they are the caller's, so the value is exact.
//!
//! Lock order: the registry, then the device state (`ctx::lock`), then a
//! table's points or its mask cache. The registry is never taken while a
//! table's lock is held.
//!
//! # Bases outside G1
//!
//! The `abs` step of `msm_digits` (a digit set of `r - s` with the signs
//! flipped) and the skew mask (`s + r` reduced modulo the group order) give
//! `sum s_i P_i` only when `q P_i = O` for every base (`q` the group order).
//! The stock CPU MSM makes no such assumption. So every base the device uses
//! is checked to be a point of G1: on the curve (blst
//! `blst_p1_affine_on_curve`) and in the prime-order subgroup
//! (`G1Affine::is_torsion_free` of midnight-curves, blst
//! `blst_p1_affine_in_g1`); the point at infinity (all zero, the crate's
//! encoding) is in G1. [`register_srs`] checks every point once and keeps
//! the first index that fails: an MSM on a prefix that reaches it declines
//! (the stock CPU code then runs, exact for any input, a point off the curve
//! included); shorter prefixes are served. The one-shot path checks every
//! base at every call before any device work and declines at the first
//! failure; the check is never skipped or sampled. Both checks run on several
//! threads; the decline reason is [`BASES_REASON`].
//!
//! # Values derived from the scalars
//!
//! Every chunk's device buffers are per-request scratch (zeroed by
//! [`crate::end_request`] and put back into the pool of zeroed buffers, or
//! released when one fails the erasure check; outside a request they are
//! zeroed at the end of the hook call, before it returns). Every host buffer
//! holding values derived from the scalars (the copies of the bucket sums,
//! the window sums, the partial sums, the CPU share's sum) is a guard
//! ([`secret::Secret`], [`secret::SecretVec`]) that zeroes it with volatile
//! writes when dropped, on every exit, unwinding included; and every thread
//! that ran the host collection's or the CPU share's `blst` arithmetic, the
//! caller's included, zeroes 64 KiB of its stack below its frame afterwards
//! ([`secret::scrub_stack`], the counterpart of the CUDA patch's
//! `scrub_stack` in sppark `collect`). A panic anywhere under [`msm_g1`] is
//! a decline.
//!
//! # Where the port departs from sppark
//!
//! - Grid-wide barriers (`cooperative_groups::this_grid().sync()` in the sort,
//!   `accumulate` and `batch_addition`) do not exist in Metal: each is a
//!   dispatch boundary in the same command buffer, so the sort is four kernels.
//!   The per-group counters of the upper sort have their own buffer instead of
//!   the spare entries of the histogram row.
//! - Warp shuffles become Metal SIMD-group functions (SIMD-groups of 32 on
//!   Apple GPUs): the prefix sums of the sort use `simd_prefix_*_sum`, the
//!   `batch_addition` reduction `simd_shuffle_down`. The SIMD-group width of
//!   every MSM pipeline is checked to be 32 before any device work; another
//!   width declines.
//! - `accumulate` hands out buckets through a global atomic counter to
//!   persistent warps; here each bucket is one grid thread and the hardware
//!   schedules them.
//! - sppark sorts two windows per cooperative launch; here a pass sorts as many
//!   windows as the temporary buffer holds (all of them up to `2^18`-point
//!   chunks).
//! - The upper-sort grid size is fixed at `SORT_GROUPS` groups per window
//!   (sppark: a third of the SM count, a power of two: 8 for a 40-core part).
//! - sppark's register-saving `uadd` forms are not used (the plain formulas
//!   give the same points).
//! - The host sums are Jacobian (`blst_p1`), each XYZZ record converted as `(X
//!   ZZ, Y ZZZ, ZZ)`; sppark's host code adds in XYZZ and Jacobian.
//! - The plan (window width and chunk count) is a table measured on Apple M4
//!   Max instead of sppark's formula (`lg2(1.5 n) - 8` clamped to `[10, 18]`,
//!   and `2^(lg2(1.5 n) - wbits) / 64` chunks), which gives too few bucket
//!   threads to fill this GPU. It follows the MSM length on both paths
//!   (sppark's cached context takes it from the table length).
//! - A chunk holds at most `CHUNK_ADDS / nwins` points, so that one command
//!   buffer stays near or below 0.5 s of GPU time.
//! - A chunk whose sort gives a bucket of more than
//!   [`HOT_BUCKET_LIMIT`](pipeline::HOT_BUCKET_LIMIT) points declines before
//!   its accumulation.
//! - The mask is added on the device in `msm_digits` (each element computed
//!   from its index); the CUDA crate builds `r` and `s + r` on the host.
//! - The host's thread starts (`std::thread::Builder::spawn_scoped`) fall back
//!   to the calling thread when a thread cannot be made.

use std::sync::Arc;

use blst::{blst_p1, blst_p1_affine};

use crate::{
    G1AffineLimbs, G1JacobianLimbs, Scalar,
    account::{self, Class, Reason},
    ctx::{self, Error},
};

/// Records into the report of [`msm_g1_stages`] (feature `test-support`);
/// in other builds the trace is always `None` and this is empty.
macro_rules! trace {
    ($trace:expr, |$st:ident| $body:expr) => {
        if let Some($st) = $trace.as_deref_mut() {
            #[cfg(any(test, feature = "test-support"))]
            $body;
            #[cfg(not(any(test, feature = "test-support")))]
            match *$st {}
        }
    };
}

mod checks;
mod collect;
mod effective;
mod par;
mod pipeline;
mod plan;
mod registry;
mod secret;
mod split;

pub(crate) use effective::{effective_need, effective_size_reaches};
#[cfg(any(test, feature = "test-support"))]
pub use registry::one_shot_g1_checks;
use registry::{ONE_SHOT, Store, TABLES_SERVED, Table};
pub use registry::{SrsHandle, msm_counts, register_srs, srs_handles, srs_stale};
use secret::{Secret, scrub_stack};
use split::{Points, split};

/// An XYZZ record of the device (`g1::Xyzz`): `X, Y, ZZZ, ZZ`, six limbs
/// each.
type Xyzz = [u64; 24];

// The host types the code reinterprets: the scalars as `Fq` (the CPU share),
// the points as `G1Affine` and `blst_p1_affine` (the G1 check, the CPU
// share), `G1Projective` as `blst_p1` (the CPU share's sum).
const _: () = {
    use std::mem::{align_of, size_of};
    assert!(size_of::<midnight_curves::Fq>() == size_of::<Scalar>());
    assert!(align_of::<midnight_curves::Fq>() == align_of::<Scalar>());
    assert!(size_of::<midnight_curves::G1Affine>() == size_of::<G1AffineLimbs>());
    assert!(align_of::<midnight_curves::G1Affine>() == align_of::<G1AffineLimbs>());
    assert!(size_of::<blst_p1_affine>() == size_of::<G1AffineLimbs>());
    assert!(align_of::<blst_p1_affine>() <= align_of::<G1AffineLimbs>());
    assert!(size_of::<midnight_curves::G1Projective>() == size_of::<blst_p1>());
};

/// The decline reason of an MSM with a base that is not a point of G1:
/// `bases-not-in-subgroup`.
pub(crate) const BASES_REASON: Reason = Reason::BasesNotInSubgroup;

// ---------------------------------------------------------------------------
// Admission.

/// A registered table that serves an MSM, pinned ([`lookup`]): every point
/// of the caller's slice was compared with it, and no point of the prefix is
/// outside G1.
pub(crate) struct Pinned(Arc<Table>);

/// What [`lookup`] found for an MSM's bases.
pub(crate) enum Lookup {
    /// A registered table serves them.
    Table(Pinned),
    /// A registered table holds them, but a point of the prefix is not in G1
    /// ([`BASES_REASON`]).
    NotInG1,
    /// No registered table serves them: none was registered for the slice,
    /// or the registered one no longer holds the caller's points (it was
    /// dropped as stale now, or before).
    Unregistered,
}

/// The registered table that serves an MSM of `n` scalars on `bases`: the
/// registry lookup by address and length, then the comparison of every
/// point (a table that does not match is dropped), then the table's G1 check
/// of the prefix. Public facts only: no scalar is read. A part of the
/// comparison that panicked counts as a difference.
pub(crate) fn lookup(bases: &[G1AffineLimbs], n: usize) -> Lookup {
    match registry::serving_table(bases, n) {
        Some(t) if n > t.first_bad => Lookup::NotInG1,
        Some(t) => Lookup::Table(Pinned(t)),
        None => Lookup::Unregistered,
    }
}

// ---------------------------------------------------------------------------
// Dispatch.

/// `sum scalars[i] * bases[i]`, or `None` (declined), on the pinned table
/// `table` ([`lookup`]), or on the one-shot path without one (the caller has
/// admitted it: an explicit threshold), which first checks every base.
///
/// A base that is not a point of G1 declines [`BASES_REASON`]. Nothing
/// panics out of this function: a panic under it is a `panic` decline.
/// Every host buffer of values derived from the scalars is zeroed before it
/// returns, and so is `secret::SCRUB_BYTES` of the calling thread's stack
/// below this frame.
pub(crate) fn msm_g1(
    scalars: &[Scalar],
    bases: &[G1AffineLimbs],
    table: Option<Pinned>,
) -> Option<G1JacobianLimbs> {
    if scalars.is_empty() || scalars.len() != bases.len() {
        account::decline(Class::Msm, Reason::Shape);
        return None;
    }
    let ran = par::catch(|| dispatch(scalars, bases, table.map(|p| p.0), None));
    scrub_stack();
    match ran {
        Some(Ok((r, ns))) => {
            account::served(Class::Msm, ns);
            // the result leaves; the guard `r` zeroes its own copy
            Some(*r)
        }
        Some(Err(Fail::Bases)) => {
            account::decline(Class::Msm, BASES_REASON);
            None
        }
        Some(Err(Fail::Ctx(e))) => {
            account::decline(Class::Msm, Reason::of(&e));
            None
        }
        None => {
            account::decline(Class::Msm, Reason::Panic);
            None
        }
    }
}

/// Why [`dispatch`] did not serve a call.
enum Fail {
    /// A base the device would use is not a point of G1.
    Bases,
    /// The device context failed.
    Ctx(Error),
}

impl From<Error> for Fail {
    fn from(e: Error) -> Fail {
        Fail::Ctx(e)
    }
}

/// The result of a served MSM (a guard) and its GPU time in nanoseconds.
type Served = (Secret<G1JacobianLimbs>, u64);

/// The table path on the pinned `table`, or the one-shot path without one.
fn dispatch(
    scalars: &[Scalar],
    bases: &[G1AffineLimbs],
    table: Option<Arc<Table>>,
    trace: Trace<'_>,
) -> Result<Served, Fail> {
    match table {
        Some(t) => table_path(scalars, bases, t, trace),
        None => one_shot_path(scalars, bases, trace),
    }
}

/// The MSM on the pinned table `t`: every point of `bases` equals the
/// table's, and the table's G1 check covers them ([`lookup`]). The first MSM
/// on a table uploads its points.
fn table_path(
    scalars: &[Scalar],
    bases: &[G1AffineLimbs],
    t: Arc<Table>,
    mut trace: Trace<'_>,
) -> Result<Served, Fail> {
    debug_assert!(scalars.len() <= t.first_bad);
    let mut s = ctx::lock()?;
    t.upload(s.ctx())?;
    TABLES_SERVED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    trace!(trace, |st| st.table = true);
    let store = par::read(&t.store);
    let Store::Device(points) = &*store else {
        return Err(Error::Device("msm: the table was not uploaded".into()).into());
    };
    let r = split(
        &mut s,
        Points::Table(points),
        scalars,
        bases,
        Some(&t.masks),
        trace,
    )?;
    drop(store);
    // the session ends before the table may be freed (if its handle was
    // dropped meanwhile, this is the last reference)
    drop(s);
    drop(t);
    Ok(r)
}

/// The MSM on the caller's points, copied for the call, after the G1 check
/// of every base.
fn one_shot_path(
    scalars: &[Scalar],
    bases: &[G1AffineLimbs],
    trace: Trace<'_>,
) -> Result<Served, Fail> {
    #[cfg(any(test, feature = "test-support"))]
    registry::G1_CHECKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if !checks::all_in_g1(bases) {
        return Err(Fail::Bases);
    }
    ONE_SHOT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut s = ctx::lock()?;
    Ok(split(
        &mut s,
        Points::Host(bases),
        scalars,
        bases,
        None,
        trace,
    )?)
}

// ---------------------------------------------------------------------------
// The test entry.

/// Where [`msm_g1_stages`] collects its report.
#[cfg(any(test, feature = "test-support"))]
type Trace<'a> = Option<&'a mut MsmStages>;
/// Always `None`: only the test entry reports.
#[cfg(not(any(test, feature = "test-support")))]
type Trace<'a> = Option<&'a mut std::convert::Infallible>;

/// The path and the plan of one MSM ([`msm_g1_stages`]): public facts only.
#[cfg(any(test, feature = "test-support"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MsmStages {
    /// Window bits of the device share.
    pub wbits: u32,
    /// Chunks (command buffers) of each device MSM pass.
    pub batches: u32,
    /// Device MSM passes: 2 (`MSM(s + r)` and `MSM(r)`), or 1 when the table
    /// had `MSM(r)` cached.
    pub passes: u32,
    /// Whether a registered table served the call.
    pub table: bool,
    /// Points of the device share (`n` when the CPU takes no share).
    pub device_points: u64,
}

/// [`crate::msm_g1`] on the path [`lookup`] gives, with the length and
/// device checks of its admission but not its thresholds, reporting which
/// path and plan the MSM took. Not counted in [`crate::stats`] or
/// [`crate::declines`]. `None`: a shape, scalar or base [`crate::msm_g1`]
/// would decline, or a device failure. Feature `test-support` only; not a
/// stable interface.
#[doc(hidden)]
#[cfg(any(test, feature = "test-support"))]
pub fn msm_g1_stages(
    scalars: &[Scalar],
    bases: &[G1AffineLimbs],
) -> Option<(G1JacobianLimbs, MsmStages)> {
    if scalars.is_empty() || scalars.len() != bases.len() || scalars.len() > 1 << crate::MAX_MSM_LOG
    {
        return None;
    }
    if !crate::device() || !crate::field::all_reduced(scalars) {
        return None;
    }
    let mut st = MsmStages::default();
    let ran = par::catch(|| {
        let table = match lookup(bases, scalars.len()) {
            Lookup::Table(p) => Some(p.0),
            Lookup::NotInG1 => return Err(Fail::Bases),
            Lookup::Unregistered => None,
        };
        dispatch(scalars, bases, table, Some(&mut st))
    });
    scrub_stack();
    let (r, _) = ran?.ok()?;
    Some((*r, st))
}
