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
//! [`crate::msm_g1`] has admitted the call (equal non-zero lengths up to
//! `2^MAX_MSM_LOG`, at least `2^min_k`, a usable device, every scalar
//! reduced) before it calls [`msm_g1`]. That function ends with exactly one
//! of `account::served` or `account::decline`.
//!
//! # Algorithm
//!
//! For `n` points, the window width `wbits` and the chunk count come from a
//! table measured on this hardware (`MEASURED`, see `plan`), with `nwins =
//! ceil(255 / wbits)` windows and `2^(wbits-1)` buckets per window (signed
//! digits). The points are split into `batch` chunks of `stride` points;
//! each chunk is one command buffer:
//!
//! 1. `msm_digits` (sppark `breakdown`): each scalar out of Montgomery form,
//!    made "positive" (`abs`), Booth-encoded into `nwins` signed digits.
//! 2. The sort of each window's point indices by bucket (sppark `sort.cuh`,
//!    the two-level counting sort): `msm_sort_count`, `msm_sort_scan`,
//!    `msm_sort_scatter`, `msm_sort_lower`, giving the sorted indices and the
//!    bucket end offsets (`hist`).
//! 3. `msm_ones` (sppark `batch_addition`): the bucket of digit 1 in window
//!    0, summed by the whole grid.
//! 4. `msm_accumulate`: one thread per bucket, mixed additions into XYZZ.
//! 5. `msm_integrate`: per window, `2 * threads` partial sums of the
//!    weighted bucket sum.
//!
//! The host then combines each window's partial sums and the windows with
//! the doubling ladder (sppark `integrate_row` and `collect`) with `blst`,
//! while the device runs the next chunk.
//!
//! # Skew mask
//!
//! As in the CUDA crate (`gpu/src/msm.rs`): a non-zero scalar repeated many
//! times would send most points of a window to one bucket, whose thread
//! then serialises them. The MSM is computed as `MSM(s + r) - MSM(r)`, with
//! `r` the CUDA crate's public splitmix64 vector (a function of the index
//! only, see `shaders/include/msm.h`). `MSM(r)` is cached per registered
//! table and length; the one-shot path computes it at every call. The result
//! is the same group element. No statistic of the scalars is computed,
//! logged or branched on (except by [`msm_g1_stages`], a measurement entry).
//!
//! # Registered base tables
//!
//! [`register_srs`] keeps a host copy of an SRS base slice under an
//! [`SrsHandle`]. The first MSM on that slice, or on a prefix of it (same
//! start address, not longer), copies the points into a shared Metal buffer
//! (the host copy is then released: on unified memory the buffer is the
//! host-visible copy) that later MSMs use without a transfer. Before each
//! use the caller's slice is compared with the table: every point in a
//! debug build, eight sampled points in a release build. A slice that fails
//! the check drops its table and takes the one-shot path. The table holds
//! public points only; it lives until the handle is dropped.
//!
//! # Where the port departs from sppark
//!
//! - Grid-wide barriers (`cooperative_groups::this_grid().sync()` in the
//!   sort, `accumulate` and `batch_addition`) do not exist in Metal: each is
//!   a dispatch boundary in the same command buffer, so the sort is four
//!   kernels. The per-group counters of the upper sort have their own
//!   buffer instead of the spare entries of the histogram row.
//! - Warp shuffles become Metal SIMD-group functions (SIMD-groups of 32 on
//!   Apple GPUs): the prefix sums of the sort use `simd_prefix_*_sum`, the
//!   `batch_addition` reduction `simd_shuffle_down`.
//! - `accumulate` hands out buckets through a global atomic counter to
//!   persistent warps; here each bucket is one grid thread and the hardware
//!   schedules them.
//! - sppark sorts two windows per cooperative launch; here a pass sorts as
//!   many windows as the temporary buffer holds (all of them up to
//!   `2^18`-point chunks).
//! - The upper-sort grid size is fixed at `SORT_GROUPS` groups per window
//!   (sppark: a third of the SM count, a power of two: 8 for a 40-core part).
//! - sppark's register-saving `uadd` forms are not used (the plain formulas
//!   give the same points).
//! - The host sums are Jacobian (`blst_p1`), each XYZZ record converted as
//!   `(X ZZ, Y ZZZ, ZZ)`; sppark's host code adds in XYZZ and Jacobian.
//! - The plan (window width and chunk count) is a table measured on Apple
//!   M4 Max (`MEASURED`) instead of sppark's formula (`lg2(1.5 n) - 8`
//!   clamped to `[10, 18]`, and `2^(lg2(1.5 n) - wbits) / 64` chunks), which
//!   gives too few bucket threads to fill this GPU. It follows the MSM
//!   length on both paths (sppark's cached context takes it from the table
//!   length).
//! - A chunk holds at most `CHUNK_ADDS / nwins` points, so that one command
//!   buffer stays near or below 0.5 s of GPU time (sppark has no such
//!   bound).
//! - The mask is added on the device in `msm_digits` (each element computed
//!   from its index); the CUDA crate builds `r` and `s + r` on the host.
//! - Every chunk's buffers are per-request scratch (zeroed and released by
//!   [`crate::end_request`]); the copies of the bucket sums on the host are
//!   zeroed after use.

use std::{
    collections::HashMap,
    sync::{
        Mutex, MutexGuard,
        atomic::{AtomicU64, Ordering::Relaxed},
    },
    time::Instant,
};

use blst::{blst_fp, blst_fp_mul, blst_p1, blst_p1_add_or_double, blst_p1_cneg, blst_p1_double};
use objc2_metal::MTLComputePipelineState;

use crate::{
    G1AffineLimbs, G1JacobianLimbs, Scalar,
    account::{self, Class, Reason},
    ctx::{self, Arg, Buffer, Command, Ctx, Error, Session},
};

/// Bits of a scalar (sppark `scalar_t::bit_length()`).
const NBITS: u32 = 255;
/// Upper-sort groups per window (sppark: `sm_count / 3`, a power of two).
const SORT_GROUPS: u32 = 8;
/// Threads of a sort group (sppark `SORT_BLOCKDIM`).
const SORT_BLOCK: usize = 1024;
/// Groups and threads of `msm_ones` (sppark: `sm_count` blocks of
/// `BATCH_ADD_BLOCK_SIZE`).
const ONES_GROUPS: usize = 32;
const ONES_BLOCK: usize = 256;
/// Threads per window of `msm_integrate` (sppark `MSM_NTHREADS`).
const MSM_NTHREADS: usize = 256;
/// The most window digits (bucket additions) of one chunk, so that a
/// command buffer stays near or below 0.5 s of GPU time: measured on Apple
/// M4 Max at 5 to 10 ns per mixed addition, sort and integrate included.
const CHUNK_ADDS: usize = 1 << 25;
/// Bytes of the temporary sort buffer one pass may use.
const SORT_TEMP_BYTES: usize = 64 << 20;
/// The largest low-digit width the lower sort holds in threadgroup memory
/// (`MAX_LOW` in msm.metal).
const MAX_LOW_BITS: u32 = 12;
/// The largest top-digit width of the upper sort (`MAX_TOP`).
const MAX_TOP_BITS: u32 = 9;

/// An XYZZ record of the device (`g1::Xyzz`): `X, Y, ZZZ, ZZ`, six limbs each.
type Xyzz = [u64; 24];

// Scalar modes of `msm_digits`.
const MODE_MASKED: u32 = 1;
const MODE_MASK: u32 = 2;

// ---------------------------------------------------------------------------
// The plan of one MSM (sppark's `msm_t` constructor and `invoke` choose it
// by formula; here a measured table).

#[derive(Clone, Copy, Debug)]
struct Plan {
    n: usize,
    wbits: u32,
    nwins: u32,
    batch: usize,
    stride: usize,
    /// `wbits - 1`: bucket index bits.
    bits: u32,
    top_bits: u32,
    low_bits: u32,
    /// Bits of the top window (sppark `top`).
    top: u32,
    /// Windows per sort pass.
    pass: u32,
}

/// sppark `lg2`: floor of log2, 0 for 0.
const fn lg2(mut n: usize) -> u32 {
    let mut r = 0;
    while n > 1 {
        n >>= 1;
        r += 1;
    }
    r
}

/// The plan table: `(log2 n, wbits, chunks before the command-buffer
/// bound)`.
///
/// Measured on Apple M4 Max (40-core GPU, 128 GB, macOS), 2026-11-18, release
/// build: the registered-table path (one MSM pass) over random scalars and
/// random points, for `wbits` in 10..=16 (sizes up to 2^18) or 12..=17
/// (2^19 to 2^22) and chunk counts {1, 2, 4} (more where the command-buffer
/// bound applies), each configuration
/// checked against `msm_pippenger` and timed as the minimum of 4 calls,
/// sizes with close candidates repeated two or three times and the minimum
/// over the runs taken; the one-shot path was timed on the leading
/// candidates and agreed. Other processes shared the machine. Minimum wall
/// time per call (ms), best and runner-up:
///
/// | n    | best (wbits, chunks) | ms    | runner-up | ms    | sppark (wbits, chunks) |
/// |------|----------------------|-------|-----------|-------|------------------------|
/// | 2^14 | 11, 1                | 6.75  | 10, 1     | 7.34  | 10, 1                  |
/// | 2^15 | 11, 1                | 9.57  | 12, 1     | 9.10* | 10, 1                  |
/// | 2^16 | 13, 1                | 13.57 | 12, 1     | 14.12 | 10, 1                  |
/// | 2^17 | 13, 1                | 21.47 | 12, 1     | 23.13 | 10, 2                  |
/// | 2^18 | 13, 1                | 34.13 | 12, 1     | 39.15 | 10, 4                  |
/// | 2^19 | 13, 1                | 57.92 | 14, 1     | 62.55 | 11, 4                  |
/// | 2^20 | 13, 1                | 105.3 | 15, 1     | 108.6 | 12, 4                  |
/// | 2^21 | 16, 1                | 204.0 | 13, 1     | 212.5 | 13, 4                  |
/// | 2^22 | 14, 1 (3: the bound) | 412.5 | 13, 4     | 417.2 | 14, 4                  |
///
/// (*) At 2^15 one table run favoured 12 by 0.5 ms and the other two runs
/// and the one-shot path favoured 11; 11 is kept.
///
/// Outside the table: below `2^14` the `2^14` entry; from `2^23` the
/// `2^22` entry, where the command-buffer bound makes every chunk at most
/// `CHUNK_ADDS / nwins` points, the chunk size measured at `2^22`. A length
/// that is not a power of two takes the entry of its floor log2.
const MEASURED: [(u32, u32, usize); 9] = [
    (14, 11, 1),
    (15, 11, 1),
    (16, 13, 1),
    (17, 13, 1),
    (18, 13, 1),
    (19, 13, 1),
    (20, 13, 1),
    (21, 16, 1),
    (22, 14, 1),
];

/// `(wbits, chunks)` of `MEASURED` for `n` points.
fn measured(n: usize) -> (u32, usize) {
    let (lo, hi) = (MEASURED[0].0, MEASURED[MEASURED.len() - 1].0);
    let k = lg2(n).clamp(lo, hi);
    let e = MEASURED[(k - lo) as usize];
    (e.1, e.2)
}

fn plan(n: usize) -> Plan {
    let (wbits, batch) = measured(n);
    let nwins = (NBITS - 1) / wbits + 1;
    // the time budget of one command buffer: at most CHUNK_ADDS bucket
    // additions per chunk (this port; sppark has no such bound)
    let max_stride = (CHUNK_ADDS / nwins as usize) & !31;
    let stride = n.div_ceil(batch).next_multiple_of(32).min(max_stride);
    let batch = n.div_ceil(stride);
    let bits = wbits - 1;
    let top_bits = bits / 2;
    let low_bits = bits - top_bits;
    let pass = (SORT_TEMP_BYTES / (8 * stride)).clamp(2, nwins as usize) as u32;
    Plan {
        n,
        wbits,
        nwins,
        batch,
        stride,
        bits,
        top_bits,
        low_bits,
        top: NBITS - wbits * (nwins - 1),
        pass,
    }
}

// ---------------------------------------------------------------------------
// Registered tables.

/// A registered table: the host copy of the points until the first MSM, then
/// the device buffer; the mask MSMs by length.
struct Table {
    id: u64,
    host: Vec<G1AffineLimbs>,
    dev: Option<Buffer>,
    masks: HashMap<usize, G1JacobianLimbs>,
}

impl Table {
    fn points(&self) -> &[G1AffineLimbs] {
        match &self.dev {
            Some(b) => &b.as_slice::<G1AffineLimbs>()[..self.len()],
            None => &self.host,
        }
    }

    fn len(&self) -> usize {
        match &self.dev {
            Some(b) => b.len_bytes() / size_of::<G1AffineLimbs>(),
            None => self.host.len(),
        }
    }

    /// Whether the table still has the caller's points (a prefix of it):
    /// every point in a debug build, eight sampled points in a release build.
    fn serves(&self, bases: &[G1AffineLimbs]) -> bool {
        let t = self.points();
        let n = bases.len();
        if n > t.len() {
            return false;
        }
        if cfg!(debug_assertions) || n <= 8 {
            return t[..n] == *bases;
        }
        [
            0,
            n / 7,
            2 * n / 7,
            3 * n / 7,
            4 * n / 7,
            5 * n / 7,
            6 * n / 7,
            n - 1,
        ]
        .iter()
        .all(|&i| t[i] == bases[i])
    }
}

/// Tables keyed by (address, length) of the registered slice.
static TABLES: Mutex<Option<HashMap<(usize, usize), Table>>> = Mutex::new(None);
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
static STALE: AtomicU64 = AtomicU64::new(0);
static UPLOADS: AtomicU64 = AtomicU64::new(0);
static CACHED: AtomicU64 = AtomicU64::new(0);
static ONE_SHOT: AtomicU64 = AtomicU64::new(0);

fn tables() -> MutexGuard<'static, Option<HashMap<(usize, usize), Table>>> {
    TABLES.lock().unwrap_or_else(|e| e.into_inner())
}

/// The registration of an SRS base slice. While it lives, the registrant
/// promises to keep the slice alive and unchanged; MSMs on it or on a prefix
/// of it use a device table. Dropping it frees the table.
#[derive(Debug)]
pub struct SrsHandle {
    id: u64,
    key: (usize, usize),
}

impl SrsHandle {
    /// Number of registered points.
    pub fn len(&self) -> usize {
        self.key.1
    }

    /// Whether no point is registered.
    pub fn is_empty(&self) -> bool {
        self.key.1 == 0
    }

    /// Whether `bases` is the registered slice with the registered content
    /// (every point compared).
    pub fn verify(&self, bases: &[G1AffineLimbs]) -> bool {
        (bases.as_ptr() as usize, bases.len()) == self.key
            && tables()
                .as_ref()
                .and_then(|t| t.get(&self.key))
                .filter(|e| e.id == self.id)
                .is_some_and(|e| e.points() == bases)
    }
}

impl Drop for SrsHandle {
    fn drop(&mut self) {
        let removed = {
            let mut t = tables();
            match t.as_mut() {
                Some(m) if m.get(&self.key).is_some_and(|e| e.id == self.id) => m.remove(&self.key),
                _ => None,
            }
        };
        drop(removed);
    }
}

/// Registers a host copy of an SRS base slice (no device work: the first MSM
/// on the slice uploads it). Registering the same slice again replaces the
/// table.
pub fn register_srs(bases: &[G1AffineLimbs]) -> SrsHandle {
    let key = (bases.as_ptr() as usize, bases.len());
    let id = NEXT_ID.fetch_add(1, Relaxed);
    let t = Table {
        id,
        host: bases.to_vec(),
        dev: None,
        masks: HashMap::new(),
    };
    let old = tables().get_or_insert_with(HashMap::new).insert(key, t);
    drop(old);
    SrsHandle { id, key }
}

/// Live registered tables.
pub fn srs_handles() -> usize {
    tables().as_ref().map_or(0, |t| t.len())
}

/// Tables dropped at an MSM because the caller's slice no longer matched.
pub fn srs_stale() -> u64 {
    STALE.load(Relaxed)
}

/// (MSMs served from a registered table, one-shot MSMs, table uploads) since
/// process start.
pub fn msm_counts() -> (u64, u64, u64) {
    (
        CACHED.load(Relaxed),
        ONE_SHOT.load(Relaxed),
        UPLOADS.load(Relaxed),
    )
}

/// The table that serves an MSM of `m` scalars on the slice `key`: the exact
/// entry, else the shortest entry at the same address that is longer.
fn table_for(
    t: &HashMap<(usize, usize), Table>,
    key: (usize, usize),
    m: usize,
) -> Option<(usize, usize)> {
    if m > key.1 {
        return None;
    }
    if t.contains_key(&key) {
        return Some(key);
    }
    t.keys().filter(|k| k.0 == key.0 && k.1 > key.1).min_by_key(|k| k.1).copied()
}

// ---------------------------------------------------------------------------
// Entry points.

/// `sum scalars[i] * bases[i]`, or `None` (declined).
pub(crate) fn msm_g1(scalars: &[Scalar], bases: &[G1AffineLimbs]) -> Option<G1JacobianLimbs> {
    match dispatch(scalars, bases, None) {
        Ok((r, ns)) => {
            account::served(Class::Msm, ns);
            Some(r)
        }
        Err(e) => {
            account::decline(Class::Msm, Reason::of(&e));
            None
        }
    }
}

/// The time of each stage of one MSM ([`msm_g1_stages`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MsmStages {
    /// Window bits.
    pub wbits: u32,
    /// Windows.
    pub nwins: u32,
    /// Chunks (command buffers per MSM pass).
    pub batches: u32,
    /// Device MSM passes run: 2 (`MSM(s + r)` and `MSM(r)`), or 1 when the
    /// table had `MSM(r)` cached.
    pub passes: u32,
    /// Whether a registered table served the call.
    pub table: bool,
    /// Host copies into device buffers (scalars, and points on the one-shot
    /// path or a table's first use), wall time.
    pub upload_ns: u64,
    /// GPU time of `msm_digits`.
    pub digits_ns: u64,
    /// GPU time of the sort kernels.
    pub sort_ns: u64,
    /// GPU time of `msm_ones` and `msm_accumulate`.
    pub accumulate_ns: u64,
    /// GPU time of `msm_integrate`.
    pub integrate_ns: u64,
    /// Host time reading the partial sums and combining them (`blst`).
    pub collect_ns: u64,
    /// The most points in one bucket (any window, any chunk, any pass).
    pub bucket_max: u32,
    /// Points summed in buckets, over all windows, chunks and passes
    /// (`bucket_entries / bucket_slots` is the mean bucket load).
    pub bucket_entries: u64,
    /// Buckets, over all windows, chunks and passes.
    pub bucket_slots: u64,
}

/// [`crate::msm_g1`] with each stage in its own command buffer and timed (a
/// measurement entry: the stages do not overlap and the host collection
/// does not overlap the device, so the total is above that of
/// [`crate::msm_g1`]). Not counted in [`crate::stats`] or
/// [`crate::declines`]. `None`: a shape or scalar [`crate::msm_g1`] would
/// decline, or a device failure.
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
    let (r, _) = dispatch(scalars, bases, Some(&mut st)).ok()?;
    Some((r, st))
}

/// The table or one-shot path, then the masked MSM.
fn dispatch(
    scalars: &[Scalar],
    bases: &[G1AffineLimbs],
    mut prof: Option<&mut MsmStages>,
) -> Result<(G1JacobianLimbs, u64), Error> {
    let n = scalars.len();
    let key = (bases.as_ptr() as usize, bases.len());
    let mut guard = tables();
    if let Some(t) = guard.as_mut() {
        while let Some(tk) = table_for(t, key, n).filter(|tk| !t[tk].serves(bases)) {
            STALE.fetch_add(1, Relaxed);
            t.remove(&tk);
        }
        if let Some(tk) = table_for(t, key, n) {
            let Table {
                host, dev, masks, ..
            } = t.get_mut(&tk).unwrap();
            let mut s = ctx::lock()?;
            if dev.is_none() {
                let t0 = Instant::now();
                *dev = Some(s.ctx().upload(host)?);
                *host = Vec::new();
                UPLOADS.fetch_add(1, Relaxed);
                if let Some(p) = prof.as_deref_mut() {
                    p.upload_ns += t0.elapsed().as_nanos() as u64;
                }
            }
            CACHED.fetch_add(1, Relaxed);
            if let Some(p) = prof.as_deref_mut() {
                p.table = true;
            }
            let points = dev.as_ref().unwrap();
            return masked(&mut s, Points::Table(points), scalars, Some(masks), prof);
        }
    }
    drop(guard);
    ONE_SHOT.fetch_add(1, Relaxed);
    let mut s = ctx::lock()?;
    masked(&mut s, Points::Host(bases), scalars, None, prof)
}

enum Points<'a> {
    /// A registered table's device buffer (a prefix of it is used).
    Table(&'a Buffer),
    /// The caller's points, copied for the call.
    Host(&'a [G1AffineLimbs]),
}

const K_SCALARS: &str = "msm-scalars";
const K_POINTS: &str = "msm-points";
const K_DIGITS: &str = "msm-digits";
const K_TEMPS: &str = "msm-temps";
const K_COUNTS: &str = "msm-counts";
const K_HIST: &str = "msm-hist";
const K_BUCKETS: &str = "msm-buckets";
const K_ONES: &str = "msm-ones";

/// `MSM(s + r) - MSM(r)` (see the module text). Returns the result and the
/// GPU time.
fn masked(
    s: &mut Session,
    points: Points<'_>,
    scalars: &[Scalar],
    masks: Option<&mut HashMap<usize, G1JacobianLimbs>>,
    mut prof: Option<&mut MsmStages>,
) -> Result<(G1JacobianLimbs, u64), Error> {
    let n = scalars.len();
    let pl = plan(n);
    let c = s.ctx();
    let geo = Geometry::of(c, &pl)?;

    let t0 = Instant::now();
    let ntop = 1usize << pl.top_bits;
    let rows = (pl.nwins as usize) << pl.bits;
    s.scratch(K_DIGITS, pl.nwins as usize * pl.stride * 4)?;
    s.scratch(K_TEMPS, pl.pass as usize * pl.stride * 8)?;
    s.scratch(K_COUNTS, pl.pass as usize * ntop * SORT_GROUPS as usize * 4)?;
    s.scratch(K_HIST, rows * 4)?;
    s.scratch(K_BUCKETS, rows * size_of::<Xyzz>())?;
    s.scratch(K_ONES, geo.ones_slots() * size_of::<Xyzz>())?;
    s.scratch(K_SCALARS, size_of_val(scalars))?.write(scalars);
    if let Points::Host(b) = points {
        s.scratch(K_POINTS, size_of_val(b))?.write(b);
    }
    if let Some(p) = prof.as_deref_mut() {
        p.upload_ns += t0.elapsed().as_nanos() as u64;
        p.wbits = pl.wbits;
        p.nwins = pl.nwins;
        p.batches = pl.batch as u32;
    }
    let [digits, temps, counts, hist, buckets, ones, sc] = s
        .scratch_many([
            K_DIGITS, K_TEMPS, K_COUNTS, K_HIST, K_BUCKETS, K_ONES, K_SCALARS,
        ])
        .ok_or_else(|| Error::Memory("msm scratch".into()))?;
    let points = match points {
        Points::Table(b) => b,
        Points::Host(_) => {
            s.get_scratch(K_POINTS).ok_or_else(|| Error::Memory("msm scratch".into()))?
        }
    };
    let bufs = Bufs {
        points,
        scalars: sc,
        digits,
        temps,
        counts,
        hist,
        buckets,
        ones,
    };

    let mut ns = 0u64;
    let cached = masks.as_ref().and_then(|m| m.get(&n).copied());
    let rr = match cached {
        Some(rr) => rr,
        None => {
            let (rr, t) = run(c, &bufs, &pl, &geo, MODE_MASK, prof.as_deref_mut())?;
            ns += t;
            let rr = limbs_of(&rr);
            if let Some(m) = masks {
                m.insert(n, rr);
            }
            rr
        }
    };
    let (mut sr, t) = run(c, &bufs, &pl, &geo, MODE_MASKED, prof.as_deref_mut())?;
    ns += t;
    if let Some(p) = prof {
        p.passes = if cached.is_some() { 1 } else { 2 };
    }
    let mut nr = p1_of(&rr);
    // SAFETY: valid blst points.
    unsafe {
        blst_p1_cneg(&mut nr, true);
    }
    add(&mut sr, &nr);
    let out = limbs_of(&sr);
    wipe(std::slice::from_mut(&mut sr));
    Ok((out, ns))
}

/// Thread counts that depend on the pipelines.
#[derive(Clone, Copy, Debug)]
struct Geometry {
    sort_block: usize,
    ones_block: usize,
    /// `msm_integrate` threads per window (a power of two).
    nthr: usize,
}

impl Geometry {
    fn of(c: &Ctx, pl: &Plan) -> Result<Geometry, Error> {
        let max = |k: &str| -> Result<usize, Error> {
            Ok(c.pipeline(k)?.maxTotalThreadsPerThreadgroup())
        };
        let pow2_floor = |x: usize| if x == 0 { 0 } else { 1usize << lg2(x) };
        let sort_block = [
            "msm_sort_count",
            "msm_sort_scan",
            "msm_sort_scatter",
            "msm_sort_lower",
        ]
        .iter()
        .map(|k| max(k))
        .try_fold(SORT_BLOCK, |a, m| m.map(|m| a.min(pow2_floor(m))))?;
        let ones_block = pow2_floor(max("msm_ones")?.min(ONES_BLOCK));
        let nthr = pow2_floor(max("msm_integrate")?.min(MSM_NTHREADS));
        if sort_block < 32 || ones_block < 32 || nthr < 2 || lg2(nthr) >= pl.bits {
            return Err(Error::Device(format!(
                "msm: threadgroup sizes {sort_block}/{ones_block}/{nthr} too small"
            )));
        }
        if pl.low_bits > MAX_LOW_BITS || pl.top_bits > MAX_TOP_BITS {
            return Err(Error::Device("msm: window too wide for the sort".into()));
        }
        Ok(Geometry {
            sort_block,
            ones_block,
            nthr,
        })
    }

    fn ones_slots(&self) -> usize {
        ONES_GROUPS * self.ones_block / 32
    }
}

struct Bufs<'a> {
    points: &'a Buffer,
    scalars: &'a Buffer,
    digits: &'a Buffer,
    temps: &'a Buffer,
    counts: &'a Buffer,
    hist: &'a Buffer,
    buckets: &'a Buffer,
    ones: &'a Buffer,
}

/// The kernel parameters of one chunk.
struct ChunkParams {
    digits: [u32; 6],
    sort: Vec<[u32; 9]>,
    acc: [u32; 3],
    int: [u32; 3],
}

fn chunk_params(pl: &Plan, mode: u32, h_off: usize, num: usize) -> ChunkParams {
    let sort = (0..pl.nwins)
        .step_by(pl.pass as usize)
        .map(|w0| {
            [
                num as u32,
                pl.stride as u32,
                w0,
                pl.nwins,
                SORT_GROUPS,
                pl.top_bits,
                pl.low_bits,
                pl.bits,
                pl.top - 1,
            ]
        })
        .collect();
    ChunkParams {
        digits: [
            num as u32,
            h_off as u32,
            pl.nwins,
            pl.wbits,
            pl.stride as u32,
            mode,
        ],
        sort,
        acc: [pl.nwins, pl.bits, pl.stride as u32],
        int: [pl.nwins, pl.wbits, NBITS],
    }
}

fn enc_digits<'a>(
    cmd: &mut Command<'a>,
    b: &Bufs<'a>,
    p: &'a ChunkParams,
    h_off: usize,
    num: usize,
) -> Result<(), Error> {
    let sc = if p.digits[5] == MODE_MASK {
        Arg::Buffer(b.digits)
    } else {
        Arg::BufferAt(b.scalars, h_off * size_of::<Scalar>())
    };
    cmd.dispatch(
        "msm_digits",
        num,
        &[sc, Arg::Buffer(b.digits), Arg::value(&p.digits)],
    )
}

fn enc_sort<'a>(
    cmd: &mut Command<'a>,
    b: &Bufs<'a>,
    p: &'a ChunkParams,
    pl: &Plan,
    g: &Geometry,
) -> Result<(), Error> {
    let ntop = 1usize << pl.top_bits;
    for sp in &p.sort {
        let pw = (pl.nwins - sp[2]).min(pl.pass) as usize;
        let groups = pw * SORT_GROUPS as usize;
        let prm = Arg::value(sp);
        cmd.dispatch_groups(
            "msm_sort_count",
            groups,
            g.sort_block,
            &[Arg::Buffer(b.digits), Arg::Buffer(b.counts), prm],
            &[],
        )?;
        cmd.dispatch_groups(
            "msm_sort_scan",
            pw,
            g.sort_block,
            &[Arg::Buffer(b.counts), Arg::Buffer(b.hist), prm],
            &[],
        )?;
        cmd.dispatch_groups(
            "msm_sort_scatter",
            groups,
            g.sort_block,
            &[
                Arg::Buffer(b.digits),
                Arg::Buffer(b.counts),
                Arg::Buffer(b.temps),
                prm,
            ],
            &[],
        )?;
        cmd.dispatch_groups(
            "msm_sort_lower",
            pw * ntop,
            g.sort_block,
            &[
                Arg::Buffer(b.digits),
                Arg::Buffer(b.temps),
                Arg::Buffer(b.hist),
                prm,
            ],
            &[],
        )?;
    }
    Ok(())
}

fn enc_accumulate<'a>(
    cmd: &mut Command<'a>,
    b: &Bufs<'a>,
    p: &'a ChunkParams,
    pl: &Plan,
    g: &Geometry,
    h_off: usize,
) -> Result<(), Error> {
    let pts = Arg::BufferAt(b.points, h_off * size_of::<G1AffineLimbs>());
    cmd.dispatch_groups(
        "msm_ones",
        ONES_GROUPS,
        g.ones_block,
        &[
            Arg::Buffer(b.ones),
            pts,
            Arg::Buffer(b.digits),
            Arg::Buffer(b.hist),
        ],
        &[],
    )?;
    cmd.dispatch(
        "msm_accumulate",
        (pl.nwins as usize) << pl.bits,
        &[
            Arg::Buffer(b.buckets),
            pts,
            Arg::Buffer(b.digits),
            Arg::Buffer(b.hist),
            Arg::value(&p.acc),
        ],
    )
}

fn enc_integrate<'a>(
    cmd: &mut Command<'a>,
    b: &Bufs<'a>,
    p: &'a ChunkParams,
    pl: &Plan,
    g: &Geometry,
) -> Result<(), Error> {
    cmd.dispatch_groups(
        "msm_integrate",
        pl.nwins as usize,
        g.nthr,
        &[Arg::Buffer(b.buckets), Arg::value(&p.int)],
        &[],
    )
}

/// One device MSM (sppark `invoke`) over the first `pl.n` points of
/// `b.points` with the scalars of `mode`. Returns the sum and the GPU time.
fn run(
    c: &Ctx,
    b: &Bufs<'_>,
    pl: &Plan,
    g: &Geometry,
    mode: u32,
    mut prof: Option<&mut MsmStages>,
) -> Result<(blst_p1, u64), Error> {
    let per_row = 2 * g.nthr;
    let row_len = 1usize << pl.bits;
    std::thread::scope(|sc| {
        let mut pending = Vec::with_capacity(pl.batch);
        let mut gpu = 0u64;
        let mut h_off = 0usize;
        let mut err = None;
        for _ in 0..pl.batch {
            let num = pl.stride.min(pl.n - h_off);
            let p = chunk_params(pl, mode, h_off, num);
            let ran = match prof.as_deref_mut() {
                None => c
                    .run(|cmd| {
                        enc_digits(cmd, b, &p, h_off, num)?;
                        enc_sort(cmd, b, &p, pl, g)?;
                        enc_accumulate(cmd, b, &p, pl, g, h_off)?;
                        enc_integrate(cmd, b, &p, pl, g)
                    })
                    .map(|r| r.1),
                Some(st) => staged(c, b, &p, pl, g, h_off, num, st),
            };
            match ran {
                Ok(ns) => gpu += ns,
                Err(e) => {
                    err = Some(e);
                    break;
                }
            }
            // the partial sums of this chunk, copied out of the device
            // buffers before the next chunk reuses them
            let t0 = Instant::now();
            let bk = b.buckets.as_slice::<Xyzz>();
            let mut rows: Vec<Xyzz> = Vec::with_capacity(pl.nwins as usize * per_row);
            for y in 0..pl.nwins as usize {
                rows.extend_from_slice(&bk[y * row_len..y * row_len + per_row]);
            }
            let ones = b.ones.as_slice::<Xyzz>()[..g.ones_slots()].to_vec();
            match prof.as_deref_mut() {
                None => pending.push(sc.spawn(move || collect(rows, ones, pl, g.nthr))),
                Some(st) => {
                    let r = collect(rows, ones, pl, g.nthr);
                    st.collect_ns += t0.elapsed().as_nanos() as u64;
                    pending.push(sc.spawn(move || r));
                }
            }
            h_off += num;
        }
        let t0 = Instant::now();
        let mut out = blst_p1::default();
        for h in pending {
            match h.join() {
                Ok(mut p) => {
                    add(&mut out, &p);
                    wipe(std::slice::from_mut(&mut p));
                }
                Err(_) => err = err.or(Some(Error::Device("msm: host collect failed".into()))),
            }
        }
        if let Some(st) = prof {
            st.collect_ns += t0.elapsed().as_nanos() as u64;
        }
        match err {
            Some(e) => {
                wipe(std::slice::from_mut(&mut out));
                Err(e)
            }
            None => Ok((out, gpu)),
        }
    })
}

/// One chunk with each stage in its own command buffer, timed.
#[allow(clippy::too_many_arguments)]
fn staged(
    c: &Ctx,
    b: &Bufs<'_>,
    p: &ChunkParams,
    pl: &Plan,
    g: &Geometry,
    h_off: usize,
    num: usize,
    st: &mut MsmStages,
) -> Result<u64, Error> {
    let d = c.run(|cmd| enc_digits(cmd, b, p, h_off, num))?.1;
    let s = c.run(|cmd| enc_sort(cmd, b, p, pl, g))?.1;
    // bucket loads, from the sorted histogram
    let hist = b.hist.as_slice::<u32>();
    let row_len = 1usize << pl.bits;
    let mut total = 0u64;
    for y in 0..pl.nwins as usize {
        let h = &hist[y * row_len..(y + 1) * row_len];
        let mut prev = 0;
        for &e in h {
            st.bucket_max = st.bucket_max.max(e - prev);
            prev = e;
        }
        total += u64::from(prev);
    }
    st.bucket_entries += total;
    st.bucket_slots += pl.nwins as u64 * row_len as u64;
    let a = c.run(|cmd| enc_accumulate(cmd, b, p, pl, g, h_off))?.1;
    let i = c.run(|cmd| enc_integrate(cmd, b, p, pl, g))?.1;
    st.digits_ns += d;
    st.sort_ns += s;
    st.accumulate_ns += a;
    st.integrate_ns += i;
    Ok(d + s + a + i)
}

// ---------------------------------------------------------------------------
// Host collection (sppark `collect`, `integrate_row`, `sum_up`) with blst.

fn add(a: &mut blst_p1, b: &blst_p1) {
    let t = *a;
    // SAFETY: valid blst points; the output does not alias the inputs.
    unsafe { blst_p1_add_or_double(a, &t, b) };
}

fn dbl(a: &mut blst_p1) {
    let t = *a;
    // SAFETY: as above.
    unsafe { blst_p1_double(a, &t) };
}

/// An XYZZ record as Jacobian coordinates `(X ZZ, Y ZZZ, ZZ)` (sppark's
/// `xyzz_t` to `jacobian_t` conversion); infinity (`ZZ = 0`) stays infinity.
fn p1_of_xyzz(r: &Xyzz) -> blst_p1 {
    let fp = |i: usize| blst_fp {
        l: r[6 * i..6 * i + 6].try_into().unwrap(),
    };
    let (x, y, zzz, zz) = (fp(0), fp(1), fp(2), fp(3));
    let mut p = blst_p1 {
        z: zz,
        ..Default::default()
    };
    // SAFETY: valid field elements; outputs do not alias inputs.
    unsafe {
        blst_fp_mul(&mut p.x, &x, &zz);
        blst_fp_mul(&mut p.y, &y, &zzz);
    }
    p
}

fn p1_of(l: &G1JacobianLimbs) -> blst_p1 {
    let fp = |i: usize| blst_fp {
        l: l[6 * i..6 * i + 6].try_into().unwrap(),
    };
    blst_p1 {
        x: fp(0),
        y: fp(1),
        z: fp(2),
    }
}

fn limbs_of(p: &blst_p1) -> G1JacobianLimbs {
    let mut out = [0u64; 18];
    out[..6].copy_from_slice(&p.x.l);
    out[6..12].copy_from_slice(&p.y.l);
    out[12..].copy_from_slice(&p.z.l);
    out
}

/// Writes zero over `v` (host copies of values derived from the scalars).
fn wipe<T: Copy>(v: &mut [T]) {
    // SAFETY: `T` here is a plain-data type (limb arrays, blst points) for
    // which all-zero bytes are a valid value.
    unsafe { std::ptr::write_bytes(v.as_mut_ptr(), 0, v.len()) };
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

/// sppark `integrate_row`: the window sum from its `nthr` pairs (weighted
/// run sum, plain run sum) of `msm_integrate`; `lsbits` is the window's
/// width.
fn integrate_row(row: &[Xyzz], lsbits: u32, nthrbits: u32) -> blst_p1 {
    let rec = |i: usize, j: usize| p1_of_xyzz(&row[2 * i + j]);
    let mut i = (1usize << nthrbits) - 1;
    if lsbits - 1 <= nthrbits {
        let mask = (1usize << (nthrbits - (lsbits - 1))) - 1;
        let mut acc = rec(i, 1);
        let mut res = if mask != 0 { blst_p1::default() } else { acc };
        while i > 0 {
            i -= 1;
            add(&mut acc, &rec(i, 1));
            if i & mask == 0 {
                add(&mut res, &acc);
            }
        }
        wipe(std::slice::from_mut(&mut acc));
        return res;
    }
    let mut res = rec(i, 0);
    let mut acc = rec(i, 1);
    while i > 0 {
        i -= 1;
        let mut raise = acc;
        for _ in 0..(lsbits - 1 - nthrbits) {
            dbl(&mut raise);
        }
        add(&mut res, &raise);
        add(&mut res, &rec(i, 0));
        if i != 0 {
            add(&mut acc, &rec(i, 1));
        }
        wipe(std::slice::from_mut(&mut raise));
    }
    wipe(std::slice::from_mut(&mut acc));
    res
}

/// sppark `collect`: the window sums (rows integrated in parallel), the
/// doubling ladder from the top window, plus the bucket of digit 1 of window
/// 0 (`ones`). Zeroes its inputs.
fn collect(mut rows: Vec<Xyzz>, mut ones: Vec<Xyzz>, pl: &Plan, nthr: usize) -> blst_p1 {
    let nwins = pl.nwins as usize;
    let per_row = 2 * nthr;
    let nthrbits = lg2(nthr);
    let mut sums = vec![blst_p1::default(); nwins];
    let workers = std::thread::available_parallelism().map_or(4, |t| t.get()).clamp(1, nwins);
    let per = nwins.div_ceil(workers);
    std::thread::scope(|s| {
        for (k, out) in sums.chunks_mut(per).enumerate() {
            let rows = &rows;
            s.spawn(move || {
                for (j, o) in out.iter_mut().enumerate() {
                    let y = k * per + j;
                    let dy = if y == nwins - 1 { pl.top } else { pl.wbits };
                    *o = integrate_row(&rows[y * per_row..(y + 1) * per_row], dy, nthrbits);
                }
            });
        }
    });
    let mut out = blst_p1::default();
    for y in (0..nwins).rev() {
        add(&mut out, &sums[y]);
        if y > 0 {
            for _ in 0..pl.wbits {
                dbl(&mut out);
            }
        }
    }
    for r in &ones {
        add(&mut out, &p1_of_xyzz(r));
    }
    wipe(&mut sums);
    wipe(&mut rows);
    wipe(&mut ones);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_follows_measured_table() {
        // (log2 n, wbits, nwins, chunks after the command-buffer bound)
        for (k, wbits, nwins, batch) in [
            (14, 11, 24, 1),
            (15, 11, 24, 1),
            (16, 13, 20, 1),
            (17, 13, 20, 1),
            (18, 13, 20, 1),
            (19, 13, 20, 1),
            (20, 13, 20, 1),
            (21, 16, 16, 1),
            (22, 14, 19, 3),
        ] {
            let p = plan(1 << k);
            assert_eq!((p.wbits, p.nwins, p.batch), (wbits, nwins, batch), "2^{k}");
            assert_eq!(p.top, NBITS - wbits * (nwins - 1));
            assert!(p.stride * p.batch >= 1 << k, "2^{k}");
        }
        // outside the table: the end entries; a non-power of two: its floor
        assert_eq!(plan(1).wbits, 11);
        assert_eq!(plan(1000).wbits, 11);
        assert_eq!(plan((1 << 21) - 1).wbits, 13);
        assert_eq!(plan((1 << 21) + 5).wbits, 16);
        // the command-buffer bound up to 2^28
        for k in 0..=crate::MAX_MSM_LOG {
            for n in [1usize << k, (1usize << k) + 33] {
                let p = plan(n);
                assert!(p.stride * p.nwins as usize <= CHUNK_ADDS, "{n}");
                assert!(
                    p.stride * p.batch >= n && p.stride * (p.batch - 1) < n,
                    "{n}"
                );
                assert!(p.stride.is_multiple_of(32), "{n}");
                if k >= 22 {
                    assert_eq!(p.wbits, 14, "{n}");
                }
            }
        }
        assert_eq!(plan(1 << 23).batch, 5);
        assert_eq!(plan(1 << 28).batch, 153);
    }
}
