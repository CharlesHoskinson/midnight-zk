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
//! `2^MAX_MSM_LOG`, at least `2^msm_table` of [`crate::thresholds`], and at
//! least `2^msm_one_shot` unless a registered table serves the bases, a
//! usable device, every scalar reduced) before it calls [`msm_g1`]. That
//! function ends with exactly one of `account::served` or `account::decline`
//! (`below-min-k` when the table it counted on turned out stale and the
//! call's effective size is below the one-shot threshold).
//!
//! # Effective size
//!
//! The size compared with the thresholds is the MSM's count of non-zero
//! scalars (1 when all are zero), not its length: a zero scalar adds nothing,
//! and the masked device MSM costs the same for 2 non-zero scalars out of
//! `2^20` as for a dense one (about 108 ms on Apple M4 Max), while the stock
//! CPU MSM, whose time already follows the sparsity, takes about a
//! millisecond. [`crate::msm_g1`] checks it (with
//! [`effective_size_reaches`]) after the length admission and before any
//! device work, against the need of the path the call would take
//! ([`effective_need`]): `2^12 + n/5` with a registered table,
//! else `max(2^msm_one_shot, n/3)`, the measured crossovers of the stock CPU
//! MSM on sparse inputs ([`SPARSE_TABLE_DIV`]); below it, the call declines
//! `below-min-k`, the ordinary reason of a short MSM. The count itself is
//! not kept, logged, counted or returned: the decline is the only
//! observable, and it is the one a short dense MSM gives. An explicit
//! threshold ([`crate::set_min_k`], `MIDNIGHT_METAL_MIN_K`) replaces the
//! need entirely (`2^min_k`, no fraction of `n`), so `set_min_k(0)` sends
//! every MSM, the all-zero one included, to the device.
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
//!    Each threadgroup first orders its buckets by length (longest first),
//!    so that the 32 lanes of a SIMD-group sum buckets of similar length;
//!    the field multiplication of the curve formulas is one out-of-line
//!    function (see `shaders/include/g1.h`).
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
//! # The CPU share
//!
//! While the device runs, the CPU computes part of the MSM: the device takes
//! the points `[0, m)` (masked, as above, with `MSM(r)` cached per table
//! and share length) and the stock `msm_pippenger` of midnight-curves takes
//! `[m, n)` with the plain scalars on another thread at the same time; the
//! two sums are added on the host, which gives the same group element. `m`
//! is a fixed function of `n` and of the path (table or one-shot) only,
//! measured on this hardware (`MEASURED_SHARE`); scalar values never move
//! it (the effective-size decline above runs before it). The CPU share runs
//! on its own rayon pool, never on the caller's: the prover calls MSMs from
//! rayon workers, which can all be blocked on the device lock while the
//! holder waits for its CPU share, so a share queued on their pool could
//! never start. It reads the caller's slices in place (no host copy of the
//! scalars is made for it; `msm_pippenger` makes its own, as the stock code
//! does). `MIDNIGHT_METAL_MSM_CPU_SHARE=off` gives `m = n` (device only).
//! The device time accounted for the call is the device share's GPU time.
//! A failure of either share declines the whole call.
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
//! public points only; it lives until the handle is dropped
//! ([`SrsHandle::device_bytes`] says how many bytes it holds on the device).
//!
//! A table serves the registered slice and its prefixes, by address: the
//! caller has to register the very slice the prover passes. The KZG prover of
//! `midnight-proofs` commits from four SRS vectors of `ParamsKZG`, each a
//! prefix: `bases::<Coeff>()` (the quotient limbs, the multi-open),
//! `bases::<LagrangeCoeff>()` (committed instances, and the fixed columns at
//! key generation), `bases::<LagrangeDeltaCoeff>()` (the advice
//! columns and LogUp multiplicities, phase 1) and
//! `bases::<LagrangeDoubleDeltaCoeff>()` (permutation products, LogUp
//! helpers and aggregators, trash, phase 2); a registrant registers all four.
//! Every other MSM base slice of the prover (the residue bases of an
//! interleaved commitment, a copy) takes the one-shot path.
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
    lg2_nearest,
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
/// Threads of an `msm_accumulate` group (`ACC_GROUP` in msm.metal).
const ACC_GROUP: usize = 256;
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
/// Re-checked on 2026-11-19 after the accumulation kernel changed (an
/// out-of-line field multiplication, buckets ordered by load within each
/// threadgroup), on the table path with the CPU share off, `wbits` in
/// 12..=17 around the entries of 2^18 to 2^22 and at the device-share sizes
/// of the split (`MEASURED_SHARE`), GPU time plus host collection, minimum
/// of 3 calls, two rounds (another process ran GPU work at the same time;
/// the rounds differed by up to 20% at 2^22, the minima are given). Wider
/// windows now win from 2^20; 2^18, 2^19 and 2^21 keep their entries:
///
/// | n          | best (wbits) | ms    | runner-up | ms    | old entry | ms    |
/// |------------|--------------|-------|-----------|-------|-----------|-------|
/// | 2^18       | 13           | 22.2  | 14        | 24.3  | 13        |       |
/// | 2^19       | 15           | 41.3  | 13        | 41.5  | 13 (kept) |       |
/// | 2^19.81    | 15           | 65.7  | 16        | 69.2  | (13)      | 70.3  |
/// | 2^20       | 15           | 74.2  | 16        | 76.9  | 13        | 79.3  |
/// | 2^21       | 16           | 145.6 | 17        | 151.4 | 16        |       |
/// | 2^21.81    | 17           | 269.6 | 16        | 243.1*| (16)      |       |
/// | 2^22       | 16           | 273.5 | 15        | 286.4 | 14        | 313.7 |
///
/// (*) At 2^21.81 (the device share of 2^22) the two rounds disagreed
/// (16: 243.1 then 307.8; 17: 269.6 once); the 2^22 entry, 16, applies.
///
/// 2^13 (taken on a registered table with the default thresholds since the
/// table threshold became 13, see the crate text): measured on 2026-11-19 on
/// the table path, warm device, `wbits` 10 to 12 with the device shares 64,
/// 56 and 48, minimum of 15 calls (load average 4 to 6 from other
/// processes). `wbits` 11 was the best at every share (device only: 2.80
/// ms; 10: 4.56; 12: 2.98), as at 2^14.
///
/// Outside the table: below `2^13` the `2^13` entry; from `2^23` the
/// `2^22` entry, where the command-buffer bound makes every chunk at most
/// `CHUNK_ADDS / nwins` points, the chunk size measured at `2^22`. A length
/// that is not a power of two takes the entry of its nearest log2 (on the
/// log scale), so that the device share of a split MSM, 81% to 84% of a
/// power of two, takes that power's entry.
const MEASURED: [(u32, u32, usize); 10] = [
    (13, 11, 1),
    (14, 11, 1),
    (15, 11, 1),
    (16, 13, 1),
    (17, 13, 1),
    (18, 13, 1),
    (19, 13, 1),
    (20, 15, 1),
    (21, 16, 1),
    (22, 16, 1),
];

/// `(wbits, chunks)` of `MEASURED` for `n` points.
fn measured(n: usize) -> (u32, usize) {
    let (lo, hi) = (MEASURED[0].0, MEASURED[MEASURED.len() - 1].0);
    let k = lg2_nearest(n).clamp(lo, hi);
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

    /// Bytes this table holds in device memory now: its points once the
    /// first MSM on it uploaded them, 0 before that and after the table was
    /// dropped as stale. The host copy of a registration that was not
    /// uploaded yet is not counted.
    pub fn device_bytes(&self) -> u64 {
        tables()
            .as_ref()
            .and_then(|t| t.get(&self.key))
            .filter(|e| e.id == self.id)
            .and_then(|e| e.dev.as_ref())
            .map_or(0, |b| b.len_bytes() as u64)
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
// The effective size.

/// The non-zero scalars an MSM of `n` scalars needs for the device, under
/// the log2 threshold `k` of its path (`table`: a registered table serves
/// it). With the default thresholds, `SPARSE_TABLE_FIXED + n /
/// SPARSE_TABLE_DIV` on the table path (`k` is not used: the length
/// admission has applied it) and `max(2^k, n / SPARSE_ONE_SHOT_DIV)` on the
/// one-shot path;
/// with an explicit threshold (`explicit`, [`crate::set_min_k`] or
/// `MIDNIGHT_METAL_MIN_K`), `2^k` alone.
pub(crate) fn effective_need(n: usize, k: u32, table: bool, explicit: bool) -> usize {
    let min = 1usize << k.min(40);
    if explicit {
        return min;
    }
    if table {
        SPARSE_TABLE_FIXED + n / SPARSE_TABLE_DIV
    } else {
        min.max(n / SPARSE_ONE_SHOT_DIV)
    }
}

/// The sparse crossovers: the device takes an MSM of `n` scalars only from
/// `SPARSE_TABLE_FIXED + n / SPARSE_TABLE_DIV` non-zero scalars on the
/// table path and from `max(2^msm_one_shot, n / SPARSE_ONE_SHOT_DIV)` on
/// the one-shot path (with the default thresholds).
///
/// The device MSM costs the same for any number of non-zero scalars of a
/// given length (the masked scalars are dense), while the stock CPU MSM
/// skips the zero ones. Measured on Apple M4 Max (40-core GPU, 16 CPU
/// cores), 2026-11-19, release build, on `2^20` random points with `nz`
/// random non-zero scalars at spread positions, with the CPU share of the
/// split active (its share of a sparse input is cheap, so the device time
/// stays that of a dense MSM), each configuration checked against
/// `msm_pippenger` and timed as the minimum of 5 calls, two rounds
/// interleaved in one process (load average 9 to 20 from another process;
/// the rounds agreed within 4 ms). Wall time per call (ms), best round:
///
/// | non-zero        | CPU `msm_pippenger` | device, table | device, one-shot |
/// |-----------------|-------|------|-------|
/// | 2^17 (n/8)      | 42.8  | 61.3 | 102.2 |
/// | 3·2^16 (n/5.33) | 60.0  | 61.7 | 102.3 |
/// | 2^18 (n/4)      | 74.3  | 61.4 | 102.1 |
/// | 3·2^17 (n/2.67) | 109.8 | 61.3 | 102.4 |
/// | 2^19 (n/2)      | 152.0 | 61.3 | 105.4 |
/// | 3·2^18 (n/1.33) | 218.7 | 61.4 | 102.3 |
///
/// The CPU time crosses the table path's at about `n/5.3` and the one-shot
/// path's at about `n/2.9`: hence `n/5` and `n/3`. (Before the split, on a
/// busier machine, the crossovers were about `n/3.4` and `n/1.5`; the
/// device-only figures are in the report of that measurement, not here.)
///
/// Near the table threshold the device's fixed cost (the host collection,
/// the command buffer) is a larger part of its time, and the crossover is a
/// larger fraction of `n`. Measured on Apple M4 Max, 2026-11-19, release
/// build, warm device, the table path (device shares of that time: 56/64 at
/// `2^14` and `2^15`, device only at `2^13`), on random points with the
/// scalars at every index `i` with `i mod d >= m` zero (`m/d` of them
/// non-zero), minimum of 9 calls (load average 4 to 7 from other
/// processes). Wall time per call (ms):
///
/// | n    | device | CPU: n | 3n/4 | n/2 | 3n/8 | n/4 | crossover |
/// |------|--------|------|------|------|------|------|-----------|
/// | 2^13 | 2.8    | 3.4  | 2.6  | 1.9  | 1.6  | 1.3  | ~0.80 n   |
/// | 2^14 | 3.5    | 6.4  | 5.6  | 3.9  | 2.9  | 2.2  | ~0.45 n   |
/// | 2^15 | 5.2    | 12.2 | 9.7  | 7.8  | 5.4  | 3.7  | ~0.36 n   |
///
/// With `n/5` for the part that grows with the length, the rest is about
/// 4000 to 5300 non-zero scalars at the three sizes: the need on the table
/// path is `2^12 + n/5` ([`SPARSE_TABLE_FIXED`]; 0.70 n, 0.45 n, 0.33 n and
/// 0.20 n at `2^13`, `2^14`, `2^15` and `2^20`; at `2^13` an MSM with 70% to
/// 80% non-zero scalars goes to the device and is up to about 0.3 ms
/// slower there). A dense MSM at the table threshold (a column with a few
/// zero rows before its blinding rows) is taken; it was declined while the
/// need was `max(2^msm_table, n/5)`, which asks every scalar of a
/// `2^msm_table` MSM to be non-zero.
///
/// The need is a function of the length only, never of the values.
pub(crate) const SPARSE_TABLE_DIV: usize = 5;
/// See [`SPARSE_TABLE_DIV`]: the part of the table path's need that does not
/// grow with the length, in non-zero scalars.
pub(crate) const SPARSE_TABLE_FIXED: usize = 1 << 12;
/// See [`SPARSE_TABLE_DIV`].
pub(crate) const SPARSE_ONE_SHOT_DIV: usize = 3;

/// Whether an MSM over `scalars` has an effective size of at least `need`:
/// its count of non-zero scalars, taken as 1 when every scalar is zero (so a
/// threshold of `2^0` admits everything). The count is never stored or
/// returned; only this comparison leaves the function.
///
/// A sequential scan of a prefix stops as soon as `need` non-zero scalars
/// are seen (a dense input decides within about `need` scalars); otherwise
/// the rest is counted on several threads, each stopping once the shared
/// total reaches `need`.
pub(crate) fn effective_size_reaches(scalars: &[Scalar], need: usize) -> bool {
    use std::sync::atomic::AtomicUsize;
    const BLOCK: usize = 1 << 13;
    if need <= 1 {
        return true;
    }
    let nz = |s: &Scalar| (s[0] | s[1] | s[2] | s[3] != 0) as usize;
    let head = scalars.len().min(need.saturating_mul(2).max(BLOCK));
    let mut seen = 0usize;
    for b in scalars[..head].chunks(BLOCK) {
        seen += b.iter().map(nz).sum::<usize>();
        if seen >= need {
            return true;
        }
    }
    let rest = &scalars[head..];
    if seen + rest.len() < need {
        return false;
    }
    let total = AtomicUsize::new(seen);
    let threads = std::thread::available_parallelism().map_or(4, |t| t.get()).min(16);
    let per = rest.len().div_ceil(threads).max(BLOCK);
    std::thread::scope(|s| {
        for part in rest.chunks(per) {
            let total = &total;
            s.spawn(move || {
                for b in part.chunks(BLOCK) {
                    if total.load(Relaxed) >= need {
                        return;
                    }
                    total.fetch_add(b.iter().map(nz).sum::<usize>(), Relaxed);
                }
            });
        }
    });
    total.load(Relaxed) >= need
}

// ---------------------------------------------------------------------------
// Entry points.

/// Whether a registered table would serve an MSM on `bases` (the registered
/// slice or a prefix of it, by address and length; the content is compared
/// when the MSM runs).
pub(crate) fn has_table(bases: &[G1AffineLimbs]) -> bool {
    let key = (bases.as_ptr() as usize, bases.len());
    tables().as_ref().is_some_and(|t| table_for(t, key, bases.len()).is_some())
}

/// `sum scalars[i] * bases[i]`, or `None` (declined). Without a table that
/// serves `bases` (none registered, or the registered one no longer matches),
/// an MSM whose effective size is below `one_shot_min` (the one-shot path's
/// [`effective_need`]) declines `below-min-k`.
pub(crate) fn msm_g1(
    scalars: &[Scalar],
    bases: &[G1AffineLimbs],
    one_shot_min: usize,
) -> Option<G1JacobianLimbs> {
    match dispatch(scalars, bases, one_shot_min, None) {
        Ok(Some((r, ns))) => {
            account::served(Class::Msm, ns);
            Some(r)
        }
        Ok(None) => {
            account::decline(Class::Msm, Reason::BelowMinK);
            None
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
    /// Points of the device share (`n` when the CPU takes no share).
    pub device_points: u64,
    /// Wall time of the CPU share (`msm_pippenger` on the points the device
    /// does not take), run at the same time as the device share.
    pub cpu_ns: u64,
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
    let (r, _) = dispatch(scalars, bases, 0, Some(&mut st)).ok()??;
    Some((r, st))
}

/// The table or one-shot path, then the masked MSM. `Ok(None)`: no table
/// serves `bases` and the MSM's effective size is below `one_shot_min` (no
/// device work was done).
fn dispatch(
    scalars: &[Scalar],
    bases: &[G1AffineLimbs],
    one_shot_min: usize,
    mut prof: Option<&mut MsmStages>,
) -> Result<Option<(G1JacobianLimbs, u64)>, Error> {
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
            return split(
                &mut s,
                Points::Table(points),
                scalars,
                bases,
                Some(masks),
                prof,
            )
            .map(Some);
        }
    }
    drop(guard);
    if !effective_size_reaches(scalars, one_shot_min) {
        return Ok(None);
    }
    ONE_SHOT.fetch_add(1, Relaxed);
    let mut s = ctx::lock()?;
    split(&mut s, Points::Host(bases), scalars, bases, None, prof).map(Some)
}

/// The device share `m` of an MSM of `n` points on the table path or the
/// one-shot path (`MEASURED_SHARE`): the device computes the points `[0, m)`
/// and the CPU `[m, n)`. A function of `n` and the path only.
/// `MIDNIGHT_METAL_MSM_CPU_SHARE=off` gives `m = n`.
fn device_share(n: usize, table: bool) -> usize {
    if std::env::var("MIDNIGHT_METAL_MSM_CPU_SHARE").as_deref() == Ok("off") || cpu_pool().is_none()
    {
        return n;
    }
    let k = lg2_nearest(n);
    let (lo, hi) = (
        MEASURED_SHARE[0].0,
        MEASURED_SHARE[MEASURED_SHARE.len() - 1].0,
    );
    if k < lo {
        return n;
    }
    let e = MEASURED_SHARE[(k.min(hi) - lo) as usize];
    let frac = if table { e.1 } else { e.2 };
    (n * frac / SHARE_DEN).clamp(1, n)
}

/// The denominator of the shares in `MEASURED_SHARE`.
const SHARE_DEN: usize = 64;

/// The device share by size: `(log2 n, table path, one-shot path)`, in
/// 64ths of the points (64: device only).
///
/// Measured on Apple M4 Max (40-core GPU, 16 CPU cores, 128 GB), 2026-11-19,
/// release build, with the window plan above: for each size, random scalars
/// and random points, the share swept over 64ths (table path: 64 down to 40
/// or 44; one-shot path: 64 down to 36), every configuration checked against
/// `msm_pippenger` once and timed as the minimum of 3 calls, the
/// configurations of one size interleaved in one process. Two other
/// processes shared the machine (one CPU-heavy, load average 12 to 23; one
/// running GPU work), so single timings moved by up to 15%. The table path
/// was swept two or three times (run 1 before the window plan above changed
/// at 2^20 and 2^22); where the runs' best shares differed, the share between
/// them is taken. Wall time per call (ms): device only, and the best share
/// of each run with its time:
///
/// | n    | table: device only | best share (runs) | taken | one-shot: device only | best share | taken |
/// |------|-----------|-------------------------------|----|-------|----------|----|
/// | 2^13 | 3.6, 2.7  | 48: 2.42; 48: 2.41            | 64*| 5.5, 5.4 | 40: 4.7; 40: 4.9 | 40 |
/// | 2^14 | 5.7, 6.8  | 48: 6.5; 58 or 50: 5.2        | 64*| 11.0  | 40: 9.1  | 40 |
/// | 2^15 | 6.8, 7.1  | 56: 6.2; 50: 6.1              | 64*| 20.9  | 40: 13.9 | 40 |
/// | 2^16 | 9.0, 8.5  | 64; 64 (58: 8.6)              | 64 | 21.8  | 48: 17.8 | 48 |
/// | 2^17 | 14.0, 13.4| 58: 13.2; 52: 11.3; 54: 11.5  | 54 | 25.8  | 52: 22.3 | 52 |
/// | 2^18 | 24.2, 31.3| 56: 21.9; 50: 25.7           | 54 | 45.1  | 40: 35.4 | 40 |
/// | 2^19 | 42.7, 47.3| 58: 39.3; 52: 39.9           | 54 | 83.0  | 40: 67.3 | 40 |
/// | 2^20 | 80.8, 80.5| 54: 70.4; 52: 70.3           | 52 | 148.4 | 40: 116.2| 40 |
/// | 2^21 | 139.6, 161| 52: 131.9; 52: 134.5        | 52 | 287.5 | 48: 232.0| 48 |
/// | 2^22 | 314, 311  | 54: 237.4; 52: 266.7         | 52 | 558.7 | 44: 412.1| 44 |
///
/// The 2^13 row was measured on 2026-11-19 (warm device, shares 64, 56, 48,
/// 40 and 32 on both paths, two rounds of 15 calls each, minimum per round,
/// load average 4 to 6 from other processes); the stock CPU MSM took 3.4 ms.
///
/// The stock CPU MSM alone took 9.3, 15.4, 25.5, 39.2, 78, 144, 288, 586 and
/// 1175 ms (best of the runs, 2^14 to 2^22). At 2^13 to 2^15 the one-shot
/// optimum is at or near the end of the swept range (40); the one-shot path
/// is taken there only with an explicit threshold.
///
/// (*) Inside a proof the share does not pay at `2^13` to `2^15`: the CPU
/// share of the 8 MSMs of a proof keeps all CPU threads busy at every call,
/// and over successive proofs the GPU time of the same device share grows
/// (at `2^14` from 2.3 ms to 5 to 7 ms within 8 to 16 proofs, the proof
/// from 47 ms to 60 to 98 ms), while with the device alone it stays put.
/// Measured on 2026-11-19 with the rc1 chain circuit (`gpu-prover/bench/rc1`),
/// 16 warm proofs per process at `k = 14` and 6 at `k = 15` (load average 5
/// to 7 from other processes), proof time in ms: `k = 14`, share 56: 51.5
/// to 93.8, device only: 48.1 to 55.6 (median 49.2); `k = 15`, share 56:
/// 80.7 to 93.6, device only: 69.4 to 75.1. At `2^16` the table entry is
/// already 64; at `k = 18` and `k = 20` the shares of `2^18` and `2^20`
/// still pay inside the proof (`k = 18`: 261 to 279 ms with the share, 284
/// to 288 without; `k = 20`: 907 against 1000 to 1017). At `k = 13`, with
/// the table threshold at 13, a share of 48 gave 54.7 to 65.8 ms per proof
/// (8 proofs per process, two processes) and the device alone 33.6 to 36.3
/// ms (12 proofs per process, two processes).
///
/// Outside the table: below `2^13` (reached only with an explicit
/// threshold) device only; from `2^23` the `2^22` shares. A length that is
/// not a power of two takes the entry of its nearest log2, as the plan does.
const MEASURED_SHARE: [(u32, usize, usize); 10] = [
    (13, 64, 40),
    (14, 64, 40),
    (15, 64, 40),
    (16, 64, 48),
    (17, 54, 52),
    (18, 54, 40),
    (19, 54, 40),
    (20, 52, 40),
    (21, 52, 48),
    (22, 52, 44),
];

/// The thread pool of the CPU share: its own threads, so that the share
/// never waits for a worker of the caller's pool (the prover calls MSMs from
/// rayon workers, which may all be blocked on the device lock).
fn cpu_pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: std::sync::OnceLock<Option<rayon::ThreadPool>> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(std::thread::available_parallelism().map_or(4, |t| t.get()))
            .thread_name(|i| format!("midnight-metal-msm-cpu-{i}"))
            .build()
            .ok()
    })
    .as_ref()
}

const _: () = {
    assert!(size_of::<midnight_curves::Fq>() == size_of::<Scalar>());
    assert!(align_of::<midnight_curves::Fq>() == align_of::<Scalar>());
    assert!(size_of::<midnight_curves::G1Affine>() == size_of::<G1AffineLimbs>());
    assert!(align_of::<midnight_curves::G1Affine>() == align_of::<G1AffineLimbs>());
    assert!(size_of::<midnight_curves::G1Projective>() == size_of::<blst_p1>());
};

/// The CPU share: the stock `msm_pippenger` of midnight-curves, on the
/// share's own pool. The scalars and points are the caller's slices (no
/// copy is made here).
fn cpu_msm(scalars: &[Scalar], bases: &[G1AffineLimbs]) -> blst_p1 {
    use midnight_curves::{Fq, G1Affine, G1Projective, msm::msm_pippenger};
    // SAFETY: `Fq` is a transparent `blst_fr` (`[u64; 4]`), `G1Affine` a
    // transparent `blst_p1_affine` (`[u64; 12]`), `G1Projective` a transparent
    // `blst_p1`; sizes and alignments are checked above. The scalars were
    // checked reduced at admission.
    let (s, b) = unsafe {
        (
            &*(scalars as *const [Scalar] as *const [Fq]),
            &*(bases as *const [G1AffineLimbs] as *const [G1Affine]),
        )
    };
    let pool = cpu_pool().expect("the CPU share runs only with its pool");
    let r: G1Projective = pool.install(|| msm_pippenger(s, b));
    // SAFETY: as above.
    unsafe { std::mem::transmute::<G1Projective, blst_p1>(r) }
}

/// The MSM split between the device (`[0, m)`, masked) and the CPU (`[m, n)`,
/// on another thread at the same time), the two sums added on the host.
fn split(
    s: &mut Session,
    points: Points<'_>,
    scalars: &[Scalar],
    bases: &[G1AffineLimbs],
    masks: Option<&mut HashMap<usize, G1JacobianLimbs>>,
    mut prof: Option<&mut MsmStages>,
) -> Result<(G1JacobianLimbs, u64), Error> {
    let n = scalars.len();
    let m = device_share(n, matches!(points, Points::Table(_)));
    if let Some(p) = prof.as_deref_mut() {
        p.device_points += m as u64;
    }
    if m == n {
        return masked(s, points, scalars, masks, prof);
    }
    let points = match points {
        Points::Host(b) => Points::Host(&b[..m]),
        t => t,
    };
    let (dev, cpu) = std::thread::scope(|sc| {
        let h = sc.spawn(|| {
            let t0 = Instant::now();
            let r = cpu_msm(&scalars[m..], &bases[m..]);
            (r, t0.elapsed().as_nanos() as u64)
        });
        let dev = masked(s, points, &scalars[..m], masks, prof.as_deref_mut());
        (dev, h.join())
    });
    let (mut cpu, cpu_ns) = cpu.map_err(|_| Error::Device("msm: the CPU share failed".into()))?;
    if let Some(p) = prof {
        p.cpu_ns += cpu_ns;
    }
    let (d, ns) = match dev {
        Ok(r) => r,
        Err(e) => {
            wipe(std::slice::from_mut(&mut cpu));
            return Err(e);
        }
    };
    let mut sum = p1_of(&d);
    add(&mut sum, &cpu);
    let out = limbs_of(&sum);
    wipe(std::slice::from_mut(&mut sum));
    wipe(std::slice::from_mut(&mut cpu));
    Ok((out, ns))
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
    write_parallel(s.scratch(K_SCALARS, size_of_val(scalars))?, scalars);
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

/// Copies `data` to the start of `buf` on a few threads (a 2^20-scalar copy
/// takes about 0.5 ms on one).
fn write_parallel<T: ctx::Pod + Sync + Send>(buf: &mut Buffer, data: &[T]) {
    const PART: usize = 1 << 20;
    let bytes = size_of_val(data);
    if bytes <= PART {
        buf.write(data);
        return;
    }
    let per = data.len().div_ceil(4).max(PART / size_of::<T>().max(1));
    let dst = &mut buf.as_mut_slice::<T>()[..data.len()];
    std::thread::scope(|s| {
        for (d, x) in dst.chunks_mut(per).zip(data.chunks(per)) {
            s.spawn(move || d.copy_from_slice(x));
        }
    });
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
    cmd.dispatch_with(
        "msm_accumulate",
        (pl.nwins as usize) << pl.bits,
        ACC_GROUP,
        &[
            Arg::Buffer(b.buckets),
            pts,
            Arg::Buffer(b.digits),
            Arg::Buffer(b.hist),
            Arg::value(&p.acc),
        ],
        &[],
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
    // sppark adds `acc * 2^(lsbits - 1 - nthrbits)` at every step; the sum of
    // the running sums is doubled once at the end instead (the same point)
    let mut res = rec(i, 0);
    let mut acc = rec(i, 1);
    let mut raised = blst_p1::default();
    while i > 0 {
        i -= 1;
        add(&mut raised, &acc);
        add(&mut res, &rec(i, 0));
        if i != 0 {
            add(&mut acc, &rec(i, 1));
        }
    }
    for _ in 0..(lsbits - 1 - nthrbits) {
        dbl(&mut raised);
    }
    add(&mut res, &raised);
    wipe(std::slice::from_mut(&mut raised));
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
            (13, 11, 24, 1),
            (14, 11, 24, 1),
            (15, 11, 24, 1),
            (16, 13, 20, 1),
            (17, 13, 20, 1),
            (18, 13, 20, 1),
            (19, 13, 20, 1),
            (20, 15, 17, 1),
            (21, 16, 16, 1),
            (22, 16, 16, 2),
        ] {
            let p = plan(1 << k);
            assert_eq!((p.wbits, p.nwins, p.batch), (wbits, nwins, batch), "2^{k}");
            assert_eq!(p.top, NBITS - wbits * (nwins - 1));
            assert!(p.stride * p.batch >= 1 << k, "2^{k}");
        }
        // outside the table: the end entries; a non-power of two: its
        // nearest log2 (a device share of 52/64 of 2^20 takes the 2^20 entry)
        assert_eq!(plan(1).wbits, 11);
        assert_eq!(plan(1000).wbits, 11);
        assert_eq!(plan((1 << 20) * 52 / 64).wbits, 15);
        assert_eq!(plan((1 << 20) * 46 / 64).wbits, 15);
        assert_eq!(plan((1 << 20) * 45 / 64).wbits, 13);
        assert_eq!(plan((1 << 21) - 1).wbits, 16);
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
                    assert_eq!(p.wbits, 16, "{n}");
                }
            }
        }
        assert_eq!(plan(1 << 23).batch, 4);
        assert_eq!(plan(1 << 28).batch, 128);
    }

    #[test]
    fn device_share_follows_measured_table() {
        if std::env::var("MIDNIGHT_METAL_MSM_CPU_SHARE").as_deref() == Ok("off") {
            return;
        }
        for &(k, t, o) in &MEASURED_SHARE {
            let n = 1usize << k;
            assert_eq!(device_share(n, true), n * t / SHARE_DEN, "2^{k}");
            assert_eq!(device_share(n, false), n * o / SHARE_DEN, "2^{k}");
        }
        assert_eq!(device_share(1000, true), 1000);
        assert_eq!(device_share(1 << 12, false), 1 << 12);
        // a length just below a power of two takes that power's share
        assert_eq!(device_share((1 << 20) - 1, true), ((1 << 20) - 1) * 52 / 64);
        assert_eq!(device_share(1 << 24, true), (1 << 24) * 52 / 64);
    }
}
