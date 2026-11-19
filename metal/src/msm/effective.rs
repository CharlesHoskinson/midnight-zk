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

//! The effective size of an MSM ("Effective size" in the module text): the
//! need of each path and the comparison of the count of non-zero scalars
//! with it.

use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use super::par::{host_threads, par_parts};
use crate::Scalar;

/// The sparse crossover: with the default thresholds the device takes an MSM
/// of `n` scalars on a registered table only from `SPARSE_TABLE_FIXED + n /
/// SPARSE_TABLE_DIV` non-zero scalars.
///
/// The device MSM costs the same for any number of non-zero scalars of a
/// given length (the masked scalars are dense), while the stock CPU MSM skips
/// the zero ones. Measured on Apple M4 Max (40-core GPU, 16 CPU cores),
/// release build, on `2^20` random points with `nz` random non-zero scalars
/// at spread positions, the table path with the CPU share of the split
/// active, each configuration checked against `msm_pippenger` and timed as
/// the minimum of 5 calls, two rounds interleaved in one process (load
/// average 9 to 20 from another process; the rounds agreed within 4 ms).
/// Wall time per call (ms), best round:
///
/// | non-zero        | CPU `msm_pippenger` | device, table |
/// |-----------------|-------|------|
/// | 2^17 (n/8)      | 42.8  | 61.3 |
/// | 3·2^16 (n/5.33) | 60.0  | 61.7 |
/// | 2^18 (n/4)      | 74.3  | 61.4 |
/// | 3·2^17 (n/2.67) | 109.8 | 61.3 |
/// | 2^19 (n/2)      | 152.0 | 61.3 |
/// | 3·2^18 (n/1.33) | 218.7 | 61.4 |
///
/// The CPU time crosses the device's at about `n/5.3`: hence `n/5`.
///
/// Near the table threshold the device's fixed cost (the host collection,
/// the command buffer) is a larger part of its time, and the crossover is a
/// larger fraction of `n`. Measured on the same machine, warm device, the
/// table path (device shares of that time: 56/64 at `2^14` and `2^15`,
/// device only at `2^13`), on random points with the scalars at every index
/// `i` with `i mod d >= m` zero (`m/d` of them non-zero), minimum of 9 calls
/// (load average 4 to 7 from other processes). Wall time per call (ms):
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
/// 80% non-zero scalars goes to the device and is up to about 0.3 ms slower
/// there). A dense MSM at the table threshold with a few zero scalars (a
/// column with zero rows before its blinding rows) is taken.
///
/// The need is a function of the length only, never of the values.
const SPARSE_TABLE_DIV: usize = 5;
/// See [`SPARSE_TABLE_DIV`]: the part of the table path's need that does not
/// grow with the length, in non-zero scalars.
const SPARSE_TABLE_FIXED: usize = 1 << 12;

/// The non-zero scalars an MSM of `n` scalars needs for the device, under
/// the log2 threshold `k` of its path (`table`: a registered table serves
/// it). With the default thresholds, `SPARSE_TABLE_FIXED + n /
/// SPARSE_TABLE_DIV` on the table path (`k` is not used: the length
/// admission has applied it); with an explicit threshold (`explicit`,
/// [`crate::set_min_k`] or `MIDNIGHT_METAL_MIN_K`), `2^k` alone, on either
/// path. The one-shot path is reached only with an explicit threshold (the
/// defaults decline it `unregistered-bases`), so it has no default need.
pub(crate) fn effective_need(n: usize, k: u32, table: bool, explicit: bool) -> usize {
    if table && !explicit {
        SPARSE_TABLE_FIXED + n / SPARSE_TABLE_DIV
    } else {
        1usize << k.min(40)
    }
}

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
    // a part that panicked leaves the count short: the call then declines
    // `below-min-k` (the stock CPU code runs)
    let _ = par_parts(rest.len(), host_threads(), BLOCK, |lo, hi| {
        for b in rest[lo..hi].chunks(BLOCK) {
            if total.load(Relaxed) >= need {
                return;
            }
            total.fetch_add(b.iter().map(nz).sum::<usize>(), Relaxed);
        }
    });
    total.load(Relaxed) >= need
}
