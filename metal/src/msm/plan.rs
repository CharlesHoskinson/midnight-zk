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

//! The plan of one MSM (sppark's `msm_t` constructor and `invoke` choose it
//! by formula; here a measured table) and the CPU share of the split.
//!
//! Both tables were measured on Apple M4 Max (40-core GPU, 16 CPU cores, 128
//! GB, macOS), release build, over random scalars and random points, each
//! configuration checked against `msm_pippenger`. Other processes shared the
//! machine during the measurements; the loads are given with each table.

use crate::lg2_nearest;

/// Bits of a scalar (sppark `scalar_t::bit_length()`).
pub(super) const NBITS: u32 = 255;
/// The most window digits (bucket additions) of one chunk, so that a command
/// buffer stays near or below 0.5 s of GPU time: measured on Apple M4 Max at
/// 5 to 10 ns per mixed addition, sort and integrate included.
pub(super) const CHUNK_ADDS: usize = 1 << 25;
/// Bytes of the temporary sort buffer one pass may use.
const SORT_TEMP_BYTES: usize = 64 << 20;

/// The window plan of an MSM of `n` points.
#[derive(Clone, Copy, Debug)]
pub(super) struct Plan {
    pub(super) n: usize,
    pub(super) wbits: u32,
    pub(super) nwins: u32,
    /// Chunks (command buffers).
    pub(super) batch: usize,
    /// Points per chunk (a multiple of 32).
    pub(super) stride: usize,
    /// `wbits - 1`: bucket index bits.
    pub(super) bits: u32,
    /// Bits of the upper sort (`bits / 2`).
    pub(super) top_bits: u32,
    /// Bits of the lower sort (`bits - top_bits`).
    pub(super) low_bits: u32,
    /// Bits of the top window (sppark `top`).
    pub(super) top: u32,
    /// Windows per sort pass.
    pub(super) pass: u32,
}

/// sppark `lg2`: floor of log2, 0 for 0.
pub(super) const fn lg2(mut n: usize) -> u32 {
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
/// Measured on the table path with the CPU share off, `wbits` in 10..=17
/// around each entry and chunk counts {1, 2, 4}: the minimum wall time per
/// call (GPU time plus the host collection) over 3 to 15 calls, sizes with
/// close candidates repeated two or three times and the minimum over the
/// runs taken (other processes shared the machine, one of them running GPU
/// work at times). The rows 2^14 to 2^17 were measured with an accumulation
/// kernel that inlined the field multiplication and did not order the
/// buckets by length, which is slower; the other rows with the kernel of
/// `shaders/msm.metal`:
///
/// | n       | entry (wbits) | ms    | runner-up | ms     |
/// |---------|---------------|-------|-----------|--------|
/// | 2^13    | 11            | 2.80  | 12        | 2.98   |
/// | 2^14    | 11            | 6.75  | 10        | 7.34   |
/// | 2^15    | 11            | 9.57  | 12        | 9.10*  |
/// | 2^16    | 13            | 13.57 | 12        | 14.12  |
/// | 2^17    | 13            | 21.47 | 12        | 23.13  |
/// | 2^18    | 13            | 22.2  | 14        | 24.3   |
/// | 2^19    | 13            | 41.5  | 15        | 41.3*  |
/// | 2^19.81 | (20: 15)      | 65.7  | 16        | 69.2   |
/// | 2^20    | 15            | 74.2  | 16        | 76.9   |
/// | 2^21    | 16            | 145.6 | 17        | 151.4  |
/// | 2^21.81 | (22: 16)      | 243.1 | 17        | 269.6* |
/// | 2^22    | 16            | 273.5 | 15        | 286.4  |
///
/// (*) Within the spread of the runs: at 2^15 one run favoured 12 by 0.5 ms
/// and two runs and the one-shot path favoured 11; at 2^19 13 and 15 are
/// within 0.5%; at 2^21.81 (the device share of 2^22) the runs disagreed
/// (16: 243.1 and 307.8; 17: 269.6).
///
/// Outside the table: below `2^13` the `2^13` entry; from `2^23` the `2^22`
/// entry, where the command-buffer bound makes every chunk at most
/// `CHUNK_ADDS / nwins` points. A length that is not a power of two takes
/// the entry of its nearest log2 ([`lg2_nearest`]), so that the device share
/// of a split MSM (2^19.81 and 2^21.81 above) takes that power's entry.
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

/// `(wbits, chunks)` of [`MEASURED`] for `n` points (the index is in range:
/// `k` is clamped to the table's first and last log2, and the table has one
/// entry per log2).
fn measured(n: usize) -> (u32, usize) {
    let (lo, hi) = (MEASURED[0].0, MEASURED[MEASURED.len() - 1].0);
    let k = lg2_nearest(n).clamp(lo, hi);
    let e = MEASURED[(k - lo) as usize];
    (e.1, e.2)
}

/// The plan of an MSM of `n` points.
pub(super) fn plan(n: usize) -> Plan {
    let (wbits, batch) = measured(n);
    let nwins = (NBITS - 1) / wbits + 1;
    // the time budget of one command buffer (this port; sppark has no such
    // bound)
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

/// The denominator of the shares in [`MEASURED_SHARE`].
const SHARE_DEN: usize = 64;

/// The device share by size: `(log2 n, table path, one-shot path)`, in 64ths
/// of the points (64: device only).
///
/// Measured with the window plan above: for each size the share swept over
/// 64ths (table path: 64 down to 40 or 44; one-shot path: 64 down to 36;
/// 2^13: 64, 56, 48, 40 and 32 on both paths), each configuration timed as
/// the minimum of 3 calls (15 at 2^13), the configurations of one size
/// interleaved in one process. Other processes shared the machine (one
/// CPU-heavy, load average 4 to 23; one running GPU work), so single timings
/// moved by up to 15%. The table path was swept two or three times (the
/// first sweep with the plan entries 13 at 2^20 and 14 at 2^22); where the
/// sweeps' best shares differed, the share between them is taken. Wall
/// time per call (ms): device only, and the best share of each sweep with
/// its time:
///
/// | n    | table: device only | best share (sweeps)           | taken | one-shot: device only | best share | taken |
/// |------|-----------|-------------------------------|----|-------|----------|----|
/// | 2^13 | 3.6, 2.7  | 48: 2.42; 48: 2.41            | 64*| 5.5, 5.4 | 40: 4.7; 40: 4.9 | 40 |
/// | 2^14 | 5.7, 6.8  | 48: 6.5; 58 or 50: 5.2        | 64*| 11.0  | 40: 9.1  | 40 |
/// | 2^15 | 6.8, 7.1  | 56: 6.2; 50: 6.1              | 64*| 20.9  | 40: 13.9 | 40 |
/// | 2^16 | 9.0, 8.5  | 64; 64 (58: 8.6)              | 64 | 21.8  | 48: 17.8 | 48 |
/// | 2^17 | 14.0, 13.4| 58: 13.2; 52: 11.3; 54: 11.5  | 54 | 25.8  | 52: 22.3 | 52 |
/// | 2^18 | 24.2, 31.3| 56: 21.9; 50: 25.7            | 54 | 45.1  | 40: 35.4 | 40 |
/// | 2^19 | 42.7, 47.3| 58: 39.3; 52: 39.9            | 54 | 83.0  | 40: 67.3 | 40 |
/// | 2^20 | 80.8, 80.5| 54: 70.4; 52: 70.3            | 52 | 148.4 | 40: 116.2| 40 |
/// | 2^21 | 139.6, 161| 52: 131.9; 52: 134.5          | 52 | 287.5 | 48: 232.0| 48 |
/// | 2^22 | 314, 311  | 54: 237.4; 52: 266.7          | 52 | 558.7 | 44: 412.1| 44 |
///
/// The stock CPU MSM alone took 3.4, 9.3, 15.4, 25.5, 39.2, 78, 144, 288,
/// 586 and 1175 ms (2^13 to 2^22). At 2^13 to 2^15 the one-shot optimum is
/// at or near the end of the swept range (40).
///
/// (*) Inside a proof the share does not pay at `2^13` to `2^15`: the CPU
/// share of the MSMs of a proof keeps all CPU threads busy at every call,
/// and over successive proofs the GPU time of the same device share grows
/// (at `2^14` from 2.3 ms to 5 to 7 ms within 8 to 16 proofs), while with
/// the device alone it stays put. Measured with the rc1 chain circuit
/// (`gpu-prover/bench/rc1`), warm proofs (load average 5 to 7 from other
/// processes), proof time in ms: `k = 13`, share 48: 54.7 to 65.8, device
/// only: 33.6 to 36.3; `k = 14`, share 56: 51.5 to 93.8, device only: 48.1
/// to 55.6 (median 49.2); `k = 15`, share 56: 80.7 to 93.6, device only:
/// 69.4 to 75.1. The shares of `2^18` and `2^20` still pay inside the proof
/// (`k = 18`: 261 to 279 ms with the share, 284 to 288 without; `k = 20`:
/// 907 against 1000 to 1017).
///
/// Outside the table: below `2^13` (reached only with an explicit threshold)
/// device only; from `2^23` the `2^22` shares. A length that is not a power
/// of two takes the entry of its nearest log2, as the plan does.
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

/// The device share `m` of an MSM of `n` points on the table path (`table`)
/// or the one-shot path ([`MEASURED_SHARE`]): the device computes the points
/// `[0, m)` and the CPU `[m, n)`. A function of `n` and the path only.
/// `MIDNIGHT_METAL_MSM_CPU_SHARE=off`, or no CPU pool, gives `m = n`.
pub(super) fn device_share(n: usize, table: bool) -> usize {
    if std::env::var("MIDNIGHT_METAL_MSM_CPU_SHARE").as_deref() == Ok("off")
        || super::split::cpu_pool().is_none()
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
    // in range: `lo <= k.min(hi) <= hi`, one entry per log2
    let e = MEASURED_SHARE[(k.min(hi) - lo) as usize];
    let frac = if table { e.1 } else { e.2 };
    (n * frac / SHARE_DEN).clamp(1, n)
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
