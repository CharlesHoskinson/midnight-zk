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

//! The plan of one MSM (window width, chunks) and the device share of the
//! CPU/device split, both from the backend's [`Tuning`] tables.

use crate::{Backend, Tuning, lg2_nearest};

/// Bits of a scalar (sppark `scalar_t::bit_length()`).
pub const NBITS: u32 = 255;
/// Bytes of the temporary sort buffer one pass may use.
const SORT_TEMP_BYTES: usize = 64 << 20;

/// The window plan of an MSM of `n` points.
#[derive(Clone, Copy, Debug)]
pub struct Plan {
    pub n: usize,
    pub wbits: u32,
    pub nwins: u32,
    /// Chunks (command buffers).
    pub batch: usize,
    /// Points per chunk (a multiple of 32).
    pub stride: usize,
    /// `wbits - 1`: bucket index bits.
    pub bits: u32,
    /// Bits of the upper sort (`bits / 2`).
    pub top_bits: u32,
    /// Bits of the lower sort (`bits - top_bits`).
    pub low_bits: u32,
    /// Bits of the top window (sppark `top`).
    pub top: u32,
    /// Windows per sort pass.
    pub pass: u32,
}

/// sppark `lg2`: floor of log2, 0 for 0.
pub const fn lg2(mut n: usize) -> u32 {
    let mut r = 0;
    while n > 1 {
        n >>= 1;
        r += 1;
    }
    r
}

/// `(wbits, chunks)` of [`Tuning::msm_windows`] for `n` points.
fn measured(n: usize, t: &Tuning) -> (u32, usize) {
    let w = t.msm_windows;
    let (lo, hi) = (w[0].0, w[w.len() - 1].0);
    let k = lg2_nearest(n).clamp(lo, hi);
    let e = w[(k - lo) as usize];
    (e.1, e.2)
}

/// The plan of an MSM of `n` points.
pub fn plan(n: usize, t: &Tuning) -> Plan {
    let (wbits, batch) = measured(n, t);
    let nwins = (NBITS - 1) / wbits + 1;
    // cap the GPU time of one command buffer (sppark has no such bound)
    let max_stride = (t.msm_chunk_adds / nwins as usize) & !31;
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

/// The denominator of the shares in [`Tuning::msm_shares`].
pub const SHARE_DEN: usize = 64;

/// Current aggregate CPU frequency cap as a percentage of hardware peak
/// (`1..=100`), read from Linux/Android `cpufreq` sysfs. Independent of any
/// witness or proof data. Returns `100` on other hosts or when sysfs is
/// unavailable.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn cpu_capacity_pct() -> u32 {
    use std::sync::OnceLock;

    struct CpuFreq {
        cpus: Vec<(usize, u64)>,
        total_max: u64,
    }

    fn read_u64(path: &str) -> Option<u64> {
        std::fs::read_to_string(path).ok()?.trim().parse().ok()
    }

    static INFO: OnceLock<Option<CpuFreq>> = OnceLock::new();
    let Some(info) = INFO.get_or_init(|| {
        let mut cpus = Vec::new();
        let mut total_max = 0u64;
        for i in 0..256 {
            let max_path = format!("/sys/devices/system/cpu/cpu{i}/cpufreq/cpuinfo_max_freq");
            let Some(max_f) = read_u64(&max_path).filter(|&v| v > 0) else {
                if i >= 16 && cpus.is_empty() {
                    break;
                }
                continue;
            };
            let cur_path = format!("/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_max_freq");
            if read_u64(&cur_path).is_some() {
                cpus.push((i, max_f));
                total_max = total_max.saturating_add(max_f);
            }
        }
        (total_max > 0).then_some(CpuFreq { cpus, total_max })
    }) else {
        return 100;
    };

    let mut total_cur = 0u64;
    for &(i, max_f) in &info.cpus {
        let cur_path = format!("/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_max_freq");
        let Some(cur) = read_u64(&cur_path) else {
            return 100;
        };
        total_cur = total_cur.saturating_add(cur.min(max_f));
    }
    ((total_cur * 100) / info.total_max).clamp(1, 100) as u32
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn cpu_capacity_pct() -> u32 {
    100
}

/// Adjusts a tuned device share (in `SHARE_DEN`-ths) when CPU clocks are
/// thermally capped while the GPU continues at full speed. Uses a single
/// threshold (`< 70%` of peak CPU clock capacity) so cool runs are untouched
/// and the mask cache (`MSM(r)` keyed by `m`) stores at most one extra entry.
fn adjust_share(frac: usize, cpu_pct: u32) -> usize {
    if frac >= SHARE_DEN || cpu_pct >= 70 {
        return frac;
    }
    const THROTTLED_RATIO: usize = 58;
    let num = frac * 100 * SHARE_DEN;
    let den = frac * 100 + THROTTLED_RATIO * (SHARE_DEN - frac);
    ((num + den / 2) / den).clamp(frac, SHARE_DEN - 4)
}

/// The device share `m` of an MSM of `n` points: the device computes `[0, m)`
/// and the CPU `[m, n)`. `{ENV_PREFIX}_MSM_CPU_SHARE=off` or no CPU pool gives
/// `m = n`.
pub fn device_share<B: Backend>(n: usize, table: bool) -> usize {
    if std::env::var(format!("{}_MSM_CPU_SHARE", B::ENV_PREFIX)).as_deref() == Ok("off")
        || super::par::cpu_pool::<B>().is_none()
    {
        return n;
    }
    let k = lg2_nearest(n);
    let shares = B::tuning().msm_shares;
    let (lo, hi) = (shares[0].0, shares[shares.len() - 1].0);
    if k < lo {
        return n;
    }
    // in range: `lo <= k.min(hi) <= hi`, one entry per log2
    let e = shares[(k.min(hi) - lo) as usize];
    let frac = adjust_share(if table { e.1 } else { e.2 }, cpu_capacity_pct());
    (n * frac / SHARE_DEN).clamp(1, n)
}

#[cfg(test)]
mod tests {
    use super::{SHARE_DEN, adjust_share};

    #[test]
    fn adjust_share_preserves_cool_and_full_gpu_and_shifts_when_throttled() {
        assert_eq!(adjust_share(37, 100), 37);
        assert_eq!(adjust_share(37, 70), 37);
        assert_eq!(adjust_share(SHARE_DEN, 50), SHARE_DEN);
        assert_eq!(adjust_share(37, 54), 45);
        assert_eq!(adjust_share(39, 54), 47);
    }
}
