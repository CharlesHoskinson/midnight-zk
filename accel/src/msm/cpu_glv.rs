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

//! BLS12-381 G1 GLV batched-affine Pippenger MSM for the CPU share.
//!
//! Every base entering [`msm_glv`] is already verified to lie in the
//! prime-order subgroup `G1` (guaranteed by table registration or the one-shot
//! subgroup check), so the endomorphism `phi(x, y) = (beta * x, y) = lambda * P`
//! with `lambda = -u^2 mod r` (`u = -0xd201000000010000`) holds unconditionally.
//! Since `r = u^4 - u^2 + 1`, dividing any `s in [0, r)` by `u^2` gives exact
//! 128-bit unsigned half-scalars `s = k1 - k2 * lambda (mod r)` with
//! `0 <= k1, k2 < u^2 < 2^128`, so `s * P = k1 * P + k2 * (beta * x, -y)`.
//!
//! Splitting 255-bit scalars into 128-bit half-scalars doubles the effective
//! point count per window, allowing wider windows (`c = 13..14`, 9–10 windows
//! vs 22 without GLV) and ~17% fewer batched-affine curve additions, using
//! `blst`'s assembly field arithmetic directly.

use std::sync::{
    OnceLock,
    atomic::{AtomicUsize, Ordering},
};

use blst::{
    blst_fp, blst_fp_eucl_inverse, blst_fp_mul, blst_fp_sqr, blst_fr, blst_p1,
    blst_p1_add_or_double, blst_p1_double, blst_scalar, blst_scalar_from_fr,
};
#[cfg(not(target_arch = "aarch64"))]
use blst::{blst_fp_add, blst_fp_cneg, blst_fp_sub};
use rayon::prelude::*;

use super::secret::{ScrubOnExit, Secret, SecretVec, wipe, wipe_slice};
use crate::{G1AffineLimbs, Scalar};

/// BLS12-381 base-field modulus `p` as 6 little-endian 64-bit limbs.
#[cfg(target_arch = "aarch64")]
const FP_P: [u64; 6] = [
    0xb9fe_ffff_ffff_aaab,
    0x1eab_fffe_b153_ffff,
    0x6730_d2a0_f6b0_f624,
    0x6477_4b84_f385_12bf,
    0x4b1b_a7b6_434b_acd7,
    0x1a01_11ea_397f_e69a,
];

/// `u^2 = (-0xd201000000010000)^2`, exact 128-bit GLV divisor (`r = u^4 - u^2 + 1`).
const U2: u128 = 0xac45_a401_0001_a402_0000_0001_0000_0000;
/// `U2 / 2 < 0.673 * 2^127`, upper bound on signed GLV half-scalar magnitudes.
const HALF_U2: u128 = U2 >> 1;
/// Sign bit stored in bit 127 of each encoded half-scalar (`|k| <= HALF_U2 < 2^127`).
const SIGN_BIT: u128 = 1u128 << 127;
/// Low 128 bits of `floor(2^256 / U2) = 2^128 + MU_LO`.
const MU_LO: u128 = 0x7c6b_ecf1_e01f_aadd_63f6_e522_f6cf_ee2e;

/// Montgomery `1` in BLS12-381 `Fp`: `2^384 mod p`.
const FP_ONE: blst_fp = blst_fp {
    l: [
        0x7609_0000_0002_fffd,
        0xebf4_000b_c40c_0002,
        0x5f48_9857_53c7_58ba,
        0x77ce_5853_7052_5745,
        0x5c07_1a97_a256_ec6d,
        0x15f6_5ec3_fa80_e493,
    ],
};

/// Cube root of unity `beta` in Montgomery form (`beta * 2^384 mod p`) such
/// that `lambda * (x, y) = (beta * x, y)` for `lambda = -u^2 mod r` on G1.
const BETA_MONT: blst_fp = blst_fp {
    l: [
        0x30f1_361b_798a_64e8,
        0xf3b8_ddab_7ece_5a2a,
        0x16a8_ca3a_c615_77f7,
        0xc26a_2ff8_74fd_029b,
        0x3636_b766_6070_1c6e,
        0x051b_a4ab_241b_6160,
    ],
};

const NBITS: usize = 127;
const LANES: usize = 256;

#[inline(always)]
fn fp_zero() -> blst_fp {
    blst_fp { l: [0u64; 6] }
}

#[inline(always)]
fn fp_is_zero(a: &blst_fp) -> bool {
    (a.l[0] | a.l[1] | a.l[2] | a.l[3] | a.l[4] | a.l[5]) == 0
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn fp_add_val(a: &blst_fp, b: &blst_fp) -> blst_fp {
    let (mut r0, mut r1, mut r2, mut r3, mut r4, mut r5): (u64, u64, u64, u64, u64, u64);
    // SAFETY: pure register arithmetic + read-only loads from `a`, `b`, `FP_P`.
    unsafe {
        std::arch::asm!(
            "ldp {s0}, {s1}, [{a}, #0]",
            "ldp {t0}, {t1}, [{b}, #0]",
            "adds {s0}, {s0}, {t0}",
            "adcs {s1}, {s1}, {t1}",
            "ldp {s2}, {s3}, [{a}, #16]",
            "ldp {t0}, {t1}, [{b}, #16]",
            "adcs {s2}, {s2}, {t0}",
            "adcs {s3}, {s3}, {t1}",
            "ldp {s4}, {s5}, [{a}, #32]",
            "ldp {t0}, {t1}, [{b}, #32]",
            "adcs {s4}, {s4}, {t0}",
            "adc  {s5}, {s5}, {t1}",
            "ldp {t0}, {t1}, [{p}, #0]",
            "subs {d0}, {s0}, {t0}",
            "sbcs {d1}, {s1}, {t1}",
            "ldp {t0}, {t1}, [{p}, #16]",
            "sbcs {d2}, {s2}, {t0}",
            "sbcs {d3}, {s3}, {t1}",
            "ldp {t0}, {t1}, [{p}, #32]",
            "sbcs {d4}, {s4}, {t0}",
            "sbcs {d5}, {s5}, {t1}",
            "csel {s0}, {s0}, {d0}, cc",
            "csel {s1}, {s1}, {d1}, cc",
            "csel {s2}, {s2}, {d2}, cc",
            "csel {s3}, {s3}, {d3}, cc",
            "csel {s4}, {s4}, {d4}, cc",
            "csel {s5}, {s5}, {d5}, cc",
            a = in(reg) a.l.as_ptr(),
            b = in(reg) b.l.as_ptr(),
            p = in(reg) FP_P.as_ptr(),
            s0 = out(reg) r0,
            s1 = out(reg) r1,
            s2 = out(reg) r2,
            s3 = out(reg) r3,
            s4 = out(reg) r4,
            s5 = out(reg) r5,
            d0 = out(reg) _,
            d1 = out(reg) _,
            d2 = out(reg) _,
            d3 = out(reg) _,
            d4 = out(reg) _,
            d5 = out(reg) _,
            t0 = out(reg) _,
            t1 = out(reg) _,
            options(nostack, readonly),
        );
    }
    blst_fp {
        l: [r0, r1, r2, r3, r4, r5],
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn fp_sub_val(a: &blst_fp, b: &blst_fp) -> blst_fp {
    let (mut r0, mut r1, mut r2, mut r3, mut r4, mut r5): (u64, u64, u64, u64, u64, u64);
    // SAFETY: pure register arithmetic + read-only loads from `a`, `b`, `FP_P`.
    unsafe {
        std::arch::asm!(
            "ldp {s0}, {s1}, [{a}, #0]",
            "ldp {t0}, {t1}, [{b}, #0]",
            "subs {s0}, {s0}, {t0}",
            "sbcs {s1}, {s1}, {t1}",
            "ldp {s2}, {s3}, [{a}, #16]",
            "ldp {t0}, {t1}, [{b}, #16]",
            "sbcs {s2}, {s2}, {t0}",
            "sbcs {s3}, {s3}, {t1}",
            "ldp {s4}, {s5}, [{a}, #32]",
            "ldp {t0}, {t1}, [{b}, #32]",
            "sbcs {s4}, {s4}, {t0}",
            "sbcs {s5}, {s5}, {t1}",
            "csetm {m}, cc",
            "ldp {t0}, {t1}, [{p}, #0]",
            "and {t0}, {t0}, {m}",
            "and {t1}, {t1}, {m}",
            "adds {s0}, {s0}, {t0}",
            "adcs {s1}, {s1}, {t1}",
            "ldp {t0}, {t1}, [{p}, #16]",
            "and {t0}, {t0}, {m}",
            "and {t1}, {t1}, {m}",
            "adcs {s2}, {s2}, {t0}",
            "adcs {s3}, {s3}, {t1}",
            "ldp {t0}, {t1}, [{p}, #32]",
            "and {t0}, {t0}, {m}",
            "and {t1}, {t1}, {m}",
            "adcs {s4}, {s4}, {t0}",
            "adc  {s5}, {s5}, {t1}",
            a = in(reg) a.l.as_ptr(),
            b = in(reg) b.l.as_ptr(),
            p = in(reg) FP_P.as_ptr(),
            s0 = out(reg) r0,
            s1 = out(reg) r1,
            s2 = out(reg) r2,
            s3 = out(reg) r3,
            s4 = out(reg) r4,
            s5 = out(reg) r5,
            t0 = out(reg) _,
            t1 = out(reg) _,
            m = out(reg) _,
            options(nostack, readonly),
        );
    }
    blst_fp {
        l: [r0, r1, r2, r3, r4, r5],
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn fp_neg_val(a: &blst_fp) -> blst_fp {
    if fp_is_zero(a) {
        return fp_zero();
    }
    let (mut r0, mut r1, mut r2, mut r3, mut r4, mut r5): (u64, u64, u64, u64, u64, u64);
    // SAFETY: pure register arithmetic + read-only loads from `a` and `FP_P`.
    unsafe {
        std::arch::asm!(
            "ldp {s0}, {s1}, [{p}, #0]",
            "ldp {t0}, {t1}, [{a}, #0]",
            "subs {s0}, {s0}, {t0}",
            "sbcs {s1}, {s1}, {t1}",
            "ldp {s2}, {s3}, [{p}, #16]",
            "ldp {t0}, {t1}, [{a}, #16]",
            "sbcs {s2}, {s2}, {t0}",
            "sbcs {s3}, {s3}, {t1}",
            "ldp {s4}, {s5}, [{p}, #32]",
            "ldp {t0}, {t1}, [{a}, #32]",
            "sbcs {s4}, {s4}, {t0}",
            "sbc  {s5}, {s5}, {t1}",
            a = in(reg) a.l.as_ptr(),
            p = in(reg) FP_P.as_ptr(),
            s0 = out(reg) r0,
            s1 = out(reg) r1,
            s2 = out(reg) r2,
            s3 = out(reg) r3,
            s4 = out(reg) r4,
            s5 = out(reg) r5,
            t0 = out(reg) _,
            t1 = out(reg) _,
            options(nostack, readonly),
        );
    }
    blst_fp {
        l: [r0, r1, r2, r3, r4, r5],
    }
}

#[inline(always)]
fn fp_add(ret: &mut blst_fp, a: &blst_fp, b: &blst_fp) {
    #[cfg(target_arch = "aarch64")]
    {
        *ret = fp_add_val(a, b);
    }
    #[cfg(not(target_arch = "aarch64"))]
    // SAFETY: valid blst_fp pointers.
    unsafe {
        blst_fp_add(ret, a, b)
    };
}

#[inline(always)]
fn fp_add_assign(a: &mut blst_fp, b: &blst_fp) {
    #[cfg(target_arch = "aarch64")]
    {
        *a = fp_add_val(a, b);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let p = a as *mut blst_fp;
        // SAFETY: blst_fp_add supports in-place aliasing (`ret == a`).
        unsafe { blst_fp_add(p, p, b) };
    }
}

#[inline(always)]
fn fp_sub(ret: &mut blst_fp, a: &blst_fp, b: &blst_fp) {
    #[cfg(target_arch = "aarch64")]
    {
        *ret = fp_sub_val(a, b);
    }
    #[cfg(not(target_arch = "aarch64"))]
    unsafe {
        blst_fp_sub(ret, a, b)
    };
}

#[inline(always)]
fn fp_sub_assign(a: &mut blst_fp, b: &blst_fp) {
    #[cfg(target_arch = "aarch64")]
    {
        *a = fp_sub_val(a, b);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let p = a as *mut blst_fp;
        // SAFETY: blst_fp_sub supports in-place aliasing (`ret == a`).
        unsafe { blst_fp_sub(p, p, b) };
    }
}

#[inline(always)]
fn fp_rsub_assign(a: &mut blst_fp, b: &blst_fp) {
    #[cfg(target_arch = "aarch64")]
    {
        *a = fp_sub_val(b, a);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let p = a as *mut blst_fp;
        // SAFETY: blst_fp_sub supports in-place aliasing (`ret == b`).
        unsafe { blst_fp_sub(p, b, p) };
    }
}

#[inline(always)]
fn fp_mul(ret: &mut blst_fp, a: &blst_fp, b: &blst_fp) {
    unsafe { blst_fp_mul(ret, a, b) };
}

#[inline(always)]
fn fp_mul_assign(a: &mut blst_fp, b: &blst_fp) {
    let p = a as *mut blst_fp;
    // SAFETY: blst_fp_mul supports in-place aliasing (`ret == a`).
    unsafe { blst_fp_mul(p, p, b) };
}

#[inline(always)]
fn fp_sqr(ret: &mut blst_fp, a: &blst_fp) {
    unsafe { blst_fp_sqr(ret, a) };
}

#[inline(always)]
fn fp_cneg(ret: &mut blst_fp, a: &blst_fp, flag: bool) {
    #[cfg(target_arch = "aarch64")]
    {
        *ret = if flag { fp_neg_val(a) } else { *a };
    }
    #[cfg(not(target_arch = "aarch64"))]
    unsafe {
        blst_fp_cneg(ret, a, flag)
    };
}

#[inline(always)]
fn fp_inv(ret: &mut blst_fp, a: &blst_fp) {
    unsafe { blst_fp_eucl_inverse(ret, a) };
}

#[inline(always)]
fn mul_u128_wide(a: u128, b: u128) -> (u128, u128) {
    let (a0, a1) = (a as u64 as u128, a >> 64);
    let (b0, b1) = (b as u64 as u128, b >> 64);
    let ll = a0 * b0;
    let lh = a0 * b1;
    let hl = a1 * b0;
    let hh = a1 * b1;
    let (mid, c1) = lh.overflowing_add(hl);
    let (lo, c2) = ll.overflowing_add(mid << 64);
    let hi = hh + (mid >> 64) + ((c1 as u128) << 64) + (c2 as u128);
    (lo, hi)
}

/// Decomposes Montgomery scalar `s` into signed 127-bit half-scalars
/// `(k1 | (neg1 << 127), k2 | (neg2 << 127))` with `0 <= k1, k2 <= U2 / 2 < 2^127`
/// such that `s * P = (±k1) * P + (±k2) * phi(P)`, where `phi(x, y) = (beta * x, y)`.
#[inline(always)]
fn decompose(s: &Scalar) -> (u128, u128) {
    let mut sc = blst_scalar::default();
    // SAFETY: `Scalar` is `[u64; 4]`, identical to `blst_fr`.
    unsafe { blst_scalar_from_fr(&mut sc, &blst_fr { l: *s }) };
    let s_lo = u128::from_le_bytes(sc.b[..16].try_into().unwrap());
    let s_hi = u128::from_le_bytes(sc.b[16..].try_into().unwrap());
    let (_, q_add) = mul_u128_wide(s_hi, MU_LO);
    let mut q = s_hi + q_add;
    let (prod_lo, prod_hi) = mul_u128_wide(q, U2);
    let (mut rem, borrow) = s_lo.overflowing_sub(prod_lo);
    let mut rem_hi = s_hi.wrapping_sub(prod_hi).wrapping_sub(borrow as u128);
    while rem_hi != 0 || rem >= U2 {
        let (r2, b) = rem.overflowing_sub(U2);
        rem = r2;
        rem_hi = rem_hi.wrapping_sub(b as u128);
        q += 1;
    }
    // Center `rem` into [-HALF_U2, HALF_U2] using (U2, -1):
    // `s = rem + q * U2 = (±k1) + q' * U2`, with `0 <= q' <= U2 - 1` since
    // `r = U2 * (U2 - 1) + 1` implies `rem == 0` whenever `q == U2 - 1`.
    let (mut k1, mut neg1) = if rem > HALF_U2 {
        q += 1;
        (U2 - rem, true)
    } else {
        (rem, false)
    };
    // Center `q` into [-HALF_U2, HALF_U2] using `r = U2 * (U2 - 1) + 1`:
    // `+q * U2 = -q * lambda` (neg2 = true);
    // when `q > HALF_U2`: `q * U2 ≡ -(U2 - 1 - q) * U2 - 1 = +(U2 - 1 - q) * lambda - 1 (mod r)`.
    let (k2, neg2) = if q > HALF_U2 {
        if neg1 {
            k1 += 1;
        } else if k1 == 0 {
            k1 = 1;
            neg1 = true;
        } else {
            k1 -= 1;
        }
        ((U2 - 1) - q, false)
    } else {
        (q, true)
    };
    (
        k1 | if neg1 { SIGN_BIT } else { 0 },
        k2 | if neg2 { SIGN_BIT } else { 0 },
    )
}

/// Signed Booth digit of window `w` (width `c`) of a 127-bit unsigned magnitude,
/// in `-2^(c-1)..=2^(c-1)`.
#[inline(always)]
fn booth_u128(k: u128, w: usize, c: usize) -> i32 {
    let shifted = if w == 0 {
        (k as u32) << 1
    } else {
        let bit = w * c - 1;
        if bit >= NBITS {
            return 0;
        }
        (k >> bit) as u32
    };
    let tmp = shifted & ((1u32 << (c + 1)) - 1);
    let sign = (tmp & (1u32 << c)) == 0;
    let mag = (tmp + 1) >> 1;
    if sign {
        mag as i32
    } else {
        -(((1u32 << c) - mag) as i32)
    }
}

#[derive(Clone, Copy)]
struct Xyzz {
    x: blst_fp,
    y: blst_fp,
    zz: blst_fp,
    zzz: blst_fp,
}

impl Xyzz {
    #[inline(always)]
    fn identity() -> Self {
        Self {
            x: fp_zero(),
            y: fp_zero(),
            zz: fp_zero(),
            zzz: fp_zero(),
        }
    }

    #[inline(always)]
    fn is_identity(&self) -> bool {
        fp_is_zero(&self.zz) && fp_is_zero(&self.zzz)
    }

    fn from_affine(x: &blst_fp, y: &blst_fp) -> Self {
        Self {
            x: *x,
            y: *y,
            zz: FP_ONE,
            zzz: FP_ONE,
        }
    }

    fn double(&mut self) {
        if self.is_identity() {
            return;
        }
        let (mut u, mut v, mut w, mut s, mut xsq, mut m) = (
            fp_zero(),
            fp_zero(),
            fp_zero(),
            fp_zero(),
            fp_zero(),
            fp_zero(),
        );
        fp_add(&mut u, &self.y, &self.y);
        fp_sqr(&mut v, &u);
        fp_mul(&mut w, &v, &u);
        fp_mul(&mut s, &self.x, &v);
        fp_sqr(&mut xsq, &self.x);
        fp_add(&mut m, &xsq, &xsq);
        fp_add_assign(&mut m, &xsq);
        fp_sqr(&mut self.x, &m);
        fp_sub_assign(&mut self.x, &s);
        fp_sub_assign(&mut self.x, &s);
        fp_sub_assign(&mut s, &self.x);
        fp_mul_assign(&mut s, &m);
        fp_mul_assign(&mut self.y, &w);
        fp_rsub_assign(&mut self.y, &s);
        fp_mul_assign(&mut self.zz, &v);
        fp_mul_assign(&mut self.zzz, &w);
    }

    fn add_affine(&mut self, x2: &blst_fp, y2: &blst_fp, negate: bool) {
        let mut ny2 = fp_zero();
        let y2 = if negate {
            fp_cneg(&mut ny2, y2, true);
            &ny2
        } else {
            y2
        };
        if self.is_identity() {
            *self = Self::from_affine(x2, y2);
            return;
        }
        let (mut p, mut r) = (fp_zero(), fp_zero());
        fp_mul(&mut p, x2, &self.zz);
        fp_sub_assign(&mut p, &self.x);
        fp_mul(&mut r, y2, &self.zzz);
        fp_sub_assign(&mut r, &self.y);
        if !fp_is_zero(&p) {
            let (mut pp, mut ppp, mut q) = (fp_zero(), fp_zero(), fp_zero());
            fp_sqr(&mut pp, &p);
            fp_mul(&mut ppp, &pp, &p);
            fp_mul(&mut q, &self.x, &pp);
            fp_sqr(&mut self.x, &r);
            fp_sub_assign(&mut self.x, &ppp);
            fp_sub_assign(&mut self.x, &q);
            fp_sub_assign(&mut self.x, &q);
            fp_sub_assign(&mut q, &self.x);
            fp_mul_assign(&mut q, &r);
            fp_mul_assign(&mut self.y, &ppp);
            fp_rsub_assign(&mut self.y, &q);
            fp_mul_assign(&mut self.zz, &pp);
            fp_mul_assign(&mut self.zzz, &ppp);
        } else if fp_is_zero(&r) {
            *self = Self::from_affine(x2, y2);
            self.double();
        } else {
            *self = Self::identity();
        }
    }

    fn add(&mut self, other: &Self) {
        if other.is_identity() {
            return;
        }
        if self.is_identity() {
            *self = *other;
            return;
        }
        let (mut u1, mut s1, mut p, mut r) = (fp_zero(), fp_zero(), fp_zero(), fp_zero());
        fp_mul(&mut u1, &self.x, &other.zz);
        fp_mul(&mut s1, &self.y, &other.zzz);
        fp_mul(&mut p, &other.x, &self.zz);
        fp_sub_assign(&mut p, &u1);
        fp_mul(&mut r, &other.y, &self.zzz);
        fp_sub_assign(&mut r, &s1);
        if !fp_is_zero(&p) {
            let (mut pp, mut ppp, mut q) = (fp_zero(), fp_zero(), fp_zero());
            fp_sqr(&mut pp, &p);
            fp_mul(&mut ppp, &pp, &p);
            fp_mul(&mut q, &u1, &pp);
            fp_sqr(&mut self.x, &r);
            fp_sub_assign(&mut self.x, &ppp);
            fp_sub_assign(&mut self.x, &q);
            fp_sub_assign(&mut self.x, &q);
            fp_sub_assign(&mut q, &self.x);
            fp_mul_assign(&mut q, &r);
            fp_mul_assign(&mut s1, &ppp);
            fp_sub(&mut self.y, &q, &s1);
            fp_mul_assign(&mut self.zz, &other.zz);
            fp_mul_assign(&mut self.zz, &pp);
            fp_mul_assign(&mut self.zzz, &other.zzz);
            fp_mul_assign(&mut self.zzz, &ppp);
        } else if fp_is_zero(&r) {
            self.double();
        } else {
            *self = Self::identity();
        }
    }

    /// Converts XYZZ `(X, Y, ZZ, ZZZ)` to Jacobian `blst_p1` `(X*ZZ, Y*ZZZ, ZZ)`
    /// without any field inversion.
    fn to_p1(self) -> blst_p1 {
        let mut p = blst_p1 {
            z: self.zz,
            ..Default::default()
        };
        fp_mul(&mut p.x, &self.x, &self.zz);
        fp_mul(&mut p.y, &self.y, &self.zzz);
        p
    }
}

#[inline(always)]
fn prefetch_read<T>(ptr: *const T) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        std::arch::asm!(
            "prfm pldl1keep, [{p}]",
            p = in(reg) ptr,
            options(nostack, readonly, preserves_flags),
        );
    }
    #[cfg(not(target_arch = "aarch64"))]
    let _ = ptr;
}

const FLAG_FULL: u8 = 1;
const FLAG_QUEUED: u8 = 2;

#[derive(Clone, Copy)]
struct BucketXY {
    x: blst_fp,
    y: blst_fp,
}

impl BucketXY {
    #[inline(always)]
    fn zero() -> Self {
        Self {
            x: fp_zero(),
            y: fp_zero(),
        }
    }
}

#[derive(Clone, Copy)]
struct AffineBucket {
    x: blst_fp,
    y: blst_fp,
    full: bool,
    queued: bool,
}

impl AffineBucket {
    fn empty() -> Self {
        Self {
            x: fp_zero(),
            y: fp_zero(),
            full: false,
            queued: false,
        }
    }
}

#[derive(Clone, Copy)]
struct Pending {
    bucket: u32,
    point: u32, // (base_idx << 1) | is_phi
    negate: bool,
    same_x: bool,
}

#[inline(always)]
fn invert_all(values: &mut [blst_fp], scratch: &mut [blst_fp]) {
    let Some(last) = values.len().checked_sub(1) else {
        return;
    };
    scratch[0] = values[0];
    for j in 1..values.len() {
        let prev = scratch[j - 1];
        fp_mul(&mut scratch[j], &prev, &values[j]);
    }
    let mut inv = fp_zero();
    fp_inv(&mut inv, &scratch[last]);
    let mut tmp = fp_zero();
    for j in (1..values.len()).rev() {
        fp_mul(&mut tmp, &inv, &scratch[j - 1]);
        let vj = values[j];
        fp_mul_assign(&mut inv, &vj);
        values[j] = tmp;
    }
    values[0] = inv;
}

#[inline(always)]
fn add_with_inverse(
    x1: &mut blst_fp,
    y1: &mut blst_fp,
    x2: &blst_fp,
    y2: &blst_fp,
    negate: bool,
    inv: &blst_fp,
) {
    let (mut lambda, mut t, mut d) = (fp_zero(), fp_zero(), fp_zero());
    if negate {
        fp_add(&mut t, y2, y1);
    } else {
        fp_sub(&mut t, y2, y1);
    }
    fp_mul(&mut lambda, &t, inv);
    fp_sqr(&mut t, &lambda);
    fp_sub_assign(&mut t, x1);
    fp_sub_assign(&mut t, x2);
    if negate {
        fp_sub(&mut d, &t, x1);
    } else {
        fp_sub(&mut d, x1, &t);
    }
    *x1 = t;
    fp_mul(&mut t, &lambda, &d);
    fp_rsub_assign(y1, &t);
}

#[inline(always)]
fn xy_of<'a>(
    bases: &'a [G1AffineLimbs],
    phi_x: &'a [blst_fp],
    pt: u32,
) -> (&'a blst_fp, &'a blst_fp) {
    let idx = (pt >> 1) as usize;
    // SAFETY: `G1AffineLimbs` is `[u64; 12]`, identical layout to `[blst_fp; 2]`.
    let pair = unsafe { &*(bases.as_ptr().add(idx) as *const [blst_fp; 2]) };
    if (pt & 1) == 0 {
        (&pair[0], &pair[1])
    } else {
        (&phi_x[idx], &pair[1])
    }
}

struct AffineBuckets {
    /// Compact 1-byte-per-bucket status flags (`FLAG_FULL`, `FLAG_QUEUED`),
    /// fitting completely in L1D (8–16 KiB) so `enqueue()` avoids touching
    /// `buckets` on the common queued path.
    flags: Vec<u8>,
    buckets: Vec<BucketXY>,
    /// Allocated lazily only if a collision/overflow sends a point to XYZZ.
    side: Vec<Xyzz>,
    sides: bool,
    batch: Vec<Pending>,
    dx: Vec<blst_fp>,
    prefix: Vec<blst_fp>,
    waiting: Vec<Pending>,
    spare: Vec<Pending>,
    acc: Vec<AffineBucket>,
    sum: Vec<AffineBucket>,
}

impl AffineBuckets {
    fn new(buckets: usize, batch: usize) -> Self {
        let lanes = LANES.min(batch);
        Self {
            flags: vec![0u8; buckets],
            buckets: vec![BucketXY::zero(); buckets],
            side: Vec::new(),
            sides: false,
            batch: Vec::with_capacity(batch),
            dx: vec![fp_zero(); batch],
            prefix: vec![fp_zero(); batch],
            waiting: Vec::with_capacity(batch),
            spare: Vec::with_capacity(batch),
            acc: vec![AffineBucket::empty(); lanes],
            sum: vec![AffineBucket::empty(); lanes],
        }
    }

    fn ensure(&mut self, buckets: usize, batch: usize) {
        let lanes = LANES.min(batch);
        if self.buckets.len() < buckets {
            self.flags.resize(buckets, 0u8);
            self.buckets.resize(buckets, BucketXY::zero());
        }
        if self.dx.len() < batch {
            self.dx.resize(batch, fp_zero());
            self.prefix.resize(batch, fp_zero());
        }
        if self.acc.len() < lanes {
            self.acc.resize(lanes, AffineBucket::empty());
            self.sum.resize(lanes, AffineBucket::empty());
        }
        self.batch.clear();
        self.waiting.clear();
        self.spare.clear();
        self.sides = false;
    }

    #[inline(never)]
    fn add_side(&mut self, b: usize, x: &blst_fp, y: &blst_fp, negate: bool) {
        if self.side.len() < self.buckets.len() {
            self.side.resize(self.buckets.len(), Xyzz::identity());
        }
        self.side[b].add_affine(x, y, negate);
        self.sides = true;
    }

    fn wipe_all(&mut self, used_buckets: usize) {
        let n = used_buckets.min(self.buckets.len());
        self.flags[..n].fill(0u8);
        wipe_slice(&mut self.buckets[..n]);
        if !self.side.is_empty() {
            wipe_slice(&mut self.side);
        }
        wipe_slice(&mut self.dx);
        wipe_slice(&mut self.prefix);
        wipe_slice(&mut self.acc);
        wipe_slice(&mut self.sum);
        self.batch.clear();
        self.waiting.clear();
        self.spare.clear();
        self.sides = false;
    }

    #[inline(always)]
    fn add(
        &mut self,
        b: usize,
        pt: u32,
        negate: bool,
        batch_cap: usize,
        bases: &[G1AffineLimbs],
        phi_x: &[blst_fp],
    ) {
        let p = Pending {
            bucket: b as u32,
            point: pt,
            negate,
            same_x: false,
        };
        self.enqueue(p, batch_cap, bases, phi_x);
        if self.batch.len() == batch_cap {
            self.flush(batch_cap, bases, phi_x);
        }
    }

    #[inline(always)]
    fn enqueue(&mut self, p: Pending, batch_cap: usize, bases: &[G1AffineLimbs], phi_x: &[blst_fp]) {
        let bi = p.bucket as usize;
        let f = self.flags[bi];
        if (f & FLAG_FULL) == 0 {
            let (x, y) = xy_of(bases, phi_x, p.point);
            let bucket = &mut self.buckets[bi];
            bucket.x = *x;
            if p.negate {
                fp_cneg(&mut bucket.y, y, true);
            } else {
                bucket.y = *y;
            }
            self.flags[bi] = FLAG_FULL;
        } else if (f & FLAG_QUEUED) == 0 {
            self.flags[bi] = FLAG_FULL | FLAG_QUEUED;
            self.batch.push(p);
        } else if self.waiting.len() < batch_cap {
            self.waiting.push(p);
        } else {
            let (x, y) = xy_of(bases, phi_x, p.point);
            self.add_side(bi, x, y, p.negate);
        }
    }

    fn flush(&mut self, batch_cap: usize, bases: &[G1AffineLimbs], phi_x: &[blst_fp]) {
        loop {
            self.add_batch(bases, phi_x);
            std::mem::swap(&mut self.waiting, &mut self.spare);
            for k in 0..self.spare.len() {
                let p = self.spare[k];
                self.enqueue(p, batch_cap, bases, phi_x);
            }
            self.spare.clear();
            if self.batch.len() < batch_cap {
                return;
            }
        }
    }

    fn add_batch(&mut self, bases: &[G1AffineLimbs], phi_x: &[blst_fp]) {
        let len = self.batch.len();
        const PF: usize = 6;
        for k in 0..len {
            if k + PF < len {
                let pf = self.batch[k + PF];
                let bp = unsafe { self.buckets.as_ptr().add(pf.bucket as usize) as *const u8 };
                prefetch_read(bp);
                prefetch_read(unsafe { bp.add(64) });
                let pidx = (pf.point >> 1) as usize;
                let base_p = unsafe { bases.as_ptr().add(pidx) as *const u8 };
                prefetch_read(base_p);
                prefetch_read(unsafe { base_p.add(64) });
                if (pf.point & 1) != 0 {
                    prefetch_read(unsafe { phi_x.as_ptr().add(pidx) });
                }
            }
            let p = &mut self.batch[k];
            let (x, _) = xy_of(bases, phi_x, p.point);
            let dx = &mut self.dx[k];
            fp_sub(dx, x, &self.buckets[p.bucket as usize].x);
            p.same_x = fp_is_zero(dx);
            if p.same_x {
                *dx = FP_ONE;
            }
        }
        invert_all(&mut self.dx[..len], &mut self.prefix);
        for k in 0..len {
            if k + PF < len {
                let pf = self.batch[k + PF];
                let bp = unsafe { self.buckets.as_ptr().add(pf.bucket as usize) as *const u8 };
                prefetch_read(bp);
                prefetch_read(unsafe { bp.add(64) });
                let pidx = (pf.point >> 1) as usize;
                let base_p = unsafe { bases.as_ptr().add(pidx) as *const u8 };
                prefetch_read(base_p);
                prefetch_read(unsafe { base_p.add(64) });
                if (pf.point & 1) != 0 {
                    prefetch_read(unsafe { phi_x.as_ptr().add(pidx) });
                }
            }
            let p = self.batch[k];
            let inv = self.dx[k];
            let bi = p.bucket as usize;
            self.flags[bi] = FLAG_FULL;
            let (x, y) = xy_of(bases, phi_x, p.point);
            if p.same_x {
                self.add_side(bi, x, y, p.negate);
            } else {
                let bucket = &mut self.buckets[bi];
                add_with_inverse(&mut bucket.x, &mut bucket.y, x, y, p.negate, &inv);
            }
        }
        self.batch.clear();
    }

    /// Batched-affine accumulator for tiny top windows (`used <= 2`, e.g. the
    /// 1-bit top window of `c = 14`), avoiding single-bucket `waiting`/`side`
    /// overflow and `side` buffer allocation. Uses 64 L1-resident lanes per
    /// bucket with Montgomery batch inversion (`5M + 1S` per point).
    #[allow(clippy::too_many_arguments)]
    fn accumulate_tiny(
        &mut self,
        w: usize,
        c: usize,
        used: usize,
        start: usize,
        end: usize,
        half: &[(u128, u128)],
        bases: &[G1AffineLimbs],
        phi_slice: &[blst_fp],
    ) -> Xyzz {
        const TINY_LANES: usize = 64;
        debug_assert!(used <= 2);
        let mut lanes = [AffineBucket::empty(); 2 * TINY_LANES];
        let mut next_lane = [0usize; 2];
        let mut side = [Xyzz::identity(); 2];
        let mut pend = [Pending {
            bucket: 0,
            point: 0,
            negate: false,
            same_x: false,
        }; 2 * TINY_LANES];
        let mut plen = 0usize;

        let flush_tiny = |lanes: &mut [AffineBucket],
                          pend: &mut [Pending],
                          plen: &mut usize,
                          side: &mut [Xyzz; 2],
                          dx: &mut [blst_fp],
                          prefix: &mut [blst_fp]| {
            let len = *plen;
            if len == 0 {
                return;
            }
            for k in 0..len {
                let p = &mut pend[k];
                let (x, _) = xy_of(bases, phi_slice, p.point);
                fp_sub(&mut dx[k], x, &lanes[p.bucket as usize].x);
                p.same_x = fp_is_zero(&dx[k]);
                if p.same_x {
                    dx[k] = FP_ONE;
                }
            }
            invert_all(&mut dx[..len], prefix);
            for k in 0..len {
                let p = pend[k];
                let slot = p.bucket as usize;
                let (x, y) = xy_of(bases, phi_slice, p.point);
                if p.same_x {
                    side[slot / TINY_LANES].add_affine(x, y, p.negate);
                } else {
                    add_with_inverse(&mut lanes[slot].x, &mut lanes[slot].y, x, y, p.negate, &dx[k]);
                }
            }
            *plen = 0;
        };

        let push_pt = |b: usize,
                           pt: u32,
                           negate: bool,
                           lanes: &mut [AffineBucket],
                           next_lane: &mut [usize; 2],
                           pend: &mut [Pending],
                           plen: &mut usize,
                           side: &mut [Xyzz; 2],
                           dx: &mut [blst_fp],
                           prefix: &mut [blst_fp]| {
            let l = next_lane[b];
            next_lane[b] = (l + 1) & (TINY_LANES - 1);
            let slot = b * TINY_LANES + l;
            if !lanes[slot].full {
                let (x, y) = xy_of(bases, phi_slice, pt);
                lanes[slot].x = *x;
                if negate {
                    fp_cneg(&mut lanes[slot].y, y, true);
                } else {
                    lanes[slot].y = *y;
                }
                lanes[slot].full = true;
            } else {
                pend[*plen] = Pending {
                    bucket: slot as u32,
                    point: pt,
                    negate,
                    same_x: false,
                };
                *plen += 1;
                if *plen == TINY_LANES {
                    flush_tiny(lanes, pend, plen, side, dx, prefix);
                }
            }
        };

        for (off, &(e1, e2)) in half[start..end].iter().enumerate() {
            let i = start + off;
            let (k1, neg1) = (e1 & !SIGN_BIT, (e1 & SIGN_BIT) != 0);
            let d1 = booth_u128(k1, w, c);
            if d1 != 0 {
                let b = d1.unsigned_abs() as usize - 1;
                push_pt(
                    b,
                    (i as u32) << 1,
                    (d1 < 0) ^ neg1,
                    &mut lanes,
                    &mut next_lane,
                    &mut pend,
                    &mut plen,
                    &mut side,
                    &mut self.dx,
                    &mut self.prefix,
                );
            }
            let (k2, neg2) = (e2 & !SIGN_BIT, (e2 & SIGN_BIT) != 0);
            let d2 = booth_u128(k2, w, c);
            if d2 != 0 {
                let b = d2.unsigned_abs() as usize - 1;
                push_pt(
                    b,
                    ((i as u32) << 1) | 1,
                    (d2 < 0) ^ neg2,
                    &mut lanes,
                    &mut next_lane,
                    &mut pend,
                    &mut plen,
                    &mut side,
                    &mut self.dx,
                    &mut self.prefix,
                );
            }
        }
        flush_tiny(
            &mut lanes,
            &mut pend,
            &mut plen,
            &mut side,
            &mut self.dx,
            &mut self.prefix,
        );

        let mut b_sums = [Xyzz::identity(); 2];
        for b in 0..used {
            let mut s = side[b];
            for l in 0..TINY_LANES {
                let slot = &lanes[b * TINY_LANES + l];
                if slot.full {
                    s.add_affine(&slot.x, &slot.y, false);
                }
            }
            b_sums[b] = s;
        }
        let mut total = b_sums[0];
        if used > 1 && !b_sums[1].is_identity() {
            b_sums[1].double();
            total.add(&b_sums[1]);
        }
        wipe_slice(&mut lanes);
        wipe_slice(&mut side);
        wipe_slice(&mut b_sums);
        total
    }

    fn integrate_and_clear(
        &mut self,
        used: usize,
        batch_cap: usize,
        bases: &[G1AffineLimbs],
        phi_x: &[blst_fp],
    ) -> Xyzz {
        for _ in 0..4 {
            if self.batch.is_empty() && self.waiting.is_empty() {
                break;
            }
            self.flush(batch_cap, bases, phi_x);
        }
        for k in 0..self.batch.len() {
            let p = self.batch[k];
            self.flags[p.bucket as usize] = FLAG_FULL;
            self.waiting.push(p);
        }
        self.batch.clear();
        for k in 0..self.waiting.len() {
            let p = self.waiting[k];
            let (x, y) = xy_of(bases, phi_x, p.point);
            self.add_side(p.bucket as usize, x, y, p.negate);
        }
        self.waiting.clear();
        let lanes = (used / 8).clamp(32, LANES.min(batch_cap));
        if !self.sides
            && used >= 4 * lanes
            && let Some(sum) = self.integrate_lanes(used, lanes)
        {
            self.flags[..used].fill(0u8);
            return sum;
        }
        let mut acc = Xyzz::identity();
        let mut sum = Xyzz::identity();
        let has_side = self.sides && self.side.len() >= used;
        for i in (0..used).rev() {
            if (self.flags[i] & FLAG_FULL) != 0 {
                let bucket = &self.buckets[i];
                acc.add_affine(&bucket.x, &bucket.y, false);
                self.flags[i] = 0;
            }
            if has_side && !self.side[i].is_identity() {
                acc.add(&self.side[i]);
                self.side[i] = Xyzz::identity();
            }
            sum.add(&acc);
        }
        self.sides = false;
        sum
    }

    fn integrate_lanes(&mut self, used: usize, lanes: usize) -> Option<Xyzz> {
        let seg = used / lanes;
        for s in self.acc[..lanes].iter_mut().chain(&mut self.sum[..lanes]) {
            (s.full, s.queued) = (false, false);
        }
        for j in (0..seg).rev() {
            let (acc, sum) = (&mut self.acc[..lanes], &mut self.sum[..lanes]);
            let buckets = &self.buckets;
            let flags = &self.flags;
            if !add_lanes_xy(
                acc,
                buckets,
                flags,
                seg,
                j,
                &mut self.dx,
                &mut self.prefix,
            ) || !add_lanes(sum, |l| &acc[l], &mut self.dx, &mut self.prefix)
            {
                return None;
            }
        }
        let (mut run, mut total, mut sums) = (Xyzz::identity(), Xyzz::identity(), Xyzz::identity());
        for l in (0..lanes).rev() {
            total.add(&run);
            if self.acc[l].full {
                run.add_affine(&self.acc[l].x, &self.acc[l].y, false);
            }
            if self.sum[l].full {
                sums.add_affine(&self.sum[l].x, &self.sum[l].y, false);
            }
        }
        for _ in 0..seg.ilog2() {
            total.double();
        }
        total.add(&sums);
        Some(total)
    }
}

fn add_lanes_xy(
    dst: &mut [AffineBucket],
    buckets: &[BucketXY],
    flags: &[u8],
    seg: usize,
    j: usize,
    dx: &mut [blst_fp],
    prefix: &mut [blst_fp],
) -> bool {
    let lanes = dst.len();
    let mut m = 0;
    for (l, d) in dst.iter_mut().enumerate() {
        if l + 4 < lanes {
            let pf_idx = (l + 4) * seg + j;
            prefetch_read(unsafe { flags.as_ptr().add(pf_idx) });
            let bp = unsafe { buckets.as_ptr().add(pf_idx) as *const u8 };
            prefetch_read(bp);
            prefetch_read(unsafe { bp.add(64) });
        }
        let idx = l * seg + j;
        if (flags[idx] & FLAG_FULL) == 0 {
            continue;
        }
        let s = &buckets[idx];
        if !d.full {
            d.x = s.x;
            d.y = s.y;
            d.full = true;
            d.queued = false;
            continue;
        }
        fp_sub(&mut dx[m], &s.x, &d.x);
        if fp_is_zero(&dx[m]) {
            return false;
        }
        d.queued = true;
        m += 1;
    }
    invert_all(&mut dx[..m], prefix);
    let queued = dst.iter_mut().enumerate().filter(|(_, d)| d.queued);
    for ((l, d), inv) in queued.zip(dx.iter()) {
        let s = &buckets[l * seg + j];
        d.queued = false;
        add_with_inverse(&mut d.x, &mut d.y, &s.x, &s.y, false, inv);
    }
    true
}

fn add_lanes<'a>(
    dst: &mut [AffineBucket],
    src: impl Fn(usize) -> &'a AffineBucket,
    dx: &mut [blst_fp],
    prefix: &mut [blst_fp],
) -> bool {
    let mut m = 0;
    for (l, d) in dst.iter_mut().enumerate() {
        let s = src(l);
        if !s.full {
            continue;
        }
        if !d.full {
            *d = *s;
            continue;
        }
        fp_sub(&mut dx[m], &s.x, &d.x);
        if fp_is_zero(&dx[m]) {
            return false;
        }
        d.queued = true;
        m += 1;
    }
    invert_all(&mut dx[..m], prefix);
    let queued = dst.iter_mut().enumerate().filter(|(_, d)| d.queued);
    for ((l, d), inv) in queued.zip(dx.iter()) {
        let s = src(l);
        d.queued = false;
        add_with_inverse(&mut d.x, &mut d.y, &s.x, &s.y, false, inv);
    }
    true
}

fn window_size(n: usize) -> usize {
    // 2*n half-points per 127-bit window; c = 15 yields 8 full windows (+ 1
    // tiny 7-bit window), eliminating the 2nd-round stall on 8-core SoCs.
    match n {
        0..=4_095 => 10,
        4_096..=12_287 => 11,
        12_288..=32_767 => 12,
        32_768..=65_535 => 13,
        65_536..=90_000 => 14,
        _ => 15,
    }
}

fn chunk_count(n: usize, c: usize, threads: usize) -> usize {
    let windows = NBITS / c + 1;
    // With c = 14, window 9 is only 2 bits (tiny), so count full windows for load balance.
    let full_wins = if NBITS - (windows - 1) * c <= 3 {
        windows - 1
    } else {
        windows
    };
    let mut best = (usize::MAX, 1);
    for chunks in 1..=threads.clamp(1, 3) {
        let per_chunk = (2 * n).div_ceil(chunks);
        if chunks > 1 && per_chunk < 256 {
            break;
        }
        let rounds = (full_wins * chunks).div_ceil(threads.max(1));
        // Weight bucket integration (2^(c-1) buckets per tile) accurately so we
        // don't over-split into extra integration passes.
        let cost = rounds * (per_chunk + (2 << c)) + full_wins * chunks;
        if cost < best.0 {
            best = (cost, chunks);
        }
    }
    best.1
}

/// Precomputes `phi_x[i] = beta * bases[i].x` in parallel for a registered SRS table.
pub(crate) fn precompute_phi_x(bases: &[G1AffineLimbs]) -> Vec<blst_fp> {
    let mut phi_x = vec![fp_zero(); bases.len()];
    phi_x.par_iter_mut().enumerate().for_each(|(i, px)| {
        // SAFETY: `G1AffineLimbs` is `[u64; 12]`, identical to `[blst_fp; 2]`.
        let pair = unsafe { &*(bases.as_ptr().add(i) as *const [blst_fp; 2]) };
        if !(fp_is_zero(&pair[0]) && fp_is_zero(&pair[1])) {
            fp_mul(px, &pair[0], &BETA_MONT);
        }
    });
    phi_x
}

/// Computes `sum scalars[i] * bases[i]` on the current rayon thread pool using
/// the BLS12-381 G1 GLV endomorphism and batched-affine Pippenger.
///
/// Caller guarantees every base in `bases` is in G1 and every scalar is
/// reduced (`< r`). When `cached_phi_x` is provided, it must hold
/// `beta * bases[i].x` for every `i < bases.len()`. When `has_identity` is
/// `false`, caller guarantees no base in `bases` is the point at infinity.
pub(super) fn msm_glv(
    scalars: &[Scalar],
    bases: &[G1AffineLimbs],
    cached_phi_x: Option<&[blst_fp]>,
    has_identity: bool,
) -> Secret<blst_p1> {
    let n = scalars.len();
    debug_assert_eq!(n, bases.len());
    if n == 0 {
        return Secret(blst_p1::default());
    }

    // Decompose all scalars in parallel into a flat `SecretVec` (zeroed on
    // drop), eliminating per-block `OnceLock` overhead across all windows.
    let owned_phi: Vec<blst_fp>;
    let phi_slice: &[blst_fp] = if let Some(cached) = cached_phi_x.filter(|c| c.len() >= n) {
        &cached[..n]
    } else {
        owned_phi = precompute_phi_x(bases);
        &owned_phi
    };
    let mut half = SecretVec::from_elem((0u128, 0u128), n);
    half.par_chunks_mut(1024)
        .zip(scalars.par_chunks(1024))
        .enumerate()
        .for_each(|(ci, (dst, src))| {
            let base_off = ci * 1024;
            if has_identity {
                for (off, (d, sc)) in dst.iter_mut().zip(src.iter()).enumerate() {
                    // SAFETY: `G1AffineLimbs` is `[u64; 12]`, identical to `[blst_fp; 2]`.
                    let pair =
                        unsafe { &*(bases.as_ptr().add(base_off + off) as *const [blst_fp; 2]) };
                    if !(fp_is_zero(&pair[0]) && fp_is_zero(&pair[1])) {
                        *d = decompose(sc);
                    }
                }
            } else {
                for (d, sc) in dst.iter_mut().zip(src.iter()) {
                    *d = decompose(sc);
                }
            }
        });
    let half_slice: &[(u128, u128)] = &half;

    let c = window_size(n);
    let windows = NBITS / c + 1;
    let threads = rayon::current_num_threads();
    let full_wins = if NBITS - (windows - 1) * c <= 3 {
        windows - 1
    } else {
        windows
    };

    // Build tile list `(w, start, end)`.
    // - When `windows == threads + 1` and the top window is narrow (`used <= 256`,
    //   e.g. `c = 15` on 8 cores: 8 full 15-bit windows + 1 narrow 7-bit top
    //   window of 128 buckets), keep `chunks = 1` on windows `0..threads` and
    //   split the cheap-to-integrate top window into `threads` slices so all
    //   cores share it with zero 2nd-round stall.
    // - When `full_wins == threads + 1` (e.g. `c = 14` on 8 cores), split the
    //   extra full window into 2 halves.
    let mut tile_specs: Vec<(usize, usize, usize)> = Vec::with_capacity(windows + threads + 2);
    let top_narrow = windows == threads + 1 && NBITS.saturating_sub(threads * c) <= 8;
    if top_narrow && n >= 16_384 {
        for w in 0..threads {
            tile_specs.push((w, 0, n));
        }
        let splits = threads.min(8);
        let chunk = n.div_ceil(splits);
        for k in 0..splits {
            let start = (k * chunk).min(n);
            let end = ((k + 1) * chunk).min(n);
            if start < end {
                tile_specs.push((threads, start, end));
            }
        }
    } else if full_wins == threads + 1 && n >= 16_384 {
        for w in 0..threads {
            tile_specs.push((w, 0, n));
        }
        let mid = n / 2;
        if mid > 0 {
            tile_specs.push((threads, 0, mid));
        }
        if mid < n {
            tile_specs.push((threads, mid, n));
        }
        for w in (threads + 1)..windows {
            tile_specs.push((w, 0, n));
        }
    } else {
        let chunks = chunk_count(n, c, threads).min(2);
        let chunk = n.div_ceil(chunks);
        for w in 0..windows {
            for k in 0..chunks {
                let start = (k * chunk).min(n);
                let end = ((k + 1) * chunk).min(n);
                if start < end {
                    tile_specs.push((w, start, end));
                }
            }
        }
    }
    let tiles = tile_specs.len();
    let buckets_len = 1usize << (c - 1);
    let batch = (buckets_len / 4).clamp(64, 128).min(buckets_len);

    let results: Vec<OnceLock<Xyzz>> = (0..tiles).map(|_| OnceLock::new()).collect();
    let next = AtomicUsize::new(0);

    thread_local! {
        static TL_BUCKETS: std::cell::RefCell<Option<AffineBuckets>> = const { std::cell::RefCell::new(None) };
    }

    rayon::in_place_scope(|scope| {
        for _ in 0..threads.min(tiles) {
            let (results, next, tile_specs) = (&results, &next, &tile_specs);
            scope.spawn(move |_| {
                let _scrub = ScrubOnExit;
                TL_BUCKETS.with(|cell| {
                    let mut guard = cell.borrow_mut();
                    let affine = match guard.as_mut() {
                        Some(a) => {
                            a.ensure(buckets_len, batch);
                            a
                        }
                        None => guard.insert(AffineBuckets::new(buckets_len, batch)),
                    };
                    loop {
                        let t = next.fetch_add(1, Ordering::Relaxed);
                        if t >= tiles {
                            break;
                        }
                        let (w, start, end) = tile_specs[t];
                        let used = 1usize << (NBITS - w * c).min(c - 1);
                        let tile_batch = (used / 2).clamp(32, batch).min(used);
                        let sum = if used <= 2 {
                            affine.accumulate_tiny(
                                w, c, used, start, end, half_slice, bases, phi_slice,
                            )
                        } else {
                            // Stagger starting offset per tile so parallel windows
                            // don't hit identical cache lines in `bases`/`phi_x`.
                            let len = end - start;
                            let rot = if len > 0 { (t * 4096) % len } else { 0 };
                            let split_pt = start + rot;
                            for &(seg_lo, seg_hi) in &[(split_pt, end), (start, split_pt)] {
                                for (off, &(e1, e2)) in
                                    half_slice[seg_lo..seg_hi].iter().enumerate()
                                {
                                    let i = seg_lo + off;
                                    let (k1, neg1) = (e1 & !SIGN_BIT, (e1 & SIGN_BIT) != 0);
                                    let d1 = booth_u128(k1, w, c);
                                    if d1 != 0 {
                                        affine.add(
                                            d1.unsigned_abs() as usize - 1,
                                            (i as u32) << 1,
                                            (d1 < 0) ^ neg1,
                                            tile_batch,
                                            bases,
                                            phi_slice,
                                        );
                                    }
                                    let (k2, neg2) = (e2 & !SIGN_BIT, (e2 & SIGN_BIT) != 0);
                                    let d2 = booth_u128(k2, w, c);
                                    if d2 != 0 {
                                        affine.add(
                                            d2.unsigned_abs() as usize - 1,
                                            ((i as u32) << 1) | 1,
                                            (d2 < 0) ^ neg2,
                                            tile_batch,
                                            bases,
                                            phi_slice,
                                        );
                                    }
                                }
                            }
                            affine.integrate_and_clear(used, tile_batch, bases, phi_slice)
                        };
                        let _ = results[t].set(sum);
                    }
                    affine.wipe_all(buckets_len);
                });
            });
        }
    });

    // Combine windows via Horner's rule, most significant first.
    let mut win_sums = vec![Xyzz::identity(); windows];
    for (t, &(w, _, _)) in tile_specs.iter().enumerate() {
        if let Some(part) = results[t].get() {
            win_sums[w].add(part);
        }
    }
    let mut acc = Secret(blst_p1::default());
    for w in (0..windows).rev() {
        if w != windows - 1 {
            for _ in 0..c {
                let tmp = *acc;
                // SAFETY: valid blst_p1 point.
                unsafe { blst_p1_double(&mut *acc, &tmp) };
            }
        }
        if !win_sums[w].is_identity() {
            let p = win_sums[w].to_p1();
            let tmp = *acc;
            // SAFETY: valid blst_p1 points.
            unsafe { blst_p1_add_or_double(&mut *acc, &tmp, &p) };
        }
        wipe(&mut win_sums[w]);
    }
    acc
}

#[cfg(test)]
mod tests {
    use blst::blst_p1_is_equal;
    use ff::Field;
    use group::{Curve, Group, prime::PrimeCurveAffine};
    use midnight_curves::{Fq, G1Affine, G1Projective, msm::msm_pippenger};

    use super::*;

    fn next_u64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn fq_from_u128(v: u128) -> Fq {
        let hi = Fq::from((v >> 64) as u64);
        let lo = Fq::from(v as u64);
        let shift = Fq::from(1u64 << 32).square();
        hi * shift + lo
    }

    fn to_limbs<'a>(s: &'a [Fq], b: &'a [G1Affine]) -> (&'a [Scalar], &'a [G1AffineLimbs]) {
        // SAFETY: layout asserted in parent module.
        unsafe {
            (
                &*(s as *const [Fq] as *const [Scalar]),
                &*(b as *const [G1Affine] as *const [G1AffineLimbs]),
            )
        }
    }

    fn run_compare(scalars: &[Fq], bases: &[G1Affine]) {
        let expected: G1Projective = msm_pippenger(scalars, bases);
        let expected_p1: blst_p1 = unsafe { std::mem::transmute(expected) };
        let (sl, bl) = to_limbs(scalars, bases);
        let got = msm_glv(sl, bl, None, true);
        assert!(
            unsafe { blst_p1_is_equal(&*got, &expected_p1) },
            "GLV CPU MSM mismatch for n = {}",
            scalars.len()
        );
        let cached = precompute_phi_x(bl);
        let got_cached = msm_glv(sl, bl, Some(&cached), true);
        assert!(
            unsafe { blst_p1_is_equal(&*got_cached, &expected_p1) },
            "GLV CPU MSM (cached phi_x) mismatch for n = {}",
            scalars.len()
        );
    }

    #[test]
    fn fp_inline_ops_match_blst() {
        let mut st = 0xfeed_face_cafe_babeu64;
        let p_minus_1 = blst_fp {
            l: [
                0xb9fe_ffff_ffff_aaaa,
                0x1eab_fffe_b153_ffff,
                0x6730_d2a0_f6b0_f624,
                0x6477_4b84_f385_12bf,
                0x4b1b_a7b6_434b_acd7,
                0x1a01_11ea_397f_e69a,
            ],
        };
        let mut elems = vec![fp_zero(), FP_ONE, p_minus_1];
        for _ in 0..200 {
            let mut a = blst_fp {
                l: [
                    next_u64(&mut st),
                    next_u64(&mut st),
                    next_u64(&mut st),
                    next_u64(&mut st),
                    next_u64(&mut st),
                    next_u64(&mut st) % 0x1a01_11ea_397f_e69a,
                ],
            };
            let mut out = fp_zero();
            unsafe { blst::blst_fp_mul(&mut out, &a, &FP_ONE) };
            a = out;
            elems.push(a);
        }
        for a in &elems {
            for b in &elems {
                let (mut ga, mut ea) = (fp_zero(), fp_zero());
                fp_add(&mut ga, a, b);
                unsafe { blst::blst_fp_add(&mut ea, a, b) };
                assert_eq!(ga.l, ea.l, "fp_add mismatch");

                let (mut gs, mut es) = (fp_zero(), fp_zero());
                fp_sub(&mut gs, a, b);
                unsafe { blst::blst_fp_sub(&mut es, a, b) };
                assert_eq!(gs.l, es.l, "fp_sub mismatch");
            }
            for flag in [false, true] {
                let (mut gn, mut en) = (fp_zero(), fp_zero());
                fp_cneg(&mut gn, a, flag);
                unsafe { blst::blst_fp_cneg(&mut en, a, flag) };
                assert_eq!(gn.l, en.l, "fp_cneg mismatch");
            }
        }
    }

    #[test]
    fn glv_matches_stock_pippenger() {
        let mut st = 0x1234_5678_9abc_def0u64;
        let n = 5000usize;
        let g = G1Projective::generator();
        let pts: Vec<G1Affine> = (0..n)
            .map(|i| {
                if i == 17 || i == 2048 {
                    G1Affine::identity()
                } else if i == 33 {
                    g.to_affine()
                } else {
                    (g * Fq::from(next_u64(&mut st) | 1)).to_affine()
                }
            })
            .collect();
        let mut scs: Vec<Fq> = (0..n)
            .map(|_| {
                let a = fq_from_u128((next_u64(&mut st) as u128) | ((next_u64(&mut st) as u128) << 64));
                let b = fq_from_u128((next_u64(&mut st) as u128) | ((next_u64(&mut st) as u128) << 64));
                a * b + Fq::from(next_u64(&mut st))
            })
            .collect();
        // Inject boundary scalars: 0, 1, -1, U2, U2-1, U2+1
        scs[0] = Fq::ZERO;
        scs[1] = Fq::ONE;
        scs[2] = -Fq::ONE;
        scs[3] = fq_from_u128(U2);
        scs[4] = fq_from_u128(U2 - 1);
        scs[5] = fq_from_u128(U2 + 1);
        let mut dup_pts = pts[..256].to_vec();
        dup_pts[10] = dup_pts[5];
        dup_pts[11] = dup_pts[5];
        run_compare(&scs[..256], &dup_pts);
        run_compare(&scs[..1], &pts[..1]);
        run_compare(&scs[..100], &pts[..100]);
        run_compare(&scs, &pts);
    }
}
