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

//! Tests of the device context and of the field arithmetic of
//! `shaders/include` against blst, element for element.

use blst::{
    blst_fp, blst_fp_add, blst_fp_cneg, blst_fp_from_uint64, blst_fp_mul, blst_fp_sqr, blst_fp_sub,
    blst_fr, blst_fr_add, blst_fr_cneg, blst_fr_from_uint64, blst_fr_mul, blst_fr_sqr, blst_fr_sub,
};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

use crate::{
    Scalar,
    ctx::{self, Arg},
    field::{FP_MODULUS, FR_MODULUS, FpLimbs, fp_is_reduced, is_reduced},
};

// Operation codes of shaders/field_test.metal.
const OP_ADD: u32 = 0;
const OP_SUB: u32 = 1;
const OP_MUL: u32 = 2;
const OP_SQR: u32 = 3;
const OP_NEG: u32 = 4;
const OP_DBL: u32 = 5;
const OP_IS_REDUCED: u32 = 6;
const OP_ONE: u32 = 7;
const OP_EQ: u32 = 8;

/// The operations compared with blst on edge and random pairs.
const OPS: [(u32, &str); 8] = [
    (OP_ADD, "add"),
    (OP_SUB, "sub"),
    (OP_MUL, "mul"),
    (OP_SQR, "sqr"),
    (OP_NEG, "neg"),
    (OP_DBL, "dbl"),
    (OP_ONE, "one"),
    (OP_EQ, "eq"),
];

fn require_device() {
    assert!(
        crate::device(),
        "no usable Metal device: {:?}",
        ctx::context().err()
    );
}

/// Runs `kernel` (`field_test_fr` / `field_test_fp`) on `a`, `b`.
fn run_op<T: ctx::Pod + Default>(kernel: &str, op: u32, a: &[T], b: &[T]) -> Vec<T> {
    assert_eq!(a.len(), b.len());
    let s = ctx::lock().unwrap();
    let c = s.ctx();
    let ba = c.upload(a).unwrap();
    let bb = c.upload(b).unwrap();
    let out = c.alloc(size_of_val(a)).unwrap();
    let n = a.len() as u32;
    c.run(|cmd| {
        cmd.dispatch(
            kernel,
            a.len(),
            &[
                Arg::Buffer(&ba),
                Arg::Buffer(&bb),
                Arg::Buffer(&out),
                Arg::value(&op),
                Arg::value(&n),
            ],
        )
    })
    .unwrap();
    let mut r = vec![T::default(); a.len()];
    out.read(&mut r);
    r
}

// --- blst references ------------------------------------------------------

fn fr2(
    f: unsafe extern "C" fn(*mut blst_fr, *const blst_fr, *const blst_fr),
    a: &Scalar,
    b: &Scalar,
) -> Scalar {
    let mut r = blst_fr::default();
    // SAFETY: valid blst_fr values.
    unsafe { f(&mut r, &blst_fr { l: *a }, &blst_fr { l: *b }) };
    r.l
}

fn fp2(
    f: unsafe extern "C" fn(*mut blst_fp, *const blst_fp, *const blst_fp),
    a: &FpLimbs,
    b: &FpLimbs,
) -> FpLimbs {
    let mut r = blst_fp::default();
    // SAFETY: valid blst_fp values.
    unsafe { f(&mut r, &blst_fp { l: *a }, &blst_fp { l: *b }) };
    r.l
}

fn fr_ref(op: u32, a: &Scalar, b: &Scalar) -> Scalar {
    let mut r = blst_fr::default();
    match op {
        OP_ADD => fr2(blst_fr_add, a, b),
        OP_SUB => fr2(blst_fr_sub, a, b),
        OP_MUL => fr2(blst_fr_mul, a, b),
        OP_SQR => {
            // SAFETY: valid blst_fr values.
            unsafe { blst_fr_sqr(&mut r, &blst_fr { l: *a }) };
            r.l
        }
        OP_NEG => {
            // SAFETY: as above.
            unsafe { blst_fr_cneg(&mut r, &blst_fr { l: *a }, true) };
            r.l
        }
        OP_DBL => fr2(blst_fr_add, a, a),
        OP_ONE => {
            // SAFETY: four readable limbs.
            unsafe { blst_fr_from_uint64(&mut r, [1u64, 0, 0, 0].as_ptr()) };
            r.l
        }
        OP_IS_REDUCED => [is_reduced(a) as u64, 0, 0, 0],
        OP_EQ => [(a == b) as u64, 0, 0, 0],
        _ => unreachable!(),
    }
}

fn fp_ref(op: u32, a: &FpLimbs, b: &FpLimbs) -> FpLimbs {
    let mut r = blst_fp::default();
    match op {
        OP_ADD => fp2(blst_fp_add, a, b),
        OP_SUB => fp2(blst_fp_sub, a, b),
        OP_MUL => fp2(blst_fp_mul, a, b),
        OP_SQR => {
            // SAFETY: valid blst_fp values.
            unsafe { blst_fp_sqr(&mut r, &blst_fp { l: *a }) };
            r.l
        }
        OP_NEG => {
            // SAFETY: as above.
            unsafe { blst_fp_cneg(&mut r, &blst_fp { l: *a }, true) };
            r.l
        }
        OP_DBL => fp2(blst_fp_add, a, a),
        OP_ONE => {
            // SAFETY: six readable limbs.
            unsafe { blst_fp_from_uint64(&mut r, [1u64, 0, 0, 0, 0, 0].as_ptr()) };
            r.l
        }
        OP_IS_REDUCED => [fp_is_reduced(a) as u64, 0, 0, 0, 0, 0],
        OP_EQ => [(a == b) as u64, 0, 0, 0, 0, 0],
        _ => unreachable!(),
    }
}

// --- inputs ---------------------------------------------------------------

fn sub_small<const N: usize>(m: &[u64; N], k: u64) -> [u64; N] {
    let mut r = *m;
    let mut borrow = k;
    for l in r.iter_mut() {
        let (v, o) = l.overflowing_sub(borrow);
        *l = v;
        borrow = o as u64;
        if borrow == 0 {
            break;
        }
    }
    r
}

fn half<const N: usize>(m: &[u64; N]) -> [u64; N] {
    let mut r = [0u64; N];
    for i in 0..N {
        r[i] = (m[i] >> 1) | if i + 1 < N { m[i + 1] << 63 } else { 0 };
    }
    r
}

/// Edge values of a field: 0, 1, 2, the Montgomery one, p-1, p-2, (p-1)/2,
/// (p+1)/2, values with a single high bit, all-ones limbs below p.
fn edges<const N: usize>(m: &[u64; N], mont_one: [u64; N]) -> Vec<[u64; N]> {
    let mut v = vec![[0u64; N]];
    let mut one = [0u64; N];
    one[0] = 1;
    v.push(one);
    let mut two = [0u64; N];
    two[0] = 2;
    v.push(two);
    v.push(mont_one);
    v.push(sub_small(m, 1));
    v.push(sub_small(m, 2));
    v.push(sub_small(m, 0x1_0000_0000));
    v.push(half(&sub_small(m, 1)));
    let mut h = half(&sub_small(m, 1));
    h[0] += 1;
    v.push(h);
    let mut top = [0u64; N];
    top[N - 1] = 1 << (63 - m[N - 1].leading_zeros());
    v.push(top);
    let mut lowfull = [0u64; N];
    for l in lowfull.iter_mut().take(N - 1) {
        *l = u64::MAX;
    }
    v.push(lowfull);
    let mut word = [0u64; N];
    word[0] = 0xffff_ffff;
    v.push(word);
    v
}

fn random_below<const N: usize>(rng: &mut ChaCha20Rng, m: &[u64; N]) -> [u64; N] {
    let mask = u64::MAX >> m[N - 1].leading_zeros();
    loop {
        let mut a = [0u64; N];
        for l in a.iter_mut() {
            *l = rng.next_u64();
        }
        a[N - 1] &= mask;
        if a.iter().rev().cmp(m.iter().rev()) == std::cmp::Ordering::Less {
            return a;
        }
    }
}

/// Every pair of edge values, then `random` random pairs (some sharing an
/// operand, some near the modulus).
fn pairs<const N: usize>(
    m: &[u64; N],
    mont_one: [u64; N],
    random: usize,
    seed: u64,
) -> (Vec<[u64; N]>, Vec<[u64; N]>) {
    let e = edges(m, mont_one);
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for x in &e {
        for y in &e {
            a.push(*x);
            b.push(*y);
        }
    }
    for i in 0..random {
        let x = random_below(&mut rng, m);
        let y = match i % 8 {
            0 => x,
            1 => sub_small(m, 1 + (rng.next_u64() & 0xffff)),
            2 => e[i % e.len()],
            _ => random_below(&mut rng, m),
        };
        a.push(x);
        b.push(y);
    }
    (a, b)
}

fn fr_one_mont() -> Scalar {
    fr_ref(OP_ONE, &[0; 4], &[0; 4])
}

fn fp_one_mont() -> FpLimbs {
    fp_ref(OP_ONE, &[0; 6], &[0; 6])
}

const RANDOM: usize = 1 << 16;

fn check_fr(op: u32, name: &str) {
    require_device();
    let (a, b) = pairs(&FR_MODULUS, fr_one_mont(), RANDOM, 0x5eed ^ op as u64);
    let got = run_op::<Scalar>("field_test_fr", op, &a, &b);
    for i in 0..a.len() {
        let want = fr_ref(op, &a[i], &b[i]);
        assert_eq!(got[i], want, "Fr {name}: a = {:x?}, b = {:x?}", a[i], b[i]);
    }
}

fn check_fp(op: u32, name: &str) {
    require_device();
    let (a, b) = pairs(&FP_MODULUS, fp_one_mont(), RANDOM, 0xf00d ^ op as u64);
    let got = run_op::<FpLimbs>("field_test_fp", op, &a, &b);
    for i in 0..a.len() {
        let want = fp_ref(op, &a[i], &b[i]);
        assert_eq!(got[i], want, "Fp {name}: a = {:x?}, b = {:x?}", a[i], b[i]);
    }
}

// --- field tests ------------------------------------------------------------

#[test]
fn fr_matches_blst() {
    for (op, name) in OPS {
        check_fr(op, name);
    }
}

#[test]
fn fp_matches_blst() {
    for (op, name) in OPS {
        check_fp(op, name);
    }
}

/// Values at, just above and far above the modulus, and the largest limbs.
fn unreduced<const N: usize>(m: &[u64; N]) -> Vec<[u64; N]> {
    let mut v = vec![*m, sub_small(m, 1), [u64::MAX; N], [0; N]];
    let mut p1 = *m;
    p1[0] += 1;
    v.push(p1);
    let mut hi = *m;
    hi[N - 1] += 1;
    v.push(hi);
    let mut lo = *m;
    lo[N - 1] -= 1;
    v.push(lo);
    let mut top = [0u64; N];
    top[N - 1] = 1 << 63;
    v.push(top);
    let mut rng = ChaCha20Rng::seed_from_u64(7);
    for _ in 0..1000 {
        let mut a = [0u64; N];
        for l in a.iter_mut() {
            *l = rng.next_u64();
        }
        // the top limb at most four times the modulus' so both sides occur
        a[N - 1] %= m[N - 1].saturating_mul(4);
        v.push(a);
    }
    v
}

#[test]
fn reduced_check() {
    require_device();
    let a = unreduced(&FR_MODULUS);
    let got = run_op::<Scalar>("field_test_fr", OP_IS_REDUCED, &a, &a);
    let mut both = [0usize; 2];
    for (x, g) in a.iter().zip(&got) {
        assert_eq!(g[0], is_reduced(x) as u64, "Fr is_reduced {x:x?}");
        both[g[0] as usize] += 1;
    }
    assert!(both[0] > 0 && both[1] > 0);
    let a = unreduced(&FP_MODULUS);
    let got = run_op::<FpLimbs>("field_test_fp", OP_IS_REDUCED, &a, &a);
    for (x, g) in a.iter().zip(&got) {
        assert_eq!(g[0], fp_is_reduced(x) as u64, "Fp is_reduced {x:x?}");
    }
}

// --- context tests
// ------------------------------------------------------------

#[test]
fn context_and_info() {
    require_device();
    let info = crate::present().unwrap();
    assert!(info.unified_memory);
    assert!(info.max_buffer_length > 0 && info.recommended_max_working_set > 0);
    assert!(crate::library_source().is_some());
    let (free, total) = crate::mem_info().unwrap();
    assert!(free <= total && total as u64 == info.recommended_max_working_set);
}

#[test]
fn dispatch_and_read_back() {
    require_device();
    let s = ctx::lock().unwrap();
    let c = s.ctx();
    // a length that is not a multiple of the threadgroup size
    let n: u32 = 100_003;
    let out = c.alloc(n as usize * 4).unwrap();
    let (_, ns) = c
        .run(|cmd| {
            cmd.dispatch(
                "diag_iota",
                n as usize,
                &[Arg::Buffer(&out), Arg::value(&n)],
            )
        })
        .unwrap();
    assert!(out.as_slice::<u32>().iter().enumerate().all(|(i, v)| *v == i as u32));
    assert!(ns > 0);
}

#[test]
fn errors_are_values() {
    require_device();
    let s = ctx::lock().unwrap();
    let c = s.ctx();
    let e = c.run(|cmd| cmd.dispatch("no_such_kernel", 1, &[])).unwrap_err();
    assert_eq!(e.reason(), "device-error");
    let e = c.alloc(usize::MAX / 2).unwrap_err();
    assert_eq!(e.reason(), "memory");
    // a bad threadgroup size is refused before encoding
    let out = c.alloc(64).unwrap();
    let n = 1u32;
    let e = c
        .run(|cmd| {
            cmd.dispatch_with(
                "diag_iota",
                1,
                1 << 20,
                &[Arg::Buffer(&out), Arg::value(&n)],
                &[],
            )
        })
        .unwrap_err();
    assert_eq!(e.reason(), "device-error");
    // the context still works afterwards
    c.run(|cmd| cmd.dispatch("diag_iota", 1, &[Arg::Buffer(&out), Arg::value(&n)]))
        .unwrap();
}

/// The tests that change the thresholds, hold buffers across a session or
/// compare the byte count: one at a time.
fn serial() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

#[test]
fn request_buffers_and_byte_accounting() {
    let _g = serial();
    require_device();
    assert!(crate::end_request());
    let mut s = ctx::lock().unwrap();
    // what stays for the life of the process (the NTT twiddles once a
    // transform ran) or of a handle; the idle buffers of the pool are not in
    // use
    let before = ctx::in_use_bytes();
    {
        let b = s.scratch("test-scratch", 1 << 20).unwrap();
        b.as_mut_slice::<u64>().fill(0xdead_beef);
    }
    // the same key, not longer: the same buffer with its contents
    let b = s.scratch("test-scratch", 1 << 19).unwrap();
    assert!(b.as_slice::<u64>().iter().all(|v| *v == 0xdead_beef));
    assert!(s.scratch_bytes() >= 1 << 20);
    let mid = ctx::in_use_bytes();
    let (held, peak) = crate::device_bytes();
    assert!(mid >= before + (1 << 20) && held >= mid && peak >= held);
    {
        let owned = s.ctx().alloc(1 << 16).unwrap();
        assert!(ctx::in_use_bytes() >= mid + (1 << 16));
        drop(owned);
    }
    assert_eq!(ctx::in_use_bytes(), mid);
    drop(s);
    assert!(crate::end_request());
    // under the lock no other test holds a buffer (each drops its buffers
    // before its session)
    let s = ctx::lock().unwrap();
    assert!(s.get_scratch("test-scratch").is_none());
    assert_eq!(s.scratch_bytes(), 0);
    assert_eq!(ctx::in_use_bytes(), before);
    // the region is idle in the pool, zeroed, until the pool is released
    assert!(crate::pool_bytes() >= 1 << 20);
    drop(s);
    let (held, _) = crate::device_bytes();
    let freed = crate::release_pool();
    assert!(freed >= 1 << 20);
    assert_eq!(crate::pool_bytes(), 0);
    assert_eq!(crate::device_bytes().0, held - freed);
}

/// A pooled buffer goes back to the pool only zeroed, and comes out of it
/// zeroed; the pool keeps no more than its limit.
#[test]
fn pooled_buffers_come_back_zeroed() {
    let _g = serial();
    require_device();
    assert!(crate::end_request());
    let c = ctx::context().unwrap();
    let len = (3 << 20) + 4096;
    let mut a = c.alloc_pooled(len).unwrap();
    a.as_mut_slice::<u8>().fill(0x5a);
    let at = a.as_slice::<u8>().as_ptr() as usize;
    drop(a);
    assert!(crate::pool_bytes() >= len as u64);
    // a little shorter: from the pool (this one unless a test running at
    // the same time took it first), zero over its whole allocation
    let b = c.alloc_pooled(len - 4096).unwrap();
    assert_eq!(b.len_bytes(), len - 4096);
    let p = b.as_slice::<u8>().as_ptr();
    if p as usize == at {
        assert_eq!(b.alloc_bytes(), len);
    }
    // SAFETY: the allocation is `alloc_bytes` long
    let whole = unsafe { std::slice::from_raw_parts(p, b.alloc_bytes()) };
    assert!(whole.iter().all(|x| *x == 0));
    drop(b);
    // much shorter: not that buffer
    let small = c.alloc_pooled(4096).unwrap();
    assert_ne!(small.as_slice::<u8>().as_ptr() as usize, at);
    drop(small);
    // a buffer written in the background zeroing
    let mut d = c.alloc_pooled(len).unwrap();
    d.as_mut_slice::<u8>().fill(0x33);
    ctx::release_later(vec![d]);
    ctx::settle_releases();
    let e = c.alloc_pooled(len).unwrap();
    assert!(e.as_slice::<u8>().iter().all(|x| *x == 0));
    drop(e);
    // a limit below the pool releases what is above it
    ctx::set_pool_limit(Some(0));
    assert_eq!(crate::pool_bytes(), 0);
    let f = c.alloc_pooled(len).unwrap();
    drop(f);
    assert_eq!(crate::pool_bytes(), 0, "nothing is kept above the limit");
    ctx::set_pool_limit(None);
    assert!(crate::end_request());
}

// --- the entry points against the stock CPU code
// ------------------------------

mod stock {
    use ff::{Field, PrimeField};
    use group::{Curve, Group};
    use midnight_curves::{
        Fq, G1Affine, G1Projective,
        fft::{best_fft_with_twiddles, compute_twiddles},
    };
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    /// The standard `2^log_n`-th root of unity (as `EvaluationDomain`
    /// derives it).
    pub fn root(log_n: u32) -> Fq {
        let mut w = Fq::ROOT_OF_UNITY;
        for _ in log_n..Fq::S {
            w = w.square();
        }
        w
    }

    pub fn scalars(n: usize, seed: u64) -> Vec<Fq> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        (0..n).map(|_| Fq::random(&mut rng)).collect()
    }

    pub fn points(n: usize, seed: u64) -> Vec<G1Affine> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let step = G1Projective::random(&mut rng);
        let mut p = G1Projective::random(&mut rng);
        let proj: Vec<G1Projective> = (0..n)
            .map(|_| {
                p += step;
                p
            })
            .collect();
        let mut out = vec![G1Affine::default(); n];
        G1Projective::batch_normalize(&proj, &mut out);
        out
    }

    /// The stock forward transform (`best_fft_with_twiddles`, as the domain
    /// calls it).
    pub fn fft(a: &mut [Fq], omega: Fq, log_n: u32) {
        best_fft_with_twiddles(a, &compute_twiddles(&omega, log_n), log_n);
    }

    /// The stock inverse transform: the transform, then the divisor.
    pub fn ifft(a: &mut [Fq], omega_inv: Fq, log_n: u32, divisor: Fq) {
        fft(a, omega_inv, log_n);
        a.iter_mut().for_each(|x| *x *= divisor);
    }

    pub fn msm(s: &[Fq], b: &[G1Affine]) -> G1Affine {
        midnight_curves::msm::msm_pippenger(s, b).to_affine()
    }
}

fn decline_count(class: &str, reason: &str) -> u64 {
    crate::declines()
        .iter()
        .filter(|d| d.0 == class && d.1 == reason)
        .map(|d| d.2)
        .sum()
}

/// The quotient hook of [`entry_points_serve_small_calls`]: before the real
/// call, a copy of the job with an unreduced challenge must decline
/// `unreduced` and leave the output unchanged.
unsafe fn checked_quotient(job: *const std::ffi::c_void, out: &mut [Scalar]) -> bool {
    use midnight_proofs::gpu::abi::QuotientJobAbi;
    // SAFETY: the prover's job, within its call.
    let mut bad: QuotientJobAbi = unsafe { *(job as *const QuotientJobAbi) };
    bad.y = [u64::MAX; 4];
    let sentinel = vec![[0x5eedu64; 4]; out.len()];
    let mut o = sentinel.clone();
    let before = decline_count("quotient", "unreduced");
    // SAFETY: the job's tables, within the prover's call.
    let ok = unsafe { crate::quotient(&bad as *const QuotientJobAbi as *const _, &mut o) };
    assert!(
        !ok && o == sentinel,
        "an unreduced job declines, out unchanged"
    );
    assert_eq!(decline_count("quotient", "unreduced") - before, 1);
    // SAFETY: the caller's job.
    unsafe { crate::quotient(job, out) }
}

/// Installs [`checked_quotient`] as the prover's quotient hook (once per
/// process; the other hooks stay the stock CPU code).
fn install_checked_quotient() {
    use midnight_proofs::gpu::{self as hooks, GpuHooks};
    static INSTALL: std::sync::Once = std::sync::Once::new();
    INSTALL.call_once(|| {
        assert!(hooks::install(GpuHooks {
            msm_g1: None,
            fft: None,
            ifft: None,
            quotient: Some(checked_quotient),
        }));
    });
}

/// Each entry point serves a small valid call with the stock CPU value, and
/// declines a call of a wrong shape or with an unreduced input, leaving the
/// caller's buffer unchanged.
#[test]
fn entry_points_serve_small_calls() {
    use std::sync::atomic::Ordering::Relaxed;

    use group::Curve;
    use midnight_curves::Fq;
    use midnight_proofs::gpu::{
        self as hooks,
        adapter::{affine, from_jacobian, scalar, scalars, scalars_mut},
        test_switches::{QUOTIENT_CHECK, RandomCircuit},
    };
    let _g = serial();
    require_device();
    assert!(crate::set_min_k(1));
    let log_n = 4;
    let n = 1usize << log_n;
    let w = stock::root(log_n);
    let wi = ff::Field::invert(&w).unwrap();
    let div = ff::Field::invert(&Fq::from(n as u64)).unwrap();
    let input = stock::scalars(n, 1);

    // fft
    let served = crate::stats();
    let mut want = input.clone();
    stock::fft(&mut want, w, log_n);
    let mut got = input.clone();
    assert!(crate::fft(scalars_mut(&mut got), scalar(w), log_n));
    assert_eq!(got, want, "fft");
    // ifft
    let mut want = input.clone();
    stock::ifft(&mut want, wi, log_n, div);
    let mut got = input.clone();
    assert!(crate::ifft(
        scalars_mut(&mut got),
        scalar(wi),
        log_n,
        scalar(div)
    ));
    assert_eq!(got, want, "ifft");
    // msm (one-shot: no table registered)
    let points = stock::points(n, 2);
    let r = crate::msm_g1(scalars(&input), affine(&points)).expect("the msm is served");
    assert_eq!(
        from_jacobian(r).to_affine(),
        stock::msm(&input, &points),
        "msm"
    );
    let now = crate::stats();
    assert_eq!(now.fft.served - served.fft.served, 1);
    assert_eq!(now.ifft.served - served.ifft.served, 1);
    assert_eq!(now.msm.served - served.msm.served, 1);

    // quotient: a real job of a proof, with the hook layer's check mode
    // comparing the device quotient with the stock CPU one
    install_checked_quotient();
    let (checks, diffs) = (hooks::quotient_checks(), hooks::quotient_check_diffs());
    let q = crate::stats().quotient.served;
    QUOTIENT_CHECK.store(true, Relaxed);
    let (_, verified) = RandomCircuit::new(8, true, 7).prove();
    QUOTIENT_CHECK.store(false, Relaxed);
    assert!(verified, "the proof with the device quotient verifies");
    assert_eq!(
        crate::stats().quotient.served - q,
        1,
        "{:?}",
        crate::declines()
    );
    assert_eq!(hooks::quotient_checks() - checks, 1);
    assert_eq!(hooks::quotient_check_diffs(), diffs, "the stock quotient");

    // declines leave the buffer unchanged
    let orig = scalars(&input).to_vec();
    let mut a = orig.clone();
    let shape = decline_count("fft", "shape");
    assert!(!crate::fft(&mut a[..n - 1], scalar(w), log_n));
    assert_eq!(decline_count("fft", "shape") - shape, 1);
    let mut bad = orig.clone();
    bad[3] = FR_MODULUS;
    let kept = bad.clone();
    let unreduced = decline_count("fft", "unreduced");
    assert!(!crate::fft(&mut bad, scalar(w), log_n));
    assert_eq!(bad, kept);
    assert_eq!(decline_count("fft", "unreduced") - unreduced, 1);
    // a root other than the standard one
    assert!(!crate::fft(&mut a, scalar(w * w), log_n));
    let unreduced = decline_count("ifft", "unreduced");
    assert!(!crate::ifft(&mut a, scalar(wi), log_n, FR_MODULUS));
    assert_eq!(decline_count("ifft", "unreduced") - unreduced, 1);
    assert_eq!(a, orig);
    let shape = decline_count("msm", "shape");
    assert!(crate::msm_g1(&orig[..n - 1], affine(&points)).is_none());
    assert_eq!(decline_count("msm", "shape") - shape, 1);
    let unreduced = decline_count("msm", "unreduced");
    assert!(crate::msm_g1(&kept, affine(&points)).is_none());
    assert_eq!(decline_count("msm", "unreduced") - unreduced, 1);
    let shape = decline_count("quotient", "shape");
    // SAFETY: a null job is declined before use.
    assert!(!unsafe { crate::quotient(std::ptr::null(), &mut a) });
    assert_eq!(decline_count("quotient", "shape") - shape, 1);
    assert_eq!(a, orig);
    assert!(!crate::set_min_k(crate::MAX_LOG_N + 1));
    crate::reset_min_k();
    assert!(crate::end_request());
}

/// The size class of an MSM length: log2 rounded to the nearest integer on
/// the log scale.
#[test]
fn lg2_nearest_rounds_on_the_log_scale() {
    for k in 2..40u32 {
        let p = 1usize << k;
        assert_eq!(crate::lg2_nearest(p), k);
        assert_eq!(crate::lg2_nearest(p - 1), k);
        assert_eq!(crate::lg2_nearest(p + p / 3), k);
    }
    // the boundary of class 13: 2^12.5 = 5792.6
    assert_eq!(crate::lg2_nearest(5793), 12);
    assert_eq!(crate::lg2_nearest(5794), 13);
}

/// Without an explicit threshold each class takes its measured default; an
/// explicit one applies to every class; an MSM below the one-shot threshold
/// is served from a registered table and declines without one.
#[test]
fn per_class_thresholds() {
    use group::Curve;
    use midnight_proofs::gpu::adapter::{affine, from_jacobian, scalar, scalars, scalars_mut};
    let _g = serial();
    require_device();
    crate::reset_min_k();
    if std::env::var_os("MIDNIGHT_METAL_MIN_K").is_some() {
        eprintln!("MIDNIGHT_METAL_MIN_K is set: the defaults are not in force; skipped");
        return;
    }
    assert_eq!(crate::thresholds(), crate::DEFAULT_THRESHOLDS);
    assert_eq!(crate::min_k(), 13);
    let d = crate::DEFAULT_THRESHOLDS;
    assert_eq!(
        (d.fft, d.ifft, d.msm_table, d.msm_one_shot, d.quotient),
        (15, 15, 13, None, 14)
    );

    // a transform below its threshold declines, unchanged
    let log_n = 14;
    let input = stock::scalars(1 << log_n, 3);
    let mut a = input.clone();
    let below = decline_count("fft", "below-min-k");
    assert!(!crate::fft(
        scalars_mut(&mut a),
        scalar(stock::root(log_n)),
        log_n
    ));
    assert_eq!(a, input);
    assert_eq!(decline_count("fft", "below-min-k") - below, 1);

    // 2^14 terms: an MSM without a table declines `unregistered-bases`
    // before any device work and any G1 check; the same MSM on a prefix of
    // a registered table is served from it
    let points = stock::points(3 << 13, 4);
    let m = 1 << 14;
    let (cached, one_shot, uploads) = crate::msm_counts();
    let checks = crate::one_shot_g1_checks();
    let unregistered = decline_count("msm", "unregistered-bases");
    assert!(!crate::msm_sized(m, affine(&points[..m])));
    assert!(crate::msm_g1(scalars(&input), affine(&points[..m])).is_none());
    assert_eq!(decline_count("msm", "unregistered-bases") - unregistered, 1);
    assert_eq!(crate::msm_counts().1, one_shot, "no one-shot device MSM");
    assert_eq!(crate::one_shot_g1_checks(), checks, "no G1 check");
    let h = crate::register_srs(affine(&points));
    assert_eq!(h.device_bytes(), 0, "a registration is host memory");
    let r = crate::msm_g1(scalars(&input), affine(&points[..m])).expect("served from the table");
    assert_eq!(
        from_jacobian(r).to_affine(),
        stock::msm(&input, &points[..m])
    );
    let (cached2, one_shot2, uploads2) = crate::msm_counts();
    assert_eq!(
        (cached2 - cached, one_shot2 - one_shot, uploads2 - uploads),
        (1, 0, 1)
    );
    assert_eq!(h.device_bytes(), (points.len() * 96) as u64);
    // the table threshold goes by the size class of the length (its log2
    // rounded to the nearest integer): 2^13 - 1 is taken, a length of class
    // 12 is not, even from a table
    let below = (1 << 13) - 1;
    let r = crate::msm_g1(scalars(&input[..below]), affine(&points[..below]))
        .expect("class 13, served from the table");
    assert_eq!(
        from_jacobian(r).to_affine(),
        stock::msm(&input[..below], &points[..below])
    );
    let quarter = 1 << 12;
    assert!(crate::msm_g1(scalars(&input[..quarter]), affine(&points[..quarter])).is_none());
    assert_eq!(
        crate::msm_admission(5793, 5793, crate::MAX_MSM_LOG),
        Err("below-min-k")
    );
    assert_eq!(crate::msm_admission(5794, 5794, crate::MAX_MSM_LOG), Ok(()));
    drop(h);

    // an explicit threshold applies to every class, and opens the one-shot
    // MSM
    assert!(crate::set_min_k(10));
    assert_eq!(crate::thresholds(), crate::Thresholds::all(10));
    assert_eq!(crate::thresholds().msm_one_shot, Some(10));
    assert_eq!(crate::min_k(), 10);
    assert!(crate::msm_sized(m, affine(&points[..m])));
    let mut a = input;
    assert!(crate::fft(
        scalars_mut(&mut a),
        scalar(stock::root(log_n)),
        log_n
    ));
    crate::reset_min_k();
    assert_eq!(crate::thresholds(), crate::DEFAULT_THRESHOLDS);
    assert!(crate::end_request());
}

/// The read-back check of the erasure: a clean request reads back zero; a
/// non-zero byte planted after the zeroing is found and fails the report.
#[test]
fn erase_report_reads_back_every_region() {
    let _g = serial();
    require_device();
    assert!(crate::end_request());
    let fill = |n: usize| {
        let mut s = ctx::lock().unwrap();
        for (i, key) in ["test-erase-a", "test-erase-b"].into_iter().enumerate() {
            s.scratch(key, n << i).unwrap().as_mut_slice::<u8>().fill(0x5c);
        }
    };
    fill(4096 + 3);
    let w = crate::end_request_report(true).expect("every region reads back zero");
    // the per-request regions, then (check mode) the idle buffers of the
    // pool, read back before them
    assert_eq!(
        (
            w.regions - w.pool_regions,
            w.checked_regions - w.pool_regions
        ),
        (2, 2)
    );
    assert_eq!(w.checked_regions, w.regions);
    assert_eq!(w.bytes - w.pool_bytes, 3 * (4096 + 3) as u64);
    assert!(w.checked && w.nonzero_regions == 0 && w.check_failures == 0);
    assert_eq!(w.erasure_failures, 0);
    // without the check nothing is read back, the pool is not counted
    fill(64);
    let w = crate::end_request_report(false).unwrap();
    assert_eq!((w.regions, w.checked, w.checked_regions), (2, false, 0));
    assert_eq!((w.pool_regions, w.pool_bytes), (0, 0));
    // a non-zero byte planted after the zeroing through the raw pointer
    // (the buffer's dirty flag is not set: its drop would not zero it)
    fill(64);
    ctx::set_plant_nonzero(true);
    let r = crate::end_request_report(true);
    ctx::set_plant_nonzero(false);
    let w = r.expect_err("a non-zero region fails the report");
    assert_eq!(
        (
            w.regions - w.pool_regions,
            w.nonzero_regions,
            w.check_failures
        ),
        (2, 1, 0)
    );
    // the region was released all the same: not kept as scratch, not in the
    // pool, and every idle buffer of the pool reads zero
    assert_eq!(ctx::lock().unwrap().scratch_bytes(), 0);
    let planted = ctx::last_planted();
    assert_ne!(planted, 0);
    assert!(
        !ctx::idle_pool_addresses().contains(&planted),
        "the failed region is not in the pool"
    );
    assert!(ctx::idle_pool_reads_zero());
    assert!(ctx::request_buffers().is_empty());
    let w = crate::end_request_report(true).expect("clean afterwards");
    assert_eq!(w.nonzero_regions, 0);
}

/// The check mode reads back every idle buffer of the pool (the scoped
/// buffers that held witness values go back there): a non-zero idle buffer
/// fails the report, is zeroed again and leaves the pool.
#[test]
fn erase_report_reads_back_the_idle_pool() {
    let _g = serial();
    require_device();
    assert!(crate::end_request());
    let c = ctx::context().unwrap();
    // a scoped pooled buffer that held data, back in the pool zeroed
    let mut b = c.alloc_pooled(1 << 20).unwrap();
    b.as_mut_slice::<u8>().fill(0x42);
    drop(b);
    let idle = ctx::idle_pool_addresses().len() as u64;
    assert!(idle >= 1);
    let w = crate::end_request_report(true).expect("the idle pool reads zero");
    assert_eq!(
        (w.regions, w.pool_regions, w.checked_regions),
        (idle, idle, idle)
    );
    assert_eq!(w.bytes, w.pool_bytes);
    assert!(w.pool_bytes >= 1 << 20);
    // a stray non-zero byte in an idle buffer
    let planted = ctx::plant_in_idle_pool().expect("an idle buffer");
    let pooled = crate::pool_bytes();
    let w = crate::end_request_report(true).expect_err("a non-zero idle buffer");
    assert_eq!((w.nonzero_regions, w.check_failures), (1, 0));
    assert_eq!(w.pool_regions, idle);
    assert!(!ctx::idle_pool_addresses().contains(&planted));
    assert!(crate::pool_bytes() < pooled, "the buffer left the pool");
    assert!(ctx::idle_pool_reads_zero());
    assert!(crate::end_request_report(true).is_ok());
}

/// An upload shorter than the 16-byte minimum allocation copies exactly its
/// bytes (no read past the caller's data) and the rest reads zero.
#[test]
fn short_uploads_copy_exactly_their_bytes() {
    require_device();
    let c = ctx::context().unwrap();
    for len in [1usize, 4, 15] {
        let data: Vec<u8> = (1..=len as u8).collect();
        let b = c.upload(&data).unwrap();
        assert_eq!(b.len_bytes(), len);
        assert!(b.alloc_bytes() >= 16);
        let mut back = vec![0u8; len];
        b.read(&mut back);
        assert_eq!(back, data, "{len} bytes");
        // SAFETY: the allocation is `alloc_bytes` long; nothing writes it
        let whole =
            unsafe { std::slice::from_raw_parts(b.as_slice::<u8>().as_ptr(), b.alloc_bytes()) };
        assert!(
            whole[len..].iter().all(|x| *x == 0),
            "the tail of {len} bytes"
        );
    }
}

/// `settle_releases` returns, for every caller at the same time, only once
/// every release made before it has completed.
#[test]
fn settle_releases_waits_for_every_caller() {
    let _g = serial();
    require_device();
    assert!(crate::end_request());
    let c = ctx::context().unwrap();
    for _ in 0..2 {
        let bufs: Vec<_> = (0..2)
            .map(|_| {
                let mut b = c.alloc_pooled(32 << 20).unwrap();
                b.as_mut_slice::<u8>().fill(0x11);
                b
            })
            .collect();
        ctx::release_later(bufs);
    }
    let start = std::sync::Barrier::new(2);
    std::thread::scope(|s| {
        let hs: Vec<_> = (0..2)
            .map(|_| {
                s.spawn(|| {
                    start.wait();
                    ctx::settle_releases();
                    !ctx::releases_pending()
                })
            })
            .collect();
        for h in hs {
            assert!(
                h.join().unwrap(),
                "a caller returned before the zeroing ended"
            );
        }
    });
    assert!(ctx::idle_pool_reads_zero());
    assert!(crate::end_request());
}

/// A zeroing that panics is completed on the calling thread; the buffer is
/// released, not pooled, and the next report fails with the fault counted.
#[test]
fn a_panicking_zeroing_is_completed_and_reported() {
    let _g = serial();
    require_device();
    assert!(crate::end_request());
    let c = ctx::context().unwrap();
    let mut b = c.alloc_pooled(1 << 20).unwrap();
    b.as_mut_slice::<u8>().fill(0x66);
    let at = b.as_slice::<u8>().as_ptr() as usize;
    ctx::set_zeroing_panic(at);
    ctx::release_later(vec![b]);
    ctx::settle_releases();
    ctx::set_zeroing_panic(0);
    assert!(!ctx::idle_pool_addresses().contains(&at), "not pooled");
    assert!(ctx::idle_pool_reads_zero());
    let w = crate::end_request_report(false).expect_err("the fault is reported");
    assert_eq!(w.erasure_failures, 1);
    assert!(crate::end_request());
}

#[test]
fn off_switch_declines_no_device() {
    const CHILD: &str = "MIDNIGHT_METAL_OFF_CHILD";
    if std::env::var_os(CHILD).is_some() {
        assert!(crate::present().is_none());
        assert!(!crate::device());
        assert!(crate::mem_info().is_none());
        assert!(crate::set_min_k(1));
        let mut a = vec![[0u64; 4]; 2];
        assert!(!crate::fft(&mut a, [0; 4], 1));
        assert!(crate::msm_g1(&a, &[[0u64; 12]; 2]).is_none());
        let d = crate::declines();
        assert!(d.contains(&("fft", "no-device", 1)), "{d:?}");
        assert!(d.contains(&("msm", "no-device", 1)), "{d:?}");
        return;
    }
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "tests::off_switch_declines_no_device"])
        .env(CHILD, "1")
        .env("MIDNIGHT_METAL", "off")
        .stdout(std::process::Stdio::null())
        .status()
        .unwrap();
    assert!(status.success());
}

// --- hook calls outside a request, panics at the hook boundary ---------------

/// The per-request buffers that exist now, for messages.
fn request_buffer_keys() -> Vec<&'static str> {
    ctx::request_buffers().into_iter().map(|r| r.0).collect()
}

/// Nothing a hook call left behind: no per-request buffer exists, and every
/// idle buffer of the pool reads zero.
fn assert_wiped(after: &str) {
    let left = ctx::request_buffers();
    assert!(left.is_empty(), "after {after}: {left:?}");
    assert!(
        ctx::idle_pool_reads_zero(),
        "after {after}: an idle buffer is not zero"
    );
}

/// Without `begin_request` every hook call zeroes its per-request buffers
/// before it returns; with a request open they are kept between calls and
/// zeroed at `end_request`.
#[test]
fn hooks_outside_a_request_wipe_before_they_return() {
    use midnight_proofs::gpu::{
        adapter::{affine, scalar, scalars, scalars_mut},
        test_switches::RandomCircuit,
    };
    let _g = serial();
    require_device();
    assert!(crate::end_request());
    assert!(!ctx::request_open());
    assert!(crate::set_min_k(1));
    // two passes: data and scratch buffers
    let log_n = 12;
    let input = stock::scalars(1 << log_n, 21);
    let w = scalar(stock::root(log_n));

    let mut a = input.clone();
    assert!(crate::fft(scalars_mut(&mut a), w, log_n));
    assert_wiped("fft");
    let points = stock::points(1 << 10, 22);
    let r = crate::msm_g1(scalars(&input[..1 << 10]), affine(&points));
    assert!(r.is_some(), "{:?}", crate::declines());
    assert_wiped("msm_g1");
    install_checked_quotient();
    let q = crate::stats().quotient.served;
    let (_, verified) = RandomCircuit::new(8, true, 9).prove();
    assert!(verified);
    assert_eq!(
        crate::stats().quotient.served - q,
        1,
        "{:?}",
        crate::declines()
    );
    assert_wiped("quotient");

    // inside a request the scratch stays for the next call
    crate::begin_request();
    assert!(ctx::request_open());
    let mut a = input.clone();
    assert!(crate::fft(scalars_mut(&mut a), w, log_n));
    let first = ctx::request_buffers();
    assert!(
        first.iter().any(|r| r.0 == "ntt-data"),
        "{:?}",
        request_buffer_keys()
    );
    let mut a = input.clone();
    assert!(crate::fft(scalars_mut(&mut a), w, log_n));
    let second = ctx::request_buffers();
    let addr =
        |rows: &[(&str, usize, bool, usize)]| rows.iter().find(|r| r.0 == "ntt-data").map(|r| r.3);
    assert_eq!(addr(&first), addr(&second), "the same buffer, reused");
    assert!(crate::msm_g1(scalars(&input[..1 << 10]), affine(&points)).is_some());
    assert!(
        ctx::request_buffers().len() > second.len(),
        "the msm's scratch"
    );
    assert!(crate::end_request());
    assert!(!ctx::request_open());
    assert_wiped("end_request");
    crate::reset_min_k();
}

/// A panic inside a backend is a counted decline `panic`; the caller's
/// buffer is unchanged and the call still wipes its scratch (no request).
#[test]
fn a_backend_panic_is_a_counted_decline() {
    use midnight_proofs::gpu::adapter::{affine, scalar, scalars, scalars_mut};
    let _g = serial();
    require_device();
    assert!(crate::end_request());
    assert!(crate::set_min_k(1));
    let log_n = 12;
    let n = 1usize << log_n;
    let input = stock::scalars(n, 31);
    let w = stock::root(log_n);
    let wi = ff::Field::invert(&w).unwrap();
    let div = ff::Field::invert(&midnight_curves::Fq::from(n as u64)).unwrap();

    let panics = crate::hook_panics();
    // fft and ifft: the panic comes with the request buffers filled
    for class in ["fft", "ifft"] {
        let before = decline_count(class, "panic");
        crate::set_hook_panic(class, true);
        let mut a = input.clone();
        let ok = if class == "fft" {
            crate::fft(scalars_mut(&mut a), scalar(w), log_n)
        } else {
            crate::ifft(scalars_mut(&mut a), scalar(wi), log_n, scalar(div))
        };
        crate::set_hook_panic(class, false);
        assert!(!ok);
        assert_eq!(a, input, "{class}: the caller's buffer is unchanged");
        assert_eq!(decline_count(class, "panic") - before, 1);
        assert_wiped(class);
    }
    // msm: the panic comes after the device MSM used its scratch
    let points = stock::points(1 << 10, 32);
    let before = decline_count("msm", "panic");
    crate::set_hook_panic("msm", true);
    let r = crate::msm_g1(scalars(&input[..1 << 10]), affine(&points));
    crate::set_hook_panic("msm", false);
    assert!(r.is_none());
    assert_eq!(decline_count("msm", "panic") - before, 1);
    assert_wiped("msm_g1");
    // quotient: the panic comes before the job is read (so a placeholder
    // job is never dereferenced)
    let before = decline_count("quotient", "panic");
    let placeholder = 0u64;
    let mut out = vec![[7u64; 4]; 4];
    crate::set_hook_panic("quotient", true);
    // SAFETY: the test switch panics before the job is read.
    let ok = unsafe {
        crate::quotient(
            &placeholder as *const u64 as *const std::ffi::c_void,
            &mut out,
        )
    };
    crate::set_hook_panic("quotient", false);
    assert!(!ok);
    assert_eq!(out, vec![[7u64; 4]; 4]);
    assert_eq!(decline_count("quotient", "panic") - before, 1);
    assert_wiped("quotient");
    assert_eq!(crate::hook_panics() - panics, 4);
    // the hooks serve again afterwards
    let mut a = input;
    assert!(crate::fft(scalars_mut(&mut a), scalar(w), log_n));
    crate::reset_min_k();
    assert!(crate::end_request());
}

/// The `ntt_pass` pipeline takes every threadgroup size the plan uses, so
/// no transform size is declined for good by `dispatch_groups`.
#[test]
fn ntt_pass_takes_the_planned_threadgroups() {
    use objc2_metal::MTLComputePipelineState;
    require_device();
    let p = ctx::context().unwrap().pipeline("ntt_pass").unwrap();
    let max = p.maxTotalThreadsPerThreadgroup();
    for log_n in 1..=crate::MAX_LOG_N {
        for t in crate::ntt::pass_group_threads(log_n) {
            assert!(
                t >= 1 && t <= max,
                "2^{log_n}: {t} threads, pipeline max {max}"
            );
        }
    }
}
