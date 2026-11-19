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
const OP_MUL_CIOS32: u32 = 6;
const OP_MUL_MULHI32: u32 = 7;
const OP_MUL_FIPS32: u32 = 8;
const OP_MUL_CIOS64: u32 = 9;
const OP_IS_REDUCED: u32 = 10;
const OP_ONE: u32 = 11;
const OP_EQ: u32 = 12;

const MUL_OPS: [(u32, &str); 5] = [
    (OP_MUL, "default"),
    (OP_MUL_CIOS32, "cios32"),
    (OP_MUL_MULHI32, "mulhi32"),
    (OP_MUL_FIPS32, "fips32"),
    (OP_MUL_CIOS64, "cios64"),
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
        OP_MUL | OP_MUL_CIOS32 | OP_MUL_MULHI32 | OP_MUL_FIPS32 | OP_MUL_CIOS64 => {
            fr2(blst_fr_mul, a, b)
        }
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
        OP_MUL | OP_MUL_CIOS32 | OP_MUL_MULHI32 | OP_MUL_FIPS32 | OP_MUL_CIOS64 => {
            fp2(blst_fp_mul, a, b)
        }
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
fn fr_add_sub_neg_dbl() {
    for (op, name) in [
        (OP_ADD, "add"),
        (OP_SUB, "sub"),
        (OP_NEG, "neg"),
        (OP_DBL, "dbl"),
    ] {
        check_fr(op, name);
    }
}

#[test]
fn fr_mul_square() {
    for (op, name) in MUL_OPS {
        check_fr(op, name);
    }
    check_fr(OP_SQR, "sqr");
}

#[test]
fn fp_add_sub_neg_dbl() {
    for (op, name) in [
        (OP_ADD, "add"),
        (OP_SUB, "sub"),
        (OP_NEG, "neg"),
        (OP_DBL, "dbl"),
    ] {
        check_fp(op, name);
    }
}

#[test]
fn fp_mul_square() {
    for (op, name) in MUL_OPS {
        check_fp(op, name);
    }
    check_fp(OP_SQR, "sqr");
}

#[test]
fn one_and_eq() {
    check_fr(OP_ONE, "one");
    check_fp(OP_ONE, "one");
    check_fr(OP_EQ, "eq");
    check_fp(OP_EQ, "eq");
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

// --- context tests ------------------------------------------------------------

#[test]
fn context_and_info() {
    require_device();
    let info = crate::present().unwrap();
    assert!(info.unified_memory);
    assert!(info.max_buffer_length > 0 && info.recommended_max_working_set > 0);
    assert!(crate::library_source().is_some());
    let (free, total) = crate::mem_info().unwrap();
    assert!(free <= total && total as u64 == info.recommended_max_working_set);
    eprintln!(
        "{info:#?}\nlibrary: {:?}\nmem_info: {:?}",
        crate::library_source(),
        (free, total)
    );
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

#[test]
fn request_buffers_and_byte_accounting() {
    require_device();
    let mut s = ctx::lock().unwrap();
    let (before, _) = crate::device_bytes();
    {
        let b = s.scratch("test-scratch", 1 << 20).unwrap();
        b.as_mut_slice::<u64>().fill(0xdead_beef);
    }
    // the same key, not longer: the same buffer with its contents
    let b = s.scratch("test-scratch", 1 << 19).unwrap();
    assert!(b.as_slice::<u64>().iter().all(|v| *v == 0xdead_beef));
    assert!(s.scratch_bytes() >= 1 << 20);
    let (mid, peak) = crate::device_bytes();
    assert!(mid >= before + (1 << 20) && peak >= mid);
    {
        let owned = s.ctx().alloc(1 << 16).unwrap();
        assert!(crate::device_bytes().0 >= mid + (1 << 16));
        drop(owned);
    }
    assert_eq!(crate::device_bytes().0, mid);
    drop(s);
    assert!(crate::end_request());
    // under the lock no other test holds a buffer (each drops its buffers
    // before its session)
    let s = ctx::lock().unwrap();
    assert!(s.get_scratch("test-scratch").is_none());
    assert_eq!(s.scratch_bytes(), 0);
    assert_eq!(crate::device_bytes().0, 0);
}

#[test]
fn entry_points_decline_not_implemented() {
    require_device();
    assert!(crate::set_min_k(1));
    let n = 1 << 4;
    let mut a = vec![fr_one_mont(); n];
    let orig = a.clone();
    assert!(!crate::fft(&mut a, fr_one_mont(), 4));
    assert!(!crate::ifft(&mut a, fr_one_mont(), 4, fr_one_mont()));
    assert_eq!(a, orig);
    assert!(crate::msm_g1(&a, &vec![[0u64; 12]; n]).is_none());
    // SAFETY: a null job is declined before use.
    assert!(!unsafe { crate::quotient(std::ptr::null(), &mut a) });
    let d = crate::declines();
    for class in ["fft", "ifft", "msm"] {
        assert!(
            d.iter().any(|(c, r, k)| *c == class && *r == "not-implemented" && *k > 0),
            "{d:?}"
        );
    }
    assert!(d.iter().any(|(c, r, _)| *c == "quotient" && *r == "shape"));
    // shapes and unreduced inputs are declined before the kernel module
    assert!(!crate::fft(&mut a[..15], fr_one_mont(), 4));
    let mut bad = orig;
    bad[3] = FR_MODULUS;
    assert!(!crate::fft(&mut bad, fr_one_mont(), 4));
    let d = crate::declines();
    assert!(d.iter().any(|(c, r, _)| *c == "fft" && *r == "shape"));
    assert!(d.iter().any(|(c, r, _)| *c == "fft" && *r == "unreduced"));
    assert!(!crate::set_min_k(crate::MAX_LOG_N + 1));
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

// --- measurements (run with --ignored --nocapture) ------------------------------

/// Multiplication strategies: `2^20` threads, each a chain of 64 dependent
/// multiplications (2^26 per run), five runs each, GPU time.
#[test]
#[ignore]
fn bench_mul_strategies() {
    require_device();
    const N: usize = 1 << 20;
    const REPS: u32 = 64;
    const RUNS: usize = 5;
    let mut rng = ChaCha20Rng::seed_from_u64(1);
    let fr_a: Vec<Scalar> = (0..N).map(|_| random_below(&mut rng, &FR_MODULUS)).collect();
    let fr_b: Vec<Scalar> = (0..N).map(|_| random_below(&mut rng, &FR_MODULUS)).collect();
    let fp_a: Vec<FpLimbs> = (0..N).map(|_| random_below(&mut rng, &FP_MODULUS)).collect();
    let fp_b: Vec<FpLimbs> = (0..N).map(|_| random_below(&mut rng, &FP_MODULUS)).collect();
    let s = ctx::lock().unwrap();
    let c = s.ctx();
    for (field, a, b, bytes) in [
        (
            "fr",
            c.upload(&fr_a).unwrap(),
            c.upload(&fr_b).unwrap(),
            32usize,
        ),
        ("fp", c.upload(&fp_a).unwrap(), c.upload(&fp_b).unwrap(), 48),
    ] {
        let mut first: Option<Vec<u64>> = None;
        for strat in ["cios32", "mulhi32", "fips32", "cios64"] {
            let kernel = format!("field_bench_{field}_{strat}");
            let out = c.alloc(N * bytes).unwrap();
            let n = N as u32;
            let mut times = Vec::new();
            for _ in 0..RUNS + 1 {
                let (_, ns) = c
                    .run(|cmd| {
                        cmd.dispatch(
                            &kernel,
                            N,
                            &[
                                Arg::Buffer(&a),
                                Arg::Buffer(&b),
                                Arg::Buffer(&out),
                                Arg::value(&REPS),
                                Arg::value(&n),
                            ],
                        )
                    })
                    .unwrap();
                times.push(ns);
            }
            // the first run includes the pipeline's first use
            let mut t = times[1..].to_vec();
            t.sort();
            let muls = (N as u64) * REPS as u64;
            let res: Vec<u64> = out.as_slice::<u64>().to_vec();
            match &first {
                None => first = Some(res),
                Some(f) => assert_eq!(f, &res, "{kernel} disagrees with cios32"),
            }
            println!(
                "{field} {strat:8} min {:7.3} ms  median {:7.3} ms  {:6.2} Gmul/s (min)",
                t[0] as f64 / 1e6,
                t[t.len() / 2] as f64 / 1e6,
                muls as f64 / t[0] as f64
            );
        }
    }
}

/// How long one command buffer may run: a spinning kernel held for growing
/// durations, reporting the GPU time or the error.
#[test]
#[ignore]
fn command_buffer_duration() {
    require_device();
    let s = ctx::lock().unwrap();
    let c = s.ctx();
    let threads: u32 = 1 << 16;
    let out = c.alloc(threads as usize * 4).unwrap();
    // calibrate
    let mut iters: u32 = 1 << 16;
    let (_, ns) = c
        .run(|cmd| {
            cmd.dispatch(
                "diag_spin",
                threads as usize,
                &[Arg::Buffer(&out), Arg::value(&iters), Arg::value(&threads)],
            )
        })
        .unwrap();
    let per_iter = ns as f64 / iters as f64;
    let secs: Vec<f64> = std::env::var("MIDNIGHT_METAL_SPIN_SECS")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| vec![0.5, 1.0, 2.0, 5.0, 10.0]);
    for target in secs {
        iters = ((target * 1e9 / per_iter) as u64).min(u32::MAX as u64) as u32;
        let t0 = std::time::Instant::now();
        let r = c.run(|cmd| {
            cmd.dispatch(
                "diag_spin",
                threads as usize,
                &[Arg::Buffer(&out), Arg::value(&iters), Arg::value(&threads)],
            )
        });
        let wall = t0.elapsed().as_secs_f64();
        match r {
            Ok((_, ns)) => println!(
                "target {target:5.1} s: ok, GPU {:.3} s, wall {wall:.3} s",
                ns as f64 / 1e9
            ),
            Err(e) => println!("target {target:5.1} s: {e} (wall {wall:.3} s)"),
        }
    }
}
