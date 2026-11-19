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

//! The G1 MSM on the device: the curve kernels against blst, the MSM against
//! the stock CPU MSM (`msm_pippenger`) as group elements, registered tables,
//! and the hook through a KZG commitment.
//!
//! The differential runs sizes `2^14 ..= 2^20` in a release build (`cargo
//! test --release`) and `2^14 ..= 2^16` in a debug build, where the CPU
//! reference and the point generation are slow; a release build also runs
//! one `2^23` MSM (several command buffers). The measurements are
//! `#[ignore]`d: `cargo test --release --test msm -- --ignored --nocapture`.

#![cfg(target_os = "macos")]

use std::{
    sync::{Mutex, MutexGuard, OnceLock},
    time::Instant,
};

use blst::{
    blst_fp, blst_fp_mul, blst_fp_sqr, blst_p1, blst_p1_add_or_double, blst_p1_affine,
    blst_p1_cneg, blst_p1_double, blst_p1_from_affine, blst_p1_is_equal,
};
use ff::Field;
use group::{Curve, Group, prime::PrimeCurveAffine};
use midnight_curves::{Bls12, Fq, G1Affine, G1Projective, msm::msm_pippenger};
use midnight_metal::{
    G1AffineLimbs, Scalar,
    ctx::{self, Arg},
};
use midnight_proofs::{
    gpu::{GpuHooks, adapter, install, test_switches::without_hooks},
    poly::{
        Coeff, EvaluationDomain, PolynomialLabel,
        commitment::PolynomialCommitmentScheme,
        kzg::{KZGCommitmentScheme, params::ParamsKZG},
    },
};
use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;
use rayon::prelude::*;

/// The tests share process-wide counters and the device: one at a time.
fn serial() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn require_device() {
    assert!(midnight_metal::device(), "no usable Metal device");
}

/// `n` distinct random points: chunks of consecutive multiples `P + i Q`
/// from a random start, normalized together (fast for large `n`).
fn random_points(n: usize, seed: u64) -> Vec<G1Affine> {
    const CHUNK: usize = 1 << 12;
    let mut out = vec![G1Affine::identity(); n];
    out.par_chunks_mut(CHUNK).enumerate().for_each(|(c, chunk)| {
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ ((c as u64) << 20));
        let step = G1Projective::random(&mut rng);
        let mut p = G1Projective::random(&mut rng);
        let proj: Vec<G1Projective> = (0..chunk.len())
            .map(|_| {
                p += step;
                p
            })
            .collect();
        G1Projective::batch_normalize(&proj, chunk);
    });
    out
}

/// The largest point set of the differential, made once.
fn points_cache() -> &'static [G1Affine] {
    static P: OnceLock<Vec<G1Affine>> = OnceLock::new();
    P.get_or_init(|| random_points(1 << 20, 1))
}

fn random_scalars(n: usize, seed: u64) -> Vec<Fq> {
    (0..n)
        .into_par_iter()
        .chunks(1 << 12)
        .flat_map_iter(|c| {
            let mut rng = ChaCha8Rng::seed_from_u64(seed ^ ((c[0] as u64) << 16));
            c.into_iter().map(move |_| Fq::random(&mut rng))
        })
        .collect()
}

fn device_msm(s: &[Fq], b: &[G1Affine]) -> G1Projective {
    let r = midnight_metal::msm_g1(adapter::scalars(s), adapter::affine(b));
    adapter::from_jacobian(r.unwrap_or_else(|| {
        panic!(
            "the device declined an MSM of {}: {:?}",
            s.len(),
            midnight_metal::declines()
        )
    }))
}

// ---------------------------------------------------------------------------
// Curve kernels.

type Xyzz = [u64; 24];

const OP_MADD: u32 = 0;
const OP_ADD: u32 = 1;
const OP_DBL: u32 = 2;
const OP_FROM_AFFINE: u32 = 3;

fn fp(l: &[u64]) -> blst_fp {
    blst_fp {
        l: l.try_into().unwrap(),
    }
}

fn fmul(a: &blst_fp, b: &blst_fp) -> blst_fp {
    let mut r = blst_fp::default();
    // SAFETY: valid field elements.
    unsafe { blst_fp_mul(&mut r, a, b) };
    r
}

fn fsqr(a: &blst_fp) -> blst_fp {
    let mut r = blst_fp::default();
    // SAFETY: valid field element.
    unsafe { blst_fp_sqr(&mut r, a) };
    r
}

fn p1_of_affine(a: &G1AffineLimbs) -> blst_p1 {
    let aff = blst_p1_affine {
        x: fp(&a[..6]),
        y: fp(&a[6..]),
    };
    let mut p = blst_p1::default();
    // SAFETY: valid affine point (all zero is infinity).
    unsafe { blst_p1_from_affine(&mut p, &aff) };
    p
}

/// The XYZZ form of an affine point scaled by `z`: `(x z^2, y z^3, z^3,
/// z^2)`; infinity is all zero.
fn xyzz_of(a: &G1AffineLimbs, z: &blst_fp) -> Xyzz {
    let mut r = [0u64; 24];
    if a.iter().all(|&w| w == 0) {
        return r;
    }
    let zz = fsqr(z);
    let zzz = fmul(&zz, z);
    r[..6].copy_from_slice(&fmul(&fp(&a[..6]), &zz).l);
    r[6..12].copy_from_slice(&fmul(&fp(&a[6..]), &zzz).l);
    r[12..18].copy_from_slice(&zzz.l);
    r[18..].copy_from_slice(&zz.l);
    r
}

/// An XYZZ record as a blst point (`(X ZZ, Y ZZZ, ZZ)`); checks the
/// representation invariants (`ZZ^3 = ZZZ^2`; infinity has both zero).
fn p1_of_xyzz(r: &Xyzz) -> blst_p1 {
    let (x, y, zzz, zz) = (fp(&r[..6]), fp(&r[6..12]), fp(&r[12..18]), fp(&r[18..]));
    let zz_zero = zz.l.iter().all(|&w| w == 0);
    let zzz_zero = zzz.l.iter().all(|&w| w == 0);
    assert_eq!(zz_zero, zzz_zero, "ZZ and ZZZ disagree on infinity");
    assert_eq!(fmul(&fsqr(&zz), &zz).l, fsqr(&zzz).l, "ZZ^3 != ZZZ^2");
    blst_p1 {
        x: fmul(&x, &zz),
        y: fmul(&y, &zzz),
        z: zz,
    }
}

fn padd(a: &blst_p1, b: &blst_p1) -> blst_p1 {
    let mut r = blst_p1::default();
    // SAFETY: valid points.
    unsafe { blst_p1_add_or_double(&mut r, a, b) };
    r
}

fn pneg(a: &blst_p1) -> blst_p1 {
    let mut r = *a;
    // SAFETY: valid point.
    unsafe { blst_p1_cneg(&mut r, true) };
    r
}

fn pdbl(a: &blst_p1) -> blst_p1 {
    let mut r = blst_p1::default();
    // SAFETY: valid point.
    unsafe { blst_p1_double(&mut r, a) };
    r
}

fn peq(a: &blst_p1, b: &blst_p1) -> bool {
    // SAFETY: valid points.
    unsafe { blst_p1_is_equal(a, b) }
}

/// Runs `msm_test_g1` on the cases.
fn g1_kernel(op: u32, a: &[Xyzz], b: &[Xyzz], b_aff: &[G1AffineLimbs], flags: &[u32]) -> Vec<Xyzz> {
    let n = a.len() as u32;
    let s = ctx::lock().unwrap();
    let c = s.ctx();
    let (ba, bb, bf, bl) = (
        c.upload(a).unwrap(),
        c.upload(b).unwrap(),
        c.upload(b_aff).unwrap(),
        c.upload(flags).unwrap(),
    );
    let out = c.alloc(size_of_val(a)).unwrap();
    c.run(|cmd| {
        cmd.dispatch(
            "msm_test_g1",
            n as usize,
            &[
                Arg::Buffer(&ba),
                Arg::Buffer(&bb),
                Arg::Buffer(&bf),
                Arg::Buffer(&bl),
                Arg::Buffer(&out),
                Arg::value(&op),
                Arg::value(&n),
            ],
        )
    })
    .unwrap();
    out.as_slice::<Xyzz>().to_vec()
}

#[test]
fn g1_kernels_match_blst() {
    let _g = serial();
    require_device();
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let pts: Vec<G1Affine> = (0..64).map(|_| G1Projective::random(&mut rng).to_affine()).collect();
    let limbs = adapter::affine(&pts);
    let z = |i: usize| fp(&limbs[(i + 17) % limbs.len()][..6]);
    let lim = |p: &G1Affine| adapter::affine(std::slice::from_ref(p))[0];
    let inf = G1Affine::identity();

    // (a, b) pairs: random, a + inf, inf + b, inf + inf, P + P, P + (-P).
    let mut pts_pairs: Vec<(G1Affine, G1Affine)> = Vec::new();
    for i in 0..32 {
        pts_pairs.push((pts[i], pts[i + 32]));
    }
    for p in &pts[..4] {
        pts_pairs.push((*p, inf));
        pts_pairs.push((inf, *p));
        pts_pairs.push((*p, *p));
        pts_pairs.push((*p, -*p));
    }
    pts_pairs.push((inf, inf));
    let pairs: Vec<(G1AffineLimbs, G1AffineLimbs)> =
        pts_pairs.iter().map(|(a, b)| (lim(a), lim(b))).collect();
    let m = pairs.len();

    let a: Vec<Xyzz> = pairs.iter().enumerate().map(|(i, p)| xyzz_of(&p.0, &z(i))).collect();
    // b in XYZZ with a different scale, so P + P is not a bitwise copy
    let b: Vec<Xyzz> = pairs.iter().enumerate().map(|(i, p)| xyzz_of(&p.1, &z(i + 5))).collect();
    let b_aff: Vec<G1AffineLimbs> = pairs.iter().map(|p| p.1).collect();
    let pa: Vec<blst_p1> = pairs.iter().map(|p| p1_of_affine(&p.0)).collect();
    let pb: Vec<blst_p1> = pairs.iter().map(|p| p1_of_affine(&p.1)).collect();

    // the inputs themselves are what they claim
    for i in 0..m {
        assert!(peq(&p1_of_xyzz(&a[i]), &pa[i]), "input a {i}");
        assert!(peq(&p1_of_xyzz(&b[i]), &pb[i]), "input b {i}");
    }

    for flag in [0u32, 1] {
        let flags = vec![flag; m];
        let got = g1_kernel(OP_MADD, &a, &b, &b_aff, &flags);
        for i in 0..m {
            let exp = if flag == 1 {
                padd(&pa[i], &pneg(&pb[i]))
            } else {
                padd(&pa[i], &pb[i])
            };
            assert!(peq(&p1_of_xyzz(&got[i]), &exp), "madd {i} flag {flag}");
        }
        let got = g1_kernel(OP_FROM_AFFINE, &a, &b, &b_aff, &flags);
        for i in 0..m {
            let exp = if flag == 1 { pneg(&pb[i]) } else { pb[i] };
            assert!(
                peq(&p1_of_xyzz(&got[i]), &exp),
                "from_affine {i} flag {flag}"
            );
        }
    }
    let flags = vec![0u32; m];
    let got = g1_kernel(OP_ADD, &a, &b, &b_aff, &flags);
    for i in 0..m {
        assert!(peq(&p1_of_xyzz(&got[i]), &padd(&pa[i], &pb[i])), "add {i}");
    }
    let got = g1_kernel(OP_DBL, &a, &b, &b_aff, &flags);
    for i in 0..m {
        assert!(peq(&p1_of_xyzz(&got[i]), &pdbl(&pa[i])), "dbl {i}");
    }
    // an opposite pair gives infinity stored as all zero
    let got = g1_kernel(OP_ADD, &a, &b, &b_aff, &flags);
    for (i, (pa, pb)) in pts_pairs.iter().enumerate() {
        if *pa != inf && *pb == -*pa {
            assert_eq!(got[i], [0u64; 24], "P + (-P) at {i}");
        }
    }
}

/// The CUDA crate's mask vector (gpu/src/msm.rs `mask_vector`).
fn cuda_mask_vector(n: usize) -> Vec<Scalar> {
    let mut x: u64 = 0x2545_f491_4f6c_dd1d;
    let mut next = || {
        x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    (0..n).map(|_| [next(), next(), next(), next() >> 3]).collect()
}

#[test]
fn mask_is_the_cuda_vector() {
    let _g = serial();
    require_device();
    let n = 5000u32;
    let want = cuda_mask_vector(n as usize + 100);
    let s = ctx::lock().unwrap();
    let c = s.ctx();
    let out = c.alloc(n as usize * 32).unwrap();
    let i0 = 100u32;
    c.run(|cmd| {
        cmd.dispatch(
            "msm_test_mask",
            n as usize,
            &[Arg::Buffer(&out), Arg::value(&i0), Arg::value(&n)],
        )
    })
    .unwrap();
    assert_eq!(out.as_slice::<Scalar>(), &want[100..]);
    assert!(want.iter().all(midnight_metal::field::is_reduced));
}

// ---------------------------------------------------------------------------
// The MSM against the CPU.

fn sizes() -> Vec<u32> {
    if cfg!(debug_assertions) {
        (14..=16).collect()
    } else {
        (14..=20).collect()
    }
}

fn check(name: &str, s: &[Fq], b: &[G1Affine]) {
    let before = midnight_metal::stats().msm.served;
    let got = device_msm(s, b);
    let want = msm_pippenger(s, b);
    assert!(
        got == want,
        "{name}: MSM of {} differs from msm_pippenger",
        s.len()
    );
    assert_eq!(midnight_metal::stats().msm.served, before + 1, "{name}");
}

/// Sets `MIDNIGHT_METAL_MSM_CPU_SHARE` (`off`: device only) or removes it.
/// The tests that call it hold [`serial`], and no other thread of this binary
/// reads the environment meanwhile.
fn set_cpu_share(on: bool) {
    // SAFETY: see above.
    unsafe {
        if on {
            std::env::remove_var("MIDNIGHT_METAL_MSM_CPU_SHARE");
        } else {
            std::env::set_var("MIDNIGHT_METAL_MSM_CPU_SHARE", "off");
        }
    }
}

/// The stage report of one MSM, checked against the CPU: whether the CPU took
/// a share.
fn splits(s: &[Fq], b: &[G1Affine]) -> bool {
    let (r, st) = midnight_metal::msm_g1_stages(adapter::scalars(s), adapter::affine(b)).unwrap();
    assert!(adapter::from_jacobian(r) == msm_pippenger(s, b));
    st.device_points < s.len() as u64
}

/// The differential set, with the CPU share on or off. Every case below
/// spreads its special inputs (infinity points, equal and opposite points,
/// repeated scalars) over all indices, so with the split both the device
/// share and the CPU share hold them, and a point of one share repeats in
/// the other (`odd[i] = odd[i / 2]`).
fn differential(split: bool) {
    // an explicit threshold of 2^0: the all-zero and other cases below whose
    // effective size (non-zero scalars) is under the default thresholds
    // still run on the device
    assert!(midnight_metal::set_min_k(0));
    set_cpu_share(split);
    midnight_metal::reset_stats();
    let all = points_cache();
    for k in sizes() {
        let n = 1usize << k;
        let bases = &all[..n];
        let t0 = Instant::now();

        let random = random_scalars(n, k as u64);
        assert_eq!(splits(&random, bases), split, "2^{k}: split");
        check("random", &random, bases);
        let c = Fq::random(ChaCha8Rng::seed_from_u64(99 + k as u64));
        check("all equal", &vec![c; n], bases);
        check("all zero", &vec![Fq::ZERO; n], bases);
        assert!(bool::from(
            device_msm(&vec![Fq::ZERO; n], bases).is_identity()
        ));
        check("all one", &vec![Fq::ONE; n], bases);
        check("all r-1", &vec![-Fq::ONE; n], bases);
        let few = [
            Fq::from(3),
            -Fq::from(5),
            Fq::random(ChaCha8Rng::seed_from_u64(k as u64)),
            Fq::ONE,
        ];
        let repeated: Vec<Fq> = (0..n).map(|i| few[(i * 7 + i / 13) % few.len()]).collect();
        check("repeated", &repeated, bases);

        // infinity points and repeated points (equal and opposite pairs)
        let mut odd = bases.to_vec();
        for i in (0..n).step_by(7) {
            odd[i] = G1Affine::identity();
        }
        for i in (3..n).step_by(11) {
            odd[i] = odd[i / 2];
        }
        for i in (5..n).step_by(17) {
            odd[i] = -odd[i - 1];
        }
        let odd_random = random_scalars(n, 1000 + k as u64);
        assert_eq!(splits(&odd_random, &odd), split, "2^{k}: split");
        check("odd bases, random", &odd_random, &odd);
        check("odd bases, repeated", &repeated, &odd);
        check("odd bases, all one", &vec![Fq::ONE; n], &odd);
        eprintln!("2^{k}: 10 cases in {:.1?}", t0.elapsed());
    }
    assert!(
        midnight_metal::declines().is_empty(),
        "{:?}",
        midnight_metal::declines()
    );
    set_cpu_share(true);
    assert!(midnight_metal::set_min_k(14));
}

#[test]
fn msm_matches_cpu() {
    let _g = serial();
    require_device();
    differential(true);
}

#[test]
fn msm_matches_cpu_device_only() {
    let _g = serial();
    require_device();
    differential(false);
}

/// MSMs issued from rayon workers, as the prover issues them: from the
/// global pool, and from a small pool whose every worker calls at once (they
/// all wait on the device lock while the holder's CPU share runs).
#[test]
fn msm_from_rayon_pool() {
    let _g = serial();
    require_device();
    assert!(midnight_metal::set_min_k(14));
    set_cpu_share(true);
    let n = 1usize << 17;
    let bases = &points_cache()[..n];
    let inputs: Vec<Vec<Fq>> = (0..6).map(|i| random_scalars(n, 300 + i)).collect();
    let want: Vec<G1Projective> = inputs.iter().map(|s| msm_pippenger(s, bases)).collect();
    assert!(splits(&inputs[0], bases));
    let got: Vec<G1Projective> = inputs.par_iter().map(|s| device_msm(s, bases)).collect();
    assert!(got == want);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(3).build().unwrap();
    let got: Vec<G1Projective> =
        pool.install(|| inputs.par_iter().map(|s| device_msm(s, bases)).collect());
    assert!(got == want);
}

/// One size above the measured table, where the command-buffer bound splits
/// the MSM into five chunks: once, in a release build only (the CPU
/// reference and the 2^23 points are slow in a debug build).
#[test]
fn msm_above_the_table_matches_cpu() {
    let _g = serial();
    if cfg!(debug_assertions) {
        eprintln!("skipped in a debug build");
        return;
    }
    require_device();
    assert!(midnight_metal::set_min_k(14));
    let n = 1usize << 23;
    let bases = random_points(n, 23);
    let t = Instant::now();
    check("2^23 random", &random_scalars(n, 23), &bases);
    eprintln!("2^23: {:.1?}", t.elapsed());
}

#[test]
fn msm_odd_lengths() {
    let _g = serial();
    require_device();
    assert!(midnight_metal::set_min_k(0));
    let all = points_cache();
    for n in [
        1usize,
        2,
        31,
        33,
        193,
        1000,
        (1 << 14) + 5,
        3 << 14,
        (1 << 18) + 5,
    ] {
        let s = random_scalars(n, n as u64);
        check("odd length", &s, &all[..n]);
        // the one-shot path splits from 2^14 points
        assert_eq!(splits(&s, &all[..n]), n >= 1 << 14, "{n}");
    }
    assert!(midnight_metal::set_min_k(14));
}

// ---------------------------------------------------------------------------
// The effective size (non-zero scalars) against the default thresholds.

/// `n` scalars, zero except `nz` of them (random, at spread positions).
fn sparse_scalars(n: usize, nz: usize, seed: u64) -> Vec<Fq> {
    let mut s = vec![Fq::ZERO; n];
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let step = n / nz.max(1);
    for i in 0..nz {
        s[i * step + (i * 7919) % step.max(1)] = Fq::random(&mut rng);
    }
    s
}

/// The device MSM, or `None` when it declined; panics on any decline other
/// than `below-min-k`, and checks a decline left the counters unchanged.
fn try_device_msm(s: &[Fq], b: &[G1Affine]) -> Option<G1Projective> {
    midnight_metal::reset_stats();
    let counts = midnight_metal::msm_counts();
    let served = midnight_metal::stats().msm.served;
    match midnight_metal::msm_g1(adapter::scalars(s), adapter::affine(b)) {
        Some(r) => {
            assert_eq!(midnight_metal::stats().msm.served, served + 1);
            Some(adapter::from_jacobian(r))
        }
        None => {
            assert_eq!(
                midnight_metal::declines(),
                vec![("msm", "below-min-k", 1)],
                "only a below-min-k decline"
            );
            // no device work: no table upload, no MSM of either path
            assert_eq!(midnight_metal::msm_counts(), counts);
            assert_eq!(midnight_metal::stats().msm.served, served);
            None
        }
    }
}

#[test]
fn sparse_msm_declines_below_min_k() {
    let _g = serial();
    require_device();
    midnight_metal::reset_min_k();
    let t = midnight_metal::thresholds();
    assert!(!t.explicit, "MIDNIGHT_METAL_MIN_K is set: {t:?}");
    let n = 1usize << 20;
    let bases = &points_cache()[..n];
    // the needs with the default thresholds: max(2^threshold, n/3) one-shot,
    // max(2^threshold, n/5) on a table (the measured sparse crossovers)
    let one_shot = (1usize << t.msm_one_shot).max(n / 3);
    let table = (1usize << t.msm_table).max(n / 5);
    assert_eq!((one_shot, table), (n / 3, n / 5));

    let served = |nz: usize, seed: u64| {
        let s = sparse_scalars(n, nz, seed);
        let got = try_device_msm(&s, bases).expect("served");
        if !cfg!(debug_assertions) {
            assert!(got == msm_pippenger(&s, bases), "{nz}");
        }
    };
    // one-shot: a mostly-zero MSM (as a permutation product or a quotient
    // limb) declines, and so does one below the crossover although above the
    // length threshold; from the crossover it is served; dense is served
    for nz in [0, 2, 11, 1 << t.msm_one_shot, one_shot - 1] {
        assert!(
            try_device_msm(&sparse_scalars(n, nz, nz as u64), bases).is_none(),
            "{nz}"
        );
    }
    served(one_shot, 1);
    served(n, 2);

    // a registered table: its own need
    let h = midnight_metal::register_srs(adapter::affine(bases));
    for nz in [2, 1 << t.msm_table, table - 1] {
        assert!(
            try_device_msm(&sparse_scalars(n, nz, 5), bases).is_none(),
            "table {nz}"
        );
    }
    served(table, 6);

    // an explicit threshold replaces the need: 2^k alone, on both paths
    assert!(midnight_metal::set_min_k(14));
    served(1 << 14, 7);
    assert!(try_device_msm(&sparse_scalars(n, (1 << 14) - 1, 8), bases).is_none());
    drop(h);
    served(1 << 14, 9);
    assert!(try_device_msm(&sparse_scalars(n, (1 << 14) - 1, 10), bases).is_none());
    assert!(midnight_metal::set_min_k(14));
}

#[test]
fn hook_sparse_commitment_matches_stock() {
    let _g = serial();
    require_device();
    midnight_metal::reset_min_k();
    assert!(!midnight_metal::thresholds().explicit);
    const K: u32 = 16;
    let mut rng = ChaCha8Rng::seed_from_u64(12);
    let params = ParamsKZG::<Bls12>::unsafe_setup(K, &mut rng);
    let h = midnight_metal::register_srs(adapter::affine(params.bases::<Coeff>()));
    let domain = EvaluationDomain::<Fq>::new(1, K);
    install_hooks();
    let label = || PolynomialLabel::Custom("p".into());
    for (nz, device) in [(5usize, false), (1usize << K, true)] {
        let mut poly = domain.empty_coeff();
        let step = poly.len() / nz;
        for i in 0..nz {
            poly[i * step] = Fq::random(&mut rng);
        }
        midnight_metal::reset_stats();
        let got = KZGCommitmentScheme::<Bls12>::commit(&params, &poly, label());
        let served = midnight_metal::stats().msm.served;
        let declines = midnight_metal::declines();
        if device {
            assert_eq!(served, 1, "{declines:?}");
        } else {
            assert_eq!(served, 0);
            assert_eq!(declines, vec![("msm", "below-min-k", 1)]);
        }
        let want = without_hooks(|| KZGCommitmentScheme::<Bls12>::commit(&params, &poly, label()));
        assert!(
            got == want,
            "{nz} non-zero: the commitment differs from the stock one"
        );
    }
    drop(h);
    assert!(midnight_metal::set_min_k(14));
}

// ---------------------------------------------------------------------------
// Registered tables.

#[test]
fn registered_table_prefix_and_changed_slice() {
    let _g = serial();
    require_device();
    assert!(midnight_metal::set_min_k(14));
    let n = 1usize << 16;
    let mut bases = points_cache()[..n].to_vec();
    let handle = midnight_metal::register_srs(adapter::affine(&bases));
    assert_eq!(handle.len(), n);
    assert!(handle.verify(adapter::affine(&bases)));
    let (cached0, one0, up0) = midnight_metal::msm_counts();

    // a prefix: the first MSM uploads the table
    let s = random_scalars(1 << 14, 5);
    check("table prefix", &s, &bases[..1 << 14]);
    assert_eq!(midnight_metal::msm_counts(), (cached0 + 1, one0, up0 + 1));
    // again (the mask MSM of this length is cached), then the whole slice
    let s2 = random_scalars(1 << 14, 6);
    check("table prefix again", &s2, &bases[..1 << 14]);
    check("table whole", &random_scalars(n, 7), &bases);
    assert_eq!(midnight_metal::msm_counts(), (cached0 + 3, one0, up0 + 1));

    // a stage profile on the table: one pass (MSM(r) cached)
    let (r, st) =
        midnight_metal::msm_g1_stages(adapter::scalars(&s2), adapter::affine(&bases[..1 << 14]))
            .unwrap();
    assert!(adapter::from_jacobian(r) == msm_pippenger(&s2, &bases[..1 << 14]));
    assert!(st.table && st.passes == 1, "{st:?}");

    // the registrant breaks its promise: the table is dropped and the MSM
    // takes the one-shot path
    let stale = midnight_metal::srs_stale();
    let fresh = random_points(n, 77);
    bases.copy_from_slice(&fresh);
    assert!(!handle.verify(adapter::affine(&bases)));
    let (c1, o1, u1) = midnight_metal::msm_counts();
    check("changed slice", &s, &bases[..1 << 14]);
    assert_eq!(midnight_metal::srs_stale(), stale + 1);
    assert_eq!(midnight_metal::msm_counts(), (c1, o1 + 1, u1));
    drop(handle);
}

/// A registered table's prefixes split between the device (a prefix of the
/// table) and the CPU (the caller's points after it).
#[test]
fn registered_table_prefix_splits() {
    let _g = serial();
    require_device();
    assert!(midnight_metal::set_min_k(14));
    set_cpu_share(true);
    let n = 1usize << 18;
    let bases = &points_cache()[..n];
    let handle = midnight_metal::register_srs(adapter::affine(bases));
    for m in [1usize << 17, (1 << 17) + 5, n] {
        let s = random_scalars(m, m as u64);
        let (r, st) =
            midnight_metal::msm_g1_stages(adapter::scalars(&s), adapter::affine(&bases[..m]))
                .unwrap();
        assert!(
            adapter::from_jacobian(r) == msm_pippenger(&s, &bases[..m]),
            "{m}"
        );
        assert!(st.table && st.device_points < m as u64, "{m}: {st:?}");
        check("table prefix, split", &s, &bases[..m]);
    }
    drop(handle);
}

// ---------------------------------------------------------------------------
// The hook.

/// Installs the MSM hook (once per process: the tests of this binary that
/// use the hook all install this one).
fn install_hooks() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        assert!(install(GpuHooks {
            msm_g1: Some(midnight_metal::msm_g1),
            ..Default::default()
        }));
    });
}

#[test]
fn hook_commitment_matches_stock() {
    let _g = serial();
    require_device();
    assert!(midnight_metal::set_min_k(14));
    const K: u32 = 14;
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let params = ParamsKZG::<Bls12>::unsafe_setup(K, &mut rng);
    let domain = EvaluationDomain::<Fq>::new(1, K);
    let mut poly = domain.empty_coeff();
    for c in poly.iter_mut() {
        *c = Fq::random(&mut rng);
    }
    install_hooks();
    let label = || PolynomialLabel::Custom("p".into());
    let before = midnight_metal::stats().msm.served;
    let got = KZGCommitmentScheme::<Bls12>::commit(&params, &poly, label());
    assert!(
        midnight_metal::stats().msm.served > before,
        "the hook did not serve"
    );
    let served = midnight_metal::stats().msm.served;
    let want = without_hooks(|| KZGCommitmentScheme::<Bls12>::commit(&params, &poly, label()));
    assert_eq!(
        midnight_metal::stats().msm.served,
        served,
        "a suspended hook served"
    );
    assert!(got == want, "the commitment differs from the stock one");
}

// ---------------------------------------------------------------------------
// Measurements.

fn ms(ns: u64) -> f64 {
    ns as f64 / 1e6
}

#[test]
#[ignore = "measurement"]
fn msm_timings() {
    let _g = serial();
    require_device();
    assert!(midnight_metal::set_min_k(0));
    const REPS: usize = 3;
    let all = random_points(1 << 22, 3);
    let scal = random_scalars(1 << 22, 4);
    println!(
        "threads {}",
        std::thread::available_parallelism().map_or(0, |t| t.get())
    );
    println!("k | cpu ms | one-shot ms | table first ms | table ms | (best of {REPS})");
    for k in 14..=22u32 {
        let n = 1usize << k;
        let (s, b) = (&scal[..n], &all[..n]);
        let best = |f: &mut dyn FnMut()| {
            (0..REPS)
                .map(|_| {
                    let t = Instant::now();
                    f();
                    t.elapsed().as_secs_f64() * 1e3
                })
                .fold(f64::MAX, f64::min)
        };
        let want = msm_pippenger(s, b);
        let cpu = best(&mut || {
            std::hint::black_box(msm_pippenger(s, b));
        });
        let one = best(&mut || assert!(device_msm(s, b) == want));
        let h = midnight_metal::register_srs(adapter::affine(b));
        let t = Instant::now();
        assert!(device_msm(s, b) == want);
        let first = t.elapsed().as_secs_f64() * 1e3;
        let tab = best(&mut || assert!(device_msm(s, b) == want));
        println!("{k} | {cpu:.1} | {one:.1} | {first:.1} | {tab:.1}");
        drop(h);
    }
    println!();
    println!(
        "k | path | wbits nwins batches passes | device points | upload | digits | sort | accumulate | integrate | collect | cpu share (ms) | bucket max/mean"
    );
    for k in 14..=22u32 {
        let n = 1usize << k;
        let (s, b) = (adapter::scalars(&scal[..n]), adapter::affine(&all[..n]));
        let (_, one) = midnight_metal::msm_g1_stages(s, b).unwrap();
        let h = midnight_metal::register_srs(b);
        let _ = midnight_metal::msm_g1_stages(s, b).unwrap();
        let (_, tab) = midnight_metal::msm_g1_stages(s, b).unwrap();
        for (path, st) in [("one-shot", one), ("table", tab)] {
            println!(
                "{k} | {path} | {} {} {} {} | {} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {} / {:.1}",
                st.wbits,
                st.nwins,
                st.batches,
                st.passes,
                st.device_points,
                ms(st.upload_ns),
                ms(st.digits_ns),
                ms(st.sort_ns),
                ms(st.accumulate_ns),
                ms(st.integrate_ns),
                ms(st.collect_ns),
                ms(st.cpu_ns),
                st.bucket_max,
                st.bucket_entries as f64 / st.bucket_slots.max(1) as f64,
            );
        }
        drop(h);
    }
    assert!(midnight_metal::set_min_k(14));
}
