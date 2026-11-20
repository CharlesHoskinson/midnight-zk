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

//! The G1 MSM on the device against blst and the stock CPU MSM
//! (`msm_pippenger`). The window widths 15 and 16 and the sizes with several
//! chunks run in a release build only.

#![cfg(target_os = "macos")]

use std::sync::{Mutex, MutexGuard, OnceLock};

use blst::{
    blst_fp, blst_fp_mul, blst_fp_sqr, blst_p1, blst_p1_add_or_double, blst_p1_affine,
    blst_p1_cneg, blst_p1_from_affine, blst_p1_is_equal,
};
use ff::Field;
use group::{Curve, Group, prime::PrimeCurveAffine};
use midnight_curves::{Bls12, Fq, G1Affine, G1Projective, msm::msm_pippenger};
use midnight_metal::{
    G1AffineLimbs, MsmStages, Scalar,
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

/// Serializes the tests (thresholds and counters are process-wide) and sets
/// the threshold (`None`: the defaults) and the CPU share.
fn setup(min_k: Option<u32>, cpu_share: bool) -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    let g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    assert!(midnight_metal::device(), "no usable Metal device");
    match min_k {
        Some(k) => assert!(midnight_metal::set_min_k(k)),
        None => midnight_metal::reset_min_k(),
    }
    set_cpu_share(cpu_share);
    g
}

/// Whether `MIDNIGHT_METAL_MIN_K` overrides the defaults.
fn defaults_overridden() -> bool {
    let explicit = midnight_metal::thresholds().explicit;
    if explicit {
        eprintln!("MIDNIGHT_METAL_MIN_K is set: the defaults are not in force; skipped");
    }
    explicit
}

/// Called under the lock of [`setup`].
fn set_cpu_share(on: bool) {
    // SAFETY: no other thread of this binary reads the environment meanwhile.
    unsafe {
        if on {
            std::env::remove_var("MIDNIGHT_METAL_MSM_CPU_SHARE");
        } else {
            std::env::set_var("MIDNIGHT_METAL_MSM_CPU_SHARE", "off");
        }
    }
}

/// `n` distinct points: chunks of `P + i Q` from random `P`, `Q`.
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

/// The device MSM is served and equals `msm_pippenger`.
fn check(name: &str, s: &[Fq], b: &[G1Affine]) {
    let before = midnight_metal::stats().msm.served;
    let got = device_msm(s, b);
    assert!(
        got == msm_pippenger(s, b),
        "{name}: MSM of {} differs from msm_pippenger",
        s.len()
    );
    assert_eq!(midnight_metal::stats().msm.served, before + 1, "{name}");
}

/// The path and plan of one MSM, its value checked.
fn stages(s: &[Fq], b: &[G1Affine]) -> MsmStages {
    let (r, st) = midnight_metal::msm_g1_stages(adapter::scalars(s), adapter::affine(b))
        .unwrap_or_else(|| panic!("msm_g1_stages declined an MSM of {}", s.len()));
    assert!(
        adapter::from_jacobian(r) == msm_pippenger(s, b),
        "MSM of {} differs from msm_pippenger",
        s.len()
    );
    st
}

/// The device value, or the stock CPU MSM when it declines.
fn hook_or_stock(s: &[Fq], b: &[G1Affine]) -> G1Projective {
    midnight_metal::msm_g1(adapter::scalars(s), adapter::affine(b))
        .map_or_else(|| msm_pippenger(s, b), adapter::from_jacobian)
}

fn sized(b: &[G1Affine]) -> bool {
    midnight_metal::msm_sized(b.len(), adapter::affine(b))
}

const UNREGISTERED: &str = "unregistered-bases";

/// `msm_g1` declines `unregistered-bases` only, with no device MSM, no table
/// upload and no G1 check of the bases.
fn declines_unregistered(what: &str, s: &[Fq], b: &[G1Affine]) {
    midnight_metal::reset_stats();
    let counts = midnight_metal::msm_counts();
    let checks = midnight_metal::one_shot_g1_checks();
    let r = midnight_metal::msm_g1(adapter::scalars(s), adapter::affine(b));
    assert!(r.is_none(), "{what}: served");
    assert_eq!(midnight_metal::stats().msm.served, 0, "{what}");
    assert_eq!(
        midnight_metal::declines(),
        vec![("msm", UNREGISTERED, 1)],
        "{what}"
    );
    assert_eq!(
        midnight_metal::msm_counts(),
        counts,
        "{what}: no device MSM, no upload"
    );
    assert_eq!(
        midnight_metal::one_shot_g1_checks(),
        checks,
        "{what}: no G1 check"
    );
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

fn mask_fq(n: usize) -> Vec<Fq> {
    cuda_mask_vector(n).into_iter().map(adapter::from_scalar).collect()
}

type Xyzz = [u64; 24];

// Operation codes of `msm_test_g1` (msm_test.metal).
const OP_MADD: u32 = 0;
const OP_ADD: u32 = 1;
const OP_FROM_AFFINE: u32 = 2;

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

/// `(x z^2, y z^3, z^3, z^2)`; infinity is all zero.
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

/// An XYZZ record as a blst point, checking `ZZ^3 = ZZZ^2`.
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

fn peq(a: &blst_p1, b: &blst_p1) -> bool {
    // SAFETY: valid points.
    unsafe { blst_p1_is_equal(a, b) }
}

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
    let _g = setup(Some(14), true);
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let pts: Vec<G1Affine> = (0..64).map(|_| G1Projective::random(&mut rng).to_affine()).collect();
    let limbs = adapter::affine(&pts);
    let z = |i: usize| fp(&limbs[(i + 17) % limbs.len()][..6]);
    let lim = |p: &G1Affine| adapter::affine(std::slice::from_ref(p))[0];
    let inf = G1Affine::identity();

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
    for (i, (pa, pb)) in pts_pairs.iter().enumerate() {
        if *pa != inf && *pb == -*pa {
            assert_eq!(got[i], [0u64; 24], "P + (-P) at {i}");
        }
    }
}

#[test]
fn mask_is_the_cuda_vector() {
    let _g = setup(Some(14), true);
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

/// `(log2 n, wbits)`: one size per window width class of the one-shot path.
fn sizes(split: bool) -> &'static [(u32, u32)] {
    if split || cfg!(debug_assertions) {
        &[(14, 11), (16, 13)]
    } else {
        &[(14, 11), (16, 13), (20, 15)]
    }
}

/// Special inputs are spread over all indices so both shares of a split
/// hold them, and `odd[i] = odd[i / 2]` repeats points across the shares.
fn differential(split: bool) {
    let _g = setup(Some(0), split);
    midnight_metal::reset_stats();
    let all = points_cache();
    for &(k, wbits) in sizes(split) {
        let n = 1usize << k;
        let bases = &all[..n];
        let path = |st: MsmStages| (st.device_points < n as u64, st.wbits);

        let random = random_scalars(n, k as u64);
        assert_eq!(path(stages(&random, bases)), (split, wbits), "2^{k}");
        check("random", &random, bases);
        let c = Fq::random(ChaCha8Rng::seed_from_u64(99 + k as u64));
        check("all equal", &vec![c; n], bases);
        check("all zero", &vec![Fq::ZERO; n], bases);
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
        assert_eq!(path(stages(&odd_random, &odd)), (split, wbits), "2^{k}");
        check("odd bases, random", &odd_random, &odd);
        check("odd bases, repeated", &repeated, &odd);
        check("odd bases, all one", &vec![Fq::ONE; n], &odd);
    }
    assert!(
        midnight_metal::declines().is_empty(),
        "{:?}",
        midnight_metal::declines()
    );
}

#[test]
fn msm_matches_cpu() {
    differential(true);
    differential(false);
}

/// Every worker of a small pool waits on the device lock while the holder's
/// CPU share runs on the same pool.
#[test]
fn msm_from_rayon_pool() {
    let _g = setup(Some(14), true);
    let n = 1usize << 14;
    let bases = &points_cache()[..n];
    let inputs: Vec<Vec<Fq>> = (0..6).map(|i| random_scalars(n, 300 + i)).collect();
    let want: Vec<G1Projective> = inputs.iter().map(|s| msm_pippenger(s, bases)).collect();
    assert!(stages(&inputs[0], bases).device_points < n as u64);
    let got: Vec<G1Projective> = inputs.par_iter().map(|s| device_msm(s, bases)).collect();
    assert!(got == want);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(3).build().unwrap();
    let got: Vec<G1Projective> =
        pool.install(|| inputs.par_iter().map(|s| device_msm(s, bases)).collect());
    assert!(got == want);
}

/// Around the chunk stride of 32, below the plan table, and a split one.
#[test]
fn msm_odd_lengths() {
    let _g = setup(Some(0), true);
    let all = points_cache();
    for n in [1usize, 2, 31, 33, 193, 1000, (1 << 14) + 5] {
        let s = random_scalars(n, n as u64);
        check("odd length", &s, &all[..n]);
        assert_eq!(
            stages(&s, &all[..n]).device_points < n as u64,
            n >= 1 << 14,
            "{n}"
        );
    }
}

/// Scalars at the boundary of the `abs` step of `msm_digits`, raw and as
/// `B - mask_i` so the masked value on the device is `B`.
#[test]
fn abs_boundary_scalars_match_cpu() {
    let _g = setup(Some(0), true);
    let two = Fq::from(2);
    let half = two.invert().unwrap();
    let p2 = |e: u64| two.pow_vartime([e]);
    let top = Fq::from(0x39f6_d3a9u64);
    let boundary = [
        -half,                               // (r - 1) / 2
        half,                                // (r + 1) / 2
        -Fq::ONE,                            // r - 1
        p2(254) - Fq::ONE,                   // 2^254 - 1 reduced
        (top + Fq::ONE) * p2(224) - Fq::ONE, // top limb of s and r - s equal
        top * p2(224),                       // top limbs next to each other
    ];
    assert_eq!(-half - half, -Fq::ONE);
    let n = 1usize << 14;
    let bases = &points_cache()[..n];
    let mask = mask_fq(n);
    for split in [false, true] {
        set_cpu_share(split);
        for masked in [false, true] {
            let mut s = random_scalars(n, 81);
            for i in (0..n).step_by(3) {
                let b = boundary[(i / 3) % boundary.len()];
                s[i] = if masked { b - mask[i] } else { b };
            }
            check(
                &format!("abs boundary, masked {masked}, split {split}"),
                &s,
                bases,
            );
            let s: Vec<Fq> = (0..n)
                .map(|i| {
                    let b = boundary[i % boundary.len()];
                    if masked { b - mask[i] } else { b }
                })
                .collect();
            check(
                &format!("abs boundary only, masked {masked}, split {split}"),
                &s,
                bases,
            );
        }
    }
}

#[test]
fn wide_windows_and_chunks_match_cpu() {
    let _g = setup(Some(14), true);
    if cfg!(debug_assertions) {
        eprintln!("skipped in a debug build");
        return;
    }
    let n = 1usize << 22;
    let bases = random_points(n, 2222);
    let scal = random_scalars(n, 2223);
    let h = midnight_metal::register_srs(adapter::affine(&bases));
    for (k, wbits, batches) in [(20u32, 15u32, 1u32), (22, 16, 2)] {
        let m = 1usize << k;
        let (s, b) = (&scal[..m], &bases[..m]);
        let st = stages(s, b);
        assert!(st.table && st.device_points < m as u64, "2^{k}: {st:?}");
        assert_eq!((st.wbits, st.batches), (wbits, batches), "2^{k}: {st:?}");
        check(&format!("table 2^{k}"), s, b);
    }
    drop(h);

    set_cpu_share(false);
    let m = (1usize << 21) + 4096;
    let st = stages(&scal[..m], &bases[..m]);
    assert!(!st.table && st.device_points == m as u64, "{st:?}");
    assert_eq!((st.wbits, st.batches), (16, 2), "{st:?}");
}

/// `s_i = c - mask_i` puts every point of a window in one bucket: exact up
/// to `HOT_BUCKET_LIMIT` points, a `device-error` decline above it.
#[test]
fn hot_bucket_is_exact_or_declined() {
    let _g = setup(Some(14), true);
    let c = Fq::random(ChaCha8Rng::seed_from_u64(4242));
    for (k, served) in [(16u32, true), (17, false)] {
        let n = 1usize << k;
        let bases = &points_cache()[..n];
        let s: Vec<Fq> = mask_fq(n).iter().map(|m| c - m).collect();
        let want = msm_pippenger(&s, bases);
        for registered in [false, true] {
            let h = registered.then(|| midnight_metal::register_srs(adapter::affine(bases)));
            midnight_metal::reset_stats();
            let got = midnight_metal::msm_g1(adapter::scalars(&s), adapter::affine(bases));
            let what = format!("2^{k}, registered {registered}");
            if served {
                assert!(got.map(adapter::from_jacobian) == Some(want), "{what}");
            } else {
                assert!(got.is_none(), "{what}");
                assert_eq!(
                    midnight_metal::declines(),
                    vec![("msm", "device-error", 1)],
                    "{what}"
                );
                assert!(hook_or_stock(&s, bases) == want, "{what}");
            }
            drop(h);
        }
    }
}

/// `n` scalars, zero except `nz` of them at spread positions.
fn sparse_scalars(n: usize, nz: usize, seed: u64) -> Vec<Fq> {
    let mut s = vec![Fq::ZERO; n];
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let step = n / nz.max(1);
    for i in 0..nz {
        s[i * step + (i * 7919) % step.max(1)] = Fq::random(&mut rng);
    }
    s
}

/// The device MSM, or `None` on a `below-min-k` decline with no device work.
fn try_device_msm(s: &[Fq], b: &[G1Affine]) -> Option<G1Projective> {
    midnight_metal::reset_stats();
    let counts = midnight_metal::msm_counts();
    match midnight_metal::msm_g1(adapter::scalars(s), adapter::affine(b)) {
        Some(r) => {
            assert_eq!(midnight_metal::stats().msm.served, 1);
            Some(adapter::from_jacobian(r))
        }
        None => {
            assert_eq!(
                midnight_metal::declines(),
                vec![("msm", "below-min-k", 1)],
                "only a below-min-k decline"
            );
            assert_eq!(midnight_metal::msm_counts(), counts);
            assert_eq!(midnight_metal::stats().msm.served, 0);
            None
        }
    }
}

/// With the defaults a table needs `2^12 + n/5` non-zero scalars; an
/// explicit threshold needs `2^k` on both paths.
#[test]
fn sparse_msm_declines_below_min_k() {
    let _g = setup(None, true);
    let t = midnight_metal::thresholds();
    assert!(!t.explicit, "MIDNIGHT_METAL_MIN_K is set: {t:?}");
    assert_eq!(t.msm_one_shot, None);
    let n = 1usize << 16;
    let bases = &points_cache()[..n];
    let need = (1usize << 12) + n / 5;
    let served = |nz: usize, seed: u64| {
        let s = sparse_scalars(n, nz, seed);
        let got = try_device_msm(&s, bases).expect("served");
        assert!(got == msm_pippenger(&s, bases), "{nz}");
    };

    for nz in [0, 2, 11, n / 3, n] {
        declines_unregistered(&format!("{nz}"), &sparse_scalars(n, nz, nz as u64), bases);
    }

    let h = midnight_metal::register_srs(adapter::affine(bases));
    for nz in [2, 1 << t.msm_table, need - 1] {
        assert!(
            try_device_msm(&sparse_scalars(n, nz, 5), bases).is_none(),
            "table {nz}"
        );
    }
    served(need, 6);

    // a column with zero rows before its blinding rows; the multi-open's
    // length m - 1
    let m = 1usize << t.msm_table;
    for len in [m, m - 1] {
        let mut s = random_scalars(len, 13);
        for z in &mut s[len - 20..len - 4] {
            *z = Fq::ZERO;
        }
        let got = try_device_msm(&s, &bases[..len]).expect("a dense MSM at the threshold");
        assert!(got == msm_pippenger(&s, &bases[..len]), "{len}");
    }
    assert!(try_device_msm(&sparse_scalars(m, m / 2, 14), &bases[..m]).is_none());

    assert!(midnight_metal::set_min_k(14));
    served(1 << 14, 7);
    assert!(try_device_msm(&sparse_scalars(n, (1 << 14) - 1, 8), bases).is_none());
    drop(h);
    served(1 << 14, 9);
    assert!(try_device_msm(&sparse_scalars(n, (1 << 14) - 1, 10), bases).is_none());
}

#[test]
fn unregistered_bases_decline_by_default() {
    let _g = setup(None, true);
    if defaults_overridden() {
        return;
    }
    assert_eq!(midnight_metal::thresholds().msm_one_shot, None);
    let n = 1usize << 16;
    let bases = &points_cache()[..n];
    let s = random_scalars(n, 516);

    assert!(!sized(bases));
    declines_unregistered("unregistered", &s, bases);
    assert!(hook_or_stock(&s, bases) == msm_pippenger(&s, bases));

    let h = midnight_metal::register_srs(adapter::affine(bases));
    assert!(sized(bases));
    let checks = midnight_metal::one_shot_g1_checks();
    let (c0, o0, _) = midnight_metal::msm_counts();
    check("registered", &s, bases);
    let (c1, o1, _) = midnight_metal::msm_counts();
    assert_eq!((c1 - c0, o1 - o0), (1, 0), "from the table");
    assert_eq!(midnight_metal::one_shot_g1_checks(), checks);
    drop(h);

    assert!(!sized(bases));
    declines_unregistered("dropped", &s, bases);
}

#[test]
fn infinity_bases_and_malformed_calls() {
    let _g = setup(Some(14), true);
    let n = 1usize << 14;
    let mut bases = points_cache()[..n].to_vec();
    for i in (0..n).step_by(5) {
        bases[i] = G1Affine::identity();
    }
    let s: Vec<Fq> = random_scalars(n, 91)
        .into_iter()
        .map(|x| if x == Fq::ZERO { Fq::ONE } else { x })
        .collect();
    let (_, o0, _) = midnight_metal::msm_counts();
    check("infinity bases, one-shot", &s, &bases);
    assert_eq!(midnight_metal::msm_counts().1, o0 + 1);
    let h = midnight_metal::register_srs(adapter::affine(&bases));
    let (c0, _, _) = midnight_metal::msm_counts();
    check("infinity bases, table", &s, &bases);
    assert_eq!(midnight_metal::msm_counts().0, c0 + 1);

    for (sl, bl) in [(n - 1, n), (n, n - 1)] {
        midnight_metal::reset_stats();
        assert!(
            midnight_metal::msm_g1(adapter::scalars(&s[..sl]), adapter::affine(&bases[..bl]))
                .is_none()
        );
        assert_eq!(midnight_metal::declines(), vec![("msm", "shape", 1)]);
    }
    let mut unreduced = adapter::scalars(&s).to_vec();
    unreduced[7] = midnight_metal::field::FR_MODULUS;
    midnight_metal::reset_stats();
    assert!(midnight_metal::msm_g1(&unreduced, adapter::affine(&bases)).is_none());
    assert_eq!(midnight_metal::declines(), vec![("msm", "unreduced", 1)]);
    drop(h);
}

/// The first MSM uploads the table, the second has `MSM(r)` cached (one
/// pass), and longer prefixes split with the CPU.
#[test]
fn registered_table_prefixes() {
    let _g = setup(Some(14), true);
    let n = (1usize << 17) + 5;
    let bases = &points_cache()[..n];
    let handle = midnight_metal::register_srs(adapter::affine(bases));
    assert_eq!(handle.len(), n);
    assert_eq!(handle.device_bytes(), 0, "a registration is host memory");
    assert!(handle.verify(adapter::affine(bases)));
    let (c0, o0, u0) = midnight_metal::msm_counts();

    let m = 1usize << 14;
    let st = stages(&random_scalars(m, 5), &bases[..m]);
    assert!(st.table && st.passes == 2, "{st:?}");
    assert_eq!(midnight_metal::msm_counts(), (c0 + 1, o0, u0 + 1));
    assert_eq!(handle.device_bytes(), (n * 96) as u64);
    let st = stages(&random_scalars(m, 6), &bases[..m]);
    assert!(st.table && st.passes == 1, "{st:?}");

    for len in [1usize << 17, n] {
        let s = random_scalars(len, len as u64);
        let st = stages(&s, &bases[..len]);
        assert!(st.table && st.device_points < len as u64, "{len}: {st:?}");
        check("table prefix, split", &s, &bases[..len]);
    }
    assert_eq!(midnight_metal::msm_counts(), (c0 + 6, o0, u0 + 1));
    drop(handle);
}

/// A table changed at one point a sampled check would not read is dropped:
/// with the defaults the call declines, with an explicit threshold the
/// one-shot path serves it.
#[test]
fn stale_table_declines_by_default_and_one_shot_when_explicit() {
    let _g = setup(None, true);
    if defaults_overridden() {
        return;
    }
    let n = 1usize << 15;
    let fresh = random_points(3, 4711);
    let s = random_scalars(n, 4712);
    let i = 3 * n / 7 + 5;

    let mut bases = points_cache()[..n].to_vec();
    let h = midnight_metal::register_srs(adapter::affine(&bases));
    check("table", &s, &bases);
    bases[i] = fresh[0];
    let stale = midnight_metal::srs_stale();
    assert!(!sized(&bases));
    assert_eq!(midnight_metal::srs_stale(), stale + 1);
    declines_unregistered("stale, msm_sized first", &s, &bases);
    assert!(hook_or_stock(&s, &bases) == msm_pippenger(&s, &bases));
    assert_eq!(midnight_metal::srs_stale(), stale + 1);
    drop(h);

    let mut bases = points_cache()[..n].to_vec();
    let h = midnight_metal::register_srs(adapter::affine(&bases));
    bases[i] = fresh[1];
    let stale = midnight_metal::srs_stale();
    declines_unregistered("stale", &s, &bases);
    assert_eq!(midnight_metal::srs_stale(), stale + 1);
    assert!(!sized(&bases));
    drop(h);

    assert!(midnight_metal::set_min_k(14));
    let mut bases = points_cache()[..n].to_vec();
    let h = midnight_metal::register_srs(adapter::affine(&bases));
    bases[i] = fresh[0];
    assert!(!h.verify(adapter::affine(&bases)));
    let stale = midnight_metal::srs_stale();
    assert!(sized(&bases));
    assert_eq!(midnight_metal::srs_stale(), stale + 1);
    let checks = midnight_metal::one_shot_g1_checks();
    let (c0, o0, u0) = midnight_metal::msm_counts();
    check("stale, explicit threshold", &s, &bases);
    assert_eq!(midnight_metal::msm_counts(), (c0, o0 + 1, u0));
    assert_eq!(midnight_metal::one_shot_g1_checks(), checks + 1);
    drop(h);

    let mut bases = points_cache()[..n].to_vec();
    let h = midnight_metal::register_srs(adapter::affine(&bases));
    let m = 1usize << 14;
    let s = random_scalars(m, 62);
    check("prefix", &s, &bases[..m]);
    let j = 2 * m / 7 + 3;
    bases[j] = fresh[2];
    let stale = midnight_metal::srs_stale();
    let (c0, o0, u0) = midnight_metal::msm_counts();
    check("prefix, one point changed", &s, &bases[..m]);
    assert_eq!(midnight_metal::srs_stale(), stale + 1);
    assert_eq!(midnight_metal::msm_counts(), (c0, o0 + 1, u0));
    drop(h);
}

#[test]
fn msm_g1_sized_reports_both_entries() {
    let _g = setup(None, true);
    if defaults_overridden() {
        return;
    }
    let n = 1usize << 15;
    let s = random_scalars(n, 4721);
    let call = |s: &[Fq], b: &[G1Affine]| {
        midnight_metal::reset_stats();
        midnight_metal::msm_g1_sized(adapter::scalars(s), adapter::affine(b))
    };
    let mut bases = points_cache()[..n].to_vec();

    let (sized_call, r) = call(&s, &bases);
    assert!(!sized_call && !sized(&bases) && r.is_none());
    assert_eq!(midnight_metal::declines(), vec![("msm", UNREGISTERED, 1)]);

    let h = midnight_metal::register_srs(adapter::affine(&bases));
    let (sized_call, r) = call(&s, &bases);
    assert!(sized_call && sized(&bases));
    assert!(r.map(adapter::from_jacobian) == Some(msm_pippenger(&s, &bases)));
    assert_eq!(midnight_metal::stats().msm.served, 1);

    bases[3 * n / 7 + 5] = random_points(1, 4722)[0];
    let stale = midnight_metal::srs_stale();
    let (sized_call, r) = call(&s, &bases);
    assert!(!sized_call && r.is_none());
    assert_eq!(midnight_metal::srs_stale(), stale + 1);
    assert_eq!(midnight_metal::declines(), vec![("msm", UNREGISTERED, 1)]);
    assert!(!sized(&bases));
    drop(h);

    let (sized_call, r) = call(&s[..n - 1], &bases);
    assert!(!sized_call && r.is_none());
    assert_eq!(midnight_metal::declines(), vec![("msm", "shape", 1)]);
}

/// A point on the curve outside the prime-order subgroup.
fn non_subgroup_point() -> G1AffineLimbs {
    use blst::{
        blst_fp_add, blst_fp_from_uint64, blst_fp_sqrt, blst_p1_affine_in_g1,
        blst_p1_affine_on_curve,
    };
    let fp_of = |v: u64| {
        let mut r = blst_fp::default();
        // SAFETY: six limbs.
        unsafe { blst_fp_from_uint64(&mut r, [v, 0, 0, 0, 0, 0].as_ptr()) };
        r
    };
    let four = fp_of(4);
    for v in 1u64.. {
        let x = fp_of(v);
        let mut rhs = blst_fp::default();
        let x3 = fmul(&fsqr(&x), &x);
        // SAFETY: valid field elements.
        unsafe { blst_fp_add(&mut rhs, &x3, &four) };
        let mut y = blst_fp::default();
        // SAFETY: as above.
        if !unsafe { blst_fp_sqrt(&mut y, &rhs) } {
            continue;
        }
        let p = blst_p1_affine { x, y };
        // SAFETY: a valid affine point.
        let (on, in_g1) = unsafe { (blst_p1_affine_on_curve(&p), blst_p1_affine_in_g1(&p)) };
        assert!(on);
        if !in_g1 {
            let mut l = [0u64; 12];
            l[..6].copy_from_slice(&p.x.l);
            l[6..].copy_from_slice(&p.y.l);
            return l;
        }
    }
    unreachable!()
}

fn affine_of(l: G1AffineLimbs) -> G1Affine {
    // SAFETY: `G1Affine` is a transparent `blst_p1_affine` (`[u64; 12]`).
    unsafe { std::mem::transmute::<G1AffineLimbs, G1Affine>(l) }
}

#[test]
fn bases_outside_g1_decline() {
    let _g = setup(Some(14), true);
    let n = 1usize << 14;
    let k = n / 3;
    let bad = non_subgroup_point();
    let mut off = adapter::affine(&points_cache()[..1])[0];
    off[6] ^= 1;
    for (what, p) in [("not in the subgroup", bad), ("off the curve", off)] {
        let mut bases = points_cache()[..n].to_vec();
        bases[k] = affine_of(p);
        for (name, sc) in [("r-1", -Fq::ONE), ("> r/2", -Fq::from(12345))] {
            let mut s = random_scalars(n, 71);
            s[k] = sc;
            for registered in [false, true] {
                let h = registered.then(|| midnight_metal::register_srs(adapter::affine(&bases)));
                midnight_metal::reset_stats();
                let counts = midnight_metal::msm_counts();
                let got = midnight_metal::msm_g1(adapter::scalars(&s), adapter::affine(&bases));
                let case = format!("{what}, {name}, registered {registered}");
                assert!(got.is_none(), "{case}");
                assert_eq!(
                    midnight_metal::declines(),
                    vec![("msm", "bases-not-in-subgroup", 1)],
                    "{case}"
                );
                assert_eq!(midnight_metal::stats().msm.served, 0, "{case}");
                assert_eq!(midnight_metal::msm_counts(), counts, "{case}");
                drop(h);
            }
            // the fallback runs on this input (a debug build of
            // midnight-curves asserts every point is in the subgroup)
            if what == "not in the subgroup" && !cfg!(debug_assertions) {
                let _ = msm_pippenger(&s, &bases);
            }
        }
        assert!(midnight_metal::set_min_k(0));
        let h = midnight_metal::register_srs(adapter::affine(&bases));
        let s = random_scalars(k, 73);
        check("prefix before the bad point", &s, &bases[..k]);
        drop(h);
        assert!(midnight_metal::set_min_k(14));
    }
}

/// KZG commitments through the hook equal the stock ones, served or not.
#[test]
fn hook_commitment_matches_stock() {
    let _g = setup(None, true);
    if defaults_overridden() {
        return;
    }
    const K: u32 = 14;
    let mut rng = ChaCha8Rng::seed_from_u64(12);
    let params = ParamsKZG::<Bls12>::unsafe_setup(K, &mut rng);
    let h = midnight_metal::register_srs(adapter::affine(params.bases::<Coeff>()));
    let domain = EvaluationDomain::<Fq>::new(1, K);
    assert!(install(GpuHooks {
        msm_g1: Some(midnight_metal::msm_g1),
        ..Default::default()
    }));
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
        assert_eq!(
            midnight_metal::stats().msm.served,
            served,
            "a suspended hook served"
        );
        assert!(
            got == want,
            "{nz} non-zero: the commitment differs from the stock one"
        );
    }
    drop(h);
}
