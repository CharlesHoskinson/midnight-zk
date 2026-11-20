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

//! The device NTT against the stock CPU code of midnight-curves and
//! midnight-proofs, bit for bit.

#![cfg(target_os = "macos")]

use ff::{Field, PrimeField, WithSmallOrderMulGroup};
use midnight_curves::{
    Fq,
    fft::{best_fft_with_twiddles, compute_twiddles},
};
use midnight_metal::{Scalar, field::FR_MODULUS, ntt_test_support as dev};
use midnight_proofs::{
    gpu::adapter::{from_scalar, scalar, scalars, scalars_mut},
    poly::EvaluationDomain,
};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn setup() {
    assert!(midnight_metal::device(), "no Metal device");
    midnight_metal::set_min_k(1);
}

fn root(log_n: u32) -> Fq {
    let mut w = Fq::ROOT_OF_UNITY;
    for _ in log_n..Fq::S {
        w = w.square();
    }
    w
}

fn random(n: usize, seed: u64) -> Vec<Fq> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    (0..n).map(|_| Fq::random(&mut rng)).collect()
}

/// Random values and the edge inputs: all zero, a single one, all `p - 1`.
fn inputs(log_n: u32, seed: u64) -> Vec<(&'static str, Vec<Fq>)> {
    let n = 1usize << log_n;
    let mut single = vec![Fq::ZERO; n];
    single[(seed as usize * 7919) % n] = Fq::ONE;
    let mut first = vec![Fq::ZERO; n];
    first[0] = Fq::ONE;
    vec![
        ("random", random(n, seed)),
        ("zero", vec![Fq::ZERO; n]),
        ("one-at-0", first),
        ("one", single),
        ("p-1", vec![-Fq::ONE; n]),
    ]
}

/// The stock forward transform, as `EvaluationDomain` calls it.
fn stock_fft(a: &mut [Fq], omega: Fq, log_n: u32) {
    best_fft_with_twiddles(a, &compute_twiddles(&omega, log_n), log_n);
}

fn stock_ifft(a: &mut [Fq], omega_inv: Fq, log_n: u32, divisor: Fq) {
    stock_fft(a, omega_inv, log_n);
    for x in a.iter_mut() {
        *x *= divisor;
    }
}

fn first_diff(a: &[Fq], b: &[Fq]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y)
}

#[test]
fn roots_match_curves() {
    setup();
    for log_n in 0..=28 {
        let w = root(log_n);
        assert_eq!(dev::root(log_n, false), Some(scalar(w)), "2^{log_n}");
        assert_eq!(
            dev::root(log_n, true),
            Some(scalar(w.invert().unwrap())),
            "2^{log_n}"
        );
    }
    assert_eq!(dev::root(29, false), None);
}

fn check_fft(log_n: u32, seed: u64) {
    let n = 1usize << log_n;
    let w = root(log_n);
    let wi = w.invert().unwrap();
    let n_inv = Fq::from(n as u64).invert().unwrap();
    let d_rand = random(1, seed ^ 0xd1)[0];
    for (name, input) in inputs(log_n, seed) {
        let mut want = input.clone();
        stock_fft(&mut want, w, log_n);
        let mut got = input.clone();
        assert!(
            midnight_metal::fft(scalars_mut(&mut got), scalar(w), log_n),
            "fft declined"
        );
        assert_eq!(first_diff(&got, &want), None, "fft 2^{log_n} {name}");

        let mut want = input.clone();
        stock_fft(&mut want, wi, log_n);
        let mut got = input.clone();
        assert!(
            midnight_metal::fft(scalars_mut(&mut got), scalar(wi), log_n),
            "fft declined"
        );
        assert_eq!(first_diff(&got, &want), None, "fft(w^-1) 2^{log_n} {name}");

        for (root_name, r) in [("w^-1", wi), ("w", w)] {
            for div in [n_inv, d_rand] {
                let mut want = input.clone();
                stock_ifft(&mut want, r, log_n, div);
                let mut got = input.clone();
                assert!(
                    midnight_metal::ifft(scalars_mut(&mut got), scalar(r), log_n, scalar(div)),
                    "ifft declined"
                );
                assert_eq!(
                    first_diff(&got, &want),
                    None,
                    "ifft({root_name}) 2^{log_n} {name}"
                );
            }
        }
    }
}

/// One pass: fewer than four and exactly four elements per thread, a
/// partial and a full threadgroup. Two passes: unequal and equal radices,
/// the high twiddle table only (16), both tables (17), the largest (20).
#[test]
fn fft_matches_stock() {
    setup();
    for log_n in [1, 2, 6, 10, 11, 12, 16, 17, 20] {
        check_fft(log_n, 100 + u64::from(log_n));
    }
}

#[test]
fn fft_matches_stock_three_passes() {
    setup();
    for log_n in [21, 22] {
        assert_eq!(dev::passes(log_n).len(), 3);
        let n = 1usize << log_n;
        let w = root(log_n);
        let input = random(n, 7 + u64::from(log_n));
        let mut want = input.clone();
        stock_fft(&mut want, w, log_n);
        let mut got = input.clone();
        assert!(midnight_metal::fft(scalars_mut(&mut got), scalar(w), log_n));
        assert_eq!(first_diff(&got, &want), None, "fft 2^{log_n}");

        let wi = w.invert().unwrap();
        let div = Fq::from(n as u64).invert().unwrap();
        let mut want = input.clone();
        stock_ifft(&mut want, wi, log_n, div);
        let mut got = input;
        assert!(midnight_metal::ifft(
            scalars_mut(&mut got),
            scalar(wi),
            log_n,
            scalar(div)
        ));
        assert_eq!(first_diff(&got, &want), None, "ifft 2^{log_n}");
    }
}

#[test]
fn building_blocks_match_stock_domain() {
    setup();
    // (j, k): k == extended_k, one, two (one and both twiddle tables) and
    // three passes
    for (j, k) in [(2u32, 3u32), (4, 8), (4, 10), (8, 14), (4, 19)] {
        let domain = EvaluationDomain::<Fq>::new(j, k);
        let ext_k = domain.extended_k();
        let zeta = [Fq::ZETA, Fq::ZETA.square()];
        let zeta_out = [zeta[1], zeta[0]];

        let coeffs = random(1 << k, u64::from(j * 100 + k));
        let want = domain.coeff_to_extended(domain.coeff_from_vec(coeffs.clone()));
        let got = dev::coeff_to_extended(scalars(&coeffs), k, ext_k, zeta.map(scalar)).unwrap();
        assert_eq!(got.len(), 1 << ext_k);
        let got: Vec<Fq> = got.into_iter().map(from_scalar).collect();
        assert_eq!(
            first_diff(&got, &want),
            None,
            "coeff_to_extended k {k} ext_k {ext_k}"
        );

        let divisor = Fq::from(1u64 << ext_k).invert().unwrap();
        for values in [got, random(1 << ext_k, u64::from(j * 1000 + k))] {
            let mut ext = domain.empty_extended();
            ext.copy_from_slice(&values);
            let want = domain.extended_to_coeff(ext);
            let got = dev::extended_to_coeff(
                scalars(&values),
                ext_k,
                scalar(divisor),
                zeta_out.map(scalar),
            )
            .unwrap();
            let got: Vec<Fq> = got.into_iter().map(from_scalar).collect();
            assert_eq!(
                first_diff(&got, &want),
                None,
                "extended_to_coeff ext_k {ext_k}"
            );
        }
        let back = dev::extended_to_coeff(
            &dev::coeff_to_extended(scalars(&coeffs), k, ext_k, zeta.map(scalar)).unwrap(),
            ext_k,
            scalar(divisor),
            zeta_out.map(scalar),
        )
        .unwrap();
        assert_eq!(&back[..1 << k], scalars(&coeffs));
        assert!(back[1 << k..].iter().all(|x| *x == [0; 4]));
    }
}

/// The kernels silently mis-compute on an unreduced by-value scalar.
#[test]
fn building_blocks_refuse_unreduced_scalars() {
    setup();
    let bad = FR_MODULUS;
    let worse = [u64::MAX; 4];
    let log_n = 4;
    let values = random(1 << log_n, 600 + u64::from(log_n));
    for inverse in [false, true] {
        for s in [bad, worse] {
            let e = dev::ntt(scalars(&values), log_n, inverse, Some(s)).unwrap_err();
            assert!(e.contains("unreduced"), "ntt scale: {e}");
        }
    }
    let one = scalar(Fq::ONE);
    let z = scalar(Fq::ZETA);
    for zeta in [[bad, z], [z, bad], [worse, worse]] {
        let e = dev::coeff_to_extended(scalars(&values), log_n, log_n + 1, zeta).unwrap_err();
        assert!(e.contains("unreduced"), "coeff_to_extended zeta: {e}");
    }
    let e = dev::extended_to_coeff(scalars(&values), log_n, bad, [z, z]).unwrap_err();
    assert!(e.contains("unreduced"), "extended_to_coeff divisor: {e}");
    let e = dev::extended_to_coeff(scalars(&values), log_n, one, [z, bad]).unwrap_err();
    assert!(e.contains("unreduced"), "extended_to_coeff zeta: {e}");
    let values = random(30, 610);
    let table = random(2, 611);
    let z = scalar(Fq::ZETA);
    let e =
        dev::elementwise(scalars(&values), [z, bad], scalar(Fq::ONE), scalars(&table)).unwrap_err();
    assert!(e.contains("unreduced"), "distribute_zeta: {e}");
    let e = dev::elementwise(scalars(&values), [z, z], bad, scalars(&table)).unwrap_err();
    assert!(e.contains("unreduced"), "scale: {e}");
    let got = dev::ntt(scalars(&values[..16]), 4, false, Some(scalar(Fq::ONE))).unwrap();
    let mut want = values[..16].to_vec();
    stock_fft(&mut want, root(4), 4);
    assert_eq!(&got[..], scalars(&want));
}

fn declines_of(class: &str, reason: &str) -> u64 {
    midnight_metal::declines()
        .into_iter()
        .filter(|(c, r, _)| *c == class && *r == reason)
        .map(|(_, _, k)| k)
        .sum()
}

#[test]
fn declines_leave_the_buffer_unchanged() {
    setup();
    let log_n = 12;
    let n = 1usize << log_n;
    let input = random(n, 4242);
    let w = root(log_n);
    let div = Fq::from(n as u64).invert().unwrap();

    // another primitive root, the roots of other sizes, a non-root
    for bad in [w * w * w, root(log_n + 1), root(log_n - 1), Fq::from(5u64)] {
        let before = declines_of("fft", "shape");
        let mut a = input.clone();
        assert!(!midnight_metal::fft(
            scalars_mut(&mut a),
            scalar(bad),
            log_n
        ));
        assert_eq!(a, input);
        assert!(declines_of("fft", "shape") > before);

        let before = declines_of("ifft", "shape");
        assert!(!midnight_metal::ifft(
            scalars_mut(&mut a),
            scalar(bad),
            log_n,
            scalar(div)
        ));
        assert_eq!(a, input);
        assert!(declines_of("ifft", "shape") > before);
    }

    for (len, lg) in [
        (n - 1, log_n),
        (n, log_n + 1),
        (n, log_n - 1),
        (2 * n, log_n),
    ] {
        let mut a = random(len, 99);
        let copy = a.clone();
        let before = declines_of("fft", "shape");
        assert!(!midnight_metal::fft(
            scalars_mut(&mut a),
            scalar(root(lg)),
            lg
        ));
        assert_eq!(a, copy);
        assert!(declines_of("fft", "shape") > before);
        let before = declines_of("ifft", "shape");
        let wi = root(lg).invert().unwrap();
        assert!(!midnight_metal::ifft(
            scalars_mut(&mut a),
            scalar(wi),
            lg,
            scalar(div)
        ));
        assert_eq!(a, copy);
        assert!(declines_of("ifft", "shape") > before);
    }

    let mut raw: Vec<Scalar> = scalars(&input).to_vec();
    raw[17] = [u64::MAX; 4];
    let copy = raw.clone();
    let before = declines_of("fft", "unreduced");
    assert!(!midnight_metal::fft(&mut raw, scalar(w), log_n));
    assert_eq!(raw, copy);
    assert_eq!(declines_of("fft", "unreduced") - before, 1);
    let mut a = input.clone();
    let before = declines_of("ifft", "unreduced");
    assert!(!midnight_metal::ifft(
        scalars_mut(&mut a),
        scalar(w.invert().unwrap()),
        log_n,
        FR_MODULUS
    ));
    assert_eq!(a, input);
    assert_eq!(declines_of("ifft", "unreduced") - before, 1);
    assert!(midnight_metal::fft(scalars_mut(&mut a), scalar(w), log_n));
}

#[test]
fn hooks_serve_the_domain_transforms() {
    use midnight_proofs::gpu::{GpuHooks, install, test_switches::without_hooks};
    setup();
    assert!(install(GpuHooks {
        msm_g1: None,
        fft: Some(midnight_metal::fft),
        ifft: Some(midnight_metal::ifft),
        quotient: None,
    }));
    let k = 14;
    let domain = EvaluationDomain::<Fq>::new(4, k);
    let values = random(1 << k, 31337);

    let before = midnight_metal::stats();
    let coeffs = domain.lagrange_to_coeff(domain.lagrange_from_vec(values.clone()));
    let evals = domain.coeff_to_lagrange(domain.coeff_from_vec(values.clone()));
    let after = midnight_metal::stats();
    assert!(after.ifft.served > before.ifft.served, "ifft not served");
    assert!(after.fft.served > before.fft.served, "fft not served");

    let (want_coeffs, want_evals) = without_hooks(|| {
        (
            domain.lagrange_to_coeff(domain.lagrange_from_vec(values.clone())),
            domain.coeff_to_lagrange(domain.coeff_from_vec(values.clone())),
        )
    });
    assert_eq!(first_diff(&coeffs, &want_coeffs), None);
    assert_eq!(first_diff(&evals, &want_evals), None);
}

/// Ignored: it needs about 10 GiB of memory.
#[test]
#[ignore]
fn large_transform_round_trip() {
    setup();
    let log_n = 26;
    let n = 1usize << log_n;
    let input = random(n, 2026);
    let fwd = dev::ntt(scalars(&input), log_n, false, None).unwrap();
    let n_inv = Fq::from(n as u64).invert().unwrap();
    let back = dev::ntt(&fwd, log_n, true, Some(scalar(n_inv))).unwrap();
    assert_eq!(&back[..], scalars(&input));
}

#[test]
fn parallel_ifft_callers_match_stock() {
    setup();
    let log_n = 16;
    let n = 1usize << log_n;
    let w_inv = root(log_n).invert().unwrap();
    let divisor = Fq::from(n as u64).invert().unwrap();
    std::thread::scope(|s| {
        for t in 0..6u64 {
            s.spawn(move || {
                for rep in 0..3u64 {
                    let input = random(n, 100 + 10 * t + rep);
                    let mut want = input.clone();
                    stock_ifft(&mut want, w_inv, log_n, divisor);
                    let mut got = input.clone();
                    assert!(midnight_metal::ifft(
                        scalars_mut(&mut got),
                        scalar(w_inv),
                        log_n,
                        scalar(divisor)
                    ));
                    assert_eq!(first_diff(&got, &want), None, "caller {t}, rep {rep}");
                }
            });
        }
    });
    assert!(midnight_metal::end_request());
}
