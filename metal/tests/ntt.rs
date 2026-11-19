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
//!
//! `bench_fft` (ignored) times `fft` against the stock FFT:
//! `cargo test --release -p midnight-metal --test ntt -- --ignored --nocapture`.

#![cfg(target_os = "macos")]

use ff::{Field, PrimeField, WithSmallOrderMulGroup};
use midnight_curves::{
    Fq,
    fft::{best_fft_with_twiddles, compute_twiddles},
};
use midnight_metal::{Scalar, ntt_test_support as dev};
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

/// The stock forward transform (`best_fft_with_twiddles`, as the domain
/// calls it).
fn stock_fft(a: &mut [Fq], omega: Fq, log_n: u32) {
    best_fft_with_twiddles(a, &compute_twiddles(&omega, log_n), log_n);
}

/// The stock inverse transform (`EvaluationDomain::ifft`: the transform,
/// then every element times the divisor).
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
        // forward hook at the standard root
        let mut want = input.clone();
        stock_fft(&mut want, w, log_n);
        let mut got = input.clone();
        assert!(
            midnight_metal::fft(scalars_mut(&mut got), scalar(w), log_n),
            "fft declined"
        );
        assert_eq!(first_diff(&got, &want), None, "fft 2^{log_n} {name}");

        // forward hook at the inverse root
        let mut want = input.clone();
        stock_fft(&mut want, wi, log_n);
        let mut got = input.clone();
        assert!(
            midnight_metal::fft(scalars_mut(&mut got), scalar(wi), log_n),
            "fft declined"
        );
        assert_eq!(first_diff(&got, &want), None, "fft(w^-1) 2^{log_n} {name}");

        // inverse hook with 1/n and with another divisor, at either root
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

#[test]
fn fft_matches_stock() {
    setup();
    for log_n in 1..=20 {
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
    // (j, k): extended_k = k + ceil(log2(j - 1)); covers one, two and three
    // passes, and k == extended_k
    for (j, k) in [(2u32, 3u32), (4, 8), (4, 10), (8, 14), (5, 17), (4, 19)] {
        let domain = EvaluationDomain::<Fq>::new(j, k);
        let ext_k = domain.extended_k();
        let zeta = [Fq::ZETA, Fq::ZETA.square()];
        let zeta_out = [zeta[1], zeta[0]];

        // coeff_to_extended
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

        // extended_to_coeff, on the extended values just made and on random
        // ones
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
        // the round trip gives the coefficients back
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

#[test]
fn ntt_dev_matches_stock() {
    setup();
    for log_n in [1, 5, 10, 11, 16, 21] {
        let n = 1usize << log_n;
        let input = random(n, 500 + u64::from(log_n));
        let w = root(log_n);
        let c = random(1, 9)[0];
        for (inverse, r) in [(false, w), (true, w.invert().unwrap())] {
            let mut want = input.clone();
            stock_fft(&mut want, r, log_n);
            let got = dev::ntt(scalars(&input), log_n, inverse, None).unwrap();
            assert_eq!(
                &got[..],
                scalars(&want),
                "ntt_dev 2^{log_n} inverse {inverse}"
            );
            for x in want.iter_mut() {
                *x *= c;
            }
            let got = dev::ntt(scalars(&input), log_n, inverse, Some(scalar(c))).unwrap();
            assert_eq!(
                &got[..],
                scalars(&want),
                "ntt_dev scaled 2^{log_n} inverse {inverse}"
            );
        }
    }
}

#[test]
fn elementwise_blocks() {
    setup();
    let n = 3 * 1000 + 2;
    let values = random(n, 77);
    let z = random(2, 78);
    let c = random(1, 79)[0];
    for log_t in [0u32, 2, 3] {
        let table = random(1 << log_t, 80 + u64::from(log_t));
        let got = dev::elementwise(
            scalars(&values),
            [scalar(z[0]), scalar(z[1])],
            scalar(c),
            scalars(&table),
        )
        .unwrap();
        for (i, (g, v)) in got.iter().zip(&values).enumerate() {
            let mut want = *v;
            match i % 3 {
                1 => want *= z[0],
                2 => want *= z[1],
                _ => {}
            }
            want *= c;
            want *= table[i & ((1 << log_t) - 1)];
            assert_eq!(*g, scalar(want), "element {i}");
        }
    }
}

/// The device building blocks refuse an unreduced scalar argument (a scale,
/// a divisor, a `zeta`) before any device work, and keep serving reduced
/// ones afterwards.
#[test]
fn building_blocks_refuse_unreduced_scalars() {
    use midnight_metal::field::FR_MODULUS;
    setup();
    let bad = FR_MODULUS;
    let worse = [u64::MAX; 4];
    for log_n in [4u32, 12] {
        let values = random(1 << log_n, 600 + u64::from(log_n));
        for inverse in [false, true] {
            for s in [bad, worse] {
                let e = dev::ntt(scalars(&values), log_n, inverse, Some(s)).unwrap_err();
                assert!(e.contains("unreduced"), "ntt scale: {e}");
            }
        }
        let one = scalar(Fq::ONE);
        let z = scalar(Fq::ZETA);
        // the coset zetas of coeff_to_extended
        for zeta in [[bad, z], [z, bad], [worse, worse]] {
            let e = dev::coeff_to_extended(scalars(&values), log_n, log_n + 1, zeta).unwrap_err();
            assert!(e.contains("unreduced"), "coeff_to_extended zeta: {e}");
        }
        // the divisor and the zetas of extended_to_coeff
        let e = dev::extended_to_coeff(scalars(&values), log_n, bad, [z, z]).unwrap_err();
        assert!(e.contains("unreduced"), "extended_to_coeff divisor: {e}");
        let e = dev::extended_to_coeff(scalars(&values), log_n, one, [z, bad]).unwrap_err();
        assert!(e.contains("unreduced"), "extended_to_coeff zeta: {e}");
    }
    // the elementwise blocks: zeta, then the scale factor
    let values = random(30, 610);
    let table = random(2, 611);
    let z = scalar(Fq::ZETA);
    let e = dev::elementwise(scalars(&values), [z, bad], scalar(Fq::ONE), scalars(&table))
        .unwrap_err();
    assert!(e.contains("unreduced"), "distribute_zeta: {e}");
    let e = dev::elementwise(scalars(&values), [z, z], bad, scalars(&table)).unwrap_err();
    assert!(e.contains("unreduced"), "scale: {e}");
    // reduced arguments are served again
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

    // a root that is not the standard one: another primitive 2^log_n-th root
    // (w^3), the root of another size, and a non-root
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

    // a length that is not 2^log_n
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

    // an unreduced element
    let mut a = input;
    let mut raw: Vec<Scalar> = scalars(&a).to_vec();
    raw[17] = [u64::MAX; 4];
    let copy = raw.clone();
    assert!(!midnight_metal::fft(&mut raw, scalar(w), log_n));
    assert_eq!(raw, copy);
    // the input itself is accepted
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

/// Wall time of `fft` against the stock FFT on all cores.
#[test]
#[ignore]
fn bench_fft() {
    use std::time::Instant;
    setup();
    let reps = 5;
    println!(
        "log_n  stock_ms  metal_ms  speedup  copy_in  gpu  copy_out  other (ms, of the best run)"
    );
    for log_n in 14..=22u32 {
        let n = 1usize << log_n;
        let w = root(log_n);
        let tw = compute_twiddles(&w, log_n);
        let input = random(n, u64::from(log_n));
        let mut a = input.clone();
        // warm up both (pipelines, tables, scratch buffers)
        best_fft_with_twiddles(&mut a, &tw, log_n);
        let mut b = input.clone();
        assert!(midnight_metal::fft(scalars_mut(&mut b), scalar(w), log_n));
        assert_eq!(a, b);

        let mut stock = f64::MAX;
        let mut metal = f64::MAX;
        let mut parts = [0u64; 4];
        for _ in 0..reps {
            let mut a = input.clone();
            let t = Instant::now();
            best_fft_with_twiddles(&mut a, &tw, log_n);
            stock = stock.min(t.elapsed().as_secs_f64() * 1e3);

            let mut b = input.clone();
            let t = Instant::now();
            assert!(midnight_metal::fft(scalars_mut(&mut b), scalar(w), log_n));
            let wall = t.elapsed().as_secs_f64() * 1e3;
            if wall < metal {
                metal = wall;
                parts = dev::last_host_timing();
            }
        }
        let ms = |x: u64| x as f64 / 1e6;
        let [cin, dev_wall, gpu, cout] = parts;
        let other = metal - ms(cin) - ms(dev_wall) - ms(cout);
        println!(
            "{log_n:5}  {stock:8.3}  {metal:8.3}  {:7.2}  {:7.3}  {:6.3}  {:8.3}  {:.3} (+{:.3} launch)",
            stock / metal,
            ms(cin),
            ms(gpu),
            ms(cout),
            other,
            ms(dev_wall) - ms(gpu),
        );
    }
}

/// GPU time of back-to-back device transforms (no host copies).
#[test]
#[ignore]
fn bench_ntt_dev() {
    setup();
    for log_n in 12..=24u32 {
        let t = dev::ntt_gpu_nanos(log_n, 12).unwrap();
        let best = *t[2..].iter().min().unwrap() as f64 / 1e6;
        let n = (1u64 << log_n) as f64;
        // field multiplications of a radix-2 transform, (n / 2) log n
        let gmuls = n / 2.0 * f64::from(log_n) / (best * 1e-3) / 1e9;
        println!(
            "{log_n:3} {:?} gpu {best:8.3} ms  {gmuls:5.2} G butterfly/s",
            dev::passes(log_n)
        );
    }
}

/// The largest sizes (several command buffers each): GPU time of a forward
/// device transform, and a forward then inverse round trip on the device
/// giving the input back.
#[test]
#[ignore]
fn large_transforms() {
    setup();
    for log_n in [26u32, 27, 28] {
        let t = dev::ntt_gpu_nanos(log_n, 2).unwrap();
        println!(
            "2^{log_n} {:?} gpu {:?} ms",
            dev::passes(log_n),
            t.iter().map(|x| *x as f64 / 1e6).collect::<Vec<_>>()
        );
    }
    let log_n = 26;
    let n = 1usize << log_n;
    let input = random(n, 2026);
    let fwd = dev::ntt(scalars(&input), log_n, false, None).unwrap();
    let n_inv = Fq::from(n as u64).invert().unwrap();
    let back = dev::ntt(&fwd, log_n, true, Some(scalar(n_inv))).unwrap();
    assert_eq!(&back[..], scalars(&input));
}

/// Inverse transforms called from several threads at once (each takes its
/// own per-request buffers and the device only for the transform) give the
/// stock values.
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

/// Wall time of the `ifft` hook at `2^IFFT_BENCH_LOG` (default 20): a warm
/// call, the first call of a request (after `end_request`), and
/// `IFFT_BENCH_CALLERS` (default 4) callers on as many threads at once, the
/// way the prover's phase 1 calls it; minimum of `IFFT_BENCH_REPS` (default
/// 7) runs each.
#[test]
#[ignore]
fn bench_ifft_hook() {
    use std::time::Instant;
    setup();
    let env = |k: &str, d: usize| std::env::var(k).map_or(d, |v| v.parse().unwrap());
    let log_n = env("IFFT_BENCH_LOG", 20) as u32;
    let callers = env("IFFT_BENCH_CALLERS", 4);
    let reps = env("IFFT_BENCH_REPS", 7);
    let n = 1usize << log_n;
    let w_inv = root(log_n).invert().unwrap();
    let divisor = Fq::from(n as u64).invert().unwrap();
    let input = random(n, 7);
    let call = |a: &mut Vec<Fq>| {
        assert!(midnight_metal::ifft(
            scalars_mut(a),
            scalar(w_inv),
            log_n,
            scalar(divisor)
        ))
    };
    let ms = |x: u64| x as f64 / 1e6;
    // warm up
    let mut a = input.clone();
    call(&mut a);
    let mut warm = (f64::MAX, [0u64; 4]);
    for _ in 0..reps {
        let mut a = input.clone();
        let t = Instant::now();
        call(&mut a);
        let wall = t.elapsed().as_secs_f64() * 1e3;
        if wall < warm.0 {
            warm = (wall, dev::last_host_timing());
        }
    }
    let mut first = (f64::MAX, [0u64; 4]);
    for _ in 0..reps {
        assert!(midnight_metal::end_request());
        midnight_metal::begin_request();
        let mut a = input.clone();
        let t = Instant::now();
        call(&mut a);
        let wall = t.elapsed().as_secs_f64() * 1e3;
        if wall < first.0 {
            first = (wall, dev::last_host_timing());
        }
    }
    let mut par = f64::MAX;
    let mut gpu_par = 0u64;
    for _ in 0..reps {
        let mut bufs: Vec<Vec<Fq>> = (0..callers).map(|_| input.clone()).collect();
        let before = midnight_metal::stats().ifft.device_nanos;
        let t = Instant::now();
        std::thread::scope(|s| {
            for b in bufs.iter_mut() {
                s.spawn(|| call(b));
            }
        });
        let wall = t.elapsed().as_secs_f64() * 1e3;
        if wall < par {
            par = wall;
            gpu_par = midnight_metal::stats().ifft.device_nanos - before;
        }
    }
    println!("ifft hook at 2^{log_n} (ms, min of {reps})");
    for (what, (wall, [cin, dev_wall, gpu, cout])) in
        [("warm", warm), ("first of a request", first)]
    {
        println!(
            "| {what} | wall {wall:.2} | copy in {:.2} | device part {:.2} (gpu {:.2}) | copy out {:.2} |",
            ms(cin),
            ms(dev_wall),
            ms(gpu),
            ms(cout)
        );
    }
    println!(
        "| {callers} callers at once | wall {par:.2} | gpu {:.2} | per call {:.2} |",
        ms(gpu_par),
        par / callers as f64
    );
    assert!(midnight_metal::end_request());
}
