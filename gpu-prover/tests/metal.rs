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

//! Metal device tests. Compiled only with `--features metal` (the build script
//! then sets the cfg `midnight_metal_linked`); the byte-identity tests also
//! need `test-seed`. Each test returns early, with a line on stderr, when no
//! device is detected.
//!
//! The byte-identity tests prove one seeded request through the Metal device
//! and once more on the CPU (`NoDevice`, hooks suspended) and compare the
//! proof bytes: the example relation with the size thresholds lowered, and a
//! chain of Poseidon hashes at k = 16, which the measured per-class default
//! thresholds admit to all four kernels (release build:
//! `cargo test --release -p midnight-gpu-prover --features metal,test-seed
//! --test metal -- --nocapture`).

#![cfg(all(feature = "metal", midnight_metal_linked, target_os = "macos"))]

mod common;

use common::*;
use midnight_gpu_prover::{
    RequestOptions,
    config::DeviceSelect,
    device::{self, Device, Phases},
};

/// The hooks and the device layer's counters are process-wide: the tests of
/// this file run one at a time.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn detect() -> Option<Box<dyn Device>> {
    let d = device::detect(DeviceSelect::Auto);
    if d.info().is_none() {
        eprintln!("no Metal device detected: skipped");
        return None;
    }
    Some(d)
}

fn print_classes(what: &str) {
    let s = midnight_metal::stats();
    eprintln!(
        "{what}: served msm={} fft={} ifft={} quotient={}",
        s.msm.served, s.fft.served, s.ifft.served, s.quotient.served
    );
    let (cached, one_shot, uploads) = midnight_metal::msm_counts();
    eprintln!(
        "{what}: msm table={cached} one-shot={one_shot} uploads={uploads}, handles={}, stale={}",
        midnight_metal::srs_handles(),
        midnight_metal::srs_stale()
    );
    for (class, reason, count) in midnight_metal::declines() {
        eprintln!("{what}: declined {class} {reason} x{count}");
    }
}

#[cfg(feature = "test-seed")]
mod seeded {
    use midnight_circuits::{
        hash::poseidon::PoseidonChip,
        instructions::{AssignmentInstructions, PublicInputInstructions, hash::HashCPU},
    };
    use midnight_gpu_prover::{
        Config, ProverService, RequestOptions, ServedRelation,
        config::{ResidentPolicy, WslSelect},
        device::{Device, NoDevice},
        example::{PoseidonPreimage, example_input},
        input::SecretBytes,
        report::Label,
    };
    use midnight_proofs::{
        circuit::{Layouter, Value},
        plonk::Error,
        poly::kzg::params::ParamsKZG,
    };
    use midnight_zk_stdlib::{Relation, ZkStdLib, ZkStdLibArch};
    use rand::SeedableRng;

    use super::*;

    type F = midnight_curves::Fq;

    const K: u32 = 16;
    const ROUNDS: usize = 600;
    const SEED: u64 = 7;

    /// `ROUNDS` chained Poseidon hashes of three inputs; the last is public
    /// (the relation of the CUDA device tests, `tests/gpu.rs`).
    #[derive(Clone, Copy, Debug, Default)]
    struct PoseidonChain;

    fn chain(w: &[F; 3]) -> F {
        let mut h = <PoseidonChip<F> as HashCPU<F, F>>::hash(w);
        for _ in 1..ROUNDS {
            h = <PoseidonChip<F> as HashCPU<F, F>>::hash(&[h, w[1], w[2]]);
        }
        h
    }

    impl Relation for PoseidonChain {
        type Instance = F;
        type Witness = [F; 3];
        type Error = Error;

        fn format_instance(i: &F) -> Result<Vec<F>, Error> {
            Ok(vec![*i])
        }

        fn circuit(
            &self,
            std: &ZkStdLib,
            layouter: &mut impl Layouter<F>,
            _: Value<F>,
            witness: Value<[F; 3]>,
        ) -> Result<(), Error> {
            let w = std.assign_many(layouter, &witness.transpose_array())?;
            let mut h = std.poseidon(layouter, &w)?;
            for _ in 1..ROUNDS {
                h = std.poseidon(layouter, &[h, w[1].clone(), w[2].clone()])?;
            }
            std.constrain_as_public_input(layouter, &h)
        }

        fn used_chips(&self) -> ZkStdLibArch {
            ZkStdLibArch {
                poseidon: true,
                ..ZkStdLibArch::default()
            }
        }

        fn write_relation<W: std::io::Write>(&self, _: &mut W) -> std::io::Result<()> {
            Ok(())
        }

        fn read_relation<R: std::io::Read>(_: &mut R) -> std::io::Result<Self> {
            Ok(PoseidonChain)
        }
    }

    impl ServedRelation for PoseidonChain {
        fn name(&self) -> String {
            "poseidon-chain".into()
        }
        fn decode_input(&self, input: &[u8]) -> Result<(F, [F; 3]), String> {
            let (_, w) = PoseidonPreimage.decode_input(input)?;
            Ok((chain(&w), w))
        }
        fn k(&self) -> Option<u32> {
            Some(K)
        }
    }

    fn cfg(resident: ResidentPolicy) -> Config {
        Config {
            threads: 8,
            wsl: WslSelect::Off,
            resident,
            ..Config::default()
        }
    }

    /// Proves `input` seeded through the Metal device (every phase admitted)
    /// and once on the CPU, and checks: equal proof bytes, the stock verifier
    /// accepts, the `msm`, `ifft` and `quotient` classes served, no
    /// fallback, the erasure checked and clean. Returns the device service
    /// for further checks.
    ///
    /// The `fft` class (the forward base-domain transform) is checked as never
    /// offered: the PLONK prover of `midnight-proofs` makes no forward
    /// base-domain FFT. Its only callers in `EvaluationDomain` are
    /// `coeff_to_lagrange` and `extended_to_lagrange`, which the prover does
    /// not call; its forward transforms are the extended cosets of
    /// `coeff_to_extended`, done by the pruned `fft_coeff_to_extended` of
    /// `midnight-curves` (no hook) on the CPU path and inside the quotient job
    /// on the device. The `fft` entry point is checked against the stock FFT by
    /// the tests of `midnight-metal`.
    fn byte_identity<R: ServedRelation + Copy>(
        name: &str,
        relation: R,
        srs: impl Fn() -> ParamsKZG<midnight_curves::Bls12>,
        resident: ResidentPolicy,
        dev: Box<dyn Device>,
    ) -> ProverService<R> {
        let input = SecretBytes::new(example_input(4));
        let opts = || RequestOptions {
            zero_check: true,
            ..RequestOptions::default()
        };
        midnight_metal::reset_stats();
        let (c0, o0, u0) = midnight_metal::msm_counts();
        let mut gpu = ProverService::with_device(relation, srs(), cfg(resident), dev).unwrap();
        let g = gpu.prove_seeded(&input, opts(), SEED).unwrap();
        let r = &g.record;
        eprintln!(
            "{name}: label {:?}, admitted {:?}, device calls {:?}, fallbacks {:?}",
            r.label,
            r.admitted.names(),
            r.device_calls,
            r.fallbacks
        );
        for f in &r.declines {
            eprintln!("{name}: record decline {} {} x{}", f.phase, f.reason, f.count);
        }
        print_classes(name);
        let (c1, o1, u1) = midnight_metal::msm_counts();
        eprintln!(
            "{name}: msm_counts of the request: table {}, one-shot {}, uploads {}",
            c1 - c0,
            o1 - o0,
            u1 - u0
        );
        eprintln!("{name}: wipe {:?}", r.wipe);
        eprintln!("{name}: held after the request {:?}", gpu.held());
        eprintln!(
            "{name}: size thresholds {:?}",
            midnight_metal::thresholds().named()
        );

        let mut cpu =
            ProverService::with_device(relation, srs(), cfg(resident), Box::new(NoDevice))
                .unwrap();
        let c = midnight_proofs::gpu::test_switches::without_hooks(|| {
            cpu.prove_seeded(&input, opts(), SEED).unwrap()
        });
        assert_eq!(c.record.label, Label::Cpu);
        assert!(g.proof == c.proof, "{name}: the device proof differs from the CPU proof");

        assert_eq!(r.label, Label::Gpu, "{name}: {:?}", r.degraded_reasons);
        assert!(r.admitted.contains(Phases::ALL), "{name}: {:?}", r.admitted);
        let served = |class: &str| r.device_calls.get(class).copied().unwrap_or(0);
        let offered = |class: &str| {
            served(class)
                + r.declines.iter().filter(|d| d.phase == class).map(|d| d.count).sum::<u64>()
        };
        for class in ["msm", "ifft", "quotient"] {
            assert!(served(class) > 0, "{name}: no {class} served: {:?}", r.device_calls);
        }
        assert_eq!(offered("fft"), 0, "{name}: the prover made a forward FFT call");
        assert_eq!(r.fallback_count, 0, "{name}: {:?}", r.fallbacks);
        assert!(r.fallbacks.is_empty());
        assert!(r.wipe.checked && r.wipe.regions > 0 && r.wipe.bytes > 0, "{:?}", r.wipe);
        assert_eq!(
            (r.wipe.nonzero_regions, r.wipe.check_failures, r.wipe.erasure_failures),
            (0, 0, 0)
        );
        assert!(gpu.retired().is_none());
        // the stock verifier of the standard library accepts the proof
        let (inst, _) = relation.decode_input(&example_input(4)).unwrap();
        midnight_zk_stdlib::verify::<R, blake2b_simd::State, _>(
            gpu.verifier_params(),
            gpu.vk(),
            &inst,
            None,
            &g.proof,
        )
        .unwrap();
        gpu
    }

    /// The example relation, with the thresholds lowered so that its calls
    /// reach the kernels.
    #[test]
    fn seeded_small_proof_equals_cpu_proof_with_min_k_lowered() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let Some(d) = detect() else { return };
        assert!(midnight_metal::set_min_k(1));
        byte_identity("example", PoseidonPreimage, srs, ResidentPolicy::Evict, d);
        midnight_metal::reset_min_k();
    }

    /// k = 16 with the measured per-class default thresholds: every class
    /// reaches the device without lowering them. The service keeps the key
    /// copies and the tables (`keep`), so the held bytes are the device's.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "minutes in a debug build: run with --release")]
    fn seeded_k16_proof_equals_cpu_proof_with_default_thresholds() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let Some(d) = detect() else { return };
        midnight_metal::reset_min_k();
        if std::env::var_os("MIDNIGHT_METAL_MIN_K").is_some() {
            eprintln!("MIDNIGHT_METAL_MIN_K is set: the defaults are not in force; skipped");
            return;
        }
        assert!(!midnight_metal::thresholds().explicit);
        let srs = || ParamsKZG::unsafe_setup(K, rand_chacha::ChaCha8Rng::seed_from_u64(11));
        let gpu = byte_identity(
            "poseidon-chain k=16",
            PoseidonChain,
            srs,
            ResidentPolicy::Keep,
            d,
        );
        // every MSM of the proof had a table
        let held = gpu.held();
        assert!(held.tables > 0 && held.resident > 0, "{held:?}");
        let freed = gpu.evict(true);
        eprintln!("poseidon-chain k=16: evicted {freed} bytes");
        assert_eq!(freed, held.tables + held.resident);
        assert_eq!(gpu.held(), device::Held::default());
        midnight_metal::set_keep_resident(false);
    }
}

#[test]
fn a_proof_with_the_device_phases_admitted_verifies_and_the_erasure_succeeds() {
    use midnight_gpu_prover::{
        example::{PoseidonPreimage, example_input},
        input::SecretBytes,
    };
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(d) = detect() else { return };
    // let calls of the example circuit's size reach the kernels
    assert!(midnight_metal::set_min_k(1));
    let info = d.info().unwrap();
    eprintln!("device: {info:?}");
    let mut s = service(config(), d);
    let served = s
        .prove(
            &SecretBytes::new(example_input(1)),
            RequestOptions {
                zero_check: true,
                ..RequestOptions::default()
            },
        )
        .unwrap();
    let r = &served.record;
    eprintln!(
        "label {:?}, admitted {:?}, device calls {:?}, fallbacks {:?}",
        r.label,
        r.admitted.names(),
        r.device_calls,
        r.fallbacks
    );
    for f in &r.declines {
        eprintln!("record decline: {} {} x{}", f.phase, f.reason, f.count);
    }
    print_classes("midnight-metal");
    eprintln!(
        "wipe {:?}, device bytes now/peak {:?}",
        r.wipe,
        midnight_metal::device_bytes()
    );
    assert!(r.admitted.contains(Phases::ALL), "{:?}", r.admitted);
    eprintln!("held after the request: {:?}", s.held());
    assert!(s.retired().is_none(), "the erasure was not confirmed");
    assert!(r.wipe.checked && r.wipe.regions > 0, "{:?}", r.wipe);
    assert_eq!(r.wipe.erasure_failures, 0);
    assert!(r.verify_s > 0.0);
    // the stock verifier of the standard library accepts the proof
    let (inst, _) =
        midnight_gpu_prover::ServedRelation::decode_input(&PoseidonPreimage, &example_input(1))
            .unwrap();
    midnight_zk_stdlib::verify::<PoseidonPreimage, blake2b_simd::State, _>(
        s.verifier_params(),
        s.vk(),
        &inst,
        None,
        &served.proof,
    )
    .unwrap();
    assert!(midnight_metal::end_request());
    // the counters of the record are the crate's
    let st = midnight_metal::stats();
    for (class, served) in [
        ("msm", st.msm.served),
        ("fft", st.fft.served),
        ("ifft", st.ifft.served),
        ("quotient", st.quotient.served),
    ] {
        assert_eq!(
            r.device_calls.get(class).copied().unwrap_or(0),
            served,
            "{class}"
        );
    }
    midnight_metal::reset_min_k();
}

/// The read-back check through the device layer: a non-zero byte planted
/// after the zeroing (the crate's test switch) fails the request's erasure
/// and is counted.
#[test]
fn a_planted_nonzero_byte_fails_the_erasure_check() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(d) = detect() else { return };
    assert_eq!(d.begin_request(Phases::ALL), Phases::ALL);
    let clean = d.end_request(true).unwrap();
    assert!(clean.checked && clean.nonzero_regions == 0, "{clean:?}");
    // a per-request region, as a kernel makes one
    let fill = || {
        let mut s = midnight_metal::ctx::lock().unwrap();
        s.scratch("test-planted", 4096).unwrap().as_mut_slice::<u8>().fill(0x77);
    };
    assert_eq!(d.begin_request(Phases::ALL), Phases::ALL);
    fill();
    let w = d.end_request(true).unwrap();
    eprintln!("clean wipe: {w:?}");
    assert_eq!((w.regions, w.bytes, w.checked), (1, 4096, true));
    assert_eq!((w.nonzero_regions, w.check_failures), (0, 0));
    assert_eq!(d.begin_request(Phases::ALL), Phases::ALL);
    fill();
    midnight_metal::ctx::set_plant_nonzero(true);
    let e = d.end_request(true);
    midnight_metal::ctx::set_plant_nonzero(false);
    let e = e.expect_err("a non-zero region fails the erasure");
    eprintln!("planted: {e}");
    assert!(e.contains("1 non-zero regions, 0 failed checks"), "{e}");
    assert!(d.end_request(true).is_ok());
}

#[test]
fn the_device_layer_admits_phases_and_ends_a_request() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(d) = detect() else { return };
    d.reset_counters();
    // before the context exists no free memory is reported
    let m = d.mem().unwrap();
    assert!(m.total > 0);
    d.activate();
    let m = d.mem().unwrap();
    eprintln!("memory: {m:?}");
    assert_eq!(d.begin_request(Phases::NONE), Phases::NONE);
    assert_eq!(d.begin_request(Phases::ALL), Phases::ALL);
    assert!(d.revoke());
    assert!(!d.revoke());
    let w = d.end_request(true).unwrap();
    eprintln!("wipe: {w:?}");
    assert_eq!(w.erasure_failures, 0);
    assert_eq!(d.held(), device::Held::default());
    assert_eq!(d.evict(true), 0);
    print_classes("device layer");
}
