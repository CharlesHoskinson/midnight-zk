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

//! The device tests of a `midnight-accel` backend (`tests/metal.rs`,
//! `tests/vulkan.rs`). Each test returns early when no device of the backend
//! is detected.

use midnight_accel::{self as accel, Backend};
use midnight_gpu_prover::{
    RequestOptions,
    config::DeviceSelect,
    device::{self, Device, Phases},
};

use crate::common::*;

/// The `#[test]` functions for backend `$b`, whose device reports
/// `build_archs` starting with `$archs`.
macro_rules! device_tests {
    ($b:ty, $archs:literal) => {
        impl crate::accel::Tested for $b {
            const BUILD_ARCHS: &str = $archs;
        }

        #[cfg(feature = "test-seed")]
        mod seeded {
            #[test]
            fn seeded_small_proof_equals_cpu_proof_with_min_k_lowered() {
                crate::accel::seeded::small_proof_with_min_k_lowered::<$b>();
            }

            #[test]
            #[cfg_attr(
                debug_assertions,
                ignore = "minutes in a debug build: run with --release"
            )]
            fn seeded_k16_proof_equals_cpu_proof_with_default_thresholds() {
                crate::accel::seeded::k16_proof_with_default_thresholds::<$b>();
            }
        }

        #[test]
        fn a_planted_nonzero_byte_fails_the_erasure_check() {
            crate::accel::a_planted_nonzero_byte_fails_the_erasure_check::<$b>();
        }

        #[test]
        fn the_peak_of_a_small_request_excludes_the_pool_of_an_earlier_large_one() {
            crate::accel::the_peak_of_a_small_request_excludes_the_pool_of_an_earlier_large_one::<
                $b,
            >();
        }

        #[test]
        fn the_record_does_not_tell_a_sparse_witness_from_a_dense_one() {
            crate::accel::the_record_does_not_tell_a_sparse_witness_from_a_dense_one::<$b>();
        }
    };
}
pub(crate) use device_tests;

/// A backend under test.
pub trait Tested: Backend {
    /// The start of its device's `DeviceInfo::build_archs`.
    const BUILD_ARCHS: &str;
}

/// The hooks and counters are process-wide: one test at a time.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn detect<B: Tested>() -> Option<Box<dyn Device>> {
    let d = device::detect(DeviceSelect::Auto);
    let Some(info) = d.info() else {
        // CI sets this on a runner with a device, so that a broken detection
        // fails instead of skipping every test.
        let require = format!("{}_REQUIRE_DEVICE", B::ENV_PREFIX);
        assert!(
            std::env::var_os(&require).is_none(),
            "no {} device detected and {require} is set",
            B::NAME
        );
        eprintln!("no {} device detected: skipped", B::NAME);
        return None;
    };
    // a device tried before this backend's took the hooks
    if !info.build_archs.starts_with(B::BUILD_ARCHS) {
        eprintln!("another device took the hooks ({}): skipped", info.name);
        return None;
    }
    Some(d)
}

fn checked() -> RequestOptions {
    RequestOptions {
        zero_check: true,
        ..RequestOptions::default()
    }
}

/// Resets the size thresholds; `false` when `{ENV_PREFIX}_MIN_K` overrides
/// them.
fn default_thresholds<B: Backend>() -> bool {
    accel::reset_min_k::<B>();
    let var = format!("{}_MIN_K", B::ENV_PREFIX);
    if std::env::var_os(&var).is_some() {
        eprintln!("{var} is set: the defaults are not in force; skipped");
        return false;
    }
    assert!(!accel::thresholds::<B>().explicit);
    true
}

#[cfg(feature = "test-seed")]
pub mod seeded {
    use midnight_circuits::{
        hash::poseidon::PoseidonChip,
        instructions::{AssignmentInstructions, PublicInputInstructions, hash::HashCPU},
    };
    use midnight_gpu_prover::{
        Config, ProverService, ServedRelation,
        config::{ResidentPolicy, WslSelect},
        device::{Device, NoDevice},
        example::{PoseidonPreimage, example_input},
        input::SecretBytes,
        report::{Label, ProofRecord},
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

    /// `ROUNDS` chained Poseidon hashes; the last is public (as in
    /// `tests/gpu.rs`).
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

    /// Proves seeded on the device and on the CPU, and checks equal proof
    /// bytes, verification, the served classes, no fallback and a clean
    /// erasure. The prover makes no forward base-domain FFT (its forward
    /// transforms are inside the quotient), so `fft` is never offered.
    fn byte_identity<B: Backend, R: ServedRelation + Copy>(
        relation: R,
        srs: impl Fn() -> ParamsKZG<midnight_curves::Bls12>,
        resident: ResidentPolicy,
        dev: Box<dyn Device>,
    ) -> (ProverService<R>, ProofRecord) {
        let cfg = Config {
            threads: 8,
            wsl: WslSelect::Off,
            resident,
            ..Config::default()
        };
        let input = SecretBytes::new(example_input(4));
        accel::reset_stats::<B>();
        let mut gpu = ProverService::with_device(relation, srs(), cfg.clone(), dev).unwrap();
        let g = gpu.prove_seeded(&input, checked(), SEED).unwrap();
        let mut cpu = ProverService::with_device(relation, srs(), cfg, Box::new(NoDevice)).unwrap();
        let c = midnight_proofs::gpu::test_switches::without_hooks(|| {
            cpu.prove_seeded(&input, checked(), SEED).unwrap()
        });
        assert_eq!(c.record.label, Label::Cpu);
        assert!(
            g.proof == c.proof,
            "the device proof differs from the CPU proof"
        );

        let r = g.record;
        assert_eq!(r.label, Label::Gpu, "{:?}", r.degraded_reasons);
        assert!(r.admitted.contains(Phases::ALL));
        assert!(r.msm.count > 0);
        assert!(accel::stats::<B>().msm.served > 0);
        assert!(!r.device_calls.contains_key("msm"));
        assert!(!r.device_calls.contains_key("fft"));
        assert!(r.declines.iter().all(|d| d.phase != "msm" && d.phase != "fft"));
        assert!(r.device_calls["ifft"] > 0 && r.device_calls["quotient"] > 0);
        assert!(r.fallbacks.is_empty(), "{:?}", r.fallbacks);
        let w = &r.wipe;
        assert!(w.checked && w.regions > 0 && w.bytes > 0);
        assert_eq!(
            (w.nonzero_regions, w.check_failures, w.erasure_failures),
            (0, 0, 0)
        );
        assert!(gpu.retired().is_none());
        let (inst, _) = relation.decode_input(&example_input(4)).unwrap();
        midnight_zk_stdlib::verify::<R, blake2b_simd::State, _>(
            gpu.verifier_params(),
            gpu.vk(),
            &inst,
            None,
            &g.proof,
        )
        .unwrap();
        (gpu, r)
    }

    /// The example relation, with the thresholds lowered; the record's
    /// counters are the crate's.
    pub fn small_proof_with_min_k_lowered<B: Tested>() {
        let _g = serial();
        let Some(d) = detect::<B>() else { return };
        assert!(accel::set_min_k::<B>(1));
        let (_, r) = byte_identity::<B, _>(PoseidonPreimage, srs, ResidentPolicy::Evict, d);
        assert!(r.verify_s > 0.0);
        assert!(accel::end_request::<B>());
        let st = accel::stats::<B>();
        assert!(r.msm.count >= st.msm.served);
        for (class, served) in [
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
        accel::reset_min_k::<B>();
    }

    /// k = 16 reaches every class with the default thresholds.
    pub fn k16_proof_with_default_thresholds<B: Tested>() {
        let _g = serial();
        let Some(d) = detect::<B>() else { return };
        if !default_thresholds::<B>() {
            return;
        }
        let srs = || ParamsKZG::unsafe_setup(K, rand_chacha::ChaCha8Rng::seed_from_u64(11));
        let (gpu, _) = byte_identity::<B, _>(PoseidonChain, srs, ResidentPolicy::Keep, d);
        let held = gpu.held();
        assert!(held.tables > 0 && held.resident > 0, "{held:?}");
        let pooled = accel::pool_bytes::<B>();
        let freed = gpu.evict(true);
        assert_eq!(freed, held.tables + held.resident + pooled);
        assert_eq!(accel::pool_bytes::<B>(), 0);
        assert_eq!(gpu.held(), device::Held::default());
        accel::set_keep_resident::<B>(false);
    }
}

/// A non-zero byte planted after the zeroing fails the erasure check.
pub fn a_planted_nonzero_byte_fails_the_erasure_check<B: Tested>() {
    let _g = serial();
    let Some(d) = detect::<B>() else { return };
    assert_eq!(d.begin_request(Phases::NONE), Phases::NONE);
    assert_eq!(d.begin_request(Phases::ALL), Phases::ALL);
    assert!(d.revoke());
    assert!(!d.revoke());
    let clean = d.end_request(true).unwrap();
    assert!(clean.checked && clean.nonzero_regions == 0, "{clean:?}");
    // the check also reads back the idle pool, so empty it first
    let fill = || {
        let mut s = accel::ctx::lock::<B>().unwrap();
        s.scratch("test-planted", 4096).unwrap().as_mut_slice::<u8>().fill(0x77);
    };
    accel::release_pool::<B>();
    assert_eq!(d.begin_request(Phases::ALL), Phases::ALL);
    fill();
    let w = d.end_request(true).unwrap();
    assert_eq!((w.regions, w.bytes, w.checked), (1, 4096, true));
    assert_eq!((w.nonzero_regions, w.check_failures), (0, 0));
    assert_eq!(accel::pool_bytes::<B>(), 4096);
    accel::release_pool::<B>();
    assert_eq!(d.begin_request(Phases::ALL), Phases::ALL);
    fill();
    accel::ctx::set_plant_nonzero::<B>(true);
    let e = d.end_request(true);
    accel::ctx::set_plant_nonzero::<B>(false);
    let e = e.unwrap_err();
    assert!(e.contains("1 non-zero regions, 0 failed checks"), "{e}");
    assert!(d.end_request(true).is_ok());
}

/// The peak counts every buffer the request holds, new or from the pool, but
/// not the idle pool of an earlier, larger request.
pub fn the_peak_of_a_small_request_excludes_the_pool_of_an_earlier_large_one<B: Tested>() {
    const BIG: usize = 64 << 20;
    const SMALL: usize = 4096;
    let _g = serial();
    let Some(d) = detect::<B>() else { return };
    d.reset_counters();
    let scratch = |n: usize| {
        let mut s = accel::ctx::lock::<B>().unwrap();
        s.scratch("test-peak", n).unwrap().as_mut_slice::<u8>().fill(1);
    };
    // `(peak, bytes in use at the start, idle pool during the request)`
    let request = |n: usize| {
        let base = accel::ctx::in_use_bytes::<B>();
        assert_eq!(d.begin_request(Phases::ALL), Phases::ALL);
        scratch(n);
        let pool_during = accel::pool_bytes::<B>();
        let peak = d.counters().peak_bytes;
        d.end_request(false).unwrap();
        (peak, base, pool_during)
    };
    accel::release_pool::<B>();
    let (big, base, _) = request(BIG);
    let pooled = accel::pool_bytes::<B>();
    assert_eq!(big, base + BIG as u64);
    if pooled < BIG as u64 {
        eprintln!("the crate does not pool the large buffer: skipped");
        d.evict(true);
        return;
    }
    d.reset_counters();
    let (small, base, pool_during) = request(SMALL);
    assert_eq!(pool_during, pooled);
    assert_eq!(small, base + SMALL as u64);
    assert!(small < pooled);
    d.reset_counters();
    let pool_before = accel::pool_bytes::<B>();
    let (reused, base, pool_during) = request(BIG);
    assert_eq!(pool_before - pool_during, BIG as u64);
    assert_eq!(reused, base + BIG as u64);
    d.evict(true);
}

/// `ROWS` products of witness values at `K = 13`. The input is a
/// little-endian `u64` seed; seed 0 gives a witness of zeros.
mod mul_chain {
    use ff::Field;
    use midnight_circuits::{
        instructions::{ArithInstructions, AssignmentInstructions, PublicInputInstructions},
        types::AssignedNative,
    };
    use midnight_gpu_prover::ServedRelation;
    use midnight_proofs::{
        circuit::{Layouter, Value},
        plonk::Error,
    };
    use midnight_zk_stdlib::{Relation, ZkStdLib, ZkStdLibArch};

    type F = midnight_curves::Fq;

    pub const ROWS: usize = 6500;
    pub const K: u32 = 13;

    #[derive(Clone, Copy, Debug, Default)]
    pub struct MulChain;

    impl Relation for MulChain {
        type Instance = F;
        type Witness = Vec<F>;
        type Error = Error;

        fn format_instance(i: &F) -> Result<Vec<F>, Error> {
            Ok(vec![*i])
        }

        fn circuit(
            &self,
            std: &ZkStdLib,
            layouter: &mut impl Layouter<F>,
            _: Value<F>,
            witness: Value<Vec<F>>,
        ) -> Result<(), Error> {
            let w: Vec<AssignedNative<F>> =
                std.assign_many(layouter, &witness.transpose_vec(ROWS))?;
            let mut acc = w[0].clone();
            for x in &w[1..] {
                acc = std.mul(layouter, &acc, x, None)?;
            }
            std.constrain_as_public_input(layouter, &acc)
        }

        fn used_chips(&self) -> ZkStdLibArch {
            ZkStdLibArch::default()
        }

        fn write_relation<W: std::io::Write>(&self, _: &mut W) -> std::io::Result<()> {
            Ok(())
        }

        fn read_relation<R: std::io::Read>(_: &mut R) -> std::io::Result<Self> {
            Ok(MulChain)
        }
    }

    impl ServedRelation for MulChain {
        fn name(&self) -> String {
            "mul-chain".into()
        }
        fn decode_input(&self, input: &[u8]) -> Result<(F, Vec<F>), String> {
            let seed: [u8; 8] = input.try_into().map_err(|_| "8 bytes expected".to_string())?;
            let seed = F::from(u64::from_le_bytes(seed));
            let w: Vec<F> = (0..ROWS as u64).map(|i| seed * F::from(i + 2)).collect();
            let product = w.iter().fold(F::ONE, |a, x| a * x);
            Ok((product, w))
        }
        fn k(&self) -> Option<u32> {
            Some(K)
        }
    }
}

/// The crate serves a dense witness's MSMs and declines a sparse one's, but
/// the record's MSM total, declines, calls, fallbacks and label are the same.
pub fn the_record_does_not_tell_a_sparse_witness_from_a_dense_one<B: Tested>() {
    use midnight_gpu_prover::{ProverService, input::SecretBytes};
    use rand::SeedableRng;
    let _g = serial();
    let Some(d) = detect::<B>() else { return };
    if !default_thresholds::<B>() {
        return;
    }
    let params = midnight_proofs::poly::kzg::params::ParamsKZG::unsafe_setup(
        mul_chain::K,
        rand_chacha::ChaCha8Rng::seed_from_u64(5),
    );
    let mut s = ProverService::with_device(mul_chain::MulChain, params, config(), d).unwrap();
    // the record, and the MSMs the crate served and declined as sparse
    let mut prove = |seed: u64| {
        accel::reset_stats::<B>();
        let served = s.prove(&SecretBytes::new(seed.to_le_bytes().to_vec()), checked()).unwrap();
        let declined: u64 = accel::declines::<B>()
            .into_iter()
            .filter(|(c, r, _)| *c == "msm" && *r == "below-min-k")
            .map(|(_, _, n)| n)
            .sum();
        (served.record, accel::stats::<B>().msm.served, declined)
    };
    let (dense, dense_served, dense_declined) = prove(0x5eed);
    let (sparse, sparse_served, sparse_declined) = prove(0);
    // the crate tells them apart, the record does not
    assert!(dense_served > sparse_served);
    assert!(sparse_declined > dense_declined);
    assert_eq!(
        dense_served + dense_declined,
        sparse_served + sparse_declined
    );
    assert!(dense.msm.count > 0);
    assert_eq!(dense.msm.count, sparse.msm.count);
    assert_eq!(dense.declines, sparse.declines);
    assert_eq!(dense.device_calls, sparse.device_calls);
    assert_eq!(dense.fallbacks, sparse.fallbacks);
    assert_eq!(dense.label, sparse.label);
    assert_eq!(
        dense.label,
        midnight_gpu_prover::report::Label::Gpu,
        "{:?}",
        dense.degraded_reasons
    );
    assert!(dense.declines.iter().all(|f| f.phase != "msm"));
    assert!(!dense.device_calls.contains_key("msm"));
}
