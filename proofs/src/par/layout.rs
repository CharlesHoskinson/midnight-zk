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

//! The region layout of a proving key that holds none from keygen.
//!
//! `keygen_pk` captures the region starts of its synthesis
//! (`ProvingKey::region_starts`) and the prover hands them to the floor
//! planner, which then skips its shape pass. A key read from bytes holds no
//! layout, so the stock prover runs the shape pass on every proof. Here the
//! first proof made with such a key measures the layout of its own
//! synthesis, which runs the shape pass anyway, into the key
//! (`ProvingKey::region_starts_cache`): for each region, in synthesis order,
//! its start, its columns, its row count and the number of global constants
//! it assigns ([`MeasuredLayout`]).
//!
//! A later proof with the key places its regions at those starts without the
//! shape pass, and checks each region as it is assigned
//! ([`FloorPlanner::synthesize_checking_layout`]): the assignment of the
//! region is measured as the shape pass measures it, and the region must
//! have the measured columns, row count and constants of the layout at its
//! index, and the start the shape pass would give it after the regions
//! before it; the circuit must open as many regions as the layout. The start
//! of a region in the shape pass depends only on those quantities of the
//! regions before it, so a proof that passes the check has the stock
//! witness, and the stock proof.
//!
//! A proof whose circuit does not pass the check (a circuit of another
//! shape: more or fewer regions, regions of other sizes or columns), or
//! whose synthesis fails in any way during the check, discards its witness
//! and synthesizes again on an empty witness with the stock shape pass: it
//! makes the proof, or returns the error, that the stock prover makes. It
//! never panics on a mismatch, and it leaves the stored layout as it is, so
//! later proofs that match the stored layout keep skipping the shape pass.
//!
//! What the check cannot see is a region closure that touches other cells
//! when it runs in the shape pass (with unknown values, and on its first
//! call) than when it runs in the assignment: the shape pass would then
//! place the region by cells that the assignment does not touch. A circuit
//! doing so in its first proof is caught there (its assignment is measured
//! too, and its layout is not stored); the same holds for every later proof
//! of a circuit whose cells do not depend on witness values or on the
//! closure's own state across its calls, which the upstream reuse of the
//! keygen layout already requires.
//!
//! Two proofs that start at once with a key without a layout both run the
//! shape pass and measure layouts; the first to finish stores its own, the
//! other is dropped. A proof that fails during synthesis stores nothing.
//! Each key holds its own layout, so keys of circuits of different shapes
//! never share one. Planners other than [`SimpleFloorPlanner`] measure no
//! layout (the default of [`FloorPlanner::synthesize_measuring_layout`]), so
//! their proofs run the stock synthesis every time.
//!
//! [`SimpleFloorPlanner`]: crate::circuit::SimpleFloorPlanner

use std::sync::OnceLock;

use ff::Field;

use crate::{
    circuit::{RegionStart, layouter::MeasuredLayout, layouter::SyncDeps},
    plonk::{Assignment, Circuit, Column, Error, Fixed, FloorPlanner},
};

#[cfg(test)]
thread_local! {
    /// The proofs of this thread that did not pass the check of the stored
    /// layout and synthesized again (tests only).
    static RETRIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The number of proofs made on this thread since the last call that did
/// not pass the check of the stored layout and synthesized again with the
/// shape pass, clearing the count (tests only).
#[cfg(test)]
pub(crate) fn take_retries() -> usize {
    RETRIES.with(|r| r.replace(0))
}

/// Synthesizes `circuit` into `cs` as the stock prover does with the
/// layout `keygen` (`ProvingKey::region_starts`). When `keygen` is `None`:
/// with the layout in `cache` if the circuit matches it, and otherwise
/// (after `reset` empties `cs`) with the stock shape pass; when `cache` is
/// empty, with the stock shape pass, storing the layout it measures in
/// `cache`.
pub(crate) fn synthesize<F, CS, C>(
    cs: &mut CS,
    reset: impl FnOnce(&mut CS),
    circuit: &C,
    config: C::Config,
    constants: Vec<Column<Fixed>>,
    keygen: Option<&[RegionStart]>,
    cache: &OnceLock<MeasuredLayout>,
) -> Result<(), Error>
where
    F: Field,
    CS: Assignment<F> + SyncDeps,
    C: Circuit<F>,
{
    if keygen.is_some() {
        // The upstream path, as the stock prover takes it.
        return C::FloorPlanner::synthesize_with_cached_regions(
            cs, circuit, config, constants, keygen,
        );
    }
    if let Some(layout) = cache.get() {
        if C::FloorPlanner::synthesize_checking_layout(
            cs,
            circuit,
            config.clone(),
            constants.clone(),
            layout,
        ) {
            return Ok(());
        }
        #[cfg(test)]
        RETRIES.with(|r| r.set(r.get() + 1));
        // The partial witness is discarded; the stock synthesis of a key
        // without a layout runs on an empty one.
        reset(cs);
        return C::FloorPlanner::synthesize_with_cached_regions(
            cs, circuit, config, constants, None,
        );
    }
    // The measuring synthesis is the stock synthesis of a key without a
    // layout, which also records the layout it computes (`None` when the
    // planner does not measure, as for the default of the trait).
    if let Some(measured) =
        C::FloorPlanner::synthesize_measuring_layout(cs, circuit, config, constants)?
    {
        // Another proof with this key may have stored its layout first: the
        // proofs after it are checked against that one.
        let _ = cache.set(measured);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};

    use midnight_curves::Fq;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use crate::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        plonk::{Circuit, ConstraintSystem, Error, create_proof, keygen_pk, keygen_vk_with_k},
        test_circuit::{
            SEEDED,
            chain::{Chain, ChainConfig, Scheme, chain_setup, prove_seeded, read_back, verifies},
        },
        transcript::{CircuitTranscript, Transcript},
    };

    /// A key read from bytes captures the layout on its first proof: the
    /// layout equals the one of keygen, and the first proof and every later
    /// one equal the proofs of the key from keygen, byte for byte.
    #[test]
    fn read_key_reuses_layout_and_proves_as_keygen_key() {
        let chain = Chain {
            rows: 200,
            start: 7,
        };
        let (params, pk) = chain_setup(9, &chain);
        let keygen_layout = pk.region_starts.clone().expect("the simple planner captures");
        assert_eq!(keygen_layout.len(), 2);

        let read = read_back(&pk);
        assert!(read.region_starts.is_none());
        assert!(read.region_starts_cache.get().is_none());

        let _seeded = SEEDED.lock().unwrap_or_else(|e| e.into_inner());
        crate::plonk::prover::test_blinding::set(Some(0x5eed));
        let reference = prove_seeded(&params, &pk, &chain, 1);
        // the key from keygen never fills the cache: it has its own layout
        assert!(pk.region_starts_cache.get().is_none());
        let first = prove_seeded(&params, &read, &chain, 1);
        assert_eq!(
            read.region_starts_cache.get().map(|l| l.starts()),
            Some(keygen_layout)
        );
        super::take_retries();
        let second = prove_seeded(&params, &read, &chain, 1);
        let third = prove_seeded(&params, &read, &chain, 1);
        let other_seed = prove_seeded(&params, &read, &chain, 2);
        let other_seed_ref = prove_seeded(&params, &pk, &chain, 2);
        crate::plonk::prover::test_blinding::set(None);
        // the later proofs reused the layout, without the shape pass
        assert_eq!(super::take_retries(), 0);

        assert_eq!(first, reference);
        assert_eq!(second, reference);
        assert_eq!(third, reference);
        assert_eq!(other_seed, other_seed_ref);
        assert_ne!(other_seed, reference);
        assert!(verifies(&params, &read, &chain, &first));
        assert!(verifies(&params, &read, &chain, &second));
    }

    /// Threads proving at once with one key read from bytes, before any
    /// layout is cached, all make the proof of the key from keygen.
    #[test]
    fn read_key_concurrent_first_proofs() {
        let chain = Chain {
            rows: 300,
            start: 11,
        };
        let (params, pk) = chain_setup(10, &chain);
        let read = read_back(&pk);

        let _seeded = SEEDED.lock().unwrap_or_else(|e| e.into_inner());
        crate::plonk::prover::test_blinding::set(Some(0xc0c0));
        let reference = prove_seeded(&params, &pk, &chain, 3);
        let threads = 3;
        let barrier = Arc::new(Barrier::new(threads));
        let proofs: Vec<Vec<u8>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..threads)
                .map(|_| {
                    let barrier = barrier.clone();
                    let (params, read, chain) = (&params, &read, &chain);
                    s.spawn(move || {
                        barrier.wait();
                        prove_seeded(params, read, chain, 3)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        // and two more at once with the layout cached
        let later: Vec<Vec<u8>> = std::thread::scope(|s| {
            let a = s.spawn(|| prove_seeded(&params, &read, &chain, 3));
            let b = s.spawn(|| prove_seeded(&params, &read, &chain, 3));
            vec![a.join().unwrap(), b.join().unwrap()]
        });
        crate::plonk::prover::test_blinding::set(None);

        assert_eq!(
            read.region_starts_cache.get().map(|l| l.starts()),
            pk.region_starts
        );
        for proof in proofs.iter().chain(later.iter()) {
            assert_eq!(*proof, reference);
        }
    }

    /// Keys of two circuits with different layouts, both read from bytes,
    /// keep their own layouts.
    #[test]
    fn read_keys_of_different_shapes_keep_their_layouts() {
        let short = Chain {
            rows: 100,
            start: 3,
        };
        let long = Chain {
            rows: 400,
            start: 3,
        };
        let (params, pk_short) = chain_setup(10, &short);
        let vk_long = keygen_vk_with_k::<Fq, Scheme, _>(&params, &long, 10).unwrap();
        let pk_long = keygen_pk(vk_long, &long).unwrap();
        let (read_short, read_long) = (read_back(&pk_short), read_back(&pk_long));
        assert_ne!(pk_short.region_starts, pk_long.region_starts);

        let _seeded = SEEDED.lock().unwrap_or_else(|e| e.into_inner());
        crate::plonk::prover::test_blinding::set(Some(0xd1ff));
        for _ in 0..2 {
            assert_eq!(
                prove_seeded(&params, &read_short, &short, 4),
                prove_seeded(&params, &pk_short, &short, 4)
            );
            assert_eq!(
                prove_seeded(&params, &read_long, &long, 4),
                prove_seeded(&params, &pk_long, &long, 4)
            );
        }
        crate::plonk::prover::test_blinding::set(None);
        assert_eq!(
            read_short.region_starts_cache.get().map(|l| l.starts()),
            pk_short.region_starts
        );
        assert_eq!(
            read_long.region_starts_cache.get().map(|l| l.starts()),
            pk_long.region_starts
        );
    }

    /// A proof whose synthesis fails stores no layout.
    #[test]
    fn failed_synthesis_stores_no_layout() {
        #[derive(Clone)]
        struct Failing(Chain);
        impl Circuit<Fq> for Failing {
            type Config = ChainConfig;
            type FloorPlanner = SimpleFloorPlanner;
            #[cfg(feature = "circuit-params")]
            type Params = ();
            fn without_witnesses(&self) -> Self {
                self.clone()
            }
            fn configure(meta: &mut ConstraintSystem<Fq>) -> ChainConfig {
                Chain::configure(meta)
            }
            fn synthesize(
                &self,
                config: ChainConfig,
                mut layouter: impl Layouter<Fq>,
            ) -> Result<(), Error> {
                layouter.assign_region(
                    || "unknown",
                    |mut region| {
                        region.assign_advice(|| "x", config.advice[0], 0, Value::<Fq>::unknown)?;
                        Ok(())
                    },
                )
            }
        }
        let chain = Chain { rows: 10, start: 1 };
        let (params, pk) = chain_setup(6, &chain);
        let read = read_back(&pk);
        let mut transcript = CircuitTranscript::<blake2b_simd::State>::init();
        let res = create_proof::<Fq, Scheme, _, _>(
            &params,
            &read,
            &Failing(chain.clone()),
            #[cfg(feature = "committed-instances")]
            0,
            &[&chain.instance()],
            &mut transcript,
            ChaCha20Rng::seed_from_u64(0),
        );
        assert!(res.is_err());
        assert!(read.region_starts_cache.get().is_none());
    }
}
