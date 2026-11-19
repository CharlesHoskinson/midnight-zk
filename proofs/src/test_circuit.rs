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

//! A seeded random circuit for tests: random gates, a batched LogUp lookup,
//! a copy constraint and a trash argument (unit tests, and the feature
//! `gpu-test`).

use std::cell::Cell;

use ff::Field;
use midnight_curves::{Bls12, Fq};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

use crate::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{
        Advice, Circuit, Column, ConstraintSystem, Constraints, Error, Expression, Fixed, Instance,
        ProvingKey, Selector, TableColumn, VirtualCells, create_proof, keygen_pk, keygen_vk_with_k,
        prepare, trash,
    },
    poly::{
        Rotation,
        commitment::Guard,
        kzg::{KZGCommitmentScheme, params::ParamsKZG},
    },
    transcript::{CircuitTranscript, Transcript},
};

thread_local! {
    static SHAPE: Cell<(u64, bool)> = const { Cell::new((0, false)) };
}

/// The commitment scheme of the test circuit.
pub(crate) type Scheme = KZGCommitmentScheme<Bls12>;

/// Sets the seed of the circuit shape `configure` builds on this thread and
/// whether the witness must satisfy the circuit.
pub(crate) fn set_shape(seed: u64, satisfied: bool) {
    SHAPE.with(|s| s.set((seed, satisfied)));
}

// Static configuration of the legacy fixture reads a thread-local shape.
// Restore an enclosing key generation if this worker runs another fixture.
#[cfg(feature = "gpu")]
pub(crate) fn with_shape<T>(seed: u64, satisfied: bool, f: impl FnOnce() -> T) -> T {
    struct Restore((u64, bool));
    impl Drop for Restore {
        fn drop(&mut self) {
            SHAPE.set(self.0);
        }
    }
    let _restore = Restore(SHAPE.replace((seed, satisfied)));
    f()
}

/// A random expression over the columns of `cfg` of depth at most `depth`.
fn rand_expr(
    rng: &mut ChaCha20Rng,
    vc: &mut VirtualCells<'_, Fq>,
    cfg: &Cfg,
    depth: u32,
) -> Expression<Fq> {
    let rots = [
        Rotation::cur(),
        Rotation::next(),
        Rotation::prev(),
        Rotation(2),
    ];
    let leaf = |rng: &mut ChaCha20Rng, vc: &mut VirtualCells<'_, Fq>| -> Expression<Fq> {
        let r = rots[(rng.next_u32() % 4) as usize];
        match rng.next_u32() % 5 {
            0 => vc.query_advice(cfg.a[(rng.next_u32() % 3) as usize], r),
            1 => vc.query_fixed(cfg.f[(rng.next_u32() % 2) as usize], r),
            2 => vc.query_instance(cfg.inst, r),
            3 => Expression::Constant(Fq::from(rng.next_u64())),
            _ => vc.query_advice(cfg.a[0], Rotation::cur()),
        }
    };
    if depth == 0 || rng.next_u32().is_multiple_of(4) {
        return leaf(rng, vc);
    }
    let x = rand_expr(rng, vc, cfg, depth - 1);
    match rng.next_u32() % 6 {
        0 => x + rand_expr(rng, vc, cfg, depth - 1),
        1 => x - rand_expr(rng, vc, cfg, depth - 1),
        2 => x * rand_expr(rng, vc, cfg, depth - 1),
        3 => -x,
        // the evaluator lowers a product with the constant 2 to a doubling
        4 => x * Expression::Constant(Fq::from(2)),
        _ => x.clone() * x,
    }
}

/// The columns of [`RandCircuit`].
#[derive(Clone, Debug)]
pub(crate) struct Cfg {
    #[cfg(feature = "gpu")]
    satisfied: bool,
    a: [Column<Advice>; 3],
    f: [Column<Fixed>; 2],
    q: Column<Fixed>,
    zero: Column<Fixed>,
    s: Selector,
    t: TableColumn,
    inst: Column<Instance>,
}

/// A circuit whose gates, lookup inputs and trash constraints are random,
/// from the shape seed of [`set_shape`]. Its witness satisfies the lookup
/// and the copy constraint. Unsatisfied, the gates and the trash argument
/// do not hold (a numerator test needs no valid witness); satisfied, every
/// gate is multiplied by a zero fixed column and the trash selector is zero.
#[derive(Clone, Debug, Default)]
pub(crate) struct RandCircuit {
    /// Seed of the witness.
    pub seed: u64,
}

const TABLE: u64 = 16;
const ROWS: usize = 20;

impl Circuit<Fq> for RandCircuit {
    type Config = Cfg;
    type FloorPlanner = SimpleFloorPlanner;
    #[cfg(feature = "circuit-params")]
    type Params = ();

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(meta: &mut ConstraintSystem<Fq>) -> Cfg {
        let (seed, satisfied) = SHAPE.with(|s| s.get());
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let cfg = Cfg {
            #[cfg(feature = "gpu")]
            satisfied,
            a: [
                meta.advice_column(),
                meta.advice_column(),
                meta.advice_column(),
            ],
            f: [meta.fixed_column(), meta.fixed_column()],
            q: meta.fixed_column(),
            zero: meta.fixed_column(),
            s: meta.complex_selector(),
            t: meta.lookup_table_column(),
            inst: meta.instance_column(),
        };
        meta.enable_equality(cfg.a[0]);
        meta.enable_equality(cfg.a[1]);
        meta.enable_equality(cfg.inst);
        if rng.next_u32().is_multiple_of(2) {
            meta.enable_equality(cfg.f[0]);
        }
        let ngates = 1 + rng.next_u32() % 3;
        for _ in 0..ngates {
            let nc = 1 + rng.next_u32() % 3;
            let mut r2 = ChaCha20Rng::seed_from_u64(rng.next_u64());
            let c = cfg.clone();
            meta.create_gate("random", move |vc| {
                let z = vc.query_fixed(c.zero, Rotation::cur());
                Constraints::without_selector(
                    (0..nc)
                        .map(|_| {
                            let e = rand_expr(&mut r2, vc, &c, 3);
                            if satisfied { z.clone() * e } else { e }
                        })
                        .collect::<Vec<_>>(),
                )
            });
        }
        // A batched lookup of a[2] at a few rotations (all in the table).
        let ninputs = 1 + (rng.next_u32() % 4) as i32;
        meta.batched_lookup("lookup", Some(cfg.s), |vc| {
            let inputs = (0..ninputs).map(|r| vc.query_advice(cfg.a[2], Rotation(r))).collect();
            vec![(inputs, cfg.t)]
        });
        // A trash argument with a fixed selector and random constraints.
        let mut r3 = ChaCha20Rng::seed_from_u64(rng.next_u64());
        let ntrash = 1 + r3.next_u32() % 2;
        let mut trash_parts = None;
        let c = cfg.clone();
        meta.create_gate("trash queries", |vc| {
            let q = vc.query_fixed(c.q, Rotation::cur());
            let exprs: Vec<_> = (0..ntrash).map(|_| rand_expr(&mut r3, vc, &c, 2)).collect();
            trash_parts = Some((q.clone(), exprs));
            Constraints::without_selector(vec![q * Expression::Constant(Fq::ZERO)])
        });
        let (q, exprs) = trash_parts.unwrap();
        let idx = meta.trashcans.len();
        meta.trashcans.push(trash::Argument::new(idx, q, exprs));
        cfg
    }

    fn synthesize(&self, cfg: Cfg, mut layouter: impl Layouter<Fq>) -> Result<(), Error> {
        #[cfg(not(feature = "gpu"))]
        let satisfied = SHAPE.with(|s| s.get().1);
        #[cfg(feature = "gpu")]
        let satisfied = cfg.satisfied;
        layouter.assign_table(
            || "table",
            |mut table| {
                for i in 0..TABLE {
                    table.assign_cell(|| "t", cfg.t, i as usize, || Value::known(Fq::from(i)))?;
                }
                Ok(())
            },
        )?;
        layouter.assign_region(
            || "main",
            |mut region| {
                let mut rng = ChaCha20Rng::seed_from_u64(self.seed ^ 0x5eed);
                let mut first: Option<(Fq, crate::circuit::Cell)> = None;
                for row in 0..ROWS {
                    let v0 = Fq::from(rng.next_u64());
                    let c0 = region.assign_advice(|| "a0", cfg.a[0], row, || Value::known(v0))?;
                    let v1 = if row == 1 {
                        first.unwrap().0
                    } else {
                        Fq::from(rng.next_u64())
                    };
                    let c1 = region.assign_advice(|| "a1", cfg.a[1], row, || Value::known(v1))?;
                    if row == 0 {
                        first = Some((v0, c0.cell()));
                    }
                    if row == 1 {
                        region.constrain_equal(c1.cell(), first.unwrap().1)?;
                    }
                    let t = Fq::from(rng.next_u64() % TABLE);
                    region.assign_advice(|| "a2", cfg.a[2], row, || Value::known(t))?;
                    for f in cfg.f {
                        region.assign_fixed(
                            || "f",
                            f,
                            row,
                            || Value::known(Fq::from(rng.next_u64())),
                        )?;
                    }
                    let q = if satisfied {
                        Fq::ZERO
                    } else {
                        Fq::from(rng.next_u64() % 2)
                    };
                    region.assign_fixed(|| "q", cfg.q, row, || Value::known(q))?;
                    if row + 4 < ROWS {
                        cfg.s.enable(&mut region, row)?;
                    }
                }
                Ok(())
            },
        )
    }
}

/// A proof using an existing key needs only the column configuration for
/// synthesis. Bind it before the prover can yield to another job.
#[cfg(feature = "gpu")]
#[derive(Clone)]
pub(crate) struct ConfiguredCircuit {
    circuit: RandCircuit,
    cfg: Cfg,
}

#[cfg(feature = "gpu")]
impl ConfiguredCircuit {
    pub(crate) fn new(seed: u64, satisfied: bool) -> Self {
        let cfg = with_shape(seed, satisfied, || {
            RandCircuit::configure(&mut ConstraintSystem::default())
        });
        Self {
            circuit: RandCircuit { seed },
            cfg,
        }
    }
}

#[cfg(feature = "gpu")]
impl Circuit<Fq> for ConfiguredCircuit {
    type Config = ();
    type FloorPlanner = SimpleFloorPlanner;
    #[cfg(feature = "circuit-params")]
    type Params = ();

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(_: &mut ConstraintSystem<Fq>) {}

    fn synthesize(&self, (): (), layouter: impl Layouter<Fq>) -> Result<(), Error> {
        self.circuit.synthesize(self.cfg.clone(), layouter)
    }
}

/// The SRS and keys of the circuit of shape `seed` at `2^k` rows.
pub(crate) fn setup(
    seed: u64,
    satisfied: bool,
    k: u32,
) -> (ParamsKZG<Bls12>, ProvingKey<Fq, Scheme>) {
    set_shape(seed, satisfied);
    let params = ParamsKZG::<Bls12>::unsafe_setup(k, ChaCha20Rng::seed_from_u64(seed));
    let circuit = RandCircuit { seed };
    let vk = keygen_vk_with_k::<Fq, Scheme, _>(&params, &circuit, k).unwrap();
    let pk = keygen_pk(vk, &circuit).unwrap();
    (params, pk)
}

/// The instance of a proof.
pub(crate) fn instance(seed: u64) -> [Fq; 1] {
    [Fq::from(seed.wrapping_mul(31).wrapping_add(7))]
}

/// The shape and the blinding seed are process-wide: one seeded proof at a
/// time, so two tests that prove at once keep their own draws.
pub(crate) static SEEDED: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// A proof for the circuit of shape `seed` with every random draw seeded
/// (the quotient blinding included). Returns the proof bytes, and whether
/// the proof verifies.
pub(crate) fn prove(
    params: &ParamsKZG<Bls12>,
    pk: &ProvingKey<Fq, Scheme>,
    seed: u64,
    satisfied: bool,
) -> (Vec<u8>, bool) {
    let seeded = SEEDED.lock().unwrap_or_else(|e| e.into_inner());
    #[cfg(not(feature = "gpu"))]
    set_shape(seed, satisfied);
    // The seed is removed on every exit, unwind included, and before the
    // lock is released (locals drop in reverse order).
    #[cfg(feature = "gpu")]
    struct ClearSeed;
    #[cfg(feature = "gpu")]
    impl Drop for ClearSeed {
        fn drop(&mut self) {
            crate::plonk::prover::test_blinding::set(None);
        }
    }
    crate::plonk::prover::test_blinding::set(Some(seed ^ 0xb11d));
    #[cfg(feature = "gpu")]
    let clear = ClearSeed;
    let inst = instance(seed);
    let mut transcript = CircuitTranscript::<blake2b_simd::State>::init();
    create_proof::<Fq, Scheme, _, _>(
        params,
        pk,
        #[cfg(not(feature = "gpu"))]
        &RandCircuit { seed },
        #[cfg(feature = "gpu")]
        &ConfiguredCircuit::new(seed, satisfied),
        #[cfg(feature = "committed-instances")]
        0,
        &[&inst],
        &mut transcript,
        ChaCha20Rng::seed_from_u64(seed ^ 0x9e37),
    )
    .unwrap();
    #[cfg(feature = "gpu")]
    drop(clear);
    #[cfg(not(feature = "gpu"))]
    crate::plonk::prover::test_blinding::set(None);
    drop(seeded);
    let proof = transcript.finalize();
    let mut transcript = CircuitTranscript::<blake2b_simd::State>::init_from_bytes(&proof);
    let ok = prepare::<Fq, Scheme, _>(
        pk.get_vk(),
        #[cfg(feature = "committed-instances")]
        &[],
        &[&inst],
        &mut transcript,
    )
    .is_ok_and(|g| g.verify(&params.verifier_params()).is_ok());
    (proof, ok)
}

/// A chain of squarings, as the circuit of the rc1 benchmark of the prover
/// service, with seeded proofs (tests only).
#[cfg(test)]
pub(crate) mod chain {
    use ff::Field;
    use midnight_curves::{Bls12, Fq};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use crate::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        plonk::{
            Advice, Circuit, Column, ConstraintSystem, Constraints, Error, Expression, Fixed,
            Instance, ProvingKey, Selector, create_proof, keygen_pk, keygen_vk_with_k, prepare,
        },
        poly::{
            Rotation,
            commitment::Guard,
            kzg::{KZGCommitmentScheme, params::ParamsKZG},
        },
        transcript::{CircuitTranscript, Transcript},
        utils::SerdeFormat,
    };

    pub(crate) type Scheme = KZGCommitmentScheme<Bls12>;

    fn step(x: Fq) -> Fq {
        x.square() + Fq::from(3)
    }

    /// A chain of squarings over four advice columns, as the circuit of the
    /// rc1 benchmark, with a second region holding a fixed column so that
    /// the layout has more than one region.
    #[derive(Clone)]
    pub(crate) struct Chain {
        pub(crate) rows: usize,
        pub(crate) start: u64,
    }

    #[derive(Clone)]
    pub(crate) struct ChainConfig {
        pub(crate) advice: [Column<Advice>; 4],
        fixed: Column<Fixed>,
        public: Column<Instance>,
        active: Selector,
    }

    impl Chain {
        pub(crate) fn instance(&self) -> [Fq; 2] {
            let start = Fq::from(self.start);
            [start, (0..4 * self.rows).fold(start, |x, _| step(x))]
        }
    }

    impl Circuit<Fq> for Chain {
        type Config = ChainConfig;
        type FloorPlanner = SimpleFloorPlanner;
        #[cfg(feature = "circuit-params")]
        type Params = ();

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(meta: &mut ConstraintSystem<Fq>) -> ChainConfig {
            let advice = std::array::from_fn(|_| meta.advice_column());
            let fixed = meta.fixed_column();
            let public = meta.instance_column();
            let active = meta.selector();
            meta.enable_equality(advice[0]);
            meta.enable_equality(advice[1]);
            meta.enable_equality(public);
            meta.create_gate("four squarings", |cells| {
                let mut chain = advice.map(|col| cells.query_advice(col, Rotation::cur())).to_vec();
                chain.push(cells.query_advice(advice[0], Rotation::next()));
                Constraints::with_selector(
                    active,
                    chain
                        .windows(2)
                        .map(|pair| {
                            pair[0].clone() * pair[0].clone() + Expression::Constant(Fq::from(3))
                                - pair[1].clone()
                        })
                        .collect(),
                )
            });
            meta.create_gate("fixed copy", |cells| {
                let f = cells.query_fixed(fixed, Rotation::cur());
                let a = cells.query_advice(advice[1], Rotation::cur());
                Constraints::without_selector(vec![f.clone() * (a - f)])
            });
            ChainConfig {
                advice,
                fixed,
                public,
                active,
            }
        }

        fn synthesize(
            &self,
            config: ChainConfig,
            mut layouter: impl Layouter<Fq>,
        ) -> Result<(), Error> {
            let (start, end) = layouter.assign_region(
                || "chain",
                |mut region| {
                    let mut x = Fq::from(self.start);
                    let mut start = None;
                    for row in 0..self.rows {
                        config.active.enable(&mut region, row)?;
                        for col in config.advice {
                            let cell =
                                region.assign_advice(|| "x", col, row, || Value::known(x))?;
                            if row == 0 && col == config.advice[0] {
                                start = Some(cell.cell());
                            }
                            x = step(x);
                        }
                    }
                    let end = region.assign_advice(
                        || "endpoint",
                        config.advice[0],
                        self.rows,
                        || Value::known(x),
                    )?;
                    Ok((start.unwrap(), end.cell()))
                },
            )?;
            layouter.assign_region(
                || "fixed",
                |mut region| {
                    for row in 0..3 {
                        let v = Fq::from(5 + row as u64);
                        region.assign_fixed(|| "f", config.fixed, row, || Value::known(v))?;
                        region.assign_advice(|| "a", config.advice[1], row, || Value::known(v))?;
                    }
                    Ok(())
                },
            )?;
            layouter.constrain_instance(start, config.public, 0)?;
            layouter.constrain_instance(end, config.public, 1)
        }
    }

    /// How [`Reshaped`] departs from the regions of its chain.
    #[derive(Clone, Copy, Debug)]
    pub(crate) enum Reshape {
        /// Only the chain region: one region fewer.
        Fewer,
        /// The regions of the chain and one more after them.
        More,
    }

    /// The circuit of `chain` (its columns and gates) with a different
    /// sequence of regions: proved with a key of `chain`, it makes a proof
    /// that does not verify, but the prover still makes it.
    #[derive(Clone)]
    pub(crate) struct Reshaped {
        pub(crate) chain: Chain,
        pub(crate) reshape: Reshape,
    }

    impl Circuit<Fq> for Reshaped {
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
            match self.reshape {
                Reshape::Fewer => {
                    let start = layouter.assign_region(
                        || "chain only",
                        |mut region| {
                            let mut x = Fq::from(self.chain.start);
                            let mut start = None;
                            for row in 0..self.chain.rows {
                                config.active.enable(&mut region, row)?;
                                for col in config.advice {
                                    let cell = region.assign_advice(
                                        || "x",
                                        col,
                                        row,
                                        || Value::known(x),
                                    )?;
                                    start.get_or_insert(cell.cell());
                                    x = step(x);
                                }
                            }
                            Ok(start.unwrap())
                        },
                    )?;
                    layouter.constrain_instance(start, config.public, 0)
                }
                Reshape::More => {
                    self.chain.synthesize(config.clone(), layouter.namespace(|| "chain"))?;
                    layouter.assign_region(
                        || "extra",
                        |mut region| {
                            for row in 0..5 {
                                let v = Fq::from(11 + row as u64);
                                region.assign_advice(
                                    || "e",
                                    config.advice[1],
                                    row,
                                    || Value::known(v),
                                )?;
                            }
                            Ok(())
                        },
                    )
                }
            }
        }
    }

    /// A seeded proof of `chain` with `pk`; the caller holds
    /// `crate::test_circuit::SEEDED` and has set the blinding seed.
    pub(crate) fn prove_seeded(
        params: &ParamsKZG<Bls12>,
        pk: &ProvingKey<Fq, Scheme>,
        chain: &Chain,
        seed: u64,
    ) -> Vec<u8> {
        prove_circuit_seeded(params, pk, chain, &chain.instance(), seed)
            .expect("the prover makes the proof")
    }

    /// A seeded proof of any circuit with the columns of [`Chain`], with
    /// `pk` and the public inputs `inst`; as [`prove_seeded`].
    pub(crate) fn prove_circuit_seeded<C: Circuit<Fq>>(
        params: &ParamsKZG<Bls12>,
        pk: &ProvingKey<Fq, Scheme>,
        circuit: &C,
        inst: &[Fq],
        seed: u64,
    ) -> Result<Vec<u8>, Error> {
        let mut transcript = CircuitTranscript::<blake2b_simd::State>::init();
        create_proof::<Fq, Scheme, _, _>(
            params,
            pk,
            circuit,
            #[cfg(feature = "committed-instances")]
            0,
            &[inst],
            &mut transcript,
            ChaCha20Rng::seed_from_u64(seed),
        )?;
        Ok(transcript.finalize())
    }

    pub(crate) fn verifies(
        params: &ParamsKZG<Bls12>,
        pk: &ProvingKey<Fq, Scheme>,
        chain: &Chain,
        proof: &[u8],
    ) -> bool {
        let inst = chain.instance();
        let mut transcript = CircuitTranscript::<blake2b_simd::State>::init_from_bytes(proof);
        prepare::<Fq, Scheme, _>(
            pk.get_vk(),
            #[cfg(feature = "committed-instances")]
            &[],
            &[&inst],
            &mut transcript,
        )
        .is_ok_and(|g| g.verify(&params.verifier_params()).is_ok())
    }

    pub(crate) fn chain_setup(k: u32, chain: &Chain) -> (ParamsKZG<Bls12>, ProvingKey<Fq, Scheme>) {
        let params = ParamsKZG::<Bls12>::unsafe_setup(k, ChaCha20Rng::seed_from_u64(k as u64));
        let vk = keygen_vk_with_k::<Fq, Scheme, _>(&params, chain, k).unwrap();
        let pk = keygen_pk(vk, chain).unwrap();
        (params, pk)
    }

    pub(crate) fn read_back(pk: &ProvingKey<Fq, Scheme>) -> ProvingKey<Fq, Scheme> {
        let bytes = pk.to_bytes(SerdeFormat::RawBytes);
        ProvingKey::<Fq, Scheme>::from_bytes::<Chain>(
            &bytes,
            SerdeFormat::RawBytes,
            #[cfg(feature = "circuit-params")]
            (),
        )
        .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The proofs of a key read from bytes whose circuits do not all have
    /// the same regions are the stock proofs, in every build and in any
    /// order: the layout a first proof stores (feature `prover-par`, see
    /// `crate::par::layout`) never changes the proof of a circuit that does
    /// not match it, and a circuit that matches it keeps reusing it.
    mod layout_reuse {
        use super::super::chain::{
            Chain, Reshape, Reshaped, chain_setup, prove_circuit_seeded, prove_seeded, read_back,
            verifies,
        };
        use super::*;

        /// The proofs of this thread since the last call that did not
        /// match the stored layout and synthesized again (0 without the
        /// feature `prover-par`, which stores none).
        fn retries() -> usize {
            #[cfg(feature = "prover-par")]
            return crate::par::layout::take_retries();
            #[cfg(not(feature = "prover-par"))]
            0
        }

        /// The stored layout's starts, if any (none without the feature).
        fn stored(pk: &ProvingKey<Fq, Scheme>) -> Option<Vec<usize>> {
            #[cfg(feature = "prover-par")]
            return pk
                .region_starts_cache
                .get()
                .map(|l| l.regions.iter().map(|r| *r.start).collect());
            #[cfg(not(feature = "prover-par"))]
            {
                let _ = pk;
                None
            }
        }

        struct Fixture {
            params: ParamsKZG<Bls12>,
            pk: ProvingKey<Fq, Scheme>,
            right: Chain,
            shorter: Chain,
        }

        fn fixture() -> Fixture {
            let right = Chain {
                rows: 200,
                start: 7,
            };
            let shorter = Chain {
                rows: 150,
                start: 7,
            };
            let (params, pk) = chain_setup(9, &right);
            Fixture {
                params,
                pk,
                right,
                shorter,
            }
        }

        fn reshaped(f: &Fixture, reshape: Reshape) -> Reshaped {
            Reshaped {
                chain: f.right.clone(),
                reshape,
            }
        }

        /// The proof of `circuit` with a key read from bytes on which no
        /// proof was made: the stock synthesis with the shape pass, in every
        /// build (the reference for a circuit that does not match a stored
        /// layout).
        fn uncached<C: Circuit<Fq>>(f: &Fixture, circuit: &C, inst: &[Fq], seed: u64) -> Vec<u8> {
            let fresh = read_back(&f.pk);
            let proof = prove_circuit_seeded(&f.params, &fresh, circuit, inst, seed).unwrap();
            assert_eq!(retries(), 0);
            proof
        }

        /// A key for 200 rows read from bytes proves 150 rows first, then
        /// 200 rows: the proofs of 200 rows are those of the key from keygen
        /// and verify.
        #[test]
        fn other_rows_first() {
            let f = fixture();
            let _seeded = SEEDED.lock().unwrap_or_else(|e| e.into_inner());
            crate::plonk::prover::test_blinding::set(Some(0x1a));
            retries();
            let reference = prove_seeded(&f.params, &f.pk, &f.right, 1);
            let shorter_ref = uncached(&f, &f.shorter, &f.shorter.instance(), 1);

            let read = read_back(&f.pk);
            let first = prove_seeded(&f.params, &read, &f.shorter, 1);
            assert_eq!(retries(), 0, "the first proof measures the layout");
            let second = prove_seeded(&f.params, &read, &f.right, 1);
            let second_retries = retries();
            let third = prove_seeded(&f.params, &read, &f.right, 1);
            let third_retries = retries();
            crate::plonk::prover::test_blinding::set(None);

            assert_eq!(first, shorter_ref);
            assert_eq!(second, reference);
            assert_eq!(third, reference);
            assert!(verifies(&f.params, &read, &f.right, &second));
            assert!(!verifies(&f.params, &read, &f.shorter, &first));
            if cfg!(feature = "prover-par") {
                // the layout of 150 rows stays: the 200 rows do not match it
                assert_eq!(stored(&read), Some(vec![0, 151]));
                assert_eq!((second_retries, third_retries), (1, 1));
            }
        }

        /// A first proof of a circuit with one region fewer, or one more,
        /// then a proof of the key's circuit: no panic, and the stock proofs.
        #[test]
        fn other_region_count_first() {
            let f = fixture();
            let _seeded = SEEDED.lock().unwrap_or_else(|e| e.into_inner());
            crate::plonk::prover::test_blinding::set(Some(0x1b));
            retries();
            let reference = prove_seeded(&f.params, &f.pk, &f.right, 2);
            for (reshape, regions) in [(Reshape::Fewer, 1), (Reshape::More, 3)] {
                let other = reshaped(&f, reshape);
                let inst = f.right.instance();
                let other_ref = uncached(&f, &other, &inst, 2);

                let read = read_back(&f.pk);
                let first = prove_circuit_seeded(&f.params, &read, &other, &inst, 2).unwrap();
                assert_eq!(retries(), 0);
                assert_eq!(first, other_ref, "{reshape:?}");
                if cfg!(feature = "prover-par") {
                    assert_eq!(stored(&read).map(|s| s.len()), Some(regions));
                }
                for _ in 0..2 {
                    let proof = prove_seeded(&f.params, &read, &f.right, 2);
                    assert_eq!(proof, reference, "{reshape:?}");
                    assert!(verifies(&f.params, &read, &f.right, &proof));
                    if cfg!(feature = "prover-par") {
                        assert_eq!(retries(), 1, "{reshape:?}");
                    }
                }
            }
            crate::plonk::prover::test_blinding::set(None);
        }

        /// The key's circuit first, then circuits that do not match its
        /// layout, then the key's circuit again: each proof is the stock
        /// one, the stored layout does not change, and the key's circuit
        /// keeps reusing it.
        #[test]
        fn matching_first_then_others() {
            let f = fixture();
            let _seeded = SEEDED.lock().unwrap_or_else(|e| e.into_inner());
            crate::plonk::prover::test_blinding::set(Some(0x1c));
            retries();
            let reference = prove_seeded(&f.params, &f.pk, &f.right, 3);
            let shorter_ref = uncached(&f, &f.shorter, &f.shorter.instance(), 3);
            let inst = f.right.instance();
            let fewer = reshaped(&f, Reshape::Fewer);
            let more = reshaped(&f, Reshape::More);
            let fewer_ref = uncached(&f, &fewer, &inst, 3);
            let more_ref = uncached(&f, &more, &inst, 3);

            let read = read_back(&f.pk);
            assert_eq!(prove_seeded(&f.params, &read, &f.right, 3), reference);
            assert_eq!(retries(), 0);
            let layout = stored(&read);
            if cfg!(feature = "prover-par") {
                assert_eq!(
                    layout,
                    f.pk.region_starts.as_ref().map(|s| s.iter().map(|r| **r).collect())
                );
            }
            assert_eq!(prove_seeded(&f.params, &read, &f.shorter, 3), shorter_ref);
            let after_shorter = retries();
            let fewer_proof = prove_circuit_seeded(&f.params, &read, &fewer, &inst, 3).unwrap();
            assert_eq!(fewer_proof, fewer_ref);
            let after_fewer = retries();
            let more_proof = prove_circuit_seeded(&f.params, &read, &more, &inst, 3).unwrap();
            assert_eq!(more_proof, more_ref);
            let after_more = retries();
            let again = prove_seeded(&f.params, &read, &f.right, 3);
            let after_again = retries();
            crate::plonk::prover::test_blinding::set(None);

            assert_eq!(again, reference);
            assert!(verifies(&f.params, &read, &f.right, &again));
            assert_eq!(stored(&read), layout, "the stored layout does not change");
            if cfg!(feature = "prover-par") {
                assert_eq!(
                    (after_shorter, after_fewer, after_more, after_again),
                    (1, 1, 1, 0)
                );
            }
        }

        /// A circuit whose synthesis fails after a layout is stored returns
        /// the error of the stock prover, and the next proof of the key's
        /// circuit still reuses the layout.
        #[test]
        fn failing_synthesis_after_layout() {
            #[derive(Clone)]
            struct Failing;
            impl Circuit<Fq> for Failing {
                type Config = super::super::chain::ChainConfig;
                type FloorPlanner = SimpleFloorPlanner;
                #[cfg(feature = "circuit-params")]
                type Params = ();
                fn without_witnesses(&self) -> Self {
                    self.clone()
                }
                fn configure(meta: &mut ConstraintSystem<Fq>) -> Self::Config {
                    Chain::configure(meta)
                }
                fn synthesize(
                    &self,
                    config: Self::Config,
                    mut layouter: impl Layouter<Fq>,
                ) -> Result<(), Error> {
                    layouter.assign_region(
                        || "unknown",
                        |mut region| {
                            region.assign_advice(
                                || "x",
                                config.advice[0],
                                0,
                                Value::<Fq>::unknown,
                            )?;
                            Ok(())
                        },
                    )
                }
            }
            let f = fixture();
            let _seeded = SEEDED.lock().unwrap_or_else(|e| e.into_inner());
            crate::plonk::prover::test_blinding::set(Some(0x1d));
            retries();
            let reference = prove_seeded(&f.params, &f.pk, &f.right, 4);
            let inst = f.right.instance();
            let stock_err = prove_circuit_seeded(&f.params, &read_back(&f.pk), &Failing, &inst, 4)
                .expect_err("the stock prover fails");

            let read = read_back(&f.pk);
            assert_eq!(prove_seeded(&f.params, &read, &f.right, 4), reference);
            let err = prove_circuit_seeded(&f.params, &read, &Failing, &inst, 4)
                .expect_err("the prover fails as stock");
            assert_eq!(format!("{err:?}"), format!("{stock_err:?}"));
            retries();
            assert_eq!(prove_seeded(&f.params, &read, &f.right, 4), reference);
            assert_eq!(retries(), 0);
            crate::plonk::prover::test_blinding::set(None);
        }
    }

    /// The digest of a seeded proof. The same in every build of this crate:
    /// the feature `gpu` without installed hooks proves byte for byte as a
    /// build without it (CI runs the unit tests in both builds).
    #[test]
    fn seeded_proof_digest() {
        let (params, pk) = setup(3, true, 6);
        let (proof, ok) = prove(&params, &pk, 3, true);
        assert!(ok, "the satisfied circuit verifies");
        let (again, _) = prove(&params, &pk, 3, true);
        assert_eq!(proof, again, "seeded proofs are reproducible");
        let digest = blake2b_simd::Params::new().hash_length(32).hash(&proof);
        assert_eq!(digest.to_hex().as_str(), GOLDEN, "proof bytes changed");
    }

    const GOLDEN: &str = "5177fc0d3371a3ee809efe70428114afb17eddddc866a104b37e6369c5da96f2";

    /// Seeded proofs at `2^14` rows have the same digests in every build:
    /// the digests below were taken from a build without the feature
    /// `prover-par`. At this size the column paths of the feature take over
    /// (parallel encodings and inversions, from `2^14` rows) and the
    /// evaluations run in chunks, but the parallel divisions of the
    /// multi-open do not (`crate::par::open::PAR_MIN_LEN` is `2^15`): see
    /// [`seeded_proof_digests_at_division_size`]. The random circuit has
    /// lookups, a trash argument and copy constraints; the chain circuit is
    /// proved with its key from keygen and with the same key read back from
    /// bytes (first and second proof).
    #[test]
    fn seeded_proof_digests_at_scale() {
        use super::chain::{Chain, chain_setup, prove_seeded, read_back, verifies};

        let digest = |proof: &[u8]| {
            blake2b_simd::Params::new().hash_length(32).hash(proof).to_hex().to_string()
        };
        let k = 14;
        let mut digests = Vec::new();
        for seed in [7u64, 21] {
            let (params, pk) = setup(seed, true, k);
            let (proof, ok) = prove(&params, &pk, seed, true);
            assert!(ok, "seed {seed}: the satisfied circuit verifies");
            digests.push(digest(&proof));
        }

        let chain = Chain {
            rows: (1 << k) - 24,
            start: 7,
        };
        let (params, pk) = chain_setup(k, &chain);
        let read = read_back(&pk);
        let seeded = SEEDED.lock().unwrap_or_else(|e| e.into_inner());
        crate::plonk::prover::test_blinding::set(Some(0xc4a1));
        let from_keygen = prove_seeded(&params, &pk, &chain, 5);
        let read_first = prove_seeded(&params, &read, &chain, 5);
        let read_second = prove_seeded(&params, &read, &chain, 5);
        crate::plonk::prover::test_blinding::set(None);
        drop(seeded);
        assert!(verifies(&params, &pk, &chain, &from_keygen));
        assert_eq!(read_first, from_keygen);
        assert_eq!(read_second, from_keygen);
        digests.push(digest(&from_keygen));

        assert_eq!(digests, GOLDEN_AT_SCALE, "proof bytes changed");
    }

    /// Seeded proofs at `2^16` rows, where the multi-open of the feature
    /// `prover-par` divides in parallel blocks (`divide_by_roots` and
    /// `kate_division` of `crate::par::open`, from `PAR_MIN_LEN = 2^15`
    /// coefficients), have the same digests in every build: the digests
    /// below were taken from a build without the feature. With the feature
    /// the test also checks that both parallel divisions ran. The inner
    /// product of the multi-open takes its parallel path only from `2^19`
    /// coefficients, out of reach here; its unit test crosses that floor
    /// against the stock helper. The chain is proved with its key from
    /// keygen and with the same key read back from bytes (first and second
    /// proof, the second reusing the stored layout).
    #[test]
    fn seeded_proof_digests_at_division_size() {
        use super::chain::{Chain, chain_setup, prove_seeded, read_back, verifies};

        let digest = |proof: &[u8]| {
            blake2b_simd::Params::new().hash_length(32).hash(proof).to_hex().to_string()
        };
        let k = 16;
        #[cfg(feature = "prover-par")]
        assert!(1usize << k >= crate::par::open::PAR_MIN_LEN);
        #[cfg(feature = "prover-par")]
        let before = crate::par::open::reached::counts();
        let mut digests = Vec::new();
        let seed = 7u64;
        let (params, pk) = setup(seed, true, k);
        let (proof, ok) = prove(&params, &pk, seed, true);
        assert!(ok, "seed {seed}: the satisfied circuit verifies");
        digests.push(digest(&proof));
        drop((params, pk));

        let chain = Chain {
            rows: (1 << k) - 24,
            start: 9,
        };
        let (params, pk) = chain_setup(k, &chain);
        let read = read_back(&pk);
        let seeded = SEEDED.lock().unwrap_or_else(|e| e.into_inner());
        crate::plonk::prover::test_blinding::set(Some(0x16d1));
        let from_keygen = prove_seeded(&params, &pk, &chain, 6);
        let read_first = prove_seeded(&params, &read, &chain, 6);
        let read_second = prove_seeded(&params, &read, &chain, 6);
        crate::plonk::prover::test_blinding::set(None);
        drop(seeded);
        assert!(verifies(&params, &pk, &chain, &from_keygen));
        assert_eq!(read_first, from_keygen);
        assert_eq!(read_second, from_keygen);
        digests.push(digest(&from_keygen));

        #[cfg(feature = "prover-par")]
        {
            let after = crate::par::open::reached::counts();
            let reached = [0, 1, 2].map(|i| after[i] - before[i]);
            // `kate_division` divides through `divide_by_roots`: the
            // multi-open's own divisions are the calls beyond those.
            assert!(reached[1] > 0, "kate_division in parallel blocks");
            assert!(
                reached[0] > reached[1],
                "divide_by_roots in parallel blocks"
            );
            eprintln!(
                "parallel paths reached at 2^{k} (divide_by_roots, kate_division, \
                 poly_inner_product): {reached:?}"
            );
        }
        assert_eq!(digests, GOLDEN_AT_DIVISION_SIZE, "proof bytes changed");
    }

    const GOLDEN_AT_DIVISION_SIZE: [&str; 2] = [
        "65ba2eb4e7237076e3a1dda590f39c54b00351fbf983f94bd9b0ae6968ac001d",
        "4e808c9dcbb4c7f06ba75fb357ffa2190b44f365131b2759131633278d710890",
    ];

    const GOLDEN_AT_SCALE: [&str; 3] = [
        "08786ddbd7256bb817805f75492c6ccc957a7d825504f6dd084ab1f0dc3088ed",
        "40b4a1e13ac60011fccfea6f9032b47049c84736bc525938679caa13cd5f06e7",
        "648ff1651d7a3714197f56801c615265c020cac2a57cf669825a952e9319b510",
    ];
}
