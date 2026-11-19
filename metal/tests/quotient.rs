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

//! The device quotient against midnight-proofs: the job layout, the
//! numerator against `QuotientJob::numerator_reference` on every row, the
//! quotient against the stock CPU quotient (the hook layer's check mode and
//! `divide_reference`), whole proofs with all four hooks installed against
//! proofs without hooks, the declines, and the resident proving-key copies.
//!
//! The jobs come from real proving keys: a quotient hook installed by this
//! test sees the job of each proof (`midnight_proofs::gpu::abi::
//! QuotientJobAbi`), rebuilds the `QuotientJob` from it for the references,
//! and passes it on to `midnight_metal::quotient`.
//!
//! `bench_quotient` (ignored) times the device quotient against the stock CPU
//! quotient on the rc1 chain circuit:
//! `cargo test --release -p midnight-metal --test quotient bench_quotient --
//! --ignored --nocapture`.

#![cfg(target_os = "macos")]

use std::{
    ffi::c_void,
    mem::{align_of, offset_of, size_of},
    sync::{
        Mutex, MutexGuard, Once,
        atomic::{AtomicU64, Ordering::Relaxed},
    },
    time::Instant,
};

use ff::Field;
use midnight_curves::Fq;
use midnight_metal::{self as metal, Scalar, quotient_abi as ma, quotient_test_support as qt};
use midnight_proofs::gpu::{
    self as hooks, DeviceProgram, GpuHooks, LookupJob, PermutationJob, QuotientDivide, QuotientJob,
    TrashJob,
    abi::{self as pa, ProgramAbi},
    adapter::{from_scalar, scalar},
    test_switches::{self, QUOTIENT_CHECK, RandomCircuit},
};

// ---------------------------------------------------------------------------
// The layout

macro_rules! same {
    ($a:ty, $b:ty, $($f:ident),*) => {
        assert_eq!(size_of::<$a>(), size_of::<$b>(), stringify!($a));
        assert_eq!(align_of::<$a>(), align_of::<$b>(), stringify!($a));
        $(assert_eq!(offset_of!($a, $f), offset_of!($b, $f), concat!(stringify!($a), ".", stringify!($f)));)*
    };
}

/// The crate's copy of the job types equals `midnight_proofs::gpu::abi`
/// field by field (offsets, sizes, alignment), and the constants agree.
#[test]
fn abi_layout_matches() {
    same!(ma::DeviceOp, hooks::DeviceOp, kind, dst, a, b, c);
    same!(ma::DeviceRead, hooks::DeviceRead, kind, column, rotation);
    same!(
        ma::ProgramAbi,
        pa::ProgramAbi,
        ops,
        nops,
        reads,
        nreads,
        uniforms,
        nuniforms,
        outputs,
        noutputs,
        nslots,
        pad
    );
    same!(
        ma::LookupAbi,
        pa::LookupAbi,
        aggregator,
        multiplicities,
        helpers,
        programs,
        nchunks
    );
    same!(
        ma::TrashAbi,
        pa::TrashAbi,
        trash,
        program,
        selector_column,
        pad
    );
    same!(
        ma::QuotientJobAbi,
        pa::QuotientJobAbi,
        abi,
        k,
        ext_k,
        nslots,
        zeta,
        extended_omega,
        y,
        beta,
        gamma,
        delta_start,
        delta,
        advice,
        nadvice,
        instance,
        ninstance,
        fixed,
        nfixed,
        l0,
        l_last,
        l_active,
        gates,
        perm_ncolumns,
        perm_column_kind,
        perm_column_index,
        perm_sigma,
        perm_products,
        perm_nsets,
        perm_chunk_len,
        perm_last_rotation,
        lookups,
        nlookups,
        trash,
        ntrash,
        t_inv,
        nt_inv,
        omega_inv,
        divisor,
        zeta_out,
        len
    );
    assert_eq!(ma::QUOTIENT_ABI, pa::QUOTIENT_ABI);
    assert_eq!(
        [ma::TAG_UNIFORM, ma::TAG_READ, ma::TAG_SLOT, ma::INDEX_MASK],
        [
            hooks::TAG_UNIFORM,
            hooks::TAG_READ,
            hooks::TAG_SLOT,
            hooks::INDEX_MASK
        ]
    );
    assert_eq!(
        [
            ma::OP_ADD,
            ma::OP_SUB,
            ma::OP_MUL,
            ma::OP_SQUARE,
            ma::OP_DOUBLE,
            ma::OP_NEGATE,
            ma::OP_MUL_ADD
        ],
        [
            hooks::OP_ADD,
            hooks::OP_SUB,
            hooks::OP_MUL,
            hooks::OP_SQUARE,
            hooks::OP_DOUBLE,
            hooks::OP_NEGATE,
            hooks::OP_MUL_ADD
        ]
    );
    assert_eq!(
        [ma::COLUMN_FIXED, ma::COLUMN_ADVICE, ma::COLUMN_INSTANCE],
        [
            hooks::COLUMN_FIXED,
            hooks::COLUMN_ADVICE,
            hooks::COLUMN_INSTANCE
        ]
    );
    // the hook signature takes the entry point as it is
    let _ = GpuHooks {
        quotient: Some(metal::quotient),
        ..Default::default()
    };
}

// ---------------------------------------------------------------------------
// The harness

/// The installed hooks, the min-k, the check switch and the probe are
/// process-wide: one test at a time.
static SERIAL: Mutex<()> = Mutex::new(());

type Probe = Box<dyn FnMut(*const c_void) + Send>;
static PROBE: Mutex<Option<Probe>> = Mutex::new(None);
/// Calls of the quotient hook, and of the MSM hook.
static QUOTIENT_CALLS: AtomicU64 = AtomicU64::new(0);
/// The time of the first MSM after the last quotient hook returned
/// (nanoseconds since `EPOCH`), and when that hook returned.
static MSM_AFTER_Q: AtomicU64 = AtomicU64::new(0);
static Q_RETURNED: AtomicU64 = AtomicU64::new(0);
static EPOCH: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();

fn now_ns() -> u64 {
    // never 0: 0 marks "not yet"
    EPOCH.get_or_init(Instant::now).elapsed().as_nanos() as u64 + 1
}

fn msm(s: &[Scalar], b: &[metal::G1AffineLimbs]) -> Option<metal::G1JacobianLimbs> {
    if MSM_AFTER_Q.load(Relaxed) == 0 && Q_RETURNED.load(Relaxed) != 0 {
        MSM_AFTER_Q.store(now_ns(), Relaxed);
    }
    metal::msm_g1(s, b)
}

unsafe fn quotient_hook(job: *const c_void, out: &mut [Scalar]) -> bool {
    QUOTIENT_CALLS.fetch_add(1, Relaxed);
    if let Some(p) = PROBE.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        p(job);
    }
    // SAFETY: the prover's job, forwarded within its call.
    let ok = unsafe { metal::quotient(job, out) };
    MSM_AFTER_Q.store(0, Relaxed);
    Q_RETURNED.store(now_ns(), Relaxed);
    ok
}

fn lock() -> MutexGuard<'static, ()> {
    static INSTALL: Once = Once::new();
    let g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    INSTALL.call_once(|| {
        assert!(metal::device(), "no Metal device");
        assert!(hooks::install(GpuHooks {
            msm_g1: Some(msm),
            fft: Some(metal::fft),
            ifft: Some(metal::ifft),
            quotient: Some(quotient_hook),
        }));
    });
    set_probe(None);
    QUOTIENT_CHECK.store(false, Relaxed);
    metal::set_min_k(1);
    qt::set_memory_limit(None);
    g
}

fn set_probe(p: Option<Probe>) {
    *PROBE.lock().unwrap_or_else(|e| e.into_inner()) = p;
}

/// The job behind a hook's pointer.
///
/// # Safety
/// `job` is the prover's job, within the hook call.
unsafe fn abi<'a>(job: *const c_void) -> &'a pa::QuotientJobAbi {
    // SAFETY: the caller's contract.
    unsafe { &*(job as *const pa::QuotientJobAbi) }
}

unsafe fn sl<'a, T>(p: *const T, n: usize) -> &'a [T] {
    if n == 0 {
        &[]
    } else {
        // SAFETY: the job's pointers are valid for their counts.
        unsafe { std::slice::from_raw_parts(p, n) }
    }
}

unsafe fn poly<'a>(p: *const Scalar, n: usize) -> &'a [Fq] {
    // SAFETY: `Fq` is the transparent `blst_fr` of `Scalar`.
    unsafe { sl(p as *const Fq, n) }
}

unsafe fn program(p: &ProgramAbi) -> DeviceProgram<Fq> {
    // SAFETY: the job's contract.
    unsafe {
        DeviceProgram {
            uniforms: sl(p.uniforms, p.nuniforms).iter().map(|u| from_scalar(*u)).collect(),
            reads: sl(p.reads, p.nreads).to_vec(),
            ops: sl(p.ops, p.nops).to_vec(),
            nslots: p.nslots,
            outputs: sl(p.outputs, p.noutputs).to_vec(),
        }
    }
}

/// The `QuotientJob` of a packed job, for its references.
///
/// # Safety
/// `j` is a valid packed job whose pointers live while the result is used.
unsafe fn job_of<'a>(j: &'a pa::QuotientJobAbi) -> QuotientJob<'a, Fq> {
    let (n, en) = (1usize << j.k, 1usize << j.ext_k);
    // SAFETY (all): the job's contract.
    unsafe {
        let polys = |ps: *const *const Scalar, c: usize, len: usize| -> Vec<&'a [Fq]> {
            sl(ps, c).iter().map(|p| poly(*p, len)).collect()
        };
        let permutation = (j.perm_nsets > 0).then(|| PermutationJob {
            chunk_len: j.perm_chunk_len,
            columns: sl(j.perm_column_kind, j.perm_ncolumns)
                .iter()
                .copied()
                .zip(sl(j.perm_column_index, j.perm_ncolumns).iter().copied())
                .collect(),
            sigma: polys(j.perm_sigma, j.perm_ncolumns, en),
            products: polys(j.perm_products, j.perm_nsets, n),
            last_rotation: j.perm_last_rotation,
        });
        let lookups = sl(j.lookups, j.nlookups)
            .iter()
            .map(|l| LookupJob {
                aggregator: poly(l.aggregator, n),
                multiplicities: poly(l.multiplicities, n),
                helpers: polys(l.helpers, l.nchunks, n),
                programs: sl(l.programs, l.nchunks).iter().map(|p| program(p)).collect(),
            })
            .collect();
        let trash = sl(j.trash, j.ntrash)
            .iter()
            .map(|t| TrashJob {
                trash: poly(t.trash, n),
                selector_column: t.selector_column,
                program: program(&t.program),
            })
            .collect();
        QuotientJob {
            k: j.k,
            ext_k: j.ext_k,
            zeta: j.zeta.map(from_scalar),
            extended_omega: from_scalar(j.extended_omega),
            advice: polys(j.advice, j.nadvice, n),
            instance: polys(j.instance, j.ninstance, n),
            fixed: sl(j.fixed, j.nfixed)
                .iter()
                .map(|p| (!p.is_null()).then(|| poly(*p, en)))
                .collect(),
            l0: poly(j.l0, en),
            l_last: poly(j.l_last, en),
            l_active: poly(j.l_active, en),
            y: from_scalar(j.y),
            beta: from_scalar(j.beta),
            gamma: from_scalar(j.gamma),
            delta_start: from_scalar(j.delta_start),
            delta: from_scalar(j.delta),
            gates: program(&j.gates),
            permutation,
            lookups,
            trash,
            divide: QuotientDivide {
                t_inv: poly(j.t_inv, j.nt_inv).to_vec(),
                omega_inv: from_scalar(j.omega_inv),
                divisor: from_scalar(j.divisor),
                zeta_out: j.zeta_out.map(from_scalar),
                len: j.len,
            },
        }
    }
}

/// The shape of a job: (k, ext_k, permutation sets, chunks of each lookup,
/// trash arguments, gate operations, nslots).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Shape {
    k: u32,
    ext_k: u32,
    nsets: usize,
    chunks: Vec<usize>,
    ntrash: usize,
    gate_ops: usize,
    nslots: u32,
}

fn shape_of(j: &pa::QuotientJobAbi) -> Shape {
    // SAFETY: the job's contract.
    let lookups = unsafe { sl(j.lookups, j.nlookups) };
    Shape {
        k: j.k,
        ext_k: j.ext_k,
        nsets: j.perm_nsets,
        chunks: lookups.iter().map(|l| l.nchunks).collect(),
        ntrash: j.ntrash,
        gate_ops: j.gates.nops,
        nslots: j.nslots,
    }
}

fn reasons() -> Vec<(&'static str, u64)> {
    metal::declines()
        .into_iter()
        .filter(|d| d.0 == "quotient")
        .map(|d| (d.1, d.2))
        .collect()
}

fn count(reason: &str) -> u64 {
    reasons().iter().filter(|r| r.0 == reason).map(|r| r.1).sum()
}

/// The quotient of a job on the CPU by the job's own definition.
fn reference_quotient(j: &pa::QuotientJobAbi) -> Vec<Fq> {
    // SAFETY: within the hook call.
    let job = unsafe { job_of(j) };
    job.divide_reference(job.numerator_reference()).as_slice().to_vec()
}

// ---------------------------------------------------------------------------
// Shapes of the random circuits

/// Prints the job shapes of RandomCircuit seeds (to choose the seeds of the
/// differential tests).
#[test]
#[ignore]
fn scan_shapes() {
    let _g = lock();
    for seed in 0..48u64 {
        let rc = RandomCircuit::new(seed, false, 7);
        let s = rc.shape().expect("a job");
        println!(
            "seed {seed}: ext_k {} proof_polys {} key_polys {} chunks {:?} trash {} nslots {} ops {:?}",
            s.ext_k, s.proof_polys, s.key_polys, s.lookup_chunks, s.trash, s.nslots, s.op_kinds
        );
    }
}

// ---------------------------------------------------------------------------
// A circuit with multi-chunk lookups

mod range {
    //! `Pow2RangeChip` of midnight-circuits with nine parallel lookup columns
    //! (its lookup splits into several chunks at a low degree), a custom gate
    //! and copy constraints into the range-checked columns and the instance.

    use ff::Field;
    use midnight_circuits::{
        field::decomposition::pow2range::{Pow2RangeChip, Pow2RangeConfig},
        instructions::decomposition::Pow2RangeInstructions,
        types::AssignedNative,
    };
    use midnight_curves::{Bls12, Fq};
    use midnight_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        plonk::{
            Advice, Circuit, Column, ConstraintSystem, Constraints, Error, Instance, ProvingKey,
            Selector, create_proof, keygen_pk, keygen_vk_with_k, prepare,
        },
        poly::{
            Rotation,
            commitment::Guard,
            kzg::{KZGCommitmentScheme, params::ParamsKZG},
        },
        transcript::{Blake2b256, CircuitTranscript, Transcript},
    };
    use rand_chacha::ChaCha20Rng;
    use rand_core::{RngCore, SeedableRng};

    pub type Scheme = KZGCommitmentScheme<Bls12>;
    const BITS: usize = 8;

    #[derive(Clone, Debug)]
    pub struct Config {
        range: Pow2RangeConfig,
        src: [Column<Advice>; 2],
        inst: Column<Instance>,
        q: Selector,
    }

    #[derive(Clone, Debug, Default)]
    pub struct RangeCircuit {
        pub seed: u64,
        pub rows: usize,
    }

    impl RangeCircuit {
        fn values(&self) -> Vec<u64> {
            let mut rng = ChaCha20Rng::seed_from_u64(self.seed);
            (0..self.rows).map(|_| rng.next_u64() % (1 << BITS)).collect()
        }

        pub fn instance(&self) -> [Fq; 1] {
            [Fq::from(self.values()[0])]
        }
    }

    impl Circuit<Fq> for RangeCircuit {
        type Config = Config;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = ();

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(meta: &mut ConstraintSystem<Fq>) -> Config {
            let cols: Vec<Column<Advice>> = (0..9).map(|_| meta.advice_column()).collect();
            for c in &cols {
                meta.enable_equality(*c);
            }
            let src = [meta.advice_column(), meta.advice_column()];
            meta.enable_equality(src[0]);
            let inst = meta.instance_column();
            meta.enable_equality(inst);
            let q = meta.selector();
            // src[1] = src[0]^2 + src[0] (degree 3 with the selector)
            meta.create_gate("square plus", |vc| {
                let a = vc.query_advice(src[0], Rotation::cur());
                let b = vc.query_advice(src[1], Rotation::cur());
                Constraints::with_selector(q, vec![a.clone() * a.clone() + a - b])
            });
            let range = Pow2RangeChip::configure(meta, &cols);
            Config {
                range,
                src,
                inst,
                q,
            }
        }

        fn synthesize(&self, cfg: Config, mut layouter: impl Layouter<Fq>) -> Result<(), Error> {
            let chip = Pow2RangeChip::<Fq>::new(&cfg.range, BITS);
            let values = self.values();
            let cells = layouter.assign_region(
                || "values",
                |mut region| {
                    let mut cells = Vec::with_capacity(values.len());
                    for (row, v) in values.iter().enumerate() {
                        let x = Fq::from(*v);
                        cells.push(region.assign_advice(
                            || "v",
                            cfg.src[0],
                            row,
                            || Value::known(x),
                        )?);
                        region.assign_advice(
                            || "w",
                            cfg.src[1],
                            row,
                            || Value::known(x.square() + x),
                        )?;
                        cfg.q.enable(&mut region, row)?;
                    }
                    Ok(cells)
                },
            )?;
            layouter.constrain_instance(cells[0].cell(), cfg.inst, 0)?;
            let cells: Vec<AssignedNative<Fq>> =
                cells.into_iter().map(AssignedNative::from_cell).collect();
            chip.assert_values_lower_than_2_pow_n(&mut layouter, &cells, BITS)?;
            chip.load_table(&mut layouter)
        }
    }

    pub fn setup(seed: u64, k: u32) -> (ParamsKZG<Bls12>, ProvingKey<Fq, Scheme>, RangeCircuit) {
        let params = ParamsKZG::<Bls12>::unsafe_setup(k, ChaCha20Rng::seed_from_u64(seed));
        let circuit = RangeCircuit {
            seed,
            rows: (1 << (k - 1)) - 40,
        };
        let vk = keygen_vk_with_k::<Fq, Scheme, _>(&params, &circuit, k).unwrap();
        let pk = keygen_pk(vk, &circuit).unwrap();
        (params, pk, circuit)
    }

    /// A proof with every random draw seeded, and whether it verifies.
    pub fn prove(
        params: &ParamsKZG<Bls12>,
        pk: &ProvingKey<Fq, Scheme>,
        circuit: &RangeCircuit,
    ) -> (Vec<u8>, bool) {
        midnight_proofs::gpu::test_switches::set_quotient_blinding_seed(Some(
            circuit.seed ^ 0xb11d,
        ));
        let inst = circuit.instance();
        let mut transcript = CircuitTranscript::<Blake2b256>::init();
        create_proof::<Fq, Scheme, _, _>(
            params,
            pk,
            circuit,
            0,
            &[&inst],
            &mut transcript,
            ChaCha20Rng::seed_from_u64(circuit.seed ^ 0x9e37),
        )
        .unwrap();
        midnight_proofs::gpu::test_switches::set_quotient_blinding_seed(None);
        let proof = transcript.finalize();
        let mut transcript = CircuitTranscript::<Blake2b256>::init_from_bytes(&proof);
        let ok = prepare::<Fq, Scheme, _>(pk.get_vk(), &[], &[&inst], &mut transcript)
            .is_ok_and(|g| g.verify(&params.verifier_params()).is_ok());
        (proof, ok)
    }
}

// ---------------------------------------------------------------------------
// The numerator and the quotient against the references

static SHAPES: Mutex<Vec<Shape>> = Mutex::new(Vec::new());

/// The probe of the differential tests: the device numerator against
/// `numerator_reference` on every row, and the device quotient against
/// `divide_reference` of it.
fn differential_probe() -> Probe {
    Box::new(|job| {
        // SAFETY: within the hook call.
        let j = unsafe { abi(job) };
        let shape = shape_of(j);
        // SAFETY: as above.
        let r = unsafe { job_of(j) };
        let nu_ref = r.numerator_reference();
        // SAFETY: as above.
        let nu = unsafe { qt::numerator(job) }.expect("the device numerator");
        assert_eq!(nu.len(), nu_ref.len(), "{shape:?}");
        let bad = nu.iter().zip(nu_ref.iter()).position(|(d, c)| *d != scalar(*c));
        assert!(
            bad.is_none(),
            "{shape:?}: numerator differs from row {bad:?}"
        );
        let h_ref = r.divide_reference(nu_ref);
        let mut out = vec![[7u64; 4]; j.len];
        // SAFETY: as above.
        assert!(
            unsafe { metal::quotient(job, &mut out) },
            "{shape:?}: declined {:?}",
            reasons()
        );
        assert!(
            out.iter().zip(h_ref.iter()).all(|(d, c)| *d == scalar(*c)),
            "{shape:?}: quotient differs"
        );
        SHAPES.lock().unwrap_or_else(|e| e.into_inner()).push(shape);
    })
}

/// For jobs of real proving keys (random circuits with gates, one or two
/// permutation sets, a LogUp lookup and a trash argument; the range circuit
/// with lookups of several chunks), the device numerator equals
/// `numerator_reference` on every row of the extended domain and the device
/// quotient equals `divide_reference`; the prover's check mode compares the
/// hook's quotient with the stock CPU quotient.
#[test]
fn numerator_and_quotient_match_references() {
    let _g = lock();
    SHAPES.lock().unwrap_or_else(|e| e.into_inner()).clear();
    set_probe(Some(differential_probe()));
    QUOTIENT_CHECK.store(true, Relaxed);
    let (checks, diffs) = (hooks::quotient_checks(), hooks::quotient_check_diffs());
    let served = metal::stats().quotient.served;
    let mut proofs = 0u64;
    // seeds with one (3, 4, 6) and two (0, 8, 10) permutation sets
    for (seed, k) in [(0u64, 6u32), (3, 6), (8, 7), (4, 8), (10, 9), (6, 10)] {
        let rc = RandomCircuit::new(seed, false, k);
        rc.prove();
        proofs += 1;
    }
    for (seed, k) in [(1u64, 9u32), (2, 11)] {
        let (params, pk, c) = range::setup(seed, k);
        let (_, ok) = range::prove(&params, &pk, &c);
        assert!(ok, "the range circuit verifies");
        proofs += 1;
    }
    set_probe(None);
    QUOTIENT_CHECK.store(false, Relaxed);
    assert_eq!(
        hooks::quotient_checks() - checks,
        proofs,
        "a check per proof"
    );
    assert_eq!(
        hooks::quotient_check_diffs(),
        diffs,
        "the device quotient is the stock one"
    );
    // the probe's quotient and the hook's own one
    assert_eq!(metal::stats().quotient.served - served, 2 * proofs);
    let shapes = SHAPES.lock().unwrap_or_else(|e| e.into_inner()).clone();
    assert_eq!(shapes.len() as u64, proofs);
    for s in &shapes {
        println!("{s:?}");
    }
    assert!(
        shapes.iter().any(|s| s.nsets >= 2),
        "a permutation of several sets"
    );
    assert!(
        shapes.iter().any(|s| s.nsets == 1),
        "a permutation of one set"
    );
    assert!(
        shapes.iter().any(|s| s.chunks.iter().any(|c| *c >= 2)),
        "a lookup of several chunks"
    );
    assert!(shapes.iter().any(|s| s.ntrash >= 1), "a trash argument");
    assert!(shapes.iter().all(|s| s.gate_ops > 0), "custom gates");
    assert!(
        shapes.iter().any(|s| s.ext_k > 10),
        "a two-pass transform size"
    );
}

/// Several dispatch widths (scratch strides) and row ranges, and a domain
/// split over many command buffers, give the same numerator.
#[test]
fn numerator_independent_of_threads() {
    let _g = lock();
    SHAPES.lock().unwrap_or_else(|e| e.into_inner()).clear();
    set_probe(Some(differential_probe()));
    for t in [256u32, 1024, 1 << 20] {
        qt::set_threads(Some(t));
        RandomCircuit::new(10, false, 9).prove();
    }
    // many command buffers of a few rows each
    qt::set_threads(Some(256));
    qt::set_target_nanos(Some(1));
    RandomCircuit::new(10, false, 9).prove();
    let runs = qt::last_stages().unwrap().numerator_runs;
    qt::set_target_nanos(None);
    qt::set_threads(None);
    set_probe(None);
    assert_eq!(SHAPES.lock().unwrap_or_else(|e| e.into_inner()).len(), 4);
    assert_eq!(
        runs,
        (1 << 11) / 256,
        "one command buffer per dispatch of rows"
    );
}

// ---------------------------------------------------------------------------
// Whole proofs

/// Proves `prove` with all four hooks and without hooks: the same bytes, a
/// proof the stock verifier accepts, the quotient served by the device; then
/// once more in check mode with no difference.
fn end_to_end(name: &str, prove: impl Fn() -> (Vec<u8>, bool)) {
    let served = metal::stats().quotient.served;
    let calls = QUOTIENT_CALLS.load(Relaxed);
    let (with, ok) = prove();
    assert!(ok, "{name}: the device proof verifies");
    assert_eq!(
        QUOTIENT_CALLS.load(Relaxed) - calls,
        1,
        "{name}: the hook was offered"
    );
    assert_eq!(
        metal::stats().quotient.served - served,
        1,
        "{name}: served, {:?}",
        reasons()
    );
    let (without, ok2) = test_switches::without_hooks(&prove);
    assert!(ok2);
    assert!(
        with == without,
        "{name}: the proof bytes differ from the stock proof"
    );
    let (checks, diffs) = (hooks::quotient_checks(), hooks::quotient_check_diffs());
    QUOTIENT_CHECK.store(true, Relaxed);
    let (checked, _) = prove();
    QUOTIENT_CHECK.store(false, Relaxed);
    assert_eq!(hooks::quotient_checks() - checks, 1);
    assert_eq!(
        hooks::quotient_check_diffs(),
        diffs,
        "{name}: quotient_check_diffs"
    );
    assert!(checked == with);
    assert!(metal::end_request());
}

#[test]
fn end_to_end_small() {
    let _g = lock();
    let rc = RandomCircuit::new(8, true, 8);
    end_to_end("random k=8", || rc.prove());
    let (params, pk, c) = range::setup(5, 10);
    end_to_end("range k=10", || range::prove(&params, &pk, &c));
}

/// At `2^16` rows (release builds; minutes in a debug build).
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn end_to_end_k16() {
    let _g = lock();
    metal::set_min_k(14);
    let rc = RandomCircuit::new(10, true, 16);
    end_to_end("random k=16", || rc.prove());
    let (params, pk, c) = range::setup(7, 16);
    end_to_end("range k=16", || range::prove(&params, &pk, &c));
}

// ---------------------------------------------------------------------------
// Declines

/// A job with a wrong ABI version, below min-k, or above the memory limit
/// declines with its reason and leaves `out` unchanged.
#[test]
fn declines_leave_out_unchanged() {
    let _g = lock();
    static DONE: AtomicU64 = AtomicU64::new(0);
    DONE.store(0, Relaxed);
    set_probe(Some(Box::new(|job| {
        // SAFETY: within the hook call.
        let j = unsafe { abi(job) };
        let sentinel = [0x5eedu64; 4];
        let try_job = |j: &pa::QuotientJobAbi, reason: &str| {
            let before = count(reason);
            let mut out = vec![sentinel; j.len];
            // SAFETY: the prover's tables, within its call.
            let ok = unsafe { metal::quotient(j as *const _ as *const c_void, &mut out) };
            assert!(!ok, "{reason}: declined");
            assert!(
                out.iter().all(|x| *x == sentinel),
                "{reason}: out unchanged"
            );
            assert_eq!(count(reason) - before, 1, "{reason}: counted");
        };
        let mut bad = *j;
        bad.abi = pa::QUOTIENT_ABI + 1;
        try_job(&bad, "shape");
        metal::set_min_k(j.ext_k + 1);
        try_job(j, "below-min-k");
        metal::set_min_k(1);
        qt::set_memory_limit(Some(metal::need_bytes(j.ext_k, 1, 1, 0) as u64));
        try_job(j, "memory");
        qt::set_memory_limit(None);
        // a wrong output length
        let mut out = vec![sentinel; j.len + 1];
        let before = count("shape");
        // SAFETY: as above.
        assert!(!unsafe { metal::quotient(job, &mut out) });
        assert_eq!(count("shape") - before, 1);
        // an unreduced challenge
        let mut bad = *j;
        bad.y = [u64::MAX; 4];
        try_job(&bad, "unreduced");
        DONE.fetch_add(1, Relaxed);
    })));
    RandomCircuit::new(3, false, 7).prove();
    set_probe(None);
    assert_eq!(DONE.load(Relaxed), 1);
}

// ---------------------------------------------------------------------------
// Resident proving-key copies

/// Registered key polynomials: the first quotient makes the copy, the second
/// reads it without copying, a changed polynomial is detected, counted stale,
/// uploaded for the call and still gives the right quotient; the copies go
/// at the end of a request unless kept.
#[test]
fn registered_key_is_resident() {
    let _g = lock();
    static DONE: AtomicU64 = AtomicU64::new(0);
    DONE.store(0, Relaxed);
    set_probe(Some(Box::new(|job| {
        // SAFETY: within the hook call.
        let j = unsafe { abi(job) };
        let en = 1usize << j.ext_k;
        // our own copy of l_active, so that it can change
        // SAFETY: the job's contract.
        let mut own: Vec<Scalar> = unsafe { sl(j.l_active, en) }.to_vec();
        let mut job2 = *j;
        job2.l_active = own.as_ptr();
        // the key polynomials as gpu_resident_polys lists them
        // SAFETY: the job's contract.
        let mut polys: Vec<&[Scalar]> = unsafe {
            sl(j.fixed, j.nfixed)
                .iter()
                .filter(|p| !p.is_null())
                .chain(sl(j.perm_sigma, j.perm_ncolumns))
                .map(|p| sl(*p, en))
                .collect()
        };
        // SAFETY: as above.
        unsafe { polys.extend([sl(j.l0, en), sl(j.l_last, en)]) };
        // SAFETY: `own` outlives the registration's use below.
        polys.push(unsafe { std::slice::from_raw_parts(own.as_ptr(), en) });
        let nkeys = polys.len() as u32;
        metal::set_keep_resident(false);
        let handle = metal::register_key(&polys);
        assert!(handle.verify(&polys));
        assert_eq!(handle.resident_bytes(), 0);
        let run = |job2: &pa::QuotientJobAbi| {
            let mut out = vec![[0u64; 4]; job2.len];
            // SAFETY: the prover's tables and `own`, within the call.
            assert!(unsafe { metal::quotient(job2 as *const _ as *const c_void, &mut out) });
            let refq = reference_quotient(job2);
            assert!(
                out.iter().zip(&refq).all(|(d, c)| *d == scalar(*c)),
                "quotient"
            );
            qt::last_stages().unwrap()
        };
        let s1 = run(&job2);
        assert_eq!(
            (s1.resident_polys, s1.uploaded_polys, s1.key_copies_made),
            (nkeys, 0, nkeys)
        );
        assert_eq!(handle.resident_bytes(), (nkeys as usize * en * 32) as u64);
        metal::begin_request();
        let s2 = run(&job2);
        assert_eq!(
            (s2.resident_polys, s2.uploaded_polys, s2.key_copies_made),
            (nkeys, 0, 0)
        );
        // a changed polynomial
        let stale = metal::key_stale();
        own[3] = scalar(from_scalar(own[3]) + Fq::ONE);
        job2.l_active = own.as_ptr();
        metal::begin_request();
        let s3 = run(&job2);
        assert_eq!(metal::key_stale() - stale, 1);
        assert_eq!((s3.resident_polys, s3.uploaded_polys), (nkeys - 1, 1));
        assert_eq!(metal::keys_registered(), nkeys as usize - 1);
        // freed at the end of a request, kept with keep_resident
        assert!(metal::end_request());
        assert_eq!(handle.resident_bytes(), 0);
        metal::set_keep_resident(true);
        let s4 = run(&job2);
        assert_eq!(s4.key_copies_made, nkeys - 1);
        assert!(metal::end_request());
        assert!(handle.resident_bytes() > 0);
        assert!(metal::evict_key_copies() > 0);
        assert_eq!(handle.resident_bytes(), 0);
        metal::set_keep_resident(false);
        drop(handle);
        assert_eq!(metal::keys_registered(), 0);
        drop(own);
        DONE.fetch_add(1, Relaxed);
    })));
    RandomCircuit::new(8, false, 8).prove();
    set_probe(None);
    assert_eq!(DONE.load(Relaxed), 1);
}

// ---------------------------------------------------------------------------
// Measurement

/// The rc1 benchmark circuit of the pull request (a chain of squarings).
#[allow(dead_code)]
#[path = "../../gpu-prover/bench/rc1/circuit.rs"]
mod chain;

/// Times the device quotient against the stock CPU quotient on the rc1 chain
/// circuit at `QBENCH_KS` (default 14,16,18,20), the threads of
/// `QBENCH_THREADS` (default the crate's). The device quotient runs three
/// times on the job of each proof (inside the hook); the stock quotient is
/// the hook layer's check-mode computation, timed from the hook's return to
/// the first MSM after it (the commitment of the quotient limbs; the interval
/// also holds the comparison and the split into limbs).
#[test]
#[ignore]
fn bench_quotient() {
    use midnight_curves::Bls12;
    use midnight_proofs::{
        plonk::{create_proof, keygen_pk, keygen_vk_with_k, prepare},
        poly::{
            commitment::Guard,
            kzg::{KZGCommitmentScheme, params::ParamsKZG},
        },
        transcript::{Blake2b256, CircuitTranscript, Transcript},
    };
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;
    type Scheme = KZGCommitmentScheme<Bls12>;

    static DEV: Mutex<Vec<(u64, metal::QuotientStages)>> = Mutex::new(Vec::new());
    let _g = lock();
    let ks: Vec<u32> = std::env::var("QBENCH_KS")
        .unwrap_or("14,16,18,20".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let threads: Vec<Option<u32>> = std::env::var("QBENCH_THREADS")
        .map(|v| v.split(',').map(|s| Some(s.trim().parse().unwrap())).collect())
        .unwrap_or(vec![None]);
    let proofs: usize = std::env::var("QBENCH_PROOFS").map_or(2, |v| v.parse().unwrap());
    println!(
        "| k | ext_k | threads | device quotient ms (min of {}) | stock CPU quotient ms (min of {proofs}) | speed-up | peak device MiB | need_bytes MiB |",
        3 * proofs
    );
    for k in ks {
        // the quotient of every size measured reaches the device
        metal::set_min_k(k.min(14));
        let params = ParamsKZG::<Bls12>::unsafe_setup(k, ChaCha20Rng::seed_from_u64(k as u64));
        let circuit = chain::Chain { k };
        let vk = keygen_vk_with_k::<Fq, Scheme, _>(&params, &circuit, k).unwrap();
        let pk = keygen_pk(vk, &circuit).unwrap();
        let shape = hooks::quotient_shape(&pk).expect("a quotient job");
        let inst = circuit.instance();
        // the key registered as the prover binding registers it; its copies
        // stay between requests with QBENCH_KEEP=1
        let polys = pk.gpu_resident_polys().expect("resident polynomials");
        let key = metal::register_key(
            &polys.iter().map(|p| hooks::adapter::scalars(p)).collect::<Vec<_>>(),
        );
        metal::set_keep_resident(std::env::var("QBENCH_KEEP").as_deref() == Ok("1"));
        for t in &threads {
            qt::set_threads(*t);
            DEV.lock().unwrap().clear();
            set_probe(Some(Box::new(|job| {
                // SAFETY: within the hook call.
                let j = unsafe { abi(job) };
                for _ in 0..3 {
                    let mut out = vec![[0u64; 4]; j.len];
                    let t = Instant::now();
                    // SAFETY: as above.
                    assert!(unsafe { metal::quotient(job, &mut out) });
                    let ns = t.elapsed().as_nanos() as u64;
                    DEV.lock().unwrap().push((ns, qt::last_stages().unwrap()));
                }
            })));
            QUOTIENT_CHECK.store(true, Relaxed);
            let diffs = hooks::quotient_check_diffs();
            let mut stock = Vec::new();
            for _ in 0..proofs {
                metal::begin_request();
                let mut transcript = CircuitTranscript::<Blake2b256>::init();
                create_proof::<Fq, Scheme, _, _>(
                    &params,
                    &pk,
                    &circuit,
                    0,
                    &[&inst],
                    &mut transcript,
                    ChaCha20Rng::seed_from_u64(1),
                )
                .unwrap();
                let s = MSM_AFTER_Q.load(Relaxed).saturating_sub(Q_RETURNED.load(Relaxed));
                stock.push(s);
                let proof = transcript.finalize();
                let mut tr = CircuitTranscript::<Blake2b256>::init_from_bytes(&proof);
                assert!(
                    prepare::<Fq, Scheme, _>(pk.get_vk(), &[], &[&inst], &mut tr)
                        .is_ok_and(|g| g.verify(&params.verifier_params()).is_ok())
                );
                assert!(metal::end_request());
            }
            QUOTIENT_CHECK.store(false, Relaxed);
            set_probe(None);
            assert_eq!(hooks::quotient_check_diffs(), diffs);
            let dev = DEV.lock().unwrap().clone();
            let (dns, st) = *dev.iter().min_by_key(|d| d.0).unwrap();
            println!("|   stock samples ns: {stock:?} |");
            let sns = *stock.iter().min().unwrap();
            let peak = dev.iter().map(|d| d.1.peak_bytes).max().unwrap();
            let need = metal::need_bytes(
                shape.ext_k,
                shape.proof_polys,
                shape.key_polys,
                shape.nslots,
            );
            let ms = |ns: u64| ns as f64 / 1e6;
            println!(
                "| {k} | {} | {} | {:.1} | {:.1} | {:.1}x | {:.1} | {:.1} |",
                shape.ext_k,
                t.map_or("default".to_string(), |t| t.to_string()),
                ms(dns),
                ms(sns),
                sns as f64 / dns as f64,
                peak as f64 / 1048576.0,
                need as f64 / 1048576.0
            );
            let stages = |what: &str, ns: u64, st: &metal::QuotientStages| {
                println!(
                    "|   {what} {:.1} ms; stages ms: prepare {:.1}, keys {:.1} (overlapped), cosets {:.1} (gpu {:.1}), numerator {:.1} (gpu {:.1}, {} cb), divide {:.1} (gpu {:.1}), inverse {:.1} (gpu {:.1}), read-back {:.1}; gpu {:.1}; proof polys {}, key polys {}, nslots {}, key copies made {} |",
                    ms(ns),
                    ms(st.prepare),
                    ms(st.keys),
                    ms(st.cosets),
                    ms(st.gpu_cosets),
                    ms(st.numerator),
                    ms(st.gpu_numerator),
                    st.numerator_runs,
                    ms(st.divide),
                    ms(st.gpu_divide),
                    ms(st.inverse),
                    ms(st.gpu_inverse),
                    ms(st.readback),
                    ms(st.gpu_cosets + st.gpu_numerator + st.gpu_divide + st.gpu_inverse),
                    shape.proof_polys,
                    shape.key_polys,
                    shape.nslots,
                    st.key_copies_made
                );
            };
            stages("best call", dns, &st);
            // the first call of each proof: the one a prover makes (keys
            // checked and copied, per-request buffers new to the request)
            let (fns, fst) = *dev.iter().step_by(3).min_by_key(|d| d.0).unwrap();
            stages("first call of a request (best)", fns, &fst);
        }
        qt::set_threads(None);
        metal::set_keep_resident(false);
        drop(key);
    }
}
