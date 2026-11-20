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

//! The device quotient against the references of midnight-proofs, on the
//! jobs of real proofs seen by a quotient hook installed here.

#![cfg(target_os = "macos")]

use std::{
    ffi::c_void,
    mem::{align_of, offset_of, size_of},
    sync::{
        Mutex, MutexGuard, Once,
        atomic::{AtomicU64, Ordering::Relaxed},
    },
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

macro_rules! same {
    ($a:ty, $b:ty, $($f:ident),*) => {
        assert_eq!(size_of::<$a>(), size_of::<$b>(), stringify!($a));
        assert_eq!(align_of::<$a>(), align_of::<$b>(), stringify!($a));
        $(assert_eq!(offset_of!($a, $f), offset_of!($b, $f), concat!(stringify!($a), ".", stringify!($f)));)*
    };
}

/// The crate's copy of the job types matches `midnight_proofs::gpu::abi`.
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
}

static SERIAL: Mutex<()> = Mutex::new(());

type Probe = Box<dyn FnMut(*const c_void) + Send>;
static PROBE: Mutex<Option<Probe>> = Mutex::new(None);
static QUOTIENT_CALLS: AtomicU64 = AtomicU64::new(0);

unsafe fn quotient_hook(job: *const c_void, out: &mut [Scalar]) -> bool {
    QUOTIENT_CALLS.fetch_add(1, Relaxed);
    if let Some(p) = PROBE.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        p(job);
    }
    // SAFETY: the prover's job, forwarded within its call.
    unsafe { metal::quotient(job, out) }
}

fn lock() -> MutexGuard<'static, ()> {
    static INSTALL: Once = Once::new();
    let g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    INSTALL.call_once(|| {
        assert!(metal::device(), "no Metal device");
        assert!(hooks::install(GpuHooks {
            msm_g1: Some(metal::msm_g1),
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

/// # Safety
/// `job` is the prover's job, within the hook call.
unsafe fn abi<'a>(job: *const c_void) -> &'a pa::QuotientJobAbi {
    // SAFETY: the caller's contract.
    unsafe { &*(job as *const pa::QuotientJobAbi) }
}

/// # Safety
/// For a non-zero count, `p` is valid for `n` reads for `'a`.
unsafe fn sl<'a, T>(p: *const T, n: usize) -> &'a [T] {
    if n == 0 {
        &[]
    } else {
        // SAFETY: the caller's contract.
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

mod range {
    //! `Pow2RangeChip` with nine lookup columns: a lookup of several chunks.

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

static SHAPES: Mutex<Vec<Shape>> = Mutex::new(Vec::new());

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

/// Every row of the numerator, the quotient, and the check mode's comparison
/// with the stock CPU quotient.
#[test]
fn numerator_and_quotient_match_references() {
    let _g = lock();
    SHAPES.lock().unwrap_or_else(|e| e.into_inner()).clear();
    set_probe(Some(differential_probe()));
    QUOTIENT_CHECK.store(true, Relaxed);
    let (checks, diffs) = (hooks::quotient_checks(), hooks::quotient_check_diffs());
    let served = metal::stats().quotient.served;
    let mut proofs = 0u64;
    for seed in [0u64, 3] {
        RandomCircuit::new(seed, false, 6).prove();
        proofs += 1;
    }
    let (params, pk, c) = range::setup(1, 9);
    let (_, ok) = range::prove(&params, &pk, &c);
    assert!(ok, "the range circuit verifies");
    proofs += 1;
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
    assert_eq!(metal::stats().quotient.served - served, 2 * proofs);
    let shapes = SHAPES.lock().unwrap_or_else(|e| e.into_inner()).clone();
    assert_eq!(shapes.len() as u64, proofs);
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

/// Several rows per thread, and many command buffers.
#[test]
fn numerator_independent_of_threads() {
    let _g = lock();
    SHAPES.lock().unwrap_or_else(|e| e.into_inner()).clear();
    set_probe(Some(differential_probe()));
    qt::set_threads(Some(256));
    RandomCircuit::new(10, false, 9).prove();
    qt::set_target_nanos(Some(1));
    RandomCircuit::new(10, false, 9).prove();
    let runs = qt::last_stages().unwrap().numerator_runs;
    qt::set_target_nanos(None);
    qt::set_threads(None);
    set_probe(None);
    assert_eq!(SHAPES.lock().unwrap_or_else(|e| e.into_inner()).len(), 2);
    assert_eq!(
        runs,
        (1 << 11) / 256,
        "one command buffer per dispatch of rows"
    );
}

/// The proof with all four hooks is served, verifies and has the bytes of
/// the proof without hooks.
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

/// `ext_k > 16` reads the low twiddle table past its first entry.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn end_to_end_k16() {
    let _g = lock();
    metal::set_min_k(14);
    let rc = RandomCircuit::new(10, true, 16);
    end_to_end("random k=16", || rc.prove());
}

#[test]
fn declines_leave_out_unchanged() {
    let _g = lock();
    let j = job_of_circuit(3, 7);
    let a = j.abi;
    let mut bad = a;
    bad.abi = pa::QUOTIENT_ABI + 1;
    expect_decline(&bad, "shape");
    metal::set_min_k(a.ext_k + 1);
    expect_decline(&a, "below-min-k");
    metal::set_min_k(1);
    qt::set_memory_limit(Some(metal::need_bytes(a.ext_k, 1, 1, 0) as u64));
    expect_decline(&a, "memory");
    qt::set_memory_limit(None);
    let before = count("shape");
    let mut out = pattern(a.len + 1);
    // SAFETY: the job's storage lives in `j`.
    assert!(!unsafe { metal::quotient(j.ptr(), &mut out) });
    assert!(out == pattern(a.len + 1), "out unchanged");
    assert_eq!(count("shape") - before, 1);
    let before = count("shape");
    let mut out = pattern(a.len);
    // SAFETY: a null job is declined before use.
    assert!(!unsafe { metal::quotient(std::ptr::null(), &mut out) });
    assert!(out == pattern(a.len), "out unchanged");
    assert_eq!(count("shape") - before, 1);
    let mut bad = a;
    bad.y = [u64::MAX; 4];
    expect_decline(&bad, "unreduced");
}

#[test]
fn registered_key_is_resident() {
    let _g = lock();
    let mut j = job_of_circuit(8, 8);
    let en = 1usize << j.abi.ext_k;
    let nkeys = j.keys.len() as u32;
    metal::set_keep_resident(false);
    let handle = metal::register_key(&j.key_slices());
    assert!(handle.verify(&j.key_slices()));
    assert_eq!(handle.resident_bytes(), 0);
    let serve = |j: &OwnedJob| {
        let (out, st) = j.serve();
        assert!(out == j.reference(), "quotient");
        st
    };
    let s1 = serve(&j);
    assert_eq!(
        (s1.resident_polys, s1.uploaded_polys, s1.key_copies_made),
        (nkeys, 0, nkeys)
    );
    assert_eq!(handle.resident_bytes(), (nkeys as usize * en * 32) as u64);
    metal::begin_request();
    let s2 = serve(&j);
    assert_eq!(
        (s2.resident_polys, s2.uploaded_polys, s2.key_copies_made),
        (nkeys, 0, 0)
    );
    let stale = metal::key_stale();
    j.bump_key(nkeys as usize - 1, 3, 1);
    metal::begin_request();
    let s3 = serve(&j);
    assert_eq!(metal::key_stale() - stale, 1);
    assert_eq!((s3.resident_polys, s3.uploaded_polys), (nkeys - 1, 1));
    assert_eq!(metal::keys_registered(), nkeys as usize - 1);
    assert!(metal::end_request());
    assert_eq!(handle.resident_bytes(), 0);
    metal::set_keep_resident(true);
    let s4 = serve(&j);
    assert_eq!(s4.key_copies_made, nkeys - 1);
    assert!(metal::end_request());
    assert!(handle.resident_bytes() > 0);
    assert!(metal::evict_key_copies() > 0);
    assert_eq!(handle.resident_bytes(), 0);
    metal::set_keep_resident(false);
    drop(handle);
    assert_eq!(metal::keys_registered(), 0);
}

struct OwnedProgram {
    ops: Vec<hooks::DeviceOp>,
    reads: Vec<hooks::DeviceRead>,
    uniforms: Vec<Scalar>,
    outputs: Vec<u32>,
}

impl OwnedProgram {
    /// # Safety
    /// `p` is a valid program.
    unsafe fn of(p: &ProgramAbi) -> Self {
        // SAFETY: the caller's contract.
        unsafe {
            OwnedProgram {
                ops: sl(p.ops, p.nops).to_vec(),
                reads: sl(p.reads, p.nreads).to_vec(),
                uniforms: sl(p.uniforms, p.nuniforms).to_vec(),
                outputs: sl(p.outputs, p.noutputs).to_vec(),
            }
        }
    }

    fn abi(&self, nslots: u32) -> ProgramAbi {
        ProgramAbi {
            ops: self.ops.as_ptr(),
            nops: self.ops.len(),
            reads: self.reads.as_ptr(),
            nreads: self.reads.len(),
            uniforms: self.uniforms.as_ptr(),
            nuniforms: self.uniforms.len(),
            outputs: self.outputs.as_ptr(),
            noutputs: self.outputs.len(),
            nslots,
            pad: 0,
        }
    }
}

/// A deep copy of a hook's job that keeps the aliasing of the original. `abi`
/// points into the owned storage, whose heap buffers never move.
struct OwnedJob {
    abi: pa::QuotientJobAbi,
    /// The proof's polynomials (`2^k` each).
    proof: Vec<Vec<Scalar>>,
    /// The proving-key polynomials (`2^ext_k` each), in first-use order.
    keys: Vec<Vec<Scalar>>,
    _ptrs: Vec<Vec<*const Scalar>>,
    _progs: Vec<OwnedProgram>,
    _lookups: Vec<pa::LookupAbi>,
    _lookup_progs: Vec<Vec<ProgramAbi>>,
    _trash: Vec<pa::TrashAbi>,
    _kinds: Vec<u32>,
    _index: Vec<u32>,
    _t_inv: Vec<Scalar>,
}

// SAFETY: the raw pointers point into the job's own storage.
unsafe impl Send for OwnedJob {}

impl OwnedJob {
    /// # Safety
    /// `j` is a valid job (within the hook call).
    unsafe fn of(j: &pa::QuotientJobAbi) -> Self {
        use std::collections::HashMap;
        let (n, en) = (1usize << j.k, 1usize << j.ext_k);
        let mut proof: Vec<Vec<Scalar>> = Vec::new();
        let mut keys: Vec<Vec<Scalar>> = Vec::new();
        let mut pix: HashMap<usize, usize> = HashMap::new();
        let mut kix: HashMap<usize, usize> = HashMap::new();
        let mut proof_i = |p: *const Scalar| -> Option<usize> {
            (!p.is_null()).then(|| {
                *pix.entry(p as usize).or_insert_with(|| {
                    // SAFETY: the job's contract.
                    proof.push(unsafe { sl(p, n) }.to_vec());
                    proof.len() - 1
                })
            })
        };
        let mut key_i = |p: *const Scalar| -> Option<usize> {
            (!p.is_null()).then(|| {
                *kix.entry(p as usize).or_insert_with(|| {
                    // SAFETY: the job's contract.
                    keys.push(unsafe { sl(p, en) }.to_vec());
                    keys.len() - 1
                })
            })
        };
        // SAFETY (all `sl`): the job's contract.
        let advice: Vec<_> =
            unsafe { sl(j.advice, j.nadvice) }.iter().map(|p| proof_i(*p)).collect();
        let instance: Vec<_> =
            unsafe { sl(j.instance, j.ninstance) }.iter().map(|p| proof_i(*p)).collect();
        let fixed: Vec<_> = unsafe { sl(j.fixed, j.nfixed) }.iter().map(|p| key_i(*p)).collect();
        let lag: Vec<_> = [j.l0, j.l_last, j.l_active].iter().map(|p| key_i(*p)).collect();
        let sigma: Vec<_> =
            unsafe { sl(j.perm_sigma, j.perm_ncolumns) }.iter().map(|p| key_i(*p)).collect();
        let products: Vec<_> = unsafe { sl(j.perm_products, j.perm_nsets) }
            .iter()
            .map(|p| proof_i(*p))
            .collect();
        let lookups_in = unsafe { sl(j.lookups, j.nlookups) };
        type L = (
            Option<usize>,
            Option<usize>,
            Vec<Option<usize>>,
            Vec<OwnedProgram>,
        );
        let lks: Vec<L> = lookups_in
            .iter()
            .map(|l| unsafe {
                (
                    proof_i(l.aggregator),
                    proof_i(l.multiplicities),
                    sl(l.helpers, l.nchunks).iter().map(|h| proof_i(*h)).collect(),
                    sl(l.programs, l.nchunks).iter().map(|p| OwnedProgram::of(p)).collect(),
                )
            })
            .collect();
        let trash_in = unsafe { sl(j.trash, j.ntrash) };
        let trs: Vec<(Option<usize>, OwnedProgram)> = trash_in
            .iter()
            .map(|t| (proof_i(t.trash), unsafe { OwnedProgram::of(&t.program) }))
            .collect();
        let gates = unsafe { OwnedProgram::of(&j.gates) };
        let t_inv = unsafe { sl(j.t_inv, j.nt_inv) }.to_vec();
        let kinds = unsafe { sl(j.perm_column_kind, j.perm_ncolumns) }.to_vec();
        let index = unsafe { sl(j.perm_column_index, j.perm_ncolumns) }.to_vec();

        let pp = |i: Option<usize>| i.map_or(std::ptr::null(), |i| proof[i].as_ptr());
        let kp = |i: Option<usize>| i.map_or(std::ptr::null(), |i| keys[i].as_ptr());
        let ptrs: Vec<Vec<*const Scalar>> = vec![
            advice.iter().map(|i| pp(*i)).collect(),
            instance.iter().map(|i| pp(*i)).collect(),
            fixed.iter().map(|i| kp(*i)).collect(),
            sigma.iter().map(|i| kp(*i)).collect(),
            products.iter().map(|i| pp(*i)).collect(),
        ];
        let mut helper_ptrs: Vec<Vec<*const Scalar>> =
            lks.iter().map(|l| l.2.iter().map(|h| pp(*h)).collect()).collect();
        let lookup_progs: Vec<Vec<ProgramAbi>> = lks
            .iter()
            .zip(lookups_in)
            .map(|(l, o)| {
                l.3.iter()
                    .zip(unsafe { sl(o.programs, o.nchunks) })
                    .map(|(p, op)| p.abi(op.nslots))
                    .collect()
            })
            .collect();
        let lookups: Vec<pa::LookupAbi> = lks
            .iter()
            .zip(&helper_ptrs)
            .zip(&lookup_progs)
            .map(|((l, h), p)| pa::LookupAbi {
                aggregator: pp(l.0),
                multiplicities: pp(l.1),
                helpers: h.as_ptr(),
                programs: p.as_ptr(),
                nchunks: h.len(),
            })
            .collect();
        let trash: Vec<pa::TrashAbi> = trs
            .iter()
            .zip(trash_in)
            .map(|((t, p), o)| pa::TrashAbi {
                trash: pp(*t),
                program: p.abi(o.program.nslots),
                selector_column: o.selector_column,
                pad: 0,
            })
            .collect();
        let mut abi = *j;
        abi.advice = ptrs[0].as_ptr();
        abi.instance = ptrs[1].as_ptr();
        abi.fixed = ptrs[2].as_ptr();
        abi.perm_sigma = ptrs[3].as_ptr();
        abi.perm_products = ptrs[4].as_ptr();
        abi.perm_column_kind = kinds.as_ptr();
        abi.perm_column_index = index.as_ptr();
        abi.l0 = kp(lag[0]);
        abi.l_last = kp(lag[1]);
        abi.l_active = kp(lag[2]);
        abi.gates = gates.abi(j.gates.nslots);
        abi.lookups = lookups.as_ptr();
        abi.trash = trash.as_ptr();
        abi.t_inv = t_inv.as_ptr();
        let mut ptrs = ptrs;
        ptrs.append(&mut helper_ptrs);
        let mut progs = vec![gates];
        for (_, _, _, ps) in lks {
            progs.extend(ps);
        }
        progs.extend(trs.into_iter().map(|t| t.1));
        OwnedJob {
            abi,
            proof,
            keys,
            _ptrs: ptrs,
            _progs: progs,
            _lookups: lookups,
            _lookup_progs: lookup_progs,
            _trash: trash,
            _kinds: kinds,
            _index: index,
            _t_inv: t_inv,
        }
    }

    fn ptr(&self) -> *const c_void {
        &self.abi as *const _ as *const c_void
    }

    fn key_slices(&self) -> Vec<&[Scalar]> {
        self.keys.iter().map(|k| k.as_slice()).collect()
    }

    fn serve(&self) -> (Vec<Scalar>, metal::QuotientStages) {
        let mut out = vec![[0u64; 4]; self.abi.len];
        // SAFETY: the job's storage lives in `self`.
        assert!(
            unsafe { metal::quotient(self.ptr(), &mut out) },
            "served: {:?}",
            reasons()
        );
        (out, qt::last_stages().unwrap())
    }

    fn reference(&self) -> Vec<Scalar> {
        // SAFETY: the job's storage lives in `self`.
        let job = unsafe { job_of(&self.abi) };
        let h = job.divide_reference(job.numerator_reference());
        h.iter().map(|x| scalar(*x)).collect()
    }

    fn bump_key(&mut self, key: usize, at: usize, by: u64) {
        let v = &mut self.keys[key][at];
        *v = scalar(from_scalar(*v) + Fq::from(by));
    }
}

fn capture(prove: impl FnOnce()) -> Vec<OwnedJob> {
    static JOBS: Mutex<Vec<OwnedJob>> = Mutex::new(Vec::new());
    JOBS.lock().unwrap_or_else(|e| e.into_inner()).clear();
    set_probe(Some(Box::new(|job| {
        // SAFETY: within the hook call.
        let owned = unsafe { OwnedJob::of(abi(job)) };
        JOBS.lock().unwrap_or_else(|e| e.into_inner()).push(owned);
    })));
    prove();
    set_probe(None);
    std::mem::take(&mut *JOBS.lock().unwrap_or_else(|e| e.into_inner()))
}

fn job_of_circuit(seed: u64, k: u32) -> OwnedJob {
    let mut jobs = capture(|| {
        RandomCircuit::new(seed, false, k).prove();
    });
    assert_eq!(jobs.len(), 1);
    let j = jobs.pop().unwrap();
    assert_eq!(j.serve().0, j.reference());
    j
}

fn pattern(len: usize) -> Vec<Scalar> {
    (0..len as u64)
        .map(|i| [0x5eed_0000 + i, !i, i.rotate_left(17) ^ 0xa5a5, 7])
        .collect()
}

/// `j` declines with `reason`, counted once, and leaves `out` unchanged.
fn expect_decline(j: &pa::QuotientJobAbi, reason: &str) {
    let before = count(reason);
    let declines = metal::quotient_counts().1;
    let want = pattern(j.len);
    let mut out = want.clone();
    // SAFETY: the job's pointers live for the call.
    let ok = unsafe { metal::quotient(j as *const _ as *const c_void, &mut out) };
    assert!(!ok, "{reason}: declined");
    assert!(out == want, "{reason}: out byte-identical");
    assert_eq!(
        count(reason) - before,
        1,
        "{reason}: counted, {:?}",
        reasons()
    );
    assert_eq!(metal::quotient_counts().1 - declines, 1);
}

/// The modulus `r` plus `d`: unreduced (and equal to `d` mod `r`).
fn r_plus(d: u64) -> Scalar {
    [
        0xffff_ffff_0000_0001 + d,
        0x53bd_a402_fffe_5bfe,
        0x3339_d808_09a1_d805,
        0x73ed_a753_299d_7d48,
    ]
}

#[test]
fn malformed_jobs_decline_and_leave_out_unchanged() {
    let _g = lock();
    let j = job_of_circuit(3, 7);
    let a = j.abi;
    // SAFETY (every `unsafe` below): `a` and its variants point into the
    // storage of `j` or of vectors that outlive their use.
    let gates = unsafe { OwnedProgram::of(&a.gates) };
    assert!(!gates.ops.is_empty() && !gates.reads.is_empty());

    let with_op = |f: &dyn Fn(&mut hooks::DeviceOp)| {
        let mut p = unsafe { OwnedProgram::of(&a.gates) };
        f(&mut p.ops[0]);
        let mut bad = a;
        bad.gates = p.abi(a.gates.nslots);
        expect_decline(&bad, "shape");
    };
    let nreads = gates.reads.len() as u32;
    let nunis = gates.uniforms.len() as u32;
    with_op(&|op| op.a = (hooks::TAG_READ << 30) | nreads);
    with_op(&|op| op.b = (hooks::TAG_UNIFORM << 30) | nunis);
    with_op(&|op| op.a = (hooks::TAG_SLOT << 30) | a.nslots);
    with_op(&|op| op.dst = a.nslots);
    with_op(&|op| op.a = 3 << 30);
    {
        let mut p = unsafe { OwnedProgram::of(&a.gates) };
        p.outputs[0] = (hooks::TAG_READ << 30) | nreads;
        let mut bad = a;
        bad.gates = p.abi(a.gates.nslots);
        expect_decline(&bad, "shape");
    }

    assert!(a.nlookups > 0, "the job has a lookup");
    {
        let mut lookups: Vec<pa::LookupAbi> = unsafe { sl(a.lookups, a.nlookups) }.to_vec();
        let mut helpers: Vec<*const Scalar> =
            unsafe { sl(lookups[0].helpers, lookups[0].nchunks) }.to_vec();
        helpers[0] = std::ptr::null();
        lookups[0].helpers = helpers.as_ptr();
        let mut bad = a;
        bad.lookups = lookups.as_ptr();
        expect_decline(&bad, "shape");
    }

    let fixed_read = gates.reads.iter().find(|r| r.kind == hooks::COLUMN_FIXED);
    let fixed_read = fixed_read.expect("the gates read a fixed column");
    {
        let mut fixed: Vec<*const Scalar> = unsafe { sl(a.fixed, a.nfixed) }.to_vec();
        fixed[fixed_read.column as usize] = std::ptr::null();
        let mut bad = a;
        bad.fixed = fixed.as_ptr();
        expect_decline(&bad, "shape");
    }

    // the shift and its inverse decline `shape`, the divisor and `t_inv`
    // `unreduced`
    for v in [r_plus(0), [u64::MAX; 4]] {
        let mut bad = a;
        bad.zeta[0] = v;
        expect_decline(&bad, "shape");
        let mut bad = a;
        bad.zeta[1] = v;
        expect_decline(&bad, "shape");
        let mut bad = a;
        bad.zeta_out[1] = v;
        expect_decline(&bad, "shape");
        let mut bad = a;
        bad.divisor = v;
        expect_decline(&bad, "unreduced");
        for at in [0, a.nt_inv - 1] {
            let mut t_inv: Vec<Scalar> = unsafe { sl(a.t_inv, a.nt_inv) }.to_vec();
            t_inv[at] = v;
            let mut bad = a;
            bad.t_inv = t_inv.as_ptr();
            expect_decline(&bad, "unreduced");
        }
    }
    // `zeta + r`, equal to the shift mod r
    {
        let mut bad = a;
        let z = from_scalar(a.zeta[0]);
        let mut limbs = scalar(z);
        let mut carry = false;
        for (l, r) in limbs.iter_mut().zip(r_plus(0)) {
            let (s1, c1) = l.overflowing_add(r);
            let (s2, c2) = s1.overflowing_add(carry as u64);
            *l = s2;
            carry = c1 | c2;
        }
        if !carry {
            bad.zeta[0] = limbs;
            expect_decline(&bad, "shape");
        }
    }

    // found while copying, with the device at work
    {
        let mut adv: Vec<Scalar> = j.proof[0].clone();
        let last = adv.len() - 1;
        adv[last] = r_plus(2);
        let mut ptrs: Vec<*const Scalar> = unsafe { sl(a.advice, a.nadvice) }.to_vec();
        let target = ptrs[0];
        for p in ptrs.iter_mut().filter(|p| **p == target) {
            *p = adv.as_ptr();
        }
        assert_eq!(target, j.proof[0].as_ptr());
        let mut bad = a;
        bad.advice = ptrs.as_ptr();
        expect_decline(&bad, "unreduced");
    }
    {
        let mut la: Vec<Scalar> = unsafe { sl(a.l_active, 1 << a.ext_k) }.to_vec();
        la[5] = r_plus(3);
        let mut bad = a;
        bad.l_active = la.as_ptr();
        expect_decline(&bad, "unreduced");
    }
    // a memory decline after a registered key was lent
    {
        let handle = metal::register_key(&j.key_slices());
        // SAFETY: the job's storage lives in `j`.
        let need = unsafe { qt::need_bytes_job(j.ptr()) }.expect("checks pass") as u64;
        qt::set_memory_limit(Some(need - 1));
        expect_decline(&a, "memory");
        qt::set_memory_limit(Some(need));
        assert_eq!(j.serve().0, j.reference());
        qt::set_memory_limit(None);
        drop(handle);
    }
    assert_eq!(j.serve().0, j.reference());
    assert!(metal::end_request());
}

/// One handle holding keys of two extended lengths, in both registration
/// orders (which number the slots of a length differently).
#[test]
fn mixed_length_registration_serves_each_length() {
    let _g = lock();
    let a = job_of_circuit(3, 6);
    let b = job_of_circuit(4, 8);
    assert_ne!(a.abi.ext_k, b.abi.ext_k, "two extended lengths");
    let (ra, rb) = (a.reference(), b.reference());
    let (na, nb) = (a.keys.len() as u32, b.keys.len() as u32);
    metal::set_keep_resident(true);
    for (a_first, one_request) in [(true, false), (false, true)] {
        let polys: Vec<&[Scalar]> = if a_first {
            a.key_slices().into_iter().chain(b.key_slices()).collect()
        } else {
            b.key_slices().into_iter().chain(a.key_slices()).collect()
        };
        let handle = metal::register_key(&polys);
        let what = format!("A first {a_first}, one request {one_request}");
        if one_request {
            metal::begin_request();
        }
        let step = |j: &OwnedJob, want: &[Scalar], nkeys: u32, copies: u32, i: usize| {
            if !one_request {
                metal::begin_request();
            }
            let (out, st) = j.serve();
            assert!(out == want, "{what}: quotient {i}");
            assert_eq!(
                (st.resident_polys, st.uploaded_polys, st.key_copies_made),
                (nkeys, 0, copies),
                "{what}: placement {i}"
            );
            if !one_request {
                assert!(metal::end_request());
            }
        };
        step(&a, &ra, na, na, 0);
        step(&b, &rb, nb, nb, 1);
        step(&a, &ra, na, 0, 2);
        step(&b, &rb, nb, 0, 3);
        if one_request {
            assert!(metal::end_request());
        }
        let (out, st) = a.serve();
        assert!(out == ra, "{what}: outside a request");
        assert_eq!((st.resident_polys, st.key_copies_made), (na, 0));
        assert_eq!(
            handle.resident_bytes(),
            ((na as u64) << a.abi.ext_k) * 32 + ((nb as u64) << b.abi.ext_k) * 32,
            "{what}: one copy per length"
        );
        drop(handle);
        assert_eq!(metal::keys_registered(), 0);
    }
    metal::set_keep_resident(false);
    assert!(metal::end_request());
}

#[test]
fn outside_a_request_every_serve_checks() {
    let _g = lock();
    assert!(metal::end_request());
    let mut j = job_of_circuit(8, 7);
    let nkeys = j.keys.len() as u32;
    let handle = metal::register_key(&j.key_slices());
    let (out, st) = j.serve();
    assert!(out == j.reference());
    assert_eq!((st.resident_polys, st.key_copies_made), (nkeys, nkeys));
    let (out, st) = j.serve();
    assert!(out == j.reference());
    assert_eq!((st.resident_polys, st.key_copies_made), (nkeys, 0));
    let stale = metal::key_stale();
    let old = j.reference();
    j.bump_key(nkeys as usize - 1, 3, 1);
    let new = j.reference();
    assert!(old != new, "the change moves the quotient");
    let (out, st) = j.serve();
    assert!(out == new, "the new contents");
    assert_eq!(metal::key_stale() - stale, 1);
    assert_eq!((st.resident_polys, st.uploaded_polys), (nkeys - 1, 1));
    drop(handle);
    assert!(metal::end_request());
}

/// A change after the first serve of a request is seen at the next request.
#[test]
fn inside_a_request_checks_once() {
    let _g = lock();
    let mut j = job_of_circuit(8, 7);
    metal::set_keep_resident(true);
    metal::begin_request();
    let polys = j.key_slices();
    let nkeys = polys.len() as u32;
    let handle = metal::register_key(&polys);
    let old = j.reference();
    let (out, st) = j.serve();
    assert!(out == old);
    assert_eq!(st.key_copies_made, nkeys);
    let stale = metal::key_stale();
    j.bump_key(0, 9, 5);
    let new = j.reference();
    assert!(old != new);
    let (out, st) = j.serve();
    assert!(out == old, "once per request: the copy");
    assert_eq!((st.resident_polys, st.key_copies_made), (nkeys, 0));
    assert_eq!(metal::key_stale(), stale);
    assert!(metal::end_request());
    metal::begin_request();
    let (out, st) = j.serve();
    assert!(out == new, "the next request: the new contents");
    assert_eq!(metal::key_stale() - stale, 1);
    assert_eq!((st.resident_polys, st.uploaded_polys), (nkeys - 1, 1));
    assert!(metal::end_request());
    drop(handle);
    metal::set_keep_resident(false);
}

/// A key changed from C1 to C2 before its slot is filled is not marked
/// present; changed back to C1 later, C1 is served.
#[test]
fn slot_filled_with_changed_contents_is_not_present() {
    let _g = lock();
    let mut j = job_of_circuit(8, 7);
    metal::set_keep_resident(true);
    metal::begin_request();
    let nkeys = j.keys.len() as u32;
    let handle = metal::register_key(&j.key_slices());
    let c1 = j.reference();
    let stale = metal::key_stale();
    let victim = nkeys as usize / 2;
    j.bump_key(victim, 11, 1);
    let c2 = j.reference();
    assert!(c1 != c2);
    let (out, st) = j.serve();
    assert!(out == c2, "C2");
    assert_eq!(metal::key_stale() - stale, 1);
    assert_eq!(
        (st.resident_polys, st.uploaded_polys, st.key_copies_made),
        (nkeys - 1, 1, nkeys - 1),
        "the C2 slot is not present"
    );
    assert_eq!(metal::keys_registered(), nkeys as usize - 1);
    assert!(metal::end_request());
    let v = &mut j.keys[victim][11];
    *v = scalar(from_scalar(*v) - Fq::ONE);
    assert!(j.reference() == c1);
    metal::begin_request();
    let (out, st) = j.serve();
    assert!(out == c1, "C1 again");
    assert_eq!(metal::key_stale() - stale, 1);
    assert_eq!(
        (st.resident_polys, st.uploaded_polys, st.key_copies_made),
        (nkeys - 1, 1, 0)
    );
    assert!(metal::end_request());
    drop(handle);
    metal::set_keep_resident(false);
}

/// `need_bytes_job` and `need_bytes` bound the device bytes a call adds, for
/// a one-pass and a two-pass transform size.
#[test]
fn need_bytes_bounds_the_measured_peak() {
    let _g = lock();
    let mut jobs = vec![job_of_circuit(3, 6)];
    let (params, pk, c) = range::setup(1, 9);
    jobs.extend(capture(|| {
        range::prove(&params, &pk, &c);
    }));
    metal::set_keep_resident(true);
    let measure = |j: &OwnedJob, what: &str| {
        assert!(metal::end_request());
        metal::release_pool();
        metal::begin_request();
        // SAFETY: the job's storage lives in `j`.
        let need = unsafe { qt::need_bytes_job(j.ptr()) }.expect("checks pass");
        let (now0, _) = metal::device_bytes();
        let want = j.reference();
        let (out, _) = j.serve();
        assert!(out == want, "{what}: quotient");
        let (_, peak) = metal::device_bytes();
        let used = peak.saturating_sub(now0);
        let proof_polys = j.proof.len();
        let shape = metal::need_bytes(j.abi.ext_k, proof_polys, j.keys.len(), j.abi.nslots);
        assert!(used > 0, "{what}: measured");
        assert!(
            need as u64 >= used,
            "{what}: need_bytes_job {need} < measured {used}"
        );
        assert!(
            shape as u64 >= used,
            "{what}: need_bytes {shape} < measured {used}"
        );
    };
    for (i, j) in jobs.iter_mut().enumerate() {
        measure(j, &format!("job {i} unregistered"));
        let handle = metal::register_key(&j.key_slices());
        measure(j, &format!("job {i} copy made"));
        measure(j, &format!("job {i} resident"));
        drop(handle);
        let handle = metal::register_key(&j.key_slices());
        j.bump_key(0, 1, 1);
        measure(j, &format!("job {i} stale"));
        drop(handle);
    }
    metal::set_keep_resident(false);
    assert!(metal::end_request());
}
