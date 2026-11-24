//! K15 differential and timing bench for the CUDA hooks.
//!
//! A ZkStdLib relation in the family of the K15 wrap (Poseidon sponge chain plus
//! SHA-256, so gates, the permutation and table lookups are all exercised), at a
//! chosen k. The same source is built twice: with no feature (stock CPU prover)
//! and with `k15-diff-cuda` (midnight-proofs `gpu` plus the midnight-gpu hooks,
//! installed before key generation so the keygen MSMs and FFTs go through the
//! device too).
//!
//! `k15-diff run DIR K NPOS NSHA SEEDS WARM`
//!   writes DIR/<mode>-vk.bin, DIR/<mode>-pk.bin digests and the seeded proofs
//!   DIR/<mode>-proof-<seed>.bin, and prints one JSON line per event: the VK
//!   and PK SHA-256, each seeded proof's SHA-256, and WARM timed proofs with
//!   OS randomness.
//! `k15-diff verify DIR K NPOS NSHA MODE`
//!   (stock build only) verifies every DIR/MODE-proof-*.bin and DIR/MODE-warm-*.bin
//!   against the stock VK with the stock CPU verifier.

use std::{
    env, fs,
    path::{Path, PathBuf},
    time::Instant,
};

use ff::Field;
use midnight_circuits::{
    hash::poseidon::PoseidonChip,
    instructions::{AssignmentInstructions, PublicInputInstructions, hash::HashCPU},
    types::{AssignedByte, Instantiable},
};
use midnight_curves::{Bls12, Fq as F};
use midnight_proofs::{
    circuit::{Layouter, Value},
    plonk::Error,
    poly::kzg::params::ParamsKZG,
    utils::SerdeFormat,
};
use midnight_zk_stdlib::{KZG, MidnightPK, MidnightVK, Relation, ZkStdLib, ZkStdLibArch};
use rand::{SeedableRng, rngs::OsRng};
use rand_chacha::ChaCha20Rng;
use sha2::Digest;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const MODE: &str = match (
    cfg!(feature = "k15-diff-cuda"),
    cfg!(feature = "k15-diff-seeded"),
) {
    (true, true) => "gpu-seeded",
    (true, false) => "gpu",
    (false, true) => "cpu-seeded",
    (false, false) => "cpu",
};

/// Seeds the quotient limb blinding (feature `k15-diff-seeded`, midnight-proofs
/// `gpu-test`). The stock prover draws it from `OsRng`, so without a seed no two
/// proofs are byte-equal.
fn blinding_seed(_seed: Option<u64>) -> bool {
    #[cfg(feature = "k15-diff-seeded")]
    {
        midnight_proofs::gpu::test_switches::set_quotient_blinding_seed(_seed);
        true
    }
    #[cfg(not(feature = "k15-diff-seeded"))]
    false
}

#[derive(Clone, Default)]
struct WrapLike {
    npos: usize,
    nsha: usize,
}

type Witness = ([F; 3], [u8; 24]);

impl WrapLike {
    fn witness() -> Witness {
        let mut rng = ChaCha20Rng::seed_from_u64(0x4b3135);
        let w = core::array::from_fn(|_| F::random(&mut rng));
        let b = core::array::from_fn(|i| (i as u8).wrapping_mul(37).wrapping_add(11));
        (w, b)
    }

    fn instance(&self, (w, b): &Witness) -> Vec<F> {
        let mut x = <PoseidonChip<F> as HashCPU<F, F>>::hash(w);
        for _ in 1..self.npos {
            x = <PoseidonChip<F> as HashCPU<F, F>>::hash(&[x, w[1], w[2]]);
        }
        let mut out = vec![x];
        let mut msg = b.to_vec();
        for _ in 0..self.nsha {
            let d: [u8; 32] = sha2::Sha256::digest(&msg).into();
            out.extend(d.iter().flat_map(AssignedByte::<F>::as_public_input));
            msg = d[..24].to_vec();
        }
        out
    }
}

impl Relation for WrapLike {
    type Instance = Vec<F>;
    type Witness = Witness;
    type Error = Error;

    fn format_instance(instance: &Self::Instance) -> std::result::Result<Vec<F>, Error> {
        Ok(instance.clone())
    }

    fn circuit(
        &self,
        std_lib: &ZkStdLib,
        layouter: &mut impl Layouter<F>,
        _instance: Value<Self::Instance>,
        witness: Value<Self::Witness>,
    ) -> std::result::Result<(), Error> {
        let w = witness.map(|(w, _)| w).transpose_array();
        let b = witness.map(|(_, b)| b).transpose_array();
        let w = std_lib.assign_many(layouter, &w)?;
        let mut x = std_lib.poseidon(layouter, &w)?;
        for _ in 1..self.npos {
            x = std_lib.poseidon(layouter, &[x, w[1].clone(), w[2].clone()])?;
        }
        std_lib.constrain_as_public_input(layouter, &x)?;
        let mut msg = std_lib.assign_many(layouter, &b)?;
        for _ in 0..self.nsha {
            let d = std_lib.sha2_256(layouter, &msg)?;
            d.iter()
                .try_for_each(|byte| std_lib.constrain_as_public_input(layouter, byte))?;
            msg = d[..24].to_vec();
        }
        Ok(())
    }

    fn used_chips(&self) -> ZkStdLibArch {
        ZkStdLibArch {
            poseidon: true,
            sha2_256: true,
            ..ZkStdLibArch::default()
        }
    }

    fn write_relation<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        w.write_all(&(self.npos as u64).to_le_bytes())?;
        w.write_all(&(self.nsha as u64).to_le_bytes())
    }

    fn read_relation<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        let mut a = [0u8; 8];
        r.read_exact(&mut a)?;
        let npos = u64::from_le_bytes(a) as usize;
        r.read_exact(&mut a)?;
        Ok(WrapLike {
            npos,
            nsha: u64::from_le_bytes(a) as usize,
        })
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn sha(b: &[u8]) -> String {
    hex(&sha2::Sha256::digest(b))
}

fn params(k: u32) -> ParamsKZG<Bls12> {
    ParamsKZG::<Bls12>::unsafe_setup(k, ChaCha20Rng::seed_from_u64(0x4b31355253))
}

fn keys(
    params: &ParamsKZG<Bls12>,
    rel: &WrapLike,
) -> (MidnightVK<KZG>, MidnightPK<WrapLike, KZG>, f64) {
    let t = Instant::now();
    let vk = midnight_zk_stdlib::setup_vk::<_, KZG>(params, rel);
    let pk = midnight_zk_stdlib::setup_pk(rel, &vk);
    (vk, pk, t.elapsed().as_secs_f64())
}

#[cfg(feature = "k15-diff-cuda")]
mod device {
    use midnight_curves::Bls12;
    use midnight_gpu as gpu;
    use midnight_proofs::{
        gpu::{self as hooks, adapter},
        poly::{Coeff, LagrangeCoeff, kzg::params::ParamsKZG},
    };

    pub fn install(params: &ParamsKZG<Bls12>) -> [gpu::SrsHandle; 2] {
        assert!(gpu::present().is_some(), "CUDA device unavailable");
        let srs = [params.bases::<Coeff>(), params.bases::<LagrangeCoeff>()]
            .map(|p| gpu::register_srs(adapter::affine(p)));
        assert!(hooks::install(hooks::GpuHooks {
            msm_g1: Some(gpu::msm_g1),
            fft: Some(gpu::fft),
            ifft: Some(gpu::ifft),
            quotient: Some(gpu::quotient),
        }));
        srs
    }

    pub fn register_key(
        pk: &midnight_proofs::plonk::ProvingKey<midnight_curves::Fq, midnight_zk_stdlib::KZG>,
    ) -> Option<gpu::KeyHandle> {
        let polys = pk.gpu_resident_polys()?;
        Some(gpu::register_key(
            &polys.iter().map(|p| adapter::scalars(p)).collect::<Vec<_>>(),
        ))
    }

    pub fn begin() {
        gpu::begin_request();
        gpu::reset_stats();
        hooks::reset_hook_declines();
    }

    pub fn end() -> String {
        let s = gpu::stats();
        let q = gpu::quotient::quotient_counts();
        let out = format!(
            "\"msm\":{},\"ntt\":{},\"quotient_runs\":{},\"device_errors\":{},\"declines\":\"{:?}\",\"hook_declines\":\"{:?}\",\"peak_device_bytes\":{}",
            s.msm,
            s.ntt,
            q.0,
            s.device_errors,
            gpu::declines(),
            hooks::hook_declines(),
            gpu::device_bytes().1
        );
        gpu::end_request();
        out
    }
}

fn prove(
    params: &ParamsKZG<Bls12>,
    pk: &MidnightPK<WrapLike, KZG>,
    rel: &WrapLike,
    inst: &Vec<F>,
    rng: impl rand::RngCore + rand::CryptoRng,
) -> Result<(Vec<u8>, f64)> {
    let t = Instant::now();
    let p = midnight_zk_stdlib::prove::<WrapLike, blake2b_simd::State, KZG>(
        params,
        pk,
        rel,
        inst,
        WrapLike::witness(),
        rng,
    )?;
    Ok((p, t.elapsed().as_secs_f64()))
}

fn device_line() -> String {
    #[cfg(feature = "k15-diff-cuda")]
    return device::end();
    #[cfg(not(feature = "k15-diff-cuda"))]
    String::from("\"device\":null")
}

fn device_begin() {
    #[cfg(feature = "k15-diff-cuda")]
    device::begin();
}

fn run(dir: &Path, k: u32, rel: WrapLike, seeds: u64, warm: usize) -> Result<()> {
    fs::create_dir_all(dir)?;
    let threads = rayon::current_num_threads();
    let params = params(k);
    #[cfg(feature = "k15-diff-cuda")]
    let _srs = device::install(&params);
    device_begin();
    let (vk, pk, keygen_s) = keys(&params, &rel);
    let dev = device_line();
    let mut vkb = Vec::new();
    vk.write(&mut vkb, SerdeFormat::RawBytes)?;
    let mut pkb = Vec::new();
    pk.write(&mut pkb, SerdeFormat::RawBytes)?;
    fs::write(dir.join(format!("{MODE}-vk.bin")), &vkb)?;
    println!(
        "{{\"mode\":\"{MODE}\",\"event\":\"keygen\",\"k\":{k},\"npos\":{},\"nsha\":{},\"threads\":{threads},\"keygen_s\":{keygen_s:.4},\"vk_bytes\":{},\"vk_sha256\":\"{}\",\"pk_bytes\":{},\"pk_sha256\":\"{}\",{dev}}}",
        rel.npos,
        rel.nsha,
        vkb.len(),
        sha(&vkb),
        pkb.len(),
        sha(&pkb)
    );
    drop(pkb);
    #[cfg(feature = "k15-diff-cuda")]
    let key = device::register_key(pk.pk());
    #[cfg(feature = "k15-diff-cuda")]
    println!(
        "{{\"mode\":\"{MODE}\",\"event\":\"register_key\",\"ok\":{}}}",
        key.is_some()
    );
    let inst = rel.instance(&WrapLike::witness());
    for seed in 0..seeds {
        let seeded = blinding_seed(Some(0x51ed ^ seed));
        device_begin();
        let (p, s) = prove(&params, &pk, &rel, &inst, ChaCha20Rng::seed_from_u64(seed))?;
        let dev = device_line();
        fs::write(dir.join(format!("{MODE}-proof-{seed}.bin")), &p)?;
        // In the seeded device build, the same proof with every hook suspended (stock CPU code
        // in the same process), compared byte for byte.
        #[cfg(all(feature = "k15-diff-cuda", feature = "k15-diff-seeded"))]
        let same = {
            let (q, _) = midnight_proofs::gpu::test_switches::without_hooks(|| {
                prove(&params, &pk, &rel, &inst, ChaCha20Rng::seed_from_u64(seed))
            })?;
            format!("{}", q == p)
        };
        #[cfg(not(all(feature = "k15-diff-cuda", feature = "k15-diff-seeded")))]
        let same = "null";
        println!(
            "{{\"mode\":\"{MODE}\",\"event\":\"seeded\",\"seed\":{seed},\"blinding_seeded\":{seeded},\"prove_s\":{s:.4},\"proof_bytes\":{},\"proof_sha256\":\"{}\",\"equals_hooks_suspended\":{same},{dev}}}",
            p.len(),
            sha(&p)
        );
    }
    blinding_seed(None);
    for i in 0..warm {
        device_begin();
        let (p, s) = prove(&params, &pk, &rel, &inst, OsRng)?;
        let dev = device_line();
        fs::write(dir.join(format!("{MODE}-warm-{i}.bin")), &p)?;
        println!("{{\"mode\":\"{MODE}\",\"event\":\"warm\",\"i\":{i},\"prove_s\":{s:.4},{dev}}}");
    }
    Ok(())
}

fn verify(dir: &Path, k: u32, rel: WrapLike, mode: &str) -> Result<()> {
    let params = params(k);
    let (vk, _, _) = keys(&params, &rel);
    let mut vkb = Vec::new();
    vk.write(&mut vkb, SerdeFormat::RawBytes)?;
    let theirs = fs::read(dir.join(format!("{mode}-vk.bin")))?;
    println!(
        "{{\"event\":\"vk_equal\",\"stock_vk_sha256\":\"{}\",\"{mode}_vk_sha256\":\"{}\",\"equal\":{}}}",
        sha(&vkb),
        sha(&theirs),
        vkb == theirs
    );
    let inst = rel.instance(&WrapLike::witness());
    let vp = params.verifier_params();
    let mut files: Vec<PathBuf> = fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let n = p.file_name().unwrap().to_string_lossy().to_string();
            n.starts_with(&format!("{mode}-proof-")) || n.starts_with(&format!("{mode}-warm-"))
        })
        .collect();
    files.sort();
    let mut bad = 0;
    for f in &files {
        let proof = fs::read(f)?;
        let t = Instant::now();
        let ok = midnight_zk_stdlib::verify::<WrapLike, blake2b_simd::State, KZG>(
            &vp, &vk, &inst, None, &proof,
        )
        .is_ok();
        bad += usize::from(!ok);
        println!(
            "{{\"event\":\"verify\",\"file\":\"{}\",\"ok\":{ok},\"verify_s\":{:.4}}}",
            f.file_name().unwrap().to_string_lossy(),
            t.elapsed().as_secs_f64()
        );
    }
    println!(
        "{{\"event\":\"verify_summary\",\"mode\":\"{mode}\",\"n\":{},\"failed\":{bad}}}",
        files.len()
    );
    if bad > 0 {
        return Err("verification failed".into());
    }
    Ok(())
}

fn main() -> Result<()> {
    let a: Vec<String> = env::args().collect();
    let usage =
        "k15-diff run DIR K NPOS NSHA SEEDS WARM | verify DIR K NPOS NSHA MODE | rows NPOS NSHA";
    let arg =
        |i: usize| -> Result<&str> { a.get(i).map(|s| s.as_str()).ok_or_else(|| usage.into()) };
    match arg(1)? {
        "rows" => {
            let rel = WrapLike {
                npos: arg(2)?.parse()?,
                nsha: arg(3)?.parse()?,
            };
            println!("optimal_k={}", midnight_zk_stdlib::optimal_k(&rel));
            Ok(())
        }
        "run" => run(
            Path::new(arg(2)?),
            arg(3)?.parse()?,
            WrapLike {
                npos: arg(4)?.parse()?,
                nsha: arg(5)?.parse()?,
            },
            arg(6)?.parse()?,
            arg(7)?.parse()?,
        ),
        "verify" if MODE == "cpu" => verify(
            Path::new(arg(2)?),
            arg(3)?.parse()?,
            WrapLike {
                npos: arg(4)?.parse()?,
                nsha: arg(5)?.parse()?,
            },
            arg(6)?,
        ),
        _ => Err(usage.into()),
    }
}
