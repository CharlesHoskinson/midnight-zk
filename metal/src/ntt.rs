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

//! The scalar-field NTT on the device, and the device-side building blocks
//! of the coset transforms (for the quotient).
//!
//! # Entry points
//!
//! [`crate::fft`] / [`crate::ifft`] have admitted the call (`a.len() ==
//! 2^log_n`, `1 <= log_n <= MAX_LOG_N`, `log_n` at least the class's
//! threshold of [`crate::thresholds`], a usable device,
//! `a`, `omega` and the divisor reduced) before they call [`fft`] /
//! [`ifft`]. Those take only the standard `2^log_n`-th root of unity of `Fr`
//! (`ROOT_OF_UNITY` squared `32 - log_n` times, as `EvaluationDomain`
//! derives it) or its inverse, and decline `shape` for any other root. Each
//! ends with exactly one of `account::served` or `account::decline`, and
//! writes `a` only with the complete result, after the last command buffer
//! has completed. Field arithmetic is exact, so the device value is the
//! stock value bit for bit.
//!
//! # Algorithm (`shaders/ntt.metal`)
//!
//! A Stockham transform: `ceil(log_n / 10)` passes of near-equal radix
//! `2^d` (`d <= 10`). A pass is one threadgroup per `2^10` elements (one or
//! more strided columns of `2^d`), one thread per four elements; it loads
//! its columns into threadgroup memory with the inter-pass factors applied,
//! runs a radix-2 decimation in frequency there and writes the outputs in
//! natural order. Natural order in and out: no bit-reversal pass. Passes
//! ping-pong between the data buffer and a scratch buffer of the same
//! length; the last pass reads and writes the same positions, so it runs in
//! place when the ping-pong would end in the scratch. An inverse transform's
//! divisor is folded into the last pass (a scaled copy of one twiddle
//! table), so it costs no extra pass.
//!
//! Why (measured on an M4 Max, see the tests' `bench_ntt_dev`): the
//! transform is bound by field multiplications, not memory (2^24 moves
//! about 60 GB/s), so the design minimises multiplications and passes. The
//! threadgroup memory (32 KiB, 1024 elements) bounds the radix; two passes
//! cover `2^20`, three `2^28`. One thread per four elements beat 32 to 256
//! threads per group and smaller groups; a radix-4 variant holding four
//! elements in registers (two stages per exchange) was slower at every size,
//! by up to 45 % at `2^20`.
//!
//! Twiddles: per direction, `w_{2^16}^j` (`j < 2^16`) and `w_{2^28}^j`
//! (`j < 2^12`) serve every size up to `2^28`: a local butterfly reads one
//! entry, an inter-pass factor one entry while the pass spans at most
//! `2^16` points and the product of two above. They are public, made at the
//! first use and kept on the device for the life of the process (2 MiB, one
//! set for all sizes). A larger high table (up to `2^20`) gained under 1 %.
//!
//! A large transform is split into several command buffers (at most about
//! [`RUN_BUDGET`] field multiplications each, a fraction of a second).
//!
//! # Building blocks for the quotient
//!
//! The `pub(crate)` functions [`ntt_dev`], [`coeff_to_extended_dev`],
//! [`extended_to_coeff_dev`], [`encode_distribute_zeta`], [`encode_scale`]
//! and [`encode_mul_periodic`] work on [`Buffer`]s already on the device and
//! read nothing back. The caller holds the device ([`ctx::lock`]) for the
//! whole sequence. They mirror `mgpu_ntt_dev`, `mgpu_coeff_to_extended_dev`,
//! `mgpu_distribute_zeta_dev`, `mgpu_scale_dev` and `mgpu_mul_periodic_dev`
//! of the CUDA crate.

use std::{
    sync::{Mutex, OnceLock},
    time::Instant,
};

use blst::{blst_fr, blst_fr_inverse, blst_fr_mul, blst_fr_sqr};

use crate::{
    MAX_LOG_N, Scalar,
    account::{self, Class, Reason},
    ctx::{self, Arg, Buffer, Command, Ctx, Error, Pod},
};

/// Bytes of one scalar.
const SCALAR_BYTES: usize = 32;

/// `2^S` is the largest power of two dividing `r - 1`.
const S: u32 = 32;

/// `ROOT_OF_UNITY` of `midnight_curves::Fq` (the `2^S`-th root of unity),
/// Montgomery limbs. The tests check it against midnight-curves.
const ROOT_OF_UNITY: Scalar = [
    0xb9b5_8d8c_5f0e_466a,
    0x5b1b_4c80_1819_d7ec,
    0x0af5_3ae3_52a3_1e64,
    0x5bf3_adda_19e9_b27b,
];

/// The factors of the passes are powers of the `2^UNIVERSAL_LOG`-th root.
const UNIVERSAL_LOG: u32 = 28;
const _: () = assert!(MAX_LOG_N <= UNIVERSAL_LOG);

/// Log2 of the length of the high twiddle table (`w_{2^HI_BITS}^j`); the
/// low table (`w_{2^28}^j`) has `2^(28 - HI_BITS)` elements.
const HI_BITS: u32 = 16;

/// The largest radix (log2) of a pass: `2^MAX_PASS_LOG` elements fill a
/// threadgroup's 32 KiB.
const MAX_PASS_LOG: u32 = 10;

/// Log2 of the elements one threadgroup holds (radix times columns).
const GROUP_LOG: u32 = 10;

/// Estimated field multiplications per command buffer: about 0.35 s at the
/// 4 to 5 G multiplications per second the passes reach on an M4 Max. A
/// pass is never split, so from `2^27` each command buffer holds one pass.
pub(crate) const RUN_BUDGET: u64 = 1_500_000_000;

/// The direction of a transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Dir {
    /// At the standard root `w_n`.
    Forward,
    /// At `w_n^-1`.
    Inverse,
}

impl Dir {
    fn index(self) -> usize {
        match self {
            Dir::Forward => 0,
            Dir::Inverse => 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Host field operations (blst)

fn mul(a: &Scalar, b: &Scalar) -> Scalar {
    let mut r = blst_fr::default();
    // SAFETY: valid pointers to blst_fr values.
    unsafe { blst_fr_mul(&mut r, &blst_fr { l: *a }, &blst_fr { l: *b }) };
    r.l
}

fn square(a: &Scalar) -> Scalar {
    let mut r = blst_fr::default();
    // SAFETY: as `mul`.
    unsafe { blst_fr_sqr(&mut r, &blst_fr { l: *a }) };
    r.l
}

fn invert(a: &Scalar) -> Scalar {
    let mut r = blst_fr::default();
    // SAFETY: as `mul`; the argument is a root of unity, never zero.
    unsafe { blst_fr_inverse(&mut r, &blst_fr { l: *a }) };
    r.l
}

/// `roots[dir][m]`: the standard `2^m`-th root of unity (`m <= 28`) or its
/// inverse.
fn roots() -> &'static [[Scalar; UNIVERSAL_LOG as usize + 1]; 2] {
    static ROOTS: OnceLock<[[Scalar; UNIVERSAL_LOG as usize + 1]; 2]> = OnceLock::new();
    ROOTS.get_or_init(|| {
        let mut w = ROOT_OF_UNITY;
        for _ in UNIVERSAL_LOG..S {
            w = square(&w);
        }
        let mut fwd = [[0u64; 4]; UNIVERSAL_LOG as usize + 1];
        let mut inv = fwd;
        let mut wi = invert(&w);
        for m in (0..=UNIVERSAL_LOG as usize).rev() {
            fwd[m] = w;
            inv[m] = wi;
            w = square(&w);
            wi = square(&wi);
        }
        [fwd, inv]
    })
}

/// The standard `2^log_n`-th root of unity of `Fr` (`dir` forward) or its
/// inverse; `None` above `2^28`.
pub(crate) fn std_root(log_n: u32, dir: Dir) -> Option<Scalar> {
    (log_n <= UNIVERSAL_LOG).then(|| roots()[dir.index()][log_n as usize])
}

/// The direction of `omega` for a `2^log_n` transform, if it is the standard
/// root or its inverse.
fn direction(omega: &Scalar, log_n: u32) -> Option<Dir> {
    if std_root(log_n, Dir::Forward).as_ref() == Some(omega) {
        Some(Dir::Forward)
    } else if std_root(log_n, Dir::Inverse).as_ref() == Some(omega) {
        Some(Dir::Inverse)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Twiddle tables (public, resident)

/// `hi[dir]`: `w_{2^H}^j`, `j < 2^H`; `lo[dir]`: `w_{2^28}^j`,
/// `j < 2^(28 - H)` (`H` = `hi_bits` = [`HI_BITS`]; 2 MiB in all).
struct Tables {
    hi_bits: u32,
    hi: [Buffer; 2],
    lo: [Buffer; 2],
}

fn powers(w: &Scalar, len: usize) -> Vec<Scalar> {
    let mut v = Vec::with_capacity(len);
    // Montgomery one: 2^256 mod r
    let mut acc: Scalar = [
        0x0000_0001_ffff_fffe,
        0x5884_b7fa_0003_4802,
        0x998c_4fef_ecbc_4ff5,
        0x1824_b159_acc5_056f,
    ];
    for _ in 0..len {
        v.push(acc);
        acc = mul(&acc, w);
    }
    v
}

static TABLES: Mutex<Option<&'static Tables>> = Mutex::new(None);

fn tables(ctx: &Ctx) -> Result<&'static Tables, Error> {
    let hb = HI_BITS;
    let mut t = TABLES.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(t) = *t {
        return Ok(t);
    }
    let r = roots();
    let make = |dir: usize, m: u32, bits: u32| ctx.upload(&powers(&r[dir][m as usize], 1 << bits));
    let lb = UNIVERSAL_LOG - hb;
    let made = Tables {
        hi_bits: hb,
        hi: [make(0, hb, hb)?, make(1, hb, hb)?],
        lo: [make(0, UNIVERSAL_LOG, lb)?, make(1, UNIVERSAL_LOG, lb)?],
    };
    // public data kept for the life of the process
    let leaked: &'static Tables = Box::leak(Box::new(made));
    *t = Some(leaked);
    Ok(leaked)
}

// ---------------------------------------------------------------------------
// Plan

/// The parameters of one pass (`ntt::PassParams` of the shader).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct PassParams {
    log_n: u32,
    log_r: u32,
    log_p: u32,
    log_g: u32,
    flags: u32,
    n_src: u32,
    hi_bits: u32,
    pad: u32,
    z1: Scalar,
    z2: Scalar,
}
// SAFETY: `repr(C)`, eight `u32` then two `[u64; 4]`: no padding, every bit
// pattern valid.
unsafe impl Pod for PassParams {}

const F_TWIDDLE: u32 = 1;
const F_TWO_LEVEL: u32 = 2;
const F_COSET: u32 = 4;

/// The radices (log2) of the passes of a `2^log_n` transform: as few passes
/// as the threadgroup memory allows, of near-equal radices (larger first).
fn plan(log_n: u32) -> Vec<u32> {
    let passes = log_n.div_ceil(MAX_PASS_LOG).max(1);
    let base = log_n / passes;
    let extra = log_n % passes;
    (0..passes).map(|i| base + u32::from(i < extra)).collect()
}

/// The scratch a [`ntt_dev`] or [`coeff_to_extended_dev`] of `2^log_n`
/// elements needs, in bytes: none for a one-pass size (`log_n <= 10`), else
/// a buffer of the transform's length.
pub(crate) fn scratch_bytes(log_n: u32) -> usize {
    if plan(log_n).len() > 1 {
        SCALAR_BYTES << log_n
    } else {
        0
    }
}

/// What a sequence of passes reads and writes.
struct Job<'b> {
    log_n: u32,
    dir: Dir,
    /// The first pass's input: `2^log_n` elements, or `n_src` with `coset`.
    src: &'b Buffer,
    /// The result's buffer, and the byte offset of the result in it.
    dst: &'b Buffer,
    dst_off: usize,
    /// Scratch of `2^log_n` elements (unused by one pass).
    tmp: &'b Buffer,
    /// The divisor folded into the last pass.
    scale: Option<Scalar>,
    /// `(n_src, [z1, z2])`: the first pass reads `src` as coeff_to_extended
    /// does (index `i < n_src` times `[1, z1, z2][i mod 3]`, zero beyond).
    coset: Option<(usize, [Scalar; 2])>,
    /// `(t, log_tn)`: before the first pass, `src[i] *= t[i mod 2^log_tn]`
    /// ([`encode_mul_periodic`], in the first command buffer; `src` is then
    /// `dst` at offset 0).
    pre_mul: Option<(&'b Buffer, u32)>,
    /// After the last pass, `dst[i] *= [1, z1, z2][i mod 3]`
    /// ([`encode_distribute_zeta`], in the last command buffer).
    post_zeta: Option<[Scalar; 2]>,
}

fn check_len(b: &Buffer, elems: usize, what: &str) -> Result<(), Error> {
    if b.len_bytes() < elems * SCALAR_BYTES {
        return Err(Error::Device(format!(
            "ntt: the {what} buffer holds {} bytes, {} needed",
            b.len_bytes(),
            elems * SCALAR_BYTES
        )));
    }
    Ok(())
}

/// Runs the passes of `job`, in as many command buffers as [`RUN_BUDGET`]
/// asks. Returns the GPU time.
fn run_job(ctx: &Ctx, job: &Job<'_>) -> Result<u64, Error> {
    let log_n = job.log_n;
    if log_n == 0 || log_n > MAX_LOG_N {
        return Err(Error::Device(format!("ntt: size 2^{log_n}")));
    }
    let n = 1usize << log_n;
    let radices = plan(log_n);
    let passes = radices.len();
    if !job.dst_off.is_multiple_of(SCALAR_BYTES)
        || job.dst.len_bytes() < job.dst_off + n * SCALAR_BYTES
        || (job.dst_off != 0
            && (std::ptr::eq(job.src, job.dst) || job.pre_mul.is_some() || job.post_zeta.is_some()))
    {
        return Err(Error::Device(format!(
            "ntt: the data buffer holds {} bytes, {} needed at offset {}",
            job.dst.len_bytes(),
            n * SCALAR_BYTES,
            job.dst_off
        )));
    }
    if job.pre_mul.is_some() && (!std::ptr::eq(job.src, job.dst) || job.coset.is_some()) {
        return Err(Error::Device("ntt: a periodic factor on a copy".into()));
    }
    if passes > 1 {
        check_len(job.tmp, n, "scratch")?;
    }
    match job.coset {
        Some((n_src, _)) => {
            if n_src > n {
                return Err(Error::Device(
                    "ntt: source longer than the transform".into(),
                ));
            }
            check_len(job.src, n_src, "source")?
        }
        None => check_len(job.src, n, "source")?,
    }
    let tables = tables(ctx)?;
    let d = job.dir.index();
    let hi = &tables.hi[d];

    if passes > 1 && (std::ptr::eq(job.tmp, job.dst) || std::ptr::eq(job.tmp, job.src)) {
        return Err(Error::Device("ntt: the scratch is also the data".into()));
    }

    // the divisor, folded into the low table of the last pass (a scaled copy
    // made in the first command buffer)
    let scaled = match job.scale {
        Some(_) => Some(ctx.alloc_pooled(SCALAR_BYTES << (UNIVERSAL_LOG - tables.hi_bits))?),
        None => None,
    };

    // (input, output) of each pass, as (buffer, byte offset): the passes
    // before the last alternate between `tmp` and `dst` (never writing their
    // own input); the last pass writes `dst`, in place when its input is
    // `dst` already.
    type At<'x> = (&'x Buffer, usize);
    let io: Vec<(At, At)> = {
        let mut cur = (job.src, 0);
        (0..passes)
            .map(|q| {
                let out = if q + 1 == passes || q % 2 == 1 {
                    (job.dst, job.dst_off)
                } else {
                    (job.tmp, 0)
                };
                let pair = (cur, out);
                cur = out;
                pair
            })
            .collect()
    };
    let post_zeta = job.post_zeta;

    let mut params = Vec::with_capacity(passes);
    let mut log_p = 0u32;
    for (q, &r) in radices.iter().enumerate() {
        let last = q + 1 == passes;
        let scale_here = last && scaled.is_some();
        let log_pr = log_p + r;
        let mut flags = 0;
        if log_p > 0 || scale_here {
            flags |= F_TWIDDLE;
        }
        if log_pr > tables.hi_bits || scale_here {
            flags |= F_TWO_LEVEL;
        }
        let mut p = PassParams {
            log_n,
            log_r: r,
            log_p,
            log_g: GROUP_LOG.saturating_sub(r).min(log_n - r),
            flags,
            hi_bits: tables.hi_bits,
            ..Default::default()
        };
        if let (0, Some((n_src, z))) = (q, job.coset) {
            p.flags |= F_COSET;
            p.n_src = n_src as u32;
            p.z1 = z[0];
            p.z2 = z[1];
        }
        params.push(p);
        log_p = log_pr;
    }

    let mut ns = 0;
    let mut q = 0;
    while q < passes {
        // the passes of this command buffer
        let mut end = q;
        let mut cost = 0u64;
        while end < passes {
            let p = &params[end];
            let c = (n as u64) * (u64::from(p.log_r) / 2 + 1 + 2 * u64::from(p.flags & 1));
            if end > q && cost + c > RUN_BUDGET {
                break;
            }
            cost += c;
            end += 1;
        }
        let lo = scaled.as_ref();
        let first = q == 0;
        let last_cb = end == passes;
        ns += ctx
            .run(|cmd| {
                if let (true, Some((t, log_tn))) = (first, job.pre_mul) {
                    encode_mul_periodic(cmd, job.dst, n, t, log_tn)?;
                }
                if let (true, Some(b), Some(c)) = (first, lo, job.scale.as_ref()) {
                    cmd.dispatch(
                        "ntt_scale_copy",
                        1usize << (UNIVERSAL_LOG - tables.hi_bits),
                        &[Arg::Buffer(&tables.lo[d]), Arg::Buffer(b), Arg::value(c)],
                    )?;
                }
                for i in q..end {
                    let p = &params[i];
                    let lo_tab = match (i + 1 == passes, lo) {
                        (true, Some(s)) => s,
                        _ => &tables.lo[d],
                    };
                    let ((src, src_off), (dst, dst_off)) = io[i];
                    let args = [
                        Arg::BufferAt(src, src_off),
                        Arg::BufferAt(dst, dst_off),
                        Arg::Buffer(hi),
                        Arg::Buffer(lo_tab),
                        Arg::value(p),
                    ];
                    let e_log = p.log_r + p.log_g;
                    // one thread per four elements (two butterflies of each
                    // stage): the best of 32 to 256 threads per 1024
                    // elements measured on an M4 Max
                    let tpg = 1usize << e_log.saturating_sub(2);
                    cmd.dispatch_groups(
                        "ntt_pass",
                        n >> e_log,
                        tpg,
                        &args,
                        &[SCALAR_BYTES << e_log],
                    )?;
                }
                if let (true, Some(z)) = (last_cb, post_zeta.as_ref()) {
                    // the data is at offset 0 here (checked above)
                    encode_distribute_zeta(cmd, job.dst, n, z)?;
                }
                Ok(())
            })?
            .1;
        q = end;
    }
    Ok(ns)
}

// ---------------------------------------------------------------------------
// Building blocks (device buffers, nothing read back)

/// The NTT of the first `2^log_n` elements of `a`, in place, natural order:
/// at the standard root `w_n` (`Dir::Forward`) or `w_n^-1`
/// (`Dir::Inverse`), then every element times `scale` if given (folded into
/// the last pass). `tmp` is scratch of at least [`scratch_bytes`]`(log_n)`
/// bytes (not read for `log_n <= 10`; it may then be `a` itself); its
/// contents afterwards are intermediate values. `1 <= log_n <= 28`.
///
/// Runs one or more command buffers and returns the GPU time in
/// nanoseconds. The CUDA counterpart is `mgpu_ntt_dev` (plus
/// `mgpu_scale_dev`).
pub(crate) fn ntt_dev(
    ctx: &Ctx,
    a: &Buffer,
    tmp: &Buffer,
    log_n: u32,
    dir: Dir,
    scale: Option<&Scalar>,
) -> Result<u64, Error> {
    run_job(
        ctx,
        &Job {
            log_n,
            dir,
            src: a,
            dst: a,
            dst_off: 0,
            tmp,
            scale: scale.copied(),
            coset: None,
            pre_mul: None,
            post_zeta: None,
        },
    )
}

/// `coeff_to_extended` on the device: `dst[..2^ext_k]` = the forward NTT at
/// the standard `2^ext_k` root of `src[i] * [1, zeta[0], zeta[1]][i mod 3]`
/// for `i < 2^k`, zero-padded to `2^ext_k`. `src` (at least `2^k` elements)
/// is not modified; the distribution and the padding happen inside the
/// first pass, with no extra pass. `tmp` is scratch of at least
/// [`scratch_bytes`]`(ext_k)` bytes. `1 <= k <= ext_k <= 28`. `zeta` is
/// `[g_coset, g_coset_inv]` of the domain (`coset_zetas`).
///
/// Returns the GPU time. The CUDA counterpart is `mgpu_coeff_to_extended_dev`.
pub(crate) fn coeff_to_extended_dev(
    ctx: &Ctx,
    src: &Buffer,
    dst: &Buffer,
    tmp: &Buffer,
    k: u32,
    ext_k: u32,
    zeta: &[Scalar; 2],
) -> Result<u64, Error> {
    coeff_to_extended_dev_at(ctx, src, dst, 0, tmp, k, ext_k, zeta)
}

/// [`coeff_to_extended_dev`] into `dst` from the byte offset `dst_off` (a
/// multiple of 32): the result is `2^ext_k` elements there, so that a coset
/// is written straight into its place in a larger buffer.
#[allow(clippy::too_many_arguments)]
pub(crate) fn coeff_to_extended_dev_at(
    ctx: &Ctx,
    src: &Buffer,
    dst: &Buffer,
    dst_off: usize,
    tmp: &Buffer,
    k: u32,
    ext_k: u32,
    zeta: &[Scalar; 2],
) -> Result<u64, Error> {
    if k < 1 || k > ext_k || ext_k > MAX_LOG_N {
        return Err(Error::Device(format!(
            "coeff_to_extended: dimensions k {k}, ext_k {ext_k}"
        )));
    }
    if std::ptr::eq(src, dst) || std::ptr::eq(src, tmp) {
        return Err(Error::Device(
            "coeff_to_extended: the source is also the destination or the scratch".into(),
        ));
    }
    run_job(
        ctx,
        &Job {
            log_n: ext_k,
            dir: Dir::Forward,
            src,
            dst,
            dst_off,
            tmp,
            scale: None,
            coset: Some((1usize << k, *zeta)),
            pre_mul: None,
            post_zeta: None,
        },
    )
}

/// `extended_to_coeff` on the device, in place on `a[..2^ext_k]`: the NTT at
/// `w^-1` of the `2^ext_k` domain, every element times `divisor` (the
/// domain's `extended_ifft_divisor`, folded into the last pass), then
/// [`encode_distribute_zeta`] with `zeta_out` (`[g_coset_inv, g_coset]`, in
/// the last command buffer of the transform). `tmp` as for [`ntt_dev`].
/// Returns the GPU time.
pub(crate) fn extended_to_coeff_dev(
    ctx: &Ctx,
    a: &Buffer,
    tmp: &Buffer,
    ext_k: u32,
    divisor: &Scalar,
    zeta_out: &[Scalar; 2],
) -> Result<u64, Error> {
    run_job(
        ctx,
        &Job {
            log_n: ext_k,
            dir: Dir::Inverse,
            src: a,
            dst: a,
            dst_off: 0,
            tmp,
            scale: Some(*divisor),
            coset: None,
            pre_mul: None,
            post_zeta: Some(*zeta_out),
        },
    )
}

/// [`encode_mul_periodic`] of `a` by `t` (`2^log_tn` elements: the division
/// by the vanishing polynomial), then [`extended_to_coeff_dev`], with the
/// multiplication in the transform's first command buffer. Returns the GPU
/// time.
#[allow(clippy::too_many_arguments)]
pub(crate) fn divide_extended_to_coeff_dev(
    ctx: &Ctx,
    a: &Buffer,
    tmp: &Buffer,
    ext_k: u32,
    divisor: &Scalar,
    zeta_out: &[Scalar; 2],
    t: &Buffer,
    log_tn: u32,
) -> Result<u64, Error> {
    run_job(
        ctx,
        &Job {
            log_n: ext_k,
            dir: Dir::Inverse,
            src: a,
            dst: a,
            dst_off: 0,
            tmp,
            scale: Some(*divisor),
            coset: None,
            pre_mul: Some((t, log_tn)),
            post_zeta: Some(*zeta_out),
        },
    )
}

fn elementwise_len(a: &Buffer, n: usize) -> Result<(), Error> {
    check_len(a, n, "data")?;
    if n > 1usize << MAX_LOG_N {
        return Err(Error::Device(format!("ntt: {n} elements")));
    }
    Ok(())
}

/// `MASKS[m] = 2^m - 1`: kernel arguments that live as long as any command.
static MASKS: [u32; 32] = {
    let mut m = [0u32; 32];
    let mut i = 0;
    while i < 32 {
        m[i] = ((1u64 << i) - 1) as u32;
        i += 1;
    }
    m
};

/// Encodes `a[i] *= [1, zeta[0], zeta[1]][i mod 3]` for `i < n` into `cmd`
/// (`distribute_powers_zeta`). The CUDA counterpart is
/// `mgpu_distribute_zeta_dev`.
pub(crate) fn encode_distribute_zeta<'a>(
    cmd: &mut Command<'a>,
    a: &'a Buffer,
    n: usize,
    zeta: &'a [Scalar; 2],
) -> Result<(), Error> {
    elementwise_len(a, n)?;
    cmd.dispatch(
        "ntt_distribute_zeta",
        n,
        &[Arg::Buffer(a), Arg::value(&zeta[0]), Arg::value(&zeta[1])],
    )
}

/// Encodes `a[i] *= c` for `i < n` into `cmd`. The CUDA counterpart is
/// `mgpu_scale_dev`.
pub(crate) fn encode_scale<'a>(
    cmd: &mut Command<'a>,
    a: &'a Buffer,
    n: usize,
    c: &'a Scalar,
) -> Result<(), Error> {
    elementwise_len(a, n)?;
    cmd.dispatch("ntt_scale", n, &[Arg::Buffer(a), Arg::value(c)])
}

/// Encodes `a[i] *= t[i mod 2^log_tn]` for `i < n` into `cmd`, `t` a buffer
/// of at least `2^log_tn` elements (`divide_by_vanishing_poly`: `t` the
/// inverse evaluations of the vanishing polynomial on the coset, whose count
/// `2^(ext_k - k)` is a power of two, and the stock code indexes them with
/// `i & (len - 1)`). The CUDA counterpart is `mgpu_mul_periodic_dev`.
pub(crate) fn encode_mul_periodic<'a>(
    cmd: &mut Command<'a>,
    a: &'a Buffer,
    n: usize,
    t: &'a Buffer,
    log_tn: u32,
) -> Result<(), Error> {
    elementwise_len(a, n)?;
    if log_tn > MAX_LOG_N {
        return Err(Error::Device(format!("mul_periodic: table of 2^{log_tn}")));
    }
    check_len(t, 1usize << log_tn, "table")?;
    cmd.dispatch(
        "ntt_mul_periodic",
        n,
        &[
            Arg::Buffer(a),
            Arg::Buffer(t),
            Arg::value(&MASKS[log_tn as usize]),
        ],
    )
}

// ---------------------------------------------------------------------------
// Host copies

/// Copies `src` into `dst` (equal lengths), on several threads when long:
/// a single thread does not reach the memory bandwidth of the machine.
fn par_copy(dst: &mut [Scalar], src: &[Scalar]) {
    const PART: usize = 1 << 15;
    if src.len() <= 2 * PART {
        dst.copy_from_slice(src);
        return;
    }
    let threads = std::thread::available_parallelism().map_or(4, |t| t.get()).min(8);
    let per = src.len().div_ceil(threads).max(PART);
    std::thread::scope(|s| {
        for (d, c) in dst.chunks_mut(per).zip(src.chunks(per)) {
            s.spawn(move || d.copy_from_slice(c));
        }
    });
}

/// `[copy in (with the buffers), device part (wall, with the wait for the
/// device), GPU, copy out]` nanoseconds of the last host transform.
static LAST_HOST: Mutex<[u64; 4]> = Mutex::new([0; 4]);

/// The transform of a host slice through per-request buffers (`ntt-data`,
/// and `ntt-scratch` above one pass) that the call takes for itself
/// ([`ctx::take_request_buffer`]): the copies in and out run without the
/// device, so the copies of calls made from several threads overlap the
/// device work of the others. `a` is written only after every command
/// buffer has completed.
fn transform_host(
    a: &mut [Scalar],
    log_n: u32,
    dir: Dir,
    scale: Option<&Scalar>,
) -> Result<u64, Error> {
    let t0 = Instant::now();
    let ctx = ctx::context()?;
    let bytes = std::mem::size_of_val(a);
    let mut data = ctx::take_request_buffer("ntt-data", bytes)?;
    par_copy(&mut data.buf_mut().as_mut_slice::<Scalar>()[..a.len()], a);
    let tmp = if scratch_bytes(log_n) > 0 {
        Some(ctx::take_request_buffer("ntt-scratch", bytes)?)
    } else {
        None
    };
    let t1 = Instant::now();
    let ns = {
        let _device = ctx::lock()?;
        let d = data.buf();
        ntt_dev(
            ctx,
            d,
            tmp.as_ref().map_or(d, |t| t.buf()),
            log_n,
            dir,
            scale,
        )?
    };
    let t2 = Instant::now();
    par_copy(a, &data.buf().as_slice::<Scalar>()[..a.len()]);
    let t3 = Instant::now();
    data.give_back();
    if let Some(t) = tmp {
        t.give_back();
    }
    let nanos = |d: std::time::Duration| d.as_nanos() as u64;
    *LAST_HOST.lock().unwrap_or_else(|e| e.into_inner()) =
        [nanos(t1 - t0), nanos(t2 - t1), ns, nanos(t3 - t2)];
    Ok(ns)
}

fn finish(class: Class, r: Result<u64, Error>) -> bool {
    match r {
        Ok(ns) => {
            account::served(class, ns);
            true
        }
        Err(e) => account::decline(class, Reason::of(&e)),
    }
}

/// The FFT of `a` at `omega`, natural order, in place. `false`: declined, `a`
/// unchanged.
pub(crate) fn fft(a: &mut [Scalar], omega: Scalar, log_n: u32) -> bool {
    let Some(dir) = direction(&omega, log_n) else {
        return account::decline(Class::Fft, Reason::Shape);
    };
    finish(Class::Fft, transform_host(a, log_n, dir, None))
}

/// The FFT of `a` at `omega_inv`, then each element times `divisor`, in
/// place. `false`: declined, `a` unchanged.
pub(crate) fn ifft(a: &mut [Scalar], omega_inv: Scalar, log_n: u32, divisor: Scalar) -> bool {
    let Some(dir) = direction(&omega_inv, log_n) else {
        return account::decline(Class::Ifft, Reason::Shape);
    };
    finish(Class::Ifft, transform_host(a, log_n, dir, Some(&divisor)))
}

// ---------------------------------------------------------------------------
// Test support

/// Host-to-host wrappers of the device building blocks, for the crate's
/// integration tests (not a stable interface).
pub mod test_support {
    use super::*;

    /// Runs `f` holding the device.
    fn on_device<R>(f: impl FnOnce(&Ctx) -> Result<R, Error>) -> Result<R, String> {
        let s = ctx::lock().map_err(|e| e.to_string())?;
        f(s.ctx()).map_err(|e| e.to_string())
    }

    fn scratch(ctx: &Ctx, log_n: u32) -> Result<Buffer, Error> {
        ctx.alloc(scratch_bytes(log_n))
    }

    /// `coeff_to_extended` through [`coeff_to_extended_dev`]: `coeffs` has
    /// `2^k` elements; the result `2^ext_k`.
    pub fn coeff_to_extended(
        coeffs: &[Scalar],
        k: u32,
        ext_k: u32,
        zeta: [Scalar; 2],
    ) -> Result<Vec<Scalar>, String> {
        if coeffs.len() != 1usize << k {
            return Err("length".into());
        }
        on_device(|ctx| {
            let src = ctx.upload(coeffs)?;
            let dst = ctx.alloc(SCALAR_BYTES << ext_k)?;
            let tmp = scratch(ctx, ext_k)?;
            coeff_to_extended_dev(ctx, &src, &dst, &tmp, k, ext_k, &zeta)?;
            Ok(dst.as_slice::<Scalar>()[..1usize << ext_k].to_vec())
        })
    }

    /// `extended_to_coeff` through [`extended_to_coeff_dev`] (the whole
    /// `2^ext_k` result, before any truncation).
    pub fn extended_to_coeff(
        values: &[Scalar],
        ext_k: u32,
        divisor: Scalar,
        zeta_out: [Scalar; 2],
    ) -> Result<Vec<Scalar>, String> {
        if values.len() != 1usize << ext_k {
            return Err("length".into());
        }
        on_device(|ctx| {
            let a = ctx.upload(values)?;
            let tmp = scratch(ctx, ext_k)?;
            extended_to_coeff_dev(ctx, &a, &tmp, ext_k, &divisor, &zeta_out)?;
            Ok(a.as_slice::<Scalar>()[..values.len()].to_vec())
        })
    }

    /// [`ntt_dev`] on a copy of `values` (`inverse`: at `w^-1`).
    pub fn ntt(
        values: &[Scalar],
        log_n: u32,
        inverse: bool,
        scale: Option<Scalar>,
    ) -> Result<Vec<Scalar>, String> {
        let dir = if inverse { Dir::Inverse } else { Dir::Forward };
        on_device(|ctx| {
            let a = ctx.upload(values)?;
            let tmp = scratch(ctx, log_n)?;
            ntt_dev(ctx, &a, &tmp, log_n, dir, scale.as_ref())?;
            Ok(a.as_slice::<Scalar>()[..values.len()].to_vec())
        })
    }

    /// The GPU time in nanoseconds of `reps` back-to-back [`ntt_dev`] calls
    /// (forward) on one device buffer of `2^log_n` elements.
    pub fn ntt_gpu_nanos(log_n: u32, reps: usize) -> Result<Vec<u64>, String> {
        on_device(|ctx| {
            let a = ctx.alloc(SCALAR_BYTES << log_n)?;
            let tmp = scratch(ctx, log_n)?;
            (0..reps).map(|_| ntt_dev(ctx, &a, &tmp, log_n, Dir::Forward, None)).collect()
        })
    }

    /// The elementwise blocks on a copy of `values`, in one command buffer:
    /// [`encode_distribute_zeta`] with `zeta`, then [`encode_scale`] by `c`,
    /// then [`encode_mul_periodic`] with `table` (a power-of-two length).
    pub fn elementwise(
        values: &[Scalar],
        zeta: [Scalar; 2],
        c: Scalar,
        table: &[Scalar],
    ) -> Result<Vec<Scalar>, String> {
        if !table.len().is_power_of_two() {
            return Err("table length".into());
        }
        on_device(|ctx| {
            let a = ctx.upload(values)?;
            let t = ctx.upload(table)?;
            let n = values.len();
            ctx.run(|cmd| {
                encode_distribute_zeta(cmd, &a, n, &zeta)?;
                encode_scale(cmd, &a, n, &c)?;
                encode_mul_periodic(cmd, &a, n, &t, table.len().ilog2())
            })?;
            Ok(a.as_slice::<Scalar>()[..n].to_vec())
        })
    }

    /// The standard `2^log_n`-th root of unity (or its inverse) as this
    /// module derives it.
    pub fn root(log_n: u32, inverse: bool) -> Option<Scalar> {
        std_root(log_n, if inverse { Dir::Inverse } else { Dir::Forward })
    }

    /// `[copy in, device part (wall), GPU, copy out]` nanoseconds of the
    /// last [`crate::fft`] / [`crate::ifft`] served (copy in includes taking
    /// the device and the buffers).
    pub fn last_host_timing() -> [u64; 4] {
        *LAST_HOST.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The radices (log2) of the passes of a `2^log_n` transform.
    pub fn passes(log_n: u32) -> Vec<u32> {
        plan(log_n)
    }
}
