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

//! The evaluations and the multi-open of the prover, in parallel tasks of
//! bounded size.
//!
//! The stock code evaluates one polynomial at a time, each split into one
//! chunk per thread, and divides each point set's polynomial by its roots on
//! a single thread. Here every evaluation of a group is cut into chunks of
//! [`EVAL_CHUNK`] coefficients and all chunks of all evaluations are spread
//! over the pool at once, and a division is cut into blocks of a linear
//! recurrence that are solved in parallel and then joined.
//!
//! Where a value is regrouped (the partial sums of an evaluation, the
//! blocks of a division) the result is the same field element as stock:
//! field arithmetic is exact, so any grouping of the same sums and products
//! gives the same element, and the field keeps reduced representations, so
//! the limbs are the same too. The stock evaluation already depends on this:
//! its own grouping follows the thread count.

use ff::Field;
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
    },
    slice::ParallelSliceMut,
};

use crate::{
    poly::{Coeff, Polynomial},
    utils::arithmetic::eval_polynomial_seq,
};

/// Coefficients per evaluation task.
const EVAL_CHUNK: usize = 1 << 13;

/// Coefficients per task of an inner product of polynomials.
const SUM_CHUNK: usize = 1 << 12;

/// Rows per block of a division.
const DIV_BLOCK: usize = 1 << 13;

/// Shorter polynomials are divided by the stock code.
pub(crate) const PAR_MIN_LEN: usize = 1 << 14;

/// The evaluation of each `(polynomial, point)` of `jobs`, in order: the
/// values of [`crate::utils::arithmetic::eval_polynomial`].
pub(crate) fn eval_many<F: Field>(jobs: &[(&[F], F)]) -> Vec<F> {
    let tasks: Vec<(usize, usize)> = (jobs.iter().enumerate())
        .flat_map(|(j, (poly, _))| (0..poly.len().div_ceil(EVAL_CHUNK)).map(move |c| (j, c)))
        .collect();
    let parts: Vec<F> = tasks
        .par_iter()
        .map(|&(j, c)| {
            let (poly, x) = jobs[j];
            let start = c * EVAL_CHUNK;
            let end = (start + EVAL_CHUNK).min(poly.len());
            horner_interleaved(&poly[start..end], x) * x.pow_vartime([start as u64])
        })
        .collect();
    let mut evals = vec![F::ZERO; jobs.len()];
    for (&(j, _), part) in tasks.iter().zip(parts) {
        evals[j] += part;
    }
    evals
}

/// Independent Horner chains per evaluation task.
const CHAINS: usize = 8;

/// `eval_polynomial_seq(poly, x)`, as [`CHAINS`] interleaved Horner chains
/// over consecutive parts of `poly` (the last one also takes the remainder
/// of the division of the length), joined by Horner in `x^part`. A single
/// chain waits on each multiplication before the next; independent chains
/// keep the multiplier busy. The same value: the parts' evaluations scaled
/// by powers of `x` sum to the evaluation of `poly`.
fn horner_interleaved<F: Field>(poly: &[F], x: F) -> F {
    let part = poly.len() / CHAINS;
    if part == 0 {
        return eval_polynomial_seq(poly, x);
    }
    let mut acc = [F::ZERO; CHAINS];
    acc[CHAINS - 1] = eval_polynomial_seq(&poly[CHAINS * part..], x);
    for i in (0..part).rev() {
        for (k, acc) in acc.iter_mut().enumerate() {
            *acc = *acc * x + poly[k * part + i];
        }
    }
    let x_part = x.pow_vartime([part as u64]);
    acc.iter().rev().fold(F::ZERO, |sum, acc| sum * x_part + acc)
}

/// The inner product `Σ_i scalars_i · polys_i` of coefficient-form
/// polynomials of possibly different lengths, as the stock helper of the
/// multi-open: each coefficient is accumulated from zero over the
/// polynomials in the same order, with the same operations; only the cut of
/// the coefficient range into tasks differs.
pub(crate) fn poly_inner_product<F: Field>(
    polys: &[&Polynomial<F, Coeff>],
    scalars: impl IntoIterator<Item = F>,
) -> Polynomial<F, Coeff> {
    let scalars: Vec<F> = scalars.into_iter().take(polys.len()).collect();
    let max_len = polys.iter().map(|p| p.len()).max().unwrap_or(0);
    let mut values: Vec<F> = (0..max_len).into_par_iter().map(|_| F::ZERO).collect();
    values.par_chunks_mut(SUM_CHUNK).enumerate().for_each(|(c, chunk)| {
        let start = c * SUM_CHUNK;
        for (poly, scalar) in polys.iter().zip(scalars.iter()) {
            let pv: &[F] = poly;
            let end = (start + chunk.len()).min(pv.len());
            if start < pv.len() {
                for (out, coeff) in chunk[..end - start].iter_mut().zip(&pv[start..end]) {
                    *out += *coeff * scalar;
                }
            }
        }
    });
    Polynomial {
        values,
        _marker: std::marker::PhantomData,
    }
}

/// `crate::utils::arithmetic::divide_by_roots`: the quotient of `poly` by
/// `∏_i (X - roots_i)`, without the remainder.
///
/// With `d` roots and `v_j` the coefficients of the vanishing product, the
/// quotient `q` of length `m = len - d` satisfies, from the top down,
/// `q_k = a_{k+d} - Σ_{j<d, v_j≠0} v_j · q_{k+d-j}` (`q_k = 0` for `k ≥ m`),
/// which is the stock loop. Counting rows from the top, `p = m - 1 - k`,
/// this is a linear recurrence with lags `d - j`. The rows are cut into
/// blocks; each block is solved in parallel with a zero history, and the
/// true value is that solution plus the contribution of the block's true
/// history, through the solutions `h_s` of the homogeneous recurrence with
/// a unit history at lag `s` (the same for every block). The histories are
/// joined by a short scan over the blocks. The quotient is unique, so the
/// values are the stock ones.
pub(crate) fn divide_by_roots<F: Field>(poly: &[F], roots: &[F]) -> Vec<F> {
    // The vanishing product and its nonzero terms, as stock computes them.
    let mut vanishing = vec![F::ONE];
    for root in roots {
        let mut next = vec![F::ZERO; vanishing.len() + 1];
        for (i, coeff) in vanishing.iter().enumerate() {
            next[i + 1] += coeff;
            next[i] -= *coeff * root;
        }
        vanishing = next;
    }
    let degree = roots.len();
    // (lag, coefficient) with lag = degree - j in 1..=degree.
    let terms: Vec<(usize, F)> = (vanishing[..degree].iter().enumerate())
        .filter(|(_, coeff)| !bool::from(coeff.is_zero()))
        .map(|(j, coeff)| (degree - j, *coeff))
        .collect();
    let m = poly.len().saturating_sub(degree);
    // The history a block needs: its rows read up to `lags` rows above it.
    let lags = terms.iter().map(|(lag, _)| *lag).max().unwrap_or(0);

    assert!(lags <= DIV_BLOCK, "more roots than rows in a block");

    // The input of row `p` (from the top): `a_{k+d}` with `k = m - 1 - p`.
    let input = |p: usize| poly[m - 1 - p + degree];
    // `q` in its natural order: the block `b` from the top is the `b`-th
    // chunk from the end, and its row `t` is the element `len - 1 - t`.
    let mut q: Vec<F> = (0..m).into_par_iter().map(|_| F::ZERO).collect();

    // Solves the block starting at row `first` with a zero history.
    let solve = |first: usize, block: &mut [F]| {
        let len = block.len();
        for t in 0..len {
            let mut value = input(first + t);
            for (lag, coeff) in &terms {
                if *lag <= t {
                    value -= *coeff * block[len - 1 - (t - lag)];
                }
            }
            block[len - 1 - t] = value;
        }
    };
    let homogeneous = || -> Vec<Vec<F>> {
        // `h[s - 1][t]`: the row `t` of a block whose only nonzero history
        // is a one at lag `s` (the row `s` above its first row).
        (1..=lags)
            .into_par_iter()
            .map(|s| {
                let mut h = vec![F::ZERO; DIV_BLOCK.min(m)];
                for t in 0..h.len() {
                    let mut value = F::ZERO;
                    for (lag, coeff) in &terms {
                        if *lag <= t {
                            value -= *coeff * h[t - lag];
                        } else if *lag - t == s {
                            value -= *coeff;
                        }
                    }
                    h[t] = value;
                }
                h
            })
            .collect()
    };
    let (_, h) = rayon::join(
        || {
            q.par_rchunks_mut(DIV_BLOCK)
                .enumerate()
                .for_each(|(b, block)| solve(b * DIV_BLOCK, block))
        },
        homogeneous,
    );

    // The scan: the true history of each block from the true last rows of
    // the block above it. `hist[s - 1]` is the value `s` rows above the
    // block's first row.
    let blocks = m.div_ceil(DIV_BLOCK);
    let mut histories: Vec<Vec<F>> = Vec::with_capacity(blocks);
    histories.push(vec![F::ZERO; lags]);
    for b in 1..blocks {
        let hist_above = &histories[b - 1];
        let next: Vec<F> = (1..=lags)
            .map(|s| {
                // the true value of the row `DIV_BLOCK - s` of the block above
                let t = DIV_BLOCK - s;
                let mut value = q[m - 1 - ((b - 1) * DIV_BLOCK + t)];
                for (r, hr) in hist_above.iter().enumerate() {
                    value += *hr * h[r][t];
                }
                value
            })
            .collect();
        histories.push(next);
    }

    // The true rows: the zero-history solution plus the history's share.
    q.par_rchunks_mut(DIV_BLOCK)
        .zip(histories.par_iter())
        .skip(1)
        .for_each(|(block, hist)| {
            let len = block.len();
            for (r, hr) in hist.iter().enumerate() {
                if bool::from(hr.is_zero()) {
                    continue;
                }
                for (t, hv) in h[r][..len].iter().enumerate() {
                    block[len - 1 - t] += *hr * hv;
                }
            }
        });

    q
}

/// `crate::utils::arithmetic::kate_division` of `a - c` by `X - b`, for any
/// constant `c`: the quotient of a division by `X - b` does not depend on
/// the constant coefficient of the dividend (`q_i = Σ_{j>i} a_j b^{j-i-1}`),
/// so `a` is divided as it is, without the copy that subtracting `c` makes.
/// The same values as the stock `kate_division`, through
/// [`divide_by_roots`] with the single root `b` (its recurrence is the one
/// of `kate_division`: `q_k = a_{k+1} + b · q_{k+1}`).
///
/// # Panics
///
/// If `a` is empty, as `kate_division`.
pub(crate) fn kate_division<F: Field>(a: &[F], b: F) -> Vec<F> {
    assert!(!a.is_empty(), "kate_division of an empty polynomial");
    divide_by_roots(a, &[b])
}

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;

    use ff::{Field, PrimeField, WithSmallOrderMulGroup};
    use midnight_curves::Fq;
    use rand_chacha::ChaCha20Rng;
    use rand_core::{RngCore, SeedableRng};

    use super::*;
    use crate::utils::arithmetic::eval_polynomial;

    fn in_pool<R: Send>(threads: usize, f: impl FnOnce() -> R + Send) -> R {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(f)
    }

    fn random_poly(rng: &mut ChaCha20Rng, n: usize) -> Vec<Fq> {
        (0..n).map(|_| Fq::random(&mut *rng)).collect()
    }

    /// The stock `divide_by_roots`, verbatim.
    fn stock_divide_by_roots<F: Field>(poly: &[F], roots: &[F]) -> Vec<F> {
        let mut vanishing = vec![F::ONE];
        for root in roots {
            let mut next = vec![F::ZERO; vanishing.len() + 1];
            for (i, coeff) in vanishing.iter().enumerate() {
                next[i + 1] += coeff;
                next[i] -= *coeff * root;
            }
            vanishing = next;
        }
        let degree = roots.len();
        let terms: Vec<(usize, F)> = (vanishing[..degree].iter().enumerate())
            .filter(|(_, coeff)| !bool::from(coeff.is_zero()))
            .map(|(j, coeff)| (j, *coeff))
            .collect();
        let mut remainder = poly.to_vec();
        let mut quotient = vec![F::ZERO; poly.len().saturating_sub(degree)];
        for i in (degree..poly.len()).rev() {
            let lead = remainder[i];
            quotient[i - degree] = lead;
            for (j, coeff) in &terms {
                remainder[i - degree + j] -= lead * coeff;
            }
        }
        quotient
    }

    #[test]
    fn eval_many_matches_eval_polynomial() {
        let mut rng = ChaCha20Rng::seed_from_u64(0x3a);
        let lens = [
            0usize,
            1,
            2,
            5,
            EVAL_CHUNK - 1,
            EVAL_CHUNK,
            EVAL_CHUNK + 1,
            70_001,
        ];
        let polys: Vec<Vec<Fq>> = lens.iter().map(|&n| random_poly(&mut rng, n)).collect();
        let mut jobs = Vec::new();
        for p in &polys {
            for x in [
                Fq::ZERO,
                Fq::ONE,
                Fq::random(&mut rng),
                Fq::random(&mut rng),
            ] {
                jobs.push((p.as_slice(), x));
            }
        }
        let stock: Vec<Fq> = jobs.iter().map(|(p, x)| eval_polynomial(p, *x)).collect();
        let seq: Vec<Fq> = jobs.iter().map(|(p, x)| eval_polynomial_seq(p, *x)).collect();
        assert_eq!(stock, seq);
        for threads in [1, 3, 8] {
            assert_eq!(
                in_pool(threads, || eval_many(&jobs)),
                stock,
                "threads {threads}"
            );
        }
        assert!(eval_many::<Fq>(&[]).is_empty());
    }

    #[test]
    fn horner_interleaved_matches_horner() {
        let mut rng = ChaCha20Rng::seed_from_u64(0x3e);
        for n in (0..3 * CHAINS + 2).chain([100, 1001, EVAL_CHUNK]) {
            let poly = random_poly(&mut rng, n);
            for x in [Fq::ZERO, Fq::ONE, -Fq::ONE, Fq::random(&mut rng)] {
                assert_eq!(
                    horner_interleaved(&poly, x),
                    eval_polynomial_seq(&poly, x),
                    "n {n}"
                );
            }
        }
    }

    #[test]
    fn poly_inner_product_matches_stock() {
        let mut rng = ChaCha20Rng::seed_from_u64(0x3b);
        for lens in [
            vec![],
            vec![0],
            vec![3, 1],
            vec![SUM_CHUNK + 3, 7, 2 * SUM_CHUNK, 0, 40_000],
        ] {
            let polys: Vec<Polynomial<Fq, Coeff>> = lens
                .iter()
                .map(|&n| Polynomial {
                    values: random_poly(&mut rng, n),
                    _marker: PhantomData,
                })
                .collect();
            let refs: Vec<&Polynomial<Fq, Coeff>> = polys.iter().collect();
            let x = Fq::random(&mut rng);
            let scalars: Vec<Fq> =
                crate::utils::arithmetic::powers(x).take(lens.len() + 2).collect();
            // the stock helper's body, verbatim
            let max_len = refs.iter().map(|p| p.len()).max().unwrap_or(0);
            let mut stock = vec![Fq::ZERO; max_len];
            crate::utils::arithmetic::parallelize(&mut stock, |chunk, start| {
                for (poly, scalar) in refs.iter().zip(scalars.iter()) {
                    let pv: &[Fq] = poly;
                    let end = (start + chunk.len()).min(pv.len());
                    if start < pv.len() {
                        for (out, coeff) in chunk[..end - start].iter_mut().zip(&pv[start..end]) {
                            *out += *coeff * scalar;
                        }
                    }
                }
            });
            for threads in [1, 4] {
                let par = in_pool(threads, || poly_inner_product(&refs, scalars.clone()));
                assert_eq!(par.values, stock);
            }
        }
    }

    #[test]
    fn divide_by_roots_matches_stock() {
        let mut rng = ChaCha20Rng::seed_from_u64(0x3c);
        let x = Fq::random(&mut rng);
        let omega = Fq::ROOT_OF_UNITY.pow_vartime([1u64 << (Fq::S - 10)]);
        let zeta = Fq::ZETA;
        let root_sets: Vec<Vec<Fq>> = vec![
            vec![],
            vec![x],
            vec![Fq::ZERO],
            vec![x, omega * x],
            vec![x, omega * x, omega.invert().unwrap().pow_vartime([5]) * x],
            vec![x, x],
            vec![
                Fq::random(&mut rng),
                Fq::random(&mut rng),
                Fq::random(&mut rng),
                Fq::random(&mut rng),
            ],
            // the cube roots of x^3: the product is X^3 - x^3, sparse
            vec![x, zeta * x, zeta * zeta * x],
            // the square roots of x^2: X^2 - x^2
            vec![x, -x],
        ];
        let lens = [
            0usize,
            1,
            2,
            3,
            4,
            5,
            100,
            DIV_BLOCK - 1,
            DIV_BLOCK,
            DIV_BLOCK + 2,
            3 * DIV_BLOCK + 17,
            70_000,
        ];
        for roots in &root_sets {
            for &n in &lens {
                let poly = random_poly(&mut rng, n);
                let stock = stock_divide_by_roots(&poly, roots);
                for threads in [1, 4] {
                    let par = in_pool(threads, || divide_by_roots(&poly, roots));
                    assert_eq!(par, stock, "roots {} n {n} threads {threads}", roots.len());
                }
                // the crate function dispatches here at large sizes
                assert_eq!(
                    crate::utils::arithmetic::divide_by_roots(&poly, roots),
                    stock
                );
            }
        }
        // a random number of random roots on random lengths
        for _ in 0..20 {
            let d = (rng.next_u32() % 6) as usize;
            let roots = random_poly(&mut rng, d);
            let n = (rng.next_u32() % 50_000) as usize;
            let poly = random_poly(&mut rng, n);
            assert_eq!(
                divide_by_roots(&poly, &roots),
                stock_divide_by_roots(&poly, &roots)
            );
        }
    }

    #[test]
    fn kate_division_matches_stock() {
        use crate::utils::arithmetic::kate_division as stock_kate;
        let mut rng = ChaCha20Rng::seed_from_u64(0x3d);
        for n in [
            1usize,
            2,
            3,
            DIV_BLOCK,
            DIV_BLOCK + 1,
            2 * DIV_BLOCK + 9,
            1 << 17,
        ] {
            let a = random_poly(&mut rng, n);
            for b in [Fq::ZERO, Fq::ONE, Fq::random(&mut rng)] {
                let c = Fq::random(&mut rng);
                // the stock call of the multi-open: the dividend minus a
                // constant
                let mut shifted = a.clone();
                shifted[0] -= c;
                let stock = stock_kate(&shifted, b);
                assert_eq!(stock, stock_kate(&a, b), "the constant does not matter");
                for threads in [1, 4] {
                    assert_eq!(in_pool(threads, || kate_division(&a, b)), stock, "n {n}");
                }
            }
        }
    }
}
