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
//! Where a value is regrouped (the partial sums of an evaluation, the blocks
//! of a division) the result is the same field element as stock: field
//! arithmetic is exact, so any grouping of the same sums and products gives
//! the same element, and the field keeps reduced representations, so the
//! limbs are the same too. The stock evaluation relies on this as well: its
//! own grouping follows the thread count.

use ff::Field;
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
    },
    slice::ParallelSliceMut,
};

use super::DIVISION_MIN_LEN;
use crate::{
    poly::{Coeff, Polynomial},
    utils::arithmetic::eval_polynomial_seq,
};

/// Coefficients per evaluation task.
const EVAL_CHUNK: usize = 1 << 13;

/// Independent Horner chains per evaluation task.
const CHAINS: usize = 8;

/// Coefficients per task of an inner product of polynomials.
const SUM_CHUNK: usize = 1 << 12;

/// Rows per block of a division.
const DIV_BLOCK: usize = 1 << 13;

/// The number of calls that divided in parallel blocks (tests only): the
/// golden-digest test at division size checks that its proofs take them.
#[cfg(test)]
pub(crate) mod reached {
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// [`super::divide_by_roots`] divided in parallel blocks.
    pub(crate) static DIVIDE_BY_ROOTS: AtomicUsize = AtomicUsize::new(0);
    /// [`super::kate_division`] divided in parallel blocks.
    pub(crate) static KATE_DIVISION: AtomicUsize = AtomicUsize::new(0);

    pub(super) fn hit(counter: &AtomicUsize) {
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// The counts of `divide_by_roots` and `kate_division`, in that order.
    pub(crate) fn counts() -> [usize; 2] {
        [&DIVIDE_BY_ROOTS, &KATE_DIVISION].map(|c| c.load(Ordering::Relaxed))
    }
}

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
/// polynomials of possibly different lengths, as the stock helper of the KZG
/// multi-open: each coefficient is accumulated from zero over the
/// polynomials in the same order, with the same operations; only the cut of
/// the coefficients into tasks differs.
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

    // A history longer than a block would reach past the block above: the
    // stock division, which takes any number of roots, divides those.
    if lags > DIV_BLOCK {
        return crate::utils::arithmetic::divide_by_roots_seq(poly, roots);
    }
    #[cfg(test)]
    reached::hit(&reached::DIVIDE_BY_ROOTS);

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
/// of `kate_division`: `q_k = a_{k+1} + b · q_{k+1}`); below
/// [`DIVISION_MIN_LEN`], by the stock `kate_division` of `a` itself.
///
/// # Panics
///
/// If `a` is empty, as `kate_division`.
pub(crate) fn kate_division<F: Field>(a: &[F], b: F) -> Vec<F> {
    assert!(!a.is_empty(), "kate_division of an empty polynomial");
    if a.len() < DIVISION_MIN_LEN {
        return crate::utils::arithmetic::kate_division(a, b);
    }
    #[cfg(test)]
    reached::hit(&reached::KATE_DIVISION);
    divide_by_roots(a, &[b])
}

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;

    use ff::{Field, PrimeField, WithSmallOrderMulGroup};
    use midnight_curves::Fq;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use super::*;
    use crate::{
        par::{
            INNER_PRODUCT_MIN_LEN,
            testing::{in_pool, stock_poly_inner_product},
        },
        utils::arithmetic::{divide_by_roots_seq, eval_polynomial, kate_division as stock_kate},
    };

    fn random_poly(rng: &mut ChaCha20Rng, n: usize) -> Vec<Fq> {
        (0..n).map(|_| Fq::random(&mut *rng)).collect()
    }

    /// Every length of a single Horner task (all remainders of the
    /// interleaved chains) and lengths across task boundaries, at special
    /// and random points, with one and many threads.
    #[test]
    fn eval_many_matches_eval_polynomial() {
        let mut rng = ChaCha20Rng::seed_from_u64(0x3a);
        let lens =
            (0..3 * CHAINS + 2).chain([100, EVAL_CHUNK - 1, EVAL_CHUNK, EVAL_CHUNK + 1, 70_001]);
        let polys: Vec<Vec<Fq>> = lens.map(|n| random_poly(&mut rng, n)).collect();
        let mut jobs = Vec::new();
        for p in &polys {
            for x in [Fq::ZERO, Fq::ONE, -Fq::ONE, Fq::random(&mut rng)] {
                jobs.push((p.as_slice(), x));
            }
        }
        let stock: Vec<Fq> = jobs.iter().map(|(p, x)| eval_polynomial(p, *x)).collect();
        for threads in [1, 3, 8] {
            assert_eq!(
                in_pool(threads, || eval_many(&jobs)),
                stock,
                "threads {threads}"
            );
        }
        assert!(eval_many::<Fq>(&[]).is_empty());
    }

    /// Lengths across task boundaries and the floor of the multi-open's
    /// dispatch, with more scalars than polynomials.
    #[test]
    fn poly_inner_product_matches_stock() {
        let mut rng = ChaCha20Rng::seed_from_u64(0x3b);
        for lens in [
            vec![],
            vec![0],
            vec![3, 1],
            vec![SUM_CHUNK + 3, 7, 2 * SUM_CHUNK, 0, 40_000],
            vec![INNER_PRODUCT_MIN_LEN - 1, 3],
            vec![INNER_PRODUCT_MIN_LEN, INNER_PRODUCT_MIN_LEN - 5],
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
            let stock = stock_poly_inner_product(&refs, scalars.clone());
            for threads in [1, 4] {
                let par = in_pool(threads, || poly_inner_product(&refs, scalars.clone()));
                assert_eq!(par.values, stock);
            }
        }
    }

    /// Dense, sparse and repeated roots, a zero root and no root, on lengths
    /// across block boundaries and the floor, with one and many threads, here
    /// and through the crate function that dispatches here.
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
            random_poly(&mut rng, 4),
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
            DIVISION_MIN_LEN - 1,
            DIVISION_MIN_LEN,
            70_000,
        ];
        for roots in &root_sets {
            for &n in &lens {
                let poly = random_poly(&mut rng, n);
                let stock = divide_by_roots_seq(&poly, roots);
                for threads in [1, 4] {
                    let par = in_pool(threads, || divide_by_roots(&poly, roots));
                    assert_eq!(par, stock, "roots {} n {n} threads {threads}", roots.len());
                }
                assert_eq!(
                    crate::utils::arithmetic::divide_by_roots(&poly, roots),
                    stock
                );
            }
        }
    }

    /// More roots than rows in a block (the history of a block would reach
    /// past the block above): the stock division, at any length. The roots
    /// are the `2^13`-th roots of `x^(2^13)` and one more root `y`, so the
    /// product `(X^(2^13) - x^(2^13)) (X - y)` has four terms, the lowest at
    /// lag `DIV_BLOCK + 1`.
    #[test]
    fn divide_by_roots_more_roots_than_a_block() {
        let mut rng = ChaCha20Rng::seed_from_u64(0x3f);
        let x = Fq::random(&mut rng);
        let y = Fq::random(&mut rng);
        let omega = Fq::ROOT_OF_UNITY.pow_vartime([1u64 << (Fq::S - 13)]);
        assert_eq!(DIV_BLOCK, 1 << 13);
        let mut roots: Vec<Fq> =
            std::iter::successors(Some(x), |r| Some(*r * omega)).take(DIV_BLOCK).collect();
        roots.push(y);
        for n in [DIV_BLOCK + 1 + 5, DIVISION_MIN_LEN + 3] {
            let poly = random_poly(&mut rng, n);
            let stock = divide_by_roots_seq(&poly, &roots);
            assert_eq!(stock.len(), n - roots.len());
            assert_eq!(
                in_pool(4, || divide_by_roots(&poly, &roots)),
                stock,
                "n {n}"
            );
            assert_eq!(
                crate::utils::arithmetic::divide_by_roots(&poly, &roots),
                stock
            );
        }
    }

    /// The quotient of the dividend as given equals the stock quotient of
    /// the dividend minus a constant (the call of the stock multi-open), on
    /// both sides of the floor, with one and many threads.
    #[test]
    fn kate_division_matches_stock() {
        let mut rng = ChaCha20Rng::seed_from_u64(0x3d);
        for n in [
            1usize,
            2,
            3,
            DIV_BLOCK,
            DIV_BLOCK + 1,
            2 * DIV_BLOCK + 9,
            DIVISION_MIN_LEN - 1,
            DIVISION_MIN_LEN,
            1 << 17,
        ] {
            let a = random_poly(&mut rng, n);
            for b in [Fq::ZERO, Fq::ONE, Fq::random(&mut rng)] {
                let mut shifted = a.clone();
                shifted[0] -= Fq::random(&mut rng);
                let stock = stock_kate(&shifted, b);
                for threads in [1, 4] {
                    assert_eq!(in_pool(threads, || kate_division(&a, b)), stock, "n {n}");
                }
            }
        }
    }
}
