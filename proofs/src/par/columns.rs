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

//! Column-wise passes over whole polynomials: the inversion of assigned
//! rational columns and the delta encodings of Lagrange columns.
//!
//! Every output element is computed from the inputs alone, independently of
//! the other output elements, so splitting the columns into chunks across
//! threads changes no value.

use ff::{BatchInverter, Field};
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};

use crate::{
    poly::{LagrangeCoeff, Polynomial},
    utils::rational::Rational,
};

/// Rows per parallel task.
const ROWS: usize = 1 << 12;

/// Shorter columns are delta-encoded by the stock code.
pub(crate) const PAR_MIN_LEN: usize = 1 << 14;

/// Chunks per thread of [`parallelize`].
const CHUNKS_PER_THREAD: usize = 4;

/// The shortest chunk of [`parallelize`].
const MIN_CHUNK: usize = 1 << 10;

/// `crate::utils::arithmetic::parallelize` on a long slice: calls `f` on
/// consecutive chunks of `v` with the offset of each chunk, as the stock
/// function does, but with [`CHUNKS_PER_THREAD`] chunks per thread of the
/// pool instead of one, so that a thread that finishes early (a faster
/// core, or one not shared with another process) takes over more chunks.
/// Every element is passed to `f` exactly once, with its offset, so a
/// callback that is correct for the stock cut, which varies with the
/// thread count, gives the same values. Returns `false`, having done
/// nothing, when `v` is too short to cut into chunks of [`MIN_CHUNK`]
/// elements (the stock cut is used then).
pub(crate) fn parallelize<T: Send, F: Fn(&mut [T], usize) + Sync>(v: &mut [T], f: &F) -> bool {
    let chunks = rayon::current_num_threads() * CHUNKS_PER_THREAD;
    if chunks <= CHUNKS_PER_THREAD || v.len() < chunks * MIN_CHUNK {
        return false;
    }
    let chunk = v.len().div_ceil(chunks);
    v.par_chunks_mut(chunk).enumerate().for_each(|(i, part)| f(part, i * chunk));
    true
}

/// `crate::poly::batch_invert_rational`, over the BLS12-381 scalar field
/// (the caller checks the field; see `crate::par::invert`).
///
/// Stock inverts the denominators of all columns in one batch and then
/// multiplies each numerator by the inverse of its denominator, or by one
/// when the value has no denominator. Here each chunk of `ROWS` rows of a
/// column inverts its own denominators in one batch: an inverse is unique
/// and a zero denominator stays zero (as in `ff::BatchInvert`), so every
/// inverse is the stock one. A value without a denominator keeps its
/// numerator, which is the stock product `numerator * 1` (multiplication by
/// one is exact in a prime field and the field keeps reduced
/// representations, so the limbs are the same too).
pub(crate) fn batch_invert_rational<F: Field>(
    assigned: Vec<Polynomial<Rational<F>, LagrangeCoeff>>,
) -> Vec<Polynomial<F, LagrangeCoeff>> {
    assigned
        .par_iter()
        .map(|poly| {
            let mut values: Vec<F> = poly.values.par_iter().map(Rational::numerator).collect();
            values.par_chunks_mut(ROWS).zip(poly.values.par_chunks(ROWS)).for_each_init(
                Vec::new,
                |denoms: &mut Vec<F>, (out, rationals)| {
                    denoms.clear();
                    denoms.extend(rationals.iter().filter_map(Rational::denominator));
                    if denoms.is_empty() {
                        return;
                    }
                    let mut scratch = vec![F::ZERO; denoms.len()];
                    BatchInverter::invert_with_external_scratch(denoms, &mut scratch);
                    let mut inverses = denoms.iter();
                    for (out, rational) in out.iter_mut().zip(rationals) {
                        if rational.denominator().is_some() {
                            *out = rational.numerator() * inverses.next().unwrap();
                        }
                    }
                },
            );
            Polynomial {
                values,
                _marker: poly._marker,
            }
        })
        .collect()
}

/// `Polynomial::to_delta`, in parallel chunks: element `i` is the stock
/// expression `a_i - a_{i-1}` (and `a_0` first).
pub(crate) fn to_delta<F: Field>(a: &[F]) -> Vec<F> {
    let mut out: Vec<F> = a.par_iter().map(|_| F::ZERO).collect();
    out.par_chunks_mut(ROWS).enumerate().for_each(|(c, chunk)| {
        let start = c * ROWS;
        for (j, out) in chunk.iter_mut().enumerate() {
            let i = start + j;
            *out = if i == 0 { a[0] } else { a[i] - a[i - 1] };
        }
    });
    out
}

/// `Polynomial::to_double_delta`, in parallel chunks: element `i` is the
/// stock expression `a_i - 2 a_{i-1} + a_{i-2}` (and `a_0`, `a_1 - 2 a_0`
/// first), evaluated in the stock order.
pub(crate) fn to_double_delta<F: Field>(a: &[F]) -> Vec<F> {
    let mut out: Vec<F> = a.par_iter().map(|_| F::ZERO).collect();
    out.par_chunks_mut(ROWS).enumerate().for_each(|(c, chunk)| {
        let start = c * ROWS;
        for (j, out) in chunk.iter_mut().enumerate() {
            let i = start + j;
            *out = match i {
                0 => a[0],
                1 => a[1] - a[0].double(),
                _ => a[i] - a[i - 1].double() + a[i - 2],
            };
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;

    use ff::{BatchInvert, Field};
    use midnight_curves::Fq;
    use rand_chacha::ChaCha20Rng;
    use rand_core::{RngCore, SeedableRng};

    use super::*;

    /// The stock code of `crate::poly::batch_invert_rational`, verbatim.
    fn stock_batch_invert_rational<F: Field>(
        assigned: Vec<Polynomial<Rational<F>, LagrangeCoeff>>,
    ) -> Vec<Polynomial<F, LagrangeCoeff>> {
        let mut assigned_denominators: Vec<_> = assigned
            .iter()
            .map(|f| f.iter().map(|value| value.denominator()).collect::<Vec<_>>())
            .collect();
        assigned_denominators
            .iter_mut()
            .flat_map(|f| f.iter_mut().filter_map(|d| d.as_mut()))
            .batch_invert();
        assigned
            .iter()
            .zip(assigned_denominators)
            .map(|(poly, inv_denoms)| {
                poly.invert(inv_denoms.into_iter().map(|d| d.unwrap_or(F::ONE)))
            })
            .collect()
    }

    fn random_rational(rng: &mut ChaCha20Rng, mix: u32) -> Rational<Fq> {
        match rng.next_u32() % mix {
            0 => Rational::Zero,
            1 => Rational::Trivial(Fq::random(&mut *rng)),
            2 => Rational::Rational(Fq::random(&mut *rng), Fq::random(&mut *rng)),
            3 => Rational::Rational(Fq::random(&mut *rng), Fq::ZERO),
            4 => Rational::Rational(Fq::ZERO, Fq::random(&mut *rng)),
            5 => Rational::Rational(Fq::random(&mut *rng), Fq::ONE),
            _ => Rational::Trivial(Fq::ZERO),
        }
    }

    fn in_pool<R: Send>(threads: usize, f: impl FnOnce() -> R + Send) -> R {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(f)
    }

    #[test]
    fn batch_invert_rational_matches_stock() {
        let mut rng = ChaCha20Rng::seed_from_u64(0x1a);
        for (cols, n, mix) in [
            (0usize, 0usize, 7u32),
            (1, 1, 7),
            (3, 17, 7),
            (2, ROWS - 1, 7),
            (4, ROWS + 3, 7),
            (3, 3 * ROWS + 5, 2),
            (2, 2 * ROWS, 1),
            (5, 20_000, 7),
        ] {
            let assigned: Vec<Polynomial<Rational<Fq>, LagrangeCoeff>> = (0..cols)
                .map(|_| Polynomial {
                    values: (0..n).map(|_| random_rational(&mut rng, mix)).collect(),
                    _marker: PhantomData,
                })
                .collect();
            let stock = stock_batch_invert_rational(assigned.clone());
            for threads in [1, 3, 8] {
                let par = in_pool(threads, || batch_invert_rational(assigned.clone()));
                assert_eq!(par.len(), stock.len());
                for (p, s) in par.iter().zip(&stock) {
                    assert_eq!(p.values, s.values, "cols {cols} n {n} threads {threads}");
                }
            }
            // and through the crate function, which dispatches here
            let dispatched = crate::poly::batch_invert_rational(assigned);
            for (p, s) in dispatched.iter().zip(&stock) {
                assert_eq!(p.values, s.values);
            }
        }
    }

    #[test]
    fn parallelize_passes_every_element_once_with_its_offset() {
        for threads in [1usize, 2, 7, 16] {
            for n in [
                0usize,
                1,
                100,
                threads * CHUNKS_PER_THREAD * MIN_CHUNK - 1,
                threads * CHUNKS_PER_THREAD * MIN_CHUNK,
                300_017,
            ] {
                let mut v = vec![(0usize, 0u32); n];
                let took = in_pool(threads, || {
                    parallelize(&mut v, &|part: &mut [(usize, u32)], start: usize| {
                        for (i, x) in part.iter_mut().enumerate() {
                            *x = (start + i, x.1 + 1);
                        }
                    })
                });
                if took {
                    assert!(v.iter().enumerate().all(|(i, x)| *x == (i, 1)), "n {n}");
                } else {
                    assert!(v.iter().all(|x| x.1 == 0));
                }
                // through the crate function, at every length
                let mut w = vec![(0usize, 0u32); n];
                in_pool(threads, || {
                    crate::utils::arithmetic::parallelize(&mut w, |part, start| {
                        for (i, x) in part.iter_mut().enumerate() {
                            *x = (start + i, x.1 + 1);
                        }
                    })
                });
                assert!(w.iter().enumerate().all(|(i, x)| *x == (i, 1)), "n {n}");
            }
        }
    }

    #[test]
    fn delta_encodings_match_stock() {
        let mut rng = ChaCha20Rng::seed_from_u64(0x1b);
        for n in [
            0usize,
            1,
            2,
            3,
            ROWS - 1,
            ROWS,
            ROWS + 1,
            2 * ROWS + 2,
            50_001,
        ] {
            let values: Vec<Fq> = (0..n).map(|_| Fq::random(&mut rng)).collect();
            // the stock bodies, verbatim
            let mut delta = Vec::with_capacity(n);
            if n >= 1 {
                delta.push(values[0]);
            }
            for i in 1..n {
                delta.push(values[i] - values[i - 1]);
            }
            let mut double = Vec::with_capacity(n);
            if n >= 1 {
                double.push(values[0]);
            }
            if n >= 2 {
                double.push(values[1] - values[0].double());
            }
            for i in 2..n {
                double.push(values[i] - values[i - 1].double() + values[i - 2]);
            }
            for threads in [1, 5] {
                assert_eq!(in_pool(threads, || to_delta(&values)), delta, "n {n}");
                assert_eq!(
                    in_pool(threads, || to_double_delta(&values)),
                    double,
                    "n {n}"
                );
            }
            let poly = Polynomial::<Fq, LagrangeCoeff> {
                values,
                _marker: PhantomData,
            };
            assert_eq!(poly.to_delta().values, delta);
            assert_eq!(poly.to_double_delta().values, double);
        }
    }
}
