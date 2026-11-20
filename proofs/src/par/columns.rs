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

//! Column-wise passes over whole polynomials. Each output element depends
//! on the inputs only, so the chunking does not change any value.

use ff::{BatchInverter, Field};
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};

use super::{PARALLELIZE_CHUNKS_PER_THREAD, PARALLELIZE_MIN_CHUNK};
use crate::{
    poly::{LagrangeCoeff, Polynomial},
    utils::rational::Rational,
};

/// Rows per task.
pub(super) const CHUNK: usize = 1 << 12;

/// Elements per task of [`copy_from_slice`].
#[cfg(feature = "gpu")]
const COPY_CHUNK: usize = 1 << 14;

/// `crate::utils::arithmetic::parallelize` with
/// [`PARALLELIZE_CHUNKS_PER_THREAD`] chunks per thread instead of one. The
/// stock cut already varies with the thread count, so callbacks do not depend
/// on it.
///
/// Returns `false` without calling `f` when the pool has one thread or the
/// chunks would be shorter than [`PARALLELIZE_MIN_CHUNK`].
pub(crate) fn parallelize<T: Send, F: Fn(&mut [T], usize) + Sync>(v: &mut [T], f: &F) -> bool {
    let chunks = rayon::current_num_threads() * PARALLELIZE_CHUNKS_PER_THREAD;
    if chunks <= PARALLELIZE_CHUNKS_PER_THREAD || v.len() < chunks * PARALLELIZE_MIN_CHUNK {
        return false;
    }
    let chunk = v.len().div_ceil(chunks);
    v.par_chunks_mut(chunk).enumerate().for_each(|(i, part)| f(part, i * chunk));
    true
}

/// `crate::poly::batch_invert_rational`, batching each chunk of [`CHUNK`]
/// rows separately. Inverses are unique and a zero denominator stays zero (as
/// in `ff::BatchInvert`), so the values equal the stock ones.
pub(crate) fn batch_invert_rational<F: Field>(
    assigned: Vec<Polynomial<Rational<F>, LagrangeCoeff>>,
) -> Vec<Polynomial<F, LagrangeCoeff>> {
    assigned
        .par_iter()
        .map(|poly| {
            let mut values: Vec<F> = poly.values.par_iter().map(Rational::numerator).collect();
            values.par_chunks_mut(CHUNK).zip(poly.values.par_chunks(CHUNK)).for_each_init(
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

/// `Polynomial::to_delta`, in parallel chunks.
pub(crate) fn to_delta<F: Field>(a: &[F]) -> Vec<F> {
    let mut out: Vec<F> = a.par_iter().map(|_| F::ZERO).collect();
    out.par_chunks_mut(CHUNK).enumerate().for_each(|(c, chunk)| {
        let start = c * CHUNK;
        for (j, out) in chunk.iter_mut().enumerate() {
            let i = start + j;
            *out = if i == 0 { a[0] } else { a[i] - a[i - 1] };
        }
    });
    out
}

/// `Polynomial::to_double_delta`, in parallel chunks.
pub(crate) fn to_double_delta<F: Field>(a: &[F]) -> Vec<F> {
    let mut out: Vec<F> = a.par_iter().map(|_| F::ZERO).collect();
    out.par_chunks_mut(CHUNK).enumerate().for_each(|(c, chunk)| {
        let start = c * CHUNK;
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

/// `dst.copy_from_slice(src)` in parallel chunks.
#[cfg(feature = "gpu")]
pub(crate) fn copy_from_slice<T: Copy + Send + Sync>(dst: &mut [T], src: &[T]) {
    assert_eq!(
        dst.len(),
        src.len(),
        "copy between slices of different lengths"
    );
    dst.par_chunks_mut(COPY_CHUNK)
        .zip(src.par_chunks(COPY_CHUNK))
        .for_each(|(dst, src)| dst.copy_from_slice(src));
}

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;

    use ff::Field;
    use midnight_curves::Fq;
    use rand_chacha::ChaCha20Rng;
    use rand_core::{RngCore, SeedableRng};

    use super::*;
    use crate::{
        par::{
            DELTA_MIN_LEN,
            testing::{
                in_pool, stock_batch_invert_rational, stock_to_delta, stock_to_double_delta,
            },
        },
        poly::batch_invert_rational as dispatched_batch_invert_rational,
    };

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

    #[test]
    fn batch_invert_rational_matches_stock() {
        let mut rng = ChaCha20Rng::seed_from_u64(0x1a);
        // `mix` 2: no denominators at all
        for (cols, n, mix) in [
            (0usize, 0usize, 7u32),
            (3, 17, 7),
            (4, CHUNK + 3, 7),
            (3, 3 * CHUNK + 5, 2),
        ] {
            let assigned: Vec<Polynomial<Rational<Fq>, LagrangeCoeff>> = (0..cols)
                .map(|_| Polynomial {
                    values: (0..n).map(|_| random_rational(&mut rng, mix)).collect(),
                    _marker: PhantomData,
                })
                .collect();
            let stock = stock_batch_invert_rational(assigned.clone());
            for threads in [1, 8] {
                let par = in_pool(threads, || batch_invert_rational(assigned.clone()));
                assert_eq!(par.len(), stock.len());
                for (p, s) in par.iter().zip(&stock) {
                    assert_eq!(p.values, s.values, "cols {cols} n {n} threads {threads}");
                }
            }
            let dispatched = dispatched_batch_invert_rational(assigned);
            for (p, s) in dispatched.iter().zip(&stock) {
                assert_eq!(p.values, s.values);
            }
        }
    }

    #[test]
    fn parallelize_passes_every_element_once_with_its_offset() {
        for threads in [1usize, 7] {
            let floor = threads * PARALLELIZE_CHUNKS_PER_THREAD * PARALLELIZE_MIN_CHUNK;
            for n in [0usize, floor - 1, floor, 300_017] {
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
        for n in [0usize, 1, 2, 3, CHUNK + 1, DELTA_MIN_LEN - 1, DELTA_MIN_LEN] {
            let values: Vec<Fq> = (0..n).map(|_| Fq::random(&mut rng)).collect();
            let delta = stock_to_delta(&values);
            let double = stock_to_double_delta(&values);
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
