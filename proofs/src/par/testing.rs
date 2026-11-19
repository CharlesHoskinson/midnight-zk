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

//! The references of the tests of this module: verbatim copies of the stock
//! routines whose call sites dispatch to a parallel routine (so that calling
//! them at their call site would not reach the stock code), and a pool of a
//! given thread count.
//!
//! The stock `divide_by_roots` stays callable as
//! `crate::utils::arithmetic::divide_by_roots_seq` and needs no copy.

use ff::{BatchInvert, Field};

use crate::{
    poly::{Coeff, LagrangeCoeff, Polynomial},
    utils::rational::Rational,
};

/// Runs `f` in a new pool of `threads` threads.
pub(crate) fn in_pool<R: Send>(threads: usize, f: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(f)
}

/// The stock `crate::poly::batch_invert_rational`.
pub(crate) fn stock_batch_invert_rational<F: Field>(
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
        .map(|(poly, inv_denoms)| poly.invert(inv_denoms.into_iter().map(|d| d.unwrap_or(F::ONE))))
        .collect()
}

/// The body of the stock `Polynomial::to_delta`.
pub(crate) fn stock_to_delta<F: Field>(values: &[F]) -> Vec<F> {
    let n = values.len();
    let mut out = Vec::with_capacity(n);
    if n >= 1 {
        out.push(values[0]);
    }
    for i in 1..n {
        out.push(values[i] - values[i - 1]);
    }
    out
}

/// The body of the stock `Polynomial::to_double_delta`.
pub(crate) fn stock_to_double_delta<F: Field>(values: &[F]) -> Vec<F> {
    let n = values.len();
    let mut out = Vec::with_capacity(n);
    if n >= 1 {
        out.push(values[0]);
    }
    if n >= 2 {
        out.push(values[1] - values[0].double());
    }
    for i in 2..n {
        out.push(values[i] - values[i - 1].double() + values[i - 2]);
    }
    out
}

/// The body of the stock `poly_inner_product` helper of the KZG multi-open.
pub(crate) fn stock_poly_inner_product<F: Field>(
    polys: &[&Polynomial<F, Coeff>],
    scalars: impl IntoIterator<Item = F>,
) -> Vec<F> {
    let scalars: Vec<F> = scalars.into_iter().take(polys.len()).collect();
    let max_len = polys.iter().map(|p| p.len()).max().unwrap_or(0);
    let mut values = vec![F::ZERO; max_len];
    crate::utils::arithmetic::parallelize(&mut values, |chunk, start| {
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
    values
}

/// The running product of the stock permutation prover: `z_0 = 1`,
/// `z_r = z_{r-1} · m_{r-1}` for `r < n`.
pub(crate) fn stock_prefix_product<F: Field>(m: &[F], n: usize) -> Vec<F> {
    let mut z = Vec::with_capacity(n);
    z.push(F::ONE);
    for row in 1..n {
        let tmp = z[row - 1] * &m[row - 1];
        z.push(tmp);
    }
    z
}
