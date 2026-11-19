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

//! The running product of the permutation argument, in parallel blocks.
//!
//! The stock loop computes `z_0 = 1`, `z_r = z_{r-1} · m_{r-1}` in one
//! sequential pass. Here the rows are cut into blocks of [`BLOCK`] rows: the
//! product of each block's factors is computed in parallel, a short
//! sequential scan turns those into the value of `z` at the start of every
//! block, and each block then runs the stock recurrence from that value, in
//! parallel. Every `z_r` is the product `m_0 · … · m_{r-1}` either way.
//! Multiplication in a prime field is exact, associative and commutative,
//! so the regrouped product is the same field element, and the field keeps
//! reduced representations, so the limbs are the same too. There is no size
//! floor: up to one block, the routine is the stock loop in a single task.

use ff::Field;
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

/// Rows per block.
const BLOCK: usize = 1 << 13;

/// The `n` values `z_0 = 1`, `z_r = z_{r-1} · m_{r-1}` (`m` holds at least
/// `n - 1` factors; `n >= 1`).
pub(crate) fn prefix_product_from_one<F: Field>(m: &[F], n: usize) -> Vec<F> {
    assert!(n >= 1 && m.len() >= n - 1);
    let blocks = n.div_ceil(BLOCK);
    // The product of the factors that lead from the first row of block `b`
    // to the first row of block `b + 1` (none for the last block).
    let block_products: Vec<F> = (0..blocks.saturating_sub(1))
        .into_par_iter()
        .map(|b| m[b * BLOCK..(b + 1) * BLOCK].iter().fold(F::ONE, |acc, f| acc * f))
        .collect();
    let mut starts = Vec::with_capacity(blocks);
    starts.push(F::ONE);
    for p in &block_products {
        let next = *starts.last().unwrap() * p;
        starts.push(next);
    }

    let mut z: Vec<F> = (0..n).into_par_iter().map(|_| F::ZERO).collect();
    z.par_chunks_mut(BLOCK).zip(starts.into_par_iter()).enumerate().for_each(
        |(b, (block, start))| {
            let first = b * BLOCK;
            block[0] = start;
            for j in 1..block.len() {
                block[j] = block[j - 1] * m[first + j - 1];
            }
        },
    );
    z
}

#[cfg(test)]
mod tests {
    use ff::Field;
    use midnight_curves::Fq;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use super::*;
    use crate::par::testing::{in_pool, stock_prefix_product};

    /// Lengths across block boundaries, with factors that include zeros and
    /// ones (as a permutation row can give) and with nonzero factors only,
    /// with one and many threads.
    #[test]
    fn prefix_product_matches_stock() {
        let mut rng = ChaCha20Rng::seed_from_u64(0x2a);
        for n in [1usize, 2, 3, BLOCK - 1, BLOCK, BLOCK + 1, 3 * BLOCK + 7] {
            let mut m: Vec<Fq> = (0..n).map(|_| Fq::random(&mut rng)).collect();
            for i in (0..n).step_by(997) {
                m[i] = if i % 2 == 0 { Fq::ZERO } else { Fq::ONE };
            }
            for case in [m.clone(), m.iter().map(|x| *x + Fq::ONE).collect()] {
                let stock = stock_prefix_product(&case, n);
                for threads in [1, 4] {
                    let par = in_pool(threads, || prefix_product_from_one(&case, n));
                    assert_eq!(par, stock, "n {n} threads {threads}");
                }
            }
        }
    }
}
