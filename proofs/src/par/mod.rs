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

//! Parallel CPU paths of the prover (feature `prover-par`).
//!
//! Each function here computes exactly the values of the stock function it
//! replaces at its call site, bit for bit, whatever the thread count: no
//! random draw and no transcript operation happens in this module, so the
//! proof bytes do not depend on the feature.
#![allow(dead_code)]

pub(crate) mod columns;
pub(crate) mod invert;
pub(crate) mod layout;
pub(crate) mod open;
pub(crate) mod product;
#[cfg(test)]
mod testing;

// The size floors of the routines of this module. Below a floor the call
// site keeps the stock routine: there the cost of cutting the work into
// tasks and joining them exceeds what the threads save.

/// Shorter columns are delta-encoded by the stock `Polynomial::to_delta` and
/// `Polynomial::to_double_delta` ([`columns::to_delta`],
/// [`columns::to_double_delta`]). No timing was recorded for this floor; at
/// it a column makes four tasks of [`columns::CHUNK`] rows.
pub(crate) const DELTA_MIN_LEN: usize = 1 << 14;

/// [`columns::parallelize`] cuts a slice into this many chunks per thread
/// of the pool, so that a thread that finishes early takes over more of
/// them.
pub(crate) const PARALLELIZE_CHUNKS_PER_THREAD: usize = 4;

/// [`columns::parallelize`] leaves a slice to the stock cut (one chunk per
/// thread) when its chunks would be shorter than this. No timing was
/// recorded for this floor.
pub(crate) const PARALLELIZE_MIN_CHUNK: usize = 1 << 10;

/// Shorter polynomials are divided by the stock `divide_by_roots` and
/// `kate_division` of `crate::utils::arithmetic` ([`open::divide_by_roots`],
/// [`open::kate_division`]).
///
/// Measured on an Apple M4 Max (16 cores), release build, 16 threads, random
/// polynomials, minimum of 30 calls (5 from `2^18`), with other processes
/// running (load average 4 to 7). Wall time (ms) of the stock function and
/// of the parallel one, by length:
///
/// | length | `divide_by_roots`, 2 roots | 1 root      | `kate_division` |
/// |--------|----------------------------|-------------|-----------------|
/// | 2^13   | 0.25 / 0.37                | 0.19 / 0.33 | 0.15 / 0.36     |
/// | 2^14   | 0.49 / 0.54                | 0.37 / 0.42 | 0.30 / 0.46     |
/// | 2^15   | 0.99 / 0.58                | 0.75 / 0.52 | 0.61 / 0.53     |
/// | 2^16   | 1.97 / 0.59                | 1.48 / 0.55 | 1.21 / 0.57     |
/// | 2^20   | 31.8 / 4.39                | 23.9 / 3.32 | 19.5 / 3.22     |
///
/// Up to `2^14` (two blocks) the scan that joins the blocks costs about as
/// much as the stock loop. The stock `kate_division` timed here is the bare
/// division; the stock multi-open also evaluates the dividend and copies it,
/// which [`open::kate_division`] does not need.
pub(crate) const DIVISION_MIN_LEN: usize = 1 << 15;

/// Shorter inner products of polynomials are computed by the stock helper of
/// the KZG multi-open ([`open::poly_inner_product`]).
///
/// Measured as [`DIVISION_MIN_LEN`], for the inner product of 5 polynomials
/// and for 4 groups of 6 polynomials computed at the same time (as the
/// multi-open does for its point sets). Wall time (ms) of the stock helper
/// and of the parallel one, by length:
///
/// | length | 5 polynomials | 4 groups of 6 |
/// |--------|---------------|---------------|
/// | 2^13   | 0.11 / 0.33   | 0.22 / 0.45   |
/// | 2^15   | 0.28 / 0.40   | 0.71 / 0.89   |
/// | 2^17   | 0.74 / 0.82   | 2.61 / 2.83   |
/// | 2^18   | 1.40 / 1.41   | 5.33 / 5.76   |
/// | 2^19   | 2.73 / 2.59   | 10.5 / 11.2   |
/// | 2^20   | 4.86 / 4.54   | 20.5 / 21.7   |
///
/// The parallel helper builds its vector of zeros in parallel and cuts the
/// sums into tasks shorter than the stock helper's one chunk per thread;
/// below `2^19` that costs more than the balance gains.
pub(crate) const INNER_PRODUCT_MIN_LEN: usize = 1 << 19;
