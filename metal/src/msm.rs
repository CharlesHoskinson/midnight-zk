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

//! The G1 MSM on the device (not written yet: every call declines with
//! `not-implemented`).
//!
//! [`crate::msm_g1`] has admitted the call (equal non-zero lengths up to
//! `2^MAX_MSM_LOG`, at least `2^min_k`, a usable device, every scalar
//! reduced) before it calls [`msm_g1`]. This function ends with exactly one
//! of `account::served` or `account::decline`.

use crate::{
    G1AffineLimbs, G1JacobianLimbs, Scalar,
    account::{self, Class, Reason},
};

/// `sum scalars[i] * bases[i]`, or `None` (declined).
pub(crate) fn msm_g1(scalars: &[Scalar], bases: &[G1AffineLimbs]) -> Option<G1JacobianLimbs> {
    let _ = (scalars, bases);
    account::decline(Class::Msm, Reason::NotImplemented);
    None
}
