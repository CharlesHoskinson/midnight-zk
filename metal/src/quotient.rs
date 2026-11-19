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

//! The quotient on the device (not written yet: every call declines with
//! `not-implemented`).
//!
//! [`crate::quotient`] has checked that `job` is not null and that a device
//! is usable before it calls [`quotient`]. Everything else (the ABI version
//! and layout of `midnight_proofs::gpu::abi::QuotientJobAbi`, the size
//! against `min_k`, reduction of the job's scalars, the memory need) is this
//! module's. It ends with exactly one of `account::served` or
//! `account::decline`, and writes `out` only with the complete result.

use std::ffi::c_void;

use crate::{
    Scalar,
    account::{self, Class, Reason},
};

/// The first `out.len()` coefficients of the quotient of `job`. `false`:
/// declined, `out` unchanged.
///
/// # Safety
/// `job` points to a valid `QuotientJobAbi` whose pointers live for the call.
pub(crate) unsafe fn quotient(job: *const c_void, out: &mut [Scalar]) -> bool {
    let _ = (job, out);
    account::decline(Class::Quotient, Reason::NotImplemented)
}
