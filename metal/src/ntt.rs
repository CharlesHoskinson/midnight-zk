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

//! The scalar-field NTT on the device (not written yet: every call declines
//! with `not-implemented`).
//!
//! [`crate::fft`] / [`crate::ifft`] have admitted the call (`a.len() ==
//! 2^log_n`, `1 <= log_n <= MAX_LOG_N`, `log_n >= min_k`, a usable device,
//! `a`, `omega` and the divisor reduced) before they call these functions.
//! Which roots are taken is this module's choice (decline `shape` for the
//! others). Each function ends with exactly one of `account::served` or
//! `account::decline`, and writes `a` only with the complete result.

use crate::{
    Scalar,
    account::{self, Class, Reason},
};

/// The FFT of `a` at `omega`, natural order, in place. `false`: declined, `a`
/// unchanged.
pub(crate) fn fft(a: &mut [Scalar], omega: Scalar, log_n: u32) -> bool {
    let _ = (a, omega, log_n);
    account::decline(Class::Fft, Reason::NotImplemented)
}

/// The FFT of `a` at `omega_inv`, then each element times `divisor`, in
/// place. `false`: declined, `a` unchanged.
pub(crate) fn ifft(a: &mut [Scalar], omega_inv: Scalar, log_n: u32, divisor: Scalar) -> bool {
    let _ = (a, omega_inv, log_n, divisor);
    account::decline(Class::Ifft, Reason::NotImplemented)
}
