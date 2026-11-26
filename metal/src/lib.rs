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

//! Metal backend for the accelerator hooks of `midnight-proofs` (feature
//! `gpu`) on Apple GPUs: BLS12-381 G1 MSM, scalar-field NTT and the quotient.
//! It has the same entry points as the CUDA crate `midnight-gpu` and does not
//! depend on `midnight-proofs`: its functions match the hook fields of
//! `midnight_proofs::gpu::GpuHooks`.
//!
//! The host side is `midnight-accel`, generic over a GPU backend; this crate
//! is its Metal backend ([`ctx::Metal`]) and the entry points for it.
//!
//! ```ignore
//! midnight_proofs::gpu::install(midnight_proofs::gpu::GpuHooks {
//!     msm_g1: Some(midnight_metal::msm_g1),
//!     fft: Some(midnight_metal::fft),
//!     ifft: Some(midnight_metal::ifft),
//!     quotient: Some(midnight_metal::quotient),
//! });
//! midnight_metal::begin_request();
//! let proof = prove(/* ... */);
//! assert!(midnight_metal::end_request());
//! ```
//!
//! # Hook contract
//!
//! - Every entry point returns the exact value of the stock CPU code, or
//!   declines (`None` / `false`) and the caller runs the CPU code.
//! - A decline leaves the caller's buffer unchanged: a result is written only
//!   after the last command buffer completed without error.
//! - No entry point draws randomness or touches a transcript. Any pseudo-random
//!   value (the MSM mask) is a fixed function of the shape.
//! - Every failure, including a panic caught at the entry point, is a counted
//!   decline ([`declines`]), never a panic into the prover.
//!
//! # Requests
//!
//! Per-request device buffers can hold witness values. Between
//! [`begin_request`] and [`end_request`] they are reused across hook calls;
//! [`end_request`] zeroes them. Without an open request every hook call zeroes
//! them before it returns. Large buffers come from a pool of zeroed buffers
//! because a new buffer stalls its first command buffer while its pages are
//! mapped for the GPU.
//!
//! # Thresholds
//!
//! A call below its class's threshold ([`thresholds`]) declines `below-min-k`.
//! The defaults ([`DEFAULT_THRESHOLDS`]) were measured on whole proofs, not
//! isolated kernels, because the GPU clocks down while the CPU works between
//! device calls. By default only MSMs on a registered SRS table
//! ([`register_srs`]) go to the device: other bases need a per-call G1 check
//! that costs more than the CPU MSM. An explicit threshold ([`set_min_k`])
//! replaces every default and opens that one-shot path.
//!
//! # Environment
//!
//! - `MIDNIGHT_METAL=off`: use no device.
//! - `MIDNIGHT_METAL_MIN_K`: one threshold for every class.
//! - `MIDNIGHT_METAL_MSM_CPU_SHARE=off`: run every MSM on the device only.
//! - `MIDNIGHT_METAL_POOL_MIB`: pool size limit (default a quarter of
//!   `recommendedMaxWorkingSetSize`).
//! - `MIDNIGHT_METAL_QUOTIENT_THREADS`: see [`quotient_threads`].
//! - `MIDNIGHT_METAL_KEEP_KEYS=1`: see [`keep_resident`].
//! - `MIDNIGHT_METAL_SHADER_SOURCE=1`: compile the embedded shader source
//!   instead of the metallib ([`library_source`]); also read by `build.rs`.

#![deny(missing_docs)]
#![deny(missing_debug_implementations)]

#[cfg(target_os = "macos")]
pub mod ctx;
/// The Metal backend of `midnight-accel`, for code generic over the backend.
#[cfg(target_os = "macos")]
pub use ctx::Metal;
#[cfg(not(target_os = "macos"))]
mod stub;
/// The Metal backend of `midnight-accel`, for code generic over the backend:
/// a type without values on this target.
#[cfg(not(target_os = "macos"))]
pub use stub::Metal;
mod tuning;

midnight_accel::export_backend!(Metal, "MIDNIGHT_METAL");

/// Per-class break-evens against the stock CPU code, measured on Apple
/// silicon. The one-shot MSM never breaks even because of its G1 check.
pub const DEFAULT_THRESHOLDS: Thresholds = tuning::TUNING.thresholds;

/// The Metal device as its properties describe it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetalInfo {
    /// Device name (`MTLDevice.name`).
    pub name: String,
    /// `MTLDevice.registryID`.
    pub registry_id: u64,
    /// `hasUnifiedMemory`.
    pub unified_memory: bool,
    /// `recommendedMaxWorkingSetSize` in bytes.
    pub recommended_max_working_set: u64,
    /// `maxBufferLength`: the largest single buffer in bytes.
    pub max_buffer_length: u64,
    /// `maxThreadsPerThreadgroup.width`.
    pub max_threads_per_threadgroup: u64,
    /// The highest `MTLGPUFamilyAppleN` supported (0: none).
    pub apple_family: u32,
    /// Supports `MTLGPUFamilyMetal3`.
    pub metal3: bool,
    /// The Metal Shading Language version of the shaders.
    pub msl_version: &'static str,
}

/// Whether the Metal code is compiled in (a macOS target).
pub const fn metal_built() -> bool {
    cfg!(target_os = "macos")
}

/// The Metal device, without loading the shader library. `None` on other
/// targets, with no device, or with `MIDNIGHT_METAL=off`.
pub fn present() -> Option<&'static MetalInfo> {
    midnight_accel::present::<Metal>()?;
    #[cfg(target_os = "macos")]
    {
        ctx::probe()
    }
    #[cfg(not(target_os = "macos"))]
    None
}

/// Where the shader library came from: `metallib`, or `source ...` with the
/// reason. `None` before the context is made or when it failed.
pub fn library_source() -> Option<&'static str> {
    #[cfg(target_os = "macos")]
    {
        ctx::context().ok().map(|c| c.library_source())
    }
    #[cfg(not(target_os = "macos"))]
    None
}
