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

// Diagnostic kernels of the device context (ctx.rs).

#include <metal_stdlib>

// A dependent integer chain of `iters` steps per thread, to hold the GPU busy
// for a chosen time (the command-buffer duration experiment).
kernel void diag_spin(device uint *out [[buffer(0)]],
                      constant uint &iters [[buffer(1)]],
                      constant uint &n [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    uint x = gid * 2654435761u + 1u;
    for (uint i = 0; i < iters; i++) x = x * 1664525u + 1013904223u;
    out[gid] = x;
}

// out[i] = i: a smoke test of dispatch and read-back.
kernel void diag_iota(device uint *out [[buffer(0)]],
                      constant uint &n [[buffer(1)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    out[gid] = gid;
}
