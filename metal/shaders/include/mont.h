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

// Montgomery arithmetic over a prime field of N 32-bit limbs.
//
// An element is `Elem<P>`: N little-endian 32-bit limbs of the Montgomery form
// `a * 2^(32 N) mod p`, reduced (below p). On a little-endian device this is
// bit-identical to the host's N/2 little-endian 64-bit limbs, so a buffer of
// blst limbs is read as it is.
//
// A parameter struct P provides:
//   enum : int  { N = <32-bit limbs> };
//   enum : uint { INV = -p^-1 mod 2^32 };
//   static ulong inv64();          // -p^-1 mod 2^64
//   static uint mod(int i);        // limb i of p
//   static uint one(int i);        // limb i of 2^(32 N) mod p (Montgomery one)
// The top limb of p must be below 2^31 - 1 (the "no-carry" condition of the
// CIOS multiplication; both BLS12-381 fields meet it), so that `a + b < 2p`
// fits in N limbs and the products below never carry out of the top limb.
//
// Multiplication strategies (all exact; the benchmark in the crate's tests
// measures them, and MONT_MUL_DEFAULT names the fastest on Apple M4 Max):
//   mul_cios32  : CIOS over 32-bit limbs, 32x32->64 products and 64-bit
//                 (`ulong`) accumulators.
//   mul_mulhi32 : CIOS over 32-bit limbs, products as `a * b` and
//                 `mulhi(a, b)` with explicit 32-bit carries (no `ulong`).
//   mul_fips32  : product scanning (FIPS): column sums of the product and of
//                 the reduction in a 96-bit accumulator (`ulong` + `uint`).
//   mul_cios64  : CIOS over 64-bit limbs, products as `a * b` and
//                 `mulhi(a, b)` on `ulong` (MSL 2.3+ on macOS).

#ifndef MIDNIGHT_METAL_MONT_H
#define MIDNIGHT_METAL_MONT_H

#include <metal_stdlib>

#define MONT_INLINE inline __attribute__((always_inline))

namespace mont {

using metal::mulhi;

template <typename P>
struct Elem {
    uint l[P::N];
};

// ---------------------------------------------------------------------------
// Basic operations

template <typename P>
MONT_INLINE Elem<P> zero() {
    Elem<P> r;
#pragma unroll
    for (int i = 0; i < P::N; i++) r.l[i] = 0;
    return r;
}

// The Montgomery form of 1.
template <typename P>
MONT_INLINE Elem<P> one() {
    Elem<P> r;
#pragma unroll
    for (int i = 0; i < P::N; i++) r.l[i] = P::one(i);
    return r;
}

template <typename P>
MONT_INLINE Elem<P> modulus() {
    Elem<P> r;
#pragma unroll
    for (int i = 0; i < P::N; i++) r.l[i] = P::mod(i);
    return r;
}

template <typename P>
MONT_INLINE bool is_zero(Elem<P> a) {
    uint acc = 0;
#pragma unroll
    for (int i = 0; i < P::N; i++) acc |= a.l[i];
    return acc == 0;
}

template <typename P>
MONT_INLINE bool eq(Elem<P> a, Elem<P> b) {
    uint acc = 0;
#pragma unroll
    for (int i = 0; i < P::N; i++) acc |= a.l[i] ^ b.l[i];
    return acc == 0;
}

// a < p.
template <typename P>
MONT_INLINE bool is_reduced(Elem<P> a) {
    uint borrow = 0;
#pragma unroll
    for (int i = 0; i < P::N; i++) {
        ulong d = (ulong)a.l[i] - P::mod(i) - borrow;
        borrow = (uint)(d >> 63);
    }
    return borrow != 0;
}

// t - p if t >= p, else t (t < 2p).
template <typename P>
MONT_INLINE Elem<P> reduce_once(Elem<P> t) {
    Elem<P> d;
    uint borrow = 0;
#pragma unroll
    for (int i = 0; i < P::N; i++) {
        ulong x = (ulong)t.l[i] - P::mod(i) - borrow;
        d.l[i] = (uint)x;
        borrow = (uint)(x >> 63);
    }
    return borrow ? t : d;
}

template <typename P>
MONT_INLINE Elem<P> add(Elem<P> a, Elem<P> b) {
    Elem<P> s;
    uint carry = 0;
#pragma unroll
    for (int i = 0; i < P::N; i++) {
        ulong x = (ulong)a.l[i] + b.l[i] + carry;
        s.l[i] = (uint)x;
        carry = (uint)(x >> 32);
    }
    // a + b < 2p < 2^(32 N): no carry out of the top limb
    return reduce_once<P>(s);
}

template <typename P>
MONT_INLINE Elem<P> dbl(Elem<P> a) {
    Elem<P> s;
    uint carry = 0;
#pragma unroll
    for (int i = 0; i < P::N; i++) {
        s.l[i] = (a.l[i] << 1) | carry;
        carry = a.l[i] >> 31;
    }
    return reduce_once<P>(s);
}

template <typename P>
MONT_INLINE Elem<P> sub(Elem<P> a, Elem<P> b) {
    Elem<P> d;
    uint borrow = 0;
#pragma unroll
    for (int i = 0; i < P::N; i++) {
        ulong x = (ulong)a.l[i] - b.l[i] - borrow;
        d.l[i] = (uint)x;
        borrow = (uint)(x >> 63);
    }
    // on a borrow add p back; the mask keeps the code branch free
    uint mask = 0u - borrow;
    uint carry = 0;
#pragma unroll
    for (int i = 0; i < P::N; i++) {
        ulong x = (ulong)d.l[i] + (P::mod(i) & mask) + carry;
        d.l[i] = (uint)x;
        carry = (uint)(x >> 32);
    }
    return d;
}

template <typename P>
MONT_INLINE Elem<P> neg(Elem<P> a) {
    Elem<P> d;
    uint borrow = 0;
#pragma unroll
    for (int i = 0; i < P::N; i++) {
        ulong x = (ulong)P::mod(i) - a.l[i] - borrow;
        d.l[i] = (uint)x;
        borrow = (uint)(x >> 63);
    }
    // -0 = 0
    uint mask = is_zero<P>(a) ? 0u : ~0u;
#pragma unroll
    for (int i = 0; i < P::N; i++) d.l[i] &= mask;
    return d;
}

// ---------------------------------------------------------------------------
// Multiplication strategies

// CIOS, no-carry variant, 32-bit limbs with 64-bit accumulators.
template <typename P>
MONT_INLINE Elem<P> mul_cios32(Elem<P> a, Elem<P> b) {
    Elem<P> t = zero<P>();
#pragma unroll
    for (int i = 0; i < P::N; i++) {
        ulong x = (ulong)a.l[0] * b.l[i] + t.l[0];
        uint A = (uint)(x >> 32);
        uint t0 = (uint)x;
        uint m = t0 * (uint)P::INV;
        ulong y = (ulong)m * P::mod(0) + t0;
        uint C = (uint)(y >> 32);
#pragma unroll
        for (int j = 1; j < P::N; j++) {
            x = (ulong)a.l[j] * b.l[i] + t.l[j] + A;
            A = (uint)(x >> 32);
            y = (ulong)m * P::mod(j) + (uint)x + C;
            C = (uint)(y >> 32);
            t.l[j - 1] = (uint)y;
        }
        t.l[P::N - 1] = C + A;
    }
    return reduce_once<P>(t);
}

// lo + hi * 2^32 = a * b + c + d, with 32-bit operations only.
MONT_INLINE uint mac32(uint a, uint b, uint c, uint d, thread uint &hi) {
    uint lo = a * b;
    uint h = mulhi(a, b);
    lo += c;
    h += (lo < c) ? 1u : 0u;
    lo += d;
    h += (lo < d) ? 1u : 0u;
    hi = h;
    return lo;
}

// CIOS, no-carry variant, 32-bit limbs with `mulhi` and 32-bit carries.
template <typename P>
MONT_INLINE Elem<P> mul_mulhi32(Elem<P> a, Elem<P> b) {
    Elem<P> t = zero<P>();
#pragma unroll
    for (int i = 0; i < P::N; i++) {
        uint A, C;
        uint t0 = mac32(a.l[0], b.l[i], t.l[0], 0u, A);
        uint m = t0 * (uint)P::INV;
        (void)mac32(m, P::mod(0), t0, 0u, C);
#pragma unroll
        for (int j = 1; j < P::N; j++) {
            uint tj = mac32(a.l[j], b.l[i], t.l[j], A, A);
            t.l[j - 1] = mac32(m, P::mod(j), tj, C, C);
        }
        t.l[P::N - 1] = C + A;
    }
    return reduce_once<P>(t);
}

// Finely integrated product scanning: column k sums a[j] b[k-j] and
// m[j] p[k-j] in a 96-bit accumulator.
template <typename P>
MONT_INLINE Elem<P> mul_fips32(Elem<P> a, Elem<P> b) {
    uint m[P::N];
    Elem<P> u;
    ulong acc = 0;
    uint acc2 = 0;
#define MONT_FIPS_MAC(x, y)                \
    {                                      \
        ulong p_ = (ulong)(x) * (y);       \
        acc += p_;                         \
        acc2 += (acc < p_) ? 1u : 0u;      \
    }
#pragma unroll
    for (int i = 0; i < P::N; i++) {
#pragma unroll
        for (int j = 0; j < i; j++) {
            MONT_FIPS_MAC(a.l[j], b.l[i - j]);
            MONT_FIPS_MAC(m[j], P::mod(i - j));
        }
        MONT_FIPS_MAC(a.l[i], b.l[0]);
        m[i] = (uint)acc * (uint)P::INV;
        MONT_FIPS_MAC(m[i], P::mod(0));
        acc = (acc >> 32) | ((ulong)acc2 << 32);
        acc2 = 0;
    }
#pragma unroll
    for (int i = P::N; i < 2 * P::N - 1; i++) {
#pragma unroll
        for (int j = i - P::N + 1; j < P::N; j++) {
            MONT_FIPS_MAC(a.l[j], b.l[i - j]);
            MONT_FIPS_MAC(m[j], P::mod(i - j));
        }
        u.l[i - P::N] = (uint)acc;
        acc = (acc >> 32) | ((ulong)acc2 << 32);
        acc2 = 0;
    }
#undef MONT_FIPS_MAC
    // the result is below 2p < 2^(32 N): the accumulator holds its top limb
    u.l[P::N - 1] = (uint)acc;
    return reduce_once<P>(u);
}

// lo + hi * 2^64 = a * b + c + d on 64-bit limbs.
MONT_INLINE ulong mac64(ulong a, ulong b, ulong c, ulong d, thread ulong &hi) {
    ulong lo = a * b;
    ulong h = mulhi(a, b);
    lo += c;
    h += (lo < c) ? 1ul : 0ul;
    lo += d;
    h += (lo < d) ? 1ul : 0ul;
    hi = h;
    return lo;
}

// CIOS, no-carry variant, 64-bit limbs.
template <typename P>
MONT_INLINE Elem<P> mul_cios64(Elem<P> a, Elem<P> b) {
    const int W = P::N / 2;
    ulong x[W], y[W], q[W], t[W];
#pragma unroll
    for (int i = 0; i < W; i++) {
        x[i] = (ulong)a.l[2 * i] | ((ulong)a.l[2 * i + 1] << 32);
        y[i] = (ulong)b.l[2 * i] | ((ulong)b.l[2 * i + 1] << 32);
        q[i] = (ulong)P::mod(2 * i) | ((ulong)P::mod(2 * i + 1) << 32);
        t[i] = 0;
    }
    const ulong inv = P::inv64();
#pragma unroll
    for (int i = 0; i < W; i++) {
        ulong A, C;
        ulong t0 = mac64(x[0], y[i], t[0], 0ul, A);
        ulong m = t0 * inv;
        (void)mac64(m, q[0], t0, 0ul, C);
#pragma unroll
        for (int j = 1; j < W; j++) {
            ulong tj = mac64(x[j], y[i], t[j], A, A);
            t[j - 1] = mac64(m, q[j], tj, C, C);
        }
        t[W - 1] = C + A;
    }
    Elem<P> r;
#pragma unroll
    for (int i = 0; i < W; i++) {
        r.l[2 * i] = (uint)t[i];
        r.l[2 * i + 1] = (uint)(t[i] >> 32);
    }
    return reduce_once<P>(r);
}

// ---------------------------------------------------------------------------
// The default multiplication. It is fixed here, not per kernel file: without
// the Metal toolchain at build time every shader file is compiled as one
// translation unit, so a per-file override would not hold. A kernel that
// wants another strategy calls it by name (`mont::mul_fips32<P>`).

#define MONT_MUL_DEFAULT mul_cios32

template <typename P>
MONT_INLINE Elem<P> mul(Elem<P> a, Elem<P> b) {
    return MONT_MUL_DEFAULT<P>(a, b);
}

template <typename P>
MONT_INLINE Elem<P> sqr(Elem<P> a) {
    return MONT_MUL_DEFAULT<P>(a, a);
}

} // namespace mont

#endif // MIDNIGHT_METAL_MONT_H
