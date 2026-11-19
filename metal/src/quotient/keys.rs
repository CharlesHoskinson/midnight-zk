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

//! Resident proving-key copies: the counterpart of `register_key` and
//! `KeyHandle` of the CUDA crate (`gpu/src/quotient.rs`).
//!
//! [`register_key`] records proving-key polynomials (extended cosets, as
//! `ProvingKey::gpu_resident_polys` lists them) by address, length and a
//! content digest. The first quotient that reads them makes the device copy:
//! on unified memory a shared Metal buffer, one per handle, holding the
//! handle's polynomials one after the other (the numerator kernel reads it as
//! one bank, see `shaders/quotient.metal`). A polynomial is served from the
//! copy only after its full digest matched the registration in the current
//! request (at every serve in a debug build); a polynomial whose content
//! changed is counted ([`key_stale`]), loses its registration and is uploaded
//! for the call only. The digests of a quotient's key polynomials are
//! computed while the device transforms the proof's polynomials ([`plan`],
//! [`settle`]), and settled before the numerator, the first kernel that
//! reads a slot.
//!
//! The copies are freed at [`crate::end_request`] unless [`keep_resident`]
//! (`set_keep_resident`, or `MIDNIGHT_METAL_KEEP_KEYS=1`), by
//! [`evict_key_copies`], and when the handle is dropped. They hold public data
//! only, and are zeroed when freed like every buffer of this crate.

use std::{
    collections::HashMap,
    sync::{
        Mutex, MutexGuard, OnceLock,
        atomic::{AtomicBool, AtomicU64, Ordering::Relaxed},
    },
};

use crate::{Scalar, ctx::Buffer};

/// Bytes per hashed chunk of the digest.
const CHUNK: usize = 16 << 20;

fn sha256(b: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    // SAFETY: valid input and output buffers.
    unsafe { blst::blst_sha256(out.as_mut_ptr(), b.as_ptr(), b.len()) };
    out
}

/// The content digest of each of `vs` (public data): SHA-256 of the length
/// and the SHA-256 of each 16 MiB chunk, the same function as the CUDA
/// crate's `ident::digest`. The chunks of all of them are hashed together on
/// all cores (a polynomial of a few chunks alone would keep most idle).
pub(crate) fn digest_many(vs: &[&[Scalar]]) -> Vec<[u8; 32]> {
    let bytes: Vec<&[u8]> = vs
        .iter()
        .map(|v| {
            // SAFETY: plain limbs, no padding.
            unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, size_of_val(*v)) }
        })
        .collect();
    let chunks: Vec<&[u8]> = bytes.iter().flat_map(|b| b.chunks(CHUNK)).collect();
    let mut ds = vec![[0u8; 32]; chunks.len()];
    let threads = std::thread::available_parallelism()
        .map_or(4, |t| t.get())
        .clamp(1, 16)
        .min(chunks.len());
    if threads <= 1 {
        for (d, c) in ds.iter_mut().zip(&chunks) {
            *d = sha256(c);
        }
    } else {
        // each thread takes the next chunk
        let next = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|sc| {
            let hs: Vec<_> = (0..threads)
                .map(|_| {
                    sc.spawn(|| {
                        let mut done = Vec::new();
                        loop {
                            let i = next.fetch_add(1, Relaxed);
                            let Some(c) = chunks.get(i) else { break };
                            done.push((i, sha256(c)));
                        }
                        done
                    })
                })
                .collect();
            for h in hs {
                for (i, d) in h.join().expect("a digest thread") {
                    ds[i] = d;
                }
            }
        });
    }
    let mut at = 0;
    bytes
        .iter()
        .map(|b| {
            let n = b.len().div_ceil(CHUNK);
            let mut top = Vec::with_capacity(8 + 32 * n);
            top.extend_from_slice(&(b.len() as u64).to_le_bytes());
            for d in &ds[at..at + n] {
                top.extend_from_slice(d);
            }
            at += n;
            sha256(&top)
        })
        .collect()
}

/// A registered polynomial.
struct Entry {
    /// The handle.
    id: u64,
    /// Its index in the handle's copy.
    slot: usize,
    digest: [u8; 32],
    /// The request in which its content last matched the digest.
    seen: u64,
}

/// The device copy of one handle.
struct Copy {
    /// Polynomials of the handle (slots of the copy).
    npolys: usize,
    /// The copy; `None` before the first quotient, after an eviction, and
    /// while a quotient uses it (it is lent, see [`Lent`]).
    buf: Option<Buffer>,
    /// Which slots hold their polynomial.
    present: Vec<bool>,
}

#[derive(Default)]
struct Registry {
    entries: HashMap<(usize, usize), Entry>,
    copies: HashMap<u64, Copy>,
    /// Increments at every eviction: a copy lent before it is not returned.
    evictions: u64,
}

static REGISTRY: Mutex<Option<Registry>> = Mutex::new(None);
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
static KEY_STALE: AtomicU64 = AtomicU64::new(0);
/// The current request ([`crate::begin_request`] increments it).
static EPOCH: AtomicU64 = AtomicU64::new(1);
static KEEP: OnceLock<AtomicBool> = OnceLock::new();

fn registry() -> MutexGuard<'static, Option<Registry>> {
    REGISTRY.lock().unwrap_or_else(|e| e.into_inner())
}

fn keep() -> &'static AtomicBool {
    KEEP.get_or_init(|| {
        AtomicBool::new(std::env::var("MIDNIGHT_METAL_KEEP_KEYS").as_deref() == Ok("1"))
    })
}

/// Whether the device copies of registered proving-key polynomials stay after
/// a request (default `false`, or `MIDNIGHT_METAL_KEEP_KEYS=1`).
pub fn keep_resident() -> bool {
    keep().load(Relaxed)
}

/// Sets [`keep_resident`].
pub fn set_keep_resident(on: bool) {
    keep().store(on, Relaxed);
}

/// A new request: the digests are checked again at the next serve.
pub(crate) fn begin_request() {
    EPOCH.fetch_add(1, Relaxed);
}

/// The end of a request: frees the copies unless [`keep_resident`].
pub(crate) fn end_request() {
    if !keep_resident() {
        evict_key_copies();
    }
}

/// Frees every device copy of a registered polynomial (the registrations
/// stay; the next quotient makes the copies again). Returns the bytes freed;
/// a copy in use by a running quotient is freed when that quotient returns
/// it and is not counted.
pub fn evict_key_copies() -> u64 {
    let mut freed = Vec::new();
    {
        let mut r = registry();
        let Some(r) = r.as_mut() else { return 0 };
        r.evictions += 1;
        for c in r.copies.values_mut() {
            freed.extend(c.buf.take());
            c.present.iter_mut().for_each(|p| *p = false);
        }
    }
    // zeroed (in the background) and put back into the pool outside the
    // lock
    let bytes = freed.iter().map(|b| b.len_bytes() as u64).sum();
    crate::ctx::release_later(freed);
    bytes
}

/// The registration of proving-key polynomials. While it lives the
/// registrant keeps them alive and unchanged; the device copy is made at the
/// first quotient that reads them and freed when this is dropped.
#[derive(Debug)]
pub struct KeyHandle {
    id: u64,
    keys: Vec<(usize, usize)>,
}

impl KeyHandle {
    /// Whether every registered polynomial still has its registered content
    /// (full digests).
    pub fn verify(&self, polys: &[&[Scalar]]) -> bool {
        if polys.len() != self.keys.len() {
            return false;
        }
        let ds = digest_many(polys);
        let r = registry();
        polys.iter().zip(&self.keys).zip(&ds).all(|((p, key), d)| {
            (p.as_ptr() as usize, p.len()) == *key
                && r.as_ref()
                    .and_then(|r| r.entries.get(key))
                    .is_some_and(|e| e.id == self.id && e.digest == *d)
        })
    }

    /// The bytes of this handle's device copy now (0 before the first
    /// quotient that reads it, after an eviction, or while a quotient uses
    /// it).
    pub fn resident_bytes(&self) -> u64 {
        registry()
            .as_ref()
            .and_then(|r| r.copies.get(&self.id))
            .and_then(|c| c.buf.as_ref())
            .map_or(0, |b| b.len_bytes() as u64)
    }
}

impl Drop for KeyHandle {
    fn drop(&mut self) {
        let copy = {
            let mut r = registry();
            let Some(r) = r.as_mut() else { return };
            for key in &self.keys {
                if r.entries.get(key).is_some_and(|e| e.id == self.id) {
                    r.entries.remove(key);
                }
            }
            r.copies.remove(&self.id)
        };
        drop(copy);
    }
}

/// Registers proving-key polynomials (each an extended coset) by their
/// digests. A polynomial registered again moves to the new handle.
pub fn register_key(polys: &[&[Scalar]]) -> KeyHandle {
    let id = NEXT_ID.fetch_add(1, Relaxed);
    // hash before taking the lock
    let entries: Vec<((usize, usize), [u8; 32])> = polys
        .iter()
        .map(|p| (p.as_ptr() as usize, p.len()))
        .zip(digest_many(polys))
        .collect();
    let now = EPOCH.load(Relaxed);
    let mut r = registry();
    let r = r.get_or_insert_with(Registry::default);
    let mut keys = Vec::with_capacity(entries.len());
    for (slot, (key, d)) in entries.into_iter().enumerate() {
        r.entries.insert(
            key,
            Entry {
                id,
                slot,
                digest: d,
                seen: now,
            },
        );
        keys.push(key);
    }
    r.copies.insert(
        id,
        Copy {
            npolys: keys.len(),
            buf: None,
            present: vec![false; keys.len()],
        },
    );
    KeyHandle { id, keys }
}

/// Registered polynomials now.
pub fn keys_registered() -> usize {
    registry().as_ref().map_or(0, |r| r.entries.len())
}

/// Registered polynomials dropped at a quotient because their content
/// changed, since process start.
pub fn key_stale() -> u64 {
    KEY_STALE.load(Relaxed)
}

/// Where a proving-key polynomial of a job is read from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Place {
    /// Slot of the lent resident copy.
    Resident(usize),
    /// Uploaded for the call.
    Temp,
}

/// The resident copy lent to one quotient: returned by [`Lent::give_back`]
/// (kept when the handle still lives and no eviction happened meanwhile).
pub(crate) struct Lent {
    pub(crate) id: u64,
    /// The copy (`None`: to be allocated by the caller with
    /// `npolys * poly_bytes` bytes).
    pub(crate) buf: Option<Buffer>,
    pub(crate) npolys: usize,
    pub(crate) present: Vec<bool>,
    evictions: u64,
}

impl Lent {
    /// Returns the copy to the registry, or drops it (zeroed) when it is no
    /// longer wanted.
    pub(crate) fn give_back(self) {
        let back = {
            let mut r = registry();
            match r.as_mut() {
                Some(r) if r.evictions == self.evictions => match r.copies.get_mut(&self.id) {
                    Some(c) if c.buf.is_none() => {
                        c.buf = self.buf;
                        c.present = self.present;
                        None
                    }
                    _ => self.buf,
                },
                _ => self.buf,
            }
        };
        drop(back);
    }
}

/// The placement of a job's proving-key polynomials before their digests
/// are checked ([`plan`]), settled by [`settle`].
pub(crate) struct Plan {
    /// Where each polynomial is read from, assuming that every registered
    /// one still has its content; [`settle`] makes a stale one `Temp`.
    pub(crate) places: Vec<Place>,
    /// The copy lent for the call.
    pub(crate) lent: Option<Lent>,
    /// Indices (into the job's polynomials) of the registered ones whose
    /// digest is to be checked: not checked yet in this request, or every
    /// one in a debug build.
    pub(crate) check: Vec<usize>,
    /// `(id, slot)` of each registered polynomial as [`plan`] found it.
    regs: Vec<Option<(u64, usize)>>,
    /// The request of the plan.
    epoch: u64,
}

/// Plans the placement of the job's proving-key polynomials (`keys`: start
/// and length of each, distinct) without hashing them: the handle with the
/// most registered polynomials of the job is lent for the call and its
/// polynomials are placed in its slots; the others are uploaded. Each
/// registered polynomial that was not checked against its digest in this
/// request (every one in a debug build) is listed in [`Plan::check`]; the
/// caller hashes them ([`digest_many`]) while the device works on the proof's
/// polynomials, and [`settle`] drops a stale one before any slot is read.
pub(crate) fn plan(keys: &[&[Scalar]]) -> Plan {
    let now = EPOCH.load(Relaxed);
    let mut p = Plan {
        places: vec![Place::Temp; keys.len()],
        lent: None,
        check: Vec::new(),
        regs: vec![None; keys.len()],
        epoch: now,
    };
    let mut r = registry();
    let Some(reg) = r.as_mut() else {
        return p;
    };
    for (i, k) in keys.iter().enumerate() {
        let key = (k.as_ptr() as usize, k.len());
        let Some(e) = reg.entries.get(&key) else {
            continue;
        };
        p.regs[i] = Some((e.id, e.slot));
        if e.seen != now || cfg!(debug_assertions) {
            p.check.push(i);
        }
    }
    // the handle with the most registered polynomials
    let mut counts: HashMap<u64, usize> = HashMap::new();
    for (id, _) in p.regs.iter().flatten() {
        *counts.entry(*id).or_default() += 1;
    }
    let Some(best) = counts.into_iter().max_by_key(|(id, n)| (*n, u64::MAX - id)).map(|c| c.0)
    else {
        return p;
    };
    let Some(c) = reg.copies.get_mut(&best) else {
        return p;
    };
    for (pl, f) in p.places.iter_mut().zip(&p.regs) {
        if let Some((id, slot)) = f
            && *id == best
        {
            *pl = Place::Resident(*slot);
        }
    }
    // without a copy no slot is present
    let present = if c.buf.is_some() && c.present.len() == c.npolys {
        c.present.clone()
    } else {
        vec![false; c.npolys]
    };
    p.lent = Some(Lent {
        id: best,
        buf: c.buf.take(),
        npolys: c.npolys,
        present,
        evictions: reg.evictions,
    });
    p
}

/// Settles a [`plan`] with the digests of its [`Plan::check`] polynomials
/// (in that order): a polynomial whose digest matches its registration is
/// marked checked for the request; one whose content changed is counted
/// ([`key_stale`]), loses its registration and its slot of the lent copy,
/// and becomes `Temp`, as does one whose registration went away meanwhile.
/// Returns whether a polynomial placed in a slot became `Temp`.
pub(crate) fn settle(p: &mut Plan, keys: &[&[Scalar]], digests: &[[u8; 32]]) -> bool {
    let mut moved = false;
    let mut r = registry();
    for (&i, d) in p.check.iter().zip(digests) {
        let key = (keys[i].as_ptr() as usize, keys[i].len());
        let planned = p.regs[i];
        let fresh = match r.as_mut().and_then(|reg| reg.entries.get_mut(&key)) {
            Some(e) if Some((e.id, e.slot)) == planned => {
                if e.digest == *d {
                    e.seen = p.epoch;
                    true
                } else {
                    KEY_STALE.fetch_add(1, Relaxed);
                    let (id, slot) = (e.id, e.slot);
                    let reg = r.as_mut().expect("a registry");
                    reg.entries.remove(&key);
                    // the slot's copy no longer holds a registered polynomial
                    if let Some(c) = reg.copies.get_mut(&id)
                        && let Some(pr) = c.present.get_mut(slot)
                    {
                        *pr = false;
                    }
                    false
                }
            }
            _ => false,
        };
        if !fresh && let Place::Resident(slot) = p.places[i] {
            if let Some(l) = p.lent.as_mut()
                && let Some(pr) = l.present.get_mut(slot)
            {
                *pr = false;
            }
            p.places[i] = Place::Temp;
            moved = true;
        }
    }
    moved
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The digest of each polynomial, one at a time, as the definition
    /// reads: SHA-256 of the length and of each chunk's SHA-256.
    fn reference(v: &[Scalar]) -> [u8; 32] {
        // SAFETY: plain limbs, no padding.
        let b = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, size_of_val(v)) };
        let mut top = (b.len() as u64).to_le_bytes().to_vec();
        for c in b.chunks(CHUNK) {
            top.extend_from_slice(&sha256(c));
        }
        sha256(&top)
    }

    /// Hashing several polynomials together gives each one's own digest
    /// (empty, shorter than a chunk, several chunks with a partial last one).
    #[test]
    fn digests_of_several_polynomials_are_their_own() {
        let words = |n: usize, seed: u64| -> Vec<Scalar> {
            (0..n as u64)
                .map(|i| [i ^ seed, i.wrapping_mul(0x9e37_79b9_7f4a_7c15), seed, !i])
                .collect()
        };
        let per_chunk = CHUNK / size_of::<Scalar>();
        let polys = [
            words(0, 1),
            words(3, 2),
            words(per_chunk, 3),
            words(2 * per_chunk + 5, 4),
            words(per_chunk / 2, 5),
        ];
        let refs: Vec<&[Scalar]> = polys.iter().map(|p| p.as_slice()).collect();
        let want: Vec<[u8; 32]> = refs.iter().map(|p| reference(p)).collect();
        assert_eq!(digest_many(&refs), want);
        assert_eq!(digest_many(&refs[3..4]), want[3..4]);
        assert!(digest_many(&[]).is_empty());
    }
}
