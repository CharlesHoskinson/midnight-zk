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

//! The job's polynomials and tables in the form of the numerator kernel
//! (`shaders/include/quotient.h`): the CUDA crate's arena of device pointers
//! (`Arena` in `gpu/src/quotient.rs`) becomes a buffer of 32-bit words with
//! word offsets, and a polynomial a handle `(bank << 30) | index` into one of
//! three flat buffers. Every check of the CUDA crate's `checked` and
//! `Arena::program` is made here, on host data, before any device work.

use std::collections::HashMap;

use super::abi::{
    COLUMN_ADVICE, COLUMN_FIXED, COLUMN_INSTANCE, INDEX_MASK, OP_MUL_ADD, ProgramAbi,
    QuotientJobAbi, TAG_READ, TAG_SLOT, TAG_UNIFORM, slice,
};
use crate::{Scalar, account::Reason, field};

pub(crate) const HANDLE_SHIFT: u32 = 30;
pub(crate) const BANK_PROOF: u32 = 0;
pub(crate) const BANK_KEY_RES: u32 = 1;
pub(crate) const BANK_KEY_TMP: u32 = 2;
/// Polynomials per bank.
pub(crate) const MAX_BANK_POLYS: usize = 1 << HANDLE_SHIFT;

const OP_WORDS: usize = 5;
const PROG_WORDS: usize = 8;
const LOOKUP_WORDS: usize = 4;
const CHUNK_WORDS: usize = 2;
const TRASH_WORDS: usize = 4;

/// The job, flattened: its polynomials and the kernel's tables.
pub(crate) struct Tables<'j> {
    /// The proof's polynomials (coefficients, `2^k` each), distinct, in the
    /// order of their cosets in the proof bank.
    pub(crate) proof: Vec<&'j [Scalar]>,
    /// The proving-key polynomials (extended cosets), distinct.
    pub(crate) keys: Vec<&'j [Scalar]>,
    /// The kernel's words. A key handle is written as its index in `keys`
    /// and fixed by [`Tables::place_keys`].
    pub(crate) words: Vec<u32>,
    /// The programs' uniforms.
    pub(crate) uniforms: Vec<Scalar>,
    key_words: Vec<usize>,
    pub(crate) gates: u32,
    pub(crate) perm: u32,
    pub(crate) lookups: u32,
    pub(crate) trash: u32,
    /// Handles (key indices until placed) of `l0`, `l_last`, `l_active`.
    pub(crate) lagrange: [u32; 3],
    /// Field multiplications per row, about (the command-buffer split).
    pub(crate) cost_per_row: u64,
}

fn handle(bank: u32, index: usize) -> u32 {
    (bank << HANDLE_SHIFT) | index as u32
}

impl<'j> Tables<'j> {
    /// Flattens `j` (whose header has passed [`super::checked`]).
    ///
    /// # Safety
    /// The job's contract: every pointer and count of `j` is valid.
    pub(crate) unsafe fn new(j: &'j QuotientJobAbi) -> Result<Self, Reason> {
        let (n, en) = (1usize << j.k, 1usize << j.ext_k);
        // SAFETY (every `slice`): the job's contract.
        let advice = unsafe { slice(j.advice, j.nadvice) };
        let instance = unsafe { slice(j.instance, j.ninstance) };
        let fixed = unsafe { slice(j.fixed, j.nfixed) };
        let sigma = unsafe { slice(j.perm_sigma, j.perm_ncolumns) };
        let products = unsafe { slice(j.perm_products, j.perm_nsets) };
        let kinds = unsafe { slice(j.perm_column_kind, j.perm_ncolumns) };
        let index = unsafe { slice(j.perm_column_index, j.perm_ncolumns) };
        let lookups = unsafe { slice(j.lookups, j.nlookups) };
        let trash = unsafe { slice(j.trash, j.ntrash) };

        let mut t = Tables {
            proof: Vec::new(),
            keys: Vec::new(),
            words: Vec::new(),
            uniforms: Vec::new(),
            key_words: Vec::new(),
            gates: 0,
            perm: 0,
            lookups: 0,
            trash: 0,
            lagrange: [0; 3],
            cost_per_row: 0,
        };
        let mut proof_ix: HashMap<usize, usize> = HashMap::new();
        let mut key_ix: HashMap<usize, usize> = HashMap::new();
        // a proof polynomial's handle (its coset in the proof bank)
        let mut proof_h = |p: *const Scalar, t: &mut Tables<'j>| -> Result<u32, Reason> {
            if p.is_null() {
                return Err(Reason::Shape);
            }
            let next = proof_ix.len();
            let i = *proof_ix.entry(p as usize).or_insert_with(|| {
                // SAFETY: the job's contract: 2^k coefficients.
                t.proof.push(unsafe { slice(p, n) });
                next
            });
            if i >= MAX_BANK_POLYS {
                return Err(Reason::Shape);
            }
            Ok(handle(BANK_PROOF, i))
        };
        // a key polynomial's index in `keys`
        let mut key_i = |p: *const Scalar, t: &mut Tables<'j>| -> Result<u32, Reason> {
            if p.is_null() {
                return Err(Reason::Shape);
            }
            let next = key_ix.len();
            let i = *key_ix.entry(p as usize).or_insert_with(|| {
                // SAFETY: the job's contract: 2^ext_k values.
                t.keys.push(unsafe { slice(p, en) });
                next
            });
            if i >= MAX_BANK_POLYS {
                return Err(Reason::Shape);
            }
            Ok(i as u32)
        };

        // the columns, as handles (fixed: key indices, fixed by place_keys)
        let advice_h: Vec<u32> =
            advice.iter().map(|p| proof_h(*p, &mut t)).collect::<Result<_, _>>()?;
        let instance_h: Vec<u32> =
            instance.iter().map(|p| proof_h(*p, &mut t)).collect::<Result<_, _>>()?;
        let fixed_h: Vec<Option<u32>> = fixed
            .iter()
            .map(|p| {
                if p.is_null() {
                    Ok(None)
                } else {
                    key_i(*p, &mut t).map(Some)
                }
            })
            .collect::<Result<_, _>>()?;
        // (handle, is a key index)
        let col = |kind: u32, c: u32| -> Option<(u32, bool)> {
            match kind {
                COLUMN_FIXED => fixed_h.get(c as usize).copied().flatten().map(|h| (h, true)),
                COLUMN_ADVICE => advice_h.get(c as usize).map(|h| (*h, false)),
                COLUMN_INSTANCE => instance_h.get(c as usize).map(|h| (*h, false)),
                _ => None,
            }
        };
        let nslots = j.nslots;

        // l0, l_last, l_active
        for (i, p) in [j.l0, j.l_last, j.l_active].into_iter().enumerate() {
            t.lagrange[i] = key_i(p, &mut t)?;
        }

        // the gates
        t.gates = t.program(&j.gates, nslots, &col)?;

        // the permutation: accumulators, columns, sigmas
        if j.perm_nsets > 0 {
            let start = t.words.len();
            for p in products {
                let h = proof_h(*p, &mut t)?;
                t.words.push(h);
            }
            for (k, c) in kinds.iter().zip(index) {
                let (h, key) = col(*k, *c).ok_or(Reason::Shape)?;
                t.push_handle(h, key);
            }
            for s in sigma {
                let i = key_i(*s, &mut t)?;
                t.push_handle(i, true);
            }
            t.perm = start as u32;
            t.cost_per_row += 4 * j.perm_ncolumns as u64 + 6 * j.perm_nsets as u64;
        }

        // the lookups: programs first, then the descriptors
        let mut descs = Vec::with_capacity(lookups.len());
        for l in lookups {
            if l.nchunks == 0 {
                return Err(Reason::Shape);
            }
            // SAFETY: the job's contract.
            let (progs, helpers) =
                unsafe { (slice(l.programs, l.nchunks), slice(l.helpers, l.nchunks)) };
            let mut chunks = Vec::with_capacity(l.nchunks);
            for (p, h) in progs.iter().zip(helpers) {
                if p.noutputs != 4 {
                    return Err(Reason::Shape);
                }
                let off = t.program(p, nslots, &col)?;
                chunks.push((proof_h(*h, &mut t)?, off));
            }
            let agg = proof_h(l.aggregator, &mut t)?;
            let mult = proof_h(l.multiplicities, &mut t)?;
            descs.push((agg, mult, chunks));
            t.cost_per_row += 6 + 3 * l.nchunks as u64;
        }
        let mut chunk_offs = Vec::with_capacity(descs.len());
        for (_, _, chunks) in &descs {
            chunk_offs.push(t.words.len() as u32);
            for (h, off) in chunks {
                t.words.extend([*h, *off]);
            }
        }
        t.lookups = t.words.len() as u32;
        for ((agg, mult, chunks), off) in descs.iter().zip(chunk_offs) {
            t.words.extend([*agg, *mult, chunks.len() as u32, off]);
        }

        // the trash arguments
        let mut tdescs = Vec::with_capacity(trash.len());
        for a in trash {
            if a.program.noutputs != 1 {
                return Err(Reason::Shape);
            }
            let sel = fixed_h
                .get(a.selector_column as usize)
                .copied()
                .flatten()
                .ok_or(Reason::Shape)?;
            let off = t.program(&a.program, nslots, &col)?;
            tdescs.push((proof_h(a.trash, &mut t)?, sel, off));
            t.cost_per_row += 3;
        }
        t.trash = t.words.len() as u32;
        for (h, sel, off) in tdescs {
            t.words.push(h);
            t.push_handle(sel, true);
            t.words.extend([off, 0]);
        }
        let _ = (LOOKUP_WORDS, CHUNK_WORDS, TRASH_WORDS, PROG_WORDS);
        if t.words.len() > u32::MAX as usize / 2 || t.uniforms.len() > u32::MAX as usize / 2 {
            return Err(Reason::Shape);
        }
        Ok(t)
    }

    fn push_handle(&mut self, h: u32, key: bool) {
        if key {
            self.key_words.push(self.words.len());
        }
        self.words.push(h);
    }

    /// Packs a program after the checks of the CUDA crate's
    /// `Arena::program`: every operand in range, every slot below the job's
    /// slots, every read on a column the job has, the uniforms reduced.
    /// Returns its word offset.
    fn program(
        &mut self,
        p: &ProgramAbi,
        nslots: u32,
        col: &dyn Fn(u32, u32) -> Option<(u32, bool)>,
    ) -> Result<u32, Reason> {
        // SAFETY: the job's contract.
        let (ops, reads, uniforms, outputs) = unsafe {
            (
                slice(p.ops, p.nops),
                slice(p.reads, p.nreads),
                slice(p.uniforms, p.nuniforms),
                slice(p.outputs, p.noutputs),
            )
        };
        let ok_operand = |o: u32| {
            let i = (o & INDEX_MASK) as usize;
            match o >> 30 {
                TAG_UNIFORM => i < uniforms.len(),
                TAG_READ => i < reads.len(),
                TAG_SLOT => i < nslots as usize,
                _ => false,
            }
        };
        let ops_ok = ops.iter().all(|op| {
            op.kind <= OP_MUL_ADD
                && op.dst < nslots
                && ok_operand(op.a)
                && ok_operand(op.b)
                && ok_operand(op.c)
        });
        if !ops_ok || !outputs.iter().all(|o| ok_operand(*o)) || p.nslots > nslots {
            return Err(Reason::Shape);
        }
        let read_cols: Vec<(u32, bool)> = reads
            .iter()
            .map(|r| col(r.kind, r.column))
            .collect::<Option<_>>()
            .ok_or(Reason::Shape)?;
        if !field::all_reduced(uniforms) {
            return Err(Reason::Unreduced);
        }
        let ops_off = self.words.len() + PROG_WORDS;
        let reads_off = ops_off + OP_WORDS * ops.len();
        let outs_off = reads_off + 2 * reads.len();
        let start = self.words.len();
        let unis = self.uniforms.len();
        self.words.extend([
            ops_off as u32,
            ops.len() as u32,
            reads_off as u32,
            reads.len() as u32,
            unis as u32,
            outs_off as u32,
            outputs.len() as u32,
            0,
        ]);
        for op in ops {
            self.words.extend([op.kind, op.dst, op.a, op.b, op.c]);
        }
        for (r, (h, key)) in reads.iter().zip(read_cols) {
            self.push_handle(h, key);
            self.words.push(r.rotation as u32);
        }
        self.words.extend_from_slice(outputs);
        self.uniforms.extend_from_slice(uniforms);
        self.cost_per_row += ops.len() as u64;
        Ok(start as u32)
    }

    /// Writes the key handles: `bank_of[i]` is `(bank, index)` of key `i`.
    pub(crate) fn place_keys(&mut self, bank_of: &[(u32, usize)]) -> [u32; 3] {
        for &w in &self.key_words {
            let (b, i) = bank_of[self.words[w] as usize];
            self.words[w] = handle(b, i);
        }
        self.lagrange.map(|k| {
            let (b, i) = bank_of[k as usize];
            handle(b, i)
        })
    }
}
