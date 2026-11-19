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

//! Metal device tests. Compiled only with `--features metal` (the build script
//! then sets the cfg `midnight_metal_linked`). Each test returns early, with a
//! line on stderr, when no device is detected.
//!
//! The kernels of `midnight-metal` may serve or decline a call: the tests pass
//! either way, and print the served and declined counts of each class. A
//! served call that computed a wrong value fails the proof's verification.

#![cfg(all(feature = "metal", midnight_metal_linked, target_os = "macos"))]

mod common;

use common::*;
use midnight_gpu_prover::{
    RequestOptions,
    config::DeviceSelect,
    device::{self, Device, Phases},
    example::{PoseidonPreimage, example_input},
    input::SecretBytes,
};

/// The hooks and the device layer's counters are process-wide: the tests of
/// this file run one at a time.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn detect() -> Option<Box<dyn Device>> {
    let d = device::detect(DeviceSelect::Auto);
    if d.info().is_none() {
        eprintln!("no Metal device detected: skipped");
        return None;
    }
    Some(d)
}

fn print_classes(what: &str) {
    let s = midnight_metal::stats();
    eprintln!(
        "{what}: served msm={} fft={} ifft={} quotient={}",
        s.msm.served, s.fft.served, s.ifft.served, s.quotient.served
    );
    let (cached, one_shot, uploads) = midnight_metal::msm_counts();
    eprintln!(
        "{what}: msm table={cached} one-shot={one_shot} uploads={uploads}, handles={}, stale={}",
        midnight_metal::srs_handles(),
        midnight_metal::srs_stale()
    );
    for (class, reason, count) in midnight_metal::declines() {
        eprintln!("{what}: declined {class} {reason} x{count}");
    }
}

#[test]
fn a_proof_with_the_device_phases_admitted_verifies_and_the_erasure_succeeds() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(d) = detect() else { return };
    // let calls of the example circuit's size reach the kernels
    assert!(midnight_metal::set_min_k(1));
    let info = d.info().unwrap();
    eprintln!("device: {info:?}");
    let mut s = service(config(), d);
    let served = s
        .prove(
            &SecretBytes::new(example_input(1)),
            RequestOptions::default(),
        )
        .unwrap();
    let r = &served.record;
    eprintln!(
        "label {:?}, admitted {:?}, device calls {:?}, fallbacks {:?}",
        r.label,
        r.admitted.names(),
        r.device_calls,
        r.fallbacks
    );
    for f in &r.declines {
        eprintln!("record decline: {} {} x{}", f.phase, f.reason, f.count);
    }
    print_classes("midnight-metal");
    eprintln!(
        "wipe {:?}, device bytes now/peak {:?}",
        r.wipe,
        midnight_metal::device_bytes()
    );
    assert!(r.admitted.contains(Phases::ALL), "{:?}", r.admitted);
    eprintln!("held after the request: {:?}", s.held());
    assert!(s.retired().is_none(), "the erasure was not confirmed");
    assert_eq!(r.wipe.erasure_failures, 0);
    assert!(r.verify_s > 0.0);
    // the stock verifier of the standard library accepts the proof
    let (inst, _) =
        midnight_gpu_prover::ServedRelation::decode_input(&PoseidonPreimage, &example_input(1))
            .unwrap();
    midnight_zk_stdlib::verify::<PoseidonPreimage, blake2b_simd::State, _>(
        s.verifier_params(),
        s.vk(),
        &inst,
        None,
        &served.proof,
    )
    .unwrap();
    assert!(midnight_metal::end_request());
    // the counters of the record are the crate's
    let st = midnight_metal::stats();
    for (class, served) in [
        ("msm", st.msm.served),
        ("fft", st.fft.served),
        ("ifft", st.ifft.served),
        ("quotient", st.quotient.served),
    ] {
        assert_eq!(
            r.device_calls.get(class).copied().unwrap_or(0),
            served,
            "{class}"
        );
    }
    midnight_metal::set_min_k(14);
}

#[test]
fn the_device_layer_admits_phases_and_ends_a_request() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(d) = detect() else { return };
    d.reset_counters();
    // before the context exists no free memory is reported
    let m = d.mem().unwrap();
    assert!(m.total > 0);
    d.activate();
    let m = d.mem().unwrap();
    eprintln!("memory: {m:?}");
    assert_eq!(d.begin_request(Phases::NONE), Phases::NONE);
    assert_eq!(d.begin_request(Phases::ALL), Phases::ALL);
    assert!(d.revoke());
    assert!(!d.revoke());
    let w = d.end_request(true).unwrap();
    eprintln!("wipe: {w:?}");
    assert_eq!(w.erasure_failures, 0);
    assert_eq!(d.held(), device::Held::default());
    assert_eq!(d.evict(true), 0);
    print_classes("device layer");
}
