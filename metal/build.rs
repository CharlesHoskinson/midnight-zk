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

//! Build of the Metal shaders.
//!
//! On a macOS target every `shaders/*.metal` file (headers in
//! `shaders/include`) is compiled with `xcrun metal` to AIR and linked with
//! `xcrun metallib` into one `.metallib`, which the crate embeds and loads at
//! first use; the cfg `midnight_metal_metallib` is set. Without the Metal
//! toolchain, or after a failed compile, the crate embeds the shader source
//! instead (all files as one translation unit, the `#include "..."` headers
//! expanded once each) and compiles it at first use with
//! `newLibraryWithSource`. The expanded source is always embedded, so a
//! metallib the running system refuses also falls back to it. A cargo warning
//! says which path was taken. Other targets build the stub (no shaders).
//!
//! Environment:
//! - `MIDNIGHT_METAL_SHADER_SOURCE=1`: skip the toolchain and embed the source.
//! - `MIDNIGHT_METAL_REQUIRE_METALLIB=1`: a missing toolchain or a failed
//!   compile is a build error.
use std::{
    collections::HashSet,
    env,
    path::{Path, PathBuf},
    process::Command,
};

/// The Metal Shading Language version of every shader (`-std=metal<V>`, and
/// `MTLCompileOptions.languageVersion` on the source path).
const MSL_VERSION: &str = "3.2";
/// The oldest macOS that runs MSL [`MSL_VERSION`].
const MACOS_MIN: &str = "15.0";

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let shaders = manifest.join("shaders");
    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-check-cfg=cfg(midnight_metal_metallib)");
    for v in [
        "MIDNIGHT_METAL_SHADER_SOURCE",
        "MIDNIGHT_METAL_REQUIRE_METALLIB",
        "DEVELOPER_DIR",
    ] {
        println!("cargo:rerun-if-env-changed={v}");
    }
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    println!("cargo:rustc-env=MIDNIGHT_METAL_MSL_VERSION={MSL_VERSION}");

    let mut files: Vec<PathBuf> = std::fs::read_dir(&shaders)
        .expect("metal/shaders exists")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "metal"))
        .collect();
    files.sort();
    let include = shaders.join("include");

    // The source path: one translation unit.
    let mut source = String::new();
    let mut seen = HashSet::new();
    for f in &files {
        expand(f, &include, &mut seen, &mut source);
    }
    std::fs::write(out.join("midnight_metal.metal"), &source).unwrap();

    let require = env::var("MIDNIGHT_METAL_REQUIRE_METALLIB").as_deref() == Ok("1");
    let fallback = |why: String| {
        if require {
            panic!("midnight-metal: {why} (MIDNIGHT_METAL_REQUIRE_METALLIB=1)");
        }
        println!(
            "cargo:warning=midnight-metal: {why}; embedding the shader source, compiled with \
             newLibraryWithSource at first use"
        );
    };
    if env::var("MIDNIGHT_METAL_SHADER_SOURCE").as_deref() == Ok("1") {
        return fallback("MIDNIGHT_METAL_SHADER_SOURCE=1".into());
    }
    match metallib(&files, &include, &out) {
        Ok(()) => {
            println!("cargo:rustc-cfg=midnight_metal_metallib");
            println!(
                "cargo:warning=midnight-metal: compiled {} shader file(s) to a metallib with \
                 xcrun metal (MSL {MSL_VERSION}, macOS {MACOS_MIN}+)",
                files.len()
            );
        }
        Err(why) => fallback(why),
    }
}

/// Compiles each file to AIR and links them into `midnight_metal.metallib`.
fn metallib(files: &[PathBuf], include: &Path, out: &Path) -> Result<(), String> {
    let probe = Command::new("xcrun").args(["-sdk", "macosx", "-f", "metal"]).output();
    if !probe.as_ref().is_ok_and(|o| o.status.success()) {
        return Err(
            "the Metal toolchain was not found (`xcrun -sdk macosx -f metal` failed)".into(),
        );
    }
    let mut airs = Vec::new();
    for f in files {
        let air = out.join(f.with_extension("air").file_name().unwrap());
        let o = Command::new("xcrun")
            .args(["-sdk", "macosx", "metal"])
            .arg(format!("-std=metal{MSL_VERSION}"))
            .arg(format!("-mmacosx-version-min={MACOS_MIN}"))
            .arg("-I")
            .arg(include)
            .arg("-c")
            .arg(f)
            .arg("-o")
            .arg(&air)
            .output()
            .map_err(|e| format!("xcrun metal could not run: {e}"))?;
        if !o.status.success() {
            return Err(format!(
                "xcrun metal failed on {}: {}",
                f.display(),
                String::from_utf8_lossy(&o.stderr).replace('\n', " | ")
            ));
        }
        airs.push(air);
    }
    let lib = out.join("midnight_metal.metallib");
    let o = Command::new("xcrun")
        .args(["-sdk", "macosx", "metallib"])
        .args(&airs)
        .arg("-o")
        .arg(&lib)
        .output()
        .map_err(|e| format!("xcrun metallib could not run: {e}"))?;
    if !o.status.success() {
        return Err(format!(
            "xcrun metallib failed: {}",
            String::from_utf8_lossy(&o.stderr).replace('\n', " | ")
        ));
    }
    Ok(())
}

/// Appends `file` to `out` with each `#include "name"` replaced by the named
/// header (looked up next to the file, then in `include`) the first time it
/// is met, and dropped after that. `#line` directives keep the compiler's
/// messages pointing at the original files.
fn expand(file: &Path, include: &Path, seen: &mut HashSet<PathBuf>, out: &mut String) {
    let file = file.canonicalize().unwrap_or_else(|_| file.to_path_buf());
    if !seen.insert(file.clone()) {
        return;
    }
    let text = std::fs::read_to_string(&file)
        .unwrap_or_else(|e| panic!("midnight-metal: cannot read {}: {e}", file.display()));
    out.push_str(&format!("#line 1 \"{}\"\n", file.display()));
    for (n, line) in text.lines().enumerate() {
        let quoted = line
            .trim_start()
            .strip_prefix("#include")
            .map(str::trim)
            .and_then(|r| r.strip_prefix('"'))
            .and_then(|r| r.split('"').next());
        match quoted {
            Some(name) => {
                let local = file.parent().unwrap().join(name);
                let header = if local.is_file() {
                    local
                } else {
                    include.join(name)
                };
                if !header.is_file() {
                    panic!(
                        "midnight-metal: {} includes missing \"{name}\"",
                        file.display()
                    );
                }
                expand(&header, include, seen, out);
                out.push_str(&format!("#line {} \"{}\"\n", n + 2, file.display()));
            }
            None => {
                out.push_str(line);
                out.push('\n');
            }
        }
    }
}
