//! Build script that compiles the C library through CMake, the one place that probes which ISA
//! capabilities the compiler supports.
//!
//! File: build.rs
//! Author: Ash Vardanian
use std::env;
use std::path::{Path, PathBuf};

/// The Cargo features that switch a CMake option, each passed as ON or OFF, never left to the host.
const FEATURE_OPTIONS: &[(&str, &str)] = &[("CUDA", "NUMKONG_BUILD_CUDA"), ("ROCM", "NUMKONG_BUILD_ROCM")];

/// Rebuilds when a directory's contents change, recursing as Cargo watches only what it is told.
fn watch(directory: &Path) {
    println!("cargo:rerun-if-changed={}", directory.display());
    for entry in std::fs::read_dir(directory).into_iter().flatten().flatten() {
        let path = entry.path();
        match path.is_dir() {
            true => watch(&path),
            false => println!("cargo:rerun-if-changed={}", path.display()),
        }
    }
}

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let enabled = |feature: &str| env::var_os(format!("CARGO_FEATURE_{feature}")).is_some();

    // A directory holding `numkong_static` that CMake already built, like a parent project's
    // build tree or a release, skips the build here.
    println!("cargo:rerun-if-env-changed=NUMKONG_LIBRARY_DIR");
    let build = match env::var_os("NUMKONG_LIBRARY_DIR") {
        Some(directory) => PathBuf::from(directory),
        None => build_library(&manifest, enabled),
    };
    // Straight in the directory, or one configuration deeper on the multi-configuration generators.
    println!("cargo:rustc-link-search=native={}", build.display());
    println!("cargo:rustc-link-search=native={}", build.join("Release").display());
    // `rust/*.rs` link `numkong`, which names the archive CMake calls `numkong_static`.
    println!("cargo:rustc-link-lib=static=numkong:numkong_static");
    println!("cargo:include={}", manifest.join("include").display());

    if enabled("CUDA") || enabled("ROCM") {
        // The runtimes come from wherever CMake found the toolkits it compiled the units with, and
        // a prebuilt archive without its build tree leaves them to the linker's own search path.
        let cache = std::fs::read_to_string(build.join("CMakeCache.txt")).unwrap_or_default();
        // The directory `levels` above a path CMake cached.
        let search = |key: &str, levels: usize| {
            let cached = cache.lines().find_map(|line| line.strip_prefix(key));
            if let Some(directory) = cached.and_then(|path| Path::new(path).ancestors().nth(levels)) {
                println!("cargo:rustc-link-search=native={}", directory.display());
            }
        };
        if enabled("CUDA") {
            search("CUDA_cudart_LIBRARY:FILEPATH=", 1);
            println!("cargo:rustc-link-lib=cudart");
        }
        if enabled("ROCM") {
            // HIP's package lives in `<rocm>/lib/cmake/hip`, two levels under `libamdhip64`.
            search("hip_DIR:PATH=", 2);
            println!("cargo:rustc-link-lib=amdhip64");
        }
        // The GPU units' host side is C++, reaching its runtime for guarded statics and unwinding.
        if env::var("CARGO_CFG_TARGET_ENV").unwrap() != "msvc" {
            println!("cargo:rustc-link-lib=stdc++");
        }
    }
}

/// Builds `numkong_static` through CMake and returns the directory holding it.
fn build_library(manifest: &Path, enabled: impl Fn(&str) -> bool) -> PathBuf {
    println!("cargo:rerun-if-changed={}", manifest.join("CMakeLists.txt").display());
    for directory in ["cmake", "c", "include", "probes"] {
        watch(&manifest.join(directory));
    }

    let mut library = cmake::Config::new(manifest);
    // The crate reconfigures on every invocation by default, which reruns every ISA probe before a
    // build that has nothing to do.
    library
        .profile("Release")
        .define("NUMKONG_INSTALL", "OFF")
        .define("NUMKONG_BUILD_SHARED", "OFF")
        .build_target("numkong_static")
        .always_configure(false);
    for (feature, option) in FEATURE_OPTIONS {
        library.define(option, if enabled(feature) { "ON" } else { "OFF" });
    }

    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    if arch == "wasm32" || arch == "wasm64" {
        // An engine validates a module whole, so its one SIMD capability is the one the Rust
        // target declares.
        let target_features = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
        let declares = |feature: &str| target_features.split(',').any(|declared| declared == feature);
        let capability = match (declares("relaxed-simd"), declares("simd128")) {
            (true, _) => "v128relaxed",
            (false, true) => "v128",
            (false, false) => "serial",
        };
        library.define("NUMKONG_TARGET_ARCH", capability);
        // The checked-in toolchains name the SDK and its sysroot, which `cmake-rs` cannot, while
        // `unknown` needs no libc at all, so it cross-compiles with the clang `cmake-rs` picks.
        println!("cargo:rerun-if-env-changed=WASI_SDK_PATH");
        println!("cargo:rerun-if-env-changed=EMSDK");
        let target = env::var("TARGET").unwrap();
        let toolchain = match env::var("CARGO_CFG_TARGET_OS").unwrap().as_str() {
            "wasi" if target.ends_with("-threads") => Some("toolchain-wasm32-wasi-threads.cmake"),
            "wasi" => Some("toolchain-wasm32-wasi.cmake"),
            "emscripten" => Some("toolchain-wasm32-emscripten.cmake"),
            "unknown" | "none" => None,
            os => panic!("NumKong has no CMake toolchain for WebAssembly on `{os}`, the target `{target}`"),
        };
        match toolchain {
            Some(file) => library.define("CMAKE_TOOLCHAIN_FILE", manifest.join("cmake").join(file)),
            None => library
                .define("CMAKE_SYSTEM_NAME", "Generic")
                .define("CMAKE_SYSTEM_PROCESSOR", &arch)
                .define("CMAKE_TRY_COMPILE_TARGET_TYPE", "STATIC_LIBRARY"),
        };
    }

    // `build_target` stops short of installing, so the archive sits where the generator left it.
    library.build().join("build")
}
