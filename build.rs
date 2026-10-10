//! Build script that compiles the C library through CMake, the one place that probes which ISA
//! capabilities the compiler supports.
//!
//! File: build.rs
//! Author: Ash Vardanian
use std::{
    env,
    path::{Path, PathBuf},
};

/// The Cargo features that switch a CMake option, each passed as ON or OFF, never left to the host.
const FEATURE_OPTIONS: &[(&str, &str)] = &[("CUDA", "NUMKONG_BUILD_CUDA"), ("ROCM", "NUMKONG_BUILD_ROCM")];

/// CMake cache variables read from the environment, like `NUMKONG_CUDA_ARCHITECTURES=120f-real`.
const ENVIRONMENT_VARIABLES: &[&str] = &["NUMKONG_CUDA_ARCHITECTURES", "NUMKONG_ROCM_ARCHITECTURES"];

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let enabled = |feature: &str| env::var_os(format!("CARGO_FEATURE_{feature}")).is_some();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap();

    // A directory holding `numkong_static` that CMake already built, like a parent project's
    // build tree or a release, skips the build here. Pass the archive's exact directory.
    println!("cargo:rerun-if-env-changed=NUMKONG_LIBRARY_DIR");
    let (library, build) = match env::var_os("NUMKONG_LIBRARY_DIR") {
        Some(directory) => {
            let directory = PathBuf::from(directory);
            (directory.clone(), directory)
        }
        None => {
            let installed = build_library(&manifest, enabled);
            (installed.join("lib"), installed.join("build"))
        }
    };
    // Watch the archive itself so replacing a prebuilt library relinks the crate.
    let archive_name = if target_env == "msvc" {
        "numkong_static.lib"
    } else {
        "libnumkong_static.a"
    };
    let archive = library.join(archive_name);
    println!("cargo:rerun-if-changed={}", archive.display());
    println!("cargo:rustc-link-search=native={}", library.display());
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
        if target_env != "msvc" {
            println!("cargo:rustc-link-lib=stdc++");
        }
    }
}

/// Builds and installs `numkong_static` through CMake and returns the installation prefix.
fn build_library(manifest: &Path, enabled: impl Fn(&str) -> bool) -> PathBuf {
    for source in ["CMakeLists.txt", "VERSION", "cmake", "c", "include", "probes"] {
        println!("cargo:rerun-if-changed={}", manifest.join(source).display());
    }

    let mut library = cmake::Config::new(manifest);
    library
        .profile("Release")
        .define("NUMKONG_INSTALL", "ON")
        .define("CMAKE_INSTALL_LIBDIR", "lib")
        .define("NUMKONG_BUILD_TEST", "OFF")
        .define("NUMKONG_BUILD_BENCH", "OFF")
        .define("NUMKONG_BUILD_SHARED", "OFF");
    for (feature, option) in FEATURE_OPTIONS {
        library.define(option, if enabled(feature) { "ON" } else { "OFF" });
    }
    // The crate reconfigures on every invocation by default, which reruns every ISA probe before a
    // build that has nothing to do, but a variable from the environment may have changed since.
    let mut from_environment = false;
    for variable in ENVIRONMENT_VARIABLES {
        println!("cargo:rerun-if-env-changed={variable}");
        if let Ok(value) = env::var(variable) {
            library.define(variable, value);
            from_environment = true;
        }
    }
    library.always_configure(from_environment);

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

    // The default install target resolves configuration-specific archive paths in CMake.
    library.build()
}
