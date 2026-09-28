//! Build script that compiles the C kernels and probes which ISA tiers the compiler supports.
//!
//! File: build.rs
//! Author: Ash Vardanian
use std::collections::HashMap;
use std::env;
use std::path::Path;

fn main() { build_numkong().expect("Failed to build NumKong"); }

/// Try to compile a single probe .c file with the given flags. Uses `flag()`, a hard error, instead
/// of `flag_if_supported()` to avoid silently dropping flags the compiler doesn't recognize.
fn probe_isa(probe_file: &str, flags: &[&str]) -> bool {
    let mut build = cc::Build::new();
    build
        .file(probe_file)
        .cargo_metadata(false)
        .warnings(false)
        .opt_level(0);
    for flag in flags {
        build.flag(flag);
    }
    let name = probe_file.replace("probes/", "probe_").replace(".c", "");
    build.try_compile(&name).is_ok()
}

/// Recursively collect all files under a directory for cargo:rerun-if-changed.
fn watch_dir(dir: &str) {
    let path = Path::new(dir);
    if !path.is_dir() {
        return;
    }
    println!("cargo:rerun-if-changed={dir}");
    for entry in std::fs::read_dir(path).into_iter().flatten().flatten() {
        let p = entry.path();
        if p.is_dir() {
            watch_dir(&p.to_string_lossy());
        } else {
            println!("cargo:rerun-if-changed={}", p.display());
        }
    }
}

struct IsaProbe {
    name: &'static str,
    probe_file: &'static str,
    gcc_flags: &'static [&'static str],
    msvc_flags: &'static [&'static str],
}

// x86 probes: GCC flags are minimal — each implies its prerequisites.
// E.g., -mavx512vnni implies -mavx512f; -mavxvnni implies -mavx2.
const X86_PROBES: &[IsaProbe] = &[
    IsaProbe {
        name: "NUMKONG_TARGET_HASWELL",
        probe_file: "probes/x86_haswell.c",
        gcc_flags: &["-mavx2", "-mfma", "-mf16c"], // all 3 are independent
        msvc_flags: &["/arch:AVX2"],
    },
    IsaProbe {
        name: "NUMKONG_TARGET_SKYLAKE",
        probe_file: "probes/x86_skylake.c",
        gcc_flags: &["-mavx512f", "-mavx512bw", "-mavx512dq", "-mavx512vl"], // 4 independent sub-features
        msvc_flags: &["/arch:AVX512"],
    },
    IsaProbe {
        name: "NUMKONG_TARGET_ICELAKE",
        probe_file: "probes/x86_icelake.c",
        gcc_flags: &["-mavx512vnni", "-mavx512vl"], // vnni implies F
        msvc_flags: &["/arch:AVX512"],
    },
    IsaProbe {
        name: "NUMKONG_TARGET_GENOA",
        probe_file: "probes/x86_genoa.c",
        gcc_flags: &["-mavx512bf16", "-mavx512vl"], // bf16 implies F+BW
        msvc_flags: &["/arch:AVX512"],
    },
    IsaProbe {
        name: "NUMKONG_TARGET_SAPPHIRE",
        probe_file: "probes/x86_sapphire.c",
        gcc_flags: &["-mavx512fp16", "-mavx512vl"], // fp16 implies F+BW
        msvc_flags: &["/arch:AVX512"],
    },
    IsaProbe {
        name: "NUMKONG_TARGET_SAPPHIREAMX",
        probe_file: "probes/x86_sapphireamx.c",
        gcc_flags: &["-mamx-tile", "-mamx-int8"],
        msvc_flags: &["/arch:AVX512"],
    },
    IsaProbe {
        name: "NUMKONG_TARGET_GRANITEAMX",
        probe_file: "probes/x86_graniteamx.c",
        gcc_flags: &["-mamx-tile", "-mamx-fp16"],
        msvc_flags: &["/arch:AVX512"],
    },
    IsaProbe {
        name: "NUMKONG_TARGET_DIAMOND",
        probe_file: "probes/x86_diamond.c",
        gcc_flags: &["-mavx10.2-512"], // implies all AVX-512 + FP16 + AVX10.1
        msvc_flags: &["/arch:AVX10.2"],
    },
    IsaProbe {
        name: "NUMKONG_TARGET_DIAMONDAMX",
        probe_file: "probes/x86_diamondamx.c",
        gcc_flags: &["-mamx-tile", "-mamx-fp8", "-mamx-avx512", "-mavx10.2"],
        msvc_flags: &["/arch:AVX10.2"],
    },
    IsaProbe {
        name: "NUMKONG_TARGET_TURIN",
        probe_file: "probes/x86_turin.c",
        gcc_flags: &["-mavx512vp2intersect"], // implies F+DQ
        msvc_flags: &["/arch:AVX512"],
    },
    IsaProbe {
        name: "NUMKONG_TARGET_ALDER",
        probe_file: "probes/x86_alder.c",
        gcc_flags: &["-mavxvnni"], // implies AVX2
        msvc_flags: &["/arch:AVX2"],
    },
    IsaProbe {
        name: "NUMKONG_TARGET_SIERRA",
        probe_file: "probes/x86_sierra.c",
        gcc_flags: &["-mavxvnniint8"], // implies AVX2
        msvc_flags: &["/arch:AVX2"],
    },
];

// ARM probes: msvc_flags are empty because MSVC does not define __ARM_FEATURE_* macros via /arch:
// flags. For MSVC header-only builds, types.h infers features from __ARM_ARCH level instead.
// SVE/SME probes also have #error guards for _WIN32.
const ARM_PROBES: &[IsaProbe] = &[
    // FEAT_AdvSIMD, baseline ARM64
    IsaProbe {
        name: "NUMKONG_TARGET_NEON",
        probe_file: "probes/arm_neon.c",
        gcc_flags: &["-march=armv8-a+simd"],
        msvc_flags: &[],
    },
    // FEAT_FP16: optional from ARMv8.2, mandatory at ARMv9.0 with AdvSIMD
    IsaProbe {
        name: "NUMKONG_TARGET_NEONHALF",
        probe_file: "probes/arm_neon_half.c",
        gcc_flags: &["-march=armv8.2-a+simd+fp16"],
        msvc_flags: &["/arch:armv8.2"],
    },
    // FEAT_DotProd: optional from ARMv8.1, mandatory at ARMv8.4 with AdvSIMD
    IsaProbe {
        name: "NUMKONG_TARGET_NEONSDOT",
        probe_file: "probes/arm_neon_sdot.c",
        gcc_flags: &["-march=armv8.2-a+dotprod"],
        msvc_flags: &["/arch:armv8.4"],
    },
    // FEAT_BF16: optional from ARMv8.2, mandatory at ARMv8.6 with FP
    IsaProbe {
        name: "NUMKONG_TARGET_NEONBFDOT",
        probe_file: "probes/arm_neon_bfdot.c",
        gcc_flags: &["-march=armv8.6-a+simd+bf16"],
        msvc_flags: &["/arch:armv8.6"],
    },
    // FEAT_FHM: optional from ARMv8.1, mandatory at ARMv8.4 with FP16
    IsaProbe {
        name: "NUMKONG_TARGET_NEONFHM",
        probe_file: "probes/arm_neon_fhm.c",
        gcc_flags: &["-march=armv8.2-a+simd+fp16+fp16fml"],
        msvc_flags: &["/arch:armv8.4"],
    },
    IsaProbe {
        name: "NUMKONG_TARGET_SVE",
        probe_file: "probes/arm_sve.c",
        gcc_flags: &["-march=armv8.2-a+sve"],
        msvc_flags: &[],
    },
    IsaProbe {
        name: "NUMKONG_TARGET_SVEHALF",
        probe_file: "probes/arm_sve_half.c",
        gcc_flags: &["-march=armv8.2-a+sve+fp16"],
        msvc_flags: &[],
    },
    IsaProbe {
        name: "NUMKONG_TARGET_SVEBFDOT",
        probe_file: "probes/arm_sve_bfdot.c",
        gcc_flags: &["-march=armv8.2-a+sve+bf16"],
        msvc_flags: &[],
    },
    IsaProbe {
        name: "NUMKONG_TARGET_SVESDOT",
        probe_file: "probes/arm_sve_sdot.c",
        gcc_flags: &["-march=armv8.2-a+sve+dotprod"],
        msvc_flags: &[],
    },
    IsaProbe {
        name: "NUMKONG_TARGET_SVE2",
        probe_file: "probes/arm_sve2.c",
        gcc_flags: &["-march=armv8.2-a+sve2"],
        msvc_flags: &[],
    },
    IsaProbe {
        name: "NUMKONG_TARGET_NEONFP8",
        probe_file: "probes/arm_neonfp8.c",
        gcc_flags: &["-march=armv8-a+simd+fp8dot4"],
        msvc_flags: &[],
    },
    IsaProbe {
        name: "NUMKONG_TARGET_SME",
        probe_file: "probes/arm_sme.c",
        gcc_flags: &["-march=armv8-a+sme"],
        msvc_flags: &[],
    },
    IsaProbe {
        name: "NUMKONG_TARGET_SMEF64",
        probe_file: "probes/arm_sme_f64.c",
        gcc_flags: &["-march=armv8-a+sme+sme-f64f64"],
        msvc_flags: &[],
    },
    IsaProbe {
        name: "NUMKONG_TARGET_SMEBI32",
        probe_file: "probes/arm_sme_bi32.c",
        gcc_flags: &["-march=armv8-a+sme2"],
        msvc_flags: &[],
    },
];

const RISCV_PROBES: &[IsaProbe] = &[
    IsaProbe {
        name: "NUMKONG_TARGET_RVV",
        probe_file: "probes/riscv_rvv.c",
        gcc_flags: &["-march=rv64gcv"],
        msvc_flags: &[],
    },
    IsaProbe {
        name: "NUMKONG_TARGET_RVVHALF",
        probe_file: "probes/riscv_rvv_half.c",
        gcc_flags: &["-march=rv64gcv_zvfh"],
        msvc_flags: &[],
    },
    IsaProbe {
        name: "NUMKONG_TARGET_RVVBF16",
        probe_file: "probes/riscv_rvv_bf16.c",
        gcc_flags: &["-march=rv64gcv_zvfbfwma"],
        msvc_flags: &[],
    },
    IsaProbe {
        name: "NUMKONG_TARGET_RVVBB",
        probe_file: "probes/riscv_rvv_bb.c",
        gcc_flags: &["-march=rv64gcv_zvbb"],
        msvc_flags: &[],
    },
];

const LOONGARCH_PROBES: &[IsaProbe] = &[IsaProbe {
    name: "NUMKONG_TARGET_LOONGSONASX",
    probe_file: "probes/loongarch_lasx.c",
    gcc_flags: &["-mlasx"],
    msvc_flags: &[],
}];

const POWER_PROBES: &[IsaProbe] = &[IsaProbe {
    name: "NUMKONG_TARGET_POWERVSX",
    probe_file: "probes/power_vsx.c",
    gcc_flags: &["-mcpu=power9", "-mvsx"],
    msvc_flags: &[],
}];

const WASM_PROBES: &[IsaProbe] = &[
    IsaProbe {
        name: "NUMKONG_TARGET_V128",
        probe_file: "probes/wasm_v128.c",
        gcc_flags: &["-msimd128"],
        msvc_flags: &[],
    },
    IsaProbe {
        name: "NUMKONG_TARGET_V128RELAXED",
        probe_file: "probes/wasm_v128relaxed.c",
        gcc_flags: &["-msimd128", "-mrelaxed-simd"],
        msvc_flags: &[],
    },
];

fn build_numkong() -> Result<HashMap<String, bool>, String> {
    let mut flags = HashMap::<String, bool>::new();
    let mut build = cc::Build::new();

    // Source files
    build
        // Prefer portable flags to support MSVC and older toolchains
        .std("c11") // The library uses C11 atomics; its headers stay C99
        .file("c/numkong.c")
        // Family dispatch points
        .file("c/dispatch/attention.c")
        .file("c/dispatch/cast.c")
        .file("c/dispatch/curved.c")
        .file("c/dispatch/dot.c")
        .file("c/dispatch/dots.c")
        .file("c/dispatch/each.c")
        .file("c/dispatch/geospatial.c")
        .file("c/dispatch/maxsim.c")
        .file("c/dispatch/mesh.c")
        .file("c/dispatch/probability.c")
        .file("c/dispatch/reduce.c")
        .file("c/dispatch/scalar.c")
        .file("c/dispatch/set.c")
        .file("c/dispatch/sets.c")
        .file("c/dispatch/sparse.c")
        .file("c/dispatch/spatial.c")
        .file("c/dispatch/spatials.c")
        .file("c/dispatch/trigonometry.c")
        // Capability kernels, each compiling to nothing off its architecture
        .file("c/cpu/serial.c")
        .file("c/cpu/haswell.c")
        .file("c/cpu/alder.c")
        .file("c/cpu/sierra.c")
        .file("c/cpu/skylake.c")
        .file("c/cpu/icelake.c")
        .file("c/cpu/genoa.c")
        .file("c/cpu/turin.c")
        .file("c/cpu/sapphire.c")
        .file("c/cpu/diamond.c")
        .file("c/cpu/sapphireamx.c")
        .file("c/cpu/graniteamx.c")
        .file("c/cpu/diamondamx.c")
        .file("c/cpu/neon.c")
        .file("c/cpu/neonhalf.c")
        .file("c/cpu/neonbfdot.c")
        .file("c/cpu/neonfhm.c")
        .file("c/cpu/neonsdot.c")
        .file("c/cpu/neonfp8.c")
        .file("c/cpu/sve.c")
        .file("c/cpu/svehalf.c")
        .file("c/cpu/svesdot.c")
        .file("c/cpu/svebfdot.c")
        .file("c/cpu/sve2.c")
        .file("c/cpu/sme.c")
        .file("c/cpu/smef64.c")
        .file("c/cpu/smebi32.c")
        .file("c/cpu/rvv.c")
        .file("c/cpu/rvvhalf.c")
        .file("c/cpu/rvvbf16.c")
        .file("c/cpu/rvvbb.c")
        .file("c/cpu/v128.c")
        .file("c/cpu/v128relaxed.c")
        .file("c/cpu/powervsx.c")
        .file("c/cpu/loongsonasx.c")
        .include("include")
        .include("c")
        .define("NUMKONG_NATIVE_F16", "0")
        .define("NUMKONG_NATIVE_BF16", "0")
        .opt_level(3)
        .flag_if_supported("-pedantic") // Strict compliance when supported
        .flag_if_supported("/experimental:c11atomics") // MSVC's <stdatomic.h>, which `c/numkong.c` caches detection in
        .flag_if_supported("-Wno-psabi") // Suppress GCC ABI note for 32-byte aligned params
        .warnings(false);

    // Architecture detection
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let is_msvc = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default() == "msvc";
    let target_features = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();

    let is_wasm = target_arch == "wasm32" || target_arch == "wasm64";

    // On 32-bit x86, ensure proper stack alignment for floating-point operations
    // See: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=38534
    if target_arch == "x86" {
        build.flag_if_supported("-mstackrealign");
        build.flag_if_supported("-mpreferred-stack-boundary=4");
    }

    // Pin TU baseline to each arch's ABI floor; SIMD kernels carry per-function pragmas.
    // `NUMKONG_TARGET_ARCH=native` opts into a host-tuned, non-portable build, ignored on MSVC.
    // Keep per-arch table in sync with CMakeLists.txt, setup.py, binding.gyp.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let is_apple = target_os == "macos" || target_os == "ios";
    let is_cross = env::var("HOST").unwrap_or_default() != env::var("TARGET").unwrap_or_default();
    let march_native = env::var("NUMKONG_TARGET_ARCH").is_ok_and(|v| v == "native");
    // Portable baseline: pin TU ISA floor + forbid auto-vectorization so serial
    // fallbacks don't get silently promoted to NEON/SSE2/VSX. SIMD kernels use
    // explicit intrinsics; unaffected. MSVC has no command-line vectorizer
    // toggle; `NUMKONG_TARGET_ARCH=native` opts out for host-tuned builds.
    if march_native && !is_msvc && !is_cross {
        println!("cargo:warning=NUMKONG_TARGET_ARCH=native: building host-tuned, result will not run on older CPUs");
        // Apple Clang's `-march=native` advertises only a subset of host features
        // (no SME/SME2/FP16_FML); `-mcpu=native` is the complete knob on macOS.
        build.flag_if_supported(if is_apple { "-mcpu=native" } else { "-march=native" });
    } else if is_msvc {
        match target_arch.as_str() {
            "x86_64" => {
                build.flag_if_supported("/arch:SSE2");
            }
            "aarch64" => {
                build.flag_if_supported("/arch:armv8.0");
            }
            _ => {}
        }
    } else if is_apple {
        // `-arch` already pins the ABI floor per slice, and a universal build passes both
        // to one clang invocation, where a per-arch `-march=` conflicts with the other slice.
        build.flag_if_supported("-fno-tree-vectorize");
        build.flag_if_supported("-fno-tree-slp-vectorize");
    } else {
        build.flag_if_supported("-fno-tree-vectorize");
        build.flag_if_supported("-fno-tree-slp-vectorize");
        match target_arch.as_str() {
            "x86_64" => {
                build.flag_if_supported("-march=x86-64");
            }
            "aarch64" => {
                build.flag_if_supported("-march=armv8-a");
            }
            "riscv64" => {
                build.flag_if_supported("-march=rv64gc");
            }
            "powerpc64" => {
                build.flag_if_supported("-mcpu=power8");
            }
            "loongarch64" => {
                // LASX must be enabled at TU level: GCC's LoongArch backend
                // doesn't support per-function `target` attribute / pragma
                // until GCC 15 (Feb 2025), and Clang until 22.1 (May 2025).
                build.flag_if_supported("-march=loongarch64");
                build.flag_if_supported("-mlasx");
            }
            _ => {}
        }
    }

    // Select probe tables for this architecture
    let probe_tables: &[&[IsaProbe]] = match target_arch.as_str() {
        "x86_64" => &[X86_PROBES],
        "aarch64" => &[ARM_PROBES],
        "riscv64" => &[RISCV_PROBES],
        "loongarch64" => &[LOONGARCH_PROBES],
        "powerpc64" => &[POWER_PROBES],
        "wasm32" | "wasm64" => &[WASM_PROBES],
        _ => &[],
    };

    // Probe each ISA, and on WASM only the tiers the Rust target declares
    for table in probe_tables {
        for probe in table.iter() {
            // Environment override: NUMKONG_TARGET_FOO=0 forces off, NUMKONG_TARGET_FOO=1 forces on
            if let Ok(val) = env::var(probe.name) {
                let forced = match val.as_str() {
                    "1" | "true" | "TRUE" => Some(true),
                    "0" | "false" | "FALSE" => Some(false),
                    _ => None,
                };
                if let Some(on) = forced {
                    build.define(probe.name, if on { "1" } else { "0" });
                    flags.insert(probe.name.to_string(), on);
                    let verb = if on { "enabled" } else { "disabled" };
                    println!("cargo:warning={}: force-{verb} via environment", probe.name);
                    continue;
                }
            }

            let probe_flags = if is_msvc { probe.msvc_flags } else { probe.gcc_flags };
            // An engine validates a WebAssembly module whole, so a tier the Rust target does not declare stays off.
            let declared = !is_wasm
                || probe_flags.iter().all(|flag| {
                    let feature = flag.strip_prefix("-m");
                    target_features
                        .split(',')
                        .any(|target_feature| Some(target_feature) == feature)
                });
            let ok = declared && probe_isa(probe.probe_file, probe_flags);
            build.define(probe.name, if ok { "1" } else { "0" });
            flags.insert(probe.name.to_string(), ok);
            if declared && !ok {
                println!("cargo:warning={}: not supported by compiler", probe.name);
            }
        }
    }

    // WebAssembly selects its SIMD tier with whole-module flags rather than per-function `target`
    // attributes, so the enabled tier goes onto the compiler invocation; `types.h` reads the
    // `__wasm_simd128__` / `__wasm_relaxed_simd__` keys the flags define.
    if is_wasm {
        if *flags.get("NUMKONG_TARGET_V128RELAXED").unwrap_or(&false) {
            build.flag("-msimd128").flag("-mrelaxed-simd");
        } else if *flags.get("NUMKONG_TARGET_V128").unwrap_or(&false) {
            build.flag("-msimd128");
        }
    }

    // Compile
    build.compile("numkong");

    // Expose the include directory so dependents can find <numkong/numkong.h>
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    println!("cargo:include={}/include", manifest_dir);

    // Watch directories recursively instead of listing individual files
    watch_dir("c");
    watch_dir("include");
    watch_dir("probes");

    // Rerun on env var changes
    println!("cargo:rerun-if-env-changed=NUMKONG_TARGET_ARCH");
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_FEATURE");
    for table in [
        X86_PROBES,
        ARM_PROBES,
        RISCV_PROBES,
        LOONGARCH_PROBES,
        POWER_PROBES,
        WASM_PROBES,
    ] {
        for probe in table {
            println!("cargo:rerun-if-env-changed={}", probe.name);
        }
    }

    Ok(flags)
}
