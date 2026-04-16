use std::collections::HashMap;
use std::env;
use std::path::Path;

fn main() {
    build_numkong().expect("Failed to build NumKong");
}

/// Try to compile a single probe .c file with the given flags.
/// Uses `flag()` (hard error) instead of `flag_if_supported()` to avoid
/// silently dropping flags the compiler doesn't recognize.
fn probe_isa(probe_file: &str, flags: &[&str]) -> bool {
    let mut build = cc::Build::new();
    build.file(probe_file).warnings(false).opt_level(0);
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
        name: "NK_TARGET_HASWELL",
        probe_file: "probes/x86_haswell.c",
        gcc_flags: &["-mavx2", "-mfma", "-mf16c"], // all 3 are independent
        msvc_flags: &["/arch:AVX2"],
    },
    IsaProbe {
        name: "NK_TARGET_SKYLAKE",
        probe_file: "probes/x86_skylake.c",
        gcc_flags: &["-mavx512f", "-mavx512bw", "-mavx512dq", "-mavx512vl"], // 4 independent sub-features
        msvc_flags: &["/arch:AVX512"],
    },
    IsaProbe {
        name: "NK_TARGET_ICELAKE",
        probe_file: "probes/x86_icelake.c",
        gcc_flags: &["-mavx512vnni", "-mavx512vl"], // vnni implies F
        msvc_flags: &["/arch:AVX512"],
    },
    IsaProbe {
        name: "NK_TARGET_GENOA",
        probe_file: "probes/x86_genoa.c",
        gcc_flags: &["-mavx512bf16", "-mavx512vl"], // bf16 implies F+BW
        msvc_flags: &["/arch:AVX512"],
    },
    IsaProbe {
        name: "NK_TARGET_SAPPHIRE",
        probe_file: "probes/x86_sapphire.c",
        gcc_flags: &["-mavx512fp16", "-mavx512vl"], // fp16 implies F+BW
        msvc_flags: &["/arch:AVX512"],
    },
    IsaProbe {
        name: "NK_TARGET_SAPPHIREAMX",
        probe_file: "probes/x86_sapphireamx.c",
        gcc_flags: &["-mamx-tile", "-mamx-int8"],
        msvc_flags: &["/arch:AVX512"],
    },
    IsaProbe {
        name: "NK_TARGET_GRANITEAMX",
        probe_file: "probes/x86_graniteamx.c",
        gcc_flags: &["-mamx-tile", "-mamx-fp16"],
        msvc_flags: &["/arch:AVX512"],
    },
    IsaProbe {
        name: "NK_TARGET_DIAMOND",
        probe_file: "probes/x86_diamond.c",
        gcc_flags: &["-mavx10.2-512"], // implies all AVX-512 + FP16 + AVX10.1
        msvc_flags: &["/arch:AVX10.2"],
    },
    IsaProbe {
        name: "NK_TARGET_TURIN",
        probe_file: "probes/x86_turin.c",
        gcc_flags: &["-mavx512vp2intersect"], // implies F+DQ
        msvc_flags: &["/arch:AVX512"],
    },
    IsaProbe {
        name: "NK_TARGET_ALDER",
        probe_file: "probes/x86_alder.c",
        gcc_flags: &["-mavxvnni"], // implies AVX2
        msvc_flags: &["/arch:AVX2"],
    },
    IsaProbe {
        name: "NK_TARGET_SIERRA",
        probe_file: "probes/x86_sierra.c",
        gcc_flags: &["-mavxvnniint8"], // implies AVX2
        msvc_flags: &["/arch:AVX2"],
    },
];

// ARM probes: msvc_flags are empty because MSVC does not define __ARM_FEATURE_*
// macros via /arch: flags. For MSVC header-only builds, types.h infers features
// from __ARM_ARCH level instead. SVE/SME probes also have #error guards for _WIN32.
const ARM_PROBES: &[IsaProbe] = &[
    // FEAT_AdvSIMD (baseline ARM64)
    IsaProbe {
        name: "NK_TARGET_NEON",
        probe_file: "probes/arm_neon.c",
        gcc_flags: &["-march=armv8-a+simd"],
        msvc_flags: &[],
    },
    // FEAT_FP16: optional from ARMv8.2, mandatory at ARMv9.0 with AdvSIMD
    IsaProbe {
        name: "NK_TARGET_NEONHALF",
        probe_file: "probes/arm_neon_half.c",
        gcc_flags: &["-march=armv8.2-a+simd+fp16"],
        msvc_flags: &["/arch:armv8.2"],
    },
    // FEAT_DotProd: optional from ARMv8.1, mandatory at ARMv8.4 with AdvSIMD
    IsaProbe {
        name: "NK_TARGET_NEONSDOT",
        probe_file: "probes/arm_neon_sdot.c",
        gcc_flags: &["-march=armv8.2-a+dotprod"],
        msvc_flags: &["/arch:armv8.4"],
    },
    // FEAT_BF16: optional from ARMv8.2, mandatory at ARMv8.6 with FP
    IsaProbe {
        name: "NK_TARGET_NEONBFDOT",
        probe_file: "probes/arm_neon_bfdot.c",
        gcc_flags: &["-march=armv8.6-a+simd+bf16"],
        msvc_flags: &["/arch:armv8.6"],
    },
    // FEAT_FHM: optional from ARMv8.1, mandatory at ARMv8.4 with FP16
    IsaProbe {
        name: "NK_TARGET_NEONFHM",
        probe_file: "probes/arm_neon_fhm.c",
        gcc_flags: &["-march=armv8.2-a+simd+fp16+fp16fml"],
        msvc_flags: &["/arch:armv8.4"],
    },
    IsaProbe {
        name: "NK_TARGET_SVE",
        probe_file: "probes/arm_sve.c",
        gcc_flags: &["-march=armv8.2-a+sve"],
        msvc_flags: &[],
    },
    IsaProbe {
        name: "NK_TARGET_SVEHALF",
        probe_file: "probes/arm_sve_half.c",
        gcc_flags: &["-march=armv8.2-a+sve+fp16"],
        msvc_flags: &[],
    },
    IsaProbe {
        name: "NK_TARGET_SVEBFDOT",
        probe_file: "probes/arm_sve_bfdot.c",
        gcc_flags: &["-march=armv8.2-a+sve+bf16"],
        msvc_flags: &[],
    },
    IsaProbe {
        name: "NK_TARGET_SVESDOT",
        probe_file: "probes/arm_sve_sdot.c",
        gcc_flags: &["-march=armv8.2-a+sve+dotprod"],
        msvc_flags: &[],
    },
    IsaProbe {
        name: "NK_TARGET_SVE2",
        probe_file: "probes/arm_sve2.c",
        gcc_flags: &["-march=armv8.2-a+sve2"],
        msvc_flags: &[],
    },
    IsaProbe {
        name: "NK_TARGET_SVE2P1",
        probe_file: "probes/arm_sve2p1.c",
        gcc_flags: &["-march=armv8.2-a+sve2p1"],
        msvc_flags: &[],
    },
    IsaProbe {
        name: "NK_TARGET_NEONFP8",
        probe_file: "probes/arm_neonfp8.c",
        gcc_flags: &["-march=armv8-a+simd+fp8dot4"],
        msvc_flags: &[],
    },
    IsaProbe {
        name: "NK_TARGET_SME",
        probe_file: "probes/arm_sme.c",
        gcc_flags: &["-march=armv8-a+sme"],
        msvc_flags: &[],
    },
    IsaProbe {
        name: "NK_TARGET_SME2",
        probe_file: "probes/arm_sme2.c",
        gcc_flags: &["-march=armv8-a+sme2"],
        msvc_flags: &[],
    },
    IsaProbe {
        name: "NK_TARGET_SME2P1",
        probe_file: "probes/arm_sme2p1.c",
        gcc_flags: &["-march=armv8-a+sme2p1"],
        msvc_flags: &[],
    },
    IsaProbe {
        name: "NK_TARGET_SMEF64",
        probe_file: "probes/arm_sme_f64.c",
        gcc_flags: &["-march=armv8-a+sme+sme-f64f64"],
        msvc_flags: &[],
    },
    IsaProbe {
        name: "NK_TARGET_SMEHALF",
        probe_file: "probes/arm_sme_half.c",
        gcc_flags: &["-march=armv8-a+sme+sme-f16f16"],
        msvc_flags: &[],
    },
    IsaProbe {
        name: "NK_TARGET_SMEBF16",
        probe_file: "probes/arm_sme_bf16.c",
        gcc_flags: &["-march=armv8-a+sme2+sme-b16b16"],
        msvc_flags: &[],
    },
    IsaProbe {
        name: "NK_TARGET_SMEBI32",
        probe_file: "probes/arm_sme_bi32.c",
        gcc_flags: &["-march=armv8-a+sme2"],
        msvc_flags: &[],
    },
    IsaProbe {
        name: "NK_TARGET_SMELUT2",
        probe_file: "probes/arm_sme_lut2.c",
        gcc_flags: &["-march=armv8-a+sme2+lut"],
        msvc_flags: &[],
    },
    IsaProbe {
        name: "NK_TARGET_SMEFA64",
        probe_file: "probes/arm_sme_fa64.c",
        gcc_flags: &["-march=armv8-a+sme+sme-fa64"],
        msvc_flags: &[],
    },
];

const RISCV_PROBES: &[IsaProbe] = &[
    IsaProbe {
        name: "NK_TARGET_RVV",
        probe_file: "probes/riscv_rvv.c",
        gcc_flags: &["-march=rv64gcv"],
        msvc_flags: &[],
    },
    IsaProbe {
        name: "NK_TARGET_RVVHALF",
        probe_file: "probes/riscv_rvv_half.c",
        gcc_flags: &["-march=rv64gcv_zvfh"],
        msvc_flags: &[],
    },
    IsaProbe {
        name: "NK_TARGET_RVVBF16",
        probe_file: "probes/riscv_rvv_bf16.c",
        gcc_flags: &["-march=rv64gcv_zvfbfwma"],
        msvc_flags: &[],
    },
    IsaProbe {
        name: "NK_TARGET_RVVBB",
        probe_file: "probes/riscv_rvv_bb.c",
        gcc_flags: &["-march=rv64gcv_zvbb"],
        msvc_flags: &[],
    },
];

const LOONGARCH_PROBES: &[IsaProbe] = &[IsaProbe {
    name: "NK_TARGET_LOONGSONASX",
    probe_file: "probes/loongarch_lasx.c",
    gcc_flags: &["-mlasx"],
    msvc_flags: &[],
}];

const POWER_PROBES: &[IsaProbe] = &[IsaProbe {
    name: "NK_TARGET_POWERVSX",
    probe_file: "probes/power_vsx.c",
    gcc_flags: &["-mcpu=power9", "-mvsx"],
    msvc_flags: &[],
}];

const WASM_PROBES: &[IsaProbe] = &[IsaProbe {
    name: "NK_TARGET_V128RELAXED",
    probe_file: "probes/wasm_v128relaxed.c",
    gcc_flags: &["-mrelaxed-simd"],
    msvc_flags: &[],
}];

fn build_numkong() -> Result<HashMap<String, bool>, String> {
    let mut flags = HashMap::<String, bool>::new();
    let mut build = cc::Build::new();

    // Source files
    build
        // Prefer portable flags to support MSVC and older toolchains
        .std("c99") // Enforce C99 standard when supported
        .file("c/numkong.c")
        // Complex float dispatch files
        .file("c/dispatch_f64c.c")
        .file("c/dispatch_f32c.c")
        .file("c/dispatch_bf16c.c")
        .file("c/dispatch_f16c.c")
        // Real float dispatch files
        .file("c/dispatch_f64.c")
        .file("c/dispatch_f32.c")
        .file("c/dispatch_bf16.c")
        .file("c/dispatch_f16.c")
        // Exotic float dispatch files
        .file("c/dispatch_e5m2.c")
        .file("c/dispatch_e4m3.c")
        .file("c/dispatch_e3m2.c")
        .file("c/dispatch_e2m3.c")
        // Signed integer dispatch files
        .file("c/dispatch_i64.c")
        .file("c/dispatch_i32.c")
        .file("c/dispatch_i16.c")
        .file("c/dispatch_i8.c")
        .file("c/dispatch_i4.c")
        // Unsigned integer dispatch files
        .file("c/dispatch_u64.c")
        .file("c/dispatch_u32.c")
        .file("c/dispatch_u16.c")
        .file("c/dispatch_u8.c")
        .file("c/dispatch_u4.c")
        .file("c/dispatch_u1.c")
        // Special dispatch files
        .file("c/dispatch_other.c")
        .include("include")
        .define("NK_NATIVE_F16", "0")
        .define("NK_NATIVE_BF16", "0")
        .define("NK_DYNAMIC_DISPATCH", "1")
        .opt_level(3)
        .flag_if_supported("-pedantic") // Strict compliance when supported
        .flag_if_supported("-Wno-psabi") // Suppress GCC ABI note for 32-byte aligned params
        .warnings(false);

    // Architecture detection
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_bits = env::var("CARGO_CFG_TARGET_POINTER_WIDTH").unwrap_or_default();
    let is_msvc = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default() == "msvc";

    let is_x86_64 = target_arch == "x86_64" && target_bits == "64";
    let is_aarch64 = target_arch == "aarch64" && target_bits == "64";
    let is_riscv64 = target_arch == "riscv64" && target_bits == "64";
    let is_loongarch64 = target_arch == "loongarch64" && target_bits == "64";
    let is_power64 = target_arch == "powerpc64" && target_bits == "64";

    build.define("NK_IS_64BIT_X86", if is_x86_64 { "1" } else { "0" });
    build.define("NK_IS_64BIT_ARM", if is_aarch64 { "1" } else { "0" });
    build.define("NK_IS_64BIT_RISCV", if is_riscv64 { "1" } else { "0" });
    build.define(
        "NK_IS_64BIT_LOONGARCH",
        if is_loongarch64 { "1" } else { "0" },
    );
    build.define("NK_IS_64BIT_POWER", if is_power64 { "1" } else { "0" });

    // On 32-bit x86, ensure proper stack alignment for floating-point operations
    // See: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=38534
    if target_arch == "x86" {
        build.flag_if_supported("-mstackrealign");
        build.flag_if_supported("-mpreferred-stack-boundary=4");
    }

    // Pin TU baseline to each arch's ABI floor; SIMD kernels carry per-function pragmas.
    // `NK_MARCH_NATIVE=1` opts into a host-tuned, non-portable build (ignored on MSVC).
    // Keep per-arch table in sync with cmake/nk_compiler_flags.cmake, setup.py, binding.gyp.
    let march_native = env::var("NK_MARCH_NATIVE").is_ok_and(|v| v == "1" || v == "true");
    // Portable baseline: pin TU ISA floor + forbid auto-vectorization so serial
    // fallbacks don't get silently promoted to NEON/SSE2/VSX. SIMD kernels use
    // explicit intrinsics; unaffected. MSVC has no command-line vectorizer
    // toggle; `NK_MARCH_NATIVE=1` opts out for host-tuned builds.
    if march_native && !is_msvc {
        println!("cargo:warning=NK_MARCH_NATIVE=1: building -march=native, result will not run on older CPUs");
        build.flag_if_supported("-march=native");
    } else if is_msvc {
        match target_arch.as_str() {
            "x86_64"  => { build.flag_if_supported("/arch:SSE2"); }
            "aarch64" => { build.flag_if_supported("/arch:armv8.0"); }
            _ => {}
        }
    } else {
        build.flag_if_supported("-fno-tree-vectorize");
        build.flag_if_supported("-fno-tree-slp-vectorize");
        match target_arch.as_str() {
            "x86_64"      => { build.flag_if_supported("-march=x86-64"); }
            "aarch64"     => { build.flag_if_supported("-march=armv8-a"); }
            "riscv64"     => { build.flag_if_supported("-march=rv64gc"); }
            "powerpc64"   => { build.flag_if_supported("-mcpu=power8"); }
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

    // Probe each ISA — uniform for all architectures including NEON and WASM
    for table in probe_tables {
        for probe in table.iter() {
            // Allow env-var override: NK_TARGET_FOO=0 forces off, NK_TARGET_FOO=1 forces on
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

            let probe_flags = if is_msvc {
                probe.msvc_flags
            } else {
                probe.gcc_flags
            };
            let ok = probe_isa(probe.probe_file, probe_flags);
            build.define(probe.name, if ok { "1" } else { "0" });
            flags.insert(probe.name.to_string(), ok);
            if !ok {
                println!("cargo:warning={}: not supported by compiler", probe.name);
            }
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
    println!("cargo:rerun-if-env-changed=NK_MARCH_NATIVE");
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
