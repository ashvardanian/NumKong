/**
 *  @brief Shared definitions for the NumKong library.
 *  @file include/numkong/types.h
 *  @author Ash Vardanian
 *  @date October 2, 2023
 *
 *  Defines:
 *
 *  - Sized aliases for numeric types, like: `nk_i32_t` and `nk_f64_t`.
 *  - Macros for internal compiler/hardware checks, like: `NK_TARGET_ARM64_`.
 *  - Macros for feature controls, like: `NK_TARGET_NEON`
 *
 *  @section fp8_types FP8 Numeric Types
 *
 *  There are several variants of 8-bit floating point types supported by different industry memebers
 *  with different hardware support. None are part of the IEEE 754 standard, but some are part of the
 *  Open Compute Project (OCP) 8-bit Floating Point Specification (OFP8):
 *
 *      Format    Bias  Sign  Exp  Mant  Range   Infinity            NaN               Standard
 *      E4M3FN    7     1     4    3     ±448    ❌ No               Only 0x7F/0xFF    OCP, NVIDIA, ONNX
 *      E5M2      15    1     5    2     ±57344  ✅ Yes (0x7C/0xFC)  0x7D-7F, 0xFD-FF  OCP, IEEE-like
 *      E4M3FNUZ  8     1     4    3     ±240    ❌ No               0x80 only         GraphCore, ONNX
 *      E5M2FNUZ  16    1     5    2     ±57344  ❌ No               0x80 only         GraphCore, ONNX
 *
 *  In currently available and soon incoming harware, only two series of models prioritze FNUZ over OCP:
 *
 *  - GraphCore IPUs were the original platform proposing FNUZ
 *  - AMD MI300 series based on CDNA3 implements FNUZ, but not OCP
 *  - AMD MI350+ series based on CDNA4 switch to OCP and remove FNUZ
 *  - NVIDIA Hopper and Blackwell only support E4M3FN, E5M2
 *  - Intel AVX10.2 defines HF8 (E4M3FN) and BF8 (E5M2) - OCP-aligned
 *  - Arm implements E4M3 (meaning E4M3FN) and E5M2 with a shared `__mfp8` type and a `FPMR` format selector
 *
 *  For brevety, across NumKong, "E4M3" implies "E4M3FN".
 *
 *  @see https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1
 *  @see FP8 Formats for Deep Learning: https://arxiv.org/pdf/2209.05433
 *  @see ONNX Float8 Types: https://onnx.ai/onnx/technical/float8.html
 *
 *  @section fp6_types FP6 Numeric Types
 *
 *  The OCP Microscaling (MX) v1.0 specification defines two 6-bit floating-point formats
 *  for block-scaled quantization. Both are "FN" (finite-numeric): all bit patterns map
 *  to real numbers with no Inf or NaN codes. Stored byte-aligned with 2 bits of padding.
 *
 *      Format  Bias  Sign  Exp  Mant  Range   Subnormals  Infinity  NaN  Standard
 *      E2M3    1     1     2    3     ±7.5    14 of 64    ❌ No     ❌   OCP MX v1.0
 *      E3M2    3     1     3    2     ±28     6 of 64     ❌ No     ❌   OCP MX v1.0
 *
 *  E2M3 favors mantissa precision (3 bits) for narrow dynamic range — ideal for activations.
 *  E3M2 favors exponent range (3 bits) for wider dynamic range — suited for weights.
 *  Both follow IEEE 754 subnormal rules: when exp=0, the implicit leading bit is 0,
 *  giving value = (-1)^s × 0.mmm × 2^(1-bias). This provides gradual underflow to zero.
 *
 *  No hardware directly computes on FP6. On Arm with FEAT_FP8DOT4, E2M3 values can be
 *  losslessly promoted to E4M3 (same mantissa width, rebias exponent by +6) and E3M2 to
 *  E5M2 (same mantissa width, rebias exponent by +12), then fed to FDOT instructions.
 *  Subnormal values (exp=0) require normalization during this promotion.
 *
 *  @see https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
 *  @see https://arxiv.org/abs/2401.14112 (FP6-LLM paper)
 */
#ifndef NK_TYPES_H
#define NK_TYPES_H

// On Linux, `_GNU_SOURCE` must be defined before any system headers
// to expose `syscall` and other GNU extensions when C extensions are disabled.
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

// MSan (MemorySanitizer) cannot track data flow through SVE horizontal reductions
// like `svaddv`, which move data from vector registers to scalar registers via
// architecture-specific paths invisible to the compiler. NK_UNPOISON marks the
// resulting scalar as initialized so MSan does not report false positives.
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
#include <sanitizer/msan_interface.h>
#define NK_UNPOISON(ptr, size) __msan_unpoison((ptr), (size))
#endif
#endif
#ifndef NK_UNPOISON
#define NK_UNPOISON(ptr, size) (void)(ptr), (void)(size)
#endif

// Inferring target OS: Windows, macOS, Linux, or FreeBSD
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define NK_DEFINED_WINDOWS_ 1
#elif defined(__APPLE__) && defined(__MACH__)
#define NK_DEFINED_APPLE_ 1
#elif defined(__linux__)
#define NK_DEFINED_LINUX_ 1
#elif defined(__FreeBSD__)
#define NK_DEFINED_FREEBSD_ 1
#endif

// Annotation for the public API symbols:
//
// - `NK_PUBLIC` is used for functions that are part of the public API.
// - `NK_INTERNAL` is used for internal helper functions with unstable APIs.
// - `NK_DYNAMIC` is used for functions that are part of the public API, but are dispatched at runtime.
//
// On GCC we mark the functions as `nonnull` informing that none of the arguments can be `NULL`.
// Marking with `pure` and `const` isn't possible as outputting to a pointer is a "side effect".
#if defined(__GNUC__) || defined(__clang__)
#define NK_PUBLIC   __attribute__((unused)) inline static
#define NK_INTERNAL __attribute__((always_inline)) inline static
#elif defined(_MSC_VER)
#define NK_PUBLIC   inline static
#define NK_INTERNAL __forceinline static
#else
#define NK_PUBLIC   inline static
#define NK_INTERNAL inline static
#endif // defined(__GNUC__) || defined(__clang__)

#if NK_DYNAMIC_DISPATCH
#if defined(_WIN32) || defined(__CYGWIN__)
#define NK_DYNAMIC __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#define NK_DYNAMIC __attribute__((visibility("default")))
#else
#define NK_DYNAMIC NK_PUBLIC
#endif
#else
#define NK_DYNAMIC NK_PUBLIC
#endif // NK_DYNAMIC_DISPATCH

// Vector union types use type punning by design (write as f16, read as f32, etc.).
// Without this, GCC at -O2 assumes strict aliasing and may optimize away valid accesses.
#if defined(__GNUC__) || defined(__clang__)
#define NK_MAY_ALIAS_ __attribute__((may_alias))
#else
#define NK_MAY_ALIAS_
#endif

#if defined(__has_builtin)
#define nk_has_builtin_(x) __has_builtin(x)
#else
#define nk_has_builtin_(x) 0
#endif

// Allow SIMD kernels to redirect small inputs to serial implementations.
// Enabled by default for production use. Tests and benchmarks may disable
// this to isolate SIMD path behavior on small inputs.
#if !defined(NK_ALLOW_ISA_REDIRECT)
#define NK_ALLOW_ISA_REDIRECT 1
#endif

// Compiling for 64-bit Arm: NK_TARGET_ARM64_
// https://arm-software.github.io/acle/main/acle.html
#if !defined(NK_TARGET_ARM64_)
#if defined(__aarch64__) || defined(_M_ARM64)
#define NK_TARGET_ARM64_ 1
#else
#define NK_TARGET_ARM64_ 0
#endif // defined(__aarch64__) || defined(_M_ARM64)
#endif // !defined(NK_TARGET_ARM64_)

// Compiling for x86: NK_TARGET_X8664_
// https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2024-2/additional-predefined-macros.html
#if !defined(NK_TARGET_X8664_)
#if defined(__x86_64__) || defined(_M_X64)
#define NK_TARGET_X8664_ 1
#else
#define NK_TARGET_X8664_ 0
#endif // defined(__x86_64__) || defined(_M_X64)
#endif // !defined(NK_TARGET_X8664_)

// Compiling for RISC-V: NK_TARGET_RISCV64_
#if !defined(NK_TARGET_RISCV64_)
#if defined(__riscv) && (__riscv_xlen == 64)
#define NK_TARGET_RISCV64_ 1
#else
#define NK_TARGET_RISCV64_ 0
#endif // defined(__riscv) && (__riscv_xlen == 64)
#endif // !defined(NK_TARGET_RISCV64_)

// Compiling for LoongArch: NK_TARGET_LOONGARCH64_
#if !defined(NK_TARGET_LOONGARCH64_)
#if defined(__loongarch__)
#define NK_TARGET_LOONGARCH64_ 1
#else
#define NK_TARGET_LOONGARCH64_ 0
#endif // defined(__loongarch__)
#endif // !defined(NK_TARGET_LOONGARCH64_)

// Compiling for Power: NK_TARGET_POWER64_
#if !defined(NK_TARGET_POWER64_)
#if defined(__powerpc64__) || defined(__ppc64__) || defined(_ARCH_PPC64)
#define NK_TARGET_POWER64_ 1
#else
#define NK_TARGET_POWER64_ 0
#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(_ARCH_PPC64)
#endif // !defined(NK_TARGET_POWER64_)

// Compiling for WASM: NK_TARGET_WASM_
#if !defined(NK_TARGET_WASM_)
#if defined(__wasm__) || defined(__EMSCRIPTEN__)
#define NK_TARGET_WASM_ 1
#else
#define NK_TARGET_WASM_ 0
#endif
#endif // !defined(NK_TARGET_WASM_)

// WASI hosted mode: NK_DEFINED_WASI_
// When NK_WASI_HOSTED=ON in CMake, this is predefined to 1 so the library
// imports capability probes (nk_has_v128, nk_has_relaxed) from the host.
// Standalone runtimes (Wasmer, Wasmtime CLI) cannot supply those imports,
// so the default for plain __wasi__ builds is 0 (compile-time detection).
#if !defined(NK_DEFINED_WASI_)
#define NK_DEFINED_WASI_ 0
#endif // !defined(NK_DEFINED_WASI_)

// Compiling for WASM with Relaxed SIMD: NK_TARGET_V128RELAXED
// Requires -mrelaxed-simd for FMA instructions (f32x4.relaxed_madd, f64x2.relaxed_madd)
#if !defined(NK_TARGET_V128RELAXED) || (NK_TARGET_V128RELAXED && !NK_TARGET_WASM_)
#if defined(__wasm_relaxed_simd__)
#define NK_TARGET_V128RELAXED 1
#else
#undef NK_TARGET_V128RELAXED
#define NK_TARGET_V128RELAXED 0
#endif
#endif // !defined(NK_TARGET_V128RELAXED) || ...

// Compiling for RISC-V Vector: NK_TARGET_RVV
#if !defined(NK_TARGET_RVV) || (NK_TARGET_RVV && !NK_TARGET_RISCV64_)
#if defined(__riscv_v) && (__riscv_v >= 1000000)
#define NK_TARGET_RVV 1
#else
#undef NK_TARGET_RVV
#define NK_TARGET_RVV 0
#endif // defined(__riscv_v) && (__riscv_v >= 1000000)
#endif // !defined(NK_TARGET_RVV) || ...

// Compiling for RISC-V Vector with Zvfh (f16): NK_TARGET_RVVHALF
// Requires GCC 14+ or Clang 18+ for full intrinsic support
#if !defined(NK_TARGET_RVVHALF) || (NK_TARGET_RVVHALF && !NK_TARGET_RVV)
#if defined(__riscv_zvfh) && (__riscv_zvfh > 0)
#define NK_TARGET_RVVHALF 1
#else
#undef NK_TARGET_RVVHALF
#define NK_TARGET_RVVHALF 0
#endif // defined(__riscv_zvfh) && (__riscv_zvfh > 0)
#endif // !defined(NK_TARGET_RVVHALF) || ...

// Compiling for RISC-V Vector with Zvfbfwma (bf16 widening FMA): NK_TARGET_RVVBF16
// Requires GCC 14+ or Clang 18+ for full intrinsic support
#if !defined(NK_TARGET_RVVBF16) || (NK_TARGET_RVVBF16 && !NK_TARGET_RVV)
#if defined(__riscv_zvfbfwma) && (__riscv_zvfbfwma > 0)
#define NK_TARGET_RVVBF16 1
#else
#undef NK_TARGET_RVVBF16
#define NK_TARGET_RVVBF16 0
#endif // defined(__riscv_zvfbfwma) && (__riscv_zvfbfwma > 0)
#endif // !defined(NK_TARGET_RVVBF16) || ...

// Compiling for RISC-V Vector with Zvbb (basic bit-manipulation): NK_TARGET_RVVBB
// Provides vcpop.v (per-element popcount), vclz.v, vctz.v, vbrev.v, vrol.v, vror.v
#if !defined(NK_TARGET_RVVBB) || (NK_TARGET_RVVBB && !NK_TARGET_RVV)
#if defined(__riscv_zvbb) && (__riscv_zvbb > 0)
#define NK_TARGET_RVVBB 1
#else
#undef NK_TARGET_RVVBB
#define NK_TARGET_RVVBB 0
#endif // defined(__riscv_zvbb) && (__riscv_zvbb > 0)
#endif // !defined(NK_TARGET_RVVBB) || ...

// Compiling for LoongArch LASX (256-bit SIMD): NK_TARGET_LOONGSONASX
// LASX provides 32 × 256-bit vector registers, widening integer multiply-accumulate,
// and f32-to-f64 conversion (xvfcvtl_d_s / xvfcvth_d_s) but no widening FMA.
#if !defined(NK_TARGET_LOONGSONASX) || (NK_TARGET_LOONGSONASX && !NK_TARGET_LOONGARCH64_)
#if defined(__loongarch_asx)
#define NK_TARGET_LOONGSONASX 1
#else
#undef NK_TARGET_LOONGSONASX
#define NK_TARGET_LOONGSONASX 0
#endif // defined(__loongarch_asx)
#endif // !defined(NK_TARGET_LOONGSONASX) || ...

// Compiling for Power VSX (128-bit SIMD, POWER9+ baseline): NK_TARGET_POWERVSX
// VSX provides 64 × 128-bit registers, FMA (vec_madd), vec_msum (multiply-sum), hardware f16
// conversion (vec_extract_fp32_from_shorth/l), length-limited loads (vec_xl_len), per-byte
// popcount (vec_popcnt), and vec_cmpne. Requires POWER9 (ISA 3.0) or newer.
#if !defined(NK_TARGET_POWERVSX) || (NK_TARGET_POWERVSX && !NK_TARGET_POWER64_)
#if defined(__VSX__) && defined(__POWER9_VECTOR__)
#define NK_TARGET_POWERVSX 1
#else
#undef NK_TARGET_POWERVSX
#define NK_TARGET_POWERVSX 0
#endif // defined(__VSX__)
#endif // !defined(NK_TARGET_POWERVSX) || ...

// Compiling for Arm: NK_TARGET_NEON (AArch64 only, AArch32 NEON is not supported)
#if !defined(NK_TARGET_NEON) || (NK_TARGET_NEON && !NK_TARGET_ARM64_)
#if (defined(__ARM_NEON) && defined(__aarch64__)) || (defined(_MSC_VER) && defined(_M_ARM64))
#define NK_TARGET_NEON 1
#else
#undef NK_TARGET_NEON
#define NK_TARGET_NEON 0
#endif // (defined(__ARM_NEON) && defined(__aarch64__)) || ...
#endif // !defined(NK_TARGET_NEON) || ...

// Compiling for Arm: NK_TARGET_NEONSDOT (FEAT_DotProd, AArch64 only)
#if !defined(NK_TARGET_NEONSDOT) || (NK_TARGET_NEONSDOT && !NK_TARGET_ARM64_)
#if (defined(__ARM_FEATURE_DOTPROD) && defined(__aarch64__)) || \
    (defined(_MSC_VER) && defined(_M_ARM64) && __ARM_ARCH >= 804)
#define NK_TARGET_NEONSDOT 1
#else
#undef NK_TARGET_NEONSDOT
#define NK_TARGET_NEONSDOT 0
#endif
#endif // !defined(NK_TARGET_NEONSDOT) || ...

// Compiling for Arm: NK_TARGET_NEONHALF (FEAT_FP16, AArch64 only)
#if !defined(NK_TARGET_NEONHALF) || (NK_TARGET_NEONHALF && !NK_TARGET_ARM64_)
#if (defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(__aarch64__)) || \
    (defined(_MSC_VER) && defined(_M_ARM64) && __ARM_ARCH >= 802)
#define NK_TARGET_NEONHALF 1
#else
#undef NK_TARGET_NEONHALF
#define NK_TARGET_NEONHALF 0
#endif
#endif // !defined(NK_TARGET_NEONHALF) || ...

// Compiling for Arm: NK_TARGET_NEONFHM (FEAT_FHM, AArch64 only)
#if !defined(NK_TARGET_NEONFHM) || (NK_TARGET_NEONFHM && !NK_TARGET_ARM64_)
#if (defined(__ARM_FEATURE_FP16_FML) && defined(__aarch64__)) || \
    (defined(_MSC_VER) && defined(_M_ARM64) && __ARM_ARCH >= 804)
#define NK_TARGET_NEONFHM 1
#else
#undef NK_TARGET_NEONFHM
#define NK_TARGET_NEONFHM 0
#endif
#endif // !defined(NK_TARGET_NEONFHM) || ...

// Compiling for Arm: NK_TARGET_NEONBFDOT (FEAT_BF16, AArch64 only)
#if !defined(NK_TARGET_NEONBFDOT) || (NK_TARGET_NEONBFDOT && !NK_TARGET_ARM64_)
#if (defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) && defined(__aarch64__)) || \
    (defined(_MSC_VER) && defined(_M_ARM64) && __ARM_ARCH >= 806)
#define NK_TARGET_NEONBFDOT 1
#else
#undef NK_TARGET_NEONBFDOT
#define NK_TARGET_NEONBFDOT 0
#endif
#endif // !defined(NK_TARGET_NEONBFDOT) || ...

// Compiling for Arm: NK_TARGET_NEONFP8 (NEON FP8 extensions, FEAT_FP8DOT4)
// ACLE macro __ARM_FEATURE_FP8DOT4 defined by GCC 15+ and Clang 21+ when +fp8dot4 is enabled.
// Older compilers lack mfloat8x16_t and the fp8dot4 target attribute entirely.
#if !defined(NK_TARGET_NEONFP8) || (NK_TARGET_NEONFP8 && !NK_TARGET_ARM64_)
#if defined(__ARM_FEATURE_FP8DOT4) && defined(__aarch64__)
#define NK_TARGET_NEONFP8 1
#else
#undef NK_TARGET_NEONFP8
#define NK_TARGET_NEONFP8 0
#endif // defined(__ARM_FEATURE_FP8DOT4)
#endif // !defined(NK_TARGET_NEONFP8)  || ...

// Compiling for Arm: NK_TARGET_SVE
#if !defined(NK_TARGET_SVE) || (NK_TARGET_SVE && !NK_TARGET_ARM64_)
#if defined(__ARM_FEATURE_SVE)
#define NK_TARGET_SVE 1
#else
#undef NK_TARGET_SVE
#define NK_TARGET_SVE 0
#endif // defined(__ARM_FEATURE_SVE)
#endif // !defined(NK_TARGET_SVE) || ...

// Compiling for Arm: NK_TARGET_SVESDOT
#if !defined(NK_TARGET_SVESDOT) || (NK_TARGET_SVESDOT && !NK_TARGET_ARM64_)
#if defined(__ARM_FEATURE_SVE)
#define NK_TARGET_SVESDOT 1
#else
#undef NK_TARGET_SVESDOT
#define NK_TARGET_SVESDOT 0
#endif // defined(__ARM_FEATURE_SVE)
#endif // !defined(NK_TARGET_SVESDOT) || ...

// Compiling for Arm: NK_TARGET_SVEHALF
#if !defined(NK_TARGET_SVEHALF) || (NK_TARGET_SVEHALF && !NK_TARGET_ARM64_)
#if defined(__ARM_FEATURE_SVE)
#define NK_TARGET_SVEHALF 1
#else
#undef NK_TARGET_SVEHALF
#define NK_TARGET_SVEHALF 0
#endif // defined(__ARM_FEATURE_SVE)
#endif // !defined(NK_TARGET_SVEHALF) || ...

// Compiling for Arm: NK_TARGET_SVEBFDOT
#if !defined(NK_TARGET_SVEBFDOT) || (NK_TARGET_SVEBFDOT && !NK_TARGET_ARM64_)
#if defined(__ARM_FEATURE_SVE)
#define NK_TARGET_SVEBFDOT 1
#else
#undef NK_TARGET_SVEBFDOT
#define NK_TARGET_SVEBFDOT 0
#endif // defined(__ARM_FEATURE_SVE)
#endif // !defined(NK_TARGET_SVEBFDOT) || ...

// Compiling for Arm: NK_TARGET_SVE2
#if !defined(NK_TARGET_SVE2) || (NK_TARGET_SVE2 && !NK_TARGET_ARM64_)
#if defined(__ARM_FEATURE_SVE2)
#define NK_TARGET_SVE2 1
#else
#undef NK_TARGET_SVE2
#define NK_TARGET_SVE2 0
#endif // defined(__ARM_FEATURE_SVE2)
#endif // !defined(NK_TARGET_SVE2) || ...

// Compiling for Arm: NK_TARGET_SVE2P1
#if !defined(NK_TARGET_SVE2P1) || (NK_TARGET_SVE2P1 && !NK_TARGET_ARM64_)
#undef NK_TARGET_SVE2P1
#define NK_TARGET_SVE2P1 0
#endif // !defined(NK_TARGET_SVE2P1) || ...

// Compiling for Arm: NK_TARGET_SME (Scalable Matrix Extension)
#if !defined(NK_TARGET_SME) || (NK_TARGET_SME && !NK_TARGET_ARM64_)
#if defined(__ARM_FEATURE_SME)
#define NK_TARGET_SME 1
#else
#undef NK_TARGET_SME
#define NK_TARGET_SME 0
#endif // defined(__ARM_FEATURE_SME)
#endif // !defined(NK_TARGET_SME) || ...

#if !defined(NK_TARGET_SME2) || (NK_TARGET_SME2 && !NK_TARGET_ARM64_)
#if defined(__ARM_FEATURE_SME2)
#define NK_TARGET_SME2 1
#else
#undef NK_TARGET_SME2
#define NK_TARGET_SME2 0
#endif // defined(__ARM_FEATURE_SME2)
#endif // !defined(NK_TARGET_SME2) || ...

// Compiling for Arm: NK_TARGET_SME2P1 (FEAT_SME2p1)
// ACLE macro: __ARM_FEATURE_SME2p1 (note lowercase 'p')
#if !defined(NK_TARGET_SME2P1) || (NK_TARGET_SME2P1 && !NK_TARGET_ARM64_)
#if defined(__ARM_FEATURE_SME2p1)
#define NK_TARGET_SME2P1 1
#else
#undef NK_TARGET_SME2P1
#define NK_TARGET_SME2P1 0
#endif // defined(__ARM_FEATURE_SME2p1)
#endif // !defined(NK_TARGET_SME2P1) || ...

// AppleClang 17 exposes SME sub-features through `arm_sme.h` builtin aliases,
// not dedicated `__ARM_FEATURE_*` predefines for every matrix subtype.
#if !defined(NK_TARGET_SMEF64) || (NK_TARGET_SMEF64 && !NK_TARGET_ARM64_)
#if defined(__ARM_FEATURE_SME_F64F64) || nk_has_builtin_(__builtin_sme_svmopa_za64_f64_m)
#define NK_TARGET_SMEF64 1
#else
#undef NK_TARGET_SMEF64
#define NK_TARGET_SMEF64 0
#endif // defined(__ARM_FEATURE_SME_F64F64) || ...
#endif // !defined(NK_TARGET_SMEF64) || ...

#if !defined(NK_TARGET_SMEBI32) || (NK_TARGET_SMEBI32 && !NK_TARGET_ARM64_)
#if nk_has_builtin_(__builtin_sme_svbmopa_za32_u32_m)
#define NK_TARGET_SMEBI32 1
#else
#undef NK_TARGET_SMEBI32
#define NK_TARGET_SMEBI32 0
#endif // nk_has_builtin_(__builtin_sme_svbmopa_za32_u32_m)
#endif // !defined(NK_TARGET_SMEBI32) || ...

#if !defined(NK_TARGET_SMEHALF) || (NK_TARGET_SMEHALF && !NK_TARGET_ARM64_)
#if defined(__ARM_FEATURE_SME_F16F16) || nk_has_builtin_(__builtin_sme_svmopa_za32_f16_m)
#define NK_TARGET_SMEHALF 1
#else
#undef NK_TARGET_SMEHALF
#define NK_TARGET_SMEHALF 0
#endif // nk_has_builtin_(__builtin_sme_svmopa_za32_f16_m)
#endif // !defined(NK_TARGET_SMEHALF) || ...

#if !defined(NK_TARGET_SMEBF16) || (NK_TARGET_SMEBF16 && !NK_TARGET_ARM64_)
#if nk_has_builtin_(__builtin_sme_svmopa_za32_bf16_m)
#define NK_TARGET_SMEBF16 1
#else
#undef NK_TARGET_SMEBF16
#define NK_TARGET_SMEBF16 0
#endif // nk_has_builtin_(__builtin_sme_svmopa_za32_bf16_m)
#endif // !defined(NK_TARGET_SMEBF16) || ...

#if !defined(NK_TARGET_SMELUT2) || (NK_TARGET_SMELUT2 && !NK_TARGET_ARM64_)
#if nk_has_builtin_(__builtin_sme_svluti2_lane_zt_u8)
#define NK_TARGET_SMELUT2 1
#else
#undef NK_TARGET_SMELUT2
#define NK_TARGET_SMELUT2 0
#endif // nk_has_builtin_(__builtin_sme_svluti2_lane_zt_u8)
#endif // !defined(NK_TARGET_SMELUT2) || ...

// Compiling for Arm: NK_TARGET_SMEFA64 (FEAT_SME_FA64, full SVE2 in streaming mode)
#if !defined(NK_TARGET_SMEFA64) || (NK_TARGET_SMEFA64 && !NK_TARGET_ARM64_)
#if defined(__ARM_FEATURE_SME_FA64)
#define NK_TARGET_SMEFA64 1
#else
#undef NK_TARGET_SMEFA64
#define NK_TARGET_SMEFA64 0
#endif // defined(__ARM_FEATURE_SME_FA64)
#endif // !defined(NK_TARGET_SMEFA64) || ...

// Compiling for x86: NK_TARGET_HASWELL
//
// Starting with Ivy Bridge, Intel supports the `F16C` extensions for fast half-precision
// to single-precision floating-point conversions. On AMD those instructions
// are supported on all CPUs starting with Jaguar 2009.
// Starting with Sandy Bridge, Intel adds basic AVX support in their CPUs and in 2013
// extends it with AVX2 in the Haswell generation. Moreover, Haswell adds FMA support.
//
// On MSVC, most GCC-style ISA macros are unavailable. MSVC defines __AVX__, __AVX2__,
// __AVX512F/BW/CD/DQ/VL__, and __AVX10_VER__, but NOT __AVXVNNI__, __AVX512VNNI__,
// __AVX512BF16__, __AVX512FP16__, __AMX_*__, etc.
// Instead, MSVC makes all intrinsics available once the toolset version supports them,
// without requiring `/arch:AVX512`. We gate on _MSC_VER to auto-enable targets:
//   - _MSC_VER >= 1900 (VS 2015+): AVX2/FMA/F16C (Haswell)
//   - _MSC_VER >= 1920 (VS 2019+): AVX-512 base (Skylake, Icelake), AVX-VNNI (Alder)
//   - _MSC_VER >= 1944 (VS 2022 17.14+): BF16, FP16, VP2INTERSECT, VNNI-INT8 (Sierra), AMX
#if !defined(NK_TARGET_HASWELL) || (NK_TARGET_HASWELL && !NK_TARGET_X8664_)
#if (defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)) || (defined(_MSC_VER) && _MSC_VER >= 1900)
#define NK_TARGET_HASWELL 1
#else
#undef NK_TARGET_HASWELL
#define NK_TARGET_HASWELL 0
#endif // defined(__AVX2__)
#endif // !defined(NK_TARGET_HASWELL) || ...

// Compiling for x86: NK_TARGET_SKYLAKE, NK_TARGET_ICELAKE, NK_TARGET_GENOA,
// NK_TARGET_SAPPHIRE, NK_TARGET_TURIN, NK_TARGET_SIERRA
//
// To list all available macros for x86, take a recent compiler, like GCC 12 and run:
//      gcc-12 -march=sapphirerapids -dM -E - < /dev/null | egrep "SSE|AVX" | sort
// On Arm machines you may want to check for other flags:
//      gcc-12 -march=native -dM -E - < /dev/null | egrep "NEON|SVE|FP16|FMA" | sort
#if !defined(NK_TARGET_SKYLAKE) || (NK_TARGET_SKYLAKE && !NK_TARGET_X8664_)
#if (defined(__AVX512F__) && defined(__AVX512CD__) && defined(__AVX512VL__) && defined(__AVX512DQ__) && \
     defined(__AVX512BW__)) ||                                                                          \
    (defined(_MSC_VER) && _MSC_VER >= 1920)
#define NK_TARGET_SKYLAKE 1
#else
#undef NK_TARGET_SKYLAKE
#define NK_TARGET_SKYLAKE 0
#endif
#endif // !defined(NK_TARGET_SKYLAKE) || ...

#if !defined(NK_TARGET_ICELAKE) || (NK_TARGET_ICELAKE && !NK_TARGET_X8664_)
#if (defined(__AVX512VNNI__) && defined(__AVX512IFMA__) && defined(__AVX512BITALG__) && defined(__AVX512VBMI__) && \
     defined(__AVX512VBMI2__) && defined(__AVX512VPOPCNTDQ__)) ||                                                  \
    (defined(_MSC_VER) && _MSC_VER >= 1920)
#define NK_TARGET_ICELAKE 1
#else
#undef NK_TARGET_ICELAKE
#define NK_TARGET_ICELAKE 0
#endif
#endif // !defined(NK_TARGET_ICELAKE) || ...

#if !defined(NK_TARGET_GENOA) || (NK_TARGET_GENOA && !NK_TARGET_X8664_)
#if defined(__AVX512BF16__) || (defined(_MSC_VER) && _MSC_VER >= 1944)
#define NK_TARGET_GENOA 1
#else
#undef NK_TARGET_GENOA
#define NK_TARGET_GENOA 0
#endif // defined(__AVX512BF16__) || ...
#endif // !defined(NK_TARGET_GENOA) || ...

// Compiling for x86: NK_TARGET_DIAMOND (AVX10.2, Diamond Rapids)
// GCC 14+: defines __AVX10_2__ with -mavx10.2-512
// Clang 19+: defines __AVX10_2__ with -mavx10.2-512
// MSVC: defines __AVX10_VER__ >= 2 with /arch:AVX10.2 (VS 2026+, not yet released)
#if !defined(NK_TARGET_DIAMOND) || (NK_TARGET_DIAMOND && !NK_TARGET_X8664_)
#if defined(__AVX10_2__) || (defined(__AVX10_VER__) && __AVX10_VER__ >= 2)
#define NK_TARGET_DIAMOND 1
#else
#undef NK_TARGET_DIAMOND
#define NK_TARGET_DIAMOND 0
#endif // defined(__AVX10_2__) || ...
#endif // !defined(NK_TARGET_DIAMOND) || ...

#if !defined(NK_TARGET_SAPPHIRE) || (NK_TARGET_SAPPHIRE && !NK_TARGET_X8664_)
#if defined(__AVX512FP16__) || (defined(_MSC_VER) && _MSC_VER >= 1944)
#define NK_TARGET_SAPPHIRE 1
#else
#undef NK_TARGET_SAPPHIRE
#define NK_TARGET_SAPPHIRE 0
#endif
#endif // !defined(NK_TARGET_SAPPHIRE) || ...

#if !defined(NK_TARGET_SAPPHIREAMX) || (NK_TARGET_SAPPHIREAMX && !NK_TARGET_X8664_)
#if (defined(__AMX_TILE__) && defined(__AMX_BF16__) && defined(__AMX_INT8__)) || (defined(_MSC_VER) && _MSC_VER >= 1944)
#define NK_TARGET_SAPPHIREAMX 1
#else
#undef NK_TARGET_SAPPHIREAMX
#define NK_TARGET_SAPPHIREAMX 0
#endif
#endif // !defined(NK_TARGET_SAPPHIREAMX) || ...

#if !defined(NK_TARGET_GRANITEAMX) || (NK_TARGET_GRANITEAMX && !NK_TARGET_X8664_)
#if (defined(__AMX_TILE__) && defined(__AMX_FP16__)) || (defined(_MSC_VER) && _MSC_VER >= 1944)
#define NK_TARGET_GRANITEAMX 1
#else
#undef NK_TARGET_GRANITEAMX
#define NK_TARGET_GRANITEAMX 0
#endif
#endif // !defined(NK_TARGET_GRANITEAMX) || ...

#if !defined(NK_TARGET_TURIN) || (NK_TARGET_TURIN && !NK_TARGET_X8664_)
#if defined(__AVX512VP2INTERSECT__) || (defined(_MSC_VER) && _MSC_VER >= 1944)
#define NK_TARGET_TURIN 1
#else
#undef NK_TARGET_TURIN
#define NK_TARGET_TURIN 0
#endif
#endif // !defined(NK_TARGET_TURIN) || ...

#if !defined(NK_TARGET_ALDER) || (NK_TARGET_ALDER && !NK_TARGET_X8664_)
#if defined(__AVXVNNI__) || (defined(_MSC_VER) && _MSC_VER >= 1920)
#define NK_TARGET_ALDER 1
#else
#undef NK_TARGET_ALDER
#define NK_TARGET_ALDER 0
#endif
#endif // !defined(NK_TARGET_ALDER) || ...

#if !defined(NK_TARGET_SIERRA) || (NK_TARGET_SIERRA && !NK_TARGET_X8664_)
#if defined(__AVXVNNIINT8__) || (defined(_MSC_VER) && _MSC_VER >= 1944)
#define NK_TARGET_SIERRA 1
#else
#undef NK_TARGET_SIERRA
#define NK_TARGET_SIERRA 0
#endif
#endif // !defined(NK_TARGET_SIERRA) || ...

// Include the relevant intrinsics headers
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#if NK_TARGET_NEON
#include <arm_neon.h>
#endif
#if NK_TARGET_SVE || NK_TARGET_SVE2
#include <arm_sve.h>
#endif
#if NK_TARGET_SME || NK_TARGET_SME2 || NK_TARGET_SMEBI32
#include <arm_sme.h>
#endif
#if NK_TARGET_HASWELL || NK_TARGET_SKYLAKE
#include <immintrin.h>
#endif
#if NK_TARGET_RVV
#include <riscv_vector.h>
#endif
#if NK_TARGET_LOONGSONASX
#include <lsxintrin.h>  // `__m128i` for LSX SIMD
#include <lasxintrin.h> // `__m256i` for LASX SIMD
#endif
#if NK_TARGET_POWERVSX
#include <altivec.h>
#endif
#if NK_TARGET_V128RELAXED
#include <wasm_simd128.h>
#endif

#if !defined(NK_F64_DIVISION_EPSILON)
#define NK_F64_DIVISION_EPSILON (1e-15)
#endif

#if !defined(NK_F32_DIVISION_EPSILON)
#define NK_F32_DIVISION_EPSILON (1e-7f)
#endif

#if !defined(NK_F16_DIVISION_EPSILON)
#define NK_F16_DIVISION_EPSILON (1e-3f)
#endif

/**
 *  @brief  The compile-time constant defining the capacity of `nk_tensor_position_t`.
 *          Matches `PyBUF_MAX_NDIM` by default.
 */
#if !defined(NK_TENSOR_MAX_RANK)
#define NK_TENSOR_MAX_RANK (64)
#endif

/**
 *  @brief  Aligns a variable to a 64-byte boundary using compiler extensions for
 *          compatibility with C 99, as `alignas(64)` is only available in C 11 or C++.
 *          Used internally and recommended for external users.
 */
#if defined(_MSC_VER)
#define NK_ALIGN64 __declspec(align(64))
#elif defined(__GNUC__) || defined(__clang__)
#define NK_ALIGN64 __attribute__((aligned(64)))
#endif

/**
 *  ARM Streaming attributes (require SME-capable compiler: GCC 14+, Clang 16+).
 *  NK_STREAMING_ marks functions that require streaming SVE mode (e.g. FCVTLT).
 *  NK_STREAMING_COMPATIBLE_ marks helpers callable from both streaming and non-streaming mode.
 */
#if NK_TARGET_ARM64_ && NK_TARGET_SME
#define NK_STREAMING_            __arm_streaming
#define NK_STREAMING_COMPATIBLE_ __arm_streaming_compatible
#else
#define NK_STREAMING_
#define NK_STREAMING_COMPATIBLE_
#endif

/**
 *  @brief  Portable casts between SIMD vector types.
 *          MSVC typedefs `__m512bh`, `__m512h`, `__m256bh` as aliases for `__m512i`/`__m256i`,
 *          but rejects C-style casts between them. GCC/Clang define them as distinct types.
 */
#if NK_TARGET_X8664_
#if defined(_MSC_VER)
#define nk_m512bh_from_m512i_(x) (x)
#define nk_m512h_from_m512i_(x)  (x)
#define nk_m512i_from_m512h_(x)  (x)
#define nk_m256bh_from_m256i_(x) (x)
#define nk_m256i_from_m256bh_(x) (x)
#else
#define nk_m512bh_from_m512i_(x) ((__m512bh)(x))
#define nk_m512h_from_m512i_(x)  ((__m512h)(x))
#define nk_m512i_from_m512h_(x)  ((__m512i)(x))
#define nk_m256bh_from_m256i_(x) ((__m256bh)(x))
#define nk_m256i_from_m256bh_(x) ((__m256i)(x))
#endif
#endif

/*  AltiVec defines `bool`, `vector`, and `pixel` as macros, which conflict with C++.
 *  We use `__vector` directly in our code, so undef the problematic macros.
 */
#if NK_TARGET_POWERVSX
#ifdef __cplusplus
#undef bool
#undef vector
#undef pixel
#endif
typedef __vector unsigned char nk_vu8x16_t;
typedef __vector unsigned short nk_vu16x8_t;
typedef __vector unsigned int nk_vu32x4_t;
typedef __vector unsigned long long nk_vu64x2_t;
typedef __vector signed char nk_vi8x16_t;
typedef __vector signed short nk_vi16x8_t;
typedef __vector signed int nk_vi32x4_t;
typedef __vector signed long long nk_vi64x2_t;
typedef __vector float nk_vf32x4_t;
typedef __vector double nk_vf64x2_t;
#endif // NK_TARGET_POWERVSX

/** Copy 16 bits (2 bytes) from source to destination */
#if defined(__GNUC__) || defined(__clang__)
#define nk_copy_bytes_(destination_ptr, source_ptr, count) __builtin_memcpy((destination_ptr), (source_ptr), count)
#else
#include <string.h> // `memcpy`
#define nk_copy_bytes_(destination_ptr, source_ptr, count) memcpy((destination_ptr), (source_ptr), count)
#endif

/** Macro to mark unused parameters (cleaner than (void)variable) */
#define nk_unused_(x) ((void)(x))

/**
 *  @brief C99 static array parameter annotation for minimum array size.
 *
 *  In C, expands to `static n` enabling compiler bounds checking.
 *  In C++, expands to nothing as this syntax is not supported.
 *  @see https://lwn.net/Articles/1046840/
 *
 *  Example usage:
 *  @code{.c}
 *      void hash_digest(uint8_t digest[nk_at_least_(32)]);
 *      void lookup(uint8_t const lut[nk_at_least_(256)]);
 *  @endcode
 */
#if defined(__cplusplus) || defined(_MSC_VER)
#define nk_at_least_(n)
#else
#define nk_at_least_(n) static n
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Packed 8-bit bit-vector (8 booleans in one byte), LSB = dimension 0.
 *  Used for Hamming distance and Jaccard similarity via popcount.
 *  Dimension count must be a multiple of 8; unused bits in the final byte must be zeroed. */
typedef unsigned char nk_u1x8_t;
/** @brief Packed 4-bit signed integer pair (2 × i4 in one byte), [high nibble : low nibble].
 *  Range per element: [−8, +7]. Elements sign-extended to i8 for arithmetic.
 *  Dimension count must be a multiple of 2; unused nibbles in the final byte must be zeroed. */
typedef unsigned char nk_i4x2_t;
/** @brief Packed 4-bit unsigned integer pair (2 × u4 in one byte), [high nibble : low nibble].
 *  Range per element: [0, 15]. Elements zero-extended to u8 for arithmetic.
 *  Dimension count must be a multiple of 2; unused nibbles in the final byte must be zeroed. */
typedef unsigned char nk_u4x2_t;

/** @brief 8-bit E4M3 float (OCP FP8): sign(1) + exponent(4) + mantissa(3), bias=7.
 *  Range: ±448, no infinities (all-ones exponent → NaN at 0x7F/0xFF).
 *  114 of 254 finite values (44.9%) fall in [−1, +1]. */
typedef unsigned char nk_e4m3_t;
/** @brief 8-bit E5M2 float (OCP FP8): sign(1) + exponent(5) + mantissa(2), bias=15.
 *  Range: ±57 344, supports infinities at 0x7C/0xFC.
 *  122 of 248 finite values (49.2%) fall in [−1, +1]. */
typedef unsigned char nk_e5m2_t;
/** @brief 6-bit E2M3 micro-float (OCP MX v1.0): sign(1) + exponent(2) + mantissa(3), bias=1.
 *  Stored as 0b00SEEMMM with 2 bits of padding. Range: ±7.5, no infinities or NaN.
 *  64 total codes: 48 normal, 14 subnormal (exp=0, mant≠0), 2 zeros (±0).
 *  18 of 64 values (28.1%) fall in [−1, +1]. Subnormal values span [±0.125, ±0.875].
 *  Losslessly promotable to E4M3 by rebiasing exponent +6 (normals) or normalizing (subnormals). */
typedef unsigned char nk_e2m3_t;
/** @brief 6-bit E3M2 micro-float (OCP MX v1.0): sign(1) + exponent(3) + mantissa(2), bias=3.
 *  Stored as 0b00SEEEMM with 2 bits of padding. Range: ±28, no infinities or NaN.
 *  64 total codes: 56 normal, 6 subnormal (exp=0, mant≠0), 2 zeros (±0).
 *  26 of 64 values (40.6%) fall in [−1, +1]. Subnormal values span [±0.0625, ±0.1875].
 *  Losslessly promotable to E5M2 by rebiasing exponent +12 (normals) or normalizing (subnormals). */
typedef unsigned char nk_e3m2_t;

/** @brief Signed 8-bit integer. Range: [−128, +127]. */
typedef signed char nk_i8_t;
/** @brief Unsigned 8-bit integer. Range: [0, 255]. */
typedef unsigned char nk_u8_t;
/** @brief Signed 16-bit integer. Range: [−32 768, +32 767]. */
typedef signed short nk_i16_t;
/** @brief Unsigned 16-bit integer. Range: [0, 65 535]. */
typedef unsigned short nk_u16_t;
/** @brief Signed 32-bit integer. Range: [−2³¹, +2³¹−1]. */
typedef signed int nk_i32_t;
/** @brief Unsigned 32-bit integer. Range: [0, 2³²−1]. */
typedef unsigned int nk_u32_t;
/*  On LP64 targets (Linux ARM64, RISC-V 64), `long` and `long long` are both 64-bit but distinct types.
 *  NEON/RVV intrinsics on Linux expect `long*`, while Apple's NEON intrinsics expect `long long*`.
 *  Windows uses LLP64 where `long` is 32-bit, so it must use `long long` for 64-bit types. */
#if ((NK_TARGET_ARM64_ && !defined(NK_DEFINED_APPLE_)) || NK_TARGET_RISCV64_) && !defined(NK_DEFINED_WINDOWS_)
/** @brief Signed 64-bit integer. Range: [−2⁶³, +2⁶³−1]. */
typedef signed long nk_i64_t;
/** @brief Unsigned 64-bit integer. Range: [0, 2⁶⁴−1]. */
typedef unsigned long nk_u64_t;
#else
/** @brief Signed 64-bit integer. Range: [−2⁶³, +2⁶³−1]. */
typedef signed long long nk_i64_t;
/** @brief Unsigned 64-bit integer. Range: [0, 2⁶⁴−1]. */
typedef unsigned long long nk_u64_t;
#endif

/** @brief Single-precision (32-bit) IEEE 754 float. sign(1) + exponent(8) + mantissa(23), bias=127. */
typedef float nk_f32_t;
/** @brief Double-precision (64-bit) IEEE 754 float. sign(1) + exponent(11) + mantissa(52), bias=1023. */
typedef double nk_f64_t;

#if NK_TARGET_X8664_ || NK_TARGET_ARM64_ || NK_TARGET_RISCV64_ || NK_TARGET_POWER64_ || NK_TARGET_LOONGARCH64_
#define NK_IS_64BIT_ 1
#else
#define NK_IS_64BIT_ 0
#endif

#if NK_IS_64BIT_
typedef nk_u64_t nk_size_t;
typedef nk_i64_t nk_ssize_t;
#else
typedef nk_u32_t nk_size_t;
typedef nk_i32_t nk_ssize_t;
#endif
typedef nk_f64_t nk_fmax_t;

#define NK_SIZE_MAX ((nk_size_t) - 1)

#define NK_F64_MAX 1.7976931348623157e+308
#define NK_F64_MIN (-1.7976931348623157e+308)
#define NK_F32_MAX 3.402823466e+38f
#define NK_F32_MIN (-3.402823466e+38f)

#define NK_I64_MAX 9223372036854775807LL
#define NK_I64_MIN (-9223372036854775807LL - 1LL)
#define NK_U64_MAX 18446744073709551615ULL
#define NK_U64_MIN 0x0ULL

#define NK_I32_MAX 2147483647
#define NK_I32_MIN (-2147483647 - 1)
#define NK_U32_MAX 4294967295U
#define NK_U32_MIN 0x0U

#define NK_I16_MAX 32767
#define NK_I16_MIN (-32767 - 1)
#define NK_U16_MAX 65535U
#define NK_U16_MIN 0x0U

#define NK_I8_MAX 127
#define NK_I8_MIN (-127 - 1)
#define NK_U8_MAX 255U
#define NK_U8_MIN 0x0U

#define NK_F16_MAX_AS_U16 0x7BFF // IEEE 754 binary16: +65504.0
#define NK_F16_MIN_AS_U16 0xFBFF // IEEE 754 binary16: -65504.0

#define NK_F16_MAX nk_u16_as_f16_(0x7BFF)
#define NK_F16_MIN nk_u16_as_f16_(0xFBFF)

#define NK_BF16_MAX_AS_U16 0x7F7F // BFloat16: ~+3.39e38
#define NK_BF16_MIN_AS_U16 0xFF7F // BFloat16: ~-3.39e38

#define NK_BF16_MAX nk_u16_as_bf16_(0x7F7F)
#define NK_BF16_MIN nk_u16_as_bf16_(0xFF7F)

#define NK_E4M3_MAX 0x7E // FP8 E4M3: +448.0
#define NK_E4M3_MIN 0xFE // FP8 E4M3: -448.0

#define NK_E5M2_MAX 0x7B // FP8 E5M2: +57344.0
#define NK_E5M2_MIN 0xFB // FP8 E5M2: -57344.0

#define NK_E2M3_MAX 0x1F // FP6 E2M3: +7.5
#define NK_E2M3_MIN 0x3F // FP6 E2M3: -7.5

#define NK_E3M2_MAX 0x1F // FP6 E3M2: +28.0
#define NK_E3M2_MIN 0x3F // FP6 E3M2: -28.0

#define NK_BITS_PER_BYTE 8

/**
 *  @brief  Enumeration of supported scalar data types.
 *
 *  Includes complex type descriptors which in C code would use the real counterparts,
 *  but the independent flags contain metadata to be passed between programming language
 *  interfaces.
 */
typedef enum {
    nk_dtype_unknown_k = 0, ///< Unknown data type
    nk_u1_k = 1 << 1,       ///< Single-bit values packed into 8-bit words

    nk_i8_k = 1 << 2,  ///< 8-bit signed integer
    nk_i16_k = 1 << 3, ///< 16-bit signed integer
    nk_i32_k = 1 << 4, ///< 32-bit signed integer
    nk_i64_k = 1 << 5, ///< 64-bit signed integer

    nk_u8_k = 1 << 6,  ///< 8-bit unsigned integer
    nk_u16_k = 1 << 7, ///< 16-bit unsigned integer
    nk_u32_k = 1 << 8, ///< 32-bit unsigned integer
    nk_u64_k = 1 << 9, ///< 64-bit unsigned integer

    nk_f64_k = 1 << 10,  ///< Double precision floating point
    nk_f32_k = 1 << 11,  ///< Single precision floating point
    nk_f16_k = 1 << 12,  ///< Half precision floating point
    nk_bf16_k = 1 << 13, ///< Brain floating point

    nk_e4m3_k = 1 << 14, ///< FP8 E4M3 floating point
    nk_e5m2_k = 1 << 15, ///< FP8 E5M2 floating point
    nk_i4_k = 1 << 16,   ///< 4-bit signed integers packed into 8-bit words
    nk_u4_k = 1 << 17,   ///< 4-bit unsigned integers packed into 8-bit words
    nk_e2m3_k = 1 << 18, ///< FP6 E2M3 floating point
    nk_e3m2_k = 1 << 19, ///< FP6 E3M2 floating point

    nk_f64c_k = 1 << 20,  ///< Complex double precision floating point
    nk_f32c_k = 1 << 21,  ///< Complex single precision floating point
    nk_f16c_k = 1 << 22,  ///< Complex half precision floating point
    nk_bf16c_k = 1 << 23, ///< Complex brain floating point
} nk_dtype_t;

typedef enum {
    nk_dtype_family_unknown_k = 0,
    nk_dtype_family_float_k,
    nk_dtype_family_complex_float_k,
    nk_dtype_family_int_k,
    nk_dtype_family_uint_k,
} nk_dtype_family_t;

/** @brief Classifies the family of the dtype. */
NK_PUBLIC nk_dtype_family_t nk_dtype_family(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return nk_dtype_family_float_k;
    case nk_f32_k: return nk_dtype_family_float_k;
    case nk_f16_k: return nk_dtype_family_float_k;
    case nk_bf16_k: return nk_dtype_family_float_k;
    case nk_e4m3_k: return nk_dtype_family_float_k;
    case nk_e5m2_k: return nk_dtype_family_float_k;
    case nk_e2m3_k: return nk_dtype_family_float_k;
    case nk_e3m2_k: return nk_dtype_family_float_k;
    case nk_f64c_k: return nk_dtype_family_complex_float_k;
    case nk_f32c_k: return nk_dtype_family_complex_float_k;
    case nk_f16c_k: return nk_dtype_family_complex_float_k;
    case nk_bf16c_k: return nk_dtype_family_complex_float_k;
    case nk_u1_k: return nk_dtype_family_uint_k;
    case nk_u4_k: return nk_dtype_family_uint_k;
    case nk_u8_k: return nk_dtype_family_uint_k;
    case nk_u16_k: return nk_dtype_family_uint_k;
    case nk_u32_k: return nk_dtype_family_uint_k;
    case nk_u64_k: return nk_dtype_family_uint_k;
    case nk_i4_k: return nk_dtype_family_int_k;
    case nk_i8_k: return nk_dtype_family_int_k;
    case nk_i16_k: return nk_dtype_family_int_k;
    case nk_i32_k: return nk_dtype_family_int_k;
    case nk_i64_k: return nk_dtype_family_int_k;
    default: return nk_dtype_family_unknown_k;
    }
}

/** @brief Returns the number of bits in a single scalar of a given type. */
NK_PUBLIC nk_size_t nk_dtype_bits(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return 64;
    case nk_f32_k: return 32;
    case nk_f16_k: return 16;
    case nk_bf16_k: return 16;
    case nk_e4m3_k: return 8;
    case nk_e5m2_k: return 8;
    case nk_e2m3_k: return 8;
    case nk_e3m2_k: return 8;
    case nk_f64c_k: return 128;
    case nk_f32c_k: return 64;
    case nk_f16c_k: return 32;
    case nk_bf16c_k: return 32;
    case nk_u1_k: return 1;
    case nk_u4_k: return 4;
    case nk_u8_k: return 8;
    case nk_u16_k: return 16;
    case nk_u32_k: return 32;
    case nk_u64_k: return 64;
    case nk_i4_k: return 4;
    case nk_i8_k: return 8;
    case nk_i16_k: return 16;
    case nk_i32_k: return 32;
    case nk_i64_k: return 64;
    default: return 0;
    }
}

/** @brief Returns how many logical dimensions are packed into one storage value.
 *  For sub-byte types multiple dimensions share a single byte container.
 *  For byte-or-larger types this is always 1. */
NK_PUBLIC nk_size_t nk_dimensions_per_value(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_u1_k: return 8;
    case nk_i4_k: return 2;
    case nk_u4_k: return 2;
    default: return 1;
    }
}

/** @brief Half-precision (16-bit) IEEE 754 float.
 *
 *  Layout: sign(1) + exponent(5) + mantissa(10), bias=15.
 *  Range: ±65 504, epsilon at 1.0 ≈ 9.77×10⁻⁴. 30 722 of 63 488 finite values (48.4%) in [−1, +1].
 *
 *  - GCC or Clang on 64-bit Arm: `__fp16`, may require `-mfp16-format` option.
 *  - GCC or Clang on 64-bit x86: `_Float16`.
 *  - Default: `unsigned short`.
 */
#if !defined(NK_NATIVE_F16) || NK_NATIVE_F16
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__ARM_ARCH) || defined(__aarch64__)) && \
    (defined(__ARM_FP16_FORMAT_IEEE))
#undef NK_NATIVE_F16
#define NK_NATIVE_F16 1
typedef __fp16 nk_f16_t;
#elif ((defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__)) && \
       (defined(__AVX512FP16__)))
typedef _Float16 nk_f16_t;
#undef NK_NATIVE_F16
#define NK_NATIVE_F16 1
#else // Unknown compiler or architecture
#undef NK_NATIVE_F16
#define NK_NATIVE_F16 0
#endif // Unknown compiler or architecture
#endif // !NK_NATIVE_F16

#if !NK_NATIVE_F16
typedef unsigned short nk_f16_t;
#endif

#if !defined(NK_NATIVE_BF16) || NK_NATIVE_BF16
/** @brief BFloat16 (16-bit) float — truncated IEEE 754 single-precision.
 *
 *  Layout: sign(1) + exponent(8) + mantissa(7), bias=127.
 *  Same dynamic range as f32, epsilon ≈ 7.81×10⁻³.
 *  32 514 of 65 280 finite values (49.8%) in [−1, +1]. Wider range than f16 but lower precision.
 *
 *  - GCC or Clang: `__bf16`
 *  - Default: `unsigned short`.
 *
 *  The compilers have added `__bf16` support in compliance with the x86-64 psABI spec.
 *  The motivation for this new special type is summed up as:
 *
 *      Currently `__bfloat16` is a typedef of short, which creates a problem where the
 *      compiler does not raise any alarms if it is used to add, subtract, multiply or
 *      divide, but the result of the calculation is actually meaningless.
 *      To solve this problem, a real scalar type `__Bfloat16` needs to be introduced.
 *      It is mainly used for intrinsics, not available for C standard operators.
 *      `__Bfloat16` will also be used for movement like passing parameter, load and store,
 *      vector initialization, vector shuffle, and etc. It creates a need for a
 *      corresponding psABI.
 *
 *  @warning Apple Clang has hard time with bf16.
 *  https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms
 *  https://forums.developer.apple.com/forums/thread/726201
 *  https://www.phoronix.com/news/GCC-LLVM-bf16-BFloat16-Type
 */
#if (defined(__GNUC__) || defined(__clang__)) && ((defined(__ARM_BF16_FORMAT_ALTERNATIVE)) || (defined(__AVX512BF16__)))
#undef NK_NATIVE_BF16
#define NK_NATIVE_BF16 1
typedef __bf16 nk_bf16_t;
#else // Unknown compiler or architecture
#undef NK_NATIVE_BF16
#define NK_NATIVE_BF16 0
#endif // Unknown compiler or architecture
#endif // !NK_NATIVE_BF16

#if !NK_NATIVE_BF16
typedef unsigned short nk_bf16_t;
#endif

/**
 *  @brief  Alias for the half-precision floating-point type on Arm.
 *
 *  Clang and GCC bring the `float16_t` symbol when you compile for Aarch64.
 *  MSVC lacks it, and it's `vld1_f16`-like intrinsics are in reality macros,
 *  that cast to 16-bit integers internally, instead of using floats.
 *  Some of those are defined as aliases, so we use `#define` preprocessor
 *  directives instead of `typedef` to avoid errors.
 */
#if NK_TARGET_ARM64_
#if defined(_MSC_VER)
#define nk_f16_for_arm_simd_t  nk_f16_t
#define nk_bf16_for_arm_simd_t nk_bf16_t
#else
#define nk_f16_for_arm_simd_t  float16_t
#define nk_bf16_for_arm_simd_t bfloat16_t
#endif
#endif

/**
 *  RISC-V Vector (RVV) intrinsics use `_Float16` for half-precision floats.
 *  This is the standard C23 type, also available in GCC/Clang with RVV extensions.
 */
#if NK_TARGET_RISCV64_
#define nk_f16_for_rvv_intrinsics_t _Float16
#endif

/*
 *  Let's make sure the sizes of the types are as expected.
 *  In C the `_Static_assert` is only available with C11 and later.
 */
#define NK_STATIC_ASSERT(cond, msg) typedef char static_assertion_##msg[(cond) ? 1 : -1]
NK_STATIC_ASSERT(sizeof(nk_u1x8_t) == 1, nk_u1x8_t_must_be_1_byte);
NK_STATIC_ASSERT(sizeof(nk_i4x2_t) == 1, nk_i4_t_must_be_1_byte);
NK_STATIC_ASSERT(sizeof(nk_u4x2_t) == 1, nk_u4_t_must_be_1_byte);
NK_STATIC_ASSERT(sizeof(nk_e4m3_t) == 1, nk_e4m3_t_must_be_1_byte);
NK_STATIC_ASSERT(sizeof(nk_e5m2_t) == 1, nk_e5m2_t_must_be_1_byte);
NK_STATIC_ASSERT(sizeof(nk_i8_t) == 1, nk_i8_t_must_be_1_byte);
NK_STATIC_ASSERT(sizeof(nk_u8_t) == 1, nk_u8_t_must_be_1_byte);
NK_STATIC_ASSERT(sizeof(nk_i16_t) == 2, nk_i16_t_must_be_2_bytes);
NK_STATIC_ASSERT(sizeof(nk_u16_t) == 2, nk_u16_t_must_be_2_bytes);
NK_STATIC_ASSERT(sizeof(nk_i32_t) == 4, nk_i32_t_must_be_4_bytes);
NK_STATIC_ASSERT(sizeof(nk_u32_t) == 4, nk_u32_t_must_be_4_bytes);
NK_STATIC_ASSERT(sizeof(nk_i64_t) == 8, nk_i64_t_must_be_8_bytes);
NK_STATIC_ASSERT(sizeof(nk_u64_t) == 8, nk_u64_t_must_be_8_bytes);
NK_STATIC_ASSERT(sizeof(nk_f32_t) == 4, nk_f32_t_must_be_4_bytes);
NK_STATIC_ASSERT(sizeof(nk_f64_t) == 8, nk_f64_t_must_be_8_bytes);
NK_STATIC_ASSERT(sizeof(nk_f16_t) == 2, nk_f16_t_must_be_2_bytes);
NK_STATIC_ASSERT(sizeof(nk_bf16_t) == 2, nk_bf16_t_must_be_2_bytes);

#define nk_assign_from_to_(src, dest) (*(dest) = *(src))

/** @brief 16-bit union for f16/bf16/u16/i16 bit manipulation. */
typedef union NK_MAY_ALIAS_ {
    nk_u16_t u;
    nk_i16_t i;
    nk_f16_t f;
    nk_bf16_t bf;
} nk_fui16_t;

/** @brief 32-bit union for f32/u32/i32 bit manipulation. */
typedef union NK_MAY_ALIAS_ {
    nk_u32_t u;
    nk_i32_t i;
    nk_f32_t f;
} nk_fui32_t;

/** @brief 64-bit union for f64/u64/i64 bit manipulation. */
typedef union NK_MAY_ALIAS_ {
    nk_u64_t u;
    nk_i64_t i;
    nk_f64_t f;
} nk_fui64_t;

/** @brief Half-precision (32-bit) complex number — {real: f16, imag: f16}. Kernel outputs widened to f32c. */
typedef struct {
    nk_f16_t real;
    nk_f16_t imag;
} nk_f16c_t;

/** @brief BFloat16 (32-bit) complex number — {real: bf16, imag: bf16}. Kernel outputs widened to f32c. */
typedef struct {
    nk_bf16_t real;
    nk_bf16_t imag;
} nk_bf16c_t;

/** @brief Single-precision (64-bit) complex number — {real: f32, imag: f32}. */
typedef struct {
    nk_f32_t real;
    nk_f32_t imag;
} nk_f32c_t;

/** @brief Double-precision (128-bit) complex number — {real: f64, imag: f64}. */
typedef struct {
    nk_f64_t real;
    nk_f64_t imag;
} nk_f64c_t;

/** @brief  Small 4-byte memory slice viewable as different types. */
typedef union NK_MAY_ALIAS_ nk_b32_vec_t {
    nk_u32_t u32;
    nk_i32_t i32;
    nk_f32_t f32;
    nk_u8_t u8s[4];
    nk_i8_t i8s[4];
    nk_u16_t u16s[2];
    nk_i16_t i16s[2];
    nk_e4m3_t e4m3s[4];
    nk_e5m2_t e5m2s[4];
} nk_b32_vec_t;

/** @brief  Small 8-byte memory slice viewable as different types. */
typedef union NK_MAY_ALIAS_ nk_b64_vec_t {
#if NK_TARGET_NEON
    uint8x8_t u8x8;
    uint16x4_t u16x4;
    uint32x2_t u32x2;
    int8x8_t i8x8;
    int16x4_t i16x4;
    int32x2_t i32x2;
    float32x2_t f32x2;
#endif
#if NK_TARGET_NEONHALF
    float16x4_t f16x4;
#endif
    nk_u8_t u8s[8];
    nk_u16_t u16s[4];
    nk_u32_t u32s[2];
    nk_u64_t u64;
    nk_i8_t i8s[8];
    nk_i16_t i16s[4];
    nk_i32_t i32s[2];
    nk_i64_t i64;
    nk_f16_t f16s[4];
    nk_bf16_t bf16s[4];
    nk_f32_t f32s[2];
} nk_b64_vec_t;

/** @brief  Small 16-byte memory slice viewable as different types. */
typedef union NK_MAY_ALIAS_ nk_b128_vec_t {
#if NK_TARGET_HASWELL || NK_TARGET_LOONGSONASX
    __m128i xmm;
    __m128d xmm_pd;
    __m128 xmm_ps;
#endif
#if NK_TARGET_V128RELAXED
    v128_t v128;
#endif
#if NK_TARGET_NEON
    uint8x16_t u8x16;
    uint16x8_t u16x8;
    uint32x4_t u32x4;
    uint64x2_t u64x2;
    int8x16_t i8x16;
    int16x8_t i16x8;
    int32x4_t i32x4;
    int64x2_t i64x2;
    float32x4_t f32x4;
#endif
#if NK_TARGET_NEON && NK_TARGET_ARM64_ // double-precision NEON requires AArch64
    float64x2_t f64x2;
#endif
#if NK_TARGET_NEONHALF
    float16x8_t f16x8;
#endif
#if NK_TARGET_POWERVSX
    nk_vu8x16_t vu8x16;
    nk_vu16x8_t vu16x8;
    nk_vu32x4_t vu32x4;
    nk_vu64x2_t vu64x2;
    nk_vi8x16_t vi8x16;
    nk_vi16x8_t vi16x8;
    nk_vi32x4_t vi32x4;
    nk_vi64x2_t vi64x2;
    nk_vf32x4_t vf32x4;
    nk_vf64x2_t vf64x2;
#endif

    nk_u8_t u8s[16];
    nk_u16_t u16s[8];
    nk_u32_t u32s[4];
    nk_u64_t u64s[2];
    nk_i8_t i8s[16];
    nk_i16_t i16s[8];
    nk_i32_t i32s[4];
    nk_i64_t i64s[2];
    nk_f16_t f16s[8];
    nk_bf16_t bf16s[8];
    nk_e4m3_t e4m3s[16];
    nk_e5m2_t e5m2s[16];
    nk_e2m3_t e2m3s[16];
    nk_e3m2_t e3m2s[16];
    nk_f32_t f32s[4];
    nk_f64_t f64s[2];
} nk_b128_vec_t;

/** @brief  Small 32-byte memory slice viewable as different types. */
typedef union NK_MAY_ALIAS_ nk_b256_vec_t {
#if NK_TARGET_HASWELL || NK_TARGET_LOONGSONASX
    __m256i ymm;
    __m256d ymm_pd;
    __m256 ymm_ps;
    __m128i xmms[2];
#endif
#if NK_TARGET_V128RELAXED
    v128_t v128s[2];
#endif
#if NK_TARGET_NEON
    uint8x16_t u8x16s[2];
    uint16x8_t u16x8s[2];
    uint32x4_t u32x4s[2];
    uint64x2_t u64x2s[2];
    int8x16_t i8x16s[2];
    int16x8_t i16x8s[2];
    int32x4_t i32x4s[2];
    int64x2_t i64x2s[2];
    float32x4_t f32x4s[2];
#endif
#if NK_TARGET_NEON && NK_TARGET_ARM64_ // double-precision NEON requires AArch64
    float64x2_t f64x2s[2];
#endif
#if NK_TARGET_POWERVSX
    nk_vu8x16_t vu8x16s[2];
    nk_vu16x8_t vu16x8s[2];
    nk_vu32x4_t vu32x4s[2];
    nk_vu64x2_t vu64x2s[2];
    nk_vi8x16_t vi8x16s[2];
    nk_vi16x8_t vi16x8s[2];
    nk_vi32x4_t vi32x4s[2];
    nk_vi64x2_t vi64x2s[2];
    nk_vf32x4_t vf32x4s[2];
    nk_vf64x2_t vf64x2s[2];
#endif

    nk_u8_t u8s[32];
    nk_u16_t u16s[16];
    nk_u32_t u32s[8];
    nk_u64_t u64s[4];
    nk_i8_t i8s[32];
    nk_i16_t i16s[16];
    nk_i32_t i32s[8];
    nk_i64_t i64s[4];
    nk_f16_t f16s[16];
    nk_bf16_t bf16s[16];
    nk_e4m3_t e4m3s[32];
    nk_e5m2_t e5m2s[32];
    nk_e2m3_t e2m3s[32];
    nk_e3m2_t e3m2s[32];
    nk_f32_t f32s[8];
    nk_f64_t f64s[4];
} nk_b256_vec_t;

/** @brief  Small 64-byte memory slice viewable as different types.
 *
 *  TODO: On GCC and Clang we use `__transparent_union__` attribute to allow implicit conversions
 *  between the different vector types when passing them as function arguments. The most important side-effect
 *  of this is that the argument of such type is passed to functions using the calling convention of the first
 *  member of the union, which in our case is a register-based calling convention for SIMD types.
 */
typedef union NK_MAY_ALIAS_ nk_b512_vec_t {
#if NK_TARGET_SKYLAKE
    __m512i zmm;
    __m512d zmm_pd;
    __m512 zmm_ps;
#endif
#if NK_TARGET_HASWELL
    __m256i ymms[2];
    __m256d ymms_pd[2];
    __m256 ymms_ps[2];
    __m128i xmms[4];
    __m128d xmms_pd[4];
    __m128 xmms_ps[4];
#endif
#if NK_TARGET_NEON
    uint8x16_t u8x16s[4];
    uint16x8_t u16x8s[4];
    uint32x4_t u32x4s[4];
    uint64x2_t u64x2s[4];
#endif
    nk_u8_t u8s[64];
    nk_u16_t u16s[32];
    nk_u32_t u32s[16];
    nk_u64_t u64s[8];
    nk_i8_t i8s[64];
    nk_i16_t i16s[32];
    nk_i32_t i32s[16];
    nk_i64_t i64s[8];
    nk_f16_t f16s[32];
    nk_bf16_t bf16s[32];
    nk_f32_t f32s[16];
    nk_f64_t f64s[8];
    nk_e4m3_t e4m3s[64];
    nk_e5m2_t e5m2s[64];
    nk_e2m3_t e2m3s[64];
    nk_e3m2_t e3m2s[64];
} nk_b512_vec_t;

/**
 *  @brief Advances the Multi-Dimensional iterator to the next set of indicies.
 *  @param[in] extents The extents of the tensor, defined by an array of at least `rank` scalars.
 *  @param[in] strides The @b signed strides of the tensor in bytes, defined by an array of at least `rank` scalars.
 *  @param[in] rank The number of dimensions in the tensor (its rank).
 *  @param[inout] coordinates The array of offsets along each of `rank` dimensions, which will be updated.
 *  @param[inout] byte_offset The @b signed byte offset of the current element, which will be advanced.
 *  @return 1 if the iterator was successfully advanced, 0 if the end of iteration was reached.
 *
 *  For flexibility, the API is decoupled from from the `nk_tensor_position_t` structure, and
 *  can be used on any-rank tensors, independent of the `NK_TENSOR_MAX_RANK` constant.
 */
NK_PUBLIC int nk_tensor_position_next(                                   //
    nk_size_t const *extents, nk_ssize_t const *strides, nk_size_t rank, //
    nk_size_t *coordinates, nk_ssize_t *byte_offset) {
    // Start from last dimension and move backward
    for (nk_size_t i = rank; i-- > 0;) {
        coordinates[i]++;
        *byte_offset += strides[i];
        if (coordinates[i] < extents[i]) return 1; // Successfully moved to the next index
        coordinates[i] = 0;                        // Reset this dimension counter
        *byte_offset -= strides[i] * extents[i];   // Discard the running progress along this dimension
    }
    // If we reach here, we've iterated over all elements
    return 0; // End of iteration
}

/**
 *  @brief Advances the Multi-Dimensional iterator to the provided coordinates, updating the byte offset.
 *  @param[in] extents The extents of the tensor, defined by an array of at least `rank` scalars.
 *  @param[in] strides The @b signed strides of the tensor in bytes, defined by an array of at least `rank` scalars.
 *  @param[in] rank The number of dimensions in the tensor (its rank).
 *  @param[in] coordinates The array of offsets along each of `rank` dimensions, which will be updated.
 *  @param[out] byte_offset The byte offset of the current element, which will be advanced.
 *  @return 1 if the offset was successfully advanced, 0 if the end of iteration was reached.
 */
NK_PUBLIC int nk_tensor_position_linearize(                              //
    nk_size_t const *extents, nk_ssize_t const *strides, nk_size_t rank, //
    nk_size_t const *coordinates, nk_ssize_t *byte_offset) {

    nk_ssize_t result = 0;
    for (nk_size_t i = 0; i < rank; i++) {
        // Ensure the coordinates is within bounds for the given dimension
        if (coordinates[i] >= extents[i]) return 0; // Invalid coordinates, out of bounds
        // Update the byte offset by multiplying the coordinates by the stride
        result += coordinates[i] * strides[i];
    }
    *byte_offset = result;
    return 1; // Successfully calculated global and byte offsets
}

/**
 *  @brief  A @b beefy structure to iterate through Multi-Dimensional arrays.
 *          Occupies 512 + 8 = 520 bytes on a 64-bit machine, or @b 9 cache-lines, by default.
 *
 *  When advancing through a structure, its overall size and strides should be stored somewhere else.
 *  The `byte_offset` starts at zero and grow monotonically during iteration, if the strides are positive.
 */
typedef struct nk_tensor_position_t {
    nk_size_t coordinates[NK_TENSOR_MAX_RANK]; // Coordinate offsets along each dimension
    nk_ssize_t byte_offset;                    // Byte offset of the current element
} nk_tensor_position_t;

NK_PUBLIC void nk_tensor_position_init(nk_tensor_position_t *tensor_position) {
    for (nk_size_t i = 0; i < NK_TENSOR_MAX_RANK; i++) tensor_position->coordinates[i] = 0;
    tensor_position->byte_offset = 0;
}

/**
 *  @brief  A @b beefy structure describing the shape and memory layout of a Multi-Dimensional array.
 *          Similar to `md::span` in C++20 and `numpy.ndarray` in Python, but with a focus on compatibility.
 *          Occupies 512 + 512 + 8 = 2052 bytes on a 64-bit machine, or @b 17 cache-lines, by default.
 *
 *  Unlike NumPy and the CPython "Buffer Protocol", we don't use `suboffsets` for pointer indirection.
 *  The logic is that such layouts aren't friendly to conventional SIMD operations and dense matrix algorithms.
 *  If the tensor is sparse, consider using a different data structure or a different memory layout.
 *
 *  Most NumKong algorithms don't work with the entire structure, but expect the fields to be passed separately.
 *  It would also require storing the @b start-pointer and the @b dtype/item-size separately, as it's not
 *  stored inside the structure.
 */
typedef struct nk_tensor_shape_t {
    nk_size_t extents[NK_TENSOR_MAX_RANK];  /// Number of elements along each dimension
    nk_ssize_t strides[NK_TENSOR_MAX_RANK]; /// Strides of the tensor in bytes
    nk_size_t rank;                         /// Number of dimensions in the tensor
} nk_tensor_shape_t;

NK_PUBLIC void nk_tensor_shape_init(nk_tensor_shape_t *tensor_shape) {
    for (nk_size_t i = 0; i < NK_TENSOR_MAX_RANK; i++) tensor_shape->extents[i] = 0, tensor_shape->strides[i] = 0;
    tensor_shape->rank = 0;
}

NK_INTERNAL nk_u32_t nk_u32_rol(nk_u32_t x, int n) { return (x << n) | (x >> (32 - n)); }
NK_INTERNAL nk_u16_t nk_u16_rol(nk_u16_t x, int n) { return (x << n) | (x >> (16 - n)); }
NK_INTERNAL nk_u8_t nk_u8_rol(nk_u8_t x, int n) { return (x << n) | (x >> (8 - n)); }
NK_INTERNAL nk_u32_t nk_u32_ror(nk_u32_t x, int n) { return (x >> n) | (x << (32 - n)); }
NK_INTERNAL nk_u16_t nk_u16_ror(nk_u16_t x, int n) { return (x >> n) | (x << (16 - n)); }
NK_INTERNAL nk_u8_t nk_u8_ror(nk_u8_t x, int n) { return (x >> n) | (x << (8 - n)); }

/**
 *  @brief  SWAR population count for 64-bit integers.
 *
 *  Classic algorithm from Hacker's Delight using parallel bit summation:
 *  - Step 1: Count bits in pairs (2-bit sums)
 *  - Step 2: Count bits in nibbles (4-bit sums)
 *  - Step 3: Count bits in bytes (8-bit sums)
 *  - Step 4: Horizontal sum via multiply - each byte contributes to bits 56-63
 *
 *  Cost: ~12 ALU ops, zero memory access (vs 8 table lookups for byte-wise).
 */
NK_INTERNAL nk_u64_t nk_u64_popcount_(nk_u64_t x) {
    x = x - ((x >> 1) & 0x5555555555555555ull);
    x = (x & 0x3333333333333333ull) + ((x >> 2) & 0x3333333333333333ull);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0Full;
    return (x * 0x0101010101010101ull) >> 56;
}

NK_INTERNAL unsigned char nk_u1x8_popcount_(nk_u1x8_t x) {
    static unsigned char lookup_table[256] = {
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, //
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};
    return lookup_table[x];
}

/** @brief Divides the number rounding up to the next multiple of the given divisor. */
NK_PUBLIC nk_size_t nk_size_divide_round_up_(nk_size_t number, nk_size_t divisor) NK_STREAMING_COMPATIBLE_ {
    return (number + divisor - 1) / divisor;
}

/** @brief Rounds up the number to the next multiple of the given divisor. */
NK_PUBLIC nk_size_t nk_size_round_up_to_multiple_(nk_size_t number, nk_size_t divisor) NK_STREAMING_COMPATIBLE_ {
    return nk_size_divide_round_up_(number, divisor) * divisor;
}

NK_INTERNAL nk_f32_t nk_f32_abs_(nk_f32_t x) { return x < 0 ? -x : x; }
NK_INTERNAL nk_f64_t nk_f64_abs_(nk_f64_t x) { return x < 0 ? -x : x; }
NK_INTERNAL nk_i64_t nk_i64_abs_(nk_i64_t x) { return x < 0 ? -x : x; }
NK_INTERNAL nk_u64_t nk_u64_abs_(nk_u64_t x) { return x; }
NK_INTERNAL nk_i64_t nk_i32_abs_(nk_i32_t x) { return x < 0 ? -x : x; }
NK_INTERNAL nk_u32_t nk_u32_abs_(nk_u32_t x) { return x; }

/** @brief Extract low (bits 0-3) unsigned nibble from packed u4x2 byte. */
NK_INTERNAL nk_u8_t nk_u4x2_low_(nk_u4x2_t byte_val) { return byte_val & 0x0F; }
/** @brief Extract high (bits 4-7) unsigned nibble from packed u4x2 byte. */
NK_INTERNAL nk_u8_t nk_u4x2_high_(nk_u4x2_t byte_val) { return (byte_val >> 4) & 0x0F; }

/** @brief Extract low (bits 0-3) signed nibble from packed i4x2 byte as i8. */
NK_INTERNAL nk_i8_t nk_i4x2_low_(nk_i4x2_t byte_val) { return (nk_i8_t)(((byte_val & 0x0F) ^ 8) - 8); }
/** @brief Extract high (bits 4-7) signed nibble from packed i4x2 byte as i8. */
NK_INTERNAL nk_i8_t nk_i4x2_high_(nk_i4x2_t byte_val) { return (nk_i8_t)((((byte_val >> 4) & 0x0F) ^ 8) - 8); }

/** @brief Extract n-th nibble (n=0: low, n=1: high) — branchless. */
NK_INTERNAL nk_u8_t nk_u4x2_get_(nk_u4x2_t byte_val, int n) { return (byte_val >> ((n & 1) * 4)) & 0x0F; }
NK_INTERNAL nk_i8_t nk_i4x2_get_(nk_i4x2_t byte_val, int n) {
    nk_u8_t nibble = (byte_val >> ((n & 1) * 4)) & 0x0F;
    return (nk_i8_t)((nibble ^ 8) - 8);
}

/** @brief Extract bit at position n (0-7) from packed u1x8 byte. */
NK_INTERNAL nk_u8_t nk_u1x8_get_(nk_u1x8_t byte_val, int n) { return (byte_val >> (n & 7)) & 1; }

NK_INTERNAL nk_f16_t nk_u16_as_f16_(nk_u16_t bits) {
    nk_fui16_t c;
    c.u = bits;
    return c.f;
}
NK_INTERNAL nk_u16_t nk_f16_as_u16_(nk_f16_t x) {
    nk_fui16_t c;
    c.f = x;
    return c.u;
}
NK_INTERNAL nk_bf16_t nk_u16_as_bf16_(nk_u16_t bits) {
    nk_fui16_t c;
    c.u = bits;
    return c.bf;
}

NK_INTERNAL void nk_f64_from_i64_(nk_i64_t const *src, nk_f64_t *dest) { *dest = (nk_f64_t)*src; }
NK_INTERNAL void nk_f64_from_u64_(nk_u64_t const *src, nk_f64_t *dest) { *dest = (nk_f64_t)*src; }
NK_INTERNAL void nk_f32_from_i32_(nk_i32_t const *src, nk_f32_t *dest) { *dest = (nk_f32_t)*src; }
NK_INTERNAL void nk_f32_from_u32_(nk_u32_t const *src, nk_f32_t *dest) { *dest = (nk_f32_t)*src; }
NK_INTERNAL void nk_f32_from_f64_(nk_f64_t const *src, nk_f32_t *dest) { *dest = (nk_f32_t)*src; }

/** @brief E4M3: NaN when (raw & 0x7F) == 0x7F  (two NaN values: 0x7F, 0xFF). */
NK_INTERNAL int nk_e4m3_is_nan_(nk_e4m3_t x) { return (x & 0x7F) == 0x7F; }

/** @brief E5M2: NaN when exponent=31 and mantissa!=0, i.e. (raw & 0x7F) > 0x7C.
 *  Values: 0x7D-0x7F (positive), 0xFD-0xFF (negative). Infinity = 0x7C/0xFC is NOT NaN. */
NK_INTERNAL int nk_e5m2_is_nan_(nk_e5m2_t x) { return (x & 0x7F) > 0x7C; }

/** @brief F16: NaN when (raw & 0x7FFF) > 0x7C00. */
NK_INTERNAL int nk_f16_is_nan_(nk_f16_t x) {
    nk_fui16_t x_fui;
    x_fui.f = x;
    return (x_fui.u & 0x7FFF) > 0x7C00;
}

/** @brief BF16: NaN when (raw & 0x7FFF) > 0x7F80. */
NK_INTERNAL int nk_bf16_is_nan_(nk_bf16_t x) {
    nk_fui16_t x_fui;
    x_fui.bf = x;
    return (x_fui.u & 0x7FFF) > 0x7F80;
}

/*  Safe SVE vector-length queries usable from non-streaming context.
 *  On Apple M4 (and other SME-only-SVE cores), SVE instructions like CNTW/CNTH/CNTB
 *  trap with SIGILL outside streaming mode. These helpers bracket the query with
 *  SMSTART SM / SMSTOP SM so the calling function's ABI is unchanged.
 *  Inside `__arm_locally_streaming` functions the plain `svcntXX()` intrinsics are fine.
 */
#if NK_TARGET_ARM64_ && NK_TARGET_SME
/** @brief Streaming SVL byte-element count (SVL/8) via SMSTART SM bracket. */
NK_INTERNAL nk_size_t nk_sme_cntb_(void) {
    nk_u64_t r;
    __asm__ __volatile__("smstart sm\n\t" "cntb %0\n\t" "smstop sm" : "=r"(r));
    return (nk_size_t)r;
}
/** @brief Streaming SVL half-element count (SVL/16) via SMSTART SM bracket. */
NK_INTERNAL nk_size_t nk_sme_cnth_(void) {
    nk_u64_t r;
    __asm__ __volatile__("smstart sm\n\t" "cnth %0\n\t" "smstop sm" : "=r"(r));
    return (nk_size_t)r;
}
/** @brief Streaming SVL word-element count (SVL/32) via SMSTART SM bracket. */
NK_INTERNAL nk_size_t nk_sme_cntw_(void) {
    nk_u64_t r;
    __asm__ __volatile__("smstart sm\n\t" "cntw %0\n\t" "smstop sm" : "=r"(r));
    return (nk_size_t)r;
}
/** @brief Streaming SVL double-element count (SVL/64) via SMSTART SM bracket. */
NK_INTERNAL nk_size_t nk_sme_cntd_(void) {
    nk_u64_t r;
    __asm__ __volatile__("smstart sm\n\t" "cntd %0\n\t" "smstop sm" : "=r"(r));
    return (nk_size_t)r;
}
#endif

#ifdef __cplusplus
} // extern "C"
#endif

#endif // NK_TYPES_H
