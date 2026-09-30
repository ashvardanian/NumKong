/**
 *  @file include/numkong/types.h
 *  @author Ash Vardanian
 *  @date October 2, 2023
 *  @brief Shared definitions for the NumKong library.
 *
 *  Defines:
 *
 *  - Sized aliases for numeric types, like: @c nk_i32_t and @c nk_f64_t.
 *  - Macros for internal compiler/hardware checks, like: @c NUMKONG_ARCH_ARM64_.
 *  - Macros for feature controls, like: @c NUMKONG_TARGET_NEON
 *
 *  @section types_fp8 FP8 Numeric Types
 *
 *  There are several 8-bit floating-point variants from different industry members, with different
 *  hardware support. None are part of the IEEE 754 standard, but some are part of the Open Compute
 *  Project's OCP 8-bit Floating Point Specification, OFP8:
 *
 *  @verbatim
 *  Format    Bias  Sign  Exp  Mant  Range   Infinity        NaN               Standard
 *  E4M3FN    7     1     4    3     ±448    No              Only 0x7F/0xFF    OCP, NVIDIA, ONNX
 *  E5M2      15    1     5    2     ±57344  Yes (0x7C/0xFC) 0x7D-7F, 0xFD-FF  OCP, IEEE-like
 *  E4M3FNUZ  8     1     4    3     ±240    No              0x80 only         GraphCore, ONNX
 *  E5M2FNUZ  16    1     5    2     ±57344  No              0x80 only         GraphCore, ONNX
 *  @endverbatim
 *
 *  Only two series of currently available and upcoming models prioritize FNUZ over OCP:
 *
 *  - GraphCore IPUs were the original platform proposing FNUZ
 *  - AMD MI300 series based on CDNA3 implements FNUZ, but not OCP
 *  - AMD MI350+ series based on CDNA4 switch to OCP and remove FNUZ
 *  - NVIDIA Hopper and Blackwell only support E4M3FN, E5M2
 *  - Intel AVX10.2 defines HF8 as E4M3FN and BF8 as E5M2, OCP-aligned
 *  - Arm implements E4M3, meaning E4M3FN, and E5M2 sharing one @c __mfp8 type and @c FPMR selector
 *
 *  For brevity, across NumKong, "E4M3" implies "E4M3FN".
 *
 *  @see https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1
 *  @see FP8 Formats for Deep Learning: https://arxiv.org/pdf/2209.05433
 *  @see ONNX Float8 Types: https://onnx.ai/onnx/technical/float8.html
 *
 *  @section types_fp6 FP6 Numeric Types
 *
 *  The OCP Microscaling, MX, v1.0 specification defines two 6-bit floating-point formats for
 *  block-scaled quantization. Both are "FN", short for finite-numeric: all bit patterns map to real
 *  numbers with no Inf or NaN codes, stored byte-aligned with 2 bits of padding.
 *
 *  @verbatim
 *  Format  Bias  Sign  Exp  Mant  Range   Subnormals  Infinity  NaN  Standard
 *  E2M3    1     1     2    3     ±7.5    14 of 64    No        No   OCP MX v1.0
 *  E3M2    3     1     3    2     ±28     6 of 64     No        No   OCP MX v1.0
 *  @endverbatim
 *
 *  E2M3 favors 3-bit mantissa precision for narrow dynamic range, ideal for activations. E3M2
 *  favors 3-bit exponent range for wider dynamic range, suited for weights. Both follow IEEE 754
 *  subnormal rules: when exp=0, the implicit leading bit is 0, giving value = (-1)^s × 0.mmm ×
 *  2^(1-bias), which provides gradual underflow to zero.
 *
 *  No hardware directly computes on FP6. On Arm with FEAT_FP8DOT4, E2M3 values can be losslessly
 *  promoted to E4M3 — same mantissa width, rebias exponent by +6 — and E3M2 to E5M2 — same mantissa
 *  width, rebias exponent by +12 — then fed to FDOT instructions. Subnormal values, exp=0, require
 *  normalization during this promotion.
 *
 *  @see https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
 *  @see The FP6-LLM paper: https://arxiv.org/abs/2401.14112
 */
#ifndef NUMKONG_TYPES_H
#define NUMKONG_TYPES_H

#define NUMKONG_VERSION_MAJOR 7
#define NUMKONG_VERSION_MINOR 8
#define NUMKONG_VERSION_PATCH 2

#include <stdint.h> // Clang modules on glibc would otherwise credit `uint64_t` to ACLE

/** Debug builds check the library's invariants with @c nk_assert_, defaulting from @c DEBUG or
 *  @c _DEBUG. */
#if !defined(NUMKONG_DEBUG)
#if defined(DEBUG) || defined(_DEBUG)
#define NUMKONG_DEBUG 1
#else
#define NUMKONG_DEBUG 0
#endif
#endif

#if NUMKONG_DEBUG && __STDC_HOSTED__ && !defined(__CUDA_ARCH__) && !defined(__METAL_VERSION__)
#include <stdio.h>  // `fprintf`, `stderr`
#include <stdlib.h> // `abort`
#endif

/*  MSan, short for MemorySanitizer, cannot track data flow through SVE horizontal reductions like
 *  @c svaddv, which move data from vector registers to scalar registers via architecture-specific
 *  paths invisible to the compiler. @c nk_unpoison_ marks the resulting scalar as initialized so
 *  MSan does not report false positives. */
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
#include <sanitizer/msan_interface.h>
#define nk_unpoison_(ptr, size) __msan_unpoison((ptr), (size))
#endif
#endif
#ifndef nk_unpoison_
#define nk_unpoison_(ptr, size) (void)(ptr), (void)(size)
#endif

/* Inferring target OS: Windows, macOS, Linux, or FreeBSD */
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define NUMKONG_OS_WINDOWS_ 1
#else
#define NUMKONG_OS_WINDOWS_ 0
#endif
#if defined(__APPLE__) && defined(__MACH__)
#define NUMKONG_OS_APPLE_ 1
#else
#define NUMKONG_OS_APPLE_ 0
#endif
#if defined(__linux__)
#define NUMKONG_OS_LINUX_ 1
#else
#define NUMKONG_OS_LINUX_ 0
#endif
#if defined(__FreeBSD__)
#define NUMKONG_OS_FREEBSD_ 1
#else
#define NUMKONG_OS_FREEBSD_ 0
#endif

/** Base of function annotations:
 *  - @c NUMKONG_API every public function, a kernel of one capability or a dispatch point.
 *  - @c NUMKONG_CONSTEXPR internal helper; the compiler decides inlining.
 *  - @c NUMKONG_INLINE internal helper forced inline, for devirtualizing driver loops. */
#define NUMKONG_C_INLINE_ inline static

#if defined(__GNUC__) || defined(__clang__)
#define NUMKONG_MAYBE_UNUSED_ __attribute__((unused))
#else
#define NUMKONG_MAYBE_UNUSED_
#endif

#if defined(_MSC_VER)
#define NUMKONG_INLINE __forceinline static
#else
#define NUMKONG_INLINE NUMKONG_MAYBE_UNUSED_ __attribute__((always_inline)) NUMKONG_C_INLINE_
#endif

/** The C++ standard in effect, or zero in C. MSVC reports it in @c _MSVC_LANG and leaves
 *  @c __cplusplus at C++98. */
#if defined(_MSVC_LANG)
#define NUMKONG_CXX_STANDARD_ _MSVC_LANG
#elif defined(__cplusplus)
#define NUMKONG_CXX_STANDARD_ __cplusplus
#else
#define NUMKONG_CXX_STANDARD_ 0
#endif

/** Internal helper that callers can fold at compile time, and that CUDA kernels can call through
 *  @c --expt-relaxed-constexpr. It is @c constexpr from C++20, except under MSVC's own front end,
 *  which rejects its intrinsics there. */
#if NUMKONG_CXX_STANDARD_ >= 202002L && (!defined(_MSC_VER) || defined(__clang__) || defined(__CUDACC__))
#define NUMKONG_CONSTEXPR NUMKONG_MAYBE_UNUSED_ NUMKONG_C_INLINE_ constexpr
#else
#define NUMKONG_CONSTEXPR NUMKONG_MAYBE_UNUSED_ NUMKONG_C_INLINE_
#endif

/** Compiles every kernel into this translation unit and stubs every dispatch point and finder with
 *  @c nk_missing_library_k, instead of linking them from the NumKong library. */
#if !defined(NUMKONG_HEADER_ONLY)
#define NUMKONG_HEADER_ONLY (0) // true or false
#endif

/** Every public function: header-inline in header-only builds, otherwise defined once in the
 *  library. Windows DLLs export every symbol through CMake, so no import or export spelling. */
#if NUMKONG_HEADER_ONLY
#define NUMKONG_API NUMKONG_MAYBE_UNUSED_ NUMKONG_C_INLINE_
#elif defined(__GNUC__) || defined(__clang__)
#define NUMKONG_API __attribute__((visibility("default")))
#else
#define NUMKONG_API
#endif

/** Vector union types use type punning by design — write as f16, read as f32, and so on. Without
 *  this, GCC at -O2 assumes strict aliasing and may optimize away valid accesses. */
#if defined(__GNUC__) || defined(__clang__)
#define NUMKONG_MAY_ALIAS_ __attribute__((may_alias))
#else
#define NUMKONG_MAY_ALIAS_
#endif

#if defined(__has_builtin)
#define nk_has_builtin_(x) __has_builtin(x)
#else
#define nk_has_builtin_(x) 0
#endif

/** True when the compiler reserves @c x as a type keyword rather than treating it as a plain
 *  identifier. Clang-only; the wrapper is what lets `#if` mention it on GCC, where the preprocessor
 *  would still have to parse `__is_identifier(...)` even behind a `defined()` guard. */
#if defined(__is_identifier)
#define nk_is_keyword_(x) (!__is_identifier(x))
#else
#define nk_is_keyword_(x) 0
#endif

/** Allow SIMD kernels to redirect small inputs to serial implementations. Enabled by default for
 *  production use. Tests and benchmarks may disable this to isolate SIMD path behavior. */
#if !defined(NUMKONG_ALLOW_ISA_REDIRECT)
#define NUMKONG_ALLOW_ISA_REDIRECT 1
#endif

/*  Compiling for 64-bit Arm: NUMKONG_ARCH_ARM64_
 *  @see Arm C Language Extensions: https://arm-software.github.io/acle/main/acle.html */
#if !defined(NUMKONG_ARCH_ARM64_)
#if defined(__aarch64__) || defined(_M_ARM64)
#define NUMKONG_ARCH_ARM64_ 1
#else
#define NUMKONG_ARCH_ARM64_ 0
#endif // defined(__aarch64__) || defined(_M_ARM64)
#endif // !defined(NUMKONG_ARCH_ARM64_)

/*  Compiling for x86: NUMKONG_ARCH_X8664_
 *  @see Intel predefined macros: https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2024-2/additional-predefined-macros.html */
#if !defined(NUMKONG_ARCH_X8664_)
#if defined(__x86_64__) || defined(_M_X64)
#define NUMKONG_ARCH_X8664_ 1
#else
#define NUMKONG_ARCH_X8664_ 0
#endif // defined(__x86_64__) || defined(_M_X64)
#endif // !defined(NUMKONG_ARCH_X8664_)

/* Compiling for RISC-V: NUMKONG_ARCH_RISCV64_ */
#if !defined(NUMKONG_ARCH_RISCV64_)
#if defined(__riscv) && (__riscv_xlen == 64)
#define NUMKONG_ARCH_RISCV64_ 1
#else
#define NUMKONG_ARCH_RISCV64_ 0
#endif // defined(__riscv) && (__riscv_xlen == 64)
#endif // !defined(NUMKONG_ARCH_RISCV64_)

/* Compiling for LoongArch: NUMKONG_ARCH_LOONGARCH64_ */
#if !defined(NUMKONG_ARCH_LOONGARCH64_)
#if defined(__loongarch__)
#define NUMKONG_ARCH_LOONGARCH64_ 1
#else
#define NUMKONG_ARCH_LOONGARCH64_ 0
#endif // defined(__loongarch__)
#endif // !defined(NUMKONG_ARCH_LOONGARCH64_)

/* Compiling for Power: NUMKONG_ARCH_PPC64_ */
#if !defined(NUMKONG_ARCH_PPC64_)
#if defined(__powerpc64__) || defined(__ppc64__) || defined(_ARCH_PPC64)
#define NUMKONG_ARCH_PPC64_ 1
#else
#define NUMKONG_ARCH_PPC64_ 0
#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(_ARCH_PPC64)
#endif // !defined(NUMKONG_ARCH_PPC64_)

/* Compiling for WASM: NUMKONG_ARCH_WASM_ */
#if !defined(NUMKONG_ARCH_WASM_)
#if defined(__wasm__) || defined(__EMSCRIPTEN__)
#define NUMKONG_ARCH_WASM_ 1
#else
#define NUMKONG_ARCH_WASM_ 0
#endif
#endif // !defined(NUMKONG_ARCH_WASM_)

/*  Compiling as CUDA, by NVCC or Clang, HIP on NVIDIA included: NUMKONG_ARCH_CUDA_. The library
 *  also sets it for its host-only units, which link the kernels of its one CUDA unit. */
#if !defined(NUMKONG_ARCH_CUDA_)
#if defined(__CUDACC__) && !defined(__HIP__)
#define NUMKONG_ARCH_CUDA_ 1
#else
#define NUMKONG_ARCH_CUDA_ 0
#endif // defined(__CUDACC__) && !defined(__HIP__)
#endif // !defined(NUMKONG_ARCH_CUDA_)

/*  Compiling as HIP for AMD GPUs, keyed on Clang's own @c __HIP__: NUMKONG_ARCH_ROCM_. The library
 *  also sets it for its host-only units, which link the kernels of its one HIP unit. */
#if !defined(NUMKONG_ARCH_ROCM_)
#if defined(__HIP__)
#define NUMKONG_ARCH_ROCM_ 1
#else
#define NUMKONG_ARCH_ROCM_ 0
#endif // defined(__HIP__)
#endif // !defined(NUMKONG_ARCH_ROCM_)

/** Linking the Metal host API, set by the build alone: NUMKONG_WITH_METAL */
#if !defined(NUMKONG_WITH_METAL)
#define NUMKONG_WITH_METAL 0
#endif // !defined(NUMKONG_WITH_METAL)

/** Importing the capability probes from the host, NUMKONG_WITH_HOST_PROBES: set by the build for a
 *  WASI module whose host supplies @c nk_has_v128 and @c nk_has_relaxed. The Wasmer and Wasmtime
 *  CLIs cannot, so it defaults to 0 and a module reports the capabilities it was compiled with. */
#if !defined(NUMKONG_WITH_HOST_PROBES)
#define NUMKONG_WITH_HOST_PROBES 0
#endif // !defined(NUMKONG_WITH_HOST_PROBES)

/** Defining the serial kernels: NUMKONG_TARGET_SERIAL. Header-only builds define them in every
 *  translation unit; in the library only its serial unit does, so units that include a serial
 *  header for its helpers, like the bindings', leave its kernels to the library. */
#if !defined(NUMKONG_TARGET_SERIAL)
#define NUMKONG_TARGET_SERIAL NUMKONG_HEADER_ONLY
#endif // !defined(NUMKONG_TARGET_SERIAL)

/*  Compiling for WASM with SIMD128, NUMKONG_TARGET_V128:
 *  A module carries one SIMD capability, chosen by the toolchain's `-msimd128` flag alone. */
#if !defined(NUMKONG_TARGET_V128) || (NUMKONG_TARGET_V128 && !(NUMKONG_ARCH_WASM_ && defined(__wasm_simd128__)))
#undef NUMKONG_TARGET_V128
#if NUMKONG_ARCH_WASM_ && defined(__wasm_simd128__)
#define NUMKONG_TARGET_V128 1
#else
#define NUMKONG_TARGET_V128 0
#endif
#endif // !defined(NUMKONG_TARGET_V128) || ...

/*  Compiling for WASM with Relaxed SIMD, NUMKONG_TARGET_V128RELAXED:
 *  Requires -mrelaxed-simd for FMA instructions: f32x4.relaxed_madd, f64x2.relaxed_madd */
#if !defined(NUMKONG_TARGET_V128RELAXED) || \
    (NUMKONG_TARGET_V128RELAXED && !(NUMKONG_ARCH_WASM_ && defined(__wasm_relaxed_simd__)))
#undef NUMKONG_TARGET_V128RELAXED
#if NUMKONG_ARCH_WASM_ && defined(__wasm_relaxed_simd__)
#define NUMKONG_TARGET_V128RELAXED 1
#else
#define NUMKONG_TARGET_V128RELAXED 0
#endif
#endif // !defined(NUMKONG_TARGET_V128RELAXED) || ...

/* Compiling for RISC-V Vector: NUMKONG_TARGET_RVV */
#if !defined(NUMKONG_TARGET_RVV) || (NUMKONG_TARGET_RVV && !NUMKONG_ARCH_RISCV64_)
#if defined(__riscv_v) && (__riscv_v >= 1000000)
#define NUMKONG_TARGET_RVV 1
#else
#undef NUMKONG_TARGET_RVV
#define NUMKONG_TARGET_RVV 0
#endif // defined(__riscv_v) && (__riscv_v >= 1000000)
#endif // !defined(NUMKONG_TARGET_RVV) || ...

/*  Compiling for RISC-V Vector with Zvfh, f16, NUMKONG_TARGET_RVVHALF:
 *  Requires GCC 14+ or Clang 18+ for full intrinsic support */
#if !defined(NUMKONG_TARGET_RVVHALF) || (NUMKONG_TARGET_RVVHALF && !NUMKONG_TARGET_RVV)
#if defined(__riscv_zvfh) && (__riscv_zvfh > 0)
#define NUMKONG_TARGET_RVVHALF 1
#else
#undef NUMKONG_TARGET_RVVHALF
#define NUMKONG_TARGET_RVVHALF 0
#endif // defined(__riscv_zvfh) && (__riscv_zvfh > 0)
#endif // !defined(NUMKONG_TARGET_RVVHALF) || ...

/*  Compiling for RISC-V Vector with Zvfbfwma, bf16 widening FMA, NUMKONG_TARGET_RVVBF16:
 *  Requires GCC 14+ or Clang 18+ for full intrinsic support */
#if !defined(NUMKONG_TARGET_RVVBF16) || (NUMKONG_TARGET_RVVBF16 && !NUMKONG_TARGET_RVV)
#if defined(__riscv_zvfbfwma) && (__riscv_zvfbfwma > 0)
#define NUMKONG_TARGET_RVVBF16 1
#else
#undef NUMKONG_TARGET_RVVBF16
#define NUMKONG_TARGET_RVVBF16 0
#endif // defined(__riscv_zvfbfwma) && (__riscv_zvfbfwma > 0)
#endif // !defined(NUMKONG_TARGET_RVVBF16) || ...

/*  Compiling for RISC-V Vector with Zvbb, basic bit-manipulation, NUMKONG_TARGET_RVVBB:
 *  Provides per-element popcount via vcpop.v, plus vclz.v, vctz.v, vbrev.v, vrol.v, vror.v */
#if !defined(NUMKONG_TARGET_RVVBB) || (NUMKONG_TARGET_RVVBB && !NUMKONG_TARGET_RVV)
#if defined(__riscv_zvbb) && (__riscv_zvbb > 0)
#define NUMKONG_TARGET_RVVBB 1
#else
#undef NUMKONG_TARGET_RVVBB
#define NUMKONG_TARGET_RVVBB 0
#endif // defined(__riscv_zvbb) && (__riscv_zvbb > 0)
#endif // !defined(NUMKONG_TARGET_RVVBB) || ...

/*  Compiling for LoongArch LASX, 256-bit SIMD, NUMKONG_TARGET_LOONGSONASX:
 *  LASX provides 32 × 256-bit vector registers, widening integer multiply-accumulate, and
 *  f32-to-f64 conversion via xvfcvtl_d_s / xvfcvth_d_s, but no widening FMA. Its units compile with
 *  `-mlasx`, as `lasxintrin.h` hides its contents without the flag. */
#if !defined(NUMKONG_TARGET_LOONGSONASX) || (NUMKONG_TARGET_LOONGSONASX && !NUMKONG_ARCH_LOONGARCH64_)
#if defined(__loongarch_asx)
#define NUMKONG_TARGET_LOONGSONASX 1
#else
#undef NUMKONG_TARGET_LOONGSONASX
#define NUMKONG_TARGET_LOONGSONASX 0
#endif // defined(__loongarch_asx)
#endif // !defined(NUMKONG_TARGET_LOONGSONASX) || ...

/*  Compiling for Power VSX, 128-bit SIMD, POWER9+ baseline, NUMKONG_TARGET_POWERVSX:
 *  VSX provides 64 × 128-bit registers, FMA via vec_madd, vec_msum for multiply-sum, hardware f16
 *  conversion via vec_extract_fp32_from_shorth/l, length-limited loads via vec_xl_len, per-byte
 *  popcount via vec_popcnt, and vec_cmpne. Requires POWER9, ISA 3.0, or newer. Its units compile
 *  with `-mcpu=power9`, as `altivec.h` hides the POWER9 API without the flag. */
#if !defined(NUMKONG_TARGET_POWERVSX) || (NUMKONG_TARGET_POWERVSX && !NUMKONG_ARCH_PPC64_)
#if defined(__VSX__) && defined(__POWER9_VECTOR__)
#define NUMKONG_TARGET_POWERVSX 1
#else
#undef NUMKONG_TARGET_POWERVSX
#define NUMKONG_TARGET_POWERVSX 0
#endif // defined(__VSX__)
#endif // !defined(NUMKONG_TARGET_POWERVSX) || ...

/* Compiling for Arm: NUMKONG_TARGET_NEON, AArch64 only — AArch32 NEON is not supported. */
#if !defined(NUMKONG_TARGET_NEON) || (NUMKONG_TARGET_NEON && !NUMKONG_ARCH_ARM64_)
#if (defined(__ARM_NEON) && defined(__aarch64__)) || (defined(_MSC_VER) && defined(_M_ARM64))
#define NUMKONG_TARGET_NEON 1
#else
#undef NUMKONG_TARGET_NEON
#define NUMKONG_TARGET_NEON 0
#endif // (defined(__ARM_NEON) && defined(__aarch64__)) || ...
#endif // !defined(NUMKONG_TARGET_NEON) || ...

/* Compiling for Arm: NUMKONG_TARGET_NEONSDOT, FEAT_DotProd, AArch64 only. */
#if !defined(NUMKONG_TARGET_NEONSDOT) || (NUMKONG_TARGET_NEONSDOT && !NUMKONG_ARCH_ARM64_)
#if (defined(__ARM_FEATURE_DOTPROD) && defined(__aarch64__)) || \
    (defined(_MSC_VER) && defined(_M_ARM64) && __ARM_ARCH >= 804)
#define NUMKONG_TARGET_NEONSDOT 1
#else
#undef NUMKONG_TARGET_NEONSDOT
#define NUMKONG_TARGET_NEONSDOT 0
#endif
#endif // !defined(NUMKONG_TARGET_NEONSDOT) || ...

/* Compiling for Arm: NUMKONG_TARGET_NEONHALF, FEAT_FP16, AArch64 only. */
#if !defined(NUMKONG_TARGET_NEONHALF) || (NUMKONG_TARGET_NEONHALF && !NUMKONG_ARCH_ARM64_)
#if (defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(__aarch64__)) || \
    (defined(_MSC_VER) && defined(_M_ARM64) && __ARM_ARCH >= 802)
#define NUMKONG_TARGET_NEONHALF 1
#else
#undef NUMKONG_TARGET_NEONHALF
#define NUMKONG_TARGET_NEONHALF 0
#endif
#endif // !defined(NUMKONG_TARGET_NEONHALF) || ...

/* Compiling for Arm: NUMKONG_TARGET_NEONFHM, FEAT_FHM, AArch64 only. */
#if !defined(NUMKONG_TARGET_NEONFHM) || (NUMKONG_TARGET_NEONFHM && !NUMKONG_ARCH_ARM64_)
#if (defined(__ARM_FEATURE_FP16_FML) && defined(__aarch64__)) || \
    (defined(_MSC_VER) && defined(_M_ARM64) && __ARM_ARCH >= 804)
#define NUMKONG_TARGET_NEONFHM 1
#else
#undef NUMKONG_TARGET_NEONFHM
#define NUMKONG_TARGET_NEONFHM 0
#endif
#endif // !defined(NUMKONG_TARGET_NEONFHM) || ...

/* Compiling for Arm: NUMKONG_TARGET_NEONBFDOT, FEAT_BF16, AArch64 only. */
#if !defined(NUMKONG_TARGET_NEONBFDOT) || (NUMKONG_TARGET_NEONBFDOT && !NUMKONG_ARCH_ARM64_)
#if (defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) && defined(__aarch64__)) || \
    (defined(_MSC_VER) && defined(_M_ARM64) && __ARM_ARCH >= 806)
#define NUMKONG_TARGET_NEONBFDOT 1
#else
#undef NUMKONG_TARGET_NEONBFDOT
#define NUMKONG_TARGET_NEONBFDOT 0
#endif
#endif // !defined(NUMKONG_TARGET_NEONBFDOT) || ...

/*  Compiling for Arm: NUMKONG_TARGET_NEONFP8, NEON FP8 extensions, FEAT_FP8DOT4. ACLE macro
 *  __ARM_FEATURE_FP8DOT4 defined by GCC 15+ and Clang 21+ when +fp8dot4 is enabled. Older compilers
 *  lack mfloat8x16_t and the fp8dot4 target attribute entirely. */
#if !defined(NUMKONG_TARGET_NEONFP8) || (NUMKONG_TARGET_NEONFP8 && !NUMKONG_ARCH_ARM64_)
#if defined(__ARM_FEATURE_FP8DOT4) && defined(__aarch64__)
#define NUMKONG_TARGET_NEONFP8 1
#else
#undef NUMKONG_TARGET_NEONFP8
#define NUMKONG_TARGET_NEONFP8 0
#endif // defined(__ARM_FEATURE_FP8DOT4)
#endif // !defined(NUMKONG_TARGET_NEONFP8)  || ...

/* Compiling for Arm: NUMKONG_TARGET_SVE */
#if !defined(NUMKONG_TARGET_SVE) || (NUMKONG_TARGET_SVE && !NUMKONG_ARCH_ARM64_)
#if defined(__ARM_FEATURE_SVE)
#define NUMKONG_TARGET_SVE 1
#else
#undef NUMKONG_TARGET_SVE
#define NUMKONG_TARGET_SVE 0
#endif // defined(__ARM_FEATURE_SVE)
#endif // !defined(NUMKONG_TARGET_SVE) || ...

/* Compiling for Arm: NUMKONG_TARGET_SVESDOT */
#if !defined(NUMKONG_TARGET_SVESDOT) || (NUMKONG_TARGET_SVESDOT && !NUMKONG_ARCH_ARM64_)
#if defined(__ARM_FEATURE_SVE)
#define NUMKONG_TARGET_SVESDOT 1
#else
#undef NUMKONG_TARGET_SVESDOT
#define NUMKONG_TARGET_SVESDOT 0
#endif // defined(__ARM_FEATURE_SVE)
#endif // !defined(NUMKONG_TARGET_SVESDOT) || ...

/* Compiling for Arm: NUMKONG_TARGET_SVEHALF */
#if !defined(NUMKONG_TARGET_SVEHALF) || (NUMKONG_TARGET_SVEHALF && !NUMKONG_ARCH_ARM64_)
#if defined(__ARM_FEATURE_SVE)
#define NUMKONG_TARGET_SVEHALF 1
#else
#undef NUMKONG_TARGET_SVEHALF
#define NUMKONG_TARGET_SVEHALF 0
#endif // defined(__ARM_FEATURE_SVE)
#endif // !defined(NUMKONG_TARGET_SVEHALF) || ...

/* Compiling for Arm: NUMKONG_TARGET_SVEBFDOT */
#if !defined(NUMKONG_TARGET_SVEBFDOT) || (NUMKONG_TARGET_SVEBFDOT && !NUMKONG_ARCH_ARM64_)
#if defined(__ARM_FEATURE_SVE)
#define NUMKONG_TARGET_SVEBFDOT 1
#else
#undef NUMKONG_TARGET_SVEBFDOT
#define NUMKONG_TARGET_SVEBFDOT 0
#endif // defined(__ARM_FEATURE_SVE)
#endif // !defined(NUMKONG_TARGET_SVEBFDOT) || ...

/* Compiling for Arm: NUMKONG_TARGET_SVE2 */
#if !defined(NUMKONG_TARGET_SVE2) || (NUMKONG_TARGET_SVE2 && !NUMKONG_ARCH_ARM64_)
#if defined(__ARM_FEATURE_SVE2)
#define NUMKONG_TARGET_SVE2 1
#else
#undef NUMKONG_TARGET_SVE2
#define NUMKONG_TARGET_SVE2 0
#endif // defined(__ARM_FEATURE_SVE2)
#endif // !defined(NUMKONG_TARGET_SVE2) || ...

/* Compiling for Arm: NUMKONG_TARGET_SME, the Scalable Matrix Extension. */
#if !defined(NUMKONG_TARGET_SME) || (NUMKONG_TARGET_SME && !NUMKONG_ARCH_ARM64_)
#if defined(__ARM_FEATURE_SME)
#define NUMKONG_TARGET_SME 1
#else
#undef NUMKONG_TARGET_SME
#define NUMKONG_TARGET_SME 0
#endif // defined(__ARM_FEATURE_SME)
#endif // !defined(NUMKONG_TARGET_SME) || ...

/*  AppleClang 17 exposes SME sub-features through `arm_sme.h` builtin aliases, not dedicated
 *  `__ARM_FEATURE_*` predefines for every matrix subtype. */
#if !defined(NUMKONG_TARGET_SMEF64) || (NUMKONG_TARGET_SMEF64 && !NUMKONG_ARCH_ARM64_)
#if defined(__ARM_FEATURE_SME_F64F64) || nk_has_builtin_(__builtin_sme_svmopa_za64_f64_m)
#define NUMKONG_TARGET_SMEF64 1
#else
#undef NUMKONG_TARGET_SMEF64
#define NUMKONG_TARGET_SMEF64 0
#endif // defined(__ARM_FEATURE_SME_F64F64) || ...
#endif // !defined(NUMKONG_TARGET_SMEF64) || ...

#if !defined(NUMKONG_TARGET_SMEBI32) || (NUMKONG_TARGET_SMEBI32 && !NUMKONG_ARCH_ARM64_)
#if nk_has_builtin_(__builtin_sme_svbmopa_za32_u32_m)
#define NUMKONG_TARGET_SMEBI32 1
#else
#undef NUMKONG_TARGET_SMEBI32
#define NUMKONG_TARGET_SMEBI32 0
#endif // nk_has_builtin_(__builtin_sme_svbmopa_za32_u32_m)
#endif // !defined(NUMKONG_TARGET_SMEBI32) || ...

/*  Compiling for x86: NUMKONG_TARGET_HASWELL
 *
 *  Starting with Ivy Bridge, Intel supports the @c F16C extensions for fast half-precision to
 *  single-precision floating-point conversions. On AMD those instructions are supported on all CPUs
 *  starting with Jaguar 2009.
 *
 *  Starting with Sandy Bridge, Intel adds basic AVX support in their CPUs and in 2013 extends it
 *  with AVX2 in the Haswell generation. Moreover, Haswell adds FMA support.
 *
 *  On MSVC, most GCC-style ISA macros are unavailable. MSVC defines __AVX__, __AVX2__,
 *  __AVX512F/BW/CD/DQ/VL__, and __AVX10_VER__, but not __AVXVNNI__, __AVX512VNNI__, __AVX512BF16__,
 *  __AVX512FP16__, __AMX_*__, etc. Instead, MSVC makes all intrinsics available once the toolset
 *  version supports them, without requiring `/arch:AVX512`. We gate on _MSC_VER to auto-enable
 *  targets:
 *  - _MSC_VER >= 1900, VS 2015+: AVX2/FMA/F16C, matching Haswell
 *  - _MSC_VER >= 1920, VS 2019+: AVX-512 base plus AVX-VNNI, matching Skylake, Icelake, and Alder
 *  - _MSC_VER >= 1944, VS 2022 17.14+: BF16, FP16, VP2INTERSECT, VNNI-INT8, AMX, matching Sierra */
#if !defined(NUMKONG_TARGET_HASWELL) || (NUMKONG_TARGET_HASWELL && !NUMKONG_ARCH_X8664_)
#if (defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)) || (defined(_MSC_VER) && _MSC_VER >= 1900)
#define NUMKONG_TARGET_HASWELL 1
#else
#undef NUMKONG_TARGET_HASWELL
#define NUMKONG_TARGET_HASWELL 0
#endif // defined(__AVX2__)
#endif // !defined(NUMKONG_TARGET_HASWELL) || ...

/*
 *  Compiling for x86: NUMKONG_TARGET_SKYLAKE, NUMKONG_TARGET_ICELAKE, NUMKONG_TARGET_GENOA,
 *  NUMKONG_TARGET_SAPPHIRE, NUMKONG_TARGET_TURIN, NUMKONG_TARGET_SIERRA
 *
 *  To list all available macros for x86, take a recent compiler, like GCC 12 and run:
 *
 *  @code{.sh}
 *  gcc-12 -march=sapphirerapids -dM -E - < /dev/null | egrep "SSE|AVX" | sort
 *  @endcode
 *
 *  On Arm machines you may want to check for other flags:
 *
 *  @code{.sh}
 *  gcc-12 -march=native -dM -E - < /dev/null | egrep "NEON|SVE|FP16|FMA" | sort
 *  @endcode
 */
#if !defined(NUMKONG_TARGET_SKYLAKE) || (NUMKONG_TARGET_SKYLAKE && !NUMKONG_ARCH_X8664_)
#if (defined(__AVX512F__) && defined(__AVX512CD__) && defined(__AVX512VL__) && defined(__AVX512DQ__) && \
     defined(__AVX512BW__)) ||                                                                          \
    (defined(_MSC_VER) && _MSC_VER >= 1920)
#define NUMKONG_TARGET_SKYLAKE 1
#else
#undef NUMKONG_TARGET_SKYLAKE
#define NUMKONG_TARGET_SKYLAKE 0
#endif
#endif // !defined(NUMKONG_TARGET_SKYLAKE) || ...

#if !defined(NUMKONG_TARGET_ICELAKE) || (NUMKONG_TARGET_ICELAKE && !NUMKONG_ARCH_X8664_)
#if (defined(__AVX512VNNI__) && defined(__AVX512IFMA__) && defined(__AVX512BITALG__) && defined(__AVX512VBMI__) && \
     defined(__AVX512VBMI2__) && defined(__AVX512VPOPCNTDQ__)) ||                                                  \
    (defined(_MSC_VER) && _MSC_VER >= 1920)
#define NUMKONG_TARGET_ICELAKE 1
#else
#undef NUMKONG_TARGET_ICELAKE
#define NUMKONG_TARGET_ICELAKE 0
#endif
#endif // !defined(NUMKONG_TARGET_ICELAKE) || ...

#if !defined(NUMKONG_TARGET_GENOA) || (NUMKONG_TARGET_GENOA && !NUMKONG_ARCH_X8664_)
#if defined(__AVX512BF16__) || (defined(_MSC_VER) && _MSC_VER >= 1944)
#define NUMKONG_TARGET_GENOA 1
#else
#undef NUMKONG_TARGET_GENOA
#define NUMKONG_TARGET_GENOA 0
#endif // defined(__AVX512BF16__) || ...
#endif // !defined(NUMKONG_TARGET_GENOA) || ...

/*  Compiling for x86: NUMKONG_TARGET_DIAMOND, AVX10.2, Diamond Rapids
 *
 *  - GCC 15+ defines __AVX10_2__ with -mavx10.2, target attribute `avx10.2`.
 *  - Clang 20+ defines __AVX10_2__ with -mavx10.2. The target attribute spells it `avx10.2-512` up
 *    to Clang 21, and `avx10.2` from Clang 22 and GCC 15.
 *  - MSVC defines __AVX10_VER__ >= 2 with /arch:AVX10.2, VS 2026+, not yet released. */
#if !defined(NUMKONG_TARGET_DIAMOND) || (NUMKONG_TARGET_DIAMOND && !NUMKONG_ARCH_X8664_)
#if defined(__AVX10_2__) || (defined(__AVX10_VER__) && __AVX10_VER__ >= 2)
#define NUMKONG_TARGET_DIAMOND 1
#else
#undef NUMKONG_TARGET_DIAMOND
#define NUMKONG_TARGET_DIAMOND 0
#endif // defined(__AVX10_2__) || ...
#endif // !defined(NUMKONG_TARGET_DIAMOND) || ...

#if !defined(NUMKONG_TARGET_SAPPHIRE) || (NUMKONG_TARGET_SAPPHIRE && !NUMKONG_ARCH_X8664_)
#if defined(__AVX512FP16__) || (defined(_MSC_VER) && _MSC_VER >= 1944)
#define NUMKONG_TARGET_SAPPHIRE 1
#else
#undef NUMKONG_TARGET_SAPPHIRE
#define NUMKONG_TARGET_SAPPHIRE 0
#endif
#endif // !defined(NUMKONG_TARGET_SAPPHIRE) || ...

#if !defined(NUMKONG_TARGET_SAPPHIREAMX) || (NUMKONG_TARGET_SAPPHIREAMX && !NUMKONG_ARCH_X8664_)
#if (defined(__AMX_TILE__) && defined(__AMX_BF16__) && defined(__AMX_INT8__)) || (defined(_MSC_VER) && _MSC_VER >= 1944)
#define NUMKONG_TARGET_SAPPHIREAMX 1
#else
#undef NUMKONG_TARGET_SAPPHIREAMX
#define NUMKONG_TARGET_SAPPHIREAMX 0
#endif
#endif // !defined(NUMKONG_TARGET_SAPPHIREAMX) || ...

#if !defined(NUMKONG_TARGET_GRANITEAMX) || (NUMKONG_TARGET_GRANITEAMX && !NUMKONG_ARCH_X8664_)
#if (defined(__AMX_TILE__) && defined(__AMX_FP16__)) || (defined(_MSC_VER) && _MSC_VER >= 1944)
#define NUMKONG_TARGET_GRANITEAMX 1
#else
#undef NUMKONG_TARGET_GRANITEAMX
#define NUMKONG_TARGET_GRANITEAMX 0
#endif
#endif // !defined(NUMKONG_TARGET_GRANITEAMX) || ...

#if !defined(NUMKONG_TARGET_DIAMONDAMX) || (NUMKONG_TARGET_DIAMONDAMX && !NUMKONG_ARCH_X8664_)
#if defined(__AMX_TILE__) && defined(__AMX_BF16__) && defined(__AMX_INT8__) && defined(__AMX_FP8__) && \
    defined(__AMX_AVX512__)
#define NUMKONG_TARGET_DIAMONDAMX 1
#else
#undef NUMKONG_TARGET_DIAMONDAMX
#define NUMKONG_TARGET_DIAMONDAMX 0
#endif
#endif // !defined(NUMKONG_TARGET_DIAMONDAMX) || ...

#if !defined(NUMKONG_TARGET_TURIN) || (NUMKONG_TARGET_TURIN && !NUMKONG_ARCH_X8664_)
#if defined(__AVX512VP2INTERSECT__) || (defined(_MSC_VER) && _MSC_VER >= 1944)
#define NUMKONG_TARGET_TURIN 1
#else
#undef NUMKONG_TARGET_TURIN
#define NUMKONG_TARGET_TURIN 0
#endif
#endif // !defined(NUMKONG_TARGET_TURIN) || ...

#if !defined(NUMKONG_TARGET_ALDER) || (NUMKONG_TARGET_ALDER && !NUMKONG_ARCH_X8664_)
#if defined(__AVXVNNI__) || (defined(_MSC_VER) && _MSC_VER >= 1920)
#define NUMKONG_TARGET_ALDER 1
#else
#undef NUMKONG_TARGET_ALDER
#define NUMKONG_TARGET_ALDER 0
#endif
#endif // !defined(NUMKONG_TARGET_ALDER) || ...

#if !defined(NUMKONG_TARGET_SIERRA) || (NUMKONG_TARGET_SIERRA && !NUMKONG_ARCH_X8664_)
#if defined(__AVXVNNIINT8__) || (defined(_MSC_VER) && _MSC_VER >= 1944)
#define NUMKONG_TARGET_SIERRA 1
#else
#undef NUMKONG_TARGET_SIERRA
#define NUMKONG_TARGET_SIERRA 0
#endif
#endif // !defined(NUMKONG_TARGET_SIERRA) || ...

/*  Compiling the NVIDIA baseline, @c cuda, on the SIMT cores of every CUDA device:
 *  NUMKONG_TARGET_CUDA. The library's later-generation units turn it off, so one unit emits it. */
#if !defined(NUMKONG_TARGET_CUDA) || (NUMKONG_TARGET_CUDA && !NUMKONG_ARCH_CUDA_)
#undef NUMKONG_TARGET_CUDA
#define NUMKONG_TARGET_CUDA NUMKONG_ARCH_CUDA_
#endif // !defined(NUMKONG_TARGET_CUDA) || ...

/*  Compiling the AMD baseline, @c rocm, on the SIMT cores of every ROCm device:
 *  NUMKONG_TARGET_ROCM. The library's later-generation units turn it off, so one unit emits it. */
#if !defined(NUMKONG_TARGET_ROCM) || (NUMKONG_TARGET_ROCM && !NUMKONG_ARCH_ROCM_)
#undef NUMKONG_TARGET_ROCM
#define NUMKONG_TARGET_ROCM NUMKONG_ARCH_ROCM_
#endif // !defined(NUMKONG_TARGET_ROCM) || ...

/* Compiling for NVIDIA GPUs from compute capability 8.0, `mma.sync`: NUMKONG_TARGET_AMPERE */
#if !defined(NUMKONG_TARGET_AMPERE) || (NUMKONG_TARGET_AMPERE && !NUMKONG_ARCH_CUDA_)
#undef NUMKONG_TARGET_AMPERE
#define NUMKONG_TARGET_AMPERE 0
#endif // !defined(NUMKONG_TARGET_AMPERE) || ...

/* Compiling for NVIDIA GPUs from compute capability 8.9, FP8 `mma.sync`: NUMKONG_TARGET_ADA */
#if !defined(NUMKONG_TARGET_ADA) || (NUMKONG_TARGET_ADA && !NUMKONG_ARCH_CUDA_)
#undef NUMKONG_TARGET_ADA
#define NUMKONG_TARGET_ADA 0
#endif // !defined(NUMKONG_TARGET_ADA) || ...

/*  Compiling for NVIDIA Hopper, "90a" code, @c wgmma: NUMKONG_TARGET_HOPPER. The host pass reads
 *  "90a" and "90" alike, so a build targeting only "90a" sets it. */
#if !defined(NUMKONG_TARGET_HOPPER) || (NUMKONG_TARGET_HOPPER && !NUMKONG_ARCH_CUDA_)
#undef NUMKONG_TARGET_HOPPER
#define NUMKONG_TARGET_HOPPER 0
#endif // !defined(NUMKONG_TARGET_HOPPER) || ...

/*  Compiling for NVIDIA datacenter Blackwell, "100f" code, @c tcgen05: NUMKONG_TARGET_BLACKWELL.
 *  The host pass reads "100f" and "100" alike, so a build whose every architecture is in the 10.x
 *  family sets it. */
#if !defined(NUMKONG_TARGET_BLACKWELL) || (NUMKONG_TARGET_BLACKWELL && !NUMKONG_ARCH_CUDA_)
#undef NUMKONG_TARGET_BLACKWELL
#define NUMKONG_TARGET_BLACKWELL 0
#endif // !defined(NUMKONG_TARGET_BLACKWELL) || ...

/*  Compiling for NVIDIA GPUs of compute capability 12.x, "120f" code, FP6/FP4 MMA:
 *  NUMKONG_TARGET_BLACKWELLRTX. The host pass reads "120f" and "120" alike, so a build targeting
 *  only the 12.x family sets it. */
#if !defined(NUMKONG_TARGET_BLACKWELLRTX) || (NUMKONG_TARGET_BLACKWELLRTX && !NUMKONG_ARCH_CUDA_)
#undef NUMKONG_TARGET_BLACKWELLRTX
#define NUMKONG_TARGET_BLACKWELLRTX 0
#endif // !defined(NUMKONG_TARGET_BLACKWELLRTX) || ...

/*  Compiling for AMD Instinct MI350 GPUs, gfx950: NUMKONG_TARGET_CDNA4. The host pass never names
 *  the AMD GPU it compiles for, so a build targeting only gfx950 sets it. */
#if !defined(NUMKONG_TARGET_CDNA4) || (NUMKONG_TARGET_CDNA4 && !NUMKONG_ARCH_ROCM_)
#undef NUMKONG_TARGET_CDNA4
#define NUMKONG_TARGET_CDNA4 0
#endif // !defined(NUMKONG_TARGET_CDNA4) || ...

/*  Compiling for AMD Instinct MI400 GPUs: NUMKONG_TARGET_CDNA5. The host pass never names the AMD
 *  GPU it compiles for, so a build targeting only MI400 and newer sets it. */
#if !defined(NUMKONG_TARGET_CDNA5) || (NUMKONG_TARGET_CDNA5 && !NUMKONG_ARCH_ROCM_)
#undef NUMKONG_TARGET_CDNA5
#define NUMKONG_TARGET_CDNA5 0
#endif // !defined(NUMKONG_TARGET_CDNA5) || ...

/*  Compiling for Apple GPUs of Metal family 9, M3 and M4: NUMKONG_TARGET_APPLE9. No compiler knows
 *  which Apple GPU a build will meet, so a build targeting only M3 and newer sets it. */
#if !defined(NUMKONG_TARGET_APPLE9) || (NUMKONG_TARGET_APPLE9 && !NUMKONG_WITH_METAL)
#undef NUMKONG_TARGET_APPLE9
#define NUMKONG_TARGET_APPLE9 0
#endif // !defined(NUMKONG_TARGET_APPLE9) || ...

/*  Compiling for Apple GPUs of Metal family 10, M5: NUMKONG_TARGET_APPLE10. No compiler knows which
 *  Apple GPU a build will meet, so a build targeting only M5 and newer sets it. */
#if !defined(NUMKONG_TARGET_APPLE10) || (NUMKONG_TARGET_APPLE10 && !NUMKONG_WITH_METAL)
#undef NUMKONG_TARGET_APPLE10
#define NUMKONG_TARGET_APPLE10 0
#endif // !defined(NUMKONG_TARGET_APPLE10) || ...

/*  Defining the Metal baseline and the Metal runtime helpers: NUMKONG_TARGET_METAL. Header-only
 *  builds define them where Metal is linked, and the library in its Metal unit alone. */
#if !defined(NUMKONG_TARGET_METAL) || (NUMKONG_TARGET_METAL && !NUMKONG_WITH_METAL)
#undef NUMKONG_TARGET_METAL
#define NUMKONG_TARGET_METAL (NUMKONG_WITH_METAL && NUMKONG_HEADER_ONLY)
#endif // !defined(NUMKONG_TARGET_METAL) || ...

/** Whether a capability's helpers compile here: its own target, or any capability built on it. Each
 *  architecture's base, Haswell, NEON, RVV and V128, covers every capability of that architecture.
 *  @c NUMKONG_TARGET_* alone decides where its kernels are defined. */
#define NUMKONG_ARCH_X8664_SAPPHIREAMX_ \
    (NUMKONG_TARGET_SAPPHIREAMX || NUMKONG_TARGET_GRANITEAMX || NUMKONG_TARGET_DIAMONDAMX)
#define NUMKONG_ARCH_X8664_ICELAKE_ \
    (NUMKONG_TARGET_ICELAKE || NUMKONG_TARGET_GENOA || NUMKONG_TARGET_SAPPHIRE || NUMKONG_ARCH_X8664_SAPPHIREAMX_)
#define NUMKONG_ARCH_X8664_SKYLAKE_ \
    (NUMKONG_TARGET_SKYLAKE || NUMKONG_ARCH_X8664_ICELAKE_ || NUMKONG_TARGET_DIAMOND || NUMKONG_TARGET_TURIN)
#define NUMKONG_ARCH_X8664_HASWELL_ \
    (NUMKONG_TARGET_HASWELL || NUMKONG_ARCH_X8664_SKYLAKE_ || NUMKONG_TARGET_ALDER || NUMKONG_TARGET_SIERRA)
#define NUMKONG_ARCH_ARM64_SME_ (NUMKONG_TARGET_SME || NUMKONG_TARGET_SMEF64 || NUMKONG_TARGET_SMEBI32)
#define NUMKONG_ARCH_ARM64_SVE_                                                                           \
    (NUMKONG_TARGET_SVE || NUMKONG_TARGET_SVEHALF || NUMKONG_TARGET_SVESDOT || NUMKONG_TARGET_SVEBFDOT || \
     NUMKONG_TARGET_SVE2 || NUMKONG_ARCH_ARM64_SME_)
#define NUMKONG_ARCH_ARM64_NEON_                                                                             \
    (NUMKONG_TARGET_NEON || NUMKONG_TARGET_NEONHALF || NUMKONG_TARGET_NEONBFDOT || NUMKONG_TARGET_NEONFHM || \
     NUMKONG_TARGET_NEONSDOT || NUMKONG_TARGET_NEONFP8 || NUMKONG_ARCH_ARM64_SVE_)
#define NUMKONG_ARCH_RISCV64_RVV_ \
    (NUMKONG_TARGET_RVV || NUMKONG_TARGET_RVVHALF || NUMKONG_TARGET_RVVBF16 || NUMKONG_TARGET_RVVBB)
#define NUMKONG_ARCH_WASM_V128_ (NUMKONG_TARGET_V128 || NUMKONG_TARGET_V128RELAXED)
#define NUMKONG_ARCH_CUDA_AMPERE_ \
    (NUMKONG_TARGET_AMPERE || NUMKONG_TARGET_HOPPER || NUMKONG_TARGET_BLACKWELL || NUMKONG_TARGET_BLACKWELLRTX)
#define NUMKONG_ARCH_CUDA_ADA_ \
    (NUMKONG_TARGET_ADA || NUMKONG_TARGET_HOPPER || NUMKONG_TARGET_BLACKWELL || NUMKONG_TARGET_BLACKWELLRTX)
#define NUMKONG_ARCH_ROCM_CDNA4_ (NUMKONG_TARGET_CDNA4 || NUMKONG_TARGET_CDNA5)
#define NUMKONG_ARCH_METAL_      (NUMKONG_TARGET_METAL || NUMKONG_TARGET_APPLE9 || NUMKONG_TARGET_APPLE10)

/* Include the relevant intrinsics headers */
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#if NUMKONG_ARCH_ARM64_NEON_
#include <arm_neon.h>
#endif
#if NUMKONG_ARCH_ARM64_SVE_
#include <arm_sve.h>
#endif
#if NUMKONG_ARCH_ARM64_SME_
#include <arm_sme.h>
#endif
#if NUMKONG_ARCH_X8664_HASWELL_
#include <immintrin.h>
#endif
#if NUMKONG_ARCH_RISCV64_RVV_
#include <riscv_vector.h>
#endif
#if defined(__loongarch_asx)
#include <lsxintrin.h>  // `__m128i` for LSX SIMD
#include <lasxintrin.h> // `__m256i` for LASX SIMD
#endif
#if defined(__POWER9_VECTOR__)
#include <altivec.h>
#endif
#if NUMKONG_ARCH_WASM_V128_
#include <wasm_simd128.h>
#endif
/*  Host-only units include neither runtime, as the two clash; vendor units probe the devices. */
#if NUMKONG_ARCH_CUDA_ && defined(__CUDACC__)
#include <cuda_fp16.h>    // `__half2float`
#include <cuda_runtime.h> // `cudaLaunchKernel`, `cudaStream_t`
#endif
#if NUMKONG_ARCH_ROCM_ && defined(__HIP__)
#include <hip/hip_fp16.h>    // `__half2float`
#include <hip/hip_runtime.h> // `hipLaunchKernel`, `hipStream_t`
#endif

/** The highest tensor rank the bindings accept, matching @c PyBUF_MAX_NDIM by default. */
#if !defined(NUMKONG_TENSOR_MAX_RANK)
#define NUMKONG_TENSOR_MAX_RANK (64)
#endif

/** Aligns a variable to a 64-byte boundary using compiler extensions, since `alignas(64)` is only
 *  available in C11 or C++. Used internally and recommended for external users. */
#if defined(_MSC_VER)
#define NUMKONG_ALIGN64_ __declspec(align(64))
#elif defined(__GNUC__) || defined(__clang__)
#define NUMKONG_ALIGN64_ __attribute__((aligned(64)))
#endif

/** ARM streaming attributes, requiring an SME-capable compiler: GCC 14+, Clang 16+.
 *  @c NUMKONG_STREAMING_ marks functions that require streaming SVE mode, such as FCVTLT.
 *  @c NUMKONG_STREAMABLE_ marks helpers callable in and out of streaming mode.
 *  @c NUMKONG_OUTLINED_ replaces @c static on streaming bodies called from non-streaming kernels,
 *  which GCC would inline, then reject their SVE intrinsics or frame by VL and spill at SVL. */
#if NUMKONG_ARCH_ARM64_ && NUMKONG_ARCH_ARM64_SME_
#define NUMKONG_STREAMING_  __arm_streaming
#define NUMKONG_STREAMABLE_ __arm_streaming_compatible
#else
#define NUMKONG_STREAMING_
#define NUMKONG_STREAMABLE_
#endif
#if NUMKONG_ARCH_ARM64_ && NUMKONG_ARCH_ARM64_SME_ && defined(__GNUC__) && !defined(__clang__)
#define NUMKONG_OUTLINED_ __attribute__((noinline)) static
#else
#define NUMKONG_OUTLINED_ static
#endif

/** @c NUMKONG_DEVICE marks helpers that kernels call, forced inline on the device. CUDA and
 *  HIP share the spelling, and kernels spell their own global qualifier, the only one here. */
#if NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_
#define NUMKONG_DEVICE static __device__ __forceinline__
#endif

/** Portable casts between SIMD vector types. MSVC typedefs @c __m512bh, @c __m512h, and @c __m256bh
 *  as aliases for @c __m512i and @c __m256i, but rejects C-style casts between them; GCC and Clang
 *  define them as distinct types. */
#if NUMKONG_ARCH_X8664_
#if defined(_MSC_VER)
#define nk_m512bh_from_m512i_(x) (x)
#define nk_m512h_from_m512i_(x)  (x)
#define nk_m512i_from_m512h_(x)  (x)
#define nk_m256bh_from_m256i_(x) (x)
#else
#define nk_m512bh_from_m512i_(x) ((__m512bh)(x))
#define nk_m512h_from_m512i_(x)  ((__m512h)(x))
#define nk_m512i_from_m512h_(x)  ((__m512i)(x))
#define nk_m256bh_from_m256i_(x) ((__m256bh)(x))
#endif
#endif

/*  AltiVec defines @c bool, @c vector, and @c pixel as macros, which conflict with C++. We use
 *  @c __vector directly in our code, so undef the problematic macros. */
#if defined(__POWER9_VECTOR__)
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
#endif // defined(__POWER9_VECTOR__)

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
 *  In C it expands to `static n`, which lets the compiler check bounds at every call; C++ and MSVC
 *  have no such syntax, so there it expands to nothing:
 *
 *  @code{.c}
 *  void hash_digest(uint8_t digest[nk_at_least_(32)]);
 *  void lookup(uint8_t const lut[nk_at_least_(256)]);
 *  @endcode
 *
 *  @see LWN, static bounds on array parameters: https://lwn.net/Articles/1046840/
 */
#if defined(__cplusplus) || defined(_MSC_VER)
#define nk_at_least_(n)
#else
#define nk_at_least_(n) static n
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** Packed 8-bit bit-vector (8 booleans in one byte), LSB = dimension 0. Used for Hamming
 *  distance and Jaccard similarity via popcount. Dimension count must be a multiple of 8, the
 *  values per byte. */
typedef unsigned char nk_u1x8_t;

/** Packed 4-bit signed integer pair (2 × i4 in one byte), [high nibble : low nibble]. Range per
 *  element: [−8, +7]. Elements sign-extended to i8 for arithmetic. Dimension count must be a
 *  multiple of 2, the values per byte. */
typedef unsigned char nk_i4x2_t;

/** Packed 4-bit unsigned integer pair (2 × u4 in one byte), [high nibble : low nibble]. Range per
 *  element: [0, 15]. Elements zero-extended to u8 for arithmetic. Dimension count must be a
 *  multiple of 2, the values per byte. */
typedef unsigned char nk_u4x2_t;

/** 8-bit E4M3 float (OCP FP8): sign(1) + exponent(4) + mantissa(3), bias=7. Range: ±448, no
 *  infinities (all-ones exponent → NaN at 0x7F/0xFF). 114 of 254 finite values, or 44.9%, fall
 *  in [−1, +1]. */
typedef unsigned char nk_e4m3_t;

/** 8-bit E5M2 float (OCP FP8): sign(1) + exponent(5) + mantissa(2), bias=15. Range: ±57 344,
 *  supports infinities at 0x7C/0xFC. 122 of 248 finite values (49.2%) fall in [−1, +1]. */
typedef unsigned char nk_e5m2_t;

/** 6-bit E2M3 micro-float (OCP MX v1.0): sign(1) + exponent(2) + mantissa(3), bias=1. Stored as
 *  0b00SEEMMM with 2 bits of padding. Range: ±7.5, no infinities or NaN. 64 total codes: 48 normal,
 *  14 subnormal (exp=0, mant ≠ 0), 2 zeros (±0). 18 of 64 values (28.1%) fall in [−1, +1].
 *  Subnormal values span [±0.125, ±0.875]. Losslessly promotable to E4M3 by rebiasing exponent +6
 *  (normals) or normalizing (subnormals). */
typedef unsigned char nk_e2m3_t;

/** 6-bit E3M2 micro-float (OCP MX v1.0): sign(1) + exponent(3) + mantissa(2), bias=3. Stored as
 *  0b00SEEEMM with 2 bits of padding. Range: ±28, no infinities or NaN. 64 total codes: 56 normal,
 *  6 subnormal (exp=0, mant ≠ 0), 2 zeros (±0). 26 of 64 values (40.6%) fall in [−1, +1]. Subnormal
 *  values span [±0.0625, ±0.1875]. Losslessly promotable to E5M2 by rebiasing exponent +12
 *  (normals) or normalizing (subnormals). */
typedef unsigned char nk_e3m2_t;

/** Packed 4-bit E2M1 micro-float pair (2 × e2m1 in one byte), [high nibble : low nibble]. OCP MX
 *  v1.0 sub-format: sign(1) + exponent(2) + mantissa(1), bias=1. Range: ±6.0, no Inf or NaN. 16
 *  total codes: 8 magnitudes {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} × 2 signs (two zeros). Used as
 *  the element type of MXFP4 (block=32, UE8M0 scales) and NVFP4 (block=16, UE4M3 scales). Dimension
 *  count must be a multiple of 2, the values per byte. */
typedef unsigned char nk_e2m1x2_t;

/** Unsigned 8-bit power-of-two scale (OCP MX v1.0): 8-bit biased exponent, no sign or mantissa.
 *  Encodes 2^(v - 127) for v ∈ [1, 254]; v = 0 is zero; v = 255 is the NaN-block sentinel. Used as
 *  the per-block scale byte for MXFP4, MXFP6, MXFP8, MXINT8. */
typedef unsigned char nk_ue8m0_t;

/** Unsigned 8-bit E4M3 scale (NVFP4): sign-bit forced to 0, otherwise identical to E4M3. Range: [0,
 *  +448]. Used as the per-block scale byte for NVFP4 (block=16) alongside an f32 "tensor scale". */
typedef unsigned char nk_ue4m3_t;

/** Signed 8-bit integer. Range: [−128, +127]. */
typedef signed char nk_i8_t;

/** Unsigned 8-bit integer. Range: [0, 255]. */
typedef unsigned char nk_u8_t;

/** Signed 16-bit integer. Range: [−32 768, +32 767]. */
typedef signed short nk_i16_t;

/** Unsigned 16-bit integer. Range: [0, 65 535]. */
typedef unsigned short nk_u16_t;

/** Signed 32-bit integer. Range: [−2³¹, +2³¹−1]. */
typedef signed int nk_i32_t;

/** Unsigned 32-bit integer. Range: [0, 2³²−1]. */
typedef unsigned int nk_u32_t;
/*  On LP64 targets (Linux ARM64, RISC-V 64), @c long and `long long` are both 64-bit but
 *  distinct types. NEON/RVV intrinsics on Linux expect `long*`, while Apple's NEON intrinsics
 *  expect `long long*`. Windows uses LLP64 where @c long is 32-bit, so it must use `long long`
 *  for 64-bit types. */
#if ((NUMKONG_ARCH_ARM64_ && !NUMKONG_OS_APPLE_) || NUMKONG_ARCH_RISCV64_) && !NUMKONG_OS_WINDOWS_

/** Signed 64-bit integer. Range: [−2⁶³, +2⁶³−1]. */
typedef signed long nk_i64_t;

/** Unsigned 64-bit integer. Range: [0, 2⁶⁴−1]. */
typedef unsigned long nk_u64_t;
#else

/** Signed 64-bit integer. Range: [−2⁶³, +2⁶³−1]. */
typedef signed long long nk_i64_t;

/** Unsigned 64-bit integer. Range: [0, 2⁶⁴−1]. */
typedef unsigned long long nk_u64_t;
#endif

/** Single-precision (32-bit) IEEE 754 float. sign(1) + exponent(8) + mantissa(23), bias=127. */
typedef float nk_f32_t;

/** Double-precision (64-bit) IEEE 754 float. sign(1) + exponent(11) + mantissa(52), bias=1023. */
typedef double nk_f64_t;

#if NUMKONG_ARCH_X8664_ || NUMKONG_ARCH_ARM64_ || NUMKONG_ARCH_RISCV64_ || NUMKONG_ARCH_PPC64_ || \
    NUMKONG_ARCH_LOONGARCH64_
#define NUMKONG_ARCH_64BIT_ 1
#else
#define NUMKONG_ARCH_64BIT_ 0
#endif

#if NUMKONG_ARCH_64BIT_
typedef nk_u64_t nk_size_t;
typedef nk_i64_t nk_ssize_t;
#else
typedef nk_u32_t nk_size_t;
typedef nk_i32_t nk_ssize_t;
#endif
typedef nk_f64_t nk_fmax_t;

#define NUMKONG_SIZE_MAX ((nk_size_t) - 1)

/** @c NUMKONG_NULL, analogous to @c NULL, so headers need not pull in `<stddef.h>`. @c __null gives
 *      better null-pointer diagnostics where the compiler provides it. The @c nullptr branch
 *      matters for MSVC C++: unlike C, it forbids a `void *` arm in a typed conditional, so
 *      `(void *)0` there fails to compile. */
#ifdef __GNUG__
#define NUMKONG_NULL __null
#elif defined(__cplusplus)
#define NUMKONG_NULL nullptr
#else
#define NUMKONG_NULL ((void *)0)
#endif

#define NUMKONG_F32_MIN (-3.402823466e+38f)

/** Infinities as an overflowing product, since C names them portably only in `<math.h>`. */
#define NUMKONG_F64_INF ((nk_f64_t)(1e300 * 1e300))
#define NUMKONG_F32_INF ((nk_f32_t)(1e300 * 1e300))

/** Fundamental math constants shared across the scalar, elementwise, reduction, and probability
 *  kernels. Base-2 polynomial evaluation means natural-log/exp quantities fold through log₂e
 *  and ln2, so both the single- and double-precision spellings are provided and each site keeps
 *  its exact bits. Trigonometric range-reduction constants (π high/low, 1/π, π/2) are
 *  intentionally not here — they are a Cody-Waite set local to the `trigonometry/` sources,
 *  tuned per approximation. */
#define NUMKONG_F32_LN2_   0.6931471805599453f // ln2 (single precision)
#define NUMKONG_F64_LN2_   0.6931471805599453  // ln2 (double precision)
#define NUMKONG_F32_LOG2E_ 1.4426950408889634f // log₂e = 1/ln2 (single)
#define NUMKONG_F64_LOG2E_ 1.4426950408889634  // log₂e (double precision)

#define NUMKONG_I64_MAX 9223372036854775807LL
#define NUMKONG_I64_MIN (-9223372036854775807LL - 1LL)
#define NUMKONG_U64_MAX 18446744073709551615ULL
#define NUMKONG_U64_MIN 0x0ULL

#define NUMKONG_I32_MAX 2147483647
#define NUMKONG_I32_MIN (-2147483647 - 1)
#define NUMKONG_U32_MAX 4294967295U
#define NUMKONG_U32_MIN 0x0U

#define NUMKONG_I16_MAX 32767
#define NUMKONG_I16_MIN (-32767 - 1)
#define NUMKONG_U16_MAX 65535U
#define NUMKONG_U16_MIN 0x0U

#define NUMKONG_I8_MAX 127
#define NUMKONG_I8_MIN (-127 - 1)
#define NUMKONG_U8_MAX 255U
#define NUMKONG_U8_MIN 0x0U

#define NUMKONG_F16_MAX nk_u16_as_f16_(0x7BFF) // IEEE 754 binary16: +65504.0
#define NUMKONG_F16_MIN nk_u16_as_f16_(0xFBFF) // IEEE 754 binary16: -65504.0

#define NUMKONG_BF16_MAX nk_u16_as_bf16_(0x7F7F) // BFloat16: ~+3.39e38
#define NUMKONG_BF16_MIN nk_u16_as_bf16_(0xFF7F) // BFloat16: ~-3.39e38

#define NUMKONG_E4M3_MAX 0x7E // FP8 E4M3: +448.0
#define NUMKONG_E4M3_MIN 0xFE // FP8 E4M3: -448.0

#define NUMKONG_E5M2_MAX 0x7B // FP8 E5M2: +57344.0
#define NUMKONG_E5M2_MIN 0xFB // FP8 E5M2: -57344.0

#define NUMKONG_E2M3_MAX 0x1F // FP6 E2M3: +7.5
#define NUMKONG_E2M3_MIN 0x3F // FP6 E2M3: -7.5

#define NUMKONG_E3M2_MAX 0x1F // FP6 E3M2: +28.0
#define NUMKONG_E3M2_MIN 0x3F // FP6 E3M2: -28.0

#define NUMKONG_BITS_PER_BYTE    8
#define NUMKONG_NIBBLES_PER_BYTE 2

/**
 *  @brief Enumeration of supported scalar data types.
 *
 *  Includes complex type descriptors which in C code would use the real counterparts, but the
 *  independent flags contain metadata to be passed between programming language interfaces.
 */
typedef enum {

    /** Unknown data type. */
    nk_dtype_unknown_k = 0,

    /** Single-bit values packed into 8-bit words. */
    nk_u1_k = 1 << 1,

    /** 8-bit signed integer. */
    nk_i8_k = 1 << 2,

    /** 16-bit signed integer. */
    nk_i16_k = 1 << 3,

    /** 32-bit signed integer. */
    nk_i32_k = 1 << 4,

    /** 64-bit signed integer. */
    nk_i64_k = 1 << 5,

    /** 8-bit unsigned integer. */
    nk_u8_k = 1 << 6,

    /** 16-bit unsigned integer. */
    nk_u16_k = 1 << 7,

    /** 32-bit unsigned integer. */
    nk_u32_k = 1 << 8,

    /** 64-bit unsigned integer. */
    nk_u64_k = 1 << 9,

    /** Double precision floating point. */
    nk_f64_k = 1 << 10,

    /** Single precision floating point. */
    nk_f32_k = 1 << 11,

    /** Half precision floating point. */
    nk_f16_k = 1 << 12,

    /** Brain floating point. */
    nk_bf16_k = 1 << 13,

    /** FP8 E4M3 floating point. */
    nk_e4m3_k = 1 << 14,

    /** FP8 E5M2 floating point. */
    nk_e5m2_k = 1 << 15,

    /** 4-bit signed integers packed into 8-bit words. */
    nk_i4_k = 1 << 16,

    /** 4-bit unsigned integers packed into 8-bit words. */
    nk_u4_k = 1 << 17,

    /** FP6 E2M3 floating point. */
    nk_e2m3_k = 1 << 18,

    /** FP6 E3M2 floating point. */
    nk_e3m2_k = 1 << 19,

    /** Complex double precision floating point. */
    nk_f64c_k = 1 << 20,

    /** Complex single precision floating point. */
    nk_f32c_k = 1 << 21,

    /** Complex half precision floating point. */
    nk_f16c_k = 1 << 22,

    /** Complex brain floating point. */
    nk_bf16c_k = 1 << 23,

    /** FP4 E2M1 floating point (element of MXFP4 and NVFP4). */
    nk_e2m1_k = 1 << 24,

    /** UE8M0 unsigned pow-2 scale byte (MX family block scale). */
    nk_ue8m0_k = 1 << 25,

    /** UE4M3 unsigned E4M3 scale byte (NVFP4 block scale). */
    nk_ue4m3_k = 1 << 26,

    // Composite block-scaled formats encoded as `element_dtype | scale_dtype`. Each OR is unique
    // (popcount = 2) and cannot collide with any singleton (popcount = 1). Block size is implicit
    // from the scale dtype: `ue4m3` → 16 (NVFP4), `ue8m0` → 32 (MX family).

    /** NVIDIA NVFP4 (block=16, f32 tensor scale). */
    nk_nvfp4_k = nk_e2m1_k | nk_ue4m3_k,

    /** OCP MXFP4 (block=32). */
    nk_mxfp4_k = nk_e2m1_k | nk_ue8m0_k,

    /** OCP MXFP6 (E2M3 variant, block=32). */
    nk_mxfp6_e2m3_k = nk_e2m3_k | nk_ue8m0_k,

    /** OCP MXFP6 (E3M2 variant, block=32). */
    nk_mxfp6_e3m2_k = nk_e3m2_k | nk_ue8m0_k,

    /** OCP MXFP8 (E4M3 variant, block=32). */
    nk_mxfp8_e4m3_k = nk_e4m3_k | nk_ue8m0_k,

    /** OCP MXFP8 (E5M2 variant, block=32). */
    nk_mxfp8_e5m2_k = nk_e5m2_k | nk_ue8m0_k,

    /** OCP MXINT8 (block=32). */
    nk_mxint8_k = nk_i8_k | nk_ue8m0_k,
} nk_dtype_t;

/** Outcome of every NumKong call that can fail: zero on success, negative when nothing was
 *  written. Positive values are reserved for results written with a caveat. */
#if NUMKONG_CXX_STANDARD_ >= 201703L || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L)
typedef enum [[nodiscard]] {
#else
typedef enum {
#endif

    /** Scheduled, or finished, without error. */
    nk_success_k = 0,

    /** The queue could not take more memory, or record another block. */
    nk_bad_alloc_k = -10,

    /** Operand shapes contradict each other, like head counts that don't divide. */
    nk_unexpected_dimensions_k = -15,

    /** No GPU of the vendor this build targets answers. */
    nk_missing_gpu_k = -16,

    /** The device code lacks the kernel, or it failed to build, launch or finish. */
    nk_device_code_mismatch_k = -17,

    /** An operand lies in memory the device cannot address. */
    nk_device_memory_mismatch_k = -18,

    /** No capability in the capability mask has this kernel. */
    nk_missing_kernel_k = -19,

    /** An operand or stride breaks the alignment contract. */
    nk_misaligned_k = -20,

    /** The buffer was packed by another capability or layout than the one reading it. */
    nk_pack_mismatch_k = -21,

    /** A dispatch point or finder called from a header-only build, which links no library. */
    nk_missing_library_k = -22,
} nk_status_t;

/** Static English description of @p status, behind @c nk_status_name. */
NUMKONG_CONSTEXPR char const *nk_status_name_(nk_status_t status) {
    switch (status) {
    case nk_success_k: return "success";
    case nk_bad_alloc_k: return "out of memory";
    case nk_unexpected_dimensions_k: return "unexpected dimensions";
    case nk_missing_gpu_k: return "no GPU of this vendor";
    case nk_device_code_mismatch_k: return "no kernel ran on this device";
    case nk_device_memory_mismatch_k: return "memory the device cannot reach";
    case nk_missing_kernel_k: return "no kernel for these capabilities";
    case nk_misaligned_k: return "misaligned operand";
    case nk_pack_mismatch_k: return "packed by another capability";
    case nk_missing_library_k: return "the NumKong library is not linked; call a capability's kernel or link it";
    }
    return "an unrecognized status";
}

/** Static English description of @p status, never null. */
NUMKONG_API char const *nk_status_name(nk_status_t status);

#if NUMKONG_HEADER_ONLY
NUMKONG_API char const *nk_status_name(nk_status_t status) { return nk_status_name_(status); }
#endif

/**
 *  @brief Descriptor for a block-scaled tensor layout (OCP MX family + NVIDIA NVFP4).
 *
 *  Elements are grouped in fixed-size contiguous blocks; each block has its own
 *  @c scale_dtype scale byte stored in a separate scales buffer. An optional
 *  @c tensor_scale_dtype scalar multiplier applies to the whole tensor (NVFP4's per-tensor
 *  FP32 factor; absent for the MX family).
 *
 *  Plain (non-block-scaled) buffers are described by `scale_dtype = nk_dtype_unknown_k`,
 *  `tensor_scale_dtype = nk_dtype_unknown_k`, and `block_size = 0` — use the `nk_plain()` factory.
 */
typedef struct {

    /** Per-element dtype: e2m1/e4m3/e5m2/e2m3/e3m2/i8 (or any for plain). */
    nk_dtype_t element_dtype;

    /** Per-block scale: ue8m0 (MX) or ue4m3 (NVFP4); unknown for plain. */
    nk_dtype_t scale_dtype;

    /** Per-tensor multiplier: f32 (NVFP4) or unknown (MX, plain). */
    nk_dtype_t tensor_scale_dtype;

    /** Elements per block: 16 (NVFP4) or 32 (MX); 0 for plain. */
    nk_size_t block_size;
} nk_block_scaled_format_t;

/** The kind of number a data type holds, as @c nk_dtype_family reports it. */
typedef enum {
    nk_dtype_family_unknown_k = 0,
    nk_dtype_family_float_k,
    nk_dtype_family_complex_float_k,
    nk_dtype_family_int_k,
    nk_dtype_family_uint_k,
} nk_dtype_family_t;

/** Compares an explicit-length string against a NUL-terminated literal. */
NUMKONG_CONSTEXPR int nk_same_literal_(char const *name, nk_size_t length, char const *literal) {
    nk_size_t position = 0;
    for (; position != length; ++position)
        if (literal[position] == '\0' || name[position] != literal[position]) return 0;
    return literal[position] == '\0';
}

/** `{nk_e2m1_k, nk_ue4m3_k, nk_f32_k, 16}` — NVIDIA NVFP4 (Blackwell-native). */
NUMKONG_CONSTEXPR nk_block_scaled_format_t nk_nvfp4(void) {
    nk_block_scaled_format_t format = {nk_e2m1_k, nk_ue4m3_k, nk_f32_k, 16};
    return format;
}

/** `{nk_e2m1_k, nk_ue8m0_k, unknown, 32}` — OCP MXFP4. */
NUMKONG_CONSTEXPR nk_block_scaled_format_t nk_mxfp4(void) {
    nk_block_scaled_format_t format = {nk_e2m1_k, nk_ue8m0_k, nk_dtype_unknown_k, 32};
    return format;
}

/** `{nk_e2m3_k, nk_ue8m0_k, unknown, 32}` — OCP MXFP6 (E2M3 variant). */
NUMKONG_CONSTEXPR nk_block_scaled_format_t nk_mxfp6_e2m3(void) {
    nk_block_scaled_format_t format = {nk_e2m3_k, nk_ue8m0_k, nk_dtype_unknown_k, 32};
    return format;
}

/** `{nk_e3m2_k, nk_ue8m0_k, unknown, 32}` — OCP MXFP6 (E3M2 variant). */
NUMKONG_CONSTEXPR nk_block_scaled_format_t nk_mxfp6_e3m2(void) {
    nk_block_scaled_format_t format = {nk_e3m2_k, nk_ue8m0_k, nk_dtype_unknown_k, 32};
    return format;
}

/** `{nk_e4m3_k, nk_ue8m0_k, unknown, 32}` — OCP MXFP8 (E4M3 variant). */
NUMKONG_CONSTEXPR nk_block_scaled_format_t nk_mxfp8_e4m3(void) {
    nk_block_scaled_format_t format = {nk_e4m3_k, nk_ue8m0_k, nk_dtype_unknown_k, 32};
    return format;
}

/** `{nk_e5m2_k, nk_ue8m0_k, unknown, 32}` — OCP MXFP8 (E5M2 variant). */
NUMKONG_CONSTEXPR nk_block_scaled_format_t nk_mxfp8_e5m2(void) {
    nk_block_scaled_format_t format = {nk_e5m2_k, nk_ue8m0_k, nk_dtype_unknown_k, 32};
    return format;
}

/** `{nk_i8_k, nk_ue8m0_k, unknown, 32}` — OCP MXINT8. */
NUMKONG_CONSTEXPR nk_block_scaled_format_t nk_mxint8(void) {
    nk_block_scaled_format_t format = {nk_i8_k, nk_ue8m0_k, nk_dtype_unknown_k, 32};
    return format;
}

/** `{element_dtype, unknown, unknown, 0}` — plain scalar buffer of @p element_dtype. */
NUMKONG_CONSTEXPR nk_block_scaled_format_t nk_plain(nk_dtype_t element_dtype) {
    nk_block_scaled_format_t format = {element_dtype, nk_dtype_unknown_k, nk_dtype_unknown_k, 0};
    return format;
}

/** Build a block-scaled format descriptor from a composite @p dtype enum value. Returns
 *  `nk_plain(dtype)` when @p dtype is not a composite. */
NUMKONG_CONSTEXPR nk_block_scaled_format_t nk_block_scaled_format_of_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_nvfp4_k: return nk_nvfp4();
    case nk_mxfp4_k: return nk_mxfp4();
    case nk_mxfp6_e2m3_k: return nk_mxfp6_e2m3();
    case nk_mxfp6_e3m2_k: return nk_mxfp6_e3m2();
    case nk_mxfp8_e4m3_k: return nk_mxfp8_e4m3();
    case nk_mxfp8_e5m2_k: return nk_mxfp8_e5m2();
    case nk_mxint8_k: return nk_mxint8();
    default: return nk_plain(dtype);
    }
}

/** True when @p dtype encodes a composite block-scaled format. */
NUMKONG_CONSTEXPR int nk_dtype_is_block_scaled(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_nvfp4_k: return 1;
    case nk_mxfp4_k: return 1;
    case nk_mxfp6_e2m3_k: return 1;
    case nk_mxfp6_e3m2_k: return 1;
    case nk_mxfp8_e4m3_k: return 1;
    case nk_mxfp8_e5m2_k: return 1;
    case nk_mxint8_k: return 1;
    default: return 0;
    }
}

/** Extracts the element dtype from a composite; returns @p dtype unchanged for plain inputs. */
NUMKONG_CONSTEXPR nk_dtype_t nk_dtype_element(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_nvfp4_k: return nk_e2m1_k;
    case nk_mxfp4_k: return nk_e2m1_k;
    case nk_mxfp6_e2m3_k: return nk_e2m3_k;
    case nk_mxfp6_e3m2_k: return nk_e3m2_k;
    case nk_mxfp8_e4m3_k: return nk_e4m3_k;
    case nk_mxfp8_e5m2_k: return nk_e5m2_k;
    case nk_mxint8_k: return nk_i8_k;
    default: return dtype;
    }
}

/** Extracts the scale dtype from a composite; returns @c nk_dtype_unknown_k for plain inputs. */
NUMKONG_CONSTEXPR nk_dtype_t nk_dtype_scale(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_nvfp4_k: return nk_ue4m3_k;
    case nk_mxfp4_k: return nk_ue8m0_k;
    case nk_mxfp6_e2m3_k: return nk_ue8m0_k;
    case nk_mxfp6_e3m2_k: return nk_ue8m0_k;
    case nk_mxfp8_e4m3_k: return nk_ue8m0_k;
    case nk_mxfp8_e5m2_k: return nk_ue8m0_k;
    case nk_mxint8_k: return nk_ue8m0_k;
    default: return nk_dtype_unknown_k;
    }
}

/** Block size implied by a composite dtype; 0 for plain inputs. */
NUMKONG_CONSTEXPR nk_size_t nk_dtype_block_size(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_nvfp4_k: return 16;
    case nk_mxfp4_k: return 32;
    case nk_mxfp6_e2m3_k: return 32;
    case nk_mxfp6_e3m2_k: return 32;
    case nk_mxfp8_e4m3_k: return 32;
    case nk_mxfp8_e5m2_k: return 32;
    case nk_mxint8_k: return 32;
    default: return 0;
    }
}

/** Classifies the family of the dtype. */
NUMKONG_CONSTEXPR nk_dtype_family_t nk_dtype_family(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return nk_dtype_family_float_k;
    case nk_f32_k: return nk_dtype_family_float_k;
    case nk_f16_k: return nk_dtype_family_float_k;
    case nk_bf16_k: return nk_dtype_family_float_k;
    case nk_e4m3_k: return nk_dtype_family_float_k;
    case nk_e5m2_k: return nk_dtype_family_float_k;
    case nk_e2m3_k: return nk_dtype_family_float_k;
    case nk_e3m2_k: return nk_dtype_family_float_k;
    case nk_e2m1_k: return nk_dtype_family_float_k;
    case nk_ue8m0_k: return nk_dtype_family_float_k;
    case nk_ue4m3_k: return nk_dtype_family_float_k;
    // Composite block-scaled dtypes — family of the logical element
    case nk_nvfp4_k: return nk_dtype_family_float_k;
    case nk_mxfp4_k: return nk_dtype_family_float_k;
    case nk_mxfp6_e2m3_k: return nk_dtype_family_float_k;
    case nk_mxfp6_e3m2_k: return nk_dtype_family_float_k;
    case nk_mxfp8_e4m3_k: return nk_dtype_family_float_k;
    case nk_mxfp8_e5m2_k: return nk_dtype_family_float_k;
    case nk_mxint8_k: return nk_dtype_family_int_k;
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

/** Returns the number of bits in a single scalar of a given type. */
NUMKONG_CONSTEXPR nk_size_t nk_dtype_bits(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return 64;
    case nk_f32_k: return 32;
    case nk_f16_k: return 16;
    case nk_bf16_k: return 16;
    case nk_e4m3_k: return 8;
    case nk_e5m2_k: return 8;
    case nk_e2m3_k: return 8;
    case nk_e3m2_k: return 8;
    case nk_e2m1_k: return 4;
    case nk_ue8m0_k: return 8;
    case nk_ue4m3_k: return 8;
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
    // Composite block-scaled dtypes — bits of the whole block value. One "storage value"
    // is one block: `sizeof(nk_<composite>_t) * NUMKONG_BITS_PER_BYTE`. Pair with
    // `nk_dimensions_per_value` to compute storage bytes via `size_values * (bits / 8)`.
    case nk_nvfp4_k: return 9 * NUMKONG_BITS_PER_BYTE;       // 16 × E2M1 nibbles + 1 UE4M3 scale
    case nk_mxfp4_k: return 17 * NUMKONG_BITS_PER_BYTE;      // 32 × E2M1 nibbles + 1 UE8M0 scale
    case nk_mxfp6_e2m3_k: return 33 * NUMKONG_BITS_PER_BYTE; // 32 × E2M3 + 1 UE8M0 scale
    case nk_mxfp6_e3m2_k: return 33 * NUMKONG_BITS_PER_BYTE; // 32 × E3M2 + 1 UE8M0 scale
    case nk_mxfp8_e4m3_k: return 33 * NUMKONG_BITS_PER_BYTE; // 32 × E4M3 + 1 UE8M0 scale
    case nk_mxfp8_e5m2_k: return 33 * NUMKONG_BITS_PER_BYTE; // 32 × E5M2 + 1 UE8M0 scale
    case nk_mxint8_k: return 33 * NUMKONG_BITS_PER_BYTE;     // 32 × i8 + 1 UE8M0 scale
    default: return 0;
    }
}

/**
 *  @brief Returns the rounding error each term may add to a sum kept in @p dtype, relative to the
 *      terms' magnitudes: |result − exact| ≤ (roundings + 1) · bound · Σ|terms|.
 *
 *  Floats get their unit roundoff with two bits of slack for matrix units that truncate; integer
 *  sums are exact. Kernel families derive their bounds from it, like @c nk_dot_error_bound.
 */
NUMKONG_CONSTEXPR nk_f64_t nk_accumulation_error_bound(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return 0x1p-51;
    case nk_f64c_k: return 0x1p-51;
    case nk_f32_k: return 0x1p-22;
    case nk_f32c_k: return 0x1p-22;
    default: return 0;
    }
}

/** Canonical name of a data type - the spelling bindings parse and interchange formats carry;
 *  "unknown" for unrecognized values. */
NUMKONG_CONSTEXPR char const *nk_dtype_name(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return "f64";
    case nk_f32_k: return "f32";
    case nk_f16_k: return "f16";
    case nk_bf16_k: return "bf16";
    case nk_e4m3_k: return "e4m3";
    case nk_e5m2_k: return "e5m2";
    case nk_e2m3_k: return "e2m3";
    case nk_e3m2_k: return "e3m2";
    case nk_e2m1_k: return "e2m1";
    case nk_ue8m0_k: return "ue8m0";
    case nk_ue4m3_k: return "ue4m3";
    case nk_f64c_k: return "f64c";
    case nk_f32c_k: return "f32c";
    case nk_f16c_k: return "f16c";
    case nk_bf16c_k: return "bf16c";
    case nk_u1_k: return "u1";
    case nk_u4_k: return "u4";
    case nk_u8_k: return "u8";
    case nk_u16_k: return "u16";
    case nk_u32_k: return "u32";
    case nk_u64_k: return "u64";
    case nk_i4_k: return "i4";
    case nk_i8_k: return "i8";
    case nk_i16_k: return "i16";
    case nk_i32_k: return "i32";
    case nk_i64_k: return "i64";
    case nk_nvfp4_k: return "nvfp4";
    case nk_mxfp4_k: return "mxfp4";
    case nk_mxfp6_e2m3_k: return "mxfp6_e2m3";
    case nk_mxfp6_e3m2_k: return "mxfp6_e3m2";
    case nk_mxfp8_e4m3_k: return "mxfp8_e4m3";
    case nk_mxfp8_e5m2_k: return "mxfp8_e5m2";
    case nk_mxint8_k: return "mxint8";
    default: return "unknown";
    }
}

/** Inverse of @c nk_dtype_name over an explicit-length string; @c nk_dtype_unknown_k
 *  for unrecognized names. */
NUMKONG_CONSTEXPR nk_dtype_t nk_dtype_named(char const *name, nk_size_t length) {
    if (nk_same_literal_(name, length, "f64")) return nk_f64_k;
    if (nk_same_literal_(name, length, "f32")) return nk_f32_k;
    if (nk_same_literal_(name, length, "f16")) return nk_f16_k;
    if (nk_same_literal_(name, length, "bf16")) return nk_bf16_k;
    if (nk_same_literal_(name, length, "e4m3")) return nk_e4m3_k;
    if (nk_same_literal_(name, length, "e5m2")) return nk_e5m2_k;
    if (nk_same_literal_(name, length, "e2m3")) return nk_e2m3_k;
    if (nk_same_literal_(name, length, "e3m2")) return nk_e3m2_k;
    if (nk_same_literal_(name, length, "e2m1")) return nk_e2m1_k;
    if (nk_same_literal_(name, length, "ue8m0")) return nk_ue8m0_k;
    if (nk_same_literal_(name, length, "ue4m3")) return nk_ue4m3_k;
    if (nk_same_literal_(name, length, "f64c")) return nk_f64c_k;
    if (nk_same_literal_(name, length, "f32c")) return nk_f32c_k;
    if (nk_same_literal_(name, length, "f16c")) return nk_f16c_k;
    if (nk_same_literal_(name, length, "bf16c")) return nk_bf16c_k;
    if (nk_same_literal_(name, length, "u1")) return nk_u1_k;
    if (nk_same_literal_(name, length, "u4")) return nk_u4_k;
    if (nk_same_literal_(name, length, "u8")) return nk_u8_k;
    if (nk_same_literal_(name, length, "u16")) return nk_u16_k;
    if (nk_same_literal_(name, length, "u32")) return nk_u32_k;
    if (nk_same_literal_(name, length, "u64")) return nk_u64_k;
    if (nk_same_literal_(name, length, "i4")) return nk_i4_k;
    if (nk_same_literal_(name, length, "i8")) return nk_i8_k;
    if (nk_same_literal_(name, length, "i16")) return nk_i16_k;
    if (nk_same_literal_(name, length, "i32")) return nk_i32_k;
    if (nk_same_literal_(name, length, "i64")) return nk_i64_k;
    if (nk_same_literal_(name, length, "nvfp4")) return nk_nvfp4_k;
    if (nk_same_literal_(name, length, "mxfp4")) return nk_mxfp4_k;
    if (nk_same_literal_(name, length, "mxfp6_e2m3")) return nk_mxfp6_e2m3_k;
    if (nk_same_literal_(name, length, "mxfp6_e3m2")) return nk_mxfp6_e3m2_k;
    if (nk_same_literal_(name, length, "mxfp8_e4m3")) return nk_mxfp8_e4m3_k;
    if (nk_same_literal_(name, length, "mxfp8_e5m2")) return nk_mxfp8_e5m2_k;
    if (nk_same_literal_(name, length, "mxint8")) return nk_mxint8_k;
    return nk_dtype_unknown_k;
}

/** Returns how many logical dimensions are packed into one storage value. For sub-byte types
 *  multiple dimensions share a single byte container. For byte-or-larger types this is always 1. */
NUMKONG_CONSTEXPR nk_size_t nk_dimensions_per_value(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_u1_k: return 8;
    case nk_i4_k: return 2;
    case nk_u4_k: return 2;
    case nk_e2m1_k: return 2;
    // Composite block-scaled dtypes — one value is one whole block of logical elements.
    case nk_nvfp4_k: return 16;      // 16 nibbles per block
    case nk_mxfp4_k: return 32;      // 32 nibbles per block
    case nk_mxfp6_e2m3_k: return 32; // 32 E2M3 per block
    case nk_mxfp6_e3m2_k: return 32; // 32 E3M2 per block
    case nk_mxfp8_e4m3_k: return 32; // 32 E4M3 per block
    case nk_mxfp8_e5m2_k: return 32; // 32 E5M2 per block
    case nk_mxint8_k: return 32;     // 32 i8 per block
    default: return 1;
    }
}

/**
 *  @brief Half-precision (16-bit) IEEE 754 float.
 *
 *  Layout: sign(1) + exponent(5) + mantissa(10), bias=15.
 *  Range: ±65 504, epsilon at 1.0 ≈ 9.77×10⁻⁴. 30 722 of 63 488 finite values (48.4%) in [−1, +1].
 *
 *  - `unsigned short` by default, on every compiler and architecture.
 *  - `NUMKONG_NATIVE_F16=1`: @c __fp16 on Arm, @c _Float16 elsewhere.
 *
 *  The type crosses the exported ABI by value — @c nk_f16_sqrt_best, @c nk_f16_order_best — and a
 *  native half is passed in a different register class than `unsigned short`. Deriving it from
 *  `-march`, as this once did, let each consumer reach its own answer: Swift's header import sees a
 *  native half on Apple silicon while the C sources it links were built `unsigned short`, and
 *  `rust/scalar.rs` binds @c u16 outright. Enabling it is a decision for a whole build, never for
 *  one translation unit, so the default cannot depend on flags.
 */
#if !defined(NUMKONG_NATIVE_F16)
#define NUMKONG_NATIVE_F16 0
#endif

#if NUMKONG_NATIVE_F16
#if defined(__ARM_FP16_FORMAT_IEEE)
typedef __fp16 nk_f16_t;
/*  @c __FLT16_MAX__ covers GCC 12+ and Clang; @c nk_is_keyword_ catches any Clang that predates it.
 *  Between them no compiler-version test is needed, which matters because Apple Clang's
 *  @c __clang_major__ does not track upstream LLVM. */
#elif defined(__FLT16_MAX__) || nk_is_keyword_(_Float16)
typedef _Float16 nk_f16_t;
#else
#error "NUMKONG_NATIVE_F16=1, but this compiler has no native half type"
#endif
#else
typedef unsigned short nk_f16_t;
#endif

/**
 *  @brief BFloat16 (16-bit) float — truncated IEEE 754 single-precision.
 *
 *  Layout: sign(1) + exponent(8) + mantissa(7), bias=127.
 *  Same dynamic range as f32, epsilon ≈ 7.81×10⁻³.
 *  32 514 of 65 280 finite values (49.8%) in [−1, +1]. Wider range than f16 but lower precision.
 *
 *  - `unsigned short` by default, on every compiler and architecture.
 *  - `NUMKONG_NATIVE_BF16=1`: @c __bf16 on GCC and Clang. See @c nk_f16_t for why this is opt-in.
 *
 *  The compilers have added @c __bf16 support in compliance with the x86-64 psABI spec. The
 *  motivation for this new special type is summed up as:
 *
 *  "Currently @c __bfloat16 is a typedef of short, which creates a problem where the compiler does
 *  not raise any alarms if it is used to add, subtract, multiply or divide, but the result of the
 *  calculation is actually meaningless. To solve this problem, a real scalar type @c __Bfloat16
 *  needs to be introduced. It is mainly used for intrinsics, not available for C standard
 *  operators. @c __Bfloat16 will also be used for movement like passing parameter, load and store,
 *  vector initialization, vector shuffle, and etc. It creates a need for a corresponding psABI."
 *
 *  @warning Apple Clang has hard time with bf16.
 *  @see Writing ARM64 code for Apple platforms: https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms
 *  @see Apple Developer Forums thread on bf16: https://forums.developer.apple.com/forums/thread/726201
 *  @see GCC and LLVM bf16 support on Phoronix: https://www.phoronix.com/news/GCC-LLVM-bf16-BFloat16-Type
 */
#if !defined(NUMKONG_NATIVE_BF16)
#define NUMKONG_NATIVE_BF16 0
#endif

/*  GCC 13+ is the first to define @c __BFLT16_MAX__; Clang defines no bf16 macro at all, so it is
 *  caught by @c nk_is_keyword_ instead. The AVX512BF16 feature macro is deliberately not consulted:
 *  GCC 11 and 12 define it while rejecting @c __bf16, since it names instructions, not the type. */
#if NUMKONG_NATIVE_BF16
#if defined(__ARM_BF16_FORMAT_ALTERNATIVE) || defined(__BFLT16_MAX__) || nk_is_keyword_(__bf16)
typedef __bf16 nk_bf16_t;
#else
#error "NUMKONG_NATIVE_BF16=1, but this compiler has no native bfloat type"
#endif
#else
typedef unsigned short nk_bf16_t;
#endif

/**
 *  @brief Alias for the half-precision floating-point type on Arm.
 *
 *  Clang and GCC bring the @c float16_t symbol when you compile for AArch64.
 *  MSVC lacks it, and it's @c vld1_f16-like intrinsics are in reality macros,
 *  that cast to 16-bit integers internally, instead of using floats.
 *  Some of those are defined as aliases, so we use `#define` preprocessor
 *  directives instead of @c typedef to avoid errors.
 */
#if NUMKONG_ARCH_ARM64_
#if defined(_MSC_VER)
#define nk_f16_for_arm_simd_t  nk_f16_t
#define nk_bf16_for_arm_simd_t nk_bf16_t
#else
#define nk_f16_for_arm_simd_t  float16_t
#define nk_bf16_for_arm_simd_t bfloat16_t
#endif
#endif

/**
 *  @brief Block-scaled composite POD types — one value is one whole block.
 *
 *  Each composite combines a packed element buffer with its per-block scale byte in a single
 *  POD struct. The struct IS the value: `sizeof(nk_nvfp4_t) == 9` means one NVFP4 value, not
 *  one element. Storage in containers strides by `sizeof(struct)`; logical element count per
 *  value is reported by `nk_dimensions_per_value(composite_dtype)` (16 for NVFP4, 32 for MX).
 *  Dimension count must be a multiple of that block size.
 *
 *  The NVFP4 per-tensor f32 tensor scale multiplier is not part of the block value — it lives on
 *  the enclosing tensor and is passed explicitly to encode/decode helpers.
 */
typedef struct NUMKONG_MAY_ALIAS_ {

    /** 16 E2M1 nibbles packed 2/byte. */
    nk_e2m1x2_t elements_[8];

    /** Per-block UE4M3 scale. */
    nk_ue4m3_t scale_;

} nk_nvfp4_t;

typedef struct NUMKONG_MAY_ALIAS_ {

    /** 32 E2M1 nibbles packed 2/byte. */
    nk_e2m1x2_t elements_[16];

    /** Per-block UE8M0 pow-2 scale. */
    nk_ue8m0_t scale_;
} nk_mxfp4_t;

typedef struct NUMKONG_MAY_ALIAS_ {
    nk_e2m3_t elements_[32];
    nk_ue8m0_t scale_;
} nk_mxfp6_e2m3_t;

typedef struct NUMKONG_MAY_ALIAS_ {
    nk_e3m2_t elements_[32];
    nk_ue8m0_t scale_;
} nk_mxfp6_e3m2_t;

typedef struct NUMKONG_MAY_ALIAS_ {
    nk_e4m3_t elements_[32];
    nk_ue8m0_t scale_;
} nk_mxfp8_e4m3_t;

typedef struct NUMKONG_MAY_ALIAS_ {
    nk_e5m2_t elements_[32];
    nk_ue8m0_t scale_;
} nk_mxfp8_e5m2_t;

typedef struct NUMKONG_MAY_ALIAS_ {
    nk_i8_t elements_[32];
    nk_ue8m0_t scale_;
} nk_mxint8_t;

/**
 *  @brief Similar to @c assert, the @c nk_assert_ checks library invariants in @c NUMKONG_DEBUG
 *      builds, aborting on failure; in release it type-checks the condition without evaluating it.
 *  @note If you want to catch it, put a breakpoint at @c abort.
 */
#if defined(__METAL_VERSION__)
#define nk_assert_(condition)
#elif NUMKONG_DEBUG && defined(__CUDA_ARCH__) // ? CUDA code for GPUs
static __device__ __noinline__ void nk_assert_cuda_failure_(char const *condition, char const *file, int line) {
    printf("Assertion failed: %s, in file %s, line %d\n", condition, file, line);
    __trap();
}
#define nk_assert_(condition)                                                          \
    do {                                                                               \
        if (!(condition)) { nk_assert_cuda_failure_(#condition, __FILE__, __LINE__); } \
    } while (0)
#elif NUMKONG_DEBUG && __STDC_HOSTED__ // ? CPU code with LibC
NUMKONG_CONSTEXPR void nk_assert_failure_(char const *condition, char const *file, int line) {
    fprintf(stderr, "Assertion failed: %s, in file %s, line %d\n", condition, file, line);
    abort();
}
#define nk_assert_(condition)                                                     \
    do {                                                                          \
        if (!(condition)) { nk_assert_failure_(#condition, __FILE__, __LINE__); } \
    } while (0)
#elif NUMKONG_DEBUG && defined(_MSC_VER) && !defined(__clang__) // ? No LibC, and MSVC has no `__builtin_trap`
#define nk_assert_(condition)             \
    do {                                  \
        if (!(condition)) __debugbreak(); \
    } while (0)
#elif NUMKONG_DEBUG // ? No LibC: nothing to print with, so trap in place
#define nk_assert_(condition)               \
    do {                                    \
        if (!(condition)) __builtin_trap(); \
    } while (0)
#else
#define nk_assert_(condition) nk_unused_(sizeof(!(condition)))
#endif

/** Asserts that @p dimensions fill whole storage values of @p dtype, as the sub-byte kernels
 *  require. */
#define nk_assert_dims_(dimensions, dtype) nk_assert_((dimensions) % nk_dimensions_per_value(dtype) == 0)

/** Compile-time assert akin to C++ @c static_assert. Uses the native assertion where available
 *  (C++11 @c static_assert, C11 @c _Static_assert); the older-C typedef fallback must sit at file
 *  scope to stay clear of @c -Wunused-local-typedef. */
#if NUMKONG_CXX_STANDARD_ >= 201103L
#define nk_static_assert_(condition, name) static_assert(condition, #name)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define nk_static_assert_(condition, name) _Static_assert(condition, #name)
#elif defined(_MSC_VER)
#define nk_static_assert_(condition, name) static_assert(condition, #name)
#else
#define nk_static_assert_(condition, name) typedef char nk_static_assert_##name[(condition) ? 1 : -1]
#endif

nk_static_assert_(sizeof(nk_u1x8_t) == 1, nk_u1x8_t_must_be_1_byte);
nk_static_assert_(sizeof(nk_i4x2_t) == 1, nk_i4_t_must_be_1_byte);
nk_static_assert_(sizeof(nk_u4x2_t) == 1, nk_u4_t_must_be_1_byte);
nk_static_assert_(sizeof(nk_e4m3_t) == 1, nk_e4m3_t_must_be_1_byte);
nk_static_assert_(sizeof(nk_e5m2_t) == 1, nk_e5m2_t_must_be_1_byte);
nk_static_assert_(sizeof(nk_e2m3_t) == 1, nk_e2m3_t_must_be_1_byte);
nk_static_assert_(sizeof(nk_e3m2_t) == 1, nk_e3m2_t_must_be_1_byte);
nk_static_assert_(sizeof(nk_e2m1x2_t) == 1, nk_e2m1x2_t_must_be_1_byte);
nk_static_assert_(sizeof(nk_ue8m0_t) == 1, nk_ue8m0_t_must_be_1_byte);
nk_static_assert_(sizeof(nk_ue4m3_t) == 1, nk_ue4m3_t_must_be_1_byte);
nk_static_assert_(sizeof(nk_i8_t) == 1, nk_i8_t_must_be_1_byte);
nk_static_assert_(sizeof(nk_u8_t) == 1, nk_u8_t_must_be_1_byte);
nk_static_assert_(sizeof(nk_i16_t) == 2, nk_i16_t_must_be_2_bytes);
nk_static_assert_(sizeof(nk_u16_t) == 2, nk_u16_t_must_be_2_bytes);
nk_static_assert_(sizeof(nk_i32_t) == 4, nk_i32_t_must_be_4_bytes);
nk_static_assert_(sizeof(nk_u32_t) == 4, nk_u32_t_must_be_4_bytes);
nk_static_assert_(sizeof(nk_i64_t) == 8, nk_i64_t_must_be_8_bytes);
nk_static_assert_(sizeof(nk_u64_t) == 8, nk_u64_t_must_be_8_bytes);
nk_static_assert_(sizeof(nk_f32_t) == 4, nk_f32_t_must_be_4_bytes);
nk_static_assert_(sizeof(nk_f64_t) == 8, nk_f64_t_must_be_8_bytes);
nk_static_assert_(sizeof(nk_f16_t) == 2, nk_f16_t_must_be_2_bytes);
nk_static_assert_(sizeof(nk_bf16_t) == 2, nk_bf16_t_must_be_2_bytes);
nk_static_assert_(sizeof(nk_nvfp4_t) == 9, nk_nvfp4_t_must_be_9_bytes);
nk_static_assert_(sizeof(nk_mxfp4_t) == 17, nk_mxfp4_t_must_be_17_bytes);
nk_static_assert_(sizeof(nk_mxfp6_e2m3_t) == 33, nk_mxfp6_e2m3_t_must_be_33_bytes);
nk_static_assert_(sizeof(nk_mxfp6_e3m2_t) == 33, nk_mxfp6_e3m2_t_must_be_33_bytes);
nk_static_assert_(sizeof(nk_mxfp8_e4m3_t) == 33, nk_mxfp8_e4m3_t_must_be_33_bytes);
nk_static_assert_(sizeof(nk_mxfp8_e5m2_t) == 33, nk_mxfp8_e5m2_t_must_be_33_bytes);
nk_static_assert_(sizeof(nk_mxint8_t) == 33, nk_mxint8_t_must_be_33_bytes);

#define nk_assign_from_to_(src, dest) (*(dest) = *(src))

/** 16-bit union for f16/bf16/u16/i16 bit manipulation. */
typedef union NUMKONG_MAY_ALIAS_ {
    nk_u16_t u;
    nk_i16_t i;
    nk_f16_t f;
    nk_bf16_t bf;
} nk_fui16_t;

/** 32-bit union for f32/u32/i32 bit manipulation. */
typedef union NUMKONG_MAY_ALIAS_ {
    nk_u32_t u;
    nk_i32_t i;
    nk_f32_t f;
} nk_fui32_t;

/** 64-bit union for f64/u64/i64 bit manipulation. */
typedef union NUMKONG_MAY_ALIAS_ {
    nk_u64_t u;
    nk_i64_t i;
    nk_f64_t f;
} nk_fui64_t;

/** Half-precision (32-bit) complex number — {real: f16, imag: f16}. Kernel outputs widened to
 *  f32c. */
typedef struct {
    nk_f16_t real;
    nk_f16_t imag;
} nk_f16c_t;

/** BFloat16 (32-bit) complex number — {real: bf16, imag: bf16}. Kernel outputs widened to f32c. */
typedef struct {
    nk_bf16_t real;
    nk_bf16_t imag;
} nk_bf16c_t;

/** Single-precision (64-bit) complex number — {real: f32, imag: f32}. */
typedef struct {
    nk_f32_t real;
    nk_f32_t imag;
} nk_f32c_t;

/** Double-precision (128-bit) complex number — {real: f64, imag: f64}. */
typedef struct {
    nk_f64_t real;
    nk_f64_t imag;
} nk_f64c_t;

/** Small 4-byte memory slice viewable as different types. */
typedef union NUMKONG_MAY_ALIAS_ nk_b32_vec_t {
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

/** Small 8-byte memory slice viewable as different types. */
typedef union NUMKONG_MAY_ALIAS_ nk_b64_vec_t {
#if NUMKONG_ARCH_ARM64_NEON_
    uint8x8_t u8x8;
    uint16x4_t u16x4;
    uint32x2_t u32x2;
    int8x8_t i8x8;
    int16x4_t i16x4;
    int32x2_t i32x2;
    float32x2_t f32x2;
#endif
#if NUMKONG_ARCH_ARM64_NEON_
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

/** Small 16-byte memory slice viewable as different types. */
typedef union NUMKONG_MAY_ALIAS_ nk_b128_vec_t {
#if NUMKONG_ARCH_X8664_HASWELL_ || defined(__loongarch_asx)
    __m128i xmm;
    __m128d xmm_pd;
    __m128 xmm_ps;
#endif
#if NUMKONG_ARCH_WASM_V128_
    v128_t v128;
#endif
#if NUMKONG_ARCH_ARM64_NEON_
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
#if NUMKONG_ARCH_ARM64_NEON_ && NUMKONG_ARCH_ARM64_ // double-precision NEON requires AArch64
    float64x2_t f64x2;
#endif
#if NUMKONG_ARCH_ARM64_NEON_
    float16x8_t f16x8;
#endif
#if defined(__POWER9_VECTOR__)
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

/** Small 32-byte memory slice viewable as different types. */
typedef union NUMKONG_MAY_ALIAS_ nk_b256_vec_t {
#if NUMKONG_ARCH_X8664_HASWELL_ || defined(__loongarch_asx)
    __m256i ymm;
    __m256d ymm_pd;
    __m256 ymm_ps;
    __m128i xmms[2];
#endif
#if NUMKONG_ARCH_WASM_V128_
    v128_t v128s[2];
#endif
#if NUMKONG_ARCH_ARM64_NEON_
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
#if NUMKONG_ARCH_ARM64_NEON_ && NUMKONG_ARCH_ARM64_ // double-precision NEON requires AArch64
    float64x2_t f64x2s[2];
#endif
#if defined(__POWER9_VECTOR__)
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

/**
 *  @brief Small 64-byte memory slice viewable as different types.
 *
 *  TODO: On GCC and Clang we use @c __transparent_union__ attribute to allow implicit conversions
 *  between the different vector types when passing them as function arguments. The most important
 *  side-effect of this is that the argument of such type is passed to functions using the calling
 *  convention of the first member of the union, which in our case is a register-based calling
 *  convention for SIMD types.
 */
typedef union NUMKONG_MAY_ALIAS_ nk_b512_vec_t {
#if NUMKONG_ARCH_X8664_SKYLAKE_
    __m512i zmm;
    __m512d zmm_pd;
    __m512 zmm_ps;
#endif
#if NUMKONG_ARCH_X8664_HASWELL_
    __m256i ymms[2];
    __m256d ymms_pd[2];
    __m256 ymms_ps[2];
    __m128i xmms[4];
    __m128d xmms_pd[4];
    __m128 xmms_ps[4];
#endif
#if NUMKONG_ARCH_ARM64_NEON_
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
 *  @brief SWAR population count for 64-bit integers.
 *
 *  Classic algorithm from Hacker's Delight using parallel bit summation:
 *  - Count bits in pairs (2-bit sums)
 *  - Count bits in nibbles (4-bit sums)
 *  - Count bits in bytes (8-bit sums)
 *  - Horizontal sum via multiply - each byte contributes to bits 56-63
 *
 *  Cost: ~12 ALU ops, zero memory access (vs 8 table lookups for byte-wise).
 */
NUMKONG_CONSTEXPR nk_u64_t nk_u64_popcount_(nk_u64_t x) {
    x = x - ((x >> 1) & 0x5555555555555555ull);
    x = (x & 0x3333333333333333ull) + ((x >> 2) & 0x3333333333333333ull);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0Full;
    return (x * 0x0101010101010101ull) >> 56;
}

NUMKONG_INLINE unsigned char nk_u1x8_popcount_(nk_u1x8_t x) {
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

/** Divides the number rounding up to the next multiple of the given divisor. */
NUMKONG_CONSTEXPR nk_size_t nk_size_divide_round_up_(nk_size_t number, nk_size_t divisor) NUMKONG_STREAMABLE_ {
    return (number + divisor - 1) / divisor;
}

/** Divides rounding up in 32 bits, for device code where the @c nk_size_t form costs registers. */
NUMKONG_CONSTEXPR nk_u32_t nk_u32_divide_round_up_(nk_u32_t number, nk_u32_t divisor) NUMKONG_STREAMABLE_ {
    return (number + divisor - 1) / divisor;
}

/** Rounds up the number to the next multiple of the given divisor. */
NUMKONG_CONSTEXPR nk_size_t nk_size_round_up_to_multiple_(nk_size_t number, nk_size_t divisor) NUMKONG_STREAMABLE_ {
    return nk_size_divide_round_up_(number, divisor) * divisor;
}

/** The smaller of two values of any comparable type; each argument is evaluated twice. */
#define nk_min_of_two(first, second) ((first) < (second) ? (first) : (second))

/** The larger of two values of any comparable type; each argument is evaluated twice. */
#define nk_max_of_two(first, second) ((first) < (second) ? (second) : (first))

/** Multiplies two sizes with overflow detection. Writes the product and returns 1 on success;
 *  returns 0 (leaving @p product unchanged) when @p a * @p b would overflow @c nk_size_t. */
NUMKONG_CONSTEXPR int nk_size_mul_checked_(nk_size_t a, nk_size_t b, nk_size_t *product) NUMKONG_STREAMABLE_ {
    if (b != 0 && a > NUMKONG_SIZE_MAX / b) return 0;
    *product = a * b;
    return 1;
}

NUMKONG_CONSTEXPR nk_f32_t nk_f32_abs_(nk_f32_t x) { return x < 0 ? -x : x; }
NUMKONG_CONSTEXPR nk_f64_t nk_f64_abs_(nk_f64_t x) { return x < 0 ? -x : x; }
NUMKONG_CONSTEXPR nk_i64_t nk_i32_abs_(nk_i32_t x) { return x < 0 ? -x : x; }
NUMKONG_CONSTEXPR nk_u32_t nk_u32_abs_(nk_u32_t x) { return x; }

/** Extract low (bits 0-3) unsigned nibble from packed u4x2 byte. */
NUMKONG_CONSTEXPR nk_u8_t nk_u4x2_low_(nk_u4x2_t byte_val) { return byte_val & 0x0F; }

/** Extract high (bits 4-7) unsigned nibble from packed u4x2 byte. */
NUMKONG_CONSTEXPR nk_u8_t nk_u4x2_high_(nk_u4x2_t byte_val) { return (byte_val >> 4) & 0x0F; }

/** Extract low (bits 0-3) signed nibble from packed i4x2 byte as i8. */
NUMKONG_CONSTEXPR nk_i8_t nk_i4x2_low_(nk_i4x2_t byte_val) { return (nk_i8_t)(((byte_val & 0x0F) ^ 8) - 8); }

/** Extract high (bits 4-7) signed nibble from packed i4x2 byte as i8. */
NUMKONG_CONSTEXPR nk_i8_t nk_i4x2_high_(nk_i4x2_t byte_val) { return (nk_i8_t)((((byte_val >> 4) & 0x0F) ^ 8) - 8); }

NUMKONG_CONSTEXPR nk_f16_t nk_u16_as_f16_(nk_u16_t bits) {
    nk_fui16_t c;
    c.u = bits;
    return c.f;
}
NUMKONG_CONSTEXPR nk_bf16_t nk_u16_as_bf16_(nk_u16_t bits) {
    nk_fui16_t c;
    c.u = bits;
    return c.bf;
}

NUMKONG_CONSTEXPR void nk_f64_from_i64_(nk_i64_t const *src, nk_f64_t *dest) { *dest = (nk_f64_t)*src; }
NUMKONG_CONSTEXPR void nk_f64_from_u64_(nk_u64_t const *src, nk_f64_t *dest) { *dest = (nk_f64_t)*src; }

/**
 *  @brief Union for type-punned scalar values at language binding boundaries.
 *
 *  Bridges different type systems (Python, JavaScript, etc.) where scalars arrive as f64 but
 *  need to be passed to kernels as typed pointers. Callers fill the appropriate union member
 *  based on the target dtype, then pass the union address as `void const *`.
 */
typedef union NUMKONG_MAY_ALIAS_ nk_scalar_buffer_t {
    nk_u8_t bytes[16];
    nk_f64_t f64;
    nk_f32_t f32;
    nk_f16_t f16;
    nk_bf16_t bf16;
    nk_f64c_t f64c;
    nk_f32c_t f32c;
    nk_f16c_t f16c;
    nk_bf16c_t bf16c;
    nk_i64_t i64;
    nk_u64_t u64;
    nk_i32_t i32;
    nk_u32_t u32;
    nk_i16_t i16;
    nk_u16_t u16;
    nk_i8_t i8;
    nk_u8_t u8;
} nk_scalar_buffer_t;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // NUMKONG_TYPES_H
