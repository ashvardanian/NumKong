/**
 *  @file include/numkong/set.h
 *  @author Ash Vardanian
 *  @date March 23, 2023
 *  @brief SIMD-accelerated set similarity measures.
 *
 *  Contains following similarity measures:
 *
 *  - Bit-level Hamming distance → @c u32 counter
 *  - Byte-level Hamming distance → @c u32 counter
 *  - Bit-level Jaccard distance (Tanimoto coefficient) → @c f32 ratio
 *  - Word-level Jaccard distance for @c u16 and @c u32 MinHash vectors from StringZilla →
 *    @c f32 ratio
 *
 *  For hardware architectures:
 *
 *  - Arm: NEON, SVE
 *  - x86: Haswell, Ice Lake
 *  - RISC-V: RVV, RVV+BB
 *  - WASM: V128
 *
 *  @section set_numerical_stability Numerical Stability
 *
 *  - Hamming u1: u32 popcount accumulator, overflowing at n_bits > 2^32, about 4.3 billion.
 *  - Jaccard u1: u32 intersection and union counts, f32 division at finalization. Popcounts above
 *    2^24 lose precision in the f32 cast, as its mantissa has 24 bits.
 *  - Byte-level Hamming and Jaccard u8: u32 mismatch counter, overflowing at n > 2^32.
 *
 *  @section popcount_strategies Population Count Strategies
 *
 *  Jaccard distances are extremely common and also fairly cheap to compute on binary vectors.
 *  The hardest part of optimizing binary similarity measures is the population count operation.
 *  It's natively supported by almost every instruction set, but the throughput and latency can be
 *  suboptimal. There are several ways to optimize this operation:
 *
 *  - Lookup tables, mostly using nibbles (4-bit lookups)
 *  - Harley-Seal population counts using Carry-Save Adders (CSA)
 *
 *  @section set_x86_instructions Relevant x86 Instructions
 *
 *  On binary vectors, when computing Jaccard distance, the CPU often struggles to compute the large
 *  number of required population counts. There are several instructions we should keep in mind:
 *
 *  @verbatim
 *  Intrinsic                  Instruction                     Icelake    Genoa
 *  _mm512_popcnt_epi64        VPOPCNTQ (ZMM, K, ZMM)          3cy @ p5   2cy @ p01
 *  _mm512_shuffle_epi8        VPSHUFB (ZMM, ZMM, ZMM)         1cy @ p5   2cy @ p12
 *  _mm512_sad_epu8            VPSADBW (ZMM, ZMM, ZMM)         3cy @ p5   3cy @ p01
 *  _mm512_ternarylogic_epi64  VPTERNLOGQ (ZMM, ZMM, ZMM, I8)  1cy @ p05  1cy @ p0123
 *  _mm512_gf2p8mul_epi8       VGF2P8MULB (ZMM, ZMM, ZMM)      5cy @ p0   3cy @ p01
 *  @endverbatim
 *
 *  On Ice Lake, VPOPCNTQ bottlenecks on port 5. On AMD Genoa/Turin, it dual-issues on ports 0-1,
 *  making native popcount significantly faster without CSA tricks.
 *
 *  @section harley_seal Harley-Seal Carry-Save Adders
 *
 *  The Harley-Seal algorithm uses Carry-Save Adders, CSA, to accumulate population counts with
 *  fewer VPOPCNTQ instructions. A CSA computes (a + b + c) as (sum, carry) using only bitwise
 *  operations, deferring expensive popcounts to the final reduction.
 *
 *  Performance varies significantly by architecture and buffer size (cycles/byte):
 *
 *  @verbatim
 *  Method              Buffer      Ice Lake    Sapphire    Genoa
 *  Native VPOPCNTQ     any         ~0.12       ~0.10       ~0.06
 *  Harley-Seal CSA     1 KB        0.107       0.095       0.08
 *  Harley-Seal CSA     4 KB        0.056       0.052       0.05
 *  VPSHUFB lookup      4 KB        0.063       0.058       0.07
 *  @endverbatim
 *
 *  For small buffers (<1KB), loop overhead dominates and unrolled native VPOPCNTQ wins.
 *  Harley-Seal shines on large buffers where CSA chains amortize the setup cost.
 *  On AMD Genoa, native VPOPCNTQ is competitive even for large buffers.
 *
 *  @section jaccard_norms Jaccard Optimization via Norms
 *
 *  There is a trivial optimization to halve the number of population counts needed for binary
 *  Jaccard distance, if one knows the set magnitudes ahead of time:
 *
 *      J = |A ∩ B| / |A ∪ B| = |A ∩ B| / (|A| + |B| - |A ∩ B|)
 *
 *  At that point the problem reduces to optimizing memory accesses and register usage. The packed
 *  kernels in sets.h take that route, finishing each tile with @c nk_jaccard_f32x4_from_dot_*.
 *
 *  @section tail_handling Tail Handling
 *
 *  The trickiest part is handling the tails of the vectors when their size isn't divisible by our
 *  step size. In such cases, it's recommended to use masked loads when supported by the ISA, or
 *  fall back to scalar code and a local on-stack buffer.
 *
 *  @section set_references References
 *
 *  @see Intel Intrinsics Guide: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
 *  @see Arm Intrinsics Reference: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *  @see Muła et al. "Faster Population Counts": https://arxiv.org/pdf/1611.07612
 *  @see Muła SSE POPCOUNT experiments: https://github.com/WojciechMula/sse-popcount
 *  @see NumKong binary R&D tracker: https://github.com/ashvardanian/NumKong/pull/138
 *
 *  @section set_output_types Output Types
 *
 *  Jaccard distances are output as f32:
 *  - Jaccard = intersection / union, always ∈ [0.0, 1.0]
 *  - f32 provides ~7 decimal digits, far exceeding practical needs
 *  - Matches spatial.h convention for non-f64 distance outputs
 *  - Reduces memory footprint in large-scale binary similarity search
 *
 *  The intersection and union counts stay integral, and only their final ratio is rounded to f32.
 *
 */
#ifndef NUMKONG_SET_H
#define NUMKONG_SET_H

#include "numkong/capabilities.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Binary Hamming distance computing the number of differing bits between
 *      two binary vectors.
 *
 *  @param[in] a The first binary vector.
 *  @param[in] b The second binary vector.
 *  @param[in] n Counts dimensions, a multiple of the values per byte.
 *  @param[out] result The output distance value.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  @note The output distance value is non-negative.
 *  @note The output distance value is zero if and only if the two vectors are identical.
 */
NUMKONG_API nk_status_t nk_hamming_u1_best(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result,
                                           nk_capability_t capabilities, void *stream);

/**
 *  @brief Binary Jaccard distance computing the ratio of differing bits to the union of bits.
 *
 *  @param[in] a The first binary vector.
 *  @param[in] b The second binary vector.
 *  @param[in] n Counts dimensions, a multiple of the values per byte.
 *  @param[out] result The output distance value.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  @note The output distance value is non-negative.
 *  @note The output distance value is zero if and only if the two vectors are identical.
 */
NUMKONG_API nk_status_t nk_jaccard_u1_best(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result,
                                           nk_capability_t capabilities, void *stream);

/**
 *  @brief Integral Jaccard distance computing the ratio of differing bits to the union of bits.
 *
 *  @param[in] a The first binary vector.
 *  @param[in] b The second binary vector.
 *  @param[in] n The number of 32-bit scalars in the vectors.
 *  @param[out] result The output distance value.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  @note The output distance value is non-negative.
 *  @note The output distance value is zero if and only if the two vectors are identical.
 */
NUMKONG_API nk_status_t nk_jaccard_u32_best(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_f32_t *result,
                                            nk_capability_t capabilities, void *stream);

/**
 *  @brief Byte-level Hamming distance computing the number of differing bytes between two vectors.
 *
 *  @param[in] a The first byte vector.
 *  @param[in] b The second byte vector.
 *  @param[in] n The number of bytes in the vectors.
 *  @param[out] result The output distance value.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  @note The output distance value is non-negative.
 *  @note The output distance value is zero if and only if the two vectors are identical.
 */
NUMKONG_API nk_status_t nk_hamming_u8_best(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                           nk_capability_t capabilities, void *stream);

/**
 *  @brief Integral Jaccard distance for 16-bit unsigned integer vectors.
 *
 *  @param[in] a The first vector.
 *  @param[in] b The second vector.
 *  @param[in] n The number of 16-bit scalars in the vectors.
 *  @param[out] result The output distance value.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  @note The output distance value is non-negative.
 *  @note The output distance value is zero if and only if the two vectors are identical.
 */
NUMKONG_API nk_status_t nk_jaccard_u16_best(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_f32_t *result,
                                            nk_capability_t capabilities, void *stream);

/** @copydoc nk_hamming_u1_best */
NUMKONG_API nk_status_t nk_hamming_u1_serial(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result,
                                             void *stream);
/** @copydoc nk_hamming_u8_best */
NUMKONG_API nk_status_t nk_hamming_u8_serial(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                             void *stream);
/** @copydoc nk_jaccard_u1_best */
NUMKONG_API nk_status_t nk_jaccard_u1_serial(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result,
                                             void *stream);
/** @copydoc nk_jaccard_u32_best */
NUMKONG_API nk_status_t nk_jaccard_u32_serial(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_f32_t *result,
                                              void *stream);
/** @copydoc nk_jaccard_u16_best */
NUMKONG_API nk_status_t nk_jaccard_u16_serial(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_f32_t *result,
                                              void *stream);

#if NUMKONG_TARGET_NEON
/** @copydoc nk_hamming_u1_best */
NUMKONG_API nk_status_t nk_hamming_u1_neon(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result,
                                           void *stream);
/** @copydoc nk_hamming_u8_best */
NUMKONG_API nk_status_t nk_hamming_u8_neon(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                           void *stream);
/** @copydoc nk_jaccard_u1_best */
NUMKONG_API nk_status_t nk_jaccard_u1_neon(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result,
                                           void *stream);
/** @copydoc nk_jaccard_u32_best */
NUMKONG_API nk_status_t nk_jaccard_u32_neon(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_f32_t *result,
                                            void *stream);
/** @copydoc nk_jaccard_u16_best */
NUMKONG_API nk_status_t nk_jaccard_u16_neon(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_f32_t *result,
                                            void *stream);
#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_SVE
/** @copydoc nk_hamming_u1_best */
NUMKONG_API nk_status_t nk_hamming_u1_sve(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result,
                                          void *stream);
/** @copydoc nk_hamming_u8_best */
NUMKONG_API nk_status_t nk_hamming_u8_sve(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                          void *stream);
/** @copydoc nk_jaccard_u1_best */
NUMKONG_API nk_status_t nk_jaccard_u1_sve(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result,
                                          void *stream);
/** @copydoc nk_jaccard_u32_best */
NUMKONG_API nk_status_t nk_jaccard_u32_sve(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_f32_t *result,
                                           void *stream);
/** @copydoc nk_jaccard_u16_best */
NUMKONG_API nk_status_t nk_jaccard_u16_sve(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_f32_t *result,
                                           void *stream);
#endif // NUMKONG_TARGET_SVE

#if NUMKONG_TARGET_HASWELL
/** @copydoc nk_hamming_u1_best */
NUMKONG_API nk_status_t nk_hamming_u1_haswell(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result,
                                              void *stream);
/** @copydoc nk_hamming_u8_best */
NUMKONG_API nk_status_t nk_hamming_u8_haswell(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                              void *stream);
/** @copydoc nk_jaccard_u1_best */
NUMKONG_API nk_status_t nk_jaccard_u1_haswell(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result,
                                              void *stream);
/** @copydoc nk_jaccard_u16_best */
NUMKONG_API nk_status_t nk_jaccard_u16_haswell(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_f32_t *result,
                                               void *stream);
/** @copydoc nk_jaccard_u32_best */
NUMKONG_API nk_status_t nk_jaccard_u32_haswell(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_f32_t *result,
                                               void *stream);
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_ICELAKE
/** @copydoc nk_hamming_u1_best */
NUMKONG_API nk_status_t nk_hamming_u1_icelake(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result,
                                              void *stream);
/** @copydoc nk_hamming_u8_best */
NUMKONG_API nk_status_t nk_hamming_u8_icelake(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                              void *stream);
/** @copydoc nk_jaccard_u1_best */
NUMKONG_API nk_status_t nk_jaccard_u1_icelake(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result,
                                              void *stream);
/** @copydoc nk_jaccard_u32_best */
NUMKONG_API nk_status_t nk_jaccard_u32_icelake(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_f32_t *result,
                                               void *stream);
/** @copydoc nk_jaccard_u16_best */
NUMKONG_API nk_status_t nk_jaccard_u16_icelake(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_f32_t *result,
                                               void *stream);
#endif // NUMKONG_TARGET_ICELAKE

#if NUMKONG_TARGET_RVVBB
/** @copydoc nk_hamming_u1_best */
NUMKONG_API nk_status_t nk_hamming_u1_rvvbb(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result,
                                            void *stream);
/** @copydoc nk_jaccard_u1_best */
NUMKONG_API nk_status_t nk_jaccard_u1_rvvbb(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result,
                                            void *stream);
#endif // NUMKONG_TARGET_RVVBB

#if NUMKONG_TARGET_RVV
/** @copydoc nk_hamming_u1_best */
NUMKONG_API nk_status_t nk_hamming_u1_rvv(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result,
                                          void *stream);
/** @copydoc nk_hamming_u8_best */
NUMKONG_API nk_status_t nk_hamming_u8_rvv(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                          void *stream);
/** @copydoc nk_jaccard_u1_best */
NUMKONG_API nk_status_t nk_jaccard_u1_rvv(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result,
                                          void *stream);
/** @copydoc nk_jaccard_u16_best */
NUMKONG_API nk_status_t nk_jaccard_u16_rvv(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_f32_t *result,
                                           void *stream);
/** @copydoc nk_jaccard_u32_best */
NUMKONG_API nk_status_t nk_jaccard_u32_rvv(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_f32_t *result,
                                           void *stream);
#endif // NUMKONG_TARGET_RVV

#if NUMKONG_TARGET_V128
/** @copydoc nk_hamming_u1_best */
NUMKONG_API nk_status_t nk_hamming_u1_v128(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result,
                                           void *stream);
/** @copydoc nk_hamming_u8_best */
NUMKONG_API nk_status_t nk_hamming_u8_v128(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                           void *stream);
/** @copydoc nk_jaccard_u1_best */
NUMKONG_API nk_status_t nk_jaccard_u1_v128(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result,
                                           void *stream);
/** @copydoc nk_jaccard_u16_best */
NUMKONG_API nk_status_t nk_jaccard_u16_v128(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_f32_t *result,
                                            void *stream);
/** @copydoc nk_jaccard_u32_best */
NUMKONG_API nk_status_t nk_jaccard_u32_v128(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_f32_t *result,
                                            void *stream);
#endif // NUMKONG_TARGET_V128

#if NUMKONG_TARGET_POWERVSX
/** @copydoc nk_hamming_u1_best */
NUMKONG_API nk_status_t nk_hamming_u1_powervsx(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result,
                                               void *stream);
/** @copydoc nk_hamming_u8_best */
NUMKONG_API nk_status_t nk_hamming_u8_powervsx(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                               void *stream);
/** @copydoc nk_jaccard_u1_best */
NUMKONG_API nk_status_t nk_jaccard_u1_powervsx(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result,
                                               void *stream);
#endif // NUMKONG_TARGET_POWERVSX

#if NUMKONG_TARGET_LOONGSONASX
/** @copydoc nk_hamming_u1_best */
NUMKONG_API nk_status_t nk_hamming_u1_loongsonasx(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result,
                                                  void *stream);
/** @copydoc nk_hamming_u8_best */
NUMKONG_API nk_status_t nk_hamming_u8_loongsonasx(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                                  void *stream);
/** @copydoc nk_jaccard_u1_best */
NUMKONG_API nk_status_t nk_jaccard_u1_loongsonasx(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result,
                                                  void *stream);
#endif // NUMKONG_TARGET_LOONGSONASX

/** Returns the output dtype for Hamming distance. */
NUMKONG_INLINE nk_dtype_t nk_hamming_output_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_u1_k: return nk_u32_k;
    case nk_u8_k: return nk_u32_k;
    default: return nk_dtype_unknown_k;
    }
}

/** Returns the output dtype for Jaccard distance. */
NUMKONG_INLINE nk_dtype_t nk_jaccard_output_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_u1_k: return nk_f32_k;
    case nk_u16_k: return nk_f32_k;
    case nk_u32_k: return nk_f32_k;
    default: return nk_dtype_unknown_k;
    }
}

/**
 *  @brief Finds the set distance kernel of @p kind for @p dtype, from the best of @p capabilities.
 *  @param[out] kernel The kernel, or null when none of @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *  @return @c nk_success_k, @c nk_missing_kernel_k, or @c nk_missing_library_k in header-only builds.
 */
NUMKONG_API nk_status_t nk_set_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                           nk_kernel_punned_t *kernel, nk_capability_t *capability);

#if defined(__cplusplus)
} // extern "C"
#endif

#if NUMKONG_HEADER_ONLY
#include "numkong/set/serial.h"
#include "numkong/set/neon.h"
#include "numkong/set/sve.h"
#include "numkong/set/icelake.h"
#include "numkong/set/haswell.h"
#include "numkong/set/powervsx.h"
#include "numkong/set/v128.h"
#include "numkong/set/rvv.h"
#include "numkong/set/rvvbb.h"
#include "numkong/set/loongsonasx.h"

#if defined(__cplusplus)
extern "C" {
#endif

NUMKONG_API nk_status_t nk_jaccard_u32_best(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_f32_t *result,
                                            nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_jaccard_u16_best(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_f32_t *result,
                                            nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_hamming_u8_best(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result,
                                           nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_hamming_u1_best(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result,
                                           nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_jaccard_u1_best(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result,
                                           nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_set_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                           nk_kernel_punned_t *kernel, nk_capability_t *capability) {
    nk_unused_(kind), nk_unused_(dtype), nk_unused_(capabilities);
    *kernel = NUMKONG_NULL, *capability = 0;
    return nk_missing_library_k;
}

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // NUMKONG_HEADER_ONLY

#endif
