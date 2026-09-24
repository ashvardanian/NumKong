/**
 *  @brief SIMD-accelerated Set Similarity Measures.
 *  @file include/numkong/set.h
 *  @author Ash Vardanian
 *  @date July 1, 2023
 *
 *  Contains following similarity measures:
 *
 *  - Bit-level Hamming distance → `u32` counter
 *  - Byte-level Hamming distance → `u32` counter
 *  - Bit-level Jaccard distance (Tanimoto coefficient) → `f32` ratio
 *  - Word-level Jaccard distance for `u16` and `u32` MinHash vectors from StringZilla → `f32` ratio
 *
 *  For hardware architectures:
 *
 *  - Arm: NEON, SVE
 *  - x86: Haswell, Ice Lake
 *  - RISC-V: RVV, RVV+BB
 *  - WASM: V128Relaxed
 *
 *  @section numerical_stability Numerical Stability
 *
 *  Hamming u1: u32 popcount accumulator. Overflows at n_bits > 2^32 (~4.3 billion).
 *  The streaming u1x512 variant uses u64, safe for any practical dimension.
 *  Jaccard u1: u32 intersection/union counts, f32 division at finalization.
 *  Popcount values above 2^24 lose precision in f32 cast (24-bit mantissa).
 *  Byte-level Hamming/Jaccard u8: u32 mismatch counter. Overflows at n > 2^32.
 *
 *  @section popcount_strategies Population Count Strategies
 *
 *  Jaccard distances are extremely common and also fairly cheap to compute on binary vectors.
 *  The hardest part of optimizing binary similarity measures is the population count operation.
 *  It's natively supported by almost every instruction set, but the throughput and latency can
 *  be suboptimal. There are several ways to optimize this operation:
 *
 *  - Lookup tables, mostly using nibbles (4-bit lookups)
 *  - Harley-Seal population counts using Carry-Save Adders (CSA)
 *
 *  @section x86_instructions Relevant x86 Instructions
 *
 *  On binary vectors, when computing Jaccard distance, the CPU often struggles to compute the
 *  large number of required population counts. There are several instructions we should keep in mind:
 *
 *      Intrinsic                   Instruction                     Ice         Genoa
 *      _mm512_popcnt_epi64         VPOPCNTQ (ZMM, K, ZMM)          3cy @ p5     2cy @ p01
 *      _mm512_shuffle_epi8         VPSHUFB (ZMM, ZMM, ZMM)         1cy @ p5     2cy @ p12
 *      _mm512_sad_epu8             VPSADBW (ZMM, ZMM, ZMM)         3cy @ p5     3cy @ p01
 *      _mm512_ternarylogic_epi64   VPTERNLOGQ (ZMM, ZMM, ZMM, I8)  1cy @ p05    1cy @ p0123
 *      _mm512_gf2p8mul_epi8        VGF2P8MULB (ZMM, ZMM, ZMM)      5cy @ p0     3cy @ p01
 *
 *  On Ice Lake, VPOPCNTQ bottlenecks on port 5. On AMD Genoa/Turin, it dual-issues
 *  on ports 0-1, making native popcount significantly faster without CSA tricks.
 *
 *  @section harley_seal Harley-Seal Carry-Save Adders
 *
 *  The Harley-Seal algorithm uses Carry-Save Adders (CSA) to accumulate population counts
 *  with fewer VPOPCNTQ instructions. A CSA computes (a + b + c) as (sum, carry) using only
 *  bitwise operations, deferring expensive popcounts to the final reduction.
 *
 *  Performance varies significantly by architecture and buffer size (cycles/byte):
 *
 *      Method              Buffer      Ice Lake    Sapphire    Genoa
 *      Native VPOPCNTQ     any         ~0.12       ~0.10       ~0.06
 *      Harley-Seal CSA     1 KB        0.107       0.095       0.08
 *      Harley-Seal CSA     4 KB        0.056       0.052       0.05
 *      VPSHUFB lookup      4 KB        0.063       0.058       0.07
 *
 *  For small buffers (<1KB), loop overhead dominates and unrolled native VPOPCNTQ wins.
 *  Harley-Seal shines on large buffers where CSA chains amortize the setup cost.
 *  On AMD Genoa, native VPOPCNTQ is competitive even for large buffers.
 *
 *  @section jaccard_norms Jaccard Optimization via Norms
 *
 *  There is a trivial optimization to halve the number of population counts needed for
 *  binary Jaccard distance, if one knows the set magnitudes ahead of time:
 *
 *      J = |A ∩ B| / |A ∪ B| = |A ∩ B| / (|A| + |B| - |A ∩ B|)
 *
 *  At that point the problem reduces to optimizing memory accesses and register usage.
 *  For such cases, we provide additional function variants designed exclusively for compile-time
 *  dispatch in heavily inlined code, operating on wider vectors with known sizes:
 *
 *  - nk_jaccard_u1x512_state_<isa>_t - Smallest optimal running state
 *  - nk_jaccard_u1x512_init_<isa> - Initializes the running state
 *  - nk_jaccard_u1x512_update_<isa> - Updates the running state with 2 new 512-bit vectors
 *  - nk_jaccard_u1x512_finalize_<isa> - Finalizes the running state and produces the distance
 *
 *  @section streaming_api Streaming API
 *
 *  The streaming variants aren't always strictly equivalent to their counterparts above
 *  and their usage also differs quite drastically. For large-scale batch processing where
 *  vectors won't be reused, consider non-temporal loads (`_mm512_stream_load_si512`) to
 *  bypass the cache and avoid pollution. This is especially beneficial when computing
 *  distances across millions of vectors in a single pass.
 *
 *  @code{.c}
 *  // 1024-dimensional binary vectors, one query and four targets
 *  nk_u1x8_t query[128], target_first[128], target_second[128], target_third[128], target_fourth[128];
 *  // Precomputed popcount of 'a' as f32
 *  nk_f32_t query_popcount = ...;
 *  nk_f32_t target_popcount_first = ..., target_popcount_second = ...;
 *
 *  nk_jaccard_u1x512_state_icelake_t state_first, state_second, state_third, state_fourth;
 *  nk_jaccard_u1x512_init_icelake(&state_first);
 *  nk_jaccard_u1x512_init_icelake(&state_second);
 *  nk_jaccard_u1x512_init_icelake(&state_third);
 *  nk_jaccard_u1x512_init_icelake(&state_fourth);
 *  nk_jaccard_u1x512_update_icelake(&state_first, &query[0], &target_first[0], 0, 512); // First 512 bits
 *  nk_jaccard_u1x512_update_icelake(&state_first, &query[64], &target_first[64], 512, 512); // Second 512 bits
 *  // ... update state_second, state_third, state_fourth similarly ...
 *
 *  nk_f32_t results[4];
 *  nk_jaccard_u1x512_finalize_icelake(&state_first, &state_second, &state_third, &state_fourth,
 *      query_popcount, target_popcount_first, target_popcount_second,
 *      target_popcount_third, target_popcount_fourth, total_dimensions, results);
 *  @endcode
 *
 *  @section tail_handling Tail Handling
 *
 *  The trickiest part is handling the tails of the vectors when their size isn't divisible
 *  by our step size. In such cases, it's recommended to use masked loads when supported by
 *  the ISA, or fall back to scalar code and a local on-stack buffer.
 *
 *  @section references References
 *
 *  - Intel Intrinsics Guide: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
 *  - Arm Intrinsics Reference: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *  - Muła et al. "Faster Population Counts": https://arxiv.org/pdf/1611.07612
 *  - Muła SSE POPCOUNT experiments: https://github.com/WojciechMula/sse-popcount
 *  - NumKong binary R&D tracker: https://github.com/ashvardanian/NumKong/pull/138
 *
 *  @section Finalize Output Types
 *
 *  Jaccard similarity finalize outputs to f32:
 *  - Jaccard = intersection / union, always ∈ [0.0, 1.0]
 *  - f32 provides ~7 decimal digits, far exceeding practical needs
 *  - Matches spatial.h convention for non-f64 distance outputs
 *  - Reduces memory footprint in large-scale binary similarity search
 *
 *  The intersection and union counts are u64 internally for correctness,
 *  but the final ratio fits comfortably in f32.
 *
 */
#ifndef NK_SET_H
#define NK_SET_H

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Binary Hamming distance computing the number of differing bits between two binary vectors.
 *
 *  @param[in] a The first binary vector.
 *  @param[in] b The second binary vector.
 *  @param[in] n The number of bits in the vectors.
 *  @param[out] result The output distance value.
 *
 *  @note The output distance value is non-negative.
 *  @note The output distance value is zero if and only if the two vectors are identical.
 */
NK_DYNAMIC void nk_hamming_u1(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result);

/**
 *  @brief Binary Jaccard distance computing the ratio of differing bits to the union of bits.
 *
 *  @param[in] a The first binary vector.
 *  @param[in] b The second binary vector.
 *  @param[in] n The number of bits in the vectors.
 *  @param[out] result The output distance value.
 *
 *  @note The output distance value is non-negative.
 *  @note The output distance value is zero if and only if the two vectors are identical.
 */
NK_DYNAMIC void nk_jaccard_u1(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result);

/**
 *  @brief Integral Jaccard distance computing the ratio of differing bits to the union of bits.
 *
 *  @param[in] a The first binary vector.
 *  @param[in] b The second binary vector.
 *  @param[in] n The number of 32-bit scalars in the vectors.
 *  @param[out] result The output distance value.
 *
 *  @note The output distance value is non-negative.
 *  @note The output distance value is zero if and only if the two vectors are identical.
 */
NK_DYNAMIC void nk_jaccard_u32(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_f32_t *result);

/**
 *  @brief Byte-level Hamming distance computing the number of differing bytes between two vectors.
 *
 *  @param[in] a The first byte vector.
 *  @param[in] b The second byte vector.
 *  @param[in] n The number of bytes in the vectors.
 *  @param[out] result The output distance value.
 *
 *  @note The output distance value is non-negative.
 *  @note The output distance value is zero if and only if the two vectors are identical.
 */
NK_DYNAMIC void nk_hamming_u8(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result);

/**
 *  @brief Integral Jaccard distance for 16-bit unsigned integer vectors.
 *
 *  @param[in] a The first vector.
 *  @param[in] b The second vector.
 *  @param[in] n The number of 16-bit scalars in the vectors.
 *  @param[out] result The output distance value.
 *
 *  @note The output distance value is non-negative.
 *  @note The output distance value is zero if and only if the two vectors are identical.
 */
NK_DYNAMIC void nk_jaccard_u16(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_f32_t *result);

/** @copydoc nk_hamming_u1 */
NK_PUBLIC void nk_hamming_u1_serial(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_hamming_u8 */
NK_PUBLIC void nk_hamming_u8_serial(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_jaccard_u1 */
NK_PUBLIC void nk_jaccard_u1_serial(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_jaccard_u32 */
NK_PUBLIC void nk_jaccard_u32_serial(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_jaccard_u16 */
NK_PUBLIC void nk_jaccard_u16_serial(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_f32_t *result);

#if NK_TARGET_NEON
/** @copydoc nk_hamming_u1 */
NK_PUBLIC void nk_hamming_u1_neon(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_hamming_u8 */
NK_PUBLIC void nk_hamming_u8_neon(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_jaccard_u1 */
NK_PUBLIC void nk_jaccard_u1_neon(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_jaccard_u32 */
NK_PUBLIC void nk_jaccard_u32_neon(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_jaccard_u16 */
NK_PUBLIC void nk_jaccard_u16_neon(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_f32_t *result);

#endif // NK_TARGET_NEON

#if NK_TARGET_SVE
/** @copydoc nk_hamming_u1 */
NK_PUBLIC void nk_hamming_u1_sve(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_hamming_u8 */
NK_PUBLIC void nk_hamming_u8_sve(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_jaccard_u1 */
NK_PUBLIC void nk_jaccard_u1_sve(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_jaccard_u32 */
NK_PUBLIC void nk_jaccard_u32_sve(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_jaccard_u16 */
NK_PUBLIC void nk_jaccard_u16_sve(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_f32_t *result);
#endif // NK_TARGET_SVE

#if NK_TARGET_HASWELL
/** @copydoc nk_hamming_u1 */
NK_PUBLIC void nk_hamming_u1_haswell(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_hamming_u8 */
NK_PUBLIC void nk_hamming_u8_haswell(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_jaccard_u1 */
NK_PUBLIC void nk_jaccard_u1_haswell(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_jaccard_u16 */
NK_PUBLIC void nk_jaccard_u16_haswell(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_jaccard_u32 */
NK_PUBLIC void nk_jaccard_u32_haswell(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_f32_t *result);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_ICELAKE
/** @copydoc nk_hamming_u1 */
NK_PUBLIC void nk_hamming_u1_icelake(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_hamming_u8 */
NK_PUBLIC void nk_hamming_u8_icelake(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_jaccard_u1 */
NK_PUBLIC void nk_jaccard_u1_icelake(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_jaccard_u32 */
NK_PUBLIC void nk_jaccard_u32_icelake(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_jaccard_u16 */
NK_PUBLIC void nk_jaccard_u16_icelake(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_f32_t *result);
#endif // NK_TARGET_ICELAKE

#if NK_TARGET_RVVBB
/** @copydoc nk_hamming_u1 */
NK_PUBLIC void nk_hamming_u1_rvvbb(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_jaccard_u1 */
NK_PUBLIC void nk_jaccard_u1_rvvbb(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result);
#endif // NK_TARGET_RVVBB

#if NK_TARGET_RVV
/** @copydoc nk_hamming_u1 */
NK_PUBLIC void nk_hamming_u1_rvv(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_hamming_u8 */
NK_PUBLIC void nk_hamming_u8_rvv(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_jaccard_u1 */
NK_PUBLIC void nk_jaccard_u1_rvv(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_jaccard_u16 */
NK_PUBLIC void nk_jaccard_u16_rvv(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_jaccard_u32 */
NK_PUBLIC void nk_jaccard_u32_rvv(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_f32_t *result);
#endif // NK_TARGET_RVV

#if NK_TARGET_V128RELAXED
/** @copydoc nk_hamming_u1 */
NK_PUBLIC void nk_hamming_u1_v128relaxed(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_hamming_u8 */
NK_PUBLIC void nk_hamming_u8_v128relaxed(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result);
/** @copydoc nk_jaccard_u1 */
NK_PUBLIC void nk_jaccard_u1_v128relaxed(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_jaccard_u16 */
NK_PUBLIC void nk_jaccard_u16_v128relaxed(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_jaccard_u32 */
NK_PUBLIC void nk_jaccard_u32_v128relaxed(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_f32_t *result);
#endif // NK_TARGET_V128RELAXED

/**
 *  @brief  Returns the output dtype for Hamming distance.
 */
NK_INTERNAL nk_dtype_t nk_hamming_output_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_u1_k: return nk_u32_k;
    case nk_u8_k: return nk_u32_k;
    default: return nk_dtype_unknown_k;
    }
}

/**
 *  @brief  Returns the output dtype for Jaccard distance.
 */
NK_INTERNAL nk_dtype_t nk_jaccard_output_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_u1_k: return nk_f32_k;
    case nk_u16_k: return nk_f32_k;
    case nk_u32_k: return nk_f32_k;
    default: return nk_dtype_unknown_k;
    }
}

#if defined(__cplusplus)
} // extern "C"
#endif

#include "numkong/set/serial.h"
#include "numkong/set/neon.h"
#include "numkong/set/sve.h"
#include "numkong/set/icelake.h"
#include "numkong/set/haswell.h"
#include "numkong/set/v128relaxed.h"
#include "numkong/set/rvv.h"
#include "numkong/set/rvvbb.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if !NK_DYNAMIC_DISPATCH

NK_PUBLIC void nk_hamming_u1(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_u32_t *result) {
#if NK_TARGET_V128RELAXED
    nk_hamming_u1_v128relaxed(a, b, n, result);
#elif NK_TARGET_SVE
    nk_hamming_u1_sve(a, b, n, result);
#elif NK_TARGET_NEON
    nk_hamming_u1_neon(a, b, n, result);
#elif NK_TARGET_ICELAKE
    nk_hamming_u1_icelake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_hamming_u1_haswell(a, b, n, result);
#elif NK_TARGET_RVVBB
    nk_hamming_u1_rvvbb(a, b, n, result);
#elif NK_TARGET_RVV
    nk_hamming_u1_rvv(a, b, n, result);
#else
    nk_hamming_u1_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_jaccard_u1(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_V128RELAXED
    nk_jaccard_u1_v128relaxed(a, b, n, result);
#elif NK_TARGET_SVE
    nk_jaccard_u1_sve(a, b, n, result);
#elif NK_TARGET_NEON
    nk_jaccard_u1_neon(a, b, n, result);
#elif NK_TARGET_ICELAKE
    nk_jaccard_u1_icelake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_jaccard_u1_haswell(a, b, n, result);
#elif NK_TARGET_RVVBB
    nk_jaccard_u1_rvvbb(a, b, n, result);
#elif NK_TARGET_RVV
    nk_jaccard_u1_rvv(a, b, n, result);
#else
    nk_jaccard_u1_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_jaccard_u32(nk_u32_t const *a, nk_u32_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_V128RELAXED
    nk_jaccard_u32_v128relaxed(a, b, n, result);
#elif NK_TARGET_SVE
    nk_jaccard_u32_sve(a, b, n, result);
#elif NK_TARGET_NEON
    nk_jaccard_u32_neon(a, b, n, result);
#elif NK_TARGET_ICELAKE
    nk_jaccard_u32_icelake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_jaccard_u32_haswell(a, b, n, result);
#elif NK_TARGET_RVV
    nk_jaccard_u32_rvv(a, b, n, result);
#else
    nk_jaccard_u32_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_hamming_u8(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result) {
#if NK_TARGET_V128RELAXED
    nk_hamming_u8_v128relaxed(a, b, n, result);
#elif NK_TARGET_SVE
    nk_hamming_u8_sve(a, b, n, result);
#elif NK_TARGET_NEON
    nk_hamming_u8_neon(a, b, n, result);
#elif NK_TARGET_ICELAKE
    nk_hamming_u8_icelake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_hamming_u8_haswell(a, b, n, result);
#elif NK_TARGET_RVV
    nk_hamming_u8_rvv(a, b, n, result);
#else
    nk_hamming_u8_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_jaccard_u16(nk_u16_t const *a, nk_u16_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_V128RELAXED
    nk_jaccard_u16_v128relaxed(a, b, n, result);
#elif NK_TARGET_SVE
    nk_jaccard_u16_sve(a, b, n, result);
#elif NK_TARGET_NEON
    nk_jaccard_u16_neon(a, b, n, result);
#elif NK_TARGET_ICELAKE
    nk_jaccard_u16_icelake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_jaccard_u16_haswell(a, b, n, result);
#elif NK_TARGET_RVV
    nk_jaccard_u16_rvv(a, b, n, result);
#else
    nk_jaccard_u16_serial(a, b, n, result);
#endif
}

#endif // !NK_DYNAMIC_DISPATCH

#if defined(__cplusplus)
} // extern "C"
#endif

#endif
