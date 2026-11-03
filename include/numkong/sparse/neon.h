/**
 *  @file include/numkong/sparse/neon.h
 *  @author Ash Vardanian
 *  @date August 19, 2024
 *  @brief NEON-accelerated sparse vector operations.
 *
 *  @sa c/numkong/sparse.h
 */
#ifndef NUMKONG_SPARSE_NEON_H
#define NUMKONG_SPARSE_NEON_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_ARCH_ARM64_NEON_

#include "numkong/types.h"
#include "numkong/cast/neon.h"     // `nk_bf16x4_to_f32x4_neon_`
#include "numkong/sparse/serial.h" // `nk_sparse_intersect_u16_serial_`, `nk_sparse_intersect_u32_serial_`, `nk_sparse_intersect_u64_serial_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#endif

NUMKONG_INLINE uint32x4_t nk_intersect_u32x4_neon_(uint32x4_t a, uint32x4_t b) {
    uint32x4_t b_rot1_u32x4 = vextq_u32(b, b, 1);
    uint32x4_t b_rot2_u32x4 = vextq_u32(b, b, 2);
    uint32x4_t b_rot3_u32x4 = vextq_u32(b, b, 3);
    uint32x4_t matches_rot0_u32x4 = vceqq_u32(a, b);
    uint32x4_t matches_rot1_u32x4 = vceqq_u32(a, b_rot1_u32x4);
    uint32x4_t matches_rot2_u32x4 = vceqq_u32(a, b_rot2_u32x4);
    uint32x4_t matches_rot3_u32x4 = vceqq_u32(a, b_rot3_u32x4);
    uint32x4_t matches_u32x4 = vorrq_u32(vorrq_u32(matches_rot0_u32x4, matches_rot1_u32x4),
                                         vorrq_u32(matches_rot2_u32x4, matches_rot3_u32x4));
    return matches_u32x4;
}

NUMKONG_INLINE uint16x8_t nk_intersect_u16x8_neon_(uint16x8_t a, uint16x8_t b) {
    uint16x8_t b_rot1_u16x8 = vextq_u16(b, b, 1);
    uint16x8_t b_rot2_u16x8 = vextq_u16(b, b, 2);
    uint16x8_t b_rot3_u16x8 = vextq_u16(b, b, 3);
    uint16x8_t b_rot4_u16x8 = vextq_u16(b, b, 4);
    uint16x8_t b_rot5_u16x8 = vextq_u16(b, b, 5);
    uint16x8_t b_rot6_u16x8 = vextq_u16(b, b, 6);
    uint16x8_t b_rot7_u16x8 = vextq_u16(b, b, 7);
    uint16x8_t matches_rot0_u16x8 = vceqq_u16(a, b);
    uint16x8_t matches_rot1_u16x8 = vceqq_u16(a, b_rot1_u16x8);
    uint16x8_t matches_rot2_u16x8 = vceqq_u16(a, b_rot2_u16x8);
    uint16x8_t matches_rot3_u16x8 = vceqq_u16(a, b_rot3_u16x8);
    uint16x8_t matches_rot4_u16x8 = vceqq_u16(a, b_rot4_u16x8);
    uint16x8_t matches_rot5_u16x8 = vceqq_u16(a, b_rot5_u16x8);
    uint16x8_t matches_rot6_u16x8 = vceqq_u16(a, b_rot6_u16x8);
    uint16x8_t matches_rot7_u16x8 = vceqq_u16(a, b_rot7_u16x8);
    uint16x8_t matches_u16x8 = vorrq_u16(
        vorrq_u16(vorrq_u16(matches_rot0_u16x8, matches_rot1_u16x8), vorrq_u16(matches_rot2_u16x8, matches_rot3_u16x8)),
        vorrq_u16(vorrq_u16(matches_rot4_u16x8, matches_rot5_u16x8),
                  vorrq_u16(matches_rot6_u16x8, matches_rot7_u16x8)));
    return matches_u16x8;
}

NUMKONG_INLINE uint64x2_t nk_intersect_u64x2_neon_(uint64x2_t a, uint64x2_t b) {
    uint64x2_t b_rot1_u64x2 = vextq_u64(b, b, 1);
    uint64x2_t matches_rot0_u64x2 = vceqq_u64(a, b);
    uint64x2_t matches_rot1_u64x2 = vceqq_u64(a, b_rot1_u64x2);
    uint64x2_t matches_u64x2 = vorrq_u64(matches_rot0_u64x2, matches_rot1_u64x2);
    return matches_u64x2;
}

/** How many leading lanes of @p a and of @p b the other block's maximum covers, in 2 lanes. */
NUMKONG_INLINE uint32x2_t nk_sparse_steps_u32x4_neon_(uint32x4_t a, uint32x4_t b, nk_u32_t a_max, nk_u32_t b_max) {
    uint32x4_t a_inrange_u32x4 = vcleq_u32(a, vdupq_n_u32(b_max));
    uint32x4_t b_inrange_u32x4 = vcleq_u32(b, vdupq_n_u32(a_max));
    uint8x8_t packed_u8x8 = vmovn_u16(vcombine_u16(vmovn_u32(a_inrange_u32x4), vmovn_u32(b_inrange_u32x4)));
    return vshr_n_u32(vsub_u32(vdup_n_u32(32), vclz_u32(vreinterpret_u32_u8(packed_u8x8))), 3);
}

#if NUMKONG_TARGET_NEON

NUMKONG_API nk_status_t nk_sparse_intersect_u16_neon( //
    nk_u16_t const *a, nk_u16_t const *b,             //
    nk_size_t a_length, nk_size_t b_length,           //
    nk_u16_t *result, nk_size_t *count, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);

    // Galloping through the longer array beats any block walk
    if (a_length / 64 >= b_length || b_length / 64 >= a_length) {
        nk_sparse_intersect_u16_serial_(a, b, a_length, b_length, result, count);
        return nk_success_k;
    }

#if NUMKONG_ALLOW_ISA_REDIRECT
    // The baseline implementation for very small arrays (2 registers or less) can be quite simple:
    if (a_length < 32 && b_length < 32) {
        nk_sparse_intersect_u16_serial_(a, b, a_length, b_length, result, count);
        return nk_success_k;
    }
#endif

    nk_u16_t const *const a_end = a + a_length;
    nk_u16_t const *const b_end = b + b_length;
    nk_b128_vec_t a_vec, b_vec;
    uint16x8_t c_counts_u16x8 = vdupq_n_u16(0);
    nk_size_t stored = 0;

    while (a + 8 <= a_end && b + 8 <= b_end) {
        a_vec.u16x8 = vld1q_u16(a);
        b_vec.u16x8 = vld1q_u16(b);

        // Intersecting registers with `nk_intersect_u16x8_neon_` involves a lot of shuffling
        // and comparisons, so we want to avoid it if the slices don't overlap at all..
        nk_u16_t a_min;
        nk_u16_t a_max = a_vec.u16s[7];
        nk_u16_t b_min = b_vec.u16s[0];
        nk_u16_t b_max = b_vec.u16s[7];

        // If the slices don't overlap, advance the appropriate pointer
        while (a_max < b_min && a + 16 <= a_end) {
            a += 8;
            a_vec.u16x8 = vld1q_u16(a);
            a_max = a_vec.u16s[7];
        }
        a_min = a_vec.u16s[0];
        while (b_max < a_min && b + 16 <= b_end) {
            b += 8;
            b_vec.u16x8 = vld1q_u16(b);
            b_max = b_vec.u16s[7];
        }
        b_min = b_vec.u16s[0];

        // Transform match-masks into "ones", accumulate them between the cycles,
        // and merge all together in the end.
        uint16x8_t a_matches_u16x8 = nk_intersect_u16x8_neon_(a_vec.u16x8, b_vec.u16x8);
        if (result) {
            // Each lane spans 8 bits; the loop ends on the last match, touching only filled slots
            nk_u64_t lane_bits = vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(a_matches_u16x8, 4)), 0);
            for (nk_size_t lane = 0; lane_bits; ++lane, lane_bits >>= 8)
                result[stored] = a[lane], stored += lane_bits & 1;
        }
        else c_counts_u16x8 = vaddq_u16(c_counts_u16x8, vandq_u16(a_matches_u16x8, vdupq_n_u16(1)));

        // Use `vclz_u32` to compute leading zeros for both `a_step` and `b_step` in parallel.
        // Narrow comparison masks from 128→64→32 bits, pack both into a `uint32x2_t`.
        uint16x8_t a_inrange_u16x8 = vcleq_u16(a_vec.u16x8, vdupq_n_u16(b_max));
        uint16x8_t b_inrange_u16x8 = vcleq_u16(b_vec.u16x8, vdupq_n_u16(a_max));
        uint8x8_t a_narrow_u8x8 = vmovn_u16(a_inrange_u16x8);
        uint8x8_t b_narrow_u8x8 = vmovn_u16(b_inrange_u16x8);
        uint8x8_t packed_u8x8 = vshrn_n_u16(vreinterpretq_u16_u8(vcombine_u8(a_narrow_u8x8, b_narrow_u8x8)), 4);
        uint32x2_t clz_u32x2 = vclz_u32(vreinterpret_u32_u8(packed_u8x8));
        a += (32 - vget_lane_u32(clz_u32x2, 0)) / 4;
        b += (32 - vget_lane_u32(clz_u32x2, 1)) / 4;
    }

    nk_size_t tail_count = 0;
    nk_sparse_intersect_u16_serial_(a, b, a_end - a, b_end - b, result ? result + stored : 0, &tail_count);
    *count = stored + tail_count + (nk_size_t)vaddlvq_u16(c_counts_u16x8);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_sparse_intersect_u32_neon( //
    nk_u32_t const *a, nk_u32_t const *b,             //
    nk_size_t a_length, nk_size_t b_length,           //
    nk_u32_t *result, nk_size_t *count, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);

    // Galloping through the longer array beats any block walk
    if (a_length / 64 >= b_length || b_length / 64 >= a_length) {
        nk_sparse_intersect_u32_serial_(a, b, a_length, b_length, result, count);
        return nk_success_k;
    }

#if NUMKONG_ALLOW_ISA_REDIRECT
    // The baseline implementation for very small arrays (2 registers or less) can be quite simple:
    if (a_length < 32 && b_length < 32) {
        nk_sparse_intersect_u32_serial_(a, b, a_length, b_length, result, count);
        return nk_success_k;
    }
#endif

    nk_u32_t const *const a_end = a + a_length;
    nk_u32_t const *const b_end = b + b_length;
    nk_b128_vec_t a_vec, b_vec;
    uint32x4_t c_counts_u32x4 = vdupq_n_u32(0);
    nk_size_t stored = 0;

    while (a + 4 <= a_end && b + 4 <= b_end) {
        a_vec.u32x4 = vld1q_u32(a);
        b_vec.u32x4 = vld1q_u32(b);

        // Intersecting registers with `nk_intersect_u32x4_neon_` involves a lot of shuffling
        // and comparisons, so we want to avoid it if the slices don't overlap at all..
        nk_u32_t a_min;
        nk_u32_t a_max = a_vec.u32s[3];
        nk_u32_t b_min = b_vec.u32s[0];
        nk_u32_t b_max = b_vec.u32s[3];

        // If the slices don't overlap, advance the appropriate pointer
        while (a_max < b_min && a + 8 <= a_end) {
            a += 4;
            a_vec.u32x4 = vld1q_u32(a);
            a_max = a_vec.u32s[3];
        }
        a_min = a_vec.u32s[0];
        while (b_max < a_min && b + 8 <= b_end) {
            b += 4;
            b_vec.u32x4 = vld1q_u32(b);
            b_max = b_vec.u32s[3];
        }
        b_min = b_vec.u32s[0];

        // Transform match-masks into "ones", accumulate them between the cycles,
        // and merge all together in the end.
        uint32x4_t a_matches_u32x4 = nk_intersect_u32x4_neon_(a_vec.u32x4, b_vec.u32x4);
        if (result) {
            // Each lane spans 16 bits; the loop ends on the last match, touching only filled slots
            nk_u64_t lane_bits = vget_lane_u64(
                vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u32(a_matches_u32x4), 4)), 0);
            for (nk_size_t lane = 0; lane_bits; ++lane, lane_bits >>= 16)
                result[stored] = a[lane], stored += lane_bits & 1;
        }
        else c_counts_u32x4 = vaddq_u32(c_counts_u32x4, vandq_u32(a_matches_u32x4, vdupq_n_u32(1)));

        uint32x2_t steps_u32x2 = nk_sparse_steps_u32x4_neon_(a_vec.u32x4, b_vec.u32x4, a_max, b_max);
        a += vget_lane_u32(steps_u32x2, 0);
        b += vget_lane_u32(steps_u32x2, 1);
    }

    nk_size_t tail_count = 0;
    nk_sparse_intersect_u32_serial_(a, b, a_end - a, b_end - b, result ? result + stored : 0, &tail_count);
    *count = stored + tail_count + (nk_size_t)vaddvq_u32(c_counts_u32x4);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_sparse_intersect_u64_neon( //
    nk_u64_t const *a, nk_u64_t const *b,             //
    nk_size_t a_length, nk_size_t b_length,           //
    nk_u64_t *result, nk_size_t *count, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);

    // Two lanes per step lose to the serial merge once every match must be stored
    if (result) {
        nk_sparse_intersect_u64_serial_(a, b, a_length, b_length, result, count);
        return nk_success_k;
    }

#if NUMKONG_ALLOW_ISA_REDIRECT
    // The baseline implementation for very small arrays (2 registers or less) can be quite simple:
    if (a_length < 8 && b_length < 8) {
        nk_sparse_intersect_u64_serial_(a, b, a_length, b_length, result, count);
        return nk_success_k;
    }
#endif

    nk_u64_t const *const a_end = a + a_length;
    nk_u64_t const *const b_end = b + b_length;
    nk_b128_vec_t a_vec, b_vec;
    uint64x2_t c_counts_u64x2 = vdupq_n_u64(0);

    while (a + 2 <= a_end && b + 2 <= b_end) {
        a_vec.u64x2 = vld1q_u64(a);
        b_vec.u64x2 = vld1q_u64(b);

        // Intersecting registers with `nk_intersect_u64x2_neon_` involves comparisons,
        // so we want to avoid it if the slices don't overlap at all.
        nk_u64_t a_min;
        nk_u64_t a_max = a_vec.u64s[1];
        nk_u64_t b_min = b_vec.u64s[0];
        nk_u64_t b_max = b_vec.u64s[1];

        // If the slices don't overlap, advance the appropriate pointer
        while (a_max < b_min && a + 4 <= a_end) {
            a += 2;
            a_vec.u64x2 = vld1q_u64(a);
            a_max = a_vec.u64s[1];
        }
        a_min = a_vec.u64s[0];
        while (b_max < a_min && b + 4 <= b_end) {
            b += 2;
            b_vec.u64x2 = vld1q_u64(b);
            b_max = b_vec.u64s[1];
        }
        b_min = b_vec.u64s[0];

        // Now we are likely to have some overlap, so we can intersect the registers
        // Transform match-masks into "ones", accumulate them between the cycles,
        // and merge all together in the end.
        uint64x2_t a_matches_u64x2 = nk_intersect_u64x2_neon_(a_vec.u64x2, b_vec.u64x2);
        c_counts_u64x2 = vaddq_u64(c_counts_u64x2, vandq_u64(a_matches_u64x2, vdupq_n_u64(1)));

        uint64x2_t a_inrange_u64x2 = vcleq_u64(a_vec.u64x2, vdupq_n_u64(b_max));
        uint64x2_t b_inrange_u64x2 = vcleq_u64(b_vec.u64x2, vdupq_n_u64(a_max));
        uint16x4_t packed_u16x4 = vmovn_u32(vcombine_u32(vmovn_u64(a_inrange_u64x2), vmovn_u64(b_inrange_u64x2)));
        uint32x2_t clz_u32x2 = vclz_u32(vreinterpret_u32_u16(packed_u16x4));
        a += (32 - vget_lane_u32(clz_u32x2, 0)) / 16;
        b += (32 - vget_lane_u32(clz_u32x2, 1)) / 16;
    }

    nk_size_t tail_count = 0;
    nk_sparse_intersect_u64_serial_(a, b, a_end - a, b_end - b, 0, &tail_count);
    *count = tail_count + (nk_size_t)vaddvq_u64(c_counts_u64x2);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_sparse_dot_u32f32_neon(nk_u32_t const *a, nk_u32_t const *b, nk_f32_t const *a_weights,
                                                  nk_f32_t const *b_weights, nk_size_t a_length, nk_size_t b_length,
                                                  nk_f64_t *product, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t i = 0, j = 0;
    float64x2_t sum_low_f64x2 = vdupq_n_f64(0), sum_high_f64x2 = vdupq_n_f64(0);
    while (i + 4 <= a_length && j + 4 <= b_length) {
        uint32x4_t a_u32x4 = vld1q_u32(a + i), b_u32x4 = vld1q_u32(b + j);
        uint32x4_t b_weights_u32x4 = vreinterpretq_u32_f32(vld1q_f32(b_weights + j));
        // Masking both weights keeps unmatched NaNs and infinities out of the sum
        uint32x4_t matched_u32x4 = vdupq_n_u32(0), aligned_u32x4 = vdupq_n_u32(0);
        for (unsigned rotation = 0; rotation != 4; ++rotation) {
            uint32x4_t matches_u32x4 = vceqq_u32(a_u32x4, b_u32x4);
            matched_u32x4 = vorrq_u32(matched_u32x4, matches_u32x4);
            aligned_u32x4 = vorrq_u32(aligned_u32x4, vandq_u32(matches_u32x4, b_weights_u32x4));
            b_u32x4 = vextq_u32(b_u32x4, b_u32x4, 1);
            b_weights_u32x4 = vextq_u32(b_weights_u32x4, b_weights_u32x4, 1);
        }
        float32x4_t a_matched_f32x4 = vreinterpretq_f32_u32(
            vandq_u32(matched_u32x4, vreinterpretq_u32_f32(vld1q_f32(a_weights + i))));
        float32x4_t b_matched_f32x4 = vreinterpretq_f32_u32(aligned_u32x4);
        sum_low_f64x2 = vfmaq_f64(sum_low_f64x2, vcvt_f64_f32(vget_low_f32(a_matched_f32x4)),
                                  vcvt_f64_f32(vget_low_f32(b_matched_f32x4)));
        sum_high_f64x2 = vfmaq_f64(sum_high_f64x2, vcvt_high_f64_f32(a_matched_f32x4),
                                   vcvt_high_f64_f32(b_matched_f32x4));
        uint32x2_t steps_u32x2 = nk_sparse_steps_u32x4_neon_(a_u32x4, b_u32x4, a[i + 3], b[j + 3]);
        i += vget_lane_u32(steps_u32x2, 0);
        j += vget_lane_u32(steps_u32x2, 1);
    }
    nk_f64_t tail_product;
    nk_sparse_dot_u32f32_serial_(a + i, b + j, a_weights + i, b_weights + j, a_length - i, b_length - j, &tail_product);
    *product = vaddvq_f64(vaddq_f64(sum_low_f64x2, sum_high_f64x2)) + tail_product;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_sparse_dot_u16bf16_neon(nk_u16_t const *a, nk_u16_t const *b, nk_bf16_t const *a_weights,
                                                   nk_bf16_t const *b_weights, nk_size_t a_length, nk_size_t b_length,
                                                   nk_f32_t *product, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t i = 0, j = 0;
    float32x4_t sum_low_f32x4 = vdupq_n_f32(0), sum_high_f32x4 = vdupq_n_f32(0);
    while (i + 8 <= a_length && j + 8 <= b_length) {
        uint16x8_t a_u16x8 = vld1q_u16(a + i), b_u16x8 = vld1q_u16(b + j);
        uint16x8_t b_weights_u16x8 = vld1q_u16((nk_u16_t const *)(b_weights + j));
        nk_u16_t const a_max = a[i + 7], b_max = b[j + 7];
        uint16x8_t a_inrange_u16x8 = vcleq_u16(a_u16x8, vdupq_n_u16(b_max));
        uint16x8_t b_inrange_u16x8 = vcleq_u16(b_u16x8, vdupq_n_u16(a_max));
        // Masking both weights keeps unmatched NaNs and infinities out of the sum
        uint16x8_t matched_u16x8 = vdupq_n_u16(0), aligned_u16x8 = vdupq_n_u16(0);
        for (unsigned rotation = 0; rotation != 8; ++rotation) {
            uint16x8_t matches_u16x8 = vceqq_u16(a_u16x8, b_u16x8);
            matched_u16x8 = vorrq_u16(matched_u16x8, matches_u16x8);
            aligned_u16x8 = vorrq_u16(aligned_u16x8, vandq_u16(matches_u16x8, b_weights_u16x8));
            b_u16x8 = vextq_u16(b_u16x8, b_u16x8, 1);
            b_weights_u16x8 = vextq_u16(b_weights_u16x8, b_weights_u16x8, 1);
        }
        uint16x8_t a_matched_u16x8 = vandq_u16(matched_u16x8, vld1q_u16((nk_u16_t const *)(a_weights + i)));
        sum_low_f32x4 = vfmaq_f32(sum_low_f32x4, nk_bf16x4_to_f32x4_neon_(vget_low_u16(a_matched_u16x8)),
                                  nk_bf16x4_to_f32x4_neon_(vget_low_u16(aligned_u16x8)));
        sum_high_f32x4 = vfmaq_f32(sum_high_f32x4, nk_bf16x4_to_f32x4_neon_(vget_high_u16(a_matched_u16x8)),
                                   nk_bf16x4_to_f32x4_neon_(vget_high_u16(aligned_u16x8)));
        uint8x8_t packed_u8x8 = vshrn_n_u16(
            vreinterpretq_u16_u8(vcombine_u8(vmovn_u16(a_inrange_u16x8), vmovn_u16(b_inrange_u16x8))), 4);
        uint32x2_t clz_u32x2 = vclz_u32(vreinterpret_u32_u8(packed_u8x8));
        i += (32 - vget_lane_u32(clz_u32x2, 0)) / 4;
        j += (32 - vget_lane_u32(clz_u32x2, 1)) / 4;
    }
    nk_f32_t tail_product;
    nk_sparse_dot_u16bf16_serial_(a + i, b + j, a_weights + i, b_weights + j, a_length - i, b_length - j,
                                  &tail_product);
    *product = vaddvq_f32(vaddq_f32(sum_low_f32x4, sum_high_f32x4)) + tail_product;
    return nk_success_k;
}

#endif // NUMKONG_TARGET_NEON

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_ARM64_NEON_
#endif // NUMKONG_ARCH_ARM64_
#endif // NUMKONG_SPARSE_NEON_H
