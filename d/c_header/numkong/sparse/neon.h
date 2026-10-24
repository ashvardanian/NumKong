/**
 *  @brief NEON-accelerated Sparse Vector Operations.
 *  @file include/numkong/sparse/neon.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *
 *  @sa include/numkong/sparse.h
 */
#ifndef NK_SPARSE_NEON_H
#define NK_SPARSE_NEON_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEON

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a")
#endif

NK_INTERNAL uint32x4_t nk_intersect_u32x4_neon_(uint32x4_t a, uint32x4_t b) {
    uint32x4_t b_rot1 = vextq_u32(b, b, 1);
    uint32x4_t b_rot2 = vextq_u32(b, b, 2);
    uint32x4_t b_rot3 = vextq_u32(b, b, 3);
    uint32x4_t matches_rot0 = vceqq_u32(a, b);
    uint32x4_t matches_rot1 = vceqq_u32(a, b_rot1);
    uint32x4_t matches_rot2 = vceqq_u32(a, b_rot2);
    uint32x4_t matches_rot3 = vceqq_u32(a, b_rot3);
    uint32x4_t matches = vorrq_u32(vorrq_u32(matches_rot0, matches_rot1), vorrq_u32(matches_rot2, matches_rot3));
    return matches;
}

NK_INTERNAL uint16x8_t nk_intersect_u16x8_neon_(uint16x8_t a, uint16x8_t b) {
    uint16x8_t b_rot1 = vextq_u16(b, b, 1);
    uint16x8_t b_rot2 = vextq_u16(b, b, 2);
    uint16x8_t b_rot3 = vextq_u16(b, b, 3);
    uint16x8_t b_rot4 = vextq_u16(b, b, 4);
    uint16x8_t b_rot5 = vextq_u16(b, b, 5);
    uint16x8_t b_rot6 = vextq_u16(b, b, 6);
    uint16x8_t b_rot7 = vextq_u16(b, b, 7);
    uint16x8_t matches_rot0 = vceqq_u16(a, b);
    uint16x8_t matches_rot1 = vceqq_u16(a, b_rot1);
    uint16x8_t matches_rot2 = vceqq_u16(a, b_rot2);
    uint16x8_t matches_rot3 = vceqq_u16(a, b_rot3);
    uint16x8_t matches_rot4 = vceqq_u16(a, b_rot4);
    uint16x8_t matches_rot5 = vceqq_u16(a, b_rot5);
    uint16x8_t matches_rot6 = vceqq_u16(a, b_rot6);
    uint16x8_t matches_rot7 = vceqq_u16(a, b_rot7);
    uint16x8_t matches = vorrq_u16(
        vorrq_u16(vorrq_u16(matches_rot0, matches_rot1), vorrq_u16(matches_rot2, matches_rot3)),
        vorrq_u16(vorrq_u16(matches_rot4, matches_rot5), vorrq_u16(matches_rot6, matches_rot7)));
    return matches;
}

NK_PUBLIC void nk_sparse_intersect_u16_neon( //
    nk_u16_t const *a, nk_u16_t const *b,    //
    nk_size_t a_length, nk_size_t b_length,  //
    nk_u16_t *result, nk_size_t *count) {

    // NEON lacks compress-store, so fall back to serial for result output
    if (result) {
        nk_sparse_intersect_u16_serial(a, b, a_length, b_length, result, count);
        return;
    }

#if NK_ALLOW_ISA_REDIRECT
    // The baseline implementation for very small arrays (2 registers or less) can be quite simple:
    if (a_length < 32 && b_length < 32) {
        nk_sparse_intersect_u16_serial(a, b, a_length, b_length, result, count);
        return;
    }
#endif

    nk_u16_t const *const a_end = a + a_length;
    nk_u16_t const *const b_end = b + b_length;
    nk_b128_vec_t a_vec, b_vec;
    uint16x8_t c_counts_u16x8 = vdupq_n_u16(0);

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
        uint16x8_t a_matches = nk_intersect_u16x8_neon_(a_vec.u16x8, b_vec.u16x8);
        c_counts_u16x8 = vaddq_u16(c_counts_u16x8, vandq_u16(a_matches, vdupq_n_u16(1)));

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
    nk_sparse_intersect_u16_serial(a, b, a_end - a, b_end - b, 0, &tail_count);
    *count = tail_count + (nk_size_t)vaddvq_u16(c_counts_u16x8);
}

NK_PUBLIC void nk_sparse_intersect_u32_neon( //
    nk_u32_t const *a, nk_u32_t const *b,    //
    nk_size_t a_length, nk_size_t b_length,  //
    nk_u32_t *result, nk_size_t *count) {

    // NEON lacks compress-store, so fall back to serial for result output
    if (result) {
        nk_sparse_intersect_u32_serial(a, b, a_length, b_length, result, count);
        return;
    }

#if NK_ALLOW_ISA_REDIRECT
    // The baseline implementation for very small arrays (2 registers or less) can be quite simple:
    if (a_length < 32 && b_length < 32) {
        nk_sparse_intersect_u32_serial(a, b, a_length, b_length, result, count);
        return;
    }
#endif

    nk_u32_t const *const a_end = a + a_length;
    nk_u32_t const *const b_end = b + b_length;
    nk_b128_vec_t a_vec, b_vec;
    uint32x4_t c_counts_u32x4 = vdupq_n_u32(0);

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
        uint32x4_t a_matches = nk_intersect_u32x4_neon_(a_vec.u32x4, b_vec.u32x4);
        c_counts_u32x4 = vaddq_u32(c_counts_u32x4, vandq_u32(a_matches, vdupq_n_u32(1)));

        uint32x4_t a_inrange_u32x4 = vcleq_u32(a_vec.u32x4, vdupq_n_u32(b_max));
        uint32x4_t b_inrange_u32x4 = vcleq_u32(b_vec.u32x4, vdupq_n_u32(a_max));
        uint8x8_t packed_u8x8 = vmovn_u16(vcombine_u16(vmovn_u32(a_inrange_u32x4), vmovn_u32(b_inrange_u32x4)));
        uint32x2_t clz_u32x2 = vclz_u32(vreinterpret_u32_u8(packed_u8x8));
        a += (32 - vget_lane_u32(clz_u32x2, 0)) / 8;
        b += (32 - vget_lane_u32(clz_u32x2, 1)) / 8;
    }

    nk_size_t tail_count = 0;
    nk_sparse_intersect_u32_serial(a, b, a_end - a, b_end - b, 0, &tail_count);
    *count = tail_count + (nk_size_t)vaddvq_u32(c_counts_u32x4);
}

NK_INTERNAL uint64x2_t nk_intersect_u64x2_neon_(uint64x2_t a, uint64x2_t b) {
    uint64x2_t b_rot1 = vextq_u64(b, b, 1);
    uint64x2_t matches_rot0 = vceqq_u64(a, b);
    uint64x2_t matches_rot1 = vceqq_u64(a, b_rot1);
    uint64x2_t matches = vorrq_u64(matches_rot0, matches_rot1);
    return matches;
}

NK_PUBLIC void nk_sparse_intersect_u64_neon( //
    nk_u64_t const *a, nk_u64_t const *b,    //
    nk_size_t a_length, nk_size_t b_length,  //
    nk_u64_t *result, nk_size_t *count) {

    // NEON lacks compress-store, so fall back to serial for result output
    if (result) {
        nk_sparse_intersect_u64_serial(a, b, a_length, b_length, result, count);
        return;
    }

#if NK_ALLOW_ISA_REDIRECT
    // The baseline implementation for very small arrays (2 registers or less) can be quite simple:
    if (a_length < 8 && b_length < 8) {
        nk_sparse_intersect_u64_serial(a, b, a_length, b_length, result, count);
        return;
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
        uint64x2_t a_matches = nk_intersect_u64x2_neon_(a_vec.u64x2, b_vec.u64x2);
        c_counts_u64x2 = vaddq_u64(c_counts_u64x2, vandq_u64(a_matches, vdupq_n_u64(1)));

        uint64x2_t a_inrange_u64x2 = vcleq_u64(a_vec.u64x2, vdupq_n_u64(b_max));
        uint64x2_t b_inrange_u64x2 = vcleq_u64(b_vec.u64x2, vdupq_n_u64(a_max));
        uint16x4_t packed_u16x4 = vmovn_u32(vcombine_u32(vmovn_u64(a_inrange_u64x2), vmovn_u64(b_inrange_u64x2)));
        uint32x2_t clz_u32x2 = vclz_u32(vreinterpret_u32_u16(packed_u16x4));
        a += (32 - vget_lane_u32(clz_u32x2, 0)) / 16;
        b += (32 - vget_lane_u32(clz_u32x2, 1)) / 16;
    }

    nk_size_t tail_count = 0;
    nk_sparse_intersect_u64_serial(a, b, a_end - a, b_end - b, 0, &tail_count);
    *count = tail_count + (nk_size_t)vaddvq_u64(c_counts_u64x2);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEON
#endif // NK_TARGET_ARM_
#endif // NK_SPARSE_NEON_H
