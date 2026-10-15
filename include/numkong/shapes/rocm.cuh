/**
 *  @file include/numkong/shapes/rocm.cuh
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief Chamfer and Hausdorff between two packs on 32 lanes of every ROCm device, for kernels
 *      that score one cloud against many from inside a wavefront.
 *
 *  @sa include/numkong/shapes/simt.cuh
 *  @sa include/numkong/shapes/cuda.cuh
 *
 *  The twins of the @c cuda warp bodies over the same @c simt passes. A team of 32 lanes is a whole
 *  wavefront on RDNA and gfx1250 and half of one on CDNA, where shuffles with offsets below 32 stay
 *  within each half, so two teams share a CDNA wavefront.
 */
#ifndef NUMKONG_SHAPES_ROCM_CUH
#define NUMKONG_SHAPES_ROCM_CUH

#if NUMKONG_ARCH_ROCM_

#include "numkong/rocm.cuh"
#include "numkong/shapes/simt.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Chamfer and Hausdorff

/** The Chamfer distance between two packs of F32 points on a team of 32 lanes. Every lane calls
 *  with the same packs and its own @p lane , the lanes split each cloud's points, and the team sums
 *  their nearest distances. Distances up to @p limit are exact; past it, the team may stop early
 *  and return a lower bound that also lies past @p limit . */
NUMKONG_DEVICE nk_f64_t nk_chamfer_warp_packed_f32_rocm_(void const *a_packed, void const *b_packed, nk_f64_t limit,
                                                         unsigned lane) {
    nk_size_t const a_count = (nk_size_t)((nk_shapes_packed_header_f64_t const *)a_packed)->points;
    nk_size_t const b_count = (nk_size_t)((nk_shapes_packed_header_f64_t const *)b_packed)->points;
    if (!a_count || !b_count) return a_count == b_count ? 0 : NUMKONG_F64_INF;
    nk_size_t a_stride, b_stride;
    nk_f32_t const *a_axes = nk_shapes_packed_axes_f32_simt_(a_packed, &a_stride);
    nk_f32_t const *b_axes = nk_shapes_packed_axes_f32_simt_(b_packed, &b_stride);
    // Per flip, the finished directions' mean distances: F/n after one, F/n + B/m after both
    nk_f64_t means[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int direction = 0; direction != 2; ++direction) {
        nk_f32_t const *outer_axes = direction ? b_axes : a_axes, *inner_axes = direction ? a_axes : b_axes;
        nk_size_t const outer_stride = direction ? b_stride : a_stride, inner_stride = direction ? a_stride : b_stride;
        nk_size_t const outer_count = direction ? b_count : a_count, inner_count = direction ? a_count : b_count;
        nk_f64_t const count = (nk_f64_t)outer_count;
        nk_f64_t sums[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (nk_size_t first = 0; first < outer_count; first += 32) {
            nk_size_t const point = first + lane;
            nk_f64_t distances[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            if (point < outer_count) {
                nk_shapes_nearest_squares_f32_simt_(outer_axes[point], outer_axes[outer_stride + point],
                                                    outer_axes[2 * outer_stride + point], inner_axes, inner_stride,
                                                    inner_count, distances);
                for (int flip = 0; flip != 8; ++flip) distances[flip] = __dsqrt_rn(distances[flip]);
            }
            int exceeded = 1;
            for (int flip = 0; flip != 8; ++flip) {
                for (unsigned offset = 16; offset; offset >>= 1)
                    distances[flip] += nk_shuffle_xor_f64_rocm_(distances[flip], offset);
                sums[flip] += distances[flip];
                exceeded &= sums[flip] > (2 * limit - means[flip]) * count;
            }
            if (!exceeded) continue;
            nk_f64_t partial = NUMKONG_F64_INF;
            for (int flip = 0; flip != 8; ++flip) {
                nk_f64_t const chamfer = (means[flip] + sums[flip] / count) / 2;
                partial = chamfer < partial ? chamfer : partial;
            }
            if (partial > limit) return partial;
        }
        for (int flip = 0; flip != 8; ++flip) means[flip] += sums[flip] / count;
    }
    nk_f64_t best = means[0] / 2;
    for (int flip = 1; flip != 8; ++flip) best = means[flip] / 2 < best ? means[flip] / 2 : best;
    return best;
}

/** The Chamfer distance between two packs of F16 points on a team of 32 lanes. Every lane calls
 *  with the same packs and its own @p lane , the lanes split each cloud's points, and the team sums
 *  their nearest distances. Distances up to @p limit are exact; past it, the team may stop early
 *  and return a lower bound that also lies past @p limit . */
NUMKONG_DEVICE nk_f32_t nk_chamfer_warp_packed_f16_rocm_(void const *a_packed, void const *b_packed, nk_f32_t limit,
                                                         unsigned lane) {
    nk_size_t const a_count = ((nk_shapes_packed_header_f32_t const *)a_packed)->points;
    nk_size_t const b_count = ((nk_shapes_packed_header_f32_t const *)b_packed)->points;
    if (!a_count || !b_count) return a_count == b_count ? 0 : NUMKONG_F32_INF;
    nk_size_t a_stride, b_stride;
    nk_f16_t const *a_axes = nk_shapes_packed_axes_f16_simt_(a_packed, &a_stride);
    nk_f16_t const *b_axes = nk_shapes_packed_axes_f16_simt_(b_packed, &b_stride);
    // Per flip, the finished directions' mean distances: F/n after one, F/n + B/m after both
    nk_f32_t means[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int direction = 0; direction != 2; ++direction) {
        nk_f16_t const *outer_axes = direction ? b_axes : a_axes, *inner_axes = direction ? a_axes : b_axes;
        nk_size_t const outer_stride = direction ? b_stride : a_stride, inner_stride = direction ? a_stride : b_stride;
        nk_size_t const outer_count = direction ? b_count : a_count, inner_count = direction ? a_count : b_count;
        nk_f32_t const count = (nk_f32_t)outer_count;
        nk_f32_t sums[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (nk_size_t first = 0; first < outer_count; first += 32) {
            nk_size_t const point = first + lane;
            nk_f32_t distances[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            if (point < outer_count) {
                nk_shapes_nearest_squares_f16_simt_(
                    nk_shapes_coordinate_f16_simt_(outer_axes + point),
                    nk_shapes_coordinate_f16_simt_(outer_axes + outer_stride + point),
                    nk_shapes_coordinate_f16_simt_(outer_axes + 2 * outer_stride + point), inner_axes, inner_stride,
                    inner_count, distances);
                for (int flip = 0; flip != 8; ++flip) distances[flip] = __fsqrt_rn(distances[flip]);
            }
            int exceeded = 1;
            for (int flip = 0; flip != 8; ++flip) {
                for (unsigned offset = 16; offset; offset >>= 1)
                    distances[flip] += nk_shuffle_xor_f32_rocm_(distances[flip], offset);
                sums[flip] += distances[flip];
                exceeded &= sums[flip] > (2 * limit - means[flip]) * count;
            }
            if (!exceeded) continue;
            nk_f32_t partial = NUMKONG_F32_INF;
            for (int flip = 0; flip != 8; ++flip) {
                nk_f32_t const chamfer = (means[flip] + sums[flip] / count) / 2;
                partial = chamfer < partial ? chamfer : partial;
            }
            if (partial > limit) return partial;
        }
        for (int flip = 0; flip != 8; ++flip) means[flip] += sums[flip] / count;
    }
    nk_f32_t best = means[0] / 2;
    for (int flip = 1; flip != 8; ++flip) best = means[flip] / 2 < best ? means[flip] / 2 : best;
    return best;
}

/** The Hausdorff distance between two packs of F32 points on a team of 32 lanes. Every lane calls
 *  with the same packs and its own @p lane , the lanes split each cloud's points, and the team
 *  keeps the farthest nearest square. Distances up to @p limit are exact; past it, the team may
 *  return early, even before reading a point, with a lower bound that also lies past @p limit . */
NUMKONG_DEVICE nk_f64_t nk_hausdorff_warp_packed_f32_rocm_(void const *a_packed, void const *b_packed, nk_f64_t limit,
                                                           unsigned lane) {
    nk_size_t const a_count = (nk_size_t)((nk_shapes_packed_header_f64_t const *)a_packed)->points;
    nk_size_t const b_count = (nk_size_t)((nk_shapes_packed_header_f64_t const *)b_packed)->points;
    if (!a_count || !b_count) return a_count == b_count ? 0 : NUMKONG_F64_INF;
    nk_f64_t const bound = nk_hausdorff_bound_packed_f32_simt_(a_packed, b_packed);
    if (bound > limit) return bound;
    nk_size_t a_stride, b_stride;
    nk_f32_t const *a_axes = nk_shapes_packed_axes_f32_simt_(a_packed, &a_stride);
    nk_f32_t const *b_axes = nk_shapes_packed_axes_f32_simt_(b_packed, &b_stride);
    nk_f64_t const limit_squared = limit * limit;
    nk_f64_t farthest[8] = {0, 0, 0, 0, 0, 0, 0, 0}, nearest[8];
    for (int direction = 0; direction != 2; ++direction) {
        nk_f32_t const *outer_axes = direction ? b_axes : a_axes, *inner_axes = direction ? a_axes : b_axes;
        nk_size_t const outer_stride = direction ? b_stride : a_stride, inner_stride = direction ? a_stride : b_stride;
        nk_size_t const outer_count = direction ? b_count : a_count, inner_count = direction ? a_count : b_count;
        for (nk_size_t first = 0; first < outer_count; first += 32) {
            nk_size_t const point = first + lane;
            if (point < outer_count) {
                nk_shapes_nearest_squares_f32_simt_(outer_axes[point], outer_axes[outer_stride + point],
                                                    outer_axes[2 * outer_stride + point], inner_axes, inner_stride,
                                                    inner_count, nearest);
                for (int flip = 0; flip != 8; ++flip)
                    farthest[flip] = nearest[flip] > farthest[flip] ? nearest[flip] : farthest[flip];
            }
            nk_f64_t least = NUMKONG_F64_INF;
            for (int flip = 0; flip != 8; ++flip) {
                for (unsigned offset = 16; offset; offset >>= 1) {
                    nk_f64_t const other = nk_shuffle_xor_f64_rocm_(farthest[flip], offset);
                    farthest[flip] = other > farthest[flip] ? other : farthest[flip];
                }
                least = farthest[flip] < least ? farthest[flip] : least;
            }
            if (least > limit_squared && __dsqrt_rn(least) > limit) return __dsqrt_rn(least);
        }
    }
    nk_f64_t least = farthest[0];
    for (int flip = 1; flip != 8; ++flip) least = farthest[flip] < least ? farthest[flip] : least;
    return __dsqrt_rn(least);
}

/** The Hausdorff distance between two packs of F16 points on a team of 32 lanes. Every lane calls
 *  with the same packs and its own @p lane , the lanes split each cloud's points, and the team
 *  keeps the farthest nearest square. Distances up to @p limit are exact; past it, the team may
 *  return early, even before reading a point, with a lower bound that also lies past @p limit . */
NUMKONG_DEVICE nk_f32_t nk_hausdorff_warp_packed_f16_rocm_(void const *a_packed, void const *b_packed, nk_f32_t limit,
                                                           unsigned lane) {
    nk_size_t const a_count = ((nk_shapes_packed_header_f32_t const *)a_packed)->points;
    nk_size_t const b_count = ((nk_shapes_packed_header_f32_t const *)b_packed)->points;
    if (!a_count || !b_count) return a_count == b_count ? 0 : NUMKONG_F32_INF;
    nk_f32_t const bound = nk_hausdorff_bound_packed_f16_simt_(a_packed, b_packed);
    if (bound > limit) return bound;
    nk_size_t a_stride, b_stride;
    nk_f16_t const *a_axes = nk_shapes_packed_axes_f16_simt_(a_packed, &a_stride);
    nk_f16_t const *b_axes = nk_shapes_packed_axes_f16_simt_(b_packed, &b_stride);
    nk_f32_t const limit_squared = limit * limit;
    nk_f32_t farthest[8] = {0, 0, 0, 0, 0, 0, 0, 0}, nearest[8];
    for (int direction = 0; direction != 2; ++direction) {
        nk_f16_t const *outer_axes = direction ? b_axes : a_axes, *inner_axes = direction ? a_axes : b_axes;
        nk_size_t const outer_stride = direction ? b_stride : a_stride, inner_stride = direction ? a_stride : b_stride;
        nk_size_t const outer_count = direction ? b_count : a_count, inner_count = direction ? a_count : b_count;
        for (nk_size_t first = 0; first < outer_count; first += 32) {
            nk_size_t const point = first + lane;
            if (point < outer_count) {
                nk_shapes_nearest_squares_f16_simt_(
                    nk_shapes_coordinate_f16_simt_(outer_axes + point),
                    nk_shapes_coordinate_f16_simt_(outer_axes + outer_stride + point),
                    nk_shapes_coordinate_f16_simt_(outer_axes + 2 * outer_stride + point), inner_axes, inner_stride,
                    inner_count, nearest);
                for (int flip = 0; flip != 8; ++flip)
                    farthest[flip] = nearest[flip] > farthest[flip] ? nearest[flip] : farthest[flip];
            }
            nk_f32_t least = NUMKONG_F32_INF;
            for (int flip = 0; flip != 8; ++flip) {
                for (unsigned offset = 16; offset; offset >>= 1) {
                    nk_f32_t const other = nk_shuffle_xor_f32_rocm_(farthest[flip], offset);
                    farthest[flip] = other > farthest[flip] ? other : farthest[flip];
                }
                least = farthest[flip] < least ? farthest[flip] : least;
            }
            if (least > limit_squared && __fsqrt_rn(least) > limit) return __fsqrt_rn(least);
        }
    }
    nk_f32_t least = farthest[0];
    for (int flip = 1; flip != 8; ++flip) least = farthest[flip] < least ? farthest[flip] : least;
    return __fsqrt_rn(least);
}

#pragma endregion Chamfer and Hausdorff

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_ROCM_
#endif // NUMKONG_SHAPES_ROCM_CUH
