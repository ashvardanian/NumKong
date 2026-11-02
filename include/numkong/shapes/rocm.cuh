/**
 *  @file include/numkong/shapes/rocm.cuh
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief Many-to-many shape distances on the SIMT cores of every ROCm device, the @c rocm exports.
 *
 *  @sa include/numkong/shapes/simt.cuh
 *  @sa include/numkong/shapes/cuda.cuh
 *
 *  Everything vendor-neutral lives in @c simt.cuh: the transport walks and their schedules, the
 *  RMSD pairs, and the best-flip writers. Here stay the launches, the team tiers and the lanes'
 *  bodies. Teams that span blocks spin on each other, so they launch through a cooperative kernel,
 *  which keeps every block resident at once.
 *
 *  The warp bodies are twins of the @c cuda ones over the same @c simt passes. A team of 32 lanes
 *  is a whole wavefront on RDNA and gfx1250 and half of one on CDNA, where shuffles with offsets
 *  below 32 stay within each half, so two teams share a CDNA wavefront.
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

/** The Chamfer distances between two packs of F32 points under each of the 8 flips on a team of 32
 *  lanes, which @c nk_chamfer_warp_packed_f32_rocm_ reduces. Distances up to @p limit are exact;
 *  past it, the team may stop early, leaving each flip's lower bound, the least of which also lies
 *  past @p limit . */
NUMKONG_DEVICE void nk_chamfer_flips_warp_packed_f32_rocm_(void const *a_packed, void const *b_packed, nk_f64_t limit,
                                                           unsigned lane, nk_f64_t *flip_distances) {
    nk_size_t const a_count = (nk_size_t)((nk_shapes_packed_header_f64_t const *)a_packed)->points;
    nk_size_t const b_count = (nk_size_t)((nk_shapes_packed_header_f64_t const *)b_packed)->points;
    if (!a_count || !b_count) {
        for (int flip = 0; flip != 8; ++flip) flip_distances[flip] = a_count == b_count ? 0 : NUMKONG_F64_INF;
        return;
    }
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
                flip_distances[flip] = (means[flip] + sums[flip] / count) / 2;
                partial = flip_distances[flip] < partial ? flip_distances[flip] : partial;
            }
            if (partial > limit) return;
        }
        for (int flip = 0; flip != 8; ++flip) means[flip] += sums[flip] / count;
    }
    for (int flip = 0; flip != 8; ++flip) flip_distances[flip] = means[flip] / 2;
}

/** The Chamfer distance between two packs of F32 points on a team of 32 lanes. Every lane calls
 *  with the same packs and its own @p lane , the lanes split each cloud's points, and the team sums
 *  their nearest distances. Distances up to @p limit are exact; past it, the team may stop early
 *  and return a lower bound that also lies past @p limit . */
NUMKONG_DEVICE nk_f64_t nk_chamfer_warp_packed_f32_rocm_(void const *a_packed, void const *b_packed, nk_f64_t limit,
                                                         unsigned lane) {
    nk_f64_t flip_distances[8];
    nk_chamfer_flips_warp_packed_f32_rocm_(a_packed, b_packed, limit, lane, flip_distances);
    return nk_shapes_least_flip_f64_simt_(flip_distances);
}

/** The Chamfer distances between two packs of F16 points under each of the 8 flips on a team of 32
 *  lanes, which @c nk_chamfer_warp_packed_f16_rocm_ reduces. Distances up to @p limit are exact;
 *  past it, the team may stop early, leaving each flip's lower bound, the least of which also lies
 *  past @p limit . */
NUMKONG_DEVICE void nk_chamfer_flips_warp_packed_f16_rocm_(void const *a_packed, void const *b_packed, nk_f32_t limit,
                                                           unsigned lane, nk_f32_t *flip_distances) {
    nk_size_t const a_count = ((nk_shapes_packed_header_f32_t const *)a_packed)->points;
    nk_size_t const b_count = ((nk_shapes_packed_header_f32_t const *)b_packed)->points;
    if (!a_count || !b_count) {
        for (int flip = 0; flip != 8; ++flip) flip_distances[flip] = a_count == b_count ? 0 : NUMKONG_F32_INF;
        return;
    }
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
                flip_distances[flip] = (means[flip] + sums[flip] / count) / 2;
                partial = flip_distances[flip] < partial ? flip_distances[flip] : partial;
            }
            if (partial > limit) return;
        }
        for (int flip = 0; flip != 8; ++flip) means[flip] += sums[flip] / count;
    }
    for (int flip = 0; flip != 8; ++flip) flip_distances[flip] = means[flip] / 2;
}

/** The Chamfer distance between two packs of F16 points on a team of 32 lanes. Every lane calls
 *  with the same packs and its own @p lane , the lanes split each cloud's points, and the team sums
 *  their nearest distances. Distances up to @p limit are exact; past it, the team may stop early
 *  and return a lower bound that also lies past @p limit . */
NUMKONG_DEVICE nk_f32_t nk_chamfer_warp_packed_f16_rocm_(void const *a_packed, void const *b_packed, nk_f32_t limit,
                                                         unsigned lane) {
    nk_f32_t flip_distances[8];
    nk_chamfer_flips_warp_packed_f16_rocm_(a_packed, b_packed, limit, lane, flip_distances);
    return nk_shapes_least_flip_f32_simt_(flip_distances);
}

/** The Hausdorff distances between two packs of F32 points under each of the 8 flips on a team of
 *  32 lanes, which @c nk_hausdorff_warp_packed_f32_rocm_ reduces. Distances up to @p limit are
 *  exact; past it, the team may stop early, leaving each flip's lower bound, the least of which
 *  also lies past @p limit . */
NUMKONG_DEVICE void nk_hausdorff_flips_warp_packed_f32_rocm_(void const *a_packed, void const *b_packed, nk_f64_t limit,
                                                             unsigned lane, nk_f64_t *flip_distances) {
    nk_size_t const a_count = (nk_size_t)((nk_shapes_packed_header_f64_t const *)a_packed)->points;
    nk_size_t const b_count = (nk_size_t)((nk_shapes_packed_header_f64_t const *)b_packed)->points;
    if (!a_count || !b_count) {
        for (int flip = 0; flip != 8; ++flip) flip_distances[flip] = a_count == b_count ? 0 : NUMKONG_F64_INF;
        return;
    }
    nk_f64_t const bound = nk_hausdorff_bound_packed_f32_simt_(a_packed, b_packed);
    if (bound > limit) {
        for (int flip = 0; flip != 8; ++flip) flip_distances[flip] = bound;
        return;
    }
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
            if (least > limit_squared && __dsqrt_rn(least) > limit) {
                for (int flip = 0; flip != 8; ++flip) flip_distances[flip] = __dsqrt_rn(farthest[flip]);
                return;
            }
        }
    }
    for (int flip = 0; flip != 8; ++flip) flip_distances[flip] = __dsqrt_rn(farthest[flip]);
}

/** The Hausdorff distance between two packs of F32 points on a team of 32 lanes. Every lane calls
 *  with the same packs and its own @p lane , the lanes split each cloud's points, and the team
 *  keeps the farthest nearest square. Distances up to @p limit are exact; past it, the team may
 *  return early, even before reading a point, with a lower bound that also lies past @p limit . */
NUMKONG_DEVICE nk_f64_t nk_hausdorff_warp_packed_f32_rocm_(void const *a_packed, void const *b_packed, nk_f64_t limit,
                                                           unsigned lane) {
    nk_f64_t flip_distances[8];
    nk_hausdorff_flips_warp_packed_f32_rocm_(a_packed, b_packed, limit, lane, flip_distances);
    return nk_shapes_least_flip_f64_simt_(flip_distances);
}

/** The Hausdorff distances between two packs of F16 points under each of the 8 flips on a team of
 *  32 lanes, which @c nk_hausdorff_warp_packed_f16_rocm_ reduces. Distances up to @p limit are
 *  exact; past it, the team may stop early, leaving each flip's lower bound, the least of which
 *  also lies past @p limit . */
NUMKONG_DEVICE void nk_hausdorff_flips_warp_packed_f16_rocm_(void const *a_packed, void const *b_packed, nk_f32_t limit,
                                                             unsigned lane, nk_f32_t *flip_distances) {
    nk_size_t const a_count = ((nk_shapes_packed_header_f32_t const *)a_packed)->points;
    nk_size_t const b_count = ((nk_shapes_packed_header_f32_t const *)b_packed)->points;
    if (!a_count || !b_count) {
        for (int flip = 0; flip != 8; ++flip) flip_distances[flip] = a_count == b_count ? 0 : NUMKONG_F32_INF;
        return;
    }
    nk_f32_t const bound = nk_hausdorff_bound_packed_f16_simt_(a_packed, b_packed);
    if (bound > limit) {
        for (int flip = 0; flip != 8; ++flip) flip_distances[flip] = bound;
        return;
    }
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
            if (least > limit_squared && __fsqrt_rn(least) > limit) {
                for (int flip = 0; flip != 8; ++flip) flip_distances[flip] = __fsqrt_rn(farthest[flip]);
                return;
            }
        }
    }
    for (int flip = 0; flip != 8; ++flip) flip_distances[flip] = __fsqrt_rn(farthest[flip]);
}

/** The Hausdorff distance between two packs of F16 points on a team of 32 lanes. Every lane calls
 *  with the same packs and its own @p lane , the lanes split each cloud's points, and the team
 *  keeps the farthest nearest square. Distances up to @p limit are exact; past it, the team may
 *  return early, even before reading a point, with a lower bound that also lies past @p limit . */
NUMKONG_DEVICE nk_f32_t nk_hausdorff_warp_packed_f16_rocm_(void const *a_packed, void const *b_packed, nk_f32_t limit,
                                                           unsigned lane) {
    nk_f32_t flip_distances[8];
    nk_hausdorff_flips_warp_packed_f16_rocm_(a_packed, b_packed, limit, lane, flip_distances);
    return nk_shapes_least_flip_f32_simt_(flip_distances);
}

#pragma endregion Chamfer and Hausdorff

#pragma region Memory Reach

/** Whether the device of @p stream reaches the pack tables, @p results of @p value_bytes each, the
 *  @c rows @c × @c columns rotations and the workspace. Packs follow their tables, as @c copy and
 *  @c UnifiedAllocator gather both into one block, and one query per pack would outweigh a pair. */
NUMKONG_INLINE nk_status_t nk_shapes_accessible_rocm_(void const *const *a_packed, nk_size_t rows,
                                                      void const *const *b_packed, nk_size_t columns,
                                                      void const *results, nk_size_t results_stride,
                                                      nk_size_t value_bytes, void const *rotations,
                                                      void const *workspace, nk_size_t workspace_bytes,
                                                      nk_stream_t stream) {
    int caller = 0;
    nk_status_t const entered = nk_device_enter_rocm_(stream, &caller);
    if (entered != nk_success_k) return entered;
    nk_size_t const rotations_bytes = rotations ? rows * columns * 9 * sizeof(nk_f32_t) : 0;
    int const reached = nk_memory_accessible_rocm_(a_packed, rows * sizeof(void const *)) &&
                        nk_memory_accessible_rocm_(b_packed, columns * sizeof(void const *)) &&
                        nk_memory_accessible_rocm_(results, (rows - 1) * results_stride + columns * value_bytes) &&
                        nk_memory_accessible_rocm_(rotations, rotations_bytes) &&
                        nk_memory_accessible_rocm_(workspace, workspace_bytes);
    nk_device_leave_rocm_(caller);
    return reached ? nk_success_k : nk_device_memory_mismatch_k;
}

#pragma endregion Memory Reach

#if NUMKONG_TARGET_ROCM
#pragma region Configuration

enum {

    /** Threads of a team on clouds up to @c nk_shapes_small_cloud_rocm_k points. */
    nk_shapes_small_team_rocm_k = 32,

    /** Threads of a team on larger clouds, a whole CDNA wavefront and two of RDNA and gfx1250, kept
     *  below CUDA's 128 until a device measures the wider reductions. */
    nk_shapes_large_team_rocm_k = 64,

    /** The largest cloud the smaller teams take, CUDA's until a device measures the crossover. */
    nk_shapes_small_cloud_rocm_k = 192,

    /** Threads of a block sorting pairs into size classes, one pair each per grid stride. */
    nk_shapes_classify_threads_rocm_k = 256,
};

/** The largest cloud a team of @p team threads holds in the device's shared memory, at
 *  @p point_bytes per coordinate. */
NUMKONG_INLINE nk_status_t nk_shapes_largest_supported_rocm_(unsigned team, nk_size_t point_bytes, nk_u32_t *largest,
                                                             nk_stream_t stream) {
    int shared_limit = 0;
    nk_status_t const status = nk_device_attribute_rocm_(hipDeviceAttributeMaxSharedMemoryPerBlock, &shared_limit,
                                                         stream);
    if (status != nk_success_k) return status;
    nk_u32_t low = 0, high = 1u << 16;
    while (low + 1 < high) {
        nk_u32_t const middle = (low + high) / 2;
        // The kernels' own shared variables take a few dozen bytes beside the dynamic ones
        if (nk_shapes_team_bytes_simt_(middle, middle, team, point_bytes) + 1024 <= (nk_size_t)shared_limit)
            low = middle;
        else high = middle;
    }
    *largest = low;
    return low ? nk_success_k : nk_unexpected_dimensions_k;
}

/** Zeroes the counters of the first @p used slots, @p stride bytes apart. */
NUMKONG_INLINE nk_status_t nk_shapes_zero_counters_rocm_(unsigned char *slots, nk_size_t stride, nk_size_t used,
                                                         nk_stream_t stream) {
    return hipMemset2DAsync(slots, stride, 0, 2 * sizeof(nk_u32_t), used, (hipStream_t)stream) == hipSuccess
               ? nk_success_k
               : nk_device_code_mismatch_k;
}

#pragma endregion Configuration

#pragma region Packs

static __global__ void __launch_bounds__(nk_shapes_classify_threads_rocm_k, 1)
    nk_shapes_classify_f32_rocm_kernel_(nk_shapes_transport_schedule_simt_t const schedule) {
    for (nk_size_t slot_index = blockIdx.x * (nk_size_t)blockDim.x + threadIdx.x; slot_index < schedule.pairs;
         slot_index += (nk_size_t)gridDim.x * blockDim.x) {
        nk_size_t const pair = schedule.pair_begin + slot_index;
        nk_u32_t const rows =
            (nk_u32_t)((nk_shapes_packed_header_f64_t const *)schedule.a_packed[pair / schedule.columns])->points;
        nk_u32_t const columns =
            (nk_u32_t)((nk_shapes_packed_header_f64_t const *)schedule.b_packed[pair % schedule.columns])->points;
        nk_shapes_classify_simt_(&schedule, slot_index, rows > columns ? rows : columns);
    }
}

static __global__ void __launch_bounds__(nk_shapes_classify_threads_rocm_k, 1)
    nk_shapes_classify_f16_rocm_kernel_(nk_shapes_transport_schedule_simt_t const schedule) {
    for (nk_size_t slot_index = blockIdx.x * (nk_size_t)blockDim.x + threadIdx.x; slot_index < schedule.pairs;
         slot_index += (nk_size_t)gridDim.x * blockDim.x) {
        nk_size_t const pair = schedule.pair_begin + slot_index;
        nk_u32_t const rows =
            ((nk_shapes_packed_header_f32_t const *)schedule.a_packed[pair / schedule.columns])->points;
        nk_u32_t const columns =
            ((nk_shapes_packed_header_f32_t const *)schedule.b_packed[pair % schedule.columns])->points;
        nk_shapes_classify_simt_(&schedule, slot_index, rows > columns ? rows : columns);
    }
}

#pragma endregion Packs

#pragma region Wasserstein Procrustes

/** One team per flip of each pair, the walk inlined once over shared memory and once over global
 *  memory, so that the block tier addresses its state as shared. */
static __global__ void __launch_bounds__(nk_shapes_large_team_rocm_k, 1)
    nk_wassersteinprocrustes_packed_f32_rocm_kernel_(nk_shapes_transport_schedule_simt_t const schedule) {
    extern __shared__ nk_f64_t nk_shapes_shared_rocm_[];
    __shared__ nk_f64_t shared_rotation[9];
    __shared__ int last;
    nk_size_t const items = (schedule.class_slots ? (nk_size_t)*schedule.class_count : schedule.pairs) * 8;
    nk_shapes_team_simt_t team;
    if (!nk_shapes_find_team_simt_(&schedule, items, &team)) return;
    if (team.region)
        nk_wassersteinprocrustes_walk_f32_simt_(&schedule, &team, items, nk_shapes_shared_rocm_,
                                                team.region + nk_shapes_meeting_bytes_simt_k, shared_rotation, &last);
    else
        nk_wassersteinprocrustes_walk_f32_simt_(
            &schedule, &team, items, nk_shapes_shared_rocm_,
            (char *)nk_shapes_shared_rocm_ + nk_shapes_scratch_bytes_simt_(blockDim.x), shared_rotation, &last);
}

static __global__ void __launch_bounds__(nk_shapes_large_team_rocm_k, 1)
    nk_wassersteinprocrustes_packed_f16_rocm_kernel_(nk_shapes_transport_schedule_simt_t const schedule) {
    extern __shared__ nk_f64_t nk_shapes_shared_rocm_[];
    __shared__ nk_f32_t shared_rotation[9];
    __shared__ int last;
    nk_size_t const items = (schedule.class_slots ? (nk_size_t)*schedule.class_count : schedule.pairs) * 8;
    nk_shapes_team_simt_t team;
    if (!nk_shapes_find_team_simt_(&schedule, items, &team)) return;
    if (team.region)
        nk_wassersteinprocrustes_walk_f16_simt_(&schedule, &team, items, nk_shapes_shared_rocm_,
                                                team.region + nk_shapes_meeting_bytes_simt_k, shared_rotation, &last);
    else
        nk_wassersteinprocrustes_walk_f16_simt_(
            &schedule, &team, items, nk_shapes_shared_rocm_,
            (char *)nk_shapes_shared_rocm_ + nk_shapes_scratch_bytes_simt_(blockDim.x), shared_rotation, &last);
}

/** Walks every pair in chunks of as many pairs as the workspace has slots. A light launch lists the
 *  chunk's pairs by the class of their larger cloud, then one launch per class, sized for its
 *  largest cloud, solves the pairs it reads from its list, so no call waits on the device. Classes
 *  that fit a block's shared memory take a block per flip; larger ones take teams of blocks over
 *  the workspace, up to clouds of @p largest points, and record any pair above as unsolvable. */
NUMKONG_INLINE nk_status_t nk_shapes_transport_launch_rocm_(void const *kernel, void const *classify_kernel,
                                                            nk_shapes_transport_schedule_simt_t schedule,
                                                            nk_size_t point_bytes, nk_size_t largest, void *workspace,
                                                            nk_size_t workspace_bytes, nk_stream_t stream) {
    if ((nk_size_t)workspace & (nk_default_alignment_k - 1)) return nk_misaligned_k;
    if (!largest) return nk_unexpected_dimensions_k;
    nk_size_t const team_stride = nk_shapes_team_stride_simt_(largest, point_bytes);
    nk_size_t const teams = nk_shapes_teams_simt_(largest, point_bytes);
    nk_size_t const head_bytes = nk_shapes_workspace_bytes_simt_(0, largest, point_bytes);
    nk_size_t const per_slot = nk_shapes_workspace_bytes_simt_(1, largest, point_bytes) - head_bytes;
    nk_size_t slots = workspace_bytes > head_bytes ? (workspace_bytes - head_bytes) / per_slot : 0;
    slots = slots < nk_shapes_pair_slots_simt_k ? slots : nk_shapes_pair_slots_simt_k;
    if (!slots) return nk_unexpected_dimensions_k;
    nk_u32_t supported = 0;
    nk_status_t status = nk_shapes_largest_supported_rocm_(nk_shapes_large_team_rocm_k, point_bytes, &supported,
                                                           stream);
    if (status != nk_success_k) return status;
    nk_size_t const pairs = schedule.rows * schedule.columns;
    unsigned char *team_states = (unsigned char *)workspace;
    nk_u32_t *counts = (nk_u32_t *)(team_states + teams * team_stride), *lists = counts + nk_shapes_classes_simt_k;
    schedule.slots = (unsigned char *)(lists + nk_shapes_classes_simt_k * slots);
    schedule.slot_stride = nk_shapes_slot_bytes_simt_(), schedule.gromov = 0;
    schedule.class_counts = counts, schedule.class_limit = (nk_u32_t)largest;
    schedule.team_stride = team_stride, schedule.teams = teams;
    status = nk_shapes_zero_counters_rocm_(schedule.slots, schedule.slot_stride, pairs < slots ? pairs : slots, stream);
    if (status == nk_success_k && largest > supported)
        status = nk_shapes_zero_counters_rocm_(team_states, team_stride, teams, stream);
    for (nk_size_t begin = 0; begin < pairs && status == nk_success_k; begin += slots) {
        schedule.pair_begin = begin, schedule.pairs = pairs - begin < slots ? pairs - begin : slots;
        schedule.class_count = counts, schedule.class_slots = lists, schedule.class_ceiling = (nk_u32_t)largest;
        schedule.team_states = NUMKONG_NULL;
        if (hipMemsetAsync(counts, 0, nk_shapes_classes_simt_k * sizeof(nk_u32_t), (hipStream_t)stream) != hipSuccess)
            return nk_device_code_mismatch_k;
        void *classify_arguments[1] = {(void *)&schedule};
        nk_size_t const classify_blocks = nk_size_divide_round_up_(schedule.pairs, nk_shapes_classify_threads_rocm_k);
        status = nk_launch_rocm_(classify_kernel, classify_blocks < 1024 ? classify_blocks : 1024,
                                 nk_shapes_classify_threads_rocm_k, classify_arguments, 0, stream);
        int last = 0;
        for (nk_u32_t class_index = 0; !last && status == nk_success_k; ++class_index) {
            nk_u32_t const ceiling = nk_shapes_class_ceiling_simt_(class_index);
            last = ceiling >= largest || class_index + 1 == nk_shapes_classes_simt_k;
            schedule.class_ceiling = last ? (nk_u32_t)largest : ceiling;
            schedule.class_count = counts + class_index, schedule.class_slots = lists + class_index * schedule.pairs;
            if (schedule.class_ceiling > supported) {
                // Past one block's shared memory, teams of blocks keep their state in the workspace
                schedule.team_states = team_states;
                status = nk_launch_cooperative_rocm_(kernel, nk_shapes_large_team_rocm_k,
                                                     nk_shapes_scratch_bytes_simt_(nk_shapes_large_team_rocm_k),
                                                     &schedule, stream);
                continue;
            }
            unsigned const team = schedule.class_ceiling > nk_shapes_small_cloud_rocm_k ? nk_shapes_large_team_rocm_k
                                                                                        : nk_shapes_small_team_rocm_k;
            nk_size_t const shared_bytes = nk_shapes_team_bytes_simt_(schedule.class_ceiling, schedule.class_ceiling,
                                                                      team, point_bytes);
            status = nk_launch_resident_rocm_(kernel, team, shared_bytes, shared_bytes, schedule.pairs * 8, &schedule,
                                              stream);
        }
    }
    return status;
}

NUMKONG_API nk_status_t nk_wassersteinprocrustes_workspace_size_f32_rocm(nk_size_t row_points, nk_size_t column_points,
                                                                         nk_size_t *bytes) {
    nk_size_t const largest = row_points > column_points ? row_points : column_points;
    *bytes = nk_shapes_workspace_bytes_simt_(nk_shapes_pair_slots_simt_k, largest ? largest : 1, sizeof(nk_f64_t));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_wassersteinprocrustes_workspace_size_f16_rocm(nk_size_t row_points, nk_size_t column_points,
                                                                         nk_size_t *bytes) {
    nk_size_t const largest = row_points > column_points ? row_points : column_points;
    *bytes = nk_shapes_workspace_bytes_simt_(nk_shapes_pair_slots_simt_k, largest ? largest : 1, sizeof(nk_f32_t));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f32_rocm(void const *const *a_packed, nk_size_t rows,
                                                                 void const *const *b_packed, nk_size_t columns,
                                                                 nk_f64_t *results, nk_size_t results_stride,
                                                                 nk_f32_t *rotations, void *workspace,
                                                                 nk_size_t workspace_bytes, nk_stream_t stream) {
    if (!rows || !columns) return nk_success_k;
    if (((nk_size_t)results | results_stride) & (sizeof(nk_f64_t) - 1) || (nk_size_t)rotations & 3)
        return nk_misaligned_k;
    nk_status_t const reached = nk_shapes_accessible_rocm_(a_packed, rows, b_packed, columns, results, results_stride,
                                                           sizeof(nk_f64_t), rotations, workspace, workspace_bytes,
                                                           stream);
    if (reached != nk_success_k) return reached;
    nk_shapes_transport_schedule_simt_t schedule;
    schedule.a_packed = a_packed, schedule.b_packed = b_packed, schedule.results = results;
    schedule.rotations = rotations, schedule.rows = rows, schedule.columns = columns;
    schedule.results_stride = results_stride;
    return nk_shapes_transport_launch_rocm_(
        (void const *)&nk_wassersteinprocrustes_packed_f32_rocm_kernel_,
        (void const *)&nk_shapes_classify_f32_rocm_kernel_, schedule, sizeof(nk_f64_t),
        nk_shapes_workspace_bound_simt_(workspace_bytes, sizeof(nk_f64_t)), workspace, workspace_bytes, stream);
}

NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f16_rocm(void const *const *a_packed, nk_size_t rows,
                                                                 void const *const *b_packed, nk_size_t columns,
                                                                 nk_f32_t *results, nk_size_t results_stride,
                                                                 nk_f32_t *rotations, void *workspace,
                                                                 nk_size_t workspace_bytes, nk_stream_t stream) {
    if (!rows || !columns) return nk_success_k;
    if (((nk_size_t)results | results_stride) & (sizeof(nk_f32_t) - 1) || (nk_size_t)rotations & 3)
        return nk_misaligned_k;
    nk_status_t const reached = nk_shapes_accessible_rocm_(a_packed, rows, b_packed, columns, results, results_stride,
                                                           sizeof(nk_f32_t), rotations, workspace, workspace_bytes,
                                                           stream);
    if (reached != nk_success_k) return reached;
    nk_shapes_transport_schedule_simt_t schedule;
    schedule.a_packed = a_packed, schedule.b_packed = b_packed, schedule.results = results;
    schedule.rotations = rotations, schedule.rows = rows, schedule.columns = columns;
    schedule.results_stride = results_stride;
    return nk_shapes_transport_launch_rocm_(
        (void const *)&nk_wassersteinprocrustes_packed_f16_rocm_kernel_,
        (void const *)&nk_shapes_classify_f16_rocm_kernel_, schedule, sizeof(nk_f32_t),
        nk_shapes_workspace_bound_simt_(workspace_bytes, sizeof(nk_f32_t)), workspace, workspace_bytes, stream);
}

#pragma endregion Wasserstein Procrustes

#pragma region Gromov Wasserstein

/** One team per start of each pair, the walk inlined once over shared memory and once over global
 *  memory, so that the block tier addresses its network as shared. */
static __global__ void __launch_bounds__(nk_shapes_large_team_rocm_k, 1)
    nk_gromovwassersteins_packed_f32_rocm_kernel_(nk_shapes_transport_schedule_simt_t const schedule) {
    extern __shared__ nk_f64_t nk_shapes_shared_rocm_[];
    __shared__ nk_f64_t shared_fit[9];
    __shared__ int last;
    nk_shapes_team_simt_t team;
    if (!nk_shapes_find_team_simt_(&schedule, schedule.pairs * 3, &team)) return;
    if (team.region)
        nk_gromovwassersteins_walk_f32_simt_(&schedule, &team, nk_shapes_shared_rocm_,
                                             team.region + nk_shapes_meeting_bytes_simt_k, shared_fit, &last);
    else
        nk_gromovwassersteins_walk_f32_simt_(&schedule, &team, nk_shapes_shared_rocm_,
                                             (char *)nk_shapes_shared_rocm_ + nk_shapes_scratch_bytes_simt_(blockDim.x),
                                             shared_fit, &last);
}

static __global__ void __launch_bounds__(nk_shapes_large_team_rocm_k, 1)
    nk_gromovwassersteins_packed_f16_rocm_kernel_(nk_shapes_transport_schedule_simt_t const schedule) {
    extern __shared__ nk_f64_t nk_shapes_shared_rocm_[];
    __shared__ nk_f64_t shared_fit[9];
    __shared__ int last;
    nk_shapes_team_simt_t team;
    if (!nk_shapes_find_team_simt_(&schedule, schedule.pairs * 3, &team)) return;
    if (team.region)
        nk_gromovwassersteins_walk_f16_simt_(&schedule, &team, nk_shapes_shared_rocm_,
                                             team.region + nk_shapes_meeting_bytes_simt_k, shared_fit, &last);
    else
        nk_gromovwassersteins_walk_f16_simt_(&schedule, &team, nk_shapes_shared_rocm_,
                                             (char *)nk_shapes_shared_rocm_ + nk_shapes_scratch_bytes_simt_(blockDim.x),
                                             shared_fit, &last);
}

/** Walks every pair in chunks of as many pairs as the workspace has slots: the flips of a chunk,
 *  then its starts, for clouds of up to @p largest points, recording any pair above as unsolvable.
 *  Clouds that fit a block's shared memory take a block per flip or start; larger ones take teams
 *  of blocks over the workspace. */
NUMKONG_INLINE nk_status_t nk_gromov_launch_rocm_(void const *flips_kernel, void const *starts_kernel,
                                                  nk_shapes_transport_schedule_simt_t schedule, nk_size_t point_bytes,
                                                  nk_size_t largest, void *workspace, nk_size_t workspace_bytes,
                                                  nk_stream_t stream) {
    if ((nk_size_t)workspace & (nk_default_alignment_k - 1)) return nk_misaligned_k;
    if (!largest || nk_gromov_workspace_bytes_simt_(largest, point_bytes) > workspace_bytes)
        return nk_unexpected_dimensions_k;
    nk_u32_t supported = 0;
    nk_status_t status = nk_shapes_largest_supported_rocm_(nk_shapes_large_team_rocm_k, point_bytes, &supported,
                                                           stream);
    if (status != nk_success_k) return status;
    nk_size_t const slots = nk_gromov_slots_simt_(largest), pairs = schedule.rows * schedule.columns;
    schedule.slots = (unsigned char *)workspace, schedule.slot_stride = nk_gromov_slot_bytes_simt_(largest);
    schedule.plans_offset = nk_shapes_slot_plans_simt_k, schedule.gromov_offset = nk_gromov_dense_offset_simt_(largest);
    schedule.gromov_stride = nk_shapes_gromov_bytes_simt_(largest);
    schedule.class_counts = NUMKONG_NULL, schedule.class_count = NUMKONG_NULL, schedule.class_slots = NUMKONG_NULL;
    schedule.class_ceiling = (nk_u32_t)largest, schedule.class_limit = (nk_u32_t)largest, schedule.gromov = 1;
    schedule.team_states = NUMKONG_NULL, schedule.team_stride = nk_shapes_team_stride_simt_(largest, point_bytes);
    schedule.teams = nk_shapes_teams_simt_(largest, point_bytes);
    unsigned const flips_team = largest > nk_shapes_small_cloud_rocm_k ? nk_shapes_large_team_rocm_k
                                                                       : nk_shapes_small_team_rocm_k;
    nk_size_t const flips_shared_bytes = nk_shapes_team_bytes_simt_(largest, largest, flips_team, point_bytes);
    // The starts' dense steps want more threads than the flips' scans
    unsigned const starts_team = nk_shapes_large_team_rocm_k;
    nk_size_t const starts_shared_bytes = nk_shapes_team_bytes_simt_(largest, largest, starts_team, 0);
    status = nk_shapes_zero_counters_rocm_(schedule.slots, schedule.slot_stride, pairs < slots ? pairs : slots, stream);
    if (status == nk_success_k && largest > supported) {
        // Past one block's shared memory, teams of blocks keep their state after the slots
        schedule.team_states = (unsigned char *)workspace +
                               nk_size_round_up_to_multiple_(slots * schedule.slot_stride, 256);
        status = nk_shapes_zero_counters_rocm_(schedule.team_states, schedule.team_stride, schedule.teams, stream);
    }
    for (nk_size_t begin = 0; begin < pairs && status == nk_success_k; begin += slots) {
        schedule.pair_begin = begin, schedule.pairs = pairs - begin < slots ? pairs - begin : slots;
        if (schedule.team_states) {
            nk_size_t const scratch_bytes = nk_shapes_scratch_bytes_simt_(nk_shapes_large_team_rocm_k);
            status = nk_launch_cooperative_rocm_(flips_kernel, nk_shapes_large_team_rocm_k, scratch_bytes, &schedule,
                                                 stream);
            if (status == nk_success_k)
                status = nk_launch_cooperative_rocm_(starts_kernel, nk_shapes_large_team_rocm_k, scratch_bytes,
                                                     &schedule, stream);
            continue;
        }
        status = nk_launch_resident_rocm_(flips_kernel, flips_team, flips_shared_bytes, flips_shared_bytes,
                                          schedule.pairs * 8, &schedule, stream);
        if (status == nk_success_k)
            status = nk_launch_resident_rocm_(starts_kernel, starts_team, starts_shared_bytes, starts_shared_bytes,
                                              schedule.pairs * 3, &schedule, stream);
    }
    return status;
}

NUMKONG_API nk_status_t nk_gromovwassersteins_workspace_size_f32_rocm(nk_size_t row_points, nk_size_t column_points,
                                                                      nk_size_t *bytes) {
    *bytes = nk_gromov_workspace_bytes_simt_(row_points > column_points ? row_points : column_points, sizeof(nk_f64_t));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_gromovwassersteins_workspace_size_f16_rocm(nk_size_t row_points, nk_size_t column_points,
                                                                      nk_size_t *bytes) {
    *bytes = nk_gromov_workspace_bytes_simt_(row_points > column_points ? row_points : column_points, sizeof(nk_f32_t));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f32_rocm(void const *const *a_packed, nk_size_t rows,
                                                              void const *const *b_packed, nk_size_t columns,
                                                              nk_f64_t *results, nk_size_t results_stride,
                                                              nk_f32_t *rotations, void *workspace,
                                                              nk_size_t workspace_bytes, nk_stream_t stream) {
    if (!rows || !columns) return nk_success_k;
    if (((nk_size_t)results | results_stride) & (sizeof(nk_f64_t) - 1) || (nk_size_t)rotations & 3)
        return nk_misaligned_k;
    nk_status_t const reached = nk_shapes_accessible_rocm_(a_packed, rows, b_packed, columns, results, results_stride,
                                                           sizeof(nk_f64_t), rotations, workspace, workspace_bytes,
                                                           stream);
    if (reached != nk_success_k) return reached;
    nk_shapes_transport_schedule_simt_t schedule;
    schedule.a_packed = a_packed, schedule.b_packed = b_packed, schedule.results = results;
    schedule.rotations = rotations, schedule.rows = rows, schedule.columns = columns;
    schedule.results_stride = results_stride;
    return nk_gromov_launch_rocm_((void const *)&nk_wassersteinprocrustes_packed_f32_rocm_kernel_,
                                  (void const *)&nk_gromovwassersteins_packed_f32_rocm_kernel_, schedule,
                                  sizeof(nk_f64_t), nk_gromov_workspace_bound_simt_(workspace_bytes, sizeof(nk_f64_t)),
                                  workspace, workspace_bytes, stream);
}

NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f16_rocm(void const *const *a_packed, nk_size_t rows,
                                                              void const *const *b_packed, nk_size_t columns,
                                                              nk_f32_t *results, nk_size_t results_stride,
                                                              nk_f32_t *rotations, void *workspace,
                                                              nk_size_t workspace_bytes, nk_stream_t stream) {
    if (!rows || !columns) return nk_success_k;
    if (((nk_size_t)results | results_stride) & (sizeof(nk_f32_t) - 1) || (nk_size_t)rotations & 3)
        return nk_misaligned_k;
    nk_status_t const reached = nk_shapes_accessible_rocm_(a_packed, rows, b_packed, columns, results, results_stride,
                                                           sizeof(nk_f32_t), rotations, workspace, workspace_bytes,
                                                           stream);
    if (reached != nk_success_k) return reached;
    nk_shapes_transport_schedule_simt_t schedule;
    schedule.a_packed = a_packed, schedule.b_packed = b_packed, schedule.results = results;
    schedule.rotations = rotations, schedule.rows = rows, schedule.columns = columns;
    schedule.results_stride = results_stride;
    return nk_gromov_launch_rocm_((void const *)&nk_wassersteinprocrustes_packed_f16_rocm_kernel_,
                                  (void const *)&nk_gromovwassersteins_packed_f16_rocm_kernel_, schedule,
                                  sizeof(nk_f32_t), nk_gromov_workspace_bound_simt_(workspace_bytes, sizeof(nk_f32_t)),
                                  workspace, workspace_bytes, stream);
}

#pragma endregion Gromov Wasserstein

#pragma region Chamfer and Hausdorff Schedules

static __global__ void __launch_bounds__(nk_shapes_lanes_simt_k, 1)
    nk_chamfers_packed_f32_rocm_kernel_(nk_shapes_schedule_simt_t const schedule) {
    nk_f64_t flip_distances[8];
    for (nk_size_t pair = blockIdx.x; pair < schedule.rows * schedule.columns; pair += gridDim.x) {
        nk_size_t const row = pair / schedule.columns, column = pair % schedule.columns;
        nk_chamfer_flips_warp_packed_f32_rocm_(schedule.a_packed[row], schedule.b_packed[column],
                                               (nk_f64_t)schedule.limit, threadIdx.x, flip_distances);
        if (threadIdx.x == 0) nk_shapes_write_best_flip_f64_simt_(&schedule, row, column, flip_distances);
    }
}

static __global__ void __launch_bounds__(nk_shapes_lanes_simt_k, 1)
    nk_hausdorffs_packed_f32_rocm_kernel_(nk_shapes_schedule_simt_t const schedule) {
    nk_f64_t flip_distances[8];
    for (nk_size_t pair = blockIdx.x; pair < schedule.rows * schedule.columns; pair += gridDim.x) {
        nk_size_t const row = pair / schedule.columns, column = pair % schedule.columns;
        nk_hausdorff_flips_warp_packed_f32_rocm_(schedule.a_packed[row], schedule.b_packed[column],
                                                 (nk_f64_t)schedule.limit, threadIdx.x, flip_distances);
        if (threadIdx.x == 0) nk_shapes_write_best_flip_f64_simt_(&schedule, row, column, flip_distances);
    }
}

static __global__ void __launch_bounds__(nk_shapes_lanes_simt_k, 1)
    nk_chamfers_packed_f16_rocm_kernel_(nk_shapes_schedule_simt_t const schedule) {
    nk_f32_t flip_distances[8];
    for (nk_size_t pair = blockIdx.x; pair < schedule.rows * schedule.columns; pair += gridDim.x) {
        nk_size_t const row = pair / schedule.columns, column = pair % schedule.columns;
        nk_chamfer_flips_warp_packed_f16_rocm_(schedule.a_packed[row], schedule.b_packed[column],
                                               (nk_f32_t)schedule.limit, threadIdx.x, flip_distances);
        if (threadIdx.x == 0) nk_shapes_write_best_flip_f32_simt_(&schedule, row, column, flip_distances);
    }
}

static __global__ void __launch_bounds__(nk_shapes_lanes_simt_k, 1)
    nk_hausdorffs_packed_f16_rocm_kernel_(nk_shapes_schedule_simt_t const schedule) {
    nk_f32_t flip_distances[8];
    for (nk_size_t pair = blockIdx.x; pair < schedule.rows * schedule.columns; pair += gridDim.x) {
        nk_size_t const row = pair / schedule.columns, column = pair % schedule.columns;
        nk_hausdorff_flips_warp_packed_f16_rocm_(schedule.a_packed[row], schedule.b_packed[column],
                                                 (nk_f32_t)schedule.limit, threadIdx.x, flip_distances);
        if (threadIdx.x == 0) nk_shapes_write_best_flip_f32_simt_(&schedule, row, column, flip_distances);
    }
}

NUMKONG_API nk_status_t nk_chamfers_workspace_size_f32_rocm(nk_size_t row_points, nk_size_t column_points,
                                                            nk_size_t *bytes) {
    nk_unused_(row_points), nk_unused_(column_points);
    *bytes = 0;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_chamfers_workspace_size_f16_rocm(nk_size_t row_points, nk_size_t column_points,
                                                            nk_size_t *bytes) {
    nk_unused_(row_points), nk_unused_(column_points);
    *bytes = 0;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_chamfers_packed_f32_rocm(void const *const *a_packed, nk_size_t rows,
                                                    void const *const *b_packed, nk_size_t columns, nk_f64_t *results,
                                                    nk_size_t results_stride, nk_f32_t *rotations,
                                                    nk_f64_t const *limit, void *workspace, nk_size_t workspace_bytes,
                                                    nk_stream_t stream) {
    nk_unused_(workspace), nk_unused_(workspace_bytes);
    if (!rows || !columns) return nk_success_k;
    if (((nk_size_t)results | results_stride) & (sizeof(nk_f64_t) - 1) || (nk_size_t)rotations & 3)
        return nk_misaligned_k;
    nk_status_t const reached = nk_shapes_accessible_rocm_(a_packed, rows, b_packed, columns, results, results_stride,
                                                           sizeof(nk_f64_t), rotations, NUMKONG_NULL, 0, stream);
    if (reached != nk_success_k) return reached;
    nk_shapes_schedule_simt_t schedule = nk_shapes_schedule_simt_(a_packed, rows, b_packed, columns, results,
                                                                  results_stride, rotations, *limit);
    return nk_launch_resident_rocm_((void const *)nk_chamfers_packed_f32_rocm_kernel_, nk_shapes_lanes_simt_k, 0, 0,
                                    rows * columns, &schedule, stream);
}

NUMKONG_API nk_status_t nk_hausdorffs_packed_f32_rocm(void const *const *a_packed, nk_size_t rows,
                                                      void const *const *b_packed, nk_size_t columns, nk_f64_t *results,
                                                      nk_size_t results_stride, nk_f32_t *rotations,
                                                      nk_f64_t const *limit, nk_stream_t stream) {
    if (!rows || !columns) return nk_success_k;
    if (((nk_size_t)results | results_stride) & (sizeof(nk_f64_t) - 1) || (nk_size_t)rotations & 3)
        return nk_misaligned_k;
    nk_status_t const reached = nk_shapes_accessible_rocm_(a_packed, rows, b_packed, columns, results, results_stride,
                                                           sizeof(nk_f64_t), rotations, NUMKONG_NULL, 0, stream);
    if (reached != nk_success_k) return reached;
    nk_shapes_schedule_simt_t schedule = nk_shapes_schedule_simt_(a_packed, rows, b_packed, columns, results,
                                                                  results_stride, rotations, *limit);
    return nk_launch_resident_rocm_((void const *)nk_hausdorffs_packed_f32_rocm_kernel_, nk_shapes_lanes_simt_k, 0, 0,
                                    rows * columns, &schedule, stream);
}

NUMKONG_API nk_status_t nk_chamfers_packed_f16_rocm(void const *const *a_packed, nk_size_t rows,
                                                    void const *const *b_packed, nk_size_t columns, nk_f32_t *results,
                                                    nk_size_t results_stride, nk_f32_t *rotations,
                                                    nk_f32_t const *limit, void *workspace, nk_size_t workspace_bytes,
                                                    nk_stream_t stream) {
    nk_unused_(workspace), nk_unused_(workspace_bytes);
    if (!rows || !columns) return nk_success_k;
    if (((nk_size_t)results | results_stride) & (sizeof(nk_f32_t) - 1) || (nk_size_t)rotations & 3)
        return nk_misaligned_k;
    nk_status_t const reached = nk_shapes_accessible_rocm_(a_packed, rows, b_packed, columns, results, results_stride,
                                                           sizeof(nk_f32_t), rotations, NUMKONG_NULL, 0, stream);
    if (reached != nk_success_k) return reached;
    nk_shapes_schedule_simt_t schedule = nk_shapes_schedule_simt_(a_packed, rows, b_packed, columns, results,
                                                                  results_stride, rotations, *limit);
    return nk_launch_resident_rocm_((void const *)nk_chamfers_packed_f16_rocm_kernel_, nk_shapes_lanes_simt_k, 0, 0,
                                    rows * columns, &schedule, stream);
}

NUMKONG_API nk_status_t nk_hausdorffs_packed_f16_rocm(void const *const *a_packed, nk_size_t rows,
                                                      void const *const *b_packed, nk_size_t columns, nk_f32_t *results,
                                                      nk_size_t results_stride, nk_f32_t *rotations,
                                                      nk_f32_t const *limit, nk_stream_t stream) {
    if (!rows || !columns) return nk_success_k;
    if (((nk_size_t)results | results_stride) & (sizeof(nk_f32_t) - 1) || (nk_size_t)rotations & 3)
        return nk_misaligned_k;
    nk_status_t const reached = nk_shapes_accessible_rocm_(a_packed, rows, b_packed, columns, results, results_stride,
                                                           sizeof(nk_f32_t), rotations, NUMKONG_NULL, 0, stream);
    if (reached != nk_success_k) return reached;
    nk_shapes_schedule_simt_t schedule = nk_shapes_schedule_simt_(a_packed, rows, b_packed, columns, results,
                                                                  results_stride, rotations, *limit);
    return nk_launch_resident_rocm_((void const *)nk_hausdorffs_packed_f16_rocm_kernel_, nk_shapes_lanes_simt_k, 0, 0,
                                    rows * columns, &schedule, stream);
}

#pragma endregion Chamfer and Hausdorff Schedules

#pragma region RMSD

static __global__ void __launch_bounds__(nk_rmsds_threads_simt_k, 1)
    nk_rmsds_packed_f64_rocm_kernel_(nk_shapes_schedule_simt_t const schedule) {
    nk_size_t const pairs = schedule.rows * schedule.columns;
    for (nk_size_t pair = blockIdx.x * (nk_size_t)blockDim.x + threadIdx.x; pair < pairs;
         pair += (nk_size_t)gridDim.x * blockDim.x) {
        nk_size_t const row = pair / schedule.columns, column = pair % schedule.columns;
        nk_f64_t *results_row = (nk_f64_t *)((char *)schedule.results + row * schedule.results_stride);
        results_row[column] = nk_rmsds_pair_f64_simt_(schedule.a_packed[row], schedule.b_packed[column]);
    }
}

static __global__ void __launch_bounds__(nk_rmsds_threads_simt_k, 1)
    nk_rmsds_packed_f32_rocm_kernel_(nk_shapes_schedule_simt_t const schedule) {
    nk_size_t const pairs = schedule.rows * schedule.columns;
    for (nk_size_t pair = blockIdx.x * (nk_size_t)blockDim.x + threadIdx.x; pair < pairs;
         pair += (nk_size_t)gridDim.x * blockDim.x) {
        nk_size_t const row = pair / schedule.columns, column = pair % schedule.columns;
        nk_f64_t *results_row = (nk_f64_t *)((char *)schedule.results + row * schedule.results_stride);
        results_row[column] = nk_rmsds_pair_f32_simt_(schedule.a_packed[row], schedule.b_packed[column]);
    }
}

static __global__ void __launch_bounds__(nk_rmsds_threads_simt_k, 1)
    nk_rmsds_packed_f16_rocm_kernel_(nk_shapes_schedule_simt_t const schedule) {
    nk_size_t const pairs = schedule.rows * schedule.columns;
    for (nk_size_t pair = blockIdx.x * (nk_size_t)blockDim.x + threadIdx.x; pair < pairs;
         pair += (nk_size_t)gridDim.x * blockDim.x) {
        nk_size_t const row = pair / schedule.columns, column = pair % schedule.columns;
        nk_f32_t *results_row = (nk_f32_t *)((char *)schedule.results + row * schedule.results_stride);
        results_row[column] = nk_rmsds_pair_f16_simt_(schedule.a_packed[row], schedule.b_packed[column]);
    }
}

NUMKONG_API nk_status_t nk_rmsds_packed_f64_rocm(void const *const *a_packed, nk_size_t rows,
                                                 void const *const *b_packed, nk_size_t columns, nk_f64_t *results,
                                                 nk_size_t results_stride, nk_stream_t stream) {
    if (!rows || !columns) return nk_success_k;
    if (((nk_size_t)results | results_stride) & (sizeof(nk_f64_t) - 1)) return nk_misaligned_k;
    nk_status_t const reached = nk_shapes_accessible_rocm_(a_packed, rows, b_packed, columns, results, results_stride,
                                                           sizeof(nk_f64_t), NUMKONG_NULL, NUMKONG_NULL, 0, stream);
    if (reached != nk_success_k) return reached;
    nk_shapes_schedule_simt_t schedule = nk_shapes_schedule_simt_(a_packed, rows, b_packed, columns, results,
                                                                  results_stride, NUMKONG_NULL, 0);
    return nk_launch_resident_rocm_((void const *)nk_rmsds_packed_f64_rocm_kernel_, nk_rmsds_threads_simt_k, 0, 0,
                                    nk_size_divide_round_up_(rows * columns, nk_rmsds_threads_simt_k), &schedule,
                                    stream);
}

NUMKONG_API nk_status_t nk_rmsds_packed_f32_rocm(void const *const *a_packed, nk_size_t rows,
                                                 void const *const *b_packed, nk_size_t columns, nk_f64_t *results,
                                                 nk_size_t results_stride, nk_stream_t stream) {
    if (!rows || !columns) return nk_success_k;
    if (((nk_size_t)results | results_stride) & (sizeof(nk_f64_t) - 1)) return nk_misaligned_k;
    nk_status_t const reached = nk_shapes_accessible_rocm_(a_packed, rows, b_packed, columns, results, results_stride,
                                                           sizeof(nk_f64_t), NUMKONG_NULL, NUMKONG_NULL, 0, stream);
    if (reached != nk_success_k) return reached;
    nk_shapes_schedule_simt_t schedule = nk_shapes_schedule_simt_(a_packed, rows, b_packed, columns, results,
                                                                  results_stride, NUMKONG_NULL, 0);
    return nk_launch_resident_rocm_((void const *)nk_rmsds_packed_f32_rocm_kernel_, nk_rmsds_threads_simt_k, 0, 0,
                                    nk_size_divide_round_up_(rows * columns, nk_rmsds_threads_simt_k), &schedule,
                                    stream);
}

NUMKONG_API nk_status_t nk_rmsds_packed_f16_rocm(void const *const *a_packed, nk_size_t rows,
                                                 void const *const *b_packed, nk_size_t columns, nk_f32_t *results,
                                                 nk_size_t results_stride, nk_stream_t stream) {
    if (!rows || !columns) return nk_success_k;
    if (((nk_size_t)results | results_stride) & (sizeof(nk_f32_t) - 1)) return nk_misaligned_k;
    nk_status_t const reached = nk_shapes_accessible_rocm_(a_packed, rows, b_packed, columns, results, results_stride,
                                                           sizeof(nk_f32_t), NUMKONG_NULL, NUMKONG_NULL, 0, stream);
    if (reached != nk_success_k) return reached;
    nk_shapes_schedule_simt_t schedule = nk_shapes_schedule_simt_(a_packed, rows, b_packed, columns, results,
                                                                  results_stride, NUMKONG_NULL, 0);
    return nk_launch_resident_rocm_((void const *)nk_rmsds_packed_f16_rocm_kernel_, nk_rmsds_threads_simt_k, 0, 0,
                                    nk_size_divide_round_up_(rows * columns, nk_rmsds_threads_simt_k), &schedule,
                                    stream);
}

#pragma endregion RMSD
#endif // NUMKONG_TARGET_ROCM

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_ROCM_
#endif // NUMKONG_SHAPES_ROCM_CUH
