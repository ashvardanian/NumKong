/**
 *  @file include/numkong/shapes/cuda.cuh
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief Many-to-many optimal transport between 3D point clouds on the SIMT cores of every CUDA
 *      device, the @c cuda exports.
 *
 *  @sa include/numkong/shapes.h
 *  @sa include/numkong/shapes/simt.cuh
 *
 *  Every block is one team solving one flip of one pair, with its transport state in dynamic shared
 *  memory sized for the largest cloud of the call: 32 threads up to 192 points and 128 above, where
 *  the longer scans outweigh the wider reductions. Each flip leaves its least cost and rotation in
 *  the workspace, and the last of a pair's 8 blocks to finish picks the best flip in the serial
 *  order and writes the distance and rotation. The workspace holds a pair slot per pair in flight,
 *  so a call walks the pairs in launches of as many pairs as it has slots. The pairs read their
 *  packs on the device, and one small read before the launches fetches the largest and smallest
 *  clouds, leaving out the simplex tree when every cloud has the same size.
 *
 *  Gromov-Wasserstein runs the flips first, each also leaving its best plan in the slot, then one
 *  block per start of each pair over that slot's dense matrices, and the last start to finish
 *  writes the pair.
 */
#ifndef NUMKONG_SHAPES_CUDA_CUH
#define NUMKONG_SHAPES_CUDA_CUH

#if NUMKONG_ARCH_CUDA_

#include "numkong/cuda.cuh"
#include "numkong/shapes/simt.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

#if NUMKONG_TARGET_CUDA
#pragma region Configuration

enum {

    /** Threads of a team on clouds up to @c nk_shapes_small_cloud_cuda_k points. */
    nk_shapes_small_team_cuda_k = 32,

    /** Threads of a team on larger clouds, where the longer scans outweigh the wider reductions. */
    nk_shapes_large_team_cuda_k = 128,

    /** The largest cloud the smaller teams take. */
    nk_shapes_small_cloud_cuda_k = 192,

    /** Pairs the Wasserstein-Procrustes workspace keeps in flight, enough for every multiprocessor
     *  many times over. */
    nk_shapes_pair_slots_cuda_k = 16384,

    /** Pairs times points the Gromov-Wasserstein workspace keeps in flight, between 16 and 4096
     *  pairs, so its bytes grow with the clouds it was sized for and give that bound back. */
    nk_gromov_pair_points_cuda_k = 131072,

    /** Doubles of a flip's or start's record: its least cost or loss, then its rotation's 9. */
    nk_shapes_record_cuda_k = 10,

    /** Offset of a slot's 8 flip records, after its counters of flips and of starts done. */
    nk_shapes_slot_records_cuda_k = 8,

    /** Offset of a slot's 3 start records. */
    nk_shapes_slot_starts_cuda_k = 8 + 8 * nk_shapes_record_cuda_k * 8,

    /** Offset of a Gromov-Wasserstein slot's flip plans. */
    nk_shapes_slot_plans_cuda_k = nk_shapes_slot_starts_cuda_k + 3 * nk_shapes_record_cuda_k * 8,

    /** Blocks of the larger teams each multiprocessor keeps, which caps the registers per thread so
     *  that the smaller teams of small clouds stay 16 to a multiprocessor. */
    nk_shapes_blocks_cuda_k = 4,

    /** The same for f16 inputs, whose f32 math leaves room for 21 smaller teams. */
    nk_shapes_blocks_f16_cuda_k = 5,

    /** Most classes of cloud sizes a Wasserstein-Procrustes chunk sorts its pairs into, up to
     *  98,304 points. */
    nk_shapes_classes_cuda_k = 24,

    /** Bytes the workspace spends on teams' states in global memory, for clouds past one block. */
    nk_shapes_team_budget_cuda_k = 256 << 20,

    /** Most teams' states the workspace keeps, enough for every resident block. */
    nk_shapes_teams_cuda_k = 1024,
};

/** One transport launch: the packs, the outputs, the pair slots of the workspace, a slot's layout,
 *  and the pairs it walks, from @c pair_begin on. Gromov-Wasserstein slots add each flip's best
 *  plan at @c plans_offset , then the dense state of each start, @c gromov_stride bytes apart from
 *  @c gromov_offset on. A Wasserstein-Procrustes launch solves the @c class_count pairs listed at
 *  @c class_slots , whose larger clouds hold at most @c class_ceiling points, which sizes its
 *  state, and the last one records those above @c class_limit as unsolvable. Without a list, a
 *  launch takes every pair of its chunk. With @c team_states , teams of blocks keep their state in
 *  global memory, @c team_stride bytes apart, at most @c teams of them; without, each block keeps
 *  its own in shared memory. */
typedef struct {
    void const *const *a_packed;
    void const *const *b_packed;
    void *results;
    nk_f32_t *rotations;
    nk_size_t rows, columns, results_stride;
    nk_size_t pair_begin, pairs;
    unsigned char *slots;
    nk_size_t slot_stride, plans_offset, gromov_offset, gromov_stride;
    nk_u32_t *class_counts;
    nk_u32_t const *class_count, *class_slots;
    nk_u32_t class_ceiling, class_limit;
    int gromov;
    unsigned char *team_states;
    nk_size_t team_stride, teams;
} nk_shapes_transport_schedule_cuda_t;

/** The blocks solving one item at a time, from @c first on, @c count teams apart: one block over
 *  shared memory, or @c blocks blocks over the state at @c region , this one of @c rank . */
typedef struct {
    nk_size_t first, count;
    unsigned blocks, rank;
    unsigned char *region;
} nk_shapes_team_cuda_t;

/** Bytes of a Wasserstein-Procrustes pair slot: two counters and 8 flip records. */
NUMKONG_INLINE nk_size_t nk_shapes_slot_bytes_cuda_(void) { return nk_shapes_slot_starts_cuda_k; }

/** Bytes of a team's meeting place and state in global memory for clouds of up to @p largest
 *  points of @p point_bytes per coordinate. */
NUMKONG_INLINE nk_size_t nk_shapes_team_stride_cuda_(nk_size_t largest, nk_size_t point_bytes) {
    return nk_size_round_up_to_multiple_(
        nk_shapes_meeting_bytes_simt_k + nk_shapes_state_bytes_simt_(largest, largest, point_bytes), 256);
}

/** Teams' states a workspace keeps for clouds of up to @p largest points. */
NUMKONG_INLINE nk_size_t nk_shapes_teams_cuda_(nk_size_t largest, nk_size_t point_bytes) {
    nk_size_t const teams = (nk_size_t)nk_shapes_team_budget_cuda_k / nk_shapes_team_stride_cuda_(largest, point_bytes);
    return teams < 1 ? 1 : teams > nk_shapes_teams_cuda_k ? nk_shapes_teams_cuda_k : teams;
}

/** Bytes of a Wasserstein-Procrustes workspace of @p slots pair slots for clouds of up to
 *  @p largest points: the teams, the class counts, a list of slots per class, and the slots. */
NUMKONG_INLINE nk_size_t nk_shapes_workspace_bytes_cuda_(nk_size_t slots, nk_size_t largest, nk_size_t point_bytes) {
    return nk_shapes_teams_cuda_(largest, point_bytes) * nk_shapes_team_stride_cuda_(largest, point_bytes) +
           nk_shapes_classes_cuda_k * sizeof(nk_u32_t) * (1 + slots) + slots * nk_shapes_slot_bytes_cuda_();
}

/** The largest cloud of class @p class_index : 32, 48, 64, 96, 128 and on, two per doubling. */
NUMKONG_CONSTEXPR nk_u32_t nk_shapes_class_ceiling_cuda_(nk_u32_t class_index) {
    return (32u << (class_index / 2)) + ((class_index & 1u) ? (16u << (class_index / 2)) : 0u);
}

/** Offset of a Gromov-Wasserstein slot's dense state, after the 8 flips' plans of up to
 *  2 × @p largest arcs each. */
NUMKONG_INLINE nk_size_t nk_gromov_dense_offset_cuda_(nk_size_t largest) {
    return nk_size_round_up_to_multiple_(nk_shapes_slot_plans_cuda_k + 8 * 3 * 2 * largest * sizeof(nk_u32_t), 128);
}

/** Bytes of a Gromov-Wasserstein pair slot for clouds of up to @p largest points: the records, the
 *  plans, and the 3 starts' dense state. */
NUMKONG_INLINE nk_size_t nk_gromov_slot_bytes_cuda_(nk_size_t largest) {
    return nk_gromov_dense_offset_cuda_(largest) + 3 * nk_shapes_gromov_bytes_simt_(largest);
}

/** Pair slots of a Gromov-Wasserstein workspace for clouds of up to @p largest points. */
NUMKONG_INLINE nk_size_t nk_gromov_slots_cuda_(nk_size_t largest) {
    nk_size_t const slots = nk_gromov_pair_points_cuda_k / (largest ? largest : 1);
    return slots < 16 ? 16 : slots > 4096 ? 4096 : slots;
}

/** Bytes of a Gromov-Wasserstein workspace for clouds of up to @p largest points of @p point_bytes
 *  per coordinate: the slots, then the teams. */
NUMKONG_INLINE nk_size_t nk_gromov_workspace_bytes_cuda_(nk_size_t largest, nk_size_t point_bytes) {
    return nk_size_round_up_to_multiple_(nk_gromov_slots_cuda_(largest) * nk_gromov_slot_bytes_cuda_(largest), 256) +
           nk_shapes_teams_cuda_(largest, point_bytes) * nk_shapes_team_stride_cuda_(largest, point_bytes);
}

/** The largest cloud a team of @p team threads holds in the device's shared memory, at
 *  @p point_bytes per coordinate. */
NUMKONG_INLINE nk_status_t nk_shapes_largest_supported_cuda_(unsigned team, nk_size_t point_bytes, nk_u32_t *largest,
                                                             nk_stream_t stream) {
    int shared_limit = 0;
    nk_status_t const status = nk_device_attribute_cuda_(cudaDevAttrMaxSharedMemoryPerBlockOptin, &shared_limit,
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
NUMKONG_INLINE nk_status_t nk_shapes_zero_counters_cuda_(unsigned char *slots, nk_size_t stride, nk_size_t used,
                                                         nk_stream_t stream) {
    return cudaMemset2DAsync(slots, stride, 0, 2 * sizeof(nk_u32_t), used, (cudaStream_t)stream) == cudaSuccess
               ? nk_success_k
               : nk_device_code_mismatch_k;
}

/** Launches as many blocks of @p threads running @p kernel as stay resident on the stream's device,
 *  all at once, as the blocks of a team wait on each other, passing the one argument struct at
 *  @p arguments by value. */
NUMKONG_INLINE nk_status_t nk_shapes_launch_teams_cuda_(void const *kernel, unsigned threads, nk_size_t shared_bytes,
                                                        void *arguments, nk_stream_t stream) {
    int caller = 0;
    nk_status_t const entered = nk_device_enter_cuda_(stream, &caller);
    if (entered != nk_success_k) return entered;
    int device = 0, multiprocessors = 0, per_multiprocessor = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared_bytes) != cudaSuccess ||
        cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device) != cudaSuccess ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_multiprocessor, kernel, (int)threads, shared_bytes) !=
            cudaSuccess) {
        nk_device_leave_cuda_(caller);
        return nk_device_code_mismatch_k;
    }
    void *launch_arguments[1];
    launch_arguments[0] = arguments;
    dim3 grid, block;
    grid.x = (unsigned)(multiprocessors * per_multiprocessor), grid.y = 1, grid.z = 1;
    block.x = threads, block.y = 1, block.z = 1;
    cudaError_t const status = grid.x ? cudaLaunchCooperativeKernel(kernel, grid, block, launch_arguments, shared_bytes,
                                                                    (cudaStream_t)stream)
                                      : cudaErrorInvalidConfiguration;
    nk_device_leave_cuda_(caller);
    return status == cudaSuccess ? nk_success_k : nk_device_code_mismatch_k;
}

#pragma endregion Configuration

#pragma region Packs

/** Whether the column pack of an f32 pair goes first in the canonical order of
 *  @c nk_shapes_column_first_f32_serial_ : by points, spreads, radius and then bytes. */
NUMKONG_DEVICE int nk_shapes_column_first_f32_cuda_(void const *row_packed, void const *column_packed) {
    nk_shapes_packed_header_f64_t const *row_header = (nk_shapes_packed_header_f64_t const *)row_packed;
    nk_shapes_packed_header_f64_t const *column_header = (nk_shapes_packed_header_f64_t const *)column_packed;
    if (row_header->points != column_header->points) return column_header->points < row_header->points;
    for (int axis = 0; axis != 3; ++axis)
        if (row_header->spreads[axis] != column_header->spreads[axis])
            return column_header->spreads[axis] < row_header->spreads[axis];
    if (row_header->max_radius != column_header->max_radius) return column_header->max_radius < row_header->max_radius;
    nk_size_t const header_bytes = nk_size_round_up_to_multiple_(sizeof(nk_shapes_packed_header_f64_t),
                                                                 nk_shapes_alignment_k_);
    nk_size_t const axis_bytes = nk_size_round_up_to_multiple_(row_header->points * sizeof(nk_f32_t),
                                                               nk_shapes_alignment_k_);
    nk_size_t const points_bytes = row_header->points * sizeof(nk_f32_t);
    nk_u8_t const *row_bytes = (nk_u8_t const *)row_packed + header_bytes;
    nk_u8_t const *column_bytes = (nk_u8_t const *)column_packed + header_bytes;
    for (nk_size_t axis_offset = 0; axis_offset != 3 * axis_bytes; axis_offset += axis_bytes)
        for (nk_size_t byte_index = axis_offset; byte_index != axis_offset + points_bytes; ++byte_index)
            if (row_bytes[byte_index] != column_bytes[byte_index])
                return column_bytes[byte_index] < row_bytes[byte_index];
    return 0;
}

NUMKONG_DEVICE int nk_shapes_column_first_f16_cuda_(void const *row_packed, void const *column_packed) {
    nk_shapes_packed_header_f32_t const *row_header = (nk_shapes_packed_header_f32_t const *)row_packed;
    nk_shapes_packed_header_f32_t const *column_header = (nk_shapes_packed_header_f32_t const *)column_packed;
    if (row_header->points != column_header->points) return column_header->points < row_header->points;
    for (int axis = 0; axis != 3; ++axis)
        if (row_header->spreads[axis] != column_header->spreads[axis])
            return column_header->spreads[axis] < row_header->spreads[axis];
    if (row_header->max_radius != column_header->max_radius) return column_header->max_radius < row_header->max_radius;
    nk_size_t const header_bytes = nk_size_round_up_to_multiple_(sizeof(nk_shapes_packed_header_f32_t),
                                                                 nk_shapes_alignment_k_);
    nk_size_t const axis_bytes = nk_size_round_up_to_multiple_(row_header->points * sizeof(nk_f16_t),
                                                               nk_shapes_alignment_k_);
    nk_size_t const points_bytes = row_header->points * sizeof(nk_f16_t);
    nk_u8_t const *row_bytes = (nk_u8_t const *)row_packed + header_bytes;
    nk_u8_t const *column_bytes = (nk_u8_t const *)column_packed + header_bytes;
    for (nk_size_t axis_offset = 0; axis_offset != 3 * axis_bytes; axis_offset += axis_bytes)
        for (nk_size_t byte_index = axis_offset; byte_index != axis_offset + points_bytes; ++byte_index)
            if (row_bytes[byte_index] != column_bytes[byte_index])
                return column_bytes[byte_index] < row_bytes[byte_index];
    return 0;
}

/** The rotation 𝑄ₐᵀ × inner × 𝑄ᵦ of @c nk_shapes_rotation_f64_serial_ , taking the column cloud
 *  onto the row cloud. */
NUMKONG_DEVICE void nk_shapes_rotation_f64_cuda_(nk_f64_t const *row_frame, nk_f64_t const *inner,
                                                 nk_f64_t const *column_frame, nk_f64_t *rotation) {
    nk_f64_t row_axes[9], column_axes[9], inner_column[9];
    nk_quaternion_to_mat3x3_f64_simt_(row_frame, row_axes);
    nk_quaternion_to_mat3x3_f64_simt_(column_frame, column_axes);
    for (int row = 0; row != 3; ++row)
        for (int column = 0; column != 3; ++column)
            inner_column[row * 3 + column] = inner[row * 3] * column_axes[column] +
                                             inner[row * 3 + 1] * column_axes[3 + column] +
                                             inner[row * 3 + 2] * column_axes[6 + column];
    for (int row = 0; row != 3; ++row)
        for (int column = 0; column != 3; ++column)
            rotation[row * 3 + column] = row_axes[row] * inner_column[column] +
                                         row_axes[3 + row] * inner_column[3 + column] +
                                         row_axes[6 + row] * inner_column[6 + column];
}

NUMKONG_DEVICE void nk_shapes_rotation_f32_cuda_(nk_f32_t const *row_frame, nk_f32_t const *inner,
                                                 nk_f32_t const *column_frame, nk_f32_t *rotation) {
    nk_f32_t row_axes[9], column_axes[9], inner_column[9];
    nk_quaternion_to_mat3x3_f32_simt_(row_frame, row_axes);
    nk_quaternion_to_mat3x3_f32_simt_(column_frame, column_axes);
    for (int row = 0; row != 3; ++row)
        for (int column = 0; column != 3; ++column)
            inner_column[row * 3 + column] = inner[row * 3] * column_axes[column] +
                                             inner[row * 3 + 1] * column_axes[3 + column] +
                                             inner[row * 3 + 2] * column_axes[6 + column];
    for (int row = 0; row != 3; ++row)
        for (int column = 0; column != 3; ++column)
            rotation[row * 3 + column] = row_axes[row] * inner_column[column] +
                                         row_axes[3 + row] * inner_column[3 + column] +
                                         row_axes[6 + row] * inner_column[6 + column];
}

/** Whether a pair with @p rows and @p columns points is above the launch's largest cloud. */
NUMKONG_DEVICE int nk_shapes_pair_oversize_cuda_(nk_shapes_transport_schedule_cuda_t const *schedule, nk_u32_t rows,
                                                 nk_u32_t columns) {
    return (rows > columns ? rows : columns) > schedule->class_ceiling;
}

/** This block's team for a launch of @p items items: itself alone without team states, else as
 *  many blocks per item as the grid spares, up to 16, which every block derives alike. Tells
 *  whether the block has a team, as the grid's last blocks may not fill one. */
NUMKONG_DEVICE int nk_shapes_find_team_cuda_(nk_shapes_transport_schedule_cuda_t const *schedule, nk_size_t items,
                                             nk_shapes_team_cuda_t *team) {
    team->first = blockIdx.x, team->count = gridDim.x, team->blocks = 1, team->rank = 0;
    team->region = NUMKONG_NULL;
    if (!schedule->team_states) return 1;
    nk_size_t const spare = gridDim.x / (items ? items : 1);
    team->blocks = spare >= nk_shapes_team_blocks_simt_k ? nk_shapes_team_blocks_simt_k : spare ? (unsigned)spare : 1;
    team->count = gridDim.x / team->blocks;
    team->count = team->count < schedule->teams ? team->count : schedule->teams;
    team->first = blockIdx.x / team->blocks, team->rank = blockIdx.x % team->blocks;
    team->region = schedule->team_states + team->first * schedule->team_stride;
    return team->first < team->count;
}

/** Carves a team's network over the largest clouds of the launch, the block's scratch at @p shared
 *  and the state at @p state , and joins the team's blocks. */
NUMKONG_DEVICE void nk_shapes_team_carve_cuda_(nk_shapes_team_cuda_t const *team, void *shared, void *state,
                                               nk_u32_t largest, nk_u32_t rows, nk_u32_t columns, nk_size_t point_bytes,
                                               nk_shapes_network_simt_t *network) {
    nk_shapes_network_carve_simt_(shared, state, largest, largest, rows, columns, point_bytes, network);
    if (team->blocks != 1) nk_shapes_network_join_simt_(network, team->region, team->blocks, team->rank);
}

/** Lists each pair of the chunk under the class of its larger cloud, the last class taking every
 *  cloud above @c class_limit . */
NUMKONG_DEVICE void nk_shapes_classify_cuda_(nk_shapes_transport_schedule_cuda_t const *schedule, nk_size_t slot_index,
                                             nk_u32_t largest) {
    nk_u32_t class_index = 0;
    while (nk_shapes_class_ceiling_cuda_(class_index) < largest &&
           nk_shapes_class_ceiling_cuda_(class_index) < schedule->class_limit &&
           class_index + 1 != nk_shapes_classes_cuda_k)
        ++class_index;
    nk_u32_t const position = atomicAdd(&schedule->class_counts[class_index], 1u);
    // The classifying launch fills the lists every solving launch then reads through `class_slots`.
    nk_u32_t *const lists = (nk_u32_t *)schedule->class_slots;
    lists[class_index * schedule->pairs + position] = (nk_u32_t)slot_index;
}

static __global__ void nk_shapes_classify_f32_cuda_kernel_(nk_shapes_transport_schedule_cuda_t const schedule) {
    for (nk_size_t slot_index = blockIdx.x * (nk_size_t)blockDim.x + threadIdx.x; slot_index < schedule.pairs;
         slot_index += (nk_size_t)gridDim.x * blockDim.x) {
        nk_size_t const pair = schedule.pair_begin + slot_index;
        nk_u32_t const rows =
            (nk_u32_t)((nk_shapes_packed_header_f64_t const *)schedule.a_packed[pair / schedule.columns])->points;
        nk_u32_t const columns =
            (nk_u32_t)((nk_shapes_packed_header_f64_t const *)schedule.b_packed[pair % schedule.columns])->points;
        nk_shapes_classify_cuda_(&schedule, slot_index, rows > columns ? rows : columns);
    }
}

static __global__ void nk_shapes_classify_f16_cuda_kernel_(nk_shapes_transport_schedule_cuda_t const schedule) {
    for (nk_size_t slot_index = blockIdx.x * (nk_size_t)blockDim.x + threadIdx.x; slot_index < schedule.pairs;
         slot_index += (nk_size_t)gridDim.x * blockDim.x) {
        nk_size_t const pair = schedule.pair_begin + slot_index;
        nk_u32_t const rows =
            ((nk_shapes_packed_header_f32_t const *)schedule.a_packed[pair / schedule.columns])->points;
        nk_u32_t const columns =
            ((nk_shapes_packed_header_f32_t const *)schedule.b_packed[pair % schedule.columns])->points;
        nk_shapes_classify_cuda_(&schedule, slot_index, rows > columns ? rows : columns);
    }
}

#pragma endregion Packs

#pragma region Wasserstein Procrustes

/** Picks a pair's best flip in the serial order from its 8 records, and writes its distance and
 *  rotation. Runs on one thread of the last block of the pair. */
NUMKONG_DEVICE void nk_wassersteinprocrustes_finish_f32_cuda_(nk_shapes_transport_schedule_cuda_t const *schedule,
                                                              nk_size_t pair, nk_f64_t const volatile *records) {
    nk_size_t const row = pair / schedule->columns, column = pair % schedule->columns;
    void const *a = schedule->a_packed[row], *b = schedule->b_packed[column];
    int const swapped = nk_shapes_column_first_f32_cuda_(a, b);
    nk_shapes_packed_header_f64_t const *row_header = (nk_shapes_packed_header_f64_t const *)(swapped ? b : a);
    nk_shapes_packed_header_f64_t const *column_header = (nk_shapes_packed_header_f64_t const *)(swapped ? a : b);
    int best_flip = 0;
    for (int flip = 1; flip != 8; ++flip)
        if (records[flip * nk_shapes_record_cuda_k] < records[best_flip * nk_shapes_record_cuda_k]) best_flip = flip;
    nk_f64_t const best = records[best_flip * nk_shapes_record_cuda_k];
    nk_f64_t *results_row = (nk_f64_t *)((char *)schedule->results + row * schedule->results_stride);
    int const empty = !row_header->points || !column_header->points;
    // A pair over the launch's largest cloud records NaN, which the root would turn into zero
    results_row[column] = empty || best != best ? best : nk_sqrt_f64_simt_(best);
    if (!schedule->rotations) return;
    nk_f64_t inner[9], pair_rotation[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    for (int cell = 0; cell != 9; ++cell) inner[cell] = records[best_flip * nk_shapes_record_cuda_k + 1 + cell];
    if (!empty) nk_shapes_rotation_f64_cuda_(row_header->frame, inner, column_header->frame, pair_rotation);
    nk_f32_t *output = schedule->rotations + pair * 9;
    for (int cell = 0; cell != 9; ++cell)
        output[cell] = (nk_f32_t)pair_rotation[swapped ? cell % 3 * 3 + cell / 3 : cell];
}

NUMKONG_DEVICE void nk_wassersteinprocrustes_finish_f16_cuda_(nk_shapes_transport_schedule_cuda_t const *schedule,
                                                              nk_size_t pair, nk_f64_t const volatile *records) {
    nk_size_t const row = pair / schedule->columns, column = pair % schedule->columns;
    void const *a = schedule->a_packed[row], *b = schedule->b_packed[column];
    int const swapped = nk_shapes_column_first_f16_cuda_(a, b);
    nk_shapes_packed_header_f32_t const *row_header = (nk_shapes_packed_header_f32_t const *)(swapped ? b : a);
    nk_shapes_packed_header_f32_t const *column_header = (nk_shapes_packed_header_f32_t const *)(swapped ? a : b);
    int best_flip = 0;
    for (int flip = 1; flip != 8; ++flip)
        if (records[flip * nk_shapes_record_cuda_k] < records[best_flip * nk_shapes_record_cuda_k]) best_flip = flip;
    nk_f32_t const best = (nk_f32_t)records[best_flip * nk_shapes_record_cuda_k];
    nk_f32_t *results_row = (nk_f32_t *)((char *)schedule->results + row * schedule->results_stride);
    int const empty = !row_header->points || !column_header->points;
    results_row[column] = empty || best != best ? best : nk_sqrt_f32_simt_(best);
    if (!schedule->rotations) return;
    nk_f32_t inner[9], pair_rotation[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    for (int cell = 0; cell != 9; ++cell)
        inner[cell] = (nk_f32_t)records[best_flip * nk_shapes_record_cuda_k + 1 + cell];
    if (!empty) nk_shapes_rotation_f32_cuda_(row_header->frame, inner, column_header->frame, pair_rotation);
    nk_f32_t *output = schedule->rotations + pair * 9;
    for (int cell = 0; cell != 9; ++cell) output[cell] = pair_rotation[swapped ? cell % 3 * 3 + cell / 3 : cell];
}

/** Stores the record of the @p index th of a slot's @p count flips or starts at @p offset , and
 *  tells whether it was the last of them to arrive, which then resets the slot's @p counter . */
NUMKONG_DEVICE void nk_shapes_record_cuda_(unsigned char *slot, nk_size_t offset, nk_u32_t *counter, int index,
                                           unsigned count, nk_f64_t cost, nk_f64_t const *rotation, int *last) {
    if (threadIdx.x == 0) {
        nk_f64_t *record = (nk_f64_t *)(slot + offset) + index * nk_shapes_record_cuda_k;
        record[0] = cost;
        for (int cell = 0; cell != 9; ++cell) record[1 + cell] = rotation[cell];
        __threadfence();
        *last = atomicAdd(counter, 1u) == count - 1;
        if (*last) __threadfence(), *counter = 0;
    }
    __syncthreads();
}

/** Solves the team's flips of a launch's @p items items, the block's scratch at @p shared and the
 *  team's state at @p state . Wasserstein-Procrustes finishes each pair in its last flip; for
 *  Gromov-Wasserstein, each flip only leaves its record and best plan. */
NUMKONG_DEVICE void nk_wassersteinprocrustes_walk_f32_cuda_(nk_shapes_transport_schedule_cuda_t const *schedule,
                                                            nk_shapes_team_cuda_t const *team, nk_size_t items,
                                                            void *shared, void *state, nk_f64_t *shared_rotation,
                                                            int *last) {
    unsigned parity = 0;
    nk_u32_t const plan_arcs = 2 * schedule->class_ceiling;
    for (nk_size_t item = team->first; item < items; item += team->count) {
        nk_size_t const slot_index = schedule->class_slots ? schedule->class_slots[item >> 3] : item >> 3;
        nk_size_t const pair = schedule->pair_begin + slot_index;
        unsigned char *slot = schedule->slots + slot_index * schedule->slot_stride;
        int const flip = (int)(item & 7);
        void const *a = schedule->a_packed[pair / schedule->columns];
        void const *b = schedule->b_packed[pair % schedule->columns];
        int const swapped = nk_shapes_column_first_f32_cuda_(a, b);
        void const *row_pack = swapped ? b : a, *column_pack = swapped ? a : b;
        nk_u32_t const rows = (nk_u32_t)((nk_shapes_packed_header_f64_t const *)row_pack)->points;
        nk_u32_t const columns = (nk_u32_t)((nk_shapes_packed_header_f64_t const *)column_pack)->points;
        nk_f64_t cost, rotation[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        if (nk_shapes_pair_oversize_cuda_(schedule, rows, columns)) cost = NUMKONG_F64_INF - NUMKONG_F64_INF;
        else if (!rows || !columns) cost = rows == columns ? 0 : NUMKONG_F64_INF;
        else {
            nk_shapes_points_f32_simt_t points;
            nk_shapes_network_simt_t network;
            nk_shapes_team_carve_cuda_(team, shared, state, schedule->class_ceiling, rows, columns, sizeof(nk_f64_t),
                                       &network);
            nk_shapes_points_carve_f32_simt_(state, schedule->class_ceiling, schedule->class_ceiling, &points);
            nk_f64_t const radii = ((nk_shapes_packed_header_f64_t const *)row_pack)->max_radius +
                                   ((nk_shapes_packed_header_f64_t const *)column_pack)->max_radius;
            network.tolerance = radii * radii * 1e-12;
            nk_size_t row_stride, column_stride;
            nk_f32_t const *row_axes = nk_shapes_packed_axes_f32_simt_(row_pack, &row_stride);
            nk_f32_t const *column_axes = nk_shapes_packed_axes_f32_simt_(column_pack, &column_stride);
            nk_u32_t *saved_plan = schedule->gromov ? (nk_u32_t *)(slot + schedule->plans_offset) + flip * 3 * plan_arcs
                                                    : NUMKONG_NULL;
            nk_shapes_load_rows_f32_simt_(&network, &points, row_axes, row_stride, rows);
            nk_wassersteinprocrustes_flip_f32_simt_(&network, &points, column_axes, column_stride, flip,
                                                    shared_rotation, &cost, rotation, saved_plan, &parity);
        }
        if (team->rank != 0) continue;
        nk_shapes_record_cuda_(slot, nk_shapes_slot_records_cuda_k, (nk_u32_t *)slot, flip, 8, cost, rotation, last);
        if (*last && !schedule->gromov && threadIdx.x == 0)
            nk_wassersteinprocrustes_finish_f32_cuda_(schedule, pair,
                                                      (nk_f64_t *)(slot + nk_shapes_slot_records_cuda_k));
        __syncthreads();
    }
}

/** One team per flip of each pair, the walk inlined once over shared memory and once over global
 *  memory, so that the block tier addresses its state as shared. */
static __global__ void __launch_bounds__(nk_shapes_large_team_cuda_k, nk_shapes_blocks_cuda_k)
    nk_wassersteinprocrustes_packed_f32_cuda_kernel_(nk_shapes_transport_schedule_cuda_t const schedule) {
    extern __shared__ nk_f64_t nk_shapes_shared_cuda_[];
    __shared__ nk_f64_t shared_rotation[9];
    __shared__ int last;
    nk_size_t const items = (schedule.class_slots ? (nk_size_t)*schedule.class_count : schedule.pairs) * 8;
    nk_shapes_team_cuda_t team;
    if (!nk_shapes_find_team_cuda_(&schedule, items, &team)) return;
    if (team.region)
        nk_wassersteinprocrustes_walk_f32_cuda_(&schedule, &team, items, nk_shapes_shared_cuda_,
                                                team.region + nk_shapes_meeting_bytes_simt_k, shared_rotation, &last);
    else
        nk_wassersteinprocrustes_walk_f32_cuda_(
            &schedule, &team, items, nk_shapes_shared_cuda_,
            (char *)nk_shapes_shared_cuda_ + nk_shapes_scratch_bytes_simt_(blockDim.x), shared_rotation, &last);
}

NUMKONG_DEVICE void nk_wassersteinprocrustes_walk_f16_cuda_(nk_shapes_transport_schedule_cuda_t const *schedule,
                                                            nk_shapes_team_cuda_t const *team, nk_size_t items,
                                                            void *shared, void *state, nk_f32_t *shared_rotation,
                                                            int *last) {
    unsigned parity = 0;
    nk_u32_t const plan_arcs = 2 * schedule->class_ceiling;
    for (nk_size_t item = team->first; item < items; item += team->count) {
        nk_size_t const slot_index = schedule->class_slots ? schedule->class_slots[item >> 3] : item >> 3;
        nk_size_t const pair = schedule->pair_begin + slot_index;
        unsigned char *slot = schedule->slots + slot_index * schedule->slot_stride;
        int const flip = (int)(item & 7);
        void const *a = schedule->a_packed[pair / schedule->columns];
        void const *b = schedule->b_packed[pair % schedule->columns];
        int const swapped = nk_shapes_column_first_f16_cuda_(a, b);
        void const *row_pack = swapped ? b : a, *column_pack = swapped ? a : b;
        nk_u32_t const rows = ((nk_shapes_packed_header_f32_t const *)row_pack)->points;
        nk_u32_t const columns = ((nk_shapes_packed_header_f32_t const *)column_pack)->points;
        nk_f32_t cost, rotation[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        if (nk_shapes_pair_oversize_cuda_(schedule, rows, columns)) cost = NUMKONG_F32_INF - NUMKONG_F32_INF;
        else if (!rows || !columns) cost = rows == columns ? 0 : NUMKONG_F32_INF;
        else {
            nk_shapes_points_f16_simt_t points;
            nk_shapes_network_simt_t network;
            nk_shapes_team_carve_cuda_(team, shared, state, schedule->class_ceiling, rows, columns, sizeof(nk_f32_t),
                                       &network);
            nk_shapes_points_carve_f16_simt_(state, schedule->class_ceiling, schedule->class_ceiling, &points);
            nk_f32_t const radii = ((nk_shapes_packed_header_f32_t const *)row_pack)->max_radius +
                                   ((nk_shapes_packed_header_f32_t const *)column_pack)->max_radius;
            network.tolerance = (nk_f64_t)(radii * radii) * 1e-12;
            nk_size_t row_stride, column_stride;
            nk_f16_t const *row_axes = nk_shapes_packed_axes_f16_simt_(row_pack, &row_stride);
            nk_f16_t const *column_axes = nk_shapes_packed_axes_f16_simt_(column_pack, &column_stride);
            nk_u32_t *saved_plan = schedule->gromov ? (nk_u32_t *)(slot + schedule->plans_offset) + flip * 3 * plan_arcs
                                                    : NUMKONG_NULL;
            nk_shapes_load_rows_f16_simt_(&network, &points, row_axes, row_stride, rows);
            nk_wassersteinprocrustes_flip_f16_simt_(&network, &points, column_axes, column_stride, flip,
                                                    shared_rotation, &cost, rotation, saved_plan, &parity);
        }
        if (team->rank != 0) continue;
        nk_f64_t wide_rotation[9];
        for (int cell = 0; cell != 9; ++cell) wide_rotation[cell] = rotation[cell];
        nk_shapes_record_cuda_(slot, nk_shapes_slot_records_cuda_k, (nk_u32_t *)slot, flip, 8, cost, wide_rotation,
                               last);
        if (*last && !schedule->gromov && threadIdx.x == 0)
            nk_wassersteinprocrustes_finish_f16_cuda_(schedule, pair,
                                                      (nk_f64_t *)(slot + nk_shapes_slot_records_cuda_k));
        __syncthreads();
    }
}

static __global__ void __launch_bounds__(nk_shapes_large_team_cuda_k, nk_shapes_blocks_f16_cuda_k)
    nk_wassersteinprocrustes_packed_f16_cuda_kernel_(nk_shapes_transport_schedule_cuda_t const schedule) {
    extern __shared__ nk_f64_t nk_shapes_shared_cuda_[];
    __shared__ nk_f32_t shared_rotation[9];
    __shared__ int last;
    nk_size_t const items = (schedule.class_slots ? (nk_size_t)*schedule.class_count : schedule.pairs) * 8;
    nk_shapes_team_cuda_t team;
    if (!nk_shapes_find_team_cuda_(&schedule, items, &team)) return;
    if (team.region)
        nk_wassersteinprocrustes_walk_f16_cuda_(&schedule, &team, items, nk_shapes_shared_cuda_,
                                                team.region + nk_shapes_meeting_bytes_simt_k, shared_rotation, &last);
    else
        nk_wassersteinprocrustes_walk_f16_cuda_(
            &schedule, &team, items, nk_shapes_shared_cuda_,
            (char *)nk_shapes_shared_cuda_ + nk_shapes_scratch_bytes_simt_(blockDim.x), shared_rotation, &last);
}

/** Walks every pair in chunks of as many pairs as the workspace has slots. A light launch lists the
 *  chunk's pairs by the class of their larger cloud, then one launch per class, sized for its
 *  largest cloud, solves the pairs it reads from its list, so no call waits on the device. Classes
 *  that fit a block's shared memory take a block per flip; larger ones take teams of blocks over
 *  the workspace, up to clouds of @p largest points, and record any pair above as unsolvable. */
NUMKONG_INLINE nk_status_t nk_shapes_transport_launch_cuda_(void const *kernel, void const *classify_kernel,
                                                            nk_shapes_transport_schedule_cuda_t schedule,
                                                            nk_size_t point_bytes, nk_size_t largest, void *workspace,
                                                            nk_size_t workspace_bytes, nk_stream_t stream) {
    if ((nk_size_t)workspace & (nk_default_alignment_k - 1)) return nk_misaligned_k;
    if (!largest) return nk_unexpected_dimensions_k;
    nk_size_t const team_stride = nk_shapes_team_stride_cuda_(largest, point_bytes);
    nk_size_t const teams = nk_shapes_teams_cuda_(largest, point_bytes);
    nk_size_t const head_bytes = nk_shapes_workspace_bytes_cuda_(0, largest, point_bytes);
    nk_size_t const per_slot = nk_shapes_workspace_bytes_cuda_(1, largest, point_bytes) - head_bytes;
    nk_size_t slots = workspace_bytes > head_bytes ? (workspace_bytes - head_bytes) / per_slot : 0;
    slots = slots < nk_shapes_pair_slots_cuda_k ? slots : nk_shapes_pair_slots_cuda_k;
    if (!slots) return nk_unexpected_dimensions_k;
    nk_u32_t supported = 0;
    nk_status_t status = nk_shapes_largest_supported_cuda_(nk_shapes_large_team_cuda_k, point_bytes, &supported,
                                                           stream);
    if (status != nk_success_k) return status;
    nk_size_t const pairs = schedule.rows * schedule.columns;
    unsigned char *team_states = (unsigned char *)workspace;
    nk_u32_t *counts = (nk_u32_t *)(team_states + teams * team_stride), *lists = counts + nk_shapes_classes_cuda_k;
    schedule.slots = (unsigned char *)(lists + nk_shapes_classes_cuda_k * slots);
    schedule.slot_stride = nk_shapes_slot_bytes_cuda_(), schedule.gromov = 0;
    schedule.class_counts = counts, schedule.class_limit = (nk_u32_t)largest;
    schedule.team_stride = team_stride, schedule.teams = teams;
    status = nk_shapes_zero_counters_cuda_(schedule.slots, schedule.slot_stride, pairs < slots ? pairs : slots, stream);
    if (status == nk_success_k && largest > supported)
        status = nk_shapes_zero_counters_cuda_(team_states, team_stride, teams, stream);
    for (nk_size_t begin = 0; begin < pairs && status == nk_success_k; begin += slots) {
        schedule.pair_begin = begin, schedule.pairs = pairs - begin < slots ? pairs - begin : slots;
        schedule.class_count = counts, schedule.class_slots = lists, schedule.class_ceiling = (nk_u32_t)largest;
        schedule.team_states = NUMKONG_NULL;
        if (cudaMemsetAsync(counts, 0, nk_shapes_classes_cuda_k * sizeof(nk_u32_t), (cudaStream_t)stream) !=
            cudaSuccess)
            return nk_device_code_mismatch_k;
        void *classify_arguments[1] = {(void *)&schedule};
        nk_size_t const classify_blocks = nk_size_divide_round_up_(schedule.pairs, 256);
        status = nk_launch_cuda_(classify_kernel, classify_blocks < 1024 ? classify_blocks : 1024, 256,
                                 classify_arguments, 0, stream);
        int last = 0;
        for (nk_u32_t class_index = 0; !last && status == nk_success_k; ++class_index) {
            nk_u32_t const ceiling = nk_shapes_class_ceiling_cuda_(class_index);
            last = ceiling >= largest || class_index + 1 == nk_shapes_classes_cuda_k;
            schedule.class_ceiling = last ? (nk_u32_t)largest : ceiling;
            schedule.class_count = counts + class_index, schedule.class_slots = lists + class_index * schedule.pairs;
            if (schedule.class_ceiling > supported) {
                // Past one block's shared memory, teams of blocks keep their state in the workspace
                schedule.team_states = team_states;
                status = nk_shapes_launch_teams_cuda_(kernel, nk_shapes_large_team_cuda_k,
                                                      nk_shapes_scratch_bytes_simt_(nk_shapes_large_team_cuda_k),
                                                      &schedule, stream);
                continue;
            }
            unsigned const team = schedule.class_ceiling > nk_shapes_small_cloud_cuda_k ? nk_shapes_large_team_cuda_k
                                                                                        : nk_shapes_small_team_cuda_k;
            nk_size_t const shared_bytes = nk_shapes_team_bytes_simt_(schedule.class_ceiling, schedule.class_ceiling,
                                                                      team, point_bytes);
            status = nk_launch_resident_cuda_(kernel, team, shared_bytes, shared_bytes, schedule.pairs * 8, &schedule,
                                              stream);
        }
    }
    return status;
}

/** The largest cloud a Wasserstein-Procrustes workspace of @p workspace_bytes holds with every pair
 *  slot, as the size queries lay it out. */
NUMKONG_INLINE nk_size_t nk_shapes_workspace_bound_cuda_(nk_size_t workspace_bytes, nk_size_t point_bytes) {
    nk_size_t largest = nk_shapes_class_ceiling_cuda_(nk_shapes_classes_cuda_k - 1);
    while (largest &&
           nk_shapes_workspace_bytes_cuda_(nk_shapes_pair_slots_cuda_k, largest, point_bytes) > workspace_bytes)
        --largest;
    return largest;
}

NUMKONG_API nk_status_t nk_wassersteinprocrustes_workspace_size_f32_cuda(nk_size_t row_points, nk_size_t column_points,
                                                                         nk_size_t *bytes) {
    nk_size_t const largest = row_points > column_points ? row_points : column_points;
    *bytes = nk_shapes_workspace_bytes_cuda_(nk_shapes_pair_slots_cuda_k, largest ? largest : 1, sizeof(nk_f64_t));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_wassersteinprocrustes_workspace_size_f16_cuda(nk_size_t row_points, nk_size_t column_points,
                                                                         nk_size_t *bytes) {
    nk_size_t const largest = row_points > column_points ? row_points : column_points;
    *bytes = nk_shapes_workspace_bytes_cuda_(nk_shapes_pair_slots_cuda_k, largest ? largest : 1, sizeof(nk_f32_t));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f32_cuda(void const *const *a_packed, nk_size_t rows,
                                                                 void const *const *b_packed, nk_size_t columns,
                                                                 nk_f64_t *results, nk_size_t results_stride,
                                                                 nk_f32_t *rotations, void *workspace,
                                                                 nk_size_t workspace_bytes, nk_stream_t stream) {
    if (!rows || !columns) return nk_success_k;
    if (((nk_size_t)results | results_stride) & (sizeof(nk_f64_t) - 1) || (nk_size_t)rotations & 3)
        return nk_misaligned_k;
    nk_shapes_transport_schedule_cuda_t schedule;
    schedule.a_packed = a_packed, schedule.b_packed = b_packed, schedule.results = results;
    schedule.rotations = rotations, schedule.rows = rows, schedule.columns = columns;
    schedule.results_stride = results_stride;
    return nk_shapes_transport_launch_cuda_(
        (void const *)&nk_wassersteinprocrustes_packed_f32_cuda_kernel_,
        (void const *)&nk_shapes_classify_f32_cuda_kernel_, schedule, sizeof(nk_f64_t),
        nk_shapes_workspace_bound_cuda_(workspace_bytes, sizeof(nk_f64_t)), workspace, workspace_bytes, stream);
}

NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f16_cuda(void const *const *a_packed, nk_size_t rows,
                                                                 void const *const *b_packed, nk_size_t columns,
                                                                 nk_f32_t *results, nk_size_t results_stride,
                                                                 nk_f32_t *rotations, void *workspace,
                                                                 nk_size_t workspace_bytes, nk_stream_t stream) {
    if (!rows || !columns) return nk_success_k;
    if (((nk_size_t)results | results_stride) & (sizeof(nk_f32_t) - 1) || (nk_size_t)rotations & 3)
        return nk_misaligned_k;
    nk_shapes_transport_schedule_cuda_t schedule;
    schedule.a_packed = a_packed, schedule.b_packed = b_packed, schedule.results = results;
    schedule.rotations = rotations, schedule.rows = rows, schedule.columns = columns;
    schedule.results_stride = results_stride;
    return nk_shapes_transport_launch_cuda_(
        (void const *)&nk_wassersteinprocrustes_packed_f16_cuda_kernel_,
        (void const *)&nk_shapes_classify_f16_cuda_kernel_, schedule, sizeof(nk_f32_t),
        nk_shapes_workspace_bound_cuda_(workspace_bytes, sizeof(nk_f32_t)), workspace, workspace_bytes, stream);
}

#pragma endregion Wasserstein Procrustes

#pragma region Gromov Wasserstein

/** The flip whose plan starts @p start , the best proper flip for 0 and mirrored for 1, the first
 *  of equal ones as the serial solver keeps it. */
NUMKONG_DEVICE int nk_gromov_start_flip_cuda_(nk_f64_t const *records, int start) {
    int best = -1;
    for (int flip = 0; flip != 8; ++flip) {
        int const mirrored = ((flip & 1) + ((flip >> 1) & 1) + ((flip >> 2) & 1)) & 1;
        if (mirrored != start) continue;
        if (best < 0 || records[flip * nk_shapes_record_cuda_k] < records[best * nk_shapes_record_cuda_k]) best = flip;
    }
    return best;
}

/** Picks a pair's best start in the serial order from its 3 records, and writes its distance and
 *  rotation. Runs on one thread of the last block of the pair. */
NUMKONG_DEVICE void nk_gromovwassersteins_finish_f32_cuda_(nk_shapes_transport_schedule_cuda_t const *schedule,
                                                           nk_size_t pair, nk_f64_t const volatile *records) {
    nk_size_t const row = pair / schedule->columns, column = pair % schedule->columns;
    void const *a = schedule->a_packed[row], *b = schedule->b_packed[column];
    int const swapped = nk_shapes_column_first_f32_cuda_(a, b);
    nk_shapes_packed_header_f64_t const *row_header = (nk_shapes_packed_header_f64_t const *)(swapped ? b : a);
    nk_shapes_packed_header_f64_t const *column_header = (nk_shapes_packed_header_f64_t const *)(swapped ? a : b);
    nk_f64_t *results_row = (nk_f64_t *)((char *)schedule->results + row * schedule->results_stride);
    nk_f64_t inner[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}, pair_rotation[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    // A pair over the launch's largest cloud records NaN, which no start's loss would replace
    if (!row_header->points || !column_header->points || records[0] != records[0]) results_row[column] = records[0];
    else {
        nk_f64_t best = NUMKONG_F64_INF;
        for (int start = 0; start != 3; ++start) {
            if (!(records[start * nk_shapes_record_cuda_k] < best)) continue;
            best = records[start * nk_shapes_record_cuda_k];
            for (int cell = 0; cell != 9; ++cell) inner[cell] = records[start * nk_shapes_record_cuda_k + 1 + cell];
        }
        results_row[column] = nk_sqrt_f64_simt_(best) / 2;
        nk_shapes_rotation_f64_cuda_(row_header->frame, inner, column_header->frame, pair_rotation);
    }
    if (!schedule->rotations) return;
    nk_f32_t *output = schedule->rotations + pair * 9;
    for (int cell = 0; cell != 9; ++cell)
        output[cell] = (nk_f32_t)pair_rotation[swapped ? cell % 3 * 3 + cell / 3 : cell];
}

NUMKONG_DEVICE void nk_gromovwassersteins_finish_f16_cuda_(nk_shapes_transport_schedule_cuda_t const *schedule,
                                                           nk_size_t pair, nk_f64_t const volatile *records) {
    nk_size_t const row = pair / schedule->columns, column = pair % schedule->columns;
    void const *a = schedule->a_packed[row], *b = schedule->b_packed[column];
    int const swapped = nk_shapes_column_first_f16_cuda_(a, b);
    nk_shapes_packed_header_f32_t const *row_header = (nk_shapes_packed_header_f32_t const *)(swapped ? b : a);
    nk_shapes_packed_header_f32_t const *column_header = (nk_shapes_packed_header_f32_t const *)(swapped ? a : b);
    nk_f32_t *results_row = (nk_f32_t *)((char *)schedule->results + row * schedule->results_stride);
    nk_f32_t inner[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}, pair_rotation[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    if (!row_header->points || !column_header->points || records[0] != records[0])
        results_row[column] = (nk_f32_t)records[0];
    else {
        nk_f64_t best = NUMKONG_F64_INF;
        for (int start = 0; start != 3; ++start) {
            if (!(records[start * nk_shapes_record_cuda_k] < best)) continue;
            best = records[start * nk_shapes_record_cuda_k];
            for (int cell = 0; cell != 9; ++cell)
                inner[cell] = (nk_f32_t)records[start * nk_shapes_record_cuda_k + 1 + cell];
        }
        results_row[column] = (nk_f32_t)(nk_sqrt_f64_simt_(best) / 2);
        nk_shapes_rotation_f32_cuda_(row_header->frame, inner, column_header->frame, pair_rotation);
    }
    if (!schedule->rotations) return;
    nk_f32_t *output = schedule->rotations + pair * 9;
    for (int cell = 0; cell != 9; ++cell) output[cell] = pair_rotation[swapped ? cell % 3 * 3 + cell / 3 : cell];
}

/** Solves the team's starts of a launch's pairs, the block's scratch at @p shared and the team's
 *  network at @p state , after the Wasserstein-Procrustes launch left every flip's record and best
 *  plan in the pair's slot. */
NUMKONG_DEVICE void nk_gromovwassersteins_walk_f32_cuda_(nk_shapes_transport_schedule_cuda_t const *schedule,
                                                         nk_shapes_team_cuda_t const *team, void *shared, void *state,
                                                         nk_f64_t *shared_fit, int *last) {
    unsigned parity = 0;
    nk_u32_t const plan_arcs = 2 * schedule->class_ceiling;
    for (nk_size_t item = team->first; item < schedule->pairs * 3; item += team->count) {
        nk_size_t const slot_index = item / 3, pair = schedule->pair_begin + slot_index;
        unsigned char *slot = schedule->slots + slot_index * schedule->slot_stride;
        int const start = (int)(item % 3);
        void const *a = schedule->a_packed[pair / schedule->columns];
        void const *b = schedule->b_packed[pair % schedule->columns];
        int const swapped = nk_shapes_column_first_f32_cuda_(a, b);
        void const *row_pack = swapped ? b : a, *column_pack = swapped ? a : b;
        nk_u32_t const rows = (nk_u32_t)((nk_shapes_packed_header_f64_t const *)row_pack)->points;
        nk_u32_t const columns = (nk_u32_t)((nk_shapes_packed_header_f64_t const *)column_pack)->points;
        nk_f64_t loss, fit[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        if (nk_shapes_pair_oversize_cuda_(schedule, rows, columns)) loss = NUMKONG_F64_INF - NUMKONG_F64_INF;
        else if (!rows || !columns) loss = rows == columns ? 0 : NUMKONG_F64_INF;
        else {
            nk_shapes_network_simt_t network;
            nk_shapes_gromov_simt_t gromov;
            nk_shapes_team_carve_cuda_(team, shared, state, schedule->class_ceiling, rows, columns, 0, &network);
            nk_shapes_gromov_carve_simt_(slot + schedule->gromov_offset + start * schedule->gromov_stride,
                                         schedule->class_ceiling, &gromov);
            nk_size_t row_stride, column_stride;
            nk_f32_t const *row_axes = nk_shapes_packed_axes_f32_simt_(row_pack, &row_stride);
            nk_f32_t const *column_axes = nk_shapes_packed_axes_f32_simt_(column_pack, &column_stride);
            nk_shapes_distances_f32_simt_(&network, row_axes, row_stride, rows, gromov.row_distances);
            nk_shapes_distances_f32_simt_(&network, column_axes, column_stride, columns, gromov.column_distances);
            nk_f64_t const row_spread = nk_shapes_distance_spread_simt_(&network, gromov.row_distances, rows);
            nk_f64_t const column_spread = nk_shapes_distance_spread_simt_(&network, gromov.column_distances, columns);
            if (start == 2) nk_shapes_product_start_simt_(&network, &gromov);
            else {
                int const flip = nk_gromov_start_flip_cuda_((nk_f64_t const *)(slot + nk_shapes_slot_records_cuda_k),
                                                            start);
                nk_shapes_network_load_simt_(
                    &network, (nk_u32_t const *)(slot + schedule->plans_offset) + flip * 3 * plan_arcs, plan_arcs);
                nk_shapes_plan_start_simt_(&network, &gromov);
            }
            nk_f64_t const plan_score = nk_shapes_frank_wolfe_simt_(&network, &gromov, &parity);
            loss = row_spread + column_spread - 2 * plan_score;
            nk_shapes_plan_fit_f32_simt_(&network, &gromov, row_axes, row_stride, column_axes, column_stride,
                                         shared_fit, fit);
        }
        if (team->rank != 0) continue;
        nk_shapes_record_cuda_(slot, nk_shapes_slot_starts_cuda_k, (nk_u32_t *)slot + 1, start, 3, loss, fit, last);
        if (*last && threadIdx.x == 0)
            nk_gromovwassersteins_finish_f32_cuda_(schedule, pair, (nk_f64_t *)(slot + nk_shapes_slot_starts_cuda_k));
        __syncthreads();
    }
}

/** One team per start of each pair, the walk inlined once over shared memory and once over global
 *  memory, so that the block tier addresses its network as shared. */
static __global__ void __launch_bounds__(nk_shapes_large_team_cuda_k, nk_shapes_blocks_cuda_k)
    nk_gromovwassersteins_packed_f32_cuda_kernel_(nk_shapes_transport_schedule_cuda_t const schedule) {
    extern __shared__ nk_f64_t nk_shapes_shared_cuda_[];
    __shared__ nk_f64_t shared_fit[9];
    __shared__ int last;
    nk_shapes_team_cuda_t team;
    if (!nk_shapes_find_team_cuda_(&schedule, schedule.pairs * 3, &team)) return;
    if (team.region)
        nk_gromovwassersteins_walk_f32_cuda_(&schedule, &team, nk_shapes_shared_cuda_,
                                             team.region + nk_shapes_meeting_bytes_simt_k, shared_fit, &last);
    else
        nk_gromovwassersteins_walk_f32_cuda_(&schedule, &team, nk_shapes_shared_cuda_,
                                             (char *)nk_shapes_shared_cuda_ + nk_shapes_scratch_bytes_simt_(blockDim.x),
                                             shared_fit, &last);
}

NUMKONG_DEVICE void nk_gromovwassersteins_walk_f16_cuda_(nk_shapes_transport_schedule_cuda_t const *schedule,
                                                         nk_shapes_team_cuda_t const *team, void *shared, void *state,
                                                         nk_f64_t *shared_fit, int *last) {
    unsigned parity = 0;
    nk_u32_t const plan_arcs = 2 * schedule->class_ceiling;
    for (nk_size_t item = team->first; item < schedule->pairs * 3; item += team->count) {
        nk_size_t const slot_index = item / 3, pair = schedule->pair_begin + slot_index;
        unsigned char *slot = schedule->slots + slot_index * schedule->slot_stride;
        int const start = (int)(item % 3);
        void const *a = schedule->a_packed[pair / schedule->columns];
        void const *b = schedule->b_packed[pair % schedule->columns];
        int const swapped = nk_shapes_column_first_f16_cuda_(a, b);
        void const *row_pack = swapped ? b : a, *column_pack = swapped ? a : b;
        nk_u32_t const rows = ((nk_shapes_packed_header_f32_t const *)row_pack)->points;
        nk_u32_t const columns = ((nk_shapes_packed_header_f32_t const *)column_pack)->points;
        nk_f64_t loss, fit[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        if (nk_shapes_pair_oversize_cuda_(schedule, rows, columns)) loss = NUMKONG_F64_INF - NUMKONG_F64_INF;
        else if (!rows || !columns) loss = rows == columns ? 0 : NUMKONG_F64_INF;
        else {
            nk_shapes_network_simt_t network;
            nk_shapes_gromov_simt_t gromov;
            nk_shapes_team_carve_cuda_(team, shared, state, schedule->class_ceiling, rows, columns, 0, &network);
            nk_shapes_gromov_carve_simt_(slot + schedule->gromov_offset + start * schedule->gromov_stride,
                                         schedule->class_ceiling, &gromov);
            nk_size_t row_stride, column_stride;
            nk_f16_t const *row_axes = nk_shapes_packed_axes_f16_simt_(row_pack, &row_stride);
            nk_f16_t const *column_axes = nk_shapes_packed_axes_f16_simt_(column_pack, &column_stride);
            nk_shapes_distances_f16_simt_(&network, row_axes, row_stride, rows, gromov.row_distances);
            nk_shapes_distances_f16_simt_(&network, column_axes, column_stride, columns, gromov.column_distances);
            nk_f64_t const row_spread = nk_shapes_distance_spread_simt_(&network, gromov.row_distances, rows);
            nk_f64_t const column_spread = nk_shapes_distance_spread_simt_(&network, gromov.column_distances, columns);
            if (start == 2) nk_shapes_product_start_simt_(&network, &gromov);
            else {
                int const flip = nk_gromov_start_flip_cuda_((nk_f64_t const *)(slot + nk_shapes_slot_records_cuda_k),
                                                            start);
                nk_shapes_network_load_simt_(
                    &network, (nk_u32_t const *)(slot + schedule->plans_offset) + flip * 3 * plan_arcs, plan_arcs);
                nk_shapes_plan_start_simt_(&network, &gromov);
            }
            nk_f64_t const plan_score = nk_shapes_frank_wolfe_simt_(&network, &gromov, &parity);
            loss = row_spread + column_spread - 2 * plan_score;
            nk_shapes_plan_fit_f16_simt_(&network, &gromov, row_axes, row_stride, column_axes, column_stride,
                                         shared_fit, fit);
        }
        if (team->rank != 0) continue;
        nk_shapes_record_cuda_(slot, nk_shapes_slot_starts_cuda_k, (nk_u32_t *)slot + 1, start, 3, loss, fit, last);
        if (*last && threadIdx.x == 0)
            nk_gromovwassersteins_finish_f16_cuda_(schedule, pair, (nk_f64_t *)(slot + nk_shapes_slot_starts_cuda_k));
        __syncthreads();
    }
}

static __global__ void __launch_bounds__(nk_shapes_large_team_cuda_k, nk_shapes_blocks_cuda_k)
    nk_gromovwassersteins_packed_f16_cuda_kernel_(nk_shapes_transport_schedule_cuda_t const schedule) {
    extern __shared__ nk_f64_t nk_shapes_shared_cuda_[];
    __shared__ nk_f64_t shared_fit[9];
    __shared__ int last;
    nk_shapes_team_cuda_t team;
    if (!nk_shapes_find_team_cuda_(&schedule, schedule.pairs * 3, &team)) return;
    if (team.region)
        nk_gromovwassersteins_walk_f16_cuda_(&schedule, &team, nk_shapes_shared_cuda_,
                                             team.region + nk_shapes_meeting_bytes_simt_k, shared_fit, &last);
    else
        nk_gromovwassersteins_walk_f16_cuda_(&schedule, &team, nk_shapes_shared_cuda_,
                                             (char *)nk_shapes_shared_cuda_ + nk_shapes_scratch_bytes_simt_(blockDim.x),
                                             shared_fit, &last);
}

/** Walks every pair in chunks of as many pairs as the workspace has slots: the flips of a chunk,
 *  then its starts, for clouds of up to @p largest points, recording any pair above as unsolvable.
 *  Clouds that fit a block's shared memory take a block per flip or start; larger ones take teams
 *  of blocks over the workspace. */
NUMKONG_INLINE nk_status_t nk_gromov_launch_cuda_(void const *flips_kernel, void const *starts_kernel,
                                                  nk_shapes_transport_schedule_cuda_t schedule, nk_size_t point_bytes,
                                                  nk_size_t largest, void *workspace, nk_size_t workspace_bytes,
                                                  nk_stream_t stream) {
    if ((nk_size_t)workspace & (nk_default_alignment_k - 1)) return nk_misaligned_k;
    if (!largest || nk_gromov_workspace_bytes_cuda_(largest, point_bytes) > workspace_bytes)
        return nk_unexpected_dimensions_k;
    nk_u32_t supported = 0;
    nk_status_t status = nk_shapes_largest_supported_cuda_(nk_shapes_large_team_cuda_k, point_bytes, &supported,
                                                           stream);
    if (status != nk_success_k) return status;
    nk_size_t const slots = nk_gromov_slots_cuda_(largest), pairs = schedule.rows * schedule.columns;
    schedule.slots = (unsigned char *)workspace, schedule.slot_stride = nk_gromov_slot_bytes_cuda_(largest);
    schedule.plans_offset = nk_shapes_slot_plans_cuda_k, schedule.gromov_offset = nk_gromov_dense_offset_cuda_(largest);
    schedule.gromov_stride = nk_shapes_gromov_bytes_simt_(largest);
    schedule.class_counts = NUMKONG_NULL, schedule.class_count = NUMKONG_NULL, schedule.class_slots = NUMKONG_NULL;
    schedule.class_ceiling = (nk_u32_t)largest, schedule.class_limit = (nk_u32_t)largest, schedule.gromov = 1;
    schedule.team_states = NUMKONG_NULL, schedule.team_stride = nk_shapes_team_stride_cuda_(largest, point_bytes);
    schedule.teams = nk_shapes_teams_cuda_(largest, point_bytes);
    unsigned const flips_team = largest > nk_shapes_small_cloud_cuda_k ? nk_shapes_large_team_cuda_k
                                                                       : nk_shapes_small_team_cuda_k;
    nk_size_t const flips_shared_bytes = nk_shapes_team_bytes_simt_(largest, largest, flips_team, point_bytes);
    // The starts' dense steps want more threads than the flips' scans
    unsigned const starts_team = largest > 64 ? nk_shapes_large_team_cuda_k : 64;
    nk_size_t const starts_shared_bytes = nk_shapes_team_bytes_simt_(largest, largest, starts_team, 0);
    status = nk_shapes_zero_counters_cuda_(schedule.slots, schedule.slot_stride, pairs < slots ? pairs : slots, stream);
    if (status == nk_success_k && largest > supported) {
        // Past one block's shared memory, teams of blocks keep their state after the slots
        schedule.team_states = (unsigned char *)workspace +
                               nk_size_round_up_to_multiple_(slots * schedule.slot_stride, 256);
        status = nk_shapes_zero_counters_cuda_(schedule.team_states, schedule.team_stride, schedule.teams, stream);
    }
    for (nk_size_t begin = 0; begin < pairs && status == nk_success_k; begin += slots) {
        schedule.pair_begin = begin, schedule.pairs = pairs - begin < slots ? pairs - begin : slots;
        if (schedule.team_states) {
            nk_size_t const scratch_bytes = nk_shapes_scratch_bytes_simt_(nk_shapes_large_team_cuda_k);
            status = nk_shapes_launch_teams_cuda_(flips_kernel, nk_shapes_large_team_cuda_k, scratch_bytes, &schedule,
                                                  stream);
            if (status == nk_success_k)
                status = nk_shapes_launch_teams_cuda_(starts_kernel, nk_shapes_large_team_cuda_k, scratch_bytes,
                                                      &schedule, stream);
            continue;
        }
        status = nk_launch_resident_cuda_(flips_kernel, flips_team, flips_shared_bytes, flips_shared_bytes,
                                          schedule.pairs * 8, &schedule, stream);
        if (status == nk_success_k)
            status = nk_launch_resident_cuda_(starts_kernel, starts_team, starts_shared_bytes, starts_shared_bytes,
                                              schedule.pairs * 3, &schedule, stream);
    }
    return status;
}

/** The largest cloud a Gromov-Wasserstein workspace of @p workspace_bytes holds, as the size
 *  queries lay it out. */
NUMKONG_INLINE nk_size_t nk_gromov_workspace_bound_cuda_(nk_size_t workspace_bytes, nk_size_t point_bytes) {
    nk_size_t largest = nk_shapes_class_ceiling_cuda_(nk_shapes_classes_cuda_k - 1);
    while (largest && nk_gromov_workspace_bytes_cuda_(largest, point_bytes) > workspace_bytes) --largest;
    return largest;
}

NUMKONG_API nk_status_t nk_gromovwassersteins_workspace_size_f32_cuda(nk_size_t row_points, nk_size_t column_points,
                                                                      nk_size_t *bytes) {
    *bytes = nk_gromov_workspace_bytes_cuda_(row_points > column_points ? row_points : column_points, sizeof(nk_f64_t));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_gromovwassersteins_workspace_size_f16_cuda(nk_size_t row_points, nk_size_t column_points,
                                                                      nk_size_t *bytes) {
    *bytes = nk_gromov_workspace_bytes_cuda_(row_points > column_points ? row_points : column_points, sizeof(nk_f32_t));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f32_cuda(void const *const *a_packed, nk_size_t rows,
                                                              void const *const *b_packed, nk_size_t columns,
                                                              nk_f64_t *results, nk_size_t results_stride,
                                                              nk_f32_t *rotations, void *workspace,
                                                              nk_size_t workspace_bytes, nk_stream_t stream) {
    if (!rows || !columns) return nk_success_k;
    if (((nk_size_t)results | results_stride) & (sizeof(nk_f64_t) - 1) || (nk_size_t)rotations & 3)
        return nk_misaligned_k;
    nk_shapes_transport_schedule_cuda_t schedule;
    schedule.a_packed = a_packed, schedule.b_packed = b_packed, schedule.results = results;
    schedule.rotations = rotations, schedule.rows = rows, schedule.columns = columns;
    schedule.results_stride = results_stride;
    return nk_gromov_launch_cuda_((void const *)&nk_wassersteinprocrustes_packed_f32_cuda_kernel_,
                                  (void const *)&nk_gromovwassersteins_packed_f32_cuda_kernel_, schedule,
                                  sizeof(nk_f64_t), nk_gromov_workspace_bound_cuda_(workspace_bytes, sizeof(nk_f64_t)),
                                  workspace, workspace_bytes, stream);
}

NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f16_cuda(void const *const *a_packed, nk_size_t rows,
                                                              void const *const *b_packed, nk_size_t columns,
                                                              nk_f32_t *results, nk_size_t results_stride,
                                                              nk_f32_t *rotations, void *workspace,
                                                              nk_size_t workspace_bytes, nk_stream_t stream) {
    if (!rows || !columns) return nk_success_k;
    if (((nk_size_t)results | results_stride) & (sizeof(nk_f32_t) - 1) || (nk_size_t)rotations & 3)
        return nk_misaligned_k;
    nk_shapes_transport_schedule_cuda_t schedule;
    schedule.a_packed = a_packed, schedule.b_packed = b_packed, schedule.results = results;
    schedule.rotations = rotations, schedule.rows = rows, schedule.columns = columns;
    schedule.results_stride = results_stride;
    return nk_gromov_launch_cuda_((void const *)&nk_wassersteinprocrustes_packed_f16_cuda_kernel_,
                                  (void const *)&nk_gromovwassersteins_packed_f16_cuda_kernel_, schedule,
                                  sizeof(nk_f32_t), nk_gromov_workspace_bound_cuda_(workspace_bytes, sizeof(nk_f32_t)),
                                  workspace, workspace_bytes, stream);
}

#pragma endregion Gromov Wasserstein
#endif // NUMKONG_TARGET_CUDA

#pragma region Chamfer and Hausdorff

/** The Chamfer distance between two packs of F32 points on a whole warp. Every lane calls with the
 *  same packs and its own @p lane , the lanes split each cloud's points, and the warp sums their
 *  nearest distances. Distances up to @p limit are exact; past it, the warp may stop early and
 *  return a lower bound that also lies past @p limit . */
NUMKONG_DEVICE nk_f64_t nk_chamfer_warp_packed_f32_cuda_(void const *a_packed, void const *b_packed, nk_f64_t limit,
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
                    distances[flip] += nk_shuffle_xor_f64_cuda_(distances[flip], offset);
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

/** The Chamfer distance between two packs of F16 points on a whole warp. Every lane calls with the
 *  same packs and its own @p lane , the lanes split each cloud's points, and the warp sums their
 *  nearest distances. Distances up to @p limit are exact; past it, the warp may stop early and
 *  return a lower bound that also lies past @p limit . */
NUMKONG_DEVICE nk_f32_t nk_chamfer_warp_packed_f16_cuda_(void const *a_packed, void const *b_packed, nk_f32_t limit,
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
                    distances[flip] += nk_shuffle_xor_f32_cuda_(distances[flip], offset);
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

/** The Hausdorff distance between two packs of F32 points on a whole warp. Every lane calls with
 *  the same packs and its own @p lane , the lanes split each cloud's points, and the warp keeps the
 *  farthest nearest square. Distances up to @p limit are exact; past it, the warp may return early,
 *  even before reading a point, with a lower bound that also lies past @p limit . */
NUMKONG_DEVICE nk_f64_t nk_hausdorff_warp_packed_f32_cuda_(void const *a_packed, void const *b_packed, nk_f64_t limit,
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
                    nk_f64_t const other = nk_shuffle_xor_f64_cuda_(farthest[flip], offset);
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

/** The Hausdorff distance between two packs of F16 points on a whole warp. Every lane calls with
 *  the same packs and its own @p lane , the lanes split each cloud's points, and the warp keeps the
 *  farthest nearest square. Distances up to @p limit are exact; past it, the warp may return early,
 *  even before reading a point, with a lower bound that also lies past @p limit . */
NUMKONG_DEVICE nk_f32_t nk_hausdorff_warp_packed_f16_cuda_(void const *a_packed, void const *b_packed, nk_f32_t limit,
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
                    nk_f32_t const other = nk_shuffle_xor_f32_cuda_(farthest[flip], offset);
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

#endif // NUMKONG_ARCH_CUDA_
#endif // NUMKONG_SHAPES_CUDA_CUH
