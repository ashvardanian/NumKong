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

    /** Pairs the default Wasserstein-Procrustes workspace keeps in flight, enough for every
     *  multiprocessor many times over. */
    nk_shapes_pair_slots_cuda_k = 16384,

    /** The most pairs the default Gromov-Wasserstein workspace keeps in flight, whose slots also
     *  hold dense matrices within its byte budget. */
    nk_gromov_pair_slots_cuda_k = 4096,

    /** Doubles of a flip's or start's record: its least cost or loss, then its rotation's 9. */
    nk_shapes_record_cuda_k = 10,

    /** Bytes before the pair slots: the call's largest and smallest clouds, on one cache line. */
    nk_shapes_workspace_header_cuda_k = 128,

    /** Offset of a slot's 8 flip records, after its counters of flips and of starts done. */
    nk_shapes_slot_records_cuda_k = 8,

    /** Offset of a slot's 3 start records. */
    nk_shapes_slot_starts_cuda_k = 8 + 8 * nk_shapes_record_cuda_k * 8,

    /** Offset of a Gromov-Wasserstein slot's flip plans. */
    nk_shapes_slot_plans_cuda_k = nk_shapes_slot_starts_cuda_k + 3 * nk_shapes_record_cuda_k * 8,

    /** MiB the default Gromov-Wasserstein workspace spends on its slots. */
    nk_gromov_workspace_mebibytes_cuda_k = 2048,
};

/** One transport launch: the packs, the outputs, the pair slots of the workspace, a slot's layout,
 *  and the pairs it walks, from @c pair_begin on. Gromov-Wasserstein slots add each flip's best
 *  plan at @c plans_offset , then the dense state of each start, @c gromov_stride bytes apart from
 *  @c gromov_offset on. */
typedef struct {
    void const *const *a_packed;
    void const *const *b_packed;
    void *results;
    nk_f32_t *rotations;
    nk_size_t rows, columns, results_stride;
    nk_size_t pair_begin, pairs;
    unsigned char *slots;
    nk_size_t slot_stride, plans_offset, gromov_offset, gromov_stride;
    nk_u32_t largest;
    int gromov, unequal;
} nk_shapes_transport_schedule_cuda_t;

/** Bytes of a Wasserstein-Procrustes pair slot: two counters and 8 flip records. */
NUMKONG_INLINE nk_size_t nk_shapes_slot_bytes_cuda_(void) { return nk_shapes_slot_starts_cuda_k; }

/** Bytes of a Gromov-Wasserstein pair slot for clouds of up to @p largest points: the records, the
 *  8 flips' plans of up to 4 × @p largest arcs each, and the 3 starts' dense state. */
NUMKONG_INLINE nk_size_t nk_gromov_slot_bytes_cuda_(nk_size_t largest) {
    return nk_size_round_up_to_multiple_(nk_shapes_slot_plans_cuda_k + 8 * 3 * 4 * largest * sizeof(nk_u32_t), 128) +
           3 * nk_shapes_gromov_bytes_simt_(largest);
}

NUMKONG_INLINE nk_status_t nk_shapes_memset_cuda_(void *pointer, int value, nk_size_t bytes, nk_stream_t stream) {
    return cudaMemsetAsync(pointer, value, bytes, (cudaStream_t)stream) == cudaSuccess ? nk_success_k
                                                                                       : nk_device_code_mismatch_k;
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

/** The largest cloud of both batches of f32 packs, raised into @p largest , and the complement of
 *  the smallest into the word after it. */
static __global__ void nk_shapes_largest_f32_cuda_kernel_(void const *const *a_packed, nk_size_t rows,
                                                          void const *const *b_packed, nk_size_t columns,
                                                          nk_u32_t *largest) {
    for (nk_size_t index = blockIdx.x * (nk_size_t)blockDim.x + threadIdx.x; index < rows + columns;
         index += (nk_size_t)gridDim.x * blockDim.x) {
        void const *packed = index < rows ? a_packed[index] : b_packed[index - rows];
        nk_u32_t const points = (nk_u32_t)((nk_shapes_packed_header_f64_t const *)packed)->points;
        atomicMax(largest, points), atomicMax(largest + 1, ~points);
    }
}

static __global__ void nk_shapes_largest_f16_cuda_kernel_(void const *const *a_packed, nk_size_t rows,
                                                          void const *const *b_packed, nk_size_t columns,
                                                          nk_u32_t *largest) {
    for (nk_size_t index = blockIdx.x * (nk_size_t)blockDim.x + threadIdx.x; index < rows + columns;
         index += (nk_size_t)gridDim.x * blockDim.x) {
        void const *packed = index < rows ? a_packed[index] : b_packed[index - rows];
        nk_u32_t const points = ((nk_shapes_packed_header_f32_t const *)packed)->points;
        atomicMax(largest, points), atomicMax(largest + 1, ~points);
    }
}

/** Reads the largest cloud of the call and the complement of the smallest back through the
 *  workspace's header, into @p extremes . */
NUMKONG_INLINE nk_status_t nk_shapes_largest_cuda_(void const *kernel, void const *const *a_packed, nk_size_t rows,
                                                   void const *const *b_packed, nk_size_t columns, void *workspace,
                                                   nk_u32_t *extremes, nk_stream_t stream) {
    nk_status_t status = nk_shapes_memset_cuda_(workspace, 0, nk_shapes_workspace_header_cuda_k, stream);
    if (status != nk_success_k) return status;
    nk_u32_t *largest_device = (nk_u32_t *)workspace;
    void *arguments[5] = {(void *)&a_packed, (void *)&rows, (void *)&b_packed, (void *)&columns,
                          (void *)&largest_device};
    nk_size_t const blocks = nk_size_divide_round_up_(rows + columns, 256);
    status = nk_launch_cuda_(kernel, blocks < 1024 ? blocks : 1024, 256, arguments, 0, stream);
    if (status != nk_success_k) return status;
    return nk_read_cuda_(extremes, workspace, 2 * sizeof(nk_u32_t), stream);
}

/** The team size and shared memory of a launch over clouds of up to @p largest points, or a
 *  status if they outgrow the device. */
NUMKONG_INLINE nk_status_t nk_shapes_team_cuda_(nk_u32_t largest, nk_size_t point_bytes, int unequal, unsigned *team,
                                                nk_size_t *shared_bytes, nk_stream_t stream) {
    int shared_limit = 0;
    nk_status_t const status = nk_device_attribute_cuda_(cudaDevAttrMaxSharedMemoryPerBlockOptin, &shared_limit,
                                                         stream);
    if (status != nk_success_k) return status;
    *team = largest > nk_shapes_small_cloud_cuda_k ? nk_shapes_large_team_cuda_k : nk_shapes_small_team_cuda_k;
    *shared_bytes = nk_shapes_team_bytes_simt_(largest, largest, *team, point_bytes, unequal);
    return *shared_bytes > (nk_size_t)shared_limit ? nk_unexpected_dimensions_k : nk_success_k;
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
    results_row[column] = empty ? best : nk_sqrt_f64_simt_(best);
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
    results_row[column] = empty ? best : nk_sqrt_f32_simt_(best);
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

/** One block per flip of each pair. Wasserstein-Procrustes finishes each pair in its last block;
 *  for Gromov-Wasserstein, each flip only leaves its record and best plan. */
static __global__ void nk_wassersteinprocrustes_packed_f32_cuda_kernel_(
    nk_shapes_transport_schedule_cuda_t const schedule) {
    extern __shared__ nk_f64_t nk_shapes_shared_cuda_[];
    __shared__ nk_f64_t shared_rotation[9];
    __shared__ int last;
    unsigned parity = 0;
    nk_u32_t const plan_arcs = 4 * schedule.largest;
    for (nk_size_t item = blockIdx.x; item < schedule.pairs * 8; item += gridDim.x) {
        nk_size_t const slot_index = item >> 3, pair = schedule.pair_begin + slot_index;
        unsigned char *slot = schedule.slots + slot_index * schedule.slot_stride;
        int const flip = (int)(item & 7);
        void const *a = schedule.a_packed[pair / schedule.columns], *b = schedule.b_packed[pair % schedule.columns];
        int const swapped = nk_shapes_column_first_f32_cuda_(a, b);
        void const *row_pack = swapped ? b : a, *column_pack = swapped ? a : b;
        nk_u32_t const rows = (nk_u32_t)((nk_shapes_packed_header_f64_t const *)row_pack)->points;
        nk_u32_t const columns = (nk_u32_t)((nk_shapes_packed_header_f64_t const *)column_pack)->points;
        nk_f64_t cost, rotation[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        if (!rows || !columns) cost = rows == columns ? 0 : NUMKONG_F64_INF;
        else {
            nk_shapes_points_f32_simt_t points;
            nk_shapes_network_simt_t network;
            nk_shapes_points_carve_f32_simt_(nk_shapes_shared_cuda_, schedule.largest, schedule.largest, &points);
            nk_shapes_network_carve_simt_(nk_shapes_shared_cuda_, schedule.largest, schedule.largest, rows, columns,
                                          sizeof(nk_f64_t), &network);
            nk_f64_t const radii = ((nk_shapes_packed_header_f64_t const *)row_pack)->max_radius +
                                   ((nk_shapes_packed_header_f64_t const *)column_pack)->max_radius;
            network.tolerance = radii * radii * 1e-12;
            nk_size_t row_stride, column_stride;
            nk_f32_t const *row_axes = nk_shapes_packed_axes_f32_simt_(row_pack, &row_stride);
            nk_f32_t const *column_axes = nk_shapes_packed_axes_f32_simt_(column_pack, &column_stride);
            nk_u32_t *saved_plan = schedule.gromov ? (nk_u32_t *)(slot + schedule.plans_offset) + flip * 3 * plan_arcs
                                                   : NUMKONG_NULL;
            nk_shapes_load_rows_f32_simt_(&points, row_axes, row_stride, rows);
            nk_wassersteinprocrustes_flip_f32_simt_(&network, &points, column_axes, column_stride, flip,
                                                    shared_rotation, &cost, rotation, saved_plan, &parity);
        }
        nk_shapes_record_cuda_(slot, nk_shapes_slot_records_cuda_k, (nk_u32_t *)slot, flip, 8, cost, rotation, &last);
        if (last && !schedule.gromov && threadIdx.x == 0)
            nk_wassersteinprocrustes_finish_f32_cuda_(&schedule, pair,
                                                      (nk_f64_t *)(slot + nk_shapes_slot_records_cuda_k));
        __syncthreads();
    }
}

static __global__ void nk_wassersteinprocrustes_packed_f16_cuda_kernel_(
    nk_shapes_transport_schedule_cuda_t const schedule) {
    extern __shared__ nk_f64_t nk_shapes_shared_cuda_[];
    __shared__ nk_f32_t shared_rotation[9];
    __shared__ int last;
    unsigned parity = 0;
    nk_u32_t const plan_arcs = 4 * schedule.largest;
    for (nk_size_t item = blockIdx.x; item < schedule.pairs * 8; item += gridDim.x) {
        nk_size_t const slot_index = item >> 3, pair = schedule.pair_begin + slot_index;
        unsigned char *slot = schedule.slots + slot_index * schedule.slot_stride;
        int const flip = (int)(item & 7);
        void const *a = schedule.a_packed[pair / schedule.columns], *b = schedule.b_packed[pair % schedule.columns];
        int const swapped = nk_shapes_column_first_f16_cuda_(a, b);
        void const *row_pack = swapped ? b : a, *column_pack = swapped ? a : b;
        nk_u32_t const rows = ((nk_shapes_packed_header_f32_t const *)row_pack)->points;
        nk_u32_t const columns = ((nk_shapes_packed_header_f32_t const *)column_pack)->points;
        nk_f32_t cost, rotation[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        if (!rows || !columns) cost = rows == columns ? 0 : NUMKONG_F32_INF;
        else {
            nk_shapes_points_f16_simt_t points;
            nk_shapes_network_simt_t network;
            nk_shapes_points_carve_f16_simt_(nk_shapes_shared_cuda_, schedule.largest, schedule.largest, &points);
            nk_shapes_network_carve_simt_(nk_shapes_shared_cuda_, schedule.largest, schedule.largest, rows, columns,
                                          sizeof(nk_f32_t), &network);
            nk_f32_t const radii = ((nk_shapes_packed_header_f32_t const *)row_pack)->max_radius +
                                   ((nk_shapes_packed_header_f32_t const *)column_pack)->max_radius;
            network.tolerance = (nk_f64_t)(radii * radii) * 1e-12;
            nk_size_t row_stride, column_stride;
            nk_f16_t const *row_axes = nk_shapes_packed_axes_f16_simt_(row_pack, &row_stride);
            nk_f16_t const *column_axes = nk_shapes_packed_axes_f16_simt_(column_pack, &column_stride);
            nk_u32_t *saved_plan = schedule.gromov ? (nk_u32_t *)(slot + schedule.plans_offset) + flip * 3 * plan_arcs
                                                   : NUMKONG_NULL;
            nk_shapes_load_rows_f16_simt_(&points, row_axes, row_stride, rows);
            nk_wassersteinprocrustes_flip_f16_simt_(&network, &points, column_axes, column_stride, flip,
                                                    shared_rotation, &cost, rotation, saved_plan, &parity);
        }
        nk_f64_t wide_rotation[9];
        for (int cell = 0; cell != 9; ++cell) wide_rotation[cell] = rotation[cell];
        nk_shapes_record_cuda_(slot, nk_shapes_slot_records_cuda_k, (nk_u32_t *)slot, flip, 8, cost, wide_rotation,
                               &last);
        if (last && !schedule.gromov && threadIdx.x == 0)
            nk_wassersteinprocrustes_finish_f16_cuda_(&schedule, pair,
                                                      (nk_f64_t *)(slot + nk_shapes_slot_records_cuda_k));
        __syncthreads();
    }
}

/** Fills the schedule's slots and launch shape from the workspace, after reading the largest cloud.
 *  Zeroes every slot's counters, and returns the slots in @p slots . */
NUMKONG_INLINE nk_status_t nk_shapes_transport_prepare_cuda_(
    void const *largest_kernel, nk_shapes_transport_schedule_cuda_t *schedule, nk_size_t point_bytes, void *workspace,
    nk_size_t workspace_bytes, unsigned *team, nk_size_t *shared_bytes, nk_size_t *slots, nk_stream_t stream) {
    if ((nk_size_t)workspace & (nk_default_alignment_k - 1)) return nk_misaligned_k;
    if (workspace_bytes < nk_shapes_workspace_header_cuda_k) return nk_unexpected_dimensions_k;
    nk_u32_t extremes[2] = {0, 0};
    nk_status_t status = nk_shapes_largest_cuda_(largest_kernel, schedule->a_packed, schedule->rows, schedule->b_packed,
                                                 schedule->columns, workspace, extremes, stream);
    if (status != nk_success_k) return status;
    schedule->largest = extremes[0] ? extremes[0] : 1;
    schedule->unequal = extremes[0] != ~extremes[1];
    status = nk_shapes_team_cuda_(schedule->largest, point_bytes, schedule->unequal, team, shared_bytes, stream);
    if (status != nk_success_k) return status;
    schedule->slot_stride = schedule->gromov ? nk_gromov_slot_bytes_cuda_(schedule->largest)
                                             : nk_shapes_slot_bytes_cuda_();
    schedule->plans_offset = nk_shapes_slot_plans_cuda_k;
    schedule->gromov_stride = nk_shapes_gromov_bytes_simt_(schedule->largest);
    schedule->gromov_offset = nk_size_round_up_to_multiple_(
        nk_shapes_slot_plans_cuda_k + 8 * 3 * 4 * (nk_size_t)schedule->largest * sizeof(nk_u32_t), 128);
    schedule->slots = (unsigned char *)workspace + nk_shapes_workspace_header_cuda_k;
    *slots = (workspace_bytes - nk_shapes_workspace_header_cuda_k) / schedule->slot_stride;
    if (!*slots) return nk_unexpected_dimensions_k;
    nk_size_t const pairs = schedule->rows * schedule->columns, used = pairs < *slots ? pairs : *slots;
    return cudaMemset2DAsync(schedule->slots, schedule->slot_stride, 0, 2 * sizeof(nk_u32_t), used,
                             (cudaStream_t)stream) == cudaSuccess
               ? nk_success_k
               : nk_device_code_mismatch_k;
}

/** Walks every pair in launches of as many pairs as the workspace has slots. */
NUMKONG_INLINE nk_status_t nk_shapes_transport_launch_cuda_(void const *kernel, void const *largest_kernel,
                                                            nk_shapes_transport_schedule_cuda_t schedule,
                                                            nk_size_t point_bytes, void *workspace,
                                                            nk_size_t workspace_bytes, nk_stream_t stream) {
    unsigned team = 0;
    nk_size_t shared_bytes = 0, slots = 0;
    schedule.gromov = 0;
    nk_status_t status = nk_shapes_transport_prepare_cuda_(largest_kernel, &schedule, point_bytes, workspace,
                                                           workspace_bytes, &team, &shared_bytes, &slots, stream);
    nk_size_t const pairs = schedule.rows * schedule.columns;
    for (nk_size_t begin = 0; begin < pairs && status == nk_success_k; begin += slots) {
        schedule.pair_begin = begin, schedule.pairs = pairs - begin < slots ? pairs - begin : slots;
        status = nk_launch_resident_cuda_(kernel, team, shared_bytes, shared_bytes, schedule.pairs * 8, &schedule,
                                          stream);
    }
    return status;
}

NUMKONG_API nk_status_t nk_wassersteinprocrustes_workspace_size_f32_cuda(nk_size_t row_points, nk_size_t column_points,
                                                                         nk_size_t *bytes) {
    nk_unused_(row_points), nk_unused_(column_points);
    *bytes = nk_shapes_workspace_header_cuda_k + nk_shapes_pair_slots_cuda_k * nk_shapes_slot_bytes_cuda_();
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_wassersteinprocrustes_workspace_size_f16_cuda(nk_size_t row_points, nk_size_t column_points,
                                                                         nk_size_t *bytes) {
    nk_unused_(row_points), nk_unused_(column_points);
    *bytes = nk_shapes_workspace_header_cuda_k + nk_shapes_pair_slots_cuda_k * nk_shapes_slot_bytes_cuda_();
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
    return nk_shapes_transport_launch_cuda_((void const *)&nk_wassersteinprocrustes_packed_f32_cuda_kernel_,
                                            (void const *)&nk_shapes_largest_f32_cuda_kernel_, schedule,
                                            sizeof(nk_f64_t), workspace, workspace_bytes, stream);
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
    return nk_shapes_transport_launch_cuda_((void const *)&nk_wassersteinprocrustes_packed_f16_cuda_kernel_,
                                            (void const *)&nk_shapes_largest_f16_cuda_kernel_, schedule,
                                            sizeof(nk_f32_t), workspace, workspace_bytes, stream);
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
    if (!row_header->points || !column_header->points) results_row[column] = records[0];
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
    if (!row_header->points || !column_header->points) results_row[column] = (nk_f32_t)records[0];
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

/** One block per start of each pair, after the Wasserstein-Procrustes launch left every flip's
 *  record and best plan in the pair's slot. */
static __global__ void nk_gromovwassersteins_packed_f32_cuda_kernel_(
    nk_shapes_transport_schedule_cuda_t const schedule) {
    extern __shared__ nk_f64_t nk_shapes_shared_cuda_[];
    __shared__ nk_f64_t shared_fit[9];
    __shared__ int last;
    unsigned parity = 0;
    nk_u32_t const plan_arcs = 4 * schedule.largest;
    for (nk_size_t item = blockIdx.x; item < schedule.pairs * 3; item += gridDim.x) {
        nk_size_t const slot_index = item / 3, pair = schedule.pair_begin + slot_index;
        unsigned char *slot = schedule.slots + slot_index * schedule.slot_stride;
        int const start = (int)(item % 3);
        void const *a = schedule.a_packed[pair / schedule.columns], *b = schedule.b_packed[pair % schedule.columns];
        int const swapped = nk_shapes_column_first_f32_cuda_(a, b);
        void const *row_pack = swapped ? b : a, *column_pack = swapped ? a : b;
        nk_u32_t const rows = (nk_u32_t)((nk_shapes_packed_header_f64_t const *)row_pack)->points;
        nk_u32_t const columns = (nk_u32_t)((nk_shapes_packed_header_f64_t const *)column_pack)->points;
        nk_f64_t loss, fit[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        if (!rows || !columns) loss = rows == columns ? 0 : NUMKONG_F64_INF;
        else {
            nk_shapes_network_simt_t network;
            nk_shapes_gromov_simt_t gromov;
            nk_shapes_network_carve_simt_(nk_shapes_shared_cuda_, schedule.largest, schedule.largest, rows, columns, 0,
                                          &network);
            nk_shapes_gromov_carve_simt_(slot + schedule.gromov_offset + start * schedule.gromov_stride,
                                         schedule.largest, &gromov);
            nk_size_t row_stride, column_stride;
            nk_f32_t const *row_axes = nk_shapes_packed_axes_f32_simt_(row_pack, &row_stride);
            nk_f32_t const *column_axes = nk_shapes_packed_axes_f32_simt_(column_pack, &column_stride);
            nk_shapes_distances_f32_simt_(row_axes, row_stride, rows, gromov.row_distances);
            nk_shapes_distances_f32_simt_(column_axes, column_stride, columns, gromov.column_distances);
            nk_f64_t const row_spread = nk_shapes_distance_spread_simt_(&network, gromov.row_distances, rows);
            nk_f64_t const column_spread = nk_shapes_distance_spread_simt_(&network, gromov.column_distances, columns);
            if (start == 2) nk_shapes_product_start_simt_(&network, &gromov);
            else {
                int const flip = nk_gromov_start_flip_cuda_((nk_f64_t const *)(slot + nk_shapes_slot_records_cuda_k),
                                                            start);
                nk_shapes_network_load_simt_(
                    &network, (nk_u32_t const *)(slot + schedule.plans_offset) + flip * 3 * plan_arcs, plan_arcs);
                nk_shapes_plan_start_simt_(&network, &gromov);
            }
            nk_f64_t const plan_score = nk_shapes_frank_wolfe_simt_(&network, &gromov, &parity);
            loss = row_spread + column_spread - 2 * plan_score;
            nk_shapes_plan_fit_f32_simt_(&network, &gromov, row_axes, row_stride, column_axes, column_stride,
                                         shared_fit, fit);
        }
        nk_shapes_record_cuda_(slot, nk_shapes_slot_starts_cuda_k, (nk_u32_t *)slot + 1, start, 3, loss, fit, &last);
        if (last && threadIdx.x == 0)
            nk_gromovwassersteins_finish_f32_cuda_(&schedule, pair, (nk_f64_t *)(slot + nk_shapes_slot_starts_cuda_k));
        __syncthreads();
    }
}

static __global__ void nk_gromovwassersteins_packed_f16_cuda_kernel_(
    nk_shapes_transport_schedule_cuda_t const schedule) {
    extern __shared__ nk_f64_t nk_shapes_shared_cuda_[];
    __shared__ nk_f64_t shared_fit[9];
    __shared__ int last;
    unsigned parity = 0;
    nk_u32_t const plan_arcs = 4 * schedule.largest;
    for (nk_size_t item = blockIdx.x; item < schedule.pairs * 3; item += gridDim.x) {
        nk_size_t const slot_index = item / 3, pair = schedule.pair_begin + slot_index;
        unsigned char *slot = schedule.slots + slot_index * schedule.slot_stride;
        int const start = (int)(item % 3);
        void const *a = schedule.a_packed[pair / schedule.columns], *b = schedule.b_packed[pair % schedule.columns];
        int const swapped = nk_shapes_column_first_f16_cuda_(a, b);
        void const *row_pack = swapped ? b : a, *column_pack = swapped ? a : b;
        nk_u32_t const rows = ((nk_shapes_packed_header_f32_t const *)row_pack)->points;
        nk_u32_t const columns = ((nk_shapes_packed_header_f32_t const *)column_pack)->points;
        nk_f64_t loss, fit[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        if (!rows || !columns) loss = rows == columns ? 0 : NUMKONG_F64_INF;
        else {
            nk_shapes_network_simt_t network;
            nk_shapes_gromov_simt_t gromov;
            nk_shapes_network_carve_simt_(nk_shapes_shared_cuda_, schedule.largest, schedule.largest, rows, columns, 0,
                                          &network);
            nk_shapes_gromov_carve_simt_(slot + schedule.gromov_offset + start * schedule.gromov_stride,
                                         schedule.largest, &gromov);
            nk_size_t row_stride, column_stride;
            nk_f16_t const *row_axes = nk_shapes_packed_axes_f16_simt_(row_pack, &row_stride);
            nk_f16_t const *column_axes = nk_shapes_packed_axes_f16_simt_(column_pack, &column_stride);
            nk_shapes_distances_f16_simt_(row_axes, row_stride, rows, gromov.row_distances);
            nk_shapes_distances_f16_simt_(column_axes, column_stride, columns, gromov.column_distances);
            nk_f64_t const row_spread = nk_shapes_distance_spread_simt_(&network, gromov.row_distances, rows);
            nk_f64_t const column_spread = nk_shapes_distance_spread_simt_(&network, gromov.column_distances, columns);
            if (start == 2) nk_shapes_product_start_simt_(&network, &gromov);
            else {
                int const flip = nk_gromov_start_flip_cuda_((nk_f64_t const *)(slot + nk_shapes_slot_records_cuda_k),
                                                            start);
                nk_shapes_network_load_simt_(
                    &network, (nk_u32_t const *)(slot + schedule.plans_offset) + flip * 3 * plan_arcs, plan_arcs);
                nk_shapes_plan_start_simt_(&network, &gromov);
            }
            nk_f64_t const plan_score = nk_shapes_frank_wolfe_simt_(&network, &gromov, &parity);
            loss = row_spread + column_spread - 2 * plan_score;
            nk_shapes_plan_fit_f16_simt_(&network, &gromov, row_axes, row_stride, column_axes, column_stride,
                                         shared_fit, fit);
        }
        nk_shapes_record_cuda_(slot, nk_shapes_slot_starts_cuda_k, (nk_u32_t *)slot + 1, start, 3, loss, fit, &last);
        if (last && threadIdx.x == 0)
            nk_gromovwassersteins_finish_f16_cuda_(&schedule, pair, (nk_f64_t *)(slot + nk_shapes_slot_starts_cuda_k));
        __syncthreads();
    }
}

/** Walks every pair in chunks of as many pairs as the workspace has slots: the flips of a chunk,
 *  then its starts. */
NUMKONG_INLINE nk_status_t nk_gromov_launch_cuda_(void const *flips_kernel, void const *starts_kernel,
                                                  void const *largest_kernel,
                                                  nk_shapes_transport_schedule_cuda_t schedule, nk_size_t point_bytes,
                                                  void *workspace, nk_size_t workspace_bytes, nk_stream_t stream) {
    unsigned team = 0;
    nk_size_t shared_bytes = 0, slots = 0;
    schedule.gromov = 1;
    nk_status_t status = nk_shapes_transport_prepare_cuda_(largest_kernel, &schedule, point_bytes, workspace,
                                                           workspace_bytes, &team, &shared_bytes, &slots, stream);
    // The starts' dense steps want more threads than the flips' scans
    unsigned const starts_team = schedule.largest > 64 ? nk_shapes_large_team_cuda_k : 64;
    nk_size_t const starts_shared_bytes = nk_shapes_team_bytes_simt_(schedule.largest, schedule.largest, starts_team, 0,
                                                                     schedule.unequal);
    nk_size_t const pairs = schedule.rows * schedule.columns;
    for (nk_size_t begin = 0; begin < pairs && status == nk_success_k; begin += slots) {
        schedule.pair_begin = begin, schedule.pairs = pairs - begin < slots ? pairs - begin : slots;
        status = nk_launch_resident_cuda_(flips_kernel, team, shared_bytes, shared_bytes, schedule.pairs * 8, &schedule,
                                          stream);
        if (status == nk_success_k)
            status = nk_launch_resident_cuda_(starts_kernel, starts_team, starts_shared_bytes, starts_shared_bytes,
                                              schedule.pairs * 3, &schedule, stream);
    }
    return status;
}

/** A Gromov-Wasserstein workspace: as many pair slots as the budget holds, at least one. */
NUMKONG_INLINE nk_size_t nk_gromov_workspace_bytes_cuda_(nk_size_t row_points, nk_size_t column_points) {
    nk_size_t const largest = row_points > column_points ? row_points : column_points;
    nk_size_t const slot_bytes = nk_gromov_slot_bytes_cuda_(largest ? largest : 1);
    nk_size_t slots = ((nk_size_t)nk_gromov_workspace_mebibytes_cuda_k << 20) / slot_bytes;
    slots = slots < 1 ? 1 : slots > nk_gromov_pair_slots_cuda_k ? nk_gromov_pair_slots_cuda_k : slots;
    return nk_shapes_workspace_header_cuda_k + slots * slot_bytes;
}

NUMKONG_API nk_status_t nk_gromovwassersteins_workspace_size_f32_cuda(nk_size_t row_points, nk_size_t column_points,
                                                                      nk_size_t *bytes) {
    *bytes = nk_gromov_workspace_bytes_cuda_(row_points, column_points);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_gromovwassersteins_workspace_size_f16_cuda(nk_size_t row_points, nk_size_t column_points,
                                                                      nk_size_t *bytes) {
    *bytes = nk_gromov_workspace_bytes_cuda_(row_points, column_points);
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
                                  (void const *)&nk_gromovwassersteins_packed_f32_cuda_kernel_,
                                  (void const *)&nk_shapes_largest_f32_cuda_kernel_, schedule, sizeof(nk_f64_t),
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
                                  (void const *)&nk_gromovwassersteins_packed_f16_cuda_kernel_,
                                  (void const *)&nk_shapes_largest_f16_cuda_kernel_, schedule, sizeof(nk_f32_t),
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
