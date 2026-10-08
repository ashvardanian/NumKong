/**
 *  @file include/numkong/attention/cuda.cuh
 *  @author Ash Vardanian
 *  @date October 3, 2026
 *  @brief CUDA baseline of packed attention: the block scan, fallback and backward loops over warp
 *      shuffles, the pack, attention and RoPE launches, the generators every CUDA capability
 *      instantiates its exports with, and the baseline @c cuda exports.
 *
 *  @sa include/numkong/attention/simt.cuh
 *  @sa include/numkong/attention/rocm.cuh
 */
#ifndef NUMKONG_ATTENTION_CUDA_CUH
#define NUMKONG_ATTENTION_CUDA_CUH

#if NUMKONG_ARCH_CUDA_

#include "numkong/cuda.cuh"
#include "numkong/attention/simt.cuh"
#include "numkong/cast/simt.cuh" // `nk_bf16_to_f32_simt_`, `nk_e4m3_to_f32_simt_`

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Schedule

/** Inclusive block-wide prefix sum of one value per thread; the block total lands in @p total. */
NUMKONG_DEVICE nk_u64_t nk_attention_block_scan_cuda_(nk_u64_t value, nk_u64_t *warp_totals, nk_u64_t *total) {
    unsigned const lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
#pragma unroll
    for (unsigned offset = 1; offset < 32; offset <<= 1) {
        nk_u64_t const other = nk_shuffle_up_u64_cuda_(value, offset);
        if (lane >= offset) value += other;
    }
    if (lane == 31) warp_totals[warp] = value;
    __syncthreads();
    nk_u64_t before = 0, sum = 0;
    for (unsigned other_warp = 0; other_warp < warps; ++other_warp) {
        nk_u64_t const warp_total = warp_totals[other_warp];
        before += other_warp < warp ? warp_total : 0, sum += warp_total;
    }
    // Every thread reads the totals before a following scan rewrites them.
    __syncthreads();
    *total = sum;
    return value + before;
}

/** Fills @p prefix with the running item counts of the 128 segments from
 *  `schedule->chunk_first`. */
NUMKONG_DEVICE void nk_attention_schedule_chunk_cuda_(nk_attention_schedule_t const *schedule, nk_u64_t *prefix,
                                                      nk_u64_t *warp_totals) {
    nk_u64_t const items = nk_attention_segment_items_(schedule, schedule->chunk_first + threadIdx.x,
                                                       nk_attention_block_rows_k);
    nk_u64_t total;
    nk_u64_t const inclusive = nk_attention_block_scan_cuda_(items, warp_totals, &total);
    prefix[threadIdx.x + 1] = inclusive;
    if (threadIdx.x == 0) prefix[0] = 0;
    __syncthreads();
}

/** Finds work item @p item, returning 0 once the window has no more; every thread of the block
 *  calls it in step. */
NUMKONG_DEVICE int nk_attention_schedule_next_cuda_(nk_attention_schedule_t *schedule, nk_u64_t *prefix,
                                                    nk_u64_t *warp_totals, nk_size_t item, nk_attention_work_t *work) {
    while (item >= schedule->items_before + prefix[nk_attention_threads_k]) {
        if (schedule->chunk_first + nk_attention_threads_k >= schedule->segment_end) return 0;
        schedule->items_before += prefix[nk_attention_threads_k];
        schedule->chunk_first += nk_attention_threads_k;
        __syncthreads();
        nk_attention_schedule_chunk_cuda_(schedule, prefix, warp_totals);
    }
    nk_u64_t const local = item - schedule->items_before;
    unsigned low = 0, high = nk_attention_threads_k;
    while (high - low > 1) {
        unsigned const middle = (low + high) >> 1;
        if (prefix[middle] <= local) low = middle;
        else high = middle;
    }
    nk_attention_segment_work_(schedule, schedule->chunk_first + low, (nk_size_t)(local - prefix[low]),
                               nk_attention_block_rows_k, work);
    return 1;
}

/** @c nk_attention_schedule_init_, then primes the schedule's first chunk; every thread of the
 *  block calls it in step. */
NUMKONG_DEVICE int nk_attention_schedule_start_cuda_(nk_attention_arguments_t const *arguments,
                                                     nk_attention_schedule_t *schedule, nk_u64_t *prefix,
                                                     nk_u64_t *warp_totals) {
    if (!nk_attention_schedule_init_(arguments, schedule)) return 0;
    nk_attention_schedule_chunk_cuda_(schedule, prefix, warp_totals);
    return 1;
}

#pragma endregion Schedule

#pragma region Tile

/** The dot product of @p query_row and @p key_row, shared by @p lanes lanes and returned in each:
 *  F32 FMAs, or for I8 an exact I32 sum like the serial backend's, converted once. */
NUMKONG_DEVICE nk_f32_t nk_attention_score_cuda_(nk_dtype_t dtype, unsigned char const *query_row,
                                                 unsigned char const *key_row, nk_size_t depth, unsigned lane,
                                                 unsigned lanes) {
    if (dtype == nk_i8_k) {
        nk_u32_t sum = 0;
        for (nk_size_t element = lane; element < depth; element += lanes)
            sum += (nk_u32_t)((nk_i32_t)(signed char)query_row[element] * (nk_i32_t)(signed char)key_row[element]);
        for (unsigned offset = lanes / 2; offset != 0; offset >>= 1) sum += nk_shuffle_xor_u32_cuda_(sum, offset);
        return (nk_f32_t)(nk_i32_t)sum;
    }
    nk_f32_t sum = 0;
    for (nk_size_t element = lane; element < depth; element += lanes)
        sum = fmaf(nk_attention_decode_(dtype, query_row, element), nk_attention_decode_(dtype, key_row, element), sum);
    for (unsigned offset = lanes / 2; offset != 0; offset >>= 1) sum += nk_shuffle_xor_f32_cuda_(sum, offset);
    return sum;
}

/** The U8 weight the I8 contract gives a softmax weight in [0, 1]: ⌊255 · w + ½⌋, as F32. */
NUMKONG_DEVICE nk_f32_t nk_attention_quantize_weight_cuda_(nk_f32_t weight) {
    // A round-down add of 2²³ truncates at the full F32 rate, where sm_103 converts at 2 per clock.
    return __fadd_rd(weight * 255.0f + 0.5f, 8388608.0f) - 8388608.0f;
}

/** The whole baseline kernel, and the tensor-core capabilities' kernel past depth 256: one warp per
 *  row attending to its keys in the serial backend's two sweeps, the largest score first, then the
 *  weighted V rows summed into the output row, so that every depth fits. */
NUMKONG_DEVICE void nk_attention_fallback_cuda_(nk_dtype_t dtype, nk_attention_arguments_t const *arguments) {
    __shared__ nk_u64_t prefix[nk_attention_threads_k + 1];
    __shared__ nk_u64_t warp_totals[nk_attention_threads_k / 32];
    unsigned const lanes = nk_warp_lanes_cuda_(), lane = threadIdx.x % lanes, group = threadIdx.x / lanes;
    unsigned const groups = nk_attention_threads_k / lanes;
    nk_size_t const element_bytes = nk_attention_element_bytes_(dtype), depth = arguments->depth;
    nk_size_t const row_bytes = nk_size_round_up_to_multiple_(depth * element_bytes, nk_attention_step_bytes_k);
    nk_f32_t const scale2 = arguments->scale2;
    nk_attention_schedule_t schedule;
    if (!nk_attention_schedule_start_cuda_(arguments, &schedule, prefix, warp_totals)) return;
    nk_attention_work_t work;
    for (nk_size_t item = blockIdx.x; nk_attention_schedule_next_cuda_(&schedule, prefix, warp_totals, item, &work);
         item += gridDim.x) {
        nk_size_t length, plane_bytes;
        unsigned char const *keys_plane = nk_attention_keys_plane_(arguments, &work, row_bytes, &length, &plane_bytes);
        unsigned char const *values_plane = keys_plane + arguments->key_value_head_count * plane_bytes;
        nk_size_t const value_stride = element_bytes == 2 ? 1 : plane_bytes / row_bytes;
        for (nk_size_t local = group; local < work.row_count; local += groups) {
            unsigned char const *query_row = nk_attention_query_row_(arguments, &work, local, element_bytes);
            nk_f32_t *output_row = nk_attention_output_row_(arguments, &work, local);
            nk_size_t key_begin, key_end;
            nk_diagonal_band_row_range_simt_(arguments->band, nk_attention_row_position_(&work, local, length), length,
                                             &key_begin, &key_end);
            nk_f32_t row_max = nk_attention_negative_infinity_(), weights_sum = 0;
            for (nk_size_t position = key_begin; position < key_end; ++position) {
                unsigned char const *key_row = keys_plane + position * row_bytes;
                row_max = fmaxf(row_max,
                                nk_attention_score_cuda_(dtype, query_row, key_row, depth, lane, lanes) * scale2);
            }
            for (nk_size_t element = lane; element < depth; element += lanes) output_row[element] = 0;
            for (nk_size_t position = key_begin; position < key_end; ++position) {
                nk_f32_t const score = nk_attention_score_cuda_(dtype, query_row, keys_plane + position * row_bytes,
                                                                depth, lane, lanes);
                nk_f32_t weight = exp2f(score * scale2 - row_max);
                if (dtype == nk_i8_k) weight = nk_attention_quantize_weight_cuda_(weight);
                weights_sum += weight;
                // 1-byte V is transposed, a depth row of positions per element.
                unsigned char const *value_row = element_bytes == 2 ? values_plane + position * row_bytes
                                                                    : values_plane + nk_attention_slot_(position);
                for (nk_size_t element = lane; element < depth; element += lanes)
                    output_row[element] = fmaf(weight, nk_attention_decode_(dtype, value_row, element * value_stride),
                                               output_row[element]);
            }
            nk_f32_t const inverse = weights_sum > 0 ? 1.0f / weights_sum : 0.0f;
            for (nk_size_t element = lane; element < depth; element += lanes) output_row[element] *= inverse;
            nk_f32_t *const log_sum_exp_slot = nk_attention_log_sum_exp_slot_(arguments, &work, local);
            if (log_sum_exp_slot && lane == 0)
                *log_sum_exp_slot = nk_attention_log_sum_exp_simt_(row_max, weights_sum,
                                                                   dtype == nk_i8_k ? 255.0f : 1.0f);
        }
    }
}

#pragma endregion Tile

#pragma region Pack

/** Writes the header, recording the packing @p capability, and the directory, one block walking the
 *  segments 128 at a time. */
static __global__ void nk_attention_pack_directory_cuda_kernel_(unsigned char *packed, nk_size_t key_value_head_count,
                                                                nk_size_t depth, nk_u32_t const *segment_lengths,
                                                                nk_size_t segment_count, nk_size_t row_bytes,
                                                                nk_size_t directory_bytes, nk_capability_t capability) {
    __shared__ nk_u64_t warp_totals[nk_attention_threads_k / 32];
    nk_attention_packed_header_t *header = (nk_attention_packed_header_t *)packed;
    if (threadIdx.x == 0) {
        header->heads = (nk_u32_t)key_value_head_count;
        header->depth = (nk_u32_t)depth;
        header->segments = (nk_u32_t)segment_count;
        header->capability = capability;
        for (unsigned reserved_index = 0; reserved_index < 11; ++reserved_index) header->reserved[reserved_index] = 0;
    }
    nk_u64_t *payload_offsets = (nk_u64_t *)(packed + sizeof(nk_attention_packed_header_t));
    nk_u32_t *lengths = (nk_u32_t *)(payload_offsets + segment_count + 1);
    nk_u64_t running = 0;
    for (nk_size_t chunk = 0; chunk < segment_count; chunk += blockDim.x) {
        nk_size_t const segment = chunk + threadIdx.x;
        nk_u32_t const length = segment < segment_count ? segment_lengths[segment] : 0;
        nk_u64_t const bytes = nk_attention_pack_segment_bytes_(length, key_value_head_count, nk_attention_panel_k,
                                                                row_bytes);
        nk_u64_t total;
        nk_u64_t const inclusive = nk_attention_block_scan_cuda_(bytes, warp_totals, &total);
        if (segment < segment_count) payload_offsets[segment] = running + inclusive - bytes, lengths[segment] = length;
        running += total;
    }
    if (threadIdx.x == 0) payload_offsets[segment_count] = running;
    for (nk_size_t byte = (segment_count + 1) * sizeof(nk_u64_t) + segment_count * sizeof(nk_u32_t) + threadIdx.x;
         byte < directory_bytes; byte += blockDim.x)
        packed[sizeof(nk_attention_packed_header_t) + byte] = 0;
}

#pragma endregion Pack

#pragma region Backward

/** Finds work item @p item past the cursor at @p segment_first, @p items_before items in, moving it
 *  to the item's segment, 32 segments at a time with every lane of the calling warp. Each task
 *  splits into items of @p rows keys, or of @p rows folded query rows unless @p keys. Writes the
 *  item's @p task and its @p task_item, returning 0 once the window has no more, which items asked
 *  for in increasing order reach only at the end. */
NUMKONG_DEVICE int nk_attention_backward_next_cuda_(nk_attention_backward_arguments_t const *arguments, int keys,
                                                    nk_size_t rows, nk_size_t *segment_first, nk_size_t *items_before,
                                                    nk_size_t item, nk_size_t *task, nk_size_t *task_item) {
    unsigned const lane = threadIdx.x & 31;
    nk_size_t const heads = arguments->key_value_head_count, group = arguments->head_count / heads;
    nk_size_t const segments = ((nk_attention_packed_header_t const *)arguments->packed)->segments;
    nk_u64_t const *payload_offsets = (nk_u64_t const *)(arguments->packed + sizeof(nk_attention_packed_header_t));
    nk_u32_t const *lengths = (nk_u32_t const *)(payload_offsets + segments + 1);
    nk_size_t const tasks_end = nk_attention_backward_task_end_(arguments);
    nk_size_t const segment_end = nk_size_divide_round_up_(tasks_end, heads);
    while (*segment_first < segment_end) {
        nk_size_t const segment = *segment_first + lane, segment_task = segment * heads;
        nk_u64_t const tasks_begin = arguments->tasks_begin > segment_task ? arguments->tasks_begin : segment_task;
        nk_u64_t const task_stop = tasks_end < segment_task + heads ? tasks_end : segment_task + heads;
        nk_u64_t task_items = 0, items = 0;
        if (tasks_begin < task_stop) {
            nk_size_t const count = keys ? lengths[segment]
                                         : group * (arguments->query_offsets[segment + 1] -
                                                    arguments->query_offsets[segment]);
            task_items = nk_size_divide_round_up_(count, rows);
            items = (task_stop - tasks_begin) * task_items;
        }
        nk_u64_t inclusive = items;
#pragma unroll
        for (unsigned offset = 1; offset < 32; offset <<= 1) {
            nk_u64_t const other = nk_shuffle_up_u64_cuda_(inclusive, offset);
            if (lane >= offset) inclusive += other;
        }
        nk_u64_t const total = __shfl_sync(0xFFFFFFFFu, inclusive, 31);
        if (item < *items_before + total) {
            unsigned const found = __ffs(__ballot_sync(0xFFFFFFFFu, *items_before + inclusive > item)) - 1;
            *items_before += __shfl_sync(0xFFFFFFFFu, inclusive - items, found);
            *segment_first += found;
            nk_size_t const local = item - *items_before;
            nk_size_t const found_items = __shfl_sync(0xFFFFFFFFu, task_items, found);
            *task = __shfl_sync(0xFFFFFFFFu, tasks_begin, found) + local / found_items;
            *task_item = local % found_items;
            return 1;
        }
        *items_before += total, *segment_first += 32;
    }
    return 0;
}

/** Accumulates the gradients of one query row, of @p head at @p row of the task, against one key at
 *  @p position: dV += P · dO and dK += dS · Q, or with @p query_gradient non-null, dQ += dS · K,
 *  where P = 2^(score₂ − lse₂) and dS = P · (dO · V − D) · scale, D being the row's @p row_dot. */
NUMKONG_DEVICE void nk_attention_backward_pair_cuda_(nk_dtype_t dtype,
                                                     nk_attention_backward_arguments_t const *arguments,
                                                     nk_attention_backward_task_t const *task, nk_size_t row_bytes,
                                                     nk_size_t head, nk_size_t row, nk_size_t position,
                                                     nk_f32_t row_dot, nk_f32_t *key_gradient, nk_f32_t *value_gradient,
                                                     nk_f32_t *query_gradient, unsigned lane, unsigned lanes) {
    nk_size_t const depth = arguments->depth, element_bytes = nk_attention_element_bytes_(dtype);
    nk_size_t const token = arguments->query_offsets[task->segment] + row;
    unsigned char const *query_row = arguments->queries + token * arguments->query_stride +
                                     head * depth * element_bytes;
    nk_f32_t const *output_gradient_row = (nk_f32_t const *)((unsigned char const *)arguments->output_gradient +
                                                             token * arguments->output_stride) +
                                          head * depth;
    unsigned char const *key_row = task->keys_plane + position * row_bytes;
    nk_f32_t const score = nk_attention_score_cuda_(dtype, query_row, key_row, depth, lane, lanes);
    nk_f32_t const log_sum_exp2 = arguments->log_sum_exp[token * arguments->head_count + head] * NUMKONG_F32_LOG2E_;
    nk_f32_t const weight = exp2f(score * arguments->scale2 - log_sum_exp2);
    unsigned char const *value_row = task->values_plane + position * row_bytes;
    nk_f32_t weight_gradient = 0;
    for (nk_size_t element = lane; element < depth; element += lanes)
        weight_gradient = fmaf(output_gradient_row[element], nk_attention_decode_(dtype, value_row, element),
                               weight_gradient);
    for (unsigned offset = lanes / 2; offset != 0; offset >>= 1)
        weight_gradient += nk_shuffle_xor_f32_cuda_(weight_gradient, offset);
    nk_f32_t const score_gradient = weight * (weight_gradient - row_dot) * arguments->scale;
    if (query_gradient) {
        for (nk_size_t element = lane; element < depth; element += lanes)
            query_gradient[element] = fmaf(score_gradient, nk_attention_decode_(dtype, key_row, element),
                                           query_gradient[element]);
        return;
    }
    for (nk_size_t element = lane; element < depth; element += lanes) {
        value_gradient[element] = fmaf(weight, output_gradient_row[element], value_gradient[element]);
        key_gradient[element] = fmaf(score_gradient, nk_attention_decode_(dtype, query_row, element),
                                     key_gradient[element]);
    }
}

/** The row's D = dO · O, for @p head at @p row of the task. */
NUMKONG_DEVICE nk_f32_t nk_attention_backward_row_dot_cuda_(nk_attention_backward_arguments_t const *arguments,
                                                            nk_attention_backward_task_t const *task, nk_size_t head,
                                                            nk_size_t row, unsigned lane, unsigned lanes) {
    nk_size_t const token = arguments->query_offsets[task->segment] + row, depth = arguments->depth;
    nk_size_t const row_offset = token * arguments->output_stride;
    nk_f32_t const *output_row = (nk_f32_t const *)((unsigned char const *)arguments->output + row_offset) +
                                 head * depth;
    nk_f32_t const *output_gradient_row =
        (nk_f32_t const *)((unsigned char const *)arguments->output_gradient + row_offset) + head * depth;
    nk_f32_t sum = 0;
    for (nk_size_t element = lane; element < depth; element += lanes)
        sum = fmaf(output_gradient_row[element], output_row[element], sum);
    for (unsigned offset = lanes / 2; offset != 0; offset >>= 1) sum += nk_shuffle_xor_f32_cuda_(sum, offset);
    return sum;
}

/** The key and value gradients of every task of the window, a warp per key, summed over the query
 *  heads sharing the key-value head and their rows in order, straight into the gradient rows. */
NUMKONG_DEVICE void nk_attention_backward_keys_cuda_(nk_dtype_t dtype,
                                                     nk_attention_backward_arguments_t const *arguments) {
    unsigned const lanes = nk_warp_lanes_cuda_(), lane = threadIdx.x % lanes, warps = blockDim.x / lanes;
    nk_size_t const depth = arguments->depth, element_bytes = nk_attention_element_bytes_(dtype);
    nk_size_t const row_bytes = nk_size_round_up_to_multiple_(depth * element_bytes, nk_attention_step_bytes_k);
    nk_size_t const group = arguments->head_count / arguments->key_value_head_count;
    nk_size_t const gradient_floats = arguments->key_value_gradient_stride / sizeof(nk_f32_t);
    nk_size_t const tasks_end = nk_attention_backward_task_end_(arguments);
    for (nk_size_t task_index = arguments->tasks_begin + blockIdx.x; task_index < tasks_end; task_index += gridDim.x) {
        nk_attention_backward_task_t const task = nk_attention_backward_task_(arguments, task_index, row_bytes);
        for (nk_size_t position = threadIdx.x / lanes; position < task.length; position += warps) {
            nk_size_t const first = (arguments->key_offsets[task.segment] + position) * gradient_floats +
                                    task.key_value_head * depth;
            nk_f32_t *key_gradient = arguments->key_gradient + first,
                     *value_gradient = arguments->value_gradient + first;
            for (nk_size_t element = lane; element < depth; element += lanes)
                key_gradient[element] = 0, value_gradient[element] = 0;
            for (nk_size_t head = task.key_value_head * group; head < (task.key_value_head + 1) * group; ++head)
                for (nk_size_t row = 0; row < task.rows; ++row) {
                    nk_size_t key_begin, key_end;
                    nk_diagonal_band_row_range_simt_(arguments->band, task.first_band_row + (nk_i64_t)row, task.length,
                                                     &key_begin, &key_end);
                    if (position < key_begin || position >= key_end) continue;
                    nk_f32_t const row_dot = nk_attention_backward_row_dot_cuda_(arguments, &task, head, row, lane,
                                                                                 lanes);
                    nk_attention_backward_pair_cuda_(dtype, arguments, &task, row_bytes, head, row, position, row_dot,
                                                     key_gradient, value_gradient, NUMKONG_NULL, lane, lanes);
                }
        }
    }
}

/** The query gradients of every task of the window, a warp per query row of every head of the
 *  group, summed over the row's keys in order straight into the gradient rows. */
NUMKONG_DEVICE void nk_attention_backward_queries_cuda_(nk_dtype_t dtype,
                                                        nk_attention_backward_arguments_t const *arguments) {
    unsigned const lanes = nk_warp_lanes_cuda_(), lane = threadIdx.x % lanes, warps = blockDim.x / lanes;
    nk_size_t const depth = arguments->depth, element_bytes = nk_attention_element_bytes_(dtype);
    nk_size_t const row_bytes = nk_size_round_up_to_multiple_(depth * element_bytes, nk_attention_step_bytes_k);
    nk_size_t const group = arguments->head_count / arguments->key_value_head_count;
    nk_size_t const tasks_end = nk_attention_backward_task_end_(arguments);
    for (nk_size_t task_index = arguments->tasks_begin + blockIdx.x; task_index < tasks_end; task_index += gridDim.x) {
        nk_attention_backward_task_t const task = nk_attention_backward_task_(arguments, task_index, row_bytes);
        for (nk_size_t local = threadIdx.x / lanes; local < group * task.rows; local += warps) {
            nk_size_t const head = task.key_value_head * group + local / task.rows, row = local % task.rows;
            nk_size_t const token = arguments->query_offsets[task.segment] + row;
            nk_f32_t *query_gradient = (nk_f32_t *)((unsigned char *)arguments->query_gradient +
                                                    token * arguments->query_gradient_stride) +
                                       head * depth;
            for (nk_size_t element = lane; element < depth; element += lanes) query_gradient[element] = 0;
            nk_size_t key_begin, key_end;
            nk_diagonal_band_row_range_simt_(arguments->band, task.first_band_row + (nk_i64_t)row, task.length,
                                             &key_begin, &key_end);
            if (key_begin == key_end) continue;
            nk_f32_t const row_dot = nk_attention_backward_row_dot_cuda_(arguments, &task, head, row, lane, lanes);
            for (nk_size_t position = key_begin; position < key_end; ++position)
                nk_attention_backward_pair_cuda_(dtype, arguments, &task, row_bytes, head, row, position, row_dot,
                                                 NUMKONG_NULL, NUMKONG_NULL, query_gradient, lane, lanes);
        }
    }
}

#pragma endregion Backward

#pragma region Launchers

/** Launches the directory writer, recording the packing @p capability, for a window starting at
 *  task 0, then @p payload_kernel over the window. */
NUMKONG_INLINE nk_status_t nk_attention_pack_launch_cuda_(
    void const *payload_kernel, nk_capability_t capability, nk_size_t element_bytes, void const *keys,
    void const *values, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *segment_offsets,
    nk_u32_t const *segment_lengths, nk_size_t segment_count, nk_size_t key_stride, nk_size_t value_stride,
    void *packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    if ((nk_size_t)packed & 15) return nk_misaligned_k;
    nk_size_t const row_bytes = nk_size_round_up_to_multiple_(depth * element_bytes, nk_attention_step_bytes_k);
    // Zero the gap between the directory and the payload, so the pack depends on its inputs alone
    nk_size_t const directory_bytes = nk_attention_payload_offset_(segment_count, row_bytes) -
                                      sizeof(nk_attention_packed_header_t);
    if (tasks_begin == 0) {
        void *directory_arguments[8] = {&packed,
                                        &key_value_head_count,
                                        &depth,
                                        &segment_lengths,
                                        &segment_count,
                                        (void *)&row_bytes,
                                        (void *)&directory_bytes,
                                        (void *)&capability};
        nk_status_t const status = nk_launch_cuda_((void const *)nk_attention_pack_directory_cuda_kernel_, 1,
                                                   nk_attention_threads_k, directory_arguments, 0, stream);
        if (status != nk_success_k) return status;
    }
    nk_size_t const total_tasks = segment_count * key_value_head_count;
    nk_size_t const end = tasks_end < total_tasks ? tasks_end : total_tasks;
    if (tasks_begin >= end) return nk_success_k;
    void *payload_arguments[12] = {(void *)&keys,    (void *)&values,  &key_value_head_count, &depth,
                                   &segment_offsets, &segment_lengths, &segment_count,        &key_stride,
                                   &value_stride,    &packed,          &tasks_begin,          (void *)&end};
    return nk_launch_cuda_(payload_kernel, end - tasks_begin < 65535 ? end - tasks_begin : 65535,
                           nk_attention_pack_threads_k, payload_arguments, 0, stream);
}

/** Validates the contract and launches @p kernel with as many blocks as stay resident. */
NUMKONG_INLINE nk_status_t nk_attention_launch_cuda_(void const *kernel, void const *queries, void const *packed,
                                                     nk_f32_t *output, nk_f32_t *log_sum_exp, nk_size_t head_count,
                                                     nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *query_offsets, nk_size_t query_stride,
                                                     nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before,
                                                     nk_size_t keys_after, nk_size_t tasks_begin, nk_size_t tasks_end,
                                                     nk_stream_t stream) {
    if (((nk_size_t)packed & 15) || (((nk_size_t)output | output_stride) & 3)) return nk_misaligned_k;
    if (key_value_head_count == 0 || head_count % key_value_head_count != 0) return nk_unexpected_dimensions_k;
    if (tasks_begin >= tasks_end || depth == 0) return nk_success_k;
    nk_attention_arguments_t arguments = nk_attention_arguments_init_(
        queries, packed, output, log_sum_exp, head_count, key_value_head_count, depth, query_offsets, query_stride,
        output_stride, scale, 1, 1, keys_before, keys_after, tasks_begin, tasks_end);
    return nk_launch_resident_cuda_(kernel, nk_attention_threads_k, 0, 0, NUMKONG_SIZE_MAX, &arguments, stream);
}

/**
 *  @brief Validates the contract and launches the kernels of a backward pass.
 *
 *  Runs @p prepare_kernel unless it is null, then @p keys_kernel for the key-value gradients and
 *  @p queries_kernel for the query ones, each launched with as many blocks as stay resident, though
 *  never more than @p blocks_wanted.
 *
 *  @sa nk_launch_resident_cuda_ for @p shared_bytes and @p shared_ceiling.
 */
NUMKONG_INLINE nk_status_t nk_attention_backward_launch_kernels_cuda_(
    void const *prepare_kernel, void const *keys_kernel, void const *queries_kernel, unsigned threads,
    nk_size_t shared_bytes, nk_size_t shared_ceiling, nk_size_t blocks_wanted,
    nk_attention_backward_arguments_t arguments, nk_stream_t stream) {
    if (((nk_size_t)arguments.packed & 15) ||
        (((nk_size_t)arguments.output | (nk_size_t)arguments.output_gradient | (nk_size_t)arguments.query_gradient |
          (nk_size_t)arguments.key_gradient | (nk_size_t)arguments.value_gradient | arguments.output_stride |
          arguments.query_gradient_stride | arguments.key_value_gradient_stride) &
         3))
        return nk_misaligned_k;
    if (arguments.key_value_head_count == 0 || arguments.head_count % arguments.key_value_head_count != 0)
        return nk_unexpected_dimensions_k;
    if (arguments.tasks_begin >= arguments.tasks_end || arguments.depth == 0) return nk_success_k;
    nk_status_t status = prepare_kernel ? nk_launch_resident_cuda_(prepare_kernel, nk_attention_threads_k, 0, 0,
                                                                   NUMKONG_SIZE_MAX, &arguments, stream)
                                        : nk_success_k;
    if (status == nk_success_k)
        status = nk_launch_resident_cuda_(keys_kernel, threads, shared_bytes, shared_ceiling, blocks_wanted, &arguments,
                                          stream);
    if (status == nk_success_k)
        status = nk_launch_resident_cuda_(queries_kernel, threads, shared_bytes, shared_ceiling, blocks_wanted,
                                          &arguments, stream);
    return status;
}

/** Launches the backward kernels, keys and values first, at most a block per task. */
NUMKONG_INLINE nk_status_t nk_attention_backward_launch_cuda_(void const *keys_kernel, void const *queries_kernel,
                                                              nk_attention_backward_arguments_t arguments,
                                                              nk_stream_t stream) {
    return nk_attention_backward_launch_kernels_cuda_(NUMKONG_NULL, keys_kernel, queries_kernel, nk_attention_threads_k,
                                                      0, 0, arguments.tasks_end - arguments.tasks_begin, arguments,
                                                      stream);
}

/** Validates the contract and launches @p kernel with a warp per head of every row, at most 2²⁰
 *  blocks of them. */
NUMKONG_INLINE nk_status_t nk_attention_rope_launch_cuda_(void const *kernel, nk_size_t value_bytes, void const *x,
                                                          nk_f32_t const *cos, nk_f32_t const *sin, void *y,
                                                          nk_size_t rows, nk_size_t head_count, nk_size_t depth,
                                                          nk_size_t x_stride, nk_size_t y_stride, nk_stream_t stream) {
    if (depth % 2) return nk_unexpected_dimensions_k;
    if ((((nk_size_t)x) | x_stride | ((nk_size_t)y) | y_stride) & (value_bytes - 1) ||
        (((nk_size_t)cos) | ((nk_size_t)sin)) & 3)
        return nk_misaligned_k;
    nk_attention_rope_arguments_t arguments;
    arguments.x = (unsigned char const *)x, arguments.cos = cos, arguments.sin = sin, arguments.y = (unsigned char *)y;
    arguments.head_count = head_count, arguments.half_depth = depth / 2, arguments.heads = rows * head_count;
    arguments.x_stride = x_stride, arguments.y_stride = y_stride;
    if (arguments.heads == 0 || depth == 0) return nk_success_k;
    nk_size_t const blocks = nk_size_divide_round_up_(arguments.heads, nk_attention_threads_k / 32),
                    blocks_limit = (nk_size_t)1 << 20;
    void *launch_arguments[1];
    launch_arguments[0] = &arguments;
    return nk_launch_cuda_(kernel, blocks < blocks_limit ? blocks : blocks_limit, nk_attention_threads_k,
                           launch_arguments, 0, stream);
}

#pragma endregion Launchers

#pragma region Attention Macros

/** Generates a shape accessor that copies a device pack's header back and checks its capability. */
#define nk_define_attention_packed_shape_cuda_(input_type_name, isa_suffix)                                          \
    NUMKONG_API nk_status_t nk_attention_packed_shape_##input_type_name##_##isa_suffix(                              \
        void const *key_value_packed, nk_size_t *heads, nk_size_t *depth, nk_size_t *segments, nk_stream_t stream) { \
        nk_attention_packed_header_t header;                                                                         \
        nk_status_t const status = nk_read_cuda_(&header, key_value_packed, sizeof(header), stream);                 \
        if (status != nk_success_k) return status;                                                                   \
        if (header.capability != nk_cap_##isa_suffix##_k) return nk_pack_mismatch_k;                                 \
        *heads = header.heads, *depth = header.depth, *segments = header.segments;                                   \
        return nk_success_k;                                                                                         \
    }

/**
 *  @brief Generates a device pack: the kernel, one block per task, and the entry point, which
 *      writes the directory, recording the packing @p isa_suffix, when the window starts at task 0.
 *
 *  V keeps position rows for BF16 and takes σ-ordered depth rows for 1-byte dtypes.
 */
#define nk_define_attention_pack_cuda_(input_type_name, isa_suffix, input_value_type)                                 \
    static __global__ void nk_attention_pack_##input_type_name##_##isa_suffix##_kernel_(                              \
        unsigned char const *keys, unsigned char const *values, nk_size_t key_value_head_count, nk_size_t depth,      \
        nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, nk_size_t segment_count,                    \
        nk_size_t key_stride, nk_size_t value_stride, unsigned char *packed, nk_size_t tasks_begin,                   \
        nk_size_t tasks_end) {                                                                                        \
        nk_attention_pack_payload_(nk_##input_value_type##_k, keys, values, key_value_head_count, depth,              \
                                   segment_offsets, segment_lengths, segment_count, key_stride, value_stride, packed, \
                                   tasks_begin, tasks_end);                                                           \
    }                                                                                                                 \
    NUMKONG_API nk_status_t nk_attention_pack_##input_type_name##_##isa_suffix(                                       \
        nk_##input_value_type##_t const *keys, nk_##input_value_type##_t const *values,                               \
        nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *segment_offsets,                             \
        nk_u32_t const *segment_lengths, nk_size_t segment_count, nk_size_t key_stride, nk_size_t value_stride,       \
        void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {                     \
        return nk_attention_pack_launch_cuda_(                                                                        \
            (void const *)nk_attention_pack_##input_type_name##_##isa_suffix##_kernel_, nk_cap_##isa_suffix##_k,      \
            sizeof(nk_##input_value_type##_t), keys, values, key_value_head_count, depth, segment_offsets,            \
            segment_lengths, segment_count, key_stride, value_stride, key_value_packed, tasks_begin, tasks_end,       \
            stream);                                                                                                  \
    }

/**
 *  @brief Generates every attention entry of one dtype on the CUDA baseline @p isa_suffix, with its
 *      own pack, and the public attention entry point over the fallback kernel, which the vendor
 *      baselines run for every query.
 *
 *  The pack must come from the same capability's pack kernel: the host can't read its device header
 *  without waiting on the stream, so the entry point trusts it.
 */
#define nk_define_attention_baseline_cuda_(input_type_name, isa_suffix, input_value_type, element_bytes)               \
    nk_define_attention_pack_size_simt_(input_type_name, isa_suffix, element_bytes)                                    \
    nk_define_attention_packed_shape_cuda_(input_type_name, isa_suffix)                                                \
    nk_define_attention_pack_cuda_(input_type_name, isa_suffix, input_value_type)                                      \
    static __global__ void __launch_bounds__(nk_attention_threads_k)                                                   \
        nk_attention_packed_##input_type_name##_##isa_suffix##_kernel_(nk_attention_arguments_t arguments) {           \
        nk_attention_fallback_cuda_(nk_##input_value_type##_k, &arguments);                                            \
    }                                                                                                                  \
    NUMKONG_API nk_status_t nk_attention_packed_##input_type_name##_##isa_suffix(                                      \
        nk_##input_value_type##_t const *queries, void const *key_value_packed, nk_f32_t *output,                      \
        nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,                  \
        nk_u32_t const *query_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale,                \
        nk_size_t keys_before, nk_size_t keys_after, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) { \
        return nk_attention_launch_cuda_((void const *)nk_attention_packed_##input_type_name##_##isa_suffix##_kernel_, \
                                         queries, key_value_packed, output, log_sum_exp, head_count,                   \
                                         key_value_head_count, depth, query_offsets, query_stride, output_stride,      \
                                         scale, keys_before, keys_after, tasks_begin, tasks_end, stream);              \
    }

/**
 *  @brief Generates the narrow and wide kernels of one dtype, unmasked and masked by the band, and
 *      the fallback kernel, then the public attention entry point over them, picking the masked
 *      pair unless the band reaches every key, and passing them to the capability's launch.
 *
 *  The pack must come from the same capability's pack kernel, which the entry point trusts, as
 *  @c nk_define_attention_baseline_cuda_ explains.
 *
 *  @param[in] tile The tiling kernel family, like @c ampere, whose tile function the narrow and
 *      wide kernels run.
 *  @param[in] launch_fn The launch, taking the narrow, wide and fallback kernels in that order.
 *  @param[in] score_scale Undoes the power of two that converting Q and K puts on scores, or 1.
 *  @param[in] output_scale Undoes the power of two that converting V puts on the output, or 1.
 */
#define nk_define_attention_packed_cuda_(input_type_name, isa_suffix, tile, launch_fn, input_value_type, epilogue,     \
                                         scores_fn, values_mma_fn, weights_fn, score_scale, output_scale)              \
    static __global__ void __launch_bounds__(nk_attention_threads_k)                                                   \
        nk_attention_packed_##input_type_name##_narrow_##isa_suffix##_kernel_(nk_attention_arguments_t arguments) {    \
        nk_attention_tile_##tile##_(nk_##input_value_type##_k, nk_attention_width_128_k, nk_attention_mask_none_k,     \
                                    epilogue, scores_fn, values_mma_fn, weights_fn, &arguments);                       \
    }                                                                                                                  \
    static __global__ void __launch_bounds__(nk_attention_threads_k)                                                   \
        nk_attention_packed_##input_type_name##_wide_##isa_suffix##_kernel_(nk_attention_arguments_t arguments) {      \
        nk_attention_tile_##tile##_(nk_##input_value_type##_k, nk_attention_width_256_k, nk_attention_mask_none_k,     \
                                    epilogue, scores_fn, values_mma_fn, weights_fn, &arguments);                       \
    }                                                                                                                  \
    static __global__ void __launch_bounds__(nk_attention_threads_k)                                                   \
        nk_attention_packed_##input_type_name##_narrow_masked_##isa_suffix##_kernel_(                                  \
            nk_attention_arguments_t arguments) {                                                                      \
        nk_attention_tile_##tile##_(nk_##input_value_type##_k, nk_attention_width_128_k,                               \
                                    nk_attention_mask_diagonal_band_k, epilogue, scores_fn, values_mma_fn, weights_fn, \
                                    &arguments);                                                                       \
    }                                                                                                                  \
    static __global__ void __launch_bounds__(nk_attention_threads_k)                                                   \
        nk_attention_packed_##input_type_name##_wide_masked_##isa_suffix##_kernel_(                                    \
            nk_attention_arguments_t arguments) {                                                                      \
        nk_attention_tile_##tile##_(nk_##input_value_type##_k, nk_attention_width_256_k,                               \
                                    nk_attention_mask_diagonal_band_k, epilogue, scores_fn, values_mma_fn, weights_fn, \
                                    &arguments);                                                                       \
    }                                                                                                                  \
    static __global__ void __launch_bounds__(nk_attention_threads_k)                                                   \
        nk_attention_packed_##input_type_name##_fallback_##isa_suffix##_kernel_(nk_attention_arguments_t arguments) {  \
        nk_attention_fallback_cuda_(nk_##input_value_type##_k, &arguments);                                            \
    }                                                                                                                  \
    NUMKONG_API nk_status_t nk_attention_packed_##input_type_name##_##isa_suffix(                                      \
        nk_##input_value_type##_t const *queries, void const *key_value_packed, nk_f32_t *output,                      \
        nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,                  \
        nk_u32_t const *query_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale,                \
        nk_size_t keys_before, nk_size_t keys_after, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) { \
        int const masked = keys_before != NUMKONG_SIZE_MAX || keys_after != NUMKONG_SIZE_MAX;                          \
        return launch_fn(                                                                                              \
            masked ? (void const *)nk_attention_packed_##input_type_name##_narrow_masked_##isa_suffix##_kernel_        \
                   : (void const *)nk_attention_packed_##input_type_name##_narrow_##isa_suffix##_kernel_,              \
            masked ? (void const *)nk_attention_packed_##input_type_name##_wide_masked_##isa_suffix##_kernel_          \
                   : (void const *)nk_attention_packed_##input_type_name##_wide_##isa_suffix##_kernel_,                \
            (void const *)nk_attention_packed_##input_type_name##_fallback_##isa_suffix##_kernel_,                     \
            nk_##input_value_type##_k, queries, key_value_packed, output, log_sum_exp, head_count,                     \
            key_value_head_count, depth, query_offsets, query_stride, output_stride, scale, score_scale, output_scale, \
            keys_before, keys_after, tasks_begin, tasks_end, stream);                                                  \
    }

/** Generates the gradients entry point of one dtype over the capability's
 *  @c nk_attention_backward_keys_<type>_<capability>_kernel_ and its queries twin, which @p launch
 *  starts, as @c nk_attention_backward_launch_cuda_ does. */
#define nk_define_attention_backward_cuda_(input_type_name, isa_suffix, input_value_type, launch)                      \
    NUMKONG_API nk_status_t nk_attention_packed_gradients_##input_type_name##_##isa_suffix(                            \
        nk_##input_value_type##_t const *queries, void const *key_value_packed, nk_f32_t const *output,                \
        nk_f32_t const *output_gradient, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient,                        \
        nk_f32_t *key_gradient, nk_f32_t *value_gradient, nk_size_t head_count, nk_size_t key_value_head_count,        \
        nk_size_t depth, nk_u32_t const *query_offsets, nk_u32_t const *key_offsets, nk_size_t query_stride,           \
        nk_size_t output_stride, nk_size_t query_gradient_stride, nk_size_t key_value_gradient_stride, nk_f32_t scale, \
        nk_size_t keys_before, nk_size_t keys_after, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) { \
        return launch((void const *)nk_attention_backward_keys_##input_type_name##_##isa_suffix##_kernel_,             \
                      (void const *)nk_attention_backward_queries_##input_type_name##_##isa_suffix##_kernel_,          \
                      nk_attention_backward_arguments_init_(                                                           \
                          queries, key_value_packed, output, output_gradient, log_sum_exp, query_gradient,             \
                          key_gradient, value_gradient, head_count, key_value_head_count, depth, query_offsets,        \
                          key_offsets, query_stride, output_stride, query_gradient_stride, key_value_gradient_stride,  \
                          scale, keys_before, keys_after, tasks_begin, tasks_end),                                     \
                      stream);                                                                                         \
    }

/** Generates the RoPE kernel of @p input_type for @p isa_suffix, after @c nk_define_attention_rope_
 *  of the serial backend, and its entry point: each warp rotates one head of one row, so the row
 *  and head divisions happen once per head rather than once per pair. */
#define nk_define_attention_rope_cuda_(input_type, isa_suffix, load_and_convert, convert_and_store)                   \
    static __global__ void nk_attention_rope_##input_type##_##isa_suffix##_kernel_(                                   \
        nk_attention_rope_arguments_t arguments) {                                                                    \
        unsigned const lanes = nk_warp_lanes_cuda_(), lane = threadIdx.x % lanes;                                     \
        nk_size_t const groups = blockDim.x / lanes;                                                                  \
        for (nk_size_t head = (nk_size_t)blockIdx.x * groups + threadIdx.x / lanes; head < arguments.heads;           \
             head += (nk_size_t)gridDim.x * groups) {                                                                 \
            nk_size_t const row = head / arguments.head_count;                                                        \
            nk_size_t const first = (head - row * arguments.head_count) * 2 * arguments.half_depth;                   \
            nk_f32_t const *cos_row = arguments.cos + row * arguments.half_depth;                                     \
            nk_f32_t const *sin_row = arguments.sin + row * arguments.half_depth;                                     \
            nk_##input_type##_t const *x = (nk_##input_type##_t const *)(arguments.x + row * arguments.x_stride) +    \
                                           first;                                                                     \
            nk_##input_type##_t *y = (nk_##input_type##_t *)(arguments.y + row * arguments.y_stride) + first;         \
            for (nk_size_t pair = lane; pair < arguments.half_depth; pair += lanes) {                                 \
                nk_f32_t low, high;                                                                                   \
                load_and_convert(x + pair, &low);                                                                     \
                load_and_convert(x + pair + arguments.half_depth, &high);                                             \
                nk_f32_t const cosine = cos_row[pair], sine = sin_row[pair];                                          \
                nk_f32_t const rotated_low = low * cosine - high * sine;                                              \
                nk_f32_t const rotated_high = low * sine + high * cosine;                                             \
                convert_and_store(&rotated_low, y + pair);                                                            \
                convert_and_store(&rotated_high, y + pair + arguments.half_depth);                                    \
            }                                                                                                         \
        }                                                                                                             \
    }                                                                                                                 \
    NUMKONG_API nk_status_t nk_attention_rope_##input_type##_##isa_suffix(                                            \
        nk_##input_type##_t const *x, nk_f32_t const *cos, nk_f32_t const *sin, nk_##input_type##_t *y,               \
        nk_size_t rows, nk_size_t head_count, nk_size_t depth, nk_size_t x_stride, nk_size_t y_stride,                \
        nk_stream_t stream) {                                                                                         \
        return nk_attention_rope_launch_cuda_((void const *)&nk_attention_rope_##input_type##_##isa_suffix##_kernel_, \
                                              sizeof(nk_##input_type##_t), x, cos, sin, y, rows, head_count, depth,   \
                                              x_stride, y_stride, stream);                                            \
    }

#pragma endregion Attention Macros

#pragma region Instantiations

#if NUMKONG_TARGET_CUDA
nk_define_attention_baseline_cuda_(bf16, cuda, bf16, 2)
nk_define_attention_baseline_cuda_(f16, cuda, f16, 2)
nk_define_attention_baseline_cuda_(e4m3, cuda, e4m3, 1)
nk_define_attention_baseline_cuda_(i8, cuda, i8, 1)

/** The two backward kernels of BF16 on the CUDA baseline, keys and values first. */
static __global__ void __launch_bounds__(nk_attention_threads_k)
    nk_attention_backward_keys_bf16_cuda_kernel_(nk_attention_backward_arguments_t arguments) {
    nk_attention_backward_keys_cuda_(nk_bf16_k, &arguments);
}
static __global__ void __launch_bounds__(nk_attention_threads_k)
    nk_attention_backward_queries_bf16_cuda_kernel_(nk_attention_backward_arguments_t arguments) {
    nk_attention_backward_queries_cuda_(nk_bf16_k, &arguments);
}
nk_define_attention_backward_cuda_(bf16, cuda, bf16, nk_attention_backward_launch_cuda_)
nk_define_attention_rope_cuda_(f32, cuda, nk_assign_from_to_, nk_assign_from_to_)
nk_define_attention_rope_cuda_(bf16, cuda, nk_bf16_to_f32_simt_, nk_f32_to_bf16_simt_)
nk_define_attention_rope_cuda_(e4m3, cuda, nk_e4m3_to_f32_simt_, nk_f32_to_e4m3_simt_)
#endif // NUMKONG_TARGET_CUDA

#pragma endregion Instantiations

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_
#endif // NUMKONG_ATTENTION_CUDA_CUH
