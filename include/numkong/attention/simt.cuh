/**
 *  @file include/numkong/attention/simt.cuh
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Ragged attention and rotary embeddings on the SIMT cores of every CUDA and ROCm device.
 *
 *  @sa include/numkong/attention.h
 *  @sa include/numkong/attention/serial.h
 *
 *  The pack, the work scheduler and the one kernel they feed, which the tensor-core kernels share:
 *  a warp or wavefront per query row, two sweeps over its keys as the serial kernel makes, products
 *  in F32 and I8 scores in exact I32 like the serial kernel's. The pack keeps the serial header and
 *  directory, with 1-byte V transposed and permuted within every 16 positions so the tensor-core
 *  kernels form MMA fragments without shuffles. Only @c pack_size reads @c segment_lengths on the
 *  host, so the pack and both attention kernels need device or managed memory for the segment
 *  offsets and lengths.
 */
#ifndef NUMKONG_ATTENTION_SIMT_CUH
#define NUMKONG_ATTENTION_SIMT_CUH

#if NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_

#include "numkong/attention/serial.h" // `nk_attention_packed_header_t`, `nk_attention_pack_directory_size_`
#include "numkong/dots/simt.cuh"      // `nk_device_launch_`, `nk_device_read_`, `nk_shuffle_xor_f32_`

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Configuration

enum {
    nk_attention_threads_k = 128,
    nk_attention_block_rows_k = 64,
    nk_attention_panel_k = 64,
    nk_attention_step_bytes_k = 32,
    nk_attention_pack_threads_k = 256,
};

/** Which keys each query row sees: its whole segment, or, when causal, the @c window keys ending at
 *  the row's position `r + diagonal_offset`. */
typedef enum {
    nk_attention_mask_bidirectional_k,
    nk_attention_mask_causal_k,
} nk_attention_mask_t;

/** Which tensor-core tile a head's depth takes: up to 128 or up to 256. Deeper heads
 *  run the SIMT fallback kernel. */
typedef enum {
    nk_attention_width_128_k,
    nk_attention_width_256_k,
} nk_attention_width_t;

/** Everything one attention launch shares, passed by value as the kernels' only argument. */
typedef struct {

    /** Query rows, [query tokens, heads × depth]. */
    unsigned char const *queries;

    /** The packed buffer, header first. */
    unsigned char const *packed;

    /** F32 output rows, [query tokens, heads × depth]. */
    nk_f32_t *output;

    /** First query row of each segment, with segments + 1 entries. */
    nk_u32_t const *query_offsets;

    /** Query heads. */
    nk_size_t head_count;

    /** K and V heads. */
    nk_size_t key_value_head_count;

    /** Elements per head. */
    nk_size_t depth;

    /** Bytes between query rows. */
    nk_size_t query_stride;

    /** Bytes between output rows. */
    nk_size_t output_stride;

    /** Position of query row 0 in its segment. */
    nk_i64_t diagonal_offset;

    /** Visible keys, including the query's own. */
    nk_size_t window;

    /** First task of the window over segments × heads. */
    nk_size_t task_start;

    /** Tasks in the window, clipped on the device. */
    nk_size_t task_count;

    /** Score multiplier in base 2, scale · log₂e. */
    nk_f32_t scale2;

    /** Undoes the power of two that converting Q and K puts on scores, or 1. */
    nk_f32_t score_scale;

    /** Undoes the power of two that converting V puts on the output, or 1. */
    nk_f32_t output_scale;

    /** K panel in shared memory, per @c nk_attention_split_t. */
    nk_u32_t key_offset[2];

    /** V panel in shared memory, per @c nk_attention_split_t. */
    nk_u32_t value_offset[2];

    /** Staged Q rows in shared memory, per @c nk_attention_split_t. */
    nk_u32_t query_offset[2];
} nk_attention_arguments_t;

/** Walks the work items of a task window, one chunk of segments' item counts at a time. */
typedef struct {

    /** First query row of each segment. */
    nk_u32_t const *query_offsets;

    /** Query heads per segment. */
    nk_size_t head_count;

    /** Query heads per K and V head. */
    nk_size_t group_heads;

    /** First task of the window. */
    nk_size_t task_start;

    /** One past the last task, clipped to the grid. */
    nk_size_t task_end;

    /** One past the last segment the window touches. */
    nk_size_t segment_end;

    /** First segment of the chunk whose prefix sits in shared memory. */
    nk_size_t chunk_first;

    /** Items in the segments before the chunk. */
    nk_size_t items_before;
} nk_attention_schedule_t;

/** One work item: up to 64 rows, each row being query × heads_selected + head, of one segment
 *  against one K and V head. */
typedef struct {

    /** Segment index. */
    nk_size_t segment;

    /** K and V head. */
    nk_size_t key_value_head;

    /** First query head, counted within the segment. */
    nk_size_t head_first;

    /** Query heads the rows cycle through. */
    nk_size_t heads_selected;

    /** First row of the item. */
    nk_size_t row_first;

    /** Rows of the item. */
    nk_size_t row_count;

    /** Query token of the segment's first query. */
    nk_size_t query_first;
} nk_attention_work_t;

#pragma endregion Configuration

#pragma region Fragments

NUMKONG_DEVICE nk_f32_t nk_attention_negative_infinity_(void) { return __int_as_float(0xFF800000); }

/** 2^x to 2⁻²² relative on NVIDIA, flushing results below 2⁻¹²⁶ to zero, and 0 for −∞. */
NUMKONG_DEVICE nk_f32_t nk_f32_exp2_(nk_f32_t exponent) {
#if NUMKONG_ARCH_ROCM_
    return exp2f(exponent);
#else
    nk_f32_t power;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(power) : "f"(exponent));
    return power;
#endif
}

/** Slot of @p position in a transposed V: within every 16, position `2t + 8h + e` sits at
 *  `4t + 2h + e`. */
NUMKONG_DEVICE nk_size_t nk_attention_slot_(nk_size_t position) {
    return (position & ~(nk_size_t)15) | ((position & 6) << 1) | ((position & 8) >> 2) | (position & 1);
}

/** Position held by @p slot of a transposed V, inverting @c nk_attention_slot_. */
NUMKONG_DEVICE nk_size_t nk_attention_slot_position_(nk_size_t slot) {
    return (slot & ~(nk_size_t)15) | ((slot & 12) >> 1) | ((slot & 2) << 2) | (slot & 1);
}

/** Writes the half-open range of keys visible to the query at @p position into @p key_begin and
 *  @p key_end, with the serial backend's rule. */
NUMKONG_DEVICE void nk_attention_row_keys_(nk_i64_t position, nk_size_t window, nk_size_t length, unsigned *key_begin,
                                           unsigned *key_end) {
    if (position < 0 || window == 0) {
        *key_begin = *key_end = 0;
        return;
    }
    nk_size_t const query_position = (nk_size_t)position;
    nk_size_t const end = query_position < length ? query_position + 1 : length;
    nk_size_t const begin = window > query_position ? 0 : query_position - window + 1;
    *key_end = (unsigned)end, *key_begin = (unsigned)(begin < end ? begin : end);
}

#pragma endregion Fragments

#pragma region Schedule

/** Inclusive block-wide prefix sum of one value per thread; the block total lands in @p total. */
NUMKONG_DEVICE nk_u64_t nk_attention_block_scan_(nk_u64_t value, nk_u64_t *warp_totals, nk_u64_t *total) {
    unsigned const lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
#pragma unroll
    for (unsigned offset = 1; offset < 32; offset <<= 1) {
        nk_u64_t const other = nk_shuffle_up_u64_(value, offset);
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

NUMKONG_DEVICE nk_size_t nk_attention_row_blocks_(nk_size_t queries, nk_size_t heads) {
    return nk_size_divide_round_up_(queries * heads, nk_attention_block_rows_k);
}

/** Writes the half-open range of heads of @p segment inside the task window into @p head_begin and
 *  @p head_end, counted within the segment. */
NUMKONG_DEVICE void nk_attention_segment_heads_(nk_attention_schedule_t const *schedule, nk_size_t segment,
                                                nk_size_t *head_begin, nk_size_t *head_end) {
    nk_size_t const task_first = segment * schedule->head_count;
    *head_begin = schedule->task_start > task_first ? schedule->task_start - task_first : 0;
    *head_end = schedule->task_end < task_first + schedule->head_count ? schedule->task_end - task_first
                                                                       : schedule->head_count;
}

NUMKONG_DEVICE nk_size_t nk_attention_segment_items_(nk_attention_schedule_t const *schedule, nk_size_t segment) {
    if (segment >= schedule->segment_end) return 0;
    nk_size_t const queries = schedule->query_offsets[segment + 1] - schedule->query_offsets[segment];
    nk_size_t head_begin, head_end;
    nk_attention_segment_heads_(schedule, segment, &head_begin, &head_end);
    if (queries == 0 || head_begin >= head_end) return 0;
    nk_size_t const group = schedule->group_heads;
    nk_size_t const first = head_begin / group, last = (head_end - 1) / group;
    if (first == last) return nk_attention_row_blocks_(queries, head_end - head_begin);
    return nk_attention_row_blocks_(queries, (first + 1) * group - head_begin) +
           (last - first - 1) * nk_attention_row_blocks_(queries, group) +
           nk_attention_row_blocks_(queries, head_end - last * group);
}

/** Fills @p prefix with the running item counts of the 128 segments from
 *  `schedule->chunk_first`. */
NUMKONG_DEVICE void nk_attention_schedule_chunk_(nk_attention_schedule_t const *schedule, nk_u64_t *prefix,
                                                 nk_u64_t *warp_totals) {
    nk_u64_t const items = nk_attention_segment_items_(schedule, schedule->chunk_first + threadIdx.x);
    nk_u64_t total;
    nk_u64_t const inclusive = nk_attention_block_scan_(items, warp_totals, &total);
    prefix[threadIdx.x + 1] = inclusive;
    if (threadIdx.x == 0) prefix[0] = 0;
    __syncthreads();
}

/** Finds work item @p item, returning 0 once the window has no more; every thread of the block
 *  calls it in step. */
NUMKONG_DEVICE int nk_attention_schedule_next_(nk_attention_schedule_t *schedule, nk_u64_t *prefix,
                                               nk_u64_t *warp_totals, nk_size_t item, nk_attention_work_t *work) {
    while (item >= schedule->items_before + prefix[nk_attention_threads_k]) {
        if (schedule->chunk_first + nk_attention_threads_k >= schedule->segment_end) return 0;
        schedule->items_before += prefix[nk_attention_threads_k];
        schedule->chunk_first += nk_attention_threads_k;
        __syncthreads();
        nk_attention_schedule_chunk_(schedule, prefix, warp_totals);
    }
    nk_u64_t const local = item - schedule->items_before;
    unsigned low = 0, high = nk_attention_threads_k;
    while (high - low > 1) {
        unsigned const middle = (low + high) >> 1;
        if (prefix[middle] <= local) low = middle;
        else high = middle;
    }
    nk_size_t const segment = schedule->chunk_first + low;
    nk_size_t index = (nk_size_t)(local - prefix[low]);
    nk_size_t const queries = schedule->query_offsets[segment + 1] - schedule->query_offsets[segment];
    nk_size_t head_begin, head_end;
    nk_attention_segment_heads_(schedule, segment, &head_begin, &head_end);
    nk_size_t const group = schedule->group_heads;
    nk_size_t const first = head_begin / group, last = (head_end - 1) / group;
    nk_size_t key_value_head = first, row_block = index;
    if (first != last) {
        nk_size_t const first_items = nk_attention_row_blocks_(queries, (first + 1) * group - head_begin);
        if (index >= first_items) {
            index -= first_items;
            nk_size_t const full_items = nk_attention_row_blocks_(queries, group);
            nk_size_t const middle_items = (last - first - 1) * full_items;
            if (index < middle_items) key_value_head = first + 1 + index / full_items, row_block = index % full_items;
            else key_value_head = last, row_block = index - middle_items;
        }
    }
    work->segment = segment;
    work->key_value_head = key_value_head;
    work->head_first = key_value_head * group > head_begin ? key_value_head * group : head_begin;
    work->heads_selected = ((key_value_head + 1) * group < head_end ? (key_value_head + 1) * group : head_end) -
                           work->head_first;
    work->row_first = row_block * nk_attention_block_rows_k;
    nk_size_t const rows = queries * work->heads_selected - work->row_first;
    work->row_count = rows < nk_attention_block_rows_k ? rows : nk_attention_block_rows_k;
    work->query_first = schedule->query_offsets[segment];
    return 1;
}

/** Reads the header, returning 0 when it disagrees with the arguments, and primes the
 *  schedule's first chunk. */
NUMKONG_DEVICE int nk_attention_schedule_start_(nk_attention_arguments_t const *arguments,
                                                nk_attention_schedule_t *schedule, nk_u64_t *prefix,
                                                nk_u64_t *warp_totals) {
    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)arguments->packed;
    if (header->depth != arguments->depth || header->heads != arguments->key_value_head_count) return 0;
    nk_size_t const total_tasks = (nk_size_t)header->segments * arguments->head_count;
    if (arguments->task_start >= total_tasks) return 0;
    schedule->query_offsets = arguments->query_offsets;
    schedule->head_count = arguments->head_count;
    schedule->group_heads = arguments->head_count / arguments->key_value_head_count;
    schedule->task_start = arguments->task_start;
    schedule->task_end = arguments->task_count < total_tasks - arguments->task_start
                             ? arguments->task_start + arguments->task_count
                             : total_tasks;
    schedule->segment_end = nk_size_divide_round_up_(schedule->task_end, arguments->head_count);
    schedule->chunk_first = arguments->task_start / arguments->head_count;
    schedule->items_before = 0;
    nk_attention_schedule_chunk_(schedule, prefix, warp_totals);
    return 1;
}

/** K plane of the work item's head; its V plane follows @c key_value_head_count planes later. */
NUMKONG_DEVICE unsigned char const *nk_attention_keys_plane_(nk_attention_arguments_t const *arguments,
                                                             nk_attention_work_t const *work, nk_size_t row_bytes,
                                                             nk_size_t *length, nk_size_t *plane_bytes) {
    nk_size_t const segments = ((nk_attention_packed_header_t const *)arguments->packed)->segments;
    nk_u64_t const *payload_offsets = (nk_u64_t const *)(arguments->packed + sizeof(nk_attention_packed_header_t));
    nk_u32_t const *lengths = (nk_u32_t const *)(payload_offsets + segments + 1);
    nk_size_t const directory_bytes = nk_size_round_up_to_multiple_(
        (segments + 1) * sizeof(nk_u64_t) + segments * sizeof(nk_u32_t), 64);
    *length = lengths[work->segment];
    *plane_bytes = nk_size_round_up_to_multiple_(*length, nk_attention_panel_k) * row_bytes;
    return arguments->packed + sizeof(nk_attention_packed_header_t) + directory_bytes + payload_offsets[work->segment] +
           work->key_value_head * *plane_bytes;
}

#pragma endregion Schedule

#pragma region Tile

/** One element of a K row or V element, decoded to F32; I8 stays an exact integer in F32. */
NUMKONG_DEVICE nk_f32_t nk_attention_decode_(nk_dtype_t dtype, unsigned char const *bytes, nk_size_t index) {
    if (dtype == nk_bf16_k) return __uint_as_float((nk_u32_t)((unsigned short const *)bytes)[index] << 16);
    if (dtype == nk_i8_k) return (nk_f32_t)(signed char)bytes[index];
    unsigned const code = bytes[index];
    return __half2float(__ushort_as_half((unsigned short)(((code & 0x7Fu) << 7) | ((code & 0x80u) << 8)))) * 256.0f;
}

/** The dot product of @p query_row and @p key_row, shared by @p lanes lanes and returned in each:
 *  F32 FMAs, or for I8 an exact I32 sum like the serial backend's, converted once. */
NUMKONG_DEVICE nk_f32_t nk_attention_score_(nk_dtype_t dtype, unsigned char const *query_row,
                                            unsigned char const *key_row, nk_size_t depth, unsigned lane,
                                            unsigned lanes) {
    if (dtype == nk_i8_k) {
        nk_u32_t sum = 0;
        for (nk_size_t element = lane; element < depth; element += lanes)
            sum += (nk_u32_t)((nk_i32_t)(signed char)query_row[element] * (nk_i32_t)(signed char)key_row[element]);
        for (unsigned offset = lanes / 2; offset != 0; offset >>= 1) sum += nk_shuffle_xor_u32_(sum, offset);
        return (nk_f32_t)(nk_i32_t)sum;
    }
    nk_f32_t sum = 0;
    for (nk_size_t element = lane; element < depth; element += lanes)
        sum = fmaf(nk_attention_decode_(dtype, query_row, element), nk_attention_decode_(dtype, key_row, element), sum);
    for (unsigned offset = lanes / 2; offset != 0; offset >>= 1) sum += nk_shuffle_xor_f32_(sum, offset);
    return sum;
}

/** Attends one query row to the keys from @p key_begin up to @p key_end in the serial backend's
 *  two sweeps: the largest score first, then the weighted V rows, summed into @p output_row and
 *  divided by the sum of their weights. */
NUMKONG_DEVICE void nk_attention_fallback_row_(nk_dtype_t dtype, nk_attention_arguments_t const *arguments,
                                               unsigned char const *query_row, unsigned char const *keys_plane,
                                               unsigned char const *values_plane, nk_size_t row_bytes,
                                               nk_size_t positions_padded, unsigned key_begin, unsigned key_end,
                                               nk_f32_t *output_row, unsigned lane, unsigned lanes) {
    nk_size_t const depth = arguments->depth;
    nk_f32_t row_max = nk_attention_negative_infinity_(), weights_sum = 0;
    for (unsigned position = key_begin; position < key_end; ++position)
        row_max = fmaxf(row_max,
                        nk_attention_score_(dtype, query_row, keys_plane + position * row_bytes, depth, lane, lanes) *
                            arguments->scale2);
    for (nk_size_t element = lane; element < depth; element += lanes) output_row[element] = 0;
    for (unsigned position = key_begin; position < key_end; ++position) {
        nk_f32_t const score = nk_attention_score_(dtype, query_row, keys_plane + position * row_bytes, depth, lane,
                                                   lanes);
        nk_f32_t weight = exp2f(score * arguments->scale2 - row_max);
        // A round-down add of 2²³ truncates at the full F32 rate; sm_103 converts at 2 per clock
        if (dtype == nk_i8_k) weight = __fadd_rd(weight * 255.0f + 0.5f, 8388608.0f) - 8388608.0f;
        weights_sum += weight;
        for (nk_size_t element = lane; element < depth; element += lanes) {
            nk_f32_t const value = dtype == nk_bf16_k
                                       ? nk_attention_decode_(dtype, values_plane + position * row_bytes, element)
                                       : nk_attention_decode_(dtype, values_plane + element * positions_padded,
                                                              nk_attention_slot_(position));
            output_row[element] = fmaf(weight, value, output_row[element]);
        }
    }
    nk_f32_t const inverse = weights_sum > 0 ? 1.0f / weights_sum : 0.0f;
    for (nk_size_t element = lane; element < depth; element += lanes) output_row[element] *= inverse;
}

/** The whole baseline kernel, and the tensor-core capabilities' kernel past depth 256: one warp or
 *  wavefront per row, with the output row as the accumulator, so that every depth fits. */
NUMKONG_DEVICE void nk_attention_fallback_(nk_dtype_t dtype, nk_attention_arguments_t const *arguments) {
    __shared__ nk_u64_t prefix[nk_attention_threads_k + 1];
    __shared__ nk_u64_t warp_totals[nk_attention_threads_k / 32];
    unsigned const lanes = nk_warp_lanes_(), lane = threadIdx.x % lanes, group = threadIdx.x / lanes;
    unsigned const groups = nk_attention_threads_k / lanes;
    nk_size_t const element_bytes = dtype == nk_bf16_k ? 2 : 1, depth = arguments->depth;
    nk_size_t const row_bytes = nk_size_round_up_to_multiple_(depth * element_bytes, nk_attention_step_bytes_k);
    nk_attention_schedule_t schedule;
    if (!nk_attention_schedule_start_(arguments, &schedule, prefix, warp_totals)) return;
    nk_attention_work_t work;
    for (nk_size_t item = blockIdx.x; nk_attention_schedule_next_(&schedule, prefix, warp_totals, item, &work);
         item += gridDim.x) {
        nk_size_t length, plane_bytes;
        unsigned char const *keys_plane = nk_attention_keys_plane_(arguments, &work, row_bytes, &length, &plane_bytes);
        unsigned char const *values_plane = keys_plane + arguments->key_value_head_count * plane_bytes;
        for (nk_size_t local = group; local < work.row_count; local += groups) {
            nk_size_t const row = work.row_first + local;
            nk_size_t const query = row / work.heads_selected;
            nk_size_t const head = work.head_first + row % work.heads_selected;
            unsigned char const *query_row = arguments->queries + (work.query_first + query) * arguments->query_stride +
                                             head * depth * element_bytes;
            nk_f32_t *output_row = (nk_f32_t *)((unsigned char *)arguments->output +
                                                (work.query_first + query) * arguments->output_stride) +
                                   head * depth;
            unsigned key_begin, key_end;
            nk_attention_row_keys_((nk_i64_t)query + arguments->diagonal_offset, arguments->window, length, &key_begin,
                                   &key_end);
            nk_attention_fallback_row_(dtype, arguments, query_row, keys_plane, values_plane, row_bytes,
                                       plane_bytes / row_bytes, key_begin, key_end, output_row, lane, lanes);
        }
    }
}

#pragma endregion Tile

#pragma region Pack

/** Writes the header, recording the packing @p capability, and the directory, one block walking the
 *  segments 128 at a time. */
static __global__ void nk_attention_pack_directory_kernel_(unsigned char *packed, nk_size_t key_value_head_count,
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
        nk_u64_t const bytes = 2 * key_value_head_count * nk_size_round_up_to_multiple_(length, nk_attention_panel_k) *
                               row_bytes;
        nk_u64_t total;
        nk_u64_t const inclusive = nk_attention_block_scan_(bytes, warp_totals, &total);
        if (segment < segment_count) payload_offsets[segment] = running + inclusive - bytes, lengths[segment] = length;
        running += total;
    }
    if (threadIdx.x == 0) payload_offsets[segment_count] = running;
    for (nk_size_t byte = (segment_count + 1) * sizeof(nk_u64_t) + segment_count * sizeof(nk_u32_t) + threadIdx.x;
         byte < directory_bytes; byte += blockDim.x)
        packed[sizeof(nk_attention_packed_header_t) + byte] = 0;
}

/** Copies the K and V planes of each @b (segment,kv_head) task, one block per task, zeroing
 *  every padded element. */
NUMKONG_DEVICE void nk_attention_pack_payload_(nk_dtype_t dtype, unsigned char const *keys, unsigned char const *values,
                                               nk_size_t key_value_head_count, nk_size_t depth,
                                               nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                               nk_size_t segment_count, nk_size_t key_stride, nk_size_t value_stride,
                                               unsigned char *packed, nk_size_t task_begin, nk_size_t task_end) {
    nk_size_t const element_bytes = dtype == nk_bf16_k ? 2 : 1;
    nk_size_t const row_bytes = nk_size_round_up_to_multiple_(depth * element_bytes, nk_attention_step_bytes_k);
    nk_size_t const row_elements = row_bytes / element_bytes;
    nk_u64_t const *payload_offsets = (nk_u64_t const *)(packed + sizeof(nk_attention_packed_header_t));
    nk_size_t const directory_bytes = nk_size_round_up_to_multiple_(
        (segment_count + 1) * sizeof(nk_u64_t) + segment_count * sizeof(nk_u32_t), 64);
    unsigned char *payload = packed + sizeof(nk_attention_packed_header_t) + directory_bytes;
    for (nk_size_t task = task_begin + blockIdx.x; task < task_end; task += gridDim.x) {
        nk_size_t const segment = task / key_value_head_count, head = task % key_value_head_count;
        nk_size_t const length = segment_lengths[segment];
        if (length == 0) continue;
        nk_size_t const positions_padded = nk_size_round_up_to_multiple_(length, nk_attention_panel_k);
        nk_size_t const plane_bytes = positions_padded * row_bytes;
        unsigned char *keys_plane = payload + payload_offsets[segment] + head * plane_bytes;
        unsigned char *values_plane = keys_plane + key_value_head_count * plane_bytes;
        unsigned char const *keys_first = keys + segment_offsets[segment] * key_stride + head * depth * element_bytes;
        unsigned char const *values_first = values + segment_offsets[segment] * value_stride +
                                            head * depth * element_bytes;
        for (nk_size_t index = threadIdx.x; index < positions_padded * row_elements; index += blockDim.x) {
            nk_size_t const position = index / row_elements, element = index % row_elements;
            int const inside = position < length && element < depth;
            if (element_bytes == 2) {
                ((unsigned short *)keys_plane)[index] =
                    inside ? ((unsigned short const *)(keys_first + position * key_stride))[element]
                           : (unsigned short)0;
                ((unsigned short *)values_plane)[index] =
                    inside ? ((unsigned short const *)(values_first + position * value_stride))[element]
                           : (unsigned short)0;
            }
            else keys_plane[index] = inside ? keys_first[position * key_stride + element] : (unsigned char)0;
        }
        if (element_bytes == 2) continue;
        for (nk_size_t index = threadIdx.x; index < row_elements * positions_padded; index += blockDim.x) {
            nk_size_t const element = index / positions_padded;
            nk_size_t const position = nk_attention_slot_position_(index % positions_padded);
            values_plane[index] = position < length && element < depth ? values_first[position * value_stride + element]
                                                                       : (unsigned char)0;
        }
    }
}

/** Mirrors the device pack: header, directory, then both planes of every segment and head. */
NUMKONG_INLINE nk_size_t nk_attention_pack_size_(nk_size_t key_value_head_count, nk_size_t depth,
                                                 nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                 nk_size_t element_bytes) {
    nk_size_t const row_bytes = nk_size_round_up_to_multiple_(depth * element_bytes, nk_attention_step_bytes_k);
    nk_size_t payload_bytes = 0;
    for (nk_size_t segment = 0; segment < segment_count; ++segment)
        payload_bytes += 2 * key_value_head_count *
                         nk_size_round_up_to_multiple_(segment_lengths[segment], nk_attention_panel_k) * row_bytes;
    return sizeof(nk_attention_packed_header_t) + nk_attention_pack_directory_size_(segment_count) + payload_bytes;
}

/** Launches the directory writer, recording the packing @p capability, for a window starting at
 *  task 0, then @p payload_kernel over the window. */
NUMKONG_INLINE nk_status_t nk_attention_pack_launch_(void const *payload_kernel, nk_capability_t capability,
                                                     nk_size_t element_bytes, void const *keys, void const *values,
                                                     nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                     nk_size_t segment_count, nk_size_t key_stride,
                                                     nk_size_t value_stride, void *packed, nk_size_t task_begin,
                                                     nk_size_t task_end, void *stream) {
    if ((nk_size_t)packed & 15) return nk_misaligned_k;
    nk_size_t const row_bytes = nk_size_round_up_to_multiple_(depth * element_bytes, nk_attention_step_bytes_k);
    nk_size_t const directory_bytes = nk_attention_pack_directory_size_(segment_count);
    if (task_begin == 0) {
        void *directory_arguments[8] = {&packed,
                                        &key_value_head_count,
                                        &depth,
                                        &segment_lengths,
                                        &segment_count,
                                        (void *)&row_bytes,
                                        (void *)&directory_bytes,
                                        (void *)&capability};
        nk_status_t const status = nk_device_launch_((void const *)nk_attention_pack_directory_kernel_, 1,
                                                     nk_attention_threads_k, directory_arguments, 0, stream);
        if (status != nk_success_k) return status;
    }
    nk_size_t const total_tasks = segment_count * key_value_head_count;
    nk_size_t const end = task_end < total_tasks ? task_end : total_tasks;
    if (task_begin >= end) return nk_success_k;
    void *payload_arguments[12] = {(void *)&keys,    (void *)&values,  &key_value_head_count, &depth,
                                   &segment_offsets, &segment_lengths, &segment_count,        &key_stride,
                                   &value_stride,    &packed,          &task_begin,           (void *)&end};
    return nk_device_launch_(payload_kernel, end - task_begin < 65535 ? end - task_begin : 65535,
                             nk_attention_pack_threads_k, payload_arguments, 0, stream);
}

#pragma endregion Pack

#pragma region Launch

/** The arguments of one launch, with the shared-memory offsets zero for the capability to place. A
 *  bidirectional @p mask lets every row see its whole segment. */
NUMKONG_INLINE nk_attention_arguments_t nk_attention_arguments_init_(
    void const *queries, void const *packed, nk_f32_t *output, nk_size_t head_count, nk_size_t key_value_head_count,
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale,
    nk_f32_t score_scale, nk_f32_t output_scale, nk_attention_mask_t mask, nk_i64_t diagonal_offset, nk_size_t window,
    nk_size_t task_start, nk_size_t task_count) {
    nk_attention_arguments_t arguments;
    arguments.queries = (unsigned char const *)queries, arguments.packed = (unsigned char const *)packed;
    arguments.output = output, arguments.query_offsets = query_offsets;
    arguments.head_count = head_count, arguments.key_value_head_count = key_value_head_count;
    arguments.depth = depth, arguments.query_stride = query_stride, arguments.output_stride = output_stride;
    arguments.diagonal_offset = mask == nk_attention_mask_causal_k ? diagonal_offset : NUMKONG_I64_MAX / 2;
    arguments.window = mask == nk_attention_mask_causal_k ? window : NUMKONG_SIZE_MAX;
    arguments.task_start = task_start, arguments.task_count = task_count;
    arguments.scale2 = scale * NUMKONG_F32_LOG2E_;
    arguments.score_scale = score_scale, arguments.output_scale = output_scale;
    arguments.key_offset[0] = arguments.key_offset[1] = 0;
    arguments.value_offset[0] = arguments.value_offset[1] = 0;
    arguments.query_offset[0] = arguments.query_offset[1] = 0;
    return arguments;
}

/** Validates the contract and launches @p kernel with as many blocks as stay resident. */
NUMKONG_INLINE nk_status_t nk_attention_launch_(void const *kernel, nk_capability_t capability, void const *queries,
                                                void const *packed, nk_f32_t *output, nk_size_t head_count,
                                                nk_size_t key_value_head_count, nk_size_t depth,
                                                nk_u32_t const *query_offsets, nk_size_t query_stride,
                                                nk_size_t output_stride, nk_f32_t scale, nk_attention_mask_t mask,
                                                nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
                                                nk_size_t task_count, void *stream) {
    if (((nk_size_t)packed & 15) || (((nk_size_t)output | output_stride) & 3)) return nk_misaligned_k;
    if (key_value_head_count == 0 || head_count % key_value_head_count != 0) return nk_unexpected_dimensions_k;
    if (task_count == 0 || depth == 0) return nk_success_k;
    nk_attention_arguments_t arguments = nk_attention_arguments_init_(
        queries, packed, output, head_count, key_value_head_count, depth, query_offsets, query_stride, output_stride,
        scale, 1, 1, mask, diagonal_offset, window, task_start, task_count);
    return nk_device_launch_resident_(kernel, nk_attention_threads_k, 0, 0, NUMKONG_SIZE_MAX, &arguments, stream);
}

#pragma endregion Launch

#pragma region Attention Macros

/** Generates the host-side size of a pack: header, directory, and both planes of every
 *  segment and head. */
#define nk_define_device_attention_pack_size_(input_type_name, isa_suffix, element_bytes)                             \
    NUMKONG_API nk_status_t nk_attention_pack_size_##input_type_name##_##isa_suffix(                                  \
        nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *segment_lengths, nk_size_t segment_count,    \
        nk_size_t *bytes) {                                                                                           \
        *bytes = nk_attention_pack_size_(key_value_head_count, depth, segment_lengths, segment_count, element_bytes); \
        return nk_success_k;                                                                                          \
    }

/** Generates a shape accessor that copies a device pack's header back and checks its capability. */
#define nk_define_device_attention_packed_shape_(input_type_name, isa_suffix)                                  \
    NUMKONG_API nk_status_t nk_attention_packed_shape_##input_type_name##_##isa_suffix(                        \
        void const *key_value_packed, nk_size_t *heads, nk_size_t *depth, nk_size_t *segments, void *stream) { \
        nk_attention_packed_header_t header;                                                                   \
        nk_status_t const status = nk_device_read_(&header, key_value_packed, sizeof(header), stream);         \
        if (status != nk_success_k) return status;                                                             \
        if (header.capability != nk_cap_##isa_suffix##_k) return nk_pack_mismatch_k;                           \
        *heads = header.heads, *depth = header.depth, *segments = header.segments;                             \
        return nk_success_k;                                                                                   \
    }

/**
 *  @brief Generates a device pack: the directory, recording the packing @p isa_suffix, when
 *      the window starts at task 0, then one block per task.
 *
 *  V keeps position rows for BF16 and takes σ-ordered depth rows for 1-byte dtypes.
 */
#define nk_define_device_attention_pack_(input_type_name, isa_suffix, input_value_type)                                \
    static __global__ void nk_attention_pack_##input_type_name##_##isa_suffix##_kernel_(                               \
        unsigned char const *keys, unsigned char const *values, nk_size_t key_value_head_count, nk_size_t depth,       \
        nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, nk_size_t segment_count,                     \
        nk_size_t key_stride, nk_size_t value_stride, unsigned char *packed, nk_size_t task_begin,                     \
        nk_size_t task_end) {                                                                                          \
        nk_attention_pack_payload_(nk_##input_value_type##_k, keys, values, key_value_head_count, depth,               \
                                   segment_offsets, segment_lengths, segment_count, key_stride, value_stride, packed,  \
                                   task_begin, task_end);                                                              \
    }                                                                                                                  \
    NUMKONG_API nk_status_t nk_attention_pack_##input_type_name##_##isa_suffix(                                        \
        nk_##input_value_type##_t const *keys, nk_##input_value_type##_t const *values,                                \
        nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *segment_offsets,                              \
        nk_u32_t const *segment_lengths, nk_size_t segment_count, nk_size_t key_stride_bytes,                          \
        nk_size_t value_stride_bytes, void *key_value_packed, nk_size_t task_begin, nk_size_t task_end,                \
        void *stream) {                                                                                                \
        return nk_attention_pack_launch_((void const *)nk_attention_pack_##input_type_name##_##isa_suffix##_kernel_,   \
                                         nk_cap_##isa_suffix##_k, sizeof(nk_##input_value_type##_t), keys, values,     \
                                         key_value_head_count, depth, segment_offsets, segment_lengths, segment_count, \
                                         key_stride_bytes, value_stride_bytes, key_value_packed, task_begin, task_end, \
                                         stream);                                                                      \
    }

/**
 *  @brief Generates the fallback kernel of one dtype and both public attention entry points, each
 *      passing its @c nk_attention_mask_t to the shared launch.
 *
 *  The pack must come from the same capability's pack kernel: the host can't read its device header
 *  without waiting on the stream, so the entry points trust it.
 */
#define nk_define_device_attention_baseline_packed_(input_type_name, isa_suffix, input_value_type)                   \
    static __global__ void __launch_bounds__(nk_attention_threads_k)                                                 \
        nk_attention_packed_##input_type_name##_##isa_suffix##_kernel_(nk_attention_arguments_t arguments) {         \
        nk_attention_fallback_(nk_##input_value_type##_k, &arguments);                                               \
    }                                                                                                                \
    NUMKONG_API nk_status_t nk_attention_bidirectional_packed_##input_type_name##_##isa_suffix(                      \
        nk_##input_value_type##_t const *queries, void const *key_value_packed, nk_f32_t *output,                    \
        nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,        \
        nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start,           \
        nk_size_t task_count, void *stream) {                                                                        \
        return nk_attention_launch_((void const *)nk_attention_packed_##input_type_name##_##isa_suffix##_kernel_,    \
                                    nk_cap_##isa_suffix##_k, queries, key_value_packed, output, head_count,          \
                                    key_value_head_count, depth, query_offsets, query_stride_bytes,                  \
                                    output_stride_bytes, scale, nk_attention_mask_bidirectional_k, 0, 0, task_start, \
                                    task_count, stream);                                                             \
    }                                                                                                                \
    NUMKONG_API nk_status_t nk_attention_causal_packed_##input_type_name##_##isa_suffix(                             \
        nk_##input_value_type##_t const *queries, void const *key_value_packed, nk_f32_t *output,                    \
        nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,        \
        nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset,       \
        nk_size_t window, nk_size_t task_start, nk_size_t task_count, void *stream) {                                \
        return nk_attention_launch_((void const *)nk_attention_packed_##input_type_name##_##isa_suffix##_kernel_,    \
                                    nk_cap_##isa_suffix##_k, queries, key_value_packed, output, head_count,          \
                                    key_value_head_count, depth, query_offsets, query_stride_bytes,                  \
                                    output_stride_bytes, scale, nk_attention_mask_causal_k, diagonal_offset, window, \
                                    task_start, task_count, stream);                                                 \
    }

/** Every attention entry of one dtype on the vendor baseline @p isa_suffix, with its own pack. */
#define nk_define_device_attention_baseline_(input_type_name, isa_suffix, input_value_type, element_bytes) \
    nk_define_device_attention_pack_size_(input_type_name, isa_suffix, element_bytes)                      \
    nk_define_device_attention_packed_shape_(input_type_name, isa_suffix)                                  \
    nk_define_device_attention_pack_(input_type_name, isa_suffix, input_value_type)                        \
    nk_define_device_attention_baseline_packed_(input_type_name, isa_suffix, input_value_type)

/**
 *  @brief Generates the narrow, wide and fallback kernels of one dtype and both public attention
 *      entry points, each passing its @c nk_attention_mask_t to the capability's launch.
 *
 *  The pack must come from the same capability's pack kernel, which the entry points trust, as
 *  @c nk_define_device_attention_baseline_packed_ explains.
 *
 *  @param[in] tile The tiling kernel family, like @c ampere, whose tile function the narrow and
 *      wide kernels run.
 *  @param[in] launch_fn The launch, taking the narrow, wide and fallback kernels in that order.
 *  @param[in] score_scale Undoes the power of two that converting Q and K puts on scores, or 1.
 *  @param[in] output_scale Undoes the power of two that converting V puts on the output, or 1.
 */
#define nk_define_device_attention_packed_(input_type_name, isa_suffix, tile, launch_fn, input_value_type, epilogue,  \
                                           scores_fn, values_mma_fn, weights_fn, score_scale, output_scale)           \
    static __global__ void __launch_bounds__(nk_attention_threads_k)                                                  \
        nk_attention_packed_##input_type_name##_##isa_suffix##_narrow_kernel_(nk_attention_arguments_t arguments) {   \
        nk_attention_tile_##tile##_(nk_##input_value_type##_k, nk_attention_width_128_k, epilogue, scores_fn,         \
                                    values_mma_fn, weights_fn, &arguments);                                           \
    }                                                                                                                 \
    static __global__ void __launch_bounds__(nk_attention_threads_k)                                                  \
        nk_attention_packed_##input_type_name##_##isa_suffix##_wide_kernel_(nk_attention_arguments_t arguments) {     \
        nk_attention_tile_##tile##_(nk_##input_value_type##_k, nk_attention_width_256_k, epilogue, scores_fn,         \
                                    values_mma_fn, weights_fn, &arguments);                                           \
    }                                                                                                                 \
    static __global__ void __launch_bounds__(nk_attention_threads_k)                                                  \
        nk_attention_packed_##input_type_name##_##isa_suffix##_fallback_kernel_(nk_attention_arguments_t arguments) { \
        nk_attention_fallback_(nk_##input_value_type##_k, &arguments);                                                \
    }                                                                                                                 \
    NUMKONG_API nk_status_t nk_attention_bidirectional_packed_##input_type_name##_##isa_suffix(                       \
        nk_##input_value_type##_t const *queries, void const *key_value_packed, nk_f32_t *output,                     \
        nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,         \
        nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start,            \
        nk_size_t task_count, void *stream) {                                                                         \
        return launch_fn((void const *)nk_attention_packed_##input_type_name##_##isa_suffix##_narrow_kernel_,         \
                         (void const *)nk_attention_packed_##input_type_name##_##isa_suffix##_wide_kernel_,           \
                         (void const *)nk_attention_packed_##input_type_name##_##isa_suffix##_fallback_kernel_,       \
                         nk_##input_value_type##_k, queries, key_value_packed, output, head_count,                    \
                         key_value_head_count, depth, query_offsets, query_stride_bytes, output_stride_bytes, scale,  \
                         score_scale, output_scale, nk_attention_mask_bidirectional_k, 0, 0, task_start, task_count,  \
                         stream);                                                                                     \
    }                                                                                                                 \
    NUMKONG_API nk_status_t nk_attention_causal_packed_##input_type_name##_##isa_suffix(                              \
        nk_##input_value_type##_t const *queries, void const *key_value_packed, nk_f32_t *output,                     \
        nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,         \
        nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset,        \
        nk_size_t window, nk_size_t task_start, nk_size_t task_count, void *stream) {                                 \
        return launch_fn((void const *)nk_attention_packed_##input_type_name##_##isa_suffix##_narrow_kernel_,         \
                         (void const *)nk_attention_packed_##input_type_name##_##isa_suffix##_wide_kernel_,           \
                         (void const *)nk_attention_packed_##input_type_name##_##isa_suffix##_fallback_kernel_,       \
                         nk_##input_value_type##_k, queries, key_value_packed, output, head_count,                    \
                         key_value_head_count, depth, query_offsets, query_stride_bytes, output_stride_bytes, scale,  \
                         score_scale, output_scale, nk_attention_mask_causal_k, diagonal_offset, window, task_start,  \
                         task_count, stream);                                                                         \
    }

#pragma endregion Attention Macros

#pragma region Rotary Embeddings

/** Everything one RoPE launch shares, passed by value as the kernels' only argument. */
typedef struct {
    unsigned char const *x;
    nk_f32_t const *cos;
    nk_f32_t const *sin;
    unsigned char *y;
    nk_size_t head_count;
    nk_size_t half_depth;
    nk_size_t heads;
    nk_size_t x_stride_bytes;
    nk_size_t y_stride_bytes;
    nk_f32_t input_scale;
} nk_attention_rope_arguments_t;

/** Validates the contract and launches @p kernel with a warp or wavefront per head of every row, at
 *  most 2²⁰ blocks of them. */
NUMKONG_INLINE nk_status_t nk_attention_rope_launch_(void const *kernel, nk_size_t value_bytes, void const *x,
                                                     nk_f32_t const *cos, nk_f32_t const *sin, void *y, nk_size_t rows,
                                                     nk_size_t head_count, nk_size_t depth, nk_size_t x_stride_bytes,
                                                     nk_size_t y_stride_bytes, nk_f32_t input_scale, void *stream) {
    if (depth % 2) return nk_unexpected_dimensions_k;
    if ((((nk_size_t)x) | x_stride_bytes | ((nk_size_t)y) | y_stride_bytes) & (value_bytes - 1) ||
        (((nk_size_t)cos) | ((nk_size_t)sin)) & 3)
        return nk_misaligned_k;
    nk_attention_rope_arguments_t arguments;
    arguments.x = (unsigned char const *)x, arguments.cos = cos, arguments.sin = sin, arguments.y = (unsigned char *)y;
    arguments.head_count = head_count, arguments.half_depth = depth / 2, arguments.heads = rows * head_count;
    arguments.x_stride_bytes = x_stride_bytes, arguments.y_stride_bytes = y_stride_bytes;
    arguments.input_scale = input_scale;
    if (arguments.heads == 0 || depth == 0) return nk_success_k;
    nk_size_t const blocks = nk_size_divide_round_up_(arguments.heads, nk_attention_threads_k / 32),
                    blocks_limit = (nk_size_t)1 << 20;
    void *launch_arguments[1];
    launch_arguments[0] = &arguments;
    return nk_device_launch_(kernel, blocks < blocks_limit ? blocks : blocks_limit, nk_attention_threads_k,
                             launch_arguments, 0, stream);
}

/** Generates the RoPE kernel of @p input_type and its host entry point for @p isa_suffix, after
 *  @c nk_define_attention_rope_ of the serial backend: each warp or wavefront rotates one head of
 *  one row, so the row and head divisions happen once per head rather than once per pair. */
#define nk_define_device_attention_rope_(input_type, isa_suffix, load_and_convert, convert_and_store)               \
    static __global__ void nk_attention_rope_##input_type##_##isa_suffix##_kernel_(                                 \
        nk_attention_rope_arguments_t arguments) {                                                                  \
        unsigned const lanes = nk_warp_lanes_(), lane = threadIdx.x % lanes;                                        \
        nk_size_t const groups = blockDim.x / lanes;                                                                \
        for (nk_size_t head = (nk_size_t)blockIdx.x * groups + threadIdx.x / lanes; head < arguments.heads;         \
             head += (nk_size_t)gridDim.x * groups) {                                                               \
            nk_size_t const row = head / arguments.head_count;                                                      \
            nk_size_t const first = (head - row * arguments.head_count) * 2 * arguments.half_depth;                 \
            nk_f32_t const *cos_row = arguments.cos + row * arguments.half_depth;                                   \
            nk_f32_t const *sin_row = arguments.sin + row * arguments.half_depth;                                   \
            nk_##input_type##_t const *x =                                                                          \
                (nk_##input_type##_t const *)(arguments.x + row * arguments.x_stride_bytes) + first;                \
            nk_##input_type##_t *y = (nk_##input_type##_t *)(arguments.y + row * arguments.y_stride_bytes) + first; \
            for (nk_size_t pair = lane; pair < arguments.half_depth; pair += lanes) {                               \
                nk_f32_t low, high;                                                                                 \
                load_and_convert(x + pair, &low);                                                                   \
                load_and_convert(x + pair + arguments.half_depth, &high);                                           \
                low *= arguments.input_scale, high *= arguments.input_scale;                                        \
                nk_f32_t const cosine = cos_row[pair], sine = sin_row[pair];                                        \
                nk_f32_t const rotated_low = low * cosine - high * sine;                                            \
                nk_f32_t const rotated_high = low * sine + high * cosine;                                           \
                convert_and_store(&rotated_low, y + pair);                                                          \
                convert_and_store(&rotated_high, y + pair + arguments.half_depth);                                  \
            }                                                                                                       \
        }                                                                                                           \
    }                                                                                                               \
    NUMKONG_API nk_status_t nk_attention_rope_##input_type##_##isa_suffix(                                          \
        nk_##input_type##_t const *x, nk_f32_t const *cos, nk_f32_t const *sin, nk_##input_type##_t *y,             \
        nk_size_t rows, nk_size_t head_count, nk_size_t depth, nk_size_t x_stride_bytes, nk_size_t y_stride_bytes,  \
        nk_f32_t input_scale, void *stream) {                                                                       \
        return nk_attention_rope_launch_((void const *)&nk_attention_rope_##input_type##_##isa_suffix##_kernel_,    \
                                         sizeof(nk_##input_type##_t), x, cos, sin, y, rows, head_count, depth,      \
                                         x_stride_bytes, y_stride_bytes, input_scale, stream);                      \
    }

#pragma endregion Rotary Embeddings

#pragma region Instantiations

#if NUMKONG_TARGET_CUDA
nk_define_device_attention_baseline_(bf16, cuda, bf16, 2)
nk_define_device_attention_baseline_(e4m3, cuda, e4m3, 1)
nk_define_device_attention_baseline_(i8, cuda, i8, 1)
nk_define_device_attention_rope_(f32, cuda, nk_assign_from_to_, nk_assign_from_to_)
nk_define_device_attention_rope_(bf16, cuda, nk_bf16_to_f32_simt_, nk_f32_to_bf16_simt_)
nk_define_device_attention_rope_(e4m3, cuda, nk_e4m3_to_f32_simt_, nk_f32_to_e4m3_simt_)
#elif NUMKONG_TARGET_ROCM
nk_define_device_attention_baseline_(bf16, rocm, bf16, 2)
nk_define_device_attention_baseline_(e4m3, rocm, e4m3, 1)
nk_define_device_attention_baseline_(i8, rocm, i8, 1)
#endif

#pragma endregion Instantiations

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_
#endif // NUMKONG_ATTENTION_SIMT_CUH
