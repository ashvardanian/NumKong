/**
 *  @file include/numkong/attention/ampere.cuh
 *  @author Ash Vardanian
 *  @date September 22, 2026
 *  @brief Ragged attention for NVIDIA Ampere and newer.
 *
 *  @sa include/numkong/attention.h
 *  @sa include/numkong/dots/ampere.cuh
 *
 *  FlashAttention-2 on warp-level `mma.sync`: four warps own 64 rows folding the GQA group, stream
 *  K and V panels through `cp.async`, and keep a base-2 online softmax per row; items of at most 16
 *  rows split every panel across the warps instead. BF16 and widened E4M3 multiply on F16-class
 *  MMA, I8 on exact integer MMA with U8 probabilities, and depths above 256 fall back to a
 *  CUDA-core kernel instead of MMA.
 *
 *  The pack keeps the serial header and directory, with 1-byte V transposed and permuted within
 *  every 16 positions so probabilities become MMA fragments without shuffles. Only @c pack_size
 *  reads @c segment_lengths on the host, so the pack and both attention kernels need device or
 *  managed memory for offsets and lengths.
 */
#ifndef NK_ATTENTION_AMPERE_CUH
#define NK_ATTENTION_AMPERE_CUH

#if NK_TARGET_AMPERE

#include "numkong/attention/serial.h" // `nk_attention_packed_header_t`, `nk_attention_pack_directory_size_`
#include "numkong/dots/ampere.cuh" // `nk_mma_bf16_ampere_`, `nk_load_matrices_x4_ampere_`, `nk_copy_b128_async_ampere_`

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Configuration

enum {
    nk_attention_threads_ampere_k = 128,
    nk_attention_warp_rows_ampere_k = 16,
    nk_attention_block_rows_ampere_k = 64,
    nk_attention_panel_ampere_k = 64,
    nk_attention_wide_panel_ampere_k = 32,
    nk_attention_group_ampere_k = 16,
    nk_attention_step_bytes_ampere_k = 32,
    nk_attention_row_padding_ampere_k = 16,
    nk_attention_narrow_depth_ampere_k = 128,
    nk_attention_wide_depth_ampere_k = 256,
    nk_attention_pack_threads_ampere_k = 256,
};

/** Which keys each query row sees. */
typedef enum {
    nk_attention_mask_bidirectional_k, // every key of the row's segment
    nk_attention_mask_causal_k,        // the `window` keys ending at the row's position `r + diagonal_offset`
} nk_attention_mask_t;

/** How fragments are formed from the packed operands. */
typedef enum {
    nk_attention_kind_bf16_k,    // 2-byte elements into m16n8k16, V rows read through `ldmatrix.trans`
    nk_attention_kind_widened_k, // 1-byte E4M3 codes widened into F16 pairs for m16n8k16, V transposed
    nk_attention_kind_bytes_k,   // 1-byte codes straight into m16n8k32, V transposed
} nk_attention_kind_t;

/** Where the head's depth puts a launch. */
typedef enum {
    nk_attention_tier_narrow_k, // depth ≤ 128: 64-position panels, Q fragments in registers
    nk_attention_tier_wide_k,   // depth ≤ 256: 32-position panels, Q fragments reloaded from shared memory
} nk_attention_tier_t;

/** How the 4 warps of a block share an item. */
typedef enum {
    nk_attention_split_rows_k,      // each warp owns 16 of the 64 rows and every position of each panel
    nk_attention_split_positions_k, // each warp owns 16 positions of each 64-position panel and all 16 rows
} nk_attention_split_t;

/** Adds one 32-byte depth step of 16 rows against 16 positions: Q as @c nk_attention_query_ampere_
 *  forms it, K as `ldmatrix.x4` loads it. */
typedef void (*nk_attention_scores_ampere_t)(nk_fui32_t scores[2][4], nk_u32_t const query[8], nk_u32_t const keys[4]);

/** Packs one row's 4 probabilities of 16 positions into A-fragment registers and adds the values
 *  P · V sees to @p sum. */
typedef void (*nk_attention_weights_ampere_t)(nk_f32_t const probabilities[4], nk_u32_t packed[2], nk_f32_t *sum);

/** Everything one attention launch shares, passed by value as the kernels' only argument. */
typedef struct {
    unsigned char const *queries;   // `[query tokens, heads × depth]` rows
    unsigned char const *packed;    // the packed buffer, header first
    nk_f32_t *output;               // `[query tokens, heads × depth]` F32 rows
    nk_u32_t const *query_offsets;  // first query row of each segment, `[segments + 1]`
    nk_size_t head_count;           // query heads
    nk_size_t key_value_head_count; // K and V heads
    nk_size_t depth;                // elements per head
    nk_size_t query_stride;         // bytes between query rows
    nk_size_t output_stride;        // bytes between output rows
    nk_i64_t diagonal_offset;       // position of query row 0 in its segment
    nk_size_t window;               // visible keys including the query's own
    nk_size_t task_start;           // first task of the window over `segments × heads`
    nk_size_t task_count;           // tasks in the window, clipped on the device
    nk_f32_t scale2;                // score multiplier in base 2, `scale · log₂e`
    nk_f32_t score_scale;           // undoes the Q and K widenings in the tile, or 1
    nk_f32_t output_scale;          // undoes the V widening in the tile, or 1
    nk_u32_t key_offset[2];         // K panel in shared memory, per `nk_attention_split_t`
    nk_u32_t value_offset[2];       // V panel in shared memory, per `nk_attention_split_t`
    nk_u32_t query_offset[2];       // staged Q rows in shared memory, per `nk_attention_split_t`
} nk_attention_arguments_ampere_t;

/** Walks the work items of a task window, one chunk of segments' item counts at a time. */
typedef struct {
    nk_u32_t const *query_offsets; // first query row of each segment
    nk_size_t head_count;          // query heads per segment
    nk_size_t group_heads;         // query heads per K and V head
    nk_size_t task_start;          // first task of the window
    nk_size_t task_end;            // one past the last task, clipped to the grid
    nk_size_t segment_end;         // one past the last segment the window touches
    nk_size_t chunk_first;         // first segment of the chunk whose prefix sits in shared memory
    nk_size_t items_before;        // items in the segments before the chunk
} nk_attention_schedule_ampere_t;

/** One work item: up to 64 rows, each row being query × heads_selected + head, of one segment
 *  against one K and V head. */
typedef struct {
    nk_size_t segment;        // segment index
    nk_size_t key_value_head; // K and V head
    nk_size_t head_first;     // first query head, counted within the segment
    nk_size_t heads_selected; // query heads the rows cycle through
    nk_size_t row_first;      // first row of the item
    nk_size_t row_count;      // rows of the item
    nk_size_t query_first;    // query token of the segment's first query
} nk_attention_work_ampere_t;

#pragma endregion Configuration

#pragma region Instructions

/* Rounds two F32 into a BF16 pair, @p low in the low half. */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
NK_HELPER_DEVICE_INLINE nk_u32_t nk_f32x2_to_bf16x2_ampere_(nk_f32_t low, nk_f32_t high) {
    nk_u32_t pair;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(pair) : "f"(high), "f"(low));
    return pair;
}

/* Rounds two F32 into an F16 pair, @p low in the low half. */
NK_HELPER_DEVICE_INLINE nk_u32_t nk_f32x2_to_f16x2_ampere_(nk_f32_t low, nk_f32_t high) {
    nk_u32_t pair;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(pair) : "f"(high), "f"(low));
    return pair;
}

/* 2^x to 2⁻²² relative, flushing results below 2⁻¹²⁶ to zero, and 0 for -∞. */
NK_HELPER_DEVICE_INLINE nk_f32_t nk_f32_exp2_ampere_(nk_f32_t exponent) {
    nk_f32_t power;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(power) : "f"(exponent));
    return power;
}

#else

NK_HELPER_DEVICE_INLINE nk_f32_t nk_f32_exp2_ampere_(nk_f32_t exponent) {
    __trap();
    return 0;
}
NK_HELPER_DEVICE_INLINE nk_u32_t nk_f32x2_to_bf16x2_ampere_(nk_f32_t low, nk_f32_t high) {
    __trap();
    return 0;
}
NK_HELPER_DEVICE_INLINE nk_u32_t nk_f32x2_to_f16x2_ampere_(nk_f32_t low, nk_f32_t high) {
    __trap();
    return 0;
}

#endif // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

#pragma endregion Instructions

#pragma region Fragments

NK_HELPER_DEVICE_INLINE nk_f32_t nk_attention_negative_infinity_ampere_(void) { return __int_as_float(0xFF800000); }

NK_HELPER_DEVICE_INLINE void nk_attention_scores_bf16_ampere_(nk_fui32_t scores[2][4], nk_u32_t const query[8],
                                                              nk_u32_t const keys[4]) {
    nk_mma_bf16_ampere_(scores[0], query, keys[0], keys[1]);
    nk_mma_bf16_ampere_(scores[1], query, keys[2], keys[3]);
}

/** Widens K as the E4M3 dots do, in the depth order @c nk_attention_query_ampere_ widened Q in. */
NK_HELPER_DEVICE_INLINE void nk_attention_scores_e4m3_ampere_(nk_fui32_t scores[2][4], nk_u32_t const query[8],
                                                              nk_u32_t const keys[4]) {
#pragma unroll
    for (unsigned tile = 0; tile < 2; ++tile) {
        nk_u32_t first_low, first_high, second_low, second_high;
        nk_e4m3x4_to_f16x4_ampere_(keys[tile * 2], &first_low, &first_high);
        nk_e4m3x4_to_f16x4_ampere_(keys[tile * 2 + 1], &second_low, &second_high);
        nk_mma_f16_ampere_(scores[tile], query, first_low, first_high);
        nk_mma_f16_ampere_(scores[tile], query + 4, second_low, second_high);
    }
}

NK_HELPER_DEVICE_INLINE void nk_attention_scores_i8_ampere_(nk_fui32_t scores[2][4], nk_u32_t const query[8],
                                                            nk_u32_t const keys[4]) {
    nk_mma_i8_ampere_(scores[0], query, keys[0], keys[1]);
    nk_mma_i8_ampere_(scores[1], query, keys[2], keys[3]);
}

/** Q's A fragment for one 32-byte depth step, E4M3 codes widened once into two F16 fragments of
 *  16 depths each. */
NK_HELPER_DEVICE_INLINE void nk_attention_query_ampere_(nk_attention_kind_t kind, nk_u32_t const fragment[4],
                                                        nk_u32_t query[8]) {
    if (kind != nk_attention_kind_widened_k) {
#pragma unroll
        for (unsigned index = 0; index < 4; ++index) query[index] = fragment[index];
        return;
    }
    nk_u32_t low[4], high[4];
#pragma unroll
    for (unsigned index = 0; index < 4; ++index) nk_e4m3x4_to_f16x4_ampere_(fragment[index], &low[index], &high[index]);
    query[0] = low[0], query[1] = low[1], query[2] = high[0], query[3] = high[1];
    query[4] = low[2], query[5] = low[3], query[6] = high[2], query[7] = high[3];
}

NK_HELPER_DEVICE_INLINE void nk_attention_weights_bf16_ampere_(nk_f32_t const probabilities[4], nk_u32_t packed[2],
                                                               nk_f32_t *sum) {
    packed[0] = nk_f32x2_to_bf16x2_ampere_(probabilities[0], probabilities[1]);
    packed[1] = nk_f32x2_to_bf16x2_ampere_(probabilities[2], probabilities[3]);
    *sum += (__uint_as_float(packed[0] << 16) + __uint_as_float(packed[0] & 0xFFFF0000u)) +
            (__uint_as_float(packed[1] << 16) + __uint_as_float(packed[1] & 0xFFFF0000u));
}

NK_HELPER_DEVICE_INLINE void nk_attention_weights_f16_ampere_(nk_f32_t const probabilities[4], nk_u32_t packed[2],
                                                              nk_f32_t *sum) {
    packed[0] = nk_f32x2_to_f16x2_ampere_(probabilities[0], probabilities[1]);
    packed[1] = nk_f32x2_to_f16x2_ampere_(probabilities[2], probabilities[3]);
#pragma unroll
    for (unsigned word = 0; word < 2; ++word)
        *sum += __half2float(__ushort_as_half((unsigned short)(packed[word] & 0xFFFFu))) +
                __half2float(__ushort_as_half((unsigned short)(packed[word] >> 16)));
}

/** U8 weights round(255 · p), the max-scoring position landing on 255. */
NK_HELPER_DEVICE_INLINE void nk_attention_weights_u8_ampere_(nk_f32_t const probabilities[4], nk_u32_t packed[2],
                                                             nk_f32_t *sum) {
    nk_u32_t bytes = 0, total = 0;
#pragma unroll
    for (unsigned index = 0; index < 4; ++index) {
        nk_u32_t const weight = (nk_u32_t)(probabilities[index] * 255.0f + 0.5f);
        bytes |= weight << (index * 8), total += weight;
    }
    packed[0] = bytes, packed[1] = 0;
    *sum += (nk_f32_t)total;
}

/** Slot of @p position in a transposed V: within every 16, position `2t + 8h + e` sits at
 *  `4t + 2h + e`. */
NK_HELPER_DEVICE_INLINE nk_size_t nk_attention_slot_ampere_(nk_size_t position) {
    return (position & ~(nk_size_t)15) | ((position & 6) << 1) | ((position & 8) >> 2) | (position & 1);
}

/** Position held by @p slot of a transposed V, inverting @c nk_attention_slot_ampere_. */
NK_HELPER_DEVICE_INLINE nk_size_t nk_attention_slot_position_ampere_(nk_size_t slot) {
    return (slot & ~(nk_size_t)15) | ((slot & 12) >> 1) | ((slot & 2) << 2) | (slot & 1);
}

/** Writes the half-open range of keys visible to the query at @p position into @p key_begin and
 *  @p key_end, with the serial backend's rule. */
NK_HELPER_DEVICE_INLINE void nk_attention_row_keys_ampere_(nk_i64_t position, nk_size_t window, nk_size_t length,
                                                           unsigned *key_begin, unsigned *key_end) {
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
NK_HELPER_DEVICE_INLINE nk_u64_t nk_attention_block_scan_ampere_(nk_u64_t value, nk_u64_t *warp_totals,
                                                                 nk_u64_t *total) {
    unsigned const lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
#pragma unroll
    for (unsigned offset = 1; offset < 32; offset <<= 1) {
        nk_u64_t const other = __shfl_up_sync(0xFFFFFFFFu, value, offset);
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

NK_HELPER_DEVICE_INLINE nk_size_t nk_attention_row_blocks_ampere_(nk_size_t queries, nk_size_t heads) {
    return (queries * heads + nk_attention_block_rows_ampere_k - 1) / nk_attention_block_rows_ampere_k;
}

/** Writes the half-open range of heads of @p segment inside the task window into @p head_begin and
 *  @p head_end, counted within the segment. */
NK_HELPER_DEVICE_INLINE void nk_attention_segment_heads_ampere_(nk_attention_schedule_ampere_t const *schedule,
                                                                nk_size_t segment, nk_size_t *head_begin,
                                                                nk_size_t *head_end) {
    nk_size_t const task_first = segment * schedule->head_count;
    *head_begin = schedule->task_start > task_first ? schedule->task_start - task_first : 0;
    *head_end = schedule->task_end < task_first + schedule->head_count ? schedule->task_end - task_first
                                                                       : schedule->head_count;
}

NK_HELPER_DEVICE_INLINE nk_size_t nk_attention_segment_items_ampere_(nk_attention_schedule_ampere_t const *schedule,
                                                                     nk_size_t segment) {
    if (segment >= schedule->segment_end) return 0;
    nk_size_t const queries = schedule->query_offsets[segment + 1] - schedule->query_offsets[segment];
    nk_size_t head_begin, head_end;
    nk_attention_segment_heads_ampere_(schedule, segment, &head_begin, &head_end);
    if (queries == 0 || head_begin >= head_end) return 0;
    nk_size_t const group = schedule->group_heads;
    nk_size_t const first = head_begin / group, last = (head_end - 1) / group;
    if (first == last) return nk_attention_row_blocks_ampere_(queries, head_end - head_begin);
    return nk_attention_row_blocks_ampere_(queries, (first + 1) * group - head_begin) +
           (last - first - 1) * nk_attention_row_blocks_ampere_(queries, group) +
           nk_attention_row_blocks_ampere_(queries, head_end - last * group);
}

/** Fills @p prefix with the running item counts of the 128 segments from
 *  `schedule->chunk_first`. */
NK_HELPER_DEVICE_INLINE void nk_attention_schedule_chunk_ampere_(nk_attention_schedule_ampere_t const *schedule,
                                                                 nk_u64_t *prefix, nk_u64_t *warp_totals) {
    nk_u64_t const items = nk_attention_segment_items_ampere_(schedule, schedule->chunk_first + threadIdx.x);
    nk_u64_t total;
    nk_u64_t const inclusive = nk_attention_block_scan_ampere_(items, warp_totals, &total);
    prefix[threadIdx.x + 1] = inclusive;
    if (threadIdx.x == 0) prefix[0] = 0;
    __syncthreads();
}

/** Finds work item @p item, returning 0 once the window has no more; every thread of the block
 *  calls it in step. */
NK_HELPER_DEVICE_INLINE int nk_attention_schedule_next_ampere_(nk_attention_schedule_ampere_t *schedule,
                                                               nk_u64_t *prefix, nk_u64_t *warp_totals, nk_size_t item,
                                                               nk_attention_work_ampere_t *work) {
    while (item >= schedule->items_before + prefix[nk_attention_threads_ampere_k]) {
        if (schedule->chunk_first + nk_attention_threads_ampere_k >= schedule->segment_end) return 0;
        schedule->items_before += prefix[nk_attention_threads_ampere_k];
        schedule->chunk_first += nk_attention_threads_ampere_k;
        __syncthreads();
        nk_attention_schedule_chunk_ampere_(schedule, prefix, warp_totals);
    }
    nk_u64_t const local = item - schedule->items_before;
    unsigned low = 0, high = nk_attention_threads_ampere_k;
    while (high - low > 1) {
        unsigned const middle = (low + high) >> 1;
        if (prefix[middle] <= local) low = middle;
        else high = middle;
    }
    nk_size_t const segment = schedule->chunk_first + low;
    nk_size_t index = (nk_size_t)(local - prefix[low]);
    nk_size_t const queries = schedule->query_offsets[segment + 1] - schedule->query_offsets[segment];
    nk_size_t head_begin, head_end;
    nk_attention_segment_heads_ampere_(schedule, segment, &head_begin, &head_end);
    nk_size_t const group = schedule->group_heads;
    nk_size_t const first = head_begin / group, last = (head_end - 1) / group;
    nk_size_t key_value_head = first, row_block = index;
    if (first != last) {
        nk_size_t const first_items = nk_attention_row_blocks_ampere_(queries, (first + 1) * group - head_begin);
        if (index >= first_items) {
            index -= first_items;
            nk_size_t const full_items = nk_attention_row_blocks_ampere_(queries, group);
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
    work->row_first = row_block * nk_attention_block_rows_ampere_k;
    nk_size_t const rows = queries * work->heads_selected - work->row_first;
    work->row_count = rows < nk_attention_block_rows_ampere_k ? rows : nk_attention_block_rows_ampere_k;
    work->query_first = schedule->query_offsets[segment];
    return 1;
}

/** Reads the header, returning 0 when it disagrees with the arguments, and primes the
 *  schedule's first chunk. */
NK_HELPER_DEVICE_INLINE int nk_attention_schedule_start_ampere_(nk_attention_arguments_ampere_t const *arguments,
                                                                nk_attention_schedule_ampere_t *schedule,
                                                                nk_u64_t *prefix, nk_u64_t *warp_totals) {
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
    schedule->segment_end = (schedule->task_end + arguments->head_count - 1) / arguments->head_count;
    schedule->chunk_first = arguments->task_start / arguments->head_count;
    schedule->items_before = 0;
    nk_attention_schedule_chunk_ampere_(schedule, prefix, warp_totals);
    return 1;
}

/** K plane of the work item's head; its V plane follows @c key_value_head_count planes later. */
NK_HELPER_DEVICE_INLINE unsigned char const *nk_attention_keys_plane_ampere_(
    nk_attention_arguments_ampere_t const *arguments, nk_attention_work_ampere_t const *work, nk_size_t row_bytes,
    nk_size_t *length, nk_size_t *plane_bytes) {
    nk_size_t const segments = ((nk_attention_packed_header_t const *)arguments->packed)->segments;
    nk_u64_t const *payload_offsets = (nk_u64_t const *)(arguments->packed + sizeof(nk_attention_packed_header_t));
    nk_u32_t const *lengths = (nk_u32_t const *)(payload_offsets + segments + 1);
    nk_size_t const directory_bytes = ((segments + 1) * sizeof(nk_u64_t) + segments * sizeof(nk_u32_t) + 63) & ~63ull;
    *length = lengths[work->segment];
    *plane_bytes = ((*length + nk_attention_panel_ampere_k - 1) & ~(nk_size_t)(nk_attention_panel_ampere_k - 1)) *
                   row_bytes;
    return arguments->packed + sizeof(nk_attention_packed_header_t) + directory_bytes + payload_offsets[work->segment] +
           work->key_value_head * *plane_bytes;
}

#pragma endregion Schedule

#pragma region Tile

/** Issues @p rows × @p row_bytes bytes of a position-major plane from @p first_position into
 *  padded shared rows. */
NK_HELPER_DEVICE_INLINE void nk_attention_stage_rows_ampere_(unsigned char *shared, unsigned char const *plane,
                                                             nk_size_t first_position, unsigned rows,
                                                             unsigned row_bytes) {
    unsigned const chunks_per_row = row_bytes >> 4, chunks = rows * chunks_per_row;
    unsigned const row_stride = row_bytes + nk_attention_row_padding_ampere_k;
    for (unsigned chunk = threadIdx.x; chunk < chunks; chunk += nk_attention_threads_ampere_k) {
        unsigned const row = chunk / chunks_per_row, column = chunk - row * chunks_per_row;
        nk_copy_b128_async_ampere_(nk_shared_address_ampere_(shared + row * row_stride + (column << 4)),
                                   plane + (first_position + row) * row_bytes + (column << 4), 16);
    }
}

/** Issues @p panel slots of every depth row of a transposed V plane from @p first_position. */
NK_HELPER_DEVICE_INLINE void nk_attention_stage_columns_ampere_(unsigned char *shared, unsigned char const *plane,
                                                                nk_size_t positions_padded, nk_size_t first_position,
                                                                unsigned panel, unsigned depth_rows) {
    unsigned const chunks_per_row = panel >> 4, chunks = depth_rows * chunks_per_row;
    unsigned const row_stride = panel + nk_attention_row_padding_ampere_k;
    for (unsigned chunk = threadIdx.x; chunk < chunks; chunk += nk_attention_threads_ampere_k) {
        unsigned const row = chunk / chunks_per_row, column = chunk - row * chunks_per_row;
        nk_copy_b128_async_ampere_(nk_shared_address_ampere_(shared + row * row_stride + (column << 4)),
                                   plane + row * positions_padded + first_position + (column << 4), 16);
    }
}

/**
 *  @brief One work item on one block: scores, online softmax and P · V over every panel its rows
 *      see, then the output.
 *  @param[in] kind How fragments are formed, see @c nk_attention_kind_t.
 *  @param[in] tier Panel width and where Q fragments live, see @c nk_attention_tier_t.
 *  @param[in] split Whether warps own rows or positions, see @c nk_attention_split_t.
 *  @param[in] epilogue F32 scores and P · V sums, or exact I32 ones converted per panel.
 *  @param[in] scores One depth step of S, see @c nk_attention_scores_ampere_t.
 *  @param[in] values_mma One 16 × 8 step of P · V on the packed P and V fragments.
 *  @param[in] weights P from probabilities, see @c nk_attention_weights_ampere_t.
 */
NK_HELPER_DEVICE_INLINE void nk_attention_block_ampere_(
    nk_attention_kind_t kind, nk_attention_tier_t tier, nk_attention_split_t split, nk_cross_epilogue_t epilogue,
    nk_attention_scores_ampere_t scores, nk_cross_mma_ampere_t values_mma, nk_attention_weights_ampere_t weights,
    nk_attention_arguments_ampere_t const *arguments, nk_attention_work_ampere_t const *work, unsigned char *shared,
    unsigned (*unions)[2]) {

    nk_f32_t const negative_infinity = nk_attention_negative_infinity_ampere_();
    unsigned const lane = threadIdx.x & 31, warp = threadIdx.x >> 5, group = lane >> 2, quad = lane & 3;
    unsigned const element_bytes = kind == nk_attention_kind_bf16_k ? 2 : 1;
    unsigned const panel = split == nk_attention_split_positions_k || tier == nk_attention_tier_narrow_k
                               ? nk_attention_panel_ampere_k
                               : nk_attention_wide_panel_ampere_k;
    unsigned const groups = split == nk_attention_split_positions_k ? 1 : panel / nk_attention_group_ampere_k;
    unsigned const max_tiles = tier == nk_attention_tier_narrow_k ? nk_attention_narrow_depth_ampere_k / 8
                                                                  : nk_attention_wide_depth_ampere_k / 8;
    unsigned const warp_row_base = split == nk_attention_split_rows_k ? warp * nk_attention_warp_rows_ampere_k : 0;
    unsigned const chunk_offset = split == nk_attention_split_positions_k ? warp * nk_attention_group_ampere_k : 0;
    unsigned const block_rows = split == nk_attention_split_rows_k ? nk_attention_block_rows_ampere_k
                                                                   : nk_attention_warp_rows_ampere_k;

    nk_size_t const depth = arguments->depth;
    unsigned const row_bytes = (unsigned)((depth * element_bytes + nk_attention_step_bytes_ampere_k - 1) &
                                          ~(nk_size_t)(nk_attention_step_bytes_ampere_k - 1));
    unsigned const row_stride = row_bytes + nk_attention_row_padding_ampere_k;
    unsigned const depth_steps = row_bytes / nk_attention_step_bytes_ampere_k;
    unsigned const depth_padded = row_bytes / element_bytes, depth_tiles = depth_padded / 8;
    unsigned const value_stride = kind == nk_attention_kind_bf16_k ? row_stride
                                                                   : panel + nk_attention_row_padding_ampere_k;
    unsigned char *keys_shared = shared + arguments->key_offset[split];
    unsigned char *values_shared = shared + arguments->value_offset[split];
    unsigned char *queries_shared = shared + arguments->query_offset[split];

    nk_size_t length, plane_bytes;
    unsigned char const *keys_plane = nk_attention_keys_plane_ampere_(arguments, work, row_bytes, &length,
                                                                      &plane_bytes);
    unsigned char const *values_plane = keys_plane + arguments->key_value_head_count * plane_bytes;
    nk_size_t const positions_padded = plane_bytes / row_bytes;

    // Each thread holds rows `group` and `group + 8` of its warp's 16.
    unsigned key_begin[2], key_end[2];
    unsigned union_begin = 0xFFFFFFFFu, union_end = 0, common_begin = 0, common_end = 0xFFFFFFFFu;
#pragma unroll
    for (unsigned half = 0; half < 2; ++half) {
        unsigned const local = warp_row_base + group + half * 8;
        key_begin[half] = key_end[half] = 0;
        if (local >= work->row_count) continue;
        nk_size_t const query = (work->row_first + local) / work->heads_selected;
        nk_attention_row_keys_ampere_((nk_i64_t)query + arguments->diagonal_offset, arguments->window, length,
                                      &key_begin[half], &key_end[half]);
        common_begin = max(common_begin, key_begin[half]), common_end = min(common_end, key_end[half]);
        if (key_begin[half] < key_end[half])
            union_begin = min(union_begin, key_begin[half]), union_end = max(union_end, key_end[half]);
    }
#pragma unroll
    for (unsigned offset = 1; offset < 32; offset <<= 1) {
        union_begin = min(union_begin, __shfl_xor_sync(0xFFFFFFFFu, union_begin, offset));
        union_end = max(union_end, __shfl_xor_sync(0xFFFFFFFFu, union_end, offset));
        common_begin = max(common_begin, __shfl_xor_sync(0xFFFFFFFFu, common_begin, offset));
        common_end = min(common_end, __shfl_xor_sync(0xFFFFFFFFu, common_end, offset));
    }
    if (lane == 0) unions[warp][0] = union_begin, unions[warp][1] = union_end;
    // Also retires the previous item's reads of every buffer this item refills.
    __syncthreads();
    unsigned block_begin = unions[0][0], block_end = unions[0][1];
#pragma unroll
    for (unsigned other = 1; other < 4; ++other)
        block_begin = min(block_begin, unions[other][0]), block_end = max(block_end, unions[other][1]);
    unsigned const panel_first = block_begin < block_end ? block_begin / panel : 0;
    unsigned const panel_end = block_begin < block_end ? (block_end + panel - 1) / panel : 0;

    if (panel_first < panel_end)
        nk_attention_stage_rows_ampere_(keys_shared, keys_plane, (nk_size_t)panel_first * panel, panel, row_bytes);
    nk_commit_async_ampere_();

    for (unsigned local = warp; local < block_rows; local += 4) {
        unsigned char *destination = queries_shared + local * row_stride;
        int const valid = local < work->row_count;
        unsigned char const *source = arguments->queries;
        if (valid) {
            nk_size_t const row = work->row_first + local;
            nk_size_t const query = row / work->heads_selected;
            nk_size_t const head = work->head_first + row % work->heads_selected;
            source += (work->query_first + query) * arguments->query_stride + head * depth * element_bytes;
        }
        if (kind == nk_attention_kind_bf16_k)
            for (unsigned element = lane; element < depth_padded; element += 32)
                ((unsigned short *)destination)[element] = valid && element < depth
                                                               ? ((unsigned short const *)source)[element]
                                                               : (unsigned short)0;
        else
            for (unsigned element = lane; element < depth_padded; element += 32)
                destination[element] = valid && element < depth ? source[element] : (unsigned char)0;
    }
    __syncthreads();

    nk_u32_t query_registers[nk_attention_narrow_depth_ampere_k * 2 / nk_attention_step_bytes_ampere_k][8];
    unsigned const query_row = warp_row_base + (lane & 7) + ((lane >> 3) & 1) * 8;
    int const queries_in_registers = tier == nk_attention_tier_narrow_k && kind != nk_attention_kind_widened_k;
    // Widened F16 fragments are twice the size, so E4M3 rereads its codes, as the wide tier does.
    if (queries_in_registers) {
#pragma unroll
        for (unsigned step = 0; step < nk_attention_narrow_depth_ampere_k * 2 / nk_attention_step_bytes_ampere_k;
             ++step)
            if (step < depth_steps) {
                nk_u32_t fragment[4];
                nk_load_matrices_x4_ampere_(
                    nk_shared_address_ampere_(queries_shared + query_row * row_stride + step * 32 + (lane >> 4) * 16),
                    fragment);
                nk_attention_query_ampere_(kind, fragment, query_registers[step]);
            }
    }

    nk_f32_t row_max[2] = {negative_infinity, negative_infinity}, row_sum[2] = {0, 0};
    nk_fui32_t output[nk_attention_wide_depth_ampere_k / 8][4];
#pragma unroll
    for (unsigned tile = 0; tile < max_tiles; ++tile)
#pragma unroll
        for (unsigned element = 0; element < 4; ++element) output[tile][element].u = 0;
    nk_f32_t const scale2 = arguments->scale2 * arguments->score_scale;

    for (unsigned panel_index = panel_first; panel_index < panel_end; ++panel_index) {
        nk_wait_async_ampere_(0);
        // K has landed, and every warp is done with the V buffer this refills.
        __syncthreads();
        nk_size_t const panel_position = (nk_size_t)panel_index * panel;
        if (kind == nk_attention_kind_bf16_k)
            nk_attention_stage_rows_ampere_(values_shared, values_plane, panel_position, panel, row_bytes);
        else
            nk_attention_stage_columns_ampere_(values_shared, values_plane, positions_padded, panel_position, panel,
                                               depth_padded);
        nk_commit_async_ampere_();

        unsigned const chunk_begin = (unsigned)panel_position + chunk_offset;
        unsigned const chunk_end = chunk_begin + groups * nk_attention_group_ampere_k;
        int const active = chunk_begin < union_end && union_begin < chunk_end;
        nk_u32_t probabilities[4][4];
        nk_f32_t correction[2] = {1.0f, 1.0f};
        if (active) {
            nk_fui32_t tile_scores[8][4];
#pragma unroll
            for (unsigned tile = 0; tile < 2 * groups; ++tile)
#pragma unroll
                for (unsigned element = 0; element < 4; ++element) tile_scores[tile][element].u = 0;
            unsigned const key_row = chunk_offset + (lane & 7) + (lane >> 4) * 8;
            unsigned const key_column = ((lane >> 3) & 1) * 16;
            if (queries_in_registers) {
#pragma unroll
                for (unsigned step = 0;
                     step < nk_attention_narrow_depth_ampere_k * 2 / nk_attention_step_bytes_ampere_k; ++step) {
                    if (step >= depth_steps) continue;
#pragma unroll
                    for (unsigned position_group = 0; position_group < groups; ++position_group) {
                        nk_u32_t keys[4];
                        nk_load_matrices_x4_ampere_(
                            nk_shared_address_ampere_(keys_shared + (key_row + position_group * 16) * row_stride +
                                                      step * 32 + key_column),
                            keys);
                        scores(tile_scores + position_group * 2, query_registers[step], keys);
                    }
                }
            }
            else {
                for (unsigned step = 0; step < depth_steps; ++step) {
                    nk_u32_t fragment[4], query[8];
                    nk_load_matrices_x4_ampere_(nk_shared_address_ampere_(queries_shared + query_row * row_stride +
                                                                          step * 32 + (lane >> 4) * 16),
                                                fragment);
                    nk_attention_query_ampere_(kind, fragment, query);
#pragma unroll
                    for (unsigned position_group = 0; position_group < groups; ++position_group) {
                        nk_u32_t keys[4];
                        nk_load_matrices_x4_ampere_(
                            nk_shared_address_ampere_(keys_shared + (key_row + position_group * 16) * row_stride +
                                                      step * 32 + key_column),
                            keys);
                        scores(tile_scores + position_group * 2, query, keys);
                    }
                }
            }

#pragma unroll
            for (unsigned tile = 0; tile < 2 * groups; ++tile)
#pragma unroll
                for (unsigned element = 0; element < 4; ++element)
                    tile_scores[tile][element].f = (epilogue == nk_cross_epilogue_i32_to_f32_k
                                                        ? (nk_f32_t)tile_scores[tile][element].i
                                                        : tile_scores[tile][element].f) *
                                                   scale2;
            // Padded tails and causal edges fall outside some row's keys; whole chunks inside every row skip this.
            if (!(common_begin <= chunk_begin && chunk_end <= common_end)) {
#pragma unroll
                for (unsigned tile = 0; tile < 2 * groups; ++tile)
#pragma unroll
                    for (unsigned element = 0; element < 4; ++element) {
                        unsigned const position = chunk_begin + tile * 8 + quad * 2 + (element & 1);
                        unsigned const half = element >> 1;
                        if (position < key_begin[half] || position >= key_end[half])
                            tile_scores[tile][element].f = negative_infinity;
                    }
            }
            nk_f32_t subtrahend[2];
#pragma unroll
            for (unsigned half = 0; half < 2; ++half) {
                nk_f32_t chunk_max = negative_infinity;
#pragma unroll
                for (unsigned tile = 0; tile < 2 * groups; ++tile)
                    chunk_max = fmaxf(chunk_max,
                                      fmaxf(tile_scores[tile][half * 2].f, tile_scores[tile][half * 2 + 1].f));
                chunk_max = fmaxf(chunk_max, __shfl_xor_sync(0xFFFFFFFFu, chunk_max, 1));
                chunk_max = fmaxf(chunk_max, __shfl_xor_sync(0xFFFFFFFFu, chunk_max, 2));
                nk_f32_t const new_max = fmaxf(row_max[half], chunk_max);
                // With every key so far masked, subtracting 0 keeps `exp2(-∞ - max)` from turning into NaN.
                subtrahend[half] = new_max == negative_infinity ? 0.0f : new_max;
                correction[half] = nk_f32_exp2_ampere_(row_max[half] - subtrahend[half]);
                row_max[half] = new_max;
                row_sum[half] *= correction[half];
            }
#pragma unroll
            for (unsigned tile = 0; tile < 2 * groups; ++tile)
#pragma unroll
                for (unsigned element = 0; element < 4; ++element)
                    tile_scores[tile][element].f = nk_f32_exp2_ampere_(tile_scores[tile][element].f -
                                                                       subtrahend[element >> 1]);
#pragma unroll
            for (unsigned position_group = 0; position_group < groups; ++position_group)
#pragma unroll
                for (unsigned half = 0; half < 2; ++half) {
                    nk_f32_t const four[4] = {tile_scores[position_group * 2][half * 2].f,
                                              tile_scores[position_group * 2][half * 2 + 1].f,
                                              tile_scores[position_group * 2 + 1][half * 2].f,
                                              tile_scores[position_group * 2 + 1][half * 2 + 1].f};
                    nk_u32_t packed[2];
                    weights(four, packed, &row_sum[half]);
                    if (kind == nk_attention_kind_bytes_k)
                        probabilities[position_group >> 1][(position_group & 1) * 2 + half] = packed[0];
                    else
                        probabilities[position_group][half] = packed[0],
                        probabilities[position_group][half + 2] = packed[1];
                }
            if (kind == nk_attention_kind_bytes_k && groups == 1) probabilities[0][2] = probabilities[0][3] = 0;
            if (epilogue == nk_cross_epilogue_f32_k) {
#pragma unroll
                for (unsigned tile = 0; tile < max_tiles; ++tile)
                    if (tile < depth_tiles)
#pragma unroll
                        for (unsigned element = 0; element < 4; ++element)
                            output[tile][element].f *= correction[element >> 1];
            }
        }

        nk_wait_async_ampere_(0);
        // V has landed, and every warp is done with the K buffer this refills.
        __syncthreads();
        if (panel_index + 1 < panel_end)
            nk_attention_stage_rows_ampere_(keys_shared, keys_plane, panel_position + panel, panel, row_bytes);
        nk_commit_async_ampere_();
        if (!active) continue;

        if (kind == nk_attention_kind_bf16_k) {
#pragma unroll
            for (unsigned pair = 0; pair < max_tiles / 2; ++pair) {
                if (pair * 2 >= depth_tiles) continue;
#pragma unroll
                for (unsigned position_group = 0; position_group < groups; ++position_group) {
                    unsigned const value_row = chunk_offset + position_group * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    nk_u32_t values[4];
                    nk_load_matrices_x4_transposed_ampere_(
                        nk_shared_address_ampere_(values_shared + value_row * value_stride + pair * 32 +
                                                  (lane >> 4) * 16),
                        values);
                    values_mma(output[pair * 2], probabilities[position_group], values[0], values[1]);
                    values_mma(output[pair * 2 + 1], probabilities[position_group], values[2], values[3]);
                }
            }
        }
        else if (kind == nk_attention_kind_widened_k || groups == 1) {
            // One 16-byte chunk per depth row: widened into both halves of a k16 step, or half of a k32 one.
#pragma unroll
            for (unsigned quartet = 0; quartet < max_tiles / 4; ++quartet) {
                if (quartet * 4 >= depth_tiles) continue;
#pragma unroll
                for (unsigned position_group = 0; position_group < groups; ++position_group) {
                    nk_u32_t values[4];
                    nk_load_matrices_x4_ampere_(
                        nk_shared_address_ampere_(values_shared + (quartet * 32 + lane) * value_stride + chunk_offset +
                                                  position_group * 16),
                        values);
#pragma unroll
                    for (unsigned index = 0; index < 4; ++index) {
                        unsigned const tile = quartet * 4 + index;
                        if (kind == nk_attention_kind_widened_k) {
                            nk_u32_t low, high;
                            nk_e4m3x4_to_f16x4_ampere_(values[index], &low, &high);
                            values_mma(output[tile], probabilities[position_group], low, high);
                        }
                        else if (epilogue == nk_cross_epilogue_f32_k)
                            values_mma(output[tile], probabilities[0], values[index], 0);
                        else {
                            nk_fui32_t sums[4] = {{0}, {0}, {0}, {0}};
                            values_mma(sums, probabilities[0], values[index], 0);
#pragma unroll
                            for (unsigned element = 0; element < 4; ++element)
                                output[tile][element].f = fmaf(output[tile][element].f, correction[element >> 1],
                                                               (nk_f32_t)sums[element].i);
                        }
                    }
                }
            }
        }
        else {
#pragma unroll
            for (unsigned pair = 0; pair < max_tiles / 2; ++pair) {
                if (pair * 2 >= depth_tiles) continue;
                unsigned const value_row = pair * 16 + (lane & 7) + (lane >> 4) * 8;
                nk_u32_t values[2][4];
#pragma unroll
                for (unsigned step = 0; step < groups / 2; ++step)
                    nk_load_matrices_x4_ampere_(nk_shared_address_ampere_(values_shared + value_row * value_stride +
                                                                          step * 32 + ((lane >> 3) & 1) * 16),
                                                values[step]);
                if (epilogue == nk_cross_epilogue_f32_k) {
#pragma unroll
                    for (unsigned step = 0; step < groups / 2; ++step) {
                        values_mma(output[pair * 2], probabilities[step], values[step][0], values[step][1]);
                        values_mma(output[pair * 2 + 1], probabilities[step], values[step][2], values[step][3]);
                    }
                    continue;
                }
                nk_fui32_t sums[2][4] = {{{0}, {0}, {0}, {0}}, {{0}, {0}, {0}, {0}}};
#pragma unroll
                for (unsigned step = 0; step < groups / 2; ++step) {
                    values_mma(sums[0], probabilities[step], values[step][0], values[step][1]);
                    values_mma(sums[1], probabilities[step], values[step][2], values[step][3]);
                }
#pragma unroll
                for (unsigned tile = 0; tile < 2; ++tile)
#pragma unroll
                    for (unsigned element = 0; element < 4; ++element)
                        output[pair * 2 + tile][element].f = fmaf(output[pair * 2 + tile][element].f,
                                                                  correction[element >> 1],
                                                                  (nk_f32_t)sums[tile][element].i);
            }
        }
    }

#pragma unroll
    for (unsigned half = 0; half < 2; ++half) {
        row_sum[half] += __shfl_xor_sync(0xFFFFFFFFu, row_sum[half], 1);
        row_sum[half] += __shfl_xor_sync(0xFFFFFFFFu, row_sum[half], 2);
    }

    if (split == nk_attention_split_positions_k) {
        // Warps 1-3 hand their states to warp 0 through the retired panel buffers.
        nk_f32_t *combined = (nk_f32_t *)shared;
        nk_f32_t *statistics = combined + 3 * depth_tiles * 128;
        __syncthreads();
        if (warp != 0) {
#pragma unroll
            for (unsigned tile = 0; tile < max_tiles; ++tile)
                if (tile < depth_tiles)
#pragma unroll
                    for (unsigned element = 0; element < 4; ++element)
                        combined[(((warp - 1) * depth_tiles + tile) * 4 + element) * 32 + lane] =
                            output[tile][element].f;
#pragma unroll
            for (unsigned half = 0; half < 2; ++half) {
                statistics[(((warp - 1) * 2 + half) * 2 + 0) * 32 + lane] = row_max[half];
                statistics[(((warp - 1) * 2 + half) * 2 + 1) * 32 + lane] = row_sum[half];
            }
        }
        __syncthreads();
        if (warp != 0) return;
        nk_f32_t merged_max[2] = {row_max[0], row_max[1]};
#pragma unroll
        for (unsigned other = 0; other < 3; ++other)
#pragma unroll
            for (unsigned half = 0; half < 2; ++half)
                merged_max[half] = fmaxf(merged_max[half], statistics[((other * 2 + half) * 2 + 0) * 32 + lane]);
        nk_f32_t subtrahend[2], own_factor[2];
#pragma unroll
        for (unsigned half = 0; half < 2; ++half) {
            subtrahend[half] = merged_max[half] == negative_infinity ? 0.0f : merged_max[half];
            own_factor[half] = nk_f32_exp2_ampere_(row_max[half] - subtrahend[half]);
            row_sum[half] *= own_factor[half];
        }
#pragma unroll
        for (unsigned tile = 0; tile < max_tiles; ++tile)
#pragma unroll
            for (unsigned element = 0; element < 4; ++element) output[tile][element].f *= own_factor[element >> 1];
#pragma unroll
        for (unsigned other = 0; other < 3; ++other) {
            nk_f32_t factor[2];
#pragma unroll
            for (unsigned half = 0; half < 2; ++half) {
                factor[half] = nk_f32_exp2_ampere_(statistics[((other * 2 + half) * 2 + 0) * 32 + lane] -
                                                   subtrahend[half]);
                row_sum[half] += statistics[((other * 2 + half) * 2 + 1) * 32 + lane] * factor[half];
            }
#pragma unroll
            for (unsigned tile = 0; tile < max_tiles; ++tile)
                if (tile < depth_tiles)
#pragma unroll
                    for (unsigned element = 0; element < 4; ++element)
                        output[tile][element].f = fmaf(
                            combined[((other * depth_tiles + tile) * 4 + element) * 32 + lane], factor[element >> 1],
                            output[tile][element].f);
        }
    }

#pragma unroll
    for (unsigned half = 0; half < 2; ++half) {
        unsigned const local = warp_row_base + group + half * 8;
        if (local >= work->row_count) continue;
        nk_size_t const row = work->row_first + local;
        nk_size_t const query = row / work->heads_selected;
        nk_size_t const head = work->head_first + row % work->heads_selected;
        nk_f32_t *destination = (nk_f32_t *)((unsigned char *)arguments->output +
                                             (work->query_first + query) * arguments->output_stride) +
                                head * depth;
        nk_f32_t const inverse = row_sum[half] > 0 ? arguments->output_scale / row_sum[half] : 0.0f;
#pragma unroll
        for (unsigned tile = 0; tile < max_tiles; ++tile)
#pragma unroll
            for (unsigned element = 0; element < 2; ++element) {
                unsigned const column = tile * 8 + quad * 2 + element;
                if (column < depth) destination[column] = output[tile][half * 2 + element].f * inverse;
            }
    }
}

/**
 *  @brief Every work item of a launch, walked with a stride of the grid, each on the split its row
 *      count calls for.
 *  @sa nk_attention_block_ampere_ for the parameters.
 */
NK_HELPER_DEVICE_INLINE void nk_attention_tile_ampere_(nk_attention_kind_t kind, nk_attention_tier_t tier,
                                                       nk_cross_epilogue_t epilogue,
                                                       nk_attention_scores_ampere_t scores,
                                                       nk_cross_mma_ampere_t values_mma,
                                                       nk_attention_weights_ampere_t weights,
                                                       nk_attention_arguments_ampere_t const *arguments) {
    extern __shared__ __align__(128) unsigned char nk_attention_shared_ampere_[];
    __shared__ nk_u64_t prefix[nk_attention_threads_ampere_k + 1];
    __shared__ nk_u64_t warp_totals[nk_attention_threads_ampere_k / 32];
    __shared__ unsigned unions[nk_attention_threads_ampere_k / 32][2];
    nk_attention_schedule_ampere_t schedule;
    if (!nk_attention_schedule_start_ampere_(arguments, &schedule, prefix, warp_totals)) return;
    nk_attention_work_ampere_t work;
    for (nk_size_t item = blockIdx.x; nk_attention_schedule_next_ampere_(&schedule, prefix, warp_totals, item, &work);
         item += gridDim.x) {
        if (work.row_count <= nk_attention_warp_rows_ampere_k)
            nk_attention_block_ampere_(kind, tier, nk_attention_split_positions_k, epilogue, scores, values_mma,
                                       weights, arguments, &work, nk_attention_shared_ampere_, unions);
        else
            nk_attention_block_ampere_(kind, tier, nk_attention_split_rows_k, epilogue, scores, values_mma, weights,
                                       arguments, &work, nk_attention_shared_ampere_, unions);
    }
}

/** One element of a K row or V element, decoded to F32; I8 stays an exact integer in F32. */
NK_HELPER_DEVICE_INLINE nk_f32_t nk_attention_decode_ampere_(nk_dtype_t dtype, unsigned char const *bytes,
                                                             nk_size_t index) {
    if (dtype == nk_bf16_k) return __uint_as_float((nk_u32_t)((unsigned short const *)bytes)[index] << 16);
    if (dtype == nk_i8_k) return (nk_f32_t)(signed char)bytes[index];
    unsigned const code = bytes[index];
    return __half2float(__ushort_as_half((unsigned short)(((code & 0x7Fu) << 7) | ((code & 0x80u) << 8)))) * 256.0f;
}

/** The CUDA-core kernel for depths past 256: one warp per row, two sweeps over its keys like the
 *  serial backend, the output row doubling as the accumulator, so any depth fits. */
NK_HELPER_DEVICE_INLINE void nk_attention_fallback_ampere_(nk_dtype_t dtype,
                                                           nk_attention_arguments_ampere_t const *arguments) {
    __shared__ nk_u64_t prefix[nk_attention_threads_ampere_k + 1];
    __shared__ nk_u64_t warp_totals[nk_attention_threads_ampere_k / 32];
    unsigned const lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    nk_size_t const element_bytes = dtype == nk_bf16_k ? 2 : 1, depth = arguments->depth;
    nk_size_t const row_bytes = (depth * element_bytes + nk_attention_step_bytes_ampere_k - 1) &
                                ~(nk_size_t)(nk_attention_step_bytes_ampere_k - 1);
    nk_attention_schedule_ampere_t schedule;
    if (!nk_attention_schedule_start_ampere_(arguments, &schedule, prefix, warp_totals)) return;
    nk_attention_work_ampere_t work;
    for (nk_size_t item = blockIdx.x; nk_attention_schedule_next_ampere_(&schedule, prefix, warp_totals, item, &work);
         item += gridDim.x) {
        nk_size_t length, plane_bytes;
        unsigned char const *keys_plane = nk_attention_keys_plane_ampere_(arguments, &work, row_bytes, &length,
                                                                          &plane_bytes);
        unsigned char const *values_plane = keys_plane + arguments->key_value_head_count * plane_bytes;
        nk_size_t const positions_padded = plane_bytes / row_bytes;
        for (nk_size_t local = warp; local < work.row_count; local += 4) {
            nk_size_t const row = work.row_first + local;
            nk_size_t const query = row / work.heads_selected;
            nk_size_t const head = work.head_first + row % work.heads_selected;
            unsigned char const *query_row = arguments->queries + (work.query_first + query) * arguments->query_stride +
                                             head * depth * element_bytes;
            nk_f32_t *output_row = (nk_f32_t *)((unsigned char *)arguments->output +
                                                (work.query_first + query) * arguments->output_stride) +
                                   head * depth;
            unsigned key_begin, key_end;
            nk_attention_row_keys_ampere_((nk_i64_t)query + arguments->diagonal_offset, arguments->window, length,
                                          &key_begin, &key_end);
            nk_f32_t row_max = nk_attention_negative_infinity_ampere_();
            for (unsigned sweep = 0; sweep < 2; ++sweep) {
                nk_f32_t weights_sum = 0;
                for (unsigned position = key_begin; position < key_end; ++position) {
                    unsigned char const *key_row = keys_plane + position * row_bytes;
                    nk_f32_t partial = 0;
                    for (nk_size_t element = lane; element < depth; element += 32)
                        partial = fmaf(nk_attention_decode_ampere_(dtype, query_row, element),
                                       nk_attention_decode_ampere_(dtype, key_row, element), partial);
#pragma unroll
                    for (unsigned offset = 16; offset != 0; offset >>= 1)
                        partial += __shfl_xor_sync(0xFFFFFFFFu, partial, offset);
                    nk_f32_t const scaled = partial * arguments->scale2;
                    if (sweep == 0) {
                        row_max = fmaxf(row_max, scaled);
                        continue;
                    }
                    nk_f32_t weight = nk_f32_exp2_ampere_(scaled - row_max);
                    if (dtype == nk_i8_k) weight = (nk_f32_t)(nk_u32_t)(weight * 255.0f + 0.5f);
                    weights_sum += weight;
                    for (nk_size_t element = lane; element < depth; element += 32) {
                        nk_f32_t const value =
                            dtype == nk_bf16_k
                                ? nk_attention_decode_ampere_(dtype, values_plane + position * row_bytes, element)
                                : nk_attention_decode_ampere_(dtype, values_plane + element * positions_padded,
                                                              nk_attention_slot_ampere_(position));
                        output_row[element] = fmaf(weight, value, output_row[element]);
                    }
                }
                if (sweep == 0)
                    for (nk_size_t element = lane; element < depth; element += 32) output_row[element] = 0;
                else {
                    nk_f32_t const inverse = weights_sum > 0 ? 1.0f / weights_sum : 0.0f;
                    for (nk_size_t element = lane; element < depth; element += 32) output_row[element] *= inverse;
                }
            }
        }
    }
}

#pragma endregion Tile

#pragma region Pack

/** Writes the header and the directory, one block walking the segments 128 at a time. */
static __global__ void nk_attention_pack_directory_ampere_kernel_(unsigned char *packed, nk_size_t key_value_head_count,
                                                                  nk_size_t depth, nk_u32_t const *segment_lengths,
                                                                  nk_size_t segment_count, nk_size_t row_bytes,
                                                                  nk_size_t directory_bytes) {
    __shared__ nk_u64_t warp_totals[nk_attention_threads_ampere_k / 32];
    nk_attention_packed_header_t *header = (nk_attention_packed_header_t *)packed;
    if (threadIdx.x == 0) {
        header->heads = (nk_u32_t)key_value_head_count;
        header->depth = (nk_u32_t)depth;
        header->segments = (nk_u32_t)segment_count;
        for (unsigned reserved_index = 0; reserved_index < 13; ++reserved_index) header->reserved[reserved_index] = 0;
    }
    nk_u64_t *payload_offsets = (nk_u64_t *)(packed + sizeof(nk_attention_packed_header_t));
    nk_u32_t *lengths = (nk_u32_t *)(payload_offsets + segment_count + 1);
    nk_u64_t running = 0;
    for (nk_size_t chunk = 0; chunk < segment_count; chunk += blockDim.x) {
        nk_size_t const segment = chunk + threadIdx.x;
        nk_u32_t const length = segment < segment_count ? segment_lengths[segment] : 0;
        nk_u64_t const bytes = 2 * key_value_head_count *
                               (((nk_u64_t)length + nk_attention_panel_ampere_k - 1) &
                                ~(nk_u64_t)(nk_attention_panel_ampere_k - 1)) *
                               row_bytes;
        nk_u64_t total;
        nk_u64_t const inclusive = nk_attention_block_scan_ampere_(bytes, warp_totals, &total);
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
NK_HELPER_DEVICE_INLINE void nk_attention_pack_payload_ampere_(
    nk_attention_kind_t kind, unsigned char const *keys, unsigned char const *values, nk_size_t key_value_head_count,
    nk_size_t depth, nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, nk_size_t segment_count,
    nk_size_t key_stride, nk_size_t value_stride, unsigned char *packed, nk_size_t task_begin, nk_size_t task_end) {
    nk_size_t const element_bytes = kind == nk_attention_kind_bf16_k ? 2 : 1;
    nk_size_t const row_bytes = (depth * element_bytes + nk_attention_step_bytes_ampere_k - 1) &
                                ~(nk_size_t)(nk_attention_step_bytes_ampere_k - 1);
    nk_size_t const row_elements = row_bytes / element_bytes;
    nk_u64_t const *payload_offsets = (nk_u64_t const *)(packed + sizeof(nk_attention_packed_header_t));
    nk_size_t const directory_bytes = ((segment_count + 1) * sizeof(nk_u64_t) + segment_count * sizeof(nk_u32_t) + 63) &
                                      ~63ull;
    unsigned char *payload = packed + sizeof(nk_attention_packed_header_t) + directory_bytes;
    for (nk_size_t task = task_begin + blockIdx.x; task < task_end; task += gridDim.x) {
        nk_size_t const segment = task / key_value_head_count, head = task % key_value_head_count;
        nk_size_t const length = segment_lengths[segment];
        if (length == 0) continue;
        nk_size_t const positions_padded = (length + nk_attention_panel_ampere_k - 1) &
                                           ~(nk_size_t)(nk_attention_panel_ampere_k - 1);
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
            nk_size_t const position = nk_attention_slot_position_ampere_(index % positions_padded);
            values_plane[index] = position < length && element < depth ? values_first[position * value_stride + element]
                                                                       : (unsigned char)0;
        }
    }
}

/** Mirrors the device pack: header, directory, then both planes of every segment and head. */
NK_HELPER_INLINE nk_size_t nk_attention_pack_size_ampere_(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                          nk_size_t element_bytes) {
    nk_size_t const row_bytes = nk_size_round_up_to_multiple_(depth * element_bytes, nk_attention_step_bytes_ampere_k);
    nk_size_t payload_bytes = 0;
    for (nk_size_t segment = 0; segment < segment_count; ++segment)
        payload_bytes += 2 * key_value_head_count *
                         nk_size_round_up_to_multiple_(segment_lengths[segment], nk_attention_panel_ampere_k) *
                         row_bytes;
    return sizeof(nk_attention_packed_header_t) + nk_attention_pack_directory_size_(segment_count) + payload_bytes;
}

/** Launches the directory writer for a window starting at task 0, then @p payload_kernel
 *  over the window. */
NK_HELPER_INLINE cudaError_t nk_attention_pack_launch_ampere_(
    void const *payload_kernel, nk_size_t element_bytes, void const *keys, void const *values,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
    nk_size_t segment_count, nk_size_t key_stride, nk_size_t value_stride, void *packed, nk_size_t task_begin,
    nk_size_t task_end, cudaStream_t stream) {
    if ((nk_size_t)packed & 15) return cudaErrorMisalignedAddress;
    nk_size_t const row_bytes = nk_size_round_up_to_multiple_(depth * element_bytes, nk_attention_step_bytes_ampere_k);
    nk_size_t const directory_bytes = nk_attention_pack_directory_size_(segment_count);
    dim3 grid, block;
    grid.y = grid.z = block.y = block.z = 1;
    cudaError_t status = cudaSuccess;
    if (task_begin == 0) {
        grid.x = 1, block.x = nk_attention_threads_ampere_k;
        void *directory_arguments[7] = {&packed,
                                        &key_value_head_count,
                                        &depth,
                                        &segment_lengths,
                                        &segment_count,
                                        (void *)&row_bytes,
                                        (void *)&directory_bytes};
        status = cudaLaunchKernel((void const *)nk_attention_pack_directory_ampere_kernel_, grid, block,
                                  directory_arguments, 0, stream);
        if (status != cudaSuccess) return status;
    }
    nk_size_t const total_tasks = segment_count * key_value_head_count;
    nk_size_t const end = task_end < total_tasks ? task_end : total_tasks;
    if (task_begin >= end) return status;
    grid.x = (unsigned)(end - task_begin < 65535 ? end - task_begin : 65535);
    block.x = nk_attention_pack_threads_ampere_k;
    void *payload_arguments[12] = {(void *)&keys,    (void *)&values,  &key_value_head_count, &depth,
                                   &segment_offsets, &segment_lengths, &segment_count,        &key_stride,
                                   &value_stride,    &packed,          &task_begin,           (void *)&end};
    return cudaLaunchKernel(payload_kernel, grid, block, payload_arguments, 0, stream);
}

#pragma endregion Pack

#pragma region Launch

/** Places the panel buffers of a tier in dynamic shared memory, returning the bytes
 *  a block needs. */
NK_HELPER_INLINE nk_size_t nk_attention_shared_layout_ampere_(nk_attention_kind_t kind, nk_attention_tier_t tier,
                                                              nk_size_t depth,
                                                              nk_attention_arguments_ampere_t *arguments) {
    nk_size_t const element_bytes = kind == nk_attention_kind_bf16_k ? 2 : 1;
    nk_size_t const row_bytes = nk_size_round_up_to_multiple_(depth * element_bytes, nk_attention_step_bytes_ampere_k);
    nk_size_t const row_stride = row_bytes + nk_attention_row_padding_ampere_k,
                    depth_padded = row_bytes / element_bytes;
    nk_size_t const keys_narrow = nk_attention_panel_ampere_k * row_stride;
    nk_size_t const keys_wide = nk_attention_wide_panel_ampere_k * row_stride;
    nk_size_t const values_narrow = kind == nk_attention_kind_bf16_k
                                        ? keys_narrow
                                        : depth_padded *
                                              (nk_attention_panel_ampere_k + nk_attention_row_padding_ampere_k);
    nk_size_t const values_wide = kind == nk_attention_kind_bf16_k ? keys_wide
                                                                   : depth_padded * (nk_attention_wide_panel_ampere_k +
                                                                                     nk_attention_row_padding_ampere_k);
    nk_size_t const queries_block = nk_attention_block_rows_ampere_k * row_stride;
    nk_size_t const queries_warp = nk_attention_warp_rows_ampere_k * row_stride;
    nk_size_t const combine = 3 * (depth_padded / 8) * 4 * 32 * sizeof(nk_f32_t) + 3 * 2 * 2 * 32 * sizeof(nk_f32_t);
    nk_size_t pipeline;
    arguments->key_offset[nk_attention_split_rows_k] = arguments->key_offset[nk_attention_split_positions_k] = 0;
    if (tier == nk_attention_tier_narrow_k && kind != nk_attention_kind_widened_k) {
        // Q is staged in the V buffer, read into registers before the first V lands.
        arguments->value_offset[nk_attention_split_rows_k] = (nk_u32_t)keys_narrow;
        arguments->value_offset[nk_attention_split_positions_k] = (nk_u32_t)keys_narrow;
        arguments->query_offset[nk_attention_split_rows_k] = (nk_u32_t)keys_narrow;
        arguments->query_offset[nk_attention_split_positions_k] = (nk_u32_t)keys_narrow;
        pipeline = keys_narrow + (values_narrow > queries_block ? values_narrow : queries_block);
    }
    else if (tier == nk_attention_tier_narrow_k) {
        arguments->value_offset[nk_attention_split_rows_k] = (nk_u32_t)keys_narrow;
        arguments->value_offset[nk_attention_split_positions_k] = (nk_u32_t)keys_narrow;
        arguments->query_offset[nk_attention_split_rows_k] = (nk_u32_t)(keys_narrow + values_narrow);
        arguments->query_offset[nk_attention_split_positions_k] = (nk_u32_t)(keys_narrow + values_narrow);
        pipeline = keys_narrow + values_narrow + queries_block;
    }
    else {
        arguments->value_offset[nk_attention_split_rows_k] = (nk_u32_t)keys_wide;
        arguments->query_offset[nk_attention_split_rows_k] = (nk_u32_t)(keys_wide + values_wide);
        arguments->value_offset[nk_attention_split_positions_k] = (nk_u32_t)keys_narrow;
        arguments->query_offset[nk_attention_split_positions_k] = (nk_u32_t)(keys_narrow + values_narrow);
        nk_size_t const rows_bytes = keys_wide + values_wide + queries_block;
        nk_size_t const positions_bytes = keys_narrow + values_narrow + queries_warp;
        pipeline = rows_bytes > positions_bytes ? rows_bytes : positions_bytes;
    }
    return pipeline > combine ? pipeline : combine;
}

/**
 *  @brief Validates the contract and launches the kernel for the depth's tier with as many blocks
 *      as stay resident.
 *  @param[in] score_scale Undoes the Q and K widenings in the tile, or 1.
 *  @param[in] output_scale Undoes the V widening in the tile, or 1.
 */
NK_HELPER_INLINE cudaError_t nk_attention_launch_ampere_(
    void const *narrow_kernel, void const *wide_kernel, void const *fallback_kernel, nk_attention_kind_t kind,
    void const *queries, void const *packed, nk_f32_t *output, nk_size_t head_count, nk_size_t key_value_head_count,
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale,
    nk_f32_t score_scale, nk_f32_t output_scale, nk_attention_mask_t mask, nk_i64_t diagonal_offset, nk_size_t window,
    nk_size_t task_start, nk_size_t task_count, cudaStream_t stream) {
    if (((nk_size_t)packed & 15) || (((nk_size_t)output | output_stride) & 3)) return cudaErrorMisalignedAddress;
    if (key_value_head_count == 0 || head_count % key_value_head_count != 0) return cudaErrorInvalidValue;
    if (task_count == 0 || depth == 0) return cudaSuccess;
    nk_attention_arguments_ampere_t arguments;
    arguments.queries = (unsigned char const *)queries, arguments.packed = (unsigned char const *)packed;
    arguments.output = output, arguments.query_offsets = query_offsets;
    arguments.head_count = head_count, arguments.key_value_head_count = key_value_head_count;
    arguments.depth = depth, arguments.query_stride = query_stride, arguments.output_stride = output_stride;
    arguments.diagonal_offset = mask == nk_attention_mask_causal_k ? diagonal_offset : NK_I64_MAX / 2;
    arguments.window = mask == nk_attention_mask_causal_k ? window : NK_SIZE_MAX;
    arguments.task_start = task_start, arguments.task_count = task_count;
    arguments.scale2 = scale * NK_F32_LOG2E_;
    arguments.score_scale = score_scale, arguments.output_scale = output_scale;

    void const *kernel = fallback_kernel;
    nk_size_t shared_bytes = 0;
    if (depth <= nk_attention_narrow_depth_ampere_k)
        kernel = narrow_kernel,
        shared_bytes = nk_attention_shared_layout_ampere_(kind, nk_attention_tier_narrow_k, depth, &arguments);
    else if (depth <= nk_attention_wide_depth_ampere_k)
        kernel = wide_kernel,
        shared_bytes = nk_attention_shared_layout_ampere_(kind, nk_attention_tier_wide_k, depth, &arguments);

    int device = 0, multiprocessors = 0, resident_per_multiprocessor = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status == cudaSuccess)
        status = cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device);
    if (status == cudaSuccess && shared_bytes)
        status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared_bytes);
    if (status == cudaSuccess)
        status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident_per_multiprocessor, kernel,
                                                               nk_attention_threads_ampere_k, shared_bytes);
    if (status != cudaSuccess) return status;
    dim3 grid, block;
    grid.x = (unsigned)(multiprocessors * (resident_per_multiprocessor > 0 ? resident_per_multiprocessor : 1));
    grid.y = grid.z = 1;
    block.x = nk_attention_threads_ampere_k, block.y = block.z = 1;
    void *launch_arguments[1];
    launch_arguments[0] = &arguments;
    return cudaLaunchKernel(kernel, grid, block, launch_arguments, shared_bytes, stream);
}

#pragma endregion Launch

#pragma region Attention Macros

/** Generates the host-side size of a pack: header, directory, and both planes of every
 *  segment and head. */
#define nk_define_attention_cuda_pack_size_(input_type_name, isa_suffix, element_bytes)                              \
    NK_API_COMPTIME nk_size_t nk_attention_pack_size_##input_type_name##_##isa_suffix(                               \
        nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *segment_lengths, nk_size_t segment_count) { \
        return nk_attention_pack_size_ampere_(key_value_head_count, depth, segment_lengths, segment_count,           \
                                              element_bytes);                                                        \
    }

/** Generates a shape accessor copying a device-resident pack's header back. */
#define nk_define_attention_cuda_packed_shape_(input_type_name, isa_suffix)                                           \
    NK_API_COMPTIME cudaError_t nk_attention_packed_shape_##input_type_name##_##isa_suffix(                           \
        void const *key_value_packed, nk_size_t *heads, nk_size_t *depth, nk_size_t *segments, cudaStream_t stream) { \
        nk_attention_packed_header_t header;                                                                          \
        cudaError_t status = cudaMemcpyAsync(&header, key_value_packed, sizeof(header), cudaMemcpyDeviceToHost,       \
                                             stream);                                                                 \
        if (status == cudaSuccess) status = cudaStreamSynchronize(stream);                                            \
        if (status != cudaSuccess) return status;                                                                     \
        *heads = header.heads, *depth = header.depth, *segments = header.segments;                                    \
        return cudaSuccess;                                                                                           \
    }

/**
 *  @brief Generates a device pack: the directory when the window starts at task 0, then one
 *      block per task.
 *  @param[in] kind Selects the V layout: position rows for BF16, σ-ordered depth rows
 *      for 1-byte codes.
 */
#define nk_define_attention_cuda_pack_(input_type_name, isa_suffix, input_value_type, kind)                      \
    static __global__ void nk_attention_pack_##input_type_name##_##isa_suffix##_kernel_(                         \
        unsigned char const *keys, unsigned char const *values, nk_size_t key_value_head_count, nk_size_t depth, \
        nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, nk_size_t segment_count,               \
        nk_size_t key_stride, nk_size_t value_stride, unsigned char *packed, nk_size_t task_begin,               \
        nk_size_t task_end) {                                                                                    \
        nk_attention_pack_payload_ampere_(kind, keys, values, key_value_head_count, depth, segment_offsets,      \
                                          segment_lengths, segment_count, key_stride, value_stride, packed,      \
                                          task_begin, task_end);                                                 \
    }                                                                                                            \
    NK_API_COMPTIME cudaError_t nk_attention_pack_##input_type_name##_##isa_suffix(                              \
        nk_##input_value_type##_t const *keys, nk_##input_value_type##_t const *values,                          \
        nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *segment_offsets,                        \
        nk_u32_t const *segment_lengths, nk_size_t segment_count, nk_size_t key_stride_bytes,                    \
        nk_size_t value_stride_bytes, void *key_value_packed, nk_size_t task_begin, nk_size_t task_end,          \
        cudaStream_t stream) {                                                                                   \
        return nk_attention_pack_launch_ampere_(                                                                 \
            (void const *)nk_attention_pack_##input_type_name##_##isa_suffix##_kernel_,                          \
            sizeof(nk_##input_value_type##_t), keys, values, key_value_head_count, depth, segment_offsets,       \
            segment_lengths, segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, task_begin,  \
            task_end, stream);                                                                                   \
    }

/**
 *  @brief Generates the narrow, wide and fallback kernels of one dtype and both public attention
 *      entry points, each passing its @c nk_attention_mask_t to the shared launch.
 *  @param[in] score_scale Undoes the power of two the tile's score widening introduces, or 1.
 *  @param[in] output_scale Undoes the power of two the tile's value widening introduces, or 1.
 *  @param[in] fallback_dtype How the CUDA-core kernel decodes the pack past depth 256.
 */
#define nk_define_attention_cuda_packed_(input_type_name, isa_suffix, input_value_type, kind, epilogue, scores_fn,  \
                                         values_mma_fn, weights_fn, score_scale, output_scale, fallback_dtype)      \
    static __global__ void __launch_bounds__(nk_attention_threads_ampere_k)                                         \
        nk_attention_packed_##input_type_name##_##isa_suffix##_narrow_kernel_(                                      \
            nk_attention_arguments_ampere_t arguments) {                                                            \
        nk_attention_tile_ampere_(kind, nk_attention_tier_narrow_k, epilogue, scores_fn, values_mma_fn, weights_fn, \
                                  &arguments);                                                                      \
    }                                                                                                               \
    static __global__ void __launch_bounds__(nk_attention_threads_ampere_k)                                         \
        nk_attention_packed_##input_type_name##_##isa_suffix##_wide_kernel_(                                        \
            nk_attention_arguments_ampere_t arguments) {                                                            \
        nk_attention_tile_ampere_(kind, nk_attention_tier_wide_k, epilogue, scores_fn, values_mma_fn, weights_fn,   \
                                  &arguments);                                                                      \
    }                                                                                                               \
    static __global__ void __launch_bounds__(nk_attention_threads_ampere_k)                                         \
        nk_attention_packed_##input_type_name##_##isa_suffix##_fallback_kernel_(                                    \
            nk_attention_arguments_ampere_t arguments) {                                                            \
        nk_attention_fallback_ampere_(fallback_dtype, &arguments);                                                  \
    }                                                                                                               \
    NK_API_COMPTIME cudaError_t nk_attention_bidirectional_packed_##input_type_name##_##isa_suffix(                 \
        nk_##input_value_type##_t const *queries, void const *key_value_packed, nk_f32_t *output,                   \
        nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,       \
        nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start,          \
        nk_size_t task_count, cudaStream_t stream) {                                                                \
        return nk_attention_launch_ampere_(                                                                         \
            (void const *)nk_attention_packed_##input_type_name##_##isa_suffix##_narrow_kernel_,                    \
            (void const *)nk_attention_packed_##input_type_name##_##isa_suffix##_wide_kernel_,                      \
            (void const *)nk_attention_packed_##input_type_name##_##isa_suffix##_fallback_kernel_, kind, queries,   \
            key_value_packed, output, head_count, key_value_head_count, depth, query_offsets, query_stride_bytes,   \
            output_stride_bytes, scale, score_scale, output_scale, nk_attention_mask_bidirectional_k, 0, 0,         \
            task_start, task_count, stream);                                                                        \
    }                                                                                                               \
    NK_API_COMPTIME cudaError_t nk_attention_causal_packed_##input_type_name##_##isa_suffix(                        \
        nk_##input_value_type##_t const *queries, void const *key_value_packed, nk_f32_t *output,                   \
        nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,       \
        nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset,      \
        nk_size_t window, nk_size_t task_start, nk_size_t task_count, cudaStream_t stream) {                        \
        return nk_attention_launch_ampere_(                                                                         \
            (void const *)nk_attention_packed_##input_type_name##_##isa_suffix##_narrow_kernel_,                    \
            (void const *)nk_attention_packed_##input_type_name##_##isa_suffix##_wide_kernel_,                      \
            (void const *)nk_attention_packed_##input_type_name##_##isa_suffix##_fallback_kernel_, kind, queries,   \
            key_value_packed, output, head_count, key_value_head_count, depth, query_offsets, query_stride_bytes,   \
            output_stride_bytes, scale, score_scale, output_scale, nk_attention_mask_causal_k, diagonal_offset,     \
            window, task_start, task_count, stream);                                                                \
    }

#pragma endregion Attention Macros

#pragma region Instantiations

nk_define_attention_cuda_pack_size_(bf16, ampere, 2)
nk_define_attention_cuda_packed_shape_(bf16, ampere)
nk_define_attention_cuda_pack_(bf16, ampere, bf16, nk_attention_kind_bf16_k)
nk_define_attention_cuda_packed_(bf16, ampere, bf16, nk_attention_kind_bf16_k, nk_cross_epilogue_f32_k,
                                 nk_attention_scores_bf16_ampere_, nk_mma_bf16_ampere_,
                                 nk_attention_weights_bf16_ampere_, 1.0f, 1.0f, nk_bf16_k)

nk_define_attention_cuda_pack_size_(e4m3, ampere, 1)
nk_define_attention_cuda_packed_shape_(e4m3, ampere)
nk_define_attention_cuda_pack_(e4m3, ampere, e4m3, nk_attention_kind_widened_k)
nk_define_attention_cuda_packed_(e4m3, ampere, e4m3, nk_attention_kind_widened_k, nk_cross_epilogue_f32_k,
                                 nk_attention_scores_e4m3_ampere_, nk_mma_f16_ampere_, nk_attention_weights_f16_ampere_,
                                 65536.0f, 256.0f, nk_e4m3_k)

nk_define_attention_cuda_pack_size_(i8, ampere, 1)
nk_define_attention_cuda_packed_shape_(i8, ampere)
nk_define_attention_cuda_pack_(i8, ampere, i8, nk_attention_kind_bytes_k)
nk_define_attention_cuda_packed_(i8, ampere, i8, nk_attention_kind_bytes_k, nk_cross_epilogue_i32_to_f32_k,
                                 nk_attention_scores_i8_ampere_, nk_mma_u8i8_ampere_, nk_attention_weights_u8_ampere_,
                                 1.0f, 1.0f, nk_i8_k)

#pragma endregion Instantiations

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_AMPERE
#endif // NK_ATTENTION_AMPERE_CUH
