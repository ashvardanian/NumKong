/**
 *  @file include/numkong/attention/simt.cuh
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Ragged attention and rotary embeddings on the SIMT cores of every CUDA and ROCm device.
 *
 *  @sa include/numkong/attention.h
 *  @sa include/numkong/attention/serial.h
 *
 *  The portable parts of the pack and the work scheduler, which every kernel of both vendors
 *  shares. The pack keeps the serial header and directory, with 1-byte V transposed and permuted
 *  within every 16 positions so the tensor-core kernels form MMA fragments without shuffles. Only
 *  @c pack_size reads no segment arrays on the host, so the pack needs device or managed memory for
 *  the key offsets and lengths, and the kernels read the directory the pack stored.
 *
 *  Everything that scans or reduces across lanes lives in the vendor baselines, `cuda.cuh` and
 *  `rocm.cuh` beside this file: the scheduler's block scan, the directory kernel, the backward
 *  loops, and the fallback kernel, a warp or wavefront per query row making the serial kernel's
 *  two sweeps over its keys, with products in F32 and I8 scores in exact I32 like the serial ones.
 */
#ifndef NUMKONG_ATTENTION_SIMT_CUH
#define NUMKONG_ATTENTION_SIMT_CUH

#if NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_

#include "numkong/simt.cuh"           // `nk_diagonal_band_tile_coverage_simt_`
#include "numkong/attention/serial.h" // `nk_attention_packed_header_t`, `nk_attention_pack_directory_size_serial_`

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

/** Which keys a kernel lets each query row see: its whole segment, or the launch's diagonal band
 *  around the row's position. */
typedef enum {
    nk_attention_mask_none_k,
    nk_attention_mask_diagonal_band_k,
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

    /** Natural log-sum-exp per query token and head, [query tokens, heads], or null. */
    nk_f32_t *log_sum_exp;

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

    /** Keys each query row sees around its position. */
    nk_diagonal_band_t band;

    /** First task of the window over query tokens × heads. */
    nk_size_t tasks_begin;

    /** One past the last task of the window, clipped on the device. */
    nk_size_t tasks_end;

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

    /** First task of the window, clipped to the grid. */
    nk_size_t tasks_begin;

    /** One past the last task, clipped to the grid. */
    nk_size_t tasks_end;

    /** One past the last segment the window touches. */
    nk_size_t segment_end;

    /** First segment of the chunk whose prefix sits in shared memory. */
    nk_size_t chunk_first;

    /** Items in the segments before the chunk. */
    nk_size_t items_before;
} nk_attention_schedule_t;

/** One work item: a block of rows, each row being query × heads_selected + head, of one segment
 *  against one K and V head, every query head of the group selected. */
typedef struct {

    /** Segment index. */
    nk_size_t segment;

    /** K and V head. */
    nk_size_t key_value_head;

    /** First query head, counted within the segment. */
    nk_size_t head_first;

    /** Query heads the rows cycle through. */
    nk_size_t heads_selected;

    /** First row of the item: where its block of the segment's rows starts, or the window does. */
    nk_size_t row_first;

    /** Rows of the item. */
    nk_size_t row_count;

    /** Query token of the segment's first query. */
    nk_size_t query_first;

    /** Queries of the segment, which align to the end of its keys. */
    nk_size_t query_count;
} nk_attention_work_t;

#pragma endregion Configuration

#pragma region Fragments

NUMKONG_DEVICE nk_f32_t nk_attention_negative_infinity_simt_(void) { return __int_as_float(0xFF800000); }

/** Slot of @p position in a transposed V: within every 16, position `2t + 8h + e` sits at
 *  `4t + 2h + e`. */
NUMKONG_DEVICE nk_size_t nk_attention_slot_simt_(nk_size_t position) {
    return (position & ~(nk_size_t)15) | ((position & 6) << 1) | ((position & 8) >> 2) | (position & 1);
}

/** Position held by @p slot of a transposed V, inverting @c nk_attention_slot_simt_. */
NUMKONG_DEVICE nk_size_t nk_attention_slot_position_simt_(nk_size_t slot) {
    return (slot & ~(nk_size_t)15) | ((slot & 12) >> 1) | ((slot & 2) << 2) | (slot & 1);
}

#pragma endregion Fragments

#pragma region Diagonal Band

/** Writes the keys of a segment of @p key_count that rows @p first_row to @p last_row see together,
 *  one range as both ends of a row's keys only grow with the row: from the first row seeing any to
 *  the last row. */
NUMKONG_DEVICE void nk_attention_rows_keys_simt_(nk_diagonal_band_t band, nk_i64_t first_row, nk_i64_t last_row,
                                                 nk_size_t key_count, nk_size_t *key_begin, nk_size_t *key_end) {
    // Rows above −superdiagonals see no key.
    nk_i64_t const first_seeing = first_row < 0 && band.superdiagonals < (nk_size_t)-first_row
                                      ? -(nk_i64_t)band.superdiagonals
                                      : first_row;
    nk_size_t unused;
    nk_diagonal_band_row_range_simt_(band, first_seeing, key_count, key_begin, &unused);
    nk_diagonal_band_row_range_simt_(band, last_row, key_count, &unused, key_end);
}

/** @c nk_diagonal_band_tile_coverage_simt_ of @p rows rows from @p row by the @p tile_keys keys
 *  from @p first_key of a segment of @p key_count, crossing wherever the tile runs past the
 *  segment's last key. */
NUMKONG_DEVICE nk_diagonal_band_coverage_t nk_attention_tile_coverage_simt_(nk_diagonal_band_t band, nk_i64_t row,
                                                                            nk_size_t rows, nk_size_t first_key,
                                                                            nk_size_t tile_keys, nk_size_t key_count) {
    if (first_key >= key_count) return nk_diagonal_band_outside_k;
    nk_size_t const keys = key_count - first_key < tile_keys ? key_count - first_key : tile_keys;
    nk_diagonal_band_coverage_t const coverage = nk_diagonal_band_tile_coverage_simt_(band, row, rows, first_key, keys);
    return coverage == nk_diagonal_band_inside_k && keys < tile_keys ? nk_diagonal_band_crossing_k : coverage;
}

#pragma endregion Diagonal Band

#pragma region Schedule

/** The rows of K and V head @p key_value_head before task @p task of a segment's own grid, row
 *  query × group + head, counting heads within the group. */
NUMKONG_DEVICE nk_size_t nk_attention_group_rows_before_simt_(nk_attention_schedule_t const *schedule, nk_size_t task,
                                                              nk_size_t key_value_head) {
    nk_size_t const group = schedule->group_heads, head = task % schedule->head_count;
    nk_size_t const head_first = key_value_head * group;
    nk_size_t const heads_before = head <= head_first ? 0 : head - head_first < group ? head - head_first : group;
    return task / schedule->head_count * group + heads_before;
}

/** Writes the rows of @p segment against @p key_value_head that the window covers into @p row_begin
 *  and @p row_end: one range, as a window of query tokens × heads cuts each head group in order. */
NUMKONG_DEVICE void nk_attention_segment_rows_simt_(nk_attention_schedule_t const *schedule, nk_size_t segment,
                                                    nk_size_t key_value_head, nk_size_t *row_begin,
                                                    nk_size_t *row_end) {
    nk_size_t const grid_first = schedule->query_offsets[segment] * schedule->head_count;
    nk_size_t const grid_end = schedule->query_offsets[segment + 1] * schedule->head_count;
    nk_size_t const begin = schedule->tasks_begin > grid_first ? schedule->tasks_begin : grid_first;
    nk_size_t const end = schedule->tasks_end < grid_end ? schedule->tasks_end : grid_end;
    *row_begin = *row_end = 0;
    if (begin >= end) return;
    *row_begin = nk_attention_group_rows_before_simt_(schedule, begin - grid_first, key_value_head);
    *row_end = nk_attention_group_rows_before_simt_(schedule, end - grid_first, key_value_head);
}

/** Whether the window covers every task of @p segment, which then cuts every head group alike. */
NUMKONG_DEVICE int nk_attention_segment_whole_simt_(nk_attention_schedule_t const *schedule, nk_size_t segment) {
    return schedule->tasks_begin <= schedule->query_offsets[segment] * schedule->head_count &&
           schedule->query_offsets[segment + 1] * schedule->head_count <= schedule->tasks_end;
}

/** Items of up to @p rows rows that @p segment splits into in the window, one K and V head each,
 *  cut at multiples of @p rows of the segment's grid so every window tiles a row alike. */
NUMKONG_DEVICE nk_size_t nk_attention_segment_items_simt_(nk_attention_schedule_t const *schedule, nk_size_t segment,
                                                          nk_size_t rows) {
    if (segment >= schedule->segment_end) return 0;
    nk_size_t const key_value_heads = schedule->head_count / schedule->group_heads;
    nk_size_t const queries = schedule->query_offsets[segment + 1] - schedule->query_offsets[segment];
    if (nk_attention_segment_whole_simt_(schedule, segment))
        return key_value_heads * nk_size_divide_round_up_(queries * schedule->group_heads, rows);
    nk_size_t items = 0;
    for (nk_size_t key_value_head = 0; key_value_head < key_value_heads; ++key_value_head) {
        nk_size_t row_begin, row_end;
        nk_attention_segment_rows_simt_(schedule, segment, key_value_head, &row_begin, &row_end);
        if (row_begin < row_end) items += nk_size_divide_round_up_(row_end, rows) - row_begin / rows;
    }
    return items;
}

/** Decodes item @p index of @p segment, of up to @p rows rows, into @p work. */
NUMKONG_DEVICE void nk_attention_segment_work_simt_(nk_attention_schedule_t const *schedule, nk_size_t segment,
                                                    nk_size_t index, nk_size_t rows, nk_attention_work_t *work) {
    nk_size_t key_value_head = 0, row_begin = 0, row_end = 0, block_first = 0;
    if (nk_attention_segment_whole_simt_(schedule, segment)) {
        row_end = (schedule->query_offsets[segment + 1] - schedule->query_offsets[segment]) * schedule->group_heads;
        nk_size_t const blocks = nk_size_divide_round_up_(row_end, rows);
        key_value_head = index / blocks, index %= blocks;
    }
    else
        for (;; ++key_value_head) {
            nk_attention_segment_rows_simt_(schedule, segment, key_value_head, &row_begin, &row_end);
            if (row_begin == row_end) continue;
            block_first = row_begin / rows;
            nk_size_t const blocks = nk_size_divide_round_up_(row_end, rows) - block_first;
            if (index < blocks) break;
            index -= blocks;
        }
    nk_size_t const block_begin = (block_first + index) * rows;
    work->segment = segment;
    work->key_value_head = key_value_head;
    work->head_first = key_value_head * schedule->group_heads;
    work->heads_selected = schedule->group_heads;
    work->row_first = block_begin > row_begin ? block_begin : row_begin;
    work->row_count = (block_begin + rows < row_end ? block_begin + rows : row_end) - work->row_first;
    work->query_first = schedule->query_offsets[segment];
    work->query_count = schedule->query_offsets[segment + 1] - work->query_first;
}

/** Reads the header, returning 0 when it disagrees with the arguments or the window is empty, and
 *  clips the window to the pack's query tokens × heads, starting at the first segment. */
NUMKONG_DEVICE int nk_attention_schedule_init_simt_(nk_attention_arguments_t const *arguments,
                                                    nk_attention_schedule_t *schedule) {
    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)arguments->packed;
    if (header->depth != arguments->depth || header->key_value_head_count != arguments->key_value_head_count) return 0;
    nk_size_t const segments = header->segments, heads = arguments->head_count;
    nk_u32_t const *query_offsets = arguments->query_offsets;
    nk_size_t const grid_begin = query_offsets[0] * heads, grid_end = query_offsets[segments] * heads;
    schedule->tasks_begin = arguments->tasks_begin > grid_begin ? arguments->tasks_begin : grid_begin;
    schedule->tasks_end = arguments->tasks_end < grid_end ? arguments->tasks_end : grid_end;
    if (schedule->tasks_begin >= schedule->tasks_end) return 0;
    schedule->query_offsets = query_offsets;
    schedule->head_count = heads;
    schedule->group_heads = heads / arguments->key_value_head_count;
    schedule->chunk_first = nk_attention_segment_of_serial_(query_offsets, segments, schedule->tasks_begin / heads);
    schedule->segment_end =
        nk_attention_segment_of_serial_(query_offsets, segments, (schedule->tasks_end - 1) / heads) + 1;
    schedule->items_before = 0;
    return 1;
}

/** The device twin of @c nk_attention_packed_payload_offsets_serial_. */
NUMKONG_DEVICE nk_u64_t const *nk_attention_packed_payload_offsets_simt_(unsigned char const *packed) {
    return (nk_u64_t const *)(packed + sizeof(nk_attention_packed_header_t));
}

/** The device twin of @c nk_attention_packed_key_offsets_serial_. */
NUMKONG_DEVICE nk_u32_t const *nk_attention_packed_key_offsets_simt_(unsigned char const *packed,
                                                                     nk_size_t segment_count) {
    return (nk_u32_t const *)(nk_attention_packed_payload_offsets_simt_(packed) + segment_count);
}

/** The device twin of @c nk_attention_packed_key_lengths_serial_. */
NUMKONG_DEVICE nk_u32_t const *nk_attention_packed_key_lengths_simt_(unsigned char const *packed,
                                                                     nk_size_t segment_count) {
    return nk_attention_packed_key_offsets_simt_(packed, segment_count) + segment_count + 1;
}

/** The device twin of @c nk_attention_pack_key_count_serial_. */
NUMKONG_DEVICE nk_size_t nk_attention_pack_key_count_simt_(nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                           nk_size_t segment) {
    return key_lengths ? key_lengths[segment] : key_offsets[segment + 1] - key_offsets[segment];
}

/** Bytes from the start of a pack of @p segment_count segments to its payload: the header and
 *  directory, rounded up to whole @p row_bytes rows, so one tensor map at the start reaches every
 *  row of every plane. */
NUMKONG_CONSTEXPR nk_size_t nk_attention_payload_offset_simt_(nk_size_t segment_count, nk_size_t row_bytes) {
    nk_size_t const directory_bytes = nk_size_round_up_to_multiple_(
        segment_count * sizeof(nk_u64_t) + (2 * segment_count + 1) * sizeof(nk_u32_t), 64);
    return nk_size_round_up_to_multiple_(sizeof(nk_attention_packed_header_t) + directory_bytes, row_bytes);
}

/** K plane of the work item's head; its V plane follows @c key_value_head_count planes later. */
NUMKONG_DEVICE unsigned char const *nk_attention_keys_plane_simt_(nk_attention_arguments_t const *arguments,
                                                                  nk_attention_work_t const *work, nk_size_t row_bytes,
                                                                  nk_size_t *length, nk_size_t *plane_bytes) {
    nk_size_t const segments = ((nk_attention_packed_header_t const *)arguments->packed)->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_simt_(arguments->packed);
    *length = nk_attention_packed_key_lengths_simt_(arguments->packed, segments)[work->segment];
    *plane_bytes = nk_size_round_up_to_multiple_(*length, nk_attention_panel_k) * row_bytes;
    return arguments->packed + nk_attention_payload_offset_simt_(segments, row_bytes) + payload_offsets[work->segment] +
           work->key_value_head * *plane_bytes;
}

/** The query row of row @p local of the work item, of @p element_bytes per element. */
NUMKONG_DEVICE unsigned char const *nk_attention_query_row_simt_(nk_attention_arguments_t const *arguments,
                                                                 nk_attention_work_t const *work, nk_size_t local,
                                                                 nk_size_t element_bytes) {
    nk_size_t const row = work->row_first + local;
    nk_size_t const query = row / work->heads_selected, head = work->head_first + row % work->heads_selected;
    return arguments->queries + (work->query_first + query) * arguments->query_stride +
           head * arguments->depth * element_bytes;
}

/** The output row of row @p local of the work item. */
NUMKONG_DEVICE nk_f32_t *nk_attention_output_row_simt_(nk_attention_arguments_t const *arguments,
                                                       nk_attention_work_t const *work, nk_size_t local) {
    nk_size_t const row = work->row_first + local;
    nk_size_t const query = row / work->heads_selected, head = work->head_first + row % work->heads_selected;
    return (nk_f32_t *)((unsigned char *)arguments->output + (work->query_first + query) * arguments->output_stride) +
           head * arguments->depth;
}

/** The signed band row of row @p local of the work item, its query aligned to the end of the
 *  segment's @p length keys. */
NUMKONG_DEVICE nk_i64_t nk_attention_row_position_simt_(nk_attention_work_t const *work, nk_size_t local,
                                                        nk_size_t length) {
    return (nk_i64_t)((work->row_first + local) / work->heads_selected + length) - (nk_i64_t)work->query_count;
}

/** The band a kernel of @p mask applies: the launch's, or one reaching every key, which the
 *  compiler folds away. */
NUMKONG_DEVICE nk_diagonal_band_t nk_attention_kernel_band_simt_(nk_attention_mask_t mask,
                                                                 nk_attention_arguments_t const *arguments) {
    nk_diagonal_band_t band = arguments->band;
    if (mask == nk_attention_mask_none_k) band.subdiagonals = band.superdiagonals = NUMKONG_SIZE_MAX;
    return band;
}

/** Where row @p local keeps its log-sum-exp, or null when the launch skips them. */
NUMKONG_DEVICE nk_f32_t *nk_attention_log_sum_exp_slot_simt_(nk_attention_arguments_t const *arguments,
                                                             nk_attention_work_t const *work, nk_size_t local) {
    if (!arguments->log_sum_exp) return NUMKONG_NULL;
    nk_size_t const row = work->row_first + local;
    nk_size_t const query = row / work->heads_selected, head = work->head_first + row % work->heads_selected;
    return arguments->log_sum_exp + (work->query_first + query) * arguments->head_count + head;
}

/** The natural log-sum-exp of a row whose base-2 scores peak at @p max2, given the @p sum of its
 *  weights relative to that peak, each @p unit for a probability of one, as U8 weights count 255
 *  and amplified E4M3 ones 256; −∞ for a row that saw no key. */
NUMKONG_DEVICE nk_f32_t nk_attention_log_sum_exp_simt_(nk_f32_t max2, nk_f32_t sum, nk_f32_t unit) {
    return sum > 0 ? (max2 + log2f(sum / unit)) * NUMKONG_F32_LN2_ : nk_attention_negative_infinity_simt_();
}

#pragma endregion Schedule

#pragma region Tile

/** Copies the first @p rows query rows of the work item, 2-byte elements, into @p queries, rows
 *  @p row_stride bytes apart, @p depth_padded elements each with zeros past the item and past the
 *  depth, @p lanes threads per row. */
NUMKONG_DEVICE void nk_attention_stage_queries_b16_simt_(nk_attention_arguments_t const *arguments,
                                                         nk_attention_work_t const *work, unsigned char *queries,
                                                         unsigned rows, unsigned row_stride, unsigned depth_padded,
                                                         unsigned lanes) {
    unsigned const lane = threadIdx.x % lanes;
    for (unsigned local = threadIdx.x / lanes; local < rows; local += nk_attention_threads_k / lanes) {
        unsigned char *destination = queries + local * row_stride;
        int const valid = local < work->row_count;
        unsigned char const *source = valid ? nk_attention_query_row_simt_(arguments, work, local, 2) : NUMKONG_NULL;
        unsigned short *destination_words = (unsigned short *)destination;
        unsigned short const *source_words = (unsigned short const *)source;
        for (unsigned element = lane; element < depth_padded; element += lanes) {
            int const inside = valid && element < arguments->depth;
            destination_words[element] = inside ? source_words[element] : (unsigned short)0;
        }
    }
}

/** The 1-byte twin of @c nk_attention_stage_queries_b16_simt_. */
NUMKONG_DEVICE void nk_attention_stage_queries_b8_simt_(nk_attention_arguments_t const *arguments,
                                                        nk_attention_work_t const *work, unsigned char *queries,
                                                        unsigned rows, unsigned row_stride, unsigned depth_padded,
                                                        unsigned lanes) {
    unsigned const lane = threadIdx.x % lanes;
    for (unsigned local = threadIdx.x / lanes; local < rows; local += nk_attention_threads_k / lanes) {
        unsigned char *destination = queries + local * row_stride;
        int const valid = local < work->row_count;
        unsigned char const *source = valid ? nk_attention_query_row_simt_(arguments, work, local, 1) : NUMKONG_NULL;
        for (unsigned element = lane; element < depth_padded; element += lanes) {
            int const inside = valid && element < arguments->depth;
            destination[element] = inside ? source[element] : (unsigned char)0;
        }
    }
}

/** One element of a BF16 K row or V element, decoded to F32. */
NUMKONG_DEVICE nk_f32_t nk_attention_decode_bf16_simt_(unsigned char const *bytes, nk_size_t index) {
    unsigned short const *words = (unsigned short const *)bytes;
    return __uint_as_float((nk_u32_t)words[index] << 16);
}

/** One element of an F16 K row or V element, decoded to F32. */
NUMKONG_DEVICE nk_f32_t nk_attention_decode_f16_simt_(unsigned char const *bytes, nk_size_t index) {
    unsigned short const *words = (unsigned short const *)bytes;
    return __half2float(__ushort_as_half(words[index]));
}

/** One element of an E4M3 K row or V element, decoded to F32 and amplified by 256. */
NUMKONG_DEVICE nk_f32_t nk_attention_decode_e4m3_simt_(unsigned char const *bytes, nk_size_t index) {
    unsigned const code = bytes[index];
    return __half2float(__ushort_as_half((unsigned short)(((code & 0x7Fu) << 7) | ((code & 0x80u) << 8)))) * 256.0f;
}

/** One element of an I8 K row or V element, an exact integer in F32. */
NUMKONG_DEVICE nk_f32_t nk_attention_decode_i8_simt_(unsigned char const *bytes, nk_size_t index) {
    return (nk_f32_t)(signed char)bytes[index];
}

#pragma endregion Tile

#pragma region Pack

/** Moves a block's cursor from segment @p segment, whose payload starts @p offset bytes in, to
 *  segment @p target, adding the payload bytes of the segments it passes as a block-wide sum, which
 *  every thread of a block of @c nk_attention_pack_threads_k threads calls in step. */
NUMKONG_DEVICE void nk_attention_pack_advance_simt_(nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                    nk_size_t key_value_head_count, nk_size_t row_bytes,
                                                    nk_size_t target, nk_size_t *segment, nk_u64_t *offset) {
    __shared__ nk_u64_t partials[nk_attention_pack_threads_k];
    if (*segment == target) return;
    nk_u64_t sum = 0;
    for (nk_size_t passed = *segment + threadIdx.x; passed < target; passed += blockDim.x)
        sum += nk_attention_pack_segment_bytes_serial_(
            nk_attention_pack_key_count_simt_(key_offsets, key_lengths, passed), key_value_head_count,
            nk_attention_panel_k, row_bytes);
    partials[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned half = blockDim.x / 2; half != 0; half >>= 1) {
        if (threadIdx.x < half) partials[threadIdx.x] += partials[threadIdx.x + half];
        __syncthreads();
    }
    *offset += partials[0], *segment = target;
    // Every thread reads the sum before a later advance rewrites it.
    __syncthreads();
}

/** Copies the K and V planes of each @b (segment,kv_head) task of 2-byte elements, one
 *  block per task, at offsets the key counts alone give, zeroing every padded element; V
 *  keeps position rows. */
NUMKONG_DEVICE void nk_attention_pack_payload_b16_simt_(unsigned char const *keys, unsigned char const *values,
                                                        nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                        nk_size_t segment_count, nk_size_t key_stride,
                                                        nk_size_t value_stride, unsigned char *packed,
                                                        nk_size_t tasks_begin, nk_size_t tasks_end) {
    nk_size_t const depth_bytes = depth * 2;
    nk_size_t const row_bytes = nk_size_round_up_to_multiple_(depth_bytes, nk_attention_step_bytes_k);
    nk_size_t const row_elements = row_bytes / 2, row_chunks = row_bytes / 16;
    unsigned char *payload = packed + nk_attention_payload_offset_simt_(segment_count, row_bytes);
    // Planes start on 16 bytes, so 16-byte chunks move whenever every source head row does too.
    int const chunked = (((nk_size_t)keys | (nk_size_t)values | key_stride | value_stride | depth_bytes) & 15) == 0;
    uint4 const zero = make_uint4(0, 0, 0, 0);
    nk_size_t cursor_segment = 0;
    nk_u64_t payload_offset = 0;
    for (nk_size_t task = tasks_begin + blockIdx.x; task < tasks_end; task += gridDim.x) {
        nk_size_t const segment = task / key_value_head_count, head = task % key_value_head_count;
        nk_attention_pack_advance_simt_(key_offsets, key_lengths, key_value_head_count, row_bytes, segment,
                                        &cursor_segment, &payload_offset);
        nk_size_t const length = nk_attention_pack_key_count_simt_(key_offsets, key_lengths, segment);
        if (length == 0) continue;
        nk_size_t const positions_padded = nk_size_round_up_to_multiple_(length, nk_attention_panel_k);
        nk_size_t const plane_bytes = positions_padded * row_bytes;
        unsigned char *keys_plane = payload + payload_offset + head * plane_bytes;
        unsigned char *values_plane = keys_plane + key_value_head_count * plane_bytes;
        unsigned char const *keys_first = keys + key_offsets[segment] * key_stride + head * depth_bytes;
        unsigned char const *values_first = values + key_offsets[segment] * value_stride + head * depth_bytes;
        uint4 *keys_quartets = (uint4 *)keys_plane;
        uint4 *values_quartets = (uint4 *)values_plane;
        unsigned short *keys_words = (unsigned short *)keys_plane;
        unsigned short *values_words = (unsigned short *)values_plane;
        if (chunked)
            for (nk_size_t index = threadIdx.x; index < plane_bytes / 16; index += blockDim.x) {
                nk_size_t const position = index / row_chunks, byte = index % row_chunks * 16;
                int const inside = position < length && byte < depth_bytes;
                keys_quartets[index] = inside ? *(uint4 const *)(keys_first + position * key_stride + byte) : zero;
                values_quartets[index] = inside ? *(uint4 const *)(values_first + position * value_stride + byte)
                                                : zero;
            }
        else
            for (nk_size_t index = threadIdx.x; index < positions_padded * row_elements; index += blockDim.x) {
                nk_size_t const position = index / row_elements, element = index % row_elements;
                int const inside = position < length && element < depth;
                unsigned char const *key = keys_first + position * key_stride + element * 2;
                unsigned char const *value = values_first + position * value_stride + element * 2;
                keys_words[index] = inside ? *(unsigned short const *)key : (unsigned short)0;
                values_words[index] = inside ? *(unsigned short const *)value : (unsigned short)0;
            }
    }
}

/** The 1-byte twin of @c nk_attention_pack_payload_b16_simt_: V is transposed, a depth row of
 *  positions per element. */
NUMKONG_DEVICE void nk_attention_pack_payload_b8_simt_(unsigned char const *keys, unsigned char const *values,
                                                       nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                       nk_size_t segment_count, nk_size_t key_stride,
                                                       nk_size_t value_stride, unsigned char *packed,
                                                       nk_size_t tasks_begin, nk_size_t tasks_end) {
    nk_size_t const depth_bytes = depth * 1;
    nk_size_t const row_bytes = nk_size_round_up_to_multiple_(depth_bytes, nk_attention_step_bytes_k);
    nk_size_t const row_elements = row_bytes / 1, row_chunks = row_bytes / 16;
    unsigned char *payload = packed + nk_attention_payload_offset_simt_(segment_count, row_bytes);
    // Planes start on 16 bytes, so 16-byte chunks move whenever every source head row does too.
    int const chunked = (((nk_size_t)keys | (nk_size_t)values | key_stride | value_stride | depth_bytes) & 15) == 0;
    uint4 const zero = make_uint4(0, 0, 0, 0);
    nk_size_t cursor_segment = 0;
    nk_u64_t payload_offset = 0;
    for (nk_size_t task = tasks_begin + blockIdx.x; task < tasks_end; task += gridDim.x) {
        nk_size_t const segment = task / key_value_head_count, head = task % key_value_head_count;
        nk_attention_pack_advance_simt_(key_offsets, key_lengths, key_value_head_count, row_bytes, segment,
                                        &cursor_segment, &payload_offset);
        nk_size_t const length = nk_attention_pack_key_count_simt_(key_offsets, key_lengths, segment);
        if (length == 0) continue;
        nk_size_t const positions_padded = nk_size_round_up_to_multiple_(length, nk_attention_panel_k);
        nk_size_t const plane_bytes = positions_padded * row_bytes;
        unsigned char *keys_plane = payload + payload_offset + head * plane_bytes;
        unsigned char *values_plane = keys_plane + key_value_head_count * plane_bytes;
        unsigned char const *keys_first = keys + key_offsets[segment] * key_stride + head * depth_bytes;
        unsigned char const *values_first = values + key_offsets[segment] * value_stride + head * depth_bytes;
        uint4 *keys_quartets = (uint4 *)keys_plane;
        if (chunked)
            for (nk_size_t index = threadIdx.x; index < plane_bytes / 16; index += blockDim.x) {
                nk_size_t const position = index / row_chunks, byte = index % row_chunks * 16;
                int const inside = position < length && byte < depth_bytes;
                keys_quartets[index] = inside ? *(uint4 const *)(keys_first + position * key_stride + byte) : zero;
            }
        else
            for (nk_size_t index = threadIdx.x; index < positions_padded * row_elements; index += blockDim.x) {
                nk_size_t const position = index / row_elements, element = index % row_elements;
                int const inside = position < length && element < depth;
                unsigned char const *key = keys_first + position * key_stride + element;
                keys_plane[index] = inside ? *key : (unsigned char)0;
            }
        for (nk_size_t index = threadIdx.x; index < row_elements * positions_padded; index += blockDim.x) {
            nk_size_t const element = index / positions_padded;
            nk_size_t const position = nk_attention_slot_position_simt_(index % positions_padded);
            values_plane[index] = position < length && element < depth ? values_first[position * value_stride + element]
                                                                       : (unsigned char)0;
        }
    }
}

/** Copies the planes of BF16 keys and values with the b16 payload. */
NUMKONG_DEVICE void nk_attention_pack_payload_bf16_simt_(unsigned char const *keys, unsigned char const *values,
                                                         nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                         nk_size_t segment_count, nk_size_t key_stride,
                                                         nk_size_t value_stride, unsigned char *packed,
                                                         nk_size_t tasks_begin, nk_size_t tasks_end) {
    nk_attention_pack_payload_b16_simt_(keys, values, key_value_head_count, depth, key_offsets, key_lengths,
                                        segment_count, key_stride, value_stride, packed, tasks_begin, tasks_end);
}

/** Copies the planes of F16 keys and values with the b16 payload. */
NUMKONG_DEVICE void nk_attention_pack_payload_f16_simt_(unsigned char const *keys, unsigned char const *values,
                                                        nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                        nk_size_t segment_count, nk_size_t key_stride,
                                                        nk_size_t value_stride, unsigned char *packed,
                                                        nk_size_t tasks_begin, nk_size_t tasks_end) {
    nk_attention_pack_payload_b16_simt_(keys, values, key_value_head_count, depth, key_offsets, key_lengths,
                                        segment_count, key_stride, value_stride, packed, tasks_begin, tasks_end);
}

/** Copies the planes of E4M3 keys and values with the b8 payload. */
NUMKONG_DEVICE void nk_attention_pack_payload_e4m3_simt_(unsigned char const *keys, unsigned char const *values,
                                                         nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                         nk_size_t segment_count, nk_size_t key_stride,
                                                         nk_size_t value_stride, unsigned char *packed,
                                                         nk_size_t tasks_begin, nk_size_t tasks_end) {
    nk_attention_pack_payload_b8_simt_(keys, values, key_value_head_count, depth, key_offsets, key_lengths,
                                       segment_count, key_stride, value_stride, packed, tasks_begin, tasks_end);
}

/** Copies the planes of I8 keys and values with the b8 payload. */
NUMKONG_DEVICE void nk_attention_pack_payload_i8_simt_(unsigned char const *keys, unsigned char const *values,
                                                       nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                       nk_size_t segment_count, nk_size_t key_stride,
                                                       nk_size_t value_stride, unsigned char *packed,
                                                       nk_size_t tasks_begin, nk_size_t tasks_end) {
    nk_attention_pack_payload_b8_simt_(keys, values, key_value_head_count, depth, key_offsets, key_lengths,
                                       segment_count, key_stride, value_stride, packed, tasks_begin, tasks_end);
}

/** Bounds the device pack of @p token_count keys in @p segment_count segments however they split:
 *  header and directory up to the payload, then both planes of every segment and head, each
 *  padded to a whole panel. */
NUMKONG_INLINE nk_size_t nk_attention_pack_size_simt_(nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_size_t token_count, nk_size_t segment_count,
                                                      nk_size_t element_bytes) {
    nk_size_t const row_bytes = nk_size_round_up_to_multiple_(depth * element_bytes, nk_attention_step_bytes_k);
    nk_size_t const positions = token_count + segment_count * (nk_attention_panel_k - 1);
    return nk_attention_payload_offset_simt_(segment_count, row_bytes) +
           2 * key_value_head_count * positions * row_bytes;
}

#pragma endregion Pack

#pragma region Backward

/** Everything one backward launch shares, passed by value as the kernels' only argument. */
typedef struct {

    /** Query rows, @c query_stride bytes apart. */
    unsigned char const *queries;

    /** The packed buffer the forward read, header first. */
    unsigned char const *packed;

    /** The forward's F32 output rows, @c output_stride bytes apart. */
    nk_f32_t const *output;

    /** Gradient of the loss with respect to the output, laid out like it. */
    nk_f32_t const *output_gradient;

    /** The forward's natural log-sum-exp per query token and head. */
    nk_f32_t const *log_sum_exp;

    /** Gradient with respect to the queries, rows @c query_gradient_stride bytes apart. */
    nk_f32_t *query_gradient;

    /** Gradients of the keys and values, rows @c key_value_gradient_stride bytes apart. */
    nk_f32_t *key_gradient;
    nk_f32_t *value_gradient;

    /** First query token of each segment. */
    nk_u32_t const *query_offsets;

    nk_size_t head_count;
    nk_size_t key_value_head_count;
    nk_size_t depth;
    nk_size_t query_stride;
    nk_size_t output_stride;
    nk_size_t query_gradient_stride;
    nk_size_t key_value_gradient_stride;

    /** Score multiplier, and the same in base 2. */
    nk_f32_t scale;
    nk_f32_t scale2;

    /** Keys each query row sees around its position. */
    nk_diagonal_band_t band;

    /** The window over segments × key-value heads, its end clipped on the device. */
    nk_size_t tasks_begin;
    nk_size_t tasks_end;

    /** Whether a dQ row holds its dO as BF16 and D past it, until the query pass overwrites it. */
    int prepared;
} nk_attention_backward_arguments_t;

/** One backward task's segment: its keys and values planes, first key token, key count and query
 *  rows, and the band row of its first query, which aligns its queries to the end of its keys. */
typedef struct {
    unsigned char const *keys_plane;
    unsigned char const *values_plane;
    nk_size_t key_first;
    nk_size_t length;
    nk_size_t rows;
    nk_size_t segment;
    nk_size_t key_value_head;
    nk_i64_t first_band_row;
} nk_attention_backward_task_t;

/** Resolves @p task of the window against the pack's directory. */
NUMKONG_DEVICE nk_attention_backward_task_t nk_attention_backward_task_simt_(
    nk_attention_backward_arguments_t const *arguments, nk_size_t task, nk_size_t row_bytes) {
    nk_size_t const segments = ((nk_attention_packed_header_t const *)arguments->packed)->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_simt_(arguments->packed);
    nk_attention_backward_task_t result;
    result.segment = task / arguments->key_value_head_count;
    result.key_value_head = task % arguments->key_value_head_count;
    result.key_first = nk_attention_packed_key_offsets_simt_(arguments->packed, segments)[result.segment];
    result.length = nk_attention_packed_key_lengths_simt_(arguments->packed, segments)[result.segment];
    result.rows = arguments->query_offsets[result.segment + 1] - arguments->query_offsets[result.segment];
    result.first_band_row = (nk_i64_t)result.length - (nk_i64_t)result.rows;
    nk_size_t const plane_bytes = nk_size_round_up_to_multiple_(result.length, nk_attention_panel_k) * row_bytes;
    result.keys_plane = arguments->packed + nk_attention_payload_offset_simt_(segments, row_bytes) +
                        payload_offsets[result.segment] + result.key_value_head * plane_bytes;
    result.values_plane = result.keys_plane + arguments->key_value_head_count * plane_bytes;
    return result;
}

/** The one past the last task of the window, clipped to the pack's segments × key-value heads. */
NUMKONG_DEVICE nk_size_t nk_attention_backward_task_end_simt_(nk_attention_backward_arguments_t const *arguments) {
    nk_size_t const tasks = ((nk_attention_packed_header_t const *)arguments->packed)->segments *
                            arguments->key_value_head_count;
    return arguments->tasks_end < tasks ? arguments->tasks_end : tasks;
}

/** Fills the backward arguments, @p tasks_end left for the device to clip. */
NUMKONG_INLINE nk_attention_backward_arguments_t nk_attention_backward_arguments_init_simt_(
    void const *queries, void const *packed, nk_f32_t const *output, nk_f32_t const *output_gradient,
    nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient, nk_f32_t *key_gradient, nk_f32_t *value_gradient,
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_stride, nk_size_t output_stride, nk_size_t query_gradient_stride,
    nk_size_t key_value_gradient_stride, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_size_t tasks_begin, nk_size_t tasks_end) {
    nk_attention_backward_arguments_t arguments;
    arguments.queries = (unsigned char const *)queries, arguments.packed = (unsigned char const *)packed;
    arguments.output = output, arguments.output_gradient = output_gradient, arguments.log_sum_exp = log_sum_exp;
    arguments.query_gradient = query_gradient, arguments.key_gradient = key_gradient;
    arguments.value_gradient = value_gradient;
    arguments.query_offsets = query_offsets;
    arguments.head_count = head_count, arguments.key_value_head_count = key_value_head_count, arguments.depth = depth;
    arguments.query_stride = query_stride, arguments.output_stride = output_stride;
    arguments.query_gradient_stride = query_gradient_stride;
    arguments.key_value_gradient_stride = key_value_gradient_stride;
    arguments.scale = scale, arguments.scale2 = scale * NUMKONG_F32_LOG2E_;
    arguments.band.subdiagonals = keys_before, arguments.band.superdiagonals = keys_after;
    arguments.tasks_begin = tasks_begin, arguments.tasks_end = tasks_end;
    arguments.prepared = 0;
    return arguments;
}

#pragma endregion Backward

#pragma region Launch Arguments

/** The arguments of one launch, with the shared-memory offsets zero for the capability to place and
 *  @p tasks_end left for the device to clip. */
NUMKONG_INLINE nk_attention_arguments_t nk_attention_arguments_init_simt_(
    void const *queries, void const *packed, nk_f32_t *output, nk_f32_t *log_sum_exp, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride,
    nk_size_t output_stride, nk_f32_t scale, nk_f32_t score_scale, nk_f32_t output_scale, nk_size_t keys_before,
    nk_size_t keys_after, nk_size_t tasks_begin, nk_size_t tasks_end) {
    nk_attention_arguments_t arguments;
    arguments.queries = (unsigned char const *)queries, arguments.packed = (unsigned char const *)packed;
    arguments.output = output, arguments.log_sum_exp = log_sum_exp, arguments.query_offsets = query_offsets;
    arguments.head_count = head_count, arguments.key_value_head_count = key_value_head_count;
    arguments.depth = depth, arguments.query_stride = query_stride, arguments.output_stride = output_stride;
    arguments.band.subdiagonals = keys_before, arguments.band.superdiagonals = keys_after;
    arguments.tasks_begin = tasks_begin, arguments.tasks_end = tasks_end;
    arguments.scale2 = scale * NUMKONG_F32_LOG2E_;
    arguments.score_scale = score_scale, arguments.output_scale = output_scale;
    arguments.key_offset[0] = arguments.key_offset[1] = 0;
    arguments.value_offset[0] = arguments.value_offset[1] = 0;
    arguments.query_offset[0] = arguments.query_offset[1] = 0;
    return arguments;
}

#pragma endregion Launch Arguments

#pragma region Attention Macros

/** Generates the host-side size of a pack: header, directory, and both planes of every
 *  segment and head. */
#define nk_define_attention_pack_size_simt_(input_type_name, isa_suffix, element_bytes)                                \
    NUMKONG_API nk_status_t nk_attention_pack_size_##input_type_name##_##isa_suffix(                                   \
        nk_size_t key_value_head_count, nk_size_t depth, nk_size_t token_count, nk_size_t segment_count,               \
        nk_size_t *bytes) {                                                                                            \
        *bytes = nk_attention_pack_size_simt_(key_value_head_count, depth, token_count, segment_count, element_bytes); \
        return nk_success_k;                                                                                           \
    }

/**
 *  @brief Generates the narrow and wide kernels of one dtype, unmasked and masked by the band, and
 *      the fallback kernel, then the public attention entry point over them, picking the masked
 *      pair unless the band reaches every key, and passing them to the capability's launch.
 *
 *  The pack must come from the same capability's pack kernel, which the entry point trusts: the
 *  host can't read its device header without waiting on the stream.
 *
 *  @param[in] input_type The dtype, like @c bf16.
 *  @param[in] isa_suffix The capability, like @c ampere, whose tile
 *      @c nk_attention_tile_<type>_<capability>_ the kernels run and whose launch
 *      @c nk_attention_launch_<type>_<capability>_ the entry point calls.
 *  @param[in] vendor The vendor floor, @c cuda or @c rocm, whose
 *      @c nk_attention_fallback_<type>_<vendor>_ runs the depths the tile does not take.
 */
#define nk_define_attention_packed_simt_(input_type, isa_suffix, vendor)                                              \
    static __global__ void __launch_bounds__(nk_attention_threads_k)                                                  \
        nk_attention_packed_##input_type##_narrow_##isa_suffix##_kernel_(nk_attention_arguments_t arguments) {        \
        nk_attention_tile_##input_type##_##isa_suffix##_(nk_attention_width_128_k, nk_attention_mask_none_k,          \
                                                         &arguments);                                                 \
    }                                                                                                                 \
    static __global__ void __launch_bounds__(nk_attention_threads_k)                                                  \
        nk_attention_packed_##input_type##_wide_##isa_suffix##_kernel_(nk_attention_arguments_t arguments) {          \
        nk_attention_tile_##input_type##_##isa_suffix##_(nk_attention_width_256_k, nk_attention_mask_none_k,          \
                                                         &arguments);                                                 \
    }                                                                                                                 \
    static __global__ void __launch_bounds__(nk_attention_threads_k)                                                  \
        nk_attention_packed_##input_type##_narrow_masked_##isa_suffix##_kernel_(nk_attention_arguments_t arguments) { \
        nk_attention_tile_##input_type##_##isa_suffix##_(nk_attention_width_128_k, nk_attention_mask_diagonal_band_k, \
                                                         &arguments);                                                 \
    }                                                                                                                 \
    static __global__ void __launch_bounds__(nk_attention_threads_k)                                                  \
        nk_attention_packed_##input_type##_wide_masked_##isa_suffix##_kernel_(nk_attention_arguments_t arguments) {   \
        nk_attention_tile_##input_type##_##isa_suffix##_(nk_attention_width_256_k, nk_attention_mask_diagonal_band_k, \
                                                         &arguments);                                                 \
    }                                                                                                                 \
    static __global__ void __launch_bounds__(nk_attention_threads_k)                                                  \
        nk_attention_packed_##input_type##_fallback_##isa_suffix##_kernel_(nk_attention_arguments_t arguments) {      \
        nk_attention_fallback_##input_type##_##vendor##_(&arguments);                                                 \
    }                                                                                                                 \
    NUMKONG_API nk_status_t nk_attention_packed_##input_type##_##isa_suffix(                                          \
        nk_##input_type##_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_f32_t *log_sum_exp,    \
        nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,         \
        nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, \
        nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {                                             \
        int const masked = keys_before != NUMKONG_SIZE_MAX || keys_after != NUMKONG_SIZE_MAX;                         \
        return nk_attention_launch_##input_type##_##isa_suffix##_(                                                    \
            masked ? (void const *)nk_attention_packed_##input_type##_narrow_masked_##isa_suffix##_kernel_            \
                   : (void const *)nk_attention_packed_##input_type##_narrow_##isa_suffix##_kernel_,                  \
            masked ? (void const *)nk_attention_packed_##input_type##_wide_masked_##isa_suffix##_kernel_              \
                   : (void const *)nk_attention_packed_##input_type##_wide_##isa_suffix##_kernel_,                    \
            (void const *)nk_attention_packed_##input_type##_fallback_##isa_suffix##_kernel_, queries,                \
            key_value_packed, output, log_sum_exp, head_count, key_value_head_count, depth, query_offsets,            \
            query_stride, output_stride, scale, keys_before, keys_after, tasks_begin, tasks_end, stream);             \
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
    nk_size_t x_stride;
    nk_size_t y_stride;
} nk_attention_rope_arguments_t;

#pragma endregion Rotary Embeddings

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_
#endif // NUMKONG_ATTENTION_SIMT_CUH
