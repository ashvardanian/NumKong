/**
 *  @file include/numkong/attention/cdna3.cuh
 *  @author Ash Vardanian
 *  @date October 7, 2026
 *  @brief Ragged attention for AMD Instinct MI300, gfx942.
 *
 *  @sa include/numkong/attention.h
 *  @sa include/numkong/dots/cdna3.cuh
 *
 *  FlashAttention-2 on 16 × 16 MFMAs: two wavefronts own the 64 rows of a work item, 32 each as two
 *  16-row query tiles, and walk 64-position K and V panels staged in shared memory with a base-2
 *  online softmax per row. Scores come out transposed, Sᵀ = K · Qᵀ, so each lane holds 4 positions
 *  of one query row, exactly the probabilities its lane feeds P · V as the A operand; K rows are
 *  picked per lane so those positions are the ones the V operand holds. BF16 multiplies BF16, and
 *  I8 runs exact integer MFMA with U8 probabilities offset into I8, each 64-byte depth step and each
 *  P · V step taking two of MI300's MFMAs. MI300's Float8 MFMA reads the FNUZ encodings rather than
 *  OCP's, so E4M3 runs the @c rocm kernel. Depths above 256 fall back to the @c rocm kernel, and so
 *  do BF16 depths above 128, whose panels outgrow MI300's 64 KiB of LDS. The pack layout and the
 *  work scheduler are the @c rocm capability's; the staging, the frame helpers and the launch serve
 *  the CDNA4 kernel too, and the staging and the launch the CDNA5 one.
 */
#ifndef NUMKONG_ATTENTION_CDNA3_CUH
#define NUMKONG_ATTENTION_CDNA3_CUH

#if NUMKONG_ARCH_ROCM_
#if NUMKONG_ARCH_ROCM_CDNA3_

#include "numkong/attention/rocm.cuh" // `nk_attention_schedule_next_rocm_`, `nk_attention_fallback_bf16_rocm_`
#include "numkong/dots/cdna3.cuh"     // `nk_mfma_bf16_cdna3_`, `nk_dot_i8x4_cdna3_`

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Configuration

enum {
    nk_attention_narrow_depth_cdna3_k = 128,
    nk_attention_wide_depth_cdna3_k = 256,
    nk_attention_row_padding_cdna3_k = 16,
    nk_attention_wave_rows_cdna3_k = 32,
    nk_attention_steps_cdna3_k = 8,
    nk_attention_tiles_cdna3_k = nk_attention_wide_depth_cdna3_k / 16,
};

#pragma endregion Configuration

#pragma region Conversions

/** Rounds @p value to the nearest BF16, ties to even; a NaN keeps its sign and top payload bits. */
NUMKONG_DEVICE nk_u32_t nk_f32_to_bf16_cdna3_(nk_f32_t value) {
    nk_u32_t const bits = __float_as_uint(value);
    // Rounding could carry a NaN into infinity or zero
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) return (bits >> 16) | 0x0040u;
    return (bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16;
}

#pragma endregion Conversions

#pragma region Staging

/** Copies @p rows rows of @p row_bytes from a position-major plane at @p first_position into
 *  shared rows @p stride bytes apart. */
NUMKONG_DEVICE void nk_attention_stage_rows_cdna3_(unsigned char *shared, unsigned char const *plane,
                                                   nk_size_t first_position, unsigned rows, unsigned row_bytes,
                                                   unsigned stride) {
    unsigned const chunks_per_row = row_bytes >> 4;
    for (unsigned chunk = threadIdx.x; chunk < rows * chunks_per_row; chunk += nk_attention_threads_k) {
        unsigned const row = chunk / chunks_per_row, column = chunk - row * chunks_per_row;
        *(uint4 *)(shared + row * stride +
                   (column << 4)) = *(uint4 const *)(plane + (first_position + row) * row_bytes + (column << 4));
    }
}

/** Stages a panel of BF16 V as Vᵀ, depth rows of 64 slots @p stride bytes apart: rows transpose
 *  into the slot order. */
NUMKONG_DEVICE void nk_attention_stage_values_bf16_cdna3_(unsigned char *shared, unsigned char const *plane,
                                                          nk_size_t positions_padded, nk_size_t first_position,
                                                          unsigned row_bytes, unsigned stride) {
    // Consecutive threads take consecutive positions, so each depth row's stores stay contiguous.
    for (unsigned chunk = threadIdx.x; chunk < (row_bytes >> 4) * nk_attention_panel_k;
         chunk += nk_attention_threads_k) {
        unsigned const position = chunk & (nk_attention_panel_k - 1), column = chunk / nk_attention_panel_k;
        uint4 const halves = *(uint4 const *)(plane + (first_position + position) * row_bytes + (column << 4));
        nk_u32_t const words[4] = {halves.x, halves.y, halves.z, halves.w};
        unsigned char *destination = shared + column * 8 * stride + nk_attention_slot_simt_(position) * 2;
#pragma unroll
        for (unsigned element = 0; element < 8; ++element)
            *(unsigned short *)(destination + element * stride) = (unsigned short)(words[element / 2] >>
                                                                                   (element % 2 * 16));
    }
}

/** Stages a panel of 1-byte V as Vᵀ, depth rows of 64 slots @p stride bytes apart: the codes copy
 *  the pack's transposed rows. */
NUMKONG_DEVICE void nk_attention_stage_values_b8_cdna3_(unsigned char *shared, unsigned char const *plane,
                                                        nk_size_t positions_padded, nk_size_t first_position,
                                                        unsigned row_bytes, unsigned stride) {
    for (unsigned chunk = threadIdx.x; chunk < row_bytes * 4; chunk += nk_attention_threads_k) {
        unsigned const row = chunk >> 2, column = chunk & 3;
        *(uint4 *)(shared + row * stride + (column << 4)) = *(uint4 const *)(plane + row * positions_padded +
                                                                             first_position + (column << 4));
    }
}

/** Loads @p words words, 4 to 16, of a lane's fragment at @p offset of a staged @p row, as zeros
 *  from @p valid_bytes on, which only ever falls between 16-byte chunks. */
NUMKONG_DEVICE void nk_attention_load_fragment_cdna3_(unsigned char const *row, unsigned offset, unsigned valid_bytes,
                                                      unsigned words, nk_u32_t *fragment) {
#pragma unroll
    for (unsigned chunk = 0; chunk < 4; ++chunk) {
        if (chunk * 4 >= words) continue;
        uint4 bytes = make_uint4(0, 0, 0, 0);
        if (offset + chunk * 16 < valid_bytes) bytes = *(uint4 const *)(row + offset + chunk * 16);
        fragment[chunk * 4 + 0] = bytes.x, fragment[chunk * 4 + 1] = bytes.y;
        fragment[chunk * 4 + 2] = bytes.z, fragment[chunk * 4 + 3] = bytes.w;
    }
}

#pragma endregion Staging

#pragma region Launch

/** Shared bytes a block needs: the K panel, then Vᵀ, which Q shares ahead of the first panel. */
NUMKONG_INLINE nk_size_t nk_attention_shared_bytes_cdna3_(nk_size_t element_bytes, nk_size_t depth) {
    nk_size_t const row_bytes = nk_size_round_up_to_multiple_(depth * element_bytes, nk_attention_step_bytes_k);
    nk_size_t const rows = nk_attention_panel_k * (row_bytes + nk_attention_row_padding_cdna3_k);
    nk_size_t const values = row_bytes / element_bytes *
                             (nk_attention_panel_k * element_bytes + nk_attention_row_padding_cdna3_k);
    return rows + (values > rows ? values : rows);
}

/**
 *  @brief Validates the contract and launches the kernel for the depth's width, or the fallback
 *      past it or past the device's shared memory, with as many blocks as stay resident.
 *  @param[in] element_bytes Bytes of one input element: 2 or 1.
 *  @param[in] score_scale Undoes the power of two that converting Q and K puts on scores, or 1.
 *  @param[in] output_scale Undoes the power of two that converting V puts on the output, or 1.
 */
NUMKONG_INLINE nk_status_t nk_attention_launch_cdna3_(
    void const *narrow_kernel, void const *wide_kernel, void const *fallback_kernel, nk_size_t element_bytes,
    void const *queries, void const *packed, nk_f32_t *output, nk_f32_t *log_sum_exp, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride,
    nk_size_t output_stride, nk_f32_t scale, nk_f32_t score_scale, nk_f32_t output_scale, nk_size_t keys_before,
    nk_size_t keys_after, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    if (((nk_size_t)packed & 15) || (((nk_size_t)output | output_stride) & 3)) return nk_misaligned_k;
    if (key_value_head_count == 0 || head_count % key_value_head_count != 0) return nk_unexpected_dimensions_k;
    if (tasks_begin >= tasks_end || depth == 0) return nk_success_k;
    nk_attention_arguments_t arguments = nk_attention_arguments_init_simt_(
        queries, packed, output, log_sum_exp, head_count, key_value_head_count, depth, query_offsets, query_stride,
        output_stride, scale, score_scale, output_scale, keys_before, keys_after, tasks_begin, tasks_end);
    nk_attention_width_t const width = depth <= nk_attention_narrow_depth_cdna3_k ? nk_attention_width_128_k
                                                                                  : nk_attention_width_256_k;
    nk_size_t const width_depth = width == nk_attention_width_128_k ? nk_attention_narrow_depth_cdna3_k
                                                                    : nk_attention_wide_depth_cdna3_k;
    int shared_limit = 0;
    nk_status_t const status = nk_device_attribute_rocm_(hipDeviceAttributeMaxSharedMemoryPerBlock, &shared_limit,
                                                         stream);
    if (status != nk_success_k) return status;
    if (depth > nk_attention_wide_depth_cdna3_k ||
        nk_attention_shared_bytes_cdna3_(element_bytes, width_depth) > (nk_size_t)shared_limit)
        return nk_launch_resident_rocm_(fallback_kernel, nk_attention_threads_k, 0, 0, NUMKONG_SIZE_MAX, &arguments,
                                        stream);
    // The attribute takes the width's largest size, so no depth's size races a concurrent launch.
    return nk_launch_resident_rocm_(width == nk_attention_width_128_k ? narrow_kernel : wide_kernel,
                                    nk_attention_threads_k, nk_attention_shared_bytes_cdna3_(element_bytes, depth),
                                    nk_attention_shared_bytes_cdna3_(element_bytes, width_depth), NUMKONG_SIZE_MAX,
                                    &arguments, stream);
}

#pragma endregion Launch

#pragma region Fragments

/** Reads @p value from lane @p source of the whole 64-lane wavefront. */
NUMKONG_DEVICE nk_f32_t nk_shuffle_f32_cdna3_(nk_f32_t value, unsigned source) {
    return __shfl(value, (int)source, 64);
}

/** Slot of a panel's Vᵀ that row @p row of Sᵀ tile @p tile covers, so each lane's scores are the
 *  probabilities in the order its V operand reads them: 32 slots per P · V step in 8 per lane
 *  group for 16-bit weights. */
NUMKONG_DEVICE unsigned nk_attention_tile_slot_b16_cdna3_(unsigned tile, unsigned row) {
    return (tile >> 1) * 32 + (row >> 2) * 8 + (tile & 1) * 4 + (row & 3);
}

/** The slot for 8-bit weights: 64 slots in 16 per lane group. */
NUMKONG_DEVICE unsigned nk_attention_tile_slot_b8_cdna3_(unsigned tile, unsigned row) {
    return (row >> 2) * 16 + tile * 4 + (row & 3);
}

NUMKONG_DEVICE void nk_attention_weights_bf16_cdna3_(nk_f32_t const probabilities[16], nk_u32_t packed[4],
                                                     nk_f32_t *sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) {
        nk_u32_t const low = nk_f32_to_bf16_cdna3_(probabilities[word * 2]);
        nk_u32_t const high = nk_f32_to_bf16_cdna3_(probabilities[word * 2 + 1]);
        packed[word] = low | (high << 16);
        *sum += __uint_as_float(low << 16) + __uint_as_float(high << 16);
    }
}

/** U8 weights round(255 · p), the max-scoring position landing on 255, offset by −128 into I8 for
 *  the signed MFMA; the tile adds 128 · Σ V back. */
NUMKONG_DEVICE void nk_attention_weights_u8_cdna3_(nk_f32_t const probabilities[16], nk_u32_t packed[4],
                                                   nk_f32_t *sum) {
    nk_u32_t total = 0;
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) {
        nk_u32_t bytes = 0;
#pragma unroll
        for (unsigned index = 0; index < 4; ++index) {
            nk_u32_t const weight = (nk_u32_t)(probabilities[word * 4 + index] * 255.0f + 0.5f);
            bytes |= weight << (index * 8), total += weight;
        }
        packed[word] = bytes ^ 0x80808080u;
    }
    *sum += (nk_f32_t)total;
}

/** The sum @c nk_attention_weights_bf16_cdna3_ adds for a probability of one, the unit a row's
 *  weights sum counts in. */
NUMKONG_DEVICE nk_f32_t nk_attention_weight_unit_bf16_cdna3_(void) {
    nk_f32_t probabilities[16] = {1};
    nk_u32_t packed[4];
    nk_f32_t unit = 0;
    nk_attention_weights_bf16_cdna3_(probabilities, packed, &unit);
    return unit;
}

/** The sum @c nk_attention_weights_u8_cdna3_ adds for a probability of one. */
NUMKONG_DEVICE nk_f32_t nk_attention_weight_unit_u8_cdna3_(void) {
    nk_f32_t probabilities[16] = {1};
    nk_u32_t packed[4];
    nk_f32_t unit = 0;
    nk_attention_weights_u8_cdna3_(probabilities, packed, &unit);
    return unit;
}

#pragma endregion Fragments

#pragma region Tile

/** Where one block's work item sits: the steps and tiles its depth takes, the shared
 *  buffers that stage a panel, the packed planes it comes from, and what the band hides
 *  from each wave's rows. */
typedef struct {
    unsigned step_bytes, step_words, depth_padded;
    unsigned row_bytes, row_stride, value_stride;
    unsigned step_mask, tile_mask, store_mask;
    unsigned char *keys_shared, *values_shared, *queries_shared;
    unsigned char const *keys_plane, *values_plane;
    nk_size_t length, positions_padded;
    nk_diagonal_band_t band;
    unsigned tile_rows[2];
    nk_i64_t tile_first[2], lane_rows[2];
    unsigned panel_first, panel_end;
} nk_attention_frame_cdna3_t;

/**
 *  @brief Places one work item on one block: the depth steps of Sᵀ and the tiles of P · V its
 *      @p width and the head's depth take, its shared buffers, and the panels its rows see.
 *  @param[in] mask Whether panels crossing the band's edges mask their scores, see
 *      @c nk_attention_mask_t; panels past the segment's keys always do.
 *  @param[in] element_bytes Bytes of one input element: 2 for BF16, 1 for 8-bit codes.
 *  @param[in] step_bytes Bytes of depth one step of Sᵀ reads: 64, or 128 for the E4M3 MFMA.
 */
NUMKONG_DEVICE nk_attention_frame_cdna3_t nk_attention_frame_cdna3_(
    nk_attention_width_t width, nk_attention_mask_t mask, unsigned element_bytes, unsigned step_bytes,
    nk_attention_arguments_t const *arguments, nk_attention_work_t const *work, unsigned char *shared) {
    nk_attention_frame_cdna3_t frame;
    unsigned const lane = threadIdx.x & 63, wave = threadIdx.x >> 6, column = lane & 15;
    frame.step_bytes = step_bytes, frame.step_words = step_bytes / 16;
    unsigned const width_depth = width == nk_attention_width_128_k ? nk_attention_narrow_depth_cdna3_k
                                                                   : nk_attention_wide_depth_cdna3_k;
    unsigned const width_steps = (unsigned)nk_size_divide_round_up_(width_depth * element_bytes, step_bytes),
                   width_tiles = width_depth / 16;

    nk_size_t const depth = arguments->depth;
    frame.row_bytes = (unsigned)nk_size_round_up_to_multiple_(depth * element_bytes, nk_attention_step_bytes_k);
    frame.depth_padded = frame.row_bytes / element_bytes;
    frame.row_stride = frame.row_bytes + nk_attention_row_padding_cdna3_k;
    frame.value_stride = nk_attention_panel_k * element_bytes + nk_attention_row_padding_cdna3_k;
    unsigned const depth_steps = nk_u32_divide_round_up_(frame.row_bytes, step_bytes);
    // Loops run to the register arrays' fixed bounds and skip steps and tiles by these masks, as a
    // bound known only after inlining leaves them rolled and the arrays in scratch.
    unsigned const width_step_mask = (1u << width_steps) - 1;
    frame.step_mask = ((1u << depth_steps) - 1) & width_step_mask;
    unsigned const depth_tiles = (unsigned)nk_size_divide_round_up_(depth, 16),
                   width_tile_mask = (1u << width_tiles) - 1;
    frame.tile_mask = ((1u << depth_tiles) - 1) & width_tile_mask;
    frame.store_mask = (column < depth ? (1u << (unsigned)nk_size_divide_round_up_(depth - column, 16)) - 1 : 0) &
                       width_tile_mask;
    frame.keys_shared = shared, frame.values_shared = shared + nk_attention_panel_k * frame.row_stride;
    // Q is staged where V goes, and read into registers before the first V lands.
    frame.queries_shared = frame.values_shared;

    nk_size_t plane_bytes;
    frame.keys_plane = nk_attention_keys_plane_simt_(arguments, work, frame.row_bytes, &frame.length, &plane_bytes);
    frame.values_plane = frame.keys_plane + arguments->key_value_head_count * plane_bytes;
    frame.positions_padded = plane_bytes / frame.row_bytes;

    // Lane `column` holds row `column` of each query tile, whose 16 rows classify panels together.
    frame.band = nk_attention_kernel_band_simt_(mask, arguments);
#pragma unroll
    for (unsigned tile = 0; tile < 2; ++tile) {
        unsigned const tile_local = wave * nk_attention_wave_rows_cdna3_k + tile * 16;
        frame.tile_rows[tile] = work->row_count > tile_local ? min((unsigned)work->row_count - tile_local, 16u) : 0;
        frame.tile_first[tile] = nk_attention_row_position_simt_(work, tile_local, frame.length);
        frame.lane_rows[tile] = nk_attention_row_position_simt_(work, tile_local + column, frame.length);
    }
    nk_size_t block_begin, block_end;
    nk_attention_rows_keys_simt_(frame.band, nk_attention_row_position_simt_(work, 0, frame.length),
                                 nk_attention_row_position_simt_(work, work->row_count - 1, frame.length), frame.length,
                                 &block_begin, &block_end);
    frame.panel_first = (unsigned)(block_begin / nk_attention_panel_k);
    frame.panel_end = block_begin < block_end ? (unsigned)nk_size_divide_round_up_(block_end, nk_attention_panel_k)
                                              : frame.panel_first;
    return frame;
}

/** Which tiles of a panel the band lets a wave's rows see: @p active ones, and of those the
 *  @p masked ones it cuts, whose lanes get the @p visible bits of their own row. */
NUMKONG_DEVICE void nk_attention_coverage_cdna3_(nk_attention_frame_cdna3_t const *frame,
                                                 nk_attention_work_t const *work, nk_size_t panel_position,
                                                 int active[2], int masked[2], nk_u32_t visible[2][2]) {
    unsigned const lane = threadIdx.x & 63, wave = threadIdx.x >> 6, column = lane & 15;
#pragma unroll
    for (unsigned tile = 0; tile < 2; ++tile) {
        nk_diagonal_band_coverage_t const coverage = nk_attention_tile_coverage_simt_(
            frame->band, frame->tile_first[tile], frame->tile_rows[tile], panel_position, nk_attention_panel_k,
            frame->length);
        active[tile] = coverage != nk_diagonal_band_outside_k;
        masked[tile] = coverage != nk_diagonal_band_inside_k;
        if (!masked[tile] || wave * nk_attention_wave_rows_cdna3_k + tile * 16 + column >= work->row_count) continue;
        visible[tile][0] = nk_diagonal_band_row_mask_simt_(frame->band, frame->lane_rows[tile], panel_position,
                                                           frame->length);
        visible[tile][1] = nk_diagonal_band_row_mask_simt_(frame->band, frame->lane_rows[tile], panel_position + 32,
                                                           frame->length);
    }
}

/** Scales a tile's raw scores to base 2, converting exact I32 ones first as @p epilogue says. */
NUMKONG_DEVICE void nk_attention_scale_cdna3_(nk_cross_epilogue_t epilogue, nk_fui32_t tile_scores[4][4],
                                              nk_f32_t scale2) {
#pragma unroll
    for (unsigned position_tile = 0; position_tile < 4; ++position_tile)
#pragma unroll
        for (unsigned element = 0; element < 4; ++element) {
            nk_fui32_t *score = &tile_scores[position_tile][element];
            score->f = (epilogue == nk_cross_epilogue_i32_to_f32_k ? (nk_f32_t)score->i : score->f) * scale2;
        }
}

/** Folds a tile's panel maxima into its row's @p row_max and rescales @p row_sum, returning what
 *  the exponentials subtract and the factor the output needs in @p correction. */
NUMKONG_DEVICE nk_f32_t nk_attention_statistics_cdna3_(nk_fui32_t const tile_scores[4][4], nk_f32_t *row_max,
                                                       nk_f32_t *row_sum, nk_f32_t *correction) {
    nk_f32_t const negative_infinity = nk_attention_negative_infinity_simt_();
    nk_f32_t chunk_max = negative_infinity;
#pragma unroll
    for (unsigned position_tile = 0; position_tile < 4; ++position_tile)
#pragma unroll
        for (unsigned element = 0; element < 4; ++element)
            chunk_max = fmaxf(chunk_max, tile_scores[position_tile][element].f);
    // The other 3 lane groups hold the rest of this lane's query row.
    chunk_max = fmaxf(chunk_max, nk_shuffle_xor_f32_rocm_(chunk_max, 16));
    chunk_max = fmaxf(chunk_max, nk_shuffle_xor_f32_rocm_(chunk_max, 32));
    nk_f32_t const new_max = fmaxf(*row_max, chunk_max);
    // Subtracting 0 while every key so far is masked keeps `exp2(-∞ - max)` from NaN.
    nk_f32_t const subtrahend = new_max == negative_infinity ? 0.0f : new_max;
    *correction = nk_f32_exp2_rocm_(*row_max - subtrahend);
    *row_max = new_max;
    *row_sum *= *correction;
    return subtrahend;
}

/** Writes the rows' outputs and log-sum-exps, @p unit being what the dtype's weights add for a
 *  probability of one. */
NUMKONG_DEVICE void nk_attention_finish_cdna3_(nk_attention_frame_cdna3_t const *frame,
                                               nk_fui32_t output[2][nk_attention_tiles_cdna3_k][4], nk_f32_t row_max[2],
                                               nk_f32_t row_sum[2], nk_f32_t unit,
                                               nk_attention_arguments_t const *arguments,
                                               nk_attention_work_t const *work) {
    unsigned const lane = threadIdx.x & 63, wave = threadIdx.x >> 6, group = lane >> 4, column = lane & 15;
#pragma unroll
    for (unsigned tile = 0; tile < 2; ++tile) {
        nk_f32_t total = row_sum[tile];
        total += nk_shuffle_xor_f32_rocm_(total, 16);
        total += nk_shuffle_xor_f32_rocm_(total, 32);
        // Lanes 0 to 15 hold one score row each, the rows the log-sum-exp needs.
        unsigned const score_local = wave * nk_attention_wave_rows_cdna3_k + tile * 16 + column;
        nk_f32_t *const log_sum_exp_slot = group == 0 && score_local < work->row_count
                                               ? nk_attention_log_sum_exp_slot_simt_(arguments, work, score_local)
                                               : NUMKONG_NULL;
        if (log_sum_exp_slot) *log_sum_exp_slot = nk_attention_log_sum_exp_simt_(row_max[tile], total, unit);
        nk_f32_t inverses[4];
#pragma unroll
        for (unsigned element = 0; element < 4; ++element) {
            nk_f32_t const sum = nk_shuffle_f32_cdna3_(total, group * 4 + element);
            inverses[element] = sum > 0 ? arguments->output_scale / sum : 0.0f;
        }
#pragma unroll
        for (unsigned element = 0; element < 4; ++element) {
            unsigned const local = wave * nk_attention_wave_rows_cdna3_k + tile * 16 + group * 4 + element;
            if (local >= work->row_count) continue;
            nk_f32_t *destination = nk_attention_output_row_simt_(arguments, work, local);
#pragma unroll
            for (unsigned depth_tile = 0; depth_tile < nk_attention_tiles_cdna3_k; ++depth_tile)
                if ((frame->store_mask >> depth_tile) & 1)
                    destination[depth_tile * 16 + column] = output[tile][depth_tile][element].f * inverses[element];
        }
    }
}

/** Reads this lane's Sᵀ operands of every query tile and depth step, which the panels reuse. */
NUMKONG_DEVICE void nk_attention_load_queries_cdna3_(nk_attention_frame_cdna3_t const *frame,
                                                     nk_u32_t queries[2][nk_attention_steps_cdna3_k][8]) {
    unsigned const lane = threadIdx.x & 63, wave = threadIdx.x >> 6, group = lane >> 4, column = lane & 15;
#pragma unroll
    for (unsigned tile = 0; tile < 2; ++tile) {
        unsigned char const *row = frame->queries_shared +
                                   (wave * nk_attention_wave_rows_cdna3_k + tile * 16 + column) * frame->row_stride;
#pragma unroll
        for (unsigned step = 0; step < nk_attention_steps_cdna3_k; ++step)
            if ((frame->step_mask >> step) & 1)
                nk_attention_load_fragment_cdna3_(row, step * frame->step_bytes + group * frame->step_words * 4,
                                                  frame->row_bytes, frame->step_words, queries[tile][step]);
    }
}

/** Dynamic shared memory of a launch, the panel buffers of every work item a block walks. */
NUMKONG_DEVICE unsigned char *nk_attention_dynamic_shared_cdna3_(void) {
    extern __shared__ __attribute__((aligned(16))) unsigned char nk_attention_shared_cdna3_[];
    return nk_attention_shared_cdna3_;
}

#pragma endregion Tile

#if NUMKONG_TARGET_CDNA3

/*  Each covers a 4-word step of a lane's fragments in two of MI300's MFMAs, the first reading words
 *  0 and 1, the second words 2 and 3, as A and B share each lane's depth or slot mapping. */
#pragma region Steps

NUMKONG_DEVICE void nk_attention_scores_bf16_cdna3_(nk_fui32_t scores[4], nk_u32_t const keys[8],
                                                    nk_u32_t const queries[8]) {
    nk_mfma_bf16_cdna3_(scores, keys, queries);
    nk_mfma_bf16_cdna3_(scores, keys + 2, queries + 2);
}

NUMKONG_DEVICE void nk_attention_scores_i8_cdna3_(nk_fui32_t scores[4], nk_u32_t const keys[8],
                                                  nk_u32_t const queries[8]) {
    nk_mfma_i8_cdna3_(scores, keys, queries);
    nk_mfma_i8_cdna3_(scores, keys + 2, queries + 2);
}

NUMKONG_DEVICE void nk_attention_values_bf16_cdna3_(nk_fui32_t output[4], nk_u32_t const probabilities[4],
                                                    nk_u32_t const values[4]) {
    nk_mfma_bf16_cdna3_(output, probabilities, values);
    nk_mfma_bf16_cdna3_(output, probabilities + 2, values + 2);
}

NUMKONG_DEVICE void nk_attention_values_i8_cdna3_(nk_fui32_t output[4], nk_u32_t const probabilities[4],
                                                  nk_u32_t const values[4]) {
    nk_mfma_i8_cdna3_(output, probabilities, values);
    nk_mfma_i8_cdna3_(output, probabilities + 2, values + 2);
}

#pragma endregion Steps

#pragma region Tile

/**
 *  @brief One BF16 work item on one block: Sᵀ, online softmax and P · V over every panel its rows
 *      see, then the output.
 *  @param[in] width The depths its registers hold, see @c nk_attention_width_t.
 *  @param[in] mask Whether panels crossing the band's edges mask their scores, see
 *      @c nk_attention_mask_t; panels past the segment's keys always do.
 *
 *  Lane l of wavefront w holds query row 32 × w + 16 × t + l % 16 of query tile t as the B operand
 *  of Sᵀ and the A operand of P · V, and rows 4 × (l / 16) + e of P · V's output, whose softmax
 *  corrections and sums it reads from the lanes holding those rows.
 */
NUMKONG_DEVICE void nk_attention_block_bf16_cdna3_(nk_attention_width_t width, nk_attention_mask_t mask,
                                                   nk_attention_arguments_t const *arguments,
                                                   nk_attention_work_t const *work, unsigned char *shared) {

    nk_f32_t const negative_infinity = nk_attention_negative_infinity_simt_();
    unsigned const lane = threadIdx.x & 63, group = lane >> 4, column = lane & 15;
    // Sᵀ reads 64 bytes of depth a step.
    nk_attention_frame_cdna3_t const frame = nk_attention_frame_cdna3_(width, mask, 2, 64, arguments, work, shared);
    nk_attention_stage_queries_b16_simt_(arguments, work, frame.queries_shared, nk_attention_block_rows_k,
                                         frame.row_stride, frame.depth_padded, 64);
    // Also retires the previous item's reads of every buffer this item refills.
    __syncthreads();

    nk_u32_t queries[2][nk_attention_steps_cdna3_k][8];
    nk_attention_load_queries_cdna3_(&frame, queries);
    // Every Q read retires before the first V lands on it.
    __syncthreads();

    nk_f32_t row_max[2] = {negative_infinity, negative_infinity}, row_sum[2] = {0, 0};
    nk_fui32_t output[2][nk_attention_tiles_cdna3_k][4];
#pragma unroll
    for (unsigned tile = 0; tile < 2; ++tile)
#pragma unroll
        for (unsigned depth_tile = 0; depth_tile < nk_attention_tiles_cdna3_k; ++depth_tile)
#pragma unroll
            for (unsigned element = 0; element < 4; ++element) output[tile][depth_tile][element].u = 0;
    nk_f32_t const scale2 = arguments->scale2 * arguments->score_scale;

    for (unsigned panel_index = frame.panel_first; panel_index < frame.panel_end; ++panel_index) {
        nk_size_t const panel_position = (nk_size_t)panel_index * nk_attention_panel_k;
        nk_attention_stage_rows_cdna3_(frame.keys_shared, frame.keys_plane, panel_position, nk_attention_panel_k,
                                       frame.row_bytes, frame.row_stride);
        nk_attention_stage_values_bf16_cdna3_(frame.values_shared, frame.values_plane, frame.positions_padded,
                                              panel_position, frame.row_bytes, frame.value_stride);
        __syncthreads();

        int active[2], masked[2];
        nk_u32_t visible[2][2] = {{0, 0}, {0, 0}};
        nk_attention_coverage_cdna3_(&frame, work, panel_position, active, masked, visible);

        nk_fui32_t tile_scores[2][4][4];
#pragma unroll
        for (unsigned tile = 0; tile < 2; ++tile)
#pragma unroll
            for (unsigned position_tile = 0; position_tile < 4; ++position_tile)
#pragma unroll
                for (unsigned element = 0; element < 4; ++element) tile_scores[tile][position_tile][element].u = 0;
#pragma unroll
        for (unsigned step = 0; step < nk_attention_steps_cdna3_k; ++step) {
            if (!((frame.step_mask >> step) & 1)) continue;
#pragma unroll
            for (unsigned position_tile = 0; position_tile < 4; ++position_tile) {
                unsigned const position = (unsigned)nk_attention_slot_position_simt_(
                    nk_attention_tile_slot_b16_cdna3_(position_tile, column));
                nk_u32_t keys[8];
                nk_attention_load_fragment_cdna3_(frame.keys_shared + position * frame.row_stride,
                                                  step * frame.step_bytes + group * frame.step_words * 4,
                                                  frame.row_bytes, frame.step_words, keys);
#pragma unroll
                for (unsigned tile = 0; tile < 2; ++tile)
                    if (active[tile])
                        nk_attention_scores_bf16_cdna3_(tile_scores[tile][position_tile], keys, queries[tile][step]);
            }
        }

        nk_u32_t probabilities[2][2][4];
        nk_f32_t correction[2] = {1.0f, 1.0f};
#pragma unroll
        for (unsigned tile = 0; tile < 2; ++tile) {
            nk_attention_scale_cdna3_(nk_cross_epilogue_f32_k, tile_scores[tile], scale2);
            if (masked[tile])
#pragma unroll
                for (unsigned position_tile = 0; position_tile < 4; ++position_tile)
#pragma unroll
                    for (unsigned element = 0; element < 4; ++element) {
                        unsigned const offset = (unsigned)nk_attention_slot_position_simt_(
                            nk_attention_tile_slot_b16_cdna3_(position_tile, group * 4 + element));
                        nk_u32_t const word = offset < 32 ? visible[tile][0] : visible[tile][1];
                        if (!((word >> (offset & 31)) & 1))
                            tile_scores[tile][position_tile][element].f = negative_infinity;
                    }
            nk_f32_t const subtrahend = nk_attention_statistics_cdna3_(tile_scores[tile], &row_max[tile],
                                                                       &row_sum[tile], &correction[tile]);
#pragma unroll
            for (unsigned value_step = 0; value_step < 2; ++value_step) {
                nk_f32_t step_probabilities[16];
#pragma unroll
                for (unsigned index = 0; index < 16; ++index) {
                    unsigned const position_tile = value_step * 2 + (index / 4 & 1);
                    step_probabilities[index] = nk_f32_exp2_rocm_(tile_scores[tile][position_tile][index % 4].f -
                                                                  subtrahend);
                }
                nk_attention_weights_bf16_cdna3_(step_probabilities, probabilities[tile][value_step], &row_sum[tile]);
            }
        }

        // This lane's 4 rows of P · V output take the corrections the lanes of those rows hold.
        nk_f32_t row_corrections[2][4];
#pragma unroll
        for (unsigned tile = 0; tile < 2; ++tile)
#pragma unroll
            for (unsigned element = 0; element < 4; ++element)
                row_corrections[tile][element] = nk_shuffle_f32_cdna3_(correction[tile], group * 4 + element);

#pragma unroll
        for (unsigned depth_tile = 0; depth_tile < nk_attention_tiles_cdna3_k; ++depth_tile) {
            if (!((frame.tile_mask >> depth_tile) & 1)) continue;
            unsigned char const *values_row = frame.values_shared + (depth_tile * 16 + column) * frame.value_stride;
#pragma unroll
            for (unsigned tile = 0; tile < 2; ++tile)
#pragma unroll
                for (unsigned element = 0; element < 4; ++element)
                    output[tile][depth_tile][element].f *= row_corrections[tile][element];
#pragma unroll
            for (unsigned value_step = 0; value_step < 2; ++value_step) {
                unsigned const slot = value_step * 32 + group * 8;
                nk_u32_t values[4];
                uint4 const halves = *(uint4 const *)(values_row + slot * 2);
                values[0] = halves.x, values[1] = halves.y, values[2] = halves.z, values[3] = halves.w;
#pragma unroll
                for (unsigned tile = 0; tile < 2; ++tile)
                    if (active[tile])
                        nk_attention_values_bf16_cdna3_(output[tile][depth_tile], probabilities[tile][value_step],
                                                        values);
            }
        }
        // Every read of the panel retires before the next one is staged over it.
        __syncthreads();
    }

    nk_attention_finish_cdna3_(&frame, output, row_max, row_sum, nk_attention_weight_unit_bf16_cdna3_(), arguments,
                               work);
}

/**
 *  @brief Every @c bf16 work item of a launch, walked with a stride of the grid.
 *  @sa nk_attention_block_bf16_cdna3_ for the parameters.
 */
NUMKONG_DEVICE void nk_attention_tile_bf16_cdna3_(nk_attention_width_t width, nk_attention_mask_t mask,
                                                  nk_attention_arguments_t const *arguments) {
    __shared__ nk_u64_t prefix[nk_attention_threads_k + 1];
    __shared__ nk_u64_t warp_totals[nk_attention_threads_k / 32];
    nk_attention_schedule_t schedule;
    if (!nk_attention_schedule_start_rocm_(arguments, &schedule, prefix, warp_totals)) return;
    nk_attention_work_t work;
    for (nk_size_t item = blockIdx.x; nk_attention_schedule_next_rocm_(&schedule, prefix, warp_totals, item, &work);
         item += gridDim.x)
        nk_attention_block_bf16_cdna3_(width, mask, arguments, &work, nk_attention_dynamic_shared_cdna3_());
}

/**
 *  @brief One I8 work item on one block: exact integer scores, U8 probabilities offset into I8, and
 *      P · V sums converted per panel.
 *  @sa nk_attention_block_bf16_cdna3_ for the parameters.
 */
NUMKONG_DEVICE void nk_attention_block_i8_cdna3_(nk_attention_width_t width, nk_attention_mask_t mask,
                                                 nk_attention_arguments_t const *arguments,
                                                 nk_attention_work_t const *work, unsigned char *shared) {

    nk_f32_t const negative_infinity = nk_attention_negative_infinity_simt_();
    unsigned const lane = threadIdx.x & 63, group = lane >> 4, column = lane & 15;
    // Sᵀ reads 64 bytes of depth a step.
    nk_attention_frame_cdna3_t const frame = nk_attention_frame_cdna3_(width, mask, 1, 64, arguments, work, shared);
    nk_attention_stage_queries_b8_simt_(arguments, work, frame.queries_shared, nk_attention_block_rows_k,
                                        frame.row_stride, frame.depth_padded, 64);
    // Also retires the previous item's reads of every buffer this item refills.
    __syncthreads();

    nk_u32_t queries[2][nk_attention_steps_cdna3_k][8];
    nk_attention_load_queries_cdna3_(&frame, queries);
    // Every Q read retires before the first V lands on it.
    __syncthreads();

    nk_f32_t row_max[2] = {negative_infinity, negative_infinity}, row_sum[2] = {0, 0};
    nk_fui32_t output[2][nk_attention_tiles_cdna3_k][4];
#pragma unroll
    for (unsigned tile = 0; tile < 2; ++tile)
#pragma unroll
        for (unsigned depth_tile = 0; depth_tile < nk_attention_tiles_cdna3_k; ++depth_tile)
#pragma unroll
            for (unsigned element = 0; element < 4; ++element) output[tile][depth_tile][element].u = 0;
    nk_f32_t const scale2 = arguments->scale2 * arguments->score_scale;

    for (unsigned panel_index = frame.panel_first; panel_index < frame.panel_end; ++panel_index) {
        nk_size_t const panel_position = (nk_size_t)panel_index * nk_attention_panel_k;
        nk_attention_stage_rows_cdna3_(frame.keys_shared, frame.keys_plane, panel_position, nk_attention_panel_k,
                                       frame.row_bytes, frame.row_stride);
        nk_attention_stage_values_b8_cdna3_(frame.values_shared, frame.values_plane, frame.positions_padded,
                                            panel_position, frame.row_bytes, frame.value_stride);
        __syncthreads();

        int active[2], masked[2];
        nk_u32_t visible[2][2] = {{0, 0}, {0, 0}};
        nk_attention_coverage_cdna3_(&frame, work, panel_position, active, masked, visible);

        nk_fui32_t tile_scores[2][4][4];
#pragma unroll
        for (unsigned tile = 0; tile < 2; ++tile)
#pragma unroll
            for (unsigned position_tile = 0; position_tile < 4; ++position_tile)
#pragma unroll
                for (unsigned element = 0; element < 4; ++element) tile_scores[tile][position_tile][element].u = 0;
#pragma unroll
        for (unsigned step = 0; step < nk_attention_steps_cdna3_k; ++step) {
            if (!((frame.step_mask >> step) & 1)) continue;
#pragma unroll
            for (unsigned position_tile = 0; position_tile < 4; ++position_tile) {
                unsigned const position = (unsigned)nk_attention_slot_position_simt_(
                    nk_attention_tile_slot_b8_cdna3_(position_tile, column));
                nk_u32_t keys[8];
                nk_attention_load_fragment_cdna3_(frame.keys_shared + position * frame.row_stride,
                                                  step * frame.step_bytes + group * frame.step_words * 4,
                                                  frame.row_bytes, frame.step_words, keys);
#pragma unroll
                for (unsigned tile = 0; tile < 2; ++tile)
                    if (active[tile])
                        nk_attention_scores_i8_cdna3_(tile_scores[tile][position_tile], keys, queries[tile][step]);
            }
        }

        nk_u32_t probabilities[2][2][4];
        nk_f32_t correction[2] = {1.0f, 1.0f};
#pragma unroll
        for (unsigned tile = 0; tile < 2; ++tile) {
            nk_attention_scale_cdna3_(nk_cross_epilogue_i32_to_f32_k, tile_scores[tile], scale2);
            if (masked[tile])
#pragma unroll
                for (unsigned position_tile = 0; position_tile < 4; ++position_tile)
#pragma unroll
                    for (unsigned element = 0; element < 4; ++element) {
                        unsigned const offset = (unsigned)nk_attention_slot_position_simt_(
                            nk_attention_tile_slot_b8_cdna3_(position_tile, group * 4 + element));
                        nk_u32_t const word = offset < 32 ? visible[tile][0] : visible[tile][1];
                        if (!((word >> (offset & 31)) & 1))
                            tile_scores[tile][position_tile][element].f = negative_infinity;
                    }
            nk_f32_t const subtrahend = nk_attention_statistics_cdna3_(tile_scores[tile], &row_max[tile],
                                                                       &row_sum[tile], &correction[tile]);
            // P · V takes 64 positions a step for 8-bit weights.
            nk_f32_t step_probabilities[16];
#pragma unroll
            for (unsigned index = 0; index < 16; ++index)
                step_probabilities[index] = nk_f32_exp2_rocm_(tile_scores[tile][index / 4][index % 4].f - subtrahend);
            nk_attention_weights_u8_cdna3_(step_probabilities, probabilities[tile][0], &row_sum[tile]);
        }

        // This lane's 4 rows of P · V output take the corrections the lanes of those rows hold.
        nk_f32_t row_corrections[2][4];
#pragma unroll
        for (unsigned tile = 0; tile < 2; ++tile)
#pragma unroll
            for (unsigned element = 0; element < 4; ++element)
                row_corrections[tile][element] = nk_shuffle_f32_cdna3_(correction[tile], group * 4 + element);

#pragma unroll
        for (unsigned depth_tile = 0; depth_tile < nk_attention_tiles_cdna3_k; ++depth_tile) {
            if (!((frame.tile_mask >> depth_tile) & 1)) continue;
            unsigned char const *values_row = frame.values_shared + (depth_tile * 16 + column) * frame.value_stride;
            nk_u32_t values[4];
            uint4 const codes = *(uint4 const *)(values_row + group * 16);
            values[0] = codes.x, values[1] = codes.y, values[2] = codes.z, values[3] = codes.w;
            // Σ V over the panel's 64 slots of this depth column restores the weights' offset.
            nk_i32_t values_sum = 0;
#pragma unroll
            for (unsigned word = 0; word < 4; ++word)
                values_sum = nk_dot_i8x4_cdna3_(values[word], 0x01010101u, values_sum);
            values_sum += nk_shuffle_xor_i32_rocm_(values_sum, 16);
            values_sum += nk_shuffle_xor_i32_rocm_(values_sum, 32);
#pragma unroll
            for (unsigned tile = 0; tile < 2; ++tile) {
                if (!active[tile]) continue;
                nk_fui32_t sums[4] = {{0}, {0}, {0}, {0}};
                nk_attention_values_i8_cdna3_(sums, probabilities[tile][0], values);
#pragma unroll
                for (unsigned element = 0; element < 4; ++element)
                    output[tile][depth_tile][element].f = fmaf(
                        output[tile][depth_tile][element].f, row_corrections[tile][element],
                        (nk_f32_t)(nk_i32_t)((nk_u32_t)sums[element].i + 128u * values_sum));
            }
        }
        // Every read of the panel retires before the next one is staged over it.
        __syncthreads();
    }

    nk_attention_finish_cdna3_(&frame, output, row_max, row_sum, nk_attention_weight_unit_u8_cdna3_(), arguments, work);
}

/**
 *  @brief Every @c i8 work item of a launch, walked with a stride of the grid.
 *  @sa nk_attention_block_i8_cdna3_ for the parameters.
 */
NUMKONG_DEVICE void nk_attention_tile_i8_cdna3_(nk_attention_width_t width, nk_attention_mask_t mask,
                                                nk_attention_arguments_t const *arguments) {
    __shared__ nk_u64_t prefix[nk_attention_threads_k + 1];
    __shared__ nk_u64_t warp_totals[nk_attention_threads_k / 32];
    nk_attention_schedule_t schedule;
    if (!nk_attention_schedule_start_rocm_(arguments, &schedule, prefix, warp_totals)) return;
    nk_attention_work_t work;
    for (nk_size_t item = blockIdx.x; nk_attention_schedule_next_rocm_(&schedule, prefix, warp_totals, item, &work);
         item += gridDim.x)
        nk_attention_block_i8_cdna3_(width, mask, arguments, &work, nk_attention_dynamic_shared_cdna3_());
}

/** The launch for BF16. */
NUMKONG_INLINE nk_status_t nk_attention_launch_bf16_cdna3_(
    void const *narrow_kernel, void const *wide_kernel, void const *fallback_kernel, void const *queries,
    void const *packed, nk_f32_t *output, nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count,
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale,
    nk_size_t keys_before, nk_size_t keys_after, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    return nk_attention_launch_cdna3_(narrow_kernel, wide_kernel, fallback_kernel, 2, queries, packed, output,
                                      log_sum_exp, head_count, key_value_head_count, depth, query_offsets, query_stride,
                                      output_stride, scale, 1.0f, 1.0f, keys_before, keys_after, tasks_begin, tasks_end,
                                      stream);
}

/** The launch for I8. */
NUMKONG_INLINE nk_status_t nk_attention_launch_i8_cdna3_(
    void const *narrow_kernel, void const *wide_kernel, void const *fallback_kernel, void const *queries,
    void const *packed, nk_f32_t *output, nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count,
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale,
    nk_size_t keys_before, nk_size_t keys_after, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    return nk_attention_launch_cdna3_(narrow_kernel, wide_kernel, fallback_kernel, 1, queries, packed, output,
                                      log_sum_exp, head_count, key_value_head_count, depth, query_offsets, query_stride,
                                      output_stride, scale, 1.0f, 1.0f, keys_before, keys_after, tasks_begin, tasks_end,
                                      stream);
}

#pragma endregion Tile

#pragma region Instantiations

nk_define_attention_pack_size_simt_(bf16, cdna3, 2)
nk_define_attention_packed_shape_rocm_(bf16, cdna3)
nk_define_attention_pack_rocm_(bf16, cdna3, bf16)
nk_define_attention_packed_simt_(bf16, cdna3, rocm)
nk_define_attention_backward_rocm_(bf16, cdna3, bf16)

nk_define_attention_pack_size_simt_(i8, cdna3, 1)
nk_define_attention_packed_shape_rocm_(i8, cdna3)
nk_define_attention_pack_rocm_(i8, cdna3, i8)
nk_define_attention_packed_simt_(i8, cdna3, rocm)

#pragma endregion Instantiations

#endif // NUMKONG_TARGET_CDNA3

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_ROCM_CDNA3_
#endif // NUMKONG_ARCH_ROCM_
#endif // NUMKONG_ATTENTION_CDNA3_CUH
