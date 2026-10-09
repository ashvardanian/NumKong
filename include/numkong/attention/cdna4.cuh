/**
 *  @file include/numkong/attention/cdna4.cuh
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Ragged attention for AMD Instinct MI350, gfx950.
 *
 *  @sa include/numkong/attention.h
 *  @sa include/numkong/attention/cdna3.cuh
 *  @sa include/numkong/dots/cdna4.cuh
 *
 *  The CDNA3 kernel with MI350's wider MFMAs, one per 64-byte depth step and per P · V step: BF16
 *  multiplies BF16, E4M3 scores take the native Float8 MFMA and its P · V runs on F16 with V cast
 *  exactly, and I8 runs exact integer MFMA with U8 probabilities offset into I8. Depths above 256
 *  fall back to the @c rocm kernel. The pack layout and the work scheduler are the @c rocm
 *  capability's; the F16 conversions serve the CDNA5 kernel too.
 */
#ifndef NUMKONG_ATTENTION_CDNA4_CUH
#define NUMKONG_ATTENTION_CDNA4_CUH

#if NUMKONG_ARCH_ROCM_
#if NUMKONG_ARCH_ROCM_CDNA4_

#include "numkong/attention/cdna3.cuh" // `nk_attention_frame_cdna3_`, `nk_attention_launch_cdna3_`
#include "numkong/dots/cdna4.cuh"      // `nk_mfma_bf16_cdna4_`, `nk_byte_permute_rocm_`

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Conversions

NUMKONG_DEVICE nk_u32_t nk_f32_to_f16_cdna4_(nk_f32_t value) { return __half_as_ushort(__float2half_rn(value)); }

/** Four E4M3 codes as two pairs of F16 patterns scaled by 2⁻⁸: each code lands in the high byte
 *  of a half and keeps only its fields. */
NUMKONG_DEVICE void nk_e4m3x4_to_f16x4_cdna4_(nk_u32_t codes, nk_u32_t *low, nk_u32_t *high) {
    nk_u32_t const low_halves = nk_byte_permute_rocm_(0, codes, 0x010C000Cu);
    nk_u32_t const high_halves = nk_byte_permute_rocm_(0, codes, 0x030C020Cu);
    // The NaN magnitude 0x7F shifts to a finite 0x3F80, and setting 0x4000 fills its exponent.
    nk_u32_t const nan_bytes = ((codes & 0x7F7F7F7Fu) + 0x01010101u) & 0x80808080u;
    *low = ((low_halves >> 1) & 0x3F803F80u) | (low_halves & 0x80008000u) |
           (nk_byte_permute_rocm_(0, nan_bytes, 0x010C000Cu) >> 1);
    *high = ((high_halves >> 1) & 0x3F803F80u) | (high_halves & 0x80008000u) |
            (nk_byte_permute_rocm_(0, nan_bytes, 0x030C020Cu) >> 1);
}

#pragma endregion Conversions

#if NUMKONG_TARGET_CDNA4

#pragma region Steps

NUMKONG_DEVICE void nk_attention_scores_bf16_cdna4_(nk_fui32_t scores[4], nk_u32_t const keys[8],
                                                    nk_u32_t const queries[8]) {
    nk_mfma_bf16_cdna4_(scores, keys, queries);
}

NUMKONG_DEVICE void nk_attention_scores_e4m3_cdna4_(nk_fui32_t scores[4], nk_u32_t const keys[8],
                                                    nk_u32_t const queries[8]) {
    nk_mfma_e4m3_cdna4_(scores, keys, queries);
}

NUMKONG_DEVICE void nk_attention_scores_i8_cdna4_(nk_fui32_t scores[4], nk_u32_t const keys[8],
                                                  nk_u32_t const queries[8]) {
    nk_mfma_i8_cdna4_(scores, keys, queries);
}

NUMKONG_DEVICE void nk_attention_values_bf16_cdna4_(nk_fui32_t output[4], nk_u32_t const probabilities[4],
                                                    nk_u32_t const values[4]) {
    nk_mfma_bf16_cdna4_(output, probabilities, values);
}

/** F16 weights against 8 E4M3 codes of V converted to F16 over 256, which the output scale
 *  undoes. */
NUMKONG_DEVICE void nk_attention_values_e4m3_cdna4_(nk_fui32_t output[4], nk_u32_t const probabilities[4],
                                                    nk_u32_t const values[4]) {
    nk_u32_t halves[4];
    nk_e4m3x4_to_f16x4_cdna4_(values[0], &halves[0], &halves[1]);
    nk_e4m3x4_to_f16x4_cdna4_(values[1], &halves[2], &halves[3]);
    nk_mfma_f16_cdna4_(output, probabilities, halves);
}

NUMKONG_DEVICE void nk_attention_values_i8_cdna4_(nk_fui32_t output[4], nk_u32_t const probabilities[4],
                                                  nk_u32_t const values[4]) {
    nk_mfma_i8_cdna4_(output, probabilities, values);
}

NUMKONG_DEVICE void nk_attention_weights_f16_cdna4_(nk_f32_t const probabilities[16], nk_u32_t packed[4],
                                                    nk_f32_t *sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) {
        nk_u32_t const low = nk_f32_to_f16_cdna4_(probabilities[word * 2]);
        nk_u32_t const high = nk_f32_to_f16_cdna4_(probabilities[word * 2 + 1]);
        packed[word] = low | (high << 16);
        *sum += __half2float(__ushort_as_half((unsigned short)low)) +
                __half2float(__ushort_as_half((unsigned short)high));
    }
}

/** The sum @c nk_attention_weights_f16_cdna4_ adds for a probability of one. */
NUMKONG_DEVICE nk_f32_t nk_attention_weight_unit_f16_cdna4_(void) {
    nk_f32_t probabilities[16] = {1};
    nk_u32_t packed[4];
    nk_f32_t unit = 0;
    nk_attention_weights_f16_cdna4_(probabilities, packed, &unit);
    return unit;
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
NUMKONG_DEVICE void nk_attention_block_bf16_cdna4_(nk_attention_width_t width, nk_attention_mask_t mask,
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
                        nk_attention_scores_bf16_cdna4_(tile_scores[tile][position_tile], keys, queries[tile][step]);
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
                        nk_attention_values_bf16_cdna4_(output[tile][depth_tile], probabilities[tile][value_step],
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
 *  @sa nk_attention_block_bf16_cdna4_ for the parameters.
 */
NUMKONG_DEVICE void nk_attention_tile_bf16_cdna4_(nk_attention_width_t width, nk_attention_mask_t mask,
                                                  nk_attention_arguments_t const *arguments) {
    __shared__ nk_u64_t prefix[nk_attention_threads_k + 1];
    __shared__ nk_u64_t warp_totals[nk_attention_threads_k / 32];
    nk_attention_schedule_t schedule;
    if (!nk_attention_schedule_start_rocm_(arguments, &schedule, prefix, warp_totals)) return;
    nk_attention_work_t work;
    for (nk_size_t item = blockIdx.x; nk_attention_schedule_next_rocm_(&schedule, prefix, warp_totals, item, &work);
         item += gridDim.x)
        nk_attention_block_bf16_cdna4_(width, mask, arguments, &work, nk_attention_dynamic_shared_cdna3_());
}

/**
 *  @brief One E4M3 work item on one block: Float8 MFMA scores, P · V on F16 with V cast exactly.
 *  @sa nk_attention_block_bf16_cdna3_ for the parameters.
 */
NUMKONG_DEVICE void nk_attention_block_e4m3_cdna4_(nk_attention_width_t width, nk_attention_mask_t mask,
                                                   nk_attention_arguments_t const *arguments,
                                                   nk_attention_work_t const *work, unsigned char *shared) {

    nk_f32_t const negative_infinity = nk_attention_negative_infinity_simt_();
    unsigned const lane = threadIdx.x & 63, group = lane >> 4, column = lane & 15;
    // Sᵀ reads 128 bytes of depth a step for the E4M3 MFMA.
    nk_attention_frame_cdna3_t const frame = nk_attention_frame_cdna3_(width, mask, 1, 128, arguments, work, shared);
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
                    nk_attention_tile_slot_b16_cdna3_(position_tile, column));
                nk_u32_t keys[8];
                nk_attention_load_fragment_cdna3_(frame.keys_shared + position * frame.row_stride,
                                                  step * frame.step_bytes + group * frame.step_words * 4,
                                                  frame.row_bytes, frame.step_words, keys);
#pragma unroll
                for (unsigned tile = 0; tile < 2; ++tile)
                    if (active[tile])
                        nk_attention_scores_e4m3_cdna4_(tile_scores[tile][position_tile], keys, queries[tile][step]);
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
                nk_attention_weights_f16_cdna4_(step_probabilities, probabilities[tile][value_step], &row_sum[tile]);
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
                uint2 const codes = *(uint2 const *)(values_row + slot);
                values[0] = codes.x, values[1] = codes.y, values[2] = 0, values[3] = 0;
#pragma unroll
                for (unsigned tile = 0; tile < 2; ++tile)
                    if (active[tile])
                        nk_attention_values_e4m3_cdna4_(output[tile][depth_tile], probabilities[tile][value_step],
                                                        values);
            }
        }
        // Every read of the panel retires before the next one is staged over it.
        __syncthreads();
    }

    nk_attention_finish_cdna3_(&frame, output, row_max, row_sum, nk_attention_weight_unit_f16_cdna4_(), arguments,
                               work);
}

/**
 *  @brief Every @c e4m3 work item of a launch, walked with a stride of the grid.
 *  @sa nk_attention_block_e4m3_cdna4_ for the parameters.
 */
NUMKONG_DEVICE void nk_attention_tile_e4m3_cdna4_(nk_attention_width_t width, nk_attention_mask_t mask,
                                                  nk_attention_arguments_t const *arguments) {
    __shared__ nk_u64_t prefix[nk_attention_threads_k + 1];
    __shared__ nk_u64_t warp_totals[nk_attention_threads_k / 32];
    nk_attention_schedule_t schedule;
    if (!nk_attention_schedule_start_rocm_(arguments, &schedule, prefix, warp_totals)) return;
    nk_attention_work_t work;
    for (nk_size_t item = blockIdx.x; nk_attention_schedule_next_rocm_(&schedule, prefix, warp_totals, item, &work);
         item += gridDim.x)
        nk_attention_block_e4m3_cdna4_(width, mask, arguments, &work, nk_attention_dynamic_shared_cdna3_());
}

/**
 *  @brief One I8 work item on one block: exact integer scores, U8 probabilities offset into I8, and
 *      P · V sums converted per panel.
 *  @sa nk_attention_block_bf16_cdna4_ for the parameters.
 */
NUMKONG_DEVICE void nk_attention_block_i8_cdna4_(nk_attention_width_t width, nk_attention_mask_t mask,
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
                        nk_attention_scores_i8_cdna4_(tile_scores[tile][position_tile], keys, queries[tile][step]);
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
                nk_attention_values_i8_cdna4_(sums, probabilities[tile][0], values);
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
 *  @sa nk_attention_block_i8_cdna4_ for the parameters.
 */
NUMKONG_DEVICE void nk_attention_tile_i8_cdna4_(nk_attention_width_t width, nk_attention_mask_t mask,
                                                nk_attention_arguments_t const *arguments) {
    __shared__ nk_u64_t prefix[nk_attention_threads_k + 1];
    __shared__ nk_u64_t warp_totals[nk_attention_threads_k / 32];
    nk_attention_schedule_t schedule;
    if (!nk_attention_schedule_start_rocm_(arguments, &schedule, prefix, warp_totals)) return;
    nk_attention_work_t work;
    for (nk_size_t item = blockIdx.x; nk_attention_schedule_next_rocm_(&schedule, prefix, warp_totals, item, &work);
         item += gridDim.x)
        nk_attention_block_i8_cdna4_(width, mask, arguments, &work, nk_attention_dynamic_shared_cdna3_());
}

/** The launch for BF16. */
NUMKONG_INLINE nk_status_t nk_attention_launch_bf16_cdna4_(
    void const *narrow_kernel, void const *wide_kernel, void const *fallback_kernel, void const *queries,
    void const *packed, nk_f32_t *output, nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count,
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_token_count, nk_size_t query_stride,
    nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_size_t tasks_begin,
    nk_size_t tasks_end, nk_stream_t stream) {
    return nk_attention_launch_cdna3_(narrow_kernel, wide_kernel, fallback_kernel, 2, queries, packed, output,
                                      log_sum_exp, head_count, key_value_head_count, depth, query_offsets,
                                      query_token_count, query_stride, output_stride, scale, 1.0f, 1.0f, keys_before,
                                      keys_after, tasks_begin, tasks_end, stream);
}

/** The launch for E4M3, whose V casts to F16 over 256. */
NUMKONG_INLINE nk_status_t nk_attention_launch_e4m3_cdna4_(
    void const *narrow_kernel, void const *wide_kernel, void const *fallback_kernel, void const *queries,
    void const *packed, nk_f32_t *output, nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count,
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_token_count, nk_size_t query_stride,
    nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_size_t tasks_begin,
    nk_size_t tasks_end, nk_stream_t stream) {
    return nk_attention_launch_cdna3_(narrow_kernel, wide_kernel, fallback_kernel, 1, queries, packed, output,
                                      log_sum_exp, head_count, key_value_head_count, depth, query_offsets,
                                      query_token_count, query_stride, output_stride, scale, 1.0f, 256.0f, keys_before,
                                      keys_after, tasks_begin, tasks_end, stream);
}

/** The launch for I8. */
NUMKONG_INLINE nk_status_t nk_attention_launch_i8_cdna4_(
    void const *narrow_kernel, void const *wide_kernel, void const *fallback_kernel, void const *queries,
    void const *packed, nk_f32_t *output, nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count,
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_token_count, nk_size_t query_stride,
    nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_size_t tasks_begin,
    nk_size_t tasks_end, nk_stream_t stream) {
    return nk_attention_launch_cdna3_(narrow_kernel, wide_kernel, fallback_kernel, 1, queries, packed, output,
                                      log_sum_exp, head_count, key_value_head_count, depth, query_offsets,
                                      query_token_count, query_stride, output_stride, scale, 1.0f, 1.0f, keys_before,
                                      keys_after, tasks_begin, tasks_end, stream);
}

#pragma endregion Tile

#pragma region Instantiations

nk_define_attention_pack_size_simt_(bf16, cdna4, 2)
nk_define_attention_packed_shape_rocm_(bf16, cdna4)
nk_define_attention_pack_rocm_(bf16, cdna4, bf16)
nk_define_attention_packed_simt_(bf16, cdna4, rocm)
nk_define_attention_backward_rocm_(bf16, cdna4, bf16)

nk_define_attention_pack_size_simt_(e4m3, cdna4, 1)
nk_define_attention_packed_shape_rocm_(e4m3, cdna4)
nk_define_attention_pack_rocm_(e4m3, cdna4, e4m3)
nk_define_attention_packed_simt_(e4m3, cdna4, rocm)

nk_define_attention_pack_size_simt_(i8, cdna4, 1)
nk_define_attention_packed_shape_rocm_(i8, cdna4)
nk_define_attention_pack_rocm_(i8, cdna4, i8)
nk_define_attention_packed_simt_(i8, cdna4, rocm)

#pragma endregion Instantiations

#endif // NUMKONG_TARGET_CDNA4

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_ROCM_CDNA4_
#endif // NUMKONG_ARCH_ROCM_
#endif // NUMKONG_ATTENTION_CDNA4_CUH
