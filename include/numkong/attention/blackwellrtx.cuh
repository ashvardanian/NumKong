/**
 *  @file include/numkong/attention/blackwellrtx.cuh
 *  @author Ash Vardanian
 *  @date September 22, 2026
 *  @brief Ragged attention for the NVIDIA compute capability 12.x family.
 *
 *  @sa include/numkong/attention.h
 *  @sa include/numkong/attention/ampere.cuh
 *
 *  The Ampere tile with E4M3 going to the tensor cores as it is: S takes one
 *  `mma.m16n8k32.kind::f8f6f4` per 16 × 8 scores, at twice the F16 rate, and P is quantized to
 *  e4m3(256 · p) for the same instruction against the transposed V codes. The × 256 keeps every
 *  weight down to 2⁻¹⁴ of the row maximum in E4M3's normal range, and the row sum adds the
 *  dequantized weights, so 256 cancels in the normalization. BF16 and I8 reuse Ampere's kernels.
 */
#ifndef NUMKONG_ATTENTION_BLACKWELLRTX_CUH
#define NUMKONG_ATTENTION_BLACKWELLRTX_CUH

#if NUMKONG_ARCH_CUDA_
#if NUMKONG_TARGET_BLACKWELLRTX

#include "numkong/attention/ampere.cuh"
#include "numkong/attention/ada.cuh"     // `nk_attention_weights_e4m3_ada_`
#include "numkong/dots/blackwellrtx.cuh" // `nk_mma_e4m3_blackwellrtx_`

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Fragments

NUMKONG_DEVICE void nk_attention_scores_e4m3_blackwellrtx_(nk_fui32_t scores[2][4], nk_u32_t const query[8],
                                                           nk_u32_t const keys[4]) {
    nk_mma_e4m3_blackwellrtx_(scores[0], query, keys[0], keys[1]);
    nk_mma_e4m3_blackwellrtx_(scores[1], query, keys[2], keys[3]);
}

#pragma endregion Fragments

#pragma region Tile

/**
 *  @brief One E4M3 work item on one block, on native E4M3 MMAs with E4M3 probabilities.
 *  @sa nk_attention_block_bf16_ampere_ for the parameters.
 */
NUMKONG_DEVICE void nk_attention_block_e4m3_blackwellrtx_(nk_attention_width_t width, nk_attention_mask_t mask,
                                                          nk_attention_split_t split,
                                                          nk_attention_arguments_t const *arguments,
                                                          nk_attention_work_t const *work, unsigned char *shared) {

    nk_f32_t const negative_infinity = nk_attention_negative_infinity_simt_();
    unsigned const lane = threadIdx.x & 31;
    nk_attention_frame_ampere_t const frame = nk_attention_frame_ampere_(width, mask, split, 1, arguments, work,
                                                                         shared);
    // Retires the previous item's reads of every buffer this item refills.
    __syncthreads();

    if (frame.panel_first < frame.panel_end)
        nk_attention_stage_rows_ampere_(frame.keys_shared, frame.keys_plane, (nk_size_t)frame.panel_first * frame.panel,
                                        frame.panel, frame.row_bytes);
    nk_commit_async_ampere_();

    nk_attention_stage_queries_b8_simt_(arguments, work, frame.queries_shared, frame.block_rows, frame.row_stride,
                                        frame.depth_padded, nk_warp_lanes_cuda_());
    __syncthreads();

    nk_u32_t query_registers[nk_attention_query_steps_ampere_k][8];
    unsigned const query_row = frame.warp_row_base + (lane & 7) + ((lane >> 3) & 1) * 8;
    int const queries_in_registers = width == nk_attention_width_128_k;
    if (queries_in_registers) {
#pragma unroll
        for (unsigned step = 0; step < nk_attention_query_steps_ampere_k; ++step)
            if (step < frame.depth_steps) {
                nk_u32_t fragment[4];
                nk_load_matrices_x4_ampere_(
                    nk_shared_address_ampere_(frame.queries_shared + query_row * frame.row_stride + step * 32 +
                                              (lane >> 4) * 16),
                    fragment);
                nk_attention_query_copy_ampere_(fragment, query_registers[step]);
            }
    }

    nk_f32_t row_max[2] = {negative_infinity, negative_infinity}, row_sum[2] = {0, 0};
    nk_fui32_t output[nk_attention_wide_depth_ampere_k / 8][4];
#pragma unroll
    for (unsigned tile = 0; tile < frame.max_tiles; ++tile)
#pragma unroll
        for (unsigned element = 0; element < 4; ++element) output[tile][element].u = 0;
    nk_f32_t const scale2 = arguments->scale2 * arguments->score_scale;

    for (unsigned panel_index = frame.panel_first; panel_index < frame.panel_end; ++panel_index) {
        nk_wait_async_ampere_(0);
        // K has landed, and every warp is done with the V buffer this refills.
        __syncthreads();
        nk_size_t const panel_position = (nk_size_t)panel_index * frame.panel;
        nk_attention_stage_columns_ampere_(frame.values_shared, frame.values_plane, frame.positions_padded,
                                           panel_position, frame.panel, frame.depth_padded);
        nk_commit_async_ampere_();

        unsigned const chunk_begin = (unsigned)panel_position + frame.chunk_offset;
        nk_diagonal_band_coverage_t const coverage = nk_attention_tile_coverage_simt_(
            frame.band, frame.warp_first, frame.warp_rows, chunk_begin, frame.groups * nk_attention_group_ampere_k,
            frame.length);
        int const active = coverage != nk_diagonal_band_outside_k;
        nk_u32_t probabilities[4][4];
        nk_f32_t correction[2] = {1.0f, 1.0f};
        if (active) {
            nk_fui32_t tile_scores[8][4];
#pragma unroll
            for (unsigned tile = 0; tile < 2 * frame.groups; ++tile)
#pragma unroll
                for (unsigned element = 0; element < 4; ++element) tile_scores[tile][element].u = 0;
            unsigned const key_row = frame.chunk_offset + (lane & 7) + (lane >> 4) * 8;
            unsigned const key_column = ((lane >> 3) & 1) * 16;
            if (queries_in_registers) {
#pragma unroll
                for (unsigned step = 0; step < nk_attention_query_steps_ampere_k; ++step) {
                    if (step >= frame.depth_steps) continue;
#pragma unroll
                    for (unsigned position_group = 0; position_group < frame.groups; ++position_group) {
                        nk_u32_t keys[4];
                        nk_load_matrices_x4_ampere_(
                            nk_shared_address_ampere_(frame.keys_shared +
                                                      (key_row + position_group * 16) * frame.row_stride + step * 32 +
                                                      key_column),
                            keys);
                        nk_attention_scores_e4m3_blackwellrtx_(tile_scores + position_group * 2, query_registers[step],
                                                               keys);
                    }
                }
            }
            else {
                for (unsigned step = 0; step < frame.depth_steps; ++step) {
                    nk_u32_t fragment[4], query[8];
                    nk_load_matrices_x4_ampere_(
                        nk_shared_address_ampere_(frame.queries_shared + query_row * frame.row_stride + step * 32 +
                                                  (lane >> 4) * 16),
                        fragment);
                    nk_attention_query_copy_ampere_(fragment, query);
#pragma unroll
                    for (unsigned position_group = 0; position_group < frame.groups; ++position_group) {
                        nk_u32_t keys[4];
                        nk_load_matrices_x4_ampere_(
                            nk_shared_address_ampere_(frame.keys_shared +
                                                      (key_row + position_group * 16) * frame.row_stride + step * 32 +
                                                      key_column),
                            keys);
                        nk_attention_scores_e4m3_blackwellrtx_(tile_scores + position_group * 2, query, keys);
                    }
                }
            }

            nk_attention_softmax_ampere_(nk_cross_epilogue_f32_k, tile_scores, &frame, work, scale2, coverage,
                                         chunk_begin, row_max, row_sum, correction);
#pragma unroll
            for (unsigned position_group = 0; position_group < frame.groups; ++position_group)
#pragma unroll
                for (unsigned half = 0; half < 2; ++half) {
                    nk_f32_t const four[4] = {tile_scores[position_group * 2][half * 2].f,
                                              tile_scores[position_group * 2][half * 2 + 1].f,
                                              tile_scores[position_group * 2 + 1][half * 2].f,
                                              tile_scores[position_group * 2 + 1][half * 2 + 1].f};
                    nk_u32_t packed[2];
                    nk_attention_weights_e4m3_ada_(four, packed, &row_sum[half]);
                    probabilities[position_group >> 1][(position_group & 1) * 2 + half] = packed[0];
                }
            if (frame.groups == 1) probabilities[0][2] = probabilities[0][3] = 0;
            nk_attention_rescale_ampere_(output, frame.max_tiles, frame.depth_tiles, correction);
        }

        nk_wait_async_ampere_(0);
        // V has landed, and every warp is done with the K buffer this refills.
        __syncthreads();
        if (panel_index + 1 < frame.panel_end)
            nk_attention_stage_rows_ampere_(frame.keys_shared, frame.keys_plane, panel_position + frame.panel,
                                            frame.panel, frame.row_bytes);
        nk_commit_async_ampere_();
        if (!active) continue;

        if (frame.groups == 1) {
            // A 16-byte chunk per depth row: half a k32 step.
#pragma unroll
            for (unsigned quartet = 0; quartet < frame.max_tiles / 4; ++quartet) {
                if (quartet * 4 >= frame.depth_tiles) continue;
#pragma unroll
                for (unsigned position_group = 0; position_group < frame.groups; ++position_group) {
                    nk_u32_t values[4];
                    nk_load_matrices_x4_ampere_(
                        nk_shared_address_ampere_(frame.values_shared + (quartet * 32 + lane) * frame.value_stride +
                                                  frame.chunk_offset + position_group * 16),
                        values);
#pragma unroll
                    for (unsigned index = 0; index < 4; ++index)
                        nk_mma_e4m3_blackwellrtx_(output[quartet * 4 + index], probabilities[0], values[index], 0);
                }
            }
        }
        else {
#pragma unroll
            for (unsigned pair = 0; pair < frame.max_tiles / 2; ++pair) {
                if (pair * 2 >= frame.depth_tiles) continue;
                unsigned const value_row = pair * 16 + (lane & 7) + (lane >> 4) * 8;
                nk_u32_t values[2][4];
#pragma unroll
                for (unsigned step = 0; step < frame.groups / 2; ++step)
                    nk_load_matrices_x4_ampere_(
                        nk_shared_address_ampere_(frame.values_shared + value_row * frame.value_stride + step * 32 +
                                                  ((lane >> 3) & 1) * 16),
                        values[step]);
#pragma unroll
                for (unsigned step = 0; step < frame.groups / 2; ++step) {
                    nk_mma_e4m3_blackwellrtx_(output[pair * 2], probabilities[step], values[step][0], values[step][1]);
                    nk_mma_e4m3_blackwellrtx_(output[pair * 2 + 1], probabilities[step], values[step][2],
                                              values[step][3]);
                }
            }
        }
    }

    nk_attention_finish_ampere_(split, &frame, output, row_max, row_sum, nk_attention_weight_unit_e4m3_ada_(), shared,
                                arguments, work);
}

/**
 *  @brief Every E4M3 work item of a launch, walked with a stride of the grid, each on the split the
 *      row count of its block calls for.
 *  @sa nk_attention_block_e4m3_blackwellrtx_ for the parameters.
 */
NUMKONG_DEVICE void nk_attention_tile_e4m3_blackwellrtx_(nk_attention_width_t width, nk_attention_mask_t mask,
                                                         nk_attention_arguments_t const *arguments) {
    __shared__ nk_u64_t prefix[nk_attention_threads_k + 1];
    __shared__ nk_u64_t warp_totals[nk_attention_threads_k / 32];
    nk_attention_schedule_t schedule;
    if (!nk_attention_schedule_start_cuda_(arguments, &schedule, prefix, warp_totals)) return;
    nk_attention_work_t work;
    for (nk_size_t item = blockIdx.x; nk_attention_schedule_next_cuda_(&schedule, prefix, warp_totals, item, &work);
         item += gridDim.x) {
        // The whole block's rows pick the split, so a window that cuts the block splits it alike.
        nk_size_t const block_first = work.row_first / nk_attention_block_rows_k * nk_attention_block_rows_k;
        if (work.query_count * work.heads_selected - block_first <= nk_attention_warp_rows_ampere_k)
            nk_attention_block_e4m3_blackwellrtx_(width, mask, nk_attention_split_positions_k, arguments, &work,
                                                  nk_attention_dynamic_shared_ampere_());
        else
            nk_attention_block_e4m3_blackwellrtx_(width, mask, nk_attention_split_rows_k, arguments, &work,
                                                  nk_attention_dynamic_shared_ampere_());
    }
}

/** The launch for native E4M3 MMAs. */
NUMKONG_INLINE nk_status_t nk_attention_launch_e4m3_blackwellrtx_(
    void const *narrow_kernel, void const *wide_kernel, void const *fallback_kernel, void const *queries,
    void const *packed, nk_f32_t *output, nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count,
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale,
    nk_size_t keys_before, nk_size_t keys_after, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    return nk_attention_launch_ampere_(narrow_kernel, wide_kernel, fallback_kernel, 1, 1, queries, packed, output,
                                       log_sum_exp, head_count, key_value_head_count, depth, query_offsets,
                                       query_stride, output_stride, scale, 1.0f, 1.0f, keys_before, keys_after,
                                       tasks_begin, tasks_end, stream);
}

#pragma endregion Tile

#pragma region Instantiations

nk_define_attention_pack_size_simt_(e4m3, blackwellrtx, 1)
nk_define_attention_packed_shape_cuda_(e4m3, blackwellrtx)
nk_define_attention_pack_cuda_(e4m3, blackwellrtx, e4m3)
nk_define_attention_packed_simt_(e4m3, blackwellrtx, cuda)

#pragma endregion Instantiations

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_BLACKWELLRTX
#endif // NUMKONG_ARCH_CUDA_
#endif // NUMKONG_ATTENTION_BLACKWELLRTX_CUH
