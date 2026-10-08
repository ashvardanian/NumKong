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
 *  K and V panels through `cp.async`, and keep a base-2 online softmax per row; blocks of at most
 *  16 rows split every panel across the warps instead. BF16 multiplies on BF16 MMA, E4M3 converts
 *  to F16 for F16 MMA, I8 runs exact integer MMA with U8 probabilities, and depths above 256 fall
 *  back to the @c cuda kernel. The pack layout and the work scheduler are the @c cuda capability's.
 *
 *  The BF16 backward is FlashAttention-2's on the same MMAs, recomputing P from the log-sum-exp:
 *  blocks of 64 keys accumulate dK and dV over the rows that see them, then blocks of 64 rows
 *  accumulate dQ over their keys, both in registers and without atomics, 128 gradient columns at a
 *  time: heads past 128 dimensions recompute S and dP for each slice. Heads past 256, or past the
 *  device's shared memory, run the @c cuda loops.
 */
#ifndef NUMKONG_ATTENTION_AMPERE_CUH
#define NUMKONG_ATTENTION_AMPERE_CUH

#if NUMKONG_ARCH_CUDA_
#if NUMKONG_ARCH_CUDA_AMPERE_

#include "numkong/attention/cuda.cuh" // `nk_attention_schedule_next_cuda_`, `nk_attention_fallback_cuda_`
#include "numkong/dots/ampere.cuh" // `nk_mma_bf16_ampere_`, `nk_load_matrices_x4_ampere_`, `nk_copy_b128_async_ampere_`
#include "numkong/cast/ampere.cuh" // `nk_f32x2_to_bf16x2_ampere_`, `nk_f32x2_to_f16x2_ampere_`

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Configuration

enum {
    nk_attention_warp_rows_ampere_k = 16,
    nk_attention_wide_panel_ampere_k = 32,
    nk_attention_group_ampere_k = 16,
    nk_attention_row_padding_ampere_k = 16,
    nk_attention_narrow_depth_ampere_k = 128,
    nk_attention_wide_depth_ampere_k = 256,
    nk_attention_query_steps_ampere_k = nk_attention_narrow_depth_ampere_k * 2 / nk_attention_step_bytes_k,
    nk_attention_padded_panel_ampere_k = nk_attention_panel_k + nk_attention_row_padding_ampere_k,
};

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

#pragma endregion Configuration

#pragma region Fragments

NUMKONG_DEVICE void nk_attention_scores_bf16_ampere_(nk_fui32_t scores[2][4], nk_u32_t const query[8],
                                                     nk_u32_t const keys[4]) {
    nk_mma_bf16_ampere_(scores[0], query, keys[0], keys[1]);
    nk_mma_bf16_ampere_(scores[1], query, keys[2], keys[3]);
}

NUMKONG_DEVICE void nk_attention_scores_f16_ampere_(nk_fui32_t scores[2][4], nk_u32_t const query[8],
                                                    nk_u32_t const keys[4]) {
    nk_mma_f16_ampere_(scores[0], query, keys[0], keys[1]);
    nk_mma_f16_ampere_(scores[1], query, keys[2], keys[3]);
}

/** Converts K to F16 as the E4M3 dots do, in the order @c nk_attention_query_ampere_ gives Q. */
NUMKONG_DEVICE void nk_attention_scores_e4m3_ampere_(nk_fui32_t scores[2][4], nk_u32_t const query[8],
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

NUMKONG_DEVICE void nk_attention_scores_i8_ampere_(nk_fui32_t scores[2][4], nk_u32_t const query[8],
                                                   nk_u32_t const keys[4]) {
    nk_mma_i8_ampere_(scores[0], query, keys[0], keys[1]);
    nk_mma_i8_ampere_(scores[1], query, keys[2], keys[3]);
}

/** Q's A fragment for one 32-byte depth step: as loaded when the MMAs take @p dtype, else E4M3
 *  codes converted once into two F16 fragments of 16 depths each. */
NUMKONG_DEVICE void nk_attention_query_ampere_(nk_dtype_t dtype, nk_dtype_t mma_dtype, nk_u32_t const fragment[4],
                                               nk_u32_t query[8]) {
    if (mma_dtype == dtype) {
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

NUMKONG_DEVICE void nk_attention_weights_bf16_ampere_(nk_f32_t const probabilities[4], nk_u32_t packed[2],
                                                      nk_f32_t *sum) {
    packed[0] = nk_f32x2_to_bf16x2_ampere_(probabilities[0], probabilities[1]);
    packed[1] = nk_f32x2_to_bf16x2_ampere_(probabilities[2], probabilities[3]);
    *sum += (__uint_as_float(packed[0] << 16) + __uint_as_float(packed[0] & 0xFFFF0000u)) +
            (__uint_as_float(packed[1] << 16) + __uint_as_float(packed[1] & 0xFFFF0000u));
}

NUMKONG_DEVICE void nk_attention_weights_f16_ampere_(nk_f32_t const probabilities[4], nk_u32_t packed[2],
                                                     nk_f32_t *sum) {
    packed[0] = nk_f32x2_to_f16x2_ampere_(probabilities[0], probabilities[1]);
    packed[1] = nk_f32x2_to_f16x2_ampere_(probabilities[2], probabilities[3]);
#pragma unroll
    for (unsigned word = 0; word < 2; ++word)
        *sum += __half2float(__ushort_as_half((unsigned short)(packed[word] & 0xFFFFu))) +
                __half2float(__ushort_as_half((unsigned short)(packed[word] >> 16)));
}

/** The sum @p weights adds for a probability of one, the unit a row's weights sum counts in. */
NUMKONG_DEVICE nk_f32_t nk_attention_weight_unit_ampere_(nk_attention_weights_ampere_t weights) {
    nk_f32_t const probabilities[4] = {1, 0, 0, 0};
    nk_u32_t packed[2];
    nk_f32_t unit = 0;
    weights(probabilities, packed, &unit);
    return unit;
}

/** U8 weights round(255 · p), the max-scoring position landing on 255. */
NUMKONG_DEVICE void nk_attention_weights_u8_ampere_(nk_f32_t const probabilities[4], nk_u32_t packed[2],
                                                    nk_f32_t *sum) {
    nk_u32_t bytes = 0, total = 0;
#pragma unroll
    for (unsigned index = 0; index < 4; ++index) {
        // A round-down add of 2²³ truncates at the full F32 rate; sm_103 converts at 2 per clock
        nk_u32_t const weight = __float_as_uint(__fadd_rd(probabilities[index] * 255.0f + 0.5f, 8388608.0f)) -
                                0x4B000000u;
        bytes |= weight << (index * 8), total += weight;
    }
    packed[0] = bytes, packed[1] = 0;
    *sum += (nk_f32_t)total;
}

#pragma endregion Fragments

#pragma region Tile

/** Issues @p rows × @p row_bytes bytes of a position-major plane from @p first_position into
 *  padded shared rows. */
NUMKONG_DEVICE void nk_attention_stage_rows_ampere_(unsigned char *shared, unsigned char const *plane,
                                                    nk_size_t first_position, unsigned rows, unsigned row_bytes) {
    unsigned const chunks_per_row = row_bytes >> 4, chunks = rows * chunks_per_row;
    unsigned const row_stride = row_bytes + nk_attention_row_padding_ampere_k;
    for (unsigned chunk = threadIdx.x; chunk < chunks; chunk += nk_attention_threads_k) {
        unsigned const row = chunk / chunks_per_row, column = chunk - row * chunks_per_row;
        nk_copy_b128_async_ampere_(nk_shared_address_ampere_(shared + row * row_stride + (column << 4)),
                                   plane + (first_position + row) * row_bytes + (column << 4), 16);
    }
}

/** Issues @p panel slots of every depth row of a transposed V plane from @p first_position. */
NUMKONG_DEVICE void nk_attention_stage_columns_ampere_(unsigned char *shared, unsigned char const *plane,
                                                       nk_size_t positions_padded, nk_size_t first_position,
                                                       unsigned panel, unsigned depth_rows) {
    unsigned const chunks_per_row = panel >> 4, chunks = depth_rows * chunks_per_row;
    unsigned const row_stride = panel + nk_attention_row_padding_ampere_k;
    for (unsigned chunk = threadIdx.x; chunk < chunks; chunk += nk_attention_threads_k) {
        unsigned const row = chunk / chunks_per_row, column = chunk - row * chunks_per_row;
        nk_copy_b128_async_ampere_(nk_shared_address_ampere_(shared + row * row_stride + (column << 4)),
                                   plane + row * positions_padded + first_position + (column << 4), 16);
    }
}

/**
 *  @brief One work item on one block: scores, online softmax and P · V over every panel its rows
 *      see, then the output.
 *  @param[in] dtype The input dtype: @c nk_bf16_k, @c nk_f16_k, @c nk_e4m3_k or @c nk_i8_k.
 *  @param[in] mma_dtype The dtype the MMAs consume: @c nk_f16_k for E4M3 converted first, else
 *      @p dtype.
 *  @param[in] width Panel width and where Q fragments live, see @c nk_attention_width_t.
 *  @param[in] mask Whether chunks crossing the band's edges mask their scores, see
 *      @c nk_attention_mask_t; chunks past the segment's keys always do.
 *  @param[in] split Whether warps own rows or positions, see @c nk_attention_split_t.
 *  @param[in] epilogue F32 scores and P · V sums, or exact I32 ones converted per panel.
 *  @param[in] scores One depth step of S, see @c nk_attention_scores_ampere_t.
 *  @param[in] values_mma One 16 × 8 step of P · V on the packed P and V fragments.
 *  @param[in] weights P from probabilities, see @c nk_attention_weights_ampere_t.
 */
NUMKONG_DEVICE void nk_attention_block_ampere_(nk_dtype_t dtype, nk_dtype_t mma_dtype, nk_attention_width_t width,
                                               nk_attention_mask_t mask, nk_attention_split_t split,
                                               nk_cross_epilogue_t epilogue, nk_attention_scores_ampere_t scores,
                                               nk_cross_mma_ampere_t values_mma, nk_attention_weights_ampere_t weights,
                                               nk_attention_arguments_t const *arguments,
                                               nk_attention_work_t const *work, unsigned char *shared) {

    nk_f32_t const negative_infinity = nk_attention_negative_infinity_();
    unsigned const lane = threadIdx.x & 31, warp = threadIdx.x >> 5, group = lane >> 2, quad = lane & 3;
    unsigned const element_bytes = nk_attention_element_bytes_(dtype);
    unsigned const panel = split == nk_attention_split_positions_k || width == nk_attention_width_128_k
                               ? nk_attention_panel_k
                               : nk_attention_wide_panel_ampere_k;
    unsigned const groups = split == nk_attention_split_positions_k ? 1 : panel / nk_attention_group_ampere_k;
    unsigned const max_tiles = width == nk_attention_width_128_k ? nk_attention_narrow_depth_ampere_k / 8
                                                                 : nk_attention_wide_depth_ampere_k / 8;
    unsigned const warp_row_base = split == nk_attention_split_rows_k ? warp * nk_attention_warp_rows_ampere_k : 0;
    unsigned const chunk_offset = split == nk_attention_split_positions_k ? warp * nk_attention_group_ampere_k : 0;
    unsigned const block_rows = split == nk_attention_split_rows_k ? nk_attention_block_rows_k
                                                                   : nk_attention_warp_rows_ampere_k;

    nk_size_t const depth = arguments->depth;
    unsigned const row_bytes = (unsigned)((depth * element_bytes + nk_attention_step_bytes_k - 1) &
                                          ~(nk_size_t)(nk_attention_step_bytes_k - 1));
    unsigned const row_stride = row_bytes + nk_attention_row_padding_ampere_k;
    unsigned const depth_steps = row_bytes / nk_attention_step_bytes_k;
    unsigned const depth_padded = row_bytes / element_bytes, depth_tiles = depth_padded / 8;
    unsigned const value_stride = element_bytes == 2 ? row_stride : panel + nk_attention_row_padding_ampere_k;
    unsigned char *keys_shared = shared + arguments->key_offset[split];
    unsigned char *values_shared = shared + arguments->value_offset[split];
    unsigned char *queries_shared = shared + arguments->query_offset[split];

    nk_size_t length, plane_bytes;
    unsigned char const *keys_plane = nk_attention_keys_plane_(arguments, work, row_bytes, &length, &plane_bytes);
    unsigned char const *values_plane = keys_plane + arguments->key_value_head_count * plane_bytes;
    nk_size_t const positions_padded = plane_bytes / row_bytes;

    // A thread holds rows `group` and `group + 8` of its warp's 16, which classify chunks jointly.
    nk_diagonal_band_t const band = nk_attention_kernel_band_(mask, arguments);
    unsigned const warp_rows = work->row_count > warp_row_base ? min((unsigned)work->row_count - warp_row_base, 16u)
                                                               : 0;
    nk_i64_t const warp_first = nk_attention_row_position_(work, warp_row_base, length);
    nk_size_t block_begin, block_end;
    nk_attention_rows_keys_(band, nk_attention_row_position_(work, 0, length),
                            nk_attention_row_position_(work, work->row_count - 1, length), length, &block_begin,
                            &block_end);
    unsigned const panel_first = (unsigned)(block_begin / panel);
    unsigned const panel_end = block_begin < block_end ? (unsigned)nk_size_divide_round_up_(block_end, panel)
                                                       : panel_first;
    // Retires the previous item's reads of every buffer this item refills.
    __syncthreads();

    if (panel_first < panel_end)
        nk_attention_stage_rows_ampere_(keys_shared, keys_plane, (nk_size_t)panel_first * panel, panel, row_bytes);
    nk_commit_async_ampere_();

    nk_attention_stage_queries_(dtype, arguments, work, queries_shared, block_rows, row_stride, depth_padded,
                                nk_warp_lanes_cuda_());
    __syncthreads();

    nk_u32_t query_registers[nk_attention_query_steps_ampere_k][8];
    unsigned const query_row = warp_row_base + (lane & 7) + ((lane >> 3) & 1) * 8;
    int const queries_in_registers = width == nk_attention_width_128_k && mma_dtype == dtype;
    // E4M3 codes convert to F16 pairs, twice the size, so they are reread, as wide heads do.
    if (queries_in_registers) {
#pragma unroll
        for (unsigned step = 0; step < nk_attention_query_steps_ampere_k; ++step)
            if (step < depth_steps) {
                nk_u32_t fragment[4];
                nk_load_matrices_x4_ampere_(
                    nk_shared_address_ampere_(queries_shared + query_row * row_stride + step * 32 + (lane >> 4) * 16),
                    fragment);
                nk_attention_query_ampere_(dtype, mma_dtype, fragment, query_registers[step]);
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
        if (element_bytes == 2)
            nk_attention_stage_rows_ampere_(values_shared, values_plane, panel_position, panel, row_bytes);
        else
            nk_attention_stage_columns_ampere_(values_shared, values_plane, positions_padded, panel_position, panel,
                                               depth_padded);
        nk_commit_async_ampere_();

        unsigned const chunk_begin = (unsigned)panel_position + chunk_offset;
        unsigned const chunk_keys = groups * nk_attention_group_ampere_k;
        nk_diagonal_band_coverage_t const coverage = nk_attention_tile_coverage_(band, warp_first, warp_rows,
                                                                                 chunk_begin, chunk_keys, length);
        int const active = coverage != nk_diagonal_band_outside_k;
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
                for (unsigned step = 0; step < nk_attention_query_steps_ampere_k; ++step) {
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
                    nk_attention_query_ampere_(dtype, mma_dtype, fragment, query);
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
            if (coverage == nk_diagonal_band_crossing_k) {
                nk_u32_t visible[2][2] = {{0, 0}, {0, 0}};
#pragma unroll
                for (unsigned half = 0; half < 2; ++half) {
                    unsigned const local = warp_row_base + group + half * 8;
                    if (local >= work->row_count) continue;
                    nk_i64_t const row = nk_attention_row_position_(work, local, length);
#pragma unroll
                    for (unsigned word = 0; word * 32 < chunk_keys; ++word)
                        visible[half][word] = nk_diagonal_band_row_mask_simt_(band, row, chunk_begin + word * 32,
                                                                              length);
                }
#pragma unroll
                for (unsigned tile = 0; tile < 2 * groups; ++tile)
#pragma unroll
                    for (unsigned element = 0; element < 4; ++element) {
                        unsigned const offset = tile * 8 + quad * 2 + (element & 1);
                        if (!((visible[element >> 1][offset >> 5] >> (offset & 31)) & 1))
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
                correction[half] = nk_f32_exp2_cuda_(row_max[half] - subtrahend[half]);
                row_max[half] = new_max;
                row_sum[half] *= correction[half];
            }
#pragma unroll
            for (unsigned tile = 0; tile < 2 * groups; ++tile)
#pragma unroll
                for (unsigned element = 0; element < 4; ++element)
                    tile_scores[tile][element].f = nk_f32_exp2_cuda_(tile_scores[tile][element].f -
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
                    if (element_bytes == 1 && mma_dtype == dtype)
                        probabilities[position_group >> 1][(position_group & 1) * 2 + half] = packed[0];
                    else
                        probabilities[position_group][half] = packed[0],
                        probabilities[position_group][half + 2] = packed[1];
                }
            if (element_bytes == 1 && mma_dtype == dtype && groups == 1) probabilities[0][2] = probabilities[0][3] = 0;
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

        if (element_bytes == 2) {
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
        else if (mma_dtype != dtype || groups == 1) {
            // A 16-byte chunk per depth row: both halves of a k16 step in F16, or half a k32 one.
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
                        if (mma_dtype != dtype) {
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
            own_factor[half] = nk_f32_exp2_cuda_(row_max[half] - subtrahend[half]);
            row_sum[half] *= own_factor[half];
            row_max[half] = merged_max[half];
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
                factor[half] = nk_f32_exp2_cuda_(statistics[((other * 2 + half) * 2 + 0) * 32 + lane] -
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
        nk_f32_t *destination = nk_attention_output_row_(arguments, work, local);
        nk_f32_t const inverse = row_sum[half] > 0 ? arguments->output_scale / row_sum[half] : 0.0f;
#pragma unroll
        for (unsigned tile = 0; tile < max_tiles; ++tile)
#pragma unroll
            for (unsigned element = 0; element < 2; ++element) {
                unsigned const column = tile * 8 + quad * 2 + element;
                if (column < depth) destination[column] = output[tile][half * 2 + element].f * inverse;
            }
        nk_f32_t *const log_sum_exp_slot = quad == 0 ? nk_attention_log_sum_exp_slot_(arguments, work, local)
                                                     : NUMKONG_NULL;
        if (log_sum_exp_slot)
            *log_sum_exp_slot = nk_attention_log_sum_exp_simt_(row_max[half], row_sum[half],
                                                               nk_attention_weight_unit_ampere_(weights));
    }
}

/**
 *  @brief Every work item of a launch, walked with a stride of the grid, each on the split the row
 *      count of its block calls for.
 *  @sa nk_attention_block_ampere_ for the parameters.
 */
NUMKONG_DEVICE void nk_attention_tile_mma_ampere_(nk_dtype_t dtype, nk_dtype_t mma_dtype, nk_attention_width_t width,
                                                  nk_attention_mask_t mask, nk_cross_epilogue_t epilogue,
                                                  nk_attention_scores_ampere_t scores, nk_cross_mma_ampere_t values_mma,
                                                  nk_attention_weights_ampere_t weights,
                                                  nk_attention_arguments_t const *arguments) {
    extern __shared__ __align__(128) unsigned char nk_attention_shared_ampere_[];
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
            nk_attention_block_ampere_(dtype, mma_dtype, width, mask, nk_attention_split_positions_k, epilogue, scores,
                                       values_mma, weights, arguments, &work, nk_attention_shared_ampere_);
        else
            nk_attention_block_ampere_(dtype, mma_dtype, width, mask, nk_attention_split_rows_k, epilogue, scores,
                                       values_mma, weights, arguments, &work, nk_attention_shared_ampere_);
    }
}

/** The tile on MMAs of the input dtype itself. */
NUMKONG_DEVICE void nk_attention_tile_ampere_(nk_dtype_t dtype, nk_attention_width_t width, nk_attention_mask_t mask,
                                              nk_cross_epilogue_t epilogue, nk_attention_scores_ampere_t scores,
                                              nk_cross_mma_ampere_t values_mma, nk_attention_weights_ampere_t weights,
                                              nk_attention_arguments_t const *arguments) {
    nk_attention_tile_mma_ampere_(dtype, dtype, width, mask, epilogue, scores, values_mma, weights, arguments);
}

/** The tile on F16 MMAs, for E4M3 converted to F16 first. */
NUMKONG_DEVICE void nk_attention_tile_ampere_f16_mma_(nk_dtype_t dtype, nk_attention_width_t width,
                                                      nk_attention_mask_t mask, nk_cross_epilogue_t epilogue,
                                                      nk_attention_scores_ampere_t scores,
                                                      nk_cross_mma_ampere_t values_mma,
                                                      nk_attention_weights_ampere_t weights,
                                                      nk_attention_arguments_t const *arguments) {
    nk_attention_tile_mma_ampere_(dtype, nk_f16_k, width, mask, epilogue, scores, values_mma, weights, arguments);
}

#pragma endregion Tile

#pragma region Backward

enum {
    // Deepest BF16 head the backward runs on tensor cores; deeper ones take the @c cuda kernels.
    nk_attention_backward_depth_ampere_k = 256,
    // Gradient columns a warp accumulates in registers at once; deeper heads take slices of them,
    // recomputing S and dP for each.
    nk_attention_backward_slice_ampere_k = 128,
};

/** Bytes of dynamic shared memory the launch gave this block, which tells the backward kernels
 *  whether their launcher picked the tensor-core path. */
NUMKONG_DEVICE nk_u32_t nk_dynamic_shared_bytes_ampere_(void) {
    nk_u32_t bytes;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(bytes));
    return bytes;
}

/** Per-row state of the folded rows a backward block holds, at most 128. */
typedef struct {
    nk_f32_t log_sum_exp2[128];

    /** D = dO · O, the softmax gradient's row offset. */
    nk_f32_t dot[128];
    unsigned key_begin[128];
    unsigned key_end[128];
} nk_attention_backward_rows_ampere_t;

/** Shared memory of a backward block: 64-row panels of K, V, Q and dO, then the rows' state. */
NUMKONG_INLINE nk_size_t nk_attention_backward_shared_ampere_(nk_size_t depth) {
    nk_size_t const row_stride = nk_size_round_up_to_multiple_(depth * sizeof(nk_bf16_t), nk_attention_step_bytes_k) +
                                 nk_attention_row_padding_ampere_k;
    return 4 * nk_attention_panel_k * row_stride + sizeof(nk_attention_backward_rows_ampere_t);
}

/**
 *  @brief Stages @p rows folded rows of @p task from @p first, its query heads interleaved per
 *      query: Q and dO as BF16 in @p chunks 16-byte chunks per row, zero past the head, and each
 *      row's state, the rows past the task seeing no keys.
 *  @param[in] block_bytes Bytes of a 128-byte column block of the 128-byte swizzle the chunks land
 *      in, or 0 for plain rows @p row_stride bytes apart.
 *  @param[in] warp This warp's index among the @p warps that stage.
 */
NUMKONG_DEVICE void nk_attention_backward_stage_rows_ampere_(
    nk_attention_backward_arguments_t const *arguments, nk_attention_backward_task_t const *task, nk_size_t first,
    unsigned rows, unsigned chunks, unsigned block_bytes, unsigned row_stride, unsigned char *queries_shared,
    unsigned char *gradients_shared, nk_attention_backward_rows_ampere_t *state, unsigned warp, unsigned warps) {
    // Each warp keeps four rows' loads in flight, a lane taking four adjacent elements of each row.
    enum { rows_in_flight = 4 };
    unsigned const lane = threadIdx.x & 31;
    nk_size_t const depth = arguments->depth, group = arguments->head_count / arguments->key_value_head_count;
    for (unsigned base = warp * rows_in_flight; base < rows; base += warps * rows_in_flight) {
        nk_size_t queries[rows_in_flight], tokens[rows_in_flight], heads[rows_in_flight];
        nk_u16_t const *query_rows[rows_in_flight];
        nk_f32_t const *output_rows[rows_in_flight], *gradient_rows[rows_in_flight];
        nk_f32_t dots[rows_in_flight];
        int lives[rows_in_flight];
#pragma unroll
        for (unsigned row = 0; row < rows_in_flight; ++row) {
            nk_size_t const folded = first + base + row;
            queries[row] = folded / group, lives[row] = base + row < rows && queries[row] < task->rows;
            tokens[row] = arguments->query_offsets[task->segment] + queries[row];
            heads[row] = task->key_value_head * group + folded % group;
            query_rows[row] = (nk_u16_t const *)(arguments->queries + tokens[row] * arguments->query_stride) +
                              heads[row] * depth;
            nk_size_t const offset = tokens[row] * arguments->output_stride;
            output_rows[row] = (nk_f32_t const *)((unsigned char const *)arguments->output + offset) +
                               heads[row] * depth;
            gradient_rows[row] = (nk_f32_t const *)((unsigned char const *)arguments->output_gradient + offset) +
                                 heads[row] * depth;
            dots[row] = 0;
        }
        for (unsigned step = 0; step < chunks * 8; step += 128) {
            unsigned const element_first = step + lane * 4;
            nk_u32_t query_words[rows_in_flight][2];
            nk_f32_t gradients[rows_in_flight][4];
#pragma unroll
            for (unsigned row = 0; row < rows_in_flight; ++row)
#pragma unroll
                for (unsigned index = 0; index < 4; ++index) {
                    nk_size_t const element = element_first + index;
                    int const inside = lives[row] && element < depth;
                    nk_u32_t const query = inside ? query_rows[row][element] : 0u;
                    gradients[row][index] = inside ? gradient_rows[row][element] : 0.0f;
                    dots[row] = fmaf(gradients[row][index], inside ? output_rows[row][element] : 0.0f, dots[row]);
                    if (index & 1) query_words[row][index / 2] |= query << 16;
                    else query_words[row][index / 2] = query;
                }
            if (element_first >= chunks * 8) continue;
#pragma unroll
            for (unsigned row = 0; row < rows_in_flight; ++row) {
                if (base + row >= rows) continue;
                unsigned const local = base + row, byte = element_first * 2;
                unsigned const destination = block_bytes ? nk_swizzled_panel_offset_(local, byte, 128, block_bytes)
                                                         : local * row_stride + byte;
                *(uint2 *)(queries_shared + destination) = make_uint2(query_words[row][0], query_words[row][1]);
                *(uint2 *)(gradients_shared + destination) = make_uint2(
                    nk_f32x2_to_bf16x2_ampere_(gradients[row][0], gradients[row][1]),
                    nk_f32x2_to_bf16x2_ampere_(gradients[row][2], gradients[row][3]));
            }
        }
#pragma unroll
        for (unsigned row = 0; row < rows_in_flight; ++row)
            for (unsigned offset = 16; offset != 0; offset >>= 1)
                dots[row] += __shfl_xor_sync(0xFFFFFFFFu, dots[row], offset);
        if (lane != 0) continue;
#pragma unroll
        for (unsigned row = 0; row < rows_in_flight; ++row) {
            if (base + row >= rows) continue;
            unsigned const local = base + row;
            nk_size_t key_begin = 0, key_end = 0;
            if (lives[row])
                nk_diagonal_band_row_range_simt_(arguments->band, task->first_band_row + (nk_i64_t)queries[row],
                                                 task->length, &key_begin, &key_end);
            state->log_sum_exp2[local] =
                lives[row]
                    ? arguments->log_sum_exp[tokens[row] * arguments->head_count + heads[row]] * NUMKONG_F32_LOG2E_
                    : 0.0f;
            state->dot[local] = dots[row];
            state->key_begin[local] = (unsigned)key_begin, state->key_end[local] = (unsigned)key_end;
        }
    }
}

/** The keys the folded rows from @p first to @p last see together. */
NUMKONG_DEVICE void nk_attention_backward_span_ampere_(nk_attention_backward_arguments_t const *arguments,
                                                       nk_attention_backward_task_t const *task, nk_size_t first,
                                                       nk_size_t last, nk_size_t *key_begin, nk_size_t *key_end) {
    nk_size_t const group = arguments->head_count / arguments->key_value_head_count;
    nk_attention_rows_keys_(arguments->band, task->first_band_row + (nk_i64_t)(first / group),
                            task->first_band_row + (nk_i64_t)(last / group), task->length, key_begin, key_end);
}

/** P = 2^(S · scale₂ − lse₂) and dS = P · (dP − D) of one 16 × 16 fragment pair, @p transposed when
 *  rows run along the fragments' columns, packed into BF16 A fragments; positions a row does not
 *  see weigh 0. */
NUMKONG_DEVICE void nk_attention_backward_weights_ampere_(nk_fui32_t const scores[2][4],
                                                          nk_fui32_t const weight_gradients[2][4],
                                                          nk_attention_backward_rows_ampere_t const *rows,
                                                          unsigned row_first, unsigned position_first, int transposed,
                                                          nk_f32_t scale2, nk_u32_t weights[4],
                                                          nk_u32_t score_gradients[4]) {
    unsigned const lane = threadIdx.x & 31, fragment_row = lane >> 2, quad = lane & 3;
    nk_f32_t weight[2][4], score_gradient[2][4];
#pragma unroll
    for (unsigned tile = 0; tile < 2; ++tile)
#pragma unroll
        for (unsigned element = 0; element < 4; ++element) {
            unsigned const across = fragment_row + (element >> 1) * 8, along = tile * 8 + quad * 2 + (element & 1);
            unsigned const local = row_first + (transposed ? along : across);
            unsigned const position = position_first + (transposed ? across : along);
            int const visible = position >= rows->key_begin[local] && position < rows->key_end[local];
            weight[tile][element] =
                visible ? nk_f32_exp2_cuda_(scores[tile][element].f * scale2 - rows->log_sum_exp2[local]) : 0.0f;
            score_gradient[tile][element] = weight[tile][element] *
                                            (weight_gradients[tile][element].f - rows->dot[local]);
        }
#pragma unroll
    for (unsigned tile = 0; tile < 2; ++tile)
#pragma unroll
        for (unsigned half = 0; half < 2; ++half) {
            weights[tile * 2 + half] = nk_f32x2_to_bf16x2_ampere_(weight[tile][half * 2], weight[tile][half * 2 + 1]);
            score_gradients[tile * 2 + half] = nk_f32x2_to_bf16x2_ampere_(score_gradient[tile][half * 2],
                                                                          score_gradient[tile][half * 2 + 1]);
        }
}

/** The key and value gradients of every task of the window, a block per 64 keys: each warp owns 16
 *  keys and walks the 64-row chunks of folded rows that see any of the block's keys, accumulating
 *  dV += Pᵀ · dO and dK += dSᵀ · Q in registers a slice of columns at a time and writing each of
 *  them once at the end. */
NUMKONG_DEVICE void nk_attention_backward_keys_ampere_(nk_attention_backward_arguments_t const *arguments) {
    extern __shared__ __align__(128) unsigned char nk_attention_shared_ampere_[];
    unsigned const lane = threadIdx.x & 31, warp = threadIdx.x >> 5, fragment_row = lane >> 2, quad = lane & 3;
    // `ldmatrix.x4` lanes: an A fragment or a transposed B pair, then a B pair.
    unsigned const a_row = (lane & 7) + ((lane >> 3) & 1) * 8, a_column = (lane >> 4) * 16;
    unsigned const b_row = (lane & 7) + (lane >> 4) * 8, b_column = ((lane >> 3) & 1) * 16;
    enum { max_tiles = nk_attention_backward_slice_ampere_k / 8 };

    nk_size_t const depth = arguments->depth, group = arguments->head_count / arguments->key_value_head_count;
    unsigned const row_bytes = (unsigned)nk_size_round_up_to_multiple_(depth * 2, nk_attention_step_bytes_k);
    unsigned const row_stride = row_bytes + nk_attention_row_padding_ampere_k;
    unsigned const depth_steps = row_bytes / nk_attention_step_bytes_k, depth_tiles = row_bytes / 16;
    unsigned const slices = nk_u32_divide_round_up_(depth_tiles, max_tiles);
    unsigned char *keys_shared = nk_attention_shared_ampere_;
    unsigned char *values_shared = keys_shared + nk_attention_panel_k * row_stride;
    unsigned char *queries_shared = values_shared + nk_attention_panel_k * row_stride;
    unsigned char *gradients_shared = queries_shared + nk_attention_panel_k * row_stride;
    nk_attention_backward_rows_ampere_t *rows =
        (nk_attention_backward_rows_ampere_t *)(gradients_shared + nk_attention_panel_k * row_stride);
    nk_size_t const gradient_floats = arguments->key_value_gradient_stride / sizeof(nk_f32_t);

    // Blocks take every grid-th 64-key block of the window's tasks in turn.
    nk_size_t segment_first = arguments->task_begin / arguments->key_value_head_count, items_before = 0;
    nk_size_t task_index, block;
    for (nk_size_t item = blockIdx.x; nk_attention_backward_next_cuda_(
             arguments, 1, nk_attention_panel_k, &segment_first, &items_before, item, &task_index, &block);
         item += gridDim.x) {
        nk_attention_backward_task_t const task = nk_attention_backward_task_(arguments, task_index, row_bytes);
        nk_size_t const folded_rows = group * task.rows;
        unsigned const key_first = (unsigned)block * nk_attention_panel_k;
        nk_wait_async_ampere_(0);
        // Every warp is done with the previous block's panels.
        __syncthreads();
        nk_attention_stage_rows_ampere_(keys_shared, task.keys_plane, key_first, nk_attention_panel_k, row_bytes);
        nk_attention_stage_rows_ampere_(values_shared, task.values_plane, key_first, nk_attention_panel_k, row_bytes);
        nk_commit_async_ampere_();

        for (unsigned slice = 0; slice < slices; ++slice) {
            unsigned const pair_first = slice * max_tiles / 2;
            nk_fui32_t key_gradient[max_tiles][4], value_gradient[max_tiles][4];
#pragma unroll
            for (unsigned tile = 0; tile < max_tiles; ++tile)
#pragma unroll
                for (unsigned element = 0; element < 4; ++element)
                    key_gradient[tile][element].u = 0, value_gradient[tile][element].u = 0;

            for (nk_size_t first = 0; first < folded_rows; first += nk_attention_panel_k) {
                nk_size_t const last =
                    (first + nk_attention_panel_k < folded_rows ? first + nk_attention_panel_k : folded_rows) - 1;
                nk_size_t span_begin, span_end;
                nk_attention_backward_span_ampere_(arguments, &task, first, last, &span_begin, &span_end);
                if (span_begin >= key_first + nk_attention_panel_k || span_end <= key_first) continue;
                // Every warp is done with the previous chunk's rows.
                __syncthreads();
                nk_attention_backward_stage_rows_ampere_(arguments, &task, first, nk_attention_panel_k, row_bytes / 16,
                                                         0, row_stride, queries_shared, gradients_shared, rows, warp,
                                                         blockDim.x >> 5);
                nk_wait_async_ampere_(0);
                __syncthreads();

                for (unsigned row_group = 0; row_group < nk_attention_panel_k / 16; ++row_group) {
                    if (first + row_group * 16 >= folded_rows) break;
                    // Sᵀ = K · Qᵀ and dPᵀ = V · dOᵀ over the full depth, keys on fragment rows.
                    nk_fui32_t scores[2][4], weight_gradients[2][4];
#pragma unroll
                    for (unsigned tile = 0; tile < 2; ++tile)
#pragma unroll
                        for (unsigned element = 0; element < 4; ++element)
                            scores[tile][element].u = 0, weight_gradients[tile][element].u = 0;
                    for (unsigned step = 0; step < depth_steps; ++step) {
                        unsigned const a_offset = (warp * 16 + a_row) * row_stride + step * 32 + a_column;
                        unsigned const b_offset = (row_group * 16 + b_row) * row_stride + step * 32 + b_column;
                        nk_u32_t keys[4], values[4], queries[4], gradients[4];
                        nk_load_matrices_x4_ampere_(nk_shared_address_ampere_(keys_shared + a_offset), keys);
                        nk_load_matrices_x4_ampere_(nk_shared_address_ampere_(values_shared + a_offset), values);
                        nk_load_matrices_x4_ampere_(nk_shared_address_ampere_(queries_shared + b_offset), queries);
                        nk_load_matrices_x4_ampere_(nk_shared_address_ampere_(gradients_shared + b_offset), gradients);
                        nk_mma_bf16_ampere_(scores[0], keys, queries[0], queries[1]);
                        nk_mma_bf16_ampere_(scores[1], keys, queries[2], queries[3]);
                        nk_mma_bf16_ampere_(weight_gradients[0], values, gradients[0], gradients[1]);
                        nk_mma_bf16_ampere_(weight_gradients[1], values, gradients[2], gradients[3]);
                    }
                    nk_u32_t weights[4], score_gradients[4];
                    nk_attention_backward_weights_ampere_(scores, weight_gradients, rows, row_group * 16,
                                                          key_first + warp * 16, 1, arguments->scale2, weights,
                                                          score_gradients);
                    // dV += Pᵀ · dO and dK += dSᵀ · Q over the 16 rows, for the slice columns.
#pragma unroll
                    for (unsigned pair = 0; pair < max_tiles / 2; ++pair) {
                        if ((pair_first + pair) * 2 >= depth_tiles) continue;
                        unsigned const offset = (row_group * 16 + a_row) * row_stride + (pair_first + pair) * 32 +
                                                a_column;
                        nk_u32_t gradients[4], queries[4];
                        nk_load_matrices_x4_transposed_ampere_(nk_shared_address_ampere_(gradients_shared + offset),
                                                               gradients);
                        nk_load_matrices_x4_transposed_ampere_(nk_shared_address_ampere_(queries_shared + offset),
                                                               queries);
                        nk_mma_bf16_ampere_(value_gradient[pair * 2], weights, gradients[0], gradients[1]);
                        nk_mma_bf16_ampere_(value_gradient[pair * 2 + 1], weights, gradients[2], gradients[3]);
                        nk_mma_bf16_ampere_(key_gradient[pair * 2], score_gradients, queries[0], queries[1]);
                        nk_mma_bf16_ampere_(key_gradient[pair * 2 + 1], score_gradients, queries[2], queries[3]);
                    }
                }
            }

#pragma unroll
            for (unsigned half = 0; half < 2; ++half) {
                unsigned const position = key_first + warp * 16 + fragment_row + half * 8;
                if (position >= task.length) continue;
                nk_size_t const first = (arguments->key_offsets[task.segment] + position) * gradient_floats +
                                        task.key_value_head * depth;
#pragma unroll
                for (unsigned tile = 0; tile < max_tiles; ++tile)
#pragma unroll
                    for (unsigned element = 0; element < 2; ++element) {
                        unsigned const column = (pair_first * 2 + tile) * 8 + quad * 2 + element;
                        if (column >= depth) continue;
                        arguments->key_gradient[first + column] = key_gradient[tile][half * 2 + element].f *
                                                                  arguments->scale;
                        arguments->value_gradient[first + column] = value_gradient[tile][half * 2 + element].f;
                    }
            }
        }
    }
}

/** The query gradients of every task of the window, a block per 64 folded rows: each warp owns 16
 *  rows and walks the 64-key panels they see, accumulating dQ += dS · K in registers a slice of
 *  columns at a time. */
NUMKONG_DEVICE void nk_attention_backward_queries_ampere_(nk_attention_backward_arguments_t const *arguments) {
    extern __shared__ __align__(128) unsigned char nk_attention_shared_ampere_[];
    unsigned const lane = threadIdx.x & 31, warp = threadIdx.x >> 5, fragment_row = lane >> 2, quad = lane & 3;
    unsigned const a_row = (lane & 7) + ((lane >> 3) & 1) * 8, a_column = (lane >> 4) * 16;
    unsigned const b_row = (lane & 7) + (lane >> 4) * 8, b_column = ((lane >> 3) & 1) * 16;
    enum { max_tiles = nk_attention_backward_slice_ampere_k / 8, position_groups = nk_attention_panel_k / 16 };

    nk_size_t const depth = arguments->depth, group = arguments->head_count / arguments->key_value_head_count;
    unsigned const row_bytes = (unsigned)nk_size_round_up_to_multiple_(depth * 2, nk_attention_step_bytes_k);
    unsigned const row_stride = row_bytes + nk_attention_row_padding_ampere_k;
    unsigned const depth_steps = row_bytes / nk_attention_step_bytes_k, depth_tiles = row_bytes / 16;
    unsigned const slices = nk_u32_divide_round_up_(depth_tiles, max_tiles);
    unsigned char *keys_shared = nk_attention_shared_ampere_;
    unsigned char *values_shared = keys_shared + nk_attention_panel_k * row_stride;
    unsigned char *queries_shared = values_shared + nk_attention_panel_k * row_stride;
    unsigned char *gradients_shared = queries_shared + nk_attention_panel_k * row_stride;
    nk_attention_backward_rows_ampere_t *rows =
        (nk_attention_backward_rows_ampere_t *)(gradients_shared + nk_attention_panel_k * row_stride);

    nk_size_t segment_first = arguments->task_begin / arguments->key_value_head_count, items_before = 0;
    nk_size_t task_index, chunk;
    for (nk_size_t item = blockIdx.x; nk_attention_backward_next_cuda_(
             arguments, 0, nk_attention_panel_k, &segment_first, &items_before, item, &task_index, &chunk);
         item += gridDim.x) {
        nk_attention_backward_task_t const task = nk_attention_backward_task_(arguments, task_index, row_bytes);
        nk_size_t const folded_rows = group * task.rows;
        nk_size_t const first = chunk * nk_attention_panel_k;
        nk_size_t const last =
            (first + nk_attention_panel_k < folded_rows ? first + nk_attention_panel_k : folded_rows) - 1;
        nk_size_t span_begin, span_end;
        nk_attention_backward_span_ampere_(arguments, &task, first, last, &span_begin, &span_end);
        unsigned const panel_first = (unsigned)(span_begin / nk_attention_panel_k);
        unsigned const panel_end = span_begin < span_end
                                       ? (unsigned)nk_size_divide_round_up_(span_end, nk_attention_panel_k)
                                       : panel_first;
        // Every warp is done with the previous chunk's rows and panels.
        __syncthreads();
        nk_attention_backward_stage_rows_ampere_(arguments, &task, first, nk_attention_panel_k, row_bytes / 16, 0,
                                                 row_stride, queries_shared, gradients_shared, rows, warp,
                                                 blockDim.x >> 5);
        int const warp_live = first + warp * 16 < folded_rows;

        for (unsigned slice = 0; slice < slices; ++slice) {
            unsigned const pair_first = slice * max_tiles / 2;
            nk_fui32_t query_gradient[max_tiles][4];
#pragma unroll
            for (unsigned tile = 0; tile < max_tiles; ++tile)
#pragma unroll
                for (unsigned element = 0; element < 4; ++element) query_gradient[tile][element].u = 0;

            for (unsigned panel_index = panel_first; panel_index < panel_end; ++panel_index) {
                unsigned const position_first = panel_index * nk_attention_panel_k;
                // The rows have landed, and every warp is done with the previous panels.
                __syncthreads();
                nk_attention_stage_rows_ampere_(keys_shared, task.keys_plane, position_first, nk_attention_panel_k,
                                                row_bytes);
                nk_attention_stage_rows_ampere_(values_shared, task.values_plane, position_first, nk_attention_panel_k,
                                                row_bytes);
                nk_commit_async_ampere_();
                nk_wait_async_ampere_(0);
                __syncthreads();
                if (!warp_live) continue;

                // S = Q · Kᵀ and dP = dO · Vᵀ over the full depth, rows on fragment rows.
                nk_fui32_t scores[2 * position_groups][4], weight_gradients[2 * position_groups][4];
#pragma unroll
                for (unsigned tile = 0; tile < 2 * position_groups; ++tile)
#pragma unroll
                    for (unsigned element = 0; element < 4; ++element)
                        scores[tile][element].u = 0, weight_gradients[tile][element].u = 0;
                for (unsigned step = 0; step < depth_steps; ++step) {
                    unsigned const a_offset = (warp * 16 + a_row) * row_stride + step * 32 + a_column;
                    nk_u32_t queries[4], gradients[4];
                    nk_load_matrices_x4_ampere_(nk_shared_address_ampere_(queries_shared + a_offset), queries);
                    nk_load_matrices_x4_ampere_(nk_shared_address_ampere_(gradients_shared + a_offset), gradients);
#pragma unroll
                    for (unsigned position_group = 0; position_group < position_groups; ++position_group) {
                        unsigned const b_offset = (position_group * 16 + b_row) * row_stride + step * 32 + b_column;
                        nk_u32_t keys[4], values[4];
                        nk_load_matrices_x4_ampere_(nk_shared_address_ampere_(keys_shared + b_offset), keys);
                        nk_load_matrices_x4_ampere_(nk_shared_address_ampere_(values_shared + b_offset), values);
                        nk_mma_bf16_ampere_(scores[position_group * 2], queries, keys[0], keys[1]);
                        nk_mma_bf16_ampere_(scores[position_group * 2 + 1], queries, keys[2], keys[3]);
                        nk_mma_bf16_ampere_(weight_gradients[position_group * 2], gradients, values[0], values[1]);
                        nk_mma_bf16_ampere_(weight_gradients[position_group * 2 + 1], gradients, values[2], values[3]);
                    }
                }
                // dQ += dS · K over the panel's positions, for the slice's columns.
#pragma unroll
                for (unsigned position_group = 0; position_group < position_groups; ++position_group) {
                    nk_u32_t weights[4], score_gradients[4];
                    nk_attention_backward_weights_ampere_(
                        scores + position_group * 2, weight_gradients + position_group * 2, rows, warp * 16,
                        position_first + position_group * 16, 0, arguments->scale2, weights, score_gradients);
#pragma unroll
                    for (unsigned pair = 0; pair < max_tiles / 2; ++pair) {
                        if ((pair_first + pair) * 2 >= depth_tiles) continue;
                        nk_u32_t keys[4];
                        nk_load_matrices_x4_transposed_ampere_(
                            nk_shared_address_ampere_(keys_shared + (position_group * 16 + a_row) * row_stride +
                                                      (pair_first + pair) * 32 + a_column),
                            keys);
                        nk_mma_bf16_ampere_(query_gradient[pair * 2], score_gradients, keys[0], keys[1]);
                        nk_mma_bf16_ampere_(query_gradient[pair * 2 + 1], score_gradients, keys[2], keys[3]);
                    }
                }
            }

#pragma unroll
            for (unsigned half = 0; half < 2; ++half) {
                nk_size_t const folded = first + warp * 16 + fragment_row + half * 8, query = folded / group;
                if (query >= task.rows) continue;
                nk_size_t const token = arguments->query_offsets[task.segment] + query;
                nk_size_t const head = task.key_value_head * group + folded % group;
                nk_f32_t *destination = (nk_f32_t *)((unsigned char *)arguments->query_gradient +
                                                     token * arguments->query_gradient_stride) +
                                        head * depth;
#pragma unroll
                for (unsigned tile = 0; tile < max_tiles; ++tile)
#pragma unroll
                    for (unsigned element = 0; element < 2; ++element) {
                        unsigned const column = (pair_first * 2 + tile) * 8 + quad * 2 + element;
                        if (column < depth)
                            destination[column] = query_gradient[tile][half * 2 + element].f * arguments->scale;
                    }
            }
        }
    }
}

#pragma endregion Backward

#pragma region Launch

/** Places the panel buffers of a width in dynamic shared memory, returning the bytes
 *  a block needs. */
NUMKONG_INLINE nk_size_t nk_attention_shared_layout_ampere_(nk_dtype_t dtype, nk_dtype_t mma_dtype,
                                                            nk_attention_width_t width, nk_size_t depth,
                                                            nk_attention_arguments_t *arguments) {
    nk_size_t const element_bytes = nk_attention_element_bytes_(dtype);
    nk_size_t const row_bytes = nk_size_round_up_to_multiple_(depth * element_bytes, nk_attention_step_bytes_k);
    nk_size_t const row_stride = row_bytes + nk_attention_row_padding_ampere_k,
                    depth_padded = row_bytes / element_bytes;
    nk_size_t const keys_narrow = nk_attention_panel_k * row_stride;
    nk_size_t const keys_wide = nk_attention_wide_panel_ampere_k * row_stride;
    nk_size_t const values_narrow = element_bytes == 2 ? keys_narrow
                                                       : depth_padded * nk_attention_padded_panel_ampere_k;
    nk_size_t const values_wide = element_bytes == 2 ? keys_wide
                                                     : depth_padded * (nk_attention_wide_panel_ampere_k +
                                                                       nk_attention_row_padding_ampere_k);
    nk_size_t const queries_block = nk_attention_block_rows_k * row_stride;
    nk_size_t const queries_warp = nk_attention_warp_rows_ampere_k * row_stride;
    nk_size_t const combine = 3 * (depth_padded / 8) * 4 * 32 * sizeof(nk_f32_t) + 3 * 2 * 2 * 32 * sizeof(nk_f32_t);
    nk_size_t pipeline;
    arguments->key_offset[nk_attention_split_rows_k] = arguments->key_offset[nk_attention_split_positions_k] = 0;
    if (width == nk_attention_width_128_k && mma_dtype == dtype) {
        // Q is staged in the V buffer, read into registers before the first V lands.
        arguments->value_offset[nk_attention_split_rows_k] = (nk_u32_t)keys_narrow;
        arguments->value_offset[nk_attention_split_positions_k] = (nk_u32_t)keys_narrow;
        arguments->query_offset[nk_attention_split_rows_k] = (nk_u32_t)keys_narrow;
        arguments->query_offset[nk_attention_split_positions_k] = (nk_u32_t)keys_narrow;
        pipeline = keys_narrow + (values_narrow > queries_block ? values_narrow : queries_block);
    }
    else if (width == nk_attention_width_128_k) {
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

/** Dynamic shared memory of a block at the deepest head @p width takes, which each of its kernels
 *  sets as its limit. */
NUMKONG_INLINE nk_size_t nk_attention_shared_ceiling_ampere_(nk_dtype_t dtype, nk_dtype_t mma_dtype,
                                                             nk_attention_width_t width) {
    nk_attention_arguments_t deepest;
    nk_size_t const depth = width == nk_attention_width_128_k ? nk_attention_narrow_depth_ampere_k
                                                              : nk_attention_wide_depth_ampere_k;
    return nk_attention_shared_layout_ampere_(dtype, mma_dtype, width, depth, &deepest);
}

/**
 *  @brief Validates the contract and launches the kernel for the depth's width with as many blocks
 *      as stay resident.
 *  @param[in] score_scale Undoes the power of two that converting Q and K puts on scores, or 1.
 *  @param[in] output_scale Undoes the power of two that converting V puts on the output, or 1.
 */
NUMKONG_INLINE nk_status_t nk_attention_launch_mma_ampere_(
    void const *narrow_kernel, void const *wide_kernel, void const *fallback_kernel, nk_dtype_t dtype,
    nk_dtype_t mma_dtype, void const *queries, void const *packed, nk_f32_t *output, nk_f32_t *log_sum_exp,
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale, nk_f32_t score_scale, nk_f32_t output_scale,
    nk_size_t keys_before, nk_size_t keys_after, nk_size_t task_begin, nk_size_t task_end, void *stream) {
    if (((nk_size_t)packed & 15) || (((nk_size_t)output | output_stride) & 3)) return nk_misaligned_k;
    if (key_value_head_count == 0 || head_count % key_value_head_count != 0) return nk_unexpected_dimensions_k;
    if (task_begin >= task_end || depth == 0) return nk_success_k;
    nk_attention_arguments_t arguments = nk_attention_arguments_init_(
        queries, packed, output, log_sum_exp, head_count, key_value_head_count, depth, query_offsets, query_stride,
        output_stride, scale, score_scale, output_scale, keys_before, keys_after, task_begin, task_end);
    if (depth > nk_attention_wide_depth_ampere_k)
        return nk_launch_resident_cuda_(fallback_kernel, nk_attention_threads_k, 0, 0, NUMKONG_SIZE_MAX, &arguments,
                                        stream);
    nk_attention_width_t const width = depth <= nk_attention_narrow_depth_ampere_k ? nk_attention_width_128_k
                                                                                   : nk_attention_width_256_k;
    nk_size_t const shared_bytes = nk_attention_shared_layout_ampere_(dtype, mma_dtype, width, depth, &arguments);
    return nk_launch_resident_cuda_(
        width == nk_attention_width_128_k ? narrow_kernel : wide_kernel, nk_attention_threads_k, shared_bytes,
        nk_attention_shared_ceiling_ampere_(dtype, mma_dtype, width), NUMKONG_SIZE_MAX, &arguments, stream);
}

/** The launch for the tile on MMAs of the input dtype itself. */
NUMKONG_INLINE nk_status_t nk_attention_launch_ampere_(
    void const *narrow_kernel, void const *wide_kernel, void const *fallback_kernel, nk_dtype_t dtype,
    void const *queries, void const *packed, nk_f32_t *output, nk_f32_t *log_sum_exp, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride,
    nk_size_t output_stride, nk_f32_t scale, nk_f32_t score_scale, nk_f32_t output_scale, nk_size_t keys_before,
    nk_size_t keys_after, nk_size_t task_begin, nk_size_t task_end, void *stream) {
    return nk_attention_launch_mma_ampere_(narrow_kernel, wide_kernel, fallback_kernel, dtype, dtype, queries, packed,
                                           output, log_sum_exp, head_count, key_value_head_count, depth, query_offsets,
                                           query_stride, output_stride, scale, score_scale, output_scale, keys_before,
                                           keys_after, task_begin, task_end, stream);
}

/** The launch for the tile on F16 MMAs, for E4M3 converted to F16 first. */
NUMKONG_INLINE nk_status_t nk_attention_launch_ampere_f16_mma_(
    void const *narrow_kernel, void const *wide_kernel, void const *fallback_kernel, nk_dtype_t dtype,
    void const *queries, void const *packed, nk_f32_t *output, nk_f32_t *log_sum_exp, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride,
    nk_size_t output_stride, nk_f32_t scale, nk_f32_t score_scale, nk_f32_t output_scale, nk_size_t keys_before,
    nk_size_t keys_after, nk_size_t task_begin, nk_size_t task_end, void *stream) {
    return nk_attention_launch_mma_ampere_(narrow_kernel, wide_kernel, fallback_kernel, dtype, nk_f16_k, queries,
                                           packed, output, log_sum_exp, head_count, key_value_head_count, depth,
                                           query_offsets, query_stride, output_stride, scale, score_scale, output_scale,
                                           keys_before, keys_after, task_begin, task_end, stream);
}

/** Launches the BF16 backward kernels of Ampere and later with as many blocks as stay resident,
 *  or, with no shared memory, a block per task of the @c cuda loops for heads too deep for the
 *  tensor cores or for the device's shared memory. */
NUMKONG_INLINE nk_status_t nk_attention_backward_launch_ampere_(void const *keys_kernel, void const *queries_kernel,
                                                                nk_attention_backward_arguments_t arguments,
                                                                void *stream) {
    nk_size_t const shared_bytes = nk_attention_backward_shared_ampere_(arguments.depth);
    int shared_limit = 0;
    nk_status_t const status = nk_device_attribute_cuda_(cudaDevAttrMaxSharedMemoryPerBlockOptin, &shared_limit,
                                                         stream);
    if (status != nk_success_k) return status;
    if (arguments.depth > nk_attention_backward_depth_ampere_k || shared_bytes > (nk_size_t)shared_limit)
        return nk_attention_backward_launch_cuda_(keys_kernel, queries_kernel, arguments, stream);
    return nk_attention_backward_launch_kernels_cuda_(NUMKONG_NULL, keys_kernel, queries_kernel, nk_attention_threads_k,
                                                      shared_bytes, shared_bytes, NUMKONG_SIZE_MAX, arguments, stream);
}

#pragma endregion Launch

/*  Later generations read the helpers above and emit only their own kernels. */
#if NUMKONG_TARGET_AMPERE

#pragma region Instantiations

nk_define_attention_pack_size_simt_(bf16, ampere, 2)
nk_define_attention_packed_shape_cuda_(bf16, ampere)
nk_define_attention_pack_cuda_(bf16, ampere, bf16)
nk_define_attention_packed_cuda_(bf16, ampere, ampere, nk_attention_launch_ampere_, bf16, nk_cross_epilogue_f32_k,
                                 nk_attention_scores_bf16_ampere_, nk_mma_bf16_ampere_,
                                 nk_attention_weights_bf16_ampere_, 1.0f, 1.0f)

/** Both BF16 backward kernels: tensor cores when the launch gives them shared memory, the
 *  @c cuda kernels' loops when it does not. */
static __global__ void __launch_bounds__(nk_attention_threads_k)
    nk_attention_backward_keys_bf16_ampere_kernel_(nk_attention_backward_arguments_t arguments) {
    if (nk_dynamic_shared_bytes_ampere_()) nk_attention_backward_keys_ampere_(&arguments);
    else nk_attention_backward_keys_cuda_(nk_bf16_k, &arguments);
}
static __global__ void __launch_bounds__(nk_attention_threads_k)
    nk_attention_backward_queries_bf16_ampere_kernel_(nk_attention_backward_arguments_t arguments) {
    if (nk_dynamic_shared_bytes_ampere_()) nk_attention_backward_queries_ampere_(&arguments);
    else nk_attention_backward_queries_cuda_(nk_bf16_k, &arguments);
}
nk_define_attention_backward_cuda_(bf16, ampere, bf16, nk_attention_backward_launch_ampere_)

nk_define_attention_pack_size_simt_(f16, ampere, 2)
nk_define_attention_packed_shape_cuda_(f16, ampere)
nk_define_attention_pack_cuda_(f16, ampere, f16)
nk_define_attention_packed_cuda_(f16, ampere, ampere, nk_attention_launch_ampere_, f16, nk_cross_epilogue_f32_k,
                                 nk_attention_scores_f16_ampere_, nk_mma_f16_ampere_, nk_attention_weights_f16_ampere_,
                                 1.0f, 1.0f)

nk_define_attention_pack_size_simt_(e4m3, ampere, 1)
nk_define_attention_packed_shape_cuda_(e4m3, ampere)
nk_define_attention_pack_cuda_(e4m3, ampere, e4m3)
nk_define_attention_packed_cuda_(e4m3, ampere, ampere_f16_mma, nk_attention_launch_ampere_f16_mma_, e4m3,
                                 nk_cross_epilogue_f32_k, nk_attention_scores_e4m3_ampere_, nk_mma_f16_ampere_,
                                 nk_attention_weights_f16_ampere_, 65536.0f, 256.0f)

nk_define_attention_pack_size_simt_(i8, ampere, 1)
nk_define_attention_packed_shape_cuda_(i8, ampere)
nk_define_attention_pack_cuda_(i8, ampere, i8)
nk_define_attention_packed_cuda_(i8, ampere, ampere, nk_attention_launch_ampere_, i8, nk_cross_epilogue_i32_to_f32_k,
                                 nk_attention_scores_i8_ampere_, nk_mma_u8i8_ampere_, nk_attention_weights_u8_ampere_,
                                 1.0f, 1.0f)

#pragma endregion Instantiations

#endif // NUMKONG_TARGET_AMPERE

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_AMPERE_
#endif // NUMKONG_ARCH_CUDA_
#endif // NUMKONG_ATTENTION_AMPERE_CUH
