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
 *  rows split every panel across the warps instead. BF16 multiplies on BF16 MMA, E4M3 converts to
 *  F16 for F16 MMA, I8 runs exact integer MMA with U8 probabilities, and depths above 256 fall back
 *  to the @c cuda kernel. The pack layout and the work scheduler are the @c cuda capability's.
 */
#ifndef NUMKONG_ATTENTION_AMPERE_CUH
#define NUMKONG_ATTENTION_AMPERE_CUH

#if NUMKONG_ARCH_CUDA_AMPERE_

#include "numkong/attention/simt.cuh" // `nk_attention_schedule_next_`, `nk_attention_fallback_`
#include "numkong/dots/ampere.cuh" // `nk_mma_bf16_ampere_`, `nk_load_matrices_x4_ampere_`, `nk_copy_b128_async_ampere_`

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

#pragma region Instructions

/* Rounds two F32 into a BF16 pair, @p low in the low half. */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
NUMKONG_DEVICE nk_u32_t nk_f32x2_to_bf16x2_ampere_(nk_f32_t low, nk_f32_t high) {
    nk_u32_t pair;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(pair) : "f"(high), "f"(low));
    return pair;
}

/* Rounds two F32 into an F16 pair, @p low in the low half. */
NUMKONG_DEVICE nk_u32_t nk_f32x2_to_f16x2_ampere_(nk_f32_t low, nk_f32_t high) {
    nk_u32_t pair;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(pair) : "f"(high), "f"(low));
    return pair;
}

#else

NUMKONG_DEVICE nk_u32_t nk_f32x2_to_bf16x2_ampere_(nk_f32_t low, nk_f32_t high) {
    __trap();
    return 0;
}
NUMKONG_DEVICE nk_u32_t nk_f32x2_to_f16x2_ampere_(nk_f32_t low, nk_f32_t high) {
    __trap();
    return 0;
}

#endif // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

#pragma endregion Instructions

#pragma region Fragments

NUMKONG_DEVICE void nk_attention_scores_bf16_ampere_(nk_fui32_t scores[2][4], nk_u32_t const query[8],
                                                     nk_u32_t const keys[4]) {
    nk_mma_bf16_ampere_(scores[0], query, keys[0], keys[1]);
    nk_mma_bf16_ampere_(scores[1], query, keys[2], keys[3]);
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
 *  @param[in] dtype The input dtype: @c nk_bf16_k, @c nk_e4m3_k or @c nk_i8_k.
 *  @param[in] mma_dtype The dtype the MMAs consume: @c nk_f16_k for E4M3 converted first, else
 *      @p dtype.
 *  @param[in] width Panel width and where Q fragments live, see @c nk_attention_width_t.
 *  @param[in] split Whether warps own rows or positions, see @c nk_attention_split_t.
 *  @param[in] epilogue F32 scores and P · V sums, or exact I32 ones converted per panel.
 *  @param[in] scores One depth step of S, see @c nk_attention_scores_ampere_t.
 *  @param[in] values_mma One 16 × 8 step of P · V on the packed P and V fragments.
 *  @param[in] weights P from probabilities, see @c nk_attention_weights_ampere_t.
 */
NUMKONG_DEVICE void nk_attention_block_ampere_(nk_dtype_t dtype, nk_dtype_t mma_dtype, nk_attention_width_t width,
                                               nk_attention_split_t split, nk_cross_epilogue_t epilogue,
                                               nk_attention_scores_ampere_t scores, nk_cross_mma_ampere_t values_mma,
                                               nk_attention_weights_ampere_t weights,
                                               nk_attention_arguments_t const *arguments,
                                               nk_attention_work_t const *work, unsigned char *shared,
                                               unsigned (*unions)[2]) {

    nk_f32_t const negative_infinity = nk_attention_negative_infinity_();
    unsigned const lane = threadIdx.x & 31, warp = threadIdx.x >> 5, group = lane >> 2, quad = lane & 3;
    unsigned const element_bytes = dtype == nk_bf16_k ? 2 : 1;
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
    unsigned const value_stride = dtype == nk_bf16_k ? row_stride : panel + nk_attention_row_padding_ampere_k;
    unsigned char *keys_shared = shared + arguments->key_offset[split];
    unsigned char *values_shared = shared + arguments->value_offset[split];
    unsigned char *queries_shared = shared + arguments->query_offset[split];

    nk_size_t length, plane_bytes;
    unsigned char const *keys_plane = nk_attention_keys_plane_(arguments, work, row_bytes, &length, &plane_bytes);
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
        nk_attention_row_keys_((nk_i64_t)query + arguments->diagonal_offset, arguments->window, length,
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
    unsigned const panel_end = block_begin < block_end ? (unsigned)nk_size_divide_round_up_(block_end, panel) : 0;

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
        if (dtype == nk_bf16_k)
            for (unsigned element = lane; element < depth_padded; element += 32)
                ((unsigned short *)destination)[element] = valid && element < depth
                                                               ? ((unsigned short const *)source)[element]
                                                               : (unsigned short)0;
        else
            for (unsigned element = lane; element < depth_padded; element += 32)
                destination[element] = valid && element < depth ? source[element] : (unsigned char)0;
    }
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
        if (dtype == nk_bf16_k)
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
                correction[half] = nk_f32_exp2_(row_max[half] - subtrahend[half]);
                row_max[half] = new_max;
                row_sum[half] *= correction[half];
            }
#pragma unroll
            for (unsigned tile = 0; tile < 2 * groups; ++tile)
#pragma unroll
                for (unsigned element = 0; element < 4; ++element)
                    tile_scores[tile][element].f = nk_f32_exp2_(tile_scores[tile][element].f -
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
                    if (dtype != nk_bf16_k && mma_dtype == dtype)
                        probabilities[position_group >> 1][(position_group & 1) * 2 + half] = packed[0];
                    else
                        probabilities[position_group][half] = packed[0],
                        probabilities[position_group][half + 2] = packed[1];
                }
            if (dtype != nk_bf16_k && mma_dtype == dtype && groups == 1) probabilities[0][2] = probabilities[0][3] = 0;
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

        if (dtype == nk_bf16_k) {
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
            own_factor[half] = nk_f32_exp2_(row_max[half] - subtrahend[half]);
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
                factor[half] = nk_f32_exp2_(statistics[((other * 2 + half) * 2 + 0) * 32 + lane] - subtrahend[half]);
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
NUMKONG_DEVICE void nk_attention_tile_mma_ampere_(nk_dtype_t dtype, nk_dtype_t mma_dtype, nk_attention_width_t width,
                                                  nk_cross_epilogue_t epilogue, nk_attention_scores_ampere_t scores,
                                                  nk_cross_mma_ampere_t values_mma,
                                                  nk_attention_weights_ampere_t weights,
                                                  nk_attention_arguments_t const *arguments) {
    extern __shared__ __align__(128) unsigned char nk_attention_shared_ampere_[];
    __shared__ nk_u64_t prefix[nk_attention_threads_k + 1];
    __shared__ nk_u64_t warp_totals[nk_attention_threads_k / 32];
    __shared__ unsigned unions[nk_attention_threads_k / 32][2];
    nk_attention_schedule_t schedule;
    if (!nk_attention_schedule_start_(arguments, &schedule, prefix, warp_totals)) return;
    nk_attention_work_t work;
    for (nk_size_t item = blockIdx.x; nk_attention_schedule_next_(&schedule, prefix, warp_totals, item, &work);
         item += gridDim.x) {
        if (work.row_count <= nk_attention_warp_rows_ampere_k)
            nk_attention_block_ampere_(dtype, mma_dtype, width, nk_attention_split_positions_k, epilogue, scores,
                                       values_mma, weights, arguments, &work, nk_attention_shared_ampere_, unions);
        else
            nk_attention_block_ampere_(dtype, mma_dtype, width, nk_attention_split_rows_k, epilogue, scores, values_mma,
                                       weights, arguments, &work, nk_attention_shared_ampere_, unions);
    }
}

/** The tile on MMAs of the input dtype itself. */
NUMKONG_DEVICE void nk_attention_tile_ampere_(nk_dtype_t dtype, nk_attention_width_t width,
                                              nk_cross_epilogue_t epilogue, nk_attention_scores_ampere_t scores,
                                              nk_cross_mma_ampere_t values_mma, nk_attention_weights_ampere_t weights,
                                              nk_attention_arguments_t const *arguments) {
    nk_attention_tile_mma_ampere_(dtype, dtype, width, epilogue, scores, values_mma, weights, arguments);
}

/** The tile on F16 MMAs, for E4M3 converted to F16 first. */
NUMKONG_DEVICE void nk_attention_tile_ampere_f16_mma_(nk_dtype_t dtype, nk_attention_width_t width,
                                                      nk_cross_epilogue_t epilogue, nk_attention_scores_ampere_t scores,
                                                      nk_cross_mma_ampere_t values_mma,
                                                      nk_attention_weights_ampere_t weights,
                                                      nk_attention_arguments_t const *arguments) {
    nk_attention_tile_mma_ampere_(dtype, nk_f16_k, width, epilogue, scores, values_mma, weights, arguments);
}

#pragma endregion Tile

#pragma region Launch

/** Places the panel buffers of a width in dynamic shared memory, returning the bytes
 *  a block needs. */
NUMKONG_INLINE nk_size_t nk_attention_shared_layout_ampere_(nk_dtype_t dtype, nk_dtype_t mma_dtype,
                                                            nk_attention_width_t width, nk_size_t depth,
                                                            nk_attention_arguments_t *arguments) {
    nk_size_t const element_bytes = dtype == nk_bf16_k ? 2 : 1;
    nk_size_t const row_bytes = nk_size_round_up_to_multiple_(depth * element_bytes, nk_attention_step_bytes_k);
    nk_size_t const row_stride = row_bytes + nk_attention_row_padding_ampere_k,
                    depth_padded = row_bytes / element_bytes;
    nk_size_t const keys_narrow = nk_attention_panel_k * row_stride;
    nk_size_t const keys_wide = nk_attention_wide_panel_ampere_k * row_stride;
    nk_size_t const values_narrow = dtype == nk_bf16_k ? keys_narrow
                                                       : depth_padded * nk_attention_padded_panel_ampere_k;
    nk_size_t const values_wide = dtype == nk_bf16_k ? keys_wide
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
    nk_dtype_t mma_dtype, void const *queries, void const *packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride,
    nk_size_t output_stride, nk_f32_t scale, nk_f32_t score_scale, nk_f32_t output_scale, nk_attention_mask_t mask,
    nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start, nk_size_t task_count, void *stream) {
    if (((nk_size_t)packed & 15) || (((nk_size_t)output | output_stride) & 3)) return nk_misaligned_k;
    if (key_value_head_count == 0 || head_count % key_value_head_count != 0) return nk_unexpected_dimensions_k;
    if (task_count == 0 || depth == 0) return nk_success_k;
    nk_attention_arguments_t arguments = nk_attention_arguments_init_(
        queries, packed, output, head_count, key_value_head_count, depth, query_offsets, query_stride, output_stride,
        scale, score_scale, output_scale, mask, diagonal_offset, window, task_start, task_count);
    if (depth > nk_attention_wide_depth_ampere_k)
        return nk_launch_resident_simt_(fallback_kernel, nk_attention_threads_k, 0, 0, NUMKONG_SIZE_MAX, &arguments,
                                        stream);
    nk_attention_width_t const width = depth <= nk_attention_narrow_depth_ampere_k ? nk_attention_width_128_k
                                                                                   : nk_attention_width_256_k;
    nk_size_t const shared_bytes = nk_attention_shared_layout_ampere_(dtype, mma_dtype, width, depth, &arguments);
    return nk_launch_resident_simt_(
        width == nk_attention_width_128_k ? narrow_kernel : wide_kernel, nk_attention_threads_k, shared_bytes,
        nk_attention_shared_ceiling_ampere_(dtype, mma_dtype, width), NUMKONG_SIZE_MAX, &arguments, stream);
}

/** The launch for the tile on MMAs of the input dtype itself. */
NUMKONG_INLINE nk_status_t nk_attention_launch_ampere_(
    void const *narrow_kernel, void const *wide_kernel, void const *fallback_kernel, nk_dtype_t dtype,
    void const *queries, void const *packed, nk_f32_t *output, nk_size_t head_count, nk_size_t key_value_head_count,
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale,
    nk_f32_t score_scale, nk_f32_t output_scale, nk_attention_mask_t mask, nk_i64_t diagonal_offset, nk_size_t window,
    nk_size_t task_start, nk_size_t task_count, void *stream) {
    return nk_attention_launch_mma_ampere_(narrow_kernel, wide_kernel, fallback_kernel, dtype, dtype, queries, packed,
                                           output, head_count, key_value_head_count, depth, query_offsets, query_stride,
                                           output_stride, scale, score_scale, output_scale, mask, diagonal_offset,
                                           window, task_start, task_count, stream);
}

/** The launch for the tile on F16 MMAs, for E4M3 converted to F16 first. */
NUMKONG_INLINE nk_status_t nk_attention_launch_ampere_f16_mma_(
    void const *narrow_kernel, void const *wide_kernel, void const *fallback_kernel, nk_dtype_t dtype,
    void const *queries, void const *packed, nk_f32_t *output, nk_size_t head_count, nk_size_t key_value_head_count,
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale,
    nk_f32_t score_scale, nk_f32_t output_scale, nk_attention_mask_t mask, nk_i64_t diagonal_offset, nk_size_t window,
    nk_size_t task_start, nk_size_t task_count, void *stream) {
    return nk_attention_launch_mma_ampere_(narrow_kernel, wide_kernel, fallback_kernel, dtype, nk_f16_k, queries,
                                           packed, output, head_count, key_value_head_count, depth, query_offsets,
                                           query_stride, output_stride, scale, score_scale, output_scale, mask,
                                           diagonal_offset, window, task_start, task_count, stream);
}

#pragma endregion Launch

/*  Later generations read the helpers above and emit only their own kernels. */
#if NUMKONG_TARGET_AMPERE

#pragma region Instantiations

nk_define_attention_pack_size_simt_(bf16, ampere, 2)
nk_define_attention_packed_shape_simt_(bf16, ampere)
nk_define_attention_pack_simt_(bf16, ampere, bf16)
nk_define_attention_packed_simt_(bf16, ampere, ampere, nk_attention_launch_ampere_, bf16, nk_cross_epilogue_f32_k,
                                 nk_attention_scores_bf16_ampere_, nk_mma_bf16_ampere_,
                                 nk_attention_weights_bf16_ampere_, 1.0f, 1.0f)

nk_define_attention_pack_size_simt_(e4m3, ampere, 1)
nk_define_attention_packed_shape_simt_(e4m3, ampere)
nk_define_attention_pack_simt_(e4m3, ampere, e4m3)
nk_define_attention_packed_simt_(e4m3, ampere, ampere_f16_mma, nk_attention_launch_ampere_f16_mma_, e4m3,
                                 nk_cross_epilogue_f32_k, nk_attention_scores_e4m3_ampere_, nk_mma_f16_ampere_,
                                 nk_attention_weights_f16_ampere_, 65536.0f, 256.0f)

nk_define_attention_pack_size_simt_(i8, ampere, 1)
nk_define_attention_packed_shape_simt_(i8, ampere)
nk_define_attention_pack_simt_(i8, ampere, i8)
nk_define_attention_packed_simt_(i8, ampere, ampere, nk_attention_launch_ampere_, i8, nk_cross_epilogue_i32_to_f32_k,
                                 nk_attention_scores_i8_ampere_, nk_mma_u8i8_ampere_, nk_attention_weights_u8_ampere_,
                                 1.0f, 1.0f)

#pragma endregion Instantiations

#endif // NUMKONG_TARGET_AMPERE

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_AMPERE_
#endif // NUMKONG_ATTENTION_AMPERE_CUH
