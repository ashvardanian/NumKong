/**
 *  @file include/numkong/attention/cdna5.cuh
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Ragged attention for AMD Instinct MI400, gfx1250 and gfx1251.
 *
 *  @sa include/numkong/attention.h
 *  @sa include/numkong/attention/cdna3.cuh
 *  @sa include/numkong/attention/cdna4.cuh
 *
 *  The CDNA3 kernel's staging, transposed scores and online softmax on 32-lane wavefronts: four of
 *  them own 16 rows each of a work item, and each lane holds 8 positions of one query row per 16 ×
 *  16 WMMA of Sᵀ. BF16 multiplies BF16, E4M3 scores take the native Float8 WMMA and its P · V runs
 *  on F16 with V converted exactly, and I8 scores run exact integer WMMA with U8 probabilities
 *  against I8 values, which the integer WMMA takes without an offset. Depths above 256 fall back to
 *  the @c rocm kernel. The pack layout and the work scheduler are the @c rocm capability's.
 */
#ifndef NUMKONG_ATTENTION_CDNA5_CUH
#define NUMKONG_ATTENTION_CDNA5_CUH

#if NUMKONG_ARCH_ROCM_
#if NUMKONG_TARGET_CDNA5

#include "numkong/attention/cdna4.cuh" // `nk_attention_launch_cdna3_`, `nk_f32_to_f16_cdna4_`
#include "numkong/dots/cdna5.cuh"      // `nk_wmma_bf16_cdna5_`, `nk_wmma_u8i8_cdna5_`

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Fragments

/** Adds one depth step of Sᵀ for 16 positions against 16 query rows: K as the A operand, Q as the
 *  B operand, 8 words a lane for BF16 and I8 and 16 for E4M3. */
typedef void (*nk_attention_scores_cdna5_t)(nk_fui32_t scores[8], nk_u32_t const keys[16], nk_u32_t const queries[16]);

/** Adds one step of P · V for 16 query rows against 16 depth columns, from the packed P and the
 *  staged codes of Vᵀ. */
typedef void (*nk_attention_values_cdna5_t)(nk_fui32_t output[8], nk_u32_t const probabilities[8],
                                            nk_u32_t const values[8]);

/** Packs one lane's probabilities of a P · V step into its A operand, adding the values that
 *  operand represents to @p sum: 16 probabilities for 16-bit weights, 32 for 8-bit ones. */
typedef void (*nk_attention_weights_cdna5_t)(nk_f32_t const probabilities[32], nk_u32_t packed[8], nk_f32_t *sum);

/** Slot of a panel's Vᵀ that row @p row of Sᵀ tile @p tile covers, so each lane's scores are the
 *  probabilities in the order its V operand reads them: 32 slots per P · V step in 16 per lane
 *  group for 16-bit weights, and 64 in 32 per lane group for 8-bit ones. */
NUMKONG_DEVICE unsigned nk_attention_tile_slot_cdna5_(nk_dtype_t dtype, unsigned tile, unsigned row) {
    if (dtype == nk_i8_k) return (row >> 3) * 32 + tile * 8 + (row & 7);
    return (tile >> 1) * 32 + (row >> 3) * 16 + (tile & 1) * 8 + (row & 7);
}

NUMKONG_DEVICE void nk_attention_scores_bf16_cdna5_(nk_fui32_t scores[8], nk_u32_t const keys[16],
                                                    nk_u32_t const queries[16]) {
    nk_wmma_bf16_cdna5_(scores, keys, queries);
}

NUMKONG_DEVICE void nk_attention_scores_e4m3_cdna5_(nk_fui32_t scores[8], nk_u32_t const keys[16],
                                                    nk_u32_t const queries[16]) {
    nk_wmma_e4m3_cdna5_(scores, keys, queries);
}

NUMKONG_DEVICE void nk_attention_scores_i8_cdna5_(nk_fui32_t scores[8], nk_u32_t const keys[16],
                                                  nk_u32_t const queries[16]) {
    nk_wmma_i8_cdna5_(scores, keys, queries);
}

NUMKONG_DEVICE void nk_attention_values_bf16_cdna5_(nk_fui32_t output[8], nk_u32_t const probabilities[8],
                                                    nk_u32_t const values[8]) {
    nk_wmma_bf16_cdna5_(output, probabilities, values);
}

/** F16 weights against 16 E4M3 codes of V converted to F16 over 256, which the output scale
 *  undoes. */
NUMKONG_DEVICE void nk_attention_values_e4m3_cdna5_(nk_fui32_t output[8], nk_u32_t const probabilities[8],
                                                    nk_u32_t const values[8]) {
    nk_u32_t halves[8];
#pragma unroll
    for (unsigned word = 0; word < 4; ++word)
        nk_e4m3x4_to_f16x4_cdna4_(values[word], &halves[word * 2], &halves[word * 2 + 1]);
    nk_wmma_f16_cdna5_(output, probabilities, halves);
}

NUMKONG_DEVICE void nk_attention_values_i8_cdna5_(nk_fui32_t output[8], nk_u32_t const probabilities[8],
                                                  nk_u32_t const values[8]) {
    nk_wmma_u8i8_cdna5_(output, probabilities, values);
}

NUMKONG_DEVICE void nk_attention_weights_bf16_cdna5_(nk_f32_t const probabilities[32], nk_u32_t packed[8],
                                                     nk_f32_t *sum) {
#pragma unroll
    for (unsigned word = 0; word < 8; ++word) {
        nk_u32_t const low = nk_f32_to_bf16_cdna3_(probabilities[word * 2]);
        nk_u32_t const high = nk_f32_to_bf16_cdna3_(probabilities[word * 2 + 1]);
        packed[word] = low | (high << 16);
        *sum += __uint_as_float(low << 16) + __uint_as_float(high << 16);
    }
}

NUMKONG_DEVICE void nk_attention_weights_f16_cdna5_(nk_f32_t const probabilities[32], nk_u32_t packed[8],
                                                    nk_f32_t *sum) {
#pragma unroll
    for (unsigned word = 0; word < 8; ++word) {
        nk_u32_t const low = nk_f32_to_f16_cdna4_(probabilities[word * 2]);
        nk_u32_t const high = nk_f32_to_f16_cdna4_(probabilities[word * 2 + 1]);
        packed[word] = low | (high << 16);
        *sum += __half2float(__ushort_as_half((unsigned short)low)) +
                __half2float(__ushort_as_half((unsigned short)high));
    }
}

/** U8 weights round(255 · p), the max-scoring position landing on 255. */
NUMKONG_DEVICE void nk_attention_weights_u8_cdna5_(nk_f32_t const probabilities[32], nk_u32_t packed[8],
                                                   nk_f32_t *sum) {
    nk_u32_t total = 0;
#pragma unroll
    for (unsigned word = 0; word < 8; ++word) {
        nk_u32_t bytes = 0;
#pragma unroll
        for (unsigned index = 0; index < 4; ++index) {
            nk_u32_t const weight = (nk_u32_t)(probabilities[word * 4 + index] * 255.0f + 0.5f);
            bytes |= weight << (index * 8), total += weight;
        }
        packed[word] = bytes;
    }
    *sum += (nk_f32_t)total;
}

/** The sum @p weights adds for a probability of one, the unit a row's weights sum counts in. */
NUMKONG_DEVICE nk_f32_t nk_attention_weight_unit_cdna5_(nk_attention_weights_cdna5_t weights) {
    nk_f32_t probabilities[32] = {1};
    nk_u32_t packed[8];
    nk_f32_t unit = 0;
    weights(probabilities, packed, &unit);
    return unit;
}

#pragma endregion Fragments

#pragma region Tile

/**
 *  @brief One work item on one block: Sᵀ, online softmax and P · V over every panel its rows see,
 *      then the output.
 *  @sa nk_attention_block_cdna3_ for the parameters.
 *
 *  Lane l of wavefront w holds query row 16 × w + l % 16 as the B operand of Sᵀ and the A operand
 *  of P · V, and rows 8 × (l / 16) + e of P · V's output, whose softmax corrections and sums it
 *  reads from the lanes holding those rows.
 */
NUMKONG_DEVICE void nk_attention_block_cdna5_(nk_dtype_t dtype, nk_attention_width_t width, nk_attention_mask_t mask,
                                              nk_cross_epilogue_t epilogue, nk_attention_scores_cdna5_t scores,
                                              nk_attention_values_cdna5_t values_mma,
                                              nk_attention_weights_cdna5_t weights,
                                              nk_attention_arguments_t const *arguments,
                                              nk_attention_work_t const *work, unsigned char *shared) {

    nk_f32_t const negative_infinity = nk_attention_negative_infinity_();
    unsigned const lane = threadIdx.x & 31, wave = threadIdx.x >> 5, group = lane >> 4, column = lane & 15;
    unsigned const element_bytes = dtype == nk_bf16_k ? 2 : 1;
    // Sᵀ reads 64 bytes of depth a step for BF16 and I8 and 128 for E4M3, half per lane group.
    unsigned const step_bytes = dtype == nk_e4m3_k ? 128 : 64, step_words = step_bytes / 8;
    unsigned const width_depth = width == nk_attention_width_128_k ? nk_attention_narrow_depth_cdna3_k
                                                                   : nk_attention_wide_depth_cdna3_k;
    unsigned const width_steps = (unsigned)nk_size_divide_round_up_(width_depth * element_bytes, step_bytes),
                   width_tiles = width_depth / 16;
    // P · V takes 32 positions a step for 16-bit weights and 64 for 8-bit ones.
    unsigned const value_steps = dtype == nk_i8_k ? 1 : 2;

    nk_size_t const depth = arguments->depth;
    unsigned const row_bytes = (unsigned)nk_size_round_up_to_multiple_(depth * element_bytes,
                                                                       nk_attention_step_bytes_k);
    unsigned const row_stride = row_bytes + nk_attention_row_padding_cdna3_k;
    unsigned const value_stride = nk_attention_panel_k * element_bytes + nk_attention_row_padding_cdna3_k;
    unsigned const depth_steps = nk_u32_divide_round_up_(row_bytes, step_bytes);
    // Loops run to the register arrays' fixed bounds and skip steps and tiles by these masks, as a
    // bound known only after inlining leaves them rolled and the arrays in scratch.
    unsigned const width_step_mask = (1u << width_steps) - 1, step_mask = ((1u << depth_steps) - 1) & width_step_mask;
    unsigned const depth_tiles = (unsigned)nk_size_divide_round_up_(depth, 16),
                   width_tile_mask = (1u << width_tiles) - 1;
    unsigned const tile_mask = ((1u << depth_tiles) - 1) & width_tile_mask;
    unsigned const store_mask =
        (column < depth ? (1u << (unsigned)nk_size_divide_round_up_(depth - column, 16)) - 1 : 0) & width_tile_mask;
    unsigned char *keys_shared = shared, *values_shared = shared + nk_attention_panel_k * row_stride;
    // Q is staged where V goes, and read into registers before the first V lands.
    unsigned char *queries_shared = values_shared;

    nk_size_t length, plane_bytes;
    unsigned char const *keys_plane = nk_attention_keys_plane_(arguments, work, row_bytes, &length, &plane_bytes);
    unsigned char const *values_plane = keys_plane + arguments->key_value_head_count * plane_bytes;
    nk_size_t const positions_padded = plane_bytes / row_bytes;

    // Lane `column` holds row `column` of its wavefront's 16, which classify panels together.
    nk_diagonal_band_t const band = nk_attention_kernel_band_(mask, arguments);
    unsigned const local_row = wave * 16 + column;
    unsigned const wave_rows = work->row_count > wave * 16 ? min((unsigned)work->row_count - wave * 16, 16u) : 0;
    nk_i64_t const wave_first = nk_attention_row_position_(work, wave * 16, length);
    nk_i64_t const lane_row = nk_attention_row_position_(work, local_row, length);
    nk_size_t block_begin, block_end;
    nk_attention_rows_keys_(band, nk_attention_row_position_(work, 0, length),
                            nk_attention_row_position_(work, work->row_count - 1, length), length, &block_begin,
                            &block_end);
    unsigned const panel_first = (unsigned)(block_begin / nk_attention_panel_k);
    unsigned const panel_end = block_begin < block_end
                                   ? (unsigned)nk_size_divide_round_up_(block_end, nk_attention_panel_k)
                                   : panel_first;

    nk_attention_stage_queries_(dtype, arguments, work, queries_shared, nk_attention_block_rows_k, row_stride,
                                row_bytes / element_bytes, 32);
    // Also retires the previous item's reads of every buffer this item refills.
    __syncthreads();

    nk_u32_t queries[nk_attention_steps_cdna3_k][16];
#pragma unroll
    for (unsigned step = 0; step < nk_attention_steps_cdna3_k; ++step)
        if ((step_mask >> step) & 1)
            nk_attention_load_fragment_cdna3_(queries_shared + local_row * row_stride,
                                              step * step_bytes + group * step_words * 4, row_bytes, step_words,
                                              queries[step]);
    // Every Q read retires before the first V lands on it.
    __syncthreads();

    nk_f32_t row_max = negative_infinity, row_sum = 0;
    nk_fui32_t output[nk_attention_tiles_cdna3_k][8];
#pragma unroll
    for (unsigned depth_tile = 0; depth_tile < nk_attention_tiles_cdna3_k; ++depth_tile)
#pragma unroll
        for (unsigned element = 0; element < 8; ++element) output[depth_tile][element].u = 0;
    nk_f32_t const scale2 = arguments->scale2 * arguments->score_scale;

    for (unsigned panel_index = panel_first; panel_index < panel_end; ++panel_index) {
        nk_size_t const panel_position = (nk_size_t)panel_index * nk_attention_panel_k;
        nk_attention_stage_rows_cdna3_(keys_shared, keys_plane, panel_position, nk_attention_panel_k, row_bytes,
                                       row_stride);
        nk_attention_stage_values_cdna3_(dtype, values_shared, values_plane, positions_padded, panel_position,
                                         row_bytes, value_stride);
        __syncthreads();

        nk_diagonal_band_coverage_t const coverage = nk_attention_tile_coverage_(
            band, wave_first, wave_rows, panel_position, nk_attention_panel_k, length);
        int const active = coverage != nk_diagonal_band_outside_k, masked = coverage != nk_diagonal_band_inside_k;
        nk_u32_t visible[2] = {0, 0};
        if (masked && local_row < work->row_count) {
            visible[0] = nk_diagonal_band_row_mask_simt_(band, lane_row, panel_position, length);
            visible[1] = nk_diagonal_band_row_mask_simt_(band, lane_row, panel_position + 32, length);
        }

        nk_fui32_t tile_scores[4][8];
#pragma unroll
        for (unsigned position_tile = 0; position_tile < 4; ++position_tile)
#pragma unroll
            for (unsigned element = 0; element < 8; ++element) tile_scores[position_tile][element].u = 0;
        if (active)
#pragma unroll
            for (unsigned step = 0; step < nk_attention_steps_cdna3_k; ++step) {
                if (!((step_mask >> step) & 1)) continue;
#pragma unroll
                for (unsigned position_tile = 0; position_tile < 4; ++position_tile) {
                    unsigned const position = (unsigned)nk_attention_slot_position_(
                        nk_attention_tile_slot_cdna5_(dtype, position_tile, column));
                    nk_u32_t keys[16];
                    nk_attention_load_fragment_cdna3_(keys_shared + position * row_stride,
                                                      step * step_bytes + group * step_words * 4, row_bytes, step_words,
                                                      keys);
                    scores(tile_scores[position_tile], keys, queries[step]);
                }
            }

#pragma unroll
        for (unsigned position_tile = 0; position_tile < 4; ++position_tile)
#pragma unroll
            for (unsigned element = 0; element < 8; ++element) {
                nk_fui32_t *score = &tile_scores[position_tile][element];
                score->f = (epilogue == nk_cross_epilogue_i32_to_f32_k ? (nk_f32_t)score->i : score->f) * scale2;
            }
        if (masked)
#pragma unroll
            for (unsigned position_tile = 0; position_tile < 4; ++position_tile)
#pragma unroll
                for (unsigned element = 0; element < 8; ++element) {
                    unsigned const offset = (unsigned)nk_attention_slot_position_(
                        nk_attention_tile_slot_cdna5_(dtype, position_tile, group * 8 + element));
                    nk_u32_t const word = offset < 32 ? visible[0] : visible[1];
                    if (!((word >> (offset & 31)) & 1)) tile_scores[position_tile][element].f = negative_infinity;
                }
        nk_f32_t chunk_max = negative_infinity;
#pragma unroll
        for (unsigned position_tile = 0; position_tile < 4; ++position_tile)
#pragma unroll
            for (unsigned element = 0; element < 8; ++element)
                chunk_max = fmaxf(chunk_max, tile_scores[position_tile][element].f);
        // The other lane group holds the rest of this lane's query row.
        chunk_max = fmaxf(chunk_max, nk_shuffle_xor_f32_rocm_(chunk_max, 16));
        nk_f32_t const new_max = fmaxf(row_max, chunk_max);
        // Subtracting 0 while every key so far is masked keeps `exp2(-∞ - max)` from NaN.
        nk_f32_t const subtrahend = new_max == negative_infinity ? 0.0f : new_max;
        nk_f32_t const correction = nk_f32_exp2_rocm_(row_max - subtrahend);
        row_max = new_max;
        row_sum *= correction;
        nk_u32_t probabilities[2][8];
#pragma unroll
        for (unsigned value_step = 0; value_step < 2; ++value_step) {
            if (value_step >= value_steps) continue;
            nk_f32_t step_probabilities[32];
#pragma unroll
            for (unsigned index = 0; index < 32; ++index) {
                unsigned const position_tile = dtype == nk_i8_k ? index / 8 : value_step * 2 + (index / 8 & 1);
                step_probabilities[index] = nk_f32_exp2_rocm_(tile_scores[position_tile][index % 8].f - subtrahend);
            }
            weights(step_probabilities, probabilities[value_step], &row_sum);
        }

        // This lane's 8 rows of P · V output take the corrections the lanes of those rows hold.
        nk_f32_t row_corrections[8];
#pragma unroll
        for (unsigned element = 0; element < 8; ++element)
            row_corrections[element] = __shfl(correction, group * 8 + element, 32);

        if (active)
#pragma unroll
            for (unsigned depth_tile = 0; depth_tile < nk_attention_tiles_cdna3_k; ++depth_tile) {
                if (!((tile_mask >> depth_tile) & 1)) continue;
                unsigned char const *values_row = values_shared + (depth_tile * 16 + column) * value_stride;
                if (dtype == nk_i8_k) {
                    nk_u32_t values[8];
#pragma unroll
                    for (unsigned chunk = 0; chunk < 2; ++chunk) {
                        uint4 const codes = *(uint4 const *)(values_row + group * 32 + chunk * 16);
                        values[chunk * 4 + 0] = codes.x, values[chunk * 4 + 1] = codes.y;
                        values[chunk * 4 + 2] = codes.z, values[chunk * 4 + 3] = codes.w;
                    }
                    nk_fui32_t sums[8] = {{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}};
                    values_mma(sums, probabilities[0], values);
#pragma unroll
                    for (unsigned element = 0; element < 8; ++element)
                        output[depth_tile][element].f = fmaf(output[depth_tile][element].f, row_corrections[element],
                                                             (nk_f32_t)sums[element].i);
                    continue;
                }
#pragma unroll
                for (unsigned element = 0; element < 8; ++element)
                    output[depth_tile][element].f *= row_corrections[element];
#pragma unroll
                for (unsigned value_step = 0; value_step < 2; ++value_step) {
                    unsigned const slot = value_step * 32 + group * 16;
                    nk_u32_t values[8];
                    if (dtype == nk_bf16_k)
#pragma unroll
                        for (unsigned chunk = 0; chunk < 2; ++chunk) {
                            uint4 const halves = *(uint4 const *)(values_row + slot * 2 + chunk * 16);
                            values[chunk * 4 + 0] = halves.x, values[chunk * 4 + 1] = halves.y;
                            values[chunk * 4 + 2] = halves.z, values[chunk * 4 + 3] = halves.w;
                        }
                    else {
                        uint4 const codes = *(uint4 const *)(values_row + slot);
                        values[0] = codes.x, values[1] = codes.y, values[2] = codes.z, values[3] = codes.w;
                        values[4] = values[5] = values[6] = values[7] = 0;
                    }
                    values_mma(output[depth_tile], probabilities[value_step], values);
                }
            }
        // Every read of the panel retires before the next one is staged over it.
        __syncthreads();
    }

    nk_f32_t const total = row_sum + nk_shuffle_xor_f32_rocm_(row_sum, 16);
    // Lanes 0 to 15 hold one score row each, the rows the log-sum-exp needs.
    nk_f32_t *const log_sum_exp_slot = group == 0 && local_row < work->row_count
                                           ? nk_attention_log_sum_exp_slot_(arguments, work, local_row)
                                           : NUMKONG_NULL;
    if (log_sum_exp_slot)
        *log_sum_exp_slot = nk_attention_log_sum_exp_simt_(row_max, total, nk_attention_weight_unit_cdna5_(weights));
    nk_f32_t inverses[8];
#pragma unroll
    for (unsigned element = 0; element < 8; ++element) {
        nk_f32_t const sum = __shfl(total, group * 8 + element, 32);
        inverses[element] = sum > 0 ? arguments->output_scale / sum : 0.0f;
    }
#pragma unroll
    for (unsigned element = 0; element < 8; ++element) {
        unsigned const local = wave * 16 + group * 8 + element;
        if (local >= work->row_count) continue;
        nk_f32_t *destination = nk_attention_output_row_(arguments, work, local);
#pragma unroll
        for (unsigned depth_tile = 0; depth_tile < nk_attention_tiles_cdna3_k; ++depth_tile)
            if ((store_mask >> depth_tile) & 1)
                destination[depth_tile * 16 + column] = output[depth_tile][element].f * inverses[element];
    }
}

/**
 *  @brief Every work item of a launch, walked with a stride of the grid.
 *  @sa nk_attention_block_cdna3_ for the parameters.
 */
NUMKONG_DEVICE void nk_attention_tile_cdna5_(nk_dtype_t dtype, nk_attention_width_t width, nk_attention_mask_t mask,
                                             nk_cross_epilogue_t epilogue, nk_attention_scores_cdna5_t scores,
                                             nk_attention_values_cdna5_t values_mma,
                                             nk_attention_weights_cdna5_t weights,
                                             nk_attention_arguments_t const *arguments) {
    extern __shared__ __attribute__((aligned(16))) unsigned char nk_attention_shared_cdna5_[];
    __shared__ nk_u64_t prefix[nk_attention_threads_k + 1];
    __shared__ nk_u64_t warp_totals[nk_attention_threads_k / 32];
    nk_attention_schedule_t schedule;
    if (!nk_attention_schedule_start_rocm_(arguments, &schedule, prefix, warp_totals)) return;
    nk_attention_work_t work;
    for (nk_size_t item = blockIdx.x; nk_attention_schedule_next_rocm_(&schedule, prefix, warp_totals, item, &work);
         item += gridDim.x)
        nk_attention_block_cdna5_(dtype, width, mask, epilogue, scores, values_mma, weights, arguments, &work,
                                  nk_attention_shared_cdna5_);
}

#pragma endregion Tile

#pragma region Instantiations

nk_define_attention_pack_size_simt_(bf16, cdna5, 2)
nk_define_attention_packed_shape_rocm_(bf16, cdna5)
nk_define_attention_pack_rocm_(bf16, cdna5, bf16)
nk_define_attention_packed_rocm_(bf16, cdna5, cdna5, nk_attention_launch_cdna3_, bf16, nk_cross_epilogue_f32_k,
                                 nk_attention_scores_bf16_cdna5_, nk_attention_values_bf16_cdna5_,
                                 nk_attention_weights_bf16_cdna5_, 1.0f, 1.0f)
nk_define_attention_backward_rocm_(bf16, cdna5, bf16)

nk_define_attention_pack_size_simt_(e4m3, cdna5, 1)
nk_define_attention_packed_shape_rocm_(e4m3, cdna5)
nk_define_attention_pack_rocm_(e4m3, cdna5, e4m3)
nk_define_attention_packed_rocm_(e4m3, cdna5, cdna5, nk_attention_launch_cdna3_, e4m3, nk_cross_epilogue_f32_k,
                                 nk_attention_scores_e4m3_cdna5_, nk_attention_values_e4m3_cdna5_,
                                 nk_attention_weights_f16_cdna5_, 1.0f, 256.0f)

nk_define_attention_pack_size_simt_(i8, cdna5, 1)
nk_define_attention_packed_shape_rocm_(i8, cdna5)
nk_define_attention_pack_rocm_(i8, cdna5, i8)
nk_define_attention_packed_rocm_(i8, cdna5, cdna5, nk_attention_launch_cdna3_, i8, nk_cross_epilogue_i32_to_f32_k,
                                 nk_attention_scores_i8_cdna5_, nk_attention_values_i8_cdna5_,
                                 nk_attention_weights_u8_cdna5_, 1.0f, 1.0f)

#pragma endregion Instantiations

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_CDNA5
#endif // NUMKONG_ARCH_ROCM_
#endif // NUMKONG_ATTENTION_CDNA5_CUH
