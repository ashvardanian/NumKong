/**
 *  @file include/numkong/attention/cdna4.cuh
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Ragged attention for AMD Instinct MI350, gfx950.
 *
 *  @sa include/numkong/attention.h
 *  @sa include/numkong/dots/cdna4.cuh
 *
 *  FlashAttention-2 on 16 × 16 MFMAs: two wavefronts own the 64 rows of a work item, 32 each as two
 *  16-row query tiles, and walk 64-position K and V panels staged in shared memory with a base-2
 *  online softmax per row. Scores come out transposed, Sᵀ = K · Qᵀ, so each lane holds 4 positions
 *  of one query row, exactly the probabilities its lane feeds P · V as the A operand; K rows are
 *  picked per lane so those positions are the ones the V operand holds. BF16 multiplies BF16, E4M3
 *  scores take the native Float8 MFMA and its P · V runs on F16 with V cast exactly, and I8 runs
 *  exact integer MFMA with U8 probabilities offset into I8. Depths above 256 fall back to the
 *  @c rocm kernel. The pack layout and the work scheduler are the @c rocm capability's; the staging
 *  and the launch serve the CDNA5 kernel too.
 */
#ifndef NUMKONG_ATTENTION_CDNA4_CUH
#define NUMKONG_ATTENTION_CDNA4_CUH

#if NUMKONG_ARCH_ROCM_CDNA4_

#include "numkong/attention/simt.cuh" // `nk_attention_schedule_next_`, `nk_attention_fallback_`
#include "numkong/dots/cdna4.cuh"     // `nk_mfma_bf16_cdna4_`, `nk_mfma_i8_cdna4_`, `nk_byte_permute_cdna4_`

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Configuration

enum {
    nk_attention_narrow_depth_cdna4_k = 128,
    nk_attention_wide_depth_cdna4_k = 256,
    nk_attention_row_padding_cdna4_k = 16,
    nk_attention_wave_rows_cdna4_k = 32,
    nk_attention_steps_cdna4_k = 8,
    nk_attention_tiles_cdna4_k = nk_attention_wide_depth_cdna4_k / 16,
};

#pragma endregion Configuration

#pragma region Conversions

/** Rounds @p value to the nearest BF16, ties to even; a NaN keeps its sign and top payload bits. */
NUMKONG_DEVICE nk_u32_t nk_f32_to_bf16_cdna4_(nk_f32_t value) {
    nk_u32_t const bits = __float_as_uint(value);
    // Rounding could carry a NaN into infinity or zero
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) return (bits >> 16) | 0x0040u;
    return (bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16;
}

NUMKONG_DEVICE nk_u32_t nk_f32_to_f16_cdna4_(nk_f32_t value) { return __half_as_ushort(__float2half_rn(value)); }

/** Four E4M3 codes as two pairs of F16 patterns scaled by 2⁻⁸: each code lands in the high byte
 *  of a half and keeps only its fields. */
NUMKONG_DEVICE void nk_e4m3x4_to_f16x4_cdna4_(nk_u32_t codes, nk_u32_t *low, nk_u32_t *high) {
    nk_u32_t const low_halves = nk_byte_permute_cdna4_(0, codes, 0x010C000Cu);
    nk_u32_t const high_halves = nk_byte_permute_cdna4_(0, codes, 0x030C020Cu);
    // The NaN magnitude 0x7F shifts to a finite 0x3F80, and setting 0x4000 fills its exponent.
    nk_u32_t const nan_bytes = ((codes & 0x7F7F7F7Fu) + 0x01010101u) & 0x80808080u;
    *low = ((low_halves >> 1) & 0x3F803F80u) | (low_halves & 0x80008000u) |
           (nk_byte_permute_cdna4_(0, nan_bytes, 0x010C000Cu) >> 1);
    *high = ((high_halves >> 1) & 0x3F803F80u) | (high_halves & 0x80008000u) |
            (nk_byte_permute_cdna4_(0, nan_bytes, 0x030C020Cu) >> 1);
}

#pragma endregion Conversions

#pragma region Staging

/** Copies @p rows rows of @p row_bytes from a position-major plane at @p first_position into
 *  shared rows @p stride bytes apart. */
NUMKONG_DEVICE void nk_attention_stage_rows_cdna4_(unsigned char *shared, unsigned char const *plane,
                                                   nk_size_t first_position, unsigned rows, unsigned row_bytes,
                                                   unsigned stride) {
    unsigned const chunks_per_row = row_bytes >> 4;
    for (unsigned chunk = threadIdx.x; chunk < rows * chunks_per_row; chunk += nk_attention_threads_k) {
        unsigned const row = chunk / chunks_per_row, column = chunk - row * chunks_per_row;
        *(uint4 *)(shared + row * stride +
                   (column << 4)) = *(uint4 const *)(plane + (first_position + row) * row_bytes + (column << 4));
    }
}

/** Stages a panel of V as Vᵀ, depth rows of 64 slots @p stride bytes apart: 1-byte codes copy
 *  the pack's transposed rows, BF16 rows transpose into the same slot order. */
NUMKONG_DEVICE void nk_attention_stage_values_cdna4_(nk_dtype_t dtype, unsigned char *shared,
                                                     unsigned char const *plane, nk_size_t positions_padded,
                                                     nk_size_t first_position, unsigned row_bytes, unsigned stride) {
    if (dtype != nk_bf16_k) {
        for (unsigned chunk = threadIdx.x; chunk < row_bytes * 4; chunk += nk_attention_threads_k) {
            unsigned const row = chunk >> 2, column = chunk & 3;
            *(uint4 *)(shared + row * stride + (column << 4)) = *(uint4 const *)(plane + row * positions_padded +
                                                                                 first_position + (column << 4));
        }
        return;
    }
    // Consecutive threads take consecutive positions, so each depth row's stores stay contiguous.
    for (unsigned chunk = threadIdx.x; chunk < (row_bytes >> 4) * nk_attention_panel_k;
         chunk += nk_attention_threads_k) {
        unsigned const position = chunk & (nk_attention_panel_k - 1), column = chunk / nk_attention_panel_k;
        uint4 const halves = *(uint4 const *)(plane + (first_position + position) * row_bytes + (column << 4));
        nk_u32_t const words[4] = {halves.x, halves.y, halves.z, halves.w};
        unsigned char *destination = shared + column * 8 * stride + nk_attention_slot_(position) * 2;
#pragma unroll
        for (unsigned element = 0; element < 8; ++element)
            *(unsigned short *)(destination + element * stride) = (unsigned short)(words[element / 2] >>
                                                                                   (element % 2 * 16));
    }
}

/** Loads @p words words, 4 to 16, of a lane's fragment at @p offset of a staged @p row, as zeros
 *  from @p valid_bytes on, which only ever falls between 16-byte chunks. */
NUMKONG_DEVICE void nk_attention_load_fragment_cdna4_(unsigned char const *row, unsigned offset, unsigned valid_bytes,
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
NUMKONG_INLINE nk_size_t nk_attention_shared_bytes_cdna4_(nk_dtype_t dtype, nk_size_t depth) {
    nk_size_t const element_bytes = dtype == nk_bf16_k ? 2 : 1;
    nk_size_t const row_bytes = nk_size_round_up_to_multiple_(depth * element_bytes, nk_attention_step_bytes_k);
    nk_size_t const rows = nk_attention_panel_k * (row_bytes + nk_attention_row_padding_cdna4_k);
    nk_size_t const values = row_bytes / element_bytes *
                             (nk_attention_panel_k * element_bytes + nk_attention_row_padding_cdna4_k);
    return rows + (values > rows ? values : rows);
}

/**
 *  @brief Validates the contract and launches the kernel for the depth's width, or the fallback
 *      past it, with as many blocks as stay resident.
 *  @param[in] score_scale Undoes the power of two that converting Q and K puts on scores, or 1.
 *  @param[in] output_scale Undoes the power of two that converting V puts on the output, or 1.
 */
NUMKONG_INLINE nk_status_t nk_attention_launch_cdna4_(
    void const *narrow_kernel, void const *wide_kernel, void const *fallback_kernel, nk_dtype_t dtype,
    void const *queries, void const *packed, nk_f32_t *output, nk_size_t head_count, nk_size_t key_value_head_count,
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale,
    nk_f32_t score_scale, nk_f32_t output_scale, nk_attention_mask_t mask, nk_i64_t diagonal_offset, nk_size_t window,
    nk_size_t task_start, nk_size_t task_count, void *stream) {
    if (((nk_size_t)packed & 15) || (((nk_size_t)output | output_stride) & 3)) return nk_misaligned_k;
    if (key_value_head_count == 0 || head_count % key_value_head_count != 0) return nk_unexpected_dimensions_k;
    if (task_count == 0 || depth == 0) return nk_success_k;
    nk_attention_arguments_t arguments = nk_attention_arguments_init_(
        queries, packed, output, head_count, key_value_head_count, depth, query_offsets, query_stride, output_stride,
        scale, score_scale, output_scale, mask, diagonal_offset, window, task_start, task_count);
    if (depth > nk_attention_wide_depth_cdna4_k)
        return nk_launch_resident_(fallback_kernel, nk_attention_threads_k, 0, 0, NUMKONG_SIZE_MAX, &arguments, stream);
    nk_attention_width_t const width = depth <= nk_attention_narrow_depth_cdna4_k ? nk_attention_width_128_k
                                                                                  : nk_attention_width_256_k;
    nk_size_t const width_depth = width == nk_attention_width_128_k ? nk_attention_narrow_depth_cdna4_k
                                                                    : nk_attention_wide_depth_cdna4_k;
    // The attribute takes the width's largest size, so no depth's size races a concurrent launch.
    return nk_launch_resident_(width == nk_attention_width_128_k ? narrow_kernel : wide_kernel, nk_attention_threads_k,
                               nk_attention_shared_bytes_cdna4_(dtype, depth),
                               nk_attention_shared_bytes_cdna4_(dtype, width_depth), NUMKONG_SIZE_MAX, &arguments,
                               stream);
}

#pragma endregion Launch

#if NUMKONG_TARGET_CDNA4

#pragma region Fragments

/** Adds one depth step of Sᵀ for 16 positions against 16 query rows: K as the A operand, Q as the
 *  B operand, 4 words a lane for BF16 and I8 and 8 for E4M3. */
typedef void (*nk_attention_scores_cdna4_t)(nk_fui32_t scores[4], nk_u32_t const keys[8], nk_u32_t const queries[8]);

/** Adds one step of P · V for 16 query rows against 16 depth columns, from the packed P and the
 *  staged codes of Vᵀ. */
typedef void (*nk_attention_values_cdna4_t)(nk_fui32_t output[4], nk_u32_t const probabilities[4],
                                            nk_u32_t const values[4]);

/** Packs one lane's probabilities of a P · V step into its A operand, adding the values that
 *  operand represents to @p sum: 8 probabilities for 16-bit weights, 16 for 8-bit ones. */
typedef void (*nk_attention_weights_cdna4_t)(nk_f32_t const probabilities[16], nk_u32_t packed[4], nk_f32_t *sum);

/** Reads @p value from lane @p source of the whole 64-lane wavefront. */
NUMKONG_DEVICE nk_f32_t nk_shuffle_f32_cdna4_(nk_f32_t value, unsigned source) {
    return __shfl(value, (int)source, 64);
}

/** Slot of a panel's Vᵀ that row @p row of Sᵀ tile @p tile covers, so each lane's scores are the
 *  probabilities in the order its V operand reads them: 32 slots per P · V step in 8 per lane
 *  group for 16-bit weights, and 64 in 16 per lane group for 8-bit ones. */
NUMKONG_DEVICE unsigned nk_attention_tile_slot_cdna4_(nk_dtype_t dtype, unsigned tile, unsigned row) {
    if (dtype == nk_i8_k) return (row >> 2) * 16 + tile * 4 + (row & 3);
    return (tile >> 1) * 32 + (row >> 2) * 8 + (tile & 1) * 4 + (row & 3);
}

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

NUMKONG_DEVICE void nk_attention_weights_bf16_cdna4_(nk_f32_t const probabilities[16], nk_u32_t packed[4],
                                                     nk_f32_t *sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) {
        nk_u32_t const low = nk_f32_to_bf16_cdna4_(probabilities[word * 2]);
        nk_u32_t const high = nk_f32_to_bf16_cdna4_(probabilities[word * 2 + 1]);
        packed[word] = low | (high << 16);
        *sum += __uint_as_float(low << 16) + __uint_as_float(high << 16);
    }
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

/** U8 weights round(255 · p), the max-scoring position landing on 255, offset by −128 into I8 for
 *  the signed MFMA; the tile adds 128 · Σ V back. */
NUMKONG_DEVICE void nk_attention_weights_u8_cdna4_(nk_f32_t const probabilities[16], nk_u32_t packed[4],
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

#pragma endregion Fragments

#pragma region Tile

/**
 *  @brief One work item on one block: Sᵀ, online softmax and P · V over every panel its rows see,
 *      then the output.
 *  @param[in] dtype The input dtype: @c nk_bf16_k throughout, @c nk_e4m3_k scores with F16 P · V,
 *      or @c nk_i8_k throughout.
 *  @param[in] width The depths its registers hold, see @c nk_attention_width_t.
 *  @param[in] epilogue F32 scores and P · V sums, or exact I32 ones converted per panel.
 *  @param[in] scores One depth step of Sᵀ, see @c nk_attention_scores_cdna4_t.
 *  @param[in] values_mma One step of P · V, see @c nk_attention_values_cdna4_t.
 *  @param[in] weights P from probabilities, see @c nk_attention_weights_cdna4_t.
 *
 *  Lane l of wavefront w holds query row 32 × w + 16 × t + l % 16 of query tile t as the B operand
 *  of Sᵀ and the A operand of P · V, and rows 4 × (l / 16) + e of P · V's output, whose softmax
 *  corrections and sums it reads from the lanes holding those rows.
 */
NUMKONG_DEVICE void nk_attention_block_cdna4_(nk_dtype_t dtype, nk_attention_width_t width,
                                              nk_cross_epilogue_t epilogue, nk_attention_scores_cdna4_t scores,
                                              nk_attention_values_cdna4_t values_mma,
                                              nk_attention_weights_cdna4_t weights,
                                              nk_attention_arguments_t const *arguments,
                                              nk_attention_work_t const *work, unsigned char *shared,
                                              unsigned (*unions)[2]) {

    nk_f32_t const negative_infinity = nk_attention_negative_infinity_();
    unsigned const lane = threadIdx.x & 63, wave = threadIdx.x >> 6, group = lane >> 4, column = lane & 15;
    unsigned const element_bytes = dtype == nk_bf16_k ? 2 : 1;
    // Sᵀ reads 64 bytes of depth a step for BF16 and I8, and 128 for the E4M3 MFMA.
    unsigned const step_bytes = dtype == nk_e4m3_k ? 128 : 64, step_words = step_bytes / 16;
    unsigned const width_depth = width == nk_attention_width_128_k ? nk_attention_narrow_depth_cdna4_k
                                                                   : nk_attention_wide_depth_cdna4_k;
    unsigned const width_steps = (unsigned)nk_size_divide_round_up_(width_depth * element_bytes, step_bytes),
                   width_tiles = width_depth / 16;
    // P · V takes 32 positions a step for 16-bit weights and 64 for 8-bit ones.
    unsigned const value_steps = dtype == nk_i8_k ? 1 : 2;

    nk_size_t const depth = arguments->depth;
    unsigned const row_bytes = (unsigned)nk_size_round_up_to_multiple_(depth * element_bytes,
                                                                       nk_attention_step_bytes_k);
    unsigned const row_stride = row_bytes + nk_attention_row_padding_cdna4_k;
    unsigned const value_stride = nk_attention_panel_k * element_bytes + nk_attention_row_padding_cdna4_k;
    // 32-bit here and in `panel_end`, where the `nk_size_t` round-up helper costs registers.
    unsigned const depth_steps = (row_bytes + step_bytes - 1) / step_bytes;
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

    unsigned key_begin[2], key_end[2];
    unsigned union_begin = 0xFFFFFFFFu, union_end = 0;
#pragma unroll
    for (unsigned tile = 0; tile < 2; ++tile) {
        unsigned const local = wave * nk_attention_wave_rows_cdna4_k + tile * 16 + column;
        key_begin[tile] = key_end[tile] = 0;
        if (local >= work->row_count) continue;
        nk_size_t const query = (work->row_first + local) / work->heads_selected;
        nk_attention_row_keys_((nk_i64_t)query + arguments->diagonal_offset, arguments->window, length,
                               &key_begin[tile], &key_end[tile]);
        if (key_begin[tile] < key_end[tile])
            union_begin = min(union_begin, key_begin[tile]), union_end = max(union_end, key_end[tile]);
    }
#pragma unroll
    for (unsigned offset = 1; offset < 64; offset <<= 1) {
        union_begin = min(union_begin, nk_shuffle_xor_u32_(union_begin, offset));
        union_end = max(union_end, nk_shuffle_xor_u32_(union_end, offset));
    }
    if (lane == 0) unions[wave][0] = union_begin, unions[wave][1] = union_end;

    for (unsigned local = wave; local < nk_attention_block_rows_k; local += 2) {
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
            for (unsigned element = lane; element < row_bytes / 2; element += 64)
                ((unsigned short *)destination)[element] = valid && element < depth
                                                               ? ((unsigned short const *)source)[element]
                                                               : (unsigned short)0;
        else
            for (unsigned element = lane; element < row_bytes; element += 64)
                destination[element] = valid && element < depth ? source[element] : (unsigned char)0;
    }
    // Also retires the previous item's reads of every buffer this item refills.
    __syncthreads();
    unsigned const block_begin = min(unions[0][0], unions[1][0]), block_end = max(unions[0][1], unions[1][1]);
    unsigned const panel_first = block_begin < block_end ? block_begin / nk_attention_panel_k : 0;
    unsigned const panel_end = block_begin < block_end ? (block_end + nk_attention_panel_k - 1) / nk_attention_panel_k
                                                       : 0;

    nk_u32_t queries[2][nk_attention_steps_cdna4_k][8];
#pragma unroll
    for (unsigned tile = 0; tile < 2; ++tile) {
        unsigned char const *row = queries_shared +
                                   (wave * nk_attention_wave_rows_cdna4_k + tile * 16 + column) * row_stride;
#pragma unroll
        for (unsigned step = 0; step < nk_attention_steps_cdna4_k; ++step)
            if ((step_mask >> step) & 1)
                nk_attention_load_fragment_cdna4_(row, step * step_bytes + group * step_words * 4, row_bytes,
                                                  step_words, queries[tile][step]);
    }
    // Every Q read retires before the first V lands on it.
    __syncthreads();

    nk_f32_t row_max[2] = {negative_infinity, negative_infinity}, row_sum[2] = {0, 0};
    nk_fui32_t output[2][nk_attention_tiles_cdna4_k][4];
#pragma unroll
    for (unsigned tile = 0; tile < 2; ++tile)
#pragma unroll
        for (unsigned depth_tile = 0; depth_tile < nk_attention_tiles_cdna4_k; ++depth_tile)
#pragma unroll
            for (unsigned element = 0; element < 4; ++element) output[tile][depth_tile][element].u = 0;
    nk_f32_t const scale2 = arguments->scale2 * arguments->score_scale;
    int const active[2] = {wave * nk_attention_wave_rows_cdna4_k < work->row_count,
                           wave * nk_attention_wave_rows_cdna4_k + 16 < work->row_count};

    for (unsigned panel_index = panel_first; panel_index < panel_end; ++panel_index) {
        nk_size_t const panel_position = (nk_size_t)panel_index * nk_attention_panel_k;
        nk_attention_stage_rows_cdna4_(keys_shared, keys_plane, panel_position, nk_attention_panel_k, row_bytes,
                                       row_stride);
        nk_attention_stage_values_cdna4_(dtype, values_shared, values_plane, positions_padded, panel_position,
                                         row_bytes, value_stride);
        __syncthreads();

        nk_fui32_t tile_scores[2][4][4];
#pragma unroll
        for (unsigned tile = 0; tile < 2; ++tile)
#pragma unroll
            for (unsigned position_tile = 0; position_tile < 4; ++position_tile)
#pragma unroll
                for (unsigned element = 0; element < 4; ++element) tile_scores[tile][position_tile][element].u = 0;
#pragma unroll
        for (unsigned step = 0; step < nk_attention_steps_cdna4_k; ++step) {
            if (!((step_mask >> step) & 1)) continue;
#pragma unroll
            for (unsigned position_tile = 0; position_tile < 4; ++position_tile) {
                unsigned const position = (unsigned)nk_attention_slot_position_(
                    nk_attention_tile_slot_cdna4_(dtype, position_tile, column));
                nk_u32_t keys[8];
                nk_attention_load_fragment_cdna4_(keys_shared + position * row_stride,
                                                  step * step_bytes + group * step_words * 4, row_bytes, step_words,
                                                  keys);
#pragma unroll
                for (unsigned tile = 0; tile < 2; ++tile)
                    if (active[tile]) scores(tile_scores[tile][position_tile], keys, queries[tile][step]);
            }
        }

        nk_u32_t probabilities[2][2][4];
        nk_f32_t correction[2] = {1.0f, 1.0f};
#pragma unroll
        for (unsigned tile = 0; tile < 2; ++tile) {
            nk_f32_t chunk_max = negative_infinity;
#pragma unroll
            for (unsigned position_tile = 0; position_tile < 4; ++position_tile)
#pragma unroll
                for (unsigned element = 0; element < 4; ++element) {
                    nk_fui32_t *score = &tile_scores[tile][position_tile][element];
                    score->f = (epilogue == nk_cross_epilogue_i32_to_f32_k ? (nk_f32_t)score->i : score->f) * scale2;
                    nk_size_t const position = panel_position +
                                               nk_attention_slot_position_(nk_attention_tile_slot_cdna4_(
                                                   dtype, position_tile, group * 4 + element));
                    if (position < key_begin[tile] || position >= key_end[tile]) score->f = negative_infinity;
                    chunk_max = fmaxf(chunk_max, score->f);
                }
            // The other 3 lane groups hold the rest of this lane's query row.
            chunk_max = fmaxf(chunk_max, nk_shuffle_xor_f32_(chunk_max, 16));
            chunk_max = fmaxf(chunk_max, nk_shuffle_xor_f32_(chunk_max, 32));
            nk_f32_t const new_max = fmaxf(row_max[tile], chunk_max);
            // Subtracting 0 while every key so far is masked keeps `exp2(-∞ - max)` from NaN.
            nk_f32_t const subtrahend = new_max == negative_infinity ? 0.0f : new_max;
            correction[tile] = nk_f32_exp2_(row_max[tile] - subtrahend);
            row_max[tile] = new_max;
            row_sum[tile] *= correction[tile];
#pragma unroll
            for (unsigned value_step = 0; value_step < 2; ++value_step) {
                if (value_step >= value_steps) continue;
                nk_f32_t step_probabilities[16];
#pragma unroll
                for (unsigned index = 0; index < 16; ++index) {
                    unsigned const position_tile = dtype == nk_i8_k ? index / 4 : value_step * 2 + (index / 4 & 1);
                    step_probabilities[index] = nk_f32_exp2_(tile_scores[tile][position_tile][index % 4].f -
                                                             subtrahend);
                }
                weights(step_probabilities, probabilities[tile][value_step], &row_sum[tile]);
            }
        }

        // This lane's 4 rows of P · V output take the corrections the lanes of those rows hold.
        nk_f32_t row_corrections[2][4];
#pragma unroll
        for (unsigned tile = 0; tile < 2; ++tile)
#pragma unroll
            for (unsigned element = 0; element < 4; ++element)
                row_corrections[tile][element] = nk_shuffle_f32_cdna4_(correction[tile], group * 4 + element);

#pragma unroll
        for (unsigned depth_tile = 0; depth_tile < nk_attention_tiles_cdna4_k; ++depth_tile) {
            if (!((tile_mask >> depth_tile) & 1)) continue;
            unsigned char const *values_row = values_shared + (depth_tile * 16 + column) * value_stride;
            if (dtype == nk_i8_k) {
                nk_u32_t values[4];
                uint4 const codes = *(uint4 const *)(values_row + group * 16);
                values[0] = codes.x, values[1] = codes.y, values[2] = codes.z, values[3] = codes.w;
                // Σ V over the panel's 64 slots of this depth column restores the weights' offset.
                nk_i32_t values_sum = 0;
#pragma unroll
                for (unsigned word = 0; word < 4; ++word)
                    values_sum = nk_dot_i8x4_(values[word], 0x01010101u, values_sum);
                values_sum += nk_shuffle_xor_i32_(values_sum, 16);
                values_sum += nk_shuffle_xor_i32_(values_sum, 32);
#pragma unroll
                for (unsigned tile = 0; tile < 2; ++tile) {
                    if (!active[tile]) continue;
                    nk_fui32_t sums[4] = {{0}, {0}, {0}, {0}};
                    values_mma(sums, probabilities[tile][0], values);
#pragma unroll
                    for (unsigned element = 0; element < 4; ++element)
                        output[tile][depth_tile][element].f = fmaf(
                            output[tile][depth_tile][element].f, row_corrections[tile][element],
                            (nk_f32_t)(nk_i32_t)((nk_u32_t)sums[element].i + 128u * values_sum));
                }
                continue;
            }
#pragma unroll
            for (unsigned tile = 0; tile < 2; ++tile)
#pragma unroll
                for (unsigned element = 0; element < 4; ++element)
                    output[tile][depth_tile][element].f *= row_corrections[tile][element];
#pragma unroll
            for (unsigned value_step = 0; value_step < 2; ++value_step) {
                unsigned const slot = value_step * 32 + group * 8;
                nk_u32_t values[4];
                if (dtype == nk_bf16_k) {
                    uint4 const halves = *(uint4 const *)(values_row + slot * 2);
                    values[0] = halves.x, values[1] = halves.y, values[2] = halves.z, values[3] = halves.w;
                }
                else {
                    uint2 const codes = *(uint2 const *)(values_row + slot);
                    values[0] = codes.x, values[1] = codes.y, values[2] = 0, values[3] = 0;
                }
#pragma unroll
                for (unsigned tile = 0; tile < 2; ++tile)
                    if (active[tile]) values_mma(output[tile][depth_tile], probabilities[tile][value_step], values);
            }
        }
        // Every read of the panel retires before the next one is staged over it.
        __syncthreads();
    }

#pragma unroll
    for (unsigned tile = 0; tile < 2; ++tile) {
        nk_f32_t total = row_sum[tile];
        total += nk_shuffle_xor_f32_(total, 16);
        total += nk_shuffle_xor_f32_(total, 32);
        nk_f32_t inverses[4];
#pragma unroll
        for (unsigned element = 0; element < 4; ++element) {
            nk_f32_t const sum = nk_shuffle_f32_cdna4_(total, group * 4 + element);
            inverses[element] = sum > 0 ? arguments->output_scale / sum : 0.0f;
        }
#pragma unroll
        for (unsigned element = 0; element < 4; ++element) {
            unsigned const local = wave * nk_attention_wave_rows_cdna4_k + tile * 16 + group * 4 + element;
            if (local >= work->row_count) continue;
            nk_size_t const row = work->row_first + local;
            nk_size_t const query = row / work->heads_selected;
            nk_size_t const head = work->head_first + row % work->heads_selected;
            nk_f32_t *destination = (nk_f32_t *)((unsigned char *)arguments->output +
                                                 (work->query_first + query) * arguments->output_stride) +
                                    head * depth;
#pragma unroll
            for (unsigned depth_tile = 0; depth_tile < nk_attention_tiles_cdna4_k; ++depth_tile)
                if ((store_mask >> depth_tile) & 1)
                    destination[depth_tile * 16 + column] = output[tile][depth_tile][element].f * inverses[element];
        }
    }
}

/**
 *  @brief Every work item of a launch, walked with a stride of the grid.
 *  @sa nk_attention_block_cdna4_ for the parameters.
 */
NUMKONG_DEVICE void nk_attention_tile_cdna4_(nk_dtype_t dtype, nk_attention_width_t width, nk_cross_epilogue_t epilogue,
                                             nk_attention_scores_cdna4_t scores, nk_attention_values_cdna4_t values_mma,
                                             nk_attention_weights_cdna4_t weights,
                                             nk_attention_arguments_t const *arguments) {
    extern __shared__ __attribute__((aligned(16))) unsigned char nk_attention_shared_cdna4_[];
    __shared__ nk_u64_t prefix[nk_attention_threads_k + 1];
    __shared__ nk_u64_t warp_totals[nk_attention_threads_k / 32];
    __shared__ unsigned unions[2][2];
    nk_attention_schedule_t schedule;
    if (!nk_attention_schedule_start_(arguments, &schedule, prefix, warp_totals)) return;
    nk_attention_work_t work;
    for (nk_size_t item = blockIdx.x; nk_attention_schedule_next_(&schedule, prefix, warp_totals, item, &work);
         item += gridDim.x)
        nk_attention_block_cdna4_(dtype, width, epilogue, scores, values_mma, weights, arguments, &work,
                                  nk_attention_shared_cdna4_, unions);
}

#pragma endregion Tile

#pragma region Instantiations

nk_define_device_attention_pack_size_(bf16, cdna4, 2)
nk_define_device_attention_packed_shape_(bf16, cdna4)
nk_define_device_attention_pack_(bf16, cdna4, bf16)
nk_define_device_attention_packed_(bf16, cdna4, cdna4, nk_attention_launch_cdna4_, bf16, nk_cross_epilogue_f32_k,
                                   nk_attention_scores_bf16_cdna4_, nk_attention_values_bf16_cdna4_,
                                   nk_attention_weights_bf16_cdna4_, 1.0f, 1.0f)

nk_define_device_attention_pack_size_(e4m3, cdna4, 1)
nk_define_device_attention_packed_shape_(e4m3, cdna4)
nk_define_device_attention_pack_(e4m3, cdna4, e4m3)
nk_define_device_attention_packed_(e4m3, cdna4, cdna4, nk_attention_launch_cdna4_, e4m3, nk_cross_epilogue_f32_k,
                                   nk_attention_scores_e4m3_cdna4_, nk_attention_values_e4m3_cdna4_,
                                   nk_attention_weights_f16_cdna4_, 1.0f, 256.0f)

nk_define_device_attention_pack_size_(i8, cdna4, 1)
nk_define_device_attention_packed_shape_(i8, cdna4)
nk_define_device_attention_pack_(i8, cdna4, i8)
nk_define_device_attention_packed_(i8, cdna4, cdna4, nk_attention_launch_cdna4_, i8, nk_cross_epilogue_i32_to_f32_k,
                                   nk_attention_scores_i8_cdna4_, nk_attention_values_i8_cdna4_,
                                   nk_attention_weights_u8_cdna4_, 1.0f, 1.0f)

#pragma endregion Instantiations

#endif // NUMKONG_TARGET_CDNA4

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_ROCM_CDNA4_
#endif // NUMKONG_ATTENTION_CDNA4_CUH
