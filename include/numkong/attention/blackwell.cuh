/**
 *  @file include/numkong/attention/blackwell.cuh
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Ragged attention for the NVIDIA compute capability 10.x family.
 *
 *  @sa include/numkong/attention.h
 *  @sa include/numkong/attention/ampere.cuh
 *  @sa include/numkong/dots/blackwell.cuh
 *
 *  The Ampere FlashAttention-2 item, its 64 rows and 64-position panels, with both products on
 *  `tcgen05.mma` instead of warp-level MMA. One thread multiplies Q by a K panel into tensor memory
 *  as a 64-row tile, whose `16x256b` loads hand every warp its 16 rows in the very fragment layout
 *  Ampere's online softmax reads, then multiplies P, written back to shared memory, by the V panel
 *  into the other half of the same tensor-memory lanes, which the warps add to their register
 *  accumulators. Every operand is staged with `cp.async` into the canonical swizzled layouts, and P
 *  is quantized to e4m3(256 · p) against the transposed V codes. BF16 stays on Ampere's kernels,
 *  faster than these 64-row steps. The pack layout and the work scheduler come from @c cuda, and
 *  depths above 256 fall back to its kernel.
 */
#ifndef NUMKONG_ATTENTION_BLACKWELL_CUH
#define NUMKONG_ATTENTION_BLACKWELL_CUH

#if NUMKONG_TARGET_BLACKWELL

#include "numkong/attention/ampere.cuh" // `nk_attention_weights_ampere_t`, `nk_copy_b128_async_ampere_`
#include "numkong/attention/ada.cuh"    // `nk_attention_weights_e4m3_ada_`
#include "numkong/dots/blackwell.cuh"   // `nk_mma_f8f6f4_blackwell_`, `nk_mbarrier_wait_blackwell_`

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Configuration

enum {
    nk_attention_chunk_bytes_blackwell_k = nk_attention_block_rows_k * 128,
    nk_attention_narrow_depth_blackwell_k = 128,
    nk_attention_wide_depth_blackwell_k = 256,
};

/** Issues one 64-row step into the accumulator at tensor-memory address @p accumulator, @p columns
 *  wide, adding to it unless @p accumulate is zero. */
typedef void (*nk_attention_mma_blackwell_t)(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b, nk_u32_t columns,
                                             nk_u32_t accumulate);

#pragma endregion Configuration

#pragma region Instructions

#if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1000 && \
    __CUDA_ARCH_FAMILY_SPECIFIC__ < 1100

/* Reads 8 columns of 16 lanes into the accumulator fragment of an m16n8 step: this thread's rows
 *  lane / 4 and lane / 4 + 8, columns 2 · (lane % 4) and the next. */
NUMKONG_DEVICE void nk_tmem_load_16x256b_x2_blackwell_(nk_u32_t address, nk_u32_t values[8]) {
    asm volatile("tcgen05.ld.sync.aligned.16x256b.x2.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n" //
                 "tcgen05.wait::ld.sync.aligned;\n"
                 : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]), "=r"(values[3]), "=r"(values[4]), "=r"(values[5]),
                   "=r"(values[6]), "=r"(values[7])
                 : "r"(address)
                 : "memory");
}

/* Reads 64 columns of 16 lanes as 8 consecutive m16n8 accumulator fragments. */
NUMKONG_DEVICE void nk_tmem_load_16x256b_x8_blackwell_(nk_u32_t address, nk_u32_t values[32]) {
    asm volatile("tcgen05.ld.sync.aligned.16x256b.x8.b32 "                                                   //
                 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "                   //
                 "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];\n" //
                 "tcgen05.wait::ld.sync.aligned;\n"
                 : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]), "=r"(values[3]), "=r"(values[4]), "=r"(values[5]),
                   "=r"(values[6]), "=r"(values[7]), "=r"(values[8]), "=r"(values[9]), "=r"(values[10]),
                   "=r"(values[11]), "=r"(values[12]), "=r"(values[13]), "=r"(values[14]), "=r"(values[15]),
                   "=r"(values[16]), "=r"(values[17]), "=r"(values[18]), "=r"(values[19]), "=r"(values[20]),
                   "=r"(values[21]), "=r"(values[22]), "=r"(values[23]), "=r"(values[24]), "=r"(values[25]),
                   "=r"(values[26]), "=r"(values[27]), "=r"(values[28]), "=r"(values[29]), "=r"(values[30]),
                   "=r"(values[31])
                 : "r"(address)
                 : "memory");
}

#else

NUMKONG_DEVICE void nk_tmem_load_16x256b_x2_blackwell_(nk_u32_t address, nk_u32_t values[8]) { __trap(); }
NUMKONG_DEVICE void nk_tmem_load_16x256b_x8_blackwell_(nk_u32_t address, nk_u32_t values[32]) { __trap(); }

#endif // __CUDA_ARCH_FAMILY_SPECIFIC__ in the 10.x family

#pragma endregion Instructions

#pragma region Fragments

/** The shared-memory descriptor of an operand at @p shared: K-major 128-byte swizzled rows for
 *  @p layout 2 or 64-byte ones for 4, eight-row groups @p stride_bytes apart, and for an MN-major
 *  operand, swizzle-wide column blocks @p leading_bytes apart. */
NUMKONG_DEVICE nk_u64_t nk_attention_descriptor_blackwell_(nk_u32_t shared, nk_u32_t leading_bytes,
                                                           nk_u32_t stride_bytes, nk_u32_t layout) {
    return (nk_u64_t)((shared & 0x3FFFFu) >> 4) | ((nk_u64_t)(leading_bytes >> 4) << 16) |
           ((nk_u64_t)(stride_bytes >> 4) << 32) | ((nk_u64_t)1 << 46) | ((nk_u64_t)layout << 61);
}

/** The instruction descriptor of a 64-row step from K-major E4M3 A and B into @p columns F32
 *  accumulator columns. */
NUMKONG_DEVICE nk_u32_t nk_attention_instruction_blackwell_(nk_u32_t columns) {
    return (1u << 4) | ((columns >> 3) << 17) | ((nk_u32_t)(nk_attention_block_rows_k >> 4) << 24);
}

/** One step of S or P · V from E4M3 codes, both K-major. */
NUMKONG_DEVICE void nk_attention_mma_e4m3_blackwell_(nk_u32_t accumulator, nk_u64_t a, nk_u64_t b, nk_u32_t columns,
                                                     nk_u32_t accumulate) {
    nk_mma_f8f6f4_blackwell_(accumulator, a, b, nk_attention_instruction_blackwell_(columns), accumulate);
}

/** Byte offset of byte @p byte of row @p row in 64 rows of 128-byte swizzled chunks, 8 KB apart. */
NUMKONG_DEVICE nk_u32_t nk_attention_swizzled_blackwell_(unsigned row, unsigned byte) {
    return (byte >> 7) * nk_attention_chunk_bytes_blackwell_k + row * 128 + ((((byte >> 4) & 7) ^ (row & 7)) << 4) +
           (byte & 15);
}

/** Byte offset of byte @p byte of row @p row in 64-byte swizzled rows. */
NUMKONG_DEVICE nk_u32_t nk_attention_swizzled_narrow_blackwell_(unsigned row, unsigned byte) {
    return row * 64 + ((((byte >> 4) & 3) ^ ((row >> 1) & 3)) << 4) + (byte & 15);
}

/** Issues a 64-position panel of a position-major plane into 128-byte swizzled chunks. */
NUMKONG_DEVICE void nk_attention_stage_rows_blackwell_(unsigned char *shared, unsigned char const *plane,
                                                       nk_size_t first_position, unsigned row_bytes) {
    unsigned const chunks_per_row = row_bytes >> 4, chunks = nk_attention_panel_k * chunks_per_row;
    for (unsigned chunk = threadIdx.x; chunk < chunks; chunk += nk_attention_threads_k) {
        unsigned const row = chunk / chunks_per_row, column = chunk - row * chunks_per_row;
        nk_copy_b128_async_ampere_(
            nk_shared_address_ampere_(shared + nk_attention_swizzled_blackwell_(row, column << 4)),
            plane + (first_position + row) * row_bytes + (column << 4), 16);
    }
}

/** Issues 64 panel slots of every depth row of a transposed V plane into 64-byte swizzled rows. */
NUMKONG_DEVICE void nk_attention_stage_columns_blackwell_(unsigned char *shared, unsigned char const *plane,
                                                          nk_size_t positions_padded, nk_size_t first_position,
                                                          unsigned depth_rows) {
    for (unsigned chunk = threadIdx.x; chunk < depth_rows * 4; chunk += nk_attention_threads_k) {
        unsigned const row = chunk >> 2, column = chunk & 3;
        nk_copy_b128_async_ampere_(
            nk_shared_address_ampere_(shared + nk_attention_swizzled_narrow_blackwell_(row, column << 4)),
            plane + row * positions_padded + first_position + (column << 4), 16);
    }
}

/** Makes this thread's staged operands visible to the tensor cores, then meets every thread. */
NUMKONG_DEVICE void nk_attention_publish_blackwell_(void) {
    nk_fence_async_shared_blackwell_();
    nk_tmem_fence_before_blackwell_();
    __syncthreads();
    nk_tmem_fence_after_blackwell_();
}

#pragma endregion Fragments

#pragma region Tile

/**
 *  @brief One work item on one block: S = Q · Kᵀ and O += P · V per 64-position panel on the tensor
 *      cores, with Ampere's online softmax in between.
 *  @param[in] width Sizes the output accumulators: 128 or 256 depths.
 *  @param[in] epilogue Only @c nk_cross_epilogue_f32_k: both products accumulate in F32.
 *  @param[in] scores One step of S, see @c nk_attention_mma_blackwell_t.
 *  @param[in] values_mma One step of P · V, see @c nk_attention_mma_blackwell_t.
 *  @param[in] weights P from probabilities, see @c nk_attention_weights_ampere_t.
 *  @param[in] tmem The tensor-memory address of this block's columns.
 *  @param[inout] phase The parity both tensor-core barriers wait on next.
 *
 *  Warp @p w holds rows 16w to 16w + 15: scores in tensor-memory lanes 32w to 32w + 15 and P · V in
 *  the 16 lanes after them, both from the first column.
 */
NUMKONG_DEVICE void nk_attention_block_blackwell_(
    nk_attention_width_t width, nk_cross_epilogue_t epilogue, nk_attention_mma_blackwell_t scores,
    nk_attention_mma_blackwell_t values_mma, nk_attention_weights_ampere_t weights,
    nk_attention_arguments_t const *arguments, nk_attention_work_t const *work, unsigned char *shared,
    nk_u32_t shared_address, nk_u32_t tmem, nk_u32_t *phase, unsigned (*unions)[2]) {
    nk_unused_(epilogue);

    nk_f32_t const negative_infinity = nk_attention_negative_infinity_();
    unsigned const lane = threadIdx.x & 31, warp = threadIdx.x >> 5, group = lane >> 2, quad = lane & 3;
    unsigned const max_tiles = width == nk_attention_width_narrow_k ? nk_attention_narrow_depth_blackwell_k / 8
                                                                    : nk_attention_wide_depth_blackwell_k / 8;
    nk_size_t const depth = arguments->depth;
    unsigned const row_bytes = (unsigned)nk_size_round_up_to_multiple_(depth, nk_attention_step_bytes_k);
    unsigned const depth_steps = row_bytes / nk_attention_step_bytes_k, depth_tiles = row_bytes / 8;
    unsigned const chunk_bytes = (unsigned)nk_size_divide_round_up_(row_bytes, 128) *
                                 nk_attention_chunk_bytes_blackwell_k;
    unsigned const value_bytes = row_bytes * nk_attention_panel_k;
    unsigned const keys_offset = chunk_bytes, values_offset = 3 * chunk_bytes;
    unsigned const weights_offset = values_offset + 2 * value_bytes;
    nk_u32_t const barriers = shared_address + weights_offset + 4096;
    nk_u32_t const scores_barrier = barriers, values_barrier = barriers + 8;

    nk_size_t length, plane_bytes;
    unsigned char const *keys_plane = nk_attention_keys_plane_(arguments, work, row_bytes, &length, &plane_bytes);
    unsigned char const *values_plane = keys_plane + arguments->key_value_head_count * plane_bytes;
    nk_size_t const positions_padded = plane_bytes / row_bytes;

    // Each thread holds rows `group` and `group + 8` of its warp's 16.
    unsigned key_begin[2], key_end[2];
    unsigned union_begin = 0xFFFFFFFFu, union_end = 0, common_begin = 0, common_end = 0xFFFFFFFFu;
#pragma unroll
    for (unsigned half = 0; half < 2; ++half) {
        unsigned const local = warp * 16 + group + half * 8;
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
    unsigned const panel = nk_attention_panel_k;
    unsigned const panel_first = block_begin < block_end ? block_begin / panel : 0;
    unsigned const panel_end = block_begin < block_end ? (unsigned)nk_size_divide_round_up_(block_end, panel) : 0;
    // No panel means no barrier below, and the next item would rewrite `unions` under slow warps.
    if (panel_first == panel_end) __syncthreads();

    if (panel_first < panel_end)
        nk_attention_stage_rows_blackwell_(shared + keys_offset, keys_plane, (nk_size_t)panel_first * panel, row_bytes);
    nk_commit_async_ampere_();

    for (unsigned local = warp; local < nk_attention_block_rows_k; local += 4) {
        int const valid = local < work->row_count;
        unsigned char const *source = arguments->queries;
        if (valid) {
            nk_size_t const row = work->row_first + local;
            nk_size_t const query = row / work->heads_selected;
            nk_size_t const head = work->head_first + row % work->heads_selected;
            source += (work->query_first + query) * arguments->query_stride + head * depth;
        }
        for (unsigned element = lane; element < row_bytes; element += 32)
            shared[nk_attention_swizzled_blackwell_(local, element)] = valid && element < depth ? source[element]
                                                                                                : (unsigned char)0;
    }

    nk_f32_t row_max[2] = {negative_infinity, negative_infinity}, row_sum[2] = {0, 0};
    nk_fui32_t output[nk_attention_wide_depth_blackwell_k / 8][4];
#pragma unroll
    for (unsigned tile = 0; tile < max_tiles; ++tile)
#pragma unroll
        for (unsigned element = 0; element < 4; ++element) output[tile][element].u = 0;
    nk_f32_t const scale2 = arguments->scale2 * arguments->score_scale;
    nk_u32_t const scores_lanes = tmem + ((warp * 32) << 16), values_lanes = tmem + ((warp * 32 + 16) << 16);

    for (unsigned panel_index = panel_first; panel_index < panel_end; ++panel_index) {
        unsigned const buffer = (panel_index - panel_first) & 1;
        nk_size_t const panel_position = (nk_size_t)panel_index * panel;
        unsigned char *const keys_shared = shared + keys_offset + buffer * chunk_bytes;
        unsigned char *const values_shared = shared + values_offset + buffer * value_bytes;
        nk_attention_stage_columns_blackwell_(values_shared, values_plane, positions_padded, panel_position, row_bytes);
        nk_commit_async_ampere_();
        if (panel_index + 1 < panel_end)
            nk_attention_stage_rows_blackwell_(shared + keys_offset + (buffer ^ 1) * chunk_bytes, keys_plane,
                                               panel_position + panel, row_bytes);
        nk_commit_async_ampere_();

        // This panel's K has landed; its V and the next K may still be in flight.
        nk_wait_async_ampere_(2);
        nk_attention_publish_blackwell_();
        if (threadIdx.x == 0) {
            nk_u32_t const queries_address = shared_address,
                           keys_address = shared_address + (nk_u32_t)(keys_shared - shared);
            for (unsigned step = 0; step < depth_steps; ++step) {
                nk_u32_t const offset = (step >> 2) * nk_attention_chunk_bytes_blackwell_k + (step & 3) * 32;
                nk_u64_t const queries = nk_attention_descriptor_blackwell_(queries_address + offset, 16, 1024, 2);
                nk_u64_t const keys = nk_attention_descriptor_blackwell_(keys_address + offset, 16, 1024, 2);
                scores(tmem, queries, keys, panel, step != 0);
            }
            nk_mma_commit_blackwell_(scores_barrier);
        }
        nk_mbarrier_wait_blackwell_(scores_barrier, *phase);
        nk_tmem_fence_after_blackwell_();

        nk_fui32_t tile_scores[8][4];
        nk_tmem_load_16x256b_x8_blackwell_(scores_lanes, (nk_u32_t *)tile_scores);
#pragma unroll
        for (unsigned tile = 0; tile < 8; ++tile)
#pragma unroll
            for (unsigned element = 0; element < 4; ++element) tile_scores[tile][element].f *= scale2;
        unsigned const chunk_begin = (unsigned)panel_position, chunk_end = chunk_begin + panel;
        // Padded tails and causal edges fall outside some row's keys; inner panels skip this.
        if (!(common_begin <= chunk_begin && chunk_end <= common_end)) {
#pragma unroll
            for (unsigned tile = 0; tile < 8; ++tile)
#pragma unroll
                for (unsigned element = 0; element < 4; ++element) {
                    unsigned const position = chunk_begin + tile * 8 + quad * 2 + (element & 1);
                    unsigned const half = element >> 1;
                    if (position < key_begin[half] || position >= key_end[half])
                        tile_scores[tile][element].f = negative_infinity;
                }
        }
        nk_f32_t subtrahend[2], correction[2];
#pragma unroll
        for (unsigned half = 0; half < 2; ++half) {
            nk_f32_t chunk_max = negative_infinity;
#pragma unroll
            for (unsigned tile = 0; tile < 8; ++tile)
                chunk_max = fmaxf(chunk_max, fmaxf(tile_scores[tile][half * 2].f, tile_scores[tile][half * 2 + 1].f));
            chunk_max = fmaxf(chunk_max, __shfl_xor_sync(0xFFFFFFFFu, chunk_max, 1));
            chunk_max = fmaxf(chunk_max, __shfl_xor_sync(0xFFFFFFFFu, chunk_max, 2));
            nk_f32_t const new_max = fmaxf(row_max[half], chunk_max);
            // With every key so far masked, subtracting 0 keeps exp2(-∞ - max) from being NaN.
            subtrahend[half] = new_max == negative_infinity ? 0.0f : new_max;
            correction[half] = nk_f32_exp2_(row_max[half] - subtrahend[half]);
            row_max[half] = new_max;
            row_sum[half] *= correction[half];
        }
#pragma unroll
        for (unsigned tile = 0; tile < 8; ++tile)
#pragma unroll
            for (unsigned element = 0; element < 4; ++element)
                tile_scores[tile][element].f = nk_f32_exp2_(tile_scores[tile][element].f - subtrahend[element >> 1]);
#pragma unroll
        for (unsigned position_group = 0; position_group < 4; ++position_group)
#pragma unroll
            for (unsigned half = 0; half < 2; ++half) {
                nk_f32_t const four[4] = {tile_scores[position_group * 2][half * 2].f,
                                          tile_scores[position_group * 2][half * 2 + 1].f,
                                          tile_scores[position_group * 2 + 1][half * 2].f,
                                          tile_scores[position_group * 2 + 1][half * 2 + 1].f};
                nk_u32_t packed[2];
                weights(four, packed, &row_sum[half]);
                unsigned const row = warp * 16 + group + half * 8;
                // P follows the V pack's slot order.
                *(nk_u32_t *)(shared + weights_offset +
                              nk_attention_swizzled_narrow_blackwell_(row, position_group * 16 + quad * 4)) = packed[0];
            }
#pragma unroll
        for (unsigned tile = 0; tile < max_tiles; ++tile)
            if (tile < depth_tiles)
#pragma unroll
                for (unsigned element = 0; element < 4; ++element) output[tile][element].f *= correction[element >> 1];

        // V has landed, and P is written; the next K may still be in flight.
        nk_wait_async_ampere_(1);
        nk_attention_publish_blackwell_();
        if (threadIdx.x == 0) {
            nk_u32_t const weights_address = shared_address + weights_offset;
            nk_u32_t const values_address = shared_address + (nk_u32_t)(values_shared - shared);
            for (unsigned step = 0; step < panel / 32; ++step) {
                nk_u64_t const probabilities = nk_attention_descriptor_blackwell_(weights_address + step * 32, 16, 512,
                                                                                  4);
                nk_u64_t const values = nk_attention_descriptor_blackwell_(values_address + step * 32, 16, 512, 4);
                values_mma(tmem + (16 << 16), probabilities, values, row_bytes, step != 0);
            }
            nk_mma_commit_blackwell_(values_barrier);
        }
        nk_mbarrier_wait_blackwell_(values_barrier, *phase);
        nk_tmem_fence_after_blackwell_();
#pragma unroll
        for (unsigned pair = 0; pair < max_tiles / 2; ++pair) {
            if (pair * 2 >= depth_tiles) continue;
            nk_fui32_t sums[2][4];
            nk_tmem_load_16x256b_x2_blackwell_(values_lanes + pair * 16, (nk_u32_t *)sums);
#pragma unroll
            for (unsigned element = 0; element < 4; ++element)
                output[pair * 2][element].f += sums[0][element].f,
                    output[pair * 2 + 1][element].f += sums[1][element].f;
        }
        nk_tmem_fence_before_blackwell_();
        *phase ^= 1;
    }

#pragma unroll
    for (unsigned half = 0; half < 2; ++half) {
        row_sum[half] += __shfl_xor_sync(0xFFFFFFFFu, row_sum[half], 1);
        row_sum[half] += __shfl_xor_sync(0xFFFFFFFFu, row_sum[half], 2);
    }
#pragma unroll
    for (unsigned half = 0; half < 2; ++half) {
        unsigned const local = warp * 16 + group + half * 8;
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
 *  @brief Every work item of a launch, walked with a stride of the grid, after allocating the
 *      tensor-memory columns both products share.
 *  @sa nk_attention_block_blackwell_ for the parameters.
 */
NUMKONG_DEVICE void nk_attention_tile_blackwell_(nk_attention_kind_t kind, nk_attention_width_t width,
                                                 nk_cross_epilogue_t epilogue, nk_attention_mma_blackwell_t scores,
                                                 nk_attention_mma_blackwell_t values_mma,
                                                 nk_attention_weights_ampere_t weights,
                                                 nk_attention_arguments_t const *arguments) {
    extern __shared__ unsigned char nk_attention_shared_blackwell_[];
    __shared__ nk_u64_t prefix[nk_attention_threads_k + 1];
    __shared__ nk_u64_t warp_totals[nk_attention_threads_k / 32];
    __shared__ unsigned unions[nk_attention_threads_k / 32][2];
    __shared__ nk_u32_t holder;
    nk_attention_schedule_t schedule;
    if (!nk_attention_schedule_start_(arguments, &schedule, prefix, warp_totals)) return;

    nk_u32_t const shared_origin = nk_shared_address_ampere_(nk_attention_shared_blackwell_);
    nk_u32_t const padding = (1024 - (shared_origin & 1023)) & 1023;
    unsigned char *const shared = nk_attention_shared_blackwell_ + padding;
    nk_u32_t const shared_address = shared_origin + padding;
    nk_u32_t const columns = width == nk_attention_width_narrow_k ? nk_attention_narrow_depth_blackwell_k
                                                                  : nk_attention_wide_depth_blackwell_k;
    // Both barriers sit past the operands, wherever the depth puts them.
    nk_unused_(kind);
    nk_size_t const row_bytes = nk_size_round_up_to_multiple_(arguments->depth, nk_attention_step_bytes_k);
    nk_size_t const chunk_bytes = nk_size_divide_round_up_(row_bytes, 128) * nk_attention_chunk_bytes_blackwell_k;
    nk_size_t const value_bytes = row_bytes * nk_attention_panel_k;
    nk_u32_t const barriers = shared_address + (nk_u32_t)(3 * chunk_bytes + 2 * value_bytes) + 4096;
    if (threadIdx.x == 0) {
        nk_mbarrier_init_blackwell_(barriers, 1);
        nk_mbarrier_init_blackwell_(barriers + 8, 1);
        nk_mbarrier_init_fence_blackwell_();
    }
    if (threadIdx.x < 32) nk_tmem_alloc_blackwell_(nk_shared_address_ampere_(&holder), columns);
    nk_tmem_fence_before_blackwell_();
    __syncthreads();
    nk_tmem_fence_after_blackwell_();
    nk_u32_t const tmem = holder;

    nk_u32_t phase = 0;
    nk_attention_work_t work;
    for (nk_size_t item = blockIdx.x; nk_attention_schedule_next_(&schedule, prefix, warp_totals, item, &work);
         item += gridDim.x)
        nk_attention_block_blackwell_(width, epilogue, scores, values_mma, weights, arguments, &work, shared,
                                      shared_address, tmem, &phase, unions);

    nk_tmem_fence_before_blackwell_();
    __syncthreads();
    if (threadIdx.x < 32) {
        nk_tmem_fence_after_blackwell_();
        nk_tmem_dealloc_blackwell_(tmem, columns);
    }
}

#pragma endregion Tile

#pragma region Launch

/** Dynamic shared memory of one block: Q, two K and two V panels, P and both barriers, plus the
 *  slack that aligns them to 1024 bytes. */
NUMKONG_INLINE nk_size_t nk_attention_shared_bytes_blackwell_(nk_size_t depth) {
    nk_size_t const row_bytes = nk_size_round_up_to_multiple_(depth, nk_attention_step_bytes_k);
    nk_size_t const chunk_bytes = nk_size_divide_round_up_(row_bytes, 128) * nk_attention_chunk_bytes_blackwell_k;
    return 1024 + 3 * chunk_bytes + 2 * row_bytes * nk_attention_panel_k + 4096 + 16;
}

/**
 *  @brief Validates the contract and launches the kernel for the depth's width, capping the blocks
 *      per multiprocessor at what tensor memory holds.
 *  @param[in] score_scale Undoes the Q and K widenings in the tile, or 1.
 *  @param[in] output_scale Undoes the V widening in the tile, or 1.
 */
NUMKONG_INLINE nk_status_t nk_attention_launch_blackwell_(
    void const *narrow_kernel, void const *wide_kernel, void const *fallback_kernel, nk_attention_kind_t kind,
    void const *queries, void const *packed, nk_f32_t *output, nk_size_t head_count, nk_size_t key_value_head_count,
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale,
    nk_f32_t score_scale, nk_f32_t output_scale, nk_attention_mask_t mask, nk_i64_t diagonal_offset, nk_size_t window,
    nk_size_t task_start, nk_size_t task_count, void *stream) {
    nk_unused_(kind);
    if (((nk_size_t)packed & 15) || (((nk_size_t)output | output_stride) & 3)) return nk_misaligned_k;
    if (key_value_head_count == 0 || head_count % key_value_head_count != 0) return nk_unexpected_dimensions_k;
    if (task_count == 0 || depth == 0) return nk_success_k;
    nk_attention_arguments_t arguments = nk_attention_arguments_init_(
        queries, packed, output, head_count, key_value_head_count, depth, query_offsets, query_stride, output_stride,
        scale, score_scale, output_scale, mask, diagonal_offset, window, task_start, task_count);
    if (depth > nk_attention_wide_depth_blackwell_k)
        return nk_launch_resident_(fallback_kernel, nk_attention_threads_k, 0, 0, NUMKONG_SIZE_MAX, &arguments, stream);
    nk_attention_width_t const width = depth <= nk_attention_narrow_depth_blackwell_k ? nk_attention_width_narrow_k
                                                                                      : nk_attention_width_wide_k;
    // Blocks take one tensor-memory column per depth of their width, and a multiprocessor has 512.
    nk_size_t const columns = width == nk_attention_width_narrow_k ? nk_attention_narrow_depth_blackwell_k
                                                                   : nk_attention_wide_depth_blackwell_k;
    int multiprocessors = 0;
    nk_status_t const status = nk_device_attribute_(cudaDevAttrMultiProcessorCount, &multiprocessors);
    if (status != nk_success_k) return status;
    return nk_launch_resident_(width == nk_attention_width_narrow_k ? narrow_kernel : wide_kernel,
                               nk_attention_threads_k, nk_attention_shared_bytes_blackwell_(depth),
                               nk_attention_shared_bytes_blackwell_(columns),
                               (nk_size_t)multiprocessors * (512 / columns), &arguments, stream);
}

#pragma endregion Launch

#pragma region E4M3

nk_define_device_attention_pack_size_(e4m3, blackwell, 1)
nk_define_device_attention_packed_shape_(e4m3, blackwell)
nk_define_device_attention_pack_(e4m3, blackwell, e4m3, nk_attention_kind_bytes_k)
nk_define_device_attention_packed_(e4m3, blackwell, blackwell, nk_attention_launch_blackwell_, e4m3,
                                   nk_attention_kind_bytes_k, nk_cross_epilogue_f32_k, nk_attention_mma_e4m3_blackwell_,
                                   nk_attention_mma_e4m3_blackwell_, nk_attention_weights_e4m3_ada_, 1.0f, 1.0f,
                                   nk_e4m3_k)

#pragma endregion E4M3

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_BLACKWELL
#endif // NUMKONG_ATTENTION_BLACKWELL_CUH
