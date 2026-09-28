/**
 *  @file include/numkong/attention/hopper.cuh
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Ragged attention for NVIDIA Hopper, compute capability 9.0.
 *
 *  @sa include/numkong/attention.h
 *  @sa include/numkong/dots/hopper.cuh
 *
 *  Warpgroup `wgmma.mma_async` over shared-memory descriptors, which only the "90a" code carries,
 *  so no other generation compiles these kernels. One warpgroup owns each work item of up to 64
 *  rows, the Ampere tile's rows split over a 64-position panel: S is @c m64n64 steps on Q and K
 *  both in shared memory, and P · V takes P from the registers the online softmax packed it into,
 *  whose fragments match the Ampere tile's per warp. BF16 reads V as it is packed, position-major,
 *  through the transposed-B form; E4M3 and I8 read the pack's transposed, permuted V, E4M3 with
 *  weights e4m3(256 · p) as the Blackwell RTX kernels form them, I8 with U8 weights. Each panel's
 *  Float8 or integer P · V starts from zero and joins the running output in F32. Depths above 256
 *  fall back to the @c cuda kernel, whose pack layout and work scheduler Hopper shares.
 */
#ifndef NUMKONG_ATTENTION_HOPPER_CUH
#define NUMKONG_ATTENTION_HOPPER_CUH

#if NUMKONG_TARGET_HOPPER

#include "numkong/attention/ampere.cuh" // `nk_attention_weights_bf16_ampere_`, `nk_attention_wide_depth_ampere_k`
#include "numkong/attention/ada.cuh"    // `nk_attention_weights_e4m3_ada_`
#include "numkong/dots/hopper.cuh"      // `nk_wgmma_descriptor_hopper_`, `nk_wgmma_fence_hopper_`

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Configuration

enum {
    nk_attention_swizzle_rows_hopper_k = 64,
    nk_attention_swizzle_bytes_hopper_k = 64,
    nk_attention_block_bytes_hopper_k = nk_attention_swizzle_rows_hopper_k * nk_attention_swizzle_bytes_hopper_k,
};

/** One 32-byte depth step of S for the warpgroup's 64 rows against 64 positions, Q and K both
 *  from shared memory, adding to @p scores or, when @p accumulate is 0, replacing them. */
typedef void (*nk_attention_scores_hopper_t)(nk_fui32_t scores[32], nk_u64_t query_descriptor, nk_u64_t key_descriptor,
                                             int accumulate);

/** One step of P · V into 64 output columns, P as A fragments in @p weights and V from shared
 *  memory, adding to @p output or, when @p accumulate is 0, replacing it. */
typedef void (*nk_attention_values_hopper_t)(nk_fui32_t output[32], nk_u32_t const weights[4],
                                             nk_u64_t value_descriptor, int accumulate);

#pragma endregion Configuration

#pragma region Instructions

/** The operands of 32 F32 accumulators. */
#define nk_wgmma_n64_f32_operands_hopper_(d)                                                                        \
    "+f"(d[0].f), "+f"(d[1].f), "+f"(d[2].f), "+f"(d[3].f), "+f"(d[4].f), "+f"(d[5].f), "+f"(d[6].f), "+f"(d[7].f), \
        "+f"(d[8].f), "+f"(d[9].f), "+f"(d[10].f), "+f"(d[11].f), "+f"(d[12].f), "+f"(d[13].f), "+f"(d[14].f),      \
        "+f"(d[15].f), "+f"(d[16].f), "+f"(d[17].f), "+f"(d[18].f), "+f"(d[19].f), "+f"(d[20].f), "+f"(d[21].f),    \
        "+f"(d[22].f), "+f"(d[23].f), "+f"(d[24].f), "+f"(d[25].f), "+f"(d[26].f), "+f"(d[27].f), "+f"(d[28].f),    \
        "+f"(d[29].f), "+f"(d[30].f), "+f"(d[31].f)

/** The operands of 32 integer accumulators. */
#define nk_wgmma_n64_s32_operands_hopper_(d)                                                                        \
    "+r"(d[0].u), "+r"(d[1].u), "+r"(d[2].u), "+r"(d[3].u), "+r"(d[4].u), "+r"(d[5].u), "+r"(d[6].u), "+r"(d[7].u), \
        "+r"(d[8].u), "+r"(d[9].u), "+r"(d[10].u), "+r"(d[11].u), "+r"(d[12].u), "+r"(d[13].u), "+r"(d[14].u),      \
        "+r"(d[15].u), "+r"(d[16].u), "+r"(d[17].u), "+r"(d[18].u), "+r"(d[19].u), "+r"(d[20].u), "+r"(d[21].u),    \
        "+r"(d[22].u), "+r"(d[23].u), "+r"(d[24].u), "+r"(d[25].u), "+r"(d[26].u), "+r"(d[27].u), "+r"(d[28].u),    \
        "+r"(d[29].u), "+r"(d[30].u), "+r"(d[31].u)

#if defined(__CUDA_ARCH_SPECIFIC__) && __CUDA_ARCH_SPECIFIC__ == 900

NUMKONG_DEVICE void nk_attention_scores_bf16_hopper_(nk_fui32_t scores[32], nk_u64_t query_descriptor,
                                                     nk_u64_t key_descriptor, int accumulate) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"                                        //
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "                             //
                 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, " //
                 "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}"             //
                 ", %32, %33, p, 1, 1, 0, 0;\n}\n"
                 : nk_wgmma_n64_f32_operands_hopper_(scores)
                 : "l"(query_descriptor), "l"(key_descriptor), "r"(accumulate)
                 : "memory");
}

NUMKONG_DEVICE void nk_attention_scores_e4m3_hopper_(nk_fui32_t scores[32], nk_u64_t query_descriptor,
                                                     nk_u64_t key_descriptor, int accumulate) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"                                        //
                 "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "                             //
                 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, " //
                 "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}"             //
                 ", %32, %33, p, 1, 1;\n}\n"
                 : nk_wgmma_n64_f32_operands_hopper_(scores)
                 : "l"(query_descriptor), "l"(key_descriptor), "r"(accumulate)
                 : "memory");
}

NUMKONG_DEVICE void nk_attention_scores_i8_hopper_(nk_fui32_t scores[32], nk_u64_t query_descriptor,
                                                   nk_u64_t key_descriptor, int accumulate) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"                                        //
                 "wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.s8 "                                 //
                 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, " //
                 "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}"             //
                 ", %32, %33, p;\n}\n"
                 : nk_wgmma_n64_s32_operands_hopper_(scores)
                 : "l"(query_descriptor), "l"(key_descriptor), "r"(accumulate)
                 : "memory");
}

/*  V is position-major here, N-major for the MMA, which only the 16-bit forms take transposed. */
NUMKONG_DEVICE void nk_attention_values_bf16_hopper_(nk_fui32_t output[32], nk_u32_t const weights[4],
                                                     nk_u64_t value_descriptor, int accumulate) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"                                        //
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "                             //
                 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, " //
                 "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}"             //
                 ", {%32, %33, %34, %35}, %36, p, 1, 1, 1;\n}\n"
                 : nk_wgmma_n64_f32_operands_hopper_(output)
                 : "r"(weights[0]), "r"(weights[1]), "r"(weights[2]), "r"(weights[3]), "l"(value_descriptor),
                   "r"(accumulate)
                 : "memory");
}

NUMKONG_DEVICE void nk_attention_values_e4m3_hopper_(nk_fui32_t output[32], nk_u32_t const weights[4],
                                                     nk_u64_t value_descriptor, int accumulate) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"                                        //
                 "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "                             //
                 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, " //
                 "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}"             //
                 ", {%32, %33, %34, %35}, %36, p, 1, 1;\n}\n"
                 : nk_wgmma_n64_f32_operands_hopper_(output)
                 : "r"(weights[0]), "r"(weights[1]), "r"(weights[2]), "r"(weights[3]), "l"(value_descriptor),
                   "r"(accumulate)
                 : "memory");
}

NUMKONG_DEVICE void nk_attention_values_u8i8_hopper_(nk_fui32_t output[32], nk_u32_t const weights[4],
                                                     nk_u64_t value_descriptor, int accumulate) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"                                        //
                 "wgmma.mma_async.sync.aligned.m64n64k32.s32.u8.s8 "                                 //
                 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, " //
                 "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}"             //
                 ", {%32, %33, %34, %35}, %36, p;\n}\n"
                 : nk_wgmma_n64_s32_operands_hopper_(output)
                 : "r"(weights[0]), "r"(weights[1]), "r"(weights[2]), "r"(weights[3]), "l"(value_descriptor),
                   "r"(accumulate)
                 : "memory");
}

#else

NUMKONG_DEVICE void nk_attention_scores_bf16_hopper_(nk_fui32_t scores[32], nk_u64_t query_descriptor,
                                                     nk_u64_t key_descriptor, int accumulate) {
    __trap();
}
NUMKONG_DEVICE void nk_attention_scores_e4m3_hopper_(nk_fui32_t scores[32], nk_u64_t query_descriptor,
                                                     nk_u64_t key_descriptor, int accumulate) {
    __trap();
}
NUMKONG_DEVICE void nk_attention_scores_i8_hopper_(nk_fui32_t scores[32], nk_u64_t query_descriptor,
                                                   nk_u64_t key_descriptor, int accumulate) {
    __trap();
}
NUMKONG_DEVICE void nk_attention_values_bf16_hopper_(nk_fui32_t output[32], nk_u32_t const weights[4],
                                                     nk_u64_t value_descriptor, int accumulate) {
    __trap();
}
NUMKONG_DEVICE void nk_attention_values_e4m3_hopper_(nk_fui32_t output[32], nk_u32_t const weights[4],
                                                     nk_u64_t value_descriptor, int accumulate) {
    __trap();
}
NUMKONG_DEVICE void nk_attention_values_u8i8_hopper_(nk_fui32_t output[32], nk_u32_t const weights[4],
                                                     nk_u64_t value_descriptor, int accumulate) {
    __trap();
}

#endif // defined(__CUDA_ARCH_SPECIFIC__) && __CUDA_ARCH_SPECIFIC__ == 900

#undef nk_wgmma_n64_f32_operands_hopper_
#undef nk_wgmma_n64_s32_operands_hopper_

#pragma endregion Instructions

#pragma region Fragments

/**
 *  @brief The descriptor of a position-major V block, 64-byte swizzled, read transposed: 8
 *      positions of 32 BF16 depth columns per 512-byte atom, the next 8 positions 512 bytes on, and
 *      the next 32 columns @c nk_attention_block_bytes_hopper_k bytes on.
 *
 *  For an N-major operand the leading offset, bits 16-29, steps between 64-byte column groups and
 *  the stride offset, bits 32-45, between 8-row groups of K.
 */
NUMKONG_DEVICE nk_u64_t nk_attention_transposed_descriptor_hopper_(nk_u32_t shared_address) {
    nk_u64_t const start = (shared_address & 0x3FFFFu) >> 4, leading = nk_attention_block_bytes_hopper_k >> 4;
    nk_u64_t const stride = 512 >> 4, swizzle = 2;
    return start | (leading << 16) | (stride << 32) | (swizzle << 62);
}

#pragma endregion Fragments

#pragma region Tile

/** Issues 64 rows of a position-major plane from @p first_position into 64-byte swizzled blocks,
 *  depth bytes 64 · b to 64 · b + 63 of every row in block @c b. */
NUMKONG_DEVICE void nk_attention_stage_rows_hopper_(unsigned char *shared, unsigned char const *plane,
                                                    nk_size_t first_position, unsigned row_bytes) {
    unsigned const chunks_per_row = row_bytes >> 4, chunks = nk_attention_swizzle_rows_hopper_k * chunks_per_row;
    for (unsigned chunk = threadIdx.x; chunk < chunks; chunk += nk_attention_threads_k) {
        unsigned const row = chunk / chunks_per_row, column = chunk - row * chunks_per_row;
        unsigned const offset = (column >> 2) * nk_attention_block_bytes_hopper_k +
                                row * nk_attention_swizzle_bytes_hopper_k + (((column & 3) ^ ((row >> 1) & 3)) << 4);
        nk_copy_b128_async_ampere_(nk_shared_address_ampere_(shared + offset),
                                   plane + (first_position + row) * row_bytes + (column << 4), 16);
    }
}

/** Issues 64 slots from @p first_position of every depth row of a transposed V plane, one
 *  64-byte swizzled row per depth row. */
NUMKONG_DEVICE void nk_attention_stage_columns_hopper_(unsigned char *shared, unsigned char const *plane,
                                                       nk_size_t positions_padded, nk_size_t first_position,
                                                       unsigned depth_rows) {
    unsigned const chunks = depth_rows * 4;
    for (unsigned chunk = threadIdx.x; chunk < chunks; chunk += nk_attention_threads_k) {
        unsigned const row = chunk >> 2, column = chunk & 3;
        unsigned const offset = row * nk_attention_swizzle_bytes_hopper_k + ((column ^ ((row >> 1) & 3)) << 4);
        nk_copy_b128_async_ampere_(nk_shared_address_ampere_(shared + offset),
                                   plane + row * positions_padded + first_position + (column << 4), 16);
    }
}

/**
 *  @brief One work item on one warpgroup: scores, online softmax and P · V over every panel its
 *      rows see, then the output.
 *  @param[in] dtype The input dtype, which the MMAs take as it is: @c nk_bf16_k, @c nk_e4m3_k or
 *      @c nk_i8_k.
 *  @param[in] width Two or four 64-column output blocks in registers, see @c nk_attention_width_t.
 *  @param[in] epilogue F32 scores, or exact I32 ones converted per panel.
 *  @param[in] scores One depth step of S, see @c nk_attention_scores_hopper_t.
 *  @param[in] values_mma One step of P · V, see @c nk_attention_values_hopper_t.
 *  @param[in] weights P from probabilities, see @c nk_attention_weights_ampere_t.
 *
 *  Q, K and a BF16 V sit in 64-byte swizzled blocks of 64 rows, one block per 64 depth bytes; a
 *  transposed V sits as one swizzled 64-byte row per depth row. Warp @c w holds rows 16w to 16w +
 *  15, with scores and output in the Ampere tile's order: register 4t + 2h + e of a thread is row
 *  16w + lane / 4 + 8h at column 8t + 2 · (lane mod 4) + e.
 */
NUMKONG_DEVICE void nk_attention_block_hopper_(nk_dtype_t dtype, nk_attention_width_t width,
                                               nk_cross_epilogue_t epilogue, nk_attention_scores_hopper_t scores,
                                               nk_attention_values_hopper_t values_mma,
                                               nk_attention_weights_ampere_t weights,
                                               nk_attention_arguments_t const *arguments,
                                               nk_attention_work_t const *work, unsigned char *shared,
                                               unsigned (*unions)[2]) {

    nk_f32_t const negative_infinity = nk_attention_negative_infinity_();
    unsigned const lane = threadIdx.x & 31, warp = threadIdx.x >> 5, group = lane >> 2, quad = lane & 3;
    unsigned const element_bytes = dtype == nk_bf16_k ? 2 : 1;
    unsigned const max_blocks = width == nk_attention_width_128_k ? nk_attention_narrow_depth_ampere_k / 64
                                                                  : nk_attention_wide_depth_ampere_k / 64;

    nk_size_t const depth = arguments->depth;
    unsigned const row_bytes = (unsigned)nk_size_round_up_to_multiple_(depth * element_bytes,
                                                                       nk_attention_step_bytes_k);
    unsigned const depth_steps = row_bytes / nk_attention_step_bytes_k;
    unsigned const depth_padded = row_bytes / element_bytes,
                   depth_blocks = (unsigned)nk_size_divide_round_up_(depth_padded, 64);
    unsigned char *keys_shared = shared + arguments->key_offset[0];
    unsigned char *values_shared = shared + arguments->value_offset[0];
    unsigned char *queries_shared = shared + arguments->query_offset[0];
    nk_u32_t const queries_address = nk_shared_address_ampere_(queries_shared);
    nk_u32_t const keys_address = nk_shared_address_ampere_(keys_shared);
    nk_u32_t const values_address = nk_shared_address_ampere_(values_shared);

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
        nk_attention_stage_rows_hopper_(keys_shared, keys_plane, (nk_size_t)panel_first * panel, row_bytes);
    nk_commit_async_ampere_();

    // Q rows need not be 16-byte aligned, so they are stored element by element.
    for (unsigned local = warp; local < nk_attention_block_rows_k; local += 4) {
        int const valid = local < work->row_count;
        unsigned char const *source = arguments->queries;
        if (valid) {
            nk_size_t const row = work->row_first + local;
            nk_size_t const query = row / work->heads_selected;
            nk_size_t const head = work->head_first + row % work->heads_selected;
            source += (work->query_first + query) * arguments->query_stride + head * depth * element_bytes;
        }
        for (unsigned element = lane; element < depth_padded; element += 32) {
            unsigned const byte = element * element_bytes, column = (byte >> 4) & 3;
            unsigned char *destination = queries_shared + (byte >> 6) * nk_attention_block_bytes_hopper_k +
                                         local * nk_attention_swizzle_bytes_hopper_k +
                                         ((column ^ ((local >> 1) & 3)) << 4) + (byte & 15);
            if (dtype == nk_bf16_k)
                *(unsigned short *)destination = valid && element < depth ? ((unsigned short const *)source)[element]
                                                                          : (unsigned short)0;
            else *destination = valid && element < depth ? source[element] : (unsigned char)0;
        }
    }

    nk_f32_t row_max[2] = {negative_infinity, negative_infinity}, row_sum[2] = {0, 0};
    nk_fui32_t output[nk_attention_wide_depth_ampere_k / 64 * 32];
#pragma unroll
    for (unsigned index = 0; index < max_blocks * 32; ++index) output[index].u = 0;
    nk_f32_t const scale2 = arguments->scale2 * arguments->score_scale;

    for (unsigned panel_index = panel_first; panel_index < panel_end; ++panel_index) {
        nk_wait_async_ampere_(0);
        nk_fence_proxy_async_hopper_();
        // K has landed, Q too on the first panel, and every warp is done with the V this refills.
        __syncthreads();
        nk_size_t const panel_position = (nk_size_t)panel_index * panel;
        if (dtype == nk_bf16_k) nk_attention_stage_rows_hopper_(values_shared, values_plane, panel_position, row_bytes);
        else
            nk_attention_stage_columns_hopper_(values_shared, values_plane, positions_padded, panel_position,
                                               depth_padded);
        nk_commit_async_ampere_();

        nk_fui32_t tile_scores[32];
#pragma unroll
        for (unsigned index = 0; index < 32; ++index) tile_scores[index].u = 0;
        nk_wgmma_fence_operands_hopper_(tile_scores, 32, epilogue);
        nk_wgmma_fence_hopper_();
#pragma unroll
        for (unsigned step = 0; step < max_blocks * 4; ++step) {
            if (step >= depth_steps) continue;
            nk_u32_t const offset = (step >> 1) * nk_attention_block_bytes_hopper_k + (step & 1) * 32;
            scores(tile_scores, nk_wgmma_descriptor_hopper_(queries_address + offset),
                   nk_wgmma_descriptor_hopper_(keys_address + offset), step != 0);
        }
        nk_wgmma_commit_hopper_();
        nk_wgmma_wait_hopper_();
        nk_wgmma_fence_operands_hopper_(tile_scores, 32, epilogue);

#pragma unroll
        for (unsigned index = 0; index < 32; ++index)
            tile_scores[index].f = (epilogue == nk_cross_epilogue_i32_to_f32_k ? (nk_f32_t)tile_scores[index].i
                                                                               : tile_scores[index].f) *
                                   scale2;
        unsigned const chunk_begin = (unsigned)panel_position, chunk_end = chunk_begin + panel;
        // Padded tails and causal edges fall outside some row's keys; other panels skip this.
        if (!(common_begin <= chunk_begin && chunk_end <= common_end)) {
#pragma unroll
            for (unsigned index = 0; index < 32; ++index) {
                unsigned const position = chunk_begin + (index >> 2) * 8 + quad * 2 + (index & 1);
                unsigned const half = (index >> 1) & 1;
                if (position < key_begin[half] || position >= key_end[half]) tile_scores[index].f = negative_infinity;
            }
        }
        nk_f32_t subtrahend[2], correction[2];
#pragma unroll
        for (unsigned half = 0; half < 2; ++half) {
            nk_f32_t chunk_max = negative_infinity;
#pragma unroll
            for (unsigned tile = 0; tile < 8; ++tile)
                chunk_max = fmaxf(chunk_max,
                                  fmaxf(tile_scores[tile * 4 + half * 2].f, tile_scores[tile * 4 + half * 2 + 1].f));
            chunk_max = fmaxf(chunk_max, __shfl_xor_sync(0xFFFFFFFFu, chunk_max, 1));
            chunk_max = fmaxf(chunk_max, __shfl_xor_sync(0xFFFFFFFFu, chunk_max, 2));
            nk_f32_t const new_max = fmaxf(row_max[half], chunk_max);
            // With every key so far masked, subtracting 0 keeps exp2(-∞ - max) from turning NaN.
            subtrahend[half] = new_max == negative_infinity ? 0.0f : new_max;
            correction[half] = nk_f32_exp2_(row_max[half] - subtrahend[half]);
            row_max[half] = new_max;
            row_sum[half] *= correction[half];
        }
#pragma unroll
        for (unsigned index = 0; index < 32; ++index)
            tile_scores[index].f = nk_f32_exp2_(tile_scores[index].f - subtrahend[(index >> 1) & 1]);
        nk_u32_t probabilities[4][4];
#pragma unroll
        for (unsigned position_group = 0; position_group < 4; ++position_group)
#pragma unroll
            for (unsigned half = 0; half < 2; ++half) {
                nk_f32_t const four[4] = {tile_scores[position_group * 8 + half * 2].f,
                                          tile_scores[position_group * 8 + half * 2 + 1].f,
                                          tile_scores[position_group * 8 + 4 + half * 2].f,
                                          tile_scores[position_group * 8 + 4 + half * 2 + 1].f};
                nk_u32_t packed[2];
                weights(four, packed, &row_sum[half]);
                if (dtype != nk_bf16_k) probabilities[position_group >> 1][(position_group & 1) * 2 + half] = packed[0];
                else
                    probabilities[position_group][half] = packed[0],
                    probabilities[position_group][half + 2] = packed[1];
            }
        if (dtype == nk_bf16_k)
#pragma unroll
            for (unsigned index = 0; index < max_blocks * 32; ++index) output[index].f *= correction[(index >> 1) & 1];

        nk_wait_async_ampere_(0);
        nk_fence_proxy_async_hopper_();
        // V has landed, and every warp is done with the K this refills.
        __syncthreads();
        if (panel_index + 1 < panel_end)
            nk_attention_stage_rows_hopper_(keys_shared, keys_plane, panel_position + panel, row_bytes);
        nk_commit_async_ampere_();

#pragma unroll
        for (unsigned position_group = 0; position_group < (dtype == nk_bf16_k ? 4 : 2); ++position_group)
            nk_wgmma_fence_operands_hopper_((nk_fui32_t *)probabilities[position_group], 4, nk_cross_epilogue_i32_k);
        if (dtype == nk_bf16_k) {
            nk_wgmma_fence_operands_hopper_(output, max_blocks * 32, nk_cross_epilogue_f32_k);
            nk_wgmma_fence_hopper_();
#pragma unroll
            for (unsigned block = 0; block < max_blocks; ++block) {
                if (block >= depth_blocks) continue;
#pragma unroll
                for (unsigned position_group = 0; position_group < 4; ++position_group)
                    values_mma(output + block * 32, probabilities[position_group],
                               nk_attention_transposed_descriptor_hopper_(
                                   values_address + block * 2 * nk_attention_block_bytes_hopper_k +
                                   position_group * 16 * nk_attention_swizzle_bytes_hopper_k),
                               1);
            }
            nk_wgmma_commit_hopper_();
            nk_wgmma_wait_hopper_();
            nk_wgmma_fence_operands_hopper_(output, max_blocks * 32, nk_cross_epilogue_f32_k);
            continue;
        }
#pragma unroll
        for (unsigned block = 0; block < max_blocks; ++block) {
            if (block >= depth_blocks) continue;
            nk_fui32_t sums[32];
#pragma unroll
            for (unsigned index = 0; index < 32; ++index) sums[index].u = 0;
            nk_wgmma_fence_operands_hopper_(sums, 32, epilogue);
            nk_wgmma_fence_hopper_();
#pragma unroll
            for (unsigned step = 0; step < 2; ++step)
                values_mma(
                    sums, probabilities[step],
                    nk_wgmma_descriptor_hopper_(values_address + block * nk_attention_block_bytes_hopper_k + step * 32),
                    step);
            nk_wgmma_commit_hopper_();
            nk_wgmma_wait_hopper_();
            nk_wgmma_fence_operands_hopper_(sums, 32, epilogue);
#pragma unroll
            for (unsigned index = 0; index < 32; ++index)
                output[block * 32 + index].f = fmaf(
                    output[block * 32 + index].f, correction[(index >> 1) & 1],
                    epilogue == nk_cross_epilogue_i32_to_f32_k ? (nk_f32_t)sums[index].i : sums[index].f);
        }
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
        for (unsigned tile = 0; tile < max_blocks * 8; ++tile)
#pragma unroll
            for (unsigned element = 0; element < 2; ++element) {
                unsigned const column = tile * 8 + quad * 2 + element;
                if (column < depth) destination[column] = output[tile * 4 + half * 2 + element].f * inverse;
            }
    }
}

/**
 *  @brief Every work item of a launch, walked with a stride of the grid, on a dynamic shared
 *      buffer aligned up to the 512-byte boundary the 64-byte swizzle repeats on.
 *  @sa nk_attention_block_hopper_ for the parameters.
 */
NUMKONG_DEVICE void nk_attention_tile_hopper_(nk_dtype_t dtype, nk_attention_width_t width,
                                              nk_cross_epilogue_t epilogue, nk_attention_scores_hopper_t scores,
                                              nk_attention_values_hopper_t values_mma,
                                              nk_attention_weights_ampere_t weights,
                                              nk_attention_arguments_t const *arguments) {
    extern __shared__ __align__(128) unsigned char nk_attention_shared_hopper_[];
    __shared__ nk_u64_t prefix[nk_attention_threads_k + 1];
    __shared__ nk_u64_t warp_totals[nk_attention_threads_k / 32];
    __shared__ unsigned unions[nk_attention_threads_k / 32][2];
    unsigned char *shared = nk_attention_shared_hopper_ +
                            ((512 - (nk_shared_address_ampere_(nk_attention_shared_hopper_) & 511)) & 511);
    nk_attention_schedule_t schedule;
    if (!nk_attention_schedule_start_(arguments, &schedule, prefix, warp_totals)) return;
    nk_attention_work_t work;
    for (nk_size_t item = blockIdx.x; nk_attention_schedule_next_(&schedule, prefix, warp_totals, item, &work);
         item += gridDim.x)
        nk_attention_block_hopper_(dtype, width, epilogue, scores, values_mma, weights, arguments, &work, shared,
                                   unions);
}

#pragma endregion Tile

#pragma region Launch

/** Places Q, K and V in dynamic shared memory, returning the bytes a block needs. */
NUMKONG_INLINE nk_size_t nk_attention_shared_layout_hopper_(nk_dtype_t dtype, nk_size_t depth,
                                                            nk_attention_arguments_t *arguments) {
    nk_size_t const element_bytes = dtype == nk_bf16_k ? 2 : 1;
    nk_size_t const row_bytes = nk_size_round_up_to_multiple_(depth * element_bytes, nk_attention_step_bytes_k);
    nk_size_t const row_blocks = nk_size_divide_round_up_(row_bytes, nk_attention_swizzle_bytes_hopper_k);
    nk_size_t const depth_blocks = nk_size_divide_round_up_(row_bytes / element_bytes, 64);
    // Every 64-column output block reads whole blocks of V, past the depth the pack holds.
    nk_size_t const rows_block_bytes = row_blocks * nk_attention_block_bytes_hopper_k;
    nk_size_t const values_bytes = depth_blocks * element_bytes * nk_attention_block_bytes_hopper_k;
    arguments->query_offset[0] = 0, arguments->key_offset[0] = (nk_u32_t)rows_block_bytes;
    arguments->value_offset[0] = (nk_u32_t)(2 * rows_block_bytes);
    return 2 * rows_block_bytes + values_bytes + 512;
}

/** Dynamic shared memory of a block at the deepest head @p width takes, which each of its kernels
 *  sets as its limit. */
NUMKONG_INLINE nk_size_t nk_attention_shared_ceiling_hopper_(nk_dtype_t dtype, nk_attention_width_t width) {
    nk_attention_arguments_t deepest;
    nk_size_t const depth = width == nk_attention_width_128_k ? nk_attention_narrow_depth_ampere_k
                                                              : nk_attention_wide_depth_ampere_k;
    return nk_attention_shared_layout_hopper_(dtype, depth, &deepest);
}

/**
 *  @brief Validates the contract and launches the kernel for the depth's width with as many blocks
 *      as stay resident, Q, K and V each taking 64-byte swizzled blocks past a 512-byte alignment.
 *  @param[in] score_scale Undoes the power of two that converting Q and K puts on scores, or 1.
 *  @param[in] output_scale Undoes the power of two that converting V puts on the output, or 1.
 */
NUMKONG_INLINE nk_status_t nk_attention_launch_hopper_(
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
    if (depth > nk_attention_wide_depth_ampere_k)
        return nk_launch_resident_(fallback_kernel, nk_attention_threads_k, 0, 0, NUMKONG_SIZE_MAX, &arguments, stream);
    nk_attention_width_t const width = depth <= nk_attention_narrow_depth_ampere_k ? nk_attention_width_128_k
                                                                                   : nk_attention_width_256_k;
    nk_size_t const shared_bytes = nk_attention_shared_layout_hopper_(dtype, depth, &arguments);
    return nk_launch_resident_(width == nk_attention_width_128_k ? narrow_kernel : wide_kernel, nk_attention_threads_k,
                               shared_bytes, nk_attention_shared_ceiling_hopper_(dtype, width), NUMKONG_SIZE_MAX,
                               &arguments, stream);
}

#pragma endregion Launch

#pragma region Instantiations

nk_define_device_attention_pack_size_(bf16, hopper, 2)
nk_define_device_attention_packed_shape_(bf16, hopper)
nk_define_device_attention_pack_(bf16, hopper, bf16)
nk_define_device_attention_packed_(bf16, hopper, hopper, nk_attention_launch_hopper_, bf16, nk_cross_epilogue_f32_k,
                                   nk_attention_scores_bf16_hopper_, nk_attention_values_bf16_hopper_,
                                   nk_attention_weights_bf16_ampere_, 1.0f, 1.0f)

nk_define_device_attention_pack_size_(e4m3, hopper, 1)
nk_define_device_attention_packed_shape_(e4m3, hopper)
nk_define_device_attention_pack_(e4m3, hopper, e4m3)
nk_define_device_attention_packed_(e4m3, hopper, hopper, nk_attention_launch_hopper_, e4m3, nk_cross_epilogue_f32_k,
                                   nk_attention_scores_e4m3_hopper_, nk_attention_values_e4m3_hopper_,
                                   nk_attention_weights_e4m3_ada_, 1.0f, 1.0f)

nk_define_device_attention_pack_size_(i8, hopper, 1)
nk_define_device_attention_packed_shape_(i8, hopper)
nk_define_device_attention_pack_(i8, hopper, i8)
nk_define_device_attention_packed_(i8, hopper, hopper, nk_attention_launch_hopper_, i8, nk_cross_epilogue_i32_to_f32_k,
                                   nk_attention_scores_i8_hopper_, nk_attention_values_u8i8_hopper_,
                                   nk_attention_weights_u8_ampere_, 1.0f, 1.0f)

#pragma endregion Instantiations

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_HOPPER
#endif // NUMKONG_ATTENTION_HOPPER_CUH
