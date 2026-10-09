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

#if NUMKONG_ARCH_CUDA_
#if NUMKONG_TARGET_HOPPER

#include "numkong/attention/ampere.cuh" // `nk_attention_weights_bf16_ampere_`, `nk_attention_wide_depth_ampere_k`
#include "numkong/attention/ada.cuh"    // `nk_attention_weights_e4m3_ada_`
#include "numkong/dots/hopper.cuh"      // `nk_smem_descriptor_hopper_`, `nk_wgmma_fence_hopper_`

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Configuration

enum {
    nk_attention_swizzle_rows_hopper_k = 64,
    nk_attention_swizzle_bytes_hopper_k = 64,
    nk_attention_block_bytes_hopper_k = nk_attention_swizzle_rows_hopper_k * nk_attention_swizzle_bytes_hopper_k,
};

#pragma endregion Configuration

#pragma region Instructions

NUMKONG_DEVICE void nk_attention_scores_bf16_hopper_(nk_fui32_t scores[32], nk_u64_t query_descriptor,
                                                     nk_u64_t key_descriptor, int accumulate) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"                                        //
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "                             //
                 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, " //
                 "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}"             //
                 ", %32, %33, p, 1, 1, 0, 0;\n}\n"
                 : "+f"(scores[0].f), "+f"(scores[1].f), "+f"(scores[2].f), "+f"(scores[3].f), "+f"(scores[4].f),
                   "+f"(scores[5].f), "+f"(scores[6].f), "+f"(scores[7].f), "+f"(scores[8].f), "+f"(scores[9].f),
                   "+f"(scores[10].f), "+f"(scores[11].f), "+f"(scores[12].f), "+f"(scores[13].f), "+f"(scores[14].f),
                   "+f"(scores[15].f), "+f"(scores[16].f), "+f"(scores[17].f), "+f"(scores[18].f), "+f"(scores[19].f),
                   "+f"(scores[20].f), "+f"(scores[21].f), "+f"(scores[22].f), "+f"(scores[23].f), "+f"(scores[24].f),
                   "+f"(scores[25].f), "+f"(scores[26].f), "+f"(scores[27].f), "+f"(scores[28].f), "+f"(scores[29].f),
                   "+f"(scores[30].f), "+f"(scores[31].f)
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
                 : "+f"(scores[0].f), "+f"(scores[1].f), "+f"(scores[2].f), "+f"(scores[3].f), "+f"(scores[4].f),
                   "+f"(scores[5].f), "+f"(scores[6].f), "+f"(scores[7].f), "+f"(scores[8].f), "+f"(scores[9].f),
                   "+f"(scores[10].f), "+f"(scores[11].f), "+f"(scores[12].f), "+f"(scores[13].f), "+f"(scores[14].f),
                   "+f"(scores[15].f), "+f"(scores[16].f), "+f"(scores[17].f), "+f"(scores[18].f), "+f"(scores[19].f),
                   "+f"(scores[20].f), "+f"(scores[21].f), "+f"(scores[22].f), "+f"(scores[23].f), "+f"(scores[24].f),
                   "+f"(scores[25].f), "+f"(scores[26].f), "+f"(scores[27].f), "+f"(scores[28].f), "+f"(scores[29].f),
                   "+f"(scores[30].f), "+f"(scores[31].f)
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
                 : "+r"(scores[0].u), "+r"(scores[1].u), "+r"(scores[2].u), "+r"(scores[3].u), "+r"(scores[4].u),
                   "+r"(scores[5].u), "+r"(scores[6].u), "+r"(scores[7].u), "+r"(scores[8].u), "+r"(scores[9].u),
                   "+r"(scores[10].u), "+r"(scores[11].u), "+r"(scores[12].u), "+r"(scores[13].u), "+r"(scores[14].u),
                   "+r"(scores[15].u), "+r"(scores[16].u), "+r"(scores[17].u), "+r"(scores[18].u), "+r"(scores[19].u),
                   "+r"(scores[20].u), "+r"(scores[21].u), "+r"(scores[22].u), "+r"(scores[23].u), "+r"(scores[24].u),
                   "+r"(scores[25].u), "+r"(scores[26].u), "+r"(scores[27].u), "+r"(scores[28].u), "+r"(scores[29].u),
                   "+r"(scores[30].u), "+r"(scores[31].u)
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
                 : "+f"(output[0].f), "+f"(output[1].f), "+f"(output[2].f), "+f"(output[3].f), "+f"(output[4].f),
                   "+f"(output[5].f), "+f"(output[6].f), "+f"(output[7].f), "+f"(output[8].f), "+f"(output[9].f),
                   "+f"(output[10].f), "+f"(output[11].f), "+f"(output[12].f), "+f"(output[13].f), "+f"(output[14].f),
                   "+f"(output[15].f), "+f"(output[16].f), "+f"(output[17].f), "+f"(output[18].f), "+f"(output[19].f),
                   "+f"(output[20].f), "+f"(output[21].f), "+f"(output[22].f), "+f"(output[23].f), "+f"(output[24].f),
                   "+f"(output[25].f), "+f"(output[26].f), "+f"(output[27].f), "+f"(output[28].f), "+f"(output[29].f),
                   "+f"(output[30].f), "+f"(output[31].f)
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
                 : "+f"(output[0].f), "+f"(output[1].f), "+f"(output[2].f), "+f"(output[3].f), "+f"(output[4].f),
                   "+f"(output[5].f), "+f"(output[6].f), "+f"(output[7].f), "+f"(output[8].f), "+f"(output[9].f),
                   "+f"(output[10].f), "+f"(output[11].f), "+f"(output[12].f), "+f"(output[13].f), "+f"(output[14].f),
                   "+f"(output[15].f), "+f"(output[16].f), "+f"(output[17].f), "+f"(output[18].f), "+f"(output[19].f),
                   "+f"(output[20].f), "+f"(output[21].f), "+f"(output[22].f), "+f"(output[23].f), "+f"(output[24].f),
                   "+f"(output[25].f), "+f"(output[26].f), "+f"(output[27].f), "+f"(output[28].f), "+f"(output[29].f),
                   "+f"(output[30].f), "+f"(output[31].f)
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
                 : "+r"(output[0].u), "+r"(output[1].u), "+r"(output[2].u), "+r"(output[3].u), "+r"(output[4].u),
                   "+r"(output[5].u), "+r"(output[6].u), "+r"(output[7].u), "+r"(output[8].u), "+r"(output[9].u),
                   "+r"(output[10].u), "+r"(output[11].u), "+r"(output[12].u), "+r"(output[13].u), "+r"(output[14].u),
                   "+r"(output[15].u), "+r"(output[16].u), "+r"(output[17].u), "+r"(output[18].u), "+r"(output[19].u),
                   "+r"(output[20].u), "+r"(output[21].u), "+r"(output[22].u), "+r"(output[23].u), "+r"(output[24].u),
                   "+r"(output[25].u), "+r"(output[26].u), "+r"(output[27].u), "+r"(output[28].u), "+r"(output[29].u),
                   "+r"(output[30].u), "+r"(output[31].u)
                 : "r"(weights[0]), "r"(weights[1]), "r"(weights[2]), "r"(weights[3]), "l"(value_descriptor),
                   "r"(accumulate)
                 : "memory");
}

#pragma endregion Instructions

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

/** Where one warpgroup's work item sits: its output blocks, the shared buffers that stage them, and
 *  the packed planes they come from. */
typedef struct {
    unsigned max_blocks, row_bytes, depth_steps, depth_padded, depth_blocks;
    unsigned char *keys_shared, *values_shared, *queries_shared;
    nk_u32_t queries_address, keys_address, values_address;
    unsigned char const *keys_plane, *values_plane;
    nk_size_t length, positions_padded;
    nk_diagonal_band_t band;
    unsigned warp_rows, panel_first, panel_end;
    nk_i64_t warp_first;
} nk_attention_frame_hopper_t;

/**
 *  @brief Places one work item on one warpgroup: how many 64-column output blocks it holds per
 *      @p width, its shared buffers, and the panels its rows see.
 *  @param[in] mask Whether panels crossing the band's edges mask their scores, see
 *      @c nk_attention_mask_t; panels past the segment's keys always do.
 *  @param[in] element_bytes Bytes of one input element: 2 keeps V position-major, 1 transposes it.
 */
NUMKONG_DEVICE nk_attention_frame_hopper_t nk_attention_frame_hopper_(nk_attention_width_t width,
                                                                      nk_attention_mask_t mask, unsigned element_bytes,
                                                                      nk_attention_arguments_t const *arguments,
                                                                      nk_attention_work_t const *work,
                                                                      unsigned char *shared) {
    nk_attention_frame_hopper_t frame;
    unsigned const warp = threadIdx.x >> 5;
    frame.max_blocks = width == nk_attention_width_128_k ? nk_attention_narrow_depth_ampere_k / 64
                                                         : nk_attention_wide_depth_ampere_k / 64;
    nk_size_t const depth = arguments->depth;
    frame.row_bytes = (unsigned)nk_size_round_up_to_multiple_(depth * element_bytes, nk_attention_step_bytes_k);
    frame.depth_steps = frame.row_bytes / nk_attention_step_bytes_k;
    frame.depth_padded = frame.row_bytes / element_bytes;
    frame.depth_blocks = (unsigned)nk_size_divide_round_up_(frame.depth_padded, 64);
    frame.keys_shared = shared + arguments->key_offset[0];
    frame.values_shared = shared + arguments->value_offset[0];
    frame.queries_shared = shared + arguments->query_offset[0];
    frame.queries_address = nk_shared_address_ampere_(frame.queries_shared);
    frame.keys_address = nk_shared_address_ampere_(frame.keys_shared);
    frame.values_address = nk_shared_address_ampere_(frame.values_shared);

    nk_size_t plane_bytes;
    frame.keys_plane = nk_attention_keys_plane_simt_(arguments, work, frame.row_bytes, &frame.length, &plane_bytes);
    frame.values_plane = frame.keys_plane + arguments->key_value_head_count * plane_bytes;
    frame.positions_padded = plane_bytes / frame.row_bytes;

    // A thread holds rows `group` and `group + 8` of its warp's 16, which classify panels jointly.
    frame.band = nk_attention_kernel_band_simt_(mask, arguments);
    frame.warp_rows = work->row_count > warp * 16 ? min((unsigned)work->row_count - warp * 16, 16u) : 0;
    frame.warp_first = nk_attention_row_position_simt_(work, warp * 16, frame.length);
    nk_size_t block_begin, block_end;
    nk_attention_rows_keys_simt_(frame.band, nk_attention_row_position_simt_(work, 0, frame.length),
                                 nk_attention_row_position_simt_(work, work->row_count - 1, frame.length), frame.length,
                                 &block_begin, &block_end);
    frame.panel_first = (unsigned)(block_begin / nk_attention_panel_k);
    frame.panel_end = block_begin < block_end ? (unsigned)nk_size_divide_round_up_(block_end, nk_attention_panel_k)
                                              : frame.panel_first;
    return frame;
}

/** Stores the work item's Q rows of 2-byte elements into 64-byte swizzled blocks. Rows need not be
 *  16-byte aligned, so they are stored element by element. */
NUMKONG_DEVICE void nk_attention_stage_queries_b16_hopper_(nk_attention_frame_hopper_t const *frame,
                                                           nk_attention_arguments_t const *arguments,
                                                           nk_attention_work_t const *work) {
    unsigned const lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (unsigned local = warp; local < nk_attention_block_rows_k; local += 4) {
        int const valid = local < work->row_count;
        unsigned char const *source = valid ? nk_attention_query_row_simt_(arguments, work, local, 2) : NUMKONG_NULL;
        unsigned short const *source_words = (unsigned short const *)source;
        for (unsigned element = lane; element < frame->depth_padded; element += 32) {
            unsigned const byte = element * 2, column = (byte >> 4) & 3;
            unsigned char *destination = frame->queries_shared + (byte >> 6) * nk_attention_block_bytes_hopper_k +
                                         local * nk_attention_swizzle_bytes_hopper_k +
                                         ((column ^ ((local >> 1) & 3)) << 4) + (byte & 15);
            int const inside = valid && element < arguments->depth;
            *(unsigned short *)destination = inside ? source_words[element] : (unsigned short)0;
        }
    }
}

/** The 1-byte twin of @c nk_attention_stage_queries_b16_hopper_. */
NUMKONG_DEVICE void nk_attention_stage_queries_b8_hopper_(nk_attention_frame_hopper_t const *frame,
                                                          nk_attention_arguments_t const *arguments,
                                                          nk_attention_work_t const *work) {
    unsigned const lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (unsigned local = warp; local < nk_attention_block_rows_k; local += 4) {
        int const valid = local < work->row_count;
        unsigned char const *source = valid ? nk_attention_query_row_simt_(arguments, work, local, 1) : NUMKONG_NULL;
        for (unsigned element = lane; element < frame->depth_padded; element += 32) {
            unsigned const byte = element, column = (byte >> 4) & 3;
            unsigned char *destination = frame->queries_shared + (byte >> 6) * nk_attention_block_bytes_hopper_k +
                                         local * nk_attention_swizzle_bytes_hopper_k +
                                         ((column ^ ((local >> 1) & 3)) << 4) + (byte & 15);
            int const inside = valid && element < arguments->depth;
            *destination = inside ? source[element] : (unsigned char)0;
        }
    }
}

/**
 *  @brief Turns a panel's raw scores into the exponentials of its online softmax: scales them,
 *      masks the columns the band hides, folds the panel's maxima into @p row_max and rescales
 *      @p row_sum, with the factor the output needs in @p correction.
 *  @param[in] epilogue F32 scores, or exact I32 ones converted first.
 */
NUMKONG_DEVICE void nk_attention_softmax_hopper_(nk_cross_epilogue_t epilogue, nk_fui32_t tile_scores[32],
                                                 nk_attention_frame_hopper_t const *frame,
                                                 nk_attention_work_t const *work, nk_f32_t scale2,
                                                 nk_size_t panel_position, nk_f32_t row_max[2], nk_f32_t row_sum[2],
                                                 nk_f32_t correction[2]) {
    nk_f32_t const negative_infinity = nk_attention_negative_infinity_simt_();
    unsigned const lane = threadIdx.x & 31, warp = threadIdx.x >> 5, group = lane >> 2, quad = lane & 3;
#pragma unroll
    for (unsigned index = 0; index < 32; ++index)
        tile_scores[index].f = (epilogue == nk_cross_epilogue_i32_to_f32_k ? (nk_f32_t)tile_scores[index].i
                                                                           : tile_scores[index].f) *
                               scale2;
    // The warpgroup multiplies every panel, so a warp outside one masks all of it.
    if (nk_attention_tile_coverage_simt_(frame->band, frame->warp_first, frame->warp_rows, panel_position,
                                         nk_attention_panel_k, frame->length) != nk_diagonal_band_inside_k) {
        nk_u32_t visible[2][2] = {{0, 0}, {0, 0}};
#pragma unroll
        for (unsigned half = 0; half < 2; ++half) {
            unsigned const local = warp * 16 + group + half * 8;
            if (local >= work->row_count) continue;
            nk_i64_t const row = nk_attention_row_position_simt_(work, local, frame->length);
            visible[half][0] = nk_diagonal_band_row_mask_simt_(frame->band, row, panel_position, frame->length);
            visible[half][1] = nk_diagonal_band_row_mask_simt_(frame->band, row, panel_position + 32, frame->length);
        }
#pragma unroll
        for (unsigned index = 0; index < 32; ++index) {
            unsigned const offset = (index >> 2) * 8 + quad * 2 + (index & 1);
            if (!((visible[(index >> 1) & 1][offset >> 5] >> (offset & 31)) & 1))
                tile_scores[index].f = negative_infinity;
        }
    }
    nk_f32_t subtrahend[2];
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
        correction[half] = nk_f32_exp2_cuda_(row_max[half] - subtrahend[half]);
        row_max[half] = new_max;
        row_sum[half] *= correction[half];
    }
#pragma unroll
    for (unsigned index = 0; index < 32; ++index)
        tile_scores[index].f = nk_f32_exp2_cuda_(tile_scores[index].f - subtrahend[(index >> 1) & 1]);
}

/** Writes the rows' outputs and log-sum-exps, @p unit being what the dtype's weights add for a
 *  probability of one. */
NUMKONG_DEVICE void nk_attention_finish_hopper_(nk_attention_frame_hopper_t const *frame, nk_fui32_t output[],
                                                nk_f32_t row_max[2], nk_f32_t row_sum[2], nk_f32_t unit,
                                                nk_attention_arguments_t const *arguments,
                                                nk_attention_work_t const *work) {
    unsigned const lane = threadIdx.x & 31, warp = threadIdx.x >> 5, group = lane >> 2, quad = lane & 3;
    nk_size_t const depth = arguments->depth;
#pragma unroll
    for (unsigned half = 0; half < 2; ++half) {
        row_sum[half] += __shfl_xor_sync(0xFFFFFFFFu, row_sum[half], 1);
        row_sum[half] += __shfl_xor_sync(0xFFFFFFFFu, row_sum[half], 2);
    }

#pragma unroll
    for (unsigned half = 0; half < 2; ++half) {
        unsigned const local = warp * 16 + group + half * 8;
        if (local >= work->row_count) continue;
        nk_f32_t *destination = nk_attention_output_row_simt_(arguments, work, local);
        nk_f32_t const inverse = row_sum[half] > 0 ? arguments->output_scale / row_sum[half] : 0.0f;
#pragma unroll
        for (unsigned tile = 0; tile < frame->max_blocks * 8; ++tile)
#pragma unroll
            for (unsigned element = 0; element < 2; ++element) {
                unsigned const column = tile * 8 + quad * 2 + element;
                if (column < depth) destination[column] = output[tile * 4 + half * 2 + element].f * inverse;
            }
        nk_f32_t *const log_sum_exp_slot = quad == 0 ? nk_attention_log_sum_exp_slot_simt_(arguments, work, local)
                                                     : NUMKONG_NULL;
        if (log_sum_exp_slot) *log_sum_exp_slot = nk_attention_log_sum_exp_simt_(row_max[half], row_sum[half], unit);
    }
}

/**
 *  @brief One BF16 work item on one warpgroup: scores, online softmax and P · V over every panel
 *      its rows see, then the output.
 *  @param[in] width Two or four 64-column output blocks in registers, see @c nk_attention_width_t.
 *  @param[in] mask Whether panels crossing the band's edges mask their scores, see
 *      @c nk_attention_mask_t; panels past the segment's keys always do.
 *
 *  Q, K and a BF16 V sit in 64-byte swizzled blocks of 64 rows, one block per 64 depth bytes; a
 *  transposed V sits as one swizzled 64-byte row per depth row. Warp @c w holds rows 16w to 16w +
 *  15, with scores and output in the Ampere tile's order: register 4t + 2h + e of a thread is row
 *  16w + lane / 4 + 8h at column 8t + 2 · (lane mod 4) + e.
 */
NUMKONG_DEVICE void nk_attention_block_bf16_hopper_(nk_attention_width_t width, nk_attention_mask_t mask,
                                                    nk_attention_arguments_t const *arguments,
                                                    nk_attention_work_t const *work, unsigned char *shared) {

    nk_f32_t const negative_infinity = nk_attention_negative_infinity_simt_();
    nk_attention_frame_hopper_t const frame = nk_attention_frame_hopper_(width, mask, 2, arguments, work, shared);
    // Retires the previous item's reads of every buffer this item refills.
    __syncthreads();

    if (frame.panel_first < frame.panel_end)
        nk_attention_stage_rows_hopper_(frame.keys_shared, frame.keys_plane,
                                        (nk_size_t)frame.panel_first * nk_attention_panel_k, frame.row_bytes);
    nk_commit_async_ampere_();

    nk_attention_stage_queries_b16_hopper_(&frame, arguments, work);

    nk_f32_t row_max[2] = {negative_infinity, negative_infinity}, row_sum[2] = {0, 0};
    nk_fui32_t output[nk_attention_wide_depth_ampere_k / 64 * 32];
#pragma unroll
    for (unsigned index = 0; index < frame.max_blocks * 32; ++index) output[index].u = 0;
    nk_f32_t const scale2 = arguments->scale2 * arguments->score_scale;

    for (unsigned panel_index = frame.panel_first; panel_index < frame.panel_end; ++panel_index) {
        nk_wait_async_ampere_(0);
        nk_fence_proxy_async_hopper_();
        // K has landed, Q too on the first panel, and every warp is done with the V this refills.
        __syncthreads();
        nk_size_t const panel_position = (nk_size_t)panel_index * nk_attention_panel_k;
        nk_attention_stage_rows_hopper_(frame.values_shared, frame.values_plane, panel_position, frame.row_bytes);
        nk_commit_async_ampere_();

        nk_fui32_t tile_scores[32];
#pragma unroll
        for (unsigned index = 0; index < 32; ++index) tile_scores[index].u = 0;
        nk_wgmma_fence_operands_hopper_(tile_scores, 32, nk_cross_epilogue_f32_k);
        nk_wgmma_fence_hopper_();
#pragma unroll
        for (unsigned step = 0; step < frame.max_blocks * 4; ++step) {
            if (step >= frame.depth_steps) continue;
            // K-major rows, the leading offset unread, each step starting where row 0 holds it.
            nk_u32_t const offset = (step >> 1) * nk_attention_block_bytes_hopper_k + (step & 1) * 32;
            nk_attention_scores_bf16_hopper_(
                tile_scores,
                nk_smem_descriptor_hopper_(frame.queries_address + offset, nk_smem_swizzle_64_hopper_k, 16, 512),
                nk_smem_descriptor_hopper_(frame.keys_address + offset, nk_smem_swizzle_64_hopper_k, 16, 512),
                step != 0);
        }
        nk_wgmma_commit_hopper_();
        nk_wgmma_wait_hopper_();
        nk_wgmma_fence_operands_hopper_(tile_scores, 32, nk_cross_epilogue_f32_k);

        nk_f32_t correction[2];
        nk_attention_softmax_hopper_(nk_cross_epilogue_f32_k, tile_scores, &frame, work, scale2, panel_position,
                                     row_max, row_sum, correction);
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
                nk_attention_weights_bf16_ampere_(four, packed, &row_sum[half]);
                probabilities[position_group][half] = packed[0];
                probabilities[position_group][half + 2] = packed[1];
            }
#pragma unroll
        for (unsigned index = 0; index < frame.max_blocks * 32; ++index)
            output[index].f *= correction[(index >> 1) & 1];

        nk_wait_async_ampere_(0);
        nk_fence_proxy_async_hopper_();
        // V has landed, and every warp is done with the K this refills.
        __syncthreads();
        if (panel_index + 1 < frame.panel_end)
            nk_attention_stage_rows_hopper_(frame.keys_shared, frame.keys_plane, panel_position + nk_attention_panel_k,
                                            frame.row_bytes);
        nk_commit_async_ampere_();

#pragma unroll
        for (unsigned position_group = 0; position_group < 4; ++position_group)
            nk_wgmma_fence_operands_hopper_((nk_fui32_t *)probabilities[position_group], 4, nk_cross_epilogue_i32_k);
        nk_wgmma_fence_operands_hopper_(output, frame.max_blocks * 32, nk_cross_epilogue_f32_k);
        nk_wgmma_fence_hopper_();
        // N-major rows of 32 columns, the next 32 a block on, in 8-row groups 512 bytes apart.
#pragma unroll
        for (unsigned block = 0; block < frame.max_blocks; ++block) {
            if (block >= frame.depth_blocks) continue;
#pragma unroll
            for (unsigned position_group = 0; position_group < 4; ++position_group) {
                nk_u32_t const address = frame.values_address + block * 2 * nk_attention_block_bytes_hopper_k +
                                         position_group * 16 * nk_attention_swizzle_bytes_hopper_k;
                nk_attention_values_bf16_hopper_(output + block * 32, probabilities[position_group],
                                                 nk_smem_descriptor_hopper_(address, nk_smem_swizzle_64_hopper_k,
                                                                            nk_attention_block_bytes_hopper_k, 512),
                                                 1);
            }
        }
        nk_wgmma_commit_hopper_();
        nk_wgmma_wait_hopper_();
        nk_wgmma_fence_operands_hopper_(output, frame.max_blocks * 32, nk_cross_epilogue_f32_k);
    }

    nk_attention_finish_hopper_(&frame, output, row_max, row_sum, nk_attention_weight_unit_bf16_ampere_(), arguments,
                                work);
}

/**
 *  @brief One E4M3 work item on one warpgroup, on native E4M3 MMAs with E4M3 probabilities.
 *  @sa nk_attention_block_bf16_hopper_ for the parameters.
 */
NUMKONG_DEVICE void nk_attention_block_e4m3_hopper_(nk_attention_width_t width, nk_attention_mask_t mask,
                                                    nk_attention_arguments_t const *arguments,
                                                    nk_attention_work_t const *work, unsigned char *shared) {

    nk_f32_t const negative_infinity = nk_attention_negative_infinity_simt_();
    nk_attention_frame_hopper_t const frame = nk_attention_frame_hopper_(width, mask, 1, arguments, work, shared);
    // Retires the previous item's reads of every buffer this item refills.
    __syncthreads();

    if (frame.panel_first < frame.panel_end)
        nk_attention_stage_rows_hopper_(frame.keys_shared, frame.keys_plane,
                                        (nk_size_t)frame.panel_first * nk_attention_panel_k, frame.row_bytes);
    nk_commit_async_ampere_();

    nk_attention_stage_queries_b8_hopper_(&frame, arguments, work);

    nk_f32_t row_max[2] = {negative_infinity, negative_infinity}, row_sum[2] = {0, 0};
    nk_fui32_t output[nk_attention_wide_depth_ampere_k / 64 * 32];
#pragma unroll
    for (unsigned index = 0; index < frame.max_blocks * 32; ++index) output[index].u = 0;
    nk_f32_t const scale2 = arguments->scale2 * arguments->score_scale;

    for (unsigned panel_index = frame.panel_first; panel_index < frame.panel_end; ++panel_index) {
        nk_wait_async_ampere_(0);
        nk_fence_proxy_async_hopper_();
        // K has landed, Q too on the first panel, and every warp is done with the V this refills.
        __syncthreads();
        nk_size_t const panel_position = (nk_size_t)panel_index * nk_attention_panel_k;
        nk_attention_stage_columns_hopper_(frame.values_shared, frame.values_plane, frame.positions_padded,
                                           panel_position, frame.depth_padded);
        nk_commit_async_ampere_();

        nk_fui32_t tile_scores[32];
#pragma unroll
        for (unsigned index = 0; index < 32; ++index) tile_scores[index].u = 0;
        nk_wgmma_fence_operands_hopper_(tile_scores, 32, nk_cross_epilogue_f32_k);
        nk_wgmma_fence_hopper_();
#pragma unroll
        for (unsigned step = 0; step < frame.max_blocks * 4; ++step) {
            if (step >= frame.depth_steps) continue;
            // K-major rows, the leading offset unread, each step starting where row 0 holds it.
            nk_u32_t const offset = (step >> 1) * nk_attention_block_bytes_hopper_k + (step & 1) * 32;
            nk_attention_scores_e4m3_hopper_(
                tile_scores,
                nk_smem_descriptor_hopper_(frame.queries_address + offset, nk_smem_swizzle_64_hopper_k, 16, 512),
                nk_smem_descriptor_hopper_(frame.keys_address + offset, nk_smem_swizzle_64_hopper_k, 16, 512),
                step != 0);
        }
        nk_wgmma_commit_hopper_();
        nk_wgmma_wait_hopper_();
        nk_wgmma_fence_operands_hopper_(tile_scores, 32, nk_cross_epilogue_f32_k);

        nk_f32_t correction[2];
        nk_attention_softmax_hopper_(nk_cross_epilogue_f32_k, tile_scores, &frame, work, scale2, panel_position,
                                     row_max, row_sum, correction);
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
                nk_attention_weights_e4m3_ada_(four, packed, &row_sum[half]);
                probabilities[position_group >> 1][(position_group & 1) * 2 + half] = packed[0];
            }

        nk_wait_async_ampere_(0);
        nk_fence_proxy_async_hopper_();
        // V has landed, and every warp is done with the K this refills.
        __syncthreads();
        if (panel_index + 1 < frame.panel_end)
            nk_attention_stage_rows_hopper_(frame.keys_shared, frame.keys_plane, panel_position + nk_attention_panel_k,
                                            frame.row_bytes);
        nk_commit_async_ampere_();

#pragma unroll
        for (unsigned position_group = 0; position_group < 2; ++position_group)
            nk_wgmma_fence_operands_hopper_((nk_fui32_t *)probabilities[position_group], 4, nk_cross_epilogue_i32_k);
#pragma unroll
        for (unsigned block = 0; block < frame.max_blocks; ++block) {
            if (block >= frame.depth_blocks) continue;
            nk_fui32_t sums[32];
#pragma unroll
            for (unsigned index = 0; index < 32; ++index) sums[index].u = 0;
            nk_wgmma_fence_operands_hopper_(sums, 32, nk_cross_epilogue_f32_k);
            nk_wgmma_fence_hopper_();
#pragma unroll
            for (unsigned step = 0; step < 2; ++step)
                nk_attention_values_e4m3_hopper_(
                    sums, probabilities[step],
                    nk_smem_descriptor_hopper_(
                        frame.values_address + block * nk_attention_block_bytes_hopper_k + step * 32,
                        nk_smem_swizzle_64_hopper_k, 16, 512),
                    step);
            nk_wgmma_commit_hopper_();
            nk_wgmma_wait_hopper_();
            nk_wgmma_fence_operands_hopper_(sums, 32, nk_cross_epilogue_f32_k);
#pragma unroll
            for (unsigned index = 0; index < 32; ++index)
                output[block * 32 + index].f = fmaf(output[block * 32 + index].f, correction[(index >> 1) & 1],
                                                    sums[index].f);
        }
    }

    nk_attention_finish_hopper_(&frame, output, row_max, row_sum, nk_attention_weight_unit_e4m3_ada_(), arguments,
                                work);
}

/**
 *  @brief One I8 work item on one warpgroup: exact integer scores, U8 probabilities, and P · V
 *      sums converted per panel.
 *  @sa nk_attention_block_bf16_hopper_ for the parameters.
 */
NUMKONG_DEVICE void nk_attention_block_i8_hopper_(nk_attention_width_t width, nk_attention_mask_t mask,
                                                  nk_attention_arguments_t const *arguments,
                                                  nk_attention_work_t const *work, unsigned char *shared) {

    nk_f32_t const negative_infinity = nk_attention_negative_infinity_simt_();
    nk_attention_frame_hopper_t const frame = nk_attention_frame_hopper_(width, mask, 1, arguments, work, shared);
    // Retires the previous item's reads of every buffer this item refills.
    __syncthreads();

    if (frame.panel_first < frame.panel_end)
        nk_attention_stage_rows_hopper_(frame.keys_shared, frame.keys_plane,
                                        (nk_size_t)frame.panel_first * nk_attention_panel_k, frame.row_bytes);
    nk_commit_async_ampere_();

    nk_attention_stage_queries_b8_hopper_(&frame, arguments, work);

    nk_f32_t row_max[2] = {negative_infinity, negative_infinity}, row_sum[2] = {0, 0};
    nk_fui32_t output[nk_attention_wide_depth_ampere_k / 64 * 32];
#pragma unroll
    for (unsigned index = 0; index < frame.max_blocks * 32; ++index) output[index].u = 0;
    nk_f32_t const scale2 = arguments->scale2 * arguments->score_scale;

    for (unsigned panel_index = frame.panel_first; panel_index < frame.panel_end; ++panel_index) {
        nk_wait_async_ampere_(0);
        nk_fence_proxy_async_hopper_();
        // K has landed, Q too on the first panel, and every warp is done with the V this refills.
        __syncthreads();
        nk_size_t const panel_position = (nk_size_t)panel_index * nk_attention_panel_k;
        nk_attention_stage_columns_hopper_(frame.values_shared, frame.values_plane, frame.positions_padded,
                                           panel_position, frame.depth_padded);
        nk_commit_async_ampere_();

        nk_fui32_t tile_scores[32];
#pragma unroll
        for (unsigned index = 0; index < 32; ++index) tile_scores[index].u = 0;
        nk_wgmma_fence_operands_hopper_(tile_scores, 32, nk_cross_epilogue_i32_to_f32_k);
        nk_wgmma_fence_hopper_();
#pragma unroll
        for (unsigned step = 0; step < frame.max_blocks * 4; ++step) {
            if (step >= frame.depth_steps) continue;
            // K-major rows, the leading offset unread, each step starting where row 0 holds it.
            nk_u32_t const offset = (step >> 1) * nk_attention_block_bytes_hopper_k + (step & 1) * 32;
            nk_attention_scores_i8_hopper_(
                tile_scores,
                nk_smem_descriptor_hopper_(frame.queries_address + offset, nk_smem_swizzle_64_hopper_k, 16, 512),
                nk_smem_descriptor_hopper_(frame.keys_address + offset, nk_smem_swizzle_64_hopper_k, 16, 512),
                step != 0);
        }
        nk_wgmma_commit_hopper_();
        nk_wgmma_wait_hopper_();
        nk_wgmma_fence_operands_hopper_(tile_scores, 32, nk_cross_epilogue_i32_to_f32_k);

        nk_f32_t correction[2];
        nk_attention_softmax_hopper_(nk_cross_epilogue_i32_to_f32_k, tile_scores, &frame, work, scale2, panel_position,
                                     row_max, row_sum, correction);
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
                nk_attention_weights_u8_ampere_(four, packed, &row_sum[half]);
                probabilities[position_group >> 1][(position_group & 1) * 2 + half] = packed[0];
            }

        nk_wait_async_ampere_(0);
        nk_fence_proxy_async_hopper_();
        // V has landed, and every warp is done with the K this refills.
        __syncthreads();
        if (panel_index + 1 < frame.panel_end)
            nk_attention_stage_rows_hopper_(frame.keys_shared, frame.keys_plane, panel_position + nk_attention_panel_k,
                                            frame.row_bytes);
        nk_commit_async_ampere_();

#pragma unroll
        for (unsigned position_group = 0; position_group < 2; ++position_group)
            nk_wgmma_fence_operands_hopper_((nk_fui32_t *)probabilities[position_group], 4, nk_cross_epilogue_i32_k);
#pragma unroll
        for (unsigned block = 0; block < frame.max_blocks; ++block) {
            if (block >= frame.depth_blocks) continue;
            nk_fui32_t sums[32];
#pragma unroll
            for (unsigned index = 0; index < 32; ++index) sums[index].u = 0;
            nk_wgmma_fence_operands_hopper_(sums, 32, nk_cross_epilogue_i32_to_f32_k);
            nk_wgmma_fence_hopper_();
#pragma unroll
            for (unsigned step = 0; step < 2; ++step)
                nk_attention_values_u8i8_hopper_(
                    sums, probabilities[step],
                    nk_smem_descriptor_hopper_(
                        frame.values_address + block * nk_attention_block_bytes_hopper_k + step * 32,
                        nk_smem_swizzle_64_hopper_k, 16, 512),
                    step);
            nk_wgmma_commit_hopper_();
            nk_wgmma_wait_hopper_();
            nk_wgmma_fence_operands_hopper_(sums, 32, nk_cross_epilogue_i32_to_f32_k);
#pragma unroll
            for (unsigned index = 0; index < 32; ++index)
                output[block * 32 + index].f = fmaf(output[block * 32 + index].f, correction[(index >> 1) & 1],
                                                    (nk_f32_t)sums[index].i);
        }
    }

    nk_attention_finish_hopper_(&frame, output, row_max, row_sum, nk_attention_weight_unit_u8_ampere_(), arguments,
                                work);
}

/** Dynamic shared memory of a launch, aligned up to the 512-byte boundary the 64-byte swizzle
 *  repeats on. */
NUMKONG_DEVICE unsigned char *nk_attention_dynamic_shared_hopper_(void) {
    extern __shared__ __align__(128) unsigned char nk_attention_shared_hopper_[];
    return nk_shared_aligned_ampere_(nk_attention_shared_hopper_, 512);
}

/**
 *  @brief Every @c bf16 work item of a launch, walked with a stride of the grid, on a dynamic
 *      shared buffer aligned up to the 512-byte boundary the 64-byte swizzle repeats on.
 *  @sa nk_attention_block_bf16_hopper_ for the parameters.
 */
NUMKONG_DEVICE void nk_attention_tile_bf16_hopper_(nk_attention_width_t width, nk_attention_mask_t mask,
                                                   nk_attention_arguments_t const *arguments) {
    __shared__ nk_u64_t prefix[nk_attention_threads_k + 1];
    __shared__ nk_u64_t warp_totals[nk_attention_threads_k / 32];
    unsigned char *shared = nk_attention_dynamic_shared_hopper_();
    nk_attention_schedule_t schedule;
    if (!nk_attention_schedule_start_cuda_(arguments, &schedule, prefix, warp_totals)) return;
    nk_attention_work_t work;
    for (nk_size_t item = blockIdx.x; nk_attention_schedule_next_cuda_(&schedule, prefix, warp_totals, item, &work);
         item += gridDim.x)
        nk_attention_block_bf16_hopper_(width, mask, arguments, &work, shared);
}

/**
 *  @brief Every @c e4m3 work item of a launch, walked with a stride of the grid, on a dynamic
 *      shared buffer aligned up to the 512-byte boundary the 64-byte swizzle repeats on.
 *  @sa nk_attention_block_e4m3_hopper_ for the parameters.
 */
NUMKONG_DEVICE void nk_attention_tile_e4m3_hopper_(nk_attention_width_t width, nk_attention_mask_t mask,
                                                   nk_attention_arguments_t const *arguments) {
    __shared__ nk_u64_t prefix[nk_attention_threads_k + 1];
    __shared__ nk_u64_t warp_totals[nk_attention_threads_k / 32];
    unsigned char *shared = nk_attention_dynamic_shared_hopper_();
    nk_attention_schedule_t schedule;
    if (!nk_attention_schedule_start_cuda_(arguments, &schedule, prefix, warp_totals)) return;
    nk_attention_work_t work;
    for (nk_size_t item = blockIdx.x; nk_attention_schedule_next_cuda_(&schedule, prefix, warp_totals, item, &work);
         item += gridDim.x)
        nk_attention_block_e4m3_hopper_(width, mask, arguments, &work, shared);
}

/**
 *  @brief Every @c i8 work item of a launch, walked with a stride of the grid, on a dynamic shared
 *      buffer aligned up to the 512-byte boundary the 64-byte swizzle repeats on.
 *  @sa nk_attention_block_i8_hopper_ for the parameters.
 */
NUMKONG_DEVICE void nk_attention_tile_i8_hopper_(nk_attention_width_t width, nk_attention_mask_t mask,
                                                 nk_attention_arguments_t const *arguments) {
    __shared__ nk_u64_t prefix[nk_attention_threads_k + 1];
    __shared__ nk_u64_t warp_totals[nk_attention_threads_k / 32];
    unsigned char *shared = nk_attention_dynamic_shared_hopper_();
    nk_attention_schedule_t schedule;
    if (!nk_attention_schedule_start_cuda_(arguments, &schedule, prefix, warp_totals)) return;
    nk_attention_work_t work;
    for (nk_size_t item = blockIdx.x; nk_attention_schedule_next_cuda_(&schedule, prefix, warp_totals, item, &work);
         item += gridDim.x)
        nk_attention_block_i8_hopper_(width, mask, arguments, &work, shared);
}

#pragma endregion Tile

#pragma region Launch

/** Places Q, K and V in dynamic shared memory, returning the bytes a block needs. */
NUMKONG_INLINE nk_size_t nk_attention_shared_layout_hopper_(nk_size_t element_bytes, nk_size_t depth,
                                                            nk_attention_arguments_t *arguments) {
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
NUMKONG_INLINE nk_size_t nk_attention_shared_ceiling_hopper_(nk_size_t element_bytes, nk_attention_width_t width) {
    nk_attention_arguments_t deepest;
    nk_size_t const depth = width == nk_attention_width_128_k ? nk_attention_narrow_depth_ampere_k
                                                              : nk_attention_wide_depth_ampere_k;
    return nk_attention_shared_layout_hopper_(element_bytes, depth, &deepest);
}

/**
 *  @brief Validates the contract and launches the kernel for the depth's width with as many blocks
 *      as stay resident, Q, K and V each taking 64-byte swizzled blocks past a 512-byte alignment.
 *  @param[in] element_bytes Bytes of one input element: 2 or 1.
 *  @param[in] score_scale Undoes the power of two that converting Q and K puts on scores, or 1.
 *  @param[in] output_scale Undoes the power of two that converting V puts on the output, or 1.
 */
NUMKONG_INLINE nk_status_t nk_attention_launch_hopper_(
    void const *narrow_kernel, void const *wide_kernel, void const *fallback_kernel, nk_size_t element_bytes,
    void const *queries, void const *packed, nk_f32_t *output, nk_f32_t *log_sum_exp, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_token_count,
    nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale, nk_f32_t score_scale, nk_f32_t output_scale,
    nk_size_t keys_before, nk_size_t keys_after, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    if (((nk_size_t)packed & 15) || (((nk_size_t)output | output_stride) & 3)) return nk_misaligned_k;
    if (key_value_head_count == 0 || head_count % key_value_head_count != 0) return nk_unexpected_dimensions_k;
    if (tasks_begin >= tasks_end || depth == 0) return nk_success_k;
    nk_attention_arguments_t arguments = nk_attention_arguments_init_simt_(
        queries, packed, output, log_sum_exp, head_count, key_value_head_count, depth, query_offsets, query_token_count,
        query_stride, output_stride, scale, score_scale, output_scale, keys_before, keys_after, tasks_begin, tasks_end);
    if (depth > nk_attention_wide_depth_ampere_k)
        return nk_launch_resident_cuda_(fallback_kernel, nk_attention_threads_k, 0, 0, NUMKONG_SIZE_MAX, &arguments,
                                        stream);
    nk_attention_width_t const width = depth <= nk_attention_narrow_depth_ampere_k ? nk_attention_width_128_k
                                                                                   : nk_attention_width_256_k;
    nk_size_t const shared_bytes = nk_attention_shared_layout_hopper_(element_bytes, depth, &arguments);
    return nk_launch_resident_cuda_(
        width == nk_attention_width_128_k ? narrow_kernel : wide_kernel, nk_attention_threads_k, shared_bytes,
        nk_attention_shared_ceiling_hopper_(element_bytes, width), NUMKONG_SIZE_MAX, &arguments, stream);
}

/** The launch for BF16. */
NUMKONG_INLINE nk_status_t nk_attention_launch_bf16_hopper_(
    void const *narrow_kernel, void const *wide_kernel, void const *fallback_kernel, void const *queries,
    void const *packed, nk_f32_t *output, nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count,
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_token_count, nk_size_t query_stride,
    nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_size_t tasks_begin,
    nk_size_t tasks_end, nk_stream_t stream) {
    return nk_attention_launch_hopper_(narrow_kernel, wide_kernel, fallback_kernel, 2, queries, packed, output,
                                       log_sum_exp, head_count, key_value_head_count, depth, query_offsets,
                                       query_token_count, query_stride, output_stride, scale, 1.0f, 1.0f, keys_before,
                                       keys_after, tasks_begin, tasks_end, stream);
}

/** The launch for E4M3. */
NUMKONG_INLINE nk_status_t nk_attention_launch_e4m3_hopper_(
    void const *narrow_kernel, void const *wide_kernel, void const *fallback_kernel, void const *queries,
    void const *packed, nk_f32_t *output, nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count,
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_token_count, nk_size_t query_stride,
    nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_size_t tasks_begin,
    nk_size_t tasks_end, nk_stream_t stream) {
    return nk_attention_launch_hopper_(narrow_kernel, wide_kernel, fallback_kernel, 1, queries, packed, output,
                                       log_sum_exp, head_count, key_value_head_count, depth, query_offsets,
                                       query_token_count, query_stride, output_stride, scale, 1.0f, 1.0f, keys_before,
                                       keys_after, tasks_begin, tasks_end, stream);
}

/** The launch for I8. */
NUMKONG_INLINE nk_status_t nk_attention_launch_i8_hopper_(
    void const *narrow_kernel, void const *wide_kernel, void const *fallback_kernel, void const *queries,
    void const *packed, nk_f32_t *output, nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count,
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_token_count, nk_size_t query_stride,
    nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_size_t tasks_begin,
    nk_size_t tasks_end, nk_stream_t stream) {
    return nk_attention_launch_hopper_(narrow_kernel, wide_kernel, fallback_kernel, 1, queries, packed, output,
                                       log_sum_exp, head_count, key_value_head_count, depth, query_offsets,
                                       query_token_count, query_stride, output_stride, scale, 1.0f, 1.0f, keys_before,
                                       keys_after, tasks_begin, tasks_end, stream);
}

#pragma endregion Launch

#pragma region Instantiations

nk_define_attention_pack_size_simt_(bf16, hopper, 2)
nk_define_attention_packed_shape_cuda_(bf16, hopper)
nk_define_attention_pack_cuda_(bf16, hopper, bf16)
nk_define_attention_packed_simt_(bf16, hopper, cuda)

/** Both BF16 backward kernels, on the Ampere tensor-core loops when the launch gives them shared
 *  memory and the @c cuda kernels' loops when it does not. */
static __global__ void __launch_bounds__(nk_attention_threads_k)
    nk_attention_backward_keys_bf16_hopper_kernel_(nk_attention_backward_arguments_t arguments) {
    if (nk_dynamic_shared_bytes_ampere_()) nk_attention_backward_keys_ampere_(&arguments);
    else nk_attention_backward_keys_bf16_cuda_(&arguments);
}
static __global__ void __launch_bounds__(nk_attention_threads_k)
    nk_attention_backward_queries_bf16_hopper_kernel_(nk_attention_backward_arguments_t arguments) {
    if (nk_dynamic_shared_bytes_ampere_()) nk_attention_backward_queries_ampere_(&arguments);
    else nk_attention_backward_queries_bf16_cuda_(&arguments);
}
nk_define_attention_backward_cuda_(bf16, hopper, bf16, nk_attention_backward_launch_ampere_)

nk_define_attention_pack_size_simt_(e4m3, hopper, 1)
nk_define_attention_packed_shape_cuda_(e4m3, hopper)
nk_define_attention_pack_cuda_(e4m3, hopper, e4m3)
nk_define_attention_packed_simt_(e4m3, hopper, cuda)

nk_define_attention_pack_size_simt_(i8, hopper, 1)
nk_define_attention_packed_shape_cuda_(i8, hopper)
nk_define_attention_pack_cuda_(i8, hopper, i8)
nk_define_attention_packed_simt_(i8, hopper, cuda)

#pragma endregion Instantiations

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_HOPPER
#endif // NUMKONG_ARCH_CUDA_
#endif // NUMKONG_ATTENTION_HOPPER_CUH
