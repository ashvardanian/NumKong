/**
 *  @file include/numkong/dots/cdna3.cuh
 *  @author Ash Vardanian
 *  @date October 7, 2026
 *  @brief Batched Dot Products for AMD Instinct MI300, gfx942, on its SIMT cores.
 *
 *  @sa include/numkong/dots.h
 *  @sa include/numkong/dots/rocm.cuh
 *
 *  The @c rocm baseline's B32 tile folding through the dot instructions CDNA has and MI400 lacks:
 *  @c v_dot4_i32_i8 for I8, @c v_dot8_i32_i4 for I4, and @c v_dot2_f32_f16 for F16 and the Float8,
 *  Float6 and Float4 codes the tile widens into F16 pairs. The packs keep the @c rocm layout, and
 *  every other type runs the @c rocm kernels. CDNA4 inherits these helpers for its norms.
 */
#ifndef NUMKONG_DOTS_CDNA3_CUH
#define NUMKONG_DOTS_CDNA3_CUH

#if NUMKONG_ARCH_ROCM_CDNA3_

#include "numkong/dots/rocm.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Instructions

/** Adds the four signed byte products of @p a and @p b to @p sum with CDNA3's @c v_dot4_i32_i8 . */
NUMKONG_DEVICE nk_i32_t nk_dot_i8x4_cdna3_(nk_u32_t a, nk_u32_t b, nk_i32_t sum) {
    return __builtin_amdgcn_sdot4((int)a, (int)b, sum, 0);
}

/** Adds the eight signed nibble products of @p a and @p b, paired by position, to @p sum with
 *  CDNA3's @c v_dot8_i32_i4 . */
NUMKONG_DEVICE nk_i32_t nk_dot_i4x8_cdna3_(nk_u32_t a, nk_u32_t b, nk_i32_t sum) {
    return __builtin_amdgcn_sdot8((int)a, (int)b, sum, 0);
}

/** Adds both F16 products of @p a and @p b to @p sum with CDNA3's @c v_dot2_f32_f16 . */
NUMKONG_DEVICE nk_f32_t nk_dot_f16x2_cdna3_(nk_u32_t a, nk_u32_t b, nk_f32_t sum) {
    union {
        nk_u32_t bits;
        _Float16_2 halves;
    } const a_pair = {a}, b_pair = {b};
    return __builtin_amdgcn_fdot2(a_pair.halves, b_pair.halves, sum, 0);
}

#pragma endregion Instructions

#if NUMKONG_TARGET_CDNA3

#pragma region Tile

enum { nk_cross_threads_b32_cdna3_k = nk_cross_threads_simt_k, nk_cross_tile_b32_cdna3_k = nk_cross_tile_simt_k };

/** Folds one 32-bit word of each operand into @p sum through CDNA3's dots, for the F16 pair, byte
 *  and nibble accumulations the @c cdna3 kernels take. */
NUMKONG_DEVICE void nk_cross_fold_b32_cdna3_(nk_cross_accumulation_t accumulation, nk_fui32_t *sum, nk_u32_t a,
                                             nk_u32_t b) {
    switch (accumulation) {
    case nk_cross_accumulation_f16x2_k: sum->f = nk_dot_f16x2_cdna3_(a, b, sum->f); break;
    case nk_cross_accumulation_i8x4_k: sum->i = nk_dot_i8x4_cdna3_(a, b, sum->i); break;
    default: sum->i = nk_dot_i4x8_cdna3_(a, b, sum->i); break;
    }
}

/** Folds one staged slab into this thread's grid of 32-bit sums. */
NUMKONG_DEVICE void nk_cross_fold_slab_b32_cdna3_(
    nk_cross_accumulation_t accumulation, nk_u32_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
    nk_u32_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
    nk_fui32_t sums[nk_cross_thread_tile_simt_k][nk_cross_thread_tile_simt_k]) {
    unsigned const thread_column = threadIdx.x % nk_cross_grid_side_simt_k;
    unsigned const thread_row = threadIdx.x / nk_cross_grid_side_simt_k;
#pragma unroll
    for (unsigned offset = 0; offset < nk_cross_slab_simt_k; ++offset) {
        nk_u32_t a_words[nk_cross_thread_tile_simt_k], b_words[nk_cross_thread_tile_simt_k];
#pragma unroll
        for (unsigned step = 0; step < nk_cross_thread_tile_simt_k; ++step)
            a_words[step] = a_slab[offset][thread_row + nk_cross_grid_side_simt_k * step],
            b_words[step] = b_slab[offset][thread_column + nk_cross_grid_side_simt_k * step];
#pragma unroll
        for (unsigned row_step = 0; row_step < nk_cross_thread_tile_simt_k; ++row_step)
#pragma unroll
            for (unsigned column_step = 0; column_step < nk_cross_thread_tile_simt_k; ++column_step)
                nk_cross_fold_b32_cdna3_(accumulation, &sums[row_step][column_step], a_words[row_step],
                                         b_words[column_step]);
    }
}

/** @c nk_cross_tile_b32_rocm_ folding through CDNA3's dots. */
NUMKONG_DEVICE void nk_cross_tile_b32_cdna3_(nk_dtype_t dtype, nk_cross_accumulation_t accumulation,
                                             nk_cross_triangle_t triangle, nk_cross_metric_t metric,
                                             nk_cross_tile_arguments_t const *arguments) {
    // One extra column breaks the 64-word stride that would put a slab's stores on one bank.
    __shared__ nk_u32_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1];
    __shared__ nk_u32_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1];
    __shared__ nk_fui32_t norms[2][nk_cross_tile_simt_k];
    nk_size_t const words = nk_size_divide_round_up_(arguments->depth, nk_cross_b32_dimensions_(accumulation));

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        nk_size_t const first_row = arguments->row_start + tile / arguments->column_tiles * nk_cross_tile_simt_k;
        nk_size_t const first_column = tile % arguments->column_tiles * nk_cross_tile_simt_k;
        if (triangle == nk_cross_triangle_upper_k && first_column + nk_cross_tile_simt_k <= first_row) continue;
        nk_fui32_t sums[nk_cross_thread_tile_simt_k][nk_cross_thread_tile_simt_k] = {{{0}}};
        nk_fui32_t a_norms[nk_cross_loads_simt_k] = {{0}}, b_norms[nk_cross_loads_simt_k] = {{0}};
        for (nk_size_t slab = 0; slab < words; slab += nk_cross_slab_simt_k) {
            nk_u32_t a_words[nk_cross_loads_simt_k], b_words[nk_cross_loads_simt_k];
            nk_cross_stage_slab_b32_simt_(dtype, accumulation, arguments, first_row, first_column, slab, words, a_slab,
                                          b_slab, a_words, b_words);
            if (metric != nk_cross_metric_dot_k)
#pragma unroll
                for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step) {
                    nk_cross_fold_b32_cdna3_(accumulation, &a_norms[step], a_words[step], a_words[step]);
                    if (triangle == nk_cross_triangle_upper_k)
                        nk_cross_fold_b32_cdna3_(accumulation, &b_norms[step], b_words[step], b_words[step]);
                }
            __syncthreads();
            nk_cross_fold_slab_b32_cdna3_(accumulation, a_slab, b_slab, sums);
            __syncthreads();
        }
        if (metric != nk_cross_metric_dot_k) {
#pragma unroll
            for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step)
#pragma unroll
                for (unsigned offset = nk_cross_slab_simt_k / 2; offset != 0; offset >>= 1) {
                    a_norms[step] = nk_cross_merge_lanes_b32_rocm_(accumulation, offset, a_norms[step]);
                    if (triangle == nk_cross_triangle_upper_k)
                        b_norms[step] = nk_cross_merge_lanes_b32_rocm_(accumulation, offset, b_norms[step]);
                }
            nk_cross_publish_norms_b32_simt_(triangle, arguments, first_column, a_norms, b_norms, norms);
        }
        nk_cross_store_tile_b32_simt_(accumulation, triangle, metric, arguments, first_row, first_column, sums, norms);
    }
}

#pragma endregion Tile

#pragma region Instantiations

nk_define_cross_pack_rocm_(f16, cdna3, f16, f16, nk_load_b8_, f32, nk_f16_lane_sumsq_, 8, 1)
nk_define_cross_rocm_(dot, f16, cdna3, b32_cdna3, f16, f16, f32, 8, 1, nk_f16_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_pack_rocm_(e5m2, cdna3, e5m2, e5m2, nk_load_b8_, f32, nk_e5m2_lane_sumsq_, 16, 1)
nk_define_cross_rocm_(dot, e5m2, cdna3, b32_cdna3, e5m2, e5m2, f32, 16, 1, nk_e5m2_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_pack_rocm_(e4m3, cdna3, e4m3, e4m3, nk_load_b8_, f32, nk_e4m3_lane_sumsq_, 16, 1)
nk_define_cross_rocm_(dot, e4m3, cdna3, b32_cdna3, e4m3, e4m3, f32, 16, 1, nk_e4m3_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_pack_rocm_(e3m2, cdna3, e3m2, e3m2, nk_load_b8_, f32, nk_e3m2_lane_sumsq_, 16, 1)
nk_define_cross_rocm_(dot, e3m2, cdna3, b32_cdna3, e3m2, e3m2, f32, 16, 1, nk_e3m2_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_pack_rocm_(e2m3, cdna3, e2m3, e2m3, nk_load_b8_, f32, nk_e2m3_lane_sumsq_, 16, 1)
nk_define_cross_rocm_(dot, e2m3, cdna3, b32_cdna3, e2m3, e2m3, f32, 16, 1, nk_e2m3_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_pack_rocm_(e2m1, cdna3, e2m1x2, e2m1x2, nk_load_b8_, f32, nk_e2m1_lane_sumsq_, 32, 2)
nk_define_cross_rocm_(dot, e2m1, cdna3, b32_cdna3, e2m1x2, e2m1x2, f32, 32, 2, nk_e2m1_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_pack_rocm_(i8, cdna3, i8, i8, nk_load_b8_, u32, nk_i8_lane_sumsq_, 16, 1)
nk_define_cross_rocm_(dot, i8, cdna3, b32_cdna3, i8, i8, i32, 16, 1, nk_i8_k, nk_cross_accumulation_i8x4_k)
nk_define_cross_pack_rocm_(i4, cdna3, i4x2, i4x2, nk_load_b8_, u32, nk_i4_lane_sumsq_, 32, 2)
nk_define_cross_rocm_(dot, i4, cdna3, b32_cdna3, i4x2, i4x2, i32, 32, 2, nk_i4_k, nk_cross_accumulation_i4x8_k)

#pragma endregion Instantiations

#endif // NUMKONG_TARGET_CDNA3

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_ROCM_CDNA3_
#endif // NUMKONG_DOTS_CDNA3_CUH
