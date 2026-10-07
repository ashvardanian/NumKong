/**
 *  @file include/numkong/dots/rocm.cuh
 *  @author Ash Vardanian
 *  @date October 3, 2026
 *  @brief The @c rocm baseline of batched dot products: its F64 and B32 tiles over the portable
 *      stages of `dots/simt.cuh`, the pack and tile launches, the generators every ROCm capability
 *      instantiates its exports with, and the baseline exports.
 *
 *  AMD stages depth in F16 pairs and nibbles as packed. Every ROCm device runs @c v_dot4_u32_u8,
 *  @c v_dot8_u32_u4 and @c v_perm_b32, but MI400 lacks CDNA's F16 pair dot and spells the signed
 *  byte dots differently, so the baseline folds those portably and `cdna3.cuh` adds CDNA's.
 *
 *  @sa include/numkong/dots/simt.cuh
 *  @sa include/numkong/dots/cuda.cuh
 *  @sa include/numkong/dots/cdna3.cuh
 */
#ifndef NUMKONG_DOTS_ROCM_CUH
#define NUMKONG_DOTS_ROCM_CUH

#include "numkong/rocm.cuh"
#include "numkong/dots/simt.cuh"

#if NUMKONG_ARCH_ROCM_

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Instructions

/** Adds the four unsigned byte products of @p a and @p b to @p sum with @c v_dot4_u32_u8 . */
NUMKONG_DEVICE nk_u32_t nk_dot_u8x4_rocm_(nk_u32_t a, nk_u32_t b, nk_u32_t sum) {
    return __builtin_amdgcn_udot4(a, b, sum, 0);
}

/** Adds the eight unsigned nibble products of @p a and @p b to @p sum with @c v_dot8_u32_u4 . */
NUMKONG_DEVICE nk_u32_t nk_dot_u4x8_rocm_(nk_u32_t a, nk_u32_t b, nk_u32_t sum) {
    return __builtin_amdgcn_udot8(a, b, sum, 0);
}

/** Adds the four signed byte products of @p a and @p b to @p sum through the portable fold, as CDNA
 *  and MI400 spell the signed dot differently. */
NUMKONG_DEVICE nk_i32_t nk_dot_i8x4_rocm_(nk_u32_t a, nk_u32_t b, nk_i32_t sum) { return nk_dot_i8x4_simt_(a, b, sum); }

/** Adds the eight signed nibble products of @p a and @p b to @p sum through the portable fold, as
 *  CDNA and MI400 spell the signed dot differently. */
NUMKONG_DEVICE nk_i32_t nk_dot_i4x8_rocm_(nk_u32_t a, nk_u32_t b, nk_i32_t sum) { return nk_dot_i4x8_simt_(a, b, sum); }

/** Adds both F16 products of @p a and @p b to @p sum through the portable fold, as MI400 has no
 *  @c v_dot2_f32_f16 . */
NUMKONG_DEVICE nk_f32_t nk_dot_f16x2_rocm_(nk_u32_t a, nk_u32_t b, nk_f32_t sum) {
    return nk_dot_f16x2_simt_(a, b, sum);
}

/** Byte @c i of the result is byte @c selector.u8[i] of the 8 bytes @p high : @p low, as
 *  @c v_perm_b32 picks them. */
NUMKONG_DEVICE nk_u32_t nk_byte_permute_rocm_(nk_u32_t high, nk_u32_t low, nk_u32_t selector) {
    return __builtin_amdgcn_perm(high, low, selector);
}

#pragma endregion Instructions

/*  The baseline tiles over the portable stages: each launches the SIMT shape under its own stem, so
 *  the generators paste it, and folds, merges and rounds through AMD's instructions. */
#pragma region Baseline Tiles

enum {
    nk_cross_threads_f64_rocm_k = nk_cross_threads_simt_k,
    nk_cross_tile_f64_rocm_k = nk_cross_tile_simt_k,
    nk_cross_threads_b32_rocm_k = nk_cross_threads_simt_k,
    nk_cross_tile_b32_rocm_k = nk_cross_tile_simt_k,
};

/** Adds a × b into a running sum: one F64 FMA, or Dot2's TwoProd and TwoSum with both errors kept
 *  apart. */
NUMKONG_DEVICE void nk_cross_step_f64_rocm_(nk_cross_accumulation_t accumulation, nk_f64_t a, nk_f64_t b, nk_f64_t *sum,
                                            nk_f64_t *compensation) {
    if (accumulation == nk_cross_accumulation_f64_k) {
        *sum = __fma_rn(a, b, *sum);
        return;
    }
    nk_f64_t const product = nk_f64_mul_rn_rocm_(a, b), product_error = __fma_rn(a, b, -product);
    nk_f64_t const total = nk_f64_add_rn_rocm_(*sum, product), virtual_addend = nk_f64_sub_rn_rocm_(total, *sum);
    nk_f64_t const augend_error = nk_f64_sub_rn_rocm_(*sum, nk_f64_sub_rn_rocm_(total, virtual_addend));
    nk_f64_t const sum_error = nk_f64_add_rn_rocm_(augend_error, nk_f64_sub_rn_rocm_(product, virtual_addend));
    *sum = total;
    *compensation = nk_f64_add_rn_rocm_(*compensation, nk_f64_add_rn_rocm_(sum_error, product_error));
}

/** Merges the running sum of lane `lane ^ offset` into this one, through TwoSum under Dot2. */
NUMKONG_DEVICE void nk_cross_merge_lanes_f64_rocm_(nk_cross_accumulation_t accumulation, unsigned offset, nk_f64_t *sum,
                                                   nk_f64_t *compensation) {
    nk_f64_t const other_sum = nk_shuffle_xor_f64_rocm_(*sum, offset);
    if (accumulation == nk_cross_accumulation_f64_k) {
        *sum = nk_f64_add_rn_rocm_(*sum, other_sum);
        return;
    }
    nk_f64_t const other_compensation = nk_shuffle_xor_f64_rocm_(*compensation, offset);
    nk_f64_t const total = nk_f64_add_rn_rocm_(*sum, other_sum), virtual_addend = nk_f64_sub_rn_rocm_(total, *sum);
    nk_f64_t const augend_error = nk_f64_sub_rn_rocm_(*sum, nk_f64_sub_rn_rocm_(total, virtual_addend));
    nk_f64_t const sum_error = nk_f64_add_rn_rocm_(augend_error, nk_f64_sub_rn_rocm_(other_sum, virtual_addend));
    *sum = total;
    *compensation = nk_f64_add_rn_rocm_(nk_f64_add_rn_rocm_(*compensation, other_compensation), sum_error);
}

/** Folds one staged slab into this thread's grid of F64 sums and their compensations. */
NUMKONG_DEVICE void nk_cross_fold_slab_f64_rocm_(
    nk_cross_accumulation_t accumulation, nk_f64_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
    nk_f64_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
    nk_f64_t sums[nk_cross_thread_tile_simt_k][nk_cross_thread_tile_simt_k],
    nk_f64_t compensations[nk_cross_thread_tile_simt_k][nk_cross_thread_tile_simt_k]) {
    unsigned const thread_column = threadIdx.x % nk_cross_grid_side_simt_k;
    unsigned const thread_row = threadIdx.x / nk_cross_grid_side_simt_k;
#pragma unroll
    for (unsigned offset = 0; offset < nk_cross_slab_simt_k; ++offset) {
        nk_f64_t a_values[nk_cross_thread_tile_simt_k], b_values[nk_cross_thread_tile_simt_k];
#pragma unroll
        for (unsigned step = 0; step < nk_cross_thread_tile_simt_k; ++step)
            a_values[step] = a_slab[offset][thread_row + nk_cross_grid_side_simt_k * step],
            b_values[step] = b_slab[offset][thread_column + nk_cross_grid_side_simt_k * step];
#pragma unroll
        for (unsigned row_step = 0; row_step < nk_cross_thread_tile_simt_k; ++row_step)
#pragma unroll
            for (unsigned column_step = 0; column_step < nk_cross_thread_tile_simt_k; ++column_step)
                nk_cross_step_f64_rocm_(accumulation, a_values[row_step], b_values[column_step],
                                        &sums[row_step][column_step], &compensations[row_step][column_step]);
    }
}

/** @c nk_cross_tile_f64_cuda_ through AMD's shuffles and rounding. */
NUMKONG_DEVICE void nk_cross_tile_f64_rocm_(nk_dtype_t dtype, nk_cross_accumulation_t accumulation,
                                            nk_cross_triangle_t triangle, nk_cross_metric_t metric,
                                            nk_cross_tile_arguments_t const *arguments) {
    // One extra column breaks the 64-element stride that would put a slab's stores on one bank.
    __shared__ nk_f64_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1];
    __shared__ nk_f64_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1];
    __shared__ nk_f64_t norms[2][nk_cross_tile_simt_k];

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        nk_size_t const first_row = arguments->row_start + tile / arguments->column_tiles * nk_cross_tile_simt_k;
        nk_size_t const first_column = tile % arguments->column_tiles * nk_cross_tile_simt_k;
        if (triangle == nk_cross_triangle_upper_k && first_column + nk_cross_tile_simt_k <= first_row) continue;
        nk_f64_t sums[nk_cross_thread_tile_simt_k][nk_cross_thread_tile_simt_k] = {{0}};
        nk_f64_t compensations[nk_cross_thread_tile_simt_k][nk_cross_thread_tile_simt_k] = {{0}};
        nk_f64_t a_norms[nk_cross_loads_simt_k][2] = {{0}}, b_norms[nk_cross_loads_simt_k][2] = {{0}};
        for (nk_size_t slab = 0; slab < arguments->depth; slab += nk_cross_slab_simt_k) {
            nk_f64_t a_values[nk_cross_loads_simt_k], b_values[nk_cross_loads_simt_k];
            nk_cross_stage_slab_f64_simt_(dtype, arguments, first_row, first_column, slab, a_slab, b_slab, a_values,
                                          b_values);
            if (metric != nk_cross_metric_dot_k)
#pragma unroll
                for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step) {
                    nk_cross_step_f64_rocm_(nk_cross_accumulation_dot2_k, a_values[step], a_values[step],
                                            &a_norms[step][0], &a_norms[step][1]);
                    if (triangle == nk_cross_triangle_upper_k)
                        nk_cross_step_f64_rocm_(nk_cross_accumulation_dot2_k, b_values[step], b_values[step],
                                                &b_norms[step][0], &b_norms[step][1]);
                }
            __syncthreads();
            nk_cross_fold_slab_f64_rocm_(accumulation, a_slab, b_slab, sums, compensations);
            __syncthreads();
        }
        if (metric != nk_cross_metric_dot_k) {
            nk_f64_t a_rounded[nk_cross_loads_simt_k], b_rounded[nk_cross_loads_simt_k];
#pragma unroll
            for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step) {
#pragma unroll
                for (unsigned offset = nk_cross_slab_simt_k / 2; offset != 0; offset >>= 1) {
                    nk_cross_merge_lanes_f64_rocm_(nk_cross_accumulation_dot2_k, offset, &a_norms[step][0],
                                                   &a_norms[step][1]);
                    if (triangle == nk_cross_triangle_upper_k)
                        nk_cross_merge_lanes_f64_rocm_(nk_cross_accumulation_dot2_k, offset, &b_norms[step][0],
                                                       &b_norms[step][1]);
                }
                a_rounded[step] = nk_f64_add_rn_rocm_(a_norms[step][0], a_norms[step][1]);
                b_rounded[step] = nk_f64_add_rn_rocm_(b_norms[step][0], b_norms[step][1]);
            }
            nk_cross_publish_norms_f64_simt_(triangle, arguments, first_column, a_rounded, b_rounded, norms);
        }
#pragma unroll
        for (unsigned row_step = 0; row_step < nk_cross_thread_tile_simt_k; ++row_step)
#pragma unroll
            for (unsigned column_step = 0; column_step < nk_cross_thread_tile_simt_k; ++column_step)
                sums[row_step][column_step] = nk_f64_add_rn_rocm_(sums[row_step][column_step],
                                                                  compensations[row_step][column_step]);
        nk_cross_store_tile_f64_simt_(triangle, metric, arguments, first_row, first_column, sums, norms);
    }
}

/** Folds one 32-bit word of each operand into @p sum, by the instruction @p accumulation names: FMA
 *  for F32 words, and the @c rocm dots for F16 pairs, bytes and nibbles. */
NUMKONG_DEVICE void nk_cross_fold_b32_rocm_(nk_cross_accumulation_t accumulation, nk_fui32_t *sum, nk_u32_t a,
                                            nk_u32_t b) {
    switch (accumulation) {
    case nk_cross_accumulation_f32_k: sum->f = __fmaf_rn(__uint_as_float(a), __uint_as_float(b), sum->f); break;
    case nk_cross_accumulation_f16x2_k: sum->f = nk_dot_f16x2_rocm_(a, b, sum->f); break;
    case nk_cross_accumulation_i8x4_k: sum->i = nk_dot_i8x4_rocm_(a, b, sum->i); break;
    case nk_cross_accumulation_u8x4_k: sum->u = nk_dot_u8x4_rocm_(a, b, sum->u); break;
    case nk_cross_accumulation_i4x8_k: sum->i = nk_dot_i4x8_rocm_(a, b, sum->i); break;
    default: sum->u = nk_dot_u4x8_rocm_(a, b, sum->u); break;
    }
}

/** Adds lane `lane ^ offset`'s B32 sum into @p sum: in F32 for floats, wrapping for integers. */
NUMKONG_DEVICE nk_fui32_t nk_cross_merge_lanes_b32_rocm_(nk_cross_accumulation_t accumulation, unsigned offset,
                                                         nk_fui32_t sum) {
    nk_fui32_t other;
    other.u = nk_shuffle_xor_u32_rocm_(sum.u, offset);
    if (accumulation == nk_cross_accumulation_f32_k || accumulation == nk_cross_accumulation_f16x2_k) sum.f += other.f;
    else sum.u += other.u;
    return sum;
}

/** Folds one staged slab into this thread's grid of 32-bit sums. */
NUMKONG_DEVICE void nk_cross_fold_slab_b32_rocm_(
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
                nk_cross_fold_b32_rocm_(accumulation, &sums[row_step][column_step], a_words[row_step],
                                        b_words[column_step]);
    }
}

/** @c nk_cross_tile_b32_cuda_ folding through the @c rocm dots. */
NUMKONG_DEVICE void nk_cross_tile_b32_rocm_(nk_dtype_t dtype, nk_cross_accumulation_t accumulation,
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
                    nk_cross_fold_b32_rocm_(accumulation, &a_norms[step], a_words[step], a_words[step]);
                    if (triangle == nk_cross_triangle_upper_k)
                        nk_cross_fold_b32_rocm_(accumulation, &b_norms[step], b_words[step], b_words[step]);
                }
            __syncthreads();
            nk_cross_fold_slab_b32_rocm_(accumulation, a_slab, b_slab, sums);
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

#pragma endregion Baseline Tiles

/*  A lane's Dot2-compensated share of an F64 or F32 column's sum of squares, and the merges of 32
 *  lanes' shares into one packed norm, which the pack generator calls for every ROCm capability. */
#pragma region Pack Norms

NUMKONG_DEVICE nk_f64_t nk_f64_lane_sumsq_rocm_(unsigned char const *row, nk_size_t depth, unsigned lane) {
    nk_f64_t sum = 0, compensation = 0;
    for (nk_size_t index = lane; index < depth; index += 32) {
        unsigned long long bits = 0;
        for (unsigned byte = 0; byte < 8; ++byte) bits |= (unsigned long long)row[index * 8 + byte] << (byte * 8);
        nk_f64_t const value = __longlong_as_double((long long)bits);
        nk_cross_step_f64_rocm_(nk_cross_accumulation_dot2_k, value, value, &sum, &compensation);
    }
    return nk_f64_add_rn_rocm_(sum, compensation);
}

NUMKONG_DEVICE nk_f64_t nk_f32_lane_sumsq_rocm_(unsigned char const *row, nk_size_t depth, unsigned lane) {
    nk_f64_t sum = 0, compensation = 0;
    for (nk_size_t index = lane; index < depth; index += 32) {
        nk_u32_t bits = 0;
        for (unsigned byte = 0; byte < 4; ++byte) bits |= (nk_u32_t)row[index * 4 + byte] << (byte * 8);
        nk_f64_t const value = (nk_f64_t)__uint_as_float(bits);
        nk_cross_step_f64_rocm_(nk_cross_accumulation_dot2_k, value, value, &sum, &compensation);
    }
    return nk_f64_add_rn_rocm_(sum, compensation);
}

/** A column's norm from each of 32 lanes' F64 shares, merged through TwoSum and rounded once. */
NUMKONG_DEVICE nk_f64_t nk_cross_pack_norm_f64_rocm_(nk_f64_t share) {
    nk_f64_t compensation = 0;
    for (unsigned offset = 16; offset != 0; offset >>= 1)
        nk_cross_merge_lanes_f64_rocm_(nk_cross_accumulation_dot2_k, offset, &share, &compensation);
    return nk_f64_add_rn_rocm_(share, compensation);
}

/** A column's F32 norm from each of 32 lanes' F64 shares, summed in F64 and rounded once. */
NUMKONG_DEVICE nk_f32_t nk_cross_pack_norm_f32_rocm_(nk_f64_t share) {
    return (nk_f32_t)nk_cross_pack_norm_f64_rocm_(share);
}

/** A column's U32 norm from each of 32 lanes' exact shares, truncated like the serial backends'. */
NUMKONG_DEVICE nk_u32_t nk_cross_pack_norm_u32_rocm_(nk_u64_t share) {
    for (unsigned offset = 16; offset != 0; offset >>= 1) share += nk_shuffle_xor_u64_rocm_(share, offset);
    return (nk_u32_t)share;
}

#pragma endregion Pack Norms

#pragma region Launchers

/** Validates the contract and launches as many blocks of @p kernel as stay resident, each walking
 *  @p tile × @p tile output tiles with a stride of the grid. @p b_norms holds the packed column
 *  norms a @c packed metric reads, or is null. @p block_size is the block of a block-scaled dtype,
 *  whose @p depth it must divide and whose operands must carry scales, or zero for plain dtypes.
 *  Codes need 16-byte rows, while scales may sit at any byte, as dense rows of them do. */
NUMKONG_INLINE nk_status_t nk_cross_launch_rocm_(void const *kernel, unsigned tile, unsigned threads,
                                                 nk_cross_operand_t const *a, nk_cross_operand_t const *b,
                                                 void const *b_norms, void *c, nk_size_t result_bytes,
                                                 nk_size_t row_start, nk_size_t row_end, nk_size_t column_count,
                                                 nk_size_t depth, nk_size_t block_size, nk_size_t depth_bytes,
                                                 nk_size_t a_stride, nk_size_t b_stride, nk_size_t c_stride,
                                                 void *stream) {
    if (block_size && (depth % block_size || !a->scales || !b->scales)) return nk_unexpected_dimensions_k;
    if ((((nk_size_t)a->elements) | a_stride | ((nk_size_t)b->elements) | b_stride) & 15 ||
        (((nk_size_t)c) | c_stride) & (result_bytes - 1))
        return nk_misaligned_k;
    if (row_end <= row_start || column_count == 0) return nk_success_k;
    nk_size_t const column_tiles = nk_size_divide_round_up_(column_count, tile);
    nk_size_t const tiles = nk_size_divide_round_up_(row_end - row_start, tile) * column_tiles;
    nk_cross_tile_arguments_t arguments;
    arguments.a = (unsigned char const *)a->elements, arguments.b = (unsigned char const *)b->elements;
    arguments.c = c;
    arguments.row_start = row_start, arguments.row_end = row_end, arguments.column_count = column_count;
    arguments.depth = depth, arguments.depth_bytes = depth_bytes, arguments.a_stride = a_stride;
    arguments.b_stride = b_stride;
    arguments.c_stride = c_stride, arguments.column_tiles = column_tiles, arguments.tiles = tiles;
    arguments.depth_slabs = nk_size_divide_round_up_(depth_bytes, 64);
    arguments.b_norms = b_norms;
    arguments.a_scales = a->scales, arguments.b_scales = b->scales;
    arguments.a_scales_stride = a->scales_stride, arguments.b_scales_stride = b->scales_stride;
    arguments.a_tensor_scale = a->tensor_scale, arguments.b_tensor_scale = b->tensor_scale;
    return nk_launch_resident_rocm_(kernel, threads, 0, 0, tiles, &arguments, stream);
}

/** Launches @p kernel with one 32-lane group per packed column, walked with a grid stride, which
 *  records the packing @p capability and the tensor scale of @p b. */
NUMKONG_INLINE nk_status_t nk_cross_pack_launch_rocm_(void const *kernel, nk_cross_operand_t const *b,
                                                      nk_size_t column_count, nk_size_t depth, nk_size_t depth_bytes,
                                                      nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                      nk_size_t columns_end, nk_size_t depth_values_padded,
                                                      nk_capability_t capability, nk_size_t scales_stride,
                                                      void *stream) {
    nk_size_t const columns = columns_end > columns_begin ? columns_end - columns_begin : 0;
    nk_size_t const needed = nk_size_divide_round_up_(columns, nk_cross_pack_groups_k);
    nk_size_t const blocks = needed == 0 ? 1 : needed < 65535 ? needed : 65535;
    void const *elements = b->elements;
    nk_u8_t const *scale_values = b->scales;
    nk_size_t scale_values_stride = b->scales_stride;
    nk_f32_t const *tensor_scale = b->tensor_scale;
    void *arguments[14];
    arguments[0] = &elements, arguments[1] = &column_count, arguments[2] = &depth, arguments[3] = &depth_bytes;
    arguments[4] = &b_stride, arguments[5] = &b_packed, arguments[6] = &columns_begin, arguments[7] = &columns_end;
    arguments[8] = &depth_values_padded, arguments[9] = &capability, arguments[10] = &scale_values;
    arguments[11] = &scale_values_stride, arguments[12] = &tensor_scale, arguments[13] = &scales_stride;
    return nk_launch_rocm_(kernel, blocks, nk_cross_pack_groups_k * 32, arguments, 0, stream);
}

#pragma endregion Launchers

#pragma region Cross Macros

/**
 *  @brief Generates a pack into the serial layout on the device: its size, the shape accessor
 *      copying a device-resident packed buffer's header back, its kernel and the launching entry.
 *
 *  The kernel gives one 32-lane group to each column. The header is written only when
 *  @p columns_begin is zero, and each packed column gets its row and its norm, so disjoint column
 *  ranges may be packed by separate calls, exactly as with @c nk_define_cross_pack_.
 *
 *  @param[in] load_fn Device map from an input byte to its packed byte, the identity unless the
 *      multiply wants a remap.
 *  @param[in] norm_value_type The serial backends' norm type: @c f64, @c f32 or @c u32.
 *  @param[in] compute_norm_fn Device share of a column's sum of squares for one lane of 32, which
 *      @c nk_cross_pack_norm_f64_rocm_, @c nk_cross_pack_norm_f32_rocm_ or
 *      @c nk_cross_pack_norm_u32_rocm_ merges.
 *  @sa nk_define_cross_pack_ and nk_define_cross_packed_shape_ for the host originals.
 */
#define nk_define_cross_pack_rocm_(input_type_name, isa_suffix, input_value_type, packed_value_type, load_fn,          \
                                   norm_value_type, compute_norm_fn, depth_simd_dimensions, dimensions_per_value)      \
    nk_define_cross_pack_size_simt_(input_type_name, isa_suffix, packed_value_type, norm_value_type,                   \
                                    depth_simd_dimensions, dimensions_per_value)                                       \
    NUMKONG_API nk_status_t nk_dots_packed_shape_##input_type_name##_##isa_suffix(                                     \
        void const *b_packed, nk_size_t *columns, nk_size_t *depth, void *stream) {                                    \
        if ((nk_size_t)b_packed & 15) return nk_misaligned_k;                                                          \
        nk_cross_packed_buffer_header_t header;                                                                        \
        nk_status_t const status = nk_read_rocm_(&header, b_packed, sizeof(header), stream);                           \
        if (status != nk_success_k) return status;                                                                     \
        if (header.capability != nk_cap_##isa_suffix##_k) return nk_pack_mismatch_k;                                   \
        *columns = header.column_count, *depth = header.depth_dimensions;                                              \
        return nk_success_k;                                                                                           \
    }                                                                                                                  \
    static __global__ void nk_dots_pack_##input_type_name##_##isa_suffix##_kernel_(                                    \
        unsigned char const *b, nk_size_t column_count, nk_size_t depth, nk_size_t depth_bytes, nk_size_t b_stride,    \
        unsigned char *b_packed, nk_size_t columns_begin, nk_size_t columns_end, nk_size_t depth_values_padded,        \
        nk_capability_t capability, unsigned char const *b_scales, nk_size_t b_scales_stride,                          \
        nk_f32_t const *b_tensor_scale, nk_size_t scales_stride) {                                                     \
        nk_size_t const row_bytes = depth_values_padded * sizeof(nk_##packed_value_type##_t);                          \
        nk_f32_t const tensor_scale = b_tensor_scale ? *b_tensor_scale : 1;                                            \
        if (columns_begin == 0 && blockIdx.x == 0 && threadIdx.x == 0) {                                               \
            nk_cross_packed_buffer_header_t *header = (nk_cross_packed_buffer_header_t *)b_packed;                     \
            header->column_count = (nk_u32_t)column_count;                                                             \
            header->depth_dimensions = (nk_u32_t)depth;                                                                \
            header->depth_padded_values = (nk_u32_t)depth_values_padded;                                               \
            header->scales_stride = (nk_u32_t)scales_stride;                                                           \
            header->tensor_scale = tensor_scale;                                                                       \
            header->capability = capability;                                                                           \
            for (unsigned reserved_index = 0; reserved_index < 9; ++reserved_index)                                    \
                header->reserved[reserved_index] = 0;                                                                  \
        }                                                                                                              \
        unsigned char *rows = b_packed + sizeof(nk_cross_packed_buffer_header_t);                                      \
        unsigned char *scales = rows + column_count * row_bytes;                                                       \
        nk_##norm_value_type##_t *norms = (nk_##norm_value_type##_t *)(scales + column_count * scales_stride);         \
        unsigned const lane = threadIdx.x & 31;                                                                        \
        nk_size_t const groups = (nk_size_t)gridDim.x * (blockDim.x >> 5);                                             \
        for (nk_size_t column = columns_begin + (nk_size_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);        \
             column < columns_end; column += groups) {                                                                 \
            unsigned char const *source = b + column * b_stride;                                                       \
            unsigned char *destination = rows + column * row_bytes;                                                    \
            for (nk_size_t byte = lane; byte < row_bytes; byte += 32)                                                  \
                destination[byte] = byte < depth_bytes ? load_fn(source[byte]) : 0;                                    \
            unsigned char const *source_scales = b_scales + column * b_scales_stride;                                  \
            for (nk_size_t byte = lane; byte < scales_stride; byte += 32)                                              \
                scales[column * scales_stride + byte] = byte < nk_cross_scale_blocks_(nk_##input_type_name##_k, depth) \
                                                            ? source_scales[byte]                                      \
                                                            : 0;                                                       \
            nk_##norm_value_type##_t const norm = nk_cross_pack_norm_##norm_value_type##_rocm_(                        \
                scales_stride ? nk_cross_scaled_sumsq_simt_(nk_##input_type_name##_k, source, source_scales,           \
                                                            tensor_scale, depth, lane, 32)                             \
                              : compute_norm_fn(source, depth, lane));                                                 \
            if (lane == 0) norms[column] = norm;                                                                       \
        }                                                                                                              \
    }                                                                                                                  \
    NUMKONG_API nk_status_t nk_dots_pack_##input_type_name##_##isa_suffix(                                             \
        nk_cross_##input_type_name##_operand_t const *b_operand, nk_size_t column_count, nk_size_t depth,              \
        nk_size_t b_stride, void *b_packed, nk_size_t columns_begin, nk_size_t columns_end, void *stream) {            \
        nk_cross_operand_t const b = nk_cross_operand_(nk_##input_type_name##_k, b_operand, b_stride);                 \
        nk_size_t const depth_values_padded = nk_cross_padded_values_simt_(                                            \
            depth, depth_simd_dimensions, dimensions_per_value, sizeof(nk_##packed_value_type##_t));                   \
        nk_size_t const depth_bytes = depth / dimensions_per_value * sizeof(nk_##input_value_type##_t);                \
        nk_size_t const scales_stride = nk_cross_scales_stride_(nk_##input_type_name##_k, depth);                      \
        if (!nk_cross_whole_blocks_(nk_##input_type_name##_k, depth) || (scales_stride && !b.scales))                  \
            return nk_unexpected_dimensions_k;                                                                         \
        return nk_cross_pack_launch_rocm_((void const *)nk_dots_pack_##input_type_name##_##isa_suffix##_kernel_, &b,   \
                                          column_count, depth, depth_bytes, b_stride, b_packed, columns_begin,         \
                                          columns_end, depth_values_padded, nk_cap_##isa_suffix##_k, scales_stride,    \
                                          stream);                                                                     \
    }

/**
 *  @brief Generates both shapes of one metric on @p tile: C = A × Bᵀ, or its angular or euclidean
 *      distances, over a B packed by @c nk_define_cross_pack_rocm_, and the Gram matrix C = A × Aᵀ,
 *      or its distances, over rows [row_start, row_start + row_count), each with its kernel.
 *  @param[in] metric @c dot, @c angular or @c euclidean, naming both the entry and the epilogue.
 *  @param[in] tile The tile stem, like @c b32_rocm, naming its function and launch shape.
 *  @param[in] ... The tile's own leading arguments, which precede the triangle, the metric and
 *      the launch arguments. The symmetric kernel skips the tiles wholly below its triangle.
 *  @sa nk_define_cross_packed_ and nk_define_cross_symmetric_ for the host originals.
 */
#define nk_define_cross_rocm_(metric, input_type_name, isa_suffix, tile, input_value_type, packed_value_type,         \
                              result_value_type, depth_simd_dimensions, dimensions_per_value, ...)                    \
    static __global__ void __launch_bounds__(nk_cross_threads_##tile##_k)                                             \
        nk_##metric##s_packed_##input_type_name##_##isa_suffix##_kernel_(nk_cross_tile_arguments_t arguments) {       \
        nk_cross_tile_##tile##_(__VA_ARGS__, nk_cross_triangle_full_k, nk_cross_metric_##metric##_k, &arguments);     \
    }                                                                                                                 \
    NUMKONG_API nk_status_t nk_##metric##s_packed_##input_type_name##_##isa_suffix(                                   \
        nk_cross_##input_type_name##_operand_t const *a_operand, void const *b_packed_buffer,                         \
        nk_##result_value_type##_t *c_matrix, nk_size_t row_count, nk_size_t column_count, nk_size_t depth,           \
        nk_size_t a_stride, nk_size_t c_stride, void *stream) {                                                       \
        nk_size_t const row_bytes = nk_cross_padded_values_simt_(depth, depth_simd_dimensions, dimensions_per_value,  \
                                                                 sizeof(nk_##packed_value_type##_t)) *                \
                                    sizeof(nk_##packed_value_type##_t);                                               \
        nk_size_t const scales_stride = nk_cross_scales_stride_(nk_##input_type_name##_k, depth);                     \
        nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed_buffer;     \
        nk_u8_t const *b_rows = (nk_u8_t const *)(header + 1);                                                        \
        nk_cross_operand_t const a = nk_cross_operand_(nk_##input_type_name##_k, a_operand, a_stride);                \
        nk_cross_operand_t const b = {b_rows, scales_stride ? b_rows + column_count * row_bytes : NUMKONG_NULL,       \
                                      scales_stride, &header->tensor_scale};                                          \
        return nk_cross_launch_rocm_(                                                                                 \
            (void const *)nk_##metric##s_packed_##input_type_name##_##isa_suffix##_kernel_, nk_cross_tile_##tile##_k, \
            nk_cross_threads_##tile##_k, &a, &b, b_rows + column_count * (row_bytes + scales_stride), c_matrix,       \
            sizeof(nk_##result_value_type##_t), 0, row_count, column_count, depth,                                    \
            nk_block_scaled_format_of_dtype(nk_##input_type_name##_k).block_size,                                     \
            depth / dimensions_per_value * sizeof(nk_##input_value_type##_t), a_stride, row_bytes, c_stride, stream); \
    }                                                                                                                 \
    static __global__ void __launch_bounds__(nk_cross_threads_##tile##_k)                                             \
        nk_##metric##s_symmetric_##input_type_name##_##isa_suffix##_kernel_(nk_cross_tile_arguments_t arguments) {    \
        nk_cross_tile_##tile##_(__VA_ARGS__, nk_cross_triangle_upper_k, nk_cross_metric_##metric##_k, &arguments);    \
    }                                                                                                                 \
    NUMKONG_API nk_status_t nk_##metric##s_symmetric_##input_type_name##_##isa_suffix(                                \
        nk_cross_##input_type_name##_operand_t const *vectors_operand, nk_size_t vectors_count, nk_size_t depth,      \
        nk_size_t stride, nk_##result_value_type##_t *result, nk_size_t result_stride, nk_size_t row_start,           \
        nk_size_t row_count, void *stream) {                                                                          \
        nk_size_t const row_end = row_start + row_count < vectors_count ? row_start + row_count : vectors_count;      \
        nk_cross_operand_t const vectors = nk_cross_operand_(nk_##input_type_name##_k, vectors_operand, stride);      \
        return nk_cross_launch_rocm_(                                                                                 \
            (void const *)nk_##metric##s_symmetric_##input_type_name##_##isa_suffix##_kernel_,                        \
            nk_cross_tile_##tile##_k, nk_cross_threads_##tile##_k, &vectors, &vectors, 0, result,                     \
            sizeof(nk_##result_value_type##_t), row_start, row_end, vectors_count, depth,                             \
            nk_block_scaled_format_of_dtype(nk_##input_type_name##_k).block_size,                                     \
            depth / dimensions_per_value * sizeof(nk_##input_value_type##_t), stride, stride, result_stride, stream); \
    }

#pragma endregion Cross Macros

#pragma region Baseline Kernels

#if NUMKONG_TARGET_ROCM
nk_define_cross_pack_rocm_(f64, rocm, f64, f64, nk_load_b8_, f64, nk_f64_lane_sumsq_rocm_, 2, 1)
nk_define_cross_rocm_(dot, f64, rocm, f64_rocm, f64, f64, f64, 2, 1, nk_f64_k, nk_cross_accumulation_dot2_k)
nk_define_cross_pack_rocm_(f32, rocm, f32, f32, nk_load_b8_, f64, nk_f32_lane_sumsq_rocm_, 4, 1)
nk_define_cross_rocm_(dot, f32, rocm, f64_rocm, f32, f32, f64, 4, 1, nk_f32_k, nk_cross_accumulation_f64_k)
nk_define_cross_pack_rocm_(bf16, rocm, bf16, bf16, nk_load_b8_, f32, nk_bf16_lane_sumsq_, 8, 1)
nk_define_cross_rocm_(dot, bf16, rocm, b32_rocm, bf16, bf16, f32, 8, 1, nk_bf16_k, nk_cross_accumulation_f32_k)
nk_define_cross_pack_rocm_(f16, rocm, f16, f16, nk_load_b8_, f32, nk_f16_lane_sumsq_, 8, 1)
nk_define_cross_rocm_(dot, f16, rocm, b32_rocm, f16, f16, f32, 8, 1, nk_f16_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_pack_rocm_(e5m2, rocm, e5m2, e5m2, nk_load_b8_, f32, nk_e5m2_lane_sumsq_, 16, 1)
nk_define_cross_rocm_(dot, e5m2, rocm, b32_rocm, e5m2, e5m2, f32, 16, 1, nk_e5m2_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_pack_rocm_(e4m3, rocm, e4m3, e4m3, nk_load_b8_, f32, nk_e4m3_lane_sumsq_, 16, 1)
nk_define_cross_rocm_(dot, e4m3, rocm, b32_rocm, e4m3, e4m3, f32, 16, 1, nk_e4m3_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_pack_rocm_(e3m2, rocm, e3m2, e3m2, nk_load_b8_, f32, nk_e3m2_lane_sumsq_, 16, 1)
nk_define_cross_rocm_(dot, e3m2, rocm, b32_rocm, e3m2, e3m2, f32, 16, 1, nk_e3m2_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_pack_rocm_(e2m3, rocm, e2m3, e2m3, nk_load_b8_, f32, nk_e2m3_lane_sumsq_, 16, 1)
nk_define_cross_rocm_(dot, e2m3, rocm, b32_rocm, e2m3, e2m3, f32, 16, 1, nk_e2m3_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_pack_rocm_(e2m1, rocm, e2m1x2, e2m1x2, nk_load_b8_, f32, nk_e2m1_lane_sumsq_, 32, 2)
nk_define_cross_rocm_(dot, e2m1, rocm, b32_rocm, e2m1x2, e2m1x2, f32, 32, 2, nk_e2m1_k, nk_cross_accumulation_f16x2_k)
nk_define_cross_pack_rocm_(i8, rocm, i8, i8, nk_load_b8_, u32, nk_i8_lane_sumsq_, 16, 1)
nk_define_cross_rocm_(dot, i8, rocm, b32_rocm, i8, i8, i32, 16, 1, nk_i8_k, nk_cross_accumulation_i8x4_k)
nk_define_cross_pack_rocm_(u8, rocm, u8, u8, nk_load_b8_, u32, nk_u8_lane_sumsq_, 16, 1)
nk_define_cross_rocm_(dot, u8, rocm, b32_rocm, u8, u8, u32, 16, 1, nk_u8_k, nk_cross_accumulation_u8x4_k)
nk_define_cross_pack_rocm_(i4, rocm, i4x2, i4x2, nk_load_b8_, u32, nk_i4_lane_sumsq_, 32, 2)
nk_define_cross_rocm_(dot, i4, rocm, b32_rocm, i4x2, i4x2, i32, 32, 2, nk_i4_k, nk_cross_accumulation_i4x8_k)
nk_define_cross_pack_rocm_(u4, rocm, u4x2, u4x2, nk_load_b8_, u32, nk_u4_lane_sumsq_, 32, 2)
nk_define_cross_rocm_(dot, u4, rocm, b32_rocm, u4x2, u4x2, u32, 32, 2, nk_u4_k, nk_cross_accumulation_u4x8_k)
#endif // NUMKONG_TARGET_ROCM

#pragma endregion Baseline Kernels

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_ROCM_
#endif // NUMKONG_DOTS_ROCM_CUH
