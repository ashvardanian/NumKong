/**
 *  @file include/numkong/dots/cuda.cuh
 *  @author Ash Vardanian
 *  @date October 3, 2026
 *  @brief The @c cuda baseline of batched dot products: its F64, B32 and block-scaled tiles over
 *      the portable stages of `dots/simt.cuh`, the pack and tile launches, the generators every
 *      CUDA capability instantiates its exports with, and the baseline exports.
 *
 *  NVIDIA stages depth in F32 words for its FMA, and folds bytes and nibbles widened to bytes
 *  through @c dp4a, which every device since Pascal runs.
 *
 *  @sa include/numkong/dots/simt.cuh
 *  @sa include/numkong/dots/rocm.cuh
 */
#ifndef NUMKONG_DOTS_CUDA_CUH
#define NUMKONG_DOTS_CUDA_CUH

#include "numkong/cuda.cuh"
#include "numkong/dots/simt.cuh"

#if NUMKONG_ARCH_CUDA_

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Instructions

/** Adds the four signed byte products of @p a and @p b to @p sum with Pascal's @c dp4a . */
NUMKONG_DEVICE nk_i32_t nk_dot_i8x4_cuda_(nk_u32_t a, nk_u32_t b, nk_i32_t sum) { return __dp4a((int)a, (int)b, sum); }

/** Adds the four unsigned byte products of @p a and @p b to @p sum with Pascal's @c dp4a . */
NUMKONG_DEVICE nk_u32_t nk_dot_u8x4_cuda_(nk_u32_t a, nk_u32_t b, nk_u32_t sum) { return __dp4a(a, b, sum); }

#pragma endregion Instructions

/*  The baseline tiles over the portable stages: each launches the SIMT shape under its own stem, so
 *  the generators paste it, and folds, merges and rounds through NVIDIA's instructions. */
#pragma region Baseline Tiles

enum {
    nk_cross_threads_f64_cuda_k = nk_cross_threads_simt_k,
    nk_cross_tile_f64_cuda_k = nk_cross_tile_simt_k,
    nk_cross_threads_b32_cuda_k = nk_cross_threads_simt_k,
    nk_cross_tile_b32_cuda_k = nk_cross_tile_simt_k,
    nk_cross_threads_scaled_cuda_k = nk_cross_threads_simt_k,
    nk_cross_tile_scaled_cuda_k = nk_cross_tile_simt_k,
};

/** Adds a × b into a running sum: one F64 FMA, or Dot2's TwoProd and TwoSum with both errors kept
 *  apart. */
NUMKONG_DEVICE void nk_cross_step_f64_cuda_(nk_cross_accumulation_t accumulation, nk_f64_t a, nk_f64_t b, nk_f64_t *sum,
                                            nk_f64_t *compensation) {
    if (accumulation == nk_cross_accumulation_f64_k) {
        *sum = __fma_rn(a, b, *sum);
        return;
    }
    nk_f64_t const product = nk_f64_mul_rn_cuda_(a, b), product_error = __fma_rn(a, b, -product);
    nk_f64_t const total = nk_f64_add_rn_cuda_(*sum, product), virtual_addend = nk_f64_sub_rn_cuda_(total, *sum);
    nk_f64_t const augend_error = nk_f64_sub_rn_cuda_(*sum, nk_f64_sub_rn_cuda_(total, virtual_addend));
    nk_f64_t const sum_error = nk_f64_add_rn_cuda_(augend_error, nk_f64_sub_rn_cuda_(product, virtual_addend));
    *sum = total;
    *compensation = nk_f64_add_rn_cuda_(*compensation, nk_f64_add_rn_cuda_(sum_error, product_error));
}

/** Merges the running sum of lane `lane ^ offset` into this one, through TwoSum under Dot2. */
NUMKONG_DEVICE void nk_cross_merge_lanes_f64_cuda_(nk_cross_accumulation_t accumulation, unsigned offset, nk_f64_t *sum,
                                                   nk_f64_t *compensation) {
    nk_f64_t const other_sum = nk_shuffle_xor_f64_cuda_(*sum, offset);
    if (accumulation == nk_cross_accumulation_f64_k) {
        *sum = nk_f64_add_rn_cuda_(*sum, other_sum);
        return;
    }
    nk_f64_t const other_compensation = nk_shuffle_xor_f64_cuda_(*compensation, offset);
    nk_f64_t const total = nk_f64_add_rn_cuda_(*sum, other_sum), virtual_addend = nk_f64_sub_rn_cuda_(total, *sum);
    nk_f64_t const augend_error = nk_f64_sub_rn_cuda_(*sum, nk_f64_sub_rn_cuda_(total, virtual_addend));
    nk_f64_t const sum_error = nk_f64_add_rn_cuda_(augend_error, nk_f64_sub_rn_cuda_(other_sum, virtual_addend));
    *sum = total;
    *compensation = nk_f64_add_rn_cuda_(nk_f64_add_rn_cuda_(*compensation, other_compensation), sum_error);
}

/** Folds one staged slab into this thread's grid of F64 sums and their compensations. */
NUMKONG_DEVICE void nk_cross_fold_slab_f64_cuda_(
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
                nk_cross_step_f64_cuda_(accumulation, a_values[row_step], b_values[column_step],
                                        &sums[row_step][column_step], &compensations[row_step][column_step]);
    }
}

/**
 *  @brief The GEMM of one 64 × 64 output tile of F64 or F32 inputs, in F64 or Dot2.
 *  @param[in] dtype @c nk_f64_k or @c nk_f32_k, the rows' element type.
 *  @param[in] accumulation @c nk_cross_accumulation_dot2_k for F64 inputs and
 *      @c nk_cross_accumulation_f64_k for F32 ones; the norms always sum in Dot2.
 *
 *  Tensor-core F64 MMA never exposes a product's rounding error, which Dot2 needs, and outside the
 *  8.0 and 9.0 datacenter parts it runs no faster than these FMAs.
 */
NUMKONG_DEVICE void nk_cross_tile_f64_cuda_(nk_dtype_t dtype, nk_cross_accumulation_t accumulation,
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
                    nk_cross_step_f64_cuda_(nk_cross_accumulation_dot2_k, a_values[step], a_values[step],
                                            &a_norms[step][0], &a_norms[step][1]);
                    if (triangle == nk_cross_triangle_upper_k)
                        nk_cross_step_f64_cuda_(nk_cross_accumulation_dot2_k, b_values[step], b_values[step],
                                                &b_norms[step][0], &b_norms[step][1]);
                }
            __syncthreads();
            nk_cross_fold_slab_f64_cuda_(accumulation, a_slab, b_slab, sums, compensations);
            __syncthreads();
        }
        if (metric != nk_cross_metric_dot_k) {
            nk_f64_t a_rounded[nk_cross_loads_simt_k], b_rounded[nk_cross_loads_simt_k];
#pragma unroll
            for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step) {
#pragma unroll
                for (unsigned offset = nk_cross_slab_simt_k / 2; offset != 0; offset >>= 1) {
                    nk_cross_merge_lanes_f64_cuda_(nk_cross_accumulation_dot2_k, offset, &a_norms[step][0],
                                                   &a_norms[step][1]);
                    if (triangle == nk_cross_triangle_upper_k)
                        nk_cross_merge_lanes_f64_cuda_(nk_cross_accumulation_dot2_k, offset, &b_norms[step][0],
                                                       &b_norms[step][1]);
                }
                a_rounded[step] = nk_f64_add_rn_cuda_(a_norms[step][0], a_norms[step][1]);
                b_rounded[step] = nk_f64_add_rn_cuda_(b_norms[step][0], b_norms[step][1]);
            }
            nk_cross_publish_norms_f64_simt_(triangle, arguments, first_column, a_rounded, b_rounded, norms);
        }
#pragma unroll
        for (unsigned row_step = 0; row_step < nk_cross_thread_tile_simt_k; ++row_step)
#pragma unroll
            for (unsigned column_step = 0; column_step < nk_cross_thread_tile_simt_k; ++column_step)
                sums[row_step][column_step] = nk_f64_add_rn_cuda_(sums[row_step][column_step],
                                                                  compensations[row_step][column_step]);
        nk_cross_store_tile_f64_simt_(triangle, metric, arguments, first_row, first_column, sums, norms);
    }
}

/** Folds one 32-bit word of each operand into @p sum, by the instruction @p accumulation names: FMA
 *  for F32 words, @c dp4a for bytes and for nibbles widened to bytes. */
NUMKONG_DEVICE void nk_cross_fold_b32_cuda_(nk_cross_accumulation_t accumulation, nk_fui32_t *sum, nk_u32_t a,
                                            nk_u32_t b) {
    switch (accumulation) {
    case nk_cross_accumulation_f32_k: sum->f = __fmaf_rn(__uint_as_float(a), __uint_as_float(b), sum->f); break;
    case nk_cross_accumulation_i8x4_k:
    case nk_cross_accumulation_i4x4_k: sum->i = nk_dot_i8x4_cuda_(a, b, sum->i); break;
    default: sum->u = nk_dot_u8x4_cuda_(a, b, sum->u); break;
    }
}

/** Adds lane `lane ^ offset`'s B32 sum into @p sum: in F32 for floats, wrapping for integers. */
NUMKONG_DEVICE nk_fui32_t nk_cross_merge_lanes_b32_cuda_(nk_cross_accumulation_t accumulation, unsigned offset,
                                                         nk_fui32_t sum) {
    nk_fui32_t other;
    other.u = nk_shuffle_xor_u32_cuda_(sum.u, offset);
    if (accumulation == nk_cross_accumulation_f32_k) sum.f += other.f;
    else sum.u += other.u;
    return sum;
}

/** Folds one staged slab into this thread's grid of 32-bit sums. */
NUMKONG_DEVICE void nk_cross_fold_slab_b32_cuda_(
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
                nk_cross_fold_b32_cuda_(accumulation, &sums[row_step][column_step], a_words[row_step],
                                        b_words[column_step]);
    }
}

/**
 *  @brief The GEMM of one 64 × 64 output tile of 16-bit or narrower inputs, folding 32-bit words.
 *  @param[in] dtype The rows' element type, which a float word decodes from.
 *  @param[in] accumulation How each word holds depth and which instruction folds it; the norms fold
 *      the same way.
 */
NUMKONG_DEVICE void nk_cross_tile_b32_cuda_(nk_dtype_t dtype, nk_cross_accumulation_t accumulation,
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
                    nk_cross_fold_b32_cuda_(accumulation, &a_norms[step], a_words[step], a_words[step]);
                    if (triangle == nk_cross_triangle_upper_k)
                        nk_cross_fold_b32_cuda_(accumulation, &b_norms[step], b_words[step], b_words[step]);
                }
            __syncthreads();
            nk_cross_fold_slab_b32_cuda_(accumulation, a_slab, b_slab, sums);
            __syncthreads();
        }
        if (metric != nk_cross_metric_dot_k) {
#pragma unroll
            for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step)
#pragma unroll
                for (unsigned offset = nk_cross_slab_simt_k / 2; offset != 0; offset >>= 1) {
                    a_norms[step] = nk_cross_merge_lanes_b32_cuda_(accumulation, offset, a_norms[step]);
                    if (triangle == nk_cross_triangle_upper_k)
                        b_norms[step] = nk_cross_merge_lanes_b32_cuda_(accumulation, offset, b_norms[step]);
                }
            nk_cross_publish_norms_b32_simt_(triangle, arguments, first_column, a_norms, b_norms, norms);
        }
        nk_cross_store_tile_b32_simt_(accumulation, triangle, metric, arguments, first_row, first_column, sums, norms);
    }
}

/** The GEMM of a block-scaled @p dtype, each block walking 64 × 64 output tiles with a stride of
 *  the grid. Every 16-element slab lies inside one block, so its products gather in F32 partial
 *  sums that take both scales once the block ends. */
NUMKONG_DEVICE void nk_cross_tile_scaled_cuda_(nk_dtype_t dtype, nk_cross_triangle_t triangle, nk_cross_metric_t metric,
                                               nk_cross_tile_arguments_t const *arguments) {
    __shared__ nk_u32_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1];
    __shared__ nk_u32_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1];
    __shared__ nk_f32_t a_scales[nk_cross_tile_simt_k], b_scales[nk_cross_tile_simt_k];
    __shared__ nk_fui32_t norms[2][nk_cross_tile_simt_k];
    nk_block_scaled_format_t const format = nk_block_scaled_format_of_dtype(dtype);
    nk_cross_accumulation_t const accumulation = nk_cross_accumulation_f32_k;
    unsigned const thread_column = threadIdx.x % nk_cross_grid_side_simt_k;
    unsigned const thread_row = threadIdx.x / nk_cross_grid_side_simt_k;
    nk_f32_t const a_tensor_scale = arguments->a_tensor_scale ? *arguments->a_tensor_scale : 1;
    nk_f32_t const b_tensor_scale = arguments->b_tensor_scale ? *arguments->b_tensor_scale : 1;

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        nk_size_t const first_row = arguments->row_start + tile / arguments->column_tiles * nk_cross_tile_simt_k;
        nk_size_t const first_column = tile % arguments->column_tiles * nk_cross_tile_simt_k;
        if (triangle == nk_cross_triangle_upper_k && first_column + nk_cross_tile_simt_k <= first_row) continue;
        nk_fui32_t partials[nk_cross_thread_tile_simt_k][nk_cross_thread_tile_simt_k] = {{{0}}};
        nk_fui32_t totals[nk_cross_thread_tile_simt_k][nk_cross_thread_tile_simt_k] = {{{0}}};
        for (nk_size_t slab = 0; slab < arguments->depth; slab += nk_cross_slab_simt_k) {
            // The block norms come from the scaled rows below, so the staged words go unused.
            nk_u32_t a_words[nk_cross_loads_simt_k], b_words[nk_cross_loads_simt_k];
            nk_cross_stage_slab_b32_simt_(format.element_dtype, accumulation, arguments, first_row, first_column, slab,
                                          arguments->depth, a_slab, b_slab, a_words, b_words);
            nk_size_t const block = slab / format.block_size;
            if (threadIdx.x < nk_cross_tile_simt_k) {
                nk_cross_stage_scale_simt_(format, arguments->a_scales, arguments->a_scales_stride, a_tensor_scale,
                                           first_row, arguments->row_end, block, a_scales);
                nk_cross_stage_scale_simt_(format, arguments->b_scales, arguments->b_scales_stride, b_tensor_scale,
                                           first_column, arguments->column_count, block, b_scales);
            }
            __syncthreads();
            nk_cross_fold_slab_b32_cuda_(accumulation, a_slab, b_slab, partials);
            if ((slab + nk_cross_slab_simt_k) % format.block_size == 0 ||
                slab + nk_cross_slab_simt_k >= arguments->depth)
#pragma unroll
                for (unsigned row_step = 0; row_step < nk_cross_thread_tile_simt_k; ++row_step)
#pragma unroll
                    for (unsigned column_step = 0; column_step < nk_cross_thread_tile_simt_k; ++column_step) {
                        totals[row_step][column_step].f +=
                            partials[row_step][column_step].f *
                            a_scales[thread_row + nk_cross_grid_side_simt_k * row_step] *
                            b_scales[thread_column + nk_cross_grid_side_simt_k * column_step];
                        partials[row_step][column_step].f = 0;
                    }
            __syncthreads();
        }
        if (metric != nk_cross_metric_dot_k) {
            // The first barrier retires the previous tile's norm reads, the second publishes these.
            __syncthreads();
            if (threadIdx.x < nk_cross_tile_simt_k) {
                nk_size_t const row = first_row + threadIdx.x, column = first_column + threadIdx.x;
                norms[0][threadIdx.x].f = row < arguments->row_end
                                              ? (nk_f32_t)nk_cross_scaled_sumsq_simt_(
                                                    dtype, arguments->a + row * arguments->a_stride,
                                                    arguments->a_scales + row * arguments->a_scales_stride,
                                                    a_tensor_scale, arguments->depth, 0, 1)
                                              : 0;
                if (column >= arguments->column_count) norms[1][threadIdx.x].f = 0;
                else if (triangle == nk_cross_triangle_upper_k)
                    norms[1][threadIdx.x].f = (nk_f32_t)nk_cross_scaled_sumsq_simt_(
                        dtype, arguments->b + column * arguments->b_stride,
                        arguments->b_scales + column * arguments->b_scales_stride, b_tensor_scale, arguments->depth, 0,
                        1);
                else norms[1][threadIdx.x].f = ((nk_f32_t const *)arguments->b_norms)[column];
            }
            __syncthreads();
        }
        nk_cross_store_tile_b32_simt_(accumulation, triangle, metric, arguments, first_row, first_column, totals,
                                      norms);
    }
}

#pragma endregion Baseline Tiles

/*  A lane's Dot2-compensated share of an F64 or F32 column's sum of squares, and the merges of 32
 *  lanes' shares into one packed norm, which the pack generators call for every CUDA capability. */
#pragma region Pack Norms

NUMKONG_DEVICE nk_f64_t nk_f64_lane_sumsq_cuda_(unsigned char const *row, nk_size_t depth, unsigned lane) {
    nk_f64_t sum = 0, compensation = 0;
    for (nk_size_t index = lane; index < depth; index += 32) {
        unsigned long long bits = 0;
        for (unsigned byte = 0; byte < 8; ++byte) bits |= (unsigned long long)row[index * 8 + byte] << (byte * 8);
        nk_f64_t const value = __longlong_as_double((long long)bits);
        nk_cross_step_f64_cuda_(nk_cross_accumulation_dot2_k, value, value, &sum, &compensation);
    }
    return nk_f64_add_rn_cuda_(sum, compensation);
}

NUMKONG_DEVICE nk_f64_t nk_f32_lane_sumsq_cuda_(unsigned char const *row, nk_size_t depth, unsigned lane) {
    nk_f64_t sum = 0, compensation = 0;
    for (nk_size_t index = lane; index < depth; index += 32) {
        nk_u32_t bits = 0;
        for (unsigned byte = 0; byte < 4; ++byte) bits |= (nk_u32_t)row[index * 4 + byte] << (byte * 8);
        nk_f64_t const value = (nk_f64_t)__uint_as_float(bits);
        nk_cross_step_f64_cuda_(nk_cross_accumulation_dot2_k, value, value, &sum, &compensation);
    }
    return nk_f64_add_rn_cuda_(sum, compensation);
}

/** A column's norm from each of 32 lanes' F64 shares, merged through TwoSum and rounded once. */
NUMKONG_DEVICE nk_f64_t nk_cross_pack_norm_f64_cuda_(nk_f64_t share) {
    nk_f64_t compensation = 0;
    for (unsigned offset = 16; offset != 0; offset >>= 1)
        nk_cross_merge_lanes_f64_cuda_(nk_cross_accumulation_dot2_k, offset, &share, &compensation);
    return nk_f64_add_rn_cuda_(share, compensation);
}

/** A column's F32 norm from each of 32 lanes' F64 shares, summed in F64 and rounded once. */
NUMKONG_DEVICE nk_f32_t nk_cross_pack_norm_f32_cuda_(nk_f64_t share) {
    return (nk_f32_t)nk_cross_pack_norm_f64_cuda_(share);
}

/** A column's U32 norm from each of 32 lanes' exact shares, truncated like the serial backends'. */
NUMKONG_DEVICE nk_u32_t nk_cross_pack_norm_u32_cuda_(nk_u64_t share) {
    for (unsigned offset = 16; offset != 0; offset >>= 1) share += nk_shuffle_xor_u64_cuda_(share, offset);
    return (nk_u32_t)share;
}

#pragma endregion Pack Norms

#pragma region Launchers

/** Validates the contract and launches as many blocks of @p kernel as stay resident, each walking
 *  @p tile × @p tile output tiles with a stride of the grid. @p b_norms holds the packed column
 *  norms a @c packed metric reads, or is null. @p block_size is the block of a block-scaled dtype,
 *  whose @p depth it must divide and whose operands must carry scales, or zero for plain dtypes.
 *  Codes need 16-byte rows, while scales may sit at any byte, as dense rows of them do. */
NUMKONG_INLINE nk_status_t nk_cross_launch_cuda_(void const *kernel, unsigned tile, unsigned threads,
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
    return nk_launch_resident_cuda_(kernel, threads, 0, 0, tiles, &arguments, stream);
}

/** Launches @p kernel with one 32-lane group per packed column, walked with a grid stride, which
 *  records the packing @p capability and the tensor scale of @p b. */
NUMKONG_INLINE nk_status_t nk_cross_pack_launch_cuda_(void const *kernel, nk_cross_operand_t const *b,
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
    return nk_launch_cuda_(kernel, blocks, nk_cross_pack_groups_k * 32, arguments, 0, stream);
}

#pragma endregion Launchers

#pragma region Cross Macros

/**
 *  @brief Generates a packed-shape accessor copying a device-resident packed buffer's header back.
 *  @sa nk_define_cross_packed_shape_ for the host-resident original.
 */
#define nk_define_cross_packed_shape_cuda_(input_type_name, isa_suffix)                      \
    NUMKONG_API nk_status_t nk_dots_packed_shape_##input_type_name##_##isa_suffix(           \
        void const *b_packed, nk_size_t *columns, nk_size_t *depth, void *stream) {          \
        if ((nk_size_t)b_packed & 15) return nk_misaligned_k;                                \
        nk_cross_packed_buffer_header_t header;                                              \
        nk_status_t const status = nk_read_cuda_(&header, b_packed, sizeof(header), stream); \
        if (status != nk_success_k) return status;                                           \
        if (header.capability != nk_cap_##isa_suffix##_k) return nk_pack_mismatch_k;         \
        *columns = header.column_count, *depth = header.depth_dimensions;                    \
        return nk_success_k;                                                                 \
    }

/**
 *  @brief Generates a pack into the serial layout on the device: its kernel, which gives one
 *      32-lane group to each column, and the launching entry.
 *
 *  The header is written only when @p columns_begin is zero, and each packed column gets its row
 *  and its norm, so disjoint column ranges may be packed by separate calls, exactly as with
 *  @c nk_define_cross_pack_.
 *
 *  @param[in] load_fn Device map from an input byte to its packed byte, the identity unless the
 *      multiply wants a remap.
 *  @param[in] norm_value_type The serial backends' norm type: @c f64, @c f32 or @c u32.
 *  @param[in] compute_norm_fn Device share of a column's sum of squares for one lane of 32, which
 *      @c nk_cross_pack_norm_f64_cuda_, @c nk_cross_pack_norm_f32_cuda_ or
 *      @c nk_cross_pack_norm_u32_cuda_ merges.
 *  @sa nk_define_cross_pack_ for the host original.
 */
#define nk_define_cross_pack_rows_cuda_(input_type_name, isa_suffix, input_value_type, packed_value_type, load_fn,     \
                                        norm_value_type, compute_norm_fn, depth_simd_dimensions, dimensions_per_value) \
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
            nk_##norm_value_type##_t const norm = nk_cross_pack_norm_##norm_value_type##_cuda_(                        \
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
        return nk_cross_pack_launch_cuda_((void const *)nk_dots_pack_##input_type_name##_##isa_suffix##_kernel_, &b,   \
                                          column_count, depth, depth_bytes, b_stride, b_packed, columns_begin,         \
                                          columns_end, depth_values_padded, nk_cap_##isa_suffix##_k, scales_stride,    \
                                          stream);                                                                     \
    }

/** The pack's size, shape reader and kernel, which always ship together, with the norm share of
 *  the input type, @c nk_<input_type_name>_lane_sumsq_. */
#define nk_define_cross_pack_cuda_(input_type_name, isa_suffix, input_value_type, packed_value_type, load_fn,   \
                                   norm_value_type, depth_simd_dimensions, dimensions_per_value)                \
    nk_define_cross_pack_size_simt_(input_type_name, isa_suffix, packed_value_type, norm_value_type,            \
                                    depth_simd_dimensions, dimensions_per_value)                                \
    nk_define_cross_packed_shape_cuda_(input_type_name, isa_suffix)                                             \
    nk_define_cross_pack_rows_cuda_(input_type_name, isa_suffix, input_value_type, packed_value_type, load_fn,  \
                                    norm_value_type, nk_##input_type_name##_lane_sumsq_, depth_simd_dimensions, \
                                    dimensions_per_value)

/**
 *  @brief Generates C = A × Bᵀ, or its angular or euclidean distances, on @p tile over a B packed
 *      by @c nk_define_cross_pack_cuda_, with a kernel whose blocks stride the grid over tiles.
 *  @param[in] metric @c dot, @c angular or @c euclidean, naming both the entry and the epilogue.
 *  @param[in] tile The stem of the tile's function and launch shape, like @c b32_cuda or @c ampere.
 *  @param[in] ... The tile's own leading arguments, which precede the triangle, the metric
 *      and the launch arguments.
 *  @sa nk_define_cross_packed_ for the host original.
 */
#define nk_define_cross_packed_cuda_(metric, input_type_name, isa_suffix, tile, input_value_type, packed_value_type,  \
                                     result_value_type, depth_simd_dimensions, dimensions_per_value, ...)             \
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
        return nk_cross_launch_cuda_(                                                                                 \
            (void const *)nk_##metric##s_packed_##input_type_name##_##isa_suffix##_kernel_, nk_cross_tile_##tile##_k, \
            nk_cross_threads_##tile##_k, &a, &b, b_rows + column_count * (row_bytes + scales_stride), c_matrix,       \
            sizeof(nk_##result_value_type##_t), 0, row_count, column_count, depth,                                    \
            nk_block_scaled_format_of_dtype(nk_##input_type_name##_k).block_size,                                     \
            depth / dimensions_per_value * sizeof(nk_##input_value_type##_t), a_stride, row_bytes, c_stride, stream); \
    }

/**
 *  @brief Generates the Gram matrix C = A × Aᵀ, or its angular or euclidean distances, on @p tile
 *      over rows [row_start, row_start + row_count), with its kernel, which walks the upper
 *      triangle and skips tiles wholly below it.
 *
 *  Takes the parameters of @c nk_define_cross_packed_cuda_, so one bundle feeds both.
 *
 *  @sa nk_define_cross_symmetric_ for the host original.
 */
#define nk_define_cross_symmetric_cuda_(metric, input_type_name, isa_suffix, tile, input_value_type,                  \
                                        packed_value_type, result_value_type, depth_simd_dimensions,                  \
                                        dimensions_per_value, ...)                                                    \
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
        return nk_cross_launch_cuda_(                                                                                 \
            (void const *)nk_##metric##s_symmetric_##input_type_name##_##isa_suffix##_kernel_,                        \
            nk_cross_tile_##tile##_k, nk_cross_threads_##tile##_k, &vectors, &vectors, 0, result,                     \
            sizeof(nk_##result_value_type##_t), row_start, row_end, vectors_count, depth,                             \
            nk_block_scaled_format_of_dtype(nk_##input_type_name##_k).block_size,                                     \
            depth / dimensions_per_value * sizeof(nk_##input_value_type##_t), stride, stride, result_stride, stream); \
    }

/** Both shapes of one metric on @p tile, packed and symmetric. */
#define nk_define_cross_cuda_(metric, input_type_name, isa_suffix, tile, input_value_type, packed_value_type,       \
                              result_value_type, depth_simd_dimensions, dimensions_per_value, ...)                  \
    nk_define_cross_packed_cuda_(metric, input_type_name, isa_suffix, tile, input_value_type, packed_value_type,    \
                                 result_value_type, depth_simd_dimensions, dimensions_per_value, __VA_ARGS__)       \
    nk_define_cross_symmetric_cuda_(metric, input_type_name, isa_suffix, tile, input_value_type, packed_value_type, \
                                    result_value_type, depth_simd_dimensions, dimensions_per_value, __VA_ARGS__)

#pragma endregion Cross Macros

#pragma region Baseline Kernels

#if NUMKONG_TARGET_CUDA
nk_define_cross_pack_size_simt_(f64, cuda, f64, f64, 2, 1)
nk_define_cross_packed_shape_cuda_(f64, cuda)
nk_define_cross_pack_rows_cuda_(f64, cuda, f64, f64, nk_load_b8_, f64, nk_f64_lane_sumsq_cuda_, 2, 1)
nk_define_cross_cuda_(dot, f64, cuda, f64_cuda, f64, f64, f64, 2, 1, nk_f64_k, nk_cross_accumulation_dot2_k)
nk_define_cross_pack_size_simt_(f32, cuda, f32, f64, 4, 1)
nk_define_cross_packed_shape_cuda_(f32, cuda)
nk_define_cross_pack_rows_cuda_(f32, cuda, f32, f32, nk_load_b8_, f64, nk_f32_lane_sumsq_cuda_, 4, 1)
nk_define_cross_cuda_(dot, f32, cuda, f64_cuda, f32, f32, f64, 4, 1, nk_f32_k, nk_cross_accumulation_f64_k)
nk_define_cross_pack_cuda_(bf16, cuda, bf16, bf16, nk_load_b8_, f32, 8, 1)
nk_define_cross_cuda_(dot, bf16, cuda, b32_cuda, bf16, bf16, f32, 8, 1, nk_bf16_k, nk_cross_accumulation_f32_k)
nk_define_cross_pack_cuda_(f16, cuda, f16, f16, nk_load_b8_, f32, 8, 1)
nk_define_cross_cuda_(dot, f16, cuda, b32_cuda, f16, f16, f32, 8, 1, nk_f16_k, nk_cross_accumulation_f32_k)
nk_define_cross_pack_cuda_(e5m2, cuda, e5m2, e5m2, nk_load_b8_, f32, 16, 1)
nk_define_cross_cuda_(dot, e5m2, cuda, b32_cuda, e5m2, e5m2, f32, 16, 1, nk_e5m2_k, nk_cross_accumulation_f32_k)
nk_define_cross_pack_cuda_(e4m3, cuda, e4m3, e4m3, nk_load_b8_, f32, 16, 1)
nk_define_cross_cuda_(dot, e4m3, cuda, b32_cuda, e4m3, e4m3, f32, 16, 1, nk_e4m3_k, nk_cross_accumulation_f32_k)
nk_define_cross_pack_cuda_(e3m2, cuda, e3m2, e3m2, nk_load_b8_, f32, 16, 1)
nk_define_cross_cuda_(dot, e3m2, cuda, b32_cuda, e3m2, e3m2, f32, 16, 1, nk_e3m2_k, nk_cross_accumulation_f32_k)
nk_define_cross_pack_cuda_(e2m3, cuda, e2m3, e2m3, nk_load_b8_, f32, 16, 1)
nk_define_cross_cuda_(dot, e2m3, cuda, b32_cuda, e2m3, e2m3, f32, 16, 1, nk_e2m3_k, nk_cross_accumulation_f32_k)
nk_define_cross_pack_cuda_(e2m1, cuda, e2m1x2, e2m1x2, nk_load_b8_, f32, 32, 2)
nk_define_cross_cuda_(dot, e2m1, cuda, b32_cuda, e2m1x2, e2m1x2, f32, 32, 2, nk_e2m1_k, nk_cross_accumulation_f32_k)
nk_define_cross_pack_cuda_(i8, cuda, i8, i8, nk_load_b8_, u32, 16, 1)
nk_define_cross_cuda_(dot, i8, cuda, b32_cuda, i8, i8, i32, 16, 1, nk_i8_k, nk_cross_accumulation_i8x4_k)
nk_define_cross_pack_cuda_(u8, cuda, u8, u8, nk_load_b8_, u32, 16, 1)
nk_define_cross_cuda_(dot, u8, cuda, b32_cuda, u8, u8, u32, 16, 1, nk_u8_k, nk_cross_accumulation_u8x4_k)
nk_define_cross_pack_cuda_(i4, cuda, i4x2, i4x2, nk_load_b8_, u32, 32, 2)
nk_define_cross_cuda_(dot, i4, cuda, b32_cuda, i4x2, i4x2, i32, 32, 2, nk_i4_k, nk_cross_accumulation_i4x4_k)
nk_define_cross_pack_cuda_(u4, cuda, u4x2, u4x2, nk_load_b8_, u32, 32, 2)
nk_define_cross_cuda_(dot, u4, cuda, b32_cuda, u4x2, u4x2, u32, 32, 2, nk_u4_k, nk_cross_accumulation_u4x4_k)
nk_define_cross_pack_size_simt_(nvfp4, cuda, e2m1x2, f32, 32, 2)
nk_define_cross_packed_shape_cuda_(nvfp4, cuda)
nk_define_cross_pack_rows_cuda_(nvfp4, cuda, e2m1x2, e2m1x2, nk_load_b8_, f32, nk_e2m1_lane_sumsq_, 32, 2)
nk_define_cross_cuda_(dot, nvfp4, cuda, scaled_cuda, e2m1x2, e2m1x2, f32, 32, 2, nk_nvfp4_k)
nk_define_cross_pack_size_simt_(mxfp4, cuda, e2m1x2, f32, 32, 2)
nk_define_cross_packed_shape_cuda_(mxfp4, cuda)
nk_define_cross_pack_rows_cuda_(mxfp4, cuda, e2m1x2, e2m1x2, nk_load_b8_, f32, nk_e2m1_lane_sumsq_, 32, 2)
nk_define_cross_cuda_(dot, mxfp4, cuda, scaled_cuda, e2m1x2, e2m1x2, f32, 32, 2, nk_mxfp4_k)
nk_define_cross_pack_size_simt_(mxfp8e4m3, cuda, e4m3, f32, 16, 1)
nk_define_cross_packed_shape_cuda_(mxfp8e4m3, cuda)
nk_define_cross_pack_rows_cuda_(mxfp8e4m3, cuda, e4m3, e4m3, nk_load_b8_, f32, nk_e4m3_lane_sumsq_, 16, 1)
nk_define_cross_cuda_(dot, mxfp8e4m3, cuda, scaled_cuda, e4m3, e4m3, f32, 16, 1, nk_mxfp8e4m3_k)
nk_define_cross_pack_size_simt_(mxfp8e5m2, cuda, e5m2, f32, 16, 1)
nk_define_cross_packed_shape_cuda_(mxfp8e5m2, cuda)
nk_define_cross_pack_rows_cuda_(mxfp8e5m2, cuda, e5m2, e5m2, nk_load_b8_, f32, nk_e5m2_lane_sumsq_, 16, 1)
nk_define_cross_cuda_(dot, mxfp8e5m2, cuda, scaled_cuda, e5m2, e5m2, f32, 16, 1, nk_mxfp8e5m2_k)
#endif // NUMKONG_TARGET_CUDA

#pragma endregion Baseline Kernels

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_
#endif // NUMKONG_DOTS_CUDA_CUH
