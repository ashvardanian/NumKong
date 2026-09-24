/**
 *  @file include/numkong/spatials/ampere.cuh
 *  @author Ash Vardanian
 *  @date September 22, 2026
 *  @brief SIMD-accelerated batched spatial distances for NVIDIA Ampere and newer.
 *
 *  @sa include/numkong/spatials.h
 *  @sa include/numkong/dots/ampere.cuh
 *
 *  The dots tile with a metric in its epilogue: squared norms of A, and for @c symmetric of the
 *  column vectors too, accumulate from the staged slabs while the products do, and @c packed reads
 *  the column norms its pack stored. Integer codes square exactly through @c dp4a, E2M3 and E2M1
 *  included; the other narrow floats sum each slab apart in F32 before adding it to the running F32
 *  sum, and F32 and F64 square in F64 on the CUDA cores. Output precision, zero-norm handling and
 *  the triangle @c symmetric writes follow the serial backends: F64 for F64 and F32 inputs, F32
 *  otherwise, and I8 and I4 norms read as I32.
 */
#ifndef NK_SPATIALS_AMPERE_CUH
#define NK_SPATIALS_AMPERE_CUH

#if NK_TARGET_AMPERE

#include "numkong/dots/ampere.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Cross Macros

/**
 *  @brief Generates angular or euclidean distances between A and a B packed by
 *      @c nk_define_cross_cuda_pack_.
 *  @param[in] norm How the pack stored the column norms, and the precision of the metric.
 *  @param[in] norm_update_fn Device fold of 16 staged bytes' squares, see
 *      @c nk_cross_norm_update_ampere_t.
 *  @param[in] norm_scale Undoes the power of two the norm update's widening introduced, or 1.
 *  @sa nk_define_cross_normalized_packed_ for the host original.
 */
#define nk_define_cross_cuda_normalized_packed_(                                                                       \
    metric_name, input_type_name, isa_suffix, input_value_type, packed_value_type, multiply_fn, epilogue,              \
    output_scale, norm, norm_update_fn, norm_scale, depth_simd_dimensions, dimensions_per_value)                       \
    static __global__ void __launch_bounds__(nk_cross_threads_ampere_k)                                                \
        nk_##metric_name##s_packed_##input_type_name##_##isa_suffix##_kernel_(                                         \
            nk_cross_tile_arguments_ampere_t arguments) {                                                              \
        nk_cross_tile_ampere_(multiply_fn, epilogue, output_scale, nk_cross_triangle_full_k,                           \
                              nk_cross_metric_##metric_name##_k, norm, norm_update_fn, norm_scale, &arguments);        \
    }                                                                                                                  \
    NK_API_COMPTIME cudaError_t nk_##metric_name##s_packed_##input_type_name##_##isa_suffix(                           \
        nk_##input_value_type##_t const *a_matrix, void const *b_packed_buffer, nk_f32_t *c_matrix,                    \
        nk_size_t row_count, nk_size_t column_count, nk_size_t depth, nk_size_t a_stride_in_bytes,                     \
        nk_size_t c_stride_in_bytes, cudaStream_t stream) {                                                            \
        nk_size_t const row_bytes = nk_cross_padded_values_ampere_(depth, depth_simd_dimensions, dimensions_per_value, \
                                                                   sizeof(nk_##packed_value_type##_t)) *               \
                                    sizeof(nk_##packed_value_type##_t);                                                \
        unsigned char const *b_rows = (unsigned char const *)b_packed_buffer +                                         \
                                      sizeof(nk_cross_packed_buffer_header_t);                                         \
        return nk_cross_launch_ampere_(                                                                                \
            (void const *)nk_##metric_name##s_packed_##input_type_name##_##isa_suffix##_kernel_,                       \
            nk_cross_tile_ampere_k, nk_cross_threads_ampere_k, a_matrix, b_rows, b_rows + column_count * row_bytes,    \
            c_matrix, sizeof(nk_f32_t), 0, row_count, column_count, depth,                                             \
            depth / dimensions_per_value * sizeof(nk_##input_value_type##_t), a_stride_in_bytes, row_bytes,            \
            c_stride_in_bytes, stream);                                                                                \
    }

/**
 *  @brief Generates angular or euclidean distances among rows [row_start, row_start + row_count)
 *      and every vector, writing above the diagonal, zeros on it, and nothing below it.
 *  @sa nk_define_cross_normalized_symmetric_ for the host original.
 */
#define nk_define_cross_cuda_normalized_symmetric_(metric_name, input_type_name, isa_suffix, input_value_type,         \
                                                   multiply_fn, epilogue, output_scale, norm, norm_update_fn,          \
                                                   norm_scale, dimensions_per_value)                                   \
    static __global__ void __launch_bounds__(nk_cross_threads_ampere_k)                                                \
        nk_##metric_name##s_symmetric_##input_type_name##_##isa_suffix##_kernel_(                                      \
            nk_cross_tile_arguments_ampere_t arguments) {                                                              \
        nk_cross_tile_ampere_(multiply_fn, epilogue, output_scale, nk_cross_triangle_upper_k,                          \
                              nk_cross_metric_##metric_name##_k, norm, norm_update_fn, norm_scale, &arguments);        \
    }                                                                                                                  \
    NK_API_COMPTIME cudaError_t nk_##metric_name##s_symmetric_##input_type_name##_##isa_suffix(                        \
        nk_##input_value_type##_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_in_bytes, \
        nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count,                  \
        cudaStream_t stream) {                                                                                         \
        nk_size_t const row_end = row_start + row_count < vectors_count ? row_start + row_count : vectors_count;       \
        return nk_cross_launch_ampere_(                                                                                \
            (void const *)nk_##metric_name##s_symmetric_##input_type_name##_##isa_suffix##_kernel_,                    \
            nk_cross_tile_ampere_k, nk_cross_threads_ampere_k, vectors, vectors, 0, result, sizeof(nk_f32_t),          \
            row_start, row_end, vectors_count, depth,                                                                  \
            depth / dimensions_per_value * sizeof(nk_##input_value_type##_t), stride_in_bytes, stride_in_bytes,        \
            result_stride_in_bytes, stream);                                                                           \
    }

/**
 *  @brief Generates angular or euclidean distances in F64 between A and a packed B on the CUDA
 *      cores, for the F32 and F64 inputs, with F64 column norms read from the pack.
 *  @sa nk_define_cross_normalized_packed_ for the host original.
 */
#define nk_define_cross_cuda_fma_normalized_packed_(metric_name, input_type_name, isa_suffix, input_value_type,        \
                                                    packed_value_type, load_fn, accumulation, depth_simd_dimensions,   \
                                                    dimensions_per_value)                                              \
    static __global__ void __launch_bounds__(nk_cross_fma_threads_ampere_k)                                            \
        nk_##metric_name##s_packed_##input_type_name##_##isa_suffix##_kernel_(                                         \
            nk_cross_tile_arguments_ampere_t arguments) {                                                              \
        nk_cross_fma_tile_ampere_(load_fn, accumulation, nk_cross_triangle_full_k, nk_cross_metric_##metric_name##_k,  \
                                  &arguments);                                                                         \
    }                                                                                                                  \
    NK_API_COMPTIME cudaError_t nk_##metric_name##s_packed_##input_type_name##_##isa_suffix(                           \
        nk_##input_value_type##_t const *a_matrix, void const *b_packed_buffer, nk_f64_t *c_matrix,                    \
        nk_size_t row_count, nk_size_t column_count, nk_size_t depth, nk_size_t a_stride_in_bytes,                     \
        nk_size_t c_stride_in_bytes, cudaStream_t stream) {                                                            \
        nk_size_t const row_bytes = nk_cross_padded_values_ampere_(depth, depth_simd_dimensions, dimensions_per_value, \
                                                                   sizeof(nk_##packed_value_type##_t)) *               \
                                    sizeof(nk_##packed_value_type##_t);                                                \
        unsigned char const *b_rows = (unsigned char const *)b_packed_buffer +                                         \
                                      sizeof(nk_cross_packed_buffer_header_t);                                         \
        return nk_cross_launch_ampere_(                                                                                \
            (void const *)nk_##metric_name##s_packed_##input_type_name##_##isa_suffix##_kernel_,                       \
            nk_cross_fma_tile_ampere_k, nk_cross_fma_threads_ampere_k, a_matrix, b_rows,                               \
            b_rows + column_count * row_bytes, c_matrix, sizeof(nk_f64_t), 0, row_count, column_count, depth,          \
            depth * sizeof(nk_##input_value_type##_t), a_stride_in_bytes, row_bytes, c_stride_in_bytes, stream);       \
    }

/**
 *  @brief Generates angular or euclidean distances in F64 on the CUDA cores among every vector and
 *      rows [row_start, row_start + row_count), writing above the diagonal and zeros on it.
 *  @sa nk_define_cross_normalized_symmetric_ for the host original.
 */
#define nk_define_cross_cuda_fma_normalized_symmetric_(metric_name, input_type_name, isa_suffix, input_value_type,     \
                                                       load_fn, accumulation)                                          \
    static __global__ void __launch_bounds__(nk_cross_fma_threads_ampere_k)                                            \
        nk_##metric_name##s_symmetric_##input_type_name##_##isa_suffix##_kernel_(                                      \
            nk_cross_tile_arguments_ampere_t arguments) {                                                              \
        nk_cross_fma_tile_ampere_(load_fn, accumulation, nk_cross_triangle_upper_k, nk_cross_metric_##metric_name##_k, \
                                  &arguments);                                                                         \
    }                                                                                                                  \
    NK_API_COMPTIME cudaError_t nk_##metric_name##s_symmetric_##input_type_name##_##isa_suffix(                        \
        nk_##input_value_type##_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_in_bytes, \
        nk_f64_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count,                  \
        cudaStream_t stream) {                                                                                         \
        nk_size_t const row_end = row_start + row_count < vectors_count ? row_start + row_count : vectors_count;       \
        return nk_cross_launch_ampere_(                                                                                \
            (void const *)nk_##metric_name##s_symmetric_##input_type_name##_##isa_suffix##_kernel_,                    \
            nk_cross_fma_tile_ampere_k, nk_cross_fma_threads_ampere_k, vectors, vectors, 0, result, sizeof(nk_f64_t),  \
            row_start, row_end, vectors_count, depth, depth * sizeof(nk_##input_value_type##_t), stride_in_bytes,      \
            stride_in_bytes, result_stride_in_bytes, stream);                                                          \
    }

#pragma endregion Cross Macros

#pragma region F64

nk_define_cross_cuda_fma_normalized_packed_(angular, f64, ampere, f64, f64, nk_load_f64_ampere_,
                                            nk_cross_accumulation_dot2_k, /*depth_simd_dimensions=*/8,
                                            /*dimensions_per_value=*/1)
nk_define_cross_cuda_fma_normalized_packed_(euclidean, f64, ampere, f64, f64, nk_load_f64_ampere_,
                                            nk_cross_accumulation_dot2_k, /*depth_simd_dimensions=*/8,
                                            /*dimensions_per_value=*/1)
nk_define_cross_cuda_fma_normalized_symmetric_(angular, f64, ampere, f64, nk_load_f64_ampere_,
                                               nk_cross_accumulation_dot2_k)
nk_define_cross_cuda_fma_normalized_symmetric_(euclidean, f64, ampere, f64, nk_load_f64_ampere_,
                                               nk_cross_accumulation_dot2_k)

#pragma endregion F64

#pragma region F32

nk_define_cross_cuda_fma_normalized_packed_(angular, f32, ampere, f32, f32, nk_load_f32_to_f64_ampere_,
                                            nk_cross_accumulation_f64_k, /*depth_simd_dimensions=*/16,
                                            /*dimensions_per_value=*/1)
nk_define_cross_cuda_fma_normalized_packed_(euclidean, f32, ampere, f32, f32, nk_load_f32_to_f64_ampere_,
                                            nk_cross_accumulation_f64_k, /*depth_simd_dimensions=*/16,
                                            /*dimensions_per_value=*/1)
nk_define_cross_cuda_fma_normalized_symmetric_(angular, f32, ampere, f32, nk_load_f32_to_f64_ampere_,
                                               nk_cross_accumulation_f64_k)
nk_define_cross_cuda_fma_normalized_symmetric_(euclidean, f32, ampere, f32, nk_load_f32_to_f64_ampere_,
                                               nk_cross_accumulation_f64_k)

#pragma endregion F32

#pragma region BF16

nk_define_cross_cuda_normalized_packed_(angular, bf16, ampere, bf16, bf16, nk_dots_bf16_multiply_ampere_,
                                        nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, nk_cross_norm_f32_k,
                                        nk_bf16_norm_update_ampere_, /*norm_scale=*/1.0f, /*depth_simd_dimensions=*/32,
                                        /*dimensions_per_value=*/1)
nk_define_cross_cuda_normalized_packed_(euclidean, bf16, ampere, bf16, bf16, nk_dots_bf16_multiply_ampere_,
                                        nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, nk_cross_norm_f32_k,
                                        nk_bf16_norm_update_ampere_, /*norm_scale=*/1.0f, /*depth_simd_dimensions=*/32,
                                        /*dimensions_per_value=*/1)
nk_define_cross_cuda_normalized_symmetric_(angular, bf16, ampere, bf16, nk_dots_bf16_multiply_ampere_,
                                           nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, nk_cross_norm_f32_k,
                                           nk_bf16_norm_update_ampere_, /*norm_scale=*/1.0f, /*dimensions_per_value=*/1)
nk_define_cross_cuda_normalized_symmetric_(euclidean, bf16, ampere, bf16, nk_dots_bf16_multiply_ampere_,
                                           nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, nk_cross_norm_f32_k,
                                           nk_bf16_norm_update_ampere_, /*norm_scale=*/1.0f, /*dimensions_per_value=*/1)

#pragma endregion BF16

#pragma region F16

nk_define_cross_cuda_normalized_packed_(angular, f16, ampere, f16, f16, nk_dots_f16_multiply_ampere_,
                                        nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, nk_cross_norm_f32_k,
                                        nk_f16_norm_update_ampere_, /*norm_scale=*/1.0f, /*depth_simd_dimensions=*/32,
                                        /*dimensions_per_value=*/1)
nk_define_cross_cuda_normalized_packed_(euclidean, f16, ampere, f16, f16, nk_dots_f16_multiply_ampere_,
                                        nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, nk_cross_norm_f32_k,
                                        nk_f16_norm_update_ampere_, /*norm_scale=*/1.0f, /*depth_simd_dimensions=*/32,
                                        /*dimensions_per_value=*/1)
nk_define_cross_cuda_normalized_symmetric_(angular, f16, ampere, f16, nk_dots_f16_multiply_ampere_,
                                           nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, nk_cross_norm_f32_k,
                                           nk_f16_norm_update_ampere_, /*norm_scale=*/1.0f, /*dimensions_per_value=*/1)
nk_define_cross_cuda_normalized_symmetric_(euclidean, f16, ampere, f16, nk_dots_f16_multiply_ampere_,
                                           nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, nk_cross_norm_f32_k,
                                           nk_f16_norm_update_ampere_, /*norm_scale=*/1.0f, /*dimensions_per_value=*/1)

#pragma endregion F16

#pragma region E5M2

nk_define_cross_cuda_normalized_packed_(angular, e5m2, ampere, e5m2, e5m2, nk_dots_e5m2_multiply_ampere_,
                                        nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, nk_cross_norm_f32_k,
                                        nk_e5m2_norm_update_ampere_, /*norm_scale=*/1.0f, /*depth_simd_dimensions=*/64,
                                        /*dimensions_per_value=*/1)
nk_define_cross_cuda_normalized_packed_(euclidean, e5m2, ampere, e5m2, e5m2, nk_dots_e5m2_multiply_ampere_,
                                        nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, nk_cross_norm_f32_k,
                                        nk_e5m2_norm_update_ampere_, /*norm_scale=*/1.0f, /*depth_simd_dimensions=*/64,
                                        /*dimensions_per_value=*/1)
nk_define_cross_cuda_normalized_symmetric_(angular, e5m2, ampere, e5m2, nk_dots_e5m2_multiply_ampere_,
                                           nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, nk_cross_norm_f32_k,
                                           nk_e5m2_norm_update_ampere_, /*norm_scale=*/1.0f, /*dimensions_per_value=*/1)
nk_define_cross_cuda_normalized_symmetric_(euclidean, e5m2, ampere, e5m2, nk_dots_e5m2_multiply_ampere_,
                                           nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, nk_cross_norm_f32_k,
                                           nk_e5m2_norm_update_ampere_, /*norm_scale=*/1.0f, /*dimensions_per_value=*/1)

#pragma endregion E5M2

#pragma region E4M3

nk_define_cross_cuda_normalized_packed_(angular, e4m3, ampere, e4m3, e4m3, nk_dots_e4m3_multiply_ampere_,
                                        nk_cross_epilogue_f32_k, /*output_scale=*/65536.0f, nk_cross_norm_f32_k,
                                        nk_e4m3_norm_update_ampere_, /*norm_scale=*/65536.0f,
                                        /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_cuda_normalized_packed_(euclidean, e4m3, ampere, e4m3, e4m3, nk_dots_e4m3_multiply_ampere_,
                                        nk_cross_epilogue_f32_k, /*output_scale=*/65536.0f, nk_cross_norm_f32_k,
                                        nk_e4m3_norm_update_ampere_, /*norm_scale=*/65536.0f,
                                        /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_cuda_normalized_symmetric_(angular, e4m3, ampere, e4m3, nk_dots_e4m3_multiply_ampere_,
                                           nk_cross_epilogue_f32_k, /*output_scale=*/65536.0f, nk_cross_norm_f32_k,
                                           nk_e4m3_norm_update_ampere_, /*norm_scale=*/65536.0f,
                                           /*dimensions_per_value=*/1)
nk_define_cross_cuda_normalized_symmetric_(euclidean, e4m3, ampere, e4m3, nk_dots_e4m3_multiply_ampere_,
                                           nk_cross_epilogue_f32_k, /*output_scale=*/65536.0f, nk_cross_norm_f32_k,
                                           nk_e4m3_norm_update_ampere_, /*norm_scale=*/65536.0f,
                                           /*dimensions_per_value=*/1)

#pragma endregion E4M3

#pragma region E3M2

nk_define_cross_cuda_normalized_packed_(angular, e3m2, ampere, e3m2, e3m2, nk_dots_e3m2_multiply_ampere_,
                                        nk_cross_epilogue_f32_k, /*output_scale=*/16777216.0f, nk_cross_norm_f32_k,
                                        nk_e3m2_norm_update_ampere_, /*norm_scale=*/16777216.0f,
                                        /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_cuda_normalized_packed_(euclidean, e3m2, ampere, e3m2, e3m2, nk_dots_e3m2_multiply_ampere_,
                                        nk_cross_epilogue_f32_k, /*output_scale=*/16777216.0f, nk_cross_norm_f32_k,
                                        nk_e3m2_norm_update_ampere_, /*norm_scale=*/16777216.0f,
                                        /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_cuda_normalized_symmetric_(angular, e3m2, ampere, e3m2, nk_dots_e3m2_multiply_ampere_,
                                           nk_cross_epilogue_f32_k, /*output_scale=*/16777216.0f, nk_cross_norm_f32_k,
                                           nk_e3m2_norm_update_ampere_, /*norm_scale=*/16777216.0f,
                                           /*dimensions_per_value=*/1)
nk_define_cross_cuda_normalized_symmetric_(euclidean, e3m2, ampere, e3m2, nk_dots_e3m2_multiply_ampere_,
                                           nk_cross_epilogue_f32_k, /*output_scale=*/16777216.0f, nk_cross_norm_f32_k,
                                           nk_e3m2_norm_update_ampere_, /*norm_scale=*/16777216.0f,
                                           /*dimensions_per_value=*/1)

#pragma endregion E3M2

#pragma region E2M3

nk_define_cross_cuda_normalized_packed_(angular, e2m3, ampere, e2m3, i8, nk_dots_e2m3_packed_multiply_ampere_,
                                        nk_cross_epilogue_i32_to_f32_k, /*output_scale=*/0.015625f, nk_cross_norm_f32_k,
                                        nk_e2m3_norm_update_ampere_, /*norm_scale=*/0.015625f,
                                        /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_cuda_normalized_packed_(euclidean, e2m3, ampere, e2m3, i8, nk_dots_e2m3_packed_multiply_ampere_,
                                        nk_cross_epilogue_i32_to_f32_k, /*output_scale=*/0.015625f, nk_cross_norm_f32_k,
                                        nk_e2m3_norm_update_ampere_, /*norm_scale=*/0.015625f,
                                        /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_cuda_normalized_symmetric_(angular, e2m3, ampere, e2m3, nk_dots_e2m3_multiply_ampere_,
                                           nk_cross_epilogue_i32_to_f32_k, /*output_scale=*/0.015625f,
                                           nk_cross_norm_f32_k, nk_e2m3_norm_update_ampere_, /*norm_scale=*/0.015625f,
                                           /*dimensions_per_value=*/1)
nk_define_cross_cuda_normalized_symmetric_(euclidean, e2m3, ampere, e2m3, nk_dots_e2m3_multiply_ampere_,
                                           nk_cross_epilogue_i32_to_f32_k, /*output_scale=*/0.015625f,
                                           nk_cross_norm_f32_k, nk_e2m3_norm_update_ampere_, /*norm_scale=*/0.015625f,
                                           /*dimensions_per_value=*/1)

#pragma endregion E2M3

#pragma region E2M1

nk_define_cross_cuda_normalized_packed_(angular, e2m1, ampere, e2m1x2, e2m1x2, nk_dots_e2m1_multiply_ampere_,
                                        nk_cross_epilogue_i32_to_f32_k, /*output_scale=*/0.25f, nk_cross_norm_f32_k,
                                        nk_e2m1_norm_update_ampere_, /*norm_scale=*/0.25f,
                                        /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/2)
nk_define_cross_cuda_normalized_packed_(euclidean, e2m1, ampere, e2m1x2, e2m1x2, nk_dots_e2m1_multiply_ampere_,
                                        nk_cross_epilogue_i32_to_f32_k, /*output_scale=*/0.25f, nk_cross_norm_f32_k,
                                        nk_e2m1_norm_update_ampere_, /*norm_scale=*/0.25f,
                                        /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/2)
nk_define_cross_cuda_normalized_symmetric_(angular, e2m1, ampere, e2m1x2, nk_dots_e2m1_multiply_ampere_,
                                           nk_cross_epilogue_i32_to_f32_k, /*output_scale=*/0.25f, nk_cross_norm_f32_k,
                                           nk_e2m1_norm_update_ampere_, /*norm_scale=*/0.25f,
                                           /*dimensions_per_value=*/2)
nk_define_cross_cuda_normalized_symmetric_(euclidean, e2m1, ampere, e2m1x2, nk_dots_e2m1_multiply_ampere_,
                                           nk_cross_epilogue_i32_to_f32_k, /*output_scale=*/0.25f, nk_cross_norm_f32_k,
                                           nk_e2m1_norm_update_ampere_, /*norm_scale=*/0.25f,
                                           /*dimensions_per_value=*/2)

#pragma endregion E2M1

#pragma region I8

nk_define_cross_cuda_normalized_packed_(angular, i8, ampere, i8, i8, nk_dots_i8_multiply_ampere_,
                                        nk_cross_epilogue_i32_k, /*output_scale=*/1.0f, nk_cross_norm_i32_k,
                                        nk_i8_norm_update_ampere_, /*norm_scale=*/1.0f, /*depth_simd_dimensions=*/64,
                                        /*dimensions_per_value=*/1)
nk_define_cross_cuda_normalized_packed_(euclidean, i8, ampere, i8, i8, nk_dots_i8_multiply_ampere_,
                                        nk_cross_epilogue_i32_k, /*output_scale=*/1.0f, nk_cross_norm_i32_k,
                                        nk_i8_norm_update_ampere_, /*norm_scale=*/1.0f, /*depth_simd_dimensions=*/64,
                                        /*dimensions_per_value=*/1)
nk_define_cross_cuda_normalized_symmetric_(angular, i8, ampere, i8, nk_dots_i8_multiply_ampere_,
                                           nk_cross_epilogue_i32_k, /*output_scale=*/1.0f, nk_cross_norm_i32_k,
                                           nk_i8_norm_update_ampere_, /*norm_scale=*/1.0f, /*dimensions_per_value=*/1)
nk_define_cross_cuda_normalized_symmetric_(euclidean, i8, ampere, i8, nk_dots_i8_multiply_ampere_,
                                           nk_cross_epilogue_i32_k, /*output_scale=*/1.0f, nk_cross_norm_i32_k,
                                           nk_i8_norm_update_ampere_, /*norm_scale=*/1.0f, /*dimensions_per_value=*/1)

#pragma endregion I8

#pragma region I4

nk_define_cross_cuda_normalized_packed_(angular, i4, ampere, i4x2, i4x2, nk_dots_i4_multiply_ampere_,
                                        nk_cross_epilogue_i32_k, /*output_scale=*/1.0f, nk_cross_norm_i32_k,
                                        nk_i4_norm_update_ampere_, /*norm_scale=*/1.0f, /*depth_simd_dimensions=*/128,
                                        /*dimensions_per_value=*/2)
nk_define_cross_cuda_normalized_packed_(euclidean, i4, ampere, i4x2, i4x2, nk_dots_i4_multiply_ampere_,
                                        nk_cross_epilogue_i32_k, /*output_scale=*/1.0f, nk_cross_norm_i32_k,
                                        nk_i4_norm_update_ampere_, /*norm_scale=*/1.0f, /*depth_simd_dimensions=*/128,
                                        /*dimensions_per_value=*/2)
nk_define_cross_cuda_normalized_symmetric_(angular, i4, ampere, i4x2, nk_dots_i4_multiply_ampere_,
                                           nk_cross_epilogue_i32_k, /*output_scale=*/1.0f, nk_cross_norm_i32_k,
                                           nk_i4_norm_update_ampere_, /*norm_scale=*/1.0f, /*dimensions_per_value=*/2)
nk_define_cross_cuda_normalized_symmetric_(euclidean, i4, ampere, i4x2, nk_dots_i4_multiply_ampere_,
                                           nk_cross_epilogue_i32_k, /*output_scale=*/1.0f, nk_cross_norm_i32_k,
                                           nk_i4_norm_update_ampere_, /*norm_scale=*/1.0f, /*dimensions_per_value=*/2)

#pragma endregion I4

#pragma region U8

nk_define_cross_cuda_normalized_packed_(angular, u8, ampere, u8, u8, nk_dots_u8_multiply_ampere_,
                                        nk_cross_epilogue_i32_k, /*output_scale=*/1.0f, nk_cross_norm_u32_k,
                                        nk_u8_norm_update_ampere_, /*norm_scale=*/1.0f, /*depth_simd_dimensions=*/64,
                                        /*dimensions_per_value=*/1)
nk_define_cross_cuda_normalized_packed_(euclidean, u8, ampere, u8, u8, nk_dots_u8_multiply_ampere_,
                                        nk_cross_epilogue_i32_k, /*output_scale=*/1.0f, nk_cross_norm_u32_k,
                                        nk_u8_norm_update_ampere_, /*norm_scale=*/1.0f, /*depth_simd_dimensions=*/64,
                                        /*dimensions_per_value=*/1)
nk_define_cross_cuda_normalized_symmetric_(angular, u8, ampere, u8, nk_dots_u8_multiply_ampere_,
                                           nk_cross_epilogue_i32_k, /*output_scale=*/1.0f, nk_cross_norm_u32_k,
                                           nk_u8_norm_update_ampere_, /*norm_scale=*/1.0f, /*dimensions_per_value=*/1)
nk_define_cross_cuda_normalized_symmetric_(euclidean, u8, ampere, u8, nk_dots_u8_multiply_ampere_,
                                           nk_cross_epilogue_i32_k, /*output_scale=*/1.0f, nk_cross_norm_u32_k,
                                           nk_u8_norm_update_ampere_, /*norm_scale=*/1.0f, /*dimensions_per_value=*/1)

#pragma endregion U8

#pragma region U4

nk_define_cross_cuda_normalized_packed_(angular, u4, ampere, u4x2, u4x2, nk_dots_u4_multiply_ampere_,
                                        nk_cross_epilogue_i32_k, /*output_scale=*/1.0f, nk_cross_norm_u32_k,
                                        nk_u4_norm_update_ampere_, /*norm_scale=*/1.0f, /*depth_simd_dimensions=*/128,
                                        /*dimensions_per_value=*/2)
nk_define_cross_cuda_normalized_packed_(euclidean, u4, ampere, u4x2, u4x2, nk_dots_u4_multiply_ampere_,
                                        nk_cross_epilogue_i32_k, /*output_scale=*/1.0f, nk_cross_norm_u32_k,
                                        nk_u4_norm_update_ampere_, /*norm_scale=*/1.0f, /*depth_simd_dimensions=*/128,
                                        /*dimensions_per_value=*/2)
nk_define_cross_cuda_normalized_symmetric_(angular, u4, ampere, u4x2, nk_dots_u4_multiply_ampere_,
                                           nk_cross_epilogue_i32_k, /*output_scale=*/1.0f, nk_cross_norm_u32_k,
                                           nk_u4_norm_update_ampere_, /*norm_scale=*/1.0f, /*dimensions_per_value=*/2)
nk_define_cross_cuda_normalized_symmetric_(euclidean, u4, ampere, u4x2, nk_dots_u4_multiply_ampere_,
                                           nk_cross_epilogue_i32_k, /*output_scale=*/1.0f, nk_cross_norm_u32_k,
                                           nk_u4_norm_update_ampere_, /*norm_scale=*/1.0f, /*dimensions_per_value=*/2)

#pragma endregion U4

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_AMPERE
#endif // NK_SPATIALS_AMPERE_CUH
