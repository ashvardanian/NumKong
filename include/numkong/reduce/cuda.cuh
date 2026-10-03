/**
 *  @file include/numkong/reduce/cuda.cuh
 *  @author Ash Vardanian
 *  @date October 3, 2026
 *  @brief CUDA host side of moments and min/max reductions: the clears and launches, the export
 *      generators and the @c cuda exports, over the kernels of `reduce/simt.cuh`.
 *
 *  @sa include/numkong/reduce/simt.cuh
 */
#ifndef NUMKONG_REDUCE_CUDA_CUH
#define NUMKONG_REDUCE_CUDA_CUH

#include "numkong/cuda.cuh"
#include "numkong/reduce/simt.cuh"

#if NUMKONG_ARCH_CUDA_ && defined(__CUDACC__) && !defined(__HIP__)

#if defined(__cplusplus)
extern "C" {
#endif

/** Sets @p bytes at device-reachable @p pointer to @p value, in order on @p stream. */
NUMKONG_INLINE nk_status_t nk_reduce_memset_cuda_(void *pointer, int value, nk_size_t bytes, void *stream) {
    return cudaMemsetAsync(pointer, value, bytes, (cudaStream_t)stream) == cudaSuccess ? nk_success_k
                                                                                       : nk_device_code_mismatch_k;
}

/** Zeroes the @p sum_bytes and @p sumsq_bytes wide outputs, then launches @p kernel's resident
 *  grid, whose blocks add into them. */
NUMKONG_INLINE nk_status_t nk_reduce_moments_launch_cuda_(void const *kernel, nk_reduce_arguments_t arguments,
                                                          nk_size_t sum_bytes, nk_size_t sumsq_bytes, void *stream) {
    nk_status_t status = nk_reduce_memset_cuda_(arguments.first, 0, sum_bytes, stream);
    if (status == nk_success_k) status = nk_reduce_memset_cuda_(arguments.second, 0, sumsq_bytes, stream);
    if (status != nk_success_k || !arguments.count) return status;
    return nk_launch_resident_cuda_(
        kernel, nk_reduce_threads_simt_k, 0, 0,
        nk_size_divide_round_up_(arguments.count, nk_reduce_threads_simt_k * nk_reduce_batch_simt_k), &arguments,
        stream);
}

/** Clears both indices, launches @p kernel's resident grid, whose blocks swap their winners in, and
 *  then @p finish_kernel, which writes the winners' values. */
NUMKONG_INLINE nk_status_t nk_reduce_minmax_launch_cuda_(void const *kernel, void const *finish_kernel,
                                                         nk_reduce_arguments_t arguments, void *stream) {
    nk_status_t status = nk_reduce_memset_cuda_(arguments.first_index, 0xFF, sizeof(nk_size_t), stream);
    if (status == nk_success_k)
        status = nk_reduce_memset_cuda_(arguments.second_index, 0xFF, sizeof(nk_size_t), stream);
    if (status == nk_success_k && arguments.count)
        status = nk_launch_resident_cuda_(
            kernel, nk_reduce_threads_simt_k, 0, 0,
            nk_size_divide_round_up_(arguments.count, nk_reduce_threads_simt_k * nk_reduce_batch_simt_k), &arguments,
            stream);
    void *launch_arguments[1];
    launch_arguments[0] = &arguments;
    return status == nk_success_k ? nk_launch_cuda_(finish_kernel, 1, 1, launch_arguments, 0, stream) : status;
}

/** Generates the moments entry point of @p input_type over the kernel of
 *  @c nk_define_reduce_moments_kernel_simt_. */
#define nk_define_reduce_moments_cuda_(input_type, input_value_type, family, sum_type, sumsq_type, isa_suffix)   \
    nk_define_reduce_moments_kernel_simt_(input_type, family, isa_suffix) NUMKONG_API nk_status_t                \
    nk_reduce_moments_##input_type##_##isa_suffix(nk_##input_value_type##_t const *data, nk_size_t count,        \
                                                  nk_size_t stride, nk_##sum_type##_t *sum,                      \
                                                  nk_##sumsq_type##_t *sumsq, void *stream) {                    \
        return nk_reduce_moments_launch_cuda_(                                                                   \
            (void const *)&nk_reduce_moments_##input_type##_##isa_suffix##_kernel_,                              \
            nk_reduce_arguments_simt_(sizeof(nk_##input_value_type##_t), data, count, stride, sum, NUMKONG_NULL, \
                                      sumsq, NUMKONG_NULL),                                                      \
            sizeof(nk_##sum_type##_t), sizeof(nk_##sumsq_type##_t), stream);                                     \
    }

/** Generates the min/max entry point of @p input_type over the kernels of
 *  @c nk_define_reduce_minmax_kernels_simt_. */
#define nk_define_reduce_minmax_cuda_(input_type, input_value_type, output_type, min_sentinel, max_sentinel,          \
                                      isa_suffix)                                                                     \
    nk_define_reduce_minmax_kernels_simt_(input_type, min_sentinel, max_sentinel, isa_suffix) NUMKONG_API nk_status_t \
    nk_reduce_minmax_##input_type##_##isa_suffix(                                                                     \
        nk_##input_value_type##_t const *data, nk_size_t count, nk_size_t stride, nk_##output_type##_t *min_value,    \
        nk_size_t *min_index, nk_##output_type##_t *max_value, nk_size_t *max_index, void *stream) {                  \
        return nk_reduce_minmax_launch_cuda_(                                                                         \
            (void const *)&nk_reduce_minmax_##input_type##_##isa_suffix##_kernel_,                                    \
            (void const *)&nk_reduce_minmax_##input_type##_finish_##isa_suffix##_kernel_,                             \
            nk_reduce_arguments_simt_(sizeof(nk_##input_value_type##_t), data, count, stride, min_value, min_index,   \
                                      max_value, max_index),                                                          \
            stream);                                                                                                  \
    }

#if NUMKONG_TARGET_CUDA
nk_define_reduce_moments_cuda_(f64, f64, f64, f64, f64, cuda)
nk_define_reduce_moments_cuda_(f32, f32, pairs, f64, f64, cuda)
nk_define_reduce_moments_cuda_(f16, f16, f32, f32, f32, cuda)
nk_define_reduce_moments_cuda_(bf16, bf16, pairs, f32, f32, cuda)
nk_define_reduce_moments_cuda_(e4m3, e4m3, f32, f32, f32, cuda)
nk_define_reduce_moments_cuda_(e5m2, e5m2, f32, f32, f32, cuda)
nk_define_reduce_moments_cuda_(e2m3, e2m3, f32, f32, f32, cuda)
nk_define_reduce_moments_cuda_(e3m2, e3m2, f32, f32, f32, cuda)
nk_define_reduce_moments_cuda_(e2m1, e2m1x2, integer, f32, f32, cuda)
nk_define_reduce_moments_cuda_(i8, i8, integer, i64, u64, cuda)
nk_define_reduce_moments_cuda_(u8, u8, integer, u64, u64, cuda)
nk_define_reduce_moments_cuda_(i16, i16, integer, i64, u64, cuda)
nk_define_reduce_moments_cuda_(u16, u16, integer, u64, u64, cuda)
nk_define_reduce_moments_cuda_(i32, i32, integer, i64, u64, cuda)
nk_define_reduce_moments_cuda_(u32, u32, integer, u64, u64, cuda)
nk_define_reduce_moments_cuda_(i64, i64, integer, i64, u64, cuda)
nk_define_reduce_moments_cuda_(u64, u64, integer, u64, u64, cuda)
nk_define_reduce_moments_cuda_(i4, i4x2, integer, i64, u64, cuda)
nk_define_reduce_moments_cuda_(u4, u4x2, integer, u64, u64, cuda)
nk_define_reduce_moments_cuda_(u1, u1x8, integer, u64, u64, cuda)
nk_define_reduce_minmax_cuda_(f64, f64, f64, 0x7FF0000000000000ull, 0xFFF0000000000000ull, cuda)
nk_define_reduce_minmax_cuda_(f32, f32, f32, 0x7F800000u, 0xFF800000u, cuda)
nk_define_reduce_minmax_cuda_(f16, f16, f16, 0x7BFFu, 0xFBFFu, cuda)
nk_define_reduce_minmax_cuda_(bf16, bf16, bf16, 0x7F7Fu, 0xFF7Fu, cuda)
nk_define_reduce_minmax_cuda_(e4m3, e4m3, e4m3, NUMKONG_E4M3_MAX, NUMKONG_E4M3_MIN, cuda)
nk_define_reduce_minmax_cuda_(e5m2, e5m2, e5m2, NUMKONG_E5M2_MAX, NUMKONG_E5M2_MIN, cuda)
nk_define_reduce_minmax_cuda_(e2m3, e2m3, e2m3, NUMKONG_E2M3_MAX, NUMKONG_E2M3_MIN, cuda)
nk_define_reduce_minmax_cuda_(e3m2, e3m2, e3m2, NUMKONG_E3M2_MAX, NUMKONG_E3M2_MIN, cuda)
nk_define_reduce_minmax_cuda_(i8, i8, i8, 0x7Fu, 0x80u, cuda)
nk_define_reduce_minmax_cuda_(u8, u8, u8, 0xFFu, 0u, cuda)
nk_define_reduce_minmax_cuda_(i16, i16, i16, 0x7FFFu, 0x8000u, cuda)
nk_define_reduce_minmax_cuda_(u16, u16, u16, 0xFFFFu, 0u, cuda)
nk_define_reduce_minmax_cuda_(i32, i32, i32, 0x7FFFFFFFu, 0x80000000u, cuda)
nk_define_reduce_minmax_cuda_(u32, u32, u32, 0xFFFFFFFFu, 0u, cuda)
nk_define_reduce_minmax_cuda_(i64, i64, i64, 0x7FFFFFFFFFFFFFFFull, 0x8000000000000000ull, cuda)
nk_define_reduce_minmax_cuda_(u64, u64, u64, NUMKONG_U64_MAX, 0u, cuda)
nk_define_reduce_minmax_cuda_(i4, i4x2, i8, 0x07u, 0xF8u, cuda)
nk_define_reduce_minmax_cuda_(u4, u4x2, u8, 0x0Fu, 0u, cuda)
nk_define_reduce_minmax_cuda_(u1, u1x8, u8, 1u, 0u, cuda)
#endif // NUMKONG_TARGET_CUDA

#undef nk_define_reduce_moments_cuda_
#undef nk_define_reduce_minmax_cuda_

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_ && defined(__CUDACC__) && !defined(__HIP__)
#endif // NUMKONG_REDUCE_CUDA_CUH
