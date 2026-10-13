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

#if NUMKONG_ARCH_CUDA_

#include "numkong/cuda.cuh"
#include "numkong/reduce/simt.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

/** Sets @p bytes at device-reachable @p pointer to @p value, in order on @p stream. */
NUMKONG_INLINE nk_status_t nk_reduce_memset_cuda_(void *pointer, int value, nk_size_t bytes, nk_stream_t stream) {
    return cudaMemsetAsync(pointer, value, bytes, (cudaStream_t)stream) == cudaSuccess ? nk_success_k
                                                                                       : nk_device_code_mismatch_k;
}

/** Zeroes the @p sum_bytes and @p sumsq_bytes wide outputs, then launches @p kernel's resident
 *  grid, whose blocks add into them. */
NUMKONG_INLINE nk_status_t nk_reduce_moments_launch_cuda_(void const *kernel, nk_reduce_arguments_t arguments,
                                                          nk_size_t sum_bytes, nk_size_t sumsq_bytes,
                                                          nk_stream_t stream) {
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
                                                         nk_reduce_arguments_t arguments, nk_stream_t stream) {
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

#if NUMKONG_TARGET_CUDA
nk_define_reduce_moments_simt_(f64, f64, 1, compensated, f64, f64, cuda, cuda, nk_assign_from_to_)
nk_define_reduce_moments_simt_(f32, f32, 1, f64, f64, f64, cuda, cuda, nk_assign_from_to_)
nk_define_reduce_moments_simt_(f16, f16, 1, f32, f32, f32, cuda, cuda, nk_f16_to_f32_simt_)
nk_define_reduce_moments_simt_(bf16, bf16, 1, f32, f32, f32, cuda, cuda, nk_bf16_to_f32_simt_)
nk_define_reduce_moments_simt_(e4m3, e4m3, 1, f32, f32, f32, cuda, cuda, nk_e4m3_to_f32_simt_)
nk_define_reduce_moments_simt_(e5m2, e5m2, 1, f32, f32, f32, cuda, cuda, nk_e5m2_to_f32_simt_)
nk_define_reduce_moments_simt_(e2m3, e2m3, 1, f32, f32, f32, cuda, cuda, nk_e2m3_to_f32_simt_)
nk_define_reduce_moments_simt_(e3m2, e3m2, 1, f32, f32, f32, cuda, cuda, nk_e3m2_to_f32_simt_)
nk_define_reduce_moments_simt_(e2m1, e2m1x2, 2, e2m1, f32, f32, cuda, cuda, nk_e2m1x2_to_f32x2_simt_)
nk_define_reduce_moments_simt_(i8, i8, 1, i64, i64, u64, cuda, cuda, nk_assign_from_to_)
nk_define_reduce_moments_simt_(u8, u8, 1, u64, u64, u64, cuda, cuda, nk_assign_from_to_)
nk_define_reduce_moments_simt_(i16, i16, 1, i64, i64, u64, cuda, cuda, nk_assign_from_to_)
nk_define_reduce_moments_simt_(u16, u16, 1, u64, u64, u64, cuda, cuda, nk_assign_from_to_)
nk_define_reduce_moments_simt_(i32, i32, 1, i64, i64, u64, cuda, cuda, nk_assign_from_to_)
nk_define_reduce_moments_simt_(u32, u32, 1, u64, u64, u64, cuda, cuda, nk_assign_from_to_)
nk_define_reduce_moments_simt_(i64, i64, 1, i64, i64, u64, cuda, cuda, nk_assign_from_to_)
nk_define_reduce_moments_simt_(u64, u64, 1, u64, u64, u64, cuda, cuda, nk_assign_from_to_)
nk_define_reduce_moments_simt_(i4, i4x2, 2, i64, i64, u64, cuda, cuda, nk_i4x2_to_i64x2_simt_)
nk_define_reduce_moments_simt_(u4, u4x2, 2, u64, u64, u64, cuda, cuda, nk_u4x2_to_u64x2_simt_)
nk_define_reduce_moments_simt_(u1, u1x8, 8, u64, u64, u64, cuda, cuda, nk_u1x8_to_u64x8_simt_)
nk_define_reduce_minmax_simt_(f64, f64, f64, 0x7FF0000000000000ull, 0xFFF0000000000000ull, cuda, cuda)
nk_define_reduce_minmax_simt_(f32, f32, f32, 0x7F800000u, 0xFF800000u, cuda, cuda)
nk_define_reduce_minmax_simt_(f16, f16, f16, 0x7BFFu, 0xFBFFu, cuda, cuda)
nk_define_reduce_minmax_simt_(bf16, bf16, bf16, 0x7F7Fu, 0xFF7Fu, cuda, cuda)
nk_define_reduce_minmax_simt_(e4m3, e4m3, e4m3, NUMKONG_E4M3_MAX, NUMKONG_E4M3_MIN, cuda, cuda)
nk_define_reduce_minmax_simt_(e5m2, e5m2, e5m2, NUMKONG_E5M2_MAX, NUMKONG_E5M2_MIN, cuda, cuda)
nk_define_reduce_minmax_simt_(e2m3, e2m3, e2m3, NUMKONG_E2M3_MAX, NUMKONG_E2M3_MIN, cuda, cuda)
nk_define_reduce_minmax_simt_(e3m2, e3m2, e3m2, NUMKONG_E3M2_MAX, NUMKONG_E3M2_MIN, cuda, cuda)
nk_define_reduce_minmax_simt_(i8, i8, i8, 0x7Fu, 0x80u, cuda, cuda)
nk_define_reduce_minmax_simt_(u8, u8, u8, 0xFFu, 0u, cuda, cuda)
nk_define_reduce_minmax_simt_(i16, i16, i16, 0x7FFFu, 0x8000u, cuda, cuda)
nk_define_reduce_minmax_simt_(u16, u16, u16, 0xFFFFu, 0u, cuda, cuda)
nk_define_reduce_minmax_simt_(i32, i32, i32, 0x7FFFFFFFu, 0x80000000u, cuda, cuda)
nk_define_reduce_minmax_simt_(u32, u32, u32, 0xFFFFFFFFu, 0u, cuda, cuda)
nk_define_reduce_minmax_simt_(i64, i64, i64, 0x7FFFFFFFFFFFFFFFull, 0x8000000000000000ull, cuda, cuda)
nk_define_reduce_minmax_simt_(u64, u64, u64, NUMKONG_U64_MAX, 0u, cuda, cuda)
nk_define_reduce_minmax_simt_(i4, i4x2, i8, 0x07u, 0xF8u, cuda, cuda)
nk_define_reduce_minmax_simt_(u4, u4x2, u8, 0x0Fu, 0u, cuda, cuda)
nk_define_reduce_minmax_simt_(u1, u1x8, u8, 1u, 0u, cuda, cuda)
#endif // NUMKONG_TARGET_CUDA

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_
#endif // NUMKONG_REDUCE_CUDA_CUH
