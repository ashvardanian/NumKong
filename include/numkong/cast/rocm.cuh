/**
 *  @file include/numkong/cast/rocm.cuh
 *  @author Ash Vardanian
 *  @date October 12, 2026
 *  @brief ROCm host side of bulk casts: the block-scaled launches and the @c rocm export, over the
 *      conversions of `cast/simt.cuh`.
 *
 *  @sa include/numkong/cast/simt.cuh
 *  @sa include/numkong/cast/cuda.cuh
 */
#ifndef NUMKONG_CAST_ROCM_CUH
#define NUMKONG_CAST_ROCM_CUH

#if NUMKONG_ARCH_ROCM_

#include "numkong/rocm.cuh"
#include "numkong/cast/simt.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

/** Validates a cast with a block-scaled side and queues its launches of @p kernel: a destination
 *  tensor scale adds the mark, abs-max and derive launches before the chunks. */
NUMKONG_INLINE nk_status_t nk_cast_block_scaled_launch_rocm_(void const *kernel, nk_cast_operand_t const &from,
                                                             nk_block_scaled_format_t const *from_format,
                                                             nk_cast_operand_t const &to,
                                                             nk_block_scaled_format_t const *to_format, nk_size_t count,
                                                             nk_stream_t stream) {
    int const from_plain = from_format->scale_dtype == nk_dtype_unknown_k || from_format->block_size == 0;
    int const to_plain = to_format->scale_dtype == nk_dtype_unknown_k || to_format->block_size == 0;
    nk_cast_block_scaled_arguments_t arguments;
    if ((!from_plain && from_format->block_size > 32) || (!to_plain && to_format->block_size > 32))
        return nk_unexpected_dimensions_k;
    if (((nk_size_t)from.codes & (nk_dtype_alignment_serial_(from_format->element_dtype) - 1)) ||
        ((nk_size_t)to.codes & (nk_dtype_alignment_serial_(to_format->element_dtype) - 1)))
        return nk_misaligned_k;
    arguments.from = (unsigned char const *)from.codes, arguments.to = (unsigned char *)to.codes;
    arguments.from_scales = from.scales, arguments.to_scales = to.scales;
    arguments.from_tensor_scale = from.tensor_scale, arguments.to_tensor_scale = to.tensor_scale;
    arguments.count = count, arguments.from_dtype = from_format->element_dtype;
    arguments.to_dtype = to_format->element_dtype;
    arguments.from_scale_dtype = from_plain ? nk_dtype_unknown_k : from_format->scale_dtype;
    arguments.to_scale_dtype = to_plain ? nk_dtype_unknown_k : to_format->scale_dtype;
    arguments.from_block = from_plain ? 1u : (unsigned)from_format->block_size;
    arguments.to_block = to_plain ? 1u : (unsigned)to_format->block_size;
    arguments.chunks = nk_size_divide_round_up_(
        count, arguments.from_block > arguments.to_block ? arguments.from_block : arguments.to_block);
    nk_status_t status = nk_success_k;
    if (!count && !arguments.to_tensor_scale) return status;
    if (arguments.to_tensor_scale && count) {
        arguments.phase = nk_cast_block_scaled_amax_k;
        status = nk_launch_resident_rocm_(kernel, nk_cast_threads_simt_k, 0, 0,
                                          nk_size_divide_round_up_(count, nk_cast_threads_simt_k), &arguments, stream);
    }
    if (status != nk_success_k) return status;
    arguments.phase = nk_cast_block_scaled_chunks_k;
    // One block at least, so an empty cast still writes the tensor scale it derives
    nk_size_t const blocks = nk_size_divide_round_up_(arguments.chunks, nk_cast_threads_simt_k);
    return nk_launch_resident_rocm_(kernel, nk_cast_threads_simt_k, 0, 0, blocks ? blocks : 1, &arguments, stream);
}

#if NUMKONG_TARGET_ROCM

static __global__ void __launch_bounds__(nk_cast_threads_simt_k, 1)
    nk_cast_rocm_kernel_(nk_cast_arguments_t arguments) {
    nk_size_t const stride = (nk_size_t)gridDim.x * blockDim.x;
    for (nk_size_t unit = (nk_size_t)blockIdx.x * blockDim.x + threadIdx.x; unit < arguments.plan.units; unit += stride)
        nk_cast_unit_simt_(&arguments, unit);
}

static __global__ void __launch_bounds__(nk_cast_threads_simt_k, 1)
    nk_cast_vectors_rocm_kernel_(nk_cast_arguments_t arguments) {
    nk_cast_vectors_simt_(&arguments);
}

static __global__ void __launch_bounds__(nk_cast_threads_simt_k, 1)
    nk_cast_block_scaled_rocm_kernel_(nk_cast_block_scaled_arguments_t arguments) {
    __shared__ nk_u32_t partials[256];
    nk_cast_block_scaled_phase_simt_(&arguments, partials);
}

NUMKONG_API nk_status_t nk_cast_rocm(void const *from, nk_dtype_t from_dtype, void *to, nk_dtype_t to_dtype,
                                     nk_size_t count, nk_stream_t stream) {
    nk_cast_arguments_t arguments;
    if (nk_dtype_is_block_scaled(from_dtype) || nk_dtype_is_block_scaled(to_dtype)) {
        nk_block_scaled_format_t const from_format = nk_block_scaled_format_of_dtype(from_dtype);
        nk_block_scaled_format_t const to_format = nk_block_scaled_format_of_dtype(to_dtype);
        return nk_cast_block_scaled_launch_rocm_((void const *)&nk_cast_block_scaled_rocm_kernel_,
                                                 nk_cast_operand_serial_(from_dtype, from), &from_format,
                                                 nk_cast_operand_serial_(to_dtype, to), &to_format, count, stream);
    }
    if (((nk_size_t)from & (nk_dtype_alignment_serial_(from_dtype) - 1)) ||
        ((nk_size_t)to & (nk_dtype_alignment_serial_(to_dtype) - 1)))
        return nk_misaligned_k;
    arguments.from = (unsigned char const *)from, arguments.to = (unsigned char *)to;
    if (!nk_cast_plan_serial_(from_dtype, to_dtype, count, (nk_size_t)from | (nk_size_t)to, &arguments.plan))
        return nk_success_k;
    void const *const kernel = nk_cast_plan_vectors_simt_(&arguments) ? (void const *)&nk_cast_vectors_rocm_kernel_
                                                                      : (void const *)&nk_cast_rocm_kernel_;
    return nk_launch_resident_rocm_(kernel, nk_cast_threads_simt_k, 0, 0,
                                    nk_size_divide_round_up_(arguments.plan.units, nk_cast_threads_simt_k), &arguments,
                                    stream);
}

#endif // NUMKONG_TARGET_ROCM

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_ROCM_
#endif // NUMKONG_CAST_ROCM_CUH
