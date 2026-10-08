/**
 *  @file include/numkong/cast/cuda.cuh
 *  @author Ash Vardanian
 *  @date October 3, 2026
 *  @brief CUDA host side of bulk casts: the plain and block-scaled launches and the @c cuda export,
 *      over the conversions of `cast/simt.cuh`.
 *
 *  The @c ampere and @c ada exports launch through the same functions, with vector kernels that
 *  use their conversion instructions.
 *
 *  @sa include/numkong/cast/simt.cuh
 *  @sa include/numkong/cast/ampere.cuh
 *  @sa include/numkong/cast/ada.cuh
 */
#ifndef NUMKONG_CAST_CUDA_CUH
#define NUMKONG_CAST_CUDA_CUH

#if NUMKONG_ARCH_CUDA_

#include "numkong/cuda.cuh"
#include "numkong/cast/simt.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

/** Validates a cast with a block-scaled side and queues its launches of @p kernel: a destination
 *  tensor scale adds the mark, abs-max and derive launches before the chunks. */
NUMKONG_INLINE nk_status_t nk_cast_block_scaled_launch_cuda_(void const *kernel, nk_cast_operand_t const &from,
                                                             nk_block_scaled_format_t const *from_format,
                                                             nk_cast_operand_t const &to,
                                                             nk_block_scaled_format_t const *to_format, nk_size_t count,
                                                             void *stream) {
    int const from_plain = from_format->scale_dtype == nk_dtype_unknown_k || from_format->block_size == 0;
    int const to_plain = to_format->scale_dtype == nk_dtype_unknown_k || to_format->block_size == 0;
    nk_cast_block_scaled_arguments_t arguments;
    if ((!from_plain && from_format->block_size > 32) || (!to_plain && to_format->block_size > 32))
        return nk_unexpected_dimensions_k;
    if (((nk_size_t)from.codes & (nk_dtype_alignment_(from_format->element_dtype) - 1)) ||
        ((nk_size_t)to.codes & (nk_dtype_alignment_(to_format->element_dtype) - 1)))
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
        status = nk_launch_resident_cuda_(kernel, 256, 0, 0, nk_size_divide_round_up_(count, 256), &arguments, stream);
    }
    if (status != nk_success_k) return status;
    arguments.phase = nk_cast_block_scaled_chunks_k;
    // One block at least, so an empty cast still writes the tensor scale it derives
    nk_size_t const blocks = nk_size_divide_round_up_(arguments.chunks, 256);
    return nk_launch_resident_cuda_(kernel, 256, 0, 0, blocks ? blocks : 1, &arguments, stream);
}

/** Validates and launches a cast with the kernels of one capability: @p block_scaled_kernel when a
 *  side is block-scaled, else @p vectors_kernel where the cast has a vector path, else @p kernel,
 *  as many 256-thread blocks as stay resident walking the units. */
NUMKONG_INLINE nk_status_t nk_cast_launch_cuda_(void const *kernel, void const *vectors_kernel,
                                                void const *block_scaled_kernel, void const *from,
                                                nk_dtype_t from_dtype, void *to, nk_dtype_t to_dtype, nk_size_t count,
                                                void *stream) {
    nk_cast_arguments_t arguments;
    if (nk_dtype_is_block_scaled(from_dtype) || nk_dtype_is_block_scaled(to_dtype)) {
        nk_block_scaled_format_t const from_format = nk_block_scaled_format_of_dtype(from_dtype);
        nk_block_scaled_format_t const to_format = nk_block_scaled_format_of_dtype(to_dtype);
        return nk_cast_block_scaled_launch_cuda_(block_scaled_kernel, nk_cast_operand_(from_dtype, from), &from_format,
                                                 nk_cast_operand_(to_dtype, to), &to_format, count, stream);
    }
    if (((nk_size_t)from & (nk_dtype_alignment_(from_dtype) - 1)) ||
        ((nk_size_t)to & (nk_dtype_alignment_(to_dtype) - 1)))
        return nk_misaligned_k;
    if (!nk_cast_plan_simt_(from, from_dtype, count, to, to_dtype, &arguments)) return nk_success_k;
    if (nk_cast_plan_vectors_simt_(&arguments)) kernel = vectors_kernel;
    return nk_launch_resident_cuda_(kernel, 256, 0, 0, nk_size_divide_round_up_(arguments.units, 256), &arguments,
                                    stream);
}

#if NUMKONG_TARGET_CUDA

static __global__ void nk_cast_cuda_kernel_(nk_cast_arguments_t arguments) {
    nk_size_t const stride = (nk_size_t)gridDim.x * blockDim.x;
    for (nk_size_t unit = (nk_size_t)blockIdx.x * blockDim.x + threadIdx.x; unit < arguments.units; unit += stride)
        nk_cast_unit_simt_(&arguments, unit);
}

static __global__ void nk_cast_vectors_cuda_kernel_(nk_cast_arguments_t arguments) {
    nk_cast_vectors_simt_(&arguments);
}

static __global__ void nk_cast_block_scaled_cuda_kernel_(nk_cast_block_scaled_arguments_t arguments) {
    __shared__ nk_u32_t partials[256];
    nk_cast_block_scaled_phase_simt_(&arguments, partials);
}

NUMKONG_API nk_status_t nk_cast_cuda(void const *from, nk_dtype_t from_dtype, void *to, nk_dtype_t to_dtype,
                                     nk_size_t count, void *stream) {
    return nk_cast_launch_cuda_((void const *)&nk_cast_cuda_kernel_, (void const *)&nk_cast_vectors_cuda_kernel_,
                                (void const *)&nk_cast_block_scaled_cuda_kernel_, from, from_dtype, to, to_dtype, count,
                                stream);
}

#endif // NUMKONG_TARGET_CUDA

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_
#endif // NUMKONG_CAST_CUDA_CUH
