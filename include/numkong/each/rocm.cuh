/**
 *  @file include/numkong/each/rocm.cuh
 *  @author Ash Vardanian
 *  @date October 3, 2026
 *  @brief ROCm host side of element-wise sums, RMSNorm and SwiGLU: the launches, the export
 *      generators and the @c rocm exports, over the kernels of `each/simt.cuh`.
 *
 *  @sa include/numkong/each/simt.cuh
 *  @sa include/numkong/each/cuda.cuh
 */
#ifndef NUMKONG_EACH_ROCM_CUH
#define NUMKONG_EACH_ROCM_CUH

#include "numkong/rocm.cuh"
#include "numkong/each/simt.cuh"

#if NUMKONG_ARCH_ROCM_ && defined(__HIP__)

#if defined(__cplusplus)
extern "C" {
#endif

/** Launches enough blocks of @p kernel for @p count elements, at most 2²⁰ of them, passing the one
 *  argument struct at @p arguments by value. */
NUMKONG_INLINE nk_status_t nk_each_launch_rocm_(void const *kernel, nk_size_t count, void *arguments, void *stream) {
    if (count == 0) return nk_success_k;
    nk_size_t const blocks = nk_size_divide_round_up_(count, nk_each_threads_simt_k), blocks_limit = (nk_size_t)1 << 20;
    void *launch_arguments[1];
    launch_arguments[0] = arguments;
    return nk_launch_rocm_(kernel, blocks < blocks_limit ? blocks : blocks_limit, nk_each_threads_simt_k,
                           launch_arguments, 0, stream);
}

NUMKONG_INLINE nk_status_t nk_each_swiglu_launch_rocm_(void const *kernel, nk_size_t value_bytes, void const *gate,
                                                       void const *up, void *y, nk_size_t rows, nk_size_t columns,
                                                       nk_size_t gate_stride, nk_size_t up_stride, nk_size_t y_stride,
                                                       nk_f32_t gate_scale, nk_f32_t output_scale, void *stream) {
    if ((((nk_size_t)gate) | gate_stride | ((nk_size_t)up) | (up ? up_stride : 0) | ((nk_size_t)y) | y_stride) &
        (value_bytes - 1))
        return nk_misaligned_k;
    nk_each_swiglu_arguments_t arguments;
    arguments.gate = (unsigned char const *)gate, arguments.up = (unsigned char const *)up;
    arguments.y = (unsigned char *)y, arguments.columns = columns, arguments.count = rows * columns;
    arguments.gate_stride = gate_stride, arguments.up_stride = up_stride, arguments.y_stride = y_stride;
    arguments.gate_scale = gate_scale, arguments.output_scale = output_scale;
    return nk_each_launch_rocm_(kernel, arguments.count, &arguments, stream);
}

/** Validates the contract, @p value_bytes alignment for the vectors and @p scale_bytes for @p alpha
 *  and @p beta, and launches @p kernel over @p n elements. */
NUMKONG_INLINE nk_status_t nk_each_elementwise_launch_rocm_(void const *kernel, nk_size_t value_bytes,
                                                            nk_size_t scale_bytes, void const *a, void const *b,
                                                            void const *c, void const *alpha, void const *beta,
                                                            nk_size_t n, void *result, void *stream) {
    if ((((nk_size_t)a) | ((nk_size_t)b) | ((nk_size_t)c) | ((nk_size_t)result)) & (value_bytes - 1) ||
        (((nk_size_t)alpha) | ((nk_size_t)beta)) & (scale_bytes - 1))
        return nk_misaligned_k;
    nk_each_elementwise_arguments_t arguments;
    arguments.a = (unsigned char const *)a, arguments.b = (unsigned char const *)b;
    arguments.c = (unsigned char const *)c, arguments.alpha = alpha, arguments.beta = beta;
    arguments.result = (unsigned char *)result, arguments.count = n;
    return nk_each_launch_rocm_(kernel, n, &arguments, stream);
}

/** Validates the contract and launches one block per normalized vector, at most 2³⁰ of them. */
NUMKONG_INLINE nk_status_t nk_each_rmsnorm_launch_rocm_(void const *kernel, nk_size_t value_bytes, void const *x,
                                                        nk_f32_t const *gamma, void *y, nk_size_t rows,
                                                        nk_size_t groups, nk_size_t columns, nk_size_t x_stride,
                                                        nk_size_t y_stride, nk_f32_t epsilon, void *stream) {
    if ((((nk_size_t)x) | x_stride | ((nk_size_t)y) | y_stride) & (value_bytes - 1) || ((nk_size_t)gamma & 3))
        return nk_misaligned_k;
    nk_each_rmsnorm_arguments_t arguments;
    arguments.x = (unsigned char const *)x, arguments.gamma = gamma, arguments.y = (unsigned char *)y;
    arguments.groups = groups, arguments.columns = columns, arguments.vectors = rows * groups;
    arguments.x_stride = x_stride, arguments.y_stride = y_stride, arguments.epsilon = epsilon;
    if (arguments.vectors == 0 || columns == 0) return nk_success_k;
    nk_size_t const blocks_limit = (nk_size_t)1 << 30;
    void *launch_arguments[1];
    launch_arguments[0] = &arguments;
    return nk_launch_rocm_(kernel, arguments.vectors < blocks_limit ? arguments.vectors : blocks_limit,
                           nk_each_threads_simt_k, launch_arguments, 0, stream);
}

/** Generates the SwiGLU entry point of @p input_type over the kernel of
 *  @c nk_define_each_swiglu_kernel_simt_. */
#define nk_define_each_swiglu_rocm_(input_type, isa_suffix, load_and_convert, convert_and_store)                   \
    nk_define_each_swiglu_kernel_simt_(input_type, isa_suffix, load_and_convert, convert_and_store)                \
        NUMKONG_API nk_status_t                                                                                    \
        nk_each_swiglu_##input_type##_##isa_suffix(nk_##input_type##_t const *gate, nk_##input_type##_t const *up, \
                                                   nk_##input_type##_t *y, nk_size_t rows, nk_size_t columns,      \
                                                   nk_size_t gate_stride, nk_size_t up_stride, nk_size_t y_stride, \
                                                   nk_f32_t gate_scale, nk_f32_t output_scale, void *stream) {     \
        return nk_each_swiglu_launch_rocm_((void const *)&nk_each_swiglu_##input_type##_##isa_suffix##_kernel_,    \
                                           sizeof(nk_##input_type##_t), gate, up, y, rows, columns, gate_stride,   \
                                           up_stride, y_stride, gate_scale, output_scale, stream);                 \
    }

/** Generates the element-wise sum entry point of @p input_type over the kernel of
 *  @c nk_define_each_sum_kernel_simt_. */
#define nk_define_each_sum_rocm_(input_type, accumulator_type, isa_suffix, load_and_convert, convert_and_store)    \
    nk_define_each_sum_kernel_simt_(input_type, accumulator_type, isa_suffix, load_and_convert, convert_and_store) \
        NUMKONG_API nk_status_t                                                                                    \
        nk_each_sum_##input_type##_##isa_suffix(nk_##input_type##_t const *a, nk_##input_type##_t const *b,        \
                                                nk_size_t n, nk_##input_type##_t *result, void *stream) {          \
        return nk_each_elementwise_launch_rocm_((void const *)&nk_each_sum_##input_type##_##isa_suffix##_kernel_,  \
                                                sizeof(nk_##input_type##_t), 1, a, b, NUMKONG_NULL, NUMKONG_NULL,  \
                                                NUMKONG_NULL, n, result, stream);                                  \
    }

/** Generates the RMSNorm entry point of @p input_type over the kernel of
 *  @c nk_define_each_rmsnorm_kernel_simt_. */
#define nk_define_each_rmsnorm_rocm_(input_type, isa_suffix, load_and_convert, convert_and_store)                      \
    nk_define_each_rmsnorm_kernel_simt_(input_type, isa_suffix, load_and_convert, convert_and_store)                   \
        NUMKONG_API nk_status_t                                                                                        \
        nk_each_rmsnorm_##input_type##_##isa_suffix(nk_##input_type##_t const *x, nk_f32_t const *gamma,               \
                                                    nk_##input_type##_t *y, nk_size_t rows, nk_size_t groups,          \
                                                    nk_size_t columns, nk_size_t x_stride, nk_size_t y_stride,         \
                                                    nk_f32_t epsilon, void *stream) {                                  \
        return nk_each_rmsnorm_launch_rocm_((void const *)&nk_each_rmsnorm_##input_type##_##isa_suffix##_kernel_,      \
                                            sizeof(nk_##input_type##_t), x, gamma, y, rows, groups, columns, x_stride, \
                                            y_stride, epsilon, stream);                                                \
    }

#if NUMKONG_TARGET_ROCM
nk_define_each_sum_rocm_(f32, f32, rocm, nk_assign_from_to_, nk_assign_from_to_)
nk_define_each_sum_rocm_(f16, f32, rocm, nk_f16_to_f32_simt_, nk_f32_to_f16_simt_)
nk_define_each_sum_rocm_(bf16, f32, rocm, nk_bf16_to_f32_simt_, nk_f32_to_bf16_simt_)
nk_define_each_rmsnorm_rocm_(f32, rocm, nk_assign_from_to_, nk_assign_from_to_)
nk_define_each_rmsnorm_rocm_(bf16, rocm, nk_bf16_to_f32_simt_, nk_f32_to_bf16_simt_)
nk_define_each_rmsnorm_rocm_(e4m3, rocm, nk_e4m3_to_f32_simt_, nk_f32_to_e4m3_simt_)
nk_define_each_swiglu_rocm_(f32, rocm, nk_assign_from_to_, nk_assign_from_to_)
nk_define_each_swiglu_rocm_(bf16, rocm, nk_bf16_to_f32_simt_, nk_f32_to_bf16_simt_)
nk_define_each_swiglu_rocm_(e4m3, rocm, nk_e4m3_to_f32_simt_, nk_f32_to_e4m3_simt_)
#endif // NUMKONG_TARGET_ROCM

#undef nk_define_each_sum_rocm_
#undef nk_define_each_rmsnorm_rocm_
#undef nk_define_each_swiglu_rocm_

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_ROCM_ && defined(__HIP__)
#endif // NUMKONG_EACH_ROCM_CUH
