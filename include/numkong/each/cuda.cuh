/**
 *  @file include/numkong/each/cuda.cuh
 *  @author Ash Vardanian
 *  @date October 3, 2026
 *  @brief CUDA host side of element-wise scales, sums, blends, FMAs, RMSNorm and SwiGLU: the
 *      launches, the export generators and the @c cuda exports, over the kernels of `each/simt.cuh`,
 *      with the integer and complex kernels only NVIDIA runs.
 *
 *  @sa include/numkong/each/simt.cuh
 *  @sa include/numkong/each/rocm.cuh
 */
#ifndef NUMKONG_EACH_CUDA_CUH
#define NUMKONG_EACH_CUDA_CUH

#include "numkong/cuda.cuh"
#include "numkong/each/simt.cuh"

#if NUMKONG_ARCH_CUDA_ && defined(__CUDACC__) && !defined(__HIP__)

#if defined(__cplusplus)
extern "C" {
#endif

/** Launches enough blocks of @p kernel for @p count elements, at most 2²⁰ of them, passing the one
 *  argument struct at @p arguments by value. */
NUMKONG_INLINE nk_status_t nk_each_launch_cuda_(void const *kernel, nk_size_t count, void *arguments, void *stream) {
    if (count == 0) return nk_success_k;
    nk_size_t const blocks = nk_size_divide_round_up_(count, nk_each_threads_simt_k), blocks_limit = (nk_size_t)1 << 20;
    void *launch_arguments[1];
    launch_arguments[0] = arguments;
    return nk_launch_cuda_(kernel, blocks < blocks_limit ? blocks : blocks_limit, nk_each_threads_simt_k,
                           launch_arguments, 0, stream);
}

NUMKONG_INLINE nk_status_t nk_each_swiglu_launch_cuda_(void const *kernel, nk_size_t value_bytes, void const *gate,
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
    return nk_each_launch_cuda_(kernel, arguments.count, &arguments, stream);
}

/** Validates the contract, @p value_bytes alignment for the vectors and @p scale_bytes for @p alpha
 *  and @p beta, and launches @p kernel over @p n elements. */
NUMKONG_INLINE nk_status_t nk_each_elementwise_launch_cuda_(void const *kernel, nk_size_t value_bytes,
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
    return nk_each_launch_cuda_(kernel, n, &arguments, stream);
}

/** Validates the contract and launches one block per normalized vector, at most 2³⁰ of them. */
NUMKONG_INLINE nk_status_t nk_each_rmsnorm_launch_cuda_(void const *kernel, nk_size_t value_bytes, void const *x,
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
    return nk_launch_cuda_(kernel, arguments.vectors < blocks_limit ? arguments.vectors : blocks_limit,
                           nk_each_threads_simt_k, launch_arguments, 0, stream);
}

/** Generates the SwiGLU entry point of @p input_type over the kernel of
 *  @c nk_define_each_swiglu_kernel_simt_. */
#define nk_define_each_swiglu_cuda_(input_type, isa_suffix, load_and_convert, convert_and_store)                   \
    nk_define_each_swiglu_kernel_simt_(input_type, isa_suffix, load_and_convert, convert_and_store)                \
        NUMKONG_API nk_status_t                                                                                    \
        nk_each_swiglu_##input_type##_##isa_suffix(nk_##input_type##_t const *gate, nk_##input_type##_t const *up, \
                                                   nk_##input_type##_t *y, nk_size_t rows, nk_size_t columns,      \
                                                   nk_size_t gate_stride, nk_size_t up_stride, nk_size_t y_stride, \
                                                   nk_f32_t gate_scale, nk_f32_t output_scale, void *stream) {     \
        return nk_each_swiglu_launch_cuda_((void const *)&nk_each_swiglu_##input_type##_##isa_suffix##_kernel_,    \
                                           sizeof(nk_##input_type##_t), gate, up, y, rows, columns, gate_stride,   \
                                           up_stride, y_stride, gate_scale, output_scale, stream);                 \
    }

/** Generates the element-wise sum entry point of @p input_type over the kernel of
 *  @c nk_define_each_sum_kernel_simt_. */
#define nk_define_each_sum_cuda_(input_type, accumulator_type, isa_suffix, load_and_convert, convert_and_store)    \
    nk_define_each_sum_kernel_simt_(input_type, accumulator_type, isa_suffix, load_and_convert, convert_and_store) \
        NUMKONG_API nk_status_t                                                                                    \
        nk_each_sum_##input_type##_##isa_suffix(nk_##input_type##_t const *a, nk_##input_type##_t const *b,        \
                                                nk_size_t n, nk_##input_type##_t *result, void *stream) {          \
        return nk_each_elementwise_launch_cuda_((void const *)&nk_each_sum_##input_type##_##isa_suffix##_kernel_,  \
                                                sizeof(nk_##input_type##_t), 1, a, b, NUMKONG_NULL, NUMKONG_NULL,  \
                                                NUMKONG_NULL, n, result, stream);                                  \
    }

/** Generates the element-wise scale entry point of @p input_type over the kernel of
 *  @c nk_define_each_scale_kernel_simt_. */
#define nk_define_each_scale_cuda_(input_type, accumulator_type, isa_suffix, load_and_convert, convert_and_store)    \
    nk_define_each_scale_kernel_simt_(input_type, accumulator_type, isa_suffix, load_and_convert, convert_and_store) \
        NUMKONG_API nk_status_t                                                                                      \
        nk_each_scale_##input_type##_##isa_suffix(                                                                   \
            nk_##input_type##_t const *a, nk_size_t n, nk_##accumulator_type##_t const *alpha,                       \
            nk_##accumulator_type##_t const *beta, nk_##input_type##_t *result, void *stream) {                      \
        return nk_each_elementwise_launch_cuda_((void const *)&nk_each_scale_##input_type##_##isa_suffix##_kernel_,  \
                                                sizeof(nk_##input_type##_t), sizeof(nk_##accumulator_type##_t), a,   \
                                                NUMKONG_NULL, NUMKONG_NULL, alpha, beta, n, result, stream);         \
    }

/** Generates the element-wise blend entry point of @p input_type over the kernel of
 *  @c nk_define_each_blend_kernel_simt_. */
#define nk_define_each_blend_cuda_(input_type, accumulator_type, isa_suffix, load_and_convert, convert_and_store)     \
    nk_define_each_blend_kernel_simt_(input_type, accumulator_type, isa_suffix, load_and_convert, convert_and_store)  \
        NUMKONG_API nk_status_t                                                                                       \
        nk_each_blend_##input_type##_##isa_suffix(nk_##input_type##_t const *a, nk_##input_type##_t const *b,         \
                                                  nk_size_t n, nk_##accumulator_type##_t const *alpha,                \
                                                  nk_##accumulator_type##_t const *beta, nk_##input_type##_t *result, \
                                                  void *stream) {                                                     \
        return nk_each_elementwise_launch_cuda_((void const *)&nk_each_blend_##input_type##_##isa_suffix##_kernel_,   \
                                                sizeof(nk_##input_type##_t), sizeof(nk_##accumulator_type##_t), a, b, \
                                                NUMKONG_NULL, alpha, beta, n, result, stream);                        \
    }

/** Generates the element-wise FMA entry point of @p input_type over the kernel of
 *  @c nk_define_each_fma_kernel_simt_. */
#define nk_define_each_fma_cuda_(input_type, accumulator_type, isa_suffix, load_and_convert, convert_and_store)       \
    nk_define_each_fma_kernel_simt_(input_type, accumulator_type, isa_suffix, load_and_convert, convert_and_store)    \
        NUMKONG_API nk_status_t                                                                                       \
        nk_each_fma_##input_type##_##isa_suffix(                                                                      \
            nk_##input_type##_t const *a, nk_##input_type##_t const *b, nk_##input_type##_t const *c, nk_size_t n,    \
            nk_##accumulator_type##_t const *alpha, nk_##accumulator_type##_t const *beta,                            \
            nk_##input_type##_t *result, void *stream) {                                                              \
        return nk_each_elementwise_launch_cuda_((void const *)&nk_each_fma_##input_type##_##isa_suffix##_kernel_,     \
                                                sizeof(nk_##input_type##_t), sizeof(nk_##accumulator_type##_t), a, b, \
                                                c, alpha, beta, n, result, stream);                                   \
    }

/** Generates the RMSNorm entry point of @p input_type over the kernel of
 *  @c nk_define_each_rmsnorm_kernel_simt_. */
#define nk_define_each_rmsnorm_cuda_(input_type, isa_suffix, load_and_convert, convert_and_store)                      \
    nk_define_each_rmsnorm_kernel_simt_(input_type, isa_suffix, load_and_convert, convert_and_store)                   \
        NUMKONG_API nk_status_t                                                                                        \
        nk_each_rmsnorm_##input_type##_##isa_suffix(nk_##input_type##_t const *x, nk_f32_t const *gamma,               \
                                                    nk_##input_type##_t *y, nk_size_t rows, nk_size_t groups,          \
                                                    nk_size_t columns, nk_size_t x_stride, nk_size_t y_stride,         \
                                                    nk_f32_t epsilon, void *stream) {                                  \
        return nk_each_rmsnorm_launch_cuda_((void const *)&nk_each_rmsnorm_##input_type##_##isa_suffix##_kernel_,      \
                                            sizeof(nk_##input_type##_t), x, gamma, y, rows, groups, columns, x_stride, \
                                            y_stride, epsilon, stream);                                                \
    }

#if NUMKONG_TARGET_CUDA
nk_define_each_sum_cuda_(f64, f64, cuda, nk_assign_from_to_, nk_assign_from_to_)
nk_define_each_sum_cuda_(f32, f32, cuda, nk_assign_from_to_, nk_assign_from_to_)
nk_define_each_sum_cuda_(f16, f32, cuda, nk_f16_to_f32_simt_, nk_f32_to_f16_simt_)
nk_define_each_sum_cuda_(bf16, f32, cuda, nk_bf16_to_f32_simt_, nk_f32_to_bf16_simt_)
nk_define_each_sum_cuda_(e4m3, f32, cuda, nk_e4m3_to_f32_simt_, nk_f32_to_e4m3_simt_)
nk_define_each_sum_cuda_(e5m2, f32, cuda, nk_e5m2_to_f32_simt_, nk_f32_to_e5m2_simt_)
nk_define_each_sum_cuda_(e2m3, f32, cuda, nk_e2m3_to_f32_simt_, nk_f32_to_e2m3_simt_)
nk_define_each_sum_cuda_(e3m2, f32, cuda, nk_e3m2_to_f32_simt_, nk_f32_to_e3m2_simt_)
nk_define_each_sum_cuda_(i8, i64, cuda, nk_assign_from_to_, nk_i64_to_i8_serial_)
nk_define_each_sum_cuda_(u8, i64, cuda, nk_assign_from_to_, nk_i64_to_u8_serial_)
nk_define_each_sum_cuda_(i16, i64, cuda, nk_assign_from_to_, nk_i64_to_i16_serial_)
nk_define_each_sum_cuda_(u16, i64, cuda, nk_assign_from_to_, nk_i64_to_u16_serial_)
nk_define_each_sum_cuda_(i32, i64, cuda, nk_assign_from_to_, nk_i64_to_i32_serial_)
nk_define_each_sum_cuda_(u32, i64, cuda, nk_assign_from_to_, nk_i64_to_u32_serial_)

static __global__ void nk_each_sum_i64_cuda_kernel_(nk_each_elementwise_arguments_t arguments) {
    nk_i64_t const *a = (nk_i64_t const *)arguments.a, *b = (nk_i64_t const *)arguments.b;
    nk_size_t const stride = (nk_size_t)gridDim.x * blockDim.x;
    for (nk_size_t cell = (nk_size_t)blockIdx.x * blockDim.x + threadIdx.x; cell < arguments.count; cell += stride)
        ((nk_i64_t *)arguments.result)[cell] = nk_i64_saturating_add_(a[cell], b[cell]);
}

NUMKONG_API nk_status_t nk_each_sum_i64_cuda(nk_i64_t const *a, nk_i64_t const *b, nk_size_t n, nk_i64_t *result,
                                             void *stream) {
    return nk_each_elementwise_launch_cuda_((void const *)&nk_each_sum_i64_cuda_kernel_, sizeof(nk_i64_t), 1, a, b,
                                            NUMKONG_NULL, NUMKONG_NULL, NUMKONG_NULL, n, result, stream);
}

static __global__ void nk_each_sum_u64_cuda_kernel_(nk_each_elementwise_arguments_t arguments) {
    nk_u64_t const *a = (nk_u64_t const *)arguments.a, *b = (nk_u64_t const *)arguments.b;
    nk_size_t const stride = (nk_size_t)gridDim.x * blockDim.x;
    for (nk_size_t cell = (nk_size_t)blockIdx.x * blockDim.x + threadIdx.x; cell < arguments.count; cell += stride)
        ((nk_u64_t *)arguments.result)[cell] = nk_u64_saturating_add_(a[cell], b[cell]);
}

NUMKONG_API nk_status_t nk_each_sum_u64_cuda(nk_u64_t const *a, nk_u64_t const *b, nk_size_t n, nk_u64_t *result,
                                             void *stream) {
    return nk_each_elementwise_launch_cuda_((void const *)&nk_each_sum_u64_cuda_kernel_, sizeof(nk_u64_t), 1, a, b,
                                            NUMKONG_NULL, NUMKONG_NULL, NUMKONG_NULL, n, result, stream);
}

nk_define_each_scale_cuda_(f64, f64, cuda, nk_assign_from_to_, nk_assign_from_to_)
nk_define_each_scale_cuda_(f32, f32, cuda, nk_assign_from_to_, nk_assign_from_to_)
nk_define_each_scale_cuda_(f16, f32, cuda, nk_f16_to_f32_simt_, nk_f32_to_f16_simt_)
nk_define_each_scale_cuda_(bf16, f32, cuda, nk_bf16_to_f32_simt_, nk_f32_to_bf16_simt_)
nk_define_each_scale_cuda_(e4m3, f32, cuda, nk_e4m3_to_f32_simt_, nk_f32_to_e4m3_simt_)
nk_define_each_scale_cuda_(e5m2, f32, cuda, nk_e5m2_to_f32_simt_, nk_f32_to_e5m2_simt_)
nk_define_each_scale_cuda_(e2m3, f32, cuda, nk_e2m3_to_f32_simt_, nk_f32_to_e2m3_simt_)
nk_define_each_scale_cuda_(e3m2, f32, cuda, nk_e3m2_to_f32_simt_, nk_f32_to_e3m2_simt_)
nk_define_each_scale_cuda_(i8, f32, cuda, nk_assign_from_to_, nk_f32_to_i8_simt_)
nk_define_each_scale_cuda_(u8, f32, cuda, nk_assign_from_to_, nk_f32_to_u8_simt_)
nk_define_each_scale_cuda_(i16, f32, cuda, nk_assign_from_to_, nk_f32_to_i16_simt_)
nk_define_each_scale_cuda_(u16, f32, cuda, nk_assign_from_to_, nk_f32_to_u16_simt_)
nk_define_each_scale_cuda_(i32, f64, cuda, nk_assign_from_to_, nk_f64_to_i32_simt_)
nk_define_each_scale_cuda_(u32, f64, cuda, nk_assign_from_to_, nk_f64_to_u32_simt_)
nk_define_each_scale_cuda_(i64, f64, cuda, nk_assign_from_to_, nk_f64_to_i64_simt_)
nk_define_each_scale_cuda_(u64, f64, cuda, nk_assign_from_to_, nk_f64_to_u64_simt_)
nk_define_each_blend_cuda_(f64, f64, cuda, nk_assign_from_to_, nk_assign_from_to_)
nk_define_each_blend_cuda_(f32, f32, cuda, nk_assign_from_to_, nk_assign_from_to_)
nk_define_each_blend_cuda_(f16, f32, cuda, nk_f16_to_f32_simt_, nk_f32_to_f16_simt_)
nk_define_each_blend_cuda_(bf16, f32, cuda, nk_bf16_to_f32_simt_, nk_f32_to_bf16_simt_)
nk_define_each_blend_cuda_(e4m3, f32, cuda, nk_e4m3_to_f32_simt_, nk_f32_to_e4m3_simt_)
nk_define_each_blend_cuda_(e5m2, f32, cuda, nk_e5m2_to_f32_simt_, nk_f32_to_e5m2_simt_)
nk_define_each_blend_cuda_(e2m3, f32, cuda, nk_e2m3_to_f32_simt_, nk_f32_to_e2m3_simt_)
nk_define_each_blend_cuda_(e3m2, f32, cuda, nk_e3m2_to_f32_simt_, nk_f32_to_e3m2_simt_)
nk_define_each_blend_cuda_(i8, f32, cuda, nk_assign_from_to_, nk_f32_to_i8_simt_)
nk_define_each_blend_cuda_(u8, f32, cuda, nk_assign_from_to_, nk_f32_to_u8_simt_)
nk_define_each_blend_cuda_(i16, f32, cuda, nk_assign_from_to_, nk_f32_to_i16_simt_)
nk_define_each_blend_cuda_(u16, f32, cuda, nk_assign_from_to_, nk_f32_to_u16_simt_)
nk_define_each_blend_cuda_(i32, f64, cuda, nk_assign_from_to_, nk_f64_to_i32_simt_)
nk_define_each_blend_cuda_(u32, f64, cuda, nk_assign_from_to_, nk_f64_to_u32_simt_)
nk_define_each_blend_cuda_(i64, f64, cuda, nk_assign_from_to_, nk_f64_to_i64_simt_)
nk_define_each_blend_cuda_(u64, f64, cuda, nk_assign_from_to_, nk_f64_to_u64_simt_)
nk_define_each_fma_cuda_(f64, f64, cuda, nk_assign_from_to_, nk_assign_from_to_)
nk_define_each_fma_cuda_(f32, f32, cuda, nk_assign_from_to_, nk_assign_from_to_)
nk_define_each_fma_cuda_(f16, f32, cuda, nk_f16_to_f32_simt_, nk_f32_to_f16_simt_)
nk_define_each_fma_cuda_(bf16, f32, cuda, nk_bf16_to_f32_simt_, nk_f32_to_bf16_simt_)
nk_define_each_fma_cuda_(e4m3, f32, cuda, nk_e4m3_to_f32_simt_, nk_f32_to_e4m3_simt_)
nk_define_each_fma_cuda_(e5m2, f32, cuda, nk_e5m2_to_f32_simt_, nk_f32_to_e5m2_simt_)
nk_define_each_fma_cuda_(e2m3, f32, cuda, nk_e2m3_to_f32_simt_, nk_f32_to_e2m3_simt_)
nk_define_each_fma_cuda_(e3m2, f32, cuda, nk_e3m2_to_f32_simt_, nk_f32_to_e3m2_simt_)
nk_define_each_fma_cuda_(i8, f32, cuda, nk_assign_from_to_, nk_f32_to_i8_simt_)
nk_define_each_fma_cuda_(u8, f32, cuda, nk_assign_from_to_, nk_f32_to_u8_simt_)
nk_define_each_fma_cuda_(i16, f32, cuda, nk_assign_from_to_, nk_f32_to_i16_simt_)
nk_define_each_fma_cuda_(u16, f32, cuda, nk_assign_from_to_, nk_f32_to_u16_simt_)
nk_define_each_fma_cuda_(i32, f64, cuda, nk_assign_from_to_, nk_f64_to_i32_simt_)
nk_define_each_fma_cuda_(u32, f64, cuda, nk_assign_from_to_, nk_f64_to_u32_simt_)
nk_define_each_fma_cuda_(i64, f64, cuda, nk_assign_from_to_, nk_f64_to_i64_simt_)
nk_define_each_fma_cuda_(u64, f64, cuda, nk_assign_from_to_, nk_f64_to_u64_simt_)
NUMKONG_API nk_status_t nk_each_sum_f32c_cuda(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f32c_t *result,
                                              void *stream) {
    return nk_each_elementwise_launch_cuda_((void const *)&nk_each_sum_f32_cuda_kernel_, sizeof(nk_f32_t), 1, a, b,
                                            NUMKONG_NULL, NUMKONG_NULL, NUMKONG_NULL, 2 * n, result, stream);
}

static __global__ void nk_each_scale_f32c_cuda_kernel_(nk_each_elementwise_arguments_t arguments) {
    nk_f32c_t const *a = (nk_f32c_t const *)arguments.a, *alpha = (nk_f32c_t const *)arguments.alpha;
    nk_f32c_t const *beta = (nk_f32c_t const *)arguments.beta;
    nk_f32c_t *result = (nk_f32c_t *)arguments.result;
    nk_f32_t alpha_real = alpha->real, alpha_imag = alpha->imag;
    nk_f32_t beta_real = beta->real, beta_imag = beta->imag;
    nk_size_t const stride = (nk_size_t)gridDim.x * blockDim.x;
    for (nk_size_t i = (nk_size_t)blockIdx.x * blockDim.x + threadIdx.x; i < arguments.count; i += stride) {
        nk_f32_t a_real = a[i].real, a_imag = a[i].imag;
        result[i].real = alpha_real * a_real - alpha_imag * a_imag + beta_real;
        result[i].imag = alpha_real * a_imag + alpha_imag * a_real + beta_imag;
    }
}

NUMKONG_API nk_status_t nk_each_scale_f32c_cuda(nk_f32c_t const *a, nk_size_t n, nk_f32c_t const *alpha,
                                                nk_f32c_t const *beta, nk_f32c_t *result, void *stream) {
    return nk_each_elementwise_launch_cuda_((void const *)&nk_each_scale_f32c_cuda_kernel_, sizeof(nk_f32_t),
                                            sizeof(nk_f32_t), a, NUMKONG_NULL, NUMKONG_NULL, alpha, beta, n, result,
                                            stream);
}

static __global__ void nk_each_blend_f32c_cuda_kernel_(nk_each_elementwise_arguments_t arguments) {
    nk_f32c_t const *a = (nk_f32c_t const *)arguments.a, *b = (nk_f32c_t const *)arguments.b;
    nk_f32c_t const *alpha = (nk_f32c_t const *)arguments.alpha, *beta = (nk_f32c_t const *)arguments.beta;
    nk_f32c_t *result = (nk_f32c_t *)arguments.result;
    nk_f32_t alpha_real = alpha->real, alpha_imag = alpha->imag;
    nk_f32_t beta_real = beta->real, beta_imag = beta->imag;
    nk_size_t const stride = (nk_size_t)gridDim.x * blockDim.x;
    for (nk_size_t i = (nk_size_t)blockIdx.x * blockDim.x + threadIdx.x; i < arguments.count; i += stride) {
        nk_f32_t a_real = a[i].real, a_imag = a[i].imag;
        nk_f32_t b_real = b[i].real, b_imag = b[i].imag;
        nk_f32_t alpha_a_real = alpha_real * a_real - alpha_imag * a_imag;
        nk_f32_t alpha_a_imag = alpha_real * a_imag + alpha_imag * a_real;
        nk_f32_t beta_b_real = beta_real * b_real - beta_imag * b_imag;
        nk_f32_t beta_b_imag = beta_real * b_imag + beta_imag * b_real;
        result[i].real = alpha_a_real + beta_b_real;
        result[i].imag = alpha_a_imag + beta_b_imag;
    }
}

NUMKONG_API nk_status_t nk_each_blend_f32c_cuda(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n,
                                                nk_f32c_t const *alpha, nk_f32c_t const *beta, nk_f32c_t *result,
                                                void *stream) {
    return nk_each_elementwise_launch_cuda_((void const *)&nk_each_blend_f32c_cuda_kernel_, sizeof(nk_f32_t),
                                            sizeof(nk_f32_t), a, b, NUMKONG_NULL, alpha, beta, n, result, stream);
}

static __global__ void nk_each_fma_f32c_cuda_kernel_(nk_each_elementwise_arguments_t arguments) {
    nk_f32c_t const *a = (nk_f32c_t const *)arguments.a, *b = (nk_f32c_t const *)arguments.b;
    nk_f32c_t const *c = (nk_f32c_t const *)arguments.c;
    nk_f32c_t const *alpha = (nk_f32c_t const *)arguments.alpha, *beta = (nk_f32c_t const *)arguments.beta;
    nk_f32c_t *result = (nk_f32c_t *)arguments.result;
    nk_f32_t alpha_real = alpha->real, alpha_imag = alpha->imag;
    nk_f32_t beta_real = beta->real, beta_imag = beta->imag;
    nk_size_t const stride = (nk_size_t)gridDim.x * blockDim.x;
    for (nk_size_t i = (nk_size_t)blockIdx.x * blockDim.x + threadIdx.x; i < arguments.count; i += stride) {
        nk_f32_t a_real = a[i].real, a_imag = a[i].imag;
        nk_f32_t b_real = b[i].real, b_imag = b[i].imag;
        nk_f32_t c_real = c[i].real, c_imag = c[i].imag;
        nk_f32_t product_real = a_real * b_real - a_imag * b_imag;
        nk_f32_t product_imag = a_real * b_imag + a_imag * b_real;
        nk_f32_t alpha_product_real = alpha_real * product_real - alpha_imag * product_imag;
        nk_f32_t alpha_product_imag = alpha_real * product_imag + alpha_imag * product_real;
        nk_f32_t beta_c_real = beta_real * c_real - beta_imag * c_imag;
        nk_f32_t beta_c_imag = beta_real * c_imag + beta_imag * c_real;
        result[i].real = alpha_product_real + beta_c_real;
        result[i].imag = alpha_product_imag + beta_c_imag;
    }
}

NUMKONG_API nk_status_t nk_each_fma_f32c_cuda(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c, nk_size_t n,
                                              nk_f32c_t const *alpha, nk_f32c_t const *beta, nk_f32c_t *result,
                                              void *stream) {
    return nk_each_elementwise_launch_cuda_((void const *)&nk_each_fma_f32c_cuda_kernel_, sizeof(nk_f32_t),
                                            sizeof(nk_f32_t), a, b, c, alpha, beta, n, result, stream);
}

NUMKONG_API nk_status_t nk_each_sum_f64c_cuda(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result,
                                              void *stream) {
    return nk_each_elementwise_launch_cuda_((void const *)&nk_each_sum_f64_cuda_kernel_, sizeof(nk_f64_t), 1, a, b,
                                            NUMKONG_NULL, NUMKONG_NULL, NUMKONG_NULL, 2 * n, result, stream);
}

static __global__ void nk_each_scale_f64c_cuda_kernel_(nk_each_elementwise_arguments_t arguments) {
    nk_f64c_t const *a = (nk_f64c_t const *)arguments.a, *alpha = (nk_f64c_t const *)arguments.alpha;
    nk_f64c_t const *beta = (nk_f64c_t const *)arguments.beta;
    nk_f64c_t *result = (nk_f64c_t *)arguments.result;
    nk_f64_t alpha_real = alpha->real, alpha_imag = alpha->imag;
    nk_f64_t beta_real = beta->real, beta_imag = beta->imag;
    nk_size_t const stride = (nk_size_t)gridDim.x * blockDim.x;
    for (nk_size_t i = (nk_size_t)blockIdx.x * blockDim.x + threadIdx.x; i < arguments.count; i += stride) {
        nk_f64_t a_real = a[i].real, a_imag = a[i].imag;
        result[i].real = alpha_real * a_real - alpha_imag * a_imag + beta_real;
        result[i].imag = alpha_real * a_imag + alpha_imag * a_real + beta_imag;
    }
}

NUMKONG_API nk_status_t nk_each_scale_f64c_cuda(nk_f64c_t const *a, nk_size_t n, nk_f64c_t const *alpha,
                                                nk_f64c_t const *beta, nk_f64c_t *result, void *stream) {
    return nk_each_elementwise_launch_cuda_((void const *)&nk_each_scale_f64c_cuda_kernel_, sizeof(nk_f64_t),
                                            sizeof(nk_f64_t), a, NUMKONG_NULL, NUMKONG_NULL, alpha, beta, n, result,
                                            stream);
}

static __global__ void nk_each_blend_f64c_cuda_kernel_(nk_each_elementwise_arguments_t arguments) {
    nk_f64c_t const *a = (nk_f64c_t const *)arguments.a, *b = (nk_f64c_t const *)arguments.b;
    nk_f64c_t const *alpha = (nk_f64c_t const *)arguments.alpha, *beta = (nk_f64c_t const *)arguments.beta;
    nk_f64c_t *result = (nk_f64c_t *)arguments.result;
    nk_f64_t alpha_real = alpha->real, alpha_imag = alpha->imag;
    nk_f64_t beta_real = beta->real, beta_imag = beta->imag;
    nk_size_t const stride = (nk_size_t)gridDim.x * blockDim.x;
    for (nk_size_t i = (nk_size_t)blockIdx.x * blockDim.x + threadIdx.x; i < arguments.count; i += stride) {
        nk_f64_t a_real = a[i].real, a_imag = a[i].imag;
        nk_f64_t b_real = b[i].real, b_imag = b[i].imag;
        nk_f64_t alpha_a_real = alpha_real * a_real - alpha_imag * a_imag;
        nk_f64_t alpha_a_imag = alpha_real * a_imag + alpha_imag * a_real;
        nk_f64_t beta_b_real = beta_real * b_real - beta_imag * b_imag;
        nk_f64_t beta_b_imag = beta_real * b_imag + beta_imag * b_real;
        result[i].real = alpha_a_real + beta_b_real;
        result[i].imag = alpha_a_imag + beta_b_imag;
    }
}

NUMKONG_API nk_status_t nk_each_blend_f64c_cuda(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n,
                                                nk_f64c_t const *alpha, nk_f64c_t const *beta, nk_f64c_t *result,
                                                void *stream) {
    return nk_each_elementwise_launch_cuda_((void const *)&nk_each_blend_f64c_cuda_kernel_, sizeof(nk_f64_t),
                                            sizeof(nk_f64_t), a, b, NUMKONG_NULL, alpha, beta, n, result, stream);
}

static __global__ void nk_each_fma_f64c_cuda_kernel_(nk_each_elementwise_arguments_t arguments) {
    nk_f64c_t const *a = (nk_f64c_t const *)arguments.a, *b = (nk_f64c_t const *)arguments.b;
    nk_f64c_t const *c = (nk_f64c_t const *)arguments.c;
    nk_f64c_t const *alpha = (nk_f64c_t const *)arguments.alpha, *beta = (nk_f64c_t const *)arguments.beta;
    nk_f64c_t *result = (nk_f64c_t *)arguments.result;
    nk_f64_t alpha_real = alpha->real, alpha_imag = alpha->imag;
    nk_f64_t beta_real = beta->real, beta_imag = beta->imag;
    nk_size_t const stride = (nk_size_t)gridDim.x * blockDim.x;
    for (nk_size_t i = (nk_size_t)blockIdx.x * blockDim.x + threadIdx.x; i < arguments.count; i += stride) {
        nk_f64_t a_real = a[i].real, a_imag = a[i].imag;
        nk_f64_t b_real = b[i].real, b_imag = b[i].imag;
        nk_f64_t c_real = c[i].real, c_imag = c[i].imag;
        nk_f64_t product_real = a_real * b_real - a_imag * b_imag;
        nk_f64_t product_imag = a_real * b_imag + a_imag * b_real;
        nk_f64_t alpha_product_real = alpha_real * product_real - alpha_imag * product_imag;
        nk_f64_t alpha_product_imag = alpha_real * product_imag + alpha_imag * product_real;
        nk_f64_t beta_c_real = beta_real * c_real - beta_imag * c_imag;
        nk_f64_t beta_c_imag = beta_real * c_imag + beta_imag * c_real;
        result[i].real = alpha_product_real + beta_c_real;
        result[i].imag = alpha_product_imag + beta_c_imag;
    }
}

NUMKONG_API nk_status_t nk_each_fma_f64c_cuda(nk_f64c_t const *a, nk_f64c_t const *b, nk_f64c_t const *c, nk_size_t n,
                                              nk_f64c_t const *alpha, nk_f64c_t const *beta, nk_f64c_t *result,
                                              void *stream) {
    return nk_each_elementwise_launch_cuda_((void const *)&nk_each_fma_f64c_cuda_kernel_, sizeof(nk_f64_t),
                                            sizeof(nk_f64_t), a, b, c, alpha, beta, n, result, stream);
}

nk_define_each_rmsnorm_cuda_(f32, cuda, nk_assign_from_to_, nk_assign_from_to_)
nk_define_each_rmsnorm_cuda_(bf16, cuda, nk_bf16_to_f32_simt_, nk_f32_to_bf16_simt_)
nk_define_each_rmsnorm_cuda_(e4m3, cuda, nk_e4m3_to_f32_simt_, nk_f32_to_e4m3_simt_)
nk_define_each_swiglu_cuda_(f32, cuda, nk_assign_from_to_, nk_assign_from_to_)
nk_define_each_swiglu_cuda_(bf16, cuda, nk_bf16_to_f32_simt_, nk_f32_to_bf16_simt_)
nk_define_each_swiglu_cuda_(e4m3, cuda, nk_e4m3_to_f32_simt_, nk_f32_to_e4m3_simt_)
#endif // NUMKONG_TARGET_CUDA

#undef nk_define_each_sum_cuda_
#undef nk_define_each_scale_cuda_
#undef nk_define_each_blend_cuda_
#undef nk_define_each_fma_cuda_
#undef nk_define_each_rmsnorm_cuda_
#undef nk_define_each_swiglu_cuda_

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_ && defined(__CUDACC__) && !defined(__HIP__)
#endif // NUMKONG_EACH_CUDA_CUH
