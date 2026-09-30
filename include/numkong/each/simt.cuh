/**
 *  @file include/numkong/each/simt.cuh
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Element-wise sums, RMSNorm and SwiGLU on the SIMT cores of every CUDA and ROCm device.
 *
 *  @sa include/numkong/each.h
 *  @sa include/numkong/cast/simt.cuh
 *
 *  Every thread owns elements a stride of the grid apart, computes in F32 and rounds like the
 *  serial casts. SwiGLU takes separate gate and up rows, so a fused @b [rows,2×ffn] buffer passes
 *  gate = base and up = base + ffn with one row stride, and a NULL @c up is plain SiLU. RMSNorm
 *  runs one block per normalized vector instead, summing its squares in F64 for every input type.
 */
#ifndef NUMKONG_EACH_SIMT_CUH
#define NUMKONG_EACH_SIMT_CUH

#if NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_

#include "numkong/cast/simt.cuh" // `nk_bf16_to_f32_simt_`, `nk_f32_to_bf16_simt_`
#include "numkong/dots/simt.cuh" // `nk_device_launch_`

#if defined(__cplusplus)
extern "C" {
#endif

enum { nk_each_threads_simt_k = 256 };

/** Everything one SwiGLU launch shares, passed by value as the kernels' only argument. */
typedef struct {
    unsigned char const *gate;
    unsigned char const *up;
    unsigned char *y;
    nk_size_t cols;
    nk_size_t count;
    nk_size_t gate_stride_bytes;
    nk_size_t up_stride_bytes;
    nk_size_t y_stride_bytes;
    nk_f32_t input_scale;
} nk_each_swiglu_arguments_t;

/** Everything one element-wise sum shares, passed by value as the kernels' only argument. */
typedef struct {
    unsigned char const *a;
    unsigned char const *b;
    unsigned char *result;
    nk_size_t count;
} nk_each_sum_arguments_t;

/** Launches enough blocks of @p kernel for @p count elements, at most 2²⁰ of them, passing the one
 *  argument struct at @p arguments by value. */
NUMKONG_INLINE nk_status_t nk_each_launch_(void const *kernel, nk_size_t count, void *arguments, void *stream) {
    if (count == 0) return nk_success_k;
    nk_size_t const blocks = nk_size_divide_round_up_(count, nk_each_threads_simt_k), blocks_limit = (nk_size_t)1 << 20;
    void *launch_arguments[1];
    launch_arguments[0] = arguments;
    return nk_device_launch_(kernel, blocks < blocks_limit ? blocks : blocks_limit, nk_each_threads_simt_k,
                             launch_arguments, 0, stream);
}

NUMKONG_INLINE nk_status_t nk_each_swiglu_launch_(void const *kernel, nk_size_t value_bytes, void const *gate,
                                                  void const *up, void *y, nk_size_t rows, nk_size_t cols,
                                                  nk_size_t gate_stride_bytes, nk_size_t up_stride_bytes,
                                                  nk_size_t y_stride_bytes, nk_f32_t input_scale, void *stream) {
    if ((((nk_size_t)gate) | gate_stride_bytes | ((nk_size_t)up) | (up ? up_stride_bytes : 0) | ((nk_size_t)y) |
         y_stride_bytes) &
        (value_bytes - 1))
        return nk_misaligned_k;
    nk_each_swiglu_arguments_t arguments;
    arguments.gate = (unsigned char const *)gate, arguments.up = (unsigned char const *)up;
    arguments.y = (unsigned char *)y, arguments.cols = cols, arguments.count = rows * cols;
    arguments.gate_stride_bytes = gate_stride_bytes, arguments.up_stride_bytes = up_stride_bytes;
    arguments.y_stride_bytes = y_stride_bytes, arguments.input_scale = input_scale;
    return nk_each_launch_(kernel, arguments.count, &arguments, stream);
}

NUMKONG_INLINE nk_status_t nk_each_sum_launch_(void const *kernel, nk_size_t value_bytes, void const *a, void const *b,
                                               nk_size_t n, void *result, void *stream) {
    if ((((nk_size_t)a) | ((nk_size_t)b) | ((nk_size_t)result)) & (value_bytes - 1)) return nk_misaligned_k;
    nk_each_sum_arguments_t arguments;
    arguments.a = (unsigned char const *)a, arguments.b = (unsigned char const *)b;
    arguments.result = (unsigned char *)result, arguments.count = n;
    return nk_each_launch_(kernel, n, &arguments, stream);
}

/** Generates the SwiGLU kernel of @p input_type and its host entry point for @p isa_suffix, after
 *  @c nk_define_each_swiglu_ of the serial backend. */
#define nk_define_device_each_swiglu_(input_type, isa_suffix, load_and_convert, convert_and_store)                    \
    static __global__ void nk_each_swiglu_##input_type##_##isa_suffix##_kernel_(                                      \
        nk_each_swiglu_arguments_t arguments) {                                                                       \
        nk_size_t const stride = (nk_size_t)gridDim.x * blockDim.x;                                                   \
        for (nk_size_t cell = (nk_size_t)blockIdx.x * blockDim.x + threadIdx.x; cell < arguments.count;               \
             cell += stride) {                                                                                        \
            nk_size_t const row = cell / arguments.cols, col = cell % arguments.cols;                                 \
            nk_f32_t gate_value, up_value;                                                                            \
            load_and_convert((nk_##input_type##_t const *)(arguments.gate + row * arguments.gate_stride_bytes) + col, \
                             &gate_value);                                                                            \
            gate_value *= arguments.input_scale;                                                                      \
            nk_f32_t result = gate_value / (1.0f + expf(-gate_value));                                                \
            if (arguments.up) {                                                                                       \
                load_and_convert((nk_##input_type##_t const *)(arguments.up + row * arguments.up_stride_bytes) + col, \
                                 &up_value);                                                                          \
                result *= up_value * arguments.input_scale;                                                           \
            }                                                                                                         \
            convert_and_store(&result, (nk_##input_type##_t *)(arguments.y + row * arguments.y_stride_bytes) + col);  \
        }                                                                                                             \
    }                                                                                                                 \
    NUMKONG_API nk_status_t nk_each_swiglu_##input_type##_##isa_suffix(                                               \
        nk_##input_type##_t const *gate, nk_##input_type##_t const *up, nk_##input_type##_t *y, nk_size_t rows,       \
        nk_size_t cols, nk_size_t gate_stride_bytes, nk_size_t up_stride_bytes, nk_size_t y_stride_bytes,             \
        nk_f32_t input_scale, void *stream) {                                                                         \
        return nk_each_swiglu_launch_((void const *)&nk_each_swiglu_##input_type##_##isa_suffix##_kernel_,            \
                                      sizeof(nk_##input_type##_t), gate, up, y, rows, cols, gate_stride_bytes,        \
                                      up_stride_bytes, y_stride_bytes, input_scale, stream);                          \
    }

/** Generates the element-wise sum kernel of @p input_type and its host entry point for
 *  @p isa_suffix, after @c nk_define_each_sum_ of the serial backend. */
#define nk_define_device_each_sum_(input_type, isa_suffix, load_and_convert, convert_and_store)                   \
    static __global__ void nk_each_sum_##input_type##_##isa_suffix##_kernel_(nk_each_sum_arguments_t arguments) { \
        nk_size_t const stride = (nk_size_t)gridDim.x * blockDim.x;                                               \
        for (nk_size_t cell = (nk_size_t)blockIdx.x * blockDim.x + threadIdx.x; cell < arguments.count;           \
             cell += stride) {                                                                                    \
            nk_f32_t a_value, b_value;                                                                            \
            load_and_convert((nk_##input_type##_t const *)arguments.a + cell, &a_value);                          \
            load_and_convert((nk_##input_type##_t const *)arguments.b + cell, &b_value);                          \
            nk_f32_t const sum = a_value + b_value;                                                               \
            convert_and_store(&sum, (nk_##input_type##_t *)arguments.result + cell);                              \
        }                                                                                                         \
    }                                                                                                             \
    NUMKONG_API nk_status_t nk_each_sum_##input_type##_##isa_suffix(nk_##input_type##_t const *a,                 \
                                                                    nk_##input_type##_t const *b, nk_size_t n,    \
                                                                    nk_##input_type##_t *result, void *stream) {  \
        return nk_each_sum_launch_((void const *)&nk_each_sum_##input_type##_##isa_suffix##_kernel_,              \
                                   sizeof(nk_##input_type##_t), a, b, n, result, stream);                         \
    }

/** Everything one RMSNorm launch shares, passed by value as the kernels' only argument. */
typedef struct {
    unsigned char const *x;
    nk_f32_t const *gamma;
    unsigned char *y;
    nk_size_t groups;
    nk_size_t cols;
    nk_size_t vectors;
    nk_size_t x_stride_bytes;
    nk_size_t y_stride_bytes;
    nk_f32_t eps;
    nk_f32_t input_scale;
} nk_each_rmsnorm_arguments_t;

/** Sums @p value across the block, returning the total to every thread, through @p partials. */
NUMKONG_DEVICE nk_f64_t nk_each_block_sum_f64_simt_(nk_f64_t value, nk_f64_t *partials) {
    partials[threadIdx.x] = value;
    __syncthreads();
    for (unsigned half = blockDim.x / 2; half; half >>= 1) {
        if (threadIdx.x < half) partials[threadIdx.x] += partials[threadIdx.x + half];
        __syncthreads();
    }
    nk_f64_t const total = partials[0];
    __syncthreads();
    return total;
}

/** Validates the contract and launches one block per normalized vector, at most 2³⁰ of them. */
NUMKONG_INLINE nk_status_t nk_each_rmsnorm_launch_(void const *kernel, nk_size_t value_bytes, void const *x,
                                                   nk_f32_t const *gamma, void *y, nk_size_t rows, nk_size_t groups,
                                                   nk_size_t cols, nk_size_t x_stride_bytes, nk_size_t y_stride_bytes,
                                                   nk_f32_t eps, nk_f32_t input_scale, void *stream) {
    if ((((nk_size_t)x) | x_stride_bytes | ((nk_size_t)y) | y_stride_bytes) & (value_bytes - 1) ||
        ((nk_size_t)gamma & 3))
        return nk_misaligned_k;
    nk_each_rmsnorm_arguments_t arguments;
    arguments.x = (unsigned char const *)x, arguments.gamma = gamma, arguments.y = (unsigned char *)y;
    arguments.groups = groups, arguments.cols = cols, arguments.vectors = rows * groups;
    arguments.x_stride_bytes = x_stride_bytes, arguments.y_stride_bytes = y_stride_bytes;
    arguments.eps = eps, arguments.input_scale = input_scale;
    if (arguments.vectors == 0 || cols == 0) return nk_success_k;
    nk_size_t const blocks_limit = (nk_size_t)1 << 30;
    void *launch_arguments[1];
    launch_arguments[0] = &arguments;
    return nk_device_launch_(kernel, arguments.vectors < blocks_limit ? arguments.vectors : blocks_limit,
                             nk_each_threads_simt_k, launch_arguments, 0, stream);
}

/** Generates the RMSNorm kernel of @p input_type and its host entry point for @p isa_suffix, after
 *  @c nk_define_each_rmsnorm_ of the serial backend, one block per vector. */
#define nk_define_device_each_rmsnorm_(input_type, isa_suffix, load_and_convert, convert_and_store)                    \
    static __global__ void nk_each_rmsnorm_##input_type##_##isa_suffix##_kernel_(                                      \
        nk_each_rmsnorm_arguments_t arguments) {                                                                       \
        __shared__ nk_f64_t partials[nk_each_threads_simt_k];                                                          \
        nk_f64_t const scale_squared = (nk_f64_t)arguments.input_scale * (nk_f64_t)arguments.input_scale;              \
        for (nk_size_t vector = blockIdx.x; vector < arguments.vectors; vector += gridDim.x) {                         \
            nk_size_t const row = vector / arguments.groups, first = vector % arguments.groups * arguments.cols;       \
            nk_##input_type##_t const *x =                                                                             \
                (nk_##input_type##_t const *)(arguments.x + row * arguments.x_stride_bytes) + first;                   \
            nk_##input_type##_t *y = (nk_##input_type##_t *)(arguments.y + row * arguments.y_stride_bytes) + first;    \
            nk_f64_t sum_squares = 0;                                                                                  \
            nk_f32_t value;                                                                                            \
            for (nk_size_t col = threadIdx.x; col < arguments.cols; col += blockDim.x) {                               \
                load_and_convert(x + col, &value);                                                                     \
                sum_squares += (nk_f64_t)value * (nk_f64_t)value;                                                      \
            }                                                                                                          \
            sum_squares = nk_each_block_sum_f64_simt_(sum_squares, partials);                                          \
            nk_f32_t const mean_square = (nk_f32_t)(scale_squared * sum_squares / (nk_f64_t)arguments.cols);           \
            nk_f32_t const inverse_rms = 1.0f / sqrtf(mean_square + arguments.eps);                                    \
            for (nk_size_t col = threadIdx.x; col < arguments.cols; col += blockDim.x) {                               \
                load_and_convert(x + col, &value);                                                                     \
                nk_f32_t const result = value * arguments.input_scale * inverse_rms *                                  \
                                        (arguments.gamma ? arguments.gamma[col] : 1.0f);                               \
                convert_and_store(&result, y + col);                                                                   \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    NUMKONG_API nk_status_t nk_each_rmsnorm_##input_type##_##isa_suffix(                                               \
        nk_##input_type##_t const *x, nk_f32_t const *gamma, nk_##input_type##_t *y, nk_size_t rows, nk_size_t groups, \
        nk_size_t cols, nk_size_t x_stride_bytes, nk_size_t y_stride_bytes, nk_f32_t eps, nk_f32_t input_scale,        \
        void *stream) {                                                                                                \
        return nk_each_rmsnorm_launch_((void const *)&nk_each_rmsnorm_##input_type##_##isa_suffix##_kernel_,           \
                                       sizeof(nk_##input_type##_t), x, gamma, y, rows, groups, cols, x_stride_bytes,   \
                                       y_stride_bytes, eps, input_scale, stream);                                      \
    }

#if NUMKONG_TARGET_CUDA
nk_define_device_each_sum_(f32, cuda, nk_assign_from_to_, nk_assign_from_to_)
nk_define_device_each_sum_(f16, cuda, nk_f16_to_f32_simt_, nk_f32_to_f16_simt_)
nk_define_device_each_sum_(bf16, cuda, nk_bf16_to_f32_simt_, nk_f32_to_bf16_simt_)
nk_define_device_each_rmsnorm_(f32, cuda, nk_assign_from_to_, nk_assign_from_to_)
nk_define_device_each_rmsnorm_(bf16, cuda, nk_bf16_to_f32_simt_, nk_f32_to_bf16_simt_)
nk_define_device_each_rmsnorm_(e4m3, cuda, nk_e4m3_to_f32_simt_, nk_f32_to_e4m3_simt_)
nk_define_device_each_swiglu_(f32, cuda, nk_assign_from_to_, nk_assign_from_to_)
nk_define_device_each_swiglu_(bf16, cuda, nk_bf16_to_f32_simt_, nk_f32_to_bf16_simt_)
nk_define_device_each_swiglu_(e4m3, cuda, nk_e4m3_to_f32_simt_, nk_f32_to_e4m3_simt_)
#elif NUMKONG_TARGET_ROCM
nk_define_device_each_sum_(f32, rocm, nk_assign_from_to_, nk_assign_from_to_)
nk_define_device_each_sum_(f16, rocm, nk_f16_to_f32_simt_, nk_f32_to_f16_simt_)
nk_define_device_each_sum_(bf16, rocm, nk_bf16_to_f32_simt_, nk_f32_to_bf16_simt_)
nk_define_device_each_rmsnorm_(f32, rocm, nk_assign_from_to_, nk_assign_from_to_)
nk_define_device_each_rmsnorm_(bf16, rocm, nk_bf16_to_f32_simt_, nk_f32_to_bf16_simt_)
nk_define_device_each_rmsnorm_(e4m3, rocm, nk_e4m3_to_f32_simt_, nk_f32_to_e4m3_simt_)
nk_define_device_each_swiglu_(f32, rocm, nk_assign_from_to_, nk_assign_from_to_)
nk_define_device_each_swiglu_(bf16, rocm, nk_bf16_to_f32_simt_, nk_f32_to_bf16_simt_)
nk_define_device_each_swiglu_(e4m3, rocm, nk_e4m3_to_f32_simt_, nk_f32_to_e4m3_simt_)
#endif

#undef nk_define_device_each_sum_
#undef nk_define_device_each_rmsnorm_
#undef nk_define_device_each_swiglu_

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_
#endif // NUMKONG_EACH_SIMT_CUH
