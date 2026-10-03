/**
 *  @file include/numkong/each/simt.cuh
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Element-wise scales, sums, blends, FMAs, RMSNorm and SwiGLU on the SIMT cores of every
 *      CUDA and ROCm device.
 *
 *  @sa include/numkong/each.h
 *  @sa include/numkong/cast/simt.cuh
 *
 *  Every thread owns elements a stride of the grid apart, computes in the serial kernel's
 *  accumulator type and rounds like the serial casts, integers saturating. Every thread reads α and
 *  β from device memory, so a launch never waits on the host. SwiGLU takes separate gate and up
 *  rows, so a fused @b [rows,2×ffn] buffer passes gate = base and up = base + ffn with one row
 *  stride, and a NULL @c up is plain SiLU. RMSNorm runs one block per normalized vector instead,
 *  summing its squares as compensated F32 pairs, TwoProduct through FMA and TwoSum, rather than in
 *  F64, which sm_103 runs at 2 results per clock per SM, and rounding their mean once.
 *
 *  Only the kernels live here; each vendor launches and exports them from its own host side, in
 *  `cuda.cuh` and `rocm.cuh` beside this file.
 *
 *  @sa include/numkong/each/cuda.cuh
 *  @sa include/numkong/each/rocm.cuh
 */
#ifndef NUMKONG_EACH_SIMT_CUH
#define NUMKONG_EACH_SIMT_CUH

#if NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_

#include "numkong/cast/simt.cuh"   // `nk_bf16_to_f32_simt_`, `nk_f32_to_i8_simt_`
#include "numkong/reduce/simt.cuh" // `nk_f32_two_sum_simt_`, `nk_f32_mul_rn_simt_`

#if defined(__cplusplus)
extern "C" {
#endif

enum { nk_each_threads_simt_k = 256 };

/** Everything one SwiGLU launch shares, passed by value as the kernels' only argument. */
typedef struct {
    unsigned char const *gate;
    unsigned char const *up;
    unsigned char *y;
    nk_size_t columns;
    nk_size_t count;
    nk_size_t gate_stride;
    nk_size_t up_stride;
    nk_size_t y_stride;
    nk_f32_t gate_scale;
    nk_f32_t output_scale;
} nk_each_swiglu_arguments_t;

/** Everything one element-wise scale, sum, blend or FMA shares, passed by value as the kernels'
 *  only argument, each verb reading the operands it takes. */
typedef struct {
    unsigned char const *a;
    unsigned char const *b;
    unsigned char const *c;
    void const *alpha;
    void const *beta;
    unsigned char *result;
    nk_size_t count;
} nk_each_elementwise_arguments_t;

/** Generates the SwiGLU kernel of @p input_type for @p isa_suffix, after @c nk_define_each_swiglu_
 *  of the serial backend. */
#define nk_define_each_swiglu_kernel_simt_(input_type, isa_suffix, load_and_convert, convert_and_store)         \
    static __global__ void nk_each_swiglu_##input_type##_##isa_suffix##_kernel_(                                \
        nk_each_swiglu_arguments_t arguments) {                                                                 \
        nk_size_t const stride = (nk_size_t)gridDim.x * blockDim.x;                                             \
        for (nk_size_t cell = (nk_size_t)blockIdx.x * blockDim.x + threadIdx.x; cell < arguments.count;         \
             cell += stride) {                                                                                  \
            nk_size_t const row = cell / arguments.columns, col = cell % arguments.columns;                     \
            nk_f32_t gate_value, up_value;                                                                      \
            load_and_convert((nk_##input_type##_t const *)(arguments.gate + row * arguments.gate_stride) + col, \
                             &gate_value);                                                                      \
            gate_value *= arguments.gate_scale;                                                                 \
            nk_f32_t result = gate_value / (1.0f + expf(-gate_value));                                          \
            if (arguments.up) {                                                                                 \
                load_and_convert((nk_##input_type##_t const *)(arguments.up + row * arguments.up_stride) + col, \
                                 &up_value);                                                                    \
                result *= up_value;                                                                             \
            }                                                                                                   \
            result *= arguments.output_scale;                                                                   \
            convert_and_store(&result, (nk_##input_type##_t *)(arguments.y + row * arguments.y_stride) + col);  \
        }                                                                                                       \
    }

/** Generates the element-wise sum kernel of @p input_type for @p isa_suffix, after
 *  @c nk_define_each_sum_ of the serial backend. */
#define nk_define_each_sum_kernel_simt_(input_type, accumulator_type, isa_suffix, load_and_convert, convert_and_store) \
    static __global__ void nk_each_sum_##input_type##_##isa_suffix##_kernel_(                                          \
        nk_each_elementwise_arguments_t arguments) {                                                                   \
        nk_size_t const stride = (nk_size_t)gridDim.x * blockDim.x;                                                    \
        for (nk_size_t cell = (nk_size_t)blockIdx.x * blockDim.x + threadIdx.x; cell < arguments.count;                \
             cell += stride) {                                                                                         \
            nk_##accumulator_type##_t a_value, b_value;                                                                \
            load_and_convert((nk_##input_type##_t const *)arguments.a + cell, &a_value);                               \
            load_and_convert((nk_##input_type##_t const *)arguments.b + cell, &b_value);                               \
            nk_##accumulator_type##_t const sum = a_value + b_value;                                                   \
            convert_and_store(&sum, (nk_##input_type##_t *)arguments.result + cell);                                   \
        }                                                                                                              \
    }

/** Generates the element-wise scale kernel of @p input_type, α · a + β, for @p isa_suffix, after
 *  @c nk_define_each_scale_ of the serial backend. */
#define nk_define_each_scale_kernel_simt_(input_type, accumulator_type, isa_suffix, load_and_convert,   \
                                          convert_and_store)                                            \
    static __global__ void nk_each_scale_##input_type##_##isa_suffix##_kernel_(                         \
        nk_each_elementwise_arguments_t arguments) {                                                    \
        nk_##accumulator_type##_t const alpha = *(nk_##accumulator_type##_t const *)arguments.alpha;    \
        nk_##accumulator_type##_t const beta = *(nk_##accumulator_type##_t const *)arguments.beta;      \
        nk_size_t const stride = (nk_size_t)gridDim.x * blockDim.x;                                     \
        for (nk_size_t cell = (nk_size_t)blockIdx.x * blockDim.x + threadIdx.x; cell < arguments.count; \
             cell += stride) {                                                                          \
            nk_##accumulator_type##_t a_value;                                                          \
            load_and_convert((nk_##input_type##_t const *)arguments.a + cell, &a_value);                \
            nk_##accumulator_type##_t const result = alpha * a_value + beta;                            \
            convert_and_store(&result, (nk_##input_type##_t *)arguments.result + cell);                 \
        }                                                                                               \
    }

/** Generates the element-wise blend kernel of @p input_type, α · a + β · b, for @p isa_suffix,
 *  after @c nk_define_each_blend_ of the serial backend. */
#define nk_define_each_blend_kernel_simt_(input_type, accumulator_type, isa_suffix, load_and_convert,   \
                                          convert_and_store)                                            \
    static __global__ void nk_each_blend_##input_type##_##isa_suffix##_kernel_(                         \
        nk_each_elementwise_arguments_t arguments) {                                                    \
        nk_##accumulator_type##_t const alpha = *(nk_##accumulator_type##_t const *)arguments.alpha;    \
        nk_##accumulator_type##_t const beta = *(nk_##accumulator_type##_t const *)arguments.beta;      \
        nk_size_t const stride = (nk_size_t)gridDim.x * blockDim.x;                                     \
        for (nk_size_t cell = (nk_size_t)blockIdx.x * blockDim.x + threadIdx.x; cell < arguments.count; \
             cell += stride) {                                                                          \
            nk_##accumulator_type##_t a_value, b_value;                                                 \
            load_and_convert((nk_##input_type##_t const *)arguments.a + cell, &a_value);                \
            load_and_convert((nk_##input_type##_t const *)arguments.b + cell, &b_value);                \
            nk_##accumulator_type##_t const result = a_value * alpha + b_value * beta;                  \
            convert_and_store(&result, (nk_##input_type##_t *)arguments.result + cell);                 \
        }                                                                                               \
    }

/** Generates the element-wise FMA kernel of @p input_type, α · a · b + β · c, for @p isa_suffix,
 *  after @c nk_define_each_fma_ of the serial backend. */
#define nk_define_each_fma_kernel_simt_(input_type, accumulator_type, isa_suffix, load_and_convert, convert_and_store) \
    static __global__ void nk_each_fma_##input_type##_##isa_suffix##_kernel_(                                          \
        nk_each_elementwise_arguments_t arguments) {                                                                   \
        nk_##accumulator_type##_t const alpha = *(nk_##accumulator_type##_t const *)arguments.alpha;                   \
        nk_##accumulator_type##_t const beta = *(nk_##accumulator_type##_t const *)arguments.beta;                     \
        nk_size_t const stride = (nk_size_t)gridDim.x * blockDim.x;                                                    \
        for (nk_size_t cell = (nk_size_t)blockIdx.x * blockDim.x + threadIdx.x; cell < arguments.count;                \
             cell += stride) {                                                                                         \
            nk_##accumulator_type##_t a_value, b_value, c_value;                                                       \
            load_and_convert((nk_##input_type##_t const *)arguments.a + cell, &a_value);                               \
            load_and_convert((nk_##input_type##_t const *)arguments.b + cell, &b_value);                               \
            load_and_convert((nk_##input_type##_t const *)arguments.c + cell, &c_value);                               \
            nk_##accumulator_type##_t const result = a_value * b_value * alpha + c_value * beta;                       \
            convert_and_store(&result, (nk_##input_type##_t *)arguments.result + cell);                                \
        }                                                                                                              \
    }

/** Everything one RMSNorm launch shares, passed by value as the kernels' only argument. */
typedef struct {
    unsigned char const *x;
    nk_f32_t const *gamma;
    unsigned char *y;
    nk_size_t groups;
    nk_size_t columns;
    nk_size_t vectors;
    nk_size_t x_stride;
    nk_size_t y_stride;
    nk_f32_t epsilon;
} nk_each_rmsnorm_arguments_t;

/** Merges every thread's compensated @p sum and @p compensation across the block through
 *  @p partials, two values per thread, and returns the mean square of @p count values
 *  to every thread, rounded once by thread 0 in F64. */
NUMKONG_DEVICE nk_f32_t nk_each_block_mean_square_simt_(nk_f32_t sum, nk_f32_t compensation, nk_size_t count,
                                                        nk_f32_t *partials) {
    nk_f32_t *compensations = partials + blockDim.x;
    partials[threadIdx.x] = sum, compensations[threadIdx.x] = compensation;
    __syncthreads();
    for (unsigned half = blockDim.x / 2; half; half >>= 1) {
        if (threadIdx.x < half) {
            compensations[threadIdx.x] += compensations[threadIdx.x + half];
            nk_f32_two_sum_simt_(partials[threadIdx.x + half], partials + threadIdx.x, compensations + threadIdx.x);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        partials[0] = (nk_f32_t)(((nk_f64_t)partials[0] + (nk_f64_t)compensations[0]) / (nk_f64_t)count);
    __syncthreads();
    nk_f32_t const mean_square = partials[0];
    __syncthreads();
    return mean_square;
}

/** Generates the RMSNorm kernel of @p input_type for @p isa_suffix, after @c nk_define_each_rmsnorm_
 *  of the serial backend, one block per vector. */
#define nk_define_each_rmsnorm_kernel_simt_(input_type, isa_suffix, load_and_convert, convert_and_store)               \
    static __global__ void nk_each_rmsnorm_##input_type##_##isa_suffix##_kernel_(                                      \
        nk_each_rmsnorm_arguments_t arguments) {                                                                       \
        __shared__ nk_f32_t partials[2 * nk_each_threads_simt_k];                                                      \
        for (nk_size_t vector = blockIdx.x; vector < arguments.vectors; vector += gridDim.x) {                         \
            nk_size_t const row = vector / arguments.groups, first = vector % arguments.groups * arguments.columns;    \
            nk_##input_type##_t const *x = (nk_##input_type##_t const *)(arguments.x + row * arguments.x_stride) +     \
                                           first;                                                                      \
            nk_##input_type##_t *y = (nk_##input_type##_t *)(arguments.y + row * arguments.y_stride) + first;          \
            nk_f32_t sum_squares = 0, compensation = 0, value;                                                         \
            for (nk_size_t col = threadIdx.x; col < arguments.columns; col += blockDim.x) {                            \
                load_and_convert(x + col, &value);                                                                     \
                nk_f32_t const square = nk_f32_mul_rn_simt_(value, value);                                             \
                compensation += fmaf(value, value, -square);                                                           \
                nk_f32_two_sum_simt_(square, &sum_squares, &compensation);                                             \
            }                                                                                                          \
            nk_f32_t const mean_square = nk_each_block_mean_square_simt_(sum_squares, compensation, arguments.columns, \
                                                                         partials);                                    \
            nk_f32_t const inverse_rms = 1.0f / sqrtf(mean_square + arguments.epsilon);                                \
            for (nk_size_t col = threadIdx.x; col < arguments.columns; col += blockDim.x) {                            \
                load_and_convert(x + col, &value);                                                                     \
                nk_f32_t const result = value * inverse_rms * (arguments.gamma ? arguments.gamma[col] : 1.0f);         \
                convert_and_store(&result, y + col);                                                                   \
            }                                                                                                          \
        }                                                                                                              \
    }

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_
#endif // NUMKONG_EACH_SIMT_CUH
