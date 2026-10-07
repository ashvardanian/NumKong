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
 *  Every kernel moves whole 16-byte chunks when its operands start on 16-byte boundaries, and the
 *  values past the last whole chunk, or every value otherwise, one at a time, each through the same
 *  helper. Each thread computes in the serial kernel's accumulator type and rounds like the serial
 *  casts, integers saturating, and reads α and β from device memory, so a launch never waits on the
 *  host. Scale, sum, blend and FMA walk their chunks a grid apart. SwiGLU gives each row to a team
 *  of 32 lanes and takes separate gate and up rows, so a fused @b [rows,2×ffn] buffer passes gate =
 *  base and up = base + ffn with one row stride, and a NULL @c up is plain SiLU.
 *
 *  RMSNorm, and its downcasting form reading F32, give each normalized vector one block, sized so
 *  that every thread keeps at most 8 16-byte pieces of it in registers between the two passes,
 *  reloading any beyond. A chunk there is 16 bytes of the narrower side, so every store moves 16
 *  bytes. Squares sum as compensated F32 pairs, TwoProduct through FMA and TwoSum, rather than in
 *  F64, which sm_103 runs at 2 results per clock per SM. The pairs merge through xor shuffles
 *  across a team's lanes and through shared memory across teams, and their mean rounds once in F64.
 *
 *  The element-wise kernels live here, and each vendor launches and exports them from its own host
 *  side, in `cuda.cuh` and `rocm.cuh` beside this file, with the RMSNorm kernels and their sums of
 *  squares, which round and shuffle through that vendor's primitives.
 *
 *  @sa include/numkong/each/cuda.cuh
 *  @sa include/numkong/each/rocm.cuh
 */
#ifndef NUMKONG_EACH_SIMT_CUH
#define NUMKONG_EACH_SIMT_CUH

#if NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_

#include "numkong/cast/simt.cuh"   // `nk_bf16_to_f32_simt_`
#include "numkong/reduce/simt.cuh" // `nk_f32_two_sum_simt_`

#if defined(__cplusplus)
extern "C" {
#endif

/** Launch shapes the element-wise kernels share. */
enum {

    /** Threads of every block, and the most an RMSNorm block takes. */
    nk_each_threads_simt_k = 256,

    /** Lanes sharing one row or one vector: a warp on NVIDIA and half a wavefront on CDNA. */
    nk_each_team_lanes_simt_k = 32,

    /** 16-byte pieces of its vector each RMSNorm thread keeps in registers between both passes. */
    nk_each_kept_pieces_simt_k = 8,
};

/** Everything one SwiGLU launch shares, passed by value as the kernels' only argument. */
typedef struct {
    unsigned char const *gate;
    unsigned char const *up;
    unsigned char *y;
    nk_size_t rows;
    nk_size_t columns;
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

/** Generates the kernel of the element-wise @p verb over @p operands inputs of @p input_type for
 *  @p isa_suffix, calling for every value the helper that the verb's own generator defines. Its
 *  chunk loop stays rolled, as unrolled copies cannot load before the stores they follow and only
 *  add registers, which cost the F64 kernels the occupancy that B300's FP64 pipe needs. */
#define nk_define_each_elementwise_kernel_simt_(verb, operands, input_type, accumulator_type, isa_suffix)         \
    static __global__ void nk_each_##verb##_##input_type##_##isa_suffix##_kernel_(                                \
        nk_each_elementwise_arguments_t arguments) {                                                              \
        nk_##input_type##_t const *a = (nk_##input_type##_t const *)arguments.a;                                  \
        nk_##input_type##_t const *b = operands > 1 ? (nk_##input_type##_t const *)arguments.b : a;               \
        nk_##input_type##_t const *c = operands > 2 ? (nk_##input_type##_t const *)arguments.c : a;               \
        nk_##input_type##_t *result = (nk_##input_type##_t *)arguments.result;                                    \
        nk_##accumulator_type##_t const *alpha_address = (nk_##accumulator_type##_t const *)arguments.alpha;      \
        nk_##accumulator_type##_t const *beta_address = (nk_##accumulator_type##_t const *)arguments.beta;        \
        nk_##accumulator_type##_t const alpha = alpha_address ? *alpha_address : 0;                               \
        nk_##accumulator_type##_t const beta = beta_address ? *beta_address : 0;                                  \
        unsigned const chunk_values = 16 / sizeof(nk_##input_type##_t);                                           \
        nk_size_t const threads = (nk_size_t)gridDim.x * blockDim.x;                                              \
        nk_size_t const first = (nk_size_t)blockIdx.x * blockDim.x + threadIdx.x;                                 \
        int const aligned = !(((nk_size_t)a | (nk_size_t)b | (nk_size_t)c | (nk_size_t)result) & 15);             \
        nk_size_t const chunks = aligned ? arguments.count / chunk_values : 0;                                    \
        uint4 inputs[3], output;                                                                                  \
        nk_b128_vec_t const *const input_vectors = (nk_b128_vec_t const *)inputs;                                 \
        nk_b128_vec_t *const output_vector = (nk_b128_vec_t *)&output;                                            \
        _Pragma("unroll 1") for (nk_size_t chunk = first; chunk < chunks; chunk += threads) {                     \
            inputs[0] = ((uint4 const *)a)[chunk];                                                                \
            inputs[1] = operands > 1 ? ((uint4 const *)b)[chunk] : inputs[0];                                     \
            inputs[2] = operands > 2 ? ((uint4 const *)c)[chunk] : inputs[0];                                     \
            _Pragma("unroll") for (unsigned offset = 0; offset != chunk_values; ++offset)                         \
                nk_each_##verb##_##input_type##_##isa_suffix##_value_(                                            \
                    input_vectors[0].input_type##s + offset, input_vectors[1].input_type##s + offset,             \
                    input_vectors[2].input_type##s + offset, alpha, beta, output_vector->input_type##s + offset); \
            ((uint4 *)result)[chunk] = output;                                                                    \
        }                                                                                                         \
        for (nk_size_t index = chunks * chunk_values + first; index < arguments.count; index += threads)          \
            nk_each_##verb##_##input_type##_##isa_suffix##_value_(a + index, b + index, c + index, alpha, beta,   \
                                                                  result + index);                                \
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

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_
#endif // NUMKONG_EACH_SIMT_CUH
