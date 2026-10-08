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

#if NUMKONG_ARCH_ROCM_

#include "numkong/rocm.cuh"
#include "numkong/each/simt.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

/** Launches @p blocks_wanted blocks of @p threads running @p kernel, but at most four times as many
 *  as stay resident across the stream's device, passing the argument struct at @p arguments. */
NUMKONG_INLINE nk_status_t nk_each_launch_rocm_(void const *kernel, unsigned threads, nk_size_t blocks_wanted,
                                                void *arguments, nk_stream_t stream) {
    int caller = 0;
    nk_status_t const entered = nk_device_enter_rocm_(stream, &caller);
    if (entered != nk_success_k) return entered;
    int device = 0, multiprocessors = 0, per_multiprocessor = 0;
    if (hipGetDevice(&device) != hipSuccess ||
        hipDeviceGetAttribute(&multiprocessors, hipDeviceAttributeMultiprocessorCount, device) != hipSuccess ||
        hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_multiprocessor, kernel, (int)threads, 0) != hipSuccess) {
        nk_device_leave_rocm_(caller);
        return nk_device_code_mismatch_k;
    }
    nk_size_t const oversubscribed = 4 * (nk_size_t)multiprocessors * (nk_size_t)per_multiprocessor;
    nk_size_t const blocks = oversubscribed < blocks_wanted ? oversubscribed : blocks_wanted;
    void *launch_arguments[1];
    launch_arguments[0] = arguments;
    dim3 grid, block;
    grid.x = (unsigned)blocks, grid.y = 1, grid.z = 1;
    block.x = threads, block.y = 1, block.z = 1;
    hipError_t const status = blocks ? hipLaunchKernel(kernel, grid, block, launch_arguments, 0, (hipStream_t)stream)
                                     : hipErrorInvalidConfiguration;
    nk_device_leave_rocm_(caller);
    return status == hipSuccess ? nk_success_k : nk_device_code_mismatch_k;
}

NUMKONG_INLINE nk_status_t nk_each_swiglu_launch_rocm_(void const *kernel, nk_size_t value_bytes, void const *gate,
                                                       void const *up, void *y, nk_size_t rows, nk_size_t columns,
                                                       nk_size_t gate_stride, nk_size_t up_stride, nk_size_t y_stride,
                                                       nk_f32_t gate_scale, nk_f32_t output_scale, nk_stream_t stream) {
    if ((((nk_size_t)gate) | gate_stride | ((nk_size_t)up) | (up ? up_stride : 0) | ((nk_size_t)y) | y_stride) &
        (value_bytes - 1))
        return nk_misaligned_k;
    if (rows == 0 || columns == 0) return nk_success_k;
    nk_each_swiglu_arguments_t arguments;
    arguments.gate = (unsigned char const *)gate, arguments.up = (unsigned char const *)up;
    arguments.y = (unsigned char *)y, arguments.rows = rows, arguments.columns = columns;
    arguments.gate_stride = gate_stride, arguments.up_stride = up_stride, arguments.y_stride = y_stride;
    arguments.gate_scale = gate_scale, arguments.output_scale = output_scale;
    nk_size_t const block_teams = nk_each_threads_simt_k / nk_each_team_lanes_simt_k;
    return nk_each_launch_rocm_(kernel, nk_each_threads_simt_k, nk_size_divide_round_up_(rows, block_teams), &arguments,
                                stream);
}

/** Validates the contract, @p value_bytes alignment for the vectors and @p scale_bytes for @p alpha
 *  and @p beta, and launches @p kernel over @p n elements. */
NUMKONG_INLINE nk_status_t nk_each_elementwise_launch_rocm_(void const *kernel, nk_size_t value_bytes,
                                                            nk_size_t scale_bytes, void const *a, void const *b,
                                                            void const *c, void const *alpha, void const *beta,
                                                            nk_size_t n, void *result, nk_stream_t stream) {
    nk_size_t const addresses = ((nk_size_t)a) | ((nk_size_t)b) | ((nk_size_t)c) | ((nk_size_t)result);
    if ((addresses & (value_bytes - 1)) || (((nk_size_t)alpha) | ((nk_size_t)beta)) & (scale_bytes - 1))
        return nk_misaligned_k;
    if (n == 0) return nk_success_k;
    nk_each_elementwise_arguments_t arguments;
    arguments.a = (unsigned char const *)a, arguments.b = (unsigned char const *)b;
    arguments.c = (unsigned char const *)c, arguments.alpha = alpha, arguments.beta = beta;
    arguments.result = (unsigned char *)result, arguments.count = n;
    nk_size_t const block_values = nk_each_threads_simt_k * (addresses & 15 ? 1 : 16 / value_bytes);
    return nk_each_launch_rocm_(kernel, nk_each_threads_simt_k, nk_size_divide_round_up_(n, block_values), &arguments,
                                stream);
}

/** Validates the contract and launches one block per normalized vector, of enough whole wavefronts
 *  that each thread keeps at most @c nk_each_kept_pieces_simt_k 16-byte pieces of it. */
NUMKONG_INLINE nk_status_t nk_each_rmsnorm_launch_rocm_(void const *kernel, nk_size_t input_bytes,
                                                        nk_size_t output_bytes, void const *x, nk_f32_t const *gamma,
                                                        void *y, nk_size_t rows, nk_size_t groups, nk_size_t columns,
                                                        nk_size_t x_stride, nk_size_t y_stride, nk_f32_t epsilon,
                                                        nk_stream_t stream) {
    nk_size_t const narrower_bytes = input_bytes < output_bytes ? input_bytes : output_bytes;
    nk_size_t const chunk_values = 16 / narrower_bytes, chunk_pieces = chunk_values * input_bytes / 16;
    if ((((nk_size_t)x) | x_stride) & (input_bytes - 1) || (((nk_size_t)y) | y_stride) & (output_bytes - 1) ||
        ((nk_size_t)gamma & 3))
        return nk_misaligned_k;
    nk_each_rmsnorm_arguments_t arguments;
    arguments.x = (unsigned char const *)x, arguments.gamma = gamma, arguments.y = (unsigned char *)y;
    arguments.groups = groups, arguments.columns = columns, arguments.vectors = rows * groups;
    arguments.x_stride = x_stride, arguments.y_stride = y_stride, arguments.epsilon = epsilon;
    if (arguments.vectors == 0 || columns == 0) return nk_success_k;
    // CDNA wavefronts hold 64 lanes, two teams each
    nk_size_t const chunks = nk_size_divide_round_up_(columns, chunk_values);
    nk_size_t const kept_chunks = nk_each_kept_pieces_simt_k / chunk_pieces;
    nk_size_t threads = nk_size_round_up_to_multiple_(nk_size_divide_round_up_(chunks, kept_chunks), 64);
    if (threads > nk_each_threads_simt_k) threads = nk_each_threads_simt_k;
    return nk_each_launch_rocm_(kernel, (unsigned)threads, arguments.vectors, &arguments, stream);
}

/** Generates SwiGLU of @p input_type for @p isa_suffix, after @c nk_define_each_swiglu_ of the
 *  serial backend: the entry point, the kernel with one team per row, and the helper for every
 *  value. The helper multiplies SiLU of the scaled gate by the up value, unless @c up is NULL, and
 *  by the output scale. */
#define nk_define_each_swiglu_rocm_(input_type, isa_suffix, load_and_convert, convert_and_store)                       \
    NUMKONG_DEVICE void nk_each_swiglu_##input_type##_##isa_suffix##_value_(                                           \
        nk_##input_type##_t const *gate, nk_##input_type##_t const *up, nk_f32_t gate_scale, nk_f32_t output_scale,    \
        nk_##input_type##_t *y) {                                                                                      \
        nk_f32_t gate_value, up_value = 1.0f;                                                                          \
        load_and_convert(gate, &gate_value);                                                                           \
        if (up) load_and_convert(up, &up_value);                                                                       \
        nk_f32_t const scaled = gate_value * gate_scale;                                                               \
        nk_f32_t const result = scaled / (1.0f + expf(-scaled)) * up_value * output_scale;                             \
        convert_and_store(&result, y);                                                                                 \
    }                                                                                                                  \
    static __global__ void nk_each_swiglu_##input_type##_##isa_suffix##_kernel_(                                       \
        nk_each_swiglu_arguments_t arguments) {                                                                        \
        unsigned const chunk_values = 16 / sizeof(nk_##input_type##_t),                                                \
                       lane = threadIdx.x % nk_each_team_lanes_simt_k;                                                 \
        nk_size_t const block_teams = blockDim.x / nk_each_team_lanes_simt_k, teams = gridDim.x * block_teams;         \
        nk_size_t const up_stride = arguments.up ? arguments.up_stride : 0;                                            \
        int const aligned = !(((nk_size_t)arguments.gate | arguments.gate_stride | (nk_size_t)arguments.up |           \
                               up_stride | (nk_size_t)arguments.y | arguments.y_stride) &                              \
                              15);                                                                                     \
        nk_size_t const chunks = aligned ? arguments.columns / chunk_values : 0;                                       \
        uint4 inputs[2], output;                                                                                       \
        nk_b128_vec_t const *const input_vectors = (nk_b128_vec_t const *)inputs;                                      \
        nk_b128_vec_t *const output_vector = (nk_b128_vec_t *)&output;                                                 \
        for (nk_size_t row = blockIdx.x * block_teams + threadIdx.x / nk_each_team_lanes_simt_k; row < arguments.rows; \
             row += teams) {                                                                                           \
            unsigned char const *gate_row = arguments.gate + row * arguments.gate_stride;                              \
            unsigned char const *up_row = arguments.up ? arguments.up + row * arguments.up_stride : NUMKONG_NULL;      \
            nk_##input_type##_t const *gate = (nk_##input_type##_t const *)gate_row;                                   \
            nk_##input_type##_t const *up = (nk_##input_type##_t const *)up_row;                                       \
            nk_##input_type##_t *y = (nk_##input_type##_t *)(arguments.y + row * arguments.y_stride);                  \
            uint4 const *gate_chunks = (uint4 const *)gate;                                                            \
            uint4 const *up_chunks = (uint4 const *)up;                                                                \
            uint4 *y_chunks = (uint4 *)y;                                                                              \
            for (nk_size_t chunk = lane; chunk < chunks; chunk += nk_each_team_lanes_simt_k) {                         \
                inputs[0] = gate_chunks[chunk];                                                                        \
                if (up) inputs[1] = up_chunks[chunk];                                                                  \
                _Pragma("unroll") for (unsigned offset = 0; offset != chunk_values; ++offset)                          \
                    nk_each_swiglu_##input_type##_##isa_suffix##_value_(                                               \
                        input_vectors[0].input_type##s + offset,                                                       \
                        up ? input_vectors[1].input_type##s + offset : NUMKONG_NULL, arguments.gate_scale,             \
                        arguments.output_scale, output_vector->input_type##s + offset);                                \
                y_chunks[chunk] = output;                                                                              \
            }                                                                                                          \
            for (nk_size_t column = chunks * chunk_values + lane; column < arguments.columns;                          \
                 column += nk_each_team_lanes_simt_k)                                                                  \
                nk_each_swiglu_##input_type##_##isa_suffix##_value_(gate + column, up ? up + column : NUMKONG_NULL,    \
                                                                    arguments.gate_scale, arguments.output_scale,      \
                                                                    y + column);                                       \
        }                                                                                                              \
    }                                                                                                                  \
    NUMKONG_API nk_status_t nk_each_swiglu_##input_type##_##isa_suffix(                                                \
        nk_##input_type##_t const *gate, nk_##input_type##_t const *up, nk_##input_type##_t *y, nk_size_t rows,        \
        nk_size_t columns, nk_size_t gate_stride, nk_size_t up_stride, nk_size_t y_stride, nk_f32_t gate_scale,        \
        nk_f32_t output_scale, nk_stream_t stream) {                                                                   \
        return nk_each_swiglu_launch_rocm_((void const *)&nk_each_swiglu_##input_type##_##isa_suffix##_kernel_,        \
                                           sizeof(nk_##input_type##_t), gate, up, y, rows, columns, gate_stride,       \
                                           up_stride, y_stride, gate_scale, output_scale, stream);                     \
    }

/** Generates the element-wise sum of @p input_type for @p isa_suffix, after @c nk_define_each_sum_
 *  of the serial backend: the helper for every value, its kernel and the entry point. */
#define nk_define_each_sum_rocm_(input_type, accumulator_type, isa_suffix, load_and_convert, convert_and_store)        \
    NUMKONG_DEVICE void nk_each_sum_##input_type##_##isa_suffix##_value_(                                              \
        nk_##input_type##_t const *a, nk_##input_type##_t const *b, nk_##input_type##_t const *c,                      \
        nk_##accumulator_type##_t alpha, nk_##accumulator_type##_t beta, nk_##input_type##_t *result) {                \
        nk_##accumulator_type##_t a_value, b_value;                                                                    \
        nk_unused_(c), nk_unused_(alpha), nk_unused_(beta);                                                            \
        load_and_convert(a, &a_value), load_and_convert(b, &b_value);                                                  \
        nk_##accumulator_type##_t const sum = a_value + b_value;                                                       \
        convert_and_store(&sum, result);                                                                               \
    }                                                                                                                  \
    nk_define_each_elementwise_kernel_simt_(sum, 2, input_type, accumulator_type, isa_suffix)                          \
    NUMKONG_API nk_status_t nk_each_sum_##input_type##_##isa_suffix(nk_##input_type##_t const *a,                      \
                                                                    nk_##input_type##_t const *b, nk_size_t n,         \
                                                                    nk_##input_type##_t *result, nk_stream_t stream) { \
        return nk_each_elementwise_launch_rocm_((void const *)&nk_each_sum_##input_type##_##isa_suffix##_kernel_,      \
                                                sizeof(nk_##input_type##_t), 1, a, b, NUMKONG_NULL, NUMKONG_NULL,      \
                                                NUMKONG_NULL, n, result, stream);                                      \
    }

/** Adds the square of @p value to the compensated @p sum: TwoProduct through FMA, then TwoSum.
 *  @p exact_squares skips the product's error for the values narrower than F32, as F32 holds
 *  their squares exactly. */
NUMKONG_DEVICE void nk_each_square_add_rocm_(nk_f32_t value, int exact_squares, nk_f32_t *sum, nk_f32_t *compensation) {
    nk_f32_t const square = nk_f32_mul_rn_rocm_(value, value);
    if (!exact_squares) *compensation += fmaf(value, value, -square);
    nk_f32_two_sum_simt_(square, sum, compensation);
}

/** Merges the compensated sums of squares of every thread of the block into the inverse RMS of
 *  @p count values, the same in every thread: across each team's lanes through xor shuffles, then
 *  across teams through @p partials, two values per team, in the same order everywhere. */
NUMKONG_DEVICE nk_f32_t nk_each_inverse_rms_rocm_(nk_f32_t sum, nk_f32_t compensation, nk_size_t count,
                                                  nk_f32_t epsilon, nk_f32_t *partials) {
    unsigned const teams = blockDim.x / nk_each_team_lanes_simt_k;
    for (unsigned offset = nk_each_team_lanes_simt_k / 2; offset; offset >>= 1) {
        nk_f32_t const other_sum = nk_shuffle_xor_f32_rocm_(sum, offset);
        compensation += nk_shuffle_xor_f32_rocm_(compensation, offset);
        nk_f32_two_sum_simt_(other_sum, &sum, &compensation);
    }
    if (teams > 1) {
        unsigned const team = threadIdx.x / nk_each_team_lanes_simt_k;
        if (threadIdx.x % nk_each_team_lanes_simt_k == 0) partials[team] = sum, partials[teams + team] = compensation;
        __syncthreads();
        sum = partials[0], compensation = partials[teams];
        for (unsigned other = 1; other < teams; ++other) {
            compensation += partials[teams + other];
            nk_f32_two_sum_simt_(partials[other], &sum, &compensation);
        }
        __syncthreads();
    }
    nk_f32_t const mean_square = (nk_f32_t)(((nk_f64_t)sum + (nk_f64_t)compensation) / (nk_f64_t)count);
    return 1.0f / sqrtf(mean_square + epsilon);
}

#if NUMKONG_TARGET_ROCM
nk_define_each_sum_rocm_(f32, f32, rocm, nk_assign_from_to_, nk_assign_from_to_)
nk_define_each_sum_rocm_(f16, f32, rocm, nk_f16_to_f32_simt_, nk_f32_to_f16_simt_)
nk_define_each_sum_rocm_(bf16, f32, rocm, nk_bf16_to_f32_simt_, nk_f32_to_bf16_simt_)
nk_define_each_rmsnorm_simt_(rmsnorm, f32, f32, rocm, rocm, nk_assign_from_to_, nk_assign_from_to_)
nk_define_each_rmsnorm_simt_(rmsnorm, bf16, bf16, rocm, rocm, nk_bf16_to_f32_simt_, nk_f32_to_bf16_simt_)
nk_define_each_rmsnorm_simt_(rmsnorm, e4m3, e4m3, rocm, rocm, nk_e4m3_to_f32_simt_, nk_f32_to_e4m3_simt_)
nk_define_each_rmsnorm_simt_(rmscast, f32, bf16, rocm, rocm, nk_assign_from_to_, nk_f32_to_bf16_simt_)
nk_define_each_rmsnorm_simt_(rmscast, f32, f16, rocm, rocm, nk_assign_from_to_, nk_f32_to_f16_simt_)
nk_define_each_rmsnorm_simt_(rmscast, f32, e4m3, rocm, rocm, nk_assign_from_to_, nk_f32_to_e4m3_simt_)
nk_define_each_rmsnorm_simt_(rmscast, f32, e5m2, rocm, rocm, nk_assign_from_to_, nk_f32_to_e5m2_simt_)
nk_define_each_rmsnorm_simt_(rmscast, f32, e2m3, rocm, rocm, nk_assign_from_to_, nk_f32_to_e2m3_simt_)
nk_define_each_rmsnorm_simt_(rmscast, f32, e3m2, rocm, rocm, nk_assign_from_to_, nk_f32_to_e3m2_simt_)
nk_define_each_rmsnorm_simt_(rmscast, f64, f32, rocm, rocm, nk_f64_to_f32_simt_, nk_assign_from_to_)
nk_define_each_rmsnorm_simt_(rmscast, i32, i8, rocm, rocm, nk_assign_from_to_, nk_f32_to_i8_simt_)
nk_define_each_rmsnorm_simt_(rmscast, u32, u8, rocm, rocm, nk_assign_from_to_, nk_f32_to_u8_simt_)
nk_define_each_swiglu_rocm_(f32, rocm, nk_assign_from_to_, nk_assign_from_to_)
nk_define_each_swiglu_rocm_(bf16, rocm, nk_bf16_to_f32_simt_, nk_f32_to_bf16_simt_)
nk_define_each_swiglu_rocm_(e4m3, rocm, nk_e4m3_to_f32_simt_, nk_f32_to_e4m3_simt_)
#endif // NUMKONG_TARGET_ROCM

#undef nk_define_each_sum_rocm_
#undef nk_define_each_swiglu_rocm_

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_ROCM_
#endif // NUMKONG_EACH_ROCM_CUH
