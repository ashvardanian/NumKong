/**
 *  @file include/numkong/each/cuda.cuh
 *  @author Ash Vardanian
 *  @date October 3, 2026
 *  @brief CUDA host side of element-wise scales, sums, blends, FMAs, RMSNorm and SwiGLU: the
 *      launches, the export generators and the @c cuda exports, over the kernels of `each/simt.cuh`,
 *      with the integer and complex kernels only NVIDIA runs.
 *
 *  The @c ampere and @c ada capabilities export through the same generators, with converters that
 *  use their conversion instructions.
 *
 *  @sa include/numkong/each/simt.cuh
 *  @sa include/numkong/each/rocm.cuh
 *  @sa include/numkong/each/ampere.cuh
 *  @sa include/numkong/each/ada.cuh
 */
#ifndef NUMKONG_EACH_CUDA_CUH
#define NUMKONG_EACH_CUDA_CUH

#if NUMKONG_ARCH_CUDA_

#include "numkong/cuda.cuh"
#include "numkong/each/simt.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

/** Launches @p blocks_wanted blocks of @p threads running @p kernel, but at most four times as many
 *  as stay resident across the stream's device, passing the argument struct at @p arguments. */
NUMKONG_INLINE nk_status_t nk_each_launch_cuda_(void const *kernel, unsigned threads, nk_size_t blocks_wanted,
                                                void *arguments, void *stream) {
    int caller = 0;
    nk_status_t const entered = nk_device_enter_cuda_(stream, &caller);
    if (entered != nk_success_k) return entered;
    int device = 0, multiprocessors = 0, per_multiprocessor = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device) != cudaSuccess ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_multiprocessor, kernel, (int)threads, 0) != cudaSuccess) {
        nk_device_leave_cuda_(caller);
        return nk_device_code_mismatch_k;
    }
    nk_size_t const oversubscribed = 4 * (nk_size_t)multiprocessors * (nk_size_t)per_multiprocessor;
    nk_size_t const blocks = oversubscribed < blocks_wanted ? oversubscribed : blocks_wanted;
    void *launch_arguments[1];
    launch_arguments[0] = arguments;
    dim3 grid, block;
    grid.x = (unsigned)blocks, grid.y = 1, grid.z = 1;
    block.x = threads, block.y = 1, block.z = 1;
    cudaError_t const status = blocks ? cudaLaunchKernel(kernel, grid, block, launch_arguments, 0, (cudaStream_t)stream)
                                      : cudaErrorInvalidConfiguration;
    nk_device_leave_cuda_(caller);
    return status == cudaSuccess ? nk_success_k : nk_device_code_mismatch_k;
}

NUMKONG_INLINE nk_status_t nk_each_swiglu_launch_cuda_(void const *kernel, nk_size_t value_bytes, void const *gate,
                                                       void const *up, void *y, nk_size_t rows, nk_size_t columns,
                                                       nk_size_t gate_stride, nk_size_t up_stride, nk_size_t y_stride,
                                                       nk_f32_t gate_scale, nk_f32_t output_scale, void *stream) {
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
    return nk_each_launch_cuda_(kernel, nk_each_threads_simt_k, nk_size_divide_round_up_(rows, block_teams), &arguments,
                                stream);
}

/** Validates the contract, @p value_bytes alignment for the vectors and @p scale_bytes for @p alpha
 *  and @p beta, and launches @p kernel over @p n elements. */
NUMKONG_INLINE nk_status_t nk_each_elementwise_launch_cuda_(void const *kernel, nk_size_t value_bytes,
                                                            nk_size_t scale_bytes, void const *a, void const *b,
                                                            void const *c, void const *alpha, void const *beta,
                                                            nk_size_t n, void *result, void *stream) {
    nk_size_t const addresses = ((nk_size_t)a) | ((nk_size_t)b) | ((nk_size_t)c) | ((nk_size_t)result);
    if ((addresses & (value_bytes - 1)) || (((nk_size_t)alpha) | ((nk_size_t)beta)) & (scale_bytes - 1))
        return nk_misaligned_k;
    if (n == 0) return nk_success_k;
    nk_each_elementwise_arguments_t arguments;
    arguments.a = (unsigned char const *)a, arguments.b = (unsigned char const *)b;
    arguments.c = (unsigned char const *)c, arguments.alpha = alpha, arguments.beta = beta;
    arguments.result = (unsigned char *)result, arguments.count = n;
    nk_size_t const block_values = nk_each_threads_simt_k * (addresses & 15 ? 1 : 16 / value_bytes);
    return nk_each_launch_cuda_(kernel, nk_each_threads_simt_k, nk_size_divide_round_up_(n, block_values), &arguments,
                                stream);
}

/** Validates the contract and launches one block per normalized vector, of enough teams that each
 *  thread keeps at most @c nk_each_kept_pieces_simt_k 16-byte pieces of it. */
NUMKONG_INLINE nk_status_t nk_each_rmsnorm_launch_cuda_(void const *kernel, nk_size_t input_bytes,
                                                        nk_size_t output_bytes, void const *x, nk_f32_t const *gamma,
                                                        void *y, nk_size_t rows, nk_size_t groups, nk_size_t columns,
                                                        nk_size_t x_stride, nk_size_t y_stride, nk_f32_t epsilon,
                                                        void *stream) {
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
    nk_size_t const chunks = nk_size_divide_round_up_(columns, chunk_values);
    nk_size_t const kept_chunks = nk_each_kept_pieces_simt_k / chunk_pieces;
    nk_size_t threads = nk_size_round_up_to_multiple_(nk_size_divide_round_up_(chunks, kept_chunks),
                                                      nk_each_team_lanes_simt_k);
    if (threads > nk_each_threads_simt_k) threads = nk_each_threads_simt_k;
    return nk_each_launch_cuda_(kernel, (unsigned)threads, arguments.vectors, &arguments, stream);
}

/** Generates SwiGLU of @p input_type for @p isa_suffix, after @c nk_define_each_swiglu_ of the
 *  serial backend: the entry point, the kernel with one team per row, and the helper for every
 *  value. The helper multiplies SiLU of the scaled gate by the up value, unless @c up is NULL, and
 *  by the output scale. */
#define nk_define_each_swiglu_cuda_(input_type, isa_suffix, load_and_convert, convert_and_store)                       \
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
            for (nk_size_t chunk = lane; chunk < chunks; chunk += nk_each_team_lanes_simt_k) {                         \
                inputs[0] = ((uint4 const *)gate)[chunk];                                                              \
                if (up) inputs[1] = ((uint4 const *)up)[chunk];                                                        \
                _Pragma("unroll") for (unsigned offset = 0; offset != chunk_values; ++offset)                          \
                    nk_each_swiglu_##input_type##_##isa_suffix##_value_(                                               \
                        input_vectors[0].input_type##s + offset,                                                       \
                        up ? input_vectors[1].input_type##s + offset : NUMKONG_NULL, arguments.gate_scale,             \
                        arguments.output_scale, output_vector->input_type##s + offset);                                \
                ((uint4 *)y)[chunk] = output;                                                                          \
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
        nk_f32_t output_scale, void *stream) {                                                                         \
        return nk_each_swiglu_launch_cuda_((void const *)&nk_each_swiglu_##input_type##_##isa_suffix##_kernel_,        \
                                           sizeof(nk_##input_type##_t), gate, up, y, rows, columns, gate_stride,       \
                                           up_stride, y_stride, gate_scale, output_scale, stream);                     \
    }

/** Generates the element-wise sum of @p input_type for @p isa_suffix, after @c nk_define_each_sum_
 *  of the serial backend: the helper for every value, its kernel and the entry point. */
#define nk_define_each_sum_cuda_(input_type, accumulator_type, isa_suffix, load_and_convert, convert_and_store)       \
    NUMKONG_DEVICE void nk_each_sum_##input_type##_##isa_suffix##_value_(                                             \
        nk_##input_type##_t const *a, nk_##input_type##_t const *b, nk_##input_type##_t const *c,                     \
        nk_##accumulator_type##_t alpha, nk_##accumulator_type##_t beta, nk_##input_type##_t *result) {               \
        nk_##accumulator_type##_t a_value, b_value;                                                                   \
        nk_unused_(c), nk_unused_(alpha), nk_unused_(beta);                                                           \
        load_and_convert(a, &a_value), load_and_convert(b, &b_value);                                                 \
        nk_##accumulator_type##_t const sum = a_value + b_value;                                                      \
        convert_and_store(&sum, result);                                                                              \
    }                                                                                                                 \
    nk_define_each_elementwise_kernel_simt_(sum, 2, input_type, accumulator_type, isa_suffix) NUMKONG_API nk_status_t \
    nk_each_sum_##input_type##_##isa_suffix(nk_##input_type##_t const *a, nk_##input_type##_t const *b, nk_size_t n,  \
                                            nk_##input_type##_t *result, void *stream) {                              \
        return nk_each_elementwise_launch_cuda_((void const *)&nk_each_sum_##input_type##_##isa_suffix##_kernel_,     \
                                                sizeof(nk_##input_type##_t), 1, a, b, NUMKONG_NULL, NUMKONG_NULL,     \
                                                NUMKONG_NULL, n, result, stream);                                     \
    }

/** Generates the element-wise scale of @p input_type, α · a + β, for @p isa_suffix, after
 *  @c nk_define_each_scale_ of the serial backend: the entry point, its kernel and the helper. */
#define nk_define_each_scale_cuda_(input_type, accumulator_type, isa_suffix, load_and_convert, convert_and_store)   \
    NUMKONG_DEVICE void nk_each_scale_##input_type##_##isa_suffix##_value_(                                         \
        nk_##input_type##_t const *a, nk_##input_type##_t const *b, nk_##input_type##_t const *c,                   \
        nk_##accumulator_type##_t alpha, nk_##accumulator_type##_t beta, nk_##input_type##_t *result) {             \
        nk_##accumulator_type##_t a_value;                                                                          \
        nk_unused_(b), nk_unused_(c);                                                                               \
        load_and_convert(a, &a_value);                                                                              \
        nk_##accumulator_type##_t const scaled = alpha * a_value + beta;                                            \
        convert_and_store(&scaled, result);                                                                         \
    }                                                                                                               \
    nk_define_each_elementwise_kernel_simt_(scale, 1, input_type, accumulator_type, isa_suffix)                     \
        NUMKONG_API nk_status_t                                                                                     \
        nk_each_scale_##input_type##_##isa_suffix(                                                                  \
            nk_##input_type##_t const *a, nk_size_t n, nk_##accumulator_type##_t const *alpha,                      \
            nk_##accumulator_type##_t const *beta, nk_##input_type##_t *result, void *stream) {                     \
        return nk_each_elementwise_launch_cuda_((void const *)&nk_each_scale_##input_type##_##isa_suffix##_kernel_, \
                                                sizeof(nk_##input_type##_t), sizeof(nk_##accumulator_type##_t), a,  \
                                                NUMKONG_NULL, NUMKONG_NULL, alpha, beta, n, result, stream);        \
    }

/** Generates the element-wise blend of @p input_type, α · a + β · b, for @p isa_suffix, after
 *  @c nk_define_each_blend_ of the serial backend: the entry point, its kernel and the helper. */
#define nk_define_each_blend_cuda_(input_type, accumulator_type, isa_suffix, load_and_convert, convert_and_store)     \
    NUMKONG_DEVICE void nk_each_blend_##input_type##_##isa_suffix##_value_(                                           \
        nk_##input_type##_t const *a, nk_##input_type##_t const *b, nk_##input_type##_t const *c,                     \
        nk_##accumulator_type##_t alpha, nk_##accumulator_type##_t beta, nk_##input_type##_t *result) {               \
        nk_##accumulator_type##_t a_value, b_value;                                                                   \
        nk_unused_(c);                                                                                                \
        load_and_convert(a, &a_value), load_and_convert(b, &b_value);                                                 \
        nk_##accumulator_type##_t const blended = a_value * alpha + b_value * beta;                                   \
        convert_and_store(&blended, result);                                                                          \
    }                                                                                                                 \
    nk_define_each_elementwise_kernel_simt_(blend, 2, input_type, accumulator_type, isa_suffix)                       \
        NUMKONG_API nk_status_t                                                                                       \
        nk_each_blend_##input_type##_##isa_suffix(nk_##input_type##_t const *a, nk_##input_type##_t const *b,         \
                                                  nk_size_t n, nk_##accumulator_type##_t const *alpha,                \
                                                  nk_##accumulator_type##_t const *beta, nk_##input_type##_t *result, \
                                                  void *stream) {                                                     \
        return nk_each_elementwise_launch_cuda_((void const *)&nk_each_blend_##input_type##_##isa_suffix##_kernel_,   \
                                                sizeof(nk_##input_type##_t), sizeof(nk_##accumulator_type##_t), a, b, \
                                                NUMKONG_NULL, alpha, beta, n, result, stream);                        \
    }

/** Generates the element-wise FMA of @p input_type, α · a · b + β · c, for @p isa_suffix, after
 *  @c nk_define_each_fma_ of the serial backend: the entry point, its kernel and the helper. */
#define nk_define_each_fma_cuda_(input_type, accumulator_type, isa_suffix, load_and_convert, convert_and_store)       \
    NUMKONG_DEVICE void nk_each_fma_##input_type##_##isa_suffix##_value_(                                             \
        nk_##input_type##_t const *a, nk_##input_type##_t const *b, nk_##input_type##_t const *c,                     \
        nk_##accumulator_type##_t alpha, nk_##accumulator_type##_t beta, nk_##input_type##_t *result) {               \
        nk_##accumulator_type##_t a_value, b_value, c_value;                                                          \
        load_and_convert(a, &a_value), load_and_convert(b, &b_value), load_and_convert(c, &c_value);                  \
        nk_##accumulator_type##_t const fused = a_value * b_value * alpha + c_value * beta;                           \
        convert_and_store(&fused, result);                                                                            \
    }                                                                                                                 \
    nk_define_each_elementwise_kernel_simt_(fma, 3, input_type, accumulator_type, isa_suffix) NUMKONG_API nk_status_t \
    nk_each_fma_##input_type##_##isa_suffix(                                                                          \
        nk_##input_type##_t const *a, nk_##input_type##_t const *b, nk_##input_type##_t const *c, nk_size_t n,        \
        nk_##accumulator_type##_t const *alpha, nk_##accumulator_type##_t const *beta, nk_##input_type##_t *result,   \
        void *stream) {                                                                                               \
        return nk_each_elementwise_launch_cuda_((void const *)&nk_each_fma_##input_type##_##isa_suffix##_kernel_,     \
                                                sizeof(nk_##input_type##_t), sizeof(nk_##accumulator_type##_t), a, b, \
                                                c, alpha, beta, n, result, stream);                                   \
    }

/** Adds the square of @p value to the compensated @p sum: TwoProduct through FMA, then TwoSum.
 *  @p exact_squares skips the product's error for the values narrower than F32, as F32 holds
 *  their squares exactly. */
NUMKONG_DEVICE void nk_each_square_add_cuda_(nk_f32_t value, int exact_squares, nk_f32_t *sum, nk_f32_t *compensation) {
    nk_f32_t const square = nk_f32_mul_rn_cuda_(value, value);
    if (!exact_squares) *compensation += fmaf(value, value, -square);
    nk_f32_two_sum_simt_(square, sum, compensation);
}

/** Merges the compensated sums of squares of every thread of the block into the inverse RMS of
 *  @p count values, the same in every thread: across each team's lanes through xor shuffles, then
 *  across teams through @p partials, two values per team, in the same order everywhere. */
NUMKONG_DEVICE nk_f32_t nk_each_inverse_rms_cuda_(nk_f32_t sum, nk_f32_t compensation, nk_size_t count,
                                                  nk_f32_t epsilon, nk_f32_t *partials) {
    unsigned const teams = blockDim.x / nk_each_team_lanes_simt_k;
    for (unsigned offset = nk_each_team_lanes_simt_k / 2; offset; offset >>= 1) {
        nk_f32_t const other_sum = nk_shuffle_xor_f32_cuda_(sum, offset);
        compensation += nk_shuffle_xor_f32_cuda_(compensation, offset);
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

/** Generates @p verb for @p isa_suffix, an RMSNorm of @p input_type vectors into @p output_type
 *  ones after @c nk_define_each_rmsnorm_ of the serial backend: the helper that normalizes one
 *  chunk out of its 16-byte @c pieces, the kernel, one block per vector, and the entry point. */
#define nk_define_each_rmsnorm_cuda_(verb, input_type, output_type, isa_suffix, load_and_convert, convert_and_store)   \
    NUMKONG_DEVICE void nk_each_##verb##_##output_type##_##isa_suffix##_chunk_(                                        \
        nk_b128_vec_t const *pieces, nk_f32_t const *gamma, nk_f32_t inverse_rms, nk_##output_type##_t *y) {           \
        unsigned const piece_values = 16 / sizeof(nk_##input_type##_t);                                                \
        unsigned const chunk_values = 16 / sizeof(nk_##output_type##_t) < piece_values                                 \
                                          ? piece_values                                                               \
                                          : 16 / sizeof(nk_##output_type##_t);                                         \
        uint4 gains[4], normalized;                                                                                    \
        nk_b128_vec_t const *const gain_vectors = (nk_b128_vec_t const *)gains;                                        \
        nk_##output_type##_t *const outputs = ((nk_b128_vec_t *)&normalized)->output_type##s;                          \
        nk_f32_t value, result;                                                                                        \
        if (gamma) {                                                                                                   \
            _Pragma("unroll") for (unsigned quad = 0; quad != chunk_values / 4; ++quad)                                \
                gains[quad] = ((uint4 const *)gamma)[quad];                                                            \
        }                                                                                                              \
        _Pragma("unroll") for (unsigned offset = 0; offset != chunk_values; ++offset) {                                \
            load_and_convert(pieces[offset / piece_values].input_type##s + offset % piece_values, &value);             \
            result = value * inverse_rms * (gamma ? gain_vectors[offset / 4].f32s[offset % 4] : 1.0f);                 \
            convert_and_store(&result, outputs + offset);                                                              \
        }                                                                                                              \
        *(uint4 *)y = normalized;                                                                                      \
    }                                                                                                                  \
    static __global__ void nk_each_##verb##_##output_type##_##isa_suffix##_kernel_(                                    \
        nk_each_rmsnorm_arguments_t arguments) {                                                                       \
        __shared__ nk_f32_t partials[2 * nk_each_threads_simt_k / nk_each_team_lanes_simt_k];                          \
        unsigned const piece_values = 16 / sizeof(nk_##input_type##_t);                                                \
        unsigned const chunk_values = 16 / sizeof(nk_##output_type##_t) < piece_values                                 \
                                          ? piece_values                                                               \
                                          : 16 / sizeof(nk_##output_type##_t);                                         \
        unsigned const chunk_pieces = chunk_values / piece_values;                                                     \
        unsigned const kept_chunks = nk_each_kept_pieces_simt_k / chunk_pieces;                                        \
        int const exact_squares = sizeof(nk_##input_type##_t) < sizeof(nk_f32_t);                                      \
        nk_f32_t const *gamma = arguments.gamma;                                                                       \
        int const aligned = arguments.columns % chunk_values == 0 &&                                                   \
                            !(((nk_size_t)arguments.x | arguments.x_stride | (nk_size_t)arguments.y |                  \
                               arguments.y_stride | (nk_size_t)gamma) &                                                \
                              15);                                                                                     \
        nk_size_t const chunks = aligned ? arguments.columns / chunk_values : 0;                                       \
        nk_b128_vec_t kept[nk_each_kept_pieces_simt_k], loaded[4];                                                     \
        for (nk_size_t vector = blockIdx.x; vector < arguments.vectors; vector += gridDim.x) {                         \
            nk_size_t const row = vector / arguments.groups, first = vector % arguments.groups * arguments.columns;    \
            nk_##input_type##_t const *x = (nk_##input_type##_t const *)(arguments.x + row * arguments.x_stride) +     \
                                           first;                                                                      \
            nk_##output_type##_t *y = (nk_##output_type##_t *)(arguments.y + row * arguments.y_stride) + first;        \
            nk_f32_t sum = 0, compensation = 0, value;                                                                 \
            _Pragma("unroll") for (unsigned piece = 0; piece != nk_each_kept_pieces_simt_k; ++piece) {                 \
                nk_size_t const chunk = threadIdx.x + piece / chunk_pieces * blockDim.x;                               \
                if (chunk >= chunks) continue;                                                                         \
                uint4 const bits = *(uint4 const *)(x + chunk * chunk_values + piece % chunk_pieces * piece_values);   \
                nk_b128_vec_t const vector = {.u32s = {bits.x, bits.y, bits.z, bits.w}};                               \
                kept[piece] = vector;                                                                                  \
            }                                                                                                          \
            _Pragma("unroll") for (unsigned piece = 0; piece != nk_each_kept_pieces_simt_k; ++piece) {                 \
                if (threadIdx.x + piece / chunk_pieces * blockDim.x >= chunks) break;                                  \
                _Pragma("unroll") for (unsigned offset = 0; offset != piece_values; ++offset) {                        \
                    load_and_convert(kept[piece].input_type##s + offset, &value);                                      \
                    nk_each_square_add_cuda_(value, exact_squares, &sum, &compensation);                               \
                }                                                                                                      \
            }                                                                                                          \
            for (nk_size_t chunk = threadIdx.x + kept_chunks * blockDim.x; chunk < chunks; chunk += blockDim.x)        \
                _Pragma("unroll") for (unsigned piece = 0; piece != chunk_pieces; ++piece) {                           \
                    uint4 const bits = *(uint4 const *)(x + chunk * chunk_values + piece * piece_values);              \
                    nk_b128_vec_t const vector = {.u32s = {bits.x, bits.y, bits.z, bits.w}};                           \
                    loaded[0] = vector;                                                                                \
                    _Pragma("unroll") for (unsigned offset = 0; offset != piece_values; ++offset) {                    \
                        load_and_convert(loaded[0].input_type##s + offset, &value);                                    \
                        nk_each_square_add_cuda_(value, exact_squares, &sum, &compensation);                           \
                    }                                                                                                  \
                }                                                                                                      \
            for (nk_size_t column = chunks * chunk_values + threadIdx.x; column < arguments.columns;                   \
                 column += blockDim.x) {                                                                               \
                load_and_convert(x + column, &value);                                                                  \
                nk_each_square_add_cuda_(value, exact_squares, &sum, &compensation);                                   \
            }                                                                                                          \
            nk_f32_t const inverse_rms = nk_each_inverse_rms_cuda_(sum, compensation, arguments.columns,               \
                                                                   arguments.epsilon, partials);                       \
            _Pragma("unroll") for (unsigned slot = 0; slot != kept_chunks; ++slot) {                                   \
                nk_size_t const chunk = threadIdx.x + slot * blockDim.x;                                               \
                if (chunk >= chunks) break;                                                                            \
                nk_each_##verb##_##output_type##_##isa_suffix##_chunk_(                                                \
                    kept + slot * chunk_pieces, gamma ? gamma + chunk * chunk_values : NUMKONG_NULL, inverse_rms,      \
                    y + chunk * chunk_values);                                                                         \
            }                                                                                                          \
            for (nk_size_t chunk = threadIdx.x + kept_chunks * blockDim.x; chunk < chunks; chunk += blockDim.x) {      \
                _Pragma("unroll") for (unsigned piece = 0; piece != chunk_pieces; ++piece) {                           \
                    uint4 const bits = *(uint4 const *)(x + chunk * chunk_values + piece * piece_values);              \
                    nk_b128_vec_t const vector = {.u32s = {bits.x, bits.y, bits.z, bits.w}};                           \
                    loaded[piece] = vector;                                                                            \
                }                                                                                                      \
                nk_each_##verb##_##output_type##_##isa_suffix##_chunk_(                                                \
                    loaded, gamma ? gamma + chunk * chunk_values : NUMKONG_NULL, inverse_rms,                          \
                    y + chunk * chunk_values);                                                                         \
            }                                                                                                          \
            for (nk_size_t column = chunks * chunk_values + threadIdx.x; column < arguments.columns;                   \
                 column += blockDim.x) {                                                                               \
                load_and_convert(x + column, &value);                                                                  \
                nk_f32_t const result = value * inverse_rms * (gamma ? gamma[column] : 1.0f);                          \
                convert_and_store(&result, y + column);                                                                \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    NUMKONG_API nk_status_t nk_each_##verb##_##output_type##_##isa_suffix(                                             \
        nk_##input_type##_t const *x, nk_f32_t const *gamma, nk_##output_type##_t *y, nk_size_t rows,                  \
        nk_size_t groups, nk_size_t columns, nk_size_t x_stride, nk_size_t y_stride, nk_f32_t epsilon, void *stream) { \
        return nk_each_rmsnorm_launch_cuda_((void const *)&nk_each_##verb##_##output_type##_##isa_suffix##_kernel_,    \
                                            sizeof(nk_##input_type##_t), sizeof(nk_##output_type##_t), x, gamma, y,    \
                                            rows, groups, columns, x_stride, y_stride, epsilon, stream);               \
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

nk_define_each_rmsnorm_cuda_(rmsnorm, f32, f32, cuda, nk_assign_from_to_, nk_assign_from_to_)
nk_define_each_rmsnorm_cuda_(rmsnorm, bf16, bf16, cuda, nk_bf16_to_f32_simt_, nk_f32_to_bf16_simt_)
nk_define_each_rmsnorm_cuda_(rmsnorm, e4m3, e4m3, cuda, nk_e4m3_to_f32_simt_, nk_f32_to_e4m3_simt_)
nk_define_each_rmsnorm_cuda_(rmscast, f32, bf16, cuda, nk_assign_from_to_, nk_f32_to_bf16_simt_)
nk_define_each_rmsnorm_cuda_(rmscast, f32, f16, cuda, nk_assign_from_to_, nk_f32_to_f16_simt_)
nk_define_each_rmsnorm_cuda_(rmscast, f32, e4m3, cuda, nk_assign_from_to_, nk_f32_to_e4m3_simt_)
nk_define_each_rmsnorm_cuda_(rmscast, f32, e5m2, cuda, nk_assign_from_to_, nk_f32_to_e5m2_simt_)
nk_define_each_rmsnorm_cuda_(rmscast, f32, e2m3, cuda, nk_assign_from_to_, nk_f32_to_e2m3_simt_)
nk_define_each_rmsnorm_cuda_(rmscast, f32, e3m2, cuda, nk_assign_from_to_, nk_f32_to_e3m2_simt_)
nk_define_each_rmsnorm_cuda_(rmscast, f64, f32, cuda, nk_f64_to_f32_simt_, nk_assign_from_to_)
nk_define_each_rmsnorm_cuda_(rmscast, i32, i8, cuda, nk_assign_from_to_, nk_f32_to_i8_simt_)
nk_define_each_rmsnorm_cuda_(rmscast, u32, u8, cuda, nk_assign_from_to_, nk_f32_to_u8_simt_)
nk_define_each_swiglu_cuda_(f32, cuda, nk_assign_from_to_, nk_assign_from_to_)
nk_define_each_swiglu_cuda_(bf16, cuda, nk_bf16_to_f32_simt_, nk_f32_to_bf16_simt_)
nk_define_each_swiglu_cuda_(e4m3, cuda, nk_e4m3_to_f32_simt_, nk_f32_to_e4m3_simt_)
#endif // NUMKONG_TARGET_CUDA

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_
#endif // NUMKONG_EACH_CUDA_CUH
