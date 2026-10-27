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
 *  RMSNorm, and its downcasting form, give each normalized vector one block, sized so that every
 *  thread keeps at most 8 16-byte pieces of it in registers between the two passes, reloading any
 *  beyond. A chunk there is 16 bytes of the narrower side, so every store moves 16 bytes. Squares
 *  sum plainly one tier above the narrower of the input and output: in F64 for the RMSNorm of F32
 *  and the downcast of F64 to F32, and in F32 for the rest, downcasts from F32 and 32-bit integers
 *  included. The sums merge through xor shuffles across a team's lanes and through shared memory
 *  across teams, and their inverse RMS comes out in that same tier, while every value scales in
 *  F32, but F64 ones in F64.
 *
 *  The element-wise kernels live here, and each vendor launches and exports them from its own host
 *  side, in `cuda.cuh` and `rocm.cuh` beside this file, whose rounded products and shuffles the
 *  RMSNorm kernels call.
 *
 *  @sa include/numkong/each/cuda.cuh
 *  @sa include/numkong/each/rocm.cuh
 */
#ifndef NUMKONG_EACH_SIMT_CUH
#define NUMKONG_EACH_SIMT_CUH

#if NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_

#include "numkong/cast/simt.cuh" // `nk_bf16_to_f32_simt_`
#include "numkong/each/serial.h" // `nk_each_rmsnorm_arguments_t`

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
    static __global__ void __launch_bounds__(nk_each_threads_simt_k, 1)                                           \
        nk_each_##verb##_##input_type##_##isa_suffix##_kernel_(nk_each_elementwise_arguments_t arguments) {       \
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
        uint4 const *a_chunks = (uint4 const *)a;                                                                 \
        uint4 const *b_chunks = (uint4 const *)b;                                                                 \
        uint4 const *c_chunks = (uint4 const *)c;                                                                 \
        uint4 *result_chunks = (uint4 *)result;                                                                   \
        _Pragma("unroll 1") for (nk_size_t chunk = first; chunk < chunks; chunk += threads) {                     \
            inputs[0] = a_chunks[chunk];                                                                          \
            inputs[1] = operands > 1 ? b_chunks[chunk] : inputs[0];                                               \
            inputs[2] = operands > 2 ? c_chunks[chunk] : inputs[0];                                               \
            _Pragma("unroll") for (unsigned offset = 0; offset != chunk_values; ++offset)                         \
                nk_each_##verb##_##input_type##_##isa_suffix##_value_(                                            \
                    input_vectors[0].input_type##s + offset, input_vectors[1].input_type##s + offset,             \
                    input_vectors[2].input_type##s + offset, alpha, beta, output_vector->input_type##s + offset); \
            result_chunks[chunk] = output;                                                                        \
        }                                                                                                         \
        for (nk_size_t index = chunks * chunk_values + first; index < arguments.count; index += threads)          \
            nk_each_##verb##_##input_type##_##isa_suffix##_value_(a + index, b + index, c + index, alpha, beta,   \
                                                                  result + index);                                \
    }

/** The unit @p scale and the F32 inverse RMS @p factor of @p count values from the merged F32 sum
 *  of their squares, all in F32, the inverse from the vendor's @c rsqrtf. */
NUMKONG_DEVICE void nk_each_inverse_rms_f32_simt_(nk_f32_t sum, nk_size_t count, nk_f32_t epsilon, nk_f32_t *scale,
                                                  nk_f32_t *factor) {
    *scale = 1.0f, *factor = rsqrtf(sum / (nk_f32_t)count + epsilon);
}

/** The unit @p scale and the F32 inverse RMS @p factor of @p count 32-bit integers from the merged
 *  F32 sum of their squares, whose mean rounds once from F64 to F32 before adding @p epsilon. */
NUMKONG_DEVICE void nk_each_inverse_rms_integer_simt_(nk_f32_t sum, nk_size_t count, nk_f32_t epsilon, nk_f32_t *scale,
                                                      nk_f32_t *factor) {
    nk_f32_t const mean_square = (nk_f32_t)((nk_f64_t)sum / (nk_f64_t)count);
    *scale = 1.0f, *factor = 1.0f / sqrtf(mean_square + epsilon);
}

/** The unit @p scale and the F64 inverse RMS @p factor of @p count values from the merged F64 sum
 *  of their squares, all in F64. */
NUMKONG_DEVICE void nk_each_inverse_rms_f64_simt_(nk_f64_t sum, nk_size_t count, nk_f32_t epsilon, nk_f64_t *scale,
                                                  nk_f64_t *factor) {
    *scale = 1.0, *factor = 1.0 / sqrt(sum / (nk_f64_t)count + (nk_f64_t)epsilon);
}

/** The F64 inverse RMS of @p count F32 values, from the merged F64 sum of their squares, as a power
 *  of two @p scale and the F32 @p factor rounded from the rest within 2⁶⁴ of one, which F32 values
 *  multiply by without leaving the F32 range, like @c nk_each_prescale_f32_serial_. */
NUMKONG_DEVICE void nk_each_inverse_rms_f64_to_f32_simt_(nk_f64_t sum, nk_size_t count, nk_f32_t epsilon,
                                                         nk_f32_t *scale, nk_f32_t *factor) {
    nk_f64_t unit, inverse_rms;
    nk_each_inverse_rms_f64_simt_(sum, count, epsilon, &unit, &inverse_rms);
    nk_u64_t const bits = (nk_u64_t)__double_as_longlong(inverse_rms);
    int const exponent = (int)((bits >> 52) & 0x7FFu) - 1023;
    int const shift = exponent > 64 ? exponent - 64 : exponent < -64 ? exponent + 64 : 0;
    nk_f64_t const power = __longlong_as_double((long long)((nk_u64_t)(1023 + shift) << 52));
    nk_f64_t const remainder = __longlong_as_double((long long)((nk_u64_t)(1023 - shift) << 52));
    *scale = (nk_f32_t)power, *factor = (nk_f32_t)(inverse_rms * remainder);
}

/** Generates @p verb for @p isa_suffix, an RMSNorm of @p input_type vectors into @p output_type
 *  ones after @c nk_define_each_rmsnorm_ of the serial backend: the helper that normalizes one
 *  chunk out of its 16-byte @c pieces, the device body that normalizes every vector of a launch,
 *  one block per vector, the kernel over it, and the entry point.
 *  Squares sum in @p reduction_type, rounded and shuffled through the primitives of @p vendor,
 *  @c cuda or @c rocm, the @p inverse_rms helper turns them into a scale and a factor of
 *  @p element_type, and every value scales in that @p element_type. F64 vectors whose sum of
 *  squares leaves @c [2⁻⁹⁶⁸,2⁹⁶⁸] sum again scaled by the power of two of their largest
 *  magnitude, as @c nk_each_inverse_rms_f64_serial_ does. */
#define nk_define_each_rmsnorm_simt_(verb, input_type, output_type, reduction_type, element_type, inverse_rms,         \
                                     isa_suffix, vendor, load_and_convert, convert_and_store)                          \
    NUMKONG_DEVICE void nk_each_##verb##_##output_type##_##isa_suffix##_chunk_(                                        \
        nk_b128_vec_t const *pieces, nk_f32_t const *gamma, nk_##element_type##_t scale, nk_##element_type##_t factor, \
        nk_##output_type##_t *y) {                                                                                     \
        unsigned const piece_values = 16 / sizeof(nk_##input_type##_t);                                                \
        unsigned const chunk_values = 16 / sizeof(nk_##output_type##_t) < piece_values                                 \
                                          ? piece_values                                                               \
                                          : 16 / sizeof(nk_##output_type##_t);                                         \
        uint4 gains[4], normalized;                                                                                    \
        nk_b128_vec_t const *const gain_vectors = (nk_b128_vec_t const *)gains;                                        \
        nk_##output_type##_t *const outputs = ((nk_b128_vec_t *)&normalized)->output_type##s;                          \
        nk_##element_type##_t value, result;                                                                           \
        uint4 const *gamma_chunks = (uint4 const *)gamma;                                                              \
        if (gamma) {                                                                                                   \
            _Pragma("unroll") for (unsigned quad = 0; quad != chunk_values / 4; ++quad)                                \
                gains[quad] = gamma_chunks[quad];                                                                      \
        }                                                                                                              \
        _Pragma("unroll") for (unsigned offset = 0; offset != chunk_values; ++offset) {                                \
            load_and_convert(pieces[offset / piece_values].input_type##s + offset % piece_values, &value);             \
            result = value * scale * factor * (gamma ? gain_vectors[offset / 4].f32s[offset % 4] : 1.0f);              \
            convert_and_store(&result, outputs + offset);                                                              \
        }                                                                                                              \
        *(uint4 *)y = normalized;                                                                                      \
    }                                                                                                                  \
    NUMKONG_DEVICE void nk_each_##verb##_##output_type##_##isa_suffix##_vectors_(                                      \
        nk_each_rmsnorm_arguments_t arguments) {                                                                       \
        __shared__ nk_##reduction_type##_t partials[nk_each_threads_simt_k / nk_each_team_lanes_simt_k];               \
        unsigned const piece_values = 16 / sizeof(nk_##input_type##_t);                                                \
        unsigned const chunk_values = 16 / sizeof(nk_##output_type##_t) < piece_values                                 \
                                          ? piece_values                                                               \
                                          : 16 / sizeof(nk_##output_type##_t);                                         \
        unsigned const chunk_pieces = chunk_values / piece_values;                                                     \
        unsigned const kept_chunks = nk_each_kept_pieces_simt_k / chunk_pieces;                                        \
        unsigned const teams = blockDim.x / nk_each_team_lanes_simt_k;                                                 \
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
            nk_##reduction_type##_t sum = 0;                                                                           \
            nk_##element_type##_t value;                                                                               \
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
                    sum += nk_##reduction_type##_mul_rn_##vendor##_(value, value);                                     \
                }                                                                                                      \
            }                                                                                                          \
            for (nk_size_t chunk = threadIdx.x + kept_chunks * blockDim.x; chunk < chunks; chunk += blockDim.x)        \
                _Pragma("unroll") for (unsigned piece = 0; piece != chunk_pieces; ++piece) {                           \
                    uint4 const bits = *(uint4 const *)(x + chunk * chunk_values + piece * piece_values);              \
                    nk_b128_vec_t const vector = {.u32s = {bits.x, bits.y, bits.z, bits.w}};                           \
                    loaded[0] = vector;                                                                                \
                    _Pragma("unroll") for (unsigned offset = 0; offset != piece_values; ++offset) {                    \
                        load_and_convert(loaded[0].input_type##s + offset, &value);                                    \
                        sum += nk_##reduction_type##_mul_rn_##vendor##_(value, value);                                 \
                    }                                                                                                  \
                }                                                                                                      \
            for (nk_size_t column = chunks * chunk_values + threadIdx.x; column < arguments.columns;                   \
                 column += blockDim.x) {                                                                               \
                load_and_convert(x + column, &value);                                                                  \
                sum += nk_##reduction_type##_mul_rn_##vendor##_(value, value);                                         \
            }                                                                                                          \
            for (unsigned offset = nk_each_team_lanes_simt_k / 2; offset; offset >>= 1)                                \
                sum += nk_shuffle_xor_##reduction_type##_##vendor##_(sum, offset);                                     \
            if (teams > 1) {                                                                                           \
                if (threadIdx.x % nk_each_team_lanes_simt_k == 0)                                                      \
                    partials[threadIdx.x / nk_each_team_lanes_simt_k] = sum;                                           \
                __syncthreads();                                                                                       \
                sum = partials[0];                                                                                     \
                for (unsigned team = 1; team < teams; ++team) sum += partials[team];                                   \
                __syncthreads();                                                                                       \
            }                                                                                                          \
            nk_##element_type##_t scale, factor;                                                                       \
            inverse_rms(sum, arguments.columns, arguments.epsilon, &scale, &factor);                                   \
            if (sizeof(nk_##input_type##_t) == 8 && sum == sum &&                                                      \
                !(sum >= (nk_##reduction_type##_t)0x1p-968 && sum <= (nk_##reduction_type##_t)0x1p968)) {              \
                nk_##reduction_type##_t largest = 0, scaled_sum = 0;                                                   \
                for (nk_size_t column = threadIdx.x; column < arguments.columns; column += blockDim.x) {               \
                    load_and_convert(x + column, &value);                                                              \
                    largest = fmax(largest, fabs((nk_##reduction_type##_t)value));                                     \
                }                                                                                                      \
                for (unsigned offset = nk_each_team_lanes_simt_k / 2; offset; offset >>= 1)                            \
                    largest = fmax(largest, nk_shuffle_xor_##reduction_type##_##vendor##_(largest, offset));           \
                if (teams > 1) {                                                                                       \
                    if (threadIdx.x % nk_each_team_lanes_simt_k == 0)                                                  \
                        partials[threadIdx.x / nk_each_team_lanes_simt_k] = largest;                                   \
                    __syncthreads();                                                                                   \
                    largest = partials[0];                                                                             \
                    for (unsigned team = 1; team < teams; ++team) largest = fmax(largest, partials[team]);             \
                    __syncthreads();                                                                                   \
                }                                                                                                      \
                if (largest != 0 && largest != (nk_##reduction_type##_t)NUMKONG_F64_INF) {                             \
                    int exponent = (int)(((nk_u64_t)__double_as_longlong(largest) >> 52) & 0x7FFu) - 1023;             \
                    exponent = exponent < -1022 ? -1022 : exponent > 1022 ? 1022 : exponent;                           \
                    nk_##reduction_type##_t const power = __longlong_as_double(                                        \
                        (long long)((nk_u64_t)(1023 - exponent) << 52));                                               \
                    for (nk_size_t column = threadIdx.x; column < arguments.columns; column += blockDim.x) {           \
                        load_and_convert(x + column, &value);                                                          \
                        nk_##reduction_type##_t const scaled = (nk_##reduction_type##_t)value * power;                 \
                        scaled_sum += nk_##reduction_type##_mul_rn_##vendor##_(scaled, scaled);                        \
                    }                                                                                                  \
                    for (unsigned offset = nk_each_team_lanes_simt_k / 2; offset; offset >>= 1)                        \
                        scaled_sum += nk_shuffle_xor_##reduction_type##_##vendor##_(scaled_sum, offset);               \
                    if (teams > 1) {                                                                                   \
                        if (threadIdx.x % nk_each_team_lanes_simt_k == 0)                                              \
                            partials[threadIdx.x / nk_each_team_lanes_simt_k] = scaled_sum;                            \
                        __syncthreads();                                                                               \
                        scaled_sum = partials[0];                                                                      \
                        for (unsigned team = 1; team < teams; ++team) scaled_sum += partials[team];                    \
                        __syncthreads();                                                                               \
                    }                                                                                                  \
                    nk_##reduction_type##_t const scaled_epsilon = (nk_##reduction_type##_t)arguments.epsilon *        \
                                                                   power * power;                                      \
                    scale = 1;                                                                                         \
                    factor =                                                                                           \
                        scaled_epsilon == (nk_##reduction_type##_t)NUMKONG_F64_INF                                     \
                            ? 1 / sqrt((nk_##reduction_type##_t)arguments.epsilon)                                     \
                            : 1 / sqrt(scaled_sum / (nk_##reduction_type##_t)arguments.columns + scaled_epsilon) *     \
                                  power;                                                                               \
                }                                                                                                      \
            }                                                                                                          \
            _Pragma("unroll") for (unsigned slot = 0; slot != kept_chunks; ++slot) {                                   \
                nk_size_t const chunk = threadIdx.x + slot * blockDim.x;                                               \
                if (chunk >= chunks) break;                                                                            \
                nk_each_##verb##_##output_type##_##isa_suffix##_chunk_(                                                \
                    kept + slot * chunk_pieces, gamma ? gamma + chunk * chunk_values : NUMKONG_NULL, scale, factor,    \
                    y + chunk * chunk_values);                                                                         \
            }                                                                                                          \
            for (nk_size_t chunk = threadIdx.x + kept_chunks * blockDim.x; chunk < chunks; chunk += blockDim.x) {      \
                _Pragma("unroll") for (unsigned piece = 0; piece != chunk_pieces; ++piece) {                           \
                    uint4 const bits = *(uint4 const *)(x + chunk * chunk_values + piece * piece_values);              \
                    nk_b128_vec_t const vector = {.u32s = {bits.x, bits.y, bits.z, bits.w}};                           \
                    loaded[piece] = vector;                                                                            \
                }                                                                                                      \
                nk_each_##verb##_##output_type##_##isa_suffix##_chunk_(                                                \
                    loaded, gamma ? gamma + chunk * chunk_values : NUMKONG_NULL, scale, factor,                        \
                    y + chunk * chunk_values);                                                                         \
            }                                                                                                          \
            for (nk_size_t column = chunks * chunk_values + threadIdx.x; column < arguments.columns;                   \
                 column += blockDim.x) {                                                                               \
                load_and_convert(x + column, &value);                                                                  \
                nk_##element_type##_t const result = value * scale * factor * (gamma ? gamma[column] : 1.0f);          \
                convert_and_store(&result, y + column);                                                                \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    static __global__ void __launch_bounds__(nk_each_threads_simt_k, 1)                                                \
        nk_each_##verb##_##output_type##_##isa_suffix##_kernel_(nk_each_rmsnorm_arguments_t arguments) {               \
        nk_each_##verb##_##output_type##_##isa_suffix##_vectors_(arguments);                                           \
    }                                                                                                                  \
    NUMKONG_API nk_status_t nk_each_##verb##_##output_type##_##isa_suffix(                                             \
        nk_##input_type##_t const *x, nk_f32_t const *gamma, nk_##output_type##_t *y, nk_size_t rows,                  \
        nk_size_t groups, nk_size_t columns, nk_size_t x_stride, nk_size_t y_stride, nk_f32_t epsilon,                 \
        nk_stream_t stream) {                                                                                          \
        return nk_each_rmsnorm_launch_##vendor##_(                                                                     \
            (void const *)&nk_each_##verb##_##output_type##_##isa_suffix##_kernel_, sizeof(nk_##input_type##_t),       \
            sizeof(nk_##output_type##_t), x, gamma, y, rows, groups, columns, x_stride, y_stride, epsilon, stream);    \
    }

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_
#endif // NUMKONG_EACH_SIMT_CUH
