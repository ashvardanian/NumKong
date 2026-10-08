/**
 *  @file include/numkong/reduce/cuda.cuh
 *  @author Ash Vardanian
 *  @date October 3, 2026
 *  @brief CUDA host side of moments and min/max reductions: the clears and launches, the export
 *      generators and the @c cuda exports, over the kernels of `reduce/simt.cuh`.
 *
 *  @sa include/numkong/reduce/simt.cuh
 */
#ifndef NUMKONG_REDUCE_CUDA_CUH
#define NUMKONG_REDUCE_CUDA_CUH

#if NUMKONG_ARCH_CUDA_

#include "numkong/cuda.cuh"
#include "numkong/reduce/simt.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

/** Sets @p bytes at device-reachable @p pointer to @p value, in order on @p stream. */
NUMKONG_INLINE nk_status_t nk_reduce_memset_cuda_(void *pointer, int value, nk_size_t bytes, nk_stream_t stream) {
    return cudaMemsetAsync(pointer, value, bytes, (cudaStream_t)stream) == cudaSuccess ? nk_success_k
                                                                                       : nk_device_code_mismatch_k;
}

/** Zeroes the @p sum_bytes and @p sumsq_bytes wide outputs, then launches @p kernel's resident
 *  grid, whose blocks add into them. */
NUMKONG_INLINE nk_status_t nk_reduce_moments_launch_cuda_(void const *kernel, nk_reduce_arguments_t arguments,
                                                          nk_size_t sum_bytes, nk_size_t sumsq_bytes,
                                                          nk_stream_t stream) {
    nk_status_t status = nk_reduce_memset_cuda_(arguments.first, 0, sum_bytes, stream);
    if (status == nk_success_k) status = nk_reduce_memset_cuda_(arguments.second, 0, sumsq_bytes, stream);
    if (status != nk_success_k || !arguments.count) return status;
    return nk_launch_resident_cuda_(
        kernel, nk_reduce_threads_simt_k, 0, 0,
        nk_size_divide_round_up_(arguments.count, nk_reduce_threads_simt_k * nk_reduce_batch_simt_k), &arguments,
        stream);
}

/** Clears both indices, launches @p kernel's resident grid, whose blocks swap their winners in, and
 *  then @p finish_kernel, which writes the winners' values. */
NUMKONG_INLINE nk_status_t nk_reduce_minmax_launch_cuda_(void const *kernel, void const *finish_kernel,
                                                         nk_reduce_arguments_t arguments, nk_stream_t stream) {
    nk_status_t status = nk_reduce_memset_cuda_(arguments.first_index, 0xFF, sizeof(nk_size_t), stream);
    if (status == nk_success_k)
        status = nk_reduce_memset_cuda_(arguments.second_index, 0xFF, sizeof(nk_size_t), stream);
    if (status == nk_success_k && arguments.count)
        status = nk_launch_resident_cuda_(
            kernel, nk_reduce_threads_simt_k, 0, 0,
            nk_size_divide_round_up_(arguments.count, nk_reduce_threads_simt_k * nk_reduce_batch_simt_k), &arguments,
            stream);
    void *launch_arguments[1];
    launch_arguments[0] = &arguments;
    return status == nk_success_k ? nk_launch_cuda_(finish_kernel, 1, 1, launch_arguments, 0, stream) : status;
}

/** Sums F64 values and their squares with Neumaier compensation, like the serial kernel. Misaligned
 *  values run in order on one thread, as the CPU tiers fall back to the serial loop for them, so a
 *  sum overflowing midway overflows at the same step. */
NUMKONG_DEVICE void nk_reduce_f64_moments_cuda_(nk_dtype_t dtype, nk_reduce_arguments_t const *arguments) {
    nk_size_t const lane = nk_reduce_lane_simt_();
    nk_size_t const first = arguments->aligned ? lane : lane ? arguments->count : 0;
    nk_size_t const step = arguments->aligned ? nk_reduce_lanes_simt_() : 1;
    nk_reduce_f64_moments_t state;
    state.sum = 0, state.sum_compensation = 0, state.sumsq = 0, state.sumsq_compensation = 0;
    for (nk_size_t batch = first; batch < arguments->count; batch += step * nk_reduce_batch_simt_k) {
        nk_u64_t raws[nk_reduce_batch_simt_k];
        nk_reduce_batch_simt_(dtype, arguments, batch, step, arguments->count, raws);
#pragma unroll
        for (unsigned slot = 0; slot != nk_reduce_batch_simt_k; ++slot) {
            if (batch + slot * step >= arguments->count) break;
            nk_f64_t const value = __longlong_as_double((long long)raws[slot]);
            nk_reduce_neumaier_f64_simt_(&state.sum, &state.sum_compensation, value);
            nk_reduce_neumaier_f64_simt_(&state.sumsq, &state.sumsq_compensation, nk_f64_mul_rn_cuda_(value, value));
        }
    }
    nk_reduce_f64_moments_merge_simt_(&state);
    if (threadIdx.x) return;
    atomicAdd((nk_f64_t *)arguments->first, state.sum + state.sum_compensation);
    atomicAdd((nk_f64_t *)arguments->second, state.sumsq + state.sumsq_compensation);
}

/** Sums the narrow float @p dtype values and their squares in F32 with Neumaier compensation, their
 *  squares exact in F32. */
NUMKONG_DEVICE void nk_reduce_f32_moments_cuda_(nk_dtype_t dtype, nk_reduce_arguments_t const *arguments) {
    __shared__ nk_reduce_f32_moments_t states[nk_reduce_threads_simt_k];
    nk_reduce_f32_moments_t state;
    state.sum = 0, state.sum_compensation = 0, state.sumsq = 0, state.sumsq_compensation = 0;
    for (nk_size_t first = nk_reduce_lane_simt_(); first < arguments->count;
         first += nk_reduce_lanes_simt_() * nk_reduce_batch_simt_k) {
        nk_u64_t raws[nk_reduce_batch_simt_k];
        nk_reduce_batch_simt_(dtype, arguments, first, nk_reduce_lanes_simt_(), arguments->count, raws);
#pragma unroll
        for (unsigned slot = 0; slot != nk_reduce_batch_simt_k; ++slot) {
            if (first + slot * nk_reduce_lanes_simt_() >= arguments->count) break;
            nk_f32_t const value = nk_reduce_f32_simt_(dtype, raws[slot]);
            nk_f32_two_sum_simt_(value, &state.sum, &state.sum_compensation);
            nk_f32_two_sum_simt_(nk_f32_mul_rn_cuda_(value, value), &state.sumsq, &state.sumsq_compensation);
        }
    }
    states[threadIdx.x] = state;
    __syncthreads();
    for (unsigned half = blockDim.x / 2; half; half >>= 1) {
        if (threadIdx.x < half) {
            nk_reduce_f32_moments_t *into = &states[threadIdx.x];
            nk_reduce_f32_moments_t const *other = &states[threadIdx.x + half];
            nk_f32_two_sum_simt_(other->sum, &into->sum, &into->sum_compensation);
            nk_f32_two_sum_simt_(other->sumsq, &into->sumsq, &into->sumsq_compensation);
            into->sum_compensation += other->sum_compensation, into->sumsq_compensation += other->sumsq_compensation;
        }
        __syncthreads();
    }
    if (threadIdx.x) return;
    atomicAdd((nk_f32_t *)arguments->first, states[0].sum + states[0].sum_compensation);
    atomicAdd((nk_f32_t *)arguments->second, states[0].sumsq + states[0].sumsq_compensation);
}

/** Sums F32 or BF16 values and their squares into F64 totals, mostly without F64 arithmetic: values
 *  F32 can square exactly sum as unevaluated F32 pairs, their squares split exactly by an FMA, and
 *  the rest take a second walk of the thread's values in F64. */
NUMKONG_DEVICE void nk_reduce_pairs_moments_cuda_(nk_dtype_t dtype, nk_reduce_arguments_t const *arguments) {
    nk_f32_t sum_high = 0, sum_low = 0, sumsq_high = 0, sumsq_low = 0;
    nk_reduce_f64_moments_t state;
    int rest = 0;
    state.sum = 0, state.sum_compensation = 0, state.sumsq = 0, state.sumsq_compensation = 0;
    for (nk_size_t first = nk_reduce_lane_simt_(); first < arguments->count;
         first += nk_reduce_lanes_simt_() * nk_reduce_batch_simt_k) {
        nk_u64_t raws[nk_reduce_batch_simt_k];
        nk_reduce_batch_simt_(dtype, arguments, first, nk_reduce_lanes_simt_(), arguments->count, raws);
#pragma unroll
        for (unsigned slot = 0; slot != nk_reduce_batch_simt_k; ++slot) {
            if (first + slot * nk_reduce_lanes_simt_() >= arguments->count) break;
            nk_f32_t const value = nk_reduce_f32_simt_(dtype, raws[slot]);
            nk_f32_t const square = nk_f32_mul_rn_cuda_(value, value);
            if (!nk_reduce_pairs_hold_simt_(value)) {
                rest = 1;
                continue;
            }
            nk_reduce_two_sum_f32_simt_(&sum_high, &sum_low, value);
            nk_reduce_two_sum_f32_simt_(&sumsq_high, &sumsq_low, square);
            sumsq_low += __fmaf_rn(value, value, -square);
        }
    }
    if (rest)
        for (nk_size_t index = nk_reduce_lane_simt_(); index < arguments->count; index += nk_reduce_lanes_simt_()) {
            nk_f64_t const value = nk_reduce_f32_simt_(dtype, nk_reduce_raw_simt_(dtype, arguments, index));
            if (nk_reduce_pairs_hold_simt_((nk_f32_t)value)) continue;
            state.sum += value, state.sumsq += nk_f64_mul_rn_cuda_(value, value);
        }
    nk_reduce_neumaier_f64_simt_(&state.sum, &state.sum_compensation, sum_high);
    nk_reduce_neumaier_f64_simt_(&state.sum, &state.sum_compensation, sum_low);
    nk_reduce_neumaier_f64_simt_(&state.sumsq, &state.sumsq_compensation, sumsq_high);
    nk_reduce_neumaier_f64_simt_(&state.sumsq, &state.sumsq_compensation, sumsq_low);
    nk_reduce_f64_moments_merge_simt_(&state);
    if (threadIdx.x) return;
    nk_f64_t const sum = state.sum + state.sum_compensation, sumsq = state.sumsq + state.sumsq_compensation;
    if (dtype == nk_f32_k)
        atomicAdd((nk_f64_t *)arguments->first, sum), atomicAdd((nk_f64_t *)arguments->second, sumsq);
    else
        atomicAdd((nk_f32_t *)arguments->first, (nk_f32_t)sum),
            atomicAdd((nk_f32_t *)arguments->second, (nk_f32_t)sumsq);
}

/** Generates the moments kernel of @p input_type for @p isa_suffix, summing through @p moments_fn,
 *  and its entry point. */
#define nk_define_reduce_moments_cuda_(input_type, input_value_type, moments_fn, sum_type, sumsq_type, isa_suffix) \
    static __global__ void __launch_bounds__(nk_reduce_threads_simt_k)                                             \
        nk_reduce_moments_##input_type##_##isa_suffix##_kernel_(nk_reduce_arguments_t arguments) {                 \
        moments_fn(nk_##input_type##_k, &arguments);                                                               \
    }                                                                                                              \
    NUMKONG_API nk_status_t nk_reduce_moments_##input_type##_##isa_suffix(                                         \
        nk_##input_value_type##_t const *data, nk_size_t count, nk_size_t stride, nk_##sum_type##_t *sum,          \
        nk_##sumsq_type##_t *sumsq, nk_stream_t stream) {                                                          \
        return nk_reduce_moments_launch_cuda_(                                                                     \
            (void const *)&nk_reduce_moments_##input_type##_##isa_suffix##_kernel_,                                \
            nk_reduce_arguments_simt_(sizeof(nk_##input_value_type##_t), data, count, stride, sum, NUMKONG_NULL,   \
                                      sumsq, NUMKONG_NULL),                                                        \
            sizeof(nk_##sum_type##_t), sizeof(nk_##sumsq_type##_t), stream);                                       \
    }

/** Generates the entry point and the min/max and finishing kernels of @p input_type for
 *  @p isa_suffix, their sides starting from the serial kernels' @p min_sentinel and
 *  @p max_sentinel bits. */
#define nk_define_reduce_minmax_cuda_(input_type, input_value_type, output_type, min_sentinel, max_sentinel,        \
                                      isa_suffix)                                                                   \
    static __global__ void __launch_bounds__(nk_reduce_threads_simt_k)                                              \
        nk_reduce_minmax_##input_type##_##isa_suffix##_kernel_(nk_reduce_arguments_t arguments) {                   \
        nk_reduce_minmax_simt_(nk_##input_type##_k, &arguments);                                                    \
    }                                                                                                               \
    static __global__ void nk_reduce_minmax_##input_type##_finish_##isa_suffix##_kernel_(                           \
        nk_reduce_arguments_t arguments) {                                                                          \
        nk_reduce_minmax_finish_simt_(nk_##input_type##_k, min_sentinel, max_sentinel, &arguments);                 \
    }                                                                                                               \
    NUMKONG_API nk_status_t nk_reduce_minmax_##input_type##_##isa_suffix(                                           \
        nk_##input_value_type##_t const *data, nk_size_t count, nk_size_t stride, nk_##output_type##_t *min_value,  \
        nk_size_t *min_index, nk_##output_type##_t *max_value, nk_size_t *max_index, nk_stream_t stream) {          \
        return nk_reduce_minmax_launch_cuda_(                                                                       \
            (void const *)&nk_reduce_minmax_##input_type##_##isa_suffix##_kernel_,                                  \
            (void const *)&nk_reduce_minmax_##input_type##_finish_##isa_suffix##_kernel_,                           \
            nk_reduce_arguments_simt_(sizeof(nk_##input_value_type##_t), data, count, stride, min_value, min_index, \
                                      max_value, max_index),                                                        \
            stream);                                                                                                \
    }

#if NUMKONG_TARGET_CUDA
nk_define_reduce_moments_cuda_(f64, f64, nk_reduce_f64_moments_cuda_, f64, f64, cuda)
nk_define_reduce_moments_cuda_(f32, f32, nk_reduce_pairs_moments_cuda_, f64, f64, cuda)
nk_define_reduce_moments_cuda_(f16, f16, nk_reduce_f32_moments_cuda_, f32, f32, cuda)
nk_define_reduce_moments_cuda_(bf16, bf16, nk_reduce_pairs_moments_cuda_, f32, f32, cuda)
nk_define_reduce_moments_cuda_(e4m3, e4m3, nk_reduce_f32_moments_cuda_, f32, f32, cuda)
nk_define_reduce_moments_cuda_(e5m2, e5m2, nk_reduce_f32_moments_cuda_, f32, f32, cuda)
nk_define_reduce_moments_cuda_(e2m3, e2m3, nk_reduce_f32_moments_cuda_, f32, f32, cuda)
nk_define_reduce_moments_cuda_(e3m2, e3m2, nk_reduce_f32_moments_cuda_, f32, f32, cuda)
nk_define_reduce_moments_cuda_(e2m1, e2m1x2, nk_reduce_integer_moments_simt_, f32, f32, cuda)
nk_define_reduce_moments_cuda_(i8, i8, nk_reduce_integer_moments_simt_, i64, u64, cuda)
nk_define_reduce_moments_cuda_(u8, u8, nk_reduce_integer_moments_simt_, u64, u64, cuda)
nk_define_reduce_moments_cuda_(i16, i16, nk_reduce_integer_moments_simt_, i64, u64, cuda)
nk_define_reduce_moments_cuda_(u16, u16, nk_reduce_integer_moments_simt_, u64, u64, cuda)
nk_define_reduce_moments_cuda_(i32, i32, nk_reduce_integer_moments_simt_, i64, u64, cuda)
nk_define_reduce_moments_cuda_(u32, u32, nk_reduce_integer_moments_simt_, u64, u64, cuda)
nk_define_reduce_moments_cuda_(i64, i64, nk_reduce_integer_moments_simt_, i64, u64, cuda)
nk_define_reduce_moments_cuda_(u64, u64, nk_reduce_integer_moments_simt_, u64, u64, cuda)
nk_define_reduce_moments_cuda_(i4, i4x2, nk_reduce_integer_moments_simt_, i64, u64, cuda)
nk_define_reduce_moments_cuda_(u4, u4x2, nk_reduce_integer_moments_simt_, u64, u64, cuda)
nk_define_reduce_moments_cuda_(u1, u1x8, nk_reduce_integer_moments_simt_, u64, u64, cuda)
nk_define_reduce_minmax_cuda_(f64, f64, f64, 0x7FF0000000000000ull, 0xFFF0000000000000ull, cuda)
nk_define_reduce_minmax_cuda_(f32, f32, f32, 0x7F800000u, 0xFF800000u, cuda)
nk_define_reduce_minmax_cuda_(f16, f16, f16, 0x7BFFu, 0xFBFFu, cuda)
nk_define_reduce_minmax_cuda_(bf16, bf16, bf16, 0x7F7Fu, 0xFF7Fu, cuda)
nk_define_reduce_minmax_cuda_(e4m3, e4m3, e4m3, NUMKONG_E4M3_MAX, NUMKONG_E4M3_MIN, cuda)
nk_define_reduce_minmax_cuda_(e5m2, e5m2, e5m2, NUMKONG_E5M2_MAX, NUMKONG_E5M2_MIN, cuda)
nk_define_reduce_minmax_cuda_(e2m3, e2m3, e2m3, NUMKONG_E2M3_MAX, NUMKONG_E2M3_MIN, cuda)
nk_define_reduce_minmax_cuda_(e3m2, e3m2, e3m2, NUMKONG_E3M2_MAX, NUMKONG_E3M2_MIN, cuda)
nk_define_reduce_minmax_cuda_(i8, i8, i8, 0x7Fu, 0x80u, cuda)
nk_define_reduce_minmax_cuda_(u8, u8, u8, 0xFFu, 0u, cuda)
nk_define_reduce_minmax_cuda_(i16, i16, i16, 0x7FFFu, 0x8000u, cuda)
nk_define_reduce_minmax_cuda_(u16, u16, u16, 0xFFFFu, 0u, cuda)
nk_define_reduce_minmax_cuda_(i32, i32, i32, 0x7FFFFFFFu, 0x80000000u, cuda)
nk_define_reduce_minmax_cuda_(u32, u32, u32, 0xFFFFFFFFu, 0u, cuda)
nk_define_reduce_minmax_cuda_(i64, i64, i64, 0x7FFFFFFFFFFFFFFFull, 0x8000000000000000ull, cuda)
nk_define_reduce_minmax_cuda_(u64, u64, u64, NUMKONG_U64_MAX, 0u, cuda)
nk_define_reduce_minmax_cuda_(i4, i4x2, i8, 0x07u, 0xF8u, cuda)
nk_define_reduce_minmax_cuda_(u4, u4x2, u8, 0x0Fu, 0u, cuda)
nk_define_reduce_minmax_cuda_(u1, u1x8, u8, 1u, 0u, cuda)
#endif // NUMKONG_TARGET_CUDA

#undef nk_define_reduce_moments_cuda_
#undef nk_define_reduce_minmax_cuda_

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_
#endif // NUMKONG_REDUCE_CUDA_CUH
