/**
 *  @file include/numkong/each/metal.metal
 *  @author Ash Vardanian
 *  @date October 5, 2026
 *  @brief SwiGLU, grouped RMSNorm and its downcasting form on the SIMT cores of every Apple GPU.
 *
 *  @sa include/numkong/each/metal.h, which embeds and launches this source
 *  @sa include/numkong/each/simt.cuh, the CUDA and ROCm sibling
 *
 *  SwiGLU runs a grid of 16-byte chunks by rows, each thread one chunk of one row. RMSNorm gives
 *  each normalized vector one threadgroup, in a grid of groups by rows, whose threads take 16-byte
 *  chunks of the output a threadgroup apart, reading the input twice rather than keeping it.
 *  Squares sum as compensated F32 pairs, TwoProduct through FMA and TwoSum, merged down each
 *  SIMD-group through shuffles and then across SIMD-groups. The host picks the kernel for a
 *  missing @c up or @c gamma, so no thread branches on either.
 */

/** The launch record of SwiGLU, laid out as the C launcher's record of the same name. */
struct nk_each_swiglu_arguments_metal_t {
    ulong gate_stride, up_stride, y_stride;
    uint columns;
    float gate_scale, output_scale;
};

/** The launch record of RMSNorm, laid out as the C launcher's record of the same name. */
struct nk_each_rmsnorm_arguments_metal_t {
    ulong x_stride, y_stride;
    uint columns;
    float epsilon;
};

static_assert(sizeof(nk_each_swiglu_arguments_metal_t) == 40, "mirrors the C record in each/metal.h");
static_assert(sizeof(nk_each_rmsnorm_arguments_metal_t) == 24, "mirrors the C record in each/metal.h");

/** SwiGLU multiplying SiLU of the gate by the up row. */
struct nk_each_gated_metal_t {};

/** SwiGLU without an up row, which is plain SiLU. */
struct nk_each_ungated_metal_t {};

/** RMSNorm multiplying every column by its gain. */
struct nk_each_scaled_metal_t {};

/** RMSNorm without gains. */
struct nk_each_unscaled_metal_t {};

/** The up value multiplying @p column of a gated row, or one without an up row. */
template <typename dtype_>
inline float nk_each_up_metal_(nk_each_gated_metal_t, device uchar const *up, uint column) {
    return dtype_::load(up, column);
}

template <typename dtype_>
inline float nk_each_up_metal_(nk_each_ungated_metal_t, device uchar const *, uint) {
    return 1.0f;
}

/** The gain multiplying @p column of a scaled vector, or one without gains. */
inline float nk_each_gain_metal_(nk_each_scaled_metal_t, device float const *gamma, uint column) {
    return gamma[column];
}

inline float nk_each_gain_metal_(nk_each_unscaled_metal_t, device float const *, uint) { return 1.0f; }

/** SiLU of @p value, x / (1 + e^-x), keeping the subnormal precision of tiny reciprocals. */
inline float nk_each_silu_metal_(float value) {
#pragma clang fp contract(off) reassociate(off)
    float const denominator = 1.0f + nk_exp2_metal_(-value * 1.4426950408889634074f);
    // Scale subnormal reciprocals into integers before Metal flushes them
    return denominator > 0x1p126f
               ? ldexp(value * (rint(precise::divide(0x1p126f, denominator) * 0x1p23f) * 0x1p-23f), -126)
               : value * precise::divide(1.0f, denominator);
}

/** SwiGLU of one 16-byte chunk of one row, @p position holding the chunk and the row. */
template <typename dtype_, typename gating_>
void nk_each_swiglu_metal_(device uchar const *gate, device uchar const *up, device uchar *y,
                           constant nk_each_swiglu_arguments_metal_t &arguments, uint2 position) {
#pragma clang fp contract(off) reassociate(off)
    uint const columns = arguments.columns;
    if (position.x >= nk::divide_round_up(columns, dtype_::chunk_values)) return;
    uint const first = position.x * dtype_::chunk_values, end = first + min(columns - first, dtype_::chunk_values);
    gate += position.y * arguments.gate_stride, up += position.y * arguments.up_stride;
    y += position.y * arguments.y_stride;
    for (uint column = first; column != end; ++column) {
        float const silu = nk_each_silu_metal_(dtype_::load(gate, column) * arguments.gate_scale);
        dtype_::store(y, column, silu * nk_each_up_metal_<dtype_>(gating_(), up, column) * arguments.output_scale);
    }
}

/** Adds @p addend to the compensated pair of @p sum and @p compensation by TwoSum. */
inline void nk_each_two_sum_metal_(float addend, thread float &sum, thread float &compensation) {
#pragma clang fp contract(off) reassociate(off)
    float const total = sum + addend, virtual_addend = total - sum;
    compensation += (sum - (total - virtual_addend)) + (addend - virtual_addend);
    sum = total;
}

/** The inverse RMS of @p count values from each thread's compensated sum of their squares, the
 *  same in every thread: down each SIMD-group through shuffles, then across the SIMD-groups of
 *  @p threads through @p partials, eight sums followed by eight compensations. */
inline float nk_each_inverse_rms_metal_(float sum, float compensation, uint count, float epsilon, uint lane,
                                        uint threads, threadgroup float *partials) {
#pragma clang fp contract(off) reassociate(off)
    uint const simd_lane = lane % 32, simd_group = lane / 32, simd_groups = threads / 32;
    for (uint offset = 16; offset; offset >>= 1) {
        float const other_sum = simd_shuffle_down(sum, offset);
        float const other_compensation = simd_shuffle_down(compensation, offset);
        if (simd_lane < offset) {
            compensation += other_compensation;
            nk_each_two_sum_metal_(other_sum, sum, compensation);
        }
    }
    if (!simd_lane) partials[simd_group] = sum, partials[8 + simd_group] = compensation;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!simd_group) {
        sum = simd_lane < simd_groups ? partials[simd_lane] : 0;
        compensation = simd_lane < simd_groups ? partials[8 + simd_lane] : 0;
        for (uint offset = 4; offset; offset >>= 1) {
            float const other_sum = simd_shuffle_down(sum, offset);
            float const other_compensation = simd_shuffle_down(compensation, offset);
            if (simd_lane < offset) {
                compensation += other_compensation;
                nk_each_two_sum_metal_(other_sum, sum, compensation);
            }
        }
        if (!simd_lane) partials[0] = precise::divide(sum + compensation, float(count));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return precise::divide(1.0f, precise::sqrt(partials[0] + epsilon));
}

/** RMSNorm of one vector of @p input_ values into @p output_ ones, @p vector holding its group and
 *  its row, each of the @p threads taking chunks of the output a threadgroup apart. */
template <typename input_, typename output_, typename scaling_>
void nk_each_rmsnorm_metal_(device uchar const *x, device float const *gamma, device uchar *y,
                            constant nk_each_rmsnorm_arguments_metal_t &arguments, uint2 vector, uint lane,
                            uint threads, threadgroup float *partials) {
#pragma clang fp contract(off) reassociate(off)
    uint const columns = arguments.columns, chunks = nk::divide_round_up(columns, output_::chunk_values);
    ulong const first = ulong(vector.x) * columns;
    x += vector.y * arguments.x_stride + first * sizeof(typename input_::raw_t);
    y += vector.y * arguments.y_stride + first * sizeof(typename output_::raw_t);
    float sum = 0, compensation = 0;
    for (uint chunk = lane; chunk < chunks; chunk += threads) {
        uint const begin = chunk * output_::chunk_values;
        uint const end = begin + min(columns - begin, output_::chunk_values);
        for (uint column = begin; column != end; ++column) {
            float const value = input_::load(x, column), square = value * value;
            compensation += fma(value, value, -square);
            nk_each_two_sum_metal_(square, sum, compensation);
        }
    }
    float const inverse_rms = nk_each_inverse_rms_metal_(sum, compensation, columns, arguments.epsilon, lane, threads,
                                                         partials);
    for (uint chunk = lane; chunk < chunks; chunk += threads) {
        uint const begin = chunk * output_::chunk_values;
        uint const end = begin + min(columns - begin, output_::chunk_values);
        for (uint column = begin; column != end; ++column) {
            float const value = input_::load(x, column);
            output_::store(y, column, value * inverse_rms * nk_each_gain_metal_(scaling_(), gamma, column));
        }
    }
}

kernel void nk_each_swiglu_f32_gated_metal_kernel_(device uchar const *gate [[buffer(0)]],
                                                   device uchar const *up [[buffer(1)]], device uchar *y [[buffer(2)]],
                                                   constant nk_each_swiglu_arguments_metal_t &arguments [[buffer(3)]],
                                                   uint2 position [[thread_position_in_grid]]) {
    nk_each_swiglu_metal_<nk::f32_t, nk_each_gated_metal_t>(gate, up, y, arguments, position);
}

kernel void nk_each_swiglu_f32_ungated_metal_kernel_(device uchar const *gate [[buffer(0)]],
                                                     device uchar const *up [[buffer(1)]],
                                                     device uchar *y [[buffer(2)]],
                                                     constant nk_each_swiglu_arguments_metal_t &arguments [[buffer(3)]],
                                                     uint2 position [[thread_position_in_grid]]) {
    nk_each_swiglu_metal_<nk::f32_t, nk_each_ungated_metal_t>(gate, up, y, arguments, position);
}

kernel void nk_each_swiglu_f16_gated_metal_kernel_(device uchar const *gate [[buffer(0)]],
                                                   device uchar const *up [[buffer(1)]], device uchar *y [[buffer(2)]],
                                                   constant nk_each_swiglu_arguments_metal_t &arguments [[buffer(3)]],
                                                   uint2 position [[thread_position_in_grid]]) {
    nk_each_swiglu_metal_<nk::f16_t, nk_each_gated_metal_t>(gate, up, y, arguments, position);
}

kernel void nk_each_swiglu_f16_ungated_metal_kernel_(device uchar const *gate [[buffer(0)]],
                                                     device uchar const *up [[buffer(1)]],
                                                     device uchar *y [[buffer(2)]],
                                                     constant nk_each_swiglu_arguments_metal_t &arguments [[buffer(3)]],
                                                     uint2 position [[thread_position_in_grid]]) {
    nk_each_swiglu_metal_<nk::f16_t, nk_each_ungated_metal_t>(gate, up, y, arguments, position);
}

kernel void nk_each_swiglu_bf16_gated_metal_kernel_(device uchar const *gate [[buffer(0)]],
                                                    device uchar const *up [[buffer(1)]], device uchar *y [[buffer(2)]],
                                                    constant nk_each_swiglu_arguments_metal_t &arguments [[buffer(3)]],
                                                    uint2 position [[thread_position_in_grid]]) {
    nk_each_swiglu_metal_<nk::bf16_t, nk_each_gated_metal_t>(gate, up, y, arguments, position);
}

kernel void nk_each_swiglu_bf16_ungated_metal_kernel_(
    device uchar const *gate [[buffer(0)]], device uchar const *up [[buffer(1)]], device uchar *y [[buffer(2)]],
    constant nk_each_swiglu_arguments_metal_t &arguments [[buffer(3)]], uint2 position [[thread_position_in_grid]]) {
    nk_each_swiglu_metal_<nk::bf16_t, nk_each_ungated_metal_t>(gate, up, y, arguments, position);
}

kernel void nk_each_swiglu_e4m3_gated_metal_kernel_(device uchar const *gate [[buffer(0)]],
                                                    device uchar const *up [[buffer(1)]], device uchar *y [[buffer(2)]],
                                                    constant nk_each_swiglu_arguments_metal_t &arguments [[buffer(3)]],
                                                    uint2 position [[thread_position_in_grid]]) {
    nk_each_swiglu_metal_<nk::e4m3_t, nk_each_gated_metal_t>(gate, up, y, arguments, position);
}

kernel void nk_each_swiglu_e4m3_ungated_metal_kernel_(
    device uchar const *gate [[buffer(0)]], device uchar const *up [[buffer(1)]], device uchar *y [[buffer(2)]],
    constant nk_each_swiglu_arguments_metal_t &arguments [[buffer(3)]], uint2 position [[thread_position_in_grid]]) {
    nk_each_swiglu_metal_<nk::e4m3_t, nk_each_ungated_metal_t>(gate, up, y, arguments, position);
}

kernel void nk_each_rmsnorm_f32_scaled_metal_kernel_(
    device uchar const *x [[buffer(0)]], device float const *gamma [[buffer(1)]], device uchar *y [[buffer(2)]],
    constant nk_each_rmsnorm_arguments_metal_t &arguments [[buffer(3)]], uint2 vector [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]], uint2 threads [[threads_per_threadgroup]]) {
    threadgroup float partials[16];
    nk_each_rmsnorm_metal_<nk::f32_t, nk::f32_t, nk_each_scaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                         threads.x, partials);
}

kernel void nk_each_rmsnorm_f32_unscaled_metal_kernel_(
    device uchar const *x [[buffer(0)]], device float const *gamma [[buffer(1)]], device uchar *y [[buffer(2)]],
    constant nk_each_rmsnorm_arguments_metal_t &arguments [[buffer(3)]], uint2 vector [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]], uint2 threads [[threads_per_threadgroup]]) {
    threadgroup float partials[16];
    nk_each_rmsnorm_metal_<nk::f32_t, nk::f32_t, nk_each_unscaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                           threads.x, partials);
}

kernel void nk_each_rmsnorm_f16_scaled_metal_kernel_(
    device uchar const *x [[buffer(0)]], device float const *gamma [[buffer(1)]], device uchar *y [[buffer(2)]],
    constant nk_each_rmsnorm_arguments_metal_t &arguments [[buffer(3)]], uint2 vector [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]], uint2 threads [[threads_per_threadgroup]]) {
    threadgroup float partials[16];
    nk_each_rmsnorm_metal_<nk::f16_t, nk::f16_t, nk_each_scaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                         threads.x, partials);
}

kernel void nk_each_rmsnorm_f16_unscaled_metal_kernel_(
    device uchar const *x [[buffer(0)]], device float const *gamma [[buffer(1)]], device uchar *y [[buffer(2)]],
    constant nk_each_rmsnorm_arguments_metal_t &arguments [[buffer(3)]], uint2 vector [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]], uint2 threads [[threads_per_threadgroup]]) {
    threadgroup float partials[16];
    nk_each_rmsnorm_metal_<nk::f16_t, nk::f16_t, nk_each_unscaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                           threads.x, partials);
}

kernel void nk_each_rmsnorm_bf16_scaled_metal_kernel_(
    device uchar const *x [[buffer(0)]], device float const *gamma [[buffer(1)]], device uchar *y [[buffer(2)]],
    constant nk_each_rmsnorm_arguments_metal_t &arguments [[buffer(3)]], uint2 vector [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]], uint2 threads [[threads_per_threadgroup]]) {
    threadgroup float partials[16];
    nk_each_rmsnorm_metal_<nk::bf16_t, nk::bf16_t, nk_each_scaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                           threads.x, partials);
}

kernel void nk_each_rmsnorm_bf16_unscaled_metal_kernel_(
    device uchar const *x [[buffer(0)]], device float const *gamma [[buffer(1)]], device uchar *y [[buffer(2)]],
    constant nk_each_rmsnorm_arguments_metal_t &arguments [[buffer(3)]], uint2 vector [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]], uint2 threads [[threads_per_threadgroup]]) {
    threadgroup float partials[16];
    nk_each_rmsnorm_metal_<nk::bf16_t, nk::bf16_t, nk_each_unscaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                             threads.x, partials);
}

kernel void nk_each_rmsnorm_e4m3_scaled_metal_kernel_(
    device uchar const *x [[buffer(0)]], device float const *gamma [[buffer(1)]], device uchar *y [[buffer(2)]],
    constant nk_each_rmsnorm_arguments_metal_t &arguments [[buffer(3)]], uint2 vector [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]], uint2 threads [[threads_per_threadgroup]]) {
    threadgroup float partials[16];
    nk_each_rmsnorm_metal_<nk::e4m3_t, nk::e4m3_t, nk_each_scaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                           threads.x, partials);
}

kernel void nk_each_rmsnorm_e4m3_unscaled_metal_kernel_(
    device uchar const *x [[buffer(0)]], device float const *gamma [[buffer(1)]], device uchar *y [[buffer(2)]],
    constant nk_each_rmsnorm_arguments_metal_t &arguments [[buffer(3)]], uint2 vector [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]], uint2 threads [[threads_per_threadgroup]]) {
    threadgroup float partials[16];
    nk_each_rmsnorm_metal_<nk::e4m3_t, nk::e4m3_t, nk_each_unscaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                             threads.x, partials);
}

kernel void nk_each_rmscast_bf16_scaled_metal_kernel_(
    device uchar const *x [[buffer(0)]], device float const *gamma [[buffer(1)]], device uchar *y [[buffer(2)]],
    constant nk_each_rmsnorm_arguments_metal_t &arguments [[buffer(3)]], uint2 vector [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]], uint2 threads [[threads_per_threadgroup]]) {
    threadgroup float partials[16];
    nk_each_rmsnorm_metal_<nk::f32_t, nk::bf16_t, nk_each_scaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                          threads.x, partials);
}

kernel void nk_each_rmscast_bf16_unscaled_metal_kernel_(
    device uchar const *x [[buffer(0)]], device float const *gamma [[buffer(1)]], device uchar *y [[buffer(2)]],
    constant nk_each_rmsnorm_arguments_metal_t &arguments [[buffer(3)]], uint2 vector [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]], uint2 threads [[threads_per_threadgroup]]) {
    threadgroup float partials[16];
    nk_each_rmsnorm_metal_<nk::f32_t, nk::bf16_t, nk_each_unscaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                            threads.x, partials);
}

kernel void nk_each_rmscast_f16_scaled_metal_kernel_(
    device uchar const *x [[buffer(0)]], device float const *gamma [[buffer(1)]], device uchar *y [[buffer(2)]],
    constant nk_each_rmsnorm_arguments_metal_t &arguments [[buffer(3)]], uint2 vector [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]], uint2 threads [[threads_per_threadgroup]]) {
    threadgroup float partials[16];
    nk_each_rmsnorm_metal_<nk::f32_t, nk::f16_t, nk_each_scaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                         threads.x, partials);
}

kernel void nk_each_rmscast_f16_unscaled_metal_kernel_(
    device uchar const *x [[buffer(0)]], device float const *gamma [[buffer(1)]], device uchar *y [[buffer(2)]],
    constant nk_each_rmsnorm_arguments_metal_t &arguments [[buffer(3)]], uint2 vector [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]], uint2 threads [[threads_per_threadgroup]]) {
    threadgroup float partials[16];
    nk_each_rmsnorm_metal_<nk::f32_t, nk::f16_t, nk_each_unscaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                           threads.x, partials);
}

kernel void nk_each_rmscast_e4m3_scaled_metal_kernel_(
    device uchar const *x [[buffer(0)]], device float const *gamma [[buffer(1)]], device uchar *y [[buffer(2)]],
    constant nk_each_rmsnorm_arguments_metal_t &arguments [[buffer(3)]], uint2 vector [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]], uint2 threads [[threads_per_threadgroup]]) {
    threadgroup float partials[16];
    nk_each_rmsnorm_metal_<nk::f32_t, nk::e4m3_t, nk_each_scaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                          threads.x, partials);
}

kernel void nk_each_rmscast_e4m3_unscaled_metal_kernel_(
    device uchar const *x [[buffer(0)]], device float const *gamma [[buffer(1)]], device uchar *y [[buffer(2)]],
    constant nk_each_rmsnorm_arguments_metal_t &arguments [[buffer(3)]], uint2 vector [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]], uint2 threads [[threads_per_threadgroup]]) {
    threadgroup float partials[16];
    nk_each_rmsnorm_metal_<nk::f32_t, nk::e4m3_t, nk_each_unscaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                            threads.x, partials);
}

kernel void nk_each_rmscast_e5m2_scaled_metal_kernel_(
    device uchar const *x [[buffer(0)]], device float const *gamma [[buffer(1)]], device uchar *y [[buffer(2)]],
    constant nk_each_rmsnorm_arguments_metal_t &arguments [[buffer(3)]], uint2 vector [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]], uint2 threads [[threads_per_threadgroup]]) {
    threadgroup float partials[16];
    nk_each_rmsnorm_metal_<nk::f32_t, nk::e5m2_t, nk_each_scaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                          threads.x, partials);
}

kernel void nk_each_rmscast_e5m2_unscaled_metal_kernel_(
    device uchar const *x [[buffer(0)]], device float const *gamma [[buffer(1)]], device uchar *y [[buffer(2)]],
    constant nk_each_rmsnorm_arguments_metal_t &arguments [[buffer(3)]], uint2 vector [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]], uint2 threads [[threads_per_threadgroup]]) {
    threadgroup float partials[16];
    nk_each_rmsnorm_metal_<nk::f32_t, nk::e5m2_t, nk_each_unscaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                            threads.x, partials);
}

kernel void nk_each_rmscast_e2m3_scaled_metal_kernel_(
    device uchar const *x [[buffer(0)]], device float const *gamma [[buffer(1)]], device uchar *y [[buffer(2)]],
    constant nk_each_rmsnorm_arguments_metal_t &arguments [[buffer(3)]], uint2 vector [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]], uint2 threads [[threads_per_threadgroup]]) {
    threadgroup float partials[16];
    nk_each_rmsnorm_metal_<nk::f32_t, nk::e2m3_t, nk_each_scaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                          threads.x, partials);
}

kernel void nk_each_rmscast_e2m3_unscaled_metal_kernel_(
    device uchar const *x [[buffer(0)]], device float const *gamma [[buffer(1)]], device uchar *y [[buffer(2)]],
    constant nk_each_rmsnorm_arguments_metal_t &arguments [[buffer(3)]], uint2 vector [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]], uint2 threads [[threads_per_threadgroup]]) {
    threadgroup float partials[16];
    nk_each_rmsnorm_metal_<nk::f32_t, nk::e2m3_t, nk_each_unscaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                            threads.x, partials);
}

kernel void nk_each_rmscast_e3m2_scaled_metal_kernel_(
    device uchar const *x [[buffer(0)]], device float const *gamma [[buffer(1)]], device uchar *y [[buffer(2)]],
    constant nk_each_rmsnorm_arguments_metal_t &arguments [[buffer(3)]], uint2 vector [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]], uint2 threads [[threads_per_threadgroup]]) {
    threadgroup float partials[16];
    nk_each_rmsnorm_metal_<nk::f32_t, nk::e3m2_t, nk_each_scaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                          threads.x, partials);
}

kernel void nk_each_rmscast_e3m2_unscaled_metal_kernel_(
    device uchar const *x [[buffer(0)]], device float const *gamma [[buffer(1)]], device uchar *y [[buffer(2)]],
    constant nk_each_rmsnorm_arguments_metal_t &arguments [[buffer(3)]], uint2 vector [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]], uint2 threads [[threads_per_threadgroup]]) {
    threadgroup float partials[16];
    nk_each_rmsnorm_metal_<nk::f32_t, nk::e3m2_t, nk_each_unscaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                            threads.x, partials);
}

kernel void nk_each_rmscast_i8_scaled_metal_kernel_(
    device uchar const *x [[buffer(0)]], device float const *gamma [[buffer(1)]], device uchar *y [[buffer(2)]],
    constant nk_each_rmsnorm_arguments_metal_t &arguments [[buffer(3)]], uint2 vector [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]], uint2 threads [[threads_per_threadgroup]]) {
    threadgroup float partials[16];
    nk_each_rmsnorm_metal_<nk::i32_t, nk::i8_t, nk_each_scaled_metal_t>(x, gamma, y, arguments, vector, lane, threads.x,
                                                                        partials);
}

kernel void nk_each_rmscast_i8_unscaled_metal_kernel_(
    device uchar const *x [[buffer(0)]], device float const *gamma [[buffer(1)]], device uchar *y [[buffer(2)]],
    constant nk_each_rmsnorm_arguments_metal_t &arguments [[buffer(3)]], uint2 vector [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]], uint2 threads [[threads_per_threadgroup]]) {
    threadgroup float partials[16];
    nk_each_rmsnorm_metal_<nk::i32_t, nk::i8_t, nk_each_unscaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                          threads.x, partials);
}

kernel void nk_each_rmscast_u8_scaled_metal_kernel_(
    device uchar const *x [[buffer(0)]], device float const *gamma [[buffer(1)]], device uchar *y [[buffer(2)]],
    constant nk_each_rmsnorm_arguments_metal_t &arguments [[buffer(3)]], uint2 vector [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]], uint2 threads [[threads_per_threadgroup]]) {
    threadgroup float partials[16];
    nk_each_rmsnorm_metal_<nk::u32_t, nk::u8_t, nk_each_scaled_metal_t>(x, gamma, y, arguments, vector, lane, threads.x,
                                                                        partials);
}

kernel void nk_each_rmscast_u8_unscaled_metal_kernel_(
    device uchar const *x [[buffer(0)]], device float const *gamma [[buffer(1)]], device uchar *y [[buffer(2)]],
    constant nk_each_rmsnorm_arguments_metal_t &arguments [[buffer(3)]], uint2 vector [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]], uint2 threads [[threads_per_threadgroup]]) {
    threadgroup float partials[16];
    nk_each_rmsnorm_metal_<nk::u32_t, nk::u8_t, nk_each_unscaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                          threads.x, partials);
}
