/**
 *  @file include/numkong/each/metal.metal
 *  @author Ash Vardanian
 *  @date October 5, 2026
 *  @brief Element-wise sums, SwiGLU, grouped RMSNorm and its downcasting form on the SIMT cores of
 *      every Apple GPU.
 *
 *  @sa include/numkong/each/metal.h, which embeds and launches this source
 *  @sa include/numkong/each/simt.cuh, the CUDA and ROCm sibling
 *
 *  Sums run a grid of 16-byte chunks, each thread adding one chunk in F32 and rounding once to the
 *  stored type. SwiGLU runs a grid of 16-byte chunks by rows, each thread one chunk of one row.
 *  RMSNorm gives each normalized vector one threadgroup, in a grid of groups by rows, whose threads
 *  take 16-byte chunks of the output a threadgroup apart, reading the input twice rather than
 *  keeping it. Squares sum in plain F32, merged down each SIMD-group through shuffles and then
 *  across SIMD-groups. Only the RMSNorm of F32 stays out, as it reduces in F64, which no Apple GPU
 *  has, while the downcasts from F32 reduce in F32. The host picks the kernel for a missing @c up
 *  or @c gamma, so no thread branches on either.
 *
 *  Every kernel binds only its launch record, at buffer slot 0, and reads its operands through the
 *  GPU addresses that open it, so a caller holding its own heaps may fill the record directly.
 */

/** The launch record of SwiGLU, laid out as the C launcher's record of the same name. It opens
 *  with the GPU addresses of the operands, and @c up is 0 when absent, as only the ungated kernels
 *  take it, which run plain SiLU. */
struct nk_each_swiglu_arguments_metal_t {
    ulong gate, up, y;
    ulong gate_stride, up_stride, y_stride;
    uint columns;
    float gate_scale, output_scale;
};

/** The launch record of RMSNorm, laid out as the C launcher's record of the same name. It opens
 *  with the GPU addresses of the operands, and @c gamma is 0 when absent, as only the unscaled
 *  kernels take it, which run plain RMS. */
struct nk_each_rmsnorm_arguments_metal_t {
    ulong x, gamma, y;
    ulong x_stride, y_stride;
    uint columns;
    float epsilon;
};

/** The launch record of a sum, laid out as the C launcher's record of the same name, opening with
 *  the GPU addresses of the operands. */
struct nk_each_sum_arguments_metal_t {
    ulong a, b, result;
    ulong count;
};

static_assert(sizeof(nk_each_swiglu_arguments_metal_t) == 64, "mirrors the C record in each/metal.h");
static_assert(sizeof(nk_each_rmsnorm_arguments_metal_t) == 48, "mirrors the C record in each/metal.h");
static_assert(sizeof(nk_each_sum_arguments_metal_t) == 32, "mirrors the C record in each/metal.h");

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

/** SiLU of @p value, x / (1 + e^-x), through @c precise::exp, within a few ULP as CUDA's @c expf
 *  is, keeping the subnormal precision of tiny reciprocals. */
inline float nk_each_silu_metal_(float value) {
#pragma clang fp contract(off) reassociate(off)
    float const denominator = 1.0f + precise::exp(-value);
    // Scale subnormal reciprocals into integers before Metal flushes them
    return denominator > 0x1p126f
               ? ldexp(value * (rint(precise::divide(0x1p126f, denominator) * 0x1p23f) * 0x1p-23f), -126)
               : value * precise::divide(1.0f, denominator);
}

/** The sum of one 16-byte @p chunk of @p a and @p b, widened to F32 and rounded once on store. */
template <typename dtype_>
void nk_each_sum_metal_(device uchar const *a, device uchar const *b, device uchar *result,
                        constant nk_each_sum_arguments_metal_t &arguments, uint chunk) {
    // Fast math may flush subnormal operands and sums, which the serial sum keeps
#pragma METAL fp math_mode(safe)
    ulong const first = ulong(chunk) * dtype_::chunk_values;
    if (first >= arguments.count) return;
    ulong const end = min(arguments.count, first + dtype_::chunk_values);
    // A row pointer per chunk keeps every index within the 32 bits `load` and `store` take
    ulong const offset = first * sizeof(typename dtype_::raw_t);
    a += offset, b += offset, result += offset;
    for (uint index = 0; index != uint(end - first); ++index)
        dtype_::store(result, index, dtype_::load(a, index) + dtype_::load(b, index));
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

/** The inverse RMS of @p count values from each thread's sum of their squares, the same in every
 *  thread: down each SIMD-group through shuffles, then across the SIMD-groups of @p threads through
 *  @p partials, one sum per SIMD-group. */
inline float nk_each_inverse_rms_metal_(float sum, uint count, float epsilon, uint lane, uint threads,
                                        threadgroup float *partials) {
    uint const simd_lane = lane % 32, simd_group = lane / 32, simd_groups = threads / 32;
    for (uint offset = 16; offset; offset >>= 1) sum += simd_shuffle_down(sum, offset);
    if (!simd_lane) partials[simd_group] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!simd_group) {
        sum = simd_lane < simd_groups ? partials[simd_lane] : 0;
        for (uint offset = 4; offset; offset >>= 1) sum += simd_shuffle_down(sum, offset);
        if (!simd_lane) partials[0] = precise::divide(sum, float(count));
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
    float sum = 0;
    for (uint chunk = lane; chunk < chunks; chunk += threads) {
        uint const begin = chunk * output_::chunk_values;
        uint const end = begin + min(columns - begin, output_::chunk_values);
        for (uint column = begin; column != end; ++column) {
            float const value = input_::load(x, column);
            sum += value * value;
        }
    }
    float const inverse_rms = nk_each_inverse_rms_metal_(sum, columns, arguments.epsilon, lane, threads, partials);
    for (uint chunk = lane; chunk < chunks; chunk += threads) {
        uint const begin = chunk * output_::chunk_values;
        uint const end = begin + min(columns - begin, output_::chunk_values);
        for (uint column = begin; column != end; ++column) {
            float const value = input_::load(x, column);
            output_::store(y, column, value * inverse_rms * nk_each_gain_metal_(scaling_(), gamma, column));
        }
    }
}

kernel void nk_each_sum_f32_metal_kernel_(constant nk_each_sum_arguments_metal_t &arguments [[buffer(0)]],
                                          uint chunk [[thread_position_in_grid]]) {
    device uchar const *a = (device uchar const *)arguments.a;
    device uchar const *b = (device uchar const *)arguments.b;
    device uchar *result = (device uchar *)arguments.result;
    nk_each_sum_metal_<nk::f32_t>(a, b, result, arguments, chunk);
}

kernel void nk_each_sum_f16_metal_kernel_(constant nk_each_sum_arguments_metal_t &arguments [[buffer(0)]],
                                          uint chunk [[thread_position_in_grid]]) {
    device uchar const *a = (device uchar const *)arguments.a;
    device uchar const *b = (device uchar const *)arguments.b;
    device uchar *result = (device uchar *)arguments.result;
    nk_each_sum_metal_<nk::f16_t>(a, b, result, arguments, chunk);
}

kernel void nk_each_sum_bf16_metal_kernel_(constant nk_each_sum_arguments_metal_t &arguments [[buffer(0)]],
                                           uint chunk [[thread_position_in_grid]]) {
    device uchar const *a = (device uchar const *)arguments.a;
    device uchar const *b = (device uchar const *)arguments.b;
    device uchar *result = (device uchar *)arguments.result;
    nk_each_sum_metal_<nk::bf16_t>(a, b, result, arguments, chunk);
}

kernel void nk_each_swiglu_f32_gated_metal_kernel_(constant nk_each_swiglu_arguments_metal_t &arguments [[buffer(0)]],
                                                   uint2 position [[thread_position_in_grid]]) {
    device uchar const *gate = (device uchar const *)arguments.gate;
    device uchar const *up = (device uchar const *)arguments.up;
    device uchar *y = (device uchar *)arguments.y;
    nk_each_swiglu_metal_<nk::f32_t, nk_each_gated_metal_t>(gate, up, y, arguments, position);
}

kernel void nk_each_swiglu_f32_ungated_metal_kernel_(constant nk_each_swiglu_arguments_metal_t &arguments [[buffer(0)]],
                                                     uint2 position [[thread_position_in_grid]]) {
    device uchar const *gate = (device uchar const *)arguments.gate;
    device uchar const *up = (device uchar const *)arguments.up;
    device uchar *y = (device uchar *)arguments.y;
    nk_each_swiglu_metal_<nk::f32_t, nk_each_ungated_metal_t>(gate, up, y, arguments, position);
}

kernel void nk_each_swiglu_f16_gated_metal_kernel_(constant nk_each_swiglu_arguments_metal_t &arguments [[buffer(0)]],
                                                   uint2 position [[thread_position_in_grid]]) {
    device uchar const *gate = (device uchar const *)arguments.gate;
    device uchar const *up = (device uchar const *)arguments.up;
    device uchar *y = (device uchar *)arguments.y;
    nk_each_swiglu_metal_<nk::f16_t, nk_each_gated_metal_t>(gate, up, y, arguments, position);
}

kernel void nk_each_swiglu_f16_ungated_metal_kernel_(constant nk_each_swiglu_arguments_metal_t &arguments [[buffer(0)]],
                                                     uint2 position [[thread_position_in_grid]]) {
    device uchar const *gate = (device uchar const *)arguments.gate;
    device uchar const *up = (device uchar const *)arguments.up;
    device uchar *y = (device uchar *)arguments.y;
    nk_each_swiglu_metal_<nk::f16_t, nk_each_ungated_metal_t>(gate, up, y, arguments, position);
}

kernel void nk_each_swiglu_bf16_gated_metal_kernel_(constant nk_each_swiglu_arguments_metal_t &arguments [[buffer(0)]],
                                                    uint2 position [[thread_position_in_grid]]) {
    device uchar const *gate = (device uchar const *)arguments.gate;
    device uchar const *up = (device uchar const *)arguments.up;
    device uchar *y = (device uchar *)arguments.y;
    nk_each_swiglu_metal_<nk::bf16_t, nk_each_gated_metal_t>(gate, up, y, arguments, position);
}

kernel void nk_each_swiglu_bf16_ungated_metal_kernel_(constant nk_each_swiglu_arguments_metal_t &arguments
                                                      [[buffer(0)]],
                                                      uint2 position [[thread_position_in_grid]]) {
    device uchar const *gate = (device uchar const *)arguments.gate;
    device uchar const *up = (device uchar const *)arguments.up;
    device uchar *y = (device uchar *)arguments.y;
    nk_each_swiglu_metal_<nk::bf16_t, nk_each_ungated_metal_t>(gate, up, y, arguments, position);
}

kernel void nk_each_swiglu_e4m3_gated_metal_kernel_(constant nk_each_swiglu_arguments_metal_t &arguments [[buffer(0)]],
                                                    uint2 position [[thread_position_in_grid]]) {
    device uchar const *gate = (device uchar const *)arguments.gate;
    device uchar const *up = (device uchar const *)arguments.up;
    device uchar *y = (device uchar *)arguments.y;
    nk_each_swiglu_metal_<nk::e4m3_t, nk_each_gated_metal_t>(gate, up, y, arguments, position);
}

kernel void nk_each_swiglu_e4m3_ungated_metal_kernel_(constant nk_each_swiglu_arguments_metal_t &arguments
                                                      [[buffer(0)]],
                                                      uint2 position [[thread_position_in_grid]]) {
    device uchar const *gate = (device uchar const *)arguments.gate;
    device uchar const *up = (device uchar const *)arguments.up;
    device uchar *y = (device uchar *)arguments.y;
    nk_each_swiglu_metal_<nk::e4m3_t, nk_each_ungated_metal_t>(gate, up, y, arguments, position);
}

kernel void nk_each_rmsnorm_f16_scaled_metal_kernel_(constant nk_each_rmsnorm_arguments_metal_t &arguments
                                                     [[buffer(0)]],
                                                     uint2 vector [[threadgroup_position_in_grid]],
                                                     uint lane [[thread_index_in_threadgroup]],
                                                     uint2 threads [[threads_per_threadgroup]]) {
    device uchar const *x = (device uchar const *)arguments.x;
    device float const *gamma = (device float const *)arguments.gamma;
    device uchar *y = (device uchar *)arguments.y;
    threadgroup float partials[8];
    nk_each_rmsnorm_metal_<nk::f16_t, nk::f16_t, nk_each_scaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                         threads.x, partials);
}

kernel void nk_each_rmsnorm_f16_unscaled_metal_kernel_(constant nk_each_rmsnorm_arguments_metal_t &arguments
                                                       [[buffer(0)]],
                                                       uint2 vector [[threadgroup_position_in_grid]],
                                                       uint lane [[thread_index_in_threadgroup]],
                                                       uint2 threads [[threads_per_threadgroup]]) {
    device uchar const *x = (device uchar const *)arguments.x;
    device float const *gamma = (device float const *)arguments.gamma;
    device uchar *y = (device uchar *)arguments.y;
    threadgroup float partials[8];
    nk_each_rmsnorm_metal_<nk::f16_t, nk::f16_t, nk_each_unscaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                           threads.x, partials);
}

kernel void nk_each_rmsnorm_bf16_scaled_metal_kernel_(constant nk_each_rmsnorm_arguments_metal_t &arguments
                                                      [[buffer(0)]],
                                                      uint2 vector [[threadgroup_position_in_grid]],
                                                      uint lane [[thread_index_in_threadgroup]],
                                                      uint2 threads [[threads_per_threadgroup]]) {
    device uchar const *x = (device uchar const *)arguments.x;
    device float const *gamma = (device float const *)arguments.gamma;
    device uchar *y = (device uchar *)arguments.y;
    threadgroup float partials[8];
    nk_each_rmsnorm_metal_<nk::bf16_t, nk::bf16_t, nk_each_scaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                           threads.x, partials);
}

kernel void nk_each_rmsnorm_bf16_unscaled_metal_kernel_(constant nk_each_rmsnorm_arguments_metal_t &arguments
                                                        [[buffer(0)]],
                                                        uint2 vector [[threadgroup_position_in_grid]],
                                                        uint lane [[thread_index_in_threadgroup]],
                                                        uint2 threads [[threads_per_threadgroup]]) {
    device uchar const *x = (device uchar const *)arguments.x;
    device float const *gamma = (device float const *)arguments.gamma;
    device uchar *y = (device uchar *)arguments.y;
    threadgroup float partials[8];
    nk_each_rmsnorm_metal_<nk::bf16_t, nk::bf16_t, nk_each_unscaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                             threads.x, partials);
}

kernel void nk_each_rmsnorm_e4m3_scaled_metal_kernel_(constant nk_each_rmsnorm_arguments_metal_t &arguments
                                                      [[buffer(0)]],
                                                      uint2 vector [[threadgroup_position_in_grid]],
                                                      uint lane [[thread_index_in_threadgroup]],
                                                      uint2 threads [[threads_per_threadgroup]]) {
    device uchar const *x = (device uchar const *)arguments.x;
    device float const *gamma = (device float const *)arguments.gamma;
    device uchar *y = (device uchar *)arguments.y;
    threadgroup float partials[8];
    nk_each_rmsnorm_metal_<nk::e4m3_t, nk::e4m3_t, nk_each_scaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                           threads.x, partials);
}

kernel void nk_each_rmsnorm_e4m3_unscaled_metal_kernel_(constant nk_each_rmsnorm_arguments_metal_t &arguments
                                                        [[buffer(0)]],
                                                        uint2 vector [[threadgroup_position_in_grid]],
                                                        uint lane [[thread_index_in_threadgroup]],
                                                        uint2 threads [[threads_per_threadgroup]]) {
    device uchar const *x = (device uchar const *)arguments.x;
    device float const *gamma = (device float const *)arguments.gamma;
    device uchar *y = (device uchar *)arguments.y;
    threadgroup float partials[8];
    nk_each_rmsnorm_metal_<nk::e4m3_t, nk::e4m3_t, nk_each_unscaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                             threads.x, partials);
}

kernel void nk_each_rmscast_bf16_scaled_metal_kernel_(constant nk_each_rmsnorm_arguments_metal_t &arguments
                                                      [[buffer(0)]],
                                                      uint2 vector [[threadgroup_position_in_grid]],
                                                      uint lane [[thread_index_in_threadgroup]],
                                                      uint2 threads [[threads_per_threadgroup]]) {
    device uchar const *x = (device uchar const *)arguments.x;
    device float const *gamma = (device float const *)arguments.gamma;
    device uchar *y = (device uchar *)arguments.y;
    threadgroup float partials[8];
    nk_each_rmsnorm_metal_<nk::f32_t, nk::bf16_t, nk_each_scaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                          threads.x, partials);
}

kernel void nk_each_rmscast_bf16_unscaled_metal_kernel_(constant nk_each_rmsnorm_arguments_metal_t &arguments
                                                        [[buffer(0)]],
                                                        uint2 vector [[threadgroup_position_in_grid]],
                                                        uint lane [[thread_index_in_threadgroup]],
                                                        uint2 threads [[threads_per_threadgroup]]) {
    device uchar const *x = (device uchar const *)arguments.x;
    device float const *gamma = (device float const *)arguments.gamma;
    device uchar *y = (device uchar *)arguments.y;
    threadgroup float partials[8];
    nk_each_rmsnorm_metal_<nk::f32_t, nk::bf16_t, nk_each_unscaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                            threads.x, partials);
}

kernel void nk_each_rmscast_f16_scaled_metal_kernel_(constant nk_each_rmsnorm_arguments_metal_t &arguments
                                                     [[buffer(0)]],
                                                     uint2 vector [[threadgroup_position_in_grid]],
                                                     uint lane [[thread_index_in_threadgroup]],
                                                     uint2 threads [[threads_per_threadgroup]]) {
    device uchar const *x = (device uchar const *)arguments.x;
    device float const *gamma = (device float const *)arguments.gamma;
    device uchar *y = (device uchar *)arguments.y;
    threadgroup float partials[8];
    nk_each_rmsnorm_metal_<nk::f32_t, nk::f16_t, nk_each_scaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                         threads.x, partials);
}

kernel void nk_each_rmscast_f16_unscaled_metal_kernel_(constant nk_each_rmsnorm_arguments_metal_t &arguments
                                                       [[buffer(0)]],
                                                       uint2 vector [[threadgroup_position_in_grid]],
                                                       uint lane [[thread_index_in_threadgroup]],
                                                       uint2 threads [[threads_per_threadgroup]]) {
    device uchar const *x = (device uchar const *)arguments.x;
    device float const *gamma = (device float const *)arguments.gamma;
    device uchar *y = (device uchar *)arguments.y;
    threadgroup float partials[8];
    nk_each_rmsnorm_metal_<nk::f32_t, nk::f16_t, nk_each_unscaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                           threads.x, partials);
}

kernel void nk_each_rmscast_e4m3_scaled_metal_kernel_(constant nk_each_rmsnorm_arguments_metal_t &arguments
                                                      [[buffer(0)]],
                                                      uint2 vector [[threadgroup_position_in_grid]],
                                                      uint lane [[thread_index_in_threadgroup]],
                                                      uint2 threads [[threads_per_threadgroup]]) {
    device uchar const *x = (device uchar const *)arguments.x;
    device float const *gamma = (device float const *)arguments.gamma;
    device uchar *y = (device uchar *)arguments.y;
    threadgroup float partials[8];
    nk_each_rmsnorm_metal_<nk::f32_t, nk::e4m3_t, nk_each_scaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                          threads.x, partials);
}

kernel void nk_each_rmscast_e4m3_unscaled_metal_kernel_(constant nk_each_rmsnorm_arguments_metal_t &arguments
                                                        [[buffer(0)]],
                                                        uint2 vector [[threadgroup_position_in_grid]],
                                                        uint lane [[thread_index_in_threadgroup]],
                                                        uint2 threads [[threads_per_threadgroup]]) {
    device uchar const *x = (device uchar const *)arguments.x;
    device float const *gamma = (device float const *)arguments.gamma;
    device uchar *y = (device uchar *)arguments.y;
    threadgroup float partials[8];
    nk_each_rmsnorm_metal_<nk::f32_t, nk::e4m3_t, nk_each_unscaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                            threads.x, partials);
}

kernel void nk_each_rmscast_e5m2_scaled_metal_kernel_(constant nk_each_rmsnorm_arguments_metal_t &arguments
                                                      [[buffer(0)]],
                                                      uint2 vector [[threadgroup_position_in_grid]],
                                                      uint lane [[thread_index_in_threadgroup]],
                                                      uint2 threads [[threads_per_threadgroup]]) {
    device uchar const *x = (device uchar const *)arguments.x;
    device float const *gamma = (device float const *)arguments.gamma;
    device uchar *y = (device uchar *)arguments.y;
    threadgroup float partials[8];
    nk_each_rmsnorm_metal_<nk::f32_t, nk::e5m2_t, nk_each_scaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                          threads.x, partials);
}

kernel void nk_each_rmscast_e5m2_unscaled_metal_kernel_(constant nk_each_rmsnorm_arguments_metal_t &arguments
                                                        [[buffer(0)]],
                                                        uint2 vector [[threadgroup_position_in_grid]],
                                                        uint lane [[thread_index_in_threadgroup]],
                                                        uint2 threads [[threads_per_threadgroup]]) {
    device uchar const *x = (device uchar const *)arguments.x;
    device float const *gamma = (device float const *)arguments.gamma;
    device uchar *y = (device uchar *)arguments.y;
    threadgroup float partials[8];
    nk_each_rmsnorm_metal_<nk::f32_t, nk::e5m2_t, nk_each_unscaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                            threads.x, partials);
}

kernel void nk_each_rmscast_e2m3_scaled_metal_kernel_(constant nk_each_rmsnorm_arguments_metal_t &arguments
                                                      [[buffer(0)]],
                                                      uint2 vector [[threadgroup_position_in_grid]],
                                                      uint lane [[thread_index_in_threadgroup]],
                                                      uint2 threads [[threads_per_threadgroup]]) {
    device uchar const *x = (device uchar const *)arguments.x;
    device float const *gamma = (device float const *)arguments.gamma;
    device uchar *y = (device uchar *)arguments.y;
    threadgroup float partials[8];
    nk_each_rmsnorm_metal_<nk::f32_t, nk::e2m3_t, nk_each_scaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                          threads.x, partials);
}

kernel void nk_each_rmscast_e2m3_unscaled_metal_kernel_(constant nk_each_rmsnorm_arguments_metal_t &arguments
                                                        [[buffer(0)]],
                                                        uint2 vector [[threadgroup_position_in_grid]],
                                                        uint lane [[thread_index_in_threadgroup]],
                                                        uint2 threads [[threads_per_threadgroup]]) {
    device uchar const *x = (device uchar const *)arguments.x;
    device float const *gamma = (device float const *)arguments.gamma;
    device uchar *y = (device uchar *)arguments.y;
    threadgroup float partials[8];
    nk_each_rmsnorm_metal_<nk::f32_t, nk::e2m3_t, nk_each_unscaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                            threads.x, partials);
}

kernel void nk_each_rmscast_e3m2_scaled_metal_kernel_(constant nk_each_rmsnorm_arguments_metal_t &arguments
                                                      [[buffer(0)]],
                                                      uint2 vector [[threadgroup_position_in_grid]],
                                                      uint lane [[thread_index_in_threadgroup]],
                                                      uint2 threads [[threads_per_threadgroup]]) {
    device uchar const *x = (device uchar const *)arguments.x;
    device float const *gamma = (device float const *)arguments.gamma;
    device uchar *y = (device uchar *)arguments.y;
    threadgroup float partials[8];
    nk_each_rmsnorm_metal_<nk::f32_t, nk::e3m2_t, nk_each_scaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                          threads.x, partials);
}

kernel void nk_each_rmscast_e3m2_unscaled_metal_kernel_(constant nk_each_rmsnorm_arguments_metal_t &arguments
                                                        [[buffer(0)]],
                                                        uint2 vector [[threadgroup_position_in_grid]],
                                                        uint lane [[thread_index_in_threadgroup]],
                                                        uint2 threads [[threads_per_threadgroup]]) {
    device uchar const *x = (device uchar const *)arguments.x;
    device float const *gamma = (device float const *)arguments.gamma;
    device uchar *y = (device uchar *)arguments.y;
    threadgroup float partials[8];
    nk_each_rmsnorm_metal_<nk::f32_t, nk::e3m2_t, nk_each_unscaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                            threads.x, partials);
}

kernel void nk_each_rmscast_i8_scaled_metal_kernel_(constant nk_each_rmsnorm_arguments_metal_t &arguments [[buffer(0)]],
                                                    uint2 vector [[threadgroup_position_in_grid]],
                                                    uint lane [[thread_index_in_threadgroup]],
                                                    uint2 threads [[threads_per_threadgroup]]) {
    device uchar const *x = (device uchar const *)arguments.x;
    device float const *gamma = (device float const *)arguments.gamma;
    device uchar *y = (device uchar *)arguments.y;
    threadgroup float partials[8];
    nk_each_rmsnorm_metal_<nk::i32_t, nk::i8_t, nk_each_scaled_metal_t>(x, gamma, y, arguments, vector, lane, threads.x,
                                                                        partials);
}

kernel void nk_each_rmscast_i8_unscaled_metal_kernel_(constant nk_each_rmsnorm_arguments_metal_t &arguments
                                                      [[buffer(0)]],
                                                      uint2 vector [[threadgroup_position_in_grid]],
                                                      uint lane [[thread_index_in_threadgroup]],
                                                      uint2 threads [[threads_per_threadgroup]]) {
    device uchar const *x = (device uchar const *)arguments.x;
    device float const *gamma = (device float const *)arguments.gamma;
    device uchar *y = (device uchar *)arguments.y;
    threadgroup float partials[8];
    nk_each_rmsnorm_metal_<nk::i32_t, nk::i8_t, nk_each_unscaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                          threads.x, partials);
}

kernel void nk_each_rmscast_u8_scaled_metal_kernel_(constant nk_each_rmsnorm_arguments_metal_t &arguments [[buffer(0)]],
                                                    uint2 vector [[threadgroup_position_in_grid]],
                                                    uint lane [[thread_index_in_threadgroup]],
                                                    uint2 threads [[threads_per_threadgroup]]) {
    device uchar const *x = (device uchar const *)arguments.x;
    device float const *gamma = (device float const *)arguments.gamma;
    device uchar *y = (device uchar *)arguments.y;
    threadgroup float partials[8];
    nk_each_rmsnorm_metal_<nk::u32_t, nk::u8_t, nk_each_scaled_metal_t>(x, gamma, y, arguments, vector, lane, threads.x,
                                                                        partials);
}

kernel void nk_each_rmscast_u8_unscaled_metal_kernel_(constant nk_each_rmsnorm_arguments_metal_t &arguments
                                                      [[buffer(0)]],
                                                      uint2 vector [[threadgroup_position_in_grid]],
                                                      uint lane [[thread_index_in_threadgroup]],
                                                      uint2 threads [[threads_per_threadgroup]]) {
    device uchar const *x = (device uchar const *)arguments.x;
    device float const *gamma = (device float const *)arguments.gamma;
    device uchar *y = (device uchar *)arguments.y;
    threadgroup float partials[8];
    nk_each_rmsnorm_metal_<nk::u32_t, nk::u8_t, nk_each_unscaled_metal_t>(x, gamma, y, arguments, vector, lane,
                                                                          threads.x, partials);
}
