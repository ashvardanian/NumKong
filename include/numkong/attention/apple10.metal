/**
 *  @file include/numkong/attention/apple10.metal
 *  @author Ash Vardanian
 *  @date October 5, 2026
 *  @brief Tiled ragged attention on the tensor operations of Apple family 10 GPUs.
 *
 *  @sa include/numkong/attention/apple10.h, which embeds this source after `metal.metal`
 *
 *  @c matmul2d runs Q · K of a 32-row tile against a 64-key panel and P · V against 32 channels of
 *  its values, four simdgroups cooperating, on operands staged in @c half, or @c bfloat for BF16.
 *  I8 stages its codes and U8 weights offset by −128 as @c int8_t, accumulates in @c int, and adds
 *  the offset back as 128 times each channel's value sum.
 */
#include <metal_tensor>

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

/** The Q · K and P · V tiles of @c nk_attention_matrix_metal_ on @c matmul2d. */
struct nk_attention_operations_apple10_t {
    template <typename stage_type_, typename element_type_>
    static inline void qk(device uchar const *queries, thread nk_attention_work_metal_t const &work,
                          constant nk_attention_arguments_metal_t &a, ulong head, ulong row_first, ulong key_first,
                          threadgroup stage_type_ *left, threadgroup stage_type_ *right, threadgroup float *scores,
                          uint thread_index, [[maybe_unused]] uint warp) {
        using stage_t = stage_type_;
        constexpr uint step = sizeof(stage_t) == 1 ? 64 : 32;
        using result_t = conditional_t<sizeof(stage_t) == 1, int, float>;
        constexpr auto descriptor = matmul2d_descriptor(32, 64, step, false, true, false,
                                                        matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<descriptor, execution_simdgroups<4>> multiply;
        tensor<threadgroup stage_t, extents<int32_t, step, 32>, tensor_inline> query(left,
                                                                                     extents<int32_t, step, 32>());
        tensor<threadgroup stage_t, extents<int32_t, step, 64>, tensor_inline> key(right, extents<int32_t, step, 64>());
        auto tile = multiply.template get_destination_cooperative_tensor<decltype(query), decltype(key), result_t>();
        for (uint16_t i = 0; i < tile.get_capacity(); ++i)
            if (tile.is_valid_element(i)) tile[i] = 0;
        for (ulong depth_first = 0; depth_first < a.depth; depth_first += step) {
            nk_attention_stage_qk_metal_<stage_t, step, element_type_>(queries, work, a, head, row_first, key_first,
                                                                       depth_first, left, right, thread_index);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            multiply.run(query, key, tile);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint16_t i = 0; i < tile.get_capacity(); ++i) {
            if (!tile.is_valid_element(i)) continue;
            auto const coordinate = tile.get_multidimensional_index(i);
            scores[coordinate[1] * 64 + coordinate[0]] = float(tile[i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    template <typename stage_type_>
    static inline void pv(threadgroup stage_type_ *left, threadgroup stage_type_ *right, threadgroup float *scores,
                          uint thread_index, [[maybe_unused]] uint warp) {
        using stage_t = stage_type_;
        using result_t = conditional_t<sizeof(stage_t) == 1, int, float>;
        constexpr auto descriptor = matmul2d_descriptor(32, 32, 64, false, false, false);
        matmul2d<descriptor, execution_simdgroups<4>> multiply;
        tensor<threadgroup stage_t, extents<int32_t, 64, 32>, tensor_inline> weights(left, extents<int32_t, 64, 32>());
        tensor<threadgroup stage_t, extents<int32_t, 32, 64>, tensor_inline> values(right, extents<int32_t, 32, 64>());
        auto tile =
            multiply.template get_destination_cooperative_tensor<decltype(weights), decltype(values), result_t>();
        for (uint16_t i = 0; i < tile.get_capacity(); ++i)
            if (tile.is_valid_element(i)) tile[i] = 0;
        threadgroup int *bias = (threadgroup int *)(scores + 1024);
        if constexpr (sizeof(stage_t) == 1) {
            if (thread_index < 32) {
                int sum = 0;
                for (uint key = 0; key < 64; ++key) sum += int(right[key * 32 + thread_index]);
                bias[thread_index] = 128 * sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        multiply.run(weights, values, tile);
        for (uint16_t i = 0; i < tile.get_capacity(); ++i) {
            if (!tile.is_valid_element(i)) continue;
            auto const coordinate = tile.get_multidimensional_index(i);
            result_t value = tile[i];
            if constexpr (sizeof(stage_t) == 1) value += bias[coordinate[0]];
            scores[coordinate[1] * 32 + coordinate[0]] = float(value);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
};

/** The @c apple10 attention of one dtype, staged in @p stage_type_: tiles accumulated in registers
 *  up to depth 128, in the output rows past it. */
template <typename stage_type_, typename element_type_>
inline void nk_attention_tiles_apple10_(device uchar const *queries, device uchar const *packed, device uchar *output,
                                        device uint const *query_offsets, device float *log_sum_exp,
                                        constant nk_attention_arguments_metal_t &a, uint2 group, uint2 groups,
                                        uint thread_index, uint warp, threadgroup stage_type_ *left,
                                        threadgroup stage_type_ *right, threadgroup float *scores,
                                        threadgroup float *maximum, threadgroup float *corrections,
                                        threadgroup float *sums) {
    if (a.depth <= 128)
        nk_attention_matrix_metal_<stage_type_, nk_attention_operations_apple10_t, element_type_,
                                   nk_attention_accumulate_registers_metal_k>(
            queries, packed, output, query_offsets, log_sum_exp, a, group, groups, thread_index, warp, left, right,
            scores, maximum, corrections, sums);
    else
        nk_attention_matrix_metal_<stage_type_, nk_attention_operations_apple10_t, element_type_,
                                   nk_attention_accumulate_device_metal_k>(
            queries, packed, output, query_offsets, log_sum_exp, a, group, groups, thread_index, warp, left, right,
            scores, maximum, corrections, sums);
}

kernel void nk_attention_packed_bf16_apple10_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_arguments_metal_t &a [[buffer(4)]], device float *log_sum_exp [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 groups [[threadgroups_per_grid]],
    uint thread_index [[thread_index_in_threadgroup]], uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup bfloat left[32 * 64], right[64 * 32];
    threadgroup float scores[32 * 64], maximum[32], corrections[32], sums[32];
    nk_attention_tiles_apple10_<bfloat, nk::bf16_t>(queries, packed, output, query_offsets, log_sum_exp, a, group,
                                                    groups, thread_index, warp, left, right, scores, maximum,
                                                    corrections, sums);
}

kernel void nk_attention_packed_f16_apple10_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_arguments_metal_t &a [[buffer(4)]], device float *log_sum_exp [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 groups [[threadgroups_per_grid]],
    uint thread_index [[thread_index_in_threadgroup]], uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup half left[32 * 64], right[64 * 32];
    threadgroup float scores[32 * 64], maximum[32], corrections[32], sums[32];
    nk_attention_tiles_apple10_<half, nk::f16_t>(queries, packed, output, query_offsets, log_sum_exp, a, group, groups,
                                                 thread_index, warp, left, right, scores, maximum, corrections, sums);
}

kernel void nk_attention_packed_e4m3_apple10_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_arguments_metal_t &a [[buffer(4)]], device float *log_sum_exp [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 groups [[threadgroups_per_grid]],
    uint thread_index [[thread_index_in_threadgroup]], uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup half left[32 * 64], right[64 * 32];
    threadgroup float scores[32 * 64], maximum[32], corrections[32], sums[32];
    nk_attention_tiles_apple10_<half, nk::e4m3_t>(queries, packed, output, query_offsets, log_sum_exp, a, group, groups,
                                                  thread_index, warp, left, right, scores, maximum, corrections, sums);
}

kernel void nk_attention_packed_i8_apple10_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_arguments_metal_t &a [[buffer(4)]], device float *log_sum_exp [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 groups [[threadgroups_per_grid]],
    uint thread_index [[thread_index_in_threadgroup]], uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup int8_t left[32 * 64], right[64 * 64];
    threadgroup float scores[32 * 64], maximum[32], corrections[32], sums[32];
    nk_attention_tiles_apple10_<int8_t, nk::i8_t>(queries, packed, output, query_offsets, log_sum_exp, a, group, groups,
                                                  thread_index, warp, left, right, scores, maximum, corrections, sums);
}
