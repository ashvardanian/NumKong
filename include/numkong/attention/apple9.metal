/**
 *  @file include/numkong/attention/apple9.metal
 *  @author Ash Vardanian
 *  @date October 5, 2026
 *  @brief Tiled ragged attention on the SIMD-group matrices of Apple family 9 GPUs.
 *
 *  @sa include/numkong/attention/apple9.h, which embeds this source after `metal.metal`
 *
 *  Each of four simdgroups owns eight rows of a 32-row tile: Q · K multiplies them into eight 8 × 8
 *  score blocks per 64-key panel, and P · V four 8 × 8 blocks per 32 channels of the values, on
 *  operands staged in @c half, or @c bfloat for BF16, with @c float accumulators.
 */

/** The Q · K and P · V tiles of @c nk_attention_matrix_metal_ on @c simdgroup_matrix products. */
struct nk_attention_operations_apple9_t {
    template <typename stage_type_, typename element_type_>
    static inline void qk(device uchar const *queries, thread nk_attention_work_metal_t const &work,
                          constant nk_attention_arguments_metal_t &a, ulong head, ulong row_first, ulong key_first,
                          threadgroup stage_type_ *left, threadgroup stage_type_ *right, threadgroup float *scores,
                          uint thread_index, uint warp) {
        using stage_t = stage_type_;
        simdgroup_matrix<float, 8, 8> sums[8];
        for (uint column = 0; column < 8; ++column) sums[column] = make_filled_simdgroup_matrix<float, 8, 8>(0);
        for (ulong depth_first = 0; depth_first < a.depth; depth_first += 32) {
            nk_attention_stage_qk_metal_<stage_t, 32, element_type_>(queries, work, a, head, row_first, key_first,
                                                                     depth_first, left, right, thread_index);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint depth = 0; depth < 32; depth += 8) {
                simdgroup_matrix<stage_t, 8, 8> query;
                simdgroup_load(query, left + warp * 8 * 32 + depth, 32);
                for (uint column = 0; column < 8; ++column) {
                    simdgroup_matrix<stage_t, 8, 8> key;
                    simdgroup_load(key, right + column * 8 * 32 + depth, 32, ulong2(0), true);
                    simdgroup_multiply_accumulate(sums[column], query, key, sums[column]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint column = 0; column < 8; ++column)
            simdgroup_store(sums[column], scores + warp * 8 * 64 + column * 8, 64);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    template <typename stage_type_>
    static inline void pv(threadgroup stage_type_ const *left, threadgroup stage_type_ const *right,
                          threadgroup float *scores, [[maybe_unused]] uint thread_index, uint warp) {
        for (uint column = 0; column < 4; ++column) {
            simdgroup_matrix<float, 8, 8> sum = make_filled_simdgroup_matrix<float, 8, 8>(0);
            for (uint key = 0; key < 64; key += 8) {
                simdgroup_matrix<stage_type_, 8, 8> weights, values;
                simdgroup_load(weights, left + warp * 8 * 64 + key, 64);
                simdgroup_load(values, right + key * 32 + column * 8, 32);
                simdgroup_multiply_accumulate(sum, weights, values, sum);
            }
            simdgroup_store(sum, scores + warp * 8 * 32 + column * 8, 32);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
};

/** The @c apple9 attention of one dtype, staged in @p stage_type_: tiles accumulated in registers
 *  up to depth 128, in the output rows past it. */
template <typename stage_type_, typename element_type_>
inline void nk_attention_tiles_apple9_(device uchar const *queries, device uchar const *packed, device uchar *output,
                                       device uint const *query_offsets, device float *log_sum_exp,
                                       constant nk_attention_arguments_metal_t &a, uint2 group, uint2 groups,
                                       uint thread_index, uint warp, threadgroup stage_type_ *left,
                                       threadgroup stage_type_ *right, threadgroup float *scores,
                                       threadgroup float *maximum, threadgroup float *corrections,
                                       threadgroup float *sums) {
    if (a.depth <= 128)
        nk_attention_matrix_metal_<stage_type_, nk_attention_operations_apple9_t, element_type_,
                                   nk_attention_accumulate_registers_metal_k>(
            queries, packed, output, query_offsets, log_sum_exp, a, group, groups, thread_index, warp, left, right,
            scores, maximum, corrections, sums);
    else
        nk_attention_matrix_metal_<stage_type_, nk_attention_operations_apple9_t, element_type_,
                                   nk_attention_accumulate_device_metal_k>(
            queries, packed, output, query_offsets, log_sum_exp, a, group, groups, thread_index, warp, left, right,
            scores, maximum, corrections, sums);
}

kernel void nk_attention_packed_bf16_apple9_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_arguments_metal_t &a [[buffer(4)]], device float *log_sum_exp [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 groups [[threadgroups_per_grid]],
    uint thread_index [[thread_index_in_threadgroup]], uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup bfloat left[32 * 64], right[64 * 32];
    threadgroup float scores[32 * 64], maximum[32], corrections[32], sums[32];
    nk_attention_tiles_apple9_<bfloat, nk::bf16_t>(queries, packed, output, query_offsets, log_sum_exp, a, group,
                                                   groups, thread_index, warp, left, right, scores, maximum,
                                                   corrections, sums);
}

kernel void nk_attention_packed_f16_apple9_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_arguments_metal_t &a [[buffer(4)]], device float *log_sum_exp [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 groups [[threadgroups_per_grid]],
    uint thread_index [[thread_index_in_threadgroup]], uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup half left[32 * 64], right[64 * 32];
    threadgroup float scores[32 * 64], maximum[32], corrections[32], sums[32];
    nk_attention_tiles_apple9_<half, nk::f16_t>(queries, packed, output, query_offsets, log_sum_exp, a, group, groups,
                                                thread_index, warp, left, right, scores, maximum, corrections, sums);
}

kernel void nk_attention_packed_e4m3_apple9_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_arguments_metal_t &a [[buffer(4)]], device float *log_sum_exp [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 groups [[threadgroups_per_grid]],
    uint thread_index [[thread_index_in_threadgroup]], uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup half left[32 * 64], right[64 * 32];
    threadgroup float scores[32 * 64], maximum[32], corrections[32], sums[32];
    nk_attention_tiles_apple9_<half, nk::e4m3_t>(queries, packed, output, query_offsets, log_sum_exp, a, group, groups,
                                                 thread_index, warp, left, right, scores, maximum, corrections, sums);
}
