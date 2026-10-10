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
 *  operands staged in @c half, or @c bfloat for BF16 and MX, with @c float accumulators. The BF16
 *  gradients split their 32-row tiles alike, each simdgroup summing its rows of a 128-column
 *  gradient slice in sixteen 8 × 8 blocks.
 */

/** The Q · K and P · V tiles of @c nk_attention_matrix_metal_ on @c simdgroup_matrix products. */
struct nk_attention_operations_apple9_t {
    template <typename stage_type_, typename element_type_, typename queries_type_>
    static inline void qk(queries_type_ queries, thread nk_attention_work_metal_t const &work,
                          constant nk_attention_arguments_metal_t &a, ulong head, ulong row_first, ulong key_first,
                          threadgroup stage_type_ *left, threadgroup stage_type_ *right, threadgroup float *scores,
                          threadgroup int const *bases, uint thread_index, uint warp) {
        using stage_t = stage_type_;
        simdgroup_matrix<float, 8, 8> sums[8];
        for (uint column = 0; column < 8; ++column) sums[column] = make_filled_simdgroup_matrix<float, 8, 8>(0);
        for (ulong depth_first = 0; depth_first < a.depth; depth_first += 32) {
            nk_attention_stage_qk_metal_<stage_t, 32, element_type_>(queries, work, a, head, row_first, key_first,
                                                                     depth_first, left, right, bases, thread_index);
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
template <typename stage_type_, typename element_type_, typename queries_type_>
inline void nk_attention_tiles_apple9_(queries_type_ queries, device uchar const *packed, device uchar *output,
                                       device uint const *query_offsets, device float *log_sum_exp,
                                       constant nk_attention_arguments_metal_t &a, uint2 group, uint2 groups,
                                       uint thread_index, uint warp, threadgroup stage_type_ *left,
                                       threadgroup stage_type_ *right, threadgroup float *scores,
                                       threadgroup float *maximum, threadgroup float *corrections,
                                       threadgroup float *sums, threadgroup float *multipliers,
                                       threadgroup int *bases) {
    if (a.depth <= 128)
        nk_attention_matrix_metal_<stage_type_, nk_attention_operations_apple9_t, element_type_,
                                   nk_attention_accumulate_registers_metal_k>(
            queries, packed, output, query_offsets, log_sum_exp, a, group, groups, thread_index, warp, left, right,
            scores, maximum, corrections, sums, multipliers, bases);
    else
        nk_attention_matrix_metal_<stage_type_, nk_attention_operations_apple9_t, element_type_,
                                   nk_attention_accumulate_device_metal_k>(
            queries, packed, output, query_offsets, log_sum_exp, a, group, groups, thread_index, warp, left, right,
            scores, maximum, corrections, sums, multipliers, bases);
}

kernel void nk_attention_packed_bf16_apple9_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_arguments_metal_t &a [[buffer(4)]], device float *log_sum_exp [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 groups [[threadgroups_per_grid]],
    uint thread_index [[thread_index_in_threadgroup]], uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup bfloat left[32 * 64], right[64 * 32];
    threadgroup float scores[32 * 64], maximum[32], corrections[32], sums[32], multipliers[32];
    threadgroup int bases[32];
    nk_attention_tiles_apple9_<bfloat, nk::bf16_t>(queries, packed, output, query_offsets, log_sum_exp, a, group,
                                                   groups, thread_index, warp, left, right, scores, maximum,
                                                   corrections, sums, multipliers, bases);
}

kernel void nk_attention_packed_f16_apple9_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_arguments_metal_t &a [[buffer(4)]], device float *log_sum_exp [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 groups [[threadgroups_per_grid]],
    uint thread_index [[thread_index_in_threadgroup]], uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup half left[32 * 64], right[64 * 32];
    threadgroup float scores[32 * 64], maximum[32], corrections[32], sums[32], multipliers[32];
    threadgroup int bases[32];
    nk_attention_tiles_apple9_<half, nk::f16_t>(queries, packed, output, query_offsets, log_sum_exp, a, group, groups,
                                                thread_index, warp, left, right, scores, maximum, corrections, sums,
                                                multipliers, bases);
}

kernel void nk_attention_packed_e4m3_apple9_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_arguments_metal_t &a [[buffer(4)]], device float *log_sum_exp [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 groups [[threadgroups_per_grid]],
    uint thread_index [[thread_index_in_threadgroup]], uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup half left[32 * 64], right[64 * 32];
    threadgroup float scores[32 * 64], maximum[32], corrections[32], sums[32], multipliers[32];
    threadgroup int bases[32];
    nk_attention_tiles_apple9_<half, nk::e4m3_t>(queries, packed, output, query_offsets, log_sum_exp, a, group, groups,
                                                 thread_index, warp, left, right, scores, maximum, corrections, sums,
                                                 multipliers, bases);
}

kernel void nk_attention_packed_nvfp4_apple9_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_arguments_metal_t &a [[buffer(4)]], device float *log_sum_exp [[buffer(5)]],
    device uchar const *query_scales [[buffer(7)]], device float const *query_tensor_scale [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 groups [[threadgroups_per_grid]],
    uint thread_index [[thread_index_in_threadgroup]], uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup half left[32 * 64], right[64 * 32];
    threadgroup float scores[32 * 64], maximum[32], corrections[32], sums[32], multipliers[32];
    threadgroup int bases[32];
    nk_attention_tiles_apple9_<half, nk::f16_t>(
        nk_attention_scaled_queries_metal_<nk::e2m1x2_t, 16, nk_cross_scale_e4m3_metal_k>(
            queries, query_scales, query_tensor_scale, packed, a),
        packed, output, query_offsets, log_sum_exp, a, group, groups, thread_index, warp, left, right, scores, maximum,
        corrections, sums, multipliers, bases);
}

kernel void nk_attention_packed_mxfp4_apple9_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_arguments_metal_t &a [[buffer(4)]], device float *log_sum_exp [[buffer(5)]],
    device uchar const *query_scales [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]], uint thread_index [[thread_index_in_threadgroup]],
    uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup bfloat left[32 * 64], right[64 * 32];
    threadgroup float scores[32 * 64], maximum[32], corrections[32], sums[32], multipliers[32];
    threadgroup int bases[32];
    nk_attention_tiles_apple9_<bfloat, nk::bf16_t>(
        nk_attention_scaled_queries_metal_<nk::e2m1x2_t, 32, nk_cross_scale_e8m0_metal_k>(queries, query_scales,
                                                                                          nullptr, packed, a),
        packed, output, query_offsets, log_sum_exp, a, group, groups, thread_index, warp, left, right, scores, maximum,
        corrections, sums, multipliers, bases);
}

kernel void nk_attention_packed_mxfp6e2m3_apple9_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_arguments_metal_t &a [[buffer(4)]], device float *log_sum_exp [[buffer(5)]],
    device uchar const *query_scales [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]], uint thread_index [[thread_index_in_threadgroup]],
    uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup bfloat left[32 * 64], right[64 * 32];
    threadgroup float scores[32 * 64], maximum[32], corrections[32], sums[32], multipliers[32];
    threadgroup int bases[32];
    nk_attention_tiles_apple9_<bfloat, nk::bf16_t>(
        nk_attention_scaled_queries_metal_<nk::e2m3_t, 32, nk_cross_scale_e8m0_metal_k>(queries, query_scales, nullptr,
                                                                                        packed, a),
        packed, output, query_offsets, log_sum_exp, a, group, groups, thread_index, warp, left, right, scores, maximum,
        corrections, sums, multipliers, bases);
}

kernel void nk_attention_packed_mxfp6e3m2_apple9_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_arguments_metal_t &a [[buffer(4)]], device float *log_sum_exp [[buffer(5)]],
    device uchar const *query_scales [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]], uint thread_index [[thread_index_in_threadgroup]],
    uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup bfloat left[32 * 64], right[64 * 32];
    threadgroup float scores[32 * 64], maximum[32], corrections[32], sums[32], multipliers[32];
    threadgroup int bases[32];
    nk_attention_tiles_apple9_<bfloat, nk::bf16_t>(
        nk_attention_scaled_queries_metal_<nk::e3m2_t, 32, nk_cross_scale_e8m0_metal_k>(queries, query_scales, nullptr,
                                                                                        packed, a),
        packed, output, query_offsets, log_sum_exp, a, group, groups, thread_index, warp, left, right, scores, maximum,
        corrections, sums, multipliers, bases);
}

kernel void nk_attention_packed_mxfp8e4m3_apple9_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_arguments_metal_t &a [[buffer(4)]], device float *log_sum_exp [[buffer(5)]],
    device uchar const *query_scales [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]], uint thread_index [[thread_index_in_threadgroup]],
    uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup bfloat left[32 * 64], right[64 * 32];
    threadgroup float scores[32 * 64], maximum[32], corrections[32], sums[32], multipliers[32];
    threadgroup int bases[32];
    nk_attention_tiles_apple9_<bfloat, nk::bf16_t>(
        nk_attention_scaled_queries_metal_<nk::e4m3_t, 32, nk_cross_scale_e8m0_metal_k>(queries, query_scales, nullptr,
                                                                                        packed, a),
        packed, output, query_offsets, log_sum_exp, a, group, groups, thread_index, warp, left, right, scores, maximum,
        corrections, sums, multipliers, bases);
}

kernel void nk_attention_packed_mxfp8e5m2_apple9_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_arguments_metal_t &a [[buffer(4)]], device float *log_sum_exp [[buffer(5)]],
    device uchar const *query_scales [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]], uint thread_index [[thread_index_in_threadgroup]],
    uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup bfloat left[32 * 64], right[64 * 32];
    threadgroup float scores[32 * 64], maximum[32], corrections[32], sums[32], multipliers[32];
    threadgroup int bases[32];
    nk_attention_tiles_apple9_<bfloat, nk::bf16_t>(
        nk_attention_scaled_queries_metal_<nk::e5m2_t, 32, nk_cross_scale_e8m0_metal_k>(queries, query_scales, nullptr,
                                                                                        packed, a),
        packed, output, query_offsets, log_sum_exp, a, group, groups, thread_index, warp, left, right, scores, maximum,
        corrections, sums, multipliers, bases);
}

/** The S, dP and gradient products of @c nk_attention_backward_keys_matrix_metal_ and
 *  @c nk_attention_backward_queries_matrix_metal_ on @c simdgroup_matrix products, each of four
 *  simdgroups owning eight of a tile's 32 rows. */
struct nk_attention_backward_operations_apple9_t {

    /** A simdgroup's eight rows by 128 gradient columns. */
    struct gradient_t {
        simdgroup_matrix<float, 8, 8> blocks[16];
    };

    static inline void zero(thread gradient_t &gradient) {
        for (uint column = 0; column < 16; ++column)
            gradient.blocks[column] = make_filled_simdgroup_matrix<float, 8, 8>(0);
    }

    static inline void scores(device uchar const *queries, device uchar const *output_gradient,
                              thread nk_attention_work_metal_t const &work, constant nk_attention_arguments_metal_t &a,
                              ulong key_value_head, ulong first, ulong key_first, threadgroup bfloat *stage,
                              threadgroup float *scores, threadgroup float *weight_gradients, uint thread_index,
                              uint warp) {
        simdgroup_matrix<float, 8, 8> score_sums[4], gradient_sums[4];
        for (uint column = 0; column < 4; ++column)
            score_sums[column] = make_filled_simdgroup_matrix<float, 8, 8>(0),
            gradient_sums[column] = make_filled_simdgroup_matrix<float, 8, 8>(0);
        for (ulong depth_first = 0; depth_first < a.depth; depth_first += 32) {
            nk_attention_backward_stage_rows_metal_(queries, output_gradient, work, a, key_value_head, first,
                                                    depth_first, 32, stage, stage + 1024, thread_index);
            nk_attention_backward_stage_plane_metal_(work.keys, work, a, key_first, depth_first, 32, stage + 2048,
                                                     thread_index);
            nk_attention_backward_stage_plane_metal_(work.values, work, a, key_first, depth_first, 32, stage + 3072,
                                                     thread_index);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint depth = 0; depth < 32; depth += 8) {
                simdgroup_matrix<bfloat, 8, 8> query, gradient;
                simdgroup_load(query, stage + warp * 8 * 32 + depth, 32);
                simdgroup_load(gradient, stage + 1024 + warp * 8 * 32 + depth, 32);
                for (uint column = 0; column < 4; ++column) {
                    simdgroup_matrix<bfloat, 8, 8> key, value;
                    simdgroup_load(key, stage + 2048 + column * 8 * 32 + depth, 32, ulong2(0), true);
                    simdgroup_load(value, stage + 3072 + column * 8 * 32 + depth, 32, ulong2(0), true);
                    simdgroup_multiply_accumulate(score_sums[column], query, key, score_sums[column]);
                    simdgroup_multiply_accumulate(gradient_sums[column], gradient, value, gradient_sums[column]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint column = 0; column < 4; ++column) {
            simdgroup_store(score_sums[column], scores + warp * 8 * 32 + column * 8, 32);
            simdgroup_store(gradient_sums[column], weight_gradients + warp * 8 * 32 + column * 8, 32);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    static inline void accumulate(thread gradient_t &gradient, threadgroup bfloat *weights, threadgroup bfloat *values,
                                  uint warp) {
        for (uint depth = 0; depth < 32; depth += 8) {
            simdgroup_matrix<bfloat, 8, 8> weight;
            simdgroup_load(weight, weights + warp * 8 * 32 + depth, 32);
            for (uint column = 0; column < 16; ++column) {
                simdgroup_matrix<bfloat, 8, 8> value;
                simdgroup_load(value, values + depth * 128 + column * 8, 128);
                simdgroup_multiply_accumulate(gradient.blocks[column], weight, value, gradient.blocks[column]);
            }
        }
    }

    static inline void store(thread gradient_t &gradient, threadgroup float *tile, [[maybe_unused]] uint thread_index,
                             uint warp) {
        for (uint column = 0; column < 16; ++column)
            simdgroup_store(gradient.blocks[column], tile + warp * 8 * 128 + column * 8, 128);
    }
};

kernel void nk_attention_backward_keys_bf16_apple9_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar const *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_backward_arguments_metal_t &b [[buffer(4)]], device float const *log_sum_exp [[buffer(5)]],
    device uchar const *output_gradient [[buffer(6)]], device uchar *key_gradient [[buffer(8)]],
    device uchar *value_gradient [[buffer(9)]], uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]], uint thread_index [[thread_index_in_threadgroup]],
    uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup float stage[32 * 128], scores[32 * 32], weight_gradients[32 * 32];
    threadgroup bfloat weights[32 * 32], score_gradients[32 * 32];
    threadgroup nk_attention_backward_panel_metal_t panel;
    nk_attention_backward_keys_matrix_metal_<nk_attention_backward_operations_apple9_t>(
        queries, packed, output, query_offsets, b, log_sum_exp, output_gradient, key_gradient, value_gradient, group,
        groups, thread_index, warp, (threadgroup bfloat *)stage, scores, weight_gradients, weights, score_gradients,
        panel);
}

kernel void nk_attention_backward_queries_bf16_apple9_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar const *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_backward_arguments_metal_t &b [[buffer(4)]], device float const *log_sum_exp [[buffer(5)]],
    device uchar const *output_gradient [[buffer(6)]], device uchar *query_gradient [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 groups [[threadgroups_per_grid]],
    uint thread_index [[thread_index_in_threadgroup]], uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup float stage[32 * 128], scores[32 * 32], weight_gradients[32 * 32];
    threadgroup bfloat weights[32 * 32], score_gradients[32 * 32];
    threadgroup nk_attention_backward_panel_metal_t panel;
    nk_attention_backward_queries_matrix_metal_<nk_attention_backward_operations_apple9_t>(
        queries, packed, output, query_offsets, b, log_sum_exp, output_gradient, query_gradient, group, groups,
        thread_index, warp, (threadgroup bfloat *)stage, scores, weight_gradients, weights, score_gradients, panel);
}
