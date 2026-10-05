/**
 *  @file include/numkong/spatials/metal.metal
 *  @author Ash Vardanian
 *  @date October 5, 2026
 *  @brief Metal kernels for batched angular and Euclidean distances.
 *
 *  @sa include/numkong/spatials.h
 */
kernel void nk_angulars_i8_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                         device float *c [[buffer(2)]],
                                         constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                         uint2 group [[threadgroup_position_in_grid]],
                                         uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::i8_t>(a, b, arguments, group, thread_index, norms);
    threadgroup nk::i8_t::dot_result_t a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_tile_metal_<nk::i8_t, float, nk_cross_angular_metal_k>(a, b, c, arguments, group, thread_index, a_slab,
                                                                    b_slab, norms);
}

kernel void nk_angulars_u8_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                         device float *c [[buffer(2)]],
                                         constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                         uint2 group [[threadgroup_position_in_grid]],
                                         uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::u8_t>(a, b, arguments, group, thread_index, norms);
    threadgroup nk::u8_t::dot_result_t a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_tile_metal_<nk::u8_t, float, nk_cross_angular_metal_k>(a, b, c, arguments, group, thread_index, a_slab,
                                                                    b_slab, norms);
}

kernel void nk_angulars_i4_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                         device float *c [[buffer(2)]],
                                         constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                         uint2 group [[threadgroup_position_in_grid]],
                                         uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::i4x2_t>(a, b, arguments, group, thread_index, norms);
    threadgroup nk::i4x2_t::dot_result_t a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_tile_metal_<nk::i4x2_t, float, nk_cross_angular_metal_k>(a, b, c, arguments, group, thread_index, a_slab,
                                                                      b_slab, norms);
}

kernel void nk_angulars_u4_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                         device float *c [[buffer(2)]],
                                         constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                         uint2 group [[threadgroup_position_in_grid]],
                                         uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::u4x2_t>(a, b, arguments, group, thread_index, norms);
    threadgroup nk::u4x2_t::dot_result_t a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_tile_metal_<nk::u4x2_t, float, nk_cross_angular_metal_k>(a, b, c, arguments, group, thread_index, a_slab,
                                                                      b_slab, norms);
}

kernel void nk_angulars_f16_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                          device float *c [[buffer(2)]],
                                          constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                          uint2 group [[threadgroup_position_in_grid]],
                                          uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::f16_t>(a, b, arguments, group, thread_index, norms);
    threadgroup nk::f16_t::dot_result_t a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_tile_metal_<nk::f16_t, float, nk_cross_angular_metal_k>(a, b, c, arguments, group, thread_index, a_slab,
                                                                     b_slab, norms);
}

kernel void nk_angulars_bf16_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                           device float *c [[buffer(2)]],
                                           constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                           uint2 group [[threadgroup_position_in_grid]],
                                           uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::bf16_t>(a, b, arguments, group, thread_index, norms);
    threadgroup nk::bf16_t::dot_result_t a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_tile_metal_<nk::bf16_t, float, nk_cross_angular_metal_k>(a, b, c, arguments, group, thread_index, a_slab,
                                                                      b_slab, norms);
}

kernel void nk_angulars_e4m3_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                           device float *c [[buffer(2)]],
                                           constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                           uint2 group [[threadgroup_position_in_grid]],
                                           uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::e4m3_t>(a, b, arguments, group, thread_index, norms);
    threadgroup nk::e4m3_t::dot_result_t a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_tile_metal_<nk::e4m3_t, float, nk_cross_angular_metal_k>(a, b, c, arguments, group, thread_index, a_slab,
                                                                      b_slab, norms);
}

kernel void nk_angulars_e5m2_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                           device float *c [[buffer(2)]],
                                           constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                           uint2 group [[threadgroup_position_in_grid]],
                                           uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::e5m2_t>(a, b, arguments, group, thread_index, norms);
    threadgroup nk::e5m2_t::dot_result_t a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_tile_metal_<nk::e5m2_t, float, nk_cross_angular_metal_k>(a, b, c, arguments, group, thread_index, a_slab,
                                                                      b_slab, norms);
}

kernel void nk_angulars_e3m2_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                           device float *c [[buffer(2)]],
                                           constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                           uint2 group [[threadgroup_position_in_grid]],
                                           uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::e3m2_t>(a, b, arguments, group, thread_index, norms);
    threadgroup nk::e3m2_t::dot_result_t a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_tile_metal_<nk::e3m2_t, float, nk_cross_angular_metal_k>(a, b, c, arguments, group, thread_index, a_slab,
                                                                      b_slab, norms);
}

kernel void nk_angulars_e2m3_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                           device float *c [[buffer(2)]],
                                           constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                           uint2 group [[threadgroup_position_in_grid]],
                                           uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::e2m3_t>(a, b, arguments, group, thread_index, norms);
    threadgroup nk::e2m3_t::dot_result_t a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_tile_metal_<nk::e2m3_t, float, nk_cross_angular_metal_k>(a, b, c, arguments, group, thread_index, a_slab,
                                                                      b_slab, norms);
}

kernel void nk_angulars_e2m1_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                           device float *c [[buffer(2)]],
                                           constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                           uint2 group [[threadgroup_position_in_grid]],
                                           uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::e2m1x2_t>(a, b, arguments, group, thread_index, norms);
    threadgroup nk::e2m1x2_t::dot_result_t a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_tile_metal_<nk::e2m1x2_t, float, nk_cross_angular_metal_k>(a, b, c, arguments, group, thread_index, a_slab,
                                                                        b_slab, norms);
}

kernel void nk_euclideans_i8_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                           device float *c [[buffer(2)]],
                                           constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                           uint2 group [[threadgroup_position_in_grid]],
                                           uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::i8_t>(a, b, arguments, group, thread_index, norms);
    threadgroup nk::i8_t::dot_result_t a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_tile_metal_<nk::i8_t, float, nk_cross_euclidean_metal_k>(a, b, c, arguments, group, thread_index, a_slab,
                                                                      b_slab, norms);
}

kernel void nk_euclideans_u8_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                           device float *c [[buffer(2)]],
                                           constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                           uint2 group [[threadgroup_position_in_grid]],
                                           uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::u8_t>(a, b, arguments, group, thread_index, norms);
    threadgroup nk::u8_t::dot_result_t a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_tile_metal_<nk::u8_t, float, nk_cross_euclidean_metal_k>(a, b, c, arguments, group, thread_index, a_slab,
                                                                      b_slab, norms);
}

kernel void nk_euclideans_i4_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                           device float *c [[buffer(2)]],
                                           constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                           uint2 group [[threadgroup_position_in_grid]],
                                           uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::i4x2_t>(a, b, arguments, group, thread_index, norms);
    threadgroup nk::i4x2_t::dot_result_t a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_tile_metal_<nk::i4x2_t, float, nk_cross_euclidean_metal_k>(a, b, c, arguments, group, thread_index, a_slab,
                                                                        b_slab, norms);
}

kernel void nk_euclideans_u4_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                           device float *c [[buffer(2)]],
                                           constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                           uint2 group [[threadgroup_position_in_grid]],
                                           uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::u4x2_t>(a, b, arguments, group, thread_index, norms);
    threadgroup nk::u4x2_t::dot_result_t a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_tile_metal_<nk::u4x2_t, float, nk_cross_euclidean_metal_k>(a, b, c, arguments, group, thread_index, a_slab,
                                                                        b_slab, norms);
}

kernel void nk_euclideans_f16_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                            device float *c [[buffer(2)]],
                                            constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                            uint2 group [[threadgroup_position_in_grid]],
                                            uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::f16_t>(a, b, arguments, group, thread_index, norms);
    threadgroup nk::f16_t::dot_result_t a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_tile_metal_<nk::f16_t, float, nk_cross_euclidean_metal_k>(a, b, c, arguments, group, thread_index, a_slab,
                                                                       b_slab, norms);
}

kernel void nk_euclideans_bf16_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                             device float *c [[buffer(2)]],
                                             constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                             uint2 group [[threadgroup_position_in_grid]],
                                             uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::bf16_t>(a, b, arguments, group, thread_index, norms);
    threadgroup nk::bf16_t::dot_result_t a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_tile_metal_<nk::bf16_t, float, nk_cross_euclidean_metal_k>(a, b, c, arguments, group, thread_index, a_slab,
                                                                        b_slab, norms);
}

kernel void nk_euclideans_e4m3_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                             device float *c [[buffer(2)]],
                                             constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                             uint2 group [[threadgroup_position_in_grid]],
                                             uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::e4m3_t>(a, b, arguments, group, thread_index, norms);
    threadgroup nk::e4m3_t::dot_result_t a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_tile_metal_<nk::e4m3_t, float, nk_cross_euclidean_metal_k>(a, b, c, arguments, group, thread_index, a_slab,
                                                                        b_slab, norms);
}

kernel void nk_euclideans_e5m2_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                             device float *c [[buffer(2)]],
                                             constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                             uint2 group [[threadgroup_position_in_grid]],
                                             uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::e5m2_t>(a, b, arguments, group, thread_index, norms);
    threadgroup nk::e5m2_t::dot_result_t a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_tile_metal_<nk::e5m2_t, float, nk_cross_euclidean_metal_k>(a, b, c, arguments, group, thread_index, a_slab,
                                                                        b_slab, norms);
}

kernel void nk_euclideans_e3m2_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                             device float *c [[buffer(2)]],
                                             constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                             uint2 group [[threadgroup_position_in_grid]],
                                             uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::e3m2_t>(a, b, arguments, group, thread_index, norms);
    threadgroup nk::e3m2_t::dot_result_t a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_tile_metal_<nk::e3m2_t, float, nk_cross_euclidean_metal_k>(a, b, c, arguments, group, thread_index, a_slab,
                                                                        b_slab, norms);
}

kernel void nk_euclideans_e2m3_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                             device float *c [[buffer(2)]],
                                             constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                             uint2 group [[threadgroup_position_in_grid]],
                                             uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::e2m3_t>(a, b, arguments, group, thread_index, norms);
    threadgroup nk::e2m3_t::dot_result_t a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_tile_metal_<nk::e2m3_t, float, nk_cross_euclidean_metal_k>(a, b, c, arguments, group, thread_index, a_slab,
                                                                        b_slab, norms);
}

kernel void nk_euclideans_e2m1_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                             device float *c [[buffer(2)]],
                                             constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                             uint2 group [[threadgroup_position_in_grid]],
                                             uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::e2m1x2_t>(a, b, arguments, group, thread_index, norms);
    threadgroup nk::e2m1x2_t::dot_result_t a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_tile_metal_<nk::e2m1x2_t, float, nk_cross_euclidean_metal_k>(a, b, c, arguments, group, thread_index,
                                                                          a_slab, b_slab, norms);
}

kernel void nk_angulars_mxfp8e4m3_metal_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_scaled_norms_metal_<nk::e4m3_t, 32, nk_cross_scale_e8m0_metal_k>(
        a, b, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, norms);
    threadgroup float a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_scaled_tile_metal_<nk::e4m3_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_angular_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, a_slab, b_slab, norms);
}

kernel void nk_angulars_mxfp8e5m2_metal_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_scaled_norms_metal_<nk::e5m2_t, 32, nk_cross_scale_e8m0_metal_k>(
        a, b, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, norms);
    threadgroup float a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_scaled_tile_metal_<nk::e5m2_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_angular_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, a_slab, b_slab, norms);
}

kernel void nk_angulars_mxfp6e2m3_metal_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_scaled_norms_metal_<nk::e2m3_t, 32, nk_cross_scale_e8m0_metal_k>(
        a, b, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, norms);
    threadgroup float a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_scaled_tile_metal_<nk::e2m3_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_angular_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, a_slab, b_slab, norms);
}

kernel void nk_angulars_mxfp6e3m2_metal_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_scaled_norms_metal_<nk::e3m2_t, 32, nk_cross_scale_e8m0_metal_k>(
        a, b, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, norms);
    threadgroup float a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_scaled_tile_metal_<nk::e3m2_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_angular_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, a_slab, b_slab, norms);
}

kernel void nk_angulars_mxfp4_metal_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_scaled_norms_metal_<nk::e2m1x2_t, 32, nk_cross_scale_e8m0_metal_k>(
        a, b, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, norms);
    threadgroup float a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_scaled_tile_metal_<nk::e2m1x2_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_angular_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, a_slab, b_slab, norms);
}

kernel void nk_angulars_nvfp4_metal_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_scaled_norms_metal_<nk::e2m1x2_t, 16, nk_cross_scale_e4m3_metal_k>(
        a, b, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, norms);
    threadgroup float a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_scaled_tile_metal_<nk::e2m1x2_t, 16, nk_cross_scale_e4m3_metal_k, nk_cross_angular_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, a_slab, b_slab, norms);
}

kernel void nk_euclideans_mxfp8e4m3_metal_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_scaled_norms_metal_<nk::e4m3_t, 32, nk_cross_scale_e8m0_metal_k>(
        a, b, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, norms);
    threadgroup float a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_scaled_tile_metal_<nk::e4m3_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_euclidean_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, a_slab, b_slab, norms);
}

kernel void nk_euclideans_mxfp8e5m2_metal_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_scaled_norms_metal_<nk::e5m2_t, 32, nk_cross_scale_e8m0_metal_k>(
        a, b, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, norms);
    threadgroup float a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_scaled_tile_metal_<nk::e5m2_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_euclidean_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, a_slab, b_slab, norms);
}

kernel void nk_euclideans_mxfp6e2m3_metal_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_scaled_norms_metal_<nk::e2m3_t, 32, nk_cross_scale_e8m0_metal_k>(
        a, b, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, norms);
    threadgroup float a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_scaled_tile_metal_<nk::e2m3_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_euclidean_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, a_slab, b_slab, norms);
}

kernel void nk_euclideans_mxfp6e3m2_metal_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_scaled_norms_metal_<nk::e3m2_t, 32, nk_cross_scale_e8m0_metal_k>(
        a, b, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, norms);
    threadgroup float a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_scaled_tile_metal_<nk::e3m2_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_euclidean_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, a_slab, b_slab, norms);
}

kernel void nk_euclideans_mxfp4_metal_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_scaled_norms_metal_<nk::e2m1x2_t, 32, nk_cross_scale_e8m0_metal_k>(
        a, b, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, norms);
    threadgroup float a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_scaled_tile_metal_<nk::e2m1x2_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_euclidean_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, a_slab, b_slab, norms);
}

kernel void nk_euclideans_nvfp4_metal_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_scaled_norms_metal_<nk::e2m1x2_t, 16, nk_cross_scale_e4m3_metal_k>(
        a, b, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, norms);
    threadgroup float a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_scaled_tile_metal_<nk::e2m1x2_t, 16, nk_cross_scale_e4m3_metal_k, nk_cross_euclidean_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, a_slab, b_slab, norms);
}
