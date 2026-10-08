/**
 *  @file include/numkong/spatials/apple10.metal
 *  @author Ash Vardanian
 *  @date October 5, 2026
 *  @brief Batched spatial distances using Apple family 10 matrix operations.
 *
 *  @sa include/numkong/spatials.h
 */
kernel void nk_angulars_i8_apple10_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                           device float *c [[buffer(2)]],
                                           constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                           uint2 group [[threadgroup_position_in_grid]],
                                           uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::i8_t>(a, b, arguments, group, thread_index, norms);
    nk_cross_tile_apple10_<int8_t, int, nk_cross_angular_metal_k, float>(
        (device int8_t const *)a, (device int8_t const *)b, c, arguments, group, norms);
}

kernel void nk_angulars_u8_apple10_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                           device float *c [[buffer(2)]],
                                           constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                           uint2 group [[threadgroup_position_in_grid]],
                                           uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::u8_t>(a, b, arguments, group, thread_index, norms);
    nk_cross_tile_apple10_<uchar, int, nk_cross_angular_metal_k, float>(
        (device uchar const *)a, (device uchar const *)b, c, arguments, group, norms);
}

kernel void nk_angulars_f16_apple10_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                            device float *c [[buffer(2)]],
                                            constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                            uint2 group [[threadgroup_position_in_grid]],
                                            uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::f16_t>(a, b, arguments, group, thread_index, norms);
    nk_cross_tile_apple10_<half, float, nk_cross_angular_metal_k, float>((device half const *)a, (device half const *)b,
                                                                         c, arguments, group, norms);
}

kernel void nk_angulars_bf16_apple10_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                             device float *c [[buffer(2)]],
                                             constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                             uint2 group [[threadgroup_position_in_grid]],
                                             uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::bf16_t>(a, b, arguments, group, thread_index, norms);
    nk_cross_tile_apple10_<bfloat, float, nk_cross_angular_metal_k, float>(
        (device bfloat const *)a, (device bfloat const *)b, c, arguments, group, norms);
}

kernel void nk_angulars_e4m3_apple10_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                             device float *c [[buffer(2)]],
                                             constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                             uint2 group [[threadgroup_position_in_grid]],
                                             uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::e4m3_t>(a, b, arguments, group, thread_index, norms);
    alignas(16) threadgroup half a_stage[nk_cross_tile_apple10_k][nk_cross_tile_apple10_k];
    alignas(16) threadgroup half b_stage[nk_cross_tile_apple10_k][nk_cross_tile_apple10_k];
    nk_cross_widened_tile_apple10_<nk::e4m3_t, nk_cross_angular_metal_k>(a, b, c, arguments, group, thread_index,
                                                                         a_stage, b_stage, norms);
}

kernel void nk_angulars_e5m2_apple10_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                             device float *c [[buffer(2)]],
                                             constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                             uint2 group [[threadgroup_position_in_grid]],
                                             uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::e5m2_t>(a, b, arguments, group, thread_index, norms);
    alignas(16) threadgroup half a_stage[nk_cross_tile_apple10_k][nk_cross_tile_apple10_k];
    alignas(16) threadgroup half b_stage[nk_cross_tile_apple10_k][nk_cross_tile_apple10_k];
    nk_cross_widened_tile_apple10_<nk::e5m2_t, nk_cross_angular_metal_k>(a, b, c, arguments, group, thread_index,
                                                                         a_stage, b_stage, norms);
}

kernel void nk_angulars_e3m2_apple10_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                             device float *c [[buffer(2)]],
                                             constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                             uint2 group [[threadgroup_position_in_grid]],
                                             uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::e3m2_t>(a, b, arguments, group, thread_index, norms);
    alignas(16) threadgroup half a_stage[nk_cross_tile_apple10_k][nk_cross_tile_apple10_k];
    alignas(16) threadgroup half b_stage[nk_cross_tile_apple10_k][nk_cross_tile_apple10_k];
    nk_cross_widened_tile_apple10_<nk::e3m2_t, nk_cross_angular_metal_k>(a, b, c, arguments, group, thread_index,
                                                                         a_stage, b_stage, norms);
}

kernel void nk_angulars_e2m3_apple10_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                             device float *c [[buffer(2)]],
                                             constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                             uint2 group [[threadgroup_position_in_grid]],
                                             uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::e2m3_t>(a, b, arguments, group, thread_index, norms);
    alignas(16) threadgroup half a_stage[nk_cross_tile_apple10_k][nk_cross_tile_apple10_k];
    alignas(16) threadgroup half b_stage[nk_cross_tile_apple10_k][nk_cross_tile_apple10_k];
    nk_cross_widened_tile_apple10_<nk::e2m3_t, nk_cross_angular_metal_k>(a, b, c, arguments, group, thread_index,
                                                                         a_stage, b_stage, norms);
}

kernel void nk_angulars_e2m1_apple10_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                             device float *c [[buffer(2)]],
                                             constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                             uint2 group [[threadgroup_position_in_grid]],
                                             uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::e2m1x2_t>(a, b, arguments, group, thread_index, norms);
    alignas(16) threadgroup half a_stage[nk_cross_tile_apple10_k][nk_cross_tile_apple10_k];
    alignas(16) threadgroup half b_stage[nk_cross_tile_apple10_k][nk_cross_tile_apple10_k];
    nk_cross_widened_tile_apple10_<nk::e2m1x2_t, nk_cross_angular_metal_k>(a, b, c, arguments, group, thread_index,
                                                                           a_stage, b_stage, norms);
}

kernel void nk_euclideans_i8_apple10_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                             device float *c [[buffer(2)]],
                                             constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                             uint2 group [[threadgroup_position_in_grid]],
                                             uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::i8_t>(a, b, arguments, group, thread_index, norms);
    nk_cross_tile_apple10_<int8_t, int, nk_cross_euclidean_metal_k, float>(
        (device int8_t const *)a, (device int8_t const *)b, c, arguments, group, norms);
}

kernel void nk_euclideans_u8_apple10_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                             device float *c [[buffer(2)]],
                                             constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                             uint2 group [[threadgroup_position_in_grid]],
                                             uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::u8_t>(a, b, arguments, group, thread_index, norms);
    nk_cross_tile_apple10_<uchar, int, nk_cross_euclidean_metal_k, float>(
        (device uchar const *)a, (device uchar const *)b, c, arguments, group, norms);
}

kernel void nk_euclideans_f16_apple10_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                              device float *c [[buffer(2)]],
                                              constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                              uint2 group [[threadgroup_position_in_grid]],
                                              uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::f16_t>(a, b, arguments, group, thread_index, norms);
    nk_cross_tile_apple10_<half, float, nk_cross_euclidean_metal_k, float>(
        (device half const *)a, (device half const *)b, c, arguments, group, norms);
}

kernel void nk_euclideans_bf16_apple10_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                               device float *c [[buffer(2)]],
                                               constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                               uint2 group [[threadgroup_position_in_grid]],
                                               uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::bf16_t>(a, b, arguments, group, thread_index, norms);
    nk_cross_tile_apple10_<bfloat, float, nk_cross_euclidean_metal_k, float>(
        (device bfloat const *)a, (device bfloat const *)b, c, arguments, group, norms);
}

kernel void nk_euclideans_e4m3_apple10_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                               device float *c [[buffer(2)]],
                                               constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                               uint2 group [[threadgroup_position_in_grid]],
                                               uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::e4m3_t>(a, b, arguments, group, thread_index, norms);
    alignas(16) threadgroup half a_stage[nk_cross_tile_apple10_k][nk_cross_tile_apple10_k];
    alignas(16) threadgroup half b_stage[nk_cross_tile_apple10_k][nk_cross_tile_apple10_k];
    nk_cross_widened_tile_apple10_<nk::e4m3_t, nk_cross_euclidean_metal_k>(a, b, c, arguments, group, thread_index,
                                                                           a_stage, b_stage, norms);
}

kernel void nk_euclideans_e5m2_apple10_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                               device float *c [[buffer(2)]],
                                               constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                               uint2 group [[threadgroup_position_in_grid]],
                                               uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::e5m2_t>(a, b, arguments, group, thread_index, norms);
    alignas(16) threadgroup half a_stage[nk_cross_tile_apple10_k][nk_cross_tile_apple10_k];
    alignas(16) threadgroup half b_stage[nk_cross_tile_apple10_k][nk_cross_tile_apple10_k];
    nk_cross_widened_tile_apple10_<nk::e5m2_t, nk_cross_euclidean_metal_k>(a, b, c, arguments, group, thread_index,
                                                                           a_stage, b_stage, norms);
}

kernel void nk_euclideans_e3m2_apple10_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                               device float *c [[buffer(2)]],
                                               constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                               uint2 group [[threadgroup_position_in_grid]],
                                               uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::e3m2_t>(a, b, arguments, group, thread_index, norms);
    alignas(16) threadgroup half a_stage[nk_cross_tile_apple10_k][nk_cross_tile_apple10_k];
    alignas(16) threadgroup half b_stage[nk_cross_tile_apple10_k][nk_cross_tile_apple10_k];
    nk_cross_widened_tile_apple10_<nk::e3m2_t, nk_cross_euclidean_metal_k>(a, b, c, arguments, group, thread_index,
                                                                           a_stage, b_stage, norms);
}

kernel void nk_euclideans_e2m3_apple10_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                               device float *c [[buffer(2)]],
                                               constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                               uint2 group [[threadgroup_position_in_grid]],
                                               uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::e2m3_t>(a, b, arguments, group, thread_index, norms);
    alignas(16) threadgroup half a_stage[nk_cross_tile_apple10_k][nk_cross_tile_apple10_k];
    alignas(16) threadgroup half b_stage[nk_cross_tile_apple10_k][nk_cross_tile_apple10_k];
    nk_cross_widened_tile_apple10_<nk::e2m3_t, nk_cross_euclidean_metal_k>(a, b, c, arguments, group, thread_index,
                                                                           a_stage, b_stage, norms);
}

kernel void nk_euclideans_e2m1_apple10_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                               device float *c [[buffer(2)]],
                                               constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                               uint2 group [[threadgroup_position_in_grid]],
                                               uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float norms[2][nk_cross_tile_metal_k];
    nk_cross_norms_metal_<nk::e2m1x2_t>(a, b, arguments, group, thread_index, norms);
    alignas(16) threadgroup half a_stage[nk_cross_tile_apple10_k][nk_cross_tile_apple10_k];
    alignas(16) threadgroup half b_stage[nk_cross_tile_apple10_k][nk_cross_tile_apple10_k];
    nk_cross_widened_tile_apple10_<nk::e2m1x2_t, nk_cross_euclidean_metal_k>(a, b, c, arguments, group, thread_index,
                                                                             a_stage, b_stage, norms);
}

kernel void nk_angulars_mxfp8e4m3_apple10_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup float norms[2][nk_cross_scaled_tile_apple10_k];
    nk_cross_scaled_norms_metal_<nk::e4m3_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_scaled_tile_apple10_k>(
        a, b, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, norms);
    threadgroup half a_stage[nk_cross_scaled_tile_apple10_k][32], b_stage[nk_cross_scaled_tile_apple10_k][32];
    nk_cross_scaled_tile_apple10_<nk::e4m3_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_angular_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, a_stage, b_stage, norms);
}

kernel void nk_angulars_mxfp8e5m2_apple10_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup float norms[2][nk_cross_scaled_tile_apple10_k];
    nk_cross_scaled_norms_metal_<nk::e5m2_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_scaled_tile_apple10_k>(
        a, b, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, norms);
    threadgroup half a_stage[nk_cross_scaled_tile_apple10_k][32], b_stage[nk_cross_scaled_tile_apple10_k][32];
    nk_cross_scaled_tile_apple10_<nk::e5m2_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_angular_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, a_stage, b_stage, norms);
}

kernel void nk_angulars_mxfp6e2m3_apple10_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup float norms[2][nk_cross_scaled_tile_apple10_k];
    nk_cross_scaled_norms_metal_<nk::e2m3_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_scaled_tile_apple10_k>(
        a, b, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, norms);
    threadgroup half a_stage[nk_cross_scaled_tile_apple10_k][32], b_stage[nk_cross_scaled_tile_apple10_k][32];
    nk_cross_scaled_tile_apple10_<nk::e2m3_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_angular_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, a_stage, b_stage, norms);
}

kernel void nk_angulars_mxfp6e3m2_apple10_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup float norms[2][nk_cross_scaled_tile_apple10_k];
    nk_cross_scaled_norms_metal_<nk::e3m2_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_scaled_tile_apple10_k>(
        a, b, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, norms);
    threadgroup half a_stage[nk_cross_scaled_tile_apple10_k][32], b_stage[nk_cross_scaled_tile_apple10_k][32];
    nk_cross_scaled_tile_apple10_<nk::e3m2_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_angular_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, a_stage, b_stage, norms);
}

kernel void nk_angulars_mxfp4_apple10_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup float norms[2][nk_cross_scaled_tile_apple10_k];
    nk_cross_scaled_norms_metal_<nk::e2m1x2_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_scaled_tile_apple10_k>(
        a, b, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, norms);
    threadgroup half a_stage[nk_cross_scaled_tile_apple10_k][32], b_stage[nk_cross_scaled_tile_apple10_k][32];
    nk_cross_scaled_tile_apple10_<nk::e2m1x2_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_angular_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, a_stage, b_stage, norms);
}

kernel void nk_angulars_nvfp4_apple10_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup float norms[2][nk_cross_scaled_tile_apple10_k];
    nk_cross_scaled_norms_metal_<nk::e2m1x2_t, 16, nk_cross_scale_e4m3_metal_k, nk_cross_scaled_tile_apple10_k>(
        a, b, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, norms);
    threadgroup half a_stage[nk_cross_scaled_tile_apple10_k][16], b_stage[nk_cross_scaled_tile_apple10_k][16];
    nk_cross_scaled_tile_apple10_<nk::e2m1x2_t, 16, nk_cross_scale_e4m3_metal_k, nk_cross_angular_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, a_stage, b_stage, norms);
}

kernel void nk_euclideans_mxfp8e4m3_apple10_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup float norms[2][nk_cross_scaled_tile_apple10_k];
    nk_cross_scaled_norms_metal_<nk::e4m3_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_scaled_tile_apple10_k>(
        a, b, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, norms);
    threadgroup half a_stage[nk_cross_scaled_tile_apple10_k][32], b_stage[nk_cross_scaled_tile_apple10_k][32];
    nk_cross_scaled_tile_apple10_<nk::e4m3_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_euclidean_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, a_stage, b_stage, norms);
}

kernel void nk_euclideans_mxfp8e5m2_apple10_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup float norms[2][nk_cross_scaled_tile_apple10_k];
    nk_cross_scaled_norms_metal_<nk::e5m2_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_scaled_tile_apple10_k>(
        a, b, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, norms);
    threadgroup half a_stage[nk_cross_scaled_tile_apple10_k][32], b_stage[nk_cross_scaled_tile_apple10_k][32];
    nk_cross_scaled_tile_apple10_<nk::e5m2_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_euclidean_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, a_stage, b_stage, norms);
}

kernel void nk_euclideans_mxfp6e2m3_apple10_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup float norms[2][nk_cross_scaled_tile_apple10_k];
    nk_cross_scaled_norms_metal_<nk::e2m3_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_scaled_tile_apple10_k>(
        a, b, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, norms);
    threadgroup half a_stage[nk_cross_scaled_tile_apple10_k][32], b_stage[nk_cross_scaled_tile_apple10_k][32];
    nk_cross_scaled_tile_apple10_<nk::e2m3_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_euclidean_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, a_stage, b_stage, norms);
}

kernel void nk_euclideans_mxfp6e3m2_apple10_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup float norms[2][nk_cross_scaled_tile_apple10_k];
    nk_cross_scaled_norms_metal_<nk::e3m2_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_scaled_tile_apple10_k>(
        a, b, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, norms);
    threadgroup half a_stage[nk_cross_scaled_tile_apple10_k][32], b_stage[nk_cross_scaled_tile_apple10_k][32];
    nk_cross_scaled_tile_apple10_<nk::e3m2_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_euclidean_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, a_stage, b_stage, norms);
}

kernel void nk_euclideans_mxfp4_apple10_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup float norms[2][nk_cross_scaled_tile_apple10_k];
    nk_cross_scaled_norms_metal_<nk::e2m1x2_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_scaled_tile_apple10_k>(
        a, b, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, norms);
    threadgroup half a_stage[nk_cross_scaled_tile_apple10_k][32], b_stage[nk_cross_scaled_tile_apple10_k][32];
    nk_cross_scaled_tile_apple10_<nk::e2m1x2_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_euclidean_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, a_stage, b_stage, norms);
}

kernel void nk_euclideans_nvfp4_apple10_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup float norms[2][nk_cross_scaled_tile_apple10_k];
    nk_cross_scaled_norms_metal_<nk::e2m1x2_t, 16, nk_cross_scale_e4m3_metal_k, nk_cross_scaled_tile_apple10_k>(
        a, b, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, norms);
    threadgroup half a_stage[nk_cross_scaled_tile_apple10_k][16], b_stage[nk_cross_scaled_tile_apple10_k][16];
    nk_cross_scaled_tile_apple10_<nk::e2m1x2_t, 16, nk_cross_scale_e4m3_metal_k, nk_cross_euclidean_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, a_stage, b_stage, norms);
}

kernel void nk_angulars_i4_apple10_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                           device float *c [[buffer(2)]],
                                           constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                           uint2 group [[threadgroup_position_in_grid]],
                                           uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup int8_t a_stage[64][64];
    alignas(128) threadgroup nk_i4_storage_apple10_t b_stage[64][nk_cross_int4_pitch_apple10_k];
    threadgroup float norms[2][64];
    nk_cross_norms_metal_<nk::i4x2_t, 64>(a, b, arguments, group, thread_index, norms);
    nk_cross_int4_tile_apple10_<nk::i4x2_t, int8_t, nk_i4_operand_apple10_t, nk_cross_angular_metal_k, 64>(
        a, b, c, arguments, group, thread_index, a_stage, b_stage, norms);
}

kernel void nk_angulars_u4_apple10_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                           device float *c [[buffer(2)]],
                                           constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                           uint2 group [[threadgroup_position_in_grid]],
                                           uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup uchar a_stage[64][64];
    alignas(128) threadgroup nk_u4_storage_apple10_t b_stage[64][nk_cross_int4_pitch_apple10_k];
    threadgroup float norms[2][64];
    nk_cross_norms_metal_<nk::u4x2_t, 64>(a, b, arguments, group, thread_index, norms);
    nk_cross_int4_tile_apple10_<nk::u4x2_t, uchar, nk_u4_operand_apple10_t, nk_cross_angular_metal_k, 64>(
        a, b, c, arguments, group, thread_index, a_stage, b_stage, norms);
}

kernel void nk_euclideans_i4_apple10_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                             device float *c [[buffer(2)]],
                                             constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                             uint2 group [[threadgroup_position_in_grid]],
                                             uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup int8_t a_stage[64][64];
    alignas(128) threadgroup nk_i4_storage_apple10_t b_stage[64][nk_cross_int4_pitch_apple10_k];
    threadgroup float norms[2][64];
    nk_cross_norms_metal_<nk::i4x2_t, 64>(a, b, arguments, group, thread_index, norms);
    nk_cross_int4_tile_apple10_<nk::i4x2_t, int8_t, nk_i4_operand_apple10_t, nk_cross_euclidean_metal_k, 64>(
        a, b, c, arguments, group, thread_index, a_stage, b_stage, norms);
}

kernel void nk_euclideans_u4_apple10_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                             device float *c [[buffer(2)]],
                                             constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                             uint2 group [[threadgroup_position_in_grid]],
                                             uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup uchar a_stage[64][64];
    alignas(128) threadgroup nk_u4_storage_apple10_t b_stage[64][nk_cross_int4_pitch_apple10_k];
    threadgroup float norms[2][64];
    nk_cross_norms_metal_<nk::u4x2_t, 64>(a, b, arguments, group, thread_index, norms);
    nk_cross_int4_tile_apple10_<nk::u4x2_t, uchar, nk_u4_operand_apple10_t, nk_cross_euclidean_metal_k, 64>(
        a, b, c, arguments, group, thread_index, a_stage, b_stage, norms);
}

kernel void nk_angulars_small_i4_apple10_kernel_(device uchar const *a [[buffer(0)]],
                                                 device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
                                                 constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                                 uint2 group [[threadgroup_position_in_grid]],
                                                 uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup int8_t a_stage[32][64];
    alignas(128) threadgroup nk_i4_storage_apple10_t b_stage[32][nk_cross_int4_pitch_apple10_k];
    threadgroup float norms[2][32];
    nk_cross_norms_metal_<nk::i4x2_t, 32>(a, b, arguments, group, thread_index, norms);
    nk_cross_int4_tile_apple10_<nk::i4x2_t, int8_t, nk_i4_operand_apple10_t, nk_cross_angular_metal_k, 32>(
        a, b, c, arguments, group, thread_index, a_stage, b_stage, norms);
}

kernel void nk_angulars_small_u4_apple10_kernel_(device uchar const *a [[buffer(0)]],
                                                 device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
                                                 constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                                 uint2 group [[threadgroup_position_in_grid]],
                                                 uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup uchar a_stage[32][64];
    alignas(128) threadgroup nk_u4_storage_apple10_t b_stage[32][nk_cross_int4_pitch_apple10_k];
    threadgroup float norms[2][32];
    nk_cross_norms_metal_<nk::u4x2_t, 32>(a, b, arguments, group, thread_index, norms);
    nk_cross_int4_tile_apple10_<nk::u4x2_t, uchar, nk_u4_operand_apple10_t, nk_cross_angular_metal_k, 32>(
        a, b, c, arguments, group, thread_index, a_stage, b_stage, norms);
}

kernel void nk_euclideans_small_i4_apple10_kernel_(device uchar const *a [[buffer(0)]],
                                                   device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
                                                   constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                                   uint2 group [[threadgroup_position_in_grid]],
                                                   uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup int8_t a_stage[32][64];
    alignas(128) threadgroup nk_i4_storage_apple10_t b_stage[32][nk_cross_int4_pitch_apple10_k];
    threadgroup float norms[2][32];
    nk_cross_norms_metal_<nk::i4x2_t, 32>(a, b, arguments, group, thread_index, norms);
    nk_cross_int4_tile_apple10_<nk::i4x2_t, int8_t, nk_i4_operand_apple10_t, nk_cross_euclidean_metal_k, 32>(
        a, b, c, arguments, group, thread_index, a_stage, b_stage, norms);
}

kernel void nk_euclideans_small_u4_apple10_kernel_(device uchar const *a [[buffer(0)]],
                                                   device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
                                                   constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                                   uint2 group [[threadgroup_position_in_grid]],
                                                   uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup uchar a_stage[32][64];
    alignas(128) threadgroup nk_u4_storage_apple10_t b_stage[32][nk_cross_int4_pitch_apple10_k];
    threadgroup float norms[2][32];
    nk_cross_norms_metal_<nk::u4x2_t, 32>(a, b, arguments, group, thread_index, norms);
    nk_cross_int4_tile_apple10_<nk::u4x2_t, uchar, nk_u4_operand_apple10_t, nk_cross_euclidean_metal_k, 32>(
        a, b, c, arguments, group, thread_index, a_stage, b_stage, norms);
}
