/**
 *  @file include/numkong/spatials/blackwellrtx.cuh
 *  @author Ash Vardanian
 *  @date September 22, 2026
 *  @brief SIMD-accelerated batched spatial distances for the NVIDIA compute capability 12.x family.
 *
 *  @sa include/numkong/spatials.h
 *  @sa include/numkong/dots/blackwellrtx.cuh
 *
 *  The native Float8, Float6 and Float4 products with the Ampere norm updates, whose widenings read
 *  a Float6 code from its low 6 bits exactly as the MMA does, so both see the same values.
 */
#ifndef NK_SPATIALS_BLACKWELLRTX_CUH
#define NK_SPATIALS_BLACKWELLRTX_CUH

#if NK_TARGET_BLACKWELLRTX

#include "numkong/dots/blackwellrtx.cuh"
#include "numkong/spatials/ampere.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region E5M2

nk_define_cross_cuda_normalized_packed_(angular, e5m2, blackwellrtx, e5m2, e5m2, nk_dots_e5m2_multiply_blackwellrtx_,
                                        nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, nk_cross_norm_f32_k,
                                        nk_e5m2_norm_update_ampere_, /*norm_scale=*/1.0f, /*depth_simd_dimensions=*/64,
                                        /*dimensions_per_value=*/1)
nk_define_cross_cuda_normalized_packed_(euclidean, e5m2, blackwellrtx, e5m2, e5m2, nk_dots_e5m2_multiply_blackwellrtx_,
                                        nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, nk_cross_norm_f32_k,
                                        nk_e5m2_norm_update_ampere_, /*norm_scale=*/1.0f, /*depth_simd_dimensions=*/64,
                                        /*dimensions_per_value=*/1)
nk_define_cross_cuda_normalized_symmetric_(angular, e5m2, blackwellrtx, e5m2, nk_dots_e5m2_multiply_blackwellrtx_,
                                           nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, nk_cross_norm_f32_k,
                                           nk_e5m2_norm_update_ampere_, /*norm_scale=*/1.0f, /*dimensions_per_value=*/1)
nk_define_cross_cuda_normalized_symmetric_(euclidean, e5m2, blackwellrtx, e5m2, nk_dots_e5m2_multiply_blackwellrtx_,
                                           nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, nk_cross_norm_f32_k,
                                           nk_e5m2_norm_update_ampere_, /*norm_scale=*/1.0f, /*dimensions_per_value=*/1)

#pragma endregion E5M2

#pragma region E4M3

nk_define_cross_cuda_normalized_packed_(angular, e4m3, blackwellrtx, e4m3, e4m3, nk_dots_e4m3_multiply_blackwellrtx_,
                                        nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, nk_cross_norm_f32_k,
                                        nk_e4m3_norm_update_ampere_, /*norm_scale=*/65536.0f,
                                        /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_cuda_normalized_packed_(euclidean, e4m3, blackwellrtx, e4m3, e4m3, nk_dots_e4m3_multiply_blackwellrtx_,
                                        nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, nk_cross_norm_f32_k,
                                        nk_e4m3_norm_update_ampere_, /*norm_scale=*/65536.0f,
                                        /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_cuda_normalized_symmetric_(angular, e4m3, blackwellrtx, e4m3, nk_dots_e4m3_multiply_blackwellrtx_,
                                           nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, nk_cross_norm_f32_k,
                                           nk_e4m3_norm_update_ampere_, /*norm_scale=*/65536.0f,
                                           /*dimensions_per_value=*/1)
nk_define_cross_cuda_normalized_symmetric_(euclidean, e4m3, blackwellrtx, e4m3, nk_dots_e4m3_multiply_blackwellrtx_,
                                           nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, nk_cross_norm_f32_k,
                                           nk_e4m3_norm_update_ampere_, /*norm_scale=*/65536.0f,
                                           /*dimensions_per_value=*/1)

#pragma endregion E4M3

#pragma region E3M2

nk_define_cross_cuda_normalized_packed_(angular, e3m2, blackwellrtx, e3m2, e3m2, nk_dots_e3m2_multiply_blackwellrtx_,
                                        nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, nk_cross_norm_f32_k,
                                        nk_e3m2_norm_update_ampere_, /*norm_scale=*/16777216.0f,
                                        /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_cuda_normalized_packed_(euclidean, e3m2, blackwellrtx, e3m2, e3m2, nk_dots_e3m2_multiply_blackwellrtx_,
                                        nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, nk_cross_norm_f32_k,
                                        nk_e3m2_norm_update_ampere_, /*norm_scale=*/16777216.0f,
                                        /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_cuda_normalized_symmetric_(angular, e3m2, blackwellrtx, e3m2, nk_dots_e3m2_multiply_blackwellrtx_,
                                           nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, nk_cross_norm_f32_k,
                                           nk_e3m2_norm_update_ampere_, /*norm_scale=*/16777216.0f,
                                           /*dimensions_per_value=*/1)
nk_define_cross_cuda_normalized_symmetric_(euclidean, e3m2, blackwellrtx, e3m2, nk_dots_e3m2_multiply_blackwellrtx_,
                                           nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, nk_cross_norm_f32_k,
                                           nk_e3m2_norm_update_ampere_, /*norm_scale=*/16777216.0f,
                                           /*dimensions_per_value=*/1)

#pragma endregion E3M2

#pragma region E2M3

nk_define_cross_cuda_normalized_packed_(angular, e2m3, blackwellrtx, e2m3, e2m3, nk_dots_e2m3_multiply_blackwellrtx_,
                                        nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, nk_cross_norm_f32_k,
                                        nk_e2m3_norm_update_ampere_, /*norm_scale=*/0.015625f,
                                        /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_cuda_normalized_packed_(euclidean, e2m3, blackwellrtx, e2m3, e2m3, nk_dots_e2m3_multiply_blackwellrtx_,
                                        nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, nk_cross_norm_f32_k,
                                        nk_e2m3_norm_update_ampere_, /*norm_scale=*/0.015625f,
                                        /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_cuda_normalized_symmetric_(angular, e2m3, blackwellrtx, e2m3, nk_dots_e2m3_multiply_blackwellrtx_,
                                           nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, nk_cross_norm_f32_k,
                                           nk_e2m3_norm_update_ampere_, /*norm_scale=*/0.015625f,
                                           /*dimensions_per_value=*/1)
nk_define_cross_cuda_normalized_symmetric_(euclidean, e2m3, blackwellrtx, e2m3, nk_dots_e2m3_multiply_blackwellrtx_,
                                           nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, nk_cross_norm_f32_k,
                                           nk_e2m3_norm_update_ampere_, /*norm_scale=*/0.015625f,
                                           /*dimensions_per_value=*/1)

#pragma endregion E2M3

#pragma region E2M1

nk_define_cross_cuda_normalized_packed_(angular, e2m1, blackwellrtx, e2m1x2, e2m1x2,
                                        nk_dots_e2m1_multiply_blackwellrtx_, nk_cross_epilogue_f32_k,
                                        /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_e2m1_norm_update_ampere_,
                                        /*norm_scale=*/0.25f, /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/2)
nk_define_cross_cuda_normalized_packed_(euclidean, e2m1, blackwellrtx, e2m1x2, e2m1x2,
                                        nk_dots_e2m1_multiply_blackwellrtx_, nk_cross_epilogue_f32_k,
                                        /*output_scale=*/1.0f, nk_cross_norm_f32_k, nk_e2m1_norm_update_ampere_,
                                        /*norm_scale=*/0.25f, /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/2)
nk_define_cross_cuda_normalized_symmetric_(angular, e2m1, blackwellrtx, e2m1x2, nk_dots_e2m1_multiply_blackwellrtx_,
                                           nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, nk_cross_norm_f32_k,
                                           nk_e2m1_norm_update_ampere_, /*norm_scale=*/0.25f,
                                           /*dimensions_per_value=*/2)
nk_define_cross_cuda_normalized_symmetric_(euclidean, e2m1, blackwellrtx, e2m1x2, nk_dots_e2m1_multiply_blackwellrtx_,
                                           nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, nk_cross_norm_f32_k,
                                           nk_e2m1_norm_update_ampere_, /*norm_scale=*/0.25f,
                                           /*dimensions_per_value=*/2)

#pragma endregion E2M1

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_BLACKWELLRTX
#endif // NK_SPATIALS_BLACKWELLRTX_CUH
