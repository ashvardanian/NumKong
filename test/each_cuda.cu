/**
 *  @file test/each_cuda.cu
 *  @author Ash Vardanian
 *  @date October 5, 2026
 *  @brief Elementwise operation tests on CUDA.
 */
#include "harness.hpp"

#if NUMKONG_ARCH_CUDA_
#include "each.hpp"
#endif // NUMKONG_ARCH_CUDA_

namespace ashvardanian::numkong::test {

#if NUMKONG_ARCH_CUDA_
void test_each_cuda(error_stats_section_t &check) {
#if NUMKONG_TARGET_CUDA
    check.section("Elementwise Operations CUDA", nk_cap_cuda_k);
    check("each_swiglu_f32_cuda", test_swiglu<f32_t, cuda_backend_t>, nk_each_swiglu_f32_cuda);
    check("each_swiglu_bf16_cuda", test_swiglu<bf16_t, cuda_backend_t>, nk_each_swiglu_bf16_cuda);
    check("each_swiglu_e4m3_cuda", test_swiglu<e4m3_t, cuda_backend_t>, nk_each_swiglu_e4m3_cuda);
    check("each_sum_f32_cuda", test_sum<f32_t, cuda_backend_t>, nk_each_sum_f32_cuda);
    check("each_sum_f16_cuda", test_sum<f16_t, cuda_backend_t>, nk_each_sum_f16_cuda);
    check("each_sum_bf16_cuda", test_sum<bf16_t, cuda_backend_t>, nk_each_sum_bf16_cuda);
    check("each_sum_f64_cuda", test_sum<f64_t, cuda_backend_t>, nk_each_sum_f64_cuda);
    check("each_sum_e4m3_cuda", test_sum<e4m3_t, cuda_backend_t>, nk_each_sum_e4m3_cuda);
    check("each_sum_e5m2_cuda", test_sum<e5m2_t, cuda_backend_t>, nk_each_sum_e5m2_cuda);
    check("each_sum_e2m3_cuda", test_sum<e2m3_t, cuda_backend_t>, nk_each_sum_e2m3_cuda);
    check("each_sum_e3m2_cuda", test_sum<e3m2_t, cuda_backend_t>, nk_each_sum_e3m2_cuda);
    check("each_sum_i8_cuda", test_sum<i8_t, cuda_backend_t>, nk_each_sum_i8_cuda);
    check("each_sum_u8_cuda", test_sum<u8_t, cuda_backend_t>, nk_each_sum_u8_cuda);
    check("each_sum_i16_cuda", test_sum<i16_t, cuda_backend_t>, nk_each_sum_i16_cuda);
    check("each_sum_u16_cuda", test_sum<u16_t, cuda_backend_t>, nk_each_sum_u16_cuda);
    check("each_sum_i32_cuda", test_sum<i32_t, cuda_backend_t>, nk_each_sum_i32_cuda);
    check("each_sum_u32_cuda", test_sum<u32_t, cuda_backend_t>, nk_each_sum_u32_cuda);
    check("each_sum_i64_cuda", test_sum<i64_t, cuda_backend_t>, nk_each_sum_i64_cuda);
    check("each_sum_u64_cuda", test_sum<u64_t, cuda_backend_t>, nk_each_sum_u64_cuda);
    check("each_sum_f32c_cuda", test_sum<f32c_t, cuda_backend_t>, nk_each_sum_f32c_cuda);
    check("each_sum_f64c_cuda", test_sum<f64c_t, cuda_backend_t>, nk_each_sum_f64c_cuda);
    check("each_scale_f64_cuda", test_scale<f64_t, cuda_backend_t>, nk_each_scale_f64_cuda);
    check("each_scale_f32_cuda", test_scale<f32_t, cuda_backend_t>, nk_each_scale_f32_cuda);
    check("each_scale_f16_cuda", test_scale<f16_t, cuda_backend_t>, nk_each_scale_f16_cuda);
    check("each_scale_bf16_cuda", test_scale<bf16_t, cuda_backend_t>, nk_each_scale_bf16_cuda);
    check("each_scale_e4m3_cuda", test_scale<e4m3_t, cuda_backend_t>, nk_each_scale_e4m3_cuda);
    check("each_scale_e5m2_cuda", test_scale<e5m2_t, cuda_backend_t>, nk_each_scale_e5m2_cuda);
    check("each_scale_e2m3_cuda", test_scale<e2m3_t, cuda_backend_t>, nk_each_scale_e2m3_cuda);
    check("each_scale_e3m2_cuda", test_scale<e3m2_t, cuda_backend_t>, nk_each_scale_e3m2_cuda);
    check("each_scale_i8_cuda", test_scale<i8_t, cuda_backend_t>, nk_each_scale_i8_cuda);
    check("each_scale_u8_cuda", test_scale<u8_t, cuda_backend_t>, nk_each_scale_u8_cuda);
    check("each_scale_i16_cuda", test_scale<i16_t, cuda_backend_t>, nk_each_scale_i16_cuda);
    check("each_scale_u16_cuda", test_scale<u16_t, cuda_backend_t>, nk_each_scale_u16_cuda);
    check("each_scale_i32_cuda", test_scale<i32_t, cuda_backend_t>, nk_each_scale_i32_cuda);
    check("each_scale_u32_cuda", test_scale<u32_t, cuda_backend_t>, nk_each_scale_u32_cuda);
    check("each_scale_i64_cuda", test_scale<i64_t, cuda_backend_t>, nk_each_scale_i64_cuda);
    check("each_scale_u64_cuda", test_scale<u64_t, cuda_backend_t>, nk_each_scale_u64_cuda);
    check("each_scale_f32c_cuda", test_scale<f32c_t, cuda_backend_t>, nk_each_scale_f32c_cuda);
    check("each_scale_f64c_cuda", test_scale<f64c_t, cuda_backend_t>, nk_each_scale_f64c_cuda);
    check("each_blend_f64_cuda", test_blend<f64_t, cuda_backend_t>, nk_each_blend_f64_cuda);
    check("each_blend_f32_cuda", test_blend<f32_t, cuda_backend_t>, nk_each_blend_f32_cuda);
    check("each_blend_f16_cuda", test_blend<f16_t, cuda_backend_t>, nk_each_blend_f16_cuda);
    check("each_blend_bf16_cuda", test_blend<bf16_t, cuda_backend_t>, nk_each_blend_bf16_cuda);
    check("each_blend_e4m3_cuda", test_blend<e4m3_t, cuda_backend_t>, nk_each_blend_e4m3_cuda);
    check("each_blend_e5m2_cuda", test_blend<e5m2_t, cuda_backend_t>, nk_each_blend_e5m2_cuda);
    check("each_blend_e2m3_cuda", test_blend<e2m3_t, cuda_backend_t>, nk_each_blend_e2m3_cuda);
    check("each_blend_e3m2_cuda", test_blend<e3m2_t, cuda_backend_t>, nk_each_blend_e3m2_cuda);
    check("each_blend_i8_cuda", test_blend<i8_t, cuda_backend_t>, nk_each_blend_i8_cuda);
    check("each_blend_u8_cuda", test_blend<u8_t, cuda_backend_t>, nk_each_blend_u8_cuda);
    check("each_blend_i16_cuda", test_blend<i16_t, cuda_backend_t>, nk_each_blend_i16_cuda);
    check("each_blend_u16_cuda", test_blend<u16_t, cuda_backend_t>, nk_each_blend_u16_cuda);
    check("each_blend_i32_cuda", test_blend<i32_t, cuda_backend_t>, nk_each_blend_i32_cuda);
    check("each_blend_u32_cuda", test_blend<u32_t, cuda_backend_t>, nk_each_blend_u32_cuda);
    check("each_blend_i64_cuda", test_blend<i64_t, cuda_backend_t>, nk_each_blend_i64_cuda);
    check("each_blend_u64_cuda", test_blend<u64_t, cuda_backend_t>, nk_each_blend_u64_cuda);
    check("each_blend_f32c_cuda", test_blend<f32c_t, cuda_backend_t>, nk_each_blend_f32c_cuda);
    check("each_blend_f64c_cuda", test_blend<f64c_t, cuda_backend_t>, nk_each_blend_f64c_cuda);
    check("each_fma_f64_cuda", test_fma<f64_t, cuda_backend_t>, nk_each_fma_f64_cuda);
    check("each_fma_f32_cuda", test_fma<f32_t, cuda_backend_t>, nk_each_fma_f32_cuda);
    check("each_fma_f16_cuda", test_fma<f16_t, cuda_backend_t>, nk_each_fma_f16_cuda);
    check("each_fma_bf16_cuda", test_fma<bf16_t, cuda_backend_t>, nk_each_fma_bf16_cuda);
    check("each_fma_e4m3_cuda", test_fma<e4m3_t, cuda_backend_t>, nk_each_fma_e4m3_cuda);
    check("each_fma_e5m2_cuda", test_fma<e5m2_t, cuda_backend_t>, nk_each_fma_e5m2_cuda);
    check("each_fma_e2m3_cuda", test_fma<e2m3_t, cuda_backend_t>, nk_each_fma_e2m3_cuda);
    check("each_fma_e3m2_cuda", test_fma<e3m2_t, cuda_backend_t>, nk_each_fma_e3m2_cuda);
    check("each_fma_i8_cuda", test_fma<i8_t, cuda_backend_t>, nk_each_fma_i8_cuda);
    check("each_fma_u8_cuda", test_fma<u8_t, cuda_backend_t>, nk_each_fma_u8_cuda);
    check("each_fma_i16_cuda", test_fma<i16_t, cuda_backend_t>, nk_each_fma_i16_cuda);
    check("each_fma_u16_cuda", test_fma<u16_t, cuda_backend_t>, nk_each_fma_u16_cuda);
    check("each_fma_i32_cuda", test_fma<i32_t, cuda_backend_t>, nk_each_fma_i32_cuda);
    check("each_fma_u32_cuda", test_fma<u32_t, cuda_backend_t>, nk_each_fma_u32_cuda);
    check("each_fma_i64_cuda", test_fma<i64_t, cuda_backend_t>, nk_each_fma_i64_cuda);
    check("each_fma_u64_cuda", test_fma<u64_t, cuda_backend_t>, nk_each_fma_u64_cuda);
    check("each_fma_f32c_cuda", test_fma<f32c_t, cuda_backend_t>, nk_each_fma_f32c_cuda);
    check("each_fma_f64c_cuda", test_fma<f64c_t, cuda_backend_t>, nk_each_fma_f64c_cuda);
    check("each_rmsnorm_f32_cuda", test_rmsnorm<f32_t, cuda_backend_t>, nk_each_rmsnorm_f32_cuda);
    check("each_rmsnorm_bf16_cuda", test_rmsnorm<bf16_t, cuda_backend_t>, nk_each_rmsnorm_bf16_cuda);
    check("each_rmsnorm_e4m3_cuda", test_rmsnorm<e4m3_t, cuda_backend_t>, nk_each_rmsnorm_e4m3_cuda);
    check("each_rmscast_bf16_cuda", test_rmsnorm<bf16_t, cuda_backend_t, f32_t>, nk_each_rmscast_bf16_cuda);
    check("each_rmscast_f16_cuda", test_rmsnorm<f16_t, cuda_backend_t, f32_t>, nk_each_rmscast_f16_cuda);
    check("each_rmscast_e4m3_cuda", test_rmsnorm<e4m3_t, cuda_backend_t, f32_t>, nk_each_rmscast_e4m3_cuda);
    check("each_rmscast_e5m2_cuda", test_rmsnorm<e5m2_t, cuda_backend_t, f32_t>, nk_each_rmscast_e5m2_cuda);
    check("each_rmscast_e2m3_cuda", test_rmsnorm<e2m3_t, cuda_backend_t, f32_t>, nk_each_rmscast_e2m3_cuda);
    check("each_rmscast_e3m2_cuda", test_rmsnorm<e3m2_t, cuda_backend_t, f32_t>, nk_each_rmscast_e3m2_cuda);
    check("each_rmscast_f32_cuda", test_rmsnorm<f32_t, cuda_backend_t, f64_t>, nk_each_rmscast_f32_cuda);
    check("each_rmscast_i8_cuda", test_rmsnorm<i8_t, cuda_backend_t, i32_t>, nk_each_rmscast_i8_cuda);
    check("each_rmscast_u8_cuda", test_rmsnorm<u8_t, cuda_backend_t, u32_t>, nk_each_rmscast_u8_cuda);
#endif // NUMKONG_TARGET_CUDA
#if NUMKONG_TARGET_AMPERE
    check.section("Elementwise Operations Ampere", nk_cap_ampere_k);
    check("each_sum_bf16_ampere", test_sum<bf16_t, cuda_backend_t>, nk_each_sum_bf16_ampere);
    check("each_scale_bf16_ampere", test_scale<bf16_t, cuda_backend_t>, nk_each_scale_bf16_ampere);
    check("each_blend_bf16_ampere", test_blend<bf16_t, cuda_backend_t>, nk_each_blend_bf16_ampere);
    check("each_fma_bf16_ampere", test_fma<bf16_t, cuda_backend_t>, nk_each_fma_bf16_ampere);
    check("each_swiglu_bf16_ampere", test_swiglu<bf16_t, cuda_backend_t>, nk_each_swiglu_bf16_ampere);
    check("each_rmsnorm_bf16_ampere", test_rmsnorm<bf16_t, cuda_backend_t>, nk_each_rmsnorm_bf16_ampere);
    check("each_rmscast_bf16_ampere", test_rmsnorm<bf16_t, cuda_backend_t, f32_t>, nk_each_rmscast_bf16_ampere);
#endif // NUMKONG_TARGET_AMPERE
#if NUMKONG_TARGET_ADA
    check.section("Elementwise Operations Ada", nk_cap_ada_k);
    check("each_sum_e4m3_ada", test_sum<e4m3_t, cuda_backend_t>, nk_each_sum_e4m3_ada);
    check("each_scale_e4m3_ada", test_scale<e4m3_t, cuda_backend_t>, nk_each_scale_e4m3_ada);
    check("each_blend_e4m3_ada", test_blend<e4m3_t, cuda_backend_t>, nk_each_blend_e4m3_ada);
    check("each_fma_e4m3_ada", test_fma<e4m3_t, cuda_backend_t>, nk_each_fma_e4m3_ada);
    check("each_swiglu_e4m3_ada", test_swiglu<e4m3_t, cuda_backend_t>, nk_each_swiglu_e4m3_ada);
    check("each_rmsnorm_e4m3_ada", test_rmsnorm<e4m3_t, cuda_backend_t>, nk_each_rmsnorm_e4m3_ada);
    check("each_rmscast_e4m3_ada", test_rmsnorm<e4m3_t, cuda_backend_t, f32_t>, nk_each_rmscast_e4m3_ada);
    check("each_rmscast_e5m2_ada", test_rmsnorm<e5m2_t, cuda_backend_t, f32_t>, nk_each_rmscast_e5m2_ada);
#endif // NUMKONG_TARGET_ADA
}

#else  // !NUMKONG_ARCH_CUDA_
void test_each_cuda(error_stats_section_t &) {}
#endif // NUMKONG_ARCH_CUDA_

} // namespace ashvardanian::numkong::test
