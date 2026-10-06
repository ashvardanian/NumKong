/**
 *  @file test/each_rocm.hip
 *  @author Ash Vardanian
 *  @date October 5, 2026
 *  @brief Elementwise operation tests on ROCm.
 */
#include "harness.hpp"

#if NUMKONG_ARCH_ROCM_
#include "each.hpp"
#endif // NUMKONG_ARCH_ROCM_

namespace ashvardanian::numkong::test {

#if NUMKONG_ARCH_ROCM_
void test_each_rocm(error_stats_section_t &check) {
    check.section("Elementwise Operations ROCm", nk_cap_rocm_k);
    check("each_swiglu_f32_rocm", test_swiglu<f32_t, rocm_backend_t>, nk_each_swiglu_f32_rocm);
    check("each_swiglu_bf16_rocm", test_swiglu<bf16_t, rocm_backend_t>, nk_each_swiglu_bf16_rocm);
    check("each_swiglu_e4m3_rocm", test_swiglu<e4m3_t, rocm_backend_t>, nk_each_swiglu_e4m3_rocm);
    check("each_sum_f32_rocm", test_sum<f32_t, rocm_backend_t>, nk_each_sum_f32_rocm);
    check("each_sum_f16_rocm", test_sum<f16_t, rocm_backend_t>, nk_each_sum_f16_rocm);
    check("each_sum_bf16_rocm", test_sum<bf16_t, rocm_backend_t>, nk_each_sum_bf16_rocm);
    check("each_rmsnorm_f32_rocm", test_rmsnorm<f32_t, rocm_backend_t>, nk_each_rmsnorm_f32_rocm);
    check("each_rmsnorm_bf16_rocm", test_rmsnorm<bf16_t, rocm_backend_t>, nk_each_rmsnorm_bf16_rocm);
    check("each_rmsnorm_e4m3_rocm", test_rmsnorm<e4m3_t, rocm_backend_t>, nk_each_rmsnorm_e4m3_rocm);
    check("each_rmscast_bf16_rocm", test_rmsnorm<bf16_t, rocm_backend_t, f32_t>, nk_each_rmscast_bf16_rocm);
    check("each_rmscast_f16_rocm", test_rmsnorm<f16_t, rocm_backend_t, f32_t>, nk_each_rmscast_f16_rocm);
    check("each_rmscast_e4m3_rocm", test_rmsnorm<e4m3_t, rocm_backend_t, f32_t>, nk_each_rmscast_e4m3_rocm);
    check("each_rmscast_e5m2_rocm", test_rmsnorm<e5m2_t, rocm_backend_t, f32_t>, nk_each_rmscast_e5m2_rocm);
    check("each_rmscast_e2m3_rocm", test_rmsnorm<e2m3_t, rocm_backend_t, f32_t>, nk_each_rmscast_e2m3_rocm);
    check("each_rmscast_e3m2_rocm", test_rmsnorm<e3m2_t, rocm_backend_t, f32_t>, nk_each_rmscast_e3m2_rocm);
    check("each_rmscast_f32_rocm", test_rmsnorm<f32_t, rocm_backend_t, f64_t>, nk_each_rmscast_f32_rocm);
    check("each_rmscast_i8_rocm", test_rmsnorm<i8_t, rocm_backend_t, i32_t>, nk_each_rmscast_i8_rocm);
    check("each_rmscast_u8_rocm", test_rmsnorm<u8_t, rocm_backend_t, u32_t>, nk_each_rmscast_u8_rocm);
}

#else  // !NUMKONG_ARCH_ROCM_
void test_each_rocm(error_stats_section_t &) {}
#endif // NUMKONG_ARCH_ROCM_

} // namespace ashvardanian::numkong::test
