/**
 *  @file test/reduce_cuda.cu
 *  @author Ash Vardanian
 *  @date October 5, 2026
 *  @brief Reduction tests on CUDA.
 */
#include "harness.hpp"

#if NUMKONG_ARCH_CUDA_
#include "reduce.hpp"
#endif // NUMKONG_ARCH_CUDA_

namespace ashvardanian::numkong::test {

#if NUMKONG_ARCH_CUDA_
void test_reduce_cuda(error_stats_section_t &check) {
    check.section("Reductions CUDA", nk_cap_cuda_k);
    check("reduce_moments_f32_cuda", test_reduce_moments<f32_t, cuda_backend_t>, nk_reduce_moments_f32_cuda);
    check("reduce_moments_f64_cuda", test_reduce_moments<f64_t, cuda_backend_t>, nk_reduce_moments_f64_cuda);
    check("reduce_moments_i8_cuda", test_reduce_moments<i8_t, cuda_backend_t>, nk_reduce_moments_i8_cuda);
    check("reduce_moments_u8_cuda", test_reduce_moments<u8_t, cuda_backend_t>, nk_reduce_moments_u8_cuda);
    check("reduce_moments_i16_cuda", test_reduce_moments<i16_t, cuda_backend_t>, nk_reduce_moments_i16_cuda);
    check("reduce_moments_u16_cuda", test_reduce_moments<u16_t, cuda_backend_t>, nk_reduce_moments_u16_cuda);
    check("reduce_moments_i32_cuda", test_reduce_moments<i32_t, cuda_backend_t>, nk_reduce_moments_i32_cuda);
    check("reduce_moments_u32_cuda", test_reduce_moments<u32_t, cuda_backend_t>, nk_reduce_moments_u32_cuda);
    check("reduce_moments_i64_cuda", test_reduce_moments<i64_t, cuda_backend_t>, nk_reduce_moments_i64_cuda);
    check("reduce_moments_u64_cuda", test_reduce_moments<u64_t, cuda_backend_t>, nk_reduce_moments_u64_cuda);
    check("reduce_moments_f16_cuda", test_reduce_moments<f16_t, cuda_backend_t>, nk_reduce_moments_f16_cuda);
    check("reduce_moments_bf16_cuda", test_reduce_moments<bf16_t, cuda_backend_t>, nk_reduce_moments_bf16_cuda);
    check("reduce_moments_e4m3_cuda", test_reduce_moments<e4m3_t, cuda_backend_t>, nk_reduce_moments_e4m3_cuda);
    check("reduce_moments_e5m2_cuda", test_reduce_moments<e5m2_t, cuda_backend_t>, nk_reduce_moments_e5m2_cuda);
    check("reduce_moments_e2m3_cuda", test_reduce_moments<e2m3_t, cuda_backend_t>, nk_reduce_moments_e2m3_cuda);
    check("reduce_moments_e3m2_cuda", test_reduce_moments<e3m2_t, cuda_backend_t>, nk_reduce_moments_e3m2_cuda);
    check("reduce_moments_e2m1_cuda", test_reduce_moments<e2m1x2_t, cuda_backend_t>, nk_reduce_moments_e2m1_cuda);
    check("reduce_moments_i4_cuda", test_reduce_moments<i4x2_t, cuda_backend_t>, nk_reduce_moments_i4_cuda);
    check("reduce_moments_u4_cuda", test_reduce_moments<u4x2_t, cuda_backend_t>, nk_reduce_moments_u4_cuda);
    check("reduce_moments_u1_cuda", test_reduce_moments<u1x8_t, cuda_backend_t>, nk_reduce_moments_u1_cuda);
    check("reduce_minmax_f32_cuda", test_reduce_minmax<f32_t, cuda_backend_t>, nk_reduce_minmax_f32_cuda);
    check("reduce_minmax_f64_cuda", test_reduce_minmax<f64_t, cuda_backend_t>, nk_reduce_minmax_f64_cuda);
    check("reduce_minmax_i8_cuda", test_reduce_minmax<i8_t, cuda_backend_t>, nk_reduce_minmax_i8_cuda);
    check("reduce_minmax_u8_cuda", test_reduce_minmax<u8_t, cuda_backend_t>, nk_reduce_minmax_u8_cuda);
    check("reduce_minmax_i16_cuda", test_reduce_minmax<i16_t, cuda_backend_t>, nk_reduce_minmax_i16_cuda);
    check("reduce_minmax_u16_cuda", test_reduce_minmax<u16_t, cuda_backend_t>, nk_reduce_minmax_u16_cuda);
    check("reduce_minmax_i32_cuda", test_reduce_minmax<i32_t, cuda_backend_t>, nk_reduce_minmax_i32_cuda);
    check("reduce_minmax_u32_cuda", test_reduce_minmax<u32_t, cuda_backend_t>, nk_reduce_minmax_u32_cuda);
    check("reduce_minmax_i64_cuda", test_reduce_minmax<i64_t, cuda_backend_t>, nk_reduce_minmax_i64_cuda);
    check("reduce_minmax_u64_cuda", test_reduce_minmax<u64_t, cuda_backend_t>, nk_reduce_minmax_u64_cuda);
    check("reduce_minmax_f16_cuda", test_reduce_minmax<f16_t, cuda_backend_t>, nk_reduce_minmax_f16_cuda);
    check("reduce_minmax_bf16_cuda", test_reduce_minmax<bf16_t, cuda_backend_t>, nk_reduce_minmax_bf16_cuda);
    check("reduce_minmax_e4m3_cuda", test_reduce_minmax<e4m3_t, cuda_backend_t>, nk_reduce_minmax_e4m3_cuda);
    check("reduce_minmax_e5m2_cuda", test_reduce_minmax<e5m2_t, cuda_backend_t>, nk_reduce_minmax_e5m2_cuda);
    check("reduce_minmax_e2m3_cuda", test_reduce_minmax<e2m3_t, cuda_backend_t>, nk_reduce_minmax_e2m3_cuda);
    check("reduce_minmax_e3m2_cuda", test_reduce_minmax<e3m2_t, cuda_backend_t>, nk_reduce_minmax_e3m2_cuda);
    check("reduce_minmax_i4_cuda", test_reduce_minmax<i4x2_t, cuda_backend_t>, nk_reduce_minmax_i4_cuda);
    check("reduce_minmax_u4_cuda", test_reduce_minmax<u4x2_t, cuda_backend_t>, nk_reduce_minmax_u4_cuda);
    check("reduce_minmax_u1_cuda", test_reduce_minmax<u1x8_t, cuda_backend_t>, nk_reduce_minmax_u1_cuda);
}

#else  // !NUMKONG_ARCH_CUDA_
void test_reduce_cuda(error_stats_section_t &) {}
#endif // NUMKONG_ARCH_CUDA_

} // namespace ashvardanian::numkong::test
