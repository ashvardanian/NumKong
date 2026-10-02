/**
 *  @file test/cross_simt.cuh
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Batch operation tests - the dispatch scenario CUDA and ROCm share, over either backend.
 *
 *  Included by one translation unit per binary, `cross_cuda.cu` or `cross_rocm.hip`, which adds the
 *  capability sections of its vendor.
 */
#pragma once
#ifndef NUMKONG_TEST_CROSS_SIMT_CUH
#define NUMKONG_TEST_CROSS_SIMT_CUH

#include "harness.cuh" // `device_capabilities`, `device_best`
#include "cross.hpp"   // `test_dots_packed`, `attention_weights_t`

namespace ashvardanian::numkong::test {

/** The dispatching entry points, over the capabilities of the device @p backend_type_ runs on. */
template <typename backend_type_>
void test_cross_dispatch(error_stats_section_t &check) {
    using runtime_t = typename backend_type_::runtime_t;
    check.section("Cross Dispatch", nk_cap_cuda_k | nk_cap_rocm_k);
#if NUMKONG_HEADER_ONLY
    check("dots_packed_i8_dispatch", [](settings_t const &settings) {
        return test_missing_library<nk_dots_packed_i8_best>(settings, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 0, 0,
                                                            0, nullptr);
    });
#else
    check("dots_packed_i8_dispatch", test_dots_packed<i8_t, backend_type_>,
          device_best<runtime_t, nk_dots_pack_size_i8_best>, device_best<runtime_t, nk_dots_pack_i8_best>,
          device_best<runtime_t, nk_dots_packed_i8_best>);
#endif
}

} // namespace ashvardanian::numkong::test

#endif // NUMKONG_TEST_CROSS_SIMT_CUH
