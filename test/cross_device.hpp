/**
 *  @file test/cross_device.hpp
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Shared dispatch checks for CUDA, ROCm, and Metal backends.
 */
#pragma once
#ifndef NUMKONG_TEST_CROSS_DEVICE_HPP
#define NUMKONG_TEST_CROSS_DEVICE_HPP

#include "harness.hpp" // `device_backend`
#include "cross.hpp"   // `test_dots_packed`, `attention_weights_t`

namespace ashvardanian::numkong::test {

/** The dispatching entry points, over the capabilities of the device @p backend_type_ runs on. */
template <typename backend_type_>
void test_cross_dispatch(error_stats_section_t &check) {
    check.section("Cross Dispatch", nk_cap_cuda_k | nk_cap_rocm_k | nk_cap_metal_k);
    check(
        "dots_packed_i8_dispatch", test_dots_packed<i8_t, backend_type_>,
        [capabilities = check.available](auto... arguments) noexcept {
            return call_best<nk_dots_pack_size_i8_best>(capabilities, arguments...);
        },
        [capabilities = check.available](auto... arguments) noexcept {
            return call_best<nk_dots_pack_i8_best>(capabilities, arguments...);
        },
        [capabilities = check.available](auto... arguments) noexcept {
            return call_best<nk_dots_packed_i8_best>(capabilities, arguments...);
        });
}

} // namespace ashvardanian::numkong::test

#endif // NUMKONG_TEST_CROSS_DEVICE_HPP
