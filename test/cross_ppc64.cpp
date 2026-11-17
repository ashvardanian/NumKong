/**
 *  @file test/cross_ppc64.cpp
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief Batch operation tests - Power ISA family, VSX and MMA.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include "harness.hpp"
#include "cross.hpp"

namespace ashvardanian::numkong::test {

void run_cross_ppc64_cases() {
#if NUMKONG_TARGET_POWERVSX
    auto const powervsx = nk_cap_powervsx_k;
    run_kernel_case("nk_dots_packed_f64_powervsx", test_dots_packed<f64_t>, powervsx, nk_dots_pack_size_f64_powervsx,
                    nk_dots_pack_f64_powervsx, nk_dots_packed_f64_powervsx);
    run_kernel_case("nk_dots_pack_f64_powervsx",
                    test_dots_pack_layout<f64_t, host_backend_t, nk_dots_pack_size_f64_powervsx,
                                          nk_dots_packed_shape_f64_powervsx, nk_dots_pack_f64_powervsx>,
                    powervsx);
    run_kernel_case("nk_dots_packed_f32_powervsx", test_dots_packed<f32_t>, powervsx, nk_dots_pack_size_f32_powervsx,
                    nk_dots_pack_f32_powervsx, nk_dots_packed_f32_powervsx);
    run_kernel_case("nk_dots_pack_f32_powervsx",
                    test_dots_pack_layout<f32_t, host_backend_t, nk_dots_pack_size_f32_powervsx,
                                          nk_dots_packed_shape_f32_powervsx, nk_dots_pack_f32_powervsx>,
                    powervsx);
    run_kernel_case("nk_dots_packed_bf16_powervsx", test_dots_packed<bf16_t>, powervsx, nk_dots_pack_size_bf16_powervsx,
                    nk_dots_pack_bf16_powervsx, nk_dots_packed_bf16_powervsx);
    run_kernel_case("nk_dots_pack_bf16_powervsx",
                    test_dots_pack_layout<bf16_t, host_backend_t, nk_dots_pack_size_bf16_powervsx,
                                          nk_dots_packed_shape_bf16_powervsx, nk_dots_pack_bf16_powervsx>,
                    powervsx);
    run_kernel_case("nk_dots_packed_f16_powervsx", test_dots_packed<f16_t>, powervsx, nk_dots_pack_size_f16_powervsx,
                    nk_dots_pack_f16_powervsx, nk_dots_packed_f16_powervsx);
    run_kernel_case("nk_dots_pack_f16_powervsx",
                    test_dots_pack_layout<f16_t, host_backend_t, nk_dots_pack_size_f16_powervsx,
                                          nk_dots_packed_shape_f16_powervsx, nk_dots_pack_f16_powervsx>,
                    powervsx);
    run_kernel_case("nk_dots_packed_i8_powervsx", test_dots_packed<i8_t>, powervsx, nk_dots_pack_size_i8_powervsx,
                    nk_dots_pack_i8_powervsx, nk_dots_packed_i8_powervsx);
    run_kernel_case("nk_dots_pack_i8_powervsx",
                    test_dots_pack_layout<i8_t, host_backend_t, nk_dots_pack_size_i8_powervsx,
                                          nk_dots_packed_shape_i8_powervsx, nk_dots_pack_i8_powervsx>,
                    powervsx);
    run_kernel_case("nk_dots_packed_u8_powervsx", test_dots_packed<u8_t>, powervsx, nk_dots_pack_size_u8_powervsx,
                    nk_dots_pack_u8_powervsx, nk_dots_packed_u8_powervsx);
    run_kernel_case("nk_dots_pack_u8_powervsx",
                    test_dots_pack_layout<u8_t, host_backend_t, nk_dots_pack_size_u8_powervsx,
                                          nk_dots_packed_shape_u8_powervsx, nk_dots_pack_u8_powervsx>,
                    powervsx);
    run_kernel_case("nk_dots_packed_u1_powervsx", test_dots_packed<u1x8_t>, powervsx, nk_dots_pack_size_u1_powervsx,
                    nk_dots_pack_u1_powervsx, nk_dots_packed_u1_powervsx);
    run_kernel_case("nk_dots_pack_u1_powervsx",
                    test_dots_pack_layout<u1x8_t, host_backend_t, nk_dots_pack_size_u1_powervsx,
                                          nk_dots_packed_shape_u1_powervsx, nk_dots_pack_u1_powervsx>,
                    powervsx);

    run_kernel_case("nk_dots_symmetric_f64_powervsx", test_dots_symmetric<f64_t>, powervsx,
                    nk_dots_symmetric_f64_powervsx);
    run_kernel_case("nk_dots_symmetric_f32_powervsx", test_dots_symmetric<f32_t>, powervsx,
                    nk_dots_symmetric_f32_powervsx);
    run_kernel_case("nk_dots_symmetric_bf16_powervsx", test_dots_symmetric<bf16_t>, powervsx,
                    nk_dots_symmetric_bf16_powervsx);
    run_kernel_case("nk_dots_symmetric_f16_powervsx", test_dots_symmetric<f16_t>, powervsx,
                    nk_dots_symmetric_f16_powervsx);
    run_kernel_case("nk_dots_symmetric_i8_powervsx", test_dots_symmetric<i8_t>, powervsx,
                    nk_dots_symmetric_i8_powervsx);
    run_kernel_case("nk_dots_symmetric_u8_powervsx", test_dots_symmetric<u8_t>, powervsx,
                    nk_dots_symmetric_u8_powervsx);
    run_kernel_case("nk_dots_symmetric_u1_powervsx", test_dots_symmetric<u1x8_t>, powervsx,
                    nk_dots_symmetric_u1_powervsx);

    run_kernel_case("nk_angulars_packed_f64_powervsx", test_angulars_packed<f64_t>, powervsx,
                    nk_dots_pack_size_f64_powervsx, nk_dots_pack_f64_powervsx, nk_angulars_packed_f64_powervsx);
    run_kernel_case("nk_angulars_packed_f32_powervsx", test_angulars_packed<f32_t>, powervsx,
                    nk_dots_pack_size_f32_powervsx, nk_dots_pack_f32_powervsx, nk_angulars_packed_f32_powervsx);
    run_kernel_case("nk_angulars_packed_bf16_powervsx", test_angulars_packed<bf16_t>, powervsx,
                    nk_dots_pack_size_bf16_powervsx, nk_dots_pack_bf16_powervsx, nk_angulars_packed_bf16_powervsx);
    run_kernel_case("nk_angulars_packed_f16_powervsx", test_angulars_packed<f16_t>, powervsx,
                    nk_dots_pack_size_f16_powervsx, nk_dots_pack_f16_powervsx, nk_angulars_packed_f16_powervsx);

    run_kernel_case("nk_angulars_symmetric_f64_powervsx", test_angulars_symmetric<f64_t>, powervsx,
                    nk_angulars_symmetric_f64_powervsx);
    run_kernel_case("nk_angulars_symmetric_f32_powervsx", test_angulars_symmetric<f32_t>, powervsx,
                    nk_angulars_symmetric_f32_powervsx);
    run_kernel_case("nk_angulars_symmetric_bf16_powervsx", test_angulars_symmetric<bf16_t>, powervsx,
                    nk_angulars_symmetric_bf16_powervsx);
    run_kernel_case("nk_angulars_symmetric_f16_powervsx", test_angulars_symmetric<f16_t>, powervsx,
                    nk_angulars_symmetric_f16_powervsx);

    run_kernel_case("nk_euclideans_packed_f64_powervsx", test_euclideans_packed<f64_t>, powervsx,
                    nk_dots_pack_size_f64_powervsx, nk_dots_pack_f64_powervsx, nk_euclideans_packed_f64_powervsx);
    run_kernel_case("nk_euclideans_packed_f32_powervsx", test_euclideans_packed<f32_t>, powervsx,
                    nk_dots_pack_size_f32_powervsx, nk_dots_pack_f32_powervsx, nk_euclideans_packed_f32_powervsx);
    run_kernel_case("nk_euclideans_packed_bf16_powervsx", test_euclideans_packed<bf16_t>, powervsx,
                    nk_dots_pack_size_bf16_powervsx, nk_dots_pack_bf16_powervsx, nk_euclideans_packed_bf16_powervsx);
    run_kernel_case("nk_euclideans_packed_f16_powervsx", test_euclideans_packed<f16_t>, powervsx,
                    nk_dots_pack_size_f16_powervsx, nk_dots_pack_f16_powervsx, nk_euclideans_packed_f16_powervsx);

    run_kernel_case("nk_euclideans_symmetric_f64_powervsx", test_euclideans_symmetric<f64_t>, powervsx,
                    nk_euclideans_symmetric_f64_powervsx);
    run_kernel_case("nk_euclideans_symmetric_f32_powervsx", test_euclideans_symmetric<f32_t>, powervsx,
                    nk_euclideans_symmetric_f32_powervsx);
    run_kernel_case("nk_euclideans_symmetric_bf16_powervsx", test_euclideans_symmetric<bf16_t>, powervsx,
                    nk_euclideans_symmetric_bf16_powervsx);
    run_kernel_case("nk_euclideans_symmetric_f16_powervsx", test_euclideans_symmetric<f16_t>, powervsx,
                    nk_euclideans_symmetric_f16_powervsx);

    run_kernel_case("nk_hammings_packed_u1_powervsx", test_hammings_packed<u1x8_t>, powervsx,
                    nk_dots_pack_size_u1_powervsx, nk_dots_pack_u1_powervsx, nk_hammings_packed_u1_powervsx);
    run_kernel_case("nk_hammings_symmetric_u1_powervsx", test_hammings_symmetric<u1x8_t>, powervsx,
                    nk_hammings_symmetric_u1_powervsx);

    run_kernel_case("nk_jaccards_packed_u1_powervsx", test_jaccards_packed<u1x8_t>, powervsx,
                    nk_dots_pack_size_u1_powervsx, nk_dots_pack_u1_powervsx, nk_jaccards_packed_u1_powervsx);
    run_kernel_case("nk_jaccards_symmetric_u1_powervsx", test_jaccards_symmetric<u1x8_t>, powervsx,
                    nk_jaccards_symmetric_u1_powervsx);
#endif // NUMKONG_TARGET_POWERVSX

#if NUMKONG_TARGET_POWERMMA
    auto const powermma = nk_cap_powermma_k;
    run_kernel_case("nk_dots_packed_f32_powermma", test_dots_packed<f32_t>, powermma, nk_dots_pack_size_f32_powermma,
                    nk_dots_pack_f32_powermma, nk_dots_packed_f32_powermma);
    run_kernel_case("nk_dots_pack_f32_powermma",
                    test_dots_pack_layout<f32_t, host_backend_t, nk_dots_pack_size_f32_powermma,
                                          nk_dots_packed_shape_f32_powermma, nk_dots_pack_f32_powermma>,
                    powermma);
    run_kernel_case("nk_dots_packed_bf16_powermma", test_dots_packed<bf16_t>, powermma, nk_dots_pack_size_bf16_powermma,
                    nk_dots_pack_bf16_powermma, nk_dots_packed_bf16_powermma);
    run_kernel_case("nk_dots_pack_bf16_powermma",
                    test_dots_pack_layout<bf16_t, host_backend_t, nk_dots_pack_size_bf16_powermma,
                                          nk_dots_packed_shape_bf16_powermma, nk_dots_pack_bf16_powermma>,
                    powermma);
    run_kernel_case("nk_dots_packed_f16_powermma", test_dots_packed<f16_t>, powermma, nk_dots_pack_size_f16_powermma,
                    nk_dots_pack_f16_powermma, nk_dots_packed_f16_powermma);
    run_kernel_case("nk_dots_pack_f16_powermma",
                    test_dots_pack_layout<f16_t, host_backend_t, nk_dots_pack_size_f16_powermma,
                                          nk_dots_packed_shape_f16_powermma, nk_dots_pack_f16_powermma>,
                    powermma);
    run_kernel_case("nk_dots_packed_i8_powermma", test_dots_packed<i8_t>, powermma, nk_dots_pack_size_i8_powermma,
                    nk_dots_pack_i8_powermma, nk_dots_packed_i8_powermma);
    run_kernel_case("nk_dots_pack_i8_powermma",
                    test_dots_pack_layout<i8_t, host_backend_t, nk_dots_pack_size_i8_powermma,
                                          nk_dots_packed_shape_i8_powermma, nk_dots_pack_i8_powermma>,
                    powermma);
    run_kernel_case("nk_dots_packed_u8_powermma", test_dots_packed<u8_t>, powermma, nk_dots_pack_size_u8_powermma,
                    nk_dots_pack_u8_powermma, nk_dots_packed_u8_powermma);
    run_kernel_case("nk_dots_pack_u8_powermma",
                    test_dots_pack_layout<u8_t, host_backend_t, nk_dots_pack_size_u8_powermma,
                                          nk_dots_packed_shape_u8_powermma, nk_dots_pack_u8_powermma>,
                    powermma);

    run_kernel_case("nk_dots_symmetric_f32_powermma", test_dots_symmetric<f32_t>, powermma,
                    nk_dots_symmetric_f32_powermma);
    run_kernel_case("nk_dots_symmetric_bf16_powermma", test_dots_symmetric<bf16_t>, powermma,
                    nk_dots_symmetric_bf16_powermma);
    run_kernel_case("nk_dots_symmetric_f16_powermma", test_dots_symmetric<f16_t>, powermma,
                    nk_dots_symmetric_f16_powermma);
    run_kernel_case("nk_dots_symmetric_i8_powermma", test_dots_symmetric<i8_t>, powermma,
                    nk_dots_symmetric_i8_powermma);
    run_kernel_case("nk_dots_symmetric_u8_powermma", test_dots_symmetric<u8_t>, powermma,
                    nk_dots_symmetric_u8_powermma);

    run_kernel_case("nk_angulars_packed_f32_powermma", test_angulars_packed<f32_t>, powermma,
                    nk_dots_pack_size_f32_powermma, nk_dots_pack_f32_powermma, nk_angulars_packed_f32_powermma);
    run_kernel_case("nk_angulars_packed_bf16_powermma", test_angulars_packed<bf16_t>, powermma,
                    nk_dots_pack_size_bf16_powermma, nk_dots_pack_bf16_powermma, nk_angulars_packed_bf16_powermma);
    run_kernel_case("nk_angulars_packed_f16_powermma", test_angulars_packed<f16_t>, powermma,
                    nk_dots_pack_size_f16_powermma, nk_dots_pack_f16_powermma, nk_angulars_packed_f16_powermma);
    run_kernel_case("nk_angulars_packed_i8_powermma", test_angulars_packed<i8_t>, powermma,
                    nk_dots_pack_size_i8_powermma, nk_dots_pack_i8_powermma, nk_angulars_packed_i8_powermma);
    run_kernel_case("nk_angulars_packed_u8_powermma", test_angulars_packed<u8_t>, powermma,
                    nk_dots_pack_size_u8_powermma, nk_dots_pack_u8_powermma, nk_angulars_packed_u8_powermma);

    run_kernel_case("nk_angulars_symmetric_f32_powermma", test_angulars_symmetric<f32_t>, powermma,
                    nk_angulars_symmetric_f32_powermma);
    run_kernel_case("nk_angulars_symmetric_bf16_powermma", test_angulars_symmetric<bf16_t>, powermma,
                    nk_angulars_symmetric_bf16_powermma);
    run_kernel_case("nk_angulars_symmetric_f16_powermma", test_angulars_symmetric<f16_t>, powermma,
                    nk_angulars_symmetric_f16_powermma);
    run_kernel_case("nk_angulars_symmetric_i8_powermma", test_angulars_symmetric<i8_t>, powermma,
                    nk_angulars_symmetric_i8_powermma);
    run_kernel_case("nk_angulars_symmetric_u8_powermma", test_angulars_symmetric<u8_t>, powermma,
                    nk_angulars_symmetric_u8_powermma);

    run_kernel_case("nk_euclideans_packed_f32_powermma", test_euclideans_packed<f32_t>, powermma,
                    nk_dots_pack_size_f32_powermma, nk_dots_pack_f32_powermma, nk_euclideans_packed_f32_powermma);
    run_kernel_case("nk_euclideans_packed_bf16_powermma", test_euclideans_packed<bf16_t>, powermma,
                    nk_dots_pack_size_bf16_powermma, nk_dots_pack_bf16_powermma, nk_euclideans_packed_bf16_powermma);
    run_kernel_case("nk_euclideans_packed_f16_powermma", test_euclideans_packed<f16_t>, powermma,
                    nk_dots_pack_size_f16_powermma, nk_dots_pack_f16_powermma, nk_euclideans_packed_f16_powermma);
    run_kernel_case("nk_euclideans_packed_i8_powermma", test_euclideans_packed<i8_t>, powermma,
                    nk_dots_pack_size_i8_powermma, nk_dots_pack_i8_powermma, nk_euclideans_packed_i8_powermma);
    run_kernel_case("nk_euclideans_packed_u8_powermma", test_euclideans_packed<u8_t>, powermma,
                    nk_dots_pack_size_u8_powermma, nk_dots_pack_u8_powermma, nk_euclideans_packed_u8_powermma);

    run_kernel_case("nk_euclideans_symmetric_f32_powermma", test_euclideans_symmetric<f32_t>, powermma,
                    nk_euclideans_symmetric_f32_powermma);
    run_kernel_case("nk_euclideans_symmetric_bf16_powermma", test_euclideans_symmetric<bf16_t>, powermma,
                    nk_euclideans_symmetric_bf16_powermma);
    run_kernel_case("nk_euclideans_symmetric_f16_powermma", test_euclideans_symmetric<f16_t>, powermma,
                    nk_euclideans_symmetric_f16_powermma);
    run_kernel_case("nk_euclideans_symmetric_i8_powermma", test_euclideans_symmetric<i8_t>, powermma,
                    nk_euclideans_symmetric_i8_powermma);
    run_kernel_case("nk_euclideans_symmetric_u8_powermma", test_euclideans_symmetric<u8_t>, powermma,
                    nk_euclideans_symmetric_u8_powermma);
#endif // NUMKONG_TARGET_POWERMMA
}

} // namespace ashvardanian::numkong::test
