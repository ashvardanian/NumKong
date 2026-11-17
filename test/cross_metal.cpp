/**
 *  @file test/cross_metal.cpp
 *  @author Ash Vardanian
 *  @date September 24, 2026
 *  @brief Batch operation tests - Metal capabilities.
 *
 *  Runs the dots scenarios of `cross.hpp` through a @c metal_backend_t for the Metal baseline and
 *  every Apple GPU family the device runs, against the serial `nk::` references on the host.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include "cross_device.hpp"

namespace ashvardanian::numkong::test {

#if NUMKONG_ARCH_METAL_

static error_stats_t test_metal_bound_lifetime() {
    error_stats_t stats(comparison_family_t::exact_k);
    auto origin = make_backend<metal_backend_t>();
    auto consumer = make_backend<metal_backend_t>();
    auto packed = make_vector<std::byte>(consumer, pack_size_bytes(stats, nk_dots_pack_size_i8_metal, 1, 16));
    {
        auto input = make_vector<i8_t>(origin, 16);
        std::fill_n(input.data(), 16, i8_t(1));
        consumer.call(nk_dots_pack_i8_metal, input.raw_values_data(), 1, 16, 16, packed.raw_values_data(), 0, 1);
    }
    auto query = make_vector<i8_t>(consumer, 16);
    auto result = make_vector<i32_t>(consumer, 1);
    std::fill_n(query.data(), 16, i8_t(1));
    consumer.call(nk_dots_packed_i8_metal, query.raw_values_data(), packed.raw_values_data(), result.raw_values_data(),
                  1, 1, 16, 16, sizeof(nk_i32_t), nullptr, 0);
    if (!stats.expect_status(consumer.synchronize())) return stats;
    stats.expect_true(result[0] == i32_t(16), "queued work lost an allocation released on another stream");
    return stats;
}

/** Every Metal baseline entry point, on any Apple GPU of family 7 or newer. */
static void run_cross_metal_baseline_cases() {
    auto const metal = nk_cap_metal_k;
    metal_backend_t const backend = make_backend<metal_backend_t>();
    run_kernel_case("nk_dots_packed_nvfp4_metal", test_dots_packed<nvfp4_t, metal_backend_t>, metal, backend,
                    nk_dots_pack_size_nvfp4_metal, nk_dots_pack_nvfp4_metal, nk_dots_packed_nvfp4_metal);
    run_kernel_case("nk_dots_symmetric_nvfp4_metal", test_dots_symmetric<nvfp4_t, metal_backend_t>, metal, backend,
                    nk_dots_symmetric_nvfp4_metal);
    run_kernel_case("nk_angulars_packed_nvfp4_metal", test_angulars_packed<nvfp4_t, metal_backend_t>, metal,
                    nk_dots_pack_size_nvfp4_metal, nk_dots_pack_nvfp4_metal, nk_angulars_packed_nvfp4_metal);
    run_kernel_case("nk_angulars_symmetric_nvfp4_metal", test_angulars_symmetric<nvfp4_t, metal_backend_t>, metal,
                    nk_angulars_symmetric_nvfp4_metal);
    run_kernel_case("nk_euclideans_packed_nvfp4_metal", test_euclideans_packed<nvfp4_t, metal_backend_t>, metal,
                    nk_dots_pack_size_nvfp4_metal, nk_dots_pack_nvfp4_metal, nk_euclideans_packed_nvfp4_metal);
    run_kernel_case("nk_euclideans_symmetric_nvfp4_metal", test_euclideans_symmetric<nvfp4_t, metal_backend_t>, metal,
                    nk_euclideans_symmetric_nvfp4_metal);
    run_kernel_case("nk_dots_packed_mxfp4_metal", test_dots_packed<mxfp4_t, metal_backend_t>, metal, backend,
                    nk_dots_pack_size_mxfp4_metal, nk_dots_pack_mxfp4_metal, nk_dots_packed_mxfp4_metal);
    run_kernel_case("nk_dots_symmetric_mxfp4_metal", test_dots_symmetric<mxfp4_t, metal_backend_t>, metal, backend,
                    nk_dots_symmetric_mxfp4_metal);
    run_kernel_case("nk_angulars_packed_mxfp4_metal", test_angulars_packed<mxfp4_t, metal_backend_t>, metal,
                    nk_dots_pack_size_mxfp4_metal, nk_dots_pack_mxfp4_metal, nk_angulars_packed_mxfp4_metal);
    run_kernel_case("nk_angulars_symmetric_mxfp4_metal", test_angulars_symmetric<mxfp4_t, metal_backend_t>, metal,
                    nk_angulars_symmetric_mxfp4_metal);
    run_kernel_case("nk_euclideans_packed_mxfp4_metal", test_euclideans_packed<mxfp4_t, metal_backend_t>, metal,
                    nk_dots_pack_size_mxfp4_metal, nk_dots_pack_mxfp4_metal, nk_euclideans_packed_mxfp4_metal);
    run_kernel_case("nk_euclideans_symmetric_mxfp4_metal", test_euclideans_symmetric<mxfp4_t, metal_backend_t>, metal,
                    nk_euclideans_symmetric_mxfp4_metal);
    run_kernel_case("nk_dots_packed_mxfp6e2m3_metal", test_dots_packed<mxfp6e2m3_t, metal_backend_t>, metal, backend,
                    nk_dots_pack_size_mxfp6e2m3_metal, nk_dots_pack_mxfp6e2m3_metal, nk_dots_packed_mxfp6e2m3_metal);
    run_kernel_case("nk_dots_symmetric_mxfp6e2m3_metal", test_dots_symmetric<mxfp6e2m3_t, metal_backend_t>, metal,
                    backend, nk_dots_symmetric_mxfp6e2m3_metal);
    run_kernel_case("nk_angulars_packed_mxfp6e2m3_metal", test_angulars_packed<mxfp6e2m3_t, metal_backend_t>, metal,
                    nk_dots_pack_size_mxfp6e2m3_metal, nk_dots_pack_mxfp6e2m3_metal,
                    nk_angulars_packed_mxfp6e2m3_metal);
    run_kernel_case("nk_angulars_symmetric_mxfp6e2m3_metal", test_angulars_symmetric<mxfp6e2m3_t, metal_backend_t>,
                    metal, nk_angulars_symmetric_mxfp6e2m3_metal);
    run_kernel_case("nk_euclideans_packed_mxfp6e2m3_metal", test_euclideans_packed<mxfp6e2m3_t, metal_backend_t>, metal,
                    nk_dots_pack_size_mxfp6e2m3_metal, nk_dots_pack_mxfp6e2m3_metal,
                    nk_euclideans_packed_mxfp6e2m3_metal);
    run_kernel_case("nk_euclideans_symmetric_mxfp6e2m3_metal", test_euclideans_symmetric<mxfp6e2m3_t, metal_backend_t>,
                    metal, nk_euclideans_symmetric_mxfp6e2m3_metal);
    run_kernel_case("nk_dots_packed_mxfp6e3m2_metal", test_dots_packed<mxfp6e3m2_t, metal_backend_t>, metal, backend,
                    nk_dots_pack_size_mxfp6e3m2_metal, nk_dots_pack_mxfp6e3m2_metal, nk_dots_packed_mxfp6e3m2_metal);
    run_kernel_case("nk_dots_symmetric_mxfp6e3m2_metal", test_dots_symmetric<mxfp6e3m2_t, metal_backend_t>, metal,
                    backend, nk_dots_symmetric_mxfp6e3m2_metal);
    run_kernel_case("nk_angulars_packed_mxfp6e3m2_metal", test_angulars_packed<mxfp6e3m2_t, metal_backend_t>, metal,
                    nk_dots_pack_size_mxfp6e3m2_metal, nk_dots_pack_mxfp6e3m2_metal,
                    nk_angulars_packed_mxfp6e3m2_metal);
    run_kernel_case("nk_angulars_symmetric_mxfp6e3m2_metal", test_angulars_symmetric<mxfp6e3m2_t, metal_backend_t>,
                    metal, nk_angulars_symmetric_mxfp6e3m2_metal);
    run_kernel_case("nk_euclideans_packed_mxfp6e3m2_metal", test_euclideans_packed<mxfp6e3m2_t, metal_backend_t>, metal,
                    nk_dots_pack_size_mxfp6e3m2_metal, nk_dots_pack_mxfp6e3m2_metal,
                    nk_euclideans_packed_mxfp6e3m2_metal);
    run_kernel_case("nk_euclideans_symmetric_mxfp6e3m2_metal", test_euclideans_symmetric<mxfp6e3m2_t, metal_backend_t>,
                    metal, nk_euclideans_symmetric_mxfp6e3m2_metal);
    run_kernel_case("nk_dots_packed_mxfp8e4m3_metal", test_dots_packed<mxfp8e4m3_t, metal_backend_t>, metal, backend,
                    nk_dots_pack_size_mxfp8e4m3_metal, nk_dots_pack_mxfp8e4m3_metal, nk_dots_packed_mxfp8e4m3_metal);
    run_kernel_case("nk_dots_symmetric_mxfp8e4m3_metal", test_dots_symmetric<mxfp8e4m3_t, metal_backend_t>, metal,
                    backend, nk_dots_symmetric_mxfp8e4m3_metal);
    run_kernel_case("nk_angulars_packed_mxfp8e4m3_metal", test_angulars_packed<mxfp8e4m3_t, metal_backend_t>, metal,
                    nk_dots_pack_size_mxfp8e4m3_metal, nk_dots_pack_mxfp8e4m3_metal,
                    nk_angulars_packed_mxfp8e4m3_metal);
    run_kernel_case("nk_angulars_symmetric_mxfp8e4m3_metal", test_angulars_symmetric<mxfp8e4m3_t, metal_backend_t>,
                    metal, nk_angulars_symmetric_mxfp8e4m3_metal);
    run_kernel_case("nk_euclideans_packed_mxfp8e4m3_metal", test_euclideans_packed<mxfp8e4m3_t, metal_backend_t>, metal,
                    nk_dots_pack_size_mxfp8e4m3_metal, nk_dots_pack_mxfp8e4m3_metal,
                    nk_euclideans_packed_mxfp8e4m3_metal);
    run_kernel_case("nk_euclideans_symmetric_mxfp8e4m3_metal", test_euclideans_symmetric<mxfp8e4m3_t, metal_backend_t>,
                    metal, nk_euclideans_symmetric_mxfp8e4m3_metal);
    run_kernel_case("nk_dots_packed_mxfp8e5m2_metal", test_dots_packed<mxfp8e5m2_t, metal_backend_t>, metal, backend,
                    nk_dots_pack_size_mxfp8e5m2_metal, nk_dots_pack_mxfp8e5m2_metal, nk_dots_packed_mxfp8e5m2_metal);
    run_kernel_case("nk_dots_symmetric_mxfp8e5m2_metal", test_dots_symmetric<mxfp8e5m2_t, metal_backend_t>, metal,
                    backend, nk_dots_symmetric_mxfp8e5m2_metal);
    run_kernel_case("nk_angulars_packed_mxfp8e5m2_metal", test_angulars_packed<mxfp8e5m2_t, metal_backend_t>, metal,
                    nk_dots_pack_size_mxfp8e5m2_metal, nk_dots_pack_mxfp8e5m2_metal,
                    nk_angulars_packed_mxfp8e5m2_metal);
    run_kernel_case("nk_angulars_symmetric_mxfp8e5m2_metal", test_angulars_symmetric<mxfp8e5m2_t, metal_backend_t>,
                    metal, nk_angulars_symmetric_mxfp8e5m2_metal);
    run_kernel_case("nk_euclideans_packed_mxfp8e5m2_metal", test_euclideans_packed<mxfp8e5m2_t, metal_backend_t>, metal,
                    nk_dots_pack_size_mxfp8e5m2_metal, nk_dots_pack_mxfp8e5m2_metal,
                    nk_euclideans_packed_mxfp8e5m2_metal);
    run_kernel_case("nk_euclideans_symmetric_mxfp8e5m2_metal", test_euclideans_symmetric<mxfp8e5m2_t, metal_backend_t>,
                    metal, nk_euclideans_symmetric_mxfp8e5m2_metal);
    run_kernel_case("test_metal_bound_lifetime", test_metal_bound_lifetime, metal);
    run_kernel_case("nk_dots_packed_i8_metal", test_dots_packed<i8_t, metal_backend_t>, metal, backend,
                    nk_dots_pack_size_i8_metal, nk_dots_pack_i8_metal, nk_dots_packed_i8_metal);
    run_kernel_case("nk_dots_pack_i8_metal",
                    test_dots_pack_layout<i8_t, metal_backend_t, nk_dots_pack_size_i8_metal,
                                          nk_dots_packed_shape_i8_metal, nk_dots_pack_i8_metal>,
                    metal, backend);
    run_kernel_case("test_dots_launch_contract<i8_t, metal_backend_t, nk_dots_pack_size_i8_metal, " //
                    "nk_dots_packed_i8_metal, nk_dots_symmetric_i8_metal>(backend)",
                    test_dots_launch_contract<i8_t, metal_backend_t, nk_dots_pack_size_i8_metal,
                                              nk_dots_packed_i8_metal, nk_dots_symmetric_i8_metal>,
                    metal, backend);
    run_kernel_case("nk_dots_symmetric_i8_metal", test_dots_symmetric<i8_t, metal_backend_t>, metal, backend,
                    nk_dots_symmetric_i8_metal);
    run_kernel_case("nk_dots_packed_u8_metal", test_dots_packed<u8_t, metal_backend_t>, metal, backend,
                    nk_dots_pack_size_u8_metal, nk_dots_pack_u8_metal, nk_dots_packed_u8_metal);
    run_kernel_case("nk_dots_pack_u8_metal",
                    test_dots_pack_layout<u8_t, metal_backend_t, nk_dots_pack_size_u8_metal,
                                          nk_dots_packed_shape_u8_metal, nk_dots_pack_u8_metal>,
                    metal, backend);
    run_kernel_case("test_dots_launch_contract<u8_t, metal_backend_t, nk_dots_pack_size_u8_metal, " //
                    "nk_dots_packed_u8_metal, nk_dots_symmetric_u8_metal>(backend)",
                    test_dots_launch_contract<u8_t, metal_backend_t, nk_dots_pack_size_u8_metal,
                                              nk_dots_packed_u8_metal, nk_dots_symmetric_u8_metal>,
                    metal, backend);
    run_kernel_case("nk_dots_symmetric_u8_metal", test_dots_symmetric<u8_t, metal_backend_t>, metal, backend,
                    nk_dots_symmetric_u8_metal);
    run_kernel_case("nk_dots_packed_i4_metal", test_dots_packed<i4x2_t, metal_backend_t>, metal, backend,
                    nk_dots_pack_size_i4_metal, nk_dots_pack_i4_metal, nk_dots_packed_i4_metal);
    run_kernel_case("nk_dots_pack_i4_metal",
                    test_dots_pack_layout<i4x2_t, metal_backend_t, nk_dots_pack_size_i4_metal,
                                          nk_dots_packed_shape_i4_metal, nk_dots_pack_i4_metal>,
                    metal, backend);
    run_kernel_case("test_dots_launch_contract<i4x2_t, metal_backend_t, nk_dots_pack_size_i4_metal, " //
                    "nk_dots_packed_i4_metal, nk_dots_symmetric_i4_metal>(backend)",
                    test_dots_launch_contract<i4x2_t, metal_backend_t, nk_dots_pack_size_i4_metal,
                                              nk_dots_packed_i4_metal, nk_dots_symmetric_i4_metal>,
                    metal, backend);
    run_kernel_case("nk_dots_symmetric_i4_metal", test_dots_symmetric<i4x2_t, metal_backend_t>, metal, backend,
                    nk_dots_symmetric_i4_metal);
    run_kernel_case("nk_dots_packed_u4_metal", test_dots_packed<u4x2_t, metal_backend_t>, metal, backend,
                    nk_dots_pack_size_u4_metal, nk_dots_pack_u4_metal, nk_dots_packed_u4_metal);
    run_kernel_case("nk_dots_pack_u4_metal",
                    test_dots_pack_layout<u4x2_t, metal_backend_t, nk_dots_pack_size_u4_metal,
                                          nk_dots_packed_shape_u4_metal, nk_dots_pack_u4_metal>,
                    metal, backend);
    run_kernel_case("test_dots_launch_contract<u4x2_t, metal_backend_t, nk_dots_pack_size_u4_metal, " //
                    "nk_dots_packed_u4_metal, nk_dots_symmetric_u4_metal>(backend)",
                    test_dots_launch_contract<u4x2_t, metal_backend_t, nk_dots_pack_size_u4_metal,
                                              nk_dots_packed_u4_metal, nk_dots_symmetric_u4_metal>,
                    metal, backend);
    run_kernel_case("nk_dots_symmetric_u4_metal", test_dots_symmetric<u4x2_t, metal_backend_t>, metal, backend,
                    nk_dots_symmetric_u4_metal);
    run_kernel_case("nk_dots_packed_f16_metal", test_dots_packed<f16_t, metal_backend_t>, metal, backend,
                    nk_dots_pack_size_f16_metal, nk_dots_pack_f16_metal, nk_dots_packed_f16_metal);
    run_kernel_case("nk_dots_pack_f16_metal",
                    test_dots_pack_layout<f16_t, metal_backend_t, nk_dots_pack_size_f16_metal,
                                          nk_dots_packed_shape_f16_metal, nk_dots_pack_f16_metal>,
                    metal, backend);
    run_kernel_case("test_dots_launch_contract<f16_t, metal_backend_t, nk_dots_pack_size_f16_metal, " //
                    "nk_dots_packed_f16_metal, nk_dots_symmetric_f16_metal>(backend)",
                    test_dots_launch_contract<f16_t, metal_backend_t, nk_dots_pack_size_f16_metal,
                                              nk_dots_packed_f16_metal, nk_dots_symmetric_f16_metal>,
                    metal, backend);
    run_kernel_case("nk_dots_symmetric_f16_metal", test_dots_symmetric<f16_t, metal_backend_t>, metal, backend,
                    nk_dots_symmetric_f16_metal);
    run_kernel_case("nk_dots_packed_bf16_metal", test_dots_packed<bf16_t, metal_backend_t>, metal, backend,
                    nk_dots_pack_size_bf16_metal, nk_dots_pack_bf16_metal, nk_dots_packed_bf16_metal);
    run_kernel_case("nk_dots_pack_bf16_metal",
                    test_dots_pack_layout<bf16_t, metal_backend_t, nk_dots_pack_size_bf16_metal,
                                          nk_dots_packed_shape_bf16_metal, nk_dots_pack_bf16_metal>,
                    metal, backend);
    run_kernel_case("test_dots_launch_contract<bf16_t, metal_backend_t, nk_dots_pack_size_bf16_metal, " //
                    "nk_dots_packed_bf16_metal, nk_dots_symmetric_bf16_metal>(backend)",
                    test_dots_launch_contract<bf16_t, metal_backend_t, nk_dots_pack_size_bf16_metal,
                                              nk_dots_packed_bf16_metal, nk_dots_symmetric_bf16_metal>,
                    metal, backend);
    run_kernel_case("nk_dots_symmetric_bf16_metal", test_dots_symmetric<bf16_t, metal_backend_t>, metal, backend,
                    nk_dots_symmetric_bf16_metal);
    run_kernel_case("nk_dots_packed_e4m3_metal", test_dots_packed<e4m3_t, metal_backend_t>, metal, backend,
                    nk_dots_pack_size_e4m3_metal, nk_dots_pack_e4m3_metal, nk_dots_packed_e4m3_metal);
    run_kernel_case("nk_dots_pack_e4m3_metal",
                    test_dots_pack_layout<e4m3_t, metal_backend_t, nk_dots_pack_size_e4m3_metal,
                                          nk_dots_packed_shape_e4m3_metal, nk_dots_pack_e4m3_metal>,
                    metal, backend);
    run_kernel_case("test_dots_launch_contract<e4m3_t, metal_backend_t, nk_dots_pack_size_e4m3_metal, " //
                    "nk_dots_packed_e4m3_metal, nk_dots_symmetric_e4m3_metal>(backend)",
                    test_dots_launch_contract<e4m3_t, metal_backend_t, nk_dots_pack_size_e4m3_metal,
                                              nk_dots_packed_e4m3_metal, nk_dots_symmetric_e4m3_metal>,
                    metal, backend);
    run_kernel_case("nk_dots_symmetric_e4m3_metal", test_dots_symmetric<e4m3_t, metal_backend_t>, metal, backend,
                    nk_dots_symmetric_e4m3_metal);
    run_kernel_case("nk_dots_packed_e5m2_metal", test_dots_packed<e5m2_t, metal_backend_t>, metal, backend,
                    nk_dots_pack_size_e5m2_metal, nk_dots_pack_e5m2_metal, nk_dots_packed_e5m2_metal);
    run_kernel_case("nk_dots_pack_e5m2_metal",
                    test_dots_pack_layout<e5m2_t, metal_backend_t, nk_dots_pack_size_e5m2_metal,
                                          nk_dots_packed_shape_e5m2_metal, nk_dots_pack_e5m2_metal>,
                    metal, backend);
    run_kernel_case("test_dots_launch_contract<e5m2_t, metal_backend_t, nk_dots_pack_size_e5m2_metal, " //
                    "nk_dots_packed_e5m2_metal, nk_dots_symmetric_e5m2_metal>(backend)",
                    test_dots_launch_contract<e5m2_t, metal_backend_t, nk_dots_pack_size_e5m2_metal,
                                              nk_dots_packed_e5m2_metal, nk_dots_symmetric_e5m2_metal>,
                    metal, backend);
    run_kernel_case("nk_dots_symmetric_e5m2_metal", test_dots_symmetric<e5m2_t, metal_backend_t>, metal, backend,
                    nk_dots_symmetric_e5m2_metal);
    run_kernel_case("nk_dots_packed_e3m2_metal", test_dots_packed<e3m2_t, metal_backend_t>, metal, backend,
                    nk_dots_pack_size_e3m2_metal, nk_dots_pack_e3m2_metal, nk_dots_packed_e3m2_metal);
    run_kernel_case("nk_dots_pack_e3m2_metal",
                    test_dots_pack_layout<e3m2_t, metal_backend_t, nk_dots_pack_size_e3m2_metal,
                                          nk_dots_packed_shape_e3m2_metal, nk_dots_pack_e3m2_metal>,
                    metal, backend);
    run_kernel_case("test_dots_launch_contract<e3m2_t, metal_backend_t, nk_dots_pack_size_e3m2_metal, " //
                    "nk_dots_packed_e3m2_metal, nk_dots_symmetric_e3m2_metal>(backend)",
                    test_dots_launch_contract<e3m2_t, metal_backend_t, nk_dots_pack_size_e3m2_metal,
                                              nk_dots_packed_e3m2_metal, nk_dots_symmetric_e3m2_metal>,
                    metal, backend);
    run_kernel_case("nk_dots_symmetric_e3m2_metal", test_dots_symmetric<e3m2_t, metal_backend_t>, metal, backend,
                    nk_dots_symmetric_e3m2_metal);
    run_kernel_case("nk_dots_packed_e2m3_metal", test_dots_packed<e2m3_t, metal_backend_t>, metal, backend,
                    nk_dots_pack_size_e2m3_metal, nk_dots_pack_e2m3_metal, nk_dots_packed_e2m3_metal);
    run_kernel_case("nk_dots_pack_e2m3_metal",
                    test_dots_pack_layout<e2m3_t, metal_backend_t, nk_dots_pack_size_e2m3_metal,
                                          nk_dots_packed_shape_e2m3_metal, nk_dots_pack_e2m3_metal>,
                    metal, backend);
    run_kernel_case("test_dots_launch_contract<e2m3_t, metal_backend_t, nk_dots_pack_size_e2m3_metal, " //
                    "nk_dots_packed_e2m3_metal, nk_dots_symmetric_e2m3_metal>(backend)",
                    test_dots_launch_contract<e2m3_t, metal_backend_t, nk_dots_pack_size_e2m3_metal,
                                              nk_dots_packed_e2m3_metal, nk_dots_symmetric_e2m3_metal>,
                    metal, backend);
    run_kernel_case("nk_dots_symmetric_e2m3_metal", test_dots_symmetric<e2m3_t, metal_backend_t>, metal, backend,
                    nk_dots_symmetric_e2m3_metal);
    run_kernel_case("nk_dots_packed_e2m1_metal", test_dots_packed<e2m1x2_t, metal_backend_t>, metal, backend,
                    nk_dots_pack_size_e2m1_metal, nk_dots_pack_e2m1_metal, nk_dots_packed_e2m1_metal);
    run_kernel_case("nk_dots_pack_e2m1_metal",
                    test_dots_pack_layout<e2m1x2_t, metal_backend_t, nk_dots_pack_size_e2m1_metal,
                                          nk_dots_packed_shape_e2m1_metal, nk_dots_pack_e2m1_metal>,
                    metal, backend);
    run_kernel_case("test_dots_launch_contract<e2m1x2_t, metal_backend_t, nk_dots_pack_size_e2m1_metal, " //
                    "nk_dots_packed_e2m1_metal, nk_dots_symmetric_e2m1_metal>(backend)",
                    test_dots_launch_contract<e2m1x2_t, metal_backend_t, nk_dots_pack_size_e2m1_metal,
                                              nk_dots_packed_e2m1_metal, nk_dots_symmetric_e2m1_metal>,
                    metal, backend);
    run_kernel_case("nk_dots_symmetric_e2m1_metal", test_dots_symmetric<e2m1x2_t, metal_backend_t>, metal, backend,
                    nk_dots_symmetric_e2m1_metal);
    run_kernel_case("nk_angulars_packed_i8_metal", test_angulars_packed<i8_t, metal_backend_t>, metal,
                    nk_dots_pack_size_i8_metal, nk_dots_pack_i8_metal, nk_angulars_packed_i8_metal);
    run_kernel_case("nk_angulars_symmetric_i8_metal", test_angulars_symmetric<i8_t, metal_backend_t>, metal,
                    nk_angulars_symmetric_i8_metal);
    run_kernel_case("nk_euclideans_packed_i8_metal", test_euclideans_packed<i8_t, metal_backend_t>, metal,
                    nk_dots_pack_size_i8_metal, nk_dots_pack_i8_metal, nk_euclideans_packed_i8_metal);
    run_kernel_case("nk_euclideans_symmetric_i8_metal", test_euclideans_symmetric<i8_t, metal_backend_t>, metal,
                    nk_euclideans_symmetric_i8_metal);
    run_kernel_case("nk_angulars_packed_u8_metal", test_angulars_packed<u8_t, metal_backend_t>, metal,
                    nk_dots_pack_size_u8_metal, nk_dots_pack_u8_metal, nk_angulars_packed_u8_metal);
    run_kernel_case("nk_angulars_symmetric_u8_metal", test_angulars_symmetric<u8_t, metal_backend_t>, metal,
                    nk_angulars_symmetric_u8_metal);
    run_kernel_case("nk_euclideans_packed_u8_metal", test_euclideans_packed<u8_t, metal_backend_t>, metal,
                    nk_dots_pack_size_u8_metal, nk_dots_pack_u8_metal, nk_euclideans_packed_u8_metal);
    run_kernel_case("nk_euclideans_symmetric_u8_metal", test_euclideans_symmetric<u8_t, metal_backend_t>, metal,
                    nk_euclideans_symmetric_u8_metal);
    run_kernel_case("nk_angulars_packed_i4_metal", test_angulars_packed<i4x2_t, metal_backend_t>, metal,
                    nk_dots_pack_size_i4_metal, nk_dots_pack_i4_metal, nk_angulars_packed_i4_metal);
    run_kernel_case("nk_angulars_symmetric_i4_metal", test_angulars_symmetric<i4x2_t, metal_backend_t>, metal,
                    nk_angulars_symmetric_i4_metal);
    run_kernel_case("nk_euclideans_packed_i4_metal", test_euclideans_packed<i4x2_t, metal_backend_t>, metal,
                    nk_dots_pack_size_i4_metal, nk_dots_pack_i4_metal, nk_euclideans_packed_i4_metal);
    run_kernel_case("nk_euclideans_symmetric_i4_metal", test_euclideans_symmetric<i4x2_t, metal_backend_t>, metal,
                    nk_euclideans_symmetric_i4_metal);
    run_kernel_case("nk_angulars_packed_u4_metal", test_angulars_packed<u4x2_t, metal_backend_t>, metal,
                    nk_dots_pack_size_u4_metal, nk_dots_pack_u4_metal, nk_angulars_packed_u4_metal);
    run_kernel_case("nk_angulars_symmetric_u4_metal", test_angulars_symmetric<u4x2_t, metal_backend_t>, metal,
                    nk_angulars_symmetric_u4_metal);
    run_kernel_case("nk_euclideans_packed_u4_metal", test_euclideans_packed<u4x2_t, metal_backend_t>, metal,
                    nk_dots_pack_size_u4_metal, nk_dots_pack_u4_metal, nk_euclideans_packed_u4_metal);
    run_kernel_case("nk_euclideans_symmetric_u4_metal", test_euclideans_symmetric<u4x2_t, metal_backend_t>, metal,
                    nk_euclideans_symmetric_u4_metal);
    run_kernel_case("nk_angulars_packed_f16_metal", test_angulars_packed<f16_t, metal_backend_t>, metal,
                    nk_dots_pack_size_f16_metal, nk_dots_pack_f16_metal, nk_angulars_packed_f16_metal);
    run_kernel_case("nk_angulars_symmetric_f16_metal", test_angulars_symmetric<f16_t, metal_backend_t>, metal,
                    nk_angulars_symmetric_f16_metal);
    run_kernel_case("nk_euclideans_packed_f16_metal", test_euclideans_packed<f16_t, metal_backend_t>, metal,
                    nk_dots_pack_size_f16_metal, nk_dots_pack_f16_metal, nk_euclideans_packed_f16_metal);
    run_kernel_case("nk_euclideans_symmetric_f16_metal", test_euclideans_symmetric<f16_t, metal_backend_t>, metal,
                    nk_euclideans_symmetric_f16_metal);
    run_kernel_case("nk_angulars_packed_bf16_metal", test_angulars_packed<bf16_t, metal_backend_t>, metal,
                    nk_dots_pack_size_bf16_metal, nk_dots_pack_bf16_metal, nk_angulars_packed_bf16_metal);
    run_kernel_case("nk_angulars_symmetric_bf16_metal", test_angulars_symmetric<bf16_t, metal_backend_t>, metal,
                    nk_angulars_symmetric_bf16_metal);
    run_kernel_case("nk_euclideans_packed_bf16_metal", test_euclideans_packed<bf16_t, metal_backend_t>, metal,
                    nk_dots_pack_size_bf16_metal, nk_dots_pack_bf16_metal, nk_euclideans_packed_bf16_metal);
    run_kernel_case("nk_euclideans_symmetric_bf16_metal", test_euclideans_symmetric<bf16_t, metal_backend_t>, metal,
                    nk_euclideans_symmetric_bf16_metal);
    run_kernel_case("nk_angulars_packed_e4m3_metal", test_angulars_packed<e4m3_t, metal_backend_t>, metal,
                    nk_dots_pack_size_e4m3_metal, nk_dots_pack_e4m3_metal, nk_angulars_packed_e4m3_metal);
    run_kernel_case("nk_angulars_symmetric_e4m3_metal", test_angulars_symmetric<e4m3_t, metal_backend_t>, metal,
                    nk_angulars_symmetric_e4m3_metal);
    run_kernel_case("nk_euclideans_packed_e4m3_metal", test_euclideans_packed<e4m3_t, metal_backend_t>, metal,
                    nk_dots_pack_size_e4m3_metal, nk_dots_pack_e4m3_metal, nk_euclideans_packed_e4m3_metal);
    run_kernel_case("nk_euclideans_symmetric_e4m3_metal", test_euclideans_symmetric<e4m3_t, metal_backend_t>, metal,
                    nk_euclideans_symmetric_e4m3_metal);
    run_kernel_case("nk_angulars_packed_e5m2_metal", test_angulars_packed<e5m2_t, metal_backend_t>, metal,
                    nk_dots_pack_size_e5m2_metal, nk_dots_pack_e5m2_metal, nk_angulars_packed_e5m2_metal);
    run_kernel_case("nk_angulars_symmetric_e5m2_metal", test_angulars_symmetric<e5m2_t, metal_backend_t>, metal,
                    nk_angulars_symmetric_e5m2_metal);
    run_kernel_case("nk_euclideans_packed_e5m2_metal", test_euclideans_packed<e5m2_t, metal_backend_t>, metal,
                    nk_dots_pack_size_e5m2_metal, nk_dots_pack_e5m2_metal, nk_euclideans_packed_e5m2_metal);
    run_kernel_case("nk_euclideans_symmetric_e5m2_metal", test_euclideans_symmetric<e5m2_t, metal_backend_t>, metal,
                    nk_euclideans_symmetric_e5m2_metal);
    run_kernel_case("nk_angulars_packed_e3m2_metal", test_angulars_packed<e3m2_t, metal_backend_t>, metal,
                    nk_dots_pack_size_e3m2_metal, nk_dots_pack_e3m2_metal, nk_angulars_packed_e3m2_metal);
    run_kernel_case("nk_angulars_symmetric_e3m2_metal", test_angulars_symmetric<e3m2_t, metal_backend_t>, metal,
                    nk_angulars_symmetric_e3m2_metal);
    run_kernel_case("nk_euclideans_packed_e3m2_metal", test_euclideans_packed<e3m2_t, metal_backend_t>, metal,
                    nk_dots_pack_size_e3m2_metal, nk_dots_pack_e3m2_metal, nk_euclideans_packed_e3m2_metal);
    run_kernel_case("nk_euclideans_symmetric_e3m2_metal", test_euclideans_symmetric<e3m2_t, metal_backend_t>, metal,
                    nk_euclideans_symmetric_e3m2_metal);
    run_kernel_case("nk_angulars_packed_e2m3_metal", test_angulars_packed<e2m3_t, metal_backend_t>, metal,
                    nk_dots_pack_size_e2m3_metal, nk_dots_pack_e2m3_metal, nk_angulars_packed_e2m3_metal);
    run_kernel_case("nk_angulars_symmetric_e2m3_metal", test_angulars_symmetric<e2m3_t, metal_backend_t>, metal,
                    nk_angulars_symmetric_e2m3_metal);
    run_kernel_case("nk_euclideans_packed_e2m3_metal", test_euclideans_packed<e2m3_t, metal_backend_t>, metal,
                    nk_dots_pack_size_e2m3_metal, nk_dots_pack_e2m3_metal, nk_euclideans_packed_e2m3_metal);
    run_kernel_case("nk_euclideans_symmetric_e2m3_metal", test_euclideans_symmetric<e2m3_t, metal_backend_t>, metal,
                    nk_euclideans_symmetric_e2m3_metal);
    run_kernel_case("nk_angulars_packed_e2m1_metal", test_angulars_packed<e2m1x2_t, metal_backend_t>, metal,
                    nk_dots_pack_size_e2m1_metal, nk_dots_pack_e2m1_metal, nk_angulars_packed_e2m1_metal);
    run_kernel_case("nk_angulars_symmetric_e2m1_metal", test_angulars_symmetric<e2m1x2_t, metal_backend_t>, metal,
                    nk_angulars_symmetric_e2m1_metal);
    run_kernel_case("nk_euclideans_packed_e2m1_metal", test_euclideans_packed<e2m1x2_t, metal_backend_t>, metal,
                    nk_dots_pack_size_e2m1_metal, nk_dots_pack_e2m1_metal, nk_euclideans_packed_e2m1_metal);
    run_kernel_case("nk_euclideans_symmetric_e2m1_metal", test_euclideans_symmetric<e2m1x2_t, metal_backend_t>, metal,
                    nk_euclideans_symmetric_e2m1_metal);
    run_kernel_case("nk_attention_packed_bf16_metal", test_attention_packed<bf16_t, metal_backend_t>, metal,
                    nk_attention_pack_size_bf16_metal, nk_attention_pack_bf16_metal, nk_attention_packed_bf16_metal);
    run_kernel_case("nk_attention_packed_f16_metal", test_attention_packed<f16_t, metal_backend_t>, metal,
                    nk_attention_pack_size_f16_metal, nk_attention_pack_f16_metal, nk_attention_packed_f16_metal);
    run_kernel_case("nk_attention_packed_e4m3_metal", test_attention_packed<e4m3_t, metal_backend_t>, metal,
                    nk_attention_pack_size_e4m3_metal, nk_attention_pack_e4m3_metal, nk_attention_packed_e4m3_metal);
    run_kernel_case("nk_attention_packed_i8_metal",
                    test_attention_packed<i8_t, metal_backend_t, attention_weights_t::bits_8_k>, metal,
                    nk_attention_pack_size_i8_metal, nk_attention_pack_i8_metal, nk_attention_packed_i8_metal);
    run_kernel_case("nk_attention_packed_nvfp4_metal", test_attention_packed<nvfp4_t, metal_backend_t>, metal,
                    nk_attention_pack_size_nvfp4_metal, nk_attention_pack_nvfp4_metal, nk_attention_packed_nvfp4_metal);
    run_kernel_case("nk_attention_packed_mxfp4_metal", test_attention_packed<mxfp4_t, metal_backend_t>, metal,
                    nk_attention_pack_size_mxfp4_metal, nk_attention_pack_mxfp4_metal, nk_attention_packed_mxfp4_metal);
    run_kernel_case("nk_attention_packed_mxfp6e2m3_metal", test_attention_packed<mxfp6e2m3_t, metal_backend_t>, metal,
                    nk_attention_pack_size_mxfp6e2m3_metal, nk_attention_pack_mxfp6e2m3_metal,
                    nk_attention_packed_mxfp6e2m3_metal);
    run_kernel_case("nk_attention_packed_mxfp6e3m2_metal", test_attention_packed<mxfp6e3m2_t, metal_backend_t>, metal,
                    nk_attention_pack_size_mxfp6e3m2_metal, nk_attention_pack_mxfp6e3m2_metal,
                    nk_attention_packed_mxfp6e3m2_metal);
    run_kernel_case("nk_attention_packed_mxfp8e4m3_metal", test_attention_packed<mxfp8e4m3_t, metal_backend_t>, metal,
                    nk_attention_pack_size_mxfp8e4m3_metal, nk_attention_pack_mxfp8e4m3_metal,
                    nk_attention_packed_mxfp8e4m3_metal);
    run_kernel_case("nk_attention_packed_mxfp8e5m2_metal", test_attention_packed<mxfp8e5m2_t, metal_backend_t>, metal,
                    nk_attention_pack_size_mxfp8e5m2_metal, nk_attention_pack_mxfp8e5m2_metal,
                    nk_attention_packed_mxfp8e5m2_metal);
    run_kernel_case("nk_attention_packed_gradients_bf16_metal",
                    test_attention_packed_gradients<bf16_t, metal_backend_t>, metal, nk_attention_pack_size_bf16_metal,
                    nk_attention_pack_bf16_metal, nk_attention_packed_gradients_bf16_metal);
    run_kernel_case("nk_attention_rope_f32_metal", test_attention_rope<f32_t, metal_backend_t>, metal,
                    nk_attention_rope_f32_metal);
    run_kernel_case("nk_attention_rope_bf16_metal", test_attention_rope<bf16_t, metal_backend_t>, metal,
                    nk_attention_rope_bf16_metal);
    run_kernel_case("nk_attention_rope_e4m3_metal", test_attention_rope<e4m3_t, metal_backend_t>, metal,
                    nk_attention_rope_e4m3_metal);
    run_kernel_case("nk_chamfers_packed_f16_metal",
                    test_shapes_flips<f16_t, nk_kernel_chamfers_packed_k, metal_backend_t>, metal,
                    nk_chamfers_packed_f16_metal);
    run_kernel_case("nk_hausdorffs_packed_f16_metal",
                    test_shapes_flips<f16_t, nk_kernel_hausdorffs_packed_k, metal_backend_t>, metal,
                    nk_hausdorffs_packed_f16_metal);
}

/** Every Apple9 entry point, on devices whose families include it. */
static void run_cross_apple9_cases() {
#if NUMKONG_TARGET_APPLE9
    auto const apple9 = nk_cap_apple9_k;
    metal_backend_t const backend = make_backend<metal_backend_t>();
    run_kernel_case("nk_dots_packed_nvfp4_apple9", test_dots_packed<nvfp4_t, metal_backend_t>, apple9, backend,
                    nk_dots_pack_size_nvfp4_apple9, nk_dots_pack_nvfp4_apple9, nk_dots_packed_nvfp4_apple9);
    run_kernel_case("nk_dots_symmetric_nvfp4_apple9", test_dots_symmetric<nvfp4_t, metal_backend_t>, apple9, backend,
                    nk_dots_symmetric_nvfp4_apple9);
    run_kernel_case("nk_angulars_packed_nvfp4_apple9", test_angulars_packed<nvfp4_t, metal_backend_t>, apple9,
                    nk_dots_pack_size_nvfp4_apple9, nk_dots_pack_nvfp4_apple9, nk_angulars_packed_nvfp4_apple9);
    run_kernel_case("nk_angulars_symmetric_nvfp4_apple9", test_angulars_symmetric<nvfp4_t, metal_backend_t>, apple9,
                    nk_angulars_symmetric_nvfp4_apple9);
    run_kernel_case("nk_euclideans_packed_nvfp4_apple9", test_euclideans_packed<nvfp4_t, metal_backend_t>, apple9,
                    nk_dots_pack_size_nvfp4_apple9, nk_dots_pack_nvfp4_apple9, nk_euclideans_packed_nvfp4_apple9);
    run_kernel_case("nk_euclideans_symmetric_nvfp4_apple9", test_euclideans_symmetric<nvfp4_t, metal_backend_t>, apple9,
                    nk_euclideans_symmetric_nvfp4_apple9);
    run_kernel_case("nk_dots_packed_mxfp4_apple9", test_dots_packed<mxfp4_t, metal_backend_t>, apple9, backend,
                    nk_dots_pack_size_mxfp4_apple9, nk_dots_pack_mxfp4_apple9, nk_dots_packed_mxfp4_apple9);
    run_kernel_case("nk_dots_symmetric_mxfp4_apple9", test_dots_symmetric<mxfp4_t, metal_backend_t>, apple9, backend,
                    nk_dots_symmetric_mxfp4_apple9);
    run_kernel_case("nk_angulars_packed_mxfp4_apple9", test_angulars_packed<mxfp4_t, metal_backend_t>, apple9,
                    nk_dots_pack_size_mxfp4_apple9, nk_dots_pack_mxfp4_apple9, nk_angulars_packed_mxfp4_apple9);
    run_kernel_case("nk_angulars_symmetric_mxfp4_apple9", test_angulars_symmetric<mxfp4_t, metal_backend_t>, apple9,
                    nk_angulars_symmetric_mxfp4_apple9);
    run_kernel_case("nk_euclideans_packed_mxfp4_apple9", test_euclideans_packed<mxfp4_t, metal_backend_t>, apple9,
                    nk_dots_pack_size_mxfp4_apple9, nk_dots_pack_mxfp4_apple9, nk_euclideans_packed_mxfp4_apple9);
    run_kernel_case("nk_euclideans_symmetric_mxfp4_apple9", test_euclideans_symmetric<mxfp4_t, metal_backend_t>, apple9,
                    nk_euclideans_symmetric_mxfp4_apple9);
    run_kernel_case("nk_dots_packed_mxfp6e2m3_apple9", test_dots_packed<mxfp6e2m3_t, metal_backend_t>, apple9, backend,
                    nk_dots_pack_size_mxfp6e2m3_apple9, nk_dots_pack_mxfp6e2m3_apple9, nk_dots_packed_mxfp6e2m3_apple9);
    run_kernel_case("nk_dots_symmetric_mxfp6e2m3_apple9", test_dots_symmetric<mxfp6e2m3_t, metal_backend_t>, apple9,
                    backend, nk_dots_symmetric_mxfp6e2m3_apple9);
    run_kernel_case("nk_angulars_packed_mxfp6e2m3_apple9", test_angulars_packed<mxfp6e2m3_t, metal_backend_t>, apple9,
                    nk_dots_pack_size_mxfp6e2m3_apple9, nk_dots_pack_mxfp6e2m3_apple9,
                    nk_angulars_packed_mxfp6e2m3_apple9);
    run_kernel_case("nk_angulars_symmetric_mxfp6e2m3_apple9", test_angulars_symmetric<mxfp6e2m3_t, metal_backend_t>,
                    apple9, nk_angulars_symmetric_mxfp6e2m3_apple9);
    run_kernel_case("nk_euclideans_packed_mxfp6e2m3_apple9", test_euclideans_packed<mxfp6e2m3_t, metal_backend_t>,
                    apple9, nk_dots_pack_size_mxfp6e2m3_apple9, nk_dots_pack_mxfp6e2m3_apple9,
                    nk_euclideans_packed_mxfp6e2m3_apple9);
    run_kernel_case("nk_euclideans_symmetric_mxfp6e2m3_apple9", test_euclideans_symmetric<mxfp6e2m3_t, metal_backend_t>,
                    apple9, nk_euclideans_symmetric_mxfp6e2m3_apple9);
    run_kernel_case("nk_dots_packed_mxfp6e3m2_apple9", test_dots_packed<mxfp6e3m2_t, metal_backend_t>, apple9, backend,
                    nk_dots_pack_size_mxfp6e3m2_apple9, nk_dots_pack_mxfp6e3m2_apple9, nk_dots_packed_mxfp6e3m2_apple9);
    run_kernel_case("nk_dots_symmetric_mxfp6e3m2_apple9", test_dots_symmetric<mxfp6e3m2_t, metal_backend_t>, apple9,
                    backend, nk_dots_symmetric_mxfp6e3m2_apple9);
    run_kernel_case("nk_angulars_packed_mxfp6e3m2_apple9", test_angulars_packed<mxfp6e3m2_t, metal_backend_t>, apple9,
                    nk_dots_pack_size_mxfp6e3m2_apple9, nk_dots_pack_mxfp6e3m2_apple9,
                    nk_angulars_packed_mxfp6e3m2_apple9);
    run_kernel_case("nk_angulars_symmetric_mxfp6e3m2_apple9", test_angulars_symmetric<mxfp6e3m2_t, metal_backend_t>,
                    apple9, nk_angulars_symmetric_mxfp6e3m2_apple9);
    run_kernel_case("nk_euclideans_packed_mxfp6e3m2_apple9", test_euclideans_packed<mxfp6e3m2_t, metal_backend_t>,
                    apple9, nk_dots_pack_size_mxfp6e3m2_apple9, nk_dots_pack_mxfp6e3m2_apple9,
                    nk_euclideans_packed_mxfp6e3m2_apple9);
    run_kernel_case("nk_euclideans_symmetric_mxfp6e3m2_apple9", test_euclideans_symmetric<mxfp6e3m2_t, metal_backend_t>,
                    apple9, nk_euclideans_symmetric_mxfp6e3m2_apple9);
    run_kernel_case("nk_dots_packed_mxfp8e4m3_apple9", test_dots_packed<mxfp8e4m3_t, metal_backend_t>, apple9, backend,
                    nk_dots_pack_size_mxfp8e4m3_apple9, nk_dots_pack_mxfp8e4m3_apple9, nk_dots_packed_mxfp8e4m3_apple9);
    run_kernel_case("nk_dots_symmetric_mxfp8e4m3_apple9", test_dots_symmetric<mxfp8e4m3_t, metal_backend_t>, apple9,
                    backend, nk_dots_symmetric_mxfp8e4m3_apple9);
    run_kernel_case("nk_angulars_packed_mxfp8e4m3_apple9", test_angulars_packed<mxfp8e4m3_t, metal_backend_t>, apple9,
                    nk_dots_pack_size_mxfp8e4m3_apple9, nk_dots_pack_mxfp8e4m3_apple9,
                    nk_angulars_packed_mxfp8e4m3_apple9);
    run_kernel_case("nk_angulars_symmetric_mxfp8e4m3_apple9", test_angulars_symmetric<mxfp8e4m3_t, metal_backend_t>,
                    apple9, nk_angulars_symmetric_mxfp8e4m3_apple9);
    run_kernel_case("nk_euclideans_packed_mxfp8e4m3_apple9", test_euclideans_packed<mxfp8e4m3_t, metal_backend_t>,
                    apple9, nk_dots_pack_size_mxfp8e4m3_apple9, nk_dots_pack_mxfp8e4m3_apple9,
                    nk_euclideans_packed_mxfp8e4m3_apple9);
    run_kernel_case("nk_euclideans_symmetric_mxfp8e4m3_apple9", test_euclideans_symmetric<mxfp8e4m3_t, metal_backend_t>,
                    apple9, nk_euclideans_symmetric_mxfp8e4m3_apple9);
    run_kernel_case("nk_dots_packed_mxfp8e5m2_apple9", test_dots_packed<mxfp8e5m2_t, metal_backend_t>, apple9, backend,
                    nk_dots_pack_size_mxfp8e5m2_apple9, nk_dots_pack_mxfp8e5m2_apple9, nk_dots_packed_mxfp8e5m2_apple9);
    run_kernel_case("nk_dots_symmetric_mxfp8e5m2_apple9", test_dots_symmetric<mxfp8e5m2_t, metal_backend_t>, apple9,
                    backend, nk_dots_symmetric_mxfp8e5m2_apple9);
    run_kernel_case("nk_angulars_packed_mxfp8e5m2_apple9", test_angulars_packed<mxfp8e5m2_t, metal_backend_t>, apple9,
                    nk_dots_pack_size_mxfp8e5m2_apple9, nk_dots_pack_mxfp8e5m2_apple9,
                    nk_angulars_packed_mxfp8e5m2_apple9);
    run_kernel_case("nk_angulars_symmetric_mxfp8e5m2_apple9", test_angulars_symmetric<mxfp8e5m2_t, metal_backend_t>,
                    apple9, nk_angulars_symmetric_mxfp8e5m2_apple9);
    run_kernel_case("nk_euclideans_packed_mxfp8e5m2_apple9", test_euclideans_packed<mxfp8e5m2_t, metal_backend_t>,
                    apple9, nk_dots_pack_size_mxfp8e5m2_apple9, nk_dots_pack_mxfp8e5m2_apple9,
                    nk_euclideans_packed_mxfp8e5m2_apple9);
    run_kernel_case("nk_euclideans_symmetric_mxfp8e5m2_apple9", test_euclideans_symmetric<mxfp8e5m2_t, metal_backend_t>,
                    apple9, nk_euclideans_symmetric_mxfp8e5m2_apple9);
    run_kernel_case("nk_dots_packed_f16_apple9", test_dots_packed<f16_t, metal_backend_t>, apple9, backend,
                    nk_dots_pack_size_f16_apple9, nk_dots_pack_f16_apple9, nk_dots_packed_f16_apple9);
    run_kernel_case("nk_dots_pack_f16_apple9",
                    test_dots_pack_layout<f16_t, metal_backend_t, nk_dots_pack_size_f16_apple9,
                                          nk_dots_packed_shape_f16_apple9, nk_dots_pack_f16_apple9>,
                    apple9, backend);
    run_kernel_case("test_dots_launch_contract<f16_t, metal_backend_t, nk_dots_pack_size_f16_apple9, " //
                    "nk_dots_packed_f16_apple9, nk_dots_symmetric_f16_apple9>(backend)",
                    test_dots_launch_contract<f16_t, metal_backend_t, nk_dots_pack_size_f16_apple9,
                                              nk_dots_packed_f16_apple9, nk_dots_symmetric_f16_apple9>,
                    apple9, backend);
    run_kernel_case("nk_dots_symmetric_f16_apple9", test_dots_symmetric<f16_t, metal_backend_t>, apple9, backend,
                    nk_dots_symmetric_f16_apple9);
    run_kernel_case("nk_dots_packed_bf16_apple9", test_dots_packed<bf16_t, metal_backend_t>, apple9, backend,
                    nk_dots_pack_size_bf16_apple9, nk_dots_pack_bf16_apple9, nk_dots_packed_bf16_apple9);
    run_kernel_case("nk_dots_pack_bf16_apple9",
                    test_dots_pack_layout<bf16_t, metal_backend_t, nk_dots_pack_size_bf16_apple9,
                                          nk_dots_packed_shape_bf16_apple9, nk_dots_pack_bf16_apple9>,
                    apple9, backend);
    run_kernel_case("test_dots_launch_contract<bf16_t, metal_backend_t, nk_dots_pack_size_bf16_apple9, " //
                    "nk_dots_packed_bf16_apple9, nk_dots_symmetric_bf16_apple9>(backend)",
                    test_dots_launch_contract<bf16_t, metal_backend_t, nk_dots_pack_size_bf16_apple9,
                                              nk_dots_packed_bf16_apple9, nk_dots_symmetric_bf16_apple9>,
                    apple9, backend);
    run_kernel_case("nk_dots_symmetric_bf16_apple9", test_dots_symmetric<bf16_t, metal_backend_t>, apple9, backend,
                    nk_dots_symmetric_bf16_apple9);
    run_kernel_case("nk_dots_packed_e4m3_apple9", test_dots_packed<e4m3_t, metal_backend_t>, apple9, backend,
                    nk_dots_pack_size_e4m3_apple9, nk_dots_pack_e4m3_apple9, nk_dots_packed_e4m3_apple9);
    run_kernel_case("nk_dots_pack_e4m3_apple9",
                    test_dots_pack_layout<e4m3_t, metal_backend_t, nk_dots_pack_size_e4m3_apple9,
                                          nk_dots_packed_shape_e4m3_apple9, nk_dots_pack_e4m3_apple9>,
                    apple9, backend);
    run_kernel_case("test_dots_launch_contract<e4m3_t, metal_backend_t, nk_dots_pack_size_e4m3_apple9, " //
                    "nk_dots_packed_e4m3_apple9, nk_dots_symmetric_e4m3_apple9>(backend)",
                    test_dots_launch_contract<e4m3_t, metal_backend_t, nk_dots_pack_size_e4m3_apple9,
                                              nk_dots_packed_e4m3_apple9, nk_dots_symmetric_e4m3_apple9>,
                    apple9, backend);
    run_kernel_case("nk_dots_symmetric_e4m3_apple9", test_dots_symmetric<e4m3_t, metal_backend_t>, apple9, backend,
                    nk_dots_symmetric_e4m3_apple9);
    run_kernel_case("nk_dots_packed_e5m2_apple9", test_dots_packed<e5m2_t, metal_backend_t>, apple9, backend,
                    nk_dots_pack_size_e5m2_apple9, nk_dots_pack_e5m2_apple9, nk_dots_packed_e5m2_apple9);
    run_kernel_case("nk_dots_pack_e5m2_apple9",
                    test_dots_pack_layout<e5m2_t, metal_backend_t, nk_dots_pack_size_e5m2_apple9,
                                          nk_dots_packed_shape_e5m2_apple9, nk_dots_pack_e5m2_apple9>,
                    apple9, backend);
    run_kernel_case("test_dots_launch_contract<e5m2_t, metal_backend_t, nk_dots_pack_size_e5m2_apple9, " //
                    "nk_dots_packed_e5m2_apple9, nk_dots_symmetric_e5m2_apple9>(backend)",
                    test_dots_launch_contract<e5m2_t, metal_backend_t, nk_dots_pack_size_e5m2_apple9,
                                              nk_dots_packed_e5m2_apple9, nk_dots_symmetric_e5m2_apple9>,
                    apple9, backend);
    run_kernel_case("nk_dots_symmetric_e5m2_apple9", test_dots_symmetric<e5m2_t, metal_backend_t>, apple9, backend,
                    nk_dots_symmetric_e5m2_apple9);
    run_kernel_case("nk_dots_packed_e3m2_apple9", test_dots_packed<e3m2_t, metal_backend_t>, apple9, backend,
                    nk_dots_pack_size_e3m2_apple9, nk_dots_pack_e3m2_apple9, nk_dots_packed_e3m2_apple9);
    run_kernel_case("nk_dots_pack_e3m2_apple9",
                    test_dots_pack_layout<e3m2_t, metal_backend_t, nk_dots_pack_size_e3m2_apple9,
                                          nk_dots_packed_shape_e3m2_apple9, nk_dots_pack_e3m2_apple9>,
                    apple9, backend);
    run_kernel_case("test_dots_launch_contract<e3m2_t, metal_backend_t, nk_dots_pack_size_e3m2_apple9, " //
                    "nk_dots_packed_e3m2_apple9, nk_dots_symmetric_e3m2_apple9>(backend)",
                    test_dots_launch_contract<e3m2_t, metal_backend_t, nk_dots_pack_size_e3m2_apple9,
                                              nk_dots_packed_e3m2_apple9, nk_dots_symmetric_e3m2_apple9>,
                    apple9, backend);
    run_kernel_case("nk_dots_symmetric_e3m2_apple9", test_dots_symmetric<e3m2_t, metal_backend_t>, apple9, backend,
                    nk_dots_symmetric_e3m2_apple9);
    run_kernel_case("nk_dots_packed_e2m3_apple9", test_dots_packed<e2m3_t, metal_backend_t>, apple9, backend,
                    nk_dots_pack_size_e2m3_apple9, nk_dots_pack_e2m3_apple9, nk_dots_packed_e2m3_apple9);
    run_kernel_case("nk_dots_pack_e2m3_apple9",
                    test_dots_pack_layout<e2m3_t, metal_backend_t, nk_dots_pack_size_e2m3_apple9,
                                          nk_dots_packed_shape_e2m3_apple9, nk_dots_pack_e2m3_apple9>,
                    apple9, backend);
    run_kernel_case("test_dots_launch_contract<e2m3_t, metal_backend_t, nk_dots_pack_size_e2m3_apple9, " //
                    "nk_dots_packed_e2m3_apple9, nk_dots_symmetric_e2m3_apple9>(backend)",
                    test_dots_launch_contract<e2m3_t, metal_backend_t, nk_dots_pack_size_e2m3_apple9,
                                              nk_dots_packed_e2m3_apple9, nk_dots_symmetric_e2m3_apple9>,
                    apple9, backend);
    run_kernel_case("nk_dots_symmetric_e2m3_apple9", test_dots_symmetric<e2m3_t, metal_backend_t>, apple9, backend,
                    nk_dots_symmetric_e2m3_apple9);
    run_kernel_case("nk_dots_packed_e2m1_apple9", test_dots_packed<e2m1x2_t, metal_backend_t>, apple9, backend,
                    nk_dots_pack_size_e2m1_apple9, nk_dots_pack_e2m1_apple9, nk_dots_packed_e2m1_apple9);
    run_kernel_case("nk_dots_pack_e2m1_apple9",
                    test_dots_pack_layout<e2m1x2_t, metal_backend_t, nk_dots_pack_size_e2m1_apple9,
                                          nk_dots_packed_shape_e2m1_apple9, nk_dots_pack_e2m1_apple9>,
                    apple9, backend);
    run_kernel_case("test_dots_launch_contract<e2m1x2_t, metal_backend_t, nk_dots_pack_size_e2m1_apple9, " //
                    "nk_dots_packed_e2m1_apple9, nk_dots_symmetric_e2m1_apple9>(backend)",
                    test_dots_launch_contract<e2m1x2_t, metal_backend_t, nk_dots_pack_size_e2m1_apple9,
                                              nk_dots_packed_e2m1_apple9, nk_dots_symmetric_e2m1_apple9>,
                    apple9, backend);
    run_kernel_case("nk_dots_symmetric_e2m1_apple9", test_dots_symmetric<e2m1x2_t, metal_backend_t>, apple9, backend,
                    nk_dots_symmetric_e2m1_apple9);
    run_kernel_case("nk_angulars_packed_f16_apple9", test_angulars_packed<f16_t, metal_backend_t>, apple9,
                    nk_dots_pack_size_f16_apple9, nk_dots_pack_f16_apple9, nk_angulars_packed_f16_apple9);
    run_kernel_case("nk_angulars_symmetric_f16_apple9", test_angulars_symmetric<f16_t, metal_backend_t>, apple9,
                    nk_angulars_symmetric_f16_apple9);
    run_kernel_case("nk_euclideans_packed_f16_apple9", test_euclideans_packed<f16_t, metal_backend_t>, apple9,
                    nk_dots_pack_size_f16_apple9, nk_dots_pack_f16_apple9, nk_euclideans_packed_f16_apple9);
    run_kernel_case("nk_euclideans_symmetric_f16_apple9", test_euclideans_symmetric<f16_t, metal_backend_t>, apple9,
                    nk_euclideans_symmetric_f16_apple9);
    run_kernel_case("nk_angulars_packed_bf16_apple9", test_angulars_packed<bf16_t, metal_backend_t>, apple9,
                    nk_dots_pack_size_bf16_apple9, nk_dots_pack_bf16_apple9, nk_angulars_packed_bf16_apple9);
    run_kernel_case("nk_angulars_symmetric_bf16_apple9", test_angulars_symmetric<bf16_t, metal_backend_t>, apple9,
                    nk_angulars_symmetric_bf16_apple9);
    run_kernel_case("nk_euclideans_packed_bf16_apple9", test_euclideans_packed<bf16_t, metal_backend_t>, apple9,
                    nk_dots_pack_size_bf16_apple9, nk_dots_pack_bf16_apple9, nk_euclideans_packed_bf16_apple9);
    run_kernel_case("nk_euclideans_symmetric_bf16_apple9", test_euclideans_symmetric<bf16_t, metal_backend_t>, apple9,
                    nk_euclideans_symmetric_bf16_apple9);
    run_kernel_case("nk_angulars_packed_e4m3_apple9", test_angulars_packed<e4m3_t, metal_backend_t>, apple9,
                    nk_dots_pack_size_e4m3_apple9, nk_dots_pack_e4m3_apple9, nk_angulars_packed_e4m3_apple9);
    run_kernel_case("nk_angulars_symmetric_e4m3_apple9", test_angulars_symmetric<e4m3_t, metal_backend_t>, apple9,
                    nk_angulars_symmetric_e4m3_apple9);
    run_kernel_case("nk_euclideans_packed_e4m3_apple9", test_euclideans_packed<e4m3_t, metal_backend_t>, apple9,
                    nk_dots_pack_size_e4m3_apple9, nk_dots_pack_e4m3_apple9, nk_euclideans_packed_e4m3_apple9);
    run_kernel_case("nk_euclideans_symmetric_e4m3_apple9", test_euclideans_symmetric<e4m3_t, metal_backend_t>, apple9,
                    nk_euclideans_symmetric_e4m3_apple9);
    run_kernel_case("nk_angulars_packed_e5m2_apple9", test_angulars_packed<e5m2_t, metal_backend_t>, apple9,
                    nk_dots_pack_size_e5m2_apple9, nk_dots_pack_e5m2_apple9, nk_angulars_packed_e5m2_apple9);
    run_kernel_case("nk_angulars_symmetric_e5m2_apple9", test_angulars_symmetric<e5m2_t, metal_backend_t>, apple9,
                    nk_angulars_symmetric_e5m2_apple9);
    run_kernel_case("nk_euclideans_packed_e5m2_apple9", test_euclideans_packed<e5m2_t, metal_backend_t>, apple9,
                    nk_dots_pack_size_e5m2_apple9, nk_dots_pack_e5m2_apple9, nk_euclideans_packed_e5m2_apple9);
    run_kernel_case("nk_euclideans_symmetric_e5m2_apple9", test_euclideans_symmetric<e5m2_t, metal_backend_t>, apple9,
                    nk_euclideans_symmetric_e5m2_apple9);
    run_kernel_case("nk_angulars_packed_e3m2_apple9", test_angulars_packed<e3m2_t, metal_backend_t>, apple9,
                    nk_dots_pack_size_e3m2_apple9, nk_dots_pack_e3m2_apple9, nk_angulars_packed_e3m2_apple9);
    run_kernel_case("nk_angulars_symmetric_e3m2_apple9", test_angulars_symmetric<e3m2_t, metal_backend_t>, apple9,
                    nk_angulars_symmetric_e3m2_apple9);
    run_kernel_case("nk_euclideans_packed_e3m2_apple9", test_euclideans_packed<e3m2_t, metal_backend_t>, apple9,
                    nk_dots_pack_size_e3m2_apple9, nk_dots_pack_e3m2_apple9, nk_euclideans_packed_e3m2_apple9);
    run_kernel_case("nk_euclideans_symmetric_e3m2_apple9", test_euclideans_symmetric<e3m2_t, metal_backend_t>, apple9,
                    nk_euclideans_symmetric_e3m2_apple9);
    run_kernel_case("nk_angulars_packed_e2m3_apple9", test_angulars_packed<e2m3_t, metal_backend_t>, apple9,
                    nk_dots_pack_size_e2m3_apple9, nk_dots_pack_e2m3_apple9, nk_angulars_packed_e2m3_apple9);
    run_kernel_case("nk_angulars_symmetric_e2m3_apple9", test_angulars_symmetric<e2m3_t, metal_backend_t>, apple9,
                    nk_angulars_symmetric_e2m3_apple9);
    run_kernel_case("nk_euclideans_packed_e2m3_apple9", test_euclideans_packed<e2m3_t, metal_backend_t>, apple9,
                    nk_dots_pack_size_e2m3_apple9, nk_dots_pack_e2m3_apple9, nk_euclideans_packed_e2m3_apple9);
    run_kernel_case("nk_euclideans_symmetric_e2m3_apple9", test_euclideans_symmetric<e2m3_t, metal_backend_t>, apple9,
                    nk_euclideans_symmetric_e2m3_apple9);
    run_kernel_case("nk_angulars_packed_e2m1_apple9", test_angulars_packed<e2m1x2_t, metal_backend_t>, apple9,
                    nk_dots_pack_size_e2m1_apple9, nk_dots_pack_e2m1_apple9, nk_angulars_packed_e2m1_apple9);
    run_kernel_case("nk_angulars_symmetric_e2m1_apple9", test_angulars_symmetric<e2m1x2_t, metal_backend_t>, apple9,
                    nk_angulars_symmetric_e2m1_apple9);
    run_kernel_case("nk_euclideans_packed_e2m1_apple9", test_euclideans_packed<e2m1x2_t, metal_backend_t>, apple9,
                    nk_dots_pack_size_e2m1_apple9, nk_dots_pack_e2m1_apple9, nk_euclideans_packed_e2m1_apple9);
    run_kernel_case("nk_euclideans_symmetric_e2m1_apple9", test_euclideans_symmetric<e2m1x2_t, metal_backend_t>, apple9,
                    nk_euclideans_symmetric_e2m1_apple9);
    run_kernel_case("nk_attention_packed_bf16_apple9",
                    test_attention_packed<bf16_t, metal_backend_t, attention_weights_t::bits_8_k>, apple9,
                    nk_attention_pack_size_bf16_apple9, nk_attention_pack_bf16_apple9, nk_attention_packed_bf16_apple9);
    run_kernel_case("nk_attention_packed_f16_apple9",
                    test_attention_packed<f16_t, metal_backend_t, attention_weights_t::bits_11_k>, apple9,
                    nk_attention_pack_size_f16_apple9, nk_attention_pack_f16_apple9, nk_attention_packed_f16_apple9);
    run_kernel_case("nk_attention_packed_e4m3_apple9",
                    test_attention_packed<e4m3_t, metal_backend_t, attention_weights_t::bits_11_k>, apple9,
                    nk_attention_pack_size_e4m3_apple9, nk_attention_pack_e4m3_apple9, nk_attention_packed_e4m3_apple9);
    run_kernel_case("nk_attention_packed_nvfp4_apple9",
                    test_attention_packed<nvfp4_t, metal_backend_t, attention_weights_t::bits_11_k>, apple9,
                    nk_attention_pack_size_nvfp4_apple9, nk_attention_pack_nvfp4_apple9,
                    nk_attention_packed_nvfp4_apple9);
    run_kernel_case("nk_attention_packed_mxfp4_apple9",
                    test_attention_packed<mxfp4_t, metal_backend_t, attention_weights_t::bits_8_k>, apple9,
                    nk_attention_pack_size_mxfp4_apple9, nk_attention_pack_mxfp4_apple9,
                    nk_attention_packed_mxfp4_apple9);
    run_kernel_case("nk_attention_packed_mxfp6e2m3_apple9",
                    test_attention_packed<mxfp6e2m3_t, metal_backend_t, attention_weights_t::bits_8_k>, apple9,
                    nk_attention_pack_size_mxfp6e2m3_apple9, nk_attention_pack_mxfp6e2m3_apple9,
                    nk_attention_packed_mxfp6e2m3_apple9);
    run_kernel_case("nk_attention_packed_mxfp6e3m2_apple9",
                    test_attention_packed<mxfp6e3m2_t, metal_backend_t, attention_weights_t::bits_8_k>, apple9,
                    nk_attention_pack_size_mxfp6e3m2_apple9, nk_attention_pack_mxfp6e3m2_apple9,
                    nk_attention_packed_mxfp6e3m2_apple9);
    run_kernel_case("nk_attention_packed_mxfp8e4m3_apple9",
                    test_attention_packed<mxfp8e4m3_t, metal_backend_t, attention_weights_t::bits_8_k>, apple9,
                    nk_attention_pack_size_mxfp8e4m3_apple9, nk_attention_pack_mxfp8e4m3_apple9,
                    nk_attention_packed_mxfp8e4m3_apple9);
    run_kernel_case("nk_attention_packed_mxfp8e5m2_apple9",
                    test_attention_packed<mxfp8e5m2_t, metal_backend_t, attention_weights_t::bits_8_k>, apple9,
                    nk_attention_pack_size_mxfp8e5m2_apple9, nk_attention_pack_mxfp8e5m2_apple9,
                    nk_attention_packed_mxfp8e5m2_apple9);
    run_kernel_case(
        "nk_attention_packed_gradients_bf16_apple9", test_attention_packed_gradients<bf16_t, metal_backend_t>, apple9,
        nk_attention_pack_size_bf16_apple9, nk_attention_pack_bf16_apple9, nk_attention_packed_gradients_bf16_apple9);
#endif // NUMKONG_TARGET_APPLE9
}

/** Every Apple10 entry point, on devices whose families include it. */
static void run_cross_apple10_cases() {
#if NUMKONG_TARGET_APPLE10
    auto const apple10 = nk_cap_apple10_k;
    metal_backend_t const backend = make_backend<metal_backend_t>();
    run_kernel_case("nk_dots_packed_nvfp4_apple10", test_dots_packed<nvfp4_t, metal_backend_t>, apple10, backend,
                    nk_dots_pack_size_nvfp4_apple10, nk_dots_pack_nvfp4_apple10, nk_dots_packed_nvfp4_apple10);
    run_kernel_case("nk_dots_symmetric_nvfp4_apple10", test_dots_symmetric<nvfp4_t, metal_backend_t>, apple10, backend,
                    nk_dots_symmetric_nvfp4_apple10);
    run_kernel_case("nk_angulars_packed_nvfp4_apple10", test_angulars_packed<nvfp4_t, metal_backend_t>, apple10,
                    nk_dots_pack_size_nvfp4_apple10, nk_dots_pack_nvfp4_apple10, nk_angulars_packed_nvfp4_apple10);
    run_kernel_case("nk_angulars_symmetric_nvfp4_apple10", test_angulars_symmetric<nvfp4_t, metal_backend_t>, apple10,
                    nk_angulars_symmetric_nvfp4_apple10);
    run_kernel_case("nk_euclideans_packed_nvfp4_apple10", test_euclideans_packed<nvfp4_t, metal_backend_t>, apple10,
                    nk_dots_pack_size_nvfp4_apple10, nk_dots_pack_nvfp4_apple10, nk_euclideans_packed_nvfp4_apple10);
    run_kernel_case("nk_euclideans_symmetric_nvfp4_apple10", test_euclideans_symmetric<nvfp4_t, metal_backend_t>,
                    apple10, nk_euclideans_symmetric_nvfp4_apple10);
    run_kernel_case("nk_dots_packed_mxfp4_apple10", test_dots_packed<mxfp4_t, metal_backend_t>, apple10, backend,
                    nk_dots_pack_size_mxfp4_apple10, nk_dots_pack_mxfp4_apple10, nk_dots_packed_mxfp4_apple10);
    run_kernel_case("nk_dots_symmetric_mxfp4_apple10", test_dots_symmetric<mxfp4_t, metal_backend_t>, apple10, backend,
                    nk_dots_symmetric_mxfp4_apple10);
    run_kernel_case("nk_angulars_packed_mxfp4_apple10", test_angulars_packed<mxfp4_t, metal_backend_t>, apple10,
                    nk_dots_pack_size_mxfp4_apple10, nk_dots_pack_mxfp4_apple10, nk_angulars_packed_mxfp4_apple10);
    run_kernel_case("nk_angulars_symmetric_mxfp4_apple10", test_angulars_symmetric<mxfp4_t, metal_backend_t>, apple10,
                    nk_angulars_symmetric_mxfp4_apple10);
    run_kernel_case("nk_euclideans_packed_mxfp4_apple10", test_euclideans_packed<mxfp4_t, metal_backend_t>, apple10,
                    nk_dots_pack_size_mxfp4_apple10, nk_dots_pack_mxfp4_apple10, nk_euclideans_packed_mxfp4_apple10);
    run_kernel_case("nk_euclideans_symmetric_mxfp4_apple10", test_euclideans_symmetric<mxfp4_t, metal_backend_t>,
                    apple10, nk_euclideans_symmetric_mxfp4_apple10);
    run_kernel_case("nk_dots_packed_mxfp6e2m3_apple10", test_dots_packed<mxfp6e2m3_t, metal_backend_t>, apple10,
                    backend, nk_dots_pack_size_mxfp6e2m3_apple10, nk_dots_pack_mxfp6e2m3_apple10,
                    nk_dots_packed_mxfp6e2m3_apple10);
    run_kernel_case("nk_dots_symmetric_mxfp6e2m3_apple10", test_dots_symmetric<mxfp6e2m3_t, metal_backend_t>, apple10,
                    backend, nk_dots_symmetric_mxfp6e2m3_apple10);
    run_kernel_case("nk_angulars_packed_mxfp6e2m3_apple10", test_angulars_packed<mxfp6e2m3_t, metal_backend_t>, apple10,
                    nk_dots_pack_size_mxfp6e2m3_apple10, nk_dots_pack_mxfp6e2m3_apple10,
                    nk_angulars_packed_mxfp6e2m3_apple10);
    run_kernel_case("nk_angulars_symmetric_mxfp6e2m3_apple10", test_angulars_symmetric<mxfp6e2m3_t, metal_backend_t>,
                    apple10, nk_angulars_symmetric_mxfp6e2m3_apple10);
    run_kernel_case("nk_euclideans_packed_mxfp6e2m3_apple10", test_euclideans_packed<mxfp6e2m3_t, metal_backend_t>,
                    apple10, nk_dots_pack_size_mxfp6e2m3_apple10, nk_dots_pack_mxfp6e2m3_apple10,
                    nk_euclideans_packed_mxfp6e2m3_apple10);
    run_kernel_case("nk_euclideans_symmetric_mxfp6e2m3_apple10",
                    test_euclideans_symmetric<mxfp6e2m3_t, metal_backend_t>, apple10,
                    nk_euclideans_symmetric_mxfp6e2m3_apple10);
    run_kernel_case("nk_dots_packed_mxfp6e3m2_apple10", test_dots_packed<mxfp6e3m2_t, metal_backend_t>, apple10,
                    backend, nk_dots_pack_size_mxfp6e3m2_apple10, nk_dots_pack_mxfp6e3m2_apple10,
                    nk_dots_packed_mxfp6e3m2_apple10);
    run_kernel_case("nk_dots_symmetric_mxfp6e3m2_apple10", test_dots_symmetric<mxfp6e3m2_t, metal_backend_t>, apple10,
                    backend, nk_dots_symmetric_mxfp6e3m2_apple10);
    run_kernel_case("nk_angulars_packed_mxfp6e3m2_apple10", test_angulars_packed<mxfp6e3m2_t, metal_backend_t>, apple10,
                    nk_dots_pack_size_mxfp6e3m2_apple10, nk_dots_pack_mxfp6e3m2_apple10,
                    nk_angulars_packed_mxfp6e3m2_apple10);
    run_kernel_case("nk_angulars_symmetric_mxfp6e3m2_apple10", test_angulars_symmetric<mxfp6e3m2_t, metal_backend_t>,
                    apple10, nk_angulars_symmetric_mxfp6e3m2_apple10);
    run_kernel_case("nk_euclideans_packed_mxfp6e3m2_apple10", test_euclideans_packed<mxfp6e3m2_t, metal_backend_t>,
                    apple10, nk_dots_pack_size_mxfp6e3m2_apple10, nk_dots_pack_mxfp6e3m2_apple10,
                    nk_euclideans_packed_mxfp6e3m2_apple10);
    run_kernel_case("nk_euclideans_symmetric_mxfp6e3m2_apple10",
                    test_euclideans_symmetric<mxfp6e3m2_t, metal_backend_t>, apple10,
                    nk_euclideans_symmetric_mxfp6e3m2_apple10);
    run_kernel_case("nk_dots_packed_mxfp8e4m3_apple10", test_dots_packed<mxfp8e4m3_t, metal_backend_t>, apple10,
                    backend, nk_dots_pack_size_mxfp8e4m3_apple10, nk_dots_pack_mxfp8e4m3_apple10,
                    nk_dots_packed_mxfp8e4m3_apple10);
    run_kernel_case("nk_dots_symmetric_mxfp8e4m3_apple10", test_dots_symmetric<mxfp8e4m3_t, metal_backend_t>, apple10,
                    backend, nk_dots_symmetric_mxfp8e4m3_apple10);
    run_kernel_case("nk_angulars_packed_mxfp8e4m3_apple10", test_angulars_packed<mxfp8e4m3_t, metal_backend_t>, apple10,
                    nk_dots_pack_size_mxfp8e4m3_apple10, nk_dots_pack_mxfp8e4m3_apple10,
                    nk_angulars_packed_mxfp8e4m3_apple10);
    run_kernel_case("nk_angulars_symmetric_mxfp8e4m3_apple10", test_angulars_symmetric<mxfp8e4m3_t, metal_backend_t>,
                    apple10, nk_angulars_symmetric_mxfp8e4m3_apple10);
    run_kernel_case("nk_euclideans_packed_mxfp8e4m3_apple10", test_euclideans_packed<mxfp8e4m3_t, metal_backend_t>,
                    apple10, nk_dots_pack_size_mxfp8e4m3_apple10, nk_dots_pack_mxfp8e4m3_apple10,
                    nk_euclideans_packed_mxfp8e4m3_apple10);
    run_kernel_case("nk_euclideans_symmetric_mxfp8e4m3_apple10",
                    test_euclideans_symmetric<mxfp8e4m3_t, metal_backend_t>, apple10,
                    nk_euclideans_symmetric_mxfp8e4m3_apple10);
    run_kernel_case("nk_dots_packed_mxfp8e5m2_apple10", test_dots_packed<mxfp8e5m2_t, metal_backend_t>, apple10,
                    backend, nk_dots_pack_size_mxfp8e5m2_apple10, nk_dots_pack_mxfp8e5m2_apple10,
                    nk_dots_packed_mxfp8e5m2_apple10);
    run_kernel_case("nk_dots_symmetric_mxfp8e5m2_apple10", test_dots_symmetric<mxfp8e5m2_t, metal_backend_t>, apple10,
                    backend, nk_dots_symmetric_mxfp8e5m2_apple10);
    run_kernel_case("nk_angulars_packed_mxfp8e5m2_apple10", test_angulars_packed<mxfp8e5m2_t, metal_backend_t>, apple10,
                    nk_dots_pack_size_mxfp8e5m2_apple10, nk_dots_pack_mxfp8e5m2_apple10,
                    nk_angulars_packed_mxfp8e5m2_apple10);
    run_kernel_case("nk_angulars_symmetric_mxfp8e5m2_apple10", test_angulars_symmetric<mxfp8e5m2_t, metal_backend_t>,
                    apple10, nk_angulars_symmetric_mxfp8e5m2_apple10);
    run_kernel_case("nk_euclideans_packed_mxfp8e5m2_apple10", test_euclideans_packed<mxfp8e5m2_t, metal_backend_t>,
                    apple10, nk_dots_pack_size_mxfp8e5m2_apple10, nk_dots_pack_mxfp8e5m2_apple10,
                    nk_euclideans_packed_mxfp8e5m2_apple10);
    run_kernel_case("nk_euclideans_symmetric_mxfp8e5m2_apple10",
                    test_euclideans_symmetric<mxfp8e5m2_t, metal_backend_t>, apple10,
                    nk_euclideans_symmetric_mxfp8e5m2_apple10);
    run_kernel_case("nk_dots_packed_i4_apple10", test_dots_packed<i4x2_t, metal_backend_t>, apple10, backend,
                    nk_dots_pack_size_i4_apple10, nk_dots_pack_i4_apple10, nk_dots_packed_i4_apple10);
    run_kernel_case("nk_dots_pack_i4_apple10",
                    test_dots_pack_layout<i4x2_t, metal_backend_t, nk_dots_pack_size_i4_apple10,
                                          nk_dots_packed_shape_i4_apple10, nk_dots_pack_i4_apple10>,
                    apple10, backend);
    run_kernel_case("test_dots_launch_contract<i4x2_t, metal_backend_t, nk_dots_pack_size_i4_apple10, " //
                    "nk_dots_packed_i4_apple10, nk_dots_symmetric_i4_apple10>(backend)",
                    test_dots_launch_contract<i4x2_t, metal_backend_t, nk_dots_pack_size_i4_apple10,
                                              nk_dots_packed_i4_apple10, nk_dots_symmetric_i4_apple10>,
                    apple10, backend);
    run_kernel_case("nk_dots_symmetric_i4_apple10", test_dots_symmetric<i4x2_t, metal_backend_t>, apple10, backend,
                    nk_dots_symmetric_i4_apple10);
    run_kernel_case("nk_dots_packed_u4_apple10", test_dots_packed<u4x2_t, metal_backend_t>, apple10, backend,
                    nk_dots_pack_size_u4_apple10, nk_dots_pack_u4_apple10, nk_dots_packed_u4_apple10);
    run_kernel_case("nk_dots_pack_u4_apple10",
                    test_dots_pack_layout<u4x2_t, metal_backend_t, nk_dots_pack_size_u4_apple10,
                                          nk_dots_packed_shape_u4_apple10, nk_dots_pack_u4_apple10>,
                    apple10, backend);
    run_kernel_case("test_dots_launch_contract<u4x2_t, metal_backend_t, nk_dots_pack_size_u4_apple10, " //
                    "nk_dots_packed_u4_apple10, nk_dots_symmetric_u4_apple10>(backend)",
                    test_dots_launch_contract<u4x2_t, metal_backend_t, nk_dots_pack_size_u4_apple10,
                                              nk_dots_packed_u4_apple10, nk_dots_symmetric_u4_apple10>,
                    apple10, backend);
    run_kernel_case("nk_dots_symmetric_u4_apple10", test_dots_symmetric<u4x2_t, metal_backend_t>, apple10, backend,
                    nk_dots_symmetric_u4_apple10);
    run_kernel_case("nk_angulars_packed_i4_apple10", test_angulars_packed<i4x2_t, metal_backend_t>, apple10,
                    nk_dots_pack_size_i4_apple10, nk_dots_pack_i4_apple10, nk_angulars_packed_i4_apple10);
    run_kernel_case("nk_angulars_symmetric_i4_apple10", test_angulars_symmetric<i4x2_t, metal_backend_t>, apple10,
                    nk_angulars_symmetric_i4_apple10);
    run_kernel_case("nk_euclideans_packed_i4_apple10", test_euclideans_packed<i4x2_t, metal_backend_t>, apple10,
                    nk_dots_pack_size_i4_apple10, nk_dots_pack_i4_apple10, nk_euclideans_packed_i4_apple10);
    run_kernel_case("nk_euclideans_symmetric_i4_apple10", test_euclideans_symmetric<i4x2_t, metal_backend_t>, apple10,
                    nk_euclideans_symmetric_i4_apple10);
    run_kernel_case("nk_angulars_packed_u4_apple10", test_angulars_packed<u4x2_t, metal_backend_t>, apple10,
                    nk_dots_pack_size_u4_apple10, nk_dots_pack_u4_apple10, nk_angulars_packed_u4_apple10);
    run_kernel_case("nk_angulars_symmetric_u4_apple10", test_angulars_symmetric<u4x2_t, metal_backend_t>, apple10,
                    nk_angulars_symmetric_u4_apple10);
    run_kernel_case("nk_euclideans_packed_u4_apple10", test_euclideans_packed<u4x2_t, metal_backend_t>, apple10,
                    nk_dots_pack_size_u4_apple10, nk_dots_pack_u4_apple10, nk_euclideans_packed_u4_apple10);
    run_kernel_case("nk_euclideans_symmetric_u4_apple10", test_euclideans_symmetric<u4x2_t, metal_backend_t>, apple10,
                    nk_euclideans_symmetric_u4_apple10);
    run_kernel_case("nk_dots_packed_i8_apple10", test_dots_packed<i8_t, metal_backend_t>, apple10, backend,
                    nk_dots_pack_size_i8_apple10, nk_dots_pack_i8_apple10, nk_dots_packed_i8_apple10);
    run_kernel_case("nk_dots_pack_i8_apple10",
                    test_dots_pack_layout<i8_t, metal_backend_t, nk_dots_pack_size_i8_apple10,
                                          nk_dots_packed_shape_i8_apple10, nk_dots_pack_i8_apple10>,
                    apple10, backend);
    run_kernel_case("test_dots_launch_contract<i8_t, metal_backend_t, nk_dots_pack_size_i8_apple10, " //
                    "nk_dots_packed_i8_apple10, nk_dots_symmetric_i8_apple10>(backend)",
                    test_dots_launch_contract<i8_t, metal_backend_t, nk_dots_pack_size_i8_apple10,
                                              nk_dots_packed_i8_apple10, nk_dots_symmetric_i8_apple10>,
                    apple10, backend);
    run_kernel_case("nk_dots_symmetric_i8_apple10", test_dots_symmetric<i8_t, metal_backend_t>, apple10, backend,
                    nk_dots_symmetric_i8_apple10);
    run_kernel_case("nk_dots_packed_u8_apple10", test_dots_packed<u8_t, metal_backend_t>, apple10, backend,
                    nk_dots_pack_size_u8_apple10, nk_dots_pack_u8_apple10, nk_dots_packed_u8_apple10);
    run_kernel_case("nk_dots_pack_u8_apple10",
                    test_dots_pack_layout<u8_t, metal_backend_t, nk_dots_pack_size_u8_apple10,
                                          nk_dots_packed_shape_u8_apple10, nk_dots_pack_u8_apple10>,
                    apple10, backend);
    run_kernel_case("test_dots_launch_contract<u8_t, metal_backend_t, nk_dots_pack_size_u8_apple10, " //
                    "nk_dots_packed_u8_apple10, nk_dots_symmetric_u8_apple10>(backend)",
                    test_dots_launch_contract<u8_t, metal_backend_t, nk_dots_pack_size_u8_apple10,
                                              nk_dots_packed_u8_apple10, nk_dots_symmetric_u8_apple10>,
                    apple10, backend);
    run_kernel_case("nk_dots_symmetric_u8_apple10", test_dots_symmetric<u8_t, metal_backend_t>, apple10, backend,
                    nk_dots_symmetric_u8_apple10);
    run_kernel_case("nk_dots_packed_f16_apple10", test_dots_packed<f16_t, metal_backend_t>, apple10, backend,
                    nk_dots_pack_size_f16_apple10, nk_dots_pack_f16_apple10, nk_dots_packed_f16_apple10);
    run_kernel_case("nk_dots_pack_f16_apple10",
                    test_dots_pack_layout<f16_t, metal_backend_t, nk_dots_pack_size_f16_apple10,
                                          nk_dots_packed_shape_f16_apple10, nk_dots_pack_f16_apple10>,
                    apple10, backend);
    run_kernel_case("test_dots_launch_contract<f16_t, metal_backend_t, nk_dots_pack_size_f16_apple10, " //
                    "nk_dots_packed_f16_apple10, nk_dots_symmetric_f16_apple10>(backend)",
                    test_dots_launch_contract<f16_t, metal_backend_t, nk_dots_pack_size_f16_apple10,
                                              nk_dots_packed_f16_apple10, nk_dots_symmetric_f16_apple10>,
                    apple10, backend);
    run_kernel_case("nk_dots_symmetric_f16_apple10", test_dots_symmetric<f16_t, metal_backend_t>, apple10, backend,
                    nk_dots_symmetric_f16_apple10);
    run_kernel_case("nk_dots_packed_bf16_apple10", test_dots_packed<bf16_t, metal_backend_t>, apple10, backend,
                    nk_dots_pack_size_bf16_apple10, nk_dots_pack_bf16_apple10, nk_dots_packed_bf16_apple10);
    run_kernel_case("nk_dots_pack_bf16_apple10",
                    test_dots_pack_layout<bf16_t, metal_backend_t, nk_dots_pack_size_bf16_apple10,
                                          nk_dots_packed_shape_bf16_apple10, nk_dots_pack_bf16_apple10>,
                    apple10, backend);
    run_kernel_case("test_dots_launch_contract<bf16_t, metal_backend_t, nk_dots_pack_size_bf16_apple10, " //
                    "nk_dots_packed_bf16_apple10, nk_dots_symmetric_bf16_apple10>(backend)",
                    test_dots_launch_contract<bf16_t, metal_backend_t, nk_dots_pack_size_bf16_apple10,
                                              nk_dots_packed_bf16_apple10, nk_dots_symmetric_bf16_apple10>,
                    apple10, backend);
    run_kernel_case("nk_dots_symmetric_bf16_apple10", test_dots_symmetric<bf16_t, metal_backend_t>, apple10, backend,
                    nk_dots_symmetric_bf16_apple10);
    run_kernel_case("nk_dots_packed_e4m3_apple10", test_dots_packed<e4m3_t, metal_backend_t>, apple10, backend,
                    nk_dots_pack_size_e4m3_apple10, nk_dots_pack_e4m3_apple10, nk_dots_packed_e4m3_apple10);
    run_kernel_case("nk_dots_pack_e4m3_apple10",
                    test_dots_pack_layout<e4m3_t, metal_backend_t, nk_dots_pack_size_e4m3_apple10,
                                          nk_dots_packed_shape_e4m3_apple10, nk_dots_pack_e4m3_apple10>,
                    apple10, backend);
    run_kernel_case("test_dots_launch_contract<e4m3_t, metal_backend_t, nk_dots_pack_size_e4m3_apple10, " //
                    "nk_dots_packed_e4m3_apple10, nk_dots_symmetric_e4m3_apple10>(backend)",
                    test_dots_launch_contract<e4m3_t, metal_backend_t, nk_dots_pack_size_e4m3_apple10,
                                              nk_dots_packed_e4m3_apple10, nk_dots_symmetric_e4m3_apple10>,
                    apple10, backend);
    run_kernel_case("nk_dots_symmetric_e4m3_apple10", test_dots_symmetric<e4m3_t, metal_backend_t>, apple10, backend,
                    nk_dots_symmetric_e4m3_apple10);
    run_kernel_case("nk_dots_packed_e5m2_apple10", test_dots_packed<e5m2_t, metal_backend_t>, apple10, backend,
                    nk_dots_pack_size_e5m2_apple10, nk_dots_pack_e5m2_apple10, nk_dots_packed_e5m2_apple10);
    run_kernel_case("nk_dots_pack_e5m2_apple10",
                    test_dots_pack_layout<e5m2_t, metal_backend_t, nk_dots_pack_size_e5m2_apple10,
                                          nk_dots_packed_shape_e5m2_apple10, nk_dots_pack_e5m2_apple10>,
                    apple10, backend);
    run_kernel_case("test_dots_launch_contract<e5m2_t, metal_backend_t, nk_dots_pack_size_e5m2_apple10, " //
                    "nk_dots_packed_e5m2_apple10, nk_dots_symmetric_e5m2_apple10>(backend)",
                    test_dots_launch_contract<e5m2_t, metal_backend_t, nk_dots_pack_size_e5m2_apple10,
                                              nk_dots_packed_e5m2_apple10, nk_dots_symmetric_e5m2_apple10>,
                    apple10, backend);
    run_kernel_case("nk_dots_symmetric_e5m2_apple10", test_dots_symmetric<e5m2_t, metal_backend_t>, apple10, backend,
                    nk_dots_symmetric_e5m2_apple10);
    run_kernel_case("nk_dots_packed_e3m2_apple10", test_dots_packed<e3m2_t, metal_backend_t>, apple10, backend,
                    nk_dots_pack_size_e3m2_apple10, nk_dots_pack_e3m2_apple10, nk_dots_packed_e3m2_apple10);
    run_kernel_case("nk_dots_pack_e3m2_apple10",
                    test_dots_pack_layout<e3m2_t, metal_backend_t, nk_dots_pack_size_e3m2_apple10,
                                          nk_dots_packed_shape_e3m2_apple10, nk_dots_pack_e3m2_apple10>,
                    apple10, backend);
    run_kernel_case("test_dots_launch_contract<e3m2_t, metal_backend_t, nk_dots_pack_size_e3m2_apple10, " //
                    "nk_dots_packed_e3m2_apple10, nk_dots_symmetric_e3m2_apple10>(backend)",
                    test_dots_launch_contract<e3m2_t, metal_backend_t, nk_dots_pack_size_e3m2_apple10,
                                              nk_dots_packed_e3m2_apple10, nk_dots_symmetric_e3m2_apple10>,
                    apple10, backend);
    run_kernel_case("nk_dots_symmetric_e3m2_apple10", test_dots_symmetric<e3m2_t, metal_backend_t>, apple10, backend,
                    nk_dots_symmetric_e3m2_apple10);
    run_kernel_case("nk_dots_packed_e2m3_apple10", test_dots_packed<e2m3_t, metal_backend_t>, apple10, backend,
                    nk_dots_pack_size_e2m3_apple10, nk_dots_pack_e2m3_apple10, nk_dots_packed_e2m3_apple10);
    run_kernel_case("nk_dots_pack_e2m3_apple10",
                    test_dots_pack_layout<e2m3_t, metal_backend_t, nk_dots_pack_size_e2m3_apple10,
                                          nk_dots_packed_shape_e2m3_apple10, nk_dots_pack_e2m3_apple10>,
                    apple10, backend);
    run_kernel_case("test_dots_launch_contract<e2m3_t, metal_backend_t, nk_dots_pack_size_e2m3_apple10, " //
                    "nk_dots_packed_e2m3_apple10, nk_dots_symmetric_e2m3_apple10>(backend)",
                    test_dots_launch_contract<e2m3_t, metal_backend_t, nk_dots_pack_size_e2m3_apple10,
                                              nk_dots_packed_e2m3_apple10, nk_dots_symmetric_e2m3_apple10>,
                    apple10, backend);
    run_kernel_case("nk_dots_symmetric_e2m3_apple10", test_dots_symmetric<e2m3_t, metal_backend_t>, apple10, backend,
                    nk_dots_symmetric_e2m3_apple10);
    run_kernel_case("nk_dots_packed_e2m1_apple10", test_dots_packed<e2m1x2_t, metal_backend_t>, apple10, backend,
                    nk_dots_pack_size_e2m1_apple10, nk_dots_pack_e2m1_apple10, nk_dots_packed_e2m1_apple10);
    run_kernel_case("nk_dots_pack_e2m1_apple10",
                    test_dots_pack_layout<e2m1x2_t, metal_backend_t, nk_dots_pack_size_e2m1_apple10,
                                          nk_dots_packed_shape_e2m1_apple10, nk_dots_pack_e2m1_apple10>,
                    apple10, backend);
    run_kernel_case("test_dots_launch_contract<e2m1x2_t, metal_backend_t, nk_dots_pack_size_e2m1_apple10, " //
                    "nk_dots_packed_e2m1_apple10, nk_dots_symmetric_e2m1_apple10>(backend)",
                    test_dots_launch_contract<e2m1x2_t, metal_backend_t, nk_dots_pack_size_e2m1_apple10,
                                              nk_dots_packed_e2m1_apple10, nk_dots_symmetric_e2m1_apple10>,
                    apple10, backend);
    run_kernel_case("nk_dots_symmetric_e2m1_apple10", test_dots_symmetric<e2m1x2_t, metal_backend_t>, apple10, backend,
                    nk_dots_symmetric_e2m1_apple10);
    run_kernel_case("nk_angulars_packed_i8_apple10", test_angulars_packed<i8_t, metal_backend_t>, apple10,
                    nk_dots_pack_size_i8_apple10, nk_dots_pack_i8_apple10, nk_angulars_packed_i8_apple10);
    run_kernel_case("nk_angulars_symmetric_i8_apple10", test_angulars_symmetric<i8_t, metal_backend_t>, apple10,
                    nk_angulars_symmetric_i8_apple10);
    run_kernel_case("nk_euclideans_packed_i8_apple10", test_euclideans_packed<i8_t, metal_backend_t>, apple10,
                    nk_dots_pack_size_i8_apple10, nk_dots_pack_i8_apple10, nk_euclideans_packed_i8_apple10);
    run_kernel_case("nk_euclideans_symmetric_i8_apple10", test_euclideans_symmetric<i8_t, metal_backend_t>, apple10,
                    nk_euclideans_symmetric_i8_apple10);
    run_kernel_case("nk_angulars_packed_u8_apple10", test_angulars_packed<u8_t, metal_backend_t>, apple10,
                    nk_dots_pack_size_u8_apple10, nk_dots_pack_u8_apple10, nk_angulars_packed_u8_apple10);
    run_kernel_case("nk_angulars_symmetric_u8_apple10", test_angulars_symmetric<u8_t, metal_backend_t>, apple10,
                    nk_angulars_symmetric_u8_apple10);
    run_kernel_case("nk_euclideans_packed_u8_apple10", test_euclideans_packed<u8_t, metal_backend_t>, apple10,
                    nk_dots_pack_size_u8_apple10, nk_dots_pack_u8_apple10, nk_euclideans_packed_u8_apple10);
    run_kernel_case("nk_euclideans_symmetric_u8_apple10", test_euclideans_symmetric<u8_t, metal_backend_t>, apple10,
                    nk_euclideans_symmetric_u8_apple10);
    run_kernel_case("nk_angulars_packed_f16_apple10", test_angulars_packed<f16_t, metal_backend_t>, apple10,
                    nk_dots_pack_size_f16_apple10, nk_dots_pack_f16_apple10, nk_angulars_packed_f16_apple10);
    run_kernel_case("nk_angulars_symmetric_f16_apple10", test_angulars_symmetric<f16_t, metal_backend_t>, apple10,
                    nk_angulars_symmetric_f16_apple10);
    run_kernel_case("nk_euclideans_packed_f16_apple10", test_euclideans_packed<f16_t, metal_backend_t>, apple10,
                    nk_dots_pack_size_f16_apple10, nk_dots_pack_f16_apple10, nk_euclideans_packed_f16_apple10);
    run_kernel_case("nk_euclideans_symmetric_f16_apple10", test_euclideans_symmetric<f16_t, metal_backend_t>, apple10,
                    nk_euclideans_symmetric_f16_apple10);
    run_kernel_case("nk_angulars_packed_bf16_apple10", test_angulars_packed<bf16_t, metal_backend_t>, apple10,
                    nk_dots_pack_size_bf16_apple10, nk_dots_pack_bf16_apple10, nk_angulars_packed_bf16_apple10);
    run_kernel_case("nk_angulars_symmetric_bf16_apple10", test_angulars_symmetric<bf16_t, metal_backend_t>, apple10,
                    nk_angulars_symmetric_bf16_apple10);
    run_kernel_case("nk_euclideans_packed_bf16_apple10", test_euclideans_packed<bf16_t, metal_backend_t>, apple10,
                    nk_dots_pack_size_bf16_apple10, nk_dots_pack_bf16_apple10, nk_euclideans_packed_bf16_apple10);
    run_kernel_case("nk_euclideans_symmetric_bf16_apple10", test_euclideans_symmetric<bf16_t, metal_backend_t>, apple10,
                    nk_euclideans_symmetric_bf16_apple10);
    run_kernel_case("nk_angulars_packed_e4m3_apple10", test_angulars_packed<e4m3_t, metal_backend_t>, apple10,
                    nk_dots_pack_size_e4m3_apple10, nk_dots_pack_e4m3_apple10, nk_angulars_packed_e4m3_apple10);
    run_kernel_case("nk_angulars_symmetric_e4m3_apple10", test_angulars_symmetric<e4m3_t, metal_backend_t>, apple10,
                    nk_angulars_symmetric_e4m3_apple10);
    run_kernel_case("nk_euclideans_packed_e4m3_apple10", test_euclideans_packed<e4m3_t, metal_backend_t>, apple10,
                    nk_dots_pack_size_e4m3_apple10, nk_dots_pack_e4m3_apple10, nk_euclideans_packed_e4m3_apple10);
    run_kernel_case("nk_euclideans_symmetric_e4m3_apple10", test_euclideans_symmetric<e4m3_t, metal_backend_t>, apple10,
                    nk_euclideans_symmetric_e4m3_apple10);
    run_kernel_case("nk_angulars_packed_e5m2_apple10", test_angulars_packed<e5m2_t, metal_backend_t>, apple10,
                    nk_dots_pack_size_e5m2_apple10, nk_dots_pack_e5m2_apple10, nk_angulars_packed_e5m2_apple10);
    run_kernel_case("nk_angulars_symmetric_e5m2_apple10", test_angulars_symmetric<e5m2_t, metal_backend_t>, apple10,
                    nk_angulars_symmetric_e5m2_apple10);
    run_kernel_case("nk_euclideans_packed_e5m2_apple10", test_euclideans_packed<e5m2_t, metal_backend_t>, apple10,
                    nk_dots_pack_size_e5m2_apple10, nk_dots_pack_e5m2_apple10, nk_euclideans_packed_e5m2_apple10);
    run_kernel_case("nk_euclideans_symmetric_e5m2_apple10", test_euclideans_symmetric<e5m2_t, metal_backend_t>, apple10,
                    nk_euclideans_symmetric_e5m2_apple10);
    run_kernel_case("nk_angulars_packed_e3m2_apple10", test_angulars_packed<e3m2_t, metal_backend_t>, apple10,
                    nk_dots_pack_size_e3m2_apple10, nk_dots_pack_e3m2_apple10, nk_angulars_packed_e3m2_apple10);
    run_kernel_case("nk_angulars_symmetric_e3m2_apple10", test_angulars_symmetric<e3m2_t, metal_backend_t>, apple10,
                    nk_angulars_symmetric_e3m2_apple10);
    run_kernel_case("nk_euclideans_packed_e3m2_apple10", test_euclideans_packed<e3m2_t, metal_backend_t>, apple10,
                    nk_dots_pack_size_e3m2_apple10, nk_dots_pack_e3m2_apple10, nk_euclideans_packed_e3m2_apple10);
    run_kernel_case("nk_euclideans_symmetric_e3m2_apple10", test_euclideans_symmetric<e3m2_t, metal_backend_t>, apple10,
                    nk_euclideans_symmetric_e3m2_apple10);
    run_kernel_case("nk_angulars_packed_e2m3_apple10", test_angulars_packed<e2m3_t, metal_backend_t>, apple10,
                    nk_dots_pack_size_e2m3_apple10, nk_dots_pack_e2m3_apple10, nk_angulars_packed_e2m3_apple10);
    run_kernel_case("nk_angulars_symmetric_e2m3_apple10", test_angulars_symmetric<e2m3_t, metal_backend_t>, apple10,
                    nk_angulars_symmetric_e2m3_apple10);
    run_kernel_case("nk_euclideans_packed_e2m3_apple10", test_euclideans_packed<e2m3_t, metal_backend_t>, apple10,
                    nk_dots_pack_size_e2m3_apple10, nk_dots_pack_e2m3_apple10, nk_euclideans_packed_e2m3_apple10);
    run_kernel_case("nk_euclideans_symmetric_e2m3_apple10", test_euclideans_symmetric<e2m3_t, metal_backend_t>, apple10,
                    nk_euclideans_symmetric_e2m3_apple10);
    run_kernel_case("nk_angulars_packed_e2m1_apple10", test_angulars_packed<e2m1x2_t, metal_backend_t>, apple10,
                    nk_dots_pack_size_e2m1_apple10, nk_dots_pack_e2m1_apple10, nk_angulars_packed_e2m1_apple10);
    run_kernel_case("nk_angulars_symmetric_e2m1_apple10", test_angulars_symmetric<e2m1x2_t, metal_backend_t>, apple10,
                    nk_angulars_symmetric_e2m1_apple10);
    run_kernel_case("nk_euclideans_packed_e2m1_apple10", test_euclideans_packed<e2m1x2_t, metal_backend_t>, apple10,
                    nk_dots_pack_size_e2m1_apple10, nk_dots_pack_e2m1_apple10, nk_euclideans_packed_e2m1_apple10);
    run_kernel_case("nk_euclideans_symmetric_e2m1_apple10", test_euclideans_symmetric<e2m1x2_t, metal_backend_t>,
                    apple10, nk_euclideans_symmetric_e2m1_apple10);
    run_kernel_case("nk_attention_packed_bf16_apple10",
                    test_attention_packed<bf16_t, metal_backend_t, attention_weights_t::bits_8_k>, apple10,
                    nk_attention_pack_size_bf16_apple10, nk_attention_pack_bf16_apple10,
                    nk_attention_packed_bf16_apple10);
    run_kernel_case("nk_attention_packed_f16_apple10",
                    test_attention_packed<f16_t, metal_backend_t, attention_weights_t::bits_11_k>, apple10,
                    nk_attention_pack_size_f16_apple10, nk_attention_pack_f16_apple10, nk_attention_packed_f16_apple10);
    run_kernel_case("nk_attention_packed_e4m3_apple10",
                    test_attention_packed<e4m3_t, metal_backend_t, attention_weights_t::bits_11_k>, apple10,
                    nk_attention_pack_size_e4m3_apple10, nk_attention_pack_e4m3_apple10,
                    nk_attention_packed_e4m3_apple10);
    run_kernel_case("nk_attention_packed_i8_apple10",
                    test_attention_packed<i8_t, metal_backend_t, attention_weights_t::bits_8_k>, apple10,
                    nk_attention_pack_size_i8_apple10, nk_attention_pack_i8_apple10, nk_attention_packed_i8_apple10);
    run_kernel_case("nk_attention_packed_nvfp4_apple10",
                    test_attention_packed<nvfp4_t, metal_backend_t, attention_weights_t::bits_11_k>, apple10,
                    nk_attention_pack_size_nvfp4_apple10, nk_attention_pack_nvfp4_apple10,
                    nk_attention_packed_nvfp4_apple10);
    run_kernel_case("nk_attention_packed_mxfp4_apple10",
                    test_attention_packed<mxfp4_t, metal_backend_t, attention_weights_t::bits_8_k>, apple10,
                    nk_attention_pack_size_mxfp4_apple10, nk_attention_pack_mxfp4_apple10,
                    nk_attention_packed_mxfp4_apple10);
    run_kernel_case("nk_attention_packed_mxfp6e2m3_apple10",
                    test_attention_packed<mxfp6e2m3_t, metal_backend_t, attention_weights_t::bits_8_k>, apple10,
                    nk_attention_pack_size_mxfp6e2m3_apple10, nk_attention_pack_mxfp6e2m3_apple10,
                    nk_attention_packed_mxfp6e2m3_apple10);
    run_kernel_case("nk_attention_packed_mxfp6e3m2_apple10",
                    test_attention_packed<mxfp6e3m2_t, metal_backend_t, attention_weights_t::bits_8_k>, apple10,
                    nk_attention_pack_size_mxfp6e3m2_apple10, nk_attention_pack_mxfp6e3m2_apple10,
                    nk_attention_packed_mxfp6e3m2_apple10);
    run_kernel_case("nk_attention_packed_mxfp8e4m3_apple10",
                    test_attention_packed<mxfp8e4m3_t, metal_backend_t, attention_weights_t::bits_8_k>, apple10,
                    nk_attention_pack_size_mxfp8e4m3_apple10, nk_attention_pack_mxfp8e4m3_apple10,
                    nk_attention_packed_mxfp8e4m3_apple10);
    run_kernel_case("nk_attention_packed_mxfp8e5m2_apple10",
                    test_attention_packed<mxfp8e5m2_t, metal_backend_t, attention_weights_t::bits_8_k>, apple10,
                    nk_attention_pack_size_mxfp8e5m2_apple10, nk_attention_pack_mxfp8e5m2_apple10,
                    nk_attention_packed_mxfp8e5m2_apple10);
    run_kernel_case("nk_attention_packed_gradients_bf16_apple10",
                    test_attention_packed_gradients<bf16_t, metal_backend_t>, apple10,
                    nk_attention_pack_size_bf16_apple10, nk_attention_pack_bf16_apple10,
                    nk_attention_packed_gradients_bf16_apple10);
    run_kernel_case("nk_rmsds_packed_f16_apple10", test_rmsds_packed<f16_t, metal_backend_t>, apple10,
                    nk_rmsds_packed_f16_apple10);
    run_kernel_case("nk_chamfers_packed_f16_apple10",
                    test_shapes_flips<f16_t, nk_kernel_chamfers_packed_k, metal_backend_t>, apple10,
                    nk_chamfers_packed_f16_apple10);
    run_kernel_case("nk_hausdorffs_packed_f16_apple10",
                    test_shapes_flips<f16_t, nk_kernel_hausdorffs_packed_k, metal_backend_t>, apple10,
                    nk_hausdorffs_packed_f16_apple10);
#endif // NUMKONG_TARGET_APPLE10
}

void run_cross_metal_cases() {
    run_cross_metal_baseline_cases();
    run_cross_apple9_cases();
    run_cross_apple10_cases();
    run_cross_dispatch_cases<metal_backend_t>();
}

#else  // !NUMKONG_ARCH_METAL_
void run_cross_metal_cases() {}
#endif // NUMKONG_ARCH_METAL_

} // namespace ashvardanian::numkong::test
