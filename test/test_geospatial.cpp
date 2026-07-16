/**
 *  @brief Haversine and Vincenty distance tests.
 *  @file test/test_geospatial.cpp
 *  @author Ash Vardanian
 *  @date December 28, 2025
 */

#include "test.hpp"
#include "numkong/geospatial.hpp" // `nk::haversine`

/**
 *  @brief Test Haversine distance.
 */
template <typename scalar_type_>
error_stats_t test_haversine(typename scalar_type_::geospatial_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using reference_t = reference_for<scalar_t>;

    error_stats_t ulp_stats(comparison_family_t::geospatial_k), abs_stats(comparison_family_t::geospatial_k);
    std::mt19937 generator(global_config.seed);
    auto a_lats = make_vector<scalar_t>(global_config.dense_dimensions),
         a_lons = make_vector<scalar_t>(global_config.dense_dimensions);
    auto b_lats = make_vector<scalar_t>(global_config.dense_dimensions),
         b_lons = make_vector<scalar_t>(global_config.dense_dimensions);
    auto results = make_vector<scalar_t>(global_config.dense_dimensions),
         haversine_ref = make_vector<scalar_t>(global_config.dense_dimensions),
         vincenty_ref = make_vector<scalar_t>(global_config.dense_dimensions);

    double const max_separation_rad = double(global_config.max_coord_angle) * 3.14159265358979323846 / 180.0;
    for (auto start = test_start_time(); within_time_budget(start);) {
        nk::fill_coordinates(generator, a_lats.values_data(), a_lons.values_data(), global_config.dense_dimensions);
        nk::fill_nearby_coordinates(generator, a_lats.values_data(), a_lons.values_data(), b_lats.values_data(),
                                    b_lons.values_data(), global_config.dense_dimensions, max_separation_rad);

        kernel(a_lats.raw_values_data(), a_lons.raw_values_data(), b_lats.raw_values_data(), b_lons.raw_values_data(),
               global_config.dense_dimensions, results.raw_values_data());
        nk::haversine<scalar_t, reference_t, nk::no_simd_k>(
            a_lats.values_data(), a_lons.values_data(), b_lats.values_data(), b_lons.values_data(),
            global_config.dense_dimensions, haversine_ref.values_data());
        nk::vincenty<scalar_t, reference_t, nk::no_simd_k>(a_lats.values_data(), a_lons.values_data(),
                                                           b_lats.values_data(), b_lons.values_data(),
                                                           global_config.dense_dimensions, vincenty_ref.values_data());

        for (std::size_t i = 0; i < global_config.dense_dimensions; i++) {
            ulp_stats.accumulate(results[i], haversine_ref[i]);
            abs_stats.accumulate(results[i], vincenty_ref[i]);
        }
    }

    // ULP measures implementation precision (vs haversine f118),
    // abs/rel measures formula accuracy in meters/ratio (vs vincenty f118)
    error_stats_t combined(comparison_family_t::geospatial_k);
    combined.min_ulp = ulp_stats.min_ulp;
    combined.max_ulp = ulp_stats.max_ulp;
    combined.sum_ulp = ulp_stats.sum_ulp;
    combined.count = ulp_stats.count;
    combined.exact_matches = ulp_stats.exact_matches;
    combined.min_abs_err = abs_stats.min_abs_err;
    combined.max_abs_err = abs_stats.max_abs_err;
    combined.sum_abs_err = abs_stats.sum_abs_err;
    combined.min_rel_err = abs_stats.min_rel_err;
    combined.max_rel_err = abs_stats.max_rel_err;
    combined.sum_rel_err = abs_stats.sum_rel_err;
    return combined;
}

/**
 *  @brief Test Vincenty distance.
 */
template <typename scalar_type_>
error_stats_t test_vincenty(typename scalar_type_::geospatial_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using reference_t = reference_for<scalar_t>;

    error_stats_t stats(comparison_family_t::geospatial_k);
    std::mt19937 generator(global_config.seed);
    auto a_lats = make_vector<scalar_t>(global_config.dense_dimensions),
         a_lons = make_vector<scalar_t>(global_config.dense_dimensions);
    auto b_lats = make_vector<scalar_t>(global_config.dense_dimensions),
         b_lons = make_vector<scalar_t>(global_config.dense_dimensions);
    auto results = make_vector<scalar_t>(global_config.dense_dimensions),
         reference = make_vector<scalar_t>(global_config.dense_dimensions);

    double const max_separation_rad = double(global_config.max_coord_angle) * 3.14159265358979323846 / 180.0;
    for (auto start = test_start_time(); within_time_budget(start);) {
        nk::fill_coordinates(generator, a_lats.values_data(), a_lons.values_data(), global_config.dense_dimensions);
        nk::fill_nearby_coordinates(generator, a_lats.values_data(), a_lons.values_data(), b_lats.values_data(),
                                    b_lons.values_data(), global_config.dense_dimensions, max_separation_rad);

        kernel(a_lats.raw_values_data(), a_lons.raw_values_data(), b_lats.raw_values_data(), b_lons.raw_values_data(),
               global_config.dense_dimensions, results.raw_values_data());
        nk::vincenty<scalar_t, reference_t, nk::no_simd_k>(a_lats.values_data(), a_lons.values_data(),
                                                           b_lats.values_data(), b_lons.values_data(),
                                                           global_config.dense_dimensions, reference.values_data());

        for (std::size_t i = 0; i < global_config.dense_dimensions; i++) stats.accumulate(results[i], reference[i]);
    }
    return stats;
}

void test_geospatial() {
    error_stats_section_t check("Geospatial Functions");

#if NK_RUNTIME_DISPATCH
    check("haversine_f64", test_haversine<f64_t>, nk_haversine_f64);
    check("haversine_f32", test_haversine<f32_t>, nk_haversine_f32);
    check("vincenty_f64", test_vincenty<f64_t>, nk_vincenty_f64);
    check("vincenty_f32", test_vincenty<f32_t>, nk_vincenty_f32);
#else

#if NK_TARGET_NEON
    check("haversine_f64_neon", test_haversine<f64_t>, nk_haversine_f64_neon);
    check("haversine_f32_neon", test_haversine<f32_t>, nk_haversine_f32_neon);
    check("vincenty_f64_neon", test_vincenty<f64_t>, nk_vincenty_f64_neon);
    check("vincenty_f32_neon", test_vincenty<f32_t>, nk_vincenty_f32_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_HASWELL
    check("haversine_f64_haswell", test_haversine<f64_t>, nk_haversine_f64_haswell);
    check("haversine_f32_haswell", test_haversine<f32_t>, nk_haversine_f32_haswell);
    check("vincenty_f64_haswell", test_vincenty<f64_t>, nk_vincenty_f64_haswell);
    check("vincenty_f32_haswell", test_vincenty<f32_t>, nk_vincenty_f32_haswell);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
    check("haversine_f64_skylake", test_haversine<f64_t>, nk_haversine_f64_skylake);
    check("haversine_f32_skylake", test_haversine<f32_t>, nk_haversine_f32_skylake);
    check("vincenty_f64_skylake", test_vincenty<f64_t>, nk_vincenty_f64_skylake);
    check("vincenty_f32_skylake", test_vincenty<f32_t>, nk_vincenty_f32_skylake);
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_RVV
    check("haversine_f64_rvv", test_haversine<f64_t>, nk_haversine_f64_rvv);
    check("haversine_f32_rvv", test_haversine<f32_t>, nk_haversine_f32_rvv);
    check("vincenty_f64_rvv", test_vincenty<f64_t>, nk_vincenty_f64_rvv);
    check("vincenty_f32_rvv", test_vincenty<f32_t>, nk_vincenty_f32_rvv);
#endif // NK_TARGET_RVV

#if NK_TARGET_V128RELAXED
    check("haversine_f64_v128relaxed", test_haversine<f64_t>, nk_haversine_f64_v128relaxed);
    check("haversine_f32_v128relaxed", test_haversine<f32_t>, nk_haversine_f32_v128relaxed);
    check("vincenty_f64_v128relaxed", test_vincenty<f64_t>, nk_vincenty_f64_v128relaxed);
    check("vincenty_f32_v128relaxed", test_vincenty<f32_t>, nk_vincenty_f32_v128relaxed);
#endif // NK_TARGET_V128RELAXED

    // Serial always runs - baseline test
    check("haversine_f64_serial", test_haversine<f64_t>, nk_haversine_f64_serial);
    check("haversine_f32_serial", test_haversine<f32_t>, nk_haversine_f32_serial);
    check("vincenty_f64_serial", test_vincenty<f64_t>, nk_vincenty_f64_serial);
    check("vincenty_f32_serial", test_vincenty<f32_t>, nk_vincenty_f32_serial);

#endif // NK_RUNTIME_DISPATCH
}
