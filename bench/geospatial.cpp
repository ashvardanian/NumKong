/**
 *  @file bench/geospatial.cpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 *  @brief Geospatial distance benchmarks, haversine, vincenty.
 */

#include "numkong/geospatial.h"

#include "harness.hpp"

namespace ashvardanian::numkong::bench {

/**
 *  @brief Measures geospatial operations, Haversine or Vincenty.
 *  @param[inout] loop The timed loop, which takes the counters.
 *  @param[in] kernel The kernel function to benchmark.
 *  @param[in] coordinates_count The number of coordinate pairs to process.
 */
template <nk_dtype_t dtype_, typename kernel_type_ = void>
void measure_geospatial(loop_t &loop, environment_t const &env, kernel_type_ kernel, std::size_t coordinates_count) {

    using scalar_t = typename nk::type_for<dtype_>::type;
    using vector_t = nk::vector<scalar_t>;

    // Preallocate coordinate arrays: latitude1, longitude1, latitude2, longitude2
    std::size_t const batches_count = input_sets_count(dtype_bytes(dtype_, 4 * coordinates_count));
    std::vector<vector_t> latitudes_first(batches_count), longitudes_first(batches_count);
    std::vector<vector_t> latitudes_second(batches_count), longitudes_second(batches_count);
    std::mt19937 generator(env.settings.seed.value);
    double const max_separation_rad = double(env.settings.max_coord_angle_degrees) * 3.14159265358979323846 / 180.0;
    for (std::size_t index = 0; index != batches_count; ++index) {
        latitudes_first[index] = make_vector<scalar_t>(coordinates_count);
        longitudes_first[index] = make_vector<scalar_t>(coordinates_count);
        latitudes_second[index] = make_vector<scalar_t>(coordinates_count);
        longitudes_second[index] = make_vector<scalar_t>(coordinates_count);
        nk::fill_coordinates(generator, latitudes_first[index].values_data(), longitudes_first[index].values_data(),
                             coordinates_count);
        nk::fill_nearby_coordinates(generator, latitudes_first[index].values_data(),
                                    longitudes_first[index].values_data(), latitudes_second[index].values_data(),
                                    longitudes_second[index].values_data(), coordinates_count, max_separation_rad);
    }

    // Output distances buffer
    vector_t distances = make_vector<scalar_t>(coordinates_count);

    // Benchmark loop
    for (std::size_t call : loop) {
        std::size_t const index = call & (batches_count - 1);
        if (!succeeded(loop,
                       kernel(latitudes_first[index].raw_values_data(), longitudes_first[index].raw_values_data(),
                              latitudes_second[index].raw_values_data(), longitudes_second[index].raw_values_data(),
                              coordinates_count, distances.raw_values_data(), nullptr)))
            break;
        do_not_optimize(distances.raw_values_data());
    }

    loop.rate("ops", coordinates_count);
}

template <nk_dtype_t dtype_, typename kernel_type_ = void>
void run_geospatial(environment_t const &env, std::string name, kernel_type_ *kernel) {
    std::string bench_name = name + "<" + std::to_string(env.settings.batch_per_core) + "d," +
                             std::to_string(static_cast<int>(env.settings.max_coord_angle_degrees)) + "°>";
    run_benchmark(env, bench_name, measure_geospatial<dtype_, kernel_type_ *>, kernel, env.settings.batch_per_core);
}

void bench_geospatial(environment_t const &env) {
    constexpr nk_dtype_t f64_k = nk_f64_k;
    constexpr nk_dtype_t f32_k = nk_f32_k;

#if NUMKONG_TARGET_NEON
    if (section(env, "Geospatial Functions NEON", nk_cap_neon_k)) {
        run_geospatial<f32_k>(env, "haversine_f32_neon", nk_haversine_f32_neon);
        run_geospatial<f64_k>(env, "haversine_f64_neon", nk_haversine_f64_neon);
        run_geospatial<f32_k>(env, "vincenty_f32_neon", nk_vincenty_f32_neon);
        run_geospatial<f64_k>(env, "vincenty_f64_neon", nk_vincenty_f64_neon);
    }
#endif

#if NUMKONG_TARGET_HASWELL
    if (section(env, "Geospatial Functions Haswell", nk_cap_haswell_k)) {
        run_geospatial<f32_k>(env, "haversine_f32_haswell", nk_haversine_f32_haswell);
        run_geospatial<f64_k>(env, "haversine_f64_haswell", nk_haversine_f64_haswell);
        run_geospatial<f32_k>(env, "vincenty_f32_haswell", nk_vincenty_f32_haswell);
        run_geospatial<f64_k>(env, "vincenty_f64_haswell", nk_vincenty_f64_haswell);
    }
#endif

#if NUMKONG_TARGET_SKYLAKE
    if (section(env, "Geospatial Functions Skylake", nk_cap_skylake_k)) {
        run_geospatial<f32_k>(env, "haversine_f32_skylake", nk_haversine_f32_skylake);
        run_geospatial<f64_k>(env, "haversine_f64_skylake", nk_haversine_f64_skylake);
        run_geospatial<f32_k>(env, "vincenty_f32_skylake", nk_vincenty_f32_skylake);
        run_geospatial<f64_k>(env, "vincenty_f64_skylake", nk_vincenty_f64_skylake);
    }
#endif

#if NUMKONG_TARGET_RVV
    if (section(env, "Geospatial Functions RVV", nk_cap_rvv_k)) {
        run_geospatial<f32_k>(env, "haversine_f32_rvv", nk_haversine_f32_rvv);
        run_geospatial<f64_k>(env, "haversine_f64_rvv", nk_haversine_f64_rvv);
        run_geospatial<f32_k>(env, "vincenty_f32_rvv", nk_vincenty_f32_rvv);
        run_geospatial<f64_k>(env, "vincenty_f64_rvv", nk_vincenty_f64_rvv);
    }
#endif

#if NUMKONG_TARGET_V128RELAXED
    if (section(env, "Geospatial Functions V128 Relaxed", nk_cap_v128relaxed_k)) {
        run_geospatial<f32_k>(env, "haversine_f32_v128relaxed", nk_haversine_f32_v128relaxed);
        run_geospatial<f64_k>(env, "haversine_f64_v128relaxed", nk_haversine_f64_v128relaxed);
        run_geospatial<f32_k>(env, "vincenty_f32_v128relaxed", nk_vincenty_f32_v128relaxed);
        run_geospatial<f64_k>(env, "vincenty_f64_v128relaxed", nk_vincenty_f64_v128relaxed);
    }
#endif

    // Serial fallbacks
    section(env, "Geospatial Functions Serial", nk_cap_serial_k);
    run_geospatial<f32_k>(env, "haversine_f32_serial", nk_haversine_f32_serial);
    run_geospatial<f64_k>(env, "haversine_f64_serial", nk_haversine_f64_serial);
    run_geospatial<f32_k>(env, "vincenty_f32_serial", nk_vincenty_f32_serial);
    run_geospatial<f64_k>(env, "vincenty_f64_serial", nk_vincenty_f64_serial);
}

} // namespace ashvardanian::numkong::bench
