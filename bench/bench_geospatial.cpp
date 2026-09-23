/**
 *  @brief Geospatial distance benchmarks (haversine, vincenty).
 *  @file bench/bench_geospatial.cpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 */

#include "numkong/geospatial.h"

#include "bench.hpp"

using namespace ashvardanian::numkong::bench;

/**
 *  @brief Measures the performance of geospatial operations (Haversine/Vincenty) using Google Benchmark.
 *  @param state The benchmark state object provided by Google Benchmark.
 *  @param kernel The kernel function to benchmark.
 *  @param coordinates_count The number of coordinate pairs to process.
 */
template <nk_dtype_t dtype_, typename kernel_type_ = void>
void measure_geospatial(bm::State &state, kernel_type_ kernel, std::size_t coordinates_count) {

    using scalar_t = typename nk::type_for<dtype_>::type;
    using vector_t = nk::vector<scalar_t>;

    // Preallocate coordinate arrays: latitude1, longitude1, latitude2, longitude2
    constexpr std::size_t batches_count = 1024;
    std::vector<vector_t> latitudes_first(batches_count), longitudes_first(batches_count);
    std::vector<vector_t> latitudes_second(batches_count), longitudes_second(batches_count);
    auto generator = make_random_engine();
    double const max_separation_rad = double(bench_config.max_coord_angle) * 3.14159265358979323846 / 180.0;
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
    std::size_t iterations = 0;
    for (auto _ : state) {
        std::size_t const index = iterations & (batches_count - 1);
        kernel(latitudes_first[index].raw_values_data(), longitudes_first[index].raw_values_data(),
               latitudes_second[index].raw_values_data(), longitudes_second[index].raw_values_data(), coordinates_count,
               distances.raw_values_data());
        bm::ClobberMemory();
        iterations++;
    }

    state.counters["ops"] = bm::Counter(1.0 * iterations * coordinates_count, bm::Counter::kIsRate);
    state.counters["calls"] = bm::Counter(iterations, bm::Counter::kIsRate);
}

template <nk_dtype_t dtype_, typename kernel_type_ = void>
void run_geospatial(std::string name, kernel_type_ *kernel) {
    std::string bench_name = name + "<" + std::to_string(bench_config.dense_dimensions) + "d," +
                             std::to_string(static_cast<int>(bench_config.max_coord_angle)) + "°>";
    bm::RegisterBenchmark(bench_name.c_str(), measure_geospatial<dtype_, kernel_type_ *>, kernel,
                          bench_config.dense_dimensions);
}

void bench_geospatial() {
    constexpr nk_dtype_t f64_k = nk_f64_k;
    constexpr nk_dtype_t f32_k = nk_f32_k;

#if NK_TARGET_NEON
    run_geospatial<f32_k>("haversine_f32_neon", nk_haversine_f32_neon);
    run_geospatial<f64_k>("haversine_f64_neon", nk_haversine_f64_neon);
    run_geospatial<f32_k>("vincenty_f32_neon", nk_vincenty_f32_neon);
    run_geospatial<f64_k>("vincenty_f64_neon", nk_vincenty_f64_neon);
#endif

#if NK_TARGET_HASWELL
    run_geospatial<f32_k>("haversine_f32_haswell", nk_haversine_f32_haswell);
    run_geospatial<f64_k>("haversine_f64_haswell", nk_haversine_f64_haswell);
    run_geospatial<f32_k>("vincenty_f32_haswell", nk_vincenty_f32_haswell);
    run_geospatial<f64_k>("vincenty_f64_haswell", nk_vincenty_f64_haswell);
#endif

#if NK_TARGET_SKYLAKE
    run_geospatial<f32_k>("haversine_f32_skylake", nk_haversine_f32_skylake);
    run_geospatial<f64_k>("haversine_f64_skylake", nk_haversine_f64_skylake);
    run_geospatial<f32_k>("vincenty_f32_skylake", nk_vincenty_f32_skylake);
    run_geospatial<f64_k>("vincenty_f64_skylake", nk_vincenty_f64_skylake);
#endif

#if NK_TARGET_RVV
    run_geospatial<f32_k>("haversine_f32_rvv", nk_haversine_f32_rvv);
    run_geospatial<f64_k>("haversine_f64_rvv", nk_haversine_f64_rvv);
    run_geospatial<f32_k>("vincenty_f32_rvv", nk_vincenty_f32_rvv);
    run_geospatial<f64_k>("vincenty_f64_rvv", nk_vincenty_f64_rvv);
#endif

#if NK_TARGET_V128RELAXED
    run_geospatial<f32_k>("haversine_f32_v128relaxed", nk_haversine_f32_v128relaxed);
    run_geospatial<f64_k>("haversine_f64_v128relaxed", nk_haversine_f64_v128relaxed);
    run_geospatial<f32_k>("vincenty_f32_v128relaxed", nk_vincenty_f32_v128relaxed);
    run_geospatial<f64_k>("vincenty_f64_v128relaxed", nk_vincenty_f64_v128relaxed);
#endif

    // Serial fallbacks
    run_geospatial<f32_k>("haversine_f32_serial", nk_haversine_f32_serial);
    run_geospatial<f64_k>("haversine_f64_serial", nk_haversine_f64_serial);
    run_geospatial<f32_k>("vincenty_f32_serial", nk_vincenty_f32_serial);
    run_geospatial<f64_k>("vincenty_f64_serial", nk_vincenty_f64_serial);
}
