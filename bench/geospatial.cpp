/**
 *  @file bench/geospatial.cpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 *  @brief Many-to-many 3D squares, Haversine, Vincenty and terrain clearance benchmarks.
 */

#include "numkong/geospatial.h"

#include "harness.hpp"

namespace ashvardanian::numkong::bench {

using sqeuclideans3d_i32_kernel_t = nk_status_t (*)(nk_i32_t const *, nk_size_t, nk_i32_t const *, nk_size_t,
                                                    nk_u64_t *, nk_size_t, nk_stream_t);
using sqeuclideans3d_f64_kernel_t = nk_status_t (*)(nk_f64_t const *, nk_size_t, nk_f64_t const *, nk_size_t,
                                                    nk_f64_t *, nk_size_t, nk_stream_t);
using geodesics_packed_kernel_t = nk_status_t (*)(void const *, void const *, nk_f64_t *, nk_size_t, nk_stream_t);
using geodesics_pack_size_kernel_t = nk_status_t (*)(nk_size_t, nk_size_t *);
using geodesics_pack_kernel_t = nk_status_t (*)(nk_f64_t const *, nk_f64_t const *, nk_size_t, void *, nk_stream_t);
using clearances_packed_kernel_t = nk_status_t (*)(nk_f32_t const *, void const *, nk_f32_t const *, nk_f32_t const *,
                                                   nk_size_t, nk_f32_t const *, nk_size_t, nk_f64_t, nk_f32_t *,
                                                   nk_size_t, nk_stream_t);

/** Measures 3D squares between @p rows and @p columns Earth-fixed points. */
template <typename scalar_type_, typename result_type_, typename kernel_type_>
void measure_sqeuclideans3d(loop_t &loop, environment_t const &env, kernel_type_ kernel, std::size_t rows,
                            std::size_t columns) {
    std::mt19937 generator(env.settings.seed.value);
    std::uniform_real_distribution<double> coordinate(-6.4e8, 6.4e8);
    std::vector<scalar_type_> a(rows * 3), b(columns * 3);
    for (scalar_type_ &value : a) value = scalar_type_(coordinate(generator));
    for (scalar_type_ &value : b) value = scalar_type_(coordinate(generator));
    std::vector<result_type_> results(rows * columns);
    for (std::size_t call : loop) {
        nk_unused_(call);
        if (!succeeded(loop, kernel(a.data(), rows, b.data(), columns, results.data(), columns * sizeof(result_type_),
                                    nullptr)))
            break;
        do_not_optimize(results.data());
    }
    loop.rate("distances", static_cast<double>(rows * columns));
}

/** Measures a many-to-many geodesic kernel over packs of @p rows and @p columns points drawn
 *  anywhere on Earth. */
void measure_geodesics(loop_t &loop, environment_t const &env, geodesics_pack_size_kernel_t pack_size,
                       geodesics_pack_kernel_t pack, geodesics_packed_kernel_t kernel, std::size_t rows,
                       std::size_t columns) {
    std::mt19937 generator(env.settings.seed.value);
    std::vector<nk_f64_t> a_latitudes(rows), a_longitudes(rows), b_latitudes(columns), b_longitudes(columns);
    nk::fill_coordinates(generator, a_latitudes.data(), a_longitudes.data(), rows);
    nk::fill_coordinates(generator, b_latitudes.data(), b_longitudes.data(), columns);
    nk_size_t a_bytes = 0, b_bytes = 0;
    if (!succeeded(loop, pack_size(rows, &a_bytes)) || !succeeded(loop, pack_size(columns, &b_bytes))) return;
    auto a_packed = make_vector<char>(a_bytes), b_packed = make_vector<char>(b_bytes);
    if (!succeeded(loop, pack(a_latitudes.data(), a_longitudes.data(), rows, a_packed.raw_values_data(), nullptr)) ||
        !succeeded(loop, pack(b_latitudes.data(), b_longitudes.data(), columns, b_packed.raw_values_data(), nullptr)))
        return;
    std::vector<nk_f64_t> results(rows * columns);
    for (std::size_t call : loop) {
        nk_unused_(call);
        if (!succeeded(loop, kernel(a_packed.raw_values_data(), b_packed.raw_values_data(), results.data(),
                                    columns * sizeof(nk_f64_t), nullptr)))
            break;
        do_not_optimize(results.data());
    }
    loop.rate("distances", static_cast<double>(rows * columns));
}

/** Measures clearances of @p rows viewpoints to @p columns targets across a random-walk terrain of
 *  @p cells × @p cells, 30 m apart, 2 m above its ground. */
void measure_clearances(loop_t &loop, environment_t const &env, clearances_packed_kernel_t kernel, std::size_t rows,
                        std::size_t columns, std::size_t cells) {
    std::mt19937 generator(env.settings.seed.value);
    std::normal_distribution<float> step(0.0f, 4.0f);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    std::vector<nk_f32_t> heights(cells * cells);
    for (std::size_t row = 0; row != cells; ++row)
        for (std::size_t column = 0; column != cells; ++column) {
            float const north = row ? heights[(row - 1) * cells + column] : 100.0f;
            float const west = column ? heights[row * cells + column - 1] : north;
            heights[row * cells + column] = (north + west) / 2 + step(generator);
        }
    nk_size_t bytes = 0;
    if (!succeeded(loop, nk_terrain_pack_size_f32_serial(cells, cells, &bytes))) return;
    auto packed = make_vector<char>(bytes);
    if (!succeeded(loop, nk_terrain_pack_f32_serial(heights.data(), cells, cells, cells * sizeof(nk_f32_t), 30.0, 30.0,
                                                    packed.raw_values_data(), nullptr)))
        return;
    auto place = [&](std::vector<nk_f32_t> &points, std::size_t count) {
        points.resize(count * 3);
        for (std::size_t point = 0; point != count; ++point) {
            std::size_t const row = std::size_t(unit(generator) * float(cells - 1));
            std::size_t const column = std::size_t(unit(generator) * float(cells - 1));
            points[point * 3] = float(column), points[point * 3 + 1] = float(row);
            points[point * 3 + 2] = heights[row * cells + column] + 2.0f;
        }
    };
    std::vector<nk_f32_t> viewpoints, targets;
    place(viewpoints, rows), place(targets, columns);
    std::vector<nk_f32_t> results(rows * columns);
    for (std::size_t call : loop) {
        nk_unused_(call);
        if (!succeeded(
                loop, kernel(heights.data(), packed.raw_values_data(), viewpoints.data(), nullptr, rows, targets.data(),
                             columns, 1.0 / 7, results.data(), columns * sizeof(nk_f32_t), nullptr)))
            break;
        do_not_optimize(results.data());
    }
    loop.rate("clearances", static_cast<double>(rows * columns));
}

/** Runs the three point families of one capability on the configured matrix shape. */
void run_geospatial(environment_t const &env, std::string const &suffix, sqeuclideans3d_i32_kernel_t i32_kernel,
                    sqeuclideans3d_f64_kernel_t f64_kernel, geodesics_packed_kernel_t haversines,
                    geodesics_packed_kernel_t vincentys) {
    std::size_t const rows = env.settings.matrix_height, columns = env.settings.matrix_width;
    std::string const shape = "<" + std::to_string(rows) + "x" + std::to_string(columns) + ">";
    run_benchmark(env, "sqeuclideans3d_i32_" + suffix + shape,
                  measure_sqeuclideans3d<nk_i32_t, nk_u64_t, sqeuclideans3d_i32_kernel_t>, i32_kernel, rows, columns);
    run_benchmark(env, "sqeuclideans3d_f64_" + suffix + shape,
                  measure_sqeuclideans3d<nk_f64_t, nk_f64_t, sqeuclideans3d_f64_kernel_t>, f64_kernel, rows, columns);
    run_benchmark(env, "haversines_packed_f64_" + suffix + shape, measure_geodesics, nk_haversines_pack_size_f64_serial,
                  nk_haversines_pack_f64_serial, haversines, rows, columns);
    run_benchmark(env, "vincentys_packed_f64_" + suffix + shape, measure_geodesics, nk_vincentys_pack_size_f64_serial,
                  nk_vincentys_pack_f64_serial, vincentys, rows, columns);
}

void bench_geospatial(environment_t const &env) {
#if NUMKONG_TARGET_NEON
    if (section(env, "Geospatial Functions NEON", nk_cap_neon_k))
        run_geospatial(env, "neon", nk_sqeuclideans3d_i32_neon, nk_sqeuclideans3d_f64_neon,
                       nk_haversines_packed_f64_neon, nk_vincentys_packed_f64_neon);
#endif
#if NUMKONG_TARGET_HASWELL
    if (section(env, "Geospatial Functions Haswell", nk_cap_haswell_k))
        run_geospatial(env, "haswell", nk_sqeuclideans3d_i32_haswell, nk_sqeuclideans3d_f64_haswell,
                       nk_haversines_packed_f64_haswell, nk_vincentys_packed_f64_haswell);
#endif
#if NUMKONG_TARGET_SKYLAKE
    if (section(env, "Geospatial Functions Skylake", nk_cap_skylake_k))
        run_geospatial(env, "skylake", nk_sqeuclideans3d_i32_skylake, nk_sqeuclideans3d_f64_skylake,
                       nk_haversines_packed_f64_skylake, nk_vincentys_packed_f64_skylake);
#endif
#if NUMKONG_TARGET_RVV
    if (section(env, "Geospatial Functions RVV", nk_cap_rvv_k))
        run_geospatial(env, "rvv", nk_sqeuclideans3d_i32_rvv, nk_sqeuclideans3d_f64_rvv, nk_haversines_packed_f64_rvv,
                       nk_vincentys_packed_f64_rvv);
#endif
#if NUMKONG_TARGET_V128RELAXED
    if (section(env, "Geospatial Functions V128 Relaxed", nk_cap_v128relaxed_k))
        run_geospatial(env, "v128relaxed", nk_sqeuclideans3d_i32_v128relaxed, nk_sqeuclideans3d_f64_v128relaxed,
                       nk_haversines_packed_f64_v128relaxed, nk_vincentys_packed_f64_v128relaxed);
#endif

    section(env, "Geospatial Functions Serial", nk_cap_serial_k);
    run_geospatial(env, "serial", nk_sqeuclideans3d_i32_serial, nk_sqeuclideans3d_f64_serial,
                   nk_haversines_packed_f64_serial, nk_vincentys_packed_f64_serial);
    run_benchmark(env, "clearances_packed_f32_serial<16x64,1024²>", measure_clearances, nk_clearances_packed_f32_serial,
                  std::size_t(16), std::size_t(64), std::size_t(1024));
}

} // namespace ashvardanian::numkong::bench
