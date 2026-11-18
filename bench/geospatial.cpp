/**
 *  @file bench/geospatial.cpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 *  @brief Many-to-many 3D squares, Haversine, Vincenty and terrain clearance benchmarks.
 */

#include <fmt/format.h> // `fmt::format`

#include "numkong/geospatial.hpp" // `nk::haversines_pack`, `nk::terrain_pack`

#include "harness.hpp"

namespace ashvardanian::numkong::bench {

using sqeuclideans3d_i32_kernel_t = nk_status_t (*)(nk_i32_t const *, nk_size_t, nk_i32_t const *, nk_size_t,
                                                    nk_u64_t *, nk_size_t, nk_stream_t);
using sqeuclideans3d_f64_kernel_t = nk_status_t (*)(nk_f64_t const *, nk_size_t, nk_f64_t const *, nk_size_t,
                                                    nk_f64_t *, nk_size_t, nk_stream_t);
using geodesics_packed_kernel_t = nk_status_t (*)(void const *, void const *, nk_f64_t *, nk_size_t, nk_stream_t);
using clearances_packed_kernel_t = nk_status_t (*)(nk_f32_t const *, void const *, nk_f32_t const *, nk_f32_t const *,
                                                   nk_size_t, nk_f32_t const *, nk_size_t, nk_f64_t, nk_f32_t *,
                                                   nk_size_t, nk_stream_t);

/** Measures 3D squares between @p rows and @p columns Earth-fixed points. */
template <typename scalar_type_, typename result_type_, typename kernel_type_>
void measure_sqeuclideans3d(loop_t &loop, bench_settings_t const &env, kernel_type_ kernel, std::size_t rows,
                            std::size_t columns) {
    split_mix_generator_t generator(env.seed);
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

/** Measures a many-to-many geodesic kernel of @p kind_ over packs of @p rows and @p columns points
 *  drawn anywhere on Earth, packed through the public dispatch. */
template <nk_kernel_kind_t kind_>
void measure_geodesics(loop_t &loop, bench_settings_t const &env, geodesics_packed_kernel_t kernel, std::size_t rows,
                       std::size_t columns) {
    constexpr bool spherical = kind_ == nk_kernel_haversines_packed_k;
    auto const pack_size = spherical ? nk::haversines_pack_size : nk::vincentys_pack_size;
    auto const pack = spherical ? nk::haversines_pack<> : nk::vincentys_pack<>;
    nk_capability_t const capabilities = nk::default_capabilities();
    split_mix_generator_t generator(env.seed);
    std::vector<f64_t> a_latitudes(rows), a_longitudes(rows), b_latitudes(columns), b_longitudes(columns);
    nk::fill_coordinates(generator, a_latitudes.data(), a_longitudes.data(), rows);
    nk::fill_coordinates(generator, b_latitudes.data(), b_longitudes.data(), columns);
    auto a_packed = make_vector<std::byte>(pack_size(rows, capabilities).value);
    auto b_packed = make_vector<std::byte>(pack_size(columns, capabilities).value);
    nk::status_t const a_status = pack(a_latitudes.data(), a_longitudes.data(), rows, a_packed.raw_values_data(),
                                       capabilities, nullptr);
    if (!nk::succeeded(a_status)) return loop.skip(nk::status_name(a_status));
    nk::status_t const b_status = pack(b_latitudes.data(), b_longitudes.data(), columns, b_packed.raw_values_data(),
                                       capabilities, nullptr);
    if (!nk::succeeded(b_status)) return loop.skip(nk::status_name(b_status));
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
void measure_clearances(loop_t &loop, bench_settings_t const &env, clearances_packed_kernel_t kernel, std::size_t rows,
                        std::size_t columns, std::size_t cells) {
    split_mix_generator_t generator(env.seed);
    std::normal_distribution<float> step(0.0f, 4.0f);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    auto heights = make_vector<f32_t>(cells * cells);
    nk_f32_t *terrain = heights.raw_values_data();
    for (std::size_t row = 0; row != cells; ++row)
        for (std::size_t column = 0; column != cells; ++column) {
            float const north = row ? terrain[(row - 1) * cells + column] : 100.0f;
            float const west = column ? terrain[row * cells + column - 1] : north;
            terrain[row * cells + column] = (north + west) / 2 + step(generator);
        }
    auto packed = make_vector<std::byte>(nk::terrain_pack_size(cells, cells).value);
    nk::status_t const pack_status = nk::terrain_pack(heights.values_data(), cells, cells, cells * sizeof(nk_f32_t),
                                                      30.0, 30.0, packed.raw_values_data());
    if (!nk::succeeded(pack_status)) return loop.skip(nk::status_name(pack_status));
    auto place = [&](std::vector<nk_f32_t> &points, std::size_t count) {
        points.resize(count * 3);
        for (std::size_t point = 0; point != count; ++point) {
            std::size_t const row = std::size_t(unit(generator) * float(cells - 1));
            std::size_t const column = std::size_t(unit(generator) * float(cells - 1));
            points[point * 3] = float(column), points[point * 3 + 1] = float(row);
            points[point * 3 + 2] = terrain[row * cells + column] + 2.0f;
        }
    };
    std::vector<nk_f32_t> viewpoints, targets;
    place(viewpoints, rows), place(targets, columns);
    std::vector<nk_f32_t> results(rows * columns);
    for (std::size_t call : loop) {
        nk_unused_(call);
        if (!succeeded(loop, kernel(terrain, packed.raw_values_data(), viewpoints.data(), nullptr, rows, targets.data(),
                                    columns, 1.0 / 7, results.data(), columns * sizeof(nk_f32_t), nullptr)))
            break;
        do_not_optimize(results.data());
    }
    loop.rate("clearances", static_cast<double>(rows * columns));
}

/** Runs the three point families of one capability on the configured matrix shape. */
void run_geospatial(bench_settings_t const &env, std::string const &suffix, sqeuclideans3d_i32_kernel_t i32_kernel,
                    sqeuclideans3d_f64_kernel_t f64_kernel, geodesics_packed_kernel_t haversines,
                    geodesics_packed_kernel_t vincentys) {
    std::size_t const rows = env.matrix_height, columns = env.matrix_width;
    run_benchmark(env, fmt::format("sqeuclideans3d_i32_{}<{}x{}>", suffix, rows, columns),
                  measure_sqeuclideans3d<nk_i32_t, nk_u64_t, sqeuclideans3d_i32_kernel_t>, i32_kernel, rows, columns);
    run_benchmark(env, fmt::format("sqeuclideans3d_f64_{}<{}x{}>", suffix, rows, columns),
                  measure_sqeuclideans3d<nk_f64_t, nk_f64_t, sqeuclideans3d_f64_kernel_t>, f64_kernel, rows, columns);
    run_benchmark(env, fmt::format("haversines_packed_f64_{}<{}x{}>", suffix, rows, columns),
                  measure_geodesics<nk_kernel_haversines_packed_k>, haversines, rows, columns);
    run_benchmark(env, fmt::format("vincentys_packed_f64_{}<{}x{}>", suffix, rows, columns),
                  measure_geodesics<nk_kernel_vincentys_packed_k>, vincentys, rows, columns);
}

void bench_geospatial(bench_settings_t const &env) {
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
#if NUMKONG_TARGET_POWERVSX
    if (section(env, "Geospatial Functions Power VSX", nk_cap_powervsx_k))
        run_geospatial(env, "powervsx", nk_sqeuclideans3d_i32_powervsx, nk_sqeuclideans3d_f64_powervsx, nullptr,
                       nullptr);
#endif

    section(env, "Geospatial Functions Serial", nk_cap_serial_k);
    run_geospatial(env, "serial", nk_sqeuclideans3d_i32_serial, nk_sqeuclideans3d_f64_serial,
                   nk_haversines_packed_f64_serial, nk_vincentys_packed_f64_serial);
    run_benchmark(env, "clearances_packed_f32_serial<16x64,1024²>", measure_clearances, nk_clearances_packed_f32_serial,
                  std::size_t(16), std::size_t(64), std::size_t(1024));
}

} // namespace ashvardanian::numkong::bench
