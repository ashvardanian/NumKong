/**
 *  @file test/geospatial.cpp
 *  @author Ash Vardanian
 *  @date December 28, 2025
 *  @brief Many-to-many 3D squares, Haversine, Vincenty and terrain clearance tests.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include "geospatial.hpp" // `test_geodesics_packed`, `test_terrain_packed`

namespace ashvardanian::numkong::test {

using sqeuclideans3d_i32_kernel_t = nk_status_t (*)(nk_i32_t const *, nk_size_t, nk_i32_t const *, nk_size_t,
                                                    nk_u64_t *, nk_size_t, nk_stream_t);
using sqeuclideans3d_f64_kernel_t = nk_status_t (*)(nk_f64_t const *, nk_size_t, nk_f64_t const *, nk_size_t,
                                                    nk_f64_t *, nk_size_t, nk_stream_t);

/** One rows × columns case of a many-to-many family. */
struct geospatial_shape_t {
    std::size_t rows, columns;
};

/** The configured shape, then single cells and odd edges off every vector width, as
 *  @c matrix_shapes lists them. */
static std::vector<geospatial_shape_t> geospatial_shapes() {
    return {{test_run.settings.matrix_height, test_run.settings.matrix_width}, {1, 1}, {2, 3}, {3, 2}, {17, 33}};
}

/** Exact squares of Earth-fixed centimetres against brute force in 64-bit integers. */
error_stats_t test_sqeuclideans3d_i32(sqeuclideans3d_i32_kernel_t kernel) {
    error_stats_t stats(0.0);
    split_mix_generator_t generator(case_stream_key());
    // Earth-fixed centimetres reach ±6.4e8 on the surface, so their differences fit 32 bits.
    std::uniform_int_distribution<nk_i32_t> coordinate(-680'000'000, 680'000'000);
    nk::vector<nk_i32_t> a, b;
    nk::vector<nk_u64_t> results;
    for (deadline_t const deadline = test_run.settings.deadline(); deadline.ahead();)
        for (geospatial_shape_t const &shape : geospatial_shapes()) {
            resize_vector(a, shape.rows * 3), resize_vector(b, shape.columns * 3);
            resize_vector(results, shape.rows * shape.columns);
            // A cell the kernel skips reads as zero, not as the previous shape's value.
            std::memset(results.data(), 0, results.size() * sizeof(*results.data()));
            for (nk_i32_t &value : a) value = coordinate(generator);
            for (nk_i32_t &value : b) value = coordinate(generator);
            // A coincident pair pins the zero.
            for (std::size_t axis = 0; axis != 3; ++axis) b[axis] = a[axis];
            if (!stats.expect_status(kernel(a.data(), shape.rows, b.data(), shape.columns, results.data(),
                                            shape.columns * sizeof(nk_u64_t), nullptr)))
                return stats;
            for (std::size_t row = 0; row != shape.rows; ++row)
                for (std::size_t column = 0; column != shape.columns; ++column) {
                    nk_u64_t expected = 0;
                    for (std::size_t axis = 0; axis != 3; ++axis) {
                        nk_i64_t const difference = (nk_i64_t)a[row * 3 + axis] - (nk_i64_t)b[column * 3 + axis];
                        expected += (nk_u64_t)(difference * difference);
                    }
                    stats.expect_true(results[row * shape.columns + column] == expected,
                                      "squared distance differs from brute force");
                }
        }
    return stats;
}

/** Dot2 squares of doubles against the double-double sum of exact differences. */
error_stats_t test_sqeuclideans3d_f64(sqeuclideans3d_f64_kernel_t kernel) {
    error_stats_t stats(nk_accumulation_error_bound(nk_f64_k));
    split_mix_generator_t generator(case_stream_key());
    std::uniform_real_distribution<double> coordinate(-1e7, 1e7);
    nk::vector<nk_f64_t> a, b, results;
    for (deadline_t const deadline = test_run.settings.deadline(); deadline.ahead();)
        for (geospatial_shape_t const &shape : geospatial_shapes()) {
            resize_vector(a, shape.rows * 3), resize_vector(b, shape.columns * 3);
            resize_vector(results, shape.rows * shape.columns);
            // A cell the kernel skips reads as zero, not as the previous shape's value.
            std::memset(results.data(), 0, results.size() * sizeof(*results.data()));
            for (nk_f64_t &value : a) value = coordinate(generator);
            for (nk_f64_t &value : b) value = coordinate(generator);
            if (!stats.expect_status(kernel(a.data(), shape.rows, b.data(), shape.columns, results.data(),
                                            shape.columns * sizeof(nk_f64_t), nullptr)))
                return stats;
            for (std::size_t row = 0; row != shape.rows; ++row)
                for (std::size_t column = 0; column != shape.columns; ++column) {
                    f118_t expected = f118_t(0.0);
                    for (std::size_t axis = 0; axis != 3; ++axis) {
                        f118_t const difference = f118_t(a[row * 3 + axis]) - f118_t(b[column * 3 + axis]);
                        expected = expected + difference * difference;
                    }
                    // A difference rounds once, its square doubles that: 2 roundings a term.
                    stats.accumulate_bounded(f64_t(results[row * shape.columns + column]), expected,
                                             3 * stats.term_error_bound * to_double(expected));
                }
        }
    return stats;
}

void run_geospatial_cases() {
    auto const serial = nk_cap_serial_k;

    run_kernel_case("nk_sqeuclideans3d_i32_serial", test_sqeuclideans3d_i32, serial, nk_sqeuclideans3d_i32_serial);
    run_kernel_case("nk_sqeuclideans3d_f64_serial", test_sqeuclideans3d_f64, serial, nk_sqeuclideans3d_f64_serial);
    run_kernel_case("nk_haversines_packed_f64_serial", test_geodesics_packed<f64_t, nk_kernel_haversines_packed_k>,
                    serial, nk_haversines_pack_size_f64_serial, nk_haversines_pack_f64_serial,
                    nk_haversines_packed_f64_serial);
    run_kernel_case("nk_vincentys_packed_f64_serial", test_geodesics_packed<f64_t, nk_kernel_vincentys_packed_k>,
                    serial, nk_vincentys_pack_size_f64_serial, nk_vincentys_pack_f64_serial,
                    nk_vincentys_packed_f64_serial);
    run_kernel_case("nk_clearances_packed_f32_serial", test_terrain_packed<f32_t>, serial,
                    nk_terrain_pack_size_f32_serial, nk_terrain_pack_f32_serial, nk_terrain_heights_packed_f32_serial,
                    nk_clearances_packed_f32_serial);

    run_kernel_case("nk_sqeuclideans3d_i32_best", test_sqeuclideans3d_i32, serial,
                    cpu_best<nk_sqeuclideans3d_i32_best>);
    run_kernel_case("nk_sqeuclideans3d_f64_best", test_sqeuclideans3d_f64, serial,
                    cpu_best<nk_sqeuclideans3d_f64_best>);
    run_kernel_case("nk_haversines_packed_f64_best", test_geodesics_packed<f64_t, nk_kernel_haversines_packed_k>,
                    serial, nk_haversines_pack_size_f64_serial, nk_haversines_pack_f64_serial,
                    cpu_best<nk_haversines_packed_f64_best>);
    run_kernel_case("nk_vincentys_packed_f64_best", test_geodesics_packed<f64_t, nk_kernel_vincentys_packed_k>, serial,
                    nk_vincentys_pack_size_f64_serial, nk_vincentys_pack_f64_serial,
                    cpu_best<nk_vincentys_packed_f64_best>);
    run_kernel_case("nk_clearances_packed_f32_best", test_terrain_packed<f32_t>, serial,
                    nk_terrain_pack_size_f32_serial, nk_terrain_pack_f32_serial, nk_terrain_heights_packed_f32_serial,
                    cpu_best<nk_clearances_packed_f32_best>);

#if NUMKONG_TARGET_NEON
    auto const neon = nk_cap_neon_k;
    run_kernel_case("nk_sqeuclideans3d_i32_neon", test_sqeuclideans3d_i32, neon, nk_sqeuclideans3d_i32_neon);
    run_kernel_case("nk_sqeuclideans3d_f64_neon", test_sqeuclideans3d_f64, neon, nk_sqeuclideans3d_f64_neon);
    run_kernel_case("nk_haversines_packed_f64_neon", test_geodesics_packed<f64_t, nk_kernel_haversines_packed_k>, neon,
                    nk_haversines_pack_size_f64_serial, nk_haversines_pack_f64_serial, nk_haversines_packed_f64_neon);
    run_kernel_case("nk_vincentys_packed_f64_neon", test_geodesics_packed<f64_t, nk_kernel_vincentys_packed_k>, neon,
                    nk_vincentys_pack_size_f64_serial, nk_vincentys_pack_f64_serial, nk_vincentys_packed_f64_neon);
#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_HASWELL
    auto const haswell = nk_cap_haswell_k;
    run_kernel_case("nk_sqeuclideans3d_i32_haswell", test_sqeuclideans3d_i32, haswell, nk_sqeuclideans3d_i32_haswell);
    run_kernel_case("nk_sqeuclideans3d_f64_haswell", test_sqeuclideans3d_f64, haswell, nk_sqeuclideans3d_f64_haswell);
    run_kernel_case("nk_haversines_packed_f64_haswell", test_geodesics_packed<f64_t, nk_kernel_haversines_packed_k>,
                    haswell, nk_haversines_pack_size_f64_serial, nk_haversines_pack_f64_serial,
                    nk_haversines_packed_f64_haswell);
    run_kernel_case("nk_vincentys_packed_f64_haswell", test_geodesics_packed<f64_t, nk_kernel_vincentys_packed_k>,
                    haswell, nk_vincentys_pack_size_f64_serial, nk_vincentys_pack_f64_serial,
                    nk_vincentys_packed_f64_haswell);
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SKYLAKE
    auto const skylake = nk_cap_skylake_k;
    run_kernel_case("nk_sqeuclideans3d_i32_skylake", test_sqeuclideans3d_i32, skylake, nk_sqeuclideans3d_i32_skylake);
    run_kernel_case("nk_sqeuclideans3d_f64_skylake", test_sqeuclideans3d_f64, skylake, nk_sqeuclideans3d_f64_skylake);
    run_kernel_case("nk_haversines_packed_f64_skylake", test_geodesics_packed<f64_t, nk_kernel_haversines_packed_k>,
                    skylake, nk_haversines_pack_size_f64_serial, nk_haversines_pack_f64_serial,
                    nk_haversines_packed_f64_skylake);
    run_kernel_case("nk_vincentys_packed_f64_skylake", test_geodesics_packed<f64_t, nk_kernel_vincentys_packed_k>,
                    skylake, nk_vincentys_pack_size_f64_serial, nk_vincentys_pack_f64_serial,
                    nk_vincentys_packed_f64_skylake);
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_RVV
    auto const rvv = nk_cap_rvv_k;
    run_kernel_case("nk_sqeuclideans3d_i32_rvv", test_sqeuclideans3d_i32, rvv, nk_sqeuclideans3d_i32_rvv);
    run_kernel_case("nk_sqeuclideans3d_f64_rvv", test_sqeuclideans3d_f64, rvv, nk_sqeuclideans3d_f64_rvv);
    run_kernel_case("nk_haversines_packed_f64_rvv", test_geodesics_packed<f64_t, nk_kernel_haversines_packed_k>, rvv,
                    nk_haversines_pack_size_f64_serial, nk_haversines_pack_f64_serial, nk_haversines_packed_f64_rvv);
    run_kernel_case("nk_vincentys_packed_f64_rvv", test_geodesics_packed<f64_t, nk_kernel_vincentys_packed_k>, rvv,
                    nk_vincentys_pack_size_f64_serial, nk_vincentys_pack_f64_serial, nk_vincentys_packed_f64_rvv);
#endif // NUMKONG_TARGET_RVV

#if NUMKONG_TARGET_V128RELAXED
    auto const v128relaxed = nk_cap_v128relaxed_k;
    run_kernel_case("nk_sqeuclideans3d_i32_v128relaxed", test_sqeuclideans3d_i32, v128relaxed,
                    nk_sqeuclideans3d_i32_v128relaxed);
    run_kernel_case("nk_sqeuclideans3d_f64_v128relaxed", test_sqeuclideans3d_f64, v128relaxed,
                    nk_sqeuclideans3d_f64_v128relaxed);
    run_kernel_case("nk_haversines_packed_f64_v128relaxed", test_geodesics_packed<f64_t, nk_kernel_haversines_packed_k>,
                    v128relaxed, nk_haversines_pack_size_f64_serial, nk_haversines_pack_f64_serial,
                    nk_haversines_packed_f64_v128relaxed);
    run_kernel_case("nk_vincentys_packed_f64_v128relaxed", test_geodesics_packed<f64_t, nk_kernel_vincentys_packed_k>,
                    v128relaxed, nk_vincentys_pack_size_f64_serial, nk_vincentys_pack_f64_serial,
                    nk_vincentys_packed_f64_v128relaxed);
#endif // NUMKONG_TARGET_V128RELAXED

#if NUMKONG_TARGET_POWERVSX
    auto const powervsx = nk_cap_powervsx_k;
    run_kernel_case("nk_sqeuclideans3d_i32_powervsx", test_sqeuclideans3d_i32, powervsx,
                    nk_sqeuclideans3d_i32_powervsx);
    run_kernel_case("nk_sqeuclideans3d_f64_powervsx", test_sqeuclideans3d_f64, powervsx,
                    nk_sqeuclideans3d_f64_powervsx);
#endif // NUMKONG_TARGET_POWERVSX
}

} // namespace ashvardanian::numkong::test
