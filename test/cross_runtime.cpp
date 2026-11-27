/**
 *  @file test/cross_runtime.cpp
 *  @author Ash Vardanian
 *  @date January 14, 2025
 *  @brief Batch operation tests - runtime dispatch through the library.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include "numkong/dots.hpp" // `nk::dots_pack`, `nk::hammings_packed`, `nk::jaccards_packed`

#include "harness.hpp"
#include "cross.hpp"

namespace ashvardanian::numkong::test {

/** Batched Hamming or Jaccard distances, as @p kind_ picks, through the SDK's own pack and
 *  @c nk::packed_matrix over the default capabilities, exact against the packed reference loops. */
template <nk_kernel_kind_t kind_>
static error_stats_t test_sets_packed_sdk() {
    constexpr bool jaccard = kind_ == nk_kernel_jaccard_k;
    using result_t = std::conditional_t<jaccard, f32_t, u32_t>;
    error_stats_t stats(comparison_family_t::exact_k);
    split_mix_generator_t generator(case_stream_key());
    for (deadline_t const deadline = test_run.settings.deadline(); deadline.ahead();)
        for (matrix_shape_t const &shape : matrix_shapes()) {
            std::size_t const rows = shape.rows, columns = shape.columns;
            std::size_t const depth = nk::divide_round_up(shape.depth, 8) * 8, stride = depth / 8;
            std::size_t const c_stride = columns * sizeof(result_t);
            auto a = make_vector<u1x8_t>(rows * depth), b = make_vector<u1x8_t>(columns * depth);
            auto c = make_vector<result_t>(rows * columns), c_reference = make_vector<result_t>(rows * columns);
            fill_random(generator, a), fill_random(generator, b);
            if constexpr (jaccard)
                std::memset(a.raw_values_data(), 0, stride), std::memset(b.raw_values_data(), 0, stride);
            if constexpr (jaccard)
                jaccards_packed_reference<u1x8_t>(a.values_data(), b.values_data(), c_reference.values_data(), rows,
                                                  columns, depth, stride, stride, c_stride);
            else
                hammings_packed_reference<u1x8_t>(a.values_data(), b.values_data(), c_reference.values_data(), rows,
                                                  columns, depth, stride, stride, c_stride);

            nk::matrix_view<u1x8_t> const a_view(a.values_data(), rows, depth), b_view(b.values_data(), columns, depth);
            nk::matrix_span<result_t> c_span(c.values_data(), rows, columns);
            auto [packed_matrix, packed_status] = nk::dots_pack<u1x8_t>(b_view);
            if (!stats.expect_status(packed_status)) continue;
            if constexpr (jaccard) stats.expect_status(nk::jaccards_packed<u1x8_t>(a_view, packed_matrix, c_span));
            else stats.expect_status(nk::hammings_packed<u1x8_t>(a_view, packed_matrix, c_span));
            for (std::size_t index = 0; index < rows * columns; index++) stats.accumulate(c[index], c_reference[index]);
        }
    return stats;
}

void run_cross_runtime_cases() {
    auto const serial = nk_cap_serial_k;
    run_kernel_case("nk_dots_packed_f64_best", test_dots_packed<f64_t>, serial, cpu_best<nk_dots_pack_size_f64_best>,
                    cpu_best<nk_dots_pack_f64_best>, cpu_best<nk_dots_packed_f64_best>);
    run_kernel_case("nk_dots_pack_f64_best", test_dots_pack_layout<f64_t, host_backend_t>, serial,
                    cpu_best<nk_dots_pack_size_f64_best>, cpu_best<nk_dots_packed_shape_f64_best>,
                    cpu_best<nk_dots_pack_f64_best>);
    run_kernel_case("nk_dots_packed_f32_best", test_dots_packed<f32_t>, serial, cpu_best<nk_dots_pack_size_f32_best>,
                    cpu_best<nk_dots_pack_f32_best>, cpu_best<nk_dots_packed_f32_best>);
    run_kernel_case("nk_dots_pack_f32_best", test_dots_pack_layout<f32_t, host_backend_t>, serial,
                    cpu_best<nk_dots_pack_size_f32_best>, cpu_best<nk_dots_packed_shape_f32_best>,
                    cpu_best<nk_dots_pack_f32_best>);
    run_kernel_case("nk_dots_packed_bf16_best", test_dots_packed<bf16_t>, serial, cpu_best<nk_dots_pack_size_bf16_best>,
                    cpu_best<nk_dots_pack_bf16_best>, cpu_best<nk_dots_packed_bf16_best>);
    run_kernel_case("nk_dots_pack_bf16_best", test_dots_pack_layout<bf16_t, host_backend_t>, serial,
                    cpu_best<nk_dots_pack_size_bf16_best>, cpu_best<nk_dots_packed_shape_bf16_best>,
                    cpu_best<nk_dots_pack_bf16_best>);
    run_kernel_case("nk_dots_packed_f16_best", test_dots_packed<f16_t>, serial, cpu_best<nk_dots_pack_size_f16_best>,
                    cpu_best<nk_dots_pack_f16_best>, cpu_best<nk_dots_packed_f16_best>);
    run_kernel_case("nk_dots_pack_f16_best", test_dots_pack_layout<f16_t, host_backend_t>, serial,
                    cpu_best<nk_dots_pack_size_f16_best>, cpu_best<nk_dots_packed_shape_f16_best>,
                    cpu_best<nk_dots_pack_f16_best>);
    run_kernel_case("nk_dots_packed_i8_best", test_dots_packed<i8_t>, serial, cpu_best<nk_dots_pack_size_i8_best>,
                    cpu_best<nk_dots_pack_i8_best>, cpu_best<nk_dots_packed_i8_best>);
    run_kernel_case("nk_dots_pack_i8_best", test_dots_pack_layout<i8_t, host_backend_t>, serial,
                    cpu_best<nk_dots_pack_size_i8_best>, cpu_best<nk_dots_packed_shape_i8_best>,
                    cpu_best<nk_dots_pack_i8_best>);
    run_kernel_case("nk_dots_packed_i4_best", test_dots_packed<i4x2_t>, serial, cpu_best<nk_dots_pack_size_i4_best>,
                    cpu_best<nk_dots_pack_i4_best>, cpu_best<nk_dots_packed_i4_best>);
    run_kernel_case("nk_dots_pack_i4_best", test_dots_pack_layout<i4x2_t, host_backend_t>, serial,
                    cpu_best<nk_dots_pack_size_i4_best>, cpu_best<nk_dots_packed_shape_i4_best>,
                    cpu_best<nk_dots_pack_i4_best>);
    run_kernel_case("nk_dots_packed_u4_best", test_dots_packed<u4x2_t>, serial, cpu_best<nk_dots_pack_size_u4_best>,
                    cpu_best<nk_dots_pack_u4_best>, cpu_best<nk_dots_packed_u4_best>);
    run_kernel_case("nk_dots_pack_u4_best", test_dots_pack_layout<u4x2_t, host_backend_t>, serial,
                    cpu_best<nk_dots_pack_size_u4_best>, cpu_best<nk_dots_packed_shape_u4_best>,
                    cpu_best<nk_dots_pack_u4_best>);
    run_kernel_case("nk_dots_packed_u1_best", test_dots_packed<u1x8_t>, serial, cpu_best<nk_dots_pack_size_u1_best>,
                    cpu_best<nk_dots_pack_u1_best>, cpu_best<nk_dots_packed_u1_best>);
    run_kernel_case("nk_dots_pack_u1_best", test_dots_pack_layout<u1x8_t, host_backend_t>, serial,
                    cpu_best<nk_dots_pack_size_u1_best>, cpu_best<nk_dots_packed_shape_u1_best>,
                    cpu_best<nk_dots_pack_u1_best>);

    run_kernel_case("nk_dots_symmetric_f64_best", test_dots_symmetric<f64_t>, serial,
                    cpu_best<nk_dots_symmetric_f64_best>);
    run_kernel_case("nk_dots_symmetric_f32_best", test_dots_symmetric<f32_t>, serial,
                    cpu_best<nk_dots_symmetric_f32_best>);
    run_kernel_case("nk_dots_symmetric_bf16_best", test_dots_symmetric<bf16_t>, serial,
                    cpu_best<nk_dots_symmetric_bf16_best>);
    run_kernel_case("nk_dots_symmetric_f16_best", test_dots_symmetric<f16_t>, serial,
                    cpu_best<nk_dots_symmetric_f16_best>);
    run_kernel_case("nk_dots_symmetric_e5m2_best", test_dots_symmetric<e5m2_t>, serial,
                    cpu_best<nk_dots_symmetric_e5m2_best>);
    run_kernel_case("nk_dots_symmetric_e4m3_best", test_dots_symmetric<e4m3_t>, serial,
                    cpu_best<nk_dots_symmetric_e4m3_best>);
    run_kernel_case("nk_dots_symmetric_i8_best", test_dots_symmetric<i8_t>, serial,
                    cpu_best<nk_dots_symmetric_i8_best>);
    run_kernel_case("nk_dots_symmetric_i4_best", test_dots_symmetric<i4x2_t>, serial,
                    cpu_best<nk_dots_symmetric_i4_best>);
    run_kernel_case("nk_dots_symmetric_u8_best", test_dots_symmetric<u8_t>, serial,
                    cpu_best<nk_dots_symmetric_u8_best>);
    run_kernel_case("nk_dots_symmetric_u4_best", test_dots_symmetric<u4x2_t>, serial,
                    cpu_best<nk_dots_symmetric_u4_best>);
    run_kernel_case("nk_dots_symmetric_u1_best", test_dots_symmetric<u1x8_t>, serial,
                    cpu_best<nk_dots_symmetric_u1_best>);

    run_kernel_case("nk_hammings_packed_u1_best", test_hammings_packed<u1x8_t>, serial,
                    cpu_best<nk_dots_pack_size_u1_best>, cpu_best<nk_dots_pack_u1_best>,
                    cpu_best<nk_hammings_packed_u1_best>);
    run_kernel_case("nk_hammings_symmetric_u1_best", test_hammings_symmetric<u1x8_t>, serial,
                    cpu_best<nk_hammings_symmetric_u1_best>);

    run_kernel_case("nk_jaccards_packed_u1_best", test_jaccards_packed<u1x8_t>, serial,
                    cpu_best<nk_dots_pack_size_u1_best>, cpu_best<nk_dots_pack_u1_best>,
                    cpu_best<nk_jaccards_packed_u1_best>);
    run_kernel_case("nk_jaccards_symmetric_u1_best", test_jaccards_symmetric<u1x8_t>, serial,
                    cpu_best<nk_jaccards_symmetric_u1_best>);
    run_kernel_case("nk::hammings_packed<u1x8_t>", test_sets_packed_sdk<nk_kernel_hamming_k>, serial);
    run_kernel_case("nk::jaccards_packed<u1x8_t>", test_sets_packed_sdk<nk_kernel_jaccard_k>, serial);

    run_kernel_case("nk_dots_packed_e5m2_best", test_dots_packed<e5m2_t>, serial, cpu_best<nk_dots_pack_size_e5m2_best>,
                    cpu_best<nk_dots_pack_e5m2_best>, cpu_best<nk_dots_packed_e5m2_best>);
    run_kernel_case("nk_dots_pack_e5m2_best", test_dots_pack_layout<e5m2_t, host_backend_t>, serial,
                    cpu_best<nk_dots_pack_size_e5m2_best>, cpu_best<nk_dots_packed_shape_e5m2_best>,
                    cpu_best<nk_dots_pack_e5m2_best>);
    run_kernel_case("nk_dots_packed_e4m3_best", test_dots_packed<e4m3_t>, serial, cpu_best<nk_dots_pack_size_e4m3_best>,
                    cpu_best<nk_dots_pack_e4m3_best>, cpu_best<nk_dots_packed_e4m3_best>);
    run_kernel_case("nk_dots_pack_e4m3_best", test_dots_pack_layout<e4m3_t, host_backend_t>, serial,
                    cpu_best<nk_dots_pack_size_e4m3_best>, cpu_best<nk_dots_packed_shape_e4m3_best>,
                    cpu_best<nk_dots_pack_e4m3_best>);
    run_kernel_case("nk_dots_packed_e3m2_best", test_dots_packed<e3m2_t>, serial, cpu_best<nk_dots_pack_size_e3m2_best>,
                    cpu_best<nk_dots_pack_e3m2_best>, cpu_best<nk_dots_packed_e3m2_best>);
    run_kernel_case("nk_dots_pack_e3m2_best", test_dots_pack_layout<e3m2_t, host_backend_t>, serial,
                    cpu_best<nk_dots_pack_size_e3m2_best>, cpu_best<nk_dots_packed_shape_e3m2_best>,
                    cpu_best<nk_dots_pack_e3m2_best>);
    run_kernel_case("nk_dots_packed_e2m3_best", test_dots_packed<e2m3_t>, serial, cpu_best<nk_dots_pack_size_e2m3_best>,
                    cpu_best<nk_dots_pack_e2m3_best>, cpu_best<nk_dots_packed_e2m3_best>);
    run_kernel_case("nk_dots_pack_e2m3_best", test_dots_pack_layout<e2m3_t, host_backend_t>, serial,
                    cpu_best<nk_dots_pack_size_e2m3_best>, cpu_best<nk_dots_packed_shape_e2m3_best>,
                    cpu_best<nk_dots_pack_e2m3_best>);
    run_kernel_case("nk_dots_packed_e2m1_best", test_dots_packed<e2m1x2_t>, serial,
                    cpu_best<nk_dots_pack_size_e2m1_best>, cpu_best<nk_dots_pack_e2m1_best>,
                    cpu_best<nk_dots_packed_e2m1_best>);
    run_kernel_case("nk_dots_pack_e2m1_best", test_dots_pack_layout<e2m1x2_t, host_backend_t>, serial,
                    cpu_best<nk_dots_pack_size_e2m1_best>, cpu_best<nk_dots_packed_shape_e2m1_best>,
                    cpu_best<nk_dots_pack_e2m1_best>);
    run_kernel_case("nk_dots_packed_u8_best", test_dots_packed<u8_t>, serial, cpu_best<nk_dots_pack_size_u8_best>,
                    cpu_best<nk_dots_pack_u8_best>, cpu_best<nk_dots_packed_u8_best>);
    run_kernel_case("nk_dots_pack_u8_best", test_dots_pack_layout<u8_t, host_backend_t>, serial,
                    cpu_best<nk_dots_pack_size_u8_best>, cpu_best<nk_dots_packed_shape_u8_best>,
                    cpu_best<nk_dots_pack_u8_best>);

    run_kernel_case("nk_dots_symmetric_e3m2_best", test_dots_symmetric<e3m2_t>, serial,
                    cpu_best<nk_dots_symmetric_e3m2_best>);
    run_kernel_case("nk_dots_symmetric_e2m3_best", test_dots_symmetric<e2m3_t>, serial,
                    cpu_best<nk_dots_symmetric_e2m3_best>);
    run_kernel_case("nk_dots_symmetric_e2m1_best", test_dots_symmetric<e2m1x2_t>, serial,
                    cpu_best<nk_dots_symmetric_e2m1_best>);

    run_kernel_case("nk_angulars_packed_f64_best", test_angulars_packed<f64_t>, serial,
                    cpu_best<nk_dots_pack_size_f64_best>, cpu_best<nk_dots_pack_f64_best>,
                    cpu_best<nk_angulars_packed_f64_best>);
    run_kernel_case("nk_angulars_packed_f32_best", test_angulars_packed<f32_t>, serial,
                    cpu_best<nk_dots_pack_size_f32_best>, cpu_best<nk_dots_pack_f32_best>,
                    cpu_best<nk_angulars_packed_f32_best>);
    run_kernel_case("nk_angulars_packed_bf16_best", test_angulars_packed<bf16_t>, serial,
                    cpu_best<nk_dots_pack_size_bf16_best>, cpu_best<nk_dots_pack_bf16_best>,
                    cpu_best<nk_angulars_packed_bf16_best>);
    run_kernel_case("nk_angulars_packed_f16_best", test_angulars_packed<f16_t>, serial,
                    cpu_best<nk_dots_pack_size_f16_best>, cpu_best<nk_dots_pack_f16_best>,
                    cpu_best<nk_angulars_packed_f16_best>);
    run_kernel_case("nk_angulars_packed_e5m2_best", test_angulars_packed<e5m2_t>, serial,
                    cpu_best<nk_dots_pack_size_e5m2_best>, cpu_best<nk_dots_pack_e5m2_best>,
                    cpu_best<nk_angulars_packed_e5m2_best>);
    run_kernel_case("nk_angulars_packed_e4m3_best", test_angulars_packed<e4m3_t>, serial,
                    cpu_best<nk_dots_pack_size_e4m3_best>, cpu_best<nk_dots_pack_e4m3_best>,
                    cpu_best<nk_angulars_packed_e4m3_best>);
    run_kernel_case("nk_angulars_packed_e3m2_best", test_angulars_packed<e3m2_t>, serial,
                    cpu_best<nk_dots_pack_size_e3m2_best>, cpu_best<nk_dots_pack_e3m2_best>,
                    cpu_best<nk_angulars_packed_e3m2_best>);
    run_kernel_case("nk_angulars_packed_e2m3_best", test_angulars_packed<e2m3_t>, serial,
                    cpu_best<nk_dots_pack_size_e2m3_best>, cpu_best<nk_dots_pack_e2m3_best>,
                    cpu_best<nk_angulars_packed_e2m3_best>);
    run_kernel_case("nk_angulars_packed_e2m1_best", test_angulars_packed<e2m1x2_t>, serial,
                    cpu_best<nk_dots_pack_size_e2m1_best>, cpu_best<nk_dots_pack_e2m1_best>,
                    cpu_best<nk_angulars_packed_e2m1_best>);
    run_kernel_case("nk_angulars_packed_i8_best", test_angulars_packed<i8_t>, serial,
                    cpu_best<nk_dots_pack_size_i8_best>, cpu_best<nk_dots_pack_i8_best>,
                    cpu_best<nk_angulars_packed_i8_best>);
    run_kernel_case("nk_angulars_packed_i4_best", test_angulars_packed<i4x2_t>, serial,
                    cpu_best<nk_dots_pack_size_i4_best>, cpu_best<nk_dots_pack_i4_best>,
                    cpu_best<nk_angulars_packed_i4_best>);
    run_kernel_case("nk_angulars_packed_u8_best", test_angulars_packed<u8_t>, serial,
                    cpu_best<nk_dots_pack_size_u8_best>, cpu_best<nk_dots_pack_u8_best>,
                    cpu_best<nk_angulars_packed_u8_best>);
    run_kernel_case("nk_angulars_packed_u4_best", test_angulars_packed<u4x2_t>, serial,
                    cpu_best<nk_dots_pack_size_u4_best>, cpu_best<nk_dots_pack_u4_best>,
                    cpu_best<nk_angulars_packed_u4_best>);

    run_kernel_case("nk_angulars_symmetric_f64_best", test_angulars_symmetric<f64_t>, serial,
                    cpu_best<nk_angulars_symmetric_f64_best>);
    run_kernel_case("nk_angulars_symmetric_f32_best", test_angulars_symmetric<f32_t>, serial,
                    cpu_best<nk_angulars_symmetric_f32_best>);
    run_kernel_case("nk_angulars_symmetric_bf16_best", test_angulars_symmetric<bf16_t>, serial,
                    cpu_best<nk_angulars_symmetric_bf16_best>);
    run_kernel_case("nk_angulars_symmetric_f16_best", test_angulars_symmetric<f16_t>, serial,
                    cpu_best<nk_angulars_symmetric_f16_best>);
    run_kernel_case("nk_angulars_symmetric_e5m2_best", test_angulars_symmetric<e5m2_t>, serial,
                    cpu_best<nk_angulars_symmetric_e5m2_best>);
    run_kernel_case("nk_angulars_symmetric_e4m3_best", test_angulars_symmetric<e4m3_t>, serial,
                    cpu_best<nk_angulars_symmetric_e4m3_best>);
    run_kernel_case("nk_angulars_symmetric_e3m2_best", test_angulars_symmetric<e3m2_t>, serial,
                    cpu_best<nk_angulars_symmetric_e3m2_best>);
    run_kernel_case("nk_angulars_symmetric_e2m3_best", test_angulars_symmetric<e2m3_t>, serial,
                    cpu_best<nk_angulars_symmetric_e2m3_best>);
    run_kernel_case("nk_angulars_symmetric_e2m1_best", test_angulars_symmetric<e2m1x2_t>, serial,
                    cpu_best<nk_angulars_symmetric_e2m1_best>);
    run_kernel_case("nk_angulars_symmetric_i8_best", test_angulars_symmetric<i8_t>, serial,
                    cpu_best<nk_angulars_symmetric_i8_best>);
    run_kernel_case("nk_angulars_symmetric_i4_best", test_angulars_symmetric<i4x2_t>, serial,
                    cpu_best<nk_angulars_symmetric_i4_best>);
    run_kernel_case("nk_angulars_symmetric_u8_best", test_angulars_symmetric<u8_t>, serial,
                    cpu_best<nk_angulars_symmetric_u8_best>);
    run_kernel_case("nk_angulars_symmetric_u4_best", test_angulars_symmetric<u4x2_t>, serial,
                    cpu_best<nk_angulars_symmetric_u4_best>);

    run_kernel_case("nk_euclideans_packed_f64_best", test_euclideans_packed<f64_t>, serial,
                    cpu_best<nk_dots_pack_size_f64_best>, cpu_best<nk_dots_pack_f64_best>,
                    cpu_best<nk_euclideans_packed_f64_best>);
    run_kernel_case("nk_euclideans_packed_f32_best", test_euclideans_packed<f32_t>, serial,
                    cpu_best<nk_dots_pack_size_f32_best>, cpu_best<nk_dots_pack_f32_best>,
                    cpu_best<nk_euclideans_packed_f32_best>);
    run_kernel_case("nk_euclideans_packed_bf16_best", test_euclideans_packed<bf16_t>, serial,
                    cpu_best<nk_dots_pack_size_bf16_best>, cpu_best<nk_dots_pack_bf16_best>,
                    cpu_best<nk_euclideans_packed_bf16_best>);
    run_kernel_case("nk_euclideans_packed_f16_best", test_euclideans_packed<f16_t>, serial,
                    cpu_best<nk_dots_pack_size_f16_best>, cpu_best<nk_dots_pack_f16_best>,
                    cpu_best<nk_euclideans_packed_f16_best>);
    run_kernel_case("nk_euclideans_packed_e5m2_best", test_euclideans_packed<e5m2_t>, serial,
                    cpu_best<nk_dots_pack_size_e5m2_best>, cpu_best<nk_dots_pack_e5m2_best>,
                    cpu_best<nk_euclideans_packed_e5m2_best>);
    run_kernel_case("nk_euclideans_packed_e4m3_best", test_euclideans_packed<e4m3_t>, serial,
                    cpu_best<nk_dots_pack_size_e4m3_best>, cpu_best<nk_dots_pack_e4m3_best>,
                    cpu_best<nk_euclideans_packed_e4m3_best>);
    run_kernel_case("nk_euclideans_packed_e3m2_best", test_euclideans_packed<e3m2_t>, serial,
                    cpu_best<nk_dots_pack_size_e3m2_best>, cpu_best<nk_dots_pack_e3m2_best>,
                    cpu_best<nk_euclideans_packed_e3m2_best>);
    run_kernel_case("nk_euclideans_packed_e2m3_best", test_euclideans_packed<e2m3_t>, serial,
                    cpu_best<nk_dots_pack_size_e2m3_best>, cpu_best<nk_dots_pack_e2m3_best>,
                    cpu_best<nk_euclideans_packed_e2m3_best>);
    run_kernel_case("nk_euclideans_packed_e2m1_best", test_euclideans_packed<e2m1x2_t>, serial,
                    cpu_best<nk_dots_pack_size_e2m1_best>, cpu_best<nk_dots_pack_e2m1_best>,
                    cpu_best<nk_euclideans_packed_e2m1_best>);
    run_kernel_case("nk_euclideans_packed_i8_best", test_euclideans_packed<i8_t>, serial,
                    cpu_best<nk_dots_pack_size_i8_best>, cpu_best<nk_dots_pack_i8_best>,
                    cpu_best<nk_euclideans_packed_i8_best>);
    run_kernel_case("nk_euclideans_packed_i4_best", test_euclideans_packed<i4x2_t>, serial,
                    cpu_best<nk_dots_pack_size_i4_best>, cpu_best<nk_dots_pack_i4_best>,
                    cpu_best<nk_euclideans_packed_i4_best>);
    run_kernel_case("nk_euclideans_packed_u8_best", test_euclideans_packed<u8_t>, serial,
                    cpu_best<nk_dots_pack_size_u8_best>, cpu_best<nk_dots_pack_u8_best>,
                    cpu_best<nk_euclideans_packed_u8_best>);
    run_kernel_case("nk_euclideans_packed_u4_best", test_euclideans_packed<u4x2_t>, serial,
                    cpu_best<nk_dots_pack_size_u4_best>, cpu_best<nk_dots_pack_u4_best>,
                    cpu_best<nk_euclideans_packed_u4_best>);

    run_kernel_case("nk_euclideans_symmetric_f64_best", test_euclideans_symmetric<f64_t>, serial,
                    cpu_best<nk_euclideans_symmetric_f64_best>);
    run_kernel_case("nk_euclideans_symmetric_f32_best", test_euclideans_symmetric<f32_t>, serial,
                    cpu_best<nk_euclideans_symmetric_f32_best>);
    run_kernel_case("nk_euclideans_symmetric_bf16_best", test_euclideans_symmetric<bf16_t>, serial,
                    cpu_best<nk_euclideans_symmetric_bf16_best>);
    run_kernel_case("nk_euclideans_symmetric_f16_best", test_euclideans_symmetric<f16_t>, serial,
                    cpu_best<nk_euclideans_symmetric_f16_best>);
    run_kernel_case("nk_euclideans_symmetric_e5m2_best", test_euclideans_symmetric<e5m2_t>, serial,
                    cpu_best<nk_euclideans_symmetric_e5m2_best>);
    run_kernel_case("nk_euclideans_symmetric_e4m3_best", test_euclideans_symmetric<e4m3_t>, serial,
                    cpu_best<nk_euclideans_symmetric_e4m3_best>);
    run_kernel_case("nk_euclideans_symmetric_e3m2_best", test_euclideans_symmetric<e3m2_t>, serial,
                    cpu_best<nk_euclideans_symmetric_e3m2_best>);
    run_kernel_case("nk_euclideans_symmetric_e2m3_best", test_euclideans_symmetric<e2m3_t>, serial,
                    cpu_best<nk_euclideans_symmetric_e2m3_best>);
    run_kernel_case("nk_euclideans_symmetric_e2m1_best", test_euclideans_symmetric<e2m1x2_t>, serial,
                    cpu_best<nk_euclideans_symmetric_e2m1_best>);
    run_kernel_case("nk_euclideans_symmetric_i8_best", test_euclideans_symmetric<i8_t>, serial,
                    cpu_best<nk_euclideans_symmetric_i8_best>);
    run_kernel_case("nk_euclideans_symmetric_i4_best", test_euclideans_symmetric<i4x2_t>, serial,
                    cpu_best<nk_euclideans_symmetric_i4_best>);
    run_kernel_case("nk_euclideans_symmetric_u8_best", test_euclideans_symmetric<u8_t>, serial,
                    cpu_best<nk_euclideans_symmetric_u8_best>);
    run_kernel_case("nk_euclideans_symmetric_u4_best", test_euclideans_symmetric<u4x2_t>, serial,
                    cpu_best<nk_euclideans_symmetric_u4_best>);
}

} // namespace ashvardanian::numkong::test
