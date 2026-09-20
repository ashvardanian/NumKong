/**
 *  @brief C++ vector type instantiation tests.
 *  @file test/test_tensor.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 */
#include <array>
#include <cassert>
#include <complex>

#include "test.hpp"

#include "numkong/cast.hpp"
#include "numkong/dot.hpp"
#include "numkong/spatial.hpp"
#include "numkong/curved.hpp"
#include "numkong/reduce.hpp"
#include "numkong/trigonometry.hpp"

#if __has_include(<format>)
#include <format>
#if defined(__cpp_lib_format) && __cpp_lib_format >= 202110L
#define NK_TEST_FORMAT_ 1
#endif
#endif
#ifndef NK_TEST_FORMAT_
#define NK_TEST_FORMAT_ 0
#endif

#if NK_TEST_FORMAT_
error_stats_t test_format_scalars();
#endif

// Explicit instantiations for tensor types — forces full compilation of all APIs
template class nk::tensor<nk::f32_t>;
template class nk::tensor<nk::f64_t>;
template class nk::tensor<nk::f16_t>;
template class nk::tensor<nk::bf16_t>;
template class nk::tensor<nk::i8_t>;

// Views and spans for rank-2 (matrix) and default rank
template class nk::tensor_view<nk::f32_t, 2>;
template class nk::tensor_view<nk::f32_t, 8>;
template class nk::tensor_span<nk::f32_t, 2>;
template class nk::tensor_span<nk::f32_t, 8>;
template class nk::tensor_view<nk::bf16_t, 2>;
template class nk::tensor_span<nk::bf16_t, 2>;

template <typename value_type_>
error_stats_t test_vector_basics() {
    error_stats_t stats(comparison_family_t::exact_k);
    constexpr std::size_t dims_per_value = nk::dimensions_per_value<value_type_>();
    constexpr std::size_t test_dims = 64 * dims_per_value;
    auto v = make_vector<value_type_>(test_dims);
    stats.expect(v.size() == test_dims, "vector size");
    stats.expect(v.size_values() == test_dims / dims_per_value, "vector value count");
    std::size_t count = 0;
    for (auto it = v.begin(); it != v.end(); ++it) ++count;
    stats.expect(count == test_dims, "iterated element count");
    return stats;
}

error_stats_t test_signed_indexing() {
    error_stats_t stats(comparison_family_t::exact_k);
    auto v = make_vector<float>(100);
    v[50] = 3.14f;
    stats.expect(v[50] == 3.14f, "float operator[] failed");
    v[-1] = 42.0f;
    stats.expect(v[99] == 42.0f, "float signed indexing failed");
    return stats;
}

error_stats_t test_integral_indexing_api() {
    error_stats_t stats(comparison_family_t::exact_k);
    auto v = make_vector<float>(5);
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>(i + 1);

    auto view = nk::vector_view<float>(v.values_data(), unsigned(v.size()));
    auto span = nk::vector_span<float>(v.values_data(), unsigned(v.size()));
    auto strided = nk::vector_view<float>(reinterpret_cast<char const *>(v.values_data()), 3u, sizeof(float));

    stats.expect(v[std::size_t {2}] == 3.0f, "vector unsigned indexing failed");
    stats.expect(view[2u] == 3.0f, "view unsigned indexing failed");
    stats.expect(view[std::ptrdiff_t {-1}] == 5.0f, "view signed indexing failed");
    stats.expect(span[unsigned {3}] == 4.0f, "span unsigned indexing failed");
    stats.expect(strided[2u] == 3.0f, "raw view unsigned stride ctor failed");

    auto sub = view[nk::range(1u, 4u)];
    stats.expect(sub.size() == 3, "unsigned range size mismatch");
    stats.expect(sub[0u] == 2.0f, "unsigned range first element mismatch");

    auto tail = view[nk::range(-3, -1)];
    stats.expect(tail.size() == 2, "signed range size mismatch");
    stats.expect(tail[0u] == 3.0f, "signed range first element mismatch");
    stats.expect(tail[1u] == 4.0f, "signed range last element mismatch");
    return stats;
}

error_stats_t test_tensor_operator_indexing() {
    error_stats_t stats(comparison_family_t::exact_k);
    auto t = nk::tensor<float>::try_zeros({2, 3});
    stats.expect(!t.empty(), "tensor allocation failed");

    for (int i = 0; i < 6; ++i) t[i] = static_cast<float>(i + 1);

    stats.expect(t[0] == 1.0f, "flat tensor lookup failed");
    stats.expect(t[-1] == 6.0f, "negative flat tensor lookup failed");
    stats.expect((t(0, 0) == 1.0f), "exact tensor lookup failed");
    stats.expect((t(1, -1) == 6.0f), "negative exact tensor lookup failed");
#if NK_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    stats.expect((t[0, 0] == 1.0f), "exact tensor lookup via operator[] failed");
    stats.expect((t[1, -1] == 6.0f), "negative exact tensor lookup via operator[] failed");
#endif

    auto whole = t[nk::slice];
    stats.expect(whole.rank() == 2, "slice identity rank mismatch");
    stats.expect(whole.extent(0) == 2 && whole.extent(1) == 3, "slice identity extents mismatch");

    auto row1 = t(1, nk::slice);
    stats.expect(row1.rank() == 1, "row slice rank mismatch");
    stats.expect(row1.extent(0) == 3, "row slice extent mismatch");
    stats.expect(row1[0] == 4.0f && row1[-1] == 6.0f, "row slice values mismatch");
    row1[1] = 42.0f;
    stats.expect((t(1, 1) == 42.0f), "row slice write-through failed");
#if NK_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    stats.expect((t[1, 1] == 42.0f), "operator[] row slice write-through failed");
    auto row1_subscript = t[1, nk::slice];
    stats.expect(row1_subscript.extent(0) == row1.extent(0), "operator[] row slice mismatch");
#endif

    auto cell = t(1, 1, nk::slice);
    stats.expect(cell.rank() == 0, "scalar slice rank mismatch");
    stats.expect(cell.scalar() == 42.0f, "scalar slice value mismatch");
    cell.scalar_ref() = 24.0f;
    stats.expect((t(1, 1) == 24.0f), "scalar slice write-through failed");
#if NK_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    stats.expect((t[1, 1] == 24.0f), "operator[] scalar slice write-through failed");
    auto cell_subscript = t[1, 1, nk::slice];
    stats.expect(cell_subscript.rank() == 0, "operator[] scalar slice rank mismatch");
#endif

    auto const &ct = t;
    auto const last_row = ct(-1, nk::slice);
    stats.expect(last_row.rank() == 1, "const row slice rank mismatch");
    stats.expect(last_row[0] == 4.0f, "const row slice mismatch");
#if NK_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    auto const last_row_subscript = ct[-1, nk::slice];
    stats.expect(last_row_subscript.rank() == 1, "operator[] const row slice mismatch");
#endif

    auto cube = nk::tensor<float>::try_zeros({2, 3, 4});
    stats.expect(!cube.empty(), "cube allocation failed");
    for (int i = 0; i < 24; ++i) cube[i] = static_cast<float>(i);

    auto plane = cube(1, nk::slice);
    stats.expect(plane.rank() == 2 && plane.extent(0) == 3 && plane.extent(1) == 4, "plane slice mismatch");
    auto line = cube(1, 2, nk::slice);
    stats.expect(line.rank() == 1 && line.extent(0) == 4, "line slice mismatch");
    stats.expect((line[3] == cube(1, 2, 3)), "line slice element mismatch");
    auto point = cube(1, 2, 3, nk::slice);
    stats.expect((point.rank() == 0 && point.scalar() == cube(1, 2, 3)), "point slice mismatch");
#if NK_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    auto plane_subscript = cube[1, nk::slice];
    auto line_subscript = cube[1, 2, nk::slice];
    auto point_subscript = cube[1, 2, 3, nk::slice];
    stats.expect((line_subscript[3] == cube[1, 2, 3]), "operator[] line slice element mismatch");
    stats.expect((point_subscript.scalar() == cube[1, 2, 3]), "operator[] point slice mismatch");
    stats.expect(plane_subscript.rank() == 2, "operator[] plane slice rank mismatch");
#endif

    // all_t slicing: extract a column
    auto second_column = t(nk::all, 1, nk::slice);
    stats.expect(second_column.rank() == 1, "all_t column rank mismatch");
    stats.expect(second_column.numel() == 2, "all_t column numel mismatch");

    // range slicing: extract a sub-range of rows
    auto first_two_planes = cube(nk::range(0, 2), nk::slice);
    stats.expect(first_two_planes.rank() == 3, "range slice rank mismatch");
    stats.expect(first_two_planes.extent(0) == 2, "range slice extent mismatch");

    // combined: range + all_t + slice on a 3D tensor
    auto sub = cube(nk::range(0, 2), nk::all, nk::slice);
    stats.expect(sub.rank() == 3 && sub.extent(0) == 2 && sub.extent(1) == 3, "range+all slice mismatch");
#if NK_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    auto second_column_subscript = t[nk::all, 1, nk::slice];
    auto first_two_planes_subscript = cube[nk::range(0, 2), nk::slice];
    auto sub_subscript = cube[nk::range(0, 2), nk::all, nk::slice];
    stats.expect(second_column_subscript.numel() == 2, "operator[] all_t column mismatch");
    stats.expect(first_two_planes_subscript.extent(0) == 2, "operator[] range slice mismatch");
    stats.expect(sub_subscript.extent(1) == 3, "operator[] range+all slice mismatch");
#endif

    // row() access
    auto row0 = t.row(0);
    stats.expect(row0.rank() == 1 && row0.extent(0) == 3, "row() rank/extent mismatch");
    auto row0_via_slice = t(0, nk::slice);
    stats.expect(row0[0] == row0_via_slice[0], "row() should match t(0, slice)");
#if NK_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    auto row0_via_subscript = t[0, nk::slice];
    stats.expect(row0[0] == row0_via_subscript[0], "row() should match t[0, slice]");
#endif
    return stats;
}

error_stats_t test_packed_tensor_operator_indexing() {
    error_stats_t stats(comparison_family_t::exact_k);
    auto t4 = nk::tensor<nk::u4x2_t>::try_zeros({2, 4});
    stats.expect(!t4.empty(), "packed u4 tensor allocation failed");

    for (int i = 0; i < 8; ++i) t4[i] = i + 1;

    stats.expect(int(t4[0]) == 1, "packed flat lookup failed");
    stats.expect(int(t4[-1]) == 8, "packed negative flat lookup failed");
    stats.expect((int(t4(0, 3)) == 4), "packed exact lookup failed");
    stats.expect((int(t4(1, -1)) == 8), "packed negative exact lookup failed");
#if NK_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    stats.expect((int(t4[0, 3]) == 4), "packed operator[] exact lookup failed");
    stats.expect((int(t4[1, -1]) == 8), "packed operator[] negative exact lookup failed");
#endif

    auto second_row = t4(1, nk::slice);
    stats.expect(second_row.rank() == 1 && second_row.extent(0) == 4, "packed row slice rank mismatch");
    stats.expect(int(second_row[0]) == 5 && int(second_row[-1]) == 8, "packed row slice values mismatch");
    second_row[1] = 14;
    stats.expect((int(t4(1, 1)) == 14), "packed row slice write-through failed");
#if NK_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    auto second_row_subscript = t4[1, nk::slice];
    stats.expect(second_row_subscript.extent(0) == 4, "packed operator[] row slice rank mismatch");
    stats.expect((int(t4[1, 1]) == 14), "packed operator[] row slice write-through failed");
#endif

    auto t1 = nk::tensor<nk::u1x8_t>::try_zeros({2, 8});
    stats.expect(!t1.empty(), "packed u1 tensor allocation failed");
    t1[0] = true;
    t1[7] = true;
    t1[11] = true;
    t1[-1] = true;

    stats.expect(bool(t1[0]), "packed bit flat lookup failed");
    stats.expect((bool(t1(0, 7))), "packed bit exact lookup failed");
    stats.expect((bool(t1(1, 3))), "packed bit second-row lookup failed");
    stats.expect(bool(t1[-1]), "packed bit negative flat lookup failed");
#if NK_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    stats.expect((bool(t1[0, 7])), "packed bit operator[] exact lookup failed");
    stats.expect((bool(t1[1, 3])), "packed bit operator[] second-row lookup failed");
#endif

    auto bits = t1(1, nk::slice);
    stats.expect(bits.rank() == 1 && bits.extent(0) == 8, "packed bit slice rank mismatch");
    bits[4] = true;
    stats.expect((bool(t1(1, 4))), "packed bit slice write-through failed");
#if NK_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    auto bits_subscript = t1[1, nk::slice];
    stats.expect(bits_subscript.extent(0) == 8, "packed bit operator[] slice rank mismatch");
    stats.expect((bool(t1[1, 4])), "packed bit operator[] slice write-through failed");
#endif
    return stats;
}

error_stats_t test_move_semantics() {
    error_stats_t stats(comparison_family_t::exact_k);
    auto v1 = make_vector<nk::f32_t>(100);
    v1[50] = nk::f32_t(42.0f);

    nk::vector<nk::f32_t> v2 = std::move(v1);
    stats.expect(v2.size() == 100, "move ctor size mismatch");
    stats.expect(v2[50] == nk::f32_t(42.0f), "move ctor value mismatch");
    stats.expect(v1.size() == 0, "moved-from vector not empty"); // NOLINT(bugprone-use-after-move)

    nk::vector<nk::f32_t> v3;
    v3 = std::move(v2);
    stats.expect(v3.size() == 100, "move assign size mismatch");
    stats.expect(v3[50] == nk::f32_t(42.0f), "move assign value mismatch");
    return stats;
}

error_stats_t test_swap() {
    error_stats_t stats(comparison_family_t::exact_k);
    auto v1 = make_vector<nk::i8_t>(10);
    auto v2 = make_vector<nk::i8_t>(20);
    v1[0] = nk::i8_t(1);
    v2[0] = nk::i8_t(2);

    swap(v1, v2);
    stats.expect(v1.size() == 20, "swap v1 size mismatch");
    stats.expect(v2.size() == 10, "swap v2 size mismatch");
    stats.expect(v1[0] == nk::i8_t(2), "swap v1 value mismatch");
    stats.expect(v2[0] == nk::i8_t(1), "swap v2 value mismatch");
    return stats;
}

error_stats_t test_view_span_rev() {
    error_stats_t stats(comparison_family_t::exact_k);
    auto v = make_vector<float>(5);
    v[0] = 1.0f;
    v[1] = 2.0f;
    v[2] = 3.0f;
    v[3] = 4.0f;
    v[4] = 5.0f;

    auto view = v.view();
    stats.expect(view.size() == 5, "view size mismatch");
    stats.expect(view[-1] == 5.0f, "view signed indexing failed");

    auto span = v.span();
    span[0] = 10.0f;
    stats.expect(v[0] == 10.0f, "span write-through failed");

    auto rev = view.rev();
    stats.expect(rev[0] == 5.0f, "reversed view first element mismatch");
    stats.expect(rev[4] == 10.0f, "reversed view last element mismatch");
    return stats;
}

error_stats_t test_range_slicing() {
    error_stats_t stats(comparison_family_t::exact_k);
    auto v = make_vector<float>(5);
    v[0] = 1.0f;
    v[1] = 2.0f;
    v[2] = 3.0f;
    v[3] = 4.0f;
    v[4] = 5.0f;

    auto sub = v[nk::range(1, 4)];
    stats.expect(sub.size() == 3, "range slice size mismatch");
    stats.expect(sub[0] == 2.0f, "range slice first element mismatch");
    stats.expect(sub[2] == 4.0f, "range slice last element mismatch");
    return stats;
}

error_stats_t test_sub_byte_i4x2() {
    error_stats_t stats(comparison_family_t::exact_k);
    auto v = make_vector<nk::i4x2_t>(100);
    stats.expect(v.size() == 100, "i4x2_t size mismatch");
    stats.expect(v.size_values() == 50, "i4x2_t size_values mismatch (should be dims/2)");

    v[0] = 5, v[1] = -3;
    stats.expect(v[0] == 5, "i4x2_t dim 0 mismatch");
    stats.expect(v[1] == -3, "i4x2_t dim 1 mismatch");
    return stats;
}

error_stats_t test_sub_byte_u1x8() {
    error_stats_t stats(comparison_family_t::exact_k);
    auto v = make_vector<nk::u1x8_t>(64);
    stats.expect(v.size() == 64, "u1x8_t size mismatch");
    stats.expect(v.size_values() == 8, "u1x8_t size_values mismatch (should be dims/8)");

    v[0] = true, v[1] = false, v[7] = true;
    stats.expect(v[0] == true, "u1x8_t dim 0 mismatch");
    stats.expect(v[1] == false, "u1x8_t dim 1 mismatch");
    stats.expect(v[7] == true, "u1x8_t dim 7 mismatch");
    return stats;
}

error_stats_t test_custom_allocator() {
    error_stats_t stats(comparison_family_t::exact_k);
    using custom_alloc_t = nk::aligned_allocator<nk::f32_t, 128>;
    auto v = nk::vector<nk::f32_t, custom_alloc_t>::try_zeros(256);
    stats.expect(v.size() == 256, "custom allocator size mismatch");
    v[128] = nk::f32_t(99.0f);
    stats.expect(v[128] == nk::f32_t(99.0f), "custom allocator value mismatch");
    return stats;
}

template <typename value_type_, std::size_t cols_>
void test_sub_byte_tensor_axis_reduction_case(error_stats_t &stats, std::array<int, cols_> const &first_row,
                                              std::array<int, cols_> const &second_row,
                                              std::array<int, cols_> const &expected_sums,
                                              std::array<int, cols_> const &expected_mins,
                                              std::array<int, cols_> const &expected_maxs) {
    using tensor_t = nk::tensor<value_type_>;
    using sum_t = typename value_type_::reduce_moments_sum_t;
    using minmax_t = typename value_type_::reduce_minmax_value_t;

    auto t = tensor_t::try_zeros({2, cols_});
    stats.expect(!t.empty(), "tensor allocation failed");

    auto span = t.span();
    auto row0 = span.slice_leading(0).as_vector();
    auto row1 = span.slice_leading(1).as_vector();
    for (std::size_t i = 0; i < cols_; ++i) {
        row0[i] = first_row[i];
        row1[i] = second_row[i];
    }

    auto sums = nk::try_sum<value_type_>(t.view(), 0);
    stats.expect(!sums.empty(), "axis-0 sum failed");
    auto sum_view = sums.as_vector_view();
    for (std::size_t i = 0; i < cols_; ++i) stats.expect(sum_view[i] == sum_t(expected_sums[i]), "axis-0 sum mismatch");

    auto minmax = nk::try_minmax<value_type_>(t.view(), 0);
    stats.expect(!minmax.min_value.empty() && !minmax.max_value.empty(), "axis-0 minmax failed");
    auto min_view = minmax.min_value.as_vector_view();
    auto max_view = minmax.max_value.as_vector_view();
    for (std::size_t i = 0; i < cols_; ++i) {
        stats.expect(min_view[i] == minmax_t(expected_mins[i]), "axis-0 min mismatch");
        stats.expect(max_view[i] == minmax_t(expected_maxs[i]), "axis-0 max mismatch");
    }
}

template <typename tensor_type_, typename expected_type_, std::size_t dims_>
void assert_flat_tensor_equals(error_stats_t &stats, tensor_type_ const &tensor,
                               std::array<expected_type_, dims_> const &expected) {
    auto flat = tensor.view().flatten();
    stats.expect(!flat.empty(), "tensor flatten failed");
    auto vec = flat.as_vector();
    stats.expect(vec.size() == dims_, "flattened tensor size mismatch");
    using actual_t = typename tensor_type_::value_type;
    for (std::size_t i = 0; i < dims_; ++i)
        stats.expect(vec[i] == actual_t(expected[i]), "flattened tensor value mismatch");
}

template <typename tensor_type_, typename expected_type_>
void assert_scalar_tensor_equals(error_stats_t &stats, tensor_type_ const &tensor, expected_type_ expected) {
    auto flat = tensor.view().flatten();
    stats.expect(!flat.empty(), "tensor flatten failed");
    auto vec = flat.as_vector();
    stats.expect(vec.size() == 1, "scalar tensor should flatten to one value");
    using actual_t = typename tensor_type_::value_type;
    stats.expect(vec[0u] == actual_t(expected), "scalar tensor value mismatch");
}

template <typename value_type_, std::size_t cols_>
void test_sub_byte_tensor_rank3_axis_case(
    error_stats_t &stats, std::array<int, cols_> const &a00, std::array<int, cols_> const &a01,
    std::array<int, cols_> const &a10, std::array<int, cols_> const &a11,
    std::array<int, cols_ * 2> const &expected_sum_axis0, std::array<int, cols_ * 2> const &expected_sum_axis1,
    std::array<int, 4> const &expected_sum_axis2, std::array<int, cols_ * 2> const &expected_min_axis0,
    std::array<int, cols_ * 2> const &expected_max_axis0, std::array<int, cols_ * 2> const &expected_min_axis1,
    std::array<int, cols_ * 2> const &expected_max_axis1, std::array<int, 4> const &expected_min_axis2,
    std::array<int, 4> const &expected_max_axis2) {
    using tensor_t = nk::tensor<value_type_>;
    using sum_t = typename value_type_::reduce_moments_sum_t;
    using minmax_t = typename value_type_::reduce_minmax_value_t;

    auto t = tensor_t::try_zeros({2, 2, cols_});
    stats.expect(!t.empty(), "rank-3 tensor allocation failed");

    auto span = t.span();
    auto row00 = span.slice_leading(0).slice_leading(0).as_vector();
    auto row01 = span.slice_leading(0).slice_leading(1).as_vector();
    auto row10 = span.slice_leading(1).slice_leading(0).as_vector();
    auto row11 = span.slice_leading(1).slice_leading(1).as_vector();
    for (std::size_t i = 0; i < cols_; ++i) {
        row00[i] = a00[i];
        row01[i] = a01[i];
        row10[i] = a10[i];
        row11[i] = a11[i];
    }

    auto sums0 = nk::try_sum<value_type_>(t.view(), 0);
    auto sums1 = nk::try_sum<value_type_>(t.view(), 1);
    auto sums2 = nk::try_sum<value_type_>(t.view(), 2);
    assert_flat_tensor_equals(stats, sums0, expected_sum_axis0);
    assert_flat_tensor_equals(stats, sums1, expected_sum_axis1);
    assert_flat_tensor_equals(stats, sums2, expected_sum_axis2);

    auto moments0 = nk::try_moments<value_type_>(t.view(), 0);
    auto moments1 = nk::try_moments<value_type_>(t.view(), 1);
    auto moments2 = nk::try_moments<value_type_>(t.view(), 2);
    assert_flat_tensor_equals(stats, moments0.sum, expected_sum_axis0);
    assert_flat_tensor_equals(stats, moments1.sum, expected_sum_axis1);
    assert_flat_tensor_equals(stats, moments2.sum, expected_sum_axis2);

    auto minmax0 = nk::try_minmax<value_type_>(t.view(), 0);
    auto minmax1 = nk::try_minmax<value_type_>(t.view(), 1);
    auto minmax2 = nk::try_minmax<value_type_>(t.view(), 2);
    assert_flat_tensor_equals(stats, minmax0.min_value, expected_min_axis0);
    assert_flat_tensor_equals(stats, minmax0.max_value, expected_max_axis0);
    assert_flat_tensor_equals(stats, minmax1.min_value, expected_min_axis1);
    assert_flat_tensor_equals(stats, minmax1.max_value, expected_max_axis1);
    assert_flat_tensor_equals(stats, minmax2.min_value, expected_min_axis2);
    assert_flat_tensor_equals(stats, minmax2.max_value, expected_max_axis2);
}

error_stats_t test_sub_byte_tensor_axis_reductions() {
    error_stats_t stats(comparison_family_t::exact_k);
    test_sub_byte_tensor_axis_reduction_case<nk::i4x2_t, 4>(stats, {1, -2, 7, -8}, {-3, 4, -5, 6}, {-2, 2, 2, -2},
                                                            {-3, -2, -5, -8}, {1, 4, 7, 6});
    test_sub_byte_tensor_axis_reduction_case<nk::u4x2_t, 4>(stats, {1, 15, 3, 8}, {14, 2, 9, 7}, {15, 17, 12, 15},
                                                            {1, 2, 3, 7}, {14, 15, 9, 8});
    test_sub_byte_tensor_axis_reduction_case<nk::u1x8_t, 8>(stats, {1, 0, 1, 1, 0, 0, 1, 0}, {0, 1, 1, 0, 1, 0, 0, 1},
                                                            {1, 1, 2, 1, 1, 0, 1, 1}, {0, 0, 1, 0, 0, 0, 0, 0},
                                                            {1, 1, 1, 1, 1, 0, 1, 1});
    return stats;
}

error_stats_t test_sub_byte_tensor_rank3_axis_reductions() {
    error_stats_t stats(comparison_family_t::exact_k);
    test_sub_byte_tensor_rank3_axis_case<nk::i4x2_t, 4>(
        stats, {1, -2, 3, -4}, {5, -6, 7, -8}, {-1, 2, -3, 4}, {-5, 6, -7, 7}, {0, 0, 0, 0, 0, 0, 0, -1},
        {6, -8, 10, -12, -6, 8, -10, 11}, {-2, -2, 2, 1}, {-1, -2, -3, -4, -5, -6, -7, -8}, {1, 2, 3, 4, 5, 6, 7, 7},
        {1, -6, 3, -8, -5, 2, -7, 4}, {5, -2, 7, -4, -1, 6, -3, 7}, {-4, -8, -3, -7}, {3, 7, 4, 7});

    test_sub_byte_tensor_rank3_axis_case<nk::u4x2_t, 4>(
        stats, {1, 2, 3, 4}, {5, 6, 7, 8}, {14, 13, 12, 11}, {10, 9, 8, 7}, {15, 15, 15, 15, 15, 15, 15, 15},
        {6, 8, 10, 12, 24, 22, 20, 18}, {10, 26, 50, 34}, {1, 2, 3, 4, 5, 6, 7, 7}, {14, 13, 12, 11, 10, 9, 8, 8},
        {1, 2, 3, 4, 10, 9, 8, 7}, {5, 6, 7, 8, 14, 13, 12, 11}, {1, 5, 11, 7}, {4, 8, 14, 10});

    test_sub_byte_tensor_rank3_axis_case<nk::u1x8_t, 8>(
        stats, {1, 0, 1, 0, 1, 0, 1, 0}, {0, 1, 0, 1, 0, 1, 0, 1}, {1, 1, 0, 0, 1, 1, 0, 0}, {0, 0, 1, 1, 0, 0, 1, 1},
        {2, 1, 1, 0, 2, 1, 1, 0, 0, 1, 1, 2, 0, 1, 1, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        {4, 4, 4, 4}, {1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1},
        {1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0}, {1, 1, 1, 1});
    return stats;
}

error_stats_t test_rank1_negative_stride_reductions() {
    error_stats_t stats(comparison_family_t::exact_k);
    using value_t = nk::f32_t;
    using sum_t = typename value_t::reduce_moments_sum_t;
    using minmax_t = typename value_t::reduce_minmax_value_t;

    value_t data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    nk::shape_storage_<8> shape {};
    shape.rank = 1;
    shape.extents[0] = 4;
    shape.strides[0] = -static_cast<std::ptrdiff_t>(sizeof(value_t));
    nk::tensor_view<value_t> reversed(reinterpret_cast<char const *>(data + 3), shape);

    auto m = nk::moments(reversed);
    auto mm = nk::minmax(reversed);
    stats.expect(m.sum == sum_t(10.0), "negative-stride sum mismatch");
    stats.expect(m.sumsq == typename value_t::reduce_moments_sumsq_t(30.0), "negative-stride sumsq mismatch");
    stats.expect(mm.min_value == minmax_t(1.0f), "negative-stride min mismatch");
    stats.expect(mm.max_value == minmax_t(4.0f), "negative-stride max mismatch");
    return stats;
}

error_stats_t test_rank1_axis_reductions() {
    error_stats_t stats(comparison_family_t::exact_k);
    auto v = nk::tensor<nk::i8_t>::try_zeros({4});
    stats.expect(!v.empty(), "rank-1 tensor allocation failed");
    auto values = v.as_vector_span();
    values[0u] = 4;
    values[1u] = -2;
    values[2u] = 7;
    values[3u] = -5;

    auto sums = nk::try_sum<nk::i8_t>(v.view(), 0);
    auto moments = nk::try_moments<nk::i8_t>(v.view(), 0);
    auto mins = nk::try_min<nk::i8_t>(v.view(), 0);
    auto maxs = nk::try_max<nk::i8_t>(v.view(), 0);
    auto argmins = nk::try_argmin<nk::i8_t>(v.view(), 0);
    auto argmaxs = nk::try_argmax<nk::i8_t>(v.view(), 0);

    stats.expect(!sums.empty() && !moments.sum.empty(), "rank-1 axis moments failed");
    stats.expect(!mins.empty() && !maxs.empty() && !argmins.empty() && !argmaxs.empty(), "rank-1 axis minmax failed");
    stats.expect(sums.rank() == 0 && moments.sum.rank() == 0,
                 "collapsed rank-1 reductions should produce rank-0 tensors");
    assert_scalar_tensor_equals(stats, sums, 4);
    assert_scalar_tensor_equals(stats, moments.sum, 4);
    assert_scalar_tensor_equals(stats, mins, -5);
    assert_scalar_tensor_equals(stats, maxs, 7);
    assert_scalar_tensor_equals(stats, argmins, 3);
    assert_scalar_tensor_equals(stats, argmaxs, 2);
    return stats;
}

error_stats_t test_packed_tensor_fail_closed_views() {
    error_stats_t stats(comparison_family_t::exact_k);
    auto packed = nk::tensor<nk::i4x2_t>::try_zeros({2, 4});
    stats.expect(!packed.empty(), "packed tensor allocation failed");
    stats.expect(packed.view().transpose().empty(), "packed transpose should fail closed");
    stats.expect((!packed(1, nk::slice).empty()), "packed row slice should remain supported");
    stats.expect((packed(1, 2, nk::slice).empty()), "packed scalar trailing slice should fail closed");
#if NK_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    stats.expect((!packed[1, nk::slice].empty()), "packed operator[] row slice should remain supported");
    stats.expect((packed[1, 2, nk::slice].empty()), "packed operator[] scalar trailing slice should fail closed");
#endif
    return stats;
}

/**
 *  @brief Smoke-test for the vector-shaped reduction wrappers (`nk::moments`/`minmax`/`sum`/...).
 *  Exercises every accessor variant on a small random vector. Numerical accuracy of the
 *  underlying kernels is validated by `test_reduce_moments` / `test_reduce_minmax` above.
 */
template <typename value_type_>
error_stats_t test_vector_reductions_for_type() {
    error_stats_t stats(comparison_family_t::exact_k);
    auto v = make_vector<value_type_>(32);
    std::mt19937 generator(42);
    fill_random(generator, v);
    auto view = nk::vector_view<value_type_>(v.values_data(), static_cast<std::size_t>(v.size()));

    auto mm = nk::minmax(view);
    stats.expect(!(mm.max_value < mm.min_value), "minmax inverted");
    stats.expect(mm.min_index < v.size(), "argmin out of range");
    stats.expect(mm.max_index < v.size(), "argmax out of range");
    stats.expect(nk::min(view) == mm.min_value, "min disagrees with minmax");
    stats.expect(nk::max(view) == mm.max_value, "max disagrees with minmax");
    stats.expect(nk::argmin(view) == mm.min_index, "argmin disagrees with minmax");
    stats.expect(nk::argmax(view) == mm.max_index, "argmax disagrees with minmax");
    stats.expect(nk::sum(view) == nk::moments(view).sum, "sum disagrees with moments");
    return stats;
}

void test_vector_types() {
    error_stats_section_t check;
    check.section("Vectors", nk_cap_serial_k);

    check("vector_basics_f32", test_vector_basics<float>);
    check("vector_basics_f64", test_vector_basics<double>);
    check("vector_basics_f16", test_vector_basics<nk::f16_t>);
    check("vector_basics_bf16", test_vector_basics<nk::bf16_t>);
    check("vector_basics_i8", test_vector_basics<nk::i8_t>);
    check("vector_basics_f32c", test_vector_basics<nk::f32c_t>);
    check("vector_basics_f64c", test_vector_basics<std::complex<double>>);
    check("vector_basics_i4", test_vector_basics<nk::i4x2_t>);
    check("vector_basics_u1", test_vector_basics<nk::u1x8_t>);

    check("vector_reductions_f32", test_vector_reductions_for_type<nk::f32_t>);
    check("vector_reductions_f64", test_vector_reductions_for_type<nk::f64_t>);
    check("vector_reductions_f16", test_vector_reductions_for_type<nk::f16_t>);
    check("vector_reductions_bf16", test_vector_reductions_for_type<nk::bf16_t>);
    check("vector_reductions_i8", test_vector_reductions_for_type<nk::i8_t>);
    check("vector_reductions_u8", test_vector_reductions_for_type<nk::u8_t>);

    check("vector_signed_indexing", test_signed_indexing);
    check("vector_integral_indexing", test_integral_indexing_api);
    check("vector_move_semantics", test_move_semantics);
    check("vector_swap", test_swap);
    check("vector_view_span_rev", test_view_span_rev);
    check("vector_range_slicing", test_range_slicing);
    check("vector_sub_byte_i4", test_sub_byte_i4x2);
    check("vector_sub_byte_u1", test_sub_byte_u1x8);
    check("vector_custom_allocator", test_custom_allocator);

#if NK_TEST_FORMAT_
    check("vector_format_scalars", test_format_scalars);
#endif
}

/**
 *  @brief Explicit template instantiation test for all tensor-level operations.
 *
 *  Forces the compiler to fully instantiate every type × operation combination,
 *  catching signature mismatches, missing type traits, and implicit conversion errors
 *  that syntax-only checks miss.
 */
template <typename value_type_>
error_stats_t test_tensor_ops_for_type() {
    error_stats_t stats(comparison_family_t::exact_k);
    using tensor_t = nk::tensor<value_type_>;

    // Create small test tensors
    auto a = tensor_t::try_zeros({4, 8});
    auto b = tensor_t::try_zeros({4, 8});
    stats.expect(!a.empty() && !b.empty(), "tensor allocation");

    auto av = a.view();
    auto bv = b.view();

    // Scalar reductions
    { [[maybe_unused]] auto r = nk::sum<value_type_>(av); }
    { [[maybe_unused]] auto r = nk::moments<value_type_>(av); }
    { [[maybe_unused]] auto r = nk::min<value_type_>(av); }
    { [[maybe_unused]] auto r = nk::max<value_type_>(av); }
    { [[maybe_unused]] auto r = nk::argmin<value_type_>(av); }
    { [[maybe_unused]] auto r = nk::argmax<value_type_>(av); }
    { [[maybe_unused]] auto r = nk::minmax<value_type_>(av); }

    // Axis reductions
    stats.expect(!(nk::try_sum<value_type_>(av, 0)).empty(), "try_sum returned empty");
    stats.expect(!(nk::try_sum<value_type_>(av, 1, nk::keep_dims_k)).empty(), "try_sum returned empty");
    { [[maybe_unused]] auto r = nk::try_moments<value_type_>(av, 1); }
    { [[maybe_unused]] auto r = nk::try_minmax<value_type_>(av, 0); }
    { [[maybe_unused]] auto r = nk::try_minmax<value_type_>(av, 1, nk::keep_dims_k); }
    stats.expect(!(nk::try_min<value_type_>(av, 0)).empty(), "try_min returned empty");
    stats.expect(!(nk::try_min<value_type_>(av, 1, nk::keep_dims_k)).empty(), "try_min returned empty");
    stats.expect(!(nk::try_max<value_type_>(av, 0)).empty(), "try_max returned empty");
    stats.expect(!(nk::try_max<value_type_>(av, 1, nk::keep_dims_k)).empty(), "try_max returned empty");
    stats.expect(!(nk::try_argmin<value_type_>(av, 0)).empty(), "try_argmin returned empty");
    stats.expect(!(nk::try_argmax<value_type_>(av, 1, nk::keep_dims_k)).empty(), "try_argmax returned empty");

    // Elementwise binary
    stats.expect(!(nk::try_add<value_type_>(av, bv)).empty(), "try_add returned empty");
    stats.expect(!(nk::try_sub<value_type_>(av, bv)).empty(), "try_sub returned empty");
    stats.expect(!(nk::try_mul<value_type_>(av, bv)).empty(), "try_mul returned empty");

    // Elementwise binary with scalar
    using scale_t = typename value_type_::scale_t;
    scale_t scalar {1};
    stats.expect(!(nk::try_add<value_type_>(av, scalar)).empty(), "try_add returned empty");
    stats.expect(!(nk::try_sub<value_type_>(av, scalar)).empty(), "try_sub returned empty");
    stats.expect(!(nk::try_mul<value_type_>(av, scalar)).empty(), "try_mul returned empty");

    // Elementwise into
    auto out = tensor_t::try_zeros({4, 8});
    stats.expect(nk::add<value_type_>(av, bv, out.span()), "add into span failed");
    stats.expect(nk::sub<value_type_>(av, bv, out.span()), "sub into span failed");
    stats.expect(nk::mul<value_type_>(av, bv, out.span()), "mul into span failed");
    stats.expect(nk::add<value_type_>(av, scalar, out.span()), "add into span failed");
    stats.expect(nk::sub<value_type_>(av, scalar, out.span()), "sub into span failed");
    stats.expect(nk::mul<value_type_>(av, scalar, out.span()), "mul into span failed");

    // Affine
    scale_t alpha {1}, beta {0};
    stats.expect(!(nk::try_scale<value_type_>(av, alpha, beta)).empty(), "try_scale returned empty");
    stats.expect(!(nk::try_blend<value_type_>(av, bv, alpha, beta)).empty(), "try_blend returned empty");
    stats.expect(!(nk::try_fma<value_type_>(av, bv, av, alpha, beta)).empty(), "try_fma returned empty");
    stats.expect(nk::scale<value_type_>(av, alpha, beta, out.span()), "scale into span failed");
    stats.expect(nk::blend<value_type_>(av, bv, alpha, beta, out.span()), "blend into span failed");
    stats.expect(nk::fma<value_type_>(av, bv, av, alpha, beta, out.span()), "fma into span failed");

    // try_from 1D
    {
        auto from1d = tensor_t::try_from({value_type_ {}, value_type_ {}, value_type_ {}});
        stats.expect(!from1d.empty(), "try_from 1D failed");
        stats.expect(from1d.rank() == 1 && from1d.numel() == 3, "try_from 1D shape mismatch");
    }

    // try_from 2D
    {
        auto from2d = tensor_t::try_from({{value_type_ {}, value_type_ {}}, {value_type_ {}, value_type_ {}}});
        stats.expect(!from2d.empty(), "try_from 2D failed");
        stats.expect(from2d.rank() == 2 && from2d.extent(0) == 2 && from2d.extent(1) == 2,
                     "try_from 2D shape mismatch");
    }

    // row() access
    {
        auto row0 = a.row(0);
        stats.expect(row0.rank() == 1 && row0.extent(0) == 8, "row() shape mismatch");
    }

    // Convenience view constructor (ptr, rows, cols)
    {
        nk::tensor_view<value_type_> view_from_ptr(a.data(), 4, 8);
        stats.expect(view_from_ptr.rank() == 2 && view_from_ptr.extent(0) == 4, "convenience view ctor mismatch");
    }
    return stats;
}

template <typename value_type_>
error_stats_t test_tensor_symmetric_for_type() {
    error_stats_t stats(comparison_family_t::exact_k);
    using tensor_t = nk::tensor<value_type_>;
    auto a = tensor_t::try_zeros({4, 8});
    auto am = a.as_matrix_view();

    stats.expect(!(nk::try_dots_symmetric<value_type_>(am)).empty(), "try_dots_symmetric returned empty");
    stats.expect(!(nk::try_angulars_symmetric<value_type_>(am)).empty(), "try_angulars_symmetric returned empty");
    stats.expect(!(nk::try_euclideans_symmetric<value_type_>(am)).empty(), "try_euclideans_symmetric returned empty");
    return stats;
}

template <typename value_type_>
error_stats_t test_tensor_packed_for_type() {
    error_stats_t stats(comparison_family_t::exact_k);
    using tensor_t = nk::tensor<value_type_>;
    auto a = tensor_t::try_zeros({4, 8});
    auto b = tensor_t::try_zeros({6, 8});

    // packed_matrix
    auto bm = b.as_matrix_view();
    auto packed = nk::packed_matrix<value_type_, nk::aligned_allocator<char>>::try_pack(bm);
    auto am = a.as_matrix_view();
    auto result = nk::matrix<typename value_type_::dot_result_t>::try_zeros({4, 6});
    stats.expect(!packed.empty(), "packed matrix empty");
    stats.expect(!result.empty(), "packed result empty");
    nk::dots_packed<value_type_>(am, packed, result.span());
    return stats;
}

template <typename value_type_>
error_stats_t test_tensor_maxsim_for_type() {
    error_stats_t stats(comparison_family_t::exact_k);
    using tensor_t = nk::tensor<value_type_>;
    auto q = tensor_t::try_zeros({3, 16});
    auto d = tensor_t::try_zeros({5, 16});

    auto qm = q.as_matrix_view();
    auto dm = d.as_matrix_view();

    auto pq = nk::packed_maxsim<value_type_>::try_pack(qm);
    auto pd = nk::packed_maxsim<value_type_>::try_pack(dm);
    stats.expect(!pq.empty(), "packed query empty");
    stats.expect(!pd.empty(), "packed document empty");
    { [[maybe_unused]] auto r = nk::maxsim(pq, pd); }
    return stats;
}

error_stats_t test_view_overloads() {
    error_stats_t stats(comparison_family_t::exact_k);
    nk::f32_t a_data[8] {}, b_data[8] {}, c_data[64] {};
    nk::f32_t result {};
    auto a_view = nk::vector_view<nk::f32_t>(a_data, 8u);
    auto b_view = nk::vector_view<nk::f32_t>(b_data, 8u);

    // Inputs are all zero, so the metrics that are defined there must come back zero.
    nk::dot(a_view, b_view, 8, &result);
    stats.expect(result == nk::f32_t(0), "dot of zero vectors");
    nk::euclidean(a_view, b_view, 8, &result);
    stats.expect(result == nk::f32_t(0), "euclidean of zero vectors");
    nk::sqeuclidean(a_view, b_view, 8, &result);
    stats.expect(result == nk::f32_t(0), "sqeuclidean of zero vectors");
    nk::angular(a_view, b_view, 8, &result);

    auto c_view = nk::vector_view<nk::f32_t>(c_data, 64u);
    nk::bilinear(a_view, b_view, c_view, 8, &result);
    nk::mahalanobis(a_view, b_view, c_view, 8, &result);
    return stats;
}

error_stats_t test_custom_allocator_try_fns() {
    error_stats_t stats(comparison_family_t::exact_k);
    using custom_alloc_t = nk::aligned_allocator<nk::f32_t, 128>;
    auto a = nk::tensor<nk::f32_t>::try_zeros({4, 8});
    auto av = a.view();

    stats.expect(!nk::try_scale<nk::f32_t, 8, custom_alloc_t>(av, 1.0, 0.0).empty(), "try_scale returned empty");
    stats.expect(!nk::try_sin<nk::f32_t, 8, custom_alloc_t>(av).empty(), "try_sin returned empty");

    using sum_alloc_t = nk::aligned_allocator<nk::f64_t, 128>;
    stats.expect(!nk::try_sum<nk::f32_t, 8, sum_alloc_t>(av, 0).empty(), "try_sum returned empty");
    return stats;
}

template <typename from_type_, typename to_type_>
error_stats_t test_cast_for_types() {
    error_stats_t stats(comparison_family_t::exact_k);
    auto src = make_vector<from_type_>(64);
    auto dst = make_vector<to_type_>(64);
    std::mt19937 generator(42);
    fill_random(generator, src);

    auto src_view = nk::vector_view<from_type_>(src.values_data(), static_cast<std::size_t>(src.size()));
    auto dst_span = nk::vector_span<to_type_>(dst.values_data(), static_cast<std::size_t>(dst.size()));

    // The pointer-level and view/span APIs must agree element for element.
    auto dst_via_views = make_vector<to_type_>(64);
    auto dst_via_views_span = nk::vector_span<to_type_>(dst_via_views.values_data(),
                                                        static_cast<std::size_t>(dst_via_views.size()));
    nk::cast<from_type_, to_type_>(src.values_data(), src.size(), dst.values_data());
    nk::cast<from_type_, to_type_>(src_view, dst_via_views_span);
    for (std::size_t i = 0; i < dst.size(); ++i) stats.accumulate(dst[i], dst_via_views[i]);
    return stats;
}

#if NK_TEST_FORMAT_
error_stats_t test_format_scalars() {
    error_stats_t stats(comparison_family_t::exact_k);
    // Float scalar formatters
    { [[maybe_unused]] auto s = std::format("{}", nk::f16_t(3.14f)); }
    { [[maybe_unused]] auto s = std::format("{:#}", nk::f16_t(3.14f)); }
    { [[maybe_unused]] auto s = std::format("{:.2f}", nk::f16_t(3.14f)); }
    { [[maybe_unused]] auto s = std::format("{:x}", nk::f16_t(3.14f)); }
    { [[maybe_unused]] auto s = std::format("{:b}", nk::f16_t(3.14f)); }
    { [[maybe_unused]] auto s = std::format("{}", nk::bf16_t(2.5f)); }
    { [[maybe_unused]] auto s = std::format("{}", nk::e4m3_t(1.0f)); }
    { [[maybe_unused]] auto s = std::format("{}", nk::e5m2_t(1.0f)); }
    { [[maybe_unused]] auto s = std::format("{}", nk::e2m3_t(1.0f)); }
    { [[maybe_unused]] auto s = std::format("{}", nk::e3m2_t(1.0f)); }

    // Packed type formatters
    { [[maybe_unused]] auto s = std::format("{}", nk::i4x2_t {}); }
    { [[maybe_unused]] auto s = std::format("{:x}", nk::u4x2_t {}); }
    { [[maybe_unused]] auto s = std::format("{}", nk::u1x8_t {}); }

    // Complex type formatters
    { [[maybe_unused]] auto s = std::format("{}", nk::f16c_t(nk::f16_t(1), nk::f16_t(2))); }
    { [[maybe_unused]] auto s = std::format("{:#}", nk::bf16c_t(nk::bf16_t(1), nk::bf16_t(2))); }

    // Sub-byte ref formatters
    nk_i4x2_t packed_i = 0x53;
    nk::sub_byte_ref<nk::i4x2_t> iref(&packed_i, 0);
    stats.expect(std::format("{}", iref) == "3", "i4 sub_byte_ref default format");
    stats.expect(std::format("{:x}", iref) == "3", "i4 sub_byte_ref hex format");
    stats.expect(std::format("{:b}", iref) == "0011", "i4 sub_byte_ref binary format");
    stats.expect(std::format("{:#}", iref) == "3 [0x3]", "i4 sub_byte_ref annotated format");

    nk_u4x2_t packed_u = 0xA7;
    nk::sub_byte_ref<nk::u4x2_t> uref(&packed_u, 1);
    stats.expect(std::format("{}", uref) == "10", "u4 sub_byte_ref default format");
    stats.expect(std::format("{:x}", uref) == "a", "u4 sub_byte_ref hex format");
    stats.expect(std::format("{:b}", uref) == "1010", "u4 sub_byte_ref binary format");

    nk_u1x8_t packed_b = 0x05;
    nk::sub_byte_ref<nk::u1x8_t> bref(&packed_b, 0);
    stats.expect(std::format("{}", bref) == "1", "u1 sub_byte_ref format");
    return stats;
}
#endif // NK_TEST_FORMAT_

/**
 *  @brief Smoke-test for the tensor-shaped trig wrappers (`nk::try_sin`/`cos`/`atan`).
 *  Runs allocating + into-span variants on a small zero tensor — just exercises the dispatch
 *  paths, not the numerical accuracy (the latter is covered by the kernel tests above).
 */
template <typename value_type_>
error_stats_t test_tensor_trig_for_type() {
    error_stats_t stats(comparison_family_t::exact_k);
    using tensor_t = nk::tensor<value_type_>;
    auto a = tensor_t::try_zeros({4, 8});
    auto out = tensor_t::try_zeros({4, 8});
    auto av = a.view();

    stats.expect(!(nk::try_sin<value_type_>(av)).empty(), "try_sin returned empty");
    stats.expect(!(nk::try_cos<value_type_>(av)).empty(), "try_cos returned empty");
    stats.expect(!(nk::try_atan<value_type_>(av)).empty(), "try_atan returned empty");
    stats.expect(nk::sin<value_type_>(av, out.span()), "sin into span failed");
    stats.expect(nk::cos<value_type_>(av, out.span()), "cos into span failed");
    stats.expect(nk::atan<value_type_>(av, out.span()), "atan into span failed");
    return stats;
}

void test_tensor_ops() {
    error_stats_section_t check;
    check.section("Tensors", nk_cap_serial_k);

    check("tensor_ops_f32", test_tensor_ops_for_type<nk::f32_t>);
    check("tensor_ops_f64", test_tensor_ops_for_type<nk::f64_t>);
    check("tensor_ops_f16", test_tensor_ops_for_type<nk::f16_t>);
    check("tensor_ops_bf16", test_tensor_ops_for_type<nk::bf16_t>);
    check("tensor_ops_i8", test_tensor_ops_for_type<nk::i8_t>);
    check("tensor_ops_u8", test_tensor_ops_for_type<nk::u8_t>);

    check("tensor_trig_f32", test_tensor_trig_for_type<nk::f32_t>);
    check("tensor_trig_f64", test_tensor_trig_for_type<nk::f64_t>);
    check("tensor_trig_f16", test_tensor_trig_for_type<nk::f16_t>);
    check("tensor_trig_bf16", test_tensor_trig_for_type<nk::bf16_t>);

    check("tensor_axis_sub_byte", test_sub_byte_tensor_axis_reductions);
    check("tensor_axis_rank3_packed", test_sub_byte_tensor_rank3_axis_reductions);
    check("tensor_negative_stride", test_rank1_negative_stride_reductions);
    check("tensor_rank1_axis", test_rank1_axis_reductions);
    check("tensor_operator_indexing", test_tensor_operator_indexing);
    check("tensor_packed_operator_indexing", test_packed_tensor_operator_indexing);
    check("tensor_fail_closed_views", test_packed_tensor_fail_closed_views);

    check("tensor_symmetric_f32", test_tensor_symmetric_for_type<nk::f32_t>);
    check("tensor_symmetric_f64", test_tensor_symmetric_for_type<nk::f64_t>);
    check("tensor_symmetric_f16", test_tensor_symmetric_for_type<nk::f16_t>);
    check("tensor_symmetric_bf16", test_tensor_symmetric_for_type<nk::bf16_t>);
    check("tensor_symmetric_i8", test_tensor_symmetric_for_type<nk::i8_t>);

    check("tensor_packed_f32", test_tensor_packed_for_type<nk::f32_t>);
    check("tensor_packed_f64", test_tensor_packed_for_type<nk::f64_t>);
    check("tensor_packed_f16", test_tensor_packed_for_type<nk::f16_t>);
    check("tensor_packed_bf16", test_tensor_packed_for_type<nk::bf16_t>);
    check("tensor_packed_i8", test_tensor_packed_for_type<nk::i8_t>);

    check("tensor_maxsim_bf16", test_tensor_maxsim_for_type<nk::bf16_t>);
    check("tensor_maxsim_f32", test_tensor_maxsim_for_type<nk::f32_t>);
    check("tensor_maxsim_f16", test_tensor_maxsim_for_type<nk::f16_t>);

    check("tensor_view_overloads", test_view_overloads);
    check("tensor_custom_allocator_try", test_custom_allocator_try_fns);

    check("tensor_cast_f32_to_f16", test_cast_for_types<nk::f32_t, nk::f16_t>);
    check("tensor_cast_f16_to_f32", test_cast_for_types<nk::f16_t, nk::f32_t>);
    check("tensor_cast_f32_to_bf16", test_cast_for_types<nk::f32_t, nk::bf16_t>);
    check("tensor_cast_bf16_to_f32", test_cast_for_types<nk::bf16_t, nk::f32_t>);
    check("tensor_cast_f32_to_e4m3", test_cast_for_types<nk::f32_t, nk::e4m3_t>);
    check("tensor_cast_e4m3_to_f32", test_cast_for_types<nk::e4m3_t, nk::f32_t>);
    check("tensor_cast_i8_to_i32", test_cast_for_types<nk::i8_t, nk::i32_t>);
    check("tensor_cast_f64_to_f32", test_cast_for_types<nk::f64_t, nk::f32_t>);
}
