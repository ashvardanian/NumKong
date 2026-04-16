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
void test_format_scalars();
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
void test_vector_basics() {
    constexpr std::size_t dims_per_value = nk::dimensions_per_value<value_type_>();
    constexpr std::size_t test_dims = 64 * dims_per_value;
    auto v = make_vector<value_type_>(test_dims);
    assert(v.size() == test_dims);
    assert(v.size_values() == test_dims / dims_per_value);
    std::size_t count = 0;
    for (auto it = v.begin(); it != v.end(); ++it) ++count;
    assert(count == test_dims);
}

void test_signed_indexing() {
    auto v = make_vector<float>(100);
    v[50] = 3.14f;
    assert(v[50] == 3.14f && "float operator[] failed");
    v[-1] = 42.0f;
    assert(v[99] == 42.0f && "float signed indexing failed");
}

void test_integral_indexing_api() {
    auto v = make_vector<float>(5);
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>(i + 1);

    auto view = nk::vector_view<float>(v.values_data(), unsigned(v.size()));
    auto span = nk::vector_span<float>(v.values_data(), unsigned(v.size()));
    auto strided = nk::vector_view<float>(reinterpret_cast<char const *>(v.values_data()), 3u, sizeof(float));

    assert(v[std::size_t {2}] == 3.0f && "vector unsigned indexing failed");
    assert(view[2u] == 3.0f && "view unsigned indexing failed");
    assert(view[std::ptrdiff_t {-1}] == 5.0f && "view signed indexing failed");
    assert(span[unsigned {3}] == 4.0f && "span unsigned indexing failed");
    assert(strided[2u] == 3.0f && "raw view unsigned stride ctor failed");

    auto sub = view[nk::range(1u, 4u)];
    assert(sub.size() == 3 && "unsigned range size mismatch");
    assert(sub[0u] == 2.0f && "unsigned range first element mismatch");

    auto tail = view[nk::range(-3, -1)];
    assert(tail.size() == 2 && "signed range size mismatch");
    assert(tail[0u] == 3.0f && "signed range first element mismatch");
    assert(tail[1u] == 4.0f && "signed range last element mismatch");
}

void test_tensor_operator_indexing() {
    auto t = nk::tensor<float>::try_zeros({2, 3});
    assert(!t.empty() && "tensor allocation failed");

    for (int i = 0; i < 6; ++i) t[i] = static_cast<float>(i + 1);

    assert(t[0] == 1.0f && "flat tensor lookup failed");
    assert(t[-1] == 6.0f && "negative flat tensor lookup failed");
    assert((t(0, 0) == 1.0f) && "exact tensor lookup failed");
    assert((t(1, -1) == 6.0f) && "negative exact tensor lookup failed");
#if NK_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    assert((t[0, 0] == 1.0f) && "exact tensor lookup via operator[] failed");
    assert((t[1, -1] == 6.0f) && "negative exact tensor lookup via operator[] failed");
#endif

    auto whole = t[nk::slice];
    assert(whole.rank() == 2 && "slice identity rank mismatch");
    assert(whole.extent(0) == 2 && whole.extent(1) == 3 && "slice identity extents mismatch");

    auto row1 = t(1, nk::slice);
    assert(row1.rank() == 1 && "row slice rank mismatch");
    assert(row1.extent(0) == 3 && "row slice extent mismatch");
    assert(row1[0] == 4.0f && row1[-1] == 6.0f && "row slice values mismatch");
    row1[1] = 42.0f;
    assert((t(1, 1) == 42.0f) && "row slice write-through failed");
#if NK_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    assert((t[1, 1] == 42.0f) && "operator[] row slice write-through failed");
    auto row1_subscript = t[1, nk::slice];
    assert(row1_subscript.extent(0) == row1.extent(0) && "operator[] row slice mismatch");
#endif

    auto cell = t(1, 1, nk::slice);
    assert(cell.rank() == 0 && "scalar slice rank mismatch");
    assert(cell.scalar() == 42.0f && "scalar slice value mismatch");
    cell.scalar_ref() = 24.0f;
    assert((t(1, 1) == 24.0f) && "scalar slice write-through failed");
#if NK_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    assert((t[1, 1] == 24.0f) && "operator[] scalar slice write-through failed");
    auto cell_subscript = t[1, 1, nk::slice];
    assert(cell_subscript.rank() == 0 && "operator[] scalar slice rank mismatch");
#endif

    auto const &ct = t;
    auto const last_row = ct(-1, nk::slice);
    assert(last_row.rank() == 1 && "const row slice rank mismatch");
    assert(last_row[0] == 4.0f && "const row slice mismatch");
#if NK_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    auto const last_row_subscript = ct[-1, nk::slice];
    assert(last_row_subscript.rank() == 1 && "operator[] const row slice mismatch");
#endif

    auto cube = nk::tensor<float>::try_zeros({2, 3, 4});
    assert(!cube.empty() && "cube allocation failed");
    for (int i = 0; i < 24; ++i) cube[i] = static_cast<float>(i);

    auto plane = cube(1, nk::slice);
    assert(plane.rank() == 2 && plane.extent(0) == 3 && plane.extent(1) == 4 && "plane slice mismatch");
    auto line = cube(1, 2, nk::slice);
    assert(line.rank() == 1 && line.extent(0) == 4 && "line slice mismatch");
    assert((line[3] == cube(1, 2, 3)) && "line slice element mismatch");
    auto point = cube(1, 2, 3, nk::slice);
    assert((point.rank() == 0 && point.scalar() == cube(1, 2, 3)) && "point slice mismatch");
#if NK_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    auto plane_subscript = cube[1, nk::slice];
    auto line_subscript = cube[1, 2, nk::slice];
    auto point_subscript = cube[1, 2, 3, nk::slice];
    assert((line_subscript[3] == cube[1, 2, 3]) && "operator[] line slice element mismatch");
    assert((point_subscript.scalar() == cube[1, 2, 3]) && "operator[] point slice mismatch");
    assert(plane_subscript.rank() == 2 && "operator[] plane slice rank mismatch");
#endif

    // all_t slicing: extract a column
    auto second_column = t(nk::all, 1, nk::slice);
    assert(second_column.rank() == 1 && "all_t column rank mismatch");
    assert(second_column.numel() == 2 && "all_t column numel mismatch");

    // range slicing: extract a sub-range of rows
    auto first_two_planes = cube(nk::range(0, 2), nk::slice);
    assert(first_two_planes.rank() == 3 && "range slice rank mismatch");
    assert(first_two_planes.extent(0) == 2 && "range slice extent mismatch");

    // combined: range + all_t + slice on a 3D tensor
    auto sub = cube(nk::range(0, 2), nk::all, nk::slice);
    assert(sub.rank() == 3 && sub.extent(0) == 2 && sub.extent(1) == 3 && "range+all slice mismatch");
#if NK_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    auto second_column_subscript = t[nk::all, 1, nk::slice];
    auto first_two_planes_subscript = cube[nk::range(0, 2), nk::slice];
    auto sub_subscript = cube[nk::range(0, 2), nk::all, nk::slice];
    assert(second_column_subscript.numel() == 2 && "operator[] all_t column mismatch");
    assert(first_two_planes_subscript.extent(0) == 2 && "operator[] range slice mismatch");
    assert(sub_subscript.extent(1) == 3 && "operator[] range+all slice mismatch");
#endif

    // row() access
    auto row0 = t.row(0);
    assert(row0.rank() == 1 && row0.extent(0) == 3 && "row() rank/extent mismatch");
    auto row0_via_slice = t(0, nk::slice);
    assert(row0[0] == row0_via_slice[0] && "row() should match t(0, slice)");
#if NK_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    auto row0_via_subscript = t[0, nk::slice];
    assert(row0[0] == row0_via_subscript[0] && "row() should match t[0, slice]");
#endif
}

void test_packed_tensor_operator_indexing() {
    auto t4 = nk::tensor<nk::u4x2_t>::try_zeros({2, 4});
    assert(!t4.empty() && "packed u4 tensor allocation failed");

    for (int i = 0; i < 8; ++i) t4[i] = i + 1;

    assert(int(t4[0]) == 1 && "packed flat lookup failed");
    assert(int(t4[-1]) == 8 && "packed negative flat lookup failed");
    assert((int(t4(0, 3)) == 4) && "packed exact lookup failed");
    assert((int(t4(1, -1)) == 8) && "packed negative exact lookup failed");
#if NK_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    assert((int(t4[0, 3]) == 4) && "packed operator[] exact lookup failed");
    assert((int(t4[1, -1]) == 8) && "packed operator[] negative exact lookup failed");
#endif

    auto second_row = t4(1, nk::slice);
    assert(second_row.rank() == 1 && second_row.extent(0) == 4 && "packed row slice rank mismatch");
    assert(int(second_row[0]) == 5 && int(second_row[-1]) == 8 && "packed row slice values mismatch");
    second_row[1] = 14;
    assert((int(t4(1, 1)) == 14) && "packed row slice write-through failed");
#if NK_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    auto second_row_subscript = t4[1, nk::slice];
    assert(second_row_subscript.extent(0) == 4 && "packed operator[] row slice rank mismatch");
    assert((int(t4[1, 1]) == 14) && "packed operator[] row slice write-through failed");
#endif

    auto t1 = nk::tensor<nk::u1x8_t>::try_zeros({2, 8});
    assert(!t1.empty() && "packed u1 tensor allocation failed");
    t1[0] = true;
    t1[7] = true;
    t1[11] = true;
    t1[-1] = true;

    assert(bool(t1[0]) && "packed bit flat lookup failed");
    assert((bool(t1(0, 7))) && "packed bit exact lookup failed");
    assert((bool(t1(1, 3))) && "packed bit second-row lookup failed");
    assert(bool(t1[-1]) && "packed bit negative flat lookup failed");
#if NK_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    assert((bool(t1[0, 7])) && "packed bit operator[] exact lookup failed");
    assert((bool(t1[1, 3])) && "packed bit operator[] second-row lookup failed");
#endif

    auto bits = t1(1, nk::slice);
    assert(bits.rank() == 1 && bits.extent(0) == 8 && "packed bit slice rank mismatch");
    bits[4] = true;
    assert((bool(t1(1, 4))) && "packed bit slice write-through failed");
#if NK_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    auto bits_subscript = t1[1, nk::slice];
    assert(bits_subscript.extent(0) == 8 && "packed bit operator[] slice rank mismatch");
    assert((bool(t1[1, 4])) && "packed bit operator[] slice write-through failed");
#endif
}

void test_move_semantics() {
    auto v1 = make_vector<nk::f32_t>(100);
    v1[50] = nk::f32_t(42.0f);

    nk::vector<nk::f32_t> v2 = std::move(v1);
    assert(v2.size() == 100 && "move ctor size mismatch");
    assert(v2[50] == nk::f32_t(42.0f) && "move ctor value mismatch");
    assert(v1.size() == 0 && "moved-from vector not empty"); // NOLINT(bugprone-use-after-move)

    nk::vector<nk::f32_t> v3;
    v3 = std::move(v2);
    assert(v3.size() == 100 && "move assign size mismatch");
    assert(v3[50] == nk::f32_t(42.0f) && "move assign value mismatch");
}

void test_swap() {
    auto v1 = make_vector<nk::i8_t>(10);
    auto v2 = make_vector<nk::i8_t>(20);
    v1[0] = nk::i8_t(1);
    v2[0] = nk::i8_t(2);

    swap(v1, v2);
    assert(v1.size() == 20 && "swap v1 size mismatch");
    assert(v2.size() == 10 && "swap v2 size mismatch");
    assert(v1[0] == nk::i8_t(2) && "swap v1 value mismatch");
    assert(v2[0] == nk::i8_t(1) && "swap v2 value mismatch");
}

void test_view_span_rev() {
    auto v = make_vector<float>(5);
    v[0] = 1.0f;
    v[1] = 2.0f;
    v[2] = 3.0f;
    v[3] = 4.0f;
    v[4] = 5.0f;

    auto view = v.view();
    assert(view.size() == 5 && "view size mismatch");
    assert(view[-1] == 5.0f && "view signed indexing failed");

    auto span = v.span();
    span[0] = 10.0f;
    assert(v[0] == 10.0f && "span write-through failed");

    auto rev = view.rev();
    assert(rev[0] == 5.0f && "reversed view first element mismatch");
    assert(rev[4] == 10.0f && "reversed view last element mismatch");
}

void test_range_slicing() {
    auto v = make_vector<float>(5);
    v[0] = 1.0f;
    v[1] = 2.0f;
    v[2] = 3.0f;
    v[3] = 4.0f;
    v[4] = 5.0f;

    auto sub = v[nk::range(1, 4)];
    assert(sub.size() == 3 && "range slice size mismatch");
    assert(sub[0] == 2.0f && "range slice first element mismatch");
    assert(sub[2] == 4.0f && "range slice last element mismatch");
}

void test_sub_byte_i4x2() {
    auto v = make_vector<nk::i4x2_t>(100);
    assert(v.size() == 100 && "i4x2_t size mismatch");
    assert(v.size_values() == 50 && "i4x2_t size_values mismatch (should be dims/2)");

    v[0] = 5, v[1] = -3;
    assert(v[0] == 5 && "i4x2_t dim 0 mismatch");
    assert(v[1] == -3 && "i4x2_t dim 1 mismatch");
}

void test_sub_byte_u1x8() {
    auto v = make_vector<nk::u1x8_t>(64);
    assert(v.size() == 64 && "u1x8_t size mismatch");
    assert(v.size_values() == 8 && "u1x8_t size_values mismatch (should be dims/8)");

    v[0] = true, v[1] = false, v[7] = true;
    assert(v[0] == true && "u1x8_t dim 0 mismatch");
    assert(v[1] == false && "u1x8_t dim 1 mismatch");
    assert(v[7] == true && "u1x8_t dim 7 mismatch");
}

void test_custom_allocator() {
    using custom_alloc_t = nk::aligned_allocator<nk::f32_t, 128>;
    auto v = nk::vector<nk::f32_t, custom_alloc_t>::try_zeros(256);
    assert(v.size() == 256 && "custom allocator size mismatch");
    v[128] = nk::f32_t(99.0f);
    assert(v[128] == nk::f32_t(99.0f) && "custom allocator value mismatch");
}

template <typename value_type_, std::size_t cols_>
void test_sub_byte_tensor_axis_reduction_case(std::array<int, cols_> const &first_row,
                                              std::array<int, cols_> const &second_row,
                                              std::array<int, cols_> const &expected_sums,
                                              std::array<int, cols_> const &expected_mins,
                                              std::array<int, cols_> const &expected_maxs) {
    using tensor_t = nk::tensor<value_type_>;
    using sum_t = typename value_type_::reduce_moments_sum_t;
    using minmax_t = typename value_type_::reduce_minmax_value_t;

    auto t = tensor_t::try_zeros({2, cols_});
    assert(!t.empty() && "tensor allocation failed");

    auto span = t.span();
    auto row0 = span.slice_leading(0).as_vector();
    auto row1 = span.slice_leading(1).as_vector();
    for (std::size_t i = 0; i < cols_; ++i) {
        row0[i] = first_row[i];
        row1[i] = second_row[i];
    }

    auto sums = nk::try_sum<value_type_>(t.view(), 0);
    assert(!sums.empty() && "axis-0 sum failed");
    auto sum_view = sums.as_vector_view();
    for (std::size_t i = 0; i < cols_; ++i) assert(sum_view[i] == sum_t(expected_sums[i]) && "axis-0 sum mismatch");

    auto minmax = nk::try_minmax<value_type_>(t.view(), 0);
    assert(!minmax.min_value.empty() && !minmax.max_value.empty() && "axis-0 minmax failed");
    auto min_view = minmax.min_value.as_vector_view();
    auto max_view = minmax.max_value.as_vector_view();
    for (std::size_t i = 0; i < cols_; ++i) {
        assert(min_view[i] == minmax_t(expected_mins[i]) && "axis-0 min mismatch");
        assert(max_view[i] == minmax_t(expected_maxs[i]) && "axis-0 max mismatch");
    }
}

template <typename tensor_type_, typename expected_type_, std::size_t dims_>
void assert_flat_tensor_equals(tensor_type_ const &tensor, std::array<expected_type_, dims_> const &expected) {
    auto flat = tensor.view().flatten();
    assert(!flat.empty() && "tensor flatten failed");
    auto vec = flat.as_vector();
    assert(vec.size() == dims_ && "flattened tensor size mismatch");
    using actual_t = typename tensor_type_::value_type;
    for (std::size_t i = 0; i < dims_; ++i)
        assert(vec[i] == actual_t(expected[i]) && "flattened tensor value mismatch");
}

template <typename tensor_type_, typename expected_type_>
void assert_scalar_tensor_equals(tensor_type_ const &tensor, expected_type_ expected) {
    auto flat = tensor.view().flatten();
    assert(!flat.empty() && "tensor flatten failed");
    auto vec = flat.as_vector();
    assert(vec.size() == 1 && "scalar tensor should flatten to one value");
    using actual_t = typename tensor_type_::value_type;
    assert(vec[0u] == actual_t(expected) && "scalar tensor value mismatch");
}

template <typename value_type_, std::size_t cols_>
void test_sub_byte_tensor_rank3_axis_case(
    std::array<int, cols_> const &a00, std::array<int, cols_> const &a01, std::array<int, cols_> const &a10,
    std::array<int, cols_> const &a11, std::array<int, cols_ * 2> const &expected_sum_axis0,
    std::array<int, cols_ * 2> const &expected_sum_axis1, std::array<int, 4> const &expected_sum_axis2,
    std::array<int, cols_ * 2> const &expected_min_axis0, std::array<int, cols_ * 2> const &expected_max_axis0,
    std::array<int, cols_ * 2> const &expected_min_axis1, std::array<int, cols_ * 2> const &expected_max_axis1,
    std::array<int, 4> const &expected_min_axis2, std::array<int, 4> const &expected_max_axis2) {
    using tensor_t = nk::tensor<value_type_>;
    using sum_t = typename value_type_::reduce_moments_sum_t;
    using minmax_t = typename value_type_::reduce_minmax_value_t;

    auto t = tensor_t::try_zeros({2, 2, cols_});
    assert(!t.empty() && "rank-3 tensor allocation failed");

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
    assert_flat_tensor_equals(sums0, expected_sum_axis0);
    assert_flat_tensor_equals(sums1, expected_sum_axis1);
    assert_flat_tensor_equals(sums2, expected_sum_axis2);

    auto moments0 = nk::try_moments<value_type_>(t.view(), 0);
    auto moments1 = nk::try_moments<value_type_>(t.view(), 1);
    auto moments2 = nk::try_moments<value_type_>(t.view(), 2);
    assert_flat_tensor_equals(moments0.sum, expected_sum_axis0);
    assert_flat_tensor_equals(moments1.sum, expected_sum_axis1);
    assert_flat_tensor_equals(moments2.sum, expected_sum_axis2);

    auto minmax0 = nk::try_minmax<value_type_>(t.view(), 0);
    auto minmax1 = nk::try_minmax<value_type_>(t.view(), 1);
    auto minmax2 = nk::try_minmax<value_type_>(t.view(), 2);
    assert_flat_tensor_equals(minmax0.min_value, expected_min_axis0);
    assert_flat_tensor_equals(minmax0.max_value, expected_max_axis0);
    assert_flat_tensor_equals(minmax1.min_value, expected_min_axis1);
    assert_flat_tensor_equals(minmax1.max_value, expected_max_axis1);
    assert_flat_tensor_equals(minmax2.min_value, expected_min_axis2);
    assert_flat_tensor_equals(minmax2.max_value, expected_max_axis2);
}

void test_sub_byte_tensor_axis_reductions() {
    test_sub_byte_tensor_axis_reduction_case<nk::i4x2_t, 4>({1, -2, 7, -8}, {-3, 4, -5, 6}, {-2, 2, 2, -2},
                                                            {-3, -2, -5, -8}, {1, 4, 7, 6});
    test_sub_byte_tensor_axis_reduction_case<nk::u4x2_t, 4>({1, 15, 3, 8}, {14, 2, 9, 7}, {15, 17, 12, 15},
                                                            {1, 2, 3, 7}, {14, 15, 9, 8});
    test_sub_byte_tensor_axis_reduction_case<nk::u1x8_t, 8>({1, 0, 1, 1, 0, 0, 1, 0}, {0, 1, 1, 0, 1, 0, 0, 1},
                                                            {1, 1, 2, 1, 1, 0, 1, 1}, {0, 0, 1, 0, 0, 0, 0, 0},
                                                            {1, 1, 1, 1, 1, 0, 1, 1});
}

void test_sub_byte_tensor_rank3_axis_reductions() {
    test_sub_byte_tensor_rank3_axis_case<nk::i4x2_t, 4>(
        {1, -2, 3, -4}, {5, -6, 7, -8}, {-1, 2, -3, 4}, {-5, 6, -7, 7}, {0, 0, 0, 0, 0, 0, 0, -1},
        {6, -8, 10, -12, -6, 8, -10, 11}, {-2, -2, 2, 1}, {-1, -2, -3, -4, -5, -6, -7, -8}, {1, 2, 3, 4, 5, 6, 7, 7},
        {1, -6, 3, -8, -5, 2, -7, 4}, {5, -2, 7, -4, -1, 6, -3, 7}, {-4, -8, -3, -7}, {3, 7, 4, 7});

    test_sub_byte_tensor_rank3_axis_case<nk::u4x2_t, 4>(
        {1, 2, 3, 4}, {5, 6, 7, 8}, {14, 13, 12, 11}, {10, 9, 8, 7}, {15, 15, 15, 15, 15, 15, 15, 15},
        {6, 8, 10, 12, 24, 22, 20, 18}, {10, 26, 50, 34}, {1, 2, 3, 4, 5, 6, 7, 7}, {14, 13, 12, 11, 10, 9, 8, 8},
        {1, 2, 3, 4, 10, 9, 8, 7}, {5, 6, 7, 8, 14, 13, 12, 11}, {1, 5, 11, 7}, {4, 8, 14, 10});

    test_sub_byte_tensor_rank3_axis_case<nk::u1x8_t, 8>(
        {1, 0, 1, 0, 1, 0, 1, 0}, {0, 1, 0, 1, 0, 1, 0, 1}, {1, 1, 0, 0, 1, 1, 0, 0}, {0, 0, 1, 1, 0, 0, 1, 1},
        {2, 1, 1, 0, 2, 1, 1, 0, 0, 1, 1, 2, 0, 1, 1, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        {4, 4, 4, 4}, {1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1},
        {1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0}, {1, 1, 1, 1});
}

void test_rank1_negative_stride_reductions() {
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
    assert(m.sum == sum_t(10.0) && "negative-stride sum mismatch");
    assert(m.sumsq == typename value_t::reduce_moments_sumsq_t(30.0) && "negative-stride sumsq mismatch");
    assert(mm.min_value == minmax_t(1.0f) && "negative-stride min mismatch");
    assert(mm.max_value == minmax_t(4.0f) && "negative-stride max mismatch");
}

void test_rank1_axis_reductions() {
    auto v = nk::tensor<nk::i8_t>::try_zeros({4});
    assert(!v.empty() && "rank-1 tensor allocation failed");
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

    assert(!sums.empty() && !moments.sum.empty() && "rank-1 axis moments failed");
    assert(!mins.empty() && !maxs.empty() && !argmins.empty() && !argmaxs.empty() && "rank-1 axis minmax failed");
    assert(sums.rank() == 0 && moments.sum.rank() == 0 && "collapsed rank-1 reductions should produce rank-0 tensors");
    assert_scalar_tensor_equals(sums, 4);
    assert_scalar_tensor_equals(moments.sum, 4);
    assert_scalar_tensor_equals(mins, -5);
    assert_scalar_tensor_equals(maxs, 7);
    assert_scalar_tensor_equals(argmins, 3);
    assert_scalar_tensor_equals(argmaxs, 2);
}

void test_packed_tensor_fail_closed_views() {
    auto packed = nk::tensor<nk::i4x2_t>::try_zeros({2, 4});
    assert(!packed.empty() && "packed tensor allocation failed");
    assert(packed.view().transpose().empty() && "packed transpose should fail closed");
    assert((!packed(1, nk::slice).empty()) && "packed row slice should remain supported");
    assert((packed(1, 2, nk::slice).empty()) && "packed scalar trailing slice should fail closed");
#if NK_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    assert((!packed[1, nk::slice].empty()) && "packed operator[] row slice should remain supported");
    assert((packed[1, 2, nk::slice].empty()) && "packed operator[] scalar trailing slice should fail closed");
#endif
}

void test_vector_types() {
    std::printf("Testing vector type instantiations...\n");

    // Template-based type coverage
    test_vector_basics<float>();
    test_vector_basics<double>();
    test_vector_basics<nk::f16_t>();
    test_vector_basics<nk::bf16_t>();
    test_vector_basics<nk::i8_t>();
    test_vector_basics<nk::f32c_t>();
    test_vector_basics<std::complex<double>>();
    test_vector_basics<nk::i4x2_t>();
    test_vector_basics<nk::u1x8_t>();
    std::printf("  vector basics (9 types):      OK\n");

    // Feature tests (non-template, using specific types)
    test_signed_indexing();
    std::printf("  signed indexing:              OK\n");

    test_integral_indexing_api();
    std::printf("  integral indexing api:        OK\n");

    test_move_semantics();
    std::printf("  move semantics:               OK\n");

    test_swap();
    std::printf("  swap:                         OK\n");

    test_view_span_rev();
    std::printf("  view/span/rev:                OK\n");

    test_range_slicing();
    std::printf("  range slicing:                OK\n");

    test_sub_byte_i4x2();
    test_sub_byte_u1x8();
    std::printf("  sub-byte i4x2/u1x8:           OK\n");

    test_custom_allocator();
    std::printf("  custom allocator:             OK\n");

#if NK_TEST_FORMAT_
    test_format_scalars();
    std::printf("  std::format scalars+refs:     OK\n");
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
void test_tensor_ops_for_type() {
    using tensor_t = nk::tensor<value_type_>;

    // Create small test tensors
    auto a = tensor_t::try_zeros({4, 8});
    auto b = tensor_t::try_zeros({4, 8});
    assert(!a.empty() && !b.empty());

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
    { [[maybe_unused]] auto r = nk::try_sum<value_type_>(av, 0); }
    { [[maybe_unused]] auto r = nk::try_sum<value_type_>(av, 1, nk::keep_dims_k); }
    { [[maybe_unused]] auto r = nk::try_moments<value_type_>(av, 1); }
    { [[maybe_unused]] auto r = nk::try_minmax<value_type_>(av, 0); }
    { [[maybe_unused]] auto r = nk::try_minmax<value_type_>(av, 1, nk::keep_dims_k); }
    { [[maybe_unused]] auto r = nk::try_min<value_type_>(av, 0); }
    { [[maybe_unused]] auto r = nk::try_min<value_type_>(av, 1, nk::keep_dims_k); }
    { [[maybe_unused]] auto r = nk::try_max<value_type_>(av, 0); }
    { [[maybe_unused]] auto r = nk::try_max<value_type_>(av, 1, nk::keep_dims_k); }
    { [[maybe_unused]] auto r = nk::try_argmin<value_type_>(av, 0); }
    { [[maybe_unused]] auto r = nk::try_argmax<value_type_>(av, 1, nk::keep_dims_k); }

    // Elementwise binary
    { [[maybe_unused]] auto r = nk::try_add<value_type_>(av, bv); }
    { [[maybe_unused]] auto r = nk::try_sub<value_type_>(av, bv); }
    { [[maybe_unused]] auto r = nk::try_mul<value_type_>(av, bv); }

    // Elementwise binary with scalar
    using scale_t = typename value_type_::scale_t;
    scale_t scalar {1};
    { [[maybe_unused]] auto r = nk::try_add<value_type_>(av, scalar); }
    { [[maybe_unused]] auto r = nk::try_sub<value_type_>(av, scalar); }
    { [[maybe_unused]] auto r = nk::try_mul<value_type_>(av, scalar); }

    // Elementwise into
    auto out = tensor_t::try_zeros({4, 8});
    { [[maybe_unused]] bool ok = nk::add<value_type_>(av, bv, out.span()); }
    { [[maybe_unused]] bool ok = nk::sub<value_type_>(av, bv, out.span()); }
    { [[maybe_unused]] bool ok = nk::mul<value_type_>(av, bv, out.span()); }
    { [[maybe_unused]] bool ok = nk::add<value_type_>(av, scalar, out.span()); }
    { [[maybe_unused]] bool ok = nk::sub<value_type_>(av, scalar, out.span()); }
    { [[maybe_unused]] bool ok = nk::mul<value_type_>(av, scalar, out.span()); }

    // Affine
    scale_t alpha {1}, beta {0};
    { [[maybe_unused]] auto r = nk::try_scale<value_type_>(av, alpha, beta); }
    { [[maybe_unused]] auto r = nk::try_blend<value_type_>(av, bv, alpha, beta); }
    { [[maybe_unused]] auto r = nk::try_fma<value_type_>(av, bv, av, alpha, beta); }
    { [[maybe_unused]] bool ok = nk::scale<value_type_>(av, alpha, beta, out.span()); }
    { [[maybe_unused]] bool ok = nk::blend<value_type_>(av, bv, alpha, beta, out.span()); }
    { [[maybe_unused]] bool ok = nk::fma<value_type_>(av, bv, av, alpha, beta, out.span()); }

    // try_from 1D
    {
        auto from1d = tensor_t::try_from({value_type_ {}, value_type_ {}, value_type_ {}});
        assert(!from1d.empty() && "try_from 1D failed");
        assert(from1d.rank() == 1 && from1d.numel() == 3 && "try_from 1D shape mismatch");
    }

    // try_from 2D
    {
        auto from2d = tensor_t::try_from({{value_type_ {}, value_type_ {}}, {value_type_ {}, value_type_ {}}});
        assert(!from2d.empty() && "try_from 2D failed");
        assert(from2d.rank() == 2 && from2d.extent(0) == 2 && from2d.extent(1) == 2 && "try_from 2D shape mismatch");
    }

    // row() access
    {
        auto row0 = a.row(0);
        assert(row0.rank() == 1 && row0.extent(0) == 8 && "row() shape mismatch");
    }

    // Convenience view constructor (ptr, rows, cols)
    {
        nk::tensor_view<value_type_> view_from_ptr(a.data(), 4, 8);
        assert(view_from_ptr.rank() == 2 && view_from_ptr.extent(0) == 4 && "convenience view ctor mismatch");
    }
}

template <typename value_type_>
void test_tensor_trig_for_type() {
    using tensor_t = nk::tensor<value_type_>;
    auto a = tensor_t::try_zeros({4, 8});
    auto out = tensor_t::try_zeros({4, 8});
    auto av = a.view();

    { [[maybe_unused]] auto r = nk::try_sin<value_type_>(av); }
    { [[maybe_unused]] auto r = nk::try_cos<value_type_>(av); }
    { [[maybe_unused]] auto r = nk::try_atan<value_type_>(av); }
    { [[maybe_unused]] bool ok = nk::sin<value_type_>(av, out.span()); }
    { [[maybe_unused]] bool ok = nk::cos<value_type_>(av, out.span()); }
    { [[maybe_unused]] bool ok = nk::atan<value_type_>(av, out.span()); }
}

template <typename value_type_>
void test_tensor_symmetric_for_type() {
    using tensor_t = nk::tensor<value_type_>;
    auto a = tensor_t::try_zeros({4, 8});
    auto am = a.as_matrix_view();

    { [[maybe_unused]] auto r = nk::try_dots_symmetric<value_type_>(am); }
    { [[maybe_unused]] auto r = nk::try_angulars_symmetric<value_type_>(am); }
    { [[maybe_unused]] auto r = nk::try_euclideans_symmetric<value_type_>(am); }
}

template <typename value_type_>
void test_tensor_packed_for_type() {
    using tensor_t = nk::tensor<value_type_>;
    auto a = tensor_t::try_zeros({4, 8});
    auto b = tensor_t::try_zeros({6, 8});

    // packed_matrix
    auto bm = b.as_matrix_view();
    auto packed = nk::packed_matrix<value_type_, nk::aligned_allocator<char>>::try_pack(bm);
    auto am = a.as_matrix_view();
    auto result = nk::matrix<typename value_type_::dot_result_t>::try_zeros({4, 6});
    nk::dots_packed<value_type_>(am, packed, result.span());
}

template <typename value_type_>
void test_tensor_maxsim_for_type() {
    using tensor_t = nk::tensor<value_type_>;
    auto q = tensor_t::try_zeros({3, 16});
    auto d = tensor_t::try_zeros({5, 16});

    auto qm = q.as_matrix_view();
    auto dm = d.as_matrix_view();

    auto pq = nk::packed_maxsim<value_type_>::try_pack(qm);
    auto pd = nk::packed_maxsim<value_type_>::try_pack(dm);
    { [[maybe_unused]] auto r = nk::maxsim(pq, pd); }
}

void test_view_overloads() {
    nk::f32_t a_data[8] {}, b_data[8] {}, c_data[64] {};
    nk::f32_t result {};
    auto a_view = nk::vector_view<nk::f32_t>(a_data, 8u);
    auto b_view = nk::vector_view<nk::f32_t>(b_data, 8u);

    nk::dot(a_view, b_view, 8, &result);
    nk::euclidean(a_view, b_view, 8, &result);
    nk::sqeuclidean(a_view, b_view, 8, &result);
    nk::angular(a_view, b_view, 8, &result);

    auto c_view = nk::vector_view<nk::f32_t>(c_data, 64u);
    nk::bilinear(a_view, b_view, c_view, 8, &result);
    nk::mahalanobis(a_view, b_view, c_view, 8, &result);
}

void test_custom_allocator_try_fns() {
    using custom_alloc_t = nk::aligned_allocator<nk::f32_t, 128>;
    auto a = nk::tensor<nk::f32_t>::try_zeros({4, 8});
    auto av = a.view();

    { auto r = nk::try_scale<nk::f32_t, 8, custom_alloc_t>(av, 1.0, 0.0); }
    { auto r = nk::try_sin<nk::f32_t, 8, custom_alloc_t>(av); }

    using sum_alloc_t = nk::aligned_allocator<nk::f64_t, 128>;
    { auto r = nk::try_sum<nk::f32_t, 8, sum_alloc_t>(av, 0); }
}

template <typename value_type_>
void test_vector_reductions_for_type() {
    auto v = make_vector<value_type_>(32);
    std::mt19937 generator(42);
    fill_random(generator, v);
    auto view = nk::vector_view<value_type_>(v.values_data(), static_cast<std::size_t>(v.size()));

    { [[maybe_unused]] auto r = nk::moments(view); }
    { [[maybe_unused]] auto r = nk::minmax(view); }
    { [[maybe_unused]] auto r = nk::sum(view); }
    { [[maybe_unused]] auto r = nk::min(view); }
    { [[maybe_unused]] auto r = nk::max(view); }
    { [[maybe_unused]] auto r = nk::argmin(view); }
    { [[maybe_unused]] auto r = nk::argmax(view); }
}

void test_vector_reductions_correctness() {
    nk::f32_t data[] = {nk::f32_t(1), nk::f32_t(2), nk::f32_t(3), nk::f32_t(4), nk::f32_t(5)};
    auto view = nk::vector_view<nk::f32_t>(data, std::size_t {5});

    auto m = nk::moments(view);
    assert(m.sum.raw_ == 15.0 && "vector moments sum");
    assert(m.sumsq.raw_ == 55.0 && "vector moments sumsq");

    auto mm = nk::minmax(view);
    assert(mm.min_value.raw_ == 1.0f && "vector min");
    assert(mm.max_value.raw_ == 5.0f && "vector max");
    assert(mm.min_index == 0 && "vector argmin");
    assert(mm.max_index == 4 && "vector argmax");
}

template <typename from_type_, typename to_type_>
void test_cast_for_types() {
    auto src = make_vector<from_type_>(64);
    auto dst = make_vector<to_type_>(64);
    std::mt19937 generator(42);
    fill_random(generator, src);

    auto src_view = nk::vector_view<from_type_>(src.values_data(), static_cast<std::size_t>(src.size()));
    auto dst_span = nk::vector_span<to_type_>(dst.values_data(), static_cast<std::size_t>(dst.size()));

    // Pointer-level API
    nk::cast<from_type_, to_type_>(src.values_data(), src.size(), dst.values_data());
    // Vector view/span API
    nk::cast<from_type_, to_type_>(src_view, dst_span);
}

#if NK_TEST_FORMAT_
void test_format_scalars() {
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
    assert(std::format("{}", iref) == "3" && "i4 sub_byte_ref default format");
    assert(std::format("{:x}", iref) == "3" && "i4 sub_byte_ref hex format");
    assert(std::format("{:b}", iref) == "0011" && "i4 sub_byte_ref binary format");
    assert(std::format("{:#}", iref) == "3 [0x3]" && "i4 sub_byte_ref annotated format");

    nk_u4x2_t packed_u = 0xA7;
    nk::sub_byte_ref<nk::u4x2_t> uref(&packed_u, 1);
    assert(std::format("{}", uref) == "10" && "u4 sub_byte_ref default format");
    assert(std::format("{:x}", uref) == "a" && "u4 sub_byte_ref hex format");
    assert(std::format("{:b}", uref) == "1010" && "u4 sub_byte_ref binary format");

    nk_u1x8_t packed_b = 0x05;
    nk::sub_byte_ref<nk::u1x8_t> bref(&packed_b, 0);
    assert(std::format("{}", bref) == "1" && "u1 sub_byte_ref format");
}
#endif // NK_TEST_FORMAT_

void test_tensor_ops() {
    std::printf("Testing tensor op instantiations...\n");

    // Core numeric types: all operations
    test_tensor_ops_for_type<nk::f32_t>();
    test_tensor_ops_for_type<nk::f64_t>();
    test_tensor_ops_for_type<nk::f16_t>();
    test_tensor_ops_for_type<nk::bf16_t>();
    test_tensor_ops_for_type<nk::i8_t>();
    test_tensor_ops_for_type<nk::u8_t>();
    std::printf("  ops (6 types):                OK\n");

    test_sub_byte_tensor_axis_reductions();
    std::printf("  tensor axis sub-byte:         OK\n");

    test_sub_byte_tensor_rank3_axis_reductions();
    std::printf("  tensor axis rank-3 packed:    OK\n");

    test_rank1_negative_stride_reductions();
    std::printf("  tensor negative stride:       OK\n");

    test_rank1_axis_reductions();
    std::printf("  tensor rank-1 axis:           OK\n");

    test_tensor_operator_indexing();
    std::printf("  tensor operator[]:            OK\n");

    test_packed_tensor_operator_indexing();
    std::printf("  tensor packed operator[]:     OK\n");

    test_packed_tensor_fail_closed_views();
    std::printf("  tensor fail-closed views:     OK\n");

    // Trig (float-capable types)
    test_tensor_trig_for_type<nk::f32_t>();
    test_tensor_trig_for_type<nk::f64_t>();
    test_tensor_trig_for_type<nk::f16_t>();
    test_tensor_trig_for_type<nk::bf16_t>();
    std::printf("  trig (4 types):               OK\n");

    // Symmetric distances
    test_tensor_symmetric_for_type<nk::f32_t>();
    test_tensor_symmetric_for_type<nk::f64_t>();
    test_tensor_symmetric_for_type<nk::f16_t>();
    test_tensor_symmetric_for_type<nk::bf16_t>();
    test_tensor_symmetric_for_type<nk::i8_t>();
    std::printf("  symmetric dist (5 types):     OK\n");

    // Packed GEMM
    test_tensor_packed_for_type<nk::f32_t>();
    test_tensor_packed_for_type<nk::f64_t>();
    test_tensor_packed_for_type<nk::f16_t>();
    test_tensor_packed_for_type<nk::bf16_t>();
    test_tensor_packed_for_type<nk::i8_t>();
    std::printf("  packed GEMM (5 types):        OK\n");

    // MaxSim (bf16, f32, f16 only)
    test_tensor_maxsim_for_type<nk::bf16_t>();
    test_tensor_maxsim_for_type<nk::f32_t>();
    test_tensor_maxsim_for_type<nk::f16_t>();
    std::printf("  packed MaxSim (3 types):      OK\n");

    test_view_overloads();
    std::printf("  view overloads:               OK\n");

    test_custom_allocator_try_fns();
    std::printf("  custom allocator try_fns:     OK\n");

    // Vector-level reductions
    test_vector_reductions_for_type<nk::f32_t>();
    test_vector_reductions_for_type<nk::f64_t>();
    test_vector_reductions_for_type<nk::f16_t>();
    test_vector_reductions_for_type<nk::bf16_t>();
    test_vector_reductions_for_type<nk::i8_t>();
    test_vector_reductions_for_type<nk::u8_t>();
    std::printf("  vector reductions (6 types):  OK\n");

    test_vector_reductions_correctness();
    std::printf("  vector reductions correct:    OK\n");

    // Cast wrapper
    test_cast_for_types<nk::f32_t, nk::f16_t>();
    test_cast_for_types<nk::f16_t, nk::f32_t>();
    test_cast_for_types<nk::f32_t, nk::bf16_t>();
    test_cast_for_types<nk::bf16_t, nk::f32_t>();
    test_cast_for_types<nk::f32_t, nk::e4m3_t>();
    test_cast_for_types<nk::e4m3_t, nk::f32_t>();
    test_cast_for_types<nk::i8_t, nk::i32_t>();
    test_cast_for_types<nk::f64_t, nk::f32_t>();
    std::printf("  cast wrapper (8 pairs):       OK\n");
}
