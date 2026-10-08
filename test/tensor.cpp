/**
 *  @file test/tensor.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *  @brief C++ vector type instantiation tests.
 */
#include <array>
#include <cassert>
#include <complex>
#include <limits>
#include <span>
#include <vector>

#include "harness.hpp"

#include "numkong/attention.hpp"
#include "numkong/cast.hpp"
#include "numkong/dot.hpp"
#include "numkong/spatial.hpp"
#include "numkong/curved.hpp"
#include "numkong/reduce.hpp"
#include "numkong/set.hpp"
#include "numkong/trigonometry.hpp"

#if __has_include(<format>)
#include <format>
#if defined(__cpp_lib_format) && __cpp_lib_format >= 202110L
#define NUMKONG_TEST_FORMAT_ 1
#endif
#endif
#ifndef NUMKONG_TEST_FORMAT_
#define NUMKONG_TEST_FORMAT_ 0
#endif

/*  Explicit instantiations for tensor types, forcing full compilation of all APIs. */
template struct nk::tensor<nk::f32_t>;
template struct nk::tensor<nk::f64_t>;
template struct nk::tensor<nk::f16_t>;
template struct nk::tensor<nk::bf16_t>;
template struct nk::tensor<nk::i8_t>;

/*  Views and spans for rank-2 matrices and the default rank. */
template struct nk::tensor_view<nk::f32_t, 2>;
template struct nk::tensor_view<nk::f32_t, 8>;
template struct nk::tensor_span<nk::f32_t, 2>;
template struct nk::tensor_span<nk::f32_t, 8>;
template struct nk::tensor_view<nk::bf16_t, 2>;
template struct nk::tensor_span<nk::bf16_t, 2>;

namespace ashvardanian::numkong::test {

#if NUMKONG_TEST_FORMAT_
error_stats_t test_format_scalars(settings_t const &);
#endif

template <typename value_type_>
error_stats_t test_vector_basics(settings_t const &) {
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

error_stats_t test_signed_indexing(settings_t const &) {
    error_stats_t stats(comparison_family_t::exact_k);
    auto v = make_vector<float>(100);
    v[50] = 3.14f;
    stats.expect(v[50] == 3.14f, "float operator[] failed");
    v[-1] = 42.0f;
    stats.expect(v[99] == 42.0f, "float signed indexing failed");
    return stats;
}

error_stats_t test_integral_indexing_api(settings_t const &) {
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

error_stats_t test_tensor_operator_indexing(settings_t const &) {
    error_stats_t stats(comparison_family_t::exact_k);
    auto [t, t_status] = nk::tensor<float>::zeros({2, 3});
    stats.expect(nk::succeeded(t_status), "tensor allocation failed");

    for (int i = 0; i < 6; ++i) t[i] = static_cast<float>(i + 1);

    stats.expect(t[0] == 1.0f, "flat tensor lookup failed");
    stats.expect(t[-1] == 6.0f, "negative flat tensor lookup failed");
    stats.expect((t(0, 0) == 1.0f), "exact tensor lookup failed");
    stats.expect((t(1, -1) == 6.0f), "negative exact tensor lookup failed");
#if NUMKONG_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
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
#if NUMKONG_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    stats.expect((t[1, 1] == 42.0f), "operator[] row slice write-through failed");
    auto row1_subscript = t[1, nk::slice];
    stats.expect(row1_subscript.extent(0) == row1.extent(0), "operator[] row slice mismatch");
#endif

    auto cell = t(1, 1, nk::slice);
    stats.expect(cell.rank() == 0, "scalar slice rank mismatch");
    stats.expect(cell.scalar() == 42.0f, "scalar slice value mismatch");
    cell.scalar_ref() = 24.0f;
    stats.expect((t(1, 1) == 24.0f), "scalar slice write-through failed");
#if NUMKONG_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    stats.expect((t[1, 1] == 24.0f), "operator[] scalar slice write-through failed");
    auto cell_subscript = t[1, 1, nk::slice];
    stats.expect(cell_subscript.rank() == 0, "operator[] scalar slice rank mismatch");
#endif

    auto const &ct = t;
    auto const last_row = ct(-1, nk::slice);
    stats.expect(last_row.rank() == 1, "const row slice rank mismatch");
    stats.expect(last_row[0] == 4.0f, "const row slice mismatch");
#if NUMKONG_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    auto const last_row_subscript = ct[-1, nk::slice];
    stats.expect(last_row_subscript.rank() == 1, "operator[] const row slice mismatch");
#endif

    auto [cube, cube_status] = nk::tensor<float>::zeros({2, 3, 4});
    stats.expect(nk::succeeded(cube_status), "cube allocation failed");
    for (int i = 0; i < 24; ++i) cube[i] = static_cast<float>(i);

    auto plane = cube(1, nk::slice);
    stats.expect(plane.rank() == 2 && plane.extent(0) == 3 && plane.extent(1) == 4, "plane slice mismatch");
    auto line = cube(1, 2, nk::slice);
    stats.expect(line.rank() == 1 && line.extent(0) == 4, "line slice mismatch");
    stats.expect((line[3] == cube(1, 2, 3)), "line slice element mismatch");
    auto point = cube(1, 2, 3, nk::slice);
    stats.expect((point.rank() == 0 && point.scalar() == cube(1, 2, 3)), "point slice mismatch");
#if NUMKONG_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
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
#if NUMKONG_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
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
#if NUMKONG_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    auto row0_via_subscript = t[0, nk::slice];
    stats.expect(row0[0] == row0_via_subscript[0], "row() should match t[0, slice]");
#endif
    return stats;
}

error_stats_t test_packed_tensor_operator_indexing(settings_t const &) {
    error_stats_t stats(comparison_family_t::exact_k);
    auto [t4, t4_status] = nk::tensor<nk::u4x2_t>::zeros({2, 4});
    stats.expect(nk::succeeded(t4_status), "packed u4 tensor allocation failed");

    for (int i = 0; i < 8; ++i) t4[i] = i + 1;

    stats.expect(int(t4[0]) == 1, "packed flat lookup failed");
    stats.expect(int(t4[-1]) == 8, "packed negative flat lookup failed");
    stats.expect((int(t4(0, 3)) == 4), "packed exact lookup failed");
    stats.expect((int(t4(1, -1)) == 8), "packed negative exact lookup failed");
#if NUMKONG_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    stats.expect((int(t4[0, 3]) == 4), "packed operator[] exact lookup failed");
    stats.expect((int(t4[1, -1]) == 8), "packed operator[] negative exact lookup failed");
#endif

    auto second_row = t4(1, nk::slice);
    stats.expect(second_row.rank() == 1 && second_row.extent(0) == 4, "packed row slice rank mismatch");
    stats.expect(int(second_row[0]) == 5 && int(second_row[-1]) == 8, "packed row slice values mismatch");
    second_row[1] = 14;
    stats.expect((int(t4(1, 1)) == 14), "packed row slice write-through failed");
#if NUMKONG_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    auto second_row_subscript = t4[1, nk::slice];
    stats.expect(second_row_subscript.extent(0) == 4, "packed operator[] row slice rank mismatch");
    stats.expect((int(t4[1, 1]) == 14), "packed operator[] row slice write-through failed");
#endif

    auto [t1, t1_status] = nk::tensor<nk::u1x8_t>::zeros({2, 8});
    stats.expect(nk::succeeded(t1_status), "packed u1 tensor allocation failed");
    t1[0] = true;
    t1[7] = true;
    t1[11] = true;
    t1[-1] = true;

    stats.expect(bool(t1[0]), "packed bit flat lookup failed");
    stats.expect((bool(t1(0, 7))), "packed bit exact lookup failed");
    stats.expect((bool(t1(1, 3))), "packed bit second-row lookup failed");
    stats.expect(bool(t1[-1]), "packed bit negative flat lookup failed");
#if NUMKONG_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    stats.expect((bool(t1[0, 7])), "packed bit operator[] exact lookup failed");
    stats.expect((bool(t1[1, 3])), "packed bit operator[] second-row lookup failed");
#endif

    auto bits = t1(1, nk::slice);
    stats.expect(bits.rank() == 1 && bits.extent(0) == 8, "packed bit slice rank mismatch");
    bits[4] = true;
    stats.expect((bool(t1(1, 4))), "packed bit slice write-through failed");
#if NUMKONG_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    auto bits_subscript = t1[1, nk::slice];
    stats.expect(bits_subscript.extent(0) == 8, "packed bit operator[] slice rank mismatch");
    stats.expect((bool(t1[1, 4])), "packed bit operator[] slice write-through failed");
#endif
    return stats;
}

error_stats_t test_move_semantics(settings_t const &) {
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

error_stats_t test_swap(settings_t const &) {
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

error_stats_t test_view_span_rev(settings_t const &) {
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

error_stats_t test_range_slicing(settings_t const &) {
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

error_stats_t test_sub_byte_i4x2(settings_t const &) {
    error_stats_t stats(comparison_family_t::exact_k);
    auto v = make_vector<nk::i4x2_t>(100);
    stats.expect(v.size() == 100, "i4x2_t size mismatch");
    stats.expect(v.size_values() == 50, "i4x2_t size_values mismatch (should be dims/2)");

    v[0] = 5, v[1] = -3;
    stats.expect(v[0] == 5, "i4x2_t dim 0 mismatch");
    stats.expect(v[1] == -3, "i4x2_t dim 1 mismatch");

    stats.expect(nk::vector<nk::i4x2_t>::zeros(7).status == nk::status_t::unexpected_dimensions_k,
                 "i4x2_t allocated half a byte");
    stats.expect(v.resize(99) == nk::status_t::unexpected_dimensions_k && v.size() == 100,
                 "i4x2_t resized to half a byte");
    return stats;
}

error_stats_t test_sub_byte_u1x8(settings_t const &) {
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

error_stats_t test_block_scaled_composites(settings_t const &) {
    error_stats_t stats(comparison_family_t::exact_k);
    // NVFP4: 9 bytes per block × 7 blocks for 112 logical dims.
    auto nvfp4_vec = make_vector<nk::nvfp4_t>(112);
    stats.expect(nvfp4_vec.size() == 112, "nvfp4_t size mismatch");
    stats.expect(nvfp4_vec.size_values() == 7, "nvfp4_t size_values mismatch (112/16)");
    stats.expect(nvfp4_vec.size_bytes() == 63, "nvfp4_t size_bytes mismatch (7 × 9)");

    // MXFP4: 17 bytes per block × 4 blocks for 128 logical dims.
    auto mxfp4_vec = make_vector<nk::mxfp4_t>(128);
    stats.expect(mxfp4_vec.size() == 128, "mxfp4_t size mismatch");
    stats.expect(mxfp4_vec.size_values() == 4, "mxfp4_t size_values mismatch (128/32)");
    stats.expect(mxfp4_vec.size_bytes() == 68, "mxfp4_t size_bytes mismatch (4 × 17)");

    // MXFP8 E4M3: 33 bytes per block × 4 blocks for 128 logical dims.
    auto mxfp8e4m3_vec = make_vector<nk::mxfp8e4m3_t>(128);
    stats.expect(mxfp8e4m3_vec.size() == 128, "mxfp8e4m3_t size mismatch");
    stats.expect(mxfp8e4m3_vec.size_values() == 4, "mxfp8e4m3_t size_values mismatch");
    stats.expect(mxfp8e4m3_vec.size_bytes() == 132, "mxfp8e4m3_t size_bytes mismatch (4 × 33)");

    // Round-trip: encode 16 random f32s into an NVFP4 block, decode back, check bounded error.
    float const src[16] = {-5.3f, 2.1f,  0.5f, -0.1f, 3.7f,  -4.2f, 1.0f, 0.0f,
                           6.0f,  -6.0f, 2.5f, 1.25f, -3.5f, 0.75f, 4.0f, -1.5f};
    nk::nvfp4_t block = nk::nvfp4_t::encode_from(src, /*global=*/1.0f);
    float decoded[16];
    block.decode_to(decoded, /*global=*/1.0f);
    float max_error = 0.0f;
    for (unsigned i = 0; i < 16; ++i) {
        float err = decoded[i] - src[i];
        if (err < 0) err = -err;
        if (err > max_error) max_error = err;
    }
    stats.expect(max_error <= 1.67f, "nvfp4_t round-trip error exceeds quantisation bound");
    return stats;
}

/** Detects whether a @c scaled_tensor lookalike has the per-tensor `tensor_scale()` accessor. */
template <typename scaled_type_>
concept exposes_tensor_scale_ = requires(scaled_type_ const &t) { t.tensor_scale(); };

/**
 *  @brief End-to-end test of the @c scaled_tensor family: encode via @c cast, inspect the SoA
 *      components, slice rows / block-aligned column tiles, materialize back to dense, and verify
 *      the per-tensor scale is exposed for NVFP4 but compile-time absent for the MX family.
 *
 *  Every numeric path is checked byte-for-byte against @c nk_cast_serial.
 */
error_stats_t test_scaled_tensor(settings_t const &) {
    error_stats_t stats(comparison_family_t::exact_k);
    using nk::f32_t;
    using nk::u8_t;

    // 4 rows × 64 columns. 64 is a multiple of both the NVFP4 (16) and MX (32) block sizes.
    constexpr std::size_t rows = 4, columns = 64;
    auto [weights, weights_status] = nk::tensor<f32_t>::uninitialized({rows, columns});
    stats.expect(nk::succeeded(weights_status), "weights allocation failed");
    {
        auto writable = weights.span();
        for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t c = 0; c < columns; ++c)
                writable(r, c) = f32_t(static_cast<float>(static_cast<int>((r * 7 + c * 3) % 17) - 8) * 0.6f);
    }

    // Encode (quantize) to NVFP4 into a preallocated scaled_tensor.
    auto [quantized, quantized_status] = nk::scaled_tensor<nk::nvfp4_t>::uninitialized({rows, columns});
    stats.expect(nk::succeeded(quantized_status), "quantized allocation failed");
    stats.expect(nk::cast(weights.view(), quantized.span()));
    stats.expect(!quantized.empty(), "encode produced an empty scaled_tensor");
    stats.expect(quantized.rank() == 2 && quantized.extent(0) == rows && quantized.extent(1) == columns,
                 "encoded shape mismatch");
    // elements() keeps the logical shape; block_scales() divides the last axis by block_size (16).
    stats.expect(quantized.elements().extent(0) == rows && quantized.elements().extent(1) == columns,
                 "elements() shape mismatch");
    stats.expect(quantized.block_scales().extent(0) == rows && quantized.block_scales().extent(1) == columns / 16,
                 "block_scales() shape mismatch");

    // Byte-identical to the serial C reference.
    nk_block_scaled_format_t const nvfp4_format = nk_nvfp4();
    auto reference_elements = make_vector<u8_t>(nk_block_scaled_elements_size(rows * columns, nvfp4_format));
    auto reference_scales = make_vector<u8_t>(nk_block_scaled_scales_size(rows * columns, nvfp4_format));
    nk_f32_t reference_tensor_scale = 0; // zero → derive, matching the C++ factory
    nk_nvfp4_ref_t reference_encoded = {reinterpret_cast<nk_e2m1x2_t *>(reference_elements.raw_values_data()),
                                        reinterpret_cast<nk_ue4m3_t *>(reference_scales.raw_values_data()),
                                        &reference_tensor_scale};
    stats.expect(nk_cast_serial(weights.data(), nk_f32_k, &reference_encoded, nk_nvfp4_k, rows * columns, nullptr));

    auto const *encoded_elements = reinterpret_cast<unsigned char const *>(quantized.elements().byte_data());
    for (std::size_t i = 0; i < reference_elements.size_values(); ++i)
        stats.expect(encoded_elements[i] == reference_elements.raw_values_data()[i],
                     "NVFP4 elements differ from reference");
    auto const *encoded_scales = reinterpret_cast<unsigned char const *>(quantized.block_scales().byte_data());
    for (std::size_t i = 0; i < reference_scales.size_values(); ++i)
        stats.expect(encoded_scales[i] == reference_scales.raw_values_data()[i], "NVFP4 scales differ from reference");
    stats.expect(quantized.tensor_scale().raw_ == reference_tensor_scale,
                 "derived tensor_scale differs from reference");

    std::size_t const row_element_bytes = nk_block_scaled_elements_size(columns, nvfp4_format); // 32
    std::size_t const row_scale_bytes = nk_block_scaled_scales_size(columns, nvfp4_format);     // 4

    // Slice one row and materialize it to a dense f32 vector.
    auto restored_row = make_vector<f32_t>(columns);
    stats.expect(nk::cast<nk::nvfp4_t>(quantized.row(1), restored_row.span()));
    {
        auto reference_row = make_vector<f32_t>(columns);
        nk_f32_t tensor_scale = quantized.tensor_scale().raw_;
        nk_nvfp4_cref_t const row_reference = {
            reinterpret_cast<nk_e2m1x2_t const *>(reference_elements.raw_values_data() + 1 * row_element_bytes),
            reinterpret_cast<nk_ue4m3_t const *>(reference_scales.raw_values_data() + 1 * row_scale_bytes),
            &tensor_scale};
        stats.expect(
            nk_cast_serial(&row_reference, nk_nvfp4_k, reference_row.raw_values_data(), nk_f32_k, columns, nullptr));
        for (std::size_t c = 0; c < columns; ++c)
            stats.expect(restored_row.raw_values_data()[c] == reference_row.raw_values_data()[c],
                         "row materialization differs from reference");
    }

    // Block-aligned column tile of the first two NVFP4 blocks, then materialize it.
    auto column_tile = quantized.columns(0, 32);
    stats.expect(column_tile.extent(0) == rows && column_tile.extent(1) == 32, "column tile shape mismatch");
    stats.expect(column_tile.block_scales().extent(1) == 32 / 16, "column tile block_scales extent mismatch");
    // A sub-block (non-aligned) range is rejected, not silently truncated.
    stats.expect(quantized.columns(0, 24).empty(), "sub-block column ranges must be rejected");
    auto [restored_tile, restored_tile_status] = nk::tensor<f32_t>::uninitialized({rows, std::size_t {32}});
    stats.expect(nk::succeeded(restored_tile_status), "restored tile allocation failed");
    stats.expect(nk::cast<nk::nvfp4_t>(column_tile, restored_tile.span()));
    {
        auto const *tile_raw = reinterpret_cast<float const *>(restored_tile.data());
        auto reference_tile_row = make_vector<f32_t>(32);
        for (std::size_t r = 0; r < rows; ++r) {
            nk_f32_t tensor_scale = quantized.tensor_scale().raw_;
            nk_nvfp4_cref_t const row_reference = {
                reinterpret_cast<nk_e2m1x2_t const *>(reference_elements.raw_values_data() + r * row_element_bytes),
                reinterpret_cast<nk_ue4m3_t const *>(reference_scales.raw_values_data() + r * row_scale_bytes),
                &tensor_scale};
            stats.expect(nk_cast_serial(&row_reference, nk_nvfp4_k, reference_tile_row.raw_values_data(), nk_f32_k, 32,
                                        nullptr));
            for (std::size_t c = 0; c < 32; ++c)
                stats.expect(tile_raw[r * 32 + c] == reference_tile_row.raw_values_data()[c],
                             "column-tile materialization differs from reference");
        }
    }

    // Iterate leading-axis rows.
    std::size_t iterated_rows = 0;
    for (nk::scaled_tensor_view<nk::nvfp4_t> row_view : quantized.rows_views()) {
        stats.expect(row_view.rank() == 1 && row_view.extent(0) == columns, "row view shape mismatch");
        stats.expect(row_view.block_scales().extent(0) == columns / 16, "row view block_scales extent mismatch");
        ++iterated_rows;
    }
    stats.expect(iterated_rows == rows, "rows_views() did not visit every row");

    // The per-tensor scale exists for NVFP4 and is compile-time absent for the MX family.
    static_assert(exposes_tensor_scale_<nk::scaled_tensor<nk::nvfp4_t>>, "NVFP4 must expose tensor_scale()");
    static_assert(!exposes_tensor_scale_<nk::scaled_tensor<nk::mxfp8e4m3_t>>,
                  "MX formats must not expose tensor_scale()");

    // Block-aligned column tile at a non-zero start exercises the element and scale byte offsets.
    {
        auto mid_tile = quantized.columns(16, 48); // two NVFP4 blocks starting at column 16
        stats.expect(mid_tile.extent(1) == 32 && mid_tile.block_scales().extent(1) == 32 / 16,
                     "mid tile shape mismatch");
        auto [restored_mid, restored_mid_status] = nk::tensor<f32_t>::uninitialized({rows, std::size_t {32}});
        stats.expect(nk::succeeded(restored_mid_status), "restored mid allocation failed");
        stats.expect(nk::cast<nk::nvfp4_t>(mid_tile, restored_mid.span()));
        auto const *mid_raw = reinterpret_cast<float const *>(restored_mid.data());
        auto reference_mid = make_vector<f32_t>(32);
        for (std::size_t r = 0; r < rows; ++r) {
            nk_f32_t tensor_scale = quantized.tensor_scale().raw_;
            nk_nvfp4_cref_t const row_reference = {
                reinterpret_cast<nk_e2m1x2_t const *>(reference_elements.raw_values_data() + r * row_element_bytes +
                                                      16 / 2), // column 16 → byte 8
                reinterpret_cast<nk_ue4m3_t const *>(reference_scales.raw_values_data() + r * row_scale_bytes +
                                                     16 / 16), // block 1
                &tensor_scale};
            stats.expect(
                nk_cast_serial(&row_reference, nk_nvfp4_k, reference_mid.raw_values_data(), nk_f32_k, 32, nullptr));
            for (std::size_t c = 0; c < 32; ++c)
                stats.expect(mid_raw[r * 32 + c] == reference_mid.raw_values_data()[c],
                             "non-zero-start column tile differs from reference");
        }
    }

    // MXFP8 whole-tensor decode equals the serial reference byte-for-byte.
    auto [mx, mx_status] = nk::scaled_tensor<nk::mxfp8e4m3_t>::uninitialized({rows, columns});
    stats.expect(nk::succeeded(mx_status), "MXFP8 allocation failed");
    stats.expect(nk::cast(weights.view(), mx.span()));
    stats.expect(!mx.empty() && mx.block_scales().extent(0) == rows && mx.block_scales().extent(1) == columns / 32,
                 "MXFP8 encode shape mismatch");
    {
        auto [mx_restored, mx_restored_status] = nk::tensor<f32_t>::uninitialized({rows, columns});
        stats.expect(nk::succeeded(mx_restored_status), "MXFP8 restore allocation failed");
        stats.expect(nk::cast<nk::mxfp8e4m3_t>(mx.view(), mx_restored.span()));
        // Reference: decode the same bytes through the serial kernel and require bit-identical output.
        nk_mxfp8e4m3_cref_t const mx_reference = {reinterpret_cast<nk_e4m3_t const *>(mx.elements().byte_data()),
                                                  reinterpret_cast<nk_ue8m0_t const *>(mx.block_scales().byte_data())};
        auto reference_restored = make_vector<f32_t>(rows * columns);
        stats.expect(nk_cast_serial(&mx_reference, nk_mxfp8e4m3_k, reference_restored.raw_values_data(), nk_f32_k,
                                    rows * columns, nullptr));
        auto const *restored_raw = reinterpret_cast<float const *>(mx_restored.data());
        for (std::size_t i = 0; i < rows * columns; ++i)
            stats.expect(restored_raw[i] == reference_restored.raw_values_data()[i],
                         "MXFP8 whole-tensor decode differs from serial reference");
    }

    // Transcode MXFP8 E4M3 to NVFP4 (block-scaled to block-scaled).
    {
        auto [transcoded, transcoded_status] = nk::scaled_tensor<nk::nvfp4_t>::uninitialized({rows, columns});
        stats.expect(nk::succeeded(transcoded_status), "transcode allocation failed");
        auto destination = transcoded.span();
        stats.expect(nk::cast(mx.view(), destination));
        // The transcode result must match decoding MXFP8→dense then re-encoding to NVFP4.
        auto [dense, dense_status] = nk::tensor<f32_t>::uninitialized({rows, columns});
        stats.expect(nk::succeeded(dense_status), "dense allocation failed");
        stats.expect(nk::cast<nk::mxfp8e4m3_t>(mx.view(), dense.span()));
        auto [reference_nvfp4, reference_nvfp4_status] = nk::scaled_tensor<nk::nvfp4_t>::uninitialized({rows, columns});
        stats.expect(nk::succeeded(reference_nvfp4_status), "reference allocation failed");
        stats.expect(nk::cast(dense.view(), reference_nvfp4.span()));
        auto const *transcoded_elements = reinterpret_cast<unsigned char const *>(transcoded.elements().byte_data());
        auto const *reference_nvfp4_elements = reinterpret_cast<unsigned char const *>(
            reference_nvfp4.elements().byte_data());
        std::size_t transcoded_byte_count = nk_block_scaled_elements_size(rows * columns, nvfp4_format);
        for (std::size_t i = 0; i < transcoded_byte_count; ++i)
            stats.expect(transcoded_elements[i] == reference_nvfp4_elements[i],
                         "transcode elements differ from decode-then-encode");
    }
    return stats;
}

/**
 *  @brief Per-format bidirectional round-trip checks, one block, covering three regimes:
 *
 *  - B exactly-representable: a block of powers of two — amax = 2 is a power of two, no scale clip
 *    — must round-trip bit-exactly through UE8M0 formats; NVFP4's two-level f32 × UE4M3 scale only
 *    reaches it within the element resolution, so that case asserts the same relative bound as C.
 *  - C narrow range [1, 1.5): every value's mantissa is below each element format's max mantissa,
 *    so nothing clips and the relative error is bounded by the element resolution
 *    @p narrow_relative_bound.
 *  - D idempotence: re-quantizing an already-quantized block is a fixed point, bit-stable.
 */
template <typename format_>
error_stats_t test_scaled_roundtrip(settings_t const &, float narrow_relative_bound) {
    error_stats_t stats(comparison_family_t::exact_k);
    using nk::f32_t;
    constexpr std::size_t block = format_::elements();
    auto abs_diff = [](float a, float b) { return a > b ? a - b : b - a; };

    auto encode_decode = [&stats](float const *input_values) {
        auto [input, input_status] = nk::tensor<f32_t>::uninitialized({std::size_t {1}, block});
        stats.expect(nk::succeeded(input_status), "input allocation failed");
        auto writable = input.span();
        for (std::size_t i = 0; i < block; ++i) writable(0, i) = f32_t(input_values[i]);
        auto [quantized, quantized_status] = nk::scaled_tensor<format_>::uninitialized({std::size_t {1}, block});
        stats.expect(nk::succeeded(quantized_status), "quantized allocation failed");
        stats.expect(nk::cast(input.view(), quantized.span()));
        auto restored = make_vector<f32_t>(block);
        stats.expect(nk::cast<format_>(quantized.row(0), restored.span()));
        return restored;
    };

    // B — powers of two: amax = 2.0 is a power of two so the scale is exact and nothing clips.
    {
        float values[32];
        for (std::size_t i = 0; i < block; ++i) values[i] = (i % 2 == 0) ? 2.0f : 1.0f;
        auto restored = encode_decode(values);
        for (std::size_t i = 0; i < block; ++i) {
            if constexpr (format_::has_tensor_scale())
                stats.expect(abs_diff(restored.raw_values_data()[i], values[i]) <= narrow_relative_bound * values[i],
                             "NVFP4 power-of-two round-trip outside element resolution");
            else
                stats.expect(restored.raw_values_data()[i] == values[i],
                             "UE8M0 power-of-two round-trip is not bit-exact");
        }
    }
    // C — narrow range [1, 1.5): no clipping, relative error bounded by element resolution.
    {
        float values[32];
        for (std::size_t i = 0; i < block; ++i) values[i] = 1.0f + 0.5f * (static_cast<float>(i % 4) / 4.0f);
        auto restored = encode_decode(values);
        float max_relative_error = 0.0f;
        for (std::size_t i = 0; i < block; ++i) {
            float relative = abs_diff(restored.raw_values_data()[i], values[i]) / values[i];
            if (relative > max_relative_error) max_relative_error = relative;
        }
        stats.expect(max_relative_error <= narrow_relative_bound, "narrow-range round-trip exceeds element resolution");
    }
    // D — idempotence: a second round-trip reproduces the first bit-for-bit.
    {
        float values[32];
        for (std::size_t i = 0; i < block; ++i)
            values[i] = static_cast<float>(static_cast<int>((i * 5) % 19) - 9) * 0.3f;
        auto first = encode_decode(values);
        auto second = encode_decode(first.raw_values_data());
        for (std::size_t i = 0; i < block; ++i)
            stats.expect(second.raw_values_data()[i] == first.raw_values_data()[i], "round-trip is not idempotent");
    }
    return stats;
}

/** Degenerate-input handling: all-zero blocks decode to zero; a NaN poisons only its block. */
error_stats_t test_scaled_tensor_degenerate(settings_t const &) {
    error_stats_t stats(comparison_family_t::exact_k);
    using nk::f32_t;
    auto abs_diff = [](float a, float b) { return a > b ? a - b : b - a; };
    constexpr std::size_t block = 32; // MXFP8 block size
    // All-zero block → zero scale → all-zero decode (no division-by-zero, no NaN).
    {
        auto [input, input_status] = nk::tensor<f32_t>::zeros({std::size_t {1}, block});
        auto [quantized,
              quantized_status] = nk::scaled_tensor<nk::mxfp8e4m3_t>::uninitialized({std::size_t {1}, block});
        stats.expect(nk::succeeded(input_status) && nk::succeeded(quantized_status), "allocation failed");
        stats.expect(nk::cast(input.view(), quantized.span()));
        auto restored = make_vector<f32_t>(block);
        stats.expect(nk::cast<nk::mxfp8e4m3_t>(quantized.row(0), restored.span()));
        for (std::size_t i = 0; i < block; ++i)
            stats.expect(restored.raw_values_data()[i] == 0.0f, "all-zero block did not decode to zero");
    }
    // A NaN in one block sets that block's scale to the NaN sentinel; a clean block is unaffected.
    {
        auto [input, input_status] = nk::tensor<f32_t>::uninitialized({std::size_t {2}, block});
        stats.expect(nk::succeeded(input_status), "input allocation failed");
        auto writable = input.span();
        float const quiet_nan = std::numeric_limits<float>::quiet_NaN();
        for (std::size_t i = 0; i < block; ++i) {
            writable(0, i) = f32_t(i == 3 ? quiet_nan : 1.5f); // row 0 poisoned
            writable(1, i) = f32_t(1.5f);                      // row 1 clean
        }
        auto [quantized,
              quantized_status] = nk::scaled_tensor<nk::mxfp8e4m3_t>::uninitialized({std::size_t {2}, block});
        stats.expect(nk::succeeded(quantized_status), "quantized allocation failed");
        stats.expect(nk::cast(input.view(), quantized.span()));
        auto [restored, restored_status] = nk::tensor<f32_t>::uninitialized({std::size_t {2}, block});
        stats.expect(nk::succeeded(restored_status), "restored allocation failed");
        stats.expect(nk::cast<nk::mxfp8e4m3_t>(quantized.view(), restored.span()));
        auto const *clean_row = reinterpret_cast<float const *>(restored.data()) + block;
        for (std::size_t i = 0; i < block; ++i)
            stats.expect(abs_diff(clean_row[i], 1.5f) <= 0.1f, "clean block corrupted by a NaN in another block");
    }
    return stats;
}

error_stats_t test_custom_allocator(settings_t const &) {
    error_stats_t stats(comparison_family_t::exact_k);
    using custom_alloc_t = nk::aligned_allocator<nk::f32_t, 128>;
    auto [v, v_status] = nk::vector<nk::f32_t, custom_alloc_t>::zeros(256);
    stats.expect(nk::succeeded(v_status), "custom allocator allocation failed");
    stats.expect(v.size() == 256, "custom allocator size mismatch");
    v[128] = nk::f32_t(99.0f);
    stats.expect(v[128] == nk::f32_t(99.0f), "custom allocator value mismatch");
    stats.expect(reinterpret_cast<std::size_t>(v.data()) % 128 == 0, "custom allocator alignment mismatch");
    stats.expect(custom_alloc_t().allocate((std::numeric_limits<std::size_t>::max)()) == nullptr,
                 "allocation size overflow accepted");

    alignas(64) char storage[256];
    nk_allocator_t policy;
    stats.expect(nk_allocator_init_arena(&policy, storage, sizeof(storage), 64) == nk_success_k,
                 "arena initialization failed");
    nk::allocator<float> arena(policy);
    nk::allocator<char> rebound(arena);
    stats.expect(rebound == arena, "rebind lost allocation context");
    stats.expect(rebound.allocate(sizeof(storage)) == nullptr, "arena accepted oversized allocation");
    auto *first = rebound.allocate(64);
    stats.expect(first == storage + 64, "failed allocation consumed arena space");
    auto const saved = policy;
    stats.expect(nk_allocator_init_heap(&policy, 3) == nk_unexpected_dimensions_k &&
                     policy.allocate == saved.allocate && policy.free == saved.free && policy.handle == saved.handle,
                 "invalid initializer changed allocation policy");
    rebound.deallocate(first, 64);

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

    auto [t, t_status] = tensor_t::zeros({2, cols_});
    stats.expect(nk::succeeded(t_status), "tensor allocation failed");

    auto span = t.span();
    auto row0 = span.slice_leading(0).as_vector();
    auto row1 = span.slice_leading(1).as_vector();
    for (std::size_t i = 0; i < cols_; ++i) {
        row0[i] = first_row[i];
        row1[i] = second_row[i];
    }

    auto [sums, sums_status] = nk::sum<value_type_>(t.view(), 0);
    stats.expect(nk::succeeded(sums_status) && !sums.empty(), "axis-0 sum failed");
    auto sum_view = sums.as_vector_view();
    for (std::size_t i = 0; i < cols_; ++i) stats.expect(sum_view[i] == sum_t(expected_sums[i]), "axis-0 sum mismatch");

    auto [minmax, minmax_status] = nk::minmax<value_type_>(t.view(), 0);
    stats.expect(nk::succeeded(minmax_status) && !minmax.min_value.empty() && !minmax.max_value.empty(),
                 "axis-0 minmax failed");
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

template <typename value_type_>
void expect_nonempty(error_stats_t &stats, nk::expected<value_type_> const &result, char const *what) {
    stats.expect(nk::succeeded(result.status) && !result.value.empty(), what);
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

    auto [t, t_status] = tensor_t::zeros({2, 2, cols_});
    stats.expect(nk::succeeded(t_status), "rank-3 tensor allocation failed");

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

    auto [sums0, sums0_status] = nk::sum<value_type_>(t.view(), 0);
    auto [sums1, sums1_status] = nk::sum<value_type_>(t.view(), 1);
    auto [sums2, sums2_status] = nk::sum<value_type_>(t.view(), 2);
    stats.expect(nk::succeeded(sums0_status) && nk::succeeded(sums1_status) && nk::succeeded(sums2_status),
                 "rank-3 axis sum failed");
    assert_flat_tensor_equals(stats, sums0, expected_sum_axis0);
    assert_flat_tensor_equals(stats, sums1, expected_sum_axis1);
    assert_flat_tensor_equals(stats, sums2, expected_sum_axis2);

    auto [moments0, moments0_status] = nk::moments<value_type_>(t.view(), 0);
    auto [moments1, moments1_status] = nk::moments<value_type_>(t.view(), 1);
    auto [moments2, moments2_status] = nk::moments<value_type_>(t.view(), 2);
    stats.expect(nk::succeeded(moments0_status) && nk::succeeded(moments1_status) && nk::succeeded(moments2_status),
                 "rank-3 axis moments failed");
    assert_flat_tensor_equals(stats, moments0.sum, expected_sum_axis0);
    assert_flat_tensor_equals(stats, moments1.sum, expected_sum_axis1);
    assert_flat_tensor_equals(stats, moments2.sum, expected_sum_axis2);

    auto [minmax0, minmax0_status] = nk::minmax<value_type_>(t.view(), 0);
    auto [minmax1, minmax1_status] = nk::minmax<value_type_>(t.view(), 1);
    auto [minmax2, minmax2_status] = nk::minmax<value_type_>(t.view(), 2);
    stats.expect(nk::succeeded(minmax0_status) && nk::succeeded(minmax1_status) && nk::succeeded(minmax2_status),
                 "rank-3 axis minmax failed");
    assert_flat_tensor_equals(stats, minmax0.min_value, expected_min_axis0);
    assert_flat_tensor_equals(stats, minmax0.max_value, expected_max_axis0);
    assert_flat_tensor_equals(stats, minmax1.min_value, expected_min_axis1);
    assert_flat_tensor_equals(stats, minmax1.max_value, expected_max_axis1);
    assert_flat_tensor_equals(stats, minmax2.min_value, expected_min_axis2);
    assert_flat_tensor_equals(stats, minmax2.max_value, expected_max_axis2);
}

error_stats_t test_sub_byte_tensor_axis_reductions(settings_t const &) {
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

error_stats_t test_sub_byte_tensor_rank3_axis_reductions(settings_t const &) {
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

error_stats_t test_rank1_negative_stride_reductions(settings_t const &) {
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

error_stats_t test_rank1_axis_reductions(settings_t const &) {
    error_stats_t stats(comparison_family_t::exact_k);
    auto [v, v_status] = nk::tensor<nk::i8_t>::zeros({4});
    stats.expect(nk::succeeded(v_status), "rank-1 tensor allocation failed");
    auto values = v.as_vector_span();
    values[0u] = 4;
    values[1u] = -2;
    values[2u] = 7;
    values[3u] = -5;

    auto [sums, sums_status] = nk::sum<nk::i8_t>(v.view(), 0);
    auto [moments, moments_status] = nk::moments<nk::i8_t>(v.view(), 0);
    auto [mins, mins_status] = nk::min<nk::i8_t>(v.view(), 0);
    auto [maxs, maxs_status] = nk::max<nk::i8_t>(v.view(), 0);
    auto [argmins, argmins_status] = nk::argmin<nk::i8_t>(v.view(), 0);
    auto [argmaxs, argmaxs_status] = nk::argmax<nk::i8_t>(v.view(), 0);

    stats.expect(nk::succeeded(sums_status) && nk::succeeded(moments_status) && !sums.empty() && !moments.sum.empty(),
                 "rank-1 axis moments failed");
    stats.expect(nk::succeeded(mins_status) && nk::succeeded(maxs_status) && nk::succeeded(argmins_status) &&
                     nk::succeeded(argmaxs_status) && !mins.empty() && !maxs.empty() && !argmins.empty() &&
                     !argmaxs.empty(),
                 "rank-1 axis minmax failed");
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

error_stats_t test_packed_tensor_fail_closed_views(settings_t const &) {
    error_stats_t stats(comparison_family_t::exact_k);
    auto [packed, packed_status] = nk::tensor<nk::i4x2_t>::zeros({2, 4});
    stats.expect(nk::succeeded(packed_status), "packed tensor allocation failed");
    stats.expect(packed.view().transpose().empty(), "packed transpose should fail closed");
    stats.expect((!packed(1, nk::slice).empty()), "packed row slice should remain supported");
    stats.expect((packed(1, 2, nk::slice).empty()), "packed scalar trailing slice should fail closed");
#if NUMKONG_HAS_MULTIDIMENSIONAL_SUBSCRIPT_
    stats.expect((!packed[1, nk::slice].empty()), "packed operator[] row slice should remain supported");
    stats.expect((packed[1, 2, nk::slice].empty()), "packed operator[] scalar trailing slice should fail closed");
#endif
    return stats;
}

/** Smoke-test for the vector-shaped reduction wrappers @c nk::moments, @c minmax, @c sum, and more,
 *  exercising every accessor variant on a small random vector. Numerical accuracy of the underlying
 *  kernels is validated by @c test_reduce_moments and @c test_reduce_minmax above. */
template <typename value_type_>
error_stats_t test_vector_reductions_for_type(settings_t const &settings) {
    error_stats_t stats(comparison_family_t::exact_k);
    auto v = make_vector<value_type_>(32);
    std::mt19937 generator(42);
    fill_random(settings, generator, v);
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

void test_vector_types(error_stats_section_t &check) {
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
    check("vector_block_scaled_composites", test_block_scaled_composites);
    check("vector_scaled_tensor", test_scaled_tensor);

    // Per-element resolution = 2^-mantissa_bits (E2M1:1, E3M2/E5M2:2, E2M3/E4M3:3 mantissa bits);
    // MXINT8 resolves to ~1/64 over the narrow band.
    check("vector_scaled_roundtrip_nvfp4", test_scaled_roundtrip<nk::nvfp4_t>, 0.5f);
    check("vector_scaled_roundtrip_mxfp4", test_scaled_roundtrip<nk::mxfp4_t>, 0.5f);
    check("vector_scaled_roundtrip_mxfp6e2m3", test_scaled_roundtrip<nk::mxfp6e2m3_t>, 0.125f);
    check("vector_scaled_roundtrip_mxfp6e3m2", test_scaled_roundtrip<nk::mxfp6e3m2_t>, 0.25f);
    check("vector_scaled_roundtrip_mxfp8e4m3", test_scaled_roundtrip<nk::mxfp8e4m3_t>, 0.125f);
    check("vector_scaled_roundtrip_mxfp8e5m2", test_scaled_roundtrip<nk::mxfp8e5m2_t>, 0.25f);
    check("vector_scaled_roundtrip_mxint8", test_scaled_roundtrip<nk::mxint8_t>, 0.05f);
    check("vector_scaled_tensor_degenerate", test_scaled_tensor_degenerate);

    check("vector_custom_allocator", test_custom_allocator);

#if NUMKONG_TEST_FORMAT_
    check("vector_format_scalars", test_format_scalars);
#endif
}

/**
 *  @brief Explicit template instantiation test for all tensor-level operations.
 *
 *  Forces the compiler to fully instantiate every type × operation combination, catching signature
 *  mismatches, missing type traits, and implicit conversion errors that syntax-only checks miss.
 */
template <typename value_type_>
error_stats_t test_tensor_ops_for_type(settings_t const &) {
    error_stats_t stats(comparison_family_t::exact_k);
    using tensor_t = nk::tensor<value_type_>;

    // Create small test tensors
    auto [a, a_status] = tensor_t::zeros({4, 8});
    auto [b, b_status] = tensor_t::zeros({4, 8});
    stats.expect(nk::succeeded(a_status) && nk::succeeded(b_status), "tensor allocation");

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
    expect_nonempty(stats, nk::sum<value_type_>(av, 0), "sum returned empty");
    expect_nonempty(stats, nk::sum<value_type_>(av, 1, nk::keep_dims_k), "sum returned empty");
    { [[maybe_unused]] auto r = nk::moments<value_type_>(av, 1); }
    { [[maybe_unused]] auto r = nk::minmax<value_type_>(av, 0); }
    { [[maybe_unused]] auto r = nk::minmax<value_type_>(av, 1, nk::keep_dims_k); }
    expect_nonempty(stats, nk::min<value_type_>(av, 0), "min returned empty");
    expect_nonempty(stats, nk::min<value_type_>(av, 1, nk::keep_dims_k), "min returned empty");
    expect_nonempty(stats, nk::max<value_type_>(av, 0), "max returned empty");
    expect_nonempty(stats, nk::max<value_type_>(av, 1, nk::keep_dims_k), "max returned empty");
    expect_nonempty(stats, nk::argmin<value_type_>(av, 0), "argmin returned empty");
    expect_nonempty(stats, nk::argmax<value_type_>(av, 1, nk::keep_dims_k), "argmax returned empty");

    // Elementwise binary
    expect_nonempty(stats, nk::add<value_type_>(av, bv), "add returned empty");
    expect_nonempty(stats, nk::sub<value_type_>(av, bv), "sub returned empty");
    expect_nonempty(stats, nk::mul<value_type_>(av, bv), "mul returned empty");

    // Elementwise binary with scalar
    using scale_t = typename value_type_::scale_t;
    scale_t scalar {1};
    expect_nonempty(stats, nk::add<value_type_>(av, scalar), "add returned empty");
    expect_nonempty(stats, nk::sub<value_type_>(av, scalar), "sub returned empty");
    expect_nonempty(stats, nk::mul<value_type_>(av, scalar), "mul returned empty");

    // Elementwise into
    auto [out, out_status] = tensor_t::zeros({4, 8});
    stats.expect(nk::succeeded(out_status), "output allocation");
    stats.expect(nk::succeeded(nk::add<value_type_>(av, bv, out.span())), "add into span failed");
    stats.expect(nk::succeeded(nk::sub<value_type_>(av, bv, out.span())), "sub into span failed");
    stats.expect(nk::succeeded(nk::mul<value_type_>(av, bv, out.span())), "mul into span failed");
    stats.expect(nk::succeeded(nk::add<value_type_>(av, scalar, out.span())), "add into span failed");
    stats.expect(nk::succeeded(nk::sub<value_type_>(av, scalar, out.span())), "sub into span failed");
    stats.expect(nk::succeeded(nk::mul<value_type_>(av, scalar, out.span())), "mul into span failed");

    // Affine
    scale_t alpha {1}, beta {0};
    expect_nonempty(stats, nk::scale<value_type_>(av, alpha, beta), "scale returned empty");
    expect_nonempty(stats, nk::blend<value_type_>(av, bv, alpha, beta), "blend returned empty");
    expect_nonempty(stats, nk::fma<value_type_>(av, bv, av, alpha, beta), "fma returned empty");
    stats.expect(nk::succeeded(nk::scale<value_type_>(av, alpha, beta, out.span())), "scale into span failed");
    stats.expect(nk::succeeded(nk::blend<value_type_>(av, bv, alpha, beta, out.span())), "blend into span failed");
    stats.expect(nk::succeeded(nk::fma<value_type_>(av, bv, av, alpha, beta, out.span())), "fma into span failed");

    // from 1D
    {
        auto [from1d, from1d_status] = tensor_t::from({value_type_ {}, value_type_ {}, value_type_ {}});
        stats.expect(nk::succeeded(from1d_status), "from 1D failed");
        stats.expect(from1d.rank() == 1 && from1d.numel() == 3, "from 1D shape mismatch");
    }

    // from 2D
    {
        auto [from2d,
              from2d_status] = tensor_t::from({{value_type_ {}, value_type_ {}}, {value_type_ {}, value_type_ {}}});
        stats.expect(nk::succeeded(from2d_status), "from 2D failed");
        stats.expect(from2d.rank() == 2 && from2d.extent(0) == 2 && from2d.extent(1) == 2, "from 2D shape mismatch");
    }

    // row() access
    {
        auto row0 = a.row(0);
        stats.expect(row0.rank() == 1 && row0.extent(0) == 8, "row() shape mismatch");
    }

    // Convenience view constructor (ptr, rows, columns)
    {
        nk::tensor_view<value_type_> view_from_ptr(a.data(), 4, 8);
        stats.expect(view_from_ptr.rank() == 2 && view_from_ptr.extent(0) == 4, "convenience view ctor mismatch");
    }
    return stats;
}

template <typename value_type_>
error_stats_t test_tensor_symmetric_for_type(settings_t const &) {
    error_stats_t stats(comparison_family_t::exact_k);
    using tensor_t = nk::tensor<value_type_>;
    auto [a, a_status] = tensor_t::zeros({4, 8});
    stats.expect(nk::succeeded(a_status), "tensor allocation");
    auto am = a.as_matrix_view();

    expect_nonempty(stats, nk::dots_symmetric<value_type_>(am), "dots_symmetric returned empty");
    expect_nonempty(stats, nk::angulars_symmetric<value_type_>(am), "angulars_symmetric returned empty");
    expect_nonempty(stats, nk::euclideans_symmetric<value_type_>(am), "euclideans_symmetric returned empty");
    return stats;
}

template <typename value_type_>
error_stats_t test_tensor_packed_for_type(settings_t const &) {
    error_stats_t stats(comparison_family_t::exact_k);
    using tensor_t = nk::tensor<value_type_>;
    auto [a, a_status] = tensor_t::zeros({4, 8});
    auto [b, b_status] = tensor_t::zeros({6, 8});
    stats.expect(nk::succeeded(a_status) && nk::succeeded(b_status), "tensor allocation");

    // packed_matrix
    auto bm = b.as_matrix_view();
    auto [packed, packed_status] = nk::packed_matrix<value_type_, nk::aligned_allocator<char>>::make(bm);
    auto am = a.as_matrix_view();
    auto [result, result_status] = nk::matrix<typename value_type_::dot_result_t>::zeros({4, 6});
    stats.expect(nk::succeeded(packed_status) && !packed.empty(), "packed matrix empty");
    stats.expect(nk::succeeded(result_status) && !result.empty(), "packed result empty");
    stats.expect(nk::succeeded(nk::dots_packed<value_type_>(am, packed, result.span())), "dots_packed failed");
    return stats;
}

template <typename value_type_>
error_stats_t test_tensor_maxsim_for_type(settings_t const &) {
    error_stats_t stats(comparison_family_t::exact_k);
    using tensor_t = nk::tensor<value_type_>;
    auto [q, q_status] = tensor_t::zeros({3, 16});
    auto [d, d_status] = tensor_t::zeros({5, 16});
    stats.expect(nk::succeeded(q_status) && nk::succeeded(d_status), "tensor allocation");

    auto qm = q.as_matrix_view();
    auto dm = d.as_matrix_view();

    auto [pq, pq_status] = nk::packed_maxsim<value_type_>::make(qm);
    auto [pd, pd_status] = nk::packed_maxsim<value_type_>::make(dm);
    stats.expect(nk::succeeded(pq_status) && !pq.empty(), "packed query empty");
    stats.expect(nk::succeeded(pd_status) && !pd.empty(), "packed document empty");
    { [[maybe_unused]] auto r = nk::maxsim(pq, pd); }
    return stats;
}

error_stats_t test_view_overloads(settings_t const &) {
    error_stats_t stats(comparison_family_t::exact_k);
    nk::f32_t a_data[8] {}, c_data[64] {};
    std::array<nk::f32_t, 8> b_array {};
    std::vector<nk::f32_t> b_vector(8);
    std::vector<float> b_raw(8);
    auto a_view = nk::vector_view<nk::f32_t>(a_data, 8u);
    auto c_view = nk::vector_view<nk::f32_t>(c_data, 64u);

    // Inputs are all zero, so the metrics that are defined there must come back zero.
    auto [dot, dot_status] = nk::dot<nk::f32_t>(a_data, b_array);
    stats.expect(nk::succeeded(dot_status) && dot == nk::f32_t(0), "dot of zero vectors");
    auto [euclidean, euclidean_status] = nk::euclidean<nk::f32_t>(a_view, b_vector);
    stats.expect(nk::succeeded(euclidean_status) && euclidean == nk::f32_t(0), "euclidean of zero vectors");
    auto [sqeuclidean, sqeuclidean_status] = nk::sqeuclidean<nk::f32_t>(std::span<nk::f32_t const>(b_vector), a_view);
    stats.expect(nk::succeeded(sqeuclidean_status) && sqeuclidean == nk::f32_t(0), "sqeuclidean of zero vectors");
    auto [raw_dot, raw_dot_status] = nk::dot<nk::f32_t>(a_view, b_raw);
    stats.expect(nk::succeeded(raw_dot_status) && raw_dot == nk::f32_t(0), "dot of raw storage");
    float raw_rows[16] {};
    double raw_products[16] {};
    stats.expect(nk::dots_symmetric<nk::f32_t>(nk::matrix_view<float>(raw_rows, 4u, 4u),
                                               nk::matrix_span<double>(raw_products, 4u, 4u)));
    stats.expect(nk::angular<nk::f32_t>(a_view, b_array).status);
    stats.expect(nk::bilinear<nk::f32_t>(a_view, b_array, c_view).status);
    stats.expect(nk::mahalanobis<nk::f32_t>(a_view, b_vector, c_data).status);

    auto short_view = nk::vector_view<nk::f32_t>(a_data, 7u);
    stats.expect(nk::dot<nk::f32_t>(a_view, short_view).status == nk::status_t::unexpected_dimensions_k,
                 "dot of mismatched lengths");
    stats.expect(nk::bilinear<nk::f32_t>(a_view, b_array, short_view).status == nk::status_t::unexpected_dimensions_k,
                 "bilinear with a non-square metric");
    auto strided_view = nk::vector_view<nk::f32_t>(reinterpret_cast<char const *>(a_data), 4u, 2 * sizeof(nk::f32_t));
    stats.expect(nk::dot<nk::f32_t>(strided_view, strided_view).status == nk::status_t::unexpected_dimensions_k,
                 "dot of strided views");
    stats.expect(strided_view.values().status == nk::status_t::unexpected_dimensions_k, "values of a strided view");
    stats.expect(a_view.values() && a_view.values().value.size() == 8, "values of a contiguous view");

    // Sub-byte runs must fill whole storage values: 5 nibbles or 13 bits end mid-byte.
    nk::i4x2_t nibbles[4] {};
    nk::u1x8_t bits[2] {};
    auto odd_nibbles = nk::vector_view<nk::i4x2_t>(nibbles, 5u);
    auto odd_bits = nk::vector_view<nk::u1x8_t>(bits, 13u);
    stats.expect(odd_nibbles.values().status == nk::status_t::unexpected_dimensions_k, "values of 5 nibbles");
    stats.expect(nk::dot<nk::i4x2_t>(odd_nibbles, odd_nibbles).status == nk::status_t::unexpected_dimensions_k,
                 "dot of 5 nibbles");
    stats.expect(nk::hamming<nk::u1x8_t>(odd_bits, odd_bits).status == nk::status_t::unexpected_dimensions_k,
                 "hamming of 13 bits");
    auto even_nibbles = nk::vector_view<nk::i4x2_t>(nibbles, 6u);
    auto [nibbles_dot, nibbles_status] = nk::dot<nk::i4x2_t>(even_nibbles, even_nibbles);
    stats.expect(nk::succeeded(nibbles_status) && nibbles_dot == nk::i32_t(0), "dot of 6 nibbles");

    std::vector<nk::f32_t> output(8);
    stats.expect(nk::scale<nk::f32_t>(a_view, 2.0f, 1.0f, output));
    stats.expect(output[7] == nk::f32_t(1), "scale into a std::vector");
    stats.expect(nk::add<nk::f32_t>(output, b_array, nk::vector_span<nk::f32_t>(a_data, 8u)));
    stats.expect(a_data[0] == nk::f32_t(1), "add into a span");
    stats.expect(nk::sin<nk::f32_t>(b_array, output));
    stats.expect(output[0] == nk::f32_t(0), "sin into a std::vector");
    stats.expect(nk::add<nk::f32_t>(a_view, short_view, output) == nk::status_t::unexpected_dimensions_k,
                 "add of mismatched lengths");

    auto [moments, moments_status] = nk::reduce_moments<nk::f32_t>(a_data);
    stats.expect(nk::succeeded(moments_status) && static_cast<double>(moments.sum) == 8.0, "moments of ones");
    auto [extremes, extremes_status] = nk::reduce_minmax<nk::f32_t>(b_vector);
    stats.expect(nk::succeeded(extremes_status) && extremes.min_index == 0, "minmax of zeros");
    return stats;
}

/** Stateful allocator counting its allocations through a shared counter, which proves the
 *  factories use the instance they were given. */
template <typename value_type_>
struct counting_allocator {
    using value_type = value_type_;
    std::size_t *allocations = nullptr;

    counting_allocator() noexcept = default;
    explicit counting_allocator(std::size_t *counter) noexcept : allocations(counter) {}
    template <typename other_type_>
    counting_allocator(counting_allocator<other_type_> const &other) noexcept : allocations(other.allocations) {}

    value_type_ *allocate(std::size_t n) noexcept {
        if (allocations) ++*allocations;
        return nk::aligned_allocator<value_type_>().allocate(n);
    }
    void deallocate(value_type_ *p, std::size_t n) noexcept { nk::aligned_allocator<value_type_>().deallocate(p, n); }

    template <typename other_type_>
    bool operator==(counting_allocator<other_type_> const &other) const noexcept {
        return allocations == other.allocations;
    }
};

error_stats_t test_custom_allocator_factories(settings_t const &) {
    error_stats_t stats(comparison_family_t::exact_k);
    using custom_alloc_t = nk::aligned_allocator<nk::f32_t, 128>;
    auto [a, a_status] = nk::tensor<nk::f32_t>::zeros({4, 8});
    stats.expect(nk::succeeded(a_status), "tensor allocation");
    auto av = a.view();

    expect_nonempty(stats, nk::scale<nk::f32_t, 8, custom_alloc_t>(av, 1.0, 0.0), "scale returned empty");
    expect_nonempty(stats, nk::sin<nk::f32_t, 8, custom_alloc_t>(av), "sin returned empty");

    using sum_alloc_t = nk::aligned_allocator<nk::f64_t, 128>;
    expect_nonempty(stats, nk::sum<nk::f32_t, 8, sum_alloc_t>(av, 0), "sum returned empty");

    std::size_t allocations = 0;
    counting_allocator<nk::f32_t> counting(&allocations);
    auto expect_counted = [&](auto const &result, std::size_t expected, char const *what) {
        expect_nonempty(stats, result, what);
        stats.expect(allocations == expected, what);
    };
    expect_counted(nk::scale(av, 1.0, 0.0, counting), 1, "scale ignored the allocator instance");
    expect_counted(nk::sin(av, counting), 2, "sin ignored the allocator instance");
    expect_counted(nk::copy(av, counting), 3, "copy ignored the allocator instance");
    expect_counted(nk::sum(av, 0, nk::collapse_dims_k, counting_allocator<nk::f64_t>(counting)), 4,
                   "sum ignored the allocator instance");
    auto rows = a.as_matrix_view();
    expect_counted(nk::dots_symmetric<nk::f32_t>(rows, counting_allocator<nk::f64_t>(counting)), 5,
                   "dots_symmetric ignored the allocator instance");

    using mx_t = nk::mxfp8e4m3_t;
    using mx_tensor_t = nk::scaled_tensor<mx_t, counting_allocator<mx_t::element_t>, counting_allocator<mx_t::scale_t>>;
    expect_counted(mx_tensor_t::uninitialized({4, 64}, counting_allocator<mx_t::element_t>(counting),
                                              counting_allocator<mx_t::scale_t>(counting)),
                   7, "scaled_tensor ignored the allocator instances");
    return stats;
}

template <typename from_type_, typename to_type_>
error_stats_t test_cast_for_types(settings_t const &settings) {
    error_stats_t stats(comparison_family_t::exact_k);
    auto src = make_vector<from_type_>(64);
    auto dst = make_vector<to_type_>(64);
    std::mt19937 generator(42);
    fill_random(settings, generator, src);

    auto src_view = nk::vector_view<from_type_>(src.values_data(), static_cast<std::size_t>(src.size()));
    auto dst_span = nk::vector_span<to_type_>(dst.values_data(), static_cast<std::size_t>(dst.size()));

    // The pointer-level and view/span APIs must agree element for element.
    auto dst_via_views = make_vector<to_type_>(64);
    auto dst_via_views_span = nk::vector_span<to_type_>(dst_via_views.values_data(),
                                                        static_cast<std::size_t>(dst_via_views.size()));
    stats.expect(nk::cast<from_type_, to_type_>(src.values_data(), src.size(), dst.values_data()));
    stats.expect(nk::cast<from_type_, to_type_>(src_view, dst_via_views_span));
    for (std::size_t i = 0; i < dst.size(); ++i) stats.accumulate(dst[i], dst_via_views[i]);
    return stats;
}

#if NUMKONG_TEST_FORMAT_
error_stats_t test_format_scalars(settings_t const &) {
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
    nk::sub_byte_ref<nk::i4x2_t> iref(&packed_i, 1);
    stats.expect(std::format("{}", iref) == "3", "i4 sub_byte_ref default format");
    stats.expect(std::format("{:x}", iref) == "3", "i4 sub_byte_ref hex format");
    stats.expect(std::format("{:b}", iref) == "0011", "i4 sub_byte_ref binary format");
    stats.expect(std::format("{:#}", iref) == "3 [0x3]", "i4 sub_byte_ref annotated format");

    nk_u4x2_t packed_u = 0xA7;
    nk::sub_byte_ref<nk::u4x2_t> uref(&packed_u, 0);
    stats.expect(std::format("{}", uref) == "10", "u4 sub_byte_ref default format");
    stats.expect(std::format("{:x}", uref) == "a", "u4 sub_byte_ref hex format");
    stats.expect(std::format("{:b}", uref) == "1010", "u4 sub_byte_ref binary format");

    nk_u1x8_t packed_b = 0xA0;
    nk::sub_byte_ref<nk::u1x8_t> bref(&packed_b, 0);
    stats.expect(std::format("{}", bref) == "1", "u1 sub_byte_ref format");
    return stats;
}
#endif // NUMKONG_TEST_FORMAT_

/** Typed-pointer ctors — count, initializer_list, @c std::array — with an out-of-range guard. */
error_stats_t test_typed_pointer_ctors(settings_t const &) {
    error_stats_t stats(comparison_family_t::exact_k);
    alignas(64) float buf[64];
    for (int i = 0; i < 64; ++i) buf[i] = static_cast<float>(i);
    auto *p = reinterpret_cast<nk::f32_t *>(buf);

    nk::tensor_view<nk::f32_t> v1(p, 6); // rank-1 (ptr, count)
    stats.expect(v1.rank() == 1 && v1.numel() == 6, "typed-ptr view (count) ctor");
    nk::tensor_span<nk::f32_t> s1(p, 6);
    stats.expect(s1.rank() == 1 && s1.numel() == 6, "typed-ptr span (count) ctor");

    nk::tensor_view<nk::f32_t> v2(p, {2, 3}); // (ptr, initializer_list)
    stats.expect(v2.rank() == 2 && v2.numel() == 6, "typed-ptr view (list) ctor");

    std::array<std::size_t, 2> ext {3, 4}; // (ptr, std::array)
    nk::tensor_view<nk::f32_t> v3(p, ext);
    stats.expect(v3.rank() == 2 && v3.numel() == 12, "typed-ptr view (array) ctor");

    // An out-of-range rank (too many, or empty) fails closed to an empty handle — like reshape() —
    // rather than overflowing the fixed-capacity shape storage.
    nk::tensor_view<nk::f32_t, 2> over(p, {2, 3, 4});
    stats.expect(over.empty(), "over-rank list -> empty handle");
    nk::tensor_span<nk::f32_t, 4> empty_list(p, std::initializer_list<std::size_t> {});
    stats.expect(empty_list.empty(), "empty extents list -> empty handle");
    return stats;
}

/** `explicit operator bool` on every owning + non-owning handle type. */
error_stats_t test_operator_bool(settings_t const &) {
    error_stats_t stats(comparison_family_t::exact_k);
    auto [t, t_status] = nk::tensor<nk::f32_t>::uninitialized({2, 3});
    nk::tensor<nk::f32_t> te {};
    stats.expect(nk::succeeded(t_status) && static_cast<bool>(t) && !te, "tensor bool");
    stats.expect(static_cast<bool>(t.view()) && !te.view(), "tensor_view bool");
    stats.expect(static_cast<bool>(t.span()), "tensor_span bool");

    auto v = make_vector<nk::f32_t>(4);
    nk::vector<nk::f32_t> ve {};
    stats.expect(static_cast<bool>(v) && !ve, "vector bool");
    stats.expect(static_cast<bool>(v.view()) && static_cast<bool>(v.span()), "vector view/span bool");

    auto [q, q_status] = nk::scaled_tensor<nk::nvfp4_t>::uninitialized({2, 32});
    nk::scaled_tensor<nk::nvfp4_t> qe {};
    stats.expect(nk::succeeded(q_status) && static_cast<bool>(q) && !qe, "scaled_tensor bool");
    stats.expect(static_cast<bool>(q.view()) && static_cast<bool>(q.span()), "scaled view/span bool");
    return stats;
}

/** Templated `flatten<out_rank_>()` with an explicit non-default output rank. */
error_stats_t test_flatten_out_rank(settings_t const &) {
    error_stats_t stats(comparison_family_t::exact_k);
    auto [t, t_status] = nk::tensor<nk::f32_t>::uninitialized({2, 3, 4});
    stats.expect(nk::succeeded(t_status), "tensor allocation");
    auto f1 = t.flatten<1>();
    stats.expect(f1.rank() == 1 && f1.numel() == 24, "tensor flatten<1>");
    auto fv = t.view().flatten<1>();
    stats.expect(fv.rank() == 1 && fv.numel() == 24, "view flatten<1>");
    auto fs = t.span().flatten<2>();
    stats.expect(fs.numel() == 24, "span flatten<2>");
    return stats;
}

/** Fixed-capacity resize contract: data()-stability, beyond-capacity fail, reserve/clear/move. */
error_stats_t test_resize_capacity(settings_t const &) {
    error_stats_t stats(comparison_family_t::exact_k);
    auto [t, t_status] = nk::tensor<nk::f32_t>::uninitialized({8, 4}); // capacity 32
    stats.expect(nk::succeeded(t_status) && t.capacity() == 32, "capacity from initial shape");
    auto *p0 = t.data();
    stats.expect(nk::succeeded(t.resize({2, 4})) && t.numel() == 8 && t.data() == p0,
                 "resize within capacity keeps data()");
    stats.expect(t.resize({100, 100}) == nk::status_t::unexpected_dimensions_k && t.numel() == 8,
                 "resize beyond capacity fails, shape kept");
    std::size_t ext[2] = {4, 4};
    stats.expect(nk::succeeded(t.resize(ext, 2)) && t.numel() == 16, "resize (ptr,rank) overload");
    stats.expect(nk::succeeded(t.reserve(64)) && t.capacity() >= 64, "reserve grows capacity");
    stats.expect(nk::succeeded(t.resize({8, 8})), "resize into grown capacity");
    t.clear();
    stats.expect(t.empty() && t.capacity() >= 64, "clear -> empty, capacity kept");

    auto [t2, t2_status] = nk::tensor<nk::f32_t>::uninitialized({4, 4});
    stats.expect(nk::succeeded(t2_status), "tensor allocation");
    auto cap = t2.capacity();
    nk::tensor<nk::f32_t> t3 = std::move(t2);
    stats.expect(t3.capacity() == cap, "move preserves capacity");

    auto [q, q_status] = nk::scaled_tensor<nk::nvfp4_t>::uninitialized({2, 32}); // block_size 16
    stats.expect(nk::succeeded(q_status), "scaled tensor allocation");
    stats.expect(nk::succeeded(q.resize({2, 16})) && q.extent(1) == 16, "scaled coordinated resize");
    stats.expect(q.resize({2, 20}) == nk::status_t::unexpected_dimensions_k,
                 "scaled resize rejects non-block-aligned last extent");
    stats.expect(q.resize({8, 64}) == nk::status_t::unexpected_dimensions_k, "scaled resize beyond capacity fails");
    stats.expect(nk::succeeded(q.reserve(256)) && nk::succeeded(q.resize({4, 64})), "scaled reserve then resize");

    auto [v, v_status] = nk::vector<nk::f32_t>::uninitialized(16);
    stats.expect(nk::succeeded(v_status) && v.capacity() == 16, "vector capacity");
    auto *vp0 = v.values_data();
    stats.expect(nk::succeeded(v.resize(8)) && v.size() == 8 && v.values_data() == vp0, "vector resize keeps data()");
    stats.expect(v.resize(100) == nk::status_t::unexpected_dimensions_k, "vector resize beyond capacity fails");
    stats.expect(nk::succeeded(v.reserve(64)) && nk::succeeded(v.resize(50)), "vector reserve then resize");
    v.clear();
    stats.expect(v.empty() && v.capacity() >= 64, "vector clear");
    return stats;
}

/** Smoke-test for the tensor-shaped trig wrappers @c nk::sin, @c cos and @c atan, running
 *  allocating and into-span variants on a small zero tensor, just exercising the dispatch paths,
 *  not the numerical accuracy, which the kernel tests above cover. */
template <typename value_type_>
error_stats_t test_tensor_trig_for_type(settings_t const &) {
    error_stats_t stats(comparison_family_t::exact_k);
    using tensor_t = nk::tensor<value_type_>;
    auto [a, a_status] = tensor_t::zeros({4, 8});
    auto [out, out_status] = tensor_t::zeros({4, 8});
    stats.expect(nk::succeeded(a_status) && nk::succeeded(out_status), "tensor allocation");
    auto av = a.view();

    expect_nonempty(stats, nk::sin<value_type_>(av), "sin returned empty");
    expect_nonempty(stats, nk::cos<value_type_>(av), "cos returned empty");
    expect_nonempty(stats, nk::atan<value_type_>(av), "atan returned empty");
    stats.expect(nk::succeeded(nk::sin<value_type_>(av, out.span())), "sin into span failed");
    stats.expect(nk::succeeded(nk::cos<value_type_>(av, out.span())), "cos into span failed");
    stats.expect(nk::succeeded(nk::atan<value_type_>(av, out.span())), "atan into span failed");
    return stats;
}

/** Packs through @c packed_attention and runs the view overloads, which must match the raw-pointer
 *  layer bit for bit on the same capabilities, in place and allocating. */
template <typename value_type_>
error_stats_t test_tensor_attention_for_type(settings_t const &settings) {
    using result_t = typename value_type_::attention_result_t;
    error_stats_t stats(comparison_family_t::exact_k);
    std::mt19937 generator(settings.seed.value);
    constexpr std::size_t tokens = 10, heads = 4, key_value_heads = 2, depth = 32;
    constexpr nk_f32_t scale = 0.125f;
    nk_u32_t const offsets[] = {0, 4, 10}, lengths[] = {4, 6};
    nk::vector_view<nk_u32_t> const segment_offsets(offsets, 3u), segment_lengths(lengths, 2u);

    auto keys = nk::tensor<value_type_>::zeros({tokens, key_value_heads, depth});
    auto values = nk::tensor<value_type_>::zeros({tokens, key_value_heads, depth});
    auto queries = nk::tensor<value_type_>::zeros({tokens, heads, depth});
    auto output = nk::tensor<result_t>::zeros({tokens, heads, depth});
    auto reference = nk::tensor<result_t>::zeros({tokens, heads, depth});
    stats.expect(keys && values && queries && output && reference, "attention operands allocation failed");
    if (stats.failed_expectations) return stats;
    nk::fill_uniform(generator, keys.value.data(), keys.value.numel());
    nk::fill_uniform(generator, values.value.data(), values.value.numel());
    nk::fill_uniform(generator, queries.value.data(), queries.value.numel());

    auto packed = nk::packed_attention<value_type_>::make(keys.value.view(), values.value.view(), segment_offsets,
                                                          segment_lengths);
    stats.expect(packed.status);
    if (!packed) return stats;
    stats.expect(packed.value.key_value_head_count() == key_value_heads && packed.value.depth() == depth &&
                     packed.value.segment_count() == 2,
                 "packed attention shape mismatch");

    auto packed_bytes = nk::attention_pack_size<value_type_>(key_value_heads, depth, tokens, 2);
    stats.expect(packed_bytes.status);
    if (!packed_bytes) return stats;
    auto raw_packed = make_vector<char>(packed_bytes.value);
    std::size_t const key_value_stride = key_value_heads * depth * sizeof(value_type_),
                      query_stride = heads * depth * sizeof(value_type_),
                      output_stride = heads * depth * sizeof(result_t);
    stats.expect(nk::attention_pack<value_type_>(keys.value.data(), values.value.data(), key_value_heads, depth,
                                                 offsets, lengths, 2, key_value_stride, key_value_stride,
                                                 raw_packed.values_data()));
    auto expect_equal = [&](nk::tensor_view<result_t> actual) {
        for (std::size_t i = 0; i < actual.numel(); ++i) stats.accumulate(actual[i], reference.value[i]);
    };

    stats.expect(nk::attention_packed<value_type_>(queries.value.data(), raw_packed.values_data(),
                                                   reference.value.data(), nullptr, heads, key_value_heads, depth,
                                                   offsets, query_stride, output_stride, scale));
    stats.expect(nk::attention_packed<value_type_>(queries.value.view(), packed.value, output.value.span(), scale));
    expect_equal(output.value.view());
    auto bidirectional = nk::attention_packed<value_type_>(queries.value.view(), packed.value, scale);
    stats.expect(bidirectional.status);
    if (bidirectional) expect_equal(bidirectional.value.view());

    // A sliding window of three keys, causal
    std::size_t const keys_before = 2, keys_after = 0;
    stats.expect(nk::attention_packed<value_type_>(
        queries.value.data(), raw_packed.values_data(), reference.value.data(), nullptr, heads, key_value_heads, depth,
        offsets, query_stride, output_stride, scale, keys_before, keys_after));
    stats.expect(nk::attention_packed<value_type_>(queries.value.view(), packed.value, output.value.span(), scale,
                                                   keys_before, keys_after));
    expect_equal(output.value.view());
    auto windowed = nk::attention_packed<value_type_>(queries.value.view(), packed.value, scale, keys_before,
                                                      keys_after);
    stats.expect(windowed.status);
    if (windowed) expect_equal(windowed.value.view());

    auto narrow = nk::tensor<result_t>::zeros({tokens, heads, depth / 2});
    stats.expect(narrow && nk::attention_packed<value_type_>(queries.value.view(), packed.value, narrow.value.span(),
                                                             scale) == nk::status_t::unexpected_dimensions_k,
                 "mismatched output shape must be rejected");
    stats.expect(nk::packed_attention<value_type_>::make(keys.value.view(), values.value.view(), segment_lengths,
                                                         segment_lengths)
                         .status == nk::status_t::unexpected_dimensions_k,
                 "segment offsets must hold one more entry than lengths");
    return stats;
}

void test_tensor_ops(error_stats_section_t &check) {
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
    check("tensor_custom_allocator_factories", test_custom_allocator_factories);

    check("tensor_cast_f32_to_f16", test_cast_for_types<nk::f32_t, nk::f16_t>);
    check("tensor_cast_f16_to_f32", test_cast_for_types<nk::f16_t, nk::f32_t>);
    check("tensor_cast_f32_to_bf16", test_cast_for_types<nk::f32_t, nk::bf16_t>);
    check("tensor_cast_bf16_to_f32", test_cast_for_types<nk::bf16_t, nk::f32_t>);
    check("tensor_cast_f32_to_e4m3", test_cast_for_types<nk::f32_t, nk::e4m3_t>);
    check("tensor_cast_e4m3_to_f32", test_cast_for_types<nk::e4m3_t, nk::f32_t>);
    check("tensor_cast_i8_to_i32", test_cast_for_types<nk::i8_t, nk::i32_t>);
    check("tensor_cast_f64_to_f32", test_cast_for_types<nk::f64_t, nk::f32_t>);

    check("tensor_typed_pointer_ctors", test_typed_pointer_ctors);
    check("tensor_operator_bool", test_operator_bool);
    check("tensor_flatten_out_rank", test_flatten_out_rank);
    check("tensor_resize_capacity", test_resize_capacity);

    check("tensor_attention_bf16", test_tensor_attention_for_type<nk::bf16_t>);
    check("tensor_attention_e4m3", test_tensor_attention_for_type<nk::e4m3_t>);
    check("tensor_attention_i8", test_tensor_attention_for_type<nk::i8_t>);
}

} // namespace ashvardanian::numkong::test
