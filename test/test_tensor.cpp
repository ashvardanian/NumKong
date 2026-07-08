/**
 *  @brief C++ vector type instantiation tests.
 *  @file test/test_tensor.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 */
#include <array>
#include <cassert>
#include <complex>
#include <limits>

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

void test_block_scaled_composites() {
    // NVFP4: 9 bytes per block × 7 blocks for 100 logical dims.
    auto nvfp4_vec = make_vector<nk::nvfp4_t>(100);
    assert(nvfp4_vec.size() == 100 && "nvfp4_t size mismatch");
    assert(nvfp4_vec.size_values() == 7 && "nvfp4_t size_values mismatch (⌈100/16⌉)");
    assert(nvfp4_vec.size_bytes() == 63 && "nvfp4_t size_bytes mismatch (7 × 9)");

    // MXFP4: 17 bytes per block × 4 blocks for 100 logical dims.
    auto mxfp4_vec = make_vector<nk::mxfp4_t>(100);
    assert(mxfp4_vec.size() == 100 && "mxfp4_t size mismatch");
    assert(mxfp4_vec.size_values() == 4 && "mxfp4_t size_values mismatch (⌈100/32⌉)");
    assert(mxfp4_vec.size_bytes() == 68 && "mxfp4_t size_bytes mismatch (4 × 17)");

    // MXFP8 E4M3: 33 bytes per block × 4 blocks for 100 logical dims.
    auto mxfp8_vec = make_vector<nk::mxfp8_e4m3_t>(100);
    assert(mxfp8_vec.size() == 100 && "mxfp8_e4m3_t size mismatch");
    assert(mxfp8_vec.size_values() == 4 && "mxfp8_e4m3_t size_values mismatch");
    assert(mxfp8_vec.size_bytes() == 132 && "mxfp8_e4m3_t size_bytes mismatch (4 × 33)");

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
    assert(max_error <= 1.67f && "nvfp4_t round-trip error exceeds quantisation bound");
}

/** @brief Detects whether a `scaled_tensor`-like type exposes the per-tensor `tensor_scale()` accessor. */
template <typename scaled_type_>
concept exposes_tensor_scale_ = requires(scaled_type_ const &t) { t.tensor_scale(); };

/**
 *  @brief End-to-end test of the `scaled_tensor` family: encode via `cast`, inspect the SoA
 *  components, slice rows / block-aligned column tiles, materialize back to dense, and verify the
 *  per-tensor scale is exposed for NVFP4 but compile-time absent for the MX family. Every numeric
 *  path is checked byte-for-byte against `nk_cast_block_scaled_serial`.
 */
void test_scaled_tensor() {
    using nk::f32_t;
    using nk::u8_t;
    auto abs_diff = [](float a, float b) { return a > b ? a - b : b - a; };

    // 4 rows × 64 cols. 64 is a multiple of both the NVFP4 (16) and MX (32) block sizes.
    constexpr std::size_t rows = 4, cols = 64;
    auto weights = nk::tensor<f32_t>::try_empty({rows, cols});
    assert(!weights.empty() && "weights allocation failed");
    {
        auto writable = weights.span();
        for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t c = 0; c < cols; ++c)
                writable(r, c) = f32_t(static_cast<float>(static_cast<int>((r * 7 + c * 3) % 17) - 8) * 0.6f);
    }
    auto const *weights_raw = reinterpret_cast<float const *>(weights.data());

    // Encode (quantize) to NVFP4 into a preallocated scaled_tensor.
    auto quantized = nk::scaled_tensor<nk::nvfp4_t>::try_empty({rows, cols});
    nk::cast(weights.view(), quantized.span());
    assert(!quantized.empty() && "encode produced an empty scaled_tensor");
    assert(quantized.rank() == 2 && quantized.extent(0) == rows && quantized.extent(1) == cols);
    // elements() keeps the logical shape; block_scales() divides the last axis by block_size (16).
    assert(quantized.elements().extent(0) == rows && quantized.elements().extent(1) == cols);
    assert(quantized.block_scales().extent(0) == rows && quantized.block_scales().extent(1) == cols / 16);

    // Byte-identical to the serial C reference.
    nk_block_scaled_format_t const nvfp4_format = nk_nvfp4();
    nk_block_scaled_format_t const f32_format = nk_plain(nk_f32_k);
    auto reference_elements = make_vector<u8_t>(nk_block_scaled_elements_size(rows * cols, nvfp4_format));
    auto reference_scales = make_vector<u8_t>(nk_block_scaled_scales_size(rows * cols, nvfp4_format));
    nk_scalar_buffer_t reference_tensor_scale = {};    // zero → derive, matching the C++ factory
    nk_cast_block_scaled_serial(                       //
        weights.data(), nullptr, nullptr, &f32_format, //
        reference_elements.raw_values_data(), reference_scales.raw_values_data(), //
        &reference_tensor_scale, &nvfp4_format, rows * cols);

    auto const *encoded_elements = reinterpret_cast<unsigned char const *>(quantized.elements().byte_data());
    for (std::size_t i = 0; i < reference_elements.size_values(); ++i)
        assert(encoded_elements[i] == reference_elements.raw_values_data()[i] &&
               "NVFP4 elements differ from reference");
    auto const *encoded_scales = reinterpret_cast<unsigned char const *>(quantized.block_scales().byte_data());
    for (std::size_t i = 0; i < reference_scales.size_values(); ++i)
        assert(encoded_scales[i] == reference_scales.raw_values_data()[i] && "NVFP4 scales differ from reference");
    assert(quantized.tensor_scale() == reference_tensor_scale.f32 && "derived tensor_scale differs from reference");

    std::size_t const row_element_bytes = nk_block_scaled_elements_size(cols, nvfp4_format); // 32
    std::size_t const row_scale_bytes = nk_block_scaled_scales_size(cols, nvfp4_format);     // 4

    // Slice one row and materialize it to a dense f32 vector.
    auto restored_row = make_vector<f32_t>(cols);
    nk::cast<nk::nvfp4_t>(quantized.row(1), restored_row.span());
    {
        auto reference_row = make_vector<f32_t>(cols);
        nk_scalar_buffer_t tensor_scale;
        tensor_scale.f32 = quantized.tensor_scale();
        nk_cast_block_scaled_serial(                                      //
            reference_elements.raw_values_data() + 1 * row_element_bytes, //
            reference_scales.raw_values_data() + 1 * row_scale_bytes,     //
            &tensor_scale, &nvfp4_format,                                 //
            reference_row.raw_values_data(), nullptr, nullptr, &f32_format, cols);
        for (std::size_t c = 0; c < cols; ++c)
            assert(restored_row.raw_values_data()[c] == reference_row.raw_values_data()[c] &&
                   "row materialization differs from reference");
    }

    // Block-aligned column tile of the first two NVFP4 blocks, then materialize it.
    auto column_tile = quantized.columns(0, 32);
    assert(column_tile.extent(0) == rows && column_tile.extent(1) == 32);
    assert(column_tile.block_scales().extent(1) == 32 / 16);
    // A sub-block (non-aligned) range is rejected, not silently truncated.
    assert(quantized.columns(0, 24).empty() && "sub-block column ranges must be rejected");
    auto restored_tile = nk::tensor<f32_t>::try_empty({rows, std::size_t {32}});
    nk::cast<nk::nvfp4_t>(column_tile, restored_tile.span());
    {
        auto const *tile_raw = reinterpret_cast<float const *>(restored_tile.data());
        auto reference_tile_row = make_vector<f32_t>(32);
        for (std::size_t r = 0; r < rows; ++r) {
            nk_scalar_buffer_t tensor_scale;
            tensor_scale.f32 = quantized.tensor_scale();
            nk_cast_block_scaled_serial(                                      //
                reference_elements.raw_values_data() + r * row_element_bytes, //
                reference_scales.raw_values_data() + r * row_scale_bytes,     //
                &tensor_scale, &nvfp4_format,                                 //
                reference_tile_row.raw_values_data(), nullptr, nullptr, &f32_format, 32);
            for (std::size_t c = 0; c < 32; ++c)
                assert(tile_raw[r * 32 + c] == reference_tile_row.raw_values_data()[c] &&
                       "column-tile materialization differs from reference");
        }
    }

    // Iterate leading-axis rows.
    std::size_t iterated_rows = 0;
    for (nk::scaled_tensor_view<nk::nvfp4_t> row_view : quantized.rows_views()) {
        assert(row_view.rank() == 1 && row_view.extent(0) == cols);
        assert(row_view.block_scales().extent(0) == cols / 16);
        ++iterated_rows;
    }
    assert(iterated_rows == rows && "rows_views() did not visit every row");

    // The per-tensor scale exists for NVFP4 and is compile-time absent for the MX family.
    static_assert(exposes_tensor_scale_<nk::scaled_tensor<nk::nvfp4_t>>, "NVFP4 must expose tensor_scale()");
    static_assert(!exposes_tensor_scale_<nk::scaled_tensor<nk::mxfp8_e4m3_t>>,
                  "MX formats must not expose tensor_scale()");

    // Block-aligned column tile at a non-zero start exercises the element and scale byte offsets.
    {
        auto mid_tile = quantized.columns(16, 48); // two NVFP4 blocks starting at column 16
        assert(mid_tile.extent(1) == 32 && mid_tile.block_scales().extent(1) == 32 / 16);
        auto restored_mid = nk::tensor<f32_t>::try_empty({rows, std::size_t {32}});
        nk::cast<nk::nvfp4_t>(mid_tile, restored_mid.span());
        auto const *mid_raw = reinterpret_cast<float const *>(restored_mid.data());
        auto reference_mid = make_vector<f32_t>(32);
        for (std::size_t r = 0; r < rows; ++r) {
            nk_scalar_buffer_t tensor_scale;
            tensor_scale.f32 = quantized.tensor_scale();
            nk_cast_block_scaled_serial(                                               //
                reference_elements.raw_values_data() + r * row_element_bytes + 16 / 2, // column 16 → byte 8
                reference_scales.raw_values_data() + r * row_scale_bytes + 16 / 16,    // block 1
                &tensor_scale, &nvfp4_format,                                          //
                reference_mid.raw_values_data(), nullptr, nullptr, &f32_format, 32);
            for (std::size_t c = 0; c < 32; ++c)
                assert(mid_raw[r * 32 + c] == reference_mid.raw_values_data()[c] &&
                       "non-zero-start column tile differs from reference");
        }
    }

    // MXFP8 whole-tensor decode equals the serial reference byte-for-byte.
    auto mx = nk::scaled_tensor<nk::mxfp8_e4m3_t>::try_empty({rows, cols});
    nk::cast(weights.view(), mx.span());
    assert(!mx.empty() && mx.block_scales().extent(0) == rows && mx.block_scales().extent(1) == cols / 32);
    {
        auto mx_restored = nk::tensor<f32_t>::try_empty({rows, cols});
        nk::cast<nk::mxfp8_e4m3_t>(mx.view(), mx_restored.span());
        // Reference: decode the same bytes through the serial kernel and require bit-identical output.
        nk_block_scaled_format_t const mx_format = nk_mxfp8_e4m3();
        auto reference_restored = make_vector<f32_t>(rows * cols);
        nk_cast_block_scaled_serial( //
            mx.elements().byte_data(), mx.block_scales().byte_data(), nullptr, &mx_format,
            reference_restored.raw_values_data(), nullptr, nullptr, &f32_format, rows * cols);
        auto const *restored_raw = reinterpret_cast<float const *>(mx_restored.data());
        for (std::size_t i = 0; i < rows * cols; ++i)
            assert(restored_raw[i] == reference_restored.raw_values_data()[i] &&
                   "MXFP8 whole-tensor decode differs from serial reference");
    }

    // Transcode MXFP8 E4M3 to NVFP4 (block-scaled to block-scaled).
    {
        auto transcoded = nk::scaled_tensor<nk::nvfp4_t>::try_empty({rows, cols});
        assert(!transcoded.empty());
        auto destination = transcoded.span();
        nk::cast(mx.view(), destination);
        // The transcode result must match decoding MXFP8→dense then re-encoding to NVFP4.
        auto dense = nk::tensor<f32_t>::try_empty({rows, cols});
        nk::cast<nk::mxfp8_e4m3_t>(mx.view(), dense.span());
        auto reference_nvfp4 = nk::scaled_tensor<nk::nvfp4_t>::try_empty({rows, cols});
        nk::cast(dense.view(), reference_nvfp4.span());
        auto const *transcoded_elements = reinterpret_cast<unsigned char const *>(transcoded.elements().byte_data());
        auto const *reference_nvfp4_elements = reinterpret_cast<unsigned char const *>(
            reference_nvfp4.elements().byte_data());
        std::size_t transcoded_byte_count = nk_block_scaled_elements_size(rows * cols, nvfp4_format);
        for (std::size_t i = 0; i < transcoded_byte_count; ++i)
            assert(transcoded_elements[i] == reference_nvfp4_elements[i] &&
                   "transcode elements differ from decode-then-encode");
    }
}

/**
 *  @brief Per-format bidirectional round-trip checks (one block) covering three regimes:
 *   - B exactly-representable: a block of powers of two (amax = 2 is a power of two, no scale clip)
 *     must round-trip bit-exactly through UE8M0 formats; NVFP4's two-level f32×UE4M3 scale only
 *     reaches it within the element resolution, so that case asserts the same relative bound as C.
 *   - C narrow range [1, 1.5): every value's mantissa is below each element format's max mantissa, so
 *     nothing clips and the relative error is bounded by the element resolution `narrow_relative_bound`.
 *   - D idempotence: re-quantizing an already-quantized block is a fixed point (bit-stable).
 */
template <typename format_>
void test_scaled_roundtrip(float narrow_relative_bound) {
    using nk::f32_t;
    constexpr std::size_t block = format_::elements();
    auto abs_diff = [](float a, float b) { return a > b ? a - b : b - a; };

    auto encode_decode = [](float const *input_values) {
        auto input = nk::tensor<f32_t>::try_empty({std::size_t {1}, block});
        auto writable = input.span();
        for (std::size_t i = 0; i < block; ++i) writable(0, i) = f32_t(input_values[i]);
        auto quantized = nk::scaled_tensor<format_>::try_empty({std::size_t {1}, block});
        nk::cast(input.view(), quantized.span());
        auto restored = make_vector<f32_t>(block);
        nk::cast<format_>(quantized.row(0), restored.span());
        return restored;
    };

    // B — powers of two: amax = 2.0 is a power of two so the scale is exact and nothing clips.
    {
        float values[32];
        for (std::size_t i = 0; i < block; ++i) values[i] = (i % 2 == 0) ? 2.0f : 1.0f;
        auto restored = encode_decode(values);
        for (std::size_t i = 0; i < block; ++i) {
            if constexpr (format_::has_tensor_scale())
                assert(abs_diff(restored.raw_values_data()[i], values[i]) <= narrow_relative_bound * values[i] &&
                       "NVFP4 power-of-two round-trip outside element resolution");
            else assert(restored.raw_values_data()[i] == values[i] && "UE8M0 power-of-two round-trip is not bit-exact");
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
        assert(max_relative_error <= narrow_relative_bound && "narrow-range round-trip exceeds element resolution");
    }
    // D — idempotence: a second round-trip reproduces the first bit-for-bit.
    {
        float values[32];
        for (std::size_t i = 0; i < block; ++i)
            values[i] = static_cast<float>(static_cast<int>((i * 5) % 19) - 9) * 0.3f;
        auto first = encode_decode(values);
        auto second = encode_decode(first.raw_values_data());
        for (std::size_t i = 0; i < block; ++i)
            assert(second.raw_values_data()[i] == first.raw_values_data()[i] && "round-trip is not idempotent");
    }
}

/** @brief Degenerate-input handling: all-zero blocks decode to zero; a NaN poisons only its block. */
void test_scaled_tensor_degenerate() {
    using nk::f32_t;
    auto abs_diff = [](float a, float b) { return a > b ? a - b : b - a; };
    constexpr std::size_t block = 32; // MXFP8 block size
    // All-zero block → zero scale → all-zero decode (no division-by-zero, no NaN).
    {
        auto input = nk::tensor<f32_t>::try_zeros({std::size_t {1}, block});
        auto quantized = nk::scaled_tensor<nk::mxfp8_e4m3_t>::try_empty({std::size_t {1}, block});
        nk::cast(input.view(), quantized.span());
        auto restored = make_vector<f32_t>(block);
        nk::cast<nk::mxfp8_e4m3_t>(quantized.row(0), restored.span());
        for (std::size_t i = 0; i < block; ++i)
            assert(restored.raw_values_data()[i] == 0.0f && "all-zero block did not decode to zero");
    }
    // A NaN in one block sets that block's scale to the NaN sentinel; a clean block is unaffected.
    {
        auto input = nk::tensor<f32_t>::try_empty({std::size_t {2}, block});
        auto writable = input.span();
        float const quiet_nan = std::numeric_limits<float>::quiet_NaN();
        for (std::size_t i = 0; i < block; ++i) {
            writable(0, i) = f32_t(i == 3 ? quiet_nan : 1.5f); // row 0 poisoned
            writable(1, i) = f32_t(1.5f);                      // row 1 clean
        }
        auto quantized = nk::scaled_tensor<nk::mxfp8_e4m3_t>::try_empty({std::size_t {2}, block});
        nk::cast(input.view(), quantized.span());
        auto restored = nk::tensor<f32_t>::try_empty({std::size_t {2}, block});
        nk::cast<nk::mxfp8_e4m3_t>(quantized.view(), restored.span());
        auto const *clean_row = reinterpret_cast<float const *>(restored.data()) + block;
        for (std::size_t i = 0; i < block; ++i)
            assert(abs_diff(clean_row[i], 1.5f) <= 0.1f && "clean block corrupted by a NaN in another block");
    }
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

    test_block_scaled_composites();
    std::printf("  block-scaled composites:      OK\n");

    test_scaled_tensor();
    std::printf("  scaled_tensor encode/slice:   OK\n");

    // Per-element resolution = 2^-mantissa_bits (E2M1:1, E3M2/E5M2:2, E2M3/E4M3:3 mantissa bits);
    // MXINT8 resolves to ~1/64 over the narrow band.
    test_scaled_roundtrip<nk::nvfp4_t>(0.5f);
    test_scaled_roundtrip<nk::mxfp4_t>(0.5f);
    test_scaled_roundtrip<nk::mxfp6_e2m3_t>(0.125f);
    test_scaled_roundtrip<nk::mxfp6_e3m2_t>(0.25f);
    test_scaled_roundtrip<nk::mxfp8_e4m3_t>(0.125f);
    test_scaled_roundtrip<nk::mxfp8_e5m2_t>(0.25f);
    test_scaled_roundtrip<nk::mxint8_t>(0.05f);
    test_scaled_tensor_degenerate();
    std::printf("  scaled_tensor round-trips:    OK\n");

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

/** @brief Typed-pointer convenience ctors (count / initializer_list / std::array) + out-of-range guard. */
void test_typed_pointer_ctors() {
    alignas(64) float buf[64];
    for (int i = 0; i < 64; ++i) buf[i] = static_cast<float>(i);
    auto *p = reinterpret_cast<nk::f32_t *>(buf);

    nk::tensor_view<nk::f32_t> v1(p, 6); // rank-1 (ptr, count)
    assert(v1.rank() == 1 && v1.numel() == 6 && "typed-ptr view (count) ctor");
    nk::tensor_span<nk::f32_t> s1(p, 6);
    assert(s1.rank() == 1 && s1.numel() == 6 && "typed-ptr span (count) ctor");

    nk::tensor_view<nk::f32_t> v2(p, {2, 3}); // (ptr, initializer_list)
    assert(v2.rank() == 2 && v2.numel() == 6 && "typed-ptr view (list) ctor");

    std::array<std::size_t, 2> ext {3, 4}; // (ptr, std::array)
    nk::tensor_view<nk::f32_t> v3(p, ext);
    assert(v3.rank() == 2 && v3.numel() == 12 && "typed-ptr view (array) ctor");

    // An out-of-range rank (too many, or empty) fails closed to an empty handle — like reshape() —
    // rather than overflowing the fixed-capacity shape storage.
    nk::tensor_view<nk::f32_t, 2> over(p, {2, 3, 4});
    assert(over.empty() && "over-rank list -> empty handle");
    nk::tensor_span<nk::f32_t, 4> empty_list(p, std::initializer_list<std::size_t> {});
    assert(empty_list.empty() && "empty extents list -> empty handle");
    std::printf("  typed-pointer ctors + guard:  OK\n");
}

/** @brief `explicit operator bool` on every owning + non-owning handle type. */
void test_operator_bool() {
    auto t = nk::tensor<nk::f32_t>::try_empty({2, 3});
    nk::tensor<nk::f32_t> te {};
    assert(static_cast<bool>(t) && !te && "tensor bool");
    assert(static_cast<bool>(t.view()) && !te.view() && "tensor_view bool");
    assert(static_cast<bool>(t.span()) && "tensor_span bool");

    auto v = make_vector<nk::f32_t>(4);
    nk::vector<nk::f32_t> ve {};
    assert(static_cast<bool>(v) && !ve && "vector bool");
    assert(static_cast<bool>(v.view()) && static_cast<bool>(v.span()) && "vector view/span bool");

    auto q = nk::scaled_tensor<nk::nvfp4_t>::try_empty({2, 32});
    nk::scaled_tensor<nk::nvfp4_t> qe {};
    assert(static_cast<bool>(q) && !qe && "scaled_tensor bool");
    assert(static_cast<bool>(q.view()) && static_cast<bool>(q.span()) && "scaled view/span bool");
    std::printf("  operator bool (8 handles):    OK\n");
}

/** @brief Templated `flatten<out_rank_>()` with an explicit non-default output rank. */
void test_flatten_out_rank() {
    auto t = nk::tensor<nk::f32_t>::try_empty({2, 3, 4});
    auto f1 = t.flatten<1>();
    assert(f1.rank() == 1 && f1.numel() == 24 && "tensor flatten<1>");
    auto fv = t.view().flatten<1>();
    assert(fv.rank() == 1 && fv.numel() == 24 && "view flatten<1>");
    auto fs = t.span().flatten<2>();
    assert(fs.numel() == 24 && "span flatten<2>");
    std::printf("  flatten<out_rank_>:           OK\n");
}

/** @brief Fixed-capacity resize contract: data()-stability, beyond-capacity fail, reserve/clear/move. */
void test_resize_capacity() {
    auto t = nk::tensor<nk::f32_t>::try_empty({8, 4}); // capacity 32
    assert(t.capacity() == 32 && "capacity from initial shape");
    auto *p0 = t.data();
    assert(t.try_resize({2, 4}) && t.numel() == 8 && t.data() == p0 && "resize within capacity keeps data()");
    assert(!t.try_resize({100, 100}) && t.numel() == 8 && "resize beyond capacity fails, shape kept");
    std::size_t ext[2] = {4, 4};
    assert(t.try_resize(ext, 2) && t.numel() == 16 && "resize (ptr,rank) overload");
    assert(t.reserve(64) && t.capacity() >= 64 && "reserve grows capacity");
    assert(t.try_resize({8, 8}) && "resize into grown capacity");
    t.clear();
    assert(t.empty() && t.capacity() >= 64 && "clear -> empty, capacity kept");

    auto t2 = nk::tensor<nk::f32_t>::try_empty({4, 4});
    auto cap = t2.capacity();
    nk::tensor<nk::f32_t> t3 = std::move(t2);
    assert(t3.capacity() == cap && "move preserves capacity");

    auto q = nk::scaled_tensor<nk::nvfp4_t>::try_empty({2, 32}); // block_size 16
    assert(q.try_resize({2, 16}) && q.extent(1) == 16 && "scaled coordinated resize");
    assert(!q.try_resize({2, 20}) && "scaled resize rejects non-block-aligned last extent");
    assert(!q.try_resize({8, 64}) && "scaled resize beyond capacity fails");
    assert(q.reserve(256) && q.try_resize({4, 64}) && "scaled reserve then resize");

    auto v = nk::vector<nk::f32_t>::try_empty(16);
    assert(v.capacity() == 16 && "vector capacity");
    auto *vp0 = v.values_data();
    assert(v.try_resize(8) && v.size() == 8 && v.values_data() == vp0 && "vector resize keeps data()");
    assert(!v.try_resize(100) && "vector resize beyond capacity fails");
    assert(v.reserve(64) && v.try_resize(50) && "vector reserve then resize");
    v.clear();
    assert(v.empty() && v.capacity() >= 64 && "vector clear");
    std::printf("  resize/capacity/reserve:      OK\n");
}

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

    // (Trig wrappers moved to test_each.cpp — they are elementwise operations.)

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

    // (Vector-level reduction wrappers moved to test_reduce.cpp.)

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

    test_typed_pointer_ctors();
    test_operator_bool();
    test_flatten_out_rank();
    test_resize_capacity();
}
