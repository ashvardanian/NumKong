/**
 *  @file test/cross.cuh
 *  @author Ash Vardanian
 *  @date January 14, 2025
 *  @brief Backend-neutral cross-kernel scenarios: batched dots, spatial and set distances, and
 *      ragged attention windows.
 *
 *  Every scenario is a template over the scalar type, its kernels, and a backend owning where
 *  kernel operands live, how a kernel is called, when its results become readable, and how its
 *  products accumulate: @c host_backend_t by default, @c cuda_backend_t from `test.cuh` for the
 *  CUDA suite. Set distances run on the host only. References always run the serial `nk::`
 *  templates on the host. Outputs start filled with @c canary_k bytes, so a stray write shows.
 */
#pragma once
#ifndef NK_TEST_CROSS_CUH
#define NK_TEST_CROSS_CUH

#include <cmath>   // `std::ldexp`, `std::nextafter`
#include <cstdint> // `std::int64_t`, `std::uint64_t`
#include <cstring> // `std::memset`, `std::memcmp`

#include <bit>              // `std::bit_cast`, `std::countr_zero`
#include <initializer_list> // `std::initializer_list`
#include <random>           // `std::uniform_real_distribution`
#include <vector>           // `std::vector`

#include "numkong/attention.hpp" // `nk::attention_bidirectional_packed`, `nk::attention_pack`
#include "numkong/cast.h"        // `nk_cast_serial`
#include "numkong/dots.hpp"      // `nk::dots_packed`, `nk::dots_symmetric`
#include "numkong/matrix.hpp"    // `nk::dots_pack_size`, `nk::dots_pack`
#include "numkong/reduce.hpp"    // `nk::reduce_moments`
#include "numkong/spatials.h"    // `nk_angulars_packed_*`, `nk_euclideans_packed_*`

#include "test.hpp"

namespace ashvardanian::numkong::test {

#pragma region Backend Policy

/** Significant bits of the softmax weights as P · V reads them, which floors how close an attention
 *  output lands. */
enum class attention_weights_t : unsigned {

    /** Judged by the scale threshold alone. */
    unquantized_k = 0,

    /** E4M3 weights. */
    bits_4_k = 4,

    /** BF16 or U8 weights. */
    bits_8_k = 8,

    /** F16 weights. */
    bits_11_k = 11,
};

/** Waits for @p backend, failing @p stats with the name of its first failed call. */
template <typename backend_type_>
void synchronize(backend_type_ &backend, error_stats_t &stats) noexcept {
    if (char const *failure = backend.synchronize()) stats.expect(false, failure);
}

#pragma endregion Backend Policy

#pragma region Tolerances

/** The family a dot product is judged in: ULP thresholds, bit for bit, or a bound of its own. */
constexpr comparison_family_t dots_family(accumulation_t accumulation) noexcept {
    return accumulation == accumulation_t::family_thresholds_k ? comparison_family_t::approximate_k
           : accumulation == accumulation_t::exact_k           ? comparison_family_t::exact_k
                                                               : comparison_family_t::bounded_k;
}

/** The family a distance is judged in: the ULP thresholds, or a tolerance around the reference. */
constexpr comparison_family_t spatials_family(accumulation_t accumulation) noexcept {
    return accumulation == accumulation_t::family_thresholds_k ? comparison_family_t::approximate_k
                                                               : comparison_family_t::bounded_k;
}

/** The family an attention output is judged in: the scale threshold, or that threshold floored by
 *  the weights. */
constexpr comparison_family_t attention_family(attention_weights_t weights) noexcept {
    return weights == attention_weights_t::unquantized_k ? comparison_family_t::normalized_reduction_k
                                                         : comparison_family_t::bounded_k;
}

/** The exponent of the lowest set bit of @p value's significand: @p value is an odd multiple of
 *  that power of two. */
inline int grid_exponent(double value) noexcept {
    std::uint64_t const bits = std::bit_cast<std::uint64_t>(value);
    int const biased_exponent = static_cast<int>((bits >> 52) & 0x7FF);
    std::uint64_t const significand = (bits & ((1ull << 52) - 1)) | (biased_exponent ? 1ull << 52 : 0);
    return std::max(biased_exponent, 1) - 1075 + std::countr_zero(significand);
}

/** How far a dot product of @p depth terms may land from @p reference under @p accumulation, given
 *  Σ|a · b| as @p magnitude and the grid every product lies on as @p grid. A sum in p bits of terms
 *  on one grid is exact while Σ|a · b| stays below 2ᵖ grid steps, so F32 and F64 accumulations must
 *  then match exactly. */
inline double dot_bound(accumulation_t accumulation, double reference, double magnitude, int grid,
                        std::size_t depth) noexcept {
    double const steps = static_cast<double>(depth);
    switch (accumulation) {
    case accumulation_t::dot2_k: {
        double const gamma = steps * 0x1p-53, absolute = std::fabs(reference);
        return 2 * (std::nextafter(absolute, INFINITY) - absolute) + 4 * gamma * gamma * magnitude;
    }
    case accumulation_t::f64_k: return magnitude < std::ldexp(1.0, 53 + grid) ? 0 : (steps + 1) * 0x1p-53 * magnitude;
    case accumulation_t::f32_k: return magnitude < std::ldexp(1.0, 24 + grid) ? 0 : (steps + 1) * 0x1p-24 * magnitude;
    case accumulation_t::tensor_core_k: return (steps / 32 + 1) * 0x1p-22 * magnitude + 1e-30;
    default: return 0;
    }
}

/** A distance's tolerance over @p depth terms: absolute for angular, relative to ‖a‖² + ‖b‖² for
 *  squared euclidean. */
inline double spatial_tolerance(accumulation_t accumulation, std::size_t depth) noexcept {
    double const steps = static_cast<double>(depth);
    switch (accumulation) {
    case accumulation_t::dot2_k:
    case accumulation_t::f64_k: return 0x1p-40;
    case accumulation_t::exact_k: return 0x1p-16;
    case accumulation_t::tensor_core_k: return 4 * ((steps / 32 + 1) * 0x1p-22 + steps * 0x1p-24) + 0x1p-16;
    default: return 0;
    }
}

/** Decodes @p rows rows of @p depth dimensions, @p row_stride_values values apart, into F64. */
template <typename scalar_type_, typename allocator_type_>
std::vector<double> decode_rows(nk::vector<scalar_type_, allocator_type_> const &matrix, std::size_t rows,
                                std::size_t depth, std::size_t row_stride_values) {
    std::vector<double> decoded(rows * depth);
    for (std::size_t row = 0; row < rows; row++)
        nk_cast_serial(matrix.raw_values_data() + row * row_stride_values, scalar_type_::dtype(), depth,
                       decoded.data() + row * depth, nk_f64_k);
    return decoded;
}

/** Decodes @p rows rows of @p matrix when @p accumulation_ judges dots by a bound, and nothing
 *  otherwise. */
template <accumulation_t accumulation_, typename vector_type_>
std::vector<double> decode_for(vector_type_ const &matrix, std::size_t rows, std::size_t depth,
                               std::size_t row_stride_values) {
    if constexpr (dots_family(accumulation_) != comparison_family_t::bounded_k) return {};
    else return decode_rows(matrix, rows, depth, row_stride_values);
}

/** Folds one dot product into @p stats, bounded by the terms of the decoded rows when @p
 *  accumulation_ needs it. */
template <accumulation_t accumulation_, typename result_type_, typename reference_type_>
void accumulate_dot(error_stats_t &stats, result_type_ result, reference_type_ reference,
                    std::vector<double> const &first_rows, std::size_t first_row,
                    std::vector<double> const &second_rows, std::size_t second_row, std::size_t depth) {
    if constexpr (dots_family(accumulation_) != comparison_family_t::bounded_k) stats.accumulate(result, reference);
    else {
        double const *first = first_rows.data() + first_row * depth, *second = second_rows.data() + second_row * depth;
        double magnitude = 0;
        int grid = std::numeric_limits<double>::max_exponent;
        for (std::size_t index = 0; index < depth; index++) {
            double const product = first[index] * second[index];
            magnitude += std::fabs(product);
            if (product != 0) grid = std::min(grid, grid_exponent(product));
        }
        double const expected = static_cast<double>(reference);
        stats.accumulate_bounded(result, expected, dot_bound(accumulation_, expected, magnitude, grid, depth));
    }
}

/** Folds one angular distance into @p stats, within the tolerance of @p accumulation_ when it has
 *  one. */
template <accumulation_t accumulation_, typename result_type_, typename reference_type_>
void accumulate_angular(error_stats_t &stats, result_type_ result, reference_type_ reference, std::size_t depth) {
    if constexpr (accumulation_ == accumulation_t::family_thresholds_k) stats.accumulate(result, reference);
    else stats.accumulate_bounded(result, static_cast<double>(reference), spatial_tolerance(accumulation_, depth));
}

/** Folds one euclidean distance into @p stats; a tolerance bounds its square, relative to both
 *  squared norms. */
template <accumulation_t accumulation_, typename result_type_, typename reference_type_>
void accumulate_euclidean(error_stats_t &stats, result_type_ result, reference_type_ reference,
                          reference_type_ squared_norms, std::size_t depth) {
    if constexpr (accumulation_ == accumulation_t::family_thresholds_k) stats.accumulate(result, reference);
    else {
        double const expected = static_cast<double>(reference), computed = static_cast<double>(result);
        double const scale = static_cast<double>(squared_norms) > 0 ? static_cast<double>(squared_norms) : 1;
        double const sum = std::max(computed + expected, std::numeric_limits<double>::min());
        stats.accumulate_bounded(result, expected, spatial_tolerance(accumulation_, depth) * scale / sum);
    }
}

/** Folds every attention output into @p stats: the scale threshold of the largest reference,
 *  floored by @p weights_. */
template <attention_weights_t weights_, typename output_vector_, typename reference_vector_, typename values_vector_>
void accumulate_attention(error_stats_t &stats, output_vector_ const &output, reference_vector_ const &reference,
                          values_vector_ const &values) {
    if constexpr (weights_ == attention_weights_t::unquantized_k) {
        for (std::size_t index = 0; index < output.size_values(); index++)
            stats.accumulate(output[index], reference[index]);
    }
    else {
        double largest_value = 0, largest_reference = 0;
        for (double value : decode_rows(values, 1, values.size(), 0))
            largest_value = std::max(largest_value, std::fabs(value));
        for (std::size_t index = 0; index < reference.size_values(); index++)
            largest_reference = std::max(largest_reference, std::fabs(static_cast<double>(reference[index])));
        double const bound = std::max(global_config.scale_threshold * largest_reference,
                                      std::ldexp(largest_value, -static_cast<int>(weights_) - 1));
        for (std::size_t index = 0; index < output.size_values(); index++)
            stats.accumulate_bounded(output[index], static_cast<double>(reference[index]), bound);
    }
}

#pragma endregion Tolerances

#pragma region Canaries

/** The byte outputs start as and stride padding holds: NaN in every 8- and 16-bit float, so a read
 *  past a row shows. */
constexpr unsigned char canary_k = 0xFF;

/** Fills every byte of @p vector with @c canary_k. */
template <typename vector_type_>
void fill_canary(vector_type_ &vector) noexcept {
    std::memset(vector.raw_values_data(), canary_k, vector.size_bytes());
}

/** Fills the bytes past @p row_bytes of each of @p rows rows, @p stride bytes apart, with
 *  @c canary_k. */
template <typename vector_type_>
void fill_padding_canary(vector_type_ &matrix, std::size_t rows, std::size_t row_bytes, std::size_t stride) noexcept {
    auto *bytes = reinterpret_cast<unsigned char *>(matrix.raw_values_data());
    for (std::size_t row = 0; row < rows; row++)
        std::memset(bytes + row * stride + row_bytes, canary_k, stride - row_bytes);
}

/** Count of bytes from @p begin up to @p end of @p vector whose value differs from @c canary_k. */
template <typename vector_type_>
std::size_t overwritten_bytes(vector_type_ const &vector, std::size_t begin, std::size_t end) noexcept {
    auto const *bytes = reinterpret_cast<unsigned char const *>(vector.raw_values_data());
    std::size_t overwritten = 0;
    for (std::size_t byte = begin; byte < end; byte++) overwritten += bytes[byte] != canary_k;
    return overwritten;
}

/** Expects the stride padding past @p row_bytes of each of @p rows output rows to still hold
 *  @c canary_k. */
template <typename vector_type_>
void expect_padding_untouched(error_stats_t &stats, vector_type_ const &output, std::size_t rows, std::size_t row_bytes,
                              std::size_t stride) noexcept {
    std::size_t overwritten = 0;
    for (std::size_t row = 0; row < rows; row++)
        overwritten += overwritten_bytes(output, row * stride + row_bytes, (row + 1) * stride);
    stats.expect(overwritten == 0, "wrote into the output stride padding");
}

/** Expects a Gram matrix of @p count rows to hold @c canary_k below the diagonal and outside the
 *  rows from @p row_start up to @p row_end. */
template <typename result_type_, typename vector_type_>
void expect_symmetric_untouched(error_stats_t &stats, vector_type_ const &output, std::size_t count, std::size_t stride,
                                std::size_t row_start, std::size_t row_end) noexcept {
    std::size_t below_diagonal = 0, outside_rows = 0;
    for (std::size_t row = 0; row < count; row++) {
        bool const computed = row >= row_start && row < row_end;
        std::size_t const untouched_columns = computed ? row : count;
        (computed ? below_diagonal : outside_rows) += overwritten_bytes(
            output, row * stride, row * stride + untouched_columns * sizeof(result_type_));
    }
    stats.expect(below_diagonal == 0, "wrote below the diagonal");
    stats.expect(outside_rows == 0, "wrote rows outside [row_start, row_start + row_count)");
    expect_padding_untouched(stats, output, count, count * sizeof(result_type_), stride);
}

#pragma endregion Canaries

#pragma region Operands

/** A ragged batch: keys per segment and the exclusive prefix sums of key and query counts, all
 *  kernel-readable. */
template <typename backend_type_>
struct attention_segments {

    /** Counts in @p backend_type_ memory. */
    using counts_t = nk::vector<nk_u32_t, typename backend_type_::template allocator<nk_u32_t>>;

    /** Keys per segment. */
    counts_t lengths;

    /** First key row of every segment, then the key total. */
    counts_t key_offsets;

    /** First query row of every segment, then the query total. */
    counts_t query_offsets;

    /** Segments in the batch. */
    std::size_t count() const noexcept { return lengths.size(); }

    /** Key rows across every segment. */
    std::size_t key_tokens() const noexcept { return key_offsets.values_data()[count()]; }

    /** Query rows across every segment. */
    std::size_t query_tokens() const noexcept { return query_offsets.values_data()[count()]; }

    /** Query rows of @p segment. */
    std::size_t queries(std::size_t segment) const noexcept {
        return query_offsets.values_data()[segment + 1] - query_offsets.values_data()[segment];
    }
};

/** Lays out segments of @p lengths keys, matched with the entries of @p query_counts queries. */
template <typename backend_type_>
attention_segments<backend_type_> make_attention_segments(std::initializer_list<nk_u32_t> lengths,
                                                          std::initializer_list<nk_u32_t> query_counts) {
    using counts_t = typename attention_segments<backend_type_>::counts_t;
    attention_segments<backend_type_> segments {counts_t::try_zeros(lengths.size()),
                                                counts_t::try_zeros(lengths.size() + 1),
                                                counts_t::try_zeros(lengths.size() + 1)};
    std::size_t segment = 0;
    for (auto length = lengths.begin(), queries = query_counts.begin(); length != lengths.end();
         ++length, ++queries, ++segment) {
        segments.lengths.values_data()[segment] = *length;
        segments.key_offsets.values_data()[segment + 1] = segments.key_offsets.values_data()[segment] + *length;
        segments.query_offsets.values_data()[segment + 1] = segments.query_offsets.values_data()[segment] + *queries;
    }
    return segments;
}

/** Head counts, depth, and softmax scale of one attention call. */
struct attention_layout_t {

    /** Query heads per token. */
    std::size_t head_count;

    /** Key and value heads per token, each shared by a group of query heads. */
    std::size_t key_value_head_count;

    /** Elements per head. */
    std::size_t depth;

    /** Logit multiplier ahead of the softmax. */
    nk_f32_t scale;

    /** Elements in one token's query or output row. */
    std::size_t query_width() const noexcept { return head_count * depth; }

    /** Elements in one token's key or value row. */
    std::size_t key_value_width() const noexcept { return key_value_head_count * depth; }
};

/** Key and value heads of every attention case; each case's group sets how many query heads share
 *  one. */
constexpr std::size_t attention_key_value_heads_k = 2;

/** One bidirectional case: the query heads sharing each key-value head, and the head depth. */
struct attention_bidirectional_case_t {

    /** Query heads per key-value head. */
    std::size_t group;

    /** Elements per head. */
    std::size_t depth;
};

/** GQA 2:1 at depths on both sides of every panel edge, then GQA 1:1, 4:1 and 8:1 at one full
 *  panel. */
inline std::vector<attention_bidirectional_case_t> attention_bidirectional_cases() {
    std::vector<attention_bidirectional_case_t> cases;
    for (std::size_t depth : {1ul, 64ul, 65ul, 127ul, 128ul, 129ul, 255ul, 257ul}) cases.push_back({2, depth});
    for (std::size_t group : {1ul, 4ul, 8ul}) cases.push_back({group, 128});
    return cases;
}

/** One causal case: the long segment's key count, the GQA group, the head depth, and the mask it
 *  runs under. */
struct attention_causal_case_t {

    /** Keys of the long segment, which a pad and a 33-key decode segment follow. */
    nk_u32_t main_length;

    /** Query heads per key-value head. */
    std::size_t group;

    /** Elements per head. */
    std::size_t depth;

    /** Position of query row 0 among its segment's keys. */
    std::int64_t diagonal_offset;

    /** Keys a query row may see, ending at its own position. */
    std::size_t window;
};

/** Queries of the long causal segment: about half its keys, capped to keep the per-row reference
 *  cheap. */
inline nk_u32_t attention_causal_queries(nk_u32_t main_length) noexcept {
    return std::min<nk_u32_t>(main_length / 2 + 2, 24);
}

/** Panel edges and a 1000-key prefill in the main length, odd depths, cached and shifted diagonals,
 *  windows, and GQA. */
inline std::vector<attention_causal_case_t> attention_causal_cases() {
    std::size_t const unbounded_window = static_cast<std::size_t>(-1);
    std::vector<attention_causal_case_t> cases;
    for (nk_u32_t main_length : {1u, 31u, 32u, 33u, 513u, 1000u}) {
        std::int64_t const length = main_length, cache_offset = length - attention_causal_queries(main_length);
        for (std::size_t depth : {1ul, 65ul, 128ul, 257ul})
            for (std::int64_t diagonal_offset : {std::int64_t(0), cache_offset, std::int64_t(-3), length + 5})
                for (std::size_t window : {std::size_t(1), std::size_t(7), std::size_t(31), std::size_t(33),
                                           std::size_t(511), std::size_t(513), unbounded_window, std::size_t(0)})
                    cases.push_back({main_length, 2, depth, diagonal_offset, window});
    }
    std::int64_t const prefill_cache_offset = 1000 - attention_causal_queries(1000);
    for (std::size_t group : {1ul, 4ul, 8ul})
        for (std::int64_t diagonal_offset : {std::int64_t(0), prefill_cache_offset})
            for (std::size_t window : {std::size_t(65), unbounded_window})
                cases.push_back({1000, group, 128, diagonal_offset, window});
    return cases;
}

/** Packs every segment through @p pack_fn in two task windows, the second clipped from past the
 *  grid. */
template <typename backend_type_, typename pack_kernel_type_, typename scalar_vector_type_,
          typename packed_vector_type_>
void pack_attention_in_two_windows(backend_type_ &backend, pack_kernel_type_ pack_fn, scalar_vector_type_ const &keys,
                                   scalar_vector_type_ const &values, attention_segments<backend_type_> const &segments,
                                   attention_layout_t const &layout, packed_vector_type_ &key_value_packed) {
    using scalar_t = typename scalar_vector_type_::value_type;
    std::size_t const stride_bytes = layout.key_value_width() * sizeof(scalar_t);
    std::size_t const tasks = segments.count() * layout.key_value_head_count;
    std::size_t const windows[2][2] = {{0, 1}, {1, tasks + 7}};
    for (auto const &window : windows)
        backend.call(pack_fn, keys.raw_values_data(), values.raw_values_data(), layout.key_value_head_count,
                     layout.depth, segments.key_offsets.values_data(), segments.lengths.values_data(), segments.count(),
                     stride_bytes, stride_bytes, key_value_packed.raw_values_data(), window[0], window[1]);
}

/** Causal reference: the serial bidirectional kernel per query row, over a pack of exactly the keys
 *  that row may see, with zeros for rows that see none. */
template <typename scalar_type_, typename allocator_type_, typename backend_type_>
nk::vector<typename scalar_type_::attention_result_t> reference_by_rows(
    nk::vector<scalar_type_, allocator_type_> const &queries, nk::vector<scalar_type_, allocator_type_> const &keys,
    nk::vector<scalar_type_, allocator_type_> const &values, attention_segments<backend_type_> const &segments,
    attention_layout_t const &layout, std::int64_t diagonal_offset, std::size_t window) {
    using result_t = typename scalar_type_::attention_result_t;
    std::size_t const query_stride_bytes = layout.query_width() * sizeof(scalar_type_),
                      key_value_stride_bytes = layout.key_value_width() * sizeof(scalar_type_),
                      output_stride_bytes = layout.query_width() * sizeof(result_t);
    nk_u32_t const single_query_offsets[2] = {0, 1};
    auto reference = make_vector<result_t>(segments.query_tokens() * layout.query_width());
    for (std::size_t segment = 0; segment < segments.count(); segment++) {
        std::size_t const length = segments.lengths.values_data()[segment];
        for (std::size_t row = 0; row < segments.queries(segment); row++) {
            std::int64_t const position = static_cast<std::int64_t>(row) + diagonal_offset;
            if (position < 0 || window == 0) continue;
            std::size_t const query_position = static_cast<std::size_t>(position);
            std::size_t const key_end = std::min<std::size_t>(query_position + 1, length);
            std::size_t const key_begin = window > query_position ? 0 : std::min(query_position - window + 1, key_end);
            if (key_begin == key_end) continue;

            std::size_t const first_key = segments.key_offsets.values_data()[segment];
            nk_u32_t const visible_offsets[2] = {static_cast<nk_u32_t>(first_key + key_begin),
                                                 static_cast<nk_u32_t>(first_key + key_end)};
            nk_u32_t const visible_length = static_cast<nk_u32_t>(key_end - key_begin);
            auto key_value_reference = make_vector<char>(nk::attention_pack_size<scalar_type_, nk::no_simd_k>(
                layout.key_value_head_count, layout.depth, &visible_length, 1));
            nk::attention_pack<scalar_type_, nk::no_simd_k>(
                keys.values_data(), values.values_data(), layout.key_value_head_count, layout.depth, visible_offsets,
                &visible_length, 1, key_value_stride_bytes, key_value_stride_bytes,
                key_value_reference.raw_values_data());
            std::size_t const query_row = segments.query_offsets.values_data()[segment] + row;
            nk::attention_bidirectional_packed<scalar_type_, result_t, nk::no_simd_k>(
                queries.values_data() + query_row * layout.query_width(), key_value_reference.raw_values_data(),
                reference.values_data() + query_row * layout.query_width(), layout.head_count,
                layout.key_value_head_count, layout.depth, single_query_offsets, query_stride_bytes,
                output_stride_bytes, layout.scale);
        }
    }
    return reference;
}

#pragma endregion Operands

#pragma region Dots

/** Row strides of one dots case: the tightest the backend takes, or every operand padded past its
 *  row. */
enum class dots_strides_t {

    /** A and vectors at the backend's row stride, B and C exactly one row. */
    tight_k,

    /** A and vectors 16 bytes past it, B one value past its row and off 16 bytes, C three results
     *  past. */
    padded_k,
};

/** The values one dots case multiplies. */
enum class dots_operands_t {

    /** Drawn from the configured distribution. */
    random_k,

    /** F64 halves cancelling to ~2⁻³³ of Σ|a · b|, which plain F64 accumulation visibly misses. */
    ill_conditioned_k,
};

/** One case: C shaped @b [height,width] equals A shaped @b [height,depth] times B transposed,
 *  shaped @b [width,depth]. */
struct dots_packed_case_t {

    /** Rows of A and C. */
    std::size_t height;

    /** Rows of B and columns of C. */
    std::size_t width;

    /** Dimensions per row, rounded up to whole values before use. */
    std::size_t depth;

    /** Tight or padded row strides. */
    dots_strides_t strides;

    /** Random or ill-conditioned values. */
    dots_operands_t operands;
};

/** The configured shape, single cells, odd edges, exact tiles, a deep reduction, and an
 *  ill-conditioned F64 product. */
template <typename scalar_type_>
std::vector<dots_packed_case_t> dots_packed_cases() {
    dots_strides_t const tight = dots_strides_t::tight_k, padded = dots_strides_t::padded_k;
    dots_operands_t const random = dots_operands_t::random_k;
    std::vector<dots_packed_case_t> cases {
        {global_config.matrix_height, global_config.matrix_width, global_config.matrix_depth, tight, random},
        {1, 1, 1, tight, random},
        {1, 7, 3, tight, random},
        {17, 33, 65, tight, random},
        {17, 33, 65, padded, random},
        {128, 128, 64, tight, random},
        {257, 129, 300, tight, random},
        {257, 129, 300, padded, random},
        {33, 100, 4096, tight, random},
    };
    if constexpr (std::is_same_v<scalar_type_, f64_t>)
        cases.push_back({32, 48, 300, tight, dots_operands_t::ill_conditioned_k});
    return cases;
}

/** Fills A and B so every product cancels: B repeats its first half, A negates its first half up to
 *  a 2⁻³³ nudge. */
template <typename vector_type_, typename generator_type_>
void fill_ill_conditioned(generator_type_ &generator, vector_type_ &first, std::size_t first_stride_values,
                          vector_type_ &second, std::size_t second_stride_values, std::size_t height, std::size_t width,
                          std::size_t depth) {
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    std::size_t const half = depth / 2;
    for (std::size_t row = 0; row < height; row++) {
        double *values = first.raw_values_data() + row * first_stride_values;
        for (std::size_t index = 0; index < half; index++) {
            values[index] = unit(generator);
            values[half + index] = -values[index] + values[index] * 0x1p-33 * unit(generator);
        }
    }
    for (std::size_t row = 0; row < width; row++) {
        double *values = second.raw_values_data() + row * second_stride_values;
        for (std::size_t index = 0; index < half; index++) values[index] = values[half + index] = unit(generator);
    }
}

/** The largest error of plain F64 accumulation over every cell, in ulp of the reference. */
template <typename reference_vector_type_>
double naive_error_ulps(std::vector<double> const &first_rows, std::vector<double> const &second_rows,
                        reference_vector_type_ const &reference, std::size_t height, std::size_t width,
                        std::size_t depth) {
    double worst = 0;
    for (std::size_t row = 0; row < height; row++)
        for (std::size_t column = 0; column < width; column++) {
            double naive = 0;
            for (std::size_t index = 0; index < depth; index++)
                naive += first_rows[row * depth + index] * second_rows[column * depth + index];
            double const expected = static_cast<double>(reference[row * width + column]);
            double const absolute = std::fabs(expected), ulp = std::nextafter(absolute, INFINITY) - absolute;
            worst = std::max(worst, std::fabs(naive - expected) / ulp);
        }
    return worst;
}

/** Packed GEMM over @c dots_packed_cases against the serial `nk::` reference, with C stride padding
 *  left untouched. */
template <typename scalar_type_, typename backend_type_ = host_backend_t,
          accumulation_t accumulation_ = backend_type_::template dots_accumulation<scalar_type_>(),
          typename pack_size_kernel_type_, typename pack_kernel_type_, typename dots_kernel_type_>
error_stats_t test_dots_packed(pack_size_kernel_type_ packed_size_fn, pack_kernel_type_ pack_fn,
                               dots_kernel_type_ dots_fn) {
    using scalar_t = scalar_type_;
    using result_t = typename scalar_t::dot_result_t;
    using reference_t = reference_for<scalar_t, result_t>;
    using scalars_t = nk::vector<scalar_t, typename backend_type_::template allocator<scalar_t>>;
    using results_t = nk::vector<result_t, typename backend_type_::template allocator<result_t>>;
    using bytes_t = nk::vector<char, typename backend_type_::template allocator<char>>;

    backend_type_ backend;
    error_stats_t stats(dots_family(accumulation_));
    std::mt19937 generator(global_config.seed);
    std::size_t const dimensions_per_value = nk::dimensions_per_value<scalar_t>();
    std::vector<dots_packed_case_t> const cases = dots_packed_cases<scalar_t>();

    for (auto start = test_start_time(); within_time_budget(start);) {
        for (dots_packed_case_t const &test_case : cases) {
            std::size_t const height = test_case.height, width = test_case.width;
            std::size_t const depth = nk::divide_round_up(test_case.depth, dimensions_per_value) * dimensions_per_value;
            std::size_t const row_bytes = depth / dimensions_per_value * sizeof(scalar_t);
            bool const padded = test_case.strides == dots_strides_t::padded_k;
            std::size_t const a_stride = backend.row_stride(row_bytes) + (padded ? 16 : 0);
            std::size_t b_stride = row_bytes + (padded ? sizeof(scalar_t) : 0);
            if (padded && b_stride % 16 == 0) b_stride += sizeof(scalar_t);
            std::size_t const c_stride = (width + (padded ? 3 : 0)) * sizeof(result_t);
            std::size_t const a_stride_values = a_stride / sizeof(scalar_t),
                              b_stride_values = b_stride / sizeof(scalar_t);

            auto a = scalars_t::try_zeros(height * a_stride_values * dimensions_per_value),
                 b = scalars_t::try_zeros(width * b_stride_values * dimensions_per_value);
            auto c = results_t::try_zeros(height * c_stride / sizeof(result_t));
            auto b_packed = bytes_t::try_zeros(packed_size_fn(width, depth));
            auto c_reference = make_vector<reference_t>(height * width);
            auto b_packed_reference = make_vector<char>(nk::dots_pack_size<scalar_t, nk::no_simd_k>(width, depth));

            if constexpr (std::is_same_v<scalar_t, f64_t>) {
                if (test_case.operands == dots_operands_t::ill_conditioned_k)
                    fill_ill_conditioned(generator, a, a_stride_values, b, b_stride_values, height, width, depth);
                else fill_random(generator, a), fill_random(generator, b);
            }
            else fill_random(generator, a), fill_random(generator, b);
            fill_padding_canary(a, height, row_bytes, a_stride), fill_padding_canary(b, width, row_bytes, b_stride);
            fill_canary(c);

            // Run kernel being tested
            backend.call(pack_fn, b.raw_values_data(), width, depth, b_stride, b_packed.raw_values_data(), 0, width);
            backend.call(dots_fn, a.raw_values_data(), b_packed.raw_values_data(), c.raw_values_data(), height, width,
                         depth, a_stride, c_stride);
            synchronize(backend, stats);

            // Compute reference using nk:: template
            nk::dots_pack<scalar_t, nk::no_simd_k>(b.values_data(), width, depth, b_stride,
                                                   b_packed_reference.raw_values_data());
            nk::dots_packed<scalar_t, reference_t, nk::no_simd_k>(a.values_data(), b_packed_reference.raw_values_data(),
                                                                  c_reference.values_data(), height, width, depth,
                                                                  a_stride, width * sizeof(reference_t));
            std::vector<double> const a_rows = decode_for<accumulation_>(a, height, depth, a_stride_values),
                                      b_rows = decode_for<accumulation_>(b, width, depth, b_stride_values);

            for (std::size_t row = 0; row < height; row++)
                for (std::size_t column = 0; column < width; column++)
                    accumulate_dot<accumulation_>(stats, c[row * c_stride / sizeof(result_t) + column],
                                                  c_reference[row * width + column], a_rows, row, b_rows, column,
                                                  depth);
            expect_padding_untouched(stats, c, height, width * sizeof(result_t), c_stride);
            if constexpr (std::is_same_v<scalar_t, f64_t> &&
                          dots_family(accumulation_) == comparison_family_t::bounded_k)
                if (test_case.operands == dots_operands_t::ill_conditioned_k)
                    stats.expect(naive_error_ulps(a_rows, b_rows, c_reference, height, width, depth) >= 1e3,
                                 "the ill-conditioned case is within 1000 ulp for plain F64");
        }
    }
    return stats;
}

/** The packed B layout over the widths and depths of @c dots_packed_cases: packing two column
 *  windows equals packing all columns at once byte for byte, and @c packed_shape_fn_ reads back the
 *  width and depth the pack was given. */
template <typename scalar_type_, typename backend_type_, auto packed_size_fn_, auto packed_shape_fn_, auto pack_fn_>
error_stats_t test_dots_pack_layout() {
    using scalar_t = scalar_type_;
    using scalars_t = nk::vector<scalar_t, typename backend_type_::template allocator<scalar_t>>;
    using bytes_t = nk::vector<char, typename backend_type_::template allocator<char>>;

    backend_type_ backend;
    error_stats_t stats(comparison_family_t::exact_k);
    std::mt19937 generator(global_config.seed);
    std::size_t const dimensions_per_value = nk::dimensions_per_value<scalar_t>();
    std::vector<dots_packed_case_t> const cases = dots_packed_cases<scalar_t>();

    for (auto start = test_start_time(); within_time_budget(start);) {
        for (dots_packed_case_t const &test_case : cases) {
            std::size_t const width = test_case.width;
            std::size_t const depth = nk::divide_round_up(test_case.depth, dimensions_per_value) * dimensions_per_value;
            std::size_t const row_bytes = depth / dimensions_per_value * sizeof(scalar_t);
            auto b = scalars_t::try_zeros(width * depth);
            auto whole = bytes_t::try_zeros(packed_size_fn_(width, depth)),
                 windows = bytes_t::try_zeros(packed_size_fn_(width, depth));
            fill_random(generator, b);
            fill_canary(whole), fill_canary(windows);

            nk_size_t shape_width = 0, shape_depth = 0;
            // Run kernel being tested: one pack of every column, one in two column windows, then the shape
            backend.call(pack_fn_, b.raw_values_data(), width, depth, row_bytes, whole.raw_values_data(), 0, width);
            backend.call(pack_fn_, b.raw_values_data(), width, depth, row_bytes, windows.raw_values_data(), 0,
                         width / 2);
            backend.call(pack_fn_, b.raw_values_data(), width, depth, row_bytes, windows.raw_values_data(), width / 2,
                         width);
            backend.call(packed_shape_fn_, whole.raw_values_data(), &shape_width, &shape_depth);
            synchronize(backend, stats);

            stats.expect(shape_width == width && shape_depth == depth, "packed_shape disagrees with the pack");
            stats.expect(std::memcmp(whole.raw_values_data(), windows.raw_values_data(), whole.size_bytes()) == 0,
                         "two column windows pack differently from one pack of every column");
        }
    }
    return stats;
}

/** One Gram-matrix case over @c count vectors, computing rows [row_start, row_start + row_count)
 *  clipped to @c count. */
struct dots_symmetric_case_t {

    /** Vectors, and rows and columns of the result. */
    std::size_t count;

    /** Dimensions per vector, rounded up to whole values before use. */
    std::size_t depth;

    /** First result row computed. */
    std::size_t row_start;

    /** Result rows computed, clipped at the last vector. */
    std::size_t row_count;

    /** Tight or padded strides for the vectors and the result. */
    dots_strides_t strides;
};

/** The configured shape, then a single cell, whole matrices, a range cut through a tile, and one
 *  clipped at the end. */
inline std::vector<dots_symmetric_case_t> dots_symmetric_cases() {
    dots_strides_t const tight = dots_strides_t::tight_k, padded = dots_strides_t::padded_k;
    return {
        {global_config.matrix_height, global_config.matrix_depth, 0, global_config.matrix_height, tight},
        {1, 1, 0, 1, tight},
        {17, 65, 0, 17, padded},
        {129, 300, 0, 129, padded},
        {129, 300, 5, 40, padded},
        {300, 1024, 256, 100, padded},
    };
}

/** Symmetric GEMM, A × Aᵀ, over @c dots_symmetric_cases against the serial `nk::` reference, on and
 *  above the diagonal of the computed rows, with everything else in the output left untouched. */
template <typename scalar_type_, typename backend_type_ = host_backend_t,
          accumulation_t accumulation_ = backend_type_::template dots_accumulation<scalar_type_>(),
          typename symmetric_kernel_type_>
error_stats_t test_dots_symmetric(symmetric_kernel_type_ symmetric_fn) {
    using scalar_t = scalar_type_;
    using result_t = typename scalar_t::dot_result_t;
    using reference_t = reference_for<scalar_t, result_t>;
    using scalars_t = nk::vector<scalar_t, typename backend_type_::template allocator<scalar_t>>;
    using results_t = nk::vector<result_t, typename backend_type_::template allocator<result_t>>;

    backend_type_ backend;
    error_stats_t stats(dots_family(accumulation_));
    std::mt19937 generator(global_config.seed);
    std::size_t const dimensions_per_value = nk::dimensions_per_value<scalar_t>();
    std::vector<dots_symmetric_case_t> const cases = dots_symmetric_cases();

    for (auto start = test_start_time(); within_time_budget(start);) {
        for (dots_symmetric_case_t const &test_case : cases) {
            std::size_t const count = test_case.count, row_start = test_case.row_start;
            std::size_t const row_end = std::min(count, row_start + test_case.row_count);
            std::size_t const depth = nk::divide_round_up(test_case.depth, dimensions_per_value) * dimensions_per_value;
            std::size_t const row_bytes = depth / dimensions_per_value * sizeof(scalar_t);
            bool const padded = test_case.strides == dots_strides_t::padded_k;
            std::size_t const stride = backend.row_stride(row_bytes) + (padded ? 16 : 0);
            std::size_t const c_stride = (count + (padded ? 3 : 0)) * sizeof(result_t);
            std::size_t const stride_values = stride / sizeof(scalar_t);

            auto a = scalars_t::try_zeros(count * stride_values * dimensions_per_value);
            auto c = results_t::try_zeros(count * c_stride / sizeof(result_t));
            auto c_reference = make_vector<reference_t>(count * count);
            fill_random(generator, a);
            fill_padding_canary(a, count, row_bytes, stride), fill_canary(c);

            // Run kernel being tested
            backend.call(symmetric_fn, a.raw_values_data(), count, depth, stride, c.raw_values_data(), c_stride,
                         row_start, test_case.row_count);
            synchronize(backend, stats);

            // Compute reference using nk:: template
            nk::dots_symmetric<scalar_t, reference_t, nk::no_simd_k>(
                a.values_data(), count, depth, stride, c_reference.values_data(), count * sizeof(reference_t),
                row_start, row_end - row_start);
            std::vector<double> const rows = decode_for<accumulation_>(a, count, depth, stride_values);

            for (std::size_t row = row_start; row < row_end; row++)
                for (std::size_t column = row; column < count; column++)
                    accumulate_dot<accumulation_>(stats, c[row * c_stride / sizeof(result_t) + column],
                                                  c_reference[row * count + column], rows, row, rows, column, depth);
            expect_symmetric_untouched<result_t>(stats, c, count, c_stride, row_start, row_end);
        }
    }
    return stats;
}

/** The launch contract of a backend refusing misaligned operands, which a CPU backend has no part
 *  in: pack sizes over @c dots_packed_cases hold a 64-byte header, every B row and one norm per
 *  column, and an A or vectors pointer or stride off 16 bytes, or a C stride off the result size,
 *  is refused before launch with its output untouched. */
template <typename scalar_type_, typename backend_type_, auto packed_size_fn_, auto dots_fn_, auto symmetric_fn_>
error_stats_t test_dots_launch_contract() {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_t::dot_result_t;
    using norm_t = std::conditional_t<nk::is_integral_dtype<result_t>(), nk_u32_t, result_t>;
    using results_t = nk::vector<result_t, typename backend_type_::template allocator<result_t>>;
    using bytes_t = nk::vector<char, typename backend_type_::template allocator<char>>;

    backend_type_ backend;
    error_stats_t stats(comparison_family_t::exact_k);
    std::size_t const dimensions_per_value = nk::dimensions_per_value<scalar_t>();

    for (dots_packed_case_t const &test_case : dots_packed_cases<scalar_t>()) {
        std::size_t const width = test_case.width;
        std::size_t const depth = nk::divide_round_up(test_case.depth, dimensions_per_value) * dimensions_per_value;
        std::size_t const row_bytes = depth / dimensions_per_value * sizeof(scalar_t);
        stats.expect(packed_size_fn_(width, depth) >= 64 + width * row_bytes + width * sizeof(norm_t),
                     "pack size below a header, every row, and every norm");
    }

    std::size_t const count = 4, depth = 16 * dimensions_per_value, row_bytes = 16 * sizeof(scalar_t);
    std::size_t const aligned_stride = backend.row_stride(row_bytes), output_stride = count * sizeof(result_t);
    std::size_t const odd_stride = row_bytes % 16 == 15 ? row_bytes + 2 : row_bytes + 1;
    auto rows = bytes_t::try_zeros(count * (aligned_stride + odd_stride) + 16);
    auto packed = bytes_t::try_zeros(packed_size_fn_(count, depth));
    auto output = results_t::try_zeros(count * 2 * count);
    fill_canary(output);

    auto const *aligned = reinterpret_cast<raw_t const *>(rows.raw_values_data());
    auto const *shifted = reinterpret_cast<raw_t const *>(rows.raw_values_data() + 1);
    auto *cells = output.raw_values_data();
    void const *b_packed = packed.raw_values_data();
    stats.expect(backend.refuses_misaligned(dots_fn_, shifted, b_packed, cells, count, count, depth, aligned_stride,
                                            output_stride),
                 "packed took an A off 16 bytes");
    stats.expect(
        backend.refuses_misaligned(dots_fn_, aligned, b_packed, cells, count, count, depth, odd_stride, output_stride),
        "packed took an A stride off 16 bytes");
    stats.expect(backend.refuses_misaligned(symmetric_fn_, shifted, count, depth, aligned_stride, cells, output_stride,
                                            0, count),
                 "symmetric took vectors off 16 bytes");
    stats.expect(
        backend.refuses_misaligned(symmetric_fn_, aligned, count, depth, odd_stride, cells, output_stride, 0, count),
        "symmetric took a vectors stride off 16 bytes");
    // Four bytes of padding keep 4-byte results aligned, so only 8-byte results can be caught off their size
    if constexpr (sizeof(result_t) == 8) {
        std::size_t const word_stride = output_stride + 4;
        stats.expect(backend.refuses_misaligned(dots_fn_, aligned, b_packed, cells, count, count, depth, aligned_stride,
                                                word_stride),
                     "packed took a C stride off the result size");
        stats.expect(backend.refuses_misaligned(symmetric_fn_, aligned, count, depth, aligned_stride, cells,
                                                word_stride, 0, count),
                     "symmetric took a result stride off the result size");
    }
    synchronize(backend, stats);
    stats.expect(overwritten_bytes(output, 0, output.size_bytes()) == 0, "a refused call wrote its output");
    return stats;
}

#pragma endregion Dots

#pragma region Set Distances

/** Batched Hamming distances with a packed B matrix, exact against the serial `nk::` reference. */
template <typename scalar_type_>
error_stats_t test_hammings_packed(typename scalar_type_::hammings_pack_size_kernel_t packed_size_fn,
                                   typename scalar_type_::hammings_pack_kernel_t pack_fn,
                                   typename scalar_type_::hammings_packed_kernel_t hammings_fn) {
    using scalar_t = scalar_type_;
    using result_t = u32_t;

    error_stats_t stats(comparison_family_t::exact_k);
    std::mt19937 generator(global_config.seed);

    std::size_t m = global_config.matrix_height, n = global_config.matrix_width;
    std::size_t const dims_per_value = nk::dimensions_per_value<scalar_t>();
    std::size_t const k = nk::divide_round_up(global_config.dense_dimensions, dims_per_value) * dims_per_value;
    std::size_t const stride = nk::divide_round_up(k, 8) * sizeof(scalar_t);
    std::size_t const stride_dimensions = stride / sizeof(scalar_t) * dims_per_value;
    std::size_t c_stride = n * sizeof(result_t);

    auto a = make_vector<scalar_t>(m * stride_dimensions), b = make_vector<scalar_t>(n * stride_dimensions);
    auto c = make_vector<result_t>(m * n);
    auto c_ref = make_vector<result_t>(m * n);
    auto b_packed = make_vector<char>(packed_size_fn(n, k));
    auto b_packed_ref = make_vector<char>(nk::dots_pack_size<scalar_t, nk::no_simd_k>(n, k));

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        // Run kernel being tested
        pack_fn(b.raw_values_data(), n, k, stride, b_packed.raw_values_data(), 0, n);
        hammings_fn(a.raw_values_data(), b_packed.raw_values_data(), c.raw_values_data(), m, n, k, stride, c_stride);

        // Compute reference using C++ template with no_simd_k
        nk::dots_pack<scalar_t, nk::no_simd_k>(b.values_data(), n, k, stride, b_packed_ref.raw_values_data());
        nk::hammings_packed<scalar_t, result_t, nk::no_simd_k>(a.values_data(), b_packed_ref.raw_values_data(),
                                                               c_ref.values_data(), m, n, k, stride, c_stride);

        // Hamming distances are exact integers
        for (std::size_t i = 0; i < m * n; i++) stats.accumulate(c[i], c_ref[i]);
    }
    return stats;
}

/** Symmetric Hamming distances, exact against the serial `nk::` reference over the upper triangle,
 *  and untouched below it. */
template <typename scalar_type_>
error_stats_t test_hammings_symmetric(typename scalar_type_::hammings_symmetric_kernel_t symmetric_fn) {
    using scalar_t = scalar_type_;
    using result_t = u32_t;

    error_stats_t stats(comparison_family_t::exact_k);
    std::mt19937 generator(global_config.seed);

    std::size_t n = global_config.matrix_height;
    std::size_t const dims_per_value = nk::dimensions_per_value<scalar_t>();
    std::size_t const k = nk::divide_round_up(global_config.dense_dimensions, dims_per_value) * dims_per_value;
    std::size_t const stride = nk::divide_round_up(k, 8) * sizeof(scalar_t);
    std::size_t const stride_dimensions = stride / sizeof(scalar_t) * dims_per_value;
    std::size_t c_stride = n * sizeof(result_t);

    auto a = make_vector<scalar_t>(n * stride_dimensions);
    auto c = make_vector<result_t>(n * n);
    auto c_ref = make_vector<result_t>(n * n);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_canary(c);

        // Run kernel being tested
        symmetric_fn(a.raw_values_data(), n, k, stride, c.raw_values_data(), c_stride, 0, n);

        // Compute reference using nk:: template
        nk::hammings_symmetric<scalar_t, result_t, nk::no_simd_k>(a.values_data(), n, k, stride, c_ref.values_data(),
                                                                  n * sizeof(result_t));

        // Hamming distances are exact integers on and above the diagonal, with nothing written below it
        for (std::size_t i = 0; i < n; i++)
            for (std::size_t j = i; j < n; j++) stats.accumulate(c[i * n + j], c_ref[i * n + j]);
        expect_symmetric_untouched<result_t>(stats, c, n, c_stride, 0, n);
    }
    return stats;
}

/** Batched Jaccard distances with a packed B matrix, exact against the serial `nk::` reference. */
template <typename scalar_type_>
error_stats_t test_jaccards_packed(typename scalar_type_::jaccards_pack_size_kernel_t packed_size_fn,
                                   typename scalar_type_::jaccards_pack_kernel_t pack_fn,
                                   typename scalar_type_::jaccards_packed_kernel_t jaccards_fn) {
    using scalar_t = scalar_type_;
    using result_t = f32_t;

    error_stats_t stats(comparison_family_t::exact_k);
    std::mt19937 generator(global_config.seed);

    std::size_t m = global_config.matrix_height, n = global_config.matrix_width;
    std::size_t const dims_per_value = nk::dimensions_per_value<scalar_t>();
    std::size_t const k = nk::divide_round_up(global_config.dense_dimensions, dims_per_value) * dims_per_value;
    std::size_t const stride = nk::divide_round_up(k, 8) * sizeof(scalar_t);
    std::size_t const stride_dimensions = stride / sizeof(scalar_t) * dims_per_value;
    std::size_t c_stride = n * sizeof(result_t);

    auto a = make_vector<scalar_t>(m * stride_dimensions), b = make_vector<scalar_t>(n * stride_dimensions);
    auto c = make_vector<result_t>(m * n);
    auto c_ref = make_vector<result_t>(m * n);
    auto b_packed = make_vector<char>(packed_size_fn(n, k));
    auto b_packed_ref = make_vector<char>(nk::dots_pack_size<scalar_t, nk::no_simd_k>(n, k));

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        // Run kernel being tested
        pack_fn(b.raw_values_data(), n, k, stride, b_packed.raw_values_data(), 0, n);
        jaccards_fn(a.raw_values_data(), b_packed.raw_values_data(), c.raw_values_data(), m, n, k, stride, c_stride);

        // Compute reference using C++ template with no_simd_k
        nk::dots_pack<scalar_t, nk::no_simd_k>(b.values_data(), n, k, stride, b_packed_ref.raw_values_data());
        nk::jaccards_packed<scalar_t, result_t, nk::no_simd_k>(a.values_data(), b_packed_ref.raw_values_data(),
                                                               c_ref.values_data(), m, n, k, stride, c_stride);

        for (std::size_t i = 0; i < m * n; i++) stats.accumulate(c[i], c_ref[i]);
    }
    return stats;
}

/** Symmetric Jaccard distances, exact against the serial `nk::` reference over the upper triangle,
 *  and untouched below it. */
template <typename scalar_type_>
error_stats_t test_jaccards_symmetric(typename scalar_type_::jaccards_symmetric_kernel_t symmetric_fn) {
    using scalar_t = scalar_type_;
    using result_t = f32_t;

    error_stats_t stats(comparison_family_t::exact_k);
    std::mt19937 generator(global_config.seed);

    std::size_t n = global_config.matrix_height;
    std::size_t const dims_per_value = nk::dimensions_per_value<scalar_t>();
    std::size_t const k = nk::divide_round_up(global_config.dense_dimensions, dims_per_value) * dims_per_value;
    std::size_t const stride = nk::divide_round_up(k, 8) * sizeof(scalar_t);
    std::size_t const stride_dimensions = stride / sizeof(scalar_t) * dims_per_value;
    std::size_t c_stride = n * sizeof(result_t);

    auto a = make_vector<scalar_t>(n * stride_dimensions);
    auto c = make_vector<result_t>(n * n);
    auto c_ref = make_vector<result_t>(n * n);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_canary(c);

        // Run kernel being tested
        symmetric_fn(a.raw_values_data(), n, k, stride, c.raw_values_data(), c_stride, 0, n);

        // Compute reference using nk:: template
        nk::jaccards_symmetric<scalar_t, result_t, nk::no_simd_k>(a.values_data(), n, k, stride, c_ref.values_data(),
                                                                  n * sizeof(result_t));

        // Values on and above the diagonal, with nothing written below it
        for (std::size_t i = 0; i < n; i++)
            for (std::size_t j = i; j < n; j++) stats.accumulate(c[i * n + j], c_ref[i * n + j]);
        expect_symmetric_untouched<result_t>(stats, c, n, c_stride, 0, n);
    }
    return stats;
}

#pragma endregion Set Distances

#pragma region Spatial Distances

/** Batched angular distances, 1 − dot / √(‖a‖² · ‖b‖²), with B packed in two column windows. */
template <typename scalar_type_, typename backend_type_ = host_backend_t,
          accumulation_t accumulation_ = backend_type_::template spatials_accumulation<scalar_type_>(),
          typename pack_size_kernel_type_, typename pack_kernel_type_, typename angulars_kernel_type_>
error_stats_t test_angulars_packed(pack_size_kernel_type_ packed_size_fn, pack_kernel_type_ pack_fn,
                                   angulars_kernel_type_ angulars_fn) {
    using scalar_t = scalar_type_;
    using result_t = typename scalar_t::angular_result_t;
    using reference_t = reference_for<scalar_t, result_t>;
    using scalars_t = nk::vector<scalar_t, typename backend_type_::template allocator<scalar_t>>;
    using results_t = nk::vector<result_t, typename backend_type_::template allocator<result_t>>;
    using bytes_t = nk::vector<char, typename backend_type_::template allocator<char>>;

    backend_type_ backend;
    error_stats_t stats(spatials_family(accumulation_));
    std::mt19937 generator(global_config.seed);

    std::size_t m = global_config.matrix_height, n = global_config.matrix_width;
    std::size_t const dims_per_value = nk::dimensions_per_value<scalar_t>();
    std::size_t c_stride = n * sizeof(result_t);

    std::size_t const k = nk::divide_round_up(global_config.matrix_depth, dims_per_value) * dims_per_value;
    std::size_t const stride = backend.row_stride(nk::divide_round_up(k, dims_per_value) * sizeof(scalar_t));
    std::size_t const stride_values = stride / sizeof(scalar_t);

    auto a = scalars_t::try_zeros(m * stride_values * dims_per_value),
         b = scalars_t::try_zeros(n * stride_values * dims_per_value);
    auto c = results_t::try_zeros(m * n);
    auto c_ref = make_vector<reference_t>(m * n);
    auto b_packed = bytes_t::try_zeros(packed_size_fn(n, k));
    auto b_packed_ref = make_vector<char>(nk::dots_pack_size<scalar_t, nk::no_simd_k>(n, k));
    auto a_sumsqs = make_vector<reference_t>(m);
    auto b_sumsqs = make_vector<reference_t>(n);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        // Reference: compute dot products in reference precision
        nk::dots_pack<scalar_t, nk::no_simd_k>(b.values_data(), n, k, stride, b_packed_ref.raw_values_data());
        nk::dots_packed<scalar_t, reference_t, nk::no_simd_k>(a.values_data(), b_packed_ref.raw_values_data(),
                                                              c_ref.values_data(), m, n, k, stride,
                                                              n * sizeof(reference_t));

        reference_t sum_unused;
        for (std::size_t i = 0; i < m; ++i)
            nk::reduce_moments<scalar_t, reference_t, reference_t, nk::no_simd_k>(
                a.values_data() + i * stride_values, k, sizeof(scalar_t), &sum_unused, a_sumsqs.values_data() + i);
        for (std::size_t j = 0; j < n; ++j)
            nk::reduce_moments<scalar_t, reference_t, reference_t, nk::no_simd_k>(
                b.values_data() + j * stride_values, k, sizeof(scalar_t), &sum_unused, b_sumsqs.values_data() + j);

        // Convert dots to angular distances: 1 - dot / sqrt(sumsq_a * sumsq_b)
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                reference_t ab_sumsq = a_sumsqs[i] * b_sumsqs[j];
                reference_t &c_cell = c_ref[i * n + j];
                c_cell = ab_sumsq > reference_t(0) ? (reference_t(1) - c_cell * ab_sumsq.rsqrt()) : reference_t(0);
            }

        // Run kernel being tested: the norms of the second window's columns come from a pack not starting at zero
        backend.call(pack_fn, b.raw_values_data(), n, k, stride, b_packed.raw_values_data(), 0, n / 2);
        backend.call(pack_fn, b.raw_values_data(), n, k, stride, b_packed.raw_values_data(), n / 2, n);
        backend.call(angulars_fn, a.raw_values_data(), b_packed.raw_values_data(), c.raw_values_data(), m, n, k, stride,
                     c_stride);
        synchronize(backend, stats);

        for (std::size_t i = 0; i < m * n; i++) accumulate_angular<accumulation_>(stats, c[i], c_ref[i], k);
    }
    return stats;
}

/** Batched euclidean distances, √max(0, ‖a‖² + ‖b‖² − 2 · dot), with B packed in two column
 *  windows. Row 0 of A is zero, so row 0 of C reads every packed norm back as √‖b‖². */
template <typename scalar_type_, typename backend_type_ = host_backend_t,
          accumulation_t accumulation_ = backend_type_::template spatials_accumulation<scalar_type_>(),
          typename pack_size_kernel_type_, typename pack_kernel_type_, typename euclideans_kernel_type_>
error_stats_t test_euclideans_packed(pack_size_kernel_type_ packed_size_fn, pack_kernel_type_ pack_fn,
                                     euclideans_kernel_type_ euclideans_fn) {
    using scalar_t = scalar_type_;
    using result_t = typename scalar_t::euclidean_result_t;
    using reference_t = reference_for<scalar_t, result_t>;
    using scalars_t = nk::vector<scalar_t, typename backend_type_::template allocator<scalar_t>>;
    using results_t = nk::vector<result_t, typename backend_type_::template allocator<result_t>>;
    using bytes_t = nk::vector<char, typename backend_type_::template allocator<char>>;

    backend_type_ backend;
    error_stats_t stats(spatials_family(accumulation_));
    std::mt19937 generator(global_config.seed);

    std::size_t m = global_config.matrix_height, n = global_config.matrix_width;
    std::size_t const dims_per_value = nk::dimensions_per_value<scalar_t>();
    std::size_t c_stride = n * sizeof(result_t);

    std::size_t const k = nk::divide_round_up(global_config.matrix_depth, dims_per_value) * dims_per_value;
    std::size_t const stride = backend.row_stride(nk::divide_round_up(k, dims_per_value) * sizeof(scalar_t));
    std::size_t const stride_values = stride / sizeof(scalar_t);

    auto a = scalars_t::try_zeros(m * stride_values * dims_per_value),
         b = scalars_t::try_zeros(n * stride_values * dims_per_value);
    auto c = results_t::try_zeros(m * n);
    auto c_ref = make_vector<reference_t>(m * n);
    auto b_packed = bytes_t::try_zeros(packed_size_fn(n, k));
    auto b_packed_ref = make_vector<char>(nk::dots_pack_size<scalar_t, nk::no_simd_k>(n, k));
    auto a_sumsqs = make_vector<reference_t>(m);
    auto b_sumsqs = make_vector<reference_t>(n);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);
        std::memset(a.raw_values_data(), 0, stride);

        // Reference: compute dot products in reference precision
        nk::dots_pack<scalar_t, nk::no_simd_k>(b.values_data(), n, k, stride, b_packed_ref.raw_values_data());
        nk::dots_packed<scalar_t, reference_t, nk::no_simd_k>(a.values_data(), b_packed_ref.raw_values_data(),
                                                              c_ref.values_data(), m, n, k, stride,
                                                              n * sizeof(reference_t));

        reference_t sum_unused;
        for (std::size_t i = 0; i < m; ++i)
            nk::reduce_moments<scalar_t, reference_t, reference_t, nk::no_simd_k>(
                a.values_data() + i * stride_values, k, sizeof(scalar_t), &sum_unused, a_sumsqs.values_data() + i);
        for (std::size_t j = 0; j < n; ++j)
            nk::reduce_moments<scalar_t, reference_t, reference_t, nk::no_simd_k>(
                b.values_data() + j * stride_values, k, sizeof(scalar_t), &sum_unused, b_sumsqs.values_data() + j);

        // Convert dots to euclidean distances: sqrt(max(0, sumsq_a + sumsq_b - 2*dot))
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                reference_t &c_cell = c_ref[i * n + j];
                reference_t diff = a_sumsqs[i] + b_sumsqs[j] - reference_t(2) * c_cell;
                c_cell = diff > reference_t(0) ? diff.sqrt() : reference_t(0);
            }

        // Run kernel being tested: the norms of the second window's columns come from a pack not starting at zero
        backend.call(pack_fn, b.raw_values_data(), n, k, stride, b_packed.raw_values_data(), 0, n / 2);
        backend.call(pack_fn, b.raw_values_data(), n, k, stride, b_packed.raw_values_data(), n / 2, n);
        backend.call(euclideans_fn, a.raw_values_data(), b_packed.raw_values_data(), c.raw_values_data(), m, n, k,
                     stride, c_stride);
        synchronize(backend, stats);

        for (std::size_t i = 0; i < m; i++)
            for (std::size_t j = 0; j < n; j++)
                accumulate_euclidean<accumulation_>(stats, c[i * n + j], c_ref[i * n + j],
                                                    reference_t(a_sumsqs[i] + b_sumsqs[j]), k);
    }
    return stats;
}

/** Symmetric angular distances over the upper triangle, zeros on the diagonal, and untouched below
 *  it. */
template <typename scalar_type_, typename backend_type_ = host_backend_t,
          accumulation_t accumulation_ = backend_type_::template spatials_accumulation<scalar_type_>(),
          typename symmetric_kernel_type_>
error_stats_t test_angulars_symmetric(symmetric_kernel_type_ symmetric_fn) {
    using scalar_t = scalar_type_;
    using result_t = typename scalar_t::angular_result_t;
    using reference_t = reference_for<scalar_t, result_t>;
    using scalars_t = nk::vector<scalar_t, typename backend_type_::template allocator<scalar_t>>;
    using results_t = nk::vector<result_t, typename backend_type_::template allocator<result_t>>;

    backend_type_ backend;
    error_stats_t stats(spatials_family(accumulation_));
    std::mt19937 generator(global_config.seed);

    std::size_t n = global_config.matrix_height;
    std::size_t const dims_per_value = nk::dimensions_per_value<scalar_t>();
    std::size_t c_stride = n * sizeof(result_t);

    std::size_t const k = nk::divide_round_up(global_config.matrix_depth, dims_per_value) * dims_per_value;
    std::size_t const stride = backend.row_stride(nk::divide_round_up(k, dims_per_value) * sizeof(scalar_t));
    std::size_t const stride_values = stride / sizeof(scalar_t);

    auto a = scalars_t::try_zeros(n * stride_values * dims_per_value);
    auto c = results_t::try_zeros(n * n);
    auto c_ref = make_vector<reference_t>(n * n);
    auto sumsqs = make_vector<reference_t>(n);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_canary(c);

        // Reference: compute dots symmetric in reference precision
        nk::dots_symmetric<scalar_t, reference_t, nk::no_simd_k>(a.values_data(), n, k, stride, c_ref.values_data(),
                                                                 n * sizeof(reference_t));

        reference_t sum_unused;
        for (std::size_t i = 0; i < n; ++i)
            nk::reduce_moments<scalar_t, reference_t, reference_t, nk::no_simd_k>(
                a.values_data() + i * stride_values, k, sizeof(scalar_t), &sum_unused, sumsqs.values_data() + i);

        // Convert dots to angular distances: diagonal=0, upper triangle uses formula
        for (std::size_t i = 0; i < n; ++i) {
            c_ref[i * n + i] = reference_t(0);
            for (std::size_t j = i + 1; j < n; ++j) {
                reference_t ab_sumsq = sumsqs[i] * sumsqs[j];
                reference_t &c_cell = c_ref[i * n + j];
                c_cell = ab_sumsq > reference_t(0) ? (reference_t(1) - c_cell * ab_sumsq.rsqrt()) : reference_t(0);
            }
        }

        // Run kernel being tested
        backend.call(symmetric_fn, a.raw_values_data(), n, k, stride, c.raw_values_data(), c_stride, 0, n);
        synchronize(backend, stats);

        // Values on and above the diagonal, with nothing written below it
        for (std::size_t i = 0; i < n; i++)
            for (std::size_t j = i; j < n; j++)
                accumulate_angular<accumulation_>(stats, c[i * n + j], c_ref[i * n + j], k);
        expect_symmetric_untouched<result_t>(stats, c, n, c_stride, 0, n);
    }
    return stats;
}

/** Symmetric euclidean distances over the upper triangle, zeros on the diagonal, and untouched
 *  below it. */
template <typename scalar_type_, typename backend_type_ = host_backend_t,
          accumulation_t accumulation_ = backend_type_::template spatials_accumulation<scalar_type_>(),
          typename symmetric_kernel_type_>
error_stats_t test_euclideans_symmetric(symmetric_kernel_type_ symmetric_fn) {
    using scalar_t = scalar_type_;
    using result_t = typename scalar_t::euclidean_result_t;
    using reference_t = reference_for<scalar_t, result_t>;
    using scalars_t = nk::vector<scalar_t, typename backend_type_::template allocator<scalar_t>>;
    using results_t = nk::vector<result_t, typename backend_type_::template allocator<result_t>>;

    backend_type_ backend;
    error_stats_t stats(spatials_family(accumulation_));
    std::mt19937 generator(global_config.seed);

    std::size_t n = global_config.matrix_height;
    std::size_t const dims_per_value = nk::dimensions_per_value<scalar_t>();
    std::size_t c_stride = n * sizeof(result_t);

    std::size_t const k = nk::divide_round_up(global_config.matrix_depth, dims_per_value) * dims_per_value;
    std::size_t const stride = backend.row_stride(nk::divide_round_up(k, dims_per_value) * sizeof(scalar_t));
    std::size_t const stride_values = stride / sizeof(scalar_t);

    auto a = scalars_t::try_zeros(n * stride_values * dims_per_value);
    auto c = results_t::try_zeros(n * n);
    auto c_ref = make_vector<reference_t>(n * n);
    auto sumsqs = make_vector<reference_t>(n);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_canary(c);

        // Reference: compute dots symmetric in reference precision
        nk::dots_symmetric<scalar_t, reference_t, nk::no_simd_k>(a.values_data(), n, k, stride, c_ref.values_data(),
                                                                 n * sizeof(reference_t));

        reference_t sum_unused;
        for (std::size_t i = 0; i < n; ++i)
            nk::reduce_moments<scalar_t, reference_t, reference_t, nk::no_simd_k>(
                a.values_data() + i * stride_values, k, sizeof(scalar_t), &sum_unused, sumsqs.values_data() + i);

        // Convert dots to euclidean distances: diagonal=0, upper triangle uses formula
        for (std::size_t i = 0; i < n; ++i) {
            c_ref[i * n + i] = reference_t(0);
            for (std::size_t j = i + 1; j < n; ++j) {
                reference_t &c_cell = c_ref[i * n + j];
                reference_t diff = sumsqs[i] + sumsqs[j] - reference_t(2) * c_cell;
                c_cell = diff > reference_t(0) ? diff.sqrt() : reference_t(0);
            }
        }

        // Run kernel being tested
        backend.call(symmetric_fn, a.raw_values_data(), n, k, stride, c.raw_values_data(), c_stride, 0, n);
        synchronize(backend, stats);

        // Values on and above the diagonal, with nothing written below it
        for (std::size_t i = 0; i < n; i++)
            for (std::size_t j = i; j < n; j++)
                accumulate_euclidean<accumulation_>(stats, c[i * n + j], c_ref[i * n + j],
                                                    reference_t(sumsqs[i] + sumsqs[j]), k);
        expect_symmetric_untouched<result_t>(stats, c, n, c_stride, 0, n);
    }
    return stats;
}

#pragma endregion Spatial Distances

#pragma region Attention

/** Ragged bidirectional attention over @c attention_bidirectional_cases against the serial backend:
 *  a segment mix with a zero-length pad, one spanning two 512-key panels, and a 1000-key segment,
 *  packed in two task windows. */
template <typename scalar_type_, typename backend_type_ = host_backend_t,
          attention_weights_t weights_ = attention_weights_t::unquantized_k, typename pack_size_kernel_type_,
          typename pack_kernel_type_, typename attention_kernel_type_>
error_stats_t test_attention_bidirectional_packed(pack_size_kernel_type_ packed_size_fn, pack_kernel_type_ pack_fn,
                                                  attention_kernel_type_ attention_fn) {
    using scalar_t = scalar_type_;
    using result_t = typename scalar_t::attention_result_t;
    using scalars_t = nk::vector<scalar_t, typename backend_type_::template allocator<scalar_t>>;
    using results_t = nk::vector<result_t, typename backend_type_::template allocator<result_t>>;
    using bytes_t = nk::vector<char, typename backend_type_::template allocator<char>>;

    backend_type_ backend;
    error_stats_t stats(attention_family(weights_));
    std::mt19937 generator(global_config.seed);

    auto const segments = make_attention_segments<backend_type_>({60, 130, 0, 33, 600, 1000},
                                                                 {60, 130, 0, 33, 600, 24});
    std::vector<attention_bidirectional_case_t> const cases = attention_bidirectional_cases();

    for (auto start = test_start_time(); within_time_budget(start);) {
        for (attention_bidirectional_case_t const &test_case : cases) {
            attention_layout_t const layout {attention_key_value_heads_k * test_case.group, attention_key_value_heads_k,
                                             test_case.depth, 0.05f};
            std::size_t const query_stride_bytes = layout.query_width() * sizeof(scalar_t),
                              key_value_stride_bytes = layout.key_value_width() * sizeof(scalar_t),
                              output_stride_bytes = layout.query_width() * sizeof(result_t);
            std::size_t const total_tasks = segments.count() * layout.head_count;
            auto queries = scalars_t::try_zeros(segments.query_tokens() * layout.query_width());
            auto keys = scalars_t::try_zeros(segments.key_tokens() * layout.key_value_width()),
                 values = scalars_t::try_zeros(segments.key_tokens() * layout.key_value_width());
            fill_random(generator, queries), fill_random(generator, keys), fill_random(generator, values);

            auto key_value_packed = bytes_t::try_zeros(packed_size_fn(
                layout.key_value_head_count, layout.depth, segments.lengths.values_data(), segments.count()));
            // Run kernel being tested: pack in two windows, then attention over the whole task grid
            pack_attention_in_two_windows(backend, pack_fn, keys, values, segments, layout, key_value_packed);
            auto output = results_t::try_zeros(segments.query_tokens() * layout.query_width());
            backend.call(attention_fn, queries.raw_values_data(), key_value_packed.raw_values_data(),
                         output.raw_values_data(), layout.head_count, layout.key_value_head_count, layout.depth,
                         segments.query_offsets.values_data(), query_stride_bytes, output_stride_bytes, layout.scale, 0,
                         total_tasks);
            synchronize(backend, stats);

            auto key_value_reference = make_vector<char>(nk::attention_pack_size<scalar_t, nk::no_simd_k>(
                layout.key_value_head_count, layout.depth, segments.lengths.values_data(), segments.count()));
            // Compute reference through the serial backend via the C++ wrappers
            nk::attention_pack<scalar_t, nk::no_simd_k>(
                keys.values_data(), values.values_data(), layout.key_value_head_count, layout.depth,
                segments.key_offsets.values_data(), segments.lengths.values_data(), segments.count(),
                key_value_stride_bytes, key_value_stride_bytes, key_value_reference.raw_values_data());
            auto reference = make_vector<result_t>(segments.query_tokens() * layout.query_width());
            nk::attention_bidirectional_packed<scalar_t, result_t, nk::no_simd_k>(
                queries.values_data(), key_value_reference.raw_values_data(), reference.values_data(),
                layout.head_count, layout.key_value_head_count, layout.depth, segments.query_offsets.values_data(),
                query_stride_bytes, output_stride_bytes, layout.scale);

            accumulate_attention<weights_>(stats, output, reference, values);
        }
    }
    return stats;
}

/** Ragged causal attention over @c attention_causal_cases, against @c reference_by_rows. The pack
 *  runs in two task windows, and two attention task windows cover the grid, the second one relying
 *  on @c task_count clipping. */
template <typename scalar_type_, typename backend_type_ = host_backend_t,
          attention_weights_t weights_ = attention_weights_t::unquantized_k, typename pack_size_kernel_type_,
          typename pack_kernel_type_, typename attention_kernel_type_>
error_stats_t test_attention_causal_packed(pack_size_kernel_type_ packed_size_fn, pack_kernel_type_ pack_fn,
                                           attention_kernel_type_ attention_fn) {
    using scalar_t = scalar_type_;
    using result_t = typename scalar_t::attention_result_t;
    using scalars_t = nk::vector<scalar_t, typename backend_type_::template allocator<scalar_t>>;
    using results_t = nk::vector<result_t, typename backend_type_::template allocator<result_t>>;
    using bytes_t = nk::vector<char, typename backend_type_::template allocator<char>>;

    backend_type_ backend;
    error_stats_t stats(attention_family(weights_));
    std::mt19937 generator(global_config.seed);

    std::size_t const unbounded_window = static_cast<std::size_t>(-1);
    std::vector<attention_causal_case_t> const cases = attention_causal_cases();

    for (auto start = test_start_time(); within_time_budget(start);) {
        for (attention_causal_case_t const &test_case : cases) {
            auto const segments = make_attention_segments<backend_type_>( // long block, pad without keys, decode
                {test_case.main_length, 0, 33}, {attention_causal_queries(test_case.main_length), 2, 1});
            attention_layout_t const layout {attention_key_value_heads_k * test_case.group, attention_key_value_heads_k,
                                             test_case.depth, 0.05f};
            std::size_t const query_stride_bytes = layout.query_width() * sizeof(scalar_t),
                              output_stride_bytes = layout.query_width() * sizeof(result_t);
            std::size_t const first_window_tasks = segments.count() * layout.head_count / 2;

            auto queries = scalars_t::try_zeros(segments.query_tokens() * layout.query_width());
            auto keys = scalars_t::try_zeros(segments.key_tokens() * layout.key_value_width()),
                 values = scalars_t::try_zeros(segments.key_tokens() * layout.key_value_width());
            fill_random(generator, queries), fill_random(generator, keys), fill_random(generator, values);

            auto key_value_packed = bytes_t::try_zeros(packed_size_fn(
                layout.key_value_head_count, layout.depth, segments.lengths.values_data(), segments.count()));
            auto output = results_t::try_zeros(segments.query_tokens() * layout.query_width());
            pack_attention_in_two_windows(backend, pack_fn, keys, values, segments, layout, key_value_packed);
            backend.call(attention_fn, queries.raw_values_data(), key_value_packed.raw_values_data(),
                         output.raw_values_data(), layout.head_count, layout.key_value_head_count, layout.depth,
                         segments.query_offsets.values_data(), query_stride_bytes, output_stride_bytes, layout.scale,
                         test_case.diagonal_offset, test_case.window, 0, first_window_tasks);
            backend.call(attention_fn, queries.raw_values_data(), key_value_packed.raw_values_data(),
                         output.raw_values_data(), layout.head_count, layout.key_value_head_count, layout.depth,
                         segments.query_offsets.values_data(), query_stride_bytes, output_stride_bytes, layout.scale,
                         test_case.diagonal_offset, test_case.window, first_window_tasks, unbounded_window);
            synchronize(backend, stats);

            auto const reference = reference_by_rows(queries, keys, values, segments, layout, test_case.diagonal_offset,
                                                     test_case.window);
            accumulate_attention<weights_>(stats, output, reference, values);
        }
    }
    return stats;
}

#pragma endregion Attention

} // namespace ashvardanian::numkong::test

#endif // NK_TEST_CROSS_CUH
