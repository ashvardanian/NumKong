/**
 *  @file test/cross.hpp
 *  @author Ash Vardanian
 *  @date January 14, 2025
 *  @brief Backend-neutral cross-kernel scenarios: batched dots, spatial and set distances, and
 *      ragged attention windows.
 *
 *  Every scenario is a template over the scalar type, its kernels, and a backend owning where
 *  kernel operands live, how a kernel is called, and when its results become readable: the backends
 *  of `harness.hpp`, @c host_backend_t by default. Every backend is held to the same
 *  `nk_*_error_bound` of each family. Set distances run on the host only. References always run the
 *  serial `nk::` templates on the host. Outputs start filled with @c canary_k bytes, so any stray
 *  write shows up.
 */
#pragma once
#ifndef NUMKONG_TEST_CROSS_HPP
#define NUMKONG_TEST_CROSS_HPP

#include <cmath>   // `std::ldexp`, `std::nextafter`
#include <cstdint> // `std::int64_t`, `std::uint64_t`
#include <cstring> // `std::memset`, `std::memcmp`

#include <initializer_list> // `std::initializer_list`
#include <random>           // `std::uniform_real_distribution`
#include <vector>           // `std::vector`

#include "numkong/attention.hpp" // `nk::attention_packed`, `nk::attention_pack`
#include "numkong/cast.h"        // `nk_cast_serial`
#include "numkong/dots.hpp"      // `nk::dots_packed`, `nk::dots_symmetric`
#include "numkong/matrix.hpp"    // `nk::dots_pack_size`, `nk::dots_pack`
#include "numkong/spatials.h"    // `nk_angulars_packed_*`, `nk_euclideans_packed_*`

#include "harness.hpp"

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

/** A vector of @p value_type_ in the memory the kernels of @p backend_type_ read and write. */
template <typename value_type_, typename backend_type_>
using backend_vector = nk::vector<value_type_, typename backend_type_::template allocator<value_type_>>;

/** A zeroed vector of @p count values in @p backend memory, empty when the allocation fails. */
template <typename value_type_, typename backend_type_>
[[nodiscard]] backend_vector<value_type_, backend_type_> make_vector(backend_type_ const &backend, std::size_t count) {
    return backend_vector<value_type_, backend_type_>::zeros(count, allocator_of<value_type_>(backend)).value;
}

/** The bytes @p packed_size_fn asks to pack its @p arguments into, failing @p stats when it has no
 *  kernel. Pack sizes are host arithmetic on every backend, so they run here. */
template <typename packed_size_kernel_type_, typename... arguments_types_>
std::size_t pack_size_bytes(error_stats_t &stats, packed_size_kernel_type_ packed_size_fn,
                            arguments_types_... arguments) noexcept {
    nk_size_t bytes = 0;
    stats.expect(packed_size_fn(arguments..., &bytes));
    return bytes;
}

/** Waits for the calls queued on @p backend, the first that failed having returned @p status, and
 *  expects both to succeed: true when their results are readable. */
template <typename backend_type_>
bool expect_completed(error_stats_t &stats, backend_type_ &backend, nk_status_t status) noexcept {
    nk_status_t const synchronization_status = backend.synchronize();
    stats.expect(status);
    stats.expect(synchronization_status);
    return status == nk_success_k && synchronization_status == nk_success_k;
}

#pragma endregion Backend Policy

#pragma region Tolerances

/** The family an attention output is judged in: the scale threshold, or that threshold floored by
 *  the weights. */
constexpr comparison_family_t attention_family(attention_weights_t weights) noexcept {
    return weights == attention_weights_t::unquantized_k ? comparison_family_t::normalized_reduction_k
                                                         : comparison_family_t::bounded_k;
}

/** The @p kind_ distance of two vectors from their @p dot and squared norms @p first and @p second:
 *  1 − dot / √(‖a‖² · ‖b‖²) for angular, √max(0, ‖a‖² + ‖b‖² − 2 · dot) for euclidean. */
template <nk_kernel_kind_t kind_, typename reference_type_>
reference_type_ spatial_distance(reference_type_ dot, reference_type_ first, reference_type_ second) {
    reference_type_ const zero(0);
    if constexpr (kind_ == nk_kernel_angular_k) {
        reference_type_ const product = first * second;
        return product > zero ? reference_type_(1) - dot * product.rsqrt() : zero;
    }
    else {
        reference_type_ const squared = first + second - reference_type_(2) * dot;
        return squared > zero ? squared.sqrt() : zero;
    }
}

/** Folds one @p kind_ distance into @p stats, within a tolerance over the @p depth terms of the
 *  dots beneath it, each adding up to @c term_error_bound: absolute for angular, and relative to
 *  the @p squared_norms of both vectors for euclidean, whose square it bounds. Both norms and the
 *  dot err by depth + 1 terms relative to ‖a‖ · ‖b‖, and the finish rounds once more. */
template <nk_kernel_kind_t kind_, typename result_type_, typename reference_type_>
void accumulate_spatial(error_stats_t &stats, result_type_ result, reference_type_ reference,
                        reference_type_ squared_norms, std::size_t depth) {
    double const tolerance = 4 * static_cast<double>(depth + 2) * stats.term_error_bound;
    double const expected = static_cast<double>(reference);
    if constexpr (kind_ == nk_kernel_angular_k) stats.accumulate_bounded(result, expected, tolerance);
    else {
        double const scale = static_cast<double>(squared_norms) > 0 ? static_cast<double>(squared_norms) : 1;
        double const sum = std::max(static_cast<double>(result) + expected, std::numeric_limits<double>::min());
        stats.accumulate_bounded(result, expected, tolerance * scale / sum);
    }
}

/** Decodes @p rows rows of @p depth dimensions, @p row_stride_values values apart, into F64. */
template <typename scalar_type_, typename allocator_type_>
std::vector<double> decode_rows(nk::vector<scalar_type_, allocator_type_> const &matrix, std::size_t rows,
                                std::size_t depth, std::size_t row_stride_values) {
    std::vector<double> decoded(rows * depth);
    for (std::size_t row = 0; row < rows; row++)
        if (nk_cast_serial(matrix.raw_values_data() + row * row_stride_values, scalar_type_::dtype(),
                           decoded.data() + row * depth, nk_f64_k, depth, nullptr) != nk_success_k)
            std::fill_n(decoded.data() + row * depth, depth, std::numeric_limits<double>::quiet_NaN());
    return decoded;
}

/** Folds every attention output into @p stats: the scale threshold of the largest reference,
 *  floored by @p weights_. */
template <attention_weights_t weights_, typename output_vector_, typename reference_vector_, typename values_vector_>
void accumulate_attention(settings_t const &settings, error_stats_t &stats, output_vector_ const &output,
                          reference_vector_ const &reference, values_vector_ const &values) {
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
        double const bound = std::max(settings.scale_threshold * largest_reference,
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

    /** Keys per segment. */
    backend_vector<nk_u32_t, backend_type_> lengths;

    /** First key row of every segment, then the key total. */
    backend_vector<nk_u32_t, backend_type_> key_offsets;

    /** First query row of every segment, then the query total. */
    backend_vector<nk_u32_t, backend_type_> query_offsets;

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

/** Lays out segments of @p lengths keys, matched with the entries of @p query_counts queries, in
 *  memory of @p backend. */
template <typename backend_type_>
attention_segments<backend_type_> make_attention_segments(backend_type_ const &backend,
                                                          std::initializer_list<nk_u32_t> lengths,
                                                          std::initializer_list<nk_u32_t> query_counts) {
    attention_segments<backend_type_> segments {make_vector<nk_u32_t>(backend, lengths.size()),
                                                make_vector<nk_u32_t>(backend, lengths.size() + 1),
                                                make_vector<nk_u32_t>(backend, lengths.size() + 1)};
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

/** Keys a query sees before and after its own position: every key, the default. */
constexpr std::size_t attention_all_keys_k = std::numeric_limits<std::size_t>::max();

/** One attention case: the long segment's key count, the GQA group, the head depth, and the band
 *  of keys each query sees around its position. */
struct attention_case_t {

    /** Keys of the long segment, which segments of other query-to-key offsets follow. */
    nk_u32_t main_length;

    /** Query heads per key-value head. */
    std::size_t group;

    /** Elements per head. */
    std::size_t depth;

    /** Keys visible before each query's position. */
    std::size_t keys_before;

    /** Keys visible after each query's position. */
    std::size_t keys_after;

    /** Whether every key of a segment is visible to all of its queries. */
    bool unmasked() const noexcept { return keys_before == attention_all_keys_k && keys_after == attention_all_keys_k; }
};

/** Queries of the long segment: about half its keys, capped to keep the per-row reference cheap. */
inline nk_u32_t attention_main_queries(nk_u32_t main_length) noexcept {
    return std::min<nk_u32_t>(main_length / 2 + 2, 24);
}

/** Bidirectional cases at depths on both sides of every panel edge and GQA 1:1, 4:1 and 8:1; then
 *  panel edges and a 1000-key prefill in the main length under causal, sliding-window, diagonal,
 *  upper-triangle and two-sided bands, at odd depths and GQA. */
inline std::vector<attention_case_t> attention_cases() {
    std::size_t const all = attention_all_keys_k;
    std::vector<attention_case_t> cases;
    for (std::size_t depth : {1ul, 64ul, 65ul, 127ul, 128ul, 129ul, 255ul, 257ul})
        cases.push_back({1000, 2, depth, all, all});
    for (std::size_t group : {1ul, 4ul, 8ul}) cases.push_back({1000, group, 128, all, all});
    nk_diagonal_band_t const bands[] = {{all, 0}, {0, 0},   {6, 0}, {30, 0}, {32, 0},
                                        {510, 0}, {512, 0}, {3, 5}, {0, all}};
    for (nk_u32_t main_length : {1u, 31u, 32u, 33u, 513u, 1000u})
        for (std::size_t depth : {1ul, 65ul, 128ul, 257ul})
            for (nk_diagonal_band_t const band : bands)
                cases.push_back({main_length, 2, depth, band.subdiagonals, band.superdiagonals});
    for (std::size_t group : {1ul, 4ul, 8ul})
        for (std::size_t keys_before : {all, std::size_t(64)}) cases.push_back({1000, group, 128, keys_before, 0});
    return cases;
}

/** Packs every segment through @p pack_fn in two task windows, the later one first so neither may
 *  rely on the other, that one clipped from past the grid. */
template <typename backend_type_, typename pack_kernel_type_, typename scalar_vector_type_,
          typename packed_vector_type_>
nk_status_t pack_attention_in_two_windows(backend_type_ &backend, pack_kernel_type_ pack_fn,
                                          scalar_vector_type_ const &keys, scalar_vector_type_ const &values,
                                          attention_segments<backend_type_> const &segments,
                                          attention_layout_t const &layout, packed_vector_type_ &key_value_packed) {
    using scalar_t = typename scalar_vector_type_::value_type;
    std::size_t const stride = layout.key_value_width() * sizeof(scalar_t);
    std::size_t const tasks = segments.count() * layout.key_value_head_count;
    std::size_t const windows[2][2] = {{1, tasks + 7}, {0, 1}};
    for (auto const &window : windows) {
        nk_status_t const status = backend.call(
            pack_fn, keys.raw_values_data(), values.raw_values_data(), layout.key_value_head_count, layout.depth,
            segments.key_offsets.values_data(), segments.lengths.values_data(), segments.count(), stride, stride,
            key_value_packed.raw_values_data(), window[0], window[1]);
        if (status != nk_success_k) return status;
    }
    return nk_success_k;
}

/** Keys of one segment that one query row sees, from @c begin up to @c end. */
struct attention_key_range_t {
    std::size_t begin, end;
};

/** The keys of a segment of @p query_count queries aligned to the end of its @p key_count keys that
 *  query @p row sees, @p keys_before and @p keys_after around its position, or an empty range. */
inline attention_key_range_t attention_visible_keys(std::size_t row, std::size_t query_count, std::size_t key_count,
                                                    std::size_t keys_before, std::size_t keys_after) noexcept {
    // No position is more than every query and key away from another, so wider bands reach all keys
    std::size_t const reach = query_count + key_count;
    std::int64_t const position = static_cast<std::int64_t>(row + key_count) - static_cast<std::int64_t>(query_count);
    std::int64_t const before = static_cast<std::int64_t>(std::min(keys_before, reach)),
                       after = static_cast<std::int64_t>(std::min(keys_after, reach)),
                       length = static_cast<std::int64_t>(key_count);
    return {static_cast<std::size_t>(std::clamp<std::int64_t>(position - before, 0, length)),
            static_cast<std::size_t>(std::clamp<std::int64_t>(position + after + 1, 0, length))};
}

/** K and V of the @p count segments of @p lengths keys from the rows at @p key_offsets, packed into
 *  host memory by the serial `nk::` wrappers. */
template <typename scalar_type_, typename allocator_type_>
nk::vector<char> serial_attention_pack(error_stats_t &stats, nk::vector<scalar_type_, allocator_type_> const &keys,
                                       nk::vector<scalar_type_, allocator_type_> const &values,
                                       attention_layout_t const &layout, nk_u32_t const *key_offsets,
                                       nk_u32_t const *lengths, std::size_t count) {
    std::size_t const stride = layout.key_value_width() * sizeof(scalar_type_);
    std::size_t key_tokens = 0;
    for (std::size_t segment = 0; segment < count; segment++) key_tokens += lengths[segment];
    auto const size = nk::attention_pack_size<scalar_type_>(layout.key_value_head_count, layout.depth, key_tokens,
                                                            count, 0);
    stats.expect(size.status);
    auto packed = make_vector<char>(size.value);
    stats.expect(nk::attention_pack<scalar_type_>(keys.values_data(), values.values_data(), layout.key_value_head_count,
                                                  layout.depth, key_offsets, lengths, count, stride, stride,
                                                  packed.raw_values_data(), 0, static_cast<std::size_t>(-1), 0));
    return packed;
}

/** A forward reference: output rows and their natural log-sum-exps, −∞ on rows without keys. */
struct attention_reference_t {
    nk::vector<f32_t> output, log_sum_exp;
};

/** Unmasked reference: the serial kernel over a serial pack of every segment, every key visible. */
template <typename scalar_type_, typename allocator_type_, typename backend_type_>
attention_reference_t reference_attention_unmasked(error_stats_t &stats,
                                                   nk::vector<scalar_type_, allocator_type_> const &queries,
                                                   nk::vector<scalar_type_, allocator_type_> const &keys,
                                                   nk::vector<scalar_type_, allocator_type_> const &values,
                                                   attention_segments<backend_type_> const &segments,
                                                   attention_layout_t const &layout) {
    std::size_t const query_stride = layout.query_width() * sizeof(scalar_type_),
                      output_stride = layout.query_width() * sizeof(f32_t);
    attention_reference_t reference {
        .output = make_vector<f32_t>(segments.query_tokens() * layout.query_width()),
        .log_sum_exp = make_vector<f32_t>(segments.query_tokens() * layout.head_count),
    };
    auto const key_value_packed_reference = serial_attention_pack(stats, keys, values, layout,
                                                                  segments.key_offsets.values_data(),
                                                                  segments.lengths.values_data(), segments.count());
    stats.expect(nk::attention_packed<scalar_type_, f32_t>(
        queries.values_data(), key_value_packed_reference.raw_values_data(), reference.output.values_data(),
        reference.log_sum_exp.values_data(), layout.head_count, layout.key_value_head_count, layout.depth,
        segments.query_offsets.values_data(), query_stride, output_stride, layout.scale, attention_all_keys_k,
        attention_all_keys_k, 0, attention_all_keys_k, 0));
    return reference;
}

/** Masked reference: the serial kernel per query row with every key visible, over a serial pack of
 *  exactly the keys @c attention_visible_keys admits, leaving zeros and −∞ on every row that sees
 *  no key at all. */
template <typename scalar_type_, typename allocator_type_, typename backend_type_>
attention_reference_t reference_attention_masked(error_stats_t &stats,
                                                 nk::vector<scalar_type_, allocator_type_> const &queries,
                                                 nk::vector<scalar_type_, allocator_type_> const &keys,
                                                 nk::vector<scalar_type_, allocator_type_> const &values,
                                                 attention_segments<backend_type_> const &segments,
                                                 attention_layout_t const &layout, std::size_t keys_before,
                                                 std::size_t keys_after) {
    std::size_t const query_stride = layout.query_width() * sizeof(scalar_type_),
                      output_stride = layout.query_width() * sizeof(f32_t);
    nk_u32_t const single_query_offsets[2] = {0, 1};
    attention_reference_t reference {
        .output = make_vector<f32_t>(segments.query_tokens() * layout.query_width()),
        .log_sum_exp = make_vector<f32_t>(segments.query_tokens() * layout.head_count),
    };
    stats.expect(reference.log_sum_exp.fill(f32_t(-std::numeric_limits<float>::infinity())));
    for (std::size_t segment = 0; segment < segments.count(); segment++) {
        std::size_t const length = segments.lengths.values_data()[segment];
        for (std::size_t row = 0; row < segments.queries(segment); row++) {
            auto const [key_begin, key_end] = attention_visible_keys(row, segments.queries(segment), length,
                                                                     keys_before, keys_after);
            if (key_begin == key_end) continue;

            std::size_t const first_key = segments.key_offsets.values_data()[segment];
            nk_u32_t const visible_offsets[2] = {static_cast<nk_u32_t>(first_key + key_begin),
                                                 static_cast<nk_u32_t>(first_key + key_end)};
            nk_u32_t const visible_length = static_cast<nk_u32_t>(key_end - key_begin);
            auto const key_value_packed_reference = serial_attention_pack(stats, keys, values, layout, visible_offsets,
                                                                          &visible_length, 1);
            std::size_t const query_row = segments.query_offsets.values_data()[segment] + row;
            stats.expect(nk::attention_packed<scalar_type_, f32_t>(
                queries.values_data() + query_row * layout.query_width(), key_value_packed_reference.raw_values_data(),
                reference.output.values_data() + query_row * layout.query_width(),
                reference.log_sum_exp.values_data() + query_row * layout.head_count, layout.head_count,
                layout.key_value_head_count, layout.depth, single_query_offsets, query_stride, output_stride,
                layout.scale, attention_all_keys_k, attention_all_keys_k, 0, attention_all_keys_k, 0));
        }
    }
    return reference;
}

/** Attention and its gradients in F64 over dense rows: the output and natural log-sum-exp a forward
 *  pass hands its backward, −∞ for rows that see no keys, the query, key and value gradients of an
 *  output gradient, and the sums of the absolute values of each gradient's terms, which scale its
 *  rounding even where the terms cancel, as dP − D does for a row of one key. */
struct attention_gradients_t {
    std::vector<double> output, log_sum_exp, query_gradient, key_gradient, value_gradient;
    std::vector<double> query_magnitude, key_magnitude, value_magnitude;
};

/** Computes @c attention_gradients_t one row at a time, straight from the definition. */
template <typename backend_type_>
attention_gradients_t reference_attention_gradients(std::vector<double> const &queries, std::vector<double> const &keys,
                                                    std::vector<double> const &values,
                                                    std::vector<double> const &output_gradient,
                                                    attention_segments<backend_type_> const &segments,
                                                    attention_layout_t const &layout, std::size_t keys_before,
                                                    std::size_t keys_after) {
    std::size_t const depth = layout.depth, group = layout.head_count / layout.key_value_head_count;
    std::size_t const query_count = segments.query_tokens() * layout.query_width(),
                      key_count = segments.key_tokens() * layout.key_value_width();
    attention_gradients_t result {
        .output = std::vector<double>(query_count),
        .log_sum_exp = std::vector<double>(segments.query_tokens() * layout.head_count,
                                           -std::numeric_limits<double>::infinity()),
        .query_gradient = std::vector<double>(query_count),
        .key_gradient = std::vector<double>(key_count),
        .value_gradient = std::vector<double>(key_count),
        .query_magnitude = std::vector<double>(query_count),
        .key_magnitude = std::vector<double>(key_count),
        .value_magnitude = std::vector<double>(key_count),
    };
    struct dot_t {
        double sum, magnitude;
    };
    auto const dot = [depth](double const *a, double const *b) {
        dot_t products {0, 0};
        for (std::size_t channel = 0; channel < depth; channel++)
            products.sum += a[channel] * b[channel], products.magnitude += std::fabs(a[channel] * b[channel]);
        return products;
    };
    std::vector<double> weights;
    for (std::size_t segment = 0; segment < segments.count(); segment++)
        for (std::size_t head = 0; head < layout.head_count; head++)
            for (std::size_t row = 0; row < segments.queries(segment); row++) {
                auto const [key_begin, key_end] = attention_visible_keys(
                    row, segments.queries(segment), segments.lengths.values_data()[segment], keys_before, keys_after);
                if (key_begin == key_end) continue;
                std::size_t const token = segments.query_offsets.values_data()[segment] + row;
                std::size_t const query_offset = token * layout.query_width() + head * depth;
                std::size_t const first_key_offset = (segments.key_offsets.values_data()[segment] + key_begin) *
                                                         layout.key_value_width() +
                                                     head / group * depth;
                double const *query = queries.data() + query_offset, *gradient = output_gradient.data() + query_offset;
                double *output = result.output.data() + query_offset;

                weights.resize(key_end - key_begin);
                for (std::size_t key = 0; key < weights.size(); key++)
                    weights[key] = layout.scale *
                                   dot(query, keys.data() + first_key_offset + key * layout.key_value_width()).sum;
                double const maximum = *std::max_element(weights.begin(), weights.end());
                double sum = 0;
                for (double &weight : weights) weight = std::exp(weight - maximum), sum += weight;
                result.log_sum_exp[token * layout.head_count + head] = maximum + std::log(sum);
                for (std::size_t key = 0; key < weights.size(); key++) {
                    weights[key] /= sum;
                    double const *value = values.data() + first_key_offset + key * layout.key_value_width();
                    for (std::size_t channel = 0; channel < depth; channel++)
                        output[channel] += weights[key] * value[channel];
                }

                dot_t const output_products = dot(gradient, output);
                for (std::size_t key = 0; key < weights.size(); key++) {
                    std::size_t const key_offset = first_key_offset + key * layout.key_value_width();
                    dot_t const value_products = dot(gradient, values.data() + key_offset);
                    double const score_gradient = weights[key] * (value_products.sum - output_products.sum) *
                                                  layout.scale;
                    double const score_magnitude = weights[key] *
                                                   (value_products.magnitude + output_products.magnitude) *
                                                   layout.scale;
                    for (std::size_t channel = 0; channel < depth; channel++) {
                        double const key_element = keys[key_offset + channel];
                        result.query_gradient[query_offset + channel] += score_gradient * key_element;
                        result.key_gradient[key_offset + channel] += score_gradient * query[channel];
                        result.value_gradient[key_offset + channel] += weights[key] * gradient[channel];
                        result.query_magnitude[query_offset + channel] += score_magnitude * std::fabs(key_element);
                        result.key_magnitude[key_offset + channel] += score_magnitude * std::fabs(query[channel]);
                        result.value_magnitude[key_offset + channel] += weights[key] * std::fabs(gradient[channel]);
                    }
                }
            }
    return result;
}

#pragma endregion Operands

#pragma region Block Scales

template <typename scalar_type_>
struct cref_of_ {
    struct none {};
    using type = none;
};

template <typename scalar_type_>
    requires requires { typename scalar_type_::cref_t; }
struct cref_of_<scalar_type_> {
    using type = typename scalar_type_::cref_t;
};

/** Bytes a padded row stride adds: 16, or a whole block of codes when that is more, keeping scale
 *  rows a whole number of blocks apart. */
template <typename scalar_type_>
constexpr std::size_t stride_padding_v = std::max<std::size_t>(
    16, nk_block_scaled_format_of_dtype(scalar_type_::dtype()).block_bytes);

/** Bytes an odd row stride adds: one value, or a whole block of codes for block-scaled formats. */
template <typename scalar_type_>
constexpr std::size_t stride_step_v =
    nk_block_scaled_format_of_dtype(scalar_type_::dtype()).block_bytes
        ? nk_block_scaled_format_of_dtype(scalar_type_::dtype()).block_bytes
        : sizeof(typename nk::type_for<nk_block_scaled_format_of_dtype(scalar_type_::dtype()).element_dtype>::type);

/** The scales of one operand in @p backend_type_'s memory: one byte per block of each row, rows
 *  @c stride / block_bytes apart, using finite nonzero UE4M3 scales or UE8M0 powers of two within
 *  three binades of one, plus for NVFP4 one non-unit tensor scale. @c operand wraps codes into what
 *  kernels take: codes for plain dtypes, else a reference to them and these scales. */
template <typename scalar_type_, typename backend_type_>
struct operand_scales {
    static constexpr nk_block_scaled_format_t format = nk_block_scaled_format_of_dtype(scalar_type_::dtype());
    using scale_t = typename nk::type_for<format.block_size ? format.scale_dtype : nk_u8_k>::type;
    using blocks_t = nk::tensor<scale_t, typename backend_type_::template allocator<scale_t>, 2>;

    blocks_t blocks;
    backend_vector<f32_t, backend_type_> tensor_scale;
    std::size_t scale_stride = 0;
    mutable typename cref_of_<scalar_type_>::type reference {};

    /** The operand of the rows from @p row on, whose codes @p codes points at. */
    template <typename codes_pointer_type_>
    auto operand(codes_pointer_type_ codes, std::size_t row = 0) const noexcept {
        if constexpr (!format.block_size) return codes;
        else {
            reference.elements = reinterpret_cast<decltype(reference.elements)>(codes);
            reference.scales = reinterpret_cast<decltype(reference.scales)>(
                reinterpret_cast<char const *>(blocks.data()) + row * scale_stride);
            if constexpr (format.tensor_scale_dtype == nk_f32_k)
                reference.tensor_scale = tensor_scale.raw_values_data();
            return &reference;
        }
    }
};

/** Scales for @p rows rows of @p depth dimensions whose codes lie @p stride bytes apart. */
template <typename scalar_type_, typename backend_type_>
auto random_scales(backend_type_ &backend, std::mt19937 &generator, std::size_t rows, std::size_t depth,
                   std::size_t stride, float tensor_scale) {
    using operand_t = operand_scales<scalar_type_, backend_type_>;
    using scale_t = typename operand_t::scale_t;
    operand_t operand;
    if constexpr (operand_t::format.block_size) {
        std::size_t const blocks = depth / operand_t::format.block_size,
                          scale_stride = stride / operand_t::format.block_bytes;
        operand.scale_stride = scale_stride;
        operand.blocks = operand_t::blocks_t::zeros({rows, scale_stride}, allocator_of<scale_t>(backend)).value;
        std::uniform_int_distribution<int> exponents(-3, 3), finite_scales(1, 126);
        for (std::size_t row = 0; row != rows; ++row)
            for (std::size_t block = 0; block != blocks; ++block) {
                if constexpr (operand_t::format.scale_dtype == nk_ue4m3_k)
                    operand.blocks.data()[row * scale_stride + block] = scale_t::from_raw(finite_scales(generator));
                else
                    operand.blocks.data()[row * scale_stride + block] = scale_t(std::ldexp(1.0f, exponents(generator)));
            }
        if constexpr (operand_t::format.tensor_scale_dtype == nk_f32_k) {
            operand.tensor_scale = make_vector<f32_t>(backend, 1);
            operand.tensor_scale[0] = f32_t(tensor_scale);
        }
    }
    return operand;
}

#pragma endregion Block Scales

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

/** One case: C shaped @b [rows,columns] equals A shaped @b [rows,depth] times B transposed,
 *  shaped @b [columns,depth]. */
struct dots_packed_case_t {

    /** Rows of A and C. */
    std::size_t rows;

    /** Rows of B and columns of C. */
    std::size_t columns;

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
std::vector<dots_packed_case_t> dots_packed_cases(settings_t const &settings) {
    dots_strides_t const tight = dots_strides_t::tight_k, padded = dots_strides_t::padded_k;
    dots_operands_t const random = dots_operands_t::random_k;
    std::vector<dots_packed_case_t> cases {
        {settings.matrix_height, settings.matrix_width, settings.matrix_depth, tight, random},
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
                          vector_type_ &second, std::size_t second_stride_values, std::size_t rows, std::size_t columns,
                          std::size_t depth) {
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    std::size_t const half = depth / 2;
    for (std::size_t row = 0; row < rows; row++) {
        double *values = first.raw_values_data() + row * first_stride_values;
        for (std::size_t index = 0; index < half; index++) {
            values[index] = unit(generator);
            values[half + index] = -values[index] + values[index] * 0x1p-33 * unit(generator);
        }
    }
    for (std::size_t row = 0; row < columns; row++) {
        double *values = second.raw_values_data() + row * second_stride_values;
        for (std::size_t index = 0; index < half; index++) values[index] = values[half + index] = unit(generator);
    }
}

/** The largest error over every cell of @p computed, in ulp of the tracked @p reference, where
 *  @p computed is called with a row and a column. */
template <typename computed_type_, typename reference_vector_type_>
double worst_error_ulps(computed_type_ computed, reference_vector_type_ const &reference, std::size_t rows,
                        std::size_t columns) {
    double worst = 0;
    for (std::size_t row = 0; row < rows; row++)
        for (std::size_t column = 0; column < columns; column++) {
            double const expected = static_cast<double>(reference[row * columns + column].value);
            double const absolute = std::fabs(expected), ulp = std::nextafter(absolute, INFINITY) - absolute;
            worst = std::max(worst, std::fabs(computed(row, column) - expected) / ulp);
        }
    return worst;
}

/** Packed GEMM over @c dots_packed_cases against the serial `nk::` reference, with C stride padding
 *  left untouched. */
template <typename scalar_type_, typename backend_type_ = host_backend_t, typename pack_size_kernel_type_,
          typename pack_kernel_type_, typename dots_kernel_type_>
error_stats_t test_dots_packed(settings_t const &settings, backend_type_ backend, pack_size_kernel_type_ packed_size_fn,
                               pack_kernel_type_ pack_fn, dots_kernel_type_ dots_fn) {
    constexpr nk_block_scaled_format_t format = nk_block_scaled_format_of_dtype(scalar_type_::dtype());
    using scalar_t = typename nk::type_for<format.element_dtype>::type;
    using result_t = typename scalar_type_::dot_result_t;
    using reference_t = bounded_reference_for<scalar_type_, result_t>;

    error_stats_t stats(nk_dot_error_bound(scalar_type_::dtype()));
    std::mt19937 generator(settings.seed.value);
    std::size_t const dimensions_per_value = nk::dimensions_per_value<scalar_t>();
    std::size_t const depth_multiple = std::max<std::size_t>(format.block_size, dimensions_per_value);
    std::vector<dots_packed_case_t> const cases = dots_packed_cases<scalar_type_>(settings);

    for (time_point_t const deadline = steady_clock_t::now() + settings.time_limit_per_kernel;
         steady_clock_t::now() < deadline;) {
        for (dots_packed_case_t const &test_case : cases) {
            std::size_t const rows = test_case.rows, columns = test_case.columns;
            std::size_t const depth = nk::divide_round_up(test_case.depth, depth_multiple) * depth_multiple;
            std::size_t const row_bytes = depth / dimensions_per_value * sizeof(scalar_t);
            bool const padded = test_case.strides == dots_strides_t::padded_k;
            std::size_t const a_stride = backend.row_stride(row_bytes) + (padded ? stride_padding_v<scalar_type_> : 0);
            std::size_t b_stride = row_bytes + (padded ? stride_step_v<scalar_type_> : 0);
            if (padded && b_stride % 16 == 0) b_stride += stride_step_v<scalar_type_>;
            std::size_t const c_stride = (columns + (padded ? 3 : 0)) * sizeof(result_t);
            std::size_t const a_stride_values = a_stride / sizeof(scalar_t),
                              b_stride_values = b_stride / sizeof(scalar_t);

            auto a = make_vector<scalar_t>(backend, rows * a_stride_values * dimensions_per_value),
                 b = make_vector<scalar_t>(backend, columns * b_stride_values * dimensions_per_value);
            auto c = make_vector<result_t>(backend, rows * c_stride / sizeof(result_t));
            auto b_packed = make_vector<char>(backend, pack_size_bytes(stats, packed_size_fn, columns, depth));

            if constexpr (std::is_same_v<scalar_t, f64_t>) {
                if (test_case.operands == dots_operands_t::ill_conditioned_k)
                    fill_ill_conditioned(generator, a, a_stride_values, b, b_stride_values, rows, columns, depth);
                else fill_random(settings, generator, a), fill_random(settings, generator, b);
            }
            else fill_random(settings, generator, a), fill_random(settings, generator, b);
            if constexpr (scalar_type_::dtype() == nk_mxfp6e2m3_k || scalar_type_::dtype() == nk_mxfp6e3m2_k)
                if (rows == 1 && columns == 1) {
                    std::memset(a.raw_values_data(), 0x1F, row_bytes);
                    std::memset(b.raw_values_data(), 0x3F, row_bytes);
                }
            fill_padding_canary(a, rows, row_bytes, a_stride), fill_padding_canary(b, columns, row_bytes, b_stride);
            fill_canary(c);
            auto const a_scales = random_scales<scalar_type_>(backend, generator, rows, depth, a_stride, 1.5f),
                       b_scales = random_scales<scalar_type_>(backend, generator, columns, depth, b_stride, 0.75f);

            nk_status_t status = backend.call(pack_fn, b_scales.operand(b.raw_values_data()), columns, depth, b_stride,
                                              b_packed.raw_values_data(), 0, columns);
            if (status == nk_success_k)
                status = backend.call(dots_fn, a_scales.operand(a.raw_values_data()), b_packed.raw_values_data(),
                                      c.raw_values_data(), rows, columns, depth, a_stride, c_stride);
            if (!expect_completed(stats, backend, status)) return stats;

            std::vector<reference_t> c_reference(rows * columns);
            auto b_packed_reference = make_vector<char>(
                nk::dots_pack_size<scalar_type_>(columns, depth, no_tiers_k).value);
            stats.expect(nk::dots_pack<scalar_type_>(b_scales.operand(b.values_data()), columns, depth, b_stride,
                                                     b_packed_reference.raw_values_data(), no_tiers_k, nullptr));
            stats.expect(nk::dots_packed<scalar_type_, reference_t>(
                a_scales.operand(a.values_data()), b_packed_reference.raw_values_data(), c_reference.data(), rows,
                columns, depth, a_stride, columns * sizeof(reference_t), no_tiers_k, nullptr));

            for (std::size_t row = 0; row < rows; row++)
                for (std::size_t column = 0; column < columns; column++)
                    stats.accumulate(c[row * c_stride / sizeof(result_t) + column],
                                     c_reference[row * columns + column]);
            expect_padding_untouched(stats, c, rows, columns * sizeof(result_t), c_stride);

            // F64 dots compensate their sums, which the ill-conditioned case tells from plain F64 ones
            if constexpr (std::is_same_v<scalar_t, f64_t>)
                if (test_case.operands == dots_operands_t::ill_conditioned_k) {
                    std::vector<double> const a_rows = decode_rows(a, rows, depth, a_stride_values),
                                              b_rows = decode_rows(b, columns, depth, b_stride_values);
                    auto const naive = [&](std::size_t row, std::size_t column) {
                        double sum = 0;
                        for (std::size_t index = 0; index < depth; index++)
                            sum += a_rows[row * depth + index] * b_rows[column * depth + index];
                        return sum;
                    };
                    auto const computed = [&](std::size_t row, std::size_t column) {
                        return static_cast<double>(c[row * c_stride / sizeof(result_t) + column]);
                    };
                    stats.expect(worst_error_ulps(naive, c_reference, rows, columns) >= 1e3,
                                 "the ill-conditioned case is beyond 1000 ulp for plain F64");
                    stats.expect(worst_error_ulps(computed, c_reference, rows, columns) <= 4,
                                 "compensated F64 dots stay within 4 ulp on the ill-conditioned case");
                }
        }
    }
    return stats;
}

/** @c test_dots_packed on the backend selected by @p settings. */
template <typename scalar_type_, typename backend_type_ = host_backend_t, typename pack_size_kernel_type_,
          typename pack_kernel_type_, typename dots_kernel_type_>
error_stats_t test_dots_packed(settings_t const &settings, pack_size_kernel_type_ packed_size_fn,
                               pack_kernel_type_ pack_fn, dots_kernel_type_ dots_fn) {
    return test_dots_packed<scalar_type_, backend_type_>(settings, make_backend<backend_type_>(settings),
                                                         packed_size_fn, pack_fn, dots_fn);
}

/** The packed B layout over the widths and depths of @c dots_packed_cases: packing two column
 *  windows equals packing all columns at once byte for byte, and @c packed_shape_fn_ reads back the
 *  columns and depth the pack was given. */
template <typename scalar_type_, typename backend_type_, auto packed_size_fn_, auto packed_shape_fn_, auto pack_fn_>
error_stats_t test_dots_pack_layout(settings_t const &settings, backend_type_ backend) {
    constexpr nk_block_scaled_format_t format = nk_block_scaled_format_of_dtype(scalar_type_::dtype());
    using scalar_t = typename nk::type_for<format.element_dtype>::type;

    error_stats_t stats(comparison_family_t::exact_k);
    std::mt19937 generator(settings.seed.value);
    std::size_t const dimensions_per_value = nk::dimensions_per_value<scalar_t>();
    std::size_t const depth_multiple = std::max<std::size_t>(format.block_size, dimensions_per_value);
    std::vector<dots_packed_case_t> const cases = dots_packed_cases<scalar_type_>(settings);

    for (time_point_t const deadline = steady_clock_t::now() + settings.time_limit_per_kernel;
         steady_clock_t::now() < deadline;) {
        for (dots_packed_case_t const &test_case : cases) {
            std::size_t const columns = test_case.columns;
            std::size_t const depth = nk::divide_round_up(test_case.depth, depth_multiple) * depth_multiple;
            std::size_t const row_bytes = depth / dimensions_per_value * sizeof(scalar_t);
            std::size_t const packed_size = pack_size_bytes(stats, packed_size_fn_, columns, depth);
            auto b = make_vector<scalar_t>(backend, columns * depth);
            auto whole = make_vector<char>(backend, packed_size), windows = make_vector<char>(backend, packed_size);
            fill_random(settings, generator, b);
            fill_canary(whole), fill_canary(windows);
            auto const b_scales = random_scales<scalar_type_>(backend, generator, columns, depth, row_bytes, 0.75f);
            auto const b_operand = b_scales.operand(b.raw_values_data());

            // One pack of every column, one in two column windows, then the shape
            nk_size_t shape_width = 0, shape_depth = 0;
            nk_status_t status = backend.call(pack_fn_, b_operand, columns, depth, row_bytes, whole.raw_values_data(),
                                              0, columns);
            if (status == nk_success_k)
                status = backend.call(pack_fn_, b_operand, columns, depth, row_bytes, windows.raw_values_data(), 0,
                                      columns / 2);
            if (status == nk_success_k)
                status = backend.call(pack_fn_, b_operand, columns, depth, row_bytes, windows.raw_values_data(),
                                      columns / 2, columns);
            if (status == nk_success_k)
                status = backend.call(packed_shape_fn_, whole.raw_values_data(), &shape_width, &shape_depth);
            if (!expect_completed(stats, backend, status)) return stats;

            stats.expect(shape_width == columns && shape_depth == depth, "packed_shape disagrees with the pack");
            stats.expect(std::memcmp(whole.raw_values_data(), windows.raw_values_data(), whole.size_bytes()) == 0,
                         "two column windows pack differently from one pack of every column");
        }
    }
    return stats;
}

/** @c test_dots_pack_layout on a default-constructed @p backend_type_. */
template <typename scalar_type_, typename backend_type_, auto packed_size_fn_, auto packed_shape_fn_, auto pack_fn_>
error_stats_t test_dots_pack_layout(settings_t const &settings) {
    return test_dots_pack_layout<scalar_type_, backend_type_, packed_size_fn_, packed_shape_fn_, pack_fn_>(
        settings, make_backend<backend_type_>(settings));
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
inline std::vector<dots_symmetric_case_t> dots_symmetric_cases(settings_t const &settings) {
    dots_strides_t const tight = dots_strides_t::tight_k, padded = dots_strides_t::padded_k;
    return {
        {settings.matrix_height, settings.matrix_depth, 0, settings.matrix_height, tight},
        {1, 1, 0, 1, tight},
        {17, 65, 0, 17, padded},
        {129, 300, 0, 129, padded},
        {129, 300, 5, 40, padded},
        {300, 1024, 256, 100, padded},
    };
}

/** Symmetric GEMM, A × Aᵀ, over @c dots_symmetric_cases against the serial `nk::` reference, on and
 *  above the diagonal of the computed rows, with everything else in the output left untouched.
 *  External baselines summing less precisely than NumKong pass their own @p term_error_bound. */
template <typename scalar_type_, typename backend_type_ = host_backend_t, typename symmetric_kernel_type_>
error_stats_t test_dots_symmetric(settings_t const &settings, backend_type_ backend,
                                  symmetric_kernel_type_ symmetric_fn, nk_f64_t term_error_bound) {
    constexpr nk_block_scaled_format_t format = nk_block_scaled_format_of_dtype(scalar_type_::dtype());
    using scalar_t = typename nk::type_for<format.element_dtype>::type;
    using result_t = typename scalar_type_::dot_result_t;
    using reference_t = bounded_reference_for<scalar_type_, result_t>;

    error_stats_t stats(term_error_bound);
    std::mt19937 generator(settings.seed.value);
    std::size_t const dimensions_per_value = nk::dimensions_per_value<scalar_t>();
    std::size_t const depth_multiple = std::max<std::size_t>(format.block_size, dimensions_per_value);
    std::vector<dots_symmetric_case_t> const cases = dots_symmetric_cases(settings);

    for (time_point_t const deadline = steady_clock_t::now() + settings.time_limit_per_kernel;
         steady_clock_t::now() < deadline;) {
        for (dots_symmetric_case_t const &test_case : cases) {
            std::size_t const count = test_case.count, row_start = test_case.row_start;
            std::size_t const row_end = std::min(count, row_start + test_case.row_count);
            std::size_t const depth = nk::divide_round_up(test_case.depth, depth_multiple) * depth_multiple;
            std::size_t const row_bytes = depth / dimensions_per_value * sizeof(scalar_t);
            bool const padded = test_case.strides == dots_strides_t::padded_k;
            std::size_t const stride = backend.row_stride(row_bytes) + (padded ? stride_padding_v<scalar_type_> : 0);
            std::size_t const c_stride = (count + (padded ? 3 : 0)) * sizeof(result_t);
            std::size_t const stride_values = stride / sizeof(scalar_t);

            auto a = make_vector<scalar_t>(backend, count * stride_values * dimensions_per_value);
            auto c = make_vector<result_t>(backend, count * c_stride / sizeof(result_t));
            fill_random(settings, generator, a);
            fill_padding_canary(a, count, row_bytes, stride), fill_canary(c);
            auto const scales = random_scales<scalar_type_>(backend, generator, count, depth, stride, 1.5f);

            if (!expect_completed(stats, backend,
                                  backend.call(symmetric_fn, scales.operand(a.raw_values_data()), count, depth, stride,
                                               c.raw_values_data(), c_stride, row_start, test_case.row_count)))
                return stats;

            std::vector<reference_t> c_reference(count * count);
            stats.expect(nk::dots_symmetric<scalar_type_, reference_t>(
                scales.operand(a.values_data()), count, depth, stride, c_reference.data(), count * sizeof(reference_t),
                row_start, row_end - row_start, no_tiers_k, nullptr));

            for (std::size_t row = row_start; row < row_end; row++)
                for (std::size_t column = row; column < count; column++)
                    stats.accumulate(c[row * c_stride / sizeof(result_t) + column], c_reference[row * count + column]);
            expect_symmetric_untouched<result_t>(stats, c, count, c_stride, row_start, row_end);
        }
    }
    return stats;
}

/** @c test_dots_symmetric on @p backend, held to the @c nk_dot_error_bound of @p scalar_type_. */
template <typename scalar_type_, typename backend_type_, typename symmetric_kernel_type_>
error_stats_t test_dots_symmetric(settings_t const &settings, backend_type_ backend,
                                  symmetric_kernel_type_ symmetric_fn) {
    return test_dots_symmetric<scalar_type_, backend_type_>(settings, backend, symmetric_fn,
                                                            nk_dot_error_bound(scalar_type_::dtype()));
}

/** @c test_dots_symmetric on a default-constructed @p backend_type_. */
template <typename scalar_type_, typename backend_type_ = host_backend_t, typename symmetric_kernel_type_>
error_stats_t test_dots_symmetric(settings_t const &settings, symmetric_kernel_type_ symmetric_fn,
                                  nk_f64_t term_error_bound) {
    return test_dots_symmetric<scalar_type_, backend_type_>(settings, make_backend<backend_type_>(settings),
                                                            symmetric_fn, term_error_bound);
}

/** @c test_dots_symmetric held to the @c nk_dot_error_bound of @p scalar_type_. */
template <typename scalar_type_, typename backend_type_ = host_backend_t, typename symmetric_kernel_type_>
error_stats_t test_dots_symmetric(settings_t const &settings, symmetric_kernel_type_ symmetric_fn) {
    return test_dots_symmetric<scalar_type_, backend_type_>(settings, symmetric_fn,
                                                            nk_dot_error_bound(scalar_type_::dtype()));
}

/** The launch contract of a backend refusing misaligned operands, which a CPU backend has no part
 *  in: pack sizes over @c dots_packed_cases hold a 64-byte header, every B row and one norm per
 *  column, and an A or vectors pointer or stride off 16 bytes, or a C stride off the result size,
 *  is refused before launch with its output untouched. A NaN input comes out NaN. */
template <typename scalar_type_, typename backend_type_, auto packed_size_fn_, auto dots_fn_, auto symmetric_fn_>
error_stats_t test_dots_launch_contract(settings_t const &settings, backend_type_ backend) {
    constexpr nk_block_scaled_format_t format = nk_block_scaled_format_of_dtype(scalar_type_::dtype());
    using scalar_t = typename nk::type_for<format.element_dtype>::type;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_type_::dot_result_t;
    using norm_t = std::conditional_t<nk::is_integral_dtype<result_t>(), nk_u32_t, result_t>;

    error_stats_t stats(comparison_family_t::exact_k);
    std::size_t const dimensions_per_value = nk::dimensions_per_value<scalar_t>();
    std::size_t const depth_multiple = std::max<std::size_t>(format.block_size, dimensions_per_value);

    for (dots_packed_case_t const &test_case : dots_packed_cases<scalar_type_>(settings)) {
        std::size_t const columns = test_case.columns;
        std::size_t const depth = nk::divide_round_up(test_case.depth, depth_multiple) * depth_multiple;
        std::size_t const row_bytes = depth / dimensions_per_value * sizeof(scalar_t);
        stats.expect(pack_size_bytes(stats, packed_size_fn_, columns, depth) >=
                         64 + columns * row_bytes + columns * sizeof(norm_t),
                     "pack size below a header, every row, and every norm");
    }

    std::size_t const count = 4,
                      depth = nk::divide_round_up(16 * dimensions_per_value, depth_multiple) * depth_multiple;
    std::size_t const row_bytes = depth / dimensions_per_value * sizeof(scalar_t);
    std::mt19937 generator(settings.seed.value);
    std::size_t const aligned_stride = backend.row_stride(row_bytes), output_stride = count * sizeof(result_t);
    std::size_t const odd_stride = row_bytes % 16 == 15 ? row_bytes + 2 : row_bytes + 1;
    auto const scales = random_scales<scalar_type_>(backend, generator, count, depth, aligned_stride, 1.5f);
    auto rows = make_vector<char>(backend, count * (aligned_stride + odd_stride) + 16);
    auto packed = make_vector<char>(backend, pack_size_bytes(stats, packed_size_fn_, count, depth));
    auto output = make_vector<result_t>(backend, count * 2 * count);
    fill_canary(output);

    auto const *aligned = reinterpret_cast<raw_t const *>(rows.raw_values_data());
    auto const *shifted = reinterpret_cast<raw_t const *>(rows.raw_values_data() + 1);
    auto *cells = output.raw_values_data();
    void const *b_packed = packed.raw_values_data();
    stats.expect(backend.refuses_misaligned(dots_fn_, scales.operand(shifted), b_packed, cells, count, count, depth,
                                            aligned_stride, output_stride),
                 "packed took an A off 16 bytes");
    stats.expect(backend.refuses_misaligned(dots_fn_, scales.operand(aligned), b_packed, cells, count, count, depth,
                                            odd_stride, output_stride),
                 "packed took an A stride off 16 bytes");
    stats.expect(backend.refuses_misaligned(symmetric_fn_, scales.operand(shifted), count, depth, aligned_stride, cells,
                                            output_stride, 0, count),
                 "symmetric took vectors off 16 bytes");
    stats.expect(backend.refuses_misaligned(symmetric_fn_, scales.operand(aligned), count, depth, odd_stride, cells,
                                            output_stride, 0, count),
                 "symmetric took a vectors stride off 16 bytes");
    // Four bytes of padding keep 4-byte results aligned, so only 8-byte results can be caught off their size
    if constexpr (sizeof(result_t) == 8) {
        std::size_t const word_stride = output_stride + 4;
        stats.expect(backend.refuses_misaligned(dots_fn_, scales.operand(aligned), b_packed, cells, count, count, depth,
                                                aligned_stride, word_stride),
                     "packed took a C stride off the result size");
        stats.expect(backend.refuses_misaligned(symmetric_fn_, scales.operand(aligned), count, depth, aligned_stride,
                                                cells, word_stride, 0, count),
                     "symmetric took a result stride off the result size");
    }
    if (!expect_completed(stats, backend, nk_success_k)) return stats;
    stats.expect(overwritten_bytes(output, 0, output.size_bytes()) == 0, "a refused call wrote its output");

    // A NaN input reaches every sum it enters, however the capability widens its codes
    if constexpr (nk::nan_capable_dtype<scalar_t>) {
        reinterpret_cast<scalar_t *>(rows.raw_values_data())[0] = scalar_t::quiet_nan();
        if (!expect_completed(stats, backend,
                              backend.call(symmetric_fn_, scales.operand(aligned), count, depth, aligned_stride, cells,
                                           output_stride, 0, count)))
            return stats;
        stats.expect(std::isnan(static_cast<double>(cells[0])), "a NaN input summed to a finite dot");
    }
    return stats;
}

/** @c test_dots_launch_contract on a default-constructed @p backend_type_. */
template <typename scalar_type_, typename backend_type_, auto packed_size_fn_, auto dots_fn_, auto symmetric_fn_>
error_stats_t test_dots_launch_contract(settings_t const &settings) {
    return test_dots_launch_contract<scalar_type_, backend_type_, packed_size_fn_, dots_fn_, symmetric_fn_>(
        settings, make_backend<backend_type_>(settings));
}

#pragma endregion Dots

#pragma region Set Distances

/** Rows of A, rows of B, and dimensions per row of one set or spatial distance matrix. */
struct matrix_shape_t {
    std::size_t rows, columns, depth;
};

/** The configured shape, then one to three rows and columns over depths off every vector width, and
 *  a depth past one 512-bit tile over odd tile edges. */
inline std::vector<matrix_shape_t> matrix_shapes(settings_t const &settings) {
    return {{settings.matrix_height, settings.matrix_width, settings.matrix_depth},
            {1, 1, 3},
            {2, 3, 17},
            {3, 2, 33},
            {17, 33, 523}};
}

/** Batched Hamming or Jaccard distances, as @p kind_ picks, with a packed B matrix over
 *  @c matrix_shapes, exact against the serial `nk::` reference. Row 0 of A and column 0 of B are
 *  empty for Jaccard, so cell (0, 0) has an empty union. */
template <typename scalar_type_, nk_kernel_kind_t kind_, typename pack_size_kernel_type_, typename pack_kernel_type_,
          typename sets_kernel_type_>
error_stats_t test_sets_packed(settings_t const &settings, pack_size_kernel_type_ packed_size_fn,
                               pack_kernel_type_ pack_fn, sets_kernel_type_ sets_fn) {
    constexpr bool jaccard = kind_ == nk_kernel_jaccard_k;
    using scalar_t = scalar_type_;
    using result_t = std::conditional_t<jaccard, f32_t, u32_t>;

    error_stats_t stats(comparison_family_t::exact_k);
    std::mt19937 generator(settings.seed.value);
    std::size_t const dimensions_per_value = nk::dimensions_per_value<scalar_t>();

    for (time_point_t const deadline = steady_clock_t::now() + settings.time_limit_per_kernel;
         steady_clock_t::now() < deadline;)
        for (matrix_shape_t const &shape : matrix_shapes(settings)) {
            std::size_t const rows = shape.rows, columns = shape.columns;
            std::size_t const depth = nk::divide_round_up(shape.depth, dimensions_per_value) * dimensions_per_value;
            std::size_t const stride = nk::divide_round_up(depth, 8) * sizeof(scalar_t);
            std::size_t const stride_dimensions = stride / sizeof(scalar_t) * dimensions_per_value;
            std::size_t const c_stride = columns * sizeof(result_t);

            auto a = make_vector<scalar_t>(rows * stride_dimensions),
                 b = make_vector<scalar_t>(columns * stride_dimensions);
            auto c = make_vector<result_t>(rows * columns), c_reference = make_vector<result_t>(rows * columns);
            auto b_packed = make_vector<char>(pack_size_bytes(stats, packed_size_fn, columns, depth)),
                 b_packed_reference = make_vector<char>(nk::dots_pack_size<scalar_t>(columns, depth, no_tiers_k).value);
            fill_random(settings, generator, a);
            fill_random(settings, generator, b);
            if constexpr (jaccard)
                std::memset(a.raw_values_data(), 0, stride), std::memset(b.raw_values_data(), 0, stride);

            stats.expect(
                pack_fn(b.raw_values_data(), columns, depth, stride, b_packed.raw_values_data(), 0, columns, nullptr));
            stats.expect(sets_fn(a.raw_values_data(), b_packed.raw_values_data(), c.raw_values_data(), rows, columns,
                                 depth, stride, c_stride, nullptr));

            stats.expect(nk::dots_pack<scalar_t>(b.values_data(), columns, depth, stride,
                                                 b_packed_reference.raw_values_data(), no_tiers_k));
            if constexpr (jaccard)
                stats.expect(nk::jaccards_packed<scalar_t, result_t>(
                    a.values_data(), b_packed_reference.raw_values_data(), c_reference.values_data(), rows, columns,
                    depth, stride, c_stride, no_tiers_k));
            else
                stats.expect(nk::hammings_packed<scalar_t, result_t>(
                    a.values_data(), b_packed_reference.raw_values_data(), c_reference.values_data(), rows, columns,
                    depth, stride, c_stride, no_tiers_k));

            for (std::size_t index = 0; index < rows * columns; index++) stats.accumulate(c[index], c_reference[index]);
        }
    return stats;
}

/** Symmetric Hamming or Jaccard distances, as @p kind_ picks, over @c matrix_shapes, exact against
 *  the serial `nk::` reference over the upper triangle, and untouched below it. Row 0 is empty for
 *  Jaccard, so the first diagonal cell has an empty union. */
template <typename scalar_type_, nk_kernel_kind_t kind_, typename symmetric_kernel_type_>
error_stats_t test_sets_symmetric(settings_t const &settings, symmetric_kernel_type_ symmetric_fn) {
    constexpr bool jaccard = kind_ == nk_kernel_jaccard_k;
    using scalar_t = scalar_type_;
    using result_t = std::conditional_t<jaccard, f32_t, u32_t>;

    error_stats_t stats(comparison_family_t::exact_k);
    std::mt19937 generator(settings.seed.value);
    std::size_t const dimensions_per_value = nk::dimensions_per_value<scalar_t>();

    for (time_point_t const deadline = steady_clock_t::now() + settings.time_limit_per_kernel;
         steady_clock_t::now() < deadline;)
        for (matrix_shape_t const &shape : matrix_shapes(settings)) {
            std::size_t const count = shape.rows;
            std::size_t const depth = nk::divide_round_up(shape.depth, dimensions_per_value) * dimensions_per_value;
            std::size_t const stride = nk::divide_round_up(depth, 8) * sizeof(scalar_t);
            std::size_t const stride_dimensions = stride / sizeof(scalar_t) * dimensions_per_value;
            std::size_t const c_stride = count * sizeof(result_t);

            auto a = make_vector<scalar_t>(count * stride_dimensions);
            auto c = make_vector<result_t>(count * count), c_reference = make_vector<result_t>(count * count);
            fill_random(settings, generator, a);
            if constexpr (jaccard) std::memset(a.raw_values_data(), 0, stride);
            fill_canary(c);

            stats.expect(symmetric_fn(a.raw_values_data(), count, depth, stride, c.raw_values_data(), c_stride, 0,
                                      count, nullptr));
            if constexpr (jaccard)
                stats.expect(nk::jaccards_symmetric<scalar_t, result_t>(
                    a.values_data(), count, depth, stride, c_reference.values_data(), c_stride, 0, count, no_tiers_k));
            else
                stats.expect(nk::hammings_symmetric<scalar_t, result_t>(
                    a.values_data(), count, depth, stride, c_reference.values_data(), c_stride, 0, count, no_tiers_k));

            for (std::size_t row = 0; row < count; row++)
                for (std::size_t column = row; column < count; column++)
                    stats.accumulate(c[row * count + column], c_reference[row * count + column]);
            expect_symmetric_untouched<result_t>(stats, c, count, c_stride, 0, count);
        }
    return stats;
}

/** @c test_sets_packed over Hamming distances. */
template <typename scalar_type_, typename... kernels_types_>
error_stats_t test_hammings_packed(settings_t const &settings, kernels_types_... kernels) {
    return test_sets_packed<scalar_type_, nk_kernel_hamming_k>(settings, kernels...);
}

/** @c test_sets_symmetric over Hamming distances. */
template <typename scalar_type_, typename... kernels_types_>
error_stats_t test_hammings_symmetric(settings_t const &settings, kernels_types_... kernels) {
    return test_sets_symmetric<scalar_type_, nk_kernel_hamming_k>(settings, kernels...);
}

/** @c test_sets_packed over Jaccard distances. */
template <typename scalar_type_, typename... kernels_types_>
error_stats_t test_jaccards_packed(settings_t const &settings, kernels_types_... kernels) {
    return test_sets_packed<scalar_type_, nk_kernel_jaccard_k>(settings, kernels...);
}

/** @c test_sets_symmetric over Jaccard distances. */
template <typename scalar_type_, typename... kernels_types_>
error_stats_t test_jaccards_symmetric(settings_t const &settings, kernels_types_... kernels) {
    return test_sets_symmetric<scalar_type_, nk_kernel_jaccard_k>(settings, kernels...);
}

#pragma endregion Set Distances

#pragma region Spatial Distances

/** Batched angular or euclidean distances, as @p kind_ picks, with B packed in two column windows.
 *  Row 0 of A is zero for euclidean, so row 0 of C reads every packed norm back as √‖b‖². */
template <typename scalar_type_, typename backend_type_, nk_kernel_kind_t kind_, typename pack_size_kernel_type_,
          typename pack_kernel_type_, typename spatials_kernel_type_>
error_stats_t test_spatials_packed(settings_t const &settings, pack_size_kernel_type_ packed_size_fn,
                                   pack_kernel_type_ pack_fn, spatials_kernel_type_ spatials_fn) {
    constexpr bool angular = kind_ == nk_kernel_angular_k;
    constexpr nk_block_scaled_format_t format = nk_block_scaled_format_of_dtype(scalar_type_::dtype());
    using scalar_t = typename nk::type_for<format.element_dtype>::type;
    using result_t =
        std::conditional_t<angular, typename scalar_type_::angular_result_t, typename scalar_type_::euclidean_result_t>;
    using reference_t = reference_for<scalar_type_>;

    backend_type_ backend = make_backend<backend_type_>(settings);
    error_stats_t stats(angular ? nk_angular_error_bound(scalar_type_::dtype())
                                : nk_euclidean_error_bound(scalar_type_::dtype()));
    std::mt19937 generator(settings.seed.value);
    std::size_t const dimensions_per_value = nk::dimensions_per_value<scalar_t>();
    std::size_t const depth_multiple = std::max<std::size_t>(format.block_size, dimensions_per_value);

    for (time_point_t const deadline = steady_clock_t::now() + settings.time_limit_per_kernel;
         steady_clock_t::now() < deadline;)
        for (matrix_shape_t const &shape : matrix_shapes(settings)) {
            std::size_t const rows = shape.rows, columns = shape.columns;
            std::size_t const depth = nk::divide_round_up(shape.depth, depth_multiple) * depth_multiple;
            std::size_t const stride = backend.row_stride(nk::divide_round_up(depth, dimensions_per_value) *
                                                          sizeof(scalar_t));
            std::size_t const stride_values = stride / sizeof(scalar_t);

            auto a = make_vector<scalar_t>(backend, rows * stride_values * dimensions_per_value),
                 b = make_vector<scalar_t>(backend, columns * stride_values * dimensions_per_value);
            auto c = make_vector<result_t>(backend, rows * columns);
            auto b_packed = make_vector<char>(backend, pack_size_bytes(stats, packed_size_fn, columns, depth));
            fill_random(settings, generator, a);
            fill_random(settings, generator, b);
            if constexpr (!angular) std::memset(a.raw_values_data(), 0, stride);
            auto const a_scales = random_scales<scalar_type_>(backend, generator, rows, depth, stride, 1.5f),
                       b_scales = random_scales<scalar_type_>(backend, generator, columns, depth, stride, 0.75f);

            // The norms of the second window's columns come from a pack not starting at zero
            nk_status_t status = backend.call(pack_fn, b_scales.operand(b.raw_values_data()), columns, depth, stride,
                                              b_packed.raw_values_data(), 0, columns / 2);
            if (status == nk_success_k)
                status = backend.call(pack_fn, b_scales.operand(b.raw_values_data()), columns, depth, stride,
                                      b_packed.raw_values_data(), columns / 2, columns);
            if (status == nk_success_k)
                status = backend.call(spatials_fn, a_scales.operand(a.raw_values_data()), b_packed.raw_values_data(),
                                      c.raw_values_data(), rows, columns, depth, stride, columns * sizeof(result_t));
            if (!expect_completed(stats, backend, status)) return stats;

            auto b_packed_reference = make_vector<char>(
                nk::dots_pack_size<scalar_type_>(columns, depth, no_tiers_k).value);
            auto dots_reference = make_vector<reference_t>(rows * columns);
            auto a_squared_norms = make_vector<reference_t>(rows), b_squared_norms = make_vector<reference_t>(columns);
            stats.expect(nk::dots_pack<scalar_type_>(b_scales.operand(b.values_data()), columns, depth, stride,
                                                     b_packed_reference.raw_values_data(), no_tiers_k, nullptr));
            stats.expect(nk::dots_packed<scalar_type_, reference_t>(
                a_scales.operand(a.values_data()), b_packed_reference.raw_values_data(), dots_reference.values_data(),
                rows, columns, depth, stride, columns * sizeof(reference_t), no_tiers_k, nullptr));
            for (std::size_t row = 0; row < rows; row++)
                stats.expect(nk::dots_symmetric<scalar_type_, reference_t>(
                    a_scales.operand(a.values_data() + row * stride_values, row), 1, depth, stride,
                    a_squared_norms.values_data() + row, sizeof(reference_t), 0, 1, no_tiers_k, nullptr));
            for (std::size_t column = 0; column < columns; column++)
                stats.expect(nk::dots_symmetric<scalar_type_, reference_t>(
                    b_scales.operand(b.values_data() + column * stride_values, column), 1, depth, stride,
                    b_squared_norms.values_data() + column, sizeof(reference_t), 0, 1, no_tiers_k, nullptr));

            for (std::size_t row = 0; row < rows; row++)
                for (std::size_t column = 0; column < columns; column++) {
                    reference_t const first = a_squared_norms[row], second = b_squared_norms[column];
                    accumulate_spatial<kind_>(
                        stats, c[row * columns + column],
                        spatial_distance<kind_>(dots_reference[row * columns + column], first, second),
                        reference_t(first + second), depth);
                }
        }
    return stats;
}

/** Symmetric angular or euclidean distances, as @p kind_ picks, over the upper triangle, zeros on
 *  the diagonal, and untouched below it. */
template <typename scalar_type_, typename backend_type_, nk_kernel_kind_t kind_, typename symmetric_kernel_type_>
error_stats_t test_spatials_symmetric(settings_t const &settings, symmetric_kernel_type_ symmetric_fn) {
    constexpr bool angular = kind_ == nk_kernel_angular_k;
    constexpr nk_block_scaled_format_t format = nk_block_scaled_format_of_dtype(scalar_type_::dtype());
    using scalar_t = typename nk::type_for<format.element_dtype>::type;
    using result_t =
        std::conditional_t<angular, typename scalar_type_::angular_result_t, typename scalar_type_::euclidean_result_t>;
    using reference_t = reference_for<scalar_type_>;

    backend_type_ backend = make_backend<backend_type_>(settings);
    error_stats_t stats(angular ? nk_angular_error_bound(scalar_type_::dtype())
                                : nk_euclidean_error_bound(scalar_type_::dtype()));
    std::mt19937 generator(settings.seed.value);
    std::size_t const dimensions_per_value = nk::dimensions_per_value<scalar_t>();
    std::size_t const depth_multiple = std::max<std::size_t>(format.block_size, dimensions_per_value);

    for (time_point_t const deadline = steady_clock_t::now() + settings.time_limit_per_kernel;
         steady_clock_t::now() < deadline;)
        for (matrix_shape_t const &shape : matrix_shapes(settings)) {
            std::size_t const count = shape.rows, c_stride = count * sizeof(result_t);
            std::size_t const depth = nk::divide_round_up(shape.depth, depth_multiple) * depth_multiple;
            std::size_t const stride = backend.row_stride(nk::divide_round_up(depth, dimensions_per_value) *
                                                          sizeof(scalar_t));
            std::size_t const stride_values = stride / sizeof(scalar_t);

            auto a = make_vector<scalar_t>(backend, count * stride_values * dimensions_per_value);
            auto c = make_vector<result_t>(backend, count * count);
            fill_random(settings, generator, a);
            fill_canary(c);
            auto const scales = random_scales<scalar_type_>(backend, generator, count, depth, stride, 1.5f);

            if (!expect_completed(stats, backend,
                                  backend.call(symmetric_fn, scales.operand(a.raw_values_data()), count, depth, stride,
                                               c.raw_values_data(), c_stride, 0, count)))
                return stats;

            auto dots_reference = make_vector<reference_t>(count * count);
            stats.expect(nk::dots_symmetric<scalar_type_, reference_t>(
                scales.operand(a.values_data()), count, depth, stride, dots_reference.values_data(),
                count * sizeof(reference_t), 0, count, no_tiers_k, nullptr));
            for (std::size_t row = 0; row < count; row++)
                for (std::size_t column = row; column < count; column++) {
                    reference_t const first = dots_reference[row * count + row],
                                      second = dots_reference[column * count + column];
                    reference_t const distance = row == column
                                                     ? reference_t(0)
                                                     : spatial_distance<kind_>(dots_reference[row * count + column],
                                                                               first, second);
                    accumulate_spatial<kind_>(stats, c[row * count + column], distance, reference_t(first + second),
                                              depth);
                }
            expect_symmetric_untouched<result_t>(stats, c, count, c_stride, 0, count);
        }
    return stats;
}

/** @c test_spatials_packed over angular distances. */
template <typename scalar_type_, typename backend_type_ = host_backend_t, typename... kernels_types_>
error_stats_t test_angulars_packed(settings_t const &settings, kernels_types_... kernels) {
    return test_spatials_packed<scalar_type_, backend_type_, nk_kernel_angular_k>(settings, kernels...);
}

/** @c test_spatials_symmetric over angular distances. */
template <typename scalar_type_, typename backend_type_ = host_backend_t, typename... kernels_types_>
error_stats_t test_angulars_symmetric(settings_t const &settings, kernels_types_... kernels) {
    return test_spatials_symmetric<scalar_type_, backend_type_, nk_kernel_angular_k>(settings, kernels...);
}

/** @c test_spatials_packed over euclidean distances. */
template <typename scalar_type_, typename backend_type_ = host_backend_t, typename... kernels_types_>
error_stats_t test_euclideans_packed(settings_t const &settings, kernels_types_... kernels) {
    return test_spatials_packed<scalar_type_, backend_type_, nk_kernel_euclidean_k>(settings, kernels...);
}

/** @c test_spatials_symmetric over euclidean distances. */
template <typename scalar_type_, typename backend_type_ = host_backend_t, typename... kernels_types_>
error_stats_t test_euclideans_symmetric(settings_t const &settings, kernels_types_... kernels) {
    return test_spatials_symmetric<scalar_type_, backend_type_, nk_kernel_euclidean_k>(settings, kernels...);
}

#pragma endregion Spatial Distances

#pragma region Attention

/** Expects every log-sum-exp within the scale threshold of its reference, relative to its magnitude
 *  past 1, or within what @p weights_ rounding moves a sum of weights by, and exactly the
 *  reference's −∞ on rows that see no keys. */
template <attention_weights_t weights_ = attention_weights_t::unquantized_k, typename actual_vector_type_,
          typename expected_vector_type_>
void expect_log_sum_exp(settings_t const &settings, error_stats_t &stats, actual_vector_type_ const &actual,
                        expected_vector_type_ const &expected) {
    double const rounding = weights_ == attention_weights_t::unquantized_k
                                ? 0
                                : std::log1p(std::ldexp(1.0, -static_cast<int>(weights_)));
    bool held = true;
    for (std::size_t index = 0; index < actual.size_values(); index++) {
        double const result = static_cast<double>(actual[index]), reference = static_cast<double>(expected[index]);
        double const bound = std::max(settings.scale_threshold * std::max(1.0, std::fabs(reference)), rounding);
        held = held && (std::isinf(reference) ? result == reference : std::fabs(result - reference) <= bound);
    }
    stats.expect(held, "log-sum-exp");
}

/** Ragged attention over @c attention_cases against @c reference_attention_unmasked, or against
 *  @c reference_attention_masked under a narrower band. Every case's batch holds the long segment
 *  with about half as many queries, a pad of two queries without keys, a one-query decode, five
 *  keys under nine queries, and a 60-token prefill, so each segment offsets its queries
 *  differently. The pack runs in two task windows, and random windows of the query tokens × heads
 *  grid, run in shuffled order and clipped to it, must match one call over the grid bit for bit. */
template <typename scalar_type_, typename backend_type_ = host_backend_t,
          attention_weights_t weights_ = attention_weights_t::unquantized_k, typename pack_size_kernel_type_,
          typename pack_kernel_type_, typename attention_kernel_type_>
error_stats_t test_attention_packed(settings_t const &settings, pack_size_kernel_type_ packed_size_fn,
                                    pack_kernel_type_ pack_fn, attention_kernel_type_ attention_fn) {
    using scalar_t = scalar_type_;
    using result_t = typename scalar_t::attention_result_t;

    backend_type_ backend = make_backend<backend_type_>(settings);
    error_stats_t stats(attention_family(weights_));
    std::mt19937 generator(settings.seed.value);

    std::vector<attention_case_t> const cases = attention_cases();

    for (time_point_t const deadline = steady_clock_t::now() + settings.time_limit_per_kernel;
         steady_clock_t::now() < deadline;) {
        for (attention_case_t const &test_case : cases) {
            auto const segments = make_attention_segments(backend, {test_case.main_length, 0, 33, 5, 60},
                                                          {attention_main_queries(test_case.main_length), 2, 1, 9, 60});
            attention_layout_t const layout {attention_key_value_heads_k * test_case.group, attention_key_value_heads_k,
                                             test_case.depth, 0.05f};
            std::size_t const query_tokens = segments.query_tokens(), key_tokens = segments.key_tokens();
            std::size_t const query_stride = layout.query_width() * sizeof(scalar_t),
                              output_stride = layout.query_width() * sizeof(result_t);
            std::vector<std::size_t> bounds {0, NUMKONG_SIZE_MAX}, windows {0, 1, 2, 3, 4, 5};
            std::uniform_int_distribution<std::size_t> bound_distribution(0, query_tokens * layout.head_count);
            while (bounds.size() < windows.size() + 1) bounds.push_back(bound_distribution(generator));
            std::sort(bounds.begin(), bounds.end());
            std::shuffle(windows.begin(), windows.end(), generator);

            auto queries = make_vector<scalar_t>(backend, query_tokens * layout.query_width());
            auto keys = make_vector<scalar_t>(backend, key_tokens * layout.key_value_width()),
                 values = make_vector<scalar_t>(backend, key_tokens * layout.key_value_width());
            fill_random(settings, generator, queries), fill_random(settings, generator, keys),
                fill_random(settings, generator, values);
            auto key_value_packed = make_vector<char>(
                backend, pack_size_bytes(stats, packed_size_fn, layout.key_value_head_count, layout.depth, key_tokens,
                                         segments.count()));
            // Allocated before any launch: Windows faults on host writes to managed memory then
            auto output = make_vector<result_t>(backend, query_tokens * layout.query_width()),
                 windowed_output = make_vector<result_t>(backend, query_tokens * layout.query_width());
            auto log_sum_exp = make_vector<result_t>(backend, query_tokens * layout.head_count),
                 windowed_log_sum_exp = make_vector<result_t>(backend, query_tokens * layout.head_count);

            nk_status_t status = pack_attention_in_two_windows(backend, pack_fn, keys, values, segments, layout,
                                                               key_value_packed);
            auto const attend = [&](auto &into, auto &into_log_sum_exp, std::size_t task_begin, std::size_t task_end) {
                return backend.call(attention_fn, queries.raw_values_data(), key_value_packed.raw_values_data(),
                                    into.raw_values_data(), into_log_sum_exp.raw_values_data(), layout.head_count,
                                    layout.key_value_head_count, layout.depth, segments.query_offsets.values_data(),
                                    query_stride, output_stride, layout.scale, test_case.keys_before,
                                    test_case.keys_after, task_begin, task_end);
            };
            if (status == nk_success_k) status = attend(output, log_sum_exp, 0, NUMKONG_SIZE_MAX);
            for (std::size_t window : windows)
                if (status == nk_success_k)
                    status = attend(windowed_output, windowed_log_sum_exp, bounds[window], bounds[window + 1]);
            if (!expect_completed(stats, backend, status)) return stats;
            void const *outputs[2] = {output.raw_values_data(), windowed_output.raw_values_data()};
            void const *log_sum_exps[2] = {log_sum_exp.raw_values_data(), windowed_log_sum_exp.raw_values_data()};
            stats.expect(std::memcmp(outputs[0], outputs[1], output.size_bytes()) == 0,
                         "random task windows output differently from one call over the grid");
            stats.expect(std::memcmp(log_sum_exps[0], log_sum_exps[1], log_sum_exp.size_bytes()) == 0,
                         "random task windows log-sum-exp differently from one call over the grid");

            auto const reference = test_case.unmasked()
                                       ? reference_attention_unmasked(stats, queries, keys, values, segments, layout)
                                       : reference_attention_masked(stats, queries, keys, values, segments, layout,
                                                                    test_case.keys_before, test_case.keys_after);
            accumulate_attention<weights_>(settings, stats, output, reference.output, values);
            expect_log_sum_exp<weights_>(settings, stats, log_sum_exp, reference.log_sum_exp);
        }
    }
    return stats;
}

/** Attention gradients against @c reference_attention_gradients, fed that reference's own output
 *  and log-sum-exp, over ragged segments that offset their queries differently, one without keys.
 *  Unmasked cases cross panel-edge depths and GQA groups; masked ones cross causal, sliding-window,
 *  diagonal and two-sided bands. Gradients start as canaries, the query ones in rows wider than the
 *  output's, and two task windows cover the grid, the second relying on @c task_end clipping. */
template <typename scalar_type_, typename backend_type_ = host_backend_t, typename pack_size_kernel_type_,
          typename pack_kernel_type_, typename gradients_kernel_type_>
error_stats_t test_attention_packed_gradients(settings_t const &settings, pack_size_kernel_type_ packed_size_fn,
                                              pack_kernel_type_ pack_fn, gradients_kernel_type_ gradients_fn) {
    using scalar_t = scalar_type_;

    backend_type_ backend = make_backend<backend_type_>(settings);
    error_stats_t stats(comparison_family_t::bounded_k);
    std::mt19937 generator(settings.seed.value);

    std::size_t const all = attention_all_keys_k;
    std::vector<attention_case_t> cases;
    for (std::size_t depth : {1ul, 64ul, 65ul, 128ul, 129ul, 256ul, 257ul}) cases.push_back({130, 2, depth, all, all});
    for (std::size_t group : {1ul, 4ul}) cases.push_back({130, group, 64, all, all});
    for (std::size_t keys_before : {all, std::size_t(0), std::size_t(32)})
        cases.push_back({130, 2, 65, keys_before, 0});
    cases.push_back({130, 2, 65, 3, 5});
    cases.push_back({130, 2, 192, 32, 0});
    auto const segments = make_attention_segments(backend, {60, 130, 0, 33, 5}, {60, 24, 2, 1, 9});
    std::size_t const query_tokens = segments.query_tokens(), key_tokens = segments.key_tokens();

    for (time_point_t const deadline = steady_clock_t::now() + settings.time_limit_per_kernel;
         steady_clock_t::now() < deadline;) {
        for (attention_case_t const &test_case : cases) {
            attention_layout_t const layout {attention_key_value_heads_k * test_case.group, attention_key_value_heads_k,
                                             test_case.depth, 0.05f};
            // Query gradient rows sit in a wider buffer, so their stride differs from the output's
            std::size_t const query_gradient_width = layout.query_width() + 4;
            std::size_t const query_stride = layout.query_width() * sizeof(scalar_t),
                              output_stride = layout.query_width() * sizeof(f32_t),
                              query_gradient_stride = query_gradient_width * sizeof(f32_t),
                              gradient_stride = layout.key_value_width() * sizeof(f32_t);

            auto queries = make_vector<scalar_t>(backend, query_tokens * layout.query_width());
            auto keys = make_vector<scalar_t>(backend, key_tokens * layout.key_value_width()),
                 values = make_vector<scalar_t>(backend, key_tokens * layout.key_value_width());
            auto output_gradient = make_vector<f32_t>(backend, query_tokens * layout.query_width());
            fill_random(settings, generator, queries), fill_random(settings, generator, keys),
                fill_random(settings, generator, values), fill_random(settings, generator, output_gradient);
            attention_gradients_t const reference = reference_attention_gradients(
                decode_rows(queries, query_tokens, layout.query_width(), layout.query_width()),
                decode_rows(keys, key_tokens, layout.key_value_width(), layout.key_value_width()),
                decode_rows(values, key_tokens, layout.key_value_width(), layout.key_value_width()),
                decode_rows(output_gradient, query_tokens, layout.query_width(), layout.query_width()), segments,
                layout, test_case.keys_before, test_case.keys_after);

            // The serial forward's log-sum-exps, the bar for every other tier, against the F64 ones
            attention_reference_t const serial =
                test_case.unmasked() ? reference_attention_unmasked(stats, queries, keys, values, segments, layout)
                                     : reference_attention_masked(stats, queries, keys, values, segments, layout,
                                                                  test_case.keys_before, test_case.keys_after);
            expect_log_sum_exp(settings, stats, serial.log_sum_exp, reference.log_sum_exp);

            auto output = make_vector<f32_t>(backend, query_tokens * layout.query_width());
            auto log_sum_exp = make_vector<f32_t>(backend, query_tokens * layout.head_count);
            for (std::size_t index = 0; index < reference.output.size(); index++)
                output[index] = static_cast<float>(reference.output[index]);
            for (std::size_t index = 0; index < reference.log_sum_exp.size(); index++)
                log_sum_exp[index] = static_cast<float>(reference.log_sum_exp[index]);
            auto query_gradient = make_vector<f32_t>(backend, query_tokens * query_gradient_width);
            auto key_gradient = make_vector<f32_t>(backend, key_tokens * layout.key_value_width()),
                 value_gradient = make_vector<f32_t>(backend, key_tokens * layout.key_value_width());
            fill_canary(query_gradient), fill_canary(key_gradient), fill_canary(value_gradient);
            auto key_value_packed = make_vector<char>(
                backend, pack_size_bytes(stats, packed_size_fn, layout.key_value_head_count, layout.depth, key_tokens,
                                         segments.count()));

            nk_status_t status = pack_attention_in_two_windows(backend, pack_fn, keys, values, segments, layout,
                                                               key_value_packed);
            std::size_t const windows[2][2] = {{0, 1}, {1, NUMKONG_SIZE_MAX}};
            for (auto const &window : windows)
                if (status == nk_success_k)
                    status = backend.call(
                        gradients_fn, queries.raw_values_data(), key_value_packed.raw_values_data(),
                        output.raw_values_data(), output_gradient.raw_values_data(), log_sum_exp.raw_values_data(),
                        query_gradient.raw_values_data(), key_gradient.raw_values_data(),
                        value_gradient.raw_values_data(), layout.head_count, layout.key_value_head_count, layout.depth,
                        segments.query_offsets.values_data(), segments.key_offsets.values_data(), query_stride,
                        output_stride, query_gradient_stride, gradient_stride, layout.scale, test_case.keys_before,
                        test_case.keys_after, window[0], window[1]);
            if (!expect_completed(stats, backend, status)) return stats;

            auto const accumulate_gradient = [&](backend_vector<f32_t, backend_type_> const &actual,
                                                 std::size_t actual_width, std::vector<double> const &expected,
                                                 std::vector<double> const &magnitudes, std::size_t width) {
                double largest = 0;
                for (double magnitude : magnitudes) largest = std::max(largest, magnitude);
                for (std::size_t index = 0; index < expected.size(); index++)
                    stats.accumulate_bounded(actual[index / width * actual_width + index % width], expected[index],
                                             settings.scale_threshold * largest);
            };
            accumulate_gradient(query_gradient, query_gradient_width, reference.query_gradient,
                                reference.query_magnitude, layout.query_width());
            accumulate_gradient(key_gradient, layout.key_value_width(), reference.key_gradient, reference.key_magnitude,
                                layout.key_value_width());
            accumulate_gradient(value_gradient, layout.key_value_width(), reference.value_gradient,
                                reference.value_magnitude, layout.key_value_width());
            bool padding_kept = true;
            for (std::size_t token = 0; token < query_tokens; token++)
                for (std::size_t column = layout.query_width(); column < query_gradient_width; column++)
                    padding_kept &= std::isnan(
                        static_cast<double>(query_gradient[token * query_gradient_width + column]));
            stats.expect(padding_kept, "query gradient written past its rows");
        }
    }
    return stats;
}

/** NeoX split-half RoPE of two heads over padded rows against an F64 reference: into a separate
 *  output, then in place. */
template <typename scalar_type_, typename backend_type_ = host_backend_t, typename rope_kernel_type_>
error_stats_t test_attention_rope(settings_t const &settings, rope_kernel_type_ rope_fn) {
    using scalar_t = scalar_type_;

    backend_type_ backend = make_backend<backend_type_>(settings);
    error_stats_t stats(nk_attention_rope_error_bound(scalar_t::dtype()));
    std::mt19937 generator(settings.seed.value);
    std::uniform_real_distribution<float> angle_distribution(-3.0f, 3.0f);
    std::size_t const rows = 33, head_count = 2, depth = 74, half_depth = depth / 2;
    std::size_t const row_values = head_count * depth + 8, row_bytes = row_values * sizeof(scalar_t);

    for (time_point_t const deadline = steady_clock_t::now() + settings.time_limit_per_kernel;
         steady_clock_t::now() < deadline;)
        for (bool const in_place : {false, true}) {
            auto x = make_vector<scalar_t>(backend, rows * row_values),
                 y = make_vector<scalar_t>(backend, rows * row_values);
            auto cosines = make_vector<f32_t>(backend, rows * half_depth),
                 sines = make_vector<f32_t>(backend, rows * half_depth);
            fill_random(settings, generator, x);
            for (std::size_t index = 0; index < rows * half_depth; index++) {
                float const angle = angle_distribution(generator);
                cosines[index] = std::cos(angle), sines[index] = std::sin(angle);
            }

            nk_status_t status = in_place ? backend.copy(y.raw_values_data(), x.raw_values_data(), rows * row_bytes)
                                          : nk_success_k;
            if (status == nk_success_k)
                status = backend.call(rope_fn, in_place ? y.raw_values_data() : x.raw_values_data(),
                                      cosines.raw_values_data(), sines.raw_values_data(), y.raw_values_data(), rows,
                                      head_count, depth, row_bytes, row_bytes);
            if (!expect_completed(stats, backend, status)) return stats;

            for (std::size_t row = 0; row < rows; row++)
                for (std::size_t head = 0; head < head_count; head++)
                    for (std::size_t pair = 0; pair < half_depth; pair++) {
                        std::size_t const first = row * row_values + head * depth + pair;
                        double const low = static_cast<double>(x[first]);
                        double const high = static_cast<double>(x[first + half_depth]);
                        double const cosine = static_cast<double>(cosines[row * half_depth + pair]);
                        double const sine = static_cast<double>(sines[row * half_depth + pair]);
                        stats.accumulate_bounded(
                            y[first], low * cosine - high * sine,
                            stats.term_error_bound * (std::fabs(low * cosine) + std::fabs(high * sine)));
                        stats.accumulate_bounded(
                            y[first + half_depth], low * sine + high * cosine,
                            stats.term_error_bound * (std::fabs(low * sine) + std::fabs(high * cosine)));
                    }
        }
    return stats;
}

#pragma endregion Attention

} // namespace ashvardanian::numkong::test

#endif // NUMKONG_TEST_CROSS_HPP
