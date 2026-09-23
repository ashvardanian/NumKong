/**
 *  @brief Template test functions for batch operations (dots, hammings).
 *  @file test/test_cross.hpp
 *  @author Ash Vardanian
 *  @date January 14, 2025
 *
 *  This header contains the template test implementations that are shared
 *  across all ISA-specific test files.
 */
#pragma once
#ifndef NK_TEST_CROSS_HPP
#define NK_TEST_CROSS_HPP

#include "numkong/attention.hpp" // `nk::attention_bidirectional_packed`, `nk::attention_pack`
#include "numkong/spatials.h"    // `nk_angulars_packed_*`, `nk_euclideans_packed_*`

#include "numkong/dots.hpp"   // `nk::dots_packed`, `nk::dots_symmetric`
#include "numkong/matrix.hpp" // `nk::dots_pack_size`, `nk::dots_pack`
#include "numkong/reduce.hpp" // `nk::reduce_moments`

#include "test.hpp"

/**
 *  @brief Generic GEMM test against f118_t reference.
 *  Works for all types: f32, f64, f16, bf16, i8.
 */
template <typename scalar_type_>
error_stats_t test_dots_packed(typename scalar_type_::dots_pack_size_kernel_t packed_size_fn,
                               typename scalar_type_::dots_pack_kernel_t pack_fn,
                               typename scalar_type_::dots_packed_kernel_t dots_fn) {
    using scalar_t = scalar_type_;
    using result_t = typename scalar_t::dot_result_t;
    using reference_t = reference_for<scalar_t, result_t>;

    error_stats_t stats(comparison_family_t::approximate_k);
    std::mt19937 generator(global_config.seed);

    std::size_t m = global_config.matrix_height, n = global_config.matrix_width;
    std::size_t const dims_per_value = nk::dimensions_per_value<scalar_t>();
    std::size_t k_values = nk::divide_round_up(global_config.matrix_depth, dims_per_value);
    std::size_t k = k_values * dims_per_value;
    std::size_t a_stride = k_values * sizeof(scalar_t);
    std::size_t b_stride = k_values * sizeof(scalar_t);
    std::size_t c_stride = n * sizeof(result_t);

    auto a = make_vector<scalar_t>(m * k), b = make_vector<scalar_t>(n * k);
    auto c = make_vector<result_t>(m * n);
    auto c_ref = make_vector<reference_t>(m * n);

    nk_size_t packed_size = packed_size_fn(n, k);
    auto b_packed = make_vector<char>(packed_size);
    nk_size_t ref_pack_size = nk::dots_pack_size<scalar_t, nk::no_simd_k>(n, k);
    auto b_packed_ref = make_vector<char>(ref_pack_size);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        // Run kernel being tested
        pack_fn(b.raw_values_data(), n, k, b_stride, b_packed.raw_values_data(), 0, n);
        dots_fn(a.raw_values_data(), b_packed.raw_values_data(), c.raw_values_data(), m, n, k, a_stride, c_stride);

        // Compute reference using nk:: template
        nk::dots_pack<scalar_t, nk::no_simd_k>(b.values_data(), n, k, b_stride, b_packed_ref.raw_values_data());
        nk::dots_packed<scalar_t, reference_t, nk::no_simd_k>(a.values_data(), b_packed_ref.raw_values_data(),
                                                              c_ref.values_data(), m, n, k, a_stride,
                                                              n * sizeof(reference_t));

        for (std::size_t i = 0; i < m * n; i++) stats.accumulate(c[i], c_ref[i]);
    }
    return stats;
}

/**
 *  @brief Generic symmetric GEMM (A x A^T) test against f118_t reference.
 *  Works for all types: f32, f64, f16, bf16, i8, u8, i4, u4, e4m3, e5m2.
 */
template <typename scalar_type_>
error_stats_t test_dots_symmetric(typename scalar_type_::dots_symmetric_kernel_t kernel_fn) {
    using scalar_t = scalar_type_;
    using result_t = typename scalar_t::dot_result_t;
    using reference_t = reference_for<scalar_t, result_t>;

    error_stats_t stats(comparison_family_t::approximate_k);
    std::mt19937 generator(global_config.seed);

    std::size_t n = global_config.matrix_height;
    std::size_t const dims_per_value = nk::dimensions_per_value<scalar_t>();
    std::size_t k_values = nk::divide_round_up(global_config.matrix_depth, dims_per_value);
    std::size_t k = k_values * dims_per_value;
    std::size_t a_stride = k_values * sizeof(scalar_t);
    std::size_t c_stride = n * sizeof(result_t);

    auto a = make_vector<scalar_t>(n * k);
    auto c = make_vector<result_t>(n * n);
    auto c_ref = make_vector<reference_t>(n * n);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);

        // Run kernel being tested
        kernel_fn(a.raw_values_data(), n, k, a_stride, c.raw_values_data(), c_stride, 0, n);

        // Compute reference using nk:: template
        nk::dots_symmetric<scalar_t, reference_t, nk::no_simd_k>(a.values_data(), n, k, a_stride, c_ref.values_data(),
                                                                 n * sizeof(reference_t));

        // Only check upper triangle and diagonal
        for (std::size_t i = 0; i < n; i++)
            for (std::size_t j = i; j < n; j++) stats.accumulate(c[i * n + j], c_ref[i * n + j]);
    }
    return stats;
}

/**
 *  @brief Ragged bidirectional attention test against the serial backend as reference: fixed segment
 *         mix with a zero-length PAD segment, GQA 2:1, executed over the whole task grid.
 */
template <typename scalar_type_>
error_stats_t test_attention_bidirectional_packed(
    typename scalar_type_::attention_pack_size_kernel_t packed_size_fn,
    typename scalar_type_::attention_pack_kernel_t pack_fn,
    typename scalar_type_::attention_bidirectional_packed_kernel_t attention_fn) {
    using scalar_t = scalar_type_;
    using result_t = typename scalar_t::attention_result_t;

    error_stats_t stats(comparison_family_t::normalized_reduction_k);
    std::mt19937 generator(global_config.seed);

    std::vector<nk_u32_t> const lengths = {60, 130, 0, 33};
    std::vector<nk_u32_t> offsets = {0};
    for (auto length : lengths) offsets.push_back(offsets.back() + length);
    std::size_t const tokens = offsets.back(), head_count = 4, key_value_head_count = 2;
    std::size_t const total_tasks = lengths.size() * head_count;
    nk_f32_t const scale = 0.05f;

    for (auto start = test_start_time(); within_time_budget(start);) {
        for (std::size_t depth : {1ul, 64ul, 65ul, 127ul, 128ul, 129ul, 255ul, 257ul}) {
            std::size_t const queries_row_width = head_count * depth,
                              key_value_row_width = key_value_head_count * depth;
            std::size_t const query_stride_bytes = queries_row_width * sizeof(scalar_t),
                              key_value_stride_bytes = key_value_row_width * sizeof(scalar_t);
            auto queries = make_vector<scalar_t>(tokens * queries_row_width);
            auto keys = make_vector<scalar_t>(tokens * key_value_row_width),
                 values = make_vector<scalar_t>(tokens * key_value_row_width);
            fill_random(generator, queries), fill_random(generator, keys), fill_random(generator, values);

            auto key_value_packed = make_vector<char>(
                packed_size_fn(key_value_head_count, depth, lengths.data(), lengths.size()));
            // Run kernel being tested: pack once, then attention over the whole task grid
            pack_fn(keys.raw_values_data(), values.raw_values_data(), key_value_head_count, depth, offsets.data(),
                    lengths.data(), lengths.size(), key_value_stride_bytes, key_value_stride_bytes,
                    key_value_packed.raw_values_data(), 0, lengths.size() * key_value_head_count);
            auto output = make_vector<result_t>(tokens * queries_row_width);
            attention_fn(queries.raw_values_data(), key_value_packed.raw_values_data(), output.raw_values_data(),
                         head_count, key_value_head_count, depth, offsets.data(), query_stride_bytes,
                         queries_row_width * sizeof(nk_f32_t), scale, 0, total_tasks);

            auto key_value_reference = make_vector<char>(nk::attention_pack_size<scalar_t, nk::no_simd_k>(
                key_value_head_count, depth, lengths.data(), lengths.size()));
            // Compute reference through the serial backend via the C++ wrappers
            nk::attention_pack<scalar_t, nk::no_simd_k>(
                keys.values_data(), values.values_data(), key_value_head_count, depth, offsets.data(), lengths.data(),
                lengths.size(), key_value_stride_bytes, key_value_stride_bytes, key_value_reference.raw_values_data());
            auto reference = make_vector<result_t>(tokens * queries_row_width);
            nk::attention_bidirectional_packed<scalar_t, result_t, nk::no_simd_k>(
                queries.values_data(), key_value_reference.raw_values_data(), reference.values_data(), head_count,
                key_value_head_count, depth, offsets.data(), query_stride_bytes, queries_row_width * sizeof(nk_f32_t),
                scale);

            for (std::size_t index = 0; index < output.size_values(); index++)
                stats.accumulate(output[index], reference[index]);
        }
    }
    return stats;
}

/**
 *  @brief Ragged causal attention test: the reference runs the serial bidirectional kernel per query row
 *         over a pack of exactly the keys that row may see, and expects zeros for rows that see none.
 *         Two task windows cover the grid, the second one relying on `task_count` clipping.
 */
template <typename scalar_type_>
error_stats_t test_attention_causal_packed(typename scalar_type_::attention_pack_size_kernel_t packed_size_fn,
                                           typename scalar_type_::attention_pack_kernel_t pack_fn,
                                           typename scalar_type_::attention_causal_packed_kernel_t attention_fn) {
    using scalar_t = scalar_type_;
    using result_t = typename scalar_t::attention_result_t;

    error_stats_t stats(comparison_family_t::normalized_reduction_k);
    std::mt19937 generator(global_config.seed);

    std::size_t const head_count = 4, key_value_head_count = 2;
    nk_f32_t const scale = 0.05f;
    std::size_t const unbounded_window = static_cast<std::size_t>(-1);

    for (auto start = test_start_time(); within_time_budget(start);) {
        for (nk_u32_t main_length : {1u, 31u, 32u, 33u, 513u}) {
            nk_u32_t const main_queries = std::min<nk_u32_t>(main_length / 2 + 2, 24);
            std::vector<nk_u32_t> const lengths = {main_length, 0, 33};
            std::vector<nk_u32_t> const query_counts = {main_queries, 2, 1}; // long block, PAD without keys, decode
            std::vector<nk_u32_t> key_offsets = {0}, query_offsets = {0};
            for (std::size_t segment = 0; segment < lengths.size(); segment++)
                key_offsets.push_back(key_offsets.back() + lengths[segment]),
                    query_offsets.push_back(query_offsets.back() + query_counts[segment]);
            std::size_t const key_tokens = key_offsets.back(), query_tokens = query_offsets.back();
            std::size_t const total_tasks = lengths.size() * head_count;

            for (std::size_t depth : {1ul, 65ul, 128ul, 257ul}) {
                std::size_t const queries_row_width = head_count * depth,
                                  key_value_row_width = key_value_head_count * depth;
                std::size_t const query_stride_bytes = queries_row_width * sizeof(scalar_t),
                                  key_value_stride_bytes = key_value_row_width * sizeof(scalar_t),
                                  output_stride_bytes = queries_row_width * sizeof(nk_f32_t);
                auto queries = make_vector<scalar_t>(query_tokens * queries_row_width);
                auto keys = make_vector<scalar_t>(key_tokens * key_value_row_width),
                     values = make_vector<scalar_t>(key_tokens * key_value_row_width);
                fill_random(generator, queries), fill_random(generator, keys), fill_random(generator, values);

                auto key_value_packed = make_vector<char>(
                    packed_size_fn(key_value_head_count, depth, lengths.data(), lengths.size()));
                pack_fn(keys.raw_values_data(), values.raw_values_data(), key_value_head_count, depth,
                        key_offsets.data(), lengths.data(), lengths.size(), key_value_stride_bytes,
                        key_value_stride_bytes, key_value_packed.raw_values_data(), 0,
                        lengths.size() * key_value_head_count);

                std::int64_t const cache_offset = static_cast<std::int64_t>(main_length) - main_queries;
                for (std::int64_t diagonal_offset :
                     {std::int64_t(0), cache_offset, std::int64_t(-3), static_cast<std::int64_t>(main_length) + 5}) {
                    for (std::size_t window : {std::size_t(1), std::size_t(7), std::size_t(31), std::size_t(33),
                                               std::size_t(511), std::size_t(513), unbounded_window, std::size_t(0)}) {
                        auto output = make_vector<result_t>(query_tokens * queries_row_width);
                        std::size_t const first_window_tasks = total_tasks / 2;
                        // Run kernel being tested over two windows, the second one clipped to the grid
                        attention_fn(queries.raw_values_data(), key_value_packed.raw_values_data(),
                                     output.raw_values_data(), head_count, key_value_head_count, depth,
                                     query_offsets.data(), query_stride_bytes, output_stride_bytes, scale,
                                     diagonal_offset, window, 0, first_window_tasks);
                        attention_fn(queries.raw_values_data(), key_value_packed.raw_values_data(),
                                     output.raw_values_data(), head_count, key_value_head_count, depth,
                                     query_offsets.data(), query_stride_bytes, output_stride_bytes, scale,
                                     diagonal_offset, window, first_window_tasks, unbounded_window);

                        auto reference = make_vector<result_t>(query_tokens * queries_row_width);
                        for (std::size_t index = 0; index < reference.size_values(); index++) reference[index] = 0;
                        // Reference: one single-row bidirectional call per query over its visible keys
                        for (std::size_t segment = 0; segment < lengths.size(); segment++) {
                            for (std::size_t row = 0; row < query_counts[segment]; row++) {
                                std::int64_t const position = static_cast<std::int64_t>(row) + diagonal_offset;
                                if (position < 0 || window == 0) continue;
                                std::size_t const query_position = static_cast<std::size_t>(position);
                                std::size_t const key_end = std::min<std::size_t>(query_position + 1, lengths[segment]);
                                std::size_t const key_begin = window > query_position
                                                                  ? 0
                                                                  : std::min(query_position - window + 1, key_end);
                                if (key_begin == key_end) continue;
                                nk_u32_t const visible_offsets[2] = {
                                    static_cast<nk_u32_t>(key_offsets[segment] + key_begin),
                                    static_cast<nk_u32_t>(key_offsets[segment] + key_end)};
                                nk_u32_t const visible_length = static_cast<nk_u32_t>(key_end - key_begin);
                                nk_u32_t const single_query_offsets[2] = {0, 1};
                                auto key_value_reference = make_vector<char>(
                                    nk::attention_pack_size<scalar_t, nk::no_simd_k>(key_value_head_count, depth,
                                                                                     &visible_length, 1));
                                nk::attention_pack<scalar_t, nk::no_simd_k>(
                                    keys.values_data(), values.values_data(), key_value_head_count, depth,
                                    visible_offsets, &visible_length, 1, key_value_stride_bytes, key_value_stride_bytes,
                                    key_value_reference.raw_values_data());
                                std::size_t const query_row = query_offsets[segment] + row;
                                nk::attention_bidirectional_packed<scalar_t, result_t, nk::no_simd_k>(
                                    queries.values_data() + query_row * queries_row_width,
                                    key_value_reference.raw_values_data(),
                                    reference.values_data() + query_row * queries_row_width, head_count,
                                    key_value_head_count, depth, single_query_offsets, query_stride_bytes,
                                    output_stride_bytes, scale);
                            }
                        }

                        for (std::size_t index = 0; index < output.size_values(); index++)
                            stats.accumulate(output[index], reference[index]);
                    }
                }
            }
        }
    }
    return stats;
}

/**
 *  @brief Test batched Hamming distance computation with packed B matrix.
 */
template <typename scalar_type_>
error_stats_t test_hammings_packed(typename scalar_type_::hammings_pack_size_kernel_t packed_size_fn,
                                   typename scalar_type_::hammings_pack_kernel_t pack_fn,
                                   typename scalar_type_::hammings_packed_kernel_t hammings_fn) {
    using scalar_t = scalar_type_;
    using result_t = u32_t;

    error_stats_t stats(comparison_family_t::exact_k);
    std::mt19937 generator(global_config.seed);

    std::size_t m = global_config.matrix_height, n = global_config.matrix_width, k = global_config.dense_dimensions;
    std::size_t k_bytes = nk::divide_round_up(k, 8);
    std::size_t a_stride = k_bytes * sizeof(scalar_t);
    std::size_t b_stride = k_bytes * sizeof(scalar_t);
    std::size_t c_stride = n * sizeof(result_t);

    auto a = make_vector<scalar_t>(m * k), b = make_vector<scalar_t>(n * k);
    auto c = make_vector<result_t>(m * n);
    auto c_ref = make_vector<result_t>(m * n);

    nk_size_t packed_size = packed_size_fn(n, k);
    auto b_packed = make_vector<char>(packed_size);

    // Allocate buffer for reference computation
    nk_size_t packed_size_ref = nk::dots_pack_size<scalar_t, nk::no_simd_k>(n, k);
    auto b_packed_ref = make_vector<char>(packed_size_ref);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        // Run kernel being tested
        pack_fn(b.raw_values_data(), n, k, b_stride, b_packed.raw_values_data(), 0, n);
        hammings_fn(a.raw_values_data(), b_packed.raw_values_data(), c.raw_values_data(), m, n, k, a_stride, c_stride);

        // Compute reference using C++ template with no_simd_k
        nk::dots_pack<scalar_t, nk::no_simd_k>(b.values_data(), n, k, b_stride, b_packed_ref.raw_values_data());
        nk::hammings_packed<scalar_t, result_t, nk::no_simd_k>(a.values_data(), b_packed_ref.raw_values_data(),
                                                               c_ref.values_data(), m, n, k, a_stride, c_stride);

        // Hamming distances are exact integers
        for (std::size_t i = 0; i < m * n; i++) stats.accumulate(c[i], c_ref[i]);
    }
    return stats;
}

/**
 *  @brief Test symmetric Hamming distance matrix computation.
 */
template <typename scalar_type_>
error_stats_t test_hammings_symmetric(typename scalar_type_::hammings_symmetric_kernel_t kernel_fn) {
    using scalar_t = scalar_type_;
    using result_t = u32_t;

    error_stats_t stats(comparison_family_t::exact_k);
    std::mt19937 generator(global_config.seed);

    std::size_t n = global_config.matrix_height, k = global_config.dense_dimensions;
    std::size_t k_bytes = nk::divide_round_up(k, 8);
    std::size_t a_stride = k_bytes * sizeof(scalar_t);
    std::size_t c_stride = n * sizeof(result_t);

    auto a = make_vector<scalar_t>(n * k);
    auto c = make_vector<result_t>(n * n);
    auto c_ref = make_vector<result_t>(n * n);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);

        // Run kernel being tested
        kernel_fn(a.raw_values_data(), n, k, a_stride, c.raw_values_data(), c_stride, 0, n);

        // Compute reference using nk:: template
        nk::hammings_symmetric<scalar_t, result_t, nk::no_simd_k>(a.values_data(), n, k, a_stride, c_ref.values_data(),
                                                                  n * sizeof(result_t));

        // Hamming distances are exact integers — check upper triangle only
        for (std::size_t i = 0; i < n; i++)
            for (std::size_t j = i; j < n; j++) stats.accumulate(c[i * n + j], c_ref[i * n + j]);
    }
    return stats;
}

/**
 *  @brief Test batched Jaccard distance computation with packed B matrix.
 */
template <typename scalar_type_>
error_stats_t test_jaccards_packed(typename scalar_type_::jaccards_pack_size_kernel_t packed_size_fn,
                                   typename scalar_type_::jaccards_pack_kernel_t pack_fn,
                                   typename scalar_type_::jaccards_packed_kernel_t jaccards_fn) {
    using scalar_t = scalar_type_;
    using result_t = f32_t;

    error_stats_t stats(comparison_family_t::exact_k);
    std::mt19937 generator(global_config.seed);

    std::size_t m = global_config.matrix_height, n = global_config.matrix_width, k = global_config.dense_dimensions;
    std::size_t k_bytes = nk::divide_round_up(k, 8);
    std::size_t a_stride = k_bytes * sizeof(scalar_t);
    std::size_t b_stride = k_bytes * sizeof(scalar_t);
    std::size_t c_stride = n * sizeof(result_t);

    auto a = make_vector<scalar_t>(m * k), b = make_vector<scalar_t>(n * k);
    auto c = make_vector<result_t>(m * n);
    auto c_ref = make_vector<result_t>(m * n);

    nk_size_t packed_size = packed_size_fn(n, k);
    auto b_packed = make_vector<char>(packed_size);

    // Allocate buffer for reference computation
    nk_size_t packed_size_ref = nk::dots_pack_size<scalar_t, nk::no_simd_k>(n, k);
    auto b_packed_ref = make_vector<char>(packed_size_ref);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        // Run kernel being tested
        pack_fn(b.raw_values_data(), n, k, b_stride, b_packed.raw_values_data(), 0, n);
        jaccards_fn(a.raw_values_data(), b_packed.raw_values_data(), c.raw_values_data(), m, n, k, a_stride, c_stride);

        // Compute reference using C++ template with no_simd_k
        nk::dots_pack<scalar_t, nk::no_simd_k>(b.values_data(), n, k, b_stride, b_packed_ref.raw_values_data());
        nk::jaccards_packed<scalar_t, result_t, nk::no_simd_k>(a.values_data(), b_packed_ref.raw_values_data(),
                                                               c_ref.values_data(), m, n, k, a_stride, c_stride);

        for (std::size_t i = 0; i < m * n; i++) stats.accumulate(c[i], c_ref[i]);
    }
    return stats;
}

/**
 *  @brief Test symmetric Jaccard distance matrix computation.
 */
template <typename scalar_type_>
error_stats_t test_jaccards_symmetric(typename scalar_type_::jaccards_symmetric_kernel_t kernel_fn) {
    using scalar_t = scalar_type_;
    using result_t = f32_t;

    error_stats_t stats(comparison_family_t::exact_k);
    std::mt19937 generator(global_config.seed);

    std::size_t n = global_config.matrix_height, k = global_config.dense_dimensions;
    std::size_t k_bytes = nk::divide_round_up(k, 8);
    std::size_t a_stride = k_bytes * sizeof(scalar_t);
    std::size_t c_stride = n * sizeof(result_t);

    auto a = make_vector<scalar_t>(n * k);
    auto c = make_vector<result_t>(n * n);
    auto c_ref = make_vector<result_t>(n * n);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);

        // Run kernel being tested
        kernel_fn(a.raw_values_data(), n, k, a_stride, c.raw_values_data(), c_stride, 0, n);

        // Compute reference using nk:: template
        nk::jaccards_symmetric<scalar_t, result_t, nk::no_simd_k>(a.values_data(), n, k, a_stride, c_ref.values_data(),
                                                                  n * sizeof(result_t));

        // Check upper triangle only
        for (std::size_t i = 0; i < n; i++)
            for (std::size_t j = i; j < n; j++) stats.accumulate(c[i * n + j], c_ref[i * n + j]);
    }
    return stats;
}

/**
 *  @brief Test batched angular distance computation with packed B matrix.
 *  Angular distance: 1 - dot(a,b) / sqrt(sumsq(a) * sumsq(b))
 */
template <typename scalar_type_>
error_stats_t test_angulars_packed(typename scalar_type_::dots_pack_size_kernel_t packed_size_fn,
                                   typename scalar_type_::dots_pack_kernel_t pack_fn,
                                   typename scalar_type_::angulars_packed_kernel_t angulars_fn) {
    using scalar_t = scalar_type_;
    using result_t = typename scalar_t::angular_result_t;
    using reference_t = reference_for<scalar_t, result_t>;

    error_stats_t stats(comparison_family_t::approximate_k);
    std::mt19937 generator(global_config.seed);

    std::size_t m = global_config.matrix_height, n = global_config.matrix_width;
    std::size_t const dims_per_value = nk::dimensions_per_value<scalar_t>();
    std::size_t k_values = nk::divide_round_up(global_config.matrix_depth, dims_per_value);
    std::size_t k = k_values * dims_per_value;
    std::size_t a_stride = k_values * sizeof(scalar_t);
    std::size_t b_stride = k_values * sizeof(scalar_t);
    std::size_t c_stride = n * sizeof(result_t);

    auto a = make_vector<scalar_t>(m * k), b = make_vector<scalar_t>(n * k);
    auto c = make_vector<result_t>(m * n);
    auto c_ref = make_vector<reference_t>(m * n);

    nk_size_t packed_size = packed_size_fn(n, k);
    auto b_packed = make_vector<char>(packed_size);

    nk_size_t ref_pack_size = nk::dots_pack_size<scalar_t, nk::no_simd_k>(n, k);
    auto b_packed_ref = make_vector<char>(ref_pack_size);
    auto a_sumsqs = make_vector<reference_t>(m);
    auto b_sumsqs = make_vector<reference_t>(n);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        // Run kernel being tested
        pack_fn(b.raw_values_data(), n, k, b_stride, b_packed.raw_values_data(), 0, n);
        angulars_fn(a.raw_values_data(), b_packed.raw_values_data(), c.raw_values_data(), m, n, k, a_stride, c_stride);

        // Reference: compute dot products in reference precision
        nk::dots_pack<scalar_t, nk::no_simd_k>(b.values_data(), n, k, b_stride, b_packed_ref.raw_values_data());
        nk::dots_packed<scalar_t, reference_t, nk::no_simd_k>(a.values_data(), b_packed_ref.raw_values_data(),
                                                              c_ref.values_data(), m, n, k, a_stride,
                                                              n * sizeof(reference_t));

        // Compute sumsqs using reduce_moments
        reference_t sum_unused;
        for (std::size_t i = 0; i < m; ++i)
            nk::reduce_moments<scalar_t, reference_t, reference_t, nk::no_simd_k>(
                a.values_data() + i * k_values, k, sizeof(scalar_t), &sum_unused, a_sumsqs.values_data() + i);
        for (std::size_t j = 0; j < n; ++j)
            nk::reduce_moments<scalar_t, reference_t, reference_t, nk::no_simd_k>(
                b.values_data() + j * k_values, k, sizeof(scalar_t), &sum_unused, b_sumsqs.values_data() + j);

        // Convert dots to angular distances: 1 - dot / sqrt(sumsq_a * sumsq_b)
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                reference_t ab_sumsq = a_sumsqs[i] * b_sumsqs[j];
                reference_t &c_cell = c_ref[i * n + j];
                c_cell = ab_sumsq > reference_t(0) ? (reference_t(1) - c_cell * ab_sumsq.rsqrt()) : reference_t(0);
            }

        for (std::size_t i = 0; i < m * n; i++) stats.accumulate(c[i], c_ref[i]);
    }
    return stats;
}

/**
 *  @brief Test batched euclidean distance computation with packed B matrix.
 *  Euclidean distance: sqrt(max(0, sumsq(a) + sumsq(b) - 2*dot(a,b)))
 */
template <typename scalar_type_>
error_stats_t test_euclideans_packed(typename scalar_type_::dots_pack_size_kernel_t packed_size_fn,
                                     typename scalar_type_::dots_pack_kernel_t pack_fn,
                                     typename scalar_type_::euclideans_packed_kernel_t euclideans_fn) {
    using scalar_t = scalar_type_;
    using result_t = typename scalar_t::euclidean_result_t;
    using reference_t = reference_for<scalar_t, result_t>;

    error_stats_t stats(comparison_family_t::approximate_k);
    std::mt19937 generator(global_config.seed);

    std::size_t m = global_config.matrix_height, n = global_config.matrix_width;
    std::size_t const dims_per_value = nk::dimensions_per_value<scalar_t>();
    std::size_t k_values = nk::divide_round_up(global_config.matrix_depth, dims_per_value);
    std::size_t k = k_values * dims_per_value;
    std::size_t a_stride = k_values * sizeof(scalar_t);
    std::size_t b_stride = k_values * sizeof(scalar_t);
    std::size_t c_stride = n * sizeof(result_t);

    auto a = make_vector<scalar_t>(m * k), b = make_vector<scalar_t>(n * k);
    auto c = make_vector<result_t>(m * n);
    auto c_ref = make_vector<reference_t>(m * n);

    nk_size_t packed_size = packed_size_fn(n, k);
    auto b_packed = make_vector<char>(packed_size);

    nk_size_t ref_pack_size = nk::dots_pack_size<scalar_t, nk::no_simd_k>(n, k);
    auto b_packed_ref = make_vector<char>(ref_pack_size);
    auto a_sumsqs = make_vector<reference_t>(m);
    auto b_sumsqs = make_vector<reference_t>(n);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        // Run kernel being tested
        pack_fn(b.raw_values_data(), n, k, b_stride, b_packed.raw_values_data(), 0, n);
        euclideans_fn(a.raw_values_data(), b_packed.raw_values_data(), c.raw_values_data(), m, n, k, a_stride,
                      c_stride);

        // Reference: compute dot products in reference precision
        nk::dots_pack<scalar_t, nk::no_simd_k>(b.values_data(), n, k, b_stride, b_packed_ref.raw_values_data());
        nk::dots_packed<scalar_t, reference_t, nk::no_simd_k>(a.values_data(), b_packed_ref.raw_values_data(),
                                                              c_ref.values_data(), m, n, k, a_stride,
                                                              n * sizeof(reference_t));

        // Compute sumsqs using reduce_moments
        reference_t sum_unused;
        for (std::size_t i = 0; i < m; ++i)
            nk::reduce_moments<scalar_t, reference_t, reference_t, nk::no_simd_k>(
                a.values_data() + i * k_values, k, sizeof(scalar_t), &sum_unused, a_sumsqs.values_data() + i);
        for (std::size_t j = 0; j < n; ++j)
            nk::reduce_moments<scalar_t, reference_t, reference_t, nk::no_simd_k>(
                b.values_data() + j * k_values, k, sizeof(scalar_t), &sum_unused, b_sumsqs.values_data() + j);

        // Convert dots to euclidean distances: sqrt(max(0, sumsq_a + sumsq_b - 2*dot))
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                reference_t &c_cell = c_ref[i * n + j];
                reference_t diff = a_sumsqs[i] + b_sumsqs[j] - reference_t(2) * c_cell;
                c_cell = diff > reference_t(0) ? diff.sqrt() : reference_t(0);
            }

        for (std::size_t i = 0; i < m * n; i++) stats.accumulate(c[i], c_ref[i]);
    }
    return stats;
}

/**
 *  @brief Test symmetric angular distance matrix computation.
 */
template <typename scalar_type_>
error_stats_t test_angulars_symmetric(typename scalar_type_::angulars_symmetric_kernel_t kernel_fn) {
    using scalar_t = scalar_type_;
    using result_t = typename scalar_t::angular_result_t;
    using reference_t = reference_for<scalar_t, result_t>;

    error_stats_t stats(comparison_family_t::approximate_k);
    std::mt19937 generator(global_config.seed);

    std::size_t n = global_config.matrix_height;
    std::size_t const dims_per_value = nk::dimensions_per_value<scalar_t>();
    std::size_t k_values = nk::divide_round_up(global_config.matrix_depth, dims_per_value);
    std::size_t k = k_values * dims_per_value;
    std::size_t a_stride = k_values * sizeof(scalar_t);
    std::size_t c_stride = n * sizeof(result_t);

    auto a = make_vector<scalar_t>(n * k);
    auto c = make_vector<result_t>(n * n);
    auto c_ref = make_vector<reference_t>(n * n);
    auto sumsqs = make_vector<reference_t>(n);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);

        // Run kernel being tested
        kernel_fn(a.raw_values_data(), n, k, a_stride, c.raw_values_data(), c_stride, 0, n);

        // Reference: compute dots symmetric in reference precision
        nk::dots_symmetric<scalar_t, reference_t, nk::no_simd_k>(a.values_data(), n, k, a_stride, c_ref.values_data(),
                                                                 n * sizeof(reference_t));

        // Compute sumsqs using reduce_moments
        reference_t sum_unused;
        for (std::size_t i = 0; i < n; ++i)
            nk::reduce_moments<scalar_t, reference_t, reference_t, nk::no_simd_k>(
                a.values_data() + i * k_values, k, sizeof(scalar_t), &sum_unused, sumsqs.values_data() + i);

        // Convert dots to angular distances: diagonal=0, upper triangle uses formula
        for (std::size_t i = 0; i < n; ++i) {
            c_ref[i * n + i] = reference_t(0);
            for (std::size_t j = i + 1; j < n; ++j) {
                reference_t ab_sumsq = sumsqs[i] * sumsqs[j];
                reference_t &c_cell = c_ref[i * n + j];
                c_cell = ab_sumsq > reference_t(0) ? (reference_t(1) - c_cell * ab_sumsq.rsqrt()) : reference_t(0);
            }
        }

        // Only check upper triangle and diagonal
        for (std::size_t i = 0; i < n; i++)
            for (std::size_t j = i; j < n; j++) stats.accumulate(c[i * n + j], c_ref[i * n + j]);
    }
    return stats;
}

/**
 *  @brief Test symmetric euclidean distance matrix computation.
 */
template <typename scalar_type_>
error_stats_t test_euclideans_symmetric(typename scalar_type_::euclideans_symmetric_kernel_t kernel_fn) {
    using scalar_t = scalar_type_;
    using result_t = typename scalar_t::euclidean_result_t;
    using reference_t = reference_for<scalar_t, result_t>;

    error_stats_t stats(comparison_family_t::approximate_k);
    std::mt19937 generator(global_config.seed);

    std::size_t n = global_config.matrix_height;
    std::size_t const dims_per_value = nk::dimensions_per_value<scalar_t>();
    std::size_t k_values = nk::divide_round_up(global_config.matrix_depth, dims_per_value);
    std::size_t k = k_values * dims_per_value;
    std::size_t a_stride = k_values * sizeof(scalar_t);
    std::size_t c_stride = n * sizeof(result_t);

    auto a = make_vector<scalar_t>(n * k);
    auto c = make_vector<result_t>(n * n);
    auto c_ref = make_vector<reference_t>(n * n);
    auto sumsqs = make_vector<reference_t>(n);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);

        // Run kernel being tested
        kernel_fn(a.raw_values_data(), n, k, a_stride, c.raw_values_data(), c_stride, 0, n);

        // Reference: compute dots symmetric in reference precision
        nk::dots_symmetric<scalar_t, reference_t, nk::no_simd_k>(a.values_data(), n, k, a_stride, c_ref.values_data(),
                                                                 n * sizeof(reference_t));

        // Compute sumsqs using reduce_moments
        reference_t sum_unused;
        for (std::size_t i = 0; i < n; ++i)
            nk::reduce_moments<scalar_t, reference_t, reference_t, nk::no_simd_k>(
                a.values_data() + i * k_values, k, sizeof(scalar_t), &sum_unused, sumsqs.values_data() + i);

        // Convert dots to euclidean distances: diagonal=0, upper triangle uses formula
        for (std::size_t i = 0; i < n; ++i) {
            c_ref[i * n + i] = reference_t(0);
            for (std::size_t j = i + 1; j < n; ++j) {
                reference_t &c_cell = c_ref[i * n + j];
                reference_t diff = sumsqs[i] + sumsqs[j] - reference_t(2) * c_cell;
                c_cell = diff > reference_t(0) ? diff.sqrt() : reference_t(0);
            }
        }

        // Only check upper triangle and diagonal
        for (std::size_t i = 0; i < n; i++)
            for (std::size_t j = i; j < n; j++) stats.accumulate(c[i * n + j], c_ref[i * n + j]);
    }
    return stats;
}

// Forward declarations for cross functions
void test_cross_serial();
void test_cross_x86();
void test_cross_amx();
void test_cross_arm();
void test_cross_sme();
void test_cross_blas();

#endif // NK_TEST_CROSS_HPP
