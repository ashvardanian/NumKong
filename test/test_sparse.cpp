/**
 *  @brief Sparse operations tests.
 *  @file test/test_sparse.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 */

#include "test.hpp"
#include "numkong/sparse.hpp"

/**
 *  @brief Test set intersection (unified template for u16/u32 index types).
 */
template <typename index_type_>
error_stats_t test_intersect(typename index_type_::sparse_intersect_kernel_t kernel) {
    using index_t = index_type_;
    error_stats_t stats(comparison_family_t::exact_k);
    std::mt19937 generator(global_config.seed);
    std::size_t dim = global_config.sparse_dimensions;
    auto a = make_vector<index_t>(dim), b = make_vector<index_t>(dim);
    auto matched = make_vector<index_t>(dim), expected = make_vector<index_t>(dim);

    for (auto start = test_start_time(); within_time_budget(start);) {
        // Every 8th case crosses the 64:1 ratio where the serial kernel switches to galloping
        // search, and `a`'s wider range makes it overshoot `b`'s last element.
        std::size_t b_length = 1 + generator() % dim;
        std::size_t a_length = (generator() % 8) ? 1 + generator() % dim : 1 + b_length / 128;
        nk::fill_sorted_unique(generator, a.values_data(), a_length, index_t(dim * 4));
        nk::fill_sorted_unique(generator, b.values_data(), b_length, index_t(dim * 2));

        nk_size_t count;
        kernel(a.raw_values_data(), b.raw_values_data(), a_length, b_length, matched.raw_values_data(), &count);

        nk_size_t ref;
        nk::sparse_intersect<index_t, nk::no_simd_k>(a.values_data(), b.values_data(), a_length, b_length,
                                                     expected.values_data(), &ref);
        stats.accumulate(count, ref);
        for (nk_size_t k = 0; k < ref; ++k) stats.accumulate(matched[k], expected[k]);
    }
    return stats;
}

/**
 *  @brief Test sparse dot product (unified template, parameterized by weight type).
 *
 *  Dispatch is by weight type (matching numkong.h dispatch tables):
 *  - bf16_t weights -> u16_t indices
 *  - f32_t weights -> u32_t indices
 */
template <typename weight_type_>
error_stats_t test_sparse_dot(typename weight_type_::sparse_dot_kernel_t kernel) {
    using weight_t = weight_type_;
    using index_t = typename weight_t::sparse_dot_index_t;
    using reference_t = reference_for<weight_t>;

    error_stats_t stats(comparison_family_t::mixed_precision_reduction_k);
    std::mt19937 generator(global_config.seed);
    std::size_t dim = global_config.sparse_dimensions;
    auto a_idx = make_vector<index_t>(dim), b_idx = make_vector<index_t>(dim);
    auto a_weights = make_vector<weight_t>(dim), b_weights = make_vector<weight_t>(dim);

    for (auto start = test_start_time(); within_time_budget(start);) {
        nk::fill_sorted_unique(generator, a_idx.values_data(), a_idx.size_values(), index_t(dim * 4));
        nk::fill_sorted_unique(generator, b_idx.values_data(), b_idx.size_values(), index_t(dim * 4));
        fill_random(generator, a_weights);
        fill_random(generator, b_weights);

        typename weight_t::dot_result_t result;
        kernel(a_idx.raw_values_data(), b_idx.raw_values_data(), a_weights.raw_values_data(),
               b_weights.raw_values_data(), dim, dim, &result.raw_);

        reference_t ref;
        nk::sparse_dot<index_t, weight_t, reference_t, nk::no_simd_k>(
            a_idx.values_data(), b_idx.values_data(), a_weights.values_data(), b_weights.values_data(), dim, dim, &ref);
        stats.accumulate(result, ref);
    }
    return stats;
}

void test_sparse() {
    error_stats_section_t check("Sparse Operations");

#if NK_DYNAMIC_DISPATCH
    check("sparse_intersect_u16", test_intersect<u16_t>, nk_sparse_intersect_u16);
    check("sparse_intersect_u32", test_intersect<u32_t>, nk_sparse_intersect_u32);
    check("sparse_intersect_u64", test_intersect<u64_t>, nk_sparse_intersect_u64);
    check("sparse_dot_u32f32", test_sparse_dot<f32_t>, nk_sparse_dot_u32f32);
    check("sparse_dot_u16bf16", test_sparse_dot<bf16_t>, nk_sparse_dot_u16bf16);
#else

#if NK_TARGET_NEON
    check("sparse_intersect_u16_neon", test_intersect<u16_t>, nk_sparse_intersect_u16_neon);
    check("sparse_intersect_u32_neon", test_intersect<u32_t>, nk_sparse_intersect_u32_neon);
    check("sparse_intersect_u64_neon", test_intersect<u64_t>, nk_sparse_intersect_u64_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_SVE2
    check("sparse_intersect_u16_sve2", test_intersect<u16_t>, nk_sparse_intersect_u16_sve2);
    check("sparse_intersect_u32_sve2", test_intersect<u32_t>, nk_sparse_intersect_u32_sve2);
    check("sparse_intersect_u64_sve2", test_intersect<u64_t>, nk_sparse_intersect_u64_sve2);
    check("sparse_dot_u32f32_sve2", test_sparse_dot<f32_t>, nk_sparse_dot_u32f32_sve2);
    check("sparse_dot_u16bf16_sve2", test_sparse_dot<bf16_t>, nk_sparse_dot_u16bf16_sve2);
#endif // NK_TARGET_SVE2

#if NK_TARGET_ICELAKE
    check("sparse_intersect_u16_icelake", test_intersect<u16_t>, nk_sparse_intersect_u16_icelake);
    check("sparse_intersect_u32_icelake", test_intersect<u32_t>, nk_sparse_intersect_u32_icelake);
    check("sparse_intersect_u64_icelake", test_intersect<u64_t>, nk_sparse_intersect_u64_icelake);
    check("sparse_dot_u32f32_icelake", test_sparse_dot<f32_t>, nk_sparse_dot_u32f32_icelake);
#endif // NK_TARGET_ICELAKE

#if NK_TARGET_TURIN
    check("sparse_intersect_u16_turin", test_intersect<u16_t>, nk_sparse_intersect_u16_turin);
    check("sparse_intersect_u32_turin", test_intersect<u32_t>, nk_sparse_intersect_u32_turin);
    check("sparse_intersect_u64_turin", test_intersect<u64_t>, nk_sparse_intersect_u64_turin);
    check("sparse_dot_u32f32_turin", test_sparse_dot<f32_t>, nk_sparse_dot_u32f32_turin);
    check("sparse_dot_u16bf16_turin", test_sparse_dot<bf16_t>, nk_sparse_dot_u16bf16_turin);
#endif // NK_TARGET_TURIN

    // Serial always runs - baseline test
    check("sparse_intersect_u16_serial", test_intersect<u16_t>, nk_sparse_intersect_u16_serial);
    check("sparse_intersect_u32_serial", test_intersect<u32_t>, nk_sparse_intersect_u32_serial);
    check("sparse_intersect_u64_serial", test_intersect<u64_t>, nk_sparse_intersect_u64_serial);
    check("sparse_dot_u32f32_serial", test_sparse_dot<f32_t>, nk_sparse_dot_u32f32_serial);
    check("sparse_dot_u16bf16_serial", test_sparse_dot<bf16_t>, nk_sparse_dot_u16bf16_serial);

#endif // NK_DYNAMIC_DISPATCH
}
