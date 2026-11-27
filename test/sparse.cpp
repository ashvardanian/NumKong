/**
 *  @file test/sparse.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *  @brief Sparse operations tests.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include "harness.hpp"
#include "numkong/sparse.hpp"
#include "numkong/random.hpp" // `nk::fill_sorted_unique`

namespace ashvardanian::numkong::test {

/** The indices two sorted runs share, written to @p result when it is not null, and their count. */
template <typename index_type_>
void sparse_intersect_reference(index_type_ const *a, index_type_ const *b, std::size_t a_length, std::size_t b_length,
                                index_type_ *result, std::size_t *count) noexcept {
    std::size_t matches = 0, i = 0, j = 0;
    while (i < a_length && j < b_length) {
        if (a[i] < b[j]) i++;
        else if (b[j] < a[i]) j++;
        else {
            if (result) result[matches] = a[i];
            matches++, i++, j++;
        }
    }
    *count = matches;
}

/** Σ of weight products over the indices two sorted runs share, in @p result_type_. */
template <typename index_type_, typename weight_type_, typename result_type_>
void sparse_dot_reference(index_type_ const *a, index_type_ const *b, weight_type_ const *a_weights,
                          weight_type_ const *b_weights, std::size_t a_length, std::size_t b_length,
                          result_type_ *product) noexcept {
    result_type_ sum {};
    std::size_t i = 0, j = 0;
    while (i < a_length && j < b_length) {
        if (a[i] < b[j]) i++;
        else if (b[j] < a[i]) j++;
        else sum = fma_reference<weight_type_, result_type_>(a_weights[i], b_weights[j], sum), i++, j++;
    }
    *product = sum;
}

/** Test set intersection, a unified template for u16/u32 index types. */
template <typename index_type_>
error_stats_t test_intersect(typename index_type_::sparse_intersect_kernel_t kernel) {
    using index_t = index_type_;
    error_stats_t stats(comparison_family_t::exact_k);
    split_mix_generator_t generator(case_stream_key());
    std::size_t dim = test_run.settings.sparse_dimensions;
    auto a = make_vector<index_t>(dim), b = make_vector<index_t>(dim);
    auto matched = make_vector<index_t>(dim), expected = make_vector<index_t>(dim);

    for (deadline_t const deadline = test_run.settings.deadline(); deadline.ahead();) {
        // Every 8th case crosses the 64:1 ratio where the serial kernel switches to galloping
        // search, and `a`'s wider range makes it overshoot `b`'s last element.
        std::size_t b_length = 1 + generator() % dim;
        std::size_t a_length = (generator() % 8) ? 1 + generator() % dim : 1 + b_length / 128;
        nk::fill_sorted_unique(generator, a.values_data(), a_length, index_t(dim * 4));
        nk::fill_sorted_unique(generator, b.values_data(), b_length, index_t(dim * 2));

        // Same input twice: counting and storing take separate paths through the block walk
        nk_size_t count, stored_count;
        stats.expect_status(
            kernel(a.raw_values_data(), b.raw_values_data(), a_length, b_length, nullptr, &count, nullptr));
        stats.expect_status(kernel(a.raw_values_data(), b.raw_values_data(), a_length, b_length,
                                   matched.raw_values_data(), &stored_count, nullptr));

        std::size_t ref;
        sparse_intersect_reference<index_t>(a.values_data(), b.values_data(), a_length, b_length,
                                            expected.values_data(), &ref);
        stats.accumulate(count, ref);
        stats.accumulate(stored_count, ref);
        for (nk_size_t k = 0; k < ref; ++k) stats.accumulate(matched[k], expected[k]);
    }
    return stats;
}

/**
 *  @brief Test sparse dot product (unified template, parameterized by weight type).
 *
 *  Dispatch is by weight type, matching the numkong.h dispatch tables:
 *  - bf16_t weights → u16_t indices
 *  - f32_t weights → u32_t indices
 */
template <typename weight_type_>
error_stats_t test_sparse_dot(typename weight_type_::sparse_dot_kernel_t kernel) {
    using weight_t = weight_type_;
    using index_t = typename weight_t::sparse_dot_index_t;
    using reference_t = tracked<reference_for<weight_t>>;

    error_stats_t stats(nk_sparse_dot_error_bound(weight_t::dtype()));
    split_mix_generator_t generator(case_stream_key());
    std::size_t dim = test_run.settings.sparse_dimensions;
    auto a_idx = make_vector<index_t>(dim), b_idx = make_vector<index_t>(dim);
    auto a_weights = make_vector<weight_t>(dim), b_weights = make_vector<weight_t>(dim);

    for (deadline_t const deadline = test_run.settings.deadline(); deadline.ahead();) {
        nk::fill_sorted_unique(generator, a_idx.values_data(), a_idx.size_values(), index_t(dim * 4));
        nk::fill_sorted_unique(generator, b_idx.values_data(), b_idx.size_values(), index_t(dim * 4));
        fill_random(generator, a_weights);
        fill_random(generator, b_weights);

        typename weight_t::dot_result_t result;
        stats.expect_status(kernel(a_idx.raw_values_data(), b_idx.raw_values_data(), a_weights.raw_values_data(),
                                   b_weights.raw_values_data(), dim, dim, &result.raw_, nullptr));

        reference_t ref;
        sparse_dot_reference<index_t, weight_t, reference_t>(
            a_idx.values_data(), b_idx.values_data(), a_weights.values_data(), b_weights.values_data(), dim, dim, &ref);
        stats.accumulate(result, ref);
    }
    return stats;
}

void run_sparse_cases() {
    auto const serial = nk_cap_serial_k;

    run_kernel_case("nk_sparse_intersect_u16_serial", test_intersect<u16_t>, serial, nk_sparse_intersect_u16_serial);
    run_kernel_case("nk_sparse_intersect_u32_serial", test_intersect<u32_t>, serial, nk_sparse_intersect_u32_serial);
    run_kernel_case("nk_sparse_intersect_u64_serial", test_intersect<u64_t>, serial, nk_sparse_intersect_u64_serial);
    run_kernel_case("nk_sparse_dot_u32f32_serial", test_sparse_dot<f32_t>, serial, nk_sparse_dot_u32f32_serial);
    run_kernel_case("nk_sparse_dot_u16bf16_serial", test_sparse_dot<bf16_t>, serial, nk_sparse_dot_u16bf16_serial);

    run_kernel_case("nk_sparse_intersect_u16_best", test_intersect<u16_t>, serial,
                    cpu_best<nk_sparse_intersect_u16_best>);
    run_kernel_case("nk_sparse_intersect_u32_best", test_intersect<u32_t>, serial,
                    cpu_best<nk_sparse_intersect_u32_best>);
    run_kernel_case("nk_sparse_intersect_u64_best", test_intersect<u64_t>, serial,
                    cpu_best<nk_sparse_intersect_u64_best>);
    run_kernel_case("nk_sparse_dot_u32f32_best", test_sparse_dot<f32_t>, serial, cpu_best<nk_sparse_dot_u32f32_best>);
    run_kernel_case("nk_sparse_dot_u16bf16_best", test_sparse_dot<bf16_t>, serial,
                    cpu_best<nk_sparse_dot_u16bf16_best>);

#if NUMKONG_TARGET_NEON
    run_kernel_case("nk_sparse_intersect_u16_neon", test_intersect<u16_t>, nk_cap_neon_k, nk_sparse_intersect_u16_neon);
    run_kernel_case("nk_sparse_intersect_u32_neon", test_intersect<u32_t>, nk_cap_neon_k, nk_sparse_intersect_u32_neon);
    run_kernel_case("nk_sparse_intersect_u64_neon", test_intersect<u64_t>, nk_cap_neon_k, nk_sparse_intersect_u64_neon);
    run_kernel_case("nk_sparse_dot_u32f32_neon", test_sparse_dot<f32_t>, nk_cap_neon_k, nk_sparse_dot_u32f32_neon);
    run_kernel_case("nk_sparse_dot_u16bf16_neon", test_sparse_dot<bf16_t>, nk_cap_neon_k, nk_sparse_dot_u16bf16_neon);
#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_SVE2
    run_kernel_case("nk_sparse_intersect_u16_sve2", test_intersect<u16_t>, nk_cap_sve2_k, nk_sparse_intersect_u16_sve2);
    run_kernel_case("nk_sparse_intersect_u32_sve2", test_intersect<u32_t>, nk_cap_sve2_k, nk_sparse_intersect_u32_sve2);
    run_kernel_case("nk_sparse_intersect_u64_sve2", test_intersect<u64_t>, nk_cap_sve2_k, nk_sparse_intersect_u64_sve2);
    run_kernel_case("nk_sparse_dot_u32f32_sve2", test_sparse_dot<f32_t>, nk_cap_sve2_k, nk_sparse_dot_u32f32_sve2);
    run_kernel_case("nk_sparse_dot_u16bf16_sve2", test_sparse_dot<bf16_t>, nk_cap_sve2_k, nk_sparse_dot_u16bf16_sve2);
#endif // NUMKONG_TARGET_SVE2

#if NUMKONG_TARGET_ICELAKE
    auto const icelake = nk_cap_icelake_k;
    run_kernel_case("nk_sparse_intersect_u16_icelake", test_intersect<u16_t>, icelake, nk_sparse_intersect_u16_icelake);
    run_kernel_case("nk_sparse_intersect_u32_icelake", test_intersect<u32_t>, icelake, nk_sparse_intersect_u32_icelake);
    run_kernel_case("nk_sparse_intersect_u64_icelake", test_intersect<u64_t>, icelake, nk_sparse_intersect_u64_icelake);
    run_kernel_case("nk_sparse_dot_u32f32_icelake", test_sparse_dot<f32_t>, icelake, nk_sparse_dot_u32f32_icelake);
#endif // NUMKONG_TARGET_ICELAKE

#if NUMKONG_TARGET_HASWELL
    auto const haswell = nk_cap_haswell_k;
    run_kernel_case("nk_sparse_intersect_u16_haswell", test_intersect<u16_t>, haswell, nk_sparse_intersect_u16_haswell);
    run_kernel_case("nk_sparse_intersect_u32_haswell", test_intersect<u32_t>, haswell, nk_sparse_intersect_u32_haswell);
    run_kernel_case("nk_sparse_intersect_u64_haswell", test_intersect<u64_t>, haswell, nk_sparse_intersect_u64_haswell);
    run_kernel_case("nk_sparse_dot_u32f32_haswell", test_sparse_dot<f32_t>, haswell, nk_sparse_dot_u32f32_haswell);
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_TURIN
    auto const turin = nk_cap_turin_k;
    run_kernel_case("nk_sparse_intersect_u16_turin", test_intersect<u16_t>, turin, nk_sparse_intersect_u16_turin);
    run_kernel_case("nk_sparse_intersect_u32_turin", test_intersect<u32_t>, turin, nk_sparse_intersect_u32_turin);
    run_kernel_case("nk_sparse_intersect_u64_turin", test_intersect<u64_t>, turin, nk_sparse_intersect_u64_turin);
    run_kernel_case("nk_sparse_dot_u32f32_turin", test_sparse_dot<f32_t>, turin, nk_sparse_dot_u32f32_turin);
    run_kernel_case("nk_sparse_dot_u16bf16_turin", test_sparse_dot<bf16_t>, turin, nk_sparse_dot_u16bf16_turin);
#endif // NUMKONG_TARGET_TURIN
}

} // namespace ashvardanian::numkong::test
