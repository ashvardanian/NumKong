/**
 *  @file test/cross_blas.cpp
 *  @author Ash Vardanian
 *  @date January 14, 2025
 *  @brief Batch operation tests - BLAS/MKL comparisons.
 */
#include "numkong/dot.hpp" // `nk::dot` for BLAS comparison

#include "harness.hpp"
#include "cross.hpp"

using namespace ashvardanian::numkong::test;

#if NUMKONG_COMPARE_TO_BLAS || NUMKONG_COMPARE_TO_MKL || NUMKONG_COMPARE_TO_ACCELERATE

/**
 *  @brief Unified template to test unpacked GEMM against high-precision reference.
 *
 *  Validates BLAS/MKL/Accelerate GEMM implementations by comparing against @c nk::dots_unpacked
 *  with high-precision reference accumulation.
 *
 *  @tparam scalar_type_ Input element type (e.g., f32_t, bf16_t)
 *  @tparam accumulator_type_ Output type from BLAS kernel (e.g., f32_t for bf16 GEMM)
 *  @tparam kernel_type_ Deduced function pointer type for the BLAS kernel
 *
 *  @param[in] term_error_bound What each term may add in the precision the routine sums in
 */
template <typename scalar_type_, typename accumulator_type_, typename kernel_type_>
error_stats_t test_dots_unpacked(kernel_type_ dots_fn, nk_f64_t term_error_bound) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = accumulator_type_;
    using reference_t = bounded_reference_for<scalar_t, result_t>;

    error_stats_t stats(term_error_bound);
    std::mt19937 generator(global_config.seed);

    std::size_t m = global_config.matrix_height, n = global_config.matrix_width, k = global_config.matrix_depth;
    std::size_t a_stride = k * sizeof(raw_t);
    std::size_t b_stride = k * sizeof(raw_t);
    std::size_t c_stride = n * sizeof(typename result_t::raw_t);

    auto a_buf = make_vector<scalar_t>(m * k), b_buf = make_vector<scalar_t>(n * k);
    auto c = make_vector<result_t>(m * n);
    std::vector<reference_t> c_ref(m * n);
    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a_buf);
        fill_random(generator, b_buf);

        nk::dots_unpacked<scalar_t, reference_t>(a_buf.values_data(), b_buf.values_data(), c_ref.data(), m, n, k,
                                                 a_stride, b_stride, n * sizeof(reference_t));
        dots_fn(a_buf.values_data(), b_buf.values_data(), c.values_data(), m, n, k, a_stride, c_stride);

        for (std::size_t i = 0; i < m * n; i++) stats.accumulate(c[i], c_ref[i]);
    }
    return stats;
}

/**
 *  @brief Like test_dots_unpacked, but uses conjugated reference (C = A × B^H).
 *
 *  For complex GEMM, BLAS computes the Hermitian inner product when called with CblasConjTrans. The
 *  reference must also conjugate B to match.
 */
template <typename scalar_type_, typename accumulator_type_, typename kernel_type_>
error_stats_t test_dots_unpacked_conjugated(kernel_type_ dots_fn, nk_f64_t term_error_bound) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = accumulator_type_;
    using reference_t = bounded_reference_for<scalar_t, result_t>;

    error_stats_t stats(term_error_bound);
    std::mt19937 generator(global_config.seed);

    std::size_t m = global_config.matrix_height, n = global_config.matrix_width, k = global_config.matrix_depth;
    std::size_t a_stride = k * sizeof(raw_t);
    std::size_t b_stride = k * sizeof(raw_t);
    std::size_t c_stride = n * sizeof(typename result_t::raw_t);

    auto a_buf = make_vector<scalar_t>(m * k), b_buf = make_vector<scalar_t>(n * k);
    auto c = make_vector<result_t>(m * n);
    std::vector<reference_t> c_ref(m * n);
    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a_buf);
        fill_random(generator, b_buf);

        nk::dots_unpacked_conjugated<scalar_t, reference_t>(a_buf.values_data(), b_buf.values_data(), c_ref.data(), m,
                                                            n, k, a_stride, b_stride, n * sizeof(reference_t));
        dots_fn(a_buf.values_data(), b_buf.values_data(), c.values_data(), m, n, k, a_stride, c_stride);

        for (std::size_t i = 0; i < m * n; i++) stats.accumulate(c[i], c_ref[i]);
    }
    return stats;
}

nk_status_t dot_f32_with_blas(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result, void *) {
    *result = cblas_dsdot(static_cast<int>(n), a, 1, b, 1);
    return nk_success_k;
}

nk_status_t dot_f64_with_blas(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result, void *) {
    *result = cblas_ddot(static_cast<int>(n), a, 1, b, 1);
    return nk_success_k;
}

nk_status_t dot_f32c_with_blas(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f64c_t *result, void *) {
    nk_f32c_t reduced_result_f32;
#if NUMKONG_COMPARE_TO_ACCELERATE
    cblas_cdotu_sub(static_cast<int>(n), reinterpret_cast<__LAPACK_float_complex const *>(a), 1,
                    reinterpret_cast<__LAPACK_float_complex const *>(b), 1,
                    reinterpret_cast<__LAPACK_float_complex *>(&reduced_result_f32));
#else
    cblas_cdotu_sub(static_cast<int>(n), a, 1, b, 1, &reduced_result_f32);
#endif
    result->real = (nk_f64_t)reduced_result_f32.real, result->imag = (nk_f64_t)reduced_result_f32.imag;
    return nk_success_k;
}

nk_status_t vdot_f32c_with_blas(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f64c_t *result, void *) {
    nk_f32c_t reduced_result_f32;
#if NUMKONG_COMPARE_TO_ACCELERATE
    cblas_cdotc_sub(static_cast<int>(n), reinterpret_cast<__LAPACK_float_complex const *>(a), 1,
                    reinterpret_cast<__LAPACK_float_complex const *>(b), 1,
                    reinterpret_cast<__LAPACK_float_complex *>(&reduced_result_f32)); // conjugated
#else
    cblas_cdotc_sub(static_cast<int>(n), a, 1, b, 1, &reduced_result_f32); // conjugated
#endif
    result->real = (nk_f64_t)reduced_result_f32.real, result->imag = (nk_f64_t)reduced_result_f32.imag;
    return nk_success_k;
}

nk_status_t dot_f64c_with_blas(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result, void *) {
#if NUMKONG_COMPARE_TO_ACCELERATE
    cblas_zdotu_sub(static_cast<int>(n), reinterpret_cast<__LAPACK_double_complex const *>(a), 1,
                    reinterpret_cast<__LAPACK_double_complex const *>(b), 1,
                    reinterpret_cast<__LAPACK_double_complex *>(result));
#else
    cblas_zdotu_sub(static_cast<int>(n), a, 1, b, 1, result);
#endif
    return nk_success_k;
}

nk_status_t vdot_f64c_with_blas(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result, void *) {
#if NUMKONG_COMPARE_TO_ACCELERATE
    cblas_zdotc_sub(static_cast<int>(n), reinterpret_cast<__LAPACK_double_complex const *>(a), 1,
                    reinterpret_cast<__LAPACK_double_complex const *>(b), 1,
                    reinterpret_cast<__LAPACK_double_complex *>(result)); // conjugated
#else
    cblas_zdotc_sub(static_cast<int>(n), a, 1, b, 1, result); // conjugated
#endif
    return nk_success_k;
}

void dots_f32_with_blas(f32_t const *a, f32_t const *b, f64_t *c, nk_size_t m, nk_size_t n, nk_size_t k,
                        nk_size_t a_stride, nk_size_t c_stride) {
    nk_size_t leading_dimension_a = a_stride / sizeof(nk_f32_t);
    nk_size_t leading_dimension_c = c_stride / sizeof(nk_f64_t);
    // Reuse the first half of the f64 output buffer as a packed f32 staging matrix, then widen in place backwards.
    nk_f32_t *reduced_result_f32 = reinterpret_cast<nk_f32_t *>(c);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                1.0f, &a->raw_, static_cast<int>(leading_dimension_a), &b->raw_, static_cast<int>(k), 0.0f,
                reduced_result_f32, static_cast<int>(leading_dimension_c));
    for (std::size_t row = m; row-- > 0;)
        for (std::size_t column = n; column-- > 0;)
            c[row * leading_dimension_c + column] = f64_t(reduced_result_f32[row * leading_dimension_c + column]);
}

void dots_f64_with_blas(f64_t const *a, f64_t const *b, f64_t *c, nk_size_t m, nk_size_t n, nk_size_t k,
                        nk_size_t a_stride, nk_size_t c_stride) {
    nk_unused_(a_stride);
    nk_unused_(c_stride);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                1.0, &a->raw_, static_cast<int>(k), &b->raw_, static_cast<int>(k), 0.0, &c->raw_, static_cast<int>(n));
}

void dots_f32c_with_blas(f32c_t const *a, f32c_t const *b, f32c_t *c, nk_size_t m, nk_size_t n, nk_size_t k,
                         nk_size_t a_stride, nk_size_t c_stride) {
    nk_unused_(a_stride);
    nk_unused_(c_stride);
    nk_f32c_t alpha = {1.0f, 0.0f}, beta = {0.0f, 0.0f};
#if NUMKONG_COMPARE_TO_ACCELERATE
    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, static_cast<int>(m), static_cast<int>(n),
                static_cast<int>(k), reinterpret_cast<__LAPACK_float_complex const *>(&alpha),
                reinterpret_cast<__LAPACK_float_complex const *>(&a->raw_), static_cast<int>(k),
                reinterpret_cast<__LAPACK_float_complex const *>(&b->raw_), static_cast<int>(k),
                reinterpret_cast<__LAPACK_float_complex const *>(&beta),
                reinterpret_cast<__LAPACK_float_complex *>(&c->raw_), static_cast<int>(n));
#else
    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, static_cast<int>(m), static_cast<int>(n),
                static_cast<int>(k), &alpha, &a->raw_, static_cast<int>(k), &b->raw_, static_cast<int>(k), &beta,
                &c->raw_, static_cast<int>(n));
#endif
}

void dots_f64c_with_blas(f64c_t const *a, f64c_t const *b, f64c_t *c, nk_size_t m, nk_size_t n, nk_size_t k,
                         nk_size_t a_stride, nk_size_t c_stride) {
    nk_unused_(a_stride);
    nk_unused_(c_stride);
    nk_f64c_t alpha = {1.0, 0.0}, beta = {0.0, 0.0};
#if NUMKONG_COMPARE_TO_ACCELERATE
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, static_cast<int>(m), static_cast<int>(n),
                static_cast<int>(k), reinterpret_cast<__LAPACK_double_complex const *>(&alpha),
                reinterpret_cast<__LAPACK_double_complex const *>(&a->raw_), static_cast<int>(k),
                reinterpret_cast<__LAPACK_double_complex const *>(&b->raw_), static_cast<int>(k),
                reinterpret_cast<__LAPACK_double_complex const *>(&beta),
                reinterpret_cast<__LAPACK_double_complex *>(&c->raw_), static_cast<int>(n));
#else
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, static_cast<int>(m), static_cast<int>(n),
                static_cast<int>(k), &alpha, &a->raw_, static_cast<int>(k), &b->raw_, static_cast<int>(k), &beta,
                &c->raw_, static_cast<int>(n));
#endif
}

/** SYRK over all of A into scratch, copying back the upper triangle of only the requested rows. */
template <typename scalar_type_>
nk_status_t dots_symmetric_with_blas(scalar_type_ const *a, nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_f64_t *c,
                                     nk_size_t c_stride, nk_size_t row_start, nk_size_t row_count, void *) {
    std::vector<scalar_type_> full(n * n);
    int const size = static_cast<int>(n), depth = static_cast<int>(k);
    int const leading_dimension_a = static_cast<int>(a_stride / sizeof(scalar_type_));
    if constexpr (std::is_same_v<scalar_type_, nk_f32_t>)
        cblas_ssyrk(CblasRowMajor, CblasUpper, CblasNoTrans, size, depth, 1, a, leading_dimension_a, 0, full.data(),
                    size);
    else
        cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans, size, depth, 1, a, leading_dimension_a, 0, full.data(),
                    size);
    for (nk_size_t row = row_start; row < std::min(n, row_start + row_count); row++)
        std::copy(&full[row * n + row], &full[row * n] + n, c + row * (c_stride / sizeof(nk_f64_t)) + row);
    return nk_success_k;
}

#endif // NUMKONG_COMPARE_TO_BLAS || NUMKONG_COMPARE_TO_MKL || NUMKONG_COMPARE_TO_ACCELERATE

#if NUMKONG_COMPARE_TO_MKL
void dots_bf16_with_mkl(bf16_t const *a, bf16_t const *b, f32_t *c, nk_size_t m, nk_size_t n, nk_size_t k,
                        nk_size_t a_stride, nk_size_t c_stride) {
    nk_unused_(a_stride);
    nk_unused_(c_stride);
    cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<MKL_INT>(m), static_cast<MKL_INT>(n),
                           static_cast<MKL_INT>(k), 1.0f, &a->raw_, static_cast<MKL_INT>(k), &b->raw_,
                           static_cast<MKL_INT>(k), 0.0f, &c->raw_, static_cast<MKL_INT>(n));
}

void dots_f16_with_mkl(f16_t const *a, f16_t const *b, f32_t *c, nk_size_t m, nk_size_t n, nk_size_t k,
                       nk_size_t a_stride, nk_size_t c_stride) {
    nk_unused_(a_stride);
    nk_unused_(c_stride);
    cblas_gemm_f16f16f32(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<MKL_INT>(m), static_cast<MKL_INT>(n),
                         static_cast<MKL_INT>(k), 1.0f, reinterpret_cast<MKL_F16 const *>(&a->raw_),
                         static_cast<MKL_INT>(k), reinterpret_cast<MKL_F16 const *>(&b->raw_), static_cast<MKL_INT>(k),
                         0.0f, &c->raw_, static_cast<MKL_INT>(n));
}

void dots_i16_with_mkl(i16_t const *a, i16_t const *b, i32_t *c, nk_size_t m, nk_size_t n, nk_size_t k,
                       nk_size_t a_stride, nk_size_t c_stride) {
    nk_unused_(a_stride);
    nk_unused_(c_stride);
    MKL_INT32 c_offset = 0;
    cblas_gemm_s16s16s32(CblasRowMajor, CblasNoTrans, CblasTrans, CblasFixOffset, static_cast<MKL_INT>(m),
                         static_cast<MKL_INT>(n), static_cast<MKL_INT>(k), 1.0f, &a->raw_, static_cast<MKL_INT>(k), 0,
                         &b->raw_, static_cast<MKL_INT>(k), 0, 0.0f, &c->raw_, static_cast<MKL_INT>(n), &c_offset);
}

#endif // NUMKONG_COMPARE_TO_MKL

/** Single dot product test for BLAS, each term adding up to @p term_error_bound in the precision
 *  the routine sums in. */
template <typename scalar_type_>
error_stats_t test_dot_blas(typename scalar_type_::dot_kernel_t kernel, nk_f64_t term_error_bound) {
    using scalar_t = scalar_type_;
    using result_t = typename scalar_t::dot_result_t;
    using reference_t = bounded_reference_for<scalar_t, result_t>;

    error_stats_t stats(term_error_bound);
    std::mt19937 generator(global_config.seed);
    auto a = make_vector<scalar_t>(global_config.dense_dimensions),
         b = make_vector<scalar_t>(global_config.dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        result_t result;
        stats.expect(
            kernel(a.raw_values_data(), b.raw_values_data(), global_config.dense_dimensions, &result.raw_, nullptr));

        reference_t reference;
        stats.expect(nk::dot<scalar_t, reference_t>(a.values_data(), b.values_data(), global_config.dense_dimensions,
                                                    &reference, no_tiers_k));

        stats.accumulate(result, reference);
    }
    return stats;
}

/** Conjugate dot product test for BLAS (vdot = conj(a) * b). */
template <typename scalar_type_>
error_stats_t test_vdot_blas(typename scalar_type_::vdot_kernel_t kernel, nk_f64_t term_error_bound) {
    using scalar_t = scalar_type_;
    using result_t = typename scalar_t::vdot_result_t;
    using reference_t = bounded_reference_for<scalar_t, result_t>;

    error_stats_t stats(term_error_bound);
    std::mt19937 generator(global_config.seed);
    auto a = make_vector<scalar_t>(global_config.dense_dimensions),
         b = make_vector<scalar_t>(global_config.dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        result_t result;
        stats.expect(
            kernel(a.raw_values_data(), b.raw_values_data(), global_config.dense_dimensions, &result.raw_, nullptr));

        reference_t reference;
        stats.expect(nk::vdot<scalar_t, reference_t>(a.values_data(), b.values_data(), global_config.dense_dimensions,
                                                     &reference, no_tiers_k));

        stats.accumulate(result, reference);
    }
    return stats;
}

void test_cross_blas() {
#if NUMKONG_COMPARE_TO_BLAS || NUMKONG_COMPARE_TO_MKL || NUMKONG_COMPARE_TO_ACCELERATE
    error_stats_section_t check;
    check.section("Cross External Baselines", nk_cap_serial_k);

    // Each routine is held to the precision it sums in, which for `dsdot` is F64 over F32 inputs
    nk_f64_t const in_f64 = nk_accumulation_error_bound(nk_f64_k), in_f32 = nk_accumulation_error_bound(nk_f32_k);
    check("dot_with_blas_f64", test_dot_blas<f64_t>, dot_f64_with_blas, in_f64);
    check("dot_with_blas_f32", test_dot_blas<f32_t>, dot_f32_with_blas, in_f64);
    check("dot_with_blas_f32c", test_dot_blas<f32c_t>, dot_f32c_with_blas, in_f32);
    check("vdot_with_blas_f32c", test_vdot_blas<f32c_t>, vdot_f32c_with_blas, in_f32);
    check("dot_with_blas_f64c", test_dot_blas<f64c_t>, dot_f64c_with_blas, in_f64);
    check("vdot_with_blas_f64c", test_vdot_blas<f64c_t>, vdot_f64c_with_blas, in_f64);

    // BLAS/MKL/Accelerate GEMM precision comparison
    check("dots_with_blas_f64", test_dots_unpacked<f64_t, f64_t, decltype(&dots_f64_with_blas)>, dots_f64_with_blas,
          in_f64);
    check("dots_with_blas_f32", test_dots_unpacked<f32_t, f64_t, decltype(&dots_f32_with_blas)>, dots_f32_with_blas,
          in_f32);
    check("dots_with_blas_f32c", test_dots_unpacked_conjugated<f32c_t, f32c_t, decltype(&dots_f32c_with_blas)>,
          dots_f32c_with_blas, in_f32);
    check("dots_with_blas_f64c", test_dots_unpacked_conjugated<f64c_t, f64c_t, decltype(&dots_f64c_with_blas)>,
          dots_f64c_with_blas, in_f64);

    // BLAS SYRK precision comparison (symmetric A x A^T)
    check("dots_symmetric_with_blas_f64",
          [&] { return test_dots_symmetric<f64_t>(dots_symmetric_with_blas<nk_f64_t>, in_f64); });
    check("dots_symmetric_with_blas_f32",
          [&] { return test_dots_symmetric<f32_t>(dots_symmetric_with_blas<nk_f32_t>, in_f32); });
#endif

#if NUMKONG_COMPARE_TO_MKL
    // MKL-specific GEMM with widening accumulation
    check("dots_with_mkl_bf16", test_dots_unpacked<bf16_t, f32_t, decltype(&dots_bf16_with_mkl)>, dots_bf16_with_mkl,
          nk_accumulation_error_bound(nk_f32_k));
    check("dots_with_mkl_f16", test_dots_unpacked<f16_t, f32_t, decltype(&dots_f16_with_mkl)>, dots_f16_with_mkl,
          nk_accumulation_error_bound(nk_f32_k));
    check("dots_with_mkl_i16", test_dots_unpacked<i16_t, i32_t, decltype(&dots_i16_with_mkl)>, dots_i16_with_mkl,
          nk_accumulation_error_bound(nk_i32_k));
#endif
}
