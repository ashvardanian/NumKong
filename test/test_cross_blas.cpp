/**
 *  @brief Batch operation tests - BLAS/MKL comparisons.
 *  @file test/test_cross_blas.cpp
 *  @author Ash Vardanian
 *  @date January 14, 2025
 */
#include "test.hpp"
#include "test_cross.hpp"
#include "numkong/dot.hpp" // `nk::dot` for BLAS comparison

#if NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE

/**
 *  @brief Unified template to test unpacked GEMM against high-precision reference.
 *
 *  Validates BLAS/MKL/Accelerate GEMM implementations by comparing against
 *  nk::dots_unpacked with high-precision reference accumulation.
 *
 *  @tparam scalar_type_ Input element type (e.g., f32_t, bf16_t)
 *  @tparam accumulator_type_ Output type from BLAS kernel (e.g., f32_t for bf16 GEMM)
 *  @tparam kernel_type_ Deduced function pointer type for the BLAS kernel
 */
template <typename scalar_type_, typename accumulator_type_, typename kernel_type_>
error_stats_t test_dots_unpacked(kernel_type_ dots_fn) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = accumulator_type_;
    using reference_t = reference_for<scalar_t, result_t>;

    error_stats_t stats(comparison_family_t::approximate_k);
    std::mt19937 generator(global_config.seed);

    std::size_t m = global_config.matrix_height, n = global_config.matrix_width, k = global_config.matrix_depth;
    std::size_t a_stride = k * sizeof(raw_t);
    std::size_t b_stride = k * sizeof(raw_t);
    std::size_t c_stride = n * sizeof(typename result_t::raw_t);

    auto a_buf = make_vector<scalar_t>(m * k), b_buf = make_vector<scalar_t>(n * k);
    auto c = make_vector<result_t>(m * n);
    auto c_ref = make_vector<reference_t>(m * n);
    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a_buf);
        fill_random(generator, b_buf);

        nk::dots_unpacked<scalar_t, reference_t>(a_buf.values_data(), b_buf.values_data(), c_ref.values_data(), m, n, k,
                                                 a_stride, b_stride, n * sizeof(reference_t));
        dots_fn(a_buf.values_data(), b_buf.values_data(), c.values_data(), m, n, k, a_stride, c_stride);

        for (std::size_t i = 0; i < m * n; i++) stats.accumulate(c[i], c_ref[i]);
    }
    return stats;
}

/**
 *  @brief Like test_dots_unpacked, but uses conjugated reference (C = A × B^H).
 *
 *  For complex GEMM, BLAS computes the Hermitian inner product when called with
 *  CblasConjTrans. The reference must also conjugate B to match.
 */
template <typename scalar_type_, typename accumulator_type_, typename kernel_type_>
error_stats_t test_dots_unpacked_conjugated(kernel_type_ dots_fn) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = accumulator_type_;
    using reference_t = reference_for<scalar_t, result_t>;

    error_stats_t stats(comparison_family_t::approximate_k);
    std::mt19937 generator(global_config.seed);

    std::size_t m = global_config.matrix_height, n = global_config.matrix_width, k = global_config.matrix_depth;
    std::size_t a_stride = k * sizeof(raw_t);
    std::size_t b_stride = k * sizeof(raw_t);
    std::size_t c_stride = n * sizeof(typename result_t::raw_t);

    auto a_buf = make_vector<scalar_t>(m * k), b_buf = make_vector<scalar_t>(n * k);
    auto c = make_vector<result_t>(m * n);
    auto c_ref = make_vector<reference_t>(m * n);
    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a_buf);
        fill_random(generator, b_buf);

        nk::dots_unpacked_conjugated<scalar_t, reference_t>(a_buf.values_data(), b_buf.values_data(),
                                                            c_ref.values_data(), m, n, k, a_stride, b_stride,
                                                            n * sizeof(reference_t));
        dots_fn(a_buf.values_data(), b_buf.values_data(), c.values_data(), m, n, k, a_stride, c_stride);

        for (std::size_t i = 0; i < m * n; i++) stats.accumulate(c[i], c_ref[i]);
    }
    return stats;
}

void dot_f32_with_blas(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result) {
    *result = cblas_dsdot(static_cast<int>(n), a, 1, b, 1);
}

void dot_f64_with_blas(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    *result = cblas_ddot(static_cast<int>(n), a, 1, b, 1);
}

void dot_f32c_with_blas(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f64c_t *result) {
    nk_f32c_t reduced_result_f32;
#if NK_COMPARE_TO_ACCELERATE
    cblas_cdotu_sub(static_cast<int>(n), reinterpret_cast<__LAPACK_float_complex const *>(a), 1,
                    reinterpret_cast<__LAPACK_float_complex const *>(b), 1,
                    reinterpret_cast<__LAPACK_float_complex *>(&reduced_result_f32));
#else
    cblas_cdotu_sub(static_cast<int>(n), a, 1, b, 1, &reduced_result_f32);
#endif
    result->real = (nk_f64_t)reduced_result_f32.real, result->imag = (nk_f64_t)reduced_result_f32.imag;
}

void vdot_f32c_with_blas(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f64c_t *result) {
    nk_f32c_t reduced_result_f32;
#if NK_COMPARE_TO_ACCELERATE
    cblas_cdotc_sub(static_cast<int>(n), reinterpret_cast<__LAPACK_float_complex const *>(a), 1,
                    reinterpret_cast<__LAPACK_float_complex const *>(b), 1,
                    reinterpret_cast<__LAPACK_float_complex *>(&reduced_result_f32)); // conjugated
#else
    cblas_cdotc_sub(static_cast<int>(n), a, 1, b, 1, &reduced_result_f32); // conjugated
#endif
    result->real = (nk_f64_t)reduced_result_f32.real, result->imag = (nk_f64_t)reduced_result_f32.imag;
}

void dot_f64c_with_blas(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result) {
#if NK_COMPARE_TO_ACCELERATE
    cblas_zdotu_sub(static_cast<int>(n), reinterpret_cast<__LAPACK_double_complex const *>(a), 1,
                    reinterpret_cast<__LAPACK_double_complex const *>(b), 1,
                    reinterpret_cast<__LAPACK_double_complex *>(result));
#else
    cblas_zdotu_sub(static_cast<int>(n), a, 1, b, 1, result);
#endif
}

void vdot_f64c_with_blas(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result) {
#if NK_COMPARE_TO_ACCELERATE
    cblas_zdotc_sub(static_cast<int>(n), reinterpret_cast<__LAPACK_double_complex const *>(a), 1,
                    reinterpret_cast<__LAPACK_double_complex const *>(b), 1,
                    reinterpret_cast<__LAPACK_double_complex *>(result)); // conjugated
#else
    cblas_zdotc_sub(static_cast<int>(n), a, 1, b, 1, result); // conjugated
#endif
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
#if NK_COMPARE_TO_ACCELERATE
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
#if NK_COMPARE_TO_ACCELERATE
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

void dots_symmetric_f32_with_blas(nk_f32_t const *a, nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_f64_t *c,
                                  nk_size_t c_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_unused_(row_start);
    nk_unused_(row_count);
    nk_size_t leading_dimension_c = c_stride / sizeof(nk_f64_t);
    // Reuse the first half of the f64 output buffer as a packed f32 staging matrix, zero it for ssyrk, then widen in
    // place backwards.
    nk_f32_t *reduced_result_f32 = reinterpret_cast<nk_f32_t *>(c);
    std::fill_n(reduced_result_f32, n * leading_dimension_c, 0.0f);
    cblas_ssyrk(CblasRowMajor, CblasUpper, CblasNoTrans, static_cast<int>(n), static_cast<int>(k), 1.0f, a,
                static_cast<int>(a_stride / sizeof(nk_f32_t)), 0.0f, reduced_result_f32,
                static_cast<int>(leading_dimension_c));
    for (std::size_t row = n; row-- > 0;)
        for (std::size_t column = leading_dimension_c; column-- > 0;)
            c[row * leading_dimension_c + column] = (nk_f64_t)reduced_result_f32[row * leading_dimension_c + column];
}

void dots_symmetric_f64_with_blas(nk_f64_t const *a, nk_size_t n, nk_size_t k, nk_size_t a_stride, nk_f64_t *c,
                                  nk_size_t c_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_unused_(row_start);
    nk_unused_(row_count);
    cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans, static_cast<int>(n), static_cast<int>(k), 1.0, a,
                static_cast<int>(a_stride / sizeof(nk_f64_t)), 0.0, c, static_cast<int>(c_stride / sizeof(nk_f64_t)));
}

#endif // NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE

#if NK_COMPARE_TO_MKL
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

#endif // NK_COMPARE_TO_MKL

/**
 *  @brief Single dot product test for BLAS.
 */
template <typename scalar_type_>
error_stats_t test_dot_blas(typename scalar_type_::dot_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using result_t = typename scalar_t::dot_result_t;
    using reference_t = reference_for<scalar_t, result_t>;

    error_stats_t stats(comparison_family_t::approximate_k);
    std::mt19937 generator(global_config.seed);
    auto a = make_vector<scalar_t>(global_config.dense_dimensions),
         b = make_vector<scalar_t>(global_config.dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        result_t result;
        kernel(a.raw_values_data(), b.raw_values_data(), global_config.dense_dimensions, &result.raw_);

        reference_t reference;
        nk::dot<scalar_t, reference_t, nk::no_simd_k>(a.values_data(), b.values_data(), global_config.dense_dimensions,
                                                      &reference);

        stats.accumulate(result, reference);
    }
    return stats;
}

/**
 *  @brief Conjugate dot product test for BLAS (vdot = conj(a) * b).
 */
template <typename scalar_type_>
error_stats_t test_vdot_blas(typename scalar_type_::vdot_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using result_t = typename scalar_t::vdot_result_t;
    using reference_t = reference_for<scalar_t, result_t>;

    error_stats_t stats(comparison_family_t::approximate_k);
    std::mt19937 generator(global_config.seed);
    auto a = make_vector<scalar_t>(global_config.dense_dimensions),
         b = make_vector<scalar_t>(global_config.dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        result_t result;
        kernel(a.raw_values_data(), b.raw_values_data(), global_config.dense_dimensions, &result.raw_);

        reference_t reference;
        nk::vdot<scalar_t, reference_t, nk::no_simd_k>(a.values_data(), b.values_data(), global_config.dense_dimensions,
                                                       &reference);

        stats.accumulate(result, reference);
    }
    return stats;
}

void test_cross_blas() {
#if NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE
    error_stats_section_t check;
    check.section("Cross External Baselines", nk_cap_serial_k);

    // Single-vector dot product BLAS precision comparison
    check("dot_with_blas_f32", test_dot_blas<f32_t>, dot_f32_with_blas);
    check("dot_with_blas_f64", test_dot_blas<f64_t>, dot_f64_with_blas);
    check("dot_with_blas_f32c", test_dot_blas<f32c_t>, dot_f32c_with_blas);
    check("vdot_with_blas_f32c", test_vdot_blas<f32c_t>, vdot_f32c_with_blas);
    check("dot_with_blas_f64c", test_dot_blas<f64c_t>, dot_f64c_with_blas);
    check("vdot_with_blas_f64c", test_vdot_blas<f64c_t>, vdot_f64c_with_blas);

    // BLAS/MKL/Accelerate GEMM precision comparison
    check("dots_with_blas_f32", test_dots_unpacked<f32_t, f64_t, decltype(&dots_f32_with_blas)>, dots_f32_with_blas);
    check("dots_with_blas_f64", test_dots_unpacked<f64_t, f64_t, decltype(&dots_f64_with_blas)>, dots_f64_with_blas);
    check("dots_with_blas_f32c", test_dots_unpacked_conjugated<f32c_t, f32c_t, decltype(&dots_f32c_with_blas)>,
          dots_f32c_with_blas);
    check("dots_with_blas_f64c", test_dots_unpacked_conjugated<f64c_t, f64c_t, decltype(&dots_f64c_with_blas)>,
          dots_f64c_with_blas);

    // BLAS SYRK precision comparison (symmetric A x A^T)
    check("dots_symmetric_with_blas_f32", test_dots_symmetric<f32_t>, dots_symmetric_f32_with_blas);
    check("dots_symmetric_with_blas_f64", test_dots_symmetric<f64_t>, dots_symmetric_f64_with_blas);
#endif

#if NK_COMPARE_TO_MKL
    // MKL-specific GEMM with widening accumulation
    check("dots_with_mkl_bf16", test_dots_unpacked<bf16_t, f32_t, decltype(&dots_bf16_with_mkl)>, dots_bf16_with_mkl);
    check("dots_with_mkl_f16", test_dots_unpacked<f16_t, f32_t, decltype(&dots_f16_with_mkl)>, dots_f16_with_mkl);
    check("dots_with_mkl_i16", test_dots_unpacked<i16_t, i32_t, decltype(&dots_i16_with_mkl)>, dots_i16_with_mkl);
#endif
}
