/**
 *  @file bench/cross_blas.cpp
 *  @author Ash Vardanian
 *  @date January 14, 2025
 *  @brief Batch operation benchmarks, BLAS/MKL comparisons.
 */

#include <cstring> // `std::memcpy`

#include <vector> // `std::vector`

#include "harness.hpp"

namespace ashvardanian::numkong::bench {

#if NUMKONG_COMPARE_TO_BLAS || NUMKONG_COMPARE_TO_MKL || NUMKONG_COMPARE_TO_ACCELERATE

template <typename input_type_, typename input_b_type_ = input_type_, typename output_type_ = input_type_,
          typename kernel_type_>
void measure_dots_unpacked(loop_t &loop, environment_t const &env, std::size_t m, std::size_t n, std::size_t k,
                           kernel_type_ kernel) {

    bytes_t const per_set {m * k * sizeof(input_type_) + n * k * sizeof(input_b_type_) + m * n * sizeof(output_type_)};
    std::size_t const sets_count = input_sets_count(per_set);

    struct gemm_set_t {
        std::vector<input_type_> a;
        std::vector<input_b_type_> b;
        std::vector<output_type_> c;
    };
    std::vector<gemm_set_t> sets(sets_count);
    std::mt19937 generator(env.settings.seed.value);
    for (auto &s : sets) {
        s.a.resize(m * k);
        s.b.resize(n * k);
        s.c.resize(m * n);
        nk::fill_uniform(generator, s.a.data(), s.a.size());
        nk::fill_uniform(generator, s.b.data(), s.b.size());
    }

    for (std::size_t call : loop) {
        auto &s = sets[call & (sets_count - 1)];
        do_not_optimize(s.c.data());
        kernel(s.a.data(), s.b.data(), s.c.data(), m, n, k);
    }
    loop.rate("scalar-ops", 2.0 * m * n * k);
}

void measure_dots_f32_with_blas(loop_t &loop, environment_t const &env, std::size_t m, std::size_t n, std::size_t k) {
    measure_dots_unpacked<float>(loop, env, m, n, k,
                                 [](float *a, float *b, float *c, std::size_t m, std::size_t n, std::size_t k) {
                                     cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(m),
                                                 static_cast<int>(n), static_cast<int>(k), 1.0f, a, static_cast<int>(k),
                                                 b, static_cast<int>(k), 0.0f, c, static_cast<int>(n));
                                 });
}

void measure_dots_f64_with_blas(loop_t &loop, environment_t const &env, std::size_t m, std::size_t n, std::size_t k) {
    measure_dots_unpacked<double>(loop, env, m, n, k,
                                  [](double *a, double *b, double *c, std::size_t m, std::size_t n, std::size_t k) {
                                      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(m),
                                                  static_cast<int>(n), static_cast<int>(k), 1.0, a, static_cast<int>(k),
                                                  b, static_cast<int>(k), 0.0, c, static_cast<int>(n));
                                  });
}

template <typename input_type_, typename output_type_ = input_type_, typename kernel_type_>
void measure_dots_symmetric_unpacked(loop_t &loop, environment_t const &env, std::size_t n, std::size_t k,
                                     kernel_type_ kernel) {
    bytes_t const per_set {n * k * sizeof(input_type_) + n * n * sizeof(output_type_)};
    std::size_t const sets_count = input_sets_count(per_set);

    struct syrk_set_t {
        std::vector<input_type_> a;
        std::vector<output_type_> c;
    };
    std::vector<syrk_set_t> sets(sets_count);
    std::mt19937 generator(env.settings.seed.value);
    for (auto &s : sets) {
        s.a.resize(n * k);
        s.c.resize(n * n);
        nk::fill_uniform(generator, s.a.data(), s.a.size());
    }

    for (std::size_t call : loop) {
        auto &s = sets[call & (sets_count - 1)];
        do_not_optimize(s.c.data());
        kernel(s.a.data(), s.c.data(), n, k);
    }
    loop.rate("scalar-ops", n * (n + 1) * k);
}

void measure_dots_symmetric_f32_with_blas(loop_t &loop, environment_t const &env, std::size_t n, std::size_t k) {
    measure_dots_symmetric_unpacked<float>(loop, env, n, k, [](float *a, float *c, std::size_t n, std::size_t k) {
        cblas_ssyrk(CblasRowMajor, CblasUpper, CblasNoTrans, static_cast<int>(n), static_cast<int>(k), 1.0f, a,
                    static_cast<int>(k), 0.0f, c, static_cast<int>(n));
    });
}

void measure_dots_symmetric_f64_with_blas(loop_t &loop, environment_t const &env, std::size_t n, std::size_t k) {
    measure_dots_symmetric_unpacked<double>(loop, env, n, k, [](double *a, double *c, std::size_t n, std::size_t k) {
        cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans, static_cast<int>(n), static_cast<int>(k), 1.0, a,
                    static_cast<int>(k), 0.0, c, static_cast<int>(n));
    });
}

#endif // NUMKONG_COMPARE_TO_BLAS || NUMKONG_COMPARE_TO_MKL || NUMKONG_COMPARE_TO_ACCELERATE

#if NUMKONG_COMPARE_TO_ACCELERATE

static BNNSNDArrayDescriptor bnns_matrix_desc(BNNSDataType dtype, void *data, std::size_t rows, std::size_t cols) {
    BNNSNDArrayDescriptor desc = {};
    desc.layout = BNNSDataLayout2DFirstMajor;
    desc.size[0] = rows;
    desc.size[1] = cols;
    desc.data_type = dtype;
    desc.data = data;
    return desc;
}

void measure_dots_f16_with_accelerate(loop_t &loop, environment_t const &env, std::size_t m, std::size_t n,
                                      std::size_t k) {
    measure_dots_unpacked<nk_f16_t, nk_f16_t, float>(
        loop, env, m, n, k, [](nk_f16_t *a, nk_f16_t *b, float *c, std::size_t m, std::size_t n, std::size_t k) {
            auto a_desc = bnns_matrix_desc(BNNSDataTypeFloat16, a, m, k);
            auto b_desc = bnns_matrix_desc(BNNSDataTypeFloat16, b, n, k);
            auto c_desc = bnns_matrix_desc(BNNSDataTypeFloat32, c, m, n);
            BNNSMatMul(false, true, 1.0f, &a_desc, &b_desc, &c_desc, nullptr, nullptr);
        });
}

void measure_dots_bf16_with_accelerate(loop_t &loop, environment_t const &env, std::size_t m, std::size_t n,
                                       std::size_t k) {
    measure_dots_unpacked<nk_bf16_t, nk_bf16_t, float>(
        loop, env, m, n, k, [](nk_bf16_t *a, nk_bf16_t *b, float *c, std::size_t m, std::size_t n, std::size_t k) {
            auto a_desc = bnns_matrix_desc(BNNSDataTypeBFloat16, a, m, k);
            auto b_desc = bnns_matrix_desc(BNNSDataTypeBFloat16, b, n, k);
            auto c_desc = bnns_matrix_desc(BNNSDataTypeFloat32, c, m, n);
            BNNSMatMul(false, true, 1.0f, &a_desc, &b_desc, &c_desc, nullptr, nullptr);
        });
}

#endif // NUMKONG_COMPARE_TO_ACCELERATE

#if NUMKONG_COMPARE_TO_MKL

void measure_dots_f32_with_mkl(loop_t &loop, environment_t const &env, std::size_t m, std::size_t n, std::size_t k) {
    measure_dots_unpacked<float>(loop, env, m, n, k,
                                 [](float *a, float *b, float *c, std::size_t m, std::size_t n, std::size_t k) {
                                     cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (MKL_INT)m, (MKL_INT)n,
                                                 (MKL_INT)k, 1.0f, a, (MKL_INT)k, b, (MKL_INT)k, 0.0f, c, (MKL_INT)n);
                                 });
}

void measure_dots_bf16_with_mkl(loop_t &loop, environment_t const &env, std::size_t m, std::size_t n, std::size_t k) {
    measure_dots_unpacked<MKL_BF16, MKL_BF16, float>(
        loop, env, m, n, k, [](MKL_BF16 *a, MKL_BF16 *b, float *c, std::size_t m, std::size_t n, std::size_t k) {
            cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasTrans, (MKL_INT)m, (MKL_INT)n, (MKL_INT)k, 1.0f, a,
                                   (MKL_INT)k, b, (MKL_INT)k, 0.0f, c, (MKL_INT)n);
        });
}

void measure_dots_f16_with_mkl(loop_t &loop, environment_t const &env, std::size_t m, std::size_t n, std::size_t k) {
    measure_dots_unpacked<MKL_F16, MKL_F16, float>(
        loop, env, m, n, k, [](MKL_F16 *a, MKL_F16 *b, float *c, std::size_t m, std::size_t n, std::size_t k) {
            cblas_gemm_f16f16f32(CblasRowMajor, CblasNoTrans, CblasTrans, (MKL_INT)m, (MKL_INT)n, (MKL_INT)k, 1.0f, a,
                                 (MKL_INT)k, b, (MKL_INT)k, 0.0f, c, (MKL_INT)n);
        });
}

void measure_dots_f64_with_mkl(loop_t &loop, environment_t const &env, std::size_t m, std::size_t n, std::size_t k) {
    measure_dots_unpacked<double>(loop, env, m, n, k,
                                  [](double *a, double *b, double *c, std::size_t m, std::size_t n, std::size_t k) {
                                      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (MKL_INT)m, (MKL_INT)n,
                                                  (MKL_INT)k, 1.0, a, (MKL_INT)k, b, (MKL_INT)k, 0.0, c, (MKL_INT)n);
                                  });
}

void measure_dots_i8u8_with_mkl(loop_t &loop, environment_t const &env, std::size_t m, std::size_t n, std::size_t k) {
    measure_dots_unpacked<std::uint8_t, std::int8_t, std::int32_t>(
        loop, env, m, n, k,
        [](std::uint8_t *a, std::int8_t *b, std::int32_t *c, std::size_t m, std::size_t n, std::size_t k) {
            MKL_INT32 c_offset = 0;
            cblas_gemm_s8u8s32(CblasRowMajor, CblasNoTrans, CblasTrans, CblasFixOffset, (MKL_INT)m, (MKL_INT)n,
                               (MKL_INT)k, 1.0f, a, (MKL_INT)k, 0, b, (MKL_INT)k, 0, 0.0f, c, (MKL_INT)n, &c_offset);
        });
}

void measure_dots_i16_with_mkl(loop_t &loop, environment_t const &env, std::size_t m, std::size_t n, std::size_t k) {
    measure_dots_unpacked<std::int16_t, std::int16_t, std::int32_t>(
        loop, env, m, n, k,
        [](std::int16_t *a, std::int16_t *b, std::int32_t *c, std::size_t m, std::size_t n, std::size_t k) {
            MKL_INT32 c_offset = 0;
            cblas_gemm_s16s16s32(CblasRowMajor, CblasNoTrans, CblasTrans, CblasFixOffset, (MKL_INT)m, (MKL_INT)n,
                                 (MKL_INT)k, 1.0f, a, (MKL_INT)k, 0, b, (MKL_INT)k, 0, 0.0f, c, (MKL_INT)n, &c_offset);
        });
}

#endif // NUMKONG_COMPARE_TO_MKL

void bench_cross_blas(environment_t const &env) {

    std::string syrk_dims = std::to_string(env.settings.matrix_height) + "x" +
                            std::to_string(env.settings.matrix_depth);
    std::string gemm_dims = std::to_string(env.settings.matrix_height) + "x" +
                            std::to_string(env.settings.matrix_width) + "x" + std::to_string(env.settings.matrix_depth);

    nk_unused_(syrk_dims);
    nk_unused_(gemm_dims);

#if NUMKONG_COMPARE_TO_BLAS || NUMKONG_COMPARE_TO_MKL || NUMKONG_COMPARE_TO_ACCELERATE
    section(env, "Cross External Baselines", nk_cap_serial_k);
    // BLAS GEMM baselines for matmul comparison (same layout as NumKong: A x B^T)
    run_benchmark(env, "dots_packed_f32_with_blas<" + gemm_dims + ">", measure_dots_f32_with_blas,
                  env.settings.matrix_height, env.settings.matrix_width, env.settings.matrix_depth);
    run_benchmark(env, "dots_packed_f64_with_blas<" + gemm_dims + ">", measure_dots_f64_with_blas,
                  env.settings.matrix_height, env.settings.matrix_width, env.settings.matrix_depth);

    // BLAS SYRK baselines for symmetric operations (correct operation for dots_symmetric: A x A^T)
    run_benchmark(env, "dots_symmetric_f32_with_blas<" + syrk_dims + ">", measure_dots_symmetric_f32_with_blas,
                  env.settings.matrix_height, env.settings.matrix_depth);
    run_benchmark(env, "dots_symmetric_f64_with_blas<" + syrk_dims + ">", measure_dots_symmetric_f64_with_blas,
                  env.settings.matrix_height, env.settings.matrix_depth);
#endif

#if NUMKONG_COMPARE_TO_ACCELERATE
    run_benchmark(env, "dots_packed_f16_with_accelerate<" + gemm_dims + ">", measure_dots_f16_with_accelerate,
                  env.settings.matrix_height, env.settings.matrix_width, env.settings.matrix_depth);
    run_benchmark(env, "dots_packed_bf16_with_accelerate<" + gemm_dims + ">", measure_dots_bf16_with_accelerate,
                  env.settings.matrix_height, env.settings.matrix_width, env.settings.matrix_depth);
#endif

#if NUMKONG_COMPARE_TO_MKL
    run_benchmark(env, "dots_packed_f32_with_mkl<" + gemm_dims + ">", measure_dots_f32_with_mkl,
                  env.settings.matrix_height, env.settings.matrix_width, env.settings.matrix_depth);
    run_benchmark(env, "dots_packed_bf16_with_mkl<" + gemm_dims + ">", measure_dots_bf16_with_mkl,
                  env.settings.matrix_height, env.settings.matrix_width, env.settings.matrix_depth);
    run_benchmark(env, "dots_packed_f16_with_mkl<" + gemm_dims + ">", measure_dots_f16_with_mkl,
                  env.settings.matrix_height, env.settings.matrix_width, env.settings.matrix_depth);
    run_benchmark(env, "dots_packed_f64_with_mkl<" + gemm_dims + ">", measure_dots_f64_with_mkl,
                  env.settings.matrix_height, env.settings.matrix_width, env.settings.matrix_depth);
    run_benchmark(env, "dots_packed_i8u8_with_mkl<" + gemm_dims + ">", measure_dots_i8u8_with_mkl,
                  env.settings.matrix_height, env.settings.matrix_width, env.settings.matrix_depth);
    run_benchmark(env, "dots_packed_i16_with_mkl<" + gemm_dims + ">", measure_dots_i16_with_mkl,
                  env.settings.matrix_height, env.settings.matrix_width, env.settings.matrix_depth);
#endif
}

} // namespace ashvardanian::numkong::bench
