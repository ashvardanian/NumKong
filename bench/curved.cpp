/**
 *  @file bench/curved.cpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 *  @brief Bilinear and Mahalanobis distance benchmarks.
 */

#include <complex> // std::complex
#include <vector>  // std::vector

#include "numkong/curved.h"

#include "harness.hpp"

using namespace ashvardanian::numkong::bench;

#if NUMKONG_COMPARE_TO_BLAS || NUMKONG_COMPARE_TO_MKL || NUMKONG_COMPARE_TO_ACCELERATE

void bilinear_f32_with_blas(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n, nk_f64_t *result) {
    static thread_local std::vector<nk_f32_t> intermediate;
    if (intermediate.size() < n) intermediate.resize(n);
    int const ni = static_cast<int>(n);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, ni, ni, 1.0f, c, ni, b, 1, 0.0f, intermediate.data(), 1);
    *result = cblas_dsdot(ni, a, 1, intermediate.data(), 1);
}

void bilinear_f64_with_blas(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n, nk_f64_t *result) {
    static thread_local std::vector<nk_f64_t> intermediate;
    if (intermediate.size() < n) intermediate.resize(n);
    int const ni = static_cast<int>(n);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, ni, ni, 1.0, c, ni, b, 1, 0.0, intermediate.data(), 1);
    *result = cblas_ddot(ni, a, 1, intermediate.data(), 1);
}

void bilinear_f32c_with_blas(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c, nk_size_t n,
                             nk_f64c_t *results) {
    static thread_local std::vector<nk_f32c_t> intermediate;
    nk_f32c_t reduced_result_f32;
    if (intermediate.size() < n) intermediate.resize(n);
    int const ni = static_cast<int>(n);
#if NUMKONG_COMPARE_TO_ACCELERATE
    std::complex<float> alpha = {1.0f, 0.0f}, beta = {0.0f, 0.0f};
    cblas_cgemv(CblasRowMajor, CblasNoTrans, ni, ni, &alpha, reinterpret_cast<std::complex<float> const *>(c), ni,
                reinterpret_cast<std::complex<float> const *>(b), 1, &beta,
                reinterpret_cast<std::complex<float> *>(intermediate.data()), 1);
    cblas_cdotu_sub(ni, reinterpret_cast<std::complex<float> const *>(a), 1,
                    reinterpret_cast<std::complex<float> const *>(intermediate.data()), 1,
                    reinterpret_cast<std::complex<float> *>(&reduced_result_f32));
#else
    nk_f32c_t alpha = {1.0f, 0.0f}, beta = {0.0f, 0.0f};
    cblas_cgemv(CblasRowMajor, CblasNoTrans, ni, ni, &alpha, c, ni, b, 1, &beta, intermediate.data(), 1);
    cblas_cdotu_sub(ni, reinterpret_cast<nk_f32_t const *>(a), 1,
                    reinterpret_cast<nk_f32_t const *>(intermediate.data()), 1,
                    reinterpret_cast<nk_f32_t *>(&reduced_result_f32));
#endif
    results->real = (nk_f64_t)reduced_result_f32.real;
    results->imag = (nk_f64_t)reduced_result_f32.imag;
}

void bilinear_f64c_with_blas(nk_f64c_t const *a, nk_f64c_t const *b, nk_f64c_t const *c, nk_size_t n,
                             nk_f64c_t *results) {
    static thread_local std::vector<nk_f64c_t> intermediate;
    if (intermediate.size() < n) intermediate.resize(n);
    int const ni = static_cast<int>(n);
#if NUMKONG_COMPARE_TO_ACCELERATE
    std::complex<double> alpha = {1.0, 0.0}, beta = {0.0, 0.0};
    cblas_zgemv(CblasRowMajor, CblasNoTrans, ni, ni, &alpha, reinterpret_cast<std::complex<double> const *>(c), ni,
                reinterpret_cast<std::complex<double> const *>(b), 1, &beta,
                reinterpret_cast<std::complex<double> *>(intermediate.data()), 1);
    cblas_zdotu_sub(ni, reinterpret_cast<std::complex<double> const *>(a), 1,
                    reinterpret_cast<std::complex<double> const *>(intermediate.data()), 1,
                    reinterpret_cast<std::complex<double> *>(results));
#else
    nk_f64c_t alpha = {1.0, 0.0}, beta = {0.0, 0.0};
    cblas_zgemv(CblasRowMajor, CblasNoTrans, ni, ni, &alpha, c, ni, b, 1, &beta, intermediate.data(), 1);
    cblas_zdotu_sub(ni, reinterpret_cast<nk_f64_t const *>(a), 1,
                    reinterpret_cast<nk_f64_t const *>(intermediate.data()), 1, reinterpret_cast<nk_f64_t *>(results));
#endif
}

#endif // NUMKONG_COMPARE_TO_BLAS || NUMKONG_COMPARE_TO_MKL || NUMKONG_COMPARE_TO_ACCELERATE

/**
 *  @brief Measures a @b curved kernel, bilinear or Mahalanobis, using Google Benchmark.
 *  @param[inout] state The benchmark state object provided by Google Benchmark.
 *  @param[in] kernel The kernel function to benchmark.
 *  @param[in] dimensions The number of dimensions in the vectors.
 */
template <nk_dtype_t input_dtype_, typename kernel_type_ = void>
void measure_curved(bm::State &state, kernel_type_ kernel, std::size_t dimensions) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using output_t = typename input_t::curved_result_t;
    using input_vector_t = nk::vector<input_t>;

    // Preallocate inputs: pairs of vectors + metric tensors (dimensions x dimensions)
    std::size_t bytes_per_set = bench_dtype_bytes(input_dtype_, 2 * dimensions + dimensions * dimensions);
    std::size_t const vectors_count = bench_input_count(bytes_per_set);
    std::vector<input_vector_t> first_vectors(vectors_count), second_vectors(vectors_count), tensors(vectors_count);
    auto generator = make_random_engine();
    for (std::size_t index = 0; index != vectors_count; ++index) {
        first_vectors[index] = make_vector<input_t>(dimensions);
        second_vectors[index] = make_vector<input_t>(dimensions);
        tensors[index] = make_vector<input_t>(dimensions * dimensions);
        nk::fill_uniform(generator, first_vectors[index].values_data(), first_vectors[index].size_values());
        nk::fill_uniform(generator, second_vectors[index].values_data(), second_vectors[index].size_values());
        nk::fill_uniform(generator, tensors[index].values_data(), tensors[index].size_values());
    }

    // Benchmark loop
    std::size_t iterations = 0;
    for (auto _ : state) {
        output_t output[2] = {};
        std::size_t const index = iterations & (vectors_count - 1);
        kernel(first_vectors[index].raw_values_data(), second_vectors[index].raw_values_data(),
               tensors[index].raw_values_data(), dimensions, &output[0].raw_);
        bm::DoNotOptimize(output);
        iterations++;
    }

    state.counters["bytes"] = bm::Counter(2.0 * iterations * first_vectors[0].size_bytes(), bm::Counter::kIsRate);
    state.counters["calls"] = bm::Counter(iterations, bm::Counter::kIsRate);
    state.counters["scalar-ops"] = bm::Counter(2.0 * iterations * (dimensions * dimensions + dimensions),
                                               bm::Counter::kIsRate);
}

template <nk_dtype_t input_dtype_, typename kernel_type_ = void>
void run_curved(std::string name, kernel_type_ *kernel) {
    std::string bench_name = name + "<" + std::to_string(bench_config.curved_dimensions) + "d>";
    bm::RegisterBenchmark(bench_name.c_str(), measure_curved<input_dtype_, kernel_type_ *>, kernel,
                          bench_config.curved_dimensions);
}

void bench_curved() {
    constexpr nk_dtype_t f64_k = nk_f64_k;
    constexpr nk_dtype_t f32_k = nk_f32_k;
    constexpr nk_dtype_t f16_k = nk_f16_k;
    constexpr nk_dtype_t bf16_k = nk_bf16_k;
    constexpr nk_dtype_t f64c_k = nk_f64c_k;
    constexpr nk_dtype_t f32c_k = nk_f32c_k;
    constexpr nk_dtype_t f16c_k = nk_f16c_k;
    constexpr nk_dtype_t bf16c_k = nk_bf16c_k;

#if NUMKONG_COMPARE_TO_BLAS || NUMKONG_COMPARE_TO_MKL || NUMKONG_COMPARE_TO_ACCELERATE
    run_curved<f64_k>("bilinear_f64_with_blas", bilinear_f64_with_blas);
    run_curved<f64c_k>("bilinear_f64c_with_blas", bilinear_f64c_with_blas);
    run_curved<f32_k>("bilinear_f32_with_blas", bilinear_f32_with_blas);
    run_curved<f32c_k>("bilinear_f32c_with_blas", bilinear_f32c_with_blas);
#endif

#if NUMKONG_TARGET_NEON
    run_curved<f32_k>("bilinear_f32_neon", nk_bilinear_f32_neon);
    run_curved<f32_k>("mahalanobis_f32_neon", nk_mahalanobis_f32_neon);
    run_curved<f32c_k>("bilinear_f32c_neon", nk_bilinear_f32c_neon);
    run_curved<f16_k>("bilinear_f16_neon", nk_bilinear_f16_neon);
    run_curved<f16_k>("mahalanobis_f16_neon", nk_mahalanobis_f16_neon);
    run_curved<f16c_k>("bilinear_f16c_neon", nk_bilinear_f16c_neon);
#endif

#if NUMKONG_TARGET_NEONBFDOT
    run_curved<bf16_k>("bilinear_bf16_neonbfdot", nk_bilinear_bf16_neonbfdot);
    run_curved<bf16_k>("mahalanobis_bf16_neonbfdot", nk_mahalanobis_bf16_neonbfdot);
    run_curved<bf16c_k>("bilinear_bf16c_neonbfdot", nk_bilinear_bf16c_neonbfdot);
#endif

#if NUMKONG_TARGET_SMEF64
    run_curved<f32_k>("bilinear_f32_smef64", nk_bilinear_f32_smef64);
    run_curved<f32c_k>("bilinear_f32c_smef64", nk_bilinear_f32c_smef64);
    run_curved<f32_k>("mahalanobis_f32_smef64", nk_mahalanobis_f32_smef64);
    run_curved<f64_k>("bilinear_f64_smef64", nk_bilinear_f64_smef64);
    run_curved<f64c_k>("bilinear_f64c_smef64", nk_bilinear_f64c_smef64);
    run_curved<f64_k>("mahalanobis_f64_smef64", nk_mahalanobis_f64_smef64);
#endif

#if NUMKONG_TARGET_HASWELL
    run_curved<f32_k>("bilinear_f32_haswell", nk_bilinear_f32_haswell);
    run_curved<f32_k>("mahalanobis_f32_haswell", nk_mahalanobis_f32_haswell);
    run_curved<f16_k>("bilinear_f16_haswell", nk_bilinear_f16_haswell);
    run_curved<f16_k>("mahalanobis_f16_haswell", nk_mahalanobis_f16_haswell);
    run_curved<bf16_k>("bilinear_bf16_haswell", nk_bilinear_bf16_haswell);
    run_curved<bf16_k>("mahalanobis_bf16_haswell", nk_mahalanobis_bf16_haswell);
#endif

#if NUMKONG_TARGET_SKYLAKE
    run_curved<f32_k>("bilinear_f32_skylake", nk_bilinear_f32_skylake);
    run_curved<f32c_k>("bilinear_f32c_skylake", nk_bilinear_f32c_skylake);
    run_curved<f64_k>("bilinear_f64_skylake", nk_bilinear_f64_skylake);
    run_curved<f64c_k>("bilinear_f64c_skylake", nk_bilinear_f64c_skylake);
    run_curved<f32_k>("mahalanobis_f32_skylake", nk_mahalanobis_f32_skylake);
    run_curved<f64_k>("mahalanobis_f64_skylake", nk_mahalanobis_f64_skylake);
#endif

#if NUMKONG_TARGET_GENOA
    run_curved<bf16_k>("bilinear_bf16_genoa", nk_bilinear_bf16_genoa);
    run_curved<bf16_k>("mahalanobis_bf16_genoa", nk_mahalanobis_bf16_genoa);
    run_curved<bf16c_k>("bilinear_bf16c_genoa", nk_bilinear_bf16c_genoa);
#endif

    // Serial fallbacks
    run_curved<f64_k>("bilinear_f64_serial", nk_bilinear_f64_serial);
    run_curved<f64c_k>("bilinear_f64c_serial", nk_bilinear_f64c_serial);
    run_curved<f64_k>("mahalanobis_f64_serial", nk_mahalanobis_f64_serial);
    run_curved<f32_k>("bilinear_f32_serial", nk_bilinear_f32_serial);
    run_curved<f32c_k>("bilinear_f32c_serial", nk_bilinear_f32c_serial);
    run_curved<f32_k>("mahalanobis_f32_serial", nk_mahalanobis_f32_serial);
    run_curved<f16_k>("bilinear_f16_serial", nk_bilinear_f16_serial);
    run_curved<f16c_k>("bilinear_f16c_serial", nk_bilinear_f16c_serial);
    run_curved<f16_k>("mahalanobis_f16_serial", nk_mahalanobis_f16_serial);
    run_curved<bf16_k>("bilinear_bf16_serial", nk_bilinear_bf16_serial);
    run_curved<bf16c_k>("bilinear_bf16c_serial", nk_bilinear_bf16c_serial);
    run_curved<bf16_k>("mahalanobis_bf16_serial", nk_mahalanobis_bf16_serial);
}
