/**
 *  @brief Dot product benchmarks.
 *  @file bench/bench_dot.cpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 */

#include <complex> // std::complex

#include "numkong/dot.h"

#include "bench.hpp"

using namespace ashvardanian::numkong::bench;

#if NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE

void dot_f32_with_blas(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result) {
    *result = cblas_dsdot(static_cast<int>(n), a, 1, b, 1);
}

void dot_f64_with_blas(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    *result = cblas_ddot(static_cast<int>(n), a, 1, b, 1);
}

void dot_f32c_with_blas(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f64c_t *result) {
    nk_f32c_t reduced_result_f32;
#if NK_COMPARE_TO_ACCELERATE
    cblas_cdotu_sub(static_cast<int>(n), reinterpret_cast<std::complex<float> const *>(a), 1,
                    reinterpret_cast<std::complex<float> const *>(b), 1,
                    reinterpret_cast<std::complex<float> *>(&reduced_result_f32));
#else
    cblas_cdotu_sub(static_cast<int>(n), reinterpret_cast<nk_f32_t const *>(a), 1,
                    reinterpret_cast<nk_f32_t const *>(b), 1, reinterpret_cast<nk_f32_t *>(&reduced_result_f32));
#endif
    result->real = (nk_f64_t)reduced_result_f32.real;
    result->imag = (nk_f64_t)reduced_result_f32.imag;
}

void dot_f64c_with_blas(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result) {
#if NK_COMPARE_TO_ACCELERATE
    cblas_zdotu_sub(static_cast<int>(n), reinterpret_cast<std::complex<double> const *>(a), 1,
                    reinterpret_cast<std::complex<double> const *>(b), 1,
                    reinterpret_cast<std::complex<double> *>(result));
#else
    cblas_zdotu_sub(static_cast<int>(n), reinterpret_cast<nk_f64_t const *>(a), 1,
                    reinterpret_cast<nk_f64_t const *>(b), 1, reinterpret_cast<nk_f64_t *>(result));
#endif
}

void vdot_f32c_with_blas(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f64c_t *result) {
    nk_f32c_t reduced_result_f32;
#if NK_COMPARE_TO_ACCELERATE
    cblas_cdotc_sub(static_cast<int>(n), reinterpret_cast<std::complex<float> const *>(a), 1,
                    reinterpret_cast<std::complex<float> const *>(b), 1,
                    reinterpret_cast<std::complex<float> *>(&reduced_result_f32));
#else
    cblas_cdotc_sub(static_cast<int>(n), reinterpret_cast<nk_f32_t const *>(a), 1,
                    reinterpret_cast<nk_f32_t const *>(b), 1, reinterpret_cast<nk_f32_t *>(&reduced_result_f32));
#endif
    result->real = (nk_f64_t)reduced_result_f32.real;
    result->imag = (nk_f64_t)reduced_result_f32.imag;
}

void vdot_f64c_with_blas(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result) {
#if NK_COMPARE_TO_ACCELERATE
    cblas_zdotc_sub(static_cast<int>(n), reinterpret_cast<std::complex<double> const *>(a), 1,
                    reinterpret_cast<std::complex<double> const *>(b), 1,
                    reinterpret_cast<std::complex<double> *>(result));
#else
    cblas_zdotc_sub(static_cast<int>(n), reinterpret_cast<nk_f64_t const *>(a), 1,
                    reinterpret_cast<nk_f64_t const *>(b), 1, reinterpret_cast<nk_f64_t *>(result));
#endif
}

#endif // NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE

void bench_dot() {
    constexpr nk_dtype_t f64_k = nk_f64_k;
    constexpr nk_dtype_t f32_k = nk_f32_k;
    constexpr nk_dtype_t bf16_k = nk_bf16_k;
    constexpr nk_dtype_t f16_k = nk_f16_k;
    constexpr nk_dtype_t e5m2_k = nk_e5m2_k;
    constexpr nk_dtype_t e4m3_k = nk_e4m3_k;
    constexpr nk_dtype_t e3m2_k = nk_e3m2_k;
    constexpr nk_dtype_t e2m3_k = nk_e2m3_k;
    constexpr nk_dtype_t e2m1_k = nk_e2m1_k;
    constexpr nk_dtype_t i32_k = nk_i32_k;
    constexpr nk_dtype_t i8_k = nk_i8_k;
    constexpr nk_dtype_t i4_k = nk_i4_k;
    constexpr nk_dtype_t u32_k = nk_u32_k;
    constexpr nk_dtype_t u8_k = nk_u8_k;
    constexpr nk_dtype_t u4_k = nk_u4_k;
    constexpr nk_dtype_t u1_k = nk_u1_k;
    constexpr nk_dtype_t f64c_k = nk_f64c_k;
    constexpr nk_dtype_t f32c_k = nk_f32c_k;
    constexpr nk_dtype_t f16c_k = nk_f16c_k;
    constexpr nk_dtype_t bf16c_k = nk_bf16c_k;

#if NK_COMPARE_TO_BLAS || NK_COMPARE_TO_MKL || NK_COMPARE_TO_ACCELERATE
    run_dense<f64c_k, f64c_k>("dot_f64c_with_blas", dot_f64c_with_blas);
    run_dense<f64c_k, f64c_k>("vdot_f64c_with_blas", vdot_f64c_with_blas);
    run_dense<f32c_k, f64c_k>("dot_f32c_with_blas", dot_f32c_with_blas);
    run_dense<f32c_k, f64c_k>("vdot_f32c_with_blas", vdot_f32c_with_blas);
    run_dense<f64_k, f64_k>("dot_f64_with_blas", dot_f64_with_blas);
    run_dense<f32_k, f64_k>("dot_f32_with_blas", dot_f32_with_blas);
#endif

#if NK_TARGET_NEON
    run_dense<f64c_k, f64c_k>("dot_f64c_neon", nk_dot_f64c_neon);
    run_dense<f64c_k, f64c_k>("vdot_f64c_neon", nk_vdot_f64c_neon);
    run_dense<f32c_k, f64c_k>("dot_f32c_neon", nk_dot_f32c_neon);
    run_dense<f32c_k, f64c_k>("vdot_f32c_neon", nk_vdot_f32c_neon);
    run_dense<f64_k, f64_k>("dot_f64_neon", nk_dot_f64_neon);
    run_dense<f32_k, f64_k>("dot_f32_neon", nk_dot_f32_neon);
    run_dense<bf16_k, f32_k>("dot_bf16_neon", nk_dot_bf16_neon);
    run_dense<f16_k, f32_k>("dot_f16_neon", nk_dot_f16_neon);
    run_dense<e5m2_k, f32_k>("dot_e5m2_neon", nk_dot_e5m2_neon);
    run_dense<e4m3_k, f32_k>("dot_e4m3_neon", nk_dot_e4m3_neon);
    run_dense<e3m2_k, f32_k>("dot_e3m2_neon", nk_dot_e3m2_neon);
    run_dense<e2m3_k, f32_k>("dot_e2m3_neon", nk_dot_e2m3_neon);
    run_dense<f16c_k, f32c_k>("dot_f16c_neon", nk_dot_f16c_neon);
    run_dense<f16c_k, f32c_k>("vdot_f16c_neon", nk_vdot_f16c_neon);
    run_dense<u1_k, u32_k>("dot_u1_neon", nk_dot_u1_neon);
#endif

#if NK_TARGET_NEONSDOT
    run_dense<e3m2_k, f32_k>("dot_e3m2_neonsdot", nk_dot_e3m2_neonsdot);
    run_dense<e2m3_k, f32_k>("dot_e2m3_neonsdot", nk_dot_e2m3_neonsdot);
    run_dense<e2m1_k, f32_k>("dot_e2m1_neonsdot", nk_dot_e2m1_neonsdot);
    run_dense<i8_k, i32_k>("dot_i8_neonsdot", nk_dot_i8_neonsdot);
    run_dense<i4_k, i32_k>("dot_i4_neonsdot", nk_dot_i4_neonsdot);
    run_dense<u8_k, u32_k>("dot_u8_neonsdot", nk_dot_u8_neonsdot);
    run_dense<u4_k, u32_k>("dot_u4_neonsdot", nk_dot_u4_neonsdot);
#endif

#if NK_TARGET_NEONFHM
    run_dense<f16c_k, f32c_k>("dot_f16c_neonfhm", nk_dot_f16c_neonfhm);
    run_dense<f16c_k, f32c_k>("vdot_f16c_neonfhm", nk_vdot_f16c_neonfhm);
    run_dense<f16_k, f32_k>("dot_f16_neonfhm", nk_dot_f16_neonfhm);
    run_dense<e5m2_k, f32_k>("dot_e5m2_neonfhm", nk_dot_e5m2_neonfhm);
    run_dense<e4m3_k, f32_k>("dot_e4m3_neonfhm", nk_dot_e4m3_neonfhm);
#endif

#if NK_TARGET_NEONBFDOT
    run_dense<bf16c_k, f32c_k>("dot_bf16c_neonbfdot", nk_dot_bf16c_neonbfdot);
    run_dense<bf16c_k, f32c_k>("vdot_bf16c_neonbfdot", nk_vdot_bf16c_neonbfdot);
    run_dense<bf16_k, f32_k>("dot_bf16_neonbfdot", nk_dot_bf16_neonbfdot);
    run_dense<e5m2_k, f32_k>("dot_e5m2_neonbfdot", nk_dot_e5m2_neonbfdot);
    run_dense<e4m3_k, f32_k>("dot_e4m3_neonbfdot", nk_dot_e4m3_neonbfdot);
#endif

#if NK_TARGET_SVE
    run_dense<f64c_k, f64c_k>("dot_f64c_sve", nk_dot_f64c_sve);
    run_dense<f64c_k, f64c_k>("vdot_f64c_sve", nk_vdot_f64c_sve);
    run_dense<f32c_k, f64c_k>("dot_f32c_sve", nk_dot_f32c_sve);
    run_dense<f32c_k, f64c_k>("vdot_f32c_sve", nk_vdot_f32c_sve);
    run_dense<f64_k, f64_k>("dot_f64_sve", nk_dot_f64_sve);
    run_dense<f32_k, f64_k>("dot_f32_sve", nk_dot_f32_sve);
#endif

#if NK_TARGET_SVEHALF
    run_dense<f16c_k, f32c_k>("dot_f16c_svehalf", nk_dot_f16c_svehalf);
    run_dense<f16c_k, f32c_k>("vdot_f16c_svehalf", nk_vdot_f16c_svehalf);
    run_dense<f16_k, f32_k>("dot_f16_svehalf", nk_dot_f16_svehalf);
#endif

#if NK_TARGET_SVEBFDOT
    run_dense<bf16_k, f32_k>("dot_bf16_svebfdot", nk_dot_bf16_svebfdot);
#endif

#if NK_TARGET_SVESDOT
    run_dense<i8_k, i32_k>("dot_i8_svesdot", nk_dot_i8_svesdot);
    run_dense<u8_k, u32_k>("dot_u8_svesdot", nk_dot_u8_svesdot);
#endif

#if NK_TARGET_NEONFP8
    run_dense<e5m2_k, f32_k>("dot_e5m2_neonfp8", nk_dot_e5m2_neonfp8);
    run_dense<e4m3_k, f32_k>("dot_e4m3_neonfp8", nk_dot_e4m3_neonfp8);
    run_dense<e3m2_k, f32_k>("dot_e3m2_neonfp8", nk_dot_e3m2_neonfp8);
    run_dense<e2m3_k, f32_k>("dot_e2m3_neonfp8", nk_dot_e2m3_neonfp8);
#endif

#if NK_TARGET_HASWELL
    run_dense<f64c_k, f64c_k>("dot_f64c_haswell", nk_dot_f64c_haswell);
    run_dense<f64c_k, f64c_k>("vdot_f64c_haswell", nk_vdot_f64c_haswell);
    run_dense<f32c_k, f64c_k>("dot_f32c_haswell", nk_dot_f32c_haswell);
    run_dense<f32c_k, f64c_k>("vdot_f32c_haswell", nk_vdot_f32c_haswell);
    run_dense<bf16c_k, f32c_k>("dot_bf16c_haswell", nk_dot_bf16c_haswell);
    run_dense<bf16c_k, f32c_k>("vdot_bf16c_haswell", nk_vdot_bf16c_haswell);
    run_dense<f16c_k, f32c_k>("dot_f16c_haswell", nk_dot_f16c_haswell);
    run_dense<f16c_k, f32c_k>("vdot_f16c_haswell", nk_vdot_f16c_haswell);
    run_dense<f64_k, f64_k>("dot_f64_haswell", nk_dot_f64_haswell);
    run_dense<f32_k, f64_k>("dot_f32_haswell", nk_dot_f32_haswell);
    run_dense<bf16_k, f32_k>("dot_bf16_haswell", nk_dot_bf16_haswell);
    run_dense<f16_k, f32_k>("dot_f16_haswell", nk_dot_f16_haswell);
    run_dense<e5m2_k, f32_k>("dot_e5m2_haswell", nk_dot_e5m2_haswell);
    run_dense<e4m3_k, f32_k>("dot_e4m3_haswell", nk_dot_e4m3_haswell);
    run_dense<e3m2_k, f32_k>("dot_e3m2_haswell", nk_dot_e3m2_haswell);
    run_dense<e2m3_k, f32_k>("dot_e2m3_haswell", nk_dot_e2m3_haswell);
    run_dense<e2m1_k, f32_k>("dot_e2m1_haswell", nk_dot_e2m1_haswell);
    run_dense<i8_k, i32_k>("dot_i8_haswell", nk_dot_i8_haswell);
    run_dense<i4_k, i32_k>("dot_i4_haswell", nk_dot_i4_haswell);
    run_dense<u8_k, u32_k>("dot_u8_haswell", nk_dot_u8_haswell);
    run_dense<u4_k, u32_k>("dot_u4_haswell", nk_dot_u4_haswell);
    run_dense<u1_k, u32_k>("dot_u1_haswell", nk_dot_u1_haswell);
#endif

#if NK_TARGET_SKYLAKE
    run_dense<f64c_k, f64c_k>("dot_f64c_skylake", nk_dot_f64c_skylake);
    run_dense<f64c_k, f64c_k>("vdot_f64c_skylake", nk_vdot_f64c_skylake);
    run_dense<f32c_k, f64c_k>("dot_f32c_skylake", nk_dot_f32c_skylake);
    run_dense<f32c_k, f64c_k>("vdot_f32c_skylake", nk_vdot_f32c_skylake);
    run_dense<f64_k, f64_k>("dot_f64_skylake", nk_dot_f64_skylake);
    run_dense<f32_k, f64_k>("dot_f32_skylake", nk_dot_f32_skylake);
    run_dense<bf16_k, f32_k>("dot_bf16_skylake", nk_dot_bf16_skylake);
    run_dense<f16_k, f32_k>("dot_f16_skylake", nk_dot_f16_skylake);
    run_dense<e5m2_k, f32_k>("dot_e5m2_skylake", nk_dot_e5m2_skylake);
    run_dense<e4m3_k, f32_k>("dot_e4m3_skylake", nk_dot_e4m3_skylake);
    run_dense<e3m2_k, f32_k>("dot_e3m2_skylake", nk_dot_e3m2_skylake);
    run_dense<e2m3_k, f32_k>("dot_e2m3_skylake", nk_dot_e2m3_skylake);
    run_dense<e2m1_k, f32_k>("dot_e2m1_skylake", nk_dot_e2m1_skylake);
    run_dense<i8_k, i32_k>("dot_i8_skylake", nk_dot_i8_skylake);
    run_dense<u8_k, u32_k>("dot_u8_skylake", nk_dot_u8_skylake);
#endif

#if NK_TARGET_ICELAKE
    run_dense<e4m3_k, f32_k>("dot_e4m3_icelake", nk_dot_e4m3_icelake);
    run_dense<e3m2_k, f32_k>("dot_e3m2_icelake", nk_dot_e3m2_icelake);
    run_dense<e2m3_k, f32_k>("dot_e2m3_icelake", nk_dot_e2m3_icelake);
    run_dense<i8_k, i32_k>("dot_i8_icelake", nk_dot_i8_icelake);
    run_dense<i4_k, i32_k>("dot_i4_icelake", nk_dot_i4_icelake);
    run_dense<u8_k, u32_k>("dot_u8_icelake", nk_dot_u8_icelake);
    run_dense<u4_k, u32_k>("dot_u4_icelake", nk_dot_u4_icelake);
    run_dense<u1_k, u32_k>("dot_u1_icelake", nk_dot_u1_icelake);
#endif

#if NK_TARGET_ALDER
    run_dense<e2m3_k, f32_k>("dot_e2m3_alder", nk_dot_e2m3_alder);
    run_dense<e2m1_k, f32_k>("dot_e2m1_alder", nk_dot_e2m1_alder);
    run_dense<i8_k, i32_k>("dot_i8_alder", nk_dot_i8_alder);
    run_dense<u8_k, u32_k>("dot_u8_alder", nk_dot_u8_alder);
#endif

#if NK_TARGET_SIERRA
    run_dense<e2m3_k, f32_k>("dot_e2m3_sierra", nk_dot_e2m3_sierra);
    run_dense<e2m1_k, f32_k>("dot_e2m1_sierra", nk_dot_e2m1_sierra);
    run_dense<i8_k, i32_k>("dot_i8_sierra", nk_dot_i8_sierra);
    run_dense<u8_k, u32_k>("dot_u8_sierra", nk_dot_u8_sierra);
#endif

#if NK_TARGET_GENOA
    run_dense<bf16c_k, f32c_k>("dot_bf16c_genoa", nk_dot_bf16c_genoa);
    run_dense<bf16c_k, f32c_k>("vdot_bf16c_genoa", nk_vdot_bf16c_genoa);
    run_dense<bf16_k, f32_k>("dot_bf16_genoa", nk_dot_bf16_genoa);
    run_dense<e5m2_k, f32_k>("dot_e5m2_genoa", nk_dot_e5m2_genoa);
#endif

#if NK_TARGET_DIAMOND
    run_dense<f16_k, f32_k>("dot_f16_diamond", nk_dot_f16_diamond);
    run_dense<e5m2_k, f32_k>("dot_e5m2_diamond", nk_dot_e5m2_diamond);
    run_dense<e4m3_k, f32_k>("dot_e4m3_diamond", nk_dot_e4m3_diamond);
#endif

#if NK_TARGET_RVV
    run_dense<f64_k, f64_k>("dot_f64_rvv", nk_dot_f64_rvv);
    run_dense<f32_k, f64_k>("dot_f32_rvv", nk_dot_f32_rvv);
    run_dense<i8_k, i32_k>("dot_i8_rvv", nk_dot_i8_rvv);
    run_dense<u8_k, u32_k>("dot_u8_rvv", nk_dot_u8_rvv);
    run_dense<u1_k, u32_k>("dot_u1_rvv", nk_dot_u1_rvv);
#endif

#if NK_TARGET_V128RELAXED
    run_dense<f64c_k, f64c_k>("dot_f64c_v128relaxed", nk_dot_f64c_v128relaxed);
    run_dense<f64c_k, f64c_k>("vdot_f64c_v128relaxed", nk_vdot_f64c_v128relaxed);
    run_dense<f32c_k, f64c_k>("dot_f32c_v128relaxed", nk_dot_f32c_v128relaxed);
    run_dense<f32c_k, f64c_k>("vdot_f32c_v128relaxed", nk_vdot_f32c_v128relaxed);
    run_dense<f64_k, f64_k>("dot_f64_v128relaxed", nk_dot_f64_v128relaxed);
    run_dense<f32_k, f64_k>("dot_f32_v128relaxed", nk_dot_f32_v128relaxed);
    run_dense<bf16_k, f32_k>("dot_bf16_v128relaxed", nk_dot_bf16_v128relaxed);
    run_dense<f16_k, f32_k>("dot_f16_v128relaxed", nk_dot_f16_v128relaxed);
    run_dense<e5m2_k, f32_k>("dot_e5m2_v128relaxed", nk_dot_e5m2_v128relaxed);
    run_dense<e4m3_k, f32_k>("dot_e4m3_v128relaxed", nk_dot_e4m3_v128relaxed);
    run_dense<e3m2_k, f32_k>("dot_e3m2_v128relaxed", nk_dot_e3m2_v128relaxed);
    run_dense<e2m3_k, f32_k>("dot_e2m3_v128relaxed", nk_dot_e2m3_v128relaxed);
    run_dense<e2m1_k, f32_k>("dot_e2m1_v128relaxed", nk_dot_e2m1_v128relaxed);
    run_dense<i8_k, i32_k>("dot_i8_v128relaxed", nk_dot_i8_v128relaxed);
    run_dense<i4_k, i32_k>("dot_i4_v128relaxed", nk_dot_i4_v128relaxed);
    run_dense<u8_k, u32_k>("dot_u8_v128relaxed", nk_dot_u8_v128relaxed);
    run_dense<u4_k, u32_k>("dot_u4_v128relaxed", nk_dot_u4_v128relaxed);
#endif

#if NK_TARGET_V128
    run_dense<bf16_k, f32_k>("dot_bf16_v128", nk_dot_bf16_v128);
    run_dense<i8_k, i32_k>("dot_i8_v128", nk_dot_i8_v128);
    run_dense<u8_k, u32_k>("dot_u8_v128", nk_dot_u8_v128);
    run_dense<u1_k, u32_k>("dot_u1_v128", nk_dot_u1_v128);
#endif

#if NK_TARGET_LOONGSONASX
    run_dense<f64_k, f64_k>("dot_f64_loongsonasx", nk_dot_f64_loongsonasx);
    run_dense<f32_k, f64_k>("dot_f32_loongsonasx", nk_dot_f32_loongsonasx);
    run_dense<bf16_k, f32_k>("dot_bf16_loongsonasx", nk_dot_bf16_loongsonasx);
    run_dense<i8_k, i32_k>("dot_i8_loongsonasx", nk_dot_i8_loongsonasx);
    run_dense<u8_k, u32_k>("dot_u8_loongsonasx", nk_dot_u8_loongsonasx);
#endif

    // Serial fallbacks
    run_dense<f64c_k, f64c_k>("dot_f64c_serial", nk_dot_f64c_serial);
    run_dense<f64c_k, f64c_k>("vdot_f64c_serial", nk_vdot_f64c_serial);
    run_dense<f32c_k, f64c_k>("dot_f32c_serial", nk_dot_f32c_serial);
    run_dense<f32c_k, f64c_k>("vdot_f32c_serial", nk_vdot_f32c_serial);
    run_dense<bf16c_k, f32c_k>("dot_bf16c_serial", nk_dot_bf16c_serial);
    run_dense<bf16c_k, f32c_k>("vdot_bf16c_serial", nk_vdot_bf16c_serial);
    run_dense<f16c_k, f32c_k>("dot_f16c_serial", nk_dot_f16c_serial);
    run_dense<f16c_k, f32c_k>("vdot_f16c_serial", nk_vdot_f16c_serial);
    run_dense<f64_k, f64_k>("dot_f64_serial", nk_dot_f64_serial);
    run_dense<f32_k, f64_k>("dot_f32_serial", nk_dot_f32_serial);
    run_dense<bf16_k, f32_k>("dot_bf16_serial", nk_dot_bf16_serial);
    run_dense<f16_k, f32_k>("dot_f16_serial", nk_dot_f16_serial);
    run_dense<e5m2_k, f32_k>("dot_e5m2_serial", nk_dot_e5m2_serial);
    run_dense<e4m3_k, f32_k>("dot_e4m3_serial", nk_dot_e4m3_serial);
    run_dense<e3m2_k, f32_k>("dot_e3m2_serial", nk_dot_e3m2_serial);
    run_dense<e2m3_k, f32_k>("dot_e2m3_serial", nk_dot_e2m3_serial);
    run_dense<e2m1_k, f32_k>("dot_e2m1_serial", nk_dot_e2m1_serial);
    run_dense<i8_k, i32_k>("dot_i8_serial", nk_dot_i8_serial);
    run_dense<i4_k, i32_k>("dot_i4_serial", nk_dot_i4_serial);
    run_dense<u8_k, u32_k>("dot_u8_serial", nk_dot_u8_serial);
    run_dense<u4_k, u32_k>("dot_u4_serial", nk_dot_u4_serial);
    run_dense<u1_k, u32_k>("dot_u1_serial", nk_dot_u1_serial);
}
