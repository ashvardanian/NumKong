/**
 *  @file bench/spatial.cpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 *  @brief Spatial distance benchmarks, angular, sqeuclidean and euclidean.
 */

#include "numkong/spatial.h"

#include "harness.hpp"

namespace ashvardanian::numkong::bench {

void bench_spatial(environment_t const &env) {
    constexpr nk_dtype_t i4_k = nk_i4_k;
    constexpr nk_dtype_t u4_k = nk_u4_k;
    constexpr nk_dtype_t i8_k = nk_i8_k;
    constexpr nk_dtype_t u8_k = nk_u8_k;
    constexpr nk_dtype_t u32_k = nk_u32_k;
    constexpr nk_dtype_t f64_k = nk_f64_k;
    constexpr nk_dtype_t f32_k = nk_f32_k;
    constexpr nk_dtype_t f16_k = nk_f16_k;
    constexpr nk_dtype_t bf16_k = nk_bf16_k;
    constexpr nk_dtype_t e4m3_k = nk_e4m3_k;
    constexpr nk_dtype_t e5m2_k = nk_e5m2_k;
    constexpr nk_dtype_t e2m3_k = nk_e2m3_k;
    constexpr nk_dtype_t e3m2_k = nk_e3m2_k;

#if NUMKONG_TARGET_NEON
    if (section(env, "Spatial Distances NEON", nk_cap_neon_k)) {
        run_dense<f64_k, f64_k>(env, "angular_f64_neon", nk_angular_f64_neon);
        run_dense<f64_k, f64_k>(env, "sqeuclidean_f64_neon", nk_sqeuclidean_f64_neon);
        run_dense<f64_k, f64_k>(env, "euclidean_f64_neon", nk_euclidean_f64_neon);
        run_dense<f32_k, f64_k>(env, "angular_f32_neon", nk_angular_f32_neon);
        run_dense<f32_k, f64_k>(env, "sqeuclidean_f32_neon", nk_sqeuclidean_f32_neon);
        run_dense<f32_k, f64_k>(env, "euclidean_f32_neon", nk_euclidean_f32_neon);
        run_dense<bf16_k, f32_k>(env, "angular_bf16_neon", nk_angular_bf16_neon);
        run_dense<bf16_k, f32_k>(env, "sqeuclidean_bf16_neon", nk_sqeuclidean_bf16_neon);
        run_dense<bf16_k, f32_k>(env, "euclidean_bf16_neon", nk_euclidean_bf16_neon);
        run_dense<f16_k, f32_k>(env, "angular_f16_neon", nk_angular_f16_neon);
        run_dense<f16_k, f32_k>(env, "sqeuclidean_f16_neon", nk_sqeuclidean_f16_neon);
        run_dense<f16_k, f32_k>(env, "euclidean_f16_neon", nk_euclidean_f16_neon);
        run_dense<e5m2_k, f32_k>(env, "angular_e5m2_neon", nk_angular_e5m2_neon);
        run_dense<e5m2_k, f32_k>(env, "sqeuclidean_e5m2_neon", nk_sqeuclidean_e5m2_neon);
        run_dense<e5m2_k, f32_k>(env, "euclidean_e5m2_neon", nk_euclidean_e5m2_neon);
        run_dense<e4m3_k, f32_k>(env, "angular_e4m3_neon", nk_angular_e4m3_neon);
        run_dense<e4m3_k, f32_k>(env, "sqeuclidean_e4m3_neon", nk_sqeuclidean_e4m3_neon);
        run_dense<e4m3_k, f32_k>(env, "euclidean_e4m3_neon", nk_euclidean_e4m3_neon);
        run_dense<e3m2_k, f32_k>(env, "angular_e3m2_neon", nk_angular_e3m2_neon);
        run_dense<e3m2_k, f32_k>(env, "sqeuclidean_e3m2_neon", nk_sqeuclidean_e3m2_neon);
        run_dense<e3m2_k, f32_k>(env, "euclidean_e3m2_neon", nk_euclidean_e3m2_neon);
        run_dense<e2m3_k, f32_k>(env, "angular_e2m3_neon", nk_angular_e2m3_neon);
        run_dense<e2m3_k, f32_k>(env, "sqeuclidean_e2m3_neon", nk_sqeuclidean_e2m3_neon);
        run_dense<e2m3_k, f32_k>(env, "euclidean_e2m3_neon", nk_euclidean_e2m3_neon);
    }
#endif

#if NUMKONG_TARGET_NEONSDOT
    if (section(env, "Spatial Distances NEON I8", nk_cap_neonsdot_k)) {
        run_dense<i8_k, f32_k>(env, "angular_i8_neonsdot", nk_angular_i8_neonsdot);
        run_dense<i8_k, u32_k>(env, "sqeuclidean_i8_neonsdot", nk_sqeuclidean_i8_neonsdot);
        run_dense<i8_k, f32_k>(env, "euclidean_i8_neonsdot", nk_euclidean_i8_neonsdot);
        run_dense<u8_k, f32_k>(env, "angular_u8_neonsdot", nk_angular_u8_neonsdot);
        run_dense<u8_k, u32_k>(env, "sqeuclidean_u8_neonsdot", nk_sqeuclidean_u8_neonsdot);
        run_dense<u8_k, f32_k>(env, "euclidean_u8_neonsdot", nk_euclidean_u8_neonsdot);
        run_dense<i4_k, f32_k>(env, "angular_i4_neonsdot", nk_angular_i4_neonsdot);
        run_dense<i4_k, u32_k>(env, "sqeuclidean_i4_neonsdot", nk_sqeuclidean_i4_neonsdot);
        run_dense<i4_k, f32_k>(env, "euclidean_i4_neonsdot", nk_euclidean_i4_neonsdot);
        run_dense<u4_k, f32_k>(env, "angular_u4_neonsdot", nk_angular_u4_neonsdot);
        run_dense<u4_k, u32_k>(env, "sqeuclidean_u4_neonsdot", nk_sqeuclidean_u4_neonsdot);
        run_dense<u4_k, f32_k>(env, "euclidean_u4_neonsdot", nk_euclidean_u4_neonsdot);
    }
#endif

#if NUMKONG_TARGET_NEONBFDOT
    if (section(env, "Spatial Distances NEON BF16", nk_cap_neonbfdot_k)) {
        run_dense<bf16_k, f32_k>(env, "angular_bf16_neonbfdot", nk_angular_bf16_neonbfdot);
        run_dense<bf16_k, f32_k>(env, "sqeuclidean_bf16_neonbfdot", nk_sqeuclidean_bf16_neonbfdot);
        run_dense<bf16_k, f32_k>(env, "euclidean_bf16_neonbfdot", nk_euclidean_bf16_neonbfdot);
    }
#endif

#if NUMKONG_TARGET_SVE
    if (section(env, "Spatial Distances SVE", nk_cap_sve_k)) {
        run_dense<f64_k, f64_k>(env, "angular_f64_sve", nk_angular_f64_sve);
        run_dense<f64_k, f64_k>(env, "sqeuclidean_f64_sve", nk_sqeuclidean_f64_sve);
        run_dense<f64_k, f64_k>(env, "euclidean_f64_sve", nk_euclidean_f64_sve);
        run_dense<f32_k, f64_k>(env, "angular_f32_sve", nk_angular_f32_sve);
        run_dense<f32_k, f64_k>(env, "sqeuclidean_f32_sve", nk_sqeuclidean_f32_sve);
        run_dense<f32_k, f64_k>(env, "euclidean_f32_sve", nk_euclidean_f32_sve);
    }
#endif

#if NUMKONG_TARGET_SVEHALF
    if (section(env, "Spatial Distances SVE HALF", nk_cap_svehalf_k)) {
        run_dense<f16_k, f32_k>(env, "angular_f16_svehalf", nk_angular_f16_svehalf);
        run_dense<f16_k, f32_k>(env, "sqeuclidean_f16_svehalf", nk_sqeuclidean_f16_svehalf);
        run_dense<f16_k, f32_k>(env, "euclidean_f16_svehalf", nk_euclidean_f16_svehalf);
    }
#endif

#if NUMKONG_TARGET_SVEBFDOT
    if (section(env, "Spatial Distances SVE BF16", nk_cap_svebfdot_k)) {
        run_dense<bf16_k, f32_k>(env, "angular_bf16_svebfdot", nk_angular_bf16_svebfdot);
        run_dense<bf16_k, f32_k>(env, "sqeuclidean_bf16_svebfdot", nk_sqeuclidean_bf16_svebfdot);
        run_dense<bf16_k, f32_k>(env, "euclidean_bf16_svebfdot", nk_euclidean_bf16_svebfdot);
    }
#endif

#if NUMKONG_TARGET_SVESDOT
    if (section(env, "Spatial Distances SVE I8", nk_cap_svesdot_k)) {
        run_dense<i8_k, f32_k>(env, "angular_i8_svesdot", nk_angular_i8_svesdot);
        run_dense<i8_k, u32_k>(env, "sqeuclidean_i8_svesdot", nk_sqeuclidean_i8_svesdot);
        run_dense<i8_k, f32_k>(env, "euclidean_i8_svesdot", nk_euclidean_i8_svesdot);
        run_dense<u8_k, f32_k>(env, "angular_u8_svesdot", nk_angular_u8_svesdot);
        run_dense<u8_k, u32_k>(env, "sqeuclidean_u8_svesdot", nk_sqeuclidean_u8_svesdot);
        run_dense<u8_k, f32_k>(env, "euclidean_u8_svesdot", nk_euclidean_u8_svesdot);
    }
#endif

#if NUMKONG_TARGET_NEONFP8
    if (section(env, "Spatial Distances NEON FP8", nk_cap_neonfp8_k)) {
        run_dense<e4m3_k, f32_k>(env, "angular_e4m3_neonfp8", nk_angular_e4m3_neonfp8);
        run_dense<e4m3_k, f32_k>(env, "sqeuclidean_e4m3_neonfp8", nk_sqeuclidean_e4m3_neonfp8);
        run_dense<e4m3_k, f32_k>(env, "euclidean_e4m3_neonfp8", nk_euclidean_e4m3_neonfp8);
        run_dense<e5m2_k, f32_k>(env, "angular_e5m2_neonfp8", nk_angular_e5m2_neonfp8);
        run_dense<e5m2_k, f32_k>(env, "sqeuclidean_e5m2_neonfp8", nk_sqeuclidean_e5m2_neonfp8);
        run_dense<e5m2_k, f32_k>(env, "euclidean_e5m2_neonfp8", nk_euclidean_e5m2_neonfp8);
        run_dense<e2m3_k, f32_k>(env, "angular_e2m3_neonfp8", nk_angular_e2m3_neonfp8);
        run_dense<e2m3_k, f32_k>(env, "sqeuclidean_e2m3_neonfp8", nk_sqeuclidean_e2m3_neonfp8);
        run_dense<e2m3_k, f32_k>(env, "euclidean_e2m3_neonfp8", nk_euclidean_e2m3_neonfp8);
        run_dense<e3m2_k, f32_k>(env, "angular_e3m2_neonfp8", nk_angular_e3m2_neonfp8);
        run_dense<e3m2_k, f32_k>(env, "sqeuclidean_e3m2_neonfp8", nk_sqeuclidean_e3m2_neonfp8);
        run_dense<e3m2_k, f32_k>(env, "euclidean_e3m2_neonfp8", nk_euclidean_e3m2_neonfp8);
    }
#endif

#if NUMKONG_TARGET_HASWELL
    if (section(env, "Spatial Distances Haswell", nk_cap_haswell_k)) {
        run_dense<f64_k, f64_k>(env, "angular_f64_haswell", nk_angular_f64_haswell);
        run_dense<f64_k, f64_k>(env, "sqeuclidean_f64_haswell", nk_sqeuclidean_f64_haswell);
        run_dense<f64_k, f64_k>(env, "euclidean_f64_haswell", nk_euclidean_f64_haswell);
        run_dense<f32_k, f64_k>(env, "angular_f32_haswell", nk_angular_f32_haswell);
        run_dense<f32_k, f64_k>(env, "sqeuclidean_f32_haswell", nk_sqeuclidean_f32_haswell);
        run_dense<f32_k, f64_k>(env, "euclidean_f32_haswell", nk_euclidean_f32_haswell);
        run_dense<bf16_k, f32_k>(env, "angular_bf16_haswell", nk_angular_bf16_haswell);
        run_dense<bf16_k, f32_k>(env, "sqeuclidean_bf16_haswell", nk_sqeuclidean_bf16_haswell);
        run_dense<bf16_k, f32_k>(env, "euclidean_bf16_haswell", nk_euclidean_bf16_haswell);
        run_dense<f16_k, f32_k>(env, "angular_f16_haswell", nk_angular_f16_haswell);
        run_dense<f16_k, f32_k>(env, "sqeuclidean_f16_haswell", nk_sqeuclidean_f16_haswell);
        run_dense<f16_k, f32_k>(env, "euclidean_f16_haswell", nk_euclidean_f16_haswell);
        run_dense<e5m2_k, f32_k>(env, "angular_e5m2_haswell", nk_angular_e5m2_haswell);
        run_dense<e5m2_k, f32_k>(env, "sqeuclidean_e5m2_haswell", nk_sqeuclidean_e5m2_haswell);
        run_dense<e5m2_k, f32_k>(env, "euclidean_e5m2_haswell", nk_euclidean_e5m2_haswell);
        run_dense<e4m3_k, f32_k>(env, "angular_e4m3_haswell", nk_angular_e4m3_haswell);
        run_dense<e4m3_k, f32_k>(env, "sqeuclidean_e4m3_haswell", nk_sqeuclidean_e4m3_haswell);
        run_dense<e4m3_k, f32_k>(env, "euclidean_e4m3_haswell", nk_euclidean_e4m3_haswell);
        run_dense<e3m2_k, f32_k>(env, "angular_e3m2_haswell", nk_angular_e3m2_haswell);
        run_dense<e3m2_k, f32_k>(env, "sqeuclidean_e3m2_haswell", nk_sqeuclidean_e3m2_haswell);
        run_dense<e3m2_k, f32_k>(env, "euclidean_e3m2_haswell", nk_euclidean_e3m2_haswell);
        run_dense<e2m3_k, f32_k>(env, "angular_e2m3_haswell", nk_angular_e2m3_haswell);
        run_dense<e2m3_k, f32_k>(env, "sqeuclidean_e2m3_haswell", nk_sqeuclidean_e2m3_haswell);
        run_dense<e2m3_k, f32_k>(env, "euclidean_e2m3_haswell", nk_euclidean_e2m3_haswell);
        run_dense<i8_k, f32_k>(env, "angular_i8_haswell", nk_angular_i8_haswell);
        run_dense<i8_k, u32_k>(env, "sqeuclidean_i8_haswell", nk_sqeuclidean_i8_haswell);
        run_dense<i8_k, f32_k>(env, "euclidean_i8_haswell", nk_euclidean_i8_haswell);
        run_dense<u8_k, f32_k>(env, "angular_u8_haswell", nk_angular_u8_haswell);
        run_dense<u8_k, u32_k>(env, "sqeuclidean_u8_haswell", nk_sqeuclidean_u8_haswell);
        run_dense<u8_k, f32_k>(env, "euclidean_u8_haswell", nk_euclidean_u8_haswell);
    }
#endif

#if NUMKONG_TARGET_SKYLAKE
    if (section(env, "Spatial Distances Skylake", nk_cap_skylake_k)) {
        run_dense<f64_k, f64_k>(env, "angular_f64_skylake", nk_angular_f64_skylake);
        run_dense<f64_k, f64_k>(env, "sqeuclidean_f64_skylake", nk_sqeuclidean_f64_skylake);
        run_dense<f64_k, f64_k>(env, "euclidean_f64_skylake", nk_euclidean_f64_skylake);
        run_dense<f32_k, f64_k>(env, "angular_f32_skylake", nk_angular_f32_skylake);
        run_dense<f32_k, f64_k>(env, "sqeuclidean_f32_skylake", nk_sqeuclidean_f32_skylake);
        run_dense<f32_k, f64_k>(env, "euclidean_f32_skylake", nk_euclidean_f32_skylake);
        run_dense<f16_k, f32_k>(env, "angular_f16_skylake", nk_angular_f16_skylake);
        run_dense<f16_k, f32_k>(env, "sqeuclidean_f16_skylake", nk_sqeuclidean_f16_skylake);
        run_dense<f16_k, f32_k>(env, "euclidean_f16_skylake", nk_euclidean_f16_skylake);
        run_dense<e5m2_k, f32_k>(env, "angular_e5m2_skylake", nk_angular_e5m2_skylake);
        run_dense<e5m2_k, f32_k>(env, "sqeuclidean_e5m2_skylake", nk_sqeuclidean_e5m2_skylake);
        run_dense<e5m2_k, f32_k>(env, "euclidean_e5m2_skylake", nk_euclidean_e5m2_skylake);
        run_dense<e4m3_k, f32_k>(env, "angular_e4m3_skylake", nk_angular_e4m3_skylake);
        run_dense<e4m3_k, f32_k>(env, "sqeuclidean_e4m3_skylake", nk_sqeuclidean_e4m3_skylake);
        run_dense<e4m3_k, f32_k>(env, "euclidean_e4m3_skylake", nk_euclidean_e4m3_skylake);
        run_dense<e3m2_k, f32_k>(env, "angular_e3m2_skylake", nk_angular_e3m2_skylake);
        run_dense<e3m2_k, f32_k>(env, "sqeuclidean_e3m2_skylake", nk_sqeuclidean_e3m2_skylake);
        run_dense<e3m2_k, f32_k>(env, "euclidean_e3m2_skylake", nk_euclidean_e3m2_skylake);
        run_dense<e2m3_k, f32_k>(env, "angular_e2m3_skylake", nk_angular_e2m3_skylake);
        run_dense<e2m3_k, f32_k>(env, "sqeuclidean_e2m3_skylake", nk_sqeuclidean_e2m3_skylake);
        run_dense<e2m3_k, f32_k>(env, "euclidean_e2m3_skylake", nk_euclidean_e2m3_skylake);
    }
#endif

#if NUMKONG_TARGET_ALDER
    if (section(env, "Spatial Distances Alder", nk_cap_alder_k)) {
        run_dense<e3m2_k, f32_k>(env, "angular_e3m2_alder", nk_angular_e3m2_alder);
        run_dense<e3m2_k, f32_k>(env, "sqeuclidean_e3m2_alder", nk_sqeuclidean_e3m2_alder);
        run_dense<e3m2_k, f32_k>(env, "euclidean_e3m2_alder", nk_euclidean_e3m2_alder);
        run_dense<e2m3_k, f32_k>(env, "angular_e2m3_alder", nk_angular_e2m3_alder);
        run_dense<e2m3_k, f32_k>(env, "sqeuclidean_e2m3_alder", nk_sqeuclidean_e2m3_alder);
        run_dense<e2m3_k, f32_k>(env, "euclidean_e2m3_alder", nk_euclidean_e2m3_alder);
        run_dense<i8_k, f32_k>(env, "angular_i8_alder", nk_angular_i8_alder);
        run_dense<i8_k, u32_k>(env, "sqeuclidean_i8_alder", nk_sqeuclidean_i8_alder);
        run_dense<i8_k, f32_k>(env, "euclidean_i8_alder", nk_euclidean_i8_alder);
        run_dense<u8_k, f32_k>(env, "angular_u8_alder", nk_angular_u8_alder);
        run_dense<u8_k, u32_k>(env, "sqeuclidean_u8_alder", nk_sqeuclidean_u8_alder);
        run_dense<u8_k, f32_k>(env, "euclidean_u8_alder", nk_euclidean_u8_alder);
    }
#endif

#if NUMKONG_TARGET_SIERRA
    if (section(env, "Spatial Distances Sierra", nk_cap_sierra_k)) {
        run_dense<e2m3_k, f32_k>(env, "angular_e2m3_sierra", nk_angular_e2m3_sierra);
        run_dense<e2m3_k, f32_k>(env, "sqeuclidean_e2m3_sierra", nk_sqeuclidean_e2m3_sierra);
        run_dense<e2m3_k, f32_k>(env, "euclidean_e2m3_sierra", nk_euclidean_e2m3_sierra);
        run_dense<e3m2_k, f32_k>(env, "angular_e3m2_sierra", nk_angular_e3m2_sierra);
        run_dense<e3m2_k, f32_k>(env, "sqeuclidean_e3m2_sierra", nk_sqeuclidean_e3m2_sierra);
        run_dense<e3m2_k, f32_k>(env, "euclidean_e3m2_sierra", nk_euclidean_e3m2_sierra);
        run_dense<i8_k, f32_k>(env, "angular_i8_sierra", nk_angular_i8_sierra);
        run_dense<i8_k, u32_k>(env, "sqeuclidean_i8_sierra", nk_sqeuclidean_i8_sierra);
        run_dense<i8_k, f32_k>(env, "euclidean_i8_sierra", nk_euclidean_i8_sierra);
        run_dense<u8_k, f32_k>(env, "angular_u8_sierra", nk_angular_u8_sierra);
        run_dense<u8_k, u32_k>(env, "sqeuclidean_u8_sierra", nk_sqeuclidean_u8_sierra);
        run_dense<u8_k, f32_k>(env, "euclidean_u8_sierra", nk_euclidean_u8_sierra);
    }
#endif

#if NUMKONG_TARGET_ICELAKE
    if (section(env, "Spatial Distances Ice Lake", nk_cap_icelake_k)) {
        run_dense<i8_k, f32_k>(env, "angular_i8_icelake", nk_angular_i8_icelake);
        run_dense<i8_k, u32_k>(env, "sqeuclidean_i8_icelake", nk_sqeuclidean_i8_icelake);
        run_dense<i8_k, f32_k>(env, "euclidean_i8_icelake", nk_euclidean_i8_icelake);
        run_dense<u8_k, f32_k>(env, "angular_u8_icelake", nk_angular_u8_icelake);
        run_dense<u8_k, u32_k>(env, "sqeuclidean_u8_icelake", nk_sqeuclidean_u8_icelake);
        run_dense<u8_k, f32_k>(env, "euclidean_u8_icelake", nk_euclidean_u8_icelake);
        run_dense<i4_k, f32_k>(env, "angular_i4_icelake", nk_angular_i4_icelake);
        run_dense<i4_k, u32_k>(env, "sqeuclidean_i4_icelake", nk_sqeuclidean_i4_icelake);
        run_dense<i4_k, f32_k>(env, "euclidean_i4_icelake", nk_euclidean_i4_icelake);
        run_dense<u4_k, f32_k>(env, "angular_u4_icelake", nk_angular_u4_icelake);
        run_dense<u4_k, u32_k>(env, "sqeuclidean_u4_icelake", nk_sqeuclidean_u4_icelake);
        run_dense<u4_k, f32_k>(env, "euclidean_u4_icelake", nk_euclidean_u4_icelake);
        run_dense<e4m3_k, f32_k>(env, "angular_e4m3_icelake", nk_angular_e4m3_icelake);
        run_dense<e4m3_k, f32_k>(env, "sqeuclidean_e4m3_icelake", nk_sqeuclidean_e4m3_icelake);
        run_dense<e4m3_k, f32_k>(env, "euclidean_e4m3_icelake", nk_euclidean_e4m3_icelake);
        run_dense<e2m3_k, f32_k>(env, "angular_e2m3_icelake", nk_angular_e2m3_icelake);
        run_dense<e2m3_k, f32_k>(env, "sqeuclidean_e2m3_icelake", nk_sqeuclidean_e2m3_icelake);
        run_dense<e2m3_k, f32_k>(env, "euclidean_e2m3_icelake", nk_euclidean_e2m3_icelake);
        run_dense<e3m2_k, f32_k>(env, "angular_e3m2_icelake", nk_angular_e3m2_icelake);
        run_dense<e3m2_k, f32_k>(env, "sqeuclidean_e3m2_icelake", nk_sqeuclidean_e3m2_icelake);
        run_dense<e3m2_k, f32_k>(env, "euclidean_e3m2_icelake", nk_euclidean_e3m2_icelake);
    }
#endif

#if NUMKONG_TARGET_GENOA
    if (section(env, "Spatial Distances Genoa", nk_cap_genoa_k)) {
        run_dense<bf16_k, f32_k>(env, "angular_bf16_genoa", nk_angular_bf16_genoa);
        run_dense<bf16_k, f32_k>(env, "sqeuclidean_bf16_genoa", nk_sqeuclidean_bf16_genoa);
        run_dense<bf16_k, f32_k>(env, "euclidean_bf16_genoa", nk_euclidean_bf16_genoa);
    }
#endif

#if NUMKONG_TARGET_DIAMOND
    if (section(env, "Spatial Distances Diamond", nk_cap_diamond_k)) {
        run_dense<f16_k, f32_k>(env, "angular_f16_diamond", nk_angular_f16_diamond);
        run_dense<f16_k, f32_k>(env, "sqeuclidean_f16_diamond", nk_sqeuclidean_f16_diamond);
        run_dense<f16_k, f32_k>(env, "euclidean_f16_diamond", nk_euclidean_f16_diamond);
        run_dense<e4m3_k, f32_k>(env, "angular_e4m3_diamond", nk_angular_e4m3_diamond);
        run_dense<e4m3_k, f32_k>(env, "sqeuclidean_e4m3_diamond", nk_sqeuclidean_e4m3_diamond);
        run_dense<e4m3_k, f32_k>(env, "euclidean_e4m3_diamond", nk_euclidean_e4m3_diamond);
        run_dense<e5m2_k, f32_k>(env, "angular_e5m2_diamond", nk_angular_e5m2_diamond);
        run_dense<e5m2_k, f32_k>(env, "sqeuclidean_e5m2_diamond", nk_sqeuclidean_e5m2_diamond);
        run_dense<e5m2_k, f32_k>(env, "euclidean_e5m2_diamond", nk_euclidean_e5m2_diamond);
    }
#endif

#if NUMKONG_TARGET_RVV
    if (section(env, "Spatial Distances RVV", nk_cap_rvv_k)) {
        run_dense<f64_k, f64_k>(env, "angular_f64_rvv", nk_angular_f64_rvv);
        run_dense<f64_k, f64_k>(env, "sqeuclidean_f64_rvv", nk_sqeuclidean_f64_rvv);
        run_dense<f32_k, f64_k>(env, "angular_f32_rvv", nk_angular_f32_rvv);
        run_dense<f32_k, f64_k>(env, "sqeuclidean_f32_rvv", nk_sqeuclidean_f32_rvv);
    }
#endif

#if NUMKONG_TARGET_V128RELAXED
    if (section(env, "Spatial Distances V128 Relaxed", nk_cap_v128relaxed_k)) {
        run_dense<f64_k, f64_k>(env, "angular_f64_v128relaxed", nk_angular_f64_v128relaxed);
        run_dense<f64_k, f64_k>(env, "sqeuclidean_f64_v128relaxed", nk_sqeuclidean_f64_v128relaxed);
        run_dense<f64_k, f64_k>(env, "euclidean_f64_v128relaxed", nk_euclidean_f64_v128relaxed);
        run_dense<f32_k, f64_k>(env, "angular_f32_v128relaxed", nk_angular_f32_v128relaxed);
        run_dense<f32_k, f64_k>(env, "sqeuclidean_f32_v128relaxed", nk_sqeuclidean_f32_v128relaxed);
        run_dense<f32_k, f64_k>(env, "euclidean_f32_v128relaxed", nk_euclidean_f32_v128relaxed);
        run_dense<bf16_k, f32_k>(env, "angular_bf16_v128relaxed", nk_angular_bf16_v128relaxed);
        run_dense<bf16_k, f32_k>(env, "sqeuclidean_bf16_v128relaxed", nk_sqeuclidean_bf16_v128relaxed);
        run_dense<bf16_k, f32_k>(env, "euclidean_bf16_v128relaxed", nk_euclidean_bf16_v128relaxed);
        run_dense<f16_k, f32_k>(env, "angular_f16_v128relaxed", nk_angular_f16_v128relaxed);
        run_dense<f16_k, f32_k>(env, "sqeuclidean_f16_v128relaxed", nk_sqeuclidean_f16_v128relaxed);
        run_dense<f16_k, f32_k>(env, "euclidean_f16_v128relaxed", nk_euclidean_f16_v128relaxed);
        run_dense<i8_k, f32_k>(env, "angular_i8_v128relaxed", nk_angular_i8_v128relaxed);
        run_dense<i8_k, u32_k>(env, "sqeuclidean_i8_v128relaxed", nk_sqeuclidean_i8_v128relaxed);
        run_dense<i8_k, f32_k>(env, "euclidean_i8_v128relaxed", nk_euclidean_i8_v128relaxed);
        run_dense<u8_k, f32_k>(env, "angular_u8_v128relaxed", nk_angular_u8_v128relaxed);
        run_dense<u8_k, u32_k>(env, "sqeuclidean_u8_v128relaxed", nk_sqeuclidean_u8_v128relaxed);
        run_dense<u8_k, f32_k>(env, "euclidean_u8_v128relaxed", nk_euclidean_u8_v128relaxed);
        run_dense<e4m3_k, f32_k>(env, "angular_e4m3_v128relaxed", nk_angular_e4m3_v128relaxed);
        run_dense<e4m3_k, f32_k>(env, "sqeuclidean_e4m3_v128relaxed", nk_sqeuclidean_e4m3_v128relaxed);
        run_dense<e4m3_k, f32_k>(env, "euclidean_e4m3_v128relaxed", nk_euclidean_e4m3_v128relaxed);
        run_dense<e5m2_k, f32_k>(env, "angular_e5m2_v128relaxed", nk_angular_e5m2_v128relaxed);
        run_dense<e5m2_k, f32_k>(env, "sqeuclidean_e5m2_v128relaxed", nk_sqeuclidean_e5m2_v128relaxed);
        run_dense<e5m2_k, f32_k>(env, "euclidean_e5m2_v128relaxed", nk_euclidean_e5m2_v128relaxed);
        run_dense<e2m3_k, f32_k>(env, "angular_e2m3_v128relaxed", nk_angular_e2m3_v128relaxed);
        run_dense<e2m3_k, f32_k>(env, "sqeuclidean_e2m3_v128relaxed", nk_sqeuclidean_e2m3_v128relaxed);
        run_dense<e2m3_k, f32_k>(env, "euclidean_e2m3_v128relaxed", nk_euclidean_e2m3_v128relaxed);
        run_dense<e3m2_k, f32_k>(env, "angular_e3m2_v128relaxed", nk_angular_e3m2_v128relaxed);
        run_dense<e3m2_k, f32_k>(env, "sqeuclidean_e3m2_v128relaxed", nk_sqeuclidean_e3m2_v128relaxed);
        run_dense<e3m2_k, f32_k>(env, "euclidean_e3m2_v128relaxed", nk_euclidean_e3m2_v128relaxed);
    }
#endif

#if NUMKONG_TARGET_V128
    if (section(env, "Spatial Distances V128", nk_cap_v128_k)) {
        run_dense<bf16_k, f32_k>(env, "angular_bf16_v128", nk_angular_bf16_v128);
        run_dense<bf16_k, f32_k>(env, "sqeuclidean_bf16_v128", nk_sqeuclidean_bf16_v128);
        run_dense<bf16_k, f32_k>(env, "euclidean_bf16_v128", nk_euclidean_bf16_v128);
        run_dense<i8_k, f32_k>(env, "angular_i8_v128", nk_angular_i8_v128);
        run_dense<i8_k, u32_k>(env, "sqeuclidean_i8_v128", nk_sqeuclidean_i8_v128);
        run_dense<i8_k, f32_k>(env, "euclidean_i8_v128", nk_euclidean_i8_v128);
        run_dense<u8_k, f32_k>(env, "angular_u8_v128", nk_angular_u8_v128);
        run_dense<u8_k, u32_k>(env, "sqeuclidean_u8_v128", nk_sqeuclidean_u8_v128);
        run_dense<u8_k, f32_k>(env, "euclidean_u8_v128", nk_euclidean_u8_v128);
    }
#endif

#if NUMKONG_TARGET_LOONGSONASX
    if (section(env, "Spatial Distances LoongArch LASX", nk_cap_loongsonasx_k)) {
        run_dense<f64_k, f64_k>(env, "angular_f64_loongsonasx", nk_angular_f64_loongsonasx);
        run_dense<f64_k, f64_k>(env, "sqeuclidean_f64_loongsonasx", nk_sqeuclidean_f64_loongsonasx);
        run_dense<f64_k, f64_k>(env, "euclidean_f64_loongsonasx", nk_euclidean_f64_loongsonasx);
        run_dense<f32_k, f64_k>(env, "angular_f32_loongsonasx", nk_angular_f32_loongsonasx);
        run_dense<f32_k, f64_k>(env, "sqeuclidean_f32_loongsonasx", nk_sqeuclidean_f32_loongsonasx);
        run_dense<f32_k, f64_k>(env, "euclidean_f32_loongsonasx", nk_euclidean_f32_loongsonasx);
        run_dense<bf16_k, f32_k>(env, "angular_bf16_loongsonasx", nk_angular_bf16_loongsonasx);
        run_dense<bf16_k, f32_k>(env, "sqeuclidean_bf16_loongsonasx", nk_sqeuclidean_bf16_loongsonasx);
        run_dense<bf16_k, f32_k>(env, "euclidean_bf16_loongsonasx", nk_euclidean_bf16_loongsonasx);
        run_dense<i8_k, f32_k>(env, "angular_i8_loongsonasx", nk_angular_i8_loongsonasx);
        run_dense<i8_k, f32_k>(env, "sqeuclidean_i8_loongsonasx", nk_sqeuclidean_i8_loongsonasx);
        run_dense<i8_k, f32_k>(env, "euclidean_i8_loongsonasx", nk_euclidean_i8_loongsonasx);
        run_dense<u8_k, f32_k>(env, "angular_u8_loongsonasx", nk_angular_u8_loongsonasx);
        run_dense<u8_k, f32_k>(env, "sqeuclidean_u8_loongsonasx", nk_sqeuclidean_u8_loongsonasx);
        run_dense<u8_k, f32_k>(env, "euclidean_u8_loongsonasx", nk_euclidean_u8_loongsonasx);
    }
#endif

    // Serial fallbacks
    section(env, "Spatial Distances Serial", nk_cap_serial_k);
    run_dense<f64_k, f64_k>(env, "angular_f64_serial", nk_angular_f64_serial);
    run_dense<f64_k, f64_k>(env, "sqeuclidean_f64_serial", nk_sqeuclidean_f64_serial);
    run_dense<f64_k, f64_k>(env, "euclidean_f64_serial", nk_euclidean_f64_serial);
    run_dense<f32_k, f64_k>(env, "angular_f32_serial", nk_angular_f32_serial);
    run_dense<f32_k, f64_k>(env, "sqeuclidean_f32_serial", nk_sqeuclidean_f32_serial);
    run_dense<f32_k, f64_k>(env, "euclidean_f32_serial", nk_euclidean_f32_serial);
    run_dense<bf16_k, f32_k>(env, "angular_bf16_serial", nk_angular_bf16_serial);
    run_dense<bf16_k, f32_k>(env, "sqeuclidean_bf16_serial", nk_sqeuclidean_bf16_serial);
    run_dense<bf16_k, f32_k>(env, "euclidean_bf16_serial", nk_euclidean_bf16_serial);
    run_dense<f16_k, f32_k>(env, "angular_f16_serial", nk_angular_f16_serial);
    run_dense<f16_k, f32_k>(env, "sqeuclidean_f16_serial", nk_sqeuclidean_f16_serial);
    run_dense<f16_k, f32_k>(env, "euclidean_f16_serial", nk_euclidean_f16_serial);
    run_dense<e5m2_k, f32_k>(env, "angular_e5m2_serial", nk_angular_e5m2_serial);
    run_dense<e5m2_k, f32_k>(env, "sqeuclidean_e5m2_serial", nk_sqeuclidean_e5m2_serial);
    run_dense<e5m2_k, f32_k>(env, "euclidean_e5m2_serial", nk_euclidean_e5m2_serial);
    run_dense<e4m3_k, f32_k>(env, "angular_e4m3_serial", nk_angular_e4m3_serial);
    run_dense<e4m3_k, f32_k>(env, "sqeuclidean_e4m3_serial", nk_sqeuclidean_e4m3_serial);
    run_dense<e4m3_k, f32_k>(env, "euclidean_e4m3_serial", nk_euclidean_e4m3_serial);
    run_dense<e3m2_k, f32_k>(env, "angular_e3m2_serial", nk_angular_e3m2_serial);
    run_dense<e3m2_k, f32_k>(env, "sqeuclidean_e3m2_serial", nk_sqeuclidean_e3m2_serial);
    run_dense<e3m2_k, f32_k>(env, "euclidean_e3m2_serial", nk_euclidean_e3m2_serial);
    run_dense<e2m3_k, f32_k>(env, "angular_e2m3_serial", nk_angular_e2m3_serial);
    run_dense<e2m3_k, f32_k>(env, "sqeuclidean_e2m3_serial", nk_sqeuclidean_e2m3_serial);
    run_dense<e2m3_k, f32_k>(env, "euclidean_e2m3_serial", nk_euclidean_e2m3_serial);
    run_dense<i8_k, f32_k>(env, "angular_i8_serial", nk_angular_i8_serial);
    run_dense<i8_k, u32_k>(env, "sqeuclidean_i8_serial", nk_sqeuclidean_i8_serial);
    run_dense<i8_k, f32_k>(env, "euclidean_i8_serial", nk_euclidean_i8_serial);
    run_dense<u8_k, f32_k>(env, "angular_u8_serial", nk_angular_u8_serial);
    run_dense<u8_k, u32_k>(env, "sqeuclidean_u8_serial", nk_sqeuclidean_u8_serial);
    run_dense<u8_k, f32_k>(env, "euclidean_u8_serial", nk_euclidean_u8_serial);
    run_dense<i4_k, f32_k>(env, "angular_i4_serial", nk_angular_i4_serial);
    run_dense<i4_k, u32_k>(env, "sqeuclidean_i4_serial", nk_sqeuclidean_i4_serial);
    run_dense<i4_k, f32_k>(env, "euclidean_i4_serial", nk_euclidean_i4_serial);
    run_dense<u4_k, f32_k>(env, "angular_u4_serial", nk_angular_u4_serial);
    run_dense<u4_k, u32_k>(env, "sqeuclidean_u4_serial", nk_sqeuclidean_u4_serial);
    run_dense<u4_k, f32_k>(env, "euclidean_u4_serial", nk_euclidean_u4_serial);
}

} // namespace ashvardanian::numkong::bench
