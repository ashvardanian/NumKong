/**
 *  @brief RMSD, Kabsch, and Umeyama alignment tests.
 *  @file test/test_mesh.cpp
 *  @author Ash Vardanian
 *  @date December 28, 2025
 */

#include "test.hpp"
#include "numkong/mesh.hpp" // `nk::rmsd`

/**
 *  @brief Test RMSD kernel.
 */
template <typename scalar_type_>
error_stats_t test_rmsd(typename scalar_type_::mesh_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using transform_t = typename scalar_t::mesh_transform_t;
    using metric_t = typename scalar_t::mesh_metric_t;
    using reference_t = reference_for<scalar_t>;

    error_stats_t stats(comparison_family_t::mixed_precision_reduction_k);
    std::mt19937 generator(global_config.seed);

    std::size_t n = global_config.mesh_points;
    auto a = make_vector<scalar_t>(n * 3), b = make_vector<scalar_t>(n * 3);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        transform_t a_centroid[3], b_centroid[3], rot[9], scale;
        metric_t result;
        kernel(a.raw_values_data(), b.raw_values_data(), n, &a_centroid[0].raw_, &b_centroid[0].raw_, &rot[0].raw_,
               &scale.raw_, &result.raw_);
        reference_t a_centroid_ref[3], b_centroid_ref[3], rot_ref[9], scale_ref, reference;
        nk::rmsd<scalar_t, reference_t, reference_t, nk::no_simd_k>(a.values_data(), b.values_data(), n, a_centroid_ref,
                                                                    b_centroid_ref, rot_ref, &scale_ref, &reference);

        stats.accumulate(result, reference);
    }
    return stats;
}

/**
 *  @brief Test Kabsch alignment kernel.
 */
template <typename scalar_type_>
error_stats_t test_kabsch(typename scalar_type_::mesh_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using transform_t = typename scalar_t::mesh_transform_t;
    using metric_t = typename scalar_t::mesh_metric_t;
    using reference_t = reference_for<scalar_t>;

    error_stats_t stats(comparison_family_t::mixed_precision_reduction_k);
    std::mt19937 generator(global_config.seed);

    std::size_t n = global_config.mesh_points;
    auto a = make_vector<scalar_t>(n * 3), b = make_vector<scalar_t>(n * 3);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        transform_t a_centroid[3], b_centroid[3], rot[9], scale;
        metric_t result;
        kernel(a.raw_values_data(), b.raw_values_data(), n, &a_centroid[0].raw_, &b_centroid[0].raw_, &rot[0].raw_,
               &scale.raw_, &result.raw_);
        reference_t a_centroid_ref[3], b_centroid_ref[3], rot_ref[9], scale_ref, reference;
        nk::kabsch<scalar_t, reference_t, reference_t, nk::no_simd_k>(
            a.values_data(), b.values_data(), n, a_centroid_ref, b_centroid_ref, rot_ref, &scale_ref, &reference);

        stats.accumulate(result, reference);
    }
    return stats;
}

/**
 *  @brief Test Umeyama alignment kernel.
 */
template <typename scalar_type_>
error_stats_t test_umeyama(typename scalar_type_::mesh_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using transform_t = typename scalar_t::mesh_transform_t;
    using metric_t = typename scalar_t::mesh_metric_t;
    using reference_t = reference_for<scalar_t>;

    error_stats_t stats(comparison_family_t::mixed_precision_reduction_k);
    std::mt19937 generator(global_config.seed);

    std::size_t n = global_config.mesh_points;
    auto a = make_vector<scalar_t>(n * 3), b = make_vector<scalar_t>(n * 3);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        transform_t a_centroid[3], b_centroid[3], rot[9], scale;
        metric_t result;
        kernel(a.raw_values_data(), b.raw_values_data(), n, &a_centroid[0].raw_, &b_centroid[0].raw_, &rot[0].raw_,
               &scale.raw_, &result.raw_);
        reference_t a_centroid_ref[3], b_centroid_ref[3], rot_ref[9], scale_ref, reference;
        nk::umeyama<scalar_t, reference_t, reference_t, nk::no_simd_k>(
            a.values_data(), b.values_data(), n, a_centroid_ref, b_centroid_ref, rot_ref, &scale_ref, &reference);

        stats.accumulate(result, reference);
    }
    return stats;
}

void test_mesh() {
    error_stats_section_t check("Mesh Operations");

#if NK_DYNAMIC_DISPATCH
    check("rmsd_f64", test_rmsd<f64_t>, nk_rmsd_f64);
    check("rmsd_f32", test_rmsd<f32_t>, nk_rmsd_f32);
    check("kabsch_f64", test_kabsch<f64_t>, nk_kabsch_f64);
    check("kabsch_f32", test_kabsch<f32_t>, nk_kabsch_f32);
    check("umeyama_f64", test_umeyama<f64_t>, nk_umeyama_f64);
    check("umeyama_f32", test_umeyama<f32_t>, nk_umeyama_f32);
#else

#if NK_TARGET_NEON
    check("rmsd_f64_neon", test_rmsd<f64_t>, nk_rmsd_f64_neon);
    check("rmsd_f32_neon", test_rmsd<f32_t>, nk_rmsd_f32_neon);
    check("kabsch_f64_neon", test_kabsch<f64_t>, nk_kabsch_f64_neon);
    check("kabsch_f32_neon", test_kabsch<f32_t>, nk_kabsch_f32_neon);
    check("umeyama_f64_neon", test_umeyama<f64_t>, nk_umeyama_f64_neon);
    check("umeyama_f32_neon", test_umeyama<f32_t>, nk_umeyama_f32_neon);
    check("rmsd_f16_neon", test_rmsd<f16_t>, nk_rmsd_f16_neon);
    check("kabsch_f16_neon", test_kabsch<f16_t>, nk_kabsch_f16_neon);
    check("umeyama_f16_neon", test_umeyama<f16_t>, nk_umeyama_f16_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_NEONBFDOT
    check("rmsd_bf16_neonbfdot", test_rmsd<bf16_t>, nk_rmsd_bf16_neonbfdot);
    check("kabsch_bf16_neonbfdot", test_kabsch<bf16_t>, nk_kabsch_bf16_neonbfdot);
    check("umeyama_bf16_neonbfdot", test_umeyama<bf16_t>, nk_umeyama_bf16_neonbfdot);
#endif // NK_TARGET_NEONBFDOT

#if NK_TARGET_NEONFHM
    check("rmsd_f16_neonfhm", test_rmsd<f16_t>, nk_rmsd_f16_neonfhm);
    check("kabsch_f16_neonfhm", test_kabsch<f16_t>, nk_kabsch_f16_neonfhm);
    check("umeyama_f16_neonfhm", test_umeyama<f16_t>, nk_umeyama_f16_neonfhm);
#endif // NK_TARGET_NEONFHM

#if NK_TARGET_HASWELL
    check("rmsd_f64_haswell", test_rmsd<f64_t>, nk_rmsd_f64_haswell);
    check("rmsd_f32_haswell", test_rmsd<f32_t>, nk_rmsd_f32_haswell);
    check("kabsch_f64_haswell", test_kabsch<f64_t>, nk_kabsch_f64_haswell);
    check("kabsch_f32_haswell", test_kabsch<f32_t>, nk_kabsch_f32_haswell);
    check("umeyama_f64_haswell", test_umeyama<f64_t>, nk_umeyama_f64_haswell);
    check("umeyama_f32_haswell", test_umeyama<f32_t>, nk_umeyama_f32_haswell);
    check("rmsd_f16_haswell", test_rmsd<f16_t>, nk_rmsd_f16_haswell);
    check("kabsch_f16_haswell", test_kabsch<f16_t>, nk_kabsch_f16_haswell);
    check("umeyama_f16_haswell", test_umeyama<f16_t>, nk_umeyama_f16_haswell);
    check("rmsd_bf16_haswell", test_rmsd<bf16_t>, nk_rmsd_bf16_haswell);
    check("kabsch_bf16_haswell", test_kabsch<bf16_t>, nk_kabsch_bf16_haswell);
    check("umeyama_bf16_haswell", test_umeyama<bf16_t>, nk_umeyama_bf16_haswell);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
    check("rmsd_f64_skylake", test_rmsd<f64_t>, nk_rmsd_f64_skylake);
    check("rmsd_f32_skylake", test_rmsd<f32_t>, nk_rmsd_f32_skylake);
    check("kabsch_f64_skylake", test_kabsch<f64_t>, nk_kabsch_f64_skylake);
    check("kabsch_f32_skylake", test_kabsch<f32_t>, nk_kabsch_f32_skylake);
    check("umeyama_f64_skylake", test_umeyama<f64_t>, nk_umeyama_f64_skylake);
    check("umeyama_f32_skylake", test_umeyama<f32_t>, nk_umeyama_f32_skylake);
    check("rmsd_f16_skylake", test_rmsd<f16_t>, nk_rmsd_f16_skylake);
    check("rmsd_bf16_skylake", test_rmsd<bf16_t>, nk_rmsd_bf16_skylake);
    check("kabsch_f16_skylake", test_kabsch<f16_t>, nk_kabsch_f16_skylake);
    check("kabsch_bf16_skylake", test_kabsch<bf16_t>, nk_kabsch_bf16_skylake);
    check("umeyama_f16_skylake", test_umeyama<f16_t>, nk_umeyama_f16_skylake);
    check("umeyama_bf16_skylake", test_umeyama<bf16_t>, nk_umeyama_bf16_skylake);
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_GENOA
    check("rmsd_bf16_genoa", test_rmsd<bf16_t>, nk_rmsd_bf16_genoa);
    check("kabsch_bf16_genoa", test_kabsch<bf16_t>, nk_kabsch_bf16_genoa);
    check("umeyama_bf16_genoa", test_umeyama<bf16_t>, nk_umeyama_bf16_genoa);
#endif // NK_TARGET_GENOA

#if NK_TARGET_RVV
    check("rmsd_f64_rvv", test_rmsd<f64_t>, nk_rmsd_f64_rvv);
    check("rmsd_f32_rvv", test_rmsd<f32_t>, nk_rmsd_f32_rvv);
    check("rmsd_f16_rvv", test_rmsd<f16_t>, nk_rmsd_f16_rvv);
    check("rmsd_bf16_rvv", test_rmsd<bf16_t>, nk_rmsd_bf16_rvv);
    check("kabsch_f64_rvv", test_kabsch<f64_t>, nk_kabsch_f64_rvv);
    check("kabsch_f32_rvv", test_kabsch<f32_t>, nk_kabsch_f32_rvv);
    check("kabsch_f16_rvv", test_kabsch<f16_t>, nk_kabsch_f16_rvv);
    check("kabsch_bf16_rvv", test_kabsch<bf16_t>, nk_kabsch_bf16_rvv);
    check("umeyama_f64_rvv", test_umeyama<f64_t>, nk_umeyama_f64_rvv);
    check("umeyama_f32_rvv", test_umeyama<f32_t>, nk_umeyama_f32_rvv);
    check("umeyama_f16_rvv", test_umeyama<f16_t>, nk_umeyama_f16_rvv);
    check("umeyama_bf16_rvv", test_umeyama<bf16_t>, nk_umeyama_bf16_rvv);
#endif // NK_TARGET_RVV

#if NK_TARGET_V128RELAXED
    check("rmsd_f32_v128relaxed", test_rmsd<f32_t>, nk_rmsd_f32_v128relaxed);
    check("rmsd_f64_v128relaxed", test_rmsd<f64_t>, nk_rmsd_f64_v128relaxed);
    check("kabsch_f32_v128relaxed", test_kabsch<f32_t>, nk_kabsch_f32_v128relaxed);
    check("kabsch_f64_v128relaxed", test_kabsch<f64_t>, nk_kabsch_f64_v128relaxed);
    check("umeyama_f32_v128relaxed", test_umeyama<f32_t>, nk_umeyama_f32_v128relaxed);
    check("umeyama_f64_v128relaxed", test_umeyama<f64_t>, nk_umeyama_f64_v128relaxed);
#endif // NK_TARGET_V128RELAXED

    // Serial always runs - baseline test
    check("rmsd_f64_serial", test_rmsd<f64_t>, nk_rmsd_f64_serial);
    check("rmsd_f32_serial", test_rmsd<f32_t>, nk_rmsd_f32_serial);
    check("kabsch_f64_serial", test_kabsch<f64_t>, nk_kabsch_f64_serial);
    check("kabsch_f32_serial", test_kabsch<f32_t>, nk_kabsch_f32_serial);
    check("umeyama_f64_serial", test_umeyama<f64_t>, nk_umeyama_f64_serial);
    check("umeyama_f32_serial", test_umeyama<f32_t>, nk_umeyama_f32_serial);

#endif // NK_DYNAMIC_DISPATCH
}
