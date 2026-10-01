/**
 *  @file test/mesh.cpp
 *  @author Ash Vardanian
 *  @date December 28, 2025
 *  @brief RMSD, Kabsch, and Umeyama alignment tests.
 */

#include "harness.hpp"
#include "numkong/mesh.hpp" // `nk::rmsd`

namespace ashvardanian::numkong::test {

/** Half-precision kernels are judged by @ref mesh_error_bound, the rest in ULPs. */
template <typename scalar_type_>
constexpr bool mesh_bounded = scalar_type_::dtype() == nk_f16_k || scalar_type_::dtype() == nk_bf16_k;

/**
 *  @brief Bounds the RMSD error of a half-precision mesh kernel by the arithmetic it performs.
 *
 *  Shifting each point by its cloud's first one and rounding it back to the input type moves it by
 *  up to that type's unit roundoff, and so the RMSD by that share of @p shift_spread, the clouds'
 *  RMS distance from their pivots. Summing the moments in f32 adds up to n roundings of their
 *  total, n · @p term_spread², which the square root turns into its own root or its ratio to the
 *  RMSD. Clouds far from the origin summed without the shift would cancel well past it.
 */
template <typename scalar_type_>
nk_f64_t mesh_error_bound(std::size_t n, nk_f64_t rmsd, nk_f64_t shift_spread, nk_f64_t term_spread) noexcept {
    nk_f64_t const input_roundoff = std::ldexp(1.0, -static_cast<int>(scalar_type_::component_t::mantissa_bits()) - 1);
    nk_f64_t const ssd_error = n * nk_accumulation_error_bound(nk_f32_k) * term_spread * term_spread;
    nk_f64_t const root_error = rmsd > 0 ? std::min(std::sqrt(ssd_error), ssd_error / rmsd) : std::sqrt(ssd_error);
    return input_roundoff * shift_spread + root_error;
}

/** RMS distance of a cloud's points from its first one, the spread its shifted moments see. */
template <typename scalar_type_>
nk_f64_t pivot_spread(scalar_type_ const *points, std::size_t n) noexcept {
    nk_f64_t sum = 0;
    for (std::size_t i = 0; i != n * 3; ++i) {
        nk_f64_t const delta = static_cast<nk_f64_t>(points[i]) - static_cast<nk_f64_t>(points[i % 3]);
        sum += delta * delta;
    }
    return n ? std::sqrt(sum / n) : 0;
}

/** Moves every coordinate by @p offset, far from the origin relative to the cloud's spread. */
template <typename vector_type_>
void offset_points(vector_type_ &points, nk_f64_t offset) noexcept {
    using value_t = std::remove_reference_t<decltype(points.values_data()[0])>;
    for (std::size_t i = 0; i != points.size_values(); ++i)
        points.values_data()[i] = value_t(static_cast<nk_f64_t>(points.values_data()[i]) + offset);
}

/** Test RMSD kernel. */
template <typename scalar_type_>
error_stats_t test_rmsd(settings_t const &settings, typename scalar_type_::mesh_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using transform_t = typename scalar_t::mesh_transform_t;
    using metric_t = typename scalar_t::mesh_metric_t;
    using reference_t = reference_for<scalar_t>;

    error_stats_t stats(mesh_bounded<scalar_t> ? comparison_family_t::bounded_k : comparison_family_t::approximate_k);
    std::mt19937 generator(settings.seed.value);

    std::size_t n = settings.mesh_points;
    auto a = make_vector<scalar_t>(n * 3), b = make_vector<scalar_t>(n * 3);

    // Degenerate empty cloud (n == 0): the kernel must match the serial oracle's neutral result
    // (0) instead of a 1/n-induced NaN/Inf — the rmsd/kabsch/umeyama empty-input hazard.
    {
        transform_t a_centroid[3], b_centroid[3], rot[9], scale;
        metric_t result;
        stats.expect(kernel(a.raw_values_data(), b.raw_values_data(), 0, &a_centroid[0].raw_, &b_centroid[0].raw_,
                            &rot[0].raw_, &scale.raw_, &result.raw_, nullptr));
        reference_t a_centroid_ref[3], b_centroid_ref[3], rot_ref[9], scale_ref, reference;
        stats.expect(nk::rmsd<scalar_t, reference_t, reference_t>(a.values_data(), b.values_data(), 0, a_centroid_ref,
                                                                  b_centroid_ref, rot_ref, &scale_ref, &reference,
                                                                  no_tiers_k));
        stats.accumulate(result, reference);
    }

    bool far_from_origin = false;
    for (time_point_t const deadline = steady_clock_t::now() + settings.time_limit_per_kernel;
         steady_clock_t::now() < deadline;) {
        fill_random(settings, generator, a);
        fill_random(settings, generator, b);
        if ((far_from_origin = !far_from_origin)) offset_points(a, 64), offset_points(b, 64);

        transform_t a_centroid[3], b_centroid[3], rot[9], scale;
        metric_t result;
        stats.expect(kernel(a.raw_values_data(), b.raw_values_data(), n, &a_centroid[0].raw_, &b_centroid[0].raw_,
                            &rot[0].raw_, &scale.raw_, &result.raw_, nullptr));
        reference_t a_centroid_ref[3], b_centroid_ref[3], rot_ref[9], scale_ref, reference;
        stats.expect(nk::rmsd<scalar_t, reference_t, reference_t>(a.values_data(), b.values_data(), n, a_centroid_ref,
                                                                  b_centroid_ref, rot_ref, &scale_ref, &reference,
                                                                  no_tiers_k));

        nk_f64_t const rmsd = static_cast<nk_f64_t>(reference);
        if constexpr (mesh_bounded<scalar_t>)
            stats.accumulate_bounded(result, reference, mesh_error_bound<scalar_t>(n, rmsd, 0, rmsd));
        else stats.accumulate(result, reference);
    }
    return stats;
}

/** Test Kabsch alignment kernel. */
template <typename scalar_type_>
error_stats_t test_kabsch(settings_t const &settings, typename scalar_type_::mesh_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using transform_t = typename scalar_t::mesh_transform_t;
    using metric_t = typename scalar_t::mesh_metric_t;
    using reference_t = reference_for<scalar_t>;

    error_stats_t stats(mesh_bounded<scalar_t> ? comparison_family_t::bounded_k : comparison_family_t::approximate_k);
    std::mt19937 generator(settings.seed.value);

    std::size_t n = settings.mesh_points;
    auto a = make_vector<scalar_t>(n * 3), b = make_vector<scalar_t>(n * 3);

    // Degenerate empty cloud (n == 0): the kernel must match the serial oracle's neutral result
    // (0) instead of a 1/n-induced NaN/Inf — the rmsd/kabsch/umeyama empty-input hazard.
    {
        transform_t a_centroid[3], b_centroid[3], rot[9], scale;
        metric_t result;
        stats.expect(kernel(a.raw_values_data(), b.raw_values_data(), 0, &a_centroid[0].raw_, &b_centroid[0].raw_,
                            &rot[0].raw_, &scale.raw_, &result.raw_, nullptr));
        reference_t a_centroid_ref[3], b_centroid_ref[3], rot_ref[9], scale_ref, reference;
        stats.expect(nk::kabsch<scalar_t, reference_t, reference_t>(a.values_data(), b.values_data(), 0, a_centroid_ref,
                                                                    b_centroid_ref, rot_ref, &scale_ref, &reference,
                                                                    no_tiers_k));
        stats.accumulate(result, reference);
    }

    bool far_from_origin = false;
    for (time_point_t const deadline = steady_clock_t::now() + settings.time_limit_per_kernel;
         steady_clock_t::now() < deadline;) {
        fill_random(settings, generator, a);
        fill_random(settings, generator, b);
        if ((far_from_origin = !far_from_origin)) offset_points(a, 64), offset_points(b, 64);

        transform_t a_centroid[3], b_centroid[3], rot[9], scale;
        metric_t result;
        stats.expect(kernel(a.raw_values_data(), b.raw_values_data(), n, &a_centroid[0].raw_, &b_centroid[0].raw_,
                            &rot[0].raw_, &scale.raw_, &result.raw_, nullptr));
        reference_t a_centroid_ref[3], b_centroid_ref[3], rot_ref[9], scale_ref, reference;
        stats.expect(nk::kabsch<scalar_t, reference_t, reference_t>(a.values_data(), b.values_data(), n, a_centroid_ref,
                                                                    b_centroid_ref, rot_ref, &scale_ref, &reference,
                                                                    no_tiers_k));

        if constexpr (mesh_bounded<scalar_t>) {
            nk_f64_t const spread = pivot_spread(a.values_data(), n) + pivot_spread(b.values_data(), n);
            stats.accumulate_bounded(result, reference,
                                     mesh_error_bound<scalar_t>(n, static_cast<nk_f64_t>(reference), spread, spread));
        }
        else stats.accumulate(result, reference);
    }
    return stats;
}

/** Test Umeyama alignment kernel. */
template <typename scalar_type_>
error_stats_t test_umeyama(settings_t const &settings, typename scalar_type_::mesh_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using transform_t = typename scalar_t::mesh_transform_t;
    using metric_t = typename scalar_t::mesh_metric_t;
    using reference_t = reference_for<scalar_t>;

    error_stats_t stats(mesh_bounded<scalar_t> ? comparison_family_t::bounded_k : comparison_family_t::approximate_k);
    std::mt19937 generator(settings.seed.value);

    std::size_t n = settings.mesh_points;
    auto a = make_vector<scalar_t>(n * 3), b = make_vector<scalar_t>(n * 3);

    // Degenerate empty cloud (n == 0): the kernel must match the serial oracle's neutral result
    // (0) instead of a 1/n-induced NaN/Inf — the rmsd/kabsch/umeyama empty-input hazard.
    {
        transform_t a_centroid[3], b_centroid[3], rot[9], scale;
        metric_t result;
        stats.expect(kernel(a.raw_values_data(), b.raw_values_data(), 0, &a_centroid[0].raw_, &b_centroid[0].raw_,
                            &rot[0].raw_, &scale.raw_, &result.raw_, nullptr));
        reference_t a_centroid_ref[3], b_centroid_ref[3], rot_ref[9], scale_ref, reference;
        stats.expect(nk::umeyama<scalar_t, reference_t, reference_t>(a.values_data(), b.values_data(), 0,
                                                                     a_centroid_ref, b_centroid_ref, rot_ref,
                                                                     &scale_ref, &reference, no_tiers_k));
        stats.accumulate(result, reference);
    }

    bool far_from_origin = false;
    for (time_point_t const deadline = steady_clock_t::now() + settings.time_limit_per_kernel;
         steady_clock_t::now() < deadline;) {
        fill_random(settings, generator, a);
        fill_random(settings, generator, b);
        if ((far_from_origin = !far_from_origin)) offset_points(a, 64), offset_points(b, 64);

        transform_t a_centroid[3], b_centroid[3], rot[9], scale;
        metric_t result;
        stats.expect(kernel(a.raw_values_data(), b.raw_values_data(), n, &a_centroid[0].raw_, &b_centroid[0].raw_,
                            &rot[0].raw_, &scale.raw_, &result.raw_, nullptr));
        reference_t a_centroid_ref[3], b_centroid_ref[3], rot_ref[9], scale_ref, reference;
        stats.expect(nk::umeyama<scalar_t, reference_t, reference_t>(a.values_data(), b.values_data(), n,
                                                                     a_centroid_ref, b_centroid_ref, rot_ref,
                                                                     &scale_ref, &reference, no_tiers_k));

        if constexpr (mesh_bounded<scalar_t>) {
            nk_f64_t const spread = static_cast<nk_f64_t>(scale_ref) * pivot_spread(a.values_data(), n) +
                                    pivot_spread(b.values_data(), n);
            stats.accumulate_bounded(result, reference,
                                     mesh_error_bound<scalar_t>(n, static_cast<nk_f64_t>(reference), spread, spread));
        }
        else stats.accumulate(result, reference);
    }
    return stats;
}

void test_mesh(error_stats_section_t &check) {

    check.section("Mesh Operations Serial", nk_cap_serial_k);
    check("rmsd_f64_serial", test_rmsd<f64_t>, nk_rmsd_f64_serial);
    check("rmsd_f32_serial", test_rmsd<f32_t>, nk_rmsd_f32_serial);
    check("kabsch_f64_serial", test_kabsch<f64_t>, nk_kabsch_f64_serial);
    check("kabsch_f32_serial", test_kabsch<f32_t>, nk_kabsch_f32_serial);
    check("umeyama_f64_serial", test_umeyama<f64_t>, nk_umeyama_f64_serial);
    check("umeyama_f32_serial", test_umeyama<f32_t>, nk_umeyama_f32_serial);

#if !NUMKONG_HEADER_ONLY
    check.section("Mesh Operations Runtime Dispatch", nk_cap_serial_k);
    check("rmsd_f64", test_rmsd<f64_t>, cpu_best<nk_rmsd_f64_best>);
    check("rmsd_f32", test_rmsd<f32_t>, cpu_best<nk_rmsd_f32_best>);
    check("kabsch_f64", test_kabsch<f64_t>, cpu_best<nk_kabsch_f64_best>);
    check("kabsch_f32", test_kabsch<f32_t>, cpu_best<nk_kabsch_f32_best>);
    check("umeyama_f64", test_umeyama<f64_t>, cpu_best<nk_umeyama_f64_best>);
    check("umeyama_f32", test_umeyama<f32_t>, cpu_best<nk_umeyama_f32_best>);
#endif

#if NUMKONG_TARGET_NEON
    check.section("Mesh Operations NEON", nk_cap_neon_k);
    check("rmsd_f64_neon", test_rmsd<f64_t>, nk_rmsd_f64_neon);
    check("rmsd_f32_neon", test_rmsd<f32_t>, nk_rmsd_f32_neon);
    check("kabsch_f64_neon", test_kabsch<f64_t>, nk_kabsch_f64_neon);
    check("kabsch_f32_neon", test_kabsch<f32_t>, nk_kabsch_f32_neon);
    check("umeyama_f64_neon", test_umeyama<f64_t>, nk_umeyama_f64_neon);
    check("umeyama_f32_neon", test_umeyama<f32_t>, nk_umeyama_f32_neon);
    check("rmsd_f16_neon", test_rmsd<f16_t>, nk_rmsd_f16_neon);
    check("kabsch_f16_neon", test_kabsch<f16_t>, nk_kabsch_f16_neon);
    check("umeyama_f16_neon", test_umeyama<f16_t>, nk_umeyama_f16_neon);
#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_NEONBFDOT
    check.section("Mesh Operations NEON BF16", nk_cap_neonbfdot_k);
    check("rmsd_bf16_neonbfdot", test_rmsd<bf16_t>, nk_rmsd_bf16_neonbfdot);
    check("kabsch_bf16_neonbfdot", test_kabsch<bf16_t>, nk_kabsch_bf16_neonbfdot);
    check("umeyama_bf16_neonbfdot", test_umeyama<bf16_t>, nk_umeyama_bf16_neonbfdot);
#endif // NUMKONG_TARGET_NEONBFDOT

#if NUMKONG_TARGET_NEONFHM
    check.section("Mesh Operations NEON FHM", nk_cap_neonfhm_k);
    check("rmsd_f16_neonfhm", test_rmsd<f16_t>, nk_rmsd_f16_neonfhm);
    check("kabsch_f16_neonfhm", test_kabsch<f16_t>, nk_kabsch_f16_neonfhm);
    check("umeyama_f16_neonfhm", test_umeyama<f16_t>, nk_umeyama_f16_neonfhm);
#endif // NUMKONG_TARGET_NEONFHM

#if NUMKONG_TARGET_HASWELL
    check.section("Mesh Operations Haswell", nk_cap_haswell_k);
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
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SKYLAKE
    check.section("Mesh Operations Skylake", nk_cap_skylake_k);
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
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_GENOA
    check.section("Mesh Operations Genoa", nk_cap_genoa_k);
    check("rmsd_bf16_genoa", test_rmsd<bf16_t>, nk_rmsd_bf16_genoa);
    check("kabsch_bf16_genoa", test_kabsch<bf16_t>, nk_kabsch_bf16_genoa);
    check("umeyama_bf16_genoa", test_umeyama<bf16_t>, nk_umeyama_bf16_genoa);
#endif // NUMKONG_TARGET_GENOA

#if NUMKONG_TARGET_RVV
    check.section("Mesh Operations RVV", nk_cap_rvv_k);
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
#endif // NUMKONG_TARGET_RVV

#if NUMKONG_TARGET_V128RELAXED
    check.section("Mesh Operations V128 Relaxed", nk_cap_v128relaxed_k);
    check("rmsd_f32_v128relaxed", test_rmsd<f32_t>, nk_rmsd_f32_v128relaxed);
    check("rmsd_f64_v128relaxed", test_rmsd<f64_t>, nk_rmsd_f64_v128relaxed);
    check("kabsch_f32_v128relaxed", test_kabsch<f32_t>, nk_kabsch_f32_v128relaxed);
    check("kabsch_f64_v128relaxed", test_kabsch<f64_t>, nk_kabsch_f64_v128relaxed);
    check("umeyama_f32_v128relaxed", test_umeyama<f32_t>, nk_umeyama_f32_v128relaxed);
    check("umeyama_f64_v128relaxed", test_umeyama<f64_t>, nk_umeyama_f64_v128relaxed);
#endif // NUMKONG_TARGET_V128RELAXED
}

} // namespace ashvardanian::numkong::test
