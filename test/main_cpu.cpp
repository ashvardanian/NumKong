/**
 *  @file test/main_cpu.cpp
 *  @author Ash Vardanian
 *  @date December 28, 2025
 *  @brief Test suite entry point and configuration.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#if defined(_MSC_VER)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "numkong/capabilities.h" // `nk_capabilities_detected_cpu`, `nk_configure_thread_cpu`
#include "numkong/devices.hpp"    // `nk::device_t`, `nk::default_capabilities`
#include "numkong/random.hpp"     // `nk::fill_uniform`

#include "harness.hpp"

using namespace ashvardanian::numkong::test;

/*  Explicit instantiations verify that `random.hpp` compiles for every code path: f64_t for the
 *  scalar float path, i16_t for the scalar signed integer path, bf16c_t for the complex path, and
 *  i4x2_t for the packed sub-byte path. */

/** Compiles the bounded @c fill_uniform overload for f64, i16, bf16c and i4x2 through implicit
 *  instantiation. */
inline void compile_bounded_fill_uniform_paths(split_mix_generator_t &generator) noexcept {
    nk::fill_uniform(generator, static_cast<nk::f64_t *>(nullptr), 0, nk::f64_t::component_t {},
                     nk::f64_t::component_t {});
    nk::fill_uniform(generator, static_cast<nk::i16_t *>(nullptr), 0, nk::i16_t::component_t {},
                     nk::i16_t::component_t {});
    nk::fill_uniform(generator, static_cast<nk::bf16c_t *>(nullptr), 0, nk::bf16c_t::component_t {},
                     nk::bf16c_t::component_t {});
    nk::fill_uniform(generator, static_cast<nk::i4x2_t *>(nullptr), 0, nk::i4x2_t::component_t {},
                     nk::i4x2_t::component_t {});
}
template class nk::vector<int>;
template class nk::vector<nk::i32_t>;
template class nk::vector<nk::u1x8_t>;
template class nk::vector<nk::i4x2_t>;
template class nk::vector<nk::f64c_t>;
template class nk::vector<std::complex<float>>;
template void nk::fill_uniform(split_mix_generator_t &, nk::f64_t *, std::size_t);
template void nk::fill_uniform(split_mix_generator_t &, nk::i16_t *, std::size_t);
template void nk::fill_uniform(split_mix_generator_t &, nk::bf16c_t *, std::size_t);
template void nk::fill_uniform(split_mix_generator_t &, nk::i4x2_t *, std::size_t);
template void nk::fill_lognormal(split_mix_generator_t &, nk::f64_t *, std::size_t, double, double);
template void nk::fill_lognormal(split_mix_generator_t &, nk::i16_t *, std::size_t, double, double);
template void nk::fill_lognormal(split_mix_generator_t &, nk::bf16c_t *, std::size_t, double, double);
template void nk::fill_lognormal(split_mix_generator_t &, nk::i4x2_t *, std::size_t, double, double);
template void nk::fill_cauchy(split_mix_generator_t &, nk::f64_t *, std::size_t, double, double);
template void nk::fill_cauchy(split_mix_generator_t &, nk::i16_t *, std::size_t, double, double);
template void nk::fill_cauchy(split_mix_generator_t &, nk::bf16c_t *, std::size_t, double, double);
template void nk::fill_cauchy(split_mix_generator_t &, nk::i4x2_t *, std::size_t, double, double);

#pragma region Dispatch Points

/** One capability's own kernel, which a dispatch point must run under a mask of it alone. */
template <typename kernel_type_>
struct capability_kernel {
    nk_capability_t capability;
    kernel_type_ kernel;
};

/** The CPU capabilities this machine runs and this binary holds. */
static nk_capability_t runnable_capabilities() noexcept {
    return capabilities_detected_cpu() & capabilities_compiled_cpu();
}

/** @c nk_dot_f32_best over each runnable capability alone runs that capability's kernel, and a GPU
 *  mask finds no kernel at all. */
static error_stats_t test_best_dot_f32() {
    capability_kernel<f32_t::dot_kernel_t> const capabilities[] = {
        {nk_cap_serial_k, nk_dot_f32_serial},
#if NUMKONG_TARGET_NEON
        {nk_cap_neon_k, nk_dot_f32_neon},
#endif // NUMKONG_TARGET_NEON
#if NUMKONG_TARGET_SVE
        {nk_cap_sve_k, nk_dot_f32_sve},
#endif // NUMKONG_TARGET_SVE
#if NUMKONG_TARGET_HASWELL
        {nk_cap_haswell_k, nk_dot_f32_haswell},
#endif // NUMKONG_TARGET_HASWELL
#if NUMKONG_TARGET_SKYLAKE
        {nk_cap_skylake_k, nk_dot_f32_skylake},
#endif // NUMKONG_TARGET_SKYLAKE
#if NUMKONG_TARGET_RVV
        {nk_cap_rvv_k, nk_dot_f32_rvv},
#endif // NUMKONG_TARGET_RVV
#if NUMKONG_TARGET_V128RELAXED
        {nk_cap_v128relaxed_k, nk_dot_f32_v128relaxed},
#endif // NUMKONG_TARGET_V128RELAXED
#if NUMKONG_TARGET_POWERVSX
        {nk_cap_powervsx_k, nk_dot_f32_powervsx},
#endif // NUMKONG_TARGET_POWERVSX
#if NUMKONG_TARGET_LOONGSONASX
        {nk_cap_loongsonasx_k, nk_dot_f32_loongsonasx},
#endif // NUMKONG_TARGET_LOONGSONASX
    };
    error_stats_t stats(comparison_family_t::exact_k);
    split_mix_generator_t generator(case_stream_key());
    std::size_t const n = test_run.settings.dense_dimensions;
    auto a = make_vector<f32_t>(n), b = make_vector<f32_t>(n);
    fill_random(generator, a), fill_random(generator, b);
    for (auto const &[capability, kernel] : capabilities) {
        if (!(capability & runnable_capabilities())) continue;
        nk_f64_t expected = 0, dispatched = 0;
        stats.expect_status(kernel(a.raw_values_data(), b.raw_values_data(), n, &expected, nullptr));
        stats.expect_status(
            nk_dot_f32_best(a.raw_values_data(), b.raw_values_data(), n, &dispatched, capability, nullptr));
        stats.expect_true(dispatched == expected, "the dispatch point ran another capability's kernel");
    }
    nk_f64_t unused = 0;
    stats.expect_true(nk_dot_f32_best(a.raw_values_data(), b.raw_values_data(), n, &unused, nk_cap_cuda_k, nullptr) ==
                          nk_missing_kernel_k,
                      "a GPU mask found a kernel");
    return stats;
}

/** @c nk_dots_pack_bf16_best and @c nk_dots_packed_bf16_best over each runnable capability alone
 *  pack and multiply as its own kernels do, and a GPU mask finds none in a binary without one. */
static error_stats_t test_best_dots_packed_bf16() {
    struct capability_kernels_t {
        nk_capability_t capability;
        bf16_t::dots_pack_size_kernel_t pack_size;
        bf16_t::dots_pack_kernel_t pack;
        bf16_t::dots_packed_kernel_t packed;
    };
    capability_kernels_t const capabilities[] = {
        {nk_cap_serial_k, nk_dots_pack_size_bf16_serial, nk_dots_pack_bf16_serial, nk_dots_packed_bf16_serial},
#if NUMKONG_TARGET_NEON
        {nk_cap_neon_k, nk_dots_pack_size_bf16_neon, nk_dots_pack_bf16_neon, nk_dots_packed_bf16_neon},
#endif // NUMKONG_TARGET_NEON
#if NUMKONG_TARGET_NEONBFDOT
        {nk_cap_neonbfdot_k, nk_dots_pack_size_bf16_neonbfdot, nk_dots_pack_bf16_neonbfdot,
         nk_dots_packed_bf16_neonbfdot},
#endif // NUMKONG_TARGET_NEONBFDOT
#if NUMKONG_TARGET_SME
        {nk_cap_sme_k, nk_dots_pack_size_bf16_sme, nk_dots_pack_bf16_sme, nk_dots_packed_bf16_sme},
#endif // NUMKONG_TARGET_SME
#if NUMKONG_TARGET_HASWELL
        {nk_cap_haswell_k, nk_dots_pack_size_bf16_haswell, nk_dots_pack_bf16_haswell, nk_dots_packed_bf16_haswell},
#endif // NUMKONG_TARGET_HASWELL
#if NUMKONG_TARGET_SKYLAKE
        {nk_cap_skylake_k, nk_dots_pack_size_bf16_skylake, nk_dots_pack_bf16_skylake, nk_dots_packed_bf16_skylake},
#endif // NUMKONG_TARGET_SKYLAKE
#if NUMKONG_TARGET_GENOA
        {nk_cap_genoa_k, nk_dots_pack_size_bf16_genoa, nk_dots_pack_bf16_genoa, nk_dots_packed_bf16_genoa},
#endif // NUMKONG_TARGET_GENOA
#if NUMKONG_TARGET_SAPPHIREAMX
        {nk_cap_sapphireamx_k, nk_dots_pack_size_bf16_sapphireamx, nk_dots_pack_bf16_sapphireamx,
         nk_dots_packed_bf16_sapphireamx},
#endif // NUMKONG_TARGET_SAPPHIREAMX
#if NUMKONG_TARGET_RVV
        {nk_cap_rvv_k, nk_dots_pack_size_bf16_rvv, nk_dots_pack_bf16_rvv, nk_dots_packed_bf16_rvv},
#endif // NUMKONG_TARGET_RVV
#if NUMKONG_TARGET_V128
        {nk_cap_v128_k, nk_dots_pack_size_bf16_v128, nk_dots_pack_bf16_v128, nk_dots_packed_bf16_v128},
#endif // NUMKONG_TARGET_V128
#if NUMKONG_TARGET_V128RELAXED
        {nk_cap_v128relaxed_k, nk_dots_pack_size_bf16_v128relaxed, nk_dots_pack_bf16_v128relaxed,
         nk_dots_packed_bf16_v128relaxed},
#endif // NUMKONG_TARGET_V128RELAXED
#if NUMKONG_TARGET_POWERVSX
        {nk_cap_powervsx_k, nk_dots_pack_size_bf16_powervsx, nk_dots_pack_bf16_powervsx, nk_dots_packed_bf16_powervsx},
#endif // NUMKONG_TARGET_POWERVSX
#if NUMKONG_TARGET_LOONGSONASX
        {nk_cap_loongsonasx_k, nk_dots_pack_size_bf16_loongsonasx, nk_dots_pack_bf16_loongsonasx,
         nk_dots_packed_bf16_loongsonasx},
#endif // NUMKONG_TARGET_LOONGSONASX
#if NUMKONG_TARGET_POWERMMA
        {nk_cap_powermma_k, nk_dots_pack_size_bf16_powermma, nk_dots_pack_bf16_powermma, nk_dots_packed_bf16_powermma},
#endif // NUMKONG_TARGET_POWERMMA
    };
    error_stats_t stats(comparison_family_t::exact_k);
    split_mix_generator_t generator(case_stream_key());
    std::size_t const rows = 17, columns = 33, depth = 70, row_bytes = depth * sizeof(bf16_t);
    std::size_t const c_stride = columns * sizeof(nk_f32_t);
    auto a = make_vector<bf16_t>(rows * depth), b = make_vector<bf16_t>(columns * depth);
    auto expected = make_vector<f32_t>(rows * columns), dispatched = make_vector<f32_t>(rows * columns);
    fill_random(generator, a), fill_random(generator, b);
    for (auto const &[capability, pack_size, pack, packed] : capabilities) {
        if (!(capability & runnable_capabilities())) continue;
        nk_size_t expected_bytes = 0, dispatched_bytes = 0;
        stats.expect_status(pack_size(columns, depth, &expected_bytes));
        stats.expect_status(nk_dots_pack_size_bf16_best(columns, depth, capability, &dispatched_bytes));
        stats.expect_true(dispatched_bytes == expected_bytes, "the dispatch point sized another capability's pack");
        auto expected_pack = make_vector<std::byte>(expected_bytes),
             dispatched_pack = make_vector<std::byte>(expected_bytes);
        stats.expect_status(
            pack(b.raw_values_data(), columns, depth, row_bytes, expected_pack.raw_values_data(), 0, columns, nullptr));
        stats.expect_status(nk_dots_pack_bf16_best(b.raw_values_data(), columns, depth, row_bytes,
                                                   dispatched_pack.raw_values_data(), 0, columns, capability, nullptr));
        stats.expect_status(packed(a.raw_values_data(), expected_pack.raw_values_data(), expected.raw_values_data(),
                                   rows, columns, depth, row_bytes, c_stride, nullptr, 0, nullptr));
        stats.expect_status(nk_dots_packed_bf16_best(a.raw_values_data(), dispatched_pack.raw_values_data(),
                                                     dispatched.raw_values_data(), rows, columns, depth, row_bytes,
                                                     c_stride, nullptr, 0, capability, nullptr));
        stats.expect_true(
            std::memcmp(expected.raw_values_data(), dispatched.raw_values_data(), expected.size_bytes()) == 0,
            "the dispatch point ran another capability's kernel");
    }
    nk_kernel_punned_t cuda_kernel = nullptr;
    nk_capability_t cuda_capability = 0;
    if (nk_find_kernel_punned(nk_kernel_dots_packed_k, nk_bf16_k, nk_cap_cuda_k, &cuda_kernel, &cuda_capability) !=
        nk_success_k)
        stats.expect_true(nk_dots_packed_bf16_best(a.raw_values_data(), b.raw_values_data(), expected.raw_values_data(),
                                                   rows, columns, depth, row_bytes, c_stride, nullptr, 0, nk_cap_cuda_k,
                                                   nullptr) == nk_missing_kernel_k,
                          "a GPU mask found a kernel in a binary without one");
    return stats;
}

/** @c nk_reduce_moments_f32_best over each runnable capability alone runs its own kernel. */
static error_stats_t test_best_reduce_moments_f32() {
    capability_kernel<f32_t::reduce_moments_kernel_t> const capabilities[] = {
        {nk_cap_serial_k, nk_reduce_moments_f32_serial},
#if NUMKONG_TARGET_NEON
        {nk_cap_neon_k, nk_reduce_moments_f32_neon},
#endif // NUMKONG_TARGET_NEON
#if NUMKONG_TARGET_HASWELL
        {nk_cap_haswell_k, nk_reduce_moments_f32_haswell},
#endif // NUMKONG_TARGET_HASWELL
#if NUMKONG_TARGET_SKYLAKE
        {nk_cap_skylake_k, nk_reduce_moments_f32_skylake},
#endif // NUMKONG_TARGET_SKYLAKE
#if NUMKONG_TARGET_RVV
        {nk_cap_rvv_k, nk_reduce_moments_f32_rvv},
#endif // NUMKONG_TARGET_RVV
#if NUMKONG_TARGET_V128RELAXED
        {nk_cap_v128relaxed_k, nk_reduce_moments_f32_v128relaxed},
#endif // NUMKONG_TARGET_V128RELAXED
    };
    error_stats_t stats(comparison_family_t::exact_k);
    split_mix_generator_t generator(case_stream_key());
    std::size_t const n = test_run.settings.dense_dimensions;
    auto data = make_vector<f32_t>(n);
    fill_random(generator, data);
    for (auto const &[capability, kernel] : capabilities) {
        if (!(capability & runnable_capabilities())) continue;
        nk_f64_t expected[2] = {0, 0}, dispatched[2] = {0, 0};
        stats.expect_status(kernel(data.raw_values_data(), n, sizeof(nk_f32_t), &expected[0], &expected[1], nullptr));
        stats.expect_status(nk_reduce_moments_f32_best(data.raw_values_data(), n, sizeof(nk_f32_t), &dispatched[0],
                                                       &dispatched[1], capability, nullptr));
        stats.expect_true(dispatched[0] == expected[0] && dispatched[1] == expected[1],
                          "the dispatch point ran another capability's kernel");
    }
    return stats;
}

/** @c nk_cast_best over each runnable capability alone runs that capability's kernel. */
static error_stats_t test_best_cast() {
    using cast_kernel_t = nk_status_t (*)(void const *, nk_dtype_t, void *, nk_dtype_t, nk_size_t, nk_stream_t);
    capability_kernel<cast_kernel_t> const capabilities[] = {
        {nk_cap_serial_k, nk_cast_serial},
#if NUMKONG_TARGET_NEON
        {nk_cap_neon_k, nk_cast_neon},
#endif // NUMKONG_TARGET_NEON
#if NUMKONG_TARGET_HASWELL
        {nk_cap_haswell_k, nk_cast_haswell},
#endif // NUMKONG_TARGET_HASWELL
#if NUMKONG_TARGET_SKYLAKE
        {nk_cap_skylake_k, nk_cast_skylake},
#endif // NUMKONG_TARGET_SKYLAKE
#if NUMKONG_TARGET_ICELAKE
        {nk_cap_icelake_k, nk_cast_icelake},
#endif // NUMKONG_TARGET_ICELAKE
#if NUMKONG_TARGET_SAPPHIRE
        {nk_cap_sapphire_k, nk_cast_sapphire},
#endif // NUMKONG_TARGET_SAPPHIRE
#if NUMKONG_TARGET_RVV
        {nk_cap_rvv_k, nk_cast_rvv},
#endif // NUMKONG_TARGET_RVV
#if NUMKONG_TARGET_V128RELAXED
        {nk_cap_v128relaxed_k, nk_cast_v128relaxed},
#endif // NUMKONG_TARGET_V128RELAXED
#if NUMKONG_TARGET_POWERVSX
        {nk_cap_powervsx_k, nk_cast_powervsx},
#endif // NUMKONG_TARGET_POWERVSX
    };
    error_stats_t stats(comparison_family_t::exact_k);
    split_mix_generator_t generator(case_stream_key());
    std::size_t const n = test_run.settings.dense_dimensions;
    auto source = make_vector<f32_t>(n);
    auto expected = make_vector<f16_t>(n), dispatched = make_vector<f16_t>(n);
    fill_random(generator, source);
    for (auto const &[capability, kernel] : capabilities) {
        if (!(capability & runnable_capabilities())) continue;
        stats.expect_status(
            kernel(source.raw_values_data(), nk_f32_k, expected.raw_values_data(), nk_f16_k, n, nullptr));
        stats.expect_status(nk_cast_best(source.raw_values_data(), nk_f32_k, dispatched.raw_values_data(), nk_f16_k, n,
                                         capability, nullptr));
        stats.expect_true(
            std::memcmp(expected.raw_values_data(), dispatched.raw_values_data(), expected.size_bytes()) == 0,
            "the dispatch point ran another capability's kernel");
    }
    return stats;
}

/** Checks square roots and reciprocals on each runnable backend and their dispatch points. */
template <typename scalar_type_>
static error_stats_t test_best_sqrt() {
    auto select = [](auto f32, auto f64) {
        if constexpr (std::is_same_v<scalar_type_, nk_f32_t>) return f32;
        else return f64;
    };
    struct {
        nk_capability_t capability;
        scalar_type_ (*sqrt)(scalar_type_), (*rsqrt)(scalar_type_);
    } const capabilities[] = {
        {nk_cap_serial_k, select(nk_f32_sqrt_serial, nk_f64_sqrt_serial),
         select(nk_f32_rsqrt_serial, nk_f64_rsqrt_serial)},
#if NUMKONG_TARGET_NEON
        {nk_cap_neon_k, select(nk_f32_sqrt_neon, nk_f64_sqrt_neon), select(nk_f32_rsqrt_neon, nk_f64_rsqrt_neon)},
#endif
#if NUMKONG_TARGET_HASWELL
        {nk_cap_haswell_k, select(nk_f32_sqrt_haswell, nk_f64_sqrt_haswell),
         select(nk_f32_rsqrt_haswell, nk_f64_rsqrt_haswell)},
#endif
#if NUMKONG_TARGET_RVV
        {nk_cap_rvv_k, select(nk_f32_sqrt_rvv, nk_f64_sqrt_rvv), select(nk_f32_rsqrt_rvv, nk_f64_rsqrt_rvv)},
#endif
#if NUMKONG_TARGET_V128
        {nk_cap_v128_k, select(nk_f32_sqrt_v128, nk_f64_sqrt_v128), select(nk_f32_rsqrt_v128, nk_f64_rsqrt_v128)},
#endif
#if NUMKONG_TARGET_POWERVSX
        {nk_cap_powervsx_k, select(nk_f32_sqrt_powervsx, nk_f64_sqrt_powervsx),
         select(nk_f32_rsqrt_powervsx, nk_f64_rsqrt_powervsx)},
#endif
#if NUMKONG_TARGET_LOONGSONASX
        {nk_cap_loongsonasx_k, select(nk_f32_sqrt_loongsonasx, nk_f64_sqrt_loongsonasx),
         select(nk_f32_rsqrt_loongsonasx, nk_f64_rsqrt_loongsonasx)},
#endif
    };
    auto const sqrt_best = select(nk_f32_sqrt_best, nk_f64_sqrt_best);
    auto const rsqrt_best = select(nk_f32_rsqrt_best, nk_f64_rsqrt_best);
    auto const sqrt_serial = capabilities[0].sqrt, rsqrt_serial = capabilities[0].rsqrt;
    using limits = std::numeric_limits<scalar_type_>;
    scalar_type_ const tolerance = 4 * limits::epsilon();
    scalar_type_ const tiny = limits::denorm_min(), normal = (limits::min)();
    scalar_type_ const inputs[] = {0, tiny, 3 * tiny,        normal - tiny,     normal, normal + tiny, 0.5,
                                   2, 3,    (limits::max)(), limits::infinity()};
    error_stats_t stats(comparison_family_t::exact_k);
    nk_capability_t const runnable = runnable_capabilities();
    for (auto const &[capability, sqrt, rsqrt] : capabilities) {
        if (!(capability & runnable)) continue;
        for (scalar_type_ input : inputs) {
            scalar_type_ const expected = std::sqrt(input), root = sqrt(input);
            stats.expect_true(root == expected || std::abs(root / expected - 1) <= tolerance,
                              "square root differs from the reference");
            stats.expect_true(sqrt_best(input, capability) == sqrt(input),
                              "square-root dispatch ran another capability's function");
            if (input == 0) continue;
            scalar_type_ const inverse_expected = 1 / expected, inverse_root = rsqrt(input);
            stats.expect_true(
                inverse_root == inverse_expected || std::abs(inverse_root / inverse_expected - 1) <= tolerance,
                "reciprocal square root differs from the reference");
            stats.expect_true(rsqrt_best(input, capability) == rsqrt(input),
                              "reciprocal-square-root dispatch ran another capability's function");
        }
    }
    stats.expect_true(sqrt_best(2, nk_cap_cuda_k) == sqrt_serial(2), "a GPU mask lost the serial square root");
    stats.expect_true(rsqrt_best(2, nk_cap_cuda_k) == rsqrt_serial(2),
                      "a GPU mask lost the serial reciprocal square root");
    return stats;
}

/** For every kind and dtype, a mask of the runnable capabilities up to each one finds a kernel of
 *  that capability or a lower one, or reports none with null outputs. */
static error_stats_t test_find_kernel() {
    nk_kernel_kind_t const kinds[] = {
        nk_kernel_dot_k,
        nk_kernel_vdot_k,
        nk_kernel_angular_k,
        nk_kernel_euclidean_k,
        nk_kernel_sqeuclidean_k,
        nk_kernel_hamming_k,
        nk_kernel_jaccard_k,
        nk_kernel_bilinear_k,
        nk_kernel_mahalanobis_k,
        nk_kernel_sqeuclideans3d_k,
        nk_kernel_haversines_pack_size_k,
        nk_kernel_haversines_pack_k,
        nk_kernel_haversines_packed_k,
        nk_kernel_vincentys_pack_size_k,
        nk_kernel_vincentys_pack_k,
        nk_kernel_vincentys_packed_k,
        nk_kernel_terrain_pack_size_k,
        nk_kernel_terrain_pack_k,
        nk_kernel_terrain_heights_packed_k,
        nk_kernel_clearances_packed_k,
        nk_kernel_kld_k,
        nk_kernel_jsd_k,
        nk_kernel_sparse_dot_k,
        nk_kernel_sparse_intersect_k,
        nk_kernel_each_scale_k,
        nk_kernel_each_sum_k,
        nk_kernel_each_blend_k,
        nk_kernel_each_fma_k,
        nk_kernel_each_swiglu_k,
        nk_kernel_each_rmsnorm_k,
        nk_kernel_each_rmscast_k,
        nk_kernel_trig_sin_k,
        nk_kernel_trig_cos_k,
        nk_kernel_trig_atan_k,
        nk_kernel_reduce_moments_k,
        nk_kernel_reduce_minmax_k,
        nk_kernel_dots_pack_size_k,
        nk_kernel_dots_pack_k,
        nk_kernel_dots_packed_k,
        nk_kernel_dots_packed_shape_k,
        nk_kernel_dots_symmetric_k,
        nk_kernel_dots_packed_workspace_size_k,
        nk_kernel_dots_symmetric_workspace_size_k,
        nk_kernel_hammings_packed_k,
        nk_kernel_hammings_symmetric_k,
        nk_kernel_jaccards_packed_k,
        nk_kernel_jaccards_symmetric_k,
        nk_kernel_hammings_packed_workspace_size_k,
        nk_kernel_hammings_symmetric_workspace_size_k,
        nk_kernel_jaccards_packed_workspace_size_k,
        nk_kernel_jaccards_symmetric_workspace_size_k,
        nk_kernel_angulars_packed_k,
        nk_kernel_angulars_symmetric_k,
        nk_kernel_euclideans_packed_k,
        nk_kernel_euclideans_symmetric_k,
        nk_kernel_angulars_packed_workspace_size_k,
        nk_kernel_angulars_symmetric_workspace_size_k,
        nk_kernel_euclideans_packed_workspace_size_k,
        nk_kernel_euclideans_symmetric_workspace_size_k,
        nk_kernel_maxsim_pack_size_k,
        nk_kernel_maxsim_pack_k,
        nk_kernel_maxsim_packed_k,
        nk_kernel_maxsim_packed_shape_k,
        nk_kernel_shapes_pack_size_k,
        nk_kernel_shapes_pack_k,
        nk_kernel_shapes_pack_normalized_k,
        nk_kernel_shapes_packed_shape_k,
        nk_kernel_shapes_unpack_k,
        nk_kernel_chamfers_workspace_size_k,
        nk_kernel_chamfers_packed_k,
        nk_kernel_hausdorffs_packed_k,
        nk_kernel_wassersteinprocrustes_workspace_size_k,
        nk_kernel_wassersteinprocrustes_packed_k,
        nk_kernel_gromovwassersteins_workspace_size_k,
        nk_kernel_gromovwassersteins_packed_k,
        nk_kernel_rmsds_packed_k,
        nk_kernel_attention_pack_size_k,
        nk_kernel_attention_pack_k,
        nk_kernel_attention_packed_k,
        nk_kernel_attention_packed_gradients_k,
        nk_kernel_attention_packed_shape_k,
        nk_kernel_attention_rope_k,
        nk_kernel_cast_k,
    };
    nk_dtype_t const dtypes[] = {
        nk_dtype_unknown_k, nk_f64c_k, nk_f32c_k, nk_bf16c_k, nk_f16c_k, nk_f64_k,  nk_f32_k,
        nk_bf16_k,          nk_f16_k,  nk_e5m2_k, nk_e4m3_k,  nk_e3m2_k, nk_e2m3_k, nk_e2m1_k,
        nk_i64_k,           nk_i32_k,  nk_i16_k,  nk_i8_k,    nk_i4_k,   nk_u64_k,  nk_u32_k,
        nk_u16_k,           nk_u8_k,   nk_u4_k,   nk_u1_k,
    };
    error_stats_t stats(comparison_family_t::exact_k);
    nk_capability_t const runnable = runnable_capabilities();
    for (nk_kernel_kind_t kind : kinds)
        for (nk_dtype_t dtype : dtypes)
            for (nk_capability_t highest = 1; highest && highest <= runnable; highest <<= 1) {
                if (!(highest & runnable)) continue;
                nk_capability_t const mask = runnable & (highest | (highest - 1));
                nk_kernel_punned_t kernel = nullptr;
                nk_capability_t capability = nk_cap_any_k;
                nk_status_t const status = nk_find_kernel_punned(kind, dtype, mask, &kernel, &capability);
                if (status != nk_success_k)
                    stats.expect_true(status == nk_missing_kernel_k && !kernel && !capability,
                                      "a missing kernel came back with outputs");
                else
                    stats.expect_true(
                        kernel && std::has_single_bit(capability) && (capability & mask) && capability <= highest,
                        "a found kernel reported a capability outside its mask");
            }
    return stats;
}

/** The CPU device answers as the C queries do, with one ordinal, and every kind refuses the ordinal
 *  past its last device. */
static error_stats_t test_device_capabilities() {
    error_stats_t stats(comparison_family_t::exact_k);
    nk::device_t const cpu = nk::device_t::cpu();
    auto const detected = cpu.capabilities_detected();
    auto const enabled = cpu.capabilities_enabled();
    stats.expect_true(detected && detected.value == capabilities_detected_cpu(), "the CPU detected another mask");
    stats.expect_true(cpu.capabilities_compiled() == capabilities_compiled_cpu(), "the CPU compiled another mask");
    stats.expect_true(enabled && enabled.value == (detected.value & cpu.capabilities_compiled()),
                      "the CPU enabled more than it detected and compiled");
    stats.expect_true(nk::default_capabilities() == enabled.value,
                      "the wrappers' default mask is not the CPU's enabled one");
    stats.expect_true(nk::succeeded(cpu.configure_thread(enabled.value)), "the CPU refused its own mask");
    for (nk::device_kind_t kind :
         {nk::device_kind_t::cpu_k, nk::device_kind_t::cuda_k, nk::device_kind_t::rocm_k, nk::device_kind_t::metal_k}) {
        auto const devices = nk::device_t::count(kind);
        stats.expect_true(kind != nk::device_kind_t::cpu_k || (devices && devices.value == 1), "the CPU is not one");
        auto const past = nk::device_t::make(kind, devices.value);
        stats.expect_true(past.status == nk::status_t::missing_gpu_k, "a device past the last one was made");
        if (kind == nk::device_kind_t::cpu_k || !devices) continue;
        auto const gpu = nk::device_t::make(kind, 0);
        stats.expect_true(gpu && !(gpu.value.capabilities_compiled() & nk_cap_cpus_k),
                          "a GPU compiled a CPU capability");
        stats.expect_true(gpu.value.configure_thread(nk_cap_any_k) == nk::status_t::missing_kernel_k,
                          "a GPU configured a CPU thread");
    }
    return stats;
}

static error_stats_t test_gpu_capabilities(nk::device_t device) {
    error_stats_t stats(comparison_family_t::exact_k);
    auto const detected = device.capabilities_detected();
    auto const enabled = device.capabilities_enabled();
    nk_capability_t const compiled = device.capabilities_compiled();
    nk_capability_t const baseline = device.kind() == nk::device_kind_t::cuda_k   ? nk_cap_cuda_k
                                     : device.kind() == nk::device_kind_t::rocm_k ? nk_cap_rocm_k
                                                                                  : nk_cap_metal_k;
    stats.expect_true(detected && (detected.value & baseline), "the device baseline is missing");
    stats.expect_true(!(compiled & nk_cap_cpus_k), "a GPU compiled a CPU capability");
    stats.expect_true(enabled && enabled.value == (detected.value & compiled),
                      "the device enabled more than it detected and compiled");
    stats.expect_true(!(detected.value & nk_cap_apple10_k) || (detected.value & nk_cap_apple9_k),
                      "Apple10 is reported without Apple9");
    return stats;
}

static void run_dispatch_cases() {
    run_kernel_case("test_best_dot_f32", test_best_dot_f32, nk_cap_serial_k);
    run_kernel_case("test_best_dots_packed_bf16", test_best_dots_packed_bf16, nk_cap_serial_k);
    run_kernel_case("test_best_reduce_moments_f32", test_best_reduce_moments_f32, nk_cap_serial_k);
    run_kernel_case("test_best_cast", test_best_cast, nk_cap_serial_k);
    run_kernel_case("test_best_sqrt<nk_f32_t>", test_best_sqrt<nk_f32_t>, nk_cap_serial_k);
    run_kernel_case("test_best_sqrt<nk_f64_t>", test_best_sqrt<nk_f64_t>, nk_cap_serial_k);
    run_kernel_case("test_find_kernel", test_find_kernel, nk_cap_serial_k);
    run_kernel_case("test_device_capabilities", test_device_capabilities, nk_cap_serial_k);
}

#pragma endregion Dispatch Points

/** A requested GPU, before checking whether its runtime and ordinal are available. */
struct device_selection_t {
    nk::device_kind_t backend;
    std::size_t ordinal;
};

static std::string_view device_name(nk::device_kind_t kind) noexcept {
    switch (kind) {
    case nk::device_kind_t::cpu_k: return "cpu";
    case nk::device_kind_t::cuda_k: return "cuda";
    case nk::device_kind_t::rocm_k: return "rocm";
    case nk::device_kind_t::metal_k: return "metal";
    }
    return "unrecognized";
}

static std::optional<std::vector<device_selection_t>> parse_devices(std::string_view text) {
    std::vector<device_selection_t> devices;
    do {
        std::size_t const comma = text.find(',');
        std::string_view const entry = text.substr(0, comma);
        std::size_t const colon = entry.find(':');
        if (colon == std::string_view::npos) return std::nullopt;
        std::string_view const vendor = entry.substr(0, colon);
        nk::device_kind_t backend;
        if (vendor == "cuda") backend = nk::device_kind_t::cuda_k;
        else if (vendor == "rocm") backend = nk::device_kind_t::rocm_k;
        else if (vendor == "metal") backend = nk::device_kind_t::metal_k;
        else return std::nullopt;
        std::string_view const number = entry.substr(colon + 1);
        std::size_t ordinal = 0;
        auto const [end, error] = std::from_chars(number.data(), number.data() + number.size(), ordinal);
        if (error != std::errc {} || end != number.data() + number.size()) return std::nullopt;
        devices.push_back({backend, ordinal});
        if (comma == std::string_view::npos) return devices;
        text.remove_prefix(comma + 1);
    } while (!text.empty());
    return std::nullopt;
}

/** The devices @c NUMKONG_DEVICES names, or the first of each kind found when it is unset, each
 *  printed in the grammar the setting parses from. */
static std::vector<nk::device_t> selected_devices(int argc, char **argv) {
    std::vector<nk::device_t> devices;
    char const *text = setting_text(argc, argv, "NUMKONG_DEVICES", "--devices");
    if (!text) {
        fmt::println("- Devices: auto");
        for (nk::device_kind_t kind :
             {nk::device_kind_t::cuda_k, nk::device_kind_t::rocm_k, nk::device_kind_t::metal_k})
            if (auto device = nk::device_t::make(kind, 0)) devices.push_back(device.value);
        return devices;
    }
    std::optional<std::vector<device_selection_t>> const requested = parse_devices(text);
    if (!requested) exit_unparsed("NUMKONG_DEVICES", text, "comma-separated devices such as cuda:0,rocm:1");
    for (device_selection_t const &selection : *requested) {
        fmt::println("- Device: {}:{}", device_name(selection.backend), selection.ordinal);
        auto const device = nk::device_t::make(selection.backend, selection.ordinal);
        if (!device) {
            fmt::println(stderr, "{}:{}: {}", device_name(selection.backend), selection.ordinal,
                         nk::status_name(device.status));
            std::exit(1);
        }
        devices.push_back(device.value);
    }
    return devices;
}

/** Runs every case of @p device , over the capabilities it enables. */
static nk::status_t run_device_cases(nk::device_t device) {
    auto const capabilities = device.capabilities_enabled();
    if (!capabilities) return capabilities.status;
    fmt::println("- {}:{}: {}", device_name(device.kind()), device.ordinal(), capability_names_t {capabilities.value});
    test_run.settings.device_ordinal = device.ordinal();
    test_run.settings.capabilities = capabilities.value;
    run_kernel_case("test_gpu_capabilities(device)", test_gpu_capabilities,
                    nk_cap_cuda_k | nk_cap_rocm_k | nk_cap_metal_k, device);
    switch (device.kind()) {
    case nk::device_kind_t::cuda_k:
        run_tensor_cuda_cases();
        run_cast_cuda_cases();
        run_each_cuda_cases();
        run_reduce_cuda_cases();
        run_shapes_cuda_cases();
        run_cross_cuda_cases();
        break;
    case nk::device_kind_t::rocm_k:
        run_cast_rocm_cases();
        run_each_rocm_cases();
        run_shapes_rocm_cases();
        run_cross_rocm_cases();
        break;
    case nk::device_kind_t::metal_k:
        run_cast_metal_cases();
        run_each_metal_cases();
        run_shapes_metal_cases();
        run_cross_metal_cases();
        break;
    default: return nk::status_t::missing_gpu_k;
    }
    return nk::status_t::success_k;
}

int main(int argc, char **argv) {
    start_test_run(argc, argv);
    machine_t const machine = probe_machine();
    print(test_run.settings);
    // ctest hides passing output; the blank line keeps the next binary's header out of this list
    if (char const *summary_path = std::getenv("GITHUB_STEP_SUMMARY"))
        if (std::FILE *summary = std::fopen(summary_path, "a"))
            std::fputc('\n', summary), print(machine, summary), std::fclose(summary);
    auto const devices = selected_devices(argc, argv);
    [[maybe_unused]] nk_status_t const configured = nk_configure_thread_cpu(machine.detected); // Also enables AMX

    run_vector_cases();
    run_tensor_cases();
    run_dispatch_cases();

    run_cast_cases();

    // Core operation tests
    run_dot_cases();
    run_spatial_cases();
    run_curved_cases();
    run_probability_cases();
    run_set_cases();
    run_each_cases();
    run_trigonometry_cases();
    run_reduce_cases();
    run_geospatial_cases();
    run_shapes_cases();
    run_sparse_cases();
    run_maxsim_cases();

    // Cross/batch tests, from ISA-family files for parallel compilation
    run_cross_serial_cases();
    run_cross_runtime_cases();
    run_cross_x8664_cases();
    run_cross_x8664amx_cases();
    run_cross_arm64_cases();
    run_cross_blas_cases();
    run_cross_riscv64_cases();
    run_cross_ppc64_cases();
    run_cross_loongarch64_cases();
    run_cross_wasm_cases();

    for (nk::device_t device : devices)
        if (auto const status = run_device_cases(device); nk::failed(status)) {
            fmt::println(stderr, "{}:{}: {}", device_name(device.kind()), device.ordinal(), nk::status_name(status));
            return 1;
        }
    return finish_test_run();
}
