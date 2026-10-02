/**
 *  @file test/main_cpu.cpp
 *  @author Ash Vardanian
 *  @date December 28, 2025
 *  @brief Test suite entry point and configuration.
 */
#if defined(_MSC_VER)
#define _CRT_SECURE_NO_WARNINGS
#include <io.h> // `_write`
#endif

#if __has_include(<unistd.h>)
#include <unistd.h> // `isatty`, `write`
#endif

#include <string_view> // `std::string_view`

#include "numkong/capabilities.h" // nk_cpu_capabilities, nk_cpu_configure_thread

#if !NUMKONG_ARCH_WASM_
#include <csignal> // `std::signal`, `SIGILL`
#define NUMKONG_HAS_SIGNAL_ 1
#else
#define NUMKONG_HAS_SIGNAL_ 0
#endif

#include "harness.hpp"

using namespace ashvardanian::numkong::test;

char const *volatile nk::test::nk_test_current_kernel_ = nullptr;

/*  Explicit instantiations verify that `random.hpp` compiles for every code path: f64_t for the
 *  scalar float path, i16_t for the scalar signed integer path, bf16c_t for the complex path, and
 *  i4x2_t for the packed sub-byte path. */
template void nk::fill_uniform(std::mt19937 &, nk::f64_t *, std::size_t, nk::f64_t::component_t,
                               nk::f64_t::component_t);
template void nk::fill_uniform(std::mt19937 &, nk::i16_t *, std::size_t, nk::i16_t::component_t,
                               nk::i16_t::component_t);
template void nk::fill_uniform(std::mt19937 &, nk::bf16c_t *, std::size_t, nk::bf16c_t::component_t,
                               nk::bf16c_t::component_t);
template void nk::fill_uniform(std::mt19937 &, nk::i4x2_t *, std::size_t, nk::i4x2_t::component_t,
                               nk::i4x2_t::component_t);
template void nk::fill_uniform(std::mt19937 &, nk::f64_t *, std::size_t);
template void nk::fill_uniform(std::mt19937 &, nk::i16_t *, std::size_t);
template void nk::fill_uniform(std::mt19937 &, nk::bf16c_t *, std::size_t);
template void nk::fill_uniform(std::mt19937 &, nk::i4x2_t *, std::size_t);
template void nk::fill_lognormal(std::mt19937 &, nk::f64_t *, std::size_t, double, double);
template void nk::fill_lognormal(std::mt19937 &, nk::i16_t *, std::size_t, double, double);
template void nk::fill_lognormal(std::mt19937 &, nk::bf16c_t *, std::size_t, double, double);
template void nk::fill_lognormal(std::mt19937 &, nk::i4x2_t *, std::size_t, double, double);
template void nk::fill_cauchy(std::mt19937 &, nk::f64_t *, std::size_t, double, double);
template void nk::fill_cauchy(std::mt19937 &, nk::i16_t *, std::size_t, double, double);
template void nk::fill_cauchy(std::mt19937 &, nk::bf16c_t *, std::size_t, double, double);
template void nk::fill_cauchy(std::mt19937 &, nk::i4x2_t *, std::size_t, double, double);

#if NUMKONG_HAS_SIGNAL_

/** Fatal signal handler that names the signal and the faulting kernel before exiting. */
static void crash_handler(int sig) {
    // Only async-signal-safe calls allowed: write(2) and _exit(2).
    std::string_view sig_name = "unknown signal";
    switch (sig) {
    case SIGILL: sig_name = "SIGILL (illegal instruction)"; break;
    case SIGSEGV: sig_name = "SIGSEGV (segmentation fault)"; break;
#if defined(SIGBUS)
    case SIGBUS: sig_name = "SIGBUS (bus error)"; break;
#endif
    case SIGFPE: sig_name = "SIGFPE (arithmetic exception)"; break;
    case SIGABRT: sig_name = "SIGABRT (abort)"; break;
    }
    auto const emit = [](std::string_view text) noexcept {
#if defined(_WIN32)
        _write(2, text.data(), static_cast<unsigned>(text.size()));
#else
        [[maybe_unused]] auto const written = write(2, text.data(), text.size());
#endif
    };
    char const *const kernel = nk_test_current_kernel_;
    emit(sig_name), emit(" in kernel '"), emit(kernel ? kernel : "(unknown)"), emit("'\n");
    _exit(128 + sig);
}
#endif // NUMKONG_HAS_SIGNAL_

#pragma region Dispatch Points

/** One capability's own kernel, which a dispatch point must run under a mask of it alone. */
template <typename kernel_type_>
struct capability_kernel {
    nk_capability_t capability;
    kernel_type_ kernel;
};

/** The CPU capabilities this machine runs and this binary holds. */
static nk_capability_t runnable_capabilities() noexcept {
    return cpu_capabilities_detected() & cpu_capabilities_compiled();
}

#if !NUMKONG_HEADER_ONLY

/** @c nk_dot_f32_best over each runnable capability alone runs that capability's kernel, and a GPU
 *  mask finds no kernel at all. */
static error_stats_t test_best_dot_f32(settings_t const &settings) {
    capability_kernel<f32_t::dot_kernel_t> const capabilities[] = {
        {nk_cap_serial_k, nk_dot_f32_serial},
#if NUMKONG_TARGET_NEON
        {nk_cap_neon_k, nk_dot_f32_neon},
#endif
#if NUMKONG_TARGET_SVE
        {nk_cap_sve_k, nk_dot_f32_sve},
#endif
#if NUMKONG_TARGET_HASWELL
        {nk_cap_haswell_k, nk_dot_f32_haswell},
#endif
#if NUMKONG_TARGET_SKYLAKE
        {nk_cap_skylake_k, nk_dot_f32_skylake},
#endif
#if NUMKONG_TARGET_RVV
        {nk_cap_rvv_k, nk_dot_f32_rvv},
#endif
#if NUMKONG_TARGET_V128RELAXED
        {nk_cap_v128relaxed_k, nk_dot_f32_v128relaxed},
#endif
#if NUMKONG_TARGET_POWERVSX
        {nk_cap_powervsx_k, nk_dot_f32_powervsx},
#endif
#if NUMKONG_TARGET_LOONGSONASX
        {nk_cap_loongsonasx_k, nk_dot_f32_loongsonasx},
#endif
    };
    error_stats_t stats(comparison_family_t::exact_k);
    std::mt19937 generator(settings.seed.value);
    std::size_t const n = settings.dense_dimensions;
    auto a = make_vector<f32_t>(n), b = make_vector<f32_t>(n);
    fill_random(settings, generator, a), fill_random(settings, generator, b);
    for (auto const &[capability, kernel] : capabilities) {
        if (!(capability & runnable_capabilities())) continue;
        nk_f64_t expected = 0, dispatched = 0;
        stats.expect(kernel(a.raw_values_data(), b.raw_values_data(), n, &expected, nullptr));
        stats.expect(nk_dot_f32_best(a.raw_values_data(), b.raw_values_data(), n, &dispatched, capability, nullptr));
        stats.expect(dispatched == expected, "the dispatch point ran another capability's kernel");
    }
    nk_f64_t unused = 0;
    stats.expect(nk_dot_f32_best(a.raw_values_data(), b.raw_values_data(), n, &unused, nk_cap_cuda_k, nullptr) ==
                     nk_missing_kernel_k,
                 "a GPU mask found a kernel");
    return stats;
}

/** @c nk_dots_pack_bf16_best and @c nk_dots_packed_bf16_best over each runnable capability alone
 *  pack and multiply as its own kernels do, and a GPU mask finds none in a binary without one. */
static error_stats_t test_best_dots_packed_bf16(settings_t const &settings) {
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
#endif
#if NUMKONG_TARGET_NEONBFDOT
        {nk_cap_neonbfdot_k, nk_dots_pack_size_bf16_neonbfdot, nk_dots_pack_bf16_neonbfdot,
         nk_dots_packed_bf16_neonbfdot},
#endif
#if NUMKONG_TARGET_SME
        {nk_cap_sme_k, nk_dots_pack_size_bf16_sme, nk_dots_pack_bf16_sme, nk_dots_packed_bf16_sme},
#endif
#if NUMKONG_TARGET_HASWELL
        {nk_cap_haswell_k, nk_dots_pack_size_bf16_haswell, nk_dots_pack_bf16_haswell, nk_dots_packed_bf16_haswell},
#endif
#if NUMKONG_TARGET_SKYLAKE
        {nk_cap_skylake_k, nk_dots_pack_size_bf16_skylake, nk_dots_pack_bf16_skylake, nk_dots_packed_bf16_skylake},
#endif
#if NUMKONG_TARGET_GENOA
        {nk_cap_genoa_k, nk_dots_pack_size_bf16_genoa, nk_dots_pack_bf16_genoa, nk_dots_packed_bf16_genoa},
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        {nk_cap_sapphireamx_k, nk_dots_pack_size_bf16_sapphireamx, nk_dots_pack_bf16_sapphireamx,
         nk_dots_packed_bf16_sapphireamx},
#endif
#if NUMKONG_TARGET_RVV
        {nk_cap_rvv_k, nk_dots_pack_size_bf16_rvv, nk_dots_pack_bf16_rvv, nk_dots_packed_bf16_rvv},
#endif
#if NUMKONG_TARGET_V128
        {nk_cap_v128_k, nk_dots_pack_size_bf16_v128, nk_dots_pack_bf16_v128, nk_dots_packed_bf16_v128},
#endif
#if NUMKONG_TARGET_V128RELAXED
        {nk_cap_v128relaxed_k, nk_dots_pack_size_bf16_v128relaxed, nk_dots_pack_bf16_v128relaxed,
         nk_dots_packed_bf16_v128relaxed},
#endif
#if NUMKONG_TARGET_POWERVSX
        {nk_cap_powervsx_k, nk_dots_pack_size_bf16_powervsx, nk_dots_pack_bf16_powervsx, nk_dots_packed_bf16_powervsx},
#endif
#if NUMKONG_TARGET_LOONGSONASX
        {nk_cap_loongsonasx_k, nk_dots_pack_size_bf16_loongsonasx, nk_dots_pack_bf16_loongsonasx,
         nk_dots_packed_bf16_loongsonasx},
#endif
    };
    error_stats_t stats(comparison_family_t::exact_k);
    std::mt19937 generator(settings.seed.value);
    std::size_t const height = 17, width = 33, depth = 70, row_bytes = depth * sizeof(bf16_t);
    std::size_t const c_stride = width * sizeof(nk_f32_t);
    auto a = make_vector<bf16_t>(height * depth), b = make_vector<bf16_t>(width * depth);
    auto expected = make_vector<f32_t>(height * width), dispatched = make_vector<f32_t>(height * width);
    fill_random(settings, generator, a), fill_random(settings, generator, b);
    for (auto const &[capability, pack_size, pack, packed] : capabilities) {
        if (!(capability & runnable_capabilities())) continue;
        nk_size_t expected_bytes = 0, dispatched_bytes = 0;
        stats.expect(pack_size(width, depth, &expected_bytes));
        stats.expect(nk_dots_pack_size_bf16_best(width, depth, capability, &dispatched_bytes));
        stats.expect(dispatched_bytes == expected_bytes, "the dispatch point sized another capability's pack");
        auto expected_pack = make_vector<char>(expected_bytes), dispatched_pack = make_vector<char>(expected_bytes);
        stats.expect(pack(b.raw_values_data(), nullptr, width, depth, row_bytes, 0, expected_pack.raw_values_data(), 0,
                          width, nullptr));
        stats.expect(nk_dots_pack_bf16_best(b.raw_values_data(), nullptr, width, depth, row_bytes, 0,
                                            dispatched_pack.raw_values_data(), 0, width, capability, nullptr));
        stats.expect(packed(a.raw_values_data(), nullptr, expected_pack.raw_values_data(), expected.raw_values_data(),
                            height, width, depth, row_bytes, 0, c_stride, nullptr));
        stats.expect(nk_dots_packed_bf16_best(a.raw_values_data(), nullptr, dispatched_pack.raw_values_data(),
                                              dispatched.raw_values_data(), height, width, depth, row_bytes, 0,
                                              c_stride, capability, nullptr));
        stats.expect(std::memcmp(expected.raw_values_data(), dispatched.raw_values_data(), expected.size_bytes()) == 0,
                     "the dispatch point ran another capability's kernel");
    }
    nk_capability_t cuda = 0;
    stats.expect(nk_cuda_capabilities_compiled(&cuda));
    if (!cuda)
        stats.expect(nk_dots_packed_bf16_best(a.raw_values_data(), nullptr, b.raw_values_data(),
                                              expected.raw_values_data(), height, width, depth, row_bytes, 0, c_stride,
                                              nk_cap_cuda_k, nullptr) == nk_missing_kernel_k,
                     "a GPU mask found a kernel in a binary without one");
    return stats;
}

/** @c nk_reduce_moments_f32_best over each runnable capability alone runs its own kernel. */
static error_stats_t test_best_reduce_moments_f32(settings_t const &settings) {
    capability_kernel<f32_t::reduce_moments_kernel_t> const capabilities[] = {
        {nk_cap_serial_k, nk_reduce_moments_f32_serial},
#if NUMKONG_TARGET_NEON
        {nk_cap_neon_k, nk_reduce_moments_f32_neon},
#endif
#if NUMKONG_TARGET_HASWELL
        {nk_cap_haswell_k, nk_reduce_moments_f32_haswell},
#endif
#if NUMKONG_TARGET_SKYLAKE
        {nk_cap_skylake_k, nk_reduce_moments_f32_skylake},
#endif
#if NUMKONG_TARGET_RVV
        {nk_cap_rvv_k, nk_reduce_moments_f32_rvv},
#endif
#if NUMKONG_TARGET_V128RELAXED
        {nk_cap_v128relaxed_k, nk_reduce_moments_f32_v128relaxed},
#endif
    };
    error_stats_t stats(comparison_family_t::exact_k);
    std::mt19937 generator(settings.seed.value);
    std::size_t const n = settings.dense_dimensions;
    auto data = make_vector<f32_t>(n);
    fill_random(settings, generator, data);
    for (auto const &[capability, kernel] : capabilities) {
        if (!(capability & runnable_capabilities())) continue;
        nk_f64_t expected[2] = {0, 0}, dispatched[2] = {0, 0};
        stats.expect(kernel(data.raw_values_data(), n, sizeof(nk_f32_t), &expected[0], &expected[1], nullptr));
        stats.expect(nk_reduce_moments_f32_best(data.raw_values_data(), n, sizeof(nk_f32_t), &dispatched[0],
                                                &dispatched[1], capability, nullptr));
        stats.expect(dispatched[0] == expected[0] && dispatched[1] == expected[1],
                     "the dispatch point ran another capability's kernel");
    }
    return stats;
}

/** @c nk_cast_best over each runnable capability alone runs that capability's kernel. */
static error_stats_t test_best_cast(settings_t const &settings) {
    using cast_kernel_t = nk_status_t (*)(void const *, nk_dtype_t, nk_size_t, void *, nk_dtype_t, void *);
    capability_kernel<cast_kernel_t> const capabilities[] = {
        {nk_cap_serial_k, nk_cast_serial},
#if NUMKONG_TARGET_NEON
        {nk_cap_neon_k, nk_cast_neon},
#endif
#if NUMKONG_TARGET_HASWELL
        {nk_cap_haswell_k, nk_cast_haswell},
#endif
#if NUMKONG_TARGET_SKYLAKE
        {nk_cap_skylake_k, nk_cast_skylake},
#endif
#if NUMKONG_TARGET_ICELAKE
        {nk_cap_icelake_k, nk_cast_icelake},
#endif
#if NUMKONG_TARGET_SAPPHIRE
        {nk_cap_sapphire_k, nk_cast_sapphire},
#endif
#if NUMKONG_TARGET_RVV
        {nk_cap_rvv_k, nk_cast_rvv},
#endif
#if NUMKONG_TARGET_V128RELAXED
        {nk_cap_v128relaxed_k, nk_cast_v128relaxed},
#endif
#if NUMKONG_TARGET_POWERVSX
        {nk_cap_powervsx_k, nk_cast_powervsx},
#endif
    };
    error_stats_t stats(comparison_family_t::exact_k);
    std::mt19937 generator(settings.seed.value);
    std::size_t const n = settings.dense_dimensions;
    auto source = make_vector<f32_t>(n);
    auto expected = make_vector<f16_t>(n), dispatched = make_vector<f16_t>(n);
    fill_random(settings, generator, source);
    for (auto const &[capability, kernel] : capabilities) {
        if (!(capability & runnable_capabilities())) continue;
        stats.expect(kernel(source.raw_values_data(), nk_f32_k, n, expected.raw_values_data(), nk_f16_k, nullptr));
        stats.expect(nk_cast_best(source.raw_values_data(), nk_f32_k, n, dispatched.raw_values_data(), nk_f16_k,
                                  capability, nullptr));
        stats.expect(std::memcmp(expected.raw_values_data(), dispatched.raw_values_data(), expected.size_bytes()) == 0,
                     "the dispatch point ran another capability's kernel");
    }
    return stats;
}

#endif // !NUMKONG_HEADER_ONLY

/** @c nk_f32_sqrt_best over each runnable capability alone runs that capability's function, or
 *  serial in a header-only build, and keeps serial for a mask of no CPU capability. */
static error_stats_t test_best_f32_sqrt(settings_t const &) {
    capability_kernel<nk_f32_t (*)(nk_f32_t)> const capabilities[] = {
        {nk_cap_serial_k, nk_f32_sqrt_serial},
#if NUMKONG_TARGET_NEON
        {nk_cap_neon_k, nk_f32_sqrt_neon},
#endif
#if NUMKONG_TARGET_HASWELL
        {nk_cap_haswell_k, nk_f32_sqrt_haswell},
#endif
#if NUMKONG_TARGET_RVV
        {nk_cap_rvv_k, nk_f32_sqrt_rvv},
#endif
#if NUMKONG_TARGET_V128
        {nk_cap_v128_k, nk_f32_sqrt_v128},
#endif
#if NUMKONG_TARGET_POWERVSX
        {nk_cap_powervsx_k, nk_f32_sqrt_powervsx},
#endif
#if NUMKONG_TARGET_LOONGSONASX
        {nk_cap_loongsonasx_k, nk_f32_sqrt_loongsonasx},
#endif
    };
    error_stats_t stats(comparison_family_t::exact_k);
    nk_f32_t const inputs[] = {0.0f, 1e-30f, 0.5f, 2.0f, 3.0f, 1e30f};
    for (auto const &[capability, function] : capabilities) {
        if (!(capability & runnable_capabilities())) continue;
        for (nk_f32_t input : inputs)
            stats.expect(
                nk_f32_sqrt_best(input, capability) == (NUMKONG_HEADER_ONLY ? nk_f32_sqrt_serial : function)(input),
                "the dispatch point ran another capability's function");
    }
    stats.expect(nk_f32_sqrt_best(2.0f, nk_cap_cuda_k) == nk_f32_sqrt_serial(2.0f),
                 "a GPU mask lost the serial capability");
    return stats;
}

#if !NUMKONG_HEADER_ONLY

/** For every kind and dtype, a mask of the runnable capabilities up to each one finds a kernel of
 *  that capability or a lower one, or reports none with null outputs. */
static error_stats_t test_find_kernel(settings_t const &) {
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
        nk_kernel_haversine_k,
        nk_kernel_vincenty_k,
        nk_kernel_kld_k,
        nk_kernel_jsd_k,
        nk_kernel_rmsd_k,
        nk_kernel_kabsch_k,
        nk_kernel_umeyama_k,
        nk_kernel_sparse_dot_k,
        nk_kernel_sparse_intersect_k,
        nk_kernel_each_scale_k,
        nk_kernel_each_sum_k,
        nk_kernel_each_blend_k,
        nk_kernel_each_fma_k,
        nk_kernel_each_swiglu_k,
        nk_kernel_each_rmsnorm_k,
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
        nk_kernel_hammings_packed_k,
        nk_kernel_hammings_symmetric_k,
        nk_kernel_jaccards_packed_k,
        nk_kernel_jaccards_symmetric_k,
        nk_kernel_angulars_packed_k,
        nk_kernel_angulars_symmetric_k,
        nk_kernel_euclideans_packed_k,
        nk_kernel_euclideans_symmetric_k,
        nk_kernel_maxsim_pack_size_k,
        nk_kernel_maxsim_pack_k,
        nk_kernel_maxsim_packed_k,
        nk_kernel_maxsim_packed_shape_k,
        nk_kernel_attention_pack_size_k,
        nk_kernel_attention_pack_k,
        nk_kernel_attention_bidirectional_packed_k,
        nk_kernel_attention_causal_packed_k,
        nk_kernel_attention_packed_shape_k,
        nk_kernel_attention_rope_k,
        nk_kernel_cast_k,
        nk_kernel_cast_block_scaled_k,
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
                    stats.expect(status == nk_missing_kernel_k && !kernel && !capability,
                                 "a missing kernel came back with outputs");
                else
                    stats.expect(
                        kernel && std::has_single_bit(capability) && (capability & mask) && capability <= highest,
                        "a found kernel reported a capability outside its mask");
            }
    return stats;
}

#else

/** The header-only finder is a stub, reporting the missing library with null outputs. */
static error_stats_t test_find_kernel(settings_t const &) {
    error_stats_t stats(comparison_family_t::exact_k);
    nk_kernel_punned_t kernel = nullptr;
    nk_capability_t capability = nk_cap_any_k;
    nk_status_t const status = nk_find_kernel_punned(nk_kernel_dot_k, nk_f32_k, nk_cap_any_k, &kernel, &capability);
    stats.expect(status == nk_missing_library_k && !kernel && !capability, "the header-only finder found a kernel");
    return stats;
}

#endif // !NUMKONG_HEADER_ONLY

/** The CPU device answers as the C queries do, with one ordinal, and every kind refuses the ordinal
 *  past its last device. */
static error_stats_t test_device_capabilities(settings_t const &) {
    error_stats_t stats(comparison_family_t::exact_k);
    nk::device_t const cpu = nk::device_t::cpu();
    auto const detected = cpu.capabilities_detected();
    auto const enabled = cpu.capabilities_enabled();
    stats.expect(detected && detected.value == cpu_capabilities_detected(), "the CPU detected another mask");
    stats.expect(cpu.capabilities_compiled() == cpu_capabilities_compiled(), "the CPU compiled another mask");
    stats.expect(enabled && enabled.value == (detected.value & cpu.capabilities_compiled()),
                 "the CPU enabled more than it detected and compiled");
    stats.expect(nk::default_capabilities() == (NUMKONG_HEADER_ONLY ? 0 : enabled.value),
                 "the wrappers' default mask is not the CPU's enabled one");
    stats.expect(nk::succeeded(cpu.configure_thread(enabled.value)), "the CPU refused its own mask");
    for (nk::device_kind_t kind :
         {nk::device_kind_t::cpu_k, nk::device_kind_t::cuda_k, nk::device_kind_t::rocm_k, nk::device_kind_t::metal_k}) {
        auto const devices = nk::device_t::count(kind);
        stats.expect(kind != nk::device_kind_t::cpu_k || (devices && devices.value == 1), "the CPU is not one");
        auto const past = nk::device_t::make(kind, devices.value);
        stats.expect(past.status == nk::status_t::missing_gpu_k, "a device past the last one was made");
        if (kind == nk::device_kind_t::cpu_k || !devices) continue;
        auto const gpu = nk::device_t::make(kind, 0);
        stats.expect(gpu && !(gpu.value.capabilities_compiled() & nk_cap_cpus_k), "a GPU compiled a CPU capability");
        stats.expect(gpu.value.configure_thread(nk_cap_any_k) == nk::status_t::missing_kernel_k,
                     "a GPU configured a CPU thread");
    }
    return stats;
}

static void test_dispatch_points(error_stats_section_t &check) {
    check.section("Dispatch Points", nk_cap_serial_k);
#if NUMKONG_HEADER_ONLY
    check("best_dot_f32", [](settings_t const &settings) {
        return test_missing_library<nk_dot_f32_best>(settings, nullptr, nullptr, 0, nullptr, nullptr);
    });
    check("best_dots_packed_bf16", [](settings_t const &settings) {
        return test_missing_library<nk_dots_packed_bf16_best>(settings, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 0,
                                                              0, 0, nullptr);
    });
    check("best_reduce_moments_f32", [](settings_t const &settings) {
        return test_missing_library<nk_reduce_moments_f32_best>(settings, nullptr, 0, 0, nullptr, nullptr, nullptr);
    });
    check("best_cast", [](settings_t const &settings) {
        return test_missing_library<nk_cast_best>(settings, nullptr, nk_f32_k, 0, nullptr, nk_f16_k, nullptr);
    });
#else
    check("best_dot_f32", test_best_dot_f32);
    check("best_dots_packed_bf16", test_best_dots_packed_bf16);
    check("best_reduce_moments_f32", test_best_reduce_moments_f32);
    check("best_cast", test_best_cast);
#endif
    check("best_f32_sqrt", test_best_f32_sqrt);
    check("find_kernel", test_find_kernel);
    check("device_capabilities", test_device_capabilities);
}

#pragma endregion Dispatch Points

int main(int, char **argv) {
    environment_t const env {read_settings(argv[0]), probe_machine()};

#if NUMKONG_HAS_SIGNAL_
    std::signal(SIGILL, crash_handler);
    std::signal(SIGSEGV, crash_handler);
#if defined(SIGBUS)
    std::signal(SIGBUS, crash_handler);
#endif
    std::signal(SIGFPE, crash_handler);
    std::signal(SIGABRT, crash_handler);
#endif // NUMKONG_HAS_SIGNAL_

    // Breadcrumb for crash_handler: if SIGILL fires here, the log shows which call faulted.
    nk_test_current_kernel_ = "nk_cpu_configure_thread()";
    [[maybe_unused]] nk_status_t const configured = nk_cpu_configure_thread(env.machine.detected); // Also enables AMX
    nk_test_current_kernel_ = nullptr;

    print(env.machine);
    print(env.settings);

    error_stats_section_t check(env.settings, env.machine.detected);
    test_vector_types(check);
    test_tensor_ops(check);
    test_dispatch_points(check);

    test_casts(check);

    // Core operation tests
    test_dot(check);
    test_spatial(check);
    test_curved(check);
    test_probability(check);
    test_set(check);
    test_each(check);
    test_trigonometry(check);
    test_reduce(check);
    test_geospatial(check);
    test_mesh(check);
    test_sparse(check);
    test_maxsim(check);

    // Cross/batch tests (ISA-family files for parallel compilation); each prints its own section
    test_cross_serial(check);
    test_cross_x8664(check);
    test_cross_arm64(check);
    test_cross_blas(check);
    test_cross_riscv64(check);
    test_cross_ppc64(check);
    test_cross_loongarch64(check);
    test_cross_wasm(check);

    if (check.failure_count > 0) {
        fmt::println("\n{} kernel(s) failed accuracy checks.", check.failure_count);
        return env.settings.on_failure == on_failure_t::exit_k ? 1 : 0;
    }
    fmt::println("\nAll tests passed.");
    return 0;
}
