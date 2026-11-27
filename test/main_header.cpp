/**
 *  @file test/main_header.cpp
 *  @author Ash Vardanian
 *  @date November 27, 2026
 *  @brief Entry point of the header-only suite: every capability's kernels compiled inline, over
 *      the C headers and the @c numkong::types containers alone.
 *
 *  The C++ SDK links the library, so this suite runs the per-capability cross files, whose kernels
 *  it calls by name, and checks that the dispatch points and the finder report the missing library.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#if defined(_MSC_VER)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "numkong/capabilities.h" // `nk_configure_thread_cpu`, `nk_find_kernel_punned`

#include "harness.hpp"

using namespace ashvardanian::numkong::test;

/** The header-only stub of @p best_, called with the arguments of its capability kernels, which it
 *  never reads, reports the missing library over every mask. */
template <auto best_, typename... arguments_types_>
static error_stats_t test_missing_library(arguments_types_... arguments) {
    error_stats_t stats(comparison_family_t::exact_k);
    stats.expect_true(call_best<best_>(nk_cap_any_k, arguments...) == nk_missing_library_k,
                      "a header-only dispatch point ran a kernel");
    return stats;
}

/** The header-only finder is a stub, reporting the missing library with null outputs. */
static error_stats_t test_find_kernel() {
    error_stats_t stats(comparison_family_t::exact_k);
    nk_kernel_punned_t kernel = nullptr;
    nk_capability_t capability = nk_cap_any_k;
    nk_status_t const status = nk_find_kernel_punned(nk_kernel_dot_k, nk_f32_k, nk_cap_any_k, &kernel, &capability);
    stats.expect_true(status == nk_missing_library_k && !kernel && !capability,
                      "the header-only finder found a kernel");
    return stats;
}

static void run_dispatch_cases() {
    run_kernel_case(
        "test_missing_library<nk_dot_f32_best>",
        [] { return test_missing_library<nk_dot_f32_best>(nullptr, nullptr, 0, nullptr, nullptr); }, nk_cap_serial_k);
    run_kernel_case(
        "test_missing_library<nk_dots_packed_bf16_best>",
        [] {
            return test_missing_library<nk_dots_packed_bf16_best>(nullptr, nullptr, nullptr, 0, 0, 0, 0, 0, nullptr, 0,
                                                                  nullptr);
        },
        nk_cap_serial_k);
    run_kernel_case(
        "test_missing_library<nk_reduce_moments_f32_best>",
        [] { return test_missing_library<nk_reduce_moments_f32_best>(nullptr, 0, 0, nullptr, nullptr, nullptr); },
        nk_cap_serial_k);
    run_kernel_case(
        "test_missing_library<nk_cast_best>",
        [] { return test_missing_library<nk_cast_best>(nullptr, nk_f32_k, nullptr, nk_f16_k, 0, nullptr); },
        nk_cap_serial_k);
    run_kernel_case("test_find_kernel", test_find_kernel, nk_cap_serial_k);
}

int main(int argc, char **argv) {
    start_test_run(argc, argv);
    machine_t const machine = probe_machine();
    print(test_run.settings);
    [[maybe_unused]] nk_status_t const configured = nk_configure_thread_cpu(machine.detected); // Also enables AMX

    run_dispatch_cases();
    run_cross_serial_cases();
    run_cross_x8664_cases();
    run_cross_x8664amx_cases();
    run_cross_arm64_cases();
    run_cross_riscv64_cases();
    run_cross_ppc64_cases();
    run_cross_loongarch64_cases();
    run_cross_wasm_cases();
    return finish_test_run();
}
