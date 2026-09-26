/**
 *  @file test/main.cpp
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

#include "numkong/capabilities.h" // nk_capabilities, nk_cpu_configure_thread

#if !NUMKONG_ARCH_WASM_
#include <csignal> // `std::signal`, `SIGILL`
#define NUMKONG_HAS_SIGNAL_ 1
#else
#define NUMKONG_HAS_SIGNAL_ 0
#endif

#include "harness.hpp"

using namespace ashvardanian::numkong::test;

test_config_t nk::test::global_config;
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

int main(int argc, char **argv) {

#if NUMKONG_HAS_SIGNAL_
    std::signal(SIGILL, crash_handler);
    std::signal(SIGSEGV, crash_handler);
#if defined(SIGBUS)
    std::signal(SIGBUS, crash_handler);
#endif
    std::signal(SIGFPE, crash_handler);
    std::signal(SIGABRT, crash_handler);
#endif // NUMKONG_HAS_SIGNAL_

    // The environment first, so the command line overrides it
    global_config.load_environment();
    global_config.program = argv[0];
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--filter=", 9) == 0) { global_config.set_filter(argv[i] + 9); }
        else if (std::strcmp(argv[i], "--filter") == 0 && i + 1 < argc) { global_config.set_filter(argv[++i]); }
        else if (std::strcmp(argv[i], "--assert") == 0) { global_config.assert_on_failure = true; }
        else if (std::strcmp(argv[i], "--verbose") == 0) { global_config.verbose = true; }
        else if (std::strncmp(argv[i], "--budget-secs=", 14) == 0) {
            global_config.budget_seconds = std::atof(argv[i] + 14);
        }
        else if (std::strcmp(argv[i], "--budget-secs") == 0 && i + 1 < argc) {
            global_config.budget_seconds = std::atof(argv[++i]);
        }
        // Foreign flags from GTest
        else if (std::strncmp(argv[i], "--gtest_filter=", 15) == 0) {
            global_config.set_filter(argv[i] + 15);
            fmt::println(stderr, "Note: Mapped --gtest_filter to --filter. Prefer: --filter='{}'",
                         global_config.filter);
        }
        else if (std::strncmp(argv[i], "--gtest_", 8) == 0) {
            fmt::println(stderr, "Note: GTest flag '{}' is not supported in numkong_test. Ignoring.", argv[i]);
        }
        // Foreign flags from Google Benchmark
        else if (std::strncmp(argv[i], "--benchmark_filter=", 19) == 0) {
            global_config.set_filter(argv[i] + 19);
            fmt::println(stderr, "Note: Mapped --benchmark_filter to --filter. Prefer: --filter='{}'",
                         global_config.filter);
        }
        else if (std::strncmp(argv[i], "--benchmark_min_time=", 21) == 0) {
            // `std::atof` stops at a trailing 's', so "10s" reads as 10 seconds
            global_config.budget_seconds = std::atof(argv[i] + 21);
            fmt::println(stderr, "Note: Mapped --benchmark_min_time to --budget-secs. Prefer: --budget-secs={}",
                         global_config.budget_seconds);
        }
        else if (std::strncmp(argv[i], "--benchmark_", 12) == 0) {
            fmt::println(stderr, "Note: Google Benchmark flag '{}' is not supported in numkong_test. Ignoring.",
                         argv[i]);
        }
        else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            fmt::print(                                                                                              //
                "Usage: numkong_test [--filter=<regex>] [--budget-secs=<seconds>] [--assert] [--verbose] [--help]\n" //
                "\n"                                                                                                 //
                "Arguments:\n"                                                                                       //
                "  --filter=<regex>          Filter tests by name (regex or substring)\n"                            //
                "  --budget-secs=<seconds>   Time budget per kernel in seconds (default: 1)\n"                       //
                "  --assert                  Exit 1 when any kernel fails its accuracy check\n"                      //
                "  --verbose                 Verbose output\n"                                                       //
                "\n"                                                                                                 //
                "Environment Variables:\n"                                                                           //
                "  NUMKONG_FILTER=<regex>          Same as --filter\n"                                               //
                "  NUMKONG_BUDGET_SECS=<seconds>   Same as --budget-secs\n"                                          //
                "  NUMKONG_SEED=<int|random>       Random seed (default: 42)\n"                                      //
                "  NUMKONG_IN_QEMU=1               Shrink dimensions for emulated runs\n"                            //
                "  NUMKONG_ASSERT=1                Same as --assert\n"                                               //
                "  NUMKONG_VERBOSE=1               Same as --verbose\n"                                              //
                "  NUMKONG_ULP_THRESHOLD_F32=N     ULP tolerance for f32\n"                                          //
                "  NUMKONG_SCALE_THRESHOLD=X       Max abs error over reference scale (attention)\n"                 //
                "  NUMKONG_ULP_THRESHOLD_F16=N     ULP tolerance for f16\n"                                          //
                "  NUMKONG_ULP_THRESHOLD_BF16=N    ULP tolerance for bf16\n"                                         //
                "  NUMKONG_RANDOM_DISTRIBUTION=X   uniform_k, cauchy_k, lognormal_k\n"                               //
                "  NUMKONG_DENSE_DIMENSIONS=N      Override dense vector dimensions\n"                               //
                "  NUMKONG_CURVED_DIMENSIONS=N     Override curved vector dimensions\n"                              //
                "  NUMKONG_SPARSE_DIMENSIONS=N     Override sparse vector dimensions\n"                              //
                "  NUMKONG_MAX_COORD_ANGLE=N       Max angular separation in degrees (default: 180)\n");             //
            return 0;
        }
        else {
            fmt::println(stderr, "Error: unrecognized argument '{}'. Try --help.", argv[i]);
            return 1;
        }
    }

    // Breadcrumbs for crash_handler: if SIGILL fires here, the log shows which call faulted.
    nk_test_current_kernel_ = "nk_cpu_capabilities_detected()";
    nk_capability_t runtime_caps = nk_cpu_capabilities_detected();
    nk_test_current_kernel_ = "nk_cpu_configure_thread()";
    nk_cpu_configure_thread(runtime_caps); // Also enables AMX if available
    nk_test_current_kernel_ = nullptr;

    log_environment();
    fmt::println("- Seed: {}", global_config.seed);
    fmt::println("- Rerun one test: NUMKONG_SEED={} NUMKONG_FILTER='^<name>$' {}", global_config.seed, argv[0]);
    fmt::println("  Dimensions: dense={}  curved={}  sparse={}  mesh={}  matrix={}x{}x{}",
                 global_config.dense_dimensions, global_config.curved_dimensions, global_config.sparse_dimensions,
                 global_config.mesh_points, global_config.matrix_height, global_config.matrix_width,
                 global_config.matrix_depth);
    fmt::println("  ULP: f32 \xe2\x89\xa4 {}  f16 \xe2\x89\xa4 {}  bf16 \xe2\x89\xa4 {}",
                 global_config.ulp_threshold_f32, global_config.ulp_threshold_f16, global_config.ulp_threshold_bf16);
    fmt::println("  Test: budget={}s  distribution={}  assert={}  qemu={}  native_f16={}  native_bf16={}  mkl={}\n",
                 global_config.budget_seconds, global_config.distribution_name(),
                 global_config.assert_on_failure ? "on" : "off", global_config.running_in_qemu ? "yes" : "no",
                 NUMKONG_NATIVE_F16 ? "yes" : "no", NUMKONG_NATIVE_BF16 ? "yes" : "no",
                 NUMKONG_COMPARE_TO_MKL ? "yes" : "no");

    test_vector_types();
    test_tensor_ops();

    test_casts();

    // Core operation tests
    test_dot();
    test_spatial();
    test_curved();
    test_probability();
    test_set();
    test_each();
    test_trigonometry();
    test_reduce();
    test_geospatial();
    test_mesh();
    test_sparse();
    test_maxsim();

    // Cross/batch tests (ISA-family files for parallel compilation); each prints its own section
    test_cross_serial();
    test_cross_x86();
    test_cross_amx();
    test_cross_arm();
    test_cross_sme();
    test_cross_blas();
    test_cross_rvv();
    test_cross_power();
    test_cross_loongarch();
    test_cross_wasm();

    if (global_config.failure_count > 0) {
        fmt::println("\n{} kernel(s) failed accuracy checks.", global_config.failure_count);
        return global_config.assert_on_failure;
    }
    fmt::println("\nAll tests passed.");
    return 0;
}
