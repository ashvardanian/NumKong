/**
 *  @brief Test suite entry point and configuration.
 *  @file test/test.cpp
 *  @author Ash Vardanian
 *  @date December 28, 2025
 */
#if defined(_MSC_VER)
#define _CRT_SECURE_NO_WARNINGS
#include <io.h> // `_write`
#endif

#if __has_include(<unistd.h>)
#include <unistd.h> // `isatty`, `write`
#endif

#include "numkong/capabilities.h" // nk_capabilities, nk_configure_thread

#if !NK_TARGET_WASM_
#include <csignal> // `std::signal`, `SIGILL`
#define NK_HAS_SIGNAL_ 1
#else
#define NK_HAS_SIGNAL_ 0
#endif

#include "test.hpp"

using namespace ashvardanian::numkong::test;

test_config_t nk::test::global_config;
char const *volatile nk::test::nk_test_current_kernel_ = nullptr;

// Explicit instantiations to verify `random.hpp` compiles for all code paths:
//  - f64_t:   scalar float path
//  - i16_t:   scalar signed integer path
//  - bf16c_t: complex path
//  - i4x2_t:  packed sub-byte path
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

#if NK_HAS_SIGNAL_
/** @brief  Fatal signal handler that logs the signal and faulting kernel before exiting. */
static void crash_handler(int sig) {
    // Only async-signal-safe calls allowed: write(2) and _exit(2).
    char const *sig_name = "unknown signal";
    switch (sig) {
    case SIGILL: sig_name = "SIGILL (illegal instruction)"; break;
    case SIGSEGV: sig_name = "SIGSEGV (segmentation fault)"; break;
#if defined(SIGBUS)
    case SIGBUS: sig_name = "SIGBUS (bus error)"; break;
#endif
    case SIGFPE: sig_name = "SIGFPE (arithmetic exception)"; break;
    case SIGABRT: sig_name = "SIGABRT (abort)"; break;
    }
    char const *name = nk_test_current_kernel_ ? nk_test_current_kernel_ : "(unknown)";
    char buf[512];
    std::size_t len = 0;
    for (std::size_t i = 0; sig_name[i] && len < sizeof(buf); ++i) buf[len++] = sig_name[i];
    char const mid[] = " in kernel '";
    for (std::size_t i = 0; mid[i] && len < sizeof(buf); ++i) buf[len++] = mid[i];
    for (std::size_t i = 0; name[i] && len + 2 < sizeof(buf); ++i) buf[len++] = name[i];
    buf[len++] = '\'';
    buf[len++] = '\n';
#if defined(_WIN32)
    _write(2, buf, (unsigned)len);
#else
    (void)!write(2, buf, len);
#endif
    _exit(128 + sig);
}
#endif // NK_HAS_SIGNAL_

int main(int argc, char **argv) {

#if NK_HAS_SIGNAL_
    std::signal(SIGILL, crash_handler);
    std::signal(SIGSEGV, crash_handler);
#if defined(SIGBUS)
    std::signal(SIGBUS, crash_handler);
#endif
    std::signal(SIGFPE, crash_handler);
    std::signal(SIGABRT, crash_handler);
#endif // NK_HAS_SIGNAL_

    // Parse CLI arguments
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--filter=", 9) == 0) { global_config.filter = argv[i] + 9; }
        else if (std::strcmp(argv[i], "--filter") == 0 && i + 1 < argc) { global_config.filter = argv[++i]; }
        else if (std::strcmp(argv[i], "--assert") == 0) { global_config.assert_on_failure = true; }
        else if (std::strcmp(argv[i], "--verbose") == 0) { global_config.verbose = true; }
        else if (std::strncmp(argv[i], "--time-budget=", 14) == 0) {
            global_config.time_budget_ms = static_cast<std::size_t>(std::atoll(argv[i] + 14));
        }
        else if (std::strcmp(argv[i], "--time-budget") == 0 && i + 1 < argc) {
            global_config.time_budget_ms = static_cast<std::size_t>(std::atoll(argv[++i]));
        }
        // Foreign flags from GTest
        else if (std::strncmp(argv[i], "--gtest_filter=", 15) == 0) {
            global_config.filter = argv[i] + 15;
            std::fprintf(stderr, "Note: Mapped --gtest_filter to --filter. Prefer: --filter='%s'\n",
                         global_config.filter);
        }
        else if (std::strncmp(argv[i], "--gtest_", 8) == 0) {
            std::fprintf(stderr, "Note: GTest flag '%s' is not supported in nk_test. Ignoring.\n", argv[i]);
        }
        // Foreign flags from Google Benchmark
        else if (std::strncmp(argv[i], "--benchmark_filter=", 19) == 0) {
            global_config.filter = argv[i] + 19;
            std::fprintf(stderr, "Note: Mapped --benchmark_filter to --filter. Prefer: --filter='%s'\n",
                         global_config.filter);
        }
        else if (std::strncmp(argv[i], "--benchmark_min_time=", 21) == 0) {
            // Parse value, stripping trailing 's' if present (e.g., "10s" -> 10000 ms)
            char const *val = argv[i] + 21;
            double seconds = std::atof(val);
            global_config.time_budget_ms = static_cast<std::size_t>(seconds * 1000);
            std::fprintf(stderr, "Note: Mapped --benchmark_min_time to --time-budget. Prefer: --time-budget=%zu\n",
                         global_config.time_budget_ms);
        }
        else if (std::strncmp(argv[i], "--benchmark_", 12) == 0) {
            std::fprintf(stderr, "Note: Google Benchmark flag '%s' is not supported in nk_test. Ignoring.\n", argv[i]);
        }
        else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            std::fprintf( //
                stdout,
                "Usage: nk_test [--filter=<regex>] [--time-budget=<ms>] [--assert] [--verbose] [--help]\n" //
                "\n"                                                                                       //
                "Arguments:\n"                                                                             //
                "  --filter=<regex>     Filter tests by name (regex or substring)\n"                       //
                "  --time-budget=<ms>   Time budget per kernel in milliseconds (default: 1000)\n"          //
                "  --assert             Abort on first failure\n"                                          //
                "  --verbose            Verbose output\n"                                                  //
                "\n"                                                                                       //
                "Environment Variables:\n"                                                                 //
                "  NK_FILTER=<regex>          Same as --filter\n"                                          //
                "  NK_BUDGET_SECS=<seconds>   Time budget per kernel (default: 1)\n"                       //
                "  NK_SEED=<int>              Random seed\n"                                               //
                "  NK_IN_QEMU=1               Shrink dimensions for emulated runs\n"                       //
                "  NK_TEST_ASSERT=1           Same as --assert\n"                                          //
                "  NK_TEST_VERBOSE=1          Same as --verbose\n"                                         //
                "  NK_ULP_THRESHOLD_F32=N     ULP tolerance for f32\n"                                     //
                "  NK_SCALE_THRESHOLD=X       Max abs error over reference scale (attention)\n"            //
                "  NK_ULP_THRESHOLD_F16=N     ULP tolerance for f16\n"                                     //
                "  NK_ULP_THRESHOLD_BF16=N    ULP tolerance for bf16\n"                                    //
                "  NK_RANDOM_DISTRIBUTION=X   uniform_k, cauchy_k, lognormal_k\n"                          //
                "  NK_DENSE_DIMENSIONS=N      Override dense vector dimensions\n"                          //
                "  NK_CURVED_DIMENSIONS=N     Override curved vector dimensions\n"                         //
                "  NK_SPARSE_DIMENSIONS=N     Override sparse vector dimensions\n"                         //
                "  NK_MAX_COORD_ANGLE=N       Max angular separation in degrees (default: 180)\n");        //
            return 0;
        }
        else {
            std::fprintf(stderr, "Error: unrecognized argument '%s'. Try --help.\n", argv[i]);
            return 1;
        }
    }

    global_config.load_environment();

    // Breadcrumbs for crash_handler: if SIGILL fires here, the log shows which call faulted.
    nk_test_current_kernel_ = "nk_capabilities_detected()";
    nk_capability_t runtime_caps = nk_capabilities_detected();
    nk_test_current_kernel_ = "nk_configure_thread()";
    nk_configure_thread(runtime_caps); // Also enables AMX if available
    nk_test_current_kernel_ = nullptr;

    print_suite_header("NumKong Precision Testing Suite", runtime_caps);

    // Dimensions row
    std::printf("  Dimensions: dense=%zu  curved=%zu  sparse=%zu  mesh=%zu  matrix=%zux%zux%zu\n",
                global_config.dense_dimensions, global_config.curved_dimensions, global_config.sparse_dimensions,
                global_config.mesh_points, global_config.matrix_height, global_config.matrix_width,
                global_config.matrix_depth);

    // ULP distance thresholds
    std::printf("  ULP: f32 \xe2\x89\xa4 %llu  f16 \xe2\x89\xa4 %llu  bf16 \xe2\x89\xa4 %llu\n",
                (unsigned long long)global_config.ulp_threshold_f32,
                (unsigned long long)global_config.ulp_threshold_f16,
                (unsigned long long)global_config.ulp_threshold_bf16);

    // Test-specific config
    std::printf("  Test: seed=%u  budget=%zums  distribution=%s  assert=%s  qemu=%s\n", global_config.seed,
                global_config.time_budget_ms, global_config.distribution_name(),
                global_config.assert_on_failure ? "on" : "off", global_config.running_in_qemu ? "yes" : "no");
    std::printf("\n");

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
        std::puts("");
        std::printf("%zu kernel(s) failed accuracy checks.\n", global_config.failure_count);
        return 1;
    }
    std::puts("");
    std::printf("All tests passed.\n");
    return 0;
}
