/**
 *  @brief NumKong C++ Benchmark Suite using Google Benchmark - Main entry point.
 *  @file bench/bench.cpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 *
 *  Comprehensive benchmarks for NumKong SIMD-optimized functions measuring
 *  throughput performance. Run with:
 *
 *  ```bash
 *  cmake -B build_release -D NK_BUILD_BENCH=1
 *  cmake --build build_release
 *  build_release/nk_bench
 *  ```
 *
 *  Environment Variables:
 *    NK_FILTER=<pattern>           - Filter benchmarks by name regex (default: run all)
 *    NK_SEED=N                     - RNG seed (default: 42)
 *    NK_BUDGET_SECS=<seconds>      - Min time per benchmark (default: 10)
 *    NK_BUDGET_MB=N                - Memory budget in MB for inputs (default: 1024)
 *
 *    NK_DENSE_DIMENSIONS=N         - Vector dimension for dot/spatial benchmarks (default: 1536)
 *    NK_MESH_POINTS=N              - Point count for mesh benchmarks (default: 1000)
 *    NK_MATRIX_HEIGHT=N            - GEMM M dimension (default: 1024), like dataset size for kNN
 *    NK_MATRIX_WIDTH=N             - GEMM N dimension (default: 128), like queries count for kNN
 *    NK_MATRIX_DEPTH=N             - GEMM K dimension (default: 1536), like vector dimensions in KNN
 *
 *    NK_CURVED_DIMENSIONS=N        - Vector dimension for curved benchmarks (default: 64)
 *    NK_SPARSE_FIRST_LENGTH=N      - First set size for sparse benchmarks (default: 1024)
 *    NK_SPARSE_SECOND_LENGTH=N     - Second set size for sparse benchmarks (default: 8192)
 *    NK_SPARSE_INTERSECTION=F      - Intersection share 0.0-1.0 (default: 0.5)
 */

#include <cstdio>  // `std::printf`, `std::fprintf`
#include <cstdlib> // `std::getenv`, `std::atoll`
#include <cstring> // `std::strcmp`
#if !defined(_WIN32) && !defined(__wasi__)
#include <unistd.h>
#endif

#include "numkong/capabilities.h" // Runtime capability detection

#include "bench.hpp"

static bool colors_enabled() {
    static bool const result = [] {
        if (std::getenv("NO_COLOR")) return false;
        if (std::getenv("FORCE_COLOR")) return true;
#if !defined(_WIN32) && !defined(__wasi__)
        return isatty(fileno(stdout)) != 0;
#else
        return false;
#endif
    }();
    return result;
}

static void print_indicator(bool on) {
    if (on) std::printf(colors_enabled() ? "\033[32m\xe2\x97\x8f\033[0m" : "\xe2\x97\x8f");
    else std::printf(colors_enabled() ? "\033[2m\xe2\x97\x8b\033[0m" : "\xe2\x97\x8b");
}

/**
 *  Tri-state glyph for "compiled in" vs "runtime supports":
 *    ● compiled & runtime usable kernel    — green
 *    ◐ compiled but runtime lacks it       — red (invoking this kernel will SIGILL)
 *    ◑ runtime has it but not compiled in  — yellow (perf left on the table)
 *    ○ neither                             — dim
 */
static void print_indicator_dual(bool compiled, bool runtime) {
    char const *glyph;
    char const *color;
    if (compiled && runtime) glyph = "\xe2\x97\x8f", color = "\033[32m";
    else if (compiled && !runtime) glyph = "\xe2\x97\x90", color = "\033[31m";
    else if (!compiled && runtime) glyph = "\xe2\x97\x91", color = "\033[33m";
    else glyph = "\xe2\x97\x8b", color = "\033[2m";
    if (colors_enabled()) std::printf("%s%s\033[0m", color, glyph);
    else std::printf("%s", glyph);
}

static void print_isa(char const *name, int compiled, nk_capability_t cap, nk_capability_t runtime_caps) {
    bool const runtime = (runtime_caps & cap) != 0;
    if (!compiled && !runtime) return;
    std::printf("  %s ", name);
    print_indicator_dual(compiled != 0, runtime);
}

bench_config_t bench_config;

int main(int argc, char **argv) {
    nk_capability_t runtime_caps = nk_capabilities_detected();
    nk_configure_thread(runtime_caps); // Also enables AMX if available

#if NK_COMPARE_TO_MKL
    mkl_set_num_threads(1);
#elif NK_COMPARE_TO_BLAS
    if (openblas_set_num_threads) openblas_set_num_threads(1);
#endif

    // Override dimensions from environment variables if provided
    if (char const *env_dense = std::getenv("NK_DENSE_DIMENSIONS")) {
        std::size_t parsed_dense = static_cast<std::size_t>(std::atoll(env_dense));
        if (parsed_dense > 0) bench_config.dense_dimensions = parsed_dense;
    }
    if (char const *env_curved = std::getenv("NK_CURVED_DIMENSIONS")) {
        std::size_t parsed_curved = static_cast<std::size_t>(std::atoll(env_curved));
        if (parsed_curved > 0) bench_config.curved_dimensions = parsed_curved;
    }
    if (char const *env_mesh = std::getenv("NK_MESH_POINTS")) {
        std::size_t parsed_mesh = static_cast<std::size_t>(std::atoll(env_mesh));
        if (parsed_mesh > 0) bench_config.mesh_points = parsed_mesh;
    }
    if (char const *env_matrix_height = std::getenv("NK_MATRIX_HEIGHT")) {
        std::size_t parsed = static_cast<std::size_t>(std::atoll(env_matrix_height));
        if (parsed > 0) bench_config.matrix_height = parsed;
    }
    if (char const *env_matrix_width = std::getenv("NK_MATRIX_WIDTH")) {
        std::size_t parsed = static_cast<std::size_t>(std::atoll(env_matrix_width));
        if (parsed > 0) bench_config.matrix_width = parsed;
    }
    if (char const *env_matrix_depth = std::getenv("NK_MATRIX_DEPTH")) {
        std::size_t parsed = static_cast<std::size_t>(std::atoll(env_matrix_depth));
        if (parsed > 0) bench_config.matrix_depth = parsed;
    }
    if (char const *env_seed = std::getenv("NK_SEED")) {
        std::uint32_t parsed = static_cast<std::uint32_t>(std::atoll(env_seed));
        bench_config.seed = parsed;
    }
    if (char const *env_sparse_first = std::getenv("NK_SPARSE_FIRST_LENGTH")) {
        std::size_t parsed = static_cast<std::size_t>(std::atoll(env_sparse_first));
        if (parsed > 0) bench_config.sparse_first_length = parsed;
    }
    if (char const *env_sparse_second = std::getenv("NK_SPARSE_SECOND_LENGTH")) {
        std::size_t parsed = static_cast<std::size_t>(std::atoll(env_sparse_second));
        if (parsed > 0) bench_config.sparse_second_length = parsed;
    }
    if (char const *env_sparse_intersection = std::getenv("NK_SPARSE_INTERSECTION")) {
        double parsed = std::atof(env_sparse_intersection);
        if (parsed >= 0.0 && parsed <= 1.0) bench_config.sparse_intersection_share = parsed;
    }
    if (char const *env_geo_angle = std::getenv("NK_MAX_COORD_ANGLE")) {
        float parsed = static_cast<float>(std::atof(env_geo_angle));
        if (parsed > 0.0f && parsed <= 180.0f) bench_config.max_coord_angle = parsed;
    }
    if (char const *env_budget = std::getenv("NK_BUDGET_MB")) {
        std::size_t parsed_mb = static_cast<std::size_t>(std::atoll(env_budget));
        if (parsed_mb > 0) bench_config.budget_bytes = parsed_mb * 1024 * 1024;
    }

    std::printf(colors_enabled() ? "\033[1mNumKong Benchmarking Suite v%d.%d.%d\033[0m\n"
                                 : "NumKong Benchmarking Suite v%d.%d.%d\n",
                NK_VERSION_MAJOR, NK_VERSION_MINOR, NK_VERSION_PATCH);

    // Compilation row
    std::printf("  Compilation: F16 ");
    print_indicator(NK_NATIVE_F16);
    std::printf("  BF16 ");
    print_indicator(NK_NATIVE_BF16);
    std::printf("  MKL ");
    print_indicator(NK_COMPARE_TO_MKL);
    std::printf("\n");

    // ISA row
    std::printf("  ISA:");
    // x86
    print_isa("Haswell", NK_TARGET_HASWELL, nk_cap_haswell_k, runtime_caps);
    print_isa("Alder", NK_TARGET_ALDER, nk_cap_alder_k, runtime_caps);
    print_isa("Sierra", NK_TARGET_SIERRA, nk_cap_sierra_k, runtime_caps);
    print_isa("Skylake", NK_TARGET_SKYLAKE, nk_cap_skylake_k, runtime_caps);
    print_isa("Ice Lake", NK_TARGET_ICELAKE, nk_cap_icelake_k, runtime_caps);
    print_isa("Genoa", NK_TARGET_GENOA, nk_cap_genoa_k, runtime_caps);
    print_isa("Turin", NK_TARGET_TURIN, nk_cap_turin_k, runtime_caps);
    print_isa("Sapphire", NK_TARGET_SAPPHIRE, nk_cap_sapphire_k, runtime_caps);
    print_isa("Sapphire AMX", NK_TARGET_SAPPHIREAMX, nk_cap_sapphireamx_k, runtime_caps);
    print_isa("Granite AMX", NK_TARGET_GRANITEAMX, nk_cap_graniteamx_k, runtime_caps);
    print_isa("Diamond", NK_TARGET_DIAMOND, nk_cap_diamond_k, runtime_caps);
    // Arm
    print_isa("NEON", NK_TARGET_NEON, nk_cap_neon_k, runtime_caps);
    print_isa("NEON F16", NK_TARGET_NEONHALF, nk_cap_neonhalf_k, runtime_caps);
    print_isa("NEON BF16", NK_TARGET_NEONBFDOT, nk_cap_neonbfdot_k, runtime_caps);
    print_isa("NEON I8", NK_TARGET_NEONSDOT, nk_cap_neonsdot_k, runtime_caps);
    print_isa("NEON FHM", NK_TARGET_NEONFHM, nk_cap_neonfhm_k, runtime_caps);
    print_isa("NEON FP8", NK_TARGET_NEONFP8, nk_cap_neonfp8_k, runtime_caps);
    print_isa("SVE", NK_TARGET_SVE, nk_cap_sve_k, runtime_caps);
    print_isa("SVE F16", NK_TARGET_SVEHALF, nk_cap_svehalf_k, runtime_caps);
    print_isa("SVE BF16", NK_TARGET_SVEBFDOT, nk_cap_svebfdot_k, runtime_caps);
    print_isa("SVE I8", NK_TARGET_SVESDOT, nk_cap_svesdot_k, runtime_caps);
    print_isa("SVE2", NK_TARGET_SVE2, nk_cap_sve2_k, runtime_caps);
    print_isa("SVE2P1", NK_TARGET_SVE2P1, nk_cap_sve2p1_k, runtime_caps);
    print_isa("SME", NK_TARGET_SME, nk_cap_sme_k, runtime_caps);
    print_isa("SME2", NK_TARGET_SME2, nk_cap_sme2_k, runtime_caps);
    print_isa("SME2P1", NK_TARGET_SME2P1, nk_cap_sme2p1_k, runtime_caps);
    print_isa("SME F64", NK_TARGET_SMEF64, nk_cap_smef64_k, runtime_caps);
    print_isa("SME F16", NK_TARGET_SMEHALF, nk_cap_smehalf_k, runtime_caps);
    print_isa("SME BF16", NK_TARGET_SMEBF16, nk_cap_smebf16_k, runtime_caps);
    print_isa("SME BI32", NK_TARGET_SMEBI32, nk_cap_smebi32_k, runtime_caps);
    print_isa("SME FA64", NK_TARGET_SMEFA64, nk_cap_smefa64_k, runtime_caps);
    print_isa("SME LUT2", NK_TARGET_SMELUT2, nk_cap_smelut2_k, runtime_caps);
    // RISC-V
    print_isa("RVV", NK_TARGET_RVV, nk_cap_rvv_k, runtime_caps);
    print_isa("RVV HALF", NK_TARGET_RVVHALF, nk_cap_rvvhalf_k, runtime_caps);
    print_isa("RVV BF16", NK_TARGET_RVVBF16, nk_cap_rvvbf16_k, runtime_caps);
    print_isa("RVV BB", NK_TARGET_RVVBB, nk_cap_rvvbb_k, runtime_caps);
    // LoongArch
    print_isa("LoongArch LASX", NK_TARGET_LOONGSONASX, nk_cap_loongsonasx_k, runtime_caps);
    // Power
    print_isa("Power VSX", NK_TARGET_POWERVSX, nk_cap_powervsx_k, runtime_caps);
    // WASM
    print_isa("V128 Relaxed", NK_TARGET_V128RELAXED, nk_cap_v128relaxed_k, runtime_caps);
    std::printf("\n");

    // Dimensions row
    std::printf("  Dimensions: dense=%zu  curved=%zu  mesh=%zu  matrix=%zux%zux%zu  sparse=%zu/%zu@%.2f  geo=%.0f°\n",
                bench_config.dense_dimensions, bench_config.curved_dimensions, bench_config.mesh_points,
                bench_config.matrix_height, bench_config.matrix_width, bench_config.matrix_depth,
                bench_config.sparse_first_length, bench_config.sparse_second_length,
                bench_config.sparse_intersection_share, bench_config.max_coord_angle);

    // Bench-specific config
    std::printf("  Bench: seed=%u\n", bench_config.seed);
    std::printf("\n");

    // Build args for Google Benchmark: translate foreign flags, inject env var overrides.
    // A single vector<string> owns all argument storage; char* pointers are built at the end.
    std::vector<std::string> args = {argv[0]};
    bool user_set_min_time = false;
    bool wants_help = false;

    for (int i = 1; i < argc; ++i) {
        // Foreign flags from nk_test
        if (std::strncmp(argv[i], "--filter=", 9) == 0) {
            args.push_back(std::string("--benchmark_filter=") + (argv[i] + 9));
            std::fprintf(stderr, "Note: Mapped --filter to --benchmark_filter. Prefer: --benchmark_filter='%s'\n",
                         argv[i] + 9);
        }
        else if (std::strcmp(argv[i], "--filter") == 0 && i + 1 < argc) {
            args.push_back(std::string("--benchmark_filter=") + argv[++i]);
            std::fprintf(stderr, "Note: Mapped --filter to --benchmark_filter. Prefer: --benchmark_filter='%s'\n",
                         argv[i]);
        }
        else if (std::strcmp(argv[i], "--assert") == 0 || std::strcmp(argv[i], "--verbose") == 0) {
            std::fprintf(stderr, "Note: '%s' is an nk_test flag, not supported in nk_bench. Ignoring.\n", argv[i]);
        }
        // Foreign flags from GTest
        else if (std::strncmp(argv[i], "--gtest_filter=", 15) == 0) {
            args.push_back(std::string("--benchmark_filter=") + (argv[i] + 15));
            std::fprintf(stderr, "Note: Mapped --gtest_filter to --benchmark_filter. Prefer: --benchmark_filter='%s'\n",
                         argv[i] + 15);
        }
        else if (std::strncmp(argv[i], "--gtest_", 8) == 0) {
            std::fprintf(stderr, "Note: GTest flag '%s' is not supported in nk_bench. Ignoring.\n", argv[i]);
        }
        // Track user-provided --benchmark_min_time so we don't override it
        else if (std::strncmp(argv[i], "--benchmark_min_time", 20) == 0) {
            user_set_min_time = true;
            args.push_back(argv[i]);
        }
        else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            wants_help = true;
            args.push_back(argv[i]);
        }
        // Everything else passes through to Google Benchmark
        else { args.push_back(argv[i]); }
    }

    // Inject from env vars
    if (char const *env_filter = std::getenv("NK_FILTER")) {
        args.push_back(std::string("--benchmark_filter=") + env_filter);
        std::printf("Applying benchmark filter from NK_FILTER: %s\n\n", env_filter);
    }
    if (!user_set_min_time) {
        if (char const *env_time = std::getenv("NK_BUDGET_SECS"))
            args.push_back(std::string("--benchmark_min_time=") + env_time + "s");
        else args.push_back("--benchmark_min_time=10s");
    }

    // Build char* array for bm::Initialize (must not outlive `args`)
    std::vector<char *> argv_ptrs;
    for (auto &a : args) argv_ptrs.push_back(a.data());
    int bench_argc = static_cast<int>(argv_ptrs.size());

    // Print help if requested
    if (wants_help) {
        std::fprintf( //
            stdout,
            "Usage: nk_bench [--benchmark_filter=<regex>] [--benchmark_min_time=<N>s] [--help]\n" //
            "\n"                                                                                  //
            "NumKong Environment Variables:\n"                                                    //
            "  NK_FILTER=<regex>              Same as --benchmark_filter\n"                       //
            "  NK_BUDGET_SECS=<seconds>       Min time per benchmark (default: 10)\n"             //
            "  NK_SEED=<int>                  Random seed\n"                                      //
            "  NK_DENSE_DIMENSIONS=N          Dense vector dimensions (default: 1536)\n"          //
            "  NK_CURVED_DIMENSIONS=N         Curved vector dimensions (default: 64)\n"           //
            "  NK_MESH_POINTS=N               Mesh point count (default: 1000)\n"                 //
            "  NK_MATRIX_HEIGHT=N             Matrix height\n"                                    //
            "  NK_MATRIX_WIDTH=N              Matrix width\n"                                     //
            "  NK_MATRIX_DEPTH=N              Matrix depth\n"                                     //
            "  NK_SPARSE_FIRST_LENGTH=N       First sparse vector length\n"                       //
            "  NK_SPARSE_SECOND_LENGTH=N      Second sparse vector length\n"                      //
            "  NK_SPARSE_INTERSECTION=F       Intersection share [0.0, 1.0]\n"                    //
            "  NK_MAX_COORD_ANGLE=F           Max angular separation in degrees (default: 180)\n" //
            "  NK_BUDGET_MB=N                 Memory budget in MB for inputs (default: %zu)\n"    //
            "  NO_COLOR=1                     Disable colored output\n"                           //
            "  FORCE_COLOR=1                  Force colored output\n"                             //
            "\n"                                                                                  //
            "Google Benchmark flags (passed through):\n",
            bench_config.budget_bytes / (1024 * 1024)); //
    }

    bm::Initialize(&bench_argc, argv_ptrs.data());
    if (bm::ReportUnrecognizedArguments(bench_argc, argv_ptrs.data())) return 1;

    // Register all benchmarks from split files
    bench_dot();
    bench_spatial();
    bench_set();
    bench_curved();
    bench_probability();
    bench_each();
    bench_trigonometry();
    bench_geospatial();
    bench_mesh();
    bench_sparse();
    bench_sparse_dot();
    bench_cast();
    bench_reduce();
    bench_maxsim();

    // Register cross/batch benchmarks (ISA-family files for parallel compilation)
    bench_cross_serial();
    bench_cross_x86();
    bench_cross_amx();
    bench_cross_arm();
    bench_cross_sme();
    bench_cross_blas();
    bench_cross_rvv();
    bench_cross_power();
    bench_cross_wasm();
    bench_cross_loongarch();

    bm::RunSpecifiedBenchmarks();
    bm::Shutdown();
    return 0;
}
