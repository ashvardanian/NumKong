/**
 *  @file bench/main.cpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 *  @brief NumKong C++ benchmark suite using Google Benchmark, main entry point.
 *
 *  Comprehensive benchmarks for NumKong SIMD-optimized functions measuring throughput performance.
 *  Run the benchmarks with:
 *
 *  @code{.sh}
 *  cmake -B build_release -D NUMKONG_BUILD_BENCH=1
 *  cmake --build build_release
 *  build_release/numkong_bench
 *  @endcode
 *
 *  Environment Variables:
 *
 *  @verbatim
 *  NUMWARS_FILTER=<pattern>           - Filter benchmarks by name regex, default run all
 *  NUMKONG_SEED=N                     - RNG seed, or random to draw one, default 42
 *  NUMWARS_PROFILE_SECONDS=<seconds>  - Min time per benchmark, default 10
 *  NUMKONG_BUDGET_MB=N                - Memory budget in MB for inputs, default 1024
 *
 *  NUMWARS_DIMS=N                     - Vector dimension for dot/spatial benchmarks, default 1536
 *  NUMWARS_MESH_POINTS=N              - Point count for mesh benchmarks, default 1000
 *  NUMWARS_DIMS_HEIGHT=N              - GEMM M dimension, default 1024, like dataset size for kNN
 *  NUMWARS_DIMS_WIDTH=N               - GEMM N dimension, default 128, like query count for kNN
 *  NUMWARS_DIMS_DEPTH=N               - GEMM K dimension, default 1536, like vector dimension for kNN
 *
 *  NUMKONG_CURVED_DIMENSIONS=N        - Vector dimension for curved benchmarks, default 64
 *  NUMKONG_SPARSE_FIRST_LENGTH=N      - First set size for sparse benchmarks, default 1024
 *  NUMKONG_SPARSE_SECOND_LENGTH=N     - Second set size for sparse benchmarks, default 8192
 *  NUMKONG_SPARSE_INTERSECTION=F      - Intersection share 0.0-1.0, default 0.5
 *  @endverbatim
 */

#include <cstring> // `std::strcmp`, `std::strncmp`
#include <string>  // `std::string`
#include <vector>  // `std::vector`

#include <fmt/format.h> // `fmt::format`, `fmt::println`

#include "numkong/capabilities.h" // Runtime capability detection

#include "../test/harness.hpp" // `env_variable`, `log_environment`
#include "harness.hpp"

using namespace ashvardanian::numkong::bench;
using ashvardanian::numkong::test::env_variable;
using ashvardanian::numkong::test::log_environment;

bench_config_t nk::bench::bench_config;

void bench_config_t::load_environment() noexcept {
    auto const positive = [](char const *name, auto fallback) noexcept {
        auto const value = env_variable(name, fallback);
        if (value > 0) return value;
        fmt::println(stderr, "{} must be positive", name);
        std::abort();
    };
    auto const within = [](char const *name, auto fallback, double low, double high) noexcept {
        auto const value = env_variable(name, fallback);
        if (value >= low && value <= high) return value;
        fmt::println(stderr, "{}={} is outside [{}, {}]", name, value, low, high);
        std::abort();
    };
    dense_dimensions = positive("NUMWARS_DIMS", dense_dimensions);
    curved_dimensions = positive("NUMKONG_CURVED_DIMENSIONS", curved_dimensions);
    mesh_points = positive("NUMWARS_MESH_POINTS", mesh_points);
    matrix_height = positive("NUMWARS_DIMS_HEIGHT", matrix_height);
    matrix_width = positive("NUMWARS_DIMS_WIDTH", matrix_width);
    matrix_depth = positive("NUMWARS_DIMS_DEPTH", matrix_depth);
    sparse_first_length = positive("NUMKONG_SPARSE_FIRST_LENGTH", sparse_first_length);
    sparse_second_length = positive("NUMKONG_SPARSE_SECOND_LENGTH", sparse_second_length);
    bool const random_seed = std::strcmp(env_variable("NUMKONG_SEED", ""), "random") == 0;
    seed = random_seed ? std::random_device {}() : env_variable("NUMKONG_SEED", seed);
    sparse_intersection_share = within("NUMKONG_SPARSE_INTERSECTION", sparse_intersection_share, 0, 1);
    max_coord_angle = within("NUMKONG_MAX_COORD_ANGLE", max_coord_angle, 0, 180);
    budget_bytes = positive("NUMKONG_BUDGET_MB", budget_bytes >> 20) << 20;
}

/** Google Benchmark's arguments: foreign flags translated, and @c NUMWARS_FILTER and
 *  @c NUMWARS_PROFILE_SECONDS injected. Prints the NumKong variables ahead of its `--help`. */
static std::vector<std::string> benchmark_arguments(int argc, char **argv) {
    std::vector<std::string> arguments = {argv[0]};
    bool user_set_min_time = false;
    bool wants_help = false;

    for (int index = 1; index < argc; ++index) {
        // Foreign flags from numkong_test
        if (std::strncmp(argv[index], "--filter=", 9) == 0) {
            arguments.push_back(std::string("--benchmark_filter=") + (argv[index] + 9));
            fmt::println(stderr, "Note: Mapped --filter to --benchmark_filter. Prefer: --benchmark_filter='{}'",
                         argv[index] + 9);
        }
        else if (std::strcmp(argv[index], "--filter") == 0 && index + 1 < argc) {
            arguments.push_back(std::string("--benchmark_filter=") + argv[++index]);
            fmt::println(stderr, "Note: Mapped --filter to --benchmark_filter. Prefer: --benchmark_filter='{}'",
                         argv[index]);
        }
        else if (std::strcmp(argv[index], "--assert") == 0 || std::strcmp(argv[index], "--verbose") == 0) {
            fmt::println(stderr, "Note: '{}' is a numkong_test flag, not supported in numkong_bench. Ignoring.",
                         argv[index]);
        }
        // Foreign flags from GTest
        else if (std::strncmp(argv[index], "--gtest_filter=", 15) == 0) {
            arguments.push_back(std::string("--benchmark_filter=") + (argv[index] + 15));
            fmt::println(stderr, "Note: Mapped --gtest_filter to --benchmark_filter. Prefer: --benchmark_filter='{}'",
                         argv[index] + 15);
        }
        else if (std::strncmp(argv[index], "--gtest_", 8) == 0) {
            fmt::println(stderr, "Note: GTest flag '{}' is not supported in numkong_bench. Ignoring.", argv[index]);
        }
        // Track user-provided --benchmark_min_time so we don't override it
        else if (std::strncmp(argv[index], "--benchmark_min_time", 20) == 0) {
            user_set_min_time = true;
            arguments.push_back(argv[index]);
        }
        else if (std::strcmp(argv[index], "--help") == 0 || std::strcmp(argv[index], "-h") == 0) {
            wants_help = true;
            arguments.push_back(argv[index]);
        }
        // Everything else passes through to Google Benchmark
        else { arguments.push_back(argv[index]); }
    }

    // Inject from env vars
    if (char const *filter = env_variable<char const *>("NUMWARS_FILTER", nullptr)) {
        arguments.push_back(std::string("--benchmark_filter=") + filter);
        fmt::println("Applying benchmark filter from NUMWARS_FILTER: {}\n", filter);
    }
    if (!user_set_min_time)
        arguments.push_back(fmt::format("--benchmark_min_time={}s", env_variable("NUMWARS_PROFILE_SECONDS", 10.0)));

    if (wants_help) {
        fmt::print(                                                                                    //
            "Usage: numkong_bench [--benchmark_filter=<regex>] [--benchmark_min_time=<N>s] [--help]\n" //
            "\n"                                                                                       //
            "NumKong Environment Variables:\n"                                                         //
            "  NUMWARS_FILTER=<regex>             Same as --benchmark_filter\n"                        //
            "  NUMWARS_PROFILE_SECONDS=<seconds>  Min time per benchmark (default: 10)\n"              //
            "  NUMKONG_SEED=<int|random>          Random seed (default: 42)\n"                         //
            "  NUMWARS_DIMS=N                     Dense vector dimensions (default: 1536)\n"           //
            "  NUMKONG_CURVED_DIMENSIONS=N        Curved vector dimensions (default: 64)\n"            //
            "  NUMWARS_MESH_POINTS=N              Mesh point count (default: 1000)\n"                  //
            "  NUMWARS_DIMS_HEIGHT=N              Matrix height\n"                                     //
            "  NUMWARS_DIMS_WIDTH=N               Matrix width\n"                                      //
            "  NUMWARS_DIMS_DEPTH=N               Matrix depth\n"                                      //
            "  NUMKONG_SPARSE_FIRST_LENGTH=N      First sparse vector length\n"                        //
            "  NUMKONG_SPARSE_SECOND_LENGTH=N     Second sparse vector length\n"                       //
            "  NUMKONG_SPARSE_INTERSECTION=F      Intersection share [0.0, 1.0]\n"                     //
            "  NUMKONG_MAX_COORD_ANGLE=F          Max angular separation in degrees (default: 180)\n"  //
            "  NUMKONG_BUDGET_MB=N                Memory budget in MB for inputs (default: {})\n"      //
            "\n"                                                                                       //
            "Google Benchmark flags (passed through):\n",
            bench_config.budget_bytes >> 20);
    }
    return arguments;
}

int main(int argc, char **argv) {
    nk_capability_t runtime_caps = nk_capabilities_detected();
    nk_configure_thread(runtime_caps); // Also enables AMX if available

#if NUMKONG_COMPARE_TO_MKL
    mkl_set_num_threads(1);
#elif NUMKONG_COMPARE_TO_BLAS
    if (openblas_set_num_threads) openblas_set_num_threads(1);
#endif

    bench_config.load_environment();

    log_environment();
    fmt::println("- Seed: {}", bench_config.seed);
#if NUMKONG_BUILD_CUDA
    print_cuda_header();
#endif
    fmt::println(
        "  Dimensions: dense={}  curved={}  mesh={}  matrix={}x{}x{}  sparse={}/{}@{:.2f}  geo={:.0f}\xc2\xb0\n",
        bench_config.dense_dimensions, bench_config.curved_dimensions, bench_config.mesh_points,
        bench_config.matrix_height, bench_config.matrix_width, bench_config.matrix_depth,
        bench_config.sparse_first_length, bench_config.sparse_second_length, bench_config.sparse_intersection_share,
        bench_config.max_coord_angle);

    // Google Benchmark keeps `argv[0]` for its `Running` line, so both vectors outlive the run
    std::vector<std::string> arguments = benchmark_arguments(argc, argv);
    std::vector<char *> argument_pointers;
    for (auto &argument : arguments) argument_pointers.push_back(argument.data());
    int arguments_count = static_cast<int>(argument_pointers.size());
    bm::Initialize(&arguments_count, argument_pointers.data());
    if (bm::ReportUnrecognizedArguments(arguments_count, argument_pointers.data())) return 1;

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
#if NUMKONG_BUILD_CUDA
    bench_cross_cuda();
#endif

    bm::RunSpecifiedBenchmarks();
    bm::Shutdown();
    return 0;
}
