/**
 *  @file bench/bench.cpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 *  @brief NumKong C++ benchmark suite using Google Benchmark, main entry point.
 *
 *  Comprehensive benchmarks for NumKong SIMD-optimized functions measuring throughput performance.
 *  Run the benchmarks with:
 *
 *  @code{.sh}
 *  cmake -B build_release -D NK_BUILD_BENCH=1
 *  cmake --build build_release
 *  build_release/nk_bench
 *  @endcode
 *
 *  Environment Variables:
 *
 *  @verbatim
 *  NK_FILTER=<pattern>        - Filter benchmarks by name regex, default run all
 *  NK_SEED=N                  - RNG seed, default 42
 *  NK_BUDGET_SECS=<seconds>   - Min time per benchmark, default 10
 *  NK_BUDGET_MB=N             - Memory budget in MB for inputs, default 1024
 *
 *  NK_DENSE_DIMENSIONS=N      - Vector dimension for dot/spatial benchmarks, default 1536
 *  NK_MESH_POINTS=N           - Point count for mesh benchmarks, default 1000
 *  NK_MATRIX_HEIGHT=N         - GEMM M dimension, default 1024, like dataset size for kNN
 *  NK_MATRIX_WIDTH=N          - GEMM N dimension, default 128, like query count for kNN
 *  NK_MATRIX_DEPTH=N          - GEMM K dimension, default 1536, like vector dimension for kNN
 *
 *  NK_CURVED_DIMENSIONS=N     - Vector dimension for curved benchmarks, default 64
 *  NK_SPARSE_FIRST_LENGTH=N   - First set size for sparse benchmarks, default 1024
 *  NK_SPARSE_SECOND_LENGTH=N  - Second set size for sparse benchmarks, default 8192
 *  NK_SPARSE_INTERSECTION=F   - Intersection share 0.0-1.0, default 0.5
 *  @endverbatim
 */

#include <cstdio> // `std::printf`

#include "numkong/capabilities.h" // Runtime capability detection

#include "../test/test.hpp" // `print_suite_header`
#include "bench.hpp"

using namespace ashvardanian::numkong::bench;
using ashvardanian::numkong::test::print_suite_header;

bench_config_t nk::bench::bench_config;

int run_benchmarks(int argc, char **argv) {
    nk_capability_t runtime_caps = nk_capabilities_detected();
    nk_configure_thread(runtime_caps); // Also enables AMX if available

#if NK_COMPARE_TO_MKL
    mkl_set_num_threads(1);
#elif NK_COMPARE_TO_BLAS
    if (openblas_set_num_threads) openblas_set_num_threads(1);
#endif

    bench_config.load_environment();

    print_suite_header("NumKong Benchmarking Suite", runtime_caps);
#if NK_BUILD_CUDA_BENCH
    print_cuda_header();
#endif

    // Dimensions row
    std::printf("  Dimensions: dense=%zu  curved=%zu  mesh=%zu  matrix=%zux%zux%zu  sparse=%zu/%zu@%.2f  geo=%.0f°\n",
                bench_config.dense_dimensions, bench_config.curved_dimensions, bench_config.mesh_points,
                bench_config.matrix_height, bench_config.matrix_width, bench_config.matrix_depth,
                bench_config.sparse_first_length, bench_config.sparse_second_length,
                bench_config.sparse_intersection_share, bench_config.max_coord_angle);

    // Bench-specific config
    std::printf("  Bench: seed=%u\n", bench_config.seed);
    std::printf("\n");

    if (!initialize_benchmarks(argc, argv)) return 1;

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
#if NK_BUILD_CUDA_BENCH
    bench_cross_cuda();
#endif

    bm::RunSpecifiedBenchmarks();
    bm::Shutdown();
    return 0;
}

int main(int argc, char **argv) { return run_benchmarks(argc, argv); }
