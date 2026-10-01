/**
 *  @file bench/main.cpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 *  @brief NumKong C++ benchmark suite, main entry point.
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
 *  The variables it reads are listed in `harness.hpp`.
 */

#include <fmt/format.h> // `fmt::println`

#include "numkong/capabilities.h" // Runtime capability detection

#include "harness.hpp"

using namespace ashvardanian::numkong::bench;

int main() {
    environment_t const env {read_settings(), probe_machine()};
    [[maybe_unused]] nk_status_t const configured = nk_cpu_configure_thread(env.machine.detected); // Also enables AMX

#if NUMKONG_COMPARE_TO_MKL
    mkl_set_num_threads(1);
#elif NUMKONG_COMPARE_TO_BLAS
    if (openblas_set_num_threads) openblas_set_num_threads(1);
#endif

    print(env.machine);
    print(env.settings);
    fmt::println("");

    // Run all benchmarks from split files
    bench_dot(env);
    bench_spatial(env);
    bench_set(env);
    bench_curved(env);
    bench_probability(env);
    bench_each(env);
    bench_trigonometry(env);
    bench_geospatial(env);
    bench_mesh(env);
    bench_sparse(env);
    bench_sparse_dot(env);
    bench_cast(env);
    bench_reduce(env);
    bench_maxsim(env);

    // Cross/batch benchmarks, ISA-family files for parallel compilation
    bench_cross_serial(env);
    bench_cross_x8664(env);
    bench_cross_arm64(env);
    bench_cross_blas(env);
    bench_cross_riscv64(env);
    bench_cross_ppc64(env);
    bench_cross_wasm(env);
    bench_cross_loongarch64(env);
    bench_cross_cuda(env);
    bench_cross_metal(env);
    return 0;
}
