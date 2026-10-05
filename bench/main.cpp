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

static std::vector<nk::device_t> select_devices(std::optional<std::vector<device_selection_t>> const &requested) {
    std::vector<nk::device_t> devices;
    if constexpr (NUMKONG_HEADER_ONLY) {
        if (requested) {
            fmt::println(stderr, "This header-only executable accepts only CPU workloads");
            std::exit(1);
        }
        return devices;
    }
    if (requested) {
        for (device_selection_t const &selection : *requested) {
            auto const device = nk::device_t::make(selection.backend, selection.ordinal);
            if (!device) {
                fmt::println(stderr, "{}:{}: {}", device_name(selection.backend), selection.ordinal,
                             nk::status_name(device.status));
                std::exit(1);
            }
            devices.push_back(device.value);
        }
    }
    else {
        for (nk::device_kind_t kind :
             {nk::device_kind_t::cuda_k, nk::device_kind_t::rocm_k, nk::device_kind_t::metal_k})
            if (auto device = nk::device_t::make(kind, 0)) devices.push_back(device.value);
    }
    return devices;
}

static nk::status_t bench_device(environment_t const &env, nk::device_t device) {
    auto const capabilities = device.capabilities_enabled();
    if (!capabilities) return capabilities.status;
    auto [stream, stream_status] = nk::stream_t::make(device);
    if (nk::failed(stream_status)) return stream_status;
    auto [memory, memory_status] = nk::allocator<char>::make(capabilities.value, stream.get(),
                                                             nk_allocator_init_device_best);
    if (nk::failed(memory_status)) return memory_status;
    char families[NUMKONG_CAPABILITIES_NAME_CAPACITY];
    nk_capabilities_name(capabilities.value, families, sizeof(families));
    fmt::println("- {}:{}: {}", device_name(device.kind()), device.ordinal(), families);
    device_backend_t const runtime {device, capabilities.value, memory};
    switch (device.kind()) {
    case nk::device_kind_t::cuda_k: return bench_cross_cuda(env, runtime);
    case nk::device_kind_t::rocm_k: return bench_cross_rocm(env, runtime);
    case nk::device_kind_t::metal_k: return bench_cross_metal(env, runtime);
    default: return nk::status_t::missing_gpu_k;
    }
}

int main() {
    environment_t const env {read_settings(), probe_machine()};
    auto const devices = select_devices(env.settings.devices);
    [[maybe_unused]] nk_status_t const configured = nk_cpu_configure_thread(env.machine.detected); // Also enables AMX

#if NUMKONG_COMPARE_TO_MKL
    mkl_set_num_threads(1);
#elif NUMKONG_COMPARE_TO_BLAS
    if (openblas_set_num_threads) openblas_set_num_threads(1);
#endif // NUMKONG_COMPARE_TO_MKL || NUMKONG_COMPARE_TO_BLAS

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
    for (nk::device_t device : devices)
        if (auto const status = bench_device(env, device); nk::failed(status)) {
            fmt::println(stderr, "{}:{}: {}", device_name(device.kind()), device.ordinal(), nk::status_name(status));
            return 1;
        }
    return 0;
}
