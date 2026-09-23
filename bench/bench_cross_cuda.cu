/**
 *  @brief Batch operation benchmarks - CUDA kernels against cuBLASLt and cuBLAS.
 *  @file bench/bench_cross_cuda.cu
 *  @author Ash Vardanian
 *  @date September 22, 2026
 *
 *  Runs the drivers of `bench_cross.cuh` through the shared `cuda_backend_t` over device-resident operands, launching
 *  on `cudaStreamPerThread`, with timed windows of launches bracketed by CUDA events and reported through
 *  `UseManualTime`, so launch latency and host synchronization stay outside the measurement. The Time column is per
 *  window, and the `calls` counter recovers the per-call rate. Input sets rotate until their footprint is at least
 * twice the L2.
 *
 *  The `attention` rows time prefill, 4096 queries on 4096 keys, and decode, 1 query on 4096 keys, with 32 query heads
 *  over 8 K and V heads of depth 128. The baselines share the inputs, timing and counters of the NumKong rows.
 *
 *  Environment Variables:
 *    NK_FILTER=<pattern>           - Filter benchmarks by name regex (default: run all)
 *    NK_SEED=N                     - RNG seed (default: 42)
 *    NK_BUDGET_SECS=<seconds>      - Min time per benchmark (default: 10)
 *    NK_BUDGET_MB=N                - Device memory budget in MB for input sets (default: 1024)
 *    NK_MATRIX_HEIGHT=N            - GEMM M dimension (default: 1024)
 *    NK_MATRIX_WIDTH=N             - GEMM N dimension (default: 128)
 *    NK_MATRIX_DEPTH=N             - GEMM K dimension (default: 1536)
 */

#include <cmath>   // `std::fabs`, `std::sqrt`
#include <cstdint> // `std::int32_t`, `std::int64_t`
#include <cstdio>  // `std::printf`
#include <cstdlib> // `std::getenv`, `std::atoll`
#include <cstring> // `std::memcpy`

#include <algorithm>     // `std::clamp`
#include <array>         // `std::array`
#include <bit>           // `std::bit_ceil`, `std::bit_floor`
#include <optional>      // `std::optional`
#include <string>        // `std::string`
#include <unordered_map> // `std::unordered_map`
#include <vector>        // `std::vector`

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "numkong/numkong.h"

#include "../test/test.cuh" // `test::cuda_backend_t`, `device_vector`
#include "bench_cross.cuh"

using namespace ashvardanian::numkong::bench;
using nk::test::device_vector;

bench_config_t nk::bench::bench_config;

#pragma region CUDA Backend

/** The shared CUDA backend over device memory, timing windows of launches with CUDA events. */
struct cuda_backend_t : nk::test::cuda_backend_t {
    /** Device memory, so timed buffers never page-migrate. */
    template <typename value_type_>
    using allocator = nk::test::cuda_device_allocator<value_type_>;

    /** Rotation sets of @p bytes_per_set each: enough to cover twice the L2, a power of two within the budget. */
    std::size_t input_sets(std::size_t bytes_per_set) const noexcept {
        int l2_bytes = 0, device = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, device);
        std::size_t const set_bytes = std::max(bytes_per_set, std::size_t(1));
        std::size_t const wanted = std::max(nk::divide_round_up(2 * std::size_t(l2_bytes), set_bytes), std::size_t(1));
        std::size_t const affordable = std::max(bench_config.budget_bytes / set_bytes, std::size_t(1));
        return std::bit_floor(std::clamp(std::bit_ceil(wanted), std::size_t(1), affordable));
    }

    /** Prefill and decode segments at the end of a 4096-key cache, Llama-style heads. */
    static std::vector<attention_shape_t> attention_shapes() {
        return {{"prefill", 32, 8, 128, 4096, 4096}, {"decode", 32, 8, 128, 1, 4096}};
    }

    /** Times batches of @p launch lasting at least a millisecond each, returning the call count. */
    template <typename launch_type_>
    std::size_t time(bm::State &state, std::size_t sets_count, launch_type_ &launch) {
        cudaEvent_t start = nullptr, stop = nullptr;
        cudaEventCreate(&start), cudaEventCreate(&stop);
        float calibration_milliseconds = 0;
        cudaEventRecord(start, stream);
        for (std::size_t index = 0; index != sets_count; ++index) launch(index);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&calibration_milliseconds, start, stop);
        double const per_call_milliseconds = std::max(double(calibration_milliseconds) / double(sets_count), 1e-4);
        std::size_t const batch = std::clamp<std::size_t>(std::size_t(1.0 / per_call_milliseconds) + 1, 1, 1 << 16);

        std::size_t calls = 0;
        for (auto _ : state) {
            cudaEventRecord(start, stream);
            for (std::size_t index = 0; index != batch && status == cudaSuccess; ++index, ++calls)
                launch(calls & (sets_count - 1));
            cudaEventRecord(stop, stream);
            keep(cudaEventSynchronize(stop));
            if (status != cudaSuccess) break;
            float window_milliseconds = 0;
            cudaEventElapsedTime(&window_milliseconds, start, stop);
            state.SetIterationTime(double(window_milliseconds) / 1e3);
        }
        cudaEventDestroy(start), cudaEventDestroy(stop);
        return calls;
    }

    /** Reports the event-timed windows instead of wall time. */
    static void configure(bm::internal::Benchmark *benchmark) { benchmark->UseManualTime(); }
};

/** Fills every set with copies of @p a_host and @p b_host, reporting whether every allocation succeeded. */
template <typename output_type_>
bool upload_unpacked_sets(cuda_backend_t &backend, std::vector<matrix_set<cuda_backend_t, output_type_>> &sets,
                          void const *a_host, std::size_t a_bytes, void const *b_host, std::size_t b_bytes,
                          std::size_t c_count) {
    for (auto &set : sets) {
        set.a = device_vector<char>::try_empty(a_bytes);
        set.b = device_vector<char>::try_empty(b_bytes);
        set.c = device_vector<output_type_>::try_empty(c_count);
        if (set.a.empty() || set.b.empty() || set.c.empty()) return false;
        backend.copy(set.a.raw_values_data(), a_host, a_bytes);
        backend.copy(set.b.raw_values_data(), b_host, b_bytes);
    }
    return true;
}

#pragma endregion CUDA Backend

#pragma region Registrations

/** Every Ampere entry point, compiled only when the architecture list includes the family. */
void bench_cross_ampere() {
#if NK_TARGET_AMPERE
    using backend_t = cuda_backend_t;
    run_dots_packed<nk_f64_k, backend_t>("dots_packed_f64_ampere", nk_dots_pack_size_f64_ampere,
                                         nk_dots_pack_f64_ampere, nk_dots_packed_f64_ampere);
    run_dots_packed<nk_f32_k, backend_t>("dots_packed_f32_ampere", nk_dots_pack_size_f32_ampere,
                                         nk_dots_pack_f32_ampere, nk_dots_packed_f32_ampere);
    run_dots_packed<nk_bf16_k, backend_t>("dots_packed_bf16_ampere", nk_dots_pack_size_bf16_ampere,
                                          nk_dots_pack_bf16_ampere, nk_dots_packed_bf16_ampere);
    run_dots_packed<nk_f16_k, backend_t>("dots_packed_f16_ampere", nk_dots_pack_size_f16_ampere,
                                         nk_dots_pack_f16_ampere, nk_dots_packed_f16_ampere);
    run_dots_packed<nk_e5m2_k, backend_t>("dots_packed_e5m2_ampere", nk_dots_pack_size_e5m2_ampere,
                                          nk_dots_pack_e5m2_ampere, nk_dots_packed_e5m2_ampere);
    run_dots_packed<nk_e4m3_k, backend_t>("dots_packed_e4m3_ampere", nk_dots_pack_size_e4m3_ampere,
                                          nk_dots_pack_e4m3_ampere, nk_dots_packed_e4m3_ampere);
    run_dots_packed<nk_e3m2_k, backend_t>("dots_packed_e3m2_ampere", nk_dots_pack_size_e3m2_ampere,
                                          nk_dots_pack_e3m2_ampere, nk_dots_packed_e3m2_ampere);
    run_dots_packed<nk_e2m3_k, backend_t>("dots_packed_e2m3_ampere", nk_dots_pack_size_e2m3_ampere,
                                          nk_dots_pack_e2m3_ampere, nk_dots_packed_e2m3_ampere);
    run_dots_packed<nk_e2m1_k, backend_t>("dots_packed_e2m1_ampere", nk_dots_pack_size_e2m1_ampere,
                                          nk_dots_pack_e2m1_ampere, nk_dots_packed_e2m1_ampere);
    run_dots_packed<nk_i8_k, backend_t>("dots_packed_i8_ampere", nk_dots_pack_size_i8_ampere, nk_dots_pack_i8_ampere,
                                        nk_dots_packed_i8_ampere);
    run_dots_packed<nk_i4_k, backend_t>("dots_packed_i4_ampere", nk_dots_pack_size_i4_ampere, nk_dots_pack_i4_ampere,
                                        nk_dots_packed_i4_ampere);
    run_dots_packed<nk_u8_k, backend_t>("dots_packed_u8_ampere", nk_dots_pack_size_u8_ampere, nk_dots_pack_u8_ampere,
                                        nk_dots_packed_u8_ampere);
    run_dots_packed<nk_u4_k, backend_t>("dots_packed_u4_ampere", nk_dots_pack_size_u4_ampere, nk_dots_pack_u4_ampere,
                                        nk_dots_packed_u4_ampere);

    run_dots_symmetric<nk_f64_k, backend_t>("dots_symmetric_f64_ampere", nk_dots_symmetric_f64_ampere);
    run_dots_symmetric<nk_f32_k, backend_t>("dots_symmetric_f32_ampere", nk_dots_symmetric_f32_ampere);
    run_dots_symmetric<nk_bf16_k, backend_t>("dots_symmetric_bf16_ampere", nk_dots_symmetric_bf16_ampere);
    run_dots_symmetric<nk_f16_k, backend_t>("dots_symmetric_f16_ampere", nk_dots_symmetric_f16_ampere);
    run_dots_symmetric<nk_e5m2_k, backend_t>("dots_symmetric_e5m2_ampere", nk_dots_symmetric_e5m2_ampere);
    run_dots_symmetric<nk_e4m3_k, backend_t>("dots_symmetric_e4m3_ampere", nk_dots_symmetric_e4m3_ampere);
    run_dots_symmetric<nk_e3m2_k, backend_t>("dots_symmetric_e3m2_ampere", nk_dots_symmetric_e3m2_ampere);
    run_dots_symmetric<nk_e2m3_k, backend_t>("dots_symmetric_e2m3_ampere", nk_dots_symmetric_e2m3_ampere);
    run_dots_symmetric<nk_e2m1_k, backend_t>("dots_symmetric_e2m1_ampere", nk_dots_symmetric_e2m1_ampere);
    run_dots_symmetric<nk_i8_k, backend_t>("dots_symmetric_i8_ampere", nk_dots_symmetric_i8_ampere);
    run_dots_symmetric<nk_i4_k, backend_t>("dots_symmetric_i4_ampere", nk_dots_symmetric_i4_ampere);
    run_dots_symmetric<nk_u8_k, backend_t>("dots_symmetric_u8_ampere", nk_dots_symmetric_u8_ampere);
    run_dots_symmetric<nk_u4_k, backend_t>("dots_symmetric_u4_ampere", nk_dots_symmetric_u4_ampere);

    run_angulars_packed<nk_f64_k, backend_t>("angulars_packed_f64_ampere", nk_dots_pack_size_f64_ampere,
                                             nk_dots_pack_f64_ampere, nk_angulars_packed_f64_ampere);
    run_angulars_packed<nk_f32_k, backend_t>("angulars_packed_f32_ampere", nk_dots_pack_size_f32_ampere,
                                             nk_dots_pack_f32_ampere, nk_angulars_packed_f32_ampere);
    run_angulars_packed<nk_bf16_k, backend_t>("angulars_packed_bf16_ampere", nk_dots_pack_size_bf16_ampere,
                                              nk_dots_pack_bf16_ampere, nk_angulars_packed_bf16_ampere);
    run_angulars_packed<nk_f16_k, backend_t>("angulars_packed_f16_ampere", nk_dots_pack_size_f16_ampere,
                                             nk_dots_pack_f16_ampere, nk_angulars_packed_f16_ampere);
    run_angulars_packed<nk_e5m2_k, backend_t>("angulars_packed_e5m2_ampere", nk_dots_pack_size_e5m2_ampere,
                                              nk_dots_pack_e5m2_ampere, nk_angulars_packed_e5m2_ampere);
    run_angulars_packed<nk_e4m3_k, backend_t>("angulars_packed_e4m3_ampere", nk_dots_pack_size_e4m3_ampere,
                                              nk_dots_pack_e4m3_ampere, nk_angulars_packed_e4m3_ampere);
    run_angulars_packed<nk_e3m2_k, backend_t>("angulars_packed_e3m2_ampere", nk_dots_pack_size_e3m2_ampere,
                                              nk_dots_pack_e3m2_ampere, nk_angulars_packed_e3m2_ampere);
    run_angulars_packed<nk_e2m3_k, backend_t>("angulars_packed_e2m3_ampere", nk_dots_pack_size_e2m3_ampere,
                                              nk_dots_pack_e2m3_ampere, nk_angulars_packed_e2m3_ampere);
    run_angulars_packed<nk_e2m1_k, backend_t>("angulars_packed_e2m1_ampere", nk_dots_pack_size_e2m1_ampere,
                                              nk_dots_pack_e2m1_ampere, nk_angulars_packed_e2m1_ampere);
    run_angulars_packed<nk_i8_k, backend_t>("angulars_packed_i8_ampere", nk_dots_pack_size_i8_ampere,
                                            nk_dots_pack_i8_ampere, nk_angulars_packed_i8_ampere);
    run_angulars_packed<nk_i4_k, backend_t>("angulars_packed_i4_ampere", nk_dots_pack_size_i4_ampere,
                                            nk_dots_pack_i4_ampere, nk_angulars_packed_i4_ampere);
    run_angulars_packed<nk_u8_k, backend_t>("angulars_packed_u8_ampere", nk_dots_pack_size_u8_ampere,
                                            nk_dots_pack_u8_ampere, nk_angulars_packed_u8_ampere);
    run_angulars_packed<nk_u4_k, backend_t>("angulars_packed_u4_ampere", nk_dots_pack_size_u4_ampere,
                                            nk_dots_pack_u4_ampere, nk_angulars_packed_u4_ampere);

    run_angulars_symmetric<nk_f64_k, backend_t>("angulars_symmetric_f64_ampere", nk_angulars_symmetric_f64_ampere);
    run_angulars_symmetric<nk_f32_k, backend_t>("angulars_symmetric_f32_ampere", nk_angulars_symmetric_f32_ampere);
    run_angulars_symmetric<nk_bf16_k, backend_t>("angulars_symmetric_bf16_ampere", nk_angulars_symmetric_bf16_ampere);
    run_angulars_symmetric<nk_f16_k, backend_t>("angulars_symmetric_f16_ampere", nk_angulars_symmetric_f16_ampere);
    run_angulars_symmetric<nk_e5m2_k, backend_t>("angulars_symmetric_e5m2_ampere", nk_angulars_symmetric_e5m2_ampere);
    run_angulars_symmetric<nk_e4m3_k, backend_t>("angulars_symmetric_e4m3_ampere", nk_angulars_symmetric_e4m3_ampere);
    run_angulars_symmetric<nk_e3m2_k, backend_t>("angulars_symmetric_e3m2_ampere", nk_angulars_symmetric_e3m2_ampere);
    run_angulars_symmetric<nk_e2m3_k, backend_t>("angulars_symmetric_e2m3_ampere", nk_angulars_symmetric_e2m3_ampere);
    run_angulars_symmetric<nk_e2m1_k, backend_t>("angulars_symmetric_e2m1_ampere", nk_angulars_symmetric_e2m1_ampere);
    run_angulars_symmetric<nk_i8_k, backend_t>("angulars_symmetric_i8_ampere", nk_angulars_symmetric_i8_ampere);
    run_angulars_symmetric<nk_i4_k, backend_t>("angulars_symmetric_i4_ampere", nk_angulars_symmetric_i4_ampere);
    run_angulars_symmetric<nk_u8_k, backend_t>("angulars_symmetric_u8_ampere", nk_angulars_symmetric_u8_ampere);
    run_angulars_symmetric<nk_u4_k, backend_t>("angulars_symmetric_u4_ampere", nk_angulars_symmetric_u4_ampere);

    run_euclideans_packed<nk_f64_k, backend_t>("euclideans_packed_f64_ampere", nk_dots_pack_size_f64_ampere,
                                               nk_dots_pack_f64_ampere, nk_euclideans_packed_f64_ampere);
    run_euclideans_packed<nk_f32_k, backend_t>("euclideans_packed_f32_ampere", nk_dots_pack_size_f32_ampere,
                                               nk_dots_pack_f32_ampere, nk_euclideans_packed_f32_ampere);
    run_euclideans_packed<nk_bf16_k, backend_t>("euclideans_packed_bf16_ampere", nk_dots_pack_size_bf16_ampere,
                                                nk_dots_pack_bf16_ampere, nk_euclideans_packed_bf16_ampere);
    run_euclideans_packed<nk_f16_k, backend_t>("euclideans_packed_f16_ampere", nk_dots_pack_size_f16_ampere,
                                               nk_dots_pack_f16_ampere, nk_euclideans_packed_f16_ampere);
    run_euclideans_packed<nk_e5m2_k, backend_t>("euclideans_packed_e5m2_ampere", nk_dots_pack_size_e5m2_ampere,
                                                nk_dots_pack_e5m2_ampere, nk_euclideans_packed_e5m2_ampere);
    run_euclideans_packed<nk_e4m3_k, backend_t>("euclideans_packed_e4m3_ampere", nk_dots_pack_size_e4m3_ampere,
                                                nk_dots_pack_e4m3_ampere, nk_euclideans_packed_e4m3_ampere);
    run_euclideans_packed<nk_e3m2_k, backend_t>("euclideans_packed_e3m2_ampere", nk_dots_pack_size_e3m2_ampere,
                                                nk_dots_pack_e3m2_ampere, nk_euclideans_packed_e3m2_ampere);
    run_euclideans_packed<nk_e2m3_k, backend_t>("euclideans_packed_e2m3_ampere", nk_dots_pack_size_e2m3_ampere,
                                                nk_dots_pack_e2m3_ampere, nk_euclideans_packed_e2m3_ampere);
    run_euclideans_packed<nk_e2m1_k, backend_t>("euclideans_packed_e2m1_ampere", nk_dots_pack_size_e2m1_ampere,
                                                nk_dots_pack_e2m1_ampere, nk_euclideans_packed_e2m1_ampere);
    run_euclideans_packed<nk_i8_k, backend_t>("euclideans_packed_i8_ampere", nk_dots_pack_size_i8_ampere,
                                              nk_dots_pack_i8_ampere, nk_euclideans_packed_i8_ampere);
    run_euclideans_packed<nk_i4_k, backend_t>("euclideans_packed_i4_ampere", nk_dots_pack_size_i4_ampere,
                                              nk_dots_pack_i4_ampere, nk_euclideans_packed_i4_ampere);
    run_euclideans_packed<nk_u8_k, backend_t>("euclideans_packed_u8_ampere", nk_dots_pack_size_u8_ampere,
                                              nk_dots_pack_u8_ampere, nk_euclideans_packed_u8_ampere);
    run_euclideans_packed<nk_u4_k, backend_t>("euclideans_packed_u4_ampere", nk_dots_pack_size_u4_ampere,
                                              nk_dots_pack_u4_ampere, nk_euclideans_packed_u4_ampere);

    run_euclideans_symmetric<nk_f64_k, backend_t>("euclideans_symmetric_f64_ampere",
                                                  nk_euclideans_symmetric_f64_ampere);
    run_euclideans_symmetric<nk_f32_k, backend_t>("euclideans_symmetric_f32_ampere",
                                                  nk_euclideans_symmetric_f32_ampere);
    run_euclideans_symmetric<nk_bf16_k, backend_t>("euclideans_symmetric_bf16_ampere",
                                                   nk_euclideans_symmetric_bf16_ampere);
    run_euclideans_symmetric<nk_f16_k, backend_t>("euclideans_symmetric_f16_ampere",
                                                  nk_euclideans_symmetric_f16_ampere);
    run_euclideans_symmetric<nk_e5m2_k, backend_t>("euclideans_symmetric_e5m2_ampere",
                                                   nk_euclideans_symmetric_e5m2_ampere);
    run_euclideans_symmetric<nk_e4m3_k, backend_t>("euclideans_symmetric_e4m3_ampere",
                                                   nk_euclideans_symmetric_e4m3_ampere);
    run_euclideans_symmetric<nk_e3m2_k, backend_t>("euclideans_symmetric_e3m2_ampere",
                                                   nk_euclideans_symmetric_e3m2_ampere);
    run_euclideans_symmetric<nk_e2m3_k, backend_t>("euclideans_symmetric_e2m3_ampere",
                                                   nk_euclideans_symmetric_e2m3_ampere);
    run_euclideans_symmetric<nk_e2m1_k, backend_t>("euclideans_symmetric_e2m1_ampere",
                                                   nk_euclideans_symmetric_e2m1_ampere);
    run_euclideans_symmetric<nk_i8_k, backend_t>("euclideans_symmetric_i8_ampere", nk_euclideans_symmetric_i8_ampere);
    run_euclideans_symmetric<nk_i4_k, backend_t>("euclideans_symmetric_i4_ampere", nk_euclideans_symmetric_i4_ampere);
    run_euclideans_symmetric<nk_u8_k, backend_t>("euclideans_symmetric_u8_ampere", nk_euclideans_symmetric_u8_ampere);
    run_euclideans_symmetric<nk_u4_k, backend_t>("euclideans_symmetric_u4_ampere", nk_euclideans_symmetric_u4_ampere);

    run_attention_bidirectional<nk_bf16_k, backend_t>("attention_bidirectional_packed_bf16_ampere",
                                                      nk_attention_pack_size_bf16_ampere, nk_attention_pack_bf16_ampere,
                                                      nk_attention_bidirectional_packed_bf16_ampere);
    run_attention_causal<nk_bf16_k, backend_t>("attention_causal_packed_bf16_ampere",
                                               nk_attention_pack_size_bf16_ampere, nk_attention_pack_bf16_ampere,
                                               nk_attention_causal_packed_bf16_ampere);
    run_attention_bidirectional<nk_e4m3_k, backend_t>("attention_bidirectional_packed_e4m3_ampere",
                                                      nk_attention_pack_size_e4m3_ampere, nk_attention_pack_e4m3_ampere,
                                                      nk_attention_bidirectional_packed_e4m3_ampere);
    run_attention_causal<nk_e4m3_k, backend_t>("attention_causal_packed_e4m3_ampere",
                                               nk_attention_pack_size_e4m3_ampere, nk_attention_pack_e4m3_ampere,
                                               nk_attention_causal_packed_e4m3_ampere);
    run_attention_bidirectional<nk_i8_k, backend_t>("attention_bidirectional_packed_i8_ampere",
                                                    nk_attention_pack_size_i8_ampere, nk_attention_pack_i8_ampere,
                                                    nk_attention_bidirectional_packed_i8_ampere);
    run_attention_causal<nk_i8_k, backend_t>("attention_causal_packed_i8_ampere", nk_attention_pack_size_i8_ampere,
                                             nk_attention_pack_i8_ampere, nk_attention_causal_packed_i8_ampere);
#endif // NK_TARGET_AMPERE
}

/** Every Blackwell RTX entry point, compiled only when the architecture list includes the family. */
void bench_cross_blackwellrtx() {
#if NK_TARGET_BLACKWELLRTX
    using backend_t = cuda_backend_t;
    run_dots_packed<nk_e5m2_k, backend_t>("dots_packed_e5m2_blackwellrtx", nk_dots_pack_size_e5m2_blackwellrtx,
                                          nk_dots_pack_e5m2_blackwellrtx, nk_dots_packed_e5m2_blackwellrtx);
    run_dots_packed<nk_e4m3_k, backend_t>("dots_packed_e4m3_blackwellrtx", nk_dots_pack_size_e4m3_blackwellrtx,
                                          nk_dots_pack_e4m3_blackwellrtx, nk_dots_packed_e4m3_blackwellrtx);
    run_dots_packed<nk_e3m2_k, backend_t>("dots_packed_e3m2_blackwellrtx", nk_dots_pack_size_e3m2_blackwellrtx,
                                          nk_dots_pack_e3m2_blackwellrtx, nk_dots_packed_e3m2_blackwellrtx);
    run_dots_packed<nk_e2m3_k, backend_t>("dots_packed_e2m3_blackwellrtx", nk_dots_pack_size_e2m3_blackwellrtx,
                                          nk_dots_pack_e2m3_blackwellrtx, nk_dots_packed_e2m3_blackwellrtx);
    run_dots_packed<nk_e2m1_k, backend_t>("dots_packed_e2m1_blackwellrtx", nk_dots_pack_size_e2m1_blackwellrtx,
                                          nk_dots_pack_e2m1_blackwellrtx, nk_dots_packed_e2m1_blackwellrtx);

    run_dots_symmetric<nk_e5m2_k, backend_t>("dots_symmetric_e5m2_blackwellrtx", nk_dots_symmetric_e5m2_blackwellrtx);
    run_dots_symmetric<nk_e4m3_k, backend_t>("dots_symmetric_e4m3_blackwellrtx", nk_dots_symmetric_e4m3_blackwellrtx);
    run_dots_symmetric<nk_e3m2_k, backend_t>("dots_symmetric_e3m2_blackwellrtx", nk_dots_symmetric_e3m2_blackwellrtx);
    run_dots_symmetric<nk_e2m3_k, backend_t>("dots_symmetric_e2m3_blackwellrtx", nk_dots_symmetric_e2m3_blackwellrtx);
    run_dots_symmetric<nk_e2m1_k, backend_t>("dots_symmetric_e2m1_blackwellrtx", nk_dots_symmetric_e2m1_blackwellrtx);

    run_angulars_packed<nk_e5m2_k, backend_t>("angulars_packed_e5m2_blackwellrtx", nk_dots_pack_size_e5m2_blackwellrtx,
                                              nk_dots_pack_e5m2_blackwellrtx, nk_angulars_packed_e5m2_blackwellrtx);
    run_angulars_packed<nk_e4m3_k, backend_t>("angulars_packed_e4m3_blackwellrtx", nk_dots_pack_size_e4m3_blackwellrtx,
                                              nk_dots_pack_e4m3_blackwellrtx, nk_angulars_packed_e4m3_blackwellrtx);
    run_angulars_packed<nk_e3m2_k, backend_t>("angulars_packed_e3m2_blackwellrtx", nk_dots_pack_size_e3m2_blackwellrtx,
                                              nk_dots_pack_e3m2_blackwellrtx, nk_angulars_packed_e3m2_blackwellrtx);
    run_angulars_packed<nk_e2m3_k, backend_t>("angulars_packed_e2m3_blackwellrtx", nk_dots_pack_size_e2m3_blackwellrtx,
                                              nk_dots_pack_e2m3_blackwellrtx, nk_angulars_packed_e2m3_blackwellrtx);
    run_angulars_packed<nk_e2m1_k, backend_t>("angulars_packed_e2m1_blackwellrtx", nk_dots_pack_size_e2m1_blackwellrtx,
                                              nk_dots_pack_e2m1_blackwellrtx, nk_angulars_packed_e2m1_blackwellrtx);

    run_angulars_symmetric<nk_e5m2_k, backend_t>("angulars_symmetric_e5m2_blackwellrtx",
                                                 nk_angulars_symmetric_e5m2_blackwellrtx);
    run_angulars_symmetric<nk_e4m3_k, backend_t>("angulars_symmetric_e4m3_blackwellrtx",
                                                 nk_angulars_symmetric_e4m3_blackwellrtx);
    run_angulars_symmetric<nk_e3m2_k, backend_t>("angulars_symmetric_e3m2_blackwellrtx",
                                                 nk_angulars_symmetric_e3m2_blackwellrtx);
    run_angulars_symmetric<nk_e2m3_k, backend_t>("angulars_symmetric_e2m3_blackwellrtx",
                                                 nk_angulars_symmetric_e2m3_blackwellrtx);
    run_angulars_symmetric<nk_e2m1_k, backend_t>("angulars_symmetric_e2m1_blackwellrtx",
                                                 nk_angulars_symmetric_e2m1_blackwellrtx);

    run_euclideans_packed<nk_e5m2_k, backend_t>("euclideans_packed_e5m2_blackwellrtx",
                                                nk_dots_pack_size_e5m2_blackwellrtx, nk_dots_pack_e5m2_blackwellrtx,
                                                nk_euclideans_packed_e5m2_blackwellrtx);
    run_euclideans_packed<nk_e4m3_k, backend_t>("euclideans_packed_e4m3_blackwellrtx",
                                                nk_dots_pack_size_e4m3_blackwellrtx, nk_dots_pack_e4m3_blackwellrtx,
                                                nk_euclideans_packed_e4m3_blackwellrtx);
    run_euclideans_packed<nk_e3m2_k, backend_t>("euclideans_packed_e3m2_blackwellrtx",
                                                nk_dots_pack_size_e3m2_blackwellrtx, nk_dots_pack_e3m2_blackwellrtx,
                                                nk_euclideans_packed_e3m2_blackwellrtx);
    run_euclideans_packed<nk_e2m3_k, backend_t>("euclideans_packed_e2m3_blackwellrtx",
                                                nk_dots_pack_size_e2m3_blackwellrtx, nk_dots_pack_e2m3_blackwellrtx,
                                                nk_euclideans_packed_e2m3_blackwellrtx);
    run_euclideans_packed<nk_e2m1_k, backend_t>("euclideans_packed_e2m1_blackwellrtx",
                                                nk_dots_pack_size_e2m1_blackwellrtx, nk_dots_pack_e2m1_blackwellrtx,
                                                nk_euclideans_packed_e2m1_blackwellrtx);

    run_euclideans_symmetric<nk_e5m2_k, backend_t>("euclideans_symmetric_e5m2_blackwellrtx",
                                                   nk_euclideans_symmetric_e5m2_blackwellrtx);
    run_euclideans_symmetric<nk_e4m3_k, backend_t>("euclideans_symmetric_e4m3_blackwellrtx",
                                                   nk_euclideans_symmetric_e4m3_blackwellrtx);
    run_euclideans_symmetric<nk_e3m2_k, backend_t>("euclideans_symmetric_e3m2_blackwellrtx",
                                                   nk_euclideans_symmetric_e3m2_blackwellrtx);
    run_euclideans_symmetric<nk_e2m3_k, backend_t>("euclideans_symmetric_e2m3_blackwellrtx",
                                                   nk_euclideans_symmetric_e2m3_blackwellrtx);
    run_euclideans_symmetric<nk_e2m1_k, backend_t>("euclideans_symmetric_e2m1_blackwellrtx",
                                                   nk_euclideans_symmetric_e2m1_blackwellrtx);

    run_attention_bidirectional<nk_e4m3_k, backend_t>(
        "attention_bidirectional_packed_e4m3_blackwellrtx", nk_attention_pack_size_e4m3_blackwellrtx,
        nk_attention_pack_e4m3_blackwellrtx, nk_attention_bidirectional_packed_e4m3_blackwellrtx);
    run_attention_causal<nk_e4m3_k, backend_t>(
        "attention_causal_packed_e4m3_blackwellrtx", nk_attention_pack_size_e4m3_blackwellrtx,
        nk_attention_pack_e4m3_blackwellrtx, nk_attention_causal_packed_e4m3_blackwellrtx);
#endif // NK_TARGET_BLACKWELLRTX
}

#pragma endregion Registrations

#pragma region cuBLASLt

/** How cuBLASLt scales an input: not at all, one UE8M0 exponent per 32 elements, or one UE4M3 per 16. */
enum class cublaslt_scaling_t { unscaled_k, block32_ue8m0_k, block16_ue4m3_k };

/** One cuBLASLt configuration per dtype: storage, accumulation, scalar and output types, and block scaling. */
struct cublaslt_types_t {
    cudaDataType_t input;        ///< A and B storage
    cublasComputeType_t compute; ///< accumulation
    cudaDataType_t scalar;       ///< alpha and beta
    cudaDataType_t output;       ///< C storage
    cublaslt_scaling_t scaling;  ///< block scales applied to A and B
};

template <nk_dtype_t input_dtype_>
constexpr cublaslt_types_t cublaslt_types_k = {};
template <>
constexpr cublaslt_types_t cublaslt_types_k<nk_f64_k> = {CUDA_R_64F, CUBLAS_COMPUTE_64F, CUDA_R_64F, CUDA_R_64F,
                                                         cublaslt_scaling_t::unscaled_k};
template <>
constexpr cublaslt_types_t cublaslt_types_k<nk_f32_k> = {CUDA_R_32F, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F,
                                                         cublaslt_scaling_t::unscaled_k};
template <>
constexpr cublaslt_types_t cublaslt_types_k<nk_bf16_k> = {CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F,
                                                          cublaslt_scaling_t::unscaled_k};
template <>
constexpr cublaslt_types_t cublaslt_types_k<nk_f16_k> = {CUDA_R_16F, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F,
                                                         cublaslt_scaling_t::unscaled_k};
template <>
constexpr cublaslt_types_t cublaslt_types_k<nk_e5m2_k> = {CUDA_R_8F_E5M2, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F,
                                                          cublaslt_scaling_t::unscaled_k};
template <>
constexpr cublaslt_types_t cublaslt_types_k<nk_e4m3_k> = {CUDA_R_8F_E4M3, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F,
                                                          cublaslt_scaling_t::unscaled_k};
template <>
constexpr cublaslt_types_t cublaslt_types_k<nk_e3m2_k> = {CUDA_R_6F_E3M2, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F,
                                                          cublaslt_scaling_t::block32_ue8m0_k};
template <>
constexpr cublaslt_types_t cublaslt_types_k<nk_e2m3_k> = {CUDA_R_6F_E2M3, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F,
                                                          cublaslt_scaling_t::block32_ue8m0_k};
template <>
constexpr cublaslt_types_t cublaslt_types_k<nk_e2m1_k> = {CUDA_R_4F_E2M1, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F,
                                                          cublaslt_scaling_t::block16_ue4m3_k};
template <>
constexpr cublaslt_types_t cublaslt_types_k<nk_i8_k> = {CUDA_R_8I, CUBLAS_COMPUTE_32I, CUDA_R_32I, CUDA_R_32I,
                                                        cublaslt_scaling_t::unscaled_k};
template <>
constexpr cublaslt_types_t cublaslt_types_k<nk_i4_k> = {CUDA_R_4I, CUBLAS_COMPUTE_32I, CUDA_R_32I, CUDA_R_32I,
                                                        cublaslt_scaling_t::unscaled_k};
template <>
constexpr cublaslt_types_t cublaslt_types_k<nk_u8_k> = {CUDA_R_8U, CUBLAS_COMPUTE_32I, CUDA_R_32I, CUDA_R_32I,
                                                        cublaslt_scaling_t::unscaled_k};
template <>
constexpr cublaslt_types_t cublaslt_types_k<nk_u4_k> = {CUDA_R_4U, CUBLAS_COMPUTE_32I, CUDA_R_32I, CUDA_R_32I,
                                                        cublaslt_scaling_t::unscaled_k};

/** How the operands reach cuBLASLt: as stored, or widened to F64 on the host first. */
enum class cublaslt_operands_t { native_k, widened_f64_k };

/**
 *  @brief One planned `cublasLtMatmul` for row-major C = A × Bᵀ.
 *
 *  Row-major C is column-major Cᵀ = B × Aᵀ, so B goes in as the transposed first operand and A as the second.
 *  Leading dimensions count elements, so a nibble-pair row of K elements has a leading dimension of K.
 */
struct cublaslt_plan_t {
    cublasLtHandle_t handle = nullptr;              ///< library context
    cublasLtMatmulDesc_t operation = nullptr;       ///< compute type, transposes and scale modes
    cublasLtMatrixLayout_t first_layout = nullptr;  ///< B, transposed
    cublasLtMatrixLayout_t second_layout = nullptr; ///< A
    cublasLtMatrixLayout_t output_layout = nullptr; ///< C
    cublasLtMatmulHeuristicResult_t heuristic {};   ///< the chosen algorithm
    device_vector<char> workspace;                  ///< scratch for the algorithm
    device_vector<char> first_scales;               ///< unit block scales of B
    device_vector<char> second_scales;              ///< unit block scales of A
    cudaDataType_t scalar_type = CUDA_R_32F;        ///< type of alpha and beta
    std::string failure;                            ///< why `build` failed

    ~cublaslt_plan_t() {
        if (output_layout) cublasLtMatrixLayoutDestroy(output_layout);
        if (second_layout) cublasLtMatrixLayoutDestroy(second_layout);
        if (first_layout) cublasLtMatrixLayoutDestroy(first_layout);
        if (operation) cublasLtMatmulDescDestroy(operation);
        if (handle) cublasLtDestroy(handle);
    }

    /** Builds the plan, or leaves the reason in `failure` when cuBLASLt offers no algorithm. */
    bool build(cublaslt_types_t types, std::size_t height, std::size_t width, std::size_t depth,
               std::size_t a_leading) {
        auto fails = [&](char const *step, cublasStatus_t status) {
            failure = std::string(step) + ": " + cublasLtGetStatusName(status);
            return false;
        };
        scalar_type = types.scalar;
        cublasStatus_t status;
        if ((status = cublasLtCreate(&handle)) != CUBLAS_STATUS_SUCCESS) return fails("cublasLtCreate", status);
        if ((status = cublasLtMatmulDescCreate(&operation, types.compute, types.scalar)) != CUBLAS_STATUS_SUCCESS)
            return fails("cublasLtMatmulDescCreate", status);
        cublasOperation_t const transposed = CUBLAS_OP_T, kept = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA, &transposed, sizeof(transposed));
        cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSB, &kept, sizeof(kept));

        if (types.scaling != cublaslt_scaling_t::unscaled_k) {
            // Unit scales - UE8M0 code 127 is 2⁰, UE4M3 code 0x38 is 1.0 - so the product is the unscaled one.
            bool const ue8m0 = types.scaling == cublaslt_scaling_t::block32_ue8m0_k;
            std::size_t const block = ue8m0 ? 32 : 16;
            int const unit_code = ue8m0 ? 127 : 0x38;
            auto scales_bytes = [&](std::size_t outer) {
                return nk::divide_round_up(outer, std::size_t(128)) * 128 *
                       nk::divide_round_up(nk::divide_round_up(depth, block), std::size_t(4)) * 4;
            };
            first_scales = device_vector<char>::try_empty(scales_bytes(width));
            second_scales = device_vector<char>::try_empty(scales_bytes(height));
            if (first_scales.empty() || second_scales.empty())
                return fails("scale allocation", CUBLAS_STATUS_ALLOC_FAILED);
            cudaMemset(first_scales.raw_values_data(), unit_code, first_scales.size_bytes());
            cudaMemset(second_scales.raw_values_data(), unit_code, second_scales.size_bytes());
            void const *first_scales_address = first_scales.raw_values_data();
            void const *second_scales_address = second_scales.raw_values_data();
            cublasLtMatmulMatrixScale_t const mode = ue8m0 ? CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0
                                                           : CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
            cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode));
            cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode));
            cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &first_scales_address,
                                           sizeof(first_scales_address));
            cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &second_scales_address,
                                           sizeof(second_scales_address));
        }

        if ((status = cublasLtMatrixLayoutCreate(&first_layout, types.input, depth, width, depth)) !=
            CUBLAS_STATUS_SUCCESS)
            return fails("first layout", status);
        if ((status = cublasLtMatrixLayoutCreate(&second_layout, types.input, depth, height, a_leading)) !=
            CUBLAS_STATUS_SUCCESS)
            return fails("second layout", status);
        if ((status = cublasLtMatrixLayoutCreate(&output_layout, types.output, width, height, width)) !=
            CUBLAS_STATUS_SUCCESS)
            return fails("output layout", status);

        std::size_t const workspace_bytes = std::size_t(64) << 20;
        workspace = device_vector<char>::try_empty(workspace_bytes);
        if (workspace.empty()) return fails("workspace allocation", CUBLAS_STATUS_ALLOC_FAILED);
        cublasLtMatmulPreference_t preference = nullptr;
        cublasLtMatmulPreferenceCreate(&preference);
        cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes,
                                             sizeof(workspace_bytes));
        int found = 0;
        status = cublasLtMatmulAlgoGetHeuristic(handle, operation, first_layout, second_layout, output_layout,
                                                output_layout, preference, 1, &heuristic, &found);
        cublasLtMatmulPreferenceDestroy(preference);
        if (status != CUBLAS_STATUS_SUCCESS) return fails("cublasLtMatmulAlgoGetHeuristic", status);
        if (found == 0) return fails("cublasLtMatmulAlgoGetHeuristic", CUBLAS_STATUS_NOT_SUPPORTED);
        return true;
    }

    /** Enqueues C = A × Bᵀ on @p stream. */
    cudaError_t launch(void const *a, void const *b, void *c, cudaStream_t stream) noexcept {
        float const alpha_f32 = 1, beta_f32 = 0;
        double const alpha_f64 = 1, beta_f64 = 0;
        std::int32_t const alpha_i32 = 1, beta_i32 = 0;
        void const *alpha = &alpha_f32, *beta = &beta_f32;
        if (scalar_type == CUDA_R_64F) alpha = &alpha_f64, beta = &beta_f64;
        if (scalar_type == CUDA_R_32I) alpha = &alpha_i32, beta = &beta_i32;
        cublasStatus_t const status = cublasLtMatmul(handle, operation, alpha, b, first_layout, a, second_layout, beta,
                                                     c, output_layout, c, output_layout, &heuristic.algo,
                                                     workspace.raw_values_data(), workspace.size_bytes(), stream);
        return status == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cudaErrorUnknown;
    }
};

/** A's leading dimension in elements: the backend's padded stride natively, dense rows once widened. */
template <nk_dtype_t input_dtype_>
std::size_t cublaslt_a_leading(cublaslt_operands_t operands, std::size_t depth) {
    using input_t = typename nk::type_for<input_dtype_>::type;
    if (operands == cublaslt_operands_t::widened_f64_k) return depth;
    std::size_t const row_bytes = nk::divide_round_up(depth, nk::dimensions_per_value<input_t>()) *
                                  sizeof(typename input_t::raw_t);
    return cuda_backend_t {}.row_stride(row_bytes) / sizeof(typename input_t::raw_t) *
           nk::dimensions_per_value<input_t>();
}

/** Rows of @p rows decoded into dense F64 rows, the operands a widened cuBLASLt row multiplies. */
template <nk_dtype_t input_dtype_>
std::vector<double> decode_rows(nk::tensor_view<typename nk::type_for<input_dtype_>::type, 2> rows, std::size_t depth) {
    std::vector<double> decoded(rows.extent(0) * depth);
    for (std::size_t row = 0; row != rows.extent(0); ++row)
        nk_cast_serial(rows.byte_data() + row * rows.stride_bytes(0), input_dtype_, depth, decoded.data() + row * depth,
                       nk_f64_k);
    return decoded;
}

template <nk_dtype_t input_dtype_, typename output_type_>
void measure_dots_with_cublaslt(bm::State &state, cublaslt_types_t types, cublaslt_operands_t operands,
                                std::size_t height, std::size_t width, std::size_t depth) {
    cuda_backend_t backend;
    auto const [a, b] = random_matrices<input_dtype_, cuda_backend_t>(height, width, depth);
    std::size_t const a_bytes = height * a.stride_bytes(0), b_bytes = width * b.stride_bytes(0);
    cublaslt_plan_t plan;
    if (!plan.build(types, height, width, depth, cublaslt_a_leading<input_dtype_>(operands, depth)))
        return state.SkipWithError(plan.failure.c_str());

    std::vector<matrix_set<cuda_backend_t, output_type_>> sets;
    std::size_t const c_bytes = height * width * sizeof(typename output_type_::raw_t);
    bool uploaded = false;
    if (operands == cublaslt_operands_t::widened_f64_k) {
        std::vector<double> const a_widened = decode_rows<input_dtype_>(a.view(), depth);
        std::vector<double> const b_widened = decode_rows<input_dtype_>(b.view(), depth);
        std::size_t const a_widened_bytes = a_widened.size() * sizeof(double),
                          b_widened_bytes = b_widened.size() * sizeof(double);
        sets.resize(backend.input_sets(a_widened_bytes + b_widened_bytes + c_bytes));
        uploaded = upload_unpacked_sets(backend, sets, a_widened.data(), a_widened_bytes, b_widened.data(),
                                        b_widened_bytes, height * width);
    }
    else {
        sets.resize(backend.input_sets(a_bytes + b_bytes + c_bytes));
        uploaded = upload_unpacked_sets(backend, sets, a.data(), a_bytes, b.data(), b_bytes, height * width);
    }
    if (!uploaded) return state.SkipWithError("set allocation failed");
    std::size_t const calls = time_rotating(state, backend, sets.size(), [&](std::size_t index) {
        auto &set = sets[index];
        backend.call([&](cudaStream_t stream) {
            return plan.launch(set.a.raw_values_data(), set.b.raw_values_data(), set.c.raw_values_data(), stream);
        });
    });
    if (!calls) return;
    double const score = sampled_accuracy<input_dtype_, output_type_>(
        backend, sets[0].c, a.view(), b.view(), depth, reference_metric_t::dot_k, written_entries_t::full_k);
    report_matrix<output_type_>(state, calls, 2.0 * height * width * depth, score);
}

/** Registers the cuBLASLt row, or prints once why cuBLASLt has no algorithm for this configuration. */
template <nk_dtype_t input_dtype_, typename output_type_ = typename nk::type_for<input_dtype_>::type::dot_result_t>
void run_dots_with_cublaslt(std::string const &name, cublaslt_types_t types = cublaslt_types_k<input_dtype_>,
                            cublaslt_operands_t operands = cublaslt_operands_t::native_k) {
    std::size_t const height = bench_config.matrix_height, width = bench_config.matrix_width,
                      depth = bench_config.matrix_depth;
    cublaslt_plan_t probe;
    if (!probe.build(types, height, width, depth, cublaslt_a_leading<input_dtype_>(operands, depth))) {
        std::printf("  Skipping %s: %s\n", name.c_str(), probe.failure.c_str());
        return;
    }
    cuda_backend_t::configure(bm::RegisterBenchmark(matrix_row_name(name, height, width, depth).c_str(),
                                                    measure_dots_with_cublaslt<input_dtype_, output_type_>, types,
                                                    operands, height, width, depth));
}

#pragma endregion cuBLASLt

#pragma region cuBLAS

/**
 *  @brief DGEMM through cuBLAS's fixed-point emulation on the integer tensor cores, with the mantissa width cuBLAS
 *      derives to match F64 accuracy.
 *
 *  Reached through the handle API: on compute capability 12.0, cuBLASLt accepts the same emulated compute type and
 *  emulation descriptor, then runs native F64.
 */
void measure_dots_f64_emulated_with_cublas(bm::State &state, std::size_t height, std::size_t width, std::size_t depth) {
    cuda_backend_t backend;
    auto const [a, b] = random_matrices<nk_f64_k, cuda_backend_t>(height, width, depth);
    std::size_t const a_bytes = height * a.stride_bytes(0), b_bytes = width * b.stride_bytes(0);
    std::vector<matrix_set<cuda_backend_t, nk::f64_t>> sets(
        backend.input_sets(a_bytes + b_bytes + height * width * sizeof(nk_f64_t)));
    if (!upload_unpacked_sets(backend, sets, a.data(), a_bytes, b.data(), b_bytes, height * width))
        return state.SkipWithError("set allocation failed");
    cublasHandle_t handle = nullptr;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return state.SkipWithError("cublasCreate failed");
    cublasSetStream(handle, backend.stream);
    cublasSetEmulationStrategy(handle, CUBLAS_EMULATION_STRATEGY_EAGER);
    cublasSetMathMode(handle, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH);
    int const a_leading = int(a.stride_bytes(0) / sizeof(nk_f64_t));
    std::size_t const calls = time_rotating(state, backend, sets.size(), [&](std::size_t index) {
        auto &set = sets[index];
        backend.call([&](cudaStream_t) {
            double const alpha = 1, beta = 0;
            cublasStatus_t const status = cublasGemmEx(
                handle, CUBLAS_OP_T, CUBLAS_OP_N, int(width), int(height), int(depth), &alpha, set.b.raw_values_data(),
                CUDA_R_64F, int(depth), set.a.raw_values_data(), CUDA_R_64F, a_leading, &beta, set.c.raw_values_data(),
                CUDA_R_64F, int(width), CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, CUBLAS_GEMM_DEFAULT);
            return status == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cudaErrorUnknown;
        });
    });
    if (calls) {
        double const score = sampled_accuracy<nk_f64_k, nk::f64_t>(
            backend, sets[0].c, a.view(), b.view(), depth, reference_metric_t::dot_k, written_entries_t::full_k);
        report_matrix<nk::f64_t>(state, calls, 2.0 * height * width * depth, score);
    }
    cublasDestroy(handle);
}

#pragma endregion cuBLAS

int run_benchmarks(int argc, char **argv) {
    auto parse_size = [](char const *name, std::size_t &value) {
        if (char const *text = std::getenv(name))
            if (std::size_t const parsed = static_cast<std::size_t>(std::atoll(text)); parsed > 0) value = parsed;
    };
    parse_size("NK_MATRIX_HEIGHT", bench_config.matrix_height);
    parse_size("NK_MATRIX_WIDTH", bench_config.matrix_width);
    parse_size("NK_MATRIX_DEPTH", bench_config.matrix_depth);
    if (char const *text = std::getenv("NK_SEED")) bench_config.seed = static_cast<std::uint32_t>(std::atoll(text));
    std::size_t budget_mb = 0;
    parse_size("NK_BUDGET_MB", budget_mb);
    if (budget_mb) bench_config.budget_bytes = budget_mb << 20;

    cudaDeviceProp properties {};
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        std::printf("No usable CUDA device\n");
        return 1;
    }
    nk_capability_t const capabilities = nk_capabilities_cuda_available(device);
    std::printf("NumKong CUDA Benchmarks v%d.%d.%d\n", NK_VERSION_MAJOR, NK_VERSION_MINOR, NK_VERSION_PATCH);
    std::printf("  Device: %s, compute capability %d.%d, %d SMs, %d MB L2\n", properties.name, properties.major,
                properties.minor, properties.multiProcessorCount, properties.l2CacheSize >> 20);
    std::printf("  Families: ampere %d, hopper %d, blackwell %d, blackwellrtx %d, cuBLASLt %zu\n\n",
                (capabilities & nk_cap_ampere_k) != 0, (capabilities & nk_cap_hopper_k) != 0,
                (capabilities & nk_cap_blackwell_k) != 0, (capabilities & nk_cap_blackwellrtx_k) != 0,
                cublasLtGetVersion());

    std::vector<std::string> arguments(argv, argv + argc);
    bool user_set_min_time = false;
    for (auto const &argument : arguments)
        if (argument.rfind("--benchmark_min_time", 0) == 0) user_set_min_time = true;
    if (char const *filter = std::getenv("NK_FILTER")) arguments.push_back(std::string("--benchmark_filter=") + filter);
    if (!user_set_min_time) {
        char const *seconds = std::getenv("NK_BUDGET_SECS");
        arguments.push_back(std::string("--benchmark_min_time=") + (seconds ? seconds : "10") + "s");
    }
    std::vector<char *> argument_pointers;
    for (auto &argument : arguments) argument_pointers.push_back(argument.data());
    int arguments_count = static_cast<int>(argument_pointers.size());
    bm::Initialize(&arguments_count, argument_pointers.data());
    if (bm::ReportUnrecognizedArguments(arguments_count, argument_pointers.data())) return 1;

    if (capabilities & nk_cap_ampere_k) bench_cross_ampere();
    if (capabilities & nk_cap_blackwellrtx_k) bench_cross_blackwellrtx();

    run_dots_with_cublaslt<nk_f64_k>("dots_packed_f64_with_cublaslt");
    cuda_backend_t::configure(
        bm::RegisterBenchmark(matrix_row_name("dots_packed_f64_emulated_with_cublas", bench_config.matrix_height,
                                              bench_config.matrix_width, bench_config.matrix_depth)
                                  .c_str(),
                              measure_dots_f64_emulated_with_cublas, bench_config.matrix_height,
                              bench_config.matrix_width, bench_config.matrix_depth));
    run_dots_with_cublaslt<nk_f32_k, nk::f32_t>("dots_packed_f32_with_cublaslt");
    run_dots_with_cublaslt<nk_f32_k, nk::f64_t>("dots_packed_f32_widened_with_cublaslt", cublaslt_types_k<nk_f64_k>,
                                                cublaslt_operands_t::widened_f64_k);
    run_dots_with_cublaslt<nk_bf16_k>("dots_packed_bf16_with_cublaslt");
    run_dots_with_cublaslt<nk_f16_k>("dots_packed_f16_with_cublaslt");
    run_dots_with_cublaslt<nk_e5m2_k>("dots_packed_e5m2_with_cublaslt");
    run_dots_with_cublaslt<nk_e4m3_k>("dots_packed_e4m3_with_cublaslt");
    run_dots_with_cublaslt<nk_e3m2_k>("dots_packed_e3m2_with_cublaslt");
    run_dots_with_cublaslt<nk_e2m3_k>("dots_packed_e2m3_with_cublaslt");
    run_dots_with_cublaslt<nk_e2m1_k>("dots_packed_e2m1_with_cublaslt");
    run_dots_with_cublaslt<nk_i8_k>("dots_packed_i8_with_cublaslt");
    run_dots_with_cublaslt<nk_i4_k>("dots_packed_i4_with_cublaslt");
    run_dots_with_cublaslt<nk_u8_k>("dots_packed_u8_with_cublaslt");
    run_dots_with_cublaslt<nk_u4_k>("dots_packed_u4_with_cublaslt");

    bm::RunSpecifiedBenchmarks();
    bm::Shutdown();
    return 0;
}

int main(int argc, char **argv) { return run_benchmarks(argc, argv); }
