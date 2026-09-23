/**
 *  @brief Batch operation benchmarks - CUDA kernels against cuBLASLt, cuBLAS, cuDNN and cuVS.
 *  @file bench/cross_cuda.cu
 *  @author Ash Vardanian
 *  @date September 22, 2026
 *
 *  Runs the drivers of `cross.cuh` through the shared `cuda_backend_t` over device-resident operands, launching
 *  on `cudaStreamPerThread`, with timed windows of launches bracketed by CUDA events and reported through
 *  `UseManualTime`, so launch latency and host synchronization stay outside the measurement. The Time column is per
 *  window, and the `calls` counter recovers the per-call rate. Input sets rotate until their footprint is at least
 *  twice the L2.
 *
 *  The `attention` rows time prefill, 4096 queries on 4096 keys, and decode, 1 query on 4096 keys, with 32 query heads
 *  over 8 K and V heads of depth 128. Every baseline enters the same drivers as a kernel callable, so it shares their
 *  inputs, timing and counters, and compiles in only under its `NK_COMPARE_TO_*` CMake option.
 */

#include <cmath>   // `std::sqrt`, `INFINITY`
#include <cstdint> // `std::int32_t`, `std::int64_t`
#include <cstdio>  // `std::printf`
#include <cstring> // `std::memcpy`

#include <algorithm>   // `std::clamp`, `std::max`
#include <array>       // `std::array`
#include <bit>         // `std::bit_ceil`, `std::bit_floor`
#include <memory>      // `std::shared_ptr`
#include <string>      // `std::string`
#include <type_traits> // `std::remove_pointer_t`
#include <utility>     // `std::pair`
#include <vector>      // `std::vector`

#if NK_COMPARE_TO_CUBLAS
#include <cublasLt.h>
#include <cublas_v2.h>
#endif
#include <cuda_runtime.h>
#if NK_COMPARE_TO_CUDNN
#include <cudnn.h>
#endif
#if NK_COMPARE_TO_CUVS
#include <cuvs/core/c_api.h>
#include <cuvs/distance/pairwise_distance.h>
#endif

#include "numkong/numkong.h"

#include "../test/test.cuh" // `test::cuda_backend_t`, `device_vector`
#include "cross.cuh"

using namespace ashvardanian::numkong::bench;
using nk::test::device_vector;
using nk::test::print_indicator;
using nk::test::print_isa;

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

/** Prints once why a baseline row is missing. */
void print_skipped(std::string const &name, char const *reason) {
    std::printf("  Skipping %s: %s\n", name.c_str(), reason);
}

/** Registers a baseline @p kernel over dense B rows through `register_packed`, with a copy of B as its pack. */
template <nk_dtype_t input_dtype_, typename output_type_, typename kernel_type_>
void register_unpacked(std::string const &name, reference_metric_t metric, kernel_type_ kernel) {
    using input_t = typename nk::type_for<input_dtype_>::type;
    auto const packed_size = [](std::size_t width, std::size_t depth) {
        return width * nk::divide_round_up(depth, nk::dimensions_per_value<input_t>()) * sizeof(input_t);
    };
    auto const copy = [](void const *b, std::size_t width, std::size_t, std::size_t row_bytes, void *packed,
                         std::size_t, std::size_t, cudaStream_t stream) {
        return cudaMemcpyAsync(packed, b, width * row_bytes, cudaMemcpyDeviceToDevice, stream);
    };
    register_packed<input_dtype_, output_type_, cuda_backend_t>(name, metric, packed_size, copy, kernel);
}

#pragma endregion CUDA Backend

#pragma region Registrations

/** Every Ampere entry point, compiled only when the architecture list includes the family. */
void bench_cross_ampere([[maybe_unused]] nk_capability_t available) {
#if NK_TARGET_AMPERE
    if (!(available & nk_cap_ampere_k)) return;
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
void bench_cross_blackwellrtx([[maybe_unused]] nk_capability_t available) {
#if NK_TARGET_BLACKWELLRTX
    if (!(available & nk_cap_blackwellrtx_k)) return;
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

#pragma region cuBLAS
#if NK_COMPARE_TO_CUBLAS

/** cuBLASLt's storage type for @p dtype. */
cudaDataType_t cublaslt_input_type(nk_dtype_t dtype) noexcept {
    switch (dtype) {
    case nk_f64_k: return CUDA_R_64F;
    case nk_f32_k: return CUDA_R_32F;
    case nk_bf16_k: return CUDA_R_16BF;
    case nk_f16_k: return CUDA_R_16F;
    case nk_e5m2_k: return CUDA_R_8F_E5M2;
    case nk_e4m3_k: return CUDA_R_8F_E4M3;
    case nk_e3m2_k: return CUDA_R_6F_E3M2;
    case nk_e2m3_k: return CUDA_R_6F_E2M3;
    case nk_e2m1_k: return CUDA_R_4F_E2M1;
    case nk_i8_k: return CUDA_R_8I;
    case nk_i4_k: return CUDA_R_4I;
    case nk_u8_k: return CUDA_R_8U;
    default: return CUDA_R_4U;
    }
}

/** cuBLASLt's accumulator, scalar and output type for @p dtype: F64, I32 for integers, F32 for other floats. */
cudaDataType_t cublaslt_output_type(nk_dtype_t dtype) noexcept {
    switch (dtype) {
    case nk_f64_k: return CUDA_R_64F;
    case nk_i8_k:
    case nk_i4_k:
    case nk_u8_k:
    case nk_u4_k: return CUDA_R_32I;
    default: return CUDA_R_32F;
    }
}

/** Elements per block scale cuBLASLt requires of @p dtype: 32 for 6-bit floats, 16 for 4-bit ones, 0 for none. */
std::size_t cublaslt_scale_block(nk_dtype_t dtype) noexcept {
    switch (dtype) {
    case nk_e3m2_k:
    case nk_e2m3_k: return 32;
    case nk_e2m1_k: return 16;
    default: return 0;
    }
}

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
    device_vector<char> workspace;                  ///< scratch the algorithm asked for
    device_vector<char> scales;                     ///< unit block scales, shared by A and B
    cudaDataType_t output_type = CUDA_R_32F;        ///< type of alpha, beta and C

    ~cublaslt_plan_t() {
        if (output_layout) cublasLtMatrixLayoutDestroy(output_layout);
        if (second_layout) cublasLtMatrixLayoutDestroy(second_layout);
        if (first_layout) cublasLtMatrixLayoutDestroy(first_layout);
        if (operation) cublasLtMatmulDescDestroy(operation);
        if (handle) cublasLtDestroy(handle);
    }

    /** Builds the plan for A of @p a_leading elements per row, or returns why cuBLASLt offers no algorithm. */
    cublasStatus_t build(nk_dtype_t dtype, std::size_t height, std::size_t width, std::size_t depth,
                         std::size_t a_leading) {
        cudaDataType_t const input_type = cublaslt_input_type(dtype);
        output_type = cublaslt_output_type(dtype);
        cublasComputeType_t const compute = output_type == CUDA_R_64F   ? CUBLAS_COMPUTE_64F
                                            : output_type == CUDA_R_32I ? CUBLAS_COMPUTE_32I
                                                                        : CUBLAS_COMPUTE_32F;
        cublasStatus_t status = cublasLtCreate(&handle);
        if (!status) status = cublasLtMatmulDescCreate(&operation, compute, output_type);
        if (!status) status = cublasLtMatrixLayoutCreate(&first_layout, input_type, depth, width, depth);
        if (!status) status = cublasLtMatrixLayoutCreate(&second_layout, input_type, depth, height, a_leading);
        if (!status) status = cublasLtMatrixLayoutCreate(&output_layout, output_type, width, height, width);
        if (status) return status;
        cublasOperation_t const transposed = CUBLAS_OP_T;
        cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA, &transposed, sizeof(transposed));

        // Unit scales - UE8M0 code 127 is 2⁰, UE4M3 code 0x38 is 1.0 - so the product is the unscaled one.
        if (std::size_t const block = cublaslt_scale_block(dtype)) {
            scales = device_vector<char>::try_empty(nk::divide_round_up(std::max(height, width), std::size_t(128)) *
                                                    128 * nk::divide_round_up(depth, block * 4) * 4);
            if (scales.empty()) return CUBLAS_STATUS_ALLOC_FAILED;
            cudaMemset(scales.raw_values_data(), block == 32 ? 127 : 0x38, scales.size_bytes());
            void const *scales_address = scales.raw_values_data();
            cublasLtMatmulMatrixScale_t const mode = block == 32 ? CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0
                                                                 : CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
            for (cublasLtMatmulDescAttributes_t attribute :
                 {CUBLASLT_MATMUL_DESC_A_SCALE_MODE, CUBLASLT_MATMUL_DESC_B_SCALE_MODE})
                cublasLtMatmulDescSetAttribute(operation, attribute, &mode, sizeof(mode));
            for (cublasLtMatmulDescAttributes_t attribute :
                 {CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER})
                cublasLtMatmulDescSetAttribute(operation, attribute, &scales_address, sizeof(scales_address));
        }

        std::size_t const workspace_limit = std::size_t(64) << 20;
        cublasLtMatmulPreference_t preference = nullptr;
        cublasLtMatmulPreferenceCreate(&preference);
        cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_limit,
                                             sizeof(workspace_limit));
        int found = 0;
        status = cublasLtMatmulAlgoGetHeuristic(handle, operation, first_layout, second_layout, output_layout,
                                                output_layout, preference, 1, &heuristic, &found);
        cublasLtMatmulPreferenceDestroy(preference);
        if (status || !found) return status ? status : CUBLAS_STATUS_NOT_SUPPORTED;
        workspace = device_vector<char>::try_empty(std::max<std::size_t>(heuristic.workspaceSize, 1));
        return workspace.empty() ? CUBLAS_STATUS_ALLOC_FAILED : CUBLAS_STATUS_SUCCESS;
    }

    /** Enqueues C = A × Bᵀ on @p stream. */
    cudaError_t launch(void const *a, void const *b, void *c, cudaStream_t stream) noexcept {
        double const alpha_f64 = 1, beta_f64 = 0;
        float const alpha_f32 = 1, beta_f32 = 0;
        std::int32_t const alpha_i32 = 1, beta_i32 = 0;
        void const *alpha = &alpha_f32, *beta = &beta_f32;
        if (output_type == CUDA_R_64F) alpha = &alpha_f64, beta = &beta_f64;
        if (output_type == CUDA_R_32I) alpha = &alpha_i32, beta = &beta_i32;
        cublasStatus_t const status = cublasLtMatmul(handle, operation, alpha, b, first_layout, a, second_layout, beta,
                                                     c, output_layout, c, output_layout, &heuristic.algo,
                                                     workspace.raw_values_data(), workspace.size_bytes(), stream);
        return status == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cudaErrorUnknown;
    }
};

/** Registers a cuBLASLt row, or prints why cuBLASLt has no algorithm for it. */
template <nk_dtype_t input_dtype_, typename output_type_ = typename nk::type_for<input_dtype_>::type::dot_result_t>
void register_dots_with_cublaslt(std::string const &name) {
    using input_t = typename nk::type_for<input_dtype_>::type;
    std::size_t const dimensions_per_value = nk::dimensions_per_value<input_t>();
    std::size_t const a_row_bytes = nk::divide_round_up(bench_config.matrix_depth, dimensions_per_value) *
                                    sizeof(input_t);
    auto const plan = std::make_shared<cublaslt_plan_t>();
    if (cublasStatus_t const status = plan->build(
            input_dtype_, bench_config.matrix_height, bench_config.matrix_width, bench_config.matrix_depth,
            cuda_backend_t::row_stride(a_row_bytes) / sizeof(input_t) * dimensions_per_value))
        return print_skipped(name, cublasLtGetStatusName(status));
    register_unpacked<input_dtype_, output_type_>(
        name, reference_metric_t::dot_k,
        [plan](void const *a, void const *b, void *c, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t,
               cudaStream_t stream) { return plan->launch(a, b, c, stream); });
}

/** Registers DGEMM through cuBLAS's fixed-point emulation, which only the handle API runs on 12.x devices. */
void register_dots_f64_with_cublas(std::string const &name) {
    cublasHandle_t raw_handle = nullptr;
    if (cublasStatus_t const status = cublasCreate(&raw_handle))
        return print_skipped(name, cublasGetStatusName(status));
    std::shared_ptr<std::remove_pointer_t<cublasHandle_t>> const handle(raw_handle, cublasDestroy);
    cublasSetEmulationStrategy(raw_handle, CUBLAS_EMULATION_STRATEGY_EAGER);
    cublasSetMathMode(raw_handle, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH);
    register_unpacked<nk_f64_k, nk::f64_t>(
        name, reference_metric_t::dot_k,
        [handle](void const *a, void const *b, void *c, std::size_t height, std::size_t width, std::size_t depth,
                 std::size_t a_stride, std::size_t, cudaStream_t stream) {
            double const alpha = 1, beta = 0;
            cublasSetStream(handle.get(), stream);
            cublasStatus_t const status = cublasGemmEx(handle.get(), CUBLAS_OP_T, CUBLAS_OP_N, int(width), int(height),
                                                       int(depth), &alpha, b, CUDA_R_64F, int(depth), a, CUDA_R_64F,
                                                       int(a_stride / sizeof(double)), &beta, c, CUDA_R_64F, int(width),
                                                       CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, CUBLAS_GEMM_DEFAULT);
            return status == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cudaErrorUnknown;
        });
}

#endif // NK_COMPARE_TO_CUBLAS

/** Every cuBLASLt row, one per dtype, and cuBLAS's emulated DGEMM. */
void bench_cross_cublas() {
#if NK_COMPARE_TO_CUBLAS
    register_dots_with_cublaslt<nk_f64_k>("dots_packed_f64_with_cublaslt");
    register_dots_f64_with_cublas("dots_packed_f64_with_cublas");
    register_dots_with_cublaslt<nk_f32_k, nk::f32_t>("dots_packed_f32_with_cublaslt");
    register_dots_with_cublaslt<nk_bf16_k>("dots_packed_bf16_with_cublaslt");
    register_dots_with_cublaslt<nk_f16_k>("dots_packed_f16_with_cublaslt");
    register_dots_with_cublaslt<nk_e5m2_k>("dots_packed_e5m2_with_cublaslt");
    register_dots_with_cublaslt<nk_e4m3_k>("dots_packed_e4m3_with_cublaslt");
    register_dots_with_cublaslt<nk_e3m2_k>("dots_packed_e3m2_with_cublaslt");
    register_dots_with_cublaslt<nk_e2m3_k>("dots_packed_e2m3_with_cublaslt");
    register_dots_with_cublaslt<nk_e2m1_k>("dots_packed_e2m1_with_cublaslt");
    register_dots_with_cublaslt<nk_i8_k>("dots_packed_i8_with_cublaslt");
    register_dots_with_cublaslt<nk_i4_k>("dots_packed_i4_with_cublaslt");
    register_dots_with_cublaslt<nk_u8_k>("dots_packed_u8_with_cublaslt");
    register_dots_with_cublaslt<nk_u4_k>("dots_packed_u4_with_cublaslt");
#endif // NK_COMPARE_TO_CUBLAS
}
#pragma endregion cuBLAS

#pragma region cuDNN
#if NK_COMPARE_TO_CUDNN

/** cuDNN's storage type for Q, K, V and O of @p dtype. */
cudnnDataType_t cudnn_data_type(nk_dtype_t dtype) noexcept {
    switch (dtype) {
    case nk_bf16_k: return CUDNN_DATA_BFLOAT16;
    case nk_e4m3_k: return CUDNN_DATA_FP8_E4M3;
    default: return CUDNN_DATA_FLOAT;
    }
}

/** Where a graph tensor lives: in device memory, in a host scalar passed by value, or only between operations. */
enum class cudnn_binding_t { device_k, host_k, virtual_k };

/** Graph tensor identifiers; from `query_length_k` on, each also indexes the 32-bit words of `parameters`. */
enum class cudnn_uid_t : std::int64_t {
    queries_k = 1,
    keys_k,
    values_k,
    output_k,
    scale_k,
    negative_infinity_k,
    window_k,
    scores_k,
    causal_scores_k,
    window_scores_k,
    query_length_k,
    key_length_k,
    query_offsets_k,
    key_offsets_k = query_offsets_k + 2,
    descale_queries_k = key_offsets_k + 2,
    descale_keys_k,
    descale_values_k,
    descale_probabilities_k,
    scale_probabilities_k,
    scale_output_k,
    amax_probabilities_k,
    amax_output_k,
    end_k,
};

/**
 *  @brief One cuDNN SDPA forward plan over a ragged segment of queries at the end of a key cache.
 *
 *  Query heads are grouped over K and V heads, and causal masks are a bottom-right diagonal-band subgraph.
 *  E4M3 plans quantize probabilities with a scale of 256, so the 1/4096-sized weights of a flat softmax stay normal.
 */
struct cudnn_attention_plan_t {
    cudnnHandle_t handle = nullptr;                    ///< library context, bound to the per-thread stream
    std::vector<cudnnBackendDescriptor_t> descriptors; ///< every descriptor built, destroyed in reverse
    cudnnBackendDescriptor_t plan = nullptr;           ///< the finalized execution plan
    std::vector<std::int64_t> uids;                    ///< variant-pack identifiers, Q, K, V and O first
    std::vector<void *> addresses;                     ///< variant-pack addresses matching `uids`
    device_vector<std::uint32_t> parameters;           ///< lengths, ragged offsets, E4M3 scales and maxima
    device_vector<char> workspace;                     ///< scratch for the plan
    float scale = 0;                                   ///< softmax scale, passed by value
    float negative_infinity = -INFINITY;               ///< what masked scores become, passed by value
    std::int32_t window = 0;                           ///< visible keys per query of a windowed row, passed by value
    std::size_t key_bytes = 0;                         ///< bytes of K, and of V after it in the packed cache
    cudnnStatus_t status = CUDNN_STATUS_SUCCESS;       ///< first failure while building

    ~cudnn_attention_plan_t() {
        for (auto descriptor = descriptors.rbegin(); descriptor != descriptors.rend(); ++descriptor)
            cudnnBackendDestroyDescriptor(*descriptor);
        if (handle) cudnnDestroy(handle);
    }

    /** A new descriptor of @p type, owned by the plan. */
    cudnnBackendDescriptor_t create(cudnnBackendDescriptorType_t type) {
        cudnnBackendDescriptor_t descriptor = nullptr;
        if (!status) status = cudnnBackendCreateDescriptor(type, &descriptor);
        if (descriptor) descriptors.push_back(descriptor);
        return descriptor;
    }

    /** Sets @p count values of attribute @p name. */
    void set(cudnnBackendDescriptor_t descriptor, cudnnBackendAttributeName_t name, cudnnBackendAttributeType_t type,
             std::int64_t count, void const *values) {
        if (!status) status = cudnnBackendSetAttribute(descriptor, name, type, count, values);
    }

    /** Sets attribute @p name to the descriptor @p value. */
    void link(cudnnBackendDescriptor_t descriptor, cudnnBackendAttributeName_t name, cudnnBackendDescriptor_t value) {
        set(descriptor, name, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &value);
    }

    /** Finalizes @p descriptor, so it can be linked or executed. */
    void finalize(cudnnBackendDescriptor_t descriptor) {
        if (!status) status = cudnnBackendFinalize(descriptor);
    }

    /** A finalized tensor, entered into the variant pack at @p address unless that is null. */
    cudnnBackendDescriptor_t tensor(cudnn_uid_t uid, cudnnDataType_t type, std::array<std::int64_t, 4> dimensions,
                                    std::array<std::int64_t, 4> strides, cudnn_binding_t binding, void *address,
                                    cudnnBackendDescriptor_t ragged_offsets = nullptr) {
        cudnnBackendDescriptor_t const descriptor = create(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
        std::int64_t const identifier = std::int64_t(uid), alignment = 16;
        bool const is_virtual = binding == cudnn_binding_t::virtual_k, is_by_value = binding == cudnn_binding_t::host_k;
        set(descriptor, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &identifier);
        set(descriptor, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &type);
        set(descriptor, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, dimensions.data());
        set(descriptor, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, strides.data());
        set(descriptor, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment);
        set(descriptor, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, 1, &is_virtual);
        set(descriptor, CUDNN_ATTR_TENSOR_IS_BY_VALUE, CUDNN_TYPE_BOOLEAN, 1, &is_by_value);
        if (ragged_offsets) link(descriptor, CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC, ragged_offsets);
        finalize(descriptor);
        if (address) uids.push_back(identifier), addresses.push_back(address);
        return descriptor;
    }

    /** A finalized one-element tensor. */
    cudnnBackendDescriptor_t scalar(cudnn_uid_t uid, cudnnDataType_t type, cudnn_binding_t binding, void *address) {
        return tensor(uid, type, {1, 1, 1, 1}, {1, 1, 1, 1}, binding, address);
    }

    /** Builds the graph and the first plan cuDNN's heuristics offer that finalizes, or returns why none did. */
    cudnnStatus_t build(nk_dtype_t dtype, attention_visibility_t visibility, attention_shape_t shape) {
        std::int64_t const heads = shape.head_count, key_value_heads = shape.key_value_head_count, depth = shape.depth,
                           queries = shape.queries, keys = shape.keys;
        cudnnDataType_t const io_type = cudnn_data_type(dtype);
        scale = 1.0f / std::sqrt(float(depth)),
        window = std::int32_t(std::min<nk_size_t>(attention_window(visibility), shape.keys));
        key_bytes = std::size_t(keys * key_value_heads * depth) * nk_dtype_bits(dtype) / 8;
        uids = {std::int64_t(cudnn_uid_t::queries_k), std::int64_t(cudnn_uid_t::keys_k),
                std::int64_t(cudnn_uid_t::values_k), std::int64_t(cudnn_uid_t::output_k)};
        addresses.assign(uids.size(), nullptr);

        // Device words: lengths, then the ragged element offsets of the one segment, then the E4M3 scalars.
        std::size_t const first_word = std::size_t(cudnn_uid_t::query_length_k);
        std::array<std::uint32_t, std::size_t(cudnn_uid_t::end_k) - first_word> words {};
        auto const word = [&](cudnn_uid_t uid) -> std::uint32_t & { return words[std::size_t(uid) - first_word]; };
        word(cudnn_uid_t::query_length_k) = std::uint32_t(queries),
        word(cudnn_uid_t::key_length_k) = std::uint32_t(keys);
        (&word(cudnn_uid_t::query_offsets_k))[1] = std::uint32_t(queries * heads * depth);
        (&word(cudnn_uid_t::key_offsets_k))[1] = std::uint32_t(keys * key_value_heads * depth);
        float const unit = 1, probability_scale = 256, probability_descale = 1.0f / 256;
        float const scales[6] = {unit, unit, unit, probability_descale, probability_scale, unit};
        std::memcpy(&word(cudnn_uid_t::descale_queries_k), scales, sizeof(scales));
        parameters = device_vector<std::uint32_t>::try_empty(words.size());
        if (parameters.empty()) return CUDNN_STATUS_ALLOC_FAILED;
        cudaMemcpy(parameters.raw_values_data(), words.data(), sizeof(words), cudaMemcpyHostToDevice);
        auto const device = [&](cudnn_uid_t uid) -> void * {
            return parameters.raw_values_data() + (std::size_t(uid) - first_word);
        };
        if ((status = cudnnCreate(&handle))) return status;
        cudnnSetStream(handle, cudaStreamPerThread);

        auto const offsets = [&](cudnn_uid_t uid) {
            return tensor(uid, CUDNN_DATA_INT32, {2, 1, 1, 1}, {1, 1, 1, 1}, cudnn_binding_t::device_k, device(uid));
        };
        auto const rows = [&](cudnn_uid_t uid, std::int64_t head_count, std::int64_t length,
                              cudnnBackendDescriptor_t ragged_offsets) {
            return tensor(uid, io_type, {1, head_count, length, depth},
                          {length * head_count * depth, depth, head_count * depth, 1}, cudnn_binding_t::device_k,
                          nullptr, ragged_offsets);
        };
        cudnnBackendDescriptor_t const query_offsets = offsets(cudnn_uid_t::query_offsets_k);
        cudnnBackendDescriptor_t const key_offsets = offsets(cudnn_uid_t::key_offsets_k);
        cudnnBackendDescriptor_t const query_length = scalar(cudnn_uid_t::query_length_k, CUDNN_DATA_INT32,
                                                             cudnn_binding_t::device_k,
                                                             device(cudnn_uid_t::query_length_k));
        cudnnBackendDescriptor_t const key_length = scalar(
            cudnn_uid_t::key_length_k, CUDNN_DATA_INT32, cudnn_binding_t::device_k, device(cudnn_uid_t::key_length_k));
        cudnnBackendDescriptor_t const attention = create(CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR);
        link(attention, CUDNN_ATTR_OPERATION_SDPA_FWD_QDESC,
             rows(cudnn_uid_t::queries_k, heads, queries, query_offsets));
        link(attention, CUDNN_ATTR_OPERATION_SDPA_FWD_KDESC,
             rows(cudnn_uid_t::keys_k, key_value_heads, keys, key_offsets));
        link(attention, CUDNN_ATTR_OPERATION_SDPA_FWD_VDESC,
             rows(cudnn_uid_t::values_k, key_value_heads, keys, key_offsets));
        link(attention, CUDNN_ATTR_OPERATION_SDPA_FWD_ODESC,
             rows(cudnn_uid_t::output_k, heads, queries, query_offsets));
        link(attention, CUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC,
             scalar(cudnn_uid_t::scale_k, CUDNN_DATA_FLOAT, cudnn_binding_t::host_k, &scale));
        link(attention, CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_QDESC, query_length);
        link(attention, CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_KVDESC, key_length);

        // Masks run on the scores between the two matmuls: keys past the query's position, then keys past the window.
        if (visibility != attention_visibility_t::bidirectional_k) {
            std::size_t const masks_count = visibility == attention_visibility_t::causal_window_1024_k ? 2 : 1;
            cudnnBackendDescriptor_t const minimum = scalar(cudnn_uid_t::negative_infinity_k, CUDNN_DATA_FLOAT,
                                                            cudnn_binding_t::host_k, &negative_infinity);
            std::array<cudnnBackendDescriptor_t, 3> scores {};
            for (std::size_t index = 0; index <= masks_count; ++index)
                scores[index] = tensor(cudnn_uid_t(std::int64_t(cudnn_uid_t::scores_k) + index), CUDNN_DATA_FLOAT,
                                       {1, heads, queries, keys}, {heads * queries * keys, queries * keys, keys, 1},
                                       cudnn_binding_t::virtual_k, nullptr);
            std::array<cudnnBackendDescriptor_t, 2> masks {};
            cudnnPointwiseMode_t const comparisons[2] = {CUDNN_POINTWISE_CMP_GE, CUDNN_POINTWISE_CMP_GT};
            for (std::size_t index = 0; index != masks_count; ++index) {
                masks[index] = create(CUDNN_BACKEND_OPERATION_DIAGONAL_BAND_MASK_DESCRIPTOR);
                link(masks[index], CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_XDESC, scores[index]);
                link(masks[index], CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_BDESC, minimum);
                link(masks[index], CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SEQ_LEN_QDESC, query_length);
                link(masks[index], CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SEQ_LEN_KVDESC, key_length);
                link(masks[index], CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_YDESC, scores[index + 1]);
                set(masks[index], CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_COMPARISON_MODE, CUDNN_TYPE_POINTWISE_MODE, 1,
                    &comparisons[index]);
                if (index == 1)
                    link(masks[index], CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_LEFT_BOUND_DESC,
                         scalar(cudnn_uid_t::window_k, CUDNN_DATA_INT32, cudnn_binding_t::host_k, &window));
                finalize(masks[index]);
            }
            cudnnBackendDescriptor_t const subgraph = create(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
            set(subgraph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, masks_count, masks.data());
            finalize(subgraph);
            std::int64_t const input_uid = std::int64_t(cudnn_uid_t::scores_k), output_uid = input_uid + masks_count;
            link(attention, CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH, subgraph);
            set(attention, CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH_INPUT_UID, CUDNN_TYPE_INT64, 1, &input_uid);
            set(attention, CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH_OUTPUT_UID, CUDNN_TYPE_INT64, 1, &output_uid);
        }
        std::pair<cudnnBackendAttributeName_t, cudnn_uid_t> const scalings[] = {
            {CUDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_QDESC, cudnn_uid_t::descale_queries_k},
            {CUDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_KDESC, cudnn_uid_t::descale_keys_k},
            {CUDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_VDESC, cudnn_uid_t::descale_values_k},
            {CUDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_SDESC, cudnn_uid_t::descale_probabilities_k},
            {CUDNN_ATTR_OPERATION_SDPA_FWD_SCALE_SDESC, cudnn_uid_t::scale_probabilities_k},
            {CUDNN_ATTR_OPERATION_SDPA_FWD_SCALE_ODESC, cudnn_uid_t::scale_output_k},
            {CUDNN_ATTR_OPERATION_SDPA_FWD_AMAX_SDESC, cudnn_uid_t::amax_probabilities_k},
            {CUDNN_ATTR_OPERATION_SDPA_FWD_AMAX_ODESC, cudnn_uid_t::amax_output_k}};
        if (io_type == CUDNN_DATA_FP8_E4M3)
            for (auto [attribute, uid] : scalings)
                link(attention, attribute, scalar(uid, CUDNN_DATA_FLOAT, cudnn_binding_t::device_k, device(uid)));
        finalize(attention);

        cudnnBackendDescriptor_t const graph = create(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
        set(graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &attention);
        set(graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle);
        finalize(graph);
        cudnnBackendDescriptor_t const heuristics = create(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
        cudnnBackendHeurMode_t const mode = CUDNN_HEUR_MODE_A;
        link(heuristics, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, graph);
        set(heuristics, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &mode);
        finalize(heuristics);
        std::int64_t configs_count = 0;
        if (!status)
            status = cudnnBackendGetAttribute(heuristics, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                              0, &configs_count, nullptr);
        std::vector<cudnnBackendDescriptor_t> configs(static_cast<std::size_t>(configs_count));
        for (cudnnBackendDescriptor_t &config : configs) config = create(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
        if (!status)
            status = cudnnBackendGetAttribute(heuristics, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                              configs_count, &configs_count, configs.data());
        for (std::int64_t index = 0; index != configs_count && !status && !plan; ++index) {
            cudnnBackendDescriptor_t const candidate = create(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
            link(candidate, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, configs[index]);
            if (!status && cudnnBackendFinalize(candidate) == CUDNN_STATUS_SUCCESS) plan = candidate;
        }
        if (!status && !plan) status = CUDNN_STATUS_NOT_SUPPORTED;
        std::int64_t workspace_bytes = 0;
        if (!status)
            status = cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1,
                                              nullptr, &workspace_bytes);
        workspace = device_vector<char>::try_empty(std::size_t(std::max<std::int64_t>(workspace_bytes, 1)));
        if (!status && workspace.empty()) status = CUDNN_STATUS_ALLOC_FAILED;
        return status;
    }

    /** Enqueues the plan over @p queries and a @p packed cache holding K, then V. */
    cudaError_t launch(void const *queries, void const *packed, void *output) noexcept {
        addresses[0] = const_cast<void *>(queries), addresses[1] = const_cast<void *>(packed);
        addresses[2] = static_cast<char *>(addresses[1]) + key_bytes, addresses[3] = output;
        void *workspace_address = workspace.raw_values_data();
        cudnnBackendDescriptor_t pack = nullptr;
        cudnnStatus_t result = cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &pack);
        if (!result)
            result = cudnnBackendSetAttribute(pack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64,
                                              std::int64_t(uids.size()), uids.data());
        if (!result)
            result = cudnnBackendSetAttribute(pack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR,
                                              std::int64_t(addresses.size()), addresses.data());
        if (!result)
            result = cudnnBackendSetAttribute(pack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1,
                                              &workspace_address);
        if (!result) result = cudnnBackendFinalize(pack);
        if (!result) result = cudnnBackendExecute(handle, plan, pack);
        if (pack) cudnnBackendDestroyDescriptor(pack);
        return result == CUDNN_STATUS_SUCCESS ? cudaSuccess : cudaErrorUnknown;
    }
};

/** Registers one cuDNN row of @p shape through `register_attention`, with K and V copied into each set's cache. */
template <nk_dtype_t input_dtype_, attention_visibility_t visibility_>
void register_attention_with_cudnn(std::string const &name, attention_shape_t shape) {
    auto const plan = std::make_shared<cudnn_attention_plan_t>();
    if (cudnnStatus_t const status = plan->build(input_dtype_, visibility_, shape))
        return print_skipped(attention_row_name(name, visibility_, shape), cudnnGetErrorString(status));
    auto const packed_size = [](std::size_t key_value_heads, std::size_t depth, nk_u32_t const *lengths, std::size_t) {
        return 2 * std::size_t(lengths[0]) * key_value_heads * depth * nk_dtype_bits(input_dtype_) / 8;
    };
    auto const pack = [key_bytes = plan->key_bytes](void const *keys, void const *values, std::size_t, std::size_t,
                                                    nk_u32_t const *, nk_u32_t const *, std::size_t, std::size_t,
                                                    std::size_t, void *packed, std::size_t, std::size_t,
                                                    cudaStream_t stream) {
        cudaMemcpyAsync(packed, keys, key_bytes, cudaMemcpyDeviceToDevice, stream);
        return cudaMemcpyAsync(static_cast<char *>(packed) + key_bytes, values, key_bytes, cudaMemcpyDeviceToDevice,
                               stream);
    };
    register_attention<input_dtype_, visibility_, cuda_backend_t>(
        name, packed_size, pack,
        [plan](void const *queries, void const *packed, void *output, auto...) {
            return plan->launch(queries, packed, output);
        },
        shape);
}

/** Registers cuDNN rows beside the NumKong ones: bidirectional, causal, and windowed where the window clips. */
template <nk_dtype_t input_dtype_>
void run_attention_with_cudnn(std::string const &bidirectional_name, std::string const &causal_name) {
    for (attention_shape_t const shape : cuda_backend_t::attention_shapes()) {
        register_attention_with_cudnn<input_dtype_, attention_visibility_t::bidirectional_k>(bidirectional_name, shape);
        register_attention_with_cudnn<input_dtype_, attention_visibility_t::causal_k>(causal_name, shape);
        if (attention_window_clips(shape))
            register_attention_with_cudnn<input_dtype_, attention_visibility_t::causal_window_1024_k>(causal_name,
                                                                                                      shape);
    }
}

#endif // NK_COMPARE_TO_CUDNN

/** Every cuDNN row: BF16 and E4M3 attention, bidirectional, causal, and windowed where the window clips. */
void bench_cross_cudnn() {
#if NK_COMPARE_TO_CUDNN
    run_attention_with_cudnn<nk_bf16_k>("attention_bidirectional_bf16_with_cudnn", "attention_causal_bf16_with_cudnn");
    run_attention_with_cudnn<nk_e4m3_k>("attention_bidirectional_e4m3_with_cudnn", "attention_causal_e4m3_with_cudnn");
#endif // NK_COMPARE_TO_CUDNN
}
#pragma endregion cuDNN

#pragma region cuVS
#if NK_COMPARE_TO_CUVS

/** A dense row-major floating-point device matrix of @p shape as a DLPack tensor. */
DLManagedTensor dlpack_matrix(void const *data, std::int64_t *shape, std::uint8_t bits) noexcept {
    DLManagedTensor tensor {};
    tensor.dl_tensor.data = const_cast<void *>(data);
    tensor.dl_tensor.device = {kDLCUDA, 0};
    tensor.dl_tensor.ndim = 2;
    tensor.dl_tensor.dtype = {kDLFloat, bits, 1};
    tensor.dl_tensor.shape = shape;
    return tensor;
}

/** Registers `cuvsPairwiseDistance` as a cosine or L2-expanded row beside NumKong's angular or euclidean one. */
template <nk_dtype_t input_dtype_>
void register_spatials_with_cuvs(std::string const &name, reference_metric_t metric) {
    std::shared_ptr<cuvsResources_t> const resources(new cuvsResources_t {}, [](cuvsResources_t *resources) {
        cuvsResourcesDestroy(*resources);
        delete resources;
    });
    if (cuvsResourcesCreate(resources.get()) != CUVS_SUCCESS) return print_skipped(name, cuvsGetLastErrorText());
    cuvsDistanceType const distance = metric == reference_metric_t::angular_k ? CosineExpanded : L2SqrtExpanded;
    std::uint8_t const bits = std::uint8_t(nk_dtype_bits(input_dtype_));
    register_unpacked<input_dtype_, nk::f32_t>(
        name, metric,
        [resources, distance, bits](void const *a, void const *b, void *c, std::size_t height, std::size_t width,
                                    std::size_t depth, std::size_t a_stride, std::size_t, cudaStream_t stream) {
            if (a_stride * 8 != depth * bits) return cudaErrorInvalidPitchValue;
            std::int64_t a_shape[2] = {std::int64_t(height), std::int64_t(depth)};
            std::int64_t b_shape[2] = {std::int64_t(width), std::int64_t(depth)};
            std::int64_t c_shape[2] = {std::int64_t(height), std::int64_t(width)};
            DLManagedTensor a_tensor = dlpack_matrix(a, a_shape, bits), b_tensor = dlpack_matrix(b, b_shape, bits),
                            c_tensor = dlpack_matrix(c, c_shape, 32);
            cuvsStreamSet(*resources, stream);
            return cuvsPairwiseDistance(*resources, &a_tensor, &b_tensor, &c_tensor, distance, 2.0f) == CUVS_SUCCESS
                       ? cudaSuccess
                       : cudaErrorUnknown;
        });
}

#endif // NK_COMPARE_TO_CUVS

/** Every cuVS row: cosine and L2 distances over F32 and F16. */
void bench_cross_cuvs() {
#if NK_COMPARE_TO_CUVS
    register_spatials_with_cuvs<nk_f32_k>("angulars_packed_f32_with_cuvs", reference_metric_t::angular_k);
    register_spatials_with_cuvs<nk_f32_k>("euclideans_packed_f32_with_cuvs", reference_metric_t::euclidean_k);
    register_spatials_with_cuvs<nk_f16_k>("angulars_packed_f16_with_cuvs", reference_metric_t::angular_k);
    register_spatials_with_cuvs<nk_f16_k>("euclideans_packed_f16_with_cuvs", reference_metric_t::euclidean_k);
#endif // NK_COMPARE_TO_CUVS
}
#pragma endregion cuVS

/** Prints the device, the baselines compiled in, and the kernel families compiled in or runnable on it. */
void print_cuda_header() {
    cudaDeviceProp properties {};
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        std::printf("  CUDA: no usable device\n");
        return;
    }
    nk_capability_t const capabilities = nk_capabilities_cuda_available(device);
    std::printf("  CUDA: %s, compute capability %d.%d, %d SMs, %d MB L2\n", properties.name, properties.major,
                properties.minor, properties.multiProcessorCount, properties.l2CacheSize >> 20);
    std::printf("  CUDA baselines: cuBLAS ");
    print_indicator(NK_COMPARE_TO_CUBLAS);
    std::printf("  cuDNN ");
    print_indicator(NK_COMPARE_TO_CUDNN);
    std::printf("  cuVS ");
    print_indicator(NK_COMPARE_TO_CUVS);
    std::printf("\n  CUDA families:");
    print_isa("Ampere", NK_TARGET_AMPERE, nk_cap_ampere_k, capabilities);
    print_isa("Hopper", NK_TARGET_HOPPER, nk_cap_hopper_k, capabilities);
    print_isa("Blackwell", NK_TARGET_BLACKWELL, nk_cap_blackwell_k, capabilities);
    print_isa("Blackwell RTX", NK_TARGET_BLACKWELLRTX, nk_cap_blackwellrtx_k, capabilities);
    std::printf("\n");
}

/** Every CUDA row: the kernel families this device runs, then the baselines compiled in. */
void bench_cross_cuda() {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return;
    nk_capability_t const capabilities = nk_capabilities_cuda_available(device);
    bench_cross_ampere(capabilities);
    bench_cross_blackwellrtx(capabilities);
    bench_cross_cublas();
    bench_cross_cudnn();
    bench_cross_cuvs();
}
