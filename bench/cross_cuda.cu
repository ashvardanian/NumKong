/**
 *  @file bench/cross_cuda.cu
 *  @author Ash Vardanian
 *  @date September 22, 2026
 *  @brief Batch operation benchmarks, CUDA kernels against cuBLASLt, cuBLAS, cuDNN and cuVS.
 *
 *  Runs the drivers of `cross.hpp` through a @c cuda_backend_t over device-resident operands,
 *  launching on @c cudaStreamPerThread, with every window of launches bracketed by CUDA events
 *  whose elapsed time replaces the wall time, so launch latency and host synchronization stay
 *  outside the measurement. Input sets rotate until their footprint is at least twice the L2.
 *
 *  The @c attention rows time prefill, 4096 queries on 4096 keys, and decode, 1 query on 4096 keys,
 *  with 32 query heads over 8 K and V heads of depth 128. Every baseline enters the same drivers as
 *  a kernel callable, so it shares their inputs, timing and counters, and compiles in only under
 *  its `NUMKONG_COMPARE_TO_*` CMake option.
 */

#include <cmath>   // `std::sqrt`, `INFINITY`
#include <cstddef> // `std::size_t`, `std::ptrdiff_t`
#include <cstdint> // `std::int32_t`, `std::int64_t`
#include <cstring> // `std::memcpy`

#include <algorithm>   // `std::max`, `std::min`
#include <array>       // `std::array`
#include <bit>         // `std::bit_ceil`
#include <memory>      // `std::shared_ptr`
#include <string>      // `std::string`
#include <type_traits> // `std::remove_pointer_t`, `std::true_type`
#include <utility>     // `std::pair`
#include <vector>      // `std::vector`

#include "numkong/numkong.h"

#include "cross.hpp"

#if NUMKONG_ARCH_CUDA_

#if NUMKONG_COMPARE_TO_CUBLAS
#include <cublasLt.h>
#include <cublas_v2.h>
#endif
#include <cuda_runtime.h>
#if NUMKONG_COMPARE_TO_CUDNN
#include <cudnn.h>
#endif
#if NUMKONG_COMPARE_TO_CUVS
#include <cuvs/core/c_api.h>
#include <cuvs/distance/pairwise_distance.h>
#endif

#pragma region CUDA Backend

namespace ashvardanian::numkong::bench {

/** Device memory: only kernels dereference it, so build with @c uninitialized and fill through
 *  @c cudaMemcpy. Returns @c nullptr on failure, which the allocating factories report. */
template <typename value_type_>
struct cuda_device_allocator {
    using value_type = value_type_;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    template <typename other_type_>
    struct rebind {
        using other = cuda_device_allocator<other_type_>;
    };

    constexpr cuda_device_allocator() noexcept = default;

    template <typename other_type_>
    constexpr cuda_device_allocator(cuda_device_allocator<other_type_> const &) noexcept {}

    [[nodiscard]] value_type *allocate(std::size_t count) noexcept {
        void *pointer = nullptr;
        if (count == 0 || cudaMalloc(&pointer, count * sizeof(value_type)) != cudaSuccess) return nullptr;
        return static_cast<value_type *>(pointer);
    }

    void deallocate(value_type *pointer, std::size_t) noexcept {
        if (!pointer) return;
        [[maybe_unused]] cudaError_t const status = cudaFree(pointer);
    }

    template <typename other_type_>
    constexpr bool operator==(cuda_device_allocator<other_type_> const &) const noexcept {
        return true;
    }
};

/** An @c nk::vector in device memory. */
template <typename value_type_>
using device_vector = nk::vector<value_type_, cuda_device_allocator<value_type_>>;

/** Runs the CUDA kernels on @c stream over device memory, keeping the first failed status and
 *  timing windows of launches with CUDA events. */
struct cuda_backend_t {

    /** Device memory, so timed buffers never page-migrate. */
    template <typename value_type_>
    using allocator = cuda_device_allocator<value_type_>;

    /** Where every call launches. */
    void *stream = cudaStreamPerThread;

    /** The first failure since the last synchronization. */
    nk_status_t status = nk_success_k;

    /** Row stride for rows of @p row_bytes: rounded up to the 16 bytes `cp.async` requires of A
     *  rows. */
    static constexpr std::size_t row_stride(std::size_t row_bytes) noexcept { return (row_bytes + 15) / 16 * 16; }

    /** Rotation sets of @p per_set each: enough to cover twice the L2, at most
     *  @c input_sets_count. */
    std::size_t input_sets(bytes_t per_set) const noexcept {
        int l2_bytes = 0, device = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, device);
        std::size_t const set_bytes = std::max(per_set.value, std::size_t(1));
        std::size_t const wanted = std::max(nk::divide_round_up(2 * std::size_t(l2_bytes), set_bytes), std::size_t(1));
        return std::min(std::bit_ceil(wanted), input_sets_count(per_set));
    }

    /** Rows a token-row benchmark batches: a 4096-token prefill. */
    static std::size_t token_rows(environment_t const &) noexcept { return 4096; }

    /** Prefill and decode segments at the end of a 4096-key cache, Llama-style heads. */
    static std::vector<attention_shape_t> attention_shapes(environment_t const &) {
        return {{"prefill", 32, 8, 128, 4096, 4096}, {"decode", 32, 8, 128, 1, 4096}};
    }

    /** Copies @p bytes in whichever direction the pointers imply. */
    void copy(void *destination, void const *source, std::size_t bytes) noexcept {
        keep(cudaMemcpy(destination, source, bytes, cudaMemcpyDefault));
    }

    /** Zeroes @p bytes of a device buffer. */
    void zero(void *destination, std::size_t bytes) noexcept { keep(cudaMemset(destination, 0, bytes)); }

    /** Launches @p kernel with @p arguments on the stream. */
    template <typename kernel_type_, typename... arguments_types_>
    void call(kernel_type_ kernel, arguments_types_... arguments) noexcept {
        keep(kernel(arguments..., stream));
    }

    /** Times windows of @p launch calls between two CUDA events, doubling a window until it spans a
     *  millisecond. */
    template <typename launch_type_>
    void time(loop_t &loop, std::size_t sets_count, launch_type_ &launch) {
        cudaEvent_t start = nullptr, stop = nullptr;
        cudaEventCreate(&start), cudaEventCreate(&stop);
        std::size_t calls = 0, window = 1;
        for ([[maybe_unused]] std::size_t call : loop) {
            cudaEventRecord(start, (cudaStream_t)stream);
            for (std::size_t index = 0; index != window; ++index) launch((calls + index) & (sets_count - 1));
            cudaEventRecord(stop, (cudaStream_t)stream);
            keep(cudaEventSynchronize(stop));
            if (status != nk_success_k) break;
            float milliseconds = 0;
            keep(cudaEventElapsedTime(&milliseconds, start, stop));
            std::chrono::duration<float, std::milli> const elapsed {milliseconds};
            loop.add_window(elapsed, window);
            calls += window;
            // Launch latency only amortizes over a millisecond of work
            if (elapsed < std::chrono::milliseconds(1)) window *= 2;
        }
        cudaEventDestroy(start), cudaEventDestroy(stop);
    }

    /** Waits for the stream, returning the name of the first failure since the last call, or
     *  @c nullptr. */
    char const *synchronize() noexcept {
        keep(cudaStreamSynchronize((cudaStream_t)stream));
        nk_status_t const failure = status;
        status = nk_success_k;
        return failure == nk_success_k ? nullptr : nk_status_name(failure);
    }

    /** Remembers @p result unless an earlier failure is pending. */
    void keep(nk_status_t result) noexcept {
        if (status == nk_success_k) status = result;
    }

    /** Remembers a baseline's or runtime call's failure as the kernel it would have failed. */
    void keep(cudaError_t result) noexcept { keep(result == cudaSuccess ? nk_success_k : nk_device_code_mismatch_k); }
};

/** Prints why a baseline row is missing, when @p name passes the filter. */
void print_skipped(environment_t const &env, std::string const &name, char const *reason) {
    if (env.settings.selects(name)) print(row_t {name, 0, {}, std::string_view(reason)});
}

/** Runs a baseline @p kernel over dense B rows through @c run_packed, with a copy of B as
 *  its pack. */
template <nk_dtype_t input_dtype_, typename output_type_, typename kernel_type_>
void run_unpacked(environment_t const &env, std::string const &name, reference_metric_t metric, kernel_type_ kernel,
                  cuda_backend_t const &backend) {
    using input_t = typename nk::type_for<input_dtype_>::type;
    auto const packed_size = [](std::size_t width, std::size_t depth, nk_size_t *bytes) {
        *bytes = width * nk::divide_round_up(depth, nk::dimensions_per_value<input_t>()) * sizeof(input_t);
        return nk_success_k;
    };
    auto const copy = [](void const *b, void const *, std::size_t width, std::size_t, std::size_t row_bytes,
                         std::size_t, void *packed, std::size_t, std::size_t, void *stream) {
        return cudaMemcpyAsync(packed, b, width * row_bytes, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
    };
    run_packed<input_dtype_, output_type_, cuda_backend_t>(env, name, metric, packed_size, copy, kernel, backend);
}

#pragma endregion CUDA Backend

#pragma region Rows

/** The CUDA baseline rows the tensor-core capabilities have no faster path for, on every device. */
void bench_cross_cuda(environment_t const &env, cuda_backend_t const &backend, nk_capability_t enabled) {
    if (!(enabled & nk_cap_cuda_k)) return;
    run_dots_packed<nk_f64_k>(env, "dots_packed_f64_cuda", nk_dots_pack_size_f64_cuda, nk_dots_pack_f64_cuda,
                              nk_dots_packed_f64_cuda, backend);
    run_dots_packed<nk_f32_k>(env, "dots_packed_f32_cuda", nk_dots_pack_size_f32_cuda, nk_dots_pack_f32_cuda,
                              nk_dots_packed_f32_cuda, backend);
    run_dots_symmetric<nk_f64_k>(env, "dots_symmetric_f64_cuda", nk_dots_symmetric_f64_cuda, backend);
    run_dots_symmetric<nk_f32_k>(env, "dots_symmetric_f32_cuda", nk_dots_symmetric_f32_cuda, backend);
    run_angulars_packed<nk_f64_k>(env, "angulars_packed_f64_cuda", nk_dots_pack_size_f64_cuda, nk_dots_pack_f64_cuda,
                                  nk_angulars_packed_f64_cuda, backend);
    run_angulars_packed<nk_f32_k>(env, "angulars_packed_f32_cuda", nk_dots_pack_size_f32_cuda, nk_dots_pack_f32_cuda,
                                  nk_angulars_packed_f32_cuda, backend);
    run_angulars_symmetric<nk_f64_k>(env, "angulars_symmetric_f64_cuda", nk_angulars_symmetric_f64_cuda, backend);
    run_angulars_symmetric<nk_f32_k>(env, "angulars_symmetric_f32_cuda", nk_angulars_symmetric_f32_cuda, backend);
    run_euclideans_packed<nk_f64_k>(env, "euclideans_packed_f64_cuda", nk_dots_pack_size_f64_cuda,
                                    nk_dots_pack_f64_cuda, nk_euclideans_packed_f64_cuda, backend);
    run_euclideans_packed<nk_f32_k>(env, "euclideans_packed_f32_cuda", nk_dots_pack_size_f32_cuda,
                                    nk_dots_pack_f32_cuda, nk_euclideans_packed_f32_cuda, backend);
    run_euclideans_symmetric<nk_f64_k>(env, "euclideans_symmetric_f64_cuda", nk_euclideans_symmetric_f64_cuda, backend);
    run_euclideans_symmetric<nk_f32_k>(env, "euclideans_symmetric_f32_cuda", nk_euclideans_symmetric_f32_cuda, backend);
}

/** The element-wise and RoPE CUDA baseline rows over the backend's token rows. */
void bench_each_cuda(environment_t const &env, cuda_backend_t const &backend, nk_capability_t enabled) {
    if (!(enabled & nk_cap_cuda_k)) return;
    constexpr nk_kernel_kind_t sum_k = nk_kernel_each_sum_k, scale_k = nk_kernel_each_scale_k;
    constexpr nk_kernel_kind_t blend_k = nk_kernel_each_blend_k, fma_k = nk_kernel_each_fma_k;
    run_each<nk_f64_k, sum_k, nk_f64_k>(env, "each_sum_f64_cuda", nk_each_sum_f64_cuda, backend);
    run_each<nk_f64_k, scale_k, nk_f64_k>(env, "each_scale_f64_cuda", nk_each_scale_f64_cuda, backend);
    run_each<nk_f64_k, blend_k, nk_f64_k>(env, "each_blend_f64_cuda", nk_each_blend_f64_cuda, backend);
    run_each<nk_f64_k, fma_k, nk_f64_k>(env, "each_fma_f64_cuda", nk_each_fma_f64_cuda, backend);
    run_each<nk_f32_k, sum_k, nk_f32_k>(env, "each_sum_f32_cuda", nk_each_sum_f32_cuda, backend);
    run_each<nk_f32_k, scale_k, nk_f32_k>(env, "each_scale_f32_cuda", nk_each_scale_f32_cuda, backend);
    run_each<nk_f32_k, blend_k, nk_f32_k>(env, "each_blend_f32_cuda", nk_each_blend_f32_cuda, backend);
    run_each<nk_f32_k, fma_k, nk_f32_k>(env, "each_fma_f32_cuda", nk_each_fma_f32_cuda, backend);
    run_each<nk_bf16_k, sum_k, nk_f32_k>(env, "each_sum_bf16_cuda", nk_each_sum_bf16_cuda, backend);
    run_each<nk_bf16_k, scale_k, nk_f32_k>(env, "each_scale_bf16_cuda", nk_each_scale_bf16_cuda, backend);
    run_each<nk_bf16_k, blend_k, nk_f32_k>(env, "each_blend_bf16_cuda", nk_each_blend_bf16_cuda, backend);
    run_each<nk_bf16_k, fma_k, nk_f32_k>(env, "each_fma_bf16_cuda", nk_each_fma_bf16_cuda, backend);
    run_each<nk_e4m3_k, sum_k, nk_f32_k>(env, "each_sum_e4m3_cuda", nk_each_sum_e4m3_cuda, backend);
    run_each<nk_e4m3_k, scale_k, nk_f32_k>(env, "each_scale_e4m3_cuda", nk_each_scale_e4m3_cuda, backend);
    run_each<nk_e4m3_k, blend_k, nk_f32_k>(env, "each_blend_e4m3_cuda", nk_each_blend_e4m3_cuda, backend);
    run_each<nk_e4m3_k, fma_k, nk_f32_k>(env, "each_fma_e4m3_cuda", nk_each_fma_e4m3_cuda, backend);
    run_each<nk_i8_k, sum_k, nk_f32_k>(env, "each_sum_i8_cuda", nk_each_sum_i8_cuda, backend);
    run_each<nk_i8_k, scale_k, nk_f32_k>(env, "each_scale_i8_cuda", nk_each_scale_i8_cuda, backend);
    run_each<nk_i8_k, blend_k, nk_f32_k>(env, "each_blend_i8_cuda", nk_each_blend_i8_cuda, backend);
    run_each<nk_i8_k, fma_k, nk_f32_k>(env, "each_fma_i8_cuda", nk_each_fma_i8_cuda, backend);
    run_each<nk_u8_k, sum_k, nk_f32_k>(env, "each_sum_u8_cuda", nk_each_sum_u8_cuda, backend);
    run_each<nk_u8_k, scale_k, nk_f32_k>(env, "each_scale_u8_cuda", nk_each_scale_u8_cuda, backend);
    run_each<nk_u8_k, blend_k, nk_f32_k>(env, "each_blend_u8_cuda", nk_each_blend_u8_cuda, backend);
    run_each<nk_u8_k, fma_k, nk_f32_k>(env, "each_fma_u8_cuda", nk_each_fma_u8_cuda, backend);
    run_each<nk_i32_k, sum_k, nk_f64_k>(env, "each_sum_i32_cuda", nk_each_sum_i32_cuda, backend);
    run_each<nk_i32_k, scale_k, nk_f64_k>(env, "each_scale_i32_cuda", nk_each_scale_i32_cuda, backend);
    run_each<nk_i32_k, blend_k, nk_f64_k>(env, "each_blend_i32_cuda", nk_each_blend_i32_cuda, backend);
    run_each<nk_i32_k, fma_k, nk_f64_k>(env, "each_fma_i32_cuda", nk_each_fma_i32_cuda, backend);
    run_rmsnorm<nk_f32_k>(env, "each_rmsnorm_f32_cuda", nk_each_rmsnorm_f32_cuda, backend);
    run_rmsnorm<nk_bf16_k>(env, "each_rmsnorm_bf16_cuda", nk_each_rmsnorm_bf16_cuda, backend);
    run_rmsnorm<nk_e4m3_k>(env, "each_rmsnorm_e4m3_cuda", nk_each_rmsnorm_e4m3_cuda, backend);
    run_swiglu<nk_f32_k>(env, "each_swiglu_f32_cuda", nk_each_swiglu_f32_cuda, backend);
    run_swiglu<nk_bf16_k>(env, "each_swiglu_bf16_cuda", nk_each_swiglu_bf16_cuda, backend);
    run_swiglu<nk_e4m3_k>(env, "each_swiglu_e4m3_cuda", nk_each_swiglu_e4m3_cuda, backend);
    run_attention_rope<nk_f32_k>(env, "attention_rope_f32_cuda", nk_attention_rope_f32_cuda, backend);
    run_attention_rope<nk_bf16_k>(env, "attention_rope_bf16_cuda", nk_attention_rope_bf16_cuda, backend);
    run_attention_rope<nk_e4m3_k>(env, "attention_rope_e4m3_cuda", nk_attention_rope_e4m3_cuda, backend);
}

/** Every Ampere entry point, compiled only when the architecture list includes the family. */
void bench_cross_ampere([[maybe_unused]] environment_t const &env, [[maybe_unused]] cuda_backend_t const &backend,
                        [[maybe_unused]] nk_capability_t enabled) {
#if NUMKONG_TARGET_AMPERE
    if (!(enabled & nk_cap_ampere_k)) return;
    run_dots_packed<nk_bf16_k>(env, "dots_packed_bf16_ampere", nk_dots_pack_size_bf16_ampere, nk_dots_pack_bf16_ampere,
                               nk_dots_packed_bf16_ampere, backend);
    run_dots_packed<nk_f16_k>(env, "dots_packed_f16_ampere", nk_dots_pack_size_f16_ampere, nk_dots_pack_f16_ampere,
                              nk_dots_packed_f16_ampere, backend);
    run_dots_packed<nk_e5m2_k>(env, "dots_packed_e5m2_ampere", nk_dots_pack_size_e5m2_ampere, nk_dots_pack_e5m2_ampere,
                               nk_dots_packed_e5m2_ampere, backend);
    run_dots_packed<nk_e4m3_k>(env, "dots_packed_e4m3_ampere", nk_dots_pack_size_e4m3_ampere, nk_dots_pack_e4m3_ampere,
                               nk_dots_packed_e4m3_ampere, backend);
    run_dots_packed<nk_e3m2_k>(env, "dots_packed_e3m2_ampere", nk_dots_pack_size_e3m2_ampere, nk_dots_pack_e3m2_ampere,
                               nk_dots_packed_e3m2_ampere, backend);
    run_dots_packed<nk_e2m3_k>(env, "dots_packed_e2m3_ampere", nk_dots_pack_size_e2m3_ampere, nk_dots_pack_e2m3_ampere,
                               nk_dots_packed_e2m3_ampere, backend);
    run_dots_packed<nk_e2m1_k>(env, "dots_packed_e2m1_ampere", nk_dots_pack_size_e2m1_ampere, nk_dots_pack_e2m1_ampere,
                               nk_dots_packed_e2m1_ampere, backend);
    run_dots_packed<nk_i8_k>(env, "dots_packed_i8_ampere", nk_dots_pack_size_i8_ampere, nk_dots_pack_i8_ampere,
                             nk_dots_packed_i8_ampere, backend);
    run_dots_packed<nk_i4_k>(env, "dots_packed_i4_ampere", nk_dots_pack_size_i4_ampere, nk_dots_pack_i4_ampere,
                             nk_dots_packed_i4_ampere, backend);
    run_dots_packed<nk_u8_k>(env, "dots_packed_u8_ampere", nk_dots_pack_size_u8_ampere, nk_dots_pack_u8_ampere,
                             nk_dots_packed_u8_ampere, backend);
    run_dots_packed<nk_u4_k>(env, "dots_packed_u4_ampere", nk_dots_pack_size_u4_ampere, nk_dots_pack_u4_ampere,
                             nk_dots_packed_u4_ampere, backend);

    run_dots_symmetric<nk_bf16_k>(env, "dots_symmetric_bf16_ampere", nk_dots_symmetric_bf16_ampere, backend);
    run_dots_symmetric<nk_f16_k>(env, "dots_symmetric_f16_ampere", nk_dots_symmetric_f16_ampere, backend);
    run_dots_symmetric<nk_e5m2_k>(env, "dots_symmetric_e5m2_ampere", nk_dots_symmetric_e5m2_ampere, backend);
    run_dots_symmetric<nk_e4m3_k>(env, "dots_symmetric_e4m3_ampere", nk_dots_symmetric_e4m3_ampere, backend);
    run_dots_symmetric<nk_e3m2_k>(env, "dots_symmetric_e3m2_ampere", nk_dots_symmetric_e3m2_ampere, backend);
    run_dots_symmetric<nk_e2m3_k>(env, "dots_symmetric_e2m3_ampere", nk_dots_symmetric_e2m3_ampere, backend);
    run_dots_symmetric<nk_e2m1_k>(env, "dots_symmetric_e2m1_ampere", nk_dots_symmetric_e2m1_ampere, backend);
    run_dots_symmetric<nk_i8_k>(env, "dots_symmetric_i8_ampere", nk_dots_symmetric_i8_ampere, backend);
    run_dots_symmetric<nk_i4_k>(env, "dots_symmetric_i4_ampere", nk_dots_symmetric_i4_ampere, backend);
    run_dots_symmetric<nk_u8_k>(env, "dots_symmetric_u8_ampere", nk_dots_symmetric_u8_ampere, backend);
    run_dots_symmetric<nk_u4_k>(env, "dots_symmetric_u4_ampere", nk_dots_symmetric_u4_ampere, backend);

    run_angulars_packed<nk_bf16_k>(env, "angulars_packed_bf16_ampere", nk_dots_pack_size_bf16_ampere,
                                   nk_dots_pack_bf16_ampere, nk_angulars_packed_bf16_ampere, backend);
    run_angulars_packed<nk_f16_k>(env, "angulars_packed_f16_ampere", nk_dots_pack_size_f16_ampere,
                                  nk_dots_pack_f16_ampere, nk_angulars_packed_f16_ampere, backend);
    run_angulars_packed<nk_e5m2_k>(env, "angulars_packed_e5m2_ampere", nk_dots_pack_size_e5m2_ampere,
                                   nk_dots_pack_e5m2_ampere, nk_angulars_packed_e5m2_ampere, backend);
    run_angulars_packed<nk_e4m3_k>(env, "angulars_packed_e4m3_ampere", nk_dots_pack_size_e4m3_ampere,
                                   nk_dots_pack_e4m3_ampere, nk_angulars_packed_e4m3_ampere, backend);
    run_angulars_packed<nk_e3m2_k>(env, "angulars_packed_e3m2_ampere", nk_dots_pack_size_e3m2_ampere,
                                   nk_dots_pack_e3m2_ampere, nk_angulars_packed_e3m2_ampere, backend);
    run_angulars_packed<nk_e2m3_k>(env, "angulars_packed_e2m3_ampere", nk_dots_pack_size_e2m3_ampere,
                                   nk_dots_pack_e2m3_ampere, nk_angulars_packed_e2m3_ampere, backend);
    run_angulars_packed<nk_e2m1_k>(env, "angulars_packed_e2m1_ampere", nk_dots_pack_size_e2m1_ampere,
                                   nk_dots_pack_e2m1_ampere, nk_angulars_packed_e2m1_ampere, backend);
    run_angulars_packed<nk_i8_k>(env, "angulars_packed_i8_ampere", nk_dots_pack_size_i8_ampere, nk_dots_pack_i8_ampere,
                                 nk_angulars_packed_i8_ampere, backend);
    run_angulars_packed<nk_i4_k>(env, "angulars_packed_i4_ampere", nk_dots_pack_size_i4_ampere, nk_dots_pack_i4_ampere,
                                 nk_angulars_packed_i4_ampere, backend);
    run_angulars_packed<nk_u8_k>(env, "angulars_packed_u8_ampere", nk_dots_pack_size_u8_ampere, nk_dots_pack_u8_ampere,
                                 nk_angulars_packed_u8_ampere, backend);
    run_angulars_packed<nk_u4_k>(env, "angulars_packed_u4_ampere", nk_dots_pack_size_u4_ampere, nk_dots_pack_u4_ampere,
                                 nk_angulars_packed_u4_ampere, backend);

    run_angulars_symmetric<nk_bf16_k>(env, "angulars_symmetric_bf16_ampere", nk_angulars_symmetric_bf16_ampere,
                                      backend);
    run_angulars_symmetric<nk_f16_k>(env, "angulars_symmetric_f16_ampere", nk_angulars_symmetric_f16_ampere, backend);
    run_angulars_symmetric<nk_e5m2_k>(env, "angulars_symmetric_e5m2_ampere", nk_angulars_symmetric_e5m2_ampere,
                                      backend);
    run_angulars_symmetric<nk_e4m3_k>(env, "angulars_symmetric_e4m3_ampere", nk_angulars_symmetric_e4m3_ampere,
                                      backend);
    run_angulars_symmetric<nk_e3m2_k>(env, "angulars_symmetric_e3m2_ampere", nk_angulars_symmetric_e3m2_ampere,
                                      backend);
    run_angulars_symmetric<nk_e2m3_k>(env, "angulars_symmetric_e2m3_ampere", nk_angulars_symmetric_e2m3_ampere,
                                      backend);
    run_angulars_symmetric<nk_e2m1_k>(env, "angulars_symmetric_e2m1_ampere", nk_angulars_symmetric_e2m1_ampere,
                                      backend);
    run_angulars_symmetric<nk_i8_k>(env, "angulars_symmetric_i8_ampere", nk_angulars_symmetric_i8_ampere, backend);
    run_angulars_symmetric<nk_i4_k>(env, "angulars_symmetric_i4_ampere", nk_angulars_symmetric_i4_ampere, backend);
    run_angulars_symmetric<nk_u8_k>(env, "angulars_symmetric_u8_ampere", nk_angulars_symmetric_u8_ampere, backend);
    run_angulars_symmetric<nk_u4_k>(env, "angulars_symmetric_u4_ampere", nk_angulars_symmetric_u4_ampere, backend);

    run_euclideans_packed<nk_bf16_k>(env, "euclideans_packed_bf16_ampere", nk_dots_pack_size_bf16_ampere,
                                     nk_dots_pack_bf16_ampere, nk_euclideans_packed_bf16_ampere, backend);
    run_euclideans_packed<nk_f16_k>(env, "euclideans_packed_f16_ampere", nk_dots_pack_size_f16_ampere,
                                    nk_dots_pack_f16_ampere, nk_euclideans_packed_f16_ampere, backend);
    run_euclideans_packed<nk_e5m2_k>(env, "euclideans_packed_e5m2_ampere", nk_dots_pack_size_e5m2_ampere,
                                     nk_dots_pack_e5m2_ampere, nk_euclideans_packed_e5m2_ampere, backend);
    run_euclideans_packed<nk_e4m3_k>(env, "euclideans_packed_e4m3_ampere", nk_dots_pack_size_e4m3_ampere,
                                     nk_dots_pack_e4m3_ampere, nk_euclideans_packed_e4m3_ampere, backend);
    run_euclideans_packed<nk_e3m2_k>(env, "euclideans_packed_e3m2_ampere", nk_dots_pack_size_e3m2_ampere,
                                     nk_dots_pack_e3m2_ampere, nk_euclideans_packed_e3m2_ampere, backend);
    run_euclideans_packed<nk_e2m3_k>(env, "euclideans_packed_e2m3_ampere", nk_dots_pack_size_e2m3_ampere,
                                     nk_dots_pack_e2m3_ampere, nk_euclideans_packed_e2m3_ampere, backend);
    run_euclideans_packed<nk_e2m1_k>(env, "euclideans_packed_e2m1_ampere", nk_dots_pack_size_e2m1_ampere,
                                     nk_dots_pack_e2m1_ampere, nk_euclideans_packed_e2m1_ampere, backend);
    run_euclideans_packed<nk_i8_k>(env, "euclideans_packed_i8_ampere", nk_dots_pack_size_i8_ampere,
                                   nk_dots_pack_i8_ampere, nk_euclideans_packed_i8_ampere, backend);
    run_euclideans_packed<nk_i4_k>(env, "euclideans_packed_i4_ampere", nk_dots_pack_size_i4_ampere,
                                   nk_dots_pack_i4_ampere, nk_euclideans_packed_i4_ampere, backend);
    run_euclideans_packed<nk_u8_k>(env, "euclideans_packed_u8_ampere", nk_dots_pack_size_u8_ampere,
                                   nk_dots_pack_u8_ampere, nk_euclideans_packed_u8_ampere, backend);
    run_euclideans_packed<nk_u4_k>(env, "euclideans_packed_u4_ampere", nk_dots_pack_size_u4_ampere,
                                   nk_dots_pack_u4_ampere, nk_euclideans_packed_u4_ampere, backend);

    run_euclideans_symmetric<nk_bf16_k>(env, "euclideans_symmetric_bf16_ampere", nk_euclideans_symmetric_bf16_ampere,
                                        backend);
    run_euclideans_symmetric<nk_f16_k>(env, "euclideans_symmetric_f16_ampere", nk_euclideans_symmetric_f16_ampere,
                                       backend);
    run_euclideans_symmetric<nk_e5m2_k>(env, "euclideans_symmetric_e5m2_ampere", nk_euclideans_symmetric_e5m2_ampere,
                                        backend);
    run_euclideans_symmetric<nk_e4m3_k>(env, "euclideans_symmetric_e4m3_ampere", nk_euclideans_symmetric_e4m3_ampere,
                                        backend);
    run_euclideans_symmetric<nk_e3m2_k>(env, "euclideans_symmetric_e3m2_ampere", nk_euclideans_symmetric_e3m2_ampere,
                                        backend);
    run_euclideans_symmetric<nk_e2m3_k>(env, "euclideans_symmetric_e2m3_ampere", nk_euclideans_symmetric_e2m3_ampere,
                                        backend);
    run_euclideans_symmetric<nk_e2m1_k>(env, "euclideans_symmetric_e2m1_ampere", nk_euclideans_symmetric_e2m1_ampere,
                                        backend);
    run_euclideans_symmetric<nk_i8_k>(env, "euclideans_symmetric_i8_ampere", nk_euclideans_symmetric_i8_ampere,
                                      backend);
    run_euclideans_symmetric<nk_i4_k>(env, "euclideans_symmetric_i4_ampere", nk_euclideans_symmetric_i4_ampere,
                                      backend);
    run_euclideans_symmetric<nk_u8_k>(env, "euclideans_symmetric_u8_ampere", nk_euclideans_symmetric_u8_ampere,
                                      backend);
    run_euclideans_symmetric<nk_u4_k>(env, "euclideans_symmetric_u4_ampere", nk_euclideans_symmetric_u4_ampere,
                                      backend);

    run_attention_bidirectional<nk_bf16_k>(env, "attention_bidirectional_packed_bf16_ampere",
                                           nk_attention_pack_size_bf16_ampere, nk_attention_pack_bf16_ampere,
                                           nk_attention_bidirectional_packed_bf16_ampere, backend);
    run_attention_causal<nk_bf16_k>(env, "attention_causal_packed_bf16_ampere", nk_attention_pack_size_bf16_ampere,
                                    nk_attention_pack_bf16_ampere, nk_attention_causal_packed_bf16_ampere, backend);
    run_attention_bidirectional<nk_e4m3_k>(env, "attention_bidirectional_packed_e4m3_ampere",
                                           nk_attention_pack_size_e4m3_ampere, nk_attention_pack_e4m3_ampere,
                                           nk_attention_bidirectional_packed_e4m3_ampere, backend);
    run_attention_causal<nk_e4m3_k>(env, "attention_causal_packed_e4m3_ampere", nk_attention_pack_size_e4m3_ampere,
                                    nk_attention_pack_e4m3_ampere, nk_attention_causal_packed_e4m3_ampere, backend);
    run_attention_bidirectional<nk_i8_k>(env, "attention_bidirectional_packed_i8_ampere",
                                         nk_attention_pack_size_i8_ampere, nk_attention_pack_i8_ampere,
                                         nk_attention_bidirectional_packed_i8_ampere, backend);
    run_attention_causal<nk_i8_k>(env, "attention_causal_packed_i8_ampere", nk_attention_pack_size_i8_ampere,
                                  nk_attention_pack_i8_ampere, nk_attention_causal_packed_i8_ampere, backend);
#endif // NUMKONG_TARGET_AMPERE
}

/** Every Hopper entry point, compiled only when the architecture list includes "90a". */
void bench_cross_hopper([[maybe_unused]] environment_t const &env, [[maybe_unused]] cuda_backend_t const &backend,
                        [[maybe_unused]] nk_capability_t enabled) {
#if NUMKONG_TARGET_HOPPER
    if (!(enabled & nk_cap_hopper_k)) return;
    run_dots_packed<nk_bf16_k>(env, "dots_packed_bf16_hopper", nk_dots_pack_size_bf16_hopper, nk_dots_pack_bf16_hopper,
                               nk_dots_packed_bf16_hopper, backend);
    run_dots_packed<nk_f16_k>(env, "dots_packed_f16_hopper", nk_dots_pack_size_f16_hopper, nk_dots_pack_f16_hopper,
                              nk_dots_packed_f16_hopper, backend);
    run_dots_packed<nk_e2m3_k>(env, "dots_packed_e2m3_hopper", nk_dots_pack_size_e2m3_hopper, nk_dots_pack_e2m3_hopper,
                               nk_dots_packed_e2m3_hopper, backend);
    run_dots_packed<nk_e2m1_k>(env, "dots_packed_e2m1_hopper", nk_dots_pack_size_e2m1_hopper, nk_dots_pack_e2m1_hopper,
                               nk_dots_packed_e2m1_hopper, backend);
    run_dots_packed<nk_i8_k>(env, "dots_packed_i8_hopper", nk_dots_pack_size_i8_hopper, nk_dots_pack_i8_hopper,
                             nk_dots_packed_i8_hopper, backend);
    run_dots_packed<nk_i4_k>(env, "dots_packed_i4_hopper", nk_dots_pack_size_i4_hopper, nk_dots_pack_i4_hopper,
                             nk_dots_packed_i4_hopper, backend);
    run_dots_packed<nk_u8_k>(env, "dots_packed_u8_hopper", nk_dots_pack_size_u8_hopper, nk_dots_pack_u8_hopper,
                             nk_dots_packed_u8_hopper, backend);
    run_dots_packed<nk_u4_k>(env, "dots_packed_u4_hopper", nk_dots_pack_size_u4_hopper, nk_dots_pack_u4_hopper,
                             nk_dots_packed_u4_hopper, backend);

    run_dots_symmetric<nk_bf16_k>(env, "dots_symmetric_bf16_hopper", nk_dots_symmetric_bf16_hopper, backend);
    run_dots_symmetric<nk_f16_k>(env, "dots_symmetric_f16_hopper", nk_dots_symmetric_f16_hopper, backend);
    run_dots_symmetric<nk_e2m3_k>(env, "dots_symmetric_e2m3_hopper", nk_dots_symmetric_e2m3_hopper, backend);
    run_dots_symmetric<nk_e2m1_k>(env, "dots_symmetric_e2m1_hopper", nk_dots_symmetric_e2m1_hopper, backend);
    run_dots_symmetric<nk_i8_k>(env, "dots_symmetric_i8_hopper", nk_dots_symmetric_i8_hopper, backend);
    run_dots_symmetric<nk_i4_k>(env, "dots_symmetric_i4_hopper", nk_dots_symmetric_i4_hopper, backend);
    run_dots_symmetric<nk_u8_k>(env, "dots_symmetric_u8_hopper", nk_dots_symmetric_u8_hopper, backend);
    run_dots_symmetric<nk_u4_k>(env, "dots_symmetric_u4_hopper", nk_dots_symmetric_u4_hopper, backend);

    run_angulars_packed<nk_bf16_k>(env, "angulars_packed_bf16_hopper", nk_dots_pack_size_bf16_hopper,
                                   nk_dots_pack_bf16_hopper, nk_angulars_packed_bf16_hopper, backend);
    run_angulars_packed<nk_f16_k>(env, "angulars_packed_f16_hopper", nk_dots_pack_size_f16_hopper,
                                  nk_dots_pack_f16_hopper, nk_angulars_packed_f16_hopper, backend);
    run_angulars_packed<nk_e2m3_k>(env, "angulars_packed_e2m3_hopper", nk_dots_pack_size_e2m3_hopper,
                                   nk_dots_pack_e2m3_hopper, nk_angulars_packed_e2m3_hopper, backend);
    run_angulars_packed<nk_e2m1_k>(env, "angulars_packed_e2m1_hopper", nk_dots_pack_size_e2m1_hopper,
                                   nk_dots_pack_e2m1_hopper, nk_angulars_packed_e2m1_hopper, backend);
    run_angulars_packed<nk_i8_k>(env, "angulars_packed_i8_hopper", nk_dots_pack_size_i8_hopper, nk_dots_pack_i8_hopper,
                                 nk_angulars_packed_i8_hopper, backend);
    run_angulars_packed<nk_i4_k>(env, "angulars_packed_i4_hopper", nk_dots_pack_size_i4_hopper, nk_dots_pack_i4_hopper,
                                 nk_angulars_packed_i4_hopper, backend);
    run_angulars_packed<nk_u8_k>(env, "angulars_packed_u8_hopper", nk_dots_pack_size_u8_hopper, nk_dots_pack_u8_hopper,
                                 nk_angulars_packed_u8_hopper, backend);
    run_angulars_packed<nk_u4_k>(env, "angulars_packed_u4_hopper", nk_dots_pack_size_u4_hopper, nk_dots_pack_u4_hopper,
                                 nk_angulars_packed_u4_hopper, backend);

    run_angulars_symmetric<nk_bf16_k>(env, "angulars_symmetric_bf16_hopper", nk_angulars_symmetric_bf16_hopper,
                                      backend);
    run_angulars_symmetric<nk_f16_k>(env, "angulars_symmetric_f16_hopper", nk_angulars_symmetric_f16_hopper, backend);
    run_angulars_symmetric<nk_e2m3_k>(env, "angulars_symmetric_e2m3_hopper", nk_angulars_symmetric_e2m3_hopper,
                                      backend);
    run_angulars_symmetric<nk_e2m1_k>(env, "angulars_symmetric_e2m1_hopper", nk_angulars_symmetric_e2m1_hopper,
                                      backend);
    run_angulars_symmetric<nk_i8_k>(env, "angulars_symmetric_i8_hopper", nk_angulars_symmetric_i8_hopper, backend);
    run_angulars_symmetric<nk_i4_k>(env, "angulars_symmetric_i4_hopper", nk_angulars_symmetric_i4_hopper, backend);
    run_angulars_symmetric<nk_u8_k>(env, "angulars_symmetric_u8_hopper", nk_angulars_symmetric_u8_hopper, backend);
    run_angulars_symmetric<nk_u4_k>(env, "angulars_symmetric_u4_hopper", nk_angulars_symmetric_u4_hopper, backend);

    run_euclideans_packed<nk_bf16_k>(env, "euclideans_packed_bf16_hopper", nk_dots_pack_size_bf16_hopper,
                                     nk_dots_pack_bf16_hopper, nk_euclideans_packed_bf16_hopper, backend);
    run_euclideans_packed<nk_f16_k>(env, "euclideans_packed_f16_hopper", nk_dots_pack_size_f16_hopper,
                                    nk_dots_pack_f16_hopper, nk_euclideans_packed_f16_hopper, backend);
    run_euclideans_packed<nk_e2m3_k>(env, "euclideans_packed_e2m3_hopper", nk_dots_pack_size_e2m3_hopper,
                                     nk_dots_pack_e2m3_hopper, nk_euclideans_packed_e2m3_hopper, backend);
    run_euclideans_packed<nk_e2m1_k>(env, "euclideans_packed_e2m1_hopper", nk_dots_pack_size_e2m1_hopper,
                                     nk_dots_pack_e2m1_hopper, nk_euclideans_packed_e2m1_hopper, backend);
    run_euclideans_packed<nk_i8_k>(env, "euclideans_packed_i8_hopper", nk_dots_pack_size_i8_hopper,
                                   nk_dots_pack_i8_hopper, nk_euclideans_packed_i8_hopper, backend);
    run_euclideans_packed<nk_i4_k>(env, "euclideans_packed_i4_hopper", nk_dots_pack_size_i4_hopper,
                                   nk_dots_pack_i4_hopper, nk_euclideans_packed_i4_hopper, backend);
    run_euclideans_packed<nk_u8_k>(env, "euclideans_packed_u8_hopper", nk_dots_pack_size_u8_hopper,
                                   nk_dots_pack_u8_hopper, nk_euclideans_packed_u8_hopper, backend);
    run_euclideans_packed<nk_u4_k>(env, "euclideans_packed_u4_hopper", nk_dots_pack_size_u4_hopper,
                                   nk_dots_pack_u4_hopper, nk_euclideans_packed_u4_hopper, backend);

    run_euclideans_symmetric<nk_bf16_k>(env, "euclideans_symmetric_bf16_hopper", nk_euclideans_symmetric_bf16_hopper,
                                        backend);
    run_euclideans_symmetric<nk_f16_k>(env, "euclideans_symmetric_f16_hopper", nk_euclideans_symmetric_f16_hopper,
                                       backend);
    run_euclideans_symmetric<nk_e2m3_k>(env, "euclideans_symmetric_e2m3_hopper", nk_euclideans_symmetric_e2m3_hopper,
                                        backend);
    run_euclideans_symmetric<nk_e2m1_k>(env, "euclideans_symmetric_e2m1_hopper", nk_euclideans_symmetric_e2m1_hopper,
                                        backend);
    run_euclideans_symmetric<nk_i8_k>(env, "euclideans_symmetric_i8_hopper", nk_euclideans_symmetric_i8_hopper,
                                      backend);
    run_euclideans_symmetric<nk_i4_k>(env, "euclideans_symmetric_i4_hopper", nk_euclideans_symmetric_i4_hopper,
                                      backend);
    run_euclideans_symmetric<nk_u8_k>(env, "euclideans_symmetric_u8_hopper", nk_euclideans_symmetric_u8_hopper,
                                      backend);
    run_euclideans_symmetric<nk_u4_k>(env, "euclideans_symmetric_u4_hopper", nk_euclideans_symmetric_u4_hopper,
                                      backend);

    run_attention_bidirectional<nk_bf16_k>(env, "attention_bidirectional_packed_bf16_hopper",
                                           nk_attention_pack_size_bf16_hopper, nk_attention_pack_bf16_hopper,
                                           nk_attention_bidirectional_packed_bf16_hopper, backend);
    run_attention_causal<nk_bf16_k>(env, "attention_causal_packed_bf16_hopper", nk_attention_pack_size_bf16_hopper,
                                    nk_attention_pack_bf16_hopper, nk_attention_causal_packed_bf16_hopper, backend);
    run_attention_bidirectional<nk_e4m3_k>(env, "attention_bidirectional_packed_e4m3_hopper",
                                           nk_attention_pack_size_e4m3_hopper, nk_attention_pack_e4m3_hopper,
                                           nk_attention_bidirectional_packed_e4m3_hopper, backend);
    run_attention_causal<nk_e4m3_k>(env, "attention_causal_packed_e4m3_hopper", nk_attention_pack_size_e4m3_hopper,
                                    nk_attention_pack_e4m3_hopper, nk_attention_causal_packed_e4m3_hopper, backend);
    run_attention_bidirectional<nk_i8_k>(env, "attention_bidirectional_packed_i8_hopper",
                                         nk_attention_pack_size_i8_hopper, nk_attention_pack_i8_hopper,
                                         nk_attention_bidirectional_packed_i8_hopper, backend);
    run_attention_causal<nk_i8_k>(env, "attention_causal_packed_i8_hopper", nk_attention_pack_size_i8_hopper,
                                  nk_attention_pack_i8_hopper, nk_attention_causal_packed_i8_hopper, backend);
#endif // NUMKONG_TARGET_HOPPER
}

/** Every Blackwell entry point, compiled only when the architecture list includes the family. */
void bench_cross_blackwell([[maybe_unused]] environment_t const &env, [[maybe_unused]] cuda_backend_t const &backend,
                           [[maybe_unused]] nk_capability_t enabled) {
#if NUMKONG_TARGET_BLACKWELL
    if (!(enabled & nk_cap_blackwell_k)) return;
    run_dots_packed<nk_bf16_k>(env, "dots_packed_bf16_blackwell", nk_dots_pack_size_bf16_blackwell,
                               nk_dots_pack_bf16_blackwell, nk_dots_packed_bf16_blackwell, backend);
    run_dots_packed<nk_f16_k>(env, "dots_packed_f16_blackwell", nk_dots_pack_size_f16_blackwell,
                              nk_dots_pack_f16_blackwell, nk_dots_packed_f16_blackwell, backend);
    run_dots_packed<nk_e5m2_k>(env, "dots_packed_e5m2_blackwell", nk_dots_pack_size_e5m2_blackwell,
                               nk_dots_pack_e5m2_blackwell, nk_dots_packed_e5m2_blackwell, backend);
    run_dots_packed<nk_e4m3_k>(env, "dots_packed_e4m3_blackwell", nk_dots_pack_size_e4m3_blackwell,
                               nk_dots_pack_e4m3_blackwell, nk_dots_packed_e4m3_blackwell, backend);
    run_dots_packed<nk_e3m2_k>(env, "dots_packed_e3m2_blackwell", nk_dots_pack_size_e3m2_blackwell,
                               nk_dots_pack_e3m2_blackwell, nk_dots_packed_e3m2_blackwell, backend);
    run_dots_packed<nk_e2m3_k>(env, "dots_packed_e2m3_blackwell", nk_dots_pack_size_e2m3_blackwell,
                               nk_dots_pack_e2m3_blackwell, nk_dots_packed_e2m3_blackwell, backend);
    run_dots_packed<nk_e2m1_k>(env, "dots_packed_e2m1_blackwell", nk_dots_pack_size_e2m1_blackwell,
                               nk_dots_pack_e2m1_blackwell, nk_dots_packed_e2m1_blackwell, backend);

    run_dots_symmetric<nk_bf16_k>(env, "dots_symmetric_bf16_blackwell", nk_dots_symmetric_bf16_blackwell, backend);
    run_dots_symmetric<nk_f16_k>(env, "dots_symmetric_f16_blackwell", nk_dots_symmetric_f16_blackwell, backend);
    run_dots_symmetric<nk_e5m2_k>(env, "dots_symmetric_e5m2_blackwell", nk_dots_symmetric_e5m2_blackwell, backend);
    run_dots_symmetric<nk_e4m3_k>(env, "dots_symmetric_e4m3_blackwell", nk_dots_symmetric_e4m3_blackwell, backend);
    run_dots_symmetric<nk_e3m2_k>(env, "dots_symmetric_e3m2_blackwell", nk_dots_symmetric_e3m2_blackwell, backend);
    run_dots_symmetric<nk_e2m3_k>(env, "dots_symmetric_e2m3_blackwell", nk_dots_symmetric_e2m3_blackwell, backend);
    run_dots_symmetric<nk_e2m1_k>(env, "dots_symmetric_e2m1_blackwell", nk_dots_symmetric_e2m1_blackwell, backend);

    run_angulars_packed<nk_bf16_k>(env, "angulars_packed_bf16_blackwell", nk_dots_pack_size_bf16_blackwell,
                                   nk_dots_pack_bf16_blackwell, nk_angulars_packed_bf16_blackwell, backend);
    run_angulars_packed<nk_f16_k>(env, "angulars_packed_f16_blackwell", nk_dots_pack_size_f16_blackwell,
                                  nk_dots_pack_f16_blackwell, nk_angulars_packed_f16_blackwell, backend);
    run_angulars_packed<nk_e5m2_k>(env, "angulars_packed_e5m2_blackwell", nk_dots_pack_size_e5m2_blackwell,
                                   nk_dots_pack_e5m2_blackwell, nk_angulars_packed_e5m2_blackwell, backend);
    run_angulars_packed<nk_e4m3_k>(env, "angulars_packed_e4m3_blackwell", nk_dots_pack_size_e4m3_blackwell,
                                   nk_dots_pack_e4m3_blackwell, nk_angulars_packed_e4m3_blackwell, backend);
    run_angulars_packed<nk_e3m2_k>(env, "angulars_packed_e3m2_blackwell", nk_dots_pack_size_e3m2_blackwell,
                                   nk_dots_pack_e3m2_blackwell, nk_angulars_packed_e3m2_blackwell, backend);
    run_angulars_packed<nk_e2m3_k>(env, "angulars_packed_e2m3_blackwell", nk_dots_pack_size_e2m3_blackwell,
                                   nk_dots_pack_e2m3_blackwell, nk_angulars_packed_e2m3_blackwell, backend);
    run_angulars_packed<nk_e2m1_k>(env, "angulars_packed_e2m1_blackwell", nk_dots_pack_size_e2m1_blackwell,
                                   nk_dots_pack_e2m1_blackwell, nk_angulars_packed_e2m1_blackwell, backend);

    run_angulars_symmetric<nk_bf16_k>(env, "angulars_symmetric_bf16_blackwell", nk_angulars_symmetric_bf16_blackwell,
                                      backend);
    run_angulars_symmetric<nk_f16_k>(env, "angulars_symmetric_f16_blackwell", nk_angulars_symmetric_f16_blackwell,
                                     backend);
    run_angulars_symmetric<nk_e5m2_k>(env, "angulars_symmetric_e5m2_blackwell", nk_angulars_symmetric_e5m2_blackwell,
                                      backend);
    run_angulars_symmetric<nk_e4m3_k>(env, "angulars_symmetric_e4m3_blackwell", nk_angulars_symmetric_e4m3_blackwell,
                                      backend);
    run_angulars_symmetric<nk_e3m2_k>(env, "angulars_symmetric_e3m2_blackwell", nk_angulars_symmetric_e3m2_blackwell,
                                      backend);
    run_angulars_symmetric<nk_e2m3_k>(env, "angulars_symmetric_e2m3_blackwell", nk_angulars_symmetric_e2m3_blackwell,
                                      backend);
    run_angulars_symmetric<nk_e2m1_k>(env, "angulars_symmetric_e2m1_blackwell", nk_angulars_symmetric_e2m1_blackwell,
                                      backend);

    run_euclideans_packed<nk_bf16_k>(env, "euclideans_packed_bf16_blackwell", nk_dots_pack_size_bf16_blackwell,
                                     nk_dots_pack_bf16_blackwell, nk_euclideans_packed_bf16_blackwell, backend);
    run_euclideans_packed<nk_f16_k>(env, "euclideans_packed_f16_blackwell", nk_dots_pack_size_f16_blackwell,
                                    nk_dots_pack_f16_blackwell, nk_euclideans_packed_f16_blackwell, backend);
    run_euclideans_packed<nk_e5m2_k>(env, "euclideans_packed_e5m2_blackwell", nk_dots_pack_size_e5m2_blackwell,
                                     nk_dots_pack_e5m2_blackwell, nk_euclideans_packed_e5m2_blackwell, backend);
    run_euclideans_packed<nk_e4m3_k>(env, "euclideans_packed_e4m3_blackwell", nk_dots_pack_size_e4m3_blackwell,
                                     nk_dots_pack_e4m3_blackwell, nk_euclideans_packed_e4m3_blackwell, backend);
    run_euclideans_packed<nk_e3m2_k>(env, "euclideans_packed_e3m2_blackwell", nk_dots_pack_size_e3m2_blackwell,
                                     nk_dots_pack_e3m2_blackwell, nk_euclideans_packed_e3m2_blackwell, backend);
    run_euclideans_packed<nk_e2m3_k>(env, "euclideans_packed_e2m3_blackwell", nk_dots_pack_size_e2m3_blackwell,
                                     nk_dots_pack_e2m3_blackwell, nk_euclideans_packed_e2m3_blackwell, backend);
    run_euclideans_packed<nk_e2m1_k>(env, "euclideans_packed_e2m1_blackwell", nk_dots_pack_size_e2m1_blackwell,
                                     nk_dots_pack_e2m1_blackwell, nk_euclideans_packed_e2m1_blackwell, backend);

    run_euclideans_symmetric<nk_bf16_k>(env, "euclideans_symmetric_bf16_blackwell",
                                        nk_euclideans_symmetric_bf16_blackwell, backend);
    run_euclideans_symmetric<nk_f16_k>(env, "euclideans_symmetric_f16_blackwell", nk_euclideans_symmetric_f16_blackwell,
                                       backend);
    run_euclideans_symmetric<nk_e5m2_k>(env, "euclideans_symmetric_e5m2_blackwell",
                                        nk_euclideans_symmetric_e5m2_blackwell, backend);
    run_euclideans_symmetric<nk_e4m3_k>(env, "euclideans_symmetric_e4m3_blackwell",
                                        nk_euclideans_symmetric_e4m3_blackwell, backend);
    run_euclideans_symmetric<nk_e3m2_k>(env, "euclideans_symmetric_e3m2_blackwell",
                                        nk_euclideans_symmetric_e3m2_blackwell, backend);
    run_euclideans_symmetric<nk_e2m3_k>(env, "euclideans_symmetric_e2m3_blackwell",
                                        nk_euclideans_symmetric_e2m3_blackwell, backend);
    run_euclideans_symmetric<nk_e2m1_k>(env, "euclideans_symmetric_e2m1_blackwell",
                                        nk_euclideans_symmetric_e2m1_blackwell, backend);
    run_attention_bidirectional<nk_e4m3_k>(env, "attention_bidirectional_packed_e4m3_blackwell",
                                           nk_attention_pack_size_e4m3_blackwell, nk_attention_pack_e4m3_blackwell,
                                           nk_attention_bidirectional_packed_e4m3_blackwell, backend);
    run_attention_causal<nk_e4m3_k>(env, "attention_causal_packed_e4m3_blackwell",
                                    nk_attention_pack_size_e4m3_blackwell, nk_attention_pack_e4m3_blackwell,
                                    nk_attention_causal_packed_e4m3_blackwell, backend);
#endif // NUMKONG_TARGET_BLACKWELL
}

/** Every Blackwell RTX entry point, compiled only when the architecture list includes it. */
void bench_cross_blackwellrtx([[maybe_unused]] environment_t const &env, [[maybe_unused]] cuda_backend_t const &backend,
                              [[maybe_unused]] nk_capability_t enabled) {
#if NUMKONG_TARGET_BLACKWELLRTX
    if (!(enabled & nk_cap_blackwellrtx_k)) return;
    run_dots_packed<nk_e5m2_k>(env, "dots_packed_e5m2_blackwellrtx", nk_dots_pack_size_e5m2_blackwellrtx,
                               nk_dots_pack_e5m2_blackwellrtx, nk_dots_packed_e5m2_blackwellrtx, backend);
    run_dots_packed<nk_e4m3_k>(env, "dots_packed_e4m3_blackwellrtx", nk_dots_pack_size_e4m3_blackwellrtx,
                               nk_dots_pack_e4m3_blackwellrtx, nk_dots_packed_e4m3_blackwellrtx, backend);
    run_dots_packed<nk_e3m2_k>(env, "dots_packed_e3m2_blackwellrtx", nk_dots_pack_size_e3m2_blackwellrtx,
                               nk_dots_pack_e3m2_blackwellrtx, nk_dots_packed_e3m2_blackwellrtx, backend);
    run_dots_packed<nk_e2m3_k>(env, "dots_packed_e2m3_blackwellrtx", nk_dots_pack_size_e2m3_blackwellrtx,
                               nk_dots_pack_e2m3_blackwellrtx, nk_dots_packed_e2m3_blackwellrtx, backend);
    run_dots_packed<nk_e2m1_k>(env, "dots_packed_e2m1_blackwellrtx", nk_dots_pack_size_e2m1_blackwellrtx,
                               nk_dots_pack_e2m1_blackwellrtx, nk_dots_packed_e2m1_blackwellrtx, backend);

    run_dots_symmetric<nk_e5m2_k>(env, "dots_symmetric_e5m2_blackwellrtx", nk_dots_symmetric_e5m2_blackwellrtx,
                                  backend);
    run_dots_symmetric<nk_e4m3_k>(env, "dots_symmetric_e4m3_blackwellrtx", nk_dots_symmetric_e4m3_blackwellrtx,
                                  backend);
    run_dots_symmetric<nk_e3m2_k>(env, "dots_symmetric_e3m2_blackwellrtx", nk_dots_symmetric_e3m2_blackwellrtx,
                                  backend);
    run_dots_symmetric<nk_e2m3_k>(env, "dots_symmetric_e2m3_blackwellrtx", nk_dots_symmetric_e2m3_blackwellrtx,
                                  backend);
    run_dots_symmetric<nk_e2m1_k>(env, "dots_symmetric_e2m1_blackwellrtx", nk_dots_symmetric_e2m1_blackwellrtx,
                                  backend);

    run_angulars_packed<nk_e5m2_k>(env, "angulars_packed_e5m2_blackwellrtx", nk_dots_pack_size_e5m2_blackwellrtx,
                                   nk_dots_pack_e5m2_blackwellrtx, nk_angulars_packed_e5m2_blackwellrtx, backend);
    run_angulars_packed<nk_e4m3_k>(env, "angulars_packed_e4m3_blackwellrtx", nk_dots_pack_size_e4m3_blackwellrtx,
                                   nk_dots_pack_e4m3_blackwellrtx, nk_angulars_packed_e4m3_blackwellrtx, backend);
    run_angulars_packed<nk_e3m2_k>(env, "angulars_packed_e3m2_blackwellrtx", nk_dots_pack_size_e3m2_blackwellrtx,
                                   nk_dots_pack_e3m2_blackwellrtx, nk_angulars_packed_e3m2_blackwellrtx, backend);
    run_angulars_packed<nk_e2m3_k>(env, "angulars_packed_e2m3_blackwellrtx", nk_dots_pack_size_e2m3_blackwellrtx,
                                   nk_dots_pack_e2m3_blackwellrtx, nk_angulars_packed_e2m3_blackwellrtx, backend);
    run_angulars_packed<nk_e2m1_k>(env, "angulars_packed_e2m1_blackwellrtx", nk_dots_pack_size_e2m1_blackwellrtx,
                                   nk_dots_pack_e2m1_blackwellrtx, nk_angulars_packed_e2m1_blackwellrtx, backend);

    run_angulars_symmetric<nk_e5m2_k>(env, "angulars_symmetric_e5m2_blackwellrtx",
                                      nk_angulars_symmetric_e5m2_blackwellrtx, backend);
    run_angulars_symmetric<nk_e4m3_k>(env, "angulars_symmetric_e4m3_blackwellrtx",
                                      nk_angulars_symmetric_e4m3_blackwellrtx, backend);
    run_angulars_symmetric<nk_e3m2_k>(env, "angulars_symmetric_e3m2_blackwellrtx",
                                      nk_angulars_symmetric_e3m2_blackwellrtx, backend);
    run_angulars_symmetric<nk_e2m3_k>(env, "angulars_symmetric_e2m3_blackwellrtx",
                                      nk_angulars_symmetric_e2m3_blackwellrtx, backend);
    run_angulars_symmetric<nk_e2m1_k>(env, "angulars_symmetric_e2m1_blackwellrtx",
                                      nk_angulars_symmetric_e2m1_blackwellrtx, backend);

    run_euclideans_packed<nk_e5m2_k>(env, "euclideans_packed_e5m2_blackwellrtx", nk_dots_pack_size_e5m2_blackwellrtx,
                                     nk_dots_pack_e5m2_blackwellrtx, nk_euclideans_packed_e5m2_blackwellrtx, backend);
    run_euclideans_packed<nk_e4m3_k>(env, "euclideans_packed_e4m3_blackwellrtx", nk_dots_pack_size_e4m3_blackwellrtx,
                                     nk_dots_pack_e4m3_blackwellrtx, nk_euclideans_packed_e4m3_blackwellrtx, backend);
    run_euclideans_packed<nk_e3m2_k>(env, "euclideans_packed_e3m2_blackwellrtx", nk_dots_pack_size_e3m2_blackwellrtx,
                                     nk_dots_pack_e3m2_blackwellrtx, nk_euclideans_packed_e3m2_blackwellrtx, backend);
    run_euclideans_packed<nk_e2m3_k>(env, "euclideans_packed_e2m3_blackwellrtx", nk_dots_pack_size_e2m3_blackwellrtx,
                                     nk_dots_pack_e2m3_blackwellrtx, nk_euclideans_packed_e2m3_blackwellrtx, backend);
    run_euclideans_packed<nk_e2m1_k>(env, "euclideans_packed_e2m1_blackwellrtx", nk_dots_pack_size_e2m1_blackwellrtx,
                                     nk_dots_pack_e2m1_blackwellrtx, nk_euclideans_packed_e2m1_blackwellrtx, backend);

    run_euclideans_symmetric<nk_e5m2_k>(env, "euclideans_symmetric_e5m2_blackwellrtx",
                                        nk_euclideans_symmetric_e5m2_blackwellrtx, backend);
    run_euclideans_symmetric<nk_e4m3_k>(env, "euclideans_symmetric_e4m3_blackwellrtx",
                                        nk_euclideans_symmetric_e4m3_blackwellrtx, backend);
    run_euclideans_symmetric<nk_e3m2_k>(env, "euclideans_symmetric_e3m2_blackwellrtx",
                                        nk_euclideans_symmetric_e3m2_blackwellrtx, backend);
    run_euclideans_symmetric<nk_e2m3_k>(env, "euclideans_symmetric_e2m3_blackwellrtx",
                                        nk_euclideans_symmetric_e2m3_blackwellrtx, backend);
    run_euclideans_symmetric<nk_e2m1_k>(env, "euclideans_symmetric_e2m1_blackwellrtx",
                                        nk_euclideans_symmetric_e2m1_blackwellrtx, backend);

    run_attention_bidirectional<nk_e4m3_k>(
        env, "attention_bidirectional_packed_e4m3_blackwellrtx", nk_attention_pack_size_e4m3_blackwellrtx,
        nk_attention_pack_e4m3_blackwellrtx, nk_attention_bidirectional_packed_e4m3_blackwellrtx, backend);
    run_attention_causal<nk_e4m3_k>(env, "attention_causal_packed_e4m3_blackwellrtx",
                                    nk_attention_pack_size_e4m3_blackwellrtx, nk_attention_pack_e4m3_blackwellrtx,
                                    nk_attention_causal_packed_e4m3_blackwellrtx, backend);
#endif // NUMKONG_TARGET_BLACKWELLRTX
}

#pragma endregion Rows

#pragma region cuBLAS
#if NUMKONG_COMPARE_TO_CUBLAS

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

/** cuBLASLt's accumulator, scalar and output type for @p dtype: F64, I32 for integers, F32 for
 *  other floats. */
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

/** Elements per block scale cuBLASLt requires of @p dtype: 32 for 6-bit floats, 16 for 4-bit ones,
 *  0 for none. */
std::size_t cublaslt_scale_block(nk_dtype_t dtype) noexcept {
    switch (dtype) {
    case nk_e3m2_k:
    case nk_e2m3_k: return 32;
    case nk_e2m1_k: return 16;
    default: return 0;
    }
}

/**
 *  @brief One planned @c cublasLtMatmul for row-major C = A × Bᵀ.
 *
 *  Row-major C is column-major Cᵀ = B × Aᵀ, so B goes in as the transposed first operand and A as
 *  the second. Leading dimensions count elements, so a nibble-pair row of K elements has a leading
 *  dimension of K. Block-scaled inputs share one buffer of unit scales between A and B, and the
 *  output type also types alpha and beta.
 */
struct cublaslt_plan_t {
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t first_layout = nullptr;
    cublasLtMatrixLayout_t second_layout = nullptr;
    cublasLtMatrixLayout_t output_layout = nullptr;
    cublasLtMatmulHeuristicResult_t heuristic {};
    device_vector<char> workspace;
    device_vector<char> scales;
    cudaDataType_t output_type = CUDA_R_32F;

    ~cublaslt_plan_t() {
        if (output_layout) cublasLtMatrixLayoutDestroy(output_layout);
        if (second_layout) cublasLtMatrixLayoutDestroy(second_layout);
        if (first_layout) cublasLtMatrixLayoutDestroy(first_layout);
        if (operation) cublasLtMatmulDescDestroy(operation);
        if (handle) cublasLtDestroy(handle);
    }

    /** Builds the plan for A of @p a_leading elements per row, or returns why cuBLASLt offers no
     *  algorithm. */
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
            scales = device_vector<char>::uninitialized(nk::divide_round_up(std::max(height, width), std::size_t(128)) *
                                                        128 * nk::divide_round_up(depth, block * 4) * 4)
                         .value;
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
        workspace = device_vector<char>::uninitialized(std::max<std::size_t>(heuristic.workspaceSize, 1)).value;
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

/** Runs a cuBLASLt row, or prints why cuBLASLt has no algorithm for it. */
template <nk_dtype_t input_dtype_, typename output_type_ = typename nk::type_for<input_dtype_>::type::dot_result_t>
void run_dots_with_cublaslt(environment_t const &env, std::string const &name, cuda_backend_t const &backend) {
    using input_t = typename nk::type_for<input_dtype_>::type;
    std::size_t const dimensions_per_value = nk::dimensions_per_value<input_t>();
    std::size_t const a_row_bytes = nk::divide_round_up(env.settings.matrix_depth, dimensions_per_value) *
                                    sizeof(input_t);
    auto const plan = std::make_shared<cublaslt_plan_t>();
    if (cublasStatus_t const status = plan->build(
            input_dtype_, env.settings.matrix_height, env.settings.matrix_width, env.settings.matrix_depth,
            backend.row_stride(a_row_bytes) / sizeof(input_t) * dimensions_per_value))
        return print_skipped(env, name, cublasLtGetStatusName(status));
    run_unpacked<input_dtype_, output_type_>(
        env, name, reference_metric_t::dot_k,
        [plan](void const *a, void const *, void const *b, void *c, std::size_t, std::size_t, std::size_t, std::size_t,
               std::size_t, std::size_t, void *stream) { return plan->launch(a, b, c, (cudaStream_t)stream); },
        backend);
}

/** Runs DGEMM through cuBLAS's fixed-point emulation, which only the handle API runs on 12.x
 *  devices. */
void run_dots_f64_with_cublas(environment_t const &env, std::string const &name, cuda_backend_t const &backend) {
    cublasHandle_t raw_handle = nullptr;
    if (cublasStatus_t const status = cublasCreate(&raw_handle))
        return print_skipped(env, name, cublasGetStatusName(status));
    std::shared_ptr<std::remove_pointer_t<cublasHandle_t>> const handle(raw_handle, cublasDestroy);
    cublasSetEmulationStrategy(raw_handle, CUBLAS_EMULATION_STRATEGY_EAGER);
    cublasSetMathMode(raw_handle, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH);
    run_unpacked<nk_f64_k, nk::f64_t>(
        env, name, reference_metric_t::dot_k,
        [handle](void const *a, void const *, void const *b, void *c, std::size_t height, std::size_t width,
                 std::size_t depth, std::size_t a_stride, std::size_t, std::size_t, void *stream) {
            double const alpha = 1, beta = 0;
            cublasSetStream(handle.get(), (cudaStream_t)stream);
            cublasStatus_t const status = cublasGemmEx(handle.get(), CUBLAS_OP_T, CUBLAS_OP_N, int(width), int(height),
                                                       int(depth), &alpha, b, CUDA_R_64F, int(depth), a, CUDA_R_64F,
                                                       int(a_stride / sizeof(double)), &beta, c, CUDA_R_64F, int(width),
                                                       CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, CUBLAS_GEMM_DEFAULT);
            return status == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cudaErrorUnknown;
        },
        backend);
}

#endif // NUMKONG_COMPARE_TO_CUBLAS

/** Every cuBLASLt row, one per dtype, and cuBLAS's emulated DGEMM. */
void bench_cross_cublas([[maybe_unused]] environment_t const &env, [[maybe_unused]] cuda_backend_t const &backend) {
#if NUMKONG_COMPARE_TO_CUBLAS
    run_dots_with_cublaslt<nk_f64_k>(env, "dots_packed_f64_with_cublaslt", backend);
    run_dots_f64_with_cublas(env, "dots_packed_f64_with_cublas", backend);
    run_dots_with_cublaslt<nk_f32_k, nk::f32_t>(env, "dots_packed_f32_with_cublaslt", backend);
    run_dots_with_cublaslt<nk_bf16_k>(env, "dots_packed_bf16_with_cublaslt", backend);
    run_dots_with_cublaslt<nk_f16_k>(env, "dots_packed_f16_with_cublaslt", backend);
    run_dots_with_cublaslt<nk_e5m2_k>(env, "dots_packed_e5m2_with_cublaslt", backend);
    run_dots_with_cublaslt<nk_e4m3_k>(env, "dots_packed_e4m3_with_cublaslt", backend);
    run_dots_with_cublaslt<nk_e3m2_k>(env, "dots_packed_e3m2_with_cublaslt", backend);
    run_dots_with_cublaslt<nk_e2m3_k>(env, "dots_packed_e2m3_with_cublaslt", backend);
    run_dots_with_cublaslt<nk_e2m1_k>(env, "dots_packed_e2m1_with_cublaslt", backend);
    run_dots_with_cublaslt<nk_i8_k>(env, "dots_packed_i8_with_cublaslt", backend);
    run_dots_with_cublaslt<nk_i4_k>(env, "dots_packed_i4_with_cublaslt", backend);
    run_dots_with_cublaslt<nk_u8_k>(env, "dots_packed_u8_with_cublaslt", backend);
    run_dots_with_cublaslt<nk_u4_k>(env, "dots_packed_u4_with_cublaslt", backend);
#endif // NUMKONG_COMPARE_TO_CUBLAS
}
#pragma endregion cuBLAS

#pragma region cuDNN
#if NUMKONG_COMPARE_TO_CUDNN

/** cuDNN's storage type for Q, K, V and O of @p dtype. */
cudnnDataType_t cudnn_data_type(nk_dtype_t dtype) noexcept {
    switch (dtype) {
    case nk_bf16_k: return CUDNN_DATA_BFLOAT16;
    case nk_e4m3_k: return CUDNN_DATA_FP8_E4M3;
    default: return CUDNN_DATA_FLOAT;
    }
}

/** Where a graph tensor lives: in device memory, in a host scalar passed by value, or only between
 *  operations. */
enum class cudnn_binding_t { device_k, host_k, virtual_k };

/** Graph tensor identifiers; from @c query_length_k on, each also indexes the 32-bit words of
 *  @c parameters. */
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
 *  Query heads are grouped over K and V heads, and causal masks are a bottom-right diagonal-band
 *  subgraph. E4M3 plans quantize probabilities with a scale of 256, so the 1/4096-sized weights of
 *  a flat softmax stay normal. The handle is bound to the per-thread stream, descriptors are
 *  destroyed in reverse, and the variant pack lists Q, K, V and O first. The softmax scale, the
 *  masked-score value and the window travel by value, and @c status keeps the first build failure.
 */
struct cudnn_attention_plan_t {
    cudnnHandle_t handle = nullptr;
    std::vector<cudnnBackendDescriptor_t> descriptors;
    cudnnBackendDescriptor_t plan = nullptr;
    std::vector<std::int64_t> uids;
    std::vector<void *> addresses;
    device_vector<std::uint32_t> parameters;
    device_vector<char> workspace;
    float scale = 0;
    float negative_infinity = -INFINITY;
    std::int32_t window = 0;
    std::size_t key_bytes = 0;
    cudnnStatus_t status = CUDNN_STATUS_SUCCESS;

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

    /** Builds the graph and the first plan cuDNN's heuristics offer that finalizes, or returns why
     *  none did. */
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
        parameters = device_vector<std::uint32_t>::uninitialized(words.size()).value;
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
        workspace = device_vector<char>::uninitialized(std::size_t(std::max<std::int64_t>(workspace_bytes, 1))).value;
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

/** Runs one cuDNN row of @p shape through @c run_attention_row, with K and V copied into each
 *  set's cache. */
template <nk_dtype_t input_dtype_, attention_visibility_t visibility_>
void run_attention_row_with_cudnn(environment_t const &env, std::string const &name, attention_shape_t shape,
                                  cuda_backend_t const &backend) {
    auto const plan = std::make_shared<cudnn_attention_plan_t>();
    if (cudnnStatus_t const status = plan->build(input_dtype_, visibility_, shape))
        return print_skipped(env, attention_row_name(name, visibility_, shape), cudnnGetErrorString(status));
    auto const packed_size = [](std::size_t key_value_heads, std::size_t depth, nk_u32_t const *lengths, std::size_t,
                                nk_size_t *bytes) {
        *bytes = 2 * std::size_t(lengths[0]) * key_value_heads * depth * nk_dtype_bits(input_dtype_) / 8;
        return nk_success_k;
    };
    auto const pack = [key_bytes = plan->key_bytes](void const *keys, void const *, void const *values, void const *,
                                                    std::size_t, std::size_t, nk_u32_t const *, nk_u32_t const *,
                                                    std::size_t, std::size_t, std::size_t, std::size_t, std::size_t,
                                                    void *packed, std::size_t, std::size_t, void *stream) {
        cudaMemcpyAsync(packed, keys, key_bytes, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
        return cudaMemcpyAsync(static_cast<char *>(packed) + key_bytes, values, key_bytes, cudaMemcpyDeviceToDevice,
                               (cudaStream_t)stream);
    };
    run_attention_row<input_dtype_, visibility_, cuda_backend_t>(
        env, name, packed_size, pack,
        [plan](void const *queries, void const *, void const *packed, void *output, auto...) {
            return plan->launch(queries, packed, output);
        },
        shape, backend);
}

/** Runs cuDNN rows beside the NumKong ones: bidirectional, causal, and windowed where the
 *  window clips. */
template <nk_dtype_t input_dtype_>
void run_attention_with_cudnn(environment_t const &env, std::string const &bidirectional_name,
                              std::string const &causal_name, cuda_backend_t const &backend) {
    for (attention_shape_t const shape : backend.attention_shapes(env)) {
        run_attention_row_with_cudnn<input_dtype_, attention_visibility_t::bidirectional_k>(env, bidirectional_name,
                                                                                            shape, backend);
        run_attention_row_with_cudnn<input_dtype_, attention_visibility_t::causal_k>(env, causal_name, shape, backend);
        if (attention_window_clips(shape))
            run_attention_row_with_cudnn<input_dtype_, attention_visibility_t::causal_window_1024_k>(env, causal_name,
                                                                                                     shape, backend);
    }
}

#endif // NUMKONG_COMPARE_TO_CUDNN

/** Every cuDNN row: BF16 and E4M3 attention, bidirectional, causal, and windowed where the window
 *  clips. */
void bench_cross_cudnn([[maybe_unused]] environment_t const &env, [[maybe_unused]] cuda_backend_t const &backend) {
#if NUMKONG_COMPARE_TO_CUDNN
    run_attention_with_cudnn<nk_bf16_k>(env, "attention_bidirectional_bf16_with_cudnn",
                                        "attention_causal_bf16_with_cudnn", backend);
    run_attention_with_cudnn<nk_e4m3_k>(env, "attention_bidirectional_e4m3_with_cudnn",
                                        "attention_causal_e4m3_with_cudnn", backend);
#endif // NUMKONG_COMPARE_TO_CUDNN
}
#pragma endregion cuDNN

#pragma region cuVS
#if NUMKONG_COMPARE_TO_CUVS

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

/** Runs @c cuvsPairwiseDistance as a cosine or L2-expanded row beside NumKong's angular or
 *  euclidean one. */
template <nk_dtype_t input_dtype_>
void run_spatials_with_cuvs(environment_t const &env, std::string const &name, reference_metric_t metric,
                            cuda_backend_t const &backend) {
    std::shared_ptr<cuvsResources_t> const resources(new cuvsResources_t {}, [](cuvsResources_t *resources) {
        cuvsResourcesDestroy(*resources);
        delete resources;
    });
    if (cuvsResourcesCreate(resources.get()) != CUVS_SUCCESS) return print_skipped(env, name, cuvsGetLastErrorText());
    cuvsDistanceType const distance = metric == reference_metric_t::angular_k ? CosineExpanded : L2SqrtExpanded;
    std::uint8_t const bits = std::uint8_t(nk_dtype_bits(input_dtype_));
    run_unpacked<input_dtype_, nk::f32_t>(
        env, name, metric,
        [resources, distance, bits](void const *a, void const *, void const *b, void *c, std::size_t height,
                                    std::size_t width, std::size_t depth, std::size_t a_stride, std::size_t,
                                    std::size_t, void *stream) {
            if (a_stride * 8 != depth * bits) return cudaErrorInvalidPitchValue;
            std::int64_t a_shape[2] = {std::int64_t(height), std::int64_t(depth)};
            std::int64_t b_shape[2] = {std::int64_t(width), std::int64_t(depth)};
            std::int64_t c_shape[2] = {std::int64_t(height), std::int64_t(width)};
            DLManagedTensor a_tensor = dlpack_matrix(a, a_shape, bits), b_tensor = dlpack_matrix(b, b_shape, bits),
                            c_tensor = dlpack_matrix(c, c_shape, 32);
            cuvsStreamSet(*resources, (cudaStream_t)stream);
            return cuvsPairwiseDistance(*resources, &a_tensor, &b_tensor, &c_tensor, distance, 2.0f) == CUVS_SUCCESS
                       ? cudaSuccess
                       : cudaErrorUnknown;
        },
        backend);
}

#endif // NUMKONG_COMPARE_TO_CUVS

/** Every cuVS row: cosine and L2 distances over F32 and F16. */
void bench_cross_cuvs([[maybe_unused]] environment_t const &env, [[maybe_unused]] cuda_backend_t const &backend) {
#if NUMKONG_COMPARE_TO_CUVS
    run_spatials_with_cuvs<nk_f32_k>(env, "angulars_packed_f32_with_cuvs", reference_metric_t::angular_k, backend);
    run_spatials_with_cuvs<nk_f32_k>(env, "euclideans_packed_f32_with_cuvs", reference_metric_t::euclidean_k, backend);
    run_spatials_with_cuvs<nk_f16_k>(env, "angulars_packed_f16_with_cuvs", reference_metric_t::angular_k, backend);
    run_spatials_with_cuvs<nk_f16_k>(env, "euclideans_packed_f16_with_cuvs", reference_metric_t::euclidean_k, backend);
#endif // NUMKONG_COMPARE_TO_CUVS
}
#pragma endregion cuVS

} // namespace ashvardanian::numkong::bench

#endif // NUMKONG_ARCH_CUDA_

namespace ashvardanian::numkong::bench {

/** Every CUDA row: the kernel families this device runs, then the baselines compiled in. */
void bench_cross_cuda([[maybe_unused]] environment_t const &env) {
#if NUMKONG_ARCH_CUDA_
    cudaDeviceProp properties {};
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        fmt::println("- CUDA: no device");
        return;
    }
    fmt::println("- CUDA: {} sm_{}{}", properties.name, properties.major, properties.minor);
    fmt::println("- CUDA baselines: cuBLAS {}, cuDNN {}, cuVS {}", NUMKONG_COMPARE_TO_CUBLAS ? "on" : "off",
                 NUMKONG_COMPARE_TO_CUDNN ? "on" : "off", NUMKONG_COMPARE_TO_CUVS ? "on" : "off");

    cuda_backend_t const backend {};
    nk_capability_t capabilities = 0;
    if (nk_cuda_capabilities_enabled(0, &capabilities) != nk_success_k) capabilities = 0;
    bench_cross_cuda(env, backend, capabilities);
    bench_each_cuda(env, backend, capabilities);
    bench_cross_ampere(env, backend, capabilities);
    bench_cross_hopper(env, backend, capabilities);
    bench_cross_blackwell(env, backend, capabilities);
    bench_cross_blackwellrtx(env, backend, capabilities);
    bench_cross_cublas(env, backend);
    bench_cross_cudnn(env, backend);
    bench_cross_cuvs(env, backend);
#endif // NUMKONG_ARCH_CUDA_
}

} // namespace ashvardanian::numkong::bench
