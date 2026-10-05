/**
 *  @file bench/cross_rocm.hip
 *  @author Ash Vardanian
 *  @date October 4, 2026
 *  @brief ROCm batch benchmarks using the shared cross-operation drivers.
 */
#include <algorithm>   // `std::min`, `std::max`
#include <bit>         // `std::bit_ceil`
#include <chrono>      // `std::chrono::steady_clock`
#include <type_traits> // `std::is_same_v`
#include <vector>      // `std::vector`

#include "numkong/numkong.h"
#include "cross.hpp"

#if NUMKONG_ARCH_ROCM_
#include <hip/hip_runtime.h>

namespace ashvardanian::numkong::bench {

/** Runs kernels over device memory, timing launch windows with ROCm events. */
struct rocm_backend_t : device_backend_t {
    std::size_t l2_bytes = 0;

    /** Row stride rounded to the kernels' 16-byte alignment. */
    static constexpr std::size_t row_stride(std::size_t row_bytes) noexcept { return (row_bytes + 15) / 16 * 16; }

    /** Rotation sets of @p per_set each: enough to cover twice the L2, at most
     *  @c input_sets_count. */
    std::size_t input_sets(bytes_t per_set) const noexcept {
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

    nk_status_t copy(void *destination, void const *source, std::size_t bytes) noexcept {
        return hipMemcpy(destination, source, bytes, hipMemcpyDefault) == hipSuccess ? nk_success_k
                                                                                     : nk_device_code_mismatch_k;
    }

    nk_status_t zero(void *destination, std::size_t bytes) noexcept {
        return hipMemset(destination, 0, bytes) == hipSuccess ? nk_success_k : nk_device_code_mismatch_k;
    }

    template <typename kernel_type_, typename... arguments_types_>
    nk_status_t call(kernel_type_ kernel, arguments_types_... arguments) noexcept {
        auto const status = kernel(arguments..., memory.stream);
        if constexpr (std::is_same_v<decltype(status), hipError_t const>)
            return status == hipSuccess ? nk_success_k : nk_device_code_mismatch_k;
        else return status;
    }

    template <typename launch_type_>
    nk_status_t time(loop_t &loop, std::size_t sets_count, launch_type_ &launch) {
        hipEvent_t start = nullptr, stop = nullptr;
        if (hipEventCreate(&start) != hipSuccess) return nk_device_code_mismatch_k;
        if (hipEventCreate(&stop) != hipSuccess) {
            hipEventDestroy(start);
            return nk_device_code_mismatch_k;
        }
        nk_status_t status = nk_success_k;
        std::size_t calls = 0, window = 1;
        for ([[maybe_unused]] std::size_t call : loop) {
            if (hipEventRecord(start, (hipStream_t)memory.stream) != hipSuccess) {
                status = nk_device_code_mismatch_k;
                break;
            }
            for (std::size_t index = 0; index != window; ++index) {
                status = launch((calls + index) & (sets_count - 1));
                if (status != nk_success_k) break;
            }
            if (status != nk_success_k) break;
            if (hipEventRecord(stop, (hipStream_t)memory.stream) != hipSuccess ||
                hipEventSynchronize(stop) != hipSuccess) {
                status = nk_device_code_mismatch_k;
                break;
            }
            float milliseconds = 0;
            if (hipEventElapsedTime(&milliseconds, start, stop) != hipSuccess) {
                status = nk_device_code_mismatch_k;
                break;
            }
            std::chrono::duration<float, std::milli> const elapsed {milliseconds};
            loop.add_window(elapsed, window);
            calls += window;
            if (elapsed < std::chrono::milliseconds(1)) window *= 2;
        }
        hipError_t const start_status = hipEventDestroy(start), stop_status = hipEventDestroy(stop);
        if (status != nk_success_k) return status;
        return start_status == hipSuccess && stop_status == hipSuccess ? nk_success_k : nk_device_code_mismatch_k;
    }

    nk_status_t synchronize() noexcept {
        return hipStreamSynchronize((hipStream_t)memory.stream) == hipSuccess ? nk_success_k
                                                                              : nk_device_code_mismatch_k;
    }
};

void bench_cross_rocm(environment_t const &env, rocm_backend_t const &backend, nk_capability_t enabled) {
    if (!(enabled & nk_cap_rocm_k)) return;
    run_dots_packed<nk_f64_k>(env, "dots_packed_f64_rocm", nk_dots_pack_size_f64_rocm, nk_dots_pack_f64_rocm,
                              nk_dots_packed_f64_rocm, backend);
    run_dots_packed<nk_f32_k>(env, "dots_packed_f32_rocm", nk_dots_pack_size_f32_rocm, nk_dots_pack_f32_rocm,
                              nk_dots_packed_f32_rocm, backend);
    run_dots_symmetric<nk_f64_k>(env, "dots_symmetric_f64_rocm", nk_dots_symmetric_f64_rocm, backend);
    run_dots_symmetric<nk_f32_k>(env, "dots_symmetric_f32_rocm", nk_dots_symmetric_f32_rocm, backend);
    run_angulars_packed<nk_f64_k>(env, "angulars_packed_f64_rocm", nk_dots_pack_size_f64_rocm, nk_dots_pack_f64_rocm,
                                  nk_angulars_packed_f64_rocm, backend);
    run_angulars_packed<nk_f32_k>(env, "angulars_packed_f32_rocm", nk_dots_pack_size_f32_rocm, nk_dots_pack_f32_rocm,
                                  nk_angulars_packed_f32_rocm, backend);
    run_angulars_symmetric<nk_f64_k>(env, "angulars_symmetric_f64_rocm", nk_angulars_symmetric_f64_rocm, backend);
    run_angulars_symmetric<nk_f32_k>(env, "angulars_symmetric_f32_rocm", nk_angulars_symmetric_f32_rocm, backend);
    run_euclideans_packed<nk_f64_k>(env, "euclideans_packed_f64_rocm", nk_dots_pack_size_f64_rocm,
                                    nk_dots_pack_f64_rocm, nk_euclideans_packed_f64_rocm, backend);
    run_euclideans_packed<nk_f32_k>(env, "euclideans_packed_f32_rocm", nk_dots_pack_size_f32_rocm,
                                    nk_dots_pack_f32_rocm, nk_euclideans_packed_f32_rocm, backend);
    run_euclideans_symmetric<nk_f64_k>(env, "euclideans_symmetric_f64_rocm", nk_euclideans_symmetric_f64_rocm, backend);
    run_euclideans_symmetric<nk_f32_k>(env, "euclideans_symmetric_f32_rocm", nk_euclideans_symmetric_f32_rocm, backend);
}

/** The element-wise and RoPE ROCm baseline rows over the backend's token rows. */
void bench_each_rocm(environment_t const &env, rocm_backend_t const &backend, nk_capability_t enabled) {
    if (!(enabled & nk_cap_rocm_k)) return;
    constexpr nk_kernel_kind_t sum_k = nk_kernel_each_sum_k, scale_k = nk_kernel_each_scale_k;
    constexpr nk_kernel_kind_t blend_k = nk_kernel_each_blend_k, fma_k = nk_kernel_each_fma_k;
    run_each<nk_f32_k, sum_k, nk_f32_k>(env, "each_sum_f32_rocm", nk_each_sum_f32_rocm, backend);
    run_each<nk_bf16_k, sum_k, nk_f32_k>(env, "each_sum_bf16_rocm", nk_each_sum_bf16_rocm, backend);
    run_rmsnorm<nk_f32_k>(env, "each_rmsnorm_f32_rocm", nk_each_rmsnorm_f32_rocm, backend);
    run_rmsnorm<nk_bf16_k>(env, "each_rmsnorm_bf16_rocm", nk_each_rmsnorm_bf16_rocm, backend);
    run_rmsnorm<nk_e4m3_k>(env, "each_rmsnorm_e4m3_rocm", nk_each_rmsnorm_e4m3_rocm, backend);
    run_swiglu<nk_f32_k>(env, "each_swiglu_f32_rocm", nk_each_swiglu_f32_rocm, backend);
    run_swiglu<nk_bf16_k>(env, "each_swiglu_bf16_rocm", nk_each_swiglu_bf16_rocm, backend);
    run_swiglu<nk_e4m3_k>(env, "each_swiglu_e4m3_rocm", nk_each_swiglu_e4m3_rocm, backend);
}

void bench_cross_cdna4([[maybe_unused]] environment_t const &env, [[maybe_unused]] rocm_backend_t const &backend,
                       [[maybe_unused]] nk_capability_t enabled) {
#if NUMKONG_TARGET_CDNA4
    if (!(enabled & nk_cap_cdna4_k)) return;
    run_dots_packed<nk_bf16_k>(env, "dots_packed_bf16_cdna4", nk_dots_pack_size_bf16_cdna4, nk_dots_pack_bf16_cdna4,
                               nk_dots_packed_bf16_cdna4, backend);
    run_dots_symmetric<nk_bf16_k>(env, "dots_symmetric_bf16_cdna4", nk_dots_symmetric_bf16_cdna4, backend);
    run_angulars_packed<nk_bf16_k>(env, "angulars_packed_bf16_cdna4", nk_dots_pack_size_bf16_cdna4,
                                   nk_dots_pack_bf16_cdna4, nk_angulars_packed_bf16_cdna4, backend);
    run_angulars_symmetric<nk_bf16_k>(env, "angulars_symmetric_bf16_cdna4", nk_angulars_symmetric_bf16_cdna4, backend);
    run_euclideans_packed<nk_bf16_k>(env, "euclideans_packed_bf16_cdna4", nk_dots_pack_size_bf16_cdna4,
                                     nk_dots_pack_bf16_cdna4, nk_euclideans_packed_bf16_cdna4, backend);
    run_euclideans_symmetric<nk_bf16_k>(env, "euclideans_symmetric_bf16_cdna4", nk_euclideans_symmetric_bf16_cdna4,
                                        backend);
    run_dots_packed<nk_f16_k>(env, "dots_packed_f16_cdna4", nk_dots_pack_size_f16_cdna4, nk_dots_pack_f16_cdna4,
                              nk_dots_packed_f16_cdna4, backend);
    run_dots_symmetric<nk_f16_k>(env, "dots_symmetric_f16_cdna4", nk_dots_symmetric_f16_cdna4, backend);
    run_angulars_packed<nk_f16_k>(env, "angulars_packed_f16_cdna4", nk_dots_pack_size_f16_cdna4, nk_dots_pack_f16_cdna4,
                                  nk_angulars_packed_f16_cdna4, backend);
    run_angulars_symmetric<nk_f16_k>(env, "angulars_symmetric_f16_cdna4", nk_angulars_symmetric_f16_cdna4, backend);
    run_euclideans_packed<nk_f16_k>(env, "euclideans_packed_f16_cdna4", nk_dots_pack_size_f16_cdna4,
                                    nk_dots_pack_f16_cdna4, nk_euclideans_packed_f16_cdna4, backend);
    run_euclideans_symmetric<nk_f16_k>(env, "euclideans_symmetric_f16_cdna4", nk_euclideans_symmetric_f16_cdna4,
                                       backend);
    run_dots_packed<nk_e4m3_k>(env, "dots_packed_e4m3_cdna4", nk_dots_pack_size_e4m3_cdna4, nk_dots_pack_e4m3_cdna4,
                               nk_dots_packed_e4m3_cdna4, backend);
    run_dots_symmetric<nk_e4m3_k>(env, "dots_symmetric_e4m3_cdna4", nk_dots_symmetric_e4m3_cdna4, backend);
    run_angulars_packed<nk_e4m3_k>(env, "angulars_packed_e4m3_cdna4", nk_dots_pack_size_e4m3_cdna4,
                                   nk_dots_pack_e4m3_cdna4, nk_angulars_packed_e4m3_cdna4, backend);
    run_angulars_symmetric<nk_e4m3_k>(env, "angulars_symmetric_e4m3_cdna4", nk_angulars_symmetric_e4m3_cdna4, backend);
    run_euclideans_packed<nk_e4m3_k>(env, "euclideans_packed_e4m3_cdna4", nk_dots_pack_size_e4m3_cdna4,
                                     nk_dots_pack_e4m3_cdna4, nk_euclideans_packed_e4m3_cdna4, backend);
    run_euclideans_symmetric<nk_e4m3_k>(env, "euclideans_symmetric_e4m3_cdna4", nk_euclideans_symmetric_e4m3_cdna4,
                                        backend);
    run_dots_packed<nk_e5m2_k>(env, "dots_packed_e5m2_cdna4", nk_dots_pack_size_e5m2_cdna4, nk_dots_pack_e5m2_cdna4,
                               nk_dots_packed_e5m2_cdna4, backend);
    run_dots_symmetric<nk_e5m2_k>(env, "dots_symmetric_e5m2_cdna4", nk_dots_symmetric_e5m2_cdna4, backend);
    run_angulars_packed<nk_e5m2_k>(env, "angulars_packed_e5m2_cdna4", nk_dots_pack_size_e5m2_cdna4,
                                   nk_dots_pack_e5m2_cdna4, nk_angulars_packed_e5m2_cdna4, backend);
    run_angulars_symmetric<nk_e5m2_k>(env, "angulars_symmetric_e5m2_cdna4", nk_angulars_symmetric_e5m2_cdna4, backend);
    run_euclideans_packed<nk_e5m2_k>(env, "euclideans_packed_e5m2_cdna4", nk_dots_pack_size_e5m2_cdna4,
                                     nk_dots_pack_e5m2_cdna4, nk_euclideans_packed_e5m2_cdna4, backend);
    run_euclideans_symmetric<nk_e5m2_k>(env, "euclideans_symmetric_e5m2_cdna4", nk_euclideans_symmetric_e5m2_cdna4,
                                        backend);
    run_dots_packed<nk_e3m2_k>(env, "dots_packed_e3m2_cdna4", nk_dots_pack_size_e3m2_cdna4, nk_dots_pack_e3m2_cdna4,
                               nk_dots_packed_e3m2_cdna4, backend);
    run_dots_symmetric<nk_e3m2_k>(env, "dots_symmetric_e3m2_cdna4", nk_dots_symmetric_e3m2_cdna4, backend);
    run_angulars_packed<nk_e3m2_k>(env, "angulars_packed_e3m2_cdna4", nk_dots_pack_size_e3m2_cdna4,
                                   nk_dots_pack_e3m2_cdna4, nk_angulars_packed_e3m2_cdna4, backend);
    run_angulars_symmetric<nk_e3m2_k>(env, "angulars_symmetric_e3m2_cdna4", nk_angulars_symmetric_e3m2_cdna4, backend);
    run_euclideans_packed<nk_e3m2_k>(env, "euclideans_packed_e3m2_cdna4", nk_dots_pack_size_e3m2_cdna4,
                                     nk_dots_pack_e3m2_cdna4, nk_euclideans_packed_e3m2_cdna4, backend);
    run_euclideans_symmetric<nk_e3m2_k>(env, "euclideans_symmetric_e3m2_cdna4", nk_euclideans_symmetric_e3m2_cdna4,
                                        backend);
    run_dots_packed<nk_e2m3_k>(env, "dots_packed_e2m3_cdna4", nk_dots_pack_size_e2m3_cdna4, nk_dots_pack_e2m3_cdna4,
                               nk_dots_packed_e2m3_cdna4, backend);
    run_dots_symmetric<nk_e2m3_k>(env, "dots_symmetric_e2m3_cdna4", nk_dots_symmetric_e2m3_cdna4, backend);
    run_angulars_packed<nk_e2m3_k>(env, "angulars_packed_e2m3_cdna4", nk_dots_pack_size_e2m3_cdna4,
                                   nk_dots_pack_e2m3_cdna4, nk_angulars_packed_e2m3_cdna4, backend);
    run_angulars_symmetric<nk_e2m3_k>(env, "angulars_symmetric_e2m3_cdna4", nk_angulars_symmetric_e2m3_cdna4, backend);
    run_euclideans_packed<nk_e2m3_k>(env, "euclideans_packed_e2m3_cdna4", nk_dots_pack_size_e2m3_cdna4,
                                     nk_dots_pack_e2m3_cdna4, nk_euclideans_packed_e2m3_cdna4, backend);
    run_euclideans_symmetric<nk_e2m3_k>(env, "euclideans_symmetric_e2m3_cdna4", nk_euclideans_symmetric_e2m3_cdna4,
                                        backend);
    run_dots_packed<nk_e2m1_k>(env, "dots_packed_e2m1_cdna4", nk_dots_pack_size_e2m1_cdna4, nk_dots_pack_e2m1_cdna4,
                               nk_dots_packed_e2m1_cdna4, backend);
    run_dots_symmetric<nk_e2m1_k>(env, "dots_symmetric_e2m1_cdna4", nk_dots_symmetric_e2m1_cdna4, backend);
    run_angulars_packed<nk_e2m1_k>(env, "angulars_packed_e2m1_cdna4", nk_dots_pack_size_e2m1_cdna4,
                                   nk_dots_pack_e2m1_cdna4, nk_angulars_packed_e2m1_cdna4, backend);
    run_angulars_symmetric<nk_e2m1_k>(env, "angulars_symmetric_e2m1_cdna4", nk_angulars_symmetric_e2m1_cdna4, backend);
    run_euclideans_packed<nk_e2m1_k>(env, "euclideans_packed_e2m1_cdna4", nk_dots_pack_size_e2m1_cdna4,
                                     nk_dots_pack_e2m1_cdna4, nk_euclideans_packed_e2m1_cdna4, backend);
    run_euclideans_symmetric<nk_e2m1_k>(env, "euclideans_symmetric_e2m1_cdna4", nk_euclideans_symmetric_e2m1_cdna4,
                                        backend);
    run_dots_packed<nk_i8_k>(env, "dots_packed_i8_cdna4", nk_dots_pack_size_i8_cdna4, nk_dots_pack_i8_cdna4,
                             nk_dots_packed_i8_cdna4, backend);
    run_dots_symmetric<nk_i8_k>(env, "dots_symmetric_i8_cdna4", nk_dots_symmetric_i8_cdna4, backend);
    run_angulars_packed<nk_i8_k>(env, "angulars_packed_i8_cdna4", nk_dots_pack_size_i8_cdna4, nk_dots_pack_i8_cdna4,
                                 nk_angulars_packed_i8_cdna4, backend);
    run_angulars_symmetric<nk_i8_k>(env, "angulars_symmetric_i8_cdna4", nk_angulars_symmetric_i8_cdna4, backend);
    run_euclideans_packed<nk_i8_k>(env, "euclideans_packed_i8_cdna4", nk_dots_pack_size_i8_cdna4, nk_dots_pack_i8_cdna4,
                                   nk_euclideans_packed_i8_cdna4, backend);
    run_euclideans_symmetric<nk_i8_k>(env, "euclideans_symmetric_i8_cdna4", nk_euclideans_symmetric_i8_cdna4, backend);
    run_dots_packed<nk_u8_k>(env, "dots_packed_u8_cdna4", nk_dots_pack_size_u8_cdna4, nk_dots_pack_u8_cdna4,
                             nk_dots_packed_u8_cdna4, backend);
    run_dots_symmetric<nk_u8_k>(env, "dots_symmetric_u8_cdna4", nk_dots_symmetric_u8_cdna4, backend);
    run_angulars_packed<nk_u8_k>(env, "angulars_packed_u8_cdna4", nk_dots_pack_size_u8_cdna4, nk_dots_pack_u8_cdna4,
                                 nk_angulars_packed_u8_cdna4, backend);
    run_angulars_symmetric<nk_u8_k>(env, "angulars_symmetric_u8_cdna4", nk_angulars_symmetric_u8_cdna4, backend);
    run_euclideans_packed<nk_u8_k>(env, "euclideans_packed_u8_cdna4", nk_dots_pack_size_u8_cdna4, nk_dots_pack_u8_cdna4,
                                   nk_euclideans_packed_u8_cdna4, backend);
    run_euclideans_symmetric<nk_u8_k>(env, "euclideans_symmetric_u8_cdna4", nk_euclideans_symmetric_u8_cdna4, backend);
    run_dots_packed<nk_i4_k>(env, "dots_packed_i4_cdna4", nk_dots_pack_size_i4_cdna4, nk_dots_pack_i4_cdna4,
                             nk_dots_packed_i4_cdna4, backend);
    run_dots_symmetric<nk_i4_k>(env, "dots_symmetric_i4_cdna4", nk_dots_symmetric_i4_cdna4, backend);
    run_angulars_packed<nk_i4_k>(env, "angulars_packed_i4_cdna4", nk_dots_pack_size_i4_cdna4, nk_dots_pack_i4_cdna4,
                                 nk_angulars_packed_i4_cdna4, backend);
    run_angulars_symmetric<nk_i4_k>(env, "angulars_symmetric_i4_cdna4", nk_angulars_symmetric_i4_cdna4, backend);
    run_euclideans_packed<nk_i4_k>(env, "euclideans_packed_i4_cdna4", nk_dots_pack_size_i4_cdna4, nk_dots_pack_i4_cdna4,
                                   nk_euclideans_packed_i4_cdna4, backend);
    run_euclideans_symmetric<nk_i4_k>(env, "euclideans_symmetric_i4_cdna4", nk_euclideans_symmetric_i4_cdna4, backend);
    run_dots_packed<nk_u4_k>(env, "dots_packed_u4_cdna4", nk_dots_pack_size_u4_cdna4, nk_dots_pack_u4_cdna4,
                             nk_dots_packed_u4_cdna4, backend);
    run_dots_symmetric<nk_u4_k>(env, "dots_symmetric_u4_cdna4", nk_dots_symmetric_u4_cdna4, backend);
    run_angulars_packed<nk_u4_k>(env, "angulars_packed_u4_cdna4", nk_dots_pack_size_u4_cdna4, nk_dots_pack_u4_cdna4,
                                 nk_angulars_packed_u4_cdna4, backend);
    run_angulars_symmetric<nk_u4_k>(env, "angulars_symmetric_u4_cdna4", nk_angulars_symmetric_u4_cdna4, backend);
    run_euclideans_packed<nk_u4_k>(env, "euclideans_packed_u4_cdna4", nk_dots_pack_size_u4_cdna4, nk_dots_pack_u4_cdna4,
                                   nk_euclideans_packed_u4_cdna4, backend);
    run_euclideans_symmetric<nk_u4_k>(env, "euclideans_symmetric_u4_cdna4", nk_euclideans_symmetric_u4_cdna4, backend);
    run_attention_bidirectional<nk_bf16_k>(env, "attention_bidirectional_packed_bf16_cdna4",
                                           nk_attention_pack_size_bf16_cdna4, nk_attention_pack_bf16_cdna4,
                                           nk_attention_bidirectional_packed_bf16_cdna4, backend);
    run_attention_causal<nk_bf16_k>(env, "attention_causal_packed_bf16_cdna4", nk_attention_pack_size_bf16_cdna4,
                                    nk_attention_pack_bf16_cdna4, nk_attention_causal_packed_bf16_cdna4, backend);
    run_attention_bidirectional<nk_e4m3_k>(env, "attention_bidirectional_packed_e4m3_cdna4",
                                           nk_attention_pack_size_e4m3_cdna4, nk_attention_pack_e4m3_cdna4,
                                           nk_attention_bidirectional_packed_e4m3_cdna4, backend);
    run_attention_causal<nk_e4m3_k>(env, "attention_causal_packed_e4m3_cdna4", nk_attention_pack_size_e4m3_cdna4,
                                    nk_attention_pack_e4m3_cdna4, nk_attention_causal_packed_e4m3_cdna4, backend);
    run_attention_bidirectional<nk_i8_k>(env, "attention_bidirectional_packed_i8_cdna4",
                                         nk_attention_pack_size_i8_cdna4, nk_attention_pack_i8_cdna4,
                                         nk_attention_bidirectional_packed_i8_cdna4, backend);
    run_attention_causal<nk_i8_k>(env, "attention_causal_packed_i8_cdna4", nk_attention_pack_size_i8_cdna4,
                                  nk_attention_pack_i8_cdna4, nk_attention_causal_packed_i8_cdna4, backend);
#endif // NUMKONG_TARGET_CDNA4
}
void bench_cross_cdna5([[maybe_unused]] environment_t const &env, [[maybe_unused]] rocm_backend_t const &backend,
                       [[maybe_unused]] nk_capability_t enabled) {
#if NUMKONG_TARGET_CDNA5
    if (!(enabled & nk_cap_cdna5_k)) return;
    run_dots_packed<nk_bf16_k>(env, "dots_packed_bf16_cdna5", nk_dots_pack_size_bf16_cdna5, nk_dots_pack_bf16_cdna5,
                               nk_dots_packed_bf16_cdna5, backend);
    run_dots_symmetric<nk_bf16_k>(env, "dots_symmetric_bf16_cdna5", nk_dots_symmetric_bf16_cdna5, backend);
    run_angulars_packed<nk_bf16_k>(env, "angulars_packed_bf16_cdna5", nk_dots_pack_size_bf16_cdna5,
                                   nk_dots_pack_bf16_cdna5, nk_angulars_packed_bf16_cdna5, backend);
    run_angulars_symmetric<nk_bf16_k>(env, "angulars_symmetric_bf16_cdna5", nk_angulars_symmetric_bf16_cdna5, backend);
    run_euclideans_packed<nk_bf16_k>(env, "euclideans_packed_bf16_cdna5", nk_dots_pack_size_bf16_cdna5,
                                     nk_dots_pack_bf16_cdna5, nk_euclideans_packed_bf16_cdna5, backend);
    run_euclideans_symmetric<nk_bf16_k>(env, "euclideans_symmetric_bf16_cdna5", nk_euclideans_symmetric_bf16_cdna5,
                                        backend);
    run_dots_packed<nk_f16_k>(env, "dots_packed_f16_cdna5", nk_dots_pack_size_f16_cdna5, nk_dots_pack_f16_cdna5,
                              nk_dots_packed_f16_cdna5, backend);
    run_dots_symmetric<nk_f16_k>(env, "dots_symmetric_f16_cdna5", nk_dots_symmetric_f16_cdna5, backend);
    run_angulars_packed<nk_f16_k>(env, "angulars_packed_f16_cdna5", nk_dots_pack_size_f16_cdna5, nk_dots_pack_f16_cdna5,
                                  nk_angulars_packed_f16_cdna5, backend);
    run_angulars_symmetric<nk_f16_k>(env, "angulars_symmetric_f16_cdna5", nk_angulars_symmetric_f16_cdna5, backend);
    run_euclideans_packed<nk_f16_k>(env, "euclideans_packed_f16_cdna5", nk_dots_pack_size_f16_cdna5,
                                    nk_dots_pack_f16_cdna5, nk_euclideans_packed_f16_cdna5, backend);
    run_euclideans_symmetric<nk_f16_k>(env, "euclideans_symmetric_f16_cdna5", nk_euclideans_symmetric_f16_cdna5,
                                       backend);
    run_dots_packed<nk_e4m3_k>(env, "dots_packed_e4m3_cdna5", nk_dots_pack_size_e4m3_cdna5, nk_dots_pack_e4m3_cdna5,
                               nk_dots_packed_e4m3_cdna5, backend);
    run_dots_symmetric<nk_e4m3_k>(env, "dots_symmetric_e4m3_cdna5", nk_dots_symmetric_e4m3_cdna5, backend);
    run_angulars_packed<nk_e4m3_k>(env, "angulars_packed_e4m3_cdna5", nk_dots_pack_size_e4m3_cdna5,
                                   nk_dots_pack_e4m3_cdna5, nk_angulars_packed_e4m3_cdna5, backend);
    run_angulars_symmetric<nk_e4m3_k>(env, "angulars_symmetric_e4m3_cdna5", nk_angulars_symmetric_e4m3_cdna5, backend);
    run_euclideans_packed<nk_e4m3_k>(env, "euclideans_packed_e4m3_cdna5", nk_dots_pack_size_e4m3_cdna5,
                                     nk_dots_pack_e4m3_cdna5, nk_euclideans_packed_e4m3_cdna5, backend);
    run_euclideans_symmetric<nk_e4m3_k>(env, "euclideans_symmetric_e4m3_cdna5", nk_euclideans_symmetric_e4m3_cdna5,
                                        backend);
    run_dots_packed<nk_e5m2_k>(env, "dots_packed_e5m2_cdna5", nk_dots_pack_size_e5m2_cdna5, nk_dots_pack_e5m2_cdna5,
                               nk_dots_packed_e5m2_cdna5, backend);
    run_dots_symmetric<nk_e5m2_k>(env, "dots_symmetric_e5m2_cdna5", nk_dots_symmetric_e5m2_cdna5, backend);
    run_angulars_packed<nk_e5m2_k>(env, "angulars_packed_e5m2_cdna5", nk_dots_pack_size_e5m2_cdna5,
                                   nk_dots_pack_e5m2_cdna5, nk_angulars_packed_e5m2_cdna5, backend);
    run_angulars_symmetric<nk_e5m2_k>(env, "angulars_symmetric_e5m2_cdna5", nk_angulars_symmetric_e5m2_cdna5, backend);
    run_euclideans_packed<nk_e5m2_k>(env, "euclideans_packed_e5m2_cdna5", nk_dots_pack_size_e5m2_cdna5,
                                     nk_dots_pack_e5m2_cdna5, nk_euclideans_packed_e5m2_cdna5, backend);
    run_euclideans_symmetric<nk_e5m2_k>(env, "euclideans_symmetric_e5m2_cdna5", nk_euclideans_symmetric_e5m2_cdna5,
                                        backend);
    run_dots_packed<nk_e3m2_k>(env, "dots_packed_e3m2_cdna5", nk_dots_pack_size_e3m2_cdna5, nk_dots_pack_e3m2_cdna5,
                               nk_dots_packed_e3m2_cdna5, backend);
    run_dots_symmetric<nk_e3m2_k>(env, "dots_symmetric_e3m2_cdna5", nk_dots_symmetric_e3m2_cdna5, backend);
    run_angulars_packed<nk_e3m2_k>(env, "angulars_packed_e3m2_cdna5", nk_dots_pack_size_e3m2_cdna5,
                                   nk_dots_pack_e3m2_cdna5, nk_angulars_packed_e3m2_cdna5, backend);
    run_angulars_symmetric<nk_e3m2_k>(env, "angulars_symmetric_e3m2_cdna5", nk_angulars_symmetric_e3m2_cdna5, backend);
    run_euclideans_packed<nk_e3m2_k>(env, "euclideans_packed_e3m2_cdna5", nk_dots_pack_size_e3m2_cdna5,
                                     nk_dots_pack_e3m2_cdna5, nk_euclideans_packed_e3m2_cdna5, backend);
    run_euclideans_symmetric<nk_e3m2_k>(env, "euclideans_symmetric_e3m2_cdna5", nk_euclideans_symmetric_e3m2_cdna5,
                                        backend);
    run_dots_packed<nk_e2m3_k>(env, "dots_packed_e2m3_cdna5", nk_dots_pack_size_e2m3_cdna5, nk_dots_pack_e2m3_cdna5,
                               nk_dots_packed_e2m3_cdna5, backend);
    run_dots_symmetric<nk_e2m3_k>(env, "dots_symmetric_e2m3_cdna5", nk_dots_symmetric_e2m3_cdna5, backend);
    run_angulars_packed<nk_e2m3_k>(env, "angulars_packed_e2m3_cdna5", nk_dots_pack_size_e2m3_cdna5,
                                   nk_dots_pack_e2m3_cdna5, nk_angulars_packed_e2m3_cdna5, backend);
    run_angulars_symmetric<nk_e2m3_k>(env, "angulars_symmetric_e2m3_cdna5", nk_angulars_symmetric_e2m3_cdna5, backend);
    run_euclideans_packed<nk_e2m3_k>(env, "euclideans_packed_e2m3_cdna5", nk_dots_pack_size_e2m3_cdna5,
                                     nk_dots_pack_e2m3_cdna5, nk_euclideans_packed_e2m3_cdna5, backend);
    run_euclideans_symmetric<nk_e2m3_k>(env, "euclideans_symmetric_e2m3_cdna5", nk_euclideans_symmetric_e2m3_cdna5,
                                        backend);
    run_dots_packed<nk_e2m1_k>(env, "dots_packed_e2m1_cdna5", nk_dots_pack_size_e2m1_cdna5, nk_dots_pack_e2m1_cdna5,
                               nk_dots_packed_e2m1_cdna5, backend);
    run_dots_symmetric<nk_e2m1_k>(env, "dots_symmetric_e2m1_cdna5", nk_dots_symmetric_e2m1_cdna5, backend);
    run_angulars_packed<nk_e2m1_k>(env, "angulars_packed_e2m1_cdna5", nk_dots_pack_size_e2m1_cdna5,
                                   nk_dots_pack_e2m1_cdna5, nk_angulars_packed_e2m1_cdna5, backend);
    run_angulars_symmetric<nk_e2m1_k>(env, "angulars_symmetric_e2m1_cdna5", nk_angulars_symmetric_e2m1_cdna5, backend);
    run_euclideans_packed<nk_e2m1_k>(env, "euclideans_packed_e2m1_cdna5", nk_dots_pack_size_e2m1_cdna5,
                                     nk_dots_pack_e2m1_cdna5, nk_euclideans_packed_e2m1_cdna5, backend);
    run_euclideans_symmetric<nk_e2m1_k>(env, "euclideans_symmetric_e2m1_cdna5", nk_euclideans_symmetric_e2m1_cdna5,
                                        backend);
    run_dots_packed<nk_i8_k>(env, "dots_packed_i8_cdna5", nk_dots_pack_size_i8_cdna5, nk_dots_pack_i8_cdna5,
                             nk_dots_packed_i8_cdna5, backend);
    run_dots_symmetric<nk_i8_k>(env, "dots_symmetric_i8_cdna5", nk_dots_symmetric_i8_cdna5, backend);
    run_angulars_packed<nk_i8_k>(env, "angulars_packed_i8_cdna5", nk_dots_pack_size_i8_cdna5, nk_dots_pack_i8_cdna5,
                                 nk_angulars_packed_i8_cdna5, backend);
    run_angulars_symmetric<nk_i8_k>(env, "angulars_symmetric_i8_cdna5", nk_angulars_symmetric_i8_cdna5, backend);
    run_euclideans_packed<nk_i8_k>(env, "euclideans_packed_i8_cdna5", nk_dots_pack_size_i8_cdna5, nk_dots_pack_i8_cdna5,
                                   nk_euclideans_packed_i8_cdna5, backend);
    run_euclideans_symmetric<nk_i8_k>(env, "euclideans_symmetric_i8_cdna5", nk_euclideans_symmetric_i8_cdna5, backend);
    run_dots_packed<nk_u8_k>(env, "dots_packed_u8_cdna5", nk_dots_pack_size_u8_cdna5, nk_dots_pack_u8_cdna5,
                             nk_dots_packed_u8_cdna5, backend);
    run_dots_symmetric<nk_u8_k>(env, "dots_symmetric_u8_cdna5", nk_dots_symmetric_u8_cdna5, backend);
    run_angulars_packed<nk_u8_k>(env, "angulars_packed_u8_cdna5", nk_dots_pack_size_u8_cdna5, nk_dots_pack_u8_cdna5,
                                 nk_angulars_packed_u8_cdna5, backend);
    run_angulars_symmetric<nk_u8_k>(env, "angulars_symmetric_u8_cdna5", nk_angulars_symmetric_u8_cdna5, backend);
    run_euclideans_packed<nk_u8_k>(env, "euclideans_packed_u8_cdna5", nk_dots_pack_size_u8_cdna5, nk_dots_pack_u8_cdna5,
                                   nk_euclideans_packed_u8_cdna5, backend);
    run_euclideans_symmetric<nk_u8_k>(env, "euclideans_symmetric_u8_cdna5", nk_euclideans_symmetric_u8_cdna5, backend);
    run_dots_packed<nk_i4_k>(env, "dots_packed_i4_cdna5", nk_dots_pack_size_i4_cdna5, nk_dots_pack_i4_cdna5,
                             nk_dots_packed_i4_cdna5, backend);
    run_dots_symmetric<nk_i4_k>(env, "dots_symmetric_i4_cdna5", nk_dots_symmetric_i4_cdna5, backend);
    run_angulars_packed<nk_i4_k>(env, "angulars_packed_i4_cdna5", nk_dots_pack_size_i4_cdna5, nk_dots_pack_i4_cdna5,
                                 nk_angulars_packed_i4_cdna5, backend);
    run_angulars_symmetric<nk_i4_k>(env, "angulars_symmetric_i4_cdna5", nk_angulars_symmetric_i4_cdna5, backend);
    run_euclideans_packed<nk_i4_k>(env, "euclideans_packed_i4_cdna5", nk_dots_pack_size_i4_cdna5, nk_dots_pack_i4_cdna5,
                                   nk_euclideans_packed_i4_cdna5, backend);
    run_euclideans_symmetric<nk_i4_k>(env, "euclideans_symmetric_i4_cdna5", nk_euclideans_symmetric_i4_cdna5, backend);
    run_dots_packed<nk_u4_k>(env, "dots_packed_u4_cdna5", nk_dots_pack_size_u4_cdna5, nk_dots_pack_u4_cdna5,
                             nk_dots_packed_u4_cdna5, backend);
    run_dots_symmetric<nk_u4_k>(env, "dots_symmetric_u4_cdna5", nk_dots_symmetric_u4_cdna5, backend);
    run_angulars_packed<nk_u4_k>(env, "angulars_packed_u4_cdna5", nk_dots_pack_size_u4_cdna5, nk_dots_pack_u4_cdna5,
                                 nk_angulars_packed_u4_cdna5, backend);
    run_angulars_symmetric<nk_u4_k>(env, "angulars_symmetric_u4_cdna5", nk_angulars_symmetric_u4_cdna5, backend);
    run_euclideans_packed<nk_u4_k>(env, "euclideans_packed_u4_cdna5", nk_dots_pack_size_u4_cdna5, nk_dots_pack_u4_cdna5,
                                   nk_euclideans_packed_u4_cdna5, backend);
    run_euclideans_symmetric<nk_u4_k>(env, "euclideans_symmetric_u4_cdna5", nk_euclideans_symmetric_u4_cdna5, backend);
    run_attention_bidirectional<nk_bf16_k>(env, "attention_bidirectional_packed_bf16_cdna5",
                                           nk_attention_pack_size_bf16_cdna5, nk_attention_pack_bf16_cdna5,
                                           nk_attention_bidirectional_packed_bf16_cdna5, backend);
    run_attention_causal<nk_bf16_k>(env, "attention_causal_packed_bf16_cdna5", nk_attention_pack_size_bf16_cdna5,
                                    nk_attention_pack_bf16_cdna5, nk_attention_causal_packed_bf16_cdna5, backend);
    run_attention_bidirectional<nk_e4m3_k>(env, "attention_bidirectional_packed_e4m3_cdna5",
                                           nk_attention_pack_size_e4m3_cdna5, nk_attention_pack_e4m3_cdna5,
                                           nk_attention_bidirectional_packed_e4m3_cdna5, backend);
    run_attention_causal<nk_e4m3_k>(env, "attention_causal_packed_e4m3_cdna5", nk_attention_pack_size_e4m3_cdna5,
                                    nk_attention_pack_e4m3_cdna5, nk_attention_causal_packed_e4m3_cdna5, backend);
    run_attention_bidirectional<nk_i8_k>(env, "attention_bidirectional_packed_i8_cdna5",
                                         nk_attention_pack_size_i8_cdna5, nk_attention_pack_i8_cdna5,
                                         nk_attention_bidirectional_packed_i8_cdna5, backend);
    run_attention_causal<nk_i8_k>(env, "attention_causal_packed_i8_cdna5", nk_attention_pack_size_i8_cdna5,
                                  nk_attention_pack_i8_cdna5, nk_attention_causal_packed_i8_cdna5, backend);
#endif // NUMKONG_TARGET_CDNA5
}
} // namespace ashvardanian::numkong::bench
#endif // NUMKONG_ARCH_ROCM_

namespace ashvardanian::numkong::bench {
nk::status_t bench_cross_rocm([[maybe_unused]] environment_t const &env,
                              [[maybe_unused]] device_backend_t const &runtime) {
#if NUMKONG_ARCH_ROCM_
    device_scope<hipGetDevice, hipSetDevice> const selected(static_cast<int>(runtime.device.ordinal()));
    if (selected.status != hipSuccess) return nk::status_t::device_code_mismatch_k;
    int l2_bytes = 0;
    if (hipDeviceGetAttribute(&l2_bytes, hipDeviceAttributeL2CacheSize, static_cast<int>(runtime.device.ordinal())) !=
        hipSuccess)
        return nk::status_t::device_code_mismatch_k;
    nk_capability_t const capabilities = runtime.capabilities;
    rocm_backend_t const backend {runtime, std::size_t(std::max(l2_bytes, 0))};
    bench_cross_rocm(env, backend, capabilities);
    bench_each_rocm(env, backend, capabilities);
    bench_cross_cdna4(env, backend, capabilities);
    bench_cross_cdna5(env, backend, capabilities);
    return nk::status_t::success_k;
#else  // !NUMKONG_ARCH_ROCM_
    return nk::status_t::missing_gpu_k;
#endif // NUMKONG_ARCH_ROCM_
}
} // namespace ashvardanian::numkong::bench
