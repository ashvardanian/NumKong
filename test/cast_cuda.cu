/**
 *  @file test/cast_cuda.cu
 *  @author Ash Vardanian
 *  @date April 15, 2026
 *  @brief FP6 and FP8 conversions against CUDA intrinsics over every input encoding.
 */
#include "harness.hpp"

#if NUMKONG_ARCH_CUDA_
#include "cast.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp6.h>
#include <cuda_fp8.h>

#endif // NUMKONG_ARCH_CUDA_

namespace ashvardanian::numkong::test {

#if NUMKONG_ARCH_CUDA_

/** Batch fp32 → fp8 conversion via __nv_cvt_float_to_fp8. */
__global__ void fp32_to_fp8_kernel_(float const *source, unsigned char *destination, std::size_t count,
                                    __nv_fp8_interpretation_t interpretation, __nv_saturation_t saturate) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    destination[index] = __nv_cvt_float_to_fp8(source[index], saturate, interpretation);
}

/** Batch fp16 → fp8 conversion via __nv_cvt_halfraw_to_fp8. */
__global__ void fp16_to_fp8_kernel_(__half const *source, unsigned char *destination, std::size_t count,
                                    __nv_fp8_interpretation_t interpretation, __nv_saturation_t saturate) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    __half_raw raw;
    std::memcpy(&raw, &source[index], sizeof raw);
    destination[index] = __nv_cvt_halfraw_to_fp8(raw, saturate, interpretation);
}

/** Batch bf16 → fp8 conversion via __nv_cvt_bfloat16raw_to_fp8. */
__global__ void bf16_to_fp8_kernel_(__nv_bfloat16 const *source, unsigned char *destination, std::size_t count,
                                    __nv_fp8_interpretation_t interpretation, __nv_saturation_t saturate) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    __nv_bfloat16_raw raw;
    std::memcpy(&raw, &source[index], sizeof raw);
    destination[index] = __nv_cvt_bfloat16raw_to_fp8(raw, saturate, interpretation);
}

/** Batch fp8 → fp32 conversion, routed via halfraw since CUDA has no direct fp8 → fp32. */
__global__ void fp8_to_fp32_kernel_(unsigned char const *source, float *destination, std::size_t count,
                                    __nv_fp8_interpretation_t interpretation) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    __half_raw half_raw = __nv_cvt_fp8_to_halfraw(source[index], interpretation);
    __half half_value;
    std::memcpy(&half_value, &half_raw, sizeof half_value);
    destination[index] = __half2float(half_value);
}

/** Batch fp8 → fp16 conversion via __nv_cvt_fp8_to_halfraw. */
__global__ void fp8_to_fp16_kernel_(unsigned char const *source, __half *destination, std::size_t count,
                                    __nv_fp8_interpretation_t interpretation) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    __half_raw half_raw = __nv_cvt_fp8_to_halfraw(source[index], interpretation);
    std::memcpy(&destination[index], &half_raw, sizeof half_raw);
}

/** Batch fp8 → bf16 conversion, routed fp8 → halfraw → fp32 → bf16 as CUDA has no direct path. */
__global__ void fp8_to_bf16_kernel_(unsigned char const *source, __nv_bfloat16 *destination, std::size_t count,
                                    __nv_fp8_interpretation_t interpretation) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    __half_raw half_raw = __nv_cvt_fp8_to_halfraw(source[index], interpretation);
    __half half_value;
    std::memcpy(&half_value, &half_raw, sizeof half_value);
    destination[index] = __float2bfloat16(__half2float(half_value));
}

/** Batch fp32 → fp6 conversion via __nv_cvt_float_to_fp6. */
__global__ void fp32_to_fp6_kernel_(float const *source, unsigned char *destination, std::size_t count,
                                    __nv_fp6_interpretation_t interpretation) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    destination[index] = __nv_cvt_float_to_fp6(source[index], interpretation, cudaRoundNearest);
}

/** Batch fp16 → fp6 conversion via __nv_cvt_halfraw_to_fp6. */
__global__ void fp16_to_fp6_kernel_(__half const *source, unsigned char *destination, std::size_t count,
                                    __nv_fp6_interpretation_t interpretation) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    __half_raw raw;
    std::memcpy(&raw, &source[index], sizeof raw);
    destination[index] = __nv_cvt_halfraw_to_fp6(raw, interpretation, cudaRoundNearest);
}

/** Batch bf16 → fp6 conversion via __nv_cvt_bfloat16raw_to_fp6. */
__global__ void bf16_to_fp6_kernel_(__nv_bfloat16 const *source, unsigned char *destination, std::size_t count,
                                    __nv_fp6_interpretation_t interpretation) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    __nv_bfloat16_raw raw;
    std::memcpy(&raw, &source[index], sizeof raw);
    destination[index] = __nv_cvt_bfloat16raw_to_fp6(raw, interpretation, cudaRoundNearest);
}

/** Batch fp6 → fp32 conversion, routed via halfraw. */
__global__ void fp6_to_fp32_kernel_(unsigned char const *source, float *destination, std::size_t count,
                                    __nv_fp6_interpretation_t interpretation) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    __half_raw half_raw = __nv_cvt_fp6_to_halfraw(source[index], interpretation);
    __half half_value;
    std::memcpy(&half_value, &half_raw, sizeof half_value);
    destination[index] = __half2float(half_value);
}

/** Batch fp6 → fp16 conversion via __nv_cvt_fp6_to_halfraw. */
__global__ void fp6_to_fp16_kernel_(unsigned char const *source, __half *destination, std::size_t count,
                                    __nv_fp6_interpretation_t interpretation) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    __half_raw half_raw = __nv_cvt_fp6_to_halfraw(source[index], interpretation);
    std::memcpy(&destination[index], &half_raw, sizeof half_raw);
}

/** Batch fp6 → bf16 conversion, routed fp6 → halfraw → fp32 → bf16. */
__global__ void fp6_to_bf16_kernel_(unsigned char const *source, __nv_bfloat16 *destination, std::size_t count,
                                    __nv_fp6_interpretation_t interpretation) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    __half_raw half_raw = __nv_cvt_fp6_to_halfraw(source[index], interpretation);
    __half half_value;
    std::memcpy(&half_value, &half_raw, sizeof half_value);
    destination[index] = __float2bfloat16(__half2float(half_value));
}

/** True iff the 32-bit IEEE-754 pattern encodes any NaN. */
static bool bits_fp32_nan_(std::uint32_t bits) noexcept {
    return (bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0;
}

/** True iff the 16-bit IEEE-754 half pattern encodes any NaN. */
static bool bits_fp16_nan_(std::uint16_t bits) noexcept { return (bits & 0x7C00u) == 0x7C00u && (bits & 0x03FFu) != 0; }

/** True iff the 16-bit bfloat16 pattern encodes any NaN. */
static bool bits_bf16_nan_(std::uint16_t bits) noexcept { return (bits & 0x7F80u) == 0x7F80u && (bits & 0x007Fu) != 0; }

/** NaN-tolerant bit equality: any NaN encoding matches any other NaN of the same width, but the "is
 *  a NaN" vs "is a number" bit still matters. */
static bool f32_bits_equal_tolerant_nan_(float left, float right) {
    std::uint32_t left_bits, right_bits;
    std::memcpy(&left_bits, &left, sizeof left_bits);
    std::memcpy(&right_bits, &right, sizeof right_bits);
    if (bits_fp32_nan_(left_bits) && bits_fp32_nan_(right_bits)) return true;
    return left_bits == right_bits;
}

/** NaN-tolerant bit equality for 16-bit half. */
static bool f16_bits_equal_tolerant_nan_(__half left, __half right) {
    std::uint16_t left_bits, right_bits;
    std::memcpy(&left_bits, &left, sizeof left_bits);
    std::memcpy(&right_bits, &right, sizeof right_bits);
    if (bits_fp16_nan_(left_bits) && bits_fp16_nan_(right_bits)) return true;
    return left_bits == right_bits;
}

/** NaN-tolerant bit equality for 16-bit bfloat16. */
static bool bf16_bits_equal_tolerant_nan_(__nv_bfloat16 left, __nv_bfloat16 right) {
    std::uint16_t left_bits, right_bits;
    std::memcpy(&left_bits, &left, sizeof left_bits);
    std::memcpy(&right_bits, &right, sizeof right_bits);
    if (bits_bf16_nan_(left_bits) && bits_bf16_nan_(right_bits)) return true;
    return left_bits == right_bits;
}

/** Masked byte equality; fp6 cares only about the low 6 bits. */
static bool bytes_equal_masked_(unsigned char left, unsigned char right, unsigned mask) noexcept {
    return (left & mask) == (right & mask);
}

/** Print a single mismatch line to stderr for diagnostics. */
static void report_mismatch_(char const *label, std::uint64_t source_bits, unsigned char cuda_output,
                             unsigned char numkong_output, std::uint64_t index) {
    fmt::println(stderr, "  [{}] input #{} raw=0x{:x} → CUDA=0x{:02x} NumKong=0x{:02x}", label, index, source_bits,
                 cuda_output, numkong_output);
}

/** Batch size: 1M elements per launch keeps device buffers under 16 MB even for 16-byte element
 *  types. */
constexpr std::size_t batch_size_ = 1u << 20;

/** Upload a batch, launch the kernel, download the result. */
template <typename source_type_, typename destination_type_, typename kernel_type_>
static bool run_kernel_batch_(source_type_ const *host_source, destination_type_ *host_destination, std::size_t count,
                              kernel_type_ kernel) {
    nk_allocator_t policy;
    if (nk_allocator_init_unified_cuda(&policy) != nk_success_k) return false;
    nk::allocator<char> const allocator(policy);
    using bytes_t = nk::vector<char, nk::allocator<char>>;
    auto [device_source, source_status] = bytes_t::uninitialized(count * sizeof(source_type_), allocator);
    auto [device_destination, destination_status] = bytes_t::uninitialized(count * sizeof(destination_type_),
                                                                           allocator);
    if (nk::failed(source_status) || nk::failed(destination_status)) return false;
    std::memcpy(device_source.raw_values_data(), host_source, count * sizeof(source_type_));
    dim3 block(256);
    dim3 grid(static_cast<unsigned>(nk::divide_round_up(count, block.x)));
    kernel(reinterpret_cast<source_type_ const *>(device_source.raw_values_data()),
           reinterpret_cast<destination_type_ *>(device_destination.raw_values_data()), count, grid, block);
    nk_cuda_assert_(cudaGetLastError());
    if (nk_stream_synchronize_cuda(nullptr) != nk_success_k) return false;
    std::memcpy(host_destination, device_destination.raw_values_data(), count * sizeof(destination_type_));
    return true;
}

/**
 *  @brief Exhaustive fp32 → fp8 sweep (2^32 inputs) in 1M-element batches.
 *
 *  NumKong uses a single rounding mode, RTNE, and one overflow policy per target type: E4M3
 *  saturates with no Inf encoding, E5M2 produces ±Inf. The CUDA @c saturate dial is derived here
 *  from @p interpretation rather than plumbed through, exposing only NumKong's fixed mode.
 */
template <typename launcher_type_>
static bool sweep_fp32_to_fp8_(char const *label, nk_dtype_t destination_dtype,
                               __nv_fp8_interpretation_t interpretation, launcher_type_ launcher) {
    __nv_saturation_t const saturate = (interpretation == __NV_E4M3) ? __NV_SATFINITE : __NV_NOSAT;
    std::vector<float> host_source(batch_size_);
    std::vector<unsigned char> host_cuda(batch_size_);
    std::vector<unsigned char> host_numkong(batch_size_);
    std::size_t mismatches = 0;
    constexpr std::uint64_t total = 1ull << 32;
    for (std::uint64_t base = 0; base < total; base += batch_size_) {
        std::size_t this_batch = static_cast<std::size_t>(std::min<std::uint64_t>(batch_size_, total - base));
        for (std::size_t i = 0; i < this_batch; ++i) {
            std::uint32_t bits = static_cast<std::uint32_t>(base + i);
            std::memcpy(&host_source[i], &bits, sizeof(float));
        }
        auto kernel = [&](float const *device_source, unsigned char *device_destination, std::size_t n, dim3 grid,
                          dim3 block) {
            launcher(device_source, device_destination, n, interpretation, saturate, grid, block);
        };
        if (!run_kernel_batch_(host_source.data(), host_cuda.data(), this_batch, kernel)) return false;
        if (nk_cast_serial(host_source.data(), nk_f32_k, host_numkong.data(), destination_dtype, this_batch, nullptr) !=
            nk_success_k)
            return false;
        for (std::size_t i = 0; i < this_batch; ++i) {
            std::uint32_t source_bits;
            std::memcpy(&source_bits, &host_source[i], sizeof(float));
            if (bits_fp32_nan_(source_bits)) continue;
            if (!bytes_equal_masked_(host_cuda[i], host_numkong[i], 0xFF)) {
                if (mismatches < 4) report_mismatch_(label, source_bits, host_cuda[i], host_numkong[i], base + i);
                ++mismatches;
            }
        }
    }
    if (mismatches > 0) fmt::println(stderr, "  [{}] total mismatches: {}", label, mismatches);
    return mismatches == 0;
}

/** Exhaustive fp32 → fp6 sweep; only low 6 bits of each output byte matter. */
template <typename launcher_type_>
static bool sweep_fp32_to_fp6_(char const *label, nk_dtype_t destination_dtype,
                               __nv_fp6_interpretation_t interpretation, launcher_type_ launcher) {
    std::vector<float> host_source(batch_size_);
    std::vector<unsigned char> host_cuda(batch_size_);
    std::vector<unsigned char> host_numkong(batch_size_);
    std::size_t mismatches = 0;
    constexpr std::uint64_t total = 1ull << 32;
    for (std::uint64_t base = 0; base < total; base += batch_size_) {
        std::size_t this_batch = static_cast<std::size_t>(std::min<std::uint64_t>(batch_size_, total - base));
        for (std::size_t i = 0; i < this_batch; ++i) {
            std::uint32_t bits = static_cast<std::uint32_t>(base + i);
            std::memcpy(&host_source[i], &bits, sizeof(float));
        }
        auto kernel = [&](float const *device_source, unsigned char *device_destination, std::size_t n, dim3 grid,
                          dim3 block) { launcher(device_source, device_destination, n, interpretation, grid, block); };
        if (!run_kernel_batch_(host_source.data(), host_cuda.data(), this_batch, kernel)) return false;
        if (nk_cast_serial(host_source.data(), nk_f32_k, host_numkong.data(), destination_dtype, this_batch, nullptr) !=
            nk_success_k)
            return false;
        for (std::size_t i = 0; i < this_batch; ++i) {
            std::uint32_t source_bits;
            std::memcpy(&source_bits, &host_source[i], sizeof(float));
            if (bits_fp32_nan_(source_bits)) continue;
            if (!bytes_equal_masked_(host_cuda[i], host_numkong[i], 0x3F)) {
                if (mismatches < 4) report_mismatch_(label, source_bits, host_cuda[i], host_numkong[i], base + i);
                ++mismatches;
            }
        }
    }
    if (mismatches > 0) fmt::println(stderr, "  [{}] total mismatches: {}", label, mismatches);
    return mismatches == 0;
}

/** Exhaustive fp16-or-bf16 16-bit → fp{6,8} sweep over all 2^16 inputs. */
template <typename source_type_, typename launcher_type_>
static bool sweep_16bit_to_8bit_(char const *label, nk_dtype_t source_dtype, nk_dtype_t destination_dtype,
                                 bool (*is_nan_fn)(std::uint16_t), unsigned byte_mask, launcher_type_ launcher) {
    constexpr std::size_t count = 1u << 16;
    std::vector<source_type_> host_source(count);
    std::vector<unsigned char> host_cuda(count);
    std::vector<unsigned char> host_numkong(count);
    for (std::size_t i = 0; i < count; ++i) {
        std::uint16_t bits = static_cast<std::uint16_t>(i);
        std::memcpy(&host_source[i], &bits, sizeof(source_type_));
    }
    if (!run_kernel_batch_(host_source.data(), host_cuda.data(), count, launcher)) return false;
    if (nk_cast_serial(host_source.data(), source_dtype, host_numkong.data(), destination_dtype, count, nullptr) !=
        nk_success_k)
        return false;
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < count; ++i) {
        std::uint16_t source_bits;
        std::memcpy(&source_bits, &host_source[i], sizeof(source_type_));
        if (is_nan_fn(source_bits)) continue;
        if (!bytes_equal_masked_(host_cuda[i], host_numkong[i], byte_mask)) {
            if (mismatches < 4) report_mismatch_(label, source_bits, host_cuda[i], host_numkong[i], i);
            ++mismatches;
        }
    }
    if (mismatches > 0) fmt::println(stderr, "  [{}] total mismatches: {}", label, mismatches);
    return mismatches == 0;
}

/** Reverse-direction exhaustive sweep: 2^8 fp8 or 2^6 fp6 inputs, comparing the wider output via a
 *  caller-supplied bit-equality predicate. */
template <typename destination_type_, typename launcher_type_, typename equals_type_>
static bool sweep_small_to_wide_(char const *label, nk_dtype_t source_dtype, nk_dtype_t destination_dtype,
                                 std::size_t domain_size, launcher_type_ launcher, equals_type_ equals) {
    std::vector<unsigned char> host_source(domain_size);
    std::vector<destination_type_> host_cuda(domain_size);
    std::vector<destination_type_> host_numkong(domain_size);
    for (std::size_t i = 0; i < domain_size; ++i) host_source[i] = static_cast<unsigned char>(i);
    if (!run_kernel_batch_(host_source.data(), host_cuda.data(), domain_size, launcher)) return false;
    if (nk_cast_serial(host_source.data(), source_dtype, host_numkong.data(), destination_dtype, domain_size,
                       nullptr) != nk_success_k)
        return false;
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < domain_size; ++i) {
        if (!equals(host_cuda[i], host_numkong[i])) {
            if (mismatches < 4) fmt::println(stderr, "  [{}] input 0x{:02x} mismatch", label, i);
            ++mismatches;
        }
    }
    if (mismatches > 0) fmt::println(stderr, "  [{}] total mismatches: {}", label, mismatches);
    return mismatches == 0;
}

static void test_cast_cuda_conformance(error_stats_section_t &check) {
    check.section("CUDA reference checks", nk_cap_cuda_k);
    cuda_device_scope_t device(check.settings.device.ordinal());
    if (!cuda_check_(device.status, "select CUDA reference device", __FILE__, __LINE__)) {
        ++check.failure_count;
        return;
    }

    check("FP8 conformance: nk_cast vs __nv_cvt_* (bit-exact)", [&](settings_t const &) {
        error_stats_t stats(comparison_family_t::exact_k);
        stats.expect(
            sweep_fp32_to_fp8_( //
                "fp32 → e4m3", nk_e4m3_k, __NV_E4M3,
                [](float const *device_source, unsigned char *device_destination, std::size_t count,
                   __nv_fp8_interpretation_t interpretation, __nv_saturation_t saturate, dim3 grid, dim3 block) {
                    fp32_to_fp8_kernel_<<<grid, block>>>(device_source, device_destination, count, interpretation,
                                                         saturate);
                }),
            "fp32 → e4m3 (exhaustive 2^32)");
        stats.expect(
            sweep_fp32_to_fp8_( //
                "fp32 → e5m2", nk_e5m2_k, __NV_E5M2,
                [](float const *device_source, unsigned char *device_destination, std::size_t count,
                   __nv_fp8_interpretation_t interpretation, __nv_saturation_t saturate, dim3 grid, dim3 block) {
                    fp32_to_fp8_kernel_<<<grid, block>>>(device_source, device_destination, count, interpretation,
                                                         saturate);
                }),
            "fp32 → e5m2 (exhaustive 2^32)");
        stats.expect(sweep_16bit_to_8bit_<__half>( //
                         "fp16 → e4m3", nk_f16_k, nk_e4m3_k, bits_fp16_nan_, 0xFF,
                         [](__half const *device_source, unsigned char *device_destination, std::size_t count,
                            dim3 grid, dim3 block) {
                             fp16_to_fp8_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E4M3,
                                                                  __NV_SATFINITE);
                         }),
                     "fp16 → e4m3 (exhaustive 2^16)");
        stats.expect(sweep_16bit_to_8bit_<__half>( //
                         "fp16 → e5m2", nk_f16_k, nk_e5m2_k, bits_fp16_nan_, 0xFF,
                         [](__half const *device_source, unsigned char *device_destination, std::size_t count,
                            dim3 grid, dim3 block) {
                             fp16_to_fp8_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E5M2,
                                                                  __NV_NOSAT);
                         }),
                     "fp16 → e5m2 (exhaustive 2^16)");
        stats.expect(sweep_16bit_to_8bit_<__nv_bfloat16>( //
                         "bf16 → e4m3", nk_bf16_k, nk_e4m3_k, bits_bf16_nan_, 0xFF,
                         [](__nv_bfloat16 const *device_source, unsigned char *device_destination, std::size_t count,
                            dim3 grid, dim3 block) {
                             bf16_to_fp8_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E4M3,
                                                                  __NV_SATFINITE);
                         }),
                     "bf16 → e4m3 (exhaustive 2^16)");
        stats.expect(sweep_16bit_to_8bit_<__nv_bfloat16>( //
                         "bf16 → e5m2", nk_bf16_k, nk_e5m2_k, bits_bf16_nan_, 0xFF,
                         [](__nv_bfloat16 const *device_source, unsigned char *device_destination, std::size_t count,
                            dim3 grid, dim3 block) {
                             bf16_to_fp8_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E5M2,
                                                                  __NV_NOSAT);
                         }),
                     "bf16 → e5m2 (exhaustive 2^16)");

        return stats;
    });

    check("FP8 reverse: fp{32,16,bf16} ← e{4m3,5m2} (bit-exact)", [&](settings_t const &) {
        error_stats_t stats(comparison_family_t::exact_k);
        stats.expect(sweep_small_to_wide_<float>( //
                         "e4m3 → fp32", nk_e4m3_k, nk_f32_k, 256u,
                         [](unsigned char const *device_source, float *device_destination, std::size_t count, dim3 grid,
                            dim3 block) {
                             fp8_to_fp32_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E4M3);
                         },
                         f32_bits_equal_tolerant_nan_),
                     "e4m3 → fp32 (exhaustive 2^8)");
        stats.expect(sweep_small_to_wide_<float>( //
                         "e5m2 → fp32", nk_e5m2_k, nk_f32_k, 256u,
                         [](unsigned char const *device_source, float *device_destination, std::size_t count, dim3 grid,
                            dim3 block) {
                             fp8_to_fp32_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E5M2);
                         },
                         f32_bits_equal_tolerant_nan_),
                     "e5m2 → fp32 (exhaustive 2^8)");
        stats.expect(sweep_small_to_wide_<__half>( //
                         "e4m3 → fp16", nk_e4m3_k, nk_f16_k, 256u,
                         [](unsigned char const *device_source, __half *device_destination, std::size_t count,
                            dim3 grid, dim3 block) {
                             fp8_to_fp16_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E4M3);
                         },
                         f16_bits_equal_tolerant_nan_),
                     "e4m3 → fp16 (exhaustive 2^8)");
        stats.expect(sweep_small_to_wide_<__half>( //
                         "e5m2 → fp16", nk_e5m2_k, nk_f16_k, 256u,
                         [](unsigned char const *device_source, __half *device_destination, std::size_t count,
                            dim3 grid, dim3 block) {
                             fp8_to_fp16_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E5M2);
                         },
                         f16_bits_equal_tolerant_nan_),
                     "e5m2 → fp16 (exhaustive 2^8)");
        stats.expect(sweep_small_to_wide_<__nv_bfloat16>( //
                         "e4m3 → bf16", nk_e4m3_k, nk_bf16_k, 256u,
                         [](unsigned char const *device_source, __nv_bfloat16 *device_destination, std::size_t count,
                            dim3 grid, dim3 block) {
                             fp8_to_bf16_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E4M3);
                         },
                         bf16_bits_equal_tolerant_nan_),
                     "e4m3 → bf16 (exhaustive 2^8)");
        stats.expect(sweep_small_to_wide_<__nv_bfloat16>( //
                         "e5m2 → bf16", nk_e5m2_k, nk_bf16_k, 256u,
                         [](unsigned char const *device_source, __nv_bfloat16 *device_destination, std::size_t count,
                            dim3 grid, dim3 block) {
                             fp8_to_bf16_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E5M2);
                         },
                         bf16_bits_equal_tolerant_nan_),
                     "e5m2 → bf16 (exhaustive 2^8)");

        return stats;
    });

    check("FP6 conformance: nk_cast vs __nv_cvt_* (bit-exact)", [&](settings_t const &) {
        error_stats_t stats(comparison_family_t::exact_k);
        stats.expect(sweep_fp32_to_fp6_( //
                         "fp32 → e3m2", nk_e3m2_k, __NV_E3M2,
                         [](float const *device_source, unsigned char *device_destination, std::size_t count,
                            __nv_fp6_interpretation_t interpretation, dim3 grid, dim3 block) {
                             fp32_to_fp6_kernel_<<<grid, block>>>(device_source, device_destination, count,
                                                                  interpretation);
                         }),
                     "fp32 → e3m2 (exhaustive 2^32)");
        stats.expect(sweep_fp32_to_fp6_( //
                         "fp32 → e2m3", nk_e2m3_k, __NV_E2M3,
                         [](float const *device_source, unsigned char *device_destination, std::size_t count,
                            __nv_fp6_interpretation_t interpretation, dim3 grid, dim3 block) {
                             fp32_to_fp6_kernel_<<<grid, block>>>(device_source, device_destination, count,
                                                                  interpretation);
                         }),
                     "fp32 → e2m3 (exhaustive 2^32)");
        stats.expect(sweep_16bit_to_8bit_<__half>( //
                         "fp16 → e3m2", nk_f16_k, nk_e3m2_k, bits_fp16_nan_, 0x3F,
                         [](__half const *device_source, unsigned char *device_destination, std::size_t count,
                            dim3 grid, dim3 block) {
                             fp16_to_fp6_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E3M2);
                         }),
                     "fp16 → e3m2 (exhaustive 2^16)");
        stats.expect(sweep_16bit_to_8bit_<__half>( //
                         "fp16 → e2m3", nk_f16_k, nk_e2m3_k, bits_fp16_nan_, 0x3F,
                         [](__half const *device_source, unsigned char *device_destination, std::size_t count,
                            dim3 grid, dim3 block) {
                             fp16_to_fp6_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E2M3);
                         }),
                     "fp16 → e2m3 (exhaustive 2^16)");
        stats.expect(sweep_16bit_to_8bit_<__nv_bfloat16>( //
                         "bf16 → e3m2", nk_bf16_k, nk_e3m2_k, bits_bf16_nan_, 0x3F,
                         [](__nv_bfloat16 const *device_source, unsigned char *device_destination, std::size_t count,
                            dim3 grid, dim3 block) {
                             bf16_to_fp6_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E3M2);
                         }),
                     "bf16 → e3m2 (exhaustive 2^16)");
        stats.expect(sweep_16bit_to_8bit_<__nv_bfloat16>( //
                         "bf16 → e2m3", nk_bf16_k, nk_e2m3_k, bits_bf16_nan_, 0x3F,
                         [](__nv_bfloat16 const *device_source, unsigned char *device_destination, std::size_t count,
                            dim3 grid, dim3 block) {
                             bf16_to_fp6_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E2M3);
                         }),
                     "bf16 → e2m3 (exhaustive 2^16)");

        return stats;
    });

    check("FP6 reverse: fp{32,16,bf16} ← e{3m2,2m3} (bit-exact)", [&](settings_t const &) {
        error_stats_t stats(comparison_family_t::exact_k);
        stats.expect(sweep_small_to_wide_<float>( //
                         "e3m2 → fp32", nk_e3m2_k, nk_f32_k, 64u,
                         [](unsigned char const *device_source, float *device_destination, std::size_t count, dim3 grid,
                            dim3 block) {
                             fp6_to_fp32_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E3M2);
                         },
                         f32_bits_equal_tolerant_nan_),
                     "e3m2 → fp32 (exhaustive 2^6)");
        stats.expect(sweep_small_to_wide_<float>( //
                         "e2m3 → fp32", nk_e2m3_k, nk_f32_k, 64u,
                         [](unsigned char const *device_source, float *device_destination, std::size_t count, dim3 grid,
                            dim3 block) {
                             fp6_to_fp32_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E2M3);
                         },
                         f32_bits_equal_tolerant_nan_),
                     "e2m3 → fp32 (exhaustive 2^6)");
        stats.expect(sweep_small_to_wide_<__half>( //
                         "e3m2 → fp16", nk_e3m2_k, nk_f16_k, 64u,
                         [](unsigned char const *device_source, __half *device_destination, std::size_t count,
                            dim3 grid, dim3 block) {
                             fp6_to_fp16_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E3M2);
                         },
                         f16_bits_equal_tolerant_nan_),
                     "e3m2 → fp16 (exhaustive 2^6)");
        stats.expect(sweep_small_to_wide_<__half>( //
                         "e2m3 → fp16", nk_e2m3_k, nk_f16_k, 64u,
                         [](unsigned char const *device_source, __half *device_destination, std::size_t count,
                            dim3 grid, dim3 block) {
                             fp6_to_fp16_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E2M3);
                         },
                         f16_bits_equal_tolerant_nan_),
                     "e2m3 → fp16 (exhaustive 2^6)");
        stats.expect(sweep_small_to_wide_<__nv_bfloat16>( //
                         "e3m2 → bf16", nk_e3m2_k, nk_bf16_k, 64u,
                         [](unsigned char const *device_source, __nv_bfloat16 *device_destination, std::size_t count,
                            dim3 grid, dim3 block) {
                             fp6_to_bf16_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E3M2);
                         },
                         bf16_bits_equal_tolerant_nan_),
                     "e3m2 → bf16 (exhaustive 2^6)");
        stats.expect(sweep_small_to_wide_<__nv_bfloat16>( //
                         "e2m3 → bf16", nk_e2m3_k, nk_bf16_k, 64u,
                         [](unsigned char const *device_source, __nv_bfloat16 *device_destination, std::size_t count,
                            dim3 grid, dim3 block) {
                             fp6_to_bf16_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E2M3);
                         },
                         bf16_bits_equal_tolerant_nan_),
                     "e2m3 → bf16 (exhaustive 2^6)");

        return stats;
    });
}

void test_cast_cuda(error_stats_section_t &check) {
    test_cast_cuda_conformance(check);
    check.section("Conversions CUDA", nk_cap_cuda_k);
    check("cast_cuda", test_cast_pairs<cuda_backend_t>, nk_cast_cuda);
    check_block_scaled_casts<cuda_backend_t>(check, "cuda", nk_cast_cuda);
}

#else // !NUMKONG_ARCH_CUDA_
void test_cast_cuda(error_stats_section_t &) {}

#endif // NUMKONG_ARCH_CUDA_

} // namespace ashvardanian::numkong::test
