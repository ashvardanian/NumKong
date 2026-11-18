/**
 *  @file test/cast_rocm.hip
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief Conversion tests on ROCm, the CPU's FP6 and FP8 ones against HIP intrinsics.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include "harness.hpp"

#if NUMKONG_ARCH_ROCM_
#include "cast.hpp"

#include <hip/hip_bf16.h> // `__hip_bfloat16_raw`, `__float2bfloat16`
#include <hip/hip_fp16.h> // `__half_raw`, `__half2float`
#include <hip/hip_fp6.h>  // `__hip_cvt_float_to_fp6`, `__hip_cvt_fp6_to_halfraw`
#include <hip/hip_fp8.h>  // `__hip_cvt_float_to_fp8`, `__hip_cvt_fp8_to_halfraw`
#endif                    // NUMKONG_ARCH_ROCM_

namespace ashvardanian::numkong::test {

#if NUMKONG_ARCH_ROCM_

/** HIP's OCP FP8 encoding, saturating E4M3 and overflowing E5M2 to infinity, as NumKong does. */
struct hip_to_fp8_t {
    __hip_fp8_interpretation_t interpretation;

    __device__ __hip_saturation_t saturation() const noexcept {
        return interpretation == __HIP_E4M3 ? __HIP_SATFINITE : __HIP_NOSAT;
    }
    __device__ void operator()(float value, unsigned char *code) const noexcept {
        *code = __hip_cvt_float_to_fp8(value, saturation(), interpretation);
    }
    __device__ void operator()(__half_raw value, unsigned char *code) const noexcept {
        *code = __hip_cvt_halfraw_to_fp8(value, saturation(), interpretation);
    }
    __device__ void operator()(__hip_bfloat16_raw value, unsigned char *code) const noexcept {
        *code = __hip_cvt_bfloat16raw_to_fp8(value, saturation(), interpretation);
    }
};

/** HIP's OCP FP6 encoding, rounding to nearest. */
struct hip_to_fp6_t {
    __hip_fp6_interpretation_t interpretation;

    __device__ void operator()(float value, unsigned char *code) const noexcept {
        *code = __hip_cvt_float_to_fp6(value, interpretation, hipRoundNearest);
    }
    __device__ void operator()(__half_raw value, unsigned char *code) const noexcept {
        *code = __hip_cvt_halfraw_to_fp6(value, interpretation, hipRoundNearest);
    }
    __device__ void operator()(__hip_bfloat16_raw value, unsigned char *code) const noexcept {
        *code = __hip_cvt_bfloat16raw_to_fp6(value, interpretation, hipRoundNearest);
    }
};

/** An F16 value as itself, exactly in F32, or rounded into BF16, into @p destination. */
__device__ inline void store_widened_(__half_raw value, __half_raw *destination) { *destination = value; }
__device__ inline void store_widened_(__half_raw value, float *destination) {
    *destination = __half2float(__half(value));
}
__device__ inline void store_widened_(__half_raw value, __hip_bfloat16_raw *destination) {
    *destination = __float2bfloat16(__half2float(__half(value)));
}

/** HIP's decoding of an OCP FP8 code into F16, as HIP decodes into nothing wider. */
struct hip_from_fp8_t {
    __hip_fp8_interpretation_t interpretation;

    template <typename destination_type_>
    __device__ void operator()(unsigned char code, destination_type_ *destination) const noexcept {
        store_widened_(__hip_cvt_fp8_to_halfraw(code, interpretation), destination);
    }
};

/** HIP's decoding of an OCP FP6 code into F16, as HIP decodes into nothing wider. */
struct hip_from_fp6_t {
    __hip_fp6_interpretation_t interpretation;

    template <typename destination_type_>
    __device__ void operator()(unsigned char code, destination_type_ *destination) const noexcept {
        store_widened_(__hip_cvt_fp6_to_halfraw(code, interpretation), destination);
    }
};

template <typename source_type_, typename destination_type_, typename convert_type_>
__global__ void convert_kernel_(source_type_ const *source, destination_type_ *destination, std::size_t count,
                                convert_type_ convert) {
    std::size_t const index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) convert(source[index], destination + index);
}

/** Converts @p count values of @p host_source through @p convert in unified memory into
 *  @p host_destination. */
template <typename source_type_, typename destination_type_, typename convert_type_>
static bool convert_on_rocm_(void const *host_source, void *host_destination, std::size_t count,
                             convert_type_ convert) {
    nk_allocator_t policy;
    if (nk_allocator_init_unified_rocm(&policy) != nk_success_k) return false;
    nk::allocator<std::byte> const allocator(policy);
    using bytes_t = nk::vector<std::byte, nk::allocator<std::byte>>;
    auto [device_source, source_status] = bytes_t::uninitialized(count * sizeof(source_type_), allocator);
    auto [device_destination, destination_status] = bytes_t::uninitialized(count * sizeof(destination_type_),
                                                                           allocator);
    if (nk::failed(source_status) || nk::failed(destination_status)) return false;
    std::memcpy(device_source.raw_values_data(), host_source, count * sizeof(source_type_));
    constexpr unsigned threads = 256;
    convert_kernel_<<<static_cast<unsigned>(nk::divide_round_up(count, threads)), threads>>>(
        reinterpret_cast<source_type_ const *>(device_source.raw_values_data()),
        reinterpret_cast<destination_type_ *>(device_destination.raw_values_data()), count, convert);
    if (hipGetLastError() != hipSuccess || nk_stream_synchronize_rocm(nullptr) != nk_success_k) return false;
    std::memcpy(host_destination, device_destination.raw_values_data(), count * sizeof(destination_type_));
    return true;
}

/** Every encoding of the float @p source_type_, 2^32 or 2^16 of them in 1M batches, into the FP8 or
 *  FP6 @p destination_dtype through @p convert, HIP's, and through @p kernel, comparing the low
 *  @p mask bits. NaN inputs are skipped, as HIP and NumKong pick different NaN codes. */
template <typename source_type_, typename hip_source_type_, typename convert_type_>
static bool sweep_narrowing_rocm_(char const *label, nk_dtype_t destination_dtype, unsigned mask, cast_t kernel,
                                  convert_type_ convert) {
    using bits_t = typename source_type_::uint_t;
    constexpr std::uint64_t total = std::uint64_t(1) << (8 * sizeof(bits_t));
    constexpr std::size_t batch = static_cast<std::size_t>(nk::min_of<std::uint64_t>(total, 1u << 20));
    std::vector<bits_t> source(batch);
    std::vector<unsigned char> hip(batch), numkong(batch);
    std::size_t mismatches = 0;
    for (std::uint64_t first = 0; first < total; first += batch) {
        for (std::size_t i = 0; i != batch; ++i) source[i] = static_cast<bits_t>(first + i);
        if (!convert_on_rocm_<hip_source_type_, unsigned char>(source.data(), hip.data(), batch, convert)) return false;
        if (kernel(source.data(), source_type_::dtype(), numkong.data(), destination_dtype, batch, nullptr) !=
            nk_success_k)
            return false;
        for (std::size_t i = 0; i != batch; ++i) {
            if (source_type_::from_bits(source[i]).is_nan() || (hip[i] & mask) == (numkong[i] & mask)) continue;
            if (mismatches++ < 4)
                fmt::println(stderr, "  [{}] input 0x{:x} → HIP 0x{:02x} NumKong 0x{:02x}", label, source[i], hip[i],
                             numkong[i]);
        }
    }
    if (mismatches > 0) fmt::println(stderr, "  [{}] total mismatches: {}", label, mismatches);
    return mismatches == 0;
}

/** All @p codes FP8 or FP6 codes of @p source_dtype into the float @p destination_type_ through
 *  @p convert, HIP's, and through @p kernel, any NaN matching any other. */
template <typename destination_type_, typename hip_destination_type_, typename convert_type_>
static bool sweep_widening_rocm_(char const *label, nk_dtype_t source_dtype, std::size_t codes, cast_t kernel,
                                 convert_type_ convert) {
    using bits_t = typename destination_type_::uint_t;
    std::vector<unsigned char> source(codes);
    std::vector<bits_t> hip(codes), numkong(codes);
    for (std::size_t i = 0; i != codes; ++i) source[i] = static_cast<unsigned char>(i);
    if (!convert_on_rocm_<unsigned char, hip_destination_type_>(source.data(), hip.data(), codes, convert))
        return false;
    if (kernel(source.data(), source_dtype, numkong.data(), destination_type_::dtype(), codes, nullptr) != nk_success_k)
        return false;
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i != codes; ++i) {
        bool const both_nan = destination_type_::from_bits(hip[i]).is_nan() &&
                              destination_type_::from_bits(numkong[i]).is_nan();
        if (both_nan || hip[i] == numkong[i]) continue;
        if (mismatches++ < 4) fmt::println(stderr, "  [{}] input 0x{:02x} mismatch", label, i);
    }
    if (mismatches > 0) fmt::println(stderr, "  [{}] total mismatches: {}", label, mismatches);
    return mismatches == 0;
}

/** Every FP8 and FP6 encoding of F32, F16 and BF16 values through @p kernel against HIP's. */
static error_stats_t test_cast_to_small_rocm(cast_t kernel) {
    error_stats_t stats(comparison_family_t::exact_k);
    for (hip_to_fp8_t const convert : {hip_to_fp8_t {__HIP_E4M3}, hip_to_fp8_t {__HIP_E5M2}}) {
        nk_dtype_t const dtype = convert.interpretation == __HIP_E4M3 ? nk_e4m3_k : nk_e5m2_k;
        stats.expect_true(sweep_narrowing_rocm_<f32_t, float>("fp32 → fp8", dtype, 0xFF, kernel, convert),
                          "fp32 → fp8 (exhaustive 2^32)");
        stats.expect_true(sweep_narrowing_rocm_<f16_t, __half_raw>("fp16 → fp8", dtype, 0xFF, kernel, convert),
                          "fp16 → fp8 (exhaustive 2^16)");
        stats.expect_true(sweep_narrowing_rocm_<bf16_t, __hip_bfloat16_raw>("bf16 → fp8", dtype, 0xFF, kernel, convert),
                          "bf16 → fp8 (exhaustive 2^16)");
    }
    for (hip_to_fp6_t const convert : {hip_to_fp6_t {__HIP_E2M3}, hip_to_fp6_t {__HIP_E3M2}}) {
        nk_dtype_t const dtype = convert.interpretation == __HIP_E2M3 ? nk_e2m3_k : nk_e3m2_k;
        stats.expect_true(sweep_narrowing_rocm_<f32_t, float>("fp32 → fp6", dtype, 0x3F, kernel, convert),
                          "fp32 → fp6 (exhaustive 2^32)");
        stats.expect_true(sweep_narrowing_rocm_<f16_t, __half_raw>("fp16 → fp6", dtype, 0x3F, kernel, convert),
                          "fp16 → fp6 (exhaustive 2^16)");
        stats.expect_true(sweep_narrowing_rocm_<bf16_t, __hip_bfloat16_raw>("bf16 → fp6", dtype, 0x3F, kernel, convert),
                          "bf16 → fp6 (exhaustive 2^16)");
    }
    return stats;
}

/** Every FP8 and FP6 decoding into F32, F16 and BF16 through @p kernel against HIP's. */
static error_stats_t test_cast_from_small_rocm(cast_t kernel) {
    error_stats_t stats(comparison_family_t::exact_k);
    for (hip_from_fp8_t const convert : {hip_from_fp8_t {__HIP_E4M3}, hip_from_fp8_t {__HIP_E5M2}}) {
        nk_dtype_t const dtype = convert.interpretation == __HIP_E4M3 ? nk_e4m3_k : nk_e5m2_k;
        stats.expect_true(sweep_widening_rocm_<f32_t, float>("fp8 → fp32", dtype, 256, kernel, convert),
                          "fp8 → fp32 (exhaustive 2^8)");
        stats.expect_true(sweep_widening_rocm_<f16_t, __half_raw>("fp8 → fp16", dtype, 256, kernel, convert),
                          "fp8 → fp16 (exhaustive 2^8)");
        stats.expect_true(sweep_widening_rocm_<bf16_t, __hip_bfloat16_raw>("fp8 → bf16", dtype, 256, kernel, convert),
                          "fp8 → bf16 (exhaustive 2^8)");
    }
    for (hip_from_fp6_t const convert : {hip_from_fp6_t {__HIP_E2M3}, hip_from_fp6_t {__HIP_E3M2}}) {
        nk_dtype_t const dtype = convert.interpretation == __HIP_E2M3 ? nk_e2m3_k : nk_e3m2_k;
        stats.expect_true(sweep_widening_rocm_<f32_t, float>("fp6 → fp32", dtype, 64, kernel, convert),
                          "fp6 → fp32 (exhaustive 2^6)");
        stats.expect_true(sweep_widening_rocm_<f16_t, __half_raw>("fp6 → fp16", dtype, 64, kernel, convert),
                          "fp6 → fp16 (exhaustive 2^6)");
        stats.expect_true(sweep_widening_rocm_<bf16_t, __hip_bfloat16_raw>("fp6 → bf16", dtype, 64, kernel, convert),
                          "fp6 → bf16 (exhaustive 2^6)");
    }
    return stats;
}

void run_cast_rocm_cases() {
    // CDNA4 converts OCP FP8 and FP6, where CDNA3 has only the FNUZ formats
    auto const cdna4 = nk_cap_cdna4_k;
    run_kernel_case("test_cast_to_small_rocm(nk_cast_serial)", test_cast_to_small_rocm, cdna4, nk_cast_serial);
    run_kernel_case("test_cast_to_small_rocm(cpu_best<nk_cast_best>)", test_cast_to_small_rocm, cdna4,
                    cpu_best<nk_cast_best>);
    run_kernel_case("test_cast_from_small_rocm(nk_cast_serial)", test_cast_from_small_rocm, cdna4, nk_cast_serial);
    run_kernel_case("test_cast_from_small_rocm(cpu_best<nk_cast_best>)", test_cast_from_small_rocm, cdna4,
                    cpu_best<nk_cast_best>);

    auto const rocm = nk_cap_rocm_k;
    run_kernel_case("test_cast_pairs<rocm_backend_t>(nk_cast_rocm)", test_cast_pairs<rocm_backend_t>, rocm,
                    nk_cast_rocm);
    run_kernel_case("test_cast_block_scaled<rocm_backend_t>(nk_cast_rocm)", test_cast_block_scaled<rocm_backend_t>,
                    rocm, nk_cast_rocm);
}

#else  // !NUMKONG_ARCH_ROCM_
void run_cast_rocm_cases() {}
#endif // NUMKONG_ARCH_ROCM_

} // namespace ashvardanian::numkong::test
