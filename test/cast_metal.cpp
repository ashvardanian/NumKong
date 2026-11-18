/**
 *  @file test/cast_metal.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief Conversion tests on Metal.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include "harness.hpp"

#if NUMKONG_ARCH_METAL_
#include <cstring> // `std::memcpy`
#include <vector>  // `std::vector`

#include "cast.hpp"
#endif // NUMKONG_ARCH_METAL_

namespace ashvardanian::numkong::test {

#if NUMKONG_ARCH_METAL_

/** Every encoding of the float @p source_type_, 2^32 or 2^16 of them in 1M batches, into
 *  @p destination_dtype through @c nk_cast_metal and through @p kernel, comparing the low @p mask
 *  bits. NaN inputs are skipped, as the two may pick different NaN codes. */
template <typename source_type_>
static bool sweep_narrowing_metal_(char const *label, nk_dtype_t destination_dtype, unsigned mask, cast_t kernel) {
    using bits_t = typename source_type_::uint_t;
    constexpr std::uint64_t total = std::uint64_t(1) << (8 * sizeof(bits_t));
    constexpr std::size_t batch = static_cast<std::size_t>(nk::min_of<std::uint64_t>(total, 1u << 20));
    std::size_t const destination_bytes = nk_dtype_bits(destination_dtype) / NUMKONG_BITS_PER_BYTE;
    metal_backend_t backend = make_backend<metal_backend_t>();
    auto source = make_vector<u8_t>(backend, batch * sizeof(bits_t));
    auto metal = make_vector<u8_t>(backend, batch * destination_bytes);
    std::vector<unsigned char> numkong(batch * destination_bytes);
    auto const code = [&](unsigned char const *codes, std::size_t index) {
        unsigned value = 0;
        std::memcpy(&value, codes + index * destination_bytes, destination_bytes);
        return value & mask;
    };
    std::size_t mismatches = 0;
    for (std::uint64_t first = 0; first < total; first += batch) {
        bits_t *source_bits = reinterpret_cast<bits_t *>(source.raw_values_data());
        for (std::size_t i = 0; i != batch; ++i) source_bits[i] = static_cast<bits_t>(first + i);
        backend.call(nk_cast_metal, static_cast<void const *>(source_bits), source_type_::dtype(),
                     static_cast<void *>(metal.raw_values_data()), destination_dtype, nk_size_t(batch));
        if (backend.synchronize() != nk_success_k) return false;
        if (kernel(source_bits, source_type_::dtype(), numkong.data(), destination_dtype, batch, nullptr) !=
            nk_success_k)
            return false;
        unsigned char const *metal_codes = reinterpret_cast<unsigned char const *>(metal.raw_values_data());
        for (std::size_t i = 0; i != batch; ++i) {
            if (source_type_::from_bits(source_bits[i]).is_nan() || code(metal_codes, i) == code(numkong.data(), i))
                continue;
            if (mismatches++ < 4)
                fmt::println(stderr, "  [{}] input 0x{:x} → Metal 0x{:x} CPU 0x{:x}", label, source_bits[i],
                             code(metal_codes, i), code(numkong.data(), i));
        }
    }
    if (mismatches > 0) fmt::println(stderr, "  [{}] total mismatches: {}", label, mismatches);
    return mismatches == 0;
}

/** All @p codes codes of the float @p source_dtype into the float @p destination_type_ through
 *  @c nk_cast_metal and through @p kernel, any NaN matching any other. */
template <typename destination_type_>
static bool sweep_widening_metal_(char const *label, nk_dtype_t source_dtype, std::size_t codes, cast_t kernel) {
    using bits_t = typename destination_type_::uint_t;
    std::size_t const source_bytes = nk_dtype_bits(source_dtype) / NUMKONG_BITS_PER_BYTE;
    metal_backend_t backend = make_backend<metal_backend_t>();
    auto source = make_vector<u8_t>(backend, codes * source_bytes);
    auto metal = make_vector<u8_t>(backend, codes * sizeof(bits_t));
    std::vector<bits_t> numkong(codes);
    unsigned char *source_codes = reinterpret_cast<unsigned char *>(source.raw_values_data());
    for (std::size_t i = 0; i != codes; ++i) std::memcpy(source_codes + i * source_bytes, &i, source_bytes);
    backend.call(nk_cast_metal, static_cast<void const *>(source_codes), source_dtype,
                 static_cast<void *>(metal.raw_values_data()), destination_type_::dtype(), nk_size_t(codes));
    if (backend.synchronize() != nk_success_k) return false;
    if (kernel(source_codes, source_dtype, numkong.data(), destination_type_::dtype(), codes, nullptr) != nk_success_k)
        return false;
    bits_t const *metal_bits = reinterpret_cast<bits_t const *>(metal.raw_values_data());
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i != codes; ++i) {
        bool const both_nan = destination_type_::from_bits(metal_bits[i]).is_nan() &&
                              destination_type_::from_bits(numkong[i]).is_nan();
        if (both_nan || metal_bits[i] == numkong[i]) continue;
        if (mismatches++ < 4)
            fmt::println(stderr, "  [{}] input 0x{:x} → Metal 0x{:x} CPU 0x{:x}", label, i, metal_bits[i], numkong[i]);
    }
    if (mismatches > 0) fmt::println(stderr, "  [{}] total mismatches: {}", label, mismatches);
    return mismatches == 0;
}

/** Every F32 encoding into F16, BF16, E5M2, E4M3, E3M2 and E2M3 on Metal against @p kernel. */
static error_stats_t test_cast_to_small_metal(cast_t kernel) {
    error_stats_t stats(comparison_family_t::exact_k);
    stats.expect_true(sweep_narrowing_metal_<f32_t>("f32 → f16", nk_f16_k, 0xFFFF, kernel), "f32 → f16");
    stats.expect_true(sweep_narrowing_metal_<f32_t>("f32 → bf16", nk_bf16_k, 0xFFFF, kernel), "f32 → bf16");
    stats.expect_true(sweep_narrowing_metal_<f32_t>("f32 → e5m2", nk_e5m2_k, 0xFF, kernel), "f32 → e5m2");
    stats.expect_true(sweep_narrowing_metal_<f32_t>("f32 → e4m3", nk_e4m3_k, 0xFF, kernel), "f32 → e4m3");
    stats.expect_true(sweep_narrowing_metal_<f32_t>("f32 → e3m2", nk_e3m2_k, 0x3F, kernel), "f32 → e3m2");
    stats.expect_true(sweep_narrowing_metal_<f32_t>("f32 → e2m3", nk_e2m3_k, 0x3F, kernel), "f32 → e2m3");
    return stats;
}

/** Every F16, BF16, E5M2, E4M3, E3M2 and E2M3 code into F32 on Metal against @p kernel. */
static error_stats_t test_cast_from_small_metal(cast_t kernel) {
    error_stats_t stats(comparison_family_t::exact_k);
    stats.expect_true(sweep_widening_metal_<f32_t>("f16 → f32", nk_f16_k, 1u << 16, kernel), "f16 → f32");
    stats.expect_true(sweep_widening_metal_<f32_t>("bf16 → f32", nk_bf16_k, 1u << 16, kernel), "bf16 → f32");
    stats.expect_true(sweep_widening_metal_<f32_t>("e5m2 → f32", nk_e5m2_k, 256, kernel), "e5m2 → f32");
    stats.expect_true(sweep_widening_metal_<f32_t>("e4m3 → f32", nk_e4m3_k, 256, kernel), "e4m3 → f32");
    stats.expect_true(sweep_widening_metal_<f32_t>("e3m2 → f32", nk_e3m2_k, 64, kernel), "e3m2 → f32");
    stats.expect_true(sweep_widening_metal_<f32_t>("e2m3 → f32", nk_e2m3_k, 64, kernel), "e2m3 → f32");
    return stats;
}

void run_cast_metal_cases() {
#if NUMKONG_TARGET_METAL
    auto const metal = nk_cap_metal_k;
    run_kernel_case("test_cast_to_small_metal(nk_cast_serial)", test_cast_to_small_metal, metal, nk_cast_serial);
    run_kernel_case("test_cast_to_small_metal(cpu_best<nk_cast_best>)", test_cast_to_small_metal, metal,
                    cpu_best<nk_cast_best>);
    run_kernel_case("test_cast_from_small_metal(nk_cast_serial)", test_cast_from_small_metal, metal, nk_cast_serial);
    run_kernel_case("test_cast_from_small_metal(cpu_best<nk_cast_best>)", test_cast_from_small_metal, metal,
                    cpu_best<nk_cast_best>);
    run_kernel_case("test_cast_pairs<metal_backend_t>(nk_cast_metal)", test_cast_pairs<metal_backend_t>, metal,
                    nk_cast_metal);
    run_kernel_case("test_cast_block_scaled<metal_backend_t>(nk_cast_metal)", test_cast_block_scaled<metal_backend_t>,
                    metal, nk_cast_metal);
#endif // NUMKONG_TARGET_METAL
}

#else  // !NUMKONG_ARCH_METAL_
void run_cast_metal_cases() {}
#endif // NUMKONG_ARCH_METAL_

} // namespace ashvardanian::numkong::test
