/**
 *  @file test/cast.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *  @brief Type cast tests.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include "harness.hpp"
#include "cast.hpp" // `test_cast`, `test_cast_pairs`, `test_cast_block_scaled`

namespace ashvardanian::numkong::test {

void run_cast_cases() {
    auto const serial = nk_cap_serial_k;

    run_kernel_case("test_cast<bf16_t, f32_t>(nk_cast_serial)", test_cast<bf16_t, f32_t>, serial, nk_cast_serial);
    run_kernel_case("test_cast<f32_t, bf16_t>(nk_cast_serial)", test_cast<f32_t, bf16_t>, serial, nk_cast_serial);
    run_kernel_case("test_cast<e4m3_t, f32_t>(nk_cast_serial)", test_cast<e4m3_t, f32_t>, serial, nk_cast_serial);
    run_kernel_case("test_cast<f32_t, e4m3_t>(nk_cast_serial)", test_cast<f32_t, e4m3_t>, serial, nk_cast_serial);
    run_kernel_case("test_cast<e5m2_t, f32_t>(nk_cast_serial)", test_cast<e5m2_t, f32_t>, serial, nk_cast_serial);
    run_kernel_case("test_cast<f32_t, e5m2_t>(nk_cast_serial)", test_cast<f32_t, e5m2_t>, serial, nk_cast_serial);
    run_kernel_case("test_cast<e2m1x2_t, f32_t>(nk_cast_serial)", test_cast<e2m1x2_t, f32_t>, serial, nk_cast_serial);
    run_kernel_case("test_cast<f32_t, e2m1x2_t>(nk_cast_serial)", test_cast<f32_t, e2m1x2_t>, serial, nk_cast_serial);
    run_kernel_case("test_cast<ue8m0_t, f32_t>(nk_cast_serial)", test_cast<ue8m0_t, f32_t>, serial, nk_cast_serial);
    run_kernel_case("test_cast<f32_t, ue8m0_t>(nk_cast_serial)", test_cast<f32_t, ue8m0_t>, serial, nk_cast_serial);
    run_kernel_case("test_cast<ue4m3_t, f32_t>(nk_cast_serial)", test_cast<ue4m3_t, f32_t>, serial, nk_cast_serial);
    run_kernel_case("test_cast<f32_t, ue4m3_t>(nk_cast_serial)", test_cast<f32_t, ue4m3_t>, serial, nk_cast_serial);
    run_kernel_case("test_cast<f16_t, f32_t>(nk_cast_serial)", test_cast<f16_t, f32_t>, serial, nk_cast_serial);
    run_kernel_case("test_cast<f32_t, f16_t>(nk_cast_serial)", test_cast<f32_t, f16_t>, serial, nk_cast_serial);
    run_kernel_case("test_cast<f32_t, f64_t>(nk_cast_serial)", test_cast<f32_t, f64_t>, serial, nk_cast_serial);
    run_kernel_case("test_cast<f64_t, f32_t>(nk_cast_serial)", test_cast<f64_t, f32_t>, serial, nk_cast_serial);
    run_kernel_case("test_cast<f64_t, i32_t>(nk_cast_serial)", test_cast<f64_t, i32_t>, serial, nk_cast_serial);
    run_kernel_case("test_cast<i16_t, i64_t>(nk_cast_serial)", test_cast<i16_t, i64_t>, serial, nk_cast_serial);
    run_kernel_case("test_cast<i32_t, f64_t>(nk_cast_serial)", test_cast<i32_t, f64_t>, serial, nk_cast_serial);
    run_kernel_case("test_cast<i32_t, i8_t>(nk_cast_serial)", test_cast<i32_t, i8_t>, serial, nk_cast_serial);
    run_kernel_case("test_cast<i8_t, f64_t>(nk_cast_serial)", test_cast<i8_t, f64_t>, serial, nk_cast_serial);
    run_kernel_case("test_cast<i8_t, i32_t>(nk_cast_serial)", test_cast<i8_t, i32_t>, serial, nk_cast_serial);
    run_kernel_case("test_cast<u8_t, f32_t>(nk_cast_serial)", test_cast<u8_t, f32_t>, serial, nk_cast_serial);
    run_kernel_case("test_cast<f32_t, i4x2_t>(nk_cast_serial)", test_cast<f32_t, i4x2_t>, serial, nk_cast_serial);
    run_kernel_case("test_cast<f32_t, u4x2_t>(nk_cast_serial)", test_cast<f32_t, u4x2_t>, serial, nk_cast_serial);
    run_kernel_case("test_cast_pairs<>(nk_cast_serial)", test_cast_pairs<>, serial, nk_cast_serial);
    run_kernel_case("test_cast_block_scaled<>(nk_cast_serial)", test_cast_block_scaled<>, serial, nk_cast_serial);

#if !NUMKONG_HEADER_ONLY
    run_kernel_case("test_cast_pairs<>(cpu_best<nk_cast_best>)", test_cast_pairs<>, serial, cpu_best<nk_cast_best>);
    run_kernel_case("test_cast<f32_t, f16_t>(cpu_best<nk_cast_best>)", test_cast<f32_t, f16_t>, serial,
                    cpu_best<nk_cast_best>);
    run_kernel_case("test_cast<f16_t, f32_t>(cpu_best<nk_cast_best>)", test_cast<f16_t, f32_t>, serial,
                    cpu_best<nk_cast_best>);
    run_kernel_case("test_cast<f32_t, bf16_t>(cpu_best<nk_cast_best>)", test_cast<f32_t, bf16_t>, serial,
                    cpu_best<nk_cast_best>);
    run_kernel_case("test_cast<bf16_t, f32_t>(cpu_best<nk_cast_best>)", test_cast<bf16_t, f32_t>, serial,
                    cpu_best<nk_cast_best>);
    run_kernel_case("test_cast<f32_t, e4m3_t>(cpu_best<nk_cast_best>)", test_cast<f32_t, e4m3_t>, serial,
                    cpu_best<nk_cast_best>);
    run_kernel_case("test_cast<e4m3_t, f32_t>(cpu_best<nk_cast_best>)", test_cast<e4m3_t, f32_t>, serial,
                    cpu_best<nk_cast_best>);
    run_kernel_case("test_cast<f32_t, e5m2_t>(cpu_best<nk_cast_best>)", test_cast<f32_t, e5m2_t>, serial,
                    cpu_best<nk_cast_best>);
    run_kernel_case("test_cast<e5m2_t, f32_t>(cpu_best<nk_cast_best>)", test_cast<e5m2_t, f32_t>, serial,
                    cpu_best<nk_cast_best>);
    run_kernel_case("test_cast<f32_t, e2m3_t>(cpu_best<nk_cast_best>)", test_cast<f32_t, e2m3_t>, serial,
                    cpu_best<nk_cast_best>);
    run_kernel_case("test_cast<e2m3_t, f32_t>(cpu_best<nk_cast_best>)", test_cast<e2m3_t, f32_t>, serial,
                    cpu_best<nk_cast_best>);
    run_kernel_case("test_cast<f32_t, e3m2_t>(cpu_best<nk_cast_best>)", test_cast<f32_t, e3m2_t>, serial,
                    cpu_best<nk_cast_best>);
    run_kernel_case("test_cast<e3m2_t, f32_t>(cpu_best<nk_cast_best>)", test_cast<e3m2_t, f32_t>, serial,
                    cpu_best<nk_cast_best>);
    run_kernel_case("test_cast<f64_t, f32_t>(cpu_best<nk_cast_best>)", test_cast<f64_t, f32_t>, serial,
                    cpu_best<nk_cast_best>);
    run_kernel_case("test_cast<f32_t, f64_t>(cpu_best<nk_cast_best>)", test_cast<f32_t, f64_t>, serial,
                    cpu_best<nk_cast_best>);
    // Integer ↔ integer
    run_kernel_case("test_cast<i8_t, i32_t>(cpu_best<nk_cast_best>)", test_cast<i8_t, i32_t>, serial,
                    cpu_best<nk_cast_best>);
    run_kernel_case("test_cast<i32_t, i8_t>(cpu_best<nk_cast_best>)", test_cast<i32_t, i8_t>, serial,
                    cpu_best<nk_cast_best>);
    run_kernel_case("test_cast<u8_t, u32_t>(cpu_best<nk_cast_best>)", test_cast<u8_t, u32_t>, serial,
                    cpu_best<nk_cast_best>);
    run_kernel_case("test_cast<u32_t, u8_t>(cpu_best<nk_cast_best>)", test_cast<u32_t, u8_t>, serial,
                    cpu_best<nk_cast_best>);
    run_kernel_case("test_cast<i16_t, i64_t>(cpu_best<nk_cast_best>)", test_cast<i16_t, i64_t>, serial,
                    cpu_best<nk_cast_best>);
    run_kernel_case("test_cast<i64_t, i16_t>(cpu_best<nk_cast_best>)", test_cast<i64_t, i16_t>, serial,
                    cpu_best<nk_cast_best>);
    run_kernel_case("test_cast<i32_t, u32_t>(cpu_best<nk_cast_best>)", test_cast<i32_t, u32_t>, serial,
                    cpu_best<nk_cast_best>);
    // Integer ↔ float
    run_kernel_case("test_cast<i32_t, f64_t>(cpu_best<nk_cast_best>)", test_cast<i32_t, f64_t>, serial,
                    cpu_best<nk_cast_best>);
    run_kernel_case("test_cast<f64_t, i32_t>(cpu_best<nk_cast_best>)", test_cast<f64_t, i32_t>, serial,
                    cpu_best<nk_cast_best>);
    run_kernel_case("test_cast<i16_t, f32_t>(cpu_best<nk_cast_best>)", test_cast<i16_t, f32_t>, serial,
                    cpu_best<nk_cast_best>);
    run_kernel_case("test_cast<u8_t, f32_t>(cpu_best<nk_cast_best>)", test_cast<u8_t, f32_t>, serial,
                    cpu_best<nk_cast_best>);
    run_kernel_case("test_cast<f32_t, i8_t>(cpu_best<nk_cast_best>)", test_cast<f32_t, i8_t>, serial,
                    cpu_best<nk_cast_best>);
    run_kernel_case("test_cast<i8_t, f64_t>(cpu_best<nk_cast_best>)", test_cast<i8_t, f64_t>, serial,
                    cpu_best<nk_cast_best>);
    run_kernel_case("test_cast<f64_t, u8_t>(cpu_best<nk_cast_best>)", test_cast<f64_t, u8_t>, serial,
                    cpu_best<nk_cast_best>);
    // Verify serial fallbacks for rare paths
    run_kernel_case("test_cast<f64_t, f16_t>(cpu_best<nk_cast_best>)", test_cast<f64_t, f16_t>, serial,
                    cpu_best<nk_cast_best>);
    run_kernel_case("test_cast<f16_t, f64_t>(cpu_best<nk_cast_best>)", test_cast<f16_t, f64_t>, serial,
                    cpu_best<nk_cast_best>);
    run_kernel_case("test_cast<f64_t, bf16_t>(cpu_best<nk_cast_best>)", test_cast<f64_t, bf16_t>, serial,
                    cpu_best<nk_cast_best>);
    run_kernel_case("test_cast<bf16_t, f64_t>(cpu_best<nk_cast_best>)", test_cast<bf16_t, f64_t>, serial,
                    cpu_best<nk_cast_best>);
#endif

#if NUMKONG_TARGET_HASWELL
    auto const haswell = nk_cap_haswell_k;
    run_kernel_case("test_cast<f32_t, f16_t>(nk_cast_haswell)", test_cast<f32_t, f16_t>, haswell, nk_cast_haswell);
    run_kernel_case("test_cast<f16_t, f32_t>(nk_cast_haswell)", test_cast<f16_t, f32_t>, haswell, nk_cast_haswell);
    run_kernel_case("test_cast<f32_t, bf16_t>(nk_cast_haswell)", test_cast<f32_t, bf16_t>, haswell, nk_cast_haswell);
    run_kernel_case("test_cast<bf16_t, f32_t>(nk_cast_haswell)", test_cast<bf16_t, f32_t>, haswell, nk_cast_haswell);
    run_kernel_case("test_cast<f32_t, e4m3_t>(nk_cast_haswell)", test_cast<f32_t, e4m3_t>, haswell, nk_cast_haswell);
    run_kernel_case("test_cast<e4m3_t, f32_t>(nk_cast_haswell)", test_cast<e4m3_t, f32_t>, haswell, nk_cast_haswell);
    run_kernel_case("test_cast<f32_t, e5m2_t>(nk_cast_haswell)", test_cast<f32_t, e5m2_t>, haswell, nk_cast_haswell);
    run_kernel_case("test_cast<e5m2_t, f32_t>(nk_cast_haswell)", test_cast<e5m2_t, f32_t>, haswell, nk_cast_haswell);
    run_kernel_case("test_cast<e5m2_t, f16_t>(nk_cast_haswell)", test_cast<e5m2_t, f16_t>, haswell, nk_cast_haswell);
    run_kernel_case("test_cast<f32_t, e2m3_t>(nk_cast_haswell)", test_cast<f32_t, e2m3_t>, haswell, nk_cast_haswell);
    run_kernel_case("test_cast<e2m3_t, f32_t>(nk_cast_haswell)", test_cast<e2m3_t, f32_t>, haswell, nk_cast_haswell);
    run_kernel_case("test_cast<f32_t, e3m2_t>(nk_cast_haswell)", test_cast<f32_t, e3m2_t>, haswell, nk_cast_haswell);
    run_kernel_case("test_cast<e3m2_t, f32_t>(nk_cast_haswell)", test_cast<e3m2_t, f32_t>, haswell, nk_cast_haswell);
    run_kernel_case("test_cast<i8_t, f32_t>(nk_cast_haswell)", test_cast<i8_t, f32_t>, haswell, nk_cast_haswell);
    run_kernel_case("test_cast<f32_t, i8_t>(nk_cast_haswell)", test_cast<f32_t, i8_t>, haswell, nk_cast_haswell);
    run_kernel_case("test_cast<i16_t, f32_t>(nk_cast_haswell)", test_cast<i16_t, f32_t>, haswell, nk_cast_haswell);
    run_kernel_case("test_cast<f32_t, i16_t>(nk_cast_haswell)", test_cast<f32_t, i16_t>, haswell, nk_cast_haswell);
    run_kernel_case("test_cast<u16_t, f32_t>(nk_cast_haswell)", test_cast<u16_t, f32_t>, haswell, nk_cast_haswell);
    run_kernel_case("test_cast<f32_t, u16_t>(nk_cast_haswell)", test_cast<f32_t, u16_t>, haswell, nk_cast_haswell);
    run_kernel_case("test_cast<u8_t, f32_t>(nk_cast_haswell)", test_cast<u8_t, f32_t>, haswell, nk_cast_haswell);
    run_kernel_case("test_cast<f32_t, u8_t>(nk_cast_haswell)", test_cast<f32_t, u8_t>, haswell, nk_cast_haswell);
    run_kernel_case("test_cast<f32_t, i32_t>(nk_cast_haswell)", test_cast<f32_t, i32_t>, haswell, nk_cast_haswell);
    run_kernel_case("test_cast<f32_t, u32_t>(nk_cast_haswell)", test_cast<f32_t, u32_t>, haswell, nk_cast_haswell);
    // Verify serial fallbacks for rare paths
    run_kernel_case("test_cast<i32_t, f64_t>(nk_cast_haswell)", test_cast<i32_t, f64_t>, haswell, nk_cast_haswell);
    run_kernel_case("test_cast<f64_t, f32_t>(nk_cast_haswell)", test_cast<f64_t, f32_t>, haswell, nk_cast_haswell);
    run_kernel_case("test_cast_block_scaled<>(nk_cast_haswell)", test_cast_block_scaled<>, haswell, nk_cast_haswell);
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SKYLAKE
    auto const skylake = nk_cap_skylake_k;
    run_kernel_case("test_cast<f32_t, f16_t>(nk_cast_skylake)", test_cast<f32_t, f16_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<f16_t, f32_t>(nk_cast_skylake)", test_cast<f16_t, f32_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<f32_t, bf16_t>(nk_cast_skylake)", test_cast<f32_t, bf16_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<bf16_t, f32_t>(nk_cast_skylake)", test_cast<bf16_t, f32_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<f32_t, e4m3_t>(nk_cast_skylake)", test_cast<f32_t, e4m3_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<e4m3_t, f32_t>(nk_cast_skylake)", test_cast<e4m3_t, f32_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<f32_t, e5m2_t>(nk_cast_skylake)", test_cast<f32_t, e5m2_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<e5m2_t, f32_t>(nk_cast_skylake)", test_cast<e5m2_t, f32_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<f32_t, e2m3_t>(nk_cast_skylake)", test_cast<f32_t, e2m3_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<e2m3_t, f32_t>(nk_cast_skylake)", test_cast<e2m3_t, f32_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<f32_t, e3m2_t>(nk_cast_skylake)", test_cast<f32_t, e3m2_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<e3m2_t, f32_t>(nk_cast_skylake)", test_cast<e3m2_t, f32_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<f16_t, bf16_t>(nk_cast_skylake)", test_cast<f16_t, bf16_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<bf16_t, f16_t>(nk_cast_skylake)", test_cast<bf16_t, f16_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<e4m3_t, f16_t>(nk_cast_skylake)", test_cast<e4m3_t, f16_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<f16_t, e4m3_t>(nk_cast_skylake)", test_cast<f16_t, e4m3_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<e5m2_t, f16_t>(nk_cast_skylake)", test_cast<e5m2_t, f16_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<f16_t, e5m2_t>(nk_cast_skylake)", test_cast<f16_t, e5m2_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<e4m3_t, bf16_t>(nk_cast_skylake)", test_cast<e4m3_t, bf16_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<bf16_t, e4m3_t>(nk_cast_skylake)", test_cast<bf16_t, e4m3_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<e5m2_t, bf16_t>(nk_cast_skylake)", test_cast<e5m2_t, bf16_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<bf16_t, e5m2_t>(nk_cast_skylake)", test_cast<bf16_t, e5m2_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<f64_t, f32_t>(nk_cast_skylake)", test_cast<f64_t, f32_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<f32_t, f64_t>(nk_cast_skylake)", test_cast<f32_t, f64_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<i32_t, f64_t>(nk_cast_skylake)", test_cast<i32_t, f64_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<f64_t, i32_t>(nk_cast_skylake)", test_cast<f64_t, i32_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<i8_t, i32_t>(nk_cast_skylake)", test_cast<i8_t, i32_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<i32_t, i8_t>(nk_cast_skylake)", test_cast<i32_t, i8_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<i16_t, f32_t>(nk_cast_skylake)", test_cast<i16_t, f32_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<f32_t, i16_t>(nk_cast_skylake)", test_cast<f32_t, i16_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<u16_t, f32_t>(nk_cast_skylake)", test_cast<u16_t, f32_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<f32_t, u16_t>(nk_cast_skylake)", test_cast<f32_t, u16_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<u8_t, f32_t>(nk_cast_skylake)", test_cast<u8_t, f32_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<f32_t, u8_t>(nk_cast_skylake)", test_cast<f32_t, u8_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<i64_t, f64_t>(nk_cast_skylake)", test_cast<i64_t, f64_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<f64_t, i64_t>(nk_cast_skylake)", test_cast<f64_t, i64_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<u64_t, f64_t>(nk_cast_skylake)", test_cast<u64_t, f64_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<f64_t, u64_t>(nk_cast_skylake)", test_cast<f64_t, u64_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<u32_t, f64_t>(nk_cast_skylake)", test_cast<u32_t, f64_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<f64_t, u32_t>(nk_cast_skylake)", test_cast<f64_t, u32_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<u64_t, i64_t>(nk_cast_skylake)", test_cast<u64_t, i64_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<i64_t, u64_t>(nk_cast_skylake)", test_cast<i64_t, u64_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<u64_t, i32_t>(nk_cast_skylake)", test_cast<u64_t, i32_t>, skylake, nk_cast_skylake);
    // Verify serial fallbacks for rare paths
    run_kernel_case("test_cast<i8_t, f64_t>(nk_cast_skylake)", test_cast<i8_t, f64_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast<f64_t, bf16_t>(nk_cast_skylake)", test_cast<f64_t, bf16_t>, skylake, nk_cast_skylake);
    run_kernel_case("test_cast_block_scaled<>(nk_cast_skylake)", test_cast_block_scaled<>, skylake, nk_cast_skylake);
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_ICELAKE
    auto const icelake = nk_cap_icelake_k;
    run_kernel_case("test_cast<e4m3_t, bf16_t>(nk_cast_icelake)", test_cast<e4m3_t, bf16_t>, icelake, nk_cast_icelake);
    run_kernel_case("test_cast<bf16_t, e4m3_t>(nk_cast_icelake)", test_cast<bf16_t, e4m3_t>, icelake, nk_cast_icelake);
    run_kernel_case("test_cast<e5m2_t, bf16_t>(nk_cast_icelake)", test_cast<e5m2_t, bf16_t>, icelake, nk_cast_icelake);
    run_kernel_case("test_cast<bf16_t, e5m2_t>(nk_cast_icelake)", test_cast<bf16_t, e5m2_t>, icelake, nk_cast_icelake);
    run_kernel_case("test_cast<e4m3_t, f16_t>(nk_cast_icelake)", test_cast<e4m3_t, f16_t>, icelake, nk_cast_icelake);
    run_kernel_case("test_cast<e5m2_t, f16_t>(nk_cast_icelake)", test_cast<e5m2_t, f16_t>, icelake, nk_cast_icelake);
    run_kernel_case("test_cast<e4m3_t, f32_t>(nk_cast_icelake)", test_cast<e4m3_t, f32_t>, icelake, nk_cast_icelake);
    run_kernel_case("test_cast<f32_t, e4m3_t>(nk_cast_icelake)", test_cast<f32_t, e4m3_t>, icelake, nk_cast_icelake);
    run_kernel_case("test_cast<f16_t, f32_t>(nk_cast_icelake)", test_cast<f16_t, f32_t>, icelake, nk_cast_icelake);
    run_kernel_case("test_cast<f32_t, f16_t>(nk_cast_icelake)", test_cast<f32_t, f16_t>, icelake, nk_cast_icelake);
    run_kernel_case("test_cast<e2m3_t, f32_t>(nk_cast_icelake)", test_cast<e2m3_t, f32_t>, icelake, nk_cast_icelake);
    run_kernel_case("test_cast<f32_t, e2m3_t>(nk_cast_icelake)", test_cast<f32_t, e2m3_t>, icelake, nk_cast_icelake);
    run_kernel_case("test_cast<e3m2_t, f32_t>(nk_cast_icelake)", test_cast<e3m2_t, f32_t>, icelake, nk_cast_icelake);
    run_kernel_case("test_cast<f32_t, e3m2_t>(nk_cast_icelake)", test_cast<f32_t, e3m2_t>, icelake, nk_cast_icelake);
    run_kernel_case("test_cast_block_scaled<>(nk_cast_icelake)", test_cast_block_scaled<>, icelake, nk_cast_icelake);
#endif // NUMKONG_TARGET_ICELAKE

#if NUMKONG_TARGET_SAPPHIRE
    auto const sapphire = nk_cap_sapphire_k;
    run_kernel_case("test_cast<e4m3_t, f16_t>(nk_cast_sapphire)", test_cast<e4m3_t, f16_t>, sapphire, nk_cast_sapphire);
    run_kernel_case("test_cast<f16_t, e4m3_t>(nk_cast_sapphire)", test_cast<f16_t, e4m3_t>, sapphire, nk_cast_sapphire);
    run_kernel_case("test_cast<e5m2_t, f16_t>(nk_cast_sapphire)", test_cast<e5m2_t, f16_t>, sapphire, nk_cast_sapphire);
    run_kernel_case("test_cast<f16_t, e5m2_t>(nk_cast_sapphire)", test_cast<f16_t, e5m2_t>, sapphire, nk_cast_sapphire);
    run_kernel_case("test_cast<f16_t, f32_t>(nk_cast_sapphire)", test_cast<f16_t, f32_t>, sapphire, nk_cast_sapphire);
    run_kernel_case("test_cast<f32_t, f16_t>(nk_cast_sapphire)", test_cast<f32_t, f16_t>, sapphire, nk_cast_sapphire);
#endif // NUMKONG_TARGET_SAPPHIRE

#if NUMKONG_TARGET_NEON
    run_kernel_case("test_cast<bf16_t, f32_t>(nk_cast_neon)", test_cast<bf16_t, f32_t>, nk_cap_neon_k, nk_cast_neon);
    run_kernel_case("test_cast<f32_t, bf16_t>(nk_cast_neon)", test_cast<f32_t, bf16_t>, nk_cap_neon_k, nk_cast_neon);
    run_kernel_case("test_cast<e4m3_t, f32_t>(nk_cast_neon)", test_cast<e4m3_t, f32_t>, nk_cap_neon_k, nk_cast_neon);
    run_kernel_case("test_cast<f32_t, e4m3_t>(nk_cast_neon)", test_cast<f32_t, e4m3_t>, nk_cap_neon_k, nk_cast_neon);
    run_kernel_case("test_cast<e5m2_t, f32_t>(nk_cast_neon)", test_cast<e5m2_t, f32_t>, nk_cap_neon_k, nk_cast_neon);
    run_kernel_case("test_cast<e5m2_t, f16_t>(nk_cast_neon)", test_cast<e5m2_t, f16_t>, nk_cap_neon_k, nk_cast_neon);
    run_kernel_case("test_cast<f32_t, e5m2_t>(nk_cast_neon)", test_cast<f32_t, e5m2_t>, nk_cap_neon_k, nk_cast_neon);
    run_kernel_case("test_cast<f32_t, i4x2_t>(nk_cast_neon)", test_cast<f32_t, i4x2_t>, nk_cap_neon_k, nk_cast_neon);
    run_kernel_case("test_cast<f32_t, u4x2_t>(nk_cast_neon)", test_cast<f32_t, u4x2_t>, nk_cap_neon_k, nk_cast_neon);
    run_kernel_case("test_cast_block_scaled<>(nk_cast_neon)", test_cast_block_scaled<>, nk_cap_neon_k, nk_cast_neon);
#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_V128RELAXED
    auto const v128relaxed = nk_cap_v128relaxed_k;
    run_kernel_case("test_cast<f32_t, f16_t>(nk_cast_v128relaxed)", test_cast<f32_t, f16_t>, v128relaxed,
                    nk_cast_v128relaxed);
    run_kernel_case("test_cast<f16_t, f32_t>(nk_cast_v128relaxed)", test_cast<f16_t, f32_t>, v128relaxed,
                    nk_cast_v128relaxed);
    run_kernel_case("test_cast<f32_t, bf16_t>(nk_cast_v128relaxed)", test_cast<f32_t, bf16_t>, v128relaxed,
                    nk_cast_v128relaxed);
    run_kernel_case("test_cast<bf16_t, f32_t>(nk_cast_v128relaxed)", test_cast<bf16_t, f32_t>, v128relaxed,
                    nk_cast_v128relaxed);
    run_kernel_case("test_cast<f32_t, e4m3_t>(nk_cast_v128relaxed)", test_cast<f32_t, e4m3_t>, v128relaxed,
                    nk_cast_v128relaxed);
    run_kernel_case("test_cast<e4m3_t, f32_t>(nk_cast_v128relaxed)", test_cast<e4m3_t, f32_t>, v128relaxed,
                    nk_cast_v128relaxed);
    run_kernel_case("test_cast<f32_t, e5m2_t>(nk_cast_v128relaxed)", test_cast<f32_t, e5m2_t>, v128relaxed,
                    nk_cast_v128relaxed);
    run_kernel_case("test_cast<e5m2_t, f32_t>(nk_cast_v128relaxed)", test_cast<e5m2_t, f32_t>, v128relaxed,
                    nk_cast_v128relaxed);
    run_kernel_case("test_cast<e5m2_t, bf16_t>(nk_cast_v128relaxed)", test_cast<e5m2_t, bf16_t>, v128relaxed,
                    nk_cast_v128relaxed);
    run_kernel_case("test_cast<e5m2_t, f16_t>(nk_cast_v128relaxed)", test_cast<e5m2_t, f16_t>, v128relaxed,
                    nk_cast_v128relaxed);
    run_kernel_case("test_cast<f32_t, e2m3_t>(nk_cast_v128relaxed)", test_cast<f32_t, e2m3_t>, v128relaxed,
                    nk_cast_v128relaxed);
    run_kernel_case("test_cast<e2m3_t, f32_t>(nk_cast_v128relaxed)", test_cast<e2m3_t, f32_t>, v128relaxed,
                    nk_cast_v128relaxed);
    run_kernel_case("test_cast<f32_t, e3m2_t>(nk_cast_v128relaxed)", test_cast<f32_t, e3m2_t>, v128relaxed,
                    nk_cast_v128relaxed);
    run_kernel_case("test_cast<e3m2_t, f32_t>(nk_cast_v128relaxed)", test_cast<e3m2_t, f32_t>, v128relaxed,
                    nk_cast_v128relaxed);
    run_kernel_case("test_cast<i8_t, f32_t>(nk_cast_v128relaxed)", test_cast<i8_t, f32_t>, v128relaxed,
                    nk_cast_v128relaxed);
    run_kernel_case("test_cast<f32_t, i8_t>(nk_cast_v128relaxed)", test_cast<f32_t, i8_t>, v128relaxed,
                    nk_cast_v128relaxed);
    run_kernel_case("test_cast<u8_t, f32_t>(nk_cast_v128relaxed)", test_cast<u8_t, f32_t>, v128relaxed,
                    nk_cast_v128relaxed);
    run_kernel_case("test_cast<f32_t, u8_t>(nk_cast_v128relaxed)", test_cast<f32_t, u8_t>, v128relaxed,
                    nk_cast_v128relaxed);
#endif // NUMKONG_TARGET_V128RELAXED

#if NUMKONG_TARGET_RVV
    run_kernel_case("test_cast<bf16_t, f32_t>(nk_cast_rvv)", test_cast<bf16_t, f32_t>, nk_cap_rvv_k, nk_cast_rvv);
    run_kernel_case("test_cast<f32_t, bf16_t>(nk_cast_rvv)", test_cast<f32_t, bf16_t>, nk_cap_rvv_k, nk_cast_rvv);
    run_kernel_case("test_cast<e4m3_t, f32_t>(nk_cast_rvv)", test_cast<e4m3_t, f32_t>, nk_cap_rvv_k, nk_cast_rvv);
    run_kernel_case("test_cast<e5m2_t, f32_t>(nk_cast_rvv)", test_cast<e5m2_t, f32_t>, nk_cap_rvv_k, nk_cast_rvv);
    run_kernel_case("test_cast<e5m2_t, bf16_t>(nk_cast_rvv)", test_cast<e5m2_t, bf16_t>, nk_cap_rvv_k, nk_cast_rvv);
#endif // NUMKONG_TARGET_RVV

#if NUMKONG_TARGET_POWERVSX
    auto const powervsx = nk_cap_powervsx_k;
    run_kernel_case("test_cast<f32_t, f16_t>(nk_cast_powervsx)", test_cast<f32_t, f16_t>, powervsx, nk_cast_powervsx);
    run_kernel_case("test_cast<f16_t, f32_t>(nk_cast_powervsx)", test_cast<f16_t, f32_t>, powervsx, nk_cast_powervsx);
    run_kernel_case("test_cast<f32_t, bf16_t>(nk_cast_powervsx)", test_cast<f32_t, bf16_t>, powervsx, nk_cast_powervsx);
    run_kernel_case("test_cast<bf16_t, f32_t>(nk_cast_powervsx)", test_cast<bf16_t, f32_t>, powervsx, nk_cast_powervsx);
    run_kernel_case("test_cast<i8_t, f32_t>(nk_cast_powervsx)", test_cast<i8_t, f32_t>, powervsx, nk_cast_powervsx);
    run_kernel_case("test_cast<f32_t, i8_t>(nk_cast_powervsx)", test_cast<f32_t, i8_t>, powervsx, nk_cast_powervsx);
    run_kernel_case("test_cast<u8_t, f32_t>(nk_cast_powervsx)", test_cast<u8_t, f32_t>, powervsx, nk_cast_powervsx);
    run_kernel_case("test_cast<f32_t, u8_t>(nk_cast_powervsx)", test_cast<f32_t, u8_t>, powervsx, nk_cast_powervsx);
    run_kernel_case("test_cast<i16_t, f32_t>(nk_cast_powervsx)", test_cast<i16_t, f32_t>, powervsx, nk_cast_powervsx);
    run_kernel_case("test_cast<f32_t, i16_t>(nk_cast_powervsx)", test_cast<f32_t, i16_t>, powervsx, nk_cast_powervsx);
    run_kernel_case("test_cast<u16_t, f32_t>(nk_cast_powervsx)", test_cast<u16_t, f32_t>, powervsx, nk_cast_powervsx);
    run_kernel_case("test_cast<f32_t, u16_t>(nk_cast_powervsx)", test_cast<f32_t, u16_t>, powervsx, nk_cast_powervsx);
#endif // NUMKONG_TARGET_POWERVSX
}

} // namespace ashvardanian::numkong::test
