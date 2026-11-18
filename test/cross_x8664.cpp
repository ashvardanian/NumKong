/**
 *  @file test/cross_x8664.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *  @brief Batch operation tests - x86-64 ISA family.
 *
 *  Covers Haswell, Skylake, Ice Lake, Genoa, Diamond, Alder and Sierra, while
 *  @c cross_x8664amx.cpp covers the AMX tiles.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include "harness.hpp"
#include "cross.hpp"

namespace ashvardanian::numkong::test {

void run_cross_x8664_cases() {
#if NUMKONG_TARGET_HASWELL
    auto const haswell = nk_cap_haswell_k;
    run_kernel_case("nk_dots_packed_f64_haswell", test_dots_packed<f64_t>, haswell, nk_dots_pack_size_f64_haswell,
                    nk_dots_pack_f64_haswell, nk_dots_packed_f64_haswell);
    run_kernel_case("nk_dots_pack_f64_haswell", test_dots_pack_layout<f64_t, host_backend_t>, haswell,
                    nk_dots_pack_size_f64_haswell, nk_dots_packed_shape_f64_haswell, nk_dots_pack_f64_haswell);
    run_kernel_case("nk_dots_packed_f32_haswell", test_dots_packed<f32_t>, haswell, nk_dots_pack_size_f32_haswell,
                    nk_dots_pack_f32_haswell, nk_dots_packed_f32_haswell);
    run_kernel_case("nk_dots_pack_f32_haswell", test_dots_pack_layout<f32_t, host_backend_t>, haswell,
                    nk_dots_pack_size_f32_haswell, nk_dots_packed_shape_f32_haswell, nk_dots_pack_f32_haswell);
    run_kernel_case("nk_dots_packed_bf16_haswell", test_dots_packed<bf16_t>, haswell, nk_dots_pack_size_bf16_haswell,
                    nk_dots_pack_bf16_haswell, nk_dots_packed_bf16_haswell);
    run_kernel_case("nk_dots_pack_bf16_haswell", test_dots_pack_layout<bf16_t, host_backend_t>, haswell,
                    nk_dots_pack_size_bf16_haswell, nk_dots_packed_shape_bf16_haswell, nk_dots_pack_bf16_haswell);
    run_kernel_case("nk_dots_packed_f16_haswell", test_dots_packed<f16_t>, haswell, nk_dots_pack_size_f16_haswell,
                    nk_dots_pack_f16_haswell, nk_dots_packed_f16_haswell);
    run_kernel_case("nk_dots_pack_f16_haswell", test_dots_pack_layout<f16_t, host_backend_t>, haswell,
                    nk_dots_pack_size_f16_haswell, nk_dots_packed_shape_f16_haswell, nk_dots_pack_f16_haswell);
    run_kernel_case("nk_dots_packed_e5m2_haswell", test_dots_packed<e5m2_t>, haswell, nk_dots_pack_size_e5m2_haswell,
                    nk_dots_pack_e5m2_haswell, nk_dots_packed_e5m2_haswell);
    run_kernel_case("nk_dots_pack_e5m2_haswell", test_dots_pack_layout<e5m2_t, host_backend_t>, haswell,
                    nk_dots_pack_size_e5m2_haswell, nk_dots_packed_shape_e5m2_haswell, nk_dots_pack_e5m2_haswell);
    run_kernel_case("nk_dots_packed_e4m3_haswell", test_dots_packed<e4m3_t>, haswell, nk_dots_pack_size_e4m3_haswell,
                    nk_dots_pack_e4m3_haswell, nk_dots_packed_e4m3_haswell);
    run_kernel_case("nk_dots_pack_e4m3_haswell", test_dots_pack_layout<e4m3_t, host_backend_t>, haswell,
                    nk_dots_pack_size_e4m3_haswell, nk_dots_packed_shape_e4m3_haswell, nk_dots_pack_e4m3_haswell);
    run_kernel_case("nk_dots_packed_e3m2_haswell", test_dots_packed<e3m2_t>, haswell, nk_dots_pack_size_e3m2_haswell,
                    nk_dots_pack_e3m2_haswell, nk_dots_packed_e3m2_haswell);
    run_kernel_case("nk_dots_pack_e3m2_haswell", test_dots_pack_layout<e3m2_t, host_backend_t>, haswell,
                    nk_dots_pack_size_e3m2_haswell, nk_dots_packed_shape_e3m2_haswell, nk_dots_pack_e3m2_haswell);
    run_kernel_case("nk_dots_packed_e2m3_haswell", test_dots_packed<e2m3_t>, haswell, nk_dots_pack_size_e2m3_haswell,
                    nk_dots_pack_e2m3_haswell, nk_dots_packed_e2m3_haswell);
    run_kernel_case("nk_dots_pack_e2m3_haswell", test_dots_pack_layout<e2m3_t, host_backend_t>, haswell,
                    nk_dots_pack_size_e2m3_haswell, nk_dots_packed_shape_e2m3_haswell, nk_dots_pack_e2m3_haswell);
    run_kernel_case("nk_dots_packed_e2m1_haswell", test_dots_packed<e2m1x2_t>, haswell, nk_dots_pack_size_e2m1_haswell,
                    nk_dots_pack_e2m1_haswell, nk_dots_packed_e2m1_haswell);
    run_kernel_case("nk_dots_pack_e2m1_haswell", test_dots_pack_layout<e2m1x2_t, host_backend_t>, haswell,
                    nk_dots_pack_size_e2m1_haswell, nk_dots_packed_shape_e2m1_haswell, nk_dots_pack_e2m1_haswell);
    run_kernel_case("nk_dots_packed_i8_haswell", test_dots_packed<i8_t>, haswell, nk_dots_pack_size_i8_haswell,
                    nk_dots_pack_i8_haswell, nk_dots_packed_i8_haswell);
    run_kernel_case("nk_dots_pack_i8_haswell", test_dots_pack_layout<i8_t, host_backend_t>, haswell,
                    nk_dots_pack_size_i8_haswell, nk_dots_packed_shape_i8_haswell, nk_dots_pack_i8_haswell);
    run_kernel_case("nk_dots_packed_i4_haswell", test_dots_packed<i4x2_t>, haswell, nk_dots_pack_size_i4_haswell,
                    nk_dots_pack_i4_haswell, nk_dots_packed_i4_haswell);
    run_kernel_case("nk_dots_pack_i4_haswell", test_dots_pack_layout<i4x2_t, host_backend_t>, haswell,
                    nk_dots_pack_size_i4_haswell, nk_dots_packed_shape_i4_haswell, nk_dots_pack_i4_haswell);
    run_kernel_case("nk_dots_packed_u8_haswell", test_dots_packed<u8_t>, haswell, nk_dots_pack_size_u8_haswell,
                    nk_dots_pack_u8_haswell, nk_dots_packed_u8_haswell);
    run_kernel_case("nk_dots_pack_u8_haswell", test_dots_pack_layout<u8_t, host_backend_t>, haswell,
                    nk_dots_pack_size_u8_haswell, nk_dots_packed_shape_u8_haswell, nk_dots_pack_u8_haswell);
    run_kernel_case("nk_dots_packed_u4_haswell", test_dots_packed<u4x2_t>, haswell, nk_dots_pack_size_u4_haswell,
                    nk_dots_pack_u4_haswell, nk_dots_packed_u4_haswell);
    run_kernel_case("nk_dots_pack_u4_haswell", test_dots_pack_layout<u4x2_t, host_backend_t>, haswell,
                    nk_dots_pack_size_u4_haswell, nk_dots_packed_shape_u4_haswell, nk_dots_pack_u4_haswell);
    run_kernel_case("nk_dots_packed_u1_haswell", test_dots_packed<u1x8_t>, haswell, nk_dots_pack_size_u1_haswell,
                    nk_dots_pack_u1_haswell, nk_dots_packed_u1_haswell);

    run_kernel_case("nk_dots_symmetric_f64_haswell", test_dots_symmetric<f64_t>, haswell,
                    nk_dots_symmetric_f64_haswell);
    run_kernel_case("nk_dots_symmetric_f32_haswell", test_dots_symmetric<f32_t>, haswell,
                    nk_dots_symmetric_f32_haswell);
    run_kernel_case("nk_dots_symmetric_bf16_haswell", test_dots_symmetric<bf16_t>, haswell,
                    nk_dots_symmetric_bf16_haswell);
    run_kernel_case("nk_dots_symmetric_f16_haswell", test_dots_symmetric<f16_t>, haswell,
                    nk_dots_symmetric_f16_haswell);
    run_kernel_case("nk_dots_symmetric_e5m2_haswell", test_dots_symmetric<e5m2_t>, haswell,
                    nk_dots_symmetric_e5m2_haswell);
    run_kernel_case("nk_dots_symmetric_e4m3_haswell", test_dots_symmetric<e4m3_t>, haswell,
                    nk_dots_symmetric_e4m3_haswell);
    run_kernel_case("nk_dots_symmetric_e3m2_haswell", test_dots_symmetric<e3m2_t>, haswell,
                    nk_dots_symmetric_e3m2_haswell);
    run_kernel_case("nk_dots_symmetric_e2m3_haswell", test_dots_symmetric<e2m3_t>, haswell,
                    nk_dots_symmetric_e2m3_haswell);
    run_kernel_case("nk_dots_symmetric_e2m1_haswell", test_dots_symmetric<e2m1x2_t>, haswell,
                    nk_dots_symmetric_e2m1_haswell);
    run_kernel_case("nk_dots_symmetric_i8_haswell", test_dots_symmetric<i8_t>, haswell, nk_dots_symmetric_i8_haswell);
    run_kernel_case("nk_dots_symmetric_i4_haswell", test_dots_symmetric<i4x2_t>, haswell, nk_dots_symmetric_i4_haswell);
    run_kernel_case("nk_dots_symmetric_u8_haswell", test_dots_symmetric<u8_t>, haswell, nk_dots_symmetric_u8_haswell);
    run_kernel_case("nk_dots_symmetric_u4_haswell", test_dots_symmetric<u4x2_t>, haswell, nk_dots_symmetric_u4_haswell);
    run_kernel_case("nk_dots_symmetric_u1_haswell", test_dots_symmetric<u1x8_t>, haswell, nk_dots_symmetric_u1_haswell);

    run_kernel_case("nk_hammings_packed_u1_haswell", test_hammings_packed<u1x8_t>, haswell,
                    nk_dots_pack_size_u1_haswell, nk_dots_pack_u1_haswell, nk_hammings_packed_u1_haswell);
    run_kernel_case("nk_hammings_symmetric_u1_haswell", test_hammings_symmetric<u1x8_t>, haswell,
                    nk_hammings_symmetric_u1_haswell);

    run_kernel_case("nk_jaccards_packed_u1_haswell", test_jaccards_packed<u1x8_t>, haswell,
                    nk_dots_pack_size_u1_haswell, nk_dots_pack_u1_haswell, nk_jaccards_packed_u1_haswell);
    run_kernel_case("nk_jaccards_symmetric_u1_haswell", test_jaccards_symmetric<u1x8_t>, haswell,
                    nk_jaccards_symmetric_u1_haswell);

    run_kernel_case("nk_angulars_packed_f64_haswell", test_angulars_packed<f64_t>, haswell,
                    nk_dots_pack_size_f64_haswell, nk_dots_pack_f64_haswell, nk_angulars_packed_f64_haswell);
    run_kernel_case("nk_angulars_packed_f32_haswell", test_angulars_packed<f32_t>, haswell,
                    nk_dots_pack_size_f32_haswell, nk_dots_pack_f32_haswell, nk_angulars_packed_f32_haswell);
    run_kernel_case("nk_angulars_packed_bf16_haswell", test_angulars_packed<bf16_t>, haswell,
                    nk_dots_pack_size_bf16_haswell, nk_dots_pack_bf16_haswell, nk_angulars_packed_bf16_haswell);
    run_kernel_case("nk_angulars_packed_f16_haswell", test_angulars_packed<f16_t>, haswell,
                    nk_dots_pack_size_f16_haswell, nk_dots_pack_f16_haswell, nk_angulars_packed_f16_haswell);
    run_kernel_case("nk_angulars_packed_e5m2_haswell", test_angulars_packed<e5m2_t>, haswell,
                    nk_dots_pack_size_e5m2_haswell, nk_dots_pack_e5m2_haswell, nk_angulars_packed_e5m2_haswell);
    run_kernel_case("nk_angulars_packed_e4m3_haswell", test_angulars_packed<e4m3_t>, haswell,
                    nk_dots_pack_size_e4m3_haswell, nk_dots_pack_e4m3_haswell, nk_angulars_packed_e4m3_haswell);
    run_kernel_case("nk_angulars_packed_e3m2_haswell", test_angulars_packed<e3m2_t>, haswell,
                    nk_dots_pack_size_e3m2_haswell, nk_dots_pack_e3m2_haswell, nk_angulars_packed_e3m2_haswell);
    run_kernel_case("nk_angulars_packed_e2m3_haswell", test_angulars_packed<e2m3_t>, haswell,
                    nk_dots_pack_size_e2m3_haswell, nk_dots_pack_e2m3_haswell, nk_angulars_packed_e2m3_haswell);
    run_kernel_case("nk_angulars_packed_e2m1_haswell", test_angulars_packed<e2m1x2_t>, haswell,
                    nk_dots_pack_size_e2m1_haswell, nk_dots_pack_e2m1_haswell, nk_angulars_packed_e2m1_haswell);
    run_kernel_case("nk_angulars_packed_i8_haswell", test_angulars_packed<i8_t>, haswell, nk_dots_pack_size_i8_haswell,
                    nk_dots_pack_i8_haswell, nk_angulars_packed_i8_haswell);
    run_kernel_case("nk_angulars_packed_i4_haswell", test_angulars_packed<i4x2_t>, haswell,
                    nk_dots_pack_size_i4_haswell, nk_dots_pack_i4_haswell, nk_angulars_packed_i4_haswell);
    run_kernel_case("nk_angulars_packed_u8_haswell", test_angulars_packed<u8_t>, haswell, nk_dots_pack_size_u8_haswell,
                    nk_dots_pack_u8_haswell, nk_angulars_packed_u8_haswell);
    run_kernel_case("nk_angulars_packed_u4_haswell", test_angulars_packed<u4x2_t>, haswell,
                    nk_dots_pack_size_u4_haswell, nk_dots_pack_u4_haswell, nk_angulars_packed_u4_haswell);

    run_kernel_case("nk_angulars_symmetric_f64_haswell", test_angulars_symmetric<f64_t>, haswell,
                    nk_angulars_symmetric_f64_haswell);
    run_kernel_case("nk_angulars_symmetric_f32_haswell", test_angulars_symmetric<f32_t>, haswell,
                    nk_angulars_symmetric_f32_haswell);
    run_kernel_case("nk_angulars_symmetric_bf16_haswell", test_angulars_symmetric<bf16_t>, haswell,
                    nk_angulars_symmetric_bf16_haswell);
    run_kernel_case("nk_angulars_symmetric_f16_haswell", test_angulars_symmetric<f16_t>, haswell,
                    nk_angulars_symmetric_f16_haswell);
    run_kernel_case("nk_angulars_symmetric_e5m2_haswell", test_angulars_symmetric<e5m2_t>, haswell,
                    nk_angulars_symmetric_e5m2_haswell);
    run_kernel_case("nk_angulars_symmetric_e4m3_haswell", test_angulars_symmetric<e4m3_t>, haswell,
                    nk_angulars_symmetric_e4m3_haswell);
    run_kernel_case("nk_angulars_symmetric_e3m2_haswell", test_angulars_symmetric<e3m2_t>, haswell,
                    nk_angulars_symmetric_e3m2_haswell);
    run_kernel_case("nk_angulars_symmetric_e2m3_haswell", test_angulars_symmetric<e2m3_t>, haswell,
                    nk_angulars_symmetric_e2m3_haswell);
    run_kernel_case("nk_angulars_symmetric_e2m1_haswell", test_angulars_symmetric<e2m1x2_t>, haswell,
                    nk_angulars_symmetric_e2m1_haswell);
    run_kernel_case("nk_angulars_symmetric_i8_haswell", test_angulars_symmetric<i8_t>, haswell,
                    nk_angulars_symmetric_i8_haswell);
    run_kernel_case("nk_angulars_symmetric_i4_haswell", test_angulars_symmetric<i4x2_t>, haswell,
                    nk_angulars_symmetric_i4_haswell);
    run_kernel_case("nk_angulars_symmetric_u8_haswell", test_angulars_symmetric<u8_t>, haswell,
                    nk_angulars_symmetric_u8_haswell);
    run_kernel_case("nk_angulars_symmetric_u4_haswell", test_angulars_symmetric<u4x2_t>, haswell,
                    nk_angulars_symmetric_u4_haswell);

    run_kernel_case("nk_euclideans_packed_f64_haswell", test_euclideans_packed<f64_t>, haswell,
                    nk_dots_pack_size_f64_haswell, nk_dots_pack_f64_haswell, nk_euclideans_packed_f64_haswell);
    run_kernel_case("nk_euclideans_packed_f32_haswell", test_euclideans_packed<f32_t>, haswell,
                    nk_dots_pack_size_f32_haswell, nk_dots_pack_f32_haswell, nk_euclideans_packed_f32_haswell);
    run_kernel_case("nk_euclideans_packed_bf16_haswell", test_euclideans_packed<bf16_t>, haswell,
                    nk_dots_pack_size_bf16_haswell, nk_dots_pack_bf16_haswell, nk_euclideans_packed_bf16_haswell);
    run_kernel_case("nk_euclideans_packed_f16_haswell", test_euclideans_packed<f16_t>, haswell,
                    nk_dots_pack_size_f16_haswell, nk_dots_pack_f16_haswell, nk_euclideans_packed_f16_haswell);
    run_kernel_case("nk_euclideans_packed_e5m2_haswell", test_euclideans_packed<e5m2_t>, haswell,
                    nk_dots_pack_size_e5m2_haswell, nk_dots_pack_e5m2_haswell, nk_euclideans_packed_e5m2_haswell);
    run_kernel_case("nk_euclideans_packed_e4m3_haswell", test_euclideans_packed<e4m3_t>, haswell,
                    nk_dots_pack_size_e4m3_haswell, nk_dots_pack_e4m3_haswell, nk_euclideans_packed_e4m3_haswell);
    run_kernel_case("nk_euclideans_packed_e3m2_haswell", test_euclideans_packed<e3m2_t>, haswell,
                    nk_dots_pack_size_e3m2_haswell, nk_dots_pack_e3m2_haswell, nk_euclideans_packed_e3m2_haswell);
    run_kernel_case("nk_euclideans_packed_e2m3_haswell", test_euclideans_packed<e2m3_t>, haswell,
                    nk_dots_pack_size_e2m3_haswell, nk_dots_pack_e2m3_haswell, nk_euclideans_packed_e2m3_haswell);
    run_kernel_case("nk_euclideans_packed_e2m1_haswell", test_euclideans_packed<e2m1x2_t>, haswell,
                    nk_dots_pack_size_e2m1_haswell, nk_dots_pack_e2m1_haswell, nk_euclideans_packed_e2m1_haswell);
    run_kernel_case("nk_euclideans_packed_i8_haswell", test_euclideans_packed<i8_t>, haswell,
                    nk_dots_pack_size_i8_haswell, nk_dots_pack_i8_haswell, nk_euclideans_packed_i8_haswell);
    run_kernel_case("nk_euclideans_packed_i4_haswell", test_euclideans_packed<i4x2_t>, haswell,
                    nk_dots_pack_size_i4_haswell, nk_dots_pack_i4_haswell, nk_euclideans_packed_i4_haswell);
    run_kernel_case("nk_euclideans_packed_u8_haswell", test_euclideans_packed<u8_t>, haswell,
                    nk_dots_pack_size_u8_haswell, nk_dots_pack_u8_haswell, nk_euclideans_packed_u8_haswell);
    run_kernel_case("nk_euclideans_packed_u4_haswell", test_euclideans_packed<u4x2_t>, haswell,
                    nk_dots_pack_size_u4_haswell, nk_dots_pack_u4_haswell, nk_euclideans_packed_u4_haswell);

    run_kernel_case("nk_euclideans_symmetric_f64_haswell", test_euclideans_symmetric<f64_t>, haswell,
                    nk_euclideans_symmetric_f64_haswell);
    run_kernel_case("nk_euclideans_symmetric_f32_haswell", test_euclideans_symmetric<f32_t>, haswell,
                    nk_euclideans_symmetric_f32_haswell);
    run_kernel_case("nk_euclideans_symmetric_bf16_haswell", test_euclideans_symmetric<bf16_t>, haswell,
                    nk_euclideans_symmetric_bf16_haswell);
    run_kernel_case("nk_euclideans_symmetric_f16_haswell", test_euclideans_symmetric<f16_t>, haswell,
                    nk_euclideans_symmetric_f16_haswell);
    run_kernel_case("nk_euclideans_symmetric_e5m2_haswell", test_euclideans_symmetric<e5m2_t>, haswell,
                    nk_euclideans_symmetric_e5m2_haswell);
    run_kernel_case("nk_euclideans_symmetric_e4m3_haswell", test_euclideans_symmetric<e4m3_t>, haswell,
                    nk_euclideans_symmetric_e4m3_haswell);
    run_kernel_case("nk_euclideans_symmetric_e3m2_haswell", test_euclideans_symmetric<e3m2_t>, haswell,
                    nk_euclideans_symmetric_e3m2_haswell);
    run_kernel_case("nk_euclideans_symmetric_e2m3_haswell", test_euclideans_symmetric<e2m3_t>, haswell,
                    nk_euclideans_symmetric_e2m3_haswell);
    run_kernel_case("nk_euclideans_symmetric_e2m1_haswell", test_euclideans_symmetric<e2m1x2_t>, haswell,
                    nk_euclideans_symmetric_e2m1_haswell);
    run_kernel_case("nk_euclideans_symmetric_i8_haswell", test_euclideans_symmetric<i8_t>, haswell,
                    nk_euclideans_symmetric_i8_haswell);
    run_kernel_case("nk_euclideans_symmetric_i4_haswell", test_euclideans_symmetric<i4x2_t>, haswell,
                    nk_euclideans_symmetric_i4_haswell);
    run_kernel_case("nk_euclideans_symmetric_u8_haswell", test_euclideans_symmetric<u8_t>, haswell,
                    nk_euclideans_symmetric_u8_haswell);
    run_kernel_case("nk_euclideans_symmetric_u4_haswell", test_euclideans_symmetric<u4x2_t>, haswell,
                    nk_euclideans_symmetric_u4_haswell);

    run_kernel_case("nk_attention_packed_bf16_haswell", test_attention_packed<bf16_t>, haswell,
                    nk_attention_pack_size_bf16_haswell, nk_attention_pack_bf16_haswell,
                    nk_attention_packed_bf16_haswell);
    run_kernel_case("nk_attention_packed_e4m3_haswell", test_attention_packed<e4m3_t>, haswell,
                    nk_attention_pack_size_e4m3_haswell, nk_attention_pack_e4m3_haswell,
                    nk_attention_packed_e4m3_haswell);
    run_kernel_case("nk_attention_packed_i8_haswell", test_attention_packed<i8_t>, haswell,
                    nk_attention_pack_size_i8_haswell, nk_attention_pack_i8_haswell, nk_attention_packed_i8_haswell);
    run_kernel_case("nk_attention_rope_f32_haswell", test_attention_rope<f32_t>, haswell,
                    nk_attention_rope_f32_haswell);
    run_kernel_case("nk_attention_rope_bf16_haswell", test_attention_rope<bf16_t>, haswell,
                    nk_attention_rope_bf16_haswell);
    run_kernel_case("nk_attention_rope_e4m3_haswell", test_attention_rope<e4m3_t>, haswell,
                    nk_attention_rope_e4m3_haswell);
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SKYLAKE
    auto const skylake = nk_cap_skylake_k;
    run_kernel_case("nk_dots_packed_f64_skylake", test_dots_packed<f64_t>, skylake, nk_dots_pack_size_f64_skylake,
                    nk_dots_pack_f64_skylake, nk_dots_packed_f64_skylake);
    run_kernel_case("nk_dots_pack_f64_skylake", test_dots_pack_layout<f64_t, host_backend_t>, skylake,
                    nk_dots_pack_size_f64_skylake, nk_dots_packed_shape_f64_skylake, nk_dots_pack_f64_skylake);
    run_kernel_case("nk_dots_packed_f32_skylake", test_dots_packed<f32_t>, skylake, nk_dots_pack_size_f32_skylake,
                    nk_dots_pack_f32_skylake, nk_dots_packed_f32_skylake);
    run_kernel_case("nk_dots_pack_f32_skylake", test_dots_pack_layout<f32_t, host_backend_t>, skylake,
                    nk_dots_pack_size_f32_skylake, nk_dots_packed_shape_f32_skylake, nk_dots_pack_f32_skylake);
    run_kernel_case("nk_dots_packed_bf16_skylake", test_dots_packed<bf16_t>, skylake, nk_dots_pack_size_bf16_skylake,
                    nk_dots_pack_bf16_skylake, nk_dots_packed_bf16_skylake);
    run_kernel_case("nk_dots_pack_bf16_skylake", test_dots_pack_layout<bf16_t, host_backend_t>, skylake,
                    nk_dots_pack_size_bf16_skylake, nk_dots_packed_shape_bf16_skylake, nk_dots_pack_bf16_skylake);
    run_kernel_case("nk_dots_packed_f16_skylake", test_dots_packed<f16_t>, skylake, nk_dots_pack_size_f16_skylake,
                    nk_dots_pack_f16_skylake, nk_dots_packed_f16_skylake);
    run_kernel_case("nk_dots_pack_f16_skylake", test_dots_pack_layout<f16_t, host_backend_t>, skylake,
                    nk_dots_pack_size_f16_skylake, nk_dots_packed_shape_f16_skylake, nk_dots_pack_f16_skylake);
    run_kernel_case("nk_dots_packed_e5m2_skylake", test_dots_packed<e5m2_t>, skylake, nk_dots_pack_size_e5m2_skylake,
                    nk_dots_pack_e5m2_skylake, nk_dots_packed_e5m2_skylake);
    run_kernel_case("nk_dots_pack_e5m2_skylake", test_dots_pack_layout<e5m2_t, host_backend_t>, skylake,
                    nk_dots_pack_size_e5m2_skylake, nk_dots_packed_shape_e5m2_skylake, nk_dots_pack_e5m2_skylake);
    run_kernel_case("nk_dots_packed_e4m3_skylake", test_dots_packed<e4m3_t>, skylake, nk_dots_pack_size_e4m3_skylake,
                    nk_dots_pack_e4m3_skylake, nk_dots_packed_e4m3_skylake);
    run_kernel_case("nk_dots_pack_e4m3_skylake", test_dots_pack_layout<e4m3_t, host_backend_t>, skylake,
                    nk_dots_pack_size_e4m3_skylake, nk_dots_packed_shape_e4m3_skylake, nk_dots_pack_e4m3_skylake);
    run_kernel_case("nk_dots_packed_e3m2_skylake", test_dots_packed<e3m2_t>, skylake, nk_dots_pack_size_e3m2_skylake,
                    nk_dots_pack_e3m2_skylake, nk_dots_packed_e3m2_skylake);
    run_kernel_case("nk_dots_pack_e3m2_skylake", test_dots_pack_layout<e3m2_t, host_backend_t>, skylake,
                    nk_dots_pack_size_e3m2_skylake, nk_dots_packed_shape_e3m2_skylake, nk_dots_pack_e3m2_skylake);
    run_kernel_case("nk_dots_packed_e2m3_skylake", test_dots_packed<e2m3_t>, skylake, nk_dots_pack_size_e2m3_skylake,
                    nk_dots_pack_e2m3_skylake, nk_dots_packed_e2m3_skylake);
    run_kernel_case("nk_dots_pack_e2m3_skylake", test_dots_pack_layout<e2m3_t, host_backend_t>, skylake,
                    nk_dots_pack_size_e2m3_skylake, nk_dots_packed_shape_e2m3_skylake, nk_dots_pack_e2m3_skylake);
    run_kernel_case("nk_dots_packed_e2m1_skylake", test_dots_packed<e2m1x2_t>, skylake, nk_dots_pack_size_e2m1_skylake,
                    nk_dots_pack_e2m1_skylake, nk_dots_packed_e2m1_skylake);
    run_kernel_case("nk_dots_packed_nvfp4_skylake", test_dots_packed<nvfp4_t>, skylake, nk_dots_pack_size_nvfp4_skylake,
                    nk_dots_pack_nvfp4_skylake, nk_dots_packed_nvfp4_skylake);
    run_kernel_case("nk_dots_packed_mxfp4_skylake", test_dots_packed<mxfp4_t>, skylake, nk_dots_pack_size_mxfp4_skylake,
                    nk_dots_pack_mxfp4_skylake, nk_dots_packed_mxfp4_skylake);
    run_kernel_case("nk_dots_packed_mxfp6e2m3_skylake", test_dots_packed<mxfp6e2m3_t>, skylake,
                    nk_dots_pack_size_mxfp6e2m3_skylake, nk_dots_pack_mxfp6e2m3_skylake,
                    nk_dots_packed_mxfp6e2m3_skylake);
    run_kernel_case("nk_dots_packed_mxfp6e3m2_skylake", test_dots_packed<mxfp6e3m2_t>, skylake,
                    nk_dots_pack_size_mxfp6e3m2_skylake, nk_dots_pack_mxfp6e3m2_skylake,
                    nk_dots_packed_mxfp6e3m2_skylake);
    run_kernel_case("nk_dots_packed_mxfp8e4m3_skylake", test_dots_packed<mxfp8e4m3_t>, skylake,
                    nk_dots_pack_size_mxfp8e4m3_skylake, nk_dots_pack_mxfp8e4m3_skylake,
                    nk_dots_packed_mxfp8e4m3_skylake);
    run_kernel_case("nk_dots_packed_mxfp8e5m2_skylake", test_dots_packed<mxfp8e5m2_t>, skylake,
                    nk_dots_pack_size_mxfp8e5m2_skylake, nk_dots_pack_mxfp8e5m2_skylake,
                    nk_dots_packed_mxfp8e5m2_skylake);
    run_kernel_case("nk_dots_pack_e2m1_skylake", test_dots_pack_layout<e2m1x2_t, host_backend_t>, skylake,
                    nk_dots_pack_size_e2m1_skylake, nk_dots_packed_shape_e2m1_skylake, nk_dots_pack_e2m1_skylake);
    run_kernel_case("nk_dots_pack_nvfp4_skylake", test_dots_pack_layout<nvfp4_t, host_backend_t>, skylake,
                    nk_dots_pack_size_nvfp4_skylake, nk_dots_packed_shape_nvfp4_skylake, nk_dots_pack_nvfp4_skylake);
    run_kernel_case("nk_dots_pack_mxfp4_skylake", test_dots_pack_layout<mxfp4_t, host_backend_t>, skylake,
                    nk_dots_pack_size_mxfp4_skylake, nk_dots_packed_shape_mxfp4_skylake, nk_dots_pack_mxfp4_skylake);
    run_kernel_case("nk_dots_pack_mxfp6e2m3_skylake", test_dots_pack_layout<mxfp6e2m3_t, host_backend_t>, skylake,
                    nk_dots_pack_size_mxfp6e2m3_skylake, nk_dots_packed_shape_mxfp6e2m3_skylake,
                    nk_dots_pack_mxfp6e2m3_skylake);
    run_kernel_case("nk_dots_pack_mxfp6e3m2_skylake", test_dots_pack_layout<mxfp6e3m2_t, host_backend_t>, skylake,
                    nk_dots_pack_size_mxfp6e3m2_skylake, nk_dots_packed_shape_mxfp6e3m2_skylake,
                    nk_dots_pack_mxfp6e3m2_skylake);
    run_kernel_case("nk_dots_pack_mxfp8e4m3_skylake", test_dots_pack_layout<mxfp8e4m3_t, host_backend_t>, skylake,
                    nk_dots_pack_size_mxfp8e4m3_skylake, nk_dots_packed_shape_mxfp8e4m3_skylake,
                    nk_dots_pack_mxfp8e4m3_skylake);
    run_kernel_case("nk_dots_pack_mxfp8e5m2_skylake", test_dots_pack_layout<mxfp8e5m2_t, host_backend_t>, skylake,
                    nk_dots_pack_size_mxfp8e5m2_skylake, nk_dots_packed_shape_mxfp8e5m2_skylake,
                    nk_dots_pack_mxfp8e5m2_skylake);

    run_kernel_case("nk_dots_symmetric_f64_skylake", test_dots_symmetric<f64_t>, skylake,
                    nk_dots_symmetric_f64_skylake);
    run_kernel_case("nk_dots_symmetric_f32_skylake", test_dots_symmetric<f32_t>, skylake,
                    nk_dots_symmetric_f32_skylake);
    run_kernel_case("nk_dots_symmetric_bf16_skylake", test_dots_symmetric<bf16_t>, skylake,
                    nk_dots_symmetric_bf16_skylake);
    run_kernel_case("nk_dots_symmetric_f16_skylake", test_dots_symmetric<f16_t>, skylake,
                    nk_dots_symmetric_f16_skylake);
    run_kernel_case("nk_dots_symmetric_e5m2_skylake", test_dots_symmetric<e5m2_t>, skylake,
                    nk_dots_symmetric_e5m2_skylake);
    run_kernel_case("nk_dots_symmetric_e4m3_skylake", test_dots_symmetric<e4m3_t>, skylake,
                    nk_dots_symmetric_e4m3_skylake);
    run_kernel_case("nk_dots_symmetric_e3m2_skylake", test_dots_symmetric<e3m2_t>, skylake,
                    nk_dots_symmetric_e3m2_skylake);
    run_kernel_case("nk_dots_symmetric_e2m3_skylake", test_dots_symmetric<e2m3_t>, skylake,
                    nk_dots_symmetric_e2m3_skylake);
    run_kernel_case("nk_dots_symmetric_e2m1_skylake", test_dots_symmetric<e2m1x2_t>, skylake,
                    nk_dots_symmetric_e2m1_skylake);
    run_kernel_case("nk_dots_symmetric_nvfp4_skylake", test_dots_symmetric<nvfp4_t>, skylake,
                    nk_dots_symmetric_nvfp4_skylake);
    run_kernel_case("nk_dots_symmetric_mxfp4_skylake", test_dots_symmetric<mxfp4_t>, skylake,
                    nk_dots_symmetric_mxfp4_skylake);
    run_kernel_case("nk_dots_symmetric_mxfp6e2m3_skylake", test_dots_symmetric<mxfp6e2m3_t>, skylake,
                    nk_dots_symmetric_mxfp6e2m3_skylake);
    run_kernel_case("nk_dots_symmetric_mxfp6e3m2_skylake", test_dots_symmetric<mxfp6e3m2_t>, skylake,
                    nk_dots_symmetric_mxfp6e3m2_skylake);
    run_kernel_case("nk_dots_symmetric_mxfp8e4m3_skylake", test_dots_symmetric<mxfp8e4m3_t>, skylake,
                    nk_dots_symmetric_mxfp8e4m3_skylake);
    run_kernel_case("nk_dots_symmetric_mxfp8e5m2_skylake", test_dots_symmetric<mxfp8e5m2_t>, skylake,
                    nk_dots_symmetric_mxfp8e5m2_skylake);

    run_kernel_case("nk_angulars_packed_f64_skylake", test_angulars_packed<f64_t>, skylake,
                    nk_dots_pack_size_f64_skylake, nk_dots_pack_f64_skylake, nk_angulars_packed_f64_skylake);
    run_kernel_case("nk_angulars_packed_f32_skylake", test_angulars_packed<f32_t>, skylake,
                    nk_dots_pack_size_f32_skylake, nk_dots_pack_f32_skylake, nk_angulars_packed_f32_skylake);
    run_kernel_case("nk_angulars_packed_bf16_skylake", test_angulars_packed<bf16_t>, skylake,
                    nk_dots_pack_size_bf16_skylake, nk_dots_pack_bf16_skylake, nk_angulars_packed_bf16_skylake);
    run_kernel_case("nk_angulars_packed_f16_skylake", test_angulars_packed<f16_t>, skylake,
                    nk_dots_pack_size_f16_skylake, nk_dots_pack_f16_skylake, nk_angulars_packed_f16_skylake);
    run_kernel_case("nk_angulars_packed_e5m2_skylake", test_angulars_packed<e5m2_t>, skylake,
                    nk_dots_pack_size_e5m2_skylake, nk_dots_pack_e5m2_skylake, nk_angulars_packed_e5m2_skylake);
    run_kernel_case("nk_angulars_packed_e4m3_skylake", test_angulars_packed<e4m3_t>, skylake,
                    nk_dots_pack_size_e4m3_skylake, nk_dots_pack_e4m3_skylake, nk_angulars_packed_e4m3_skylake);
    run_kernel_case("nk_angulars_packed_e3m2_skylake", test_angulars_packed<e3m2_t>, skylake,
                    nk_dots_pack_size_e3m2_skylake, nk_dots_pack_e3m2_skylake, nk_angulars_packed_e3m2_skylake);
    run_kernel_case("nk_angulars_packed_e2m3_skylake", test_angulars_packed<e2m3_t>, skylake,
                    nk_dots_pack_size_e2m3_skylake, nk_dots_pack_e2m3_skylake, nk_angulars_packed_e2m3_skylake);
    run_kernel_case("nk_angulars_packed_e2m1_skylake", test_angulars_packed<e2m1x2_t>, skylake,
                    nk_dots_pack_size_e2m1_skylake, nk_dots_pack_e2m1_skylake, nk_angulars_packed_e2m1_skylake);
    run_kernel_case("nk_angulars_packed_nvfp4_skylake", test_angulars_packed<nvfp4_t>, skylake,
                    nk_dots_pack_size_nvfp4_skylake, nk_dots_pack_nvfp4_skylake, nk_angulars_packed_nvfp4_skylake);
    run_kernel_case("nk_angulars_packed_mxfp4_skylake", test_angulars_packed<mxfp4_t>, skylake,
                    nk_dots_pack_size_mxfp4_skylake, nk_dots_pack_mxfp4_skylake, nk_angulars_packed_mxfp4_skylake);
    run_kernel_case("nk_angulars_packed_mxfp6e2m3_skylake", test_angulars_packed<mxfp6e2m3_t>, skylake,
                    nk_dots_pack_size_mxfp6e2m3_skylake, nk_dots_pack_mxfp6e2m3_skylake,
                    nk_angulars_packed_mxfp6e2m3_skylake);
    run_kernel_case("nk_angulars_packed_mxfp6e3m2_skylake", test_angulars_packed<mxfp6e3m2_t>, skylake,
                    nk_dots_pack_size_mxfp6e3m2_skylake, nk_dots_pack_mxfp6e3m2_skylake,
                    nk_angulars_packed_mxfp6e3m2_skylake);
    run_kernel_case("nk_angulars_packed_mxfp8e4m3_skylake", test_angulars_packed<mxfp8e4m3_t>, skylake,
                    nk_dots_pack_size_mxfp8e4m3_skylake, nk_dots_pack_mxfp8e4m3_skylake,
                    nk_angulars_packed_mxfp8e4m3_skylake);
    run_kernel_case("nk_angulars_packed_mxfp8e5m2_skylake", test_angulars_packed<mxfp8e5m2_t>, skylake,
                    nk_dots_pack_size_mxfp8e5m2_skylake, nk_dots_pack_mxfp8e5m2_skylake,
                    nk_angulars_packed_mxfp8e5m2_skylake);

    run_kernel_case("nk_angulars_symmetric_f64_skylake", test_angulars_symmetric<f64_t>, skylake,
                    nk_angulars_symmetric_f64_skylake);
    run_kernel_case("nk_angulars_symmetric_f32_skylake", test_angulars_symmetric<f32_t>, skylake,
                    nk_angulars_symmetric_f32_skylake);
    run_kernel_case("nk_angulars_symmetric_bf16_skylake", test_angulars_symmetric<bf16_t>, skylake,
                    nk_angulars_symmetric_bf16_skylake);
    run_kernel_case("nk_angulars_symmetric_f16_skylake", test_angulars_symmetric<f16_t>, skylake,
                    nk_angulars_symmetric_f16_skylake);
    run_kernel_case("nk_angulars_symmetric_e5m2_skylake", test_angulars_symmetric<e5m2_t>, skylake,
                    nk_angulars_symmetric_e5m2_skylake);
    run_kernel_case("nk_angulars_symmetric_e4m3_skylake", test_angulars_symmetric<e4m3_t>, skylake,
                    nk_angulars_symmetric_e4m3_skylake);
    run_kernel_case("nk_angulars_symmetric_e3m2_skylake", test_angulars_symmetric<e3m2_t>, skylake,
                    nk_angulars_symmetric_e3m2_skylake);
    run_kernel_case("nk_angulars_symmetric_e2m3_skylake", test_angulars_symmetric<e2m3_t>, skylake,
                    nk_angulars_symmetric_e2m3_skylake);
    run_kernel_case("nk_angulars_symmetric_e2m1_skylake", test_angulars_symmetric<e2m1x2_t>, skylake,
                    nk_angulars_symmetric_e2m1_skylake);
    run_kernel_case("nk_angulars_symmetric_nvfp4_skylake", test_angulars_symmetric<nvfp4_t>, skylake,
                    nk_angulars_symmetric_nvfp4_skylake);
    run_kernel_case("nk_angulars_symmetric_mxfp4_skylake", test_angulars_symmetric<mxfp4_t>, skylake,
                    nk_angulars_symmetric_mxfp4_skylake);
    run_kernel_case("nk_angulars_symmetric_mxfp6e2m3_skylake", test_angulars_symmetric<mxfp6e2m3_t>, skylake,
                    nk_angulars_symmetric_mxfp6e2m3_skylake);
    run_kernel_case("nk_angulars_symmetric_mxfp6e3m2_skylake", test_angulars_symmetric<mxfp6e3m2_t>, skylake,
                    nk_angulars_symmetric_mxfp6e3m2_skylake);
    run_kernel_case("nk_angulars_symmetric_mxfp8e4m3_skylake", test_angulars_symmetric<mxfp8e4m3_t>, skylake,
                    nk_angulars_symmetric_mxfp8e4m3_skylake);
    run_kernel_case("nk_angulars_symmetric_mxfp8e5m2_skylake", test_angulars_symmetric<mxfp8e5m2_t>, skylake,
                    nk_angulars_symmetric_mxfp8e5m2_skylake);

    run_kernel_case("nk_euclideans_packed_f64_skylake", test_euclideans_packed<f64_t>, skylake,
                    nk_dots_pack_size_f64_skylake, nk_dots_pack_f64_skylake, nk_euclideans_packed_f64_skylake);
    run_kernel_case("nk_euclideans_packed_f32_skylake", test_euclideans_packed<f32_t>, skylake,
                    nk_dots_pack_size_f32_skylake, nk_dots_pack_f32_skylake, nk_euclideans_packed_f32_skylake);
    run_kernel_case("nk_euclideans_packed_bf16_skylake", test_euclideans_packed<bf16_t>, skylake,
                    nk_dots_pack_size_bf16_skylake, nk_dots_pack_bf16_skylake, nk_euclideans_packed_bf16_skylake);
    run_kernel_case("nk_euclideans_packed_f16_skylake", test_euclideans_packed<f16_t>, skylake,
                    nk_dots_pack_size_f16_skylake, nk_dots_pack_f16_skylake, nk_euclideans_packed_f16_skylake);
    run_kernel_case("nk_euclideans_packed_e5m2_skylake", test_euclideans_packed<e5m2_t>, skylake,
                    nk_dots_pack_size_e5m2_skylake, nk_dots_pack_e5m2_skylake, nk_euclideans_packed_e5m2_skylake);
    run_kernel_case("nk_euclideans_packed_e4m3_skylake", test_euclideans_packed<e4m3_t>, skylake,
                    nk_dots_pack_size_e4m3_skylake, nk_dots_pack_e4m3_skylake, nk_euclideans_packed_e4m3_skylake);
    run_kernel_case("nk_euclideans_packed_e3m2_skylake", test_euclideans_packed<e3m2_t>, skylake,
                    nk_dots_pack_size_e3m2_skylake, nk_dots_pack_e3m2_skylake, nk_euclideans_packed_e3m2_skylake);
    run_kernel_case("nk_euclideans_packed_e2m3_skylake", test_euclideans_packed<e2m3_t>, skylake,
                    nk_dots_pack_size_e2m3_skylake, nk_dots_pack_e2m3_skylake, nk_euclideans_packed_e2m3_skylake);
    run_kernel_case("nk_euclideans_packed_e2m1_skylake", test_euclideans_packed<e2m1x2_t>, skylake,
                    nk_dots_pack_size_e2m1_skylake, nk_dots_pack_e2m1_skylake, nk_euclideans_packed_e2m1_skylake);
    run_kernel_case("nk_euclideans_packed_nvfp4_skylake", test_euclideans_packed<nvfp4_t>, skylake,
                    nk_dots_pack_size_nvfp4_skylake, nk_dots_pack_nvfp4_skylake, nk_euclideans_packed_nvfp4_skylake);
    run_kernel_case("nk_euclideans_packed_mxfp4_skylake", test_euclideans_packed<mxfp4_t>, skylake,
                    nk_dots_pack_size_mxfp4_skylake, nk_dots_pack_mxfp4_skylake, nk_euclideans_packed_mxfp4_skylake);
    run_kernel_case("nk_euclideans_packed_mxfp6e2m3_skylake", test_euclideans_packed<mxfp6e2m3_t>, skylake,
                    nk_dots_pack_size_mxfp6e2m3_skylake, nk_dots_pack_mxfp6e2m3_skylake,
                    nk_euclideans_packed_mxfp6e2m3_skylake);
    run_kernel_case("nk_euclideans_packed_mxfp6e3m2_skylake", test_euclideans_packed<mxfp6e3m2_t>, skylake,
                    nk_dots_pack_size_mxfp6e3m2_skylake, nk_dots_pack_mxfp6e3m2_skylake,
                    nk_euclideans_packed_mxfp6e3m2_skylake);
    run_kernel_case("nk_euclideans_packed_mxfp8e4m3_skylake", test_euclideans_packed<mxfp8e4m3_t>, skylake,
                    nk_dots_pack_size_mxfp8e4m3_skylake, nk_dots_pack_mxfp8e4m3_skylake,
                    nk_euclideans_packed_mxfp8e4m3_skylake);
    run_kernel_case("nk_euclideans_packed_mxfp8e5m2_skylake", test_euclideans_packed<mxfp8e5m2_t>, skylake,
                    nk_dots_pack_size_mxfp8e5m2_skylake, nk_dots_pack_mxfp8e5m2_skylake,
                    nk_euclideans_packed_mxfp8e5m2_skylake);

    run_kernel_case("nk_euclideans_symmetric_f64_skylake", test_euclideans_symmetric<f64_t>, skylake,
                    nk_euclideans_symmetric_f64_skylake);
    run_kernel_case("nk_euclideans_symmetric_f32_skylake", test_euclideans_symmetric<f32_t>, skylake,
                    nk_euclideans_symmetric_f32_skylake);
    run_kernel_case("nk_euclideans_symmetric_bf16_skylake", test_euclideans_symmetric<bf16_t>, skylake,
                    nk_euclideans_symmetric_bf16_skylake);
    run_kernel_case("nk_euclideans_symmetric_f16_skylake", test_euclideans_symmetric<f16_t>, skylake,
                    nk_euclideans_symmetric_f16_skylake);
    run_kernel_case("nk_euclideans_symmetric_e5m2_skylake", test_euclideans_symmetric<e5m2_t>, skylake,
                    nk_euclideans_symmetric_e5m2_skylake);
    run_kernel_case("nk_euclideans_symmetric_e4m3_skylake", test_euclideans_symmetric<e4m3_t>, skylake,
                    nk_euclideans_symmetric_e4m3_skylake);
    run_kernel_case("nk_euclideans_symmetric_e3m2_skylake", test_euclideans_symmetric<e3m2_t>, skylake,
                    nk_euclideans_symmetric_e3m2_skylake);
    run_kernel_case("nk_euclideans_symmetric_e2m3_skylake", test_euclideans_symmetric<e2m3_t>, skylake,
                    nk_euclideans_symmetric_e2m3_skylake);
    run_kernel_case("nk_euclideans_symmetric_e2m1_skylake", test_euclideans_symmetric<e2m1x2_t>, skylake,
                    nk_euclideans_symmetric_e2m1_skylake);
    run_kernel_case("nk_euclideans_symmetric_nvfp4_skylake", test_euclideans_symmetric<nvfp4_t>, skylake,
                    nk_euclideans_symmetric_nvfp4_skylake);
    run_kernel_case("nk_euclideans_symmetric_mxfp4_skylake", test_euclideans_symmetric<mxfp4_t>, skylake,
                    nk_euclideans_symmetric_mxfp4_skylake);
    run_kernel_case("nk_euclideans_symmetric_mxfp6e2m3_skylake", test_euclideans_symmetric<mxfp6e2m3_t>, skylake,
                    nk_euclideans_symmetric_mxfp6e2m3_skylake);
    run_kernel_case("nk_euclideans_symmetric_mxfp6e3m2_skylake", test_euclideans_symmetric<mxfp6e3m2_t>, skylake,
                    nk_euclideans_symmetric_mxfp6e3m2_skylake);
    run_kernel_case("nk_euclideans_symmetric_mxfp8e4m3_skylake", test_euclideans_symmetric<mxfp8e4m3_t>, skylake,
                    nk_euclideans_symmetric_mxfp8e4m3_skylake);
    run_kernel_case("nk_euclideans_symmetric_mxfp8e5m2_skylake", test_euclideans_symmetric<mxfp8e5m2_t>, skylake,
                    nk_euclideans_symmetric_mxfp8e5m2_skylake);

    run_kernel_case("nk_attention_packed_bf16_skylake", test_attention_packed<bf16_t>, skylake,
                    nk_attention_pack_size_bf16_skylake, nk_attention_pack_bf16_skylake,
                    nk_attention_packed_bf16_skylake);
    run_kernel_case("nk_attention_packed_e4m3_skylake", test_attention_packed<e4m3_t>, skylake,
                    nk_attention_pack_size_e4m3_skylake, nk_attention_pack_e4m3_skylake,
                    nk_attention_packed_e4m3_skylake);
    run_kernel_case("nk_attention_rope_f32_skylake", test_attention_rope<f32_t>, skylake,
                    nk_attention_rope_f32_skylake);
    run_kernel_case("nk_attention_rope_bf16_skylake", test_attention_rope<bf16_t>, skylake,
                    nk_attention_rope_bf16_skylake);
    run_kernel_case("nk_attention_rope_e4m3_skylake", test_attention_rope<e4m3_t>, skylake,
                    nk_attention_rope_e4m3_skylake);
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_ICELAKE
    auto const icelake = nk_cap_icelake_k;
    run_kernel_case("nk_dots_packed_e2m3_icelake", test_dots_packed<e2m3_t>, icelake, nk_dots_pack_size_e2m3_icelake,
                    nk_dots_pack_e2m3_icelake, nk_dots_packed_e2m3_icelake);
    run_kernel_case("nk_dots_pack_e2m3_icelake", test_dots_pack_layout<e2m3_t, host_backend_t>, icelake,
                    nk_dots_pack_size_e2m3_icelake, nk_dots_packed_shape_e2m3_icelake, nk_dots_pack_e2m3_icelake);
    run_kernel_case("nk_dots_packed_e2m1_icelake", test_dots_packed<e2m1x2_t>, icelake, nk_dots_pack_size_e2m1_icelake,
                    nk_dots_pack_e2m1_icelake, nk_dots_packed_e2m1_icelake);
    run_kernel_case("nk_dots_pack_e2m1_icelake", test_dots_pack_layout<e2m1x2_t, host_backend_t>, icelake,
                    nk_dots_pack_size_e2m1_icelake, nk_dots_packed_shape_e2m1_icelake, nk_dots_pack_e2m1_icelake);
    run_kernel_case("nk_dots_symmetric_e2m3_icelake", test_dots_symmetric<e2m3_t>, icelake,
                    nk_dots_symmetric_e2m3_icelake);
    run_kernel_case("nk_dots_symmetric_e2m1_icelake", test_dots_symmetric<e2m1x2_t>, icelake,
                    nk_dots_symmetric_e2m1_icelake);
    run_kernel_case("nk_angulars_packed_e2m3_icelake", test_angulars_packed<e2m3_t>, icelake,
                    nk_dots_pack_size_e2m3_icelake, nk_dots_pack_e2m3_icelake, nk_angulars_packed_e2m3_icelake);
    run_kernel_case("nk_angulars_packed_e2m1_icelake", test_angulars_packed<e2m1x2_t>, icelake,
                    nk_dots_pack_size_e2m1_icelake, nk_dots_pack_e2m1_icelake, nk_angulars_packed_e2m1_icelake);
    run_kernel_case("nk_angulars_symmetric_e2m3_icelake", test_angulars_symmetric<e2m3_t>, icelake,
                    nk_angulars_symmetric_e2m3_icelake);
    run_kernel_case("nk_angulars_symmetric_e2m1_icelake", test_angulars_symmetric<e2m1x2_t>, icelake,
                    nk_angulars_symmetric_e2m1_icelake);
    run_kernel_case("nk_euclideans_packed_e2m3_icelake", test_euclideans_packed<e2m3_t>, icelake,
                    nk_dots_pack_size_e2m3_icelake, nk_dots_pack_e2m3_icelake, nk_euclideans_packed_e2m3_icelake);
    run_kernel_case("nk_euclideans_packed_e2m1_icelake", test_euclideans_packed<e2m1x2_t>, icelake,
                    nk_dots_pack_size_e2m1_icelake, nk_dots_pack_e2m1_icelake, nk_euclideans_packed_e2m1_icelake);
    run_kernel_case("nk_euclideans_symmetric_e2m3_icelake", test_euclideans_symmetric<e2m3_t>, icelake,
                    nk_euclideans_symmetric_e2m3_icelake);
    run_kernel_case("nk_euclideans_symmetric_e2m1_icelake", test_euclideans_symmetric<e2m1x2_t>, icelake,
                    nk_euclideans_symmetric_e2m1_icelake);
    run_kernel_case("nk_dots_packed_i8_icelake", test_dots_packed<i8_t>, icelake, nk_dots_pack_size_i8_icelake,
                    nk_dots_pack_i8_icelake, nk_dots_packed_i8_icelake);
    run_kernel_case("nk_dots_pack_i8_icelake", test_dots_pack_layout<i8_t, host_backend_t>, icelake,
                    nk_dots_pack_size_i8_icelake, nk_dots_packed_shape_i8_icelake, nk_dots_pack_i8_icelake);
    run_kernel_case("nk_dots_packed_i4_icelake", test_dots_packed<i4x2_t>, icelake, nk_dots_pack_size_i4_icelake,
                    nk_dots_pack_i4_icelake, nk_dots_packed_i4_icelake);
    run_kernel_case("nk_dots_pack_i4_icelake", test_dots_pack_layout<i4x2_t, host_backend_t>, icelake,
                    nk_dots_pack_size_i4_icelake, nk_dots_packed_shape_i4_icelake, nk_dots_pack_i4_icelake);
    run_kernel_case("nk_dots_packed_u8_icelake", test_dots_packed<u8_t>, icelake, nk_dots_pack_size_u8_icelake,
                    nk_dots_pack_u8_icelake, nk_dots_packed_u8_icelake);
    run_kernel_case("nk_dots_pack_u8_icelake", test_dots_pack_layout<u8_t, host_backend_t>, icelake,
                    nk_dots_pack_size_u8_icelake, nk_dots_packed_shape_u8_icelake, nk_dots_pack_u8_icelake);
    run_kernel_case("nk_dots_packed_u4_icelake", test_dots_packed<u4x2_t>, icelake, nk_dots_pack_size_u4_icelake,
                    nk_dots_pack_u4_icelake, nk_dots_packed_u4_icelake);
    run_kernel_case("nk_dots_pack_u4_icelake", test_dots_pack_layout<u4x2_t, host_backend_t>, icelake,
                    nk_dots_pack_size_u4_icelake, nk_dots_packed_shape_u4_icelake, nk_dots_pack_u4_icelake);
    run_kernel_case("nk_dots_packed_u1_icelake", test_dots_packed<u1x8_t>, icelake, nk_dots_pack_size_u1_icelake,
                    nk_dots_pack_u1_icelake, nk_dots_packed_u1_icelake);

    run_kernel_case("nk_dots_symmetric_i8_icelake", test_dots_symmetric<i8_t>, icelake, nk_dots_symmetric_i8_icelake);
    run_kernel_case("nk_dots_symmetric_i4_icelake", test_dots_symmetric<i4x2_t>, icelake, nk_dots_symmetric_i4_icelake);
    run_kernel_case("nk_dots_symmetric_u8_icelake", test_dots_symmetric<u8_t>, icelake, nk_dots_symmetric_u8_icelake);
    run_kernel_case("nk_dots_symmetric_u4_icelake", test_dots_symmetric<u4x2_t>, icelake, nk_dots_symmetric_u4_icelake);
    run_kernel_case("nk_dots_symmetric_u1_icelake", test_dots_symmetric<u1x8_t>, icelake, nk_dots_symmetric_u1_icelake);

    run_kernel_case("nk_hammings_packed_u1_icelake", test_hammings_packed<u1x8_t>, icelake,
                    nk_dots_pack_size_u1_icelake, nk_dots_pack_u1_icelake, nk_hammings_packed_u1_icelake);
    run_kernel_case("nk_hammings_symmetric_u1_icelake", test_hammings_symmetric<u1x8_t>, icelake,
                    nk_hammings_symmetric_u1_icelake);

    run_kernel_case("nk_jaccards_packed_u1_icelake", test_jaccards_packed<u1x8_t>, icelake,
                    nk_dots_pack_size_u1_icelake, nk_dots_pack_u1_icelake, nk_jaccards_packed_u1_icelake);
    run_kernel_case("nk_jaccards_symmetric_u1_icelake", test_jaccards_symmetric<u1x8_t>, icelake,
                    nk_jaccards_symmetric_u1_icelake);

    run_kernel_case("nk_angulars_packed_i8_icelake", test_angulars_packed<i8_t>, icelake, nk_dots_pack_size_i8_icelake,
                    nk_dots_pack_i8_icelake, nk_angulars_packed_i8_icelake);
    run_kernel_case("nk_angulars_packed_i4_icelake", test_angulars_packed<i4x2_t>, icelake,
                    nk_dots_pack_size_i4_icelake, nk_dots_pack_i4_icelake, nk_angulars_packed_i4_icelake);
    run_kernel_case("nk_angulars_packed_u8_icelake", test_angulars_packed<u8_t>, icelake, nk_dots_pack_size_u8_icelake,
                    nk_dots_pack_u8_icelake, nk_angulars_packed_u8_icelake);
    run_kernel_case("nk_angulars_packed_u4_icelake", test_angulars_packed<u4x2_t>, icelake,
                    nk_dots_pack_size_u4_icelake, nk_dots_pack_u4_icelake, nk_angulars_packed_u4_icelake);

    run_kernel_case("nk_angulars_symmetric_i8_icelake", test_angulars_symmetric<i8_t>, icelake,
                    nk_angulars_symmetric_i8_icelake);
    run_kernel_case("nk_angulars_symmetric_i4_icelake", test_angulars_symmetric<i4x2_t>, icelake,
                    nk_angulars_symmetric_i4_icelake);
    run_kernel_case("nk_angulars_symmetric_u8_icelake", test_angulars_symmetric<u8_t>, icelake,
                    nk_angulars_symmetric_u8_icelake);
    run_kernel_case("nk_angulars_symmetric_u4_icelake", test_angulars_symmetric<u4x2_t>, icelake,
                    nk_angulars_symmetric_u4_icelake);

    run_kernel_case("nk_euclideans_packed_i8_icelake", test_euclideans_packed<i8_t>, icelake,
                    nk_dots_pack_size_i8_icelake, nk_dots_pack_i8_icelake, nk_euclideans_packed_i8_icelake);
    run_kernel_case("nk_euclideans_packed_i4_icelake", test_euclideans_packed<i4x2_t>, icelake,
                    nk_dots_pack_size_i4_icelake, nk_dots_pack_i4_icelake, nk_euclideans_packed_i4_icelake);
    run_kernel_case("nk_euclideans_packed_u8_icelake", test_euclideans_packed<u8_t>, icelake,
                    nk_dots_pack_size_u8_icelake, nk_dots_pack_u8_icelake, nk_euclideans_packed_u8_icelake);
    run_kernel_case("nk_euclideans_packed_u4_icelake", test_euclideans_packed<u4x2_t>, icelake,
                    nk_dots_pack_size_u4_icelake, nk_dots_pack_u4_icelake, nk_euclideans_packed_u4_icelake);

    run_kernel_case("nk_euclideans_symmetric_i8_icelake", test_euclideans_symmetric<i8_t>, icelake,
                    nk_euclideans_symmetric_i8_icelake);
    run_kernel_case("nk_euclideans_symmetric_i4_icelake", test_euclideans_symmetric<i4x2_t>, icelake,
                    nk_euclideans_symmetric_i4_icelake);
    run_kernel_case("nk_euclideans_symmetric_u8_icelake", test_euclideans_symmetric<u8_t>, icelake,
                    nk_euclideans_symmetric_u8_icelake);
    run_kernel_case("nk_euclideans_symmetric_u4_icelake", test_euclideans_symmetric<u4x2_t>, icelake,
                    nk_euclideans_symmetric_u4_icelake);

    run_kernel_case("nk_attention_packed_i8_icelake", test_attention_packed<i8_t>, icelake,
                    nk_attention_pack_size_i8_icelake, nk_attention_pack_i8_icelake, nk_attention_packed_i8_icelake);
#endif // NUMKONG_TARGET_ICELAKE

#if NUMKONG_TARGET_GENOA
    auto const genoa = nk_cap_genoa_k;
    run_kernel_case("nk_dots_packed_bf16_genoa", test_dots_packed<bf16_t>, genoa, nk_dots_pack_size_bf16_genoa,
                    nk_dots_pack_bf16_genoa, nk_dots_packed_bf16_genoa);
    run_kernel_case("nk_dots_pack_bf16_genoa", test_dots_pack_layout<bf16_t, host_backend_t>, genoa,
                    nk_dots_pack_size_bf16_genoa, nk_dots_packed_shape_bf16_genoa, nk_dots_pack_bf16_genoa);
    run_kernel_case("nk_dots_packed_e5m2_genoa", test_dots_packed<e5m2_t>, genoa, nk_dots_pack_size_e5m2_genoa,
                    nk_dots_pack_e5m2_genoa, nk_dots_packed_e5m2_genoa);
    run_kernel_case("nk_dots_pack_e5m2_genoa", test_dots_pack_layout<e5m2_t, host_backend_t>, genoa,
                    nk_dots_pack_size_e5m2_genoa, nk_dots_packed_shape_e5m2_genoa, nk_dots_pack_e5m2_genoa);
    run_kernel_case("nk_dots_packed_e4m3_genoa", test_dots_packed<e4m3_t>, genoa, nk_dots_pack_size_e4m3_genoa,
                    nk_dots_pack_e4m3_genoa, nk_dots_packed_e4m3_genoa);
    run_kernel_case("nk_dots_pack_e4m3_genoa", test_dots_pack_layout<e4m3_t, host_backend_t>, genoa,
                    nk_dots_pack_size_e4m3_genoa, nk_dots_packed_shape_e4m3_genoa, nk_dots_pack_e4m3_genoa);
    run_kernel_case("nk_dots_symmetric_bf16_genoa", test_dots_symmetric<bf16_t>, genoa, nk_dots_symmetric_bf16_genoa);
    run_kernel_case("nk_dots_symmetric_e5m2_genoa", test_dots_symmetric<e5m2_t>, genoa, nk_dots_symmetric_e5m2_genoa);
    run_kernel_case("nk_dots_symmetric_e4m3_genoa", test_dots_symmetric<e4m3_t>, genoa, nk_dots_symmetric_e4m3_genoa);
    run_kernel_case("nk_angulars_packed_bf16_genoa", test_angulars_packed<bf16_t>, genoa, nk_dots_pack_size_bf16_genoa,
                    nk_dots_pack_bf16_genoa, nk_angulars_packed_bf16_genoa);
    run_kernel_case("nk_angulars_packed_e5m2_genoa", test_angulars_packed<e5m2_t>, genoa, nk_dots_pack_size_e5m2_genoa,
                    nk_dots_pack_e5m2_genoa, nk_angulars_packed_e5m2_genoa);
    run_kernel_case("nk_angulars_packed_e4m3_genoa", test_angulars_packed<e4m3_t>, genoa, nk_dots_pack_size_e4m3_genoa,
                    nk_dots_pack_e4m3_genoa, nk_angulars_packed_e4m3_genoa);
    run_kernel_case("nk_angulars_symmetric_bf16_genoa", test_angulars_symmetric<bf16_t>, genoa,
                    nk_angulars_symmetric_bf16_genoa);
    run_kernel_case("nk_angulars_symmetric_e5m2_genoa", test_angulars_symmetric<e5m2_t>, genoa,
                    nk_angulars_symmetric_e5m2_genoa);
    run_kernel_case("nk_angulars_symmetric_e4m3_genoa", test_angulars_symmetric<e4m3_t>, genoa,
                    nk_angulars_symmetric_e4m3_genoa);
    run_kernel_case("nk_euclideans_packed_bf16_genoa", test_euclideans_packed<bf16_t>, genoa,
                    nk_dots_pack_size_bf16_genoa, nk_dots_pack_bf16_genoa, nk_euclideans_packed_bf16_genoa);
    run_kernel_case("nk_euclideans_packed_e5m2_genoa", test_euclideans_packed<e5m2_t>, genoa,
                    nk_dots_pack_size_e5m2_genoa, nk_dots_pack_e5m2_genoa, nk_euclideans_packed_e5m2_genoa);
    run_kernel_case("nk_euclideans_packed_e4m3_genoa", test_euclideans_packed<e4m3_t>, genoa,
                    nk_dots_pack_size_e4m3_genoa, nk_dots_pack_e4m3_genoa, nk_euclideans_packed_e4m3_genoa);
    run_kernel_case("nk_euclideans_symmetric_bf16_genoa", test_euclideans_symmetric<bf16_t>, genoa,
                    nk_euclideans_symmetric_bf16_genoa);
    run_kernel_case("nk_euclideans_symmetric_e5m2_genoa", test_euclideans_symmetric<e5m2_t>, genoa,
                    nk_euclideans_symmetric_e5m2_genoa);
    run_kernel_case("nk_euclideans_symmetric_e4m3_genoa", test_euclideans_symmetric<e4m3_t>, genoa,
                    nk_euclideans_symmetric_e4m3_genoa);

    run_kernel_case("nk_attention_packed_bf16_genoa", test_attention_packed<bf16_t>, genoa,
                    nk_attention_pack_size_bf16_genoa, nk_attention_pack_bf16_genoa, nk_attention_packed_bf16_genoa);
    run_kernel_case("nk_attention_packed_e4m3_genoa", test_attention_packed<e4m3_t>, genoa,
                    nk_attention_pack_size_e4m3_genoa, nk_attention_pack_e4m3_genoa, nk_attention_packed_e4m3_genoa);
#endif // NUMKONG_TARGET_GENOA

#if NUMKONG_TARGET_DIAMOND
    auto const diamond = nk_cap_diamond_k;
    run_kernel_case("nk_dots_packed_e5m2_diamond", test_dots_packed<e5m2_t>, diamond, nk_dots_pack_size_e5m2_diamond,
                    nk_dots_pack_e5m2_diamond, nk_dots_packed_e5m2_diamond);
    run_kernel_case("nk_dots_pack_e5m2_diamond", test_dots_pack_layout<e5m2_t, host_backend_t>, diamond,
                    nk_dots_pack_size_e5m2_diamond, nk_dots_packed_shape_e5m2_diamond, nk_dots_pack_e5m2_diamond);
    run_kernel_case("nk_dots_packed_e4m3_diamond", test_dots_packed<e4m3_t>, diamond, nk_dots_pack_size_e4m3_diamond,
                    nk_dots_pack_e4m3_diamond, nk_dots_packed_e4m3_diamond);
    run_kernel_case("nk_dots_pack_e4m3_diamond", test_dots_pack_layout<e4m3_t, host_backend_t>, diamond,
                    nk_dots_pack_size_e4m3_diamond, nk_dots_packed_shape_e4m3_diamond, nk_dots_pack_e4m3_diamond);
    run_kernel_case("nk_dots_symmetric_e5m2_diamond", test_dots_symmetric<e5m2_t>, diamond,
                    nk_dots_symmetric_e5m2_diamond);
    run_kernel_case("nk_dots_symmetric_e4m3_diamond", test_dots_symmetric<e4m3_t>, diamond,
                    nk_dots_symmetric_e4m3_diamond);
    run_kernel_case("nk_angulars_packed_e5m2_diamond", test_angulars_packed<e5m2_t>, diamond,
                    nk_dots_pack_size_e5m2_diamond, nk_dots_pack_e5m2_diamond, nk_angulars_packed_e5m2_diamond);
    run_kernel_case("nk_angulars_packed_e4m3_diamond", test_angulars_packed<e4m3_t>, diamond,
                    nk_dots_pack_size_e4m3_diamond, nk_dots_pack_e4m3_diamond, nk_angulars_packed_e4m3_diamond);
    run_kernel_case("nk_angulars_symmetric_e5m2_diamond", test_angulars_symmetric<e5m2_t>, diamond,
                    nk_angulars_symmetric_e5m2_diamond);
    run_kernel_case("nk_angulars_symmetric_e4m3_diamond", test_angulars_symmetric<e4m3_t>, diamond,
                    nk_angulars_symmetric_e4m3_diamond);
    run_kernel_case("nk_euclideans_packed_e5m2_diamond", test_euclideans_packed<e5m2_t>, diamond,
                    nk_dots_pack_size_e5m2_diamond, nk_dots_pack_e5m2_diamond, nk_euclideans_packed_e5m2_diamond);
    run_kernel_case("nk_euclideans_packed_e4m3_diamond", test_euclideans_packed<e4m3_t>, diamond,
                    nk_dots_pack_size_e4m3_diamond, nk_dots_pack_e4m3_diamond, nk_euclideans_packed_e4m3_diamond);
    run_kernel_case("nk_euclideans_symmetric_e5m2_diamond", test_euclideans_symmetric<e5m2_t>, diamond,
                    nk_euclideans_symmetric_e5m2_diamond);
    run_kernel_case("nk_euclideans_symmetric_e4m3_diamond", test_euclideans_symmetric<e4m3_t>, diamond,
                    nk_euclideans_symmetric_e4m3_diamond);
#endif // NUMKONG_TARGET_DIAMOND

#if NUMKONG_TARGET_ALDER
    auto const alder = nk_cap_alder_k;
    run_kernel_case("nk_dots_packed_e2m3_alder", test_dots_packed<e2m3_t>, alder, nk_dots_pack_size_e2m3_alder,
                    nk_dots_pack_e2m3_alder, nk_dots_packed_e2m3_alder);
    run_kernel_case("nk_dots_pack_e2m3_alder", test_dots_pack_layout<e2m3_t, host_backend_t>, alder,
                    nk_dots_pack_size_e2m3_alder, nk_dots_packed_shape_e2m3_alder, nk_dots_pack_e2m3_alder);
    run_kernel_case("nk_dots_packed_e2m1_alder", test_dots_packed<e2m1x2_t>, alder, nk_dots_pack_size_e2m1_alder,
                    nk_dots_pack_e2m1_alder, nk_dots_packed_e2m1_alder);
    run_kernel_case("nk_dots_pack_e2m1_alder", test_dots_pack_layout<e2m1x2_t, host_backend_t>, alder,
                    nk_dots_pack_size_e2m1_alder, nk_dots_packed_shape_e2m1_alder, nk_dots_pack_e2m1_alder);
    run_kernel_case("nk_dots_packed_i8_alder", test_dots_packed<i8_t>, alder, nk_dots_pack_size_i8_alder,
                    nk_dots_pack_i8_alder, nk_dots_packed_i8_alder);
    run_kernel_case("nk_dots_pack_i8_alder", test_dots_pack_layout<i8_t, host_backend_t>, alder,
                    nk_dots_pack_size_i8_alder, nk_dots_packed_shape_i8_alder, nk_dots_pack_i8_alder);
    run_kernel_case("nk_dots_packed_u8_alder", test_dots_packed<u8_t>, alder, nk_dots_pack_size_u8_alder,
                    nk_dots_pack_u8_alder, nk_dots_packed_u8_alder);
    run_kernel_case("nk_dots_pack_u8_alder", test_dots_pack_layout<u8_t, host_backend_t>, alder,
                    nk_dots_pack_size_u8_alder, nk_dots_packed_shape_u8_alder, nk_dots_pack_u8_alder);

    run_kernel_case("nk_dots_symmetric_e2m3_alder", test_dots_symmetric<e2m3_t>, alder, nk_dots_symmetric_e2m3_alder);
    run_kernel_case("nk_dots_symmetric_e2m1_alder", test_dots_symmetric<e2m1x2_t>, alder, nk_dots_symmetric_e2m1_alder);
    run_kernel_case("nk_dots_symmetric_i8_alder", test_dots_symmetric<i8_t>, alder, nk_dots_symmetric_i8_alder);
    run_kernel_case("nk_dots_symmetric_u8_alder", test_dots_symmetric<u8_t>, alder, nk_dots_symmetric_u8_alder);

    run_kernel_case("nk_angulars_packed_i8_alder", test_angulars_packed<i8_t>, alder, nk_dots_pack_size_i8_alder,
                    nk_dots_pack_i8_alder, nk_angulars_packed_i8_alder);
    run_kernel_case("nk_angulars_symmetric_i8_alder", test_angulars_symmetric<i8_t>, alder,
                    nk_angulars_symmetric_i8_alder);

    run_kernel_case("nk_euclideans_packed_i8_alder", test_euclideans_packed<i8_t>, alder, nk_dots_pack_size_i8_alder,
                    nk_dots_pack_i8_alder, nk_euclideans_packed_i8_alder);
    run_kernel_case("nk_euclideans_symmetric_i8_alder", test_euclideans_symmetric<i8_t>, alder,
                    nk_euclideans_symmetric_i8_alder);

    run_kernel_case("nk_angulars_packed_u8_alder", test_angulars_packed<u8_t>, alder, nk_dots_pack_size_u8_alder,
                    nk_dots_pack_u8_alder, nk_angulars_packed_u8_alder);
    run_kernel_case("nk_angulars_symmetric_u8_alder", test_angulars_symmetric<u8_t>, alder,
                    nk_angulars_symmetric_u8_alder);

    run_kernel_case("nk_euclideans_packed_u8_alder", test_euclideans_packed<u8_t>, alder, nk_dots_pack_size_u8_alder,
                    nk_dots_pack_u8_alder, nk_euclideans_packed_u8_alder);
    run_kernel_case("nk_euclideans_symmetric_u8_alder", test_euclideans_symmetric<u8_t>, alder,
                    nk_euclideans_symmetric_u8_alder);

    run_kernel_case("nk_angulars_packed_e2m3_alder", test_angulars_packed<e2m3_t>, alder, nk_dots_pack_size_e2m3_alder,
                    nk_dots_pack_e2m3_alder, nk_angulars_packed_e2m3_alder);
    run_kernel_case("nk_angulars_packed_e2m1_alder", test_angulars_packed<e2m1x2_t>, alder,
                    nk_dots_pack_size_e2m1_alder, nk_dots_pack_e2m1_alder, nk_angulars_packed_e2m1_alder);
    run_kernel_case("nk_angulars_symmetric_e2m3_alder", test_angulars_symmetric<e2m3_t>, alder,
                    nk_angulars_symmetric_e2m3_alder);
    run_kernel_case("nk_angulars_symmetric_e2m1_alder", test_angulars_symmetric<e2m1x2_t>, alder,
                    nk_angulars_symmetric_e2m1_alder);

    run_kernel_case("nk_euclideans_packed_e2m3_alder", test_euclideans_packed<e2m3_t>, alder,
                    nk_dots_pack_size_e2m3_alder, nk_dots_pack_e2m3_alder, nk_euclideans_packed_e2m3_alder);
    run_kernel_case("nk_euclideans_packed_e2m1_alder", test_euclideans_packed<e2m1x2_t>, alder,
                    nk_dots_pack_size_e2m1_alder, nk_dots_pack_e2m1_alder, nk_euclideans_packed_e2m1_alder);
    run_kernel_case("nk_euclideans_symmetric_e2m3_alder", test_euclideans_symmetric<e2m3_t>, alder,
                    nk_euclideans_symmetric_e2m3_alder);
    run_kernel_case("nk_euclideans_symmetric_e2m1_alder", test_euclideans_symmetric<e2m1x2_t>, alder,
                    nk_euclideans_symmetric_e2m1_alder);
#endif // NUMKONG_TARGET_ALDER

#if NUMKONG_TARGET_SIERRA
    auto const sierra = nk_cap_sierra_k;
    run_kernel_case("nk_dots_packed_e2m3_sierra", test_dots_packed<e2m3_t>, sierra, nk_dots_pack_size_e2m3_sierra,
                    nk_dots_pack_e2m3_sierra, nk_dots_packed_e2m3_sierra);
    run_kernel_case("nk_dots_pack_e2m3_sierra", test_dots_pack_layout<e2m3_t, host_backend_t>, sierra,
                    nk_dots_pack_size_e2m3_sierra, nk_dots_packed_shape_e2m3_sierra, nk_dots_pack_e2m3_sierra);
    run_kernel_case("nk_dots_packed_e2m1_sierra", test_dots_packed<e2m1x2_t>, sierra, nk_dots_pack_size_e2m1_sierra,
                    nk_dots_pack_e2m1_sierra, nk_dots_packed_e2m1_sierra);
    run_kernel_case("nk_dots_pack_e2m1_sierra", test_dots_pack_layout<e2m1x2_t, host_backend_t>, sierra,
                    nk_dots_pack_size_e2m1_sierra, nk_dots_packed_shape_e2m1_sierra, nk_dots_pack_e2m1_sierra);
    run_kernel_case("nk_dots_packed_i8_sierra", test_dots_packed<i8_t>, sierra, nk_dots_pack_size_i8_sierra,
                    nk_dots_pack_i8_sierra, nk_dots_packed_i8_sierra);
    run_kernel_case("nk_dots_pack_i8_sierra", test_dots_pack_layout<i8_t, host_backend_t>, sierra,
                    nk_dots_pack_size_i8_sierra, nk_dots_packed_shape_i8_sierra, nk_dots_pack_i8_sierra);
    run_kernel_case("nk_dots_packed_u8_sierra", test_dots_packed<u8_t>, sierra, nk_dots_pack_size_u8_sierra,
                    nk_dots_pack_u8_sierra, nk_dots_packed_u8_sierra);
    run_kernel_case("nk_dots_pack_u8_sierra", test_dots_pack_layout<u8_t, host_backend_t>, sierra,
                    nk_dots_pack_size_u8_sierra, nk_dots_packed_shape_u8_sierra, nk_dots_pack_u8_sierra);

    run_kernel_case("nk_dots_symmetric_e2m3_sierra", test_dots_symmetric<e2m3_t>, sierra,
                    nk_dots_symmetric_e2m3_sierra);
    run_kernel_case("nk_dots_symmetric_e2m1_sierra", test_dots_symmetric<e2m1x2_t>, sierra,
                    nk_dots_symmetric_e2m1_sierra);
    run_kernel_case("nk_dots_symmetric_i8_sierra", test_dots_symmetric<i8_t>, sierra, nk_dots_symmetric_i8_sierra);
    run_kernel_case("nk_dots_symmetric_u8_sierra", test_dots_symmetric<u8_t>, sierra, nk_dots_symmetric_u8_sierra);

    run_kernel_case("nk_angulars_packed_i8_sierra", test_angulars_packed<i8_t>, sierra, nk_dots_pack_size_i8_sierra,
                    nk_dots_pack_i8_sierra, nk_angulars_packed_i8_sierra);
    run_kernel_case("nk_angulars_symmetric_i8_sierra", test_angulars_symmetric<i8_t>, sierra,
                    nk_angulars_symmetric_i8_sierra);

    run_kernel_case("nk_euclideans_packed_i8_sierra", test_euclideans_packed<i8_t>, sierra, nk_dots_pack_size_i8_sierra,
                    nk_dots_pack_i8_sierra, nk_euclideans_packed_i8_sierra);
    run_kernel_case("nk_euclideans_symmetric_i8_sierra", test_euclideans_symmetric<i8_t>, sierra,
                    nk_euclideans_symmetric_i8_sierra);

    run_kernel_case("nk_angulars_packed_u8_sierra", test_angulars_packed<u8_t>, sierra, nk_dots_pack_size_u8_sierra,
                    nk_dots_pack_u8_sierra, nk_angulars_packed_u8_sierra);
    run_kernel_case("nk_angulars_symmetric_u8_sierra", test_angulars_symmetric<u8_t>, sierra,
                    nk_angulars_symmetric_u8_sierra);

    run_kernel_case("nk_euclideans_packed_u8_sierra", test_euclideans_packed<u8_t>, sierra, nk_dots_pack_size_u8_sierra,
                    nk_dots_pack_u8_sierra, nk_euclideans_packed_u8_sierra);
    run_kernel_case("nk_euclideans_symmetric_u8_sierra", test_euclideans_symmetric<u8_t>, sierra,
                    nk_euclideans_symmetric_u8_sierra);

    run_kernel_case("nk_angulars_packed_e2m3_sierra", test_angulars_packed<e2m3_t>, sierra,
                    nk_dots_pack_size_e2m3_sierra, nk_dots_pack_e2m3_sierra, nk_angulars_packed_e2m3_sierra);
    run_kernel_case("nk_angulars_packed_e2m1_sierra", test_angulars_packed<e2m1x2_t>, sierra,
                    nk_dots_pack_size_e2m1_sierra, nk_dots_pack_e2m1_sierra, nk_angulars_packed_e2m1_sierra);
    run_kernel_case("nk_angulars_symmetric_e2m3_sierra", test_angulars_symmetric<e2m3_t>, sierra,
                    nk_angulars_symmetric_e2m3_sierra);
    run_kernel_case("nk_angulars_symmetric_e2m1_sierra", test_angulars_symmetric<e2m1x2_t>, sierra,
                    nk_angulars_symmetric_e2m1_sierra);

    run_kernel_case("nk_euclideans_packed_e2m3_sierra", test_euclideans_packed<e2m3_t>, sierra,
                    nk_dots_pack_size_e2m3_sierra, nk_dots_pack_e2m3_sierra, nk_euclideans_packed_e2m3_sierra);
    run_kernel_case("nk_euclideans_packed_e2m1_sierra", test_euclideans_packed<e2m1x2_t>, sierra,
                    nk_dots_pack_size_e2m1_sierra, nk_dots_pack_e2m1_sierra, nk_euclideans_packed_e2m1_sierra);
    run_kernel_case("nk_euclideans_symmetric_e2m3_sierra", test_euclideans_symmetric<e2m3_t>, sierra,
                    nk_euclideans_symmetric_e2m3_sierra);
    run_kernel_case("nk_euclideans_symmetric_e2m1_sierra", test_euclideans_symmetric<e2m1x2_t>, sierra,
                    nk_euclideans_symmetric_e2m1_sierra);
#endif // NUMKONG_TARGET_SIERRA
}

} // namespace ashvardanian::numkong::test
