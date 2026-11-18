/**
 *  @file test/cross_arm64.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *  @brief Batch operation tests - Arm64 ISA family.
 *
 *  Covers NEON, NEONBFDOT, NEONFHM, NEONSDOT, NEONFP8, SME, SMEBI32, SMEF64.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include "harness.hpp"
#include "cross.hpp"

namespace ashvardanian::numkong::test {

void run_cross_arm64_cases() {
#if NUMKONG_TARGET_NEON
    auto const neon = nk_cap_neon_k;
    run_kernel_case("nk_dots_packed_f64_neon", test_dots_packed<f64_t>, neon, nk_dots_pack_size_f64_neon,
                    nk_dots_pack_f64_neon, nk_dots_packed_f64_neon);
    run_kernel_case("nk_dots_pack_f64_neon", test_dots_pack_layout<f64_t, host_backend_t>, neon,
                    nk_dots_pack_size_f64_neon, nk_dots_packed_shape_f64_neon, nk_dots_pack_f64_neon);
    run_kernel_case("nk_dots_packed_f32_neon", test_dots_packed<f32_t>, neon, nk_dots_pack_size_f32_neon,
                    nk_dots_pack_f32_neon, nk_dots_packed_f32_neon);
    run_kernel_case("nk_dots_pack_f32_neon", test_dots_pack_layout<f32_t, host_backend_t>, neon,
                    nk_dots_pack_size_f32_neon, nk_dots_packed_shape_f32_neon, nk_dots_pack_f32_neon);
    run_kernel_case("nk_dots_packed_bf16_neon", test_dots_packed<bf16_t>, neon, nk_dots_pack_size_bf16_neon,
                    nk_dots_pack_bf16_neon, nk_dots_packed_bf16_neon);
    run_kernel_case("nk_dots_pack_bf16_neon", test_dots_pack_layout<bf16_t, host_backend_t>, neon,
                    nk_dots_pack_size_bf16_neon, nk_dots_packed_shape_bf16_neon, nk_dots_pack_bf16_neon);
    run_kernel_case("nk_dots_packed_u1_neon", test_dots_packed<u1x8_t>, neon, nk_dots_pack_size_u1_neon,
                    nk_dots_pack_u1_neon, nk_dots_packed_u1_neon);

    run_kernel_case("nk_dots_symmetric_f64_neon", test_dots_symmetric<f64_t>, neon, nk_dots_symmetric_f64_neon);
    run_kernel_case("nk_dots_symmetric_f32_neon", test_dots_symmetric<f32_t>, neon, nk_dots_symmetric_f32_neon);
    run_kernel_case("nk_dots_symmetric_bf16_neon", test_dots_symmetric<bf16_t>, neon, nk_dots_symmetric_bf16_neon);
    run_kernel_case("nk_dots_symmetric_u1_neon", test_dots_symmetric<u1x8_t>, neon, nk_dots_symmetric_u1_neon);

    run_kernel_case("nk_angulars_packed_f64_neon", test_angulars_packed<f64_t>, neon, nk_dots_pack_size_f64_neon,
                    nk_dots_pack_f64_neon, nk_angulars_packed_f64_neon);
    run_kernel_case("nk_angulars_packed_f32_neon", test_angulars_packed<f32_t>, neon, nk_dots_pack_size_f32_neon,
                    nk_dots_pack_f32_neon, nk_angulars_packed_f32_neon);
    run_kernel_case("nk_angulars_packed_bf16_neon", test_angulars_packed<bf16_t>, neon, nk_dots_pack_size_bf16_neon,
                    nk_dots_pack_bf16_neon, nk_angulars_packed_bf16_neon);

    run_kernel_case("nk_angulars_symmetric_f64_neon", test_angulars_symmetric<f64_t>, neon,
                    nk_angulars_symmetric_f64_neon);
    run_kernel_case("nk_angulars_symmetric_f32_neon", test_angulars_symmetric<f32_t>, neon,
                    nk_angulars_symmetric_f32_neon);
    run_kernel_case("nk_angulars_symmetric_bf16_neon", test_angulars_symmetric<bf16_t>, neon,
                    nk_angulars_symmetric_bf16_neon);

    run_kernel_case("nk_euclideans_packed_f64_neon", test_euclideans_packed<f64_t>, neon, nk_dots_pack_size_f64_neon,
                    nk_dots_pack_f64_neon, nk_euclideans_packed_f64_neon);
    run_kernel_case("nk_euclideans_packed_f32_neon", test_euclideans_packed<f32_t>, neon, nk_dots_pack_size_f32_neon,
                    nk_dots_pack_f32_neon, nk_euclideans_packed_f32_neon);
    run_kernel_case("nk_euclideans_packed_bf16_neon", test_euclideans_packed<bf16_t>, neon, nk_dots_pack_size_bf16_neon,
                    nk_dots_pack_bf16_neon, nk_euclideans_packed_bf16_neon);

    run_kernel_case("nk_euclideans_symmetric_f64_neon", test_euclideans_symmetric<f64_t>, neon,
                    nk_euclideans_symmetric_f64_neon);
    run_kernel_case("nk_euclideans_symmetric_f32_neon", test_euclideans_symmetric<f32_t>, neon,
                    nk_euclideans_symmetric_f32_neon);
    run_kernel_case("nk_euclideans_symmetric_bf16_neon", test_euclideans_symmetric<bf16_t>, neon,
                    nk_euclideans_symmetric_bf16_neon);

    run_kernel_case("nk_dots_packed_f16_neon", test_dots_packed<f16_t>, neon, nk_dots_pack_size_f16_neon,
                    nk_dots_pack_f16_neon, nk_dots_packed_f16_neon);
    run_kernel_case("nk_dots_pack_f16_neon", test_dots_pack_layout<f16_t, host_backend_t>, neon,
                    nk_dots_pack_size_f16_neon, nk_dots_packed_shape_f16_neon, nk_dots_pack_f16_neon);
    run_kernel_case("nk_dots_symmetric_f16_neon", test_dots_symmetric<f16_t>, neon, nk_dots_symmetric_f16_neon);

    run_kernel_case("nk_angulars_packed_f16_neon", test_angulars_packed<f16_t>, neon, nk_dots_pack_size_f16_neon,
                    nk_dots_pack_f16_neon, nk_angulars_packed_f16_neon);
    run_kernel_case("nk_angulars_symmetric_f16_neon", test_angulars_symmetric<f16_t>, neon,
                    nk_angulars_symmetric_f16_neon);

    run_kernel_case("nk_euclideans_packed_f16_neon", test_euclideans_packed<f16_t>, neon, nk_dots_pack_size_f16_neon,
                    nk_dots_pack_f16_neon, nk_euclideans_packed_f16_neon);
    run_kernel_case("nk_euclideans_symmetric_f16_neon", test_euclideans_symmetric<f16_t>, neon,
                    nk_euclideans_symmetric_f16_neon);

    run_kernel_case("nk_hammings_packed_u1_neon", test_hammings_packed<u1x8_t>, neon, nk_dots_pack_size_u1_neon,
                    nk_dots_pack_u1_neon, nk_hammings_packed_u1_neon);
    run_kernel_case("nk_hammings_symmetric_u1_neon", test_hammings_symmetric<u1x8_t>, neon,
                    nk_hammings_symmetric_u1_neon);

    run_kernel_case("nk_jaccards_packed_u1_neon", test_jaccards_packed<u1x8_t>, neon, nk_dots_pack_size_u1_neon,
                    nk_dots_pack_u1_neon, nk_jaccards_packed_u1_neon);
    run_kernel_case("nk_jaccards_symmetric_u1_neon", test_jaccards_symmetric<u1x8_t>, neon,
                    nk_jaccards_symmetric_u1_neon);
    run_kernel_case("nk_dots_packed_nvfp4_neon", test_dots_packed<nvfp4_t>, neon, nk_dots_pack_size_nvfp4_neon,
                    nk_dots_pack_nvfp4_neon, nk_dots_packed_nvfp4_neon);
    run_kernel_case("nk_dots_symmetric_nvfp4_neon", test_dots_symmetric<nvfp4_t>, neon, nk_dots_symmetric_nvfp4_neon);
    run_kernel_case("nk_angulars_packed_nvfp4_neon", test_angulars_packed<nvfp4_t>, neon, nk_dots_pack_size_nvfp4_neon,
                    nk_dots_pack_nvfp4_neon, nk_angulars_packed_nvfp4_neon);
    run_kernel_case("nk_angulars_symmetric_nvfp4_neon", test_angulars_symmetric<nvfp4_t>, neon,
                    nk_angulars_symmetric_nvfp4_neon);
    run_kernel_case("nk_euclideans_packed_nvfp4_neon", test_euclideans_packed<nvfp4_t>, neon,
                    nk_dots_pack_size_nvfp4_neon, nk_dots_pack_nvfp4_neon, nk_euclideans_packed_nvfp4_neon);
    run_kernel_case("nk_euclideans_symmetric_nvfp4_neon", test_euclideans_symmetric<nvfp4_t>, neon,
                    nk_euclideans_symmetric_nvfp4_neon);
    run_kernel_case("nk_dots_packed_mxfp4_neon", test_dots_packed<mxfp4_t>, neon, nk_dots_pack_size_mxfp4_neon,
                    nk_dots_pack_mxfp4_neon, nk_dots_packed_mxfp4_neon);
    run_kernel_case("nk_dots_symmetric_mxfp4_neon", test_dots_symmetric<mxfp4_t>, neon, nk_dots_symmetric_mxfp4_neon);
    run_kernel_case("nk_angulars_packed_mxfp4_neon", test_angulars_packed<mxfp4_t>, neon, nk_dots_pack_size_mxfp4_neon,
                    nk_dots_pack_mxfp4_neon, nk_angulars_packed_mxfp4_neon);
    run_kernel_case("nk_angulars_symmetric_mxfp4_neon", test_angulars_symmetric<mxfp4_t>, neon,
                    nk_angulars_symmetric_mxfp4_neon);
    run_kernel_case("nk_euclideans_packed_mxfp4_neon", test_euclideans_packed<mxfp4_t>, neon,
                    nk_dots_pack_size_mxfp4_neon, nk_dots_pack_mxfp4_neon, nk_euclideans_packed_mxfp4_neon);
    run_kernel_case("nk_euclideans_symmetric_mxfp4_neon", test_euclideans_symmetric<mxfp4_t>, neon,
                    nk_euclideans_symmetric_mxfp4_neon);
    run_kernel_case("nk_dots_packed_mxfp6e2m3_neon", test_dots_packed<mxfp6e2m3_t>, neon,
                    nk_dots_pack_size_mxfp6e2m3_neon, nk_dots_pack_mxfp6e2m3_neon, nk_dots_packed_mxfp6e2m3_neon);
    run_kernel_case("nk_dots_symmetric_mxfp6e2m3_neon", test_dots_symmetric<mxfp6e2m3_t>, neon,
                    nk_dots_symmetric_mxfp6e2m3_neon);
    run_kernel_case("nk_angulars_packed_mxfp6e2m3_neon", test_angulars_packed<mxfp6e2m3_t>, neon,
                    nk_dots_pack_size_mxfp6e2m3_neon, nk_dots_pack_mxfp6e2m3_neon, nk_angulars_packed_mxfp6e2m3_neon);
    run_kernel_case("nk_angulars_symmetric_mxfp6e2m3_neon", test_angulars_symmetric<mxfp6e2m3_t>, neon,
                    nk_angulars_symmetric_mxfp6e2m3_neon);
    run_kernel_case("nk_euclideans_packed_mxfp6e2m3_neon", test_euclideans_packed<mxfp6e2m3_t>, neon,
                    nk_dots_pack_size_mxfp6e2m3_neon, nk_dots_pack_mxfp6e2m3_neon, nk_euclideans_packed_mxfp6e2m3_neon);
    run_kernel_case("nk_euclideans_symmetric_mxfp6e2m3_neon", test_euclideans_symmetric<mxfp6e2m3_t>, neon,
                    nk_euclideans_symmetric_mxfp6e2m3_neon);
    run_kernel_case("nk_dots_packed_mxfp6e3m2_neon", test_dots_packed<mxfp6e3m2_t>, neon,
                    nk_dots_pack_size_mxfp6e3m2_neon, nk_dots_pack_mxfp6e3m2_neon, nk_dots_packed_mxfp6e3m2_neon);
    run_kernel_case("nk_dots_symmetric_mxfp6e3m2_neon", test_dots_symmetric<mxfp6e3m2_t>, neon,
                    nk_dots_symmetric_mxfp6e3m2_neon);
    run_kernel_case("nk_angulars_packed_mxfp6e3m2_neon", test_angulars_packed<mxfp6e3m2_t>, neon,
                    nk_dots_pack_size_mxfp6e3m2_neon, nk_dots_pack_mxfp6e3m2_neon, nk_angulars_packed_mxfp6e3m2_neon);
    run_kernel_case("nk_angulars_symmetric_mxfp6e3m2_neon", test_angulars_symmetric<mxfp6e3m2_t>, neon,
                    nk_angulars_symmetric_mxfp6e3m2_neon);
    run_kernel_case("nk_euclideans_packed_mxfp6e3m2_neon", test_euclideans_packed<mxfp6e3m2_t>, neon,
                    nk_dots_pack_size_mxfp6e3m2_neon, nk_dots_pack_mxfp6e3m2_neon, nk_euclideans_packed_mxfp6e3m2_neon);
    run_kernel_case("nk_euclideans_symmetric_mxfp6e3m2_neon", test_euclideans_symmetric<mxfp6e3m2_t>, neon,
                    nk_euclideans_symmetric_mxfp6e3m2_neon);
    run_kernel_case("nk_dots_packed_mxfp8e4m3_neon", test_dots_packed<mxfp8e4m3_t>, neon,
                    nk_dots_pack_size_mxfp8e4m3_neon, nk_dots_pack_mxfp8e4m3_neon, nk_dots_packed_mxfp8e4m3_neon);
    run_kernel_case("nk_dots_symmetric_mxfp8e4m3_neon", test_dots_symmetric<mxfp8e4m3_t>, neon,
                    nk_dots_symmetric_mxfp8e4m3_neon);
    run_kernel_case("nk_angulars_packed_mxfp8e4m3_neon", test_angulars_packed<mxfp8e4m3_t>, neon,
                    nk_dots_pack_size_mxfp8e4m3_neon, nk_dots_pack_mxfp8e4m3_neon, nk_angulars_packed_mxfp8e4m3_neon);
    run_kernel_case("nk_angulars_symmetric_mxfp8e4m3_neon", test_angulars_symmetric<mxfp8e4m3_t>, neon,
                    nk_angulars_symmetric_mxfp8e4m3_neon);
    run_kernel_case("nk_euclideans_packed_mxfp8e4m3_neon", test_euclideans_packed<mxfp8e4m3_t>, neon,
                    nk_dots_pack_size_mxfp8e4m3_neon, nk_dots_pack_mxfp8e4m3_neon, nk_euclideans_packed_mxfp8e4m3_neon);
    run_kernel_case("nk_euclideans_symmetric_mxfp8e4m3_neon", test_euclideans_symmetric<mxfp8e4m3_t>, neon,
                    nk_euclideans_symmetric_mxfp8e4m3_neon);
    run_kernel_case("nk_dots_packed_mxfp8e5m2_neon", test_dots_packed<mxfp8e5m2_t>, neon,
                    nk_dots_pack_size_mxfp8e5m2_neon, nk_dots_pack_mxfp8e5m2_neon, nk_dots_packed_mxfp8e5m2_neon);
    run_kernel_case("nk_dots_symmetric_mxfp8e5m2_neon", test_dots_symmetric<mxfp8e5m2_t>, neon,
                    nk_dots_symmetric_mxfp8e5m2_neon);
    run_kernel_case("nk_angulars_packed_mxfp8e5m2_neon", test_angulars_packed<mxfp8e5m2_t>, neon,
                    nk_dots_pack_size_mxfp8e5m2_neon, nk_dots_pack_mxfp8e5m2_neon, nk_angulars_packed_mxfp8e5m2_neon);
    run_kernel_case("nk_angulars_symmetric_mxfp8e5m2_neon", test_angulars_symmetric<mxfp8e5m2_t>, neon,
                    nk_angulars_symmetric_mxfp8e5m2_neon);
    run_kernel_case("nk_euclideans_packed_mxfp8e5m2_neon", test_euclideans_packed<mxfp8e5m2_t>, neon,
                    nk_dots_pack_size_mxfp8e5m2_neon, nk_dots_pack_mxfp8e5m2_neon, nk_euclideans_packed_mxfp8e5m2_neon);
    run_kernel_case("nk_euclideans_symmetric_mxfp8e5m2_neon", test_euclideans_symmetric<mxfp8e5m2_t>, neon,
                    nk_euclideans_symmetric_mxfp8e5m2_neon);

    run_kernel_case("nk_attention_packed_bf16_neon", test_attention_packed<bf16_t>, neon,
                    nk_attention_pack_size_bf16_neon, nk_attention_pack_bf16_neon, nk_attention_packed_bf16_neon);
    run_kernel_case("nk_attention_packed_f16_neon", test_attention_packed<f16_t>, neon, nk_attention_pack_size_f16_neon,
                    nk_attention_pack_f16_neon, nk_attention_packed_f16_neon);
    run_kernel_case("nk_attention_packed_e4m3_neon", test_attention_packed<e4m3_t>, neon,
                    nk_attention_pack_size_e4m3_neon, nk_attention_pack_e4m3_neon, nk_attention_packed_e4m3_neon);
    run_kernel_case("nk_attention_packed_i8_neon", test_attention_packed<i8_t>, neon, nk_attention_pack_size_i8_neon,
                    nk_attention_pack_i8_neon, nk_attention_packed_i8_neon);
    run_kernel_case("nk_attention_packed_nvfp4_neon", test_attention_packed<nvfp4_t>, neon,
                    nk_attention_pack_size_nvfp4_neon, nk_attention_pack_nvfp4_neon, nk_attention_packed_nvfp4_neon);
    run_kernel_case("nk_attention_packed_mxfp4_neon", test_attention_packed<mxfp4_t>, neon,
                    nk_attention_pack_size_mxfp4_neon, nk_attention_pack_mxfp4_neon, nk_attention_packed_mxfp4_neon);
    run_kernel_case("nk_attention_packed_mxfp6e2m3_neon", test_attention_packed<mxfp6e2m3_t>, neon,
                    nk_attention_pack_size_mxfp6e2m3_neon, nk_attention_pack_mxfp6e2m3_neon,
                    nk_attention_packed_mxfp6e2m3_neon);
    run_kernel_case("nk_attention_packed_mxfp6e3m2_neon", test_attention_packed<mxfp6e3m2_t>, neon,
                    nk_attention_pack_size_mxfp6e3m2_neon, nk_attention_pack_mxfp6e3m2_neon,
                    nk_attention_packed_mxfp6e3m2_neon);
    run_kernel_case("nk_attention_packed_mxfp8e4m3_neon", test_attention_packed<mxfp8e4m3_t>, neon,
                    nk_attention_pack_size_mxfp8e4m3_neon, nk_attention_pack_mxfp8e4m3_neon,
                    nk_attention_packed_mxfp8e4m3_neon);
    run_kernel_case("nk_attention_packed_mxfp8e5m2_neon", test_attention_packed<mxfp8e5m2_t>, neon,
                    nk_attention_pack_size_mxfp8e5m2_neon, nk_attention_pack_mxfp8e5m2_neon,
                    nk_attention_packed_mxfp8e5m2_neon);
    run_kernel_case("nk_attention_rope_f32_neon", test_attention_rope<f32_t>, neon, nk_attention_rope_f32_neon);
    run_kernel_case("nk_attention_rope_bf16_neon", test_attention_rope<bf16_t>, neon, nk_attention_rope_bf16_neon);
    run_kernel_case("nk_attention_rope_e4m3_neon", test_attention_rope<e4m3_t>, neon, nk_attention_rope_e4m3_neon);

#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_NEONBFDOT
    auto const neonbfdot = nk_cap_neonbfdot_k;
    run_kernel_case("nk_dots_packed_bf16_neonbfdot", test_dots_packed<bf16_t>, neonbfdot,
                    nk_dots_pack_size_bf16_neonbfdot, nk_dots_pack_bf16_neonbfdot, nk_dots_packed_bf16_neonbfdot);
    run_kernel_case("nk_dots_pack_bf16_neonbfdot", test_dots_pack_layout<bf16_t, host_backend_t>, neonbfdot,
                    nk_dots_pack_size_bf16_neonbfdot, nk_dots_packed_shape_bf16_neonbfdot, nk_dots_pack_bf16_neonbfdot);
    run_kernel_case("nk_dots_symmetric_bf16_neonbfdot", test_dots_symmetric<bf16_t>, neonbfdot,
                    nk_dots_symmetric_bf16_neonbfdot);

    run_kernel_case("nk_angulars_packed_bf16_neonbfdot", test_angulars_packed<bf16_t>, neonbfdot,
                    nk_dots_pack_size_bf16_neonbfdot, nk_dots_pack_bf16_neonbfdot, nk_angulars_packed_bf16_neonbfdot);
    run_kernel_case("nk_angulars_symmetric_bf16_neonbfdot", test_angulars_symmetric<bf16_t>, neonbfdot,
                    nk_angulars_symmetric_bf16_neonbfdot);

    run_kernel_case("nk_euclideans_packed_bf16_neonbfdot", test_euclideans_packed<bf16_t>, neonbfdot,
                    nk_dots_pack_size_bf16_neonbfdot, nk_dots_pack_bf16_neonbfdot, nk_euclideans_packed_bf16_neonbfdot);
    run_kernel_case("nk_euclideans_symmetric_bf16_neonbfdot", test_euclideans_symmetric<bf16_t>, neonbfdot,
                    nk_euclideans_symmetric_bf16_neonbfdot);

    run_kernel_case("nk_attention_packed_bf16_neonbfdot", test_attention_packed<bf16_t>, neonbfdot,
                    nk_attention_pack_size_bf16_neonbfdot, nk_attention_pack_bf16_neonbfdot,
                    nk_attention_packed_bf16_neonbfdot);
    run_kernel_case("nk_attention_packed_mxfp4_neonbfdot", test_attention_packed<mxfp4_t>, neonbfdot,
                    nk_attention_pack_size_mxfp4_neonbfdot, nk_attention_pack_mxfp4_neonbfdot,
                    nk_attention_packed_mxfp4_neonbfdot);
    run_kernel_case("nk_attention_packed_mxfp6e2m3_neonbfdot", test_attention_packed<mxfp6e2m3_t>, neonbfdot,
                    nk_attention_pack_size_mxfp6e2m3_neonbfdot, nk_attention_pack_mxfp6e2m3_neonbfdot,
                    nk_attention_packed_mxfp6e2m3_neonbfdot);
    run_kernel_case("nk_attention_packed_mxfp6e3m2_neonbfdot", test_attention_packed<mxfp6e3m2_t>, neonbfdot,
                    nk_attention_pack_size_mxfp6e3m2_neonbfdot, nk_attention_pack_mxfp6e3m2_neonbfdot,
                    nk_attention_packed_mxfp6e3m2_neonbfdot);
    run_kernel_case("nk_attention_packed_mxfp8e4m3_neonbfdot", test_attention_packed<mxfp8e4m3_t>, neonbfdot,
                    nk_attention_pack_size_mxfp8e4m3_neonbfdot, nk_attention_pack_mxfp8e4m3_neonbfdot,
                    nk_attention_packed_mxfp8e4m3_neonbfdot);
    run_kernel_case("nk_attention_packed_mxfp8e5m2_neonbfdot", test_attention_packed<mxfp8e5m2_t>, neonbfdot,
                    nk_attention_pack_size_mxfp8e5m2_neonbfdot, nk_attention_pack_mxfp8e5m2_neonbfdot,
                    nk_attention_packed_mxfp8e5m2_neonbfdot);
#endif // NUMKONG_TARGET_NEONBFDOT

#if NUMKONG_TARGET_NEONFHM
    auto const neonfhm = nk_cap_neonfhm_k;
    run_kernel_case("nk_dots_packed_f16_neonfhm", test_dots_packed<f16_t>, neonfhm, nk_dots_pack_size_f16_neonfhm,
                    nk_dots_pack_f16_neonfhm, nk_dots_packed_f16_neonfhm);
    run_kernel_case("nk_dots_pack_f16_neonfhm", test_dots_pack_layout<f16_t, host_backend_t>, neonfhm,
                    nk_dots_pack_size_f16_neonfhm, nk_dots_packed_shape_f16_neonfhm, nk_dots_pack_f16_neonfhm);
    run_kernel_case("nk_dots_symmetric_f16_neonfhm", test_dots_symmetric<f16_t>, neonfhm,
                    nk_dots_symmetric_f16_neonfhm);

    run_kernel_case("nk_angulars_packed_f16_neonfhm", test_angulars_packed<f16_t>, neonfhm,
                    nk_dots_pack_size_f16_neonfhm, nk_dots_pack_f16_neonfhm, nk_angulars_packed_f16_neonfhm);
    run_kernel_case("nk_angulars_packed_e5m2_neonfhm", test_angulars_packed<e5m2_t>, neonfhm,
                    nk_dots_pack_size_e5m2_neonfhm, nk_dots_pack_e5m2_neonfhm, nk_angulars_packed_e5m2_neonfhm);
    run_kernel_case("nk_angulars_packed_e4m3_neonfhm", test_angulars_packed<e4m3_t>, neonfhm,
                    nk_dots_pack_size_e4m3_neonfhm, nk_dots_pack_e4m3_neonfhm, nk_angulars_packed_e4m3_neonfhm);

    run_kernel_case("nk_angulars_symmetric_f16_neonfhm", test_angulars_symmetric<f16_t>, neonfhm,
                    nk_angulars_symmetric_f16_neonfhm);
    run_kernel_case("nk_angulars_symmetric_e5m2_neonfhm", test_angulars_symmetric<e5m2_t>, neonfhm,
                    nk_angulars_symmetric_e5m2_neonfhm);
    run_kernel_case("nk_angulars_symmetric_e4m3_neonfhm", test_angulars_symmetric<e4m3_t>, neonfhm,
                    nk_angulars_symmetric_e4m3_neonfhm);

    run_kernel_case("nk_euclideans_packed_f16_neonfhm", test_euclideans_packed<f16_t>, neonfhm,
                    nk_dots_pack_size_f16_neonfhm, nk_dots_pack_f16_neonfhm, nk_euclideans_packed_f16_neonfhm);
    run_kernel_case("nk_euclideans_packed_e5m2_neonfhm", test_euclideans_packed<e5m2_t>, neonfhm,
                    nk_dots_pack_size_e5m2_neonfhm, nk_dots_pack_e5m2_neonfhm, nk_euclideans_packed_e5m2_neonfhm);
    run_kernel_case("nk_euclideans_packed_e4m3_neonfhm", test_euclideans_packed<e4m3_t>, neonfhm,
                    nk_dots_pack_size_e4m3_neonfhm, nk_dots_pack_e4m3_neonfhm, nk_euclideans_packed_e4m3_neonfhm);

    run_kernel_case("nk_euclideans_symmetric_f16_neonfhm", test_euclideans_symmetric<f16_t>, neonfhm,
                    nk_euclideans_symmetric_f16_neonfhm);
    run_kernel_case("nk_euclideans_symmetric_e5m2_neonfhm", test_euclideans_symmetric<e5m2_t>, neonfhm,
                    nk_euclideans_symmetric_e5m2_neonfhm);
    run_kernel_case("nk_euclideans_symmetric_e4m3_neonfhm", test_euclideans_symmetric<e4m3_t>, neonfhm,
                    nk_euclideans_symmetric_e4m3_neonfhm);
    run_kernel_case("nk_dots_packed_mxfp8e4m3_neonfhm", test_dots_packed<mxfp8e4m3_t>, neonfhm,
                    nk_dots_pack_size_mxfp8e4m3_neonfhm, nk_dots_pack_mxfp8e4m3_neonfhm,
                    nk_dots_packed_mxfp8e4m3_neonfhm);
    run_kernel_case("nk_dots_symmetric_mxfp8e4m3_neonfhm", test_dots_symmetric<mxfp8e4m3_t>, neonfhm,
                    nk_dots_symmetric_mxfp8e4m3_neonfhm);
    run_kernel_case("nk_angulars_packed_mxfp8e4m3_neonfhm", test_angulars_packed<mxfp8e4m3_t>, neonfhm,
                    nk_dots_pack_size_mxfp8e4m3_neonfhm, nk_dots_pack_mxfp8e4m3_neonfhm,
                    nk_angulars_packed_mxfp8e4m3_neonfhm);
    run_kernel_case("nk_angulars_symmetric_mxfp8e4m3_neonfhm", test_angulars_symmetric<mxfp8e4m3_t>, neonfhm,
                    nk_angulars_symmetric_mxfp8e4m3_neonfhm);
    run_kernel_case("nk_euclideans_packed_mxfp8e4m3_neonfhm", test_euclideans_packed<mxfp8e4m3_t>, neonfhm,
                    nk_dots_pack_size_mxfp8e4m3_neonfhm, nk_dots_pack_mxfp8e4m3_neonfhm,
                    nk_euclideans_packed_mxfp8e4m3_neonfhm);
    run_kernel_case("nk_euclideans_symmetric_mxfp8e4m3_neonfhm", test_euclideans_symmetric<mxfp8e4m3_t>, neonfhm,
                    nk_euclideans_symmetric_mxfp8e4m3_neonfhm);
    run_kernel_case("nk_dots_packed_mxfp8e5m2_neonfhm", test_dots_packed<mxfp8e5m2_t>, neonfhm,
                    nk_dots_pack_size_mxfp8e5m2_neonfhm, nk_dots_pack_mxfp8e5m2_neonfhm,
                    nk_dots_packed_mxfp8e5m2_neonfhm);
    run_kernel_case("nk_dots_symmetric_mxfp8e5m2_neonfhm", test_dots_symmetric<mxfp8e5m2_t>, neonfhm,
                    nk_dots_symmetric_mxfp8e5m2_neonfhm);
    run_kernel_case("nk_angulars_packed_mxfp8e5m2_neonfhm", test_angulars_packed<mxfp8e5m2_t>, neonfhm,
                    nk_dots_pack_size_mxfp8e5m2_neonfhm, nk_dots_pack_mxfp8e5m2_neonfhm,
                    nk_angulars_packed_mxfp8e5m2_neonfhm);
    run_kernel_case("nk_angulars_symmetric_mxfp8e5m2_neonfhm", test_angulars_symmetric<mxfp8e5m2_t>, neonfhm,
                    nk_angulars_symmetric_mxfp8e5m2_neonfhm);
    run_kernel_case("nk_euclideans_packed_mxfp8e5m2_neonfhm", test_euclideans_packed<mxfp8e5m2_t>, neonfhm,
                    nk_dots_pack_size_mxfp8e5m2_neonfhm, nk_dots_pack_mxfp8e5m2_neonfhm,
                    nk_euclideans_packed_mxfp8e5m2_neonfhm);
    run_kernel_case("nk_euclideans_symmetric_mxfp8e5m2_neonfhm", test_euclideans_symmetric<mxfp8e5m2_t>, neonfhm,
                    nk_euclideans_symmetric_mxfp8e5m2_neonfhm);

    run_kernel_case("nk_attention_packed_f16_neonfhm", test_attention_packed<f16_t>, neonfhm,
                    nk_attention_pack_size_f16_neonfhm, nk_attention_pack_f16_neonfhm, nk_attention_packed_f16_neonfhm);
    run_kernel_case("nk_attention_packed_e4m3_neonfhm", test_attention_packed<e4m3_t>, neonfhm,
                    nk_attention_pack_size_e4m3_neonfhm, nk_attention_pack_e4m3_neonfhm,
                    nk_attention_packed_e4m3_neonfhm);
    run_kernel_case("nk_attention_packed_nvfp4_neonfhm", test_attention_packed<nvfp4_t>, neonfhm,
                    nk_attention_pack_size_nvfp4_neonfhm, nk_attention_pack_nvfp4_neonfhm,
                    nk_attention_packed_nvfp4_neonfhm);
#endif // NUMKONG_TARGET_NEONFHM

#if NUMKONG_TARGET_NEONSDOT
    auto const neonsdot = nk_cap_neonsdot_k;
    run_kernel_case("nk_dots_packed_i8_neonsdot", test_dots_packed<i8_t>, neonsdot, nk_dots_pack_size_i8_neonsdot,
                    nk_dots_pack_i8_neonsdot, nk_dots_packed_i8_neonsdot);
    run_kernel_case("nk_dots_pack_i8_neonsdot", test_dots_pack_layout<i8_t, host_backend_t>, neonsdot,
                    nk_dots_pack_size_i8_neonsdot, nk_dots_packed_shape_i8_neonsdot, nk_dots_pack_i8_neonsdot);
    run_kernel_case("nk_dots_packed_i4_neonsdot", test_dots_packed<i4x2_t>, neonsdot, nk_dots_pack_size_i4_neonsdot,
                    nk_dots_pack_i4_neonsdot, nk_dots_packed_i4_neonsdot);
    run_kernel_case("nk_dots_pack_i4_neonsdot", test_dots_pack_layout<i4x2_t, host_backend_t>, neonsdot,
                    nk_dots_pack_size_i4_neonsdot, nk_dots_packed_shape_i4_neonsdot, nk_dots_pack_i4_neonsdot);
    run_kernel_case("nk_dots_packed_u8_neonsdot", test_dots_packed<u8_t>, neonsdot, nk_dots_pack_size_u8_neonsdot,
                    nk_dots_pack_u8_neonsdot, nk_dots_packed_u8_neonsdot);
    run_kernel_case("nk_dots_pack_u8_neonsdot", test_dots_pack_layout<u8_t, host_backend_t>, neonsdot,
                    nk_dots_pack_size_u8_neonsdot, nk_dots_packed_shape_u8_neonsdot, nk_dots_pack_u8_neonsdot);
    run_kernel_case("nk_dots_packed_u4_neonsdot", test_dots_packed<u4x2_t>, neonsdot, nk_dots_pack_size_u4_neonsdot,
                    nk_dots_pack_u4_neonsdot, nk_dots_packed_u4_neonsdot);
    run_kernel_case("nk_dots_pack_u4_neonsdot", test_dots_pack_layout<u4x2_t, host_backend_t>, neonsdot,
                    nk_dots_pack_size_u4_neonsdot, nk_dots_packed_shape_u4_neonsdot, nk_dots_pack_u4_neonsdot);

    run_kernel_case("nk_dots_symmetric_i8_neonsdot", test_dots_symmetric<i8_t>, neonsdot,
                    nk_dots_symmetric_i8_neonsdot);
    run_kernel_case("nk_dots_symmetric_i4_neonsdot", test_dots_symmetric<i4x2_t>, neonsdot,
                    nk_dots_symmetric_i4_neonsdot);
    run_kernel_case("nk_dots_symmetric_u8_neonsdot", test_dots_symmetric<u8_t>, neonsdot,
                    nk_dots_symmetric_u8_neonsdot);
    run_kernel_case("nk_dots_symmetric_u4_neonsdot", test_dots_symmetric<u4x2_t>, neonsdot,
                    nk_dots_symmetric_u4_neonsdot);

    run_kernel_case("nk_angulars_packed_i8_neonsdot", test_angulars_packed<i8_t>, neonsdot,
                    nk_dots_pack_size_i8_neonsdot, nk_dots_pack_i8_neonsdot, nk_angulars_packed_i8_neonsdot);
    run_kernel_case("nk_angulars_packed_i4_neonsdot", test_angulars_packed<i4x2_t>, neonsdot,
                    nk_dots_pack_size_i4_neonsdot, nk_dots_pack_i4_neonsdot, nk_angulars_packed_i4_neonsdot);
    run_kernel_case("nk_angulars_packed_u8_neonsdot", test_angulars_packed<u8_t>, neonsdot,
                    nk_dots_pack_size_u8_neonsdot, nk_dots_pack_u8_neonsdot, nk_angulars_packed_u8_neonsdot);
    run_kernel_case("nk_angulars_packed_u4_neonsdot", test_angulars_packed<u4x2_t>, neonsdot,
                    nk_dots_pack_size_u4_neonsdot, nk_dots_pack_u4_neonsdot, nk_angulars_packed_u4_neonsdot);

    run_kernel_case("nk_angulars_symmetric_i8_neonsdot", test_angulars_symmetric<i8_t>, neonsdot,
                    nk_angulars_symmetric_i8_neonsdot);
    run_kernel_case("nk_angulars_symmetric_i4_neonsdot", test_angulars_symmetric<i4x2_t>, neonsdot,
                    nk_angulars_symmetric_i4_neonsdot);
    run_kernel_case("nk_angulars_symmetric_u8_neonsdot", test_angulars_symmetric<u8_t>, neonsdot,
                    nk_angulars_symmetric_u8_neonsdot);
    run_kernel_case("nk_angulars_symmetric_u4_neonsdot", test_angulars_symmetric<u4x2_t>, neonsdot,
                    nk_angulars_symmetric_u4_neonsdot);

    run_kernel_case("nk_euclideans_packed_i8_neonsdot", test_euclideans_packed<i8_t>, neonsdot,
                    nk_dots_pack_size_i8_neonsdot, nk_dots_pack_i8_neonsdot, nk_euclideans_packed_i8_neonsdot);
    run_kernel_case("nk_euclideans_packed_i4_neonsdot", test_euclideans_packed<i4x2_t>, neonsdot,
                    nk_dots_pack_size_i4_neonsdot, nk_dots_pack_i4_neonsdot, nk_euclideans_packed_i4_neonsdot);
    run_kernel_case("nk_euclideans_packed_u8_neonsdot", test_euclideans_packed<u8_t>, neonsdot,
                    nk_dots_pack_size_u8_neonsdot, nk_dots_pack_u8_neonsdot, nk_euclideans_packed_u8_neonsdot);
    run_kernel_case("nk_euclideans_packed_u4_neonsdot", test_euclideans_packed<u4x2_t>, neonsdot,
                    nk_dots_pack_size_u4_neonsdot, nk_dots_pack_u4_neonsdot, nk_euclideans_packed_u4_neonsdot);

    run_kernel_case("nk_euclideans_symmetric_i8_neonsdot", test_euclideans_symmetric<i8_t>, neonsdot,
                    nk_euclideans_symmetric_i8_neonsdot);
    run_kernel_case("nk_euclideans_symmetric_i4_neonsdot", test_euclideans_symmetric<i4x2_t>, neonsdot,
                    nk_euclideans_symmetric_i4_neonsdot);
    run_kernel_case("nk_euclideans_symmetric_u8_neonsdot", test_euclideans_symmetric<u8_t>, neonsdot,
                    nk_euclideans_symmetric_u8_neonsdot);
    run_kernel_case("nk_euclideans_symmetric_u4_neonsdot", test_euclideans_symmetric<u4x2_t>, neonsdot,
                    nk_euclideans_symmetric_u4_neonsdot);

    run_kernel_case("nk_dots_packed_e3m2_neonsdot", test_dots_packed<e3m2_t>, neonsdot, nk_dots_pack_size_e3m2_neonsdot,
                    nk_dots_pack_e3m2_neonsdot, nk_dots_packed_e3m2_neonsdot);
    run_kernel_case("nk_dots_pack_e3m2_neonsdot", test_dots_pack_layout<e3m2_t, host_backend_t>, neonsdot,
                    nk_dots_pack_size_e3m2_neonsdot, nk_dots_packed_shape_e3m2_neonsdot, nk_dots_pack_e3m2_neonsdot);
    run_kernel_case("nk_dots_packed_e2m3_neonsdot", test_dots_packed<e2m3_t>, neonsdot, nk_dots_pack_size_e2m3_neonsdot,
                    nk_dots_pack_e2m3_neonsdot, nk_dots_packed_e2m3_neonsdot);
    run_kernel_case("nk_dots_pack_e2m3_neonsdot", test_dots_pack_layout<e2m3_t, host_backend_t>, neonsdot,
                    nk_dots_pack_size_e2m3_neonsdot, nk_dots_packed_shape_e2m3_neonsdot, nk_dots_pack_e2m3_neonsdot);
    run_kernel_case("nk_dots_packed_e2m1_neonsdot", test_dots_packed<e2m1x2_t>, neonsdot,
                    nk_dots_pack_size_e2m1_neonsdot, nk_dots_pack_e2m1_neonsdot, nk_dots_packed_e2m1_neonsdot);
    run_kernel_case("nk_dots_pack_e2m1_neonsdot", test_dots_pack_layout<e2m1x2_t, host_backend_t>, neonsdot,
                    nk_dots_pack_size_e2m1_neonsdot, nk_dots_packed_shape_e2m1_neonsdot, nk_dots_pack_e2m1_neonsdot);
    run_kernel_case("nk_dots_symmetric_e3m2_neonsdot", test_dots_symmetric<e3m2_t>, neonsdot,
                    nk_dots_symmetric_e3m2_neonsdot);
    run_kernel_case("nk_dots_symmetric_e2m3_neonsdot", test_dots_symmetric<e2m3_t>, neonsdot,
                    nk_dots_symmetric_e2m3_neonsdot);
    run_kernel_case("nk_dots_symmetric_e2m1_neonsdot", test_dots_symmetric<e2m1x2_t>, neonsdot,
                    nk_dots_symmetric_e2m1_neonsdot);

    run_kernel_case("nk_angulars_packed_e3m2_neonsdot", test_angulars_packed<e3m2_t>, neonsdot,
                    nk_dots_pack_size_e3m2_neonsdot, nk_dots_pack_e3m2_neonsdot, nk_angulars_packed_e3m2_neonsdot);
    run_kernel_case("nk_angulars_packed_e2m3_neonsdot", test_angulars_packed<e2m3_t>, neonsdot,
                    nk_dots_pack_size_e2m3_neonsdot, nk_dots_pack_e2m3_neonsdot, nk_angulars_packed_e2m3_neonsdot);
    run_kernel_case("nk_angulars_packed_e2m1_neonsdot", test_angulars_packed<e2m1x2_t>, neonsdot,
                    nk_dots_pack_size_e2m1_neonsdot, nk_dots_pack_e2m1_neonsdot, nk_angulars_packed_e2m1_neonsdot);
    run_kernel_case("nk_angulars_symmetric_e3m2_neonsdot", test_angulars_symmetric<e3m2_t>, neonsdot,
                    nk_angulars_symmetric_e3m2_neonsdot);
    run_kernel_case("nk_angulars_symmetric_e2m3_neonsdot", test_angulars_symmetric<e2m3_t>, neonsdot,
                    nk_angulars_symmetric_e2m3_neonsdot);
    run_kernel_case("nk_angulars_symmetric_e2m1_neonsdot", test_angulars_symmetric<e2m1x2_t>, neonsdot,
                    nk_angulars_symmetric_e2m1_neonsdot);

    run_kernel_case("nk_euclideans_packed_e3m2_neonsdot", test_euclideans_packed<e3m2_t>, neonsdot,
                    nk_dots_pack_size_e3m2_neonsdot, nk_dots_pack_e3m2_neonsdot, nk_euclideans_packed_e3m2_neonsdot);
    run_kernel_case("nk_euclideans_packed_e2m3_neonsdot", test_euclideans_packed<e2m3_t>, neonsdot,
                    nk_dots_pack_size_e2m3_neonsdot, nk_dots_pack_e2m3_neonsdot, nk_euclideans_packed_e2m3_neonsdot);
    run_kernel_case("nk_euclideans_packed_e2m1_neonsdot", test_euclideans_packed<e2m1x2_t>, neonsdot,
                    nk_dots_pack_size_e2m1_neonsdot, nk_dots_pack_e2m1_neonsdot, nk_euclideans_packed_e2m1_neonsdot);
    run_kernel_case("nk_euclideans_symmetric_e3m2_neonsdot", test_euclideans_symmetric<e3m2_t>, neonsdot,
                    nk_euclideans_symmetric_e3m2_neonsdot);
    run_kernel_case("nk_euclideans_symmetric_e2m3_neonsdot", test_euclideans_symmetric<e2m3_t>, neonsdot,
                    nk_euclideans_symmetric_e2m3_neonsdot);
    run_kernel_case("nk_euclideans_symmetric_e2m1_neonsdot", test_euclideans_symmetric<e2m1x2_t>, neonsdot,
                    nk_euclideans_symmetric_e2m1_neonsdot);
    run_kernel_case("nk_dots_packed_nvfp4_neonsdot", test_dots_packed<nvfp4_t>, neonsdot,
                    nk_dots_pack_size_nvfp4_neonsdot, nk_dots_pack_nvfp4_neonsdot, nk_dots_packed_nvfp4_neonsdot);
    run_kernel_case("nk_dots_symmetric_nvfp4_neonsdot", test_dots_symmetric<nvfp4_t>, neonsdot,
                    nk_dots_symmetric_nvfp4_neonsdot);
    run_kernel_case("nk_angulars_packed_nvfp4_neonsdot", test_angulars_packed<nvfp4_t>, neonsdot,
                    nk_dots_pack_size_nvfp4_neonsdot, nk_dots_pack_nvfp4_neonsdot, nk_angulars_packed_nvfp4_neonsdot);
    run_kernel_case("nk_angulars_symmetric_nvfp4_neonsdot", test_angulars_symmetric<nvfp4_t>, neonsdot,
                    nk_angulars_symmetric_nvfp4_neonsdot);
    run_kernel_case("nk_euclideans_packed_nvfp4_neonsdot", test_euclideans_packed<nvfp4_t>, neonsdot,
                    nk_dots_pack_size_nvfp4_neonsdot, nk_dots_pack_nvfp4_neonsdot, nk_euclideans_packed_nvfp4_neonsdot);
    run_kernel_case("nk_euclideans_symmetric_nvfp4_neonsdot", test_euclideans_symmetric<nvfp4_t>, neonsdot,
                    nk_euclideans_symmetric_nvfp4_neonsdot);
    run_kernel_case("nk_dots_packed_mxfp4_neonsdot", test_dots_packed<mxfp4_t>, neonsdot,
                    nk_dots_pack_size_mxfp4_neonsdot, nk_dots_pack_mxfp4_neonsdot, nk_dots_packed_mxfp4_neonsdot);
    run_kernel_case("nk_dots_symmetric_mxfp4_neonsdot", test_dots_symmetric<mxfp4_t>, neonsdot,
                    nk_dots_symmetric_mxfp4_neonsdot);
    run_kernel_case("nk_angulars_packed_mxfp4_neonsdot", test_angulars_packed<mxfp4_t>, neonsdot,
                    nk_dots_pack_size_mxfp4_neonsdot, nk_dots_pack_mxfp4_neonsdot, nk_angulars_packed_mxfp4_neonsdot);
    run_kernel_case("nk_angulars_symmetric_mxfp4_neonsdot", test_angulars_symmetric<mxfp4_t>, neonsdot,
                    nk_angulars_symmetric_mxfp4_neonsdot);
    run_kernel_case("nk_euclideans_packed_mxfp4_neonsdot", test_euclideans_packed<mxfp4_t>, neonsdot,
                    nk_dots_pack_size_mxfp4_neonsdot, nk_dots_pack_mxfp4_neonsdot, nk_euclideans_packed_mxfp4_neonsdot);
    run_kernel_case("nk_euclideans_symmetric_mxfp4_neonsdot", test_euclideans_symmetric<mxfp4_t>, neonsdot,
                    nk_euclideans_symmetric_mxfp4_neonsdot);
    run_kernel_case("nk_dots_packed_mxfp6e2m3_neonsdot", test_dots_packed<mxfp6e2m3_t>, neonsdot,
                    nk_dots_pack_size_mxfp6e2m3_neonsdot, nk_dots_pack_mxfp6e2m3_neonsdot,
                    nk_dots_packed_mxfp6e2m3_neonsdot);
    run_kernel_case("nk_dots_symmetric_mxfp6e2m3_neonsdot", test_dots_symmetric<mxfp6e2m3_t>, neonsdot,
                    nk_dots_symmetric_mxfp6e2m3_neonsdot);
    run_kernel_case("nk_angulars_packed_mxfp6e2m3_neonsdot", test_angulars_packed<mxfp6e2m3_t>, neonsdot,
                    nk_dots_pack_size_mxfp6e2m3_neonsdot, nk_dots_pack_mxfp6e2m3_neonsdot,
                    nk_angulars_packed_mxfp6e2m3_neonsdot);
    run_kernel_case("nk_angulars_symmetric_mxfp6e2m3_neonsdot", test_angulars_symmetric<mxfp6e2m3_t>, neonsdot,
                    nk_angulars_symmetric_mxfp6e2m3_neonsdot);
    run_kernel_case("nk_euclideans_packed_mxfp6e2m3_neonsdot", test_euclideans_packed<mxfp6e2m3_t>, neonsdot,
                    nk_dots_pack_size_mxfp6e2m3_neonsdot, nk_dots_pack_mxfp6e2m3_neonsdot,
                    nk_euclideans_packed_mxfp6e2m3_neonsdot);
    run_kernel_case("nk_euclideans_symmetric_mxfp6e2m3_neonsdot", test_euclideans_symmetric<mxfp6e2m3_t>, neonsdot,
                    nk_euclideans_symmetric_mxfp6e2m3_neonsdot);

    run_kernel_case("nk_attention_packed_i8_neonsdot", test_attention_packed<i8_t>, neonsdot,
                    nk_attention_pack_size_i8_neonsdot, nk_attention_pack_i8_neonsdot, nk_attention_packed_i8_neonsdot);
#endif // NUMKONG_TARGET_NEONSDOT

#if NUMKONG_TARGET_NEONFP8
    auto const neonfp8 = nk_cap_neonfp8_k;
    run_kernel_case("nk_dots_packed_e5m2_neonfp8", test_dots_packed<e5m2_t>, neonfp8, nk_dots_pack_size_e5m2_neonfp8,
                    nk_dots_pack_e5m2_neonfp8, nk_dots_packed_e5m2_neonfp8);
    run_kernel_case("nk_dots_pack_e5m2_neonfp8", test_dots_pack_layout<e5m2_t, host_backend_t>, neonfp8,
                    nk_dots_pack_size_e5m2_neonfp8, nk_dots_packed_shape_e5m2_neonfp8, nk_dots_pack_e5m2_neonfp8);
    run_kernel_case("nk_dots_packed_e4m3_neonfp8", test_dots_packed<e4m3_t>, neonfp8, nk_dots_pack_size_e4m3_neonfp8,
                    nk_dots_pack_e4m3_neonfp8, nk_dots_packed_e4m3_neonfp8);
    run_kernel_case("nk_dots_pack_e4m3_neonfp8", test_dots_pack_layout<e4m3_t, host_backend_t>, neonfp8,
                    nk_dots_pack_size_e4m3_neonfp8, nk_dots_packed_shape_e4m3_neonfp8, nk_dots_pack_e4m3_neonfp8);
    run_kernel_case("nk_dots_packed_e3m2_neonfp8", test_dots_packed<e3m2_t>, neonfp8, nk_dots_pack_size_e3m2_neonfp8,
                    nk_dots_pack_e3m2_neonfp8, nk_dots_packed_e3m2_neonfp8);
    run_kernel_case("nk_dots_pack_e3m2_neonfp8", test_dots_pack_layout<e3m2_t, host_backend_t>, neonfp8,
                    nk_dots_pack_size_e3m2_neonfp8, nk_dots_packed_shape_e3m2_neonfp8, nk_dots_pack_e3m2_neonfp8);
    run_kernel_case("nk_dots_packed_e2m3_neonfp8", test_dots_packed<e2m3_t>, neonfp8, nk_dots_pack_size_e2m3_neonfp8,
                    nk_dots_pack_e2m3_neonfp8, nk_dots_packed_e2m3_neonfp8);
    run_kernel_case("nk_dots_pack_e2m3_neonfp8", test_dots_pack_layout<e2m3_t, host_backend_t>, neonfp8,
                    nk_dots_pack_size_e2m3_neonfp8, nk_dots_packed_shape_e2m3_neonfp8, nk_dots_pack_e2m3_neonfp8);
    run_kernel_case("nk_dots_packed_e2m1_neonfp8", test_dots_packed<e2m1x2_t>, neonfp8, nk_dots_pack_size_e2m1_neonfp8,
                    nk_dots_pack_e2m1_neonfp8, nk_dots_packed_e2m1_neonfp8);
    run_kernel_case("nk_dots_pack_e2m1_neonfp8", test_dots_pack_layout<e2m1x2_t, host_backend_t>, neonfp8,
                    nk_dots_pack_size_e2m1_neonfp8, nk_dots_packed_shape_e2m1_neonfp8, nk_dots_pack_e2m1_neonfp8);

    run_kernel_case("nk_dots_symmetric_e5m2_neonfp8", test_dots_symmetric<e5m2_t>, neonfp8,
                    nk_dots_symmetric_e5m2_neonfp8);
    run_kernel_case("nk_dots_symmetric_e4m3_neonfp8", test_dots_symmetric<e4m3_t>, neonfp8,
                    nk_dots_symmetric_e4m3_neonfp8);
    run_kernel_case("nk_dots_symmetric_e3m2_neonfp8", test_dots_symmetric<e3m2_t>, neonfp8,
                    nk_dots_symmetric_e3m2_neonfp8);
    run_kernel_case("nk_dots_symmetric_e2m3_neonfp8", test_dots_symmetric<e2m3_t>, neonfp8,
                    nk_dots_symmetric_e2m3_neonfp8);
    run_kernel_case("nk_dots_symmetric_e2m1_neonfp8", test_dots_symmetric<e2m1x2_t>, neonfp8,
                    nk_dots_symmetric_e2m1_neonfp8);

    run_kernel_case("nk_angulars_packed_e5m2_neonfp8", test_angulars_packed<e5m2_t>, neonfp8,
                    nk_dots_pack_size_e5m2_neonfp8, nk_dots_pack_e5m2_neonfp8, nk_angulars_packed_e5m2_neonfp8);
    run_kernel_case("nk_angulars_packed_e4m3_neonfp8", test_angulars_packed<e4m3_t>, neonfp8,
                    nk_dots_pack_size_e4m3_neonfp8, nk_dots_pack_e4m3_neonfp8, nk_angulars_packed_e4m3_neonfp8);
    run_kernel_case("nk_angulars_packed_e3m2_neonfp8", test_angulars_packed<e3m2_t>, neonfp8,
                    nk_dots_pack_size_e3m2_neonfp8, nk_dots_pack_e3m2_neonfp8, nk_angulars_packed_e3m2_neonfp8);
    run_kernel_case("nk_angulars_packed_e2m3_neonfp8", test_angulars_packed<e2m3_t>, neonfp8,
                    nk_dots_pack_size_e2m3_neonfp8, nk_dots_pack_e2m3_neonfp8, nk_angulars_packed_e2m3_neonfp8);
    run_kernel_case("nk_angulars_packed_e2m1_neonfp8", test_angulars_packed<e2m1x2_t>, neonfp8,
                    nk_dots_pack_size_e2m1_neonfp8, nk_dots_pack_e2m1_neonfp8, nk_angulars_packed_e2m1_neonfp8);

    run_kernel_case("nk_angulars_symmetric_e5m2_neonfp8", test_angulars_symmetric<e5m2_t>, neonfp8,
                    nk_angulars_symmetric_e5m2_neonfp8);
    run_kernel_case("nk_angulars_symmetric_e4m3_neonfp8", test_angulars_symmetric<e4m3_t>, neonfp8,
                    nk_angulars_symmetric_e4m3_neonfp8);
    run_kernel_case("nk_angulars_symmetric_e3m2_neonfp8", test_angulars_symmetric<e3m2_t>, neonfp8,
                    nk_angulars_symmetric_e3m2_neonfp8);
    run_kernel_case("nk_angulars_symmetric_e2m3_neonfp8", test_angulars_symmetric<e2m3_t>, neonfp8,
                    nk_angulars_symmetric_e2m3_neonfp8);
    run_kernel_case("nk_angulars_symmetric_e2m1_neonfp8", test_angulars_symmetric<e2m1x2_t>, neonfp8,
                    nk_angulars_symmetric_e2m1_neonfp8);

    run_kernel_case("nk_euclideans_packed_e5m2_neonfp8", test_euclideans_packed<e5m2_t>, neonfp8,
                    nk_dots_pack_size_e5m2_neonfp8, nk_dots_pack_e5m2_neonfp8, nk_euclideans_packed_e5m2_neonfp8);
    run_kernel_case("nk_euclideans_packed_e4m3_neonfp8", test_euclideans_packed<e4m3_t>, neonfp8,
                    nk_dots_pack_size_e4m3_neonfp8, nk_dots_pack_e4m3_neonfp8, nk_euclideans_packed_e4m3_neonfp8);
    run_kernel_case("nk_euclideans_packed_e3m2_neonfp8", test_euclideans_packed<e3m2_t>, neonfp8,
                    nk_dots_pack_size_e3m2_neonfp8, nk_dots_pack_e3m2_neonfp8, nk_euclideans_packed_e3m2_neonfp8);
    run_kernel_case("nk_euclideans_packed_e2m3_neonfp8", test_euclideans_packed<e2m3_t>, neonfp8,
                    nk_dots_pack_size_e2m3_neonfp8, nk_dots_pack_e2m3_neonfp8, nk_euclideans_packed_e2m3_neonfp8);
    run_kernel_case("nk_euclideans_packed_e2m1_neonfp8", test_euclideans_packed<e2m1x2_t>, neonfp8,
                    nk_dots_pack_size_e2m1_neonfp8, nk_dots_pack_e2m1_neonfp8, nk_euclideans_packed_e2m1_neonfp8);

    run_kernel_case("nk_euclideans_symmetric_e5m2_neonfp8", test_euclideans_symmetric<e5m2_t>, neonfp8,
                    nk_euclideans_symmetric_e5m2_neonfp8);
    run_kernel_case("nk_euclideans_symmetric_e4m3_neonfp8", test_euclideans_symmetric<e4m3_t>, neonfp8,
                    nk_euclideans_symmetric_e4m3_neonfp8);
    run_kernel_case("nk_euclideans_symmetric_e3m2_neonfp8", test_euclideans_symmetric<e3m2_t>, neonfp8,
                    nk_euclideans_symmetric_e3m2_neonfp8);
    run_kernel_case("nk_euclideans_symmetric_e2m3_neonfp8", test_euclideans_symmetric<e2m3_t>, neonfp8,
                    nk_euclideans_symmetric_e2m3_neonfp8);
    run_kernel_case("nk_euclideans_symmetric_e2m1_neonfp8", test_euclideans_symmetric<e2m1x2_t>, neonfp8,
                    nk_euclideans_symmetric_e2m1_neonfp8);
#endif // NUMKONG_TARGET_NEONFP8

#if NUMKONG_TARGET_SME
    auto const sme = nk_cap_sme_k;
    run_kernel_case("nk_dots_packed_bf16_sme", test_dots_packed<bf16_t>, sme, nk_dots_pack_size_bf16_sme,
                    nk_dots_pack_bf16_sme, nk_dots_packed_bf16_sme);
    run_kernel_case("nk_dots_pack_bf16_sme", test_dots_pack_layout<bf16_t, host_backend_t>, sme,
                    nk_dots_pack_size_bf16_sme, nk_dots_packed_shape_bf16_sme, nk_dots_pack_bf16_sme);
    run_kernel_case("nk_dots_packed_f16_sme", test_dots_packed<f16_t>, sme, nk_dots_pack_size_f16_sme,
                    nk_dots_pack_f16_sme, nk_dots_packed_f16_sme);
    run_kernel_case("nk_dots_pack_f16_sme", test_dots_pack_layout<f16_t, host_backend_t>, sme,
                    nk_dots_pack_size_f16_sme, nk_dots_packed_shape_f16_sme, nk_dots_pack_f16_sme);
    run_kernel_case("nk_dots_packed_e5m2_sme", test_dots_packed<e5m2_t>, sme, nk_dots_pack_size_e5m2_sme,
                    nk_dots_pack_e5m2_sme, nk_dots_packed_e5m2_sme);
    run_kernel_case("nk_dots_pack_e5m2_sme", test_dots_pack_layout<e5m2_t, host_backend_t>, sme,
                    nk_dots_pack_size_e5m2_sme, nk_dots_packed_shape_e5m2_sme, nk_dots_pack_e5m2_sme);
    run_kernel_case("nk_dots_packed_e4m3_sme", test_dots_packed<e4m3_t>, sme, nk_dots_pack_size_e4m3_sme,
                    nk_dots_pack_e4m3_sme, nk_dots_packed_e4m3_sme);
    run_kernel_case("nk_dots_pack_e4m3_sme", test_dots_pack_layout<e4m3_t, host_backend_t>, sme,
                    nk_dots_pack_size_e4m3_sme, nk_dots_packed_shape_e4m3_sme, nk_dots_pack_e4m3_sme);
    run_kernel_case("nk_dots_packed_e3m2_sme", test_dots_packed<e3m2_t>, sme, nk_dots_pack_size_e3m2_sme,
                    nk_dots_pack_e3m2_sme, nk_dots_packed_e3m2_sme);
    run_kernel_case("nk_dots_pack_e3m2_sme", test_dots_pack_layout<e3m2_t, host_backend_t>, sme,
                    nk_dots_pack_size_e3m2_sme, nk_dots_packed_shape_e3m2_sme, nk_dots_pack_e3m2_sme);
    run_kernel_case("nk_dots_packed_e2m3_sme", test_dots_packed<e2m3_t>, sme, nk_dots_pack_size_e2m3_sme,
                    nk_dots_pack_e2m3_sme, nk_dots_packed_e2m3_sme);
    run_kernel_case("nk_dots_pack_e2m3_sme", test_dots_pack_layout<e2m3_t, host_backend_t>, sme,
                    nk_dots_pack_size_e2m3_sme, nk_dots_packed_shape_e2m3_sme, nk_dots_pack_e2m3_sme);
    run_kernel_case("nk_dots_packed_e2m1_sme", test_dots_packed<e2m1x2_t>, sme, nk_dots_pack_size_e2m1_sme,
                    nk_dots_pack_e2m1_sme, nk_dots_packed_e2m1_sme);
    run_kernel_case("nk_dots_pack_e2m1_sme", test_dots_pack_layout<e2m1x2_t, host_backend_t>, sme,
                    nk_dots_pack_size_e2m1_sme, nk_dots_packed_shape_e2m1_sme, nk_dots_pack_e2m1_sme);
    run_kernel_case("nk_dots_packed_i8_sme", test_dots_packed<i8_t>, sme, nk_dots_pack_size_i8_sme, nk_dots_pack_i8_sme,
                    nk_dots_packed_i8_sme);
    run_kernel_case("nk_dots_pack_i8_sme", test_dots_pack_layout<i8_t, host_backend_t>, sme, nk_dots_pack_size_i8_sme,
                    nk_dots_packed_shape_i8_sme, nk_dots_pack_i8_sme);
    run_kernel_case("nk_dots_packed_i4_sme", test_dots_packed<i4x2_t>, sme, nk_dots_pack_size_i4_sme,
                    nk_dots_pack_i4_sme, nk_dots_packed_i4_sme);
    run_kernel_case("nk_dots_pack_i4_sme", test_dots_pack_layout<i4x2_t, host_backend_t>, sme, nk_dots_pack_size_i4_sme,
                    nk_dots_packed_shape_i4_sme, nk_dots_pack_i4_sme);
    run_kernel_case("nk_dots_packed_u8_sme", test_dots_packed<u8_t>, sme, nk_dots_pack_size_u8_sme, nk_dots_pack_u8_sme,
                    nk_dots_packed_u8_sme);
    run_kernel_case("nk_dots_pack_u8_sme", test_dots_pack_layout<u8_t, host_backend_t>, sme, nk_dots_pack_size_u8_sme,
                    nk_dots_packed_shape_u8_sme, nk_dots_pack_u8_sme);
    run_kernel_case("nk_dots_packed_u4_sme", test_dots_packed<u4x2_t>, sme, nk_dots_pack_size_u4_sme,
                    nk_dots_pack_u4_sme, nk_dots_packed_u4_sme);
    run_kernel_case("nk_dots_pack_u4_sme", test_dots_pack_layout<u4x2_t, host_backend_t>, sme, nk_dots_pack_size_u4_sme,
                    nk_dots_packed_shape_u4_sme, nk_dots_pack_u4_sme);

    run_kernel_case("nk_dots_symmetric_bf16_sme", test_dots_symmetric<bf16_t>, sme, nk_dots_symmetric_bf16_sme);
    run_kernel_case("nk_dots_symmetric_f16_sme", test_dots_symmetric<f16_t>, sme, nk_dots_symmetric_f16_sme);
    run_kernel_case("nk_dots_symmetric_e5m2_sme", test_dots_symmetric<e5m2_t>, sme, nk_dots_symmetric_e5m2_sme);
    run_kernel_case("nk_dots_symmetric_e4m3_sme", test_dots_symmetric<e4m3_t>, sme, nk_dots_symmetric_e4m3_sme);
    run_kernel_case("nk_dots_symmetric_e3m2_sme", test_dots_symmetric<e3m2_t>, sme, nk_dots_symmetric_e3m2_sme);
    run_kernel_case("nk_dots_symmetric_e2m3_sme", test_dots_symmetric<e2m3_t>, sme, nk_dots_symmetric_e2m3_sme);
    run_kernel_case("nk_dots_symmetric_e2m1_sme", test_dots_symmetric<e2m1x2_t>, sme, nk_dots_symmetric_e2m1_sme);
    run_kernel_case("nk_dots_symmetric_i8_sme", test_dots_symmetric<i8_t>, sme, nk_dots_symmetric_i8_sme);
    run_kernel_case("nk_dots_symmetric_i4_sme", test_dots_symmetric<i4x2_t>, sme, nk_dots_symmetric_i4_sme);
    run_kernel_case("nk_dots_symmetric_u8_sme", test_dots_symmetric<u8_t>, sme, nk_dots_symmetric_u8_sme);
    run_kernel_case("nk_dots_symmetric_u4_sme", test_dots_symmetric<u4x2_t>, sme, nk_dots_symmetric_u4_sme);

    run_kernel_case("nk_angulars_packed_bf16_sme", test_angulars_packed<bf16_t>, sme, nk_dots_pack_size_bf16_sme,
                    nk_dots_pack_bf16_sme, nk_angulars_packed_bf16_sme);
    run_kernel_case("nk_angulars_packed_f16_sme", test_angulars_packed<f16_t>, sme, nk_dots_pack_size_f16_sme,
                    nk_dots_pack_f16_sme, nk_angulars_packed_f16_sme);
    run_kernel_case("nk_angulars_packed_e5m2_sme", test_angulars_packed<e5m2_t>, sme, nk_dots_pack_size_e5m2_sme,
                    nk_dots_pack_e5m2_sme, nk_angulars_packed_e5m2_sme);
    run_kernel_case("nk_angulars_packed_e4m3_sme", test_angulars_packed<e4m3_t>, sme, nk_dots_pack_size_e4m3_sme,
                    nk_dots_pack_e4m3_sme, nk_angulars_packed_e4m3_sme);
    run_kernel_case("nk_angulars_packed_e3m2_sme", test_angulars_packed<e3m2_t>, sme, nk_dots_pack_size_e3m2_sme,
                    nk_dots_pack_e3m2_sme, nk_angulars_packed_e3m2_sme);
    run_kernel_case("nk_angulars_packed_e2m3_sme", test_angulars_packed<e2m3_t>, sme, nk_dots_pack_size_e2m3_sme,
                    nk_dots_pack_e2m3_sme, nk_angulars_packed_e2m3_sme);
    run_kernel_case("nk_angulars_packed_e2m1_sme", test_angulars_packed<e2m1x2_t>, sme, nk_dots_pack_size_e2m1_sme,
                    nk_dots_pack_e2m1_sme, nk_angulars_packed_e2m1_sme);
    run_kernel_case("nk_angulars_packed_i8_sme", test_angulars_packed<i8_t>, sme, nk_dots_pack_size_i8_sme,
                    nk_dots_pack_i8_sme, nk_angulars_packed_i8_sme);
    run_kernel_case("nk_angulars_packed_i4_sme", test_angulars_packed<i4x2_t>, sme, nk_dots_pack_size_i4_sme,
                    nk_dots_pack_i4_sme, nk_angulars_packed_i4_sme);
    run_kernel_case("nk_angulars_packed_u8_sme", test_angulars_packed<u8_t>, sme, nk_dots_pack_size_u8_sme,
                    nk_dots_pack_u8_sme, nk_angulars_packed_u8_sme);
    run_kernel_case("nk_angulars_packed_u4_sme", test_angulars_packed<u4x2_t>, sme, nk_dots_pack_size_u4_sme,
                    nk_dots_pack_u4_sme, nk_angulars_packed_u4_sme);

    run_kernel_case("nk_angulars_symmetric_bf16_sme", test_angulars_symmetric<bf16_t>, sme,
                    nk_angulars_symmetric_bf16_sme);
    run_kernel_case("nk_angulars_symmetric_f16_sme", test_angulars_symmetric<f16_t>, sme,
                    nk_angulars_symmetric_f16_sme);
    run_kernel_case("nk_angulars_symmetric_e5m2_sme", test_angulars_symmetric<e5m2_t>, sme,
                    nk_angulars_symmetric_e5m2_sme);
    run_kernel_case("nk_angulars_symmetric_e4m3_sme", test_angulars_symmetric<e4m3_t>, sme,
                    nk_angulars_symmetric_e4m3_sme);
    run_kernel_case("nk_angulars_symmetric_e3m2_sme", test_angulars_symmetric<e3m2_t>, sme,
                    nk_angulars_symmetric_e3m2_sme);
    run_kernel_case("nk_angulars_symmetric_e2m3_sme", test_angulars_symmetric<e2m3_t>, sme,
                    nk_angulars_symmetric_e2m3_sme);
    run_kernel_case("nk_angulars_symmetric_e2m1_sme", test_angulars_symmetric<e2m1x2_t>, sme,
                    nk_angulars_symmetric_e2m1_sme);
    run_kernel_case("nk_angulars_symmetric_i8_sme", test_angulars_symmetric<i8_t>, sme, nk_angulars_symmetric_i8_sme);
    run_kernel_case("nk_angulars_symmetric_i4_sme", test_angulars_symmetric<i4x2_t>, sme, nk_angulars_symmetric_i4_sme);
    run_kernel_case("nk_angulars_symmetric_u8_sme", test_angulars_symmetric<u8_t>, sme, nk_angulars_symmetric_u8_sme);
    run_kernel_case("nk_angulars_symmetric_u4_sme", test_angulars_symmetric<u4x2_t>, sme, nk_angulars_symmetric_u4_sme);

    run_kernel_case("nk_euclideans_packed_bf16_sme", test_euclideans_packed<bf16_t>, sme, nk_dots_pack_size_bf16_sme,
                    nk_dots_pack_bf16_sme, nk_euclideans_packed_bf16_sme);
    run_kernel_case("nk_euclideans_packed_f16_sme", test_euclideans_packed<f16_t>, sme, nk_dots_pack_size_f16_sme,
                    nk_dots_pack_f16_sme, nk_euclideans_packed_f16_sme);
    run_kernel_case("nk_euclideans_packed_e5m2_sme", test_euclideans_packed<e5m2_t>, sme, nk_dots_pack_size_e5m2_sme,
                    nk_dots_pack_e5m2_sme, nk_euclideans_packed_e5m2_sme);
    run_kernel_case("nk_euclideans_packed_e4m3_sme", test_euclideans_packed<e4m3_t>, sme, nk_dots_pack_size_e4m3_sme,
                    nk_dots_pack_e4m3_sme, nk_euclideans_packed_e4m3_sme);
    run_kernel_case("nk_euclideans_packed_e3m2_sme", test_euclideans_packed<e3m2_t>, sme, nk_dots_pack_size_e3m2_sme,
                    nk_dots_pack_e3m2_sme, nk_euclideans_packed_e3m2_sme);
    run_kernel_case("nk_euclideans_packed_e2m3_sme", test_euclideans_packed<e2m3_t>, sme, nk_dots_pack_size_e2m3_sme,
                    nk_dots_pack_e2m3_sme, nk_euclideans_packed_e2m3_sme);
    run_kernel_case("nk_euclideans_packed_e2m1_sme", test_euclideans_packed<e2m1x2_t>, sme, nk_dots_pack_size_e2m1_sme,
                    nk_dots_pack_e2m1_sme, nk_euclideans_packed_e2m1_sme);
    run_kernel_case("nk_euclideans_packed_i8_sme", test_euclideans_packed<i8_t>, sme, nk_dots_pack_size_i8_sme,
                    nk_dots_pack_i8_sme, nk_euclideans_packed_i8_sme);
    run_kernel_case("nk_euclideans_packed_i4_sme", test_euclideans_packed<i4x2_t>, sme, nk_dots_pack_size_i4_sme,
                    nk_dots_pack_i4_sme, nk_euclideans_packed_i4_sme);
    run_kernel_case("nk_euclideans_packed_u8_sme", test_euclideans_packed<u8_t>, sme, nk_dots_pack_size_u8_sme,
                    nk_dots_pack_u8_sme, nk_euclideans_packed_u8_sme);
    run_kernel_case("nk_euclideans_packed_u4_sme", test_euclideans_packed<u4x2_t>, sme, nk_dots_pack_size_u4_sme,
                    nk_dots_pack_u4_sme, nk_euclideans_packed_u4_sme);

    run_kernel_case("nk_euclideans_symmetric_bf16_sme", test_euclideans_symmetric<bf16_t>, sme,
                    nk_euclideans_symmetric_bf16_sme);
    run_kernel_case("nk_euclideans_symmetric_f16_sme", test_euclideans_symmetric<f16_t>, sme,
                    nk_euclideans_symmetric_f16_sme);
    run_kernel_case("nk_euclideans_symmetric_e5m2_sme", test_euclideans_symmetric<e5m2_t>, sme,
                    nk_euclideans_symmetric_e5m2_sme);
    run_kernel_case("nk_euclideans_symmetric_e4m3_sme", test_euclideans_symmetric<e4m3_t>, sme,
                    nk_euclideans_symmetric_e4m3_sme);
    run_kernel_case("nk_euclideans_symmetric_e3m2_sme", test_euclideans_symmetric<e3m2_t>, sme,
                    nk_euclideans_symmetric_e3m2_sme);
    run_kernel_case("nk_euclideans_symmetric_e2m3_sme", test_euclideans_symmetric<e2m3_t>, sme,
                    nk_euclideans_symmetric_e2m3_sme);
    run_kernel_case("nk_euclideans_symmetric_e2m1_sme", test_euclideans_symmetric<e2m1x2_t>, sme,
                    nk_euclideans_symmetric_e2m1_sme);
    run_kernel_case("nk_euclideans_symmetric_i8_sme", test_euclideans_symmetric<i8_t>, sme,
                    nk_euclideans_symmetric_i8_sme);
    run_kernel_case("nk_euclideans_symmetric_i4_sme", test_euclideans_symmetric<i4x2_t>, sme,
                    nk_euclideans_symmetric_i4_sme);
    run_kernel_case("nk_euclideans_symmetric_u8_sme", test_euclideans_symmetric<u8_t>, sme,
                    nk_euclideans_symmetric_u8_sme);
    run_kernel_case("nk_euclideans_symmetric_u4_sme", test_euclideans_symmetric<u4x2_t>, sme,
                    nk_euclideans_symmetric_u4_sme);

    run_kernel_case("nk_dots_packed_nvfp4_sme", test_dots_packed<nvfp4_t>, sme, nk_dots_pack_size_nvfp4_sme,
                    nk_dots_pack_nvfp4_sme, nk_dots_packed_nvfp4_sme);
    run_kernel_case("nk_dots_symmetric_nvfp4_sme", test_dots_symmetric<nvfp4_t>, sme, nk_dots_symmetric_nvfp4_sme);
    run_kernel_case("nk_angulars_packed_nvfp4_sme", test_angulars_packed<nvfp4_t>, sme, nk_dots_pack_size_nvfp4_sme,
                    nk_dots_pack_nvfp4_sme, nk_angulars_packed_nvfp4_sme);
    run_kernel_case("nk_angulars_symmetric_nvfp4_sme", test_angulars_symmetric<nvfp4_t>, sme,
                    nk_angulars_symmetric_nvfp4_sme);
    run_kernel_case("nk_euclideans_packed_nvfp4_sme", test_euclideans_packed<nvfp4_t>, sme, nk_dots_pack_size_nvfp4_sme,
                    nk_dots_pack_nvfp4_sme, nk_euclideans_packed_nvfp4_sme);
    run_kernel_case("nk_euclideans_symmetric_nvfp4_sme", test_euclideans_symmetric<nvfp4_t>, sme,
                    nk_euclideans_symmetric_nvfp4_sme);
    run_kernel_case("nk_dots_packed_mxfp4_sme", test_dots_packed<mxfp4_t>, sme, nk_dots_pack_size_mxfp4_sme,
                    nk_dots_pack_mxfp4_sme, nk_dots_packed_mxfp4_sme);
    run_kernel_case("nk_dots_symmetric_mxfp4_sme", test_dots_symmetric<mxfp4_t>, sme, nk_dots_symmetric_mxfp4_sme);
    run_kernel_case("nk_angulars_packed_mxfp4_sme", test_angulars_packed<mxfp4_t>, sme, nk_dots_pack_size_mxfp4_sme,
                    nk_dots_pack_mxfp4_sme, nk_angulars_packed_mxfp4_sme);
    run_kernel_case("nk_angulars_symmetric_mxfp4_sme", test_angulars_symmetric<mxfp4_t>, sme,
                    nk_angulars_symmetric_mxfp4_sme);
    run_kernel_case("nk_euclideans_packed_mxfp4_sme", test_euclideans_packed<mxfp4_t>, sme, nk_dots_pack_size_mxfp4_sme,
                    nk_dots_pack_mxfp4_sme, nk_euclideans_packed_mxfp4_sme);
    run_kernel_case("nk_euclideans_symmetric_mxfp4_sme", test_euclideans_symmetric<mxfp4_t>, sme,
                    nk_euclideans_symmetric_mxfp4_sme);
    run_kernel_case("nk_dots_packed_mxfp6e2m3_sme", test_dots_packed<mxfp6e2m3_t>, sme, nk_dots_pack_size_mxfp6e2m3_sme,
                    nk_dots_pack_mxfp6e2m3_sme, nk_dots_packed_mxfp6e2m3_sme);
    run_kernel_case("nk_dots_symmetric_mxfp6e2m3_sme", test_dots_symmetric<mxfp6e2m3_t>, sme,
                    nk_dots_symmetric_mxfp6e2m3_sme);
    run_kernel_case("nk_angulars_packed_mxfp6e2m3_sme", test_angulars_packed<mxfp6e2m3_t>, sme,
                    nk_dots_pack_size_mxfp6e2m3_sme, nk_dots_pack_mxfp6e2m3_sme, nk_angulars_packed_mxfp6e2m3_sme);
    run_kernel_case("nk_angulars_symmetric_mxfp6e2m3_sme", test_angulars_symmetric<mxfp6e2m3_t>, sme,
                    nk_angulars_symmetric_mxfp6e2m3_sme);
    run_kernel_case("nk_euclideans_packed_mxfp6e2m3_sme", test_euclideans_packed<mxfp6e2m3_t>, sme,
                    nk_dots_pack_size_mxfp6e2m3_sme, nk_dots_pack_mxfp6e2m3_sme, nk_euclideans_packed_mxfp6e2m3_sme);
    run_kernel_case("nk_euclideans_symmetric_mxfp6e2m3_sme", test_euclideans_symmetric<mxfp6e2m3_t>, sme,
                    nk_euclideans_symmetric_mxfp6e2m3_sme);
    run_kernel_case("nk_dots_packed_mxfp6e3m2_sme", test_dots_packed<mxfp6e3m2_t>, sme, nk_dots_pack_size_mxfp6e3m2_sme,
                    nk_dots_pack_mxfp6e3m2_sme, nk_dots_packed_mxfp6e3m2_sme);
    run_kernel_case("nk_dots_symmetric_mxfp6e3m2_sme", test_dots_symmetric<mxfp6e3m2_t>, sme,
                    nk_dots_symmetric_mxfp6e3m2_sme);
    run_kernel_case("nk_angulars_packed_mxfp6e3m2_sme", test_angulars_packed<mxfp6e3m2_t>, sme,
                    nk_dots_pack_size_mxfp6e3m2_sme, nk_dots_pack_mxfp6e3m2_sme, nk_angulars_packed_mxfp6e3m2_sme);
    run_kernel_case("nk_angulars_symmetric_mxfp6e3m2_sme", test_angulars_symmetric<mxfp6e3m2_t>, sme,
                    nk_angulars_symmetric_mxfp6e3m2_sme);
    run_kernel_case("nk_euclideans_packed_mxfp6e3m2_sme", test_euclideans_packed<mxfp6e3m2_t>, sme,
                    nk_dots_pack_size_mxfp6e3m2_sme, nk_dots_pack_mxfp6e3m2_sme, nk_euclideans_packed_mxfp6e3m2_sme);
    run_kernel_case("nk_euclideans_symmetric_mxfp6e3m2_sme", test_euclideans_symmetric<mxfp6e3m2_t>, sme,
                    nk_euclideans_symmetric_mxfp6e3m2_sme);
    run_kernel_case("nk_dots_packed_mxfp8e4m3_sme", test_dots_packed<mxfp8e4m3_t>, sme, nk_dots_pack_size_mxfp8e4m3_sme,
                    nk_dots_pack_mxfp8e4m3_sme, nk_dots_packed_mxfp8e4m3_sme);
    run_kernel_case("nk_dots_symmetric_mxfp8e4m3_sme", test_dots_symmetric<mxfp8e4m3_t>, sme,
                    nk_dots_symmetric_mxfp8e4m3_sme);
    run_kernel_case("nk_angulars_packed_mxfp8e4m3_sme", test_angulars_packed<mxfp8e4m3_t>, sme,
                    nk_dots_pack_size_mxfp8e4m3_sme, nk_dots_pack_mxfp8e4m3_sme, nk_angulars_packed_mxfp8e4m3_sme);
    run_kernel_case("nk_angulars_symmetric_mxfp8e4m3_sme", test_angulars_symmetric<mxfp8e4m3_t>, sme,
                    nk_angulars_symmetric_mxfp8e4m3_sme);
    run_kernel_case("nk_euclideans_packed_mxfp8e4m3_sme", test_euclideans_packed<mxfp8e4m3_t>, sme,
                    nk_dots_pack_size_mxfp8e4m3_sme, nk_dots_pack_mxfp8e4m3_sme, nk_euclideans_packed_mxfp8e4m3_sme);
    run_kernel_case("nk_euclideans_symmetric_mxfp8e4m3_sme", test_euclideans_symmetric<mxfp8e4m3_t>, sme,
                    nk_euclideans_symmetric_mxfp8e4m3_sme);
    run_kernel_case("nk_dots_packed_mxfp8e5m2_sme", test_dots_packed<mxfp8e5m2_t>, sme, nk_dots_pack_size_mxfp8e5m2_sme,
                    nk_dots_pack_mxfp8e5m2_sme, nk_dots_packed_mxfp8e5m2_sme);
    run_kernel_case("nk_dots_symmetric_mxfp8e5m2_sme", test_dots_symmetric<mxfp8e5m2_t>, sme,
                    nk_dots_symmetric_mxfp8e5m2_sme);
    run_kernel_case("nk_angulars_packed_mxfp8e5m2_sme", test_angulars_packed<mxfp8e5m2_t>, sme,
                    nk_dots_pack_size_mxfp8e5m2_sme, nk_dots_pack_mxfp8e5m2_sme, nk_angulars_packed_mxfp8e5m2_sme);
    run_kernel_case("nk_angulars_symmetric_mxfp8e5m2_sme", test_angulars_symmetric<mxfp8e5m2_t>, sme,
                    nk_angulars_symmetric_mxfp8e5m2_sme);
    run_kernel_case("nk_euclideans_packed_mxfp8e5m2_sme", test_euclideans_packed<mxfp8e5m2_t>, sme,
                    nk_dots_pack_size_mxfp8e5m2_sme, nk_dots_pack_mxfp8e5m2_sme, nk_euclideans_packed_mxfp8e5m2_sme);
    run_kernel_case("nk_euclideans_symmetric_mxfp8e5m2_sme", test_euclideans_symmetric<mxfp8e5m2_t>, sme,
                    nk_euclideans_symmetric_mxfp8e5m2_sme);

    run_kernel_case("nk_attention_packed_bf16_sme", test_attention_packed<bf16_t>, sme, nk_attention_pack_size_bf16_sme,
                    nk_attention_pack_bf16_sme, nk_attention_packed_bf16_sme);
    run_kernel_case("nk_attention_packed_f16_sme", test_attention_packed<f16_t>, sme, nk_attention_pack_size_f16_sme,
                    nk_attention_pack_f16_sme, nk_attention_packed_f16_sme);
    run_kernel_case("nk_attention_packed_e4m3_sme", test_attention_packed<e4m3_t>, sme, nk_attention_pack_size_e4m3_sme,
                    nk_attention_pack_e4m3_sme, nk_attention_packed_e4m3_sme);
    run_kernel_case("nk_attention_packed_i8_sme", test_attention_packed<i8_t>, sme, nk_attention_pack_size_i8_sme,
                    nk_attention_pack_i8_sme, nk_attention_packed_i8_sme);
    run_kernel_case("nk_attention_packed_nvfp4_sme", test_attention_packed<nvfp4_t>, sme,
                    nk_attention_pack_size_nvfp4_sme, nk_attention_pack_nvfp4_sme, nk_attention_packed_nvfp4_sme);
    run_kernel_case("nk_attention_packed_mxfp4_sme", test_attention_packed<mxfp4_t>, sme,
                    nk_attention_pack_size_mxfp4_sme, nk_attention_pack_mxfp4_sme, nk_attention_packed_mxfp4_sme);
    run_kernel_case("nk_attention_packed_mxfp6e2m3_sme", test_attention_packed<mxfp6e2m3_t>, sme,
                    nk_attention_pack_size_mxfp6e2m3_sme, nk_attention_pack_mxfp6e2m3_sme,
                    nk_attention_packed_mxfp6e2m3_sme);
    run_kernel_case("nk_attention_packed_mxfp6e3m2_sme", test_attention_packed<mxfp6e3m2_t>, sme,
                    nk_attention_pack_size_mxfp6e3m2_sme, nk_attention_pack_mxfp6e3m2_sme,
                    nk_attention_packed_mxfp6e3m2_sme);
    run_kernel_case("nk_attention_packed_mxfp8e4m3_sme", test_attention_packed<mxfp8e4m3_t>, sme,
                    nk_attention_pack_size_mxfp8e4m3_sme, nk_attention_pack_mxfp8e4m3_sme,
                    nk_attention_packed_mxfp8e4m3_sme);
    run_kernel_case("nk_attention_packed_mxfp8e5m2_sme", test_attention_packed<mxfp8e5m2_t>, sme,
                    nk_attention_pack_size_mxfp8e5m2_sme, nk_attention_pack_mxfp8e5m2_sme,
                    nk_attention_packed_mxfp8e5m2_sme);
#endif // NUMKONG_TARGET_SME

#if NUMKONG_TARGET_SMEBI32
    auto const smebi32 = nk_cap_smebi32_k;
    run_kernel_case("nk_dots_packed_u1_smebi32", test_dots_packed<u1x8_t>, smebi32, nk_dots_pack_size_u1_smebi32,
                    nk_dots_pack_u1_smebi32, nk_dots_packed_u1_smebi32);
    run_kernel_case("nk_dots_pack_u1_smebi32", test_dots_pack_layout<u1x8_t, host_backend_t>, smebi32,
                    nk_dots_pack_size_u1_smebi32, nk_dots_packed_shape_u1_smebi32, nk_dots_pack_u1_smebi32);
    run_kernel_case("nk_dots_symmetric_u1_smebi32", test_dots_symmetric<u1x8_t>, smebi32, nk_dots_symmetric_u1_smebi32);

    run_kernel_case("nk_hammings_packed_u1_smebi32", test_hammings_packed<u1x8_t>, smebi32,
                    nk_dots_pack_size_u1_smebi32, nk_dots_pack_u1_smebi32, nk_hammings_packed_u1_smebi32);
    run_kernel_case("nk_hammings_symmetric_u1_smebi32", test_hammings_symmetric<u1x8_t>, smebi32,
                    nk_hammings_symmetric_u1_smebi32);

    run_kernel_case("nk_jaccards_packed_u1_smebi32", test_jaccards_packed<u1x8_t>, smebi32,
                    nk_dots_pack_size_u1_smebi32, nk_dots_pack_u1_smebi32, nk_jaccards_packed_u1_smebi32);
    run_kernel_case("nk_jaccards_symmetric_u1_smebi32", test_jaccards_symmetric<u1x8_t>, smebi32,
                    nk_jaccards_symmetric_u1_smebi32);
#endif // NUMKONG_TARGET_SMEBI32

#if NUMKONG_TARGET_SMEF64
    auto const smef64 = nk_cap_smef64_k;
    run_kernel_case("nk_dots_packed_f64_smef64", test_dots_packed<f64_t>, smef64, nk_dots_pack_size_f64_smef64,
                    nk_dots_pack_f64_smef64, nk_dots_packed_f64_smef64);
    run_kernel_case("nk_dots_pack_f64_smef64", test_dots_pack_layout<f64_t, host_backend_t>, smef64,
                    nk_dots_pack_size_f64_smef64, nk_dots_packed_shape_f64_smef64, nk_dots_pack_f64_smef64);
    run_kernel_case("nk_dots_packed_f32_smef64", test_dots_packed<f32_t>, smef64, nk_dots_pack_size_f32_smef64,
                    nk_dots_pack_f32_smef64, nk_dots_packed_f32_smef64);
    run_kernel_case("nk_dots_pack_f32_smef64", test_dots_pack_layout<f32_t, host_backend_t>, smef64,
                    nk_dots_pack_size_f32_smef64, nk_dots_packed_shape_f32_smef64, nk_dots_pack_f32_smef64);
    run_kernel_case("nk_dots_symmetric_f64_smef64", test_dots_symmetric<f64_t>, smef64, nk_dots_symmetric_f64_smef64);
    run_kernel_case("nk_dots_symmetric_f32_smef64", test_dots_symmetric<f32_t>, smef64, nk_dots_symmetric_f32_smef64);

    run_kernel_case("nk_angulars_packed_f64_smef64", test_angulars_packed<f64_t>, smef64, nk_dots_pack_size_f64_smef64,
                    nk_dots_pack_f64_smef64, nk_angulars_packed_f64_smef64);
    run_kernel_case("nk_angulars_packed_f32_smef64", test_angulars_packed<f32_t>, smef64, nk_dots_pack_size_f32_smef64,
                    nk_dots_pack_f32_smef64, nk_angulars_packed_f32_smef64);

    run_kernel_case("nk_angulars_symmetric_f64_smef64", test_angulars_symmetric<f64_t>, smef64,
                    nk_angulars_symmetric_f64_smef64);
    run_kernel_case("nk_angulars_symmetric_f32_smef64", test_angulars_symmetric<f32_t>, smef64,
                    nk_angulars_symmetric_f32_smef64);

    run_kernel_case("nk_euclideans_packed_f64_smef64", test_euclideans_packed<f64_t>, smef64,
                    nk_dots_pack_size_f64_smef64, nk_dots_pack_f64_smef64, nk_euclideans_packed_f64_smef64);
    run_kernel_case("nk_euclideans_packed_f32_smef64", test_euclideans_packed<f32_t>, smef64,
                    nk_dots_pack_size_f32_smef64, nk_dots_pack_f32_smef64, nk_euclideans_packed_f32_smef64);

    run_kernel_case("nk_euclideans_symmetric_f64_smef64", test_euclideans_symmetric<f64_t>, smef64,
                    nk_euclideans_symmetric_f64_smef64);
    run_kernel_case("nk_euclideans_symmetric_f32_smef64", test_euclideans_symmetric<f32_t>, smef64,
                    nk_euclideans_symmetric_f32_smef64);
#endif // NUMKONG_TARGET_SMEF64
}

} // namespace ashvardanian::numkong::test
