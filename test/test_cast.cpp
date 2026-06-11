/**
 *  @brief Type cast tests.
 *  @file test/test_cast.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 */

#include <numeric> // `std::lcm`

#include "test.hpp"
#include "numkong/cast.h"

using cast_t = void (*)(void const *, nk_dtype_t, nk_size_t, void *, nk_dtype_t);

/**
 *  @brief Pull one logical element out of a vector as a primitive comparable value.
 *
 *  For sub-byte value types (i4x2, u4x2, u1x8, e2m1x2), `vec[i]` returns a `sub_byte_ref`
 *  whose conversion operator upcasts the nibble/bit to its natural integer type (i8/u8/bool);
 *  unary `+` triggers that conversion. For byte-sized types, the indexed wrapper struct
 *  exposes the primitive directly through `.raw_`.
 */
template <typename vec_type_>
static auto read_element_(vec_type_ const &v, std::size_t i) {
    if constexpr (nk::dimensions_per_value<typename vec_type_::value_type>() > 1) return +v[i];
    else return v[i].raw_;
}

/**
 *  @brief Test cast kernel against serial kernel.
 *  SIMD kernels must match serial output exactly for every logical element.
 */
template <typename from_type_, typename to_type_>
error_stats_t test_cast(cast_t kernel) {
    error_stats_t stats(comparison_family_t::exact_k);
    std::mt19937 generator(global_config.seed);

    // Align to lcm(dims_per_value) so both buffers land on clean storage boundaries.
    std::size_t const aligned_dims = std::lcm(nk::dimensions_per_value<from_type_>(),
                                              nk::dimensions_per_value<to_type_>());
    std::size_t const dimensions = (global_config.dense_dimensions / aligned_dims) * aligned_dims;

    auto source_vec = make_vector<from_type_>(dimensions);
    auto target_vec = make_vector<to_type_>(dimensions);
    auto reference_vec = make_vector<to_type_>(dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, source_vec);

        nk_cast_serial(source_vec.raw_values_data(), from_type_::dtype(), dimensions, reference_vec.raw_values_data(),
                       to_type_::dtype());
        kernel(source_vec.raw_values_data(), from_type_::dtype(), dimensions, target_vec.raw_values_data(),
               to_type_::dtype());

        // Per-element comparison, dispatched to the smart reference for sub-byte types.
        for (std::size_t i = 0; i < target_vec.size(); ++i)
            stats.accumulate(read_element_(target_vec, i), read_element_(reference_vec, i));
    }
    return stats;
}

using block_scaled_cast_t = void (*)(                                                         //
    void const *, void const *, nk_scalar_buffer_t const *, nk_block_scaled_format_t const *, //
    void *, void *, nk_scalar_buffer_t *, nk_block_scaled_format_t const *, nk_size_t);
using block_scaled_format_factory_t = nk_block_scaled_format_t (*)(void);

/**
 *  @brief Test block-scaled cast kernel against the serial reference.
 *  Direct analog of `test_cast`: SIMD kernels must match serial output exactly for both
 *  encoded elements and per-block scales.
 */
error_stats_t test_cast_block_scaled(block_scaled_cast_t kernel, block_scaled_format_factory_t factory) {
    error_stats_t stats(comparison_family_t::exact_k);
    std::mt19937 generator(global_config.seed);

    nk_block_scaled_format_t const target_format = factory();
    nk_block_scaled_format_t const plain_f32_format = nk_plain(nk_f32_k);
    std::size_t const dimensions = (global_config.dense_dimensions / target_format.block_size) *
                                   target_format.block_size;
    bool const has_tensor_scale = (target_format.tensor_scale_dtype == nk_f32_k);

    auto source_vec = make_vector<f32_t>(dimensions);
    auto target_elements_vec = make_vector<u8_t>(nk_block_scaled_elements_size(dimensions, target_format));
    auto target_scales_vec = make_vector<u8_t>(nk_block_scaled_scales_size(dimensions, target_format));
    auto reference_elements_vec = make_vector<u8_t>(nk_block_scaled_elements_size(dimensions, target_format));
    auto reference_scales_vec = make_vector<u8_t>(nk_block_scaled_scales_size(dimensions, target_format));

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, source_vec);

        // Pre-populated global so both kernels skip the auto-derive calibration path.
        nk_scalar_buffer_t tensor_scale_target = {}, tensor_scale_reference = {};
        tensor_scale_target.f32 = 1.0f;
        tensor_scale_reference.f32 = 1.0f;

        nk_cast_block_scaled_serial(                                                          //
            source_vec.raw_values_data(), nullptr, nullptr, &plain_f32_format,                //
            reference_elements_vec.raw_values_data(), reference_scales_vec.raw_values_data(), //
            has_tensor_scale ? &tensor_scale_reference : nullptr, &target_format, dimensions);
        kernel(                                                                         //
            source_vec.raw_values_data(), nullptr, nullptr, &plain_f32_format,          //
            target_elements_vec.raw_values_data(), target_scales_vec.raw_values_data(), //
            has_tensor_scale ? &tensor_scale_target : nullptr, &target_format, dimensions);

        auto const *target_elements_raw = target_elements_vec.raw_values_data();
        auto const *reference_elements_raw = reference_elements_vec.raw_values_data();
        for (std::size_t i = 0; i < target_elements_vec.size_values(); ++i)
            stats.accumulate(target_elements_raw[i], reference_elements_raw[i]);

        auto const *target_scales_raw = target_scales_vec.raw_values_data();
        auto const *reference_scales_raw = reference_scales_vec.raw_values_data();
        for (std::size_t i = 0; i < target_scales_vec.size_values(); ++i)
            stats.accumulate(target_scales_raw[i], reference_scales_raw[i]);
    }
    return stats;
}

void test_casts() {
    error_stats_section_t check("Type Casts");

#if NK_DYNAMIC_DISPATCH
    check("cast_f32_to_f16", test_cast<f32_t, f16_t>, nk_cast);
    check("cast_f16_to_f32", test_cast<f16_t, f32_t>, nk_cast);
    check("cast_f32_to_bf16", test_cast<f32_t, bf16_t>, nk_cast);
    check("cast_bf16_to_f32", test_cast<bf16_t, f32_t>, nk_cast);
    check("cast_f32_to_e4m3", test_cast<f32_t, e4m3_t>, nk_cast);
    check("cast_e4m3_to_f32", test_cast<e4m3_t, f32_t>, nk_cast);
    check("cast_f32_to_e5m2", test_cast<f32_t, e5m2_t>, nk_cast);
    check("cast_e5m2_to_f32", test_cast<e5m2_t, f32_t>, nk_cast);
    check("cast_f32_to_e2m3", test_cast<f32_t, e2m3_t>, nk_cast);
    check("cast_e2m3_to_f32", test_cast<e2m3_t, f32_t>, nk_cast);
    check("cast_f32_to_e3m2", test_cast<f32_t, e3m2_t>, nk_cast);
    check("cast_e3m2_to_f32", test_cast<e3m2_t, f32_t>, nk_cast);
    check("cast_f64_to_f32", test_cast<f64_t, f32_t>, nk_cast);
    check("cast_f32_to_f64", test_cast<f32_t, f64_t>, nk_cast);
    // Integer ↔ integer
    check("cast_i8_to_i32", test_cast<i8_t, i32_t>, nk_cast);
    check("cast_i32_to_i8", test_cast<i32_t, i8_t>, nk_cast);
    check("cast_u8_to_u32", test_cast<u8_t, u32_t>, nk_cast);
    check("cast_u32_to_u8", test_cast<u32_t, u8_t>, nk_cast);
    check("cast_i16_to_i64", test_cast<i16_t, i64_t>, nk_cast);
    check("cast_i64_to_i16", test_cast<i64_t, i16_t>, nk_cast);
    check("cast_i32_to_u32", test_cast<i32_t, u32_t>, nk_cast);
    // Integer ↔ float
    check("cast_i32_to_f64", test_cast<i32_t, f64_t>, nk_cast);
    check("cast_f64_to_i32", test_cast<f64_t, i32_t>, nk_cast);
    check("cast_i16_to_f32", test_cast<i16_t, f32_t>, nk_cast);
    check("cast_u8_to_f32", test_cast<u8_t, f32_t>, nk_cast);
    check("cast_f32_to_i8", test_cast<f32_t, i8_t>, nk_cast);
    check("cast_i8_to_f64", test_cast<i8_t, f64_t>, nk_cast);
    check("cast_f64_to_u8", test_cast<f64_t, u8_t>, nk_cast);
    // Verify serial fallbacks for rare paths
    check("cast_f64_to_f16", test_cast<f64_t, f16_t>, nk_cast);
    check("cast_f16_to_f64", test_cast<f16_t, f64_t>, nk_cast);
    check("cast_f64_to_bf16", test_cast<f64_t, bf16_t>, nk_cast);
    check("cast_bf16_to_f64", test_cast<bf16_t, f64_t>, nk_cast);
#endif

#if NK_TARGET_HASWELL
    check("cast_f32_to_f16_haswell", test_cast<f32_t, f16_t>, nk_cast_haswell);
    check("cast_f16_to_f32_haswell", test_cast<f16_t, f32_t>, nk_cast_haswell);
    check("cast_f32_to_bf16_haswell", test_cast<f32_t, bf16_t>, nk_cast_haswell);
    check("cast_bf16_to_f32_haswell", test_cast<bf16_t, f32_t>, nk_cast_haswell);
    check("cast_f32_to_e4m3_haswell", test_cast<f32_t, e4m3_t>, nk_cast_haswell);
    check("cast_e4m3_to_f32_haswell", test_cast<e4m3_t, f32_t>, nk_cast_haswell);
    check("cast_f32_to_e5m2_haswell", test_cast<f32_t, e5m2_t>, nk_cast_haswell);
    check("cast_e5m2_to_f32_haswell", test_cast<e5m2_t, f32_t>, nk_cast_haswell);
    check("cast_f32_to_e2m3_haswell", test_cast<f32_t, e2m3_t>, nk_cast_haswell);
    check("cast_e2m3_to_f32_haswell", test_cast<e2m3_t, f32_t>, nk_cast_haswell);
    check("cast_f32_to_e3m2_haswell", test_cast<f32_t, e3m2_t>, nk_cast_haswell);
    check("cast_e3m2_to_f32_haswell", test_cast<e3m2_t, f32_t>, nk_cast_haswell);
    check("cast_i8_to_f32_haswell", test_cast<i8_t, f32_t>, nk_cast_haswell);
    check("cast_f32_to_i8_haswell", test_cast<f32_t, i8_t>, nk_cast_haswell);
    check("cast_i16_to_f32_haswell", test_cast<i16_t, f32_t>, nk_cast_haswell);
    check("cast_f32_to_i16_haswell", test_cast<f32_t, i16_t>, nk_cast_haswell);
    check("cast_u16_to_f32_haswell", test_cast<u16_t, f32_t>, nk_cast_haswell);
    check("cast_f32_to_u16_haswell", test_cast<f32_t, u16_t>, nk_cast_haswell);
    check("cast_u8_to_f32_haswell", test_cast<u8_t, f32_t>, nk_cast_haswell);
    check("cast_f32_to_u8_haswell", test_cast<f32_t, u8_t>, nk_cast_haswell);
    check("cast_block_scaled_nvfp4_haswell", test_cast_block_scaled, nk_cast_block_scaled_haswell, nk_nvfp4);
    check("cast_block_scaled_mxfp4_haswell", test_cast_block_scaled, nk_cast_block_scaled_haswell, nk_mxfp4);
    check("cast_block_scaled_mxfp6_e2m3_haswell", test_cast_block_scaled, nk_cast_block_scaled_haswell, nk_mxfp6_e2m3);
    check("cast_block_scaled_mxfp6_e3m2_haswell", test_cast_block_scaled, nk_cast_block_scaled_haswell, nk_mxfp6_e3m2);
    check("cast_block_scaled_mxfp8_e4m3_haswell", test_cast_block_scaled, nk_cast_block_scaled_haswell, nk_mxfp8_e4m3);
    check("cast_block_scaled_mxfp8_e5m2_haswell", test_cast_block_scaled, nk_cast_block_scaled_haswell, nk_mxfp8_e5m2);
    check("cast_block_scaled_mxint8_haswell", test_cast_block_scaled, nk_cast_block_scaled_haswell, nk_mxint8);
    // Verify serial fallbacks for rare paths
    check("cast_i32_to_f64_haswell", test_cast<i32_t, f64_t>, nk_cast_haswell);
    check("cast_f64_to_f32_haswell", test_cast<f64_t, f32_t>, nk_cast_haswell);
#endif

#if NK_TARGET_SKYLAKE
    check("cast_f32_to_f16_skylake", test_cast<f32_t, f16_t>, nk_cast_skylake);
    check("cast_f16_to_f32_skylake", test_cast<f16_t, f32_t>, nk_cast_skylake);
    check("cast_f32_to_bf16_skylake", test_cast<f32_t, bf16_t>, nk_cast_skylake);
    check("cast_bf16_to_f32_skylake", test_cast<bf16_t, f32_t>, nk_cast_skylake);
    check("cast_f32_to_e4m3_skylake", test_cast<f32_t, e4m3_t>, nk_cast_skylake);
    check("cast_e4m3_to_f32_skylake", test_cast<e4m3_t, f32_t>, nk_cast_skylake);
    check("cast_f32_to_e5m2_skylake", test_cast<f32_t, e5m2_t>, nk_cast_skylake);
    check("cast_e5m2_to_f32_skylake", test_cast<e5m2_t, f32_t>, nk_cast_skylake);
    check("cast_f32_to_e2m3_skylake", test_cast<f32_t, e2m3_t>, nk_cast_skylake);
    check("cast_e2m3_to_f32_skylake", test_cast<e2m3_t, f32_t>, nk_cast_skylake);
    check("cast_f32_to_e3m2_skylake", test_cast<f32_t, e3m2_t>, nk_cast_skylake);
    check("cast_e3m2_to_f32_skylake", test_cast<e3m2_t, f32_t>, nk_cast_skylake);
    check("cast_block_scaled_nvfp4_skylake", test_cast_block_scaled, nk_cast_block_scaled_skylake, nk_nvfp4);
    check("cast_block_scaled_mxfp4_skylake", test_cast_block_scaled, nk_cast_block_scaled_skylake, nk_mxfp4);
    check("cast_block_scaled_mxfp6_e2m3_skylake", test_cast_block_scaled, nk_cast_block_scaled_skylake, nk_mxfp6_e2m3);
    check("cast_block_scaled_mxfp6_e3m2_skylake", test_cast_block_scaled, nk_cast_block_scaled_skylake, nk_mxfp6_e3m2);
    check("cast_block_scaled_mxfp8_e4m3_skylake", test_cast_block_scaled, nk_cast_block_scaled_skylake, nk_mxfp8_e4m3);
    check("cast_block_scaled_mxfp8_e5m2_skylake", test_cast_block_scaled, nk_cast_block_scaled_skylake, nk_mxfp8_e5m2);
    check("cast_block_scaled_mxint8_skylake", test_cast_block_scaled, nk_cast_block_scaled_skylake, nk_mxint8);
    check("cast_f16_to_bf16_skylake", test_cast<f16_t, bf16_t>, nk_cast_skylake);
    check("cast_bf16_to_f16_skylake", test_cast<bf16_t, f16_t>, nk_cast_skylake);
    check("cast_e4m3_to_f16_skylake", test_cast<e4m3_t, f16_t>, nk_cast_skylake);
    check("cast_f16_to_e4m3_skylake", test_cast<f16_t, e4m3_t>, nk_cast_skylake);
    check("cast_e5m2_to_f16_skylake", test_cast<e5m2_t, f16_t>, nk_cast_skylake);
    check("cast_f16_to_e5m2_skylake", test_cast<f16_t, e5m2_t>, nk_cast_skylake);
    check("cast_e4m3_to_bf16_skylake", test_cast<e4m3_t, bf16_t>, nk_cast_skylake);
    check("cast_bf16_to_e4m3_skylake", test_cast<bf16_t, e4m3_t>, nk_cast_skylake);
    check("cast_e5m2_to_bf16_skylake", test_cast<e5m2_t, bf16_t>, nk_cast_skylake);
    check("cast_bf16_to_e5m2_skylake", test_cast<bf16_t, e5m2_t>, nk_cast_skylake);
    check("cast_f64_to_f32_skylake", test_cast<f64_t, f32_t>, nk_cast_skylake);
    check("cast_f32_to_f64_skylake", test_cast<f32_t, f64_t>, nk_cast_skylake);
    check("cast_i32_to_f64_skylake", test_cast<i32_t, f64_t>, nk_cast_skylake);
    check("cast_f64_to_i32_skylake", test_cast<f64_t, i32_t>, nk_cast_skylake);
    check("cast_i8_to_i32_skylake", test_cast<i8_t, i32_t>, nk_cast_skylake);
    check("cast_i32_to_i8_skylake", test_cast<i32_t, i8_t>, nk_cast_skylake);
    check("cast_i16_to_f32_skylake", test_cast<i16_t, f32_t>, nk_cast_skylake);
    check("cast_f32_to_i16_skylake", test_cast<f32_t, i16_t>, nk_cast_skylake);
    check("cast_u16_to_f32_skylake", test_cast<u16_t, f32_t>, nk_cast_skylake);
    check("cast_f32_to_u16_skylake", test_cast<f32_t, u16_t>, nk_cast_skylake);
    check("cast_u8_to_f32_skylake", test_cast<u8_t, f32_t>, nk_cast_skylake);
    check("cast_f32_to_u8_skylake", test_cast<f32_t, u8_t>, nk_cast_skylake);
    check("cast_i64_to_f64_skylake", test_cast<i64_t, f64_t>, nk_cast_skylake);
    check("cast_f64_to_i64_skylake", test_cast<f64_t, i64_t>, nk_cast_skylake);
    check("cast_u64_to_f64_skylake", test_cast<u64_t, f64_t>, nk_cast_skylake);
    check("cast_f64_to_u64_skylake", test_cast<f64_t, u64_t>, nk_cast_skylake);
    check("cast_u32_to_f64_skylake", test_cast<u32_t, f64_t>, nk_cast_skylake);
    check("cast_f64_to_u32_skylake", test_cast<f64_t, u32_t>, nk_cast_skylake);
    // Verify serial fallbacks for rare paths
    check("cast_i8_to_f64_skylake", test_cast<i8_t, f64_t>, nk_cast_skylake);
    check("cast_f64_to_bf16_skylake", test_cast<f64_t, bf16_t>, nk_cast_skylake);
#endif

#if NK_TARGET_ICELAKE

    check("cast_e4m3_to_bf16_icelake", test_cast<e4m3_t, bf16_t>, nk_cast_icelake);
    check("cast_bf16_to_e4m3_icelake", test_cast<bf16_t, e4m3_t>, nk_cast_icelake);
    check("cast_e5m2_to_bf16_icelake", test_cast<e5m2_t, bf16_t>, nk_cast_icelake);
    check("cast_bf16_to_e5m2_icelake", test_cast<bf16_t, e5m2_t>, nk_cast_icelake);
    check("cast_e4m3_to_f16_icelake", test_cast<e4m3_t, f16_t>, nk_cast_icelake);
    check("cast_e5m2_to_f16_icelake", test_cast<e5m2_t, f16_t>, nk_cast_icelake);
    check("cast_e4m3_to_f32_icelake", test_cast<e4m3_t, f32_t>, nk_cast_icelake);
    check("cast_f32_to_e4m3_icelake", test_cast<f32_t, e4m3_t>, nk_cast_icelake);
    check("cast_f16_to_f32_icelake", test_cast<f16_t, f32_t>, nk_cast_icelake);
    check("cast_f32_to_f16_icelake", test_cast<f32_t, f16_t>, nk_cast_icelake);
    check("cast_e2m3_to_f32_icelake", test_cast<e2m3_t, f32_t>, nk_cast_icelake);
    check("cast_f32_to_e2m3_icelake", test_cast<f32_t, e2m3_t>, nk_cast_icelake);
    check("cast_e3m2_to_f32_icelake", test_cast<e3m2_t, f32_t>, nk_cast_icelake);
    check("cast_f32_to_e3m2_icelake", test_cast<f32_t, e3m2_t>, nk_cast_icelake);
    check("cast_block_scaled_nvfp4_icelake", test_cast_block_scaled, nk_cast_block_scaled_icelake, nk_nvfp4);
    check("cast_block_scaled_mxfp4_icelake", test_cast_block_scaled, nk_cast_block_scaled_icelake, nk_mxfp4);
    check("cast_block_scaled_mxfp6_e2m3_icelake", test_cast_block_scaled, nk_cast_block_scaled_icelake, nk_mxfp6_e2m3);
    check("cast_block_scaled_mxfp6_e3m2_icelake", test_cast_block_scaled, nk_cast_block_scaled_icelake, nk_mxfp6_e3m2);
    check("cast_block_scaled_mxfp8_e4m3_icelake", test_cast_block_scaled, nk_cast_block_scaled_icelake, nk_mxfp8_e4m3);
    check("cast_block_scaled_mxfp8_e5m2_icelake", test_cast_block_scaled, nk_cast_block_scaled_icelake, nk_mxfp8_e5m2);
    check("cast_block_scaled_mxint8_icelake", test_cast_block_scaled, nk_cast_block_scaled_icelake, nk_mxint8);
#endif

#if NK_TARGET_SAPPHIRE
    check("cast_e4m3_to_f16_sapphire", test_cast<e4m3_t, f16_t>, nk_cast_sapphire);
    check("cast_f16_to_e4m3_sapphire", test_cast<f16_t, e4m3_t>, nk_cast_sapphire);
    check("cast_e5m2_to_f16_sapphire", test_cast<e5m2_t, f16_t>, nk_cast_sapphire);
    check("cast_f16_to_e5m2_sapphire", test_cast<f16_t, e5m2_t>, nk_cast_sapphire);
    check("cast_f16_to_f32_sapphire", test_cast<f16_t, f32_t>, nk_cast_sapphire);
    check("cast_f32_to_f16_sapphire", test_cast<f32_t, f16_t>, nk_cast_sapphire);
#endif

#if NK_TARGET_NEON
    check("cast_e4m3_to_f32_neon", test_cast<e4m3_t, f32_t>, nk_cast_neon);
    check("cast_f32_to_e4m3_neon", test_cast<f32_t, e4m3_t>, nk_cast_neon);
    check("cast_e5m2_to_f32_neon", test_cast<e5m2_t, f32_t>, nk_cast_neon);
    check("cast_f32_to_e5m2_neon", test_cast<f32_t, e5m2_t>, nk_cast_neon);
#endif

#if NK_TARGET_V128RELAXED
    check("cast_f32_to_f16_v128relaxed", test_cast<f32_t, f16_t>, nk_cast_v128relaxed);
    check("cast_f16_to_f32_v128relaxed", test_cast<f16_t, f32_t>, nk_cast_v128relaxed);
    check("cast_f32_to_bf16_v128relaxed", test_cast<f32_t, bf16_t>, nk_cast_v128relaxed);
    check("cast_bf16_to_f32_v128relaxed", test_cast<bf16_t, f32_t>, nk_cast_v128relaxed);
    check("cast_f32_to_e4m3_v128relaxed", test_cast<f32_t, e4m3_t>, nk_cast_v128relaxed);
    check("cast_e4m3_to_f32_v128relaxed", test_cast<e4m3_t, f32_t>, nk_cast_v128relaxed);
    check("cast_f32_to_e5m2_v128relaxed", test_cast<f32_t, e5m2_t>, nk_cast_v128relaxed);
    check("cast_e5m2_to_f32_v128relaxed", test_cast<e5m2_t, f32_t>, nk_cast_v128relaxed);
    check("cast_f32_to_e2m3_v128relaxed", test_cast<f32_t, e2m3_t>, nk_cast_v128relaxed);
    check("cast_e2m3_to_f32_v128relaxed", test_cast<e2m3_t, f32_t>, nk_cast_v128relaxed);
    check("cast_f32_to_e3m2_v128relaxed", test_cast<f32_t, e3m2_t>, nk_cast_v128relaxed);
    check("cast_e3m2_to_f32_v128relaxed", test_cast<e3m2_t, f32_t>, nk_cast_v128relaxed);
    check("cast_i8_to_f32_v128relaxed", test_cast<i8_t, f32_t>, nk_cast_v128relaxed);
    check("cast_f32_to_i8_v128relaxed", test_cast<f32_t, i8_t>, nk_cast_v128relaxed);
    check("cast_u8_to_f32_v128relaxed", test_cast<u8_t, f32_t>, nk_cast_v128relaxed);
    check("cast_f32_to_u8_v128relaxed", test_cast<f32_t, u8_t>, nk_cast_v128relaxed);
#endif

#if NK_TARGET_RVV
    check("cast_bf16_to_f32_rvv", test_cast<bf16_t, f32_t>, nk_cast_rvv);
    check("cast_f32_to_bf16_rvv", test_cast<f32_t, bf16_t>, nk_cast_rvv);
    check("cast_e4m3_to_f32_rvv", test_cast<e4m3_t, f32_t>, nk_cast_rvv);
    check("cast_e5m2_to_f32_rvv", test_cast<e5m2_t, f32_t>, nk_cast_rvv);
#endif

#if NK_TARGET_POWERVSX
    check("cast_f32_to_f16_powervsx", test_cast<f32_t, f16_t>, nk_cast_powervsx);
    check("cast_f16_to_f32_powervsx", test_cast<f16_t, f32_t>, nk_cast_powervsx);
    check("cast_f32_to_bf16_powervsx", test_cast<f32_t, bf16_t>, nk_cast_powervsx);
    check("cast_bf16_to_f32_powervsx", test_cast<bf16_t, f32_t>, nk_cast_powervsx);
    check("cast_i8_to_f32_powervsx", test_cast<i8_t, f32_t>, nk_cast_powervsx);
    check("cast_f32_to_i8_powervsx", test_cast<f32_t, i8_t>, nk_cast_powervsx);
    check("cast_u8_to_f32_powervsx", test_cast<u8_t, f32_t>, nk_cast_powervsx);
    check("cast_f32_to_u8_powervsx", test_cast<f32_t, u8_t>, nk_cast_powervsx);
    check("cast_i16_to_f32_powervsx", test_cast<i16_t, f32_t>, nk_cast_powervsx);
    check("cast_f32_to_i16_powervsx", test_cast<f32_t, i16_t>, nk_cast_powervsx);
    check("cast_u16_to_f32_powervsx", test_cast<u16_t, f32_t>, nk_cast_powervsx);
    check("cast_f32_to_u16_powervsx", test_cast<f32_t, u16_t>, nk_cast_powervsx);
#endif

    // Serial always runs - baseline test
    check("cast_bf16_to_f32_serial", test_cast<bf16_t, f32_t>, nk_cast_serial);
    check("cast_f32_to_bf16_serial", test_cast<f32_t, bf16_t>, nk_cast_serial);
    check("cast_e4m3_to_f32_serial", test_cast<e4m3_t, f32_t>, nk_cast_serial);
    check("cast_f32_to_e4m3_serial", test_cast<f32_t, e4m3_t>, nk_cast_serial);
    check("cast_e5m2_to_f32_serial", test_cast<e5m2_t, f32_t>, nk_cast_serial);
    check("cast_f32_to_e5m2_serial", test_cast<f32_t, e5m2_t>, nk_cast_serial);
    check("cast_e2m1_to_f32_serial", test_cast<e2m1x2_t, f32_t>, nk_cast_serial);
    check("cast_f32_to_e2m1_serial", test_cast<f32_t, e2m1x2_t>, nk_cast_serial);
    check("cast_ue8m0_to_f32_serial", test_cast<ue8m0_t, f32_t>, nk_cast_serial);
    check("cast_f32_to_ue8m0_serial", test_cast<f32_t, ue8m0_t>, nk_cast_serial);
    check("cast_ue4m3_to_f32_serial", test_cast<ue4m3_t, f32_t>, nk_cast_serial);
    check("cast_f32_to_ue4m3_serial", test_cast<f32_t, ue4m3_t>, nk_cast_serial);
    check("cast_f16_to_f32_serial", test_cast<f16_t, f32_t>, nk_cast_serial);
    check("cast_f32_to_f16_serial", test_cast<f32_t, f16_t>, nk_cast_serial);
    check("cast_f32_to_f64_serial", test_cast<f32_t, f64_t>, nk_cast_serial);
    check("cast_f64_to_f32_serial", test_cast<f64_t, f32_t>, nk_cast_serial);
    check("cast_f64_to_i32_serial", test_cast<f64_t, i32_t>, nk_cast_serial);
    check("cast_i16_to_i64_serial", test_cast<i16_t, i64_t>, nk_cast_serial);
    check("cast_i32_to_f64_serial", test_cast<i32_t, f64_t>, nk_cast_serial);
    check("cast_i32_to_i8_serial", test_cast<i32_t, i8_t>, nk_cast_serial);
    check("cast_i8_to_f64_serial", test_cast<i8_t, f64_t>, nk_cast_serial);
    check("cast_i8_to_i32_serial", test_cast<i8_t, i32_t>, nk_cast_serial);
    check("cast_u8_to_f32_serial", test_cast<u8_t, f32_t>, nk_cast_serial);

    // Block-scaled round-trip: encode f32 → format → decode f32.
    check("cast_block_scaled_nvfp4_serial", test_cast_block_scaled, nk_cast_block_scaled_serial, nk_nvfp4);
    check("cast_block_scaled_mxfp4_serial", test_cast_block_scaled, nk_cast_block_scaled_serial, nk_mxfp4);
    check("cast_block_scaled_mxfp6_e2m3_serial", test_cast_block_scaled, nk_cast_block_scaled_serial, nk_mxfp6_e2m3);
    check("cast_block_scaled_mxfp6_e3m2_serial", test_cast_block_scaled, nk_cast_block_scaled_serial, nk_mxfp6_e3m2);
    check("cast_block_scaled_mxfp8_e4m3_serial", test_cast_block_scaled, nk_cast_block_scaled_serial, nk_mxfp8_e4m3);
    check("cast_block_scaled_mxfp8_e5m2_serial", test_cast_block_scaled, nk_cast_block_scaled_serial, nk_mxfp8_e5m2);
    check("cast_block_scaled_mxint8_serial", test_cast_block_scaled, nk_cast_block_scaled_serial, nk_mxint8);
}
