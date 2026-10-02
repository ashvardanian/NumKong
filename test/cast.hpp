/**
 *  @file test/cast.hpp
 *  @author Ash Vardanian
 *  @date October 1, 2026
 *  @brief Backend-neutral conversion scenarios, run on the host and on devices alike.
 *
 *  Every scenario is a template over a backend owning where the operands live and when results
 *  become readable, @c host_backend_t by default, and checks its kernel against the serial one.
 */
#pragma once
#ifndef NUMKONG_TEST_CAST_HPP
#define NUMKONG_TEST_CAST_HPP

#include <cstring> // `std::memset`

#include <numeric> // `std::lcm`

#include "numkong/cast.h" // `nk_cast_serial`, `nk_cast_block_scaled_serial`

#include "harness.hpp"

namespace ashvardanian::numkong::test {

using cast_t = nk_status_t (*)(void const *, nk_dtype_t, nk_size_t, void *, nk_dtype_t, void *);

/**
 *  @brief Pull one logical element out of a vector as a primitive comparable value.
 *
 *  For sub-byte value types — i4x2, u4x2, u1x8, e2m1x2 — `vec[i]` returns a @c sub_byte_ref whose
 *  conversion operator upcasts the nibble/bit to its natural integer type, i8, u8 or bool; unary
 *  `+` triggers that conversion. For byte-sized types, the indexed wrapper struct exposes the
 *  primitive directly through `.raw_`.
 */
template <typename vec_type_>
static auto read_element_(vec_type_ const &v, std::size_t i) {
    if constexpr (nk::dimensions_per_value<typename vec_type_::value_type>() > 1) return +v[i];
    else return v[i].raw_;
}

/** Tests a cast kernel against the serial kernel; SIMD kernels must match serial output exactly for
 *  every logical element. */
template <typename from_type_, typename to_type_, typename backend_type_ = host_backend_t>
error_stats_t test_cast(settings_t const &settings, cast_t kernel) {
    using sources_t = nk::vector<from_type_, typename backend_type_::template allocator<from_type_>>;
    using targets_t = nk::vector<to_type_, typename backend_type_::template allocator<to_type_>>;
    backend_type_ backend;
    error_stats_t stats(comparison_family_t::exact_k);
    std::mt19937 generator(settings.seed.value);

    // Align to lcm(dims_per_value) so both buffers land on clean storage boundaries.
    std::size_t const aligned_dims = std::lcm(nk::dimensions_per_value<from_type_>(),
                                              nk::dimensions_per_value<to_type_>());
    std::size_t const dimensions = (settings.dense_dimensions / aligned_dims) * aligned_dims;

    auto source_vec = sources_t::zeros(dimensions).value;
    auto target_vec = targets_t::zeros(dimensions).value;
    auto reference_vec = make_vector<to_type_>(dimensions);

    for (time_point_t const deadline = steady_clock_t::now() + settings.time_limit_per_kernel;
         steady_clock_t::now() < deadline;) {
        fill_random_bits(generator, source_vec);

        stats.expect(nk_cast_serial(source_vec.raw_values_data(), from_type_::dtype(), dimensions,
                                    reference_vec.raw_values_data(), to_type_::dtype(), nullptr));
        backend.call(kernel, source_vec.raw_values_data(), from_type_::dtype(), dimensions,
                     target_vec.raw_values_data(), to_type_::dtype());
        if (char const *failure = backend.synchronize()) stats.expect(false, failure);

        // Per-element comparison, dispatched to the smart reference for sub-byte types.
        for (std::size_t i = 0; i < target_vec.size(); ++i)
            stats.accumulate(read_element_(target_vec, i), read_element_(reference_vec, i));
    }
    return stats;
}

/** Casts random bits between every pair of scalar types through @p kernel, matching the serial
 *  output byte for byte, NaN payloads included. U1 into signed integers is left out, as the serial
 *  I64 hub reads no U1 and converts whatever its buffers held. */
template <typename backend_type_ = host_backend_t>
error_stats_t test_cast_pairs(settings_t const &settings, cast_t kernel) {
    using bytes_t = nk::vector<u8_t, typename backend_type_::template allocator<u8_t>>;
    nk_dtype_t const dtypes[] = {nk_f64_k,   nk_f32_k,  nk_f16_k,   nk_bf16_k,  nk_e4m3_k, nk_e5m2_k, nk_e2m3_k,
                                 nk_e3m2_k,  nk_e2m1_k, nk_ue8m0_k, nk_ue4m3_k, nk_f64c_k, nk_f32c_k, nk_f16c_k,
                                 nk_bf16c_k, nk_i64_k,  nk_i32_k,   nk_i16_k,   nk_i8_k,   nk_i4_k,   nk_u64_k,
                                 nk_u32_k,   nk_u16_k,  nk_u8_k,    nk_u4_k,    nk_u1_k};
    backend_type_ backend;
    error_stats_t stats(comparison_family_t::exact_k);
    std::mt19937 generator(settings.seed.value);
    std::size_t const count = nk::divide_round_up(settings.dense_dimensions, 8) * 8, capacity = count * 16;
    auto source = bytes_t::zeros(capacity).value, target = bytes_t::zeros(capacity).value;
    auto reference = make_vector<u8_t>(capacity);

    for (time_point_t const deadline = steady_clock_t::now() + settings.time_limit_per_kernel;
         steady_clock_t::now() < deadline;)
        for (nk_dtype_t const from_type : dtypes) {
            fill_random_bits(generator, source);
            for (nk_dtype_t const to_type : dtypes) {
                if (from_type == nk_u1_k && nk_dtype_family(to_type) == nk_dtype_family_int_k) continue;
                std::memset(target.raw_values_data(), 0, capacity);
                std::memset(reference.raw_values_data(), 0, capacity);
                stats.expect(nk_cast_serial(source.raw_values_data(), from_type, count, reference.raw_values_data(),
                                            to_type, nullptr));
                backend.call(kernel, static_cast<void const *>(source.raw_values_data()), from_type, count,
                             static_cast<void *>(target.raw_values_data()), to_type);
                if (char const *failure = backend.synchronize()) stats.expect(false, failure);
                std::size_t const to_bytes = count * nk_dtype_bits(to_type) / NUMKONG_BITS_PER_BYTE;
                for (std::size_t i = 0; i < to_bytes; ++i)
                    stats.accumulate(target.raw_values_data()[i], reference.raw_values_data()[i]);
            }
        }
    return stats;
}

using block_scaled_cast_t = nk_status_t (*)(                                                  //
    void const *, void const *, nk_scalar_buffer_t const *, nk_block_scaled_format_t const *, //
    void *, void *, nk_scalar_buffer_t *, nk_block_scaled_format_t const *, nk_size_t, void *);
using block_scaled_format_factory_t = nk_block_scaled_format_t (*)(void);

/**
 *  @brief Tests a block-scaled cast kernel against the serial reference, byte for byte, over every
 *      direction of the format: encoding F32 with a given and with a derived tensor scale, decoding
 *      the serial encoding back to F32, and transcoding it into NVFP4 with a derived tensor scale.
 *
 *  Derived scales only apply to formats with a tensor scale, so MX formats skip that encoding.
 */
template <typename backend_type_ = host_backend_t>
error_stats_t test_cast_block_scaled(settings_t const &settings, block_scaled_cast_t kernel,
                                     block_scaled_format_factory_t factory) {
    using floats_t = nk::vector<f32_t, typename backend_type_::template allocator<f32_t>>;
    using bytes_t = nk::vector<u8_t, typename backend_type_::template allocator<u8_t>>;
    backend_type_ backend;
    error_stats_t stats(comparison_family_t::exact_k);
    std::mt19937 generator(settings.seed.value);

    nk_block_scaled_format_t const format = factory(), plain_f32 = nk_plain(nk_f32_k), nvfp4 = nk_nvfp4();
    std::size_t const dimensions = (settings.dense_dimensions / format.block_size) * format.block_size;
    bool const has_tensor_scale = format.tensor_scale_dtype == nk_f32_k;

    auto source = floats_t::zeros(dimensions).value;
    auto target_decoded = floats_t::zeros(dimensions).value, reference_decoded = floats_t::zeros(dimensions).value;
    auto target_elements = bytes_t::zeros(nk_block_scaled_elements_size(dimensions, format)).value;
    auto reference_elements = bytes_t::zeros(nk_block_scaled_elements_size(dimensions, format)).value;
    auto target_scales = bytes_t::zeros(nk_block_scaled_scales_size(dimensions, format)).value;
    auto reference_scales = bytes_t::zeros(nk_block_scaled_scales_size(dimensions, format)).value;
    auto target_nvfp4 = bytes_t::zeros(nk_block_scaled_elements_size(dimensions, nvfp4)).value;
    auto reference_nvfp4 = bytes_t::zeros(nk_block_scaled_elements_size(dimensions, nvfp4)).value;
    auto target_nvfp4_scales = bytes_t::zeros(nk_block_scaled_scales_size(dimensions, nvfp4)).value;
    auto reference_nvfp4_scales = bytes_t::zeros(nk_block_scaled_scales_size(dimensions, nvfp4)).value;
    // Tensor scales of the target and the reference, in memory the kernel reaches
    auto tensor_scales = bytes_t::zeros(4 * sizeof(nk_scalar_buffer_t)).value;
    auto *target_tensor_scale = reinterpret_cast<nk_scalar_buffer_t *>(tensor_scales.raw_values_data());
    auto *reference_tensor_scale = target_tensor_scale + 1;
    auto *target_nvfp4_tensor_scale = target_tensor_scale + 2, *reference_nvfp4_tensor_scale = target_tensor_scale + 3;

    auto compare_bytes = [&](void const *target, void const *reference, std::size_t bytes) {
        for (std::size_t i = 0; i < bytes; ++i)
            stats.accumulate(static_cast<nk_u8_t const *>(target)[i], static_cast<nk_u8_t const *>(reference)[i]);
    };
    auto cast_both = [&](void const *from, void const *from_scales, nk_scalar_buffer_t const *from_tensor_scale,
                         nk_block_scaled_format_t const *from_format, void *target, void *target_scales,
                         nk_scalar_buffer_t *target_tensor_scale, void *reference, void *reference_scales,
                         nk_scalar_buffer_t *reference_tensor_scale, nk_block_scaled_format_t const *to_format) {
        stats.expect(nk_cast_block_scaled_serial(from, from_scales, from_tensor_scale, from_format, reference,
                                                 reference_scales, reference_tensor_scale, to_format, dimensions,
                                                 nullptr));
        backend.call(kernel, from, from_scales, from_tensor_scale, from_format, target, target_scales,
                     target_tensor_scale, to_format, static_cast<nk_size_t>(dimensions));
        if (char const *failure = backend.synchronize()) stats.expect(false, failure);
    };

    for (time_point_t const deadline = steady_clock_t::now() + settings.time_limit_per_kernel;
         steady_clock_t::now() < deadline;) {
        fill_random(settings, generator, source);
        std::size_t const elements_bytes = target_elements.size_values(), scales_bytes = target_scales.size_values();

        for (float const tensor_scale : {1.0f, 0.0f}) {
            if (tensor_scale == 0.0f && !has_tensor_scale) continue;
            target_tensor_scale->f32 = reference_tensor_scale->f32 = tensor_scale;
            cast_both(source.raw_values_data(), nullptr, nullptr, &plain_f32, target_elements.raw_values_data(),
                      target_scales.raw_values_data(), has_tensor_scale ? target_tensor_scale : nullptr,
                      reference_elements.raw_values_data(), reference_scales.raw_values_data(),
                      has_tensor_scale ? reference_tensor_scale : nullptr, &format);
            compare_bytes(target_elements.raw_values_data(), reference_elements.raw_values_data(), elements_bytes);
            compare_bytes(target_scales.raw_values_data(), reference_scales.raw_values_data(), scales_bytes);
            compare_bytes(target_tensor_scale, reference_tensor_scale, sizeof(nk_f32_t));
        }

        cast_both(reference_elements.raw_values_data(), reference_scales.raw_values_data(), reference_tensor_scale,
                  &format, target_decoded.raw_values_data(), nullptr, nullptr, reference_decoded.raw_values_data(),
                  nullptr, nullptr, &plain_f32);
        compare_bytes(target_decoded.raw_values_data(), reference_decoded.raw_values_data(),
                      dimensions * sizeof(nk_f32_t));

        target_nvfp4_tensor_scale->f32 = reference_nvfp4_tensor_scale->f32 = 0.0f;
        cast_both(reference_elements.raw_values_data(), reference_scales.raw_values_data(), reference_tensor_scale,
                  &format, target_nvfp4.raw_values_data(), target_nvfp4_scales.raw_values_data(),
                  target_nvfp4_tensor_scale, reference_nvfp4.raw_values_data(),
                  reference_nvfp4_scales.raw_values_data(), reference_nvfp4_tensor_scale, &nvfp4);
        compare_bytes(target_nvfp4.raw_values_data(), reference_nvfp4.raw_values_data(), target_nvfp4.size_values());
        compare_bytes(target_nvfp4_scales.raw_values_data(), reference_nvfp4_scales.raw_values_data(),
                      target_nvfp4_scales.size_values());
        compare_bytes(target_nvfp4_tensor_scale, reference_nvfp4_tensor_scale, sizeof(nk_f32_t));
    }
    return stats;
}

} // namespace ashvardanian::numkong::test

#endif // NUMKONG_TEST_CAST_HPP
