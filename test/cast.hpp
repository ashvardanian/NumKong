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

#include "numkong/cast.h" // `nk_cast_serial`

#include "harness.hpp"

namespace ashvardanian::numkong::test {

using cast_t = nk_status_t (*)(void const *, nk_dtype_t, void *, nk_dtype_t, nk_size_t, nk_stream_t);

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

/** The buffers of one side of a cast: elements, and for block-scaled dtypes their scales and the
 *  tensor scale, all in the memory of @p backend_type_. */
template <typename value_type_, typename backend_type_>
struct cast_operand {
    static constexpr nk_dtype_t dtype = value_type_::dtype();
    static constexpr bool scaled = nk_dtype_is_block_scaled(dtype);
    static constexpr bool has_tensor_scale = nk_block_scaled_format_of_dtype(dtype).tensor_scale_dtype == nk_f32_k;

    using bytes_t = nk::vector<u8_t, typename backend_type_::template allocator<u8_t>>;
    using elements_t =
        std::conditional_t<scaled, bytes_t,
                           nk::vector<value_type_, typename backend_type_::template allocator<value_type_>>>;

    elements_t elements;
    bytes_t scales;
    bytes_t tensor_scale;

    cast_operand(std::size_t dimensions, backend_type_ const &backend)
        : elements(make_elements_(dimensions, backend)),
          scales(bytes_t::zeros(nk_block_scaled_scales_size(dimensions, nk_block_scaled_format_of_dtype(dtype)) + 1,
                                allocator_of<u8_t>(backend))
                     .value),
          tensor_scale(bytes_t::zeros(sizeof(nk_f32_t), allocator_of<u8_t>(backend)).value) {}

    nk_f32_t &tensor_scale_value() { return *reinterpret_cast<nk_f32_t *>(tensor_scale.raw_values_data()); }

    /** The reference a kernel takes for block-scaled dtypes, read-only or writable alike: every one
     *  starts as NVFP4's does, and MX formats never read the tensor scale. */
    struct reference_t {
        void *elements;
        void *scales;
        nk_f32_t *tensor_scale;
    };

    reference_t reference() {
        return {elements.raw_values_data(), scales.raw_values_data(),
                has_tensor_scale ? &tensor_scale_value() : nullptr};
    }

    /** What to pass as the kernel's operand: the reference for block-scaled dtypes, else the plain
     *  elements. */
    void *operand(reference_t &reference_storage) {
        if constexpr (scaled) return &reference_storage;
        else return elements.raw_values_data();
    }

    static elements_t make_elements_(std::size_t dimensions, backend_type_ const &backend) {
        if constexpr (scaled)
            return bytes_t::zeros(nk_block_scaled_elements_size(dimensions, nk_block_scaled_format_of_dtype(dtype)),
                                  allocator_of<u8_t>(backend))
                .value;
        else return elements_t::zeros(dimensions, allocator_of<value_type_>(backend)).value;
    }
};

/** Tests a cast kernel against the serial kernel, byte for byte for block-scaled results and per
 *  logical element for plain ones. Either side may be plain or block-scaled: a block-scaled source
 *  is the serial encoding of random F32, a block-scaled destination with a tensor scale runs with a
 *  given scale and with one the kernel derives. */
template <typename from_type_, typename to_type_, typename backend_type_ = host_backend_t>
error_stats_t test_cast(settings_t const &settings, cast_t kernel) {
    using source_t = cast_operand<from_type_, backend_type_>;
    using target_t = cast_operand<to_type_, backend_type_>;
    using floats_t = nk::vector<f32_t, typename backend_type_::template allocator<f32_t>>;
    backend_type_ backend = make_backend<backend_type_>(settings);
    error_stats_t stats(comparison_family_t::exact_k);
    std::mt19937 generator(settings.seed.value);

    // Align to lcm(dims_per_value) so both buffers land on clean storage boundaries.
    std::size_t const aligned_dims = std::lcm(nk::dimensions_per_value<from_type_>(),
                                              nk::dimensions_per_value<to_type_>());
    std::size_t const dimensions = (settings.dense_dimensions / aligned_dims) * aligned_dims;

    source_t source(dimensions, backend);
    target_t target(dimensions, backend), reference(dimensions, backend);
    auto source_floats = floats_t::zeros(dimensions, allocator_of<f32_t>(backend)).value;
    typename source_t::reference_t source_reference = source.reference();
    typename target_t::reference_t target_reference = target.reference(), reference_reference = reference.reference();
    void *source_operand = source.operand(source_reference);
    void *target_operand = target.operand(target_reference);
    void *reference_operand = reference.operand(reference_reference);

    auto compare_bytes = [&](auto &target_bytes, auto &reference_bytes) {
        for (std::size_t i = 0; i < target_bytes.size_bytes(); ++i)
            stats.accumulate(target_bytes[i], reference_bytes[i]);
    };

    for (time_point_t const deadline = steady_clock_t::now() + settings.time_limit_per_kernel;
         steady_clock_t::now() < deadline;) {
        if constexpr (source_t::scaled) {
            fill_random(settings, generator, source_floats);
            source.tensor_scale_value() = 0.0f;
            stats.expect(nk_cast_serial(source_floats.raw_values_data(), nk_f32_k, source_operand, from_type_::dtype(),
                                        dimensions, nullptr));
        }
        else if constexpr (target_t::scaled) fill_random(settings, generator, source.elements);
        else fill_random_bits(generator, source.elements);

        for (float const tensor_scale : {1.0f, 0.0f}) {
            if (tensor_scale == 0.0f && !target_t::has_tensor_scale) continue;
            target.tensor_scale_value() = reference.tensor_scale_value() = tensor_scale;
            stats.expect(nk_cast_serial(source_operand, from_type_::dtype(), reference_operand, to_type_::dtype(),
                                        dimensions, nullptr));
            if (nk_status_t const status = backend.call(kernel, static_cast<void const *>(source_operand),
                                                        from_type_::dtype(), target_operand, to_type_::dtype(),
                                                        static_cast<nk_size_t>(dimensions));
                status != nk_success_k) {
                stats.expect(status);
                stats.expect(backend.synchronize());
                return stats;
            }
            if (nk_status_t const status = backend.synchronize(); status != nk_success_k) {
                stats.expect(status);
                return stats;
            }

            if constexpr (target_t::scaled) {
                compare_bytes(target.elements, reference.elements);
                compare_bytes(target.scales, reference.scales);
                if constexpr (target_t::has_tensor_scale) compare_bytes(target.tensor_scale, reference.tensor_scale);
            }
            else
                for (std::size_t i = 0; i < target.elements.size(); ++i)
                    stats.accumulate(read_element_(target.elements, i), read_element_(reference.elements, i));
        }
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
    backend_type_ backend = make_backend<backend_type_>(settings);
    error_stats_t stats(comparison_family_t::exact_k);
    std::mt19937 generator(settings.seed.value);
    std::size_t const count = nk::divide_round_up(settings.dense_dimensions, 8) * 8, capacity = count * 16;
    auto source = bytes_t::zeros(capacity, allocator_of<u8_t>(backend)).value,
         target = bytes_t::zeros(capacity, allocator_of<u8_t>(backend)).value;
    auto reference = make_vector<u8_t>(capacity);

    for (time_point_t const deadline = steady_clock_t::now() + settings.time_limit_per_kernel;
         steady_clock_t::now() < deadline;)
        for (nk_dtype_t const from_type : dtypes) {
            fill_random_bits(generator, source);
            for (nk_dtype_t const to_type : dtypes) {
                if (from_type == nk_u1_k && nk_dtype_family(to_type) == nk_dtype_family_int_k) continue;
                std::memset(target.raw_values_data(), 0, capacity);
                std::memset(reference.raw_values_data(), 0, capacity);
                stats.expect(nk_cast_serial(source.raw_values_data(), from_type, reference.raw_values_data(), to_type,
                                            count, nullptr));
                if (nk_status_t const status = backend.call(kernel, static_cast<void const *>(source.raw_values_data()),
                                                            from_type, static_cast<void *>(target.raw_values_data()),
                                                            to_type, count);
                    status != nk_success_k) {
                    stats.expect(status);
                    stats.expect(backend.synchronize());
                    return stats;
                }
                if (nk_status_t const status = backend.synchronize(); status != nk_success_k) {
                    stats.expect(status);
                    return stats;
                }
                std::size_t const to_bytes = count * nk_dtype_bits(to_type) / NUMKONG_BITS_PER_BYTE;
                for (std::size_t i = 0; i < to_bytes; ++i) stats.accumulate(target[i], reference[i]);
            }
        }
    return stats;
}

/** Registers the casts of every block-scaled format from F32 and to F32, and three transcodes
 *  covering each distinct path, as rows named `cast_<from>_to_<to>_<isa>`, each checked by
 *  @c test_cast: a 16-element block into a 32-element one, the reverse with a derived tensor scale,
 *  and a widening between two MX formats. */
template <typename backend_type_ = host_backend_t>
void check_block_scaled_casts(error_stats_section_t &check, char const *isa, cast_t kernel) {
    auto row = [&]<typename from_type_, typename to_type_>() {
        std::string const name = fmt::format("cast_{}_to_{}_{}", from_type_::dtype_name(), to_type_::dtype_name(), isa);
        check(name.c_str(), test_cast<from_type_, to_type_, backend_type_>, kernel);
    };
    [&]<typename... format_types_>(std::type_identity<format_types_>...) {
        ((row.template operator()<f32_t, format_types_>(), row.template operator()<format_types_, f32_t>()), ...);
    }(std::type_identity<nvfp4_t> {}, std::type_identity<mxfp4_t> {}, std::type_identity<mxfp6e2m3_t> {},
      std::type_identity<mxfp6e3m2_t> {}, std::type_identity<mxfp8e4m3_t> {}, std::type_identity<mxfp8e5m2_t> {},
      std::type_identity<mxint8_t> {});
    row.template operator()<nvfp4_t, mxfp8e4m3_t>();
    row.template operator()<mxfp8e4m3_t, nvfp4_t>();
    row.template operator()<mxfp4_t, mxfp8e5m2_t>();
}

} // namespace ashvardanian::numkong::test

#endif // NUMKONG_TEST_CAST_HPP
