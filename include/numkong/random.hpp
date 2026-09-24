/**
 *  @file include/numkong/random.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 *  @brief Random number generation utilities for NumKong types.
 *
 *  Lightweight header with random fill functions for testing and benchmarking.
 *  Only depends on types.hpp to minimize compilation overhead.
 */
#ifndef NK_RANDOM_HPP
#define NK_RANDOM_HPP

#include <algorithm> // `std::sort`
#include <random>    // `std::uniform_int_distribution`

#include "numkong/types.hpp"

namespace ashvardanian::numkong {

/** Lightweight clamp to avoid pulling in `<algorithm>` for @c std::clamp. */
template <typename scalar_type_>
scalar_type_ clamp(scalar_type_ val, scalar_type_ low, scalar_type_ high) noexcept {
    return val < low ? low : val > high ? high : val;
}

/**
 *  @brief Fill array with uniform random values.
 *
 *  @tparam value_type_ A NumKong wrapper type, e.g., f32_t, f64_t, i32_t.
 *  @tparam generator_type_ A random number generator type, e.g., @c std::mt19937_64.
 *
 *  @param[inout] generator The random number generator.
 *  @param[out] values_ptr Pointer to the array to fill.
 *  @param[in] values_count Number of storage values to fill, not dimensions for sub-byte types.
 */
template <typename value_type_, typename generator_type_, typename component_type_ = typename value_type_::component_t>
void fill_uniform(generator_type_ &generator, value_type_ *values_ptr, std::size_t values_count,
                  component_type_ min_val, component_type_ max_val) noexcept {

    // Packed types (u1x8, u4x2, i4x2) need special handling - just fill with random bytes
    if constexpr (dimensions_per_value<value_type_>() > 1) {
        std::uniform_int_distribution<std::int32_t> distribution(min_val, max_val);
        for (std::size_t i = 0; i < values_count; ++i)
            for (std::size_t j = 0; j < dimensions_per_value<value_type_>(); ++j)
                values_ptr[i][j] = static_cast<component_type_>(distribution(generator));
    }
    // Integer distribution types aren't always defined
    else if constexpr (is_integral_dtype<value_type_>()) {
        using small_integer_t = std::conditional_t<is_signed_dtype<value_type_>(), std::int32_t, std::uint32_t>;
        using large_integer_t = std::conditional_t<is_signed_dtype<value_type_>(), std::int64_t, std::uint64_t>;
        using distribution_integer_t = std::conditional_t<sizeof(value_type_) <= 4, small_integer_t, large_integer_t>;
        std::uniform_int_distribution<distribution_integer_t> distribution(min_val, max_val);
        for (std::size_t i = 0; i < values_count; ++i)
            values_ptr[i] = static_cast<value_type_>(distribution(generator));
    }
    // Complex types need both real and imaginary parts filled
    else if constexpr (is_complex_dtype<value_type_>()) {
        using component_t = typename value_type_::component_t;
        using distribution_float_t = std::conditional_t<sizeof(component_t) <= 4, float, double>;
        std::uniform_real_distribution<distribution_float_t> distribution(min_val, max_val);
        for (std::size_t i = 0; i < values_count; ++i) {
            auto real = distribution(generator), imag = distribution(generator);
            values_ptr[i] = value_type_(component_t(static_cast<distribution_float_t>(real)),
                                        component_t(static_cast<distribution_float_t>(imag)));
        }
    }
    // Floats and other types use a fixed range for better numerical stability
    else {
        using distribution_float_t = std::conditional_t<sizeof(value_type_) <= 4, float, double>;
        std::uniform_real_distribution<distribution_float_t> distribution(min_val, max_val);
        for (std::size_t i = 0; i < values_count; ++i)
            values_ptr[i] = static_cast<value_type_>(distribution(generator));
    }
}

/** Fill array with uniform random values within specified range. */
template <typename value_type_, typename generator_type_>
void fill_uniform(generator_type_ &generator, value_type_ *values_ptr, std::size_t values_count) noexcept {

    // Packed types (u1x8, u4x2, i4x2) need special handling - just fill with random bytes
    if constexpr (dimensions_per_value<value_type_>() > 1) {
        std::uniform_int_distribution<std::uint32_t> distribution(0, 255);
        for (std::size_t i = 0; i < values_count; ++i)
            values_ptr[i] = value_type_::from_raw(static_cast<typename value_type_::raw_t>(distribution(generator)));
    }
    else if constexpr (is_integral_dtype<value_type_>())
        return fill_uniform(generator, values_ptr, values_count, finite_min<value_type_>(), finite_max<value_type_>());
    else return fill_uniform(generator, values_ptr, values_count, -10, +10);
}

/**
 *  @brief Fill array with lognormal distribution, good for detecting numerical edge cases.
 *
 *  Probability Density Function (PDF):
 *
 *      f(x; μ, σ) = 1 / (x · σ · √(2π)) · exp(−(ln x − μ)² / (2σ²)),  x > 0
 *
 *  Equivalently, if X ~ 𝒩(μ, σ²), then eˣ ~ LogNormal(μ, σ). Values span many orders of magnitude,
 *  which is useful for exercising overflow/underflow paths in low-precision arithmetic. Common in
 *  modeling latencies, file sizes, and NLP token frequencies.
 */
template <typename value_type_, typename generator_type_>
void fill_lognormal(generator_type_ &generator, value_type_ *values_ptr, std::size_t values_count, //
                    double mean = 0.0, double stddev = 0.5) noexcept {

    // Packed types (u1x8, u4x2, i4x2) need special handling
    if constexpr (dimensions_per_value<value_type_>() > 1) {
        std::lognormal_distribution<float> distribution(static_cast<float>(mean), static_cast<float>(stddev));
        for (std::size_t i = 0; i < values_count; ++i)
            for (std::size_t j = 0; j < dimensions_per_value<value_type_>(); ++j)
                values_ptr[i][j] = static_cast<std::uint8_t>(clamp(distribution(generator), 0.0f, 255.0f));
    }
    // Complex types need both real and imaginary parts filled
    else if constexpr (is_complex_dtype<value_type_>()) {
        using component_t = typename value_type_::component_t;
        using distribution_float_t = std::conditional_t<sizeof(component_t) <= 4, float, double>;
        std::lognormal_distribution<distribution_float_t> distribution(static_cast<distribution_float_t>(mean),
                                                                       static_cast<distribution_float_t>(stddev));
        std::bernoulli_distribution sign_distribution(0.5);
        distribution_float_t const clamp_max = finite_max<component_t>();
        distribution_float_t const clamp_min = finite_min<component_t>();
        constexpr distribution_float_t signs[] = {1, -1};
        for (std::size_t i = 0; i < values_count; ++i) {
            auto real = clamp(distribution(generator) * signs[sign_distribution(generator)], clamp_min, clamp_max);
            auto imag = clamp(distribution(generator) * signs[sign_distribution(generator)], clamp_min, clamp_max);
            values_ptr[i] = value_type_(component_t(real), component_t(imag));
        }
    }
    // Floats and integers: unified path with trait-based clamping
    else {
        using distribution_float_t = std::conditional_t<sizeof(value_type_) <= 4, float, double>;
        std::lognormal_distribution<distribution_float_t> distribution(static_cast<distribution_float_t>(mean),
                                                                       static_cast<distribution_float_t>(stddev));
        std::bernoulli_distribution sign_distribution(0.5);
        distribution_float_t const clamp_max = static_cast<distribution_float_t>(finite_max<value_type_>());
        distribution_float_t const clamp_min = static_cast<distribution_float_t>(finite_min<value_type_>());
        constexpr distribution_float_t signs[] = {1, -1};
        for (std::size_t i = 0; i < values_count; ++i) {
            distribution_float_t val = distribution(generator);
            if constexpr (is_signed_dtype<value_type_>()) val *= signs[sign_distribution(generator)];
            values_ptr[i] = static_cast<value_type_>(clamp(val, clamp_min, clamp_max));
        }
    }
}

/**
 *  @brief Fill array with Cauchy distribution, heavy tails for stress testing.
 *
 *  Probability Density Function (PDF):
 *
 *      f(x; x₀, γ) = 1 / (π · γ · (1 + ((x − x₀) / γ)²))
 *
 *  The Cauchy distribution has no defined mean or variance, as all its moments diverge, making it
 *  ideal for stress-testing numerical stability with extreme outliers. Common in robust Bayesian
 *  priors, half-Cauchy, metaheuristic Levy flights, and adversarial robustness evaluation.
 */
template <typename value_type_, typename generator_type_>
void fill_cauchy(generator_type_ &generator, value_type_ *values_ptr, std::size_t values_count, //
                 double location = 0.0, double scale = 1.0) noexcept {

    // Packed types (u1x8, u4x2, i4x2) need special handling
    if constexpr (dimensions_per_value<value_type_>() > 1) {
        std::cauchy_distribution<float> distribution(static_cast<float>(location), static_cast<float>(scale));
        for (std::size_t i = 0; i < values_count; ++i)
            for (std::size_t j = 0; j < dimensions_per_value<value_type_>(); ++j)
                values_ptr[i][j] = static_cast<std::uint8_t>(clamp(distribution(generator), 0.0f, 255.0f));
    }
    // Complex types need both real and imaginary parts filled
    else if constexpr (is_complex_dtype<value_type_>()) {
        using component_t = typename value_type_::component_t;
        using distribution_float_t = std::conditional_t<sizeof(component_t) <= 4, float, double>;
        std::cauchy_distribution<distribution_float_t> distribution(static_cast<distribution_float_t>(location),
                                                                    static_cast<distribution_float_t>(scale));
        distribution_float_t const clamp_max = finite_max<component_t>();
        distribution_float_t const clamp_min = finite_min<component_t>();
        for (std::size_t i = 0; i < values_count; ++i) {
            auto real = clamp(distribution(generator), clamp_min, clamp_max);
            auto imag = clamp(distribution(generator), clamp_min, clamp_max);
            values_ptr[i] = value_type_(component_t(real), component_t(imag));
        }
    }
    // Floats and integers: unified two-sided clamp
    else {
        using distribution_float_t = std::conditional_t<sizeof(value_type_) <= 4, float, double>;
        std::cauchy_distribution<distribution_float_t> distribution(static_cast<distribution_float_t>(location),
                                                                    static_cast<distribution_float_t>(scale));
        distribution_float_t const clamp_max = static_cast<distribution_float_t>(finite_max<value_type_>());
        distribution_float_t const clamp_min = static_cast<distribution_float_t>(finite_min<value_type_>());
        for (std::size_t i = 0; i < values_count; ++i)
            values_ptr[i] = static_cast<value_type_>(clamp(distribution(generator), clamp_min, clamp_max));
    }
}

/** Fill arrays with latitude and longitude coordinate values in radians. */
template <typename value_type_, typename generator_type_>
void fill_coordinates(generator_type_ &generator, value_type_ *lats_ptr, value_type_ *lons_ptr,
                      std::size_t values_count) noexcept {

    using distribution_float_t = std::conditional_t<sizeof(value_type_) <= 4, float, double>;
    constexpr distribution_float_t pi = distribution_float_t(3.14159265358979323846);
    std::uniform_real_distribution<distribution_float_t> lat_distribution(-pi / 2, pi / 2);
    std::uniform_real_distribution<distribution_float_t> lon_distribution(-pi, pi);
    for (std::size_t i = 0; i < values_count; ++i) {
        lats_ptr[i] = static_cast<value_type_>(lat_distribution(generator));
        lons_ptr[i] = static_cast<value_type_>(lon_distribution(generator));
    }
}

/**
 *  @brief Fill destination coordinate arrays within a max angular separation of origin coordinates.
 *
 *  Uses the great-circle destination formula to place each destination point at a random angular
 *  distance (uniform in [0, @p max_separation_rad]) and random bearing from the corresponding
 *  origin point. Coordinates are in radians.
 */
template <typename value_type_, typename generator_type_>
void fill_nearby_coordinates(generator_type_ &generator, value_type_ const *origin_lats_ptr,
                             value_type_ const *origin_lons_ptr, value_type_ *dest_lats_ptr, value_type_ *dest_lons_ptr,
                             std::size_t values_count, double max_separation_rad) noexcept {

    using float_t = std::conditional_t<sizeof(value_type_) <= 4, float, double>;
    constexpr float_t pi = float_t(3.14159265358979323846);
    std::uniform_real_distribution<float_t> separation_distribution(0, float_t(max_separation_rad));
    std::uniform_real_distribution<float_t> bearing_distribution(0, float_t(2) * pi);
    for (std::size_t i = 0; i < values_count; ++i) {
        float_t origin_lat = float_t(origin_lats_ptr[i]);
        float_t origin_lon = float_t(origin_lons_ptr[i]);
        float_t separation = separation_distribution(generator);
        float_t bearing = bearing_distribution(generator);
        float_t sin_origin_lat = std::sin(origin_lat), cos_origin_lat = std::cos(origin_lat);
        float_t sin_sep = std::sin(separation), cos_sep = std::cos(separation);
        float_t dest_lat = std::asin(sin_origin_lat * cos_sep + cos_origin_lat * sin_sep * std::cos(bearing));
        float_t dest_lon = origin_lon + std::atan2(std::sin(bearing) * sin_sep * cos_origin_lat,
                                                   cos_sep - sin_origin_lat * std::sin(dest_lat));
        dest_lats_ptr[i] = value_type_(dest_lat);
        dest_lons_ptr[i] = value_type_(dest_lon);
    }
}

/** Fill array as a probability distribution (sums to 1.0). */
template <typename value_type_, typename generator_type_>
void fill_probability(generator_type_ &generator, value_type_ *values_ptr, std::size_t values_count) noexcept {

    using distribution_float_t = std::conditional_t<sizeof(value_type_) <= 4, float, double>;
    std::uniform_real_distribution<distribution_float_t> distribution(distribution_float_t(0.01),
                                                                      distribution_float_t(1));

    distribution_float_t sum = 0;
    for (std::size_t i = 0; i < values_count; ++i) {
        distribution_float_t val = distribution(generator);
        values_ptr[i] = static_cast<value_type_>(val);
        sum += val;
    }
    for (std::size_t i = 0; i < values_count; ++i)
        values_ptr[i] = static_cast<value_type_>(static_cast<distribution_float_t>(values_ptr[i]) / sum);
}

/**
 *  @brief Fill array with sorted unique random integer values, in-place.
 *
 *  Draws @p values_count values from [0, @p max_val - values_count], sorts once, then adds each
 *  element's index. Duplicates become distinct, which guarantees uniqueness and keeps sorted order
 *  in O(n log n) with one sort and no retry loop. Requires @p max_val ≥ @p values_count.
 */
template <typename value_type_, typename generator_type_>
void fill_sorted_unique(generator_type_ &generator, value_type_ *values_ptr, std::size_t values_count,
                        value_type_ max_val) noexcept {
    using raw_t = typename value_type_::raw_t;
    raw_t range = raw_t(max_val) - raw_t(values_count);
    fill_uniform(generator, values_ptr, values_count, raw_t(0), range);
    std::sort(values_ptr, values_ptr + values_count);
    for (std::size_t i = 0; i < values_count; ++i) values_ptr[i] = value_type_(raw_t(values_ptr[i]) + raw_t(i));
}

} // namespace ashvardanian::numkong

#endif // NK_RANDOM_HPP
