/**
 *  @file include/numkong/geospatial.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 *  @brief Geospatial kernels: haversine, vincenty.
 */
#ifndef NK_GEOSPATIAL_HPP
#define NK_GEOSPATIAL_HPP

#include <cstdint>     // `std::uint32_t`
#include <type_traits> // `std::is_same_v`

#include "numkong/geospatial.h"

#include "numkong/types.hpp"

namespace ashvardanian::numkong {

/**
 *  @brief Batched Haversine: 2R × arcsin(√(sin²(Δφ/2) + cos φ₁ × cos φ₂ × sin²(Δλ/2)))
 *  @param[in] a_lats Latitudes of the first points (radians)
 *  @param[in] a_lons Longitudes of the first points (radians)
 *  @param[in] b_lats Latitudes of the second points (radians)
 *  @param[in] b_lons Longitudes of the second points (radians)
 *  @param[in] d Number of point pairs
 *  @param[out] results Output array of distances (meters)
 *
 *  @tparam in_type_ Input coordinate type (f32_t, f64_t)
 *  @tparam precision_type_ Precision type for scalar fallback computations, defaults to `in_type_`
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 *
 *  @note Uses spherical Earth model with mediatorial radius (6335439.0 m)
 *  @note Accuracy: 0.3-0.6% vs WGS-84, suitable for ranking/similarity
 */
template <numeric_dtype in_type_, numeric_dtype precision_type_ = in_type_, allow_simd_t allow_simd_ = prefer_simd_k>
void haversine(in_type_ const *a_lats, in_type_ const *a_lons, in_type_ const *b_lats, in_type_ const *b_lons,
               std::size_t d, in_type_ *results) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k && std::is_same_v<in_type_, precision_type_>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd)
        nk_haversine_f64(&a_lats->raw_, &a_lons->raw_, &b_lats->raw_, &b_lons->raw_, d, &results->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd)
        nk_haversine_f32(&a_lats->raw_, &a_lons->raw_, &b_lats->raw_, &b_lons->raw_, d, &results->raw_);
    // Scalar fallback
    else {
        precision_type_ const earth_radius = precision_type_(6335439.0); // mediatorial radius in meters

        for (std::size_t i = 0; i < d; i++) {
            precision_type_ first_latitude = precision_type_(a_lats[i]);
            precision_type_ first_longitude = precision_type_(a_lons[i]);
            precision_type_ second_latitude = precision_type_(b_lats[i]);
            precision_type_ second_longitude = precision_type_(b_lons[i]);

            precision_type_ latitude_delta = second_latitude - first_latitude;
            precision_type_ longitude_delta = second_longitude - first_longitude;

            // Haversine formula: a = sin²(Δlat/2) + cos(lat1)×cos(lat2)×sin²(Δlon/2)
            precision_type_ sin_latitude_delta_half = (latitude_delta * precision_type_(0.5)).sin();
            precision_type_ sin_longitude_delta_half = (longitude_delta * precision_type_(0.5)).sin();
            precision_type_ cos_first_latitude = first_latitude.cos();
            precision_type_ cos_second_latitude = second_latitude.cos();

            precision_type_ haversine_term = sin_latitude_delta_half * sin_latitude_delta_half +
                                             cos_first_latitude * cos_second_latitude * sin_longitude_delta_half *
                                                 sin_longitude_delta_half;

            // Central angle: c = 2 * atan2(sqrt(a), sqrt(1-a))
            precision_type_ sqrt_haversine = haversine_term.sqrt();
            precision_type_ sqrt_complement = (precision_type_(1.0) - haversine_term).sqrt();
            precision_type_ central_angle = precision_type_(2.0) * sqrt_haversine.atan2(sqrt_complement);

            results[i] = in_type_(static_cast<double>(earth_radius * central_angle));
        }
    }
}

/**
 *  @brief Batched Vincenty distance (geodesic on WGS-84 ellipsoid)
 *  @param[in] a_lats Latitudes of the first points (radians)
 *  @param[in] a_lons Longitudes of the first points (radians)
 *  @param[in] b_lats Latitudes of the second points (radians)
 *  @param[in] b_lons Longitudes of the second points (radians)
 *  @param[in] d Number of point pairs
 *  @param[out] results Output array of distances (meters)
 *
 *  @tparam in_type_ Input coordinate type (f32_t, f64_t)
 *  @tparam precision_type_ Precision type for scalar fallback computations, defaults to `in_type_`
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 *
 *  @note Uses WGS-84/IERS-2003 ellipsoid model
 *  @note Accuracy: 0.01-0.2% vs WGS-84, 3-20x more accurate than Haversine
 *  @note Iterative algorithm with max 100 iterations
 */
template <numeric_dtype in_type_, numeric_dtype precision_type_ = in_type_, allow_simd_t allow_simd_ = prefer_simd_k>
void vincenty(in_type_ const *a_lats, in_type_ const *a_lons, in_type_ const *b_lats, in_type_ const *b_lons,
              std::size_t d, in_type_ *results) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k && std::is_same_v<in_type_, precision_type_>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd)
        nk_vincenty_f64(&a_lats->raw_, &a_lons->raw_, &b_lats->raw_, &b_lons->raw_, d, &results->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd)
        nk_vincenty_f32(&a_lats->raw_, &a_lons->raw_, &b_lats->raw_, &b_lons->raw_, d, &results->raw_);
    // Scalar fallback
    else {
        precision_type_ const equatorial_radius = precision_type_(6378136.6);
        precision_type_ const polar_radius = precision_type_(6356751.9);
        precision_type_ const flattening = precision_type_(1.0) / precision_type_(298.25642);
        precision_type_ const convergence_threshold = precision_type_(1e-12);
        constexpr int max_iterations = 100;

        for (std::size_t i = 0; i < d; i++) {
            precision_type_ first_latitude = precision_type_(a_lats[i]);
            precision_type_ second_latitude = precision_type_(b_lats[i]);
            precision_type_ longitude_difference = precision_type_(b_lons[i]) - precision_type_(a_lons[i]);

            // Reduced latitudes on the auxiliary sphere
            precision_type_ tan_reduced_first = (precision_type_(1.0) - flattening) * first_latitude.tan();
            precision_type_ tan_reduced_second = (precision_type_(1.0) - flattening) * second_latitude.tan();
            precision_type_ cos_reduced_first = precision_type_(1.0) /
                                                (precision_type_(1.0) + tan_reduced_first * tan_reduced_first).sqrt();
            precision_type_ sin_reduced_first = tan_reduced_first * cos_reduced_first;
            precision_type_ cos_reduced_second =
                precision_type_(1.0) / (precision_type_(1.0) + tan_reduced_second * tan_reduced_second).sqrt();
            precision_type_ sin_reduced_second = tan_reduced_second * cos_reduced_second;

            // Iterative convergence of lambda (difference in longitude on auxiliary sphere)
            precision_type_ lambda = longitude_difference;
            precision_type_ lambda_previous = longitude_difference;
            precision_type_ sin_angular_distance, cos_angular_distance, angular_distance;
            precision_type_ sin_azimuth, cos_squared_azimuth, cos_double_angular_midpoint;
            bool coincident = false;

            for (unsigned int iteration = 0; iteration < max_iterations; iteration++) {
                precision_type_ sin_lambda = lambda.sin();
                precision_type_ cos_lambda = lambda.cos();

                precision_type_ cross_term = cos_reduced_second * sin_lambda;
                precision_type_ mixed_term = cos_reduced_first * sin_reduced_second -
                                             sin_reduced_first * cos_reduced_second * cos_lambda;
                sin_angular_distance = (cross_term * cross_term + mixed_term * mixed_term).sqrt();

                if (sin_angular_distance == precision_type_(0.0)) {
                    coincident = true;
                    break;
                }

                cos_angular_distance = sin_reduced_first * sin_reduced_second +
                                       cos_reduced_first * cos_reduced_second * cos_lambda;
                angular_distance = sin_angular_distance.atan2(cos_angular_distance);

                sin_azimuth = cos_reduced_first * cos_reduced_second * sin_lambda / sin_angular_distance;
                cos_squared_azimuth = precision_type_(1.0) - sin_azimuth * sin_azimuth;

                // Handle equatorial geodesic case
                cos_double_angular_midpoint = (cos_squared_azimuth != precision_type_(0.0))
                                                  ? cos_angular_distance - precision_type_(2.0) * sin_reduced_first *
                                                                               sin_reduced_second / cos_squared_azimuth
                                                  : precision_type_(0.0);

                precision_type_ correction_factor =
                    flattening / precision_type_(16.0) * cos_squared_azimuth *
                    (precision_type_(4.0) +
                     flattening * (precision_type_(4.0) - precision_type_(3.0) * cos_squared_azimuth));

                lambda_previous = lambda;
                lambda = longitude_difference +
                         (precision_type_(1.0) - correction_factor) * flattening * sin_azimuth *
                             (angular_distance +
                              correction_factor * sin_angular_distance *
                                  (cos_double_angular_midpoint +
                                   correction_factor * cos_angular_distance *
                                       (precision_type_(-1.0) + precision_type_(2.0) * cos_double_angular_midpoint *
                                                                    cos_double_angular_midpoint)));

                if ((lambda - lambda_previous).abs() < convergence_threshold) break;
            }

            if (coincident) {
                results[i] = in_type_(0.0);
                continue;
            }

            // Final distance calculation
            precision_type_ u_squared = cos_squared_azimuth *
                                        (equatorial_radius * equatorial_radius - polar_radius * polar_radius) /
                                        (polar_radius * polar_radius);
            precision_type_ series_a =
                precision_type_(1.0) +
                u_squared / precision_type_(16384.0) *
                    (precision_type_(4096.0) +
                     u_squared * (precision_type_(-768.0) +
                                  u_squared * (precision_type_(320.0) - precision_type_(175.0) * u_squared)));
            precision_type_ series_b = u_squared / precision_type_(1024.0) *
                                       (precision_type_(256.0) +
                                        u_squared *
                                            (precision_type_(-128.0) +
                                             u_squared * (precision_type_(74.0) - precision_type_(47.0) * u_squared)));

            precision_type_ angular_correction =
                series_b * sin_angular_distance *
                (cos_double_angular_midpoint +
                 series_b / precision_type_(4.0) *
                     (cos_angular_distance *
                          (precision_type_(-1.0) +
                           precision_type_(2.0) * cos_double_angular_midpoint * cos_double_angular_midpoint) -
                      series_b / precision_type_(6.0) * cos_double_angular_midpoint *
                          (precision_type_(-3.0) + precision_type_(4.0) * sin_angular_distance * sin_angular_distance) *
                          (precision_type_(-3.0) +
                           precision_type_(4.0) * cos_double_angular_midpoint * cos_double_angular_midpoint)));

            results[i] = in_type_(
                static_cast<double>(polar_radius * series_a * (angular_distance - angular_correction)));
        }
    }
}

} // namespace ashvardanian::numkong

#include "numkong/tensor.hpp"

namespace ashvardanian::numkong {

template <numeric_dtype in_type_, numeric_dtype precision_type_ = in_type_, allow_simd_t allow_simd_ = prefer_simd_k>
void haversine(vector_view<in_type_> a_lats, vector_view<in_type_> a_lons, vector_view<in_type_> b_lats,
               vector_view<in_type_> b_lons, in_type_ *results) noexcept {
    haversine<in_type_, precision_type_, allow_simd_>(a_lats.data(), a_lons.data(), b_lats.data(), b_lons.data(),
                                                      a_lats.size(), results);
}

template <numeric_dtype in_type_, numeric_dtype precision_type_ = in_type_, allow_simd_t allow_simd_ = prefer_simd_k>
void vincenty(vector_view<in_type_> a_lats, vector_view<in_type_> a_lons, vector_view<in_type_> b_lats,
              vector_view<in_type_> b_lons, in_type_ *results) noexcept {
    vincenty<in_type_, precision_type_, allow_simd_>(a_lats.data(), a_lons.data(), b_lats.data(), b_lons.data(),
                                                     a_lats.size(), results);
}

} // namespace ashvardanian::numkong

#endif // NK_GEOSPATIAL_HPP
