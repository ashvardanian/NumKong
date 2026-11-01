/**
 *  @brief Serial Geospatial Distances.
 *  @file include/numkong/geospatial/serial.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *
 *  @sa include/numkong/geospatial.h
 */
#ifndef NK_GEOSPATIAL_SERIAL_H
#define NK_GEOSPATIAL_SERIAL_H

#include "numkong/types.h"
#include "numkong/spatial/serial.h"      // `nk_f64_sqrt_serial`, `nk_f32_sqrt_serial`
#include "numkong/trigonometry/serial.h" // `nk_f64_sin`, `nk_f64_cos`, `nk_f64_atan2`, etc.

#if defined(__cplusplus)
extern "C" {
#endif

/*  Serial implementations of geospatial distance functions.
 *  These use the trigonometric functions from trigonometry.h for sin, cos, and atan2.
 */

NK_PUBLIC void nk_haversine_f64_serial(             //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results) {

    nk_f64_t const earth_radius = NK_EARTH_MEDIATORIAL_RADIUS;

    for (nk_size_t i = 0; i != n; ++i) {
        nk_f64_t first_latitude = a_lats[i];
        nk_f64_t first_longitude = a_lons[i];
        nk_f64_t second_latitude = b_lats[i];
        nk_f64_t second_longitude = b_lons[i];

        nk_f64_t latitude_delta = second_latitude - first_latitude;
        nk_f64_t longitude_delta = second_longitude - first_longitude;

        // Haversine formula: a = sin²(Δlat/2) + cos(lat1)×cos(lat2)×sin²(Δlon/2)
        nk_f64_t sin_latitude_delta_half = nk_f64_sin(latitude_delta * 0.5);
        nk_f64_t sin_longitude_delta_half = nk_f64_sin(longitude_delta * 0.5);
        nk_f64_t cos_first_latitude = nk_f64_cos(first_latitude);
        nk_f64_t cos_second_latitude = nk_f64_cos(second_latitude);

        // Use FMA for improved precision
        nk_f64_t sin_lat_sq = sin_latitude_delta_half * sin_latitude_delta_half;
        nk_f64_t sin_lon_sq = sin_longitude_delta_half * sin_longitude_delta_half;
        nk_f64_t cos_product = cos_first_latitude * cos_second_latitude;
        nk_f64_t haversine_term = nk_f64_fma_serial(cos_product, sin_lon_sq, sin_lat_sq);
        // Clamp haversine_term to [0, 1] to prevent NaN from sqrt of negative values
        haversine_term = (haversine_term < 0.0) ? 0.0 : ((haversine_term > 1.0) ? 1.0 : haversine_term);

        // Central angle: c = 2 × atan2(√a, √(1-a))
        nk_f64_t sqrt_haversine = nk_f64_sqrt_serial(haversine_term);
        nk_f64_t sqrt_complement = nk_f64_sqrt_serial(1.0 - haversine_term);
        nk_f64_t central_angle = 2.0 * nk_f64_atan2(sqrt_haversine, sqrt_complement);

        results[i] = earth_radius * central_angle;
    }
}

NK_PUBLIC void nk_haversine_f32_serial(             //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results) {

    nk_f32_t const earth_radius = (nk_f32_t)NK_EARTH_MEDIATORIAL_RADIUS;

    for (nk_size_t i = 0; i != n; ++i) {
        nk_f32_t first_latitude = a_lats[i];
        nk_f32_t first_longitude = a_lons[i];
        nk_f32_t second_latitude = b_lats[i];
        nk_f32_t second_longitude = b_lons[i];

        nk_f32_t latitude_delta = second_latitude - first_latitude;
        nk_f32_t longitude_delta = second_longitude - first_longitude;

        // Haversine formula: a = sin²(Δlat/2) + cos(lat1)×cos(lat2)×sin²(Δlon/2)
        nk_f32_t sin_latitude_delta_half = nk_f32_sin(latitude_delta * 0.5f);
        nk_f32_t sin_longitude_delta_half = nk_f32_sin(longitude_delta * 0.5f);
        nk_f32_t cos_first_latitude = nk_f32_cos(first_latitude);
        nk_f32_t cos_second_latitude = nk_f32_cos(second_latitude);

        // Use FMA for improved precision
        nk_f32_t sin_lat_sq = sin_latitude_delta_half * sin_latitude_delta_half;
        nk_f32_t sin_lon_sq = sin_longitude_delta_half * sin_longitude_delta_half;
        nk_f32_t cos_product = cos_first_latitude * cos_second_latitude;
        nk_f32_t haversine_term = nk_f32_fma_serial(cos_product, sin_lon_sq, sin_lat_sq);

        // Clamp to [0, 1] to avoid NaN from sqrt of negative numbers (due to floating point errors)
        if (haversine_term < 0.0f) haversine_term = 0.0f;
        if (haversine_term > 1.0f) haversine_term = 1.0f;

        // Central angle: c = 2 × atan2(√a, √(1-a))
        nk_f32_t sqrt_haversine = nk_f32_sqrt_serial(haversine_term);
        nk_f32_t sqrt_complement = nk_f32_sqrt_serial(1.0f - haversine_term);
        nk_f32_t central_angle = 2.0f * nk_f32_atan2(sqrt_haversine, sqrt_complement);

        results[i] = earth_radius * central_angle;
    }
}

NK_PUBLIC void nk_vincenty_f64_serial(              //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results) {

    nk_f64_t const equatorial_radius = NK_EARTH_ELLIPSOID_EQUATORIAL_RADIUS;
    nk_f64_t const polar_radius = NK_EARTH_ELLIPSOID_POLAR_RADIUS;
    nk_f64_t const flattening = 1.0 / NK_EARTH_ELLIPSOID_INVERSE_FLATTENING;

    for (nk_size_t i = 0; i != n; ++i) {
        nk_f64_t first_latitude = a_lats[i];
        nk_f64_t second_latitude = b_lats[i];
        nk_f64_t longitude_difference = b_lons[i] - a_lons[i];

        // Reduced latitudes on the auxiliary sphere
        nk_f64_t tan_reduced_first = (1.0 - flattening) * (nk_f64_sin(first_latitude) / nk_f64_cos(first_latitude));
        nk_f64_t tan_reduced_second = (1.0 - flattening) * (nk_f64_sin(second_latitude) / nk_f64_cos(second_latitude));
        nk_f64_t cos_reduced_first = 1.0 / nk_f64_sqrt_serial(1.0 + tan_reduced_first * tan_reduced_first);
        nk_f64_t sin_reduced_first = tan_reduced_first * cos_reduced_first;
        nk_f64_t cos_reduced_second = 1.0 / nk_f64_sqrt_serial(1.0 + tan_reduced_second * tan_reduced_second);
        nk_f64_t sin_reduced_second = tan_reduced_second * cos_reduced_second;

        // Iterative convergence of lambda (difference in longitude on auxiliary sphere)
        nk_f64_t lambda = longitude_difference;
        nk_f64_t lambda_previous = longitude_difference;
        nk_f64_t sin_angular_distance, cos_angular_distance, angular_distance;
        nk_f64_t sin_azimuth, cos_squared_azimuth, cos_double_angular_midpoint;
        nk_u32_t iteration = 0;

        // Check for coincident points early
        nk_u32_t coincident = 0;
        do {
            nk_f64_t sin_lambda = nk_f64_sin(lambda);
            nk_f64_t cos_lambda = nk_f64_cos(lambda);

            nk_f64_t cross_term = cos_reduced_second * sin_lambda;
            nk_f64_t mixed_term = cos_reduced_first * sin_reduced_second -
                                  sin_reduced_first * cos_reduced_second * cos_lambda;
            sin_angular_distance = nk_f64_sqrt_serial(cross_term * cross_term + mixed_term * mixed_term);

            if (sin_angular_distance == 0.0) {
                coincident = 1;
                break;
            }

            cos_angular_distance = sin_reduced_first * sin_reduced_second +
                                   cos_reduced_first * cos_reduced_second * cos_lambda;
            angular_distance = nk_f64_atan2(sin_angular_distance, cos_angular_distance);

            sin_azimuth = cos_reduced_first * cos_reduced_second * sin_lambda / sin_angular_distance;
            cos_squared_azimuth = 1.0 - sin_azimuth * sin_azimuth;

            // Handle equatorial geodesic case
            cos_double_angular_midpoint = (cos_squared_azimuth != 0.0)
                                              ? cos_angular_distance -
                                                    2.0 * sin_reduced_first * sin_reduced_second / cos_squared_azimuth
                                              : 0.0;

            nk_f64_t correction_factor = flattening / 16.0 * cos_squared_azimuth *
                                         (4.0 + flattening * (4.0 - 3.0 * cos_squared_azimuth));

            lambda_previous = lambda;
            lambda = longitude_difference + (1.0 - correction_factor) * flattening * sin_azimuth *
                                                (angular_distance + correction_factor * sin_angular_distance *
                                                                        (cos_double_angular_midpoint +
                                                                         correction_factor * cos_angular_distance *
                                                                             (-1.0 + 2.0 * cos_double_angular_midpoint *
                                                                                         cos_double_angular_midpoint)));

            iteration++;
        } while (nk_f64_abs_(lambda - lambda_previous) > NK_VINCENTY_CONVERGENCE_THRESHOLD_F64 &&
                 iteration < NK_VINCENTY_MAX_ITERATIONS);

        if (coincident) {
            results[i] = 0.0;
            continue;
        }

        // Final distance calculation
        nk_f64_t u_squared = cos_squared_azimuth *
                             (equatorial_radius * equatorial_radius - polar_radius * polar_radius) /
                             (polar_radius * polar_radius);
        nk_f64_t series_a = 1.0 + u_squared / 16384.0 *
                                      (4096.0 + u_squared * (-768.0 + u_squared * (320.0 - 175.0 * u_squared)));
        nk_f64_t series_b = u_squared / 1024.0 * (256.0 + u_squared * (-128.0 + u_squared * (74.0 - 47.0 * u_squared)));

        nk_f64_t angular_correction =
            series_b * sin_angular_distance *
            (cos_double_angular_midpoint +
             series_b / 4.0 *
                 (cos_angular_distance * (-1.0 + 2.0 * cos_double_angular_midpoint * cos_double_angular_midpoint) -
                  series_b / 6.0 * cos_double_angular_midpoint *
                      (-3.0 + 4.0 * sin_angular_distance * sin_angular_distance) *
                      (-3.0 + 4.0 * cos_double_angular_midpoint * cos_double_angular_midpoint)));

        results[i] = polar_radius * series_a * (angular_distance - angular_correction);
    }
}

NK_PUBLIC void nk_vincenty_f32_serial(              //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results) {

    nk_f32_t const equatorial_radius = (nk_f32_t)NK_EARTH_ELLIPSOID_EQUATORIAL_RADIUS;
    nk_f32_t const polar_radius = (nk_f32_t)NK_EARTH_ELLIPSOID_POLAR_RADIUS;
    nk_f32_t const flattening = 1.0f / (nk_f32_t)NK_EARTH_ELLIPSOID_INVERSE_FLATTENING;
    nk_f32_t const convergence_threshold = 1e-7f;

    for (nk_size_t i = 0; i != n; ++i) {
        nk_f32_t first_latitude = a_lats[i];
        nk_f32_t second_latitude = b_lats[i];
        nk_f32_t longitude_difference = b_lons[i] - a_lons[i];

        // Reduced latitudes on the auxiliary sphere
        nk_f32_t tan_reduced_first = (1.0f - flattening) * (nk_f32_sin(first_latitude) / nk_f32_cos(first_latitude));
        nk_f32_t tan_reduced_second = (1.0f - flattening) * (nk_f32_sin(second_latitude) / nk_f32_cos(second_latitude));
        nk_f32_t cos_reduced_first = 1.0f / nk_f32_sqrt_serial(1.0f + tan_reduced_first * tan_reduced_first);
        nk_f32_t sin_reduced_first = tan_reduced_first * cos_reduced_first;
        nk_f32_t cos_reduced_second = 1.0f / nk_f32_sqrt_serial(1.0f + tan_reduced_second * tan_reduced_second);
        nk_f32_t sin_reduced_second = tan_reduced_second * cos_reduced_second;

        // Iterative convergence of lambda (difference in longitude on auxiliary sphere)
        nk_f32_t lambda = longitude_difference;
        nk_f32_t lambda_previous = longitude_difference;
        nk_f32_t sin_angular_distance, cos_angular_distance, angular_distance;
        nk_f32_t sin_azimuth, cos_squared_azimuth, cos_double_angular_midpoint;
        nk_u32_t iteration = 0;

        // Check for coincident points early
        nk_u32_t coincident = 0;
        do {
            nk_f32_t sin_lambda = nk_f32_sin(lambda);
            nk_f32_t cos_lambda = nk_f32_cos(lambda);

            nk_f32_t cross_term = cos_reduced_second * sin_lambda;
            nk_f32_t mixed_term = cos_reduced_first * sin_reduced_second -
                                  sin_reduced_first * cos_reduced_second * cos_lambda;
            sin_angular_distance = nk_f32_sqrt_serial(cross_term * cross_term + mixed_term * mixed_term);

            if (sin_angular_distance == 0.0f) {
                coincident = 1;
                break;
            }

            cos_angular_distance = sin_reduced_first * sin_reduced_second +
                                   cos_reduced_first * cos_reduced_second * cos_lambda;
            angular_distance = nk_f32_atan2(sin_angular_distance, cos_angular_distance);

            sin_azimuth = cos_reduced_first * cos_reduced_second * sin_lambda / sin_angular_distance;
            cos_squared_azimuth = 1.0f - sin_azimuth * sin_azimuth;

            // Handle equatorial geodesic case
            cos_double_angular_midpoint = (cos_squared_azimuth != 0.0f)
                                              ? cos_angular_distance -
                                                    2.0f * sin_reduced_first * sin_reduced_second / cos_squared_azimuth
                                              : 0.0f;

            nk_f32_t correction_factor = flattening / 16.0f * cos_squared_azimuth *
                                         (4.0f + flattening * (4.0f - 3.0f * cos_squared_azimuth));

            lambda_previous = lambda;
            lambda = longitude_difference +
                     (1.0f - correction_factor) * flattening * sin_azimuth *
                         (angular_distance +
                          correction_factor * sin_angular_distance *
                              (cos_double_angular_midpoint +
                               correction_factor * cos_angular_distance *
                                   (-1.0f + 2.0f * cos_double_angular_midpoint * cos_double_angular_midpoint)));

            iteration++;
        } while (nk_f32_abs_(lambda - lambda_previous) > convergence_threshold &&
                 iteration < NK_VINCENTY_MAX_ITERATIONS);

        if (coincident) {
            results[i] = 0.0;
            continue;
        }

        // Final distance calculation
        nk_f32_t u_squared = cos_squared_azimuth *
                             (equatorial_radius * equatorial_radius - polar_radius * polar_radius) /
                             (polar_radius * polar_radius);
        nk_f32_t series_a = 1.0f + u_squared / 16384.0f *
                                       (4096.0f + u_squared * (-768.0f + u_squared * (320.0f - 175.0f * u_squared)));
        nk_f32_t series_b = u_squared / 1024.0f *
                            (256.0f + u_squared * (-128.0f + u_squared * (74.0f - 47.0f * u_squared)));

        nk_f32_t angular_correction =
            series_b * sin_angular_distance *
            (cos_double_angular_midpoint +
             series_b / 4.0f *
                 (cos_angular_distance * (-1.0f + 2.0f * cos_double_angular_midpoint * cos_double_angular_midpoint) -
                  series_b / 6.0f * cos_double_angular_midpoint *
                      (-3.0f + 4.0f * sin_angular_distance * sin_angular_distance) *
                      (-3.0f + 4.0f * cos_double_angular_midpoint * cos_double_angular_midpoint)));

        results[i] = polar_radius * series_a * (angular_distance - angular_correction);
    }
}

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_GEOSPATIAL_SERIAL_H
