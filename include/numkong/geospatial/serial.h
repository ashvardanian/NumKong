/**
 *  @file include/numkong/geospatial/serial.h
 *  @author Ash Vardanian
 *  @date December 21, 2025
 *  @brief Serial geospatial distances, terrain visibility and the packs every capability reads.
 *
 *  @sa include/numkong/geospatial.h
 */
#ifndef NUMKONG_GEOSPATIAL_SERIAL_H
#define NUMKONG_GEOSPATIAL_SERIAL_H

#include "numkong/types.h"
#include "numkong/scalar/serial.h" // `nk_sqrt_f64_serial_`, `nk_sin_f64_serial_`, `nk_atan2_f64_serial_`, `nk_dot2_f64_serial_`

#if defined(__cplusplus)
extern "C" {
#endif

/** Opens every Haversine and Vincenty pack; three planar arrays of @c f64 follow, each on its own
 *  @c nk_default_alignment_k boundary: the x, y and z of unit vectors for Haversine, and the
 *  longitude and the sine and cosine of the reduced latitude for Vincenty. */
typedef struct nk_geospatial_packed_header_t {

    /** Number of points. */
    nk_u64_t points;

    /** Values from the start of one array to the next. */
    nk_u64_t axis_stride;
} nk_geospatial_packed_header_t;

/** Opens every terrain pack; the maxima of each level follow, the finest first, row-major. Block
 *  (i, j) of level k holds the largest height over rows i · 2ᵏ − 1 to (i + 1) · 2ᵏ + 1 and the
 *  same span of columns, clipped to the grid: one ring wider than the block, so a sample
 *  interpolated across its edge stays under its maximum. */
typedef struct nk_terrain_packed_header_t {

    /** Grid rows and columns. */
    nk_u64_t rows;
    nk_u64_t columns;

    /** Bytes between consecutive rows of the heights the pack was built from. */
    nk_u64_t heights_stride;

    /** Levels of maxima, the coarsest holding a single block. */
    nk_u64_t levels;

    /** Ground length of one row step and of one column step, in metres. */
    nk_f64_t row_metres;
    nk_f64_t column_metres;
} nk_terrain_packed_header_t;

/** The first array of a Haversine or Vincenty pack, with its point count and the values from
 *  one array to the next. */
NUMKONG_INLINE nk_f64_t const *nk_geospatial_packed_axes_f64_serial_(void const *packed, nk_size_t *points,
                                                                     nk_size_t *axis_stride) {
    nk_geospatial_packed_header_t const *header = (nk_geospatial_packed_header_t const *)packed;
    *points = (nk_size_t)header->points, *axis_stride = (nk_size_t)header->axis_stride;
    return (nk_f64_t const *)((char const *)packed + nk_default_alignment_k);
}

/** Blocks along one axis of terrain level @p level over @p cells cells. */
NUMKONG_INLINE nk_size_t nk_terrain_level_blocks_f32_serial_(nk_size_t cells, nk_size_t level) {
    nk_size_t const span = cells > 1 ? cells - 1 : 1;
    return nk_size_divide_round_up_(span, (nk_size_t)1 << level);
}

/** Levels of maxima a @p rows × @p columns grid needs until one block covers it. */
NUMKONG_INLINE nk_size_t nk_terrain_levels_f32_serial_(nk_size_t rows, nk_size_t columns) {
    nk_size_t const span = rows > columns ? rows - 1 : columns - 1;
    nk_size_t levels = 0;
    while (((nk_size_t)1 << levels) < span) ++levels;
    return levels;
}

/** The maxima of level @p level of a terrain pack, row-major over its blocks. */
NUMKONG_INLINE nk_f32_t const *nk_terrain_packed_level_f32_serial_(void const *packed, nk_size_t level) {
    nk_terrain_packed_header_t const *header = (nk_terrain_packed_header_t const *)packed;
    nk_f32_t const *maxima = (nk_f32_t const *)((char const *)packed + nk_default_alignment_k);
    for (nk_size_t finer = 1; finer < level; ++finer)
        maxima += nk_terrain_level_blocks_f32_serial_((nk_size_t)header->rows, finer) *
                  nk_terrain_level_blocks_f32_serial_((nk_size_t)header->columns, finer);
    return maxima;
}

/** The height in row @p row and column @p column of a grid with @p stride bytes between rows. */
NUMKONG_INLINE nk_f32_t nk_terrain_height_f32_serial_(nk_f32_t const *heights, nk_size_t stride, nk_size_t row,
                                                      nk_size_t column) {
    return *(nk_f32_t const *)((char const *)heights + row * stride + column * sizeof(nk_f32_t));
}

#if NUMKONG_TARGET_SERIAL
/*  Keep the serial instantiations below actually scalar, regardless of build type.
 *  See dots/serial.h for rationale. */
#if defined(__clang__)
#pragma clang attribute push(__attribute__((noinline)), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-slp-vectorize", "no-ipa-cp-clone", "no-inline")
#endif

NUMKONG_API nk_status_t nk_sqeuclideans3d_i32_serial(nk_i32_t const *a, nk_size_t rows, nk_i32_t const *b,
                                                     nk_size_t columns, nk_u64_t *results, nk_size_t results_stride,
                                                     nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_u64_t *output = (nk_u64_t *)((char *)results + row * results_stride);
        for (nk_size_t column = 0; column != columns; ++column) {
            nk_u64_t sum = 0;
            for (nk_size_t axis = 0; axis != 3; ++axis) {
                nk_i64_t const difference = (nk_i64_t)a[row * 3 + axis] - (nk_i64_t)b[column * 3 + axis];
                nk_u64_t const magnitude = (nk_u64_t)(difference < 0 ? -difference : difference);
                sum += magnitude * magnitude;
            }
            output[column] = sum;
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_sqeuclideans3d_f64_serial(nk_f64_t const *a, nk_size_t rows, nk_f64_t const *b,
                                                     nk_size_t columns, nk_f64_t *results, nk_size_t results_stride,
                                                     nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f64_t *output = (nk_f64_t *)((char *)results + row * results_stride);
        for (nk_size_t column = 0; column != columns; ++column) {
            nk_f64_t sum = 0, compensation = 0;
            for (nk_size_t axis = 0; axis != 3; ++axis) {
                nk_f64_t const difference = a[row * 3 + axis] - b[column * 3 + axis];
                nk_dot2_f64_serial_(&sum, &compensation, difference, difference);
            }
            output[column] = sum + compensation;
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_haversines_pack_size_f64_serial(nk_size_t points, nk_size_t *bytes) {
    *bytes = nk_default_alignment_k +
             3 * nk_size_round_up_to_multiple_(points, nk_default_alignment_k / sizeof(nk_f64_t)) * sizeof(nk_f64_t);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_vincentys_pack_size_f64_serial(nk_size_t points, nk_size_t *bytes) {
    *bytes = nk_default_alignment_k +
             3 * nk_size_round_up_to_multiple_(points, nk_default_alignment_k / sizeof(nk_f64_t)) * sizeof(nk_f64_t);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_haversines_pack_f64_serial(nk_f64_t const *latitudes, nk_f64_t const *longitudes,
                                                      nk_size_t count, void *packed, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const axis_stride = nk_size_round_up_to_multiple_(count, nk_default_alignment_k / sizeof(nk_f64_t));
    nk_geospatial_packed_header_t *header = (nk_geospatial_packed_header_t *)packed;
    header->points = count, header->axis_stride = axis_stride;
    nk_f64_t *packed_x = (nk_f64_t *)((char *)packed + nk_default_alignment_k);
    nk_f64_t *packed_y = packed_x + axis_stride, *packed_z = packed_y + axis_stride;
    for (nk_size_t point = 0; point != count; ++point) {
        nk_f64_t const cos_latitude = nk_cos_f64_serial_(latitudes[point]);
        packed_x[point] = cos_latitude * nk_cos_f64_serial_(longitudes[point]);
        packed_y[point] = cos_latitude * nk_sin_f64_serial_(longitudes[point]);
        packed_z[point] = nk_sin_f64_serial_(latitudes[point]);
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_vincentys_pack_f64_serial(nk_f64_t const *latitudes, nk_f64_t const *longitudes,
                                                     nk_size_t count, void *packed, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f64_t const flattening = 1.0 / NUMKONG_EARTH_ELLIPSOID_INVERSE_FLATTENING;
    nk_size_t const axis_stride = nk_size_round_up_to_multiple_(count, nk_default_alignment_k / sizeof(nk_f64_t));
    nk_geospatial_packed_header_t *header = (nk_geospatial_packed_header_t *)packed;
    header->points = count, header->axis_stride = axis_stride;
    nk_f64_t *packed_longitudes = (nk_f64_t *)((char *)packed + nk_default_alignment_k);
    nk_f64_t *packed_sines = packed_longitudes + axis_stride, *packed_cosines = packed_sines + axis_stride;
    for (nk_size_t point = 0; point != count; ++point) {
        // The reduced latitude through atan2 stays finite at the poles, where tan(φ) does not.
        nk_f64_t const reduced = nk_atan2_f64_serial_((1.0 - flattening) * nk_sin_f64_serial_(latitudes[point]),
                                                      nk_cos_f64_serial_(latitudes[point]));
        packed_longitudes[point] = longitudes[point];
        packed_sines[point] = nk_sin_f64_serial_(reduced);
        packed_cosines[point] = nk_cos_f64_serial_(reduced);
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_haversines_packed_f64_serial(void const *a_packed, void const *b_packed, nk_f64_t *results,
                                                        nk_size_t results_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t rows, columns, a_stride, b_stride;
    nk_f64_t const *a_packed_x = nk_geospatial_packed_axes_f64_serial_(a_packed, &rows, &a_stride);
    nk_f64_t const *b_packed_x = nk_geospatial_packed_axes_f64_serial_(b_packed, &columns, &b_stride);
    nk_f64_t const *a_packed_y = a_packed_x + a_stride, *a_packed_z = a_packed_y + a_stride;
    nk_f64_t const *b_packed_y = b_packed_x + b_stride, *b_packed_z = b_packed_y + b_stride;
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f64_t *output = (nk_f64_t *)((char *)results + row * results_stride);
        for (nk_size_t column = 0; column != columns; ++column) {
            nk_f64_t const difference_x = a_packed_x[row] - b_packed_x[column];
            nk_f64_t const difference_y = a_packed_y[row] - b_packed_y[column];
            nk_f64_t const difference_z = a_packed_z[row] - b_packed_z[column];
            nk_f64_t const sum_x = a_packed_x[row] + b_packed_x[column];
            nk_f64_t const sum_y = a_packed_y[row] + b_packed_y[column];
            nk_f64_t const sum_z = a_packed_z[row] + b_packed_z[column];
            nk_f64_t chord = 0, chord_compensation = 0, span = 0, span_compensation = 0;
            nk_dot2_f64_serial_(&chord, &chord_compensation, difference_x, difference_x);
            nk_dot2_f64_serial_(&chord, &chord_compensation, difference_y, difference_y);
            nk_dot2_f64_serial_(&chord, &chord_compensation, difference_z, difference_z);
            nk_dot2_f64_serial_(&span, &span_compensation, sum_x, sum_x);
            nk_dot2_f64_serial_(&span, &span_compensation, sum_y, sum_y);
            nk_dot2_f64_serial_(&span, &span_compensation, sum_z, sum_z);
            nk_f64_t const angle = 2.0 * nk_atan2_f64_serial_(nk_sqrt_f64_serial_(chord + chord_compensation),
                                                              nk_sqrt_f64_serial_(span + span_compensation));
            output[column] = NUMKONG_EARTH_MEAN_RADIUS * angle;
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_vincentys_packed_f64_serial(void const *a_packed, void const *b_packed, nk_f64_t *results,
                                                       nk_size_t results_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f64_t const equatorial_radius = NUMKONG_EARTH_ELLIPSOID_EQUATORIAL_RADIUS;
    nk_f64_t const flattening = 1.0 / NUMKONG_EARTH_ELLIPSOID_INVERSE_FLATTENING;
    nk_f64_t const polar_radius = equatorial_radius * (1.0 - flattening);
    nk_size_t rows, columns, a_stride, b_stride;
    nk_f64_t const *a_packed_longitudes = nk_geospatial_packed_axes_f64_serial_(a_packed, &rows, &a_stride);
    nk_f64_t const *b_packed_longitudes = nk_geospatial_packed_axes_f64_serial_(b_packed, &columns, &b_stride);
    nk_f64_t const *a_packed_sines = a_packed_longitudes + a_stride, *a_packed_cosines = a_packed_sines + a_stride;
    nk_f64_t const *b_packed_sines = b_packed_longitudes + b_stride, *b_packed_cosines = b_packed_sines + b_stride;

    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f64_t *output = (nk_f64_t *)((char *)results + row * results_stride);
        nk_f64_t const sin_reduced_first = a_packed_sines[row], cos_reduced_first = a_packed_cosines[row];
        for (nk_size_t column = 0; column != columns; ++column) {
            nk_f64_t const sin_reduced_second = b_packed_sines[column], cos_reduced_second = b_packed_cosines[column];
            nk_f64_t const longitude_difference = b_packed_longitudes[column] - a_packed_longitudes[row];

            // Iterative convergence of lambda, the difference in longitude on the auxiliary sphere
            nk_f64_t lambda = longitude_difference, lambda_previous;
            nk_f64_t sin_angular_distance, cos_angular_distance, angular_distance = 0;
            nk_f64_t cos_squared_azimuth = 1, cos_double_angular_midpoint = 0;
            nk_u32_t iteration = 0, coincident = 0;
            do {
                nk_f64_t const sin_lambda = nk_sin_f64_serial_(lambda), cos_lambda = nk_cos_f64_serial_(lambda);
                nk_f64_t const cross_term = cos_reduced_second * sin_lambda;
                nk_f64_t const mixed_term = cos_reduced_first * sin_reduced_second -
                                            sin_reduced_first * cos_reduced_second * cos_lambda;
                sin_angular_distance = nk_sqrt_f64_serial_(cross_term * cross_term + mixed_term * mixed_term);
                if (sin_angular_distance == 0.0) {
                    coincident = 1;
                    break;
                }
                cos_angular_distance = sin_reduced_first * sin_reduced_second +
                                       cos_reduced_first * cos_reduced_second * cos_lambda;
                angular_distance = nk_atan2_f64_serial_(sin_angular_distance, cos_angular_distance);
                nk_f64_t const sin_azimuth = cos_reduced_first * cos_reduced_second * sin_lambda / sin_angular_distance;
                cos_squared_azimuth = 1.0 - sin_azimuth * sin_azimuth;
                // An equatorial geodesic has no midpoint term
                cos_double_angular_midpoint = cos_squared_azimuth != 0.0
                                                  ? cos_angular_distance - 2.0 * sin_reduced_first *
                                                                               sin_reduced_second / cos_squared_azimuth
                                                  : 0.0;
                nk_f64_t const correction_factor = flattening / 16.0 * cos_squared_azimuth *
                                                   (4.0 + flattening * (4.0 - 3.0 * cos_squared_azimuth));
                lambda_previous = lambda;
                lambda = longitude_difference +
                         (1.0 - correction_factor) * flattening * sin_azimuth *
                             (angular_distance +
                              correction_factor * sin_angular_distance *
                                  (cos_double_angular_midpoint +
                                   correction_factor * cos_angular_distance *
                                       (-1.0 + 2.0 * cos_double_angular_midpoint * cos_double_angular_midpoint)));
                iteration++;
            } while (nk_f64_abs_(lambda - lambda_previous) > NUMKONG_VINCENTY_CONVERGENCE_THRESHOLD_F64 &&
                     iteration < NUMKONG_VINCENTY_MAX_ITERATIONS);

            if (coincident) {
                output[column] = 0.0;
                continue;
            }
            nk_f64_t const u_squared = cos_squared_azimuth *
                                       (equatorial_radius * equatorial_radius - polar_radius * polar_radius) /
                                       (polar_radius * polar_radius);
            nk_f64_t const series_a =
                1.0 + u_squared / 16384.0 * (4096.0 + u_squared * (-768.0 + u_squared * (320.0 - 175.0 * u_squared)));
            nk_f64_t const series_b = u_squared / 1024.0 *
                                      (256.0 + u_squared * (-128.0 + u_squared * (74.0 - 47.0 * u_squared)));
            nk_f64_t const angular_correction =
                series_b * sin_angular_distance *
                (cos_double_angular_midpoint +
                 series_b / 4.0 *
                     (cos_angular_distance * (-1.0 + 2.0 * cos_double_angular_midpoint * cos_double_angular_midpoint) -
                      series_b / 6.0 * cos_double_angular_midpoint *
                          (-3.0 + 4.0 * sin_angular_distance * sin_angular_distance) *
                          (-3.0 + 4.0 * cos_double_angular_midpoint * cos_double_angular_midpoint)));
            output[column] = polar_radius * series_a * (angular_distance - angular_correction);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_terrain_pack_size_f32_serial(nk_size_t rows, nk_size_t columns, nk_size_t *bytes) {
    nk_size_t total = nk_default_alignment_k;
    nk_size_t const levels = nk_terrain_levels_f32_serial_(rows, columns);
    for (nk_size_t level = 1; level <= levels; ++level)
        total += nk_terrain_level_blocks_f32_serial_(rows, level) *
                 nk_terrain_level_blocks_f32_serial_(columns, level) * sizeof(nk_f32_t);
    *bytes = total;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_terrain_pack_f32_serial(nk_f32_t const *heights, nk_size_t rows, nk_size_t columns,
                                                   nk_size_t heights_stride, nk_f64_t row_metres,
                                                   nk_f64_t column_metres, void *packed, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!rows || !columns) return nk_unexpected_dimensions_k;
    nk_size_t const levels = nk_terrain_levels_f32_serial_(rows, columns);
    nk_terrain_packed_header_t *header = (nk_terrain_packed_header_t *)packed;
    header->rows = rows, header->columns = columns, header->heights_stride = heights_stride;
    header->levels = levels, header->row_metres = row_metres, header->column_metres = column_metres;

    // Level 1 reads the heights directly: rows 2i − 1 to 2i + 3, clipped.
    // NaN never wins a comparison.
    nk_f32_t const lowest = -NUMKONG_F32_INF;
    for (nk_size_t level = 1; level <= levels; ++level) {
        nk_f32_t *maxima = (nk_f32_t *)nk_terrain_packed_level_f32_serial_(packed, level);
        nk_size_t const block_rows = nk_terrain_level_blocks_f32_serial_(rows, level);
        nk_size_t const block_columns = nk_terrain_level_blocks_f32_serial_(columns, level);
        for (nk_size_t block_row = 0; block_row != block_rows; ++block_row) {
            for (nk_size_t block_column = 0; block_column != block_columns; ++block_column) {
                nk_f32_t maximum = lowest;
                if (level == 1) {
                    nk_size_t const first_row = block_row * 2 ? block_row * 2 - 1 : 0;
                    nk_size_t const last_row = block_row * 2 + 3 < rows ? block_row * 2 + 3 : rows - 1;
                    nk_size_t const first_column = block_column * 2 ? block_column * 2 - 1 : 0;
                    nk_size_t const last_column = block_column * 2 + 3 < columns ? block_column * 2 + 3 : columns - 1;
                    for (nk_size_t row = first_row; row <= last_row; ++row)
                        for (nk_size_t column = first_column; column <= last_column; ++column) {
                            nk_f32_t const height = nk_terrain_height_f32_serial_(heights, heights_stride, row, column);
                            if (height > maximum) maximum = height;
                        }
                }
                else {
                    // Children 2i and 2i + 1 cover the same rows as their parent, rings included.
                    nk_f32_t const *finer = nk_terrain_packed_level_f32_serial_(packed, level - 1);
                    nk_size_t const finer_rows = nk_terrain_level_blocks_f32_serial_(rows, level - 1);
                    nk_size_t const finer_columns = nk_terrain_level_blocks_f32_serial_(columns, level - 1);
                    for (nk_size_t child_row = block_row * 2; child_row < block_row * 2 + 2 && child_row < finer_rows;
                         ++child_row)
                        for (nk_size_t child_column = block_column * 2;
                             child_column < block_column * 2 + 2 && child_column < finer_columns; ++child_column) {
                            nk_f32_t const child = finer[child_row * finer_columns + child_column];
                            if (child > maximum) maximum = child;
                        }
                }
                maxima[block_row * block_columns + block_column] = maximum;
            }
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_terrain_heights_packed_f32_serial(nk_f32_t const *heights, void const *packed,
                                                             nk_f32_t const *points, nk_size_t count, nk_f32_t *results,
                                                             nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_terrain_packed_header_t const *header = (nk_terrain_packed_header_t const *)packed;
    nk_size_t const rows = (nk_size_t)header->rows, columns = (nk_size_t)header->columns;
    nk_size_t const stride = (nk_size_t)header->heights_stride;
    nk_fui32_t off_grid;
    off_grid.u = 0x7FC00000u;
    for (nk_size_t point = 0; point != count; ++point) {
        nk_f64_t const column = points[point * 2], row = points[point * 2 + 1];
        if (!(column >= 0 && column <= (nk_f64_t)(columns - 1) && row >= 0 && row <= (nk_f64_t)(rows - 1))) {
            results[point] = off_grid.f;
            continue;
        }
        nk_size_t column_first = (nk_size_t)column, row_first = (nk_size_t)row;
        if (column_first + 1 == columns && columns > 1) --column_first;
        if (row_first + 1 == rows && rows > 1) --row_first;
        nk_size_t const column_second = columns > 1 ? column_first + 1 : column_first;
        nk_size_t const row_second = rows > 1 ? row_first + 1 : row_first;
        nk_f64_t const column_fraction = column - (nk_f64_t)column_first, row_fraction = row - (nk_f64_t)row_first;
        nk_f64_t const north_west = nk_terrain_height_f32_serial_(heights, stride, row_first, column_first);
        nk_f64_t const north_east = nk_terrain_height_f32_serial_(heights, stride, row_first, column_second);
        nk_f64_t const south_west = nk_terrain_height_f32_serial_(heights, stride, row_second, column_first);
        nk_f64_t const south_east = nk_terrain_height_f32_serial_(heights, stride, row_second, column_second);
        nk_f64_t const north = north_west + column_fraction * (north_east - north_west);
        nk_f64_t const south = south_west + column_fraction * (south_east - south_west);
        results[point] = (nk_f32_t)(north + row_fraction * (south - north));
    }
    return nk_success_k;
}

/** One sight line in cell coordinates: p(u) = start + u · delta for u in (0, 1), its height
 *  at u, and the scale of the curvature drop along it. */
typedef struct nk_terrain_ray_t_ {
    nk_f64_t column_start, column_delta;
    nk_f64_t row_start, row_delta;
    nk_f64_t height_start, height_delta;
    nk_f64_t distance, drop_scale;
} nk_terrain_ray_t_;

/** Folds one terrain sample at @p u, interpolated between @p first and @p second at
 *  @p fraction, into the running @p clearance; missing heights never block. */
NUMKONG_INLINE void nk_terrain_fold_sample_f32_serial_(nk_terrain_ray_t_ const *ray, nk_f64_t u, nk_f64_t first,
                                                       nk_f64_t second, nk_f64_t fraction, nk_f64_t *clearance) {
    nk_f64_t const ground = fraction == 0 ? first : first + fraction * (second - first);
    if (ground != ground) return;
    nk_f64_t const along = u * ray->distance;
    nk_f64_t const gap = ray->height_start + u * ray->height_delta - (ground - ray->drop_scale * along * along);
    if (gap < *clearance) *clearance = gap;
}

/** A cell coordinate along a sight line clamped into [0, @p last]: both ends lie on the grid,
 *  so only a rounding near an edge can step outside it. */
NUMKONG_INLINE nk_f64_t nk_terrain_clamp_f64_serial_(nk_f64_t coordinate, nk_f64_t last) {
    return coordinate < 0 ? 0 : (coordinate > last ? last : coordinate);
}

/** Folds every sample whose crossing falls between @p u_begin and @p u_end into @p clearance:
 *  the column lines, then the row lines, each interpolated along its own line. */
NUMKONG_INLINE void nk_terrain_fold_samples_f32_serial_(nk_f32_t const *heights,
                                                        nk_terrain_packed_header_t const *header,
                                                        nk_terrain_ray_t_ const *ray, nk_f64_t u_begin, nk_f64_t u_end,
                                                        nk_f64_t *clearance) {
    nk_size_t const rows = (nk_size_t)header->rows, columns = (nk_size_t)header->columns;
    nk_size_t const stride = (nk_size_t)header->heights_stride;
    nk_f64_t const last_row = (nk_f64_t)(rows - 1), last_column = (nk_f64_t)(columns - 1);
    if (ray->column_delta != 0) {
        nk_f64_t low = nk_terrain_clamp_f64_serial_(ray->column_start + u_begin * ray->column_delta, last_column);
        nk_f64_t high = nk_terrain_clamp_f64_serial_(ray->column_start + u_end * ray->column_delta, last_column);
        if (low > high) {
            nk_f64_t const swapped = low;
            low = high, high = swapped;
        }
        nk_size_t const first = (nk_size_t)low + ((nk_f64_t)(nk_size_t)low < low);
        for (nk_size_t column = first; (nk_f64_t)column <= high; ++column) {
            nk_f64_t const u = ((nk_f64_t)column - ray->column_start) / ray->column_delta;
            if (!(u > 0 && u < 1 && u >= u_begin && u <= u_end)) continue;
            nk_f64_t const row = nk_terrain_clamp_f64_serial_(ray->row_start + u * ray->row_delta, last_row);
            nk_size_t const row_first = (nk_size_t)row;
            nk_f64_t const fraction = row - (nk_f64_t)row_first;
            nk_size_t const row_second = row_first + 1 < rows ? row_first + 1 : row_first;
            nk_terrain_fold_sample_f32_serial_(
                ray, u, nk_terrain_height_f32_serial_(heights, stride, row_first, column),
                nk_terrain_height_f32_serial_(heights, stride, row_second, column), fraction, clearance);
        }
    }
    if (ray->row_delta != 0) {
        nk_f64_t low = nk_terrain_clamp_f64_serial_(ray->row_start + u_begin * ray->row_delta, last_row);
        nk_f64_t high = nk_terrain_clamp_f64_serial_(ray->row_start + u_end * ray->row_delta, last_row);
        if (low > high) {
            nk_f64_t const swapped = low;
            low = high, high = swapped;
        }
        nk_size_t const first = (nk_size_t)low + ((nk_f64_t)(nk_size_t)low < low);
        for (nk_size_t row = first; (nk_f64_t)row <= high; ++row) {
            nk_f64_t const u = ((nk_f64_t)row - ray->row_start) / ray->row_delta;
            if (!(u > 0 && u < 1 && u >= u_begin && u <= u_end)) continue;
            nk_f64_t const column = nk_terrain_clamp_f64_serial_(ray->column_start + u * ray->column_delta,
                                                                 last_column);
            nk_size_t const column_first = (nk_size_t)column;
            nk_f64_t const fraction = column - (nk_f64_t)column_first;
            nk_size_t const column_second = column_first + 1 < columns ? column_first + 1 : column_first;
            nk_terrain_fold_sample_f32_serial_(
                ray, u, nk_terrain_height_f32_serial_(heights, stride, row, column_first),
                nk_terrain_height_f32_serial_(heights, stride, row, column_second), fraction, clearance);
        }
    }
}

/** Walks the blocks of @p level the ray crosses between @p u_begin and @p u_end, skipping each
 *  whose maximum cannot lower @p clearance and descending into the rest, down to the samples
 *  themselves at level zero. */
static void nk_terrain_visit_serial_(nk_f32_t const *heights, void const *packed, nk_terrain_ray_t_ const *ray,
                                     nk_size_t level, nk_f64_t u_begin, nk_f64_t u_end, nk_f64_t *clearance) {
    nk_terrain_packed_header_t const *header = (nk_terrain_packed_header_t const *)packed;
    if (level == 0) {
        nk_terrain_fold_samples_f32_serial_(heights, header, ray, u_begin, u_end, clearance);
        return;
    }
    nk_f32_t const *maxima = nk_terrain_packed_level_f32_serial_(packed, level);
    nk_size_t const block_rows = nk_terrain_level_blocks_f32_serial_((nk_size_t)header->rows, level);
    nk_size_t const block_columns = nk_terrain_level_blocks_f32_serial_((nk_size_t)header->columns, level);
    nk_f64_t const span = (nk_f64_t)((nk_size_t)1 << level);
    nk_f64_t const infinity = NUMKONG_F64_INF;

    nk_f64_t u = u_begin;
    nk_f64_t const last_row = (nk_f64_t)(header->rows - 1), last_column = (nk_f64_t)(header->columns - 1);
    nk_size_t block_row = (nk_size_t)(nk_terrain_clamp_f64_serial_(ray->row_start + u * ray->row_delta, last_row) /
                                      span);
    nk_size_t block_column =
        (nk_size_t)(nk_terrain_clamp_f64_serial_(ray->column_start + u * ray->column_delta, last_column) / span);
    if (block_row >= block_rows) block_row = block_rows - 1;
    if (block_column >= block_columns) block_column = block_columns - 1;
    for (;;) {
        nk_f64_t u_row = infinity, u_column = infinity;
        if (ray->row_delta > 0) u_row = ((nk_f64_t)(block_row + 1) * span - ray->row_start) / ray->row_delta;
        else if (ray->row_delta < 0) u_row = ((nk_f64_t)block_row * span - ray->row_start) / ray->row_delta;
        if (ray->column_delta > 0)
            u_column = ((nk_f64_t)(block_column + 1) * span - ray->column_start) / ray->column_delta;
        else if (ray->column_delta < 0)
            u_column = ((nk_f64_t)block_column * span - ray->column_start) / ray->column_delta;
        nk_f64_t u_next = u_row < u_column ? u_row : u_column;
        if (u_next > u_end) u_next = u_end;
        if (u_next < u) u_next = u;

        // The sight line is linear in u and the drop grows with u: the segment's start bounds both.
        nk_f64_t const maximum = maxima[block_row * block_columns + block_column];
        nk_f64_t const line_begin = ray->height_start + u * ray->height_delta;
        nk_f64_t const line_end = ray->height_start + u_next * ray->height_delta;
        nk_f64_t const along = u * ray->distance;
        nk_f64_t const bound = (line_begin < line_end ? line_begin : line_end) -
                               (maximum - ray->drop_scale * along * along);
        if (bound < *clearance) nk_terrain_visit_serial_(heights, packed, ray, level - 1, u, u_next, clearance);

        if (u_next >= u_end) break;
        int const crosses_row = u_row <= u_column, crosses_column = u_column <= u_row;
        if (crosses_row) {
            if (ray->row_delta > 0 ? block_row + 1 >= block_rows : block_row == 0) break;
            block_row = ray->row_delta > 0 ? block_row + 1 : block_row - 1;
        }
        if (crosses_column) {
            if (ray->column_delta > 0 ? block_column + 1 >= block_columns : block_column == 0) break;
            block_column = ray->column_delta > 0 ? block_column + 1 : block_column - 1;
        }
        u = u_next;
    }
}

/** Whether a target at @p east, @p north and @p up metres from a viewpoint falls outside its
 *  view of @p azimuth, @p elevation, @p field and @p range. */
NUMKONG_INLINE int nk_terrain_outside_view_f32_serial_(nk_f64_t east, nk_f64_t north, nk_f64_t up, nk_f64_t azimuth,
                                                       nk_f64_t elevation, nk_f64_t field, nk_f64_t range) {
    nk_f64_t const slant = nk_sqrt_f64_serial_(east * east + north * north + up * up);
    if (range == range && slant > range) return 1;
    if (azimuth != azimuth || !(field < 2 * 3.14159265358979323846)) return 0;
    nk_f64_t const half_field = field / 2;
    if (elevation != elevation) {
        nk_f64_t const ground = nk_sqrt_f64_serial_(east * east + north * north);
        if (ground == 0) return 0;
        nk_f64_t const cosine = (east * nk_sin_f64_serial_(azimuth) + north * nk_cos_f64_serial_(azimuth)) / ground;
        return cosine < nk_cos_f64_serial_(half_field);
    }
    if (slant == 0) return 0;
    nk_f64_t const cos_elevation = nk_cos_f64_serial_(elevation);
    nk_f64_t const along = east * nk_sin_f64_serial_(azimuth) * cos_elevation +
                           north * nk_cos_f64_serial_(azimuth) * cos_elevation + up * nk_sin_f64_serial_(elevation);
    return along / slant < nk_cos_f64_serial_(half_field);
}

NUMKONG_API nk_status_t nk_clearances_packed_f32_serial(nk_f32_t const *heights, void const *packed,
                                                        nk_f32_t const *viewpoints, nk_f32_t const *views,
                                                        nk_size_t rows, nk_f32_t const *targets, nk_size_t columns,
                                                        nk_f64_t refraction, nk_f32_t *results,
                                                        nk_size_t results_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_terrain_packed_header_t const *header = (nk_terrain_packed_header_t const *)packed;
    nk_f64_t const last_row = (nk_f64_t)(header->rows - 1), last_column = (nk_f64_t)(header->columns - 1);
    nk_f64_t const drop_scale = (1.0 - refraction) / (2.0 * NUMKONG_EARTH_MEAN_RADIUS);
    nk_size_t const levels = (nk_size_t)header->levels;
    nk_fui32_t off_grid;
    off_grid.u = 0x7FC00000u;
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f32_t *output = (nk_f32_t *)((char *)results + row * results_stride);
        nk_f64_t const column_start = viewpoints[row * 3], row_start = viewpoints[row * 3 + 1];
        nk_f64_t const height_start = viewpoints[row * 3 + 2];
        int const viewpoint_on_grid = column_start >= 0 && column_start <= last_column && row_start >= 0 &&
                                      row_start <= last_row;
        for (nk_size_t column = 0; column != columns; ++column) {
            nk_f64_t const column_end = targets[column * 3], row_end = targets[column * 3 + 1];
            if (!viewpoint_on_grid ||
                !(column_end >= 0 && column_end <= last_column && row_end >= 0 && row_end <= last_row)) {
                output[column] = off_grid.f;
                continue;
            }
            nk_terrain_ray_t_ ray;
            ray.column_start = column_start, ray.column_delta = column_end - column_start;
            ray.row_start = row_start, ray.row_delta = row_end - row_start;
            nk_f64_t const east = ray.column_delta * header->column_metres;
            nk_f64_t const south = ray.row_delta * header->row_metres;
            ray.distance = nk_sqrt_f64_serial_(east * east + south * south);
            ray.drop_scale = drop_scale;
            ray.height_start = height_start;
            ray.height_delta = (nk_f64_t)targets[column * 3 + 2] - drop_scale * ray.distance * ray.distance -
                               height_start;
            if (views &&
                nk_terrain_outside_view_f32_serial_(east, -south, ray.height_delta, views[row * 4], views[row * 4 + 1],
                                                    views[row * 4 + 2], views[row * 4 + 3])) {
                output[column] = -NUMKONG_F32_INF;
                continue;
            }
            nk_f64_t clearance = NUMKONG_F64_INF;
            nk_terrain_visit_serial_(heights, packed, &ray, levels, 0, 1, &clearance);
            output[column] = (nk_f32_t)clearance;
        }
    }
    return nk_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NUMKONG_TARGET_SERIAL

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_GEOSPATIAL_SERIAL_H
