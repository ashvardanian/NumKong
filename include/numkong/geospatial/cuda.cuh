/**
 *  @file include/numkong/geospatial/cuda.cuh
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief Many-to-many geodesics and terrain clearances on the SIMT cores of every CUDA device, the
 *      @c cuda exports.
 *
 *  @sa include/numkong/geospatial.h
 *  @sa include/numkong/geospatial/serial.h
 *
 *  Every pack keeps the serial layout and the serial arithmetic: the sines, cosines and arctangents
 *  below are the serial polynomials rounded one operation at a time, so a pack built here is
 *  byte-identical to the serial one, wherever it is read.
 *
 *  Each warp of a geodesic kernel takes 32 adjacent columns of a few rows and strides over the
 *  matrix, reading both shapes from the packs' headers on the device, so a launch never waits on
 *  the host. Haversine sums both norms in Dot2 and takes hardware square roots; Vincenty iterates
 *  per pair, each warp until its slowest lane settles.
 *
 *  A clearance walks the pyramid on one thread, depth-first in the order of the serial recursion
 *  and with the serial arithmetic, so it equals the serial clearance exactly. Each warp takes 32
 *  adjacent targets of one viewpoint, and as sight lines differ in cost, the hardware schedules
 *  every block rather than a resident grid striding over them.
 */
#ifndef NUMKONG_GEOSPATIAL_CUDA_CUH
#define NUMKONG_GEOSPATIAL_CUDA_CUH

#if NUMKONG_ARCH_CUDA_

#include "numkong/cuda.cuh"
#include "numkong/dots/cuda.cuh"       // `nk_cross_step_dot2_f64_cuda_`
#include "numkong/geospatial/serial.h" // `nk_geospatial_packed_header_t`, `nk_terrain_packed_header_t`

#if defined(__cplusplus)
extern "C" {
#endif

enum {

    /** Threads of every geospatial block. */
    nk_geospatial_threads_cuda_k = 256,

    /** Rows each lane of a geodesic warp takes for its one column. */
    nk_geodesics_rows_cuda_k = 4,

    /** Pyramid levels a clearance walk keeps frames for: grids up to 2³² + 1 rows and columns. */
    nk_terrain_levels_cuda_k = 32,

    /** Clearance blocks each multiprocessor keeps resident: the walk waits on memory, so it trades
     *  registers for occupancy. */
    nk_clearances_blocks_cuda_k = 3,
};

#pragma region Serial Arithmetic

/** @c nk_sin_f64_serial_ , rounded one operation at a time. */
NUMKONG_DEVICE nk_f64_t nk_sin_f64_cuda_(nk_f64_t const angle_radians) {
    nk_f64_t const pi_high = 3.141592653589793116, pi_low = 1.2246467991473532072e-16;
    nk_f64_t const pi_reciprocal = 0.318309886183790671537767526745028724;
    nk_f64_t const coeff_0 = +0.00833333333333332974823815, coeff_1 = -0.000198412698412696162806809;
    nk_f64_t const coeff_2 = +2.75573192239198747630416e-06, coeff_3 = -2.50521083763502045810755e-08;
    nk_f64_t const coeff_4 = +1.60590430605664501629054e-10, coeff_5 = -7.64712219118158833288484e-13;
    nk_f64_t const coeff_6 = +2.81009972710863200091251e-15, coeff_7 = -7.97255955009037868891952e-18;
    nk_f64_t const coeff_8 = -0.166666666666666657414808;

    nk_f64_t const quotient = nk_f64_mul_rn_cuda_(angle_radians, pi_reciprocal);
    int const multiple_of_pi = (int)(quotient < 0 ? quotient - 0.5 : quotient + 0.5);
    nk_f64_t angle = nk_f64_sub_rn_cuda_(angle_radians, nk_f64_mul_rn_cuda_((nk_f64_t)multiple_of_pi, pi_high));
    angle = nk_f64_sub_rn_cuda_(angle, nk_f64_mul_rn_cuda_((nk_f64_t)multiple_of_pi, pi_low));
    if ((multiple_of_pi & 1) != 0) angle = -angle;
    nk_f64_t const angle_squared = nk_f64_mul_rn_cuda_(angle, angle);
    nk_f64_t const angle_cubed = nk_f64_mul_rn_cuda_(angle, angle_squared);
    nk_f64_t const angle_quartic = nk_f64_mul_rn_cuda_(angle_squared, angle_squared);
    nk_f64_t const angle_octic = nk_f64_mul_rn_cuda_(angle_quartic, angle_quartic);

    nk_f64_t const poly_67 = __fma_rn(angle_squared, coeff_7, coeff_6);
    nk_f64_t const poly_45 = __fma_rn(angle_squared, coeff_5, coeff_4);
    nk_f64_t const poly_4567 = __fma_rn(angle_quartic, poly_67, poly_45);
    nk_f64_t const poly_23 = __fma_rn(angle_squared, coeff_3, coeff_2);
    nk_f64_t const poly_01 = __fma_rn(angle_squared, coeff_1, coeff_0);
    nk_f64_t const poly_0123 = __fma_rn(angle_quartic, poly_23, poly_01);
    nk_f64_t result = __fma_rn(angle_octic, poly_4567, poly_0123);
    result = __fma_rn(result, angle_squared, coeff_8);
    result = __fma_rn(result, angle_cubed, angle);

    nk_fui64_t converter;
    converter.f = angle_radians;
    return converter.u == 0x8000000000000000ull ? angle : result;
}

/** @c nk_cos_f64_serial_ , rounded one operation at a time. */
NUMKONG_DEVICE nk_f64_t nk_cos_f64_cuda_(nk_f64_t const angle_radians) {
    nk_f64_t const pi_high_half = 3.141592653589793116 * 0.5, pi_low_half = 1.2246467991473532072e-16 * 0.5;
    nk_f64_t const pi_reciprocal = 0.318309886183790671537767526745028724;
    nk_f64_t const coeff_0 = +0.00833333333333332974823815, coeff_1 = -0.000198412698412696162806809;
    nk_f64_t const coeff_2 = +2.75573192239198747630416e-06, coeff_3 = -2.50521083763502045810755e-08;
    nk_f64_t const coeff_4 = +1.60590430605664501629054e-10, coeff_5 = -7.64712219118158833288484e-13;
    nk_f64_t const coeff_6 = +2.81009972710863200091251e-15, coeff_7 = -7.97255955009037868891952e-18;
    nk_f64_t const coeff_8 = -0.166666666666666657414808;

    nk_f64_t const quotient = nk_f64_sub_rn_cuda_(nk_f64_mul_rn_cuda_(angle_radians, pi_reciprocal), 0.5);
    int const multiple_of_pi = 2 * (int)(quotient < 0 ? quotient - 0.5 : quotient + 0.5) + 1;
    nk_f64_t angle = nk_f64_sub_rn_cuda_(angle_radians, nk_f64_mul_rn_cuda_((nk_f64_t)multiple_of_pi, pi_high_half));
    angle = nk_f64_sub_rn_cuda_(angle, nk_f64_mul_rn_cuda_((nk_f64_t)multiple_of_pi, pi_low_half));
    if ((multiple_of_pi & 2) == 0) angle = -angle;
    nk_f64_t const angle_squared = nk_f64_mul_rn_cuda_(angle, angle);
    nk_f64_t const angle_cubed = nk_f64_mul_rn_cuda_(angle, angle_squared);
    nk_f64_t const angle_quartic = nk_f64_mul_rn_cuda_(angle_squared, angle_squared);
    nk_f64_t const angle_octic = nk_f64_mul_rn_cuda_(angle_quartic, angle_quartic);

    nk_f64_t const poly_67 = __fma_rn(angle_squared, coeff_7, coeff_6);
    nk_f64_t const poly_45 = __fma_rn(angle_squared, coeff_5, coeff_4);
    nk_f64_t const poly_4567 = __fma_rn(angle_quartic, poly_67, poly_45);
    nk_f64_t const poly_23 = __fma_rn(angle_squared, coeff_3, coeff_2);
    nk_f64_t const poly_01 = __fma_rn(angle_squared, coeff_1, coeff_0);
    nk_f64_t const poly_0123 = __fma_rn(angle_quartic, poly_23, poly_01);
    nk_f64_t result = __fma_rn(angle_octic, poly_4567, poly_0123);
    result = __fma_rn(result, angle_squared, coeff_8);
    return __fma_rn(result, angle_cubed, angle);
}

/** @c nk_atan2_f64_serial_ , rounded one operation at a time. */
NUMKONG_DEVICE nk_f64_t nk_atan2_f64_cuda_(nk_f64_t const y_input, nk_f64_t const x_input) {
    nk_f64_t const coeff_19 = -1.88796008463073496563746e-05, coeff_18 = +0.000209850076645816976906797;
    nk_f64_t const coeff_17 = -0.00110611831486672482563471, coeff_16 = +0.00370026744188713119232403;
    nk_f64_t const coeff_15 = -0.00889896195887655491740809, coeff_14 = +0.016599329773529201970117;
    nk_f64_t const coeff_13 = -0.0254517624932312641616861, coeff_12 = +0.0337852580001353069993897;
    nk_f64_t const coeff_11 = -0.0407629191276836500001934, coeff_10 = +0.0466667150077840625632675;
    nk_f64_t const coeff_9 = -0.0523674852303482457616113, coeff_8 = +0.0587666392926673580854313;
    nk_f64_t const coeff_7 = -0.0666573579361080525984562, coeff_6 = +0.0769219538311769618355029;
    nk_f64_t const coeff_5 = -0.090908995008245008229153, coeff_4 = +0.111111105648261418443745;
    nk_f64_t const coeff_3 = -0.14285714266771329383765, coeff_2 = +0.199999999996591265594148;
    nk_f64_t const coeff_1 = -0.333333333333311110369124;

    nk_fui64_t x_bits, y_bits, x_abs, y_abs;
    x_bits.f = x_input, y_bits.f = y_input;
    y_abs.u = y_bits.u & 0x7FFFFFFFFFFFFFFFull;
    int quadrant = 0;
    if (x_input < 0) x_abs.f = -x_input, quadrant = -2;
    else x_abs.f = x_input;
    if (y_abs.f > x_abs.f) {
        nk_f64_t const previous_x_abs = x_abs.f;
        x_abs.f = y_abs.f;
        y_abs.f = -previous_x_abs;
        quadrant += 1;
    }

    nk_f64_t const ratio = y_abs.f / x_abs.f;
    nk_f64_t const ratio_squared = nk_f64_mul_rn_cuda_(ratio, ratio);
    nk_f64_t const ratio_cubed = nk_f64_mul_rn_cuda_(ratio, ratio_squared);
    nk_f64_t polynomial = __fma_rn(coeff_19, ratio_squared, coeff_18);
    polynomial = __fma_rn(polynomial, ratio_squared, coeff_17);
    polynomial = __fma_rn(polynomial, ratio_squared, coeff_16);
    polynomial = __fma_rn(polynomial, ratio_squared, coeff_15);
    polynomial = __fma_rn(polynomial, ratio_squared, coeff_14);
    polynomial = __fma_rn(polynomial, ratio_squared, coeff_13);
    polynomial = __fma_rn(polynomial, ratio_squared, coeff_12);
    polynomial = __fma_rn(polynomial, ratio_squared, coeff_11);
    polynomial = __fma_rn(polynomial, ratio_squared, coeff_10);
    polynomial = __fma_rn(polynomial, ratio_squared, coeff_9);
    polynomial = __fma_rn(polynomial, ratio_squared, coeff_8);
    polynomial = __fma_rn(polynomial, ratio_squared, coeff_7);
    polynomial = __fma_rn(polynomial, ratio_squared, coeff_6);
    polynomial = __fma_rn(polynomial, ratio_squared, coeff_5);
    polynomial = __fma_rn(polynomial, ratio_squared, coeff_4);
    polynomial = __fma_rn(polynomial, ratio_squared, coeff_3);
    polynomial = __fma_rn(polynomial, ratio_squared, coeff_2);
    polynomial = __fma_rn(polynomial, ratio_squared, coeff_1);

    nk_f64_t const pi = 3.14159265358979323846, pi_half = 1.5707963267948966, pi_quarter = 0.7853981633974483;
    nk_f64_t result = __fma_rn(polynomial, ratio_cubed, ratio);
    result = __fma_rn((nk_f64_t)quadrant, pi_half, result);

    int const x_is_inf = (x_bits.u == 0x7FF0000000000000ull) | (x_bits.u == 0xFFF0000000000000ull);
    int const y_is_inf = (y_bits.u == 0x7FF0000000000000ull) | (y_bits.u == 0xFFF0000000000000ull);
    nk_fui64_t result_bits;
    result_bits.f = result;
    result_bits.u ^= x_bits.u & 0x8000000000000000ull;
    if (x_is_inf | (x_bits.f == 0)) result_bits.f = pi_half - (x_is_inf ? (x_bits.f < 0 ? pi_half : 0) : 0);
    if (y_is_inf) result_bits.f = pi_half - (x_is_inf ? (x_bits.f < 0 ? pi_half : pi_quarter) : 0);
    if (y_bits.f == 0) result_bits.f = (x_bits.f < 0 ? pi : 0);
    if (x_is_inf | y_is_inf) result_bits.u = 0x7FF8000000000000ull;
    else result_bits.u ^= y_bits.u & 0x8000000000000000ull;
    return result_bits.f;
}

/** @c nk_sqrt_f64_serial_ , the product of a number and its Newton reciprocal root, rounded one
 *  operation at a time, so clearances match the serial ones. */
NUMKONG_DEVICE nk_f64_t nk_sqrt_f64_cuda_(nk_f64_t number) {
    nk_f64_t const positive = number > 0 ? number : 0;
    nk_f64_t const finite = number > 0 && number < NUMKONG_F64_INF ? number : 1.0;
    nk_fui64_t conv;
    conv.f = finite;
    int const subnormal = conv.u - 1 < 0x000FFFFFFFFFFFFFull;
    nk_f64_t const scaled = finite * (subnormal ? 0x1p54 : 1.0);
    conv.f = scaled;
    conv.u = 0x5FE6EB50C7B537A9ull - (conv.u >> 1);
    nk_f64_t root = conv.f;
    for (int step = 0; step != 4; ++step) {
        nk_f64_t const half_scaled_root_squared = nk_f64_mul_rn_cuda_(
            nk_f64_mul_rn_cuda_(nk_f64_mul_rn_cuda_(0.5, scaled), root), root);
        root = nk_f64_mul_rn_cuda_(root, nk_f64_sub_rn_cuda_(1.5, half_scaled_root_squared));
    }
    return positive * (root * (subnormal ? 0x1p27 : 1.0));
}

#pragma endregion Serial Arithmetic

#pragma region Geodesics

/** A geodesic pack's latitudes and longitudes, its point count and the pack to fill. */
typedef struct nk_geodesics_pack_arguments_cuda_t {
    nk_f64_t const *latitudes;
    nk_f64_t const *longitudes;
    nk_size_t count;
    void *packed;
} nk_geodesics_pack_arguments_cuda_t;

/** A geodesic kernel's packs and the @b [rows,columns] matrix it fills. */
typedef struct nk_geodesics_arguments_cuda_t {
    void const *a_packed;
    void const *b_packed;
    nk_f64_t *results;
    nk_size_t results_stride;
} nk_geodesics_arguments_cuda_t;

/** Great-circle metres between unit vectors @p a and @p b, both norms summed in Dot2. */
NUMKONG_DEVICE nk_f64_t nk_haversine_f64_cuda_(nk_f64_t a_x, nk_f64_t a_y, nk_f64_t a_z, nk_f64_t b_x, nk_f64_t b_y,
                                               nk_f64_t b_z) {
    nk_f64_t const difference_x = a_x - b_x, difference_y = a_y - b_y, difference_z = a_z - b_z;
    nk_f64_t const sum_x = a_x + b_x, sum_y = a_y + b_y, sum_z = a_z + b_z;
    // From zero, the first Dot2 step leaves the product and its TwoProd error
    nk_f64_t chord = nk_f64_mul_rn_cuda_(difference_x, difference_x);
    nk_f64_t chord_compensation = __fma_rn(difference_x, difference_x, -chord);
    nk_f64_t span = nk_f64_mul_rn_cuda_(sum_x, sum_x), span_compensation = __fma_rn(sum_x, sum_x, -span);
    nk_cross_step_dot2_f64_cuda_(difference_y, difference_y, &chord, &chord_compensation);
    nk_cross_step_dot2_f64_cuda_(difference_z, difference_z, &chord, &chord_compensation);
    nk_cross_step_dot2_f64_cuda_(sum_y, sum_y, &span, &span_compensation);
    nk_cross_step_dot2_f64_cuda_(sum_z, sum_z, &span, &span_compensation);
    nk_f64_t const angle = 2.0 * nk_atan2_f64_cuda_(__dsqrt_rn(chord + chord_compensation),
                                                    __dsqrt_rn(span + span_compensation));
    return NUMKONG_EARTH_MEAN_RADIUS * angle;
}

/** Geodesic metres on the WGS-84 ellipsoid between two points, @p longitude_difference apart, of
 *  the given sines and cosines of reduced latitude, iterated as the serial kernel iterates. */
NUMKONG_DEVICE nk_f64_t nk_vincenty_f64_cuda_(nk_f64_t longitude_difference, nk_f64_t sin_reduced_first,
                                              nk_f64_t cos_reduced_first, nk_f64_t sin_reduced_second,
                                              nk_f64_t cos_reduced_second) {
    nk_f64_t const equatorial_radius = NUMKONG_EARTH_ELLIPSOID_EQUATORIAL_RADIUS;
    nk_f64_t const flattening = 1.0 / NUMKONG_EARTH_ELLIPSOID_INVERSE_FLATTENING;
    nk_f64_t const polar_radius = equatorial_radius * (1.0 - flattening);
    nk_f64_t lambda = longitude_difference, lambda_previous;
    nk_f64_t sin_angular_distance = 0, cos_angular_distance = 1, angular_distance = 0;
    nk_f64_t cos_squared_azimuth = 1, cos_double_angular_midpoint = 0;
    nk_u32_t iteration = 0;
    do {
        nk_f64_t const sin_lambda = nk_sin_f64_cuda_(lambda), cos_lambda = nk_cos_f64_cuda_(lambda);
        nk_f64_t const cross_term = cos_reduced_second * sin_lambda;
        nk_f64_t const mixed_term = cos_reduced_first * sin_reduced_second -
                                    sin_reduced_first * cos_reduced_second * cos_lambda;
        sin_angular_distance = __dsqrt_rn(cross_term * cross_term + mixed_term * mixed_term);
        if (sin_angular_distance == 0.0) return 0.0;
        cos_angular_distance = sin_reduced_first * sin_reduced_second +
                               cos_reduced_first * cos_reduced_second * cos_lambda;
        angular_distance = nk_atan2_f64_cuda_(sin_angular_distance, cos_angular_distance);
        nk_f64_t const sin_azimuth = cos_reduced_first * cos_reduced_second * sin_lambda / sin_angular_distance;
        cos_squared_azimuth = 1.0 - sin_azimuth * sin_azimuth;
        // An equatorial geodesic has no midpoint term
        cos_double_angular_midpoint = cos_squared_azimuth != 0.0
                                          ? cos_angular_distance -
                                                2.0 * sin_reduced_first * sin_reduced_second / cos_squared_azimuth
                                          : 0.0;
        nk_f64_t const correction_factor = flattening / 16.0 * cos_squared_azimuth *
                                           (4.0 + flattening * (4.0 - 3.0 * cos_squared_azimuth));
        lambda_previous = lambda;
        lambda = longitude_difference + (1.0 - correction_factor) * flattening * sin_azimuth *
                                            (angular_distance + correction_factor * sin_angular_distance *
                                                                    (cos_double_angular_midpoint +
                                                                     correction_factor * cos_angular_distance *
                                                                         (-1.0 + 2.0 * cos_double_angular_midpoint *
                                                                                     cos_double_angular_midpoint)));
        iteration++;
    } while (nk_f64_abs_(lambda - lambda_previous) > NUMKONG_VINCENTY_CONVERGENCE_THRESHOLD_F64 &&
             iteration < NUMKONG_VINCENTY_MAX_ITERATIONS);

    nk_f64_t const u_squared = cos_squared_azimuth *
                               (equatorial_radius * equatorial_radius - polar_radius * polar_radius) /
                               (polar_radius * polar_radius);
    nk_f64_t const series_a = 1.0 + u_squared / 16384.0 *
                                        (4096.0 + u_squared * (-768.0 + u_squared * (320.0 - 175.0 * u_squared)));
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
    return polar_radius * series_a * (angular_distance - angular_correction);
}

/** Validates a geodesic pack and launches @p kernel with a thread per point, at least one block
 *  wide, so the header lands even for no points. */
NUMKONG_INLINE nk_status_t nk_geodesics_pack_launch_cuda_(void const *kernel, nk_f64_t const *latitudes,
                                                          nk_f64_t const *longitudes, nk_size_t count, void *packed,
                                                          nk_stream_t stream) {
    if ((((nk_size_t)latitudes) | ((nk_size_t)longitudes) | ((nk_size_t)packed)) & 7) return nk_misaligned_k;
    nk_geodesics_pack_arguments_cuda_t arguments;
    arguments.latitudes = latitudes, arguments.longitudes = longitudes;
    arguments.count = count, arguments.packed = packed;
    nk_size_t const blocks = nk_size_divide_round_up_(count, nk_geospatial_threads_cuda_k);
    return nk_launch_resident_cuda_(kernel, nk_geospatial_threads_cuda_k, 0, 0, blocks ? blocks : 1, &arguments,
                                    stream);
}

/** Validates a geodesic matrix and launches the resident grid of @p kernel, which reads both shapes
 *  from the packs. */
NUMKONG_INLINE nk_status_t nk_geodesics_launch_cuda_(void const *kernel, void const *a_packed, void const *b_packed,
                                                     nk_f64_t *results, nk_size_t results_stride, nk_stream_t stream) {
    if ((((nk_size_t)a_packed) | ((nk_size_t)b_packed) | ((nk_size_t)results) | results_stride) & 7)
        return nk_misaligned_k;
    nk_geodesics_arguments_cuda_t arguments;
    arguments.a_packed = a_packed, arguments.b_packed = b_packed;
    arguments.results = results, arguments.results_stride = results_stride;
    return nk_launch_resident_cuda_(kernel, nk_geospatial_threads_cuda_k, 0, 0, NUMKONG_SIZE_MAX, &arguments, stream);
}

#pragma endregion Geodesics

#pragma region Terrain

/** One launch of a terrain pack: the grid, the level it fills and the level below it. */
typedef struct nk_terrain_pack_arguments_cuda_t {
    nk_f32_t const *heights;
    nk_size_t heights_stride;
    nk_size_t rows, columns;
    nk_f32_t const *finer;
    nk_size_t finer_rows, finer_columns;
    nk_f32_t *maxima;
    nk_size_t block_rows, block_columns;
    nk_terrain_packed_header_t header;
    void *packed;
} nk_terrain_pack_arguments_cuda_t;

/** Bilinear heights under a list of points. */
typedef struct nk_terrain_heights_arguments_cuda_t {
    nk_f32_t const *heights;
    void const *packed;
    nk_f32_t const *points;
    nk_size_t count;
    nk_f32_t *results;
} nk_terrain_heights_arguments_cuda_t;

/** Clearances from every viewpoint to every target over one terrain. */
typedef struct nk_clearances_arguments_cuda_t {
    nk_f32_t const *heights;
    void const *packed;
    nk_f32_t const *viewpoints;
    nk_f32_t const *views;
    nk_size_t rows;
    nk_f32_t const *targets;
    nk_size_t columns;
    nk_f64_t drop_scale;
    nk_f32_t *results;
    nk_size_t results_stride;
} nk_clearances_arguments_cuda_t;

/** The shape of a terrain grid and its pyramid: where each level's maxima start, in values from the
 *  first level's, and how many blocks a level's rows and columns hold. */
typedef struct nk_terrain_grid_cuda_t {
    nk_f32_t const *heights;
    nk_f32_t const *maxima;
    nk_size_t rows, columns, heights_stride, levels;
    nk_f64_t last_row, last_column;
    nk_size_t level_offsets[nk_terrain_levels_cuda_k + 1];
    nk_u32_t level_rows[nk_terrain_levels_cuda_k + 1];
    nk_u32_t level_columns[nk_terrain_levels_cuda_k + 1];
} nk_terrain_grid_cuda_t;

/** One sight line in cell coordinates, as the serial kernel lays it out: p(u) = start + u · delta
 *  for u in (0, 1), its height at u, and the scale of the curvature drop along it. */
typedef struct nk_terrain_ray_cuda_t {
    nk_f64_t column_start, column_delta;
    nk_f64_t row_start, row_delta;
    nk_f64_t height_start, height_delta;
    nk_f64_t distance, drop_scale;
} nk_terrain_ray_cuda_t;

/** The walk's state at a coarser level while it visits a finer one: where that level's segment
 *  ends and the block it stands in. */
typedef struct nk_terrain_frame_cuda_t {
    nk_f64_t u_end;
    nk_u32_t block_row, block_column;
} nk_terrain_frame_cuda_t;

/** Blocks along one axis of terrain level @p level over @p cells cells. */
NUMKONG_DEVICE nk_size_t nk_terrain_level_blocks_f32_cuda_(nk_size_t cells, nk_size_t level) {
    nk_size_t const span = cells > 1 ? cells - 1 : 1;
    return nk_size_divide_round_up_(span, (nk_size_t)1 << level);
}

/** Fills @p grid from the header of @p packed, on one thread. */
NUMKONG_DEVICE void nk_terrain_grid_f32_cuda_(nk_f32_t const *heights, void const *packed,
                                              nk_terrain_grid_cuda_t *grid) {
    nk_terrain_packed_header_t const *header = (nk_terrain_packed_header_t const *)packed;
    grid->heights = heights;
    grid->maxima = (nk_f32_t const *)((char const *)packed + nk_default_alignment_k);
    grid->rows = (nk_size_t)header->rows, grid->columns = (nk_size_t)header->columns;
    grid->heights_stride = (nk_size_t)header->heights_stride, grid->levels = (nk_size_t)header->levels;
    grid->last_row = (nk_f64_t)(grid->rows - 1), grid->last_column = (nk_f64_t)(grid->columns - 1);
    nk_size_t offset = 0;
    for (nk_size_t level = 1; level <= grid->levels && level <= nk_terrain_levels_cuda_k; ++level) {
        nk_size_t const level_rows = nk_terrain_level_blocks_f32_cuda_(grid->rows, level);
        nk_size_t const level_columns = nk_terrain_level_blocks_f32_cuda_(grid->columns, level);
        grid->level_offsets[level] = offset;
        grid->level_rows[level] = (nk_u32_t)level_rows, grid->level_columns[level] = (nk_u32_t)level_columns;
        offset += level_rows * level_columns;
    }
}

/** The height in row @p row and column @p column of @p grid. */
NUMKONG_DEVICE nk_f64_t nk_terrain_height_f32_cuda_(nk_terrain_grid_cuda_t const *grid, nk_size_t row,
                                                    nk_size_t column) {
    return *(nk_f32_t const *)((char const *)grid->heights + row * grid->heights_stride + column * sizeof(nk_f32_t));
}

/** @c nk_terrain_clamp_f64_serial_ . */
NUMKONG_DEVICE nk_f64_t nk_terrain_clamp_f64_cuda_(nk_f64_t coordinate, nk_f64_t last) {
    return coordinate < 0 ? 0 : (coordinate > last ? last : coordinate);
}

/** @c nk_terrain_fold_sample_f32_serial_ , rounded one operation at a time. */
NUMKONG_DEVICE void nk_terrain_fold_sample_f32_cuda_(nk_terrain_ray_cuda_t const *ray, nk_f64_t u, nk_f64_t first,
                                                     nk_f64_t second, nk_f64_t fraction, nk_f64_t *clearance) {
    nk_f64_t const ground = fraction == 0 ? first
                                          : nk_f64_add_rn_cuda_(first, nk_f64_mul_rn_cuda_(fraction, second - first));
    if (ground != ground) return;
    nk_f64_t const along = u * ray->distance;
    nk_f64_t const line = nk_f64_add_rn_cuda_(ray->height_start, nk_f64_mul_rn_cuda_(u, ray->height_delta));
    nk_f64_t const drop = nk_f64_mul_rn_cuda_(nk_f64_mul_rn_cuda_(ray->drop_scale, along), along);
    nk_f64_t const gap = line - nk_f64_sub_rn_cuda_(ground, drop);
    if (gap < *clearance) *clearance = gap;
}

/** @c nk_terrain_fold_samples_f32_serial_ : every sample between @p u_begin and @p u_end, the
 *  column lines, then the row lines. */
NUMKONG_DEVICE void nk_terrain_fold_samples_f32_cuda_(nk_terrain_grid_cuda_t const *grid,
                                                      nk_terrain_ray_cuda_t const *ray, nk_f64_t u_begin,
                                                      nk_f64_t u_end, nk_f64_t *clearance) {
    if (ray->column_delta != 0) {
        nk_f64_t low = nk_terrain_clamp_f64_cuda_(
            nk_f64_add_rn_cuda_(ray->column_start, nk_f64_mul_rn_cuda_(u_begin, ray->column_delta)), grid->last_column);
        nk_f64_t high = nk_terrain_clamp_f64_cuda_(
            nk_f64_add_rn_cuda_(ray->column_start, nk_f64_mul_rn_cuda_(u_end, ray->column_delta)), grid->last_column);
        if (low > high) {
            nk_f64_t const swapped = low;
            low = high, high = swapped;
        }
        nk_size_t const first = (nk_size_t)low + ((nk_f64_t)(nk_size_t)low < low);
        for (nk_size_t column = first; (nk_f64_t)column <= high; ++column) {
            nk_f64_t const u = ((nk_f64_t)column - ray->column_start) / ray->column_delta;
            if (!(u > 0 && u < 1 && u >= u_begin && u <= u_end)) continue;
            nk_f64_t const row = nk_terrain_clamp_f64_cuda_(
                nk_f64_add_rn_cuda_(ray->row_start, nk_f64_mul_rn_cuda_(u, ray->row_delta)), grid->last_row);
            nk_size_t const row_first = (nk_size_t)row;
            nk_size_t const row_second = row_first + 1 < grid->rows ? row_first + 1 : row_first;
            nk_terrain_fold_sample_f32_cuda_(ray, u, nk_terrain_height_f32_cuda_(grid, row_first, column),
                                             nk_terrain_height_f32_cuda_(grid, row_second, column),
                                             row - (nk_f64_t)row_first, clearance);
        }
    }
    if (ray->row_delta != 0) {
        nk_f64_t low = nk_terrain_clamp_f64_cuda_(
            nk_f64_add_rn_cuda_(ray->row_start, nk_f64_mul_rn_cuda_(u_begin, ray->row_delta)), grid->last_row);
        nk_f64_t high = nk_terrain_clamp_f64_cuda_(
            nk_f64_add_rn_cuda_(ray->row_start, nk_f64_mul_rn_cuda_(u_end, ray->row_delta)), grid->last_row);
        if (low > high) {
            nk_f64_t const swapped = low;
            low = high, high = swapped;
        }
        nk_size_t const first = (nk_size_t)low + ((nk_f64_t)(nk_size_t)low < low);
        for (nk_size_t row = first; (nk_f64_t)row <= high; ++row) {
            nk_f64_t const u = ((nk_f64_t)row - ray->row_start) / ray->row_delta;
            if (!(u > 0 && u < 1 && u >= u_begin && u <= u_end)) continue;
            nk_f64_t const column = nk_terrain_clamp_f64_cuda_(
                nk_f64_add_rn_cuda_(ray->column_start, nk_f64_mul_rn_cuda_(u, ray->column_delta)), grid->last_column);
            nk_size_t const column_first = (nk_size_t)column;
            nk_size_t const column_second = column_first + 1 < grid->columns ? column_first + 1 : column_first;
            nk_terrain_fold_sample_f32_cuda_(ray, u, nk_terrain_height_f32_cuda_(grid, row, column_first),
                                             nk_terrain_height_f32_cuda_(grid, row, column_second),
                                             column - (nk_f64_t)column_first, clearance);
        }
    }
}

/** The block of @p level holding the ray at @p u, as the serial walk enters a level. */
NUMKONG_DEVICE void nk_terrain_enter_f32_cuda_(nk_terrain_grid_cuda_t const *grid, nk_terrain_ray_cuda_t const *ray,
                                               nk_size_t level, nk_f64_t u, nk_u32_t *block_row,
                                               nk_u32_t *block_column) {
    nk_f64_t const span = (nk_f64_t)((nk_size_t)1 << level);
    nk_f64_t const row = nk_terrain_clamp_f64_cuda_(
        nk_f64_add_rn_cuda_(ray->row_start, nk_f64_mul_rn_cuda_(u, ray->row_delta)), grid->last_row);
    nk_f64_t const column = nk_terrain_clamp_f64_cuda_(
        nk_f64_add_rn_cuda_(ray->column_start, nk_f64_mul_rn_cuda_(u, ray->column_delta)), grid->last_column);
    nk_u32_t const rows_last = grid->level_rows[level] - 1, columns_last = grid->level_columns[level] - 1;
    nk_u32_t const row_block = (nk_u32_t)(row / span), column_block = (nk_u32_t)(column / span);
    *block_row = row_block < rows_last ? row_block : rows_last;
    *block_column = column_block < columns_last ? column_block : columns_last;
}

/**
 *  @brief The least gap between @p ray and the terrain, by the serial recursion of
 *      @c nk_terrain_visit_serial_ unrolled into a loop over a stack of frames.
 *
 *  Each level walks the blocks the ray crosses within its parent's segment, skipping a block whose
 *  maximum cannot lower the running minimum and descending into the rest, with level 1 folding the
 *  samples of its segments directly.
 */
NUMKONG_DEVICE nk_f64_t nk_terrain_clearance_f32_cuda_(nk_terrain_grid_cuda_t const *grid,
                                                       nk_terrain_ray_cuda_t const *ray) {
    nk_f64_t clearance = NUMKONG_F64_INF;
    if (grid->levels == 0) {
        nk_terrain_fold_samples_f32_cuda_(grid, ray, 0, 1, &clearance);
        return clearance;
    }
    nk_terrain_frame_cuda_t frames[nk_terrain_levels_cuda_k + 1];
    nk_u64_t row_crossings = 0, column_crossings = 0;
    nk_size_t level = grid->levels;
    nk_f64_t u = 0, u_end = 1;
    nk_u32_t block_row, block_column;
    nk_terrain_enter_f32_cuda_(grid, ray, level, u, &block_row, &block_column);
    for (;;) {
        nk_f64_t const span = (nk_f64_t)((nk_size_t)1 << level);
        nk_f64_t const infinity = NUMKONG_F64_INF;
        nk_f64_t u_row = infinity, u_column = infinity;
        if (ray->row_delta > 0)
            u_row = nk_f64_sub_rn_cuda_(nk_f64_mul_rn_cuda_((nk_f64_t)(block_row + 1), span), ray->row_start) /
                    ray->row_delta;
        else if (ray->row_delta < 0)
            u_row = nk_f64_sub_rn_cuda_(nk_f64_mul_rn_cuda_((nk_f64_t)block_row, span), ray->row_start) /
                    ray->row_delta;
        if (ray->column_delta > 0)
            u_column = nk_f64_sub_rn_cuda_(nk_f64_mul_rn_cuda_((nk_f64_t)(block_column + 1), span), ray->column_start) /
                       ray->column_delta;
        else if (ray->column_delta < 0)
            u_column = nk_f64_sub_rn_cuda_(nk_f64_mul_rn_cuda_((nk_f64_t)block_column, span), ray->column_start) /
                       ray->column_delta;
        nk_f64_t u_next = u_row < u_column ? u_row : u_column;
        if (u_next > u_end) u_next = u_end;
        if (u_next < u) u_next = u;

        // The sight line is linear in u and the drop grows with u: the segment's start bounds both.
        nk_f64_t const maximum =
            grid->maxima[grid->level_offsets[level] + (nk_size_t)block_row * grid->level_columns[level] + block_column];
        nk_f64_t const line_begin = nk_f64_add_rn_cuda_(ray->height_start, nk_f64_mul_rn_cuda_(u, ray->height_delta));
        nk_f64_t const line_end = nk_f64_add_rn_cuda_(ray->height_start,
                                                      nk_f64_mul_rn_cuda_(u_next, ray->height_delta));
        nk_f64_t const along = u * ray->distance;
        nk_f64_t const drop = nk_f64_mul_rn_cuda_(nk_f64_mul_rn_cuda_(ray->drop_scale, along), along);
        nk_f64_t const bound = (line_begin < line_end ? line_begin : line_end) - nk_f64_sub_rn_cuda_(maximum, drop);
        int crosses_row = u_row <= u_column, crosses_column = u_column <= u_row;
        if (bound < clearance) {
            if (level == 1) nk_terrain_fold_samples_f32_cuda_(grid, ray, u, u_next, &clearance);
            else {
                frames[level].u_end = u_end, frames[level].block_row = block_row;
                frames[level].block_column = block_column;
                nk_u64_t const bit = (nk_u64_t)1 << level;
                row_crossings = crosses_row ? row_crossings | bit : row_crossings & ~bit;
                column_crossings = crosses_column ? column_crossings | bit : column_crossings & ~bit;
                --level, u_end = u_next;
                nk_terrain_enter_f32_cuda_(grid, ray, level, u, &block_row, &block_column);
                continue;
            }
        }
        // Steps to the next block, returning to coarser levels while their segments are done
        for (;;) {
            int done = u_next >= u_end;
            if (!done && crosses_row) {
                if (ray->row_delta > 0 ? block_row + 1 >= grid->level_rows[level] : block_row == 0) done = 1;
                else block_row = ray->row_delta > 0 ? block_row + 1 : block_row - 1;
            }
            if (!done && crosses_column) {
                if (ray->column_delta > 0 ? block_column + 1 >= grid->level_columns[level] : block_column == 0)
                    done = 1;
                else block_column = ray->column_delta > 0 ? block_column + 1 : block_column - 1;
            }
            if (!done) break;
            if (level == grid->levels) return clearance;
            u_next = u_end, ++level;
            u_end = frames[level].u_end, block_row = frames[level].block_row;
            block_column = frames[level].block_column;
            crosses_row = (int)((row_crossings >> level) & 1), crosses_column = (int)((column_crossings >> level) & 1);
        }
        u = u_next;
    }
}

/** @c nk_terrain_outside_view_f32_serial_ , rounded one operation at a time. */
NUMKONG_DEVICE int nk_terrain_outside_view_f32_cuda_(nk_f64_t east, nk_f64_t north, nk_f64_t up, nk_f64_t azimuth,
                                                     nk_f64_t elevation, nk_f64_t field, nk_f64_t range) {
    nk_f64_t const ground_squared = nk_f64_add_rn_cuda_(nk_f64_mul_rn_cuda_(east, east),
                                                        nk_f64_mul_rn_cuda_(north, north));
    nk_f64_t const slant = nk_sqrt_f64_cuda_(nk_f64_add_rn_cuda_(ground_squared, nk_f64_mul_rn_cuda_(up, up)));
    if (range == range && slant > range) return 1;
    if (azimuth != azimuth || !(field < 2 * 3.14159265358979323846)) return 0;
    nk_f64_t const half_field = field / 2;
    nk_f64_t const sin_azimuth = nk_sin_f64_cuda_(azimuth), cos_azimuth = nk_cos_f64_cuda_(azimuth);
    if (elevation != elevation) {
        nk_f64_t const ground = nk_sqrt_f64_cuda_(ground_squared);
        if (ground == 0) return 0;
        nk_f64_t const cosine = nk_f64_add_rn_cuda_(nk_f64_mul_rn_cuda_(east, sin_azimuth),
                                                    nk_f64_mul_rn_cuda_(north, cos_azimuth)) /
                                ground;
        return cosine < nk_cos_f64_cuda_(half_field);
    }
    if (slant == 0) return 0;
    nk_f64_t const cos_elevation = nk_cos_f64_cuda_(elevation);
    nk_f64_t const along = nk_f64_add_rn_cuda_(
        nk_f64_add_rn_cuda_(nk_f64_mul_rn_cuda_(nk_f64_mul_rn_cuda_(east, sin_azimuth), cos_elevation),
                            nk_f64_mul_rn_cuda_(nk_f64_mul_rn_cuda_(north, cos_azimuth), cos_elevation)),
        nk_f64_mul_rn_cuda_(up, nk_sin_f64_cuda_(elevation)));
    return along / slant < nk_cos_f64_cuda_(half_field);
}

/** The clearance from viewpoint @p row to target @p column, as @c nk_clearances_packed_f32_serial
 *  computes it. */
NUMKONG_DEVICE nk_f32_t nk_clearance_f32_cuda_(nk_terrain_grid_cuda_t const *grid,
                                               nk_clearances_arguments_cuda_t const *arguments, nk_size_t row,
                                               nk_size_t column) {
    nk_fui32_t off_grid;
    off_grid.u = 0x7FC00000u;
    nk_terrain_packed_header_t const *header = (nk_terrain_packed_header_t const *)arguments->packed;
    nk_f32_t const *viewpoint = arguments->viewpoints + row * 3, *target = arguments->targets + column * 3;
    nk_f64_t const column_start = viewpoint[0], row_start = viewpoint[1], height_start = viewpoint[2];
    nk_f64_t const column_end = target[0], row_end = target[1];
    if (!(column_start >= 0 && column_start <= grid->last_column && row_start >= 0 && row_start <= grid->last_row) ||
        !(column_end >= 0 && column_end <= grid->last_column && row_end >= 0 && row_end <= grid->last_row) ||
        grid->levels > nk_terrain_levels_cuda_k)
        return off_grid.f;
    nk_terrain_ray_cuda_t ray;
    ray.column_start = column_start, ray.column_delta = column_end - column_start;
    ray.row_start = row_start, ray.row_delta = row_end - row_start;
    nk_f64_t const east = nk_f64_mul_rn_cuda_(ray.column_delta, header->column_metres);
    nk_f64_t const south = nk_f64_mul_rn_cuda_(ray.row_delta, header->row_metres);
    ray.distance = nk_sqrt_f64_cuda_(
        nk_f64_add_rn_cuda_(nk_f64_mul_rn_cuda_(east, east), nk_f64_mul_rn_cuda_(south, south)));
    ray.drop_scale = arguments->drop_scale;
    ray.height_start = height_start;
    ray.height_delta = nk_f64_sub_rn_cuda_((nk_f64_t)target[2],
                                           nk_f64_mul_rn_cuda_(nk_f64_mul_rn_cuda_(arguments->drop_scale, ray.distance),
                                                               ray.distance)) -
                       height_start;
    if (arguments->views) {
        nk_f32_t const *view = arguments->views + row * 4;
        if (nk_terrain_outside_view_f32_cuda_(east, -south, ray.height_delta, view[0], view[1], view[2], view[3]))
            return -NUMKONG_F32_INF;
    }
    return (nk_f32_t)nk_terrain_clearance_f32_cuda_(grid, &ray);
}

#pragma endregion Terrain

#if NUMKONG_TARGET_CUDA

static __global__ void __launch_bounds__(nk_geospatial_threads_cuda_k, 1)
    nk_haversines_pack_f64_cuda_kernel_(nk_geodesics_pack_arguments_cuda_t arguments) {
    nk_size_t const count = arguments.count;
    nk_size_t const axis_stride = nk_size_round_up_to_multiple_(count, nk_default_alignment_k / sizeof(nk_f64_t));
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        nk_geospatial_packed_header_t *header = (nk_geospatial_packed_header_t *)arguments.packed;
        header->points = count, header->axis_stride = axis_stride;
    }
    nk_f64_t *packed_x = (nk_f64_t *)((char *)arguments.packed + nk_default_alignment_k);
    nk_f64_t *packed_y = packed_x + axis_stride, *packed_z = packed_y + axis_stride;
    nk_size_t const stride = (nk_size_t)gridDim.x * blockDim.x;
    for (nk_size_t point = (nk_size_t)blockIdx.x * blockDim.x + threadIdx.x; point < count; point += stride) {
        nk_f64_t const latitude = arguments.latitudes[point], longitude = arguments.longitudes[point];
        nk_f64_t const cos_latitude = nk_cos_f64_cuda_(latitude);
        packed_x[point] = cos_latitude * nk_cos_f64_cuda_(longitude);
        packed_y[point] = cos_latitude * nk_sin_f64_cuda_(longitude);
        packed_z[point] = nk_sin_f64_cuda_(latitude);
    }
}

static __global__ void __launch_bounds__(nk_geospatial_threads_cuda_k, 1)
    nk_vincentys_pack_f64_cuda_kernel_(nk_geodesics_pack_arguments_cuda_t arguments) {
    nk_f64_t const flattening = 1.0 / NUMKONG_EARTH_ELLIPSOID_INVERSE_FLATTENING;
    nk_size_t const count = arguments.count;
    nk_size_t const axis_stride = nk_size_round_up_to_multiple_(count, nk_default_alignment_k / sizeof(nk_f64_t));
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        nk_geospatial_packed_header_t *header = (nk_geospatial_packed_header_t *)arguments.packed;
        header->points = count, header->axis_stride = axis_stride;
    }
    nk_f64_t *packed_longitudes = (nk_f64_t *)((char *)arguments.packed + nk_default_alignment_k);
    nk_f64_t *packed_sines = packed_longitudes + axis_stride, *packed_cosines = packed_sines + axis_stride;
    nk_size_t const stride = (nk_size_t)gridDim.x * blockDim.x;
    for (nk_size_t point = (nk_size_t)blockIdx.x * blockDim.x + threadIdx.x; point < count; point += stride) {
        nk_f64_t const latitude = arguments.latitudes[point];
        // The reduced latitude through atan2 stays finite at the poles, where tan(φ) does not.
        nk_f64_t const reduced = nk_atan2_f64_cuda_((1.0 - flattening) * nk_sin_f64_cuda_(latitude),
                                                    nk_cos_f64_cuda_(latitude));
        packed_longitudes[point] = arguments.longitudes[point];
        packed_sines[point] = nk_sin_f64_cuda_(reduced);
        packed_cosines[point] = nk_cos_f64_cuda_(reduced);
    }
}

static __global__ void __launch_bounds__(nk_geospatial_threads_cuda_k, 1)
    nk_haversines_packed_f64_cuda_kernel_(nk_geodesics_arguments_cuda_t arguments) {
    nk_geospatial_packed_header_t const *a_header = (nk_geospatial_packed_header_t const *)arguments.a_packed;
    nk_geospatial_packed_header_t const *b_header = (nk_geospatial_packed_header_t const *)arguments.b_packed;
    nk_size_t const rows = (nk_size_t)a_header->points, a_stride = (nk_size_t)a_header->axis_stride;
    nk_size_t const columns = (nk_size_t)b_header->points, b_stride = (nk_size_t)b_header->axis_stride;
    nk_f64_t const *a_x = (nk_f64_t const *)((char const *)arguments.a_packed + nk_default_alignment_k);
    nk_f64_t const *b_x = (nk_f64_t const *)((char const *)arguments.b_packed + nk_default_alignment_k);
    nk_f64_t const *a_y = a_x + a_stride, *a_z = a_y + a_stride;
    nk_f64_t const *b_y = b_x + b_stride, *b_z = b_y + b_stride;
    nk_size_t const column_tiles = nk_size_divide_round_up_(columns, 32);
    nk_size_t const tiles = nk_size_divide_round_up_(rows, nk_geodesics_rows_cuda_k) * column_tiles;
    nk_size_t const warps = (nk_size_t)gridDim.x * (blockDim.x / 32);
    for (nk_size_t tile = (nk_size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32; tile < tiles; tile += warps) {
        nk_size_t const column = tile % column_tiles * 32 + threadIdx.x % 32;
        nk_size_t const row_begin = tile / column_tiles * nk_geodesics_rows_cuda_k;
        if (column >= columns) continue;
        nk_f64_t const column_x = b_x[column], column_y = b_y[column], column_z = b_z[column];
        for (nk_size_t row = row_begin; row < row_begin + nk_geodesics_rows_cuda_k && row < rows; ++row) {
            nk_f64_t *output = (nk_f64_t *)((char *)arguments.results + row * arguments.results_stride);
            output[column] = nk_haversine_f64_cuda_(a_x[row], a_y[row], a_z[row], column_x, column_y, column_z);
        }
    }
}

static __global__ void __launch_bounds__(nk_geospatial_threads_cuda_k, 1)
    nk_vincentys_packed_f64_cuda_kernel_(nk_geodesics_arguments_cuda_t arguments) {
    nk_geospatial_packed_header_t const *a_header = (nk_geospatial_packed_header_t const *)arguments.a_packed;
    nk_geospatial_packed_header_t const *b_header = (nk_geospatial_packed_header_t const *)arguments.b_packed;
    nk_size_t const rows = (nk_size_t)a_header->points, a_stride = (nk_size_t)a_header->axis_stride;
    nk_size_t const columns = (nk_size_t)b_header->points, b_stride = (nk_size_t)b_header->axis_stride;
    nk_f64_t const *a_longitudes = (nk_f64_t const *)((char const *)arguments.a_packed + nk_default_alignment_k);
    nk_f64_t const *b_longitudes = (nk_f64_t const *)((char const *)arguments.b_packed + nk_default_alignment_k);
    nk_f64_t const *a_sines = a_longitudes + a_stride, *a_cosines = a_sines + a_stride;
    nk_f64_t const *b_sines = b_longitudes + b_stride, *b_cosines = b_sines + b_stride;
    nk_size_t const column_tiles = nk_size_divide_round_up_(columns, 32);
    nk_size_t const tiles = nk_size_divide_round_up_(rows, nk_geodesics_rows_cuda_k) * column_tiles;
    nk_size_t const warps = (nk_size_t)gridDim.x * (blockDim.x / 32);
    for (nk_size_t tile = (nk_size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32; tile < tiles; tile += warps) {
        nk_size_t const column = tile % column_tiles * 32 + threadIdx.x % 32;
        nk_size_t const row_begin = tile / column_tiles * nk_geodesics_rows_cuda_k;
        if (column >= columns) continue;
        nk_f64_t const column_longitude = b_longitudes[column];
        nk_f64_t const column_sine = b_sines[column], column_cosine = b_cosines[column];
        for (nk_size_t row = row_begin; row < row_begin + nk_geodesics_rows_cuda_k && row < rows; ++row) {
            nk_f64_t *output = (nk_f64_t *)((char *)arguments.results + row * arguments.results_stride);
            output[column] = nk_vincenty_f64_cuda_(column_longitude - a_longitudes[row], a_sines[row], a_cosines[row],
                                                   column_sine, column_cosine);
        }
    }
}

static __global__ void __launch_bounds__(nk_geospatial_threads_cuda_k, 1)
    nk_terrain_pack_heights_f32_cuda_kernel_(nk_terrain_pack_arguments_cuda_t arguments) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *(nk_terrain_packed_header_t *)arguments.packed = arguments.header;
    nk_size_t const rows = arguments.rows, columns = arguments.columns, block_columns = arguments.block_columns;
    nk_size_t const blocks = arguments.block_rows * block_columns, stride = (nk_size_t)gridDim.x * blockDim.x;
    // Rows 2i − 1 to 2i + 3, clipped; NaN never wins a comparison.
    for (nk_size_t block = (nk_size_t)blockIdx.x * blockDim.x + threadIdx.x; block < blocks; block += stride) {
        nk_size_t const block_row = block / block_columns, block_column = block % block_columns;
        nk_size_t const first_row = block_row * 2 ? block_row * 2 - 1 : 0;
        nk_size_t const last_row = block_row * 2 + 3 < rows ? block_row * 2 + 3 : rows - 1;
        nk_size_t const first_column = block_column * 2 ? block_column * 2 - 1 : 0;
        nk_size_t const last_column = block_column * 2 + 3 < columns ? block_column * 2 + 3 : columns - 1;
        nk_f32_t maximum = -NUMKONG_F32_INF;
        for (nk_size_t row = first_row; row <= last_row; ++row) {
            nk_f32_t const *heights = (nk_f32_t const *)((char const *)arguments.heights +
                                                         row * arguments.heights_stride);
            for (nk_size_t column = first_column; column <= last_column; ++column)
                if (heights[column] > maximum) maximum = heights[column];
        }
        arguments.maxima[block] = maximum;
    }
}

static __global__ void __launch_bounds__(nk_geospatial_threads_cuda_k, 1)
    nk_terrain_pack_maxima_f32_cuda_kernel_(nk_terrain_pack_arguments_cuda_t arguments) {
    nk_size_t const block_columns = arguments.block_columns, finer_columns = arguments.finer_columns;
    nk_size_t const blocks = arguments.block_rows * block_columns, stride = (nk_size_t)gridDim.x * blockDim.x;
    // Children 2i and 2i + 1 cover the same rows as their parent, rings included.
    for (nk_size_t block = (nk_size_t)blockIdx.x * blockDim.x + threadIdx.x; block < blocks; block += stride) {
        nk_size_t const block_row = block / block_columns, block_column = block % block_columns;
        nk_f32_t maximum = -NUMKONG_F32_INF;
        for (nk_size_t child_row = block_row * 2; child_row < block_row * 2 + 2 && child_row < arguments.finer_rows;
             ++child_row)
            for (nk_size_t child_column = block_column * 2;
                 child_column < block_column * 2 + 2 && child_column < finer_columns; ++child_column) {
                nk_f32_t const child = arguments.finer[child_row * finer_columns + child_column];
                if (child > maximum) maximum = child;
            }
        arguments.maxima[block] = maximum;
    }
}

static __global__ void __launch_bounds__(nk_geospatial_threads_cuda_k, 1)
    nk_terrain_heights_packed_f32_cuda_kernel_(nk_terrain_heights_arguments_cuda_t arguments) {
    nk_terrain_packed_header_t const *header = (nk_terrain_packed_header_t const *)arguments.packed;
    nk_size_t const rows = (nk_size_t)header->rows, columns = (nk_size_t)header->columns;
    nk_size_t const heights_stride = (nk_size_t)header->heights_stride;
    nk_fui32_t off_grid;
    off_grid.u = 0x7FC00000u;
    nk_size_t const stride = (nk_size_t)gridDim.x * blockDim.x;
    for (nk_size_t point = (nk_size_t)blockIdx.x * blockDim.x + threadIdx.x; point < arguments.count; point += stride) {
        nk_f64_t const column = arguments.points[point * 2], row = arguments.points[point * 2 + 1];
        if (!(column >= 0 && column <= (nk_f64_t)(columns - 1) && row >= 0 && row <= (nk_f64_t)(rows - 1))) {
            arguments.results[point] = off_grid.f;
            continue;
        }
        nk_size_t column_first = (nk_size_t)column, row_first = (nk_size_t)row;
        if (column_first + 1 == columns && columns > 1) --column_first;
        if (row_first + 1 == rows && rows > 1) --row_first;
        nk_size_t const column_second = columns > 1 ? column_first + 1 : column_first;
        nk_size_t const row_second = rows > 1 ? row_first + 1 : row_first;
        nk_f64_t const column_fraction = column - (nk_f64_t)column_first, row_fraction = row - (nk_f64_t)row_first;
        nk_f32_t const *first = (nk_f32_t const *)((char const *)arguments.heights + row_first * heights_stride);
        nk_f32_t const *second = (nk_f32_t const *)((char const *)arguments.heights + row_second * heights_stride);
        nk_f64_t const north_west = first[column_first], north_east = first[column_second];
        nk_f64_t const south_west = second[column_first], south_east = second[column_second];
        nk_f64_t const north = nk_f64_add_rn_cuda_(north_west,
                                                   nk_f64_mul_rn_cuda_(column_fraction, north_east - north_west));
        nk_f64_t const south = nk_f64_add_rn_cuda_(south_west,
                                                   nk_f64_mul_rn_cuda_(column_fraction, south_east - south_west));
        arguments.results[point] = (nk_f32_t)nk_f64_add_rn_cuda_(north,
                                                                 nk_f64_mul_rn_cuda_(row_fraction, south - north));
    }
}

static __global__ void __launch_bounds__(nk_geospatial_threads_cuda_k, nk_clearances_blocks_cuda_k)
    nk_clearances_packed_f32_cuda_kernel_(nk_clearances_arguments_cuda_t arguments) {
    __shared__ nk_terrain_grid_cuda_t grid;
    if (threadIdx.x == 0) nk_terrain_grid_f32_cuda_(arguments.heights, arguments.packed, &grid);
    __syncthreads();
    nk_size_t const column_tiles = nk_size_divide_round_up_(arguments.columns, 32);
    nk_size_t const tiles = arguments.rows * column_tiles;
    nk_size_t const warps = (nk_size_t)gridDim.x * (blockDim.x / 32);
    for (nk_size_t tile = (nk_size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32; tile < tiles; tile += warps) {
        nk_size_t const row = tile / column_tiles, column = tile % column_tiles * 32 + threadIdx.x % 32;
        if (column >= arguments.columns) continue;
        nk_f32_t *output = (nk_f32_t *)((char *)arguments.results + row * arguments.results_stride);
        output[column] = nk_clearance_f32_cuda_(&grid, &arguments, row, column);
    }
}

NUMKONG_API nk_status_t nk_haversines_pack_size_f64_cuda(nk_size_t points, nk_size_t *bytes) {
    *bytes = nk_default_alignment_k +
             3 * nk_size_round_up_to_multiple_(points, nk_default_alignment_k / sizeof(nk_f64_t)) * sizeof(nk_f64_t);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_vincentys_pack_size_f64_cuda(nk_size_t points, nk_size_t *bytes) {
    *bytes = nk_default_alignment_k +
             3 * nk_size_round_up_to_multiple_(points, nk_default_alignment_k / sizeof(nk_f64_t)) * sizeof(nk_f64_t);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_haversines_pack_f64_cuda(nk_f64_t const *latitudes, nk_f64_t const *longitudes,
                                                    nk_size_t count, void *packed, nk_stream_t stream) {
    return nk_geodesics_pack_launch_cuda_((void const *)&nk_haversines_pack_f64_cuda_kernel_, latitudes, longitudes,
                                          count, packed, stream);
}

NUMKONG_API nk_status_t nk_vincentys_pack_f64_cuda(nk_f64_t const *latitudes, nk_f64_t const *longitudes,
                                                   nk_size_t count, void *packed, nk_stream_t stream) {
    return nk_geodesics_pack_launch_cuda_((void const *)&nk_vincentys_pack_f64_cuda_kernel_, latitudes, longitudes,
                                          count, packed, stream);
}

NUMKONG_API nk_status_t nk_haversines_packed_f64_cuda(void const *a_packed, void const *b_packed, nk_f64_t *results,
                                                      nk_size_t results_stride, nk_stream_t stream) {
    return nk_geodesics_launch_cuda_((void const *)&nk_haversines_packed_f64_cuda_kernel_, a_packed, b_packed, results,
                                     results_stride, stream);
}

NUMKONG_API nk_status_t nk_vincentys_packed_f64_cuda(void const *a_packed, void const *b_packed, nk_f64_t *results,
                                                     nk_size_t results_stride, nk_stream_t stream) {
    return nk_geodesics_launch_cuda_((void const *)&nk_vincentys_packed_f64_cuda_kernel_, a_packed, b_packed, results,
                                     results_stride, stream);
}

NUMKONG_API nk_status_t nk_terrain_pack_size_f32_cuda(nk_size_t rows, nk_size_t columns, nk_size_t *bytes) {
    nk_size_t const levels = nk_terrain_levels_f32_serial_(rows, columns);
    if (levels > nk_terrain_levels_cuda_k) return nk_unexpected_dimensions_k;
    nk_size_t total = nk_default_alignment_k;
    for (nk_size_t level = 1; level <= levels; ++level)
        total += nk_terrain_level_blocks_f32_serial_(rows, level) *
                 nk_terrain_level_blocks_f32_serial_(columns, level) * sizeof(nk_f32_t);
    *bytes = total;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_terrain_pack_f32_cuda(nk_f32_t const *heights, nk_size_t rows, nk_size_t columns,
                                                 nk_size_t heights_stride, nk_f64_t row_metres, nk_f64_t column_metres,
                                                 void *packed, nk_stream_t stream) {
    if (!rows || !columns) return nk_unexpected_dimensions_k;
    nk_size_t const levels = nk_terrain_levels_f32_serial_(rows, columns);
    if (levels > nk_terrain_levels_cuda_k) return nk_unexpected_dimensions_k;
    if ((((nk_size_t)heights) | heights_stride) & 3 || (nk_size_t)packed & 7) return nk_misaligned_k;
    nk_terrain_pack_arguments_cuda_t arguments;
    arguments.heights = heights, arguments.heights_stride = heights_stride;
    arguments.rows = rows, arguments.columns = columns, arguments.packed = packed;
    arguments.header.rows = rows, arguments.header.columns = columns, arguments.header.heights_stride = heights_stride;
    arguments.header.levels = levels, arguments.header.row_metres = row_metres;
    arguments.header.column_metres = column_metres;
    arguments.finer = NUMKONG_NULL, arguments.finer_rows = 0, arguments.finer_columns = 0;
    arguments.maxima = (nk_f32_t *)((char *)packed + nk_default_alignment_k);
    // The first launch writes the header, so it runs even for a grid with no levels.
    arguments.block_rows = levels ? nk_terrain_level_blocks_f32_serial_(rows, 1) : 0;
    arguments.block_columns = levels ? nk_terrain_level_blocks_f32_serial_(columns, 1) : 0;
    nk_size_t blocks = nk_size_divide_round_up_(arguments.block_rows * arguments.block_columns,
                                                nk_geospatial_threads_cuda_k);
    nk_status_t status = nk_launch_resident_cuda_((void const *)&nk_terrain_pack_heights_f32_cuda_kernel_,
                                                  nk_geospatial_threads_cuda_k, 0, 0, blocks ? blocks : 1, &arguments,
                                                  stream);
    for (nk_size_t level = 2; level <= levels && status == nk_success_k; ++level) {
        arguments.finer = arguments.maxima;
        arguments.finer_rows = arguments.block_rows, arguments.finer_columns = arguments.block_columns;
        arguments.maxima += arguments.block_rows * arguments.block_columns;
        arguments.block_rows = nk_terrain_level_blocks_f32_serial_(rows, level);
        arguments.block_columns = nk_terrain_level_blocks_f32_serial_(columns, level);
        blocks = nk_size_divide_round_up_(arguments.block_rows * arguments.block_columns, nk_geospatial_threads_cuda_k);
        status = nk_launch_resident_cuda_((void const *)&nk_terrain_pack_maxima_f32_cuda_kernel_,
                                          nk_geospatial_threads_cuda_k, 0, 0, blocks, &arguments, stream);
    }
    return status;
}

NUMKONG_API nk_status_t nk_terrain_heights_packed_f32_cuda(nk_f32_t const *heights, void const *packed,
                                                           nk_f32_t const *points, nk_size_t count, nk_f32_t *results,
                                                           nk_stream_t stream) {
    if ((((nk_size_t)heights) | ((nk_size_t)points) | ((nk_size_t)results)) & 3 || (nk_size_t)packed & 7)
        return nk_misaligned_k;
    if (!count) return nk_success_k;
    nk_terrain_heights_arguments_cuda_t arguments;
    arguments.heights = heights, arguments.packed = packed, arguments.points = points;
    arguments.count = count, arguments.results = results;
    return nk_launch_resident_cuda_((void const *)&nk_terrain_heights_packed_f32_cuda_kernel_,
                                    nk_geospatial_threads_cuda_k, 0, 0,
                                    nk_size_divide_round_up_(count, nk_geospatial_threads_cuda_k), &arguments, stream);
}

NUMKONG_API nk_status_t nk_clearances_packed_f32_cuda(nk_f32_t const *heights, void const *packed,
                                                      nk_f32_t const *viewpoints, nk_f32_t const *views, nk_size_t rows,
                                                      nk_f32_t const *targets, nk_size_t columns, nk_f64_t refraction,
                                                      nk_f32_t *results, nk_size_t results_stride, nk_stream_t stream) {
    if ((((nk_size_t)heights) | ((nk_size_t)viewpoints) | ((nk_size_t)views) | ((nk_size_t)targets) |
         ((nk_size_t)results) | results_stride) &
            3 ||
        (nk_size_t)packed & 7)
        return nk_misaligned_k;
    if (!rows || !columns) return nk_success_k;
    nk_clearances_arguments_cuda_t arguments;
    arguments.heights = heights, arguments.packed = packed;
    arguments.viewpoints = viewpoints, arguments.views = views, arguments.rows = rows;
    arguments.targets = targets, arguments.columns = columns;
    arguments.drop_scale = (1.0 - refraction) / (2.0 * NUMKONG_EARTH_MEAN_RADIUS);
    arguments.results = results, arguments.results_stride = results_stride;
    // Sight lines differ in cost, so the hardware hands out every block instead of a striding grid
    nk_size_t const tiles = rows * nk_size_divide_round_up_(columns, 32);
    nk_size_t const blocks = nk_size_divide_round_up_(tiles, nk_geospatial_threads_cuda_k / 32);
    void *launch_arguments[1];
    launch_arguments[0] = &arguments;
    return nk_launch_cuda_((void const *)&nk_clearances_packed_f32_cuda_kernel_,
                           blocks < 0x7FFFFFFF ? blocks : 0x7FFFFFFF, nk_geospatial_threads_cuda_k, launch_arguments, 0,
                           stream);
}

#endif // NUMKONG_TARGET_CUDA

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_
#endif // NUMKONG_GEOSPATIAL_CUDA_CUH
