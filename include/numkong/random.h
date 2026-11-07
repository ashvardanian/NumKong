/**
 *  @brief Stateful pseudo-random sampling, independent of host array libraries.
 *  @file include/numkong/random.h
 *  @see https://prng.di.unimi.it/splitmix64.c
 *  @see https://github.com/numpy/numpy/tree/main/numpy/random/src/distributions
 *
 *  Seeds and state are 64-bit integers. Floating distributions use host libm;
 *  their last bits can differ between platforms. Ziggurat, Johnk, Marsaglia-Tsang,
 *  and Hoermann PTRS sampling follow the algorithms used by NumPy. The retained
 *  NumPy copyright and BSD license are in random/ziggurat.h.
 */
#ifndef NK_RANDOM_H
#define NK_RANDOM_H

#include <math.h>
#include <stdint.h>

#include "numkong/types.h"
#include "numkong/random/ziggurat.h"

#if defined(__cplusplus)
extern "C" {
#endif

/** @brief Independent SplitMix64 stream. */
typedef struct nk_random_generator_t {
    uint64_t state;
} nk_random_generator_t;

/** @brief Initialize or restore a stream. */
NK_PUBLIC void nk_random_seed(nk_random_generator_t *generator, uint64_t seed) { generator->state = seed; }

/** @brief Advance the stream by one 64-bit word. */
NK_PUBLIC uint64_t nk_random_next_u64(nk_random_generator_t *generator) {
    uint64_t value = (generator->state += UINT64_C(0x9E3779B97F4A7C15));
    value = (value ^ (value >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94D049BB133111EB);
    return value ^ (value >> 31);
}

/** @brief Uniform sample in [0, 1), with 53 random mantissa bits. */
NK_PUBLIC double nk_random_uniform_f64(nk_random_generator_t *generator) {
    return (double)(nk_random_next_u64(generator) >> 11) * 0x1.0p-53;
}

/** @brief Uniform sample in [0, 1), with 24 random mantissa bits. */
NK_PUBLIC float nk_random_uniform_f32(nk_random_generator_t *generator) {
    return (float)(nk_random_next_u64(generator) >> 40) * 0x1.0p-24f;
}

/** @brief Unbiased integer in [0, bound); zero bound denotes the full uint64 range. */
NK_PUBLIC uint64_t nk_random_bounded_u64(nk_random_generator_t *generator, uint64_t bound) {
    uint64_t value = nk_random_next_u64(generator);
    if (!(bound & (bound - 1))) return bound ? value & (bound - 1) : value;
#if defined(__SIZEOF_INT128__)
    __uint128_t product = (__uint128_t)value * bound;
    uint64_t low = (uint64_t)product;
    if (low < bound) {
        uint64_t const threshold = (uint64_t)(0 - bound) % bound;
        while (low < threshold) {
            product = (__uint128_t)nk_random_next_u64(generator) * bound;
            low = (uint64_t)product;
        }
    }
    return (uint64_t)(product >> 64);
#else
    uint64_t const threshold = (uint64_t)(0 - bound) % bound;
    while (value < threshold) value = nk_random_next_u64(generator);
    return value % bound;
#endif
}

/** @brief Standard normal sample from a 256-strip Ziggurat. */
NK_INTERNAL double nk_random_standard_normal_f64_(nk_random_generator_t *generator) {
    double const radius = 3.6541528853610088;
    double const inverse_radius = 0.27366123732975828;
    double const signs[] = {1.0, -1.0};
    for (;;) {
        uint64_t bits = nk_random_next_u64(generator);
        unsigned int const index = (unsigned int)(bits & 255);
        bits >>= 8;
        uint64_t const magnitude = (bits >> 1) & UINT64_C(0x000fffffffffffff);
        double const value = (double)magnitude * nk_random_wi_double_[index] * signs[bits & 1];
        if (magnitude < nk_random_ki_double_[index]) return value;
        if (!index) {
            double tail, ordinate;
            do {
                tail = -inverse_radius * log1p(-nk_random_uniform_f64(generator));
                ordinate = -log1p(-nk_random_uniform_f64(generator));
            } while (2 * ordinate <= tail * tail);
            return ((magnitude >> 8) & 1) ? -(radius + tail) : radius + tail;
        }
        if ((nk_random_fi_double_[index - 1] - nk_random_fi_double_[index]) * nk_random_uniform_f64(generator) +
                nk_random_fi_double_[index] <
            exp(-0.5 * value * value))
            return value;
    }
}

/** @brief Standard normal sample with double precision. */
NK_PUBLIC double nk_random_standard_normal_f64(nk_random_generator_t *generator) {
    return nk_random_standard_normal_f64_(generator);
}

/** @brief Standard normal sample using single-precision Ziggurat tables. */
NK_INTERNAL float nk_random_standard_normal_f32_(nk_random_generator_t *generator) {
    float const radius = 3.6541528853610088f, inverse_radius = 0.27366123732975828f;
    float const signs[] = {1.0f, -1.0f};
    for (;;) {
        uint32_t const bits = (uint32_t)(nk_random_next_u64(generator) >> 32);
        unsigned int const index = bits & 255;
        uint32_t const magnitude = (bits >> 9) & 0x007fffff;
        float const value = (float)magnitude * nk_random_wi_float_[index] * signs[(bits >> 8) & 1];
        if (magnitude < nk_random_ki_float_[index]) return value;
        if (!index) {
            float tail, ordinate;
            do {
                tail = -inverse_radius * log1pf(-nk_random_uniform_f32(generator));
                ordinate = -log1pf(-nk_random_uniform_f32(generator));
            } while (2 * ordinate <= tail * tail);
            return ((magnitude >> 8) & 1) ? -(radius + tail) : radius + tail;
        }
        if ((nk_random_fi_float_[index - 1] - nk_random_fi_float_[index]) * nk_random_uniform_f32(generator) +
                nk_random_fi_float_[index] <
            expf(-0.5f * value * value))
            return value;
    }
}

/** @brief Standard normal sample with single precision. */
NK_PUBLIC float nk_random_standard_normal_f32(nk_random_generator_t *generator) {
    return nk_random_standard_normal_f32_(generator);
}

/** @brief Normal sample; caller validates finite location and nonnegative scale. */
NK_PUBLIC double nk_random_normal_f64(nk_random_generator_t *generator, double location, double scale) {
    return location + scale * nk_random_standard_normal_f64_(generator);
}

/** @brief Laplace sample; caller validates location and scale. */
NK_PUBLIC double nk_random_laplace_f64(nk_random_generator_t *generator, double location, double scale) {
    double value;
    do { value = nk_random_uniform_f64(generator); } while (value == 0.0);
    double const lower = value + value, upper = 2.0 - value - value;
    return location + (value >= 0.5 ? -scale : scale) * log(lower < upper ? lower : upper);
}

NK_INTERNAL double nk_random_gamma_f64_(nk_random_generator_t *generator, double shape) {
    if (shape == 1.0) return -log1p(-nk_random_uniform_f64(generator));
    double const original_shape = shape;
    int const adjust = shape < 1.0;
    if (adjust) shape += 1;
    double const base = shape - 1.0 / 3.0;
    double const factor = 1.0 / sqrt(9.0 * base);
    for (;;) {
        double normal, volume;
        do {
            normal = nk_random_standard_normal_f64(generator);
            volume = 1 + factor * normal;
        } while (volume <= 0.0);
        volume *= volume * volume;
        double const uniform = nk_random_uniform_f64(generator);
        double const squared = normal * normal;
        if (uniform < 1 - 0.0331 * squared * squared ||
            log(uniform) < 0.5 * squared + base * (1 - volume + log(volume)))
            return adjust ? base * volume * pow(nk_random_uniform_f64(generator), 1.0 / original_shape) : base * volume;
    }
}

/** @brief Beta sample for finite positive shape parameters. */
NK_PUBLIC double nk_random_beta_f64(nk_random_generator_t *generator, double alpha, double beta) {
    if (alpha <= 1.0 && beta <= 1.0) {
        if (alpha < 3e-103 && beta < 3e-103) return (alpha + beta) * nk_random_uniform_f64(generator) < alpha;
        for (;;) {
            double const first = nk_random_uniform_f64(generator), second = nk_random_uniform_f64(generator);
            double const x = pow(first, 1.0 / alpha), y = pow(second, 1.0 / beta);
            if (x + y > 1.0 || first + second == 0.0) continue;
            if (x > 0.0 && y > 0.0) return x / (x + y);
            double const difference = log(first) / alpha - log(second) / beta;
            return difference > 0 ? exp(-log1p(exp(-difference))) : exp(difference - log1p(exp(difference)));
        }
    }
    double const x = nk_random_gamma_f64_(generator, alpha), y = nk_random_gamma_f64_(generator, beta);
    return x > y ? 1.0 / (1.0 + y / x) : (x / y) / (1.0 + x / y);
}

static const double nk_random_log_factorial_table_[] = {0,
                                                        0,
                                                        0.69314718055994529,
                                                        1.791759469228055,
                                                        3.1780538303479458,
                                                        4.7874917427820458,
                                                        6.5792512120101012,
                                                        8.5251613610654147,
                                                        10.604602902745251,
                                                        12.801827480081469,
                                                        15.104412573075516,
                                                        17.502307845873887,
                                                        19.987214495661885,
                                                        22.552163853123425,
                                                        25.19122118273868,
                                                        27.89927138384089,
                                                        30.671860106080672,
                                                        33.505073450136891,
                                                        36.395445208033053,
                                                        39.339884187199495,
                                                        42.335616460753485,
                                                        45.380138898476908,
                                                        48.471181351835227,
                                                        51.606675567764377,
                                                        54.784729398112319,
                                                        58.003605222980518,
                                                        61.261701761002001,
                                                        64.557538627006338,
                                                        67.88974313718154,
                                                        71.257038967168015,
                                                        74.658236348830158,
                                                        78.092223553315307,
                                                        81.557959456115043,
                                                        85.054467017581516,
                                                        88.580827542197682,
                                                        92.136175603687093,
                                                        95.719694542143202,
                                                        99.330612454787428,
                                                        102.96819861451381,
                                                        106.63176026064346,
                                                        110.32063971475739,
                                                        114.03421178146171,
                                                        117.77188139974507,
                                                        121.53308151543864,
                                                        125.3172711493569,
                                                        129.12393363912722,
                                                        132.95257503561632,
                                                        136.80272263732635,
                                                        140.67392364823425,
                                                        144.5657439463449,
                                                        148.47776695177302,
                                                        152.40959258449735,
                                                        156.3608363030788,
                                                        160.3311282166309,
                                                        164.32011226319517,
                                                        168.32744544842765,
                                                        172.35279713916279,
                                                        176.39584840699735,
                                                        180.45629141754378,
                                                        184.53382886144948,
                                                        188.6281734236716,
                                                        192.7390472878449,
                                                        196.86618167289001,
                                                        201.00931639928152,
                                                        205.1681994826412,
                                                        209.34258675253685,
                                                        213.53224149456327,
                                                        217.73693411395422,
                                                        221.95644181913033,
                                                        226.1905483237276,
                                                        230.43904356577696,
                                                        234.70172344281826,
                                                        238.97838956183432,
                                                        243.26884900298271,
                                                        247.57291409618688,
                                                        251.89040220972319,
                                                        256.22113555000954,
                                                        260.56494097186322,
                                                        264.92164979855278,
                                                        269.29109765101981,
                                                        273.67312428569369,
                                                        278.06757344036612,
                                                        282.4742926876304,
                                                        286.89313329542699,
                                                        291.32395009427029,
                                                        295.76660135076065,
                                                        300.22094864701415,
                                                        304.68685676566872,
                                                        309.1641935801469,
                                                        313.65282994987905,
                                                        318.1526396202093,
                                                        322.66349912672615,
                                                        327.1852877037752,
                                                        331.71788719692847,
                                                        336.26118197919845,
                                                        340.81505887079902,
                                                        345.37940706226686,
                                                        349.95411804077025,
                                                        354.53908551944079,
                                                        359.1342053695754,
                                                        363.73937555556347,
                                                        368.35449607240474,
                                                        372.97946888568902,
                                                        377.61419787391867,
                                                        382.25858877306001,
                                                        386.91254912321756,
                                                        391.57598821732961,
                                                        396.24881705179155,
                                                        400.93094827891576,
                                                        405.6222961611449,
                                                        410.32277652693733,
                                                        415.03230672824964,
                                                        419.75080559954472,
                                                        424.47819341825709,
                                                        429.21439186665157,
                                                        433.95932399501481,
                                                        438.71291418612117,
                                                        443.47508812091894,
                                                        448.24577274538461,
                                                        453.02489623849613,
                                                        457.81238798127816,
                                                        462.60817852687489,
                                                        467.4121995716082,
                                                        472.22438392698058,
                                                        477.04466549258564,
                                                        481.87297922988796};

NK_INTERNAL double nk_random_log_factorial_(int64_t count) {
    if (count < (int64_t)(sizeof(nk_random_log_factorial_table_) / sizeof(double)))
        return nk_random_log_factorial_table_[count];
    double const value = (double)count, inverse = 1.0 / value;
    return (value + 0.5) * log(value) - value +
           (0.9189385332046728 + inverse * (1.0 / 12.0 - inverse * inverse / 360.0));
}

/** @brief Poisson sample; caller validates 0 <= rate <= INT64_MAX - 10*sqrt(INT64_MAX). */
NK_PUBLIC int64_t nk_random_poisson_i64(nk_random_generator_t *generator, double rate) {
    if (rate == 0.0) return 0;
    if (rate < 10.0) {
        double const limit = exp(-rate);
        double product = nk_random_uniform_f64(generator);
        int64_t count = 0;
        while (product > limit) {
            ++count;
            product *= nk_random_uniform_f64(generator);
        }
        return count;
    }
    double const b = 0.931 + 2.53 * sqrt(rate), a = -0.059 + 0.02483 * b;
    double const inverse_alpha = 1.1239 + 1.1328 / (b - 3.4), fast_limit = 0.9277 - 3.6224 / (b - 2);
    double const log_rate = log(rate);
    for (;;) {
        double const u = nk_random_uniform_f64(generator) - 0.5, v = nk_random_uniform_f64(generator);
        double const distance = 0.5 - fabs(u);
        if (distance == 0.0) continue;
        double const candidate = floor((2 * a / distance + b) * u + rate + 0.43);
        if (candidate < 0 || candidate >= 0x1.0p63) continue;
        if (distance >= 0.07 && v <= fast_limit) return (int64_t)candidate;
        if (distance < 0.013 && v > distance) continue;
        if (log(v) + log(inverse_alpha) - log(a / (distance * distance) + b) <=
            -rate + candidate * log_rate - nk_random_log_factorial_((int64_t)candidate))
            return (int64_t)candidate;
    }
}

#if defined(__cplusplus)
}
#endif
#endif
