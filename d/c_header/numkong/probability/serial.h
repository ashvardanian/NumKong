/**
 *  @brief Serial Probability Distribution Similarity Measures.
 *  @file include/numkong/probability/serial.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *
 *  @sa include/numkong/probability.h
 */
#ifndef NK_PROBABILITY_SERIAL_H
#define NK_PROBABILITY_SERIAL_H

#include "numkong/types.h"
#include "numkong/cast/serial.h"    // `nk_f16_to_f32_serial`, `nk_bf16_to_f32_serial`, `nk_assign_from_to_`
#include "numkong/spatial/serial.h" // `nk_f32_sqrt_serial`, `nk_f64_sqrt_serial`

#if defined(__cplusplus)
extern "C" {
#endif

#define nk_define_kld_(input_type, accumulator_type, output_type, load_and_convert, epsilon, compute_log)   \
    NK_PUBLIC void nk_kld_##input_type##_serial(nk_##input_type##_t const *a, nk_##input_type##_t const *b, \
                                                nk_size_t n, output_type *result) {                         \
        nk_##accumulator_type##_t d = 0, ai, bi;                                                            \
        for (nk_size_t i = 0; i != n; ++i) {                                                                \
            load_and_convert(a + i, &ai);                                                                   \
            load_and_convert(b + i, &bi);                                                                   \
            d += ai * compute_log((ai + epsilon) / (bi + epsilon));                                         \
        }                                                                                                   \
        *result = (output_type)d;                                                                           \
    }

#define nk_define_jsd_(input_type, accumulator_type, output_type, load_and_convert, epsilon, compute_log,   \
                       compute_sqrt)                                                                        \
    NK_PUBLIC void nk_jsd_##input_type##_serial(nk_##input_type##_t const *a, nk_##input_type##_t const *b, \
                                                nk_size_t n, output_type *result) {                         \
        nk_##accumulator_type##_t d = 0, ai, bi;                                                            \
        for (nk_size_t i = 0; i != n; ++i) {                                                                \
            load_and_convert(a + i, &ai);                                                                   \
            load_and_convert(b + i, &bi);                                                                   \
            nk_##accumulator_type##_t mi = (ai + bi) / 2;                                                   \
            d += ai * compute_log((ai + epsilon) / (mi + epsilon));                                         \
            d += bi * compute_log((bi + epsilon) / (mi + epsilon));                                         \
        }                                                                                                   \
        output_type d_half = ((output_type)d / 2);                                                          \
        *result = d_half > 0 ? compute_sqrt(d_half) : 0;                                                    \
    }

/**
 *  @brief  Computes `log(x)` for any positive float using IEEE 754 bit extraction
 *          and a fast-converging series expansion.
 *
 *  Exploits the IEEE 754 representation to extract the exponent and mantissa:
 *  `log(x) = log(2) * exponent + log(mantissa)`. The mantissa is reduced to the
 *  range `[√2/2, √2]` for optimal convergence. Uses the transformation
 *  `u = (m-1)/(m+1)` which converges much faster than the classic Mercator series,
 *  since `u` is bounded to approximately `[-0.17, 0.17]` after range reduction.
 *
 *  Maximum relative error is approximately 0.00001% across all positive floats,
 *  roughly 300,000x more accurate than the 3-term Mercator series (which also
 *  only converges for inputs in `(0, 2)`).
 *
 *  https://en.wikipedia.org/wiki/Logarithm#Power_series
 */
NK_INTERNAL nk_f32_t nk_f32_log_serial_(nk_f32_t x) {
    nk_fui32_t conv;
    conv.f = x;
    int exp = ((conv.u >> 23) & 0xFF) - 127;
    conv.u = (conv.u & 0x007FFFFF) | 0x3F800000; // mantissa ∈ [1, 2)
    nk_f32_t m = conv.f;
    // Range reduction: if m > √2, halve it and increment exponent
    if (m > 1.41421356f) m *= 0.5f, exp++;
    // Use (m-1)/(m+1) transformation for faster convergence
    nk_f32_t u = (m - 1.0f) / (m + 1.0f);
    nk_f32_t u2 = u * u;
    // log(m) = 2 × (u + u³/3 + u⁵/5 + u⁷/7)
    nk_f32_t log_m = 2.0f * u * (1.0f + u2 * (0.3333333333f + u2 * (0.2f + u2 * 0.142857143f)));
    return (nk_f32_t)exp * 0.6931471805599453f + log_m;
}

/**
 *  @brief  Computes `log(x)` for any positive double using IEEE 754 bit extraction
 *          and a fast-converging series expansion.
 *
 *  Exploits the IEEE 754 representation to extract the 11-bit exponent and 52-bit mantissa:
 *  `log(x) = log(2) * exponent + log(mantissa)`. The mantissa is reduced to the
 *  range `[√2/2, √2]` for optimal convergence. Uses the transformation
 *  `u = (m-1)/(m+1)` which converges much faster than the classic Mercator series,
 *  since `u` is bounded to approximately `[-0.17, 0.17]` after range reduction.
 *
 *  Uses more series terms than the f32 version to achieve near-full f64 precision,
 *  with maximum relative error approximately 0.00000000000001% across all positive doubles.
 *
 *  https://en.wikipedia.org/wiki/Logarithm#Power_series
 */
NK_INTERNAL nk_f64_t nk_f64_log_serial_(nk_f64_t x) {
    nk_fui64_t conv;
    conv.f = x;
    int exp = ((conv.u >> 52) & 0x7FF) - 1023;
    conv.u = (conv.u & 0x000FFFFFFFFFFFFFULL) | 0x3FF0000000000000ULL; // mantissa ∈ [1, 2)
    nk_f64_t m = conv.f;
    // Range reduction: if m > √2, halve it and increment exponent
    if (m > 1.4142135623730950488) m *= 0.5, exp++;
    // Use (m-1)/(m+1) transformation for faster convergence
    nk_f64_t u = (m - 1.0) / (m + 1.0);
    nk_f64_t u2 = u * u;
    // 14-term Horner: P(u²) = 1 + u²/3 + u⁴/5 + ... + u²⁶/27, matching SIMD
    nk_f64_t poly = 1.0 / 27.0;
    poly = u2 * poly + 1.0 / 25.0;
    poly = u2 * poly + 1.0 / 23.0;
    poly = u2 * poly + 1.0 / 21.0;
    poly = u2 * poly + 1.0 / 19.0;
    poly = u2 * poly + 1.0 / 17.0;
    poly = u2 * poly + 1.0 / 15.0;
    poly = u2 * poly + 1.0 / 13.0;
    poly = u2 * poly + 1.0 / 11.0;
    poly = u2 * poly + 1.0 / 9.0;
    poly = u2 * poly + 1.0 / 7.0;
    poly = u2 * poly + 1.0 / 5.0;
    poly = u2 * poly + 1.0 / 3.0;
    poly = u2 * poly + 1.0;
    return (nk_f64_t)exp * 0.6931471805599453 + 2.0 * u * poly;
}

nk_define_kld_(f32, f64, nk_f64_t, nk_assign_from_to_, NK_F32_DIVISION_EPSILON, nk_f32_log_serial_)
nk_define_jsd_(f32, f64, nk_f64_t, nk_assign_from_to_, NK_F32_DIVISION_EPSILON, nk_f32_log_serial_, nk_f64_sqrt_serial)

nk_define_kld_(f16, f32, nk_f32_t, nk_f16_to_f32_serial, NK_F32_DIVISION_EPSILON, nk_f32_log_serial_)
nk_define_jsd_(f16, f32, nk_f32_t, nk_f16_to_f32_serial, NK_F32_DIVISION_EPSILON, nk_f32_log_serial_,
               nk_f32_sqrt_serial)

nk_define_kld_(bf16, f32, nk_f32_t, nk_bf16_to_f32_serial, NK_F32_DIVISION_EPSILON, nk_f32_log_serial_)
nk_define_jsd_(bf16, f32, nk_f32_t, nk_bf16_to_f32_serial, NK_F32_DIVISION_EPSILON, nk_f32_log_serial_,
               nk_f32_sqrt_serial)

NK_PUBLIC void nk_kld_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_f64_t sum = 0, compensation = 0;
    for (nk_size_t i = 0; i != n; ++i) {
        nk_f64_t ai = a[i], bi = b[i];
        nk_f64_t term = ai * nk_f64_log_serial_((ai + NK_F64_DIVISION_EPSILON) / (bi + NK_F64_DIVISION_EPSILON));
        nk_f64_t t = sum + term;
        compensation += (nk_f64_abs_(sum) >= nk_f64_abs_(term)) ? ((sum - t) + term) : ((term - t) + sum);
        sum = t;
    }
    *result = sum + compensation;
}

NK_PUBLIC void nk_jsd_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_f64_t sum = 0, compensation = 0;
    for (nk_size_t i = 0; i != n; ++i) {
        nk_f64_t ai = a[i], bi = b[i];
        nk_f64_t mi = (ai + bi) / 2;
        nk_f64_t term_a = ai * nk_f64_log_serial_((ai + NK_F64_DIVISION_EPSILON) / (mi + NK_F64_DIVISION_EPSILON));
        nk_f64_t t = sum + term_a;
        compensation += (nk_f64_abs_(sum) >= nk_f64_abs_(term_a)) ? ((sum - t) + term_a) : ((term_a - t) + sum);
        sum = t;
        nk_f64_t term_b = bi * nk_f64_log_serial_((bi + NK_F64_DIVISION_EPSILON) / (mi + NK_F64_DIVISION_EPSILON));
        t = sum + term_b;
        compensation += (nk_f64_abs_(sum) >= nk_f64_abs_(term_b)) ? ((sum - t) + term_b) : ((term_b - t) + sum);
        sum = t;
    }
    nk_f64_t d_half = (sum + compensation) / 2;
    *result = d_half > 0 ? nk_f64_sqrt_serial(d_half) : 0;
}

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_PROBABILITY_SERIAL_H
