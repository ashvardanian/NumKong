/**
 *  @file include/numkong/spatial/serial.h
 *  @author Ash Vardanian
 *  @date March 14, 2023
 *  @brief SWAR-accelerated spatial similarity measures for SIMD-free CPUs.
 *
 *  @sa include/numkong/spatial.h
 */
#ifndef NUMKONG_SPATIAL_SERIAL_H
#define NUMKONG_SPATIAL_SERIAL_H

#include "numkong/types.h"
#include "numkong/scalar/serial.h" // `nk_f32_rsqrt_`
#include "numkong/cast/serial.h"
#include "numkong/dot/serial.h" // `nk_dot_f64x2_state_serial_t`

#if defined(__cplusplus)
extern "C" {
#endif

/** Generates @c nk_sqeuclidean_<input_type>_, a squared distance with simple accumulation. */
#define nk_define_sqeuclidean_(input_type, accumulator_type, output_type, load_and_convert)                        \
    NUMKONG_INLINE void nk_sqeuclidean_##input_type##_(nk_##input_type##_t const *a, nk_##input_type##_t const *b, \
                                                       nk_size_t n, nk_##output_type##_t *result) {                \
        nk_##accumulator_type##_t sum = 0, a_value, b_value;                                                       \
        for (nk_size_t i = 0; i != n; ++i) {                                                                       \
            load_and_convert(a + i, &a_value);                                                                     \
            load_and_convert(b + i, &b_value);                                                                     \
            nk_##accumulator_type##_t diff = a_value - b_value;                                                    \
            sum += diff * diff;                                                                                    \
        }                                                                                                          \
        *result = (nk_##output_type##_t)sum;                                                                       \
    }

/** Generates @c nk_sqeuclidean_<input_type>_serial, the public kernel over its helper. */
#define nk_define_sqeuclidean_serial_(input_type, output_type)                                                 \
    NUMKONG_API nk_status_t nk_sqeuclidean_##input_type##_serial(nk_##input_type##_t const *a,                 \
                                                                 nk_##input_type##_t const *b, nk_size_t n,    \
                                                                 nk_##output_type##_t *result, void *stream) { \
        nk_assert_(stream == NUMKONG_NULL);                                                                    \
        nk_sqeuclidean_##input_type##_(a, b, n, result);                                                       \
        return nk_success_k;                                                                                   \
    }

#define nk_define_euclidean_(input_type, accumulator_type, l2sq_output_type, output_type, load_and_convert,  \
                             compute_sqrt)                                                                   \
    NUMKONG_API nk_status_t nk_euclidean_##input_type##_serial(nk_##input_type##_t const *a,                 \
                                                               nk_##input_type##_t const *b, nk_size_t n,    \
                                                               nk_##output_type##_t *result, void *stream) { \
        nk_assert_(stream == NUMKONG_NULL);                                                                  \
        nk_##l2sq_output_type##_t distance_sq;                                                               \
        nk_sqeuclidean_##input_type##_(a, b, n, &distance_sq);                                               \
        *result = compute_sqrt((nk_##output_type##_t)distance_sq);                                           \
        return nk_success_k;                                                                                 \
    }

/** Generates @c nk_angular_<input_type>_serial, an angular distance with simple accumulation. */
#define nk_define_angular_(input_type, accumulator_type, output_type, load_and_convert, compute_rsqrt)     \
    NUMKONG_API nk_status_t nk_angular_##input_type##_serial(nk_##input_type##_t const *a,                 \
                                                             nk_##input_type##_t const *b, nk_size_t n,    \
                                                             nk_##output_type##_t *result, void *stream) { \
        nk_assert_(stream == NUMKONG_NULL);                                                                \
        nk_##accumulator_type##_t dot_product = 0, a_norm_sq = 0, b_norm_sq = 0, a_value, b_value;         \
        for (nk_size_t i = 0; i != n; ++i) {                                                               \
            load_and_convert(a + i, &a_value);                                                             \
            load_and_convert(b + i, &b_value);                                                             \
            dot_product += a_value * b_value;                                                              \
            a_norm_sq += a_value * a_value;                                                                \
            b_norm_sq += b_value * b_value;                                                                \
        }                                                                                                  \
        if (a_norm_sq == 0 && b_norm_sq == 0) { *result = 0; }                                             \
        else if (dot_product == 0) { *result = 1; }                                                        \
        else {                                                                                             \
            nk_##output_type##_t unclipped_distance = (nk_##output_type##_t)(                              \
                1 - (nk_##output_type##_t)dot_product * compute_rsqrt((nk_##output_type##_t)a_norm_sq) *   \
                        compute_rsqrt((nk_##output_type##_t)b_norm_sq));                                   \
            *result = unclipped_distance > 0 ? unclipped_distance : 0;                                     \
        }                                                                                                  \
        return nk_success_k;                                                                               \
    }

/*  GCC inlines a helper only into callers whose targets include its own, so serial code builds at
 *  the Armv8-A floor. */
#if defined(__GNUC__) && !defined(__clang__) && NUMKONG_ARCH_ARM64_
#pragma GCC push_options
#pragma GCC target("arch=armv8-a")
#endif

nk_define_sqeuclidean_(f32, f64, f64, nk_assign_from_to_) // nk_sqeuclidean_f32_
nk_define_sqeuclidean_(f16, f32, f32, nk_f16_to_f32_)     // nk_sqeuclidean_f16_
nk_define_sqeuclidean_(bf16, f32, f32, nk_bf16_to_f32_)   // nk_sqeuclidean_bf16_
nk_define_sqeuclidean_(e4m3, f32, f32, nk_e4m3_to_f32_)   // nk_sqeuclidean_e4m3_
nk_define_sqeuclidean_(e5m2, f32, f32, nk_e5m2_to_f32_)   // nk_sqeuclidean_e5m2_
nk_define_sqeuclidean_(e2m3, f32, f32, nk_e2m3_to_f32_)   // nk_sqeuclidean_e2m3_
nk_define_sqeuclidean_(e3m2, f32, f32, nk_e3m2_to_f32_)   // nk_sqeuclidean_e3m2_
nk_define_sqeuclidean_(i8, i32, u32, nk_assign_from_to_)  // nk_sqeuclidean_i8_
nk_define_sqeuclidean_(u8, u32, u32, nk_assign_from_to_)  // nk_sqeuclidean_u8_

#undef nk_define_sqeuclidean_

/** Squared Euclidean distance between @p n F64 values, summed in Dot2. */
NUMKONG_INLINE void nk_sqeuclidean_f64_(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_f64_t sum = 0, compensation = 0;
    for (nk_size_t i = 0; i != n; ++i) nk_f64_dot2_(&sum, &compensation, a[i] - b[i], a[i] - b[i]);
    *result = sum + compensation;
}

/** Squared Euclidean distance between @p n packed I4 values, exact in I32. */
NUMKONG_INLINE void nk_sqeuclidean_i4_(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_u32_t *result) {
    nk_assert_dims_(n, nk_i4_k);
    // i4 values are packed as nibbles: two 4-bit signed values per byte.
    // Sign extension: (nibble ^ 8) - 8 maps [0,15] to [-8,7]
    nk_size_t const n_bytes = n / NUMKONG_NIBBLES_PER_BYTE;
    nk_i32_t sum = 0;
    for (nk_size_t i = 0; i < n_bytes; ++i) {
        nk_i32_t a_low = (nk_i32_t)nk_i4x2_low_(a[i]);
        nk_i32_t b_low = (nk_i32_t)nk_i4x2_low_(b[i]);
        nk_i32_t a_high = (nk_i32_t)nk_i4x2_high_(a[i]);
        nk_i32_t b_high = (nk_i32_t)nk_i4x2_high_(b[i]);
        nk_i32_t diff_low = a_low - b_low, diff_high = a_high - b_high;
        sum += diff_low * diff_low + diff_high * diff_high;
    }
    *result = (nk_u32_t)sum;
}

/** Squared Euclidean distance between @p n packed U4 values, exact in U32. */
NUMKONG_INLINE void nk_sqeuclidean_u4_(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result) {
    nk_assert_dims_(n, nk_u4_k);
    // u4 values are packed as nibbles: two 4-bit unsigned values per byte.
    // No sign extension needed - values are in [0,15].
    nk_size_t const n_bytes = n / NUMKONG_NIBBLES_PER_BYTE;
    nk_u32_t sum = 0;
    for (nk_size_t i = 0; i < n_bytes; ++i) {
        nk_i32_t a_low = (nk_i32_t)nk_u4x2_low_(a[i]);
        nk_i32_t b_low = (nk_i32_t)nk_u4x2_low_(b[i]);
        nk_i32_t a_high = (nk_i32_t)nk_u4x2_high_(a[i]);
        nk_i32_t b_high = (nk_i32_t)nk_u4x2_high_(b[i]);
        nk_i32_t diff_low = a_low - b_low, diff_high = a_high - b_high;
        sum += (nk_u32_t)(diff_low * diff_low + diff_high * diff_high);
    }
    *result = sum;
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

NUMKONG_API nk_status_t nk_angular_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                              void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f64_t dot_product = 0, a_norm_sq = 0, b_norm_sq = 0;
    nk_f64_t dot_compensation = 0, a_compensation = 0, b_compensation = 0;
    for (nk_size_t i = 0; i != n; ++i) {
        nk_f64_dot2_(&dot_product, &dot_compensation, a[i], b[i]);
        nk_f64_dot2_(&a_norm_sq, &a_compensation, a[i], a[i]);
        nk_f64_dot2_(&b_norm_sq, &b_compensation, b[i], b[i]);
    }
    dot_product += dot_compensation, a_norm_sq += a_compensation, b_norm_sq += b_compensation;
    if (a_norm_sq == 0 && b_norm_sq == 0) { *result = 0; }
    else if (dot_product == 0) { *result = 1; }
    else {
        nk_f64_t unclipped_distance = 1 - dot_product * nk_f64_rsqrt_(a_norm_sq) * nk_f64_rsqrt_(b_norm_sq);
        *result = unclipped_distance > 0 ? unclipped_distance : 0;
    }
    return nk_success_k;
}
nk_define_sqeuclidean_serial_(f64, f64)                                    // nk_sqeuclidean_f64_serial
nk_define_euclidean_(f64, f64, f64, f64, nk_assign_from_to_, nk_f64_sqrt_) // nk_euclidean_f64_serial

nk_define_angular_(f32, f64, f64, nk_assign_from_to_, nk_f64_rsqrt_)       // nk_angular_f32_serial
nk_define_sqeuclidean_serial_(f32, f64)                                    // nk_sqeuclidean_f32_serial
nk_define_euclidean_(f32, f64, f64, f64, nk_assign_from_to_, nk_f64_sqrt_) // nk_euclidean_f32_serial

nk_define_angular_(f16, f32, f32, nk_f16_to_f32_, nk_f32_rsqrt_)       // nk_angular_f16_serial
nk_define_sqeuclidean_serial_(f16, f32)                                // nk_sqeuclidean_f16_serial
nk_define_euclidean_(f16, f32, f32, f32, nk_f16_to_f32_, nk_f32_sqrt_) // nk_euclidean_f16_serial

nk_define_angular_(bf16, f32, f32, nk_bf16_to_f32_, nk_f32_rsqrt_)       // nk_angular_bf16_serial
nk_define_sqeuclidean_serial_(bf16, f32)                                 // nk_sqeuclidean_bf16_serial
nk_define_euclidean_(bf16, f32, f32, f32, nk_bf16_to_f32_, nk_f32_sqrt_) // nk_euclidean_bf16_serial

nk_define_angular_(e4m3, f32, f32, nk_e4m3_to_f32_, nk_f32_rsqrt_)       // nk_angular_e4m3_serial
nk_define_sqeuclidean_serial_(e4m3, f32)                                 // nk_sqeuclidean_e4m3_serial
nk_define_euclidean_(e4m3, f32, f32, f32, nk_e4m3_to_f32_, nk_f32_sqrt_) // nk_euclidean_e4m3_serial

nk_define_angular_(e5m2, f32, f32, nk_e5m2_to_f32_, nk_f32_rsqrt_)       // nk_angular_e5m2_serial
nk_define_sqeuclidean_serial_(e5m2, f32)                                 // nk_sqeuclidean_e5m2_serial
nk_define_euclidean_(e5m2, f32, f32, f32, nk_e5m2_to_f32_, nk_f32_sqrt_) // nk_euclidean_e5m2_serial

nk_define_angular_(e2m3, f32, f32, nk_e2m3_to_f32_, nk_f32_rsqrt_)       // nk_angular_e2m3_serial
nk_define_sqeuclidean_serial_(e2m3, f32)                                 // nk_sqeuclidean_e2m3_serial
nk_define_euclidean_(e2m3, f32, f32, f32, nk_e2m3_to_f32_, nk_f32_sqrt_) // nk_euclidean_e2m3_serial

nk_define_angular_(e3m2, f32, f32, nk_e3m2_to_f32_, nk_f32_rsqrt_)       // nk_angular_e3m2_serial
nk_define_sqeuclidean_serial_(e3m2, f32)                                 // nk_sqeuclidean_e3m2_serial
nk_define_euclidean_(e3m2, f32, f32, f32, nk_e3m2_to_f32_, nk_f32_sqrt_) // nk_euclidean_e3m2_serial

nk_define_angular_(i8, i32, f32, nk_assign_from_to_, nk_f32_rsqrt_)       // nk_angular_i8_serial
nk_define_sqeuclidean_serial_(i8, u32)                                    // nk_sqeuclidean_i8_serial
nk_define_euclidean_(i8, i32, u32, f32, nk_assign_from_to_, nk_f32_sqrt_) // nk_euclidean_i8_serial

nk_define_angular_(u8, u32, f32, nk_assign_from_to_, nk_f32_rsqrt_)       // nk_angular_u8_serial
nk_define_sqeuclidean_serial_(u8, u32)                                    // nk_sqeuclidean_u8_serial
nk_define_euclidean_(u8, u32, u32, f32, nk_assign_from_to_, nk_f32_sqrt_) // nk_euclidean_u8_serial

NUMKONG_API nk_status_t nk_sqeuclidean_i4_serial(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_u32_t *result,
                                                 void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_sqeuclidean_i4_(a, b, n, result);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclidean_i4_serial(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_f32_t *result,
                                               void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_u32_t distance_sq;
    nk_sqeuclidean_i4_(a, b, n, &distance_sq);
    *result = nk_f32_sqrt_((nk_f32_t)distance_sq);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angular_i4_serial(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_f32_t *result,
                                             void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_dims_(n, nk_i4_k);
    nk_size_t const n_bytes = n / NUMKONG_NIBBLES_PER_BYTE;
    nk_i32_t dot_sum = 0, a_norm_sq = 0, b_norm_sq = 0;
    for (nk_size_t i = 0; i < n_bytes; ++i) {
        nk_i32_t a_low = (nk_i32_t)nk_i4x2_low_(a[i]);
        nk_i32_t b_low = (nk_i32_t)nk_i4x2_low_(b[i]);
        nk_i32_t a_high = (nk_i32_t)nk_i4x2_high_(a[i]);
        nk_i32_t b_high = (nk_i32_t)nk_i4x2_high_(b[i]);
        dot_sum += a_low * b_low + a_high * b_high;
        a_norm_sq += a_low * a_low + a_high * a_high;
        b_norm_sq += b_low * b_low + b_high * b_high;
    }
    if (a_norm_sq == 0 && b_norm_sq == 0) { *result = 0; }
    else if (dot_sum == 0) { *result = 1; }
    else {
        nk_f32_t unclipped = 1.0f - (nk_f32_t)dot_sum * nk_f32_rsqrt_((nk_f32_t)a_norm_sq) *
                                        nk_f32_rsqrt_((nk_f32_t)b_norm_sq);
        *result = unclipped > 0 ? unclipped : 0;
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_sqeuclidean_u4_serial(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result,
                                                 void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_sqeuclidean_u4_(a, b, n, result);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclidean_u4_serial(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_f32_t *result,
                                               void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_u32_t distance_sq;
    nk_sqeuclidean_u4_(a, b, n, &distance_sq);
    *result = nk_f32_sqrt_((nk_f32_t)distance_sq);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angular_u4_serial(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_f32_t *result,
                                             void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_dims_(n, nk_u4_k);
    nk_size_t const n_bytes = n / NUMKONG_NIBBLES_PER_BYTE;
    nk_u32_t dot_sum = 0, a_norm_sq = 0, b_norm_sq = 0;
    for (nk_size_t i = 0; i < n_bytes; ++i) {
        nk_u32_t a_low = (nk_u32_t)nk_u4x2_low_(a[i]);
        nk_u32_t b_low = (nk_u32_t)nk_u4x2_low_(b[i]);
        nk_u32_t a_high = (nk_u32_t)nk_u4x2_high_(a[i]);
        nk_u32_t b_high = (nk_u32_t)nk_u4x2_high_(b[i]);
        dot_sum += a_low * b_low + a_high * b_high;
        a_norm_sq += a_low * a_low + a_high * a_high;
        b_norm_sq += b_low * b_low + b_high * b_high;
    }
    if (a_norm_sq == 0 && b_norm_sq == 0) { *result = 0; }
    else if (dot_sum == 0) { *result = 1; }
    else {
        nk_f32_t unclipped = 1.0f - (nk_f32_t)dot_sum * nk_f32_rsqrt_((nk_f32_t)a_norm_sq) *
                                        nk_f32_rsqrt_((nk_f32_t)b_norm_sq);
        *result = unclipped > 0 ? unclipped : 0;
    }
    return nk_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // NUMKONG_TARGET_SERIAL

#undef nk_define_sqeuclidean_serial_
#undef nk_define_euclidean_
#undef nk_define_angular_

/** Angular from_dot: computes 1 − dot × rsqrt(q) × rsqrt(t) for 4 pairs (serial), where q is
 *  @p query_sumsq and t each target's sum of squares. Separate reciprocal square roots avoid
 *  overflowing the product of two finite-but-large norms. */
NUMKONG_INLINE void nk_angular_through_f32_from_dot_serial_(nk_b128_vec_t const *dots_vec, nk_f32_t query_sumsq,
                                                            nk_b128_vec_t const *target_sumsqs_vec,
                                                            nk_b128_vec_t *result_vec) {
    nk_f32_t query_rsqrt = query_sumsq > 0 ? nk_f32_rsqrt_(query_sumsq) : 0.0f;
    for (int i = 0; i < 4; ++i) {
        nk_f32_t target_sumsq = target_sumsqs_vec->f32s[i];
        if (query_sumsq > 0 && target_sumsq > 0) {
            nk_f32_t rsqrt_val = query_rsqrt * nk_f32_rsqrt_(target_sumsq);
            nk_f32_t normalized = dots_vec->f32s[i] * rsqrt_val;
            nk_f32_t result = 1.0f - normalized;
            result_vec->f32s[i] = result > 0 ? result : 0;
        }
        else { result_vec->f32s[i] = (dots_vec->f32s[i] == 0) ? 0.0f : 1.0f; }
    }
}

/** Euclidean from_dot: computes √(q + t − 2 × dot) for 4 pairs (serial), where q is @p query_sumsq
 *  and t each target's sum of squares. */
NUMKONG_INLINE void nk_euclidean_through_f32_from_dot_serial_(nk_b128_vec_t const *dots_vec, nk_f32_t query_sumsq,
                                                              nk_b128_vec_t const *target_sumsqs_vec,
                                                              nk_b128_vec_t *result_vec) {
    for (int i = 0; i < 4; ++i) {
        nk_f32_t dist_sq = query_sumsq + target_sumsqs_vec->f32s[i] - 2.0f * dots_vec->f32s[i];
        result_vec->f32s[i] = dist_sq > 0 ? nk_f32_sqrt_(dist_sq) : 0.0f;
    }
}

/** Angular from_dot for f64 precision. Separate rsqrts avoid the product overflowing. */
NUMKONG_INLINE void nk_angular_through_f64_from_dot_serial_(nk_b256_vec_t const *dots_vec, nk_f64_t query_sumsq,
                                                            nk_b256_vec_t const *target_sumsqs_vec,
                                                            nk_b256_vec_t *result_vec) {
    nk_f64_t query_rsqrt = query_sumsq > 0 ? nk_f64_rsqrt_(query_sumsq) : 0.0;
    for (int i = 0; i < 4; ++i) {
        nk_f64_t target_sumsq = target_sumsqs_vec->f64s[i];
        if (query_sumsq > 0 && target_sumsq > 0) {
            nk_f64_t rsqrt_val = query_rsqrt * nk_f64_rsqrt_(target_sumsq);
            nk_f64_t normalized = dots_vec->f64s[i] * rsqrt_val;
            nk_f64_t result = 1.0 - normalized;
            result_vec->f64s[i] = result > 0 ? result : 0;
        }
        else { result_vec->f64s[i] = (dots_vec->f64s[i] == 0) ? 0.0 : 1.0; }
    }
}

/** Euclidean from_dot for f64 precision. */
NUMKONG_INLINE void nk_euclidean_through_f64_from_dot_serial_(nk_b256_vec_t const *dots_vec, nk_f64_t query_sumsq,
                                                              nk_b256_vec_t const *target_sumsqs_vec,
                                                              nk_b256_vec_t *result_vec) {
    for (int i = 0; i < 4; ++i) {
        nk_f64_t dist_sq = query_sumsq + target_sumsqs_vec->f64s[i] - 2.0 * dots_vec->f64s[i];
        result_vec->f64s[i] = dist_sq > 0 ? nk_f64_sqrt_(dist_sq) : 0.0;
    }
}

/** Angular from_dot for i32 accumulators: cast to f32, then same math as f32 variant. */
NUMKONG_INLINE void nk_angular_through_i32_from_dot_serial_(nk_b128_vec_t const *dots_vec, nk_i32_t query_sumsq,
                                                            nk_b128_vec_t const *target_sumsqs_vec,
                                                            nk_b128_vec_t *result_vec) {
    nk_f32_t query_rsqrt = query_sumsq > 0 ? nk_f32_rsqrt_((nk_f32_t)query_sumsq) : 0.0f;
    for (int i = 0; i < 4; ++i) {
        nk_i32_t target_sumsq = target_sumsqs_vec->i32s[i];
        if (query_sumsq > 0 && target_sumsq > 0) {
            nk_f32_t rsqrt_val = query_rsqrt * nk_f32_rsqrt_((nk_f32_t)target_sumsq);
            nk_f32_t normalized = (nk_f32_t)dots_vec->i32s[i] * rsqrt_val;
            nk_f32_t result = 1.0f - normalized;
            result_vec->f32s[i] = result > 0 ? result : 0;
        }
        else { result_vec->f32s[i] = (dots_vec->i32s[i] == 0) ? 0.0f : 1.0f; }
    }
}

/** Euclidean from_dot for i32 accumulators: cast to f32, then same math as f32 variant. */
NUMKONG_INLINE void nk_euclidean_through_i32_from_dot_serial_(nk_b128_vec_t const *dots_vec, nk_i32_t query_sumsq,
                                                              nk_b128_vec_t const *target_sumsqs_vec,
                                                              nk_b128_vec_t *result_vec) {
    for (int i = 0; i < 4; ++i) {
        nk_f32_t dist_sq = (nk_f32_t)query_sumsq + (nk_f32_t)target_sumsqs_vec->i32s[i] -
                           2.0f * (nk_f32_t)dots_vec->i32s[i];
        result_vec->f32s[i] = dist_sq > 0 ? nk_f32_sqrt_(dist_sq) : 0.0f;
    }
}

/** Angular from_dot for u32 accumulators: cast to f32, then same math as f32 variant. */
NUMKONG_INLINE void nk_angular_through_u32_from_dot_serial_(nk_b128_vec_t const *dots_vec, nk_u32_t query_sumsq,
                                                            nk_b128_vec_t const *target_sumsqs_vec,
                                                            nk_b128_vec_t *result_vec) {
    nk_f32_t query_rsqrt = query_sumsq > 0 ? nk_f32_rsqrt_((nk_f32_t)query_sumsq) : 0.0f;
    for (int i = 0; i < 4; ++i) {
        nk_u32_t target_sumsq = target_sumsqs_vec->u32s[i];
        if (query_sumsq > 0 && target_sumsq > 0) {
            nk_f32_t rsqrt_val = query_rsqrt * nk_f32_rsqrt_((nk_f32_t)target_sumsq);
            nk_f32_t normalized = (nk_f32_t)dots_vec->u32s[i] * rsqrt_val;
            nk_f32_t result = 1.0f - normalized;
            result_vec->f32s[i] = result > 0 ? result : 0;
        }
        else { result_vec->f32s[i] = (dots_vec->u32s[i] == 0) ? 0.0f : 1.0f; }
    }
}

/** Euclidean from_dot for u32 accumulators: cast to f32, then same math as f32 variant. */
NUMKONG_INLINE void nk_euclidean_through_u32_from_dot_serial_(nk_b128_vec_t const *dots_vec, nk_u32_t query_sumsq,
                                                              nk_b128_vec_t const *target_sumsqs_vec,
                                                              nk_b128_vec_t *result_vec) {
    for (int i = 0; i < 4; ++i) {
        nk_f32_t dist_sq = (nk_f32_t)query_sumsq + (nk_f32_t)target_sumsqs_vec->u32s[i] -
                           2.0f * (nk_f32_t)dots_vec->u32s[i];
        result_vec->f32s[i] = dist_sq > 0 ? nk_f32_sqrt_(dist_sq) : 0.0f;
    }
}

#if defined(__GNUC__) && !defined(__clang__) && NUMKONG_ARCH_ARM64_
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_SPATIAL_SERIAL_H
