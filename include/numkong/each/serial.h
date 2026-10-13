/**
 *  @file include/numkong/each/serial.h
 *  @author Ash Vardanian
 *  @date October 18, 2024
 *  @brief SWAR-accelerated elementwise arithmetic for SIMD-free CPUs.
 *
 *  @sa include/numkong/each.h
 */
#ifndef NUMKONG_EACH_SERIAL_H
#define NUMKONG_EACH_SERIAL_H

#include "numkong/types.h"
#include "numkong/cast/serial.h"   // `nk_f16_to_f32_serial_`
#include "numkong/reduce/serial.h" // `nk_reduce_moments_strided_f32_serial_`
#include "numkong/scalar/serial.h" // `nk_f32_silu_serial_`

#if defined(__cplusplus)
extern "C" {
#endif

#define nk_define_each_scale_(input_type, accumulator_type, load_and_convert, convert_and_store)  \
    NUMKONG_API nk_status_t nk_each_scale_##input_type##_serial(                                  \
        nk_##input_type##_t const *a, nk_size_t n, nk_##accumulator_type##_t const *alpha,        \
        nk_##accumulator_type##_t const *beta, nk_##input_type##_t *result, nk_stream_t stream) { \
        nk_assert_(stream == NUMKONG_NULL);                                                       \
        nk_##accumulator_type##_t alpha_val = *alpha;                                             \
        nk_##accumulator_type##_t beta_val = *beta;                                               \
        nk_##accumulator_type##_t ai, sum;                                                        \
        for (nk_size_t i = 0; i != n; ++i) {                                                      \
            load_and_convert(a + i, &ai);                                                         \
            sum = (nk_##accumulator_type##_t)(alpha_val * ai + beta_val);                         \
            convert_and_store(&sum, result + i);                                                  \
        }                                                                                         \
        return nk_success_k;                                                                      \
    }
#define nk_define_each_sum_(input_type, accumulator_type, load_and_convert, convert_and_store)                   \
    NUMKONG_API nk_status_t nk_each_sum_##input_type##_serial(nk_##input_type##_t const *a,                      \
                                                              nk_##input_type##_t const *b, nk_size_t n,         \
                                                              nk_##input_type##_t *result, nk_stream_t stream) { \
        nk_assert_(stream == NUMKONG_NULL);                                                                      \
        nk_##accumulator_type##_t ai, bi, sum;                                                                   \
        for (nk_size_t i = 0; i != n; ++i) {                                                                     \
            load_and_convert(a + i, &ai);                                                                        \
            load_and_convert(b + i, &bi);                                                                        \
            sum = ai + bi;                                                                                       \
            convert_and_store(&sum, result + i);                                                                 \
        }                                                                                                        \
        return nk_success_k;                                                                                     \
    }

#define nk_define_each_blend_(input_type, accumulator_type, load_and_convert, convert_and_store)                    \
    NUMKONG_API nk_status_t nk_each_blend_##input_type##_serial(                                                    \
        nk_##input_type##_t const *a, nk_##input_type##_t const *b, nk_size_t n,                                    \
        nk_##accumulator_type##_t const *alpha, nk_##accumulator_type##_t const *beta, nk_##input_type##_t *result, \
        nk_stream_t stream) {                                                                                       \
        nk_assert_(stream == NUMKONG_NULL);                                                                         \
        nk_##accumulator_type##_t alpha_val = *alpha;                                                               \
        nk_##accumulator_type##_t beta_val = *beta;                                                                 \
        nk_##accumulator_type##_t ai, bi, ai_scaled, bi_scaled, sum;                                                \
        for (nk_size_t i = 0; i != n; ++i) {                                                                        \
            load_and_convert(a + i, &ai);                                                                           \
            load_and_convert(b + i, &bi);                                                                           \
            ai_scaled = ai * alpha_val;                                                                             \
            bi_scaled = bi * beta_val;                                                                              \
            sum = ai_scaled + bi_scaled;                                                                            \
            convert_and_store(&sum, result + i);                                                                    \
        }                                                                                                           \
        return nk_success_k;                                                                                        \
    }

#define nk_define_each_fma_(input_type, accumulator_type, load_and_convert, convert_and_store)                      \
    NUMKONG_API nk_status_t nk_each_fma_##input_type##_serial(                                                      \
        nk_##input_type##_t const *a, nk_##input_type##_t const *b, nk_##input_type##_t const *c, nk_size_t n,      \
        nk_##accumulator_type##_t const *alpha, nk_##accumulator_type##_t const *beta, nk_##input_type##_t *result, \
        nk_stream_t stream) {                                                                                       \
        nk_assert_(stream == NUMKONG_NULL);                                                                         \
        nk_##accumulator_type##_t alpha_val = *alpha;                                                               \
        nk_##accumulator_type##_t beta_val = *beta;                                                                 \
        nk_##accumulator_type##_t ai, bi, ci, abi_scaled, ci_scaled, sum;                                           \
        for (nk_size_t i = 0; i != n; ++i) {                                                                        \
            load_and_convert(a + i, &ai);                                                                           \
            load_and_convert(b + i, &bi);                                                                           \
            load_and_convert(c + i, &ci);                                                                           \
            abi_scaled = ai * bi * alpha_val;                                                                       \
            ci_scaled = ci * beta_val;                                                                              \
            sum = abi_scaled + ci_scaled;                                                                           \
            convert_and_store(&sum, result + i);                                                                    \
        }                                                                                                           \
        return nk_success_k;                                                                                        \
    }

/** The F64 inverse RMS of @p count F64 values whose squares sum to @p sumsq, after adding
 *  @p epsilon to their mean square. Rows whose squares leave the normal range sum them again,
 *  scaled by the power of two of their largest magnitude. */
NUMKONG_INLINE nk_f64_t nk_each_inverse_rms_f64_serial_(nk_f64_t const *data, nk_size_t count, nk_f64_t sumsq,
                                                        nk_f64_t epsilon) {
    if (sumsq >= 0x1p-968 && sumsq <= 0x1p968) return nk_rsqrt_f64_serial_(sumsq / (nk_f64_t)count + epsilon);
    nk_f64_t largest = 0;
    for (nk_size_t i = 0; i < count; ++i) {
        nk_f64_t const magnitude = nk_f64_abs_(data[i]);
        if (magnitude != magnitude) return magnitude;
        if (magnitude > largest) largest = magnitude;
    }
    if (largest == 0 || largest == NUMKONG_F64_INF) return nk_rsqrt_f64_serial_(largest + epsilon);
    nk_fui64_t bits, scale;
    bits.f = largest;
    int exponent = (int)((bits.u >> 52) & 0x7FFu) - 1023;
    exponent = exponent < -1022 ? -1022 : exponent > 1022 ? 1022 : exponent;
    scale.u = (nk_u64_t)(1023 - exponent) << 52;
    nk_f64_t scaled_sumsq = 0;
    for (nk_size_t i = 0; i < count; ++i) {
        nk_f64_t const value = data[i] * scale.f;
        scaled_sumsq += value * value;
    }
    // Epsilon scales by the square of the same power, past the F64 range only where it dominates
    nk_f64_t const scaled_epsilon = epsilon * scale.f * scale.f;
    if (scaled_epsilon == NUMKONG_F64_INF) return nk_rsqrt_f64_serial_(epsilon);
    return nk_rsqrt_f64_serial_(scaled_sumsq / (nk_f64_t)count + scaled_epsilon) * scale.f;
}

/** Splits the F64 @p inverse_rms of an F32 row into a power of two @p prescale for its values and
 *  the F32 @p factor rounded from the rest within 2⁶⁴ of one, so their F32 products scale any
 *  finite row without overflowing or flushing. The @p prescale stays one unless the inverse leaves
 *  the range of 2⁻⁶⁴ to 2⁶⁴. */
NUMKONG_INLINE void nk_each_prescale_f32_serial_(nk_f64_t inverse_rms, nk_f32_t *prescale, nk_f32_t *factor) {
    nk_fui64_t bits, power, remainder;
    bits.f = inverse_rms;
    int const exponent = (int)((bits.u >> 52) & 0x7FFu) - 1023;
    int const shift = exponent > 64 ? exponent - 64 : exponent < -64 ? exponent + 64 : 0;
    power.u = (nk_u64_t)(1023 + shift) << 52, remainder.u = (nk_u64_t)(1023 - shift) << 52;
    *prescale = (nk_f32_t)power.f, *factor = (nk_f32_t)(inverse_rms * remainder.f);
}

/** Everything one RMSNorm launch on the SIMT cores shares, passed by value as the kernels' only
 *  argument. Host code fills it, so a caller launching the kernels through its own loader needs no
 *  GPU header to plan one. */
typedef struct {
    unsigned char const *x;
    nk_f32_t const *gamma;
    unsigned char *y;
    nk_size_t groups;
    nk_size_t columns;
    nk_size_t vectors;
    nk_size_t x_stride;
    nk_size_t y_stride;
    nk_f32_t epsilon;
} nk_each_rmsnorm_arguments_t;

#if NUMKONG_TARGET_SERIAL
/*  Keep the serial instantiations below actually scalar, regardless of build type. Without this,
 *  -O3 + LTO can vectorize or clone the serial kernels under AVX-512 callers in dispatch_*.c, which
 *  wastes binary and breaks the nk_*_serial-as-scalar-oracle contract. See dots/serial.h. */
#if defined(__clang__)
#pragma clang attribute push(__attribute__((noinline)), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-slp-vectorize", "no-ipa-cp-clone", "no-inline")
#endif

/* Size bias for release. Gated on NDEBUG so Debug builds keep -O0 for stepping. */
#if defined(NDEBUG)
#if defined(_MSC_VER)
#pragma optimize("s", on)
#elif defined(__clang__)
#pragma clang attribute push(__attribute__((minsize)), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("Os")
#endif
#endif

nk_define_each_sum_(f64, f64, nk_assign_from_to_, nk_assign_from_to_)    // nk_each_sum_f64_serial
nk_define_each_sum_(f32, f32, nk_assign_from_to_, nk_assign_from_to_)    // nk_each_sum_f32_serial
nk_define_each_sum_(f16, f32, nk_f16_to_f32_serial_, nk_f32_to_f16_serial_)    // nk_each_sum_f16_serial
nk_define_each_sum_(bf16, f32, nk_bf16_to_f32_serial_, nk_f32_to_bf16_serial_) // nk_each_sum_bf16_serial
nk_define_each_sum_(e4m3, f32, nk_e4m3_to_f32_serial_, nk_f32_to_e4m3_serial_) // nk_each_sum_e4m3_serial
nk_define_each_sum_(e5m2, f32, nk_e5m2_to_f32_serial_, nk_f32_to_e5m2_serial_) // nk_each_sum_e5m2_serial
nk_define_each_sum_(e2m3, f32, nk_e2m3_to_f32_serial_, nk_f32_to_e2m3_serial_) // nk_each_sum_e2m3_serial
nk_define_each_sum_(e3m2, f32, nk_e3m2_to_f32_serial_, nk_f32_to_e3m2_serial_) // nk_each_sum_e3m2_serial
nk_define_each_sum_(i8, i64, nk_assign_from_to_, nk_i64_to_i8_serial_)   // nk_each_sum_i8_serial
nk_define_each_sum_(u8, i64, nk_assign_from_to_, nk_i64_to_u8_serial_)   // nk_each_sum_u8_serial
nk_define_each_sum_(i16, i64, nk_assign_from_to_, nk_i64_to_i16_serial_) // nk_each_sum_i16_serial
nk_define_each_sum_(u16, i64, nk_assign_from_to_, nk_i64_to_u16_serial_) // nk_each_sum_u16_serial
nk_define_each_sum_(i32, i64, nk_assign_from_to_, nk_i64_to_i32_serial_) // nk_each_sum_i32_serial
nk_define_each_sum_(u32, i64, nk_assign_from_to_, nk_i64_to_u32_serial_) // nk_each_sum_u32_serial

NUMKONG_API nk_status_t nk_each_sum_i64_serial(nk_i64_t const *a, nk_i64_t const *b, nk_size_t n, nk_i64_t *result,
                                               nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    for (nk_size_t i = 0; i != n; ++i) result[i] = nk_saturating_add_i64_serial_(a[i], b[i]);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_each_sum_u64_serial(nk_u64_t const *a, nk_u64_t const *b, nk_size_t n, nk_u64_t *result,
                                               nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    for (nk_size_t i = 0; i != n; ++i) result[i] = nk_saturating_add_u64_serial_(a[i], b[i]);
    return nk_success_k;
}

nk_define_each_scale_(f64, f64, nk_assign_from_to_, nk_assign_from_to_)    // nk_each_scale_f64_serial
nk_define_each_scale_(f32, f32, nk_assign_from_to_, nk_assign_from_to_)    // nk_each_scale_f32_serial
nk_define_each_scale_(f16, f32, nk_f16_to_f32_serial_, nk_f32_to_f16_serial_)    // nk_each_scale_f16_serial
nk_define_each_scale_(bf16, f32, nk_bf16_to_f32_serial_, nk_f32_to_bf16_serial_) // nk_each_scale_bf16_serial
nk_define_each_scale_(e4m3, f32, nk_e4m3_to_f32_serial_, nk_f32_to_e4m3_serial_) // nk_each_scale_e4m3_serial
nk_define_each_scale_(e5m2, f32, nk_e5m2_to_f32_serial_, nk_f32_to_e5m2_serial_) // nk_each_scale_e5m2_serial
nk_define_each_scale_(e2m3, f32, nk_e2m3_to_f32_serial_, nk_f32_to_e2m3_serial_) // nk_each_scale_e2m3_serial
nk_define_each_scale_(e3m2, f32, nk_e3m2_to_f32_serial_, nk_f32_to_e3m2_serial_) // nk_each_scale_e3m2_serial
nk_define_each_scale_(i8, f32, nk_assign_from_to_, nk_f32_to_i8_serial_)   // nk_each_scale_i8_serial
nk_define_each_scale_(u8, f32, nk_assign_from_to_, nk_f32_to_u8_serial_)   // nk_each_scale_u8_serial
nk_define_each_scale_(i16, f32, nk_assign_from_to_, nk_f32_to_i16_serial_) // nk_each_scale_i16_serial
nk_define_each_scale_(u16, f32, nk_assign_from_to_, nk_f32_to_u16_serial_) // nk_each_scale_u16_serial
nk_define_each_scale_(i32, f64, nk_assign_from_to_, nk_f64_to_i32_serial_) // nk_each_scale_i32_serial
nk_define_each_scale_(u32, f64, nk_assign_from_to_, nk_f64_to_u32_serial_) // nk_each_scale_u32_serial
nk_define_each_scale_(i64, f64, nk_f64_from_i64_, nk_f64_to_i64_serial_)   // nk_each_scale_i64_serial
nk_define_each_scale_(u64, f64, nk_f64_from_u64_, nk_f64_to_u64_serial_)   // nk_each_scale_u64_serial

nk_define_each_blend_(f64, f64, nk_assign_from_to_, nk_assign_from_to_)    // nk_each_blend_f64_serial
nk_define_each_blend_(f32, f32, nk_assign_from_to_, nk_assign_from_to_)    // nk_each_blend_f32_serial
nk_define_each_blend_(f16, f32, nk_f16_to_f32_serial_, nk_f32_to_f16_serial_)    // nk_each_blend_f16_serial
nk_define_each_blend_(bf16, f32, nk_bf16_to_f32_serial_, nk_f32_to_bf16_serial_) // nk_each_blend_bf16_serial
nk_define_each_blend_(e4m3, f32, nk_e4m3_to_f32_serial_, nk_f32_to_e4m3_serial_) // nk_each_blend_e4m3_serial
nk_define_each_blend_(e5m2, f32, nk_e5m2_to_f32_serial_, nk_f32_to_e5m2_serial_) // nk_each_blend_e5m2_serial
nk_define_each_blend_(e2m3, f32, nk_e2m3_to_f32_serial_, nk_f32_to_e2m3_serial_) // nk_each_blend_e2m3_serial
nk_define_each_blend_(e3m2, f32, nk_e3m2_to_f32_serial_, nk_f32_to_e3m2_serial_) // nk_each_blend_e3m2_serial
nk_define_each_blend_(i8, f32, nk_assign_from_to_, nk_f32_to_i8_serial_)   // nk_each_blend_i8_serial
nk_define_each_blend_(u8, f32, nk_assign_from_to_, nk_f32_to_u8_serial_)   // nk_each_blend_u8_serial
nk_define_each_blend_(i16, f32, nk_assign_from_to_, nk_f32_to_i16_serial_) // nk_each_blend_i16_serial
nk_define_each_blend_(u16, f32, nk_assign_from_to_, nk_f32_to_u16_serial_) // nk_each_blend_u16_serial
nk_define_each_blend_(i32, f64, nk_assign_from_to_, nk_f64_to_i32_serial_) // nk_each_blend_i32_serial
nk_define_each_blend_(u32, f64, nk_assign_from_to_, nk_f64_to_u32_serial_) // nk_each_blend_u32_serial
nk_define_each_blend_(i64, f64, nk_f64_from_i64_, nk_f64_to_i64_serial_)   // nk_each_blend_i64_serial
nk_define_each_blend_(u64, f64, nk_f64_from_u64_, nk_f64_to_u64_serial_)   // nk_each_blend_u64_serial

nk_define_each_fma_(f64, f64, nk_assign_from_to_, nk_assign_from_to_)    // nk_each_fma_f64_serial
nk_define_each_fma_(f32, f32, nk_assign_from_to_, nk_assign_from_to_)    // nk_each_fma_f32_serial
nk_define_each_fma_(f16, f32, nk_f16_to_f32_serial_, nk_f32_to_f16_serial_)    // nk_each_fma_f16_serial
nk_define_each_fma_(bf16, f32, nk_bf16_to_f32_serial_, nk_f32_to_bf16_serial_) // nk_each_fma_bf16_serial
nk_define_each_fma_(e4m3, f32, nk_e4m3_to_f32_serial_, nk_f32_to_e4m3_serial_) // nk_each_fma_e4m3_serial
nk_define_each_fma_(e5m2, f32, nk_e5m2_to_f32_serial_, nk_f32_to_e5m2_serial_) // nk_each_fma_e5m2_serial
nk_define_each_fma_(e2m3, f32, nk_e2m3_to_f32_serial_, nk_f32_to_e2m3_serial_) // nk_each_fma_e2m3_serial
nk_define_each_fma_(e3m2, f32, nk_e3m2_to_f32_serial_, nk_f32_to_e3m2_serial_) // nk_each_fma_e3m2_serial
nk_define_each_fma_(i8, f32, nk_assign_from_to_, nk_f32_to_i8_serial_)   // nk_each_fma_i8_serial
nk_define_each_fma_(u8, f32, nk_assign_from_to_, nk_f32_to_u8_serial_)   // nk_each_fma_u8_serial
nk_define_each_fma_(i16, f32, nk_assign_from_to_, nk_f32_to_i16_serial_) // nk_each_fma_i16_serial
nk_define_each_fma_(u16, f32, nk_assign_from_to_, nk_f32_to_u16_serial_) // nk_each_fma_u16_serial
nk_define_each_fma_(i32, f64, nk_assign_from_to_, nk_f64_to_i32_serial_) // nk_each_fma_i32_serial
nk_define_each_fma_(u32, f64, nk_assign_from_to_, nk_f64_to_u32_serial_) // nk_each_fma_u32_serial
nk_define_each_fma_(i64, f64, nk_f64_from_i64_, nk_f64_to_i64_serial_)   // nk_each_fma_i64_serial
nk_define_each_fma_(u64, f64, nk_f64_from_u64_, nk_f64_to_u64_serial_)   // nk_each_fma_u64_serial

#undef nk_define_each_scale_
#undef nk_define_each_sum_
#undef nk_define_each_blend_
#undef nk_define_each_fma_

NUMKONG_API nk_status_t nk_each_sum_f32c_serial(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f32c_t *result,
                                                nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f32_t const *a_scalars = (nk_f32_t const *)a, *b_scalars = (nk_f32_t const *)b;
    nk_f32_t *result_scalars = (nk_f32_t *)result;
    for (nk_size_t i = 0; i != 2 * n; ++i) result_scalars[i] = a_scalars[i] + b_scalars[i];
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_each_sum_f64c_serial(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result,
                                                nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f64_t const *a_scalars = (nk_f64_t const *)a, *b_scalars = (nk_f64_t const *)b;
    nk_f64_t *result_scalars = (nk_f64_t *)result;
    for (nk_size_t i = 0; i != 2 * n; ++i) result_scalars[i] = a_scalars[i] + b_scalars[i];
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_each_scale_f32c_serial(nk_f32c_t const *a, nk_size_t n, nk_f32c_t const *alpha,
                                                  nk_f32c_t const *beta, nk_f32c_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f32_t alpha_real = alpha->real, alpha_imag = alpha->imag;
    nk_f32_t beta_real = beta->real, beta_imag = beta->imag;
    for (nk_size_t i = 0; i != n; ++i) {
        nk_f32_t a_real = a[i].real, a_imag = a[i].imag;
        result[i].real = alpha_real * a_real - alpha_imag * a_imag + beta_real;
        result[i].imag = alpha_real * a_imag + alpha_imag * a_real + beta_imag;
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_each_scale_f64c_serial(nk_f64c_t const *a, nk_size_t n, nk_f64c_t const *alpha,
                                                  nk_f64c_t const *beta, nk_f64c_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f64_t alpha_real = alpha->real, alpha_imag = alpha->imag;
    nk_f64_t beta_real = beta->real, beta_imag = beta->imag;
    for (nk_size_t i = 0; i != n; ++i) {
        nk_f64_t a_real = a[i].real, a_imag = a[i].imag;
        result[i].real = alpha_real * a_real - alpha_imag * a_imag + beta_real;
        result[i].imag = alpha_real * a_imag + alpha_imag * a_real + beta_imag;
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_each_blend_f32c_serial(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n,
                                                  nk_f32c_t const *alpha, nk_f32c_t const *beta, nk_f32c_t *result,
                                                  nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f32_t alpha_real = alpha->real, alpha_imag = alpha->imag;
    nk_f32_t beta_real = beta->real, beta_imag = beta->imag;
    for (nk_size_t i = 0; i != n; ++i) {
        nk_f32_t a_real = a[i].real, a_imag = a[i].imag;
        nk_f32_t b_real = b[i].real, b_imag = b[i].imag;
        nk_f32_t alpha_a_real = alpha_real * a_real - alpha_imag * a_imag;
        nk_f32_t alpha_a_imag = alpha_real * a_imag + alpha_imag * a_real;
        nk_f32_t beta_b_real = beta_real * b_real - beta_imag * b_imag;
        nk_f32_t beta_b_imag = beta_real * b_imag + beta_imag * b_real;
        result[i].real = alpha_a_real + beta_b_real;
        result[i].imag = alpha_a_imag + beta_b_imag;
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_each_blend_f64c_serial(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n,
                                                  nk_f64c_t const *alpha, nk_f64c_t const *beta, nk_f64c_t *result,
                                                  nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f64_t alpha_real = alpha->real, alpha_imag = alpha->imag;
    nk_f64_t beta_real = beta->real, beta_imag = beta->imag;
    for (nk_size_t i = 0; i != n; ++i) {
        nk_f64_t a_real = a[i].real, a_imag = a[i].imag;
        nk_f64_t b_real = b[i].real, b_imag = b[i].imag;
        nk_f64_t alpha_a_real = alpha_real * a_real - alpha_imag * a_imag;
        nk_f64_t alpha_a_imag = alpha_real * a_imag + alpha_imag * a_real;
        nk_f64_t beta_b_real = beta_real * b_real - beta_imag * b_imag;
        nk_f64_t beta_b_imag = beta_real * b_imag + beta_imag * b_real;
        result[i].real = alpha_a_real + beta_b_real;
        result[i].imag = alpha_a_imag + beta_b_imag;
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_each_fma_f32c_serial(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c, nk_size_t n,
                                                nk_f32c_t const *alpha, nk_f32c_t const *beta, nk_f32c_t *result,
                                                nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f32_t alpha_real = alpha->real, alpha_imag = alpha->imag;
    nk_f32_t beta_real = beta->real, beta_imag = beta->imag;
    for (nk_size_t i = 0; i != n; ++i) {
        nk_f32_t a_real = a[i].real, a_imag = a[i].imag;
        nk_f32_t b_real = b[i].real, b_imag = b[i].imag;
        nk_f32_t c_real = c[i].real, c_imag = c[i].imag;
        nk_f32_t product_real = a_real * b_real - a_imag * b_imag;
        nk_f32_t product_imag = a_real * b_imag + a_imag * b_real;
        nk_f32_t alpha_product_real = alpha_real * product_real - alpha_imag * product_imag;
        nk_f32_t alpha_product_imag = alpha_real * product_imag + alpha_imag * product_real;
        nk_f32_t beta_c_real = beta_real * c_real - beta_imag * c_imag;
        nk_f32_t beta_c_imag = beta_real * c_imag + beta_imag * c_real;
        result[i].real = alpha_product_real + beta_c_real;
        result[i].imag = alpha_product_imag + beta_c_imag;
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_each_fma_f64c_serial(nk_f64c_t const *a, nk_f64c_t const *b, nk_f64c_t const *c, nk_size_t n,
                                                nk_f64c_t const *alpha, nk_f64c_t const *beta, nk_f64c_t *result,
                                                nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f64_t alpha_real = alpha->real, alpha_imag = alpha->imag;
    nk_f64_t beta_real = beta->real, beta_imag = beta->imag;
    for (nk_size_t i = 0; i != n; ++i) {
        nk_f64_t a_real = a[i].real, a_imag = a[i].imag;
        nk_f64_t b_real = b[i].real, b_imag = b[i].imag;
        nk_f64_t c_real = c[i].real, c_imag = c[i].imag;
        nk_f64_t product_real = a_real * b_real - a_imag * b_imag;
        nk_f64_t product_imag = a_real * b_imag + a_imag * b_real;
        nk_f64_t alpha_product_real = alpha_real * product_real - alpha_imag * product_imag;
        nk_f64_t alpha_product_imag = alpha_real * product_imag + alpha_imag * product_real;
        nk_f64_t beta_c_real = beta_real * c_real - beta_imag * c_imag;
        nk_f64_t beta_c_imag = beta_real * c_imag + beta_imag * c_real;
        result[i].real = alpha_product_real + beta_c_real;
        result[i].imag = alpha_product_imag + beta_c_imag;
    }
    return nk_success_k;
}

#if defined(NDEBUG)
#if defined(_MSC_VER)
#pragma optimize("", on)
#elif defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif

/** SwiGLU: y = silu(gate_scale × gate) ⊙ up × output_scale, and a NULL @c up collapses it to plain
 *  SiLU, y = silu(gate_scale × gate) × output_scale. Separate gate and up pointers with their own
 *  byte row-strides let UForm's fused @b [rows,2×ffn] output pass gate = base and up = base + ffn,
 *  let other split-weight models fit without a v2, and let the FFN head's fc1 → SiLU → fc2 pass a
 *  NULL @c up. The scales fold an E4M3 descale and requantization. */
#define nk_define_each_swiglu_(input_type, load_and_convert, convert_and_store)                                 \
    NUMKONG_API nk_status_t nk_each_swiglu_##input_type##_serial(                                               \
        nk_##input_type##_t const *gate, nk_##input_type##_t const *up, nk_##input_type##_t *y, nk_size_t rows, \
        nk_size_t columns, nk_size_t gate_stride, nk_size_t up_stride, nk_size_t y_stride, nk_f32_t gate_scale, \
        nk_f32_t output_scale, nk_stream_t stream) {                                                            \
        nk_assert_(stream == NUMKONG_NULL);                                                                     \
        for (nk_size_t row = 0; row != rows; ++row) {                                                           \
            nk_##input_type##_t const *gate_row = /**/                                                          \
                (nk_##input_type##_t const *)((unsigned char const *)gate + row * gate_stride);                 \
            nk_##input_type##_t const *up_row = /**/                                                            \
                up ? (nk_##input_type##_t const *)((unsigned char const *)up + row * up_stride) : NUMKONG_NULL; \
            nk_##input_type##_t *y_row = /**/                                                                   \
                (nk_##input_type##_t *)((unsigned char *)y + row * y_stride);                                   \
            for (nk_size_t col = 0; col != columns; ++col) {                                                    \
                nk_f32_t gate_value;                                                                            \
                load_and_convert(gate_row + col, &gate_value);                                                  \
                nk_f32_t result = nk_f32_silu_serial_(gate_value * gate_scale);                                 \
                if (up_row) {                                                                                   \
                    nk_f32_t up_value;                                                                          \
                    load_and_convert(up_row + col, &up_value);                                                  \
                    result *= up_value;                                                                         \
                }                                                                                               \
                result *= output_scale;                                                                         \
                convert_and_store(&result, y_row + col);                                                        \
            }                                                                                                   \
        }                                                                                                       \
        return nk_success_k;                                                                                    \
    }

nk_define_each_swiglu_(e4m3, nk_e4m3_to_f32_serial_, nk_f32_to_e4m3_serial_)
nk_define_each_swiglu_(f16, nk_f16_to_f32_serial_, nk_f32_to_f16_serial_)
nk_define_each_swiglu_(bf16, nk_bf16_to_f32_serial_, nk_f32_to_bf16_serial_)
nk_define_each_swiglu_(f32, nk_assign_from_to_, nk_assign_from_to_)
#undef nk_define_each_swiglu_

/** RMSNorm: y = x × rsqrt(mean(x²) + epsilon) × γ, where a NULL γ means unit scale. Each row, with
 *  byte strides @c x_stride and @c y_stride, holds @c groups independent vectors of
 *  @c columns elements, each normalized separately. One group with a learned γ covers the pre,
 *  post, final and head norms, while groups = heads, columns = depth and a NULL γ give the in-place
 *  unit QK-norm over the strided sections of a fused @b [tokens,3×hidden] QKV buffer. Pass 1 sums
 *  squares in @p reduction_type, F32 for floats up to F32 and F64 for 32-bit integers, and rounds
 *  their mean square to F32, where epsilon adds and the inverse root is taken. Pass 2 widens every
 *  value to F32 and multiplies it there by the inverse RMS and γ, rounding once into the output.
 *  The @c rmscast verb reads what a dot product of the output type writes, F32 or a 32-bit integer,
 *  and rounds like the cast into that type. */
#define nk_define_each_rmsnorm_(verb, input_type, output_type, reduction_type, load_and_convert, convert_and_store)    \
    NUMKONG_API nk_status_t nk_each_##verb##_##output_type##_serial(                                                   \
        nk_##input_type##_t const *x, nk_f32_t const *gamma, nk_##output_type##_t *y, nk_size_t rows,                  \
        nk_size_t groups, nk_size_t columns, nk_size_t x_stride, nk_size_t y_stride, nk_f32_t epsilon,                 \
        nk_stream_t stream) {                                                                                          \
        nk_assert_(stream == NUMKONG_NULL);                                                                            \
        for (nk_size_t r = 0; r != rows; ++r) {                                                                        \
            nk_##input_type##_t const *x_row = (nk_##input_type##_t const *)((unsigned char const *)x + r * x_stride); \
            nk_##output_type##_t *y_row = (nk_##output_type##_t *)((unsigned char *)y + r * y_stride);                 \
            for (nk_size_t group = 0; group != groups; ++group) {                                                      \
                nk_##input_type##_t const *group_input = x_row + group * columns;                                      \
                nk_##output_type##_t *group_output = y_row + group * columns;                                          \
                reduction_type sumsq = 0;                                                                              \
                for (nk_size_t c = 0; c != columns; ++c) {                                                             \
                    reduction_type value;                                                                              \
                    load_and_convert(group_input + c, &value);                                                         \
                    sumsq += value * value;                                                                            \
                }                                                                                                      \
                nk_f32_t const inverse_rms = nk_rsqrt_f32_serial_((nk_f32_t)(sumsq / (reduction_type)columns) +        \
                                                                  epsilon);                                            \
                for (nk_size_t c = 0; c != columns; ++c) {                                                             \
                    nk_f32_t value;                                                                                    \
                    load_and_convert(group_input + c, &value);                                                         \
                    nk_f32_t gamma_value = gamma ? gamma[c] : 1.0f;                                                    \
                    nk_f32_t result = value * inverse_rms * gamma_value;                                               \
                    convert_and_store(&result, group_output + c);                                                      \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
        return nk_success_k;                                                                                           \
    }

nk_define_each_rmsnorm_(rmsnorm, e4m3, e4m3, nk_f32_t, nk_e4m3_to_f32_serial_, nk_f32_to_e4m3_serial_)
nk_define_each_rmsnorm_(rmsnorm, f16, f16, nk_f32_t, nk_f16_to_f32_serial_, nk_f32_to_f16_serial_)
nk_define_each_rmsnorm_(rmsnorm, bf16, bf16, nk_f32_t, nk_bf16_to_f32_serial_, nk_f32_to_bf16_serial_)
nk_define_each_rmsnorm_(rmscast, f32, bf16, nk_f32_t, nk_assign_from_to_, nk_f32_to_bf16_serial_)
nk_define_each_rmsnorm_(rmscast, f32, f16, nk_f32_t, nk_assign_from_to_, nk_f32_to_f16_serial_)
nk_define_each_rmsnorm_(rmscast, f32, e4m3, nk_f32_t, nk_assign_from_to_, nk_f32_to_e4m3_serial_)
nk_define_each_rmsnorm_(rmscast, f32, e5m2, nk_f32_t, nk_assign_from_to_, nk_f32_to_e5m2_serial_)
nk_define_each_rmsnorm_(rmscast, f32, e2m3, nk_f32_t, nk_assign_from_to_, nk_f32_to_e2m3_serial_)
nk_define_each_rmsnorm_(rmscast, f32, e3m2, nk_f32_t, nk_assign_from_to_, nk_f32_to_e3m2_serial_)
nk_define_each_rmsnorm_(rmscast, i32, i8, nk_f64_t, nk_assign_from_to_, nk_f32_to_i8_serial_)
nk_define_each_rmsnorm_(rmscast, u32, u8, nk_f64_t, nk_assign_from_to_, nk_f32_to_u8_serial_)
#undef nk_define_each_rmsnorm_

NUMKONG_API nk_status_t nk_each_rmsnorm_f32_serial(nk_f32_t const *x, nk_f32_t const *gamma, nk_f32_t *y,
                                                   nk_size_t rows, nk_size_t groups, nk_size_t columns,
                                                   nk_size_t x_stride, nk_size_t y_stride, nk_f32_t epsilon,
                                                   nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    for (nk_size_t r = 0; r != rows; ++r) {
        nk_f32_t const *x_row = (nk_f32_t const *)((unsigned char const *)x + r * x_stride);
        nk_f32_t *y_row = (nk_f32_t *)((unsigned char *)y + r * y_stride);
        for (nk_size_t group = 0; group != groups; ++group) {
            nk_f32_t const *group_input = x_row + group * columns;
            nk_f32_t *group_output = y_row + group * columns;
            nk_f64_t sum, sumsq;
            nk_reduce_moments_strided_f32_serial_(group_input, columns, sizeof(nk_f32_t), &sum, &sumsq);
            nk_f32_t prescale, factor;
            nk_each_prescale_f32_serial_(nk_rsqrt_f64_serial_(sumsq / (nk_f64_t)columns + epsilon), &prescale, &factor);
            for (nk_size_t c = 0; c != columns; ++c)
                group_output[c] = group_input[c] * prescale * factor * (gamma ? gamma[c] : 1.0f);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_each_rmscast_f32_serial(nk_f64_t const *x, nk_f32_t const *gamma, nk_f32_t *y,
                                                   nk_size_t rows, nk_size_t groups, nk_size_t columns,
                                                   nk_size_t x_stride, nk_size_t y_stride, nk_f32_t epsilon,
                                                   nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    for (nk_size_t r = 0; r != rows; ++r) {
        nk_f64_t const *x_row = (nk_f64_t const *)((unsigned char const *)x + r * x_stride);
        nk_f32_t *y_row = (nk_f32_t *)((unsigned char *)y + r * y_stride);
        for (nk_size_t group = 0; group != groups; ++group) {
            nk_f64_t const *group_input = x_row + group * columns;
            nk_f32_t *group_output = y_row + group * columns;
            nk_f64_t sumsq = 0;
            for (nk_size_t c = 0; c != columns; ++c) sumsq += group_input[c] * group_input[c];
            nk_f64_t const inverse_rms = nk_each_inverse_rms_f64_serial_(group_input, columns, sumsq, epsilon);
            for (nk_size_t c = 0; c != columns; ++c) {
                nk_f64_t const result = group_input[c] * inverse_rms * (gamma ? gamma[c] : 1.0);
                nk_f64_to_f32_serial_(&result, group_output + c);
            }
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

#endif // NUMKONG_EACH_SERIAL_H
