/**
 *  @file include/numkong/trigonometry/serial.h
 *  @author Ash Vardanian
 *  @date November 20, 2024
 *  @brief SWAR-accelerated trigonometric functions for SIMD-free CPUs.
 *
 *  @sa include/numkong/trigonometry.h
 *  @see https://sleef.org
 */
#ifndef NUMKONG_TRIGONOMETRY_SERIAL_H
#define NUMKONG_TRIGONOMETRY_SERIAL_H

#include "numkong/types.h"
#include "numkong/cast/serial.h"   // `nk_f16_to_f32_`
#include "numkong/scalar/serial.h" // `nk_f32_sin_`, `nk_f64_atan_`

#if defined(__cplusplus)
extern "C" {
#endif

#if NUMKONG_TARGET_SERIAL

NUMKONG_API nk_status_t nk_trig_sin_f32_serial(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    for (nk_size_t i = 0; i != n; ++i) outs[i] = nk_f32_sin_(ins[i]);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_trig_cos_f32_serial(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    for (nk_size_t i = 0; i != n; ++i) outs[i] = nk_f32_cos_(ins[i]);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_trig_atan_f32_serial(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    for (nk_size_t i = 0; i != n; ++i) outs[i] = nk_f32_atan_(ins[i]);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_trig_sin_f64_serial(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    for (nk_size_t i = 0; i != n; ++i) outs[i] = nk_f64_sin_(ins[i]);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_trig_cos_f64_serial(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    for (nk_size_t i = 0; i != n; ++i) outs[i] = nk_f64_cos_(ins[i]);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_trig_atan_f64_serial(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    for (nk_size_t i = 0; i != n; ++i) outs[i] = nk_f64_atan_(ins[i]);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_trig_sin_f16_serial(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    for (nk_size_t i = 0; i != n; ++i) {
        nk_f32_t angle_f32;
        nk_f16_to_f32_(&ins[i], &angle_f32);
        nk_f32_t const result_f32 = nk_f32_sin_(angle_f32);
        nk_f32_to_f16_(&result_f32, &outs[i]);
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_trig_cos_f16_serial(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    for (nk_size_t i = 0; i != n; ++i) {
        nk_f32_t angle_f32;
        nk_f16_to_f32_(&ins[i], &angle_f32);
        nk_f32_t const result_f32 = nk_f32_cos_(angle_f32);
        nk_f32_to_f16_(&result_f32, &outs[i]);
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_trig_atan_f16_serial(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    for (nk_size_t i = 0; i != n; ++i) {
        nk_f32_t value_f32;
        nk_f16_to_f32_(&ins[i], &value_f32);
        nk_f32_t const result_f32 = nk_f32_atan_(value_f32);
        nk_f32_to_f16_(&result_f32, &outs[i]);
    }
    return nk_success_k;
}
#endif // NUMKONG_TARGET_SERIAL

/** RoPE, the NeoX split-half rotary position embedding. Each row, a token with byte stride
 *  @c x_row_stride, holds @c heads heads of 2 × half_dim channels. Every pair @c i rotates channel
 *  @c i against its split-half partner i + half_dim by the per-token angle from the
 *  @b [rows,half_dim] cosine and sine grids, where row @c r starts at r × half_dim and is shared
 *  across heads, exactly a complex multiply by (cos, sin). The whole head is written, so the output
 *  @c y, with byte stride @c y_row_stride, may alias @c x for in-place rotation, and the caller
 *  bakes position lookup and M-RoPE axis assignment into the grids. @c input_scale folds an E4M3
 *  descale onto the load, and is 1.0 for BF16 and F32. */
#define nk_define_trig_rope_(input_type, load_and_convert, convert_and_store)                                         \
    NUMKONG_API nk_status_t nk_trig_rope_##input_type##_serial(                                                       \
        nk_##input_type##_t const *x, nk_##input_type##_t *y, nk_rope_angle_t const *cos, nk_rope_angle_t const *sin, \
        nk_size_t rows, nk_size_t heads, nk_size_t half_dim, nk_size_t x_row_stride, nk_size_t y_row_stride,          \
        nk_f32_t input_scale, void *stream) {                                                                         \
        nk_assert_(stream == NUMKONG_NULL);                                                                           \
        for (nk_size_t r = 0; r != rows; ++r) {                                                                       \
            nk_f32_t const *cos_row = cos + r * half_dim;                                                             \
            nk_f32_t const *sin_row = sin + r * half_dim;                                                             \
            nk_##input_type##_t const *x_row = (nk_##input_type##_t const *)((unsigned char const *)x +               \
                                                                             r * x_row_stride);                       \
            nk_##input_type##_t *y_row = (nk_##input_type##_t *)((unsigned char *)y + r * y_row_stride);              \
            for (nk_size_t h = 0; h != heads; ++h) {                                                                  \
                nk_##input_type##_t const *x_base = x_row + h * 2 * half_dim;                                         \
                nk_##input_type##_t *y_base = y_row + h * 2 * half_dim;                                               \
                for (nk_size_t i = 0; i != half_dim; ++i) {                                                           \
                    nk_f32_t low, high;                                                                               \
                    load_and_convert(x_base + i, &low);                                                               \
                    load_and_convert(x_base + i + half_dim, &high);                                                   \
                    low *= input_scale, high *= input_scale;                                                          \
                    nk_f32_t cosine = cos_row[i], sine = sin_row[i];                                                  \
                    nk_f32_t rotated_low = low * cosine - high * sine;                                                \
                    nk_f32_t rotated_high = low * sine + high * cosine;                                               \
                    convert_and_store(&rotated_low, y_base + i);                                                      \
                    convert_and_store(&rotated_high, y_base + i + half_dim);                                          \
                }                                                                                                     \
            }                                                                                                         \
        }                                                                                                             \
        return nk_success_k;                                                                                          \
    }

#if NUMKONG_TARGET_SERIAL
nk_define_trig_rope_(f32, nk_assign_from_to_, nk_assign_from_to_)
nk_define_trig_rope_(bf16, nk_bf16_to_f32_, nk_f32_to_bf16_)
nk_define_trig_rope_(e4m3, nk_e4m3_to_f32_, nk_f32_to_e4m3_)
#endif // NUMKONG_TARGET_SERIAL
#undef nk_define_trig_rope_

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TRIGONOMETRY_SERIAL_H
