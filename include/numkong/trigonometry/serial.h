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
#include "numkong/cast/serial.h"   // `nk_f16_to_f32_serial_`
#include "numkong/scalar/serial.h" // `nk_sin_f32_serial_`, `nk_atan_f64_serial_`

#if defined(__cplusplus)
extern "C" {
#endif

#if NUMKONG_TARGET_SERIAL

NUMKONG_API nk_status_t nk_trig_sin_f32_serial(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    for (nk_size_t i = 0; i != n; ++i) outs[i] = nk_sin_f32_serial_(ins[i]);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_trig_cos_f32_serial(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    for (nk_size_t i = 0; i != n; ++i) outs[i] = nk_cos_f32_serial_(ins[i]);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_trig_atan_f32_serial(nk_f32_t const *ins, nk_size_t n, nk_f32_t *outs, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    for (nk_size_t i = 0; i != n; ++i) outs[i] = nk_atan_f32_serial_(ins[i]);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_trig_sin_f64_serial(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    for (nk_size_t i = 0; i != n; ++i) outs[i] = nk_sin_f64_serial_(ins[i]);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_trig_cos_f64_serial(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    for (nk_size_t i = 0; i != n; ++i) outs[i] = nk_cos_f64_serial_(ins[i]);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_trig_atan_f64_serial(nk_f64_t const *ins, nk_size_t n, nk_f64_t *outs, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    for (nk_size_t i = 0; i != n; ++i) outs[i] = nk_atan_f64_serial_(ins[i]);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_trig_sin_f16_serial(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    for (nk_size_t i = 0; i != n; ++i) {
        nk_f32_t angle_f32;
        nk_f16_to_f32_serial_(&ins[i], &angle_f32);
        nk_f32_t const result_f32 = nk_f32_sin_for_f16_serial_(angle_f32);
        nk_f32_to_f16_serial_(&result_f32, &outs[i]);
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_trig_cos_f16_serial(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    for (nk_size_t i = 0; i != n; ++i) {
        nk_f32_t angle_f32;
        nk_f16_to_f32_serial_(&ins[i], &angle_f32);
        nk_f32_t const result_f32 = nk_f32_cos_for_f16_serial_(angle_f32);
        nk_f32_to_f16_serial_(&result_f32, &outs[i]);
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_trig_atan_f16_serial(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    for (nk_size_t i = 0; i != n; ++i) {
        nk_f32_t value_f32;
        nk_f16_to_f32_serial_(&ins[i], &value_f32);
        nk_f32_t const result_f32 = nk_f32_atan_for_f16_serial_(value_f32);
        nk_f32_to_f16_serial_(&result_f32, &outs[i]);
    }
    return nk_success_k;
}
#endif // NUMKONG_TARGET_SERIAL

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TRIGONOMETRY_SERIAL_H
