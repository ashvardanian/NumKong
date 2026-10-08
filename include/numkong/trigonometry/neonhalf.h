/**
 *  @file include/numkong/trigonometry/neonhalf.h
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief SIMD-accelerated trigonometric functions for NEON FP16.
 *
 *  @sa include/numkong/trigonometry.h
 *
 *  @section trigonometry_neonhalf_instructions ARM NEON FP16 Instructions (ARMv8.2-FP16)
 *
 *  @verbatim
 *  Intrinsic          Instruction
 *  vfmaq_f16          FMLA (V.8H, V.8H, V.8H)
 *  vfmsq_f16          FMLS (V.8H, V.8H, V.8H)
 *  vrndnq_f16         FRINTN (V.8H, V.8H)
 *  vdivq_f16          FDIV (V.8H, V.8H, V.8H)
 *  vcagtq_f16         FACGT (V.8H, V.8H, V.8H)
 *  vcvt_high_f16_f32  FCVTN2 (V.8H, V.4S)
 *  @endverbatim
 *
 *  Sine, cosine and arctangent evaluate in F16 on 8 lanes, with polynomials fitted for F16.
 *  Sine and cosine reduce in F16 while every lane stays within |x| ≤ 256, and in F32 otherwise.
 */
#ifndef NUMKONG_TRIGONOMETRY_NEONHALF_H
#define NUMKONG_TRIGONOMETRY_NEONHALF_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_TARGET_NEONHALF

#include "numkong/types.h"
#include "numkong/cast/serial.h" // `nk_partial_load_b16x8_serial_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16")
#endif

/** Sine of 8 F16 angles already reduced to about [-π/2, π/2], odd in the reduced angle. */
NUMKONG_INLINE float16x8_t nk_sin_reduced_f16x8_neonhalf_(float16x8_t reduced_f16x8) {
    // Degree-7 odd polynomial with a unit linear term, coefficients searched in F16 arithmetic
    float16x8_t const squared_f16x8 = vmulq_f16(reduced_f16x8, reduced_f16x8);
    float16x8_t polynomial_f16x8 = vfmaq_f16(vdupq_n_f16(+0.00824737548828125), squared_f16x8,
                                             vdupq_n_f16(-0.00018310546875));
    polynomial_f16x8 = vfmaq_f16(vdupq_n_f16(-0.16650390625), polynomial_f16x8, squared_f16x8);
    return vfmaq_f16(reduced_f16x8, vmulq_f16(reduced_f16x8, squared_f16x8), polynomial_f16x8);
}

/** Subtracts @p multiples_f16x8 of π from @p angles_f16x8, for F16 angles up to 256. */
NUMKONG_INLINE float16x8_t nk_reduce_pi_f16x8_neonhalf_(float16x8_t angles_f16x8, float16x8_t multiples_f16x8) {
    // π in three F16 parts, the last scaled by 2¹² to stay clear of F16 subnormals
    float16x8_t reduced_f16x8 = vfmsq_f16(angles_f16x8, multiples_f16x8, vdupq_n_f16(3.140625));
    reduced_f16x8 = vfmsq_f16(reduced_f16x8, multiples_f16x8, vdupq_n_f16(0.0009675025939941406));
    float16x8_t const scaled_multiples_f16x8 = vmulq_f16(multiples_f16x8, vdupq_n_f16(0.000244140625));
    return vfmsq_f16(reduced_f16x8, scaled_multiples_f16x8, vdupq_n_f16(0.0006184577941894531));
}

/** Subtracts @p multiples_f32x4 of π from @p angles_f32x4, for F16 angles of any magnitude. */
NUMKONG_INLINE float32x4_t nk_reduce_pi_f32x4_neonhalf_(float32x4_t angles_f32x4, float32x4_t multiples_f32x4) {
    float32x4_t const reduced_f32x4 = vfmsq_f32(angles_f32x4, multiples_f32x4, vdupq_n_f32(3.140625f));
    return vfmsq_f32(reduced_f32x4, multiples_f32x4, vdupq_n_f32(9.676535897e-4f));
}

/** Flips the sign of @p values_f32x4 in the lanes where @p flips_i32x4 is odd. */
NUMKONG_INLINE float32x4_t nk_flip_odd_f32x4_neonhalf_(float32x4_t values_f32x4, int32x4_t flips_i32x4) {
    uint32x4_t const signs_u32x4 = vshlq_n_u32(vreinterpretq_u32_s32(flips_i32x4), 31);
    return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(values_f32x4), signs_u32x4));
}

/** Flips the sign of @p values_f16x8 in the lanes where @p flips_i16x8 is odd. */
NUMKONG_INLINE float16x8_t nk_flip_odd_f16x8_neonhalf_(float16x8_t values_f16x8, int16x8_t flips_i16x8) {
    uint16x8_t const signs_u16x8 = vshlq_n_u16(vreinterpretq_u16_s16(flips_i16x8), 15);
    return vreinterpretq_f16_u16(veorq_u16(vreinterpretq_u16_f16(values_f16x8), signs_u16x8));
}

/** Reduces 4 F32 angles by the nearest multiple of π, negated where that multiple is odd. */
NUMKONG_INLINE float32x4_t nk_sin_reduce_f32x4_neonhalf_(float32x4_t angles_f32x4) {
    int32x4_t const multiples_i32x4 = vcvtnq_s32_f32(vmulq_f32(angles_f32x4, vdupq_n_f32(0.31830988618379067154f)));
    float32x4_t const reduced_f32x4 = nk_reduce_pi_f32x4_neonhalf_(angles_f32x4, vcvtq_f32_s32(multiples_i32x4));
    return nk_flip_odd_f32x4_neonhalf_(reduced_f32x4, multiples_i32x4);
}

/** Reduces 4 F32 angles by the nearest odd multiple of π/2, signed so its sine is their cosine. */
NUMKONG_INLINE float32x4_t nk_cos_reduce_f32x4_neonhalf_(float32x4_t angles_f32x4) {
    float32x4_t const quotients_f32x4 = vfmaq_f32(vdupq_n_f32(-0.5f), angles_f32x4,
                                                  vdupq_n_f32(0.31830988618379067154f));
    int32x4_t const multiples_i32x4 = vcvtnq_s32_f32(quotients_f32x4);
    float32x4_t const offsets_f32x4 = vaddq_f32(vcvtq_f32_s32(multiples_i32x4), vdupq_n_f32(0.5f));
    float32x4_t const reduced_f32x4 = nk_reduce_pi_f32x4_neonhalf_(angles_f32x4, offsets_f32x4);
    return nk_flip_odd_f32x4_neonhalf_(reduced_f32x4, vmvnq_s32(multiples_i32x4));
}

/** Sine of 8 F16 angles within one F16 ULP. */
NUMKONG_INLINE float16x8_t nk_sin_f16x8_neonhalf_(float16x8_t angles_f16x8) {
    float16x8_t reduced_f16x8;
    // The F16 reduction holds one ULP only up to |x| ≤ 256
    if (vmaxvq_u16(vcagtq_f16(angles_f16x8, vdupq_n_f16(256)))) {
        float32x4_t const low_f32x4 = nk_sin_reduce_f32x4_neonhalf_(vcvt_f32_f16(vget_low_f16(angles_f16x8)));
        float32x4_t const high_f32x4 = nk_sin_reduce_f32x4_neonhalf_(vcvt_high_f32_f16(angles_f16x8));
        reduced_f16x8 = vcvt_high_f16_f32(vcvt_f16_f32(low_f32x4), high_f32x4);
    }
    else {
        float16x8_t const multiples_f16x8 = vrndnq_f16(vmulq_f16(angles_f16x8, vdupq_n_f16(0.31830988618379067154)));
        reduced_f16x8 = nk_reduce_pi_f16x8_neonhalf_(angles_f16x8, multiples_f16x8);
        reduced_f16x8 = nk_flip_odd_f16x8_neonhalf_(reduced_f16x8, vcvtq_s16_f16(multiples_f16x8));
    }
    return nk_sin_reduced_f16x8_neonhalf_(reduced_f16x8);
}

/** Cosine of 8 F16 angles within one F16 ULP. */
NUMKONG_INLINE float16x8_t nk_cos_f16x8_neonhalf_(float16x8_t angles_f16x8) {
    float16x8_t reduced_f16x8;
    // The F16 reduction holds one ULP only up to |x| ≤ 256
    if (vmaxvq_u16(vcagtq_f16(angles_f16x8, vdupq_n_f16(256)))) {
        float32x4_t const low_f32x4 = nk_cos_reduce_f32x4_neonhalf_(vcvt_f32_f16(vget_low_f16(angles_f16x8)));
        float32x4_t const high_f32x4 = nk_cos_reduce_f32x4_neonhalf_(vcvt_high_f32_f16(angles_f16x8));
        reduced_f16x8 = vcvt_high_f16_f32(vcvt_f16_f32(low_f32x4), high_f32x4);
    }
    else {
        float16x8_t const quotients_f16x8 = vfmaq_f16(vdupq_n_f16(-0.5), angles_f16x8,
                                                      vdupq_n_f16(0.31830988618379067154));
        float16x8_t const multiples_f16x8 = vrndnq_f16(quotients_f16x8);
        float16x8_t const offsets_f16x8 = vaddq_f16(multiples_f16x8, vdupq_n_f16(0.5));
        reduced_f16x8 = nk_reduce_pi_f16x8_neonhalf_(angles_f16x8, offsets_f16x8);
        reduced_f16x8 = nk_flip_odd_f16x8_neonhalf_(reduced_f16x8, vmvnq_s16(vcvtq_s16_f16(multiples_f16x8)));
    }
    return nk_sin_reduced_f16x8_neonhalf_(reduced_f16x8);
}

/** Arctangent of 8 F16 values within one F16 ULP. */
NUMKONG_INLINE float16x8_t nk_atan_f16x8_neonhalf_(float16x8_t values_f16x8) {
    float16x8_t const one_f16x8 = vdupq_n_f16(1), zero_f16x8 = vdupq_n_f16(0);

    // Fold |x| > 1 into [0, 1] through atan(x) = π/2 - atan(1/x)
    float16x8_t const magnitudes_f16x8 = vabsq_f16(values_f16x8);
    uint16x8_t const folded_u16x8 = vcagtq_f16(values_f16x8, one_f16x8);
    float16x8_t const reduced_f16x8 = vbslq_f16(folded_u16x8, vdivq_f16(one_f16x8, magnitudes_f16x8), magnitudes_f16x8);

    // Folded lanes add the low part of π/2 first, as its high part alone is half an F16 ULP off
    float16x8_t const signed_f16x8 = vbslq_f16(folded_u16x8, vnegq_f16(reduced_f16x8), reduced_f16x8);
    float16x8_t const bases_f16x8 = vaddq_f16(signed_f16x8,
                                              vbslq_f16(folded_u16x8, vdupq_n_f16(0.0004837512969970703), zero_f16x8));

    // Degree-7 odd polynomial with a unit linear term, coefficients searched in F16 arithmetic
    float16x8_t const squared_f16x8 = vmulq_f16(reduced_f16x8, reduced_f16x8);
    float16x8_t polynomial_f16x8 = vfmaq_f16(vdupq_n_f16(+0.1600341796875), squared_f16x8,
                                             vdupq_n_f16(-0.046722412109375));
    polynomial_f16x8 = vfmaq_f16(vdupq_n_f16(-0.328125), polynomial_f16x8, squared_f16x8);
    float16x8_t results_f16x8 = vfmaq_f16(bases_f16x8, vmulq_f16(signed_f16x8, squared_f16x8), polynomial_f16x8);
    results_f16x8 = vaddq_f16(results_f16x8, vbslq_f16(folded_u16x8, vdupq_n_f16(1.5703125), zero_f16x8));
    return vbslq_f16(vdupq_n_u16(0x8000), values_f16x8, results_f16x8);
}

NUMKONG_API nk_status_t nk_trig_sin_f16_neonhalf(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t angles_f16x8 = vld1q_f16((float16_t const *)ins + i);
        vst1q_f16((float16_t *)outs + i, nk_sin_f16x8_neonhalf_(angles_f16x8));
    }
    if (i < n) {
        nk_b128_vec_t angles_vec, results_vec;
        nk_partial_load_b16x8_serial_(ins + i, &angles_vec, n - i);
        results_vec.f16x8 = nk_sin_f16x8_neonhalf_(angles_vec.f16x8);
        nk_partial_store_b16x8_serial_(&results_vec, outs + i, n - i);
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_trig_cos_f16_neonhalf(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t angles_f16x8 = vld1q_f16((float16_t const *)ins + i);
        vst1q_f16((float16_t *)outs + i, nk_cos_f16x8_neonhalf_(angles_f16x8));
    }
    if (i < n) {
        nk_b128_vec_t angles_vec, results_vec;
        nk_partial_load_b16x8_serial_(ins + i, &angles_vec, n - i);
        results_vec.f16x8 = nk_cos_f16x8_neonhalf_(angles_vec.f16x8);
        nk_partial_store_b16x8_serial_(&results_vec, outs + i, n - i);
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_trig_atan_f16_neonhalf(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs,
                                                  nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t values_f16x8 = vld1q_f16((float16_t const *)ins + i);
        vst1q_f16((float16_t *)outs + i, nk_atan_f16x8_neonhalf_(values_f16x8));
    }
    if (i < n) {
        nk_b128_vec_t values_vec, results_vec;
        nk_partial_load_b16x8_serial_(ins + i, &values_vec, n - i);
        results_vec.f16x8 = nk_atan_f16x8_neonhalf_(values_vec.f16x8);
        nk_partial_store_b16x8_serial_(&results_vec, outs + i, n - i);
    }
    return nk_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_NEONHALF
#endif // NUMKONG_ARCH_ARM64_
#endif // NUMKONG_TRIGONOMETRY_NEONHALF_H
