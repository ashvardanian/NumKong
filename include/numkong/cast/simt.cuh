/**
 *  @file include/numkong/cast/simt.cuh
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Scalar conversions on the SIMT cores of every CUDA and ROCm device.
 *
 *  @sa include/numkong/cast/serial.h
 *
 *  Device twins of the serial casts, with their signatures, that the GPU families load and store
 *  through. Every narrowing rounds to nearest even like its serial original, so a GPU output
 *  matches a serial one bit for bit wherever both round the same F32 value.
 */
#ifndef NUMKONG_CAST_SIMT_CUH
#define NUMKONG_CAST_SIMT_CUH

#if NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/** One E4M3 code over 256, its NaN code read as 480. */
NUMKONG_DEVICE nk_f32_t nk_e4m3_to_scaled_f32_(nk_u32_t code) {
    return __half2float(__ushort_as_half((unsigned short)(((code & 0x7Fu) << 7) | ((code & 0x80u) << 8))));
}

/** Widens one BF16 value to F32, exactly. */
NUMKONG_DEVICE void nk_bf16_to_f32_simt_(nk_bf16_t const *src, nk_f32_t *dest) {
    *dest = __uint_as_float((nk_u32_t)(*(unsigned short const *)src) << 16);
}

/** Widens one F16 value to F32, exactly. */
NUMKONG_DEVICE void nk_f16_to_f32_simt_(nk_f16_t const *src, nk_f32_t *dest) {
    *dest = __half2float(__ushort_as_half(*(unsigned short const *)src));
}

/** Widens one E5M2 value to F32, exactly. */
NUMKONG_DEVICE void nk_e5m2_to_f32_simt_(nk_e5m2_t const *src, nk_f32_t *dest) {
    *dest = __half2float(__ushort_as_half((unsigned short)((nk_u32_t)*src << 8)));
}

/** Widens one E4M3FN value to F32, exactly. */
NUMKONG_DEVICE void nk_e4m3_to_f32_simt_(nk_e4m3_t const *src, nk_f32_t *dest) {
    nk_u32_t const code = *src;
    *dest = (code & 0x7Fu) == 0x7Fu ? __uint_as_float(0x7FC00000u) : nk_e4m3_to_scaled_f32_(code) * 256.0f;
}

/** Narrows one F32 value to BF16, rounding to nearest even and keeping NaNs. */
NUMKONG_DEVICE void nk_f32_to_bf16_simt_(nk_f32_t const *src, nk_bf16_t *dest) {
    nk_u32_t const bits = __float_as_uint(*src);
    *(unsigned short *)dest = (bits & 0x7FFFFFFFu) > 0x7F800000u
                                  ? (unsigned short)((bits >> 16) | 0x0040u)
                                  : (unsigned short)((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
}

/** Narrows one F32 value to F16, rounding to nearest even. */
NUMKONG_DEVICE void nk_f32_to_f16_simt_(nk_f32_t const *src, nk_f16_t *dest) {
    *(unsigned short *)dest = __half_as_ushort(__float2half_rn(*src));
}

/** Narrows one F32 value to E4M3FN, rounding to nearest even and saturating at 448. */
NUMKONG_DEVICE void nk_f32_to_e4m3_simt_(nk_f32_t const *src, nk_e4m3_t *dest) {
    nk_u32_t const bits = __float_as_uint(*src), magnitude_bits = bits & 0x7FFFFFFFu;
    unsigned const sign = (bits >> 24) & 0x80u;
    nk_f32_t const magnitude = __uint_as_float(magnitude_bits);
    if (magnitude_bits > 0x7F800000u) { *dest = (nk_e4m3_t)(sign | 0x7Fu); }
    else if (magnitude_bits == 0x7F800000u) { *dest = (nk_e4m3_t)(sign | 0x7Eu); }
    else if (magnitude < 1.0f / 64.0f) {
        // Subnormals step by 1/512 below the smallest normal; rounding up to 8 is the first normal
        nk_f32_t const scaled = magnitude * 512.0f;
        unsigned mantissa = (unsigned)scaled;
        nk_f32_t const fraction = scaled - (nk_f32_t)mantissa;
        if (fraction > 0.5f || (fraction == 0.5f && (mantissa & 1u))) ++mantissa;
        *dest = (nk_e4m3_t)(sign | (mantissa > 7u ? 0x08u : mantissa));
    }
    else {
        int exponent = (int)((magnitude_bits >> 23) & 0xFFu) - 127;
        nk_u32_t const significand = (1u << 23) | (magnitude_bits & 0x7FFFFFu);
        nk_u32_t const remainder = significand & ((1u << 20) - 1u), halfway = 1u << 19;
        nk_u32_t rounded = significand >> 20;
        if (remainder > halfway || (remainder == halfway && (rounded & 1u))) ++rounded;
        if (rounded == 16u) rounded >>= 1, ++exponent;
        unsigned const exponent_field = (unsigned)(exponent + 7), mantissa_field = rounded & 0x07u;
        // The top exponent keeps mantissa 7 for NaN, so it saturates at 6
        if (exponent > 8) { *dest = (nk_e4m3_t)(sign | 0x7Eu); }
        else if (exponent_field == 15u && mantissa_field > 6u) { *dest = (nk_e4m3_t)(sign | 0x7Eu); }
        else { *dest = (nk_e4m3_t)(sign | (exponent_field << 3) | mantissa_field); }
    }
}

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_
#endif // NUMKONG_CAST_SIMT_CUH
