/**
 *  @brief SIMD-accelerated Dot Products for NEON FP8DOT4.
 *  @file include/numkong/dot/neonfp8.h
 *  @author Ash Vardanian
 *  @date March 23, 2026
 *
 *  @sa include/numkong/dot.h
 *
 *  @section dot_neonfp8_instructions ARM NEON FP8DOT4 Instructions (FEAT_FP8DOT4)
 *
 *      Intrinsic          Instruction                V1
 *      vdotq_f32_mf8      FDOT (V.4S, V.16B, V.16B)  4cy @ 2p
 *      vld1q_u8           LD1 (V.16B)                4cy @ 2p
 *      vaddvq_f32         FADDP+FADDP (V.4S)         4cy @ 1p
 *      vpaddq_f32         FADDP (V.4S, V.4S, V.4S)   2cy @ 2p
 *
 *  FEAT_FP8DOT4 adds NEON FDOT instructions that take two 128-bit vectors of FP8 (E4M3 or E5M2),
 *  perform 4-way multiply-accumulate into FP32 per lane. Each FDOT processes 16 FP8 elements
 *  into 4 FP32 accumulators. The FP8 format is selected by the FPMR register.
 *
 *  FP6 types (E2M3, E3M2) are losslessly promoted to FP8 (E4M3, E5M2) by rebiasing the exponent.
 *  Normal values: magnitude += 48. Subnormal values (exp=0): 8-entry or 4-entry TBL lookup.
 *
 *  @section dot_neonfp8_stateful Stateful Streaming Logic
 *
 *  Defines stateful init/update/finalize helpers for tiled GEMM via the dots/ macros:
 *  - nk_dot_e4m3x16_state_neonfp8_t, nk_dot_e5m2x16_state_neonfp8_t
 *  - nk_dot_e2m3x16_state_neonfp8_t, nk_dot_e3m2x16_state_neonfp8_t
 */
#ifndef NK_DOT_NEONFP8_H
#define NK_DOT_NEONFP8_H

#if NK_TARGET_ARM64_
#if NK_TARGET_NEONFP8

#include "numkong/types.h"
#include "numkong/cast/serial.h" // `nk_partial_load_b8x16_serial_`

/** @brief FPM immediate for E4M3 × E4M3 dot products: src1=E4M3(1), src2=E4M3(1). */
#define NK_FPM_E4M3_ ((fpm_t)((1ull << 0) | (1ull << 3)))
/** @brief FPM immediate for E5M2 × E5M2 dot products: src1=E5M2(0), src2=E5M2(0). */
#define NK_FPM_E5M2_ ((fpm_t)0)

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd+fp8dot4"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd+fp8dot4")
#endif

/**
 *  @brief Convert 16 E2M3 bytes (0b00SEEMMM) to E4M3 bytes (0bSEEEEMMM).
 *
 *  Normal values (exp>0, mag>=8): rebias exponent by +6 → magnitude += 48.
 *  Subnormal values (exp=0, mag<8): 8-entry TBL lookup for normalization.
 *  Zero (mag=0): maps to E4M3 zero. Sign moved from bit 5 to bit 7.
 */
NK_HELPER_INLINE uint8x16_t nk_e2m3x16_to_e4m3x16_neonfp8_(uint8x16_t raw_u8x16) {
    uint8x16_t sign_u8x16 = vandq_u8(raw_u8x16, vdupq_n_u8(0x20));
    uint8x16_t mag_u8x16 = vandq_u8(raw_u8x16, vdupq_n_u8(0x1F));

    // Normal path: rebias exponent by +6 → add 48 to magnitude
    uint8x16_t normal_mag_u8x16 = vaddq_u8(mag_u8x16, vdupq_n_u8(48));

    // Subnormal path: 8-entry LUT for mag 0-7
    // 0→0, 1→32, 2→40, 3→44, 4→48, 5→50, 6→52, 7→54
    uint8x16_t sub_lut_u8x16 = vcombine_u8(vcreate_u8(0x363432302c282000ull), vcreate_u8(0));
    uint8x16_t sub_mag_u8x16 = vqtbl1q_u8(sub_lut_u8x16, mag_u8x16);

    // Select: subnormal (mag < 8) uses LUT, normal uses +48
    uint8x16_t is_normal_u8x16 = vcgeq_u8(mag_u8x16, vdupq_n_u8(8));
    uint8x16_t result_mag_u8x16 = vbslq_u8(is_normal_u8x16, normal_mag_u8x16, sub_mag_u8x16);

    // Move sign from bit 5 to bit 7
    uint8x16_t sign_shifted_u8x16 = vshlq_n_u8(sign_u8x16, 2);
    return vorrq_u8(sign_shifted_u8x16, result_mag_u8x16);
}

/**
 *  @brief Convert 16 E3M2 bytes (0b00SEEEMM) to E5M2 bytes (0bSEEEEEMM).
 *
 *  Normal values (exp>0, mag>=4): rebias exponent by +12 → magnitude += 48.
 *  Subnormal values (exp=0, mag<4): 4-entry TBL lookup for normalization.
 *  Zero (mag=0): maps to E5M2 zero. Sign moved from bit 5 to bit 7.
 */
NK_HELPER_INLINE uint8x16_t nk_e3m2x16_to_e5m2x16_neonfp8_(uint8x16_t raw_u8x16) {
    uint8x16_t sign_u8x16 = vandq_u8(raw_u8x16, vdupq_n_u8(0x20));
    uint8x16_t mag_u8x16 = vandq_u8(raw_u8x16, vdupq_n_u8(0x1F));

    // Normal path: rebias exponent by +12 → add 48 to magnitude
    uint8x16_t normal_mag_u8x16 = vaddq_u8(mag_u8x16, vdupq_n_u8(48));

    // Subnormal path: 4-entry LUT for mag 0-3
    // 0→0, 1→44, 2→48, 3→50
    uint8x16_t sub_lut_u8x16 = vcombine_u8(vcreate_u8(0x0000000032302c00ull), vcreate_u8(0));
    uint8x16_t sub_mag_u8x16 = vqtbl1q_u8(sub_lut_u8x16, mag_u8x16);

    // Select: subnormal (mag < 4) uses LUT, normal uses +48
    uint8x16_t is_normal_u8x16 = vcgeq_u8(mag_u8x16, vdupq_n_u8(4));
    uint8x16_t result_mag_u8x16 = vbslq_u8(is_normal_u8x16, normal_mag_u8x16, sub_mag_u8x16);

    // Move sign from bit 5 to bit 7
    uint8x16_t sign_shifted_u8x16 = vshlq_n_u8(sign_u8x16, 2);
    return vorrq_u8(sign_shifted_u8x16, result_mag_u8x16);
}

NK_API_COMPTIME void nk_dot_e4m3_neonfp8(nk_e4m3_t const *a_scalars, nk_e4m3_t const *b_scalars,
                                         nk_size_t count_scalars, nk_f32_t *result) {
    mfloat8x16_t a_mf8x16, b_mf8x16;
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
nk_dot_e4m3_neonfp8_cycle:
    if (count_scalars < 16) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b8x16_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b8x16_serial_(b_scalars, &b_vec, count_scalars);
        a_mf8x16 = vreinterpretq_mf8_u8(a_vec.u8x16);
        b_mf8x16 = vreinterpretq_mf8_u8(b_vec.u8x16);
        count_scalars = 0;
    }
    else {
        a_mf8x16 = vreinterpretq_mf8_u8(vld1q_u8((nk_u8_t const *)a_scalars));
        b_mf8x16 = vreinterpretq_mf8_u8(vld1q_u8((nk_u8_t const *)b_scalars));
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }
    sum_f32x4 = vdotq_f32_mf8_fpm(sum_f32x4, a_mf8x16, b_mf8x16, NK_FPM_E4M3_);
    if (count_scalars) goto nk_dot_e4m3_neonfp8_cycle;
    *result = vaddvq_f32(sum_f32x4);
}

NK_API_COMPTIME void nk_dot_e5m2_neonfp8(nk_e5m2_t const *a_scalars, nk_e5m2_t const *b_scalars,
                                         nk_size_t count_scalars, nk_f32_t *result) {
    mfloat8x16_t a_mf8x16, b_mf8x16;
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
nk_dot_e5m2_neonfp8_cycle:
    if (count_scalars < 16) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b8x16_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b8x16_serial_(b_scalars, &b_vec, count_scalars);
        a_mf8x16 = vreinterpretq_mf8_u8(a_vec.u8x16);
        b_mf8x16 = vreinterpretq_mf8_u8(b_vec.u8x16);
        count_scalars = 0;
    }
    else {
        a_mf8x16 = vreinterpretq_mf8_u8(vld1q_u8((nk_u8_t const *)a_scalars));
        b_mf8x16 = vreinterpretq_mf8_u8(vld1q_u8((nk_u8_t const *)b_scalars));
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }
    sum_f32x4 = vdotq_f32_mf8_fpm(sum_f32x4, a_mf8x16, b_mf8x16, NK_FPM_E5M2_);
    if (count_scalars) goto nk_dot_e5m2_neonfp8_cycle;
    *result = vaddvq_f32(sum_f32x4);
}

NK_API_COMPTIME void nk_dot_e2m3_neonfp8(nk_e2m3_t const *a_scalars, nk_e2m3_t const *b_scalars,
                                         nk_size_t count_scalars, nk_f32_t *result) {
    mfloat8x16_t a_mf8x16, b_mf8x16;
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
nk_dot_e2m3_neonfp8_cycle:
    if (count_scalars < 16) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b8x16_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b8x16_serial_(b_scalars, &b_vec, count_scalars);
        a_mf8x16 = vreinterpretq_mf8_u8(nk_e2m3x16_to_e4m3x16_neonfp8_(a_vec.u8x16));
        b_mf8x16 = vreinterpretq_mf8_u8(nk_e2m3x16_to_e4m3x16_neonfp8_(b_vec.u8x16));
        count_scalars = 0;
    }
    else {
        a_mf8x16 = vreinterpretq_mf8_u8(nk_e2m3x16_to_e4m3x16_neonfp8_(vld1q_u8((nk_u8_t const *)a_scalars)));
        b_mf8x16 = vreinterpretq_mf8_u8(nk_e2m3x16_to_e4m3x16_neonfp8_(vld1q_u8((nk_u8_t const *)b_scalars)));
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }
    sum_f32x4 = vdotq_f32_mf8_fpm(sum_f32x4, a_mf8x16, b_mf8x16, NK_FPM_E4M3_);
    if (count_scalars) goto nk_dot_e2m3_neonfp8_cycle;
    *result = vaddvq_f32(sum_f32x4);
}

NK_API_COMPTIME void nk_dot_e3m2_neonfp8(nk_e3m2_t const *a_scalars, nk_e3m2_t const *b_scalars,
                                         nk_size_t count_scalars, nk_f32_t *result) {
    mfloat8x16_t a_mf8x16, b_mf8x16;
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
nk_dot_e3m2_neonfp8_cycle:
    if (count_scalars < 16) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b8x16_serial_(a_scalars, &a_vec, count_scalars);
        nk_partial_load_b8x16_serial_(b_scalars, &b_vec, count_scalars);
        a_mf8x16 = vreinterpretq_mf8_u8(nk_e3m2x16_to_e5m2x16_neonfp8_(a_vec.u8x16));
        b_mf8x16 = vreinterpretq_mf8_u8(nk_e3m2x16_to_e5m2x16_neonfp8_(b_vec.u8x16));
        count_scalars = 0;
    }
    else {
        a_mf8x16 = vreinterpretq_mf8_u8(nk_e3m2x16_to_e5m2x16_neonfp8_(vld1q_u8((nk_u8_t const *)a_scalars)));
        b_mf8x16 = vreinterpretq_mf8_u8(nk_e3m2x16_to_e5m2x16_neonfp8_(vld1q_u8((nk_u8_t const *)b_scalars)));
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }
    sum_f32x4 = vdotq_f32_mf8_fpm(sum_f32x4, a_mf8x16, b_mf8x16, NK_FPM_E5M2_);
    if (count_scalars) goto nk_dot_e3m2_neonfp8_cycle;
    *result = vaddvq_f32(sum_f32x4);
}

typedef struct nk_dot_e4m3x16_state_neonfp8_t {
    float32x4_t sum_f32x4;
} nk_dot_e4m3x16_state_neonfp8_t;

NK_HELPER_INLINE void nk_dot_e4m3x16_init_neonfp8(nk_dot_e4m3x16_state_neonfp8_t *state) {
    state->sum_f32x4 = vdupq_n_f32(0);
}

NK_HELPER_INLINE void nk_dot_e4m3x16_update_neonfp8(nk_dot_e4m3x16_state_neonfp8_t *state, nk_b128_vec_t a,
                                                    nk_b128_vec_t b, nk_size_t depth_offset,
                                                    nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    mfloat8x16_t a_mf8x16 = vreinterpretq_mf8_u8(a.u8x16);
    mfloat8x16_t b_mf8x16 = vreinterpretq_mf8_u8(b.u8x16);
    state->sum_f32x4 = vdotq_f32_mf8_fpm(state->sum_f32x4, a_mf8x16, b_mf8x16, NK_FPM_E4M3_);
}

NK_HELPER_INLINE void nk_dot_e4m3x16_finalize_neonfp8(                                            //
    nk_dot_e4m3x16_state_neonfp8_t const *state_a, nk_dot_e4m3x16_state_neonfp8_t const *state_b, //
    nk_dot_e4m3x16_state_neonfp8_t const *state_c, nk_dot_e4m3x16_state_neonfp8_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    float32x4_t ab_f32x4 = vpaddq_f32(state_a->sum_f32x4, state_b->sum_f32x4);
    float32x4_t cd_f32x4 = vpaddq_f32(state_c->sum_f32x4, state_d->sum_f32x4);
    result->f32x4 = vpaddq_f32(ab_f32x4, cd_f32x4);
}

typedef struct nk_dot_e5m2x16_state_neonfp8_t {
    float32x4_t sum_f32x4;
} nk_dot_e5m2x16_state_neonfp8_t;

NK_HELPER_INLINE void nk_dot_e5m2x16_init_neonfp8(nk_dot_e5m2x16_state_neonfp8_t *state) {
    state->sum_f32x4 = vdupq_n_f32(0);
}

NK_HELPER_INLINE void nk_dot_e5m2x16_update_neonfp8(nk_dot_e5m2x16_state_neonfp8_t *state, nk_b128_vec_t a,
                                                    nk_b128_vec_t b, nk_size_t depth_offset,
                                                    nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    mfloat8x16_t a_mf8x16 = vreinterpretq_mf8_u8(a.u8x16);
    mfloat8x16_t b_mf8x16 = vreinterpretq_mf8_u8(b.u8x16);
    state->sum_f32x4 = vdotq_f32_mf8_fpm(state->sum_f32x4, a_mf8x16, b_mf8x16, NK_FPM_E5M2_);
}

NK_HELPER_INLINE void nk_dot_e5m2x16_finalize_neonfp8(                                            //
    nk_dot_e5m2x16_state_neonfp8_t const *state_a, nk_dot_e5m2x16_state_neonfp8_t const *state_b, //
    nk_dot_e5m2x16_state_neonfp8_t const *state_c, nk_dot_e5m2x16_state_neonfp8_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    float32x4_t ab_f32x4 = vpaddq_f32(state_a->sum_f32x4, state_b->sum_f32x4);
    float32x4_t cd_f32x4 = vpaddq_f32(state_c->sum_f32x4, state_d->sum_f32x4);
    result->f32x4 = vpaddq_f32(ab_f32x4, cd_f32x4);
}

typedef struct nk_dot_e2m3x16_state_neonfp8_t {
    float32x4_t sum_f32x4;
} nk_dot_e2m3x16_state_neonfp8_t;

NK_HELPER_INLINE void nk_dot_e2m3x16_init_neonfp8(nk_dot_e2m3x16_state_neonfp8_t *state) {
    state->sum_f32x4 = vdupq_n_f32(0);
}

NK_HELPER_INLINE void nk_dot_e2m3x16_update_neonfp8(nk_dot_e2m3x16_state_neonfp8_t *state, nk_b128_vec_t a,
                                                    nk_b128_vec_t b, nk_size_t depth_offset,
                                                    nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    mfloat8x16_t a_mf8x16 = vreinterpretq_mf8_u8(nk_e2m3x16_to_e4m3x16_neonfp8_(a.u8x16));
    mfloat8x16_t b_mf8x16 = vreinterpretq_mf8_u8(nk_e2m3x16_to_e4m3x16_neonfp8_(b.u8x16));
    state->sum_f32x4 = vdotq_f32_mf8_fpm(state->sum_f32x4, a_mf8x16, b_mf8x16, NK_FPM_E4M3_);
}

NK_HELPER_INLINE void nk_dot_e2m3x16_finalize_neonfp8(                                            //
    nk_dot_e2m3x16_state_neonfp8_t const *state_a, nk_dot_e2m3x16_state_neonfp8_t const *state_b, //
    nk_dot_e2m3x16_state_neonfp8_t const *state_c, nk_dot_e2m3x16_state_neonfp8_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    float32x4_t ab_f32x4 = vpaddq_f32(state_a->sum_f32x4, state_b->sum_f32x4);
    float32x4_t cd_f32x4 = vpaddq_f32(state_c->sum_f32x4, state_d->sum_f32x4);
    result->f32x4 = vpaddq_f32(ab_f32x4, cd_f32x4);
}

typedef struct nk_dot_e3m2x16_state_neonfp8_t {
    float32x4_t sum_f32x4;
} nk_dot_e3m2x16_state_neonfp8_t;

NK_HELPER_INLINE void nk_dot_e3m2x16_init_neonfp8(nk_dot_e3m2x16_state_neonfp8_t *state) {
    state->sum_f32x4 = vdupq_n_f32(0);
}

NK_HELPER_INLINE void nk_dot_e3m2x16_update_neonfp8(nk_dot_e3m2x16_state_neonfp8_t *state, nk_b128_vec_t a,
                                                    nk_b128_vec_t b, nk_size_t depth_offset,
                                                    nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    mfloat8x16_t a_mf8x16 = vreinterpretq_mf8_u8(nk_e3m2x16_to_e5m2x16_neonfp8_(a.u8x16));
    mfloat8x16_t b_mf8x16 = vreinterpretq_mf8_u8(nk_e3m2x16_to_e5m2x16_neonfp8_(b.u8x16));
    state->sum_f32x4 = vdotq_f32_mf8_fpm(state->sum_f32x4, a_mf8x16, b_mf8x16, NK_FPM_E5M2_);
}

NK_HELPER_INLINE void nk_dot_e3m2x16_finalize_neonfp8(                                            //
    nk_dot_e3m2x16_state_neonfp8_t const *state_a, nk_dot_e3m2x16_state_neonfp8_t const *state_b, //
    nk_dot_e3m2x16_state_neonfp8_t const *state_c, nk_dot_e3m2x16_state_neonfp8_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    float32x4_t ab_f32x4 = vpaddq_f32(state_a->sum_f32x4, state_b->sum_f32x4);
    float32x4_t cd_f32x4 = vpaddq_f32(state_c->sum_f32x4, state_d->sum_f32x4);
    result->f32x4 = vpaddq_f32(ab_f32x4, cd_f32x4);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEONFP8
#endif // NK_TARGET_ARM64_
#endif // NK_DOT_NEONFP8_H
