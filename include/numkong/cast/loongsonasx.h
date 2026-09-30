/**
 *  @file include/numkong/cast/loongsonasx.h
 *  @author Ash Vardanian
 *  @date March 23, 2026
 *  @brief SIMD-accelerated type conversions and load/store helpers for LoongArch LASX, 256-bit.
 *
 *  @sa include/numkong/cast.h
 *
 *  @section loongsonasx_cast_instructions Key LASX Load/Store Instructions
 *
 *  @verbatim
 *  Intrinsic                      Instruction       Description
 *  __lasx_xvld(address, 0)        XVLD              256-bit aligned/unaligned load
 *  __lasx_xvst(v, address, 0)     XVST              256-bit aligned/unaligned store
 *  @endverbatim
 *
 *  LASX is a 256-bit extension; all vector registers are 256-bit @c __m256i. For 128-bit
 *  @c nk_b128_vec_t operations, @c __lasx_xvld safely loads into the low 128 bits — the high 128
 *  bits are zeroed or undefined depending on context. For 128-bit stores we use @c memcpy to avoid
 *  writing beyond the intended 16 bytes. Partial loads/stores delegate to serial helpers since LASX
 *  lacks masked load/store instructions.
 */
#ifndef NUMKONG_CAST_LOONGSONASX_H
#define NUMKONG_CAST_LOONGSONASX_H

#if NUMKONG_ARCH_LOONGARCH64_
#if NUMKONG_TARGET_LOONGSONASX

#include "numkong/types.h"
#include "numkong/cast/serial.h"        // `nk_partial_load_b32x4_serial_`, `nk_partial_load_b64x4_serial_`
#include "numkong/scalar/loongsonasx.h" // `nk_xvreplgr2vr_s_128_`, `nk_xvfreplgr2vr_s_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("lasx"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("lasx")
#endif

#pragma region Type Punned Loads and Stores

/** LSX and LASX share the same physical register file, so widening __m128i → __m256i and extracting
 *  __m256i → __m128i are no-ops on hardware. Empty inline asm with "f" constraints avoids the stack
 *  round-trip that union punning causes on GCC 14. The helpers are named after the x86 intrinsics
 *  _mm256_castsi128_si256, _mm256_castsi256_si128 and _mm256_castps256_ps128. */
NUMKONG_INLINE __m256i nk_lasx_castsi128_si256_(__m128i low_i64x2) {
    __m256i wide_i64x4;
    __asm__("" : "=f"(wide_i64x4) : "f"(low_i64x2));
    return wide_i64x4;
}
NUMKONG_INLINE __m128i nk_lasx_castsi256_si128_(__m256i wide_i64x4) {
    __m128i low_i64x2;
    __asm__("" : "=f"(low_i64x2) : "f"(wide_i64x4));
    return low_i64x2;
}
NUMKONG_INLINE __m128 nk_lasx_castps256_ps128_(__m256 wide_f32x8) {
    __m128 low_f32x4;
    __asm__("" : "=f"(low_f32x4) : "f"(wide_f32x8));
    return low_f32x4;
}

/** Type-agnostic 256-bit full load (LASX). */
NUMKONG_INLINE void nk_load_b256_loongsonasx_(void const *src, nk_b256_vec_t *dst) { dst->ymm = __lasx_xvld(src, 0); }

/** Type-agnostic 256-bit full store (LASX). */
NUMKONG_INLINE void nk_store_b256_loongsonasx_(nk_b256_vec_t const *src, void *dst) { __lasx_xvst(src->ymm, dst, 0); }

/** Type-agnostic 128-bit full load (LSX subset of LASX). */
NUMKONG_INLINE void nk_load_b128_loongsonasx_(void const *src, nk_b128_vec_t *dst) { dst->xmm = __lsx_vld(src, 0); }

/** Type-agnostic 128-bit full store (LSX subset of LASX). */
NUMKONG_INLINE void nk_store_b128_loongsonasx_(nk_b128_vec_t const *src, void *dst) { __lsx_vst(src->xmm, dst, 0); }

/** Convert 8 × f16 → 8 × f32 via native LASX hardware conversion. */
NUMKONG_INLINE __m256i nk_f16x8_to_f32x8_loongsonasx_(__m128i f16_i16x8) {
    __m256i duped_f16x16 = __lasx_xvpermi_q(nk_lasx_castsi128_si256_(f16_i16x8), nk_lasx_castsi128_si256_(f16_i16x8),
                                            0x00);
    __m256i low_f32x8 = (__m256i)__lasx_xvfcvtl_s_h(duped_f16x16);
    __m256i high_f32x8 = (__m256i)__lasx_xvfcvth_s_h(duped_f16x16);
    return __lasx_xvpermi_q(high_f32x8, low_f32x8, 0x20);
}

/** Load 8 × f16 from memory, convert to 8 × f32 via native LASX conversion. */
NUMKONG_INLINE void nk_load_f16x8_to_f32x8_loongsonasx_(void const *src, nk_b256_vec_t *dst) {
    dst->ymm = nk_f16x8_to_f32x8_loongsonasx_(__lsx_vld(src, 0));
}

/** Partial load for f16 elements (up to 8) with conversion to f32 (LASX). */
NUMKONG_INLINE void nk_partial_load_f16x8_to_f32x8_loongsonasx_(nk_f16_t const *src, nk_b256_vec_t *dst, nk_size_t n) {
    nk_b128_vec_t vec;
    nk_partial_load_b16x8_serial_(src, &vec, n);
    dst->ymm = nk_f16x8_to_f32x8_loongsonasx_(vec.xmm);
}

#pragma endregion Type Punned Loads and Stores

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_LOONGSONASX
#endif // NUMKONG_ARCH_LOONGARCH64_
#endif // NUMKONG_CAST_LOONGSONASX_H
