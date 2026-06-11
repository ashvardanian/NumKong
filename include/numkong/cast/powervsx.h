/**
 *  @brief SIMD-accelerated Type Conversions for Power VSX.
 *  @file include/numkong/cast/powervsx.h
 *  @author Ash Vardanian
 *  @date March 23, 2026
 *
 *  @sa include/numkong/cast.h
 *
 *  @section powervsx_cast_instructions Power VSX Conversion Instructions (POWER9+)
 *
 *  Float16 hardware conversion (POWER9+):
 *
 *      Intrinsic                        Instruction       Notes
 *      vec_extract_fp32_from_shorth     xvcvhpsp          High 4 f16 → f32x4 (1 instruction!)
 *      vec_extract_fp32_from_shortl     xvcvhpsp          Low 4 f16 → f32x4 (1 instruction!)
 *
 *  Scalar f16 ↔ f32 (POWER9 inline asm):
 *
 *      Instruction   Notes
 *      lxsihzx       Load f16 → VSR (zero-extended)
 *      xscvhpdp      Convert half → double precision
 *      xscvdphp      Convert double → half precision
 *      stxsihx       Store f16 from VSR
 *
 *  Scalar sqrt (POWER9 inline asm):
 *
 *      Instruction   Notes
 *      xssqrtsp      Scalar single-precision sqrt
 *      xssqrtdp      Scalar double-precision sqrt
 *
 *  Float ↔ integer conversions:
 *
 *      Intrinsic   Instruction   Notes
 *      vec_cts     xvcvspsxws    f32x4 → i32x4 (truncation)
 *      vec_ctu     xvcvspuxws    f32x4 → u32x4 (truncation)
 *      vec_ctf     xvcvsxwsp     i32x4 → f32x4
 *      vec_ctf     xvcvuxwsp     u32x4 → f32x4
 *
 *  Integer narrowing/widening:
 *
 *      Intrinsic     Instruction   Notes
 *      vec_pack      vpkuwum       u32x4 → u16x8 (modular)
 *      vec_packs     vpkswss       i32x4 → i16x8 (signed saturation)
 *      vec_packsu    vpkswus       i32x4 → u16x8 (unsigned saturation from signed)
 *      vec_unpackh   vupkhsh       i16x8 → i32x4 (sign-extend high half)
 *      vec_mergeh    vmrghh        Interleave high halves (zero-extend via merge with zero)
 *
 *  Partial-length load:
 *
 *      Intrinsic     Instruction   Notes
 *      vec_xl_len    lxvl          Load up to 16 bytes with runtime length (POWER9)
 *
 *  Load/store:
 *
 *      Intrinsic   Instruction   Notes
 *      vec_xl      lxvd2x        Aligned/unaligned load
 *      vec_xst     stxvd2x       Aligned/unaligned store
 *
 *  BF16 conversions use bit manipulation (no hardware support):
 *  - bf16 → f32: zero-extend u16 → u32 via vec_mergeh with zero, reinterpret
 *  - f32 → bf16: RNE rounding + vec_sr by 16 + vec_pack
 *
 *  FP8 (E4M3/E5M2/E2M3/E3M2) types have no Power hardware support.
 *  Serial fallback via cast/serial.h is used for those formats.
 */
#ifndef NK_CAST_POWERVSX_H
#define NK_CAST_POWERVSX_H

#if NK_TARGET_POWER64_
#if NK_TARGET_POWERVSX

#include "numkong/types.h"
#include "numkong/cast/serial.h"   // `nk_cast_serial`, `nk_dtype_bits`
#include "numkong/reduce/serial.h" // `nk_reduce_moments_f32_serial`

// Power VSX vector typedefs — wrapping altivec built-in vector types.
// These may move to `numkong/types.h` in the future.
#ifndef NK_POWERVSX_TYPES_DEFINED_
#define NK_POWERVSX_TYPES_DEFINED_
#endif // NK_POWERVSX_TYPES_DEFINED_

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("power9-vector"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("power9-vector")
#endif

/** @brief Convert scalar f16 → f32 via POWER9 vector path (xvcvhpsp). */
NK_PUBLIC void nk_f16_to_f32_powervsx(nk_f16_t const *source, nk_f32_t *destination) {
    nk_vu16x8_t values_u16x8 = (nk_vu16x8_t)vec_xl_len((nk_u8_t *)source, 2);
    *destination = vec_extract(vec_extract_fp32_from_shorth(values_u16x8), 0);
}

/** @brief Convert scalar f32 → f16 via POWER9 vector path (xvcvsphp). */
NK_PUBLIC void nk_f32_to_f16_powervsx(nk_f32_t const *source, nk_f16_t *destination) {
    nk_vu16x8_t packed_u16x8 = vec_pack_to_short_fp32(vec_splats(*source), vec_splats(*source));
    *destination = vec_extract(packed_u16x8, 0);
}

/** @brief Type-agnostic 128-bit full load (Power VSX). */
NK_INTERNAL void nk_load_b128_powervsx_(void const *source, nk_b128_vec_t *destination) {
    destination->vu8x16 = vec_xl(0, (nk_u8_t const *)source);
}

/** @brief Type-agnostic 256-bit full load (Power VSX). */
NK_INTERNAL void nk_load_b256_powervsx_(void const *source, nk_b256_vec_t *destination) {
    destination->vu8x16s[0] = vec_xl(0, (nk_u8_t const *)source);
    destination->vu8x16s[1] = vec_xl(16, (nk_u8_t const *)source);
}

/** @brief Type-agnostic 128-bit full store (Power VSX). */
NK_INTERNAL void nk_store_b128_powervsx_(nk_b128_vec_t const *source, void *destination) {
    vec_xst(source->vu8x16, 0, (nk_u8_t *)destination);
}

/** @brief Type-agnostic 256-bit full store (Power VSX). */
NK_INTERNAL void nk_store_b256_powervsx_(nk_b256_vec_t const *source, void *destination) {
    vec_xst(source->vu8x16s[0], 0, (nk_u8_t *)destination);
    vec_xst(source->vu8x16s[1], 16, (nk_u8_t *)destination);
}

/** @brief Type-agnostic 64-bit load (Power VSX). */
NK_INTERNAL void nk_load_b64_powervsx_(void const *source, nk_b64_vec_t *destination) {
    destination->u64 = *(nk_u64_t const *)source;
}

/** @brief Partial load for 64-bit elements (n elements, max 4) into 256-bit vector.
 *  Uses vec_xl_len to load exactly n×8 bytes, zero-filling the remainder.
 *  vec_xl_len with length=0 produces a zero vector (no branch needed). */
NK_INTERNAL void nk_partial_load_b64x4_powervsx_(void const *source, nk_b256_vec_t *destination, nk_size_t n) {
    nk_size_t bytes = n * 8;
    nk_size_t first_half = bytes < 16 ? bytes : 16;
    nk_size_t second_half = bytes > 16 ? bytes - 16 : 0;
    destination->vu8x16s[0] = vec_xl_len((nk_u8_t *)source, first_half);
    destination->vu8x16s[1] = vec_xl_len((nk_u8_t *)source + 16, second_half);
}

/** @brief Partial load for 64-bit elements (n elements, max 2) into 128-bit vector. */
NK_INTERNAL void nk_partial_load_b64x2_powervsx_(void const *source, nk_b128_vec_t *destination, nk_size_t n) {
    destination->vu8x16 = vec_xl_len((nk_u8_t *)source, n * 8);
}

/** @brief Partial load for 32-bit elements (n elements, max 4) into 128-bit vector. */
NK_INTERNAL void nk_partial_load_b32x4_powervsx_(void const *source, nk_b128_vec_t *destination, nk_size_t n) {
    destination->vu8x16 = vec_xl_len((nk_u8_t *)source, n * 4);
}

/** @brief Partial load for 32-bit elements (n elements, max 2) into 64-bit vector. */
NK_INTERNAL void nk_partial_load_b32x2_powervsx_(void const *source, nk_b64_vec_t *destination, nk_size_t n) {
    nk_copy_bytes_(destination, source, n * 4);
}

/** @brief Partial load for 16-bit elements (n elements, max 8) into 128-bit vector. */
NK_INTERNAL void nk_partial_load_b16x8_powervsx_(void const *source, nk_b128_vec_t *destination, nk_size_t n) {
    destination->vu8x16 = vec_xl_len((nk_u8_t *)source, n * 2);
}

/** @brief Partial load for 8-bit elements (n elements, max 16) into 128-bit vector. */
NK_INTERNAL void nk_partial_load_b8x16_powervsx_(void const *source, nk_b128_vec_t *destination, nk_size_t n) {
    destination->vu8x16 = vec_xl_len((nk_u8_t *)source, n);
}

/** @brief Partial load for 1-bit elements (n bits, max 128) into 128-bit vector. */
NK_INTERNAL void nk_partial_load_b1x128_powervsx_(void const *source, nk_b128_vec_t *destination, nk_size_t n_bits) {
    destination->vu8x16 = vec_xl_len((nk_u8_t *)source, nk_size_divide_round_up_(n_bits, 8));
}

/** @brief Partial store for 64-bit elements (n elements, max 4) from 256-bit vector.
 *  vec_xst_len with length=0 stores nothing (no branch needed). */
NK_INTERNAL void nk_partial_store_b64x4_powervsx_(nk_b256_vec_t const *source, void *destination, nk_size_t n) {
    nk_size_t bytes = n * 8;
    nk_size_t first_half = bytes < 16 ? bytes : 16;
    nk_size_t second_half = bytes > 16 ? bytes - 16 : 0;
    vec_xst_len(source->vu8x16s[0], (nk_u8_t *)destination, first_half);
    vec_xst_len(source->vu8x16s[1], (nk_u8_t *)destination + 16, second_half);
}

/** @brief Partial store for 32-bit elements (n elements, max 4) from 128-bit vector. */
NK_INTERNAL void nk_partial_store_b32x4_powervsx_(nk_b128_vec_t const *source, void *destination, nk_size_t n) {
    vec_xst_len(source->vu8x16, (nk_u8_t *)destination, n * 4);
}

/** @brief Convert 4x f16 → f32x4 via POWER9 hardware (xvcvhpsp, 1 instruction!).
 *  Loads 4 f16 values into a u16x8 register and uses `vec_extract_fp32_from_shorth`. */
NK_INTERNAL nk_vf32x4_t nk_f16x4_to_f32x4_powervsx_(nk_f16_t const *source) {
    nk_vu16x8_t values_u16x8 = (nk_vu16x8_t)vec_xl_len((nk_u8_t *)source, 8);
    return vec_extract_fp32_from_shorth(values_u16x8);
}

/** @brief Convert f32x4 → 4x f16 via POWER9 hardware (xvcvsphp, 1 instruction!).
 *  Uses `vec_pack_to_short_fp32` to pack 4 f32 values into 4 f16 values. */
NK_INTERNAL nk_b64_vec_t nk_f32x4_to_f16x4_powervsx_(nk_vf32x4_t values_f32x4) {
    nk_vu16x8_t packed_u16x8 = vec_pack_to_short_fp32(values_f32x4, values_f32x4);
    nk_b64_vec_t result_vec;
    result_vec.u64 = vec_extract((nk_vu64x2_t)packed_u16x8, 0);
    return result_vec;
}

/** @brief Convert 4x bf16 → f32x4 via branchless bit manipulation (Power VSX).
 *  BF16 format: upper 16 bits of f32. Conversion is zero-extend via vec_mergeh, reinterpret. */
NK_INTERNAL nk_vf32x4_t nk_bf16x4_to_f32x4_powervsx_(nk_bf16_t const *source) {
    nk_vu16x8_t values_u16x8 = (nk_vu16x8_t)vec_xl_len((nk_u8_t *)source, 8);
    nk_vu16x8_t zero_u16x8 = vec_splats((nk_u16_t)0);
    nk_vu32x4_t bits_u32x4 = (nk_vu32x4_t)vec_mergeh(zero_u16x8, values_u16x8);
    return (nk_vf32x4_t)bits_u32x4;
}

/** @brief Convert f32x4 → bf16 packed in u16x8 with RNE rounding (Power VSX).
 *  Round-to-nearest-even: add (0x7FFF + lsb) before truncation.
 *  Uses vec_sr by 16, then vec_pack to narrow u32x4 → u16x8.
 *  Result is in low 4 lanes of the returned u16x8. */
NK_INTERNAL nk_vu16x8_t nk_f32x4_to_bf16_pack_powervsx_(nk_vf32x4_t values_f32x4) {
    nk_vu32x4_t shift_u32x4 = vec_splats((nk_u32_t)16);
    nk_vu32x4_t one_u32x4 = vec_splats((nk_u32_t)1);
    nk_vu32x4_t rounding_base_u32x4 = vec_splats((nk_u32_t)0x7FFF);

    nk_vu32x4_t bits_u32x4 = (nk_vu32x4_t)values_f32x4;

    // RNE rounding: lsb = (bits >> 16) & 1; bits += 0x7FFF + lsb
    nk_vu32x4_t lsb_u32x4 = vec_and(vec_sr(bits_u32x4, shift_u32x4), one_u32x4);
    nk_vu32x4_t rounding_u32x4 = vec_add(rounding_base_u32x4, lsb_u32x4);
    bits_u32x4 = vec_add(bits_u32x4, rounding_u32x4);
    bits_u32x4 = vec_sr(bits_u32x4, shift_u32x4);
    return vec_pack(bits_u32x4, bits_u32x4);
}

/** @brief Convert f32x4 → 4x bf16 with RNE rounding (Power VSX). Returns nk_b64_vec_t. */
NK_INTERNAL nk_b64_vec_t nk_f32x4_to_bf16x4_powervsx_(nk_vf32x4_t values_f32x4) {
    nk_b64_vec_t result_vec;
    result_vec.u64 = vec_extract((nk_vu64x2_t)nk_f32x4_to_bf16_pack_powervsx_(values_f32x4), 0);
    return result_vec;
}

/** @brief Convert 4x i16 → f32x4 (Power VSX). Sign-extend via vec_unpackh, then vec_ctf. */
NK_INTERNAL nk_vf32x4_t nk_i16x4_to_f32x4_powervsx_(nk_i16_t const *source) {
    nk_vi16x8_t values_i16x8 = (nk_vi16x8_t)vec_xl_len((nk_u8_t *)source, 8);
    nk_vi32x4_t values_i32x4 = vec_unpackh(values_i16x8);
    return vec_ctf(values_i32x4, 0);
}

/** @brief Convert 4x u16 → f32x4 (Power VSX). Zero-extend via vec_mergeh with zero, then vec_ctf. */
NK_INTERNAL nk_vf32x4_t nk_u16x4_to_f32x4_powervsx_(nk_u16_t const *source) {
    nk_vu16x8_t values_u16x8 = (nk_vu16x8_t)vec_xl_len((nk_u8_t *)source, 8);
    nk_vu16x8_t zero_u16x8 = vec_splats((nk_u16_t)0);
    nk_vu32x4_t values_u32x4 = (nk_vu32x4_t)vec_mergeh(values_u16x8, zero_u16x8);
    return vec_ctf(values_u32x4, 0);
}

/** @brief Convert 4x i8 → f32x4 (Power VSX). Double unpack via vec_unpackh (i8 → i16 → i32), then vec_ctf. */
NK_INTERNAL nk_vf32x4_t nk_i8x4_to_f32x4_powervsx_(void const *source) {
    nk_vi8x16_t values_i8x16 = (nk_vi8x16_t)vec_xl_len((nk_u8_t *)source, 4);
    nk_vi16x8_t values_i16x8 = vec_unpackh(values_i8x16);
    nk_vi32x4_t values_i32x4 = vec_unpackh(values_i16x8);
    return vec_ctf(values_i32x4, 0);
}

/** @brief Convert 4x u8 → f32x4 (Power VSX). Double merge with zero (u8 → u16 → u32), then vec_ctf. */
NK_INTERNAL nk_vf32x4_t nk_u8x4_to_f32x4_powervsx_(void const *source) {
    nk_vu8x16_t values_u8x16 = (nk_vu8x16_t)vec_xl_len((nk_u8_t *)source, 4);
    nk_vu8x16_t zero_u8x16 = vec_splats((nk_u8_t)0);
    nk_vu16x8_t values_u16x8 = (nk_vu16x8_t)vec_mergeh(values_u8x16, zero_u8x16);
    nk_vu16x8_t zero_u16x8 = vec_splats((nk_u16_t)0);
    nk_vu32x4_t values_u32x4 = (nk_vu32x4_t)vec_mergeh(values_u16x8, zero_u16x8);
    return vec_ctf(values_u32x4, 0);
}

/** @brief Convert f32x4 → 4x i16 with vector saturation (Power VSX).
 *  Uses vec_cts + vec_min/vec_max for clamping, then vec_packs to narrow. */
NK_INTERNAL nk_b64_vec_t nk_f32x4_to_i16x4_powervsx_(nk_vf32x4_t values_f32x4) {
    nk_vi32x4_t min_i32x4 = vec_splats((nk_i32_t)-32768);
    nk_vi32x4_t max_i32x4 = vec_splats((nk_i32_t)32767);

    nk_vi32x4_t values_i32x4 = vec_cts(vec_round(values_f32x4), 0);
    values_i32x4 = vec_max(values_i32x4, min_i32x4);
    values_i32x4 = vec_min(values_i32x4, max_i32x4);

    // Signed saturating pack: i32x4 → i16x8, extract low 8 bytes
    nk_vi16x8_t packed_i16x8 = vec_packs(values_i32x4, values_i32x4);
    nk_b64_vec_t result_vec;
    result_vec.u64 = vec_extract((nk_vu64x2_t)packed_i16x8, 0);
    return result_vec;
}

/** @brief Convert f32x4 → 4x u16 with vector saturation (Power VSX).
 *  Uses vec_ctu + vec_round/vec_max for clamping, then vec_pack to narrow. */
NK_INTERNAL nk_b64_vec_t nk_f32x4_to_u16x4_powervsx_(nk_vf32x4_t values_f32x4) {
    nk_vf32x4_t zero_f32x4 = vec_splats(0.0f);
    nk_vu32x4_t max_u32x4 = vec_splats((nk_u32_t)65535);

    values_f32x4 = vec_max(values_f32x4, zero_f32x4);
    nk_vu32x4_t values_u32x4 = vec_ctu(vec_round(values_f32x4), 0);
    values_u32x4 = vec_min(values_u32x4, max_u32x4);

    // Pack u32x4 → u16x8, extract low 8 bytes
    nk_vu16x8_t packed_u16x8 = vec_pack(values_u32x4, values_u32x4);
    nk_b64_vec_t result_vec;
    result_vec.u64 = vec_extract((nk_vu64x2_t)packed_u16x8, 0);
    return result_vec;
}

/** @brief Convert f32x4 → 4x i8 with vector saturation (Power VSX).
 *  Uses vec_cts + vec_min/vec_max for clamping, then vec_packs twice to narrow. */
NK_INTERNAL nk_b32_vec_t nk_f32x4_to_i8x4_powervsx_(nk_vf32x4_t values_f32x4) {
    nk_vi32x4_t min_i32x4 = vec_splats((nk_i32_t)-128);
    nk_vi32x4_t max_i32x4 = vec_splats((nk_i32_t)127);

    nk_vi32x4_t values_i32x4 = vec_cts(vec_round(values_f32x4), 0);
    values_i32x4 = vec_max(values_i32x4, min_i32x4);
    values_i32x4 = vec_min(values_i32x4, max_i32x4);

    // Narrow: i32x4 → i16x8 → i8x16, extract low 4 bytes
    nk_vi16x8_t packed_i16x8 = vec_packs(values_i32x4, values_i32x4);
    nk_vi8x16_t packed_i8x16 = vec_packs(packed_i16x8, packed_i16x8);
    nk_b32_vec_t result_vec;
    result_vec.u32 = vec_extract((nk_vu32x4_t)packed_i8x16, 0);
    return result_vec;
}

/** @brief Convert f32x4 → 4x u8 with vector saturation (Power VSX).
 *  Uses vec_ctu + vec_min/vec_max for clamping, then vec_pack twice to narrow. */
NK_INTERNAL nk_b32_vec_t nk_f32x4_to_u8x4_powervsx_(nk_vf32x4_t values_f32x4) {
    nk_vf32x4_t zero_f32x4 = vec_splats(0.0f);
    nk_vu32x4_t max_u32x4 = vec_splats((nk_u32_t)255);

    values_f32x4 = vec_max(values_f32x4, zero_f32x4);
    nk_vu32x4_t values_u32x4 = vec_ctu(vec_round(values_f32x4), 0);
    values_u32x4 = vec_min(values_u32x4, max_u32x4);

    // Narrow: u32x4 → u16x8 → u8x16, extract low 4 bytes
    nk_vu16x8_t packed_u16x8 = vec_pack(values_u32x4, values_u32x4);
    nk_vu8x16_t packed_u8x16 = vec_pack(packed_u16x8, packed_u16x8);
    nk_b32_vec_t result_vec;
    result_vec.u32 = vec_extract((nk_vu32x4_t)packed_u8x16, 0);
    return result_vec;
}

NK_PUBLIC void nk_cast_powervsx(void const *from, nk_dtype_t from_type, nk_size_t n, void *to, nk_dtype_t to_type) {
    // Same-type fast path
    if (from_type == to_type) {
        nk_size_t size_bits = nk_dtype_bits(from_type);
        if (size_bits > 0) nk_copy_bytes_(to, from, nk_size_divide_round_up_(n * size_bits, 8));
        return;
    }

    // Validate supported types (f32 and smaller, no FP8 vectorization on Power)
    int from_ok = (from_type == nk_f32_k || from_type == nk_f16_k || from_type == nk_bf16_k || from_type == nk_i8_k ||
                   from_type == nk_u8_k || from_type == nk_i16_k || from_type == nk_u16_k || from_type == nk_i32_k ||
                   from_type == nk_u32_k);
    int to_ok = (to_type == nk_f32_k || to_type == nk_f16_k || to_type == nk_bf16_k || to_type == nk_i8_k ||
                 to_type == nk_u8_k || to_type == nk_i16_k || to_type == nk_u16_k || to_type == nk_i32_k ||
                 to_type == nk_u32_k);

    // Fall back to serial for unsupported types or i32 ↔ u32 (loses precision through f32)
    if (!from_ok || !to_ok || (from_type == nk_i32_k && to_type == nk_u32_k) ||
        (from_type == nk_u32_k && to_type == nk_i32_k)) {
        nk_cast_serial(from, from_type, n, to, to_type);
        return;
    }

    // F32 hub with predicated loads/stores — no serial fallback needed
    nk_size_t from_element_bytes = nk_size_divide_round_up_(nk_dtype_bits(from_type), NK_BITS_PER_BYTE);
    nk_size_t to_element_bytes = nk_size_divide_round_up_(nk_dtype_bits(to_type), NK_BITS_PER_BYTE);
    nk_u8_t const *from_ptr = (nk_u8_t const *)from;
    nk_u8_t *to_ptr = (nk_u8_t *)to;

    for (nk_size_t index = 0; index < n; index += 4) {
        nk_size_t remaining = n - index < 4 ? n - index : 4;
        nk_size_t from_bytes = remaining * from_element_bytes;
        nk_size_t to_bytes = remaining * to_element_bytes;

        // Predicated load → upcast to f32x4 hub
        nk_vu8x16_t raw_u8x16 = vec_xl_len((nk_u8_t *)from_ptr, from_bytes);
        nk_vf32x4_t hub_f32x4;
        switch (from_type) {
        case nk_f32_k: hub_f32x4 = (nk_vf32x4_t)raw_u8x16; break;
        case nk_f16_k: hub_f32x4 = vec_extract_fp32_from_shorth((nk_vu16x8_t)raw_u8x16); break;
        case nk_bf16_k: hub_f32x4 = (nk_vf32x4_t)vec_mergeh(vec_splats((nk_u16_t)0), (nk_vu16x8_t)raw_u8x16); break;
        case nk_i32_k: hub_f32x4 = vec_ctf((nk_vi32x4_t)raw_u8x16, 0); break;
        case nk_u32_k: hub_f32x4 = vec_ctf((nk_vu32x4_t)raw_u8x16, 0); break;
        case nk_i16_k: hub_f32x4 = vec_ctf(vec_unpackh((nk_vi16x8_t)raw_u8x16), 0); break;
        case nk_u16_k:
            hub_f32x4 = vec_ctf((nk_vu32x4_t)vec_mergeh((nk_vu16x8_t)raw_u8x16, vec_splats((nk_u16_t)0)), 0);
            break;
        case nk_i8_k: hub_f32x4 = vec_ctf(vec_unpackh(vec_unpackh((nk_vi8x16_t)raw_u8x16)), 0); break;
        case nk_u8_k:
            hub_f32x4 = vec_ctf((nk_vu32x4_t)vec_mergeh((nk_vu16x8_t)vec_mergeh(raw_u8x16, vec_splats((nk_u8_t)0)),
                                                        vec_splats((nk_u16_t)0)),
                                0);
            break;
        default: hub_f32x4 = vec_splats(0.0f); break;
        }

        // Downcast from f32x4 hub → predicated store
        switch (to_type) {
        case nk_f32_k: vec_xst_len(hub_f32x4, (nk_f32_t *)to_ptr, to_bytes); break;
        case nk_f16_k:
            vec_xst_len((nk_vu8x16_t)vec_pack_to_short_fp32(hub_f32x4, hub_f32x4), (nk_u8_t *)to_ptr, to_bytes);
            break;
        case nk_bf16_k:
            vec_xst_len((nk_vu8x16_t)nk_f32x4_to_bf16_pack_powervsx_(hub_f32x4), (nk_u8_t *)to_ptr, to_bytes);
            break;
        case nk_i32_k: vec_xst_len(vec_cts(vec_round(hub_f32x4), 0), (nk_i32_t *)to_ptr, to_bytes); break;
        case nk_u32_k: vec_xst_len(vec_ctu(vec_round(hub_f32x4), 0), (nk_u32_t *)to_ptr, to_bytes); break;
        case nk_i16_k:
            vec_xst_len((nk_vu8x16_t)vec_packs(vec_cts(vec_round(hub_f32x4), 0), vec_cts(vec_round(hub_f32x4), 0)),
                        (nk_u8_t *)to_ptr, to_bytes);
            break;
        case nk_u16_k:
            vec_xst_len((nk_vu8x16_t)vec_pack(vec_ctu(vec_round(hub_f32x4), 0), vec_ctu(vec_round(hub_f32x4), 0)),
                        (nk_u8_t *)to_ptr, to_bytes);
            break;
        case nk_i8_k:
            vec_xst_len(
                (nk_vu8x16_t)vec_packs(vec_packs(vec_cts(vec_round(hub_f32x4), 0), vec_cts(vec_round(hub_f32x4), 0)),
                                       vec_packs(vec_cts(vec_round(hub_f32x4), 0), vec_cts(vec_round(hub_f32x4), 0))),
                (nk_u8_t *)to_ptr, to_bytes);
            break;
        case nk_u8_k:
            vec_xst_len(
                (nk_vu8x16_t)vec_pack(vec_pack(vec_ctu(vec_round(hub_f32x4), 0), vec_ctu(vec_round(hub_f32x4), 0)),
                                      vec_pack(vec_ctu(vec_round(hub_f32x4), 0), vec_ctu(vec_round(hub_f32x4), 0))),
                (nk_u8_t *)to_ptr, to_bytes);
            break;
        default: break;
        }

        from_ptr += from_bytes;
        to_ptr += to_bytes;
    }
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_POWERVSX
#endif // NK_TARGET_POWER64_
#endif // NK_CAST_POWERVSX_H
