/**
 *  @brief SIMD-accelerated Dot Products for POWER9 VSX.
 *  @file include/numkong/dot/powervsx.h
 *  @author Ash Vardanian
 *  @date March 23, 2026
 *
 *  @sa include/numkong/dot.h
 *
 *  @section dot_powervsx_instructions Power9 VSX Dot Product Instructions
 *
 *  Key Power9 VSX instructions for dot products:
 *
 *      Intrinsic                        Instruction           POWER9
 *      vec_madd(a, b, c)                XVMADDADP/XVMADDASP   5cy  FMA: a×b+c
 *      vec_msub(a, b, c)                XVMSUBADP/XVMSUBASP   5cy  FMS: a×b−c
 *      vec_msum(a, b, c)                VMSUMUBM/VMSUMMBM     5cy  i8/u8 widening multiply-sum → i32/u32
 *      vec_msum(a, b, c)                VMSUMSHM/VMSUMUHM     5cy  i16/u16 widening multiply-sum → i32/u32
 *      vec_doublee(a)                   XVCVSPDP              3cy  Widen even f32 lanes → f64x2
 *      vec_doubleo(a)                   XVCVSPDP (odd)        3cy  Widen odd f32 lanes → f64x2
 *      vec_unpackh(a)                   VUPKHSB/VUPKHSH       2cy  Sign-extend high half (i8→i16 or i16→i32)
 *      vec_unpackl(a)                   VUPKLSB/VUPKLSH       2cy  Sign-extend low half (i8→i16 or i16→i32)
 *      vec_xor(a, b)                    VXOR/XXLXOR           1cy  Bitwise XOR
 *      vec_xl(off, ptr)                 LXV                   5cy  Aligned 16-byte load
 *      vec_xl_len(ptr, len)             LXVL                  5cy  Partial load (Power9), zero-fills tail
 *      vec_extract_fp32_from_shorth     XVCVHPSP (high)       5cy  f16x4 → f32x4 from high half
 *      vec_extract_fp32_from_shortl     XVCVHPSP (low)        5cy  f16x4 → f32x4 from low half
 *      vec_popcnt(a)                    VPOPCNTB/H/W/D        2cy  Per-element popcount
 *      vec_sum4s(a, b)                  VSUM4UBS/VSUM4SBS     5cy  Sum groups of 4 bytes → i32/u32
 *      vec_sums(a, b)                   VSUMSWS               5cy  Signed i32x4 horizontal → i32 (lane 3)
 *
 *  Power9 (POWER ISA 3.0) provides `vec_xl_len` for partial loads that zero-fill unused bytes,
 *  enabling branchless tail handling: zero × anything = zero, so partial vectors contribute
 *  no spurious terms to dot-product accumulators.
 *
 *  @section dot_powervsx_stateful Stateful Streaming Logic
 *
 *  For memory-optimal tiled algorithms, this file defines state structures and force-inlined
 *  `NK_INTERNAL` functions:
 *
 *  - nk_dot_f32x2 state for f32 inputs with double-precision accumulation,
 *  - nk_dot_f64x2 state with Dot2 stable dot-products for f64 inputs,
 *  - nk_dot_bf16x8 state for bf16 inputs with f32 accumulation,
 *  - nk_dot_f16x8 state for f16 inputs with f32 accumulation,
 *  - nk_dot_i8x16 state for i8 inputs with i32 accumulation,
 *  - nk_dot_u8x16 state for u8 inputs with u32 accumulation,
 *  - nk_dot_u1x128 state for binary inputs with u64 popcount accumulation.
 */
#ifndef NK_DOT_POWERVSX_H
#define NK_DOT_POWERVSX_H

#if NK_TARGET_POWERVSX

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("power9-vector"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("power9-vector")
#endif

/** @brief Horizontal sum of 4 f32 lanes → scalar f32. */
NK_INTERNAL nk_f32_t nk_hsum_f32x4_powervsx_(nk_vf32x4_t values_f32x4) {
    // Rotate by 8 bytes (2 floats) and add → {v[0]+v[2], v[1]+v[3], ...}
    nk_vf32x4_t rotated_f32x4 = vec_sld(values_f32x4, values_f32x4, 8);
    nk_vf32x4_t partial_f32x4 = vec_add(values_f32x4, rotated_f32x4);
    // Rotate by 4 bytes (1 float) and add → {v[0]+v[1]+v[2]+v[3], ...}
    nk_vf32x4_t shifted_f32x4 = vec_sld(partial_f32x4, partial_f32x4, 4);
    nk_vf32x4_t total_f32x4 = vec_add(partial_f32x4, shifted_f32x4);
    return vec_extract(total_f32x4, 0);
}

/** @brief Horizontal sum of 2 f64 lanes → scalar f64 via xxpermdi (1 domain crossing). */
NK_INTERNAL nk_f64_t nk_hsum_f64x2_powervsx_(nk_vf64x2_t values_f64x2) {
    nk_vf64x2_t swapped_f64x2 = vec_xxpermdi(values_f64x2, values_f64x2, 2);
    nk_vf64x2_t sum_f64x2 = vec_add(values_f64x2, swapped_f64x2);
    return vec_extract(sum_f64x2, 0);
}

/** @brief Horizontal sum of 4 signed i32 lanes → scalar i32. */
NK_INTERNAL nk_i32_t nk_hsum_i32x4_powervsx_(nk_vi32x4_t values_i32x4) {
    // vec_sums reduces i32x4 → i32 in lane 3 of the result
    nk_vi32x4_t zero_i32x4 = vec_splats((nk_i32_t)0);
    nk_vi32x4_t sums_i32x4 = vec_sums(values_i32x4, zero_i32x4);
    return vec_extract(sums_i32x4, 3);
}

/** @brief Horizontal sum of 4 unsigned u32 lanes → scalar u32. */
NK_INTERNAL nk_u32_t nk_hsum_u32x4_powervsx_(nk_vu32x4_t values_u32x4) {
    // Rotate by 8 bytes (2 ints) and add → {v[0]+v[2], v[1]+v[3], ...}
    nk_vu32x4_t rotated_u32x4 = vec_sld(values_u32x4, values_u32x4, 8);
    nk_vu32x4_t partial_u32x4 = vec_add(values_u32x4, rotated_u32x4);
    // Rotate by 4 bytes (1 int) and add → {v[0]+v[1]+v[2]+v[3], ...}
    nk_vu32x4_t shifted_u32x4 = vec_sld(partial_u32x4, partial_u32x4, 4);
    nk_vu32x4_t total_u32x4 = vec_add(partial_u32x4, shifted_u32x4);
    return vec_extract(total_u32x4, 0);
}

/** @brief Horizontal sum of 2 unsigned u64 lanes → scalar u64 via xxpermdi. */
NK_INTERNAL nk_u64_t nk_hsum_u64x2_powervsx_(nk_vu64x2_t values_u64x2) {
    nk_vu64x2_t swapped_u64x2 = vec_xxpermdi(values_u64x2, values_u64x2, 2);
    nk_vu64x2_t sum_u64x2 = vec_add(values_u64x2, swapped_u64x2);
    return vec_extract(sum_u64x2, 0);
}

/** @brief Compensated horizontal sum of 2 f64 lanes via TwoSum. */
NK_INTERNAL nk_f64_t nk_dot_stable_sum_f64x2_powervsx_(nk_vf64x2_t sum_f64x2, nk_vf64x2_t compensation_f64x2) {
    // TwoSum merge of sum + compensation (2-wide)
    nk_vf64x2_t tentative_sum_f64x2 = vec_add(sum_f64x2, compensation_f64x2);
    nk_vf64x2_t virtual_addend_f64x2 = vec_sub(tentative_sum_f64x2, sum_f64x2);
    nk_vf64x2_t rounding_error_f64x2 = vec_add(vec_sub(sum_f64x2, vec_sub(tentative_sum_f64x2, virtual_addend_f64x2)),
                                               vec_sub(compensation_f64x2, virtual_addend_f64x2));
    // Scalar TwoSum 2 → 1
    nk_f64_t lower_sum = vec_extract(tentative_sum_f64x2, 0);
    nk_f64_t upper_sum = vec_extract(tentative_sum_f64x2, 1);
    nk_f64_t lower_error = vec_extract(rounding_error_f64x2, 0);
    nk_f64_t upper_error = vec_extract(rounding_error_f64x2, 1);
    nk_f64_t tentative_sum = lower_sum + upper_sum;
    nk_f64_t virtual_addend = tentative_sum - lower_sum;
    nk_f64_t rounding_error = (lower_sum - (tentative_sum - virtual_addend)) + (upper_sum - virtual_addend);
    return tentative_sum + (lower_error + upper_error + rounding_error);
}

#pragma region F32 and F64 Floats

NK_PUBLIC void nk_dot_f32_powervsx(nk_f32_t const *a_scalars, nk_f32_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f64_t *result) {
    // Upcast f32 → f64 for accumulation via vec_doublee (even lanes) and vec_doubleo (odd lanes)
    nk_vf64x2_t sum_even_f64x2 = vec_splats((nk_f64_t)0);
    nk_vf64x2_t sum_odd_f64x2 = vec_splats((nk_f64_t)0);
    nk_vf32x4_t a_f32x4, b_f32x4;
    nk_size_t tail_bytes;

nk_dot_f32_powervsx_cycle:
    if (count_scalars < 4) {
        tail_bytes = count_scalars * sizeof(nk_f32_t);
        a_f32x4 = vec_xl_len((nk_f32_t *)a_scalars, tail_bytes);
        b_f32x4 = vec_xl_len((nk_f32_t *)b_scalars, tail_bytes);
        count_scalars = 0;
    }
    else {
        a_f32x4 = vec_xl(0, a_scalars);
        b_f32x4 = vec_xl(0, b_scalars);
        a_scalars += 4, b_scalars += 4, count_scalars -= 4;
    }

    // Widen even/odd f32 lanes → f64x2, then FMA
    nk_vf64x2_t a_even_f64x2 = vec_doublee(a_f32x4);
    nk_vf64x2_t b_even_f64x2 = vec_doublee(b_f32x4);
    nk_vf64x2_t a_odd_f64x2 = vec_doubleo(a_f32x4);
    nk_vf64x2_t b_odd_f64x2 = vec_doubleo(b_f32x4);
    sum_even_f64x2 = vec_madd(a_even_f64x2, b_even_f64x2, sum_even_f64x2);
    sum_odd_f64x2 = vec_madd(a_odd_f64x2, b_odd_f64x2, sum_odd_f64x2);

    if (count_scalars) goto nk_dot_f32_powervsx_cycle;
    // Combine even and odd accumulators → final scalar
    nk_vf64x2_t total_f64x2 = vec_add(sum_even_f64x2, sum_odd_f64x2);
    *result = nk_hsum_f64x2_powervsx_(total_f64x2);
}

NK_PUBLIC void nk_dot_f64_powervsx(nk_f64_t const *a_scalars, nk_f64_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f64_t *result) {
    // Dot2 algorithm (Ogita-Rump-Oishi 2005) for compensated dot product
    nk_vf64x2_t sum_f64x2 = vec_splats((nk_f64_t)0);
    nk_vf64x2_t compensation_f64x2 = vec_splats((nk_f64_t)0);
    nk_vf64x2_t a_f64x2, b_f64x2;
    nk_size_t tail_bytes;

nk_dot_f64_powervsx_cycle:
    if (count_scalars < 2) {
        tail_bytes = count_scalars * sizeof(nk_f64_t);
        a_f64x2 = vec_xl_len((nk_f64_t *)a_scalars, tail_bytes);
        b_f64x2 = vec_xl_len((nk_f64_t *)b_scalars, tail_bytes);
        count_scalars = 0;
    }
    else {
        a_f64x2 = vec_xl(0, a_scalars);
        b_f64x2 = vec_xl(0, b_scalars);
        a_scalars += 2, b_scalars += 2, count_scalars -= 2;
    }

    // TwoProd: product = a * b, error = msub(a, b, product) captures rounding error
    nk_vf64x2_t product_f64x2 = vec_mul(a_f64x2, b_f64x2);
    nk_vf64x2_t product_error_f64x2 = vec_msub(a_f64x2, b_f64x2, product_f64x2);
    // TwoSum: (t, q) = TwoSum(sum, product) where t = sum + product rounded, q = error
    nk_vf64x2_t tentative_sum_f64x2 = vec_add(sum_f64x2, product_f64x2);
    nk_vf64x2_t virtual_addend_f64x2 = vec_sub(tentative_sum_f64x2, sum_f64x2);
    nk_vf64x2_t sum_error_f64x2 = vec_add(vec_sub(sum_f64x2, vec_sub(tentative_sum_f64x2, virtual_addend_f64x2)),
                                          vec_sub(product_f64x2, virtual_addend_f64x2));
    // Update: sum = t, compensation += q + r
    sum_f64x2 = tentative_sum_f64x2;
    compensation_f64x2 = vec_add(compensation_f64x2, vec_add(sum_error_f64x2, product_error_f64x2));

    if (count_scalars) goto nk_dot_f64_powervsx_cycle;
    // Compensated horizontal reduction preserving Dot2 error tracking
    *result = nk_dot_stable_sum_f64x2_powervsx_(sum_f64x2, compensation_f64x2);
}

#pragma endregion F32 and F64 Floats
#pragma region F16 and BF16 Floats

NK_PUBLIC void nk_dot_bf16_powervsx(nk_bf16_t const *a_scalars, nk_bf16_t const *b_scalars, nk_size_t count_scalars,
                                    nk_f32_t *result) {
    // bf16 → f32 via mergeh/mergel with zero: shift 16 bits into f32 upper half
    nk_vu16x8_t zero_u16x8 = vec_splats((nk_u16_t)0);
    nk_vf32x4_t sum_f32x4 = vec_splats((nk_f32_t)0);
    nk_vu16x8_t a_u16x8, b_u16x8;
    nk_size_t tail_bytes;

nk_dot_bf16_powervsx_cycle:
    if (count_scalars < 8) {
        tail_bytes = count_scalars * sizeof(nk_bf16_t);
        a_u16x8 = vec_xl_len((nk_u16_t *)a_scalars, tail_bytes);
        b_u16x8 = vec_xl_len((nk_u16_t *)b_scalars, tail_bytes);
        count_scalars = 0;
    }
    else {
        a_u16x8 = vec_xl(0, (nk_u16_t const *)a_scalars);
        b_u16x8 = vec_xl(0, (nk_u16_t const *)b_scalars);
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }

    // Convert bf16 → f32: merge with zero puts bf16 bits in upper 16 of each f32
    nk_vf32x4_t a_high_f32x4 = (nk_vf32x4_t)vec_mergeh(zero_u16x8, a_u16x8);
    nk_vf32x4_t a_low_f32x4 = (nk_vf32x4_t)vec_mergel(zero_u16x8, a_u16x8);
    nk_vf32x4_t b_high_f32x4 = (nk_vf32x4_t)vec_mergeh(zero_u16x8, b_u16x8);
    nk_vf32x4_t b_low_f32x4 = (nk_vf32x4_t)vec_mergel(zero_u16x8, b_u16x8);
    sum_f32x4 = vec_madd(a_high_f32x4, b_high_f32x4, sum_f32x4);
    sum_f32x4 = vec_madd(a_low_f32x4, b_low_f32x4, sum_f32x4);

    if (count_scalars) goto nk_dot_bf16_powervsx_cycle;
    *result = nk_hsum_f32x4_powervsx_(sum_f32x4);
}

NK_PUBLIC void nk_dot_f16_powervsx(nk_f16_t const *a_scalars, nk_f16_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    // f16 → f32 via vec_extract_fp32_from_shorth/shortl (Power9 XVCVHPSP)
    nk_vf32x4_t sum_f32x4 = vec_splats((nk_f32_t)0);
    nk_vu16x8_t a_u16x8, b_u16x8;
    nk_size_t tail_bytes;

nk_dot_f16_powervsx_cycle:
    if (count_scalars < 8) {
        tail_bytes = count_scalars * sizeof(nk_f16_t);
        a_u16x8 = vec_xl_len((nk_u16_t *)a_scalars, tail_bytes);
        b_u16x8 = vec_xl_len((nk_u16_t *)b_scalars, tail_bytes);
        count_scalars = 0;
    }
    else {
        a_u16x8 = vec_xl(0, (nk_u16_t const *)a_scalars);
        b_u16x8 = vec_xl(0, (nk_u16_t const *)b_scalars);
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }

    // Convert f16 → f32 via hardware XVCVHPSP
    nk_vf32x4_t a_high_f32x4 = vec_extract_fp32_from_shorth(a_u16x8);
    nk_vf32x4_t a_low_f32x4 = vec_extract_fp32_from_shortl(a_u16x8);
    nk_vf32x4_t b_high_f32x4 = vec_extract_fp32_from_shorth(b_u16x8);
    nk_vf32x4_t b_low_f32x4 = vec_extract_fp32_from_shortl(b_u16x8);
    sum_f32x4 = vec_madd(a_high_f32x4, b_high_f32x4, sum_f32x4);
    sum_f32x4 = vec_madd(a_low_f32x4, b_low_f32x4, sum_f32x4);

    if (count_scalars) goto nk_dot_f16_powervsx_cycle;
    *result = nk_hsum_f32x4_powervsx_(sum_f32x4);
}

#pragma endregion F16 and BF16 Floats
#pragma region I8 and U8 Integers

NK_PUBLIC void nk_dot_i8_powervsx(nk_i8_t const *a_scalars, nk_i8_t const *b_scalars, nk_size_t count_scalars,
                                  nk_i32_t *result) {
    // Algebraic transform for i8×i8 using VMSUMMBM (i8×u8 → i32):
    //   b' = b ⊕ 0x80  (reinterpret signed as unsigned)
    //   a·b = a·b' − 128·Σa
    // Σ(a+128) accumulated via VSUM4UBS; correction applied after loop.
    // Tail handling is free: vec_xl_len zero-fills unused lanes.
    //   - Product: 0 × (0⊕0x80) = 0 → no spurious contribution
    //   - Correction: (0⊕0x80) = 128 in sum_a_biased, compensated by count_padded
    nk_vu8x16_t const bias_u8x16 = vec_splats((nk_u8_t)0x80);
    nk_vi32x4_t accumulator_i32x4 = vec_splats((nk_i32_t)0);
    nk_vu32x4_t sum_a_biased_u32x4 = vec_splats((nk_u32_t)0);
    nk_size_t count_padded = ((count_scalars + 15) / 16) * 16;
    nk_vi8x16_t a_i8x16;
    nk_vu8x16_t b_biased_u8x16;
    nk_size_t tail_bytes;

nk_dot_i8_powervsx_cycle:
    if (count_scalars < 16) {
        tail_bytes = count_scalars * sizeof(nk_i8_t);
        a_i8x16 = vec_xl_len((nk_i8_t *)a_scalars, tail_bytes);
        b_biased_u8x16 = vec_xor(vec_xl_len((nk_u8_t *)b_scalars, tail_bytes), bias_u8x16);
        count_scalars = 0;
    }
    else {
        a_i8x16 = vec_xl(0, a_scalars);
        b_biased_u8x16 = vec_xor(vec_xl(0, (nk_u8_t *)b_scalars), bias_u8x16);
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }

    // VMSUMMBM: i8 × u8 → i32 (16 products per instruction)
    accumulator_i32x4 = vec_msum(a_i8x16, b_biased_u8x16, accumulator_i32x4);
    // VSUM4UBS: accumulate Σ(a+128) as unsigned (independent chain, good ILP)
    sum_a_biased_u32x4 = vec_sum4s(vec_xor((nk_vu8x16_t)a_i8x16, bias_u8x16), sum_a_biased_u32x4);

    if (count_scalars) goto nk_dot_i8_powervsx_cycle;

    // Correction: a·b = biased_dot − 128·Σa = biased_dot − 128·(Σ(a+128) − 128·count_padded)
    nk_i32_t biased_dot = nk_hsum_i32x4_powervsx_(accumulator_i32x4);
    nk_i64_t correction = 128LL * (nk_i64_t)nk_hsum_u32x4_powervsx_(sum_a_biased_u32x4) -
                          16384LL * (nk_i64_t)count_padded;
    *result = (nk_i32_t)((nk_i64_t)biased_dot - correction);
}

NK_PUBLIC void nk_dot_u8_powervsx(nk_u8_t const *a_scalars, nk_u8_t const *b_scalars, nk_size_t count_scalars,
                                  nk_u32_t *result) {
    // vec_msum: multiply u8×u8 pairs and accumulate 16 products → 4 u32 lanes per call
    nk_vu32x4_t accumulator_u32x4 = vec_splats((nk_u32_t)0);
    nk_vu8x16_t a_u8x16, b_u8x16;
    nk_size_t tail_bytes;

nk_dot_u8_powervsx_cycle:
    if (count_scalars < 16) {
        tail_bytes = count_scalars * sizeof(nk_u8_t);
        a_u8x16 = vec_xl_len((nk_u8_t *)a_scalars, tail_bytes);
        b_u8x16 = vec_xl_len((nk_u8_t *)b_scalars, tail_bytes);
        count_scalars = 0;
    }
    else {
        a_u8x16 = vec_xl(0, a_scalars);
        b_u8x16 = vec_xl(0, b_scalars);
        a_scalars += 16, b_scalars += 16, count_scalars -= 16;
    }

    // Unsigned × unsigned multiply-sum: 16 u8 products accumulated into 4 u32 lanes
    accumulator_u32x4 = vec_msum(a_u8x16, b_u8x16, accumulator_u32x4);

    if (count_scalars) goto nk_dot_u8_powervsx_cycle;
    *result = nk_hsum_u32x4_powervsx_(accumulator_u32x4);
}

#pragma endregion I8 and U8 Integers
#pragma region Binary

NK_PUBLIC void nk_dot_u1_powervsx(nk_u1x8_t const *a, nk_u1x8_t const *b, nk_size_t n_bits, nk_u32_t *result) {
    nk_size_t n_bytes = nk_size_divide_round_up_(n_bits, NK_BITS_PER_BYTE);
    nk_vu64x2_t accumulator_u64x2 = vec_splats((nk_u64_t)0);
    nk_vu8x16_t a_u8x16, b_u8x16;

nk_dot_u1_powervsx_cycle:
    if (n_bytes < 16) {
        a_u8x16 = vec_xl_len((nk_u8_t *)a, n_bytes);
        b_u8x16 = vec_xl_len((nk_u8_t *)b, n_bytes);
        n_bytes = 0;
    }
    else {
        a_u8x16 = vec_xl(0, (nk_u8_t const *)a);
        b_u8x16 = vec_xl(0, (nk_u8_t const *)b);
        a += 16, b += 16, n_bytes -= 16;
    }

    // AND → doubleword popcount (vpopcntd) → accumulate u64 lanes
    nk_vu8x16_t and_u8x16 = vec_and(a_u8x16, b_u8x16);
    nk_vu64x2_t popcnt_u64x2 = vec_popcnt((nk_vu64x2_t)and_u8x16);
    accumulator_u64x2 = vec_add(accumulator_u64x2, popcnt_u64x2);

    if (n_bytes) goto nk_dot_u1_powervsx_cycle;
    *result = (nk_u32_t)nk_hsum_u64x2_powervsx_(accumulator_u64x2);
}

#pragma endregion Binary

/**
 *  @brief Running state for 128-bit dot accumulation over f32 scalars on Power VSX.
 *
 *  Processes 2 f32 values at a time, upcasting to f64 for accumulation to avoid
 *  catastrophic cancellation in long reductions.
 */
typedef struct nk_dot_f32x2_state_powervsx_t {
    nk_vf64x2_t sum_f64x2;
} nk_dot_f32x2_state_powervsx_t;

NK_INTERNAL void nk_dot_f32x2_init_powervsx(nk_dot_f32x2_state_powervsx_t *state) {
    state->sum_f64x2 = vec_splats((nk_f64_t)0);
}

NK_INTERNAL void nk_dot_f32x2_update_powervsx(nk_dot_f32x2_state_powervsx_t *state, nk_b64_vec_t a, nk_b64_vec_t b,
                                              nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    // Load 8 bytes (2 f32s) into a vector register, zero-filling the upper 8 bytes
    nk_vf32x4_t a_f32x4 = vec_xl_len((nk_f32_t *)a.f32s, 8);
    nk_vf32x4_t b_f32x4 = vec_xl_len((nk_f32_t *)b.f32s, 8);
    // Widen even lanes (the two f32 values) → f64x2
    nk_vf64x2_t a_f64x2 = vec_doublee(a_f32x4);
    nk_vf64x2_t b_f64x2 = vec_doublee(b_f32x4);
    // Permute to get {lane0, lane2} → {a[0], a[1]} as f64x2
    a_f64x2 = vec_xxpermdi(a_f64x2, vec_doubleo(a_f32x4), 0);
    b_f64x2 = vec_xxpermdi(b_f64x2, vec_doubleo(b_f32x4), 0);
    state->sum_f64x2 = vec_madd(a_f64x2, b_f64x2, state->sum_f64x2);
}

NK_INTERNAL void nk_dot_f32x2_finalize_powervsx(                                                //
    nk_dot_f32x2_state_powervsx_t const *state_a, nk_dot_f32x2_state_powervsx_t const *state_b, //
    nk_dot_f32x2_state_powervsx_t const *state_c, nk_dot_f32x2_state_powervsx_t const *state_d, //
    nk_size_t total_dimensions, nk_b256_vec_t *result) {
    nk_unused_(total_dimensions);
    nk_vf64x2_t sum_a_f64x2 = vec_add(state_a->sum_f64x2, vec_xxpermdi(state_a->sum_f64x2, state_a->sum_f64x2, 2));
    nk_vf64x2_t sum_b_f64x2 = vec_add(state_b->sum_f64x2, vec_xxpermdi(state_b->sum_f64x2, state_b->sum_f64x2, 2));
    nk_vf64x2_t sum_c_f64x2 = vec_add(state_c->sum_f64x2, vec_xxpermdi(state_c->sum_f64x2, state_c->sum_f64x2, 2));
    nk_vf64x2_t sum_d_f64x2 = vec_add(state_d->sum_f64x2, vec_xxpermdi(state_d->sum_f64x2, state_d->sum_f64x2, 2));
    result->vf64x2s[0] = vec_xxpermdi(sum_a_f64x2, sum_b_f64x2, 0);
    result->vf64x2s[1] = vec_xxpermdi(sum_c_f64x2, sum_d_f64x2, 0);
}

/**
 *  @brief Running state for 128-bit dot accumulation over f64 scalars on Power VSX.
 *
 *  Uses the Dot2 algorithm (Ogita-Rump-Oishi 2005) for compensated dot product.
 */
typedef struct nk_dot_f64x2_state_powervsx_t {
    nk_vf64x2_t sum_f64x2;
    nk_vf64x2_t compensation_f64x2;
} nk_dot_f64x2_state_powervsx_t;

NK_INTERNAL void nk_dot_f64x2_init_powervsx(nk_dot_f64x2_state_powervsx_t *state) {
    state->sum_f64x2 = vec_splats((nk_f64_t)0);
    state->compensation_f64x2 = vec_splats((nk_f64_t)0);
}

NK_INTERNAL void nk_dot_f64x2_update_powervsx(nk_dot_f64x2_state_powervsx_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                              nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    nk_vf64x2_t sum_f64x2 = state->sum_f64x2;
    nk_vf64x2_t compensation_f64x2 = state->compensation_f64x2;
    nk_vf64x2_t a_f64x2 = a.vf64x2;
    nk_vf64x2_t b_f64x2 = b.vf64x2;

    // TwoProd: product = a × b, error = msub(a, b, product) captures rounding error
    nk_vf64x2_t product_f64x2 = vec_mul(a_f64x2, b_f64x2);
    nk_vf64x2_t product_error_f64x2 = vec_msub(a_f64x2, b_f64x2, product_f64x2);

    // TwoSum: (t, q) = TwoSum(sum, product) where t = sum + product rounded, q = error
    nk_vf64x2_t tentative_sum_f64x2 = vec_add(sum_f64x2, product_f64x2);
    nk_vf64x2_t virtual_addend_f64x2 = vec_sub(tentative_sum_f64x2, sum_f64x2);
    nk_vf64x2_t sum_error_f64x2 = vec_add(vec_sub(sum_f64x2, vec_sub(tentative_sum_f64x2, virtual_addend_f64x2)),
                                          vec_sub(product_f64x2, virtual_addend_f64x2));

    // Update: sum = t, compensation += q + r
    state->sum_f64x2 = tentative_sum_f64x2;
    state->compensation_f64x2 = vec_add(compensation_f64x2, vec_add(sum_error_f64x2, product_error_f64x2));
}

NK_INTERNAL void nk_dot_f64x2_finalize_powervsx(                                                //
    nk_dot_f64x2_state_powervsx_t const *state_a, nk_dot_f64x2_state_powervsx_t const *state_b, //
    nk_dot_f64x2_state_powervsx_t const *state_c, nk_dot_f64x2_state_powervsx_t const *state_d, //
    nk_size_t total_dimensions, nk_b256_vec_t *result) {
    nk_unused_(total_dimensions);
    // Compensated horizontal reduction preserving Dot2 error tracking per state
    result->f64s[0] = nk_dot_stable_sum_f64x2_powervsx_(state_a->sum_f64x2, state_a->compensation_f64x2);
    result->f64s[1] = nk_dot_stable_sum_f64x2_powervsx_(state_b->sum_f64x2, state_b->compensation_f64x2);
    result->f64s[2] = nk_dot_stable_sum_f64x2_powervsx_(state_c->sum_f64x2, state_c->compensation_f64x2);
    result->f64s[3] = nk_dot_stable_sum_f64x2_powervsx_(state_d->sum_f64x2, state_d->compensation_f64x2);
}

/**
 *  @brief Running state for 128-bit dot accumulation over bf16 scalars on Power VSX.
 *
 *  Processes 8 bf16 values at a time (128 bits), converting to f32 via vec_mergeh/mergel
 *  with zero for accumulation.
 */
typedef struct nk_dot_bf16x8_state_powervsx_t {
    nk_vf32x4_t sum_f32x4;
} nk_dot_bf16x8_state_powervsx_t;

NK_INTERNAL void nk_dot_bf16x8_init_powervsx(nk_dot_bf16x8_state_powervsx_t *state) {
    state->sum_f32x4 = vec_splats((nk_f32_t)0);
}

NK_INTERNAL void nk_dot_bf16x8_update_powervsx(nk_dot_bf16x8_state_powervsx_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                               nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    // Convert bf16 → f32 inline: merge with zero puts bf16 bits in upper 16 of each f32
    nk_vu16x8_t zero_u16x8 = vec_splats((nk_u16_t)0);
    nk_vu16x8_t a_u16x8 = a.vu16x8;
    nk_vu16x8_t b_u16x8 = b.vu16x8;
    nk_vf32x4_t a_high_f32x4 = (nk_vf32x4_t)vec_mergeh(zero_u16x8, a_u16x8);
    nk_vf32x4_t a_low_f32x4 = (nk_vf32x4_t)vec_mergel(zero_u16x8, a_u16x8);
    nk_vf32x4_t b_high_f32x4 = (nk_vf32x4_t)vec_mergeh(zero_u16x8, b_u16x8);
    nk_vf32x4_t b_low_f32x4 = (nk_vf32x4_t)vec_mergel(zero_u16x8, b_u16x8);
    state->sum_f32x4 = vec_madd(a_high_f32x4, b_high_f32x4, state->sum_f32x4);
    state->sum_f32x4 = vec_madd(a_low_f32x4, b_low_f32x4, state->sum_f32x4);
}

NK_INTERNAL void nk_dot_bf16x8_finalize_powervsx(                                                 //
    nk_dot_bf16x8_state_powervsx_t const *state_a, nk_dot_bf16x8_state_powervsx_t const *state_b, //
    nk_dot_bf16x8_state_powervsx_t const *state_c, nk_dot_bf16x8_state_powervsx_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    nk_vf32x4_t a_f32x4 = state_a->sum_f32x4, b_f32x4 = state_b->sum_f32x4, c_f32x4 = state_c->sum_f32x4,
                d_f32x4 = state_d->sum_f32x4;
    nk_vf32x4_t transpose_ab_low_f32x4 = vec_mergeh(a_f32x4, b_f32x4);
    nk_vf32x4_t transpose_cd_low_f32x4 = vec_mergeh(c_f32x4, d_f32x4);
    nk_vf32x4_t transpose_ab_high_f32x4 = vec_mergel(a_f32x4, b_f32x4);
    nk_vf32x4_t transpose_cd_high_f32x4 = vec_mergel(c_f32x4, d_f32x4);
    nk_vf32x4_t sum_lane0_f32x4 = (nk_vf32x4_t)vec_xxpermdi((nk_vu64x2_t)transpose_ab_low_f32x4,
                                                            (nk_vu64x2_t)transpose_cd_low_f32x4, 0);
    nk_vf32x4_t sum_lane1_f32x4 = (nk_vf32x4_t)vec_xxpermdi((nk_vu64x2_t)transpose_ab_low_f32x4,
                                                            (nk_vu64x2_t)transpose_cd_low_f32x4, 3);
    nk_vf32x4_t sum_lane2_f32x4 = (nk_vf32x4_t)vec_xxpermdi((nk_vu64x2_t)transpose_ab_high_f32x4,
                                                            (nk_vu64x2_t)transpose_cd_high_f32x4, 0);
    nk_vf32x4_t sum_lane3_f32x4 = (nk_vf32x4_t)vec_xxpermdi((nk_vu64x2_t)transpose_ab_high_f32x4,
                                                            (nk_vu64x2_t)transpose_cd_high_f32x4, 3);
    result->vf32x4 = vec_add(vec_add(sum_lane0_f32x4, sum_lane1_f32x4), vec_add(sum_lane2_f32x4, sum_lane3_f32x4));
}

/**
 *  @brief Running state for 128-bit dot accumulation over f16 scalars on Power VSX.
 *
 *  Processes 8 f16 values at a time (128 bits), converting to f32 via
 *  vec_extract_fp32_from_shorth/shortl for accumulation.
 */
typedef struct nk_dot_f16x8_state_powervsx_t {
    nk_vf32x4_t sum_f32x4;
} nk_dot_f16x8_state_powervsx_t;

NK_INTERNAL void nk_dot_f16x8_init_powervsx(nk_dot_f16x8_state_powervsx_t *state) {
    state->sum_f32x4 = vec_splats((nk_f32_t)0);
}

NK_INTERNAL void nk_dot_f16x8_update_powervsx(nk_dot_f16x8_state_powervsx_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                              nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    // Convert f16 → f32 via hardware XVCVHPSP
    nk_vu16x8_t a_u16x8 = a.vu16x8;
    nk_vu16x8_t b_u16x8 = b.vu16x8;
    nk_vf32x4_t a_high_f32x4 = vec_extract_fp32_from_shorth(a_u16x8);
    nk_vf32x4_t a_low_f32x4 = vec_extract_fp32_from_shortl(a_u16x8);
    nk_vf32x4_t b_high_f32x4 = vec_extract_fp32_from_shorth(b_u16x8);
    nk_vf32x4_t b_low_f32x4 = vec_extract_fp32_from_shortl(b_u16x8);
    state->sum_f32x4 = vec_madd(a_high_f32x4, b_high_f32x4, state->sum_f32x4);
    state->sum_f32x4 = vec_madd(a_low_f32x4, b_low_f32x4, state->sum_f32x4);
}

NK_INTERNAL void nk_dot_f16x8_finalize_powervsx(                                                //
    nk_dot_f16x8_state_powervsx_t const *state_a, nk_dot_f16x8_state_powervsx_t const *state_b, //
    nk_dot_f16x8_state_powervsx_t const *state_c, nk_dot_f16x8_state_powervsx_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    nk_vf32x4_t a_f32x4 = state_a->sum_f32x4, b_f32x4 = state_b->sum_f32x4, c_f32x4 = state_c->sum_f32x4,
                d_f32x4 = state_d->sum_f32x4;
    nk_vf32x4_t transpose_ab_low_f32x4 = vec_mergeh(a_f32x4, b_f32x4);
    nk_vf32x4_t transpose_cd_low_f32x4 = vec_mergeh(c_f32x4, d_f32x4);
    nk_vf32x4_t transpose_ab_high_f32x4 = vec_mergel(a_f32x4, b_f32x4);
    nk_vf32x4_t transpose_cd_high_f32x4 = vec_mergel(c_f32x4, d_f32x4);
    nk_vf32x4_t sum_lane0_f32x4 = (nk_vf32x4_t)vec_xxpermdi((nk_vu64x2_t)transpose_ab_low_f32x4,
                                                            (nk_vu64x2_t)transpose_cd_low_f32x4, 0);
    nk_vf32x4_t sum_lane1_f32x4 = (nk_vf32x4_t)vec_xxpermdi((nk_vu64x2_t)transpose_ab_low_f32x4,
                                                            (nk_vu64x2_t)transpose_cd_low_f32x4, 3);
    nk_vf32x4_t sum_lane2_f32x4 = (nk_vf32x4_t)vec_xxpermdi((nk_vu64x2_t)transpose_ab_high_f32x4,
                                                            (nk_vu64x2_t)transpose_cd_high_f32x4, 0);
    nk_vf32x4_t sum_lane3_f32x4 = (nk_vf32x4_t)vec_xxpermdi((nk_vu64x2_t)transpose_ab_high_f32x4,
                                                            (nk_vu64x2_t)transpose_cd_high_f32x4, 3);
    result->vf32x4 = vec_add(vec_add(sum_lane0_f32x4, sum_lane1_f32x4), vec_add(sum_lane2_f32x4, sum_lane3_f32x4));
}

/**
 *  @brief Running state for 128-bit dot accumulation over i8 scalars on Power VSX.
 *
 *  Algebraic transform: a·b = a·(b⊕0x80) − 128·Σa. Uses VMSUMMBM (i8×u8 → i32) for the biased
 *  product. Correction is applied at finalize using precomputed column sums from the compensated
 *  macro infrastructure.
 */
typedef struct nk_dot_i8x16_state_powervsx_t {
    nk_vi32x4_t biased_sum_i32x4;
} nk_dot_i8x16_state_powervsx_t;

NK_INTERNAL void nk_dot_i8x16_init_powervsx(nk_dot_i8x16_state_powervsx_t *state) {
    state->biased_sum_i32x4 = vec_splats((nk_i32_t)0);
}

NK_INTERNAL void nk_dot_i8x16_update_powervsx(nk_dot_i8x16_state_powervsx_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                              nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    // VMSUMMBM(b, a⊕0x80) = Σ(b_i · (a_i+128)) = a·b + 128·Σb
    // Swapping operands: b in signed slot, biased a in unsigned slot.
    // Correction −128·Σb uses precomputed B column sums from the compensated macro.
    nk_vu8x16_t const bias_u8x16 = vec_splats((nk_u8_t)0x80);
    nk_vu8x16_t a_biased_u8x16 = vec_xor(a.vu8x16, bias_u8x16);
    state->biased_sum_i32x4 = vec_msum(b.vi8x16, a_biased_u8x16, state->biased_sum_i32x4);
}

NK_INTERNAL void nk_dot_i8x16_finalize_powervsx(                                                //
    nk_dot_i8x16_state_powervsx_t const *state_a, nk_dot_i8x16_state_powervsx_t const *state_b, //
    nk_dot_i8x16_state_powervsx_t const *state_c, nk_dot_i8x16_state_powervsx_t const *state_d, //
    nk_size_t total_dimensions,                                                                 //
    nk_i32_t a_sum, nk_b128_vec_t const *b_sums_vec, nk_b128_vec_t *result_vec) {
    nk_unused_(total_dimensions);
    nk_unused_(a_sum);

    // Transpose-reduce biased products across 4 accumulators → one i32x4
    nk_vi32x4_t a_i32x4 = state_a->biased_sum_i32x4, b_i32x4 = state_b->biased_sum_i32x4,
                c_i32x4 = state_c->biased_sum_i32x4, d_i32x4 = state_d->biased_sum_i32x4;
    nk_vi32x4_t transpose_ab_low_i32x4 = vec_mergeh(a_i32x4, b_i32x4);
    nk_vi32x4_t transpose_cd_low_i32x4 = vec_mergeh(c_i32x4, d_i32x4);
    nk_vi32x4_t transpose_ab_high_i32x4 = vec_mergel(a_i32x4, b_i32x4);
    nk_vi32x4_t transpose_cd_high_i32x4 = vec_mergel(c_i32x4, d_i32x4);
    nk_vi32x4_t sum_lane0_i32x4 = (nk_vi32x4_t)vec_xxpermdi((nk_vu64x2_t)transpose_ab_low_i32x4,
                                                            (nk_vu64x2_t)transpose_cd_low_i32x4, 0);
    nk_vi32x4_t sum_lane1_i32x4 = (nk_vi32x4_t)vec_xxpermdi((nk_vu64x2_t)transpose_ab_low_i32x4,
                                                            (nk_vu64x2_t)transpose_cd_low_i32x4, 3);
    nk_vi32x4_t sum_lane2_i32x4 = (nk_vi32x4_t)vec_xxpermdi((nk_vu64x2_t)transpose_ab_high_i32x4,
                                                            (nk_vu64x2_t)transpose_cd_high_i32x4, 0);
    nk_vi32x4_t sum_lane3_i32x4 = (nk_vi32x4_t)vec_xxpermdi((nk_vu64x2_t)transpose_ab_high_i32x4,
                                                            (nk_vu64x2_t)transpose_cd_high_i32x4, 3);
    nk_vi32x4_t biased_i32x4 = vec_add(vec_add(sum_lane0_i32x4, sum_lane1_i32x4),
                                       vec_add(sum_lane2_i32x4, sum_lane3_i32x4));

    // Correction: VMSUMMBM(b, a⊕0x80) = Σ(b_i·(a_i+128)) = a·b + 128·Σb
    // So a·b = biased − 128·Σb. B column sums are precomputed during packing.
    nk_vu32x4_t shift_u32x4 = vec_splats((nk_u32_t)7);
    nk_vi32x4_t correction_i32x4 = (nk_vi32x4_t)vec_sl((nk_vu32x4_t)b_sums_vec->vi32x4, shift_u32x4);
    result_vec->vi32x4 = vec_sub(biased_i32x4, correction_i32x4);
}

/** @brief Running state for i8 column sum precomputation on Power VSX. */
typedef struct nk_sum_i8x16_state_powervsx_t {
    nk_vu32x4_t biased_sum_u32x4;
} nk_sum_i8x16_state_powervsx_t;

NK_INTERNAL void nk_sum_i8x16_init_powervsx(nk_sum_i8x16_state_powervsx_t *state) {
    state->biased_sum_u32x4 = vec_splats((nk_u32_t)0);
}

NK_INTERNAL void nk_sum_i8x16_update_powervsx(nk_sum_i8x16_state_powervsx_t *state, nk_b128_vec_t values_vec) {
    nk_vu8x16_t const bias_u8x16 = vec_splats((nk_u8_t)0x80);
    nk_vu8x16_t biased_u8x16 = vec_xor(values_vec.vu8x16, bias_u8x16);
    state->biased_sum_u32x4 = vec_sum4s(biased_u8x16, state->biased_sum_u32x4);
}

NK_INTERNAL nk_i32_t nk_sum_i8x16_finalize_powervsx(nk_sum_i8x16_state_powervsx_t const *state, nk_size_t count) {
    nk_u32_t biased_sum = nk_hsum_u32x4_powervsx_(state->biased_sum_u32x4);
    return (nk_i32_t)((nk_i64_t)biased_sum - 128 * (nk_i64_t)count);
}

/**
 *  @brief Running state for 128-bit dot accumulation over u8 scalars on Power VSX.
 *
 *  Processes 16 u8 values at a time via vec_msum, accumulating into 4 u32 lanes.
 */
typedef struct nk_dot_u8x16_state_powervsx_t {
    nk_vu32x4_t sum_u32x4;
} nk_dot_u8x16_state_powervsx_t;

NK_INTERNAL void nk_dot_u8x16_init_powervsx(nk_dot_u8x16_state_powervsx_t *state) {
    state->sum_u32x4 = vec_splats((nk_u32_t)0);
}

NK_INTERNAL void nk_dot_u8x16_update_powervsx(nk_dot_u8x16_state_powervsx_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                              nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    // Unsigned × unsigned multiply-sum: 16 u8 products accumulated into 4 u32 lanes
    nk_vu8x16_t a_u8x16 = a.vu8x16;
    nk_vu8x16_t b_u8x16 = b.vu8x16;
    state->sum_u32x4 = vec_msum(a_u8x16, b_u8x16, state->sum_u32x4);
}

NK_INTERNAL void nk_dot_u8x16_finalize_powervsx(                                                //
    nk_dot_u8x16_state_powervsx_t const *state_a, nk_dot_u8x16_state_powervsx_t const *state_b, //
    nk_dot_u8x16_state_powervsx_t const *state_c, nk_dot_u8x16_state_powervsx_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    nk_vu32x4_t a_u32x4 = state_a->sum_u32x4, b_u32x4 = state_b->sum_u32x4, c_u32x4 = state_c->sum_u32x4,
                d_u32x4 = state_d->sum_u32x4;
    nk_vu32x4_t transpose_ab_low_u32x4 = vec_mergeh(a_u32x4, b_u32x4);
    nk_vu32x4_t transpose_cd_low_u32x4 = vec_mergeh(c_u32x4, d_u32x4);
    nk_vu32x4_t transpose_ab_high_u32x4 = vec_mergel(a_u32x4, b_u32x4);
    nk_vu32x4_t transpose_cd_high_u32x4 = vec_mergel(c_u32x4, d_u32x4);
    nk_vu32x4_t sum_lane0_u32x4 = (nk_vu32x4_t)vec_xxpermdi((nk_vu64x2_t)transpose_ab_low_u32x4,
                                                            (nk_vu64x2_t)transpose_cd_low_u32x4, 0);
    nk_vu32x4_t sum_lane1_u32x4 = (nk_vu32x4_t)vec_xxpermdi((nk_vu64x2_t)transpose_ab_low_u32x4,
                                                            (nk_vu64x2_t)transpose_cd_low_u32x4, 3);
    nk_vu32x4_t sum_lane2_u32x4 = (nk_vu32x4_t)vec_xxpermdi((nk_vu64x2_t)transpose_ab_high_u32x4,
                                                            (nk_vu64x2_t)transpose_cd_high_u32x4, 0);
    nk_vu32x4_t sum_lane3_u32x4 = (nk_vu32x4_t)vec_xxpermdi((nk_vu64x2_t)transpose_ab_high_u32x4,
                                                            (nk_vu64x2_t)transpose_cd_high_u32x4, 3);
    result->vu32x4 = vec_add(vec_add(sum_lane0_u32x4, sum_lane1_u32x4), vec_add(sum_lane2_u32x4, sum_lane3_u32x4));
}

/**
 *  @brief Running state for 128-bit binary dot accumulation on Power VSX.
 *
 *  Processes 128 bits (16 bytes) at a time via AND + doubleword popcount (vpopcntd),
 *  accumulating bit-match counts into 2 u64 lanes.
 */
typedef struct nk_dot_u1x128_state_powervsx_t {
    nk_vu64x2_t dot_count_u64x2;
} nk_dot_u1x128_state_powervsx_t;

NK_INTERNAL void nk_dot_u1x128_init_powervsx(nk_dot_u1x128_state_powervsx_t *state) {
    state->dot_count_u64x2 = vec_splats((nk_u64_t)0);
}

NK_INTERNAL void nk_dot_u1x128_update_powervsx(nk_dot_u1x128_state_powervsx_t *state, nk_b128_vec_t a, nk_b128_vec_t b,
                                               nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_unused_(depth_offset);
    nk_unused_(active_dimensions);
    // AND → doubleword popcount (vpopcntd, 3cy ALU) → vec_add (7cy DP)
    // Simpler data flow than vpopcntb + vec_sum4s, and u64 accumulator holds larger counts
    nk_vu8x16_t a_u8x16 = a.vu8x16;
    nk_vu8x16_t b_u8x16 = b.vu8x16;
    nk_vu8x16_t and_u8x16 = vec_and(a_u8x16, b_u8x16);
    nk_vu64x2_t popcnt_u64x2 = vec_popcnt((nk_vu64x2_t)and_u8x16);
    state->dot_count_u64x2 = vec_add(state->dot_count_u64x2, popcnt_u64x2);
}

NK_INTERNAL void nk_dot_u1x128_finalize_powervsx(                                                 //
    nk_dot_u1x128_state_powervsx_t const *state_a, nk_dot_u1x128_state_powervsx_t const *state_b, //
    nk_dot_u1x128_state_powervsx_t const *state_c, nk_dot_u1x128_state_powervsx_t const *state_d, //
    nk_size_t total_dimensions, nk_b128_vec_t *result) {
    nk_unused_(total_dimensions);
    nk_vu64x2_t sum_a_u64x2 = vec_add(state_a->dot_count_u64x2,
                                      vec_xxpermdi(state_a->dot_count_u64x2, state_a->dot_count_u64x2, 2));
    nk_vu64x2_t sum_b_u64x2 = vec_add(state_b->dot_count_u64x2,
                                      vec_xxpermdi(state_b->dot_count_u64x2, state_b->dot_count_u64x2, 2));
    nk_vu64x2_t sum_c_u64x2 = vec_add(state_c->dot_count_u64x2,
                                      vec_xxpermdi(state_c->dot_count_u64x2, state_c->dot_count_u64x2, 2));
    nk_vu64x2_t sum_d_u64x2 = vec_add(state_d->dot_count_u64x2,
                                      vec_xxpermdi(state_d->dot_count_u64x2, state_d->dot_count_u64x2, 2));
    nk_vu64x2_t ab_u64x2 = vec_xxpermdi(sum_a_u64x2, sum_b_u64x2, 0);
    nk_vu64x2_t cd_u64x2 = vec_xxpermdi(sum_c_u64x2, sum_d_u64x2, 0);
    result->vu32x4 = vec_pack(ab_u64x2, cd_u64x2);
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
#endif // NK_DOT_POWERVSX_H
