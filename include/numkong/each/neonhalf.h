/**
 *  @file include/numkong/each/neonhalf.h
 *  @author Ash Vardanian
 *  @date October 18, 2024
 *  @brief SIMD-accelerated elementwise arithmetic for NEON FP16.
 *
 *  @sa include/numkong/each.h
 *
 *  @section elementwise_neonhalf_instructions ARM NEON FP16 Instructions (ARMv8.2-FP16)
 *
 *  @verbatim
 *  Intrinsic       Instruction                  A76       M5
 *  vld1q_f16       LD1 (V.8H)                   4cy @ 2p  4cy @ 3p
 *  vst1q_f16       ST1 (V.8H)                   2cy @ 2p  2cy @ 3p
 *  vaddq_f16       FADD (V.8H, V.8H, V.8H)      2cy @ 2p  2cy @ 4p
 *  @endverbatim
 *
 *  The ARMv8.2-FP16 extension enables native half-precision element-wise operations, processing 8
 *  F16 elements per instruction. Only the sum stays in F16, as one F16 addition rounds exactly like
 *  the serial F32 sum narrowed back. Scale, blend, and fma round more than once, so they widen to
 *  F32 in `each/neon.h` instead.
 */
#ifndef NUMKONG_EACH_NEONHALF_H
#define NUMKONG_EACH_NEONHALF_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_TARGET_NEONHALF

#include "numkong/types.h"
#include "numkong/cast/serial.h" // `nk_f16_to_f32_serial_`, `nk_f32_to_f16_serial_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16")
#endif

NUMKONG_API nk_status_t nk_each_sum_f16_neonhalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f16_t *result,
                                                 nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    // The main loop:
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t a_vec = vreinterpretq_f16_u16(vld1q_u16((nk_u16_t const *)a + i));
        float16x8_t b_vec = vreinterpretq_f16_u16(vld1q_u16((nk_u16_t const *)b + i));
        float16x8_t sum_vec = vaddq_f16(a_vec, b_vec);
        vst1q_u16((nk_u16_t *)result + i, vreinterpretq_u16_f16(sum_vec));
    }

    // The tail:
    for (; i < n; ++i) {
        nk_f32_t ai, bi;
        nk_f16_to_f32_serial_(a + i, &ai);
        nk_f16_to_f32_serial_(b + i, &bi);
        nk_f32_t const sum = ai + bi;
        nk_f32_to_f16_serial_(&sum, result + i);
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
#endif // NUMKONG_EACH_NEONHALF_H
