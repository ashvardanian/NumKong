/**
 *  @file probes/arm_neon_bfdot.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief NumKong ISA probe for ARMv8.6-A NEON bfloat16 dot products.
 */
#include <arm_neon.h>
int main(void) {
    bfloat16x8_t a = vdupq_n_bf16(1.0f);
    bfloat16x8_t b = vdupq_n_bf16(2.0f);
    float32x4_t c = vdupq_n_f32(0.0f);
    c = vbfdotq_f32(c, a, b);
    return vgetq_lane_f32(c, 0) > 0.0f ? 0 : 1;
}
