/**
 *  @file probes/arm_neon.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief NumKong ISA probe for NEON, the AArch64 baseline SIMD extension.
 */
#include <arm_neon.h>
int main(void) {
    float32x4_t a = vdupq_n_f32(1.0f);
    float32x4_t b = vdupq_n_f32(2.0f);
    float32x4_t c = vaddq_f32(a, b);
    return vgetq_lane_f32(c, 0) > 0.0f ? 0 : 1;
}
