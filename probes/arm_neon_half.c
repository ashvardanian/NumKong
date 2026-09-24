/**
 *  @file probes/arm_neon_half.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief NumKong ISA probe for ARMv8.2-A NEON half-precision arithmetic.
 */
#include <arm_neon.h>
int main(void) {
    float16x8_t a = vdupq_n_f16(1.0f);
    float16x8_t b = vdupq_n_f16(2.0f);
    float16x8_t c = vaddq_f16(a, b);
    return vgetq_lane_f16(c, 0) > 0.0f ? 0 : 1;
}
