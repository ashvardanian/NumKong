/**
 *  @file probes/arm_neon_sdot.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief NumKong ISA probe for the ARMv8.2-A NEON SDOT dot product.
 */
#include <arm_neon.h>
int main(void) {
    int8x16_t a = vdupq_n_s8(1);
    int8x16_t b = vdupq_n_s8(2);
    int32x4_t c = vdupq_n_s32(0);
    c = vdotq_s32(c, a, b);
    return vgetq_lane_s32(c, 0) > 0 ? 0 : 1;
}
