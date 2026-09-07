/* NumKong ISA probe: NEON FP8 (fp8dot4) */
#include <arm_neon.h>
int test_neonfp8(void) {
    mfloat8x16_t a = vreinterpretq_mf8_u8(vdupq_n_u8(0));
    float32x4_t acc = vdupq_n_f32(0.0f);
    acc = vdotq_f32_mf8_fpm(acc, a, a, 0);
    return vgetq_lane_f32(acc, 0) == 0.0f ? 0 : 1;
}
int main(void) { return test_neonfp8(); }
