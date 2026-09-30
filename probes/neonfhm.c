/**
 *  @file probes/neonfhm.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief Whether the toolchain builds the NEON FHM kernels, ARMv8.2-A FP16 fused multiply-add.
 */
#define NUMKONG_HEADER_ONLY    1
#define NUMKONG_TARGET_NEONFHM 1
#include <numkong/types.h>
#include <numkong/dot/neonfhm.h> // `nk_dot_f16_neonfhm`

int main(void) {
    nk_f16_t a[64] = {0}, b[64] = {0};
    nk_f32_t dot = 1;
    nk_dot_f16_neonfhm(a, b, 64, &dot, 0);
    return dot != 0;
}
