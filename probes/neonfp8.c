/**
 *  @file probes/neonfp8.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief Whether the toolchain builds the NEON FP8 kernels, the @c fp8dot4 instruction.
 */
#define NUMKONG_HEADER_ONLY    1
#define NUMKONG_TARGET_NEONFP8 1
#include <numkong/types.h>
#include <numkong/dot/neonfp8.h> // `nk_dot_e4m3_neonfp8`

int main(void) {
    nk_e4m3_t a[64] = {0}, b[64] = {0};
    nk_f32_t dot = 1;
    nk_dot_e4m3_neonfp8(a, b, 64, &dot, 0);
    return dot != 0;
}
