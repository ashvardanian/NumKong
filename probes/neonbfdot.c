/**
 *  @file probes/neonbfdot.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief Whether the toolchain builds the NEON BFDOT kernels, ARMv8.6-A bfloat16 dot products.
 */
#define NUMKONG_HEADER_ONLY      1
#define NUMKONG_TARGET_NEONBFDOT 1
#include <numkong/types.h>
#include <numkong/dot/neonbfdot.h> // `nk_dot_bf16_neonbfdot`

int main(void) {
    nk_bf16_t a[64] = {0}, b[64] = {0};
    nk_f32_t dot = 1;
    nk_dot_bf16_neonbfdot(a, b, 64, &dot, 0);
    return dot != 0;
}
