/**
 *  @file probes/svehalf.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief Whether the toolchain builds the SVE F16 kernels, half-precision arithmetic.
 */
#define NUMKONG_HEADER_ONLY    1
#define NUMKONG_TARGET_SVEHALF 1
#include <numkong/types.h>
#include <numkong/dot/svehalf.h> // `nk_dot_f16_svehalf`

int main(void) {
    nk_f16_t a[64] = {0}, b[64] = {0};
    nk_f32_t dot = 1;
    nk_dot_f16_svehalf(a, b, 64, &dot, 0);
    return dot != 0;
}
