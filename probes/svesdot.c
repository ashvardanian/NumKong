/**
 *  @file probes/svesdot.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief Whether the toolchain builds the SVE I8 kernels, signed dot products, @c FEAT_SVEDot.
 */
#define NUMKONG_HEADER_ONLY    1
#define NUMKONG_TARGET_SVESDOT 1
#include <numkong/types.h>
#include <numkong/dot/svesdot.h> // `nk_dot_i8_svesdot`

int main(void) {
    nk_i8_t a[64], b[64];
    nk_i32_t dot = 0;
    for (int i = 0; i != 64; ++i) a[i] = 1, b[i] = 2;
    nk_dot_i8_svesdot(a, b, 64, &dot, 0);
    return dot != 128;
}
