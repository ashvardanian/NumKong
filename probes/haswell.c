/**
 *  @file probes/haswell.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief Whether the toolchain builds the Haswell kernels, AVX2 plus FMA plus F16C.
 */
#define NUMKONG_HEADER_ONLY    1
#define NUMKONG_TARGET_HASWELL 1
#include <numkong/types.h>
#include <numkong/dot/haswell.h> // `nk_dot_f32_haswell`

int main(void) {
    nk_f32_t a[64], b[64];
    nk_f64_t dot = 0;
    for (int i = 0; i != 64; ++i) a[i] = 1, b[i] = 2;
    nk_dot_f32_haswell(a, b, 64, &dot, 0);
    return dot != 128;
}
