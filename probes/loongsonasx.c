/**
 *  @file probes/loongsonasx.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief Whether the toolchain builds the LoongArch LASX kernels, 256-bit SIMD.
 */
#define NUMKONG_HEADER_ONLY        1
#define NUMKONG_TARGET_LOONGSONASX 1
#include <numkong/types.h>
#include <numkong/dot/loongsonasx.h> // `nk_dot_f32_loongsonasx`

int main(void) {
    nk_f32_t a[64], b[64];
    nk_f64_t dot = 0;
    for (int i = 0; i != 64; ++i) a[i] = 1, b[i] = 2;
    nk_dot_f32_loongsonasx(a, b, 64, &dot, 0);
    return dot != 128;
}
