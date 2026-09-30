/**
 *  @file probes/alder.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief Whether the toolchain builds the Alder Lake kernels, AVX-VNNI 256-bit.
 */
#define NUMKONG_HEADER_ONLY  1
#define NUMKONG_TARGET_ALDER 1
#include <numkong/types.h>
#include <numkong/dot/alder.h> // `nk_dot_i8_alder`

int main(void) {
    nk_i8_t a[64], b[64];
    nk_i32_t dot = 0;
    for (int i = 0; i != 64; ++i) a[i] = 1, b[i] = 2;
    nk_dot_i8_alder(a, b, 64, &dot, 0);
    return dot != 128;
}
