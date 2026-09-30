/**
 *  @file probes/rvvbb.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief Whether the toolchain builds the RVV Zvbb kernels, basic bit manipulation.
 */
#define NUMKONG_HEADER_ONLY  1
#define NUMKONG_TARGET_RVV   1 // `types.h` drops the RVV extensions without the base
#define NUMKONG_TARGET_RVVBB 1
#include <numkong/types.h>
#include <numkong/dot/rvvbb.h> // `nk_dot_u1_rvvbb`

int main(void) {
    nk_u1x8_t a[8], b[8];
    nk_u32_t dot = 0;
    for (int i = 0; i != 8; ++i) a[i] = 0xFF, b[i] = 0xFF;
    nk_dot_u1_rvvbb(a, b, 64, &dot, 0);
    return dot != 64;
}
