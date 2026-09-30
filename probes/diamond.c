/**
 *  @file probes/diamond.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief Whether the toolchain builds the Diamond Rapids kernels, AVX10.2.
 */
#define NUMKONG_HEADER_ONLY    1
#define NUMKONG_TARGET_DIAMOND 1
#include <numkong/types.h>
#include <numkong/dot/diamond.h> // `nk_dot_e4m3_diamond`

int main(void) {
    nk_e4m3_t a[64] = {0}, b[64] = {0};
    nk_f32_t dot = 1;
    nk_dot_e4m3_diamond(a, b, 64, &dot, 0);
    return dot != 0;
}
