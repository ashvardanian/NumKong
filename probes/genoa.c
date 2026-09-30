/**
 *  @file probes/genoa.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief Whether the toolchain builds the Genoa kernels, AVX-512F/BW/DQ/VL plus BF16.
 */
#define NUMKONG_HEADER_ONLY  1
#define NUMKONG_TARGET_GENOA 1
#include <numkong/types.h>
#include <numkong/dot/genoa.h> // `nk_dot_bf16_genoa`

int main(void) {
    nk_bf16_t a[64] = {0}, b[64] = {0};
    nk_f32_t dot = 1;
    nk_dot_bf16_genoa(a, b, 64, &dot, 0);
    return dot != 0;
}
