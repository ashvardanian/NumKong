/**
 *  @file probes/rvvbf16.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief Whether the toolchain builds the RVV Zvfbfwma kernels, BF16 widening fused multiply-add.
 */
#define NUMKONG_HEADER_ONLY    1
#define NUMKONG_TARGET_RVV     1 // `types.h` drops the RVV extensions without the base
#define NUMKONG_TARGET_RVVBF16 1
#include <numkong/types.h>
#include <numkong/dot/rvvbf16.h> // `nk_dot_bf16_rvvbf16`

int main(void) {
    nk_bf16_t a[64] = {0}, b[64] = {0};
    nk_f32_t dot = 1;
    nk_dot_bf16_rvvbf16(a, b, 64, &dot, 0);
    return dot != 0;
}
