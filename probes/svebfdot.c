/**
 *  @file probes/svebfdot.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief Whether the toolchain builds the SVE BF16 kernels, the @c FEAT_BF16 dot product.
 */
#define NUMKONG_HEADER_ONLY     1
#define NUMKONG_TARGET_SVEBFDOT 1
#include <numkong/types.h>
#include <numkong/dot/svebfdot.h> // `nk_dot_bf16_svebfdot`

int main(void) {
    nk_bf16_t a[64] = {0}, b[64] = {0};
    nk_f32_t dot = 1;
    nk_dot_bf16_svebfdot(a, b, 64, &dot, 0);
    return dot != 0;
}
