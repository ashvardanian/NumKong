/**
 *  @file probes/sapphire.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief Whether the toolchain builds the Sapphire Rapids kernels, AVX-512F/BW/DQ/VL plus FP16.
 */
#define NUMKONG_HEADER_ONLY     1
#define NUMKONG_TARGET_SAPPHIRE 1
#include "numkong/types.h"
#include "numkong/each/sapphire.h" // `nk_each_sum_f16_sapphire`
#include "numkong/trigonometry/sapphire.h"

int main(void) {
    nk_f16_t a[64] = {0}, b[64] = {0}, sum[64];
    nk_each_sum_f16_sapphire(a, b, 64, sum, 0);
    return sum[0] != 0;
}
