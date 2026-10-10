/**
 *  @file probes/neonhalf.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief Whether the toolchain builds the NEON half kernels, ARMv8.2-A half-precision arithmetic.
 */
#define NUMKONG_HEADER_ONLY     1
#define NUMKONG_TARGET_NEONHALF 1
#include "numkong/types.h"
#include "numkong/each/neonhalf.h" // `nk_each_sum_f16_neonhalf`
#include "numkong/trigonometry/neonhalf.h"

int main(void) {
    nk_f16_t a[64] = {0}, b[64] = {0}, sum[64];
    nk_each_sum_f16_neonhalf(a, b, 64, sum, 0);
    return sum[0] != 0;
}
