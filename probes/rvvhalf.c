/**
 *  @file probes/rvvhalf.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief Whether the toolchain builds the RVV Zvfh kernels, half-precision vectors.
 */
#define NUMKONG_HEADER_ONLY    1
#define NUMKONG_TARGET_RVV     1 // `types.h` drops the RVV extensions without the base
#define NUMKONG_TARGET_RVVHALF 1
#include <numkong/types.h>
#include <numkong/dot/rvvhalf.h> // `nk_dot_f16_rvvhalf`

int main(void) {
    nk_f16_t a[64] = {0}, b[64] = {0};
    nk_f32_t dot = 1;
    nk_dot_f16_rvvhalf(a, b, 64, &dot, 0);
    return dot != 0;
}
