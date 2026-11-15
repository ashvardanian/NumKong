/**
 *  @file probes/powermma.c
 *  @author Ash Vardanian
 *  @date November 12, 2026
 *  @brief Whether the toolchain builds the Power MMA kernels, POWER10 Matrix-Multiply Assist.
 */
#define NUMKONG_HEADER_ONLY          1
#define NUMKONG_ARCH_PPC64_POWERMMA_ 1
#include "numkong/types.h"
#include "numkong/dots/powermma.h" // `nk_dots_symmetric_bf16_powermma`

int main(void) {
    nk_bf16_t vectors[8 * 16] = {0};
    nk_f32_t gram[8 * 8];
    nk_dots_symmetric_bf16_powermma(vectors, 8, 16, 16 * sizeof(nk_bf16_t), gram, 8 * sizeof(nk_f32_t), 0, 8, 0, 0, 0);
    return gram[0] != 0;
}
