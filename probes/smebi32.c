/**
 *  @file probes/smebi32.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief Whether the toolchain builds the SME BI32 kernels, the boolean and integer
 *      32-bit outer product.
 */
#define NUMKONG_HEADER_ONLY    1
#define NUMKONG_TARGET_SMEBI32 1
#include <numkong/types.h>
#include <numkong/dots/smebi32.h> // `nk_dots_symmetric_u1_smebi32`

int main(void) {
    nk_u1x8_t vectors[16 * 8];
    nk_u32_t gram[16 * 16];
    for (int i = 0; i != 16 * 8; ++i) vectors[i] = 0xFF;
    nk_dots_symmetric_u1_smebi32(vectors, 16, 64, 8, gram, 16 * sizeof(nk_u32_t), 0, 16, 0);
    return gram[0] != 64;
}
