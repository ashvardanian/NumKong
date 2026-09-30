/**
 *  @file probes/sme.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief Whether the toolchain builds the SME kernels, the Scalable Matrix Extension.
 */
#define NUMKONG_HEADER_ONLY 1
#define NUMKONG_TARGET_SME  1
#include <numkong/types.h>
#include <numkong/dots/sme.h> // `nk_dots_symmetric_i8_sme`

int main(void) {
    nk_i8_t vectors[16 * 64];
    nk_i32_t gram[16 * 16];
    for (int i = 0; i != 16 * 64; ++i) vectors[i] = 1;
    nk_dots_symmetric_i8_sme(vectors, 16, 64, 64, gram, 16 * sizeof(nk_i32_t), 0, 16, 0);
    return gram[0] != 64;
}
