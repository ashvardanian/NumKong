/**
 *  @file probes/smef64.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief Whether the toolchain builds the SME F64 kernels, @c FEAT_SME_F64F64.
 */
#define NUMKONG_HEADER_ONLY   1
#define NUMKONG_TARGET_SMEF64 1
#include <numkong/types.h>
#include <numkong/dots/smef64.h> // `nk_dots_symmetric_f64_smef64`

int main(void) {
    nk_f64_t vectors[8 * 8];
    nk_f64_t gram[8 * 8];
    for (int i = 0; i != 8 * 8; ++i) vectors[i] = 1;
    nk_dots_symmetric_f64_smef64(vectors, 8, 8, 8 * sizeof(nk_f64_t), gram, 8 * sizeof(nk_f64_t), 0, 8, 0);
    return gram[0] != 8;
}
