/**
 *  @file probes/turin.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief Whether the toolchain builds the Turin kernels, AVX-512F plus VP2INTERSECT.
 */
#define NUMKONG_HEADER_ONLY  1
#define NUMKONG_TARGET_TURIN 1
#include <numkong/types.h>
#include <numkong/sparse/turin.h> // `nk_sparse_intersect_u32_turin`

int main(void) {
    nk_u32_t a[16], b[16], common[16];
    nk_size_t count = 0;
    for (nk_u32_t i = 0; i != 16; ++i) a[i] = i, b[i] = 2 * i;
    nk_sparse_intersect_u32_turin(a, b, 16, 16, common, &count, 0);
    return count != 8;
}
