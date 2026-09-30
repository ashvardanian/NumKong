/**
 *  @file probes/sve2.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief Whether the toolchain builds the SVE2 kernels.
 */
#define NUMKONG_HEADER_ONLY 1
#define NUMKONG_TARGET_SVE2 1
#include <numkong/types.h>
#include <numkong/sparse/sve2.h> // `nk_sparse_intersect_u32_sve2`

int main(void) {
    nk_u32_t a[16], b[16], common[16];
    nk_size_t count = 0;
    for (nk_u32_t i = 0; i != 16; ++i) a[i] = i, b[i] = 2 * i;
    nk_sparse_intersect_u32_sve2(a, b, 16, 16, common, &count, 0);
    return count != 8;
}
