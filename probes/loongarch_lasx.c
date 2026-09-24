/**
 *  @file probes/loongarch_lasx.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief NumKong ISA probe for LoongArch LASX, 256-bit SIMD.
 */
#if !defined(__loongarch_asx)
#error "Feature not available"
#endif
#include <lasxintrin.h>
int main(void) {
    __m256i a = __lasx_xvreplgr2vr_w(1);
    __m256i b = __lasx_xvreplgr2vr_w(2);
    __m256i c = __lasx_xvadd_w(a, b);
    int r = __lasx_xvpickve2gr_w(c, 0);
    return r == 3 ? 0 : 1;
}
