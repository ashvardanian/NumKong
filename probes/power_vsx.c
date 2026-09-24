/**
 *  @file probes/power_vsx.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief NumKong ISA probe for Power VSX, POWER9+ 128-bit SIMD.
 */
#if !defined(__VSX__)
#error "Feature not available"
#endif
#include <altivec.h>
int main(void) {
    __vector float a = vec_splats(1.0f);
    __vector float b = vec_splats(2.0f);
    __vector float c = vec_madd(a, b, a);
    /* vec_extract requires POWER9+ */
    return vec_extract(c, 0) == 3.0f ? 0 : 1;
}
