/**
 *  @file probes/arm_sme_bi32.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief NumKong ISA probe for SME BI32, the boolean and integer 32-bit outer product.
 */
#if defined(_WIN32)
#error "SVE/SME not supported on Windows ARM"
#endif

#if !defined(__ARM_FEATURE_SME2)
#error "Feature not available"
#endif
#include <arm_sme.h>
__arm_new("za") __arm_locally_streaming int test_smebi32(void) {
    svuint32_t a = svdup_u32(1);
    svbool_t p = svptrue_b32();
    svbmopa_za32_u32_m(0, p, p, a, a);
    return 0;
}
int main(void) { return test_smebi32(); }
