/**
 *  @file probes/arm_sme.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief NumKong ISA probe for SME, the Scalable Matrix Extension.
 */
#if defined(_WIN32)
#error "SVE/SME not supported on Windows ARM"
#endif

#if !defined(__ARM_FEATURE_SME)
#error "Feature not available"
#endif
#include <arm_sme.h>
__arm_new("za") __arm_locally_streaming int test_sme(void) {
    svfloat32_t a = svdup_f32(1.0f);
    svbool_t p = svptrue_b32();
    svmopa_za32_f32_m(0, p, p, a, a);
    return 0;
}
int main(void) { return test_sme(); }
