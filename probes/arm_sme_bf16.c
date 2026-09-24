/**
 *  @file probes/arm_sme_bf16.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief NumKong ISA probe for SME BF16, @c FEAT_SME_B16B16.
 */
#if defined(_WIN32)
#error "SVE/SME not supported on Windows ARM"
#endif

#if !defined(__ARM_FEATURE_SME_B16B16)
#error "Feature not available"
#endif
#include <arm_sme.h>
__arm_new("za") __arm_locally_streaming int test_smebf16(void) {
    svbfloat16_t a = svdup_bf16(0.0f);
    svbool_t p = svptrue_b16();
    svmopa_za16_bf16_m(0, p, p, a, a);
    return 0;
}
int main(void) { return test_smebf16(); }
