/**
 *  @file probes/arm_sme_fa64.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief NumKong ISA probe for SME FA64, @c FEAT_SME_FA64, full SVE2 in streaming mode.
 */
#if defined(_WIN32)
#error "SVE/SME not supported on Windows ARM"
#endif

#if !defined(__ARM_FEATURE_SME)
#error "Feature not available"
#endif
#include <arm_sme.h>
__arm_locally_streaming int test_smefa64(void) {
    svfloat32_t a = svdup_f32(1.0f);
    return (int)svaddv_f32(svptrue_b32(), a) > 0 ? 0 : 1;
}
int main(void) { return test_smefa64(); }
