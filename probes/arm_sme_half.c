/* NumKong ISA probe: SME F16 (FEAT_SME_F16F16) */
#if defined(_WIN32)
#error "SVE/SME not supported on Windows ARM"
#endif

#if !defined(__ARM_FEATURE_SME_F16F16)
#error "Feature not available"
#endif
#include <arm_sme.h>
__arm_new("za") __arm_locally_streaming int test_smehalf(void) {
    svfloat16_t a = svdup_f16((__fp16)1.0f);
    svbool_t p = svptrue_b16();
    svmopa_za16_f16_m(0, p, p, a, a);
    return 0;
}
int main(void) { return test_smehalf(); }
