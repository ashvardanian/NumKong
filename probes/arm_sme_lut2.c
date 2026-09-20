/* NumKong ISA probe: SME LUT2 (FEAT_SME_LUTv2) */
#if defined(_WIN32)
#error "SVE/SME not supported on Windows ARM"
#endif

#if !defined(__ARM_FEATURE_SME2)
#error "Feature not available"
#endif
#include <arm_sme.h>
__arm_new("zt0") __arm_locally_streaming int test_smelut2(void) {
    svuint8x2_t indices = svcreate2_u8(svdup_u8(0), svdup_u8(0));
    svuint8x4_t r = svluti4_zt_u8_x4(0, indices);
    return (int)svaddv_u8(svptrue_b8(), svget4_u8(r, 0)) == 0 ? 0 : 1;
}
int main(void) { return test_smelut2(); }
