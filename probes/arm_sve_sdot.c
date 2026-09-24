/**
 *  @file probes/arm_sve_sdot.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief NumKong ISA probe for SVE I8 signed dot products, @c FEAT_SVEDot.
 */
#if defined(_WIN32)
#error "SVE/SME not supported on Windows ARM"
#endif

#if defined(__APPLE__) && defined(__aarch64__)
#error "SVE not available on Apple Silicon"
#endif

#if !defined(__ARM_FEATURE_SVE)
#error "Feature not available"
#endif
#include <arm_sve.h>
int test_svesdot(void) {
    svint32_t acc = svdup_s32(0);
    svint8_t a = svdup_s8(1);
    svint8_t b = svdup_s8(1);
    acc = svdot_s32(acc, a, b);
    return (int)svaddv_s32(svptrue_b32(), acc) >= 0 ? 0 : 1;
}
int main(void) { return test_svesdot(); }
