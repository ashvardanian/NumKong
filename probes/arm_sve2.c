/**
 *  @file probes/arm_sve2.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief NumKong ISA probe for SVE2.
 */
#if defined(_WIN32)
#error "SVE/SME not supported on Windows ARM"
#endif

#if defined(__APPLE__) && defined(__aarch64__)
#error "SVE not available on Apple Silicon"
#endif

#if !defined(__ARM_FEATURE_SVE2)
#error "Feature not available"
#endif
#include <arm_sve.h>
int test_sve2(void) {
    svint32_t a = svdup_s32(2);
    svint32_t b = svdup_s32(3);
    svint32_t c = svmul_s32_z(svptrue_b32(), a, b);
    return (int)svaddv_s32(svptrue_b32(), c) > 0 ? 0 : 1;
}
int main(void) { return test_sve2(); }
