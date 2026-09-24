/**
 *  @file probes/arm_sve2p1.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief NumKong ISA probe for SVE2P1.
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
int test_sve2p1(void) {
    svfloat32_t a = svdup_f32(1.0f);
    return (int)svaddv_f32(svptrue_b32(), a) > 0 ? 0 : 1;
}
int main(void) { return test_sve2p1(); }
