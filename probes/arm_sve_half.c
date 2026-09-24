/**
 *  @file probes/arm_sve_half.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief NumKong ISA probe for SVE F16, half-precision arithmetic.
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
int test_svehalf(void) {
    svfloat16_t z = svdup_f16((__fp16)1.0f);
    return (int)svaddv_f16(svptrue_b16(), z);
}
int main(void) { return test_svehalf() > 0 ? 0 : 1; }
