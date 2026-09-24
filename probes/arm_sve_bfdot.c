/**
 *  @file probes/arm_sve_bfdot.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief NumKong ISA probe for SVE BF16, the @c FEAT_BF16 dot product.
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
int test_svebfdot(void) {
    svfloat32_t acc = svdup_f32(0.0f);
    svbfloat16_t a = svdup_bf16(0.0f);
    acc = svbfdot_f32(acc, a, a);
    return (int)svaddv_f32(svptrue_b32(), acc) == 0 ? 0 : 1;
}
int main(void) { return test_svebfdot(); }
