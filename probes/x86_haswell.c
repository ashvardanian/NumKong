/**
 *  @file probes/x86_haswell.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief NumKong ISA probe for Haswell, AVX2 plus FMA plus F16C.
 */
#if !defined(__AVX2__)
#error "Feature not available"
#endif
#include <immintrin.h>
int main(void) {
    volatile int one = 1;
    __m256i a = _mm256_set1_epi32(one);
    __m256i b = _mm256_add_epi32(a, a);
    return _mm256_extract_epi32(b, 0) == 2 ? 0 : 1;
}
