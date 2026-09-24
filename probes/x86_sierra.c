/**
 *  @file probes/x86_sierra.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief NumKong ISA probe for Sierra Forest, AVXVNNIINT8.
 */
#if defined(__APPLE__)
#error "AVX-512 not available on macOS"
#endif

#if !defined(__AVXVNNIINT8__)
#error "Feature not available"
#endif
#include <immintrin.h>
int main(void) {
    volatile int two = 2;
    __m256i acc = _mm256_setzero_si256();
    __m256i a = _mm256_set1_epi8((char)two);
    __m256i b = _mm256_set1_epi8((char)(two + 1));
    acc = _mm256_dpbssd_epi32(acc, a, b);
    return _mm256_extract_epi32(acc, 0) == 24 ? 0 : 1;
}
