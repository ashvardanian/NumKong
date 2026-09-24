/**
 *  @file probes/x86_skylake.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief NumKong ISA probe for Skylake, AVX-512F/BW/DQ/VL.
 */
#if defined(__APPLE__)
#error "AVX-512 not available on macOS"
#endif

#if !defined(__AVX512F__)
#error "Feature not available"
#endif
#include <immintrin.h>
int main(void) {
    volatile int one = 1;
    __m512i a = _mm512_set1_epi32(one);
    __m512i b = _mm512_add_epi32(a, a);
    return (int)_mm512_reduce_add_epi32(b) == 32 ? 0 : 1;
}
