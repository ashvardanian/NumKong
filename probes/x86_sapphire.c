/**
 *  @file probes/x86_sapphire.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief NumKong ISA probe for Sapphire Rapids, AVX-512F/BW/DQ/VL plus FP16.
 */
#if defined(__APPLE__)
#error "AVX-512 not available on macOS"
#endif

#if !defined(__AVX512FP16__)
#error "Feature not available"
#endif
#include <immintrin.h>
int main(void) {
    volatile float one = 1.0f;
    __m512h a = _mm512_set1_ph((_Float16)one);
    __m512h b = _mm512_set1_ph((_Float16)(one + one));
    __m512h c = _mm512_fmadd_ph(a, b, a);
    return (int)_mm_extract_epi16(_mm256_castsi256_si128(_mm512_castsi512_si256((__m512i)c)), 0) != 0 ? 0 : 1;
}
