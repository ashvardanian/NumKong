/**
 *  @file probes/x86_genoa.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief NumKong ISA probe for Genoa, AVX-512F/BW/DQ/VL plus BF16.
 */
#if defined(__APPLE__)
#error "AVX-512 not available on macOS"
#endif

#if !defined(__AVX512BF16__)
#error "Feature not available"
#endif
#include <immintrin.h>
int main(void) {
    volatile float one = 1.0f;
    __m512 f = _mm512_set1_ps(one);
    __m256bh a = _mm512_cvtneps_pbh(f);
    __m512bh wide = (__m512bh)_mm512_castsi512_ps(_mm512_inserti64x4(_mm512_setzero_si512(), (__m256i)a, 0));
    __m512 r = _mm512_dpbf16_ps(_mm512_setzero_ps(), wide, wide);
    return _mm512_reduce_add_ps(r) >= 0.0f ? 0 : 1;
}
