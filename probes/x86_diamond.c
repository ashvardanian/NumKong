/* NumKong ISA probe: Diamond Rapids (AVX10.2) */
#if defined(__APPLE__)
#error "AVX-512 not available on macOS"
#endif

#if !defined(__AVX10_2__)
#error "Feature not available"
#endif
#include <immintrin.h>
int main(void) {
    volatile int one = 1;
    __m256i a = _mm256_set1_epi8((char)one);
    __m512h ah = _mm512_cvthf8_ph(a);
    __m512 acc = _mm512_dpph_ps(_mm512_setzero_ps(), ah, ah);
    return _mm512_reduce_add_ps(acc) != 0.0f ? 0 : 1;
}
