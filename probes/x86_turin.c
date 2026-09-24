/* NumKong ISA probe: Turin (AVX-512F + VP2INTERSECT + BF16) */
#if defined(__APPLE__)
#error "AVX-512 not available on macOS"
#endif

#if !defined(__AVX512VP2INTERSECT__) || !defined(__AVX512BF16__)
#error "Feature not available"
#endif
#include <immintrin.h>
int main(void) {
    volatile int val = 42;
    __m512i a = _mm512_set1_epi32(val);
    __m512i b = _mm512_set1_epi32(val);
    __mmask16 k0, k1;
    _mm512_2intersect_epi32(a, b, &k0, &k1);
    return k0 != 0 ? 0 : 1;
}
