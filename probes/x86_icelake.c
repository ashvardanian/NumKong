/**
 *  @file probes/x86_icelake.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief NumKong ISA probe for Ice Lake, AVX-512F/BW/DQ/VL plus VNNI, VBMI and VPOPCNTDQ.
 */
#if defined(__APPLE__)
#error "AVX-512 not available on macOS"
#endif

#if !defined(__AVX512VNNI__)
#error "Feature not available"
#endif
#include <immintrin.h>
int main(void) {
    volatile int one = 1;
    __m512i acc = _mm512_setzero_si512();
    __m512i a = _mm512_set1_epi8((char)one);
    __m512i b = _mm512_set1_epi8((char)one);
    acc = _mm512_dpbusd_epi32(acc, a, b);
    return (int)_mm512_reduce_add_epi32(acc) == 64 ? 0 : 1;
}
