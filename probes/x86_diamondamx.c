/* NumKong ISA probe: Diamond Rapids AMX (AMX-FP8 + AMX-AVX512) */
#if defined(__APPLE__)
#error "AMX not available on macOS"
#endif

#if defined(__FreeBSD__)
#error "AMX not supported on FreeBSD"
#endif

#if !defined(__AMX_FP8__) || !defined(__AMX_AVX512__)
#error "Feature not available"
#endif
#include <immintrin.h>
int main(void) {
    volatile int zero = 0;
    _tile_dphf8ps(0, 1, 2);
    volatile __m512 row = _mm512_castsi512_ps(_tile_movrow(0, zero));
    (void)row;
    _tile_release();
    return zero;
}
