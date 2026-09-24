/* NumKong ISA probe: Sapphire Rapids AMX (AMX-TILE + AMX-INT8 + AMX-BF16 + AVX-512 FP16) */
#if defined(__APPLE__)
#error "AVX-512 not available on macOS"
#endif

#if defined(__FreeBSD__)
#error "AMX not supported on FreeBSD"
#endif

#if !defined(__AMX_INT8__) || !defined(__AMX_BF16__) || !defined(__AVX512FP16__)
#error "Feature not available"
#endif
#include <immintrin.h>
int main(void) {
    volatile int zero = 0;
    _tile_release();
    return zero;
}
