/**
 *  @file probes/x86_sapphireamx.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief NumKong ISA probe for Sapphire Rapids AMX, AMX-TILE plus AMX-INT8.
 */
#if defined(__APPLE__)
#error "AVX-512 not available on macOS"
#endif

#if defined(__FreeBSD__)
#error "AMX not supported on FreeBSD"
#endif

#if !defined(__AMX_INT8__)
#error "Feature not available"
#endif
#include <immintrin.h>
int main(void) {
    volatile int zero = 0;
    _tile_release();
    return zero;
}
