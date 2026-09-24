/**
 *  @file probes/x86_graniteamx.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief NumKong ISA probe for Granite Rapids AMX, AMX-TILE plus AMX-FP16.
 */
#if defined(__APPLE__)
#error "AVX-512 not available on macOS"
#endif

#if defined(__FreeBSD__)
#error "AMX not supported on FreeBSD"
#endif

#if !defined(__AMX_FP16__)
#error "Feature not available"
#endif
#include <immintrin.h>
#include <amxfp16intrin.h>
int main(void) {
    volatile int zero = 0;
    _tile_release();
    return zero;
}
