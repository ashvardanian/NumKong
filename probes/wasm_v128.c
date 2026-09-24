/**
 *  @file probes/wasm_v128.c
 *  @author Ash Vardanian
 *  @date September 13, 2026
 *  @brief NumKong ISA probe for WASM SIMD128, @c v128.
 */
#if !defined(__wasm_simd128__)
#error "WASM SIMD128 not available"
#endif
#include <wasm_simd128.h>
int main(void) {
    v128_t a = wasm_i16x8_splat(1);
    v128_t b = wasm_i16x8_splat(2);
    v128_t c = wasm_i32x4_dot_i16x8(a, b);
    return wasm_i32x4_extract_lane(c, 0) > 0 ? 0 : 1;
}
