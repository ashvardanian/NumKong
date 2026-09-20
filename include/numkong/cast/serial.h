/**
 *  @brief SWAR-accelerated Type Conversions for SIMD-free CPUs.
 *  @file include/numkong/cast/serial.h
 *  @author Ash Vardanian
 *  @date January 2, 2026
 */
#ifndef NK_CAST_SERIAL_H
#define NK_CAST_SERIAL_H

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Type Punned Loads and Stores

/** @brief Type-agnostic 32-bit full load (scalar). */
NK_INTERNAL void nk_load_b32_serial_(void const *src, nk_b32_vec_t *dst) { dst->u32 = *(nk_u32_t const *)src; }

/** @brief Type-agnostic 64-bit full load. */
NK_INTERNAL void nk_load_b64_serial_(void const *src, nk_b64_vec_t *dst) { dst->u64 = *(nk_u64_t const *)src; }

/** @brief Type-agnostic 128-bit full load. */
NK_INTERNAL void nk_load_b128_serial_(void const *src, nk_b128_vec_t *dst) {
    nk_u64_t const *s = (nk_u64_t const *)src;
    dst->u64s[0] = s[0], dst->u64s[1] = s[1];
}

/** @brief Type-agnostic 256-bit full load. */
NK_INTERNAL void nk_load_b256_serial_(void const *src, nk_b256_vec_t *dst) {
    nk_u64_t const *s = (nk_u64_t const *)src;
    dst->u64s[0] = s[0], dst->u64s[1] = s[1], dst->u64s[2] = s[2], dst->u64s[3] = s[3];
}

/** @brief Type-agnostic 32-bit full store (scalar). */
NK_INTERNAL void nk_store_b32_serial_(nk_b32_vec_t const *src, void *dst) { *(nk_u32_t *)dst = src->u32; }

/** @brief Type-agnostic 64-bit full store (scalar). */
NK_INTERNAL void nk_store_b64_serial_(nk_b64_vec_t const *src, void *dst) { *(nk_u64_t *)dst = src->u64; }

/** @brief Type-agnostic 128-bit store (serial, word-by-word). */
NK_INTERNAL void nk_store_b128_serial_(nk_b128_vec_t const *src, void *dst) {
    nk_u64_t *d = (nk_u64_t *)dst;
    d[0] = src->u64s[0];
    d[1] = src->u64s[1];
}

/** @brief Type-agnostic 256-bit store (serial, word-by-word). */
NK_INTERNAL void nk_store_b256_serial_(nk_b256_vec_t const *src, void *dst) {
    nk_u64_t *d = (nk_u64_t *)dst;
    d[0] = src->u64s[0];
    d[1] = src->u64s[1];
    d[2] = src->u64s[2];
    d[3] = src->u64s[3];
}

/** @brief Type-agnostic partial load for 64-bit elements (4 elements max) into 256-bit vector. */
NK_INTERNAL void nk_partial_load_b64x4_serial_(void const *src, nk_b256_vec_t *dst, nk_size_t n) {
    nk_u64_t const *s = (nk_u64_t const *)src;
    dst->u64s[0] = 0, dst->u64s[1] = 0, dst->u64s[2] = 0, dst->u64s[3] = 0;
    switch (n) {
    default:
    case 4: dst->u64s[3] = s[3]; // fallthrough
    case 3: dst->u64s[2] = s[2]; // fallthrough
    case 2: dst->u64s[1] = s[1]; // fallthrough
    case 1: dst->u64s[0] = s[0]; // fallthrough
    case 0: break;
    }
}

/** @brief Type-agnostic partial store for 64-bit elements (4 elements max) from 256-bit vector. */
NK_INTERNAL void nk_partial_store_b64x4_serial_(nk_b256_vec_t const *src, void *dst, nk_size_t n) {
    nk_u64_t *d = (nk_u64_t *)dst;
    switch (n) {
    default:
    case 4: d[3] = src->u64s[3]; // fallthrough
    case 3: d[2] = src->u64s[2]; // fallthrough
    case 2: d[1] = src->u64s[1]; // fallthrough
    case 1: d[0] = src->u64s[0]; // fallthrough
    case 0: break;
    }
}

NK_INTERNAL void nk_partial_load_b64x2_serial_(void const *src, nk_b128_vec_t *dst, nk_size_t n) {
    dst->u64s[0] = 0, dst->u64s[1] = 0;
    nk_u64_t const *s = (nk_u64_t const *)src;
    switch (n) {
    default:
    case 2: dst->u64s[1] = s[1]; // fallthrough
    case 1: dst->u64s[0] = s[0]; // fallthrough
    case 0: break;
    }
}

/** @brief Type-agnostic partial store for 64-bit elements (2 elements max) from 128-bit vector. */
NK_INTERNAL void nk_partial_store_b64x2_serial_(nk_b128_vec_t const *src, void *dst, nk_size_t n) {
    nk_u64_t *d = (nk_u64_t *)dst;
    switch (n) {
    default:
    case 2: d[1] = src->u64s[1]; // fallthrough
    case 1: d[0] = src->u64s[0]; // fallthrough
    case 0: break;
    }
}

/** @brief Type-agnostic partial load for 32-bit elements (8 elements max) into 256-bit vector. */
NK_INTERNAL void nk_partial_load_b32x8_serial_(void const *src, nk_b256_vec_t *dst, nk_size_t n) {
    dst->u64s[0] = 0, dst->u64s[1] = 0, dst->u64s[2] = 0, dst->u64s[3] = 0;
    nk_u32_t const *s = (nk_u32_t const *)src;
    switch (n) {
    default:
    case 8: dst->u32s[7] = s[7]; // fallthrough
    case 7: dst->u32s[6] = s[6]; // fallthrough
    case 6: dst->u32s[5] = s[5]; // fallthrough
    case 5: dst->u32s[4] = s[4]; // fallthrough
    case 4: dst->u32s[3] = s[3]; // fallthrough
    case 3: dst->u32s[2] = s[2]; // fallthrough
    case 2: dst->u32s[1] = s[1]; // fallthrough
    case 1: dst->u32s[0] = s[0]; // fallthrough
    case 0: break;
    }
}

/** @brief Type-agnostic partial store for 32-bit elements (8 elements max) from 256-bit vector. */
NK_INTERNAL void nk_partial_store_b32x8_serial_(nk_b256_vec_t const *src, void *dst, nk_size_t n) {
    nk_u32_t *d = (nk_u32_t *)dst;
    switch (n) {
    default:
    case 8: d[7] = src->u32s[7]; // fallthrough
    case 7: d[6] = src->u32s[6]; // fallthrough
    case 6: d[5] = src->u32s[5]; // fallthrough
    case 5: d[4] = src->u32s[4]; // fallthrough
    case 4: d[3] = src->u32s[3]; // fallthrough
    case 3: d[2] = src->u32s[2]; // fallthrough
    case 2: d[1] = src->u32s[1]; // fallthrough
    case 1: d[0] = src->u32s[0]; // fallthrough
    case 0: break;
    }
}

/** @brief Type-agnostic partial load for 32-bit elements (4 elements max) into 128-bit vector. */
NK_INTERNAL void nk_partial_load_b32x4_serial_(void const *src, nk_b128_vec_t *dst, nk_size_t n) {
    dst->u64s[0] = 0, dst->u64s[1] = 0;
    nk_u32_t const *s = (nk_u32_t const *)src;
    switch (n) {
    default:
    case 4: dst->u32s[3] = s[3]; // fallthrough
    case 3: dst->u32s[2] = s[2]; // fallthrough
    case 2: dst->u32s[1] = s[1]; // fallthrough
    case 1: dst->u32s[0] = s[0]; // fallthrough
    case 0: break;
    }
}

/** @brief Type-agnostic partial store for 32-bit elements (4 elements max) from 128-bit vector. */
NK_INTERNAL void nk_partial_store_b32x4_serial_(nk_b128_vec_t const *src, void *dst, nk_size_t n) {
    nk_u32_t *d = (nk_u32_t *)dst;
    switch (n) {
    default:
    case 4: d[3] = src->u32s[3]; // fallthrough
    case 3: d[2] = src->u32s[2]; // fallthrough
    case 2: d[1] = src->u32s[1]; // fallthrough
    case 1: d[0] = src->u32s[0]; // fallthrough
    case 0: break;
    }
}

/** @brief Type-agnostic partial load for 32-bit elements (2 elements max) into 64-bit vector. */
NK_INTERNAL void nk_partial_load_b32x2_serial_(void const *src, nk_b64_vec_t *dst, nk_size_t n) {
    dst->u64 = 0;
    nk_u32_t const *s = (nk_u32_t const *)src;
    switch (n) {
    default:
    case 2: dst->u32s[1] = s[1]; // fallthrough
    case 1: dst->u32s[0] = s[0]; // fallthrough
    case 0: break;
    }
}

/** @brief Type-agnostic partial load for 16-bit elements (8 elements max) into 128-bit vector. */
NK_INTERNAL void nk_partial_load_b16x8_serial_(void const *src, nk_b128_vec_t *dst, nk_size_t n) {
    dst->u64s[0] = 0, dst->u64s[1] = 0;
    nk_u16_t const *s = (nk_u16_t const *)src;
    switch (n) {
    default:
    case 8: dst->u16s[7] = s[7]; // fallthrough
    case 7: dst->u16s[6] = s[6]; // fallthrough
    case 6: dst->u16s[5] = s[5]; // fallthrough
    case 5: dst->u16s[4] = s[4]; // fallthrough
    case 4: dst->u16s[3] = s[3]; // fallthrough
    case 3: dst->u16s[2] = s[2]; // fallthrough
    case 2: dst->u16s[1] = s[1]; // fallthrough
    case 1: dst->u16s[0] = s[0]; // fallthrough
    case 0: break;
    }
}

/** @brief Type-agnostic partial store for 16-bit elements (8 elements max) from 128-bit vector. */
NK_INTERNAL void nk_partial_store_b16x8_serial_(nk_b128_vec_t const *src, void *dst, nk_size_t n) {
    nk_u16_t *d = (nk_u16_t *)dst;
    switch (n) {
    default:
    case 8: d[7] = src->u16s[7]; // fallthrough
    case 7: d[6] = src->u16s[6]; // fallthrough
    case 6: d[5] = src->u16s[5]; // fallthrough
    case 5: d[4] = src->u16s[4]; // fallthrough
    case 4: d[3] = src->u16s[3]; // fallthrough
    case 3: d[2] = src->u16s[2]; // fallthrough
    case 2: d[1] = src->u16s[1]; // fallthrough
    case 1: d[0] = src->u16s[0]; // fallthrough
    case 0: break;
    }
}

/** @brief Type-agnostic partial load for 16-bit elements (16 elements max) into 256-bit vector. */
NK_INTERNAL void nk_partial_load_b16x16_serial_(void const *src, nk_b256_vec_t *dst, nk_size_t n) {
    dst->u64s[0] = 0, dst->u64s[1] = 0, dst->u64s[2] = 0, dst->u64s[3] = 0;
    nk_u16_t const *s = (nk_u16_t const *)src;
    switch (n) {
    default:
    case 16: dst->u16s[15] = s[15]; // fallthrough
    case 15: dst->u16s[14] = s[14]; // fallthrough
    case 14: dst->u16s[13] = s[13]; // fallthrough
    case 13: dst->u16s[12] = s[12]; // fallthrough
    case 12: dst->u16s[11] = s[11]; // fallthrough
    case 11: dst->u16s[10] = s[10]; // fallthrough
    case 10: dst->u16s[9] = s[9];   // fallthrough
    case 9: dst->u16s[8] = s[8];    // fallthrough
    case 8: dst->u16s[7] = s[7];    // fallthrough
    case 7: dst->u16s[6] = s[6];    // fallthrough
    case 6: dst->u16s[5] = s[5];    // fallthrough
    case 5: dst->u16s[4] = s[4];    // fallthrough
    case 4: dst->u16s[3] = s[3];    // fallthrough
    case 3: dst->u16s[2] = s[2];    // fallthrough
    case 2: dst->u16s[1] = s[1];    // fallthrough
    case 1: dst->u16s[0] = s[0];    // fallthrough
    case 0: break;
    }
}

/** @brief Type-agnostic partial store for 16-bit elements (16 elements max) from 256-bit vector. */
NK_INTERNAL void nk_partial_store_b16x16_serial_(nk_b256_vec_t const *src, void *dst, nk_size_t n) {
    nk_u16_t *d = (nk_u16_t *)dst;
    switch (n) {
    default:
    case 16: d[15] = src->u16s[15]; // fallthrough
    case 15: d[14] = src->u16s[14]; // fallthrough
    case 14: d[13] = src->u16s[13]; // fallthrough
    case 13: d[12] = src->u16s[12]; // fallthrough
    case 12: d[11] = src->u16s[11]; // fallthrough
    case 11: d[10] = src->u16s[10]; // fallthrough
    case 10: d[9] = src->u16s[9];   // fallthrough
    case 9: d[8] = src->u16s[8];    // fallthrough
    case 8: d[7] = src->u16s[7];    // fallthrough
    case 7: d[6] = src->u16s[6];    // fallthrough
    case 6: d[5] = src->u16s[5];    // fallthrough
    case 5: d[4] = src->u16s[4];    // fallthrough
    case 4: d[3] = src->u16s[3];    // fallthrough
    case 3: d[2] = src->u16s[2];    // fallthrough
    case 2: d[1] = src->u16s[1];    // fallthrough
    case 1: d[0] = src->u16s[0];    // fallthrough
    case 0: break;
    }
}

/** @brief Type-agnostic partial load for 16-bit elements (4 elements max) into 64-bit vector. */
NK_INTERNAL void nk_partial_load_b16x4_serial_(void const *src, nk_b64_vec_t *dst, nk_size_t n) {
    dst->u64 = 0;
    nk_u16_t const *s = (nk_u16_t const *)src;
    switch (n) {
    default:
    case 4: dst->u16s[3] = s[3]; // fallthrough
    case 3: dst->u16s[2] = s[2]; // fallthrough
    case 2: dst->u16s[1] = s[1]; // fallthrough
    case 1: dst->u16s[0] = s[0]; // fallthrough
    case 0: break;
    }
}

/** @brief Type-agnostic partial store for 16-bit elements (4 elements max) from 64-bit vector. */
NK_INTERNAL void nk_partial_store_b16x4_serial_(void *dst, nk_b64_vec_t const *src, nk_size_t n) {
    nk_u16_t *d = (nk_u16_t *)dst;
    switch (n) {
    default:
    case 4: d[3] = src->u16s[3]; // fallthrough
    case 3: d[2] = src->u16s[2]; // fallthrough
    case 2: d[1] = src->u16s[1]; // fallthrough
    case 1: d[0] = src->u16s[0]; // fallthrough
    case 0: break;
    }
}

/** @brief Type-agnostic partial load for 8-bit elements (8 elements max) into 64-bit vector. */
NK_INTERNAL void nk_partial_load_b8x8_serial_(void const *src, nk_b64_vec_t *dst, nk_size_t n) {
    dst->u64 = 0;
    nk_u8_t const *s = (nk_u8_t const *)src;
    switch (n) {
    default:
    case 8: dst->u8s[7] = s[7]; // fallthrough
    case 7: dst->u8s[6] = s[6]; // fallthrough
    case 6: dst->u8s[5] = s[5]; // fallthrough
    case 5: dst->u8s[4] = s[4]; // fallthrough
    case 4: dst->u8s[3] = s[3]; // fallthrough
    case 3: dst->u8s[2] = s[2]; // fallthrough
    case 2: dst->u8s[1] = s[1]; // fallthrough
    case 1: dst->u8s[0] = s[0]; // fallthrough
    case 0: break;
    }
}

/** @brief Type-agnostic partial store for 8-bit elements (8 elements max) from 64-bit vector. */
NK_INTERNAL void nk_partial_store_b8x8_serial_(nk_b64_vec_t const *src, void *dst, nk_size_t n) {
    nk_u8_t *d = (nk_u8_t *)dst;
    switch (n) {
    default:
    case 8: d[7] = src->u8s[7]; // fallthrough
    case 7: d[6] = src->u8s[6]; // fallthrough
    case 6: d[5] = src->u8s[5]; // fallthrough
    case 5: d[4] = src->u8s[4]; // fallthrough
    case 4: d[3] = src->u8s[3]; // fallthrough
    case 3: d[2] = src->u8s[2]; // fallthrough
    case 2: d[1] = src->u8s[1]; // fallthrough
    case 1: d[0] = src->u8s[0]; // fallthrough
    case 0: break;
    }
}

/** @brief Type-agnostic partial store for 8-bit elements (16 elements max) from 128-bit vector. */
NK_INTERNAL void nk_partial_store_b8x16_serial_(nk_b128_vec_t const *src, void *dst, nk_size_t n) {
    nk_u8_t *d = (nk_u8_t *)dst;
    switch (n) {
    default:
    case 16: d[15] = src->u8s[15]; // fallthrough
    case 15: d[14] = src->u8s[14]; // fallthrough
    case 14: d[13] = src->u8s[13]; // fallthrough
    case 13: d[12] = src->u8s[12]; // fallthrough
    case 12: d[11] = src->u8s[11]; // fallthrough
    case 11: d[10] = src->u8s[10]; // fallthrough
    case 10: d[9] = src->u8s[9];   // fallthrough
    case 9: d[8] = src->u8s[8];    // fallthrough
    case 8: d[7] = src->u8s[7];    // fallthrough
    case 7: d[6] = src->u8s[6];    // fallthrough
    case 6: d[5] = src->u8s[5];    // fallthrough
    case 5: d[4] = src->u8s[4];    // fallthrough
    case 4: d[3] = src->u8s[3];    // fallthrough
    case 3: d[2] = src->u8s[2];    // fallthrough
    case 2: d[1] = src->u8s[1];    // fallthrough
    case 1: d[0] = src->u8s[0];    // fallthrough
    case 0: break;
    }
}

/** @brief Type-agnostic partial store for 8-bit elements (32 elements max) from 256-bit vector. */
NK_INTERNAL void nk_partial_store_b8x32_serial_(nk_b256_vec_t const *src, void *dst, nk_size_t n) {
    nk_u8_t *d = (nk_u8_t *)dst;
    switch (n) {
    default:
    case 32: d[31] = src->u8s[31]; // fallthrough
    case 31: d[30] = src->u8s[30]; // fallthrough
    case 30: d[29] = src->u8s[29]; // fallthrough
    case 29: d[28] = src->u8s[28]; // fallthrough
    case 28: d[27] = src->u8s[27]; // fallthrough
    case 27: d[26] = src->u8s[26]; // fallthrough
    case 26: d[25] = src->u8s[25]; // fallthrough
    case 25: d[24] = src->u8s[24]; // fallthrough
    case 24: d[23] = src->u8s[23]; // fallthrough
    case 23: d[22] = src->u8s[22]; // fallthrough
    case 22: d[21] = src->u8s[21]; // fallthrough
    case 21: d[20] = src->u8s[20]; // fallthrough
    case 20: d[19] = src->u8s[19]; // fallthrough
    case 19: d[18] = src->u8s[18]; // fallthrough
    case 18: d[17] = src->u8s[17]; // fallthrough
    case 17: d[16] = src->u8s[16]; // fallthrough
    case 16: d[15] = src->u8s[15]; // fallthrough
    case 15: d[14] = src->u8s[14]; // fallthrough
    case 14: d[13] = src->u8s[13]; // fallthrough
    case 13: d[12] = src->u8s[12]; // fallthrough
    case 12: d[11] = src->u8s[11]; // fallthrough
    case 11: d[10] = src->u8s[10]; // fallthrough
    case 10: d[9] = src->u8s[9];   // fallthrough
    case 9: d[8] = src->u8s[8];    // fallthrough
    case 8: d[7] = src->u8s[7];    // fallthrough
    case 7: d[6] = src->u8s[6];    // fallthrough
    case 6: d[5] = src->u8s[5];    // fallthrough
    case 5: d[4] = src->u8s[4];    // fallthrough
    case 4: d[3] = src->u8s[3];    // fallthrough
    case 3: d[2] = src->u8s[2];    // fallthrough
    case 2: d[1] = src->u8s[1];    // fallthrough
    case 1: d[0] = src->u8s[0];    // fallthrough
    case 0: break;
    }
}

/** @brief Type-agnostic partial load for 8-bit elements (16 elements max) into 128-bit vector. */
NK_INTERNAL void nk_partial_load_b8x16_serial_(void const *src, nk_b128_vec_t *dst, nk_size_t n) {
    dst->u64s[0] = 0, dst->u64s[1] = 0;
    nk_u8_t const *s = (nk_u8_t const *)src;
    switch (n) {
    default:
    case 16: dst->u8s[15] = s[15]; // fallthrough
    case 15: dst->u8s[14] = s[14]; // fallthrough
    case 14: dst->u8s[13] = s[13]; // fallthrough
    case 13: dst->u8s[12] = s[12]; // fallthrough
    case 12: dst->u8s[11] = s[11]; // fallthrough
    case 11: dst->u8s[10] = s[10]; // fallthrough
    case 10: dst->u8s[9] = s[9];   // fallthrough
    case 9: dst->u8s[8] = s[8];    // fallthrough
    case 8: dst->u8s[7] = s[7];    // fallthrough
    case 7: dst->u8s[6] = s[6];    // fallthrough
    case 6: dst->u8s[5] = s[5];    // fallthrough
    case 5: dst->u8s[4] = s[4];    // fallthrough
    case 4: dst->u8s[3] = s[3];    // fallthrough
    case 3: dst->u8s[2] = s[2];    // fallthrough
    case 2: dst->u8s[1] = s[1];    // fallthrough
    case 1: dst->u8s[0] = s[0];    // fallthrough
    case 0: break;
    }
}

/** @brief Type-agnostic partial load for 8-bit elements (4 elements max) into 32-bit vector. */
NK_INTERNAL nk_b32_vec_t nk_partial_load_b8x4_serial_(void const *src, nk_size_t n) {
    nk_b32_vec_t dst = {0};
    nk_u8_t const *s = (nk_u8_t const *)src;
    switch (n) {
    default:
    case 4: dst.u8s[3] = s[3]; // fallthrough
    case 3: dst.u8s[2] = s[2]; // fallthrough
    case 2: dst.u8s[1] = s[1]; // fallthrough
    case 1: dst.u8s[0] = s[0]; // fallthrough
    case 0: break;
    }
    return dst;
}

/** @brief Partial store for 8-bit elements (up to 4) from nk_b32_vec_t. */
NK_INTERNAL void nk_partial_store_b8x4_serial_(nk_b32_vec_t const *src, void *dst, nk_size_t n) {
    nk_u8_t *d = (nk_u8_t *)dst;
    switch (n) {
    default:
    case 4: d[3] = src->u8s[3]; // fallthrough
    case 3: d[2] = src->u8s[2]; // fallthrough
    case 2: d[1] = src->u8s[1]; // fallthrough
    case 1: d[0] = src->u8s[0]; // fallthrough
    case 0: break;
    }
}

/** @brief Partial load for 8-bit elements (32 max) into 256-bit vector (zeros in remaining slots). */
NK_INTERNAL void nk_partial_load_b8x32_serial_(void const *src, nk_b256_vec_t *dst, nk_size_t n) {
    dst->u64s[0] = 0, dst->u64s[1] = 0, dst->u64s[2] = 0, dst->u64s[3] = 0;
    nk_u8_t const *s = (nk_u8_t const *)src;
    switch (n) {
    default:
    case 32: dst->u8s[31] = s[31]; // fallthrough
    case 31: dst->u8s[30] = s[30]; // fallthrough
    case 30: dst->u8s[29] = s[29]; // fallthrough
    case 29: dst->u8s[28] = s[28]; // fallthrough
    case 28: dst->u8s[27] = s[27]; // fallthrough
    case 27: dst->u8s[26] = s[26]; // fallthrough
    case 26: dst->u8s[25] = s[25]; // fallthrough
    case 25: dst->u8s[24] = s[24]; // fallthrough
    case 24: dst->u8s[23] = s[23]; // fallthrough
    case 23: dst->u8s[22] = s[22]; // fallthrough
    case 22: dst->u8s[21] = s[21]; // fallthrough
    case 21: dst->u8s[20] = s[20]; // fallthrough
    case 20: dst->u8s[19] = s[19]; // fallthrough
    case 19: dst->u8s[18] = s[18]; // fallthrough
    case 18: dst->u8s[17] = s[17]; // fallthrough
    case 17: dst->u8s[16] = s[16]; // fallthrough
    case 16: dst->u8s[15] = s[15]; // fallthrough
    case 15: dst->u8s[14] = s[14]; // fallthrough
    case 14: dst->u8s[13] = s[13]; // fallthrough
    case 13: dst->u8s[12] = s[12]; // fallthrough
    case 12: dst->u8s[11] = s[11]; // fallthrough
    case 11: dst->u8s[10] = s[10]; // fallthrough
    case 10: dst->u8s[9] = s[9];   // fallthrough
    case 9: dst->u8s[8] = s[8];    // fallthrough
    case 8: dst->u8s[7] = s[7];    // fallthrough
    case 7: dst->u8s[6] = s[6];    // fallthrough
    case 6: dst->u8s[5] = s[5];    // fallthrough
    case 5: dst->u8s[4] = s[4];    // fallthrough
    case 4: dst->u8s[3] = s[3];    // fallthrough
    case 3: dst->u8s[2] = s[2];    // fallthrough
    case 2: dst->u8s[1] = s[1];    // fallthrough
    case 1: dst->u8s[0] = s[0];    // fallthrough
    case 0: break;
    }
}

/** @brief Partial load for 4-bit nibbles (64 max = 32 bytes) into 256-bit vector (zeros in remaining slots). */
NK_INTERNAL void nk_partial_load_b4x64_serial_(void const *src, nk_b256_vec_t *dst, nk_size_t n) {
    dst->u64s[0] = 0, dst->u64s[1] = 0, dst->u64s[2] = 0, dst->u64s[3] = 0;
    nk_u8_t const *s = (nk_u8_t const *)src;
    nk_size_t n_bytes = nk_size_divide_round_up_(n, 2);
    for (nk_size_t i = 0; i < n_bytes && i < 32; i++) dst->u8s[i] = s[i];
}

/** @brief Partial load for 4-bit nibbles (32 max = 16 bytes) into 128-bit vector (zeros in remaining slots). */
NK_INTERNAL void nk_partial_load_b4x32_serial_(void const *src, nk_b128_vec_t *dst, nk_size_t n) {
    dst->u64s[0] = 0, dst->u64s[1] = 0;
    nk_u8_t const *s = (nk_u8_t const *)src;
    nk_size_t n_bytes = nk_size_divide_round_up_(n, 2);
    for (nk_size_t i = 0; i < n_bytes && i < 16; i++) dst->u8s[i] = s[i];
}

/** @brief Partial load for 1-bit elements (128 max = 16 bytes) into 128-bit vector (zeros in remaining slots). */
NK_INTERNAL void nk_partial_load_b1x128_serial_(void const *src, nk_b128_vec_t *dst, nk_size_t n_bits) {
    dst->u64s[0] = 0, dst->u64s[1] = 0;
    nk_u8_t const *s = (nk_u8_t const *)src;
    nk_size_t n_bytes = nk_size_divide_round_up_(n_bits, 8);
    for (nk_size_t i = 0; i < n_bytes && i < 16; i++) dst->u8s[i] = s[i];
}

/** @brief Partial load for binary (u1) data into 256-bit vector, converting n_bits → n_bytes. */
NK_INTERNAL void nk_partial_load_b1x256_serial_(void const *src, nk_b256_vec_t *dst, nk_size_t n_bits) {
    nk_size_t n_bytes = nk_size_divide_round_up_(n_bits, 8);
    nk_partial_load_b8x32_serial_(src, dst, n_bytes);
}

/** @brief Partial load for 4-bit nibbles (16 max = 8 bytes) into 64-bit vector (zeros in remaining slots). */
NK_INTERNAL void nk_partial_load_b4x16_serial_(void const *src, nk_b64_vec_t *dst, nk_size_t n) {
    dst->u64 = 0;
    nk_u8_t const *s = (nk_u8_t const *)src;
    nk_size_t n_bytes = nk_size_divide_round_up_(n, 2);
    for (nk_size_t i = 0; i < n_bytes && i < 8; i++) ((nk_u8_t *)&dst->u64)[i] = s[i];
}

/** @brief Strided partial load for 32-bit elements (4 max) into 128-bit vector. */
NK_INTERNAL void nk_strided_load_b32x4_serial_(void const *src, nk_size_t stride_elements, nk_b128_vec_t *dst,
                                               nk_size_t n) {
    dst->u64s[0] = 0, dst->u64s[1] = 0;
    nk_u32_t const *s = (nk_u32_t const *)src;
    for (nk_size_t i = 0; i < n && i < 4; ++i) dst->u32s[i] = s[i * stride_elements];
}

/** @brief Strided partial load for 16-bit elements (8 max) into 128-bit vector. */
NK_INTERNAL void nk_strided_load_b16x8_serial_(void const *src, nk_size_t stride_elements, nk_b128_vec_t *dst,
                                               nk_size_t n) {
    dst->u64s[0] = 0, dst->u64s[1] = 0;
    nk_u16_t const *s = (nk_u16_t const *)src;
    for (nk_size_t i = 0; i < n && i < 8; ++i) dst->u16s[i] = s[i * stride_elements];
}

/** @brief Strided partial load for 8-bit elements (16 max) into 128-bit vector. */
NK_INTERNAL void nk_strided_load_b8x16_serial_(void const *src, nk_size_t stride_elements, nk_b128_vec_t *dst,
                                               nk_size_t n) {
    dst->u64s[0] = 0, dst->u64s[1] = 0;
    nk_u8_t const *s = (nk_u8_t const *)src;
    for (nk_size_t i = 0; i < n && i < 16; ++i) dst->u8s[i] = s[i * stride_elements];
}

#pragma endregion Type Punned Loads and Stores

/*  Keep the serial conversions below actually scalar, regardless of build type.
 *  Without this, -O3 + LTO can vectorize or clone the serial kernels under AVX-512
 *  callers in dispatch_*.c, which wastes binary and breaks the nk_*_serial-as-scalar-oracle
 *  contract. See dots/serial.h. */
#if defined(__clang__)
#pragma clang attribute push(__attribute__((noinline)), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-slp-vectorize", "no-ipa-cp-clone", "no-inline")
#endif

/**
 *  @brief Expands an `f16` (IEEE-754 16-bit) to a `float`.
 *
 *  Handles all IEEE-754 edge cases:
 *
 *       Input        F16 Hex   F32 Hex       Description
 *       +0           0x0000    0x00000000    Positive zero
 *       -0           0x8000    0x80000000    Negative zero
 *       +inf         0x7C00    0x7F800000    Positive infinity
 *       -inf         0xFC00    0xFF800000    Negative infinity
 *       NaN          0x7E00    0x7FC00000    Quiet NaN (payload preserved)
 *       Min normal   0x0400    0x38800000    2⁻¹⁴
 *       Max normal   0x7BFF    0x477FE000    65504
 *       Min denorm   0x0001    0x33800000    2⁻²⁴
 *       Max denorm   0x03FF    0x387FC000    2⁻¹⁴ - 2⁻²⁴
 *
 *  https://stackoverflow.com/a/60047308
 *  https://gist.github.com/milhidaka/95863906fe828198f47991c813dbe233
 *  https://github.com/OpenCyphal/libcanard/blob/636795f4bc395f56af8d2c61d3757b5e762bb9e5/canard.c#L811-L834
 */
NK_PUBLIC void nk_f16_to_f32_serial(nk_f16_t const *src, nk_f32_t *dest) {
#if NK_NATIVE_F16
    *dest = (nk_f32_t)(*src);
#else
    unsigned short x;
    nk_copy_bytes_(&x, src, 2);

    unsigned int sign = (x >> 15) & 1;
    unsigned int exponent = (x >> 10) & 0x1F;
    unsigned int mantissa = x & 0x03FF;

    nk_fui32_t conv;

    if (exponent == 0) {
        if (mantissa == 0) {
            // Zero (preserve sign)
            conv.u = sign << 31;
        }
        else {
            // Denormal: value = mantissa × 2⁻²⁴
            // Use FPU normalization, then subtract 24 from exponent
            nk_fui32_t temp;
            temp.f = (float)mantissa;
            conv.u = (sign << 31) | (temp.u - 0x0C000000);
        }
    }
    else if (exponent == 31) {
        // Infinity (mantissa=0) or NaN (mantissa!=0)
        conv.u = (sign << 31) | 0x7F800000 | (mantissa << 13);
    }
    else {
        // Normal: rebias exponent (127-15=112), shift mantissa
        conv.u = (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13);
    }

    *dest = conv.f;
#endif
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

/** @brief Load 4 × f16 from memory and upcast them to 4 × f32. */
NK_INTERNAL void nk_load_f16x4_to_f32x4_serial_(void const *src, nk_b128_vec_t *dst) {
    nk_f16_t const *scalars = (nk_f16_t const *)src;
    nk_f16_to_f32_serial(scalars + 0, dst->f32s + 0);
    nk_f16_to_f32_serial(scalars + 1, dst->f32s + 1);
    nk_f16_to_f32_serial(scalars + 2, dst->f32s + 2);
    nk_f16_to_f32_serial(scalars + 3, dst->f32s + 3);
}

/** @brief Partial load for up to 4 × f16 with upcast to 4 × f32. */
NK_INTERNAL void nk_partial_load_f16x4_to_f32x4_serial_(nk_f16_t const *src, nk_b128_vec_t *dst, nk_size_t n) {
    dst->u64s[0] = 0, dst->u64s[1] = 0;
    switch (n) {
    default:
    case 4: nk_f16_to_f32_serial(src + 3, dst->f32s + 3); // fallthrough
    case 3: nk_f16_to_f32_serial(src + 2, dst->f32s + 2); // fallthrough
    case 2: nk_f16_to_f32_serial(src + 1, dst->f32s + 1); // fallthrough
    case 1: nk_f16_to_f32_serial(src + 0, dst->f32s + 0); // fallthrough
    case 0: break;
    }
}

#if defined(__clang__)
#pragma clang attribute push(__attribute__((noinline)), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-slp-vectorize", "no-ipa-cp-clone", "no-inline")
#endif

/**
 *  @brief Compresses a `float` to an `f16` (IEEE-754 16-bit).
 *
 *  Handles all IEEE-754 edge cases with round-to-nearest:
 *
 *      Input           F32 Hex       F16 Hex   Description
 *      +0              0x00000000    0x0000    Positive zero
 *      -0              0x80000000    0x8000    Negative zero
 *      +inf            0x7F800000    0x7C00    Positive infinity
 *      -inf            0xFF800000    0xFC00    Negative infinity
 *      NaN             0x7FC00000    0x7E00    Quiet NaN (payload truncated)
 *      1.0             0x3F800000    0x3C00    Normal number
 *      65504           0x477FE000    0x7BFF    Max f16 normal
 *      65520+          >0x477FE000   0x7C00    Overflow → infinity
 *      2⁻¹⁴           0x38800000    0x0400    Min f16 normal
 *      2⁻²⁴           0x33800000    0x0001    Min f16 denormal
 *      <2⁻²⁵          <0x33000000   0x0000    Underflow → zero
 *
 *  https://stackoverflow.com/a/60047308
 *  https://gist.github.com/milhidaka/95863906fe828198f47991c813dbe233
 *  https://github.com/OpenCyphal/libcanard/blob/636795f4bc395f56af8d2c61d3757b5e762bb9e5/canard.c#L811-L834
 */
NK_PUBLIC void nk_f32_to_f16_serial(nk_f32_t const *src, nk_f16_t *dest) {
#if NK_NATIVE_F16
    *dest = (nk_f16_t)(*src);
#else
    nk_fui32_t conv;
    conv.f = *src;

    unsigned int sign = (conv.u >> 31) & 1;
    unsigned int exponent = (conv.u >> 23) & 0xFF;
    unsigned int mantissa = conv.u & 0x007FFFFF;

    unsigned short result;

    if (exponent == 0) {
        // Zero or f32 denormal → f16 zero
        result = (unsigned short)(sign << 15);
    }
    else if (exponent == 255) {
        // Infinity or NaN
        unsigned short payload = (unsigned short)(mantissa >> 13);
        if (mantissa != 0 && payload == 0) payload = 1; // Preserve NaN-ness
        result = (unsigned short)((sign << 15) | 0x7C00 | payload);
    }
    else if (exponent <= 102) {
        // Below or at f16 denormal threshold
        // exp=102 with mant=0 is exactly 2^-25 (tie point, rounds to 0 per round-to-even)
        // exp=102 with mant>0 is above tie point (rounds to smallest denormal 0x0001)
        if (exponent == 102 && mantissa > 0) result = (unsigned short)((sign << 15) | 0x0001);
        else result = (unsigned short)(sign << 15);
    }
    else if (exponent < 113) {
        // F16 denormal range (exp 103-112) with IEEE 754 round-to-nearest-even
        unsigned int shift = 113 - exponent;
        unsigned int shift_amount = shift + 13;
        unsigned long long full_mant = 0x00800000ULL | mantissa;

        // Extract result before rounding
        unsigned int mant = (unsigned int)(full_mant >> shift_amount);

        // IEEE 754 round-to-nearest-even: round up if round_bit is set AND
        // (sticky_bits are nonzero OR result is odd)
        unsigned int round_bit = (full_mant >> (shift_amount - 1)) & 1;
        unsigned long long sticky_bits = full_mant & ((1ULL << (shift_amount - 1)) - 1);

        if (round_bit && (sticky_bits || (mant & 1))) mant++;

        result = (unsigned short)((sign << 15) | mant);
    }
    else if (exponent < 143) {
        // Normal f16 range with IEEE 754 round-to-nearest-even
        unsigned int f16_exp = exponent - 112;
        unsigned int f16_mant = mantissa >> 13;

        // IEEE 754 rounding: check round bit (bit 12) and sticky bits (bits 0-11)
        unsigned int round_bit = (mantissa >> 12) & 1;
        unsigned int sticky_bits = mantissa & 0xFFF;

        if (round_bit && (sticky_bits || (f16_mant & 1))) {
            f16_mant++;
            if (f16_mant > 0x3FF) f16_mant = 0, f16_exp++;
        }

        if (f16_exp > 30) result = (unsigned short)((sign << 15) | 0x7C00);
        else result = (unsigned short)((sign << 15) | (f16_exp << 10) | f16_mant);
    }
    else {
        // Overflow → infinity
        result = (unsigned short)((sign << 15) | 0x7C00);
    }

    nk_copy_bytes_(dest, &result, 2);
#endif
}

/**
 *  @brief For compilers that don't natively support the `__bf16` type,
 *          upcasts contents into a more conventional `float`.
 *
 *  https://stackoverflow.com/questions/55253233/convert-fp32-to-bfloat16-in-c/55254307#55254307
 *  https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus
 */
NK_PUBLIC void nk_bf16_to_f32_serial(nk_bf16_t const *src, nk_f32_t *dest) {
#if NK_NATIVE_BF16
    *dest = (nk_f32_t)(*src);
#else
    unsigned short x;
    nk_copy_bytes_(&x, src, 2);
    nk_fui32_t conv;
    conv.u = x << 16; // Zero extends the mantissa
    *dest = conv.f;
#endif
}

/**
 *  @brief Compresses a `float` to a `bf16` representation.
 *
 *  https://stackoverflow.com/questions/55253233/convert-fp32-to-bfloat16-in-c/55254307#55254307
 *  https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus
 */
NK_PUBLIC void nk_f32_to_bf16_serial(nk_f32_t const *src, nk_bf16_t *dest) {
#if NK_NATIVE_BF16
    *dest = (nk_bf16_t)(*src);
#else
    nk_fui32_t conv;
    conv.f = *src;
    // IEEE 754 round-to-nearest-even: add (0x7FFF + LSB)
    unsigned int lsb = (conv.u >> 16) & 1;
    conv.u += 0x7FFF + lsb;
    conv.u >>= 16;
    // Use an intermediate variable to ensure correct behavior on big-endian systems.
    // Copying directly from `&conv.u` would copy the wrong bytes on big-endian,
    // since the lower 16 bits are at offset 2, not offset 0.
    unsigned short result = (unsigned short)conv.u;
    nk_copy_bytes_(dest, &result, 2);
#endif
}

/**
 *  @brief Convert FP8 E4M3 to IEEE 754 single-precision float.
 *
 *  E4M3 (FP8) format: 1 sign bit, 4 exponent bits (bias=7), 3 mantissa bits.
 *  Range: [-448, +448], no ∞, only two NaN encodings (0x7F, 0xFF).
 *  Subnormal values: (-1)ˢ × mantissa × 2⁻⁹ = mantissa / 512.
 *
 *  Special value mappings (E4M3 → F32):
 *      Input        E4M3 Hex  F32 Hex       Description
 *      +0           0x00      0x00000000    Positive zero
 *      -0           0x80      0x80000000    Negative zero
 *      +NaN         0x7F      0x7FC00000    Quiet NaN (exp=15, mant!=0)
 *      -NaN         0xFF      0xFFC00000    Quiet NaN (signed)
 *      +448 (max)   0x7E      0x43E00000    Max normal = 448
 *      -448         0xFE      0xC3E00000    Min normal = -448
 *      1.0          0x38      0x3F800000    Normal (exp=7, mant=0)
 *      Min denorm   0x01      0x3B000000    1/512 = 2⁻⁹
 *      Max denorm   0x07      0x3BE00000    7/512 = 7 × 2⁻⁹
 *
 *  References:
 *      https://arxiv.org/pdf/2209.05433 (NVIDIA/Intel/Arm FP8 paper)
 *      https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1
 *      https://onnx.ai/onnx/technical/float8.html
 */
NK_PUBLIC void nk_e4m3_to_f32_serial(nk_e4m3_t const *src, nk_f32_t *dest) {
    nk_u8_t raw = *src;
    nk_u32_t sign = (nk_u32_t)(raw & 0x80) << 24;
    nk_u32_t exponent = (raw >> 3) & 0x0Fu;
    nk_u32_t mantissa = raw & 0x07u;
    nk_fui32_t conv;

    if (exponent == 0) {
        if (mantissa == 0) {
            conv.u = sign;
            *dest = conv.f;
            return;
        }
        nk_f32_t value = (nk_f32_t)mantissa * (1.0f / 512.0f);
        *dest = sign ? -value : value;
        return;
    }
    // E4M3FN has no ∞. Only exp=15 && mant=7 is NaN.
    // exp=15 && mant=0..6 are normal values (256, 288, 320, 352, 384, 416, 448).
    if (exponent == 0x0Fu && mantissa == 7) {
        conv.u = sign | 0x7FC00000u; // F32 quiet NaN
        *dest = conv.f;
        return;
    }

    nk_u32_t f32_exponent = (exponent + 120u) << 23;
    nk_u32_t f32_mantissa = mantissa << 20;
    conv.u = sign | f32_exponent | f32_mantissa;
    *dest = conv.f;
}

/**
 *  @brief Convert IEEE 754 single-precision float to FP8 E4M3.
 *
 *  E4M3 (FP8) format: 1 sign bit, 4 exponent bits (bias=7), 3 mantissa bits.
 *  Range: [-448, +448], no ∞, only two NaN encodings.
 *  Rounding: RNE (Round to Nearest Even) per IEEE 754 / OCP FP8 spec.
 *  Subnormal threshold: values with |x| < 2⁻⁶ use subnormal encoding.
 *
 *  Special value mappings (F32 → E4M3):
 *      Input        F32 Hex       E4M3 Hex  Description
 *      +0           0x00000000    0x00      Positive zero
 *      -0           0x80000000    0x80      Negative zero
 *      +inf         0x7F800000    0x7E      Saturates to max (+448)
 *      -inf         0xFF800000    0xFE      Saturates to min (-448)
 *      NaN          0x7FC00000    0x7F      Quiet NaN
 *      1.0          0x3F800000    0x38      Normal (exp=7, mant=0)
 *      448+         >0x43E00000   0x7E      Overflow → max
 *      2⁻⁶          0x3E800000    0x08      Min normal
 *      ≤2⁻¹⁰        ≤0x3A800000   0x00      Underflow → zero (RNE boundary)
 *
 *  References:
 *      https://arxiv.org/pdf/2209.05433 (NVIDIA/Intel/Arm FP8 paper)
 *      https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1
 *      https://onnx.ai/onnx/technical/float8.html
 */
NK_PUBLIC void nk_f32_to_e4m3_serial(nk_f32_t const *src, nk_e4m3_t *dest) {
    nk_f32_t x = *src;
    nk_fui32_t conv;
    conv.f = x;
    nk_u32_t sign_bit = conv.u >> 31;
    nk_u32_t abs_bits = conv.u & 0x7FFFFFFFu;
    nk_u8_t sign = (nk_u8_t)(sign_bit << 7);

    // NaN → E4M3FN NaN (0x7F or 0xFF)
    if (abs_bits > 0x7F800000u) {
        *dest = (nk_e4m3_t)(sign | 0x7Fu);
        return;
    }
    // Infinity → saturate to max (0x7E or 0xFE), E4M3FN has no ∞
    if (abs_bits == 0x7F800000u) {
        *dest = (nk_e4m3_t)(sign | 0x7Eu);
        return;
    }

    if (abs_bits == 0) {
        *dest = (nk_e4m3_t)sign;
        return;
    }

    nk_f32_t abs_x = sign_bit ? -x : x;

    // Subnormal range: [0, 1/64). Use RNE rounding via scaled * 512.
    // The RNE boundary between 0 and 1/512 is at 0.5/512, not 1/512.
    if (abs_x < (1.0f / 64.0f)) {
        nk_f32_t scaled = abs_x * 512.0f;
        nk_i32_t mant = (nk_i32_t)scaled;
        nk_f32_t frac = scaled - (nk_f32_t)mant;
        if (frac > 0.5f || (frac == 0.5f && (mant & 1))) { ++mant; }
        // If rounds to 8, promote to first normal (exp_field=1, mantissa=0)
        if (mant > 7) {
            *dest = (nk_e4m3_t)(sign | 0x08u);
            return;
        }
        if (mant == 0) { *dest = (nk_e4m3_t)sign; }
        else { *dest = (nk_e4m3_t)(sign | (nk_u8_t)mant); }
        return;
    }

    nk_i32_t exp = (nk_i32_t)((abs_bits >> 23) & 0xFFu) - 127;
    nk_u32_t mantissa = abs_bits & 0x7FFFFFu;
    nk_u32_t significand = (1u << 23) | mantissa;
    nk_i32_t shift = 23 - 3;
    nk_u32_t remainder_mask = (1u << shift) - 1;
    nk_u32_t remainder = significand & remainder_mask;
    nk_u32_t halfway = 1u << (shift - 1);
    nk_u32_t significand_rounded = significand >> shift;
    if (remainder > halfway || (remainder == halfway && (significand_rounded & 1))) { ++significand_rounded; }
    if (significand_rounded == (1u << (3 + 1))) {
        significand_rounded >>= 1;
        ++exp;
    }
    if (exp > 8) {
        // Saturate to max value 448 = 0x7E (exp=15, mantissa=6). Note: 0x7F is NaN in e4m3FN.
        *dest = (nk_e4m3_t)(sign | 0x7Eu);
        return;
    }
    if (exp < -6) {
        nk_f32_t scaled = abs_x * 512.0f;
        nk_i32_t mant = (nk_i32_t)scaled;
        nk_f32_t frac = scaled - (nk_f32_t)mant;
        if (frac > 0.5f || (frac == 0.5f && (mant & 1))) { ++mant; }
        // If rounds to 8, promote to first normal (exp_field=1, mantissa=0)
        if (mant > 7) {
            *dest = (nk_e4m3_t)(sign | 0x08u);
            return;
        }
        if (mant == 0) { *dest = (nk_e4m3_t)sign; }
        else { *dest = (nk_e4m3_t)(sign | (nk_u8_t)mant); }
        return;
    }

    nk_u8_t exp_field = (nk_u8_t)(exp + 7);
    nk_u8_t mant_field = (nk_u8_t)(significand_rounded & 0x07u);
    // For exp_field=15, clamp mantissa to 6 to avoid NaN encoding (0x7F in e4m3FN)
    if (exp_field == 15 && mant_field > 6) { mant_field = 6; }
    *dest = (nk_e4m3_t)(sign | (exp_field << 3) | mant_field);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

/**
 *  @brief Convert FP8 E4M3 to IEEE 754 half-precision float.
 *
 *  E4M3 format: 1 sign bit, 4 exponent bits (bias=7), 3 mantissa bits.
 *  F16 format:  1 sign bit, 5 exponent bits (bias=15), 10 mantissa bits.
 *
 *  Conversion notes:
 *  - Normal values: F16_exp = E4M3_exp + 8, mantissa shifted left by 7 bits
 *  - Subnormals: mant × 2⁻⁹ (where 2⁻⁹ = 0x1800 in F16)
 *  - NaN (0x7F): maps to F16 quiet NaN (0x7E00)
 */
NK_INTERNAL void nk_e4m3_to_f16_serial(nk_e4m3_t const *src, nk_f16_t *dest) {
    nk_u8_t raw = *src;
    nk_u16_t sign = ((nk_u16_t)(raw & 0x80)) << 8;
    nk_u16_t mag = raw & 0x7F;
    nk_u16_t mant = raw & 0x07;
    nk_u16_t exp = (raw >> 3) & 0x0F;
    nk_fui16_t result;

    if (mag == 0x7F) {
        result.u = sign | 0x7E00; // NaN
    }
    else if (exp == 0) {
        // Subnormal: mant × 2⁻⁹, where 2⁻⁹ = 0x1800 in F16
        nk_fui16_t scale;
        scale.u = 0x1800;
        nk_fui16_t mant_f16;
        mant_f16.f = (nk_f16_t)mant;
        result.f = mant_f16.f * scale.f;
        result.u |= sign;
    }
    else {
        // Normal: F16 = sign | ((mag << 7) + 0x2000)
        result.u = sign | ((mag << 7) + 0x2000);
    }
    *dest = result.f;
}

/**
 *  @brief Convert FP8 E5M2 to IEEE 754 single-precision float.
 *
 *  E5M2 (FP8) format: 1 sign bit, 5 exponent bits (bias=15), 2 mantissa bits.
 *  Range: [-57344, +57344], supports infinity and NaN (IEEE 754 compatible).
 *  Subnormal values: (-1)ˢ × mantissa × 2⁻¹⁶ = mantissa / 65536.
 *
 *  Special value mappings (E5M2 → F32):
 *      Input        E5M2 Hex  F32 Hex       Description
 *      +0           0x00      0x00000000    Positive zero
 *      -0           0x80      0x80000000    Negative zero
 *      +inf         0x7C      0x7F800000    Positive infinity
 *      -inf         0xFC      0xFF800000    Negative infinity
 *      +NaN         0x7D-7F   0x7FC00000    Quiet NaN (exp=31, mant!=0)
 *      -NaN         0xFD-FF   0xFFC00000    Quiet NaN (signed)
 *      +57344 (max) 0x7B      0x47600000    Max normal
 *      1.0          0x3C      0x3F800000    Normal (exp=15, mant=0)
 *      Min denorm   0x01      0x37800000    1/65536 = 2⁻¹⁶
 *      Max denorm   0x03      0x38000000    3/65536 = 3 × 2⁻¹⁶
 *
 *  References:
 *      https://arxiv.org/pdf/2209.05433 (NVIDIA/Intel/Arm FP8 paper)
 *      https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1
 *      https://onnx.ai/onnx/technical/float8.html
 */
NK_INTERNAL void nk_e5m2_to_f32_manual_(nk_e5m2_t const *src, nk_f32_t *dest) {
    nk_u8_t raw = *src;
    nk_u32_t sign = (nk_u32_t)(raw & 0x80) << 24;
    nk_u32_t exponent = (raw >> 2) & 0x1Fu;
    nk_u32_t mantissa = raw & 0x03u;
    nk_fui32_t conv;

    if (exponent == 0) {
        if (mantissa == 0) {
            conv.u = sign;
            *dest = conv.f;
            return;
        }
        nk_f32_t value = (nk_f32_t)mantissa * (1.0f / 65536.0f);
        *dest = sign ? -value : value;
        return;
    }
    if (exponent == 0x1Fu) {
        if (mantissa == 0) { conv.u = sign | 0x7F800000u; }
        else { conv.u = sign | 0x7FC00000u; }
        *dest = conv.f;
        return;
    }

    nk_u32_t f32_exponent = (exponent + 112u) << 23;
    nk_u32_t f32_mantissa = mantissa << 21;
    conv.u = sign | f32_exponent | f32_mantissa;
    *dest = conv.f;
}

#if defined(__clang__)
#pragma clang attribute push(__attribute__((noinline)), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-slp-vectorize", "no-ipa-cp-clone", "no-inline")
#endif

NK_PUBLIC void nk_e5m2_to_f32_serial(nk_e5m2_t const *src, nk_f32_t *dest) {
    static nk_u32_t const lut[128] = {
        0x00000000, 0x37800000, 0x38000000, 0x38400000, // exp=0  sub
        0x38800000, 0x38A00000, 0x38C00000, 0x38E00000, // exp=1
        0x39000000, 0x39200000, 0x39400000, 0x39600000, // exp=2
        0x39800000, 0x39A00000, 0x39C00000, 0x39E00000, // exp=3
        0x3A000000, 0x3A200000, 0x3A400000, 0x3A600000, // exp=4
        0x3A800000, 0x3AA00000, 0x3AC00000, 0x3AE00000, // exp=5
        0x3B000000, 0x3B200000, 0x3B400000, 0x3B600000, // exp=6
        0x3B800000, 0x3BA00000, 0x3BC00000, 0x3BE00000, // exp=7
        0x3C000000, 0x3C200000, 0x3C400000, 0x3C600000, // exp=8
        0x3C800000, 0x3CA00000, 0x3CC00000, 0x3CE00000, // exp=9
        0x3D000000, 0x3D200000, 0x3D400000, 0x3D600000, // exp=10
        0x3D800000, 0x3DA00000, 0x3DC00000, 0x3DE00000, // exp=11
        0x3E000000, 0x3E200000, 0x3E400000, 0x3E600000, // exp=12
        0x3E800000, 0x3EA00000, 0x3EC00000, 0x3EE00000, // exp=13
        0x3F000000, 0x3F200000, 0x3F400000, 0x3F600000, // exp=14
        0x3F800000, 0x3FA00000, 0x3FC00000, 0x3FE00000, // exp=15
        0x40000000, 0x40200000, 0x40400000, 0x40600000, // exp=16
        0x40800000, 0x40A00000, 0x40C00000, 0x40E00000, // exp=17
        0x41000000, 0x41200000, 0x41400000, 0x41600000, // exp=18
        0x41800000, 0x41A00000, 0x41C00000, 0x41E00000, // exp=19
        0x42000000, 0x42200000, 0x42400000, 0x42600000, // exp=20
        0x42800000, 0x42A00000, 0x42C00000, 0x42E00000, // exp=21
        0x43000000, 0x43200000, 0x43400000, 0x43600000, // exp=22
        0x43800000, 0x43A00000, 0x43C00000, 0x43E00000, // exp=23
        0x44000000, 0x44200000, 0x44400000, 0x44600000, // exp=24
        0x44800000, 0x44A00000, 0x44C00000, 0x44E00000, // exp=25
        0x45000000, 0x45200000, 0x45400000, 0x45600000, // exp=26
        0x45800000, 0x45A00000, 0x45C00000, 0x45E00000, // exp=27
        0x46000000, 0x46200000, 0x46400000, 0x46600000, // exp=28
        0x46800000, 0x46A00000, 0x46C00000, 0x46E00000, // exp=29
        0x47000000, 0x47200000, 0x47400000, 0x47600000, // exp=30
        0x7F800000, 0x7FC00000, 0x7FC00000, 0x7FC00000, // inf, nan
    };
    nk_u8_t raw = *src;
    nk_u32_t sign = (nk_u32_t)(raw & 0x80) << 24;
    nk_fui32_t conv;
    conv.u = sign | lut[raw & 0x7F];
    *dest = conv.f;
}

/**
 *  @brief Convert IEEE 754 single-precision float to FP8 E5M2.
 *
 *  E5M2 (FP8) format: 1 sign bit, 5 exponent bits (bias=15), 2 mantissa bits.
 *  Range: [-57344, +57344], supports infinity and NaN (IEEE 754 compatible).
 *  Rounding: RNE (Round to Nearest Even) per IEEE 754 / OCP FP8 spec.
 *  Subnormal threshold: values with |x| < 2⁻¹⁴ use subnormal encoding.
 *
 *  Special value mappings (F32 → E5M2):
 *      Input        F32 Hex       E5M2 Hex  Description
 *      +0           0x00000000    0x00      Positive zero
 *      -0           0x80000000    0x80      Negative zero
 *      +inf         0x7F800000    0x7C      Positive infinity
 *      -inf         0xFF800000    0xFC      Negative infinity
 *      NaN          0x7FC00000    0x7D      Quiet NaN
 *      1.0          0x3F800000    0x3C      Normal (exp=15, mant=0)
 *      57344+       >0x47600000   0x7C      Overflow → infinity
 *      2⁻¹⁴         0x38800000    0x04      Min normal
 *      ≤2⁻¹⁷        ≤0x37000000   0x00      Underflow → zero (RNE boundary)
 *
 *  References:
 *      https://arxiv.org/pdf/2209.05433 (NVIDIA/Intel/Arm FP8 paper)
 *      https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1
 *      https://onnx.ai/onnx/technical/float8.html
 */
NK_PUBLIC void nk_f32_to_e5m2_serial(nk_f32_t const *src, nk_e5m2_t *dest) {
    nk_f32_t x = *src;
    nk_fui32_t conv;
    conv.f = x;
    nk_u32_t sign_bit = conv.u >> 31;
    nk_u32_t abs_bits = conv.u & 0x7FFFFFFFu;
    nk_u8_t sign = (nk_u8_t)(sign_bit << 7);

    if (abs_bits >= 0x7F800000u) {
        nk_u8_t mant = (abs_bits > 0x7F800000u) ? 0x01u : 0x00u;
        *dest = (nk_e5m2_t)(sign | 0x7Cu | mant);
        return;
    }

    if (abs_bits == 0) {
        *dest = (nk_e5m2_t)sign;
        return;
    }

    nk_f32_t abs_x = sign_bit ? -x : x;

    // Subnormal range: [0, 1/16384). Use RNE rounding via scaled * 65536.
    // The RNE boundary between 0 and 1/65536 is at 0.5/65536, not 1/65536.
    if (abs_x < (1.0f / 16384.0f)) {
        nk_f32_t scaled = abs_x * 65536.0f;
        nk_i32_t mant = (nk_i32_t)scaled;
        nk_f32_t frac = scaled - (nk_f32_t)mant;
        if (frac > 0.5f || (frac == 0.5f && (mant & 1))) { ++mant; }
        // If rounds to 4, promote to first normal (exp_field=1, mantissa=0)
        if (mant > 3) {
            *dest = (nk_e5m2_t)(sign | 0x04u);
            return;
        }
        if (mant == 0) { *dest = (nk_e5m2_t)sign; }
        else { *dest = (nk_e5m2_t)(sign | (nk_u8_t)mant); }
        return;
    }

    nk_i32_t exp = (nk_i32_t)((abs_bits >> 23) & 0xFFu) - 127;
    nk_u32_t mantissa = abs_bits & 0x7FFFFFu;
    nk_u32_t significand = (1u << 23) | mantissa;
    nk_i32_t shift = 23 - 2;
    nk_u32_t remainder_mask = (1u << shift) - 1;
    nk_u32_t remainder = significand & remainder_mask;
    nk_u32_t halfway = 1u << (shift - 1);
    nk_u32_t significand_rounded = significand >> shift;
    if (remainder > halfway || (remainder == halfway && (significand_rounded & 1))) { ++significand_rounded; }
    if (significand_rounded == (1u << (2 + 1))) {
        significand_rounded >>= 1;
        ++exp;
    }
    if (exp > 15) {
        *dest = (nk_e5m2_t)(sign | 0x7Cu);
        return;
    }
    if (exp < -14) {
        nk_f32_t scaled = abs_x * 65536.0f;
        nk_i32_t mant = (nk_i32_t)scaled;
        nk_f32_t frac = scaled - (nk_f32_t)mant;
        if (frac > 0.5f || (frac == 0.5f && (mant & 1))) { ++mant; }
        // If rounds to 4, promote to first normal (exp_field=1, mantissa=0)
        if (mant > 3) {
            *dest = (nk_e5m2_t)(sign | 0x04u);
            return;
        }
        if (mant == 0) { *dest = (nk_e5m2_t)sign; }
        else { *dest = (nk_e5m2_t)(sign | (nk_u8_t)mant); }
        return;
    }

    nk_u8_t exp_field = (nk_u8_t)(exp + 15);
    nk_u8_t mant_field = (nk_u8_t)(significand_rounded & 0x03u);
    *dest = (nk_e5m2_t)(sign | (exp_field << 2) | mant_field);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

/**
 *  @brief Convert FP8 E5M2 to IEEE 754 half-precision float.
 *
 *  E5M2 format: 1 sign bit, 5 exponent bits (bias=15), 2 mantissa bits.
 *  F16 format:  1 sign bit, 5 exponent bits (bias=15), 10 mantissa bits.
 *
 *  Since E5M2 and F16 share the same exponent bias (15), normal values
 *  convert by simply shifting the magnitude left by 8 bits.
 *
 *  Conversion notes:
 *  - Normal values: F16 = sign | (mag << 8)
 *  - Subnormals: mant × 2⁻¹⁶ (where 2⁻¹⁶ = 0x0100 in F16)
 *  - Infinity (0x7C): maps to F16 infinity (0x7C00)
 *  - NaN (0x7D-0x7F): maps to F16 quiet NaN (0x7E00)
 */
NK_INTERNAL void nk_e5m2_to_f16_manual_(nk_e5m2_t const *src, nk_f16_t *dest) {
    nk_u8_t raw = *src;
    nk_u16_t sign = ((nk_u16_t)(raw & 0x80)) << 8;
    nk_u16_t mag = raw & 0x7F;
    nk_u16_t mant = raw & 0x03;
    nk_u16_t exp = (raw >> 2) & 0x1F;
    nk_fui16_t result;

    if (exp == 0) {
        if (mant == 0) {
            result.u = sign; // Zero
        }
        else {
            // Subnormal: mant × 2⁻¹⁶, where 2⁻¹⁶ = 0x0100 in F16
            nk_fui16_t scale;
            scale.u = 0x0100;
            nk_fui16_t mant_f16;
            mant_f16.f = (nk_f16_t)mant;
            result.f = mant_f16.f * scale.f;
            result.u |= sign;
        }
    }
    else if (mag == 0x7C) {
        result.u = sign | 0x7C00; // Infinity
    }
    else if (mag > 0x7C) {
        result.u = sign | 0x7E00; // NaN
    }
    else {
        // Normal: E5M2 and F16 have same bias (15), just shift magnitude
        result.u = sign | ((nk_u16_t)mag << 8);
    }
    *dest = result.f;
}

NK_INTERNAL void nk_e5m2_to_f16_serial(nk_e5m2_t const *src, nk_f16_t *dest) {
    static nk_u16_t const lut[128] = {
        0x0000, 0x0100, 0x0200, 0x0300, // exp=0  sub
        0x0400, 0x0500, 0x0600, 0x0700, // exp=1
        0x0800, 0x0900, 0x0A00, 0x0B00, // exp=2
        0x0C00, 0x0D00, 0x0E00, 0x0F00, // exp=3
        0x1000, 0x1100, 0x1200, 0x1300, // exp=4
        0x1400, 0x1500, 0x1600, 0x1700, // exp=5
        0x1800, 0x1900, 0x1A00, 0x1B00, // exp=6
        0x1C00, 0x1D00, 0x1E00, 0x1F00, // exp=7
        0x2000, 0x2100, 0x2200, 0x2300, // exp=8
        0x2400, 0x2500, 0x2600, 0x2700, // exp=9
        0x2800, 0x2900, 0x2A00, 0x2B00, // exp=10
        0x2C00, 0x2D00, 0x2E00, 0x2F00, // exp=11
        0x3000, 0x3100, 0x3200, 0x3300, // exp=12
        0x3400, 0x3500, 0x3600, 0x3700, // exp=13
        0x3800, 0x3900, 0x3A00, 0x3B00, // exp=14
        0x3C00, 0x3D00, 0x3E00, 0x3F00, // exp=15
        0x4000, 0x4100, 0x4200, 0x4300, // exp=16
        0x4400, 0x4500, 0x4600, 0x4700, // exp=17
        0x4800, 0x4900, 0x4A00, 0x4B00, // exp=18
        0x4C00, 0x4D00, 0x4E00, 0x4F00, // exp=19
        0x5000, 0x5100, 0x5200, 0x5300, // exp=20
        0x5400, 0x5500, 0x5600, 0x5700, // exp=21
        0x5800, 0x5900, 0x5A00, 0x5B00, // exp=22
        0x5C00, 0x5D00, 0x5E00, 0x5F00, // exp=23
        0x6000, 0x6100, 0x6200, 0x6300, // exp=24
        0x6400, 0x6500, 0x6600, 0x6700, // exp=25
        0x6800, 0x6900, 0x6A00, 0x6B00, // exp=26
        0x6C00, 0x6D00, 0x6E00, 0x6F00, // exp=27
        0x7000, 0x7100, 0x7200, 0x7300, // exp=28
        0x7400, 0x7500, 0x7600, 0x7700, // exp=29
        0x7800, 0x7900, 0x7A00, 0x7B00, // exp=30
        0x7C00, 0x7E00, 0x7E00, 0x7E00, // inf, nan
    };
    nk_u8_t raw = *src;
    nk_u16_t sign = ((nk_u16_t)(raw & 0x80)) << 8;
    nk_fui16_t result;
    result.u = sign | lut[raw & 0x7F];
    *dest = result.f;
}

/**
 *  @brief Convert FP6 E2M3FN to IEEE 754 single-precision float.
 *
 *  E2M3FN (FP6) format: 1 sign bit, 2 exponent bits (bias=1), 3 mantissa bits.
 *  Range: [-7.5, +7.5], no infinity or NaN (OCP Microscaling FN format).
 *  Uses precomputed lookup table for all 64 possible values.
 *
 *  References:
 *      https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
 *      https://arxiv.org/abs/2401.14112 (FP6-LLM)
 */
NK_INTERNAL void nk_e2m3_to_f32_manual_(nk_e2m3_t const *src, nk_f32_t *dest) {
    nk_u8_t raw = *src;
    nk_u32_t sign = (nk_u32_t)((raw >> 5) & 0x01u) << 31;
    nk_u32_t exponent = (raw >> 3) & 0x03u;
    nk_u32_t mantissa = raw & 0x07u;
    nk_fui32_t conv;

    // Handle zero
    if (exponent == 0 && mantissa == 0) {
        conv.u = sign;
        *dest = conv.f;
        return;
    }

    // Handle subnormal (exp=0, mant!=0)
    if (exponent == 0) {
        // Subnormal: value = 2^(1-bias) * (mantissa / 2^p) = 2^0 * (mantissa / 8) = mantissa / 8
        nk_f32_t value = (nk_f32_t)mantissa * (1.0f / 8.0f);
        *dest = sign ? -value : value;
        return;
    }

    // Normal values: rebias from E2M3 (bias=1) to F32 (bias=127)
    // E2M3 exp range: 1-3 (unbiased: 0-2)
    // F32 needs: (e2m3_exp - 1) + 127 = e2m3_exp + 126
    nk_u32_t f32_exponent = (exponent + 126u) << 23;
    nk_u32_t f32_mantissa = mantissa << 20;
    conv.u = sign | f32_exponent | f32_mantissa;
    *dest = conv.f;
}

#if defined(__clang__)
#pragma clang attribute push(__attribute__((noinline)), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-slp-vectorize", "no-ipa-cp-clone", "no-inline")
#endif

NK_PUBLIC void nk_e2m3_to_f32_serial(nk_e2m3_t const *src, nk_f32_t *dest) {
    static nk_u32_t const lut[32] = {
        0x00000000, 0x3E000000, 0x3E800000, 0x3EC00000, 0x3F000000, 0x3F200000, 0x3F400000, 0x3F600000, // exp=0 sub
        0x3F800000, 0x3F900000, 0x3FA00000, 0x3FB00000, 0x3FC00000, 0x3FD00000, 0x3FE00000, 0x3FF00000, // exp=1
        0x40000000, 0x40100000, 0x40200000, 0x40300000, 0x40400000, 0x40500000, 0x40600000, 0x40700000, // exp=2
        0x40800000, 0x40900000, 0x40A00000, 0x40B00000, 0x40C00000, 0x40D00000, 0x40E00000, 0x40F00000, // exp=3
    };
    nk_u8_t raw = *src;
    nk_u32_t sign = (nk_u32_t)((raw >> 5) & 0x01u) << 31;
    nk_fui32_t conv;
    conv.u = sign | lut[raw & 0x1F];
    *dest = conv.f;
}

/**
 *  @brief Convert IEEE 754 single-precision float to FP6 E2M3FN.
 *
 *  E2M3FN (FP6) format: 1 sign bit, 2 exponent bits (bias=1), 3 mantissa bits.
 *  Range: [-7.5, +7.5], no ∞ or NaN. Saturates to max on overflow.
 *  Rounding: RNE (Round to Nearest Even) per IEEE 754.
 *  Subnormal threshold: values with |x| < 0.5 use subnormal encoding.
 */
NK_PUBLIC void nk_f32_to_e2m3_serial(nk_f32_t const *src, nk_e2m3_t *dest) {
    nk_f32_t x = *src;
    nk_fui32_t conv;
    conv.f = x;
    nk_u32_t sign_bit = conv.u >> 31;
    nk_u32_t abs_bits = conv.u & 0x7FFFFFFFu;
    nk_u8_t sign = (nk_u8_t)(sign_bit << 5);

    // Zero
    if (abs_bits == 0) {
        *dest = (nk_e2m3_t)sign;
        return;
    }

    nk_f32_t abs_x = sign_bit ? -x : x;

    // Clamp to E2M3FN range [-7.5, 7.5]
    // Max value: exp=3, mant=7 → (1 + 7/8) * 2^(3-1) = 1.875 * 4 = 7.5
    if (abs_x >= 7.5f) {
        *dest = (nk_e2m3_t)(sign | 0x1Fu); // Max: 0b011111
        return;
    }

    // Subnormal range: [0, 1.0). exp=0, mant encodes value/0.125
    if (abs_x < 1.0f) {
        nk_f32_t scaled = abs_x * 8.0f; // Scale to mantissa range [0, 8)
        nk_i32_t mant = (nk_i32_t)scaled;
        nk_f32_t frac = scaled - (nk_f32_t)mant;
        // RNE rounding
        if (frac > 0.5f || (frac == 0.5f && (mant & 1))) { ++mant; }
        // If rounds to 8, promote to first normal (exp=1, mant=0)
        if (mant > 7) {
            *dest = (nk_e2m3_t)(sign | 0x08u);
            return;
        }
        *dest = (nk_e2m3_t)(sign | (nk_u8_t)mant);
        return;
    }

    // Normal range: extract exponent and mantissa
    nk_i32_t exp = (nk_i32_t)((abs_bits >> 23) & 0xFFu) - 127;
    nk_u32_t mantissa = abs_bits & 0x7FFFFFu;
    nk_u32_t significand = (1u << 23) | mantissa;

    // Round mantissa from 23 to 3 bits
    nk_i32_t shift = 23 - 3;
    nk_u32_t remainder_mask = (1u << shift) - 1;
    nk_u32_t remainder = significand & remainder_mask;
    nk_u32_t halfway = 1u << (shift - 1);
    nk_u32_t significand_rounded = significand >> shift;

    // RNE rounding
    if (remainder > halfway || (remainder == halfway && (significand_rounded & 1))) { ++significand_rounded; }

    // Handle carry into exponent
    if (significand_rounded == (1u << 4)) {
        significand_rounded >>= 1;
        ++exp;
    }

    // Rebias exponent: e2m3_exp = f32_exp + 1
    nk_i32_t e2m3_exp = exp + 1;

    // Clamp to valid range
    if (e2m3_exp > 3) {
        *dest = (nk_e2m3_t)(sign | 0x1Fu); // Max value
        return;
    }
    if (e2m3_exp < 0) {
        *dest = (nk_e2m3_t)sign; // Underflow to zero
        return;
    }

    nk_u8_t exp_field = (nk_u8_t)e2m3_exp;
    nk_u8_t mant_field = (nk_u8_t)(significand_rounded & 0x07u);
    *dest = (nk_e2m3_t)(sign | (exp_field << 3) | mant_field);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

/**
 *  @brief Convert FP6 E3M2FN to IEEE 754 single-precision float.
 *
 *  E3M2FN (FP6) format: 1 sign bit, 3 exponent bits (bias=3), 2 mantissa bits.
 *  Range: [-28, +28], no infinity or NaN (OCP Microscaling FN format).
 */
NK_INTERNAL void nk_e3m2_to_f32_manual_(nk_e3m2_t const *src, nk_f32_t *dest) {
    nk_u8_t raw = *src;
    nk_u32_t sign = (nk_u32_t)((raw >> 5) & 0x01u) << 31;
    nk_u32_t exponent = (raw >> 2) & 0x07u;
    nk_u32_t mantissa = raw & 0x03u;
    nk_fui32_t conv;

    // Handle zero
    if (exponent == 0 && mantissa == 0) {
        conv.u = sign;
        *dest = conv.f;
        return;
    }

    // Handle subnormal (exp=0, mant!=0)
    if (exponent == 0) {
        // Subnormal: value = 2^(-2) * (mantissa / 4)
        nk_f32_t value = (nk_f32_t)mantissa * (1.0f / 16.0f); // 2^(-2) * (1/4) = 1/16
        *dest = sign ? -value : value;
        return;
    }

    // Normal values: rebias from E3M2 (bias=3) to F32 (bias=127)
    // E3M2 exp range: 1-7 (unbiased: -2 to 4)
    // F32 needs: (e3m2_exp - 3) + 127 = e3m2_exp + 124
    nk_u32_t f32_exponent = (exponent + 124u) << 23;
    nk_u32_t f32_mantissa = mantissa << 21;
    conv.u = sign | f32_exponent | f32_mantissa;
    *dest = conv.f;
}

#if defined(__clang__)
#pragma clang attribute push(__attribute__((noinline)), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-slp-vectorize", "no-ipa-cp-clone", "no-inline")
#endif

NK_PUBLIC void nk_e3m2_to_f32_serial(nk_e3m2_t const *src, nk_f32_t *dest) {
    static nk_u32_t const lut[32] = {
        0x00000000, 0x3D800000, 0x3E000000, 0x3E400000, // exp=0 sub
        0x3E800000, 0x3EA00000, 0x3EC00000, 0x3EE00000, // exp=1
        0x3F000000, 0x3F200000, 0x3F400000, 0x3F600000, // exp=2
        0x3F800000, 0x3FA00000, 0x3FC00000, 0x3FE00000, // exp=3
        0x40000000, 0x40200000, 0x40400000, 0x40600000, // exp=4
        0x40800000, 0x40A00000, 0x40C00000, 0x40E00000, // exp=5
        0x41000000, 0x41200000, 0x41400000, 0x41600000, // exp=6
        0x41800000, 0x41A00000, 0x41C00000, 0x41E00000, // exp=7
    };
    nk_u8_t raw = *src;
    nk_u32_t sign = (nk_u32_t)((raw >> 5) & 0x01u) << 31;
    nk_fui32_t conv;
    conv.u = sign | lut[raw & 0x1F];
    *dest = conv.f;
}

/**
 *  @brief Convert IEEE 754 single-precision float to FP6 E3M2FN.
 *
 *  E3M2FN (FP6) format: 1 sign bit, 3 exponent bits (bias=3), 2 mantissa bits.
 *  Range: [-28, +28], no ∞ or NaN. Saturates to max on overflow.
 *  Rounding: RNE (Round to Nearest Even) per IEEE 754.
 *  Subnormal threshold: values with |x| < 0.25 use subnormal encoding.
 */
NK_PUBLIC void nk_f32_to_e3m2_serial(nk_f32_t const *src, nk_e3m2_t *dest) {
    nk_f32_t x = *src;
    nk_fui32_t conv;
    conv.f = x;
    nk_u32_t sign_bit = conv.u >> 31;
    nk_u32_t abs_bits = conv.u & 0x7FFFFFFFu;
    nk_u8_t sign = (nk_u8_t)(sign_bit << 5);

    // Zero
    if (abs_bits == 0) {
        *dest = (nk_e3m2_t)sign;
        return;
    }

    nk_f32_t abs_x = sign_bit ? -x : x;

    // Clamp to E3M2FN range [-28, 28]
    // Max value: exp=7, mant=2 → (1 + 2/4) * 2^(7-3) = 1.5 * 16 = 24
    // Actually max is exp=7, mant=3 → (1 + 3/4) * 2⁴ = 1.75 * 16 = 28
    if (abs_x >= 28.0f) {
        *dest = (nk_e3m2_t)(sign | 0x1Fu); // Max: 0b011111 (exp=7, mant=3)
        return;
    }

    // Subnormal range: [0, 0.25). exp=0, mant encodes value/0.0625
    if (abs_x < 0.25f) {
        nk_f32_t scaled = abs_x * 16.0f; // Scale to mantissa range [0, 4)
        nk_i32_t mant = (nk_i32_t)scaled;
        nk_f32_t frac = scaled - (nk_f32_t)mant;
        // RNE rounding
        if (frac > 0.5f || (frac == 0.5f && (mant & 1))) { ++mant; }
        // If rounds to 4, promote to first normal (exp=1, mant=0)
        if (mant > 3) {
            *dest = (nk_e3m2_t)(sign | 0x04u);
            return;
        }
        *dest = (nk_e3m2_t)(sign | (nk_u8_t)mant);
        return;
    }

    // Normal range: extract exponent and mantissa
    nk_i32_t exp = (nk_i32_t)((abs_bits >> 23) & 0xFFu) - 127;
    nk_u32_t mantissa = abs_bits & 0x7FFFFFu;
    nk_u32_t significand = (1u << 23) | mantissa;

    // Round mantissa from 23 to 2 bits
    nk_i32_t shift = 23 - 2;
    nk_u32_t remainder_mask = (1u << shift) - 1;
    nk_u32_t remainder = significand & remainder_mask;
    nk_u32_t halfway = 1u << (shift - 1);
    nk_u32_t significand_rounded = significand >> shift;

    // RNE rounding
    if (remainder > halfway || (remainder == halfway && (significand_rounded & 1))) { ++significand_rounded; }

    // Handle carry into exponent
    if (significand_rounded == (1u << 3)) {
        significand_rounded >>= 1;
        ++exp;
    }

    // Rebias exponent: e3m2_exp = f32_exp + 3
    nk_i32_t e3m2_exp = exp + 3;

    // Clamp to valid range
    if (e3m2_exp > 7) {
        *dest = (nk_e3m2_t)(sign | 0x1Fu); // Max value
        return;
    }
    if (e3m2_exp < 0) {
        *dest = (nk_e3m2_t)sign; // Underflow to zero
        return;
    }

    nk_u8_t exp_field = (nk_u8_t)e3m2_exp;
    nk_u8_t mant_field = (nk_u8_t)(significand_rounded & 0x03u);
    *dest = (nk_e3m2_t)(sign | (exp_field << 2) | mant_field);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

NK_INTERNAL void nk_f16_to_f64_serial(nk_f16_t const *x, nk_f64_t *y) {
    nk_f32_t f32;
    nk_f16_to_f32_serial(x, &f32);
    *y = (nk_f64_t)f32;
}
NK_INTERNAL void nk_f64_to_f16_serial(nk_f64_t const *x, nk_f16_t *y) {
    nk_f32_t f32 = (nk_f32_t)*x;
    nk_f32_to_f16_serial(&f32, y);
}
NK_INTERNAL void nk_bf16_to_f64_serial(nk_bf16_t const *x, nk_f64_t *y) {
    nk_f32_t f32;
    nk_bf16_to_f32_serial(x, &f32);
    *y = (nk_f64_t)f32;
}
NK_INTERNAL void nk_f64_to_bf16_serial(nk_f64_t const *x, nk_bf16_t *y) {
    nk_f32_t f32 = (nk_f32_t)*x;
    nk_f32_to_bf16_serial(&f32, y);
}

/*  Convert floating-point numbers to integers with the project-wide narrowing policy:
 *  finite values are clamped and rounded to nearest, ties to even, infinities saturate,
 *  and NaNs map to zero.
 */
NK_INTERNAL nk_i64_t nk_rint_even_f64_to_i64_serial_(nk_f64_t x) {
    nk_i64_t integer = (nk_i64_t)x;
    nk_f64_t fraction = x - (nk_f64_t)integer;
    if (fraction > 0.5 || (fraction == 0.5 && (integer & 1))) ++integer;
    else if (fraction < -0.5 || (fraction == -0.5 && (integer & 1))) --integer;
    return integer;
}

NK_INTERNAL nk_u64_t nk_rint_even_f64_to_u64_serial_(nk_f64_t x) {
    nk_u64_t integer = (nk_u64_t)x;
    nk_f64_t fraction = x - (nk_f64_t)integer;
    if (fraction > 0.5 || (fraction == 0.5 && (integer & 1))) ++integer;
    return integer;
}

NK_INTERNAL void nk_f32_to_i8_serial(nk_f32_t const *x, nk_i8_t *y) {
    if (*x != *x) *y = 0; // For IEEE floating-point, NaN is the one value that is not equal to itself
    else *y = (nk_i8_t)nk_rint_even_f64_to_i64_serial_(*x > 127.0f ? 127.0 : (*x < -128.0f ? -128.0 : (nk_f64_t)*x));
}

NK_INTERNAL void nk_f32_to_u8_serial(nk_f32_t const *x, nk_u8_t *y) {
    if (*x != *x) *y = 0; // For IEEE floating-point, NaN is the one value that is not equal to itself
    else *y = (nk_u8_t)nk_rint_even_f64_to_u64_serial_(*x > 255.0f ? 255.0 : (*x < 0 ? 0.0 : (nk_f64_t)*x));
}

NK_INTERNAL void nk_f32_to_i16_serial(nk_f32_t const *x, nk_i16_t *y) {
    if (*x != *x) *y = 0; // For IEEE floating-point, NaN is the one value that is not equal to itself
    else
        *y = (nk_i16_t)nk_rint_even_f64_to_i64_serial_(*x > 32767.0f ? 32767.0
                                                                     : (*x < -32768.0f ? -32768.0 : (nk_f64_t)*x));
}

NK_INTERNAL void nk_f32_to_u16_serial(nk_f32_t const *x, nk_u16_t *y) {
    if (*x != *x) *y = 0; // For IEEE floating-point, NaN is the one value that is not equal to itself
    else *y = (nk_u16_t)nk_rint_even_f64_to_u64_serial_(*x > 65535.0f ? 65535.0 : (*x < 0 ? 0.0 : (nk_f64_t)*x));
}

NK_INTERNAL void nk_f64_to_i8_serial(nk_f64_t const *x, nk_i8_t *y) {
    if (*x != *x) *y = 0; // For IEEE floating-point, NaN is the one value that is not equal to itself
    else *y = (nk_i8_t)nk_rint_even_f64_to_i64_serial_(*x > 127.0 ? 127.0 : (*x < -128.0 ? -128.0 : *x));
}

NK_INTERNAL void nk_f64_to_u8_serial(nk_f64_t const *x, nk_u8_t *y) {
    if (*x != *x) *y = 0; // For IEEE floating-point, NaN is the one value that is not equal to itself
    else *y = (nk_u8_t)nk_rint_even_f64_to_u64_serial_(*x > 255.0 ? 255.0 : (*x < 0 ? 0.0 : *x));
}

NK_INTERNAL void nk_f64_to_i16_serial(nk_f64_t const *x, nk_i16_t *y) {
    if (*x != *x) *y = 0; // For IEEE floating-point, NaN is the one value that is not equal to itself
    else *y = (nk_i16_t)nk_rint_even_f64_to_i64_serial_(*x > 32767.0 ? 32767.0 : (*x < -32768.0 ? -32768.0 : *x));
}

NK_INTERNAL void nk_f64_to_u16_serial(nk_f64_t const *x, nk_u16_t *y) {
    if (*x != *x) *y = 0; // For IEEE floating-point, NaN is the one value that is not equal to itself
    else *y = (nk_u16_t)nk_rint_even_f64_to_u64_serial_(*x > 65535.0 ? 65535.0 : (*x < 0 ? 0.0 : *x));
}

NK_INTERNAL void nk_f64_to_i32_serial(nk_f64_t const *x, nk_i32_t *y) {
    if (*x != *x) *y = 0; // For IEEE floating-point, NaN is the one value that is not equal to itself
    else
        *y = (nk_i32_t)nk_rint_even_f64_to_i64_serial_(*x > 2147483647.0 ? 2147483647.0
                                                                         : (*x < -2147483648.0 ? -2147483648.0 : *x));
}

NK_INTERNAL void nk_f64_to_u32_serial(nk_f64_t const *x, nk_u32_t *y) {
    if (*x != *x) *y = 0; // For IEEE floating-point, NaN is the one value that is not equal to itself
    else *y = (nk_u32_t)nk_rint_even_f64_to_u64_serial_(*x > 4294967295.0 ? 4294967295.0 : (*x < 0 ? 0.0 : *x));
}

NK_INTERNAL void nk_f64_to_i64_serial(nk_f64_t const *x, nk_i64_t *y) {
    if (*x != *x) *y = 0; // For IEEE floating-point, NaN is the one value that is not equal to itself
    else
        *y = nk_rint_even_f64_to_i64_serial_(*x > 9223372036854775807.0
                                                 ? 9223372036854775807.0
                                                 : (*x < -9223372036854775808.0 ? -9223372036854775808.0 : *x));
}

NK_INTERNAL void nk_f64_to_u64_serial(nk_f64_t const *x, nk_u64_t *y) {
    if (*x != *x) *y = 0; // For IEEE floating-point, NaN is the one value that is not equal to itself
    else
        *y = nk_rint_even_f64_to_u64_serial_(*x > 18446744073709551615.0 ? 18446744073709551615.0
                                                                         : (*x < 0 ? 0.0 : *x));
}

NK_INTERNAL void nk_i64_to_i8_serial(nk_i64_t const *x, nk_i8_t *y) {
    *y = (nk_i8_t)(*x > 127ll ? 127ll : (*x < -128ll ? -128ll : *x));
}

NK_INTERNAL void nk_i64_to_u8_serial(nk_i64_t const *x, nk_u8_t *y) {
    *y = (nk_u8_t)(*x > 255ll ? 255ll : (*x < 0ll ? 0ll : *x));
}

NK_INTERNAL void nk_i64_to_i16_serial(nk_i64_t const *x, nk_i16_t *y) {
    *y = (nk_i16_t)(*x > 32767ll ? 32767ll : (*x < -32768ll ? -32768ll : *x));
}

NK_INTERNAL void nk_i64_to_u16_serial(nk_i64_t const *x, nk_u16_t *y) {
    *y = (nk_u16_t)(*x > 65535ll ? 65535ll : (*x < 0ll ? 0ll : *x));
}

NK_INTERNAL void nk_i64_to_i32_serial(nk_i64_t const *x, nk_i32_t *y) {
    *y = (nk_i32_t)(*x > 2147483647ll ? 2147483647ll : (*x < -2147483648ll ? -2147483648ll : *x));
}

NK_INTERNAL void nk_i64_to_u32_serial(nk_i64_t const *x, nk_u32_t *y) {
    *y = (nk_u32_t)(*x > 4294967295ll ? 4294967295ll : (*x < 0ll ? 0ll : *x));
}

NK_INTERNAL void nk_u64_to_i8_serial(nk_u64_t const *x, nk_i8_t *y) { *y = (nk_i8_t)(*x > 127ull ? 127ull : *x); }
NK_INTERNAL void nk_u64_to_u8_serial(nk_u64_t const *x, nk_u8_t *y) { *y = (nk_u8_t)(*x > 255ull ? 255ull : *x); }
NK_INTERNAL void nk_u64_to_i16_serial(nk_u64_t const *x, nk_i16_t *y) {
    *y = (nk_i16_t)(*x > 32767ull ? 32767ull : *x);
}
NK_INTERNAL void nk_u64_to_u16_serial(nk_u64_t const *x, nk_u16_t *y) {
    *y = (nk_u16_t)(*x > 65535ull ? 65535ull : *x);
}

NK_INTERNAL void nk_u64_to_i32_serial(nk_u64_t const *x, nk_i32_t *y) {
    *y = (nk_i32_t)(*x > 2147483647ull ? 2147483647ull : *x);
}

NK_INTERNAL void nk_u64_to_u32_serial(nk_u64_t const *x, nk_u32_t *y) {
    *y = (nk_u32_t)(*x > 4294967295ull ? 4294967295ull : *x);
}

NK_INTERNAL void nk_u64_to_i64_serial(nk_u64_t const *x, nk_i64_t *y) {
    *y = (nk_i64_t)(*x >= 9223372036854775807ull ? 9223372036854775807ll : *x);
}

NK_INTERNAL void nk_i8_to_u64_serial(nk_i8_t const *x, nk_u64_t *y) { *y = (nk_u64_t)(*x < 0 ? 0 : *x); }
NK_INTERNAL void nk_i16_to_u64_serial(nk_i16_t const *x, nk_u64_t *y) { *y = (nk_u64_t)(*x < 0 ? 0 : *x); }
NK_INTERNAL void nk_i32_to_u64_serial(nk_i32_t const *x, nk_u64_t *y) { *y = (nk_u64_t)(*x < 0 ? 0 : *x); }
NK_INTERNAL void nk_i64_to_u64_serial(nk_i64_t const *x, nk_u64_t *y) { *y = (nk_u64_t)(*x < 0 ? 0 : *x); }

NK_INTERNAL void nk_i64_to_f16_serial(nk_i64_t const *x, nk_f16_t *y) {
    nk_f32_t f32 = (nk_f32_t)*x;
    nk_f32_to_f16_serial(&f32, y);
}
NK_INTERNAL void nk_i64_to_bf16_serial(nk_i64_t const *x, nk_bf16_t *y) {
    nk_f32_t f32 = (nk_f32_t)*x;
    nk_f32_to_bf16_serial(&f32, y);
}
NK_INTERNAL void nk_u64_to_f16_serial(nk_u64_t const *x, nk_f16_t *y) {
    nk_f32_t f32 = (nk_f32_t)*x;
    nk_f32_to_f16_serial(&f32, y);
}
NK_INTERNAL void nk_u64_to_bf16_serial(nk_u64_t const *x, nk_bf16_t *y) {
    nk_f32_t f32 = (nk_f32_t)*x;
    nk_f32_to_bf16_serial(&f32, y);
}

#if defined(__clang__)
#pragma clang attribute push(__attribute__((noinline)), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-slp-vectorize", "no-ipa-cp-clone", "no-inline")
#endif

/** @brief Convert a pair of i4 (4-bit signed integer, -8 to 7) nibbles into signed integers. */
NK_PUBLIC void nk_i4x2_to_i8x2_serial(nk_i4x2_t const *src, nk_i8_t *dest) {
    nk_u8_t byte = *(nk_u8_t const *)src;
    nk_u8_t high_nibble = byte >> 4;
    nk_u8_t low_nibble = byte & 0x0F;
    // Sign extend: 0-7 → 0-7, 8-15 → -8 to -1
    dest[0] = (nk_i8_t)((high_nibble ^ 8) - 8);
    dest[1] = (nk_i8_t)((low_nibble ^ 8) - 8);
}

/** @brief Convert a pair of u4 (4-bit unsigned integer, 0 to 15) nibbles into unsigned integers. */
NK_PUBLIC void nk_u4x2_to_u8x2_serial(nk_u4x2_t const *src, nk_u8_t *dest) {
    nk_u8_t byte = *(nk_u8_t const *)src;
    dest[0] = byte >> 4;
    dest[1] = byte & 0x0F;
}

/**
 *  @brief Union for type-punned scalar values at language binding boundaries.
 *
 *  Used to bridge different type systems (Python, JavaScript, etc.) where
 *  scalars arrive as f64 but need to be passed to kernels as typed pointers.
 *  The caller fills the appropriate union member based on the target dtype,
 *  then passes the union address as `void const *` to kernel functions.
 */
typedef union NK_MAY_ALIAS_ nk_scalar_buffer_t {
    nk_u8_t bytes[16];
    nk_f64_t f64;
    nk_f32_t f32;
    nk_f16_t f16;
    nk_bf16_t bf16;
    nk_f64c_t f64c;
    nk_f32c_t f32c;
    nk_f16c_t f16c;
    nk_bf16c_t bf16c;
    nk_i64_t i64;
    nk_u64_t u64;
    nk_i32_t i32;
    nk_u32_t u32;
    nk_i16_t i16;
    nk_u16_t u16;
    nk_i8_t i8;
    nk_u8_t u8;
} nk_scalar_buffer_t;

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

/** @brief Reads a typed scalar from @p buf and writes the widened f64c into @p result.
 *  Real types set `.imag = 0`. Safe when @p result aliases @p buf (in-place conversion).
 *  @return 1 on success, 0 for unsupported types (sub-byte, unknown). */
NK_INTERNAL int nk_scalar_buffer_to_f64c(nk_scalar_buffer_t const *buf, nk_dtype_t dtype, nk_f64c_t *result) {
    // Snapshot input so `result` may alias `buf` (e.g. in-place conversion within a union).
    nk_scalar_buffer_t local;
    local.f64c = buf->f64c;
    result->real = 0, result->imag = 0;
    switch (dtype) {
    case nk_f64_k: result->real = local.f64; break;
    case nk_f32_k: result->real = (nk_f64_t)local.f32; break;
    case nk_f16_k:
        nk_f16_to_f32_serial(&local.f16, &local.f32);
        result->real = (nk_f64_t)local.f32;
        break;
    case nk_bf16_k:
        nk_bf16_to_f32_serial(&local.bf16, &local.f32);
        result->real = (nk_f64_t)local.f32;
        break;
    case nk_f64c_k: result->real = local.f64c.real, result->imag = local.f64c.imag; break;
    case nk_f32c_k: result->real = (nk_f64_t)local.f32c.real, result->imag = (nk_f64_t)local.f32c.imag; break;
    case nk_f16c_k:
        nk_f16_to_f32_serial(&local.f16c.real, &local.f32);
        result->real = (nk_f64_t)local.f32;
        nk_f16_to_f32_serial(&local.f16c.imag, &local.f32);
        result->imag = (nk_f64_t)local.f32;
        break;
    case nk_bf16c_k:
        nk_bf16_to_f32_serial(&local.bf16c.real, &local.f32);
        result->real = (nk_f64_t)local.f32;
        nk_bf16_to_f32_serial(&local.bf16c.imag, &local.f32);
        result->imag = (nk_f64_t)local.f32;
        break;
    case nk_i64_k: result->real = (nk_f64_t)local.i64; break;
    case nk_u64_k: result->real = (nk_f64_t)local.u64; break;
    case nk_i32_k: result->real = (nk_f64_t)local.i32; break;
    case nk_u32_k: result->real = (nk_f64_t)local.u32; break;
    case nk_i16_k: result->real = (nk_f64_t)local.i16; break;
    case nk_u16_k: result->real = (nk_f64_t)local.u16; break;
    case nk_i8_k: result->real = (nk_f64_t)local.i8; break;
    case nk_u8_k: result->real = (nk_f64_t)local.u8; break;
    case nk_e4m3_k:
        nk_e4m3_to_f32_serial(&local.u8, &local.f32);
        result->real = (nk_f64_t)local.f32;
        break;
    case nk_e5m2_k:
        nk_e5m2_to_f32_serial(&local.u8, &local.f32);
        result->real = (nk_f64_t)local.f32;
        break;
    case nk_e2m3_k:
        nk_e2m3_to_f32_serial(&local.u8, &local.f32);
        result->real = (nk_f64_t)local.f32;
        break;
    case nk_e3m2_k:
        nk_e3m2_to_f32_serial(&local.u8, &local.f32);
        result->real = (nk_f64_t)local.f32;
        break;
    default: return 0;
    }
    return 1;
}

/**
 *  @brief Converts up to 8x values from `from_ptr` buffer into 8x puned buffer objects
 *  into a complex 64-bit floating point representation.
 */
NK_INTERNAL void nk_scalar_buffers_to_f64c_(                           //
    void const *from_ptr, nk_dtype_t from_dtype, nk_size_t from_count, //
    nk_scalar_buffer_t to_buffers[nk_at_least_(8)]) {

    nk_size_t i;
    switch (from_dtype) {
    // Sub-byte: u1 - 8 bits from 1 byte, MSB-first
    case nk_u1_k: {
        nk_u8_t byte = *(nk_u8_t const *)from_ptr;
        for (i = 0; i < 8; ++i) to_buffers[i].f64c.real = (byte >> (7 - i)) & 1, to_buffers[i].f64c.imag = 0;
    } break;
    // Sub-byte: i4 - 8 nibbles from 4 bytes, high nibble = even index, sign-extended
    case nk_i4_k: {
        nk_i4x2_t const *pairs = (nk_i4x2_t const *)from_ptr;
        nk_i8_t unpacked[2];
        for (i = 0; i < 4; ++i) {
            nk_i4x2_to_i8x2_serial(&pairs[i], unpacked);
            to_buffers[i * 2].f64c.real = unpacked[0], to_buffers[i * 2].f64c.imag = 0;
            to_buffers[i * 2 + 1].f64c.real = unpacked[1], to_buffers[i * 2 + 1].f64c.imag = 0;
        }
    } break;
    // Sub-byte: u4 - 8 nibbles from 4 bytes, high nibble = even index
    case nk_u4_k: {
        nk_u4x2_t const *pairs = (nk_u4x2_t const *)from_ptr;
        nk_u8_t unpacked[2];
        for (i = 0; i < 4; ++i) {
            nk_u4x2_to_u8x2_serial(&pairs[i], unpacked);
            to_buffers[i * 2].f64c.real = unpacked[0], to_buffers[i * 2].f64c.imag = 0;
            to_buffers[i * 2 + 1].f64c.real = unpacked[1], to_buffers[i * 2 + 1].f64c.imag = 0;
        }
    } break;
    // All byte-or-larger types: stage through a separate buffer to avoid
    // variable-length memcpy and type-punned read on the same union —
    // a pattern that triggers an ICE in MSVC's ARM64 optimizer (C1001).
    default: {
        nk_size_t stride = nk_dtype_bits(from_dtype) / NK_BITS_PER_BYTE;
        nk_scalar_buffer_t staged;
        for (i = 0; i < from_count; ++i) {
            staged.u64 = 0;
            nk_copy_bytes_(&staged, (char const *)from_ptr + i * stride, stride);
            nk_scalar_buffer_to_f64c(&staged, from_dtype, &to_buffers[i].f64c);
        }
    } break;
    }
}

/** @brief Narrows an f64c @p value into the appropriate typed member of @p buf.
 *  Real types use only `.real`; complex types use both components.
 *  Safe when @p value aliases @p buf (in-place conversion).
 *  @note Integer targets (i64, i32, ...) go through f64 rounding — values beyond 2^53 may lose precision.
 *  @return 1 on success, 0 for unsupported types (sub-byte, unknown). */
NK_INTERNAL int nk_scalar_buffer_from_f64c(nk_f64c_t const *value, nk_scalar_buffer_t *buf, nk_dtype_t dtype) {
    // Snapshot input so `value` may point into `buf` (e.g. in-place conversion within a union).
    nk_f64c_t local = *value;
    nk_f32_t temporary_f32;
    switch (dtype) {
    case nk_f64_k: buf->f64 = local.real; break;
    case nk_f32_k: buf->f32 = (nk_f32_t)local.real; break;
    case nk_f16_k:
        temporary_f32 = (nk_f32_t)local.real;
        nk_f32_to_f16_serial(&temporary_f32, &buf->f16);
        break;
    case nk_bf16_k:
        temporary_f32 = (nk_f32_t)local.real;
        nk_f32_to_bf16_serial(&temporary_f32, &buf->bf16);
        break;
    case nk_f64c_k:
        buf->f64c.real = local.real;
        buf->f64c.imag = local.imag;
        break;
    case nk_f32c_k:
        buf->f32c.real = (nk_f32_t)local.real;
        buf->f32c.imag = (nk_f32_t)local.imag;
        break;
    case nk_f16c_k:
        temporary_f32 = (nk_f32_t)local.real;
        nk_f32_to_f16_serial(&temporary_f32, &buf->f16c.real);
        temporary_f32 = (nk_f32_t)local.imag;
        nk_f32_to_f16_serial(&temporary_f32, &buf->f16c.imag);
        break;
    case nk_bf16c_k:
        temporary_f32 = (nk_f32_t)local.real;
        nk_f32_to_bf16_serial(&temporary_f32, &buf->bf16c.real);
        temporary_f32 = (nk_f32_t)local.imag;
        nk_f32_to_bf16_serial(&temporary_f32, &buf->bf16c.imag);
        break;
    case nk_i64_k: nk_f64_to_i64_serial(&local.real, &buf->i64); break;
    case nk_u64_k: nk_f64_to_u64_serial(&local.real, &buf->u64); break;
    case nk_i32_k: nk_f64_to_i32_serial(&local.real, &buf->i32); break;
    case nk_u32_k: nk_f64_to_u32_serial(&local.real, &buf->u32); break;
    case nk_i16_k: nk_f64_to_i16_serial(&local.real, &buf->i16); break;
    case nk_u16_k: nk_f64_to_u16_serial(&local.real, &buf->u16); break;
    case nk_i8_k: nk_f64_to_i8_serial(&local.real, &buf->i8); break;
    case nk_u8_k: nk_f64_to_u8_serial(&local.real, &buf->u8); break;
    case nk_e4m3_k:
        temporary_f32 = (nk_f32_t)local.real;
        nk_f32_to_e4m3_serial(&temporary_f32, &buf->u8);
        break;
    case nk_e5m2_k:
        temporary_f32 = (nk_f32_t)local.real;
        nk_f32_to_e5m2_serial(&temporary_f32, &buf->u8);
        break;
    case nk_e2m3_k:
        temporary_f32 = (nk_f32_t)local.real;
        nk_f32_to_e2m3_serial(&temporary_f32, &buf->u8);
        break;
    case nk_e3m2_k:
        temporary_f32 = (nk_f32_t)local.real;
        nk_f32_to_e3m2_serial(&temporary_f32, &buf->u8);
        break;
    default: return 0;
    }
    return 1;
}

/**
 *  @brief Converts up to 8x values from `from_buffers` buffer into 8x typed scalars.
 */
NK_INTERNAL void nk_scalar_buffers_from_f64c_(              //
    nk_scalar_buffer_t const from_buffers[nk_at_least_(8)], //
    void *to_ptr, nk_dtype_t to_dtype, nk_size_t to_count) {

    nk_size_t i;
    switch (to_dtype) {
    // Sub-byte: u1 - 8 bits to 1 byte, MSB-first, non-zero → 1
    case nk_u1_k: {
        nk_u8_t *p = (nk_u8_t *)to_ptr;
        nk_u8_t byte = 0;
        for (i = 0; i < 8; ++i) byte |= (from_buffers[i].f64c.real != 0) << (7 - i);
        *p = byte;
    } break;
    // Sub-byte: i4 - 8 nibbles to 4 bytes, high nibble = even index
    case nk_i4_k: {
        nk_u8_t *p = (nk_u8_t *)to_ptr;
        for (i = 0; i < 4; ++i) {
            nk_f64_t high = from_buffers[i * 2].f64c.real, low = from_buffers[i * 2 + 1].f64c.real;
            high = high > 7 ? 7 : (high < -8 ? -8 : high);
            low = low > 7 ? 7 : (low < -8 ? -8 : low);
            p[i] = (nk_u8_t)((((nk_i8_t)high & 0x0F) << 4) | ((nk_i8_t)low & 0x0F));
        }
    } break;
    // Sub-byte: u4 - 8 nibbles to 4 bytes, high nibble = even index
    case nk_u4_k: {
        nk_u8_t *p = (nk_u8_t *)to_ptr;
        for (i = 0; i < 4; ++i) {
            nk_f64_t high = from_buffers[i * 2].f64c.real, low = from_buffers[i * 2 + 1].f64c.real;
            high = high > 15 ? 15 : (high < 0 ? 0 : high);
            low = low > 15 ? 15 : (low < 0 ? 0 : low);
            p[i] = (nk_u8_t)(((nk_u8_t)high << 4) | (nk_u8_t)low);
        }
    } break;
    // All byte-or-larger types: convert, then store relevant bytes
    default: {
        nk_size_t stride = nk_dtype_bits(to_dtype) / NK_BITS_PER_BYTE;
        nk_scalar_buffer_t tmp;
        for (i = 0; i < to_count; ++i) {
            nk_scalar_buffer_from_f64c(&from_buffers[i].f64c, &tmp, to_dtype);
            nk_copy_bytes_((char *)to_ptr + i * stride, &tmp, stride);
        }
    } break;
    }
}

/**
 *  @brief Load 8 values from typed buffer into `buf[i].i64` (lossless widening for signed integers).
 */
NK_INTERNAL void nk_scalar_buffers_to_i64_(                            //
    void const *from_ptr, nk_dtype_t from_dtype, nk_size_t from_count, //
    nk_scalar_buffer_t to_buffers[nk_at_least_(8)]) {                  //
    nk_size_t i;
    switch (from_dtype) {
    case nk_i64_k: {
        nk_i64_t const *p = (nk_i64_t const *)from_ptr;
        for (i = 0; i < from_count; ++i) to_buffers[i].i64 = p[i];
    } break;
    case nk_i32_k: {
        nk_i32_t const *p = (nk_i32_t const *)from_ptr;
        for (i = 0; i < from_count; ++i) to_buffers[i].i64 = p[i];
    } break;
    case nk_i16_k: {
        nk_i16_t const *p = (nk_i16_t const *)from_ptr;
        for (i = 0; i < from_count; ++i) to_buffers[i].i64 = p[i];
    } break;
    case nk_i8_k: {
        nk_i8_t const *p = (nk_i8_t const *)from_ptr;
        for (i = 0; i < from_count; ++i) to_buffers[i].i64 = p[i];
    } break;
    // Sub-byte: i4 - 4 bytes to 8 nibbles, sign-extend each nibble
    case nk_i4_k: {
        nk_i4x2_t const *pairs = (nk_i4x2_t const *)from_ptr;
        for (i = 0; i < 4; ++i) {
            nk_i8_t unpacked[2];
            nk_i4x2_to_i8x2_serial(&pairs[i], unpacked);
            to_buffers[i * 2].i64 = unpacked[0];
            to_buffers[i * 2 + 1].i64 = unpacked[1];
        }
    } break;
    case nk_u64_k: {
        nk_u64_t const *p = (nk_u64_t const *)from_ptr;
        for (i = 0; i < from_count; ++i) to_buffers[i].i64 = (nk_i64_t)p[i];
    } break;
    case nk_u32_k: {
        nk_u32_t const *p = (nk_u32_t const *)from_ptr;
        for (i = 0; i < from_count; ++i) to_buffers[i].i64 = (nk_i64_t)p[i];
    } break;
    case nk_u16_k: {
        nk_u16_t const *p = (nk_u16_t const *)from_ptr;
        for (i = 0; i < from_count; ++i) to_buffers[i].i64 = (nk_i64_t)p[i];
    } break;
    case nk_u8_k: {
        nk_u8_t const *p = (nk_u8_t const *)from_ptr;
        for (i = 0; i < from_count; ++i) to_buffers[i].i64 = (nk_i64_t)p[i];
    } break;
    case nk_u4_k: {
        nk_u8_t const *p = (nk_u8_t const *)from_ptr;
        for (i = 0; i < 4; ++i) {
            to_buffers[i * 2].i64 = (nk_i64_t)(p[i] >> 4);
            to_buffers[i * 2 + 1].i64 = (nk_i64_t)(p[i] & 0xF);
        }
    } break;
    default: break;
    }
}

/**
 *  @brief Export 8 `buf[i].i64` values to typed buffer with saturation on downcast.
 *  @note Only handles integer and sub-byte targets. Float/complex targets are silently skipped.
 */
NK_INTERNAL void nk_scalar_buffers_from_i64_(                //
    nk_scalar_buffer_t const from_buffers[nk_at_least_(8)],  //
    void *to_ptr, nk_dtype_t to_dtype, nk_size_t to_count) { //
    nk_size_t i;
    switch (to_dtype) {
    case nk_i64_k: {
        nk_i64_t *p = (nk_i64_t *)to_ptr;
        for (i = 0; i < to_count; ++i) p[i] = from_buffers[i].i64;
    } break;
    case nk_i32_k: {
        nk_i32_t *p = (nk_i32_t *)to_ptr;
        for (i = 0; i < to_count; ++i) nk_i64_to_i32_serial(&from_buffers[i].i64, &p[i]);
    } break;
    case nk_i16_k: {
        nk_i16_t *p = (nk_i16_t *)to_ptr;
        for (i = 0; i < to_count; ++i) nk_i64_to_i16_serial(&from_buffers[i].i64, &p[i]);
    } break;
    case nk_i8_k: {
        nk_i8_t *p = (nk_i8_t *)to_ptr;
        for (i = 0; i < to_count; ++i) nk_i64_to_i8_serial(&from_buffers[i].i64, &p[i]);
    } break;
    // Unsigned targets: clamp negatives to 0
    case nk_u64_k: {
        nk_u64_t *p = (nk_u64_t *)to_ptr;
        for (i = 0; i < to_count; ++i) nk_i64_to_u64_serial(&from_buffers[i].i64, &p[i]);
    } break;
    case nk_u32_k: {
        nk_u32_t *p = (nk_u32_t *)to_ptr;
        for (i = 0; i < to_count; ++i) nk_i64_to_u32_serial(&from_buffers[i].i64, &p[i]);
    } break;
    case nk_u16_k: {
        nk_u16_t *p = (nk_u16_t *)to_ptr;
        for (i = 0; i < to_count; ++i) nk_i64_to_u16_serial(&from_buffers[i].i64, &p[i]);
    } break;
    case nk_u8_k: {
        nk_u8_t *p = (nk_u8_t *)to_ptr;
        for (i = 0; i < to_count; ++i) nk_i64_to_u8_serial(&from_buffers[i].i64, &p[i]);
    } break;
    // Sub-byte: i4 - 8 nibbles to 4 bytes, clamp [-8,7]
    case nk_i4_k: {
        nk_i4x2_t *p = (nk_i4x2_t *)to_ptr;
        for (i = 0; i < 4; ++i) {
            nk_i64_t high = from_buffers[i * 2].i64, low = from_buffers[i * 2 + 1].i64;
            high = high > 7 ? 7 : (high < -8 ? -8 : high);
            low = low > 7 ? 7 : (low < -8 ? -8 : low);
            p[i] = (nk_u8_t)(((high & 0xF) << 4) | (low & 0xF));
        }
    } break;
    default: break;
    }
}

/**
 *  @brief Load 8 values from typed buffer into `buf[i].u64` (lossless widening for unsigned integers).
 */
NK_INTERNAL void nk_scalar_buffers_to_u64_(                            //
    void const *from_ptr, nk_dtype_t from_dtype, nk_size_t from_count, //
    nk_scalar_buffer_t to_buffers[nk_at_least_(8)]) {                  //
    nk_size_t i;
    switch (from_dtype) {
    case nk_u64_k: {
        nk_u64_t const *p = (nk_u64_t const *)from_ptr;
        for (i = 0; i < from_count; ++i) to_buffers[i].u64 = p[i];
    } break;
    case nk_u32_k: {
        nk_u32_t const *p = (nk_u32_t const *)from_ptr;
        for (i = 0; i < from_count; ++i) to_buffers[i].u64 = p[i];
    } break;
    case nk_u16_k: {
        nk_u16_t const *p = (nk_u16_t const *)from_ptr;
        for (i = 0; i < from_count; ++i) to_buffers[i].u64 = p[i];
    } break;
    case nk_u8_k: {
        nk_u8_t const *p = (nk_u8_t const *)from_ptr;
        for (i = 0; i < from_count; ++i) to_buffers[i].u64 = p[i];
    } break;
    // Sub-byte: u4 - 4 bytes to 8 nibbles, zero-extend
    case nk_u4_k: {
        nk_u4x2_t const *pairs = (nk_u4x2_t const *)from_ptr;
        for (i = 0; i < 4; ++i) {
            nk_u8_t unpacked[2];
            nk_u4x2_to_u8x2_serial(&pairs[i], unpacked);
            to_buffers[i * 2].u64 = unpacked[0];
            to_buffers[i * 2 + 1].u64 = unpacked[1];
        }
    } break;
    // Sub-byte: u1 - 1 byte to 8 bits, MSB-first
    case nk_u1_k: {
        nk_u8_t byte = *(nk_u8_t const *)from_ptr;
        for (i = 0; i < 8; ++i) to_buffers[i].u64 = (byte >> (7 - i)) & 1;
    } break;
    default: break;
    }
}

/**
 *  @brief Export 8 `buf[i].u64` values to typed buffer with saturation on downcast.
 *  @note Only handles integer and sub-byte targets. Float/complex targets are silently skipped.
 */
NK_INTERNAL void nk_scalar_buffers_from_u64_(                //
    nk_scalar_buffer_t const from_buffers[nk_at_least_(8)],  //
    void *to_ptr, nk_dtype_t to_dtype, nk_size_t to_count) { //
    nk_size_t i;
    switch (to_dtype) {
    case nk_u64_k: {
        nk_u64_t *p = (nk_u64_t *)to_ptr;
        for (i = 0; i < to_count; ++i) p[i] = from_buffers[i].u64;
    } break;
    case nk_u32_k: {
        nk_u32_t *p = (nk_u32_t *)to_ptr;
        for (i = 0; i < to_count; ++i) nk_u64_to_u32_serial(&from_buffers[i].u64, &p[i]);
    } break;
    case nk_u16_k: {
        nk_u16_t *p = (nk_u16_t *)to_ptr;
        for (i = 0; i < to_count; ++i) nk_u64_to_u16_serial(&from_buffers[i].u64, &p[i]);
    } break;
    case nk_u8_k: {
        nk_u8_t *p = (nk_u8_t *)to_ptr;
        for (i = 0; i < to_count; ++i) nk_u64_to_u8_serial(&from_buffers[i].u64, &p[i]);
    } break;
    // Signed targets: clamp to i64_max
    case nk_i64_k: {
        nk_i64_t *p = (nk_i64_t *)to_ptr;
        for (i = 0; i < to_count; ++i) nk_u64_to_i64_serial(&from_buffers[i].u64, &p[i]);
    } break;
    case nk_i32_k: {
        nk_i32_t *p = (nk_i32_t *)to_ptr;
        for (i = 0; i < to_count; ++i) nk_u64_to_i32_serial(&from_buffers[i].u64, &p[i]);
    } break;
    case nk_i16_k: {
        nk_i16_t *p = (nk_i16_t *)to_ptr;
        for (i = 0; i < to_count; ++i) nk_u64_to_i16_serial(&from_buffers[i].u64, &p[i]);
    } break;
    case nk_i8_k: {
        nk_i8_t *p = (nk_i8_t *)to_ptr;
        for (i = 0; i < to_count; ++i) nk_u64_to_i8_serial(&from_buffers[i].u64, &p[i]);
    } break;
    // Sub-byte: u4 - 8 nibbles to 4 bytes, clamp [0,15]
    case nk_u4_k: {
        nk_u4x2_t *p = (nk_u4x2_t *)to_ptr;
        for (i = 0; i < 4; ++i) {
            nk_u64_t high = from_buffers[i * 2].u64, low = from_buffers[i * 2 + 1].u64;
            high = high > 15 ? 15 : high;
            low = low > 15 ? 15 : low;
            p[i] = (nk_u8_t)((high << 4) | low);
        }
    } break;
    // Sub-byte: u1 - 8 bits to 1 byte, MSB-first, non-zero becomes 1
    case nk_u1_k: {
        nk_u8_t *p = (nk_u8_t *)to_ptr;
        nk_u8_t byte = 0;
        for (i = 0; i < 8; ++i) byte |= (from_buffers[i].u64 != 0) << (7 - i);
        *p = byte;
    } break;
    default: break;
    }
}

/** @brief Widens a typed scalar from @p buf into @p result as f64 (discards imaginary part).
 *  Safe when @p result aliases @p buf (in-place conversion). */
NK_INTERNAL int nk_scalar_buffer_to_f64(nk_scalar_buffer_t const *buf, nk_dtype_t dtype, nk_f64_t *result) {
    nk_f64c_t temporary_f64c;
    int ok = nk_scalar_buffer_to_f64c(buf, dtype, &temporary_f64c);
    *result = temporary_f64c.real;
    return ok;
}

/** @brief Narrows an f64 @p value into the appropriate typed member of @p buf.
 *  Safe when @p value aliases @p buf (in-place: `buf->f64 = x; from_f64(&buf->f64, buf, dtype)`).
 *  @note Integer targets go through f64 rounding — values beyond 2^53 may lose precision. */
NK_INTERNAL int nk_scalar_buffer_from_f64(nk_f64_t const *value, nk_scalar_buffer_t *buf, nk_dtype_t dtype) {
    nk_f64c_t temporary_f64c = {*value, 0};
    return nk_scalar_buffer_from_f64c(&temporary_f64c, buf, dtype);
}

#pragma region Public API

#if defined(__clang__)
#pragma clang attribute push(__attribute__((noinline)), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-slp-vectorize", "no-ipa-cp-clone", "no-inline")
#endif

NK_PUBLIC void nk_cast_serial(void const *from, nk_dtype_t from_type, nk_size_t n, void *to, nk_dtype_t to_type) {
    if (from_type == to_type) {
        nk_size_t size_bits = nk_dtype_bits(from_type);
        nk_size_t size_bytes = nk_size_divide_round_up_(n * size_bits, NK_BITS_PER_BYTE);
        if (size_bytes > 0) nk_copy_bytes_(to, from, size_bytes);
        return;
    }

    nk_size_t from_bits = nk_dtype_bits(from_type);
    nk_size_t to_bits = nk_dtype_bits(to_type);
    if (from_bits == 0 || to_bits == 0) return;

    // Byte steps per batch of NK_BITS_PER_BYTE elements
    nk_size_t from_step = from_bits;
    nk_size_t to_step = to_bits;

    nk_u8_t const *src = (nk_u8_t const *)from;
    nk_u8_t *dst = (nk_u8_t *)to;
    nk_dtype_family_t from_family = nk_dtype_family(from_type);
    nk_dtype_family_t to_family = nk_dtype_family(to_type);

    nk_size_t batches = n / NK_BITS_PER_BYTE;
    nk_size_t tail = n % NK_BITS_PER_BYTE;
    nk_scalar_buffer_t bufs[NK_BITS_PER_BYTE];

    // Both unsigned: u64 hub
    if (from_family == nk_dtype_family_uint_k && to_family == nk_dtype_family_uint_k) {
        for (nk_size_t b = 0; b < batches; ++b, src += from_step, dst += to_step) {
            nk_scalar_buffers_to_u64_(src, from_type, NK_BITS_PER_BYTE, bufs);
            nk_scalar_buffers_from_u64_(bufs, dst, to_type, NK_BITS_PER_BYTE);
        }
        if (tail) {
            nk_scalar_buffers_to_u64_(src, from_type, tail, bufs);
            nk_scalar_buffers_from_u64_(bufs, dst, to_type, tail);
        }
        return;
    }

    // Both integers, at least one signed: i64 hub
    if ((from_family == nk_dtype_family_int_k || from_family == nk_dtype_family_uint_k) &&
        (to_family == nk_dtype_family_int_k || to_family == nk_dtype_family_uint_k)) {
        for (nk_size_t b = 0; b < batches; ++b, src += from_step, dst += to_step) {
            nk_scalar_buffers_to_i64_(src, from_type, NK_BITS_PER_BYTE, bufs);
            nk_scalar_buffers_from_i64_(bufs, dst, to_type, NK_BITS_PER_BYTE);
        }
        if (tail) {
            nk_scalar_buffers_to_i64_(src, from_type, tail, bufs);
            nk_scalar_buffers_from_i64_(bufs, dst, to_type, tail);
        }
        return;
    }

    // Everything else: f64c hub (floats, complex, cross-category)
    for (nk_size_t b = 0; b < batches; ++b, src += from_step, dst += to_step) {
        nk_scalar_buffers_to_f64c_(src, from_type, NK_BITS_PER_BYTE, bufs);
        nk_scalar_buffers_from_f64c_(bufs, dst, to_type, NK_BITS_PER_BYTE);
    }
    if (tail) {
        nk_scalar_buffers_to_f64c_(src, from_type, tail, bufs);
        nk_scalar_buffers_from_f64c_(bufs, dst, to_type, tail);
    }
}

/** @brief Convert E4M3 to BF16 via F32 intermediate. */
NK_PUBLIC void nk_e4m3_to_bf16(nk_e4m3_t const *src, nk_bf16_t *dest) {
    nk_f32_t temp;
    nk_e4m3_to_f32_serial(src, &temp);
    nk_f32_to_bf16_serial(&temp, dest);
}

/** @brief Convert E5M2 to BF16 via F32 intermediate. */
NK_PUBLIC void nk_e5m2_to_bf16(nk_e5m2_t const *src, nk_bf16_t *dest) {
    nk_f32_t temp;
    nk_e5m2_to_f32_serial(src, &temp);
    nk_f32_to_bf16_serial(&temp, dest);
}

/** @brief Convert E2M3 to BF16 via F32 intermediate. */
NK_PUBLIC void nk_e2m3_to_bf16(nk_e2m3_t const *src, nk_bf16_t *dest) {
    nk_f32_t temp;
    nk_e2m3_to_f32_serial(src, &temp);
    nk_f32_to_bf16_serial(&temp, dest);
}

/** @brief Convert E3M2 to BF16 via F32 intermediate. */
NK_PUBLIC void nk_e3m2_to_bf16(nk_e3m2_t const *src, nk_bf16_t *dest) {
    nk_f32_t temp;
    nk_e3m2_to_f32_serial(src, &temp);
    nk_f32_to_bf16_serial(&temp, dest);
}

#pragma endregion Public API

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_CAST_SERIAL_H
