/**
 *  @file include/numkong/dots/ampere.cuh
 *  @author Ash Vardanian
 *  @date September 22, 2026
 *  @brief SIMD-accelerated Batched Dot Products for NVIDIA Ampere and newer.
 *
 *  @sa include/numkong/dots.h
 *
 *  Four warps own a @b [128,128] output tile, streaming 64-byte depth slabs through a three-stage
 *  `cp.async` pipeline into swizzled shared memory, so dtypes differ only in how each 32-byte
 *  sub-slab is multiplied. Float8 and E3M2 widen exactly into F16; E2M3, E2M1, I4 and U4 widen into
 *  I8 for exact integer MMA, with one output multiply undoing each widening's power of two. F64 and
 *  F32 have no tensor-core path worth taking and stay on the @c cuda capability.
 */
#ifndef NUMKONG_DOTS_AMPERE_CUH
#define NUMKONG_DOTS_AMPERE_CUH

#if NUMKONG_ARCH_CUDA_
#if NUMKONG_ARCH_CUDA_AMPERE_

#include "numkong/dots/cuda.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Configuration

enum {
    nk_cross_threads_ampere_k = 128,
    nk_cross_tile_ampere_k = 128,
    nk_cross_slab_bytes_ampere_k = 64,
    nk_cross_stages_ampere_k = 3,
    nk_cross_stage_bytes_ampere_k = nk_cross_tile_ampere_k * nk_cross_slab_bytes_ampere_k,
};

/** Shared memory of a tile: the ring of stages, A then B in each, whose first stage holds the row
 *  norms and then the column norms once every product is done. */
typedef union {
    unsigned char stages[nk_cross_stages_ampere_k][2][nk_cross_stage_bytes_ampere_k];
    nk_fui32_t norms[2 * nk_cross_tile_ampere_k];
} nk_cross_shared_ampere_t;

/*  Each dtype launches the one shape under its own stem, so the generators paste it. */
enum {
    nk_cross_threads_bf16_ampere_k = nk_cross_threads_ampere_k,
    nk_cross_tile_bf16_ampere_k = nk_cross_tile_ampere_k,
    nk_cross_threads_f16_ampere_k = nk_cross_threads_ampere_k,
    nk_cross_tile_f16_ampere_k = nk_cross_tile_ampere_k,
    nk_cross_threads_e5m2_ampere_k = nk_cross_threads_ampere_k,
    nk_cross_tile_e5m2_ampere_k = nk_cross_tile_ampere_k,
    nk_cross_threads_e4m3_ampere_k = nk_cross_threads_ampere_k,
    nk_cross_tile_e4m3_ampere_k = nk_cross_tile_ampere_k,
    nk_cross_threads_e3m2_ampere_k = nk_cross_threads_ampere_k,
    nk_cross_tile_e3m2_ampere_k = nk_cross_tile_ampere_k,
    nk_cross_threads_e2m3_ampere_k = nk_cross_threads_ampere_k,
    nk_cross_tile_e2m3_ampere_k = nk_cross_tile_ampere_k,
    nk_cross_threads_e2m3_symmetric_ampere_k = nk_cross_threads_ampere_k,
    nk_cross_tile_e2m3_symmetric_ampere_k = nk_cross_tile_ampere_k,
    nk_cross_threads_e2m1_ampere_k = nk_cross_threads_ampere_k,
    nk_cross_tile_e2m1_ampere_k = nk_cross_tile_ampere_k,
    nk_cross_threads_i8_ampere_k = nk_cross_threads_ampere_k,
    nk_cross_tile_i8_ampere_k = nk_cross_tile_ampere_k,
    nk_cross_threads_i4_ampere_k = nk_cross_threads_ampere_k,
    nk_cross_tile_i4_ampere_k = nk_cross_tile_ampere_k,
    nk_cross_threads_u8_ampere_k = nk_cross_threads_ampere_k,
    nk_cross_tile_u8_ampere_k = nk_cross_tile_ampere_k,
    nk_cross_threads_u4_ampere_k = nk_cross_threads_ampere_k,
    nk_cross_tile_u4_ampere_k = nk_cross_tile_ampere_k,
};

#pragma endregion Configuration

#pragma region Instructions

NUMKONG_DEVICE nk_u32_t nk_shared_address_ampere_(void const *pointer) {
    return (nk_u32_t)__cvta_generic_to_shared(pointer);
}

/* Copies 16 bytes, zero-filling past @p valid_bytes, so tails need no second path. */
NUMKONG_DEVICE void nk_copy_b128_async_ampere_(nk_u32_t shared, void const *global, nk_u32_t valid_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(shared), "l"(global), "r"(valid_bytes));
}

NUMKONG_DEVICE void nk_commit_async_ampere_(void) { asm volatile("cp.async.commit_group;\n" ::); }

/*  Waits until at most @p pending committed groups are in flight, capped at 2, since PTX
 *  takes an immediate. */
NUMKONG_DEVICE void nk_wait_async_ampere_(unsigned pending) {
    switch (pending) {
    case 0: asm volatile("cp.async.wait_group 0;\n" ::); break;
    case 1: asm volatile("cp.async.wait_group 1;\n" ::); break;
    default: asm volatile("cp.async.wait_group 2;\n" ::); break;
    }
}

NUMKONG_DEVICE void nk_load_matrices_x4_ampere_(nk_u32_t shared, nk_u32_t fragments[4]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(fragments[0]), "=r"(fragments[1]), "=r"(fragments[2]), "=r"(fragments[3])
                 : "r"(shared));
}

NUMKONG_DEVICE void nk_load_matrices_x4_transposed_ampere_(nk_u32_t shared, nk_u32_t fragments[4]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(fragments[0]), "=r"(fragments[1]), "=r"(fragments[2]), "=r"(fragments[3])
                 : "r"(shared));
}

NUMKONG_DEVICE void nk_mma_bf16_ampere_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t b_first,
                                        nk_u32_t b_second) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " //
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(accumulator[0].f), "+f"(accumulator[1].f), "+f"(accumulator[2].f), "+f"(accumulator[3].f)
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b_first), "r"(b_second));
}

NUMKONG_DEVICE void nk_mma_f16_ampere_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t b_first,
                                       nk_u32_t b_second) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " //
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(accumulator[0].f), "+f"(accumulator[1].f), "+f"(accumulator[2].f), "+f"(accumulator[3].f)
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b_first), "r"(b_second));
}

NUMKONG_DEVICE void nk_mma_i8_ampere_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t b_first,
                                      nk_u32_t b_second) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " //
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+r"(accumulator[0].u), "+r"(accumulator[1].u), "+r"(accumulator[2].u), "+r"(accumulator[3].u)
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b_first), "r"(b_second));
}

NUMKONG_DEVICE void nk_mma_u8_ampere_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t b_first,
                                      nk_u32_t b_second) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 " //
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+r"(accumulator[0].u), "+r"(accumulator[1].u), "+r"(accumulator[2].u), "+r"(accumulator[3].u)
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b_first), "r"(b_second));
}

NUMKONG_DEVICE void nk_mma_u8i8_ampere_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t b_first,
                                        nk_u32_t b_second) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 " //
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+r"(accumulator[0].u), "+r"(accumulator[1].u), "+r"(accumulator[2].u), "+r"(accumulator[3].u)
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b_first), "r"(b_second));
}

/** The first byte at or past @p pointer whose shared-memory address is a multiple of @p alignment,
 *  a power of two, which dynamic shared memory promises only up to 16. */
NUMKONG_DEVICE unsigned char *nk_shared_aligned_ampere_(unsigned char *pointer, unsigned alignment) {
    return pointer + ((alignment - (nk_shared_address_ampere_(pointer) & (alignment - 1))) & (alignment - 1));
}

#pragma endregion Instructions

/*  Four codes become two pairs of F16 patterns scaled by a power of two: each first lands in the
 *  high byte of a half, `code << 8`, and then keeps only its fields. */
#pragma region Conversions

NUMKONG_DEVICE void nk_e5m2x4_to_f16x4_ampere_(nk_u32_t codes, nk_u32_t *low, nk_u32_t *high) {
    *low = __byte_perm(codes, 0, 0x1404), *high = __byte_perm(codes, 0, 0x3424);
}

NUMKONG_DEVICE void nk_e4m3x4_to_f16x4_ampere_(nk_u32_t codes, nk_u32_t *low, nk_u32_t *high) {
    nk_u32_t const low_halves = __byte_perm(codes, 0, 0x1404), high_halves = __byte_perm(codes, 0, 0x3424);
    // The NaN magnitude 0x7F shifts to a finite 0x3F80, and setting 0x4000 fills its exponent.
    nk_u32_t const nan_bytes = ((codes & 0x7F7F7F7Fu) + 0x01010101u) & 0x80808080u;
    *low = ((low_halves >> 1) & 0x3F803F80u) | (low_halves & 0x80008000u) | (__byte_perm(nan_bytes, 0, 0x1404) >> 1);
    *high = ((high_halves >> 1) & 0x3F803F80u) | (high_halves & 0x80008000u) | (__byte_perm(nan_bytes, 0, 0x3424) >> 1);
}

NUMKONG_DEVICE void nk_e3m2x4_to_f16x4_ampere_(nk_u32_t codes, nk_u32_t *low, nk_u32_t *high) {
    nk_u32_t const low_halves = __byte_perm(codes, 0, 0x1404), high_halves = __byte_perm(codes, 0, 0x3424);
    *low = (low_halves & 0x1F001F00u) | ((low_halves << 2) & 0x80008000u);
    *high = (high_halves & 0x1F001F00u) | ((high_halves << 2) & 0x80008000u);
}

/** Negates the bytes of @p magnitudes under @p sign_mask, each magnitude below 128 so no borrow
 *  crosses a byte. */
NUMKONG_DEVICE nk_u32_t nk_i8x4_apply_signs_ampere_(nk_u32_t magnitudes, nk_u32_t sign_mask) {
    nk_u32_t const negated = (0x80808080u - magnitudes) ^ 0x80808080u;
    return (magnitudes & ~sign_mask) | (negated & sign_mask);
}

/** Four E2M3 codes times 8 as i8. */
NUMKONG_DEVICE nk_u32_t nk_e2m3x4_to_i8x4_ampere_(nk_u32_t codes) {
    return nk_i8x4_apply_signs_ampere_(nk_e2m3x4_to_u8x4_magnitudes_simt_(codes), ((codes >> 5) & 0x01010101u) * 0xFFu);
}

/** Four E2M1 nibbles of @p codes times 2 as i8 through one table lookup; bit 3 of each nibble
 *  is its sign. */
NUMKONG_DEVICE nk_u32_t nk_e2m1x4_to_i8x4_ampere_(nk_u32_t codes) {
    nk_u32_t const magnitudes = __byte_perm(0x03020100u, 0x0C080604u, codes & 0x7777u);
    nk_u32_t replicated;
    // With every table byte negative, a nibble whose bit 3 is set replicates that sign as 0xFF, others read 0x80.
    asm("prmt.b32 %0, %1, %1, %2;" : "=r"(replicated) : "r"(0x80808080u), "r"(codes & 0xFFFFu));
    nk_u32_t const signs = replicated ^ 0x80808080u;
    return nk_i8x4_apply_signs_ampere_(magnitudes, signs | (signs << 1));
}

/** Eight E2M1 nibbles become two registers of four scaled i8 each. */
NUMKONG_DEVICE void nk_e2m1x8_to_i8x8_ampere_(nk_u32_t codes, nk_u32_t *low, nk_u32_t *high) {
    *low = nk_e2m1x4_to_i8x4_ampere_(codes), *high = nk_e2m1x4_to_i8x4_ampere_(codes >> 16);
}

/** Packs one E2M3 code as its value times 8 in i8, the form its multiply consumes. */
NUMKONG_DEVICE unsigned char nk_load_e2m3_to_i8_ampere_(unsigned char code) {
    return (unsigned char)nk_e2m3x4_to_i8x4_ampere_(code);
}

#pragma endregion Conversions

/*  Each dtype adds the squares of this thread's staged row, one 16-byte chunk at a time and in any
 *  order, since a sum of squares has none. Float8 and Float6 square their F16 widenings, and E2M3
 *  and E2M1 their scaled i8 ones, so the tile scales them back. Floats sum each slab apart before
 *  folding it into the running sum, and integers go straight into theirs. */
#pragma region Norms

/** The words of chunk @p chunk of this thread's staged row; the staging swizzle as the visiting
 *  order puts a quarter-warp's 16-byte reads on distinct banks. */
NUMKONG_DEVICE void nk_cross_staged_words_ampere_(unsigned char const *stage, unsigned chunk, nk_u32_t words[4]) {
    uint4 const bytes = *(
        uint4 const *)(stage + nk_swizzled_offset_simt_(threadIdx.x, chunk << 4, nk_cross_slab_bytes_ampere_k));
    words[0] = bytes.x, words[1] = bytes.y, words[2] = bytes.z, words[3] = bytes.w;
}

NUMKONG_DEVICE void nk_cross_stage_norm_bf16_ampere_(unsigned char const *stage, nk_f32_t *real_sum) {
    nk_f32_t slab_sum = 0;
#pragma unroll
    for (unsigned chunk = 0; chunk < 4; ++chunk) {
        nk_u32_t words[4];
        nk_cross_staged_words_ampere_(stage, chunk, words);
#pragma unroll
        for (unsigned word = 0; word < 4; ++word) {
            nk_f32_t const low = __uint_as_float(words[word] << 16), high = __uint_as_float(words[word] & 0xFFFF0000u);
            slab_sum = __fmaf_rn(high, high, __fmaf_rn(low, low, slab_sum));
        }
    }
    if (slab_sum != 0) *real_sum += slab_sum;
}

NUMKONG_DEVICE void nk_cross_stage_norm_f16_ampere_(unsigned char const *stage, nk_f32_t *real_sum) {
    nk_f32_t slab_sum = 0;
#pragma unroll
    for (unsigned chunk = 0; chunk < 4; ++chunk) {
        nk_u32_t words[4];
        nk_cross_staged_words_ampere_(stage, chunk, words);
#pragma unroll
        for (unsigned word = 0; word < 4; ++word) nk_f16x2_norm_update_simt_(words[word], &slab_sum);
    }
    if (slab_sum != 0) *real_sum += slab_sum;
}

NUMKONG_DEVICE void nk_cross_stage_norm_e5m2_ampere_(unsigned char const *stage, nk_f32_t *real_sum) {
    nk_f32_t slab_sum = 0;
#pragma unroll
    for (unsigned chunk = 0; chunk < 4; ++chunk) {
        nk_u32_t words[4];
        nk_cross_staged_words_ampere_(stage, chunk, words);
#pragma unroll
        for (unsigned word = 0; word < 4; ++word) {
            nk_u32_t low, high;
            nk_e5m2x4_to_f16x4_ampere_(words[word], &low, &high);
            nk_f16x2_norm_update_simt_(low, &slab_sum);
            nk_f16x2_norm_update_simt_(high, &slab_sum);
        }
    }
    if (slab_sum != 0) *real_sum += slab_sum;
}

NUMKONG_DEVICE void nk_cross_stage_norm_e4m3_ampere_(unsigned char const *stage, nk_f32_t *real_sum) {
    nk_f32_t slab_sum = 0;
#pragma unroll
    for (unsigned chunk = 0; chunk < 4; ++chunk) {
        nk_u32_t words[4];
        nk_cross_staged_words_ampere_(stage, chunk, words);
#pragma unroll
        for (unsigned word = 0; word < 4; ++word) {
            nk_u32_t low, high;
            nk_e4m3x4_to_f16x4_ampere_(words[word], &low, &high);
            nk_f16x2_norm_update_simt_(low, &slab_sum);
            nk_f16x2_norm_update_simt_(high, &slab_sum);
        }
    }
    if (slab_sum != 0) *real_sum += slab_sum;
}

NUMKONG_DEVICE void nk_cross_stage_norm_e3m2_ampere_(unsigned char const *stage, nk_f32_t *real_sum) {
    nk_f32_t slab_sum = 0;
#pragma unroll
    for (unsigned chunk = 0; chunk < 4; ++chunk) {
        nk_u32_t words[4];
        nk_cross_staged_words_ampere_(stage, chunk, words);
#pragma unroll
        for (unsigned word = 0; word < 4; ++word) {
            nk_u32_t low, high;
            nk_e3m2x4_to_f16x4_ampere_(words[word], &low, &high);
            nk_f16x2_norm_update_simt_(low, &slab_sum);
            nk_f16x2_norm_update_simt_(high, &slab_sum);
        }
    }
    if (slab_sum != 0) *real_sum += slab_sum;
}

NUMKONG_DEVICE void nk_cross_stage_norm_e2m3_ampere_(unsigned char const *stage, nk_u32_t *integer_sum) {
#pragma unroll
    for (unsigned chunk = 0; chunk < 4; ++chunk) {
        nk_u32_t words[4];
        nk_cross_staged_words_ampere_(stage, chunk, words);
#pragma unroll
        for (unsigned word = 0; word < 4; ++word) {
            nk_u32_t const magnitudes = nk_e2m3x4_to_u8x4_magnitudes_simt_(words[word]);
            *integer_sum = __dp4a(magnitudes, magnitudes, *integer_sum);
        }
    }
}

NUMKONG_DEVICE void nk_cross_stage_norm_e2m1_ampere_(unsigned char const *stage, nk_u32_t *integer_sum) {
#pragma unroll
    for (unsigned chunk = 0; chunk < 4; ++chunk) {
        nk_u32_t words[4];
        nk_cross_staged_words_ampere_(stage, chunk, words);
#pragma unroll
        for (unsigned word = 0; word < 4; ++word) {
            // Squares of twice each magnitude, {0, 1, 4, 9, 16, 36, 64, 144}, 4 nibbles per lookup.
            nk_u32_t const low = __byte_perm(0x09040100u, 0x90402410u, words[word] & 0x7777u);
            nk_u32_t const high = __byte_perm(0x09040100u, 0x90402410u, (words[word] >> 16) & 0x7777u);
            *integer_sum = __dp4a(low, 0x01010101u, *integer_sum);
            *integer_sum = __dp4a(high, 0x01010101u, *integer_sum);
        }
    }
}

NUMKONG_DEVICE void nk_cross_stage_norm_i8_ampere_(unsigned char const *stage, nk_u32_t *integer_sum) {
#pragma unroll
    for (unsigned chunk = 0; chunk < 4; ++chunk) {
        nk_u32_t words[4];
        nk_cross_staged_words_ampere_(stage, chunk, words);
#pragma unroll
        for (unsigned word = 0; word < 4; ++word) {
            *integer_sum = (nk_u32_t)__dp4a((int)words[word], (int)words[word], (int)*integer_sum);
        }
    }
}

NUMKONG_DEVICE void nk_cross_stage_norm_i4_ampere_(unsigned char const *stage, nk_u32_t *integer_sum) {
#pragma unroll
    for (unsigned chunk = 0; chunk < 4; ++chunk) {
        nk_u32_t words[4];
        nk_cross_staged_words_ampere_(stage, chunk, words);
#pragma unroll
        for (unsigned word = 0; word < 4; ++word) {
            nk_u32_t low, high;
            nk_i4x8_to_i8x8_simt_(words[word], &low, &high);
            *integer_sum = (nk_u32_t)__dp4a((int)low, (int)low, (int)*integer_sum);
            *integer_sum = (nk_u32_t)__dp4a((int)high, (int)high, (int)*integer_sum);
        }
    }
}

NUMKONG_DEVICE void nk_cross_stage_norm_u8_ampere_(unsigned char const *stage, nk_u32_t *integer_sum) {
#pragma unroll
    for (unsigned chunk = 0; chunk < 4; ++chunk) {
        nk_u32_t words[4];
        nk_cross_staged_words_ampere_(stage, chunk, words);
#pragma unroll
        for (unsigned word = 0; word < 4; ++word) { *integer_sum = __dp4a(words[word], words[word], *integer_sum); }
    }
}

NUMKONG_DEVICE void nk_cross_stage_norm_u4_ampere_(unsigned char const *stage, nk_u32_t *integer_sum) {
#pragma unroll
    for (unsigned chunk = 0; chunk < 4; ++chunk) {
        nk_u32_t words[4];
        nk_cross_staged_words_ampere_(stage, chunk, words);
#pragma unroll
        for (unsigned word = 0; word < 4; ++word) {
            nk_u32_t low, high;
            nk_u4x8_to_u8x8_simt_(words[word], &low, &high);
            *integer_sum = __dp4a(low, low, *integer_sum);
            *integer_sum = __dp4a(high, high, *integer_sum);
        }
    }
}

#pragma endregion Norms

#pragma region Tile

/** Issues one 64-byte depth slab of A and B into a pipeline stage, 16 bytes per copy, 8
 *  copies per thread. */
NUMKONG_DEVICE void nk_cross_stage_ampere_(unsigned char *a_stage, unsigned char *b_stage,
                                           nk_cross_tile_arguments_t const *arguments, nk_size_t first_row,
                                           nk_size_t first_column, nk_size_t slab_offset) {
#pragma unroll
    for (unsigned chunk = threadIdx.x; chunk < nk_cross_stage_bytes_ampere_k / 16; chunk += nk_cross_threads_ampere_k) {
        unsigned const tile_row = chunk >> 2, column = chunk & 3;
        unsigned const swizzled = nk_swizzled_offset_simt_(tile_row, column << 4, nk_cross_slab_bytes_ampere_k);
        nk_size_t const byte = slab_offset + (column << 4);
        nk_u32_t const valid = byte < arguments->depth_bytes
                                   ? (nk_u32_t)(arguments->depth_bytes - byte < 16 ? arguments->depth_bytes - byte : 16)
                                   : 0;

        nk_size_t const row = first_row + tile_row;
        unsigned char const *a_source = row < arguments->rows_end ? arguments->a + row * arguments->a_stride + byte
                                                                  : arguments->a;
        nk_copy_b128_async_ampere_(nk_shared_address_ampere_(a_stage + swizzled), a_source,
                                   row < arguments->rows_end ? valid : 0);

        nk_size_t const b_row = first_column + tile_row;
        unsigned char const *b_source = b_row < arguments->column_count
                                            ? arguments->b + b_row * arguments->b_stride + byte
                                            : arguments->b;
        nk_copy_b128_async_ampere_(nk_shared_address_ampere_(b_stage + swizzled), b_source,
                                   b_row < arguments->column_count ? valid : 0);
    }
}

/** Where tile @p tile of the grid starts, or zero when @p band hides all of it. */
NUMKONG_DEVICE int nk_cross_tile_origin_ampere_(nk_diagonal_band_t band, nk_cross_tile_arguments_t const *arguments,
                                                nk_size_t tile, nk_size_t *first_row, nk_size_t *first_column) {
    *first_row = arguments->rows_begin + tile / arguments->column_tiles * nk_cross_tile_ampere_k;
    *first_column = tile % arguments->column_tiles * nk_cross_tile_ampere_k;
    return nk_diagonal_band_tile_coverage_simt_(band, (nk_i64_t)*first_row, nk_cross_tile_ampere_k, *first_column,
                                                nk_cross_tile_ampere_k) != nk_diagonal_band_outside_k;
}

NUMKONG_DEVICE void nk_cross_accumulators_clear_ampere_(nk_fui32_t accumulators[4][8][4]) {
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
#pragma unroll
            for (unsigned element = 0; element < 4; ++element) accumulators[row_tile][column_tile][element].u = 0;
}

/** Fills all but one stage of the ring ahead of the first slab. */
NUMKONG_DEVICE void nk_cross_prologue_ampere_(nk_cross_shared_ampere_t *shared,
                                              nk_cross_tile_arguments_t const *arguments, nk_size_t first_row,
                                              nk_size_t first_column) {
#pragma unroll
    for (unsigned stage = 0; stage + 1 < nk_cross_stages_ampere_k; ++stage) {
        if (stage < arguments->depth_slabs)
            nk_cross_stage_ampere_(shared->stages[stage][0], shared->stages[stage][1], arguments, first_row,
                                   first_column, stage * nk_cross_slab_bytes_ampere_k);
        nk_commit_async_ampere_();
    }
}

/** Refills the stage read one iteration ago, which the barrier of this slab retired, with the slab
 *  @p slab + stages − 1 ahead. */
NUMKONG_DEVICE void nk_cross_prefetch_ampere_(nk_cross_shared_ampere_t *shared,
                                              nk_cross_tile_arguments_t const *arguments, nk_size_t first_row,
                                              nk_size_t first_column, nk_size_t slab, unsigned write_stage) {
    nk_size_t const prefetch = slab + nk_cross_stages_ampere_k - 1;
    if (prefetch < arguments->depth_slabs)
        nk_cross_stage_ampere_(shared->stages[write_stage][0], shared->stages[write_stage][1], arguments, first_row,
                               first_column, prefetch * nk_cross_slab_bytes_ampere_k);
    nk_commit_async_ampere_();
}

/** This warp's fragments of 32-byte sub-slab @p sub_slab: A as 4 row tiles of 16, B as 8 column
 *  tiles of 8. */
NUMKONG_DEVICE void nk_cross_load_fragments_ampere_(unsigned char const *a_stage, unsigned char const *b_stage,
                                                    unsigned sub_slab, nk_u32_t a_fragments[4][4],
                                                    nk_u32_t b_fragments[8][2]) {
    unsigned const lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    unsigned const warp_row = (warp >> 1) * 64, warp_column = (warp & 1) * 64;
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile) {
        unsigned const row = warp_row + row_tile * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
        unsigned const chunk = sub_slab * 2 + (lane >> 4);
        nk_load_matrices_x4_ampere_(
            nk_shared_address_ampere_(a_stage +
                                      nk_swizzled_offset_simt_(row, chunk << 4, nk_cross_slab_bytes_ampere_k)),
            a_fragments[row_tile]);
    }
#pragma unroll
    for (unsigned column_pair = 0; column_pair < 4; ++column_pair) {
        unsigned const row = warp_column + column_pair * 16 + (lane & 7) + (lane >> 4) * 8;
        unsigned const chunk = sub_slab * 2 + ((lane >> 3) & 1);
        nk_u32_t pair[4];
        nk_load_matrices_x4_ampere_(
            nk_shared_address_ampere_(b_stage +
                                      nk_swizzled_offset_simt_(row, chunk << 4, nk_cross_slab_bytes_ampere_k)),
            pair);
        b_fragments[column_pair * 2][0] = pair[0], b_fragments[column_pair * 2][1] = pair[1];
        b_fragments[column_pair * 2 + 1][0] = pair[2], b_fragments[column_pair * 2 + 1][1] = pair[3];
    }
}

/**
 *  @brief Writes the output tile once every product is done.
 *  @param[in] epilogue What the accumulators hold and how they reach the output.
 *  @param[in] output_scale Undoes the power of two a widening introduced, or 1.
 *  @param[in] norm How the squared norms are stored and the metric is computed; unused for dots.
 *  @param[in] metric The dot product itself, or a distance from it and the two squared norms.
 *  @param[in] row_norm This thread's row norm, finalized; unused for dots.
 *  @param[in] column_norm This thread's column norm, finalized, read only for a Gram matrix; the
 *      packed column norms come from @c b_norms.
 *
 *  Both norms go to the idle ring for the epilogue, once every warp's fragment reads retire.
 */
NUMKONG_DEVICE void nk_cross_tile_store_ampere_(nk_cross_shared_ampere_t *shared, nk_fui32_t accumulators[4][8][4],
                                                nk_cross_epilogue_t epilogue, nk_f32_t output_scale,
                                                nk_cross_norm_t norm, nk_cross_metric_t metric, nk_diagonal_band_t band,
                                                nk_size_t first_row, nk_size_t first_column, nk_fui32_t row_norm,
                                                nk_fui32_t column_norm, nk_cross_tile_arguments_t const *arguments) {
    unsigned const lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    unsigned const warp_row = (warp >> 1) * 64, warp_column = (warp & 1) * 64;
    nk_fui32_t *norms = shared->norms;
    if (metric != nk_cross_metric_dot_k) {
        __syncthreads();
        norms[threadIdx.x] = row_norm;
        nk_size_t const column = first_column + threadIdx.x;
        nk_u32_t const *column_norms = (nk_u32_t const *)arguments->b_norms;
        if (nk_cross_symmetric_simt_(band)) norms[nk_cross_tile_ampere_k + threadIdx.x] = column_norm;
        else
            norms[nk_cross_tile_ampere_k + threadIdx.x].u = column < arguments->column_count ? column_norms[column] : 0;
        __syncthreads();
    }

    unsigned const group = lane >> 2, quad = lane & 3;
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
        for (unsigned half = 0; half < 2; ++half) {
            unsigned const tile_row = warp_row + row_tile * 16 + half * 8 + group;
            nk_size_t const row = first_row + tile_row;
            if (row >= arguments->rows_end) continue;
            nk_size_t column_begin, column_end;
            nk_diagonal_band_row_range_simt_(band, (nk_i64_t)row, arguments->column_count, &column_begin, &column_end);
            unsigned char *output = (unsigned char *)arguments->c + row * arguments->c_stride;
            nk_f32_t *output_f32 = (nk_f32_t *)output;
            nk_u32_t *output_u32 = (nk_u32_t *)output;
            nk_fui32_t row_norm_shared;
            row_norm_shared.u = metric == nk_cross_metric_dot_k ? 0 : norms[tile_row].u;
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
#pragma unroll
                for (unsigned pair = 0; pair < 2; ++pair) {
                    unsigned const tile_column = warp_column + column_tile * 8 + quad * 2 + pair;
                    nk_size_t const column = first_column + tile_column;
                    if (column < column_begin || column >= column_end) continue;
                    nk_fui32_t const sum = accumulators[row_tile][column_tile][half * 2 + pair];
                    if (metric == nk_cross_metric_dot_k) {
                        if (epilogue == nk_cross_epilogue_f32_k) output_f32[column] = sum.f * output_scale;
                        else if (epilogue == nk_cross_epilogue_i32_k) output_u32[column] = sum.u;
                        else output_f32[column] = (nk_f32_t)sum.i * output_scale;
                        continue;
                    }
                    if (nk_cross_symmetric_simt_(band) && column == row) {
                        output_f32[column] = 0.0f;
                        continue;
                    }
                    nk_fui32_t const column_norm_shared = norms[nk_cross_tile_ampere_k + tile_column];
                    nk_f32_t const dot = nk_cross_dot_to_f32_simt_(sum, epilogue, output_scale);
                    if (norm != nk_cross_norm_f32_k)
                        output_f32[column] = nk_cross_integer_metric_simt_(metric, norm, sum.u, row_norm_shared.u,
                                                                           column_norm_shared.u);
                    else if (metric == nk_cross_metric_angular_k)
                        output_f32[column] = nk_f32_angular_simt_(dot, row_norm_shared.f, column_norm_shared.f);
                    else output_f32[column] = nk_f32_euclidean_simt_(dot, row_norm_shared.f, column_norm_shared.f);
                }
        }
}

#pragma endregion Tile

#pragma region Multiplies

NUMKONG_DEVICE void nk_dots_bf16_multiply_ampere_(nk_fui32_t accumulators[4][8][4], nk_u32_t const a[4][4],
                                                  nk_u32_t const b[8][2]) {
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
            nk_mma_bf16_ampere_(accumulators[row_tile][column_tile], a[row_tile], b[column_tile][0], b[column_tile][1]);
}

NUMKONG_DEVICE void nk_dots_f16_multiply_ampere_(nk_fui32_t accumulators[4][8][4], nk_u32_t const a[4][4],
                                                 nk_u32_t const b[8][2]) {
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
            nk_mma_f16_ampere_(accumulators[row_tile][column_tile], a[row_tile], b[column_tile][0], b[column_tile][1]);
}

/*  Float8 and E3M2 widen every fragment register into two F16 pairs, and nibbles into two i8
 *  quads, so each 32-byte sub-slab takes two steps of the widened type's MMA. */
NUMKONG_DEVICE void nk_dots_e5m2_multiply_ampere_(nk_fui32_t accumulators[4][8][4], nk_u32_t const a[4][4],
                                                  nk_u32_t const b[8][2]) {
    nk_u32_t b_low[8][2], b_high[8][2];
#pragma unroll
    for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
#pragma unroll
        for (unsigned depth_half = 0; depth_half < 2; ++depth_half)
            nk_e5m2x4_to_f16x4_ampere_(b[column_tile][depth_half], &b_low[column_tile][depth_half],
                                       &b_high[column_tile][depth_half]);
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile) {
        nk_u32_t a_low[4], a_high[4];
#pragma unroll
        for (unsigned fragment = 0; fragment < 4; ++fragment)
            nk_e5m2x4_to_f16x4_ampere_(a[row_tile][fragment], &a_low[fragment], &a_high[fragment]);
        nk_u32_t const first[4] = {a_low[0], a_low[1], a_high[0], a_high[1]};
        nk_u32_t const second[4] = {a_low[2], a_low[3], a_high[2], a_high[3]};
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 8; ++column_tile) {
            nk_mma_f16_ampere_(accumulators[row_tile][column_tile], first, b_low[column_tile][0],
                               b_high[column_tile][0]);
            nk_mma_f16_ampere_(accumulators[row_tile][column_tile], second, b_low[column_tile][1],
                               b_high[column_tile][1]);
        }
    }
}

NUMKONG_DEVICE void nk_dots_e4m3_multiply_ampere_(nk_fui32_t accumulators[4][8][4], nk_u32_t const a[4][4],
                                                  nk_u32_t const b[8][2]) {
    nk_u32_t b_low[8][2], b_high[8][2];
#pragma unroll
    for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
#pragma unroll
        for (unsigned depth_half = 0; depth_half < 2; ++depth_half)
            nk_e4m3x4_to_f16x4_ampere_(b[column_tile][depth_half], &b_low[column_tile][depth_half],
                                       &b_high[column_tile][depth_half]);
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile) {
        nk_u32_t a_low[4], a_high[4];
#pragma unroll
        for (unsigned fragment = 0; fragment < 4; ++fragment)
            nk_e4m3x4_to_f16x4_ampere_(a[row_tile][fragment], &a_low[fragment], &a_high[fragment]);
        nk_u32_t const first[4] = {a_low[0], a_low[1], a_high[0], a_high[1]};
        nk_u32_t const second[4] = {a_low[2], a_low[3], a_high[2], a_high[3]};
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 8; ++column_tile) {
            nk_mma_f16_ampere_(accumulators[row_tile][column_tile], first, b_low[column_tile][0],
                               b_high[column_tile][0]);
            nk_mma_f16_ampere_(accumulators[row_tile][column_tile], second, b_low[column_tile][1],
                               b_high[column_tile][1]);
        }
    }
}

NUMKONG_DEVICE void nk_dots_e3m2_multiply_ampere_(nk_fui32_t accumulators[4][8][4], nk_u32_t const a[4][4],
                                                  nk_u32_t const b[8][2]) {
    nk_u32_t b_low[8][2], b_high[8][2];
#pragma unroll
    for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
#pragma unroll
        for (unsigned depth_half = 0; depth_half < 2; ++depth_half)
            nk_e3m2x4_to_f16x4_ampere_(b[column_tile][depth_half], &b_low[column_tile][depth_half],
                                       &b_high[column_tile][depth_half]);
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile) {
        nk_u32_t a_low[4], a_high[4];
#pragma unroll
        for (unsigned fragment = 0; fragment < 4; ++fragment)
            nk_e3m2x4_to_f16x4_ampere_(a[row_tile][fragment], &a_low[fragment], &a_high[fragment]);
        nk_u32_t const first[4] = {a_low[0], a_low[1], a_high[0], a_high[1]};
        nk_u32_t const second[4] = {a_low[2], a_low[3], a_high[2], a_high[3]};
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 8; ++column_tile) {
            nk_mma_f16_ampere_(accumulators[row_tile][column_tile], first, b_low[column_tile][0],
                               b_high[column_tile][0]);
            nk_mma_f16_ampere_(accumulators[row_tile][column_tile], second, b_low[column_tile][1],
                               b_high[column_tile][1]);
        }
    }
}

/** E2M3 against the scaled i8 its pack produced: only A converts, one integer step per
 *  16 × 8 output. */
NUMKONG_DEVICE void nk_dots_e2m3_packed_multiply_ampere_(nk_fui32_t accumulators[4][8][4], nk_u32_t const a[4][4],
                                                         nk_u32_t const b[8][2]) {
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile) {
        nk_u32_t const a_scaled[4] = {
            nk_e2m3x4_to_i8x4_ampere_(a[row_tile][0]), nk_e2m3x4_to_i8x4_ampere_(a[row_tile][1]),
            nk_e2m3x4_to_i8x4_ampere_(a[row_tile][2]), nk_e2m3x4_to_i8x4_ampere_(a[row_tile][3])};
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
            nk_mma_i8_ampere_(accumulators[row_tile][column_tile], a_scaled, b[column_tile][0], b[column_tile][1]);
    }
}

/** E2M3 on both sides, for @c symmetric, where B is the raw codes again. */
NUMKONG_DEVICE void nk_dots_e2m3_multiply_ampere_(nk_fui32_t accumulators[4][8][4], nk_u32_t const a[4][4],
                                                  nk_u32_t const b[8][2]) {
    nk_u32_t b_scaled[8][2];
#pragma unroll
    for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
#pragma unroll
        for (unsigned depth_half = 0; depth_half < 2; ++depth_half)
            b_scaled[column_tile][depth_half] = nk_e2m3x4_to_i8x4_ampere_(b[column_tile][depth_half]);
    nk_dots_e2m3_packed_multiply_ampere_(accumulators, a, b_scaled);
}

NUMKONG_DEVICE void nk_dots_e2m1_multiply_ampere_(nk_fui32_t accumulators[4][8][4], nk_u32_t const a[4][4],
                                                  nk_u32_t const b[8][2]) {
    nk_u32_t b_low[8][2], b_high[8][2];
#pragma unroll
    for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
#pragma unroll
        for (unsigned depth_half = 0; depth_half < 2; ++depth_half)
            nk_e2m1x8_to_i8x8_ampere_(b[column_tile][depth_half], &b_low[column_tile][depth_half],
                                      &b_high[column_tile][depth_half]);
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile) {
        nk_u32_t a_low[4], a_high[4];
#pragma unroll
        for (unsigned fragment = 0; fragment < 4; ++fragment)
            nk_e2m1x8_to_i8x8_ampere_(a[row_tile][fragment], &a_low[fragment], &a_high[fragment]);
        nk_u32_t const first[4] = {a_low[0], a_low[1], a_high[0], a_high[1]};
        nk_u32_t const second[4] = {a_low[2], a_low[3], a_high[2], a_high[3]};
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 8; ++column_tile) {
            nk_mma_i8_ampere_(accumulators[row_tile][column_tile], first, b_low[column_tile][0],
                              b_high[column_tile][0]);
            nk_mma_i8_ampere_(accumulators[row_tile][column_tile], second, b_low[column_tile][1],
                              b_high[column_tile][1]);
        }
    }
}

NUMKONG_DEVICE void nk_dots_i8_multiply_ampere_(nk_fui32_t accumulators[4][8][4], nk_u32_t const a[4][4],
                                                nk_u32_t const b[8][2]) {
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
            nk_mma_i8_ampere_(accumulators[row_tile][column_tile], a[row_tile], b[column_tile][0], b[column_tile][1]);
}

NUMKONG_DEVICE void nk_dots_u8_multiply_ampere_(nk_fui32_t accumulators[4][8][4], nk_u32_t const a[4][4],
                                                nk_u32_t const b[8][2]) {
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
            nk_mma_u8_ampere_(accumulators[row_tile][column_tile], a[row_tile], b[column_tile][0], b[column_tile][1]);
}

/*  Nibbles widen to 8-bit steps, since from 9.0 on @c ptxas lowers 4-bit MMA to a software routine
 *  and this tier runs on every GPU since 8.0. */
NUMKONG_DEVICE void nk_dots_i4_multiply_ampere_(nk_fui32_t accumulators[4][8][4], nk_u32_t const a[4][4],
                                                nk_u32_t const b[8][2]) {
    nk_u32_t b_low[8][2], b_high[8][2];
#pragma unroll
    for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
#pragma unroll
        for (unsigned depth_half = 0; depth_half < 2; ++depth_half)
            nk_i4x8_to_i8x8_simt_(b[column_tile][depth_half], &b_low[column_tile][depth_half],
                                  &b_high[column_tile][depth_half]);
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile) {
        nk_u32_t a_low[4], a_high[4];
#pragma unroll
        for (unsigned fragment = 0; fragment < 4; ++fragment)
            nk_i4x8_to_i8x8_simt_(a[row_tile][fragment], &a_low[fragment], &a_high[fragment]);
        nk_u32_t const first[4] = {a_low[0], a_low[1], a_high[0], a_high[1]};
        nk_u32_t const second[4] = {a_low[2], a_low[3], a_high[2], a_high[3]};
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 8; ++column_tile) {
            nk_mma_i8_ampere_(accumulators[row_tile][column_tile], first, b_low[column_tile][0],
                              b_high[column_tile][0]);
            nk_mma_i8_ampere_(accumulators[row_tile][column_tile], second, b_low[column_tile][1],
                              b_high[column_tile][1]);
        }
    }
}

NUMKONG_DEVICE void nk_dots_u4_multiply_ampere_(nk_fui32_t accumulators[4][8][4], nk_u32_t const a[4][4],
                                                nk_u32_t const b[8][2]) {
    nk_u32_t b_low[8][2], b_high[8][2];
#pragma unroll
    for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
#pragma unroll
        for (unsigned depth_half = 0; depth_half < 2; ++depth_half)
            nk_u4x8_to_u8x8_simt_(b[column_tile][depth_half], &b_low[column_tile][depth_half],
                                  &b_high[column_tile][depth_half]);
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile) {
        nk_u32_t a_low[4], a_high[4];
#pragma unroll
        for (unsigned fragment = 0; fragment < 4; ++fragment)
            nk_u4x8_to_u8x8_simt_(a[row_tile][fragment], &a_low[fragment], &a_high[fragment]);
        nk_u32_t const first[4] = {a_low[0], a_low[1], a_high[0], a_high[1]};
        nk_u32_t const second[4] = {a_low[2], a_low[3], a_high[2], a_high[3]};
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 8; ++column_tile) {
            nk_mma_u8_ampere_(accumulators[row_tile][column_tile], first, b_low[column_tile][0],
                              b_high[column_tile][0]);
            nk_mma_u8_ampere_(accumulators[row_tile][column_tile], second, b_low[column_tile][1],
                              b_high[column_tile][1]);
        }
    }
}

#pragma endregion Multiplies

#pragma region Tiles

NUMKONG_DEVICE void nk_cross_tile_bf16_ampere_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                               nk_cross_tile_arguments_t const *arguments) {
    __shared__ __align__(128) nk_cross_shared_ampere_t shared;
    nk_fui32_t accumulators[4][8][4];

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        // The previous tile's last ring reads must retire before this tile's prologue refills it.
        __syncthreads();
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_ampere_(band, arguments, tile, &first_row, &first_column)) continue;
        nk_cross_accumulators_clear_ampere_(accumulators);
        nk_f32_t row_real_norm = 0, column_real_norm = 0;
        nk_cross_prologue_ampere_(&shared, arguments, first_row, first_column);
        unsigned read_stage = 0, write_stage = nk_cross_stages_ampere_k - 1;
        for (nk_size_t slab = 0; slab < arguments->depth_slabs; ++slab) {
            nk_wait_async_ampere_(nk_cross_stages_ampere_k - 2);
            __syncthreads();
            unsigned char *a_stage = shared.stages[read_stage][0], *b_stage = shared.stages[read_stage][1];
#pragma unroll
            for (unsigned sub_slab = 0; sub_slab < 2; ++sub_slab) {
                nk_u32_t a_fragments[4][4], b_fragments[8][2];
                nk_cross_load_fragments_ampere_(a_stage, b_stage, sub_slab, a_fragments, b_fragments);
                nk_dots_bf16_multiply_ampere_(accumulators, a_fragments, b_fragments);
            }
            // Issued after the products, or ptxas sinks the commit below the fragment loads.
            nk_cross_prefetch_ampere_(&shared, arguments, first_row, first_column, slab, write_stage);
            // Squares come after the products, once the fragments are dead, and read a stage no
            // copy refills until the next iteration's barrier.
            if (metric != nk_cross_metric_dot_k) {
                nk_cross_stage_norm_bf16_ampere_(a_stage, &row_real_norm);
                if (nk_cross_symmetric_simt_(band)) nk_cross_stage_norm_bf16_ampere_(b_stage, &column_real_norm);
            }
            read_stage = read_stage + 1 == nk_cross_stages_ampere_k ? 0 : read_stage + 1;
            write_stage = write_stage + 1 == nk_cross_stages_ampere_k ? 0 : write_stage + 1;
        }
        nk_cross_tile_store_ampere_(
            &shared, accumulators, nk_cross_epilogue_f32_k, 1.0f, nk_cross_norm_f32_k, metric, band, first_row,
            first_column, nk_cross_norm_finalize_simt_(nk_cross_norm_f32_k, 0, row_real_norm, 1.0f),
            nk_cross_norm_finalize_simt_(nk_cross_norm_f32_k, 0, column_real_norm, 1.0f), arguments);
    }
}

NUMKONG_DEVICE void nk_cross_tile_f16_ampere_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                              nk_cross_tile_arguments_t const *arguments) {
    __shared__ __align__(128) nk_cross_shared_ampere_t shared;
    nk_fui32_t accumulators[4][8][4];

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        // The previous tile's last ring reads must retire before this tile's prologue refills it.
        __syncthreads();
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_ampere_(band, arguments, tile, &first_row, &first_column)) continue;
        nk_cross_accumulators_clear_ampere_(accumulators);
        nk_f32_t row_real_norm = 0, column_real_norm = 0;
        nk_cross_prologue_ampere_(&shared, arguments, first_row, first_column);
        unsigned read_stage = 0, write_stage = nk_cross_stages_ampere_k - 1;
        for (nk_size_t slab = 0; slab < arguments->depth_slabs; ++slab) {
            nk_wait_async_ampere_(nk_cross_stages_ampere_k - 2);
            __syncthreads();
            unsigned char *a_stage = shared.stages[read_stage][0], *b_stage = shared.stages[read_stage][1];
#pragma unroll
            for (unsigned sub_slab = 0; sub_slab < 2; ++sub_slab) {
                nk_u32_t a_fragments[4][4], b_fragments[8][2];
                nk_cross_load_fragments_ampere_(a_stage, b_stage, sub_slab, a_fragments, b_fragments);
                nk_dots_f16_multiply_ampere_(accumulators, a_fragments, b_fragments);
            }
            // Issued after the products, or ptxas sinks the commit below the fragment loads.
            nk_cross_prefetch_ampere_(&shared, arguments, first_row, first_column, slab, write_stage);
            // Squares come after the products, once the fragments are dead, and read a stage no
            // copy refills until the next iteration's barrier.
            if (metric != nk_cross_metric_dot_k) {
                nk_cross_stage_norm_f16_ampere_(a_stage, &row_real_norm);
                if (nk_cross_symmetric_simt_(band)) nk_cross_stage_norm_f16_ampere_(b_stage, &column_real_norm);
            }
            read_stage = read_stage + 1 == nk_cross_stages_ampere_k ? 0 : read_stage + 1;
            write_stage = write_stage + 1 == nk_cross_stages_ampere_k ? 0 : write_stage + 1;
        }
        nk_cross_tile_store_ampere_(
            &shared, accumulators, nk_cross_epilogue_f32_k, 1.0f, nk_cross_norm_f32_k, metric, band, first_row,
            first_column, nk_cross_norm_finalize_simt_(nk_cross_norm_f32_k, 0, row_real_norm, 1.0f),
            nk_cross_norm_finalize_simt_(nk_cross_norm_f32_k, 0, column_real_norm, 1.0f), arguments);
    }
}

NUMKONG_DEVICE void nk_cross_tile_e5m2_ampere_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                               nk_cross_tile_arguments_t const *arguments) {
    __shared__ __align__(128) nk_cross_shared_ampere_t shared;
    nk_fui32_t accumulators[4][8][4];

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        // The previous tile's last ring reads must retire before this tile's prologue refills it.
        __syncthreads();
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_ampere_(band, arguments, tile, &first_row, &first_column)) continue;
        nk_cross_accumulators_clear_ampere_(accumulators);
        nk_f32_t row_real_norm = 0, column_real_norm = 0;
        nk_cross_prologue_ampere_(&shared, arguments, first_row, first_column);
        unsigned read_stage = 0, write_stage = nk_cross_stages_ampere_k - 1;
        for (nk_size_t slab = 0; slab < arguments->depth_slabs; ++slab) {
            nk_wait_async_ampere_(nk_cross_stages_ampere_k - 2);
            __syncthreads();
            unsigned char *a_stage = shared.stages[read_stage][0], *b_stage = shared.stages[read_stage][1];
#pragma unroll
            for (unsigned sub_slab = 0; sub_slab < 2; ++sub_slab) {
                nk_u32_t a_fragments[4][4], b_fragments[8][2];
                nk_cross_load_fragments_ampere_(a_stage, b_stage, sub_slab, a_fragments, b_fragments);
                nk_dots_e5m2_multiply_ampere_(accumulators, a_fragments, b_fragments);
            }
            // Issued after the products, or ptxas sinks the commit below the fragment loads.
            nk_cross_prefetch_ampere_(&shared, arguments, first_row, first_column, slab, write_stage);
            // Squares come after the products, once the fragments are dead, and read a stage no
            // copy refills until the next iteration's barrier.
            if (metric != nk_cross_metric_dot_k) {
                nk_cross_stage_norm_e5m2_ampere_(a_stage, &row_real_norm);
                if (nk_cross_symmetric_simt_(band)) nk_cross_stage_norm_e5m2_ampere_(b_stage, &column_real_norm);
            }
            read_stage = read_stage + 1 == nk_cross_stages_ampere_k ? 0 : read_stage + 1;
            write_stage = write_stage + 1 == nk_cross_stages_ampere_k ? 0 : write_stage + 1;
        }
        nk_cross_tile_store_ampere_(
            &shared, accumulators, nk_cross_epilogue_f32_k, 1.0f, nk_cross_norm_f32_k, metric, band, first_row,
            first_column, nk_cross_norm_finalize_simt_(nk_cross_norm_f32_k, 0, row_real_norm, 1.0f),
            nk_cross_norm_finalize_simt_(nk_cross_norm_f32_k, 0, column_real_norm, 1.0f), arguments);
    }
}

NUMKONG_DEVICE void nk_cross_tile_e4m3_ampere_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                               nk_cross_tile_arguments_t const *arguments) {
    __shared__ __align__(128) nk_cross_shared_ampere_t shared;
    nk_fui32_t accumulators[4][8][4];

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        // The previous tile's last ring reads must retire before this tile's prologue refills it.
        __syncthreads();
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_ampere_(band, arguments, tile, &first_row, &first_column)) continue;
        nk_cross_accumulators_clear_ampere_(accumulators);
        nk_f32_t row_real_norm = 0, column_real_norm = 0;
        nk_cross_prologue_ampere_(&shared, arguments, first_row, first_column);
        unsigned read_stage = 0, write_stage = nk_cross_stages_ampere_k - 1;
        for (nk_size_t slab = 0; slab < arguments->depth_slabs; ++slab) {
            nk_wait_async_ampere_(nk_cross_stages_ampere_k - 2);
            __syncthreads();
            unsigned char *a_stage = shared.stages[read_stage][0], *b_stage = shared.stages[read_stage][1];
#pragma unroll
            for (unsigned sub_slab = 0; sub_slab < 2; ++sub_slab) {
                nk_u32_t a_fragments[4][4], b_fragments[8][2];
                nk_cross_load_fragments_ampere_(a_stage, b_stage, sub_slab, a_fragments, b_fragments);
                nk_dots_e4m3_multiply_ampere_(accumulators, a_fragments, b_fragments);
            }
            // Issued after the products, or ptxas sinks the commit below the fragment loads.
            nk_cross_prefetch_ampere_(&shared, arguments, first_row, first_column, slab, write_stage);
            // Squares come after the products, once the fragments are dead, and read a stage no
            // copy refills until the next iteration's barrier.
            if (metric != nk_cross_metric_dot_k) {
                nk_cross_stage_norm_e4m3_ampere_(a_stage, &row_real_norm);
                if (nk_cross_symmetric_simt_(band)) nk_cross_stage_norm_e4m3_ampere_(b_stage, &column_real_norm);
            }
            read_stage = read_stage + 1 == nk_cross_stages_ampere_k ? 0 : read_stage + 1;
            write_stage = write_stage + 1 == nk_cross_stages_ampere_k ? 0 : write_stage + 1;
        }
        nk_cross_tile_store_ampere_(
            &shared, accumulators, nk_cross_epilogue_f32_k, 65536.0f, nk_cross_norm_f32_k, metric, band, first_row,
            first_column, nk_cross_norm_finalize_simt_(nk_cross_norm_f32_k, 0, row_real_norm, 65536.0f),
            nk_cross_norm_finalize_simt_(nk_cross_norm_f32_k, 0, column_real_norm, 65536.0f), arguments);
    }
}

NUMKONG_DEVICE void nk_cross_tile_e3m2_ampere_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                               nk_cross_tile_arguments_t const *arguments) {
    __shared__ __align__(128) nk_cross_shared_ampere_t shared;
    nk_fui32_t accumulators[4][8][4];

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        // The previous tile's last ring reads must retire before this tile's prologue refills it.
        __syncthreads();
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_ampere_(band, arguments, tile, &first_row, &first_column)) continue;
        nk_cross_accumulators_clear_ampere_(accumulators);
        nk_f32_t row_real_norm = 0, column_real_norm = 0;
        nk_cross_prologue_ampere_(&shared, arguments, first_row, first_column);
        unsigned read_stage = 0, write_stage = nk_cross_stages_ampere_k - 1;
        for (nk_size_t slab = 0; slab < arguments->depth_slabs; ++slab) {
            nk_wait_async_ampere_(nk_cross_stages_ampere_k - 2);
            __syncthreads();
            unsigned char *a_stage = shared.stages[read_stage][0], *b_stage = shared.stages[read_stage][1];
#pragma unroll
            for (unsigned sub_slab = 0; sub_slab < 2; ++sub_slab) {
                nk_u32_t a_fragments[4][4], b_fragments[8][2];
                nk_cross_load_fragments_ampere_(a_stage, b_stage, sub_slab, a_fragments, b_fragments);
                nk_dots_e3m2_multiply_ampere_(accumulators, a_fragments, b_fragments);
            }
            // Issued after the products, or ptxas sinks the commit below the fragment loads.
            nk_cross_prefetch_ampere_(&shared, arguments, first_row, first_column, slab, write_stage);
            // Squares come after the products, once the fragments are dead, and read a stage no
            // copy refills until the next iteration's barrier.
            if (metric != nk_cross_metric_dot_k) {
                nk_cross_stage_norm_e3m2_ampere_(a_stage, &row_real_norm);
                if (nk_cross_symmetric_simt_(band)) nk_cross_stage_norm_e3m2_ampere_(b_stage, &column_real_norm);
            }
            read_stage = read_stage + 1 == nk_cross_stages_ampere_k ? 0 : read_stage + 1;
            write_stage = write_stage + 1 == nk_cross_stages_ampere_k ? 0 : write_stage + 1;
        }
        nk_cross_tile_store_ampere_(
            &shared, accumulators, nk_cross_epilogue_f32_k, 16777216.0f, nk_cross_norm_f32_k, metric, band, first_row,
            first_column, nk_cross_norm_finalize_simt_(nk_cross_norm_f32_k, 0, row_real_norm, 16777216.0f),
            nk_cross_norm_finalize_simt_(nk_cross_norm_f32_k, 0, column_real_norm, 16777216.0f), arguments);
    }
}

NUMKONG_DEVICE void nk_cross_tile_e2m3_ampere_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                               nk_cross_tile_arguments_t const *arguments) {
    __shared__ __align__(128) nk_cross_shared_ampere_t shared;
    nk_fui32_t accumulators[4][8][4];

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        // The previous tile's last ring reads must retire before this tile's prologue refills it.
        __syncthreads();
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_ampere_(band, arguments, tile, &first_row, &first_column)) continue;
        nk_cross_accumulators_clear_ampere_(accumulators);
        nk_u32_t row_integer_norm = 0, column_integer_norm = 0;
        nk_cross_prologue_ampere_(&shared, arguments, first_row, first_column);
        unsigned read_stage = 0, write_stage = nk_cross_stages_ampere_k - 1;
        for (nk_size_t slab = 0; slab < arguments->depth_slabs; ++slab) {
            nk_wait_async_ampere_(nk_cross_stages_ampere_k - 2);
            __syncthreads();
            unsigned char *a_stage = shared.stages[read_stage][0], *b_stage = shared.stages[read_stage][1];
#pragma unroll
            for (unsigned sub_slab = 0; sub_slab < 2; ++sub_slab) {
                nk_u32_t a_fragments[4][4], b_fragments[8][2];
                nk_cross_load_fragments_ampere_(a_stage, b_stage, sub_slab, a_fragments, b_fragments);
                nk_dots_e2m3_packed_multiply_ampere_(accumulators, a_fragments, b_fragments);
            }
            // Issued after the products, or ptxas sinks the commit below the fragment loads.
            nk_cross_prefetch_ampere_(&shared, arguments, first_row, first_column, slab, write_stage);
            // Squares come after the products, once the fragments are dead, and read a stage no
            // copy refills until the next iteration's barrier.
            if (metric != nk_cross_metric_dot_k) {
                nk_cross_stage_norm_e2m3_ampere_(a_stage, &row_integer_norm);
                if (nk_cross_symmetric_simt_(band)) nk_cross_stage_norm_e2m3_ampere_(b_stage, &column_integer_norm);
            }
            read_stage = read_stage + 1 == nk_cross_stages_ampere_k ? 0 : read_stage + 1;
            write_stage = write_stage + 1 == nk_cross_stages_ampere_k ? 0 : write_stage + 1;
        }
        nk_cross_tile_store_ampere_(
            &shared, accumulators, nk_cross_epilogue_i32_to_f32_k, 0.015625f, nk_cross_norm_f32_k, metric, band,
            first_row, first_column, nk_cross_norm_finalize_simt_(nk_cross_norm_f32_k, row_integer_norm, 0, 0.015625f),
            nk_cross_norm_finalize_simt_(nk_cross_norm_f32_k, column_integer_norm, 0, 0.015625f), arguments);
    }
}

NUMKONG_DEVICE void nk_cross_tile_e2m3_symmetric_ampere_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                                         nk_cross_tile_arguments_t const *arguments) {
    __shared__ __align__(128) nk_cross_shared_ampere_t shared;
    nk_fui32_t accumulators[4][8][4];

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        // The previous tile's last ring reads must retire before this tile's prologue refills it.
        __syncthreads();
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_ampere_(band, arguments, tile, &first_row, &first_column)) continue;
        nk_cross_accumulators_clear_ampere_(accumulators);
        nk_u32_t row_integer_norm = 0, column_integer_norm = 0;
        nk_cross_prologue_ampere_(&shared, arguments, first_row, first_column);
        unsigned read_stage = 0, write_stage = nk_cross_stages_ampere_k - 1;
        for (nk_size_t slab = 0; slab < arguments->depth_slabs; ++slab) {
            nk_wait_async_ampere_(nk_cross_stages_ampere_k - 2);
            __syncthreads();
            unsigned char *a_stage = shared.stages[read_stage][0], *b_stage = shared.stages[read_stage][1];
#pragma unroll
            for (unsigned sub_slab = 0; sub_slab < 2; ++sub_slab) {
                nk_u32_t a_fragments[4][4], b_fragments[8][2];
                nk_cross_load_fragments_ampere_(a_stage, b_stage, sub_slab, a_fragments, b_fragments);
                nk_dots_e2m3_multiply_ampere_(accumulators, a_fragments, b_fragments);
            }
            // Issued after the products, or ptxas sinks the commit below the fragment loads.
            nk_cross_prefetch_ampere_(&shared, arguments, first_row, first_column, slab, write_stage);
            // Squares come after the products, once the fragments are dead, and read a stage no
            // copy refills until the next iteration's barrier.
            if (metric != nk_cross_metric_dot_k) {
                nk_cross_stage_norm_e2m3_ampere_(a_stage, &row_integer_norm);
                if (nk_cross_symmetric_simt_(band)) nk_cross_stage_norm_e2m3_ampere_(b_stage, &column_integer_norm);
            }
            read_stage = read_stage + 1 == nk_cross_stages_ampere_k ? 0 : read_stage + 1;
            write_stage = write_stage + 1 == nk_cross_stages_ampere_k ? 0 : write_stage + 1;
        }
        nk_cross_tile_store_ampere_(
            &shared, accumulators, nk_cross_epilogue_i32_to_f32_k, 0.015625f, nk_cross_norm_f32_k, metric, band,
            first_row, first_column, nk_cross_norm_finalize_simt_(nk_cross_norm_f32_k, row_integer_norm, 0, 0.015625f),
            nk_cross_norm_finalize_simt_(nk_cross_norm_f32_k, column_integer_norm, 0, 0.015625f), arguments);
    }
}

NUMKONG_DEVICE void nk_cross_tile_e2m1_ampere_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                               nk_cross_tile_arguments_t const *arguments) {
    __shared__ __align__(128) nk_cross_shared_ampere_t shared;
    nk_fui32_t accumulators[4][8][4];

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        // The previous tile's last ring reads must retire before this tile's prologue refills it.
        __syncthreads();
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_ampere_(band, arguments, tile, &first_row, &first_column)) continue;
        nk_cross_accumulators_clear_ampere_(accumulators);
        nk_u32_t row_integer_norm = 0, column_integer_norm = 0;
        nk_cross_prologue_ampere_(&shared, arguments, first_row, first_column);
        unsigned read_stage = 0, write_stage = nk_cross_stages_ampere_k - 1;
        for (nk_size_t slab = 0; slab < arguments->depth_slabs; ++slab) {
            nk_wait_async_ampere_(nk_cross_stages_ampere_k - 2);
            __syncthreads();
            unsigned char *a_stage = shared.stages[read_stage][0], *b_stage = shared.stages[read_stage][1];
#pragma unroll
            for (unsigned sub_slab = 0; sub_slab < 2; ++sub_slab) {
                nk_u32_t a_fragments[4][4], b_fragments[8][2];
                nk_cross_load_fragments_ampere_(a_stage, b_stage, sub_slab, a_fragments, b_fragments);
                nk_dots_e2m1_multiply_ampere_(accumulators, a_fragments, b_fragments);
            }
            // Issued after the products, or ptxas sinks the commit below the fragment loads.
            nk_cross_prefetch_ampere_(&shared, arguments, first_row, first_column, slab, write_stage);
            // Squares come after the products, once the fragments are dead, and read a stage no
            // copy refills until the next iteration's barrier.
            if (metric != nk_cross_metric_dot_k) {
                nk_cross_stage_norm_e2m1_ampere_(a_stage, &row_integer_norm);
                if (nk_cross_symmetric_simt_(band)) nk_cross_stage_norm_e2m1_ampere_(b_stage, &column_integer_norm);
            }
            read_stage = read_stage + 1 == nk_cross_stages_ampere_k ? 0 : read_stage + 1;
            write_stage = write_stage + 1 == nk_cross_stages_ampere_k ? 0 : write_stage + 1;
        }
        nk_cross_tile_store_ampere_(
            &shared, accumulators, nk_cross_epilogue_i32_to_f32_k, 0.25f, nk_cross_norm_f32_k, metric, band, first_row,
            first_column, nk_cross_norm_finalize_simt_(nk_cross_norm_f32_k, row_integer_norm, 0, 0.25f),
            nk_cross_norm_finalize_simt_(nk_cross_norm_f32_k, column_integer_norm, 0, 0.25f), arguments);
    }
}

NUMKONG_DEVICE void nk_cross_tile_i8_ampere_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                             nk_cross_tile_arguments_t const *arguments) {
    __shared__ __align__(128) nk_cross_shared_ampere_t shared;
    nk_fui32_t accumulators[4][8][4];

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        // The previous tile's last ring reads must retire before this tile's prologue refills it.
        __syncthreads();
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_ampere_(band, arguments, tile, &first_row, &first_column)) continue;
        nk_cross_accumulators_clear_ampere_(accumulators);
        nk_u32_t row_integer_norm = 0, column_integer_norm = 0;
        nk_cross_prologue_ampere_(&shared, arguments, first_row, first_column);
        unsigned read_stage = 0, write_stage = nk_cross_stages_ampere_k - 1;
        for (nk_size_t slab = 0; slab < arguments->depth_slabs; ++slab) {
            nk_wait_async_ampere_(nk_cross_stages_ampere_k - 2);
            __syncthreads();
            unsigned char *a_stage = shared.stages[read_stage][0], *b_stage = shared.stages[read_stage][1];
#pragma unroll
            for (unsigned sub_slab = 0; sub_slab < 2; ++sub_slab) {
                nk_u32_t a_fragments[4][4], b_fragments[8][2];
                nk_cross_load_fragments_ampere_(a_stage, b_stage, sub_slab, a_fragments, b_fragments);
                nk_dots_i8_multiply_ampere_(accumulators, a_fragments, b_fragments);
            }
            // Issued after the products, or ptxas sinks the commit below the fragment loads.
            nk_cross_prefetch_ampere_(&shared, arguments, first_row, first_column, slab, write_stage);
            // Squares come after the products, once the fragments are dead, and read a stage no
            // copy refills until the next iteration's barrier.
            if (metric != nk_cross_metric_dot_k) {
                nk_cross_stage_norm_i8_ampere_(a_stage, &row_integer_norm);
                if (nk_cross_symmetric_simt_(band)) nk_cross_stage_norm_i8_ampere_(b_stage, &column_integer_norm);
            }
            read_stage = read_stage + 1 == nk_cross_stages_ampere_k ? 0 : read_stage + 1;
            write_stage = write_stage + 1 == nk_cross_stages_ampere_k ? 0 : write_stage + 1;
        }
        nk_cross_tile_store_ampere_(
            &shared, accumulators, nk_cross_epilogue_i32_k, 1.0f, nk_cross_norm_i32_k, metric, band, first_row,
            first_column, nk_cross_norm_finalize_simt_(nk_cross_norm_i32_k, row_integer_norm, 0, 1.0f),
            nk_cross_norm_finalize_simt_(nk_cross_norm_i32_k, column_integer_norm, 0, 1.0f), arguments);
    }
}

NUMKONG_DEVICE void nk_cross_tile_i4_ampere_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                             nk_cross_tile_arguments_t const *arguments) {
    __shared__ __align__(128) nk_cross_shared_ampere_t shared;
    nk_fui32_t accumulators[4][8][4];

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        // The previous tile's last ring reads must retire before this tile's prologue refills it.
        __syncthreads();
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_ampere_(band, arguments, tile, &first_row, &first_column)) continue;
        nk_cross_accumulators_clear_ampere_(accumulators);
        nk_u32_t row_integer_norm = 0, column_integer_norm = 0;
        nk_cross_prologue_ampere_(&shared, arguments, first_row, first_column);
        unsigned read_stage = 0, write_stage = nk_cross_stages_ampere_k - 1;
        for (nk_size_t slab = 0; slab < arguments->depth_slabs; ++slab) {
            nk_wait_async_ampere_(nk_cross_stages_ampere_k - 2);
            __syncthreads();
            unsigned char *a_stage = shared.stages[read_stage][0], *b_stage = shared.stages[read_stage][1];
#pragma unroll
            for (unsigned sub_slab = 0; sub_slab < 2; ++sub_slab) {
                nk_u32_t a_fragments[4][4], b_fragments[8][2];
                nk_cross_load_fragments_ampere_(a_stage, b_stage, sub_slab, a_fragments, b_fragments);
                nk_dots_i4_multiply_ampere_(accumulators, a_fragments, b_fragments);
            }
            // Issued after the products, or ptxas sinks the commit below the fragment loads.
            nk_cross_prefetch_ampere_(&shared, arguments, first_row, first_column, slab, write_stage);
            // Squares come after the products, once the fragments are dead, and read a stage no
            // copy refills until the next iteration's barrier.
            if (metric != nk_cross_metric_dot_k) {
                nk_cross_stage_norm_i4_ampere_(a_stage, &row_integer_norm);
                if (nk_cross_symmetric_simt_(band)) nk_cross_stage_norm_i4_ampere_(b_stage, &column_integer_norm);
            }
            read_stage = read_stage + 1 == nk_cross_stages_ampere_k ? 0 : read_stage + 1;
            write_stage = write_stage + 1 == nk_cross_stages_ampere_k ? 0 : write_stage + 1;
        }
        nk_cross_tile_store_ampere_(
            &shared, accumulators, nk_cross_epilogue_i32_k, 1.0f, nk_cross_norm_i32_k, metric, band, first_row,
            first_column, nk_cross_norm_finalize_simt_(nk_cross_norm_i32_k, row_integer_norm, 0, 1.0f),
            nk_cross_norm_finalize_simt_(nk_cross_norm_i32_k, column_integer_norm, 0, 1.0f), arguments);
    }
}

NUMKONG_DEVICE void nk_cross_tile_u8_ampere_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                             nk_cross_tile_arguments_t const *arguments) {
    __shared__ __align__(128) nk_cross_shared_ampere_t shared;
    nk_fui32_t accumulators[4][8][4];

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        // The previous tile's last ring reads must retire before this tile's prologue refills it.
        __syncthreads();
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_ampere_(band, arguments, tile, &first_row, &first_column)) continue;
        nk_cross_accumulators_clear_ampere_(accumulators);
        nk_u32_t row_integer_norm = 0, column_integer_norm = 0;
        nk_cross_prologue_ampere_(&shared, arguments, first_row, first_column);
        unsigned read_stage = 0, write_stage = nk_cross_stages_ampere_k - 1;
        for (nk_size_t slab = 0; slab < arguments->depth_slabs; ++slab) {
            nk_wait_async_ampere_(nk_cross_stages_ampere_k - 2);
            __syncthreads();
            unsigned char *a_stage = shared.stages[read_stage][0], *b_stage = shared.stages[read_stage][1];
#pragma unroll
            for (unsigned sub_slab = 0; sub_slab < 2; ++sub_slab) {
                nk_u32_t a_fragments[4][4], b_fragments[8][2];
                nk_cross_load_fragments_ampere_(a_stage, b_stage, sub_slab, a_fragments, b_fragments);
                nk_dots_u8_multiply_ampere_(accumulators, a_fragments, b_fragments);
            }
            // Issued after the products, or ptxas sinks the commit below the fragment loads.
            nk_cross_prefetch_ampere_(&shared, arguments, first_row, first_column, slab, write_stage);
            // Squares come after the products, once the fragments are dead, and read a stage no
            // copy refills until the next iteration's barrier.
            if (metric != nk_cross_metric_dot_k) {
                nk_cross_stage_norm_u8_ampere_(a_stage, &row_integer_norm);
                if (nk_cross_symmetric_simt_(band)) nk_cross_stage_norm_u8_ampere_(b_stage, &column_integer_norm);
            }
            read_stage = read_stage + 1 == nk_cross_stages_ampere_k ? 0 : read_stage + 1;
            write_stage = write_stage + 1 == nk_cross_stages_ampere_k ? 0 : write_stage + 1;
        }
        nk_cross_tile_store_ampere_(
            &shared, accumulators, nk_cross_epilogue_i32_k, 1.0f, nk_cross_norm_u32_k, metric, band, first_row,
            first_column, nk_cross_norm_finalize_simt_(nk_cross_norm_u32_k, row_integer_norm, 0, 1.0f),
            nk_cross_norm_finalize_simt_(nk_cross_norm_u32_k, column_integer_norm, 0, 1.0f), arguments);
    }
}

NUMKONG_DEVICE void nk_cross_tile_u4_ampere_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                             nk_cross_tile_arguments_t const *arguments) {
    __shared__ __align__(128) nk_cross_shared_ampere_t shared;
    nk_fui32_t accumulators[4][8][4];

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        // The previous tile's last ring reads must retire before this tile's prologue refills it.
        __syncthreads();
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_ampere_(band, arguments, tile, &first_row, &first_column)) continue;
        nk_cross_accumulators_clear_ampere_(accumulators);
        nk_u32_t row_integer_norm = 0, column_integer_norm = 0;
        nk_cross_prologue_ampere_(&shared, arguments, first_row, first_column);
        unsigned read_stage = 0, write_stage = nk_cross_stages_ampere_k - 1;
        for (nk_size_t slab = 0; slab < arguments->depth_slabs; ++slab) {
            nk_wait_async_ampere_(nk_cross_stages_ampere_k - 2);
            __syncthreads();
            unsigned char *a_stage = shared.stages[read_stage][0], *b_stage = shared.stages[read_stage][1];
#pragma unroll
            for (unsigned sub_slab = 0; sub_slab < 2; ++sub_slab) {
                nk_u32_t a_fragments[4][4], b_fragments[8][2];
                nk_cross_load_fragments_ampere_(a_stage, b_stage, sub_slab, a_fragments, b_fragments);
                nk_dots_u4_multiply_ampere_(accumulators, a_fragments, b_fragments);
            }
            // Issued after the products, or ptxas sinks the commit below the fragment loads.
            nk_cross_prefetch_ampere_(&shared, arguments, first_row, first_column, slab, write_stage);
            // Squares come after the products, once the fragments are dead, and read a stage no
            // copy refills until the next iteration's barrier.
            if (metric != nk_cross_metric_dot_k) {
                nk_cross_stage_norm_u4_ampere_(a_stage, &row_integer_norm);
                if (nk_cross_symmetric_simt_(band)) nk_cross_stage_norm_u4_ampere_(b_stage, &column_integer_norm);
            }
            read_stage = read_stage + 1 == nk_cross_stages_ampere_k ? 0 : read_stage + 1;
            write_stage = write_stage + 1 == nk_cross_stages_ampere_k ? 0 : write_stage + 1;
        }
        nk_cross_tile_store_ampere_(
            &shared, accumulators, nk_cross_epilogue_i32_k, 1.0f, nk_cross_norm_u32_k, metric, band, first_row,
            first_column, nk_cross_norm_finalize_simt_(nk_cross_norm_u32_k, row_integer_norm, 0, 1.0f),
            nk_cross_norm_finalize_simt_(nk_cross_norm_u32_k, column_integer_norm, 0, 1.0f), arguments);
    }
}

#pragma endregion Tiles

/*  Later generations read the helpers above and emit only their own kernels. */
#if NUMKONG_TARGET_AMPERE

#pragma region BF16

nk_define_cross_pack_cuda_(bf16, ampere, bf16, bf16, nk_load_b8_simt_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_cuda_(dot, bf16, ampere, bf16_ampere, bf16, bf16, f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1)

#pragma endregion BF16

#pragma region F16

nk_define_cross_pack_cuda_(f16, ampere, f16, f16, nk_load_b8_simt_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_cuda_(dot, f16, ampere, f16_ampere, f16, f16, f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1)

#pragma endregion F16

#pragma region E5M2

nk_define_cross_pack_cuda_(e5m2, ampere, e5m2, e5m2, nk_load_b8_simt_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_cuda_(dot, e5m2, ampere, e5m2_ampere, e5m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion E5M2

#pragma region E4M3

nk_define_cross_pack_cuda_(e4m3, ampere, e4m3, e4m3, nk_load_b8_simt_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_cuda_(dot, e4m3, ampere, e4m3_ampere, e4m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion E4M3

#pragma region E3M2

nk_define_cross_pack_cuda_(e3m2, ampere, e3m2, e3m2, nk_load_b8_simt_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_cuda_(dot, e3m2, ampere, e3m2_ampere, e3m2, e3m2, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion E3M2

#pragma region E2M3

nk_define_cross_pack_cuda_(e2m3, ampere, e2m3, i8, nk_load_e2m3_to_i8_ampere_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_cuda_(dot, e2m3, ampere, e2m3_ampere, e2m3, i8, f32, /*depth_simd_dimensions=*/16,
                             /*dimensions_per_value=*/1)
nk_define_cross_symmetric_cuda_(dot, e2m3, ampere, e2m3_symmetric_ampere, e2m3, i8, f32,
                                /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

#pragma endregion E2M3

#pragma region E2M1

nk_define_cross_pack_cuda_(e2m1, ampere, e2m1x2, e2m1x2, nk_load_b8_simt_, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_cuda_(dot, e2m1, ampere, e2m1_ampere, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)

#pragma endregion E2M1

#pragma region I8

nk_define_cross_pack_cuda_(i8, ampere, i8, i8, nk_load_b8_simt_, /*norm_value_type=*/u32,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_cuda_(dot, i8, ampere, i8_ampere, i8, i8, i32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion I8

#pragma region I4

nk_define_cross_pack_cuda_(i4, ampere, i4x2, i4x2, nk_load_b8_simt_, /*norm_value_type=*/u32,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_cuda_(dot, i4, ampere, i4_ampere, i4x2, i4x2, i32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)

#pragma endregion I4

#pragma region U8

nk_define_cross_pack_cuda_(u8, ampere, u8, u8, nk_load_b8_simt_, /*norm_value_type=*/u32,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_cuda_(dot, u8, ampere, u8_ampere, u8, u8, u32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion U8

#pragma region U4

nk_define_cross_pack_cuda_(u4, ampere, u4x2, u4x2, nk_load_b8_simt_, /*norm_value_type=*/u32,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_cuda_(dot, u4, ampere, u4_ampere, u4x2, u4x2, u32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)

#pragma endregion U4

#endif // NUMKONG_TARGET_AMPERE

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_AMPERE_
#endif // NUMKONG_ARCH_CUDA_
#endif // NUMKONG_DOTS_AMPERE_CUH
