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
 *  I8 for exact integer MMA, with one output multiply undoing each widening's power of two. F32
 *  widens exactly into F64 for DMMA, whose FMA chains round as the @c cuda tile's scalar ones do.
 *  F64 runs that I8 tile on residues modulo 24 moduli and rounds each rebuilt integer product once,
 *  with its workspace allocated and freed in the caller's stream order.
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

/*  F64 inputs multiply exactly as integers: each row of A and of B scales by its own power of two
 *  onto an 84-bit grid, and 24 I8 GEMMs of residues modulo pairwise coprime moduli up to 256 give
 *  their products modulo a 184-bit product, enough for depths up to 2^15. Below a depth of 512,
 *  where B200 measured grids and reconstruction outweighing the products, F64 stays in Dot2. */
enum {
    nk_cross_moduli_f64_ampere_k = 24,
    nk_cross_limbs_f64_ampere_k = 6,
    nk_cross_grid_bits_f64_ampere_k = 84,
    nk_cross_depth_floor_f64_ampere_k = 512,
    nk_cross_depth_limit_f64_ampere_k = 32768,
    nk_cross_threads_f64_ampere_k = 256,
    nk_cross_tile_outputs_f64_ampere_k = (int)nk_cross_tile_ampere_k * (int)nk_cross_tile_ampere_k,
};

/** The moduli of the F64 products and what reconstruction needs of them, limbs low first. */
typedef struct {

    /** The pairwise coprime moduli p, from 256 down. */
    nk_u32_t moduli[nk_cross_moduli_f64_ampere_k];

    /** ⌊2³² / p⌋, for reducing sums below 2³¹. */
    nk_u32_t reciprocals[nk_cross_moduli_f64_ampere_k];

    /** 2^21, 2^42 and 2^63 mod p, the weights of a grid integer's upper 21-bit digits. */
    nk_u32_t digit_weights[nk_cross_moduli_f64_ampere_k][3];

    /** (M / p) · ((M / p)⁻¹ mod p) mod M, whose sum against the residues is the integer mod M. */
    nk_u32_t weights[nk_cross_moduli_f64_ampere_k][nk_cross_limbs_f64_ampere_k];

    /** M, the product of the moduli. */
    nk_u32_t product[nk_cross_limbs_f64_ampere_k];

    /** 1 / M, rounded. */
    nk_f64_t product_inverse;
} nk_cross_moduli_f64_ampere_t;

/** What the epilogue reads of one row of A or B: its grid exponent, and what bounds the loss of
 *  truncating its values onto that grid, infinite when a value isn't finite. */
typedef struct {

    /** Σx² in Dot2, which angulars and euclideans read. */
    nk_f64_t norm;

    /** max |x|, infinite when a value isn't finite. */
    nk_f64_t maximum;

    /** Σ|x|, rounded up. */
    nk_f64_t total;

    /** Values that lost bits on the grid. */
    nk_f64_t truncated;

    /** x · 2^shift lands on the grid, below 2^84 in magnitude. */
    nk_i32_t shift;
} nk_cross_grid_row_f64_ampere_t;

/** Everything the F64 kernels of one call share, passed by value to each. */
typedef struct {

    /** The F64 operands, in the 64 × 64 tiles of the Dot2 fallback. */
    nk_cross_tile_arguments_t tiles;

    /** The moduli and the weights that rebuild their products. */
    nk_cross_moduli_f64_ampere_t moduli;

    /** I8 residues as planes, one per modulus, of rows padded to whole 64-byte slabs. */
    unsigned char *residues;

    /** One grid row per plane row: A's rows, then B's unless they are the same vectors. */
    nk_cross_grid_row_f64_ampere_t *grid_rows;

    /** Each block's residues of its current output tile, one plane per modulus. */
    unsigned char *scratch;

    /** One byte per 64 × 64 tile, set when its outputs need Dot2. */
    unsigned char *flags;

    /** The depth padded to whole 64-byte slabs. */
    nk_size_t depth_padded;

    /** Bytes of one modulus plane. */
    nk_size_t plane_bytes;

    /** The plane row where B's rows start. */
    nk_size_t column_plane_row;

    /** Column tiles of the [128,128] residue products. */
    nk_size_t residue_column_tiles;

    /** All [128,128] tiles of the residue products. */
    nk_size_t residue_tiles;

    /** What the epilogue computes. */
    nk_cross_metric_t metric;

    /** The band of the output the call fills. */
    nk_diagonal_band_t band;
} nk_cross_residue_arguments_f64_ampere_t;

/*  F32 inputs keep F64 accumulators, 64 per thread for a [128,128] tile, so twice the warps share
 *  it and each thread squares 8 of the rows a metric needs, 16 lanes to a row. */
enum {
    nk_cross_threads_f32_ampere_k = 256,
    nk_cross_tile_f32_ampere_k = nk_cross_tile_ampere_k,
    nk_cross_norm_steps_f32_ampere_k = (int)nk_cross_tile_ampere_k * (int)nk_cross_slab_simt_k /
                                       nk_cross_threads_f32_ampere_k,
};

/** Shared memory of an F32 tile: two stages of F32 slabs, A then B in each, and the squared norms
 *  of its rows and columns, which a metric computes before the first slab. */
typedef struct {
    unsigned char stages[2][2][nk_cross_stage_bytes_ampere_k];
    nk_f64_t norms[2][nk_cross_tile_ampere_k];
} nk_cross_shared_f32_ampere_t;

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
    nk_cross_threads_nvfp4_ampere_k = nk_cross_threads_ampere_k,
    nk_cross_tile_nvfp4_ampere_k = nk_cross_tile_ampere_k,
    nk_cross_threads_mxfp4_ampere_k = nk_cross_threads_ampere_k,
    nk_cross_tile_mxfp4_ampere_k = nk_cross_tile_ampere_k,
    nk_cross_threads_mxfp8e4m3_ampere_k = nk_cross_threads_ampere_k,
    nk_cross_tile_mxfp8e4m3_ampere_k = nk_cross_tile_ampere_k,
    nk_cross_threads_mxfp8e5m2_ampere_k = nk_cross_threads_ampere_k,
    nk_cross_tile_mxfp8e5m2_ampere_k = nk_cross_tile_ampere_k,
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

/** Adds an 8 × 4 by 4 × 8 F64 product into @p accumulator as one IEEE FMA chain per output, in
 *  depth order from the accumulator, exactly as four scalar FMAs would. */
NUMKONG_DEVICE void nk_mma_f64_ampere_(nk_f64_t accumulator[2], nk_f64_t a, nk_f64_t b) {
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0, %1}, {%2}, {%3}, {%0, %1};\n"
                 : "+d"(accumulator[0]), "+d"(accumulator[1])
                 : "d"(a), "d"(b));
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

/*  F64 products come back exact from integer residues, a variant of the Ozaki scheme II. Each row
 *  of A and of B scales by its own power of two so its largest value has 84 bits, truncating what
 *  falls below that grid; the I8 residues of the grid integers modulo 24 moduli multiply on the
 *  I8 tile, one GEMM per modulus; the Chinese remainder theorem rebuilds each output's integer in
 *  192-bit limbs, and it rounds once to F64. Truncation loses at most one grid unit per value, so
 *  each output bounds its loss from its rows' truncation counts, maxima and sums, and when that
 *  bound exceeds half an ulp of the result, its 64 × 64 tile reruns in Dot2. */
#pragma region F64 Tiles

/** 2^exponent for exponents of normal F64 values. */
NUMKONG_DEVICE nk_f64_t nk_f64_power_of_two_ampere_(nk_i32_t exponent) {
    return __longlong_as_double((long long)(exponent + 1023) << 52);
}

/** @p value modulo @p modulus for values below 2^31, whose reciprocal quotient is short by at most
 *  one. */
NUMKONG_DEVICE nk_u32_t nk_cross_modulo_f64_ampere_(nk_u32_t value, nk_u32_t modulus, nk_u32_t reciprocal) {
    nk_u32_t const remainder = value - __umulhi(value, reciprocal) * modulus;
    return remainder >= modulus ? remainder - modulus : remainder;
}

/** The centered I8 residue modulo moduli[index] of ±Σ digits[i] · 2^(21 i), a grid integer. */
NUMKONG_DEVICE nk_i32_t nk_cross_residue_f64_ampere_(nk_cross_moduli_f64_ampere_t const *moduli, unsigned index,
                                                     nk_u32_t const digits[4], int negative) {
    nk_u32_t const modulus = moduli->moduli[index];
    nk_u32_t const *weights = moduli->digit_weights[index];
    // Four digits below 2^21 against weights below 2^8 sum below 2^31.
    nk_u32_t const folded = digits[0] + digits[1] * weights[0] + digits[2] * weights[1] + digits[3] * weights[2];
    nk_u32_t remainder = nk_cross_modulo_f64_ampere_(folded, modulus, moduli->reciprocals[index]);
    if (negative && remainder != 0) remainder = modulus - remainder;
    return 2 * remainder >= modulus ? (nk_i32_t)remainder - (nk_i32_t)modulus : (nk_i32_t)remainder;
}

/** An I32 sum of residue products modulo @p modulus, in [0, modulus), for sums below 2^29. */
NUMKONG_DEVICE nk_u32_t nk_cross_reduce_f64_ampere_(nk_i32_t sum, nk_u32_t modulus, nk_u32_t reciprocal) {
    // A multiple of the modulus above 2^29 makes the sum positive and keeps it below 2^31.
    return nk_cross_modulo_f64_ampere_((nk_u32_t)sum + (modulus << 22), modulus, reciprocal);
}

/** Scales one row of F64 values onto its grid, writing its residue in every plane and its grid
 *  row, with the 32 lanes of one warp. */
NUMKONG_DEVICE void nk_cross_prepare_row_f64_ampere_(nk_cross_residue_arguments_f64_ampere_t const *arguments,
                                                     nk_f64_t const *values, nk_size_t plane_row) {
    unsigned const lane = threadIdx.x & 31;
    nk_size_t const depth = arguments->tiles.depth, depth_padded = arguments->depth_padded;
    nk_f64_t maximum = 0, total = 0, sum = 0, compensation = 0;
    int special = 0;
    for (nk_size_t index = lane; index < depth; index += 32) {
        nk_f64_t const value = values[index], magnitude = fabs(value);
        special |= !(magnitude <= 0x1.fffffffffffffp+1023);
        maximum = fmax(maximum, magnitude), total += magnitude;
        nk_cross_step_dot2_f64_cuda_(value, value, &sum, &compensation);
    }
    for (unsigned offset = 16; offset != 0; offset >>= 1) {
        maximum = fmax(maximum, nk_shuffle_xor_f64_cuda_(maximum, offset));
        total += nk_shuffle_xor_f64_cuda_(total, offset);
        nk_cross_merge_lanes_dot2_f64_cuda_(offset, &sum, &compensation);
    }
    special = __any_sync(0xFFFFFFFFu, special);
    nk_i32_t const shift = special || maximum == 0 ? 0 : nk_cross_grid_bits_f64_ampere_k - 1 - ilogb(maximum);
    // Two exact steps, since the shift of a subnormal row passes the largest power of two.
    nk_f64_t const first_scale = nk_f64_power_of_two_ampere_(shift / 2);
    nk_f64_t const second_scale = nk_f64_power_of_two_ampere_(shift - shift / 2);
    nk_u32_t truncated = 0;
    unsigned char *residues = arguments->residues + plane_row * depth_padded;
    for (nk_size_t index = lane; index < depth_padded; index += 32) {
        nk_f64_t const value = index < depth && !special ? values[index] : 0;
        nk_f64_t const scaled = fabs(value) * first_scale * second_scale, integer = trunc(scaled);
        truncated += value != 0 && (integer == 0 || integer != scaled);
        // Both 42-bit halves are exact, and split into 21-bit digits as integers.
        nk_f64_t const high = floor(integer * 0x1p-42), low = __fma_rn(-high, 0x1p42, integer);
        nk_u64_t const high_bits = (nk_u64_t)high, low_bits = (nk_u64_t)low;
        nk_u32_t digits[4];
        digits[0] = (nk_u32_t)low_bits & 0x1FFFFFu, digits[1] = (nk_u32_t)(low_bits >> 21);
        digits[2] = (nk_u32_t)high_bits & 0x1FFFFFu, digits[3] = (nk_u32_t)(high_bits >> 21);
#pragma unroll
        for (unsigned modulus = 0; modulus < nk_cross_moduli_f64_ampere_k; ++modulus)
            residues[modulus * arguments->plane_bytes + index] = (unsigned char)nk_cross_residue_f64_ampere_(
                &arguments->moduli, modulus, digits, value < 0);
    }
    for (unsigned offset = 16; offset != 0; offset >>= 1) truncated += __shfl_xor_sync(0xFFFFFFFFu, truncated, offset);
    if (lane == 0) {
        nk_f64_t const infinity = __longlong_as_double(0x7FF0000000000000LL);
        nk_cross_grid_row_f64_ampere_t grid_row;
        grid_row.norm = nk_f64_add_rn_cuda_(sum, compensation);
        grid_row.maximum = special ? infinity : maximum;
        // A recursive sum of at most 2^15 values errs below 2^-37 of itself.
        grid_row.total = special ? infinity : total * (1 + 0x1p-36);
        grid_row.truncated = special ? infinity : (nk_f64_t)truncated;
        grid_row.shift = shift;
        arguments->grid_rows[plane_row] = grid_row;
    }
}

/** Rebuilds the integer of one output from its residues, @p stride bytes apart, rounds it once to
 *  F64 and scales it by 2^exponent; @p zero says whether the integer is exactly zero. */
NUMKONG_DEVICE nk_f64_t nk_cross_reconstruct_f64_ampere_(nk_cross_moduli_f64_ampere_t const *moduli,
                                                         unsigned char const *residues, nk_size_t stride,
                                                         nk_i32_t exponent, int *zero) {
    nk_u64_t sums[nk_cross_limbs_f64_ampere_k] = {0};
#pragma unroll
    for (unsigned index = 0; index < nk_cross_moduli_f64_ampere_k; ++index) {
        nk_u64_t const residue = residues[index * stride];
#pragma unroll
        for (unsigned limb = 0; limb < nk_cross_limbs_f64_ampere_k; ++limb)
            sums[limb] += residue * moduli->weights[index][limb];
    }
    // S = Σ residue · weight stays below 6144 M, 197 bits.
    nk_u32_t limbs[nk_cross_limbs_f64_ampere_k + 1];
    nk_u64_t carry = 0;
#pragma unroll
    for (unsigned limb = 0; limb < nk_cross_limbs_f64_ampere_k; ++limb)
        carry += sums[limb], limbs[limb] = (nk_u32_t)carry, carry >>= 32;
    limbs[nk_cross_limbs_f64_ampere_k] = (nk_u32_t)carry;
    // The top three limbs estimate S / M to within one, and S − q M then lands in [−M, 2M).
    nk_f64_t const estimate = (nk_f64_t)limbs[6] * 0x1p192 + (nk_f64_t)limbs[5] * 0x1p160 +
                              (nk_f64_t)limbs[4] * 0x1p128;
    nk_u64_t const quotient = (nk_u64_t)floor(estimate * moduli->product_inverse);
    nk_u64_t product_carry = 0;
    nk_i64_t borrow = 0;
#pragma unroll
    for (unsigned limb = 0; limb <= nk_cross_limbs_f64_ampere_k; ++limb) {
        nk_u64_t const subtrahend = product_carry +
                                    (limb < nk_cross_limbs_f64_ampere_k ? quotient * moduli->product[limb] : 0);
        product_carry = subtrahend >> 32;
        nk_i64_t const difference = (nk_i64_t)limbs[limb] - (nk_i64_t)(nk_u32_t)subtrahend + borrow;
        limbs[limb] = (nk_u32_t)difference, borrow = difference >> 32;
    }
    // A negative difference takes M back, and one at or past M gives it up.
    int const negative_difference = borrow != 0;
    borrow = 0;
    nk_u32_t reduced[nk_cross_limbs_f64_ampere_k];
#pragma unroll
    for (unsigned limb = 0; limb < nk_cross_limbs_f64_ampere_k; ++limb) {
        nk_i64_t const term = negative_difference ? (nk_i64_t)moduli->product[limb] : -(nk_i64_t)moduli->product[limb];
        nk_i64_t const sum = (nk_i64_t)limbs[limb] + term + borrow;
        reduced[limb] = (nk_u32_t)sum, borrow = sum >> 32;
    }
    int const keep_reduced = negative_difference || borrow == 0;
#pragma unroll
    for (unsigned limb = 0; limb < nk_cross_limbs_f64_ampere_k; ++limb)
        limbs[limb] = keep_reduced ? reduced[limb] : limbs[limb];
    // The residue in [0, M) is the integer itself below M / 2, and M less its magnitude above.
    nk_u32_t complement[nk_cross_limbs_f64_ampere_k];
    borrow = 0;
#pragma unroll
    for (unsigned limb = 0; limb < nk_cross_limbs_f64_ampere_k; ++limb) {
        nk_i64_t const difference = (nk_i64_t)moduli->product[limb] - (nk_i64_t)limbs[limb] + borrow;
        complement[limb] = (nk_u32_t)difference, borrow = difference >> 32;
    }
    int greater = 0, decided = 0;
#pragma unroll
    for (unsigned limb = nk_cross_limbs_f64_ampere_k; limb-- != 0;) {
        greater = decided ? greater : limbs[limb] > complement[limb];
        decided = decided || limbs[limb] != complement[limb];
    }
    int const negative = greater;
    nk_u32_t magnitude[nk_cross_limbs_f64_ampere_k];
#pragma unroll
    for (unsigned limb = 0; limb < nk_cross_limbs_f64_ampere_k; ++limb)
        magnitude[limb] = negative ? complement[limb] : limbs[limb];
    // Whole limbs shift up until the top one leads, then 64 bits and a sticky bit round once.
    nk_i32_t limb_shift = 0;
#pragma unroll
    for (unsigned step = 0; step + 1 < nk_cross_limbs_f64_ampere_k; ++step) {
        int const empty = magnitude[nk_cross_limbs_f64_ampere_k - 1] == 0;
#pragma unroll
        for (unsigned limb = nk_cross_limbs_f64_ampere_k - 1; limb != 0; --limb)
            magnitude[limb] = empty ? magnitude[limb - 1] : magnitude[limb];
        magnitude[0] = empty ? 0 : magnitude[0];
        limb_shift += empty;
    }
    *zero = magnitude[5] == 0;
    if (*zero) return 0;
    unsigned const leading_zeros = __clz(magnitude[5]);
    nk_u64_t const top = ((nk_u64_t)magnitude[5] << 32) | magnitude[4];
    nk_u64_t const third = (nk_u64_t)magnitude[3] << leading_zeros;
    nk_u64_t const window = (top << leading_zeros) | (third >> 32);
    nk_u32_t const sticky = (nk_u32_t)third | magnitude[2] | magnitude[1] | magnitude[0];
    nk_f64_t const rounded = __ull2double_rn(window | (sticky != 0));
    nk_f64_t const scaled = ldexp(rounded, 128 - (nk_i32_t)leading_zeros - 32 * limb_shift + exponent);
    return negative ? -scaled : scaled;
}

/** Rebuilds, checks and writes the outputs of one [128,128] tile from the block's residues, each
 *  thread down one column; outputs whose truncation bound fails flag their 64 × 64 tile instead. */
NUMKONG_DEVICE void nk_cross_finish_tile_f64_ampere_(nk_cross_residue_arguments_f64_ampere_t const *arguments,
                                                     unsigned char const *scratch, nk_size_t first_row,
                                                     nk_size_t first_column) {
    nk_cross_tile_arguments_t const *tiles = &arguments->tiles;
    nk_size_t const column = first_column + threadIdx.x;
    if (column >= tiles->column_count) return;
    nk_cross_grid_row_f64_ampere_t const column_grid = arguments->grid_rows[arguments->column_plane_row + column];
    nk_f64_t const *column_norms = (nk_f64_t const *)tiles->b_norms;
    nk_f64_t const column_norm = column_norms ? column_norms[column] : column_grid.norm;
    for (unsigned tile_row = 0; tile_row < nk_cross_tile_ampere_k; ++tile_row) {
        nk_size_t const row = first_row + tile_row;
        if (row >= tiles->rows_end) break;
        nk_size_t column_begin, column_end;
        nk_diagonal_band_row_range_simt_(arguments->band, (nk_i64_t)row, tiles->column_count, &column_begin,
                                         &column_end);
        if (column < column_begin || column >= column_end) continue;
        nk_cross_grid_row_f64_ampere_t const row_grid = arguments->grid_rows[row];
        int zero;
        nk_f64_t const dot = nk_cross_reconstruct_f64_ampere_(
            &arguments->moduli, scratch + tile_row * nk_cross_tile_ampere_k + threadIdx.x,
            nk_cross_tile_outputs_f64_ampere_k, -(row_grid.shift + column_grid.shift), &zero);
        // Each truncated value lost under one grid unit, times its partner's magnitude.
        nk_f64_t const lost = ldexp(fmin(row_grid.truncated * column_grid.maximum, column_grid.total),
                                    -row_grid.shift) +
                              ldexp(fmin(column_grid.truncated * row_grid.maximum, row_grid.total), -column_grid.shift);
        nk_f64_t const magnitude = fabs(dot);
        int const keep = zero ? lost == 0
                              : magnitude >= 0x1p-1000 && magnitude <= 0x1.fffffffffffffp+1023 &&
                                    lost * (1 + 0x1p-36) <= 0x1p-54 * magnitude;
        if (!keep) {
            arguments->flags[(row - tiles->rows_begin) / nk_cross_tile_simt_k * tiles->column_tiles +
                             column / nk_cross_tile_simt_k] = 1;
            continue;
        }
        nk_f64_t *output = (nk_f64_t *)((unsigned char *)tiles->c + row * tiles->c_stride);
        if (arguments->metric == nk_cross_metric_dot_k) output[column] = dot;
        else if (nk_cross_symmetric_simt_(arguments->band) && column == row) output[column] = 0;
        else if (arguments->metric == nk_cross_metric_angular_k)
            output[column] = nk_f64_angular_simt_(dot, row_grid.norm, column_norm);
        else output[column] = nk_f64_euclidean_simt_(dot, row_grid.norm, column_norm);
    }
}

/** The residue products of [128,128] output tiles, one I8 GEMM per modulus on the I8 tile's
 *  pipeline, then their reconstruction. */
NUMKONG_DEVICE void nk_cross_tile_residues_f64_ampere_(nk_cross_residue_arguments_f64_ampere_t const *arguments) {
    __shared__ __align__(128) nk_cross_shared_ampere_t shared;
    nk_fui32_t accumulators[4][8][4];
    nk_cross_tile_arguments_t plane = arguments->tiles;
    plane.a_stride = arguments->depth_padded, plane.b_stride = arguments->depth_padded;
    plane.depth_bytes = arguments->depth_padded;
    plane.depth_slabs = arguments->depth_padded / nk_cross_slab_bytes_ampere_k;
    plane.column_tiles = arguments->residue_column_tiles, plane.tiles = arguments->residue_tiles;
    unsigned char *scratch = arguments->scratch +
                             (nk_size_t)blockIdx.x * nk_cross_moduli_f64_ampere_k * nk_cross_tile_outputs_f64_ampere_k;
    unsigned const lane = threadIdx.x & 31, warp = threadIdx.x >> 5, group = lane >> 2, quad = lane & 3;
    unsigned const warp_row = (warp >> 1) * 64, warp_column = (warp & 1) * 64;

    for (nk_size_t tile = blockIdx.x; tile < plane.tiles; tile += gridDim.x) {
        // The previous tile's reconstruction reads its residues until every thread is done.
        __syncthreads();
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_ampere_(arguments->band, &plane, tile, &first_row, &first_column)) continue;
        for (unsigned index = 0; index < nk_cross_moduli_f64_ampere_k; ++index) {
            plane.a = arguments->residues + index * arguments->plane_bytes;
            plane.b = plane.a + arguments->column_plane_row * arguments->depth_padded;
            nk_cross_accumulators_clear_ampere_(accumulators);
            nk_cross_prologue_ampere_(&shared, &plane, first_row, first_column);
            unsigned read_stage = 0, write_stage = nk_cross_stages_ampere_k - 1;
            for (nk_size_t slab = 0; slab < plane.depth_slabs; ++slab) {
                nk_wait_async_ampere_(nk_cross_stages_ampere_k - 2);
                __syncthreads();
                unsigned char *a_stage = shared.stages[read_stage][0], *b_stage = shared.stages[read_stage][1];
#pragma unroll
                for (unsigned sub_slab = 0; sub_slab < 2; ++sub_slab) {
                    nk_u32_t a_fragments[4][4], b_fragments[8][2];
                    nk_cross_load_fragments_ampere_(a_stage, b_stage, sub_slab, a_fragments, b_fragments);
                    nk_dots_i8_multiply_ampere_(accumulators, a_fragments, b_fragments);
                }
                nk_cross_prefetch_ampere_(&shared, &plane, first_row, first_column, slab, write_stage);
                read_stage = read_stage + 1 == nk_cross_stages_ampere_k ? 0 : read_stage + 1;
                write_stage = write_stage + 1 == nk_cross_stages_ampere_k ? 0 : write_stage + 1;
            }
            nk_u32_t const modulus = arguments->moduli.moduli[index];
            nk_u32_t const reciprocal = arguments->moduli.reciprocals[index];
            unsigned char *residues = scratch + index * nk_cross_tile_outputs_f64_ampere_k;
#pragma unroll
            for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
                for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
#pragma unroll
                    for (unsigned half = 0; half < 2; ++half) {
                        unsigned const tile_row = warp_row + row_tile * 16 + half * 8 + group;
                        unsigned const tile_column = warp_column + column_tile * 8 + quad * 2;
                        nk_fui32_t const *pair = &accumulators[row_tile][column_tile][half * 2];
                        nk_u32_t const low = nk_cross_reduce_f64_ampere_(pair[0].i, modulus, reciprocal);
                        nk_u32_t const high = nk_cross_reduce_f64_ampere_(pair[1].i, modulus, reciprocal);
                        *(unsigned short *)(residues + tile_row * nk_cross_tile_ampere_k +
                                            tile_column) = (unsigned short)(low | (high << 8));
                    }
            // The last slab's ring reads retire before the next modulus refills the ring, and the
            // last modulus's residues land before any thread rebuilds from them.
            __syncthreads();
        }
        nk_cross_finish_tile_f64_ampere_(arguments, scratch, first_row, first_column);
    }
}

/** The Dot2 tile of @c cuda, over the 64 × 64 tiles @p flags marks, or every one without flags. */
NUMKONG_DEVICE void nk_cross_tile_fallback_f64_ampere_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                                       nk_cross_tile_arguments_t const *arguments,
                                                       unsigned char const *flags) {
    // One extra column breaks the 64-element stride that would put a slab's stores on one bank.
    __shared__ nk_f64_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1];
    __shared__ nk_f64_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1];
    __shared__ nk_f64_t norms[2][nk_cross_tile_simt_k];

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        if (flags && !flags[tile]) continue;
        nk_size_t const first_row = arguments->rows_begin + tile / arguments->column_tiles * nk_cross_tile_simt_k;
        nk_size_t const first_column = tile % arguments->column_tiles * nk_cross_tile_simt_k;
        if (nk_diagonal_band_tile_coverage_simt_(band, (nk_i64_t)first_row, nk_cross_tile_simt_k, first_column,
                                                 nk_cross_tile_simt_k) == nk_diagonal_band_outside_k)
            continue;
        nk_f64_t sums[nk_cross_thread_tile_simt_k][nk_cross_thread_tile_simt_k] = {{0}};
        nk_f64_t compensations[nk_cross_thread_tile_simt_k][nk_cross_thread_tile_simt_k] = {{0}};
        nk_f64_t a_norms[nk_cross_loads_simt_k][2] = {{0}}, b_norms[nk_cross_loads_simt_k][2] = {{0}};
        for (nk_size_t slab = 0; slab < arguments->depth; slab += nk_cross_slab_simt_k) {
            nk_f64_t a_values[nk_cross_loads_simt_k], b_values[nk_cross_loads_simt_k];
            nk_cross_stage_slab_f64_simt_(arguments, first_row, first_column, slab, a_slab, b_slab, a_values, b_values);
            if (metric != nk_cross_metric_dot_k)
                nk_cross_accumulate_norms_f64_cuda_(band, a_values, b_values, a_norms, b_norms);
            __syncthreads();
            nk_cross_fold_slab_dot2_f64_cuda_(a_slab, b_slab, sums, compensations);
            __syncthreads();
        }
        if (metric != nk_cross_metric_dot_k) {
            nk_f64_t a_rounded[nk_cross_loads_simt_k], b_rounded[nk_cross_loads_simt_k];
            nk_cross_round_norms_f64_cuda_(band, a_norms, b_norms, a_rounded, b_rounded);
            nk_cross_publish_norms_f64_simt_(band, arguments, first_column, a_rounded, b_rounded, norms);
        }
#pragma unroll
        for (unsigned row_step = 0; row_step < nk_cross_thread_tile_simt_k; ++row_step)
#pragma unroll
            for (unsigned column_step = 0; column_step < nk_cross_thread_tile_simt_k; ++column_step)
                sums[row_step][column_step] = nk_f64_add_rn_cuda_(sums[row_step][column_step],
                                                                  compensations[row_step][column_step]);
        nk_cross_store_tile_f64_simt_(band, metric, arguments, first_row, first_column, sums, norms);
    }
}

#pragma endregion F64 Tiles

/*  F32 products are exact in F64, and DMMA adds each output's four products in depth order from its
 *  accumulator, one FMA at a time, so this tile rounds exactly as the @c cuda one does at every
 *  step, padding included. Eight warps own a [128,128] output tile, each a [64,32] grid of 8 × 8
 *  DMMA tiles, over 16-deep F32 slabs that `cp.async` copies into two swizzled stages; fragments
 *  widen to F64 as they leave shared memory, on a pipe other than the DMMA one. A metric squares
 *  its rows first, in the @c cuda tile's Dot2 order. */
#pragma region F32 Tiles

/** Issues one 64-byte depth slab of the A and B rows of a tile into a stage, 16 bytes per copy, 4
 *  copies per thread. */
NUMKONG_DEVICE void nk_cross_stage_f32_ampere_(unsigned char *a_stage, unsigned char *b_stage,
                                               nk_cross_tile_arguments_t const *arguments, nk_size_t first_row,
                                               nk_size_t first_column, nk_size_t slab_offset) {
#pragma unroll
    for (unsigned chunk = threadIdx.x; chunk < nk_cross_stage_bytes_ampere_k / 16;
         chunk += nk_cross_threads_f32_ampere_k) {
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

/** Squared norms of the 128 F32 rows from @p first into @p norms, zero past @p count: 16 lanes sum
 *  every 16th element of a row in Dot2 and merge as the @c cuda tile's do, so both round alike. */
NUMKONG_DEVICE void nk_cross_norms_f32_ampere_(unsigned char const *rows, nk_size_t stride, nk_size_t first,
                                               nk_size_t count, nk_size_t depth,
                                               nk_f64_t norms[nk_cross_tile_ampere_k]) {
    unsigned const offset = threadIdx.x % nk_cross_slab_simt_k, tile_row = threadIdx.x / nk_cross_slab_simt_k;
    unsigned const rows_per_step = (unsigned)nk_cross_threads_f32_ampere_k / nk_cross_slab_simt_k;
    nk_f64_t sums[nk_cross_norm_steps_f32_ampere_k] = {0}, compensations[nk_cross_norm_steps_f32_ampere_k] = {0};
    // Rows past the end alias the last one, whose norm is never stored, so loads need no predicate.
    nk_f32_t const *elements[nk_cross_norm_steps_f32_ampere_k];
#pragma unroll
    for (unsigned step = 0; step < nk_cross_norm_steps_f32_ampere_k; ++step) {
        nk_size_t const row = first + tile_row + step * rows_per_step;
        elements[step] = (nk_f32_t const *)(rows + (row < count ? row : count - 1) * stride);
    }
    // Four depths of loads per row go out before any square, since L2 latency bounds this loop.
    nk_size_t index = offset;
    for (; index + 3 * nk_cross_slab_simt_k < depth; index += 4 * nk_cross_slab_simt_k) {
        nk_f32_t values[4][nk_cross_norm_steps_f32_ampere_k];
#pragma unroll
        for (unsigned unrolled = 0; unrolled < 4; ++unrolled)
#pragma unroll
            for (unsigned step = 0; step < nk_cross_norm_steps_f32_ampere_k; ++step)
                values[unrolled][step] = elements[step][index + unrolled * nk_cross_slab_simt_k];
#pragma unroll
        for (unsigned unrolled = 0; unrolled < 4; ++unrolled)
#pragma unroll
            for (unsigned step = 0; step < nk_cross_norm_steps_f32_ampere_k; ++step) {
                nk_f64_t const value = values[unrolled][step];
                nk_cross_step_dot2_f64_cuda_(value, value, &sums[step], &compensations[step]);
            }
    }
    for (; index < depth; index += nk_cross_slab_simt_k)
#pragma unroll
        for (unsigned step = 0; step < nk_cross_norm_steps_f32_ampere_k; ++step) {
            nk_f64_t const value = elements[step][index];
            nk_cross_step_dot2_f64_cuda_(value, value, &sums[step], &compensations[step]);
        }
#pragma unroll
    for (unsigned step = 0; step < nk_cross_norm_steps_f32_ampere_k; ++step) {
#pragma unroll
        for (unsigned lanes = nk_cross_slab_simt_k / 2; lanes != 0; lanes >>= 1)
            nk_cross_merge_lanes_dot2_f64_cuda_(lanes, &sums[step], &compensations[step]);
        if (offset == 0) norms[tile_row + step * rows_per_step] = nk_f64_add_rn_cuda_(sums[step], compensations[step]);
    }
}

/** Adds one staged slab into this warp's [64,32] grid, four depths per DMMA, each F32 value widened
 *  to F64 as it leaves shared memory. */
NUMKONG_DEVICE void nk_dots_f32_multiply_ampere_(nk_f64_t accumulators[8][4][2], unsigned char const *a_stage,
                                                 unsigned char const *b_stage) {
    unsigned const lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    unsigned const group = lane >> 2, quad = lane & 3;
    unsigned const warp_row = (warp >> 2) * 64 + group, warp_column = (warp & 3) * 32 + group;
#pragma unroll
    for (unsigned step = 0; step < 4; ++step) {
        unsigned const byte = step * 16 + quad * 4;
        nk_f64_t a_values[8], b_values[4];
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 8; ++row_tile) {
            unsigned const offset = nk_swizzled_offset_simt_(warp_row + row_tile * 8, byte,
                                                             nk_cross_slab_bytes_ampere_k);
            nk_f32_t const value = *(nk_f32_t const *)(a_stage + offset);
            a_values[row_tile] = (nk_f64_t)value;
        }
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 4; ++column_tile) {
            unsigned const offset = nk_swizzled_offset_simt_(warp_column + column_tile * 8, byte,
                                                             nk_cross_slab_bytes_ampere_k);
            nk_f32_t const value = *(nk_f32_t const *)(b_stage + offset);
            b_values[column_tile] = (nk_f64_t)value;
        }
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 8; ++row_tile)
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 4; ++column_tile)
                nk_mma_f64_ampere_(accumulators[row_tile][column_tile], a_values[row_tile], b_values[column_tile]);
    }
}

/** Writes this warp's [64,32] outputs as F64: the dots, or metrics with zeros on the diagonal of a
 *  Gram matrix. Lane `4g + q` holds row g and columns 2q and 2q + 1 of each 8 × 8 tile. */
NUMKONG_DEVICE void nk_cross_store_f32_ampere_(nk_cross_shared_f32_ampere_t const *shared,
                                               nk_f64_t accumulators[8][4][2], nk_cross_metric_t metric,
                                               nk_diagonal_band_t band, nk_size_t first_row, nk_size_t first_column,
                                               nk_cross_tile_arguments_t const *arguments) {
    unsigned const lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    unsigned const group = lane >> 2, quad = lane & 3;
    unsigned const warp_row = (warp >> 2) * 64 + group, warp_column = (warp & 3) * 32 + quad * 2;
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 8; ++row_tile) {
        unsigned const tile_row = warp_row + row_tile * 8;
        nk_size_t const row = first_row + tile_row;
        if (row >= arguments->rows_end) continue;
        nk_size_t column_begin, column_end;
        nk_diagonal_band_row_range_simt_(band, (nk_i64_t)row, arguments->column_count, &column_begin, &column_end);
        nk_f64_t *output = (nk_f64_t *)((unsigned char *)arguments->c + row * arguments->c_stride);
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 4; ++column_tile)
#pragma unroll
            for (unsigned pair = 0; pair < 2; ++pair) {
                unsigned const tile_column = warp_column + column_tile * 8 + pair;
                nk_size_t const column = first_column + tile_column;
                if (column < column_begin || column >= column_end) continue;
                nk_f64_t const dot = accumulators[row_tile][column_tile][pair];
                if (metric == nk_cross_metric_dot_k) output[column] = dot;
                else if (nk_cross_symmetric_simt_(band) && column == row) output[column] = 0;
                else if (metric == nk_cross_metric_angular_k)
                    output[column] = nk_f64_angular_simt_(dot, shared->norms[0][tile_row],
                                                          shared->norms[1][tile_column]);
                else
                    output[column] = nk_f64_euclidean_simt_(dot, shared->norms[0][tile_row],
                                                            shared->norms[1][tile_column]);
            }
    }
}

/** One [128,128] output tile of F32 inputs on DMMA, rounding as the @c cuda tile does. */
NUMKONG_DEVICE void nk_cross_tile_f32_ampere_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                              nk_cross_tile_arguments_t const *arguments) {
    __shared__ __align__(128) nk_cross_shared_f32_ampere_t shared;
    nk_f64_t accumulators[8][4][2];

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        // The previous tile's last stage and norm reads must retire before this tile refills them.
        __syncthreads();
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_ampere_(band, arguments, tile, &first_row, &first_column)) continue;
        if (arguments->depth_slabs)
            nk_cross_stage_f32_ampere_(shared.stages[0][0], shared.stages[0][1], arguments, first_row, first_column, 0);
        nk_commit_async_ampere_();
        if (metric != nk_cross_metric_dot_k) {
            nk_cross_norms_f32_ampere_(arguments->a, arguments->a_stride, first_row, arguments->rows_end,
                                       arguments->depth, shared.norms[0]);
            if (nk_cross_symmetric_simt_(band))
                nk_cross_norms_f32_ampere_(arguments->b, arguments->b_stride, first_column, arguments->column_count,
                                           arguments->depth, shared.norms[1]);
            else if (threadIdx.x < nk_cross_tile_ampere_k) {
                nk_size_t const column = first_column + threadIdx.x;
                nk_f64_t const *column_norms = (nk_f64_t const *)arguments->b_norms;
                shared.norms[1][threadIdx.x] = column < arguments->column_count ? column_norms[column] : 0;
            }
        }
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 8; ++row_tile)
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 4; ++column_tile)
                accumulators[row_tile][column_tile][0] = 0, accumulators[row_tile][column_tile][1] = 0;
        for (nk_size_t slab = 0; slab < arguments->depth_slabs; ++slab) {
            unsigned const stage = slab & 1;
            nk_wait_async_ampere_(0);
            __syncthreads();
            // The next slab lands while this one multiplies.
            if (slab + 1 < arguments->depth_slabs)
                nk_cross_stage_f32_ampere_(shared.stages[stage ^ 1][0], shared.stages[stage ^ 1][1], arguments,
                                           first_row, first_column, (slab + 1) * nk_cross_slab_bytes_ampere_k);
            nk_commit_async_ampere_();
            nk_dots_f32_multiply_ampere_(accumulators, shared.stages[stage][0], shared.stages[stage][1]);
        }
        // Without slabs the norms still need a barrier before other threads read them.
        __syncthreads();
        nk_cross_store_f32_ampere_(&shared, accumulators, metric, band, first_row, first_column, arguments);
    }
}

#pragma endregion F32 Tiles

/*  Block-scaled tiles keep the codes compact through shared memory and widen each warp's fragments
 *  in registers right before the products. NVFP4 scales are exact in F16, so its codes widen to F16
 *  through one byte table, take the scale by one HMUL2 per pair and multiply in F16 unrebased.
 *  MXFP4 widens to BF16 through two byte tables and takes its rebased power-of-two scale by one
 *  BF16 FMA per pair. MXFP8 widens to F16 as the plain FP8 tiles do and folds each 32-deep block's
 *  F16 products into F32 times both rebased scales, as the SIMT tiles fold their partial sums. Two
 *  stages of codes leave room for every stage's scales in 48 KB of static shared memory. */
#pragma region Block Scales

/** Shared memory of a block-scaled tile: two stages of codes, A then B in each, the multipliers of
 *  each stage's blocks for the 128 rows of A and of B, 16 bytes a row, and the rows' rebasing. */
typedef struct {
    unsigned char codes[2][2][nk_cross_stage_bytes_ampere_k];
    nk_u32_t scales[2][2][nk_cross_tile_ampere_k][4];
    nk_i32_t bases[2][nk_cross_tile_ampere_k];
    nk_i32_t spreads[2][nk_cross_tile_ampere_k];
} nk_cross_scaled_shared_ampere_t;

NUMKONG_DEVICE nk_u32_t nk_f16x2_multiply_ampere_(nk_u32_t values, nk_u32_t scales) {
    nk_u32_t products;
    asm("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(products) : "r"(values), "r"(scales));
    return products;
}

NUMKONG_DEVICE nk_u32_t nk_bf16x2_multiply_ampere_(nk_u32_t values, nk_u32_t scales) {
    nk_u32_t products;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(products) : "r"(values), "r"(scales), "r"(0x80008000u));
    return products;
}

NUMKONG_DEVICE nk_u32_t nk_f32_to_f16_bits_ampere_(nk_f32_t value) {
    unsigned short bits;
    asm("cvt.rn.f16.f32 %0, %1;\n" : "=h"(bits) : "f"(value));
    return bits;
}

NUMKONG_DEVICE nk_u32_t nk_f32_to_bf16_bits_ampere_(nk_f32_t value) {
    unsigned short bits;
    asm("cvt.rn.bf16.f32 %0, %1;\n" : "=h"(bits) : "f"(value));
    return bits;
}

/** The high bytes of four E2M1 codes' F16 or BF16 values, through tables @p low and @p high of the
 *  eight magnitudes, the sign from each nibble's bit 3. */
NUMKONG_DEVICE nk_u32_t nk_e2m1x4_high_bytes_ampere_(nk_u32_t codes, nk_u32_t low, nk_u32_t high) {
    nk_u32_t const magnitudes = __byte_perm(low, high, codes & 0x7777u);
    // Bytes b0 b0 b1 b1 hold the even codes' signs at bit 3 and the odd codes' at bit 7
    nk_u32_t const doubled = __byte_perm(codes, 0, 0x1100);
    return magnitudes | ((doubled << 4) & 0x00800080u) | (doubled & 0x80008000u);
}

/** Four E2M1 codes, the low 16 bits of @p codes, as two pairs of F16, whose low bytes are zero. */
NUMKONG_DEVICE void nk_e2m1x4_to_f16x4_ampere_(nk_u32_t codes, nk_u32_t *first, nk_u32_t *second) {
    nk_u32_t const high_bytes = nk_e2m1x4_high_bytes_ampere_(codes, 0x3E3C3800u, 0x46444240u);
    *first = __byte_perm(high_bytes, 0, 0x1404), *second = __byte_perm(high_bytes, 0, 0x3424);
}

/** Four E2M1 codes, the low 16 bits of @p codes, as two pairs of BF16. */
NUMKONG_DEVICE void nk_e2m1x4_to_bf16x4_ampere_(nk_u32_t codes, nk_u32_t *first, nk_u32_t *second) {
    nk_u32_t const high_bytes = nk_e2m1x4_high_bytes_ampere_(codes, 0x3F3F3F00u, 0x40404040u);
    nk_u32_t const low_bytes = __byte_perm(0xC0800000u, 0xC0804000u, codes & 0x7777u);
    *first = __byte_perm(low_bytes, high_bytes, 0x5140), *second = __byte_perm(low_bytes, high_bytes, 0x7362);
}

/** Copies one 64-byte slab of A and B codes into stage @p stage, or nothing past the depth, and
 *  commits a group either way, so every slab waits on one. */
NUMKONG_DEVICE void nk_cross_scaled_stage_ampere_(nk_cross_scaled_shared_ampere_t *shared, unsigned stage,
                                                  nk_cross_tile_arguments_t const *arguments, nk_size_t first_row,
                                                  nk_size_t first_column, nk_size_t slab) {
    if (slab < arguments->depth_slabs)
        nk_cross_stage_ampere_(shared->codes[stage][0], shared->codes[stage][1], arguments, first_row, first_column,
                               slab * nk_cross_slab_bytes_ampere_k);
    nk_commit_async_ampere_();
}

/** Reads @p count scale codes from block @p first of the A row and the B row this thread stages,
 *  zeros past the rows and past @p blocks. */
NUMKONG_DEVICE void nk_cross_scaled_codes_ampere_(nk_cross_tile_arguments_t const *arguments, nk_size_t first_row,
                                                  nk_size_t first_column, nk_size_t blocks, nk_size_t first,
                                                  unsigned count, unsigned char codes[2][8]) {
    nk_size_t const row = first_row + threadIdx.x, column = first_column + threadIdx.x;
#pragma unroll
    for (unsigned block = 0; block < 8; ++block) {
        int const inside = block < count && first + block < blocks;
        codes[0][block] = inside && row < arguments->rows_end
                              ? arguments->a_scales[row * arguments->a_scales_stride + first + block]
                              : 0;
        codes[1][block] = inside && column < arguments->column_count
                              ? arguments->b_scales[column * arguments->b_scales_stride + first + block]
                              : 0;
    }
}

/** One UE8M0 scale relative to @p base: 2^(code − 127 − base), 0x00 to zero, 0xFF to a NaN; codes
 *  far below the base decode garbage, as on the host, which the exact path overwrites. */
NUMKONG_DEVICE nk_f32_t nk_relative_ue8m0_to_f32_ampere_(unsigned code, nk_i32_t base) {
    return __uint_as_float(code == 0 ? 0 : code == 255 ? 0x7FC00000u : (nk_u32_t)((nk_i32_t)code - base) << 23);
}

/** @c nk_cross_scaled_base_ue8m0_simt_ over 16 independent loads at a time, as one thread scans a
 *  whole row of scales per tile. */
NUMKONG_DEVICE nk_i32_t nk_cross_scaled_base_ue8m0_ampere_(unsigned char const *scales, nk_size_t blocks,
                                                           nk_i32_t *spread) {
    unsigned low = 255, high = 0;
    for (nk_size_t first = 0; first < blocks; first += 16) {
#pragma unroll
        for (unsigned block = 0; block < 16; ++block) {
            unsigned const code = first + block < blocks ? scales[first + block] : 0;
            // Codes 0 and 255 are zero and NaN scales, which no rebasing covers
            if (code != 0 && code != 255) low = code < low ? code : low, high = code > high ? code : high;
        }
    }
    *spread = low > high ? 0 : (nk_i32_t)(high - low);
    return low > high ? 0 : (nk_i32_t)high - 127 - nk_cross_scaled_headroom_k;
}

/** Rebases the A row and the B row this thread stages over their @p blocks UE8M0 scales. */
NUMKONG_DEVICE void nk_cross_scaled_rebase_ue8m0_ampere_(nk_cross_scaled_shared_ampere_t *shared,
                                                         nk_cross_tile_arguments_t const *arguments,
                                                         nk_size_t first_row, nk_size_t first_column,
                                                         nk_size_t blocks) {
    nk_size_t const row = first_row + threadIdx.x, column = first_column + threadIdx.x;
    nk_i32_t spread = 0;
    shared->bases[0][threadIdx.x] = row < arguments->rows_end
                                        ? nk_cross_scaled_base_ue8m0_ampere_(
                                              arguments->a_scales + row * arguments->a_scales_stride, blocks, &spread)
                                        : 0;
    shared->spreads[0][threadIdx.x] = spread, spread = 0;
    shared->bases[1][threadIdx.x] = column < arguments->column_count
                                        ? nk_cross_scaled_base_ue8m0_ampere_(
                                              arguments->b_scales + column * arguments->b_scales_stride, blocks,
                                              &spread)
                                        : 0;
    shared->spreads[1][threadIdx.x] = spread;
}

#pragma endregion Block Scales

#pragma region NVFP4 Tiles

/** Stores this thread's A and B row's 8 UE4M3 scales of one slab as F16 into stage @p stage. */
NUMKONG_DEVICE void nk_cross_scaled_store_nvfp4_ampere_(nk_cross_scaled_shared_ampere_t *shared, unsigned stage,
                                                        unsigned char const codes[2][8]) {
#pragma unroll
    for (unsigned side = 0; side < 2; ++side)
#pragma unroll
        for (unsigned pair = 0; pair < 4; ++pair) {
            nk_i32_t exponent;
            nk_f32_t const first = nk_ue4m3_split_simt_(codes[side][pair * 2], &exponent);
            nk_f32_t const second = nk_ue4m3_split_simt_(codes[side][pair * 2 + 1], &exponent);
            shared->scales[stage][side][threadIdx.x][pair] = nk_f32_to_f16_bits_ampere_(first) |
                                                             (nk_f32_to_f16_bits_ampere_(second) << 16);
        }
}

/** Eight E2M1 codes of a fragment register as four F16 pairs, times @p scale of their block. */
NUMKONG_DEVICE void nk_e2m1x8_scaled_to_f16x8_ampere_(nk_u32_t codes, nk_u32_t scale, nk_u32_t pairs[4]) {
    nk_e2m1x4_to_f16x4_ampere_(codes, &pairs[0], &pairs[1]);
    nk_e2m1x4_to_f16x4_ampere_(codes >> 16, &pairs[2], &pairs[3]);
    nk_u32_t const scales = __byte_perm(scale, 0, 0x1010);
#pragma unroll
    for (unsigned pair = 0; pair < 4; ++pair) pairs[pair] = nk_f16x2_multiply_ampere_(pairs[pair], scales);
}

/*  Each fragment register holds 8 codes of one row, depths 8 · quad on of its 32-deep chunk, so of
 *  one 16-deep NVFP4 block; its first pairs and its last pairs make the two halves of K in two
 *  steps of the 16-deep MMA. */
NUMKONG_DEVICE void nk_dots_nvfp4_multiply_ampere_(nk_fui32_t accumulators[4][8][4], nk_u32_t const a[4][4],
                                                   nk_u32_t const b[8][2],
                                                   nk_cross_scaled_shared_ampere_t const *shared, unsigned stage,
                                                   unsigned sub_slab) {
    unsigned const lane = threadIdx.x & 31, warp = threadIdx.x >> 5, group = lane >> 2, quad = lane & 3;
    unsigned const warp_row = (warp >> 1) * 64, warp_column = (warp & 1) * 64;
    // Rows of 4 words each hold a slab's 8 16-bit scales, A rows then B rows
    nk_u16_t const *a_scales = (nk_u16_t const *)shared->scales[stage][0],
                   *b_scales = (nk_u16_t const *)shared->scales[stage][1];
#pragma unroll
    for (unsigned chunk = 0; chunk < 2; ++chunk) {
        unsigned const block = sub_slab * 4 + chunk * 2 + (quad >> 1);
        nk_u32_t b_pairs[8][4];
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
            nk_e2m1x8_scaled_to_f16x8_ampere_(b[column_tile][chunk],
                                              b_scales[(warp_column + column_tile * 8 + group) * 8 + block],
                                              b_pairs[column_tile]);
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile) {
            unsigned const top = warp_row + row_tile * 16 + group;
            nk_u32_t top_pairs[4], bottom_pairs[4];
            nk_e2m1x8_scaled_to_f16x8_ampere_(a[row_tile][chunk * 2], a_scales[top * 8 + block], top_pairs);
            nk_e2m1x8_scaled_to_f16x8_ampere_(a[row_tile][chunk * 2 + 1], a_scales[(top + 8) * 8 + block],
                                              bottom_pairs);
#pragma unroll
            for (unsigned step = 0; step < 2; ++step) {
                nk_u32_t const fragment[4] = {top_pairs[step], bottom_pairs[step], top_pairs[step + 2],
                                              bottom_pairs[step + 2]};
#pragma unroll
                for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
                    nk_mma_f16_ampere_(accumulators[row_tile][column_tile], fragment, b_pairs[column_tile][step],
                                       b_pairs[column_tile][step + 2]);
            }
        }
    }
}

NUMKONG_DEVICE void nk_cross_tile_nvfp4_ampere_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                                nk_cross_tile_arguments_t const *arguments) {
    __shared__ __align__(128) nk_cross_scaled_shared_ampere_t shared;
    nk_fui32_t accumulators[4][8][4];
    nk_cross_scaled_factors_cuda_t const factors = nk_cross_scaled_factors_cuda_(arguments);
    nk_size_t const blocks = arguments->depth / 16;
    unsigned const lane = threadIdx.x & 31, warp = threadIdx.x >> 5, group = lane >> 2, quad = lane & 3;
    unsigned const warp_row = (warp >> 1) * 64, warp_column = (warp & 1) * 64;

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        // The previous tile's last reads of the ring must retire before this tile refills it.
        __syncthreads();
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_ampere_(band, arguments, tile, &first_row, &first_column)) continue;
        nk_cross_accumulators_clear_ampere_(accumulators);
        unsigned char codes[2][8];
        nk_cross_scaled_stage_ampere_(&shared, 0, arguments, first_row, first_column, 0);
        nk_cross_scaled_codes_ampere_(arguments, first_row, first_column, blocks, 0, 8, codes);
        nk_cross_scaled_store_nvfp4_ampere_(&shared, 0, codes);
        for (nk_size_t slab = 0; slab < arguments->depth_slabs; ++slab) {
            unsigned const stage = slab & 1;
            nk_wait_async_ampere_(0);
            __syncthreads();
            // The next slab's codes and scales load while this one multiplies.
            nk_cross_scaled_stage_ampere_(&shared, stage ^ 1, arguments, first_row, first_column, slab + 1);
            nk_cross_scaled_codes_ampere_(arguments, first_row, first_column, blocks, (slab + 1) * 8, 8, codes);
#pragma unroll
            for (unsigned sub_slab = 0; sub_slab < 2; ++sub_slab) {
                nk_u32_t a_fragments[4][4], b_fragments[8][2];
                nk_cross_load_fragments_ampere_(shared.codes[stage][0], shared.codes[stage][1], sub_slab, a_fragments,
                                                b_fragments);
                nk_dots_nvfp4_multiply_ampere_(accumulators, a_fragments, b_fragments, &shared, stage, sub_slab);
            }
            nk_cross_scaled_store_nvfp4_ampere_(&shared, stage ^ 1, codes);
        }
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
            for (unsigned half = 0; half < 2; ++half) {
                nk_size_t const row = first_row + warp_row + row_tile * 16 + half * 8 + group;
                if (row >= arguments->rows_end) continue;
                nk_size_t column_begin, column_end;
                nk_diagonal_band_row_range_simt_(band, (nk_i64_t)row, arguments->column_count, &column_begin,
                                                 &column_end);
                nk_f32_t *output = (nk_f32_t *)((unsigned char *)arguments->c + row * arguments->c_stride);
#pragma unroll
                for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
#pragma unroll
                    for (unsigned pair = 0; pair < 2; ++pair) {
                        nk_size_t const column = first_column + warp_column + column_tile * 8 + quad * 2 + pair;
                        if (column < column_begin || column >= column_end) continue;
                        nk_cross_wide_sum_t dot, a_sumsq, b_sumsq;
                        nk_cross_scaled_wide_cuda_(accumulators[row_tile][column_tile][half * 2 + pair].f, 0, 0, 0, 0,
                                                   &factors, &dot, &a_sumsq, &b_sumsq);
                        output[column] = nk_cross_scaled_metric_simt_(metric, dot, a_sumsq, b_sumsq);
                    }
            }
    }
}

#pragma endregion NVFP4 Tiles

#pragma region MXFP4 Tiles

/** Stores this thread's A and B row's 4 UE8M0 scales of one slab, rebased, as BF16 into stage
 *  @p stage. */
NUMKONG_DEVICE void nk_cross_scaled_store_mxfp4_ampere_(nk_cross_scaled_shared_ampere_t *shared, unsigned stage,
                                                        unsigned char const codes[2][8]) {
#pragma unroll
    for (unsigned side = 0; side < 2; ++side) {
        nk_i32_t const base = shared->bases[side][threadIdx.x];
#pragma unroll
        for (unsigned pair = 0; pair < 2; ++pair)
            shared->scales[stage][side][threadIdx.x][pair] =
                nk_f32_to_bf16_bits_ampere_(nk_relative_ue8m0_to_f32_ampere_(codes[side][pair * 2], base)) |
                (nk_f32_to_bf16_bits_ampere_(nk_relative_ue8m0_to_f32_ampere_(codes[side][pair * 2 + 1], base)) << 16);
    }
}

/** Eight E2M1 codes of a fragment register as four BF16 pairs, times @p scale of their block. */
NUMKONG_DEVICE void nk_e2m1x8_scaled_to_bf16x8_ampere_(nk_u32_t codes, nk_u32_t scale, nk_u32_t pairs[4]) {
    nk_e2m1x4_to_bf16x4_ampere_(codes, &pairs[0], &pairs[1]);
    nk_e2m1x4_to_bf16x4_ampere_(codes >> 16, &pairs[2], &pairs[3]);
    nk_u32_t const scales = __byte_perm(scale, 0, 0x1010);
#pragma unroll
    for (unsigned pair = 0; pair < 4; ++pair) pairs[pair] = nk_bf16x2_multiply_ampere_(pairs[pair], scales);
}

/*  Each 32-deep chunk is one MXFP4 block, so every register of a chunk shares one scale. */
NUMKONG_DEVICE void nk_dots_mxfp4_multiply_ampere_(nk_fui32_t accumulators[4][8][4], nk_u32_t const a[4][4],
                                                   nk_u32_t const b[8][2],
                                                   nk_cross_scaled_shared_ampere_t const *shared, unsigned stage,
                                                   unsigned sub_slab) {
    unsigned const lane = threadIdx.x & 31, warp = threadIdx.x >> 5, group = lane >> 2;
    unsigned const warp_row = (warp >> 1) * 64, warp_column = (warp & 1) * 64;
    // Rows of 4 words each hold a slab's 8 16-bit scales, A rows then B rows
    nk_u16_t const *a_scales = (nk_u16_t const *)shared->scales[stage][0],
                   *b_scales = (nk_u16_t const *)shared->scales[stage][1];
#pragma unroll
    for (unsigned chunk = 0; chunk < 2; ++chunk) {
        unsigned const block = sub_slab * 2 + chunk;
        nk_u32_t b_pairs[8][4];
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
            nk_e2m1x8_scaled_to_bf16x8_ampere_(b[column_tile][chunk],
                                               b_scales[(warp_column + column_tile * 8 + group) * 8 + block],
                                               b_pairs[column_tile]);
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile) {
            unsigned const top = warp_row + row_tile * 16 + group;
            nk_u32_t top_pairs[4], bottom_pairs[4];
            nk_e2m1x8_scaled_to_bf16x8_ampere_(a[row_tile][chunk * 2], a_scales[top * 8 + block], top_pairs);
            nk_e2m1x8_scaled_to_bf16x8_ampere_(a[row_tile][chunk * 2 + 1], a_scales[(top + 8) * 8 + block],
                                               bottom_pairs);
#pragma unroll
            for (unsigned step = 0; step < 2; ++step) {
                nk_u32_t const fragment[4] = {top_pairs[step], bottom_pairs[step], top_pairs[step + 2],
                                              bottom_pairs[step + 2]};
#pragma unroll
                for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
                    nk_mma_bf16_ampere_(accumulators[row_tile][column_tile], fragment, b_pairs[column_tile][step],
                                        b_pairs[column_tile][step + 2]);
            }
        }
    }
}

/** Stores this thread's A and B row's 4 UE8M0 scales of one slab, rebased, as F16 into stage
 *  @p stage, for tiles whose rows and columns all fit F16. */
NUMKONG_DEVICE void nk_cross_scaled_store_narrow_mxfp4_ampere_(nk_cross_scaled_shared_ampere_t *shared, unsigned stage,
                                                               unsigned char const codes[2][8]) {
#pragma unroll
    for (unsigned side = 0; side < 2; ++side) {
        nk_i32_t const base = shared->bases[side][threadIdx.x];
#pragma unroll
        for (unsigned pair = 0; pair < 2; ++pair)
            shared->scales[stage][side][threadIdx.x][pair] =
                nk_f32_to_f16_bits_ampere_(nk_relative_ue8m0_to_f32_ampere_(codes[side][pair * 2], base)) |
                (nk_f32_to_f16_bits_ampere_(nk_relative_ue8m0_to_f32_ampere_(codes[side][pair * 2 + 1], base)) << 16);
    }
}

/*  Rebased 13 binades under the largest scale, E2M1 values stay below F16's 65,504, and a spread of
 *  at most 36 binades keeps the smallest at least F16's 2⁻²⁴, so these tiles run as NVFP4 tiles. */
NUMKONG_DEVICE void nk_dots_mxfp4_narrow_multiply_ampere_(nk_fui32_t accumulators[4][8][4], nk_u32_t const a[4][4],
                                                          nk_u32_t const b[8][2],
                                                          nk_cross_scaled_shared_ampere_t const *shared, unsigned stage,
                                                          unsigned sub_slab) {
    unsigned const lane = threadIdx.x & 31, warp = threadIdx.x >> 5, group = lane >> 2;
    unsigned const warp_row = (warp >> 1) * 64, warp_column = (warp & 1) * 64;
    // Rows of 4 words each hold a slab's 8 16-bit scales, A rows then B rows
    nk_u16_t const *a_scales = (nk_u16_t const *)shared->scales[stage][0],
                   *b_scales = (nk_u16_t const *)shared->scales[stage][1];
#pragma unroll
    for (unsigned chunk = 0; chunk < 2; ++chunk) {
        unsigned const block = sub_slab * 2 + chunk;
        nk_u32_t b_pairs[8][4];
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
            nk_e2m1x8_scaled_to_f16x8_ampere_(b[column_tile][chunk],
                                              b_scales[(warp_column + column_tile * 8 + group) * 8 + block],
                                              b_pairs[column_tile]);
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile) {
            unsigned const top = warp_row + row_tile * 16 + group;
            nk_u32_t top_pairs[4], bottom_pairs[4];
            nk_e2m1x8_scaled_to_f16x8_ampere_(a[row_tile][chunk * 2], a_scales[top * 8 + block], top_pairs);
            nk_e2m1x8_scaled_to_f16x8_ampere_(a[row_tile][chunk * 2 + 1], a_scales[(top + 8) * 8 + block],
                                              bottom_pairs);
#pragma unroll
            for (unsigned step = 0; step < 2; ++step) {
                nk_u32_t const fragment[4] = {top_pairs[step], bottom_pairs[step], top_pairs[step + 2],
                                              bottom_pairs[step + 2]};
#pragma unroll
                for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
                    nk_mma_f16_ampere_(accumulators[row_tile][column_tile], fragment, b_pairs[column_tile][step],
                                       b_pairs[column_tile][step + 2]);
            }
        }
    }
}

NUMKONG_DEVICE void nk_cross_tile_mxfp4_ampere_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                                nk_cross_tile_arguments_t const *arguments) {
    __shared__ __align__(128) nk_cross_scaled_shared_ampere_t shared;
    nk_fui32_t accumulators[4][8][4];
    nk_cross_scaled_factors_cuda_t const factors = nk_cross_scaled_factors_cuda_(arguments);
    nk_size_t const depth = arguments->depth, blocks = depth / 32;
    unsigned const lane = threadIdx.x & 31, warp = threadIdx.x >> 5, group = lane >> 2, quad = lane & 3;
    unsigned const warp_row = (warp >> 1) * 64, warp_column = (warp & 1) * 64;

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        // The previous tile's last reads of the ring and of the bases must retire before refills.
        __syncthreads();
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_ampere_(band, arguments, tile, &first_row, &first_column)) continue;
        nk_cross_accumulators_clear_ampere_(accumulators);
        nk_cross_scaled_rebase_ue8m0_ampere_(&shared, arguments, first_row, first_column, blocks);
        // Tiles whose rows and columns span at most 36 binades rebase 18 binades higher into F16
        int const narrow = __syncthreads_and(shared.spreads[0][threadIdx.x] <= 36 &&
                                             shared.spreads[1][threadIdx.x] <= 36);
        if (narrow) shared.bases[0][threadIdx.x] += 18, shared.bases[1][threadIdx.x] += 18;
        unsigned char codes[2][8];
        nk_cross_scaled_stage_ampere_(&shared, 0, arguments, first_row, first_column, 0);
        nk_cross_scaled_codes_ampere_(arguments, first_row, first_column, blocks, 0, 4, codes);
        // Each lane runs its own loop, so neither one's registers crowd the other's
        if (narrow) {
            nk_cross_scaled_store_narrow_mxfp4_ampere_(&shared, 0, codes);
            for (nk_size_t slab = 0; slab < arguments->depth_slabs; ++slab) {
                unsigned const stage = slab & 1;
                nk_wait_async_ampere_(0);
                __syncthreads();
                // The next slab's codes and scales load while this one multiplies.
                nk_cross_scaled_stage_ampere_(&shared, stage ^ 1, arguments, first_row, first_column, slab + 1);
                nk_cross_scaled_codes_ampere_(arguments, first_row, first_column, blocks, (slab + 1) * 4, 4, codes);
#pragma unroll
                for (unsigned sub_slab = 0; sub_slab < 2; ++sub_slab) {
                    nk_u32_t a_fragments[4][4], b_fragments[8][2];
                    nk_cross_load_fragments_ampere_(shared.codes[stage][0], shared.codes[stage][1], sub_slab,
                                                    a_fragments, b_fragments);
                    nk_dots_mxfp4_narrow_multiply_ampere_(accumulators, a_fragments, b_fragments, &shared, stage,
                                                          sub_slab);
                }
                nk_cross_scaled_store_narrow_mxfp4_ampere_(&shared, stage ^ 1, codes);
            }
        }
        else {
            nk_cross_scaled_store_mxfp4_ampere_(&shared, 0, codes);
            for (nk_size_t slab = 0; slab < arguments->depth_slabs; ++slab) {
                unsigned const stage = slab & 1;
                nk_wait_async_ampere_(0);
                __syncthreads();
                // The next slab's codes and scales load while this one multiplies.
                nk_cross_scaled_stage_ampere_(&shared, stage ^ 1, arguments, first_row, first_column, slab + 1);
                nk_cross_scaled_codes_ampere_(arguments, first_row, first_column, blocks, (slab + 1) * 4, 4, codes);
#pragma unroll
                for (unsigned sub_slab = 0; sub_slab < 2; ++sub_slab) {
                    nk_u32_t a_fragments[4][4], b_fragments[8][2];
                    nk_cross_load_fragments_ampere_(shared.codes[stage][0], shared.codes[stage][1], sub_slab,
                                                    a_fragments, b_fragments);
                    nk_dots_mxfp4_multiply_ampere_(accumulators, a_fragments, b_fragments, &shared, stage, sub_slab);
                }
                nk_cross_scaled_store_mxfp4_ampere_(&shared, stage ^ 1, codes);
            }
        }
        // Without slabs the rebasing still needs a barrier before other threads' rows read it.
        __syncthreads();
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
            for (unsigned half = 0; half < 2; ++half) {
                unsigned const tile_row = warp_row + row_tile * 16 + half * 8 + group;
                nk_size_t const row = first_row + tile_row;
                if (row >= arguments->rows_end) continue;
                nk_size_t column_begin, column_end;
                nk_diagonal_band_row_range_simt_(band, (nk_i64_t)row, arguments->column_count, &column_begin,
                                                 &column_end);
                nk_f32_t *output = (nk_f32_t *)((unsigned char *)arguments->c + row * arguments->c_stride);
#pragma unroll
                for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
#pragma unroll
                    for (unsigned pair = 0; pair < 2; ++pair) {
                        unsigned const tile_column = warp_column + column_tile * 8 + quad * 2 + pair;
                        nk_size_t const column = first_column + tile_column;
                        if (column < column_begin || column >= column_end) continue;
                        nk_cross_wide_sum_t dot, a_sumsq, b_sumsq;
                        nk_cross_scaled_wide_cuda_(accumulators[row_tile][column_tile][half * 2 + pair].f,
                                                   shared.bases[0][tile_row], shared.bases[1][tile_column], 0, 0,
                                                   &factors, &dot, &a_sumsq, &b_sumsq);
                        if (nk_cross_scaled_exceeds_simt_(shared.spreads[0][tile_row], shared.spreads[1][tile_column],
                                                          0)) {
                            dot = nk_cross_scaled_exact_wide_mxfp4_simt_(
                                arguments->a + row * arguments->a_stride,
                                arguments->a_scales + row * arguments->a_scales_stride,
                                arguments->b + column * arguments->b_stride,
                                arguments->b_scales + column * arguments->b_scales_stride, depth, &a_sumsq, &b_sumsq);
                            nk_cross_scaled_apply_factors_cuda_(&factors, &dot, &a_sumsq, &b_sumsq);
                        }
                        output[column] = nk_cross_scaled_metric_simt_(metric, dot, a_sumsq, b_sumsq);
                    }
            }
    }
}

#pragma endregion MXFP4 Tiles

#pragma region MXFP8E4M3 Tiles

/** Stores this thread's A and B row's 2 UE8M0 scales of one slab, rebased and times 2⁸ to undo the
 *  widening, as F32 into stage @p stage. */
NUMKONG_DEVICE void nk_cross_scaled_store_mxfp8e4m3_ampere_(nk_cross_scaled_shared_ampere_t *shared, unsigned stage,
                                                            unsigned char const codes[2][8]) {
#pragma unroll
    for (unsigned side = 0; side < 2; ++side) {
        nk_i32_t const base = shared->bases[side][threadIdx.x];
#pragma unroll
        for (unsigned block = 0; block < 2; ++block)
            shared->scales[stage][side][threadIdx.x][block] = __float_as_uint(
                nk_relative_ue8m0_to_f32_ampere_(codes[side][block], base) * 256.0f);
    }
}

/*  A 32-byte sub-slab is one MXFP8 block: its two F16 steps gather into a partial, which joins the
 *  accumulators times the row's and the column's scales, multiplied first so neither underflows. */
NUMKONG_DEVICE void nk_dots_mxfp8e4m3_multiply_ampere_(nk_fui32_t accumulators[4][8][4], nk_u32_t const a[4][4],
                                                       nk_u32_t const b[8][2],
                                                       nk_cross_scaled_shared_ampere_t const *shared, unsigned stage,
                                                       unsigned sub_slab) {
    unsigned const lane = threadIdx.x & 31, warp = threadIdx.x >> 5, group = lane >> 2, quad = lane & 3;
    unsigned const warp_row = (warp >> 1) * 64, warp_column = (warp & 1) * 64;
    nk_u32_t b_low[8][2], b_high[8][2];
#pragma unroll
    for (unsigned column_tile = 0; column_tile < 8; ++column_tile) {
#pragma unroll
        for (unsigned depth_half = 0; depth_half < 2; ++depth_half)
            nk_e4m3x4_to_f16x4_ampere_(b[column_tile][depth_half], &b_low[column_tile][depth_half],
                                       &b_high[column_tile][depth_half]);
    }
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile) {
        unsigned const top = warp_row + row_tile * 16 + group;
        nk_f32_t const top_scale = __uint_as_float(shared->scales[stage][0][top][sub_slab]);
        nk_f32_t const bottom_scale = __uint_as_float(shared->scales[stage][0][top + 8][sub_slab]);
        nk_u32_t a_low[4], a_high[4];
#pragma unroll
        for (unsigned fragment = 0; fragment < 4; ++fragment)
            nk_e4m3x4_to_f16x4_ampere_(a[row_tile][fragment], &a_low[fragment], &a_high[fragment]);
        nk_u32_t const first[4] = {a_low[0], a_low[1], a_high[0], a_high[1]};
        nk_u32_t const second[4] = {a_low[2], a_low[3], a_high[2], a_high[3]};
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 8; ++column_tile) {
            nk_fui32_t partial[4] = {{0}, {0}, {0}, {0}};
            nk_mma_f16_ampere_(partial, first, b_low[column_tile][0], b_high[column_tile][0]);
            nk_mma_f16_ampere_(partial, second, b_low[column_tile][1], b_high[column_tile][1]);
            // Column scales come from shared memory here, as registers hold the accumulators
            unsigned const column = warp_column + column_tile * 8 + quad * 2;
            nk_f32_t const left_scale = __uint_as_float(shared->scales[stage][1][column][sub_slab]);
            nk_f32_t const right_scale = __uint_as_float(shared->scales[stage][1][column + 1][sub_slab]);
            nk_fui32_t *sums = accumulators[row_tile][column_tile];
            sums[0].f = __fmaf_rn(partial[0].f, top_scale * left_scale, sums[0].f);
            sums[1].f = __fmaf_rn(partial[1].f, top_scale * right_scale, sums[1].f);
            sums[2].f = __fmaf_rn(partial[2].f, bottom_scale * left_scale, sums[2].f);
            sums[3].f = __fmaf_rn(partial[3].f, bottom_scale * right_scale, sums[3].f);
        }
    }
}

/** Stores this thread's A and B row's 2 UE8M0 scales of one slab, rebased and times 2⁸ to undo the
 *  widening, as F16 pairs into stage @p stage, for tiles whose rows and columns all fit F16. */
NUMKONG_DEVICE void nk_cross_scaled_store_narrow_mxfp8e4m3_ampere_(nk_cross_scaled_shared_ampere_t *shared,
                                                                   unsigned stage, unsigned char const codes[2][8]) {
#pragma unroll
    for (unsigned side = 0; side < 2; ++side) {
        nk_i32_t const base = shared->bases[side][threadIdx.x];
#pragma unroll
        for (unsigned block = 0; block < 2; ++block)
            shared->scales[stage][side][threadIdx.x][block] = __byte_perm(
                nk_f32_to_f16_bits_ampere_(nk_relative_ue8m0_to_f32_ampere_(codes[side][block], base) * 256.0f), 0,
                0x1010);
    }
}

/*  Rebased 7 binades under the largest scale, E4M3 values stay below F16's 65,504, and a spread of
 *  at most 22 binades keeps the smallest subnormal at least F16's 2⁻²⁴, so such tiles scale every
 *  pair before the F16 steps and fold nothing per block. */
NUMKONG_DEVICE void nk_dots_mxfp8e4m3_narrow_multiply_ampere_(nk_fui32_t accumulators[4][8][4], nk_u32_t const a[4][4],
                                                              nk_u32_t const b[8][2],
                                                              nk_cross_scaled_shared_ampere_t const *shared,
                                                              unsigned stage, unsigned sub_slab) {
    unsigned const lane = threadIdx.x & 31, warp = threadIdx.x >> 5, group = lane >> 2;
    unsigned const warp_row = (warp >> 1) * 64, warp_column = (warp & 1) * 64;
    nk_u32_t b_low[8][2], b_high[8][2];
#pragma unroll
    for (unsigned column_tile = 0; column_tile < 8; ++column_tile) {
        nk_u32_t const scale = shared->scales[stage][1][warp_column + column_tile * 8 + group][sub_slab];
#pragma unroll
        for (unsigned depth_half = 0; depth_half < 2; ++depth_half) {
            nk_e4m3x4_to_f16x4_ampere_(b[column_tile][depth_half], &b_low[column_tile][depth_half],
                                       &b_high[column_tile][depth_half]);
            b_low[column_tile][depth_half] = nk_f16x2_multiply_ampere_(b_low[column_tile][depth_half], scale);
            b_high[column_tile][depth_half] = nk_f16x2_multiply_ampere_(b_high[column_tile][depth_half], scale);
        }
    }
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile) {
        unsigned const top = warp_row + row_tile * 16 + group;
        nk_u32_t const scales[2] = {shared->scales[stage][0][top][sub_slab],
                                    shared->scales[stage][0][top + 8][sub_slab]};
        nk_u32_t a_low[4], a_high[4];
#pragma unroll
        for (unsigned fragment = 0; fragment < 4; ++fragment) {
            nk_e4m3x4_to_f16x4_ampere_(a[row_tile][fragment], &a_low[fragment], &a_high[fragment]);
            a_low[fragment] = nk_f16x2_multiply_ampere_(a_low[fragment], scales[fragment & 1]);
            a_high[fragment] = nk_f16x2_multiply_ampere_(a_high[fragment], scales[fragment & 1]);
        }
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

NUMKONG_DEVICE void nk_cross_tile_mxfp8e4m3_ampere_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                                    nk_cross_tile_arguments_t const *arguments) {
    __shared__ __align__(128) nk_cross_scaled_shared_ampere_t shared;
    nk_fui32_t accumulators[4][8][4];
    nk_cross_scaled_factors_cuda_t const factors = nk_cross_scaled_factors_cuda_(arguments);
    nk_size_t const depth = arguments->depth, blocks = depth / 32;
    unsigned const lane = threadIdx.x & 31, warp = threadIdx.x >> 5, group = lane >> 2, quad = lane & 3;
    unsigned const warp_row = (warp >> 1) * 64, warp_column = (warp & 1) * 64;

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        // The previous tile's last reads of the ring and of the bases must retire before refills.
        __syncthreads();
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_ampere_(band, arguments, tile, &first_row, &first_column)) continue;
        nk_cross_accumulators_clear_ampere_(accumulators);
        nk_cross_scaled_rebase_ue8m0_ampere_(&shared, arguments, first_row, first_column, blocks);
        // Tiles whose rows and columns span at most 22 binades rebase 24 binades higher into F16
        int const narrow = __syncthreads_and(shared.spreads[0][threadIdx.x] <= 22 &&
                                             shared.spreads[1][threadIdx.x] <= 22);
        if (narrow) shared.bases[0][threadIdx.x] += 24, shared.bases[1][threadIdx.x] += 24;
        unsigned char codes[2][8];
        nk_cross_scaled_stage_ampere_(&shared, 0, arguments, first_row, first_column, 0);
        nk_cross_scaled_codes_ampere_(arguments, first_row, first_column, blocks, 0, 2, codes);
        // Each lane runs its own loop, so neither one's registers crowd the other's
        if (narrow) {
            nk_cross_scaled_store_narrow_mxfp8e4m3_ampere_(&shared, 0, codes);
            for (nk_size_t slab = 0; slab < arguments->depth_slabs; ++slab) {
                unsigned const stage = slab & 1;
                nk_wait_async_ampere_(0);
                __syncthreads();
                // The next slab's codes and scales load while this one multiplies.
                nk_cross_scaled_stage_ampere_(&shared, stage ^ 1, arguments, first_row, first_column, slab + 1);
                nk_cross_scaled_codes_ampere_(arguments, first_row, first_column, blocks, (slab + 1) * 2, 2, codes);
#pragma unroll
                for (unsigned sub_slab = 0; sub_slab < 2; ++sub_slab) {
                    nk_u32_t a_fragments[4][4], b_fragments[8][2];
                    nk_cross_load_fragments_ampere_(shared.codes[stage][0], shared.codes[stage][1], sub_slab,
                                                    a_fragments, b_fragments);
                    nk_dots_mxfp8e4m3_narrow_multiply_ampere_(accumulators, a_fragments, b_fragments, &shared, stage,
                                                              sub_slab);
                }
                nk_cross_scaled_store_narrow_mxfp8e4m3_ampere_(&shared, stage ^ 1, codes);
            }
        }
        else {
            nk_cross_scaled_store_mxfp8e4m3_ampere_(&shared, 0, codes);
            for (nk_size_t slab = 0; slab < arguments->depth_slabs; ++slab) {
                unsigned const stage = slab & 1;
                nk_wait_async_ampere_(0);
                __syncthreads();
                // The next slab's codes and scales load while this one multiplies.
                nk_cross_scaled_stage_ampere_(&shared, stage ^ 1, arguments, first_row, first_column, slab + 1);
                nk_cross_scaled_codes_ampere_(arguments, first_row, first_column, blocks, (slab + 1) * 2, 2, codes);
#pragma unroll
                for (unsigned sub_slab = 0; sub_slab < 2; ++sub_slab) {
                    nk_u32_t a_fragments[4][4], b_fragments[8][2];
                    nk_cross_load_fragments_ampere_(shared.codes[stage][0], shared.codes[stage][1], sub_slab,
                                                    a_fragments, b_fragments);
                    nk_dots_mxfp8e4m3_multiply_ampere_(accumulators, a_fragments, b_fragments, &shared, stage,
                                                       sub_slab);
                }
                nk_cross_scaled_store_mxfp8e4m3_ampere_(&shared, stage ^ 1, codes);
            }
        }
        // Without slabs the rebasing still needs a barrier before other threads' rows read it.
        __syncthreads();
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
            for (unsigned half = 0; half < 2; ++half) {
                unsigned const tile_row = warp_row + row_tile * 16 + half * 8 + group;
                nk_size_t const row = first_row + tile_row;
                if (row >= arguments->rows_end) continue;
                nk_size_t column_begin, column_end;
                nk_diagonal_band_row_range_simt_(band, (nk_i64_t)row, arguments->column_count, &column_begin,
                                                 &column_end);
                nk_f32_t *output = (nk_f32_t *)((unsigned char *)arguments->c + row * arguments->c_stride);
#pragma unroll
                for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
#pragma unroll
                    for (unsigned pair = 0; pair < 2; ++pair) {
                        unsigned const tile_column = warp_column + column_tile * 8 + quad * 2 + pair;
                        nk_size_t const column = first_column + tile_column;
                        if (column < column_begin || column >= column_end) continue;
                        nk_cross_wide_sum_t dot, a_sumsq, b_sumsq;
                        nk_cross_scaled_wide_cuda_(accumulators[row_tile][column_tile][half * 2 + pair].f,
                                                   shared.bases[0][tile_row], shared.bases[1][tile_column], 0, 0,
                                                   &factors, &dot, &a_sumsq, &b_sumsq);
                        if (nk_cross_scaled_exceeds_simt_(shared.spreads[0][tile_row], shared.spreads[1][tile_column],
                                                          0)) {
                            dot = nk_cross_scaled_exact_wide_mxfp8e4m3_simt_(
                                arguments->a + row * arguments->a_stride,
                                arguments->a_scales + row * arguments->a_scales_stride,
                                arguments->b + column * arguments->b_stride,
                                arguments->b_scales + column * arguments->b_scales_stride, depth, &a_sumsq, &b_sumsq);
                            nk_cross_scaled_apply_factors_cuda_(&factors, &dot, &a_sumsq, &b_sumsq);
                        }
                        output[column] = nk_cross_scaled_metric_simt_(metric, dot, a_sumsq, b_sumsq);
                    }
            }
    }
}

#pragma endregion MXFP8E4M3 Tiles
#pragma region MXFP8E5M2 Tiles

/** Stores this thread's A and B row's 2 UE8M0 scales of one slab, rebased, as F32 into
 *  stage @p stage. */
NUMKONG_DEVICE void nk_cross_scaled_store_mxfp8e5m2_ampere_(nk_cross_scaled_shared_ampere_t *shared, unsigned stage,
                                                            unsigned char const codes[2][8]) {
#pragma unroll
    for (unsigned side = 0; side < 2; ++side) {
        nk_i32_t const base = shared->bases[side][threadIdx.x];
#pragma unroll
        for (unsigned block = 0; block < 2; ++block)
            shared->scales[stage][side][threadIdx.x][block] = __float_as_uint(
                nk_relative_ue8m0_to_f32_ampere_(codes[side][block], base));
    }
}

/*  A 32-byte sub-slab is one MXFP8 block: its two F16 steps gather into a partial, which joins the
 *  accumulators times the row's and the column's scales, multiplied first so neither underflows. */
NUMKONG_DEVICE void nk_dots_mxfp8e5m2_multiply_ampere_(nk_fui32_t accumulators[4][8][4], nk_u32_t const a[4][4],
                                                       nk_u32_t const b[8][2],
                                                       nk_cross_scaled_shared_ampere_t const *shared, unsigned stage,
                                                       unsigned sub_slab) {
    unsigned const lane = threadIdx.x & 31, warp = threadIdx.x >> 5, group = lane >> 2, quad = lane & 3;
    unsigned const warp_row = (warp >> 1) * 64, warp_column = (warp & 1) * 64;
    nk_u32_t b_low[8][2], b_high[8][2];
#pragma unroll
    for (unsigned column_tile = 0; column_tile < 8; ++column_tile) {
#pragma unroll
        for (unsigned depth_half = 0; depth_half < 2; ++depth_half)
            nk_e5m2x4_to_f16x4_ampere_(b[column_tile][depth_half], &b_low[column_tile][depth_half],
                                       &b_high[column_tile][depth_half]);
    }
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile) {
        unsigned const top = warp_row + row_tile * 16 + group;
        nk_f32_t const top_scale = __uint_as_float(shared->scales[stage][0][top][sub_slab]);
        nk_f32_t const bottom_scale = __uint_as_float(shared->scales[stage][0][top + 8][sub_slab]);
        nk_u32_t a_low[4], a_high[4];
#pragma unroll
        for (unsigned fragment = 0; fragment < 4; ++fragment)
            nk_e5m2x4_to_f16x4_ampere_(a[row_tile][fragment], &a_low[fragment], &a_high[fragment]);
        nk_u32_t const first[4] = {a_low[0], a_low[1], a_high[0], a_high[1]};
        nk_u32_t const second[4] = {a_low[2], a_low[3], a_high[2], a_high[3]};
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 8; ++column_tile) {
            nk_fui32_t partial[4] = {{0}, {0}, {0}, {0}};
            nk_mma_f16_ampere_(partial, first, b_low[column_tile][0], b_high[column_tile][0]);
            nk_mma_f16_ampere_(partial, second, b_low[column_tile][1], b_high[column_tile][1]);
            // Column scales come from shared memory here, as registers hold the accumulators
            unsigned const column = warp_column + column_tile * 8 + quad * 2;
            nk_f32_t const left_scale = __uint_as_float(shared->scales[stage][1][column][sub_slab]);
            nk_f32_t const right_scale = __uint_as_float(shared->scales[stage][1][column + 1][sub_slab]);
            nk_fui32_t *sums = accumulators[row_tile][column_tile];
            sums[0].f = __fmaf_rn(partial[0].f, top_scale * left_scale, sums[0].f);
            sums[1].f = __fmaf_rn(partial[1].f, top_scale * right_scale, sums[1].f);
            sums[2].f = __fmaf_rn(partial[2].f, bottom_scale * left_scale, sums[2].f);
            sums[3].f = __fmaf_rn(partial[3].f, bottom_scale * right_scale, sums[3].f);
        }
    }
}

/** Stores this thread's A and B row's 2 UE8M0 scales of one slab, rebased, as F16
 *  pairs into stage @p stage, for tiles whose rows and columns all fit F16. */
NUMKONG_DEVICE void nk_cross_scaled_store_narrow_mxfp8e5m2_ampere_(nk_cross_scaled_shared_ampere_t *shared,
                                                                   unsigned stage, unsigned char const codes[2][8]) {
#pragma unroll
    for (unsigned side = 0; side < 2; ++side) {
        nk_i32_t const base = shared->bases[side][threadIdx.x];
#pragma unroll
        for (unsigned block = 0; block < 2; ++block)
            shared->scales[stage][side][threadIdx.x][block] = __byte_perm(
                nk_f32_to_f16_bits_ampere_(nk_relative_ue8m0_to_f32_ampere_(codes[side][block], base)), 0, 0x1010);
    }
}

/*  Rebased to the largest scale, E5M2 values stay below F16's 65,504, and a spread of at most 8
 *  binades keeps the smallest subnormal at least F16's 2⁻²⁴, so such tiles scale every pair
 *  before the F16 steps and fold nothing per block. */
NUMKONG_DEVICE void nk_dots_mxfp8e5m2_narrow_multiply_ampere_(nk_fui32_t accumulators[4][8][4], nk_u32_t const a[4][4],
                                                              nk_u32_t const b[8][2],
                                                              nk_cross_scaled_shared_ampere_t const *shared,
                                                              unsigned stage, unsigned sub_slab) {
    unsigned const lane = threadIdx.x & 31, warp = threadIdx.x >> 5, group = lane >> 2;
    unsigned const warp_row = (warp >> 1) * 64, warp_column = (warp & 1) * 64;
    nk_u32_t b_low[8][2], b_high[8][2];
#pragma unroll
    for (unsigned column_tile = 0; column_tile < 8; ++column_tile) {
        nk_u32_t const scale = shared->scales[stage][1][warp_column + column_tile * 8 + group][sub_slab];
#pragma unroll
        for (unsigned depth_half = 0; depth_half < 2; ++depth_half) {
            nk_e5m2x4_to_f16x4_ampere_(b[column_tile][depth_half], &b_low[column_tile][depth_half],
                                       &b_high[column_tile][depth_half]);
            b_low[column_tile][depth_half] = nk_f16x2_multiply_ampere_(b_low[column_tile][depth_half], scale);
            b_high[column_tile][depth_half] = nk_f16x2_multiply_ampere_(b_high[column_tile][depth_half], scale);
        }
    }
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile) {
        unsigned const top = warp_row + row_tile * 16 + group;
        nk_u32_t const scales[2] = {shared->scales[stage][0][top][sub_slab],
                                    shared->scales[stage][0][top + 8][sub_slab]};
        nk_u32_t a_low[4], a_high[4];
#pragma unroll
        for (unsigned fragment = 0; fragment < 4; ++fragment) {
            nk_e5m2x4_to_f16x4_ampere_(a[row_tile][fragment], &a_low[fragment], &a_high[fragment]);
            a_low[fragment] = nk_f16x2_multiply_ampere_(a_low[fragment], scales[fragment & 1]);
            a_high[fragment] = nk_f16x2_multiply_ampere_(a_high[fragment], scales[fragment & 1]);
        }
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

NUMKONG_DEVICE void nk_cross_tile_mxfp8e5m2_ampere_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                                    nk_cross_tile_arguments_t const *arguments) {
    __shared__ __align__(128) nk_cross_scaled_shared_ampere_t shared;
    nk_fui32_t accumulators[4][8][4];
    nk_cross_scaled_factors_cuda_t const factors = nk_cross_scaled_factors_cuda_(arguments);
    nk_size_t const depth = arguments->depth, blocks = depth / 32;
    unsigned const lane = threadIdx.x & 31, warp = threadIdx.x >> 5, group = lane >> 2, quad = lane & 3;
    unsigned const warp_row = (warp >> 1) * 64, warp_column = (warp & 1) * 64;

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        // The previous tile's last reads of the ring and of the bases must retire before refills.
        __syncthreads();
        nk_size_t first_row, first_column;
        if (!nk_cross_tile_origin_ampere_(band, arguments, tile, &first_row, &first_column)) continue;
        nk_cross_accumulators_clear_ampere_(accumulators);
        nk_cross_scaled_rebase_ue8m0_ampere_(&shared, arguments, first_row, first_column, blocks);
        // Tiles whose rows and columns all span at most 8 binades rebase 31 binades higher into F16
        int const narrow = __syncthreads_and(shared.spreads[0][threadIdx.x] <= 8 &&
                                             shared.spreads[1][threadIdx.x] <= 8);
        if (narrow) shared.bases[0][threadIdx.x] += 31, shared.bases[1][threadIdx.x] += 31;
        unsigned char codes[2][8];
        nk_cross_scaled_stage_ampere_(&shared, 0, arguments, first_row, first_column, 0);
        nk_cross_scaled_codes_ampere_(arguments, first_row, first_column, blocks, 0, 2, codes);
        // Each lane runs its own loop, so neither one's registers crowd the other's
        if (narrow) {
            nk_cross_scaled_store_narrow_mxfp8e5m2_ampere_(&shared, 0, codes);
            for (nk_size_t slab = 0; slab < arguments->depth_slabs; ++slab) {
                unsigned const stage = slab & 1;
                nk_wait_async_ampere_(0);
                __syncthreads();
                // The next slab's codes and scales load while this one multiplies.
                nk_cross_scaled_stage_ampere_(&shared, stage ^ 1, arguments, first_row, first_column, slab + 1);
                nk_cross_scaled_codes_ampere_(arguments, first_row, first_column, blocks, (slab + 1) * 2, 2, codes);
#pragma unroll
                for (unsigned sub_slab = 0; sub_slab < 2; ++sub_slab) {
                    nk_u32_t a_fragments[4][4], b_fragments[8][2];
                    nk_cross_load_fragments_ampere_(shared.codes[stage][0], shared.codes[stage][1], sub_slab,
                                                    a_fragments, b_fragments);
                    nk_dots_mxfp8e5m2_narrow_multiply_ampere_(accumulators, a_fragments, b_fragments, &shared, stage,
                                                              sub_slab);
                }
                nk_cross_scaled_store_narrow_mxfp8e5m2_ampere_(&shared, stage ^ 1, codes);
            }
        }
        else {
            nk_cross_scaled_store_mxfp8e5m2_ampere_(&shared, 0, codes);
            for (nk_size_t slab = 0; slab < arguments->depth_slabs; ++slab) {
                unsigned const stage = slab & 1;
                nk_wait_async_ampere_(0);
                __syncthreads();
                // The next slab's codes and scales load while this one multiplies.
                nk_cross_scaled_stage_ampere_(&shared, stage ^ 1, arguments, first_row, first_column, slab + 1);
                nk_cross_scaled_codes_ampere_(arguments, first_row, first_column, blocks, (slab + 1) * 2, 2, codes);
#pragma unroll
                for (unsigned sub_slab = 0; sub_slab < 2; ++sub_slab) {
                    nk_u32_t a_fragments[4][4], b_fragments[8][2];
                    nk_cross_load_fragments_ampere_(shared.codes[stage][0], shared.codes[stage][1], sub_slab,
                                                    a_fragments, b_fragments);
                    nk_dots_mxfp8e5m2_multiply_ampere_(accumulators, a_fragments, b_fragments, &shared, stage,
                                                       sub_slab);
                }
                nk_cross_scaled_store_mxfp8e5m2_ampere_(&shared, stage ^ 1, codes);
            }
        }
        // Without slabs the rebasing still needs a barrier before other threads' rows read it.
        __syncthreads();
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
            for (unsigned half = 0; half < 2; ++half) {
                unsigned const tile_row = warp_row + row_tile * 16 + half * 8 + group;
                nk_size_t const row = first_row + tile_row;
                if (row >= arguments->rows_end) continue;
                nk_size_t column_begin, column_end;
                nk_diagonal_band_row_range_simt_(band, (nk_i64_t)row, arguments->column_count, &column_begin,
                                                 &column_end);
                nk_f32_t *output = (nk_f32_t *)((unsigned char *)arguments->c + row * arguments->c_stride);
#pragma unroll
                for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
#pragma unroll
                    for (unsigned pair = 0; pair < 2; ++pair) {
                        unsigned const tile_column = warp_column + column_tile * 8 + quad * 2 + pair;
                        nk_size_t const column = first_column + tile_column;
                        if (column < column_begin || column >= column_end) continue;
                        nk_cross_wide_sum_t dot, a_sumsq, b_sumsq;
                        nk_cross_scaled_wide_cuda_(accumulators[row_tile][column_tile][half * 2 + pair].f,
                                                   shared.bases[0][tile_row], shared.bases[1][tile_column], 0, 0,
                                                   &factors, &dot, &a_sumsq, &b_sumsq);
                        if (nk_cross_scaled_exceeds_simt_(shared.spreads[0][tile_row], shared.spreads[1][tile_column],
                                                          0)) {
                            dot = nk_cross_scaled_exact_wide_mxfp8e5m2_simt_(
                                arguments->a + row * arguments->a_stride,
                                arguments->a_scales + row * arguments->a_scales_stride,
                                arguments->b + column * arguments->b_stride,
                                arguments->b_scales + column * arguments->b_scales_stride, depth, &a_sumsq, &b_sumsq);
                            nk_cross_scaled_apply_factors_cuda_(&factors, &dot, &a_sumsq, &b_sumsq);
                        }
                        output[column] = nk_cross_scaled_metric_simt_(metric, dot, a_sumsq, b_sumsq);
                    }
            }
    }
}

#pragma endregion MXFP8E5M2 Tiles

/*  Later generations read the helpers above and emit only their own kernels. */
#if NUMKONG_TARGET_AMPERE

#pragma region F64

static __global__ void __launch_bounds__(nk_cross_threads_f64_ampere_k)
    nk_cross_prepare_f64_ampere_kernel_(__grid_constant__ nk_cross_residue_arguments_f64_ampere_t const arguments,
                                        unsigned char const *rows, nk_size_t stride, nk_size_t count,
                                        nk_size_t first_plane_row) {
    nk_size_t const warps = (nk_size_t)gridDim.x * (blockDim.x >> 5);
    for (nk_size_t row = (nk_size_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); row < count; row += warps)
        nk_cross_prepare_row_f64_ampere_(&arguments, (nk_f64_t const *)(rows + row * stride), first_plane_row + row);
}

static __global__ void __launch_bounds__(nk_cross_threads_ampere_k)
    nk_cross_residues_f64_ampere_kernel_(__grid_constant__ nk_cross_residue_arguments_f64_ampere_t const arguments) {
    nk_cross_tile_residues_f64_ampere_(&arguments);
}

static __global__ void __launch_bounds__(nk_cross_threads_simt_k)
    nk_cross_fallback_f64_ampere_kernel_(__grid_constant__ nk_cross_residue_arguments_f64_ampere_t const arguments) {
    nk_cross_tile_fallback_f64_ampere_(arguments.metric, arguments.band, &arguments.tiles, arguments.flags);
}

/** Fills the moduli of the F64 products and what reconstruction needs of them. */
NUMKONG_INLINE void nk_cross_moduli_f64_ampere_(nk_cross_moduli_f64_ampere_t *moduli) {
    static nk_u32_t const values[nk_cross_moduli_f64_ampere_k] = {256, 255, 253, 251, 247, 241, 239, 233,
                                                                  229, 227, 223, 211, 199, 197, 193, 191,
                                                                  181, 179, 173, 167, 163, 157, 151, 149};
    static nk_u32_t const weights[nk_cross_moduli_f64_ampere_k][nk_cross_limbs_f64_ampere_k] = {
        {0xCAF98D01u, 0x149EC617u, 0xE2F4F31Du, 0x8DDA2DA9u, 0xF60CCF00u, 0x00768228u},
        {0xF5ED5400u, 0x4114C791u, 0x740FFED0u, 0xADD0B845u, 0xCADBEFF0u, 0x009EA182u},
        {0x12A63200u, 0xDF81DBBAu, 0xEBF784CEu, 0x16DC001Eu, 0xA25D6D6Cu, 0x0109C12Fu},
        {0xAAF5A700u, 0xA6F15389u, 0x3513CA32u, 0xE5351366u, 0xCD8E86E1u, 0x01089AFFu},
        {0xF76DCC00u, 0x41D992D3u, 0x233CB3E6u, 0x44764DDCu, 0xB1FC3C49u, 0x0105250Eu},
        {0x262B4500u, 0xF5C9024Du, 0xE7097213u, 0xA0ADB700u, 0xBA59DB65u, 0x0042E95Cu},
        {0xD99ED000u, 0x4D103FF3u, 0xF2F69B2Fu, 0xFB05F6EDu, 0xE2839B24u, 0x00EDDD99u},
        {0xBA4EFC00u, 0xA69F4EC0u, 0xE5E3CB84u, 0x7DB8B80Fu, 0x6CB5103Fu, 0x010B739Bu},
        {0x93577800u, 0x3F106013u, 0xF37E8B69u, 0xEC0A2EEFu, 0xD0FA40B7u, 0x002FBDACu},
        {0x4D449500u, 0xD2FE088Fu, 0x41DBA407u, 0xB8AE29CCu, 0x8CF7348Cu, 0x00880E6Du},
        {0xAAE9EC00u, 0x8220D8E5u, 0x3EC9D5F5u, 0x35E89ABBu, 0x704ECD95u, 0x0035ED8Fu},
        {0xA6CBF500u, 0xAF2DE727u, 0xBACFB561u, 0xA6DD802Du, 0x99BF4024u, 0x00D08CBCu},
        {0x92D3BF00u, 0x3187FAF5u, 0xC32D44C4u, 0xA2B5A4FFu, 0xFD37CE51u, 0x00A370CFu},
        {0x32002300u, 0x6DA8B7E8u, 0x25A69763u, 0xAED34084u, 0x325163E3u, 0x00DF5EDCu},
        {0x4EC79700u, 0xC82CC3DFu, 0x623BCC58u, 0xD8AE2254u, 0x39AE068Fu, 0x0050B872u},
        {0xD4331000u, 0x597C75EAu, 0xC9C04EA7u, 0x8BBD0AFBu, 0xF60C4159u, 0x0016E548u},
        {0xE738C800u, 0x4BE095DFu, 0xB6DC3E44u, 0x4250B628u, 0x5B081568u, 0x00243DABu},
        {0x1F6D7B00u, 0x6D9469F7u, 0x7AEEC418u, 0x4A284CBFu, 0x4626573Eu, 0x00A97CA7u},
        {0x1BFC0B00u, 0x6DE54DFDu, 0xC4786627u, 0xCCE72990u, 0x7985E515u, 0x003422B4u},
        {0x5C850100u, 0x2082CCCDu, 0x5B6A91CBu, 0x04CC8303u, 0x9CAE2FDBu, 0x00ABD88Au},
        {0xB26B9000u, 0x0AB8385Cu, 0xEC911CECu, 0xD47AD8B1u, 0x549C9CB7u, 0x008624A8u},
        {0xF83D4400u, 0x88AE4C9Bu, 0xD9EA92C5u, 0xD3112CABu, 0xB61AAC22u, 0x0014E3F4u},
        {0x1BE08800u, 0xBC16A95Du, 0x396960C2u, 0x4BB76ED6u, 0xA9AEDEC8u, 0x008252BEu},
        {0xF2D98A00u, 0xEC187C37u, 0x6CBC23C1u, 0x4727C109u, 0x881EC0CEu, 0x010479F3u},
    };
    static nk_u32_t const product[nk_cross_limbs_f64_ampere_k] = {0x45218F00u, 0xBC3DBFA3u, 0xB867AAEDu,
                                                                  0xF420B31Du, 0xF9324C1Cu, 0x01115116u};
    for (unsigned index = 0; index < nk_cross_moduli_f64_ampere_k; ++index) {
        nk_u32_t const modulus = values[index];
        moduli->moduli[index] = modulus;
        moduli->reciprocals[index] = (nk_u32_t)(0x100000000ull / modulus);
        nk_u64_t weight = 1;
        for (unsigned digit = 0; digit < 3; ++digit)
            weight = (weight << 21) % modulus, moduli->digit_weights[index][digit] = (nk_u32_t)weight;
        for (unsigned limb = 0; limb < nk_cross_limbs_f64_ampere_k; ++limb)
            moduli->weights[index][limb] = weights[index][limb];
    }
    nk_f64_t product_value = 0;
    for (unsigned limb = nk_cross_limbs_f64_ampere_k; limb-- != 0;)
        moduli->product[limb] = product[limb], product_value = product_value * 0x1p32 + product[limb];
    moduli->product_inverse = 1.0 / product_value;
}

/** Launches the F64 GEMM over rows from @p rows_begin to @p rows_end: the Dot2 tiles for depths
 *  outside the residue range or when its workspace can't be had, else the grids and residues, their
 *  products and reconstruction, and Dot2 over the tiles that flagged, with the workspace freed
 *  after them in the order of @p stream. */
NUMKONG_INLINE nk_status_t nk_cross_launch_f64_ampere_(nk_cross_metric_t metric, nk_diagonal_band_t band,
                                                       unsigned char const *a, unsigned char const *b,
                                                       void const *b_norms, void *c, nk_size_t rows_begin,
                                                       nk_size_t rows_end, nk_size_t column_count, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t b_stride, nk_size_t c_stride,
                                                       nk_stream_t stream) {
    if ((((nk_size_t)a) | a_stride | ((nk_size_t)b) | b_stride) & 15 || (((nk_size_t)c) | c_stride) & 7)
        return nk_misaligned_k;
    if (rows_end <= rows_begin || column_count == 0) return nk_success_k;
    int const symmetric = band.subdiagonals == 0;
    nk_cross_residue_arguments_f64_ampere_t arguments;
    nk_cross_tile_arguments_t *tiles = &arguments.tiles;
    tiles->a = a, tiles->b = b, tiles->c = c;
    tiles->rows_begin = rows_begin, tiles->rows_end = rows_end, tiles->column_count = column_count;
    tiles->depth = depth, tiles->depth_bytes = depth * sizeof(nk_f64_t);
    tiles->a_stride = a_stride, tiles->b_stride = b_stride, tiles->c_stride = c_stride;
    tiles->a_scales = NUMKONG_NULL, tiles->b_scales = NUMKONG_NULL;
    tiles->a_scales_stride = 0, tiles->b_scales_stride = 0;
    tiles->a_tensor_scale = NUMKONG_NULL, tiles->b_tensor_scale = NUMKONG_NULL;
    tiles->depth_slabs = nk_size_divide_round_up_(tiles->depth_bytes, 64);
    tiles->column_tiles = nk_size_divide_round_up_(column_count, nk_cross_tile_simt_k);
    tiles->tiles = nk_size_divide_round_up_(rows_end - rows_begin, nk_cross_tile_simt_k) * tiles->column_tiles;
    tiles->b_norms = b_norms;
    arguments.residues = NUMKONG_NULL, arguments.grid_rows = NUMKONG_NULL, arguments.scratch = NUMKONG_NULL;
    arguments.flags = NUMKONG_NULL, arguments.metric = metric, arguments.band = band;
    void const *fallback_kernel = (void const *)nk_cross_fallback_f64_ampere_kernel_;
    if (depth < nk_cross_depth_floor_f64_ampere_k || depth > nk_cross_depth_limit_f64_ampere_k)
        return nk_launch_resident_cuda_(fallback_kernel, nk_cross_threads_simt_k, 0, 0, tiles->tiles, &arguments,
                                        stream);

    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, nk_cross_slab_bytes_ampere_k);
    nk_size_t const plane_rows = symmetric ? column_count : rows_end + column_count;
    arguments.depth_padded = depth_padded, arguments.plane_bytes = plane_rows * depth_padded;
    arguments.column_plane_row = symmetric ? 0 : rows_end;
    arguments.residue_column_tiles = nk_size_divide_round_up_(column_count, nk_cross_tile_ampere_k);
    arguments.residue_tiles = nk_size_divide_round_up_(rows_end - rows_begin, nk_cross_tile_ampere_k) *
                              arguments.residue_column_tiles;
    nk_cross_moduli_f64_ampere_(&arguments.moduli);

    void const *prepare_kernel = (void const *)nk_cross_prepare_f64_ampere_kernel_;
    void const *residues_kernel = (void const *)nk_cross_residues_f64_ampere_kernel_;
    int caller = 0, device = 0, multiprocessors = 0, per_multiprocessor = 0;
    nk_status_t status = nk_device_enter_cuda_(stream, &caller);
    if (status != nk_success_k) return status;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device) != cudaSuccess ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_multiprocessor, residues_kernel, nk_cross_threads_ampere_k,
                                                      0) != cudaSuccess) {
        nk_device_leave_cuda_(caller);
        return nk_device_code_mismatch_k;
    }
    nk_size_t const resident = (nk_size_t)multiprocessors * (nk_size_t)per_multiprocessor;
    nk_size_t const residue_blocks = resident < arguments.residue_tiles ? resident : arguments.residue_tiles;
    nk_size_t const residues_bytes = nk_size_round_up_to_multiple_(nk_cross_moduli_f64_ampere_k * arguments.plane_bytes,
                                                                   256);
    nk_size_t const rows_bytes = nk_size_round_up_to_multiple_(plane_rows * sizeof(nk_cross_grid_row_f64_ampere_t),
                                                               256);
    nk_size_t const scratch_bytes = residue_blocks * nk_cross_moduli_f64_ampere_k * nk_cross_tile_outputs_f64_ampere_k;
    void *workspace = NUMKONG_NULL;
    cudaError_t const allocated = cudaMallocAsync(
        &workspace, residues_bytes + rows_bytes + scratch_bytes + tiles->tiles, (cudaStream_t)stream);
    if (allocated != cudaSuccess) {
        cudaGetLastError();
        nk_device_leave_cuda_(caller);
        return nk_launch_resident_cuda_(fallback_kernel, nk_cross_threads_simt_k, 0, 0, tiles->tiles, &arguments,
                                        stream);
    }
    arguments.residues = (unsigned char *)workspace;
    arguments.grid_rows = (nk_cross_grid_row_f64_ampere_t *)(arguments.residues + residues_bytes);
    arguments.scratch = arguments.residues + residues_bytes + rows_bytes;
    arguments.flags = arguments.scratch + scratch_bytes;
    status = cudaMemsetAsync(arguments.flags, 0, tiles->tiles, (cudaStream_t)stream) == cudaSuccess
                 ? nk_success_k
                 : nk_device_code_mismatch_k;
    nk_device_leave_cuda_(caller);

    // A's rows take the first plane rows, and B's follow unless they are the same vectors; each
    // warp prepares a row.
    nk_size_t first_plane_row = 0, count = symmetric ? column_count : rows_end;
    nk_size_t blocks = nk_size_divide_round_up_(count, nk_cross_threads_f64_ampere_k / 32);
    void *prepare_arguments[5];
    prepare_arguments[0] = &arguments, prepare_arguments[1] = &a, prepare_arguments[2] = &a_stride;
    prepare_arguments[3] = &count, prepare_arguments[4] = &first_plane_row;
    if (status == nk_success_k)
        status = nk_launch_cuda_(prepare_kernel, blocks < 65535 ? blocks : 65535, nk_cross_threads_f64_ampere_k,
                                 prepare_arguments, 0, stream);
    if (status == nk_success_k && !symmetric) {
        first_plane_row = rows_end, count = column_count;
        blocks = nk_size_divide_round_up_(count, nk_cross_threads_f64_ampere_k / 32);
        prepare_arguments[1] = &b, prepare_arguments[2] = &b_stride;
        status = nk_launch_cuda_(prepare_kernel, blocks < 65535 ? blocks : 65535, nk_cross_threads_f64_ampere_k,
                                 prepare_arguments, 0, stream);
    }
    void *residue_arguments[1];
    residue_arguments[0] = &arguments;
    if (status == nk_success_k)
        status = nk_launch_cuda_(residues_kernel, residue_blocks, nk_cross_threads_ampere_k, residue_arguments, 0,
                                 stream);
    if (status == nk_success_k)
        status = nk_launch_resident_cuda_(fallback_kernel, nk_cross_threads_simt_k, 0, 0, tiles->tiles, &arguments,
                                          stream);
    if (nk_device_enter_cuda_(stream, &caller) == nk_success_k) {
        cudaFreeAsync(workspace, (cudaStream_t)stream);
        nk_device_leave_cuda_(caller);
    }
    return status;
}

/** C = A × Bᵀ of F64 inputs, or its distances, over a B packed by @c nk_dots_pack_f64_ampere. */
NUMKONG_INLINE nk_status_t nk_cross_packed_f64_ampere_(nk_cross_metric_t metric, nk_f64_t const *a,
                                                       void const *b_packed, nk_f64_t *c, nk_size_t rows,
                                                       nk_size_t column_count, nk_size_t depth, nk_size_t a_stride,
                                                       nk_size_t c_stride, nk_stream_t stream) {
    nk_size_t const row_bytes = nk_cross_padded_values_simt_(depth, 2, 1, sizeof(nk_f64_t)) * sizeof(nk_f64_t);
    unsigned char const *b_rows = (unsigned char const *)((nk_cross_packed_buffer_header_t const *)b_packed + 1);
    nk_diagonal_band_t const band = {NUMKONG_SIZE_MAX, NUMKONG_SIZE_MAX};
    return nk_cross_launch_f64_ampere_(metric, band, (unsigned char const *)a, b_rows,
                                       b_rows + column_count * row_bytes, c, 0, rows, column_count, depth, a_stride,
                                       row_bytes, c_stride, stream);
}

/** The Gram matrix of F64 vectors, or its distances, over rows @p rows_begin to @p rows_end. */
NUMKONG_INLINE nk_status_t nk_cross_symmetric_f64_ampere_(nk_cross_metric_t metric, nk_f64_t const *vectors,
                                                          nk_size_t vector_count, nk_size_t depth, nk_size_t stride,
                                                          nk_f64_t *result, nk_size_t result_stride,
                                                          nk_size_t rows_begin, nk_size_t rows_end,
                                                          nk_stream_t stream) {
    nk_diagonal_band_t const band = {0, NUMKONG_SIZE_MAX};
    return nk_cross_launch_f64_ampere_(metric, band, (unsigned char const *)vectors, (unsigned char const *)vectors,
                                       NUMKONG_NULL, result, rows_begin, nk_min_of_two(rows_end, vector_count),
                                       vector_count, depth, stride, stride, result_stride, stream);
}

nk_define_cross_pack_size_simt_(f64, ampere, f64, f64, /*depth_simd_dimensions=*/2, /*dimensions_per_value=*/1)
nk_define_cross_packed_shape_cuda_(f64, ampere)
nk_define_cross_pack_rows_cuda_(f64, ampere, f64, f64, nk_load_b8_simt_, f64, nk_f64_lane_sumsq_cuda_,
                                /*depth_simd_dimensions=*/2, /*dimensions_per_value=*/1)

NUMKONG_API nk_status_t nk_dots_packed_f64_ampere(nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream) {
    return nk_cross_packed_f64_ampere_(nk_cross_metric_dot_k, a, b_packed, c, rows, columns, depth, a_stride, c_stride,
                                       stream);
}

NUMKONG_API nk_status_t nk_dots_symmetric_f64_ampere(nk_f64_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    return nk_cross_symmetric_f64_ampere_(nk_cross_metric_dot_k, vectors, vector_count, depth, stride, result,
                                          result_stride, rows_begin, rows_end, stream);
}

#pragma endregion F64

#pragma region F32

nk_define_cross_pack_size_simt_(f32, ampere, f32, f64, /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)
nk_define_cross_packed_shape_cuda_(f32, ampere)
nk_define_cross_pack_rows_cuda_(f32, ampere, f32, f32, nk_load_b8_simt_, f64, nk_f32_lane_sumsq_cuda_,
                                /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)
nk_define_cross_cuda_(dot, f32, ampere, f32_ampere, f32, f32, f64, /*depth_simd_dimensions=*/4,
                      /*dimensions_per_value=*/1)

#pragma endregion F32

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

#pragma region NVFP4

nk_define_cross_pack_size_simt_(nvfp4, ampere, e2m1x2, cross_wide_sum, /*depth_simd_dimensions=*/32,
                                /*dimensions_per_value=*/2)
nk_define_cross_packed_shape_cuda_(nvfp4, ampere)
nk_define_cross_pack_rows_cuda_(nvfp4, ampere, e2m1x2, e2m1x2, nk_load_b8_simt_, cross_wide_sum,
                                nk_cross_scaled_pack_norm_nvfp4_simt_, /*depth_simd_dimensions=*/32,
                                /*dimensions_per_value=*/2)
nk_define_cross_cuda_(dot, nvfp4, ampere, nvfp4_ampere, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)

#pragma endregion NVFP4

#pragma region MXFP4

nk_define_cross_pack_size_simt_(mxfp4, ampere, e2m1x2, cross_wide_sum, /*depth_simd_dimensions=*/32,
                                /*dimensions_per_value=*/2)
nk_define_cross_packed_shape_cuda_(mxfp4, ampere)
nk_define_cross_pack_rows_cuda_(mxfp4, ampere, e2m1x2, e2m1x2, nk_load_b8_simt_, cross_wide_sum,
                                nk_cross_scaled_pack_norm_mxfp4_simt_, /*depth_simd_dimensions=*/32,
                                /*dimensions_per_value=*/2)
nk_define_cross_cuda_(dot, mxfp4, ampere, mxfp4_ampere, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)

#pragma endregion MXFP4

#pragma region MXFP8E4M3

nk_define_cross_pack_size_simt_(mxfp8e4m3, ampere, e4m3, cross_wide_sum, /*depth_simd_dimensions=*/16,
                                /*dimensions_per_value=*/1)
nk_define_cross_packed_shape_cuda_(mxfp8e4m3, ampere)
nk_define_cross_pack_rows_cuda_(mxfp8e4m3, ampere, e4m3, e4m3, nk_load_b8_simt_, cross_wide_sum,
                                nk_cross_scaled_pack_norm_mxfp8e4m3_simt_, /*depth_simd_dimensions=*/16,
                                /*dimensions_per_value=*/1)
nk_define_cross_cuda_(dot, mxfp8e4m3, ampere, mxfp8e4m3_ampere, e4m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion MXFP8E4M3

#pragma region MXFP8E5M2

nk_define_cross_pack_size_simt_(mxfp8e5m2, ampere, e5m2, cross_wide_sum, /*depth_simd_dimensions=*/16,
                                /*dimensions_per_value=*/1)
nk_define_cross_packed_shape_cuda_(mxfp8e5m2, ampere)
nk_define_cross_pack_rows_cuda_(mxfp8e5m2, ampere, e5m2, e5m2, nk_load_b8_simt_, cross_wide_sum,
                                nk_cross_scaled_pack_norm_mxfp8e5m2_simt_, /*depth_simd_dimensions=*/16,
                                /*dimensions_per_value=*/1)
nk_define_cross_cuda_(dot, mxfp8e5m2, ampere, mxfp8e5m2_ampere, e5m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)

#pragma endregion MXFP8E5M2

#endif // NUMKONG_TARGET_AMPERE

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_AMPERE_
#endif // NUMKONG_ARCH_CUDA_
#endif // NUMKONG_DOTS_AMPERE_CUH
