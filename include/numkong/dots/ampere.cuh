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
 *  sub-slab is multiplied. F64 and F32 run a CUDA-core FMA tile at Dot2 and F64 precision; Float8
 *  and E3M2 widen exactly into F16; E2M3, E2M1, I4 and U4 widen into I8 for exact integer MMA, with
 *  one output multiply undoing each widening's power of two.
 */
#ifndef NK_DOTS_AMPERE_CUH
#define NK_DOTS_AMPERE_CUH

#if NK_TARGET_AMPERE

#include "numkong/dots/serial.h"

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
    nk_cross_fma_threads_ampere_k = 256,
    nk_cross_fma_tile_ampere_k = 64,
    nk_cross_fma_slab_ampere_k = 16,
    nk_cross_fma_loads_ampere_k = nk_cross_fma_tile_ampere_k * nk_cross_fma_slab_ampere_k /
                                  nk_cross_fma_threads_ampere_k,
    nk_cross_pack_warps_ampere_k = 8,
};

/** Folds one warp's fragments of a 32-byte sub-slab, A as 4 row tiles of 16 and B as 8 column
 *  tiles of 8. */
typedef void (*nk_cross_multiply_ampere_t)(nk_fui32_t accumulators[4][8][4], nk_u32_t const a[4][4],
                                           nk_u32_t const b[8][2]);

/** Splits one fragment register into two narrower-typed ones: 4 codes into F16 pairs, or 8 nibbles
 *  into i8 quads. */
typedef void (*nk_cross_widen_ampere_t)(nk_u32_t codes, nk_u32_t *low, nk_u32_t *high);

/** One 16 × 8 tensor-core step on registers of the type a widening produced. */
typedef void (*nk_cross_mma_ampere_t)(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t b_first,
                                      nk_u32_t b_second);

/** Reads element @p index of one F32 or F64 row as F64, the FMA tile's working type. */
typedef nk_f64_t (*nk_cross_load_f64_ampere_t)(unsigned char const *row, nk_size_t index);

/** How the FMA tile accumulates, matching the serial backends. */
typedef enum {
    nk_cross_accumulation_f64_k,  // one F64 FMA per product, exact for F32 inputs
    nk_cross_accumulation_dot2_k, // Ogita-Rump-Oishi Dot2: TwoProd and TwoSum, about twice the F64 precision
} nk_cross_accumulation_t;

/** Which outputs a tile writes. */
typedef enum {
    nk_cross_triangle_full_k,  // every output, for `packed`
    nk_cross_triangle_upper_k, // the upper triangle with its diagonal, for `symmetric`
} nk_cross_triangle_t;

/** What the accumulators hold and how they reach the output. */
typedef enum {
    nk_cross_epilogue_f32_k,        // F32 sums, stored times the output scale
    nk_cross_epilogue_i32_k,        // integer sums, stored as they are
    nk_cross_epilogue_i32_to_f32_k, // integer sums of scaled codes, converted and stored times the output scale
} nk_cross_epilogue_t;

/** What a tile turns each dot product into. */
typedef enum {
    nk_cross_metric_dot_k,       // the dot product itself
    nk_cross_metric_angular_k,   // 1 − dot / (‖a‖ ‖b‖), clamped at 0
    nk_cross_metric_euclidean_k, // √(‖a‖² + ‖b‖² − 2 · dot), clamped at 0
} nk_cross_metric_t;

/** How squared norms are stored, which also fixes the precision a metric is computed in. */
typedef enum {
    nk_cross_norm_f32_k, // F32 in true units, the metric in F32
    nk_cross_norm_f64_k, // F64, the metric in F64
    nk_cross_norm_i32_k, // wrapping U32 sums read as I32, the metric in F32
    nk_cross_norm_u32_k, // wrapping U32 sums, the metric in F32
} nk_cross_norm_t;

/** Adds the squares of 16 staged bytes: exact integer codes into @p integer_sum, the others
 *  into @p real_sum. */
typedef void (*nk_cross_norm_update_ampere_t)(nk_u32_t const words[4], nk_u32_t *integer_sum, nk_f32_t *real_sum);

/** Everything one launch shares, passed by value as the kernels' only argument. */
typedef struct {
    unsigned char const *a; // row-major A, or the vectors for `symmetric`
    unsigned char const *b; // packed B rows past the header, or the vectors again for `symmetric`
    void *c;                // row-major output, indexed by absolute row
    nk_size_t row_start;    // first output row
    nk_size_t row_end;      // one past the last output row
    nk_size_t column_count; // output columns, the rows of B
    nk_size_t depth_bytes;  // bytes of depth per row, past which A and B read as zeros
    nk_size_t depth;        // elements of depth per row, which the FMA tile counts in
    nk_size_t a_stride;     // bytes between rows of A
    nk_size_t b_stride;     // bytes between rows of B
    nk_size_t c_stride;     // bytes between rows of C
    nk_size_t depth_slabs;  // 64-byte slabs of depth, which the tensor tile counts in
    nk_size_t column_tiles; // output tiles per row of tiles
    nk_size_t tiles;        // output tiles in all, which the blocks walk with a stride of the grid
    void const *b_norms;    // column norms past the packed rows, or null where the tile needs none from memory
} nk_cross_tile_arguments_ampere_t;

#pragma endregion Configuration

#pragma region Instructions

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

NK_HELPER_DEVICE_INLINE nk_u32_t nk_shared_address_ampere_(void const *pointer) {
    return (nk_u32_t)__cvta_generic_to_shared(pointer);
}

/* Copies 16 bytes, zero-filling past @p valid_bytes, so tails need no second path. */
NK_HELPER_DEVICE_INLINE void nk_copy_b128_async_ampere_(nk_u32_t shared, void const *global, nk_u32_t valid_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(shared), "l"(global), "r"(valid_bytes));
}

NK_HELPER_DEVICE_INLINE void nk_commit_async_ampere_(void) { asm volatile("cp.async.commit_group;\n" ::); }

/*  Waits until at most @p pending committed groups are in flight, capped at 2, since PTX
 *  takes an immediate. */
NK_HELPER_DEVICE_INLINE void nk_wait_async_ampere_(unsigned pending) {
    switch (pending) {
    case 0: asm volatile("cp.async.wait_group 0;\n" ::); break;
    case 1: asm volatile("cp.async.wait_group 1;\n" ::); break;
    default: asm volatile("cp.async.wait_group 2;\n" ::); break;
    }
}

NK_HELPER_DEVICE_INLINE void nk_load_matrices_x4_ampere_(nk_u32_t shared, nk_u32_t fragments[4]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(fragments[0]), "=r"(fragments[1]), "=r"(fragments[2]), "=r"(fragments[3])
                 : "r"(shared));
}

NK_HELPER_DEVICE_INLINE void nk_load_matrices_x4_transposed_ampere_(nk_u32_t shared, nk_u32_t fragments[4]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(fragments[0]), "=r"(fragments[1]), "=r"(fragments[2]), "=r"(fragments[3])
                 : "r"(shared));
}

NK_HELPER_DEVICE_INLINE void nk_mma_bf16_ampere_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t b_first,
                                                 nk_u32_t b_second) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " //
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(accumulator[0].f), "+f"(accumulator[1].f), "+f"(accumulator[2].f), "+f"(accumulator[3].f)
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b_first), "r"(b_second));
}

NK_HELPER_DEVICE_INLINE void nk_mma_f16_ampere_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t b_first,
                                                nk_u32_t b_second) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " //
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(accumulator[0].f), "+f"(accumulator[1].f), "+f"(accumulator[2].f), "+f"(accumulator[3].f)
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b_first), "r"(b_second));
}

NK_HELPER_DEVICE_INLINE void nk_mma_i8_ampere_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t b_first,
                                               nk_u32_t b_second) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " //
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+r"(accumulator[0].u), "+r"(accumulator[1].u), "+r"(accumulator[2].u), "+r"(accumulator[3].u)
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b_first), "r"(b_second));
}

NK_HELPER_DEVICE_INLINE void nk_mma_u8_ampere_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t b_first,
                                               nk_u32_t b_second) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 " //
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+r"(accumulator[0].u), "+r"(accumulator[1].u), "+r"(accumulator[2].u), "+r"(accumulator[3].u)
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b_first), "r"(b_second));
}

NK_HELPER_DEVICE_INLINE void nk_mma_u8i8_ampere_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t b_first,
                                                 nk_u32_t b_second) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 " //
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+r"(accumulator[0].u), "+r"(accumulator[1].u), "+r"(accumulator[2].u), "+r"(accumulator[3].u)
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b_first), "r"(b_second));
}

NK_HELPER_DEVICE_INLINE void nk_mma_i4_ampere_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t b_first,
                                               nk_u32_t b_second) {
    asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 " //
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+r"(accumulator[0].u), "+r"(accumulator[1].u), "+r"(accumulator[2].u), "+r"(accumulator[3].u)
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b_first), "r"(b_second));
}

NK_HELPER_DEVICE_INLINE void nk_mma_u4_ampere_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t b_first,
                                               nk_u32_t b_second) {
    asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.u4.u4.s32 " //
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+r"(accumulator[0].u), "+r"(accumulator[1].u), "+r"(accumulator[2].u), "+r"(accumulator[3].u)
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b_first), "r"(b_second));
}

#else

/*  Device passes older than 8.0 and the host pass get trapping bodies, so a cubin picked for the
 *  wrong device fails loudly rather than returning zeros. */
NK_HELPER_DEVICE_INLINE nk_u32_t nk_shared_address_ampere_(void const *pointer) {
    __trap();
    return 0;
}
NK_HELPER_DEVICE_INLINE void nk_copy_b128_async_ampere_(nk_u32_t shared, void const *global, nk_u32_t valid_bytes) {
    __trap();
}
NK_HELPER_DEVICE_INLINE void nk_commit_async_ampere_(void) { __trap(); }
NK_HELPER_DEVICE_INLINE void nk_wait_async_ampere_(unsigned pending) { __trap(); }
NK_HELPER_DEVICE_INLINE void nk_load_matrices_x4_ampere_(nk_u32_t shared, nk_u32_t fragments[4]) { __trap(); }
NK_HELPER_DEVICE_INLINE void nk_load_matrices_x4_transposed_ampere_(nk_u32_t shared, nk_u32_t fragments[4]) {
    __trap();
}
NK_HELPER_DEVICE_INLINE void nk_mma_bf16_ampere_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t b_first,
                                                 nk_u32_t b_second) {
    __trap();
}
NK_HELPER_DEVICE_INLINE void nk_mma_f16_ampere_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t b_first,
                                                nk_u32_t b_second) {
    __trap();
}
NK_HELPER_DEVICE_INLINE void nk_mma_i8_ampere_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t b_first,
                                               nk_u32_t b_second) {
    __trap();
}
NK_HELPER_DEVICE_INLINE void nk_mma_u8_ampere_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t b_first,
                                               nk_u32_t b_second) {
    __trap();
}
NK_HELPER_DEVICE_INLINE void nk_mma_u8i8_ampere_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t b_first,
                                                 nk_u32_t b_second) {
    __trap();
}
NK_HELPER_DEVICE_INLINE void nk_mma_i4_ampere_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t b_first,
                                               nk_u32_t b_second) {
    __trap();
}
NK_HELPER_DEVICE_INLINE void nk_mma_u4_ampere_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t b_first,
                                               nk_u32_t b_second) {
    __trap();
}

#endif // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

#pragma endregion Instructions

/*  Four codes become two pairs of F16 patterns scaled by a power of two: each first lands in the
 *  high byte of a half, `code << 8`, and then keeps only its fields. */
#pragma region Conversions

NK_HELPER_DEVICE_INLINE void nk_e5m2x4_to_f16x4_ampere_(nk_u32_t codes, nk_u32_t *low, nk_u32_t *high) {
    *low = __byte_perm(codes, 0, 0x1404), *high = __byte_perm(codes, 0, 0x3424);
}

NK_HELPER_DEVICE_INLINE void nk_e4m3x4_to_f16x4_ampere_(nk_u32_t codes, nk_u32_t *low, nk_u32_t *high) {
    nk_u32_t const low_halves = __byte_perm(codes, 0, 0x1404), high_halves = __byte_perm(codes, 0, 0x3424);
    *low = ((low_halves >> 1) & 0x3F803F80u) | (low_halves & 0x80008000u);
    *high = ((high_halves >> 1) & 0x3F803F80u) | (high_halves & 0x80008000u);
}

NK_HELPER_DEVICE_INLINE void nk_e3m2x4_to_f16x4_ampere_(nk_u32_t codes, nk_u32_t *low, nk_u32_t *high) {
    nk_u32_t const low_halves = __byte_perm(codes, 0, 0x1404), high_halves = __byte_perm(codes, 0, 0x3424);
    *low = (low_halves & 0x1F001F00u) | ((low_halves << 2) & 0x80008000u);
    *high = (high_halves & 0x1F001F00u) | ((high_halves << 2) & 0x80008000u);
}

/** Negates the bytes of @p magnitudes under @p sign_mask, each magnitude below 128 so no borrow
 *  crosses a byte. */
NK_HELPER_DEVICE_INLINE nk_u32_t nk_i8x4_apply_signs_ampere_(nk_u32_t magnitudes, nk_u32_t sign_mask) {
    nk_u32_t const negated = (0x80808080u - magnitudes) ^ 0x80808080u;
    return (magnitudes & ~sign_mask) | (negated & sign_mask);
}

/** Four E2M3 magnitudes times 8 as u8: `e = 0` gives @c m, otherwise `(8 + m) << (e - 1)`,
 *  at most 60. */
NK_HELPER_DEVICE_INLINE nk_u32_t nk_e2m3x4_to_u8x4_magnitudes_ampere_(nk_u32_t codes) {
    nk_u32_t const exponent_low = (codes >> 3) & 0x01010101u, exponent_high = (codes >> 4) & 0x01010101u;
    nk_u32_t const significand = (codes & 0x07070707u) | ((exponent_low | exponent_high) << 3);
    nk_u32_t const doubled_mask = exponent_high * 0xFFu, quadrupled_mask = (exponent_high & exponent_low) * 0xFFu;
    return significand + (significand & doubled_mask) + ((significand << 1) & quadrupled_mask);
}

/** Four E2M3 codes times 8 as i8. */
NK_HELPER_DEVICE_INLINE nk_u32_t nk_e2m3x4_to_i8x4_ampere_(nk_u32_t codes) {
    return nk_i8x4_apply_signs_ampere_(nk_e2m3x4_to_u8x4_magnitudes_ampere_(codes),
                                       ((codes >> 5) & 0x01010101u) * 0xFFu);
}

/** Four E2M1 nibbles of @p codes times 2 as i8 through one table lookup; bit 3 of each nibble
 *  is its sign. */
NK_HELPER_DEVICE_INLINE nk_u32_t nk_e2m1x4_to_i8x4_ampere_(nk_u32_t codes) {
    nk_u32_t const magnitudes = __byte_perm(0x03020100u, 0x0C080604u, codes & 0x7777u);
    nk_u32_t replicated;
    // With every table byte negative, a nibble whose bit 3 is set replicates that sign as 0xFF, others read 0x80.
    asm("prmt.b32 %0, %1, %1, %2;" : "=r"(replicated) : "r"(0x80808080u), "r"(codes & 0xFFFFu));
    nk_u32_t const signs = replicated ^ 0x80808080u;
    return nk_i8x4_apply_signs_ampere_(magnitudes, signs | (signs << 1));
}

/** Eight E2M1 nibbles become two registers of four scaled i8 each. */
NK_HELPER_DEVICE_INLINE void nk_e2m1x8_to_i8x8_ampere_(nk_u32_t codes, nk_u32_t *low, nk_u32_t *high) {
    *low = nk_e2m1x4_to_i8x4_ampere_(codes), *high = nk_e2m1x4_to_i8x4_ampere_(codes >> 16);
}

/** Eight i4 nibbles become two registers of four sign-extended i8, each `(v ^ 8) - 8`
 *  without cross-byte borrows. */
NK_HELPER_DEVICE_INLINE void nk_i4x8_to_i8x8_ampere_(nk_u32_t codes, nk_u32_t *low, nk_u32_t *high) {
    nk_u32_t const low_nibbles = codes & 0x0F0F0F0Fu, high_nibbles = (codes >> 4) & 0x0F0F0F0Fu;
    *low = (((low_nibbles ^ 0x08080808u) | 0x80808080u) - 0x08080808u) ^ 0x80808080u;
    *high = (((high_nibbles ^ 0x08080808u) | 0x80808080u) - 0x08080808u) ^ 0x80808080u;
}

/** Eight u4 nibbles become two registers of four u8. */
NK_HELPER_DEVICE_INLINE void nk_u4x8_to_u8x8_ampere_(nk_u32_t codes, nk_u32_t *low, nk_u32_t *high) {
    *low = codes & 0x0F0F0F0Fu, *high = (codes >> 4) & 0x0F0F0F0Fu;
}

/** Keeps a packed byte as it is. */
NK_HELPER_DEVICE_INLINE unsigned char nk_load_b8_ampere_(unsigned char value) { return value; }

/** Packs one E2M3 code as its value times 8 in i8, the form its multiply consumes. */
NK_HELPER_DEVICE_INLINE unsigned char nk_load_e2m3_to_i8_ampere_(unsigned char code) {
    return (unsigned char)nk_e2m3x4_to_i8x4_ampere_(code);
}

#pragma endregion Conversions

#pragma region Norms

NK_HELPER_DEVICE_INLINE nk_f64_t nk_load_f64_ampere_(unsigned char const *row, nk_size_t index) {
    return ((nk_f64_t const *)row)[index];
}

NK_HELPER_DEVICE_INLINE nk_f64_t nk_load_f32_to_f64_ampere_(unsigned char const *row, nk_size_t index) {
    return (nk_f64_t)((nk_f32_t const *)row)[index];
}

NK_HELPER_DEVICE_INLINE nk_f64_t nk_dots_reduce_sumsq_f64_ampere_(unsigned char const *row, nk_size_t depth,
                                                                  unsigned lane) {
    nk_f64_t sum = 0;
    for (nk_size_t index = lane; index < depth; index += 32) {
        unsigned long long bits = 0;
        for (unsigned byte = 0; byte < 8; ++byte) bits |= (unsigned long long)row[index * 8 + byte] << (byte * 8);
        nk_f64_t const value = __longlong_as_double((long long)bits);
        sum = __fma_rn(value, value, sum);
    }
    return sum;
}

NK_HELPER_DEVICE_INLINE nk_f64_t nk_dots_reduce_sumsq_f32_ampere_(unsigned char const *row, nk_size_t depth,
                                                                  unsigned lane) {
    nk_f64_t sum = 0;
    for (nk_size_t index = lane; index < depth; index += 32) {
        nk_u32_t bits = 0;
        for (unsigned byte = 0; byte < 4; ++byte) bits |= (nk_u32_t)row[index * 4 + byte] << (byte * 8);
        nk_f64_t const value = (nk_f64_t)__uint_as_float(bits);
        sum = __fma_rn(value, value, sum);
    }
    return sum;
}

/*  Each returns one lane's share of a column's sum of squares, over indices @c lane, `lane + 32`,
 *  … below @c depth; the pack adds the 32 shares. Rows are read bytewise, since @c b_stride need
 *  not be element-aligned. */

NK_HELPER_DEVICE_INLINE nk_f32_t nk_dots_reduce_sumsq_bf16_ampere_(unsigned char const *row, nk_size_t depth,
                                                                   unsigned lane) {
    nk_f32_t sum = 0;
    for (nk_size_t index = lane; index < depth; index += 32) {
        nk_f32_t const value = __uint_as_float(((nk_u32_t)row[index * 2] | ((nk_u32_t)row[index * 2 + 1] << 8)) << 16);
        sum += value * value;
    }
    return sum;
}

NK_HELPER_DEVICE_INLINE nk_f32_t nk_dots_reduce_sumsq_f16_ampere_(unsigned char const *row, nk_size_t depth,
                                                                  unsigned lane) {
    nk_f32_t sum = 0;
    for (nk_size_t index = lane; index < depth; index += 32) {
        nk_f32_t const value = __half2float(
            __ushort_as_half((unsigned short)(row[index * 2] | (row[index * 2 + 1] << 8))));
        sum += value * value;
    }
    return sum;
}

NK_HELPER_DEVICE_INLINE nk_f32_t nk_dots_reduce_sumsq_e5m2_ampere_(unsigned char const *row, nk_size_t depth,
                                                                   unsigned lane) {
    nk_f32_t sum = 0;
    for (nk_size_t index = lane; index < depth; index += 32) {
        nk_f32_t const value = __half2float(__ushort_as_half((unsigned short)(row[index] << 8)));
        sum += value * value;
    }
    return sum;
}

NK_HELPER_DEVICE_INLINE nk_f32_t nk_dots_reduce_sumsq_e4m3_ampere_(unsigned char const *row, nk_size_t depth,
                                                                   unsigned lane) {
    nk_f32_t sum = 0;
    for (nk_size_t index = lane; index < depth; index += 32) {
        unsigned const code = row[index];
        nk_f32_t const value =
            __half2float(__ushort_as_half((unsigned short)(((code & 0x7Fu) << 7) | ((code & 0x80u) << 8)))) * 256.0f;
        sum += value * value;
    }
    return sum;
}

NK_HELPER_DEVICE_INLINE nk_f32_t nk_dots_reduce_sumsq_e3m2_ampere_(unsigned char const *row, nk_size_t depth,
                                                                   unsigned lane) {
    nk_f32_t sum = 0;
    for (nk_size_t index = lane; index < depth; index += 32) {
        unsigned const code = row[index];
        nk_f32_t const value =
            __half2float(__ushort_as_half((unsigned short)(((code & 0x1Fu) << 8) | ((code & 0x20u) << 10)))) * 4096.0f;
        sum += value * value;
    }
    return sum;
}

NK_HELPER_DEVICE_INLINE nk_f32_t nk_dots_reduce_sumsq_e2m3_ampere_(unsigned char const *row, nk_size_t depth,
                                                                   unsigned lane) {
    nk_f32_t sum = 0;
    for (nk_size_t index = lane; index < depth; index += 32) {
        nk_f32_t const value = (nk_f32_t)(signed char)nk_e2m3x4_to_i8x4_ampere_(row[index]) * 0.125f;
        sum += value * value;
    }
    return sum;
}

/*  Nibble types hold element `2 × i` in the high nibble of byte @c i and element `2 × i + 1` in
 *  the low one. */

NK_HELPER_DEVICE_INLINE nk_f32_t nk_dots_reduce_sumsq_e2m1_ampere_(unsigned char const *row, nk_size_t depth,
                                                                   unsigned lane) {
    nk_f32_t sum = 0;
    for (nk_size_t index = lane; index < depth; index += 32) {
        unsigned const nibble = (index & 1) ? (row[index / 2] & 0x0Fu) : (row[index / 2] >> 4);
        nk_f32_t const value = (nk_f32_t)(signed char)nk_e2m1x4_to_i8x4_ampere_(nibble) * 0.5f;
        sum += value * value;
    }
    return sum;
}

NK_HELPER_DEVICE_INLINE nk_u32_t nk_dots_reduce_sumsq_i8_ampere_(unsigned char const *row, nk_size_t depth,
                                                                 unsigned lane) {
    nk_u32_t sum = 0;
    for (nk_size_t index = lane; index < depth; index += 32) {
        nk_i32_t const value = (signed char)row[index];
        sum += (nk_u32_t)(value * value);
    }
    return sum;
}

NK_HELPER_DEVICE_INLINE nk_u32_t nk_dots_reduce_sumsq_i4_ampere_(unsigned char const *row, nk_size_t depth,
                                                                 unsigned lane) {
    nk_u32_t sum = 0;
    for (nk_size_t index = lane; index < depth; index += 32) {
        unsigned const nibble = (index & 1) ? (row[index / 2] & 0x0Fu) : (row[index / 2] >> 4);
        nk_i32_t const value = (nk_i32_t)(nibble ^ 8u) - 8;
        sum += (nk_u32_t)(value * value);
    }
    return sum;
}

NK_HELPER_DEVICE_INLINE nk_u32_t nk_dots_reduce_sumsq_u8_ampere_(unsigned char const *row, nk_size_t depth,
                                                                 unsigned lane) {
    nk_u32_t sum = 0;
    for (nk_size_t index = lane; index < depth; index += 32) sum += (nk_u32_t)row[index] * row[index];
    return sum;
}

NK_HELPER_DEVICE_INLINE nk_u32_t nk_dots_reduce_sumsq_u4_ampere_(unsigned char const *row, nk_size_t depth,
                                                                 unsigned lane) {
    nk_u32_t sum = 0;
    for (nk_size_t index = lane; index < depth; index += 32) {
        nk_u32_t const value = (index & 1) ? (row[index / 2] & 0x0Fu) : (row[index / 2] >> 4);
        sum += value * value;
    }
    return sum;
}

/*  Each adds the squares of one 16-byte chunk of a staged row, in any order, since a sum of squares
 *  has none. Float8 and Float6 square their F16 widenings, and E2M3 and E2M1 their scaled i8 ones,
 *  so the tile scales them back. */

NK_HELPER_DEVICE_INLINE void nk_f16x2_norm_update_ampere_(nk_u32_t halves, nk_f32_t *real_sum) {
    nk_f32_t const low = __half2float(__ushort_as_half((unsigned short)(halves & 0xFFFFu)));
    nk_f32_t const high = __half2float(__ushort_as_half((unsigned short)(halves >> 16)));
    *real_sum = __fmaf_rn(high, high, __fmaf_rn(low, low, *real_sum));
}

NK_HELPER_DEVICE_INLINE void nk_bf16_norm_update_ampere_(nk_u32_t const words[4], nk_u32_t *integer_sum,
                                                         nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) {
        nk_f32_t const low = __uint_as_float(words[word] << 16), high = __uint_as_float(words[word] & 0xFFFF0000u);
        *real_sum = __fmaf_rn(high, high, __fmaf_rn(low, low, *real_sum));
    }
}

NK_HELPER_DEVICE_INLINE void nk_f16_norm_update_ampere_(nk_u32_t const words[4], nk_u32_t *integer_sum,
                                                        nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) nk_f16x2_norm_update_ampere_(words[word], real_sum);
}

NK_HELPER_DEVICE_INLINE void nk_widened_f16_norm_update_ampere_(nk_cross_widen_ampere_t widen, nk_u32_t const words[4],
                                                                nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) {
        nk_u32_t low, high;
        widen(words[word], &low, &high);
        nk_f16x2_norm_update_ampere_(low, real_sum);
        nk_f16x2_norm_update_ampere_(high, real_sum);
    }
}

NK_HELPER_DEVICE_INLINE void nk_e5m2_norm_update_ampere_(nk_u32_t const words[4], nk_u32_t *integer_sum,
                                                         nk_f32_t *real_sum) {
    nk_widened_f16_norm_update_ampere_(nk_e5m2x4_to_f16x4_ampere_, words, real_sum);
}

NK_HELPER_DEVICE_INLINE void nk_e4m3_norm_update_ampere_(nk_u32_t const words[4], nk_u32_t *integer_sum,
                                                         nk_f32_t *real_sum) {
    nk_widened_f16_norm_update_ampere_(nk_e4m3x4_to_f16x4_ampere_, words, real_sum);
}

NK_HELPER_DEVICE_INLINE void nk_e3m2_norm_update_ampere_(nk_u32_t const words[4], nk_u32_t *integer_sum,
                                                         nk_f32_t *real_sum) {
    nk_widened_f16_norm_update_ampere_(nk_e3m2x4_to_f16x4_ampere_, words, real_sum);
}

NK_HELPER_DEVICE_INLINE void nk_e2m3_norm_update_ampere_(nk_u32_t const words[4], nk_u32_t *integer_sum,
                                                         nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) {
        nk_u32_t const magnitudes = nk_e2m3x4_to_u8x4_magnitudes_ampere_(words[word]);
        *integer_sum = __dp4a(magnitudes, magnitudes, *integer_sum);
    }
}

NK_HELPER_DEVICE_INLINE void nk_e2m1_norm_update_ampere_(nk_u32_t const words[4], nk_u32_t *integer_sum,
                                                         nk_f32_t *real_sum) {
    // Squares of twice each magnitude, {0, 1, 4, 9, 16, 36, 64, 144}, looked up 4 nibbles at a time.
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) {
        nk_u32_t const low = __byte_perm(0x09040100u, 0x90402410u, words[word] & 0x7777u);
        nk_u32_t const high = __byte_perm(0x09040100u, 0x90402410u, (words[word] >> 16) & 0x7777u);
        *integer_sum = __dp4a(low, 0x01010101u, *integer_sum);
        *integer_sum = __dp4a(high, 0x01010101u, *integer_sum);
    }
}

NK_HELPER_DEVICE_INLINE void nk_i8_norm_update_ampere_(nk_u32_t const words[4], nk_u32_t *integer_sum,
                                                       nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word)
        *integer_sum = (nk_u32_t)__dp4a((int)words[word], (int)words[word], (int)*integer_sum);
}

NK_HELPER_DEVICE_INLINE void nk_i4_norm_update_ampere_(nk_u32_t const words[4], nk_u32_t *integer_sum,
                                                       nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) {
        nk_u32_t low, high;
        nk_i4x8_to_i8x8_ampere_(words[word], &low, &high);
        *integer_sum = (nk_u32_t)__dp4a((int)low, (int)low, (int)*integer_sum);
        *integer_sum = (nk_u32_t)__dp4a((int)high, (int)high, (int)*integer_sum);
    }
}

NK_HELPER_DEVICE_INLINE void nk_u8_norm_update_ampere_(nk_u32_t const words[4], nk_u32_t *integer_sum,
                                                       nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) *integer_sum = __dp4a(words[word], words[word], *integer_sum);
}

NK_HELPER_DEVICE_INLINE void nk_u4_norm_update_ampere_(nk_u32_t const words[4], nk_u32_t *integer_sum,
                                                       nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) {
        nk_u32_t low, high;
        nk_u4x8_to_u8x8_ampere_(words[word], &low, &high);
        *integer_sum = __dp4a(low, low, *integer_sum);
        *integer_sum = __dp4a(high, high, *integer_sum);
    }
}

#pragma endregion Norms

#pragma region Metrics

/** An integer dot product as F32, signed unless its norms are unsigned, the way the serial
 *  metrics read it. */
NK_HELPER_DEVICE_INLINE nk_f32_t nk_cross_dot_to_f32_ampere_(nk_fui32_t sum, nk_cross_epilogue_t epilogue,
                                                             nk_cross_norm_t norm, nk_f32_t output_scale) {
    if (epilogue == nk_cross_epilogue_f32_k) return sum.f * output_scale;
    if (epilogue == nk_cross_epilogue_i32_to_f32_k) return (nk_f32_t)sum.i * output_scale;
    return norm == nk_cross_norm_u32_k ? (nk_f32_t)sum.u : (nk_f32_t)sum.i;
}

NK_HELPER_DEVICE_INLINE nk_f32_t nk_cross_norm_to_f32_ampere_(nk_fui32_t bits, nk_cross_norm_t norm) {
    if (norm == nk_cross_norm_i32_k) return (nk_f32_t)bits.i;
    if (norm == nk_cross_norm_u32_k) return (nk_f32_t)bits.u;
    return bits.f;
}

/** A thread's accumulated squared norm in the 32 bits the epilogue reads: integers as summed,
 *  floats in true units. */
NK_HELPER_DEVICE_INLINE nk_fui32_t nk_cross_norm_finalize_ampere_(nk_cross_norm_t norm, nk_u32_t integer_sum,
                                                                  nk_f32_t real_sum, nk_f32_t norm_scale) {
    nk_fui32_t result;
    if (norm == nk_cross_norm_f32_k) result.f = ((nk_f32_t)integer_sum + real_sum) * norm_scale;
    else result.u = integer_sum;
    return result;
}

/** 1 − dot / (‖a‖ ‖b‖) clamped at 0; with a zero norm, 0 when the dot is 0 and 1 otherwise. */
NK_HELPER_DEVICE_INLINE nk_f32_t nk_f32_angular_ampere_(nk_f32_t dot, nk_f32_t row_norm, nk_f32_t column_norm) {
    if (!(row_norm > 0 && column_norm > 0)) return dot == 0 ? 0.0f : 1.0f;
    nk_f32_t const unclipped = 1.0f - dot * (rsqrtf(row_norm) * rsqrtf(column_norm));
    return unclipped > 0 ? unclipped : 0.0f;
}

NK_HELPER_DEVICE_INLINE nk_f64_t nk_f64_angular_ampere_(nk_f64_t dot, nk_f64_t row_norm, nk_f64_t column_norm) {
    if (!(row_norm > 0 && column_norm > 0)) return dot == 0 ? 0.0 : 1.0;
    nk_f64_t const unclipped = 1.0 - dot * (rsqrt(row_norm) * rsqrt(column_norm));
    return unclipped > 0 ? unclipped : 0.0;
}

/** √(‖a‖² + ‖b‖² − 2 · dot), with a negative radicand from rounding clamped to 0. */
NK_HELPER_DEVICE_INLINE nk_f32_t nk_f32_euclidean_ampere_(nk_f32_t dot, nk_f32_t row_norm, nk_f32_t column_norm) {
    nk_f32_t const squared = row_norm + column_norm - 2.0f * dot;
    return squared > 0 ? sqrtf(squared) : 0.0f;
}

NK_HELPER_DEVICE_INLINE nk_f64_t nk_f64_euclidean_ampere_(nk_f64_t dot, nk_f64_t row_norm, nk_f64_t column_norm) {
    nk_f64_t const squared = row_norm + column_norm - 2.0 * dot;
    return squared > 0 ? sqrt(squared) : 0.0;
}

#pragma endregion Metrics

#pragma region Tile

/** Issues one 64-byte depth slab of A and B into a pipeline stage, 16 bytes per copy, 8
 *  copies per thread. */
NK_HELPER_DEVICE_INLINE void nk_cross_stage_ampere_(unsigned char *a_stage, unsigned char *b_stage,
                                                    nk_cross_tile_arguments_ampere_t const *arguments,
                                                    nk_size_t first_row, nk_size_t first_column,
                                                    nk_size_t slab_offset) {
#pragma unroll
    for (unsigned chunk = threadIdx.x; chunk < nk_cross_stage_bytes_ampere_k / 16; chunk += nk_cross_threads_ampere_k) {
        unsigned const tile_row = chunk >> 2, column = chunk & 3;
        unsigned const swizzled = tile_row * nk_cross_slab_bytes_ampere_k + ((column ^ ((tile_row >> 1) & 3)) << 4);
        nk_size_t const byte = slab_offset + (column << 4);
        nk_u32_t const valid = byte < arguments->depth_bytes
                                   ? (nk_u32_t)(arguments->depth_bytes - byte < 16 ? arguments->depth_bytes - byte : 16)
                                   : 0;

        nk_size_t const row = first_row + tile_row;
        unsigned char const *a_source = row < arguments->row_end ? arguments->a + row * arguments->a_stride + byte
                                                                 : arguments->a;
        nk_copy_b128_async_ampere_(nk_shared_address_ampere_(a_stage + swizzled), a_source,
                                   row < arguments->row_end ? valid : 0);

        nk_size_t const b_row = first_column + tile_row;
        unsigned char const *b_source = b_row < arguments->column_count
                                            ? arguments->b + b_row * arguments->b_stride + byte
                                            : arguments->b;
        nk_copy_b128_async_ampere_(nk_shared_address_ampere_(b_stage + swizzled), b_source,
                                   b_row < arguments->column_count ? valid : 0);
    }
}

/** Adds the squares of this thread's staged row, summing each slab apart before folding it into
 *  the running sum. */
NK_HELPER_DEVICE_INLINE void nk_cross_stage_norm_ampere_(nk_cross_norm_update_ampere_t norm_update,
                                                         unsigned char const *stage, nk_u32_t *integer_sum,
                                                         nk_f32_t *real_sum) {
    uint4 const *row = (uint4 const *)(stage + threadIdx.x * nk_cross_slab_bytes_ampere_k);
    nk_f32_t slab_sum = 0;
    // The staging swizzle as the visiting order puts a quarter-warp's 16-byte reads on distinct banks.
#pragma unroll
    for (unsigned chunk = 0; chunk < 4; ++chunk) {
        uint4 const bytes = row[chunk ^ ((threadIdx.x >> 1) & 3)];
        nk_u32_t const words[4] = {bytes.x, bytes.y, bytes.z, bytes.w};
        norm_update(words, integer_sum, &slab_sum);
    }
    // Integer updates leave the slab sum a constant zero, and this test drops their dead fold at compile time.
    if (slab_sum != 0) *real_sum += slab_sum;
}

/**
 *  @brief The whole GEMM of one 128 × 128 output tile, shared by every dtype, both CUDA backends
 *      and every metric.
 *  @param[in] multiply Folds one 32-byte sub-slab into the warp's accumulators; inlined,
 *      being a constant.
 *  @param[in] epilogue What the accumulators hold and how they reach the output.
 *  @param[in] output_scale Undoes the power of two a widening introduced, or 1.
 *  @param[in] triangle Whether tiles and outputs below the diagonal are skipped.
 *  @param[in] metric The dot product itself, or a distance from it and the two squared norms.
 *  @param[in] norm How the squared norms are stored and the metric is computed; unused for dots.
 *  @param[in] norm_update Adds staged squares; unused for dots.
 *  @param[in] norm_scale Undoes the power of two the norm update's widening introduced, or 1.
 *
 *  Row norms, and for @c symmetric the column norms, accumulate from the staged slabs, one row per
 *  thread; @c packed reads its column norms from @c b_norms. After the loop both go to the idle
 *  ring for the epilogue.
 */
NK_HELPER_DEVICE_INLINE void nk_cross_tile_ampere_(nk_cross_multiply_ampere_t multiply, nk_cross_epilogue_t epilogue,
                                                   nk_f32_t output_scale, nk_cross_triangle_t triangle,
                                                   nk_cross_metric_t metric, nk_cross_norm_t norm,
                                                   nk_cross_norm_update_ampere_t norm_update, nk_f32_t norm_scale,
                                                   nk_cross_tile_arguments_ampere_t const *arguments) {
    __shared__ __align__(128) unsigned char staged[nk_cross_stages_ampere_k][2][nk_cross_stage_bytes_ampere_k];

    unsigned const lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    unsigned const warp_row = (warp >> 1) * 64, warp_column = (warp & 1) * 64;
    nk_size_t const slabs = arguments->depth_slabs;

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        // The previous tile's last reads of the ring must retire before this tile's prologue refills it.
        __syncthreads();
        nk_size_t const first_row = arguments->row_start + tile / arguments->column_tiles * nk_cross_tile_ampere_k;
        nk_size_t const first_column = tile % arguments->column_tiles * nk_cross_tile_ampere_k;
        if (triangle == nk_cross_triangle_upper_k && first_column + nk_cross_tile_ampere_k <= first_row) continue;

        nk_fui32_t accumulators[4][8][4];
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
#pragma unroll
                for (unsigned element = 0; element < 4; ++element) accumulators[row_tile][column_tile][element].u = 0;
        nk_u32_t row_integer_norm = 0, column_integer_norm = 0;
        nk_f32_t row_real_norm = 0, column_real_norm = 0;

#pragma unroll
        for (unsigned stage = 0; stage + 1 < nk_cross_stages_ampere_k; ++stage) {
            if (stage < slabs)
                nk_cross_stage_ampere_(staged[stage][0], staged[stage][1], arguments, first_row, first_column,
                                       stage * nk_cross_slab_bytes_ampere_k);
            nk_commit_async_ampere_();
        }

        unsigned read_stage = 0, write_stage = nk_cross_stages_ampere_k - 1;
        for (nk_size_t slab = 0; slab < slabs; ++slab) {
            nk_wait_async_ampere_(nk_cross_stages_ampere_k - 2);
            __syncthreads();
            unsigned char *a_stage = staged[read_stage][0], *b_stage = staged[read_stage][1];
#pragma unroll
            for (unsigned sub_slab = 0; sub_slab < 2; ++sub_slab) {
                nk_u32_t a_fragments[4][4], b_fragments[8][2];
#pragma unroll
                for (unsigned row_tile = 0; row_tile < 4; ++row_tile) {
                    unsigned const row = warp_row + row_tile * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    unsigned const chunk = sub_slab * 2 + (lane >> 4);
                    nk_load_matrices_x4_ampere_(nk_shared_address_ampere_(a_stage + row * nk_cross_slab_bytes_ampere_k +
                                                                          ((chunk ^ ((row >> 1) & 3)) << 4)),
                                                a_fragments[row_tile]);
                }
#pragma unroll
                for (unsigned column_pair = 0; column_pair < 4; ++column_pair) {
                    unsigned const row = warp_column + column_pair * 16 + (lane & 7) + (lane >> 4) * 8;
                    unsigned const chunk = sub_slab * 2 + ((lane >> 3) & 1);
                    nk_u32_t pair[4];
                    nk_load_matrices_x4_ampere_(nk_shared_address_ampere_(b_stage + row * nk_cross_slab_bytes_ampere_k +
                                                                          ((chunk ^ ((row >> 1) & 3)) << 4)),
                                                pair);
                    b_fragments[column_pair * 2][0] = pair[0], b_fragments[column_pair * 2][1] = pair[1];
                    b_fragments[column_pair * 2 + 1][0] = pair[2], b_fragments[column_pair * 2 + 1][1] = pair[3];
                }
                multiply(accumulators, a_fragments, b_fragments);
            }
            nk_size_t const prefetch = slab + nk_cross_stages_ampere_k - 1;
            // Issued after the products, as ptxas otherwise sinks the commit below the fragment loads, stalling them.
            // The stage refilled here was read one iteration ago, and the barrier above retired those reads.
            if (prefetch < slabs)
                nk_cross_stage_ampere_(staged[write_stage][0], staged[write_stage][1], arguments, first_row,
                                       first_column, prefetch * nk_cross_slab_bytes_ampere_k);
            nk_commit_async_ampere_();
            // Squares come after the products, once the fragments are dead, and read a stage no copy refills until the
            // next iteration's barrier.
            if (metric != nk_cross_metric_dot_k) {
                nk_cross_stage_norm_ampere_(norm_update, a_stage, &row_integer_norm, &row_real_norm);
                if (triangle == nk_cross_triangle_upper_k)
                    nk_cross_stage_norm_ampere_(norm_update, b_stage, &column_integer_norm, &column_real_norm);
            }
            read_stage = read_stage + 1 == nk_cross_stages_ampere_k ? 0 : read_stage + 1;
            write_stage = write_stage + 1 == nk_cross_stages_ampere_k ? 0 : write_stage + 1;
        }

        // Row norms, then column norms, in the ring's first stage once every warp's fragment reads retire.
        nk_fui32_t *norms = (nk_fui32_t *)staged[0][0];
        if (metric != nk_cross_metric_dot_k) {
            __syncthreads();
            norms[threadIdx.x] = nk_cross_norm_finalize_ampere_(norm, row_integer_norm, row_real_norm, norm_scale);
            nk_size_t const column = first_column + threadIdx.x;
            if (triangle == nk_cross_triangle_upper_k)
                norms[nk_cross_tile_ampere_k + threadIdx.x] = nk_cross_norm_finalize_ampere_(
                    norm, column_integer_norm, column_real_norm, norm_scale);
            else
                norms[nk_cross_tile_ampere_k + threadIdx.x].u = column < arguments->column_count
                                                                    ? ((nk_u32_t const *)arguments->b_norms)[column]
                                                                    : 0;
            __syncthreads();
        }

        unsigned const group = lane >> 2, quad = lane & 3;
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
            for (unsigned half = 0; half < 2; ++half) {
                unsigned const tile_row = warp_row + row_tile * 16 + half * 8 + group;
                nk_size_t const row = first_row + tile_row;
                if (row >= arguments->row_end) continue;
                unsigned char *output = (unsigned char *)arguments->c + row * arguments->c_stride;
                nk_f32_t const row_norm = metric == nk_cross_metric_dot_k
                                              ? 0.0f
                                              : nk_cross_norm_to_f32_ampere_(norms[tile_row], norm);
#pragma unroll
                for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
#pragma unroll
                    for (unsigned pair = 0; pair < 2; ++pair) {
                        unsigned const tile_column = warp_column + column_tile * 8 + quad * 2 + pair;
                        nk_size_t const column = first_column + tile_column;
                        if (column >= arguments->column_count ||
                            (triangle == nk_cross_triangle_upper_k && column < row))
                            continue;
                        nk_fui32_t const sum = accumulators[row_tile][column_tile][half * 2 + pair];
                        if (metric == nk_cross_metric_dot_k) {
                            if (epilogue == nk_cross_epilogue_f32_k)
                                ((nk_f32_t *)output)[column] = sum.f * output_scale;
                            else if (epilogue == nk_cross_epilogue_i32_k) ((nk_u32_t *)output)[column] = sum.u;
                            else ((nk_f32_t *)output)[column] = (nk_f32_t)sum.i * output_scale;
                            continue;
                        }
                        if (triangle == nk_cross_triangle_upper_k && column == row) {
                            ((nk_f32_t *)output)[column] = 0.0f;
                            continue;
                        }
                        nk_f32_t const dot = nk_cross_dot_to_f32_ampere_(sum, epilogue, norm, output_scale);
                        nk_f32_t const column_norm = nk_cross_norm_to_f32_ampere_(
                            norms[nk_cross_tile_ampere_k + tile_column], norm);
                        ((nk_f32_t *)output)[column] = metric == nk_cross_metric_angular_k
                                                           ? nk_f32_angular_ampere_(dot, row_norm, column_norm)
                                                           : nk_f32_euclidean_ampere_(dot, row_norm, column_norm);
                    }
            }
    }
}

/** Adds a × b into a running sum: one F64 FMA, or Dot2's TwoProd and TwoSum with both
 *  errors kept apart. */
NK_HELPER_DEVICE_INLINE void nk_cross_fma_step_ampere_(nk_cross_accumulation_t accumulation, nk_f64_t a, nk_f64_t b,
                                                       nk_f64_t *sum, nk_f64_t *compensation) {
    if (accumulation == nk_cross_accumulation_f64_k) {
        *sum = __fma_rn(a, b, *sum);
        return;
    }
    nk_f64_t const product = __dmul_rn(a, b), product_error = __fma_rn(a, b, -product);
    nk_f64_t const total = __dadd_rn(*sum, product), virtual_addend = __dsub_rn(total, *sum);
    nk_f64_t const sum_error = __dadd_rn(__dsub_rn(*sum, __dsub_rn(total, virtual_addend)),
                                         __dsub_rn(product, virtual_addend));
    *sum = total;
    *compensation = __dadd_rn(*compensation, __dadd_rn(sum_error, product_error));
}

/** Merges the running sum of lane `lane ^ offset` into this one, through TwoSum under Dot2. */
NK_HELPER_DEVICE_INLINE void nk_cross_fma_merge_lanes_ampere_(nk_cross_accumulation_t accumulation, unsigned offset,
                                                              nk_f64_t *sum, nk_f64_t *compensation) {
    nk_f64_t const other_sum = __shfl_xor_sync(0xFFFFFFFFu, *sum, offset);
    if (accumulation == nk_cross_accumulation_f64_k) {
        *sum = __dadd_rn(*sum, other_sum);
        return;
    }
    nk_f64_t const other_compensation = __shfl_xor_sync(0xFFFFFFFFu, *compensation, offset);
    nk_f64_t const total = __dadd_rn(*sum, other_sum), virtual_addend = __dsub_rn(total, *sum);
    nk_f64_t const sum_error = __dadd_rn(__dsub_rn(*sum, __dsub_rn(total, virtual_addend)),
                                         __dsub_rn(other_sum, virtual_addend));
    *sum = total;
    *compensation = __dadd_rn(__dadd_rn(*compensation, other_compensation), sum_error);
}

/**
 *  @brief The F32 and F64 GEMM of one 64 × 64 output tile on the CUDA cores, each of 256 threads
 *      owning a 4 × 4 grid of outputs strided by 16, so shared-memory reads broadcast along one
 *      axis and sweep the banks along the other.
 *
 *  Tensor-core F64 MMA never exposes a product's rounding error, which Dot2 needs, and outside the
 *  8.0 and 9.0 datacenter parts it runs no faster than these FMAs. The rounded intrinsics keep the
 *  compiler from contracting TwoSum's additions into FMAs, which would break its error terms. For a
 *  metric, each thread also squares the 4 A and, for @c symmetric, 4 B elements it loads, and the
 *  16 threads sharing a row merge their sums after the loop.
 */
NK_HELPER_DEVICE_INLINE void nk_cross_fma_tile_ampere_(nk_cross_load_f64_ampere_t load,
                                                       nk_cross_accumulation_t accumulation,
                                                       nk_cross_triangle_t triangle, nk_cross_metric_t metric,
                                                       nk_cross_tile_arguments_ampere_t const *arguments) {
    // One extra column breaks the 64-element stride that would put a slab's stores on the same banks.
    __shared__ nk_f64_t a_slab[nk_cross_fma_slab_ampere_k][nk_cross_fma_tile_ampere_k + 1];
    __shared__ nk_f64_t b_slab[nk_cross_fma_slab_ampere_k][nk_cross_fma_tile_ampere_k + 1];
    __shared__ nk_f64_t norms[2][nk_cross_fma_tile_ampere_k];

    unsigned const thread_column = threadIdx.x & 15, thread_row = threadIdx.x >> 4;

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        nk_size_t const first_row = arguments->row_start + tile / arguments->column_tiles * nk_cross_fma_tile_ampere_k;
        nk_size_t const first_column = tile % arguments->column_tiles * nk_cross_fma_tile_ampere_k;
        if (triangle == nk_cross_triangle_upper_k && first_column + nk_cross_fma_tile_ampere_k <= first_row) continue;
        nk_f64_t sums[4][4], compensations[4][4];
#pragma unroll
        for (unsigned row_step = 0; row_step < 4; ++row_step)
#pragma unroll
            for (unsigned column_step = 0; column_step < 4; ++column_step)
                sums[row_step][column_step] = 0, compensations[row_step][column_step] = 0;
        nk_f64_t a_norms[nk_cross_fma_loads_ampere_k], a_norm_compensations[nk_cross_fma_loads_ampere_k];
        nk_f64_t b_norms[nk_cross_fma_loads_ampere_k], b_norm_compensations[nk_cross_fma_loads_ampere_k];
#pragma unroll
        for (unsigned step = 0; step < nk_cross_fma_loads_ampere_k; ++step)
            a_norms[step] = 0, a_norm_compensations[step] = 0, b_norms[step] = 0, b_norm_compensations[step] = 0;

        for (nk_size_t slab = 0; slab < arguments->depth; slab += nk_cross_fma_slab_ampere_k) {
#pragma unroll
            for (unsigned element = threadIdx.x; element < nk_cross_fma_tile_ampere_k * nk_cross_fma_slab_ampere_k;
                 element += nk_cross_fma_threads_ampere_k) {
                unsigned const step = element / nk_cross_fma_threads_ampere_k;
                unsigned const tile_row = element / nk_cross_fma_slab_ampere_k;
                unsigned const offset = element % nk_cross_fma_slab_ampere_k;
                nk_size_t const index = slab + offset, row = first_row + tile_row, column = first_column + tile_row;
                nk_f64_t const a_value = row < arguments->row_end && index < arguments->depth
                                             ? load(arguments->a + row * arguments->a_stride, index)
                                             : 0;
                nk_f64_t const b_value = column < arguments->column_count && index < arguments->depth
                                             ? load(arguments->b + column * arguments->b_stride, index)
                                             : 0;
                a_slab[offset][tile_row] = a_value, b_slab[offset][tile_row] = b_value;
                if (metric != nk_cross_metric_dot_k) {
                    nk_cross_fma_step_ampere_(accumulation, a_value, a_value, &a_norms[step],
                                              &a_norm_compensations[step]);
                    if (triangle == nk_cross_triangle_upper_k)
                        nk_cross_fma_step_ampere_(accumulation, b_value, b_value, &b_norms[step],
                                                  &b_norm_compensations[step]);
                }
            }
            __syncthreads();
#pragma unroll
            for (unsigned offset = 0; offset < nk_cross_fma_slab_ampere_k; ++offset) {
                nk_f64_t a_values[4], b_values[4];
#pragma unroll
                for (unsigned step = 0; step < 4; ++step)
                    a_values[step] = a_slab[offset][thread_row + 16 * step],
                    b_values[step] = b_slab[offset][thread_column + 16 * step];
#pragma unroll
                for (unsigned row_step = 0; row_step < 4; ++row_step)
#pragma unroll
                    for (unsigned column_step = 0; column_step < 4; ++column_step)
                        nk_cross_fma_step_ampere_(accumulation, a_values[row_step], b_values[column_step],
                                                  &sums[row_step][column_step], &compensations[row_step][column_step]);
            }
            __syncthreads();
        }

        if (metric != nk_cross_metric_dot_k) {
            // The 16 threads loading one row sit in one half-warp, one per depth offset.
#pragma unroll
            for (unsigned step = 0; step < nk_cross_fma_loads_ampere_k; ++step)
#pragma unroll
                for (unsigned offset = 8; offset != 0; offset >>= 1) {
                    nk_cross_fma_merge_lanes_ampere_(accumulation, offset, &a_norms[step], &a_norm_compensations[step]);
                    if (triangle == nk_cross_triangle_upper_k)
                        nk_cross_fma_merge_lanes_ampere_(accumulation, offset, &b_norms[step],
                                                         &b_norm_compensations[step]);
                }
            // A zero-depth tile skips the slab loop's barriers, so the last tile's epilogue reads retire here.
            __syncthreads();
            if (thread_column == 0)
#pragma unroll
                for (unsigned step = 0; step < nk_cross_fma_loads_ampere_k; ++step) {
                    unsigned const tile_row = thread_row + step * (nk_cross_fma_threads_ampere_k / 16);
                    norms[0][tile_row] = __dadd_rn(a_norms[step], a_norm_compensations[step]);
                    if (triangle == nk_cross_triangle_upper_k)
                        norms[1][tile_row] = __dadd_rn(b_norms[step], b_norm_compensations[step]);
                }
            if (triangle == nk_cross_triangle_full_k && threadIdx.x < nk_cross_fma_tile_ampere_k) {
                nk_size_t const column = first_column + threadIdx.x;
                norms[1][threadIdx.x] = column < arguments->column_count
                                            ? ((nk_f64_t const *)arguments->b_norms)[column]
                                            : 0;
            }
            __syncthreads();
        }

#pragma unroll
        for (unsigned row_step = 0; row_step < 4; ++row_step) {
            nk_size_t const row = first_row + thread_row + 16 * row_step;
            if (row >= arguments->row_end) continue;
            nk_f64_t *output = (nk_f64_t *)((unsigned char *)arguments->c + row * arguments->c_stride);
#pragma unroll
            for (unsigned column_step = 0; column_step < 4; ++column_step) {
                nk_size_t const column = first_column + thread_column + 16 * column_step;
                if (column >= arguments->column_count || (triangle == nk_cross_triangle_upper_k && column < row))
                    continue;
                nk_f64_t const dot = __dadd_rn(sums[row_step][column_step], compensations[row_step][column_step]);
                if (metric == nk_cross_metric_dot_k) output[column] = dot;
                else if (triangle == nk_cross_triangle_upper_k && column == row) output[column] = 0;
                else {
                    nk_f64_t const row_norm = norms[0][thread_row + 16 * row_step];
                    nk_f64_t const column_norm = norms[1][thread_column + 16 * column_step];
                    output[column] = metric == nk_cross_metric_angular_k
                                         ? nk_f64_angular_ampere_(dot, row_norm, column_norm)
                                         : nk_f64_euclidean_ampere_(dot, row_norm, column_norm);
                }
            }
        }
    }
}

/** Mirrors @c nk_define_cross_pack_size_: depth padded to whole slabs, plus one more when the
 *  stride is a power of 2. */
NK_HELPER_INLINE nk_size_t nk_cross_padded_values_ampere_(nk_size_t depth, nk_size_t depth_simd_dimensions,
                                                          nk_size_t dimensions_per_value, nk_size_t value_bytes) {
    nk_size_t values = nk_size_round_up_to_multiple_(depth, depth_simd_dimensions) / dimensions_per_value;
    nk_size_t const stride_bytes = values * value_bytes;
    if ((stride_bytes & (stride_bytes - 1)) == 0 && stride_bytes > 0)
        values += depth_simd_dimensions / dimensions_per_value;
    return values;
}

/** Validates the contract and launches as many blocks of @p kernel as stay resident, each walking
 *  @p tile × @p tile output tiles with a stride of the grid. @p b_norms is the packed column norms
 *  a @c packed metric reads, or null. */
NK_HELPER_INLINE cudaError_t nk_cross_launch_ampere_(void const *kernel, unsigned tile, unsigned threads, void const *a,
                                                     void const *b, void const *b_norms, void *c,
                                                     nk_size_t result_bytes, nk_size_t row_start, nk_size_t row_end,
                                                     nk_size_t column_count, nk_size_t depth, nk_size_t depth_bytes,
                                                     nk_size_t a_stride, nk_size_t b_stride, nk_size_t c_stride,
                                                     cudaStream_t stream) {
    if ((((nk_size_t)a) | a_stride | ((nk_size_t)b) | b_stride) & 15 ||
        (((nk_size_t)c) | c_stride) & (result_bytes - 1))
        return cudaErrorMisalignedAddress;
    if (row_end <= row_start || column_count == 0) return cudaSuccess;
    int device = 0, multiprocessors = 0, resident_per_multiprocessor = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status == cudaSuccess)
        status = cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device);
    if (status == cudaSuccess)
        status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident_per_multiprocessor, kernel, (int)threads, 0);
    if (status != cudaSuccess) return status;
    nk_size_t const column_tiles = nk_size_divide_round_up_(column_count, tile);
    nk_size_t const tiles = nk_size_divide_round_up_(row_end - row_start, tile) * column_tiles;
    nk_size_t const resident = (nk_size_t)multiprocessors * (nk_size_t)resident_per_multiprocessor;
    nk_size_t const blocks = resident != 0 && resident < tiles ? resident : tiles;
    nk_cross_tile_arguments_ampere_t arguments;
    arguments.a = (unsigned char const *)a, arguments.b = (unsigned char const *)b, arguments.c = c;
    arguments.row_start = row_start, arguments.row_end = row_end, arguments.column_count = column_count;
    arguments.depth = depth, arguments.depth_bytes = depth_bytes, arguments.a_stride = a_stride;
    arguments.b_stride = b_stride;
    arguments.c_stride = c_stride, arguments.column_tiles = column_tiles, arguments.tiles = tiles;
    arguments.depth_slabs = nk_size_divide_round_up_(depth_bytes, nk_cross_slab_bytes_ampere_k);
    arguments.b_norms = b_norms;
    dim3 grid, block;
    grid.x = (unsigned)blocks, grid.y = 1, grid.z = 1;
    block.x = threads, block.y = 1, block.z = 1;
    void *launch_arguments[1];
    launch_arguments[0] = &arguments;
    return cudaLaunchKernel(kernel, grid, block, launch_arguments, 0, stream);
}

/** Launches @p kernel with one warp per packed column, walked with a stride of the grid. */
NK_HELPER_INLINE cudaError_t nk_cross_pack_launch_ampere_(void const *kernel, void const *b, nk_size_t column_count,
                                                          nk_size_t depth, nk_size_t depth_bytes, nk_size_t b_stride,
                                                          void *b_packed, nk_size_t columns_begin,
                                                          nk_size_t columns_end, nk_size_t depth_values_padded,
                                                          cudaStream_t stream) {
    nk_size_t const columns = columns_end > columns_begin ? columns_end - columns_begin : 0;
    nk_size_t const needed = nk_size_divide_round_up_(columns, nk_cross_pack_warps_ampere_k);
    nk_size_t const blocks = needed == 0 ? 1 : needed < 65535 ? needed : 65535;
    dim3 grid, block;
    grid.x = (unsigned)blocks, grid.y = 1, grid.z = 1;
    block.x = nk_cross_pack_warps_ampere_k * 32, block.y = 1, block.z = 1;
    void *arguments[9];
    arguments[0] = &b, arguments[1] = &column_count, arguments[2] = &depth, arguments[3] = &depth_bytes;
    arguments[4] = &b_stride, arguments[5] = &b_packed, arguments[6] = &columns_begin, arguments[7] = &columns_end;
    arguments[8] = &depth_values_padded;
    return cudaLaunchKernel(kernel, grid, block, arguments, 0, stream);
}

#pragma endregion Tile

#pragma region Cross Macros

/**
 *  @brief Generates a packed-shape accessor copying a device-resident packed buffer's header back.
 *  @sa nk_define_cross_packed_shape_ for the host-resident original.
 */
#define nk_define_cross_cuda_packed_shape_(api_name, input_type_name, isa_suffix)                                \
    NK_API_COMPTIME cudaError_t nk_##api_name##_packed_shape_##input_type_name##_##isa_suffix(                   \
        void const *b_packed, nk_size_t *width, nk_size_t *depth, cudaStream_t stream) {                         \
        nk_cross_packed_buffer_header_t header;                                                                  \
        cudaError_t status = cudaMemcpyAsync(&header, b_packed, sizeof(header), cudaMemcpyDeviceToHost, stream); \
        if (status == cudaSuccess) status = cudaStreamSynchronize(stream);                                       \
        if (status != cudaSuccess) return status;                                                                \
        *width = header.column_count, *depth = header.depth_dimensions;                                          \
        return cudaSuccess;                                                                                      \
    }

/**
 *  @brief Generates a pack into the serial layout on the device, one warp per column.
 *
 *  The header is written only when @p columns_begin is zero, and each packed column gets its row
 *  and its norm, so disjoint column ranges may be packed by separate calls, exactly as with
 *  @c nk_define_cross_pack_.
 *
 *  @param[in] load_fn Device map from an input byte to its packed byte, the identity unless the
 *      multiply wants a remap.
 *  @param[in] compute_norm_fn Device share of a column's sum of squares for one lane of 32.
 *  @sa nk_define_cross_pack_ for the host original.
 */
#define nk_define_cross_cuda_pack_(api_name, input_type_name, isa_suffix, input_value_type, packed_value_type,        \
                                   load_fn, norm_value_type, compute_norm_fn, depth_simd_dimensions,                  \
                                   dimensions_per_value)                                                              \
    static __global__ void nk_##api_name##_pack_##input_type_name##_##isa_suffix##_kernel_(                           \
        unsigned char const *b, nk_size_t column_count, nk_size_t depth, nk_size_t depth_bytes,                       \
        nk_size_t b_stride_in_bytes, unsigned char *b_packed, nk_size_t columns_begin, nk_size_t columns_end,         \
        nk_size_t depth_values_padded) {                                                                              \
        nk_size_t const row_bytes = depth_values_padded * sizeof(nk_##packed_value_type##_t);                         \
        if (columns_begin == 0 && blockIdx.x == 0 && threadIdx.x == 0) {                                              \
            nk_cross_packed_buffer_header_t *header = (nk_cross_packed_buffer_header_t *)b_packed;                    \
            header->column_count = (nk_u32_t)column_count;                                                            \
            header->depth_dimensions = (nk_u32_t)depth;                                                               \
            header->depth_padded_values = (nk_u32_t)depth_values_padded;                                              \
            for (unsigned reserved_index = 0; reserved_index < 13; ++reserved_index)                                  \
                header->reserved[reserved_index] = 0;                                                                 \
        }                                                                                                             \
        unsigned char *rows = b_packed + sizeof(nk_cross_packed_buffer_header_t);                                     \
        nk_##norm_value_type##_t *norms = (nk_##norm_value_type##_t *)(rows + column_count * row_bytes);              \
        unsigned const lane = threadIdx.x & 31;                                                                       \
        nk_size_t const warps = (nk_size_t)gridDim.x * (blockDim.x >> 5);                                             \
        for (nk_size_t column = columns_begin + (nk_size_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);       \
             column < columns_end; column += warps) {                                                                 \
            unsigned char const *source = b + column * b_stride_in_bytes;                                             \
            unsigned char *destination = rows + column * row_bytes;                                                   \
            for (nk_size_t byte = lane; byte < row_bytes; byte += 32)                                                 \
                destination[byte] = byte < depth_bytes ? load_fn(source[byte]) : 0;                                   \
            nk_##norm_value_type##_t norm = compute_norm_fn(source, depth, lane);                                     \
            for (unsigned offset = 16; offset != 0; offset >>= 1) norm += __shfl_xor_sync(0xFFFFFFFFu, norm, offset); \
            if (lane == 0) norms[column] = norm;                                                                      \
        }                                                                                                             \
    }                                                                                                                 \
    NK_API_COMPTIME cudaError_t nk_##api_name##_pack_##input_type_name##_##isa_suffix(                                \
        nk_##input_value_type##_t const *b, nk_size_t column_count, nk_size_t depth, nk_size_t b_stride_in_bytes,     \
        void *b_packed, nk_size_t columns_begin, nk_size_t columns_end, cudaStream_t stream) {                        \
        nk_size_t const depth_values_padded = nk_cross_padded_values_ampere_(                                         \
            depth, depth_simd_dimensions, dimensions_per_value, sizeof(nk_##packed_value_type##_t));                  \
        nk_size_t const depth_bytes = depth / dimensions_per_value * sizeof(nk_##input_value_type##_t);               \
        return nk_cross_pack_launch_ampere_(                                                                          \
            (void const *)nk_##api_name##_pack_##input_type_name##_##isa_suffix##_kernel_, b, column_count, depth,    \
            depth_bytes, b_stride_in_bytes, b_packed, columns_begin, columns_end, depth_values_padded, stream);       \
    }

/**
 *  @brief Generates C = A × Bᵀ over a B packed by @c nk_define_cross_cuda_pack_, one block per
 *      128 × 128 tile.
 *  @param[in] multiply_fn Device fold of one 32-byte sub-slab of fragments, see
 *      @c nk_cross_multiply_ampere_t.
 *  @param[in] epilogue What the accumulators hold and how they reach the output, see
 *      @c nk_cross_epilogue_t.
 *  @param[in] output_scale Undoes the power of two a widening introduced, or 1.
 *  @sa nk_define_cross_packed_ for the host original.
 */
#define nk_define_cross_cuda_packed_(api_name, input_type_name, isa_suffix, input_value_type, packed_value_type,       \
                                     result_value_type, multiply_fn, epilogue, output_scale, depth_simd_dimensions,    \
                                     dimensions_per_value)                                                             \
    static __global__ void __launch_bounds__(nk_cross_threads_ampere_k)                                                \
        nk_##api_name##_packed_##input_type_name##_##isa_suffix##_kernel_(                                             \
            nk_cross_tile_arguments_ampere_t arguments) {                                                              \
        nk_cross_tile_ampere_(multiply_fn, epilogue, output_scale, nk_cross_triangle_full_k, nk_cross_metric_dot_k,    \
                              nk_cross_norm_f32_k, 0, 1.0f, &arguments);                                               \
    }                                                                                                                  \
    NK_API_COMPTIME cudaError_t nk_##api_name##_packed_##input_type_name##_##isa_suffix(                               \
        nk_##input_value_type##_t const *a_matrix, void const *b_packed_buffer, nk_##result_value_type##_t *c_matrix,  \
        nk_size_t row_count, nk_size_t column_count, nk_size_t depth, nk_size_t a_stride_in_bytes,                     \
        nk_size_t c_stride_in_bytes, cudaStream_t stream) {                                                            \
        nk_size_t const row_bytes = nk_cross_padded_values_ampere_(depth, depth_simd_dimensions, dimensions_per_value, \
                                                                   sizeof(nk_##packed_value_type##_t)) *               \
                                    sizeof(nk_##packed_value_type##_t);                                                \
        return nk_cross_launch_ampere_(                                                                                \
            (void const *)nk_##api_name##_packed_##input_type_name##_##isa_suffix##_kernel_, nk_cross_tile_ampere_k,   \
            nk_cross_threads_ampere_k, a_matrix,                                                                       \
            (unsigned char const *)b_packed_buffer + sizeof(nk_cross_packed_buffer_header_t), 0, c_matrix,             \
            sizeof(nk_##result_value_type##_t), 0, row_count, column_count, depth,                                     \
            depth / dimensions_per_value * sizeof(nk_##input_value_type##_t), a_stride_in_bytes, row_bytes,            \
            c_stride_in_bytes, stream);                                                                                \
    }

/**
 *  @brief Generates the Gram matrix C = A × Aᵀ over rows [row_start, row_start + row_count),
 *      writing the upper triangle with its diagonal and skipping tiles wholly below it.
 *  @sa nk_define_cross_symmetric_ for the host original.
 */
#define nk_define_cross_cuda_symmetric_(api_name, input_type_name, isa_suffix, input_value_type, result_value_type,    \
                                        multiply_fn, epilogue, output_scale, dimensions_per_value)                     \
    static __global__ void __launch_bounds__(nk_cross_threads_ampere_k)                                                \
        nk_##api_name##_symmetric_##input_type_name##_##isa_suffix##_kernel_(                                          \
            nk_cross_tile_arguments_ampere_t arguments) {                                                              \
        nk_cross_tile_ampere_(multiply_fn, epilogue, output_scale, nk_cross_triangle_upper_k, nk_cross_metric_dot_k,   \
                              nk_cross_norm_f32_k, 0, 1.0f, &arguments);                                               \
    }                                                                                                                  \
    NK_API_COMPTIME cudaError_t nk_##api_name##_symmetric_##input_type_name##_##isa_suffix(                            \
        nk_##input_value_type##_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_in_bytes, \
        nk_##result_value_type##_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start,                     \
        nk_size_t row_count, cudaStream_t stream) {                                                                    \
        nk_size_t const row_end = row_start + row_count < vectors_count ? row_start + row_count : vectors_count;       \
        return nk_cross_launch_ampere_(                                                                                \
            (void const *)nk_##api_name##_symmetric_##input_type_name##_##isa_suffix##_kernel_,                        \
            nk_cross_tile_ampere_k, nk_cross_threads_ampere_k, vectors, vectors, 0, result,                            \
            sizeof(nk_##result_value_type##_t), row_start, row_end, vectors_count, depth,                              \
            depth / dimensions_per_value * sizeof(nk_##input_value_type##_t), stride_in_bytes, stride_in_bytes,        \
            result_stride_in_bytes, stream);                                                                           \
    }

/**
 *  @brief Generates C = A × Bᵀ over a packed B on the CUDA cores, one block per 64 × 64 tile, for
 *      the F32 and F64 inputs whose serial backends accumulate in F64 or Dot2.
 *  @param[in] load_fn Device reader of one element as F64, see @c nk_cross_load_f64_ampere_t.
 *  @param[in] accumulation Plain F64 FMAs or Dot2, see @c nk_cross_accumulation_t.
 *  @sa nk_define_cross_packed_ for the host original.
 */
#define nk_define_cross_cuda_fma_packed_(api_name, input_type_name, isa_suffix, input_value_type, packed_value_type,   \
                                         result_value_type, load_fn, accumulation, depth_simd_dimensions,              \
                                         dimensions_per_value)                                                         \
    static __global__ void __launch_bounds__(nk_cross_fma_threads_ampere_k)                                            \
        nk_##api_name##_packed_##input_type_name##_##isa_suffix##_kernel_(                                             \
            nk_cross_tile_arguments_ampere_t arguments) {                                                              \
        nk_cross_fma_tile_ampere_(load_fn, accumulation, nk_cross_triangle_full_k, nk_cross_metric_dot_k, &arguments); \
    }                                                                                                                  \
    NK_API_COMPTIME cudaError_t nk_##api_name##_packed_##input_type_name##_##isa_suffix(                               \
        nk_##input_value_type##_t const *a_matrix, void const *b_packed_buffer, nk_##result_value_type##_t *c_matrix,  \
        nk_size_t row_count, nk_size_t column_count, nk_size_t depth, nk_size_t a_stride_in_bytes,                     \
        nk_size_t c_stride_in_bytes, cudaStream_t stream) {                                                            \
        nk_size_t const row_bytes = nk_cross_padded_values_ampere_(depth, depth_simd_dimensions, dimensions_per_value, \
                                                                   sizeof(nk_##packed_value_type##_t)) *               \
                                    sizeof(nk_##packed_value_type##_t);                                                \
        return nk_cross_launch_ampere_(                                                                                \
            (void const *)nk_##api_name##_packed_##input_type_name##_##isa_suffix##_kernel_,                           \
            nk_cross_fma_tile_ampere_k, nk_cross_fma_threads_ampere_k, a_matrix,                                       \
            (unsigned char const *)b_packed_buffer + sizeof(nk_cross_packed_buffer_header_t), 0, c_matrix,             \
            sizeof(nk_##result_value_type##_t), 0, row_count, column_count, depth,                                     \
            depth * sizeof(nk_##input_value_type##_t), a_stride_in_bytes, row_bytes, c_stride_in_bytes, stream);       \
    }

/**
 *  @brief Generates the Gram matrix C = A × Aᵀ on the CUDA cores over rows [row_start, row_start +
 *      row_count), writing the upper triangle with its diagonal.
 *  @sa nk_define_cross_symmetric_ for the host original.
 */
#define nk_define_cross_cuda_fma_symmetric_(api_name, input_type_name, isa_suffix, input_value_type,                   \
                                            result_value_type, load_fn, accumulation)                                  \
    static __global__ void __launch_bounds__(nk_cross_fma_threads_ampere_k)                                            \
        nk_##api_name##_symmetric_##input_type_name##_##isa_suffix##_kernel_(                                          \
            nk_cross_tile_arguments_ampere_t arguments) {                                                              \
        nk_cross_fma_tile_ampere_(load_fn, accumulation, nk_cross_triangle_upper_k, nk_cross_metric_dot_k,             \
                                  &arguments);                                                                         \
    }                                                                                                                  \
    NK_API_COMPTIME cudaError_t nk_##api_name##_symmetric_##input_type_name##_##isa_suffix(                            \
        nk_##input_value_type##_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_in_bytes, \
        nk_##result_value_type##_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start,                     \
        nk_size_t row_count, cudaStream_t stream) {                                                                    \
        nk_size_t const row_end = row_start + row_count < vectors_count ? row_start + row_count : vectors_count;       \
        return nk_cross_launch_ampere_(                                                                                \
            (void const *)nk_##api_name##_symmetric_##input_type_name##_##isa_suffix##_kernel_,                        \
            nk_cross_fma_tile_ampere_k, nk_cross_fma_threads_ampere_k, vectors, vectors, 0, result,                    \
            sizeof(nk_##result_value_type##_t), row_start, row_end, vectors_count, depth,                              \
            depth * sizeof(nk_##input_value_type##_t), stride_in_bytes, stride_in_bytes, result_stride_in_bytes,       \
            stream);                                                                                                   \
    }

#pragma endregion Cross Macros

#pragma region Multiplies

NK_HELPER_DEVICE_INLINE void nk_dots_bf16_multiply_ampere_(nk_fui32_t accumulators[4][8][4], nk_u32_t const a[4][4],
                                                           nk_u32_t const b[8][2]) {
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
            nk_mma_bf16_ampere_(accumulators[row_tile][column_tile], a[row_tile], b[column_tile][0], b[column_tile][1]);
}

NK_HELPER_DEVICE_INLINE void nk_dots_f16_multiply_ampere_(nk_fui32_t accumulators[4][8][4], nk_u32_t const a[4][4],
                                                          nk_u32_t const b[8][2]) {
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
            nk_mma_f16_ampere_(accumulators[row_tile][column_tile], a[row_tile], b[column_tile][0], b[column_tile][1]);
}

/** Widens 1-byte fragments into F16: 4 codes per register become the depth of
 *  two m16n8k16 steps. */
NK_HELPER_DEVICE_INLINE void nk_dots_widened_f16_multiply_ampere_(nk_cross_widen_ampere_t widen,
                                                                  nk_fui32_t accumulators[4][8][4],
                                                                  nk_u32_t const a[4][4], nk_u32_t const b[8][2]) {
    nk_u32_t b_low[8][2], b_high[8][2];
#pragma unroll
    for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
#pragma unroll
        for (unsigned depth_half = 0; depth_half < 2; ++depth_half)
            widen(b[column_tile][depth_half], &b_low[column_tile][depth_half], &b_high[column_tile][depth_half]);
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile) {
        nk_u32_t a_low[4], a_high[4];
#pragma unroll
        for (unsigned fragment = 0; fragment < 4; ++fragment)
            widen(a[row_tile][fragment], &a_low[fragment], &a_high[fragment]);
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

NK_HELPER_DEVICE_INLINE void nk_dots_e5m2_multiply_ampere_(nk_fui32_t accumulators[4][8][4], nk_u32_t const a[4][4],
                                                           nk_u32_t const b[8][2]) {
    nk_dots_widened_f16_multiply_ampere_(nk_e5m2x4_to_f16x4_ampere_, accumulators, a, b);
}

NK_HELPER_DEVICE_INLINE void nk_dots_e4m3_multiply_ampere_(nk_fui32_t accumulators[4][8][4], nk_u32_t const a[4][4],
                                                           nk_u32_t const b[8][2]) {
    nk_dots_widened_f16_multiply_ampere_(nk_e4m3x4_to_f16x4_ampere_, accumulators, a, b);
}

NK_HELPER_DEVICE_INLINE void nk_dots_e3m2_multiply_ampere_(nk_fui32_t accumulators[4][8][4], nk_u32_t const a[4][4],
                                                           nk_u32_t const b[8][2]) {
    nk_dots_widened_f16_multiply_ampere_(nk_e3m2x4_to_f16x4_ampere_, accumulators, a, b);
}

/** E2M3 against the scaled i8 its pack produced: only A converts, one integer step per
 *  16 × 8 output. */
NK_HELPER_DEVICE_INLINE void nk_dots_e2m3_packed_multiply_ampere_(nk_fui32_t accumulators[4][8][4],
                                                                  nk_u32_t const a[4][4], nk_u32_t const b[8][2]) {
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
NK_HELPER_DEVICE_INLINE void nk_dots_e2m3_multiply_ampere_(nk_fui32_t accumulators[4][8][4], nk_u32_t const a[4][4],
                                                           nk_u32_t const b[8][2]) {
    nk_u32_t b_scaled[8][2];
#pragma unroll
    for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
#pragma unroll
        for (unsigned depth_half = 0; depth_half < 2; ++depth_half)
            b_scaled[column_tile][depth_half] = nk_e2m3x4_to_i8x4_ampere_(b[column_tile][depth_half]);
    nk_dots_e2m3_packed_multiply_ampere_(accumulators, a, b_scaled);
}

/** Widens nibble fragments into 1-byte ones: 8 nibbles per register become the depth of
 *  two m16n8k32 steps. */
NK_HELPER_DEVICE_INLINE void nk_dots_widened_i8_multiply_ampere_(nk_cross_widen_ampere_t widen,
                                                                 nk_cross_mma_ampere_t mma,
                                                                 nk_fui32_t accumulators[4][8][4],
                                                                 nk_u32_t const a[4][4], nk_u32_t const b[8][2]) {
    nk_u32_t b_low[8][2], b_high[8][2];
#pragma unroll
    for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
#pragma unroll
        for (unsigned depth_half = 0; depth_half < 2; ++depth_half)
            widen(b[column_tile][depth_half], &b_low[column_tile][depth_half], &b_high[column_tile][depth_half]);
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile) {
        nk_u32_t a_low[4], a_high[4];
#pragma unroll
        for (unsigned fragment = 0; fragment < 4; ++fragment)
            widen(a[row_tile][fragment], &a_low[fragment], &a_high[fragment]);
        nk_u32_t const first[4] = {a_low[0], a_low[1], a_high[0], a_high[1]};
        nk_u32_t const second[4] = {a_low[2], a_low[3], a_high[2], a_high[3]};
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 8; ++column_tile) {
            mma(accumulators[row_tile][column_tile], first, b_low[column_tile][0], b_high[column_tile][0]);
            mma(accumulators[row_tile][column_tile], second, b_low[column_tile][1], b_high[column_tile][1]);
        }
    }
}

NK_HELPER_DEVICE_INLINE void nk_dots_e2m1_multiply_ampere_(nk_fui32_t accumulators[4][8][4], nk_u32_t const a[4][4],
                                                           nk_u32_t const b[8][2]) {
    nk_dots_widened_i8_multiply_ampere_(nk_e2m1x8_to_i8x8_ampere_, nk_mma_i8_ampere_, accumulators, a, b);
}

NK_HELPER_DEVICE_INLINE void nk_dots_i8_multiply_ampere_(nk_fui32_t accumulators[4][8][4], nk_u32_t const a[4][4],
                                                         nk_u32_t const b[8][2]) {
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
            nk_mma_i8_ampere_(accumulators[row_tile][column_tile], a[row_tile], b[column_tile][0], b[column_tile][1]);
}

NK_HELPER_DEVICE_INLINE void nk_dots_u8_multiply_ampere_(nk_fui32_t accumulators[4][8][4], nk_u32_t const a[4][4],
                                                         nk_u32_t const b[8][2]) {
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
            nk_mma_u8_ampere_(accumulators[row_tile][column_tile], a[row_tile], b[column_tile][0], b[column_tile][1]);
}

/*  From 9.0 on @c ptxas lowers 4-bit MMA to a software routine, so there the nibbles widen to
 *  8-bit steps instead. */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

NK_HELPER_DEVICE_INLINE void nk_dots_i4_multiply_ampere_(nk_fui32_t accumulators[4][8][4], nk_u32_t const a[4][4],
                                                         nk_u32_t const b[8][2]) {
    nk_dots_widened_i8_multiply_ampere_(nk_i4x8_to_i8x8_ampere_, nk_mma_i8_ampere_, accumulators, a, b);
}

NK_HELPER_DEVICE_INLINE void nk_dots_u4_multiply_ampere_(nk_fui32_t accumulators[4][8][4], nk_u32_t const a[4][4],
                                                         nk_u32_t const b[8][2]) {
    nk_dots_widened_i8_multiply_ampere_(nk_u4x8_to_u8x8_ampere_, nk_mma_u8_ampere_, accumulators, a, b);
}

#else

NK_HELPER_DEVICE_INLINE void nk_dots_i4_multiply_ampere_(nk_fui32_t accumulators[4][8][4], nk_u32_t const a[4][4],
                                                         nk_u32_t const b[8][2]) {
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
            nk_mma_i4_ampere_(accumulators[row_tile][column_tile], a[row_tile], b[column_tile][0], b[column_tile][1]);
}

NK_HELPER_DEVICE_INLINE void nk_dots_u4_multiply_ampere_(nk_fui32_t accumulators[4][8][4], nk_u32_t const a[4][4],
                                                         nk_u32_t const b[8][2]) {
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
            nk_mma_u4_ampere_(accumulators[row_tile][column_tile], a[row_tile], b[column_tile][0], b[column_tile][1]);
}

#endif // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

#pragma endregion Multiplies

#pragma region F64

nk_define_cross_pack_size_(dots, f64, ampere, f64, f64, /*norm_value_type=*/f64, /*depth_simd_dimensions=*/8,
                           /*dimensions_per_value=*/1)
nk_define_cross_cuda_packed_shape_(dots, f64, ampere)
nk_define_cross_cuda_pack_(dots, f64, ampere, f64, f64, nk_load_b8_ampere_, /*norm_value_type=*/f64,
                           nk_dots_reduce_sumsq_f64_ampere_, /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_cuda_fma_symmetric_(dots, f64, ampere, f64, f64, nk_load_f64_ampere_, nk_cross_accumulation_dot2_k)
nk_define_cross_cuda_fma_packed_(dots, f64, ampere, f64, f64, f64, nk_load_f64_ampere_, nk_cross_accumulation_dot2_k,
                                 /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)

#pragma endregion F64

#pragma region F32

nk_define_cross_pack_size_(dots, f32, ampere, f32, f32, /*norm_value_type=*/f64, /*depth_simd_dimensions=*/16,
                           /*dimensions_per_value=*/1)
nk_define_cross_cuda_packed_shape_(dots, f32, ampere)
nk_define_cross_cuda_pack_(dots, f32, ampere, f32, f32, nk_load_b8_ampere_, /*norm_value_type=*/f64,
                           nk_dots_reduce_sumsq_f32_ampere_, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_cuda_fma_symmetric_(dots, f32, ampere, f32, f64, nk_load_f32_to_f64_ampere_,
                                    nk_cross_accumulation_f64_k)
nk_define_cross_cuda_fma_packed_(dots, f32, ampere, f32, f32, f64, nk_load_f32_to_f64_ampere_,
                                 nk_cross_accumulation_f64_k, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

#pragma endregion F32

#pragma region BF16

nk_define_cross_pack_size_(dots, bf16, ampere, bf16, bf16, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1)
nk_define_cross_cuda_packed_shape_(dots, bf16, ampere)
nk_define_cross_cuda_pack_(dots, bf16, ampere, bf16, bf16, nk_load_b8_ampere_, /*norm_value_type=*/f32,
                           nk_dots_reduce_sumsq_bf16_ampere_, /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_cuda_symmetric_(dots, bf16, ampere, bf16, f32, nk_dots_bf16_multiply_ampere_, nk_cross_epilogue_f32_k,
                                /*output_scale=*/1.0f, /*dimensions_per_value=*/1)
nk_define_cross_cuda_packed_(dots, bf16, ampere, bf16, bf16, f32, nk_dots_bf16_multiply_ampere_,
                             nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, /*depth_simd_dimensions=*/32,
                             /*dimensions_per_value=*/1)

#pragma endregion BF16

#pragma region F16

nk_define_cross_pack_size_(dots, f16, ampere, f16, f16, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/32,
                           /*dimensions_per_value=*/1)
nk_define_cross_cuda_packed_shape_(dots, f16, ampere)
nk_define_cross_cuda_pack_(dots, f16, ampere, f16, f16, nk_load_b8_ampere_, /*norm_value_type=*/f32,
                           nk_dots_reduce_sumsq_f16_ampere_, /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_cuda_symmetric_(dots, f16, ampere, f16, f32, nk_dots_f16_multiply_ampere_, nk_cross_epilogue_f32_k,
                                /*output_scale=*/1.0f, /*dimensions_per_value=*/1)
nk_define_cross_cuda_packed_(dots, f16, ampere, f16, f16, f32, nk_dots_f16_multiply_ampere_, nk_cross_epilogue_f32_k,
                             /*output_scale=*/1.0f, /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)

#pragma endregion F16

#pragma region E5M2

nk_define_cross_pack_size_(dots, e5m2, ampere, e5m2, e5m2, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/64,
                           /*dimensions_per_value=*/1)
nk_define_cross_cuda_packed_shape_(dots, e5m2, ampere)
nk_define_cross_cuda_pack_(dots, e5m2, ampere, e5m2, e5m2, nk_load_b8_ampere_, /*norm_value_type=*/f32,
                           nk_dots_reduce_sumsq_e5m2_ampere_, /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_cuda_symmetric_(dots, e5m2, ampere, e5m2, f32, nk_dots_e5m2_multiply_ampere_, nk_cross_epilogue_f32_k,
                                /*output_scale=*/1.0f, /*dimensions_per_value=*/1)
nk_define_cross_cuda_packed_(dots, e5m2, ampere, e5m2, e5m2, f32, nk_dots_e5m2_multiply_ampere_,
                             nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, /*depth_simd_dimensions=*/64,
                             /*dimensions_per_value=*/1)

#pragma endregion E5M2

#pragma region E4M3

nk_define_cross_pack_size_(dots, e4m3, ampere, e4m3, e4m3, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/64,
                           /*dimensions_per_value=*/1)
nk_define_cross_cuda_packed_shape_(dots, e4m3, ampere)
nk_define_cross_cuda_pack_(dots, e4m3, ampere, e4m3, e4m3, nk_load_b8_ampere_, /*norm_value_type=*/f32,
                           nk_dots_reduce_sumsq_e4m3_ampere_, /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_cuda_symmetric_(dots, e4m3, ampere, e4m3, f32, nk_dots_e4m3_multiply_ampere_, nk_cross_epilogue_f32_k,
                                /*output_scale=*/65536.0f, /*dimensions_per_value=*/1)
nk_define_cross_cuda_packed_(dots, e4m3, ampere, e4m3, e4m3, f32, nk_dots_e4m3_multiply_ampere_,
                             nk_cross_epilogue_f32_k, /*output_scale=*/65536.0f, /*depth_simd_dimensions=*/64,
                             /*dimensions_per_value=*/1)

#pragma endregion E4M3

#pragma region E3M2

nk_define_cross_pack_size_(dots, e3m2, ampere, e3m2, e3m2, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/64,
                           /*dimensions_per_value=*/1)
nk_define_cross_cuda_packed_shape_(dots, e3m2, ampere)
nk_define_cross_cuda_pack_(dots, e3m2, ampere, e3m2, e3m2, nk_load_b8_ampere_, /*norm_value_type=*/f32,
                           nk_dots_reduce_sumsq_e3m2_ampere_, /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_cuda_symmetric_(dots, e3m2, ampere, e3m2, f32, nk_dots_e3m2_multiply_ampere_, nk_cross_epilogue_f32_k,
                                /*output_scale=*/16777216.0f, /*dimensions_per_value=*/1)
nk_define_cross_cuda_packed_(dots, e3m2, ampere, e3m2, e3m2, f32, nk_dots_e3m2_multiply_ampere_,
                             nk_cross_epilogue_f32_k, /*output_scale=*/16777216.0f, /*depth_simd_dimensions=*/64,
                             /*dimensions_per_value=*/1)

#pragma endregion E3M2

#pragma region E2M3

nk_define_cross_pack_size_(dots, e2m3, ampere, e2m3, i8, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/64,
                           /*dimensions_per_value=*/1)
nk_define_cross_cuda_packed_shape_(dots, e2m3, ampere)
nk_define_cross_cuda_pack_(dots, e2m3, ampere, e2m3, i8, nk_load_e2m3_to_i8_ampere_, /*norm_value_type=*/f32,
                           nk_dots_reduce_sumsq_e2m3_ampere_, /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_cuda_symmetric_(dots, e2m3, ampere, e2m3, f32, nk_dots_e2m3_multiply_ampere_,
                                nk_cross_epilogue_i32_to_f32_k, /*output_scale=*/0.015625f,
                                /*dimensions_per_value=*/1)
nk_define_cross_cuda_packed_(dots, e2m3, ampere, e2m3, i8, f32, nk_dots_e2m3_packed_multiply_ampere_,
                             nk_cross_epilogue_i32_to_f32_k, /*output_scale=*/0.015625f,
                             /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)

#pragma endregion E2M3

#pragma region E2M1

nk_define_cross_pack_size_(dots, e2m1, ampere, e2m1x2, e2m1x2, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/2)
nk_define_cross_cuda_packed_shape_(dots, e2m1, ampere)
nk_define_cross_cuda_pack_(dots, e2m1, ampere, e2m1x2, e2m1x2, nk_load_b8_ampere_, /*norm_value_type=*/f32,
                           nk_dots_reduce_sumsq_e2m1_ampere_, /*depth_simd_dimensions=*/128,
                           /*dimensions_per_value=*/2)
nk_define_cross_cuda_symmetric_(dots, e2m1, ampere, e2m1x2, f32, nk_dots_e2m1_multiply_ampere_,
                                nk_cross_epilogue_i32_to_f32_k, /*output_scale=*/0.25f, /*dimensions_per_value=*/2)
nk_define_cross_cuda_packed_(dots, e2m1, ampere, e2m1x2, e2m1x2, f32, nk_dots_e2m1_multiply_ampere_,
                             nk_cross_epilogue_i32_to_f32_k, /*output_scale=*/0.25f, /*depth_simd_dimensions=*/128,
                             /*dimensions_per_value=*/2)

#pragma endregion E2M1

#pragma region I8

nk_define_cross_pack_size_(dots, i8, ampere, i8, i8, /*norm_value_type=*/u32, /*depth_simd_dimensions=*/64,
                           /*dimensions_per_value=*/1)
nk_define_cross_cuda_packed_shape_(dots, i8, ampere)
nk_define_cross_cuda_pack_(dots, i8, ampere, i8, i8, nk_load_b8_ampere_, /*norm_value_type=*/u32,
                           nk_dots_reduce_sumsq_i8_ampere_, /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_cuda_symmetric_(dots, i8, ampere, i8, i32, nk_dots_i8_multiply_ampere_, nk_cross_epilogue_i32_k,
                                /*output_scale=*/1.0f, /*dimensions_per_value=*/1)
nk_define_cross_cuda_packed_(dots, i8, ampere, i8, i8, i32, nk_dots_i8_multiply_ampere_, nk_cross_epilogue_i32_k,
                             /*output_scale=*/1.0f, /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)

#pragma endregion I8

#pragma region I4

nk_define_cross_pack_size_(dots, i4, ampere, i4x2, i4x2, /*norm_value_type=*/u32, /*depth_simd_dimensions=*/128,
                           /*dimensions_per_value=*/2)
nk_define_cross_cuda_packed_shape_(dots, i4, ampere)
nk_define_cross_cuda_pack_(dots, i4, ampere, i4x2, i4x2, nk_load_b8_ampere_, /*norm_value_type=*/u32,
                           nk_dots_reduce_sumsq_i4_ampere_, /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/2)
nk_define_cross_cuda_symmetric_(dots, i4, ampere, i4x2, i32, nk_dots_i4_multiply_ampere_, nk_cross_epilogue_i32_k,
                                /*output_scale=*/1.0f, /*dimensions_per_value=*/2)
nk_define_cross_cuda_packed_(dots, i4, ampere, i4x2, i4x2, i32, nk_dots_i4_multiply_ampere_, nk_cross_epilogue_i32_k,
                             /*output_scale=*/1.0f, /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/2)

#pragma endregion I4

#pragma region U8

nk_define_cross_pack_size_(dots, u8, ampere, u8, u8, /*norm_value_type=*/u32, /*depth_simd_dimensions=*/64,
                           /*dimensions_per_value=*/1)
nk_define_cross_cuda_packed_shape_(dots, u8, ampere)
nk_define_cross_cuda_pack_(dots, u8, ampere, u8, u8, nk_load_b8_ampere_, /*norm_value_type=*/u32,
                           nk_dots_reduce_sumsq_u8_ampere_, /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_cuda_symmetric_(dots, u8, ampere, u8, u32, nk_dots_u8_multiply_ampere_, nk_cross_epilogue_i32_k,
                                /*output_scale=*/1.0f, /*dimensions_per_value=*/1)
nk_define_cross_cuda_packed_(dots, u8, ampere, u8, u8, u32, nk_dots_u8_multiply_ampere_, nk_cross_epilogue_i32_k,
                             /*output_scale=*/1.0f, /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)

#pragma endregion U8

#pragma region U4

nk_define_cross_pack_size_(dots, u4, ampere, u4x2, u4x2, /*norm_value_type=*/u32, /*depth_simd_dimensions=*/128,
                           /*dimensions_per_value=*/2)
nk_define_cross_cuda_packed_shape_(dots, u4, ampere)
nk_define_cross_cuda_pack_(dots, u4, ampere, u4x2, u4x2, nk_load_b8_ampere_, /*norm_value_type=*/u32,
                           nk_dots_reduce_sumsq_u4_ampere_, /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/2)
nk_define_cross_cuda_symmetric_(dots, u4, ampere, u4x2, u32, nk_dots_u4_multiply_ampere_, nk_cross_epilogue_i32_k,
                                /*output_scale=*/1.0f, /*dimensions_per_value=*/2)
nk_define_cross_cuda_packed_(dots, u4, ampere, u4x2, u4x2, u32, nk_dots_u4_multiply_ampere_, nk_cross_epilogue_i32_k,
                             /*output_scale=*/1.0f, /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/2)

#pragma endregion U4

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_AMPERE
#endif // NK_DOTS_AMPERE_CUH
