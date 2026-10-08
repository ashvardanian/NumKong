/**
 *  @file include/numkong/dots/sapphireamx.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *  @brief SIMD-accelerated Batched Dot Products for Sapphire Rapids.
 *
 *  @sa include/numkong/dots.h
 *
 *  This file contains tiled matrix-multiplication kernels optimized for Intel AMX instructions,
 *  leveraging the new TMM registers on Intel Sapphire Rapids CPUs. Those are much larger than ZMM:
 *
 *  - BF16 tiles: 16 rows × 32 elements = 512 BF16 values = 1KB per tile
 *  - INT8 tiles: 16 rows × 64 elements = 1024 INT8 values = 1KB per tile
 *
 *  We typically use 4 registers for the 2 × 2 tile output for the matrix C accumulators, leaving 4
 *  other registers for parts of A and B matrices:
 *
 *  - TMM0, TMM1: A matrix tiles (row blocks i and i+16)
 *  - TMM2, TMM3: B matrix tiles (column blocks j and j+16)
 *  - TMM4-7: C accumulator tiles (2 × 2 output grid)
 *
 *  In most synthetic benchmarks there seems to be no major difference between aggregating into 1 or
 *  4 output tiles, implying the CPU's ability to internally pipeline the accumulation, so using 2 ×
 *  2 for outputs is more of a memory-bandwidth saving measure.
 *
 *  Lacking High Bandwidth Memory, the performance in GEMM-like BLAS workloads is dominated by
 *  memory bandwidth. Latency hiding is also extremely hard, heavily affecting performance numbers.
 *  For reference, Intel MKL SGEMM for FP32 inputs yields around 250 GigaOPS per core on Intel
 *  Sapphire Rapids, leveraging AVX-512. At the same time, for AMX:
 *
 *  - BF16 peak: ≈ 3 TeraOPS per core in theory, ≈ 500 GigaOPS per core in practice
 *  - INT8 peak: ≈ 6 TeraOPS per core in theory, ≈ 1000 GigaOPS per core in practice
 *
 *  Several optimizations are used across file:
 *
 *  - Pre-pack B matrix once for repeated inference (avoids runtime reordering)
 *  - Morton Z-curve tile ordering improves L2 cache hit rate by 5-25%
 *  - Use streaming stores for large C matrices to avoid cache pollution
 *
 *  @section amx_instructions Intel AMX Instructions (Sapphire Rapids+)
 *
 *  Tile configuration and data movement:
 *
 *  @verbatim
 *  Intrinsic                   Instruction                     Notes
 *  _tile_loadconfig            LDTILECFG (mem64)               Configure tile palette
 *  _tile_loadd                 TILELOADD (TMM, mem, stride)    Load tile from memory
 *  _tile_stored                TILESTORED (mem, TMM, stride)   Store tile to memory
 *  _tile_zero                  TILEZERO (TMM)                  Zero a tile register
 *  @endverbatim
 *
 *  BF16 matrix multiply (AMX-BF16):
 *
 *  @verbatim
 *  Intrinsic                   Instruction                     Operation
 *  _tile_dpbf16ps              TDPBF16PS (TMM, TMM, TMM)       C += A × B (bf16 → f32)
 *  @endverbatim
 *
 *  INT8 matrix multiply (AMX-INT8):
 *
 *  @verbatim
 *  Intrinsic                   Instruction                     Operation
 *  _tile_dpbssd                TDPBSSD (TMM, TMM, TMM)         C += A × B (i8 × i8 → i32)
 *  _tile_dpbsud                TDPBSUD (TMM, TMM, TMM)         C += A × B (i8 × u8 → i32)
 *  _tile_dpbusd                TDPBUSD (TMM, TMM, TMM)         C += A × B (u8 × i8 → i32)
 *  _tile_dpbuud                TDPBUUD (TMM, TMM, TMM)         C += A × B (u8 × u8 → u32)
 *  @endverbatim
 *
 *  AMX performance characteristics:
 *  - TDPBF16PS: 16 × 16 × 32 = 8192 BF16 MACs per instruction
 *  - TDPBSSD: 16 × 16 × 64 = 16384 INT8 MACs per instruction
 *  - Tile load latency is ~20-30 cycles; software pipelining essential
 *  - PDEP/PEXT used for Morton Z-curve encoding (BMI2): 2-3cy @ p1
 */
#ifndef NUMKONG_DOTS_SAPPHIREAMX_H
#define NUMKONG_DOTS_SAPPHIREAMX_H

#if NUMKONG_ARCH_X8664_
#if NUMKONG_ARCH_X8664_SAPPHIREAMX_

#include "numkong/cast/icelake.h" // For FP8 ↔ BF16 conversions
#include "numkong/dots/serial.h"  // `nk_cross_scaled_exact_wide_mxfp4_serial_`
#include "numkong/dots/skylake.h" // `nk_dots_reduce_sumsq_bf16_skylake_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(                                                                                            \
    __attribute__((target(                                                                                               \
        "avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512fp16,avx512vbmi,f16c,fma,bmi,bmi2,amx-tile,amx-bf16,amx-int8"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512fp16", "avx512vbmi", "f16c", "fma", \
                   "bmi", "bmi2", "amx-tile", "amx-bf16", "amx-int8")
#endif

/*  AMX-specific packed buffer header (64-byte aligned).
 *  Different from nk_dots_amx_packed_header_t as AMX uses tile-based layout. */
typedef struct {

    /** Columns, not padded, at offset 0, where every packed shape reader finds them. */
    nk_u32_t columns;

    /** Depth, not padded, at offset 4. */
    nk_u32_t depth;

    /** Full column tiles of 16 rows each. */
    nk_u32_t full_column_tiles;

    /** Depth tiles: 32 columns for BF16, 64 for I8. */
    nk_u32_t full_depth_tiles;

    /** Rows left after the full tiles, 0 to 15. */
    nk_u32_t column_remainder_count;

    /** Byte offset to the edge data region. */
    nk_u32_t column_edge_offset;

    /** Byte offset to the per-column norms, for angular and euclidean. */
    nk_u32_t norms_byte_offset;

    /** The packed operand's tensor scale, 1 when it has none, which every product multiplies in. */
    nk_f32_t tensor_scale;

    /** Zeroed; pads the header to 64 bytes. */
    nk_u32_t reserved[6];

    /** The capability that packed the buffer, which every consumer checks. */
    nk_capability_t capability;
} nk_dots_amx_packed_header_t;

/*  Composable tile structures for AMX operations.
 *  These enable reusable primitives and cross-correlation (A × Aᵀ) use cases. */

/*  BF16 A tile: 16 rows × 32 depth-elements, row-major layout.
 *  Loaded from source matrix, used as left operand in AMX multiply. */
typedef struct {
    nk_align_(64) nk_bf16_t data[16][32];
} nk_dots_bf16_a16x32_sapphireamx_t;

/*  BF16 B tile: 32 depth × 16 columns, pair-interleaved for TDPBF16PS.
 *  Access pattern: data[depth/2][column][depth%2] for logical B[depth, column].
 *  Pre-packed from column-major or transposed source. */
typedef struct {
    nk_align_(64) nk_bf16_t data[16][16][2];
} nk_dots_bf16_b32x16_sapphireamx_t;

/*  BF16 output state: 16 × 16 F32 accumulator tile.
 *  Holds partial sums during depth-dimension accumulation. */
typedef struct {
    nk_align_(64) nk_f32_t data[16][16];
} nk_dots_bf16_state_sapphireamx_t;

/*  INT8 A tile: 16 rows × 64 depth-elements, row-major layout. */
typedef struct {
    nk_align_(64) nk_i8_t data[16][64];
} nk_dots_i8_a16x64_sapphireamx_t;

/*  INT8 B tile: 64 depth × 16 columns, quad-interleaved for TDPBSSD.
 *  Access pattern: data[depth/4][column][depth%4] for logical B[depth, column]. */
typedef struct {
    nk_align_(64) nk_i8_t data[16][16][4];
} nk_dots_i8_b64x16_sapphireamx_t;

/*  INT8 output state: 16 × 16 I32 accumulator tile. */
typedef struct {
    nk_align_(64) nk_i32_t data[16][16];
} nk_dots_i8_state_sapphireamx_t;

/*  BF16 2 × 2 output state: 32 × 32 F32 output (4 accumulator tiles).
 *  Used for GEMM's 2 × 2 output blocking pattern. */
typedef struct {
    nk_dots_bf16_state_sapphireamx_t c[2][2]; // 4KB total
} nk_dots_bf16_state2x2_sapphireamx_t;

/*  INT8 2 × 2 output state: 32 × 32 I32 output (4 accumulator tiles). */
typedef struct {
    nk_dots_i8_state_sapphireamx_t c[2][2]; // 4KB total
} nk_dots_i8_state2x2_sapphireamx_t;

/*  UINT8 A tile: 16 rows × 64 depth-elements, row-major layout.
 *  Same layout as I8, different interpretation of signed vs unsigned. */
typedef struct {
    nk_align_(64) nk_u8_t data[16][64];
} nk_dots_u8_a16x64_sapphireamx_t;

/*  UINT8 B tile: 64 depth × 16 columns, quad-interleaved for TDPBUUD. */
typedef struct {
    nk_align_(64) nk_u8_t data[16][16][4];
} nk_dots_u8_b64x16_sapphireamx_t;

/*  UINT8 output state: 16 × 16 U32 accumulator tile. */
typedef struct {
    nk_align_(64) nk_u32_t data[16][16];
} nk_dots_u8_state_sapphireamx_t;

/*  UINT8 2 × 2 output state: 32 × 32 U32 output (4 accumulator tiles). */
typedef struct {
    nk_dots_u8_state_sapphireamx_t c[2][2]; // 4KB total
} nk_dots_u8_state2x2_sapphireamx_t;

/* Morton Z-curve encoding for cache-friendly tile traversal */
NUMKONG_INLINE nk_u64_t nk_morton_encode_sapphireamx_(nk_u32_t tile_row, nk_u32_t tile_col) {
    return _pdep_u64(tile_row, 0x5555555555555555ULL) | _pdep_u64(tile_col, 0xAAAAAAAAAAAAAAAAULL);
}

/* Configure AMX tile registers */
NUMKONG_INLINE void nk_amx_tile_configure_sapphireamx_(void) {
    nk_align_(64) nk_u8_t tile_config[64] = {0};
    tile_config[0] = 1; // palette 1 (standard tile configuration)

    nk_u16_t *bytes_per_row = (nk_u16_t *)&tile_config[16];
    nk_u8_t *rows_per_tile = &tile_config[48];

    for (int tile_idx = 0; tile_idx < 8; tile_idx++) {
        rows_per_tile[tile_idx] = 16; // 16 rows per tile
        bytes_per_row[tile_idx] = 64; // 64 bytes per row (1KB total)
    }
    _tile_loadconfig(tile_config);
}

/** Compiler memory barrier to ensure stores complete before AMX tile loads */
#if defined(_MSC_VER)
NUMKONG_INLINE void nk_compiler_barrier_sapphireamx_(void) { _ReadWriteBarrier(); }
#else
NUMKONG_INLINE void nk_compiler_barrier_sapphireamx_(void) { __asm__ volatile("" ::: "memory"); }
#endif

/* Initialize BF16 output state to zero */
NUMKONG_INLINE void nk_dots_bf16_init_sapphireamx_(nk_dots_bf16_state_sapphireamx_t *state) {
    __m512 zero_f32x16 = _mm512_setzero_ps();
    for (nk_size_t row_idx = 0; row_idx < 16; row_idx++) { _mm512_store_ps(state->data[row_idx], zero_f32x16); }
}

/* Load A tile from row-major source with masking for edge tiles */
NUMKONG_INLINE void nk_dots_bf16_load_a_sapphireamx_(    //
    nk_dots_bf16_a16x32_sapphireamx_t *a_tile,           //
    nk_bf16_t const *src, nk_size_t src_stride_elements, //
    nk_size_t valid_rows, nk_size_t valid_columns) {

    __mmask32 column_m32 = (valid_columns >= 32) ? 0xFFFFFFFF : ((__mmask32)1 << valid_columns) - 1;
    __m512i zero_i16x32 = _mm512_setzero_si512();

    for (nk_size_t row_idx = 0; row_idx < 16; row_idx++) {
        if (row_idx < valid_rows) {
            __m512i row_i16x32 = _mm512_maskz_loadu_epi16(column_m32, src + row_idx * src_stride_elements);
            _mm512_store_si512((__m512i *)a_tile->data[row_idx], row_i16x32);
        }
        else { _mm512_store_si512((__m512i *)a_tile->data[row_idx], zero_i16x32); }
    }
    nk_compiler_barrier_sapphireamx_();
}

/* Store state to output matrix with masking for edge tiles */
NUMKONG_INLINE void nk_dots_bf16_store_sapphireamx_( //
    nk_dots_bf16_state_sapphireamx_t const *state,   //
    nk_f32_t *dst, nk_size_t dst_stride_elements,    //
    nk_size_t valid_rows, nk_size_t valid_columns) {

    __mmask16 column_m16 = (valid_columns >= 16) ? 0xFFFF : ((__mmask16)1 << valid_columns) - 1;

    for (nk_size_t row_idx = 0; row_idx < valid_rows; row_idx++) {
        __m512 row_f32x16 = _mm512_load_ps(state->data[row_idx]);
        _mm512_mask_storeu_ps(dst + row_idx * dst_stride_elements, column_m16, row_f32x16);
    }
}

/** Store a 16 × 16 tile of 4-byte cells starting @p column_offset right of the diagonal, skipping
 *  the cells below it. */
NUMKONG_INLINE void nk_dots_symmetric_store_sapphireamx_(       //
    void const *tile, void *dst, nk_size_t dst_stride_elements, //
    nk_size_t valid_rows, nk_size_t valid_columns, nk_size_t column_offset) {

    __mmask16 column_m16 = (valid_columns >= 16) ? 0xFFFF : ((__mmask16)1 << valid_columns) - 1;

    for (nk_size_t row_idx = 0; row_idx < valid_rows; row_idx++) {
        __mmask16 upper_m16 = row_idx > column_offset ? column_m16 & (__mmask16)(0xFFFFu << (row_idx - column_offset))
                                                      : column_m16;
        __m512i row_b32x16 = _mm512_load_si512((nk_u32_t const *)tile + row_idx * 16);
        _mm512_mask_storeu_epi32((nk_u32_t *)dst + row_idx * dst_stride_elements, upper_m16, row_b32x16);
    }
}

/* Accumulate 3 A x B tile pairs into state using AMX TDPBF16PS */
NUMKONG_INLINE void nk_dots_bf16_update_sapphireamx_(  //
    nk_dots_bf16_state_sapphireamx_t *state,           //
    nk_dots_bf16_a16x32_sapphireamx_t const *a_tile_0, //
    nk_dots_bf16_a16x32_sapphireamx_t const *a_tile_1, //
    nk_dots_bf16_a16x32_sapphireamx_t const *a_tile_2, //
    nk_dots_bf16_b32x16_sapphireamx_t const *b_tile_0, //
    nk_dots_bf16_b32x16_sapphireamx_t const *b_tile_1, //
    nk_dots_bf16_b32x16_sapphireamx_t const *b_tile_2) {

    // Load all tiles into registers
    _tile_loadd(0, state->data, 64);    // C accumulator
    _tile_loadd(1, a_tile_0->data, 64); // A0
    _tile_loadd(2, a_tile_1->data, 64); // A1
    _tile_loadd(3, a_tile_2->data, 64); // A2
    _tile_loadd(4, b_tile_0->data, 64); // B0
    _tile_loadd(5, b_tile_1->data, 64); // B1
    _tile_loadd(6, b_tile_2->data, 64); // B2

    // Accumulate: C += A0 × B0 + A1 × B1 + A2 × B2
    _tile_dpbf16ps(0, 1, 4); // C += A0 × B0
    _tile_dpbf16ps(0, 2, 5); // C += A1 × B1
    _tile_dpbf16ps(0, 3, 6); // C += A2 × B2

    // Store result
    _tile_stored(0, state->data, 64);
}

/* Initialize INT8 output state to zero */
NUMKONG_INLINE void nk_dots_i8_init_sapphireamx_(nk_dots_i8_state_sapphireamx_t *state) {
    __m512i zero_i32x16 = _mm512_setzero_si512();
    for (nk_size_t row_idx = 0; row_idx < 16; row_idx++) {
        _mm512_store_si512((__m512i *)state->data[row_idx], zero_i32x16);
    }
}

/* Load A tile from row-major source with masking for edge tiles */
NUMKONG_INLINE void nk_dots_i8_load_a_sapphireamx_( //
    nk_dots_i8_a16x64_sapphireamx_t *a_tile,        //
    nk_i8_t const *src, nk_size_t src_stride,       //
    nk_size_t valid_rows, nk_size_t valid_columns) {

    __mmask64 column_m64 = (valid_columns >= 64) ? 0xFFFFFFFFFFFFFFFFULL : ((__mmask64)1 << valid_columns) - 1;
    __m512i zero_i8x64 = _mm512_setzero_si512();

    for (nk_size_t row_idx = 0; row_idx < 16; row_idx++) {
        if (row_idx < valid_rows) {
            __m512i row_i8x64 = _mm512_maskz_loadu_epi8(column_m64, src + row_idx * src_stride);
            _mm512_store_si512((__m512i *)a_tile->data[row_idx], row_i8x64);
        }
        else { _mm512_store_si512((__m512i *)a_tile->data[row_idx], zero_i8x64); }
    }
    nk_compiler_barrier_sapphireamx_();
}

/* Store state to output matrix with masking for edge tiles */
NUMKONG_INLINE void nk_dots_i8_store_sapphireamx_( //
    nk_dots_i8_state_sapphireamx_t const *state,   //
    nk_i32_t *dst, nk_size_t dst_stride_elements,  //
    nk_size_t valid_rows, nk_size_t valid_columns) {

    __mmask16 column_m16 = (valid_columns >= 16) ? 0xFFFF : ((__mmask16)1 << valid_columns) - 1;

    for (nk_size_t row_idx = 0; row_idx < valid_rows; row_idx++) {
        __m512i row_i32x16 = _mm512_load_si512((__m512i const *)state->data[row_idx]);
        _mm512_mask_storeu_epi32(dst + row_idx * dst_stride_elements, column_m16, row_i32x16);
    }
}

/* Accumulate 3 A x B tile pairs into state using AMX TDPBSSD */
NUMKONG_INLINE void nk_dots_i8_update_sapphireamx_(  //
    nk_dots_i8_state_sapphireamx_t *state,           //
    nk_dots_i8_a16x64_sapphireamx_t const *a_tile_0, //
    nk_dots_i8_a16x64_sapphireamx_t const *a_tile_1, //
    nk_dots_i8_a16x64_sapphireamx_t const *a_tile_2, //
    nk_dots_i8_b64x16_sapphireamx_t const *b_tile_0, //
    nk_dots_i8_b64x16_sapphireamx_t const *b_tile_1, //
    nk_dots_i8_b64x16_sapphireamx_t const *b_tile_2) {

    // Load all tiles into registers
    _tile_loadd(0, state->data, 64);    // C accumulator
    _tile_loadd(1, a_tile_0->data, 64); // A0
    _tile_loadd(2, a_tile_1->data, 64); // A1
    _tile_loadd(3, a_tile_2->data, 64); // A2
    _tile_loadd(4, b_tile_0->data, 64); // B0
    _tile_loadd(5, b_tile_1->data, 64); // B1
    _tile_loadd(6, b_tile_2->data, 64); // B2

    // Accumulate: C += A0 × B0 + A1 × B1 + A2 × B2
    _tile_dpbssd(0, 1, 4); // C += A0 × B0
    _tile_dpbssd(0, 2, 5); // C += A1 × B1
    _tile_dpbssd(0, 3, 6); // C += A2 × B2

    // Store result
    _tile_stored(0, state->data, 64);
}

/* Store BF16 2x2 state to output matrix with masking for edge tiles */
NUMKONG_INLINE void nk_dots_bf16_output2x2_sapphireamx_( //
    nk_dots_bf16_state2x2_sapphireamx_t const *state,    //
    nk_f32_t *dst, nk_size_t dst_stride_elements,        //
    nk_size_t valid_rows, nk_size_t valid_columns) {

    // Rows 0-15
    nk_size_t const rows_high = (valid_rows > 16) ? 16 : valid_rows;
    nk_size_t const columns_left = (valid_columns > 16) ? 16 : valid_columns;
    nk_size_t const columns_right = (valid_columns > 16) ? valid_columns - 16 : 0;

    if (rows_high > 0 && columns_left > 0)
        nk_dots_bf16_store_sapphireamx_(&state->c[0][0], dst, dst_stride_elements, rows_high, columns_left);
    if (rows_high > 0 && columns_right > 0)
        nk_dots_bf16_store_sapphireamx_(&state->c[0][1], dst + 16, dst_stride_elements, rows_high, columns_right);

    // Rows 16-31
    if (valid_rows > 16) {
        nk_size_t const rows_low = valid_rows - 16;
        nk_f32_t *dst_low = dst + 16 * dst_stride_elements;
        if (columns_left > 0)
            nk_dots_bf16_store_sapphireamx_(&state->c[1][0], dst_low, dst_stride_elements, rows_low, columns_left);
        if (columns_right > 0)
            nk_dots_bf16_store_sapphireamx_(&state->c[1][1], dst_low + 16, dst_stride_elements, rows_low,
                                            columns_right);
    }
}

/* Store INT8 2x2 state to output matrix with masking for edge tiles */
NUMKONG_INLINE void nk_dots_i8_output2x2_sapphireamx_( //
    nk_dots_i8_state2x2_sapphireamx_t const *state,    //
    nk_i32_t *dst, nk_size_t dst_stride_elements,      //
    nk_size_t valid_rows, nk_size_t valid_columns) {

    nk_size_t const rows_high = (valid_rows > 16) ? 16 : valid_rows;
    nk_size_t const columns_left = (valid_columns > 16) ? 16 : valid_columns;
    nk_size_t const columns_right = (valid_columns > 16) ? valid_columns - 16 : 0;

    if (rows_high > 0 && columns_left > 0)
        nk_dots_i8_store_sapphireamx_(&state->c[0][0], dst, dst_stride_elements, rows_high, columns_left);
    if (rows_high > 0 && columns_right > 0)
        nk_dots_i8_store_sapphireamx_(&state->c[0][1], dst + 16, dst_stride_elements, rows_high, columns_right);

    if (valid_rows > 16) {
        nk_size_t const rows_low = valid_rows - 16;
        nk_i32_t *dst_low = dst + 16 * dst_stride_elements;
        if (columns_left > 0)
            nk_dots_i8_store_sapphireamx_(&state->c[1][0], dst_low, dst_stride_elements, rows_low, columns_left);
        if (columns_right > 0)
            nk_dots_i8_store_sapphireamx_(&state->c[1][1], dst_low + 16, dst_stride_elements, rows_low, columns_right);
    }
}

/* Initialize UINT8 output state to zero */
NUMKONG_INLINE void nk_dots_u8_init_sapphireamx_(nk_dots_u8_state_sapphireamx_t *state) {
    nk_dots_i8_init_sapphireamx_((nk_dots_i8_state_sapphireamx_t *)state);
}

/* Load U8 A tile from row-major source with masking for edge tiles */
NUMKONG_INLINE void nk_dots_u8_load_a_sapphireamx_( //
    nk_dots_u8_a16x64_sapphireamx_t *a_tile,        //
    nk_u8_t const *src, nk_size_t src_stride,       //
    nk_size_t valid_rows, nk_size_t valid_columns) {
    nk_dots_i8_load_a_sapphireamx_(                //
        (nk_dots_i8_a16x64_sapphireamx_t *)a_tile, //
        (nk_i8_t const *)src, src_stride, valid_rows, valid_columns);
}

/* Store U8 state to output matrix with masking for edge tiles */
NUMKONG_INLINE void nk_dots_u8_store_sapphireamx_( //
    nk_dots_u8_state_sapphireamx_t const *state,   //
    nk_u32_t *dst, nk_size_t dst_stride_elements,  //
    nk_size_t valid_rows, nk_size_t valid_columns) {
    nk_dots_i8_store_sapphireamx_(                     //
        (nk_dots_i8_state_sapphireamx_t const *)state, //
        (nk_i32_t *)dst, dst_stride_elements, valid_rows, valid_columns);
}

/* Store UINT8 2x2 state to output matrix with masking for edge tiles */
NUMKONG_INLINE void nk_dots_u8_output2x2_sapphireamx_( //
    nk_dots_u8_state2x2_sapphireamx_t const *state,    //
    nk_u32_t *dst, nk_size_t dst_stride_elements,      //
    nk_size_t valid_rows, nk_size_t valid_columns) {
    nk_dots_i8_output2x2_sapphireamx_(                    //
        (nk_dots_i8_state2x2_sapphireamx_t const *)state, //
        (nk_i32_t *)dst, dst_stride_elements, valid_rows, valid_columns);
}

/* Pack U8 A transposed into B format */
NUMKONG_INLINE void nk_dots_pack_u8_transposed_sapphireamx_( //
    nk_dots_u8_a16x64_sapphireamx_t const *a_tile,           //
    nk_dots_u8_b64x16_sapphireamx_t *b_tile) {

    // Load all 16 rows - each row is 64 UINT8 = 64 bytes = 1 ZMM
    // Treat as 16 × 32-bit elements per row (each 32-bit = quad of UINT8)
    __m512i row00_i32x16 = _mm512_load_si512(&a_tile->data[0][0]);
    __m512i row01_i32x16 = _mm512_load_si512(&a_tile->data[1][0]);
    __m512i row02_i32x16 = _mm512_load_si512(&a_tile->data[2][0]);
    __m512i row03_i32x16 = _mm512_load_si512(&a_tile->data[3][0]);
    __m512i row04_i32x16 = _mm512_load_si512(&a_tile->data[4][0]);
    __m512i row05_i32x16 = _mm512_load_si512(&a_tile->data[5][0]);
    __m512i row06_i32x16 = _mm512_load_si512(&a_tile->data[6][0]);
    __m512i row07_i32x16 = _mm512_load_si512(&a_tile->data[7][0]);
    __m512i row08_i32x16 = _mm512_load_si512(&a_tile->data[8][0]);
    __m512i row09_i32x16 = _mm512_load_si512(&a_tile->data[9][0]);
    __m512i row10_i32x16 = _mm512_load_si512(&a_tile->data[10][0]);
    __m512i row11_i32x16 = _mm512_load_si512(&a_tile->data[11][0]);
    __m512i row12_i32x16 = _mm512_load_si512(&a_tile->data[12][0]);
    __m512i row13_i32x16 = _mm512_load_si512(&a_tile->data[13][0]);
    __m512i row14_i32x16 = _mm512_load_si512(&a_tile->data[14][0]);
    __m512i row15_i32x16 = _mm512_load_si512(&a_tile->data[15][0]);

    // 16×16 transpose of 32-bit elements using hierarchical unpacks
    // Stage 1: Unpack adjacent row pairs at 32-bit granularity
    __m512i t01_low_i32x16 = _mm512_unpacklo_epi32(row00_i32x16, row01_i32x16);
    __m512i t01_high_i32x16 = _mm512_unpackhi_epi32(row00_i32x16, row01_i32x16);
    __m512i t23_low_i32x16 = _mm512_unpacklo_epi32(row02_i32x16, row03_i32x16);
    __m512i t23_high_i32x16 = _mm512_unpackhi_epi32(row02_i32x16, row03_i32x16);
    __m512i t45_low_i32x16 = _mm512_unpacklo_epi32(row04_i32x16, row05_i32x16);
    __m512i t45_high_i32x16 = _mm512_unpackhi_epi32(row04_i32x16, row05_i32x16);
    __m512i t67_low_i32x16 = _mm512_unpacklo_epi32(row06_i32x16, row07_i32x16);
    __m512i t67_high_i32x16 = _mm512_unpackhi_epi32(row06_i32x16, row07_i32x16);
    __m512i t89_low_i32x16 = _mm512_unpacklo_epi32(row08_i32x16, row09_i32x16);
    __m512i t89_high_i32x16 = _mm512_unpackhi_epi32(row08_i32x16, row09_i32x16);
    __m512i tab_low_i32x16 = _mm512_unpacklo_epi32(row10_i32x16, row11_i32x16);
    __m512i tab_high_i32x16 = _mm512_unpackhi_epi32(row10_i32x16, row11_i32x16);
    __m512i tcd_low_i32x16 = _mm512_unpacklo_epi32(row12_i32x16, row13_i32x16);
    __m512i tcd_high_i32x16 = _mm512_unpackhi_epi32(row12_i32x16, row13_i32x16);
    __m512i tef_low_i32x16 = _mm512_unpacklo_epi32(row14_i32x16, row15_i32x16);
    __m512i tef_high_i32x16 = _mm512_unpackhi_epi32(row14_i32x16, row15_i32x16);

    // Stage 2: Unpack at 64-bit granularity
    __m512i u0123_ll_i32x16 = _mm512_unpacklo_epi64(t01_low_i32x16, t23_low_i32x16);
    __m512i u0123_lh_i32x16 = _mm512_unpackhi_epi64(t01_low_i32x16, t23_low_i32x16);
    __m512i u0123_hl_i32x16 = _mm512_unpacklo_epi64(t01_high_i32x16, t23_high_i32x16);
    __m512i u0123_hh_i32x16 = _mm512_unpackhi_epi64(t01_high_i32x16, t23_high_i32x16);
    __m512i u4567_ll_i32x16 = _mm512_unpacklo_epi64(t45_low_i32x16, t67_low_i32x16);
    __m512i u4567_lh_i32x16 = _mm512_unpackhi_epi64(t45_low_i32x16, t67_low_i32x16);
    __m512i u4567_hl_i32x16 = _mm512_unpacklo_epi64(t45_high_i32x16, t67_high_i32x16);
    __m512i u4567_hh_i32x16 = _mm512_unpackhi_epi64(t45_high_i32x16, t67_high_i32x16);
    __m512i u89ab_ll_i32x16 = _mm512_unpacklo_epi64(t89_low_i32x16, tab_low_i32x16);
    __m512i u89ab_lh_i32x16 = _mm512_unpackhi_epi64(t89_low_i32x16, tab_low_i32x16);
    __m512i u89ab_hl_i32x16 = _mm512_unpacklo_epi64(t89_high_i32x16, tab_high_i32x16);
    __m512i u89ab_hh_i32x16 = _mm512_unpackhi_epi64(t89_high_i32x16, tab_high_i32x16);
    __m512i ucdef_ll_i32x16 = _mm512_unpacklo_epi64(tcd_low_i32x16, tef_low_i32x16);
    __m512i ucdef_lh_i32x16 = _mm512_unpackhi_epi64(tcd_low_i32x16, tef_low_i32x16);
    __m512i ucdef_hl_i32x16 = _mm512_unpacklo_epi64(tcd_high_i32x16, tef_high_i32x16);
    __m512i ucdef_hh_i32x16 = _mm512_unpackhi_epi64(tcd_high_i32x16, tef_high_i32x16);

    // Stage 3: Shuffle 128-bit lanes
    __m512i v0_a_i32x16 = _mm512_shuffle_i32x4(u0123_ll_i32x16, u4567_ll_i32x16, 0x88);
    __m512i v0_b_i32x16 = _mm512_shuffle_i32x4(u0123_ll_i32x16, u4567_ll_i32x16, 0xDD);
    __m512i v1_a_i32x16 = _mm512_shuffle_i32x4(u0123_lh_i32x16, u4567_lh_i32x16, 0x88);
    __m512i v1_b_i32x16 = _mm512_shuffle_i32x4(u0123_lh_i32x16, u4567_lh_i32x16, 0xDD);
    __m512i v2_a_i32x16 = _mm512_shuffle_i32x4(u0123_hl_i32x16, u4567_hl_i32x16, 0x88);
    __m512i v2_b_i32x16 = _mm512_shuffle_i32x4(u0123_hl_i32x16, u4567_hl_i32x16, 0xDD);
    __m512i v3_a_i32x16 = _mm512_shuffle_i32x4(u0123_hh_i32x16, u4567_hh_i32x16, 0x88);
    __m512i v3_b_i32x16 = _mm512_shuffle_i32x4(u0123_hh_i32x16, u4567_hh_i32x16, 0xDD);
    __m512i v4_a_i32x16 = _mm512_shuffle_i32x4(u89ab_ll_i32x16, ucdef_ll_i32x16, 0x88);
    __m512i v4_b_i32x16 = _mm512_shuffle_i32x4(u89ab_ll_i32x16, ucdef_ll_i32x16, 0xDD);
    __m512i v5_a_i32x16 = _mm512_shuffle_i32x4(u89ab_lh_i32x16, ucdef_lh_i32x16, 0x88);
    __m512i v5_b_i32x16 = _mm512_shuffle_i32x4(u89ab_lh_i32x16, ucdef_lh_i32x16, 0xDD);
    __m512i v6_a_i32x16 = _mm512_shuffle_i32x4(u89ab_hl_i32x16, ucdef_hl_i32x16, 0x88);
    __m512i v6_b_i32x16 = _mm512_shuffle_i32x4(u89ab_hl_i32x16, ucdef_hl_i32x16, 0xDD);
    __m512i v7_a_i32x16 = _mm512_shuffle_i32x4(u89ab_hh_i32x16, ucdef_hh_i32x16, 0x88);
    __m512i v7_b_i32x16 = _mm512_shuffle_i32x4(u89ab_hh_i32x16, ucdef_hh_i32x16, 0xDD);

    // Stage 4: Final 256-bit shuffle to complete transpose
    __m512i out00_i32x16 = _mm512_shuffle_i32x4(v0_a_i32x16, v4_a_i32x16, 0x88);
    __m512i out01_i32x16 = _mm512_shuffle_i32x4(v1_a_i32x16, v5_a_i32x16, 0x88);
    __m512i out02_i32x16 = _mm512_shuffle_i32x4(v2_a_i32x16, v6_a_i32x16, 0x88);
    __m512i out03_i32x16 = _mm512_shuffle_i32x4(v3_a_i32x16, v7_a_i32x16, 0x88);
    __m512i out04_i32x16 = _mm512_shuffle_i32x4(v0_a_i32x16, v4_a_i32x16, 0xDD);
    __m512i out05_i32x16 = _mm512_shuffle_i32x4(v1_a_i32x16, v5_a_i32x16, 0xDD);
    __m512i out06_i32x16 = _mm512_shuffle_i32x4(v2_a_i32x16, v6_a_i32x16, 0xDD);
    __m512i out07_i32x16 = _mm512_shuffle_i32x4(v3_a_i32x16, v7_a_i32x16, 0xDD);
    __m512i out08_i32x16 = _mm512_shuffle_i32x4(v0_b_i32x16, v4_b_i32x16, 0x88);
    __m512i out09_i32x16 = _mm512_shuffle_i32x4(v1_b_i32x16, v5_b_i32x16, 0x88);
    __m512i out10_i32x16 = _mm512_shuffle_i32x4(v2_b_i32x16, v6_b_i32x16, 0x88);
    __m512i out11_i32x16 = _mm512_shuffle_i32x4(v3_b_i32x16, v7_b_i32x16, 0x88);
    __m512i out12_i32x16 = _mm512_shuffle_i32x4(v0_b_i32x16, v4_b_i32x16, 0xDD);
    __m512i out13_i32x16 = _mm512_shuffle_i32x4(v1_b_i32x16, v5_b_i32x16, 0xDD);
    __m512i out14_i32x16 = _mm512_shuffle_i32x4(v2_b_i32x16, v6_b_i32x16, 0xDD);
    __m512i out15_i32x16 = _mm512_shuffle_i32x4(v3_b_i32x16, v7_b_i32x16, 0xDD);

    // Store transposed results - each output row is one depth_group
    // Output layout: B.data[depth_group][column][quad] = 16 columns × 4 UINT8 = 64 bytes
    _mm512_store_si512(&b_tile->data[0][0][0], out00_i32x16);
    _mm512_store_si512(&b_tile->data[1][0][0], out01_i32x16);
    _mm512_store_si512(&b_tile->data[2][0][0], out02_i32x16);
    _mm512_store_si512(&b_tile->data[3][0][0], out03_i32x16);
    _mm512_store_si512(&b_tile->data[4][0][0], out08_i32x16);
    _mm512_store_si512(&b_tile->data[5][0][0], out09_i32x16);
    _mm512_store_si512(&b_tile->data[6][0][0], out10_i32x16);
    _mm512_store_si512(&b_tile->data[7][0][0], out11_i32x16);
    _mm512_store_si512(&b_tile->data[8][0][0], out04_i32x16);
    _mm512_store_si512(&b_tile->data[9][0][0], out05_i32x16);
    _mm512_store_si512(&b_tile->data[10][0][0], out06_i32x16);
    _mm512_store_si512(&b_tile->data[11][0][0], out07_i32x16);
    _mm512_store_si512(&b_tile->data[12][0][0], out12_i32x16);
    _mm512_store_si512(&b_tile->data[13][0][0], out13_i32x16);
    _mm512_store_si512(&b_tile->data[14][0][0], out14_i32x16);
    _mm512_store_si512(&b_tile->data[15][0][0], out15_i32x16);

    nk_compiler_barrier_sapphireamx_();
}

/* Accumulate 3 A x B tile pairs into state using AMX TDPBUUD */
NUMKONG_INLINE void nk_dots_u8_update_sapphireamx_(  //
    nk_dots_u8_state_sapphireamx_t *state,           //
    nk_dots_u8_a16x64_sapphireamx_t const *a_tile_0, //
    nk_dots_u8_a16x64_sapphireamx_t const *a_tile_1, //
    nk_dots_u8_a16x64_sapphireamx_t const *a_tile_2, //
    nk_dots_u8_b64x16_sapphireamx_t const *b_tile_0, //
    nk_dots_u8_b64x16_sapphireamx_t const *b_tile_1, //
    nk_dots_u8_b64x16_sapphireamx_t const *b_tile_2) {

    // Load all tiles into registers
    _tile_loadd(0, state->data, 64);    // C accumulator
    _tile_loadd(1, a_tile_0->data, 64); // A0
    _tile_loadd(2, a_tile_1->data, 64); // A1
    _tile_loadd(3, a_tile_2->data, 64); // A2
    _tile_loadd(4, b_tile_0->data, 64); // B0
    _tile_loadd(5, b_tile_1->data, 64); // B1
    _tile_loadd(6, b_tile_2->data, 64); // B2

    // Accumulate: C += A0 × B0 + A1 × B1 + A2 × B2
    _tile_dpbuud(0, 1, 4); // C += A0 × B0
    _tile_dpbuud(0, 2, 5); // C += A1 × B1
    _tile_dpbuud(0, 3, 6); // C += A2 × B2

    // Store result
    _tile_stored(0, state->data, 64);
}

/* Pack A transposed into B format for BF16 */
NUMKONG_INLINE void nk_dots_pack_bf16_transposed_sapphireamx_( //
    nk_dots_bf16_a16x32_sapphireamx_t const *a_tile,           //
    nk_dots_bf16_b32x16_sapphireamx_t *b_tile) {

    // Load all 16 rows - each row is 32 BF16 = 64 bytes = 1 ZMM
    // Treat as 16 × 32-bit elements per row (each 32-bit = pair of BF16)
    __m512i row00_i32x16 = _mm512_load_si512(&a_tile->data[0][0]);
    __m512i row01_i32x16 = _mm512_load_si512(&a_tile->data[1][0]);
    __m512i row02_i32x16 = _mm512_load_si512(&a_tile->data[2][0]);
    __m512i row03_i32x16 = _mm512_load_si512(&a_tile->data[3][0]);
    __m512i row04_i32x16 = _mm512_load_si512(&a_tile->data[4][0]);
    __m512i row05_i32x16 = _mm512_load_si512(&a_tile->data[5][0]);
    __m512i row06_i32x16 = _mm512_load_si512(&a_tile->data[6][0]);
    __m512i row07_i32x16 = _mm512_load_si512(&a_tile->data[7][0]);
    __m512i row08_i32x16 = _mm512_load_si512(&a_tile->data[8][0]);
    __m512i row09_i32x16 = _mm512_load_si512(&a_tile->data[9][0]);
    __m512i row10_i32x16 = _mm512_load_si512(&a_tile->data[10][0]);
    __m512i row11_i32x16 = _mm512_load_si512(&a_tile->data[11][0]);
    __m512i row12_i32x16 = _mm512_load_si512(&a_tile->data[12][0]);
    __m512i row13_i32x16 = _mm512_load_si512(&a_tile->data[13][0]);
    __m512i row14_i32x16 = _mm512_load_si512(&a_tile->data[14][0]);
    __m512i row15_i32x16 = _mm512_load_si512(&a_tile->data[15][0]);

    // 16×16 transpose of 32-bit elements using hierarchical unpacks
    // Stage 1: Unpack adjacent row pairs at 32-bit granularity
    __m512i t01_low_i32x16 = _mm512_unpacklo_epi32(row00_i32x16, row01_i32x16);
    __m512i t01_high_i32x16 = _mm512_unpackhi_epi32(row00_i32x16, row01_i32x16);
    __m512i t23_low_i32x16 = _mm512_unpacklo_epi32(row02_i32x16, row03_i32x16);
    __m512i t23_high_i32x16 = _mm512_unpackhi_epi32(row02_i32x16, row03_i32x16);
    __m512i t45_low_i32x16 = _mm512_unpacklo_epi32(row04_i32x16, row05_i32x16);
    __m512i t45_high_i32x16 = _mm512_unpackhi_epi32(row04_i32x16, row05_i32x16);
    __m512i t67_low_i32x16 = _mm512_unpacklo_epi32(row06_i32x16, row07_i32x16);
    __m512i t67_high_i32x16 = _mm512_unpackhi_epi32(row06_i32x16, row07_i32x16);
    __m512i t89_low_i32x16 = _mm512_unpacklo_epi32(row08_i32x16, row09_i32x16);
    __m512i t89_high_i32x16 = _mm512_unpackhi_epi32(row08_i32x16, row09_i32x16);
    __m512i tab_low_i32x16 = _mm512_unpacklo_epi32(row10_i32x16, row11_i32x16);
    __m512i tab_high_i32x16 = _mm512_unpackhi_epi32(row10_i32x16, row11_i32x16);
    __m512i tcd_low_i32x16 = _mm512_unpacklo_epi32(row12_i32x16, row13_i32x16);
    __m512i tcd_high_i32x16 = _mm512_unpackhi_epi32(row12_i32x16, row13_i32x16);
    __m512i tef_low_i32x16 = _mm512_unpacklo_epi32(row14_i32x16, row15_i32x16);
    __m512i tef_high_i32x16 = _mm512_unpackhi_epi32(row14_i32x16, row15_i32x16);

    // Stage 2: Unpack at 64-bit granularity
    __m512i u0123_ll_i32x16 = _mm512_unpacklo_epi64(t01_low_i32x16, t23_low_i32x16);
    __m512i u0123_lh_i32x16 = _mm512_unpackhi_epi64(t01_low_i32x16, t23_low_i32x16);
    __m512i u0123_hl_i32x16 = _mm512_unpacklo_epi64(t01_high_i32x16, t23_high_i32x16);
    __m512i u0123_hh_i32x16 = _mm512_unpackhi_epi64(t01_high_i32x16, t23_high_i32x16);
    __m512i u4567_ll_i32x16 = _mm512_unpacklo_epi64(t45_low_i32x16, t67_low_i32x16);
    __m512i u4567_lh_i32x16 = _mm512_unpackhi_epi64(t45_low_i32x16, t67_low_i32x16);
    __m512i u4567_hl_i32x16 = _mm512_unpacklo_epi64(t45_high_i32x16, t67_high_i32x16);
    __m512i u4567_hh_i32x16 = _mm512_unpackhi_epi64(t45_high_i32x16, t67_high_i32x16);
    __m512i u89ab_ll_i32x16 = _mm512_unpacklo_epi64(t89_low_i32x16, tab_low_i32x16);
    __m512i u89ab_lh_i32x16 = _mm512_unpackhi_epi64(t89_low_i32x16, tab_low_i32x16);
    __m512i u89ab_hl_i32x16 = _mm512_unpacklo_epi64(t89_high_i32x16, tab_high_i32x16);
    __m512i u89ab_hh_i32x16 = _mm512_unpackhi_epi64(t89_high_i32x16, tab_high_i32x16);
    __m512i ucdef_ll_i32x16 = _mm512_unpacklo_epi64(tcd_low_i32x16, tef_low_i32x16);
    __m512i ucdef_lh_i32x16 = _mm512_unpackhi_epi64(tcd_low_i32x16, tef_low_i32x16);
    __m512i ucdef_hl_i32x16 = _mm512_unpacklo_epi64(tcd_high_i32x16, tef_high_i32x16);
    __m512i ucdef_hh_i32x16 = _mm512_unpackhi_epi64(tcd_high_i32x16, tef_high_i32x16);

    // Stage 3: Shuffle 128-bit lanes using permute2x128 equivalent for 512-bit
    // Use shuffle_i32x4 to move 128-bit chunks
    __m512i v0_a_i32x16 = _mm512_shuffle_i32x4(u0123_ll_i32x16, u4567_ll_i32x16, 0x88); // lanes 0,2 from each
    __m512i v0_b_i32x16 = _mm512_shuffle_i32x4(u0123_ll_i32x16, u4567_ll_i32x16, 0xDD); // lanes 1,3 from each
    __m512i v1_a_i32x16 = _mm512_shuffle_i32x4(u0123_lh_i32x16, u4567_lh_i32x16, 0x88);
    __m512i v1_b_i32x16 = _mm512_shuffle_i32x4(u0123_lh_i32x16, u4567_lh_i32x16, 0xDD);
    __m512i v2_a_i32x16 = _mm512_shuffle_i32x4(u0123_hl_i32x16, u4567_hl_i32x16, 0x88);
    __m512i v2_b_i32x16 = _mm512_shuffle_i32x4(u0123_hl_i32x16, u4567_hl_i32x16, 0xDD);
    __m512i v3_a_i32x16 = _mm512_shuffle_i32x4(u0123_hh_i32x16, u4567_hh_i32x16, 0x88);
    __m512i v3_b_i32x16 = _mm512_shuffle_i32x4(u0123_hh_i32x16, u4567_hh_i32x16, 0xDD);
    __m512i v4_a_i32x16 = _mm512_shuffle_i32x4(u89ab_ll_i32x16, ucdef_ll_i32x16, 0x88);
    __m512i v4_b_i32x16 = _mm512_shuffle_i32x4(u89ab_ll_i32x16, ucdef_ll_i32x16, 0xDD);
    __m512i v5_a_i32x16 = _mm512_shuffle_i32x4(u89ab_lh_i32x16, ucdef_lh_i32x16, 0x88);
    __m512i v5_b_i32x16 = _mm512_shuffle_i32x4(u89ab_lh_i32x16, ucdef_lh_i32x16, 0xDD);
    __m512i v6_a_i32x16 = _mm512_shuffle_i32x4(u89ab_hl_i32x16, ucdef_hl_i32x16, 0x88);
    __m512i v6_b_i32x16 = _mm512_shuffle_i32x4(u89ab_hl_i32x16, ucdef_hl_i32x16, 0xDD);
    __m512i v7_a_i32x16 = _mm512_shuffle_i32x4(u89ab_hh_i32x16, ucdef_hh_i32x16, 0x88);
    __m512i v7_b_i32x16 = _mm512_shuffle_i32x4(u89ab_hh_i32x16, ucdef_hh_i32x16, 0xDD);

    // Stage 4: Final 256-bit shuffle to complete transpose
    __m512i out00_i32x16 = _mm512_shuffle_i32x4(v0_a_i32x16, v4_a_i32x16, 0x88);
    __m512i out01_i32x16 = _mm512_shuffle_i32x4(v1_a_i32x16, v5_a_i32x16, 0x88);
    __m512i out02_i32x16 = _mm512_shuffle_i32x4(v2_a_i32x16, v6_a_i32x16, 0x88);
    __m512i out03_i32x16 = _mm512_shuffle_i32x4(v3_a_i32x16, v7_a_i32x16, 0x88);
    __m512i out04_i32x16 = _mm512_shuffle_i32x4(v0_a_i32x16, v4_a_i32x16, 0xDD);
    __m512i out05_i32x16 = _mm512_shuffle_i32x4(v1_a_i32x16, v5_a_i32x16, 0xDD);
    __m512i out06_i32x16 = _mm512_shuffle_i32x4(v2_a_i32x16, v6_a_i32x16, 0xDD);
    __m512i out07_i32x16 = _mm512_shuffle_i32x4(v3_a_i32x16, v7_a_i32x16, 0xDD);
    __m512i out08_i32x16 = _mm512_shuffle_i32x4(v0_b_i32x16, v4_b_i32x16, 0x88);
    __m512i out09_i32x16 = _mm512_shuffle_i32x4(v1_b_i32x16, v5_b_i32x16, 0x88);
    __m512i out10_i32x16 = _mm512_shuffle_i32x4(v2_b_i32x16, v6_b_i32x16, 0x88);
    __m512i out11_i32x16 = _mm512_shuffle_i32x4(v3_b_i32x16, v7_b_i32x16, 0x88);
    __m512i out12_i32x16 = _mm512_shuffle_i32x4(v0_b_i32x16, v4_b_i32x16, 0xDD);
    __m512i out13_i32x16 = _mm512_shuffle_i32x4(v1_b_i32x16, v5_b_i32x16, 0xDD);
    __m512i out14_i32x16 = _mm512_shuffle_i32x4(v2_b_i32x16, v6_b_i32x16, 0xDD);
    __m512i out15_i32x16 = _mm512_shuffle_i32x4(v3_b_i32x16, v7_b_i32x16, 0xDD);

    // Store transposed results - each output row is one depth_group
    // Output layout: B.data[depth_group][column][pair] = 16 columns × 2 BF16 = 64 bytes
    _mm512_store_si512(&b_tile->data[0][0][0], out00_i32x16);
    _mm512_store_si512(&b_tile->data[1][0][0], out01_i32x16);
    _mm512_store_si512(&b_tile->data[2][0][0], out02_i32x16);
    _mm512_store_si512(&b_tile->data[3][0][0], out03_i32x16);
    _mm512_store_si512(&b_tile->data[4][0][0], out08_i32x16);
    _mm512_store_si512(&b_tile->data[5][0][0], out09_i32x16);
    _mm512_store_si512(&b_tile->data[6][0][0], out10_i32x16);
    _mm512_store_si512(&b_tile->data[7][0][0], out11_i32x16);
    _mm512_store_si512(&b_tile->data[8][0][0], out04_i32x16);
    _mm512_store_si512(&b_tile->data[9][0][0], out05_i32x16);
    _mm512_store_si512(&b_tile->data[10][0][0], out06_i32x16);
    _mm512_store_si512(&b_tile->data[11][0][0], out07_i32x16);
    _mm512_store_si512(&b_tile->data[12][0][0], out12_i32x16);
    _mm512_store_si512(&b_tile->data[13][0][0], out13_i32x16);
    _mm512_store_si512(&b_tile->data[14][0][0], out14_i32x16);
    _mm512_store_si512(&b_tile->data[15][0][0], out15_i32x16);

    nk_compiler_barrier_sapphireamx_();
}

/* Pack A transposed into B format for INT8 */
NUMKONG_INLINE void nk_dots_pack_i8_transposed_sapphireamx_( //
    nk_dots_i8_a16x64_sapphireamx_t const *a_tile,           //
    nk_dots_i8_b64x16_sapphireamx_t *b_tile) {

    // Load all 16 rows - each row is 64 INT8 = 64 bytes = 1 ZMM
    // Treat as 16 × 32-bit elements per row (each 32-bit = quad of INT8)
    __m512i row00_i32x16 = _mm512_load_si512(&a_tile->data[0][0]);
    __m512i row01_i32x16 = _mm512_load_si512(&a_tile->data[1][0]);
    __m512i row02_i32x16 = _mm512_load_si512(&a_tile->data[2][0]);
    __m512i row03_i32x16 = _mm512_load_si512(&a_tile->data[3][0]);
    __m512i row04_i32x16 = _mm512_load_si512(&a_tile->data[4][0]);
    __m512i row05_i32x16 = _mm512_load_si512(&a_tile->data[5][0]);
    __m512i row06_i32x16 = _mm512_load_si512(&a_tile->data[6][0]);
    __m512i row07_i32x16 = _mm512_load_si512(&a_tile->data[7][0]);
    __m512i row08_i32x16 = _mm512_load_si512(&a_tile->data[8][0]);
    __m512i row09_i32x16 = _mm512_load_si512(&a_tile->data[9][0]);
    __m512i row10_i32x16 = _mm512_load_si512(&a_tile->data[10][0]);
    __m512i row11_i32x16 = _mm512_load_si512(&a_tile->data[11][0]);
    __m512i row12_i32x16 = _mm512_load_si512(&a_tile->data[12][0]);
    __m512i row13_i32x16 = _mm512_load_si512(&a_tile->data[13][0]);
    __m512i row14_i32x16 = _mm512_load_si512(&a_tile->data[14][0]);
    __m512i row15_i32x16 = _mm512_load_si512(&a_tile->data[15][0]);

    // 16×16 transpose of 32-bit elements using hierarchical unpacks
    // Stage 1: Unpack adjacent row pairs at 32-bit granularity
    __m512i t01_low_i32x16 = _mm512_unpacklo_epi32(row00_i32x16, row01_i32x16);
    __m512i t01_high_i32x16 = _mm512_unpackhi_epi32(row00_i32x16, row01_i32x16);
    __m512i t23_low_i32x16 = _mm512_unpacklo_epi32(row02_i32x16, row03_i32x16);
    __m512i t23_high_i32x16 = _mm512_unpackhi_epi32(row02_i32x16, row03_i32x16);
    __m512i t45_low_i32x16 = _mm512_unpacklo_epi32(row04_i32x16, row05_i32x16);
    __m512i t45_high_i32x16 = _mm512_unpackhi_epi32(row04_i32x16, row05_i32x16);
    __m512i t67_low_i32x16 = _mm512_unpacklo_epi32(row06_i32x16, row07_i32x16);
    __m512i t67_high_i32x16 = _mm512_unpackhi_epi32(row06_i32x16, row07_i32x16);
    __m512i t89_low_i32x16 = _mm512_unpacklo_epi32(row08_i32x16, row09_i32x16);
    __m512i t89_high_i32x16 = _mm512_unpackhi_epi32(row08_i32x16, row09_i32x16);
    __m512i tab_low_i32x16 = _mm512_unpacklo_epi32(row10_i32x16, row11_i32x16);
    __m512i tab_high_i32x16 = _mm512_unpackhi_epi32(row10_i32x16, row11_i32x16);
    __m512i tcd_low_i32x16 = _mm512_unpacklo_epi32(row12_i32x16, row13_i32x16);
    __m512i tcd_high_i32x16 = _mm512_unpackhi_epi32(row12_i32x16, row13_i32x16);
    __m512i tef_low_i32x16 = _mm512_unpacklo_epi32(row14_i32x16, row15_i32x16);
    __m512i tef_high_i32x16 = _mm512_unpackhi_epi32(row14_i32x16, row15_i32x16);

    // Stage 2: Unpack at 64-bit granularity
    __m512i u0123_ll_i32x16 = _mm512_unpacklo_epi64(t01_low_i32x16, t23_low_i32x16);
    __m512i u0123_lh_i32x16 = _mm512_unpackhi_epi64(t01_low_i32x16, t23_low_i32x16);
    __m512i u0123_hl_i32x16 = _mm512_unpacklo_epi64(t01_high_i32x16, t23_high_i32x16);
    __m512i u0123_hh_i32x16 = _mm512_unpackhi_epi64(t01_high_i32x16, t23_high_i32x16);
    __m512i u4567_ll_i32x16 = _mm512_unpacklo_epi64(t45_low_i32x16, t67_low_i32x16);
    __m512i u4567_lh_i32x16 = _mm512_unpackhi_epi64(t45_low_i32x16, t67_low_i32x16);
    __m512i u4567_hl_i32x16 = _mm512_unpacklo_epi64(t45_high_i32x16, t67_high_i32x16);
    __m512i u4567_hh_i32x16 = _mm512_unpackhi_epi64(t45_high_i32x16, t67_high_i32x16);
    __m512i u89ab_ll_i32x16 = _mm512_unpacklo_epi64(t89_low_i32x16, tab_low_i32x16);
    __m512i u89ab_lh_i32x16 = _mm512_unpackhi_epi64(t89_low_i32x16, tab_low_i32x16);
    __m512i u89ab_hl_i32x16 = _mm512_unpacklo_epi64(t89_high_i32x16, tab_high_i32x16);
    __m512i u89ab_hh_i32x16 = _mm512_unpackhi_epi64(t89_high_i32x16, tab_high_i32x16);
    __m512i ucdef_ll_i32x16 = _mm512_unpacklo_epi64(tcd_low_i32x16, tef_low_i32x16);
    __m512i ucdef_lh_i32x16 = _mm512_unpackhi_epi64(tcd_low_i32x16, tef_low_i32x16);
    __m512i ucdef_hl_i32x16 = _mm512_unpacklo_epi64(tcd_high_i32x16, tef_high_i32x16);
    __m512i ucdef_hh_i32x16 = _mm512_unpackhi_epi64(tcd_high_i32x16, tef_high_i32x16);

    // Stage 3: Shuffle 128-bit lanes
    __m512i v0_a_i32x16 = _mm512_shuffle_i32x4(u0123_ll_i32x16, u4567_ll_i32x16, 0x88);
    __m512i v0_b_i32x16 = _mm512_shuffle_i32x4(u0123_ll_i32x16, u4567_ll_i32x16, 0xDD);
    __m512i v1_a_i32x16 = _mm512_shuffle_i32x4(u0123_lh_i32x16, u4567_lh_i32x16, 0x88);
    __m512i v1_b_i32x16 = _mm512_shuffle_i32x4(u0123_lh_i32x16, u4567_lh_i32x16, 0xDD);
    __m512i v2_a_i32x16 = _mm512_shuffle_i32x4(u0123_hl_i32x16, u4567_hl_i32x16, 0x88);
    __m512i v2_b_i32x16 = _mm512_shuffle_i32x4(u0123_hl_i32x16, u4567_hl_i32x16, 0xDD);
    __m512i v3_a_i32x16 = _mm512_shuffle_i32x4(u0123_hh_i32x16, u4567_hh_i32x16, 0x88);
    __m512i v3_b_i32x16 = _mm512_shuffle_i32x4(u0123_hh_i32x16, u4567_hh_i32x16, 0xDD);
    __m512i v4_a_i32x16 = _mm512_shuffle_i32x4(u89ab_ll_i32x16, ucdef_ll_i32x16, 0x88);
    __m512i v4_b_i32x16 = _mm512_shuffle_i32x4(u89ab_ll_i32x16, ucdef_ll_i32x16, 0xDD);
    __m512i v5_a_i32x16 = _mm512_shuffle_i32x4(u89ab_lh_i32x16, ucdef_lh_i32x16, 0x88);
    __m512i v5_b_i32x16 = _mm512_shuffle_i32x4(u89ab_lh_i32x16, ucdef_lh_i32x16, 0xDD);
    __m512i v6_a_i32x16 = _mm512_shuffle_i32x4(u89ab_hl_i32x16, ucdef_hl_i32x16, 0x88);
    __m512i v6_b_i32x16 = _mm512_shuffle_i32x4(u89ab_hl_i32x16, ucdef_hl_i32x16, 0xDD);
    __m512i v7_a_i32x16 = _mm512_shuffle_i32x4(u89ab_hh_i32x16, ucdef_hh_i32x16, 0x88);
    __m512i v7_b_i32x16 = _mm512_shuffle_i32x4(u89ab_hh_i32x16, ucdef_hh_i32x16, 0xDD);

    // Stage 4: Final 256-bit shuffle to complete transpose
    __m512i out00_i32x16 = _mm512_shuffle_i32x4(v0_a_i32x16, v4_a_i32x16, 0x88);
    __m512i out01_i32x16 = _mm512_shuffle_i32x4(v1_a_i32x16, v5_a_i32x16, 0x88);
    __m512i out02_i32x16 = _mm512_shuffle_i32x4(v2_a_i32x16, v6_a_i32x16, 0x88);
    __m512i out03_i32x16 = _mm512_shuffle_i32x4(v3_a_i32x16, v7_a_i32x16, 0x88);
    __m512i out04_i32x16 = _mm512_shuffle_i32x4(v0_a_i32x16, v4_a_i32x16, 0xDD);
    __m512i out05_i32x16 = _mm512_shuffle_i32x4(v1_a_i32x16, v5_a_i32x16, 0xDD);
    __m512i out06_i32x16 = _mm512_shuffle_i32x4(v2_a_i32x16, v6_a_i32x16, 0xDD);
    __m512i out07_i32x16 = _mm512_shuffle_i32x4(v3_a_i32x16, v7_a_i32x16, 0xDD);
    __m512i out08_i32x16 = _mm512_shuffle_i32x4(v0_b_i32x16, v4_b_i32x16, 0x88);
    __m512i out09_i32x16 = _mm512_shuffle_i32x4(v1_b_i32x16, v5_b_i32x16, 0x88);
    __m512i out10_i32x16 = _mm512_shuffle_i32x4(v2_b_i32x16, v6_b_i32x16, 0x88);
    __m512i out11_i32x16 = _mm512_shuffle_i32x4(v3_b_i32x16, v7_b_i32x16, 0x88);
    __m512i out12_i32x16 = _mm512_shuffle_i32x4(v0_b_i32x16, v4_b_i32x16, 0xDD);
    __m512i out13_i32x16 = _mm512_shuffle_i32x4(v1_b_i32x16, v5_b_i32x16, 0xDD);
    __m512i out14_i32x16 = _mm512_shuffle_i32x4(v2_b_i32x16, v6_b_i32x16, 0xDD);
    __m512i out15_i32x16 = _mm512_shuffle_i32x4(v3_b_i32x16, v7_b_i32x16, 0xDD);

    // Store transposed results - each output row is one depth_group
    // Output layout: B.data[depth_group][column][quad] = 16 columns × 4 INT8 = 64 bytes
    _mm512_store_si512(&b_tile->data[0][0][0], out00_i32x16);
    _mm512_store_si512(&b_tile->data[1][0][0], out01_i32x16);
    _mm512_store_si512(&b_tile->data[2][0][0], out02_i32x16);
    _mm512_store_si512(&b_tile->data[3][0][0], out03_i32x16);
    _mm512_store_si512(&b_tile->data[4][0][0], out08_i32x16);
    _mm512_store_si512(&b_tile->data[5][0][0], out09_i32x16);
    _mm512_store_si512(&b_tile->data[6][0][0], out10_i32x16);
    _mm512_store_si512(&b_tile->data[7][0][0], out11_i32x16);
    _mm512_store_si512(&b_tile->data[8][0][0], out04_i32x16);
    _mm512_store_si512(&b_tile->data[9][0][0], out05_i32x16);
    _mm512_store_si512(&b_tile->data[10][0][0], out06_i32x16);
    _mm512_store_si512(&b_tile->data[11][0][0], out07_i32x16);
    _mm512_store_si512(&b_tile->data[12][0][0], out12_i32x16);
    _mm512_store_si512(&b_tile->data[13][0][0], out13_i32x16);
    _mm512_store_si512(&b_tile->data[14][0][0], out14_i32x16);
    _mm512_store_si512(&b_tile->data[15][0][0], out15_i32x16);

    nk_compiler_barrier_sapphireamx_();
}

/** Copies @p count 16-bit values into @p destination, 32 at a time with masked loads and stores. */
NUMKONG_INLINE void nk_dots_copy_row_b16_sapphireamx_(void *destination, void const *source, nk_size_t count) {
    for (nk_size_t index = 0; index < count; index += 32) {
        nk_size_t const chunk = count - index < 32 ? count - index : 32;
        nk_b512_vec_t chunk_vec;
        nk_partial_load_b16x32_skylake_((nk_u16_t const *)source + index, &chunk_vec, chunk);
        nk_partial_store_b16x32_skylake_(&chunk_vec, (nk_u16_t *)destination + index, chunk);
    }
}

/** Copies @p count 8-bit values into @p destination, 64 at a time with masked loads and stores. */
NUMKONG_INLINE void nk_dots_copy_row_b8_sapphireamx_(void *destination, void const *source, nk_size_t count) {
    for (nk_size_t index = 0; index < count; index += 64) {
        nk_size_t const chunk = count - index < 64 ? count - index : 64;
        nk_b512_vec_t chunk_vec;
        nk_partial_load_b8x64_skylake_((nk_u8_t const *)source + index, &chunk_vec, chunk);
        nk_partial_store_b8x64_skylake_(&chunk_vec, (nk_u8_t *)destination + index, chunk);
    }
}

#pragma region F16 Floats

/** Bytes a pack of @p column_count BF16 columns of @p depth takes: header, AMX tiles, norms. */
NUMKONG_INLINE nk_size_t nk_dots_packed_bytes_bf16_sapphireamx_(nk_size_t column_count, nk_size_t depth) {
    nk_size_t const tmm_rows = 16;
    nk_size_t const tmm_columns = 32;
    nk_size_t const tile_bytes = 512 * sizeof(nk_bf16_t); // 16 × 32 × 2 = 1KB

    nk_size_t const full_column_tiles = column_count / tmm_rows;
    nk_size_t const tiles_along_depth = nk_size_divide_round_up_(depth, tmm_columns);
    nk_size_t const column_remainder_count = column_count - full_column_tiles * tmm_rows;

    // Header (64 bytes aligned)
    nk_size_t size = sizeof(nk_dots_amx_packed_header_t);

    // All tiles for full column rows (Morton-ordered, pair-interleaved, depth remainder zero-padded)
    size += full_column_tiles * tiles_along_depth * tile_bytes;

    // Column edge: remaining rows for all depth columns, stored row-major
    if (column_remainder_count > 0) size += column_remainder_count * depth * sizeof(nk_bf16_t);

    // Per-column norms for angular/euclidean distance (4 bytes each: f32 or u32)
    size += column_count * sizeof(nk_f32_t);
    return size;
}

/** BF16 GEMM of @p a rows against pre-packed B columns into F32 @p c,on AMX tiles. */
NUMKONG_INLINE nk_status_t nk_gemm_packed_bf16_sapphireamx_( //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,   //
    nk_size_t rows, nk_size_t column_count, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride) {
    nk_unused_(column_count);

    // Parse packed B header
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    if (header->capability != nk_cap_sapphireamx_k) return nk_pack_mismatch_k;
    nk_size_t const column_tiles_count = header->full_column_tiles;
    nk_size_t const depth_tiles_count = header->full_depth_tiles;
    nk_size_t const column_remainder_count = header->column_remainder_count;

    // Packed B data regions
    nk_bf16_t const *b_tiles_base = (nk_bf16_t const *)((char const *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_bf16_t const *col_edge_ptr = (nk_bf16_t const *)((char const *)b_packed + header->column_edge_offset);

    // Stride conversions
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_bf16_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);

    // Tile dimensions
    nk_size_t const tile_depth = 32; // depth elements per BF16 tile
    nk_size_t const tile_size = 512; // elements per packed tile
    nk_size_t const full_columns = column_tiles_count * 16;

    // Block counts (32 × 32 output blocks = 2 × 2 tiles)
    nk_size_t const row_blocks_count = nk_size_divide_round_up_(rows, 32);
    nk_size_t const col_blocks_count = column_tiles_count / 2;

    if (depth_tiles_count == 0) return nk_success_k;

    // Tile buffers for A (only used for edge tiles)
    nk_dots_bf16_a16x32_sapphireamx_t a_tile_top, a_tile_bottom;
    nk_dots_bf16_state2x2_sapphireamx_t c_accum_buffer;

    // Precompute: number of full depth-tiles (no masking needed)
    nk_size_t const full_depth_tiles_count = depth / tile_depth;
    nk_size_t const depth_remainder = depth % tile_depth;

    nk_amx_tile_configure_sapphireamx_();

    // Loop order: row_blocks outer, col_blocks inner - maximizes A tile L2 cache reuse
    // A tiles stay in L2 while we sweep through all col_blocks for a given row_block
    for (nk_size_t row_block_idx = 0; row_block_idx < row_blocks_count; row_block_idx++) {
        nk_size_t const row_block_start = row_block_idx * 32;
        nk_size_t const valid_rows_count = (row_block_start + 32 <= rows) ? 32 : (rows - row_block_start);
        nk_size_t const is_full_row_block = (valid_rows_count == 32);

        for (nk_size_t column_block_idx = 0; column_block_idx < col_blocks_count; column_block_idx++) {
            nk_size_t const col_block_start = column_block_idx * 32;
            nk_size_t const b_column_left_base = (column_block_idx * 2) * depth_tiles_count;
            nk_size_t const b_column_right_base = (column_block_idx * 2 + 1) * depth_tiles_count;

            // Zero accumulators (TMM4-7 stay resident across entire depth loop)
            _tile_zero(4);
            _tile_zero(5);
            _tile_zero(6);
            _tile_zero(7);

            // Fast path: full row-block with full depth-tiles → direct A load with 2-deep pipelining
            if (is_full_row_block && full_depth_tiles_count > 0) {
                nk_bf16_t const *a_top_base = a + row_block_start * a_stride_elements;
                nk_bf16_t const *a_bottom_base = a + (row_block_start + 16) * a_stride_elements;

                nk_dots_bf16_b32x16_sapphireamx_t const *b_tile_left =
                    (nk_dots_bf16_b32x16_sapphireamx_t const *)(b_tiles_base + b_column_left_base * tile_size);
                nk_dots_bf16_b32x16_sapphireamx_t const *b_tile_right =
                    (nk_dots_bf16_b32x16_sapphireamx_t const *)(b_tiles_base + b_column_right_base * tile_size);

                // Prologue: load first depth tile
                _tile_loadd(0, a_top_base, a_stride);
                _tile_loadd(1, a_bottom_base, a_stride);
                _tile_loadd(2, b_tile_left->data, 64);
                _tile_loadd(3, b_tile_right->data, 64);

                // Main loop: 2-deep software pipelining
                for (nk_size_t depth_tile_idx = 0; depth_tile_idx < full_depth_tiles_count - 1; depth_tile_idx++) {
                    nk_size_t const next_depth_offset = (depth_tile_idx + 1) * tile_depth;

                    _tile_dpbf16ps(4, 0, 2);
                    _tile_dpbf16ps(5, 0, 3);
                    _tile_dpbf16ps(6, 1, 2);
                    _tile_dpbf16ps(7, 1, 3);

                    _tile_loadd(0, a_top_base + next_depth_offset, a_stride);
                    _tile_loadd(1, a_bottom_base + next_depth_offset, a_stride);
                    b_tile_left = (nk_dots_bf16_b32x16_sapphireamx_t const *)(b_tiles_base + (b_column_left_base +
                                                                                              depth_tile_idx + 1) *
                                                                                                 tile_size);
                    b_tile_right = (nk_dots_bf16_b32x16_sapphireamx_t const *)(b_tiles_base + (b_column_right_base +
                                                                                               depth_tile_idx + 1) *
                                                                                                  tile_size);
                    _tile_loadd(2, b_tile_left->data, 64);
                    _tile_loadd(3, b_tile_right->data, 64);
                }

                // Epilogue: final depth tile
                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(5, 0, 3);
                _tile_dpbf16ps(6, 1, 2);
                _tile_dpbf16ps(7, 1, 3);

                // Handle partial depth-tile (if any)
                if (depth_remainder > 0) {
                    nk_size_t const depth_offset = full_depth_tiles_count * tile_depth;

                    nk_dots_bf16_load_a_sapphireamx_(&a_tile_top, a_top_base + depth_offset, a_stride_elements, 16,
                                                     depth_remainder);
                    nk_dots_bf16_load_a_sapphireamx_(&a_tile_bottom, a_bottom_base + depth_offset, a_stride_elements,
                                                     16, depth_remainder);

                    b_tile_left = (nk_dots_bf16_b32x16_sapphireamx_t const *)(b_tiles_base + (b_column_left_base +
                                                                                              full_depth_tiles_count) *
                                                                                                 tile_size);
                    b_tile_right = (nk_dots_bf16_b32x16_sapphireamx_t const *)(b_tiles_base + (b_column_right_base +
                                                                                               full_depth_tiles_count) *
                                                                                                  tile_size);

                    _tile_loadd(0, a_tile_top.data, 64);
                    _tile_loadd(1, a_tile_bottom.data, 64);
                    _tile_loadd(2, b_tile_left->data, 64);
                    _tile_loadd(3, b_tile_right->data, 64);

                    _tile_dpbf16ps(4, 0, 2);
                    _tile_dpbf16ps(5, 0, 3);
                    _tile_dpbf16ps(6, 1, 2);
                    _tile_dpbf16ps(7, 1, 3);
                }
            }
            // Full row-block but only partial depth tile (depth < tile_depth)
            else if (is_full_row_block) {
                nk_bf16_t const *a_top_base = a + row_block_start * a_stride_elements;
                nk_bf16_t const *a_bottom_base = a + (row_block_start + 16) * a_stride_elements;

                nk_dots_bf16_load_a_sapphireamx_(&a_tile_top, a_top_base, a_stride_elements, 16, depth_remainder);
                nk_dots_bf16_load_a_sapphireamx_(&a_tile_bottom, a_bottom_base, a_stride_elements, 16, depth_remainder);

                nk_dots_bf16_b32x16_sapphireamx_t const *b_tile_left =
                    (nk_dots_bf16_b32x16_sapphireamx_t const *)(b_tiles_base + b_column_left_base * tile_size);
                nk_dots_bf16_b32x16_sapphireamx_t const *b_tile_right =
                    (nk_dots_bf16_b32x16_sapphireamx_t const *)(b_tiles_base + b_column_right_base * tile_size);

                _tile_loadd(0, a_tile_top.data, 64);
                _tile_loadd(1, a_tile_bottom.data, 64);
                _tile_loadd(2, b_tile_left->data, 64);
                _tile_loadd(3, b_tile_right->data, 64);

                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(5, 0, 3);
                _tile_dpbf16ps(6, 1, 2);
                _tile_dpbf16ps(7, 1, 3);
            }
            // Slow path: edge row-block → buffered load with masking
            else {
                nk_size_t const rows_in_high_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
                nk_size_t const rows_in_low_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

                for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                    nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                    nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth
                                                                                            : depth_remainder;

                    nk_dots_bf16_load_a_sapphireamx_(&a_tile_top,
                                                     a + row_block_start * a_stride_elements + depth_offset,
                                                     a_stride_elements, rows_in_high_tile, valid_depth);
                    if (rows_in_low_tile > 0) {
                        nk_dots_bf16_load_a_sapphireamx_(&a_tile_bottom,
                                                         a + (row_block_start + 16) * a_stride_elements + depth_offset,
                                                         a_stride_elements, rows_in_low_tile, valid_depth);
                    }

                    nk_dots_bf16_b32x16_sapphireamx_t const *b_tile_left =
                        (nk_dots_bf16_b32x16_sapphireamx_t const *)(b_tiles_base +
                                                                    (b_column_left_base + depth_tile_idx) * tile_size);
                    nk_dots_bf16_b32x16_sapphireamx_t const *b_tile_right =
                        (nk_dots_bf16_b32x16_sapphireamx_t const *)(b_tiles_base +
                                                                    (b_column_right_base + depth_tile_idx) * tile_size);

                    _tile_loadd(0, a_tile_top.data, 64);
                    _tile_loadd(1, a_tile_bottom.data, 64);
                    _tile_loadd(2, b_tile_left->data, 64);
                    _tile_loadd(3, b_tile_right->data, 64);

                    _tile_dpbf16ps(4, 0, 2);
                    _tile_dpbf16ps(5, 0, 3);
                    _tile_dpbf16ps(6, 1, 2);
                    _tile_dpbf16ps(7, 1, 3);
                }
            }

            // Store accumulators to output (once per output block)
            if (is_full_row_block) {
                nk_f32_t *c_block = c + row_block_start * c_stride_elements + col_block_start;
                _tile_stored(4, c_block, c_stride);
                _tile_stored(5, c_block + 16, c_stride);
                _tile_stored(6, (nk_f32_t *)((char *)c_block + 16 * c_stride), c_stride);
                _tile_stored(7, (nk_f32_t *)((char *)c_block + 16 * c_stride) + 16, c_stride);
            }
            else {
                _tile_stored(4, c_accum_buffer.c[0][0].data, 64);
                _tile_stored(5, c_accum_buffer.c[0][1].data, 64);
                _tile_stored(6, c_accum_buffer.c[1][0].data, 64);
                _tile_stored(7, c_accum_buffer.c[1][1].data, 64);
                nk_dots_bf16_output2x2_sapphireamx_(&c_accum_buffer,
                                                    c + row_block_start * c_stride_elements + col_block_start,
                                                    c_stride_elements, valid_rows_count, 32);
            }
        }
    }

    // Handle odd column-tile (single 16-column tile if column_tiles_count is odd)
    if (column_tiles_count % 2 == 1) {
        nk_size_t const column_tile_idx = column_tiles_count - 1;
        nk_size_t const col_start = column_tile_idx * 16;
        nk_size_t const b_column_base = column_tile_idx * depth_tiles_count;

        for (nk_size_t row_block_idx = 0; row_block_idx < row_blocks_count; row_block_idx++) {
            nk_size_t const row_block_start = row_block_idx * 32;
            nk_size_t const valid_rows_count = (row_block_start + 32 <= rows) ? 32 : (rows - row_block_start);
            nk_size_t const rows_in_high_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
            nk_size_t const rows_in_low_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

            nk_dots_bf16_state_sapphireamx_t c_high_state, c_low_state;

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_bf16_load_a_sapphireamx_(&a_tile_top, a + row_block_start * a_stride_elements + depth_offset,
                                                 a_stride_elements, rows_in_high_tile, valid_depth);
                if (rows_in_low_tile > 0) {
                    nk_dots_bf16_load_a_sapphireamx_(&a_tile_bottom,
                                                     a + (row_block_start + 16) * a_stride_elements + depth_offset,
                                                     a_stride_elements, rows_in_low_tile, valid_depth);
                }

                nk_dots_bf16_b32x16_sapphireamx_t const *b_tile =
                    (nk_dots_bf16_b32x16_sapphireamx_t const *)(b_tiles_base +
                                                                (b_column_base + depth_tile_idx) * tile_size);

                _tile_loadd(0, a_tile_top.data, 64);
                _tile_loadd(1, a_tile_bottom.data, 64);
                _tile_loadd(2, b_tile->data, 64);

                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(6, 1, 2);
            }

            _tile_stored(4, c_high_state.data, 64);
            _tile_stored(6, c_low_state.data, 64);

            nk_dots_bf16_store_sapphireamx_(&c_high_state, c + row_block_start * c_stride_elements + col_start,
                                            c_stride_elements, rows_in_high_tile, 16);
            if (rows_in_low_tile > 0) {
                nk_dots_bf16_store_sapphireamx_(&c_low_state,
                                                c + (row_block_start + 16) * c_stride_elements + col_start,
                                                c_stride_elements, rows_in_low_tile, 16);
            }
        }
    }

    // Handle column-edge (remaining columns < 16) using AMX with partial tiles
    if (column_remainder_count > 0) {
        for (nk_size_t row_block_idx = 0; row_block_idx < row_blocks_count; row_block_idx++) {
            nk_size_t const row_block_start = row_block_idx * 32;
            nk_size_t const valid_rows_count = (row_block_start + 32 <= rows) ? 32 : (rows - row_block_start);
            nk_size_t const rows_in_high_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
            nk_size_t const rows_in_low_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

            nk_dots_bf16_state_sapphireamx_t c_high_state, c_low_state;
            nk_dots_bf16_a16x32_sapphireamx_t b_as_a;
            nk_dots_bf16_b32x16_sapphireamx_t b_tile;

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_bf16_load_a_sapphireamx_(&a_tile_top, a + row_block_start * a_stride_elements + depth_offset,
                                                 a_stride_elements, rows_in_high_tile, valid_depth);
                if (rows_in_low_tile > 0) {
                    nk_dots_bf16_load_a_sapphireamx_(&a_tile_bottom,
                                                     a + (row_block_start + 16) * a_stride_elements + depth_offset,
                                                     a_stride_elements, rows_in_low_tile, valid_depth);
                }

                nk_dots_bf16_load_a_sapphireamx_(&b_as_a, col_edge_ptr + depth_offset, depth, column_remainder_count,
                                                 valid_depth);
                nk_dots_pack_bf16_transposed_sapphireamx_(&b_as_a, &b_tile);

                _tile_loadd(0, a_tile_top.data, 64);
                _tile_loadd(1, a_tile_bottom.data, 64);
                _tile_loadd(2, b_tile.data, 64);

                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(6, 1, 2);
            }

            _tile_stored(4, c_high_state.data, 64);
            _tile_stored(6, c_low_state.data, 64);

            nk_dots_bf16_store_sapphireamx_(&c_high_state, c + row_block_start * c_stride_elements + full_columns,
                                            c_stride_elements, rows_in_high_tile, column_remainder_count);
            if (rows_in_low_tile > 0) {
                nk_dots_bf16_store_sapphireamx_(&c_low_state,
                                                c + (row_block_start + 16) * c_stride_elements + full_columns,
                                                c_stride_elements, rows_in_low_tile, column_remainder_count);
            }
        }
    }

    _tile_release();
    return nk_success_k;
}

/** BF16 Gram upper triangle of @p vectors on AMX tiles, rows @p rows_begin to @p rows_end. */
NUMKONG_INLINE nk_status_t nk_gram_bf16_sapphireamx_(                  //
    nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth, //
    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,       //
    nk_size_t rows_begin, nk_size_t rows_end) {
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));

    nk_size_t const stride_elements = stride / sizeof(nk_bf16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);

    // Handle row slicing: compute rows [rows_begin, rows_end)
    rows_end = nk_min_of_two(rows_end, vector_count);

    // Round depth up to multiple of 96 (3 tiles × 32 elements)
    nk_size_t const depth_tiles = nk_size_divide_round_up_(depth, 32);
    nk_size_t const depth_tile_groups = nk_size_divide_round_up_(depth_tiles, 3);

    nk_dots_bf16_a16x32_sapphireamx_t a_tiles[3];
    nk_dots_bf16_a16x32_sapphireamx_t b_src_tiles[3];
    nk_dots_bf16_b32x16_sapphireamx_t b_tiles[3];
    nk_dots_bf16_state_sapphireamx_t state;

    nk_amx_tile_configure_sapphireamx_();

    for (nk_size_t row_tile = rows_begin; row_tile < rows_end; row_tile += 16) {
        nk_size_t const valid_rows = (row_tile + 16 <= rows_end) ? 16 : (rows_end - row_tile);

        for (nk_size_t col_tile = row_tile; col_tile < vector_count; col_tile += 16) {
            nk_size_t const valid_columns = (col_tile + 16 <= vector_count) ? 16 : (vector_count - col_tile);

            nk_dots_bf16_init_sapphireamx_(&state);

            for (nk_size_t depth_group_idx = 0; depth_group_idx < depth_tile_groups; depth_group_idx++) {
                nk_size_t const depth_base = depth_group_idx * 96;

                for (int tile_idx = 0; tile_idx < 3; tile_idx++) {
                    nk_size_t const depth_start = depth_base + tile_idx * 32;
                    nk_size_t const valid_depth = (depth_start + 32 <= depth)
                                                      ? 32
                                                      : (depth > depth_start ? depth - depth_start : 0);

                    nk_dots_bf16_load_a_sapphireamx_(                       //
                        &a_tiles[tile_idx],                                 //
                        vectors + row_tile * stride_elements + depth_start, //
                        stride_elements, valid_rows, valid_depth);

                    if (row_tile == col_tile && valid_rows == valid_columns) {
                        nk_dots_pack_bf16_transposed_sapphireamx_(&a_tiles[tile_idx], &b_tiles[tile_idx]);
                    }
                    else {
                        nk_dots_bf16_load_a_sapphireamx_(                       //
                            &b_src_tiles[tile_idx],                             //
                            vectors + col_tile * stride_elements + depth_start, //
                            stride_elements, valid_columns, valid_depth);
                        nk_dots_pack_bf16_transposed_sapphireamx_(&b_src_tiles[tile_idx], &b_tiles[tile_idx]);
                    }
                }

                nk_dots_bf16_update_sapphireamx_( //
                    &state, &a_tiles[0], &a_tiles[1], &a_tiles[2], &b_tiles[0], &b_tiles[1], &b_tiles[2]);
            }

            nk_dots_symmetric_store_sapphireamx_(                                  //
                state.data, result + row_tile * result_stride_elements + col_tile, //
                result_stride_elements, valid_rows, valid_columns, col_tile - row_tile);
        }
    }
    return nk_success_k;
}

#pragma endregion F16 Floats

#pragma region Signed Integers

/** Bytes a pack of @p column_count I8 columns of @p depth takes: header, AMX tiles, norms. */
NUMKONG_INLINE nk_size_t nk_dots_packed_bytes_i8_sapphireamx_(nk_size_t column_count, nk_size_t depth) {
    nk_size_t const tmm_rows = 16;
    nk_size_t const tmm_columns = 64;
    nk_size_t const tile_bytes = 1024 * sizeof(nk_i8_t); // 16 × 64×1 = 1KB

    nk_size_t const full_column_tiles = column_count / tmm_rows;
    nk_size_t const tiles_along_depth = nk_size_divide_round_up_(depth, tmm_columns);
    nk_size_t const column_remainder_count = column_count - full_column_tiles * tmm_rows;

    // Header (64 bytes aligned)
    nk_size_t size = sizeof(nk_dots_amx_packed_header_t);

    // All tiles for full column rows (Morton-ordered, quad-interleaved, depth remainder zero-padded)
    size += full_column_tiles * tiles_along_depth * tile_bytes;

    // Column edge: remaining rows for all depth columns, stored row-major
    if (column_remainder_count > 0) size += column_remainder_count * depth * sizeof(nk_i8_t);

    // Per-column norms for angular/euclidean distance (4 bytes each: f32 or u32)
    size += column_count * sizeof(nk_u32_t);
    return size;
}

/** I8 GEMM of @p a rows against pre-packed B columns into I32 @p c,on AMX tiles. */
NUMKONG_INLINE nk_status_t nk_gemm_packed_i8_sapphireamx_( //
    nk_i8_t const *a, void const *b_packed, nk_i32_t *c,   //
    nk_size_t rows, nk_size_t column_count, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride) {
    nk_unused_(column_count);

    // Parse packed B header
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    if (header->capability != nk_cap_sapphireamx_k) return nk_pack_mismatch_k;
    nk_size_t const column_tiles_count = header->full_column_tiles;
    nk_size_t const depth_tiles_count = header->full_depth_tiles;
    nk_size_t const column_remainder_count = header->column_remainder_count;

    // Packed B data regions
    nk_i8_t const *b_tiles_base = (nk_i8_t const *)((char const *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_i8_t const *col_edge_ptr = (nk_i8_t const *)((char const *)b_packed + header->column_edge_offset);

    // Stride conversions
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_i32_t);

    // Tile dimensions
    nk_size_t const tile_depth = 64;  // depth elements per INT8 tile
    nk_size_t const tile_size = 1024; // bytes per packed tile
    nk_size_t const full_columns = column_tiles_count * 16;

    // Block counts (32 × 32 output blocks = 2 × 2 tiles)
    nk_size_t const row_blocks_count = nk_size_divide_round_up_(rows, 32);
    nk_size_t const col_blocks_count = column_tiles_count / 2;

    if (depth_tiles_count == 0) return nk_success_k;

    // Tile buffers for A (only used for edge tiles)
    nk_dots_i8_a16x64_sapphireamx_t a_tile_top, a_tile_bottom;
    nk_dots_i8_state2x2_sapphireamx_t c_accum_buffer;

    // Precompute: number of full depth-tiles (no masking needed)
    nk_size_t const full_depth_tiles_count = depth / tile_depth;
    nk_size_t const depth_remainder = depth % tile_depth;

    nk_amx_tile_configure_sapphireamx_();

    // Process all 32 × 32 row × column blocks (including partial edge blocks)
    for (nk_size_t row_block_idx = 0; row_block_idx < row_blocks_count; row_block_idx++) {
        nk_size_t const row_block_start = row_block_idx * 32;
        nk_size_t const valid_rows_count = (row_block_start + 32 <= rows) ? 32 : (rows - row_block_start);
        nk_size_t const is_full_row_block = (valid_rows_count == 32);

        // Process full column-blocks (pairs of 16-column tiles = 32 columns)
        for (nk_size_t column_block_idx = 0; column_block_idx < col_blocks_count; column_block_idx++) {
            nk_size_t const col_block_start = column_block_idx * 32;

            // B tile base indices (linear layout: col_tile × depth_tiles_count + depth_tile)
            nk_size_t const b_column_left_base = (column_block_idx * 2) * depth_tiles_count;
            nk_size_t const b_column_right_base = (column_block_idx * 2 + 1) * depth_tiles_count;

            // Zero accumulators (TMM4-7 stay resident across entire depth loop)
            _tile_zero(4); // C[upper, left]
            _tile_zero(5); // C[upper, right]
            _tile_zero(6); // C[lower, left]
            _tile_zero(7); // C[lower, right]

            // Fast path: full row-block with full depth-tiles → direct A load with 2-deep pipelining
            if (is_full_row_block && full_depth_tiles_count > 0) {
                // A row pointers for direct load
                nk_i8_t const *a_top_base = a + row_block_start * a_stride;
                nk_i8_t const *a_bottom_base = a + (row_block_start + 16) * a_stride;

                // B tile pointers
                nk_dots_i8_b64x16_sapphireamx_t const *b_tile_left =
                    (nk_dots_i8_b64x16_sapphireamx_t const *)(b_tiles_base + b_column_left_base * tile_size);
                nk_dots_i8_b64x16_sapphireamx_t const *b_tile_right =
                    (nk_dots_i8_b64x16_sapphireamx_t const *)(b_tiles_base + b_column_right_base * tile_size);

                // Prologue: load first depth tile into TMM0-3
                _tile_loadd(0, a_top_base, a_stride);
                _tile_loadd(1, a_bottom_base, a_stride);
                _tile_loadd(2, b_tile_left->data, 64);
                _tile_loadd(3, b_tile_right->data, 64);

                // Main loop: 2-deep software pipelining (compute current while loading next)
                for (nk_size_t depth_tile_idx = 0; depth_tile_idx < full_depth_tiles_count - 1; depth_tile_idx++) {
                    nk_size_t const next_depth_offset = (depth_tile_idx + 1) * tile_depth;

                    _tile_dpbssd(4, 0, 2);
                    _tile_dpbssd(5, 0, 3);
                    _tile_dpbssd(6, 1, 2);
                    _tile_dpbssd(7, 1, 3);

                    _tile_loadd(0, a_top_base + next_depth_offset, a_stride);
                    _tile_loadd(1, a_bottom_base + next_depth_offset, a_stride);
                    b_tile_left = (nk_dots_i8_b64x16_sapphireamx_t const *)(b_tiles_base +
                                                                            (b_column_left_base + depth_tile_idx + 1) *
                                                                                tile_size);
                    b_tile_right = (nk_dots_i8_b64x16_sapphireamx_t const *)(b_tiles_base + (b_column_right_base +
                                                                                             depth_tile_idx + 1) *
                                                                                                tile_size);
                    _tile_loadd(2, b_tile_left->data, 64);
                    _tile_loadd(3, b_tile_right->data, 64);
                }

                // Epilogue: final depth tile (no next to load)
                _tile_dpbssd(4, 0, 2);
                _tile_dpbssd(5, 0, 3);
                _tile_dpbssd(6, 1, 2);
                _tile_dpbssd(7, 1, 3);

                // Handle partial depth-tile (if any) with buffered load
                if (depth_remainder > 0) {
                    nk_size_t const depth_offset = full_depth_tiles_count * tile_depth;

                    nk_dots_i8_load_a_sapphireamx_(&a_tile_top, a_top_base + depth_offset, a_stride, 16,
                                                   depth_remainder);
                    nk_dots_i8_load_a_sapphireamx_(&a_tile_bottom, a_bottom_base + depth_offset, a_stride, 16,
                                                   depth_remainder);

                    b_tile_left = (nk_dots_i8_b64x16_sapphireamx_t const *)(b_tiles_base + (b_column_left_base +
                                                                                            full_depth_tiles_count) *
                                                                                               tile_size);
                    b_tile_right = (nk_dots_i8_b64x16_sapphireamx_t const *)(b_tiles_base + (b_column_right_base +
                                                                                             full_depth_tiles_count) *
                                                                                                tile_size);

                    _tile_loadd(0, a_tile_top.data, 64);
                    _tile_loadd(1, a_tile_bottom.data, 64);
                    _tile_loadd(2, b_tile_left->data, 64);
                    _tile_loadd(3, b_tile_right->data, 64);

                    _tile_dpbssd(4, 0, 2);
                    _tile_dpbssd(5, 0, 3);
                    _tile_dpbssd(6, 1, 2);
                    _tile_dpbssd(7, 1, 3);
                }
            }
            // Full row-block but only partial depth tile (depth < tile_depth)
            else if (is_full_row_block) {
                nk_i8_t const *a_top_base = a + row_block_start * a_stride;
                nk_i8_t const *a_bottom_base = a + (row_block_start + 16) * a_stride;

                nk_dots_i8_load_a_sapphireamx_(&a_tile_top, a_top_base, a_stride, 16, depth_remainder);
                nk_dots_i8_load_a_sapphireamx_(&a_tile_bottom, a_bottom_base, a_stride, 16, depth_remainder);

                nk_dots_i8_b64x16_sapphireamx_t const *b_tile_left =
                    (nk_dots_i8_b64x16_sapphireamx_t const *)(b_tiles_base + b_column_left_base * tile_size);
                nk_dots_i8_b64x16_sapphireamx_t const *b_tile_right =
                    (nk_dots_i8_b64x16_sapphireamx_t const *)(b_tiles_base + b_column_right_base * tile_size);

                _tile_loadd(0, a_tile_top.data, 64);
                _tile_loadd(1, a_tile_bottom.data, 64);
                _tile_loadd(2, b_tile_left->data, 64);
                _tile_loadd(3, b_tile_right->data, 64);

                _tile_dpbssd(4, 0, 2);
                _tile_dpbssd(5, 0, 3);
                _tile_dpbssd(6, 1, 2);
                _tile_dpbssd(7, 1, 3);
            }
            // Slow path: edge row-block → always use buffered load with masking
            else {
                nk_size_t const rows_in_high_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
                nk_size_t const rows_in_low_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

                for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                    nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                    nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth
                                                                                            : depth_remainder;

                    nk_dots_i8_load_a_sapphireamx_(&a_tile_top, a + row_block_start * a_stride + depth_offset, a_stride,
                                                   rows_in_high_tile, valid_depth);
                    if (rows_in_low_tile > 0) {
                        nk_dots_i8_load_a_sapphireamx_(&a_tile_bottom,
                                                       a + (row_block_start + 16) * a_stride + depth_offset, a_stride,
                                                       rows_in_low_tile, valid_depth);
                    }

                    nk_dots_i8_b64x16_sapphireamx_t const *b_tile_left =
                        (nk_dots_i8_b64x16_sapphireamx_t const *)(b_tiles_base +
                                                                  (b_column_left_base + depth_tile_idx) * tile_size);
                    nk_dots_i8_b64x16_sapphireamx_t const *b_tile_right =
                        (nk_dots_i8_b64x16_sapphireamx_t const *)(b_tiles_base +
                                                                  (b_column_right_base + depth_tile_idx) * tile_size);

                    _tile_loadd(0, a_tile_top.data, 64);
                    _tile_loadd(1, a_tile_bottom.data, 64);
                    _tile_loadd(2, b_tile_left->data, 64);
                    _tile_loadd(3, b_tile_right->data, 64);

                    _tile_dpbssd(4, 0, 2);
                    _tile_dpbssd(5, 0, 3);
                    _tile_dpbssd(6, 1, 2);
                    _tile_dpbssd(7, 1, 3);
                }
            }

            // Store accumulators to output (once per output block, not per depth tile)
            if (is_full_row_block) {
                nk_i32_t *c_block = c + row_block_start * c_stride_elements + col_block_start;
                _tile_stored(4, c_block, c_stride);
                _tile_stored(5, c_block + 16, c_stride);
                _tile_stored(6, (nk_i32_t *)((char *)c_block + 16 * c_stride), c_stride);
                _tile_stored(7, (nk_i32_t *)((char *)c_block + 16 * c_stride) + 16, c_stride);
            }
            else {
                // Slow path: edge row-block needs masked output
                _tile_stored(4, c_accum_buffer.c[0][0].data, 64);
                _tile_stored(5, c_accum_buffer.c[0][1].data, 64);
                _tile_stored(6, c_accum_buffer.c[1][0].data, 64);
                _tile_stored(7, c_accum_buffer.c[1][1].data, 64);
                nk_dots_i8_output2x2_sapphireamx_(&c_accum_buffer,
                                                  c + row_block_start * c_stride_elements + col_block_start,
                                                  c_stride_elements, valid_rows_count, 32);
            }
        }

        // Handle odd column-tile (single 16-column tile if column_tiles_count is odd)
        if (column_tiles_count % 2 == 1) {
            nk_size_t const column_tile_idx = column_tiles_count - 1;
            nk_size_t const col_start = column_tile_idx * 16;
            nk_size_t const b_column_base = column_tile_idx * depth_tiles_count;
            nk_size_t const rows_in_high_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
            nk_size_t const rows_in_low_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

            // Use 1 × 2 blocking for single column-tile (2 row-tiles × 1 column-tile)
            nk_dots_i8_state_sapphireamx_t c_high_state, c_low_state;

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_i8_load_a_sapphireamx_(&a_tile_top, a + row_block_start * a_stride + depth_offset, a_stride,
                                               rows_in_high_tile, valid_depth);
                if (rows_in_low_tile > 0) {
                    nk_dots_i8_load_a_sapphireamx_(&a_tile_bottom, a + (row_block_start + 16) * a_stride + depth_offset,
                                                   a_stride, rows_in_low_tile, valid_depth);
                }

                nk_dots_i8_b64x16_sapphireamx_t const *b_tile =
                    (nk_dots_i8_b64x16_sapphireamx_t const *)(b_tiles_base +
                                                              (b_column_base + depth_tile_idx) * tile_size);

                _tile_loadd(0, a_tile_top.data, 64);
                _tile_loadd(1, a_tile_bottom.data, 64);
                _tile_loadd(2, b_tile->data, 64);

                _tile_dpbssd(4, 0, 2);
                _tile_dpbssd(6, 1, 2);
            }

            _tile_stored(4, c_high_state.data, 64);
            _tile_stored(6, c_low_state.data, 64);

            nk_dots_i8_store_sapphireamx_(&c_high_state, c + row_block_start * c_stride_elements + col_start,
                                          c_stride_elements, rows_in_high_tile, 16);
            if (rows_in_low_tile > 0) {
                nk_dots_i8_store_sapphireamx_(&c_low_state, c + (row_block_start + 16) * c_stride_elements + col_start,
                                              c_stride_elements, rows_in_low_tile, 16);
            }
        }

        // Handle column-edge (remaining columns < 16) using AMX with partial tiles
        if (column_remainder_count > 0) {
            nk_size_t const rows_in_high_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
            nk_size_t const rows_in_low_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

            nk_dots_i8_state_sapphireamx_t c_high_state, c_low_state;
            nk_dots_i8_a16x64_sapphireamx_t b_as_a;
            nk_dots_i8_b64x16_sapphireamx_t b_tile;

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                // Load A tiles
                nk_dots_i8_load_a_sapphireamx_(&a_tile_top, a + row_block_start * a_stride + depth_offset, a_stride,
                                               rows_in_high_tile, valid_depth);
                if (rows_in_low_tile > 0) {
                    nk_dots_i8_load_a_sapphireamx_(&a_tile_bottom, a + (row_block_start + 16) * a_stride + depth_offset,
                                                   a_stride, rows_in_low_tile, valid_depth);
                }

                // Load B edge data (row-major: b_edge[row × depth + column]) and pack into B tile
                // Each "row" in edge data corresponds to one output column
                nk_dots_i8_load_a_sapphireamx_(&b_as_a, col_edge_ptr + depth_offset, depth, column_remainder_count,
                                               valid_depth);
                nk_dots_pack_i8_transposed_sapphireamx_(&b_as_a, &b_tile);

                _tile_loadd(0, a_tile_top.data, 64);
                _tile_loadd(1, a_tile_bottom.data, 64);
                _tile_loadd(2, b_tile.data, 64);

                _tile_dpbssd(4, 0, 2);
                _tile_dpbssd(6, 1, 2);
            }

            _tile_stored(4, c_high_state.data, 64);
            _tile_stored(6, c_low_state.data, 64);

            nk_dots_i8_store_sapphireamx_(&c_high_state, c + row_block_start * c_stride_elements + full_columns,
                                          c_stride_elements, rows_in_high_tile, column_remainder_count);
            if (rows_in_low_tile > 0) {
                nk_dots_i8_store_sapphireamx_(&c_low_state,
                                              c + (row_block_start + 16) * c_stride_elements + full_columns,
                                              c_stride_elements, rows_in_low_tile, column_remainder_count);
            }
        }
    }

    _tile_release();
    return nk_success_k;
}

/** I8 Gram upper triangle of @p vectors on AMX tiles, rows @p rows_begin to @p rows_end. */
NUMKONG_INLINE nk_status_t nk_gram_i8_sapphireamx_(                  //
    nk_i8_t const *vectors, nk_size_t vector_count, nk_size_t depth, //
    nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,     //
    nk_size_t rows_begin, nk_size_t rows_end) {
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));

    nk_size_t const result_stride_elements = result_stride / sizeof(nk_i32_t);

    // Handle row slicing: compute rows [rows_begin, rows_end)
    rows_end = nk_min_of_two(rows_end, vector_count);

    // Round depth up to multiple of 192 (3 tiles × 64 elements)
    nk_size_t const depth_tiles = nk_size_divide_round_up_(depth, 64);
    nk_size_t const depth_tile_groups = nk_size_divide_round_up_(depth_tiles, 3);

    nk_dots_i8_a16x64_sapphireamx_t a_tiles[3];
    nk_dots_i8_a16x64_sapphireamx_t b_src_tiles[3];
    nk_dots_i8_b64x16_sapphireamx_t b_tiles[3];
    nk_dots_i8_state_sapphireamx_t state;

    nk_amx_tile_configure_sapphireamx_();

    for (nk_size_t row_tile = rows_begin; row_tile < rows_end; row_tile += 16) {
        nk_size_t const valid_rows = (row_tile + 16 <= rows_end) ? 16 : (rows_end - row_tile);

        for (nk_size_t col_tile = row_tile; col_tile < vector_count; col_tile += 16) {
            nk_size_t const valid_columns = (col_tile + 16 <= vector_count) ? 16 : (vector_count - col_tile);

            nk_dots_i8_init_sapphireamx_(&state);

            for (nk_size_t depth_group_idx = 0; depth_group_idx < depth_tile_groups; depth_group_idx++) {
                nk_size_t const depth_base = depth_group_idx * 192;

                for (int tile_idx = 0; tile_idx < 3; tile_idx++) {
                    nk_size_t const depth_start = depth_base + tile_idx * 64;
                    nk_size_t const valid_depth = (depth_start + 64 <= depth)
                                                      ? 64
                                                      : (depth > depth_start ? depth - depth_start : 0);

                    nk_dots_i8_load_a_sapphireamx_(                //
                        &a_tiles[tile_idx],                        //
                        vectors + row_tile * stride + depth_start, //
                        stride, valid_rows, valid_depth);

                    if (row_tile == col_tile && valid_rows == valid_columns) {
                        nk_dots_pack_i8_transposed_sapphireamx_(&a_tiles[tile_idx], &b_tiles[tile_idx]);
                    }
                    else {
                        nk_dots_i8_load_a_sapphireamx_(                //
                            &b_src_tiles[tile_idx],                    //
                            vectors + col_tile * stride + depth_start, //
                            stride, valid_columns, valid_depth);
                        nk_dots_pack_i8_transposed_sapphireamx_(&b_src_tiles[tile_idx], &b_tiles[tile_idx]);
                    }
                }

                nk_dots_i8_update_sapphireamx_( //
                    &state, &a_tiles[0], &a_tiles[1], &a_tiles[2], &b_tiles[0], &b_tiles[1], &b_tiles[2]);
            }

            nk_dots_symmetric_store_sapphireamx_(                                  //
                state.data, result + row_tile * result_stride_elements + col_tile, //
                result_stride_elements, valid_rows, valid_columns, col_tile - row_tile);
        }
    }
    return nk_success_k;
}

#pragma endregion Signed Integers

#pragma region Unsigned Integers

/** U8 GEMM of @p a rows against pre-packed B columns into U32 @p c,on AMX tiles. */
NUMKONG_INLINE nk_status_t nk_gemm_packed_u8_sapphireamx_( //
    nk_u8_t const *a, void const *b_packed, nk_u32_t *c,   //
    nk_size_t rows, nk_size_t column_count, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride) {
    nk_unused_(column_count);

    // Parse packed B header
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    if (header->capability != nk_cap_sapphireamx_k) return nk_pack_mismatch_k;
    nk_size_t const column_tiles_count = header->full_column_tiles;
    nk_size_t const depth_tiles_count = header->full_depth_tiles;
    nk_size_t const column_remainder_count = header->column_remainder_count;

    // Packed B data regions
    nk_u8_t const *b_tiles_base = (nk_u8_t const *)((char const *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_u8_t const *col_edge_ptr = (nk_u8_t const *)((char const *)b_packed + header->column_edge_offset);

    // Stride conversions
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_u32_t);

    // Tile dimensions
    nk_size_t const tile_depth = 64;  // depth elements per U8 tile
    nk_size_t const tile_size = 1024; // bytes per packed tile
    nk_size_t const full_columns = column_tiles_count * 16;

    // Block counts (32 × 32 output blocks = 2 × 2 tiles)
    nk_size_t const row_blocks_count = nk_size_divide_round_up_(rows, 32);
    nk_size_t const col_blocks_count = column_tiles_count / 2;

    if (depth_tiles_count == 0) return nk_success_k;

    // Tile buffers for A (only used for edge tiles)
    nk_dots_u8_a16x64_sapphireamx_t a_tile_top, a_tile_bottom;
    nk_dots_u8_state2x2_sapphireamx_t c_accum_buffer;

    // Precompute: number of full depth-tiles
    nk_size_t const full_depth_tiles_count = depth / tile_depth;
    nk_size_t const depth_remainder = depth % tile_depth;

    nk_amx_tile_configure_sapphireamx_();

    // Process all 32 × 32 row × column blocks
    for (nk_size_t row_block_idx = 0; row_block_idx < row_blocks_count; row_block_idx++) {
        nk_size_t const row_block_start = row_block_idx * 32;
        nk_size_t const valid_rows_count = (row_block_start + 32 <= rows) ? 32 : (rows - row_block_start);
        nk_size_t const is_full_row_block = (valid_rows_count == 32);

        // Process full column-blocks (pairs of 16-column tiles = 32 columns)
        for (nk_size_t column_block_idx = 0; column_block_idx < col_blocks_count; column_block_idx++) {
            nk_size_t const col_block_start = column_block_idx * 32;

            // B tile base indices
            nk_size_t const b_column_left_base = (column_block_idx * 2) * depth_tiles_count;
            nk_size_t const b_column_right_base = (column_block_idx * 2 + 1) * depth_tiles_count;

            // Zero accumulators (TMM4-7 stay resident across entire depth loop)
            _tile_zero(4);
            _tile_zero(5);
            _tile_zero(6);
            _tile_zero(7);

            // Fast path: full row-block with full depth-tiles → direct A load with 2-deep pipelining
            if (is_full_row_block && full_depth_tiles_count > 0) {
                nk_u8_t const *a_top_base = a + row_block_start * a_stride;
                nk_u8_t const *a_bottom_base = a + (row_block_start + 16) * a_stride;

                nk_dots_u8_b64x16_sapphireamx_t const *b_tile_left =
                    (nk_dots_u8_b64x16_sapphireamx_t const *)(b_tiles_base + b_column_left_base * tile_size);
                nk_dots_u8_b64x16_sapphireamx_t const *b_tile_right =
                    (nk_dots_u8_b64x16_sapphireamx_t const *)(b_tiles_base + b_column_right_base * tile_size);

                // Prologue: load first depth tile into TMM0-3
                _tile_loadd(0, a_top_base, a_stride);
                _tile_loadd(1, a_bottom_base, a_stride);
                _tile_loadd(2, b_tile_left->data, 64);
                _tile_loadd(3, b_tile_right->data, 64);

                // Main loop: 2-deep software pipelining (compute current while loading next)
                for (nk_size_t depth_tile_idx = 0; depth_tile_idx < full_depth_tiles_count - 1; depth_tile_idx++) {
                    nk_size_t const next_depth_offset = (depth_tile_idx + 1) * tile_depth;

                    _tile_dpbuud(4, 0, 2);
                    _tile_dpbuud(5, 0, 3);
                    _tile_dpbuud(6, 1, 2);
                    _tile_dpbuud(7, 1, 3);

                    _tile_loadd(0, a_top_base + next_depth_offset, a_stride);
                    _tile_loadd(1, a_bottom_base + next_depth_offset, a_stride);
                    b_tile_left = (nk_dots_u8_b64x16_sapphireamx_t const *)(b_tiles_base +
                                                                            (b_column_left_base + depth_tile_idx + 1) *
                                                                                tile_size);
                    b_tile_right = (nk_dots_u8_b64x16_sapphireamx_t const *)(b_tiles_base + (b_column_right_base +
                                                                                             depth_tile_idx + 1) *
                                                                                                tile_size);
                    _tile_loadd(2, b_tile_left->data, 64);
                    _tile_loadd(3, b_tile_right->data, 64);
                }

                // Epilogue: final depth tile (no next to load)
                _tile_dpbuud(4, 0, 2);
                _tile_dpbuud(5, 0, 3);
                _tile_dpbuud(6, 1, 2);
                _tile_dpbuud(7, 1, 3);

                // Handle partial depth-tile (if any) with buffered load
                if (depth_remainder > 0) {
                    nk_size_t const depth_offset = full_depth_tiles_count * tile_depth;

                    nk_dots_u8_load_a_sapphireamx_(&a_tile_top, a_top_base + depth_offset, a_stride, 16,
                                                   depth_remainder);
                    nk_dots_u8_load_a_sapphireamx_(&a_tile_bottom, a_bottom_base + depth_offset, a_stride, 16,
                                                   depth_remainder);

                    b_tile_left = (nk_dots_u8_b64x16_sapphireamx_t const *)(b_tiles_base + (b_column_left_base +
                                                                                            full_depth_tiles_count) *
                                                                                               tile_size);
                    b_tile_right = (nk_dots_u8_b64x16_sapphireamx_t const *)(b_tiles_base + (b_column_right_base +
                                                                                             full_depth_tiles_count) *
                                                                                                tile_size);

                    _tile_loadd(0, a_tile_top.data, 64);
                    _tile_loadd(1, a_tile_bottom.data, 64);
                    _tile_loadd(2, b_tile_left->data, 64);
                    _tile_loadd(3, b_tile_right->data, 64);

                    _tile_dpbuud(4, 0, 2);
                    _tile_dpbuud(5, 0, 3);
                    _tile_dpbuud(6, 1, 2);
                    _tile_dpbuud(7, 1, 3);
                }
            }
            // Full row-block but only partial depth tile (depth < tile_depth)
            else if (is_full_row_block) {
                nk_u8_t const *a_top_base = a + row_block_start * a_stride;
                nk_u8_t const *a_bottom_base = a + (row_block_start + 16) * a_stride;

                nk_dots_u8_load_a_sapphireamx_(&a_tile_top, a_top_base, a_stride, 16, depth_remainder);
                nk_dots_u8_load_a_sapphireamx_(&a_tile_bottom, a_bottom_base, a_stride, 16, depth_remainder);

                nk_dots_u8_b64x16_sapphireamx_t const *b_tile_left =
                    (nk_dots_u8_b64x16_sapphireamx_t const *)(b_tiles_base + b_column_left_base * tile_size);
                nk_dots_u8_b64x16_sapphireamx_t const *b_tile_right =
                    (nk_dots_u8_b64x16_sapphireamx_t const *)(b_tiles_base + b_column_right_base * tile_size);

                _tile_loadd(0, a_tile_top.data, 64);
                _tile_loadd(1, a_tile_bottom.data, 64);
                _tile_loadd(2, b_tile_left->data, 64);
                _tile_loadd(3, b_tile_right->data, 64);

                _tile_dpbuud(4, 0, 2);
                _tile_dpbuud(5, 0, 3);
                _tile_dpbuud(6, 1, 2);
                _tile_dpbuud(7, 1, 3);
            }
            // Slow path: edge row-block → always use buffered load
            else {
                nk_size_t const rows_in_high_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
                nk_size_t const rows_in_low_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

                for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                    nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                    nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth
                                                                                            : depth_remainder;

                    nk_dots_u8_load_a_sapphireamx_(&a_tile_top, a + row_block_start * a_stride + depth_offset, a_stride,
                                                   rows_in_high_tile, valid_depth);
                    if (rows_in_low_tile > 0) {
                        nk_dots_u8_load_a_sapphireamx_(&a_tile_bottom,
                                                       a + (row_block_start + 16) * a_stride + depth_offset, a_stride,
                                                       rows_in_low_tile, valid_depth);
                    }

                    nk_dots_u8_b64x16_sapphireamx_t const *b_tile_left =
                        (nk_dots_u8_b64x16_sapphireamx_t const *)(b_tiles_base +
                                                                  (b_column_left_base + depth_tile_idx) * tile_size);
                    nk_dots_u8_b64x16_sapphireamx_t const *b_tile_right =
                        (nk_dots_u8_b64x16_sapphireamx_t const *)(b_tiles_base +
                                                                  (b_column_right_base + depth_tile_idx) * tile_size);

                    _tile_loadd(0, a_tile_top.data, 64);
                    _tile_loadd(1, a_tile_bottom.data, 64);
                    _tile_loadd(2, b_tile_left->data, 64);
                    _tile_loadd(3, b_tile_right->data, 64);

                    _tile_dpbuud(4, 0, 2);
                    _tile_dpbuud(5, 0, 3);
                    _tile_dpbuud(6, 1, 2);
                    _tile_dpbuud(7, 1, 3);
                }
            }

            // Store accumulators to output (once per output block, not per depth tile)
            if (is_full_row_block) {
                nk_u32_t *c_block = c + row_block_start * c_stride_elements + col_block_start;
                _tile_stored(4, c_block, c_stride);
                _tile_stored(5, c_block + 16, c_stride);
                _tile_stored(6, (nk_u32_t *)((char *)c_block + 16 * c_stride), c_stride);
                _tile_stored(7, (nk_u32_t *)((char *)c_block + 16 * c_stride) + 16, c_stride);
            }
            else {
                _tile_stored(4, c_accum_buffer.c[0][0].data, 64);
                _tile_stored(5, c_accum_buffer.c[0][1].data, 64);
                _tile_stored(6, c_accum_buffer.c[1][0].data, 64);
                _tile_stored(7, c_accum_buffer.c[1][1].data, 64);
                nk_dots_u8_output2x2_sapphireamx_(&c_accum_buffer,
                                                  c + row_block_start * c_stride_elements + col_block_start,
                                                  c_stride_elements, valid_rows_count, 32);
            }
        }

        // Handle odd column-tile (single 16-column tile if column_tiles_count is odd)
        if (column_tiles_count % 2 == 1) {
            nk_size_t const column_tile_idx = column_tiles_count - 1;
            nk_size_t const col_start = column_tile_idx * 16;
            nk_size_t const b_column_base = column_tile_idx * depth_tiles_count;
            nk_size_t const rows_in_high_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
            nk_size_t const rows_in_low_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

            nk_dots_u8_state_sapphireamx_t c_high_state, c_low_state;

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_u8_load_a_sapphireamx_(&a_tile_top, a + row_block_start * a_stride + depth_offset, a_stride,
                                               rows_in_high_tile, valid_depth);
                if (rows_in_low_tile > 0) {
                    nk_dots_u8_load_a_sapphireamx_(&a_tile_bottom, a + (row_block_start + 16) * a_stride + depth_offset,
                                                   a_stride, rows_in_low_tile, valid_depth);
                }

                nk_dots_u8_b64x16_sapphireamx_t const *b_tile =
                    (nk_dots_u8_b64x16_sapphireamx_t const *)(b_tiles_base +
                                                              (b_column_base + depth_tile_idx) * tile_size);

                _tile_loadd(0, a_tile_top.data, 64);
                _tile_loadd(1, a_tile_bottom.data, 64);
                _tile_loadd(2, b_tile->data, 64);

                _tile_dpbuud(4, 0, 2);
                _tile_dpbuud(6, 1, 2);
            }

            _tile_stored(4, c_high_state.data, 64);
            _tile_stored(6, c_low_state.data, 64);

            nk_dots_u8_store_sapphireamx_(&c_high_state, c + row_block_start * c_stride_elements + col_start,
                                          c_stride_elements, rows_in_high_tile, 16);
            if (rows_in_low_tile > 0) {
                nk_dots_u8_store_sapphireamx_(&c_low_state, c + (row_block_start + 16) * c_stride_elements + col_start,
                                              c_stride_elements, rows_in_low_tile, 16);
            }
        }

        // Handle column-edge (remaining columns < 16) using AMX with partial tiles
        if (column_remainder_count > 0) {
            nk_size_t const rows_in_high_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
            nk_size_t const rows_in_low_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

            nk_dots_u8_state_sapphireamx_t c_high_state, c_low_state;
            nk_dots_u8_a16x64_sapphireamx_t b_as_a;
            nk_dots_u8_b64x16_sapphireamx_t b_tile;

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_u8_load_a_sapphireamx_(&a_tile_top, a + row_block_start * a_stride + depth_offset, a_stride,
                                               rows_in_high_tile, valid_depth);
                if (rows_in_low_tile > 0) {
                    nk_dots_u8_load_a_sapphireamx_(&a_tile_bottom, a + (row_block_start + 16) * a_stride + depth_offset,
                                                   a_stride, rows_in_low_tile, valid_depth);
                }

                nk_dots_u8_load_a_sapphireamx_(&b_as_a, col_edge_ptr + depth_offset, depth, column_remainder_count,
                                               valid_depth);
                nk_dots_pack_u8_transposed_sapphireamx_(&b_as_a, &b_tile);

                _tile_loadd(0, a_tile_top.data, 64);
                _tile_loadd(1, a_tile_bottom.data, 64);
                _tile_loadd(2, b_tile.data, 64);

                _tile_dpbuud(4, 0, 2);
                _tile_dpbuud(6, 1, 2);
            }

            _tile_stored(4, c_high_state.data, 64);
            _tile_stored(6, c_low_state.data, 64);

            nk_dots_u8_store_sapphireamx_(&c_high_state, c + row_block_start * c_stride_elements + full_columns,
                                          c_stride_elements, rows_in_high_tile, column_remainder_count);
            if (rows_in_low_tile > 0) {
                nk_dots_u8_store_sapphireamx_(&c_low_state,
                                              c + (row_block_start + 16) * c_stride_elements + full_columns,
                                              c_stride_elements, rows_in_low_tile, column_remainder_count);
            }
        }
    }

    _tile_release();
    return nk_success_k;
}

/** U8 Gram upper triangle of @p vectors on AMX tiles, rows @p rows_begin to @p rows_end. */
NUMKONG_INLINE nk_status_t nk_gram_u8_sapphireamx_(                  //
    nk_u8_t const *vectors, nk_size_t vector_count, nk_size_t depth, //
    nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,     //
    nk_size_t rows_begin, nk_size_t rows_end) {
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));

    nk_size_t const result_stride_elements = result_stride / sizeof(nk_u32_t);

    // Handle row slicing: compute rows [rows_begin, rows_end)
    rows_end = nk_min_of_two(rows_end, vector_count);

    // Round depth up to multiple of 192 (3 tiles × 64 elements)
    nk_size_t const depth_tiles = nk_size_divide_round_up_(depth, 64);
    nk_size_t const depth_tile_groups = nk_size_divide_round_up_(depth_tiles, 3);

    nk_dots_u8_a16x64_sapphireamx_t a_tiles[3];
    nk_dots_u8_a16x64_sapphireamx_t b_src_tiles[3];
    nk_dots_u8_b64x16_sapphireamx_t b_tiles[3];
    nk_dots_u8_state_sapphireamx_t state;

    nk_amx_tile_configure_sapphireamx_();

    for (nk_size_t row_tile = rows_begin; row_tile < rows_end; row_tile += 16) {
        nk_size_t const valid_rows = (row_tile + 16 <= rows_end) ? 16 : (rows_end - row_tile);

        for (nk_size_t col_tile = row_tile; col_tile < vector_count; col_tile += 16) {
            nk_size_t const valid_columns = (col_tile + 16 <= vector_count) ? 16 : (vector_count - col_tile);

            nk_dots_u8_init_sapphireamx_(&state);

            for (nk_size_t depth_group_idx = 0; depth_group_idx < depth_tile_groups; depth_group_idx++) {
                nk_size_t const depth_base = depth_group_idx * 192;

                for (int tile_idx = 0; tile_idx < 3; tile_idx++) {
                    nk_size_t const depth_start = depth_base + tile_idx * 64;
                    nk_size_t const valid_depth = (depth_start + 64 <= depth)
                                                      ? 64
                                                      : (depth > depth_start ? depth - depth_start : 0);

                    nk_dots_u8_load_a_sapphireamx_(                //
                        &a_tiles[tile_idx],                        //
                        vectors + row_tile * stride + depth_start, //
                        stride, valid_rows, valid_depth);

                    if (row_tile == col_tile && valid_rows == valid_columns) {
                        nk_dots_pack_u8_transposed_sapphireamx_(&a_tiles[tile_idx], &b_tiles[tile_idx]);
                    }
                    else {
                        nk_dots_u8_load_a_sapphireamx_(                //
                            &b_src_tiles[tile_idx],                    //
                            vectors + col_tile * stride + depth_start, //
                            stride, valid_columns, valid_depth);
                        nk_dots_pack_u8_transposed_sapphireamx_(&b_src_tiles[tile_idx], &b_tiles[tile_idx]);
                    }
                }

                nk_dots_u8_update_sapphireamx_( //
                    &state, &a_tiles[0], &a_tiles[1], &a_tiles[2], &b_tiles[0], &b_tiles[1], &b_tiles[2]);
            }

            nk_dots_symmetric_store_sapphireamx_(                                  //
                state.data, result + row_tile * result_stride_elements + col_tile, //
                result_stride_elements, valid_rows, valid_columns, col_tile - row_tile);
        }
    }
    return nk_success_k;
}

#pragma endregion Unsigned Integers

/*  FP8 rows and block-scaled rows reach the AMX units as BF16, which holds every E4M3 and E5M2
 *  value and every block-scaled element times its scale exactly: an E2M1 value times a UE4M3
 *  scale needs 6 significant bits of BF16's 8, and the MX scales are powers of two, so only
 *  products leaving BF16's exponent range, beyond 2^±126, round. Each such dtype widens its own
 *  rows and runs its own GEMM, Gram and pack around the dtype-free tile helpers below. */
#pragma region Through BF16

/** Byte mask of the first @p valid of 32 elements. */
NUMKONG_INLINE __mmask32 nk_dots_bf16_valid_mask_sapphireamx_(nk_size_t valid) {
    return valid >= 32 ? 0xFFFFFFFFu : ((__mmask32)1 << valid) - 1;
}

/** 16 floats whose low halves are zero, as BF16. */
NUMKONG_INLINE __m256i nk_f32x16_to_exact_bf16x16_sapphireamx_(__m512 values_f32x16) {
    return _mm512_cvtepi32_epi16(_mm512_srli_epi32(_mm512_castps_si512(values_f32x16), 16));
}

/** Two halves of 16 values, each times its own scale, as 32 BF16. */
NUMKONG_INLINE __m512i nk_scaled_halves_to_bf16x32_sapphireamx_(__m512 low_f32x16, __m512 high_f32x16,
                                                                nk_f32_t low_scale, nk_f32_t high_scale) {
    __m256i const low_bf16x16 = nk_f32x16_to_exact_bf16x16_sapphireamx_(
        _mm512_mul_ps(low_f32x16, _mm512_set1_ps(low_scale)));
    __m256i const high_bf16x16 = nk_f32x16_to_exact_bf16x16_sapphireamx_(
        _mm512_mul_ps(high_f32x16, _mm512_set1_ps(high_scale)));
    return _mm512_inserti64x4(_mm512_castsi256_si512(low_bf16x16), high_bf16x16, 1);
}

NUMKONG_INLINE nk_f32_t nk_ue8m0_scale_sapphireamx_(nk_ue8m0_t const *scales, nk_size_t block) {
    nk_f32_t scale;
    nk_ue8m0_to_f32_serial_(scales + block, &scale);
    return scale;
}

NUMKONG_INLINE nk_f32_t nk_ue4m3_scale_sapphireamx_(nk_ue4m3_t const *scales, nk_size_t block) {
    nk_f32_t scale;
    nk_ue4m3_to_f32_serial_(scales + block, &scale);
    return scale;
}

/** The 16 values per half of 32 E2M1 elements from @p first of row @p row, zero past @p valid. */
NUMKONG_INLINE void nk_e2m1x32_to_f32x16x2_sapphireamx_(nk_e2m1x2_t const *values, nk_size_t stride, nk_size_t row,
                                                        nk_size_t first, nk_size_t valid, __m512 *low_f32x16,
                                                        __m512 *high_f32x16) {
    __m128i const bytes_u8x16 = _mm_maskz_loadu_epi8((__mmask16)((1u << ((valid + 1) / 2)) - 1),
                                                     values + row * stride + first / 2);
    *low_f32x16 = nk_e2m1x16_to_f32x16_skylake_(bytes_u8x16);
    *high_f32x16 = nk_e2m1x16_to_f32x16_skylake_(_mm_srli_si128(bytes_u8x16, 8));
}

/** The squares of 32 BF16, added to @p sum_f32x16 in F32. */
NUMKONG_INLINE __m512 nk_bf16x32_sumsq_step_sapphireamx_(__m512 sum_f32x16, __m512i widened_bf16x32) {
    __m512 const low_f32x16 = _mm512_castsi512_ps(
        _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_castsi512_si256(widened_bf16x32)), 16));
    __m512 const high_f32x16 = _mm512_castsi512_ps(
        _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(widened_bf16x32, 1)), 16));
    return _mm512_fmadd_ps(low_f32x16, low_f32x16, _mm512_fmadd_ps(high_f32x16, high_f32x16, sum_f32x16));
}

NUMKONG_INLINE nk_dots_bf16_b32x16_sapphireamx_t const *nk_dots_bf16_b_tile_sapphireamx_(nk_bf16_t const *b_tiles,
                                                                                         nk_size_t tile_index) {
    return (nk_dots_bf16_b32x16_sapphireamx_t const *)(b_tiles + tile_index * 512);
}

NUMKONG_INLINE nk_f32_t const *nk_dots_bf16_packed_norms_sapphireamx_(void const *b_packed) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    return (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
}

NUMKONG_INLINE void nk_dots_bf16_zero_2x2_sapphireamx_(void) {
    _tile_zero(4);
    _tile_zero(5);
    _tile_zero(6);
    _tile_zero(7);
}

NUMKONG_INLINE void nk_dots_bf16_zero_2x1_sapphireamx_(void) {
    _tile_zero(4);
    _tile_zero(6);
}

/** One depth step of two A tiles against two B tiles into TMM4-7. */
NUMKONG_INLINE void nk_dots_bf16_accumulate_2x2_sapphireamx_(nk_dots_bf16_a16x32_sapphireamx_t const *a_tile_top,
                                                             nk_dots_bf16_a16x32_sapphireamx_t const *a_tile_bottom,
                                                             nk_dots_bf16_b32x16_sapphireamx_t const *b_tile_left,
                                                             nk_dots_bf16_b32x16_sapphireamx_t const *b_tile_right) {
    _tile_loadd(0, a_tile_top->data, 64);
    _tile_loadd(1, a_tile_bottom->data, 64);
    _tile_loadd(2, b_tile_left->data, 64);
    _tile_loadd(3, b_tile_right->data, 64);

    _tile_dpbf16ps(4, 0, 2);
    _tile_dpbf16ps(5, 0, 3);
    _tile_dpbf16ps(6, 1, 2);
    _tile_dpbf16ps(7, 1, 3);
}

/** One depth step of two A tiles against one B tile into TMM4 and TMM6. */
NUMKONG_INLINE void nk_dots_bf16_accumulate_2x1_sapphireamx_(nk_dots_bf16_a16x32_sapphireamx_t const *a_tile_top,
                                                             nk_dots_bf16_a16x32_sapphireamx_t const *a_tile_bottom,
                                                             nk_dots_bf16_b32x16_sapphireamx_t const *b_tile) {
    _tile_loadd(0, a_tile_top->data, 64);
    _tile_loadd(1, a_tile_bottom->data, 64);
    _tile_loadd(2, b_tile->data, 64);

    _tile_dpbf16ps(4, 0, 2);
    _tile_dpbf16ps(6, 1, 2);
}

/** One depth step of two A tiles against the BF16 column-edge rows at @p depth_offset, which are
 *  transposed into a B tile first. */
NUMKONG_INLINE void nk_dots_bf16_accumulate_edge_sapphireamx_(nk_dots_bf16_a16x32_sapphireamx_t const *a_tile_top,
                                                              nk_dots_bf16_a16x32_sapphireamx_t const *a_tile_bottom,
                                                              nk_bf16_t const *column_edge, nk_size_t depth_offset,
                                                              nk_size_t depth, nk_size_t column_remainder_count,
                                                              nk_size_t valid_depth) {
    nk_dots_bf16_a16x32_sapphireamx_t b_as_a;
    nk_dots_bf16_b32x16_sapphireamx_t b_tile;
    nk_dots_bf16_load_a_sapphireamx_(&b_as_a, column_edge + depth_offset, depth, column_remainder_count, valid_depth);
    nk_dots_pack_bf16_transposed_sapphireamx_(&b_as_a, &b_tile);
    nk_dots_bf16_accumulate_2x1_sapphireamx_(a_tile_top, a_tile_bottom, &b_tile);
}

/** Stores TMM4-7 as a 32 × 32 block of F32 at @p c_block, clipped to @p valid_rows_count rows. */
NUMKONG_INLINE void nk_dots_bf16_store_block_sapphireamx_(nk_f32_t *c_block, nk_size_t c_stride,
                                                          nk_size_t valid_rows_count) {
    if (valid_rows_count == 32) {
        _tile_stored(4, c_block, c_stride);
        _tile_stored(5, c_block + 16, c_stride);
        _tile_stored(6, (nk_f32_t *)((char *)c_block + 16 * c_stride), c_stride);
        _tile_stored(7, (nk_f32_t *)((char *)c_block + 16 * c_stride) + 16, c_stride);
    }
    else {
        nk_dots_bf16_state2x2_sapphireamx_t c_accum_buffer;
        _tile_stored(4, c_accum_buffer.c[0][0].data, 64);
        _tile_stored(5, c_accum_buffer.c[0][1].data, 64);
        _tile_stored(6, c_accum_buffer.c[1][0].data, 64);
        _tile_stored(7, c_accum_buffer.c[1][1].data, 64);
        nk_dots_bf16_output2x2_sapphireamx_(&c_accum_buffer, c_block, c_stride / sizeof(nk_f32_t), valid_rows_count,
                                            32);
    }
}

/** Stores TMM4 and TMM6 as up to 16 columns of F32 at @p c_column, the high and low tiles clipped
 *  to their row counts. */
NUMKONG_INLINE void nk_dots_bf16_store_column_sapphireamx_(nk_f32_t *c_column, nk_size_t c_stride_elements,
                                                           nk_size_t rows_in_high_tile, nk_size_t rows_in_low_tile,
                                                           nk_size_t columns) {
    nk_dots_bf16_state_sapphireamx_t c_high_state, c_low_state;
    _tile_stored(4, c_high_state.data, 64);
    _tile_stored(6, c_low_state.data, 64);

    nk_dots_bf16_store_sapphireamx_(&c_high_state, c_column, c_stride_elements, rows_in_high_tile, columns);
    if (rows_in_low_tile > 0)
        nk_dots_bf16_store_sapphireamx_(&c_low_state, c_column + 16 * c_stride_elements, c_stride_elements,
                                        rows_in_low_tile, columns);
}

/** Zeros the @p columns_count results of each of @p rows rows, as an empty depth sums to. */
NUMKONG_INLINE void nk_dots_bf16_zero_results_sapphireamx_(nk_f32_t *c, nk_size_t rows, nk_size_t columns_count,
                                                           nk_size_t c_stride_elements) {
    for (nk_size_t row = 0; row < rows; row++)
        for (nk_size_t column = 0; column < columns_count; column += 16)
            _mm512_mask_storeu_ps(
                c + row * c_stride_elements + column,
                columns_count - column < 16 ? (__mmask16)((1u << (columns_count - column)) - 1) : (__mmask16)0xFFFF,
                _mm512_setzero_ps());
}

/** Bytes from the packed buffer's start to its column-edge rows, past the full BF16 tiles. */
NUMKONG_INLINE nk_size_t nk_dots_bf16_column_edge_offset_sapphireamx_(nk_size_t column_count, nk_size_t depth) {
    return sizeof(nk_dots_amx_packed_header_t) +
           (column_count / 16) * nk_size_divide_round_up_(depth, 32) * 512 * sizeof(nk_bf16_t);
}

/** Bytes from the packed buffer's start to its per-column norms, past the column-edge rows. */
NUMKONG_INLINE nk_size_t nk_dots_bf16_norms_offset_sapphireamx_(nk_size_t column_count, nk_size_t depth) {
    return nk_dots_bf16_column_edge_offset_sapphireamx_(column_count, depth) +
           (column_count % 16) * depth * sizeof(nk_bf16_t);
}

/** Writes the header of a BF16 pack of @p column_count columns, from the call packing column 0. */
NUMKONG_INLINE void nk_dots_bf16_pack_header_sapphireamx_(void *b_packed, nk_f32_t tensor_scale, nk_size_t column_count,
                                                          nk_size_t depth, nk_size_t columns_begin) {
    if (columns_begin != 0) return;
    nk_dots_amx_packed_header_t *header = (nk_dots_amx_packed_header_t *)b_packed;
    nk_u32_t *header_words = (nk_u32_t *)header;
    for (nk_size_t word_index = 0; word_index < sizeof(*header) / sizeof(nk_u32_t); word_index++)
        header_words[word_index] = 0;
    header->tensor_scale = tensor_scale;
    header->columns = (nk_u32_t)column_count;
    header->depth = (nk_u32_t)depth;
    header->full_column_tiles = (nk_u32_t)(column_count / 16);
    header->full_depth_tiles = (nk_u32_t)nk_size_divide_round_up_(depth, 32);
    header->column_remainder_count = (nk_u32_t)(column_count % 16);
    header->capability = nk_cap_sapphireamx_k;
    header->column_edge_offset = (nk_u32_t)nk_dots_bf16_column_edge_offset_sapphireamx_(column_count, depth);
    header->norms_byte_offset = (nk_u32_t)nk_dots_bf16_norms_offset_sapphireamx_(column_count, depth);
}

/** Transposes a widened 16 × 32 BF16 tile into the B tile layout at @p tile_output. */
NUMKONG_INLINE void nk_dots_bf16_pack_tile_sapphireamx_(nk_dots_bf16_a16x32_sapphireamx_t const *source_tile,
                                                        nk_bf16_t *tile_output) {
    nk_dots_bf16_b32x16_sapphireamx_t transposed_tile;
    nk_dots_pack_bf16_transposed_sapphireamx_(source_tile, &transposed_tile);
    for (nk_size_t i = 0; i < 512 * sizeof(nk_bf16_t); i += 64)
        _mm512_storeu_si512((char *)tile_output + i, _mm512_load_si512((char const *)&transposed_tile + i));
}

/** Bytes a block-scaled pack takes: the BF16 pack with norms, exponents and spreads padded to whole
 *  tiles, then the raw codes and scales of every column for the exact path. */
NUMKONG_INLINE nk_size_t nk_dots_packed_bytes_scaled_sapphireamx_(nk_size_t block_size, nk_size_t block_bytes,
                                                                  nk_size_t column_count, nk_size_t depth) {
    nk_size_t const padded = nk_size_round_up_to_multiple_(column_count, 16);
    return nk_dots_packed_bytes_bf16_sapphireamx_(column_count, depth) + (padded - column_count) * sizeof(nk_f32_t) +
           padded * 2 * sizeof(nk_i32_t) + column_count * depth / block_size * (block_bytes + 1);
}

/** The arrays a block-scaled pack keeps past its tiles: norms, exponents and spreads padded to
 *  whole tiles, then raw codes and scales. */
typedef struct nk_dots_scaled_view_sapphireamx_t {
    nk_f32_t *norms;
    nk_i32_t *exponents;
    nk_i32_t *spreads;
    nk_u8_t *codes;
    nk_u8_t *scales;
    nk_size_t padded;
    nk_size_t blocks;
    nk_size_t row_bytes;
} nk_dots_scaled_view_sapphireamx_t;

NUMKONG_INLINE nk_dots_scaled_view_sapphireamx_t nk_dots_scaled_view_sapphireamx_(void const *b_packed,
                                                                                  nk_size_t column_count,
                                                                                  nk_size_t depth, nk_size_t block_size,
                                                                                  nk_size_t block_bytes) {
    nk_dots_scaled_view_sapphireamx_t view;
    view.padded = nk_size_round_up_to_multiple_(column_count, 16);
    view.blocks = depth / block_size;
    view.row_bytes = view.blocks * block_bytes;
    view.norms = (nk_f32_t *)((char *)b_packed + nk_dots_bf16_norms_offset_sapphireamx_(column_count, depth));
    view.exponents = (nk_i32_t *)(view.norms + view.padded);
    view.spreads = view.exponents + view.padded;
    view.codes = (nk_u8_t *)(view.spreads + view.padded);
    view.scales = view.codes + column_count * view.row_bytes;
    return view;
}

/** The widest spread in the pack's @p padded columns. */
NUMKONG_INLINE nk_i32_t nk_dots_scaled_spread_max_sapphireamx_(nk_i32_t const *spreads, nk_size_t padded) {
    __m512i spread_max_i32x16 = _mm512_setzero_si512();
    for (nk_size_t column = 0; column < padded; column += 16)
        spread_max_i32x16 = _mm512_max_epi32(spread_max_i32x16, _mm512_loadu_si512(spreads + column));
    return _mm512_reduce_max_epi32(spread_max_i32x16);
}

/** Zeros the padding of the arrays of @p view past @p column_count, which the epilogues read four
 *  columns at a time. */
NUMKONG_INLINE void nk_dots_scaled_pad_sapphireamx_(nk_dots_scaled_view_sapphireamx_t const *view,
                                                    nk_size_t column_count) {
    for (nk_size_t column = column_count; column < view->padded; column += 16) {
        __mmask16 const padding_m16 = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)(view->padded - column));
        _mm512_mask_storeu_ps(view->norms + column, padding_m16, _mm512_setzero_ps());
        _mm512_mask_storeu_epi32(view->exponents + column, padding_m16, _mm512_setzero_si512());
        _mm512_mask_storeu_epi32(view->spreads + column, padding_m16, _mm512_setzero_si512());
    }
}

/*  AMX reads F32 subnormals as zero, so an MX row or column spreading past a dtype's limit, where
 *  its smallest element times its smallest rebased scale would be one, takes the exact path. A
 *  limit is 157 plus the binary exponent of the smallest element: 2⁻¹ for E2M1, 2⁻⁹ for E4M3,
 *  2⁻¹⁶ for E5M2. */
enum {
    nk_dots_spread_limit_mxfp4_sapphireamx_k = 156,
    nk_dots_spread_limit_mxfp8e4m3_sapphireamx_k = 148,
    nk_dots_spread_limit_mxfp8e5m2_sapphireamx_k = 141,
};

/** Whether a row and a column spreading @p row_spread and @p column_spread binades leave the
 *  rebased window, or either passes @p spread_limit. */
NUMKONG_INLINE int nk_dots_scaled_exceeds_sapphireamx_(nk_i32_t row_spread, nk_i32_t column_spread, int normalized,
                                                       nk_i32_t spread_limit) {
    return nk_cross_scaled_exceeds_serial_(row_spread, column_spread, normalized) || row_spread > spread_limit ||
           column_spread > spread_limit;
}

/** Finishes @p count relative sums at @p values into dots, four at a time with masked tails;
 *  @p column_exponents and @p column_norms are readable four lanes past @p count. */
NUMKONG_INLINE void nk_dots_scaled_finish_row_dot_sapphireamx_(nk_f32_t *values, nk_size_t count, nk_f32_t mantissa,
                                                               nk_i32_t row_exponent, nk_i32_t const *column_exponents,
                                                               nk_f32_t row_norm, nk_f32_t const *column_norms) {
    for (nk_size_t column = 0; column < count; column += 4) {
        __mmask8 const column_m8 = count - column < 4 ? (__mmask8)((1u << (count - column)) - 1) : (__mmask8)0xF;
        nk_b128_vec_t values_vec;
        values_vec.xmm_ps = _mm_maskz_loadu_ps(column_m8, values + column);
        nk_dot_f32x4_from_relative_skylake_(&values_vec, mantissa, row_exponent, column_exponents + column, row_norm,
                                            column_norms + column);
        _mm_mask_storeu_ps(values + column, column_m8, values_vec.xmm_ps);
    }
}

#pragma endregion Through BF16

#pragma region E4M3 Floats

NUMKONG_INLINE __m512i nk_e4m3_widen_bf16_sapphireamx_(nk_e4m3_t const *values, nk_size_t stride, nk_size_t row,
                                                       nk_size_t first, nk_size_t valid) {
    return nk_e4m3x32_to_bf16x32_icelake_(
        _mm256_maskz_loadu_epi8(nk_dots_bf16_valid_mask_sapphireamx_(valid), values + row * stride + first));
}

NUMKONG_INLINE nk_f32_t nk_e4m3_sumsq_sapphireamx_(nk_e4m3_t const *values, nk_size_t stride, nk_size_t row,
                                                   nk_size_t depth) {
    return nk_dots_reduce_sumsq_e4m3_skylake_(values + row * stride, depth, sizeof(nk_e4m3_t));
}

/** Loads rows @p first_row to `first_row + 16` of @p values into a BF16 A tile, zeros past
 *  @p valid_rows rows and @p valid_columns elements from @p depth_offset. */
NUMKONG_INLINE void nk_dots_load_a_e4m3_sapphireamx_(nk_dots_bf16_a16x32_sapphireamx_t *a_tile, nk_e4m3_t const *values,
                                                     nk_size_t stride, nk_size_t first_row, nk_size_t depth_offset,
                                                     nk_size_t valid_rows, nk_size_t valid_columns) {
    for (nk_size_t row = 0; row < 16; row++)
        _mm512_store_si512((__m512i *)a_tile->data[row],
                           row < valid_rows ? nk_e4m3_widen_bf16_sapphireamx_(values, stride, first_row + row,
                                                                              depth_offset, valid_columns)
                                            : _mm512_setzero_si512());
    nk_compiler_barrier_sapphireamx_();
}

/** GEMM of @p a rows widened to BF16 against B columns from @c nk_dots_pack_e4m3_sapphireamx, into
 *  F32 @p c, on BF16 AMX tiles. */
NUMKONG_INLINE nk_status_t nk_gemm_packed_e4m3_sapphireamx_( //
    nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,   //
    nk_size_t rows, nk_size_t columns_count, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    if (header->capability != nk_cap_sapphireamx_k) return nk_pack_mismatch_k;
    nk_size_t const column_tiles_count = header->full_column_tiles;
    nk_size_t const depth_tiles_count = header->full_depth_tiles;
    nk_size_t const column_remainder_count = header->column_remainder_count;
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    if (depth_tiles_count == 0) {
        nk_dots_bf16_zero_results_sapphireamx_(c, rows, columns_count, c_stride_elements);
        return nk_success_k;
    }

    nk_bf16_t const *b_tiles = (nk_bf16_t const *)((char const *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_bf16_t const *column_edge = (nk_bf16_t const *)((char const *)b_packed + header->column_edge_offset);
    nk_size_t const full_columns = column_tiles_count * 16;
    nk_size_t const row_blocks_count = nk_size_divide_round_up_(rows, 32);
    nk_size_t const column_blocks_count = column_tiles_count / 2;
    nk_size_t const full_depth_tiles_count = depth / 32;
    nk_size_t const depth_remainder = depth % 32;
    nk_dots_bf16_a16x32_sapphireamx_t a_tile_top, a_tile_bottom;

    nk_amx_tile_configure_sapphireamx_();

    for (nk_size_t row_block = 0; row_block < row_blocks_count; row_block++) {
        nk_size_t const row_block_start = row_block * 32;
        nk_size_t const valid_rows_count = (row_block_start + 32 <= rows) ? 32 : (rows - row_block_start);
        nk_size_t const rows_in_high_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
        nk_size_t const rows_in_low_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

        for (nk_size_t column_block = 0; column_block < column_blocks_count; column_block++) {
            nk_dots_bf16_zero_2x2_sapphireamx_();
            for (nk_size_t depth_tile = 0; depth_tile < depth_tiles_count; depth_tile++) {
                nk_size_t const depth_offset = depth_tile * 32;
                nk_size_t const valid_depth = (depth_tile < full_depth_tiles_count) ? 32 : depth_remainder;
                nk_dots_load_a_e4m3_sapphireamx_(&a_tile_top, a, a_stride, row_block_start, depth_offset,
                                                 rows_in_high_tile, valid_depth);
                if (rows_in_low_tile > 0)
                    nk_dots_load_a_e4m3_sapphireamx_(&a_tile_bottom, a, a_stride, row_block_start + 16, depth_offset,
                                                     rows_in_low_tile, valid_depth);
                nk_dots_bf16_accumulate_2x2_sapphireamx_(
                    &a_tile_top, &a_tile_bottom,
                    nk_dots_bf16_b_tile_sapphireamx_(b_tiles, (column_block * 2) * depth_tiles_count + depth_tile),
                    nk_dots_bf16_b_tile_sapphireamx_(b_tiles, (column_block * 2 + 1) * depth_tiles_count + depth_tile));
            }
            nk_dots_bf16_store_block_sapphireamx_(c + row_block_start * c_stride_elements + column_block * 32, c_stride,
                                                  valid_rows_count);
        }

        if (column_tiles_count % 2 == 1) {
            nk_size_t const column_tile = column_tiles_count - 1;
            nk_dots_bf16_zero_2x1_sapphireamx_();
            for (nk_size_t depth_tile = 0; depth_tile < depth_tiles_count; depth_tile++) {
                nk_size_t const depth_offset = depth_tile * 32;
                nk_size_t const valid_depth = (depth_tile < full_depth_tiles_count) ? 32 : depth_remainder;
                nk_dots_load_a_e4m3_sapphireamx_(&a_tile_top, a, a_stride, row_block_start, depth_offset,
                                                 rows_in_high_tile, valid_depth);
                if (rows_in_low_tile > 0)
                    nk_dots_load_a_e4m3_sapphireamx_(&a_tile_bottom, a, a_stride, row_block_start + 16, depth_offset,
                                                     rows_in_low_tile, valid_depth);
                nk_dots_bf16_accumulate_2x1_sapphireamx_(
                    &a_tile_top, &a_tile_bottom,
                    nk_dots_bf16_b_tile_sapphireamx_(b_tiles, column_tile * depth_tiles_count + depth_tile));
            }
            nk_dots_bf16_store_column_sapphireamx_(c + row_block_start * c_stride_elements + column_tile * 16,
                                                   c_stride_elements, rows_in_high_tile, rows_in_low_tile, 16);
        }

        if (column_remainder_count > 0) {
            nk_dots_bf16_zero_2x1_sapphireamx_();
            for (nk_size_t depth_tile = 0; depth_tile < depth_tiles_count; depth_tile++) {
                nk_size_t const depth_offset = depth_tile * 32;
                nk_size_t const valid_depth = (depth_tile < full_depth_tiles_count) ? 32 : depth_remainder;
                nk_dots_load_a_e4m3_sapphireamx_(&a_tile_top, a, a_stride, row_block_start, depth_offset,
                                                 rows_in_high_tile, valid_depth);
                if (rows_in_low_tile > 0)
                    nk_dots_load_a_e4m3_sapphireamx_(&a_tile_bottom, a, a_stride, row_block_start + 16, depth_offset,
                                                     rows_in_low_tile, valid_depth);
                nk_dots_bf16_accumulate_edge_sapphireamx_(&a_tile_top, &a_tile_bottom, column_edge, depth_offset, depth,
                                                          column_remainder_count, valid_depth);
            }
            nk_dots_bf16_store_column_sapphireamx_(c + row_block_start * c_stride_elements + full_columns,
                                                   c_stride_elements, rows_in_high_tile, rows_in_low_tile,
                                                   column_remainder_count);
        }
    }

    _tile_release();
    return nk_success_k;
}

/** Gram upper triangle of @p vectors widened to BF16, on AMX tiles, covering rows from
 *  @p rows_begin up to @p rows_end. */
NUMKONG_INLINE nk_status_t nk_gram_e4m3_sapphireamx_(                                    //
    nk_e4m3_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end) {
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    rows_end = nk_min_of_two(rows_end, vector_count);

    // Depth is walked in groups of 3 tiles × 32 elements
    nk_size_t const depth_tile_groups = nk_size_divide_round_up_(nk_size_divide_round_up_(depth, 32), 3);

    nk_dots_bf16_a16x32_sapphireamx_t a_tiles[3];
    nk_dots_bf16_a16x32_sapphireamx_t b_src_tiles[3];
    nk_dots_bf16_b32x16_sapphireamx_t b_tiles[3];
    nk_dots_bf16_state_sapphireamx_t state;

    nk_amx_tile_configure_sapphireamx_();

    for (nk_size_t row_tile = rows_begin; row_tile < rows_end; row_tile += 16) {
        nk_size_t const valid_rows = (row_tile + 16 <= rows_end) ? 16 : (rows_end - row_tile);

        for (nk_size_t column_tile = row_tile; column_tile < vector_count; column_tile += 16) {
            nk_size_t const valid_columns = (column_tile + 16 <= vector_count) ? 16 : (vector_count - column_tile);

            nk_dots_bf16_init_sapphireamx_(&state);

            for (nk_size_t depth_group = 0; depth_group < depth_tile_groups; depth_group++) {
                for (int tile_index = 0; tile_index < 3; tile_index++) {
                    nk_size_t const depth_start = depth_group * 96 + tile_index * 32;
                    nk_size_t const valid_depth = (depth_start + 32 <= depth)
                                                      ? 32
                                                      : (depth > depth_start ? depth - depth_start : 0);

                    nk_dots_load_a_e4m3_sapphireamx_(&a_tiles[tile_index], vectors, stride, row_tile, depth_start,
                                                     valid_rows, valid_depth);
                    if (row_tile == column_tile && valid_rows == valid_columns) {
                        nk_dots_pack_bf16_transposed_sapphireamx_(&a_tiles[tile_index], &b_tiles[tile_index]);
                    }
                    else {
                        nk_dots_load_a_e4m3_sapphireamx_(&b_src_tiles[tile_index], vectors, stride, column_tile,
                                                         depth_start, valid_columns, valid_depth);
                        nk_dots_pack_bf16_transposed_sapphireamx_(&b_src_tiles[tile_index], &b_tiles[tile_index]);
                    }
                }

                nk_dots_bf16_update_sapphireamx_( //
                    &state, &a_tiles[0], &a_tiles[1], &a_tiles[2], &b_tiles[0], &b_tiles[1], &b_tiles[2]);
            }

            nk_dots_symmetric_store_sapphireamx_(                                     //
                state.data, result + row_tile * result_stride_elements + column_tile, //
                result_stride_elements, valid_rows, valid_columns, column_tile - row_tile);
        }
    }
    return nk_success_k;
}

#pragma endregion E4M3 Floats

#pragma region E5M2 Floats

NUMKONG_INLINE __m512i nk_e5m2_widen_bf16_sapphireamx_(nk_e5m2_t const *values, nk_size_t stride, nk_size_t row,
                                                       nk_size_t first, nk_size_t valid) {
    return nk_e5m2x32_to_bf16x32_icelake_(
        _mm256_maskz_loadu_epi8(nk_dots_bf16_valid_mask_sapphireamx_(valid), values + row * stride + first));
}

NUMKONG_INLINE nk_f32_t nk_e5m2_sumsq_sapphireamx_(nk_e5m2_t const *values, nk_size_t stride, nk_size_t row,
                                                   nk_size_t depth) {
    return nk_dots_reduce_sumsq_e5m2_skylake_(values + row * stride, depth, sizeof(nk_e5m2_t));
}

/** Loads rows @p first_row to `first_row + 16` of @p values into a BF16 A tile, zeros past
 *  @p valid_rows rows and @p valid_columns elements from @p depth_offset. */
NUMKONG_INLINE void nk_dots_load_a_e5m2_sapphireamx_(nk_dots_bf16_a16x32_sapphireamx_t *a_tile, nk_e5m2_t const *values,
                                                     nk_size_t stride, nk_size_t first_row, nk_size_t depth_offset,
                                                     nk_size_t valid_rows, nk_size_t valid_columns) {
    for (nk_size_t row = 0; row < 16; row++)
        _mm512_store_si512((__m512i *)a_tile->data[row],
                           row < valid_rows ? nk_e5m2_widen_bf16_sapphireamx_(values, stride, first_row + row,
                                                                              depth_offset, valid_columns)
                                            : _mm512_setzero_si512());
    nk_compiler_barrier_sapphireamx_();
}

/** GEMM of @p a rows widened to BF16 against B columns from @c nk_dots_pack_e5m2_sapphireamx, into
 *  F32 @p c, on BF16 AMX tiles. */
NUMKONG_INLINE nk_status_t nk_gemm_packed_e5m2_sapphireamx_( //
    nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,   //
    nk_size_t rows, nk_size_t columns_count, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    if (header->capability != nk_cap_sapphireamx_k) return nk_pack_mismatch_k;
    nk_size_t const column_tiles_count = header->full_column_tiles;
    nk_size_t const depth_tiles_count = header->full_depth_tiles;
    nk_size_t const column_remainder_count = header->column_remainder_count;
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    if (depth_tiles_count == 0) {
        nk_dots_bf16_zero_results_sapphireamx_(c, rows, columns_count, c_stride_elements);
        return nk_success_k;
    }

    nk_bf16_t const *b_tiles = (nk_bf16_t const *)((char const *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_bf16_t const *column_edge = (nk_bf16_t const *)((char const *)b_packed + header->column_edge_offset);
    nk_size_t const full_columns = column_tiles_count * 16;
    nk_size_t const row_blocks_count = nk_size_divide_round_up_(rows, 32);
    nk_size_t const column_blocks_count = column_tiles_count / 2;
    nk_size_t const full_depth_tiles_count = depth / 32;
    nk_size_t const depth_remainder = depth % 32;
    nk_dots_bf16_a16x32_sapphireamx_t a_tile_top, a_tile_bottom;

    nk_amx_tile_configure_sapphireamx_();

    for (nk_size_t row_block = 0; row_block < row_blocks_count; row_block++) {
        nk_size_t const row_block_start = row_block * 32;
        nk_size_t const valid_rows_count = (row_block_start + 32 <= rows) ? 32 : (rows - row_block_start);
        nk_size_t const rows_in_high_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
        nk_size_t const rows_in_low_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

        for (nk_size_t column_block = 0; column_block < column_blocks_count; column_block++) {
            nk_dots_bf16_zero_2x2_sapphireamx_();
            for (nk_size_t depth_tile = 0; depth_tile < depth_tiles_count; depth_tile++) {
                nk_size_t const depth_offset = depth_tile * 32;
                nk_size_t const valid_depth = (depth_tile < full_depth_tiles_count) ? 32 : depth_remainder;
                nk_dots_load_a_e5m2_sapphireamx_(&a_tile_top, a, a_stride, row_block_start, depth_offset,
                                                 rows_in_high_tile, valid_depth);
                if (rows_in_low_tile > 0)
                    nk_dots_load_a_e5m2_sapphireamx_(&a_tile_bottom, a, a_stride, row_block_start + 16, depth_offset,
                                                     rows_in_low_tile, valid_depth);
                nk_dots_bf16_accumulate_2x2_sapphireamx_(
                    &a_tile_top, &a_tile_bottom,
                    nk_dots_bf16_b_tile_sapphireamx_(b_tiles, (column_block * 2) * depth_tiles_count + depth_tile),
                    nk_dots_bf16_b_tile_sapphireamx_(b_tiles, (column_block * 2 + 1) * depth_tiles_count + depth_tile));
            }
            nk_dots_bf16_store_block_sapphireamx_(c + row_block_start * c_stride_elements + column_block * 32, c_stride,
                                                  valid_rows_count);
        }

        if (column_tiles_count % 2 == 1) {
            nk_size_t const column_tile = column_tiles_count - 1;
            nk_dots_bf16_zero_2x1_sapphireamx_();
            for (nk_size_t depth_tile = 0; depth_tile < depth_tiles_count; depth_tile++) {
                nk_size_t const depth_offset = depth_tile * 32;
                nk_size_t const valid_depth = (depth_tile < full_depth_tiles_count) ? 32 : depth_remainder;
                nk_dots_load_a_e5m2_sapphireamx_(&a_tile_top, a, a_stride, row_block_start, depth_offset,
                                                 rows_in_high_tile, valid_depth);
                if (rows_in_low_tile > 0)
                    nk_dots_load_a_e5m2_sapphireamx_(&a_tile_bottom, a, a_stride, row_block_start + 16, depth_offset,
                                                     rows_in_low_tile, valid_depth);
                nk_dots_bf16_accumulate_2x1_sapphireamx_(
                    &a_tile_top, &a_tile_bottom,
                    nk_dots_bf16_b_tile_sapphireamx_(b_tiles, column_tile * depth_tiles_count + depth_tile));
            }
            nk_dots_bf16_store_column_sapphireamx_(c + row_block_start * c_stride_elements + column_tile * 16,
                                                   c_stride_elements, rows_in_high_tile, rows_in_low_tile, 16);
        }

        if (column_remainder_count > 0) {
            nk_dots_bf16_zero_2x1_sapphireamx_();
            for (nk_size_t depth_tile = 0; depth_tile < depth_tiles_count; depth_tile++) {
                nk_size_t const depth_offset = depth_tile * 32;
                nk_size_t const valid_depth = (depth_tile < full_depth_tiles_count) ? 32 : depth_remainder;
                nk_dots_load_a_e5m2_sapphireamx_(&a_tile_top, a, a_stride, row_block_start, depth_offset,
                                                 rows_in_high_tile, valid_depth);
                if (rows_in_low_tile > 0)
                    nk_dots_load_a_e5m2_sapphireamx_(&a_tile_bottom, a, a_stride, row_block_start + 16, depth_offset,
                                                     rows_in_low_tile, valid_depth);
                nk_dots_bf16_accumulate_edge_sapphireamx_(&a_tile_top, &a_tile_bottom, column_edge, depth_offset, depth,
                                                          column_remainder_count, valid_depth);
            }
            nk_dots_bf16_store_column_sapphireamx_(c + row_block_start * c_stride_elements + full_columns,
                                                   c_stride_elements, rows_in_high_tile, rows_in_low_tile,
                                                   column_remainder_count);
        }
    }

    _tile_release();
    return nk_success_k;
}

/** Gram upper triangle of @p vectors widened to BF16, on AMX tiles, covering rows from
 *  @p rows_begin up to @p rows_end. */
NUMKONG_INLINE nk_status_t nk_gram_e5m2_sapphireamx_(                                    //
    nk_e5m2_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end) {
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    rows_end = nk_min_of_two(rows_end, vector_count);

    // Depth is walked in groups of 3 tiles × 32 elements
    nk_size_t const depth_tile_groups = nk_size_divide_round_up_(nk_size_divide_round_up_(depth, 32), 3);

    nk_dots_bf16_a16x32_sapphireamx_t a_tiles[3];
    nk_dots_bf16_a16x32_sapphireamx_t b_src_tiles[3];
    nk_dots_bf16_b32x16_sapphireamx_t b_tiles[3];
    nk_dots_bf16_state_sapphireamx_t state;

    nk_amx_tile_configure_sapphireamx_();

    for (nk_size_t row_tile = rows_begin; row_tile < rows_end; row_tile += 16) {
        nk_size_t const valid_rows = (row_tile + 16 <= rows_end) ? 16 : (rows_end - row_tile);

        for (nk_size_t column_tile = row_tile; column_tile < vector_count; column_tile += 16) {
            nk_size_t const valid_columns = (column_tile + 16 <= vector_count) ? 16 : (vector_count - column_tile);

            nk_dots_bf16_init_sapphireamx_(&state);

            for (nk_size_t depth_group = 0; depth_group < depth_tile_groups; depth_group++) {
                for (int tile_index = 0; tile_index < 3; tile_index++) {
                    nk_size_t const depth_start = depth_group * 96 + tile_index * 32;
                    nk_size_t const valid_depth = (depth_start + 32 <= depth)
                                                      ? 32
                                                      : (depth > depth_start ? depth - depth_start : 0);

                    nk_dots_load_a_e5m2_sapphireamx_(&a_tiles[tile_index], vectors, stride, row_tile, depth_start,
                                                     valid_rows, valid_depth);
                    if (row_tile == column_tile && valid_rows == valid_columns) {
                        nk_dots_pack_bf16_transposed_sapphireamx_(&a_tiles[tile_index], &b_tiles[tile_index]);
                    }
                    else {
                        nk_dots_load_a_e5m2_sapphireamx_(&b_src_tiles[tile_index], vectors, stride, column_tile,
                                                         depth_start, valid_columns, valid_depth);
                        nk_dots_pack_bf16_transposed_sapphireamx_(&b_src_tiles[tile_index], &b_tiles[tile_index]);
                    }
                }

                nk_dots_bf16_update_sapphireamx_( //
                    &state, &a_tiles[0], &a_tiles[1], &a_tiles[2], &b_tiles[0], &b_tiles[1], &b_tiles[2]);
            }

            nk_dots_symmetric_store_sapphireamx_(                                     //
                state.data, result + row_tile * result_stride_elements + column_tile, //
                result_stride_elements, valid_rows, valid_columns, column_tile - row_tile);
        }
    }
    return nk_success_k;
}

#pragma endregion E5M2 Floats

#pragma region NVFP4 Floats

/** NVFP4 scales are not rebased, so every row has base zero and spread zero. */
NUMKONG_INLINE nk_i32_t nk_nvfp4_base_sapphireamx_(nk_ue4m3_t const *scales, nk_size_t scales_stride, nk_size_t row,
                                                   nk_size_t depth, nk_i32_t *spread) {
    nk_unused_(scales), nk_unused_(scales_stride), nk_unused_(row), nk_unused_(depth);
    *spread = 0;
    return 0;
}

NUMKONG_INLINE __m512i nk_nvfp4_widen_bf16_sapphireamx_(nk_e2m1x2_t const *values, nk_size_t stride,
                                                        nk_ue4m3_t const *scales, nk_size_t scales_stride,
                                                        nk_i32_t base, nk_size_t row, nk_size_t first,
                                                        nk_size_t valid) {
    nk_unused_(base);
    __m512 low_f32x16, high_f32x16;
    nk_e2m1x32_to_f32x16x2_sapphireamx_(values, stride, row, first, valid, &low_f32x16, &high_f32x16);
    nk_ue4m3_t const *row_scales = scales + row * scales_stride;
    return nk_scaled_halves_to_bf16x32_sapphireamx_(
        low_f32x16, high_f32x16, nk_ue4m3_scale_sapphireamx_(row_scales, first / 16),
        valid > 16 ? nk_ue4m3_scale_sapphireamx_(row_scales, first / 16 + 1) : 0);
}

/** Squared norm of row @p row over @p depth elements of the BF16 values the GEMM multiplies,
 *  rebased by @p base, summed in F32. */
NUMKONG_INLINE nk_f32_t nk_nvfp4_sumsq_sapphireamx_(nk_e2m1x2_t const *values, nk_size_t stride,
                                                    nk_ue4m3_t const *scales, nk_size_t scales_stride, nk_i32_t base,
                                                    nk_size_t row, nk_size_t depth) {
    __m512 sum_f32x16 = _mm512_setzero_ps();
    for (nk_size_t first = 0; first < depth; first += 32)
        sum_f32x16 = nk_bf16x32_sumsq_step_sapphireamx_(
            sum_f32x16, nk_nvfp4_widen_bf16_sapphireamx_(values, stride, scales, scales_stride, base, row, first,
                                                         depth - first < 32 ? depth - first : 32));
    return _mm512_reduce_add_ps(sum_f32x16);
}

/** Fills @p bases with the rebasing exponents of @p count rows of @p scales from @p first_row, and
 *  zeros up to @p capacity. */
NUMKONG_INLINE void nk_dots_bases_nvfp4_sapphireamx_(nk_ue4m3_t const *scales, nk_size_t scales_stride,
                                                     nk_size_t first_row, nk_size_t count, nk_size_t capacity,
                                                     nk_size_t depth, nk_i32_t *bases) {
    nk_i32_t spread;
    for (nk_size_t row = 0; row < capacity; row++)
        bases[row] = row < count ? nk_nvfp4_base_sapphireamx_(scales, scales_stride, first_row + row, depth, &spread)
                                 : 0;
}

/** Loads rows @p first_row to `first_row + 16` of @p values and @p scales, row r rebased by
 *  @p bases[r], into a BF16 A tile, zeros past @p valid_rows rows and @p valid_columns elements
 *  from @p depth_offset. A tile past the depth stays zero without reading scales past the row. */
NUMKONG_INLINE void nk_dots_load_a_nvfp4_sapphireamx_(nk_dots_bf16_a16x32_sapphireamx_t *a_tile,
                                                      nk_e2m1x2_t const *values, nk_size_t stride,
                                                      nk_ue4m3_t const *scales, nk_size_t scales_stride,
                                                      nk_i32_t const *bases, nk_size_t first_row,
                                                      nk_size_t depth_offset, nk_size_t valid_rows,
                                                      nk_size_t valid_columns) {
    if (!valid_columns) valid_rows = 0;
    for (nk_size_t row = 0; row < 16; row++)
        _mm512_store_si512((__m512i *)a_tile->data[row],
                           row < valid_rows
                               ? nk_nvfp4_widen_bf16_sapphireamx_(values, stride, scales, scales_stride, bases[row],
                                                                  first_row + row, depth_offset, valid_columns)
                               : _mm512_setzero_si512());
    nk_compiler_barrier_sapphireamx_();
}

/** GEMM of @p a rows widened to BF16 against the columns packed by
 *  @c nk_dots_pack_nvfp4_sapphireamx, into F32 @p c: rows and columns rebased, before
 *  any tensor scale. */
NUMKONG_INLINE nk_status_t nk_gemm_packed_nvfp4_sapphireamx_(                            //
    nk_e2m1x2_t const *a, nk_ue4m3_t const *a_scales, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns_count, nk_size_t depth, nk_size_t a_stride, nk_size_t a_scales_stride,
    nk_size_t c_stride) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    if (header->capability != nk_cap_sapphireamx_k) return nk_pack_mismatch_k;
    nk_size_t const column_tiles_count = header->full_column_tiles;
    nk_size_t const depth_tiles_count = header->full_depth_tiles;
    nk_size_t const column_remainder_count = header->column_remainder_count;
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);

    // An empty depth gives zero sums, which the epilogues still finish
    if (depth_tiles_count == 0) {
        nk_dots_bf16_zero_results_sapphireamx_(c, rows, columns_count, c_stride_elements);
        return nk_success_k;
    }

    nk_bf16_t const *b_tiles = (nk_bf16_t const *)((char const *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_bf16_t const *column_edge = (nk_bf16_t const *)((char const *)b_packed + header->column_edge_offset);
    nk_size_t const full_columns = column_tiles_count * 16;
    nk_size_t const row_blocks_count = nk_size_divide_round_up_(rows, 32);
    nk_size_t const column_blocks_count = column_tiles_count / 2;
    nk_size_t const full_depth_tiles_count = depth / 32;
    nk_size_t const depth_remainder = depth % 32;
    nk_dots_bf16_a16x32_sapphireamx_t a_tile_top, a_tile_bottom;
    nk_i32_t row_bases[32];

    nk_amx_tile_configure_sapphireamx_();

    for (nk_size_t row_block = 0; row_block < row_blocks_count; row_block++) {
        nk_size_t const row_block_start = row_block * 32;
        nk_size_t const valid_rows_count = (row_block_start + 32 <= rows) ? 32 : (rows - row_block_start);
        nk_size_t const rows_in_high_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
        nk_size_t const rows_in_low_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;
        nk_dots_bases_nvfp4_sapphireamx_(a_scales, a_scales_stride, row_block_start, valid_rows_count, 32, depth,
                                         row_bases);

        for (nk_size_t column_block = 0; column_block < column_blocks_count; column_block++) {
            nk_dots_bf16_zero_2x2_sapphireamx_();
            for (nk_size_t depth_tile = 0; depth_tile < depth_tiles_count; depth_tile++) {
                nk_size_t const depth_offset = depth_tile * 32;
                nk_size_t const valid_depth = (depth_tile < full_depth_tiles_count) ? 32 : depth_remainder;
                nk_dots_load_a_nvfp4_sapphireamx_(&a_tile_top, a, a_stride, a_scales, a_scales_stride, row_bases,
                                                  row_block_start, depth_offset, rows_in_high_tile, valid_depth);
                if (rows_in_low_tile > 0)
                    nk_dots_load_a_nvfp4_sapphireamx_(&a_tile_bottom, a, a_stride, a_scales, a_scales_stride,
                                                      row_bases + 16, row_block_start + 16, depth_offset,
                                                      rows_in_low_tile, valid_depth);
                nk_dots_bf16_accumulate_2x2_sapphireamx_(
                    &a_tile_top, &a_tile_bottom,
                    nk_dots_bf16_b_tile_sapphireamx_(b_tiles, (column_block * 2) * depth_tiles_count + depth_tile),
                    nk_dots_bf16_b_tile_sapphireamx_(b_tiles, (column_block * 2 + 1) * depth_tiles_count + depth_tile));
            }
            nk_dots_bf16_store_block_sapphireamx_(c + row_block_start * c_stride_elements + column_block * 32, c_stride,
                                                  valid_rows_count);
        }

        if (column_tiles_count % 2 == 1) {
            nk_size_t const column_tile = column_tiles_count - 1;
            nk_dots_bf16_zero_2x1_sapphireamx_();
            for (nk_size_t depth_tile = 0; depth_tile < depth_tiles_count; depth_tile++) {
                nk_size_t const depth_offset = depth_tile * 32;
                nk_size_t const valid_depth = (depth_tile < full_depth_tiles_count) ? 32 : depth_remainder;
                nk_dots_load_a_nvfp4_sapphireamx_(&a_tile_top, a, a_stride, a_scales, a_scales_stride, row_bases,
                                                  row_block_start, depth_offset, rows_in_high_tile, valid_depth);
                if (rows_in_low_tile > 0)
                    nk_dots_load_a_nvfp4_sapphireamx_(&a_tile_bottom, a, a_stride, a_scales, a_scales_stride,
                                                      row_bases + 16, row_block_start + 16, depth_offset,
                                                      rows_in_low_tile, valid_depth);
                nk_dots_bf16_accumulate_2x1_sapphireamx_(
                    &a_tile_top, &a_tile_bottom,
                    nk_dots_bf16_b_tile_sapphireamx_(b_tiles, column_tile * depth_tiles_count + depth_tile));
            }
            nk_dots_bf16_store_column_sapphireamx_(c + row_block_start * c_stride_elements + column_tile * 16,
                                                   c_stride_elements, rows_in_high_tile, rows_in_low_tile, 16);
        }

        if (column_remainder_count > 0) {
            nk_dots_bf16_zero_2x1_sapphireamx_();
            for (nk_size_t depth_tile = 0; depth_tile < depth_tiles_count; depth_tile++) {
                nk_size_t const depth_offset = depth_tile * 32;
                nk_size_t const valid_depth = (depth_tile < full_depth_tiles_count) ? 32 : depth_remainder;
                nk_dots_load_a_nvfp4_sapphireamx_(&a_tile_top, a, a_stride, a_scales, a_scales_stride, row_bases,
                                                  row_block_start, depth_offset, rows_in_high_tile, valid_depth);
                if (rows_in_low_tile > 0)
                    nk_dots_load_a_nvfp4_sapphireamx_(&a_tile_bottom, a, a_stride, a_scales, a_scales_stride,
                                                      row_bases + 16, row_block_start + 16, depth_offset,
                                                      rows_in_low_tile, valid_depth);
                nk_dots_bf16_accumulate_edge_sapphireamx_(&a_tile_top, &a_tile_bottom, column_edge, depth_offset, depth,
                                                          column_remainder_count, valid_depth);
            }
            nk_dots_bf16_store_column_sapphireamx_(c + row_block_start * c_stride_elements + full_columns,
                                                   c_stride_elements, rows_in_high_tile, rows_in_low_tile,
                                                   column_remainder_count);
        }
    }

    _tile_release();
    return nk_success_k;
}

/** Gram upper triangle of @p vectors widened to BF16, on AMX tiles, covering rows from
 *  @p rows_begin up to @p rows_end: vectors rebased, before any tensor scale. */
NUMKONG_INLINE nk_status_t nk_gram_nvfp4_sapphireamx_(                                             //
    nk_e2m1x2_t const *vectors, nk_ue4m3_t const *scales, nk_size_t vector_count, nk_size_t depth, //
    nk_size_t stride, nk_size_t scales_stride, nk_f32_t *result, nk_size_t result_stride, nk_size_t rows_begin,
    nk_size_t rows_end) {
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    rows_end = nk_min_of_two(rows_end, vector_count);

    // Depth is walked in groups of 3 tiles × 32 elements
    nk_size_t const depth_tile_groups = nk_size_divide_round_up_(nk_size_divide_round_up_(depth, 32), 3);

    nk_dots_bf16_a16x32_sapphireamx_t a_tiles[3];
    nk_dots_bf16_a16x32_sapphireamx_t b_src_tiles[3];
    nk_dots_bf16_b32x16_sapphireamx_t b_tiles[3];
    nk_dots_bf16_state_sapphireamx_t state;
    nk_i32_t row_bases[16], column_bases[16];

    nk_amx_tile_configure_sapphireamx_();

    for (nk_size_t row_tile = rows_begin; row_tile < rows_end; row_tile += 16) {
        nk_size_t const valid_rows = (row_tile + 16 <= rows_end) ? 16 : (rows_end - row_tile);
        nk_dots_bases_nvfp4_sapphireamx_(scales, scales_stride, row_tile, valid_rows, 16, depth, row_bases);

        for (nk_size_t column_tile = row_tile; column_tile < vector_count; column_tile += 16) {
            nk_size_t const valid_columns = (column_tile + 16 <= vector_count) ? 16 : (vector_count - column_tile);
            nk_dots_bases_nvfp4_sapphireamx_(scales, scales_stride, column_tile, valid_columns, 16, depth,
                                             column_bases);

            nk_dots_bf16_init_sapphireamx_(&state);

            for (nk_size_t depth_group = 0; depth_group < depth_tile_groups; depth_group++) {
                for (int tile_index = 0; tile_index < 3; tile_index++) {
                    nk_size_t const depth_start = depth_group * 96 + tile_index * 32;
                    nk_size_t const valid_depth = (depth_start + 32 <= depth)
                                                      ? 32
                                                      : (depth > depth_start ? depth - depth_start : 0);

                    nk_dots_load_a_nvfp4_sapphireamx_(&a_tiles[tile_index], vectors, stride, scales, scales_stride,
                                                      row_bases, row_tile, depth_start, valid_rows, valid_depth);
                    if (row_tile == column_tile && valid_rows == valid_columns) {
                        nk_dots_pack_bf16_transposed_sapphireamx_(&a_tiles[tile_index], &b_tiles[tile_index]);
                    }
                    else {
                        nk_dots_load_a_nvfp4_sapphireamx_(&b_src_tiles[tile_index], vectors, stride, scales,
                                                          scales_stride, column_bases, column_tile, depth_start,
                                                          valid_columns, valid_depth);
                        nk_dots_pack_bf16_transposed_sapphireamx_(&b_src_tiles[tile_index], &b_tiles[tile_index]);
                    }
                }

                nk_dots_bf16_update_sapphireamx_( //
                    &state, &a_tiles[0], &a_tiles[1], &a_tiles[2], &b_tiles[0], &b_tiles[1], &b_tiles[2]);
            }

            nk_dots_symmetric_store_sapphireamx_(                                     //
                state.data, result + row_tile * result_stride_elements + column_tile, //
                result_stride_elements, valid_rows, valid_columns, column_tile - row_tile);
        }
    }
    return nk_success_k;
}

#pragma endregion NVFP4 Floats

#pragma region MXFP4 Floats

/** The rebasing exponent of row @p row over @p depth elements, its spread into @p spread. */
NUMKONG_INLINE nk_i32_t nk_mxfp4_base_sapphireamx_(nk_ue8m0_t const *scales, nk_size_t scales_stride, nk_size_t row,
                                                   nk_size_t depth, nk_i32_t *spread) {
    return nk_cross_scaled_base_serial_(scales + row * scales_stride, depth / 32, spread);
}

NUMKONG_INLINE __m512i nk_mxfp4_widen_bf16_sapphireamx_(nk_e2m1x2_t const *values, nk_size_t stride,
                                                        nk_ue8m0_t const *scales, nk_size_t scales_stride,
                                                        nk_i32_t base, nk_size_t row, nk_size_t first,
                                                        nk_size_t valid) {
    __m512 low_f32x16, high_f32x16;
    nk_e2m1x32_to_f32x16x2_sapphireamx_(values, stride, row, first, valid, &low_f32x16, &high_f32x16);
    nk_f32_t const scale = nk_relative_ue8m0_to_f32_serial_(scales[row * scales_stride + first / 32], base);
    return nk_scaled_halves_to_bf16x32_sapphireamx_(low_f32x16, high_f32x16, scale, scale);
}

/** Squared norm of row @p row over @p depth elements of the BF16 values the GEMM multiplies,
 *  rebased by @p base, summed in F32. */
NUMKONG_INLINE nk_f32_t nk_mxfp4_sumsq_sapphireamx_(nk_e2m1x2_t const *values, nk_size_t stride,
                                                    nk_ue8m0_t const *scales, nk_size_t scales_stride, nk_i32_t base,
                                                    nk_size_t row, nk_size_t depth) {
    __m512 sum_f32x16 = _mm512_setzero_ps();
    for (nk_size_t first = 0; first < depth; first += 32)
        sum_f32x16 = nk_bf16x32_sumsq_step_sapphireamx_(
            sum_f32x16, nk_mxfp4_widen_bf16_sapphireamx_(values, stride, scales, scales_stride, base, row, first,
                                                         depth - first < 32 ? depth - first : 32));
    return _mm512_reduce_add_ps(sum_f32x16);
}

/** Fills @p bases with the rebasing exponents of @p count rows of @p scales from @p first_row, and
 *  zeros up to @p capacity. */
NUMKONG_INLINE void nk_dots_bases_mxfp4_sapphireamx_(nk_ue8m0_t const *scales, nk_size_t scales_stride,
                                                     nk_size_t first_row, nk_size_t count, nk_size_t capacity,
                                                     nk_size_t depth, nk_i32_t *bases) {
    nk_i32_t spread;
    for (nk_size_t row = 0; row < capacity; row++)
        bases[row] = row < count ? nk_mxfp4_base_sapphireamx_(scales, scales_stride, first_row + row, depth, &spread)
                                 : 0;
}

/** Loads rows @p first_row to `first_row + 16` of @p values and @p scales, row r rebased by
 *  @p bases[r], into a BF16 A tile, zeros past @p valid_rows rows and @p valid_columns elements
 *  from @p depth_offset. A tile past the depth stays zero without reading scales past the row. */
NUMKONG_INLINE void nk_dots_load_a_mxfp4_sapphireamx_(nk_dots_bf16_a16x32_sapphireamx_t *a_tile,
                                                      nk_e2m1x2_t const *values, nk_size_t stride,
                                                      nk_ue8m0_t const *scales, nk_size_t scales_stride,
                                                      nk_i32_t const *bases, nk_size_t first_row,
                                                      nk_size_t depth_offset, nk_size_t valid_rows,
                                                      nk_size_t valid_columns) {
    if (!valid_columns) valid_rows = 0;
    for (nk_size_t row = 0; row < 16; row++)
        _mm512_store_si512((__m512i *)a_tile->data[row],
                           row < valid_rows
                               ? nk_mxfp4_widen_bf16_sapphireamx_(values, stride, scales, scales_stride, bases[row],
                                                                  first_row + row, depth_offset, valid_columns)
                               : _mm512_setzero_si512());
    nk_compiler_barrier_sapphireamx_();
}

/** GEMM of @p a rows widened to BF16 against the columns packed by
 *  @c nk_dots_pack_mxfp4_sapphireamx, into F32 @p c: rows and columns rebased, before
 *  any tensor scale. */
NUMKONG_INLINE nk_status_t nk_gemm_packed_mxfp4_sapphireamx_(                            //
    nk_e2m1x2_t const *a, nk_ue8m0_t const *a_scales, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns_count, nk_size_t depth, nk_size_t a_stride, nk_size_t a_scales_stride,
    nk_size_t c_stride) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    if (header->capability != nk_cap_sapphireamx_k) return nk_pack_mismatch_k;
    nk_size_t const column_tiles_count = header->full_column_tiles;
    nk_size_t const depth_tiles_count = header->full_depth_tiles;
    nk_size_t const column_remainder_count = header->column_remainder_count;
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);

    // An empty depth gives zero sums, which the epilogues still finish
    if (depth_tiles_count == 0) {
        nk_dots_bf16_zero_results_sapphireamx_(c, rows, columns_count, c_stride_elements);
        return nk_success_k;
    }

    nk_bf16_t const *b_tiles = (nk_bf16_t const *)((char const *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_bf16_t const *column_edge = (nk_bf16_t const *)((char const *)b_packed + header->column_edge_offset);
    nk_size_t const full_columns = column_tiles_count * 16;
    nk_size_t const row_blocks_count = nk_size_divide_round_up_(rows, 32);
    nk_size_t const column_blocks_count = column_tiles_count / 2;
    nk_size_t const full_depth_tiles_count = depth / 32;
    nk_size_t const depth_remainder = depth % 32;
    nk_dots_bf16_a16x32_sapphireamx_t a_tile_top, a_tile_bottom;
    nk_i32_t row_bases[32];

    nk_amx_tile_configure_sapphireamx_();

    for (nk_size_t row_block = 0; row_block < row_blocks_count; row_block++) {
        nk_size_t const row_block_start = row_block * 32;
        nk_size_t const valid_rows_count = (row_block_start + 32 <= rows) ? 32 : (rows - row_block_start);
        nk_size_t const rows_in_high_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
        nk_size_t const rows_in_low_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;
        nk_dots_bases_mxfp4_sapphireamx_(a_scales, a_scales_stride, row_block_start, valid_rows_count, 32, depth,
                                         row_bases);

        for (nk_size_t column_block = 0; column_block < column_blocks_count; column_block++) {
            nk_dots_bf16_zero_2x2_sapphireamx_();
            for (nk_size_t depth_tile = 0; depth_tile < depth_tiles_count; depth_tile++) {
                nk_size_t const depth_offset = depth_tile * 32;
                nk_size_t const valid_depth = (depth_tile < full_depth_tiles_count) ? 32 : depth_remainder;
                nk_dots_load_a_mxfp4_sapphireamx_(&a_tile_top, a, a_stride, a_scales, a_scales_stride, row_bases,
                                                  row_block_start, depth_offset, rows_in_high_tile, valid_depth);
                if (rows_in_low_tile > 0)
                    nk_dots_load_a_mxfp4_sapphireamx_(&a_tile_bottom, a, a_stride, a_scales, a_scales_stride,
                                                      row_bases + 16, row_block_start + 16, depth_offset,
                                                      rows_in_low_tile, valid_depth);
                nk_dots_bf16_accumulate_2x2_sapphireamx_(
                    &a_tile_top, &a_tile_bottom,
                    nk_dots_bf16_b_tile_sapphireamx_(b_tiles, (column_block * 2) * depth_tiles_count + depth_tile),
                    nk_dots_bf16_b_tile_sapphireamx_(b_tiles, (column_block * 2 + 1) * depth_tiles_count + depth_tile));
            }
            nk_dots_bf16_store_block_sapphireamx_(c + row_block_start * c_stride_elements + column_block * 32, c_stride,
                                                  valid_rows_count);
        }

        if (column_tiles_count % 2 == 1) {
            nk_size_t const column_tile = column_tiles_count - 1;
            nk_dots_bf16_zero_2x1_sapphireamx_();
            for (nk_size_t depth_tile = 0; depth_tile < depth_tiles_count; depth_tile++) {
                nk_size_t const depth_offset = depth_tile * 32;
                nk_size_t const valid_depth = (depth_tile < full_depth_tiles_count) ? 32 : depth_remainder;
                nk_dots_load_a_mxfp4_sapphireamx_(&a_tile_top, a, a_stride, a_scales, a_scales_stride, row_bases,
                                                  row_block_start, depth_offset, rows_in_high_tile, valid_depth);
                if (rows_in_low_tile > 0)
                    nk_dots_load_a_mxfp4_sapphireamx_(&a_tile_bottom, a, a_stride, a_scales, a_scales_stride,
                                                      row_bases + 16, row_block_start + 16, depth_offset,
                                                      rows_in_low_tile, valid_depth);
                nk_dots_bf16_accumulate_2x1_sapphireamx_(
                    &a_tile_top, &a_tile_bottom,
                    nk_dots_bf16_b_tile_sapphireamx_(b_tiles, column_tile * depth_tiles_count + depth_tile));
            }
            nk_dots_bf16_store_column_sapphireamx_(c + row_block_start * c_stride_elements + column_tile * 16,
                                                   c_stride_elements, rows_in_high_tile, rows_in_low_tile, 16);
        }

        if (column_remainder_count > 0) {
            nk_dots_bf16_zero_2x1_sapphireamx_();
            for (nk_size_t depth_tile = 0; depth_tile < depth_tiles_count; depth_tile++) {
                nk_size_t const depth_offset = depth_tile * 32;
                nk_size_t const valid_depth = (depth_tile < full_depth_tiles_count) ? 32 : depth_remainder;
                nk_dots_load_a_mxfp4_sapphireamx_(&a_tile_top, a, a_stride, a_scales, a_scales_stride, row_bases,
                                                  row_block_start, depth_offset, rows_in_high_tile, valid_depth);
                if (rows_in_low_tile > 0)
                    nk_dots_load_a_mxfp4_sapphireamx_(&a_tile_bottom, a, a_stride, a_scales, a_scales_stride,
                                                      row_bases + 16, row_block_start + 16, depth_offset,
                                                      rows_in_low_tile, valid_depth);
                nk_dots_bf16_accumulate_edge_sapphireamx_(&a_tile_top, &a_tile_bottom, column_edge, depth_offset, depth,
                                                          column_remainder_count, valid_depth);
            }
            nk_dots_bf16_store_column_sapphireamx_(c + row_block_start * c_stride_elements + full_columns,
                                                   c_stride_elements, rows_in_high_tile, rows_in_low_tile,
                                                   column_remainder_count);
        }
    }

    _tile_release();
    return nk_success_k;
}

/** Gram upper triangle of @p vectors widened to BF16, on AMX tiles, covering rows from
 *  @p rows_begin up to @p rows_end: vectors rebased, before any tensor scale. */
NUMKONG_INLINE nk_status_t nk_gram_mxfp4_sapphireamx_(                                             //
    nk_e2m1x2_t const *vectors, nk_ue8m0_t const *scales, nk_size_t vector_count, nk_size_t depth, //
    nk_size_t stride, nk_size_t scales_stride, nk_f32_t *result, nk_size_t result_stride, nk_size_t rows_begin,
    nk_size_t rows_end) {
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    rows_end = nk_min_of_two(rows_end, vector_count);

    // Depth is walked in groups of 3 tiles × 32 elements
    nk_size_t const depth_tile_groups = nk_size_divide_round_up_(nk_size_divide_round_up_(depth, 32), 3);

    nk_dots_bf16_a16x32_sapphireamx_t a_tiles[3];
    nk_dots_bf16_a16x32_sapphireamx_t b_src_tiles[3];
    nk_dots_bf16_b32x16_sapphireamx_t b_tiles[3];
    nk_dots_bf16_state_sapphireamx_t state;
    nk_i32_t row_bases[16], column_bases[16];

    nk_amx_tile_configure_sapphireamx_();

    for (nk_size_t row_tile = rows_begin; row_tile < rows_end; row_tile += 16) {
        nk_size_t const valid_rows = (row_tile + 16 <= rows_end) ? 16 : (rows_end - row_tile);
        nk_dots_bases_mxfp4_sapphireamx_(scales, scales_stride, row_tile, valid_rows, 16, depth, row_bases);

        for (nk_size_t column_tile = row_tile; column_tile < vector_count; column_tile += 16) {
            nk_size_t const valid_columns = (column_tile + 16 <= vector_count) ? 16 : (vector_count - column_tile);
            nk_dots_bases_mxfp4_sapphireamx_(scales, scales_stride, column_tile, valid_columns, 16, depth,
                                             column_bases);

            nk_dots_bf16_init_sapphireamx_(&state);

            for (nk_size_t depth_group = 0; depth_group < depth_tile_groups; depth_group++) {
                for (int tile_index = 0; tile_index < 3; tile_index++) {
                    nk_size_t const depth_start = depth_group * 96 + tile_index * 32;
                    nk_size_t const valid_depth = (depth_start + 32 <= depth)
                                                      ? 32
                                                      : (depth > depth_start ? depth - depth_start : 0);

                    nk_dots_load_a_mxfp4_sapphireamx_(&a_tiles[tile_index], vectors, stride, scales, scales_stride,
                                                      row_bases, row_tile, depth_start, valid_rows, valid_depth);
                    if (row_tile == column_tile && valid_rows == valid_columns) {
                        nk_dots_pack_bf16_transposed_sapphireamx_(&a_tiles[tile_index], &b_tiles[tile_index]);
                    }
                    else {
                        nk_dots_load_a_mxfp4_sapphireamx_(&b_src_tiles[tile_index], vectors, stride, scales,
                                                          scales_stride, column_bases, column_tile, depth_start,
                                                          valid_columns, valid_depth);
                        nk_dots_pack_bf16_transposed_sapphireamx_(&b_src_tiles[tile_index], &b_tiles[tile_index]);
                    }
                }

                nk_dots_bf16_update_sapphireamx_( //
                    &state, &a_tiles[0], &a_tiles[1], &a_tiles[2], &b_tiles[0], &b_tiles[1], &b_tiles[2]);
            }

            nk_dots_symmetric_store_sapphireamx_(                                     //
                state.data, result + row_tile * result_stride_elements + column_tile, //
                result_stride_elements, valid_rows, valid_columns, column_tile - row_tile);
        }
    }
    return nk_success_k;
}

#pragma endregion MXFP4 Floats

#pragma region MXFP8E4M3 Floats

/** The rebasing exponent of row @p row over @p depth elements, its spread into @p spread. */
NUMKONG_INLINE nk_i32_t nk_mxfp8e4m3_base_sapphireamx_(nk_ue8m0_t const *scales, nk_size_t scales_stride, nk_size_t row,
                                                       nk_size_t depth, nk_i32_t *spread) {
    return nk_cross_scaled_base_serial_(scales + row * scales_stride, depth / 32, spread);
}

NUMKONG_INLINE __m512i nk_mxfp8e4m3_widen_bf16_sapphireamx_(nk_e4m3_t const *values, nk_size_t stride,
                                                            nk_ue8m0_t const *scales, nk_size_t scales_stride,
                                                            nk_i32_t base, nk_size_t row, nk_size_t first,
                                                            nk_size_t valid) {
    __m256i const bytes_u8x32 = _mm256_maskz_loadu_epi8(nk_dots_bf16_valid_mask_sapphireamx_(valid),
                                                        values + row * stride + first);
    nk_f32_t const scale = nk_relative_ue8m0_to_f32_serial_(scales[row * scales_stride + first / 32], base);
    return nk_scaled_halves_to_bf16x32_sapphireamx_(
        nk_e4m3x16_to_f32x16_skylake_(_mm256_castsi256_si128(bytes_u8x32)),
        nk_e4m3x16_to_f32x16_skylake_(_mm256_extracti128_si256(bytes_u8x32, 1)), scale, scale);
}

/** Squared norm of row @p row over @p depth elements of the BF16 values the GEMM multiplies,
 *  rebased by @p base, summed in F32. */
NUMKONG_INLINE nk_f32_t nk_mxfp8e4m3_sumsq_sapphireamx_(nk_e4m3_t const *values, nk_size_t stride,
                                                        nk_ue8m0_t const *scales, nk_size_t scales_stride,
                                                        nk_i32_t base, nk_size_t row, nk_size_t depth) {
    __m512 sum_f32x16 = _mm512_setzero_ps();
    for (nk_size_t first = 0; first < depth; first += 32)
        sum_f32x16 = nk_bf16x32_sumsq_step_sapphireamx_(
            sum_f32x16, nk_mxfp8e4m3_widen_bf16_sapphireamx_(values, stride, scales, scales_stride, base, row, first,
                                                             depth - first < 32 ? depth - first : 32));
    return _mm512_reduce_add_ps(sum_f32x16);
}

/** Fills @p bases with the rebasing exponents of @p count rows of @p scales from @p first_row, and
 *  zeros up to @p capacity. */
NUMKONG_INLINE void nk_dots_bases_mxfp8e4m3_sapphireamx_(nk_ue8m0_t const *scales, nk_size_t scales_stride,
                                                         nk_size_t first_row, nk_size_t count, nk_size_t capacity,
                                                         nk_size_t depth, nk_i32_t *bases) {
    nk_i32_t spread;
    for (nk_size_t row = 0; row < capacity; row++)
        bases[row] = row < count
                         ? nk_mxfp8e4m3_base_sapphireamx_(scales, scales_stride, first_row + row, depth, &spread)
                         : 0;
}

/** Loads rows @p first_row to `first_row + 16` of @p values and @p scales, row r rebased by
 *  @p bases[r], into a BF16 A tile, zeros past @p valid_rows rows and @p valid_columns elements
 *  from @p depth_offset. A tile past the depth stays zero without reading scales past the row. */
NUMKONG_INLINE void nk_dots_load_a_mxfp8e4m3_sapphireamx_(nk_dots_bf16_a16x32_sapphireamx_t *a_tile,
                                                          nk_e4m3_t const *values, nk_size_t stride,
                                                          nk_ue8m0_t const *scales, nk_size_t scales_stride,
                                                          nk_i32_t const *bases, nk_size_t first_row,
                                                          nk_size_t depth_offset, nk_size_t valid_rows,
                                                          nk_size_t valid_columns) {
    if (!valid_columns) valid_rows = 0;
    for (nk_size_t row = 0; row < 16; row++)
        _mm512_store_si512((__m512i *)a_tile->data[row],
                           row < valid_rows
                               ? nk_mxfp8e4m3_widen_bf16_sapphireamx_(values, stride, scales, scales_stride, bases[row],
                                                                      first_row + row, depth_offset, valid_columns)
                               : _mm512_setzero_si512());
    nk_compiler_barrier_sapphireamx_();
}

/** GEMM of @p a rows widened to BF16 against the columns packed by
 *  @c nk_dots_pack_mxfp8e4m3_sapphireamx, into F32 @p c: rows and columns rebased, before
 *  any tensor scale. */
NUMKONG_INLINE nk_status_t nk_gemm_packed_mxfp8e4m3_sapphireamx_(                      //
    nk_e4m3_t const *a, nk_ue8m0_t const *a_scales, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns_count, nk_size_t depth, nk_size_t a_stride, nk_size_t a_scales_stride,
    nk_size_t c_stride) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    if (header->capability != nk_cap_sapphireamx_k) return nk_pack_mismatch_k;
    nk_size_t const column_tiles_count = header->full_column_tiles;
    nk_size_t const depth_tiles_count = header->full_depth_tiles;
    nk_size_t const column_remainder_count = header->column_remainder_count;
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);

    // An empty depth gives zero sums, which the epilogues still finish
    if (depth_tiles_count == 0) {
        nk_dots_bf16_zero_results_sapphireamx_(c, rows, columns_count, c_stride_elements);
        return nk_success_k;
    }

    nk_bf16_t const *b_tiles = (nk_bf16_t const *)((char const *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_bf16_t const *column_edge = (nk_bf16_t const *)((char const *)b_packed + header->column_edge_offset);
    nk_size_t const full_columns = column_tiles_count * 16;
    nk_size_t const row_blocks_count = nk_size_divide_round_up_(rows, 32);
    nk_size_t const column_blocks_count = column_tiles_count / 2;
    nk_size_t const full_depth_tiles_count = depth / 32;
    nk_size_t const depth_remainder = depth % 32;
    nk_dots_bf16_a16x32_sapphireamx_t a_tile_top, a_tile_bottom;
    nk_i32_t row_bases[32];

    nk_amx_tile_configure_sapphireamx_();

    for (nk_size_t row_block = 0; row_block < row_blocks_count; row_block++) {
        nk_size_t const row_block_start = row_block * 32;
        nk_size_t const valid_rows_count = (row_block_start + 32 <= rows) ? 32 : (rows - row_block_start);
        nk_size_t const rows_in_high_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
        nk_size_t const rows_in_low_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;
        nk_dots_bases_mxfp8e4m3_sapphireamx_(a_scales, a_scales_stride, row_block_start, valid_rows_count, 32, depth,
                                             row_bases);

        for (nk_size_t column_block = 0; column_block < column_blocks_count; column_block++) {
            nk_dots_bf16_zero_2x2_sapphireamx_();
            for (nk_size_t depth_tile = 0; depth_tile < depth_tiles_count; depth_tile++) {
                nk_size_t const depth_offset = depth_tile * 32;
                nk_size_t const valid_depth = (depth_tile < full_depth_tiles_count) ? 32 : depth_remainder;
                nk_dots_load_a_mxfp8e4m3_sapphireamx_(&a_tile_top, a, a_stride, a_scales, a_scales_stride, row_bases,
                                                      row_block_start, depth_offset, rows_in_high_tile, valid_depth);
                if (rows_in_low_tile > 0)
                    nk_dots_load_a_mxfp8e4m3_sapphireamx_(&a_tile_bottom, a, a_stride, a_scales, a_scales_stride,
                                                          row_bases + 16, row_block_start + 16, depth_offset,
                                                          rows_in_low_tile, valid_depth);
                nk_dots_bf16_accumulate_2x2_sapphireamx_(
                    &a_tile_top, &a_tile_bottom,
                    nk_dots_bf16_b_tile_sapphireamx_(b_tiles, (column_block * 2) * depth_tiles_count + depth_tile),
                    nk_dots_bf16_b_tile_sapphireamx_(b_tiles, (column_block * 2 + 1) * depth_tiles_count + depth_tile));
            }
            nk_dots_bf16_store_block_sapphireamx_(c + row_block_start * c_stride_elements + column_block * 32, c_stride,
                                                  valid_rows_count);
        }

        if (column_tiles_count % 2 == 1) {
            nk_size_t const column_tile = column_tiles_count - 1;
            nk_dots_bf16_zero_2x1_sapphireamx_();
            for (nk_size_t depth_tile = 0; depth_tile < depth_tiles_count; depth_tile++) {
                nk_size_t const depth_offset = depth_tile * 32;
                nk_size_t const valid_depth = (depth_tile < full_depth_tiles_count) ? 32 : depth_remainder;
                nk_dots_load_a_mxfp8e4m3_sapphireamx_(&a_tile_top, a, a_stride, a_scales, a_scales_stride, row_bases,
                                                      row_block_start, depth_offset, rows_in_high_tile, valid_depth);
                if (rows_in_low_tile > 0)
                    nk_dots_load_a_mxfp8e4m3_sapphireamx_(&a_tile_bottom, a, a_stride, a_scales, a_scales_stride,
                                                          row_bases + 16, row_block_start + 16, depth_offset,
                                                          rows_in_low_tile, valid_depth);
                nk_dots_bf16_accumulate_2x1_sapphireamx_(
                    &a_tile_top, &a_tile_bottom,
                    nk_dots_bf16_b_tile_sapphireamx_(b_tiles, column_tile * depth_tiles_count + depth_tile));
            }
            nk_dots_bf16_store_column_sapphireamx_(c + row_block_start * c_stride_elements + column_tile * 16,
                                                   c_stride_elements, rows_in_high_tile, rows_in_low_tile, 16);
        }

        if (column_remainder_count > 0) {
            nk_dots_bf16_zero_2x1_sapphireamx_();
            for (nk_size_t depth_tile = 0; depth_tile < depth_tiles_count; depth_tile++) {
                nk_size_t const depth_offset = depth_tile * 32;
                nk_size_t const valid_depth = (depth_tile < full_depth_tiles_count) ? 32 : depth_remainder;
                nk_dots_load_a_mxfp8e4m3_sapphireamx_(&a_tile_top, a, a_stride, a_scales, a_scales_stride, row_bases,
                                                      row_block_start, depth_offset, rows_in_high_tile, valid_depth);
                if (rows_in_low_tile > 0)
                    nk_dots_load_a_mxfp8e4m3_sapphireamx_(&a_tile_bottom, a, a_stride, a_scales, a_scales_stride,
                                                          row_bases + 16, row_block_start + 16, depth_offset,
                                                          rows_in_low_tile, valid_depth);
                nk_dots_bf16_accumulate_edge_sapphireamx_(&a_tile_top, &a_tile_bottom, column_edge, depth_offset, depth,
                                                          column_remainder_count, valid_depth);
            }
            nk_dots_bf16_store_column_sapphireamx_(c + row_block_start * c_stride_elements + full_columns,
                                                   c_stride_elements, rows_in_high_tile, rows_in_low_tile,
                                                   column_remainder_count);
        }
    }

    _tile_release();
    return nk_success_k;
}

/** Gram upper triangle of @p vectors widened to BF16, on AMX tiles, covering rows from
 *  @p rows_begin up to @p rows_end: vectors rebased, before any tensor scale. */
NUMKONG_INLINE nk_status_t nk_gram_mxfp8e4m3_sapphireamx_(                                       //
    nk_e4m3_t const *vectors, nk_ue8m0_t const *scales, nk_size_t vector_count, nk_size_t depth, //
    nk_size_t stride, nk_size_t scales_stride, nk_f32_t *result, nk_size_t result_stride, nk_size_t rows_begin,
    nk_size_t rows_end) {
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    rows_end = nk_min_of_two(rows_end, vector_count);

    // Depth is walked in groups of 3 tiles × 32 elements
    nk_size_t const depth_tile_groups = nk_size_divide_round_up_(nk_size_divide_round_up_(depth, 32), 3);

    nk_dots_bf16_a16x32_sapphireamx_t a_tiles[3];
    nk_dots_bf16_a16x32_sapphireamx_t b_src_tiles[3];
    nk_dots_bf16_b32x16_sapphireamx_t b_tiles[3];
    nk_dots_bf16_state_sapphireamx_t state;
    nk_i32_t row_bases[16], column_bases[16];

    nk_amx_tile_configure_sapphireamx_();

    for (nk_size_t row_tile = rows_begin; row_tile < rows_end; row_tile += 16) {
        nk_size_t const valid_rows = (row_tile + 16 <= rows_end) ? 16 : (rows_end - row_tile);
        nk_dots_bases_mxfp8e4m3_sapphireamx_(scales, scales_stride, row_tile, valid_rows, 16, depth, row_bases);

        for (nk_size_t column_tile = row_tile; column_tile < vector_count; column_tile += 16) {
            nk_size_t const valid_columns = (column_tile + 16 <= vector_count) ? 16 : (vector_count - column_tile);
            nk_dots_bases_mxfp8e4m3_sapphireamx_(scales, scales_stride, column_tile, valid_columns, 16, depth,
                                                 column_bases);

            nk_dots_bf16_init_sapphireamx_(&state);

            for (nk_size_t depth_group = 0; depth_group < depth_tile_groups; depth_group++) {
                for (int tile_index = 0; tile_index < 3; tile_index++) {
                    nk_size_t const depth_start = depth_group * 96 + tile_index * 32;
                    nk_size_t const valid_depth = (depth_start + 32 <= depth)
                                                      ? 32
                                                      : (depth > depth_start ? depth - depth_start : 0);

                    nk_dots_load_a_mxfp8e4m3_sapphireamx_(&a_tiles[tile_index], vectors, stride, scales, scales_stride,
                                                          row_bases, row_tile, depth_start, valid_rows, valid_depth);
                    if (row_tile == column_tile && valid_rows == valid_columns) {
                        nk_dots_pack_bf16_transposed_sapphireamx_(&a_tiles[tile_index], &b_tiles[tile_index]);
                    }
                    else {
                        nk_dots_load_a_mxfp8e4m3_sapphireamx_(&b_src_tiles[tile_index], vectors, stride, scales,
                                                              scales_stride, column_bases, column_tile, depth_start,
                                                              valid_columns, valid_depth);
                        nk_dots_pack_bf16_transposed_sapphireamx_(&b_src_tiles[tile_index], &b_tiles[tile_index]);
                    }
                }

                nk_dots_bf16_update_sapphireamx_( //
                    &state, &a_tiles[0], &a_tiles[1], &a_tiles[2], &b_tiles[0], &b_tiles[1], &b_tiles[2]);
            }

            nk_dots_symmetric_store_sapphireamx_(                                     //
                state.data, result + row_tile * result_stride_elements + column_tile, //
                result_stride_elements, valid_rows, valid_columns, column_tile - row_tile);
        }
    }
    return nk_success_k;
}

#pragma endregion MXFP8E4M3 Floats

#pragma region MXFP8E5M2 Floats

/** The rebasing exponent of row @p row over @p depth elements, its spread into @p spread. */
NUMKONG_INLINE nk_i32_t nk_mxfp8e5m2_base_sapphireamx_(nk_ue8m0_t const *scales, nk_size_t scales_stride, nk_size_t row,
                                                       nk_size_t depth, nk_i32_t *spread) {
    return nk_cross_scaled_base_serial_(scales + row * scales_stride, depth / 32, spread);
}

NUMKONG_INLINE __m512i nk_mxfp8e5m2_widen_bf16_sapphireamx_(nk_e5m2_t const *values, nk_size_t stride,
                                                            nk_ue8m0_t const *scales, nk_size_t scales_stride,
                                                            nk_i32_t base, nk_size_t row, nk_size_t first,
                                                            nk_size_t valid) {
    __m256i const bytes_u8x32 = _mm256_maskz_loadu_epi8(nk_dots_bf16_valid_mask_sapphireamx_(valid),
                                                        values + row * stride + first);
    nk_f32_t const scale = nk_relative_ue8m0_to_f32_serial_(scales[row * scales_stride + first / 32], base);
    return nk_scaled_halves_to_bf16x32_sapphireamx_(
        nk_e5m2x16_to_f32x16_skylake_(_mm256_castsi256_si128(bytes_u8x32)),
        nk_e5m2x16_to_f32x16_skylake_(_mm256_extracti128_si256(bytes_u8x32, 1)), scale, scale);
}

/** Squared norm of row @p row over @p depth elements of the BF16 values the GEMM multiplies,
 *  rebased by @p base, summed in F32. */
NUMKONG_INLINE nk_f32_t nk_mxfp8e5m2_sumsq_sapphireamx_(nk_e5m2_t const *values, nk_size_t stride,
                                                        nk_ue8m0_t const *scales, nk_size_t scales_stride,
                                                        nk_i32_t base, nk_size_t row, nk_size_t depth) {
    __m512 sum_f32x16 = _mm512_setzero_ps();
    for (nk_size_t first = 0; first < depth; first += 32)
        sum_f32x16 = nk_bf16x32_sumsq_step_sapphireamx_(
            sum_f32x16, nk_mxfp8e5m2_widen_bf16_sapphireamx_(values, stride, scales, scales_stride, base, row, first,
                                                             depth - first < 32 ? depth - first : 32));
    return _mm512_reduce_add_ps(sum_f32x16);
}

/** Fills @p bases with the rebasing exponents of @p count rows of @p scales from @p first_row, and
 *  zeros up to @p capacity. */
NUMKONG_INLINE void nk_dots_bases_mxfp8e5m2_sapphireamx_(nk_ue8m0_t const *scales, nk_size_t scales_stride,
                                                         nk_size_t first_row, nk_size_t count, nk_size_t capacity,
                                                         nk_size_t depth, nk_i32_t *bases) {
    nk_i32_t spread;
    for (nk_size_t row = 0; row < capacity; row++)
        bases[row] = row < count
                         ? nk_mxfp8e5m2_base_sapphireamx_(scales, scales_stride, first_row + row, depth, &spread)
                         : 0;
}

/** Loads rows @p first_row to `first_row + 16` of @p values and @p scales, row r rebased by
 *  @p bases[r], into a BF16 A tile, zeros past @p valid_rows rows and @p valid_columns elements
 *  from @p depth_offset. A tile past the depth stays zero without reading scales past the row. */
NUMKONG_INLINE void nk_dots_load_a_mxfp8e5m2_sapphireamx_(nk_dots_bf16_a16x32_sapphireamx_t *a_tile,
                                                          nk_e5m2_t const *values, nk_size_t stride,
                                                          nk_ue8m0_t const *scales, nk_size_t scales_stride,
                                                          nk_i32_t const *bases, nk_size_t first_row,
                                                          nk_size_t depth_offset, nk_size_t valid_rows,
                                                          nk_size_t valid_columns) {
    if (!valid_columns) valid_rows = 0;
    for (nk_size_t row = 0; row < 16; row++)
        _mm512_store_si512((__m512i *)a_tile->data[row],
                           row < valid_rows
                               ? nk_mxfp8e5m2_widen_bf16_sapphireamx_(values, stride, scales, scales_stride, bases[row],
                                                                      first_row + row, depth_offset, valid_columns)
                               : _mm512_setzero_si512());
    nk_compiler_barrier_sapphireamx_();
}

/** GEMM of @p a rows widened to BF16 against the columns packed by
 *  @c nk_dots_pack_mxfp8e5m2_sapphireamx, into F32 @p c: rows and columns rebased, before
 *  any tensor scale. */
NUMKONG_INLINE nk_status_t nk_gemm_packed_mxfp8e5m2_sapphireamx_(                      //
    nk_e5m2_t const *a, nk_ue8m0_t const *a_scales, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns_count, nk_size_t depth, nk_size_t a_stride, nk_size_t a_scales_stride,
    nk_size_t c_stride) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    if (header->capability != nk_cap_sapphireamx_k) return nk_pack_mismatch_k;
    nk_size_t const column_tiles_count = header->full_column_tiles;
    nk_size_t const depth_tiles_count = header->full_depth_tiles;
    nk_size_t const column_remainder_count = header->column_remainder_count;
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);

    // An empty depth gives zero sums, which the epilogues still finish
    if (depth_tiles_count == 0) {
        nk_dots_bf16_zero_results_sapphireamx_(c, rows, columns_count, c_stride_elements);
        return nk_success_k;
    }

    nk_bf16_t const *b_tiles = (nk_bf16_t const *)((char const *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_bf16_t const *column_edge = (nk_bf16_t const *)((char const *)b_packed + header->column_edge_offset);
    nk_size_t const full_columns = column_tiles_count * 16;
    nk_size_t const row_blocks_count = nk_size_divide_round_up_(rows, 32);
    nk_size_t const column_blocks_count = column_tiles_count / 2;
    nk_size_t const full_depth_tiles_count = depth / 32;
    nk_size_t const depth_remainder = depth % 32;
    nk_dots_bf16_a16x32_sapphireamx_t a_tile_top, a_tile_bottom;
    nk_i32_t row_bases[32];

    nk_amx_tile_configure_sapphireamx_();

    for (nk_size_t row_block = 0; row_block < row_blocks_count; row_block++) {
        nk_size_t const row_block_start = row_block * 32;
        nk_size_t const valid_rows_count = (row_block_start + 32 <= rows) ? 32 : (rows - row_block_start);
        nk_size_t const rows_in_high_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
        nk_size_t const rows_in_low_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;
        nk_dots_bases_mxfp8e5m2_sapphireamx_(a_scales, a_scales_stride, row_block_start, valid_rows_count, 32, depth,
                                             row_bases);

        for (nk_size_t column_block = 0; column_block < column_blocks_count; column_block++) {
            nk_dots_bf16_zero_2x2_sapphireamx_();
            for (nk_size_t depth_tile = 0; depth_tile < depth_tiles_count; depth_tile++) {
                nk_size_t const depth_offset = depth_tile * 32;
                nk_size_t const valid_depth = (depth_tile < full_depth_tiles_count) ? 32 : depth_remainder;
                nk_dots_load_a_mxfp8e5m2_sapphireamx_(&a_tile_top, a, a_stride, a_scales, a_scales_stride, row_bases,
                                                      row_block_start, depth_offset, rows_in_high_tile, valid_depth);
                if (rows_in_low_tile > 0)
                    nk_dots_load_a_mxfp8e5m2_sapphireamx_(&a_tile_bottom, a, a_stride, a_scales, a_scales_stride,
                                                          row_bases + 16, row_block_start + 16, depth_offset,
                                                          rows_in_low_tile, valid_depth);
                nk_dots_bf16_accumulate_2x2_sapphireamx_(
                    &a_tile_top, &a_tile_bottom,
                    nk_dots_bf16_b_tile_sapphireamx_(b_tiles, (column_block * 2) * depth_tiles_count + depth_tile),
                    nk_dots_bf16_b_tile_sapphireamx_(b_tiles, (column_block * 2 + 1) * depth_tiles_count + depth_tile));
            }
            nk_dots_bf16_store_block_sapphireamx_(c + row_block_start * c_stride_elements + column_block * 32, c_stride,
                                                  valid_rows_count);
        }

        if (column_tiles_count % 2 == 1) {
            nk_size_t const column_tile = column_tiles_count - 1;
            nk_dots_bf16_zero_2x1_sapphireamx_();
            for (nk_size_t depth_tile = 0; depth_tile < depth_tiles_count; depth_tile++) {
                nk_size_t const depth_offset = depth_tile * 32;
                nk_size_t const valid_depth = (depth_tile < full_depth_tiles_count) ? 32 : depth_remainder;
                nk_dots_load_a_mxfp8e5m2_sapphireamx_(&a_tile_top, a, a_stride, a_scales, a_scales_stride, row_bases,
                                                      row_block_start, depth_offset, rows_in_high_tile, valid_depth);
                if (rows_in_low_tile > 0)
                    nk_dots_load_a_mxfp8e5m2_sapphireamx_(&a_tile_bottom, a, a_stride, a_scales, a_scales_stride,
                                                          row_bases + 16, row_block_start + 16, depth_offset,
                                                          rows_in_low_tile, valid_depth);
                nk_dots_bf16_accumulate_2x1_sapphireamx_(
                    &a_tile_top, &a_tile_bottom,
                    nk_dots_bf16_b_tile_sapphireamx_(b_tiles, column_tile * depth_tiles_count + depth_tile));
            }
            nk_dots_bf16_store_column_sapphireamx_(c + row_block_start * c_stride_elements + column_tile * 16,
                                                   c_stride_elements, rows_in_high_tile, rows_in_low_tile, 16);
        }

        if (column_remainder_count > 0) {
            nk_dots_bf16_zero_2x1_sapphireamx_();
            for (nk_size_t depth_tile = 0; depth_tile < depth_tiles_count; depth_tile++) {
                nk_size_t const depth_offset = depth_tile * 32;
                nk_size_t const valid_depth = (depth_tile < full_depth_tiles_count) ? 32 : depth_remainder;
                nk_dots_load_a_mxfp8e5m2_sapphireamx_(&a_tile_top, a, a_stride, a_scales, a_scales_stride, row_bases,
                                                      row_block_start, depth_offset, rows_in_high_tile, valid_depth);
                if (rows_in_low_tile > 0)
                    nk_dots_load_a_mxfp8e5m2_sapphireamx_(&a_tile_bottom, a, a_stride, a_scales, a_scales_stride,
                                                          row_bases + 16, row_block_start + 16, depth_offset,
                                                          rows_in_low_tile, valid_depth);
                nk_dots_bf16_accumulate_edge_sapphireamx_(&a_tile_top, &a_tile_bottom, column_edge, depth_offset, depth,
                                                          column_remainder_count, valid_depth);
            }
            nk_dots_bf16_store_column_sapphireamx_(c + row_block_start * c_stride_elements + full_columns,
                                                   c_stride_elements, rows_in_high_tile, rows_in_low_tile,
                                                   column_remainder_count);
        }
    }

    _tile_release();
    return nk_success_k;
}

/** Gram upper triangle of @p vectors widened to BF16, on AMX tiles, covering rows from
 *  @p rows_begin up to @p rows_end: vectors rebased, before any tensor scale. */
NUMKONG_INLINE nk_status_t nk_gram_mxfp8e5m2_sapphireamx_(                                       //
    nk_e5m2_t const *vectors, nk_ue8m0_t const *scales, nk_size_t vector_count, nk_size_t depth, //
    nk_size_t stride, nk_size_t scales_stride, nk_f32_t *result, nk_size_t result_stride, nk_size_t rows_begin,
    nk_size_t rows_end) {
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    rows_end = nk_min_of_two(rows_end, vector_count);

    // Depth is walked in groups of 3 tiles × 32 elements
    nk_size_t const depth_tile_groups = nk_size_divide_round_up_(nk_size_divide_round_up_(depth, 32), 3);

    nk_dots_bf16_a16x32_sapphireamx_t a_tiles[3];
    nk_dots_bf16_a16x32_sapphireamx_t b_src_tiles[3];
    nk_dots_bf16_b32x16_sapphireamx_t b_tiles[3];
    nk_dots_bf16_state_sapphireamx_t state;
    nk_i32_t row_bases[16], column_bases[16];

    nk_amx_tile_configure_sapphireamx_();

    for (nk_size_t row_tile = rows_begin; row_tile < rows_end; row_tile += 16) {
        nk_size_t const valid_rows = (row_tile + 16 <= rows_end) ? 16 : (rows_end - row_tile);
        nk_dots_bases_mxfp8e5m2_sapphireamx_(scales, scales_stride, row_tile, valid_rows, 16, depth, row_bases);

        for (nk_size_t column_tile = row_tile; column_tile < vector_count; column_tile += 16) {
            nk_size_t const valid_columns = (column_tile + 16 <= vector_count) ? 16 : (vector_count - column_tile);
            nk_dots_bases_mxfp8e5m2_sapphireamx_(scales, scales_stride, column_tile, valid_columns, 16, depth,
                                                 column_bases);

            nk_dots_bf16_init_sapphireamx_(&state);

            for (nk_size_t depth_group = 0; depth_group < depth_tile_groups; depth_group++) {
                for (int tile_index = 0; tile_index < 3; tile_index++) {
                    nk_size_t const depth_start = depth_group * 96 + tile_index * 32;
                    nk_size_t const valid_depth = (depth_start + 32 <= depth)
                                                      ? 32
                                                      : (depth > depth_start ? depth - depth_start : 0);

                    nk_dots_load_a_mxfp8e5m2_sapphireamx_(&a_tiles[tile_index], vectors, stride, scales, scales_stride,
                                                          row_bases, row_tile, depth_start, valid_rows, valid_depth);
                    if (row_tile == column_tile && valid_rows == valid_columns) {
                        nk_dots_pack_bf16_transposed_sapphireamx_(&a_tiles[tile_index], &b_tiles[tile_index]);
                    }
                    else {
                        nk_dots_load_a_mxfp8e5m2_sapphireamx_(&b_src_tiles[tile_index], vectors, stride, scales,
                                                              scales_stride, column_bases, column_tile, depth_start,
                                                              valid_columns, valid_depth);
                        nk_dots_pack_bf16_transposed_sapphireamx_(&b_src_tiles[tile_index], &b_tiles[tile_index]);
                    }
                }

                nk_dots_bf16_update_sapphireamx_( //
                    &state, &a_tiles[0], &a_tiles[1], &a_tiles[2], &b_tiles[0], &b_tiles[1], &b_tiles[2]);
            }

            nk_dots_symmetric_store_sapphireamx_(                                     //
                state.data, result + row_tile * result_stride_elements + column_tile, //
                result_stride_elements, valid_rows, valid_columns, column_tile - row_tile);
        }
    }
    return nk_success_k;
}

#pragma endregion MXFP8E5M2 Floats

#pragma region E2M3 Floats

/** Load E2M3 A tile with E2M3 to signed I8 conversion via VPERMB LUT. Each E2M3 byte keeps the
 *  sign in bit 5 and a 5-bit magnitude index in bits 4:0. The LUT maps the magnitude to 16 times
 *  its value, then the sign is applied via conditional negation. The result is stored in an INT8
 *  tile for use with @c _tile_dpbssd. */
NUMKONG_INLINE void nk_dots_e2m3_load_a_sapphireamx_( //
    nk_dots_i8_a16x64_sapphireamx_t *a_tile,          //
    nk_e2m3_t const *src, nk_size_t src_stride,       //
    nk_size_t valid_rows, nk_size_t valid_columns) {

    // Repeat the 32 magnitudes in both halves for VPERMB.
    nk_align_(64) static nk_u8_t const lut_bytes[64] = {
        0,  2,  4,  6,  8,  10,  12,  14,  //
        16, 18, 20, 22, 24, 26,  28,  30,  //
        32, 36, 40, 44, 48, 52,  56,  60,  //
        64, 72, 80, 88, 96, 104, 112, 120, //
        0,  2,  4,  6,  8,  10,  12,  14,  //
        16, 18, 20, 22, 24, 26,  28,  30,  //
        32, 36, 40, 44, 48, 52,  56,  60,  //
        64, 72, 80, 88, 96, 104, 112, 120, //
    };
    __m512i magnitude_lut_u8x64 = _mm512_load_si512((__m512i const *)lut_bytes);
    __m512i sign_mask_u8x64 = _mm512_set1_epi8(0x20);
    __m512i magnitude_mask_u8x64 = _mm512_set1_epi8(0x1F);
    __m512i zero_i8x64 = _mm512_setzero_si512();

    __mmask64 column_m64 = (valid_columns >= 64) ? 0xFFFFFFFFFFFFFFFFULL : ((__mmask64)1 << valid_columns) - 1;

    for (nk_size_t row = 0; row < 16; row++) {
        if (row < valid_rows) {
            __m512i raw_u8x64 = _mm512_maskz_loadu_epi8(column_m64, src + row * src_stride);
            __m512i magnitude_u8x64 = _mm512_and_si512(raw_u8x64, magnitude_mask_u8x64);
            __m512i unsigned_value_u8x64 = _mm512_permutexvar_epi8(magnitude_u8x64, magnitude_lut_u8x64);
            __mmask64 negate_m64 = _mm512_test_epi8_mask(raw_u8x64, sign_mask_u8x64);
            __m512i signed_value_i8x64 = _mm512_mask_sub_epi8(unsigned_value_u8x64, negate_m64, zero_i8x64,
                                                              unsigned_value_u8x64);
            _mm512_store_si512(a_tile->data[row], signed_value_i8x64);
        }
        else { _mm512_store_si512(a_tile->data[row], zero_i8x64); }
    }
    nk_compiler_barrier_sapphireamx_();
}

/* Store E2M3 accumulator: read I32 state, convert to F32, multiply by 1/256, store as F32. */
NUMKONG_INLINE void nk_dots_e2m3_store_sapphireamx_( //
    nk_dots_i8_state_sapphireamx_t const *state,     //
    nk_f32_t *dst, nk_size_t dst_stride_elements,    //
    nk_size_t valid_rows, nk_size_t valid_columns) {

    __mmask16 column_m16 = (valid_columns >= 16) ? 0xFFFF : ((__mmask16)1 << valid_columns) - 1;
    __m512 scale_f32x16 = _mm512_set1_ps(1.0f / 256.0f);

    for (nk_size_t row = 0; row < valid_rows; row++) {
        __m512i i32_row_i32x16 = _mm512_load_si512(state->data[row]);
        __m512 f32_row_f32x16 = _mm512_mul_ps(_mm512_cvtepi32_ps(i32_row_i32x16), scale_f32x16);
        _mm512_mask_storeu_ps(dst + row * dst_stride_elements, column_m16, f32_row_f32x16);
    }
}

/* Store E2M3 2x2 accumulator state to F32 output matrix with masking for edge tiles. */
NUMKONG_INLINE void nk_dots_e2m3_output2x2_sapphireamx_( //
    nk_dots_i8_state2x2_sapphireamx_t const *state,      //
    nk_f32_t *dst, nk_size_t dst_stride_elements,        //
    nk_size_t valid_rows, nk_size_t valid_columns) {

    nk_size_t const rows_high = (valid_rows > 16) ? 16 : valid_rows;
    nk_size_t const columns_left = (valid_columns > 16) ? 16 : valid_columns;
    nk_size_t const columns_right = (valid_columns > 16) ? valid_columns - 16 : 0;

    if (rows_high > 0 && columns_left > 0)
        nk_dots_e2m3_store_sapphireamx_(&state->c[0][0], dst, dst_stride_elements, rows_high, columns_left);
    if (rows_high > 0 && columns_right > 0)
        nk_dots_e2m3_store_sapphireamx_(&state->c[0][1], dst + 16, dst_stride_elements, rows_high, columns_right);

    if (valid_rows > 16) {
        nk_size_t const rows_low = valid_rows - 16;
        nk_f32_t *dst_low = dst + 16 * dst_stride_elements;
        if (columns_left > 0)
            nk_dots_e2m3_store_sapphireamx_(&state->c[1][0], dst_low, dst_stride_elements, rows_low, columns_left);
        if (columns_right > 0)
            nk_dots_e2m3_store_sapphireamx_(&state->c[1][1], dst_low + 16, dst_stride_elements, rows_low,
                                            columns_right);
    }
}

/** E2M3 GEMM of @p a rows against pre-packed B columns into F32 @p c,through I8 AMX tiles. */
NUMKONG_INLINE nk_status_t nk_gemm_packed_e2m3_sapphireamx_( //
    nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,   //
    nk_size_t rows, nk_size_t column_count, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride) {
    nk_unused_(column_count);

    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    if (header->capability != nk_cap_sapphireamx_k) return nk_pack_mismatch_k;
    nk_size_t const column_tiles_count = header->full_column_tiles;
    nk_size_t const depth_tiles_count = header->full_depth_tiles;
    nk_size_t const column_remainder_count = header->column_remainder_count;

    // B tiles are already in I8 format
    nk_i8_t const *b_tiles_base = (nk_i8_t const *)((char const *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_i8_t const *col_edge_ptr = (nk_i8_t const *)((char const *)b_packed + header->column_edge_offset);

    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_size_t const tile_depth = 64;
    nk_size_t const tile_size = 1024;
    nk_size_t const full_columns = column_tiles_count * 16;

    nk_size_t const row_blocks_count = nk_size_divide_round_up_(rows, 32);
    nk_size_t const col_blocks_count = column_tiles_count / 2;

    if (depth_tiles_count == 0) return nk_success_k;

    nk_dots_i8_a16x64_sapphireamx_t a_tile_top, a_tile_bottom;
    nk_dots_i8_state2x2_sapphireamx_t c_accum_buffer;

    nk_size_t const full_depth_tiles_count = depth / tile_depth;
    nk_size_t const depth_remainder = depth % tile_depth;

    nk_amx_tile_configure_sapphireamx_();

    // Loop order: row_blocks outer, col_blocks inner
    for (nk_size_t row_block_idx = 0; row_block_idx < row_blocks_count; row_block_idx++) {
        nk_size_t const row_block_start = row_block_idx * 32;
        nk_size_t const valid_rows_count = (row_block_start + 32 <= rows) ? 32 : (rows - row_block_start);
        nk_size_t const is_full_row_block = (valid_rows_count == 32);
        nk_size_t const rows_in_high_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
        nk_size_t const rows_in_low_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

        for (nk_size_t column_block_idx = 0; column_block_idx < col_blocks_count; column_block_idx++) {
            nk_size_t const col_block_start = column_block_idx * 32;
            nk_size_t const b_column_left_base = (column_block_idx * 2) * depth_tiles_count;
            nk_size_t const b_column_right_base = (column_block_idx * 2 + 1) * depth_tiles_count;

            // Zero accumulators (TMM4-7 stay resident across entire depth loop)
            _tile_zero(4);
            _tile_zero(5);
            _tile_zero(6);
            _tile_zero(7);

            // E2M3 always uses buffered load for E2M3 → I8 conversion
            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                // Load A with E2M3 → I8 conversion
                nk_dots_e2m3_load_a_sapphireamx_(&a_tile_top, a + row_block_start * a_stride + depth_offset, a_stride,
                                                 rows_in_high_tile, valid_depth);
                if (rows_in_low_tile > 0) {
                    nk_dots_e2m3_load_a_sapphireamx_(&a_tile_bottom,
                                                     a + (row_block_start + 16) * a_stride + depth_offset, a_stride,
                                                     rows_in_low_tile, valid_depth);
                }

                nk_dots_i8_b64x16_sapphireamx_t const *b_tile_left =
                    (nk_dots_i8_b64x16_sapphireamx_t const *)(b_tiles_base +
                                                              (b_column_left_base + depth_tile_idx) * tile_size);
                nk_dots_i8_b64x16_sapphireamx_t const *b_tile_right =
                    (nk_dots_i8_b64x16_sapphireamx_t const *)(b_tiles_base +
                                                              (b_column_right_base + depth_tile_idx) * tile_size);

                _tile_loadd(0, a_tile_top.data, 64);
                _tile_loadd(1, a_tile_bottom.data, 64);
                _tile_loadd(2, b_tile_left->data, 64);
                _tile_loadd(3, b_tile_right->data, 64);

                _tile_dpbssd(4, 0, 2);
                _tile_dpbssd(5, 0, 3);
                _tile_dpbssd(6, 1, 2);
                _tile_dpbssd(7, 1, 3);
            }

            // AMX stores I32 tiles; buffer them before converting to F32.
            if (is_full_row_block) {
                nk_f32_t *c_block = c + row_block_start * c_stride_elements + col_block_start;
                nk_dots_i8_state2x2_sapphireamx_t c_accum_buffer;
                _tile_stored(4, c_accum_buffer.c[0][0].data, 64);
                _tile_stored(5, c_accum_buffer.c[0][1].data, 64);
                _tile_stored(6, c_accum_buffer.c[1][0].data, 64);
                _tile_stored(7, c_accum_buffer.c[1][1].data, 64);
                nk_dots_e2m3_output2x2_sapphireamx_(&c_accum_buffer, c_block, c_stride_elements, valid_rows_count, 32);
            }
            else {
                _tile_stored(4, c_accum_buffer.c[0][0].data, 64);
                _tile_stored(5, c_accum_buffer.c[0][1].data, 64);
                _tile_stored(6, c_accum_buffer.c[1][0].data, 64);
                _tile_stored(7, c_accum_buffer.c[1][1].data, 64);
                nk_dots_e2m3_output2x2_sapphireamx_(&c_accum_buffer,
                                                    c + row_block_start * c_stride_elements + col_block_start,
                                                    c_stride_elements, valid_rows_count, 32);
            }
        }

        // Handle odd column-tile (single 16-column tile if column_tiles_count is odd)
        if (column_tiles_count % 2 == 1) {
            nk_size_t const column_tile_idx = column_tiles_count - 1;
            nk_size_t const col_start = column_tile_idx * 16;
            nk_size_t const b_column_base = column_tile_idx * depth_tiles_count;

            nk_dots_i8_state_sapphireamx_t c_high_state, c_low_state;
            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_e2m3_load_a_sapphireamx_(&a_tile_top, a + row_block_start * a_stride + depth_offset, a_stride,
                                                 rows_in_high_tile, valid_depth);
                if (rows_in_low_tile > 0) {
                    nk_dots_e2m3_load_a_sapphireamx_(&a_tile_bottom,
                                                     a + (row_block_start + 16) * a_stride + depth_offset, a_stride,
                                                     rows_in_low_tile, valid_depth);
                }

                nk_dots_i8_b64x16_sapphireamx_t const *b_tile =
                    (nk_dots_i8_b64x16_sapphireamx_t const *)(b_tiles_base +
                                                              (b_column_base + depth_tile_idx) * tile_size);

                _tile_loadd(0, a_tile_top.data, 64);
                _tile_loadd(1, a_tile_bottom.data, 64);
                _tile_loadd(2, b_tile->data, 64);

                _tile_dpbssd(4, 0, 2);
                _tile_dpbssd(6, 1, 2);
            }

            _tile_stored(4, c_high_state.data, 64);
            _tile_stored(6, c_low_state.data, 64);

            nk_dots_e2m3_store_sapphireamx_(&c_high_state, c + row_block_start * c_stride_elements + col_start,
                                            c_stride_elements, rows_in_high_tile, 16);
            if (rows_in_low_tile > 0) {
                nk_dots_e2m3_store_sapphireamx_(&c_low_state,
                                                c + (row_block_start + 16) * c_stride_elements + col_start,
                                                c_stride_elements, rows_in_low_tile, 16);
            }
        }

        // Handle column-edge (remaining columns < 16) using AMX with partial tiles
        if (column_remainder_count > 0) {
            nk_dots_i8_state_sapphireamx_t c_high_state, c_low_state;
            nk_dots_i8_a16x64_sapphireamx_t b_as_a;
            nk_dots_i8_b64x16_sapphireamx_t b_tile;

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_e2m3_load_a_sapphireamx_(&a_tile_top, a + row_block_start * a_stride + depth_offset, a_stride,
                                                 rows_in_high_tile, valid_depth);
                if (rows_in_low_tile > 0) {
                    nk_dots_e2m3_load_a_sapphireamx_(&a_tile_bottom,
                                                     a + (row_block_start + 16) * a_stride + depth_offset, a_stride,
                                                     rows_in_low_tile, valid_depth);
                }

                // B edge data is already in I8 format
                nk_dots_i8_load_a_sapphireamx_(&b_as_a, col_edge_ptr + depth_offset, depth, column_remainder_count,
                                               valid_depth);
                nk_dots_pack_i8_transposed_sapphireamx_(&b_as_a, &b_tile);

                _tile_loadd(0, a_tile_top.data, 64);
                _tile_loadd(1, a_tile_bottom.data, 64);
                _tile_loadd(2, b_tile.data, 64);

                _tile_dpbssd(4, 0, 2);
                _tile_dpbssd(6, 1, 2);
            }

            _tile_stored(4, c_high_state.data, 64);
            _tile_stored(6, c_low_state.data, 64);

            nk_dots_e2m3_store_sapphireamx_(&c_high_state, c + row_block_start * c_stride_elements + full_columns,
                                            c_stride_elements, rows_in_high_tile, column_remainder_count);
            if (rows_in_low_tile > 0) {
                nk_dots_e2m3_store_sapphireamx_(&c_low_state,
                                                c + (row_block_start + 16) * c_stride_elements + full_columns,
                                                c_stride_elements, rows_in_low_tile, column_remainder_count);
            }
        }
    }

    _tile_release();
    return nk_success_k;
}

/** E2M3 Gram upper triangle of @p vectors on AMX tiles, rows @p rows_begin to @p rows_end. */
NUMKONG_INLINE nk_status_t nk_gram_e2m3_sapphireamx_(                  //
    nk_e2m3_t const *vectors, nk_size_t vector_count, nk_size_t depth, //
    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,       //
    nk_size_t rows_begin, nk_size_t rows_end) {
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));

    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);

    // Handle row slicing: compute rows [rows_begin, rows_end)
    rows_end = nk_min_of_two(rows_end, vector_count);

    // Round depth up to multiple of 192 (3 tiles x 64 elements)
    nk_size_t const depth_tiles = nk_size_divide_round_up_(depth, 64);
    nk_size_t const depth_tile_groups = nk_size_divide_round_up_(depth_tiles, 3);

    nk_dots_i8_a16x64_sapphireamx_t a_tiles[3];
    nk_dots_i8_a16x64_sapphireamx_t b_src_tiles[3];
    nk_dots_i8_b64x16_sapphireamx_t b_tiles[3];
    nk_dots_i8_state_sapphireamx_t state;

    nk_amx_tile_configure_sapphireamx_();

    for (nk_size_t row_tile = rows_begin; row_tile < rows_end; row_tile += 16) {
        nk_size_t const valid_rows = (row_tile + 16 <= rows_end) ? 16 : (rows_end - row_tile);

        for (nk_size_t col_tile = row_tile; col_tile < vector_count; col_tile += 16) {
            nk_size_t const valid_columns = (col_tile + 16 <= vector_count) ? 16 : (vector_count - col_tile);

            nk_dots_i8_init_sapphireamx_(&state);

            for (nk_size_t depth_group_idx = 0; depth_group_idx < depth_tile_groups; depth_group_idx++) {
                nk_size_t const depth_base = depth_group_idx * 192;

                for (int tile_idx = 0; tile_idx < 3; tile_idx++) {
                    nk_size_t const depth_start = depth_base + tile_idx * 64;
                    nk_size_t const valid_depth = (depth_start + 64 <= depth)
                                                      ? 64
                                                      : (depth > depth_start ? depth - depth_start : 0);

                    nk_dots_e2m3_load_a_sapphireamx_(              //
                        &a_tiles[tile_idx],                        //
                        vectors + row_tile * stride + depth_start, //
                        stride, valid_rows, valid_depth);

                    if (row_tile == col_tile && valid_rows == valid_columns) {
                        nk_dots_pack_i8_transposed_sapphireamx_(&a_tiles[tile_idx], &b_tiles[tile_idx]);
                    }
                    else {
                        nk_dots_e2m3_load_a_sapphireamx_(              //
                            &b_src_tiles[tile_idx],                    //
                            vectors + col_tile * stride + depth_start, //
                            stride, valid_columns, valid_depth);
                        nk_dots_pack_i8_transposed_sapphireamx_(&b_src_tiles[tile_idx], &b_tiles[tile_idx]);
                    }
                }

                nk_dots_i8_update_sapphireamx_( //
                    &state, &a_tiles[0], &a_tiles[1], &a_tiles[2], &b_tiles[0], &b_tiles[1], &b_tiles[2]);
            }

            nk_align_(64) nk_f32_t scaled[16][16];
            nk_dots_e2m3_store_sapphireamx_(&state, &scaled[0][0], 16, valid_rows, valid_columns);
            nk_dots_symmetric_store_sapphireamx_(                              //
                scaled, result + row_tile * result_stride_elements + col_tile, //
                result_stride_elements, valid_rows, valid_columns, col_tile - row_tile);
        }
    }
    return nk_success_k;
}

#pragma endregion E2M3 Floats

#pragma region E2M1 Floats

/** Decode 64 E2M1 nibbles from 32 bytes into doubled signed I8 values, the high nibble of each
 *  byte first. */
NUMKONG_INLINE __m512i nk_e2m1x64_to_i8x64_sapphireamx_(__m256i packed_u8x32) {
    __m512i const lut_i8x64 = _mm512_broadcast_i32x4(
        _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12));
    __m512i bytes_u16x32 = _mm512_cvtepu8_epi16(packed_u8x32);
    __m512i high_u16x32 = _mm512_srli_epi16(bytes_u16x32, 4);
    __m512i low_u16x32 = _mm512_slli_epi16(_mm512_and_si512(bytes_u16x32, _mm512_set1_epi16(0x0F)), 8);
    __m512i nibbles_u8x64 = _mm512_or_si512(high_u16x32, low_u16x32);
    return _mm512_permutexvar_epi8(nibbles_u8x64, lut_i8x64);
}

/** Load an E2M1 A tile of up to 64 dimensions per row, decoding nibbles to doubled I8 and zeroing
 *  past @p valid_columns. */
NUMKONG_INLINE void nk_dots_e2m1_load_a_sapphireamx_( //
    nk_dots_i8_a16x64_sapphireamx_t *a_tile,          //
    nk_e2m1x2_t const *src, nk_size_t src_stride,     //
    nk_size_t valid_rows, nk_size_t valid_columns) {

    nk_size_t const valid_bytes = valid_columns / NUMKONG_NIBBLES_PER_BYTE;
    __mmask32 byte_m32 = (valid_bytes >= 32) ? 0xFFFFFFFFu : (((__mmask32)1 << valid_bytes) - 1);
    __mmask64 column_m64 = (valid_columns >= 64) ? 0xFFFFFFFFFFFFFFFFULL : ((__mmask64)1 << valid_columns) - 1;
    __m512i zero_i8x64 = _mm512_setzero_si512();

    for (nk_size_t row = 0; row < 16; row++) {
        if (row < valid_rows) {
            __m256i packed_u8x32 = _mm256_maskz_loadu_epi8(byte_m32, src + row * src_stride);
            __m512i values_i8x64 = _mm512_maskz_mov_epi8(column_m64, nk_e2m1x64_to_i8x64_sapphireamx_(packed_u8x32));
            _mm512_store_si512(a_tile->data[row], values_i8x64);
        }
        else { _mm512_store_si512(a_tile->data[row], zero_i8x64); }
    }
    nk_compiler_barrier_sapphireamx_();
}

/* Store E2M1 accumulator: read I32 state, convert to F32, multiply by 1/4, store as F32. */
NUMKONG_INLINE void nk_dots_e2m1_store_sapphireamx_( //
    nk_dots_i8_state_sapphireamx_t const *state,     //
    nk_f32_t *dst, nk_size_t dst_stride_elements,    //
    nk_size_t valid_rows, nk_size_t valid_columns) {

    __mmask16 column_m16 = (valid_columns >= 16) ? 0xFFFF : ((__mmask16)1 << valid_columns) - 1;
    __m512 scale_f32x16 = _mm512_set1_ps(0.25f);

    for (nk_size_t row = 0; row < valid_rows; row++) {
        __m512i i32_row_i32x16 = _mm512_load_si512(state->data[row]);
        __m512 f32_row_f32x16 = _mm512_mul_ps(_mm512_cvtepi32_ps(i32_row_i32x16), scale_f32x16);
        _mm512_mask_storeu_ps(dst + row * dst_stride_elements, column_m16, f32_row_f32x16);
    }
}

/* Store E2M1 2x2 accumulator state to F32 output matrix with masking for edge tiles. */
NUMKONG_INLINE void nk_dots_e2m1_output2x2_sapphireamx_( //
    nk_dots_i8_state2x2_sapphireamx_t const *state,      //
    nk_f32_t *dst, nk_size_t dst_stride_elements,        //
    nk_size_t valid_rows, nk_size_t valid_columns) {

    nk_size_t const rows_high = (valid_rows > 16) ? 16 : valid_rows;
    nk_size_t const columns_left = (valid_columns > 16) ? 16 : valid_columns;
    nk_size_t const columns_right = (valid_columns > 16) ? valid_columns - 16 : 0;

    if (rows_high > 0 && columns_left > 0)
        nk_dots_e2m1_store_sapphireamx_(&state->c[0][0], dst, dst_stride_elements, rows_high, columns_left);
    if (rows_high > 0 && columns_right > 0)
        nk_dots_e2m1_store_sapphireamx_(&state->c[0][1], dst + 16, dst_stride_elements, rows_high, columns_right);

    if (valid_rows > 16) {
        nk_size_t const rows_low = valid_rows - 16;
        nk_f32_t *dst_low = dst + 16 * dst_stride_elements;
        if (columns_left > 0)
            nk_dots_e2m1_store_sapphireamx_(&state->c[1][0], dst_low, dst_stride_elements, rows_low, columns_left);
        if (columns_right > 0)
            nk_dots_e2m1_store_sapphireamx_(&state->c[1][1], dst_low + 16, dst_stride_elements, rows_low,
                                            columns_right);
    }
}

/** E2M1 GEMM of @p a rows against pre-packed B columns into F32 @p c,through I8 AMX tiles. */
NUMKONG_INLINE nk_status_t nk_gemm_packed_e2m1_sapphireamx_( //
    nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t column_count, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride) {
    nk_unused_(column_count);

    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    if (header->capability != nk_cap_sapphireamx_k) return nk_pack_mismatch_k;
    nk_size_t const column_tiles_count = header->full_column_tiles;
    nk_size_t const depth_tiles_count = header->full_depth_tiles;
    nk_size_t const column_remainder_count = header->column_remainder_count;

    // B tiles are already in I8 format
    nk_i8_t const *b_tiles_base = (nk_i8_t const *)((char const *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_i8_t const *col_edge_ptr = (nk_i8_t const *)((char const *)b_packed + header->column_edge_offset);

    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_size_t const tile_depth = 64;
    nk_size_t const tile_size = 1024;
    nk_size_t const full_columns = column_tiles_count * 16;

    nk_size_t const row_blocks_count = nk_size_divide_round_up_(rows, 32);
    nk_size_t const col_blocks_count = column_tiles_count / 2;

    if (depth_tiles_count == 0) return nk_success_k;

    nk_dots_i8_a16x64_sapphireamx_t a_tile_top, a_tile_bottom;
    nk_dots_i8_state2x2_sapphireamx_t c_accum_buffer;

    nk_size_t const full_depth_tiles_count = depth / tile_depth;
    nk_size_t const depth_remainder = depth % tile_depth;

    nk_amx_tile_configure_sapphireamx_();

    // Loop order: row_blocks outer, col_blocks inner
    for (nk_size_t row_block_idx = 0; row_block_idx < row_blocks_count; row_block_idx++) {
        nk_size_t const row_block_start = row_block_idx * 32;
        nk_size_t const valid_rows_count = (row_block_start + 32 <= rows) ? 32 : (rows - row_block_start);
        nk_size_t const is_full_row_block = (valid_rows_count == 32);
        nk_size_t const rows_in_high_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
        nk_size_t const rows_in_low_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

        for (nk_size_t column_block_idx = 0; column_block_idx < col_blocks_count; column_block_idx++) {
            nk_size_t const col_block_start = column_block_idx * 32;
            nk_size_t const b_column_left_base = (column_block_idx * 2) * depth_tiles_count;
            nk_size_t const b_column_right_base = (column_block_idx * 2 + 1) * depth_tiles_count;

            // Zero accumulators (TMM4-7 stay resident across entire depth loop)
            _tile_zero(4);
            _tile_zero(5);
            _tile_zero(6);
            _tile_zero(7);

            // E2M1 always uses buffered load for E2M1 → I8 conversion
            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                // Load A with E2M1 → I8 conversion
                nk_dots_e2m1_load_a_sapphireamx_(&a_tile_top, a + row_block_start * a_stride + depth_offset / 2,
                                                 a_stride, rows_in_high_tile, valid_depth);
                if (rows_in_low_tile > 0) {
                    nk_dots_e2m1_load_a_sapphireamx_(&a_tile_bottom,
                                                     a + (row_block_start + 16) * a_stride + depth_offset / 2, a_stride,
                                                     rows_in_low_tile, valid_depth);
                }

                nk_dots_i8_b64x16_sapphireamx_t const *b_tile_left =
                    (nk_dots_i8_b64x16_sapphireamx_t const *)(b_tiles_base +
                                                              (b_column_left_base + depth_tile_idx) * tile_size);
                nk_dots_i8_b64x16_sapphireamx_t const *b_tile_right =
                    (nk_dots_i8_b64x16_sapphireamx_t const *)(b_tiles_base +
                                                              (b_column_right_base + depth_tile_idx) * tile_size);

                _tile_loadd(0, a_tile_top.data, 64);
                _tile_loadd(1, a_tile_bottom.data, 64);
                _tile_loadd(2, b_tile_left->data, 64);
                _tile_loadd(3, b_tile_right->data, 64);

                _tile_dpbssd(4, 0, 2);
                _tile_dpbssd(5, 0, 3);
                _tile_dpbssd(6, 1, 2);
                _tile_dpbssd(7, 1, 3);
            }

            // AMX stores I32 tiles; buffer them before converting to F32.
            if (is_full_row_block) {
                nk_f32_t *c_block = c + row_block_start * c_stride_elements + col_block_start;
                nk_dots_i8_state2x2_sapphireamx_t c_accum_buffer;
                _tile_stored(4, c_accum_buffer.c[0][0].data, 64);
                _tile_stored(5, c_accum_buffer.c[0][1].data, 64);
                _tile_stored(6, c_accum_buffer.c[1][0].data, 64);
                _tile_stored(7, c_accum_buffer.c[1][1].data, 64);
                nk_dots_e2m1_output2x2_sapphireamx_(&c_accum_buffer, c_block, c_stride_elements, valid_rows_count, 32);
            }
            else {
                _tile_stored(4, c_accum_buffer.c[0][0].data, 64);
                _tile_stored(5, c_accum_buffer.c[0][1].data, 64);
                _tile_stored(6, c_accum_buffer.c[1][0].data, 64);
                _tile_stored(7, c_accum_buffer.c[1][1].data, 64);
                nk_dots_e2m1_output2x2_sapphireamx_(&c_accum_buffer,
                                                    c + row_block_start * c_stride_elements + col_block_start,
                                                    c_stride_elements, valid_rows_count, 32);
            }
        }

        // Handle odd column-tile (single 16-column tile if column_tiles_count is odd)
        if (column_tiles_count % 2 == 1) {
            nk_size_t const column_tile_idx = column_tiles_count - 1;
            nk_size_t const col_start = column_tile_idx * 16;
            nk_size_t const b_column_base = column_tile_idx * depth_tiles_count;

            nk_dots_i8_state_sapphireamx_t c_high_state, c_low_state;
            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_e2m1_load_a_sapphireamx_(&a_tile_top, a + row_block_start * a_stride + depth_offset / 2,
                                                 a_stride, rows_in_high_tile, valid_depth);
                if (rows_in_low_tile > 0) {
                    nk_dots_e2m1_load_a_sapphireamx_(&a_tile_bottom,
                                                     a + (row_block_start + 16) * a_stride + depth_offset / 2, a_stride,
                                                     rows_in_low_tile, valid_depth);
                }

                nk_dots_i8_b64x16_sapphireamx_t const *b_tile =
                    (nk_dots_i8_b64x16_sapphireamx_t const *)(b_tiles_base +
                                                              (b_column_base + depth_tile_idx) * tile_size);

                _tile_loadd(0, a_tile_top.data, 64);
                _tile_loadd(1, a_tile_bottom.data, 64);
                _tile_loadd(2, b_tile->data, 64);

                _tile_dpbssd(4, 0, 2);
                _tile_dpbssd(6, 1, 2);
            }

            _tile_stored(4, c_high_state.data, 64);
            _tile_stored(6, c_low_state.data, 64);

            nk_dots_e2m1_store_sapphireamx_(&c_high_state, c + row_block_start * c_stride_elements + col_start,
                                            c_stride_elements, rows_in_high_tile, 16);
            if (rows_in_low_tile > 0) {
                nk_dots_e2m1_store_sapphireamx_(&c_low_state,
                                                c + (row_block_start + 16) * c_stride_elements + col_start,
                                                c_stride_elements, rows_in_low_tile, 16);
            }
        }

        // Handle column-edge (remaining columns < 16) using AMX with partial tiles
        if (column_remainder_count > 0) {
            nk_dots_i8_state_sapphireamx_t c_high_state, c_low_state;
            nk_dots_i8_a16x64_sapphireamx_t b_as_a;
            nk_dots_i8_b64x16_sapphireamx_t b_tile;

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_e2m1_load_a_sapphireamx_(&a_tile_top, a + row_block_start * a_stride + depth_offset / 2,
                                                 a_stride, rows_in_high_tile, valid_depth);
                if (rows_in_low_tile > 0) {
                    nk_dots_e2m1_load_a_sapphireamx_(&a_tile_bottom,
                                                     a + (row_block_start + 16) * a_stride + depth_offset / 2, a_stride,
                                                     rows_in_low_tile, valid_depth);
                }

                // B edge data is already in I8 format
                nk_dots_i8_load_a_sapphireamx_(&b_as_a, col_edge_ptr + depth_offset, depth, column_remainder_count,
                                               valid_depth);
                nk_dots_pack_i8_transposed_sapphireamx_(&b_as_a, &b_tile);

                _tile_loadd(0, a_tile_top.data, 64);
                _tile_loadd(1, a_tile_bottom.data, 64);
                _tile_loadd(2, b_tile.data, 64);

                _tile_dpbssd(4, 0, 2);
                _tile_dpbssd(6, 1, 2);
            }

            _tile_stored(4, c_high_state.data, 64);
            _tile_stored(6, c_low_state.data, 64);

            nk_dots_e2m1_store_sapphireamx_(&c_high_state, c + row_block_start * c_stride_elements + full_columns,
                                            c_stride_elements, rows_in_high_tile, column_remainder_count);
            if (rows_in_low_tile > 0) {
                nk_dots_e2m1_store_sapphireamx_(&c_low_state,
                                                c + (row_block_start + 16) * c_stride_elements + full_columns,
                                                c_stride_elements, rows_in_low_tile, column_remainder_count);
            }
        }
    }

    _tile_release();
    return nk_success_k;
}

/** E2M1 Gram upper triangle of @p vectors on AMX tiles, rows @p rows_begin to @p rows_end. */
NUMKONG_INLINE nk_status_t nk_gram_e2m1_sapphireamx_(                    //
    nk_e2m1x2_t const *vectors, nk_size_t vector_count, nk_size_t depth, //
    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,         //
    nk_size_t rows_begin, nk_size_t rows_end) {
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= nk_size_divide_round_up_(depth, 2) * sizeof(*vectors));

    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);

    rows_end = nk_min_of_two(rows_end, vector_count);

    nk_size_t const depth_tiles = nk_size_divide_round_up_(depth, 64);
    nk_size_t const depth_tile_groups = nk_size_divide_round_up_(depth_tiles, 3); // 3 tiles × 64 = 192 values

    nk_dots_i8_a16x64_sapphireamx_t a_tiles[3];
    nk_dots_i8_a16x64_sapphireamx_t b_src_tiles[3];
    nk_dots_i8_b64x16_sapphireamx_t b_tiles[3];
    nk_dots_i8_state_sapphireamx_t state;

    nk_amx_tile_configure_sapphireamx_();

    for (nk_size_t row_tile = rows_begin; row_tile < rows_end; row_tile += 16) {
        nk_size_t const valid_rows = (row_tile + 16 <= rows_end) ? 16 : (rows_end - row_tile);

        for (nk_size_t col_tile = row_tile; col_tile < vector_count; col_tile += 16) {
            nk_size_t const valid_columns = (col_tile + 16 <= vector_count) ? 16 : (vector_count - col_tile);

            nk_dots_i8_init_sapphireamx_(&state);

            for (nk_size_t depth_group_idx = 0; depth_group_idx < depth_tile_groups; depth_group_idx++) {
                nk_size_t const depth_base = depth_group_idx * 192;

                for (int tile_idx = 0; tile_idx < 3; tile_idx++) {
                    nk_size_t const depth_start = depth_base + tile_idx * 64;
                    nk_size_t const valid_depth = (depth_start + 64 <= depth)
                                                      ? 64
                                                      : (depth > depth_start ? depth - depth_start : 0);

                    nk_dots_e2m1_load_a_sapphireamx_(                  //
                        &a_tiles[tile_idx],                            //
                        vectors + row_tile * stride + depth_start / 2, //
                        stride, valid_rows, valid_depth);

                    if (row_tile == col_tile && valid_rows == valid_columns) {
                        nk_dots_pack_i8_transposed_sapphireamx_(&a_tiles[tile_idx], &b_tiles[tile_idx]);
                    }
                    else {
                        nk_dots_e2m1_load_a_sapphireamx_(                  //
                            &b_src_tiles[tile_idx],                        //
                            vectors + col_tile * stride + depth_start / 2, //
                            stride, valid_columns, valid_depth);
                        nk_dots_pack_i8_transposed_sapphireamx_(&b_src_tiles[tile_idx], &b_tiles[tile_idx]);
                    }
                }

                nk_dots_i8_update_sapphireamx_( //
                    &state, &a_tiles[0], &a_tiles[1], &a_tiles[2], &b_tiles[0], &b_tiles[1], &b_tiles[2]);
            }

            nk_align_(64) nk_f32_t scaled[16][16];
            nk_dots_e2m1_store_sapphireamx_(&state, &scaled[0][0], 16, valid_rows, valid_columns);
            nk_dots_symmetric_store_sapphireamx_(                              //
                scaled, result + row_tile * result_stride_elements + col_tile, //
                result_stride_elements, valid_rows, valid_columns, col_tile - row_tile);
        }
    }
    return nk_success_k;
}

#pragma endregion E2M1 Floats

#pragma region E3M2 Floats

/** Load E3M2 A tile with FP8 to BF16 conversion. */
NUMKONG_INLINE void nk_dots_e3m2_load_a_sapphireamx_( //
    nk_dots_bf16_a16x32_sapphireamx_t *a_tile,        //
    nk_e3m2_t const *src, nk_size_t src_stride,       //
    nk_size_t valid_rows, nk_size_t valid_columns) {

    __mmask32 column_m32 = (valid_columns >= 32) ? 0xFFFFFFFF : ((__mmask32)1 << valid_columns) - 1;
    __m512i zero_i16x32 = _mm512_setzero_si512();

    for (nk_size_t row_idx = 0; row_idx < 16; row_idx++) {
        if (row_idx < valid_rows) {
            __m256i e3m2_row_u8x32 = _mm256_maskz_loadu_epi8(column_m32, src + row_idx * src_stride);
            __m512i bf16_row_i16x32 = nk_e3m2x32_to_bf16x32_icelake_(e3m2_row_u8x32);
            _mm512_store_si512((__m512i *)a_tile->data[row_idx], bf16_row_i16x32);
        }
        else { _mm512_store_si512((__m512i *)a_tile->data[row_idx], zero_i16x32); }
    }
    nk_compiler_barrier_sapphireamx_();
}

/** E3M2 GEMM of @p a rows against pre-packed B columns into F32 @p c,through BF16 AMX tiles. */
NUMKONG_INLINE nk_status_t nk_gemm_packed_e3m2_sapphireamx_( //
    nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,   //
    nk_size_t rows, nk_size_t column_count, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride) {
    nk_unused_(column_count);

    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    if (header->capability != nk_cap_sapphireamx_k) return nk_pack_mismatch_k;
    nk_size_t const column_tiles_count = header->full_column_tiles;
    nk_size_t const depth_tiles_count = header->full_depth_tiles;
    nk_size_t const column_remainder_count = header->column_remainder_count;

    nk_bf16_t const *b_tiles_base = (nk_bf16_t const *)((char const *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_bf16_t const *col_edge_ptr = (nk_bf16_t const *)((char const *)b_packed + header->column_edge_offset);

    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_size_t const tile_depth = 32;
    nk_size_t const tile_size = 512;
    nk_size_t const full_columns = column_tiles_count * 16;

    nk_size_t const row_blocks_count = nk_size_divide_round_up_(rows, 32);
    nk_size_t const col_blocks_count = column_tiles_count / 2;

    if (depth_tiles_count == 0) return nk_success_k;

    nk_dots_bf16_a16x32_sapphireamx_t a_tile_top, a_tile_bottom;
    nk_dots_bf16_state2x2_sapphireamx_t c_accum_buffer;

    nk_size_t const full_depth_tiles_count = depth / tile_depth;
    nk_size_t const depth_remainder = depth % tile_depth;

    nk_amx_tile_configure_sapphireamx_();

    // Loop order: row_blocks outer, col_blocks inner
    for (nk_size_t row_block_idx = 0; row_block_idx < row_blocks_count; row_block_idx++) {
        nk_size_t const row_block_start = row_block_idx * 32;
        nk_size_t const valid_rows_count = (row_block_start + 32 <= rows) ? 32 : (rows - row_block_start);
        nk_size_t const is_full_row_block = (valid_rows_count == 32);
        nk_size_t const rows_in_high_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
        nk_size_t const rows_in_low_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

        for (nk_size_t column_block_idx = 0; column_block_idx < col_blocks_count; column_block_idx++) {
            nk_size_t const col_block_start = column_block_idx * 32;
            nk_size_t const b_column_left_base = (column_block_idx * 2) * depth_tiles_count;
            nk_size_t const b_column_right_base = (column_block_idx * 2 + 1) * depth_tiles_count;

            // Zero accumulators (TMM4-7 stay resident across entire depth loop)
            _tile_zero(4);
            _tile_zero(5);
            _tile_zero(6);
            _tile_zero(7);

            // FP8 always uses buffered load for E3M2 → BF16 conversion
            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                // Load A with FP8 → BF16 conversion
                nk_dots_e3m2_load_a_sapphireamx_(&a_tile_top, a + row_block_start * a_stride + depth_offset, a_stride,
                                                 rows_in_high_tile, valid_depth);
                if (rows_in_low_tile > 0) {
                    nk_dots_e3m2_load_a_sapphireamx_(&a_tile_bottom,
                                                     a + (row_block_start + 16) * a_stride + depth_offset, a_stride,
                                                     rows_in_low_tile, valid_depth);
                }

                nk_dots_bf16_b32x16_sapphireamx_t const *b_tile_left =
                    (nk_dots_bf16_b32x16_sapphireamx_t const *)(b_tiles_base +
                                                                (b_column_left_base + depth_tile_idx) * tile_size);
                nk_dots_bf16_b32x16_sapphireamx_t const *b_tile_right =
                    (nk_dots_bf16_b32x16_sapphireamx_t const *)(b_tiles_base +
                                                                (b_column_right_base + depth_tile_idx) * tile_size);

                _tile_loadd(0, a_tile_top.data, 64);
                _tile_loadd(1, a_tile_bottom.data, 64);
                _tile_loadd(2, b_tile_left->data, 64);
                _tile_loadd(3, b_tile_right->data, 64);

                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(5, 0, 3);
                _tile_dpbf16ps(6, 1, 2);
                _tile_dpbf16ps(7, 1, 3);
            }

            // Store accumulators to output (once per output block)
            if (is_full_row_block) {
                nk_f32_t *c_block = c + row_block_start * c_stride_elements + col_block_start;
                _tile_stored(4, c_block, c_stride);
                _tile_stored(5, c_block + 16, c_stride);
                _tile_stored(6, (nk_f32_t *)((char *)c_block + 16 * c_stride), c_stride);
                _tile_stored(7, (nk_f32_t *)((char *)c_block + 16 * c_stride) + 16, c_stride);
            }
            else {
                _tile_stored(4, c_accum_buffer.c[0][0].data, 64);
                _tile_stored(5, c_accum_buffer.c[0][1].data, 64);
                _tile_stored(6, c_accum_buffer.c[1][0].data, 64);
                _tile_stored(7, c_accum_buffer.c[1][1].data, 64);
                nk_dots_bf16_output2x2_sapphireamx_(&c_accum_buffer,
                                                    c + row_block_start * c_stride_elements + col_block_start,
                                                    c_stride_elements, valid_rows_count, 32);
            }
        }

        // Handle odd column-tile (single 16-column tile if column_tiles_count is odd)
        if (column_tiles_count % 2 == 1) {
            nk_size_t const column_tile_idx = column_tiles_count - 1;
            nk_size_t const col_start = column_tile_idx * 16;
            nk_size_t const b_column_base = column_tile_idx * depth_tiles_count;

            nk_dots_bf16_state_sapphireamx_t c_high_state, c_low_state;
            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_e3m2_load_a_sapphireamx_(&a_tile_top, a + row_block_start * a_stride + depth_offset, a_stride,
                                                 rows_in_high_tile, valid_depth);
                if (rows_in_low_tile > 0) {
                    nk_dots_e3m2_load_a_sapphireamx_(&a_tile_bottom,
                                                     a + (row_block_start + 16) * a_stride + depth_offset, a_stride,
                                                     rows_in_low_tile, valid_depth);
                }

                nk_dots_bf16_b32x16_sapphireamx_t const *b_tile =
                    (nk_dots_bf16_b32x16_sapphireamx_t const *)(b_tiles_base +
                                                                (b_column_base + depth_tile_idx) * tile_size);

                _tile_loadd(0, a_tile_top.data, 64);
                _tile_loadd(1, a_tile_bottom.data, 64);
                _tile_loadd(2, b_tile->data, 64);

                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(6, 1, 2);
            }

            _tile_stored(4, c_high_state.data, 64);
            _tile_stored(6, c_low_state.data, 64);

            nk_dots_bf16_store_sapphireamx_(&c_high_state, c + row_block_start * c_stride_elements + col_start,
                                            c_stride_elements, rows_in_high_tile, 16);
            if (rows_in_low_tile > 0) {
                nk_dots_bf16_store_sapphireamx_(&c_low_state,
                                                c + (row_block_start + 16) * c_stride_elements + col_start,
                                                c_stride_elements, rows_in_low_tile, 16);
            }
        }

        // Handle column-edge (remaining columns < 16) using AMX with partial tiles
        if (column_remainder_count > 0) {
            nk_dots_bf16_state_sapphireamx_t c_high_state, c_low_state;
            nk_dots_bf16_a16x32_sapphireamx_t b_as_a;
            nk_dots_bf16_b32x16_sapphireamx_t b_tile;

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_e3m2_load_a_sapphireamx_(&a_tile_top, a + row_block_start * a_stride + depth_offset, a_stride,
                                                 rows_in_high_tile, valid_depth);
                if (rows_in_low_tile > 0) {
                    nk_dots_e3m2_load_a_sapphireamx_(&a_tile_bottom,
                                                     a + (row_block_start + 16) * a_stride + depth_offset, a_stride,
                                                     rows_in_low_tile, valid_depth);
                }

                nk_dots_bf16_load_a_sapphireamx_(&b_as_a, col_edge_ptr + depth_offset, depth, column_remainder_count,
                                                 valid_depth);
                nk_dots_pack_bf16_transposed_sapphireamx_(&b_as_a, &b_tile);

                _tile_loadd(0, a_tile_top.data, 64);
                _tile_loadd(1, a_tile_bottom.data, 64);
                _tile_loadd(2, b_tile.data, 64);

                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(6, 1, 2);
            }

            _tile_stored(4, c_high_state.data, 64);
            _tile_stored(6, c_low_state.data, 64);

            nk_dots_bf16_store_sapphireamx_(&c_high_state, c + row_block_start * c_stride_elements + full_columns,
                                            c_stride_elements, rows_in_high_tile, column_remainder_count);
            if (rows_in_low_tile > 0) {
                nk_dots_bf16_store_sapphireamx_(&c_low_state,
                                                c + (row_block_start + 16) * c_stride_elements + full_columns,
                                                c_stride_elements, rows_in_low_tile, column_remainder_count);
            }
        }
    }

    _tile_release();
    return nk_success_k;
}

/** E3M2 Gram upper triangle of @p vectors on AMX tiles, rows @p rows_begin to @p rows_end. */
NUMKONG_INLINE nk_status_t nk_gram_e3m2_sapphireamx_(                  //
    nk_e3m2_t const *vectors, nk_size_t vector_count, nk_size_t depth, //
    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,       //
    nk_size_t rows_begin, nk_size_t rows_end) {
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));

    nk_size_t const stride_elements = stride; // sizeof(nk_e3m2_t) == 1, so bytes == elements
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);

    // Handle row slicing: compute rows [rows_begin, rows_end)
    rows_end = nk_min_of_two(rows_end, vector_count);

    // Round depth up to multiple of 96 (3 tiles x 32 bf16 elements)
    nk_size_t const depth_tiles = nk_size_divide_round_up_(depth, 32);
    nk_size_t const depth_tile_groups = nk_size_divide_round_up_(depth_tiles, 3);

    nk_dots_bf16_a16x32_sapphireamx_t a_tiles[3];
    nk_dots_bf16_a16x32_sapphireamx_t b_src_tiles[3];
    nk_dots_bf16_b32x16_sapphireamx_t b_tiles[3];
    nk_dots_bf16_state_sapphireamx_t state;

    nk_amx_tile_configure_sapphireamx_();

    for (nk_size_t row_tile = rows_begin; row_tile < rows_end; row_tile += 16) {
        nk_size_t const valid_rows = (row_tile + 16 <= rows_end) ? 16 : (rows_end - row_tile);

        for (nk_size_t col_tile = row_tile; col_tile < vector_count; col_tile += 16) {
            nk_size_t const valid_columns = (col_tile + 16 <= vector_count) ? 16 : (vector_count - col_tile);

            nk_dots_bf16_init_sapphireamx_(&state);

            for (nk_size_t depth_group_idx = 0; depth_group_idx < depth_tile_groups; depth_group_idx++) {
                nk_size_t const depth_base = depth_group_idx * 96;

                for (int tile_idx = 0; tile_idx < 3; tile_idx++) {
                    nk_size_t const depth_start = depth_base + tile_idx * 32;
                    nk_size_t const valid_depth = (depth_start + 32 <= depth)
                                                      ? 32
                                                      : (depth > depth_start ? depth - depth_start : 0);

                    nk_dots_e3m2_load_a_sapphireamx_(                       //
                        &a_tiles[tile_idx],                                 //
                        vectors + row_tile * stride_elements + depth_start, //
                        stride_elements, valid_rows, valid_depth);

                    if (row_tile == col_tile && valid_rows == valid_columns) {
                        nk_dots_pack_bf16_transposed_sapphireamx_(&a_tiles[tile_idx], &b_tiles[tile_idx]);
                    }
                    else {
                        nk_dots_e3m2_load_a_sapphireamx_(                       //
                            &b_src_tiles[tile_idx],                             //
                            vectors + col_tile * stride_elements + depth_start, //
                            stride_elements, valid_columns, valid_depth);
                        nk_dots_pack_bf16_transposed_sapphireamx_(&b_src_tiles[tile_idx], &b_tiles[tile_idx]);
                    }
                }

                nk_dots_bf16_update_sapphireamx_( //
                    &state, &a_tiles[0], &a_tiles[1], &a_tiles[2], &b_tiles[0], &b_tiles[1], &b_tiles[2]);
            }

            nk_dots_symmetric_store_sapphireamx_(                                  //
                state.data, result + row_tile * result_stride_elements + col_tile, //
                result_stride_elements, valid_rows, valid_columns, col_tile - row_tile);
        }
    }
    return nk_success_k;
}

#pragma endregion E3M2 Floats

#if NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_API nk_status_t nk_dots_pack_size_bf16_sapphireamx(nk_size_t column_count, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_packed_bytes_bf16_sapphireamx_(column_count, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_packed_shape_bf16_sapphireamx(void const *b_packed, nk_size_t *columns,
                                                              nk_size_t *depth, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    if (header->capability != nk_cap_sapphireamx_k) return nk_pack_mismatch_k;
    *columns = header->columns;
    *depth = header->depth;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_pack_bf16_sapphireamx(           //
    nk_bf16_t const *b, nk_size_t column_count, nk_size_t depth, //
    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin, nk_size_t columns_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);

    // AMX BF16 tile dimensions: 16 rows × 32 columns (512 BF16 elements = 1KB)
    nk_size_t const tmm_rows = 16;
    nk_size_t const tmm_columns = 32;
    nk_size_t const tile_elements = 512;
    nk_size_t const tile_bytes = tile_elements * sizeof(nk_bf16_t);
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_bf16_t);

    // Compute layout dimensions
    nk_size_t const column_tiles_count = column_count / tmm_rows;
    nk_size_t const depth_tiles_count = nk_size_divide_round_up_(depth, tmm_columns);
    nk_size_t const column_remainder_count = column_count - column_tiles_count * tmm_rows;
    nk_size_t const total_tiles = column_tiles_count * depth_tiles_count;

    // Memory region offsets — every window needs these to locate its output regions.
    nk_size_t const tiles_offset = sizeof(nk_dots_amx_packed_header_t);
    nk_size_t const column_edge_offset = tiles_offset + total_tiles * tile_bytes;

    // Only the window covering column 0 writes the shared header; later windows read it.
    nk_dots_amx_packed_header_t *header = (nk_dots_amx_packed_header_t *)b_packed;
    if (columns_begin == 0) {
        nk_u32_t *header_words = (nk_u32_t *)header;
        for (nk_size_t word_index = 0; word_index < sizeof(*header) / sizeof(nk_u32_t); word_index++)
            header_words[word_index] = 0;
        header->tensor_scale = 1;
        header->columns = (nk_u32_t)column_count;
        header->depth = (nk_u32_t)depth;
        header->full_column_tiles = (nk_u32_t)column_tiles_count;
        header->full_depth_tiles = (nk_u32_t)depth_tiles_count;
        header->column_remainder_count = (nk_u32_t)column_remainder_count;
        header->capability = nk_cap_sapphireamx_k;
        header->column_edge_offset = (nk_u32_t)column_edge_offset;
    }

    // Pointers to packed data regions
    nk_bf16_t *tiles_ptr = (nk_bf16_t *)((char *)b_packed + tiles_offset);
    nk_bf16_t *column_edge_ptr = (nk_bf16_t *)((char *)b_packed + column_edge_offset);

    nk_size_t tile_column_begin = nk_size_divide_round_up_(columns_begin, tmm_rows);
    nk_size_t tile_column_end = nk_size_divide_round_up_(columns_end, tmm_rows);
    if (tile_column_end > column_tiles_count) tile_column_end = column_tiles_count;

    // Pack tiles using vectorized transposer: gather 16 strided rows into an aligned
    // temporary, transpose via SIMD, then copy the result to the packed buffer.
    for (nk_size_t column_tile_idx = tile_column_begin; column_tile_idx < tile_column_end; column_tile_idx++) {
        for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {

            nk_size_t const tile_index = column_tile_idx * depth_tiles_count + depth_tile_idx;
            nk_bf16_t *tile_output = tiles_ptr + tile_index * tile_elements;

            nk_size_t const src_row_start = column_tile_idx * tmm_rows;
            nk_size_t const src_column_start = depth_tile_idx * tmm_columns;
            nk_size_t const columns_to_pack = (src_column_start + tmm_columns <= depth) ? tmm_columns
                                                                                        : (depth - src_column_start);

            // Gather 16 strided source rows into a contiguous aligned tile
            nk_dots_bf16_a16x32_sapphireamx_t source_tile;
            if (columns_to_pack == tmm_columns) {
                for (nk_size_t row_idx = 0; row_idx < tmm_rows; row_idx++) {
                    nk_bf16_t const *source_row = b + (src_row_start + row_idx) * b_stride_elements + src_column_start;
                    _mm512_store_si512(&source_tile.data[row_idx][0], _mm512_loadu_si512(source_row));
                }
            }
            else {
                __mmask32 depth_m32 = (__mmask32)((columns_to_pack < 32) ? ((1U << columns_to_pack) - 1) : ~0U);
                for (nk_size_t row_idx = 0; row_idx < tmm_rows; row_idx++) {
                    nk_bf16_t const *source_row = b + (src_row_start + row_idx) * b_stride_elements + src_column_start;
                    _mm512_store_si512(&source_tile.data[row_idx][0], _mm512_maskz_loadu_epi16(depth_m32, source_row));
                }
            }

            // Transpose into aligned local, then copy to (potentially unaligned) packed buffer
            nk_dots_bf16_b32x16_sapphireamx_t transposed_tile;
            nk_dots_pack_bf16_transposed_sapphireamx_(&source_tile, &transposed_tile);
            for (nk_size_t i = 0; i < tile_bytes; i += 64)
                _mm512_storeu_si512((char *)tile_output + i, _mm512_load_si512((char const *)&transposed_tile + i));
        }
    }

    nk_size_t const remainder_start_row = column_tiles_count * tmm_rows;
    if (column_remainder_count > 0 && remainder_start_row >= columns_begin && remainder_start_row < columns_end)
        for (nk_size_t row_idx = 0; row_idx < column_remainder_count; row_idx++)
            nk_dots_copy_row_b16_sapphireamx_(column_edge_ptr + row_idx * depth,
                                              b + (remainder_start_row + row_idx) * b_stride_elements, depth);

    // Compute and store per-column norms for angular/euclidean distance — one per column, so each
    // window computes the norms of the columns it owns.
    nk_size_t norms_offset = column_edge_offset +
                             (column_remainder_count > 0 ? column_remainder_count * depth * sizeof(nk_bf16_t) : 0);
    if (columns_begin == 0) header->norms_byte_offset = (nk_u32_t)norms_offset;
    nk_f32_t *norms = (nk_f32_t *)((char *)b_packed + norms_offset);
    for (nk_size_t col = columns_begin; col < columns_end; col++)
        norms[col] = nk_dots_reduce_sumsq_bf16_skylake_(b + col * b_stride_elements, depth, sizeof(nk_bf16_t));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_packed_bf16_sapphireamx(   //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_gemm_packed_bf16_sapphireamx_(a, b_packed, c, rows, columns, depth, a_stride, c_stride);
}

NUMKONG_API nk_status_t nk_dots_symmetric_bf16_sapphireamx(            //
    nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth, //
    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,       //
    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_gram_bf16_sapphireamx_(vectors, vector_count, depth, stride, result, result_stride, rows_begin, rows_end);
}

NUMKONG_API nk_status_t nk_dots_pack_size_i8_sapphireamx(nk_size_t column_count, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_packed_bytes_i8_sapphireamx_(column_count, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_packed_shape_i8_sapphireamx(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                            nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    if (header->capability != nk_cap_sapphireamx_k) return nk_pack_mismatch_k;
    *columns = header->columns;
    *depth = header->depth;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_pack_i8_sapphireamx(           //
    nk_i8_t const *b, nk_size_t column_count, nk_size_t depth, //
    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin, nk_size_t columns_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);

    // AMX I8 tile dimensions: 16 rows × 64 columns (1024 I8 elements = 1KB)
    nk_size_t const tmm_rows = 16;
    nk_size_t const tmm_columns = 64;
    nk_size_t const tile_elements = 1024;
    nk_size_t const tile_bytes = tile_elements * sizeof(nk_i8_t);

    // Compute layout dimensions
    nk_size_t const column_tiles_count = column_count / tmm_rows;
    nk_size_t const depth_tiles_count = nk_size_divide_round_up_(depth, tmm_columns);
    nk_size_t const column_remainder_count = column_count - column_tiles_count * tmm_rows;
    nk_size_t const total_tiles = column_tiles_count * depth_tiles_count;

    // Write header with layout metadata
    nk_dots_amx_packed_header_t *header = (nk_dots_amx_packed_header_t *)b_packed;
    if (columns_begin == 0) {
        nk_u32_t *header_words = (nk_u32_t *)header;
        for (nk_size_t word_index = 0; word_index < sizeof(*header) / sizeof(nk_u32_t); word_index++)
            header_words[word_index] = 0;
        header->tensor_scale = 1;
        header->columns = (nk_u32_t)column_count;
        header->depth = (nk_u32_t)depth;
        header->full_column_tiles = (nk_u32_t)column_tiles_count;
        header->full_depth_tiles = (nk_u32_t)depth_tiles_count;
        header->column_remainder_count = (nk_u32_t)column_remainder_count;
        header->capability = nk_cap_sapphireamx_k;
    }

    // Compute memory region offsets
    nk_size_t const tiles_offset = sizeof(nk_dots_amx_packed_header_t);
    nk_size_t const column_edge_offset = tiles_offset + total_tiles * tile_bytes;
    if (columns_begin == 0) header->column_edge_offset = (nk_u32_t)column_edge_offset;

    // Pointers to packed data regions
    nk_i8_t *tiles_ptr = (nk_i8_t *)((char *)b_packed + tiles_offset);
    nk_i8_t *column_edge_ptr = (nk_i8_t *)((char *)b_packed + column_edge_offset);

    // Stack tiles provide the alignment that the packed output may lack.
    for (nk_size_t column_tile_idx = nk_size_divide_round_up_(columns_begin, 16);
         column_tile_idx < column_tiles_count && column_tile_idx < nk_size_divide_round_up_(columns_end, 16);
         column_tile_idx++) {
        for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {

            nk_size_t const tile_index = column_tile_idx * depth_tiles_count + depth_tile_idx;
            nk_i8_t *tile_output = tiles_ptr + tile_index * tile_elements;

            nk_size_t const src_row_start = column_tile_idx * tmm_rows;
            nk_size_t const src_column_start = depth_tile_idx * tmm_columns;
            nk_size_t const columns_to_pack = (src_column_start + tmm_columns <= depth) ? tmm_columns
                                                                                        : (depth - src_column_start);

            // Gather 16 strided source rows into a contiguous aligned tile
            nk_dots_i8_a16x64_sapphireamx_t source_tile;
            if (columns_to_pack == tmm_columns) {
                for (nk_size_t row_idx = 0; row_idx < tmm_rows; row_idx++) {
                    nk_i8_t const *source_row =
                        (nk_i8_t const *)((char const *)b + (src_row_start + row_idx) * b_stride) + src_column_start;
                    _mm512_store_si512(&source_tile.data[row_idx][0], _mm512_loadu_si512(source_row));
                }
            }
            else {
                __mmask64 depth_m64 = (__mmask64)((columns_to_pack < 64) ? ((1ULL << columns_to_pack) - 1) : ~0ULL);
                for (nk_size_t row_idx = 0; row_idx < tmm_rows; row_idx++) {
                    nk_i8_t const *source_row =
                        (nk_i8_t const *)((char const *)b + (src_row_start + row_idx) * b_stride) + src_column_start;
                    _mm512_store_si512(&source_tile.data[row_idx][0], _mm512_maskz_loadu_epi8(depth_m64, source_row));
                }
            }

            // Transpose into aligned local, then copy to (potentially unaligned) packed buffer
            nk_dots_i8_b64x16_sapphireamx_t transposed_tile;
            nk_dots_pack_i8_transposed_sapphireamx_(&source_tile, &transposed_tile);
            for (nk_size_t i = 0; i < tile_elements; i += 64)
                _mm512_storeu_si512(tile_output + i, _mm512_load_si512((char const *)&transposed_tile + i));
        }
    }

    // Pack column-remainder rows in simple row-major format (for AVX-512 fallback)
    if (column_remainder_count > 0 && column_tiles_count * 16 >= columns_begin &&
        column_tiles_count * 16 < columns_end) {
        nk_size_t const remainder_start_row = column_tiles_count * tmm_rows;
        for (nk_size_t row_idx = 0; row_idx < column_remainder_count; row_idx++)
            nk_dots_copy_row_b8_sapphireamx_(column_edge_ptr + row_idx * depth,
                                             b + (remainder_start_row + row_idx) * b_stride, depth);
    }

    // Compute and store per-column norms for angular/euclidean distance
    nk_size_t norms_offset = column_edge_offset +
                             (column_remainder_count > 0 ? column_remainder_count * depth * sizeof(nk_i8_t) : 0);
    if (columns_begin == 0) header->norms_byte_offset = (nk_u32_t)norms_offset;
    nk_u32_t *norms = (nk_u32_t *)((char *)b_packed + norms_offset);
    for (nk_size_t col = columns_begin; col < columns_end; col++)
        norms[col] = nk_dots_reduce_sumsq_i8_skylake_(b + col * b_stride, depth, sizeof(nk_i8_t));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_packed_i8_sapphireamx(   //
    nk_i8_t const *a, void const *b_packed, nk_i32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_gemm_packed_i8_sapphireamx_(a, b_packed, c, rows, columns, depth, a_stride, c_stride);
}

NUMKONG_API nk_status_t nk_dots_symmetric_i8_sapphireamx(            //
    nk_i8_t const *vectors, nk_size_t vector_count, nk_size_t depth, //
    nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,     //
    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_gram_i8_sapphireamx_(vectors, vector_count, depth, stride, result, result_stride, rows_begin, rows_end);
}

NUMKONG_API nk_status_t nk_dots_pack_size_u8_sapphireamx(nk_size_t column_count, nk_size_t depth, nk_size_t *bytes) {
    // Same layout as I8 - just different type interpretation
    *bytes = nk_dots_packed_bytes_i8_sapphireamx_(column_count, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_packed_shape_u8_sapphireamx(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                            nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    if (header->capability != nk_cap_sapphireamx_k) return nk_pack_mismatch_k;
    *columns = header->columns;
    *depth = header->depth;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_pack_u8_sapphireamx(           //
    nk_u8_t const *b, nk_size_t column_count, nk_size_t depth, //
    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin, nk_size_t columns_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);

    nk_size_t const tmm_rows = 16;
    nk_size_t const tmm_columns = 64;
    nk_size_t const tile_elements = 1024;
    nk_size_t const tile_bytes = tile_elements * sizeof(nk_u8_t);

    nk_size_t const column_tiles_count = column_count / tmm_rows;
    nk_size_t const depth_tiles_count = nk_size_divide_round_up_(depth, tmm_columns);
    nk_size_t const column_remainder_count = column_count - column_tiles_count * tmm_rows;
    nk_size_t const total_tiles = column_tiles_count * depth_tiles_count;

    nk_dots_amx_packed_header_t *header = (nk_dots_amx_packed_header_t *)b_packed;
    if (columns_begin == 0) {
        nk_u32_t *header_words = (nk_u32_t *)header;
        for (nk_size_t word_index = 0; word_index < sizeof(*header) / sizeof(nk_u32_t); word_index++)
            header_words[word_index] = 0;
        header->tensor_scale = 1;
        header->columns = (nk_u32_t)column_count;
        header->depth = (nk_u32_t)depth;
        header->full_column_tiles = (nk_u32_t)column_tiles_count;
        header->full_depth_tiles = (nk_u32_t)depth_tiles_count;
        header->column_remainder_count = (nk_u32_t)column_remainder_count;
        header->capability = nk_cap_sapphireamx_k;
    }

    nk_size_t const tiles_offset = sizeof(nk_dots_amx_packed_header_t);
    nk_size_t const column_edge_offset = tiles_offset + total_tiles * tile_bytes;
    if (columns_begin == 0) header->column_edge_offset = (nk_u32_t)column_edge_offset;

    nk_u8_t *tiles_ptr = (nk_u8_t *)((char *)b_packed + tiles_offset);
    nk_u8_t *column_edge_ptr = (nk_u8_t *)((char *)b_packed + column_edge_offset);

    // Stack tiles provide the alignment that the packed output may lack.
    for (nk_size_t column_tile_idx = nk_size_divide_round_up_(columns_begin, 16);
         column_tile_idx < column_tiles_count && column_tile_idx < nk_size_divide_round_up_(columns_end, 16);
         column_tile_idx++) {
        for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {

            nk_size_t const tile_index = column_tile_idx * depth_tiles_count + depth_tile_idx;
            nk_u8_t *tile_output = tiles_ptr + tile_index * tile_elements;

            nk_size_t const src_row_start = column_tile_idx * tmm_rows;
            nk_size_t const src_column_start = depth_tile_idx * tmm_columns;
            nk_size_t const columns_to_pack = (src_column_start + tmm_columns <= depth) ? tmm_columns
                                                                                        : (depth - src_column_start);

            // Gather 16 strided source rows into a contiguous aligned tile
            nk_dots_u8_a16x64_sapphireamx_t source_tile;
            if (columns_to_pack == tmm_columns) {
                for (nk_size_t row_idx = 0; row_idx < tmm_rows; row_idx++) {
                    nk_u8_t const *source_row =
                        (nk_u8_t const *)((char const *)b + (src_row_start + row_idx) * b_stride) + src_column_start;
                    _mm512_store_si512(&source_tile.data[row_idx][0], _mm512_loadu_si512(source_row));
                }
            }
            else {
                __mmask64 depth_m64 = (__mmask64)((columns_to_pack < 64) ? ((1ULL << columns_to_pack) - 1) : ~0ULL);
                for (nk_size_t row_idx = 0; row_idx < tmm_rows; row_idx++) {
                    nk_u8_t const *source_row =
                        (nk_u8_t const *)((char const *)b + (src_row_start + row_idx) * b_stride) + src_column_start;
                    _mm512_store_si512(&source_tile.data[row_idx][0], _mm512_maskz_loadu_epi8(depth_m64, source_row));
                }
            }

            // Transpose into aligned local, then copy to (potentially unaligned) packed buffer
            nk_dots_u8_b64x16_sapphireamx_t transposed_tile;
            nk_dots_pack_u8_transposed_sapphireamx_(&source_tile, &transposed_tile);
            for (nk_size_t i = 0; i < tile_elements; i += 64)
                _mm512_storeu_si512(tile_output + i, _mm512_load_si512((char const *)&transposed_tile + i));
        }
    }

    if (column_remainder_count > 0 && column_tiles_count * 16 >= columns_begin &&
        column_tiles_count * 16 < columns_end) {
        nk_size_t const remainder_start_row = column_tiles_count * tmm_rows;
        for (nk_size_t row_idx = 0; row_idx < column_remainder_count; row_idx++)
            nk_dots_copy_row_b8_sapphireamx_(column_edge_ptr + row_idx * depth,
                                             b + (remainder_start_row + row_idx) * b_stride, depth);
    }

    // Compute and store per-column norms for angular/euclidean distance
    nk_size_t norms_offset = column_edge_offset +
                             (column_remainder_count > 0 ? column_remainder_count * depth * sizeof(nk_u8_t) : 0);
    if (columns_begin == 0) header->norms_byte_offset = (nk_u32_t)norms_offset;
    nk_u32_t *norms = (nk_u32_t *)((char *)b_packed + norms_offset);
    for (nk_size_t col = columns_begin; col < columns_end; col++)
        norms[col] = nk_dots_reduce_sumsq_u8_skylake_(b + col * b_stride, depth, sizeof(nk_u8_t));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_packed_u8_sapphireamx(   //
    nk_u8_t const *a, void const *b_packed, nk_u32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_gemm_packed_u8_sapphireamx_(a, b_packed, c, rows, columns, depth, a_stride, c_stride);
}

NUMKONG_API nk_status_t nk_dots_symmetric_u8_sapphireamx(            //
    nk_u8_t const *vectors, nk_size_t vector_count, nk_size_t depth, //
    nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,     //
    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_gram_u8_sapphireamx_(vectors, vector_count, depth, stride, result, result_stride, rows_begin, rows_end);
}

NUMKONG_API nk_status_t nk_dots_pack_size_e4m3_sapphireamx(nk_size_t column_count, nk_size_t depth, nk_size_t *bytes) {
    // FP8 uses BF16 tile layout after conversion (same element count: 32 per row)
    *bytes = nk_dots_packed_bytes_bf16_sapphireamx_(column_count, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_packed_shape_e4m3_sapphireamx(void const *b_packed, nk_size_t *columns,
                                                              nk_size_t *depth, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    if (header->capability != nk_cap_sapphireamx_k) return nk_pack_mismatch_k;
    *columns = header->columns;
    *depth = header->depth;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_pack_e4m3_sapphireamx(           //
    nk_e4m3_t const *b, nk_size_t column_count, nk_size_t depth, //
    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin, nk_size_t columns_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const column_tiles_count = column_count / 16;
    nk_size_t const depth_tiles_count = nk_size_divide_round_up_(depth, 32);
    nk_size_t const column_remainder_count = column_count % 16;
    nk_dots_bf16_pack_header_sapphireamx_(b_packed, 1, column_count, depth, columns_begin);
    nk_bf16_t *tiles = (nk_bf16_t *)((char *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_bf16_t *column_edge = (nk_bf16_t *)((char *)b_packed +
                                           nk_dots_bf16_column_edge_offset_sapphireamx_(column_count, depth));
    nk_f32_t *norms = (nk_f32_t *)((char *)b_packed + nk_dots_bf16_norms_offset_sapphireamx_(column_count, depth));

    for (nk_size_t column_tile = nk_size_divide_round_up_(columns_begin, 16);
         column_tile < column_tiles_count && column_tile < nk_size_divide_round_up_(columns_end, 16); column_tile++) {
        for (nk_size_t depth_tile = 0; depth_tile < depth_tiles_count; depth_tile++) {
            nk_size_t const depth_offset = depth_tile * 32;
            nk_dots_bf16_a16x32_sapphireamx_t source_tile;
            nk_dots_load_a_e4m3_sapphireamx_(&source_tile, b, b_stride, column_tile * 16, depth_offset, 16,
                                             nk_min_of_two(depth - depth_offset, 32));
            nk_dots_bf16_pack_tile_sapphireamx_(&source_tile,
                                                tiles + (column_tile * depth_tiles_count + depth_tile) * 512);
        }
    }

    if (column_remainder_count > 0 && column_tiles_count * 16 >= columns_begin &&
        column_tiles_count * 16 < columns_end) {
        for (nk_size_t row = 0; row < column_remainder_count; row++)
            for (nk_size_t column = 0; column < depth; column += 32) {
                nk_size_t const columns = nk_min_of_two(depth - column, 32);
                _mm512_mask_storeu_epi16(
                    column_edge + row * depth + column, nk_dots_bf16_valid_mask_sapphireamx_(columns),
                    nk_e4m3_widen_bf16_sapphireamx_(b, b_stride, column_tiles_count * 16 + row, column, columns));
            }
    }

    for (nk_size_t column = columns_begin; column < columns_end; column++)
        norms[column] = nk_e4m3_sumsq_sapphireamx_(b, b_stride, column, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_packed_e4m3_sapphireamx(   //
    nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t row_count, nk_size_t columns, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
    nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_gemm_packed_e4m3_sapphireamx_(a, b_packed, c, row_count, columns, depth, a_stride, c_stride);
}

NUMKONG_API nk_status_t nk_dots_symmetric_e4m3_sapphireamx(            //
    nk_e4m3_t const *vectors, nk_size_t vector_count, nk_size_t depth, //
    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,       //
    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_gram_e4m3_sapphireamx_(vectors, vector_count, depth, stride, result, result_stride, rows_begin, rows_end);
}

NUMKONG_API nk_status_t nk_dots_pack_size_e5m2_sapphireamx(nk_size_t column_count, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_packed_bytes_bf16_sapphireamx_(column_count, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_packed_shape_e5m2_sapphireamx(void const *b_packed, nk_size_t *columns,
                                                              nk_size_t *depth, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    if (header->capability != nk_cap_sapphireamx_k) return nk_pack_mismatch_k;
    *columns = header->columns;
    *depth = header->depth;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_pack_e5m2_sapphireamx(           //
    nk_e5m2_t const *b, nk_size_t column_count, nk_size_t depth, //
    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin, nk_size_t columns_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const column_tiles_count = column_count / 16;
    nk_size_t const depth_tiles_count = nk_size_divide_round_up_(depth, 32);
    nk_size_t const column_remainder_count = column_count % 16;
    nk_dots_bf16_pack_header_sapphireamx_(b_packed, 1, column_count, depth, columns_begin);
    nk_bf16_t *tiles = (nk_bf16_t *)((char *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_bf16_t *column_edge = (nk_bf16_t *)((char *)b_packed +
                                           nk_dots_bf16_column_edge_offset_sapphireamx_(column_count, depth));
    nk_f32_t *norms = (nk_f32_t *)((char *)b_packed + nk_dots_bf16_norms_offset_sapphireamx_(column_count, depth));

    for (nk_size_t column_tile = nk_size_divide_round_up_(columns_begin, 16);
         column_tile < column_tiles_count && column_tile < nk_size_divide_round_up_(columns_end, 16); column_tile++) {
        for (nk_size_t depth_tile = 0; depth_tile < depth_tiles_count; depth_tile++) {
            nk_size_t const depth_offset = depth_tile * 32;
            nk_dots_bf16_a16x32_sapphireamx_t source_tile;
            nk_dots_load_a_e5m2_sapphireamx_(&source_tile, b, b_stride, column_tile * 16, depth_offset, 16,
                                             nk_min_of_two(depth - depth_offset, 32));
            nk_dots_bf16_pack_tile_sapphireamx_(&source_tile,
                                                tiles + (column_tile * depth_tiles_count + depth_tile) * 512);
        }
    }

    if (column_remainder_count > 0 && column_tiles_count * 16 >= columns_begin &&
        column_tiles_count * 16 < columns_end) {
        for (nk_size_t row = 0; row < column_remainder_count; row++)
            for (nk_size_t column = 0; column < depth; column += 32) {
                nk_size_t const columns = nk_min_of_two(depth - column, 32);
                _mm512_mask_storeu_epi16(
                    column_edge + row * depth + column, nk_dots_bf16_valid_mask_sapphireamx_(columns),
                    nk_e5m2_widen_bf16_sapphireamx_(b, b_stride, column_tiles_count * 16 + row, column, columns));
            }
    }

    for (nk_size_t column = columns_begin; column < columns_end; column++)
        norms[column] = nk_e5m2_sumsq_sapphireamx_(b, b_stride, column, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_packed_e5m2_sapphireamx(   //
    nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t row_count, nk_size_t columns, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
    nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_gemm_packed_e5m2_sapphireamx_(a, b_packed, c, row_count, columns, depth, a_stride, c_stride);
}

NUMKONG_API nk_status_t nk_dots_symmetric_e5m2_sapphireamx(            //
    nk_e5m2_t const *vectors, nk_size_t vector_count, nk_size_t depth, //
    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,       //
    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_gram_e5m2_sapphireamx_(vectors, vector_count, depth, stride, result, result_stride, rows_begin, rows_end);
}

NUMKONG_API nk_status_t nk_dots_pack_size_nvfp4_sapphireamx(nk_size_t column_count, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_packed_bytes_scaled_sapphireamx_(16, 8, column_count, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_packed_shape_nvfp4_sapphireamx(void const *b_packed, nk_size_t *columns,
                                                               nk_size_t *depth, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    if (header->capability != nk_cap_sapphireamx_k) return nk_pack_mismatch_k;
    *columns = header->columns, *depth = header->depth;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_pack_nvfp4_sapphireamx(nk_nvfp4_cref_t const *b, nk_size_t column_count,
                                                       nk_size_t depth, nk_size_t b_stride, void *b_packed,
                                                       nk_size_t columns_begin, nk_size_t columns_end,
                                                       nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_cross_operand_t const b_unpacked = nk_cross_operand_serial_(nk_nvfp4_k, b, b_stride);
    nk_e2m1x2_t const *values = (nk_e2m1x2_t const *)b_unpacked.elements;
    nk_ue4m3_t const *scales = (nk_ue4m3_t const *)b_unpacked.scales;
    nk_size_t const scales_stride = b_unpacked.scales_stride;
    nk_f32_t const tensor_scale = nk_cross_tensor_scale_serial_(b_unpacked.tensor_scale);
    nk_cross_tensor_factor_t const factor = nk_cross_tensor_factor_serial_(tensor_scale, tensor_scale);
    nk_size_t const column_tiles_count = column_count / 16;
    nk_size_t const depth_tiles_count = nk_size_divide_round_up_(depth, 32);
    nk_size_t const column_remainder_count = column_count % 16;
    nk_dots_bf16_pack_header_sapphireamx_(b_packed, tensor_scale, column_count, depth, columns_begin);
    nk_bf16_t *tiles = (nk_bf16_t *)((char *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_bf16_t *column_edge = (nk_bf16_t *)((char *)b_packed +
                                           nk_dots_bf16_column_edge_offset_sapphireamx_(column_count, depth));
    nk_dots_scaled_view_sapphireamx_t const view = nk_dots_scaled_view_sapphireamx_(b_packed, column_count, depth, 16,
                                                                                    8);
    nk_i32_t bases[16];

    for (nk_size_t column_tile = nk_size_divide_round_up_(columns_begin, 16);
         column_tile < column_tiles_count && column_tile < nk_size_divide_round_up_(columns_end, 16); column_tile++) {
        nk_dots_bases_nvfp4_sapphireamx_(scales, scales_stride, column_tile * 16, 16, 16, depth, bases);
        for (nk_size_t depth_tile = 0; depth_tile < depth_tiles_count; depth_tile++) {
            nk_size_t const depth_offset = depth_tile * 32;
            nk_dots_bf16_a16x32_sapphireamx_t source_tile;
            nk_dots_load_a_nvfp4_sapphireamx_(&source_tile, values, b_stride, scales, scales_stride, bases,
                                              column_tile * 16, depth_offset, 16,
                                              nk_min_of_two(depth - depth_offset, 32));
            nk_dots_bf16_pack_tile_sapphireamx_(&source_tile,
                                                tiles + (column_tile * depth_tiles_count + depth_tile) * 512);
        }
    }

    if (column_remainder_count > 0 && column_tiles_count * 16 >= columns_begin &&
        column_tiles_count * 16 < columns_end) {
        nk_dots_bases_nvfp4_sapphireamx_(scales, scales_stride, column_tiles_count * 16, column_remainder_count, 16,
                                         depth, bases);
        for (nk_size_t row = 0; row < column_remainder_count; row++)
            for (nk_size_t column = 0; column < depth; column += 32) {
                nk_size_t const columns = nk_min_of_two(depth - column, 32);
                _mm512_mask_storeu_epi16(
                    column_edge + row * depth + column, nk_dots_bf16_valid_mask_sapphireamx_(columns),
                    nk_nvfp4_widen_bf16_sapphireamx_(values, b_stride, scales, scales_stride, bases[row],
                                                     column_tiles_count * 16 + row, column, columns));
            }
    }

    for (nk_size_t column = columns_begin; column < columns_end; column++) {
        nk_i32_t spread;
        nk_i32_t const base = nk_nvfp4_base_sapphireamx_(scales, scales_stride, column, depth, &spread);
        view.norms[column] = nk_nvfp4_sumsq_sapphireamx_(values, b_stride, scales, scales_stride, base, column, depth) *
                             factor.mantissa;
        view.exponents[column] = base + factor.exponent / 2, view.spreads[column] = spread;
        nk_dots_copy_row_b8_sapphireamx_(view.codes + column * view.row_bytes,
                                         (nk_u8_t const *)values + column * b_stride, view.row_bytes);
        nk_dots_copy_row_b8_sapphireamx_(view.scales + column * view.blocks,
                                         (nk_u8_t const *)scales + column * scales_stride, view.blocks);
    }
    if (columns_end == column_count) nk_dots_scaled_pad_sapphireamx_(&view, column_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_packed_nvfp4_sapphireamx(nk_nvfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_cross_operand_t const a_unpacked = nk_cross_operand_serial_(nk_nvfp4_k, a, a_stride);
    nk_e2m1x2_t const *values = (nk_e2m1x2_t const *)a_unpacked.elements;
    nk_ue4m3_t const *scales = (nk_ue4m3_t const *)a_unpacked.scales;
    nk_size_t const scales_stride = a_unpacked.scales_stride;
    nk_f32_t const tensor_scale = nk_cross_tensor_scale_serial_(a_unpacked.tensor_scale);
    nk_status_t const status = nk_gemm_packed_nvfp4_sapphireamx_(values, scales, b_packed, c, rows, columns, depth,
                                                                 a_stride, scales_stride, c_stride);
    if (status != nk_success_k) return status;
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_dots_scaled_view_sapphireamx_t const b = nk_dots_scaled_view_sapphireamx_(b_packed, header->columns, depth, 16,
                                                                                 8);
    nk_cross_tensor_factor_t const factor = nk_cross_tensor_factor_serial_(tensor_scale, header->tensor_scale);
    nk_cross_tensor_factor_t const a_factor = nk_cross_tensor_factor_serial_(tensor_scale, tensor_scale);

    for (nk_size_t row = 0; row < rows; row++) {
        nk_i32_t row_spread;
        nk_i32_t const row_base = nk_nvfp4_base_sapphireamx_(scales, scales_stride, row, depth, &row_spread);
        nk_f32_t *c_row = (nk_f32_t *)((char *)c + row * c_stride);
        nk_dots_scaled_finish_row_dot_sapphireamx_(c_row, columns, factor.mantissa, row_base + a_factor.exponent / 2,
                                                   b.exponents, 0, b.norms);
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_symmetric_nvfp4_sapphireamx(nk_nvfp4_cref_t const *vectors, nk_size_t vector_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t rows_begin,
                                                            nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_cross_operand_t const vectors_unpacked = nk_cross_operand_serial_(nk_nvfp4_k, vectors, stride);
    nk_e2m1x2_t const *values = (nk_e2m1x2_t const *)vectors_unpacked.elements;
    nk_ue4m3_t const *scales = (nk_ue4m3_t const *)vectors_unpacked.scales;
    nk_size_t const scales_stride = vectors_unpacked.scales_stride;
    nk_f32_t const tensor_scale = nk_cross_tensor_scale_serial_(vectors_unpacked.tensor_scale);
    nk_status_t const status = nk_gram_nvfp4_sapphireamx_(values, scales, vector_count, depth, stride, scales_stride,
                                                          result, result_stride, rows_begin, rows_end);
    if (status != nk_success_k) return status;
    nk_cross_tensor_factor_t const factor = nk_cross_tensor_factor_serial_(tensor_scale, tensor_scale);
    nk_i32_t column_exponents[256 + 4] = {0};
    nk_f32_t column_norms[256 + 4] = {0};
    for (nk_size_t chunk_start = rows_begin; chunk_start < vector_count && rows_begin < rows_end; chunk_start += 256) {
        nk_size_t const chunk_end = nk_min_of_two(chunk_start + 256, vector_count);
        for (nk_size_t column = chunk_start; column < chunk_end; column++) {
            nk_i32_t spread;
            nk_i32_t const base = nk_nvfp4_base_sapphireamx_(scales, scales_stride, column, depth, &spread);
            column_exponents[column - chunk_start] = base + factor.exponent / 2;
        }
        for (nk_size_t row = rows_begin; row < rows_end; row++) {
            nk_size_t const column_first = chunk_start > row + 0 ? chunk_start : row + 0;
            if (column_first >= chunk_end) continue;
            nk_f32_t *result_row = (nk_f32_t *)((char *)result + row * result_stride);
            nk_i32_t row_spread;
            nk_i32_t const row_base = nk_nvfp4_base_sapphireamx_(scales, scales_stride, row, depth, &row_spread);
            nk_dots_scaled_finish_row_dot_sapphireamx_(
                result_row + column_first, chunk_end - column_first, factor.mantissa, row_base + factor.exponent / 2,
                column_exponents + (column_first - chunk_start), 0, column_norms + (column_first - chunk_start));
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_pack_size_mxfp4_sapphireamx(nk_size_t column_count, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_packed_bytes_scaled_sapphireamx_(32, 16, column_count, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp4_sapphireamx(void const *b_packed, nk_size_t *columns,
                                                               nk_size_t *depth, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    if (header->capability != nk_cap_sapphireamx_k) return nk_pack_mismatch_k;
    *columns = header->columns, *depth = header->depth;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_pack_mxfp4_sapphireamx(nk_mxfp4_cref_t const *b, nk_size_t column_count,
                                                       nk_size_t depth, nk_size_t b_stride, void *b_packed,
                                                       nk_size_t columns_begin, nk_size_t columns_end,
                                                       nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_cross_operand_t const b_unpacked = nk_cross_operand_serial_(nk_mxfp4_k, b, b_stride);
    nk_e2m1x2_t const *values = (nk_e2m1x2_t const *)b_unpacked.elements;
    nk_ue8m0_t const *scales = (nk_ue8m0_t const *)b_unpacked.scales;
    nk_size_t const scales_stride = b_unpacked.scales_stride;
    nk_cross_tensor_factor_t const factor = nk_cross_tensor_factor_serial_(1, 1);
    nk_size_t const column_tiles_count = column_count / 16;
    nk_size_t const depth_tiles_count = nk_size_divide_round_up_(depth, 32);
    nk_size_t const column_remainder_count = column_count % 16;
    nk_dots_bf16_pack_header_sapphireamx_(b_packed, 1, column_count, depth, columns_begin);
    nk_bf16_t *tiles = (nk_bf16_t *)((char *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_bf16_t *column_edge = (nk_bf16_t *)((char *)b_packed +
                                           nk_dots_bf16_column_edge_offset_sapphireamx_(column_count, depth));
    nk_dots_scaled_view_sapphireamx_t const view = nk_dots_scaled_view_sapphireamx_(b_packed, column_count, depth, 32,
                                                                                    16);
    nk_i32_t bases[16];

    for (nk_size_t column_tile = nk_size_divide_round_up_(columns_begin, 16);
         column_tile < column_tiles_count && column_tile < nk_size_divide_round_up_(columns_end, 16); column_tile++) {
        nk_dots_bases_mxfp4_sapphireamx_(scales, scales_stride, column_tile * 16, 16, 16, depth, bases);
        for (nk_size_t depth_tile = 0; depth_tile < depth_tiles_count; depth_tile++) {
            nk_size_t const depth_offset = depth_tile * 32;
            nk_dots_bf16_a16x32_sapphireamx_t source_tile;
            nk_dots_load_a_mxfp4_sapphireamx_(&source_tile, values, b_stride, scales, scales_stride, bases,
                                              column_tile * 16, depth_offset, 16,
                                              nk_min_of_two(depth - depth_offset, 32));
            nk_dots_bf16_pack_tile_sapphireamx_(&source_tile,
                                                tiles + (column_tile * depth_tiles_count + depth_tile) * 512);
        }
    }

    if (column_remainder_count > 0 && column_tiles_count * 16 >= columns_begin &&
        column_tiles_count * 16 < columns_end) {
        nk_dots_bases_mxfp4_sapphireamx_(scales, scales_stride, column_tiles_count * 16, column_remainder_count, 16,
                                         depth, bases);
        for (nk_size_t row = 0; row < column_remainder_count; row++)
            for (nk_size_t column = 0; column < depth; column += 32) {
                nk_size_t const columns = nk_min_of_two(depth - column, 32);
                _mm512_mask_storeu_epi16(
                    column_edge + row * depth + column, nk_dots_bf16_valid_mask_sapphireamx_(columns),
                    nk_mxfp4_widen_bf16_sapphireamx_(values, b_stride, scales, scales_stride, bases[row],
                                                     column_tiles_count * 16 + row, column, columns));
            }
    }

    for (nk_size_t column = columns_begin; column < columns_end; column++) {
        nk_i32_t spread;
        nk_i32_t const base = nk_mxfp4_base_sapphireamx_(scales, scales_stride, column, depth, &spread);
        view.norms[column] = nk_mxfp4_sumsq_sapphireamx_(values, b_stride, scales, scales_stride, base, column, depth) *
                             factor.mantissa;
        view.exponents[column] = base + factor.exponent / 2, view.spreads[column] = spread;
        nk_dots_copy_row_b8_sapphireamx_(view.codes + column * view.row_bytes,
                                         (nk_u8_t const *)values + column * b_stride, view.row_bytes);
        nk_dots_copy_row_b8_sapphireamx_(view.scales + column * view.blocks,
                                         (nk_u8_t const *)scales + column * scales_stride, view.blocks);
    }
    if (columns_end == column_count) nk_dots_scaled_pad_sapphireamx_(&view, column_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_packed_mxfp4_sapphireamx(nk_mxfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_cross_operand_t const a_unpacked = nk_cross_operand_serial_(nk_mxfp4_k, a, a_stride);
    nk_e2m1x2_t const *values = (nk_e2m1x2_t const *)a_unpacked.elements;
    nk_ue8m0_t const *scales = (nk_ue8m0_t const *)a_unpacked.scales;
    nk_size_t const scales_stride = a_unpacked.scales_stride;
    nk_status_t const status = nk_gemm_packed_mxfp4_sapphireamx_(values, scales, b_packed, c, rows, columns, depth,
                                                                 a_stride, scales_stride, c_stride);
    if (status != nk_success_k) return status;
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_dots_scaled_view_sapphireamx_t const b = nk_dots_scaled_view_sapphireamx_(b_packed, header->columns, depth, 32,
                                                                                 16);
    nk_i32_t const column_spread_max = nk_dots_scaled_spread_max_sapphireamx_(b.spreads, b.padded);

    for (nk_size_t row = 0; row < rows; row++) {
        nk_i32_t row_spread;
        nk_i32_t const row_base = nk_mxfp4_base_sapphireamx_(scales, scales_stride, row, depth, &row_spread);
        nk_f32_t *c_row = (nk_f32_t *)((char *)c + row * c_stride);
        nk_dots_scaled_finish_row_dot_sapphireamx_(c_row, columns, 1, row_base, b.exponents, 0, b.norms);
        if (!nk_dots_scaled_exceeds_sapphireamx_(row_spread, column_spread_max, 0,
                                                 nk_dots_spread_limit_mxfp4_sapphireamx_k))
            continue;
        for (nk_size_t column = 0; column < columns; column++) {
            if (!nk_dots_scaled_exceeds_sapphireamx_(row_spread, b.spreads[column], 0,
                                                     nk_dots_spread_limit_mxfp4_sapphireamx_k))
                continue;
            nk_cross_wide_sum_t a_sumsq, b_sumsq;
            nk_cross_wide_sum_t const dot = nk_cross_scaled_exact_wide_mxfp4_serial_(
                (nk_u8_t const *)values + row * a_stride, (nk_u8_t const *)scales + row * scales_stride,
                b.codes + column * b.row_bytes, b.scales + column * b.blocks, depth, &a_sumsq, &b_sumsq);
            c_row[column] = nk_dot_from_wide_f32_serial_(dot, a_sumsq, b_sumsq);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_symmetric_mxfp4_sapphireamx(nk_mxfp4_cref_t const *vectors, nk_size_t vector_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t rows_begin,
                                                            nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_cross_operand_t const vectors_unpacked = nk_cross_operand_serial_(nk_mxfp4_k, vectors, stride);
    nk_e2m1x2_t const *values = (nk_e2m1x2_t const *)vectors_unpacked.elements;
    nk_ue8m0_t const *scales = (nk_ue8m0_t const *)vectors_unpacked.scales;
    nk_size_t const scales_stride = vectors_unpacked.scales_stride;
    nk_status_t const status = nk_gram_mxfp4_sapphireamx_(values, scales, vector_count, depth, stride, scales_stride,
                                                          result, result_stride, rows_begin, rows_end);
    if (status != nk_success_k) return status;
    nk_i32_t column_exponents[256 + 4] = {0};
    nk_i32_t column_spreads[256] = {0};
    nk_f32_t column_norms[256 + 4] = {0};
    for (nk_size_t chunk_start = rows_begin; chunk_start < vector_count && rows_begin < rows_end; chunk_start += 256) {
        nk_size_t const chunk_end = nk_min_of_two(chunk_start + 256, vector_count);
        nk_i32_t column_spread_max = 0;
        for (nk_size_t column = chunk_start; column < chunk_end; column++) {
            nk_i32_t spread;
            nk_i32_t const base = nk_mxfp4_base_sapphireamx_(scales, scales_stride, column, depth, &spread);
            column_exponents[column - chunk_start] = base;
            column_spreads[column - chunk_start] = spread;
            column_spread_max = spread > column_spread_max ? spread : column_spread_max;
        }
        for (nk_size_t row = rows_begin; row < rows_end; row++) {
            nk_size_t const column_first = chunk_start > row + 0 ? chunk_start : row + 0;
            if (column_first >= chunk_end) continue;
            nk_f32_t *result_row = (nk_f32_t *)((char *)result + row * result_stride);
            nk_i32_t row_spread;
            nk_i32_t const row_base = nk_mxfp4_base_sapphireamx_(scales, scales_stride, row, depth, &row_spread);
            nk_dots_scaled_finish_row_dot_sapphireamx_(result_row + column_first, chunk_end - column_first, 1, row_base,
                                                       column_exponents + (column_first - chunk_start), 0,
                                                       column_norms + (column_first - chunk_start));
            if (!nk_dots_scaled_exceeds_sapphireamx_(row_spread, column_spread_max, 0,
                                                     nk_dots_spread_limit_mxfp4_sapphireamx_k))
                continue;
            for (nk_size_t column = column_first; column < chunk_end; column++) {
                if (!nk_dots_scaled_exceeds_sapphireamx_(row_spread, column_spreads[column - chunk_start], 0,
                                                         nk_dots_spread_limit_mxfp4_sapphireamx_k))
                    continue;
                nk_cross_wide_sum_t row_sumsq, column_sumsq;
                nk_cross_wide_sum_t const dot = nk_cross_scaled_exact_wide_mxfp4_serial_(
                    (nk_u8_t const *)values + row * stride, (nk_u8_t const *)scales + row * scales_stride,
                    (nk_u8_t const *)values + column * stride, (nk_u8_t const *)scales + column * scales_stride, depth,
                    &row_sumsq, &column_sumsq);
                result_row[column] = nk_dot_from_wide_f32_serial_(dot, row_sumsq, column_sumsq);
            }
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_pack_size_mxfp8e4m3_sapphireamx(nk_size_t column_count, nk_size_t depth,
                                                                nk_size_t *bytes) {
    *bytes = nk_dots_packed_bytes_scaled_sapphireamx_(32, 32, column_count, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp8e4m3_sapphireamx(void const *b_packed, nk_size_t *columns,
                                                                   nk_size_t *depth, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    if (header->capability != nk_cap_sapphireamx_k) return nk_pack_mismatch_k;
    *columns = header->columns, *depth = header->depth;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_pack_mxfp8e4m3_sapphireamx(nk_mxfp8e4m3_cref_t const *b, nk_size_t column_count,
                                                           nk_size_t depth, nk_size_t b_stride, void *b_packed,
                                                           nk_size_t columns_begin, nk_size_t columns_end,
                                                           nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_cross_operand_t const b_unpacked = nk_cross_operand_serial_(nk_mxfp8e4m3_k, b, b_stride);
    nk_e4m3_t const *values = (nk_e4m3_t const *)b_unpacked.elements;
    nk_ue8m0_t const *scales = (nk_ue8m0_t const *)b_unpacked.scales;
    nk_size_t const scales_stride = b_unpacked.scales_stride;
    nk_cross_tensor_factor_t const factor = nk_cross_tensor_factor_serial_(1, 1);
    nk_size_t const column_tiles_count = column_count / 16;
    nk_size_t const depth_tiles_count = nk_size_divide_round_up_(depth, 32);
    nk_size_t const column_remainder_count = column_count % 16;
    nk_dots_bf16_pack_header_sapphireamx_(b_packed, 1, column_count, depth, columns_begin);
    nk_bf16_t *tiles = (nk_bf16_t *)((char *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_bf16_t *column_edge = (nk_bf16_t *)((char *)b_packed +
                                           nk_dots_bf16_column_edge_offset_sapphireamx_(column_count, depth));
    nk_dots_scaled_view_sapphireamx_t const view = nk_dots_scaled_view_sapphireamx_(b_packed, column_count, depth, 32,
                                                                                    32);
    nk_i32_t bases[16];

    for (nk_size_t column_tile = nk_size_divide_round_up_(columns_begin, 16);
         column_tile < column_tiles_count && column_tile < nk_size_divide_round_up_(columns_end, 16); column_tile++) {
        nk_dots_bases_mxfp8e4m3_sapphireamx_(scales, scales_stride, column_tile * 16, 16, 16, depth, bases);
        for (nk_size_t depth_tile = 0; depth_tile < depth_tiles_count; depth_tile++) {
            nk_size_t const depth_offset = depth_tile * 32;
            nk_dots_bf16_a16x32_sapphireamx_t source_tile;
            nk_dots_load_a_mxfp8e4m3_sapphireamx_(&source_tile, values, b_stride, scales, scales_stride, bases,
                                                  column_tile * 16, depth_offset, 16,
                                                  nk_min_of_two(depth - depth_offset, 32));
            nk_dots_bf16_pack_tile_sapphireamx_(&source_tile,
                                                tiles + (column_tile * depth_tiles_count + depth_tile) * 512);
        }
    }

    if (column_remainder_count > 0 && column_tiles_count * 16 >= columns_begin &&
        column_tiles_count * 16 < columns_end) {
        nk_dots_bases_mxfp8e4m3_sapphireamx_(scales, scales_stride, column_tiles_count * 16, column_remainder_count, 16,
                                             depth, bases);
        for (nk_size_t row = 0; row < column_remainder_count; row++)
            for (nk_size_t column = 0; column < depth; column += 32) {
                nk_size_t const columns = nk_min_of_two(depth - column, 32);
                _mm512_mask_storeu_epi16(
                    column_edge + row * depth + column, nk_dots_bf16_valid_mask_sapphireamx_(columns),
                    nk_mxfp8e4m3_widen_bf16_sapphireamx_(values, b_stride, scales, scales_stride, bases[row],
                                                         column_tiles_count * 16 + row, column, columns));
            }
    }

    for (nk_size_t column = columns_begin; column < columns_end; column++) {
        nk_i32_t spread;
        nk_i32_t const base = nk_mxfp8e4m3_base_sapphireamx_(scales, scales_stride, column, depth, &spread);
        view.norms[column] = nk_mxfp8e4m3_sumsq_sapphireamx_(values, b_stride, scales, scales_stride, base, column,
                                                             depth) *
                             factor.mantissa;
        view.exponents[column] = base + factor.exponent / 2, view.spreads[column] = spread;
        nk_dots_copy_row_b8_sapphireamx_(view.codes + column * view.row_bytes,
                                         (nk_u8_t const *)values + column * b_stride, view.row_bytes);
        nk_dots_copy_row_b8_sapphireamx_(view.scales + column * view.blocks,
                                         (nk_u8_t const *)scales + column * scales_stride, view.blocks);
    }
    if (columns_end == column_count) nk_dots_scaled_pad_sapphireamx_(&view, column_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_packed_mxfp8e4m3_sapphireamx(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                             nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                             nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                             nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_cross_operand_t const a_unpacked = nk_cross_operand_serial_(nk_mxfp8e4m3_k, a, a_stride);
    nk_e4m3_t const *values = (nk_e4m3_t const *)a_unpacked.elements;
    nk_ue8m0_t const *scales = (nk_ue8m0_t const *)a_unpacked.scales;
    nk_size_t const scales_stride = a_unpacked.scales_stride;
    nk_status_t const status = nk_gemm_packed_mxfp8e4m3_sapphireamx_(values, scales, b_packed, c, rows, columns, depth,
                                                                     a_stride, scales_stride, c_stride);
    if (status != nk_success_k) return status;
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_dots_scaled_view_sapphireamx_t const b = nk_dots_scaled_view_sapphireamx_(b_packed, header->columns, depth, 32,
                                                                                 32);
    nk_i32_t const column_spread_max = nk_dots_scaled_spread_max_sapphireamx_(b.spreads, b.padded);

    for (nk_size_t row = 0; row < rows; row++) {
        nk_i32_t row_spread;
        nk_i32_t const row_base = nk_mxfp8e4m3_base_sapphireamx_(scales, scales_stride, row, depth, &row_spread);
        nk_f32_t *c_row = (nk_f32_t *)((char *)c + row * c_stride);
        nk_dots_scaled_finish_row_dot_sapphireamx_(c_row, columns, 1, row_base, b.exponents, 0, b.norms);
        if (!nk_dots_scaled_exceeds_sapphireamx_(row_spread, column_spread_max, 0,
                                                 nk_dots_spread_limit_mxfp8e4m3_sapphireamx_k))
            continue;
        for (nk_size_t column = 0; column < columns; column++) {
            if (!nk_dots_scaled_exceeds_sapphireamx_(row_spread, b.spreads[column], 0,
                                                     nk_dots_spread_limit_mxfp8e4m3_sapphireamx_k))
                continue;
            nk_cross_wide_sum_t a_sumsq, b_sumsq;
            nk_cross_wide_sum_t const dot = nk_cross_scaled_exact_wide_mxfp8e4m3_serial_(
                (nk_u8_t const *)values + row * a_stride, (nk_u8_t const *)scales + row * scales_stride,
                b.codes + column * b.row_bytes, b.scales + column * b.blocks, depth, &a_sumsq, &b_sumsq);
            c_row[column] = nk_dot_from_wide_f32_serial_(dot, a_sumsq, b_sumsq);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_symmetric_mxfp8e4m3_sapphireamx(nk_mxfp8e4m3_cref_t const *vectors,
                                                                nk_size_t vector_count, nk_size_t depth,
                                                                nk_size_t stride, nk_f32_t *result,
                                                                nk_size_t result_stride, nk_size_t rows_begin,
                                                                nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_cross_operand_t const vectors_unpacked = nk_cross_operand_serial_(nk_mxfp8e4m3_k, vectors, stride);
    nk_e4m3_t const *values = (nk_e4m3_t const *)vectors_unpacked.elements;
    nk_ue8m0_t const *scales = (nk_ue8m0_t const *)vectors_unpacked.scales;
    nk_size_t const scales_stride = vectors_unpacked.scales_stride;
    nk_status_t const status = nk_gram_mxfp8e4m3_sapphireamx_(
        values, scales, vector_count, depth, stride, scales_stride, result, result_stride, rows_begin, rows_end);
    if (status != nk_success_k) return status;
    nk_i32_t column_exponents[256 + 4] = {0};
    nk_i32_t column_spreads[256] = {0};
    nk_f32_t column_norms[256 + 4] = {0};
    for (nk_size_t chunk_start = rows_begin; chunk_start < vector_count && rows_begin < rows_end; chunk_start += 256) {
        nk_size_t const chunk_end = nk_min_of_two(chunk_start + 256, vector_count);
        nk_i32_t column_spread_max = 0;
        for (nk_size_t column = chunk_start; column < chunk_end; column++) {
            nk_i32_t spread;
            nk_i32_t const base = nk_mxfp8e4m3_base_sapphireamx_(scales, scales_stride, column, depth, &spread);
            column_exponents[column - chunk_start] = base;
            column_spreads[column - chunk_start] = spread;
            column_spread_max = spread > column_spread_max ? spread : column_spread_max;
        }
        for (nk_size_t row = rows_begin; row < rows_end; row++) {
            nk_size_t const column_first = chunk_start > row + 0 ? chunk_start : row + 0;
            if (column_first >= chunk_end) continue;
            nk_f32_t *result_row = (nk_f32_t *)((char *)result + row * result_stride);
            nk_i32_t row_spread;
            nk_i32_t const row_base = nk_mxfp8e4m3_base_sapphireamx_(scales, scales_stride, row, depth, &row_spread);
            nk_dots_scaled_finish_row_dot_sapphireamx_(result_row + column_first, chunk_end - column_first, 1, row_base,
                                                       column_exponents + (column_first - chunk_start), 0,
                                                       column_norms + (column_first - chunk_start));
            if (!nk_dots_scaled_exceeds_sapphireamx_(row_spread, column_spread_max, 0,
                                                     nk_dots_spread_limit_mxfp8e4m3_sapphireamx_k))
                continue;
            for (nk_size_t column = column_first; column < chunk_end; column++) {
                if (!nk_dots_scaled_exceeds_sapphireamx_(row_spread, column_spreads[column - chunk_start], 0,
                                                         nk_dots_spread_limit_mxfp8e4m3_sapphireamx_k))
                    continue;
                nk_cross_wide_sum_t row_sumsq, column_sumsq;
                nk_cross_wide_sum_t const dot = nk_cross_scaled_exact_wide_mxfp8e4m3_serial_(
                    (nk_u8_t const *)values + row * stride, (nk_u8_t const *)scales + row * scales_stride,
                    (nk_u8_t const *)values + column * stride, (nk_u8_t const *)scales + column * scales_stride, depth,
                    &row_sumsq, &column_sumsq);
                result_row[column] = nk_dot_from_wide_f32_serial_(dot, row_sumsq, column_sumsq);
            }
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_pack_size_mxfp8e5m2_sapphireamx(nk_size_t column_count, nk_size_t depth,
                                                                nk_size_t *bytes) {
    *bytes = nk_dots_packed_bytes_scaled_sapphireamx_(32, 32, column_count, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp8e5m2_sapphireamx(void const *b_packed, nk_size_t *columns,
                                                                   nk_size_t *depth, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    if (header->capability != nk_cap_sapphireamx_k) return nk_pack_mismatch_k;
    *columns = header->columns, *depth = header->depth;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_pack_mxfp8e5m2_sapphireamx(nk_mxfp8e5m2_cref_t const *b, nk_size_t column_count,
                                                           nk_size_t depth, nk_size_t b_stride, void *b_packed,
                                                           nk_size_t columns_begin, nk_size_t columns_end,
                                                           nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_cross_operand_t const b_unpacked = nk_cross_operand_serial_(nk_mxfp8e5m2_k, b, b_stride);
    nk_e5m2_t const *values = (nk_e5m2_t const *)b_unpacked.elements;
    nk_ue8m0_t const *scales = (nk_ue8m0_t const *)b_unpacked.scales;
    nk_size_t const scales_stride = b_unpacked.scales_stride;
    nk_cross_tensor_factor_t const factor = nk_cross_tensor_factor_serial_(1, 1);
    nk_size_t const column_tiles_count = column_count / 16;
    nk_size_t const depth_tiles_count = nk_size_divide_round_up_(depth, 32);
    nk_size_t const column_remainder_count = column_count % 16;
    nk_dots_bf16_pack_header_sapphireamx_(b_packed, 1, column_count, depth, columns_begin);
    nk_bf16_t *tiles = (nk_bf16_t *)((char *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_bf16_t *column_edge = (nk_bf16_t *)((char *)b_packed +
                                           nk_dots_bf16_column_edge_offset_sapphireamx_(column_count, depth));
    nk_dots_scaled_view_sapphireamx_t const view = nk_dots_scaled_view_sapphireamx_(b_packed, column_count, depth, 32,
                                                                                    32);
    nk_i32_t bases[16];

    for (nk_size_t column_tile = nk_size_divide_round_up_(columns_begin, 16);
         column_tile < column_tiles_count && column_tile < nk_size_divide_round_up_(columns_end, 16); column_tile++) {
        nk_dots_bases_mxfp8e5m2_sapphireamx_(scales, scales_stride, column_tile * 16, 16, 16, depth, bases);
        for (nk_size_t depth_tile = 0; depth_tile < depth_tiles_count; depth_tile++) {
            nk_size_t const depth_offset = depth_tile * 32;
            nk_dots_bf16_a16x32_sapphireamx_t source_tile;
            nk_dots_load_a_mxfp8e5m2_sapphireamx_(&source_tile, values, b_stride, scales, scales_stride, bases,
                                                  column_tile * 16, depth_offset, 16,
                                                  nk_min_of_two(depth - depth_offset, 32));
            nk_dots_bf16_pack_tile_sapphireamx_(&source_tile,
                                                tiles + (column_tile * depth_tiles_count + depth_tile) * 512);
        }
    }

    if (column_remainder_count > 0 && column_tiles_count * 16 >= columns_begin &&
        column_tiles_count * 16 < columns_end) {
        nk_dots_bases_mxfp8e5m2_sapphireamx_(scales, scales_stride, column_tiles_count * 16, column_remainder_count, 16,
                                             depth, bases);
        for (nk_size_t row = 0; row < column_remainder_count; row++)
            for (nk_size_t column = 0; column < depth; column += 32) {
                nk_size_t const columns = nk_min_of_two(depth - column, 32);
                _mm512_mask_storeu_epi16(
                    column_edge + row * depth + column, nk_dots_bf16_valid_mask_sapphireamx_(columns),
                    nk_mxfp8e5m2_widen_bf16_sapphireamx_(values, b_stride, scales, scales_stride, bases[row],
                                                         column_tiles_count * 16 + row, column, columns));
            }
    }

    for (nk_size_t column = columns_begin; column < columns_end; column++) {
        nk_i32_t spread;
        nk_i32_t const base = nk_mxfp8e5m2_base_sapphireamx_(scales, scales_stride, column, depth, &spread);
        view.norms[column] = nk_mxfp8e5m2_sumsq_sapphireamx_(values, b_stride, scales, scales_stride, base, column,
                                                             depth) *
                             factor.mantissa;
        view.exponents[column] = base + factor.exponent / 2, view.spreads[column] = spread;
        nk_dots_copy_row_b8_sapphireamx_(view.codes + column * view.row_bytes,
                                         (nk_u8_t const *)values + column * b_stride, view.row_bytes);
        nk_dots_copy_row_b8_sapphireamx_(view.scales + column * view.blocks,
                                         (nk_u8_t const *)scales + column * scales_stride, view.blocks);
    }
    if (columns_end == column_count) nk_dots_scaled_pad_sapphireamx_(&view, column_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_packed_mxfp8e5m2_sapphireamx(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                             nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                             nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                             nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_cross_operand_t const a_unpacked = nk_cross_operand_serial_(nk_mxfp8e5m2_k, a, a_stride);
    nk_e5m2_t const *values = (nk_e5m2_t const *)a_unpacked.elements;
    nk_ue8m0_t const *scales = (nk_ue8m0_t const *)a_unpacked.scales;
    nk_size_t const scales_stride = a_unpacked.scales_stride;
    nk_status_t const status = nk_gemm_packed_mxfp8e5m2_sapphireamx_(values, scales, b_packed, c, rows, columns, depth,
                                                                     a_stride, scales_stride, c_stride);
    if (status != nk_success_k) return status;
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_dots_scaled_view_sapphireamx_t const b = nk_dots_scaled_view_sapphireamx_(b_packed, header->columns, depth, 32,
                                                                                 32);
    nk_i32_t const column_spread_max = nk_dots_scaled_spread_max_sapphireamx_(b.spreads, b.padded);

    for (nk_size_t row = 0; row < rows; row++) {
        nk_i32_t row_spread;
        nk_i32_t const row_base = nk_mxfp8e5m2_base_sapphireamx_(scales, scales_stride, row, depth, &row_spread);
        nk_f32_t *c_row = (nk_f32_t *)((char *)c + row * c_stride);
        nk_dots_scaled_finish_row_dot_sapphireamx_(c_row, columns, 1, row_base, b.exponents, 0, b.norms);
        if (!nk_dots_scaled_exceeds_sapphireamx_(row_spread, column_spread_max, 0,
                                                 nk_dots_spread_limit_mxfp8e5m2_sapphireamx_k))
            continue;
        for (nk_size_t column = 0; column < columns; column++) {
            if (!nk_dots_scaled_exceeds_sapphireamx_(row_spread, b.spreads[column], 0,
                                                     nk_dots_spread_limit_mxfp8e5m2_sapphireamx_k))
                continue;
            nk_cross_wide_sum_t a_sumsq, b_sumsq;
            nk_cross_wide_sum_t const dot = nk_cross_scaled_exact_wide_mxfp8e5m2_serial_(
                (nk_u8_t const *)values + row * a_stride, (nk_u8_t const *)scales + row * scales_stride,
                b.codes + column * b.row_bytes, b.scales + column * b.blocks, depth, &a_sumsq, &b_sumsq);
            c_row[column] = nk_dot_from_wide_f32_serial_(dot, a_sumsq, b_sumsq);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_symmetric_mxfp8e5m2_sapphireamx(nk_mxfp8e5m2_cref_t const *vectors,
                                                                nk_size_t vector_count, nk_size_t depth,
                                                                nk_size_t stride, nk_f32_t *result,
                                                                nk_size_t result_stride, nk_size_t rows_begin,
                                                                nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_cross_operand_t const vectors_unpacked = nk_cross_operand_serial_(nk_mxfp8e5m2_k, vectors, stride);
    nk_e5m2_t const *values = (nk_e5m2_t const *)vectors_unpacked.elements;
    nk_ue8m0_t const *scales = (nk_ue8m0_t const *)vectors_unpacked.scales;
    nk_size_t const scales_stride = vectors_unpacked.scales_stride;
    nk_status_t const status = nk_gram_mxfp8e5m2_sapphireamx_(
        values, scales, vector_count, depth, stride, scales_stride, result, result_stride, rows_begin, rows_end);
    if (status != nk_success_k) return status;
    nk_i32_t column_exponents[256 + 4] = {0};
    nk_i32_t column_spreads[256] = {0};
    nk_f32_t column_norms[256 + 4] = {0};
    for (nk_size_t chunk_start = rows_begin; chunk_start < vector_count && rows_begin < rows_end; chunk_start += 256) {
        nk_size_t const chunk_end = nk_min_of_two(chunk_start + 256, vector_count);
        nk_i32_t column_spread_max = 0;
        for (nk_size_t column = chunk_start; column < chunk_end; column++) {
            nk_i32_t spread;
            nk_i32_t const base = nk_mxfp8e5m2_base_sapphireamx_(scales, scales_stride, column, depth, &spread);
            column_exponents[column - chunk_start] = base;
            column_spreads[column - chunk_start] = spread;
            column_spread_max = spread > column_spread_max ? spread : column_spread_max;
        }
        for (nk_size_t row = rows_begin; row < rows_end; row++) {
            nk_size_t const column_first = chunk_start > row + 0 ? chunk_start : row + 0;
            if (column_first >= chunk_end) continue;
            nk_f32_t *result_row = (nk_f32_t *)((char *)result + row * result_stride);
            nk_i32_t row_spread;
            nk_i32_t const row_base = nk_mxfp8e5m2_base_sapphireamx_(scales, scales_stride, row, depth, &row_spread);
            nk_dots_scaled_finish_row_dot_sapphireamx_(result_row + column_first, chunk_end - column_first, 1, row_base,
                                                       column_exponents + (column_first - chunk_start), 0,
                                                       column_norms + (column_first - chunk_start));
            if (!nk_dots_scaled_exceeds_sapphireamx_(row_spread, column_spread_max, 0,
                                                     nk_dots_spread_limit_mxfp8e5m2_sapphireamx_k))
                continue;
            for (nk_size_t column = column_first; column < chunk_end; column++) {
                if (!nk_dots_scaled_exceeds_sapphireamx_(row_spread, column_spreads[column - chunk_start], 0,
                                                         nk_dots_spread_limit_mxfp8e5m2_sapphireamx_k))
                    continue;
                nk_cross_wide_sum_t row_sumsq, column_sumsq;
                nk_cross_wide_sum_t const dot = nk_cross_scaled_exact_wide_mxfp8e5m2_serial_(
                    (nk_u8_t const *)values + row * stride, (nk_u8_t const *)scales + row * scales_stride,
                    (nk_u8_t const *)values + column * stride, (nk_u8_t const *)scales + column * scales_stride, depth,
                    &row_sumsq, &column_sumsq);
                result_row[column] = nk_dot_from_wide_f32_serial_(dot, row_sumsq, column_sumsq);
            }
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_pack_size_e2m3_sapphireamx(nk_size_t column_count, nk_size_t depth, nk_size_t *bytes) {
    // E2M3 uses INT8 tile layout after conversion (same element count: 64 per row)
    *bytes = nk_dots_packed_bytes_i8_sapphireamx_(column_count, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_packed_shape_e2m3_sapphireamx(void const *b_packed, nk_size_t *columns,
                                                              nk_size_t *depth, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    if (header->capability != nk_cap_sapphireamx_k) return nk_pack_mismatch_k;
    *columns = header->columns;
    *depth = header->depth;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_pack_e2m3_sapphireamx(           //
    nk_e2m3_t const *b, nk_size_t column_count, nk_size_t depth, //
    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin, nk_size_t columns_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);

    // AMX I8 tile dimensions: 16 rows x 64 columns (1024 I8 elements = 1KB)
    nk_size_t const tmm_rows = 16;
    nk_size_t const tmm_columns = 64;
    nk_size_t const tile_elements = 1024;
    nk_size_t const tile_bytes = tile_elements * sizeof(nk_i8_t);

    nk_size_t const column_tiles_count = column_count / tmm_rows;
    nk_size_t const depth_tiles_count = nk_size_divide_round_up_(depth, tmm_columns);
    nk_size_t const column_remainder_count = column_count - column_tiles_count * tmm_rows;
    nk_size_t const total_tiles = column_tiles_count * depth_tiles_count;

    nk_dots_amx_packed_header_t *header = (nk_dots_amx_packed_header_t *)b_packed;
    if (columns_begin == 0) {
        nk_u32_t *header_words = (nk_u32_t *)header;
        for (nk_size_t word_index = 0; word_index < sizeof(*header) / sizeof(nk_u32_t); word_index++)
            header_words[word_index] = 0;
        header->tensor_scale = 1;
        header->columns = (nk_u32_t)column_count;
        header->depth = (nk_u32_t)depth;
        header->full_column_tiles = (nk_u32_t)column_tiles_count;
        header->full_depth_tiles = (nk_u32_t)depth_tiles_count;
        header->column_remainder_count = (nk_u32_t)column_remainder_count;
        header->capability = nk_cap_sapphireamx_k;
    }

    nk_size_t const tiles_offset = sizeof(nk_dots_amx_packed_header_t);
    nk_size_t const column_edge_offset = tiles_offset + total_tiles * tile_bytes;
    if (columns_begin == 0) header->column_edge_offset = (nk_u32_t)column_edge_offset;

    nk_i8_t *tiles_ptr = (nk_i8_t *)((char *)b_packed + tiles_offset);
    nk_i8_t *column_edge_ptr = (nk_i8_t *)((char *)b_packed + column_edge_offset);

    // Pack tiles using vectorized E2M3 → I8 conversion + SIMD transpose
    for (nk_size_t column_tile_idx = nk_size_divide_round_up_(columns_begin, 16);
         column_tile_idx < column_tiles_count && column_tile_idx < nk_size_divide_round_up_(columns_end, 16);
         column_tile_idx++) {
        for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
            nk_size_t const tile_index = column_tile_idx * depth_tiles_count + depth_tile_idx;
            nk_i8_t *tile_output = tiles_ptr + tile_index * tile_elements;

            nk_size_t const src_row_start = column_tile_idx * tmm_rows;
            nk_size_t const src_column_start = depth_tile_idx * tmm_columns;
            nk_size_t const columns_to_pack = (src_column_start + tmm_columns <= depth) ? tmm_columns
                                                                                        : (depth - src_column_start);

            // Convert E2M3 → I8 and gather into aligned source tile
            nk_dots_i8_a16x64_sapphireamx_t source_tile;
            if (columns_to_pack == tmm_columns) {
                for (nk_size_t row_idx = 0; row_idx < tmm_rows; row_idx++) {
                    __m512i raw_row_i8x64 = _mm512_loadu_si512(
                        (nk_e2m3_t const *)((char const *)b + (src_row_start + row_idx) * b_stride) + src_column_start);
                    _mm512_store_si512(&source_tile.data[row_idx][0], nk_e2m3x64_to_i8x64_skylake_(raw_row_i8x64));
                }
            }
            else {
                __mmask64 depth_m64 = (__mmask64)((columns_to_pack < 64) ? ((1ULL << columns_to_pack) - 1) : ~0ULL);
                for (nk_size_t row_idx = 0; row_idx < tmm_rows; row_idx++) {
                    __m512i raw_row_i8x64 = _mm512_maskz_loadu_epi8(
                        depth_m64,
                        (nk_e2m3_t const *)((char const *)b + (src_row_start + row_idx) * b_stride) + src_column_start);
                    _mm512_store_si512(&source_tile.data[row_idx][0], nk_e2m3x64_to_i8x64_skylake_(raw_row_i8x64));
                }
            }

            nk_dots_i8_b64x16_sapphireamx_t transposed_tile;
            nk_dots_pack_i8_transposed_sapphireamx_(&source_tile, &transposed_tile);
            for (nk_size_t i = 0; i < tile_elements; i += 64)
                _mm512_storeu_si512(tile_output + i, _mm512_load_si512((char const *)&transposed_tile + i));
        }
    }

    // Pack column-remainder rows, converting E2M3 to I8 64 values at a time
    if (column_remainder_count > 0 && column_tiles_count * 16 >= columns_begin &&
        column_tiles_count * 16 < columns_end) {
        nk_size_t const remainder_start_row = column_tiles_count * tmm_rows;
        for (nk_size_t row_idx = 0; row_idx < column_remainder_count; row_idx++) {
            nk_e2m3_t const *source_row = b + (remainder_start_row + row_idx) * b_stride;
            for (nk_size_t column_idx = 0; column_idx < depth; column_idx += 64) {
                nk_size_t const columns = depth - column_idx < 64 ? depth - column_idx : 64;
                nk_b512_vec_t codes_vec;
                nk_partial_load_b8x64_skylake_(source_row + column_idx, &codes_vec, columns);
                codes_vec.zmm = nk_e2m3x64_to_i8x64_skylake_(codes_vec.zmm);
                nk_partial_store_b8x64_skylake_(&codes_vec, column_edge_ptr + row_idx * depth + column_idx, columns);
            }
        }
    }

    // Compute and store per-column norms for angular/euclidean distance
    nk_size_t norms_offset = column_edge_offset +
                             (column_remainder_count > 0 ? column_remainder_count * depth * sizeof(nk_i8_t) : 0);
    if (columns_begin == 0) header->norms_byte_offset = (nk_u32_t)norms_offset;
    nk_f32_t *norms = (nk_f32_t *)((char *)b_packed + norms_offset);
    for (nk_size_t col = columns_begin; col < columns_end; col++)
        norms[col] = nk_dots_reduce_sumsq_e2m3_skylake_(b + col * b_stride, depth, sizeof(nk_e2m3_t));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_packed_e2m3_sapphireamx(   //
    nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_gemm_packed_e2m3_sapphireamx_(a, b_packed, c, rows, columns, depth, a_stride, c_stride);
}

NUMKONG_API nk_status_t nk_dots_symmetric_e2m3_sapphireamx(            //
    nk_e2m3_t const *vectors, nk_size_t vector_count, nk_size_t depth, //
    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,       //
    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_gram_e2m3_sapphireamx_(vectors, vector_count, depth, stride, result, result_stride, rows_begin, rows_end);
}

NUMKONG_API nk_status_t nk_dots_pack_size_e2m1_sapphireamx(nk_size_t column_count, nk_size_t depth, nk_size_t *bytes) {
    // E2M1 packs one decoded I8 per dimension, the same INT8 tile layout as I8
    *bytes = nk_dots_packed_bytes_i8_sapphireamx_(column_count, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_packed_shape_e2m1_sapphireamx(void const *b_packed, nk_size_t *columns,
                                                              nk_size_t *depth, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    if (header->capability != nk_cap_sapphireamx_k) return nk_pack_mismatch_k;
    *columns = header->columns;
    *depth = header->depth;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_pack_e2m1_sapphireamx(             //
    nk_e2m1x2_t const *b, nk_size_t column_count, nk_size_t depth, //
    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin, nk_size_t columns_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);

    nk_size_t const tmm_rows = 16;
    nk_size_t const tmm_columns = 64;
    nk_size_t const tile_elements = 1024;
    nk_size_t const tile_bytes = tile_elements * sizeof(nk_i8_t);

    nk_size_t const column_tiles_count = column_count / tmm_rows;
    nk_size_t const depth_tiles_count = nk_size_divide_round_up_(depth, tmm_columns);
    nk_size_t const column_remainder_count = column_count - column_tiles_count * tmm_rows;
    nk_size_t const total_tiles = column_tiles_count * depth_tiles_count;

    nk_dots_amx_packed_header_t *header = (nk_dots_amx_packed_header_t *)b_packed;
    if (columns_begin == 0) {
        nk_u32_t *header_words = (nk_u32_t *)header;
        for (nk_size_t word_index = 0; word_index < sizeof(*header) / sizeof(nk_u32_t); word_index++)
            header_words[word_index] = 0;
        header->tensor_scale = 1;
        header->columns = (nk_u32_t)column_count;
        header->depth = (nk_u32_t)depth;
        header->full_column_tiles = (nk_u32_t)column_tiles_count;
        header->full_depth_tiles = (nk_u32_t)depth_tiles_count;
        header->column_remainder_count = (nk_u32_t)column_remainder_count;
        header->capability = nk_cap_sapphireamx_k;
    }

    nk_size_t const tiles_offset = sizeof(nk_dots_amx_packed_header_t);
    nk_size_t const column_edge_offset = tiles_offset + total_tiles * tile_bytes;
    if (columns_begin == 0) header->column_edge_offset = (nk_u32_t)column_edge_offset;

    nk_i8_t *tiles_ptr = (nk_i8_t *)((char *)b_packed + tiles_offset);
    nk_i8_t *column_edge_ptr = (nk_i8_t *)((char *)b_packed + column_edge_offset);

    // Pack tiles decoding E2M1 nibbles into I8 rows, then transposing
    for (nk_size_t column_tile_idx = nk_size_divide_round_up_(columns_begin, 16);
         column_tile_idx < column_tiles_count && column_tile_idx < nk_size_divide_round_up_(columns_end, 16);
         column_tile_idx++) {
        for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
            nk_size_t const tile_index = column_tile_idx * depth_tiles_count + depth_tile_idx;
            nk_i8_t *tile_output = tiles_ptr + tile_index * tile_elements;

            nk_size_t const src_row_start = column_tile_idx * tmm_rows;
            nk_size_t const src_column_start = depth_tile_idx * tmm_columns;
            nk_size_t const columns_to_pack = (src_column_start + tmm_columns <= depth) ? tmm_columns
                                                                                        : (depth - src_column_start);

            nk_dots_i8_a16x64_sapphireamx_t source_tile;
            nk_dots_e2m1_load_a_sapphireamx_(
                &source_tile, (nk_e2m1x2_t const *)((char const *)b + src_row_start * b_stride) + src_column_start / 2,
                b_stride, tmm_rows, columns_to_pack);

            nk_dots_i8_b64x16_sapphireamx_t transposed_tile;
            nk_dots_pack_i8_transposed_sapphireamx_(&source_tile, &transposed_tile);
            for (nk_size_t i = 0; i < tile_elements; i += 64)
                _mm512_storeu_si512(tile_output + i, _mm512_load_si512((char const *)&transposed_tile + i));
        }
    }

    // Pack column-remainder rows, decoding each nibble to its doubled I8 value
    if (column_remainder_count > 0 && column_tiles_count * 16 >= columns_begin &&
        column_tiles_count * 16 < columns_end) {
        nk_size_t const remainder_start_row = column_tiles_count * tmm_rows;
        for (nk_size_t row_idx = 0; row_idx < column_remainder_count; row_idx++) {
            nk_u8_t const *source_row = (nk_u8_t const *)b + (remainder_start_row + row_idx) * b_stride;
            for (nk_size_t column_idx = 0; column_idx < depth; column_idx++) {
                nk_u8_t raw = source_row[column_idx / 2];
                column_edge_ptr[row_idx * depth + column_idx] = nk_e2m1_nibble_to_i8x2_serial_(
                    (column_idx & 1) ? (raw & 0x0F) : (raw >> 4));
            }
        }
    }

    nk_size_t norms_offset = column_edge_offset +
                             (column_remainder_count > 0 ? column_remainder_count * depth * sizeof(nk_i8_t) : 0);
    if (columns_begin == 0) header->norms_byte_offset = (nk_u32_t)norms_offset;
    nk_f32_t *norms = (nk_f32_t *)((char *)b_packed + norms_offset);
    // Compute and store per-column norms for angular/euclidean distance
    for (nk_size_t col = columns_begin; col < columns_end; col++)
        norms[col] = nk_dots_reduce_sumsq_e2m1_serial_((nk_e2m1x2_t const *)((char const *)b + col * b_stride), depth,
                                                       sizeof(nk_e2m1x2_t));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_packed_e2m1_sapphireamx(     //
    nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_gemm_packed_e2m1_sapphireamx_(a, b_packed, c, rows, columns, depth, a_stride, c_stride);
}

NUMKONG_API nk_status_t nk_dots_symmetric_e2m1_sapphireamx(              //
    nk_e2m1x2_t const *vectors, nk_size_t vector_count, nk_size_t depth, //
    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,         //
    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_gram_e2m1_sapphireamx_(vectors, vector_count, depth, stride, result, result_stride, rows_begin, rows_end);
}

NUMKONG_API nk_status_t nk_dots_pack_size_e3m2_sapphireamx(nk_size_t column_count, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_dots_packed_bytes_bf16_sapphireamx_(column_count, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_packed_shape_e3m2_sapphireamx(void const *b_packed, nk_size_t *columns,
                                                              nk_size_t *depth, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    if (header->capability != nk_cap_sapphireamx_k) return nk_pack_mismatch_k;
    *columns = header->columns;
    *depth = header->depth;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_pack_e3m2_sapphireamx(           //
    nk_e3m2_t const *b, nk_size_t column_count, nk_size_t depth, //
    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin, nk_size_t columns_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);

    nk_size_t const tmm_rows = 16;
    nk_size_t const tmm_columns = 32;
    nk_size_t const tile_elements = 512;
    nk_size_t const tile_bytes = tile_elements * sizeof(nk_bf16_t);

    nk_size_t const column_tiles_count = column_count / tmm_rows;
    nk_size_t const depth_tiles_count = nk_size_divide_round_up_(depth, tmm_columns);
    nk_size_t const column_remainder_count = column_count - column_tiles_count * tmm_rows;
    nk_size_t const total_tiles = column_tiles_count * depth_tiles_count;

    nk_dots_amx_packed_header_t *header = (nk_dots_amx_packed_header_t *)b_packed;
    if (columns_begin == 0) {
        nk_u32_t *header_words = (nk_u32_t *)header;
        for (nk_size_t word_index = 0; word_index < sizeof(*header) / sizeof(nk_u32_t); word_index++)
            header_words[word_index] = 0;
        header->tensor_scale = 1;
        header->columns = (nk_u32_t)column_count;
        header->depth = (nk_u32_t)depth;
        header->full_column_tiles = (nk_u32_t)column_tiles_count;
        header->full_depth_tiles = (nk_u32_t)depth_tiles_count;
        header->column_remainder_count = (nk_u32_t)column_remainder_count;
        header->capability = nk_cap_sapphireamx_k;
    }

    nk_size_t const tiles_offset = sizeof(nk_dots_amx_packed_header_t);
    nk_size_t const column_edge_offset = tiles_offset + total_tiles * tile_bytes;
    if (columns_begin == 0) header->column_edge_offset = (nk_u32_t)column_edge_offset;

    nk_bf16_t *tiles_ptr = (nk_bf16_t *)((char *)b_packed + tiles_offset);
    nk_bf16_t *column_edge_ptr = (nk_bf16_t *)((char *)b_packed + column_edge_offset);

    // Pack tiles using vectorized convert + SIMD transpose
    for (nk_size_t column_tile_idx = nk_size_divide_round_up_(columns_begin, 16);
         column_tile_idx < column_tiles_count && column_tile_idx < nk_size_divide_round_up_(columns_end, 16);
         column_tile_idx++) {
        for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
            nk_size_t const tile_index = column_tile_idx * depth_tiles_count + depth_tile_idx;
            nk_bf16_t *tile_output = tiles_ptr + tile_index * tile_elements;

            nk_size_t const src_row_start = column_tile_idx * tmm_rows;
            nk_size_t const src_column_start = depth_tile_idx * tmm_columns;
            nk_size_t const columns_to_pack = (src_column_start + tmm_columns <= depth) ? tmm_columns
                                                                                        : (depth - src_column_start);

            __mmask32 column_m32 = (columns_to_pack >= 32) ? 0xFFFFFFFF : ((__mmask32)1 << columns_to_pack) - 1;
            nk_dots_bf16_a16x32_sapphireamx_t source_tile;
            for (nk_size_t row_idx = 0; row_idx < tmm_rows; row_idx++) {
                __m256i e3m2_row_u8x32 = _mm256_maskz_loadu_epi8(
                    column_m32, b + (src_row_start + row_idx) * b_stride + src_column_start);
                _mm512_store_si512(&source_tile.data[row_idx][0], nk_e3m2x32_to_bf16x32_icelake_(e3m2_row_u8x32));
            }

            nk_dots_bf16_b32x16_sapphireamx_t transposed_tile;
            nk_dots_pack_bf16_transposed_sapphireamx_(&source_tile, &transposed_tile);
            for (nk_size_t i = 0; i < tile_bytes; i += 64)
                _mm512_storeu_si512((char *)tile_output + i, _mm512_load_si512((char const *)&transposed_tile + i));
        }
    }

    if (column_remainder_count > 0 && column_tiles_count * 16 >= columns_begin &&
        column_tiles_count * 16 < columns_end) {
        nk_size_t const remainder_start_row = column_tiles_count * tmm_rows;
        for (nk_size_t row_idx = 0; row_idx < column_remainder_count; row_idx++) {
            for (nk_size_t column_idx = 0; column_idx < depth; column_idx += 32) {
                nk_size_t columns = (column_idx + 32 <= depth) ? 32 : (depth - column_idx);
                __mmask32 column_m32 = (columns >= 32) ? 0xFFFFFFFF : ((__mmask32)1 << columns) - 1;
                __m256i e3m2_chunk_u8x32 = _mm256_maskz_loadu_epi8(
                    column_m32, b + (remainder_start_row + row_idx) * b_stride + column_idx);
                __m512i bf16_chunk_i16x32 = nk_e3m2x32_to_bf16x32_icelake_(e3m2_chunk_u8x32);
                _mm512_mask_storeu_epi16(column_edge_ptr + row_idx * depth + column_idx, column_m32, bf16_chunk_i16x32);
            }
        }
    }

    // Compute and store per-column norms for angular/euclidean distance
    nk_size_t norms_offset = column_edge_offset +
                             (column_remainder_count > 0 ? column_remainder_count * depth * sizeof(nk_bf16_t) : 0);
    if (columns_begin == 0) header->norms_byte_offset = (nk_u32_t)norms_offset;
    nk_f32_t *norms = (nk_f32_t *)((char *)b_packed + norms_offset);
    for (nk_size_t col = columns_begin; col < columns_end; col++)
        norms[col] = nk_dots_reduce_sumsq_e3m2_skylake_(b + col * b_stride, depth, sizeof(nk_e3m2_t));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_dots_packed_e3m2_sapphireamx(   //
    nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_gemm_packed_e3m2_sapphireamx_(a, b_packed, c, rows, columns, depth, a_stride, c_stride);
}

NUMKONG_API nk_status_t nk_dots_symmetric_e3m2_sapphireamx(            //
    nk_e3m2_t const *vectors, nk_size_t vector_count, nk_size_t depth, //
    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,       //
    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_gram_e3m2_sapphireamx_(vectors, vector_count, depth, stride, result, result_stride, rows_begin, rows_end);
}

#endif // NUMKONG_TARGET_SAPPHIREAMX

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_X8664_SAPPHIREAMX_
#endif // NUMKONG_ARCH_X8664_
#endif // NUMKONG_DOTS_SAPPHIREAMX_H
