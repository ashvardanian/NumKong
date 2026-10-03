/**
 *  @brief SIMD-accelerated Batched Dot Products for Sapphire Rapids.
 *  @file include/numkong/dots/sapphireamx.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/dots.h
 *
 *  This file contains tiled matrix-multiplication kernels optimized for Intel AMX instructions,
 *  leveraging the new TMM registers on Intel Sapphire Rapids CPUs. Those are much larger than ZMM:
 *
 *  - BF16 tiles: 16 rows × 32 elements = 512 BF16 values = 1KB per tile
 *  - INT8 tiles: 16 rows × 64 elements = 1024 INT8 values = 1KB per tile
 *
 *  We typically use 4 registers for the 2 × 2 tile output for the matrix C accumulators, leaving
 *  4 other registers for parts of A and B matrices:
 *
 *  - TMM0, TMM1: A matrix tiles (row blocks i and i+16)
 *  - TMM2, TMM3: B matrix tiles (column blocks j and j+16)
 *  - TMM4-7: C accumulator tiles (2 × 2 output grid)
 *
 *  In most synthetic benchmarks there seems to be no mahor difference between aggregating into 1 or 4
 *  output tiles, implying the CPU's ability to internally pipeline the accumulation; so using 2 × 2 for
 *  ouputs is more of memory-bandwidth saving measure.
 *
 *  Lacking High Bandwidth Mememory, the performance in GEMM-like BLAS workloads is dominated by memory
 *  bandwidth. Latency hiding is also extremely hard, heavily affecting performance numbers. For reference,
 *  Intel MKL SGEMM for FP32 inputs yeilds arounf 250 GigaOPS per core on Intel Sapphire Rapids, leveraging
 *  AVX-512. At the same time, for AMX:
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
 *      Intrinsic                   Instruction                     Notes
 *      _tile_loadconfig            LDTILECFG (mem64)               Configure tile palette
 *      _tile_loadd                 TILELOADD (TMM, mem, stride)    Load tile from memory
 *      _tile_stored                TILESTORED (mem, TMM, stride)   Store tile to memory
 *      _tile_zero                  TILEZERO (TMM)                  Zero a tile register
 *
 *  BF16 matrix multiply (AMX-BF16):
 *
 *      Intrinsic                   Instruction                     Operation
 *      _tile_dpbf16ps              TDPBF16PS (TMM, TMM, TMM)       C += A × B (bf16 → f32)
 *
 *  INT8 matrix multiply (AMX-INT8):
 *
 *      Intrinsic                   Instruction                     Operation
 *      _tile_dpbssd                TDPBSSD (TMM, TMM, TMM)         C += A × B (i8 × i8 → i32)
 *      _tile_dpbsud                TDPBSUD (TMM, TMM, TMM)         C += A × B (i8 × u8 → i32)
 *      _tile_dpbusd                TDPBUSD (TMM, TMM, TMM)         C += A × B (u8 × i8 → i32)
 *      _tile_dpbuud                TDPBUUD (TMM, TMM, TMM)         C += A × B (u8 × u8 → u32)
 *
 *  AMX performance characteristics:
 *  - TDPBF16PS: 16 × 16 × 32 = 8192 BF16 MACs per instruction
 *  - TDPBSSD: 16 × 16 × 64 = 16384 INT8 MACs per instruction
 *  - Tile load latency is ~20-30 cycles; software pipelining essential
 *  - PDEP/PEXT used for Morton Z-curve encoding (BMI2): 2-3cy @ p1
 */
#ifndef NK_DOTS_SAPPHIREAMX_H
#define NK_DOTS_SAPPHIREAMX_H

#if NK_TARGET_X86_
#if NK_TARGET_SAPPHIREAMX

#include "numkong/cast/icelake.h" // For FP8 ↔ BF16 conversions
#include "numkong/dots/serial.h"  // For nk_dots_reduce_sumsq_bf16_

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
 *  Different from nk_dots_amx_packed_header_t as AMX uses tile-based layout.
 */
typedef struct {
    nk_u32_t full_column_tiles;      // Number of full column tiles (16 rows each)
    nk_u32_t full_depth_tiles;       // Number of depth tiles (32 columns for BF16, 64 for I8)
    nk_u32_t column_remainder_count; // Remaining rows after full tiles (0-15)
    nk_u32_t column_edge_offset;     // Byte offset to edge data region
    nk_u32_t norms_byte_offset;      // Byte offset to per-column norms (for angular/euclidean)
    nk_u32_t reserved[11];           // Padding to 64 bytes
} nk_dots_amx_packed_header_t;

/*  Composable tile structures for AMX operations.
 *  These enable reusable primitives and cross-correlation (A × Aᵀ) use cases.
 */

/*  BF16 A tile: 16 rows × 32 depth-elements, row-major layout.
 *  Loaded from source matrix, used as left operand in AMX multiply.
 */
typedef struct {
    NK_ALIGN64 nk_bf16_t data[16][32]; // 16 rows × 32 columns = 1KB
} nk_dots_bf16_a16x32_sapphireamx_t;

/*  BF16 B tile: 32 depth × 16 columns, pair-interleaved for TDPBF16PS.
 *  Access pattern: data[depth/2][column][depth%2] for logical B[depth, column].
 *  Pre-packed from column-major or transposed source.
 */
typedef struct {
    NK_ALIGN64 nk_bf16_t data[16][16][2]; // 16 depth-groups × 16 columns × 2 = 1KB
} nk_dots_bf16_b32x16_sapphireamx_t;

/*  BF16 output state: 16 × 16 F32 accumulator tile.
 *  Holds partial sums during depth-dimension accumulation.
 */
typedef struct {
    NK_ALIGN64 nk_f32_t data[16][16]; // 16 × 16 = 1KB
} nk_dots_bf16_state_sapphireamx_t;

/*  INT8 A tile: 16 rows × 64 depth-elements, row-major layout.
 */
typedef struct {
    NK_ALIGN64 nk_i8_t data[16][64]; // 16 rows × 64 columns = 1KB
} nk_dots_i8_a16x64_sapphireamx_t;

/*  INT8 B tile: 64 depth × 16 columns, quad-interleaved for TDPBSSD.
 *  Access pattern: data[depth/4][column][depth%4] for logical B[depth, column].
 */
typedef struct {
    NK_ALIGN64 nk_i8_t data[16][16][4]; // 16 depth-groups × 16 columns × 4 = 1KB
} nk_dots_i8_b64x16_sapphireamx_t;

/*  INT8 output state: 16 × 16 I32 accumulator tile.
 */
typedef struct {
    NK_ALIGN64 nk_i32_t data[16][16]; // 16 × 16 = 1KB
} nk_dots_i8_state_sapphireamx_t;

/*  BF16 2 × 2 output state: 32 × 32 F32 output (4 accumulator tiles).
 *  Used for GEMM's 2 × 2 output blocking pattern.
 */
typedef struct {
    nk_dots_bf16_state_sapphireamx_t c[2][2]; // 4KB total
} nk_dots_bf16_state2x2_sapphireamx_t;

/*  INT8 2 × 2 output state: 32 × 32 I32 output (4 accumulator tiles).
 */
typedef struct {
    nk_dots_i8_state_sapphireamx_t c[2][2]; // 4KB total
} nk_dots_i8_state2x2_sapphireamx_t;

/*  UINT8 A tile: 16 rows × 64 depth-elements, row-major layout.
 *  Same layout as I8, different interpretation of signed vs unsigned.
 */
typedef struct {
    NK_ALIGN64 nk_u8_t data[16][64]; // 16 rows × 64 columns = 1KB
} nk_dots_u8_a16x64_sapphireamx_t;

/*  UINT8 B tile: 64 depth × 16 columns, quad-interleaved for TDPBUUD.
 */
typedef struct {
    NK_ALIGN64 nk_u8_t data[16][16][4]; // 16 depth-groups × 16 columns × 4 = 1KB
} nk_dots_u8_b64x16_sapphireamx_t;

/*  UINT8 output state: 16 × 16 U32 accumulator tile.
 */
typedef struct {
    NK_ALIGN64 nk_u32_t data[16][16]; // 16 × 16 = 1KB
} nk_dots_u8_state_sapphireamx_t;

/*  UINT8 2 × 2 output state: 32 × 32 U32 output (4 accumulator tiles).
 */
typedef struct {
    nk_dots_u8_state_sapphireamx_t c[2][2]; // 4KB total
} nk_dots_u8_state2x2_sapphireamx_t;

/* Morton Z-curve encoding for cache-friendly tile traversal */
NK_INTERNAL nk_u64_t nk_morton_encode_sapphireamx_(nk_u32_t tile_row, nk_u32_t tile_col) {
    return _pdep_u64(tile_row, 0x5555555555555555ULL) | _pdep_u64(tile_col, 0xAAAAAAAAAAAAAAAAULL);
}

/* Configure AMX tile registers */
NK_INTERNAL void nk_amx_tile_configure_sapphireamx_(void) {
    NK_ALIGN64 nk_u8_t tile_config[64] = {0};
    tile_config[0] = 1; // palette 1 (standard tile configuration)

    nk_u16_t *bytes_per_row = (nk_u16_t *)&tile_config[16];
    nk_u8_t *rows_per_tile = &tile_config[48];

    for (int tile_idx = 0; tile_idx < 8; tile_idx++) {
        rows_per_tile[tile_idx] = 16; // 16 rows per tile
        bytes_per_row[tile_idx] = 64; // 64 bytes per row (1KB total)
    }
    _tile_loadconfig(tile_config);
}

/** @brief Compiler memory barrier to ensure stores complete before AMX tile loads */
#if defined(_MSC_VER)
NK_INTERNAL void nk_compiler_barrier_sapphireamx_(void) { _ReadWriteBarrier(); }
#else
NK_INTERNAL void nk_compiler_barrier_sapphireamx_(void) { __asm__ volatile("" ::: "memory"); }
#endif

/* Initialize BF16 output state to zero */
NK_INTERNAL void nk_dots_bf16_init_sapphireamx_(nk_dots_bf16_state_sapphireamx_t *state) {
    __m512 zero = _mm512_setzero_ps();
    for (nk_size_t row_idx = 0; row_idx < 16; row_idx++) { _mm512_store_ps(state->data[row_idx], zero); }
}

/* Load A tile from row-major source with masking for edge tiles */
NK_INTERNAL void nk_dots_bf16_load_a_sapphireamx_(       //
    nk_dots_bf16_a16x32_sapphireamx_t *a_tile,           //
    nk_bf16_t const *src, nk_size_t src_stride_elements, //
    nk_size_t valid_rows, nk_size_t valid_cols) {

    __mmask32 column_mask = (valid_cols >= 32) ? 0xFFFFFFFF : ((__mmask32)1 << valid_cols) - 1;
    __m512i zero = _mm512_setzero_si512();

    for (nk_size_t row_idx = 0; row_idx < 16; row_idx++) {
        if (row_idx < valid_rows) {
            __m512i row = _mm512_maskz_loadu_epi16(column_mask, src + row_idx * src_stride_elements);
            _mm512_store_si512((__m512i *)a_tile->data[row_idx], row);
        }
        else { _mm512_store_si512((__m512i *)a_tile->data[row_idx], zero); }
    }
    nk_compiler_barrier_sapphireamx_();
}

/* Store state to output matrix with masking for edge tiles */
NK_INTERNAL void nk_dots_bf16_store_sapphireamx_(  //
    nk_dots_bf16_state_sapphireamx_t const *state, //
    nk_f32_t *dst, nk_size_t dst_stride_elements,  //
    nk_size_t valid_rows, nk_size_t valid_cols) {

    __mmask16 column_mask = (valid_cols >= 16) ? 0xFFFF : ((__mmask16)1 << valid_cols) - 1;

    for (nk_size_t row_idx = 0; row_idx < valid_rows; row_idx++) {
        __m512 row = _mm512_load_ps(state->data[row_idx]);
        _mm512_mask_storeu_ps(dst + row_idx * dst_stride_elements, column_mask, row);
    }
}

/* Accumulate 3 A x B tile pairs into state using AMX TDPBF16PS */
NK_INTERNAL void nk_dots_bf16_update_sapphireamx_(     //
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
NK_INTERNAL void nk_dots_i8_init_sapphireamx_(nk_dots_i8_state_sapphireamx_t *state) {
    __m512i zero = _mm512_setzero_si512();
    for (nk_size_t row_idx = 0; row_idx < 16; row_idx++) { _mm512_store_si512((__m512i *)state->data[row_idx], zero); }
}

/* Load A tile from row-major source with masking for edge tiles */
NK_INTERNAL void nk_dots_i8_load_a_sapphireamx_( //
    nk_dots_i8_a16x64_sapphireamx_t *a_tile,     //
    nk_i8_t const *src, nk_size_t src_stride,    //
    nk_size_t valid_rows, nk_size_t valid_cols) {

    __mmask64 column_mask = (valid_cols >= 64) ? 0xFFFFFFFFFFFFFFFFULL : ((__mmask64)1 << valid_cols) - 1;
    __m512i zero = _mm512_setzero_si512();

    for (nk_size_t row_idx = 0; row_idx < 16; row_idx++) {
        if (row_idx < valid_rows) {
            __m512i row = _mm512_maskz_loadu_epi8(column_mask, src + row_idx * src_stride);
            _mm512_store_si512((__m512i *)a_tile->data[row_idx], row);
        }
        else { _mm512_store_si512((__m512i *)a_tile->data[row_idx], zero); }
    }
    nk_compiler_barrier_sapphireamx_();
}

/* Store state to output matrix with masking for edge tiles */
NK_INTERNAL void nk_dots_i8_store_sapphireamx_(   //
    nk_dots_i8_state_sapphireamx_t const *state,  //
    nk_i32_t *dst, nk_size_t dst_stride_elements, //
    nk_size_t valid_rows, nk_size_t valid_cols) {

    __mmask16 column_mask = (valid_cols >= 16) ? 0xFFFF : ((__mmask16)1 << valid_cols) - 1;

    for (nk_size_t row_idx = 0; row_idx < valid_rows; row_idx++) {
        __m512i row = _mm512_load_si512((__m512i const *)state->data[row_idx]);
        _mm512_mask_storeu_epi32(dst + row_idx * dst_stride_elements, column_mask, row);
    }
}

/* Accumulate 3 A x B tile pairs into state using AMX TDPBSSD */
NK_INTERNAL void nk_dots_i8_update_sapphireamx_(     //
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
NK_INTERNAL void nk_dots_bf16_output2x2_sapphireamx_( //
    nk_dots_bf16_state2x2_sapphireamx_t const *state, //
    nk_f32_t *dst, nk_size_t dst_stride_elements,     //
    nk_size_t valid_rows, nk_size_t valid_cols) {

    // Rows 0-15
    nk_size_t const rows_upper = (valid_rows > 16) ? 16 : valid_rows;
    nk_size_t const cols_left = (valid_cols > 16) ? 16 : valid_cols;
    nk_size_t const cols_right = (valid_cols > 16) ? valid_cols - 16 : 0;

    if (rows_upper > 0 && cols_left > 0)
        nk_dots_bf16_store_sapphireamx_(&state->c[0][0], dst, dst_stride_elements, rows_upper, cols_left);
    if (rows_upper > 0 && cols_right > 0)
        nk_dots_bf16_store_sapphireamx_(&state->c[0][1], dst + 16, dst_stride_elements, rows_upper, cols_right);

    // Rows 16-31
    if (valid_rows > 16) {
        nk_size_t const rows_lower = valid_rows - 16;
        nk_f32_t *dst_lower = dst + 16 * dst_stride_elements;
        if (cols_left > 0)
            nk_dots_bf16_store_sapphireamx_(&state->c[1][0], dst_lower, dst_stride_elements, rows_lower, cols_left);
        if (cols_right > 0)
            nk_dots_bf16_store_sapphireamx_(&state->c[1][1], dst_lower + 16, dst_stride_elements, rows_lower,
                                            cols_right);
    }
}

/* Store INT8 2x2 state to output matrix with masking for edge tiles */
NK_INTERNAL void nk_dots_i8_output2x2_sapphireamx_( //
    nk_dots_i8_state2x2_sapphireamx_t const *state, //
    nk_i32_t *dst, nk_size_t dst_stride_elements,   //
    nk_size_t valid_rows, nk_size_t valid_cols) {

    nk_size_t const rows_upper = (valid_rows > 16) ? 16 : valid_rows;
    nk_size_t const cols_left = (valid_cols > 16) ? 16 : valid_cols;
    nk_size_t const cols_right = (valid_cols > 16) ? valid_cols - 16 : 0;

    if (rows_upper > 0 && cols_left > 0)
        nk_dots_i8_store_sapphireamx_(&state->c[0][0], dst, dst_stride_elements, rows_upper, cols_left);
    if (rows_upper > 0 && cols_right > 0)
        nk_dots_i8_store_sapphireamx_(&state->c[0][1], dst + 16, dst_stride_elements, rows_upper, cols_right);

    if (valid_rows > 16) {
        nk_size_t const rows_lower = valid_rows - 16;
        nk_i32_t *dst_lower = dst + 16 * dst_stride_elements;
        if (cols_left > 0)
            nk_dots_i8_store_sapphireamx_(&state->c[1][0], dst_lower, dst_stride_elements, rows_lower, cols_left);
        if (cols_right > 0)
            nk_dots_i8_store_sapphireamx_(&state->c[1][1], dst_lower + 16, dst_stride_elements, rows_lower, cols_right);
    }
}

/* Initialize UINT8 output state to zero */
NK_INTERNAL void nk_dots_u8_init_sapphireamx_(nk_dots_u8_state_sapphireamx_t *state) {
    nk_dots_i8_init_sapphireamx_((nk_dots_i8_state_sapphireamx_t *)state);
}

/* Load U8 A tile from row-major source with masking for edge tiles */
NK_INTERNAL void nk_dots_u8_load_a_sapphireamx_( //
    nk_dots_u8_a16x64_sapphireamx_t *a_tile,     //
    nk_u8_t const *src, nk_size_t src_stride,    //
    nk_size_t valid_rows, nk_size_t valid_cols) {
    nk_dots_i8_load_a_sapphireamx_(                //
        (nk_dots_i8_a16x64_sapphireamx_t *)a_tile, //
        (nk_i8_t const *)src, src_stride, valid_rows, valid_cols);
}

/* Store U8 state to output matrix with masking for edge tiles */
NK_INTERNAL void nk_dots_u8_store_sapphireamx_(   //
    nk_dots_u8_state_sapphireamx_t const *state,  //
    nk_u32_t *dst, nk_size_t dst_stride_elements, //
    nk_size_t valid_rows, nk_size_t valid_cols) {
    nk_dots_i8_store_sapphireamx_(                     //
        (nk_dots_i8_state_sapphireamx_t const *)state, //
        (nk_i32_t *)dst, dst_stride_elements, valid_rows, valid_cols);
}

/* Store UINT8 2x2 state to output matrix with masking for edge tiles */
NK_INTERNAL void nk_dots_u8_output2x2_sapphireamx_( //
    nk_dots_u8_state2x2_sapphireamx_t const *state, //
    nk_u32_t *dst, nk_size_t dst_stride_elements,   //
    nk_size_t valid_rows, nk_size_t valid_cols) {
    nk_dots_i8_output2x2_sapphireamx_(                    //
        (nk_dots_i8_state2x2_sapphireamx_t const *)state, //
        (nk_i32_t *)dst, dst_stride_elements, valid_rows, valid_cols);
}

/* Pack U8 A transposed into B format */
NK_INTERNAL void nk_dots_pack_u8_transposed_sapphireamx_( //
    nk_dots_u8_a16x64_sapphireamx_t const *a_tile,        //
    nk_dots_u8_b64x16_sapphireamx_t *b_tile) {

    // Load all 16 rows - each row is 64 UINT8 = 64 bytes = 1 ZMM
    // Treat as 16 × 32-bit elements per row (each 32-bit = quad of UINT8)
    __m512i row00 = _mm512_load_si512(&a_tile->data[0][0]);
    __m512i row01 = _mm512_load_si512(&a_tile->data[1][0]);
    __m512i row02 = _mm512_load_si512(&a_tile->data[2][0]);
    __m512i row03 = _mm512_load_si512(&a_tile->data[3][0]);
    __m512i row04 = _mm512_load_si512(&a_tile->data[4][0]);
    __m512i row05 = _mm512_load_si512(&a_tile->data[5][0]);
    __m512i row06 = _mm512_load_si512(&a_tile->data[6][0]);
    __m512i row07 = _mm512_load_si512(&a_tile->data[7][0]);
    __m512i row08 = _mm512_load_si512(&a_tile->data[8][0]);
    __m512i row09 = _mm512_load_si512(&a_tile->data[9][0]);
    __m512i row10 = _mm512_load_si512(&a_tile->data[10][0]);
    __m512i row11 = _mm512_load_si512(&a_tile->data[11][0]);
    __m512i row12 = _mm512_load_si512(&a_tile->data[12][0]);
    __m512i row13 = _mm512_load_si512(&a_tile->data[13][0]);
    __m512i row14 = _mm512_load_si512(&a_tile->data[14][0]);
    __m512i row15 = _mm512_load_si512(&a_tile->data[15][0]);

    // 16×16 transpose of 32-bit elements using hierarchical unpacks
    // Stage 1: Unpack adjacent row pairs at 32-bit granularity
    __m512i t01_lo = _mm512_unpacklo_epi32(row00, row01);
    __m512i t01_hi = _mm512_unpackhi_epi32(row00, row01);
    __m512i t23_lo = _mm512_unpacklo_epi32(row02, row03);
    __m512i t23_hi = _mm512_unpackhi_epi32(row02, row03);
    __m512i t45_lo = _mm512_unpacklo_epi32(row04, row05);
    __m512i t45_hi = _mm512_unpackhi_epi32(row04, row05);
    __m512i t67_lo = _mm512_unpacklo_epi32(row06, row07);
    __m512i t67_hi = _mm512_unpackhi_epi32(row06, row07);
    __m512i t89_lo = _mm512_unpacklo_epi32(row08, row09);
    __m512i t89_hi = _mm512_unpackhi_epi32(row08, row09);
    __m512i tab_lo = _mm512_unpacklo_epi32(row10, row11);
    __m512i tab_hi = _mm512_unpackhi_epi32(row10, row11);
    __m512i tcd_lo = _mm512_unpacklo_epi32(row12, row13);
    __m512i tcd_hi = _mm512_unpackhi_epi32(row12, row13);
    __m512i tef_lo = _mm512_unpacklo_epi32(row14, row15);
    __m512i tef_hi = _mm512_unpackhi_epi32(row14, row15);

    // Stage 2: Unpack at 64-bit granularity
    __m512i u0123_ll = _mm512_unpacklo_epi64(t01_lo, t23_lo);
    __m512i u0123_lh = _mm512_unpackhi_epi64(t01_lo, t23_lo);
    __m512i u0123_hl = _mm512_unpacklo_epi64(t01_hi, t23_hi);
    __m512i u0123_hh = _mm512_unpackhi_epi64(t01_hi, t23_hi);
    __m512i u4567_ll = _mm512_unpacklo_epi64(t45_lo, t67_lo);
    __m512i u4567_lh = _mm512_unpackhi_epi64(t45_lo, t67_lo);
    __m512i u4567_hl = _mm512_unpacklo_epi64(t45_hi, t67_hi);
    __m512i u4567_hh = _mm512_unpackhi_epi64(t45_hi, t67_hi);
    __m512i u89ab_ll = _mm512_unpacklo_epi64(t89_lo, tab_lo);
    __m512i u89ab_lh = _mm512_unpackhi_epi64(t89_lo, tab_lo);
    __m512i u89ab_hl = _mm512_unpacklo_epi64(t89_hi, tab_hi);
    __m512i u89ab_hh = _mm512_unpackhi_epi64(t89_hi, tab_hi);
    __m512i ucdef_ll = _mm512_unpacklo_epi64(tcd_lo, tef_lo);
    __m512i ucdef_lh = _mm512_unpackhi_epi64(tcd_lo, tef_lo);
    __m512i ucdef_hl = _mm512_unpacklo_epi64(tcd_hi, tef_hi);
    __m512i ucdef_hh = _mm512_unpackhi_epi64(tcd_hi, tef_hi);

    // Stage 3: Shuffle 128-bit lanes
    __m512i v0_a = _mm512_shuffle_i32x4(u0123_ll, u4567_ll, 0x88);
    __m512i v0_b = _mm512_shuffle_i32x4(u0123_ll, u4567_ll, 0xDD);
    __m512i v1_a = _mm512_shuffle_i32x4(u0123_lh, u4567_lh, 0x88);
    __m512i v1_b = _mm512_shuffle_i32x4(u0123_lh, u4567_lh, 0xDD);
    __m512i v2_a = _mm512_shuffle_i32x4(u0123_hl, u4567_hl, 0x88);
    __m512i v2_b = _mm512_shuffle_i32x4(u0123_hl, u4567_hl, 0xDD);
    __m512i v3_a = _mm512_shuffle_i32x4(u0123_hh, u4567_hh, 0x88);
    __m512i v3_b = _mm512_shuffle_i32x4(u0123_hh, u4567_hh, 0xDD);
    __m512i v4_a = _mm512_shuffle_i32x4(u89ab_ll, ucdef_ll, 0x88);
    __m512i v4_b = _mm512_shuffle_i32x4(u89ab_ll, ucdef_ll, 0xDD);
    __m512i v5_a = _mm512_shuffle_i32x4(u89ab_lh, ucdef_lh, 0x88);
    __m512i v5_b = _mm512_shuffle_i32x4(u89ab_lh, ucdef_lh, 0xDD);
    __m512i v6_a = _mm512_shuffle_i32x4(u89ab_hl, ucdef_hl, 0x88);
    __m512i v6_b = _mm512_shuffle_i32x4(u89ab_hl, ucdef_hl, 0xDD);
    __m512i v7_a = _mm512_shuffle_i32x4(u89ab_hh, ucdef_hh, 0x88);
    __m512i v7_b = _mm512_shuffle_i32x4(u89ab_hh, ucdef_hh, 0xDD);

    // Stage 4: Final 256-bit shuffle to complete transpose
    __m512i out00 = _mm512_shuffle_i32x4(v0_a, v4_a, 0x88);
    __m512i out01 = _mm512_shuffle_i32x4(v1_a, v5_a, 0x88);
    __m512i out02 = _mm512_shuffle_i32x4(v2_a, v6_a, 0x88);
    __m512i out03 = _mm512_shuffle_i32x4(v3_a, v7_a, 0x88);
    __m512i out04 = _mm512_shuffle_i32x4(v0_a, v4_a, 0xDD);
    __m512i out05 = _mm512_shuffle_i32x4(v1_a, v5_a, 0xDD);
    __m512i out06 = _mm512_shuffle_i32x4(v2_a, v6_a, 0xDD);
    __m512i out07 = _mm512_shuffle_i32x4(v3_a, v7_a, 0xDD);
    __m512i out08 = _mm512_shuffle_i32x4(v0_b, v4_b, 0x88);
    __m512i out09 = _mm512_shuffle_i32x4(v1_b, v5_b, 0x88);
    __m512i out10 = _mm512_shuffle_i32x4(v2_b, v6_b, 0x88);
    __m512i out11 = _mm512_shuffle_i32x4(v3_b, v7_b, 0x88);
    __m512i out12 = _mm512_shuffle_i32x4(v0_b, v4_b, 0xDD);
    __m512i out13 = _mm512_shuffle_i32x4(v1_b, v5_b, 0xDD);
    __m512i out14 = _mm512_shuffle_i32x4(v2_b, v6_b, 0xDD);
    __m512i out15 = _mm512_shuffle_i32x4(v3_b, v7_b, 0xDD);

    // Store transposed results - each output row is one depth_group
    // Output layout: B.data[depth_group][column][quad] = 16 columns × 4 UINT8 = 64 bytes
    _mm512_store_si512(&b_tile->data[0][0][0], out00);
    _mm512_store_si512(&b_tile->data[1][0][0], out01);
    _mm512_store_si512(&b_tile->data[2][0][0], out02);
    _mm512_store_si512(&b_tile->data[3][0][0], out03);
    _mm512_store_si512(&b_tile->data[4][0][0], out08);
    _mm512_store_si512(&b_tile->data[5][0][0], out09);
    _mm512_store_si512(&b_tile->data[6][0][0], out10);
    _mm512_store_si512(&b_tile->data[7][0][0], out11);
    _mm512_store_si512(&b_tile->data[8][0][0], out04);
    _mm512_store_si512(&b_tile->data[9][0][0], out05);
    _mm512_store_si512(&b_tile->data[10][0][0], out06);
    _mm512_store_si512(&b_tile->data[11][0][0], out07);
    _mm512_store_si512(&b_tile->data[12][0][0], out12);
    _mm512_store_si512(&b_tile->data[13][0][0], out13);
    _mm512_store_si512(&b_tile->data[14][0][0], out14);
    _mm512_store_si512(&b_tile->data[15][0][0], out15);

    nk_compiler_barrier_sapphireamx_();
}

/* Accumulate 3 A x B tile pairs into state using AMX TDPBUUD */
NK_INTERNAL void nk_dots_u8_update_sapphireamx_(     //
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

/* Load E4M3 A tile with FP8 to BF16 conversion */
NK_INTERNAL void nk_dots_e4m3_load_a_sapphireamx_( //
    nk_dots_bf16_a16x32_sapphireamx_t *a_tile,     //
    nk_e4m3_t const *src, nk_size_t src_stride,    //
    nk_size_t valid_rows, nk_size_t valid_cols) {

    __mmask32 column_mask = (valid_cols >= 32) ? 0xFFFFFFFF : ((__mmask32)1 << valid_cols) - 1;
    __m512i zero = _mm512_setzero_si512();

    for (nk_size_t row_idx = 0; row_idx < 16; row_idx++) {
        if (row_idx < valid_rows) {
            // Load 32 E4M3 bytes with masking
            __m256i e4m3_row = _mm256_maskz_loadu_epi8(column_mask, src + row_idx * src_stride);
            // Convert to 32 BF16 values
            __m512i bf16_row = nk_e4m3x32_to_bf16x32_icelake_(e4m3_row);
            _mm512_store_si512((__m512i *)a_tile->data[row_idx], bf16_row);
        }
        else { _mm512_store_si512((__m512i *)a_tile->data[row_idx], zero); }
    }
    nk_compiler_barrier_sapphireamx_();
}

/* Load E5M2 A tile with FP8 to BF16 conversion */
NK_INTERNAL void nk_dots_e5m2_load_a_sapphireamx_( //
    nk_dots_bf16_a16x32_sapphireamx_t *a_tile,     //
    nk_e5m2_t const *src, nk_size_t src_stride,    //
    nk_size_t valid_rows, nk_size_t valid_cols) {

    __mmask32 column_mask = (valid_cols >= 32) ? 0xFFFFFFFF : ((__mmask32)1 << valid_cols) - 1;
    __m512i zero = _mm512_setzero_si512();

    for (nk_size_t row_idx = 0; row_idx < 16; row_idx++) {
        if (row_idx < valid_rows) {
            __m256i e5m2_row = _mm256_maskz_loadu_epi8(column_mask, src + row_idx * src_stride);
            __m512i bf16_row = nk_e5m2x32_to_bf16x32_icelake_(e5m2_row);
            _mm512_store_si512((__m512i *)a_tile->data[row_idx], bf16_row);
        }
        else { _mm512_store_si512((__m512i *)a_tile->data[row_idx], zero); }
    }
    nk_compiler_barrier_sapphireamx_();
}

/* Pack A transposed into B format for BF16 */
NK_INTERNAL void nk_dots_pack_bf16_transposed_sapphireamx_( //
    nk_dots_bf16_a16x32_sapphireamx_t const *a_tile,        //
    nk_dots_bf16_b32x16_sapphireamx_t *b_tile) {

    // Load all 16 rows - each row is 32 BF16 = 64 bytes = 1 ZMM
    // Treat as 16 × 32-bit elements per row (each 32-bit = pair of BF16)
    __m512i row00 = _mm512_load_si512(&a_tile->data[0][0]);
    __m512i row01 = _mm512_load_si512(&a_tile->data[1][0]);
    __m512i row02 = _mm512_load_si512(&a_tile->data[2][0]);
    __m512i row03 = _mm512_load_si512(&a_tile->data[3][0]);
    __m512i row04 = _mm512_load_si512(&a_tile->data[4][0]);
    __m512i row05 = _mm512_load_si512(&a_tile->data[5][0]);
    __m512i row06 = _mm512_load_si512(&a_tile->data[6][0]);
    __m512i row07 = _mm512_load_si512(&a_tile->data[7][0]);
    __m512i row08 = _mm512_load_si512(&a_tile->data[8][0]);
    __m512i row09 = _mm512_load_si512(&a_tile->data[9][0]);
    __m512i row10 = _mm512_load_si512(&a_tile->data[10][0]);
    __m512i row11 = _mm512_load_si512(&a_tile->data[11][0]);
    __m512i row12 = _mm512_load_si512(&a_tile->data[12][0]);
    __m512i row13 = _mm512_load_si512(&a_tile->data[13][0]);
    __m512i row14 = _mm512_load_si512(&a_tile->data[14][0]);
    __m512i row15 = _mm512_load_si512(&a_tile->data[15][0]);

    // 16×16 transpose of 32-bit elements using hierarchical unpacks
    // Stage 1: Unpack adjacent row pairs at 32-bit granularity
    __m512i t01_lo = _mm512_unpacklo_epi32(row00, row01);
    __m512i t01_hi = _mm512_unpackhi_epi32(row00, row01);
    __m512i t23_lo = _mm512_unpacklo_epi32(row02, row03);
    __m512i t23_hi = _mm512_unpackhi_epi32(row02, row03);
    __m512i t45_lo = _mm512_unpacklo_epi32(row04, row05);
    __m512i t45_hi = _mm512_unpackhi_epi32(row04, row05);
    __m512i t67_lo = _mm512_unpacklo_epi32(row06, row07);
    __m512i t67_hi = _mm512_unpackhi_epi32(row06, row07);
    __m512i t89_lo = _mm512_unpacklo_epi32(row08, row09);
    __m512i t89_hi = _mm512_unpackhi_epi32(row08, row09);
    __m512i tab_lo = _mm512_unpacklo_epi32(row10, row11);
    __m512i tab_hi = _mm512_unpackhi_epi32(row10, row11);
    __m512i tcd_lo = _mm512_unpacklo_epi32(row12, row13);
    __m512i tcd_hi = _mm512_unpackhi_epi32(row12, row13);
    __m512i tef_lo = _mm512_unpacklo_epi32(row14, row15);
    __m512i tef_hi = _mm512_unpackhi_epi32(row14, row15);

    // Stage 2: Unpack at 64-bit granularity
    __m512i u0123_ll = _mm512_unpacklo_epi64(t01_lo, t23_lo);
    __m512i u0123_lh = _mm512_unpackhi_epi64(t01_lo, t23_lo);
    __m512i u0123_hl = _mm512_unpacklo_epi64(t01_hi, t23_hi);
    __m512i u0123_hh = _mm512_unpackhi_epi64(t01_hi, t23_hi);
    __m512i u4567_ll = _mm512_unpacklo_epi64(t45_lo, t67_lo);
    __m512i u4567_lh = _mm512_unpackhi_epi64(t45_lo, t67_lo);
    __m512i u4567_hl = _mm512_unpacklo_epi64(t45_hi, t67_hi);
    __m512i u4567_hh = _mm512_unpackhi_epi64(t45_hi, t67_hi);
    __m512i u89ab_ll = _mm512_unpacklo_epi64(t89_lo, tab_lo);
    __m512i u89ab_lh = _mm512_unpackhi_epi64(t89_lo, tab_lo);
    __m512i u89ab_hl = _mm512_unpacklo_epi64(t89_hi, tab_hi);
    __m512i u89ab_hh = _mm512_unpackhi_epi64(t89_hi, tab_hi);
    __m512i ucdef_ll = _mm512_unpacklo_epi64(tcd_lo, tef_lo);
    __m512i ucdef_lh = _mm512_unpackhi_epi64(tcd_lo, tef_lo);
    __m512i ucdef_hl = _mm512_unpacklo_epi64(tcd_hi, tef_hi);
    __m512i ucdef_hh = _mm512_unpackhi_epi64(tcd_hi, tef_hi);

    // Stage 3: Shuffle 128-bit lanes using permute2x128 equivalent for 512-bit
    // Use shuffle_i32x4 to move 128-bit chunks
    __m512i v0_a = _mm512_shuffle_i32x4(u0123_ll, u4567_ll, 0x88); // lanes 0,2 from each
    __m512i v0_b = _mm512_shuffle_i32x4(u0123_ll, u4567_ll, 0xDD); // lanes 1,3 from each
    __m512i v1_a = _mm512_shuffle_i32x4(u0123_lh, u4567_lh, 0x88);
    __m512i v1_b = _mm512_shuffle_i32x4(u0123_lh, u4567_lh, 0xDD);
    __m512i v2_a = _mm512_shuffle_i32x4(u0123_hl, u4567_hl, 0x88);
    __m512i v2_b = _mm512_shuffle_i32x4(u0123_hl, u4567_hl, 0xDD);
    __m512i v3_a = _mm512_shuffle_i32x4(u0123_hh, u4567_hh, 0x88);
    __m512i v3_b = _mm512_shuffle_i32x4(u0123_hh, u4567_hh, 0xDD);
    __m512i v4_a = _mm512_shuffle_i32x4(u89ab_ll, ucdef_ll, 0x88);
    __m512i v4_b = _mm512_shuffle_i32x4(u89ab_ll, ucdef_ll, 0xDD);
    __m512i v5_a = _mm512_shuffle_i32x4(u89ab_lh, ucdef_lh, 0x88);
    __m512i v5_b = _mm512_shuffle_i32x4(u89ab_lh, ucdef_lh, 0xDD);
    __m512i v6_a = _mm512_shuffle_i32x4(u89ab_hl, ucdef_hl, 0x88);
    __m512i v6_b = _mm512_shuffle_i32x4(u89ab_hl, ucdef_hl, 0xDD);
    __m512i v7_a = _mm512_shuffle_i32x4(u89ab_hh, ucdef_hh, 0x88);
    __m512i v7_b = _mm512_shuffle_i32x4(u89ab_hh, ucdef_hh, 0xDD);

    // Stage 4: Final 256-bit shuffle to complete transpose
    __m512i out00 = _mm512_shuffle_i32x4(v0_a, v4_a, 0x88);
    __m512i out01 = _mm512_shuffle_i32x4(v1_a, v5_a, 0x88);
    __m512i out02 = _mm512_shuffle_i32x4(v2_a, v6_a, 0x88);
    __m512i out03 = _mm512_shuffle_i32x4(v3_a, v7_a, 0x88);
    __m512i out04 = _mm512_shuffle_i32x4(v0_a, v4_a, 0xDD);
    __m512i out05 = _mm512_shuffle_i32x4(v1_a, v5_a, 0xDD);
    __m512i out06 = _mm512_shuffle_i32x4(v2_a, v6_a, 0xDD);
    __m512i out07 = _mm512_shuffle_i32x4(v3_a, v7_a, 0xDD);
    __m512i out08 = _mm512_shuffle_i32x4(v0_b, v4_b, 0x88);
    __m512i out09 = _mm512_shuffle_i32x4(v1_b, v5_b, 0x88);
    __m512i out10 = _mm512_shuffle_i32x4(v2_b, v6_b, 0x88);
    __m512i out11 = _mm512_shuffle_i32x4(v3_b, v7_b, 0x88);
    __m512i out12 = _mm512_shuffle_i32x4(v0_b, v4_b, 0xDD);
    __m512i out13 = _mm512_shuffle_i32x4(v1_b, v5_b, 0xDD);
    __m512i out14 = _mm512_shuffle_i32x4(v2_b, v6_b, 0xDD);
    __m512i out15 = _mm512_shuffle_i32x4(v3_b, v7_b, 0xDD);

    // Store transposed results - each output row is one depth_group
    // Output layout: B.data[depth_group][column][pair] = 16 columns × 2 BF16 = 64 bytes
    _mm512_store_si512(&b_tile->data[0][0][0], out00);
    _mm512_store_si512(&b_tile->data[1][0][0], out01);
    _mm512_store_si512(&b_tile->data[2][0][0], out02);
    _mm512_store_si512(&b_tile->data[3][0][0], out03);
    _mm512_store_si512(&b_tile->data[4][0][0], out08);
    _mm512_store_si512(&b_tile->data[5][0][0], out09);
    _mm512_store_si512(&b_tile->data[6][0][0], out10);
    _mm512_store_si512(&b_tile->data[7][0][0], out11);
    _mm512_store_si512(&b_tile->data[8][0][0], out04);
    _mm512_store_si512(&b_tile->data[9][0][0], out05);
    _mm512_store_si512(&b_tile->data[10][0][0], out06);
    _mm512_store_si512(&b_tile->data[11][0][0], out07);
    _mm512_store_si512(&b_tile->data[12][0][0], out12);
    _mm512_store_si512(&b_tile->data[13][0][0], out13);
    _mm512_store_si512(&b_tile->data[14][0][0], out14);
    _mm512_store_si512(&b_tile->data[15][0][0], out15);

    nk_compiler_barrier_sapphireamx_();
}

/* Pack A transposed into B format for INT8 */
NK_INTERNAL void nk_dots_pack_i8_transposed_sapphireamx_( //
    nk_dots_i8_a16x64_sapphireamx_t const *a_tile,        //
    nk_dots_i8_b64x16_sapphireamx_t *b_tile) {

    // Load all 16 rows - each row is 64 INT8 = 64 bytes = 1 ZMM
    // Treat as 16 × 32-bit elements per row (each 32-bit = quad of INT8)
    __m512i row00 = _mm512_load_si512(&a_tile->data[0][0]);
    __m512i row01 = _mm512_load_si512(&a_tile->data[1][0]);
    __m512i row02 = _mm512_load_si512(&a_tile->data[2][0]);
    __m512i row03 = _mm512_load_si512(&a_tile->data[3][0]);
    __m512i row04 = _mm512_load_si512(&a_tile->data[4][0]);
    __m512i row05 = _mm512_load_si512(&a_tile->data[5][0]);
    __m512i row06 = _mm512_load_si512(&a_tile->data[6][0]);
    __m512i row07 = _mm512_load_si512(&a_tile->data[7][0]);
    __m512i row08 = _mm512_load_si512(&a_tile->data[8][0]);
    __m512i row09 = _mm512_load_si512(&a_tile->data[9][0]);
    __m512i row10 = _mm512_load_si512(&a_tile->data[10][0]);
    __m512i row11 = _mm512_load_si512(&a_tile->data[11][0]);
    __m512i row12 = _mm512_load_si512(&a_tile->data[12][0]);
    __m512i row13 = _mm512_load_si512(&a_tile->data[13][0]);
    __m512i row14 = _mm512_load_si512(&a_tile->data[14][0]);
    __m512i row15 = _mm512_load_si512(&a_tile->data[15][0]);

    // 16×16 transpose of 32-bit elements using hierarchical unpacks
    // Stage 1: Unpack adjacent row pairs at 32-bit granularity
    __m512i t01_lo = _mm512_unpacklo_epi32(row00, row01);
    __m512i t01_hi = _mm512_unpackhi_epi32(row00, row01);
    __m512i t23_lo = _mm512_unpacklo_epi32(row02, row03);
    __m512i t23_hi = _mm512_unpackhi_epi32(row02, row03);
    __m512i t45_lo = _mm512_unpacklo_epi32(row04, row05);
    __m512i t45_hi = _mm512_unpackhi_epi32(row04, row05);
    __m512i t67_lo = _mm512_unpacklo_epi32(row06, row07);
    __m512i t67_hi = _mm512_unpackhi_epi32(row06, row07);
    __m512i t89_lo = _mm512_unpacklo_epi32(row08, row09);
    __m512i t89_hi = _mm512_unpackhi_epi32(row08, row09);
    __m512i tab_lo = _mm512_unpacklo_epi32(row10, row11);
    __m512i tab_hi = _mm512_unpackhi_epi32(row10, row11);
    __m512i tcd_lo = _mm512_unpacklo_epi32(row12, row13);
    __m512i tcd_hi = _mm512_unpackhi_epi32(row12, row13);
    __m512i tef_lo = _mm512_unpacklo_epi32(row14, row15);
    __m512i tef_hi = _mm512_unpackhi_epi32(row14, row15);

    // Stage 2: Unpack at 64-bit granularity
    __m512i u0123_ll = _mm512_unpacklo_epi64(t01_lo, t23_lo);
    __m512i u0123_lh = _mm512_unpackhi_epi64(t01_lo, t23_lo);
    __m512i u0123_hl = _mm512_unpacklo_epi64(t01_hi, t23_hi);
    __m512i u0123_hh = _mm512_unpackhi_epi64(t01_hi, t23_hi);
    __m512i u4567_ll = _mm512_unpacklo_epi64(t45_lo, t67_lo);
    __m512i u4567_lh = _mm512_unpackhi_epi64(t45_lo, t67_lo);
    __m512i u4567_hl = _mm512_unpacklo_epi64(t45_hi, t67_hi);
    __m512i u4567_hh = _mm512_unpackhi_epi64(t45_hi, t67_hi);
    __m512i u89ab_ll = _mm512_unpacklo_epi64(t89_lo, tab_lo);
    __m512i u89ab_lh = _mm512_unpackhi_epi64(t89_lo, tab_lo);
    __m512i u89ab_hl = _mm512_unpacklo_epi64(t89_hi, tab_hi);
    __m512i u89ab_hh = _mm512_unpackhi_epi64(t89_hi, tab_hi);
    __m512i ucdef_ll = _mm512_unpacklo_epi64(tcd_lo, tef_lo);
    __m512i ucdef_lh = _mm512_unpackhi_epi64(tcd_lo, tef_lo);
    __m512i ucdef_hl = _mm512_unpacklo_epi64(tcd_hi, tef_hi);
    __m512i ucdef_hh = _mm512_unpackhi_epi64(tcd_hi, tef_hi);

    // Stage 3: Shuffle 128-bit lanes
    __m512i v0_a = _mm512_shuffle_i32x4(u0123_ll, u4567_ll, 0x88);
    __m512i v0_b = _mm512_shuffle_i32x4(u0123_ll, u4567_ll, 0xDD);
    __m512i v1_a = _mm512_shuffle_i32x4(u0123_lh, u4567_lh, 0x88);
    __m512i v1_b = _mm512_shuffle_i32x4(u0123_lh, u4567_lh, 0xDD);
    __m512i v2_a = _mm512_shuffle_i32x4(u0123_hl, u4567_hl, 0x88);
    __m512i v2_b = _mm512_shuffle_i32x4(u0123_hl, u4567_hl, 0xDD);
    __m512i v3_a = _mm512_shuffle_i32x4(u0123_hh, u4567_hh, 0x88);
    __m512i v3_b = _mm512_shuffle_i32x4(u0123_hh, u4567_hh, 0xDD);
    __m512i v4_a = _mm512_shuffle_i32x4(u89ab_ll, ucdef_ll, 0x88);
    __m512i v4_b = _mm512_shuffle_i32x4(u89ab_ll, ucdef_ll, 0xDD);
    __m512i v5_a = _mm512_shuffle_i32x4(u89ab_lh, ucdef_lh, 0x88);
    __m512i v5_b = _mm512_shuffle_i32x4(u89ab_lh, ucdef_lh, 0xDD);
    __m512i v6_a = _mm512_shuffle_i32x4(u89ab_hl, ucdef_hl, 0x88);
    __m512i v6_b = _mm512_shuffle_i32x4(u89ab_hl, ucdef_hl, 0xDD);
    __m512i v7_a = _mm512_shuffle_i32x4(u89ab_hh, ucdef_hh, 0x88);
    __m512i v7_b = _mm512_shuffle_i32x4(u89ab_hh, ucdef_hh, 0xDD);

    // Stage 4: Final 256-bit shuffle to complete transpose
    __m512i out00 = _mm512_shuffle_i32x4(v0_a, v4_a, 0x88);
    __m512i out01 = _mm512_shuffle_i32x4(v1_a, v5_a, 0x88);
    __m512i out02 = _mm512_shuffle_i32x4(v2_a, v6_a, 0x88);
    __m512i out03 = _mm512_shuffle_i32x4(v3_a, v7_a, 0x88);
    __m512i out04 = _mm512_shuffle_i32x4(v0_a, v4_a, 0xDD);
    __m512i out05 = _mm512_shuffle_i32x4(v1_a, v5_a, 0xDD);
    __m512i out06 = _mm512_shuffle_i32x4(v2_a, v6_a, 0xDD);
    __m512i out07 = _mm512_shuffle_i32x4(v3_a, v7_a, 0xDD);
    __m512i out08 = _mm512_shuffle_i32x4(v0_b, v4_b, 0x88);
    __m512i out09 = _mm512_shuffle_i32x4(v1_b, v5_b, 0x88);
    __m512i out10 = _mm512_shuffle_i32x4(v2_b, v6_b, 0x88);
    __m512i out11 = _mm512_shuffle_i32x4(v3_b, v7_b, 0x88);
    __m512i out12 = _mm512_shuffle_i32x4(v0_b, v4_b, 0xDD);
    __m512i out13 = _mm512_shuffle_i32x4(v1_b, v5_b, 0xDD);
    __m512i out14 = _mm512_shuffle_i32x4(v2_b, v6_b, 0xDD);
    __m512i out15 = _mm512_shuffle_i32x4(v3_b, v7_b, 0xDD);

    // Store transposed results - each output row is one depth_group
    // Output layout: B.data[depth_group][column][quad] = 16 columns × 4 INT8 = 64 bytes
    _mm512_store_si512(&b_tile->data[0][0][0], out00);
    _mm512_store_si512(&b_tile->data[1][0][0], out01);
    _mm512_store_si512(&b_tile->data[2][0][0], out02);
    _mm512_store_si512(&b_tile->data[3][0][0], out03);
    _mm512_store_si512(&b_tile->data[4][0][0], out08);
    _mm512_store_si512(&b_tile->data[5][0][0], out09);
    _mm512_store_si512(&b_tile->data[6][0][0], out10);
    _mm512_store_si512(&b_tile->data[7][0][0], out11);
    _mm512_store_si512(&b_tile->data[8][0][0], out04);
    _mm512_store_si512(&b_tile->data[9][0][0], out05);
    _mm512_store_si512(&b_tile->data[10][0][0], out06);
    _mm512_store_si512(&b_tile->data[11][0][0], out07);
    _mm512_store_si512(&b_tile->data[12][0][0], out12);
    _mm512_store_si512(&b_tile->data[13][0][0], out13);
    _mm512_store_si512(&b_tile->data[14][0][0], out14);
    _mm512_store_si512(&b_tile->data[15][0][0], out15);

    nk_compiler_barrier_sapphireamx_();
}

#pragma region Half Precision Floats

NK_PUBLIC nk_size_t nk_dots_packed_size_bf16_sapphireamx(nk_size_t column_count, nk_size_t depth) {
    nk_size_t const tmm_rows = 16;
    nk_size_t const tmm_cols = 32;
    nk_size_t const tile_bytes = 512 * sizeof(nk_bf16_t); // 16 × 32 × 2 = 1KB

    nk_size_t const full_column_tiles = column_count / tmm_rows;
    nk_size_t const tiles_along_depth = nk_size_divide_round_up_(depth, tmm_cols);
    nk_size_t const column_remainder_count = column_count - full_column_tiles * tmm_rows;

    // Header (64 bytes aligned)
    nk_size_t size = sizeof(nk_dots_amx_packed_header_t);

    // All tiles for full column rows (Morton-ordered, pair-interleaved, depth remainder zero-padded)
    size += full_column_tiles * tiles_along_depth * tile_bytes;

    // Column edge: remaining rows for ALL depth columns, stored row-major
    if (column_remainder_count > 0) size += column_remainder_count * depth * sizeof(nk_bf16_t);

    // Per-column norms for angular/euclidean distance (4 bytes each: f32 or u32)
    size += column_count * sizeof(nk_f32_t);

    return size;
}

NK_PUBLIC void nk_dots_pack_bf16_sapphireamx(                    //
    nk_bf16_t const *b, nk_size_t column_count, nk_size_t depth, //
    nk_size_t b_stride, void *b_packed) {

    // AMX BF16 tile dimensions: 16 rows × 32 columns (512 BF16 elements = 1KB)
    nk_size_t const tmm_rows = 16;
    nk_size_t const tmm_cols = 32;
    nk_size_t const tile_elements = 512;
    nk_size_t const tile_bytes = tile_elements * sizeof(nk_bf16_t);
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_bf16_t);

    // Compute layout dimensions
    nk_size_t const column_tiles_count = column_count / tmm_rows;
    nk_size_t const depth_tiles_count = nk_size_divide_round_up_(depth, tmm_cols);
    nk_size_t const column_remainder_count = column_count - column_tiles_count * tmm_rows;
    nk_size_t const total_tiles = column_tiles_count * depth_tiles_count;

    // Write header with layout metadata
    nk_dots_amx_packed_header_t *header = (nk_dots_amx_packed_header_t *)b_packed;
    header->full_column_tiles = (nk_u32_t)column_tiles_count;
    header->full_depth_tiles = (nk_u32_t)depth_tiles_count;
    header->column_remainder_count = (nk_u32_t)column_remainder_count;

    // Compute memory region offsets
    nk_size_t const tiles_offset = sizeof(nk_dots_amx_packed_header_t);
    nk_size_t const column_edge_offset = tiles_offset + total_tiles * tile_bytes;
    header->column_edge_offset = (nk_u32_t)column_edge_offset;

    // Pointers to packed data regions
    nk_bf16_t *tiles_ptr = (nk_bf16_t *)((char *)b_packed + tiles_offset);
    nk_bf16_t *column_edge_ptr = (nk_bf16_t *)((char *)b_packed + column_edge_offset);

    // Zero-initialize all tiles (handles depth remainder padding)
    for (nk_size_t idx = 0; idx < total_tiles * tile_elements; idx++) tiles_ptr[idx] = 0;

    // Pack tiles using LINEAR ordering: tile_index = column_tile × depth_tiles_count + depth_tile
    // This provides sequential memory access when streaming along depth dimension,
    // which is critical for cache efficiency in the compute kernel.
    for (nk_size_t column_tile_idx = 0; column_tile_idx < column_tiles_count; column_tile_idx++) {
        for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {

            // Linear tile index: all depth-tiles for one column-tile are contiguous
            nk_size_t const tile_index = column_tile_idx * depth_tiles_count + depth_tile_idx;
            nk_bf16_t *tile_output = tiles_ptr + tile_index * tile_elements;

            // Source coordinates in original B matrix
            nk_size_t const src_row_start = column_tile_idx * tmm_rows;
            nk_size_t const src_column_start = depth_tile_idx * tmm_cols;
            nk_size_t const columns_to_pack = (src_column_start + tmm_cols <= depth) ? tmm_cols
                                                                                     : (depth - src_column_start);

            // Pack with pair-interleaving as required by TDPBF16PS instruction.
            // AMX expects: [col0_row0, col1_row0, col0_row1, col1_row1, col2_row0, col3_row0, ...]
            // Formula: packed_idx = (column / 2) × 32 + row × 2 + (column % 2)
            for (nk_size_t row_idx = 0; row_idx < tmm_rows; row_idx++) {
                for (nk_size_t column_idx = 0; column_idx < columns_to_pack; column_idx++) {
                    nk_size_t const src_idx = (src_row_start + row_idx) * b_stride_elements + src_column_start +
                                              column_idx;
                    nk_size_t const dst_idx = (column_idx / 2) * 32 + row_idx * 2 + (column_idx % 2);
                    tile_output[dst_idx] = b[src_idx];
                }
            }
        }
    }

    // Pack column-remainder rows in simple row-major format (for AVX-512 fallback)
    if (column_remainder_count > 0) {
        nk_size_t const remainder_start_row = column_tiles_count * tmm_rows;
        for (nk_size_t row_idx = 0; row_idx < column_remainder_count; row_idx++) {
            for (nk_size_t column_idx = 0; column_idx < depth; column_idx++) {
                column_edge_ptr[row_idx * depth + column_idx] =
                    b[(remainder_start_row + row_idx) * b_stride_elements + column_idx];
            }
        }
    }

    // Compute and store per-column norms for angular/euclidean distance
    nk_size_t norms_offset = column_edge_offset +
                             (column_remainder_count > 0 ? column_remainder_count * depth * sizeof(nk_bf16_t) : 0);
    header->norms_byte_offset = (nk_u32_t)norms_offset;
    nk_f32_t *norms = (nk_f32_t *)((char *)b_packed + norms_offset);
    for (nk_size_t col = 0; col < column_count; col++)
        norms[col] = nk_dots_reduce_sumsq_bf16_(b + col * b_stride_elements, depth);
}

NK_PUBLIC void nk_dots_packed_bf16_sapphireamx(            //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows_count, nk_size_t cols_count, nk_size_t depth, nk_size_t a_stride_bytes, nk_size_t c_stride_bytes) {
    nk_unused_(cols_count);

    // Parse packed B header
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_size_t const column_tiles_count = header->full_column_tiles;
    nk_size_t const depth_tiles_count = header->full_depth_tiles;
    nk_size_t const column_remainder_count = header->column_remainder_count;

    // Packed B data regions
    nk_bf16_t const *b_tiles_base = (nk_bf16_t const *)((char const *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_bf16_t const *col_edge_ptr = (nk_bf16_t const *)((char const *)b_packed + header->column_edge_offset);

    // Stride conversions
    nk_size_t const a_stride_elements = a_stride_bytes / sizeof(nk_bf16_t);
    nk_size_t const c_stride_elements = c_stride_bytes / sizeof(nk_f32_t);

    // Tile dimensions
    nk_size_t const tile_depth = 32; // depth elements per BF16 tile
    nk_size_t const tile_size = 512; // elements per packed tile
    nk_size_t const full_cols = column_tiles_count * 16;

    // Block counts (32 × 32 output blocks = 2 × 2 tiles)
    nk_size_t const row_blocks_count = nk_size_divide_round_up_(rows_count, 32);
    nk_size_t const col_blocks_count = column_tiles_count / 2;

    if (depth_tiles_count == 0) return;

    // Tile buffers for A (only used for edge tiles)
    nk_dots_bf16_a16x32_sapphireamx_t a_tile_upper, a_tile_lower;
    nk_dots_bf16_state2x2_sapphireamx_t c_accum_buffer;

    // Precompute: number of full depth-tiles (no masking needed)
    nk_size_t const full_depth_tiles_count = depth / tile_depth;
    nk_size_t const depth_remainder = depth % tile_depth;

    nk_amx_tile_configure_sapphireamx_();

    // Loop order: row_blocks outer, col_blocks inner - maximizes A tile L2 cache reuse
    // A tiles stay in L2 while we sweep through all col_blocks for a given row_block
    for (nk_size_t row_block_idx = 0; row_block_idx < row_blocks_count; row_block_idx++) {
        nk_size_t const row_block_start = row_block_idx * 32;
        nk_size_t const valid_rows_count = (row_block_start + 32 <= rows_count) ? 32 : (rows_count - row_block_start);
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
                nk_bf16_t const *a_upper_base = a + row_block_start * a_stride_elements;
                nk_bf16_t const *a_lower_base = a + (row_block_start + 16) * a_stride_elements;

                nk_dots_bf16_b32x16_sapphireamx_t const *b_tile_left =
                    (nk_dots_bf16_b32x16_sapphireamx_t const *)(b_tiles_base + b_column_left_base * tile_size);
                nk_dots_bf16_b32x16_sapphireamx_t const *b_tile_right =
                    (nk_dots_bf16_b32x16_sapphireamx_t const *)(b_tiles_base + b_column_right_base * tile_size);

                // Prologue: load first depth tile
                _tile_loadd(0, a_upper_base, a_stride_bytes);
                _tile_loadd(1, a_lower_base, a_stride_bytes);
                _tile_loadd(2, b_tile_left->data, 64);
                _tile_loadd(3, b_tile_right->data, 64);

                // Main loop: 2-deep software pipelining
                for (nk_size_t depth_tile_idx = 0; depth_tile_idx < full_depth_tiles_count - 1; depth_tile_idx++) {
                    nk_size_t const next_depth_offset = (depth_tile_idx + 1) * tile_depth;

                    _tile_dpbf16ps(4, 0, 2);
                    _tile_dpbf16ps(5, 0, 3);
                    _tile_dpbf16ps(6, 1, 2);
                    _tile_dpbf16ps(7, 1, 3);

                    _tile_loadd(0, a_upper_base + next_depth_offset, a_stride_bytes);
                    _tile_loadd(1, a_lower_base + next_depth_offset, a_stride_bytes);
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

                    nk_dots_bf16_load_a_sapphireamx_(&a_tile_upper, a_upper_base + depth_offset, a_stride_elements, 16,
                                                     depth_remainder);
                    nk_dots_bf16_load_a_sapphireamx_(&a_tile_lower, a_lower_base + depth_offset, a_stride_elements, 16,
                                                     depth_remainder);

                    b_tile_left = (nk_dots_bf16_b32x16_sapphireamx_t const *)(b_tiles_base + (b_column_left_base +
                                                                                              full_depth_tiles_count) *
                                                                                                 tile_size);
                    b_tile_right = (nk_dots_bf16_b32x16_sapphireamx_t const *)(b_tiles_base + (b_column_right_base +
                                                                                               full_depth_tiles_count) *
                                                                                                  tile_size);

                    _tile_loadd(0, a_tile_upper.data, 64);
                    _tile_loadd(1, a_tile_lower.data, 64);
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
                nk_bf16_t const *a_upper_base = a + row_block_start * a_stride_elements;
                nk_bf16_t const *a_lower_base = a + (row_block_start + 16) * a_stride_elements;

                nk_dots_bf16_load_a_sapphireamx_(&a_tile_upper, a_upper_base, a_stride_elements, 16, depth_remainder);
                nk_dots_bf16_load_a_sapphireamx_(&a_tile_lower, a_lower_base, a_stride_elements, 16, depth_remainder);

                nk_dots_bf16_b32x16_sapphireamx_t const *b_tile_left =
                    (nk_dots_bf16_b32x16_sapphireamx_t const *)(b_tiles_base + b_column_left_base * tile_size);
                nk_dots_bf16_b32x16_sapphireamx_t const *b_tile_right =
                    (nk_dots_bf16_b32x16_sapphireamx_t const *)(b_tiles_base + b_column_right_base * tile_size);

                _tile_loadd(0, a_tile_upper.data, 64);
                _tile_loadd(1, a_tile_lower.data, 64);
                _tile_loadd(2, b_tile_left->data, 64);
                _tile_loadd(3, b_tile_right->data, 64);

                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(5, 0, 3);
                _tile_dpbf16ps(6, 1, 2);
                _tile_dpbf16ps(7, 1, 3);
            }
            // Slow path: edge row-block → buffered load with masking
            else {
                nk_size_t const rows_in_upper_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
                nk_size_t const rows_in_lower_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

                for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                    nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                    nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth
                                                                                            : depth_remainder;

                    nk_dots_bf16_load_a_sapphireamx_(&a_tile_upper,
                                                     a + row_block_start * a_stride_elements + depth_offset,
                                                     a_stride_elements, rows_in_upper_tile, valid_depth);
                    if (rows_in_lower_tile > 0) {
                        nk_dots_bf16_load_a_sapphireamx_(&a_tile_lower,
                                                         a + (row_block_start + 16) * a_stride_elements + depth_offset,
                                                         a_stride_elements, rows_in_lower_tile, valid_depth);
                    }

                    nk_dots_bf16_b32x16_sapphireamx_t const *b_tile_left =
                        (nk_dots_bf16_b32x16_sapphireamx_t const *)(b_tiles_base +
                                                                    (b_column_left_base + depth_tile_idx) * tile_size);
                    nk_dots_bf16_b32x16_sapphireamx_t const *b_tile_right =
                        (nk_dots_bf16_b32x16_sapphireamx_t const *)(b_tiles_base +
                                                                    (b_column_right_base + depth_tile_idx) * tile_size);

                    _tile_loadd(0, a_tile_upper.data, 64);
                    _tile_loadd(1, a_tile_lower.data, 64);
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
                _tile_stored(4, c_block, c_stride_bytes);
                _tile_stored(5, c_block + 16, c_stride_bytes);
                _tile_stored(6, (nk_f32_t *)((char *)c_block + 16 * c_stride_bytes), c_stride_bytes);
                _tile_stored(7, (nk_f32_t *)((char *)c_block + 16 * c_stride_bytes) + 16, c_stride_bytes);
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
            nk_size_t const valid_rows_count = (row_block_start + 32 <= rows_count) ? 32
                                                                                    : (rows_count - row_block_start);
            nk_size_t const rows_in_upper_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
            nk_size_t const rows_in_lower_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

            nk_dots_bf16_state_sapphireamx_t c_upper_state, c_lower_state;

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_bf16_load_a_sapphireamx_(&a_tile_upper, a + row_block_start * a_stride_elements + depth_offset,
                                                 a_stride_elements, rows_in_upper_tile, valid_depth);
                if (rows_in_lower_tile > 0) {
                    nk_dots_bf16_load_a_sapphireamx_(&a_tile_lower,
                                                     a + (row_block_start + 16) * a_stride_elements + depth_offset,
                                                     a_stride_elements, rows_in_lower_tile, valid_depth);
                }

                nk_dots_bf16_b32x16_sapphireamx_t const *b_tile =
                    (nk_dots_bf16_b32x16_sapphireamx_t const *)(b_tiles_base +
                                                                (b_column_base + depth_tile_idx) * tile_size);

                _tile_loadd(0, a_tile_upper.data, 64);
                _tile_loadd(1, a_tile_lower.data, 64);
                _tile_loadd(2, b_tile->data, 64);

                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(6, 1, 2);
            }

            _tile_stored(4, c_upper_state.data, 64);
            _tile_stored(6, c_lower_state.data, 64);

            nk_dots_bf16_store_sapphireamx_(&c_upper_state, c + row_block_start * c_stride_elements + col_start,
                                            c_stride_elements, rows_in_upper_tile, 16);
            if (rows_in_lower_tile > 0) {
                nk_dots_bf16_store_sapphireamx_(&c_lower_state,
                                                c + (row_block_start + 16) * c_stride_elements + col_start,
                                                c_stride_elements, rows_in_lower_tile, 16);
            }
        }
    }

    // Handle column-edge (remaining columns < 16) using AMX with partial tiles
    if (column_remainder_count > 0) {
        for (nk_size_t row_block_idx = 0; row_block_idx < row_blocks_count; row_block_idx++) {
            nk_size_t const row_block_start = row_block_idx * 32;
            nk_size_t const valid_rows_count = (row_block_start + 32 <= rows_count) ? 32
                                                                                    : (rows_count - row_block_start);
            nk_size_t const rows_in_upper_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
            nk_size_t const rows_in_lower_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

            nk_dots_bf16_state_sapphireamx_t c_upper_state, c_lower_state;
            nk_dots_bf16_a16x32_sapphireamx_t b_as_a;
            nk_dots_bf16_b32x16_sapphireamx_t b_tile;

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_bf16_load_a_sapphireamx_(&a_tile_upper, a + row_block_start * a_stride_elements + depth_offset,
                                                 a_stride_elements, rows_in_upper_tile, valid_depth);
                if (rows_in_lower_tile > 0) {
                    nk_dots_bf16_load_a_sapphireamx_(&a_tile_lower,
                                                     a + (row_block_start + 16) * a_stride_elements + depth_offset,
                                                     a_stride_elements, rows_in_lower_tile, valid_depth);
                }

                nk_dots_bf16_load_a_sapphireamx_(&b_as_a, col_edge_ptr + depth_offset, depth, column_remainder_count,
                                                 valid_depth);
                nk_dots_pack_bf16_transposed_sapphireamx_(&b_as_a, &b_tile);

                _tile_loadd(0, a_tile_upper.data, 64);
                _tile_loadd(1, a_tile_lower.data, 64);
                _tile_loadd(2, b_tile.data, 64);

                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(6, 1, 2);
            }

            _tile_stored(4, c_upper_state.data, 64);
            _tile_stored(6, c_lower_state.data, 64);

            nk_dots_bf16_store_sapphireamx_(&c_upper_state, c + row_block_start * c_stride_elements + full_cols,
                                            c_stride_elements, rows_in_upper_tile, column_remainder_count);
            if (rows_in_lower_tile > 0) {
                nk_dots_bf16_store_sapphireamx_(&c_lower_state,
                                                c + (row_block_start + 16) * c_stride_elements + full_cols,
                                                c_stride_elements, rows_in_lower_tile, column_remainder_count);
            }
        }
    }

    _tile_release();
}

NK_PUBLIC void nk_dots_compact_bf16_sapphireamx( //
    void *c, nk_size_t row_count, nk_size_t column_count, nk_size_t c_stride) {

    nk_size_t const c_stride_f32 = c_stride / sizeof(nk_f32_t);
    nk_f32_t const *c_f32 = (nk_f32_t const *)c;
    nk_bf16_t *c_bf16 = (nk_bf16_t *)c;

    for (nk_size_t row_idx = 0; row_idx < row_count; row_idx++) {
        nk_f32_t const *src_row = c_f32 + row_idx * c_stride_f32;
        nk_bf16_t *dst_row = c_bf16 + row_idx * column_count;
        nk_size_t column_idx = 0;

        // Process 16 floats at a time using AVX512-BF16
        for (; column_idx + 16 <= column_count; column_idx += 16) {
            __m512 f32_vec = _mm512_loadu_ps(src_row + column_idx);
            __m256bh bf16_vec = _mm512_cvtneps_pbh(f32_vec);
            _mm256_storeu_si256((__m256i *)(dst_row + column_idx), nk_m256i_from_m256bh_(bf16_vec));
        }

        // Handle remaining elements with masked operations
        if (column_idx < column_count) {
            __mmask16 tail_mask = (__mmask16)((1u << (column_count - column_idx)) - 1);
            __m512 f32_vec = _mm512_maskz_loadu_ps(tail_mask, src_row + column_idx);
            __m256bh bf16_vec = _mm512_cvtneps_pbh(f32_vec);
            _mm256_mask_storeu_epi16(dst_row + column_idx, tail_mask, nk_m256i_from_m256bh_(bf16_vec));
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_bf16_sapphireamx(                  //
    nk_bf16_t const *vectors, nk_size_t n_vectors, nk_size_t depth, //
    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,    //
    nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const stride_elements = stride / sizeof(nk_bf16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);

    // Handle row slicing: compute rows [row_start, row_end)
    nk_size_t const row_end = (row_count == 0)
                                  ? n_vectors
                                  : (row_start + row_count < n_vectors ? row_start + row_count : n_vectors);

    // Round depth up to multiple of 96 (3 tiles × 32 elements)
    nk_size_t const depth_tiles = nk_size_divide_round_up_(depth, 32);
    nk_size_t const depth_tile_groups = nk_size_divide_round_up_(depth_tiles, 3);

    nk_dots_bf16_a16x32_sapphireamx_t a_tiles[3];
    nk_dots_bf16_a16x32_sapphireamx_t b_src_tiles[3];
    nk_dots_bf16_b32x16_sapphireamx_t b_tiles[3];
    nk_dots_bf16_state_sapphireamx_t state;

    nk_amx_tile_configure_sapphireamx_();

    for (nk_size_t row_tile = row_start; row_tile < row_end; row_tile += 16) {
        nk_size_t const valid_rows = (row_tile + 16 <= row_end) ? 16 : (row_end - row_tile);

        for (nk_size_t col_tile = 0; col_tile < n_vectors; col_tile += 16) {
            nk_size_t const valid_cols = (col_tile + 16 <= n_vectors) ? 16 : (n_vectors - col_tile);

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

                    if (row_tile == col_tile) {
                        nk_dots_pack_bf16_transposed_sapphireamx_(&a_tiles[tile_idx], &b_tiles[tile_idx]);
                    }
                    else {
                        nk_dots_bf16_load_a_sapphireamx_(                       //
                            &b_src_tiles[tile_idx],                             //
                            vectors + col_tile * stride_elements + depth_start, //
                            stride_elements, valid_cols, valid_depth);
                        nk_dots_pack_bf16_transposed_sapphireamx_(&b_src_tiles[tile_idx], &b_tiles[tile_idx]);
                    }
                }

                nk_dots_bf16_update_sapphireamx_( //
                    &state, &a_tiles[0], &a_tiles[1], &a_tiles[2], &b_tiles[0], &b_tiles[1], &b_tiles[2]);
            }

            nk_dots_bf16_store_sapphireamx_(                                   //
                &state, result + row_tile * result_stride_elements + col_tile, //
                result_stride_elements, valid_rows, valid_cols);
        }
    }
}

#pragma endregion // Half Precision Floats

#pragma region Signed Integers

NK_PUBLIC nk_size_t nk_dots_packed_size_i8_sapphireamx(nk_size_t column_count, nk_size_t depth) {
    nk_size_t const tmm_rows = 16;
    nk_size_t const tmm_cols = 64;
    nk_size_t const tile_bytes = 1024 * sizeof(nk_i8_t); // 16 × 64×1 = 1KB

    nk_size_t const full_column_tiles = column_count / tmm_rows;
    nk_size_t const tiles_along_depth = nk_size_divide_round_up_(depth, tmm_cols);
    nk_size_t const column_remainder_count = column_count - full_column_tiles * tmm_rows;

    // Header (64 bytes aligned)
    nk_size_t size = sizeof(nk_dots_amx_packed_header_t);

    // All tiles for full column rows (Morton-ordered, quad-interleaved, depth remainder zero-padded)
    size += full_column_tiles * tiles_along_depth * tile_bytes;

    // Column edge: remaining rows for ALL depth columns, stored row-major
    if (column_remainder_count > 0) size += column_remainder_count * depth * sizeof(nk_i8_t);

    // Per-column norms for angular/euclidean distance (4 bytes each: f32 or u32)
    size += column_count * sizeof(nk_u32_t);

    return size;
}

NK_PUBLIC void nk_dots_pack_i8_sapphireamx(                    //
    nk_i8_t const *b, nk_size_t column_count, nk_size_t depth, //
    nk_size_t b_stride, void *b_packed) {

    // AMX I8 tile dimensions: 16 rows × 64 columns (1024 I8 elements = 1KB)
    nk_size_t const tmm_rows = 16;
    nk_size_t const tmm_cols = 64;
    nk_size_t const tile_elements = 1024;
    nk_size_t const tile_bytes = tile_elements * sizeof(nk_i8_t);

    // Compute layout dimensions
    nk_size_t const column_tiles_count = column_count / tmm_rows;
    nk_size_t const depth_tiles_count = nk_size_divide_round_up_(depth, tmm_cols);
    nk_size_t const column_remainder_count = column_count - column_tiles_count * tmm_rows;
    nk_size_t const total_tiles = column_tiles_count * depth_tiles_count;

    // Write header with layout metadata
    nk_dots_amx_packed_header_t *header = (nk_dots_amx_packed_header_t *)b_packed;
    header->full_column_tiles = (nk_u32_t)column_tiles_count;
    header->full_depth_tiles = (nk_u32_t)depth_tiles_count;
    header->column_remainder_count = (nk_u32_t)column_remainder_count;

    // Compute memory region offsets
    nk_size_t const tiles_offset = sizeof(nk_dots_amx_packed_header_t);
    nk_size_t const column_edge_offset = tiles_offset + total_tiles * tile_bytes;
    header->column_edge_offset = (nk_u32_t)column_edge_offset;

    // Pointers to packed data regions
    nk_i8_t *tiles_ptr = (nk_i8_t *)((char *)b_packed + tiles_offset);
    nk_i8_t *column_edge_ptr = (nk_i8_t *)((char *)b_packed + column_edge_offset);

    // Zero-initialize all tiles (handles depth remainder padding)
    for (nk_size_t idx = 0; idx < total_tiles * tile_elements; idx++) tiles_ptr[idx] = 0;

    // Pack tiles using LINEAR ordering: tile_index = column_tile × depth_tiles_count + depth_tile
    // This provides sequential memory access when streaming along depth dimension.
    for (nk_size_t column_tile_idx = 0; column_tile_idx < column_tiles_count; column_tile_idx++) {
        for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {

            // Linear tile index: all depth-tiles for one column-tile are contiguous
            nk_size_t const tile_index = column_tile_idx * depth_tiles_count + depth_tile_idx;
            nk_i8_t *tile_output = tiles_ptr + tile_index * tile_elements;

            // Source coordinates in original B matrix
            nk_size_t const src_row_start = column_tile_idx * tmm_rows;
            nk_size_t const src_column_start = depth_tile_idx * tmm_cols;
            nk_size_t const columns_to_pack = (src_column_start + tmm_cols <= depth) ? tmm_cols
                                                                                     : (depth - src_column_start);

            // Pack with quad-interleaving as required by TDPBSSD instruction.
            // AMX expects: [col0_row0, col1_row0, col2_row0, col3_row0, col0_row1, ...]
            // Formula: packed_idx = (column / 4) × 64 + row × 4 + (column % 4)
            for (nk_size_t row_idx = 0; row_idx < tmm_rows; row_idx++) {
                for (nk_size_t column_idx = 0; column_idx < columns_to_pack; column_idx++) {
                    nk_size_t const src_idx = (src_row_start + row_idx) * b_stride + src_column_start + column_idx;
                    nk_size_t const dst_idx = (column_idx / 4) * 64 + row_idx * 4 + (column_idx % 4);
                    tile_output[dst_idx] = b[src_idx];
                }
            }
        }
    }

    // Pack column-remainder rows in simple row-major format (for AVX-512 fallback)
    if (column_remainder_count > 0) {
        nk_size_t const remainder_start_row = column_tiles_count * tmm_rows;
        for (nk_size_t row_idx = 0; row_idx < column_remainder_count; row_idx++) {
            for (nk_size_t column_idx = 0; column_idx < depth; column_idx++) {
                column_edge_ptr[row_idx * depth + column_idx] =
                    b[(remainder_start_row + row_idx) * b_stride + column_idx];
            }
        }
    }

    // Compute and store per-column norms for angular/euclidean distance
    nk_size_t norms_offset = column_edge_offset +
                             (column_remainder_count > 0 ? column_remainder_count * depth * sizeof(nk_i8_t) : 0);
    header->norms_byte_offset = (nk_u32_t)norms_offset;
    nk_u32_t *norms = (nk_u32_t *)((char *)b_packed + norms_offset);
    for (nk_size_t col = 0; col < column_count; col++) norms[col] = nk_dots_reduce_sumsq_i8_(b + col * b_stride, depth);
}

NK_PUBLIC void nk_dots_packed_i8_sapphireamx(            //
    nk_i8_t const *a, void const *b_packed, nk_i32_t *c, //
    nk_size_t rows_count, nk_size_t cols_count, nk_size_t depth, nk_size_t a_stride_bytes, nk_size_t c_stride_bytes) {
    nk_unused_(cols_count);

    // Parse packed B header
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_size_t const column_tiles_count = header->full_column_tiles;
    nk_size_t const depth_tiles_count = header->full_depth_tiles;
    nk_size_t const column_remainder_count = header->column_remainder_count;

    // Packed B data regions
    nk_i8_t const *b_tiles_base = (nk_i8_t const *)((char const *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_i8_t const *col_edge_ptr = (nk_i8_t const *)((char const *)b_packed + header->column_edge_offset);

    // Stride conversions
    nk_size_t const c_stride_elements = c_stride_bytes / sizeof(nk_i32_t);

    // Tile dimensions
    nk_size_t const tile_depth = 64;  // depth elements per INT8 tile
    nk_size_t const tile_size = 1024; // bytes per packed tile
    nk_size_t const full_cols = column_tiles_count * 16;

    // Block counts (32 × 32 output blocks = 2 × 2 tiles)
    nk_size_t const row_blocks_count = nk_size_divide_round_up_(rows_count, 32);
    nk_size_t const col_blocks_count = column_tiles_count / 2;

    if (depth_tiles_count == 0) return;

    // Tile buffers for A (only used for edge tiles)
    nk_dots_i8_a16x64_sapphireamx_t a_tile_upper, a_tile_lower;
    nk_dots_i8_state2x2_sapphireamx_t c_accum_buffer;

    // Precompute: number of full depth-tiles (no masking needed)
    nk_size_t const full_depth_tiles_count = depth / tile_depth;
    nk_size_t const depth_remainder = depth % tile_depth;

    nk_amx_tile_configure_sapphireamx_();

    // Process all 32 × 32 row × column blocks (including partial edge blocks)
    for (nk_size_t row_block_idx = 0; row_block_idx < row_blocks_count; row_block_idx++) {
        nk_size_t const row_block_start = row_block_idx * 32;
        nk_size_t const valid_rows_count = (row_block_start + 32 <= rows_count) ? 32 : (rows_count - row_block_start);
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
                nk_i8_t const *a_upper_base = a + row_block_start * a_stride_bytes;
                nk_i8_t const *a_lower_base = a + (row_block_start + 16) * a_stride_bytes;

                // B tile pointers
                nk_dots_i8_b64x16_sapphireamx_t const *b_tile_left =
                    (nk_dots_i8_b64x16_sapphireamx_t const *)(b_tiles_base + b_column_left_base * tile_size);
                nk_dots_i8_b64x16_sapphireamx_t const *b_tile_right =
                    (nk_dots_i8_b64x16_sapphireamx_t const *)(b_tiles_base + b_column_right_base * tile_size);

                // Prologue: load first depth tile into TMM0-3
                _tile_loadd(0, a_upper_base, a_stride_bytes);
                _tile_loadd(1, a_lower_base, a_stride_bytes);
                _tile_loadd(2, b_tile_left->data, 64);
                _tile_loadd(3, b_tile_right->data, 64);

                // Main loop: 2-deep software pipelining (compute current while loading next)
                for (nk_size_t depth_tile_idx = 0; depth_tile_idx < full_depth_tiles_count - 1; depth_tile_idx++) {
                    nk_size_t const next_depth_offset = (depth_tile_idx + 1) * tile_depth;

                    _tile_dpbssd(4, 0, 2);
                    _tile_dpbssd(5, 0, 3);
                    _tile_dpbssd(6, 1, 2);
                    _tile_dpbssd(7, 1, 3);

                    _tile_loadd(0, a_upper_base + next_depth_offset, a_stride_bytes);
                    _tile_loadd(1, a_lower_base + next_depth_offset, a_stride_bytes);
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

                    nk_dots_i8_load_a_sapphireamx_(&a_tile_upper, a_upper_base + depth_offset, a_stride_bytes, 16,
                                                   depth_remainder);
                    nk_dots_i8_load_a_sapphireamx_(&a_tile_lower, a_lower_base + depth_offset, a_stride_bytes, 16,
                                                   depth_remainder);

                    b_tile_left = (nk_dots_i8_b64x16_sapphireamx_t const *)(b_tiles_base + (b_column_left_base +
                                                                                            full_depth_tiles_count) *
                                                                                               tile_size);
                    b_tile_right = (nk_dots_i8_b64x16_sapphireamx_t const *)(b_tiles_base + (b_column_right_base +
                                                                                             full_depth_tiles_count) *
                                                                                                tile_size);

                    _tile_loadd(0, a_tile_upper.data, 64);
                    _tile_loadd(1, a_tile_lower.data, 64);
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
                nk_i8_t const *a_upper_base = a + row_block_start * a_stride_bytes;
                nk_i8_t const *a_lower_base = a + (row_block_start + 16) * a_stride_bytes;

                nk_dots_i8_load_a_sapphireamx_(&a_tile_upper, a_upper_base, a_stride_bytes, 16, depth_remainder);
                nk_dots_i8_load_a_sapphireamx_(&a_tile_lower, a_lower_base, a_stride_bytes, 16, depth_remainder);

                nk_dots_i8_b64x16_sapphireamx_t const *b_tile_left =
                    (nk_dots_i8_b64x16_sapphireamx_t const *)(b_tiles_base + b_column_left_base * tile_size);
                nk_dots_i8_b64x16_sapphireamx_t const *b_tile_right =
                    (nk_dots_i8_b64x16_sapphireamx_t const *)(b_tiles_base + b_column_right_base * tile_size);

                _tile_loadd(0, a_tile_upper.data, 64);
                _tile_loadd(1, a_tile_lower.data, 64);
                _tile_loadd(2, b_tile_left->data, 64);
                _tile_loadd(3, b_tile_right->data, 64);

                _tile_dpbssd(4, 0, 2);
                _tile_dpbssd(5, 0, 3);
                _tile_dpbssd(6, 1, 2);
                _tile_dpbssd(7, 1, 3);
            }
            // Slow path: edge row-block → always use buffered load with masking
            else {
                nk_size_t const rows_in_upper_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
                nk_size_t const rows_in_lower_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

                for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                    nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                    nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth
                                                                                            : depth_remainder;

                    nk_dots_i8_load_a_sapphireamx_(&a_tile_upper, a + row_block_start * a_stride_bytes + depth_offset,
                                                   a_stride_bytes, rows_in_upper_tile, valid_depth);
                    if (rows_in_lower_tile > 0) {
                        nk_dots_i8_load_a_sapphireamx_(&a_tile_lower,
                                                       a + (row_block_start + 16) * a_stride_bytes + depth_offset,
                                                       a_stride_bytes, rows_in_lower_tile, valid_depth);
                    }

                    nk_dots_i8_b64x16_sapphireamx_t const *b_tile_left =
                        (nk_dots_i8_b64x16_sapphireamx_t const *)(b_tiles_base +
                                                                  (b_column_left_base + depth_tile_idx) * tile_size);
                    nk_dots_i8_b64x16_sapphireamx_t const *b_tile_right =
                        (nk_dots_i8_b64x16_sapphireamx_t const *)(b_tiles_base +
                                                                  (b_column_right_base + depth_tile_idx) * tile_size);

                    _tile_loadd(0, a_tile_upper.data, 64);
                    _tile_loadd(1, a_tile_lower.data, 64);
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
                _tile_stored(4, c_block, c_stride_bytes);
                _tile_stored(5, c_block + 16, c_stride_bytes);
                _tile_stored(6, (nk_i32_t *)((char *)c_block + 16 * c_stride_bytes), c_stride_bytes);
                _tile_stored(7, (nk_i32_t *)((char *)c_block + 16 * c_stride_bytes) + 16, c_stride_bytes);
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
            nk_size_t const rows_in_upper_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
            nk_size_t const rows_in_lower_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

            // Use 1 × 2 blocking for single column-tile (2 row-tiles × 1 column-tile)
            nk_dots_i8_state_sapphireamx_t c_upper_state, c_lower_state;

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_i8_load_a_sapphireamx_(&a_tile_upper, a + row_block_start * a_stride_bytes + depth_offset,
                                               a_stride_bytes, rows_in_upper_tile, valid_depth);
                if (rows_in_lower_tile > 0) {
                    nk_dots_i8_load_a_sapphireamx_(&a_tile_lower,
                                                   a + (row_block_start + 16) * a_stride_bytes + depth_offset,
                                                   a_stride_bytes, rows_in_lower_tile, valid_depth);
                }

                nk_dots_i8_b64x16_sapphireamx_t const *b_tile =
                    (nk_dots_i8_b64x16_sapphireamx_t const *)(b_tiles_base +
                                                              (b_column_base + depth_tile_idx) * tile_size);

                _tile_loadd(0, a_tile_upper.data, 64);
                _tile_loadd(1, a_tile_lower.data, 64);
                _tile_loadd(2, b_tile->data, 64);

                _tile_dpbssd(4, 0, 2);
                _tile_dpbssd(6, 1, 2);
            }

            _tile_stored(4, c_upper_state.data, 64);
            _tile_stored(6, c_lower_state.data, 64);

            nk_dots_i8_store_sapphireamx_(&c_upper_state, c + row_block_start * c_stride_elements + col_start,
                                          c_stride_elements, rows_in_upper_tile, 16);
            if (rows_in_lower_tile > 0) {
                nk_dots_i8_store_sapphireamx_(&c_lower_state,
                                              c + (row_block_start + 16) * c_stride_elements + col_start,
                                              c_stride_elements, rows_in_lower_tile, 16);
            }
        }

        // Handle column-edge (remaining columns < 16) using AMX with partial tiles
        if (column_remainder_count > 0) {
            nk_size_t const rows_in_upper_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
            nk_size_t const rows_in_lower_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

            nk_dots_i8_state_sapphireamx_t c_upper_state, c_lower_state;
            nk_dots_i8_a16x64_sapphireamx_t b_as_a;
            nk_dots_i8_b64x16_sapphireamx_t b_tile;

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                // Load A tiles
                nk_dots_i8_load_a_sapphireamx_(&a_tile_upper, a + row_block_start * a_stride_bytes + depth_offset,
                                               a_stride_bytes, rows_in_upper_tile, valid_depth);
                if (rows_in_lower_tile > 0) {
                    nk_dots_i8_load_a_sapphireamx_(&a_tile_lower,
                                                   a + (row_block_start + 16) * a_stride_bytes + depth_offset,
                                                   a_stride_bytes, rows_in_lower_tile, valid_depth);
                }

                // Load B edge data (row-major: b_edge[row × depth + column]) and pack into B tile
                // Each "row" in edge data corresponds to one output column
                nk_dots_i8_load_a_sapphireamx_(&b_as_a, col_edge_ptr + depth_offset, depth, column_remainder_count,
                                               valid_depth);
                nk_dots_pack_i8_transposed_sapphireamx_(&b_as_a, &b_tile);

                _tile_loadd(0, a_tile_upper.data, 64);
                _tile_loadd(1, a_tile_lower.data, 64);
                _tile_loadd(2, b_tile.data, 64);

                _tile_dpbssd(4, 0, 2);
                _tile_dpbssd(6, 1, 2);
            }

            _tile_stored(4, c_upper_state.data, 64);
            _tile_stored(6, c_lower_state.data, 64);

            nk_dots_i8_store_sapphireamx_(&c_upper_state, c + row_block_start * c_stride_elements + full_cols,
                                          c_stride_elements, rows_in_upper_tile, column_remainder_count);
            if (rows_in_lower_tile > 0) {
                nk_dots_i8_store_sapphireamx_(&c_lower_state,
                                              c + (row_block_start + 16) * c_stride_elements + full_cols,
                                              c_stride_elements, rows_in_lower_tile, column_remainder_count);
            }
        }
    }

    _tile_release();
}

NK_PUBLIC void nk_dots_compact_i8_sapphireamx( //
    void *c, nk_size_t row_count, nk_size_t column_count, nk_size_t c_stride, nk_i32_t const *a_squared_norms,
    nk_i32_t const *b_squared_norms) {

    nk_size_t const c_stride_i32 = c_stride / sizeof(nk_i32_t);
    nk_i32_t const *c_i32 = (nk_i32_t const *)c;
    nk_i8_t *c_i8 = (nk_i8_t *)c;

    // Use space after I8 output for precomputed b_rsqrt (I8 output is 4x smaller than I32 input)
    nk_f32_t *b_rsqrt = (nk_f32_t *)(c_i8 + row_count * column_count);

    // Precompute rsqrt of all b_norms using AVX512 (16 at a time)
    __m512 half_vec = _mm512_set1_ps(0.5f);
    __m512 three_halves_vec = _mm512_set1_ps(1.5f);
    nk_size_t column_idx = 0;

    for (; column_idx + 16 <= column_count; column_idx += 16) {
        __m512i b_norms_i32 = _mm512_loadu_si512(b_squared_norms + column_idx);
        __m512 b_norms_f32 = _mm512_cvtepi32_ps(b_norms_i32);
        __m512 rsqrt_vec = _mm512_rsqrt14_ps(b_norms_f32);
        // Newton-Raphson refinement
        rsqrt_vec = _mm512_mul_ps(
            rsqrt_vec,
            _mm512_sub_ps(three_halves_vec,
                          _mm512_mul_ps(half_vec, _mm512_mul_ps(b_norms_f32, _mm512_mul_ps(rsqrt_vec, rsqrt_vec)))));
        // Zero out rsqrt where norm was zero
        __mmask16 nonzero_mask = _mm512_cmpneq_epi32_mask(b_norms_i32, _mm512_setzero_si512());
        rsqrt_vec = _mm512_maskz_mov_ps(nonzero_mask, rsqrt_vec);
        _mm512_storeu_ps(b_rsqrt + column_idx, rsqrt_vec);
    }

    // Handle remaining b_norms with masked operations
    if (column_idx < column_count) {
        __mmask16 tail_mask = (__mmask16)((1u << (column_count - column_idx)) - 1);
        __m512i b_norms_i32 = _mm512_maskz_loadu_epi32(tail_mask, b_squared_norms + column_idx);
        __m512 b_norms_f32 = _mm512_cvtepi32_ps(b_norms_i32);
        __m512 rsqrt_vec = _mm512_rsqrt14_ps(b_norms_f32);
        rsqrt_vec = _mm512_mul_ps(
            rsqrt_vec,
            _mm512_sub_ps(three_halves_vec,
                          _mm512_mul_ps(half_vec, _mm512_mul_ps(b_norms_f32, _mm512_mul_ps(rsqrt_vec, rsqrt_vec)))));
        __mmask16 nonzero_mask = _mm512_cmpneq_epi32_mask(b_norms_i32, _mm512_setzero_si512());
        rsqrt_vec = _mm512_maskz_mov_ps(nonzero_mask & tail_mask, rsqrt_vec);
        _mm512_mask_storeu_ps(b_rsqrt + column_idx, tail_mask, rsqrt_vec);
    }

    __m512 scale_vec = _mm512_set1_ps(127.0f);

    for (nk_size_t row_idx = 0; row_idx < row_count; row_idx++) {
        nk_i32_t const *src_row = c_i32 + row_idx * c_stride_i32;
        nk_i8_t *dst_row = c_i8 + row_idx * column_count;

        // Compute rsqrt of a_norm for this row, broadcast to vector
        nk_f32_t a_norm_f32 = (nk_f32_t)a_squared_norms[row_idx];
        nk_f32_t a_rsqrt_val = 0.0f;
        if (a_norm_f32 > 0.0f) {
            __m128 a_vec = _mm_set_ss(a_norm_f32);
            __m128 rsqrt_s = _mm_rsqrt_ss(a_vec);
            rsqrt_s = _mm_mul_ss(
                rsqrt_s, _mm_sub_ss(_mm_set_ss(1.5f),
                                    _mm_mul_ss(_mm_set_ss(0.5f), _mm_mul_ss(a_vec, _mm_mul_ss(rsqrt_s, rsqrt_s)))));
            a_rsqrt_val = _mm_cvtss_f32(rsqrt_s);
        }
        __m512 a_rsqrt_vec = _mm512_set1_ps(a_rsqrt_val);
        __m512 row_scale = _mm512_mul_ps(a_rsqrt_vec, scale_vec);

        column_idx = 0;

        // Process 16 elements at a time
        for (; column_idx + 16 <= column_count; column_idx += 16) {
            __m512i c_vals = _mm512_loadu_si512(src_row + column_idx);
            __m512 c_f32 = _mm512_cvtepi32_ps(c_vals);
            __m512 b_rsqrt_vec = _mm512_loadu_ps(b_rsqrt + column_idx);
            __m512 normalized = _mm512_mul_ps(_mm512_mul_ps(c_f32, row_scale), b_rsqrt_vec);
            __m512i result_i32 = _mm512_cvtps_epi32(normalized);
            // Saturating pack I32 → I8 (16 values → 16 bytes in low 128 bits)
            __m128i result_i8 = _mm512_cvtsepi32_epi8(result_i32);
            _mm_storeu_si128((__m128i *)(dst_row + column_idx), result_i8);
        }

        // Handle remaining elements with masked operations
        if (column_idx < column_count) {
            __mmask16 tail_mask = (__mmask16)((1u << (column_count - column_idx)) - 1);
            __m512i c_vals = _mm512_maskz_loadu_epi32(tail_mask, src_row + column_idx);
            __m512 c_f32 = _mm512_cvtepi32_ps(c_vals);
            __m512 b_rsqrt_vec = _mm512_maskz_loadu_ps(tail_mask, b_rsqrt + column_idx);
            __m512 normalized = _mm512_mul_ps(_mm512_mul_ps(c_f32, row_scale), b_rsqrt_vec);
            __m512i result_i32 = _mm512_cvtps_epi32(normalized);
            __m128i result_i8 = _mm512_cvtsepi32_epi8(result_i32);
            _mm_mask_storeu_epi8(dst_row + column_idx, tail_mask, result_i8);
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_i8_sapphireamx(                  //
    nk_i8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, //
    nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,  //
    nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const result_stride_elements = result_stride / sizeof(nk_i32_t);

    // Handle row slicing: compute rows [row_start, row_end)
    nk_size_t const row_end = (row_count == 0)
                                  ? n_vectors
                                  : (row_start + row_count < n_vectors ? row_start + row_count : n_vectors);

    // Round depth up to multiple of 192 (3 tiles × 64 elements)
    nk_size_t const depth_tiles = nk_size_divide_round_up_(depth, 64);
    nk_size_t const depth_tile_groups = nk_size_divide_round_up_(depth_tiles, 3);

    nk_dots_i8_a16x64_sapphireamx_t a_tiles[3];
    nk_dots_i8_a16x64_sapphireamx_t b_src_tiles[3];
    nk_dots_i8_b64x16_sapphireamx_t b_tiles[3];
    nk_dots_i8_state_sapphireamx_t state;

    nk_amx_tile_configure_sapphireamx_();

    for (nk_size_t row_tile = row_start; row_tile < row_end; row_tile += 16) {
        nk_size_t const valid_rows = (row_tile + 16 <= row_end) ? 16 : (row_end - row_tile);

        for (nk_size_t col_tile = 0; col_tile < n_vectors; col_tile += 16) {
            nk_size_t const valid_cols = (col_tile + 16 <= n_vectors) ? 16 : (n_vectors - col_tile);

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

                    if (row_tile == col_tile) {
                        nk_dots_pack_i8_transposed_sapphireamx_(&a_tiles[tile_idx], &b_tiles[tile_idx]);
                    }
                    else {
                        nk_dots_i8_load_a_sapphireamx_(                //
                            &b_src_tiles[tile_idx],                    //
                            vectors + col_tile * stride + depth_start, //
                            stride, valid_cols, valid_depth);
                        nk_dots_pack_i8_transposed_sapphireamx_(&b_src_tiles[tile_idx], &b_tiles[tile_idx]);
                    }
                }

                nk_dots_i8_update_sapphireamx_( //
                    &state, &a_tiles[0], &a_tiles[1], &a_tiles[2], &b_tiles[0], &b_tiles[1], &b_tiles[2]);
            }

            nk_dots_i8_store_sapphireamx_(                                     //
                &state, result + row_tile * result_stride_elements + col_tile, //
                result_stride_elements, valid_rows, valid_cols);
        }
    }
}

#pragma endregion // Signed Integers

#pragma region Unsigned Integers

NK_PUBLIC nk_size_t nk_dots_packed_size_u8_sapphireamx(nk_size_t column_count, nk_size_t depth) {
    // Same layout as I8 - just different type interpretation
    return nk_dots_packed_size_i8_sapphireamx(column_count, depth);
}

NK_PUBLIC void nk_dots_pack_u8_sapphireamx(                    //
    nk_u8_t const *b, nk_size_t column_count, nk_size_t depth, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const tmm_rows = 16;
    nk_size_t const tmm_cols = 64;
    nk_size_t const tile_elements = 1024;
    nk_size_t const tile_bytes = tile_elements * sizeof(nk_u8_t);

    nk_size_t const column_tiles_count = column_count / tmm_rows;
    nk_size_t const depth_tiles_count = nk_size_divide_round_up_(depth, tmm_cols);
    nk_size_t const column_remainder_count = column_count - column_tiles_count * tmm_rows;
    nk_size_t const total_tiles = column_tiles_count * depth_tiles_count;

    nk_dots_amx_packed_header_t *header = (nk_dots_amx_packed_header_t *)b_packed;
    header->full_column_tiles = (nk_u32_t)column_tiles_count;
    header->full_depth_tiles = (nk_u32_t)depth_tiles_count;
    header->column_remainder_count = (nk_u32_t)column_remainder_count;

    nk_size_t const tiles_offset = sizeof(nk_dots_amx_packed_header_t);
    nk_size_t const column_edge_offset = tiles_offset + total_tiles * tile_bytes;
    header->column_edge_offset = (nk_u32_t)column_edge_offset;

    nk_u8_t *tiles_ptr = (nk_u8_t *)((char *)b_packed + tiles_offset);
    nk_u8_t *column_edge_ptr = (nk_u8_t *)((char *)b_packed + column_edge_offset);

    for (nk_size_t idx = 0; idx < total_tiles * tile_elements; idx++) tiles_ptr[idx] = 0;

    for (nk_size_t column_tile_idx = 0; column_tile_idx < column_tiles_count; column_tile_idx++) {
        for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {

            nk_size_t const tile_index = column_tile_idx * depth_tiles_count + depth_tile_idx;
            nk_u8_t *tile_output = tiles_ptr + tile_index * tile_elements;

            nk_size_t const src_row_start = column_tile_idx * tmm_rows;
            nk_size_t const src_column_start = depth_tile_idx * tmm_cols;
            nk_size_t const columns_to_pack = (src_column_start + tmm_cols <= depth) ? tmm_cols
                                                                                     : (depth - src_column_start);

            // Pack with quad-interleaving as required by TDPBUUD instruction.
            for (nk_size_t row_idx = 0; row_idx < tmm_rows; row_idx++) {
                for (nk_size_t column_idx = 0; column_idx < columns_to_pack; column_idx++) {
                    nk_size_t const src_idx = (src_row_start + row_idx) * b_stride + src_column_start + column_idx;
                    nk_size_t const dst_idx = (column_idx / 4) * 64 + row_idx * 4 + (column_idx % 4);
                    tile_output[dst_idx] = b[src_idx];
                }
            }
        }
    }

    if (column_remainder_count > 0) {
        nk_size_t const remainder_start_row = column_tiles_count * tmm_rows;
        for (nk_size_t row_idx = 0; row_idx < column_remainder_count; row_idx++) {
            for (nk_size_t column_idx = 0; column_idx < depth; column_idx++) {
                column_edge_ptr[row_idx * depth + column_idx] =
                    b[(remainder_start_row + row_idx) * b_stride + column_idx];
            }
        }
    }

    // Compute and store per-column norms for angular/euclidean distance
    nk_size_t norms_offset = column_edge_offset +
                             (column_remainder_count > 0 ? column_remainder_count * depth * sizeof(nk_u8_t) : 0);
    header->norms_byte_offset = (nk_u32_t)norms_offset;
    nk_u32_t *norms = (nk_u32_t *)((char *)b_packed + norms_offset);
    for (nk_size_t col = 0; col < column_count; col++) norms[col] = nk_dots_reduce_sumsq_u8_(b + col * b_stride, depth);
}

NK_PUBLIC void nk_dots_packed_u8_sapphireamx(            //
    nk_u8_t const *a, void const *b_packed, nk_u32_t *c, //
    nk_size_t rows_count, nk_size_t cols_count, nk_size_t depth, nk_size_t a_stride_bytes, nk_size_t c_stride_bytes) {
    nk_unused_(cols_count);

    // Parse packed B header
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_size_t const column_tiles_count = header->full_column_tiles;
    nk_size_t const depth_tiles_count = header->full_depth_tiles;
    nk_size_t const column_remainder_count = header->column_remainder_count;

    // Packed B data regions
    nk_u8_t const *b_tiles_base = (nk_u8_t const *)((char const *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_u8_t const *col_edge_ptr = (nk_u8_t const *)((char const *)b_packed + header->column_edge_offset);

    // Stride conversions
    nk_size_t const c_stride_elements = c_stride_bytes / sizeof(nk_u32_t);

    // Tile dimensions
    nk_size_t const tile_depth = 64;  // depth elements per U8 tile
    nk_size_t const tile_size = 1024; // bytes per packed tile
    nk_size_t const full_cols = column_tiles_count * 16;

    // Block counts (32 × 32 output blocks = 2 × 2 tiles)
    nk_size_t const row_blocks_count = nk_size_divide_round_up_(rows_count, 32);
    nk_size_t const col_blocks_count = column_tiles_count / 2;

    if (depth_tiles_count == 0) return;

    // Tile buffers for A (only used for edge tiles)
    nk_dots_u8_a16x64_sapphireamx_t a_tile_upper, a_tile_lower;
    nk_dots_u8_state2x2_sapphireamx_t c_accum_buffer;

    // Precompute: number of full depth-tiles
    nk_size_t const full_depth_tiles_count = depth / tile_depth;
    nk_size_t const depth_remainder = depth % tile_depth;

    nk_amx_tile_configure_sapphireamx_();

    // Process all 32 × 32 row × column blocks
    for (nk_size_t row_block_idx = 0; row_block_idx < row_blocks_count; row_block_idx++) {
        nk_size_t const row_block_start = row_block_idx * 32;
        nk_size_t const valid_rows_count = (row_block_start + 32 <= rows_count) ? 32 : (rows_count - row_block_start);
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
                nk_u8_t const *a_upper_base = a + row_block_start * a_stride_bytes;
                nk_u8_t const *a_lower_base = a + (row_block_start + 16) * a_stride_bytes;

                nk_dots_u8_b64x16_sapphireamx_t const *b_tile_left =
                    (nk_dots_u8_b64x16_sapphireamx_t const *)(b_tiles_base + b_column_left_base * tile_size);
                nk_dots_u8_b64x16_sapphireamx_t const *b_tile_right =
                    (nk_dots_u8_b64x16_sapphireamx_t const *)(b_tiles_base + b_column_right_base * tile_size);

                // Prologue: load first depth tile into TMM0-3
                _tile_loadd(0, a_upper_base, a_stride_bytes);
                _tile_loadd(1, a_lower_base, a_stride_bytes);
                _tile_loadd(2, b_tile_left->data, 64);
                _tile_loadd(3, b_tile_right->data, 64);

                // Main loop: 2-deep software pipelining (compute current while loading next)
                for (nk_size_t depth_tile_idx = 0; depth_tile_idx < full_depth_tiles_count - 1; depth_tile_idx++) {
                    nk_size_t const next_depth_offset = (depth_tile_idx + 1) * tile_depth;

                    _tile_dpbuud(4, 0, 2);
                    _tile_dpbuud(5, 0, 3);
                    _tile_dpbuud(6, 1, 2);
                    _tile_dpbuud(7, 1, 3);

                    _tile_loadd(0, a_upper_base + next_depth_offset, a_stride_bytes);
                    _tile_loadd(1, a_lower_base + next_depth_offset, a_stride_bytes);
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

                    nk_dots_u8_load_a_sapphireamx_(&a_tile_upper, a_upper_base + depth_offset, a_stride_bytes, 16,
                                                   depth_remainder);
                    nk_dots_u8_load_a_sapphireamx_(&a_tile_lower, a_lower_base + depth_offset, a_stride_bytes, 16,
                                                   depth_remainder);

                    b_tile_left = (nk_dots_u8_b64x16_sapphireamx_t const *)(b_tiles_base + (b_column_left_base +
                                                                                            full_depth_tiles_count) *
                                                                                               tile_size);
                    b_tile_right = (nk_dots_u8_b64x16_sapphireamx_t const *)(b_tiles_base + (b_column_right_base +
                                                                                             full_depth_tiles_count) *
                                                                                                tile_size);

                    _tile_loadd(0, a_tile_upper.data, 64);
                    _tile_loadd(1, a_tile_lower.data, 64);
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
                nk_u8_t const *a_upper_base = a + row_block_start * a_stride_bytes;
                nk_u8_t const *a_lower_base = a + (row_block_start + 16) * a_stride_bytes;

                nk_dots_u8_load_a_sapphireamx_(&a_tile_upper, a_upper_base, a_stride_bytes, 16, depth_remainder);
                nk_dots_u8_load_a_sapphireamx_(&a_tile_lower, a_lower_base, a_stride_bytes, 16, depth_remainder);

                nk_dots_u8_b64x16_sapphireamx_t const *b_tile_left =
                    (nk_dots_u8_b64x16_sapphireamx_t const *)(b_tiles_base + b_column_left_base * tile_size);
                nk_dots_u8_b64x16_sapphireamx_t const *b_tile_right =
                    (nk_dots_u8_b64x16_sapphireamx_t const *)(b_tiles_base + b_column_right_base * tile_size);

                _tile_loadd(0, a_tile_upper.data, 64);
                _tile_loadd(1, a_tile_lower.data, 64);
                _tile_loadd(2, b_tile_left->data, 64);
                _tile_loadd(3, b_tile_right->data, 64);

                _tile_dpbuud(4, 0, 2);
                _tile_dpbuud(5, 0, 3);
                _tile_dpbuud(6, 1, 2);
                _tile_dpbuud(7, 1, 3);
            }
            // Slow path: edge row-block → always use buffered load
            else {
                nk_size_t const rows_in_upper_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
                nk_size_t const rows_in_lower_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

                for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                    nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                    nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth
                                                                                            : depth_remainder;

                    nk_dots_u8_load_a_sapphireamx_(&a_tile_upper, a + row_block_start * a_stride_bytes + depth_offset,
                                                   a_stride_bytes, rows_in_upper_tile, valid_depth);
                    if (rows_in_lower_tile > 0) {
                        nk_dots_u8_load_a_sapphireamx_(&a_tile_lower,
                                                       a + (row_block_start + 16) * a_stride_bytes + depth_offset,
                                                       a_stride_bytes, rows_in_lower_tile, valid_depth);
                    }

                    nk_dots_u8_b64x16_sapphireamx_t const *b_tile_left =
                        (nk_dots_u8_b64x16_sapphireamx_t const *)(b_tiles_base +
                                                                  (b_column_left_base + depth_tile_idx) * tile_size);
                    nk_dots_u8_b64x16_sapphireamx_t const *b_tile_right =
                        (nk_dots_u8_b64x16_sapphireamx_t const *)(b_tiles_base +
                                                                  (b_column_right_base + depth_tile_idx) * tile_size);

                    _tile_loadd(0, a_tile_upper.data, 64);
                    _tile_loadd(1, a_tile_lower.data, 64);
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
                _tile_stored(4, c_block, c_stride_bytes);
                _tile_stored(5, c_block + 16, c_stride_bytes);
                _tile_stored(6, (nk_u32_t *)((char *)c_block + 16 * c_stride_bytes), c_stride_bytes);
                _tile_stored(7, (nk_u32_t *)((char *)c_block + 16 * c_stride_bytes) + 16, c_stride_bytes);
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
            nk_size_t const rows_in_upper_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
            nk_size_t const rows_in_lower_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

            nk_dots_u8_state_sapphireamx_t c_upper_state, c_lower_state;

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_u8_load_a_sapphireamx_(&a_tile_upper, a + row_block_start * a_stride_bytes + depth_offset,
                                               a_stride_bytes, rows_in_upper_tile, valid_depth);
                if (rows_in_lower_tile > 0) {
                    nk_dots_u8_load_a_sapphireamx_(&a_tile_lower,
                                                   a + (row_block_start + 16) * a_stride_bytes + depth_offset,
                                                   a_stride_bytes, rows_in_lower_tile, valid_depth);
                }

                nk_dots_u8_b64x16_sapphireamx_t const *b_tile =
                    (nk_dots_u8_b64x16_sapphireamx_t const *)(b_tiles_base +
                                                              (b_column_base + depth_tile_idx) * tile_size);

                _tile_loadd(0, a_tile_upper.data, 64);
                _tile_loadd(1, a_tile_lower.data, 64);
                _tile_loadd(2, b_tile->data, 64);

                _tile_dpbuud(4, 0, 2);
                _tile_dpbuud(6, 1, 2);
            }

            _tile_stored(4, c_upper_state.data, 64);
            _tile_stored(6, c_lower_state.data, 64);

            nk_dots_u8_store_sapphireamx_(&c_upper_state, c + row_block_start * c_stride_elements + col_start,
                                          c_stride_elements, rows_in_upper_tile, 16);
            if (rows_in_lower_tile > 0) {
                nk_dots_u8_store_sapphireamx_(&c_lower_state,
                                              c + (row_block_start + 16) * c_stride_elements + col_start,
                                              c_stride_elements, rows_in_lower_tile, 16);
            }
        }

        // Handle column-edge (remaining columns < 16) using AMX with partial tiles
        if (column_remainder_count > 0) {
            nk_size_t const rows_in_upper_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
            nk_size_t const rows_in_lower_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

            nk_dots_u8_state_sapphireamx_t c_upper_state, c_lower_state;
            nk_dots_u8_a16x64_sapphireamx_t b_as_a;
            nk_dots_u8_b64x16_sapphireamx_t b_tile;

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_u8_load_a_sapphireamx_(&a_tile_upper, a + row_block_start * a_stride_bytes + depth_offset,
                                               a_stride_bytes, rows_in_upper_tile, valid_depth);
                if (rows_in_lower_tile > 0) {
                    nk_dots_u8_load_a_sapphireamx_(&a_tile_lower,
                                                   a + (row_block_start + 16) * a_stride_bytes + depth_offset,
                                                   a_stride_bytes, rows_in_lower_tile, valid_depth);
                }

                nk_dots_u8_load_a_sapphireamx_(&b_as_a, col_edge_ptr + depth_offset, depth, column_remainder_count,
                                               valid_depth);
                nk_dots_pack_u8_transposed_sapphireamx_(&b_as_a, &b_tile);

                _tile_loadd(0, a_tile_upper.data, 64);
                _tile_loadd(1, a_tile_lower.data, 64);
                _tile_loadd(2, b_tile.data, 64);

                _tile_dpbuud(4, 0, 2);
                _tile_dpbuud(6, 1, 2);
            }

            _tile_stored(4, c_upper_state.data, 64);
            _tile_stored(6, c_lower_state.data, 64);

            nk_dots_u8_store_sapphireamx_(&c_upper_state, c + row_block_start * c_stride_elements + full_cols,
                                          c_stride_elements, rows_in_upper_tile, column_remainder_count);
            if (rows_in_lower_tile > 0) {
                nk_dots_u8_store_sapphireamx_(&c_lower_state,
                                              c + (row_block_start + 16) * c_stride_elements + full_cols,
                                              c_stride_elements, rows_in_lower_tile, column_remainder_count);
            }
        }
    }

    _tile_release();
}

NK_PUBLIC void nk_dots_symmetric_u8_sapphireamx(                  //
    nk_u8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, //
    nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,  //
    nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const result_stride_elements = result_stride / sizeof(nk_u32_t);

    // Handle row slicing: compute rows [row_start, row_end)
    nk_size_t const row_end = (row_count == 0)
                                  ? n_vectors
                                  : (row_start + row_count < n_vectors ? row_start + row_count : n_vectors);

    // Round depth up to multiple of 192 (3 tiles × 64 elements)
    nk_size_t const depth_tiles = nk_size_divide_round_up_(depth, 64);
    nk_size_t const depth_tile_groups = nk_size_divide_round_up_(depth_tiles, 3);

    nk_dots_u8_a16x64_sapphireamx_t a_tiles[3];
    nk_dots_u8_a16x64_sapphireamx_t b_src_tiles[3];
    nk_dots_u8_b64x16_sapphireamx_t b_tiles[3];
    nk_dots_u8_state_sapphireamx_t state;

    nk_amx_tile_configure_sapphireamx_();

    for (nk_size_t row_tile = row_start; row_tile < row_end; row_tile += 16) {
        nk_size_t const valid_rows = (row_tile + 16 <= row_end) ? 16 : (row_end - row_tile);

        for (nk_size_t col_tile = 0; col_tile < n_vectors; col_tile += 16) {
            nk_size_t const valid_cols = (col_tile + 16 <= n_vectors) ? 16 : (n_vectors - col_tile);

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

                    if (row_tile == col_tile) {
                        nk_dots_pack_u8_transposed_sapphireamx_(&a_tiles[tile_idx], &b_tiles[tile_idx]);
                    }
                    else {
                        nk_dots_u8_load_a_sapphireamx_(                //
                            &b_src_tiles[tile_idx],                    //
                            vectors + col_tile * stride + depth_start, //
                            stride, valid_cols, valid_depth);
                        nk_dots_pack_u8_transposed_sapphireamx_(&b_src_tiles[tile_idx], &b_tiles[tile_idx]);
                    }
                }

                nk_dots_u8_update_sapphireamx_( //
                    &state, &a_tiles[0], &a_tiles[1], &a_tiles[2], &b_tiles[0], &b_tiles[1], &b_tiles[2]);
            }

            nk_dots_u8_store_sapphireamx_(                                     //
                &state, result + row_tile * result_stride_elements + col_tile, //
                result_stride_elements, valid_rows, valid_cols);
        }
    }
}

#pragma endregion // Unsigned Integers

#pragma region Quarter Precision E4M3

NK_PUBLIC nk_size_t nk_dots_packed_size_e4m3_sapphireamx(nk_size_t column_count, nk_size_t depth) {
    // FP8 uses BF16 tile layout after conversion (same element count: 32 per row)
    return nk_dots_packed_size_bf16_sapphireamx(column_count, depth);
}

NK_PUBLIC void nk_dots_pack_e4m3_sapphireamx(                    //
    nk_e4m3_t const *b, nk_size_t column_count, nk_size_t depth, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const tmm_rows = 16;
    nk_size_t const tmm_cols = 32; // Same depth granularity as BF16
    nk_size_t const tile_elements = 512;
    nk_size_t const tile_bytes = tile_elements * sizeof(nk_bf16_t);

    nk_size_t const column_tiles_count = column_count / tmm_rows;
    nk_size_t const depth_tiles_count = nk_size_divide_round_up_(depth, tmm_cols);
    nk_size_t const column_remainder_count = column_count - column_tiles_count * tmm_rows;
    nk_size_t const total_tiles = column_tiles_count * depth_tiles_count;

    nk_dots_amx_packed_header_t *header = (nk_dots_amx_packed_header_t *)b_packed;
    header->full_column_tiles = (nk_u32_t)column_tiles_count;
    header->full_depth_tiles = (nk_u32_t)depth_tiles_count;
    header->column_remainder_count = (nk_u32_t)column_remainder_count;

    nk_size_t const tiles_offset = sizeof(nk_dots_amx_packed_header_t);
    nk_size_t const column_edge_offset = tiles_offset + total_tiles * tile_bytes;
    header->column_edge_offset = (nk_u32_t)column_edge_offset;

    nk_bf16_t *tiles_ptr = (nk_bf16_t *)((char *)b_packed + tiles_offset);
    nk_bf16_t *column_edge_ptr = (nk_bf16_t *)((char *)b_packed + column_edge_offset);

    for (nk_size_t idx = 0; idx < total_tiles * tile_elements; idx++) tiles_ptr[idx] = 0;

    for (nk_size_t column_tile_idx = 0; column_tile_idx < column_tiles_count; column_tile_idx++) {
        for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
            nk_size_t const tile_index = column_tile_idx * depth_tiles_count + depth_tile_idx;
            nk_bf16_t *tile_output = tiles_ptr + tile_index * tile_elements;

            nk_size_t const src_row_start = column_tile_idx * tmm_rows;
            nk_size_t const src_column_start = depth_tile_idx * tmm_cols;
            nk_size_t const columns_to_pack = (src_column_start + tmm_cols <= depth) ? tmm_cols
                                                                                     : (depth - src_column_start);

            // Convert E4M3 to BF16 and pack with pair-interleaving
            for (nk_size_t row_idx = 0; row_idx < tmm_rows; row_idx++) {
                nk_size_t src_row = src_row_start + row_idx;
                // Load 32 E4M3 bytes and convert to BF16
                __mmask32 column_mask = (columns_to_pack >= 32) ? 0xFFFFFFFF : ((__mmask32)1 << columns_to_pack) - 1;
                __m256i e4m3_row = _mm256_maskz_loadu_epi8(column_mask, b + src_row * b_stride + src_column_start);
                __m512i bf16_row = nk_e4m3x32_to_bf16x32_icelake_(e4m3_row);
                // Store with pair-interleaving
                nk_bf16_t bf16_buf[32];
                _mm512_storeu_si512((__m512i *)bf16_buf, bf16_row);
                for (nk_size_t column_idx = 0; column_idx < columns_to_pack; column_idx++) {
                    nk_size_t const dst_idx = (column_idx / 2) * 32 + row_idx * 2 + (column_idx % 2);
                    tile_output[dst_idx] = bf16_buf[column_idx];
                }
            }
        }
    }

    // Pack column-remainder rows (convert E4M3 to BF16)
    if (column_remainder_count > 0) {
        nk_size_t const remainder_start_row = column_tiles_count * tmm_rows;
        for (nk_size_t row_idx = 0; row_idx < column_remainder_count; row_idx++) {
            for (nk_size_t column_idx = 0; column_idx < depth; column_idx += 32) {
                nk_size_t columns = (column_idx + 32 <= depth) ? 32 : (depth - column_idx);
                __mmask32 column_mask = (columns >= 32) ? 0xFFFFFFFF : ((__mmask32)1 << columns) - 1;
                __m256i e4m3_chunk = _mm256_maskz_loadu_epi8(
                    column_mask, b + (remainder_start_row + row_idx) * b_stride + column_idx);
                __m512i bf16_chunk = nk_e4m3x32_to_bf16x32_icelake_(e4m3_chunk);
                _mm512_mask_storeu_epi16(column_edge_ptr + row_idx * depth + column_idx, column_mask, bf16_chunk);
            }
        }
    }

    // Compute and store per-column norms for angular/euclidean distance
    nk_size_t norms_offset = column_edge_offset +
                             (column_remainder_count > 0 ? column_remainder_count * depth * sizeof(nk_bf16_t) : 0);
    header->norms_byte_offset = (nk_u32_t)norms_offset;
    nk_f32_t *norms = (nk_f32_t *)((char *)b_packed + norms_offset);
    for (nk_size_t col = 0; col < column_count; col++)
        norms[col] = nk_dots_reduce_sumsq_e4m3_(b + col * b_stride, depth);
}

NK_PUBLIC void nk_dots_packed_e4m3_sapphireamx(            //
    nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows_count, nk_size_t cols_count, nk_size_t depth, nk_size_t a_stride_bytes, nk_size_t c_stride_bytes) {
    nk_unused_(cols_count);

    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_size_t const column_tiles_count = header->full_column_tiles;
    nk_size_t const depth_tiles_count = header->full_depth_tiles;
    nk_size_t const column_remainder_count = header->column_remainder_count;

    // B tiles are already in BF16 format
    nk_bf16_t const *b_tiles_base = (nk_bf16_t const *)((char const *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_bf16_t const *col_edge_ptr = (nk_bf16_t const *)((char const *)b_packed + header->column_edge_offset);

    nk_size_t const c_stride_elements = c_stride_bytes / sizeof(nk_f32_t);
    nk_size_t const tile_depth = 32;
    nk_size_t const tile_size = 512;
    nk_size_t const full_cols = column_tiles_count * 16;

    nk_size_t const row_blocks_count = nk_size_divide_round_up_(rows_count, 32);
    nk_size_t const col_blocks_count = column_tiles_count / 2;

    if (depth_tiles_count == 0) return;

    nk_dots_bf16_a16x32_sapphireamx_t a_tile_upper, a_tile_lower;
    nk_dots_bf16_state2x2_sapphireamx_t c_accum_buffer;

    nk_size_t const full_depth_tiles_count = depth / tile_depth;
    nk_size_t const depth_remainder = depth % tile_depth;

    nk_amx_tile_configure_sapphireamx_();

    // Loop order: row_blocks outer, col_blocks inner
    for (nk_size_t row_block_idx = 0; row_block_idx < row_blocks_count; row_block_idx++) {
        nk_size_t const row_block_start = row_block_idx * 32;
        nk_size_t const valid_rows_count = (row_block_start + 32 <= rows_count) ? 32 : (rows_count - row_block_start);
        nk_size_t const is_full_row_block = (valid_rows_count == 32);
        nk_size_t const rows_in_upper_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
        nk_size_t const rows_in_lower_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

        for (nk_size_t column_block_idx = 0; column_block_idx < col_blocks_count; column_block_idx++) {
            nk_size_t const col_block_start = column_block_idx * 32;
            nk_size_t const b_column_left_base = (column_block_idx * 2) * depth_tiles_count;
            nk_size_t const b_column_right_base = (column_block_idx * 2 + 1) * depth_tiles_count;

            // Zero accumulators (TMM4-7 stay resident across entire depth loop)
            _tile_zero(4);
            _tile_zero(5);
            _tile_zero(6);
            _tile_zero(7);

            // FP8 always uses buffered load for E4M3 → BF16 conversion
            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                // Load A with FP8 → BF16 conversion
                nk_dots_e4m3_load_a_sapphireamx_(&a_tile_upper, a + row_block_start * a_stride_bytes + depth_offset,
                                                 a_stride_bytes, rows_in_upper_tile, valid_depth);
                if (rows_in_lower_tile > 0) {
                    nk_dots_e4m3_load_a_sapphireamx_(&a_tile_lower,
                                                     a + (row_block_start + 16) * a_stride_bytes + depth_offset,
                                                     a_stride_bytes, rows_in_lower_tile, valid_depth);
                }

                nk_dots_bf16_b32x16_sapphireamx_t const *b_tile_left =
                    (nk_dots_bf16_b32x16_sapphireamx_t const *)(b_tiles_base +
                                                                (b_column_left_base + depth_tile_idx) * tile_size);
                nk_dots_bf16_b32x16_sapphireamx_t const *b_tile_right =
                    (nk_dots_bf16_b32x16_sapphireamx_t const *)(b_tiles_base +
                                                                (b_column_right_base + depth_tile_idx) * tile_size);

                _tile_loadd(0, a_tile_upper.data, 64);
                _tile_loadd(1, a_tile_lower.data, 64);
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
                _tile_stored(4, c_block, c_stride_bytes);
                _tile_stored(5, c_block + 16, c_stride_bytes);
                _tile_stored(6, (nk_f32_t *)((char *)c_block + 16 * c_stride_bytes), c_stride_bytes);
                _tile_stored(7, (nk_f32_t *)((char *)c_block + 16 * c_stride_bytes) + 16, c_stride_bytes);
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

            nk_dots_bf16_state_sapphireamx_t c_upper_state, c_lower_state;
            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_e4m3_load_a_sapphireamx_(&a_tile_upper, a + row_block_start * a_stride_bytes + depth_offset,
                                                 a_stride_bytes, rows_in_upper_tile, valid_depth);
                if (rows_in_lower_tile > 0) {
                    nk_dots_e4m3_load_a_sapphireamx_(&a_tile_lower,
                                                     a + (row_block_start + 16) * a_stride_bytes + depth_offset,
                                                     a_stride_bytes, rows_in_lower_tile, valid_depth);
                }

                nk_dots_bf16_b32x16_sapphireamx_t const *b_tile =
                    (nk_dots_bf16_b32x16_sapphireamx_t const *)(b_tiles_base +
                                                                (b_column_base + depth_tile_idx) * tile_size);

                _tile_loadd(0, a_tile_upper.data, 64);
                _tile_loadd(1, a_tile_lower.data, 64);
                _tile_loadd(2, b_tile->data, 64);

                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(6, 1, 2);
            }

            _tile_stored(4, c_upper_state.data, 64);
            _tile_stored(6, c_lower_state.data, 64);

            nk_dots_bf16_store_sapphireamx_(&c_upper_state, c + row_block_start * c_stride_elements + col_start,
                                            c_stride_elements, rows_in_upper_tile, 16);
            if (rows_in_lower_tile > 0) {
                nk_dots_bf16_store_sapphireamx_(&c_lower_state,
                                                c + (row_block_start + 16) * c_stride_elements + col_start,
                                                c_stride_elements, rows_in_lower_tile, 16);
            }
        }

        // Handle column-edge (remaining columns < 16) using AMX with partial tiles
        if (column_remainder_count > 0) {
            nk_dots_bf16_state_sapphireamx_t c_upper_state, c_lower_state;
            nk_dots_bf16_a16x32_sapphireamx_t b_as_a;
            nk_dots_bf16_b32x16_sapphireamx_t b_tile;

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_e4m3_load_a_sapphireamx_(&a_tile_upper, a + row_block_start * a_stride_bytes + depth_offset,
                                                 a_stride_bytes, rows_in_upper_tile, valid_depth);
                if (rows_in_lower_tile > 0) {
                    nk_dots_e4m3_load_a_sapphireamx_(&a_tile_lower,
                                                     a + (row_block_start + 16) * a_stride_bytes + depth_offset,
                                                     a_stride_bytes, rows_in_lower_tile, valid_depth);
                }

                // B edge data is already in BF16 format
                nk_dots_bf16_load_a_sapphireamx_(&b_as_a, col_edge_ptr + depth_offset, depth, column_remainder_count,
                                                 valid_depth);
                nk_dots_pack_bf16_transposed_sapphireamx_(&b_as_a, &b_tile);

                _tile_loadd(0, a_tile_upper.data, 64);
                _tile_loadd(1, a_tile_lower.data, 64);
                _tile_loadd(2, b_tile.data, 64);

                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(6, 1, 2);
            }

            _tile_stored(4, c_upper_state.data, 64);
            _tile_stored(6, c_lower_state.data, 64);

            nk_dots_bf16_store_sapphireamx_(&c_upper_state, c + row_block_start * c_stride_elements + full_cols,
                                            c_stride_elements, rows_in_upper_tile, column_remainder_count);
            if (rows_in_lower_tile > 0) {
                nk_dots_bf16_store_sapphireamx_(&c_lower_state,
                                                c + (row_block_start + 16) * c_stride_elements + full_cols,
                                                c_stride_elements, rows_in_lower_tile, column_remainder_count);
            }
        }
    }

    _tile_release();
}

#pragma endregion // Quarter Precision E4M3

#pragma region Quarter Precision E5M2

NK_PUBLIC nk_size_t nk_dots_packed_size_e5m2_sapphireamx(nk_size_t column_count, nk_size_t depth) {
    return nk_dots_packed_size_bf16_sapphireamx(column_count, depth);
}

NK_PUBLIC void nk_dots_pack_e5m2_sapphireamx(                    //
    nk_e5m2_t const *b, nk_size_t column_count, nk_size_t depth, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const tmm_rows = 16;
    nk_size_t const tmm_cols = 32;
    nk_size_t const tile_elements = 512;
    nk_size_t const tile_bytes = tile_elements * sizeof(nk_bf16_t);

    nk_size_t const column_tiles_count = column_count / tmm_rows;
    nk_size_t const depth_tiles_count = nk_size_divide_round_up_(depth, tmm_cols);
    nk_size_t const column_remainder_count = column_count - column_tiles_count * tmm_rows;
    nk_size_t const total_tiles = column_tiles_count * depth_tiles_count;

    nk_dots_amx_packed_header_t *header = (nk_dots_amx_packed_header_t *)b_packed;
    header->full_column_tiles = (nk_u32_t)column_tiles_count;
    header->full_depth_tiles = (nk_u32_t)depth_tiles_count;
    header->column_remainder_count = (nk_u32_t)column_remainder_count;

    nk_size_t const tiles_offset = sizeof(nk_dots_amx_packed_header_t);
    nk_size_t const column_edge_offset = tiles_offset + total_tiles * tile_bytes;
    header->column_edge_offset = (nk_u32_t)column_edge_offset;

    nk_bf16_t *tiles_ptr = (nk_bf16_t *)((char *)b_packed + tiles_offset);
    nk_bf16_t *column_edge_ptr = (nk_bf16_t *)((char *)b_packed + column_edge_offset);

    for (nk_size_t idx = 0; idx < total_tiles * tile_elements; idx++) tiles_ptr[idx] = 0;

    for (nk_size_t column_tile_idx = 0; column_tile_idx < column_tiles_count; column_tile_idx++) {
        for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
            nk_size_t const tile_index = column_tile_idx * depth_tiles_count + depth_tile_idx;
            nk_bf16_t *tile_output = tiles_ptr + tile_index * tile_elements;

            nk_size_t const src_row_start = column_tile_idx * tmm_rows;
            nk_size_t const src_column_start = depth_tile_idx * tmm_cols;
            nk_size_t const columns_to_pack = (src_column_start + tmm_cols <= depth) ? tmm_cols
                                                                                     : (depth - src_column_start);

            for (nk_size_t row_idx = 0; row_idx < tmm_rows; row_idx++) {
                nk_size_t src_row = src_row_start + row_idx;
                __mmask32 column_mask = (columns_to_pack >= 32) ? 0xFFFFFFFF : ((__mmask32)1 << columns_to_pack) - 1;
                __m256i e5m2_row = _mm256_maskz_loadu_epi8(column_mask, b + src_row * b_stride + src_column_start);
                __m512i bf16_row = nk_e5m2x32_to_bf16x32_icelake_(e5m2_row);
                nk_bf16_t bf16_buf[32];
                _mm512_storeu_si512((__m512i *)bf16_buf, bf16_row);
                for (nk_size_t column_idx = 0; column_idx < columns_to_pack; column_idx++) {
                    nk_size_t const dst_idx = (column_idx / 2) * 32 + row_idx * 2 + (column_idx % 2);
                    tile_output[dst_idx] = bf16_buf[column_idx];
                }
            }
        }
    }

    if (column_remainder_count > 0) {
        nk_size_t const remainder_start_row = column_tiles_count * tmm_rows;
        for (nk_size_t row_idx = 0; row_idx < column_remainder_count; row_idx++) {
            for (nk_size_t column_idx = 0; column_idx < depth; column_idx += 32) {
                nk_size_t columns = (column_idx + 32 <= depth) ? 32 : (depth - column_idx);
                __mmask32 column_mask = (columns >= 32) ? 0xFFFFFFFF : ((__mmask32)1 << columns) - 1;
                __m256i e5m2_chunk = _mm256_maskz_loadu_epi8(
                    column_mask, b + (remainder_start_row + row_idx) * b_stride + column_idx);
                __m512i bf16_chunk = nk_e5m2x32_to_bf16x32_icelake_(e5m2_chunk);
                _mm512_mask_storeu_epi16(column_edge_ptr + row_idx * depth + column_idx, column_mask, bf16_chunk);
            }
        }
    }

    // Compute and store per-column norms for angular/euclidean distance
    nk_size_t norms_offset = column_edge_offset +
                             (column_remainder_count > 0 ? column_remainder_count * depth * sizeof(nk_bf16_t) : 0);
    header->norms_byte_offset = (nk_u32_t)norms_offset;
    nk_f32_t *norms = (nk_f32_t *)((char *)b_packed + norms_offset);
    for (nk_size_t col = 0; col < column_count; col++)
        norms[col] = nk_dots_reduce_sumsq_e5m2_(b + col * b_stride, depth);
}

NK_PUBLIC void nk_dots_packed_e5m2_sapphireamx(            //
    nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows_count, nk_size_t cols_count, nk_size_t depth, nk_size_t a_stride_bytes, nk_size_t c_stride_bytes) {
    nk_unused_(cols_count);

    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_size_t const column_tiles_count = header->full_column_tiles;
    nk_size_t const depth_tiles_count = header->full_depth_tiles;
    nk_size_t const column_remainder_count = header->column_remainder_count;

    nk_bf16_t const *b_tiles_base = (nk_bf16_t const *)((char const *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_bf16_t const *col_edge_ptr = (nk_bf16_t const *)((char const *)b_packed + header->column_edge_offset);

    nk_size_t const c_stride_elements = c_stride_bytes / sizeof(nk_f32_t);
    nk_size_t const tile_depth = 32;
    nk_size_t const tile_size = 512;
    nk_size_t const full_cols = column_tiles_count * 16;

    nk_size_t const row_blocks_count = nk_size_divide_round_up_(rows_count, 32);
    nk_size_t const col_blocks_count = column_tiles_count / 2;

    if (depth_tiles_count == 0) return;

    nk_dots_bf16_a16x32_sapphireamx_t a_tile_upper, a_tile_lower;
    nk_dots_bf16_state2x2_sapphireamx_t c_accum_buffer;

    nk_size_t const full_depth_tiles_count = depth / tile_depth;
    nk_size_t const depth_remainder = depth % tile_depth;

    nk_amx_tile_configure_sapphireamx_();

    // Loop order: row_blocks outer, col_blocks inner
    for (nk_size_t row_block_idx = 0; row_block_idx < row_blocks_count; row_block_idx++) {
        nk_size_t const row_block_start = row_block_idx * 32;
        nk_size_t const valid_rows_count = (row_block_start + 32 <= rows_count) ? 32 : (rows_count - row_block_start);
        nk_size_t const is_full_row_block = (valid_rows_count == 32);
        nk_size_t const rows_in_upper_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
        nk_size_t const rows_in_lower_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

        for (nk_size_t column_block_idx = 0; column_block_idx < col_blocks_count; column_block_idx++) {
            nk_size_t const col_block_start = column_block_idx * 32;
            nk_size_t const b_column_left_base = (column_block_idx * 2) * depth_tiles_count;
            nk_size_t const b_column_right_base = (column_block_idx * 2 + 1) * depth_tiles_count;

            // Zero accumulators (TMM4-7 stay resident across entire depth loop)
            _tile_zero(4);
            _tile_zero(5);
            _tile_zero(6);
            _tile_zero(7);

            // FP8 always uses buffered load for E5M2 → BF16 conversion
            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                // Load A with FP8 → BF16 conversion
                nk_dots_e5m2_load_a_sapphireamx_(&a_tile_upper, a + row_block_start * a_stride_bytes + depth_offset,
                                                 a_stride_bytes, rows_in_upper_tile, valid_depth);
                if (rows_in_lower_tile > 0) {
                    nk_dots_e5m2_load_a_sapphireamx_(&a_tile_lower,
                                                     a + (row_block_start + 16) * a_stride_bytes + depth_offset,
                                                     a_stride_bytes, rows_in_lower_tile, valid_depth);
                }

                nk_dots_bf16_b32x16_sapphireamx_t const *b_tile_left =
                    (nk_dots_bf16_b32x16_sapphireamx_t const *)(b_tiles_base +
                                                                (b_column_left_base + depth_tile_idx) * tile_size);
                nk_dots_bf16_b32x16_sapphireamx_t const *b_tile_right =
                    (nk_dots_bf16_b32x16_sapphireamx_t const *)(b_tiles_base +
                                                                (b_column_right_base + depth_tile_idx) * tile_size);

                _tile_loadd(0, a_tile_upper.data, 64);
                _tile_loadd(1, a_tile_lower.data, 64);
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
                _tile_stored(4, c_block, c_stride_bytes);
                _tile_stored(5, c_block + 16, c_stride_bytes);
                _tile_stored(6, (nk_f32_t *)((char *)c_block + 16 * c_stride_bytes), c_stride_bytes);
                _tile_stored(7, (nk_f32_t *)((char *)c_block + 16 * c_stride_bytes) + 16, c_stride_bytes);
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

            nk_dots_bf16_state_sapphireamx_t c_upper_state, c_lower_state;
            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_e5m2_load_a_sapphireamx_(&a_tile_upper, a + row_block_start * a_stride_bytes + depth_offset,
                                                 a_stride_bytes, rows_in_upper_tile, valid_depth);
                if (rows_in_lower_tile > 0) {
                    nk_dots_e5m2_load_a_sapphireamx_(&a_tile_lower,
                                                     a + (row_block_start + 16) * a_stride_bytes + depth_offset,
                                                     a_stride_bytes, rows_in_lower_tile, valid_depth);
                }

                nk_dots_bf16_b32x16_sapphireamx_t const *b_tile =
                    (nk_dots_bf16_b32x16_sapphireamx_t const *)(b_tiles_base +
                                                                (b_column_base + depth_tile_idx) * tile_size);

                _tile_loadd(0, a_tile_upper.data, 64);
                _tile_loadd(1, a_tile_lower.data, 64);
                _tile_loadd(2, b_tile->data, 64);

                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(6, 1, 2);
            }

            _tile_stored(4, c_upper_state.data, 64);
            _tile_stored(6, c_lower_state.data, 64);

            nk_dots_bf16_store_sapphireamx_(&c_upper_state, c + row_block_start * c_stride_elements + col_start,
                                            c_stride_elements, rows_in_upper_tile, 16);
            if (rows_in_lower_tile > 0) {
                nk_dots_bf16_store_sapphireamx_(&c_lower_state,
                                                c + (row_block_start + 16) * c_stride_elements + col_start,
                                                c_stride_elements, rows_in_lower_tile, 16);
            }
        }

        // Handle column-edge (remaining columns < 16) using AMX with partial tiles
        if (column_remainder_count > 0) {
            nk_dots_bf16_state_sapphireamx_t c_upper_state, c_lower_state;
            nk_dots_bf16_a16x32_sapphireamx_t b_as_a;
            nk_dots_bf16_b32x16_sapphireamx_t b_tile;

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_e5m2_load_a_sapphireamx_(&a_tile_upper, a + row_block_start * a_stride_bytes + depth_offset,
                                                 a_stride_bytes, rows_in_upper_tile, valid_depth);
                if (rows_in_lower_tile > 0) {
                    nk_dots_e5m2_load_a_sapphireamx_(&a_tile_lower,
                                                     a + (row_block_start + 16) * a_stride_bytes + depth_offset,
                                                     a_stride_bytes, rows_in_lower_tile, valid_depth);
                }

                nk_dots_bf16_load_a_sapphireamx_(&b_as_a, col_edge_ptr + depth_offset, depth, column_remainder_count,
                                                 valid_depth);
                nk_dots_pack_bf16_transposed_sapphireamx_(&b_as_a, &b_tile);

                _tile_loadd(0, a_tile_upper.data, 64);
                _tile_loadd(1, a_tile_lower.data, 64);
                _tile_loadd(2, b_tile.data, 64);

                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(6, 1, 2);
            }

            _tile_stored(4, c_upper_state.data, 64);
            _tile_stored(6, c_lower_state.data, 64);

            nk_dots_bf16_store_sapphireamx_(&c_upper_state, c + row_block_start * c_stride_elements + full_cols,
                                            c_stride_elements, rows_in_upper_tile, column_remainder_count);
            if (rows_in_lower_tile > 0) {
                nk_dots_bf16_store_sapphireamx_(&c_lower_state,
                                                c + (row_block_start + 16) * c_stride_elements + full_cols,
                                                c_stride_elements, rows_in_lower_tile, column_remainder_count);
            }
        }
    }

    _tile_release();
}

NK_PUBLIC void nk_dots_symmetric_e5m2_sapphireamx(                  //
    nk_e5m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, //
    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,    //
    nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);

    // Handle row slicing: compute rows [row_start, row_end)
    nk_size_t const row_end = (row_count == 0)
                                  ? n_vectors
                                  : (row_start + row_count < n_vectors ? row_start + row_count : n_vectors);

    // Round depth up to multiple of 96 (3 tiles × 32 elements)
    nk_size_t const depth_tiles = nk_size_divide_round_up_(depth, 32);
    nk_size_t const depth_tile_groups = nk_size_divide_round_up_(depth_tiles, 3);

    nk_dots_bf16_a16x32_sapphireamx_t a_tiles[3];
    nk_dots_bf16_a16x32_sapphireamx_t b_src_tiles[3];
    nk_dots_bf16_b32x16_sapphireamx_t b_tiles[3];
    nk_dots_bf16_state_sapphireamx_t state;

    nk_amx_tile_configure_sapphireamx_();

    for (nk_size_t row_tile = row_start; row_tile < row_end; row_tile += 16) {
        nk_size_t const valid_rows = (row_tile + 16 <= row_end) ? 16 : (row_end - row_tile);

        for (nk_size_t col_tile = 0; col_tile < n_vectors; col_tile += 16) {
            nk_size_t const valid_cols = (col_tile + 16 <= n_vectors) ? 16 : (n_vectors - col_tile);

            nk_dots_bf16_init_sapphireamx_(&state);

            for (nk_size_t depth_group_idx = 0; depth_group_idx < depth_tile_groups; depth_group_idx++) {
                nk_size_t const depth_base = depth_group_idx * 96;

                for (int tile_idx = 0; tile_idx < 3; tile_idx++) {
                    nk_size_t const depth_start = depth_base + tile_idx * 32;
                    nk_size_t const valid_depth = (depth_start + 32 <= depth)
                                                      ? 32
                                                      : (depth > depth_start ? depth - depth_start : 0);

                    nk_dots_e5m2_load_a_sapphireamx_(              //
                        &a_tiles[tile_idx],                        //
                        vectors + row_tile * stride + depth_start, //
                        stride, valid_rows, valid_depth);

                    if (row_tile == col_tile) {
                        nk_dots_pack_bf16_transposed_sapphireamx_(&a_tiles[tile_idx], &b_tiles[tile_idx]);
                    }
                    else {
                        nk_dots_e5m2_load_a_sapphireamx_(              //
                            &b_src_tiles[tile_idx],                    //
                            vectors + col_tile * stride + depth_start, //
                            stride, valid_cols, valid_depth);
                        nk_dots_pack_bf16_transposed_sapphireamx_(&b_src_tiles[tile_idx], &b_tiles[tile_idx]);
                    }
                }

                nk_dots_bf16_update_sapphireamx_( //
                    &state, &a_tiles[0], &a_tiles[1], &a_tiles[2], &b_tiles[0], &b_tiles[1], &b_tiles[2]);
            }

            nk_dots_bf16_store_sapphireamx_(                                   //
                &state, result + row_tile * result_stride_elements + col_tile, //
                result_stride_elements, valid_rows, valid_cols);
        }
    }
}

NK_PUBLIC void nk_dots_symmetric_e4m3_sapphireamx(                  //
    nk_e4m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth, //
    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,    //
    nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);

    // Handle row slicing: compute rows [row_start, row_end)
    nk_size_t const row_end = (row_count == 0)
                                  ? n_vectors
                                  : (row_start + row_count < n_vectors ? row_start + row_count : n_vectors);

    // Round depth up to multiple of 96 (3 tiles × 32 elements)
    nk_size_t const depth_tiles = nk_size_divide_round_up_(depth, 32);
    nk_size_t const depth_tile_groups = nk_size_divide_round_up_(depth_tiles, 3);

    nk_dots_bf16_a16x32_sapphireamx_t a_tiles[3];
    nk_dots_bf16_a16x32_sapphireamx_t b_src_tiles[3];
    nk_dots_bf16_b32x16_sapphireamx_t b_tiles[3];
    nk_dots_bf16_state_sapphireamx_t state;

    nk_amx_tile_configure_sapphireamx_();

    for (nk_size_t row_tile = row_start; row_tile < row_end; row_tile += 16) {
        nk_size_t const valid_rows = (row_tile + 16 <= row_end) ? 16 : (row_end - row_tile);

        for (nk_size_t col_tile = 0; col_tile < n_vectors; col_tile += 16) {
            nk_size_t const valid_cols = (col_tile + 16 <= n_vectors) ? 16 : (n_vectors - col_tile);

            nk_dots_bf16_init_sapphireamx_(&state);

            for (nk_size_t depth_group_idx = 0; depth_group_idx < depth_tile_groups; depth_group_idx++) {
                nk_size_t const depth_base = depth_group_idx * 96;

                for (int tile_idx = 0; tile_idx < 3; tile_idx++) {
                    nk_size_t const depth_start = depth_base + tile_idx * 32;
                    nk_size_t const valid_depth = (depth_start + 32 <= depth)
                                                      ? 32
                                                      : (depth > depth_start ? depth - depth_start : 0);

                    nk_dots_e4m3_load_a_sapphireamx_(              //
                        &a_tiles[tile_idx],                        //
                        vectors + row_tile * stride + depth_start, //
                        stride, valid_rows, valid_depth);

                    if (row_tile == col_tile) {
                        nk_dots_pack_bf16_transposed_sapphireamx_(&a_tiles[tile_idx], &b_tiles[tile_idx]);
                    }
                    else {
                        nk_dots_e4m3_load_a_sapphireamx_(              //
                            &b_src_tiles[tile_idx],                    //
                            vectors + col_tile * stride + depth_start, //
                            stride, valid_cols, valid_depth);
                        nk_dots_pack_bf16_transposed_sapphireamx_(&b_src_tiles[tile_idx], &b_tiles[tile_idx]);
                    }
                }

                nk_dots_bf16_update_sapphireamx_( //
                    &state, &a_tiles[0], &a_tiles[1], &a_tiles[2], &b_tiles[0], &b_tiles[1], &b_tiles[2]);
            }

            nk_dots_bf16_store_sapphireamx_(                                   //
                &state, result + row_tile * result_stride_elements + col_tile, //
                result_stride_elements, valid_rows, valid_cols);
        }
    }
}

#pragma endregion // Quarter Precision E5M2

#pragma region Micro Precision E2M3

/* Load E2M3 A tile with E2M3 to signed I8 conversion via VPERMB LUT.
 * Each E2M3 byte encodes: bit 5 = sign, bits 4:0 = magnitude (5-bit index).
 * The LUT maps 5-bit magnitude to value * 16, then sign is applied via conditional negation.
 * Result is stored in INT8 tile for use with _tile_dpbssd.
 */
NK_INTERNAL void nk_dots_e2m3_load_a_sapphireamx_( //
    nk_dots_i8_a16x64_sapphireamx_t *a_tile,       //
    nk_e2m3_t const *src, nk_size_t src_stride,    //
    nk_size_t valid_rows, nk_size_t valid_cols) {

    // Build 64-byte LUT for VPERMB: 32 entries replicated to fill both halves.
    // magnitude → value×16:
    //  e=0 (step 2): {0,2,4,6,8,10,12,14},
    //  e=1 (step 2): {16,18,20,22,24,26,28,30},
    //  e=2 (step 4): {32,36,40,44,48,52,56,60},
    //  e=3 (step 8): {64,72,80,88,96,104,112,120}
    NK_ALIGN64 static nk_u8_t const lut_bytes[64] = {
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

    __mmask64 column_mask = (valid_cols >= 64) ? 0xFFFFFFFFFFFFFFFFULL : ((__mmask64)1 << valid_cols) - 1;

    for (nk_size_t row = 0; row < 16; row++) {
        if (row < valid_rows) {
            __m512i raw_u8x64 = _mm512_maskz_loadu_epi8(column_mask, src + row * src_stride);
            __m512i magnitude_u8x64 = _mm512_and_si512(raw_u8x64, magnitude_mask_u8x64);
            __m512i unsigned_value_u8x64 = _mm512_permutexvar_epi8(magnitude_u8x64, magnitude_lut_u8x64);
            __mmask64 negate_mask = _mm512_test_epi8_mask(raw_u8x64, sign_mask_u8x64);
            __m512i signed_value_i8x64 = _mm512_mask_sub_epi8(unsigned_value_u8x64, negate_mask, zero_i8x64,
                                                              unsigned_value_u8x64);
            _mm512_store_si512(a_tile->data[row], signed_value_i8x64);
        }
        else { _mm512_store_si512(a_tile->data[row], zero_i8x64); }
    }
    nk_compiler_barrier_sapphireamx_();
}

/* Store E2M3 accumulator: read I32 state, convert to F32, multiply by 1/256, store as F32. */
NK_INTERNAL void nk_dots_e2m3_store_sapphireamx_( //
    nk_dots_i8_state_sapphireamx_t const *state,  //
    nk_f32_t *dst, nk_size_t dst_stride_elements, //
    nk_size_t valid_rows, nk_size_t valid_cols) {

    __mmask16 column_mask = (valid_cols >= 16) ? 0xFFFF : ((__mmask16)1 << valid_cols) - 1;
    __m512 scale = _mm512_set1_ps(1.0f / 256.0f);

    for (nk_size_t row = 0; row < valid_rows; row++) {
        __m512i i32_row = _mm512_load_si512(state->data[row]);
        __m512 f32_row = _mm512_mul_ps(_mm512_cvtepi32_ps(i32_row), scale);
        _mm512_mask_storeu_ps(dst + row * dst_stride_elements, column_mask, f32_row);
    }
}

/* Store E2M3 2x2 accumulator state to F32 output matrix with masking for edge tiles. */
NK_INTERNAL void nk_dots_e2m3_output2x2_sapphireamx_( //
    nk_dots_i8_state2x2_sapphireamx_t const *state,   //
    nk_f32_t *dst, nk_size_t dst_stride_elements,     //
    nk_size_t valid_rows, nk_size_t valid_cols) {

    nk_size_t const rows_upper = (valid_rows > 16) ? 16 : valid_rows;
    nk_size_t const cols_left = (valid_cols > 16) ? 16 : valid_cols;
    nk_size_t const cols_right = (valid_cols > 16) ? valid_cols - 16 : 0;

    if (rows_upper > 0 && cols_left > 0)
        nk_dots_e2m3_store_sapphireamx_(&state->c[0][0], dst, dst_stride_elements, rows_upper, cols_left);
    if (rows_upper > 0 && cols_right > 0)
        nk_dots_e2m3_store_sapphireamx_(&state->c[0][1], dst + 16, dst_stride_elements, rows_upper, cols_right);

    if (valid_rows > 16) {
        nk_size_t const rows_lower = valid_rows - 16;
        nk_f32_t *dst_lower = dst + 16 * dst_stride_elements;
        if (cols_left > 0)
            nk_dots_e2m3_store_sapphireamx_(&state->c[1][0], dst_lower, dst_stride_elements, rows_lower, cols_left);
        if (cols_right > 0)
            nk_dots_e2m3_store_sapphireamx_(&state->c[1][1], dst_lower + 16, dst_stride_elements, rows_lower,
                                            cols_right);
    }
}

NK_PUBLIC nk_size_t nk_dots_packed_size_e2m3_sapphireamx(nk_size_t column_count, nk_size_t depth) {
    // E2M3 uses INT8 tile layout after conversion (same element count: 64 per row)
    return nk_dots_packed_size_i8_sapphireamx(column_count, depth);
}

NK_PUBLIC void nk_dots_pack_e2m3_sapphireamx(                    //
    nk_e2m3_t const *b, nk_size_t column_count, nk_size_t depth, //
    nk_size_t b_stride, void *b_packed) {

    // AMX I8 tile dimensions: 16 rows x 64 columns (1024 I8 elements = 1KB)
    nk_size_t const tmm_rows = 16;
    nk_size_t const tmm_cols = 64;
    nk_size_t const tile_elements = 1024;
    nk_size_t const tile_bytes = tile_elements * sizeof(nk_i8_t);

    nk_size_t const column_tiles_count = column_count / tmm_rows;
    nk_size_t const depth_tiles_count = nk_size_divide_round_up_(depth, tmm_cols);
    nk_size_t const column_remainder_count = column_count - column_tiles_count * tmm_rows;
    nk_size_t const total_tiles = column_tiles_count * depth_tiles_count;

    nk_dots_amx_packed_header_t *header = (nk_dots_amx_packed_header_t *)b_packed;
    header->full_column_tiles = (nk_u32_t)column_tiles_count;
    header->full_depth_tiles = (nk_u32_t)depth_tiles_count;
    header->column_remainder_count = (nk_u32_t)column_remainder_count;

    nk_size_t const tiles_offset = sizeof(nk_dots_amx_packed_header_t);
    nk_size_t const column_edge_offset = tiles_offset + total_tiles * tile_bytes;
    header->column_edge_offset = (nk_u32_t)column_edge_offset;

    nk_i8_t *tiles_ptr = (nk_i8_t *)((char *)b_packed + tiles_offset);
    nk_i8_t *column_edge_ptr = (nk_i8_t *)((char *)b_packed + column_edge_offset);

    // Zero-initialize all tiles (handles depth remainder padding)
    for (nk_size_t idx = 0; idx < total_tiles * tile_elements; idx++) tiles_ptr[idx] = 0;

    // E2M3 magnitude-to-value LUT (value * 16)
    static nk_u8_t const lut_magnitude[32] = {
        0,  2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26,  28,  30,  //
        32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120, //
    };

    // Pack tiles with E2M3 -> I8 conversion and quad-interleaving
    for (nk_size_t column_tile_idx = 0; column_tile_idx < column_tiles_count; column_tile_idx++) {
        for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
            nk_size_t const tile_index = column_tile_idx * depth_tiles_count + depth_tile_idx;
            nk_i8_t *tile_output = tiles_ptr + tile_index * tile_elements;

            nk_size_t const src_row_start = column_tile_idx * tmm_rows;
            nk_size_t const src_column_start = depth_tile_idx * tmm_cols;
            nk_size_t const columns_to_pack = (src_column_start + tmm_cols <= depth) ? tmm_cols
                                                                                     : (depth - src_column_start);

            for (nk_size_t row_idx = 0; row_idx < tmm_rows; row_idx++) {
                for (nk_size_t column_idx = 0; column_idx < columns_to_pack; column_idx++) {
                    nk_size_t const src_idx = (src_row_start + row_idx) * b_stride + src_column_start + column_idx;
                    nk_size_t const dst_idx = (column_idx / 4) * 64 + row_idx * 4 + (column_idx % 4);
                    nk_u8_t raw = b[src_idx];
                    nk_u8_t magnitude = raw & 0x1F;
                    nk_i8_t val = (nk_i8_t)lut_magnitude[magnitude];
                    if (raw & 0x20) val = -val;
                    tile_output[dst_idx] = val;
                }
            }
        }
    }

    // Pack column-remainder rows (convert E2M3 to I8)
    if (column_remainder_count > 0) {
        nk_size_t const remainder_start_row = column_tiles_count * tmm_rows;
        for (nk_size_t row_idx = 0; row_idx < column_remainder_count; row_idx++) {
            for (nk_size_t column_idx = 0; column_idx < depth; column_idx++) {
                nk_u8_t raw = b[(remainder_start_row + row_idx) * b_stride + column_idx];
                nk_u8_t magnitude = raw & 0x1F;
                nk_i8_t val = (nk_i8_t)lut_magnitude[magnitude];
                if (raw & 0x20) val = -val;
                column_edge_ptr[row_idx * depth + column_idx] = val;
            }
        }
    }

    // Compute and store per-column norms for angular/euclidean distance
    nk_size_t norms_offset = column_edge_offset +
                             (column_remainder_count > 0 ? column_remainder_count * depth * sizeof(nk_i8_t) : 0);
    header->norms_byte_offset = (nk_u32_t)norms_offset;
    nk_f32_t *norms = (nk_f32_t *)((char *)b_packed + norms_offset);
    for (nk_size_t col = 0; col < column_count; col++)
        norms[col] = nk_dots_reduce_sumsq_e2m3_(b + col * b_stride, depth);
}

NK_PUBLIC void nk_dots_packed_e2m3_sapphireamx(            //
    nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows_count, nk_size_t cols_count, nk_size_t depth, nk_size_t a_stride_bytes, nk_size_t c_stride_bytes) {
    nk_unused_(cols_count);

    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_size_t const column_tiles_count = header->full_column_tiles;
    nk_size_t const depth_tiles_count = header->full_depth_tiles;
    nk_size_t const column_remainder_count = header->column_remainder_count;

    // B tiles are already in I8 format
    nk_i8_t const *b_tiles_base = (nk_i8_t const *)((char const *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_i8_t const *col_edge_ptr = (nk_i8_t const *)((char const *)b_packed + header->column_edge_offset);

    nk_size_t const c_stride_elements = c_stride_bytes / sizeof(nk_f32_t);
    nk_size_t const tile_depth = 64;
    nk_size_t const tile_size = 1024;
    nk_size_t const full_cols = column_tiles_count * 16;

    nk_size_t const row_blocks_count = nk_size_divide_round_up_(rows_count, 32);
    nk_size_t const col_blocks_count = column_tiles_count / 2;

    if (depth_tiles_count == 0) return;

    nk_dots_i8_a16x64_sapphireamx_t a_tile_upper, a_tile_lower;
    nk_dots_i8_state2x2_sapphireamx_t c_accum_buffer;

    nk_size_t const full_depth_tiles_count = depth / tile_depth;
    nk_size_t const depth_remainder = depth % tile_depth;

    nk_amx_tile_configure_sapphireamx_();

    // Loop order: row_blocks outer, col_blocks inner
    for (nk_size_t row_block_idx = 0; row_block_idx < row_blocks_count; row_block_idx++) {
        nk_size_t const row_block_start = row_block_idx * 32;
        nk_size_t const valid_rows_count = (row_block_start + 32 <= rows_count) ? 32 : (rows_count - row_block_start);
        nk_size_t const is_full_row_block = (valid_rows_count == 32);
        nk_size_t const rows_in_upper_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
        nk_size_t const rows_in_lower_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

        for (nk_size_t column_block_idx = 0; column_block_idx < col_blocks_count; column_block_idx++) {
            nk_size_t const col_block_start = column_block_idx * 32;
            nk_size_t const b_column_left_base = (column_block_idx * 2) * depth_tiles_count;
            nk_size_t const b_column_right_base = (column_block_idx * 2 + 1) * depth_tiles_count;

            // Zero accumulators (TMM4-7 stay resident across entire depth loop)
            _tile_zero(4);
            _tile_zero(5);
            _tile_zero(6);
            _tile_zero(7);

            // E2M3 always uses buffered load for E2M3 -> I8 conversion
            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                // Load A with E2M3 -> I8 conversion
                nk_dots_e2m3_load_a_sapphireamx_(&a_tile_upper, a + row_block_start * a_stride_bytes + depth_offset,
                                                 a_stride_bytes, rows_in_upper_tile, valid_depth);
                if (rows_in_lower_tile > 0) {
                    nk_dots_e2m3_load_a_sapphireamx_(&a_tile_lower,
                                                     a + (row_block_start + 16) * a_stride_bytes + depth_offset,
                                                     a_stride_bytes, rows_in_lower_tile, valid_depth);
                }

                nk_dots_i8_b64x16_sapphireamx_t const *b_tile_left =
                    (nk_dots_i8_b64x16_sapphireamx_t const *)(b_tiles_base +
                                                              (b_column_left_base + depth_tile_idx) * tile_size);
                nk_dots_i8_b64x16_sapphireamx_t const *b_tile_right =
                    (nk_dots_i8_b64x16_sapphireamx_t const *)(b_tiles_base +
                                                              (b_column_right_base + depth_tile_idx) * tile_size);

                _tile_loadd(0, a_tile_upper.data, 64);
                _tile_loadd(1, a_tile_lower.data, 64);
                _tile_loadd(2, b_tile_left->data, 64);
                _tile_loadd(3, b_tile_right->data, 64);

                _tile_dpbssd(4, 0, 2);
                _tile_dpbssd(5, 0, 3);
                _tile_dpbssd(6, 1, 2);
                _tile_dpbssd(7, 1, 3);
            }

            // Store accumulators to output (once per output block)
            // Can't directly store I32 tiles to F32 output, must buffer + convert
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

            nk_dots_i8_state_sapphireamx_t c_upper_state, c_lower_state;
            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_e2m3_load_a_sapphireamx_(&a_tile_upper, a + row_block_start * a_stride_bytes + depth_offset,
                                                 a_stride_bytes, rows_in_upper_tile, valid_depth);
                if (rows_in_lower_tile > 0) {
                    nk_dots_e2m3_load_a_sapphireamx_(&a_tile_lower,
                                                     a + (row_block_start + 16) * a_stride_bytes + depth_offset,
                                                     a_stride_bytes, rows_in_lower_tile, valid_depth);
                }

                nk_dots_i8_b64x16_sapphireamx_t const *b_tile =
                    (nk_dots_i8_b64x16_sapphireamx_t const *)(b_tiles_base +
                                                              (b_column_base + depth_tile_idx) * tile_size);

                _tile_loadd(0, a_tile_upper.data, 64);
                _tile_loadd(1, a_tile_lower.data, 64);
                _tile_loadd(2, b_tile->data, 64);

                _tile_dpbssd(4, 0, 2);
                _tile_dpbssd(6, 1, 2);
            }

            _tile_stored(4, c_upper_state.data, 64);
            _tile_stored(6, c_lower_state.data, 64);

            nk_dots_e2m3_store_sapphireamx_(&c_upper_state, c + row_block_start * c_stride_elements + col_start,
                                            c_stride_elements, rows_in_upper_tile, 16);
            if (rows_in_lower_tile > 0) {
                nk_dots_e2m3_store_sapphireamx_(&c_lower_state,
                                                c + (row_block_start + 16) * c_stride_elements + col_start,
                                                c_stride_elements, rows_in_lower_tile, 16);
            }
        }

        // Handle column-edge (remaining columns < 16) using AMX with partial tiles
        if (column_remainder_count > 0) {
            nk_dots_i8_state_sapphireamx_t c_upper_state, c_lower_state;
            nk_dots_i8_a16x64_sapphireamx_t b_as_a;
            nk_dots_i8_b64x16_sapphireamx_t b_tile;

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_e2m3_load_a_sapphireamx_(&a_tile_upper, a + row_block_start * a_stride_bytes + depth_offset,
                                                 a_stride_bytes, rows_in_upper_tile, valid_depth);
                if (rows_in_lower_tile > 0) {
                    nk_dots_e2m3_load_a_sapphireamx_(&a_tile_lower,
                                                     a + (row_block_start + 16) * a_stride_bytes + depth_offset,
                                                     a_stride_bytes, rows_in_lower_tile, valid_depth);
                }

                // B edge data is already in I8 format
                nk_dots_i8_load_a_sapphireamx_(&b_as_a, col_edge_ptr + depth_offset, depth, column_remainder_count,
                                               valid_depth);
                nk_dots_pack_i8_transposed_sapphireamx_(&b_as_a, &b_tile);

                _tile_loadd(0, a_tile_upper.data, 64);
                _tile_loadd(1, a_tile_lower.data, 64);
                _tile_loadd(2, b_tile.data, 64);

                _tile_dpbssd(4, 0, 2);
                _tile_dpbssd(6, 1, 2);
            }

            _tile_stored(4, c_upper_state.data, 64);
            _tile_stored(6, c_lower_state.data, 64);

            nk_dots_e2m3_store_sapphireamx_(&c_upper_state, c + row_block_start * c_stride_elements + full_cols,
                                            c_stride_elements, rows_in_upper_tile, column_remainder_count);
            if (rows_in_lower_tile > 0) {
                nk_dots_e2m3_store_sapphireamx_(&c_lower_state,
                                                c + (row_block_start + 16) * c_stride_elements + full_cols,
                                                c_stride_elements, rows_in_lower_tile, column_remainder_count);
            }
        }
    }

    _tile_release();
}

NK_PUBLIC void nk_dots_symmetric_e2m3_sapphireamx(                  //
    nk_e2m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth, //
    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,    //
    nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);

    // Handle row slicing: compute rows [row_start, row_end)
    nk_size_t const row_end = (row_count == 0)
                                  ? n_vectors
                                  : (row_start + row_count < n_vectors ? row_start + row_count : n_vectors);

    // Round depth up to multiple of 192 (3 tiles x 64 elements)
    nk_size_t const depth_tiles = nk_size_divide_round_up_(depth, 64);
    nk_size_t const depth_tile_groups = nk_size_divide_round_up_(depth_tiles, 3);

    nk_dots_i8_a16x64_sapphireamx_t a_tiles[3];
    nk_dots_i8_a16x64_sapphireamx_t b_src_tiles[3];
    nk_dots_i8_b64x16_sapphireamx_t b_tiles[3];
    nk_dots_i8_state_sapphireamx_t state;

    nk_amx_tile_configure_sapphireamx_();

    for (nk_size_t row_tile = row_start; row_tile < row_end; row_tile += 16) {
        nk_size_t const valid_rows = (row_tile + 16 <= row_end) ? 16 : (row_end - row_tile);

        for (nk_size_t col_tile = 0; col_tile < n_vectors; col_tile += 16) {
            nk_size_t const valid_cols = (col_tile + 16 <= n_vectors) ? 16 : (n_vectors - col_tile);

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

                    if (row_tile == col_tile) {
                        nk_dots_pack_i8_transposed_sapphireamx_(&a_tiles[tile_idx], &b_tiles[tile_idx]);
                    }
                    else {
                        nk_dots_e2m3_load_a_sapphireamx_(              //
                            &b_src_tiles[tile_idx],                    //
                            vectors + col_tile * stride + depth_start, //
                            stride, valid_cols, valid_depth);
                        nk_dots_pack_i8_transposed_sapphireamx_(&b_src_tiles[tile_idx], &b_tiles[tile_idx]);
                    }
                }

                nk_dots_i8_update_sapphireamx_( //
                    &state, &a_tiles[0], &a_tiles[1], &a_tiles[2], &b_tiles[0], &b_tiles[1], &b_tiles[2]);
            }

            nk_dots_e2m3_store_sapphireamx_(                                   //
                &state, result + row_tile * result_stride_elements + col_tile, //
                result_stride_elements, valid_rows, valid_cols);
        }
    }
}

#pragma endregion // Micro Precision E2M3

#pragma region Micro Precision E3M2

/* Load E3M2 A tile with FP8 to BF16 conversion */
NK_INTERNAL void nk_dots_e3m2_load_a_sapphireamx_( //
    nk_dots_bf16_a16x32_sapphireamx_t *a_tile,     //
    nk_e3m2_t const *src, nk_size_t src_stride,    //
    nk_size_t valid_rows, nk_size_t valid_cols) {

    __mmask32 column_mask = (valid_cols >= 32) ? 0xFFFFFFFF : ((__mmask32)1 << valid_cols) - 1;
    __m512i zero = _mm512_setzero_si512();

    for (nk_size_t row_idx = 0; row_idx < 16; row_idx++) {
        if (row_idx < valid_rows) {
            __m256i e3m2_row = _mm256_maskz_loadu_epi8(column_mask, src + row_idx * src_stride);
            __m512i bf16_row = nk_e3m2x32_to_bf16x32_icelake_(e3m2_row);
            _mm512_store_si512((__m512i *)a_tile->data[row_idx], bf16_row);
        }
        else { _mm512_store_si512((__m512i *)a_tile->data[row_idx], zero); }
    }
    nk_compiler_barrier_sapphireamx_();
}

NK_PUBLIC nk_size_t nk_dots_packed_size_e3m2_sapphireamx(nk_size_t column_count, nk_size_t depth) {
    return nk_dots_packed_size_bf16_sapphireamx(column_count, depth);
}

NK_PUBLIC void nk_dots_pack_e3m2_sapphireamx(                    //
    nk_e3m2_t const *b, nk_size_t column_count, nk_size_t depth, //
    nk_size_t b_stride, void *b_packed) {

    nk_size_t const tmm_rows = 16;
    nk_size_t const tmm_cols = 32;
    nk_size_t const tile_elements = 512;
    nk_size_t const tile_bytes = tile_elements * sizeof(nk_bf16_t);

    nk_size_t const column_tiles_count = column_count / tmm_rows;
    nk_size_t const depth_tiles_count = nk_size_divide_round_up_(depth, tmm_cols);
    nk_size_t const column_remainder_count = column_count - column_tiles_count * tmm_rows;
    nk_size_t const total_tiles = column_tiles_count * depth_tiles_count;

    nk_dots_amx_packed_header_t *header = (nk_dots_amx_packed_header_t *)b_packed;
    header->full_column_tiles = (nk_u32_t)column_tiles_count;
    header->full_depth_tiles = (nk_u32_t)depth_tiles_count;
    header->column_remainder_count = (nk_u32_t)column_remainder_count;

    nk_size_t const tiles_offset = sizeof(nk_dots_amx_packed_header_t);
    nk_size_t const column_edge_offset = tiles_offset + total_tiles * tile_bytes;
    header->column_edge_offset = (nk_u32_t)column_edge_offset;

    nk_bf16_t *tiles_ptr = (nk_bf16_t *)((char *)b_packed + tiles_offset);
    nk_bf16_t *column_edge_ptr = (nk_bf16_t *)((char *)b_packed + column_edge_offset);

    for (nk_size_t idx = 0; idx < total_tiles * tile_elements; idx++) tiles_ptr[idx] = 0;

    for (nk_size_t column_tile_idx = 0; column_tile_idx < column_tiles_count; column_tile_idx++) {
        for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
            nk_size_t const tile_index = column_tile_idx * depth_tiles_count + depth_tile_idx;
            nk_bf16_t *tile_output = tiles_ptr + tile_index * tile_elements;

            nk_size_t const src_row_start = column_tile_idx * tmm_rows;
            nk_size_t const src_column_start = depth_tile_idx * tmm_cols;
            nk_size_t const columns_to_pack = (src_column_start + tmm_cols <= depth) ? tmm_cols
                                                                                     : (depth - src_column_start);

            for (nk_size_t row_idx = 0; row_idx < tmm_rows; row_idx++) {
                nk_size_t src_row = src_row_start + row_idx;
                __mmask32 column_mask = (columns_to_pack >= 32) ? 0xFFFFFFFF : ((__mmask32)1 << columns_to_pack) - 1;
                __m256i e3m2_row = _mm256_maskz_loadu_epi8(column_mask, b + src_row * b_stride + src_column_start);
                __m512i bf16_row = nk_e3m2x32_to_bf16x32_icelake_(e3m2_row);
                nk_bf16_t bf16_buf[32];
                _mm512_storeu_si512((__m512i *)bf16_buf, bf16_row);
                for (nk_size_t column_idx = 0; column_idx < columns_to_pack; column_idx++) {
                    nk_size_t const dst_idx = (column_idx / 2) * 32 + row_idx * 2 + (column_idx % 2);
                    tile_output[dst_idx] = bf16_buf[column_idx];
                }
            }
        }
    }

    if (column_remainder_count > 0) {
        nk_size_t const remainder_start_row = column_tiles_count * tmm_rows;
        for (nk_size_t row_idx = 0; row_idx < column_remainder_count; row_idx++) {
            for (nk_size_t column_idx = 0; column_idx < depth; column_idx += 32) {
                nk_size_t columns = (column_idx + 32 <= depth) ? 32 : (depth - column_idx);
                __mmask32 column_mask = (columns >= 32) ? 0xFFFFFFFF : ((__mmask32)1 << columns) - 1;
                __m256i e3m2_chunk = _mm256_maskz_loadu_epi8(
                    column_mask, b + (remainder_start_row + row_idx) * b_stride + column_idx);
                __m512i bf16_chunk = nk_e3m2x32_to_bf16x32_icelake_(e3m2_chunk);
                _mm512_mask_storeu_epi16(column_edge_ptr + row_idx * depth + column_idx, column_mask, bf16_chunk);
            }
        }
    }

    // Compute and store per-column norms for angular/euclidean distance
    nk_size_t norms_offset = column_edge_offset +
                             (column_remainder_count > 0 ? column_remainder_count * depth * sizeof(nk_bf16_t) : 0);
    header->norms_byte_offset = (nk_u32_t)norms_offset;
    nk_f32_t *norms = (nk_f32_t *)((char *)b_packed + norms_offset);
    for (nk_size_t col = 0; col < column_count; col++)
        norms[col] = nk_dots_reduce_sumsq_e3m2_(b + col * b_stride, depth);
}

NK_PUBLIC void nk_dots_packed_e3m2_sapphireamx(            //
    nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows_count, nk_size_t cols_count, nk_size_t depth, nk_size_t a_stride_bytes, nk_size_t c_stride_bytes) {
    nk_unused_(cols_count);

    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_size_t const column_tiles_count = header->full_column_tiles;
    nk_size_t const depth_tiles_count = header->full_depth_tiles;
    nk_size_t const column_remainder_count = header->column_remainder_count;

    nk_bf16_t const *b_tiles_base = (nk_bf16_t const *)((char const *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_bf16_t const *col_edge_ptr = (nk_bf16_t const *)((char const *)b_packed + header->column_edge_offset);

    nk_size_t const c_stride_elements = c_stride_bytes / sizeof(nk_f32_t);
    nk_size_t const tile_depth = 32;
    nk_size_t const tile_size = 512;
    nk_size_t const full_cols = column_tiles_count * 16;

    nk_size_t const row_blocks_count = nk_size_divide_round_up_(rows_count, 32);
    nk_size_t const col_blocks_count = column_tiles_count / 2;

    if (depth_tiles_count == 0) return;

    nk_dots_bf16_a16x32_sapphireamx_t a_tile_upper, a_tile_lower;
    nk_dots_bf16_state2x2_sapphireamx_t c_accum_buffer;

    nk_size_t const full_depth_tiles_count = depth / tile_depth;
    nk_size_t const depth_remainder = depth % tile_depth;

    nk_amx_tile_configure_sapphireamx_();

    // Loop order: row_blocks outer, col_blocks inner
    for (nk_size_t row_block_idx = 0; row_block_idx < row_blocks_count; row_block_idx++) {
        nk_size_t const row_block_start = row_block_idx * 32;
        nk_size_t const valid_rows_count = (row_block_start + 32 <= rows_count) ? 32 : (rows_count - row_block_start);
        nk_size_t const is_full_row_block = (valid_rows_count == 32);
        nk_size_t const rows_in_upper_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
        nk_size_t const rows_in_lower_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

        for (nk_size_t column_block_idx = 0; column_block_idx < col_blocks_count; column_block_idx++) {
            nk_size_t const col_block_start = column_block_idx * 32;
            nk_size_t const b_column_left_base = (column_block_idx * 2) * depth_tiles_count;
            nk_size_t const b_column_right_base = (column_block_idx * 2 + 1) * depth_tiles_count;

            // Zero accumulators (TMM4-7 stay resident across entire depth loop)
            _tile_zero(4);
            _tile_zero(5);
            _tile_zero(6);
            _tile_zero(7);

            // FP8 always uses buffered load for E3M2 -> BF16 conversion
            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                // Load A with FP8 -> BF16 conversion
                nk_dots_e3m2_load_a_sapphireamx_(&a_tile_upper, a + row_block_start * a_stride_bytes + depth_offset,
                                                 a_stride_bytes, rows_in_upper_tile, valid_depth);
                if (rows_in_lower_tile > 0) {
                    nk_dots_e3m2_load_a_sapphireamx_(&a_tile_lower,
                                                     a + (row_block_start + 16) * a_stride_bytes + depth_offset,
                                                     a_stride_bytes, rows_in_lower_tile, valid_depth);
                }

                nk_dots_bf16_b32x16_sapphireamx_t const *b_tile_left =
                    (nk_dots_bf16_b32x16_sapphireamx_t const *)(b_tiles_base +
                                                                (b_column_left_base + depth_tile_idx) * tile_size);
                nk_dots_bf16_b32x16_sapphireamx_t const *b_tile_right =
                    (nk_dots_bf16_b32x16_sapphireamx_t const *)(b_tiles_base +
                                                                (b_column_right_base + depth_tile_idx) * tile_size);

                _tile_loadd(0, a_tile_upper.data, 64);
                _tile_loadd(1, a_tile_lower.data, 64);
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
                _tile_stored(4, c_block, c_stride_bytes);
                _tile_stored(5, c_block + 16, c_stride_bytes);
                _tile_stored(6, (nk_f32_t *)((char *)c_block + 16 * c_stride_bytes), c_stride_bytes);
                _tile_stored(7, (nk_f32_t *)((char *)c_block + 16 * c_stride_bytes) + 16, c_stride_bytes);
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

            nk_dots_bf16_state_sapphireamx_t c_upper_state, c_lower_state;
            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_e3m2_load_a_sapphireamx_(&a_tile_upper, a + row_block_start * a_stride_bytes + depth_offset,
                                                 a_stride_bytes, rows_in_upper_tile, valid_depth);
                if (rows_in_lower_tile > 0) {
                    nk_dots_e3m2_load_a_sapphireamx_(&a_tile_lower,
                                                     a + (row_block_start + 16) * a_stride_bytes + depth_offset,
                                                     a_stride_bytes, rows_in_lower_tile, valid_depth);
                }

                nk_dots_bf16_b32x16_sapphireamx_t const *b_tile =
                    (nk_dots_bf16_b32x16_sapphireamx_t const *)(b_tiles_base +
                                                                (b_column_base + depth_tile_idx) * tile_size);

                _tile_loadd(0, a_tile_upper.data, 64);
                _tile_loadd(1, a_tile_lower.data, 64);
                _tile_loadd(2, b_tile->data, 64);

                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(6, 1, 2);
            }

            _tile_stored(4, c_upper_state.data, 64);
            _tile_stored(6, c_lower_state.data, 64);

            nk_dots_bf16_store_sapphireamx_(&c_upper_state, c + row_block_start * c_stride_elements + col_start,
                                            c_stride_elements, rows_in_upper_tile, 16);
            if (rows_in_lower_tile > 0) {
                nk_dots_bf16_store_sapphireamx_(&c_lower_state,
                                                c + (row_block_start + 16) * c_stride_elements + col_start,
                                                c_stride_elements, rows_in_lower_tile, 16);
            }
        }

        // Handle column-edge (remaining columns < 16) using AMX with partial tiles
        if (column_remainder_count > 0) {
            nk_dots_bf16_state_sapphireamx_t c_upper_state, c_lower_state;
            nk_dots_bf16_a16x32_sapphireamx_t b_as_a;
            nk_dots_bf16_b32x16_sapphireamx_t b_tile;

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_e3m2_load_a_sapphireamx_(&a_tile_upper, a + row_block_start * a_stride_bytes + depth_offset,
                                                 a_stride_bytes, rows_in_upper_tile, valid_depth);
                if (rows_in_lower_tile > 0) {
                    nk_dots_e3m2_load_a_sapphireamx_(&a_tile_lower,
                                                     a + (row_block_start + 16) * a_stride_bytes + depth_offset,
                                                     a_stride_bytes, rows_in_lower_tile, valid_depth);
                }

                nk_dots_bf16_load_a_sapphireamx_(&b_as_a, col_edge_ptr + depth_offset, depth, column_remainder_count,
                                                 valid_depth);
                nk_dots_pack_bf16_transposed_sapphireamx_(&b_as_a, &b_tile);

                _tile_loadd(0, a_tile_upper.data, 64);
                _tile_loadd(1, a_tile_lower.data, 64);
                _tile_loadd(2, b_tile.data, 64);

                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(6, 1, 2);
            }

            _tile_stored(4, c_upper_state.data, 64);
            _tile_stored(6, c_lower_state.data, 64);

            nk_dots_bf16_store_sapphireamx_(&c_upper_state, c + row_block_start * c_stride_elements + full_cols,
                                            c_stride_elements, rows_in_upper_tile, column_remainder_count);
            if (rows_in_lower_tile > 0) {
                nk_dots_bf16_store_sapphireamx_(&c_lower_state,
                                                c + (row_block_start + 16) * c_stride_elements + full_cols,
                                                c_stride_elements, rows_in_lower_tile, column_remainder_count);
            }
        }
    }

    _tile_release();
}

NK_PUBLIC void nk_dots_symmetric_e3m2_sapphireamx(                  //
    nk_e3m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, //
    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,    //
    nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const stride_elements = stride; // sizeof(nk_e3m2_t) == 1, so bytes == elements
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);

    // Handle row slicing: compute rows [row_start, row_end)
    nk_size_t const row_end = (row_count == 0)
                                  ? n_vectors
                                  : (row_start + row_count < n_vectors ? row_start + row_count : n_vectors);

    // Round depth up to multiple of 96 (3 tiles x 32 bf16 elements)
    nk_size_t const depth_tiles = nk_size_divide_round_up_(depth, 32);
    nk_size_t const depth_tile_groups = nk_size_divide_round_up_(depth_tiles, 3);

    nk_dots_bf16_a16x32_sapphireamx_t a_tiles[3];
    nk_dots_bf16_a16x32_sapphireamx_t b_src_tiles[3];
    nk_dots_bf16_b32x16_sapphireamx_t b_tiles[3];
    nk_dots_bf16_state_sapphireamx_t state;

    nk_amx_tile_configure_sapphireamx_();

    for (nk_size_t row_tile = row_start; row_tile < row_end; row_tile += 16) {
        nk_size_t const valid_rows = (row_tile + 16 <= row_end) ? 16 : (row_end - row_tile);

        for (nk_size_t col_tile = 0; col_tile < n_vectors; col_tile += 16) {
            nk_size_t const valid_cols = (col_tile + 16 <= n_vectors) ? 16 : (n_vectors - col_tile);

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

                    if (row_tile == col_tile) {
                        nk_dots_pack_bf16_transposed_sapphireamx_(&a_tiles[tile_idx], &b_tiles[tile_idx]);
                    }
                    else {
                        nk_dots_e3m2_load_a_sapphireamx_(                       //
                            &b_src_tiles[tile_idx],                             //
                            vectors + col_tile * stride_elements + depth_start, //
                            stride_elements, valid_cols, valid_depth);
                        nk_dots_pack_bf16_transposed_sapphireamx_(&b_src_tiles[tile_idx], &b_tiles[tile_idx]);
                    }
                }

                nk_dots_bf16_update_sapphireamx_( //
                    &state, &a_tiles[0], &a_tiles[1], &a_tiles[2], &b_tiles[0], &b_tiles[1], &b_tiles[2]);
            }

            nk_dots_bf16_store_sapphireamx_(                                   //
                &state, result + row_tile * result_stride_elements + col_tile, //
                result_stride_elements, valid_rows, valid_cols);
        }
    }
}

#pragma endregion // Micro Precision E3M2

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SAPPHIREAMX
#endif // NK_TARGET_X86_
#endif // NK_DOTS_SAPPHIREAMX_H
