/**
 *  @brief SIMD-accelerated Batched Dot Products for Granite Rapids.
 *  @file include/numkong/dots/graniteamx.h
 *  @author Ash Vardanian
 *  @date April 9, 2026
 *
 *  @sa include/numkong/dots.h
 *
 *  Native FP16×FP16→FP32 GEMM kernels using Intel AMX-FP16 (TDPFP16PS) on Granite Rapids CPUs.
 *  Same tile geometry as BF16 (16 rows × 32 FP16 = 1KB per tile), same 2×2 output blocking,
 *  same packing format — only the tile multiply instruction differs.
 *
 *  Tile register allocation:
 *
 *  - TMM0, TMM1: A matrix tiles (row blocks i and i+16)
 *  - TMM2, TMM3: B matrix tiles (column blocks j and j+16)
 *  - TMM4-7: C accumulator tiles (2 × 2 output grid = 32×32 F32 results)
 *
 *  @section amx_fp16_instructions Intel AMX-FP16 Instructions (Granite Rapids+)
 *
 *  FP16 matrix multiply (AMX-FP16):
 *
 *      Intrinsic                   Instruction                     Operation
 *      _tile_dpfp16ps              TDPFP16PS (TMM, TMM, TMM)       C += A × B (fp16 → f32)
 *
 *  TDPFP16PS: 16 × 16 × 32 = 8192 FP16 MACs per instruction (same throughput as TDPBF16PS).
 *
 *  @section ozaki_limitations F32→F64 via Ozaki Scheme — Attempted and Abandoned
 *
 *  We explored using AMX-FP16 tiles to compute F32→F64 GEMMs via the Ozaki decomposition scheme,
 *  splitting each F32 scalar into 2 or 3 FP16 terms and performing cross-product TDPFP16PS operations.
 *
 *  Results on Intel Xeon 6776P (Granite Rapids), single-threaded:
 *
 *  | Variant              | Speed (gso/s) | Precision | Notes                                      |
 *  |----------------------|:-------------:|:---------:|:------------------------------------------:|
 *  | 2-term, 2×1 blocking |    ~150       |  ~22 bits | Split accumulators, N=16 F64 flush         |
 *  | 3-term, 1×1 blocking |    ~110       |  ~22 bits | 3 accumulators by magnitude band           |
 *  | Pipelined 2-term     |    ~156       |  ~22 bits | Double-buffered A split, AMX/AVX-512 overlap|
 *  | MKL SGEMM            |    ~170       |  ~20 bits | Pure F32, no decomposition                 |
 *  | Skylake F64 accum    |     ~50       |  ~48 bits | F32×F32 multiply, F64 accumulation         |
 *
 *  The fundamental bottleneck is TDPFP16PS's internal F32 accumulation: each instruction sums
 *  32 FP16×FP16 products into an F32 register (23-bit mantissa). Even with Ozaki cross-term
 *  separation into distinct TMM accumulators (preventing magnitude mixing) and periodic extraction
 *  to F64 running sums, the per-instruction accumulation of 32 products loses ~5 bits
 *  (log2(32) = 5), capping effective precision at ~28 - 5 = ~23 bits — barely exceeding F32 BLAS.
 *
 *  Approaches attempted:
 *
 *  - 2-term decomposition (a = a_high + a_low): 4 TDPFP16PS per depth tile, ~20-bit products.
 *    With split accumulators (main + correction) merged in F64: ~22-bit effective precision.
 *    Faster than MKL at small depths (≤512) but precision plateaus at ~22 bits.
 *
 *  - 3-term decomposition (a = a_high + a_mid + a_low): 6 TDPFP16PS per depth tile, ~30-bit
 *    products. No precision improvement over 2-term because the F32 TMM accumulation is the
 *    bottleneck, not the decomposition quality. Strictly slower and no more precise.
 *
 *  - Periodic F64 flush (extract TMM accumulators to F64 every N depth tiles): prevents precision
 *    degradation at large depths. With N=16, ~15% overhead. Effective precision still ~24 bits
 *    (limited by per-TDPFP16PS accumulation of 32 products, not by inter-tile accumulation).
 *
 *  - AMX/AVX-512 pipelining (double-buffered A splitting overlapped with AMX compute): ~7%
 *    speedup at large sizes. Does not affect precision.
 *
 *  Conclusion: AMX-FP16's F32 tile accumulation fundamentally limits Ozaki to ~22-24 bits —
 *  comparable to F32 BLAS, far short of the ~48-bit F64 precision needed to justify the complexity.
 *  For F32→F64 GEMM, pure AVX-512 with F64 FMA remains the correct approach.
 */
#ifndef NK_DOTS_GRANITEAMX_H
#define NK_DOTS_GRANITEAMX_H

#if NK_TARGET_X8664_
#if NK_TARGET_GRANITEAMX

#include "numkong/dots/serial.h"
#include "numkong/dots/sapphireamx.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(                                                                                                     \
    __attribute__((target(                                                                                                        \
        "avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512fp16,avx512vbmi,f16c,fma,bmi,bmi2,amx-tile,amx-bf16,amx-int8,amx-fp16"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512fp16", "avx512vbmi", "f16c", "fma", \
                   "bmi", "bmi2", "amx-tile", "amx-bf16", "amx-int8", "amx-fp16")
#endif

#pragma region Tile Types

typedef struct {
    NK_ALIGN64 nk_f16_t data[16][32]; // 16 rows × 32 columns = 1KB
} nk_dots_f16_a16x32_graniteamx_t;

typedef struct {
    NK_ALIGN64 nk_f16_t data[16][16][2]; // 16 depth-groups × 16 columns × 2 = 1KB (pair-interleaved)
} nk_dots_f16_b32x16_graniteamx_t;

typedef struct {
    NK_ALIGN64 nk_f32_t data[16][16]; // 16 × 16 = 1KB accumulator
} nk_dots_f16_state_graniteamx_t;

typedef struct {
    nk_dots_f16_state_graniteamx_t c[2][2]; // 4KB total (2×2 output blocking)
} nk_dots_f16_state2x2_graniteamx_t;

#pragma endregion Tile Types

#pragma region Helpers

/* Initialize FP16 output state to zero */
NK_INTERNAL void nk_dots_f16_init_graniteamx_(nk_dots_f16_state_graniteamx_t *state) {
    __m512 zero_f32x16 = _mm512_setzero_ps();
    for (nk_size_t row_idx = 0; row_idx < 16; row_idx++) { _mm512_store_ps(state->data[row_idx], zero_f32x16); }
}

/* Load A tile from FP16 row-major source with masking for edge tiles */
NK_INTERNAL void nk_dots_f16_load_a_graniteamx_(        //
    nk_dots_f16_a16x32_graniteamx_t *a_tile,            //
    nk_f16_t const *src, nk_size_t src_stride_elements, //
    nk_size_t valid_rows, nk_size_t valid_cols) {

    __mmask32 column_mask = (valid_cols >= 32) ? 0xFFFFFFFF : ((__mmask32)1 << valid_cols) - 1;
    __m512i zero_i16x32 = _mm512_setzero_si512();

    for (nk_size_t row_idx = 0; row_idx < 16; row_idx++) {
        if (row_idx < valid_rows) {
            __m512i row_i16x32 = _mm512_maskz_loadu_epi16(column_mask, src + row_idx * src_stride_elements);
            _mm512_store_si512((__m512i *)a_tile->data[row_idx], row_i16x32);
        }
        else { _mm512_store_si512((__m512i *)a_tile->data[row_idx], zero_i16x32); }
    }
    nk_compiler_barrier_sapphireamx_();
}

/* Store F32 state to output matrix with masking for edge tiles */
NK_INTERNAL void nk_dots_f16_store_graniteamx_(   //
    nk_dots_f16_state_graniteamx_t const *state,  //
    nk_f32_t *dst, nk_size_t dst_stride_elements, //
    nk_size_t valid_rows, nk_size_t valid_cols) {

    __mmask16 column_mask = (valid_cols >= 16) ? 0xFFFF : ((__mmask16)1 << valid_cols) - 1;

    for (nk_size_t row_idx = 0; row_idx < valid_rows; row_idx++) {
        __m512 row_f32x16 = _mm512_load_ps(state->data[row_idx]);
        _mm512_mask_storeu_ps(dst + row_idx * dst_stride_elements, column_mask, row_f32x16);
    }
}

NK_INTERNAL void nk_dots_f16_output2x2_graniteamx_( //
    nk_dots_f16_state2x2_graniteamx_t const *state, //
    nk_f32_t *dst, nk_size_t dst_stride_elements,   //
    nk_size_t valid_rows, nk_size_t valid_cols) {

    nk_size_t const rows_high = (valid_rows > 16) ? 16 : valid_rows;
    nk_size_t const cols_left = (valid_cols > 16) ? 16 : valid_cols;
    nk_size_t const cols_right = (valid_cols > 16) ? valid_cols - 16 : 0;

    if (rows_high > 0 && cols_left > 0)
        nk_dots_f16_store_graniteamx_(&state->c[0][0], dst, dst_stride_elements, rows_high, cols_left);
    if (rows_high > 0 && cols_right > 0)
        nk_dots_f16_store_graniteamx_(&state->c[0][1], dst + 16, dst_stride_elements, rows_high, cols_right);

    if (valid_rows > 16) {
        nk_size_t const rows_low = valid_rows - 16;
        nk_f32_t *dst_low = dst + 16 * dst_stride_elements;
        if (cols_left > 0)
            nk_dots_f16_store_graniteamx_(&state->c[1][0], dst_low, dst_stride_elements, rows_low, cols_left);
        if (cols_right > 0)
            nk_dots_f16_store_graniteamx_(&state->c[1][1], dst_low + 16, dst_stride_elements, rows_low, cols_right);
    }
}

NK_INTERNAL void nk_dots_f16_update_graniteamx_(     //
    nk_dots_f16_state_graniteamx_t *state,           //
    nk_dots_f16_a16x32_graniteamx_t const *a_tile_0, //
    nk_dots_f16_a16x32_graniteamx_t const *a_tile_1, //
    nk_dots_f16_a16x32_graniteamx_t const *a_tile_2, //
    nk_dots_f16_b32x16_graniteamx_t const *b_tile_0, //
    nk_dots_f16_b32x16_graniteamx_t const *b_tile_1, //
    nk_dots_f16_b32x16_graniteamx_t const *b_tile_2) {

    _tile_loadd(0, state->data, 64);
    _tile_loadd(1, a_tile_0->data, 64);
    _tile_loadd(2, a_tile_1->data, 64);
    _tile_loadd(3, a_tile_2->data, 64);
    _tile_loadd(4, b_tile_0->data, 64);
    _tile_loadd(5, b_tile_1->data, 64);
    _tile_loadd(6, b_tile_2->data, 64);

    _tile_dpfp16ps(0, 1, 4);
    _tile_dpfp16ps(0, 2, 5);
    _tile_dpfp16ps(0, 3, 6);

    _tile_stored(0, state->data, 64);
}

#pragma endregion Helpers

#pragma region F16 Native

NK_PUBLIC nk_size_t nk_dots_packed_size_f16_graniteamx(nk_size_t column_count, nk_size_t depth) {
    nk_size_t const tmm_rows = 16;
    nk_size_t const tmm_cols = 32;
    nk_size_t const tile_bytes = 512 * sizeof(nk_f16_t); // 16 × 32 × 2 = 1KB

    nk_size_t const full_column_tiles = column_count / tmm_rows;
    nk_size_t const tiles_along_depth = nk_size_divide_round_up_(depth, tmm_cols);
    nk_size_t const column_remainder_count = column_count - full_column_tiles * tmm_rows;

    // Header (64 bytes aligned)
    nk_size_t size = sizeof(nk_dots_amx_packed_header_t);

    // All tiles for full column rows (pair-interleaved, depth remainder zero-padded)
    size += full_column_tiles * tiles_along_depth * tile_bytes;

    // Column edge: remaining rows for ALL depth columns, stored row-major
    if (column_remainder_count > 0) size += column_remainder_count * depth * sizeof(nk_f16_t);

    // Per-column norms for angular/euclidean distance (4 bytes each: f32)
    size += column_count * sizeof(nk_f32_t);

    return size;
}

NK_PUBLIC void nk_dots_pack_f16_graniteamx(                     //
    nk_f16_t const *b, nk_size_t column_count, nk_size_t depth, //
    nk_size_t b_stride_in_bytes, void *b_packed) {

    // AMX FP16 tile dimensions: 16 rows × 32 columns (512 FP16 elements = 1KB)
    nk_size_t const tmm_rows = 16;
    nk_size_t const tmm_cols = 32;
    nk_size_t const tile_elements = 512;
    nk_size_t const tile_bytes = tile_elements * sizeof(nk_f16_t);
    nk_size_t const b_stride_elements = b_stride_in_bytes / sizeof(nk_f16_t);

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
    nk_f16_t *tiles_ptr = (nk_f16_t *)((char *)b_packed + tiles_offset);
    nk_f16_t *column_edge_ptr = (nk_f16_t *)((char *)b_packed + column_edge_offset);

    // Pack tiles: gather 16 strided rows into aligned temporary, transpose via SIMD, copy to packed buffer.
    // FP16 has the same 16-bit pair-interleaved layout as BF16, so reuse the BF16 transpose.
    for (nk_size_t column_tile_idx = 0; column_tile_idx < column_tiles_count; column_tile_idx++) {
        for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {

            nk_size_t const tile_index = column_tile_idx * depth_tiles_count + depth_tile_idx;
            nk_f16_t *tile_output = tiles_ptr + tile_index * tile_elements;

            nk_size_t const src_row_start = column_tile_idx * tmm_rows;
            nk_size_t const src_column_start = depth_tile_idx * tmm_cols;
            nk_size_t const columns_to_pack = (src_column_start + tmm_cols <= depth) ? tmm_cols
                                                                                     : (depth - src_column_start);

            // Gather 16 strided source rows into a contiguous aligned tile
            nk_dots_bf16_a16x32_sapphireamx_t source_tile;
            if (columns_to_pack == tmm_cols) {
                for (nk_size_t row_idx = 0; row_idx < tmm_rows; row_idx++) {
                    nk_f16_t const *source_row = b + (src_row_start + row_idx) * b_stride_elements + src_column_start;
                    _mm512_store_si512(&source_tile.data[row_idx][0], _mm512_loadu_si512(source_row));
                }
            }
            else {
                __mmask32 depth_mask = (__mmask32)((columns_to_pack < 32) ? ((1U << columns_to_pack) - 1) : ~0U);
                for (nk_size_t row_idx = 0; row_idx < tmm_rows; row_idx++) {
                    nk_f16_t const *source_row = b + (src_row_start + row_idx) * b_stride_elements + src_column_start;
                    _mm512_store_si512(&source_tile.data[row_idx][0], _mm512_maskz_loadu_epi16(depth_mask, source_row));
                }
            }

            // Transpose into aligned local, then copy to (potentially unaligned) packed buffer.
            // BF16 and FP16 share identical 16-bit pair-interleaved layout for TDPBF16PS/TDPFP16PS.
            nk_dots_bf16_b32x16_sapphireamx_t transposed_tile;
            nk_dots_pack_bf16_transposed_sapphireamx_(&source_tile, &transposed_tile);
            for (nk_size_t i = 0; i < tile_bytes; i += 64)
                _mm512_storeu_si512((char *)tile_output + i, _mm512_load_si512((char const *)&transposed_tile + i));
        }
    }

    // Pack column-remainder rows using vectorized masked copies
    if (column_remainder_count > 0) {
        nk_size_t const remainder_start_row = column_tiles_count * tmm_rows;
        for (nk_size_t row_idx = 0; row_idx < column_remainder_count; row_idx++) {
            nk_f16_t const *src_row = b + (remainder_start_row + row_idx) * b_stride_elements;
            nk_f16_t *dst_row = column_edge_ptr + row_idx * depth;
            nk_size_t column_idx = 0;
            for (; column_idx + 32 <= depth; column_idx += 32) {
                _mm512_storeu_si512(dst_row + column_idx, _mm512_loadu_si512(src_row + column_idx));
            }
            if (column_idx < depth) {
                __mmask32 tail_mask = (__mmask32)((1U << (depth - column_idx)) - 1);
                _mm512_mask_storeu_epi16(dst_row + column_idx, tail_mask,
                                         _mm512_maskz_loadu_epi16(tail_mask, src_row + column_idx));
            }
        }
    }

    // Compute and store per-column norms for angular/euclidean distance
    nk_size_t norms_offset = column_edge_offset +
                             (column_remainder_count > 0 ? column_remainder_count * depth * sizeof(nk_f16_t) : 0);
    header->norms_byte_offset = (nk_u32_t)norms_offset;
    nk_f32_t *norms = (nk_f32_t *)((char *)b_packed + norms_offset);
    for (nk_size_t col = 0; col < column_count; col++)
        norms[col] = nk_dots_reduce_sumsq_f16_(b + col * b_stride_elements, depth);
}

NK_PUBLIC void nk_dots_packed_f16_graniteamx(             //
    nk_f16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows_count, nk_size_t cols_count, nk_size_t depth, nk_size_t a_stride_bytes, nk_size_t c_stride_bytes) {
    nk_unused_(cols_count);

    // Parse packed B header
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_size_t const column_tiles_count = header->full_column_tiles;
    nk_size_t const depth_tiles_count = header->full_depth_tiles;
    nk_size_t const column_remainder_count = header->column_remainder_count;

    // Packed B data regions
    nk_f16_t const *b_tiles_base = (nk_f16_t const *)((char const *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_f16_t const *col_edge_ptr = (nk_f16_t const *)((char const *)b_packed + header->column_edge_offset);

    // Stride conversions
    nk_size_t const a_stride_elements = a_stride_bytes / sizeof(nk_f16_t);
    nk_size_t const c_stride_elements = c_stride_bytes / sizeof(nk_f32_t);

    // Tile dimensions
    nk_size_t const tile_depth = 32;
    nk_size_t const tile_size = 512;
    nk_size_t const full_cols = column_tiles_count * 16;

    // Block counts (32 × 32 output blocks = 2 × 2 tiles)
    nk_size_t const row_blocks_count = nk_size_divide_round_up_(rows_count, 32);
    nk_size_t const col_blocks_count = column_tiles_count / 2;

    if (depth_tiles_count == 0) return;

    // Tile buffers for A (only used for edge tiles)
    nk_dots_f16_a16x32_graniteamx_t a_tile_top, a_tile_bottom;
    nk_dots_f16_state2x2_graniteamx_t c_accum_buffer;

    // Precompute: number of full depth-tiles (no masking needed)
    nk_size_t const full_depth_tiles_count = depth / tile_depth;
    nk_size_t const depth_remainder = depth % tile_depth;

    nk_amx_tile_configure_sapphireamx_();

    // Loop order: row_blocks outer, col_blocks inner - maximizes A tile L2 cache reuse
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
                nk_f16_t const *a_top_base = a + row_block_start * a_stride_elements;
                nk_f16_t const *a_bottom_base = a + (row_block_start + 16) * a_stride_elements;

                nk_dots_f16_b32x16_graniteamx_t const *b_tile_left =
                    (nk_dots_f16_b32x16_graniteamx_t const *)(b_tiles_base + b_column_left_base * tile_size);
                nk_dots_f16_b32x16_graniteamx_t const *b_tile_right =
                    (nk_dots_f16_b32x16_graniteamx_t const *)(b_tiles_base + b_column_right_base * tile_size);

                // Prologue: load first depth tile
                _tile_loadd(0, a_top_base, a_stride_bytes);
                _tile_loadd(1, a_bottom_base, a_stride_bytes);
                _tile_loadd(2, b_tile_left->data, 64);
                _tile_loadd(3, b_tile_right->data, 64);

                // Main loop: 2-deep software pipelining
                for (nk_size_t depth_tile_idx = 0; depth_tile_idx < full_depth_tiles_count - 1; depth_tile_idx++) {
                    nk_size_t const next_depth_offset = (depth_tile_idx + 1) * tile_depth;

                    _tile_dpfp16ps(4, 0, 2);
                    _tile_dpfp16ps(5, 0, 3);
                    _tile_dpfp16ps(6, 1, 2);
                    _tile_dpfp16ps(7, 1, 3);

                    _tile_loadd(0, a_top_base + next_depth_offset, a_stride_bytes);
                    _tile_loadd(1, a_bottom_base + next_depth_offset, a_stride_bytes);
                    b_tile_left = (nk_dots_f16_b32x16_graniteamx_t const *)(b_tiles_base +
                                                                            (b_column_left_base + depth_tile_idx + 1) *
                                                                                tile_size);
                    b_tile_right = (nk_dots_f16_b32x16_graniteamx_t const *)(b_tiles_base + (b_column_right_base +
                                                                                             depth_tile_idx + 1) *
                                                                                                tile_size);
                    _tile_loadd(2, b_tile_left->data, 64);
                    _tile_loadd(3, b_tile_right->data, 64);
                }

                // Epilogue: final depth tile
                _tile_dpfp16ps(4, 0, 2);
                _tile_dpfp16ps(5, 0, 3);
                _tile_dpfp16ps(6, 1, 2);
                _tile_dpfp16ps(7, 1, 3);

                // Handle partial depth-tile (if any)
                if (depth_remainder > 0) {
                    nk_size_t const depth_offset = full_depth_tiles_count * tile_depth;

                    nk_dots_f16_load_a_graniteamx_(&a_tile_top, a_top_base + depth_offset, a_stride_elements, 16,
                                                   depth_remainder);
                    nk_dots_f16_load_a_graniteamx_(&a_tile_bottom, a_bottom_base + depth_offset, a_stride_elements, 16,
                                                   depth_remainder);

                    b_tile_left = (nk_dots_f16_b32x16_graniteamx_t const *)(b_tiles_base + (b_column_left_base +
                                                                                            full_depth_tiles_count) *
                                                                                               tile_size);
                    b_tile_right = (nk_dots_f16_b32x16_graniteamx_t const *)(b_tiles_base + (b_column_right_base +
                                                                                             full_depth_tiles_count) *
                                                                                                tile_size);

                    _tile_loadd(0, a_tile_top.data, 64);
                    _tile_loadd(1, a_tile_bottom.data, 64);
                    _tile_loadd(2, b_tile_left->data, 64);
                    _tile_loadd(3, b_tile_right->data, 64);

                    _tile_dpfp16ps(4, 0, 2);
                    _tile_dpfp16ps(5, 0, 3);
                    _tile_dpfp16ps(6, 1, 2);
                    _tile_dpfp16ps(7, 1, 3);
                }
            }
            // Full row-block but only partial depth tile (depth < tile_depth)
            else if (is_full_row_block) {
                nk_f16_t const *a_top_base = a + row_block_start * a_stride_elements;
                nk_f16_t const *a_bottom_base = a + (row_block_start + 16) * a_stride_elements;

                nk_dots_f16_load_a_graniteamx_(&a_tile_top, a_top_base, a_stride_elements, 16, depth_remainder);
                nk_dots_f16_load_a_graniteamx_(&a_tile_bottom, a_bottom_base, a_stride_elements, 16, depth_remainder);

                nk_dots_f16_b32x16_graniteamx_t const *b_tile_left =
                    (nk_dots_f16_b32x16_graniteamx_t const *)(b_tiles_base + b_column_left_base * tile_size);
                nk_dots_f16_b32x16_graniteamx_t const *b_tile_right =
                    (nk_dots_f16_b32x16_graniteamx_t const *)(b_tiles_base + b_column_right_base * tile_size);

                _tile_loadd(0, a_tile_top.data, 64);
                _tile_loadd(1, a_tile_bottom.data, 64);
                _tile_loadd(2, b_tile_left->data, 64);
                _tile_loadd(3, b_tile_right->data, 64);

                _tile_dpfp16ps(4, 0, 2);
                _tile_dpfp16ps(5, 0, 3);
                _tile_dpfp16ps(6, 1, 2);
                _tile_dpfp16ps(7, 1, 3);
            }
            // Slow path: edge row-block → buffered load with masking
            else {
                nk_size_t const rows_in_high_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
                nk_size_t const rows_in_low_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

                for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                    nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                    nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth
                                                                                            : depth_remainder;

                    nk_dots_f16_load_a_graniteamx_(&a_tile_top, a + row_block_start * a_stride_elements + depth_offset,
                                                   a_stride_elements, rows_in_high_tile, valid_depth);
                    if (rows_in_low_tile > 0) {
                        nk_dots_f16_load_a_graniteamx_(&a_tile_bottom,
                                                       a + (row_block_start + 16) * a_stride_elements + depth_offset,
                                                       a_stride_elements, rows_in_low_tile, valid_depth);
                    }

                    nk_dots_f16_b32x16_graniteamx_t const *b_tile_left =
                        (nk_dots_f16_b32x16_graniteamx_t const *)(b_tiles_base +
                                                                  (b_column_left_base + depth_tile_idx) * tile_size);
                    nk_dots_f16_b32x16_graniteamx_t const *b_tile_right =
                        (nk_dots_f16_b32x16_graniteamx_t const *)(b_tiles_base +
                                                                  (b_column_right_base + depth_tile_idx) * tile_size);

                    _tile_loadd(0, a_tile_top.data, 64);
                    _tile_loadd(1, a_tile_bottom.data, 64);
                    _tile_loadd(2, b_tile_left->data, 64);
                    _tile_loadd(3, b_tile_right->data, 64);

                    _tile_dpfp16ps(4, 0, 2);
                    _tile_dpfp16ps(5, 0, 3);
                    _tile_dpfp16ps(6, 1, 2);
                    _tile_dpfp16ps(7, 1, 3);
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
                nk_dots_f16_output2x2_graniteamx_(&c_accum_buffer,
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
            nk_size_t const rows_in_high_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
            nk_size_t const rows_in_low_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

            nk_dots_f16_state_graniteamx_t c_high_state, c_low_state;

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_f16_load_a_graniteamx_(&a_tile_top, a + row_block_start * a_stride_elements + depth_offset,
                                               a_stride_elements, rows_in_high_tile, valid_depth);
                if (rows_in_low_tile > 0) {
                    nk_dots_f16_load_a_graniteamx_(&a_tile_bottom,
                                                   a + (row_block_start + 16) * a_stride_elements + depth_offset,
                                                   a_stride_elements, rows_in_low_tile, valid_depth);
                }

                nk_dots_f16_b32x16_graniteamx_t const *b_tile =
                    (nk_dots_f16_b32x16_graniteamx_t const *)(b_tiles_base +
                                                              (b_column_base + depth_tile_idx) * tile_size);

                _tile_loadd(0, a_tile_top.data, 64);
                _tile_loadd(1, a_tile_bottom.data, 64);
                _tile_loadd(2, b_tile->data, 64);

                _tile_dpfp16ps(4, 0, 2);
                _tile_dpfp16ps(6, 1, 2);
            }

            _tile_stored(4, c_high_state.data, 64);
            _tile_stored(6, c_low_state.data, 64);

            nk_dots_f16_store_graniteamx_(&c_high_state, c + row_block_start * c_stride_elements + col_start,
                                          c_stride_elements, rows_in_high_tile, 16);
            if (rows_in_low_tile > 0) {
                nk_dots_f16_store_graniteamx_(&c_low_state, c + (row_block_start + 16) * c_stride_elements + col_start,
                                              c_stride_elements, rows_in_low_tile, 16);
            }
        }
    }

    // Handle column-edge (remaining columns < 16) using AMX with partial tiles
    if (column_remainder_count > 0) {
        for (nk_size_t row_block_idx = 0; row_block_idx < row_blocks_count; row_block_idx++) {
            nk_size_t const row_block_start = row_block_idx * 32;
            nk_size_t const valid_rows_count = (row_block_start + 32 <= rows_count) ? 32
                                                                                    : (rows_count - row_block_start);
            nk_size_t const rows_in_high_tile = (valid_rows_count > 16) ? 16 : valid_rows_count;
            nk_size_t const rows_in_low_tile = (valid_rows_count > 16) ? valid_rows_count - 16 : 0;

            nk_dots_f16_state_graniteamx_t c_high_state, c_low_state;
            nk_dots_bf16_a16x32_sapphireamx_t b_as_a;
            nk_dots_bf16_b32x16_sapphireamx_t b_tile;

            _tile_zero(4);
            _tile_zero(6);

            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles_count; depth_tile_idx++) {
                nk_size_t const depth_offset = depth_tile_idx * tile_depth;
                nk_size_t const valid_depth = (depth_tile_idx < full_depth_tiles_count) ? tile_depth : depth_remainder;

                nk_dots_f16_load_a_graniteamx_(&a_tile_top, a + row_block_start * a_stride_elements + depth_offset,
                                               a_stride_elements, rows_in_high_tile, valid_depth);
                if (rows_in_low_tile > 0) {
                    nk_dots_f16_load_a_graniteamx_(&a_tile_bottom,
                                                   a + (row_block_start + 16) * a_stride_elements + depth_offset,
                                                   a_stride_elements, rows_in_low_tile, valid_depth);
                }

                // Load edge columns as BF16-shaped tile (same 16-bit layout) and transpose on-the-fly
                nk_dots_bf16_load_a_sapphireamx_(&b_as_a, (nk_bf16_t const *)(col_edge_ptr + depth_offset), depth,
                                                 column_remainder_count, valid_depth);
                nk_dots_pack_bf16_transposed_sapphireamx_(&b_as_a, &b_tile);

                _tile_loadd(0, a_tile_top.data, 64);
                _tile_loadd(1, a_tile_bottom.data, 64);
                _tile_loadd(2, b_tile.data, 64);

                _tile_dpfp16ps(4, 0, 2);
                _tile_dpfp16ps(6, 1, 2);
            }

            _tile_stored(4, c_high_state.data, 64);
            _tile_stored(6, c_low_state.data, 64);

            nk_dots_f16_store_graniteamx_(&c_high_state, c + row_block_start * c_stride_elements + full_cols,
                                          c_stride_elements, rows_in_high_tile, column_remainder_count);
            if (rows_in_low_tile > 0) {
                nk_dots_f16_store_graniteamx_(&c_low_state, c + (row_block_start + 16) * c_stride_elements + full_cols,
                                              c_stride_elements, rows_in_low_tile, column_remainder_count);
            }
        }
    }

    _tile_release();
}

NK_PUBLIC void nk_dots_symmetric_f16_graniteamx(                                   //
    nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,             //
    nk_size_t stride_in_bytes, nk_f32_t *result, nk_size_t result_stride_in_bytes, //
    nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_f16_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);

    nk_size_t const row_end = (row_count == 0)
                                  ? vectors_count
                                  : (row_start + row_count < vectors_count ? row_start + row_count : vectors_count);

    // Round depth up to multiple of 96 (3 tiles × 32 elements)
    nk_size_t const depth_tiles = nk_size_divide_round_up_(depth, 32);
    nk_size_t const depth_tile_groups = nk_size_divide_round_up_(depth_tiles, 3);

    nk_dots_f16_a16x32_graniteamx_t a_tiles[3];
    nk_dots_f16_a16x32_graniteamx_t b_src_tiles[3];
    nk_dots_f16_b32x16_graniteamx_t b_tiles[3];
    nk_dots_f16_state_graniteamx_t state;

    nk_amx_tile_configure_sapphireamx_();

    for (nk_size_t row_tile = row_start; row_tile < row_end; row_tile += 16) {
        nk_size_t const valid_rows = (row_tile + 16 <= row_end) ? 16 : (row_end - row_tile);

        for (nk_size_t col_tile = 0; col_tile < vectors_count; col_tile += 16) {
            nk_size_t const valid_cols = (col_tile + 16 <= vectors_count) ? 16 : (vectors_count - col_tile);

            nk_dots_f16_init_graniteamx_(&state);

            for (nk_size_t depth_group_idx = 0; depth_group_idx < depth_tile_groups; depth_group_idx++) {
                nk_size_t const depth_base = depth_group_idx * 96;

                for (int tile_idx = 0; tile_idx < 3; tile_idx++) {
                    nk_size_t const depth_start = depth_base + tile_idx * 32;
                    nk_size_t const valid_depth = (depth_start + 32 <= depth)
                                                      ? 32
                                                      : (depth > depth_start ? depth - depth_start : 0);

                    nk_dots_f16_load_a_graniteamx_(                         //
                        &a_tiles[tile_idx],                                 //
                        vectors + row_tile * stride_elements + depth_start, //
                        stride_elements, valid_rows, valid_depth);

                    if (row_tile == col_tile) {
                        // Reuse A data as B (self-correlation on diagonal)
                        nk_dots_pack_bf16_transposed_sapphireamx_(
                            (nk_dots_bf16_a16x32_sapphireamx_t const *)&a_tiles[tile_idx],
                            (nk_dots_bf16_b32x16_sapphireamx_t *)&b_tiles[tile_idx]);
                    }
                    else {
                        nk_dots_f16_load_a_graniteamx_(                         //
                            &b_src_tiles[tile_idx],                             //
                            vectors + col_tile * stride_elements + depth_start, //
                            stride_elements, valid_cols, valid_depth);
                        nk_dots_pack_bf16_transposed_sapphireamx_(
                            (nk_dots_bf16_a16x32_sapphireamx_t const *)&b_src_tiles[tile_idx],
                            (nk_dots_bf16_b32x16_sapphireamx_t *)&b_tiles[tile_idx]);
                    }
                }

                nk_dots_f16_update_graniteamx_(                    //
                    &state, &a_tiles[0], &a_tiles[1], &a_tiles[2], //
                    &b_tiles[0], &b_tiles[1], &b_tiles[2]);
            }

            nk_dots_f16_store_graniteamx_(                                     //
                &state, result + row_tile * result_stride_elements + col_tile, //
                result_stride_elements, valid_rows, valid_cols);
        }
    }
}

#pragma endregion F16 Native

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_GRANITEAMX
#endif // NK_TARGET_X8664_
#endif // NK_DOTS_GRANITEAMX_H
