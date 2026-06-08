/**
 *  @brief SWAR-accelerated Batched Dot Products for SIMD-free CPUs.
 *  @file include/numkong/dots/serial.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/dots.h for API overview and use cases
 *
 *  This file provides two macro families for generating GEMM kernels:
 *
 *  - nk_define_dots_packed_: vectorized inner-products between rows of A and Bᵀ
 *  - nk_define_dots_symmetric_: vectorized inner-products between rows and columns of A
 *
 *  Both use the same B packing format (see below), enabling pack-once-use-anywhere.
 *
 *  @section packing B Matrix Packing Format
 *
 *  Computing C = A × Bᵀ where:
 *
 *  - A[row_count, depth] row-major: A[i, k] at address A + i × lda + k
 *  - B[column_count, depth] row-major (pre-transposed): B[j, k] at address B + j × ldb + k
 *  - C[row_count, column_count] row-major: C[i, j] at address C + i × ldc + j
 *
 *  The API convention stores B as Bᵀ for efficient SIMD access:
 *
 *  - A[i, k:k+4] is contiguous in row-major (good)
 *  - B[j, k:k+4] is contiguous in row-major (good - already transposed)
 *
 *  Packing adds row grouping (group_size = 16) for:
 *
 *  - Zero-padding on edges (avoids boundary checks in inner loop)
 *  - Cache-friendly blocking in outer loops
 *
 *  Memory layout example - B[8, 8] with 8 output columns (j), 8 depth (k):
 *
 *            k=0   k=1   k=2   k=3   k=4   k=5   k=6   k=7
 *         ┌─────────────────────────────────────────────────┐
 *    j=0  │  a0    a1    a2    a3    a4    a5    a6    a7   │
 *    j=1  │  b0    b1    b2    b3    b4    b5    b6    b7   │
 *    j=2  │  c0    c1    c2    c3    c4    c5    c6    c7   │
 *    j=3  │  d0    d1    d2    d3    d4    d5    d6    d7   │
 *    j=4  │  e0    e1    e2    e3    e4    e5    e6    e7   │
 *    j=5  │  f0    f1    f2    f3    f4    f5    f6    f7   │
 *    j=6  │  g0    g1    g2    g3    g4    g5    g6    g7   │
 *    j=7  │  h0    h1    h2    h3    h4    h5    h6    h7   │
 *         └─────────────────────────────────────────────────┘
 *
 *  Packed as B_packed[column_count_padded, depth] (grouped for alignment):
 *
 *    Group 0 (j=0..7, padded to 16):
 *      ┌───────────────────────────────────┐
 *      │ a0 a1 a2 a3 a4 a5 a6 a7 │  j=0    │  ← row 0 copied as-is
 *      │ b0 b1 b2 b3 b4 b5 b6 b7 │  j=1    │
 *      │ c0 c1 c2 c3 c4 c5 c6 c7 │  j=2    │
 *      │ d0 d1 d2 d3 d4 d5 d6 d7 │  j=3    │
 *      │ e0 e1 e2 e3 e4 e5 e6 e7 │  j=4    │
 *      │ f0 f1 f2 f3 f4 f5 f6 f7 │  j=5    │
 *      │ g0 g1 g2 g3 g4 g5 g6 g7 │  j=6    │
 *      │ h0 h1 h2 h3 h4 h5 h6 h7 │  j=7    │
 *      │ 00 00 00 00 00 00 00 00 │ padding │
 *      │ ...                     │ ...     │
 *      └───────────────────────────────────┘
 *
 *  Addressing formula for B_packed[j, k]:
 *
 *      group = j / group_size
 *      j_in_group = j % group_size
 *      B_packed[j, k] = packed[group * group_size * depth + j_in_group * depth + k]
 *
 *  Inner loop accesses B_packed[j, k:k+simd] which is contiguous - just ptr + k.
 */

#ifndef NK_DOTS_SERIAL_H
#define NK_DOTS_SERIAL_H

#include "numkong/types.h"
#include "numkong/cast/serial.h"    // `nk_partial_load_b32x4_serial_`
#include "numkong/dot/serial.h"     // `nk_dot_f32x4_state_serial_t`
#include "numkong/spatial/serial.h" // `nk_f32_sqrt_serial`
#include "numkong/reduce.h"         // `nk_reduce_moments_*`

/*  GCC's -Wstringop-overflow produces false positives on the padded accumulator arrays
 *  in nk_define_cross_symmetric_ macro expansions (accumulators[4][7] with runtime indexing). */
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/*  Packed buffer header (64-byte aligned).
 *  Used by all packed matmul backends (serial, NEON, AVX-512, SVE).
 *
 *  Important units clarification:
 *  - For types where dimensions_per_value = 1 (f32, i8, u8, etc.): dimensions == values
 *  - For sub-byte types (i4x2, u4x2): dimensions ≠ values
 *    - dimensions = individual 4-bit nibbles (e.g., 128 nibbles)
 *    - values = storage bytes containing nibbles (e.g., 64 bytes for 128 nibbles)
 *    - dimensions_per_value = 2 (2 nibbles per byte)
 */
typedef struct {
    nk_u32_t column_count;        // Actual number of columns (not padded)
    nk_u32_t depth_dimensions;    // Logical depth in dimensions (nibbles for i4/u4, values for i8/f32)
    nk_u32_t depth_padded_values; // Padded depth in storage values (bytes for i4/u4, values for i8/f32)
    nk_u32_t reserved[13];        // Padding to 64 bytes
} nk_cross_packed_buffer_header_t;

/*  Norm compute helpers for packing.
 *  Each computes the norm (sum-of-squares or popcount) of a contiguous row.
 *  Used by `nk_define_cross_pack_` to append per-column norms to packed buffers.
 */
NK_INTERNAL nk_f64_t nk_dots_reduce_sumsq_f64_(nk_f64_t const *data, nk_size_t count) {
    nk_f64_t sum, sumsq;
    nk_reduce_moments_f64(data, count, sizeof(nk_f64_t), &sum, &sumsq);
    return sumsq;
}
NK_INTERNAL nk_f64_t nk_dots_reduce_sumsq_f32_(nk_f32_t const *data, nk_size_t count) {
    nk_f64_t sum, sumsq;
    nk_reduce_moments_f32(data, count, sizeof(nk_f32_t), &sum, &sumsq);
    return sumsq;
}
NK_INTERNAL nk_f32_t nk_dots_reduce_sumsq_f16_(nk_f16_t const *data, nk_size_t count) {
    nk_f32_t sum, sumsq;
    nk_reduce_moments_f16(data, count, sizeof(nk_f16_t), &sum, &sumsq);
    return sumsq;
}
NK_INTERNAL nk_f32_t nk_dots_reduce_sumsq_bf16_(nk_bf16_t const *data, nk_size_t count) {
    nk_f32_t sum, sumsq;
    nk_reduce_moments_bf16(data, count, sizeof(nk_bf16_t), &sum, &sumsq);
    return sumsq;
}
NK_INTERNAL nk_f32_t nk_dots_reduce_sumsq_e4m3_(nk_e4m3_t const *data, nk_size_t count) {
    nk_f32_t sum, sumsq;
    nk_reduce_moments_e4m3(data, count, sizeof(nk_e4m3_t), &sum, &sumsq);
    return sumsq;
}
NK_INTERNAL nk_f32_t nk_dots_reduce_sumsq_e5m2_(nk_e5m2_t const *data, nk_size_t count) {
    nk_f32_t sum, sumsq;
    nk_reduce_moments_e5m2(data, count, sizeof(nk_e5m2_t), &sum, &sumsq);
    return sumsq;
}
NK_INTERNAL nk_f32_t nk_dots_reduce_sumsq_e2m3_(nk_e2m3_t const *data, nk_size_t count) {
    nk_f32_t sum, sumsq;
    nk_reduce_moments_e2m3(data, count, sizeof(nk_e2m3_t), &sum, &sumsq);
    return sumsq;
}
NK_INTERNAL nk_f32_t nk_dots_reduce_sumsq_e3m2_(nk_e3m2_t const *data, nk_size_t count) {
    nk_f32_t sum, sumsq;
    nk_reduce_moments_e3m2(data, count, sizeof(nk_e3m2_t), &sum, &sumsq);
    return sumsq;
}
NK_INTERNAL nk_u32_t nk_dots_reduce_sumsq_i8_(nk_i8_t const *data, nk_size_t count) {
    nk_i64_t sum;
    nk_u64_t sumsq;
    nk_reduce_moments_i8(data, count, sizeof(nk_i8_t), &sum, &sumsq);
    return (nk_u32_t)sumsq;
}
NK_INTERNAL nk_u32_t nk_dots_reduce_sumsq_u8_(nk_u8_t const *data, nk_size_t count) {
    nk_u64_t sum, sumsq;
    nk_reduce_moments_u8(data, count, sizeof(nk_u8_t), &sum, &sumsq);
    return (nk_u32_t)sumsq;
}
NK_INTERNAL nk_u32_t nk_dots_reduce_sumsq_i4_(nk_i4x2_t const *data, nk_size_t count) {
    nk_i64_t sum;
    nk_u64_t sumsq;
    nk_reduce_moments_i4(data, count, sizeof(nk_i4x2_t), &sum, &sumsq);
    return (nk_u32_t)sumsq;
}
NK_INTERNAL nk_u32_t nk_dots_reduce_sumsq_u4_(nk_u4x2_t const *data, nk_size_t count) {
    nk_u64_t sum, sumsq;
    nk_reduce_moments_u4(data, count, sizeof(nk_u4x2_t), &sum, &sumsq);
    return (nk_u32_t)sumsq;
}
NK_INTERNAL nk_u32_t nk_dots_reduce_sum_u1_(nk_u1x8_t const *data, nk_size_t count_bits) {
    nk_u64_t sum, sumsq;
    nk_reduce_moments_u1(data, count_bits, sizeof(nk_u1x8_t), &sum, &sumsq);
    return (nk_u32_t)sum;
}

/*  Combined moment trampolines for compensated GEMM.
 *  Each computes BOTH sum and norm (sum-of-squares) in a single nk_reduce_moments call.
 *  Used by nk_define_cross_compensated_pack_ to store both in the packed buffer.
 */
NK_INTERNAL void nk_dots_reduce_moments_i8_(nk_i8_t const *data, nk_size_t count, nk_i32_t *sum, nk_u32_t *norm) {
    nk_i64_t s;
    nk_u64_t sq;
    nk_reduce_moments_i8(data, count, sizeof(nk_i8_t), &s, &sq);
    *sum = (nk_i32_t)s;
    *norm = (nk_u32_t)sq;
}
NK_INTERNAL void nk_dots_reduce_moments_u8_(nk_u8_t const *data, nk_size_t count, nk_u32_t *sum, nk_u32_t *norm) {
    nk_u64_t s, sq;
    nk_reduce_moments_u8(data, count, sizeof(nk_u8_t), &s, &sq);
    *sum = (nk_u32_t)s;
    *norm = (nk_u32_t)sq;
}
NK_INTERNAL void nk_dots_reduce_moments_i4_(nk_i4x2_t const *data, nk_size_t count, nk_i32_t *sum, nk_u32_t *norm) {
    nk_i64_t s;
    nk_u64_t sq;
    nk_reduce_moments_i4(data, count, sizeof(nk_i4x2_t), &s, &sq);
    *sum = (nk_i32_t)s;
    *norm = (nk_u32_t)sq;
}

/*  A-row sum helpers for compensated GEMM finalization.
 *  i8/u8: no A-side correction needed, stubs return 0.
 *  i4: needs A-side sum for correction term.
 */
NK_INTERNAL nk_i32_t nk_dots_reduce_sum_i8_stub_(nk_i8_t const *d, nk_size_t c) {
    nk_unused_(d);
    nk_unused_(c);
    return 0;
}
NK_INTERNAL nk_i32_t nk_dots_reduce_sum_u8_stub_(nk_u8_t const *d, nk_size_t c) {
    nk_unused_(d);
    nk_unused_(c);
    return 0;
}
NK_INTERNAL nk_i32_t nk_dots_reduce_sum_i4_(nk_i4x2_t const *data, nk_size_t count) {
    nk_i64_t sum;
    nk_u64_t sumsq;
    nk_reduce_moments_i4(data, count, sizeof(nk_i4x2_t), &sum, &sumsq);
    return (nk_i32_t)sum;
}

/**
 *  @brief Generates function to calculate packed B matrix buffer size for GEMM micro-kernels.
 *
 *  Memory layout: B_packed[column_count, depth_padded] with header storing metadata.
 *  Buffer size: sizeof(header) + column_count × depth_padded × sizeof(intermediate_type) + column_count × sizeof(norm)
 *  Depth padding logic: Round up to `depth_simd_dimensions` multiple, then add `depth_simd_dimensions`
 *  if stride is power-of-2.
 *
 *  @param api_name Operation name (hammings, dots)
 *  @param input_type_name Original type's name of B matrix values (i4, f16, bf16, e4m3, e5m2, f32, etc.)
 *  @param isa_suffix Platform Instruct Set Architecture suffix (serial, haswell, icelake, etc.)
 *  @param input_type Original type of B matrix values (i4x2, f16, bf16, e4m3, e5m2, f32, etc.)
 *  @param intermediate_type Internal storage type in packed buffer (often bf16 or f32 for mixed precision)
 *  @param norm_value_type Type of per-column norm values (f32, f64, u32) appended after packed data
 *  @param depth_simd_dimensions SIMD vector width in values for this platform/type combination
 *  @param dimensions_per_value Number of logical dimensions in a single value of input_type_name.
 */
#define nk_define_cross_pack_size_(api_name, input_type_name, isa_suffix, input_value_type, packed_value_type,   \
                                   norm_value_type, depth_simd_dimensions, dimensions_per_value)                 \
    NK_PUBLIC nk_size_t nk_##api_name##_packed_size_##input_type_name##_##isa_suffix(nk_size_t column_count,     \
                                                                                     nk_size_t depth) {          \
        /* depth is always in logical dimensions (nibbles for i4, bytes for i8, etc.) */                         \
        /* depth_simd_dimensions is also in logical dimensions */                                                \
                                                                                                                 \
        /* Step 1: Pad depth in dimensions */                                                                    \
        nk_size_t depth_dimensions_padded = nk_size_round_up_to_multiple_(depth, depth_simd_dimensions);         \
                                                                                                                 \
        /* Step 2: Convert dimensions to storage values */                                                       \
        nk_size_t depth_values_padded = nk_size_divide_round_up_(depth_dimensions_padded, dimensions_per_value); \
                                                                                                                 \
        /* Step 3: Calculate stride in bytes for power-of-2 check */                                             \
        nk_size_t const stride_bytes = depth_values_padded * sizeof(nk_##packed_value_type##_t);                 \
                                                                                                                 \
        /* Step 4: Break power-of-2 strides for cache associativity */                                           \
        if ((stride_bytes & (stride_bytes - 1)) == 0 && stride_bytes > 0) {                                      \
            /* Add one SIMD step worth of storage values */                                                      \
            depth_values_padded += nk_size_divide_round_up_(depth_simd_dimensions, dimensions_per_value);        \
        }                                                                                                        \
                                                                                                                 \
        /* Step 5: Return total buffer size (packed data + per-column norms) */                                  \
        return sizeof(nk_cross_packed_buffer_header_t) +                                                         \
               column_count * depth_values_padded * sizeof(nk_##packed_value_type##_t) +                         \
               column_count * sizeof(nk_##norm_value_type##_t);                                                  \
    }

/**
 *  @brief Generates pack function using SIMD load/store helpers.
 *
 *  Packs the B matrix into padded row-major layout with optional type conversion,
 *  using vectorized load/store for the bulk copy and a small scalar tail for padding.
 *
 *  @param vec_type SIMD vector type (nk_b512_vec_t, nk_b256_vec_t, nk_b128_vec_t)
 *  @param load_fn Full load: void fn(void const*, vec_type*)
 *  @param partial_load_fn Masked/partial load: void fn(void const*, vec_type*, nk_size_t)
 *  @param store_fn Full store: void fn(vec_type const*, void*)
 *  @param partial_store_fn Masked/partial store: void fn(vec_type const*, void*, nk_size_t)
 *  @param simd_width Elements per SIMD load/store operation
 */
#define nk_define_cross_pack_(api_name, input_type_name, isa_suffix, input_value_type, packed_value_type, vec_type,   \
                              load_fn, partial_load_fn, store_fn, partial_store_fn, simd_width, norm_value_type,      \
                              compute_norm_fn, depth_simd_dimensions, dimensions_per_value)                           \
    NK_PUBLIC void nk_##api_name##_pack_##input_type_name##_##isa_suffix(                                             \
        nk_##input_value_type##_t const *b, nk_size_t column_count, nk_size_t depth, nk_size_t b_stride_in_bytes,     \
        void *b_packed) {                                                                                             \
        nk_size_t depth_dimensions_padded = nk_size_round_up_to_multiple_(depth, depth_simd_dimensions);              \
        nk_size_t depth_values_padded = nk_size_divide_round_up_(depth_dimensions_padded, dimensions_per_value);      \
        nk_size_t const stride_bytes = depth_values_padded * sizeof(nk_##packed_value_type##_t);                      \
        if ((stride_bytes & (stride_bytes - 1)) == 0 && stride_bytes > 0)                                             \
            depth_values_padded += nk_size_divide_round_up_(depth_simd_dimensions, dimensions_per_value);             \
        nk_size_t const depth_in_values = nk_size_divide_round_up_(depth, dimensions_per_value);                      \
                                                                                                                      \
        nk_cross_packed_buffer_header_t *header = (nk_cross_packed_buffer_header_t *)b_packed;                        \
        header->column_count = (nk_u32_t)column_count;                                                                \
        header->depth_dimensions = (nk_u32_t)depth;                                                                   \
        header->depth_padded_values = (nk_u32_t)depth_values_padded;                                                  \
                                                                                                                      \
        nk_##packed_value_type##_t *packed = (nk_##packed_value_type##_t *)((char *)b_packed +                        \
                                                                            sizeof(nk_cross_packed_buffer_header_t)); \
        nk_size_t const full_chunks = depth_in_values / (simd_width);                                                 \
        nk_size_t const remainder = depth_in_values % (simd_width);                                                   \
                                                                                                                      \
        for (nk_size_t column_index = 0; column_index < column_count; ++column_index) {                               \
            nk_##input_value_type##_t const *source_row =                                                             \
                (nk_##input_value_type##_t const *)((char const *)b + column_index * b_stride_in_bytes);              \
            nk_##packed_value_type##_t *destination_row = packed + column_index * depth_values_padded;                \
            for (nk_size_t chunk = 0; chunk < full_chunks; ++chunk) {                                                 \
                vec_type vec;                                                                                         \
                load_fn(source_row + chunk * (simd_width), &vec);                                                     \
                store_fn(&vec, destination_row + chunk * (simd_width));                                               \
            }                                                                                                         \
            if (remainder > 0) {                                                                                      \
                vec_type vec;                                                                                         \
                partial_load_fn(source_row + full_chunks * (simd_width), &vec, remainder);                            \
                partial_store_fn(&vec, destination_row + full_chunks * (simd_width), remainder);                      \
            }                                                                                                         \
            for (nk_size_t pad = depth_in_values; pad < depth_values_padded; ++pad) destination_row[pad] = 0;         \
        }                                                                                                             \
                                                                                                                      \
        nk_size_t const total_values = column_count * depth_values_padded;                                            \
        nk_##norm_value_type##_t *norms = (nk_##norm_value_type##_t *)(packed + total_values);                        \
        for (nk_size_t column_index = 0; column_index < column_count; ++column_index) {                               \
            nk_##input_value_type##_t const *source_row =                                                             \
                (nk_##input_value_type##_t const *)((char const *)b + column_index * b_stride_in_bytes);              \
            norms[column_index] = compute_norm_fn(source_row, depth);                                                 \
        }                                                                                                             \
    }

/**
 *  @brief Generates function to calculate packed B matrix buffer size for compensated GEMM.
 *
 *  Like nk_define_cross_pack_size_ but the buffer stores BOTH norms AND column sums.
 *  Layout: [ Header 64B ] [ Packed data ] [ Norms (norm_type) ] [ Column sums (sum_type) ]
 *  Norms first → existing nk_define_cross_normalized_packed_ reads norms at the same offset.
 */
#define nk_define_cross_compensated_pack_size_(api_name, input_type_name, isa_suffix, input_value_type,          \
                                               packed_value_type, sum_value_type, norm_value_type,               \
                                               depth_simd_dimensions, dimensions_per_value)                      \
    NK_PUBLIC nk_size_t nk_##api_name##_packed_size_##input_type_name##_##isa_suffix(nk_size_t column_count,     \
                                                                                     nk_size_t depth) {          \
        nk_size_t depth_dimensions_padded = nk_size_round_up_to_multiple_(depth, depth_simd_dimensions);         \
        nk_size_t depth_values_padded = nk_size_divide_round_up_(depth_dimensions_padded, dimensions_per_value); \
        nk_size_t const stride_bytes = depth_values_padded * sizeof(nk_##packed_value_type##_t);                 \
        if ((stride_bytes & (stride_bytes - 1)) == 0 && stride_bytes > 0) {                                      \
            depth_values_padded += nk_size_divide_round_up_(depth_simd_dimensions, dimensions_per_value);        \
        }                                                                                                        \
        return sizeof(nk_cross_packed_buffer_header_t) +                                                         \
               column_count * depth_values_padded * sizeof(nk_##packed_value_type##_t) +                         \
               column_count * sizeof(nk_##norm_value_type##_t) + column_count * sizeof(nk_##sum_value_type##_t); \
    }

/**
 *  @brief Like nk_define_cross_pack_ but stores both per-column norms AND column sums.
 *  Layout: [ Header 64B ] [ Packed data ] [ Norms (norm_type) ] [ Column sums (sum_type) ]
 */
#define nk_define_cross_compensated_pack_(api_name, input_type_name, isa_suffix, input_value_type, packed_value_type, \
                                          vec_type, load_fn, partial_load_fn, store_fn, partial_store_fn, simd_width, \
                                          sum_value_type, norm_value_type, compute_moments_fn, depth_simd_dimensions, \
                                          dimensions_per_value)                                                       \
    NK_PUBLIC void nk_##api_name##_pack_##input_type_name##_##isa_suffix(                                             \
        nk_##input_value_type##_t const *b, nk_size_t column_count, nk_size_t depth, nk_size_t b_stride_in_bytes,     \
        void *b_packed) {                                                                                             \
        nk_size_t depth_dimensions_padded = nk_size_round_up_to_multiple_(depth, depth_simd_dimensions);              \
        nk_size_t depth_values_padded = nk_size_divide_round_up_(depth_dimensions_padded, dimensions_per_value);      \
        nk_size_t const stride_bytes = depth_values_padded * sizeof(nk_##packed_value_type##_t);                      \
        if ((stride_bytes & (stride_bytes - 1)) == 0 && stride_bytes > 0)                                             \
            depth_values_padded += nk_size_divide_round_up_(depth_simd_dimensions, dimensions_per_value);             \
        nk_size_t const depth_in_values = nk_size_divide_round_up_(depth, dimensions_per_value);                      \
                                                                                                                      \
        nk_cross_packed_buffer_header_t *header = (nk_cross_packed_buffer_header_t *)b_packed;                        \
        header->column_count = (nk_u32_t)column_count;                                                                \
        header->depth_dimensions = (nk_u32_t)depth;                                                                   \
        header->depth_padded_values = (nk_u32_t)depth_values_padded;                                                  \
                                                                                                                      \
        nk_##packed_value_type##_t *packed = (nk_##packed_value_type##_t *)((char *)b_packed +                        \
                                                                            sizeof(nk_cross_packed_buffer_header_t)); \
        nk_size_t const full_chunks = depth_in_values / (simd_width);                                                 \
        nk_size_t const remainder = depth_in_values % (simd_width);                                                   \
                                                                                                                      \
        for (nk_size_t column_index = 0; column_index < column_count; ++column_index) {                               \
            nk_##input_value_type##_t const *source_row =                                                             \
                (nk_##input_value_type##_t const *)((char const *)b + column_index * b_stride_in_bytes);              \
            nk_##packed_value_type##_t *destination_row = packed + column_index * depth_values_padded;                \
            for (nk_size_t chunk = 0; chunk < full_chunks; ++chunk) {                                                 \
                vec_type vec;                                                                                         \
                load_fn(source_row + chunk * (simd_width), &vec);                                                     \
                store_fn(&vec, destination_row + chunk * (simd_width));                                               \
            }                                                                                                         \
            if (remainder > 0) {                                                                                      \
                vec_type vec;                                                                                         \
                partial_load_fn(source_row + full_chunks * (simd_width), &vec, remainder);                            \
                partial_store_fn(&vec, destination_row + full_chunks * (simd_width), remainder);                      \
            }                                                                                                         \
            for (nk_size_t pad = depth_in_values; pad < depth_values_padded; ++pad) destination_row[pad] = 0;         \
        }                                                                                                             \
                                                                                                                      \
        nk_size_t const total_values = column_count * depth_values_padded;                                            \
        nk_##norm_value_type##_t *norms = (nk_##norm_value_type##_t *)(packed + total_values);                        \
        nk_##sum_value_type##_t *col_sums = (nk_##sum_value_type##_t *)(norms + column_count);                        \
        for (nk_size_t column_index = 0; column_index < column_count; ++column_index) {                               \
            nk_##input_value_type##_t const *source_row =                                                             \
                (nk_##input_value_type##_t const *)((char const *)b + column_index * b_stride_in_bytes);              \
            compute_moments_fn(source_row, depth, &col_sums[column_index], &norms[column_index]);                     \
        }                                                                                                             \
    }

/**
 *  @brief Generates optimized GEMM implementation: C = A × Bᵀ with pre-packed B matrix.
 *
 *  This macro creates a complete batched matrix multiplication kernel with THREE specialized
 *  code paths that are automatically selected based on the remaining work at each blocking level.
 *  The kernel requires B to be pre-packed using nk_define_cross_pack_ before invocation.
 *
 *  @par Mathematical Operation
 *    C[row_count, column_count] = A[row_count, depth] × Bᵀ[column_count, depth]
 *  where operation can be dot product, Hamming distance, Jaccard similarity, etc.
 *
 *  @par Three Kernel Variants for Adaptive Performance
 *
 *  1. @b 4×4 @b register @b tile @b kernel (primary path, ~80% of work):
 *     - Processes 4 rows of A × 4 columns of B simultaneously
 *     - Maintains 16 independent accumulators in registers (state_type[4][4])
 *     - Achieves maximum instruction-level parallelism (16 FMAs per depth iteration)
 *     - Used when: row_count ≥ 4 AND column_count ≥ 4
 *     - Performance: Peak throughput, optimal register utilization
 *
 *  2. @b 1×8 @b register @b tile @b kernel (edge case, ~15% of work):
 *     - Processes 1 row of A × 8 columns of B when remaining rows < 4
 *     - Maintains 8 independent accumulators (state_type[1][8])
 *     - Balances vectorization with low row count
 *     - Used when: row_count < 4 AND column_count ≥ 8
 *     - Performance: Better throughput than generic fallback for wide matrices
 *
 *  3. @b Generic @b fallback @b kernel (edge cases, ~5% of work):
 *     - Handles all irregular cases (row_count < 4 AND column_count < 8)
 *     - Single accumulator, minimal unrolling
 *     - Used for: Small tiles, remainder handling
 *     - Performance: Lower throughput but handles all edge cases correctly
 *
 *  @par Cache Blocking Strategy (No Depth Blocking)
 *
 *  Unlike traditional GEMM which blocks all three dimensions (M, N, K), this implementation
 *  deliberately omits depth (K) blocking for several reasons:
 *
 *  1. @b Streaming @b access @b pattern: A and B are read sequentially along depth dimension
 *     - Prefetcher-friendly access (hardware prefetch works well)
 *     - No cache reuse along depth within a single C[i,j] computation
 *
 *  2. @b Depth @b is @b typically @b small: For ML inference, depth is often 128-4096 values
 *     - Fits in L2/L3 cache for single row of A
 *     - B is pre-packed for optimal spatial locality
 *
 *  3. @b Simplicity @b and @b instruction @b cache @b efficiency:
 *     - Fewer nested loops = better instruction cache utilization
 *     - Simpler control flow = easier for compiler to optimize
 *
 *  @par Pre-Packing Benefits
 *
 *  B matrix is pre-packed using nk_define_cross_pack_ before kernel invocation:
 *  - @b Type @b conversion @b amortization: Convert B values once (e.g., bf16→f32) rather than
 *    per A row access. Saves (row_count - 1) × column_count conversions.
 *  - @b Cache @b line @b optimization: Pad depth to break power-of-2 strides that cause cache
 *    associativity conflicts (e.g., 8192 → 8200 values).
 *  - @b Spatial @b locality: Transpose B so columns are contiguous, enabling efficient SIMD loads.
 *
 *  @par Loop Structure
 *
 *    for column_block in columns (step: varies based on available columns):
 *      for row_block in rows (step: varies based on available rows):
 *        for row_tile in row_block (step: 4 or 1 depending on variant):
 *          for column_tile in column_block (step: 4 or 8 depending on variant):
 *            accumulator_tiles[row_tile][column_tile] = init_accumulator_fn()
 *            for depth_index in depth (step: depth_simd_dimensions):
 *              a_vectors = load_a_vec_fn(A[row_tile, depth_index])
 *              b_vectors = load_b_vec_fn(B_packed[column_tile, depth_index])
 *              accumulator_tiles = inner_product_fn(accumulator_tiles, a_vectors, b_vectors)
 *            results = reduce_accumulators_fn(accumulator_tiles)
 *            partial_store_fn(results, C[row_tile, column_tile])
 *
 *  @par Generated Function
 *
 *  nk_##api_name##_packed_##input_type_name##_##isa_suffix##_aligned_(
 *      A_matrix, B_packed_buffer, C_matrix, row_count, column_count, depth,
 *      A_stride_bytes, C_stride_bytes)
 *
 *  @param api_name Operation family (dots, hammings, jaccards) for codegen namespace
 *  @param input_type_name Type identifier for codegen (f32, bf16, i8, u1, etc.)
 *  @param isa_suffix ISA backend identifier (serial, haswell, neon, sve, icelake, etc.)
 *  @param input_type C type of input matrix values (f32, bf16, i8, u1x8, etc.)
 *  @param intermediate_type Storage type in packed B buffer (often bf16 or f32 for mixed precision)
 *  @param output_type C type of output matrix C values (f32, u32, f64, etc.)
 *  @param vec_type SIMD vector type for depth dimension (e.g., __m256, nk_f32x8_t)
 *  @param state_type Accumulator state type (often vec_type or wider, e.g., __m256 or __m512)
 *  @param result_vec_type SIMD vector type for reduction results (e.g., __m128 for 4 f32 results)
 *  @param init_accumulator_fn Initialize accumulator: void fn(state_type*)
 *  @param load_a_vec_fn Full A vector load: vec_type fn(input_type const*, nk_size_t offset)
 *  @param partial_load_a_vec_fn Partial A load for remainder
 *  @param load_b_vec_fn Full B vector load: vec_type fn(intermediate_type const*, nk_size_t offset)
 *  @param partial_load_b_vec_fn Partial B load for remainder
 *  @param inner_product_fn Inner product accumulate
 *  @param reduce_accumulators_fn Reduce 4 accumulators
 *  @param store_fn Full-width store for results
 *  @param store_fn Full-width store for results
 *  @param partial_store_fn Partial store for results
 *  @param depth_simd_dimensions SIMD vector width in logical dimensions (e.g., 8 for f32 on AVX2, 128 for u1 on serial)
 *  @param dimensions_per_value Packing ratio: dimensions per storage value (1 for f32, 2 for i4x2, 8 for u1x8)
 *
 *  @sa nk_define_cross_symmetric_ for symmetric C = A × Aᵀ computation (upper triangle only)
 *  @sa nk_define_cross_pack_size_ for calculating B_packed buffer size
 *  @sa nk_define_cross_pack_ for packing B matrix into optimized layout
 *  @sa include/numkong/set/serial.h for state type definitions
 *  @sa include/numkong/cast/serial.h for load/store function implementations
 */
#define nk_define_cross_packed_(api_name, input_type_name, isa_suffix, input_value_type, packed_value_type,            \
                                result_value_type, vec_type, state_type, result_vec_type, init_accumulator_fn,         \
                                load_a_vec_fn, partial_load_a_vec_fn, load_b_vec_fn, partial_load_b_vec_fn,            \
                                inner_product_fn, reduce_accumulators_fn, store_fn, partial_store_fn,                  \
                                depth_simd_dimensions, dimensions_per_value)                                           \
    NK_INTERNAL void nk_##api_name##_packed_##input_type_name##_##isa_suffix##_aligned_(                               \
        nk_##input_value_type##_t const *a_matrix, void const *b_packed_buffer, nk_##result_value_type##_t *c_matrix,  \
        nk_size_t row_count, nk_size_t column_count, nk_size_t depth, nk_size_t a_stride_in_bytes,                     \
        nk_size_t c_stride_in_bytes) {                                                                                 \
        /* Read padded depth from header for correct stride calculation */                                             \
        nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed_buffer;      \
        nk_size_t const depth_padded = header->depth_padded_values;                                                    \
                                                                                                                       \
        nk_##packed_value_type##_t const *packed_data =                                                                \
            (nk_##packed_value_type##_t const *)((char const *)b_packed_buffer +                                       \
                                                 sizeof(nk_cross_packed_buffer_header_t));                             \
                                                                                                                       \
        /* Cache blocking parameters (no depth_block blocking - full depth accumulated per tile) */                    \
        nk_size_t const row_block_size = 128;      /* L2 cache blocking over rows */                                   \
        nk_size_t const column_block_size = 2048;  /* L3 cache blocking over columns */                                \
        nk_size_t const register_row_count = 4;    /* Rows per register tile */                                        \
        nk_size_t const register_column_count = 4; /* Columns per register tile */                                     \
        /* Correct aligned_depth calculation for sub-byte types */                                                     \
        nk_size_t const depth_dimensions_aligned = (depth / depth_simd_dimensions) * depth_simd_dimensions;            \
        nk_size_t const aligned_depth = nk_size_divide_round_up_(depth_dimensions_aligned, dimensions_per_value);      \
        /* Calculate step size in storage values for loop increment */                                                 \
        nk_size_t const depth_step_values = nk_size_divide_round_up_(depth_simd_dimensions, dimensions_per_value);     \
                                                                                                                       \
        /* Zero output matrix */                                                                                       \
        for (nk_size_t row_index = 0; row_index < row_count; ++row_index) {                                            \
            nk_##result_value_type##_t *c_row = (nk_##result_value_type##_t *)((char *)c_matrix +                      \
                                                                               row_index * c_stride_in_bytes);         \
            for (nk_size_t column_index = 0; column_index < column_count; ++column_index) c_row[column_index] = 0;     \
        }                                                                                                              \
                                                                                                                       \
        /* Loop 1: L3 cache blocking over columns */                                                                   \
        for (nk_size_t column_block_start_index = 0; column_block_start_index < column_count;                          \
             column_block_start_index += column_block_size) {                                                          \
            nk_size_t column_block_end_index = column_block_start_index + column_block_size;                           \
            if (column_block_end_index > column_count) column_block_end_index = column_count;                          \
                                                                                                                       \
            /* Loop 2: L2 cache blocking over rows */                                                                  \
            for (nk_size_t row_block_start_index = 0; row_block_start_index < row_count;                               \
                 row_block_start_index += row_block_size) {                                                            \
                nk_size_t row_block_end_index = row_block_start_index + row_block_size;                                \
                if (row_block_end_index > row_count) row_block_end_index = row_count;                                  \
                                                                                                                       \
                /* Loop 3: Register tiling over columns (register_column_count columns per batch) */                   \
                for (nk_size_t tile_column_start_index = column_block_start_index;                                     \
                     tile_column_start_index < column_block_end_index;                                                 \
                     tile_column_start_index += register_column_count) {                                               \
                                                                                                                       \
                    /* Compute B pointers once per column tile - direct column-major addressing */                     \
                    nk_##packed_value_type##_t const *b_depth_ptr_0 = packed_data +                                    \
                                                                      (tile_column_start_index + 0) * depth_padded;    \
                    nk_##packed_value_type##_t const *b_depth_ptr_1 = packed_data +                                    \
                                                                      (tile_column_start_index + 1) * depth_padded;    \
                    nk_##packed_value_type##_t const *b_depth_ptr_2 = packed_data +                                    \
                                                                      (tile_column_start_index + 2) * depth_padded;    \
                    nk_##packed_value_type##_t const *b_depth_ptr_3 = packed_data +                                    \
                                                                      (tile_column_start_index + 3) * depth_padded;    \
                                                                                                                       \
                    /* Loop 4: Register tiling over rows (register_row_count rows per tile) */                         \
                    for (nk_size_t tile_row_start_index = row_block_start_index;                                       \
                         tile_row_start_index < row_block_end_index; tile_row_start_index += register_row_count) {     \
                                                                                                                       \
                        /* Initialize register_row_count × register_column_count accumulator states */                 \
                        state_type accumulator_tiles[4][4];                                                            \
                        init_accumulator_fn(&accumulator_tiles[0][0]), init_accumulator_fn(&accumulator_tiles[0][1]),  \
                            init_accumulator_fn(&accumulator_tiles[0][2]),                                             \
                            init_accumulator_fn(&accumulator_tiles[0][3]);                                             \
                        init_accumulator_fn(&accumulator_tiles[1][0]), init_accumulator_fn(&accumulator_tiles[1][1]),  \
                            init_accumulator_fn(&accumulator_tiles[1][2]),                                             \
                            init_accumulator_fn(&accumulator_tiles[1][3]);                                             \
                        init_accumulator_fn(&accumulator_tiles[2][0]), init_accumulator_fn(&accumulator_tiles[2][1]),  \
                            init_accumulator_fn(&accumulator_tiles[2][2]),                                             \
                            init_accumulator_fn(&accumulator_tiles[2][3]);                                             \
                        init_accumulator_fn(&accumulator_tiles[3][0]), init_accumulator_fn(&accumulator_tiles[3][1]),  \
                            init_accumulator_fn(&accumulator_tiles[3][2]),                                             \
                            init_accumulator_fn(&accumulator_tiles[3][3]);                                             \
                                                                                                                       \
                        /* A row pointers */                                                                           \
                        nk_##input_value_type##_t const *a_row_ptr_0 =                                                 \
                            (nk_##input_value_type##_t const *)((char const *)a_matrix +                               \
                                                                (tile_row_start_index + 0) * a_stride_in_bytes);       \
                        nk_##input_value_type##_t const *a_row_ptr_1 =                                                 \
                            (nk_##input_value_type##_t const *)((char const *)a_matrix +                               \
                                                                (tile_row_start_index + 1) * a_stride_in_bytes);       \
                        nk_##input_value_type##_t const *a_row_ptr_2 =                                                 \
                            (nk_##input_value_type##_t const *)((char const *)a_matrix +                               \
                                                                (tile_row_start_index + 2) * a_stride_in_bytes);       \
                        nk_##input_value_type##_t const *a_row_ptr_3 =                                                 \
                            (nk_##input_value_type##_t const *)((char const *)a_matrix +                               \
                                                                (tile_row_start_index + 3) * a_stride_in_bytes);       \
                                                                                                                       \
                        /* Tight inner loop: full depth with simple depth_index addressing */                          \
                        vec_type a_vector_0, a_vector_1, a_vector_2, a_vector_3;                                       \
                        vec_type b_vector_0, b_vector_1, b_vector_2, b_vector_3;                                       \
                        for (nk_size_t depth_index = 0; depth_index < aligned_depth;                                   \
                             depth_index += depth_step_values) {                                                       \
                            /* Load next few values from 4 rows from A (unpacked, may upcast) */                       \
                            load_a_vec_fn(a_row_ptr_0 + depth_index, &a_vector_0);                                     \
                            load_a_vec_fn(a_row_ptr_1 + depth_index, &a_vector_1);                                     \
                            load_a_vec_fn(a_row_ptr_2 + depth_index, &a_vector_2);                                     \
                            load_a_vec_fn(a_row_ptr_3 + depth_index, &a_vector_3);                                     \
                                                                                                                       \
                            /* Load next few values from 4 rows from B (packed, already upcasted) */                   \
                            load_b_vec_fn(b_depth_ptr_0 + depth_index, &b_vector_0);                                   \
                            load_b_vec_fn(b_depth_ptr_1 + depth_index, &b_vector_1);                                   \
                            load_b_vec_fn(b_depth_ptr_2 + depth_index, &b_vector_2);                                   \
                            load_b_vec_fn(b_depth_ptr_3 + depth_index, &b_vector_3);                                   \
                                                                                                                       \
                            /* 16 FMAs: 4 A rows × 4 B columns */                                                      \
                            inner_product_fn(&accumulator_tiles[0][0], a_vector_0, b_vector_0,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[0][1], a_vector_0, b_vector_1,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[0][2], a_vector_0, b_vector_2,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[0][3], a_vector_0, b_vector_3,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[1][0], a_vector_1, b_vector_0,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[1][1], a_vector_1, b_vector_1,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[1][2], a_vector_1, b_vector_2,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[1][3], a_vector_1, b_vector_3,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[2][0], a_vector_2, b_vector_0,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[2][1], a_vector_2, b_vector_1,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[2][2], a_vector_2, b_vector_2,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[2][3], a_vector_2, b_vector_3,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[3][0], a_vector_3, b_vector_0,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[3][1], a_vector_3, b_vector_1,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[3][2], a_vector_3, b_vector_2,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                            inner_product_fn(&accumulator_tiles[3][3], a_vector_3, b_vector_3,                         \
                                             depth_index * dimensions_per_value, depth_simd_dimensions);               \
                        }                                                                                              \
                        /* Finalize and store register_rows x register_cols results using batched 4-way reduction */   \
                        result_vec_type result_vector;                                                                 \
                        nk_##result_value_type##_t *c_row_ptr_0 =                                                      \
                            (nk_##result_value_type##_t *)((char *)c_matrix +                                          \
                                                           (tile_row_start_index + 0) * c_stride_in_bytes);            \
                        reduce_accumulators_fn(&accumulator_tiles[0][0], &accumulator_tiles[0][1],                     \
                                               &accumulator_tiles[0][2], &accumulator_tiles[0][3], depth,              \
                                               &result_vector);                                                        \
                        store_fn(&result_vector, c_row_ptr_0 + tile_column_start_index);                               \
                        nk_##result_value_type##_t *c_row_ptr_1 =                                                      \
                            (nk_##result_value_type##_t *)((char *)c_matrix +                                          \
                                                           (tile_row_start_index + 1) * c_stride_in_bytes);            \
                        reduce_accumulators_fn(&accumulator_tiles[1][0], &accumulator_tiles[1][1],                     \
                                               &accumulator_tiles[1][2], &accumulator_tiles[1][3], depth,              \
                                               &result_vector);                                                        \
                        store_fn(&result_vector, c_row_ptr_1 + tile_column_start_index);                               \
                        nk_##result_value_type##_t *c_row_ptr_2 =                                                      \
                            (nk_##result_value_type##_t *)((char *)c_matrix +                                          \
                                                           (tile_row_start_index + 2) * c_stride_in_bytes);            \
                        reduce_accumulators_fn(&accumulator_tiles[2][0], &accumulator_tiles[2][1],                     \
                                               &accumulator_tiles[2][2], &accumulator_tiles[2][3], depth,              \
                                               &result_vector);                                                        \
                        store_fn(&result_vector, c_row_ptr_2 + tile_column_start_index);                               \
                        nk_##result_value_type##_t *c_row_ptr_3 =                                                      \
                            (nk_##result_value_type##_t *)((char *)c_matrix +                                          \
                                                           (tile_row_start_index + 3) * c_stride_in_bytes);            \
                        reduce_accumulators_fn(&accumulator_tiles[3][0], &accumulator_tiles[3][1],                     \
                                               &accumulator_tiles[3][2], &accumulator_tiles[3][3], depth,              \
                                               &result_vector);                                                        \
                        store_fn(&result_vector, c_row_ptr_3 + tile_column_start_index);                               \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    NK_INTERNAL void nk_##api_name##_packed_##input_type_name##_##isa_suffix##_1x8_aligned_(                           \
        nk_##input_value_type##_t const *a_matrix, void const *b_packed_buffer, nk_##result_value_type##_t *c_matrix,  \
        nk_size_t row_count, nk_size_t column_count, nk_size_t depth, nk_size_t a_stride_in_bytes,                     \
        nk_size_t c_stride_in_bytes) {                                                                                 \
        /* Read padded depth from header for correct stride calculation */                                             \
        nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed_buffer;      \
        nk_size_t const depth_padded = header->depth_padded_values; /* in storage values */                            \
                                                                                                                       \
        nk_##packed_value_type##_t const *packed_data =                                                                \
            (nk_##packed_value_type##_t const *)((char const *)b_packed_buffer +                                       \
                                                 sizeof(nk_cross_packed_buffer_header_t));                             \
                                                                                                                       \
        /* Cache blocking parameters (no depth_block blocking - full depth accumulated per tile) */                    \
        nk_size_t const row_block_size = 128;      /* L2 cache blocking over rows */                                   \
        nk_size_t const column_block_size = 2048;  /* L3 cache blocking over columns */                                \
        nk_size_t const register_row_count = 1;    /* Rows per register tile */                                        \
        nk_size_t const register_column_count = 8; /* Columns per register tile (2 × 4) */                             \
        /* Correct aligned_depth calculation for sub-byte types */                                                     \
        nk_size_t const depth_dimensions_aligned = (depth / depth_simd_dimensions) * depth_simd_dimensions;            \
        nk_size_t const aligned_depth = nk_size_divide_round_up_(depth_dimensions_aligned, dimensions_per_value);      \
        /* Calculate step size in storage values for loop increment */                                                 \
        nk_size_t const depth_step_values = nk_size_divide_round_up_(depth_simd_dimensions, dimensions_per_value);     \
        nk_unused_(register_row_count); /* Used in comments, loop uses 1 directly */                                   \
                                                                                                                       \
        /* Zero output matrix */                                                                                       \
        for (nk_size_t row_index = 0; row_index < row_count; ++row_index) {                                            \
            nk_##result_value_type##_t *c_row = (nk_##result_value_type##_t *)((char *)c_matrix +                      \
                                                                               row_index * c_stride_in_bytes);         \
            for (nk_size_t column_index = 0; column_index < column_count; ++column_index) c_row[column_index] = 0;     \
        }                                                                                                              \
                                                                                                                       \
        /* Loop 1: L3 cache blocking over columns */                                                                   \
        for (nk_size_t column_block_start_index = 0; column_block_start_index < column_count;                          \
             column_block_start_index += column_block_size) {                                                          \
            nk_size_t column_block_end_index = column_block_start_index + column_block_size;                           \
            if (column_block_end_index > column_count) column_block_end_index = column_count;                          \
                                                                                                                       \
            /* Loop 2: L2 cache blocking over rows */                                                                  \
            for (nk_size_t row_block_start_index = 0; row_block_start_index < row_count;                               \
                 row_block_start_index += row_block_size) {                                                            \
                nk_size_t const row_block_end_index = row_block_start_index + row_block_size < row_count               \
                                                          ? row_block_start_index + row_block_size                     \
                                                          : row_count;                                                 \
                                                                                                                       \
                /* Loop 3: Register tiling over columns (register_column_count columns per batch) */                   \
                for (nk_size_t tile_column_start_index = column_block_start_index;                                     \
                     tile_column_start_index < column_block_end_index;                                                 \
                     tile_column_start_index += register_column_count) {                                               \
                                                                                                                       \
                    /* Compute B pointers once per column tile - direct column-major addressing */                     \
                    nk_##packed_value_type##_t const *b_depth_ptr_0 = packed_data +                                    \
                                                                      (tile_column_start_index + 0) * depth_padded;    \
                    nk_##packed_value_type##_t const *b_depth_ptr_1 = packed_data +                                    \
                                                                      (tile_column_start_index + 1) * depth_padded;    \
                    nk_##packed_value_type##_t const *b_depth_ptr_2 = packed_data +                                    \
                                                                      (tile_column_start_index + 2) * depth_padded;    \
                    nk_##packed_value_type##_t const *b_depth_ptr_3 = packed_data +                                    \
                                                                      (tile_column_start_index + 3) * depth_padded;    \
                    nk_##packed_value_type##_t const *b_depth_ptr_4 = packed_data +                                    \
                                                                      (tile_column_start_index + 4) * depth_padded;    \
                    nk_##packed_value_type##_t const *b_depth_ptr_5 = packed_data +                                    \
                                                                      (tile_column_start_index + 5) * depth_padded;    \
                    nk_##packed_value_type##_t const *b_depth_ptr_6 = packed_data +                                    \
                                                                      (tile_column_start_index + 6) * depth_padded;    \
                    nk_##packed_value_type##_t const *b_depth_ptr_7 = packed_data +                                    \
                                                                      (tile_column_start_index + 7) * depth_padded;    \
                                                                                                                       \
                    /* Loop 4: Process 1 row at a time */                                                              \
                    for (nk_size_t row_index = row_block_start_index; row_index < row_block_end_index; ++row_index) {  \
                                                                                                                       \
                        /* Initialize 1 × 8 accumulator states */                                                      \
                        state_type accumulator_0, accumulator_1, accumulator_2, accumulator_3, accumulator_4,          \
                            accumulator_5, accumulator_6, accumulator_7;                                               \
                        init_accumulator_fn(&accumulator_0), init_accumulator_fn(&accumulator_1),                      \
                            init_accumulator_fn(&accumulator_2), init_accumulator_fn(&accumulator_3),                  \
                            init_accumulator_fn(&accumulator_4), init_accumulator_fn(&accumulator_5),                  \
                            init_accumulator_fn(&accumulator_6), init_accumulator_fn(&accumulator_7);                  \
                                                                                                                       \
                        /* A row pointer */                                                                            \
                        nk_##input_value_type##_t const *a_row_ptr =                                                   \
                            (nk_##input_value_type##_t const *)((char const *)a_matrix +                               \
                                                                row_index * a_stride_in_bytes);                        \
                                                                                                                       \
                        /* Tight inner loop: full depth with simple depth_index addressing */                          \
                        vec_type a_vector;                                                                             \
                        vec_type b_vector_0, b_vector_1, b_vector_2, b_vector_3, b_vector_4, b_vector_5, b_vector_6,   \
                            b_vector_7;                                                                                \
                        for (nk_size_t depth_index = 0; depth_index < aligned_depth;                                   \
                             depth_index += depth_step_values) {                                                       \
                            /* Load A vector (1 row) */                                                                \
                            load_a_vec_fn(a_row_ptr + depth_index, &a_vector);                                         \
                                                                                                                       \
                            /* Load B vectors (8 columns) */                                                           \
                            load_b_vec_fn(b_depth_ptr_0 + depth_index, &b_vector_0);                                   \
                            load_b_vec_fn(b_depth_ptr_1 + depth_index, &b_vector_1);                                   \
                            load_b_vec_fn(b_depth_ptr_2 + depth_index, &b_vector_2);                                   \
                            load_b_vec_fn(b_depth_ptr_3 + depth_index, &b_vector_3);                                   \
                            load_b_vec_fn(b_depth_ptr_4 + depth_index, &b_vector_4);                                   \
                            load_b_vec_fn(b_depth_ptr_5 + depth_index, &b_vector_5);                                   \
                            load_b_vec_fn(b_depth_ptr_6 + depth_index, &b_vector_6);                                   \
                            load_b_vec_fn(b_depth_ptr_7 + depth_index, &b_vector_7);                                   \
                                                                                                                       \
                            /* 8 FMAs: 1 A row × 8 B columns */                                                        \
                            inner_product_fn(&accumulator_0, a_vector, b_vector_0, depth_index * dimensions_per_value, \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulator_1, a_vector, b_vector_1, depth_index * dimensions_per_value, \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulator_2, a_vector, b_vector_2, depth_index * dimensions_per_value, \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulator_3, a_vector, b_vector_3, depth_index * dimensions_per_value, \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulator_4, a_vector, b_vector_4, depth_index * dimensions_per_value, \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulator_5, a_vector, b_vector_5, depth_index * dimensions_per_value, \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulator_6, a_vector, b_vector_6, depth_index * dimensions_per_value, \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulator_7, a_vector, b_vector_7, depth_index * dimensions_per_value, \
                                             depth_simd_dimensions);                                                   \
                        }                                                                                              \
                                                                                                                       \
                        /* Finalize and store 1 × 8 results using two 4-way reductions */                              \
                        result_vec_type result_vector;                                                                 \
                        nk_##result_value_type##_t *c_row_ptr =                                                        \
                            (nk_##result_value_type##_t *)((char *)c_matrix + row_index * c_stride_in_bytes);          \
                        /* First 4 columns */                                                                          \
                        reduce_accumulators_fn(&accumulator_0, &accumulator_1, &accumulator_2, &accumulator_3, depth,  \
                                               &result_vector);                                                        \
                        store_fn(&result_vector, c_row_ptr + tile_column_start_index);                                 \
                        /* Second 4 columns */                                                                         \
                        reduce_accumulators_fn(&accumulator_4, &accumulator_5, &accumulator_6, &accumulator_7, depth,  \
                                               &result_vector);                                                        \
                        store_fn(&result_vector, c_row_ptr + tile_column_start_index + 4);                             \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    NK_PUBLIC void nk_##api_name##_packed_##input_type_name##_##isa_suffix(                                            \
        nk_##input_value_type##_t const *a_matrix, void const *b_packed_buffer, nk_##result_value_type##_t *c_matrix,  \
        nk_size_t row_count, nk_size_t column_count, nk_size_t depth, nk_size_t a_stride_in_bytes,                     \
        nk_size_t c_stride_in_bytes) {                                                                                 \
        /* Read padded depth from header for correct stride calculation */                                             \
        nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed_buffer;      \
        nk_size_t const depth_padded = header->depth_padded_values;                                                    \
                                                                                                                       \
        /* Cache blocking parameters (hardcoded for optimal L1/L2/L3 utilization) */                                   \
        nk_size_t const row_block_size = 128;      /* L2 cache blocking over rows */                                   \
        nk_size_t const column_block_size = 2048;  /* L3 cache blocking over columns */                                \
        nk_size_t const register_row_count = 4;    /* Rows per register tile */                                        \
        nk_size_t const register_column_count = 4; /* Columns per register tile */                                     \
        nk_unused_(register_column_count);         /* Suppress unused warnings */                                      \
        /* Use 1 × 8 kernel when columns are aligned to 8 and many columns relative to rows */                         \
        if (column_count % 8 == 0 && column_count >= row_count * 2 && depth % depth_simd_dimensions == 0) {            \
            nk_##api_name##_packed_##input_type_name##_##isa_suffix##_1x8_aligned_(                                    \
                a_matrix, b_packed_buffer, c_matrix, row_count, column_count, depth, a_stride_in_bytes,                \
                c_stride_in_bytes);                                                                                    \
            return;                                                                                                    \
        }                                                                                                              \
        /* Use 4 × 4 kernel when dimensions are 4-aligned */                                                           \
        if (row_count % 4 == 0 && column_count % 4 == 0 && depth % depth_simd_dimensions == 0) {                       \
            nk_##api_name##_packed_##input_type_name##_##isa_suffix##_aligned_(a_matrix, b_packed_buffer, c_matrix,    \
                                                                               row_count, column_count, depth,         \
                                                                               a_stride_in_bytes, c_stride_in_bytes);  \
            return;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        /* Zero output matrix */                                                                                       \
        for (nk_size_t row_index = 0; row_index < row_count; ++row_index) {                                            \
            nk_##result_value_type##_t *c_row = (nk_##result_value_type##_t *)((char *)c_matrix +                      \
                                                                               row_index * c_stride_in_bytes);         \
            for (nk_size_t column_index = 0; column_index < column_count; ++column_index) c_row[column_index] = 0;     \
        }                                                                                                              \
                                                                                                                       \
        /* Compute aligned/remainder depth for partial loads (correct for sub-byte types) */                           \
        nk_size_t const depth_dimensions_aligned = (depth / depth_simd_dimensions) * depth_simd_dimensions;            \
        nk_size_t const aligned_depth = nk_size_divide_round_up_(depth_dimensions_aligned, dimensions_per_value);      \
        nk_size_t const depth_in_values = nk_size_divide_round_up_(depth, dimensions_per_value);                       \
        nk_size_t const remainder_depth = depth_in_values - aligned_depth;                                             \
        nk_size_t const remainder_dimensions = depth - depth_dimensions_aligned;                                       \
        /* Calculate step size in storage values for loop increment */                                                 \
        nk_size_t const depth_step_values = nk_size_divide_round_up_(depth_simd_dimensions, dimensions_per_value);     \
                                                                                                                       \
        /* Loop 1: L3 cache blocking over columns */                                                                   \
        nk_##packed_value_type##_t const *packed_data =                                                                \
            (nk_##packed_value_type##_t const *)((char const *)b_packed_buffer +                                       \
                                                 sizeof(nk_cross_packed_buffer_header_t));                             \
        for (nk_size_t column_block_start_index = 0; column_block_start_index < column_count;                          \
             column_block_start_index += column_block_size) {                                                          \
            nk_size_t column_block_end_index = column_block_start_index + column_block_size;                           \
            if (column_block_end_index > column_count) column_block_end_index = column_count;                          \
                                                                                                                       \
            /* Loop 2: L2 cache blocking over rows */                                                                  \
            for (nk_size_t row_block_start_index = 0; row_block_start_index < row_count;                               \
                 row_block_start_index += row_block_size) {                                                            \
                nk_size_t row_block_end_index = row_block_start_index + row_block_size;                                \
                if (row_block_end_index > row_count) row_block_end_index = row_count;                                  \
                                                                                                                       \
                /* Loop 4: Register tiling over columns (register_column_count columns per batch) */                   \
                for (nk_size_t tile_column_start_index = column_block_start_index;                                     \
                     tile_column_start_index < column_block_end_index;                                                 \
                     tile_column_start_index += register_column_count) {                                               \
                    nk_size_t tile_column_count = register_column_count;                                               \
                    if (tile_column_start_index + tile_column_count > column_block_end_index)                          \
                        tile_column_count = column_block_end_index - tile_column_start_index;                          \
                                                                                                                       \
                    /* Compute B pointers once per column tile - direct column-major addressing */                     \
                    nk_##packed_value_type##_t const *b_depth_ptr_0 = packed_data +                                    \
                                                                      (tile_column_start_index + 0) * depth_padded;    \
                    nk_##packed_value_type##_t const *b_depth_ptr_1 =                                                  \
                        (tile_column_count > 1) ? packed_data + (tile_column_start_index + 1) * depth_padded           \
                                                : b_depth_ptr_0;                                                       \
                    nk_##packed_value_type##_t const *b_depth_ptr_2 =                                                  \
                        (tile_column_count > 2) ? packed_data + (tile_column_start_index + 2) * depth_padded           \
                                                : b_depth_ptr_0;                                                       \
                    nk_##packed_value_type##_t const *b_depth_ptr_3 =                                                  \
                        (tile_column_count > 3) ? packed_data + (tile_column_start_index + 3) * depth_padded           \
                                                : b_depth_ptr_0;                                                       \
                                                                                                                       \
                    /* Loop 5: Register tiling over rows (register_rows rows per tile) */                              \
                    for (nk_size_t tile_row_start_index = row_block_start_index;                                       \
                         tile_row_start_index < row_block_end_index; tile_row_start_index += register_row_count) {     \
                        nk_size_t tile_row_count = register_row_count;                                                 \
                        if (tile_row_start_index + tile_row_count > row_block_end_index)                               \
                            tile_row_count = row_block_end_index - tile_row_start_index;                               \
                                                                                                                       \
                        /* Initialize register_rows x register_cols accumulator states */                              \
                        state_type accumulator_tiles[4][4];                                                            \
                        for (nk_size_t r = 0; r < tile_row_count; ++r) {                                               \
                            init_accumulator_fn(&accumulator_tiles[r][0]);                                             \
                            init_accumulator_fn(&accumulator_tiles[r][1]);                                             \
                            init_accumulator_fn(&accumulator_tiles[r][2]);                                             \
                            init_accumulator_fn(&accumulator_tiles[r][3]);                                             \
                        }                                                                                              \
                                                                                                                       \
                        /* A row pointers */                                                                           \
                        nk_##input_value_type##_t const *a_row_ptr_0 =                                                 \
                            (nk_##input_value_type##_t const *)((char const *)a_matrix +                               \
                                                                (tile_row_start_index + 0) * a_stride_in_bytes);       \
                        nk_##input_value_type##_t const *a_row_ptr_1 =                                                 \
                            (tile_row_count > 1)                                                                       \
                                ? (nk_##input_value_type##_t const *)((char const *)a_matrix +                         \
                                                                      (tile_row_start_index + 1) * a_stride_in_bytes)  \
                                : a_row_ptr_0;                                                                         \
                        nk_##input_value_type##_t const *a_row_ptr_2 =                                                 \
                            (tile_row_count > 2)                                                                       \
                                ? (nk_##input_value_type##_t const *)((char const *)a_matrix +                         \
                                                                      (tile_row_start_index + 2) * a_stride_in_bytes)  \
                                : a_row_ptr_0;                                                                         \
                        nk_##input_value_type##_t const *a_row_ptr_3 =                                                 \
                            (tile_row_count > 3)                                                                       \
                                ? (nk_##input_value_type##_t const *)((char const *)a_matrix +                         \
                                                                      (tile_row_start_index + 3) * a_stride_in_bytes)  \
                                : a_row_ptr_0;                                                                         \
                                                                                                                       \
                        /* Tight inner loop: k values with simple ptr+k addressing */                                  \
                        vec_type a_first_vec, a_second_vec, a_third_vec, a_fourth_vec;                                 \
                        vec_type b_first_vec, b_second_vec, b_third_vec, b_fourth_vec;                                 \
                        for (nk_size_t k = 0; k < aligned_depth; k += depth_step_values) {                             \
                            /* Load next few values from 4 rows from A */                                              \
                            load_a_vec_fn(a_row_ptr_0 + k, &a_first_vec);                                              \
                            load_a_vec_fn(a_row_ptr_1 + k, &a_second_vec);                                             \
                            load_a_vec_fn(a_row_ptr_2 + k, &a_third_vec);                                              \
                            load_a_vec_fn(a_row_ptr_3 + k, &a_fourth_vec);                                             \
                                                                                                                       \
                            /* Load next few values from 4 rows from B */                                              \
                            load_b_vec_fn(b_depth_ptr_0 + k, &b_first_vec);                                            \
                            load_b_vec_fn(b_depth_ptr_1 + k, &b_second_vec);                                           \
                            load_b_vec_fn(b_depth_ptr_2 + k, &b_third_vec);                                            \
                            load_b_vec_fn(b_depth_ptr_3 + k, &b_fourth_vec);                                           \
                                                                                                                       \
                            /* 16 FMAs: 4 A rows × 4 B columns */                                                      \
                            inner_product_fn(&accumulator_tiles[0][0], a_first_vec, b_first_vec,                       \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[0][1], a_first_vec, b_second_vec,                      \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[0][2], a_first_vec, b_third_vec,                       \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[0][3], a_first_vec, b_fourth_vec,                      \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[1][0], a_second_vec, b_first_vec,                      \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[1][1], a_second_vec, b_second_vec,                     \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[1][2], a_second_vec, b_third_vec,                      \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[1][3], a_second_vec, b_fourth_vec,                     \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[2][0], a_third_vec, b_first_vec,                       \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[2][1], a_third_vec, b_second_vec,                      \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[2][2], a_third_vec, b_third_vec,                       \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[2][3], a_third_vec, b_fourth_vec,                      \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[3][0], a_fourth_vec, b_first_vec,                      \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[3][1], a_fourth_vec, b_second_vec,                     \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[3][2], a_fourth_vec, b_third_vec,                      \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                            inner_product_fn(&accumulator_tiles[3][3], a_fourth_vec, b_fourth_vec,                     \
                                             k * dimensions_per_value, depth_simd_dimensions);                         \
                        }                                                                                              \
                                                                                                                       \
                        /* Handle remainder k positions with partial loads */                                          \
                        if (remainder_depth > 0) {                                                                     \
                            /* Load next few values from 4 rows from A */                                              \
                            partial_load_a_vec_fn(a_row_ptr_0 + aligned_depth, &a_first_vec, remainder_dimensions);    \
                            partial_load_a_vec_fn(a_row_ptr_1 + aligned_depth, &a_second_vec, remainder_dimensions);   \
                            partial_load_a_vec_fn(a_row_ptr_2 + aligned_depth, &a_third_vec, remainder_dimensions);    \
                            partial_load_a_vec_fn(a_row_ptr_3 + aligned_depth, &a_fourth_vec, remainder_dimensions);   \
                                                                                                                       \
                            /* Load next few values from 4 rows from B */                                              \
                            partial_load_b_vec_fn(b_depth_ptr_0 + aligned_depth, &b_first_vec, remainder_dimensions);  \
                            partial_load_b_vec_fn(b_depth_ptr_1 + aligned_depth, &b_second_vec, remainder_dimensions); \
                            partial_load_b_vec_fn(b_depth_ptr_2 + aligned_depth, &b_third_vec, remainder_dimensions);  \
                            partial_load_b_vec_fn(b_depth_ptr_3 + aligned_depth, &b_fourth_vec, remainder_dimensions); \
                                                                                                                       \
                            /* 16 FMAs: 4 A rows × 4 B columns */                                                      \
                            inner_product_fn(&accumulator_tiles[0][0], a_first_vec, b_first_vec,                       \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[0][1], a_first_vec, b_second_vec,                      \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[0][2], a_first_vec, b_third_vec,                       \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[0][3], a_first_vec, b_fourth_vec,                      \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[1][0], a_second_vec, b_first_vec,                      \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[1][1], a_second_vec, b_second_vec,                     \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[1][2], a_second_vec, b_third_vec,                      \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[1][3], a_second_vec, b_fourth_vec,                     \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[2][0], a_third_vec, b_first_vec,                       \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[2][1], a_third_vec, b_second_vec,                      \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[2][2], a_third_vec, b_third_vec,                       \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[2][3], a_third_vec, b_fourth_vec,                      \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[3][0], a_fourth_vec, b_first_vec,                      \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[3][1], a_fourth_vec, b_second_vec,                     \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[3][2], a_fourth_vec, b_third_vec,                      \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                            inner_product_fn(&accumulator_tiles[3][3], a_fourth_vec, b_fourth_vec,                     \
                                             aligned_depth * dimensions_per_value, remainder_dimensions);              \
                        }                                                                                              \
                                                                                                                       \
                        /* Finalize and store register_rows x register_cols results using batched 4-way reduction */   \
                        for (nk_size_t r = 0; r < tile_row_count; ++r) {                                               \
                            result_vec_type result_vector;                                                             \
                            reduce_accumulators_fn(&accumulator_tiles[r][0], &accumulator_tiles[r][1],                 \
                                                   &accumulator_tiles[r][2], &accumulator_tiles[r][3], depth,          \
                                                   &result_vector);                                                    \
                                                                                                                       \
                            nk_##result_value_type##_t *c_row =                                                        \
                                (nk_##result_value_type##_t *)((char *)c_matrix +                                      \
                                                               (tile_row_start_index + r) * c_stride_in_bytes);        \
                            partial_store_fn(&result_vector, c_row + tile_column_start_index, tile_column_count);      \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }

/**
 *  @brief Generates compensated GEMM: C = A × Bᵀ with precomputed B column sums.
 *
 *  Like nk_define_cross_packed_ but the finalize function receives precomputed B column sums
 *  and per-row A sums to apply algebraic correction inline. This eliminates correction
 *  accumulators from the inner loop state, halving register pressure for integer dot products.
 *
 *  The compensated_finalize_fn signature differs from the standard reduce_accumulators_fn:
 *    compensated_finalize_fn(state_a, state_b, state_c, state_d, depth, a_sum, b_sums_vec, result)
 *  where a_sum is a scalar A row sum and b_sums_vec contains 4 B column sums as SIMD vector.
 *
 *  Buffer layout: [ Header ] [ Packed data ] [ Norms ] [ Column sums ]
 *  The norms occupy the same position as in non-compensated packs, so spatial functions work.
 */
#define nk_define_cross_compensated_packed_(                                                                           \
    api_name, input_type_name, isa_suffix, input_value_type, packed_value_type, result_value_type, sum_value_type,     \
    norm_value_type, vec_type, state_type, result_vec_type, init_accumulator_fn, load_a_vec_fn, partial_load_a_vec_fn, \
    load_b_vec_fn, partial_load_b_vec_fn, inner_product_fn, compensated_finalize_fn, store_fn, partial_store_fn,       \
    load_sum_fn, partial_load_sum_fn, compute_a_sum_fn, depth_simd_dimensions, dimensions_per_value)                   \
    NK_INTERNAL void nk_##api_name##_packed_##input_type_name##_##isa_suffix##_aligned_(                               \
        nk_##input_value_type##_t const *a_matrix, void const *b_packed_buffer, nk_##result_value_type##_t *c_matrix,  \
        nk_size_t row_count, nk_size_t column_count, nk_size_t depth, nk_size_t a_stride_in_bytes,                     \
        nk_size_t c_stride_in_bytes) {                                                                                 \
        nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed_buffer;      \
        nk_size_t const depth_padded = header->depth_padded_values;                                                    \
        nk_##packed_value_type##_t const *packed_data =                                                                \
            (nk_##packed_value_type##_t const *)((char const *)b_packed_buffer +                                       \
                                                 sizeof(nk_cross_packed_buffer_header_t));                             \
        /* Locate column sums: after packed data + norms */                                                            \
        nk_size_t const total_packed_values = column_count * depth_padded;                                             \
        nk_##norm_value_type##_t const *b_norms = (nk_##norm_value_type##_t const *)(packed_data +                     \
                                                                                     total_packed_values);             \
        nk_##sum_value_type##_t const *b_sums = (nk_##sum_value_type##_t const *)(b_norms + column_count);             \
        nk_unused_(b_norms);                                                                                           \
        nk_size_t const row_block_size = 128;                                                                          \
        nk_size_t const column_block_size = 2048;                                                                      \
        nk_size_t const register_row_count = 4;                                                                        \
        nk_size_t const register_column_count = 4;                                                                     \
        nk_size_t const depth_dimensions_aligned = (depth / depth_simd_dimensions) * depth_simd_dimensions;            \
        nk_size_t const aligned_depth = nk_size_divide_round_up_(depth_dimensions_aligned, dimensions_per_value);      \
        nk_size_t const depth_step_values = nk_size_divide_round_up_(depth_simd_dimensions, dimensions_per_value);     \
        for (nk_size_t row_index = 0; row_index < row_count; ++row_index) {                                            \
            nk_##result_value_type##_t *c_row = (nk_##result_value_type##_t *)((char *)c_matrix +                      \
                                                                               row_index * c_stride_in_bytes);         \
            for (nk_size_t ci = 0; ci < column_count; ++ci) c_row[ci] = 0;                                             \
        }                                                                                                              \
        for (nk_size_t cb = 0; cb < column_count; cb += column_block_size) {                                           \
            nk_size_t ce = cb + column_block_size;                                                                     \
            if (ce > column_count) ce = column_count;                                                                  \
            for (nk_size_t rb = 0; rb < row_count; rb += row_block_size) {                                             \
                nk_size_t re = rb + row_block_size;                                                                    \
                if (re > row_count) re = row_count;                                                                    \
                for (nk_size_t tc = cb; tc < ce; tc += register_column_count) {                                        \
                    nk_##packed_value_type##_t const *b_depth_ptr_0 = packed_data + (tc + 0) * depth_padded;           \
                    nk_##packed_value_type##_t const *b_depth_ptr_1 = packed_data + (tc + 1) * depth_padded;           \
                    nk_##packed_value_type##_t const *b_depth_ptr_2 = packed_data + (tc + 2) * depth_padded;           \
                    nk_##packed_value_type##_t const *b_depth_ptr_3 = packed_data + (tc + 3) * depth_padded;           \
                    /* Load 4 B column sums as SIMD vector */                                                          \
                    result_vec_type b_sum_vec;                                                                         \
                    load_sum_fn(b_sums + tc, &b_sum_vec);                                                              \
                    for (nk_size_t tr = rb; tr < re; tr += register_row_count) {                                       \
                        state_type acc[4][4];                                                                          \
                        init_accumulator_fn(&acc[0][0]), init_accumulator_fn(&acc[0][1]),                              \
                            init_accumulator_fn(&acc[0][2]), init_accumulator_fn(&acc[0][3]);                          \
                        init_accumulator_fn(&acc[1][0]), init_accumulator_fn(&acc[1][1]),                              \
                            init_accumulator_fn(&acc[1][2]), init_accumulator_fn(&acc[1][3]);                          \
                        init_accumulator_fn(&acc[2][0]), init_accumulator_fn(&acc[2][1]),                              \
                            init_accumulator_fn(&acc[2][2]), init_accumulator_fn(&acc[2][3]);                          \
                        init_accumulator_fn(&acc[3][0]), init_accumulator_fn(&acc[3][1]),                              \
                            init_accumulator_fn(&acc[3][2]), init_accumulator_fn(&acc[3][3]);                          \
                        nk_##input_value_type##_t const *a_row_ptr_0 =                                                 \
                            (nk_##input_value_type##_t const *)((char const *)a_matrix +                               \
                                                                (tr + 0) * a_stride_in_bytes);                         \
                        nk_##input_value_type##_t const *a_row_ptr_1 =                                                 \
                            (nk_##input_value_type##_t const *)((char const *)a_matrix +                               \
                                                                (tr + 1) * a_stride_in_bytes);                         \
                        nk_##input_value_type##_t const *a_row_ptr_2 =                                                 \
                            (nk_##input_value_type##_t const *)((char const *)a_matrix +                               \
                                                                (tr + 2) * a_stride_in_bytes);                         \
                        nk_##input_value_type##_t const *a_row_ptr_3 =                                                 \
                            (nk_##input_value_type##_t const *)((char const *)a_matrix +                               \
                                                                (tr + 3) * a_stride_in_bytes);                         \
                        /* Precompute A row sums (no-op for i8/u8, real for i4) */                                     \
                        nk_##sum_value_type##_t a_sums[4];                                                             \
                        a_sums[0] = compute_a_sum_fn(a_row_ptr_0, depth);                                              \
                        a_sums[1] = compute_a_sum_fn(a_row_ptr_1, depth);                                              \
                        a_sums[2] = compute_a_sum_fn(a_row_ptr_2, depth);                                              \
                        a_sums[3] = compute_a_sum_fn(a_row_ptr_3, depth);                                              \
                        vec_type av0, av1, av2, av3, bv0, bv1, bv2, bv3;                                               \
                        for (nk_size_t di = 0; di < aligned_depth; di += depth_step_values) {                          \
                            load_a_vec_fn(a_row_ptr_0 + di, &av0);                                                     \
                            load_a_vec_fn(a_row_ptr_1 + di, &av1);                                                     \
                            load_a_vec_fn(a_row_ptr_2 + di, &av2);                                                     \
                            load_a_vec_fn(a_row_ptr_3 + di, &av3);                                                     \
                            load_b_vec_fn(b_depth_ptr_0 + di, &bv0);                                                   \
                            load_b_vec_fn(b_depth_ptr_1 + di, &bv1);                                                   \
                            load_b_vec_fn(b_depth_ptr_2 + di, &bv2);                                                   \
                            load_b_vec_fn(b_depth_ptr_3 + di, &bv3);                                                   \
                            inner_product_fn(&acc[0][0], av0, bv0, di * dimensions_per_value, depth_simd_dimensions);  \
                            inner_product_fn(&acc[0][1], av0, bv1, di * dimensions_per_value, depth_simd_dimensions);  \
                            inner_product_fn(&acc[0][2], av0, bv2, di * dimensions_per_value, depth_simd_dimensions);  \
                            inner_product_fn(&acc[0][3], av0, bv3, di * dimensions_per_value, depth_simd_dimensions);  \
                            inner_product_fn(&acc[1][0], av1, bv0, di * dimensions_per_value, depth_simd_dimensions);  \
                            inner_product_fn(&acc[1][1], av1, bv1, di * dimensions_per_value, depth_simd_dimensions);  \
                            inner_product_fn(&acc[1][2], av1, bv2, di * dimensions_per_value, depth_simd_dimensions);  \
                            inner_product_fn(&acc[1][3], av1, bv3, di * dimensions_per_value, depth_simd_dimensions);  \
                            inner_product_fn(&acc[2][0], av2, bv0, di * dimensions_per_value, depth_simd_dimensions);  \
                            inner_product_fn(&acc[2][1], av2, bv1, di * dimensions_per_value, depth_simd_dimensions);  \
                            inner_product_fn(&acc[2][2], av2, bv2, di * dimensions_per_value, depth_simd_dimensions);  \
                            inner_product_fn(&acc[2][3], av2, bv3, di * dimensions_per_value, depth_simd_dimensions);  \
                            inner_product_fn(&acc[3][0], av3, bv0, di * dimensions_per_value, depth_simd_dimensions);  \
                            inner_product_fn(&acc[3][1], av3, bv1, di * dimensions_per_value, depth_simd_dimensions);  \
                            inner_product_fn(&acc[3][2], av3, bv2, di * dimensions_per_value, depth_simd_dimensions);  \
                            inner_product_fn(&acc[3][3], av3, bv3, di * dimensions_per_value, depth_simd_dimensions);  \
                        }                                                                                              \
                        /* Compensated finalize: apply correction with precomputed sums */                             \
                        result_vec_type result_vector;                                                                 \
                        for (nk_size_t r = 0; r < register_row_count; ++r) {                                           \
                            compensated_finalize_fn(&acc[r][0], &acc[r][1], &acc[r][2], &acc[r][3], depth, a_sums[r],  \
                                                    &b_sum_vec, &result_vector);                                       \
                            nk_##result_value_type##_t *c_row =                                                        \
                                (nk_##result_value_type##_t *)((char *)c_matrix + (tr + r) * c_stride_in_bytes);       \
                            store_fn(&result_vector, c_row + tc);                                                      \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    NK_INTERNAL void nk_##api_name##_packed_##input_type_name##_##isa_suffix##_1x8_aligned_(                           \
        nk_##input_value_type##_t const *a_matrix, void const *b_packed_buffer, nk_##result_value_type##_t *c_matrix,  \
        nk_size_t row_count, nk_size_t column_count, nk_size_t depth, nk_size_t a_stride_in_bytes,                     \
        nk_size_t c_stride_in_bytes) {                                                                                 \
        nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed_buffer;      \
        nk_size_t const depth_padded = header->depth_padded_values;                                                    \
        nk_##packed_value_type##_t const *packed_data =                                                                \
            (nk_##packed_value_type##_t const *)((char const *)b_packed_buffer +                                       \
                                                 sizeof(nk_cross_packed_buffer_header_t));                             \
        nk_size_t const total_packed_values = column_count * depth_padded;                                             \
        nk_##norm_value_type##_t const *b_norms = (nk_##norm_value_type##_t const *)(packed_data +                     \
                                                                                     total_packed_values);             \
        nk_##sum_value_type##_t const *b_sums = (nk_##sum_value_type##_t const *)(b_norms + column_count);             \
        nk_unused_(b_norms);                                                                                           \
        nk_size_t const row_block_size = 128;                                                                          \
        nk_size_t const column_block_size = 2048;                                                                      \
        nk_size_t const register_column_count = 8;                                                                     \
        nk_size_t const depth_dimensions_aligned = (depth / depth_simd_dimensions) * depth_simd_dimensions;            \
        nk_size_t const aligned_depth = nk_size_divide_round_up_(depth_dimensions_aligned, dimensions_per_value);      \
        nk_size_t const depth_step_values = nk_size_divide_round_up_(depth_simd_dimensions, dimensions_per_value);     \
        for (nk_size_t row_index = 0; row_index < row_count; ++row_index) {                                            \
            nk_##result_value_type##_t *c_row = (nk_##result_value_type##_t *)((char *)c_matrix +                      \
                                                                               row_index * c_stride_in_bytes);         \
            for (nk_size_t ci = 0; ci < column_count; ++ci) c_row[ci] = 0;                                             \
        }                                                                                                              \
        for (nk_size_t cb = 0; cb < column_count; cb += column_block_size) {                                           \
            nk_size_t ce = cb + column_block_size;                                                                     \
            if (ce > column_count) ce = column_count;                                                                  \
            for (nk_size_t rb2 = 0; rb2 < row_count; rb2 += row_block_size) {                                          \
                nk_size_t re2 = rb2 + row_block_size < row_count ? rb2 + row_block_size : row_count;                   \
                for (nk_size_t tc = cb; tc < ce; tc += register_column_count) {                                        \
                    nk_##packed_value_type##_t const *bp0 = packed_data + (tc + 0) * depth_padded;                     \
                    nk_##packed_value_type##_t const *bp1 = packed_data + (tc + 1) * depth_padded;                     \
                    nk_##packed_value_type##_t const *bp2 = packed_data + (tc + 2) * depth_padded;                     \
                    nk_##packed_value_type##_t const *bp3 = packed_data + (tc + 3) * depth_padded;                     \
                    nk_##packed_value_type##_t const *bp4 = packed_data + (tc + 4) * depth_padded;                     \
                    nk_##packed_value_type##_t const *bp5 = packed_data + (tc + 5) * depth_padded;                     \
                    nk_##packed_value_type##_t const *bp6 = packed_data + (tc + 6) * depth_padded;                     \
                    nk_##packed_value_type##_t const *bp7 = packed_data + (tc + 7) * depth_padded;                     \
                    result_vec_type b_sum_low, b_sum_high;                                                             \
                    load_sum_fn(b_sums + tc, &b_sum_low);                                                              \
                    load_sum_fn(b_sums + tc + 4, &b_sum_high);                                                         \
                    for (nk_size_t ri = rb2; ri < re2; ++ri) {                                                         \
                        state_type s0, s1, s2, s3, s4, s5, s6, s7;                                                     \
                        init_accumulator_fn(&s0), init_accumulator_fn(&s1), init_accumulator_fn(&s2),                  \
                            init_accumulator_fn(&s3), init_accumulator_fn(&s4), init_accumulator_fn(&s5),              \
                            init_accumulator_fn(&s6), init_accumulator_fn(&s7);                                        \
                        nk_##input_value_type##_t const *a_row =                                                       \
                            (nk_##input_value_type##_t const *)((char const *)a_matrix + ri * a_stride_in_bytes);      \
                        nk_##sum_value_type##_t a_sum_val = compute_a_sum_fn(a_row, depth);                            \
                        vec_type av;                                                                                   \
                        vec_type bv0, bv1, bv2, bv3, bv4, bv5, bv6, bv7;                                               \
                        for (nk_size_t di = 0; di < aligned_depth; di += depth_step_values) {                          \
                            load_a_vec_fn(a_row + di, &av);                                                            \
                            load_b_vec_fn(bp0 + di, &bv0), load_b_vec_fn(bp1 + di, &bv1);                              \
                            load_b_vec_fn(bp2 + di, &bv2), load_b_vec_fn(bp3 + di, &bv3);                              \
                            load_b_vec_fn(bp4 + di, &bv4), load_b_vec_fn(bp5 + di, &bv5);                              \
                            load_b_vec_fn(bp6 + di, &bv6), load_b_vec_fn(bp7 + di, &bv7);                              \
                            inner_product_fn(&s0, av, bv0, di * dimensions_per_value, depth_simd_dimensions);          \
                            inner_product_fn(&s1, av, bv1, di * dimensions_per_value, depth_simd_dimensions);          \
                            inner_product_fn(&s2, av, bv2, di * dimensions_per_value, depth_simd_dimensions);          \
                            inner_product_fn(&s3, av, bv3, di * dimensions_per_value, depth_simd_dimensions);          \
                            inner_product_fn(&s4, av, bv4, di * dimensions_per_value, depth_simd_dimensions);          \
                            inner_product_fn(&s5, av, bv5, di * dimensions_per_value, depth_simd_dimensions);          \
                            inner_product_fn(&s6, av, bv6, di * dimensions_per_value, depth_simd_dimensions);          \
                            inner_product_fn(&s7, av, bv7, di * dimensions_per_value, depth_simd_dimensions);          \
                        }                                                                                              \
                        result_vec_type rv;                                                                            \
                        nk_##result_value_type##_t *c_row = (nk_##result_value_type##_t *)((char *)c_matrix +          \
                                                                                           ri * c_stride_in_bytes);    \
                        compensated_finalize_fn(&s0, &s1, &s2, &s3, depth, a_sum_val, &b_sum_low, &rv);                \
                        store_fn(&rv, c_row + tc);                                                                     \
                        compensated_finalize_fn(&s4, &s5, &s6, &s7, depth, a_sum_val, &b_sum_high, &rv);               \
                        store_fn(&rv, c_row + tc + 4);                                                                 \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    NK_PUBLIC void nk_##api_name##_packed_##input_type_name##_##isa_suffix(                                            \
        nk_##input_value_type##_t const *a_matrix, void const *b_packed_buffer, nk_##result_value_type##_t *c_matrix,  \
        nk_size_t row_count, nk_size_t column_count, nk_size_t depth, nk_size_t a_stride_in_bytes,                     \
        nk_size_t c_stride_in_bytes) {                                                                                 \
        nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed_buffer;      \
        nk_size_t const depth_padded = header->depth_padded_values;                                                    \
        nk_size_t const row_block_size = 128;                                                                          \
        nk_size_t const column_block_size = 2048;                                                                      \
        nk_size_t const register_row_count = 4;                                                                        \
        nk_size_t const register_column_count = 4;                                                                     \
        nk_unused_(register_column_count);                                                                             \
        if (column_count % 8 == 0 && column_count >= row_count * 2 && depth % depth_simd_dimensions == 0) {            \
            nk_##api_name##_packed_##input_type_name##_##isa_suffix##_1x8_aligned_(                                    \
                a_matrix, b_packed_buffer, c_matrix, row_count, column_count, depth, a_stride_in_bytes,                \
                c_stride_in_bytes);                                                                                    \
            return;                                                                                                    \
        }                                                                                                              \
        if (row_count % 4 == 0 && column_count % 4 == 0 && depth % depth_simd_dimensions == 0) {                       \
            nk_##api_name##_packed_##input_type_name##_##isa_suffix##_aligned_(a_matrix, b_packed_buffer, c_matrix,    \
                                                                               row_count, column_count, depth,         \
                                                                               a_stride_in_bytes, c_stride_in_bytes);  \
            return;                                                                                                    \
        }                                                                                                              \
        /* Generic fallback with partial loads and compensated finalize */                                             \
        nk_##packed_value_type##_t const *packed_data =                                                                \
            (nk_##packed_value_type##_t const *)((char const *)b_packed_buffer +                                       \
                                                 sizeof(nk_cross_packed_buffer_header_t));                             \
        nk_size_t const total_packed_values = column_count * depth_padded;                                             \
        nk_##norm_value_type##_t const *b_norms = (nk_##norm_value_type##_t const *)(packed_data +                     \
                                                                                     total_packed_values);             \
        nk_##sum_value_type##_t const *b_sums = (nk_##sum_value_type##_t const *)(b_norms + column_count);             \
        nk_unused_(b_norms);                                                                                           \
        nk_size_t const depth_dimensions_aligned = (depth / depth_simd_dimensions) * depth_simd_dimensions;            \
        nk_size_t const aligned_depth = nk_size_divide_round_up_(depth_dimensions_aligned, dimensions_per_value);      \
        nk_size_t const depth_in_values = nk_size_divide_round_up_(depth, dimensions_per_value);                       \
        nk_size_t const remainder_depth = depth_in_values - aligned_depth;                                             \
        nk_size_t const remainder_dimensions = depth - depth_dimensions_aligned;                                       \
        nk_size_t const depth_step_values = nk_size_divide_round_up_(depth_simd_dimensions, dimensions_per_value);     \
        for (nk_size_t row_index = 0; row_index < row_count; ++row_index) {                                            \
            nk_##result_value_type##_t *c_row = (nk_##result_value_type##_t *)((char *)c_matrix +                      \
                                                                               row_index * c_stride_in_bytes);         \
            for (nk_size_t ci = 0; ci < column_count; ++ci) c_row[ci] = 0;                                             \
        }                                                                                                              \
        for (nk_size_t cb = 0; cb < column_count; cb += column_block_size) {                                           \
            nk_size_t ce = cb + column_block_size;                                                                     \
            if (ce > column_count) ce = column_count;                                                                  \
            for (nk_size_t rb = 0; rb < row_count; rb += row_block_size) {                                             \
                nk_size_t re = rb + row_block_size;                                                                    \
                if (re > row_count) re = row_count;                                                                    \
                for (nk_size_t tc = cb; tc < ce; tc += register_column_count) {                                        \
                    nk_size_t tile_col_count = register_column_count;                                                  \
                    if (tc + tile_col_count > ce) tile_col_count = ce - tc;                                            \
                    nk_##packed_value_type##_t const *bdp0 = packed_data + (tc + 0) * depth_padded;                    \
                    nk_##packed_value_type##_t const *bdp1 = (tile_col_count > 1)                                      \
                                                                 ? packed_data + (tc + 1) * depth_padded               \
                                                                 : bdp0;                                               \
                    nk_##packed_value_type##_t const *bdp2 = (tile_col_count > 2)                                      \
                                                                 ? packed_data + (tc + 2) * depth_padded               \
                                                                 : bdp0;                                               \
                    nk_##packed_value_type##_t const *bdp3 = (tile_col_count > 3)                                      \
                                                                 ? packed_data + (tc + 3) * depth_padded               \
                                                                 : bdp0;                                               \
                    result_vec_type b_sum_vec;                                                                         \
                    partial_load_sum_fn(b_sums + tc, &b_sum_vec, tile_col_count);                                      \
                    for (nk_size_t tr = rb; tr < re; tr += register_row_count) {                                       \
                        nk_size_t tile_row_count = register_row_count;                                                 \
                        if (tr + tile_row_count > re) tile_row_count = re - tr;                                        \
                        state_type acc[4][4];                                                                          \
                        for (nk_size_t rr = 0; rr < tile_row_count; ++rr) {                                            \
                            init_accumulator_fn(&acc[rr][0]);                                                          \
                            init_accumulator_fn(&acc[rr][1]);                                                          \
                            init_accumulator_fn(&acc[rr][2]);                                                          \
                            init_accumulator_fn(&acc[rr][3]);                                                          \
                        }                                                                                              \
                        nk_##input_value_type##_t const *arp0 =                                                        \
                            (nk_##input_value_type##_t const *)((char const *)a_matrix +                               \
                                                                (tr + 0) * a_stride_in_bytes);                         \
                        nk_##input_value_type##_t const *arp1 =                                                        \
                            (tile_row_count > 1) ? (nk_##input_value_type##_t const *)((char const *)a_matrix +        \
                                                                                       (tr + 1) * a_stride_in_bytes)   \
                                                 : arp0;                                                               \
                        nk_##input_value_type##_t const *arp2 =                                                        \
                            (tile_row_count > 2) ? (nk_##input_value_type##_t const *)((char const *)a_matrix +        \
                                                                                       (tr + 2) * a_stride_in_bytes)   \
                                                 : arp0;                                                               \
                        nk_##input_value_type##_t const *arp3 =                                                        \
                            (tile_row_count > 3) ? (nk_##input_value_type##_t const *)((char const *)a_matrix +        \
                                                                                       (tr + 3) * a_stride_in_bytes)   \
                                                 : arp0;                                                               \
                        nk_##sum_value_type##_t a_sums[4];                                                             \
                        a_sums[0] = compute_a_sum_fn(arp0, depth);                                                     \
                        a_sums[1] = (tile_row_count > 1) ? compute_a_sum_fn(arp1, depth) : 0;                          \
                        a_sums[2] = (tile_row_count > 2) ? compute_a_sum_fn(arp2, depth) : 0;                          \
                        a_sums[3] = (tile_row_count > 3) ? compute_a_sum_fn(arp3, depth) : 0;                          \
                        vec_type av0, av1, av2, av3, bv0, bv1, bv2, bv3;                                               \
                        for (nk_size_t k = 0; k < aligned_depth; k += depth_step_values) {                             \
                            load_a_vec_fn(arp0 + k, &av0);                                                             \
                            load_a_vec_fn(arp1 + k, &av1);                                                             \
                            load_a_vec_fn(arp2 + k, &av2);                                                             \
                            load_a_vec_fn(arp3 + k, &av3);                                                             \
                            load_b_vec_fn(bdp0 + k, &bv0);                                                             \
                            load_b_vec_fn(bdp1 + k, &bv1);                                                             \
                            load_b_vec_fn(bdp2 + k, &bv2);                                                             \
                            load_b_vec_fn(bdp3 + k, &bv3);                                                             \
                            inner_product_fn(&acc[0][0], av0, bv0, k * dimensions_per_value, depth_simd_dimensions);   \
                            inner_product_fn(&acc[0][1], av0, bv1, k * dimensions_per_value, depth_simd_dimensions);   \
                            inner_product_fn(&acc[0][2], av0, bv2, k * dimensions_per_value, depth_simd_dimensions);   \
                            inner_product_fn(&acc[0][3], av0, bv3, k * dimensions_per_value, depth_simd_dimensions);   \
                            inner_product_fn(&acc[1][0], av1, bv0, k * dimensions_per_value, depth_simd_dimensions);   \
                            inner_product_fn(&acc[1][1], av1, bv1, k * dimensions_per_value, depth_simd_dimensions);   \
                            inner_product_fn(&acc[1][2], av1, bv2, k * dimensions_per_value, depth_simd_dimensions);   \
                            inner_product_fn(&acc[1][3], av1, bv3, k * dimensions_per_value, depth_simd_dimensions);   \
                            inner_product_fn(&acc[2][0], av2, bv0, k * dimensions_per_value, depth_simd_dimensions);   \
                            inner_product_fn(&acc[2][1], av2, bv1, k * dimensions_per_value, depth_simd_dimensions);   \
                            inner_product_fn(&acc[2][2], av2, bv2, k * dimensions_per_value, depth_simd_dimensions);   \
                            inner_product_fn(&acc[2][3], av2, bv3, k * dimensions_per_value, depth_simd_dimensions);   \
                            inner_product_fn(&acc[3][0], av3, bv0, k * dimensions_per_value, depth_simd_dimensions);   \
                            inner_product_fn(&acc[3][1], av3, bv1, k * dimensions_per_value, depth_simd_dimensions);   \
                            inner_product_fn(&acc[3][2], av3, bv2, k * dimensions_per_value, depth_simd_dimensions);   \
                            inner_product_fn(&acc[3][3], av3, bv3, k * dimensions_per_value, depth_simd_dimensions);   \
                        }                                                                                              \
                        if (remainder_depth > 0) {                                                                     \
                            partial_load_a_vec_fn(arp0 + aligned_depth, &av0, remainder_dimensions);                   \
                            partial_load_a_vec_fn(arp1 + aligned_depth, &av1, remainder_dimensions);                   \
                            partial_load_a_vec_fn(arp2 + aligned_depth, &av2, remainder_dimensions);                   \
                            partial_load_a_vec_fn(arp3 + aligned_depth, &av3, remainder_dimensions);                   \
                            partial_load_b_vec_fn(bdp0 + aligned_depth, &bv0, remainder_dimensions);                   \
                            partial_load_b_vec_fn(bdp1 + aligned_depth, &bv1, remainder_dimensions);                   \
                            partial_load_b_vec_fn(bdp2 + aligned_depth, &bv2, remainder_dimensions);                   \
                            partial_load_b_vec_fn(bdp3 + aligned_depth, &bv3, remainder_dimensions);                   \
                            inner_product_fn(&acc[0][0], av0, bv0, aligned_depth * dimensions_per_value,               \
                                             remainder_dimensions);                                                    \
                            inner_product_fn(&acc[0][1], av0, bv1, aligned_depth * dimensions_per_value,               \
                                             remainder_dimensions);                                                    \
                            inner_product_fn(&acc[0][2], av0, bv2, aligned_depth * dimensions_per_value,               \
                                             remainder_dimensions);                                                    \
                            inner_product_fn(&acc[0][3], av0, bv3, aligned_depth * dimensions_per_value,               \
                                             remainder_dimensions);                                                    \
                            inner_product_fn(&acc[1][0], av1, bv0, aligned_depth * dimensions_per_value,               \
                                             remainder_dimensions);                                                    \
                            inner_product_fn(&acc[1][1], av1, bv1, aligned_depth * dimensions_per_value,               \
                                             remainder_dimensions);                                                    \
                            inner_product_fn(&acc[1][2], av1, bv2, aligned_depth * dimensions_per_value,               \
                                             remainder_dimensions);                                                    \
                            inner_product_fn(&acc[1][3], av1, bv3, aligned_depth * dimensions_per_value,               \
                                             remainder_dimensions);                                                    \
                            inner_product_fn(&acc[2][0], av2, bv0, aligned_depth * dimensions_per_value,               \
                                             remainder_dimensions);                                                    \
                            inner_product_fn(&acc[2][1], av2, bv1, aligned_depth * dimensions_per_value,               \
                                             remainder_dimensions);                                                    \
                            inner_product_fn(&acc[2][2], av2, bv2, aligned_depth * dimensions_per_value,               \
                                             remainder_dimensions);                                                    \
                            inner_product_fn(&acc[2][3], av2, bv3, aligned_depth * dimensions_per_value,               \
                                             remainder_dimensions);                                                    \
                            inner_product_fn(&acc[3][0], av3, bv0, aligned_depth * dimensions_per_value,               \
                                             remainder_dimensions);                                                    \
                            inner_product_fn(&acc[3][1], av3, bv1, aligned_depth * dimensions_per_value,               \
                                             remainder_dimensions);                                                    \
                            inner_product_fn(&acc[3][2], av3, bv2, aligned_depth * dimensions_per_value,               \
                                             remainder_dimensions);                                                    \
                            inner_product_fn(&acc[3][3], av3, bv3, aligned_depth * dimensions_per_value,               \
                                             remainder_dimensions);                                                    \
                        }                                                                                              \
                        for (nk_size_t rr = 0; rr < tile_row_count; ++rr) {                                            \
                            result_vec_type rv;                                                                        \
                            compensated_finalize_fn(&acc[rr][0], &acc[rr][1], &acc[rr][2], &acc[rr][3], depth,         \
                                                    a_sums[rr], &b_sum_vec, &rv);                                      \
                            nk_##result_value_type##_t *c_row =                                                        \
                                (nk_##result_value_type##_t *)((char *)c_matrix + (tr + rr) * c_stride_in_bytes);      \
                            partial_store_fn(&rv, c_row + tc, tile_col_count);                                         \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }

/**
 *  @brief Generates compensated symmetric Gram matrix: C = A × Aᵀ with inline correction.
 *
 *  Like nk_define_cross_symmetric_ but the finalize function receives precomputed sums.
 *  For symmetric computation, both row and column vectors come from the same matrix A,
 *  so A sums serve as both row and column sums.
 *
 *  The off-diagonal helper uses 4×4 tiling (matching nk_define_cross_symmetric_) with
 *  progressive sum accumulation: SAD runs on port 5 alongside DPBUSD on ports 0+1 for
 *  zero throughput overhead on Alder Lake and Ice Lake.
 */
#define nk_define_cross_compensated_symmetric_(                                                                        \
    api_name, input_type_name, isa_suffix, input_value_type, result_value_type, sum_value_type, norm_value_type,       \
    vec_type, state_type, result_vec_type, init_accumulator_fn, load_vec_fn, partial_load_vec_fn, inner_product_fn,    \
    compensated_finalize_fn, store_fn, partial_store_fn, load_sum_fn, partial_load_sum_fn, sum_state_type,             \
    init_sum_fn, update_sum_fn, finalize_sum_fn, depth_simd_dimensions, dimensions_per_value)                          \
    NK_INTERNAL void nk_##api_name##_symmetric_diagonal_##input_type_name##_##isa_suffix##_(                           \
        nk_##input_value_type##_t const **vector_base_ptrs, nk_size_t i_macro, nk_size_t macro_size,                   \
        nk_size_t aligned_depth, nk_size_t remainder_depth, nk_size_t remainder_dimensions,                            \
        nk_size_t depth_step_values, nk_size_t dimensions_per_value_runtime, nk_##result_value_type##_t *result,       \
        nk_size_t result_stride_values, nk_size_t finalizer_batch_size, nk_size_t depth) {                             \
        nk_unused_(finalizer_batch_size);                                                                              \
        nk_unused_(dimensions_per_value_runtime);                                                                      \
        /* Compute sums via stateful helpers — separate loop is fine since diagonal is ~1.6% of work */                \
        nk_size_t padded_depth_dimensions = aligned_depth * dimensions_per_value +                                     \
                                            (remainder_depth > 0 ? depth_simd_dimensions : 0);                         \
        nk_##sum_value_type##_t precomputed_sums[32];                                                                  \
        for (nk_size_t s = 0; s < macro_size; s++) {                                                                   \
            sum_state_type ss;                                                                                         \
            init_sum_fn(&ss);                                                                                          \
            for (nk_size_t di = 0; di < aligned_depth; di += depth_step_values) {                                      \
                vec_type v;                                                                                            \
                load_vec_fn(vector_base_ptrs[s] + di, &v);                                                             \
                update_sum_fn(&ss, v);                                                                                 \
            }                                                                                                          \
            if (remainder_depth > 0) {                                                                                 \
                vec_type v;                                                                                            \
                partial_load_vec_fn(vector_base_ptrs[s] + aligned_depth, &v, remainder_dimensions);                    \
                update_sum_fn(&ss, v);                                                                                 \
            }                                                                                                          \
            precomputed_sums[s] = finalize_sum_fn(&ss, padded_depth_dimensions);                                       \
        }                                                                                                              \
        for (nk_size_t tile_row_start = 0; tile_row_start < macro_size; tile_row_start += 4) {                         \
            for (nk_size_t tile_col_start = tile_row_start; tile_col_start < macro_size; tile_col_start += 4) {        \
                nk_size_t tile_rows = (tile_row_start + 4 <= macro_size) ? 4 : (macro_size - tile_row_start);          \
                nk_size_t tile_cols = (tile_col_start + 4 <= macro_size) ? 4 : (macro_size - tile_col_start);          \
                int is_diag = (tile_row_start == tile_col_start);                                                      \
                NK_ALIGN64 state_type accumulators[4][7];                                                              \
                for (nk_size_t row = 0; row < tile_rows; row++) {                                                      \
                    nk_size_t init_start = is_diag ? row : 0;                                                          \
                    nk_size_t init_end = is_diag ? (row + 4) : tile_cols;                                              \
                    for (nk_size_t col = init_start; col < init_end; col++) {                                          \
                        init_accumulator_fn(&accumulators[row][col]);                                                  \
                    }                                                                                                  \
                }                                                                                                      \
                nk_##input_value_type##_t const *row_ptrs[4], *col_ptrs[4];                                            \
                row_ptrs[0] = vector_base_ptrs[tile_row_start + 0];                                                    \
                row_ptrs[1] = (tile_rows > 1) ? vector_base_ptrs[tile_row_start + 1] : row_ptrs[0];                    \
                row_ptrs[2] = (tile_rows > 2) ? vector_base_ptrs[tile_row_start + 2] : row_ptrs[0];                    \
                row_ptrs[3] = (tile_rows > 3) ? vector_base_ptrs[tile_row_start + 3] : row_ptrs[0];                    \
                if (is_diag) {                                                                                         \
                    col_ptrs[0] = row_ptrs[0];                                                                         \
                    col_ptrs[1] = row_ptrs[1];                                                                         \
                    col_ptrs[2] = row_ptrs[2];                                                                         \
                    col_ptrs[3] = row_ptrs[3];                                                                         \
                }                                                                                                      \
                else {                                                                                                 \
                    col_ptrs[0] = vector_base_ptrs[tile_col_start + 0];                                                \
                    col_ptrs[1] = (tile_cols > 1) ? vector_base_ptrs[tile_col_start + 1] : col_ptrs[0];                \
                    col_ptrs[2] = (tile_cols > 2) ? vector_base_ptrs[tile_col_start + 2] : col_ptrs[0];                \
                    col_ptrs[3] = (tile_cols > 3) ? vector_base_ptrs[tile_col_start + 3] : col_ptrs[0];                \
                }                                                                                                      \
                vec_type row_vecs[4], col_vecs[4];                                                                     \
                for (nk_size_t di = 0; di < aligned_depth; di += depth_step_values) {                                  \
                    load_vec_fn(row_ptrs[0] + di, &row_vecs[0]);                                                       \
                    load_vec_fn(row_ptrs[1] + di, &row_vecs[1]);                                                       \
                    load_vec_fn(row_ptrs[2] + di, &row_vecs[2]);                                                       \
                    load_vec_fn(row_ptrs[3] + di, &row_vecs[3]);                                                       \
                    if (!is_diag) {                                                                                    \
                        load_vec_fn(col_ptrs[0] + di, &col_vecs[0]);                                                   \
                        load_vec_fn(col_ptrs[1] + di, &col_vecs[1]);                                                   \
                        load_vec_fn(col_ptrs[2] + di, &col_vecs[2]);                                                   \
                        load_vec_fn(col_ptrs[3] + di, &col_vecs[3]);                                                   \
                    }                                                                                                  \
                    else {                                                                                             \
                        col_vecs[0] = row_vecs[0];                                                                     \
                        col_vecs[1] = row_vecs[1];                                                                     \
                        col_vecs[2] = row_vecs[2];                                                                     \
                        col_vecs[3] = row_vecs[3];                                                                     \
                    }                                                                                                  \
                    if (tile_rows == 4 && tile_cols == 4 && is_diag) {                                                 \
                        /* Upper triangle: 10 FMAs */                                                                  \
                        inner_product_fn(&accumulators[0][0], row_vecs[0], col_vecs[0], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[0][1], row_vecs[0], col_vecs[1], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[0][2], row_vecs[0], col_vecs[2], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[0][3], row_vecs[0], col_vecs[3], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[1][1], row_vecs[1], col_vecs[1], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[1][2], row_vecs[1], col_vecs[2], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[1][3], row_vecs[1], col_vecs[3], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[2][2], row_vecs[2], col_vecs[2], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[2][3], row_vecs[2], col_vecs[3], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[3][3], row_vecs[3], col_vecs[3], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                    }                                                                                                  \
                    else if (tile_rows == 4 && tile_cols == 4) {                                                       \
                        /* Full 4×4 rectangle: 16 FMAs */                                                              \
                        inner_product_fn(&accumulators[0][0], row_vecs[0], col_vecs[0], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[0][1], row_vecs[0], col_vecs[1], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[0][2], row_vecs[0], col_vecs[2], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[0][3], row_vecs[0], col_vecs[3], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[1][0], row_vecs[1], col_vecs[0], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[1][1], row_vecs[1], col_vecs[1], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[1][2], row_vecs[1], col_vecs[2], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[1][3], row_vecs[1], col_vecs[3], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[2][0], row_vecs[2], col_vecs[0], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[2][1], row_vecs[2], col_vecs[1], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[2][2], row_vecs[2], col_vecs[2], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[2][3], row_vecs[2], col_vecs[3], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[3][0], row_vecs[3], col_vecs[0], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[3][1], row_vecs[3], col_vecs[1], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[3][2], row_vecs[3], col_vecs[2], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[3][3], row_vecs[3], col_vecs[3], di * dimensions_per_value,     \
                                         depth_simd_dimensions);                                                       \
                    }                                                                                                  \
                    else {                                                                                             \
                        for (nk_size_t row = 0; row < tile_rows; row++) {                                              \
                            nk_size_t col_start = is_diag ? row : 0;                                                   \
                            nk_size_t col_end = is_diag ? (row < 4 ? 4 : tile_cols) : tile_cols;                       \
                            for (nk_size_t col = col_start; col < col_end; col++)                                      \
                                inner_product_fn(&accumulators[row][col], row_vecs[row], col_vecs[col],                \
                                                 di * dimensions_per_value, depth_simd_dimensions);                    \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
                if (remainder_depth > 0) {                                                                             \
                    partial_load_vec_fn(row_ptrs[0] + aligned_depth, &row_vecs[0], remainder_dimensions);              \
                    partial_load_vec_fn(row_ptrs[1] + aligned_depth, &row_vecs[1], remainder_dimensions);              \
                    partial_load_vec_fn(row_ptrs[2] + aligned_depth, &row_vecs[2], remainder_dimensions);              \
                    partial_load_vec_fn(row_ptrs[3] + aligned_depth, &row_vecs[3], remainder_dimensions);              \
                    if (!is_diag) {                                                                                    \
                        partial_load_vec_fn(col_ptrs[0] + aligned_depth, &col_vecs[0], remainder_dimensions);          \
                        partial_load_vec_fn(col_ptrs[1] + aligned_depth, &col_vecs[1], remainder_dimensions);          \
                        partial_load_vec_fn(col_ptrs[2] + aligned_depth, &col_vecs[2], remainder_dimensions);          \
                        partial_load_vec_fn(col_ptrs[3] + aligned_depth, &col_vecs[3], remainder_dimensions);          \
                    }                                                                                                  \
                    else {                                                                                             \
                        col_vecs[0] = row_vecs[0];                                                                     \
                        col_vecs[1] = row_vecs[1];                                                                     \
                        col_vecs[2] = row_vecs[2];                                                                     \
                        col_vecs[3] = row_vecs[3];                                                                     \
                    }                                                                                                  \
                    if (tile_rows == 4 && tile_cols == 4 && is_diag) {                                                 \
                        inner_product_fn(&accumulators[0][0], row_vecs[0], col_vecs[0],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[0][1], row_vecs[0], col_vecs[1],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[0][2], row_vecs[0], col_vecs[2],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[0][3], row_vecs[0], col_vecs[3],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[1][1], row_vecs[1], col_vecs[1],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[1][2], row_vecs[1], col_vecs[2],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[1][3], row_vecs[1], col_vecs[3],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[2][2], row_vecs[2], col_vecs[2],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[2][3], row_vecs[2], col_vecs[3],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[3][3], row_vecs[3], col_vecs[3],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                    }                                                                                                  \
                    else if (tile_rows == 4 && tile_cols == 4) {                                                       \
                        inner_product_fn(&accumulators[0][0], row_vecs[0], col_vecs[0],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[0][1], row_vecs[0], col_vecs[1],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[0][2], row_vecs[0], col_vecs[2],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[0][3], row_vecs[0], col_vecs[3],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[1][0], row_vecs[1], col_vecs[0],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[1][1], row_vecs[1], col_vecs[1],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[1][2], row_vecs[1], col_vecs[2],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[1][3], row_vecs[1], col_vecs[3],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[2][0], row_vecs[2], col_vecs[0],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[2][1], row_vecs[2], col_vecs[1],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[2][2], row_vecs[2], col_vecs[2],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[2][3], row_vecs[2], col_vecs[3],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[3][0], row_vecs[3], col_vecs[0],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[3][1], row_vecs[3], col_vecs[1],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[3][2], row_vecs[3], col_vecs[2],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                        inner_product_fn(&accumulators[3][3], row_vecs[3], col_vecs[3],                                \
                                         aligned_depth * dimensions_per_value, remainder_dimensions);                  \
                    }                                                                                                  \
                    else {                                                                                             \
                        for (nk_size_t row = 0; row < tile_rows; row++) {                                              \
                            nk_size_t col_start = is_diag ? row : 0;                                                   \
                            nk_size_t col_end = is_diag ? (row < 4 ? 4 : tile_cols) : tile_cols;                       \
                            for (nk_size_t col = col_start; col < col_end; col++)                                      \
                                inner_product_fn(&accumulators[row][col], row_vecs[row], col_vecs[col],                \
                                                 aligned_depth * dimensions_per_value, remainder_dimensions);          \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
                nk_##sum_value_type##_t row_sums[4] = {0}, col_sums_arr[4] = {0};                                      \
                for (nk_size_t r = 0; r < tile_rows; r++) row_sums[r] = precomputed_sums[tile_row_start + r];          \
                for (nk_size_t c = 0; c < tile_cols; c++)                                                              \
                    col_sums_arr[c] = is_diag ? row_sums[c] : precomputed_sums[tile_col_start + c];                    \
                /* Build column sums as SIMD vector — for diagonal tiles, shift per row */                             \
                result_vec_type col_sum_vec;                                                                           \
                if (!is_diag) partial_load_sum_fn(col_sums_arr, &col_sum_vec, tile_cols);                              \
                /* Finalize with compensation */                                                                       \
                for (nk_size_t row = 0; row < tile_rows; row++) {                                                      \
                    if (is_diag) {                                                                                     \
                        nk_##sum_value_type##_t shifted[4] = {0};                                                      \
                        for (nk_size_t c = 0; c < 4 && (row + c) < tile_cols; c++) shifted[c] = col_sums_arr[row + c]; \
                        partial_load_sum_fn(shifted, &col_sum_vec, 4);                                                 \
                    }                                                                                                  \
                    result_vec_type rv;                                                                                \
                    compensated_finalize_fn(                                                                           \
                        &accumulators[row][is_diag ? row : 0], &accumulators[row][(is_diag ? row : 0) + 1],            \
                        &accumulators[row][(is_diag ? row : 0) + 2], &accumulators[row][(is_diag ? row : 0) + 3],      \
                        depth, row_sums[row], &col_sum_vec, &rv);                                                      \
                    nk_size_t global_row = i_macro + tile_row_start + row;                                             \
                    nk_size_t global_col_start = i_macro + tile_col_start + (is_diag ? row : 0);                       \
                    nk_size_t store_count = is_diag ? (tile_cols - row) : tile_cols;                                   \
                    nk_##result_value_type##_t *dest = result + global_row * result_stride_values + global_col_start;  \
                    partial_store_fn(&rv, dest, store_count);                                                          \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    /* Off-diagonal helper: 4×4 tiling with inline sum accumulation (16 FMAs + up to 8 SADs per depth step) */         \
    NK_INTERNAL void nk_##api_name##_symmetric_offdiagonal_##input_type_name##_##isa_suffix##_(                        \
        nk_##input_value_type##_t const **row_ptrs_macro, nk_##input_value_type##_t const **col_ptrs_macro,            \
        nk_size_t i_macro, nk_size_t j_macro, nk_size_t macro_i_size, nk_size_t macro_j_size, nk_size_t aligned_depth, \
        nk_size_t remainder_depth, nk_size_t remainder_dimensions, nk_size_t depth_step_values,                        \
        nk_size_t dimensions_per_value_runtime, nk_##result_value_type##_t *result, nk_size_t result_stride_values,    \
        nk_size_t finalizer_batch_size, nk_size_t depth) {                                                             \
        nk_unused_(finalizer_batch_size);                                                                              \
        nk_unused_(dimensions_per_value_runtime);                                                                      \
        nk_size_t padded_depth_dimensions = aligned_depth * dimensions_per_value +                                     \
                                            (remainder_depth > 0 ? depth_simd_dimensions : 0);                         \
        /* Sum caches for this macro-tile pair — computed once, reused across tiles */                                 \
        nk_##sum_value_type##_t row_sums[32], col_sums[32];                                                            \
        for (nk_size_t tile_row_start = 0; tile_row_start < macro_i_size; tile_row_start += 4) {                       \
            for (nk_size_t tile_col_start = 0; tile_col_start < macro_j_size; tile_col_start += 4) {                   \
                nk_size_t tile_rows = (tile_row_start + 4 <= macro_i_size) ? 4 : (macro_i_size - tile_row_start);      \
                nk_size_t tile_cols = (tile_col_start + 4 <= macro_j_size) ? 4 : (macro_j_size - tile_col_start);      \
                /* Determine if this tile should compute sums — predictable branches */                                \
                int compute_row_sums_flag = (tile_col_start == 0);                                                     \
                int compute_col_sums_flag = (tile_row_start == 0);                                                     \
                /* Initialize 4×4 dot accumulators */                                                                  \
                NK_ALIGN64 state_type accumulators[4][4];                                                              \
                for (nk_size_t row = 0; row < tile_rows; row++)                                                        \
                    for (nk_size_t col = 0; col < tile_cols; col++) init_accumulator_fn(&accumulators[row][col]);      \
                /* Initialize sum accumulators (only when needed) */                                                   \
                sum_state_type rsum[4], csum[4];                                                                       \
                if (compute_row_sums_flag)                                                                             \
                    for (nk_size_t r = 0; r < tile_rows; r++) init_sum_fn(&rsum[r]);                                   \
                if (compute_col_sums_flag)                                                                             \
                    for (nk_size_t c = 0; c < tile_cols; c++) init_sum_fn(&csum[c]);                                   \
                /* Setup pointers (hoist outside depth loop) */                                                        \
                nk_##input_value_type##_t const *row_ptrs[4], *col_ptrs[4];                                            \
                row_ptrs[0] = row_ptrs_macro[tile_row_start + 0];                                                      \
                row_ptrs[1] = (tile_rows > 1) ? row_ptrs_macro[tile_row_start + 1] : row_ptrs[0];                      \
                row_ptrs[2] = (tile_rows > 2) ? row_ptrs_macro[tile_row_start + 2] : row_ptrs[0];                      \
                row_ptrs[3] = (tile_rows > 3) ? row_ptrs_macro[tile_row_start + 3] : row_ptrs[0];                      \
                col_ptrs[0] = col_ptrs_macro[tile_col_start + 0];                                                      \
                col_ptrs[1] = (tile_cols > 1) ? col_ptrs_macro[tile_col_start + 1] : col_ptrs[0];                      \
                col_ptrs[2] = (tile_cols > 2) ? col_ptrs_macro[tile_col_start + 2] : col_ptrs[0];                      \
                col_ptrs[3] = (tile_cols > 3) ? col_ptrs_macro[tile_col_start + 3] : col_ptrs[0];                      \
                /* Depth loop — innermost, 16 FMAs + up to 8 SADs per iteration */                                     \
                vec_type row_vecs[4], col_vecs[4];                                                                     \
                for (nk_size_t di = 0; di < aligned_depth; di += depth_step_values) {                                  \
                    load_vec_fn(row_ptrs[0] + di, &row_vecs[0]);                                                       \
                    load_vec_fn(row_ptrs[1] + di, &row_vecs[1]);                                                       \
                    load_vec_fn(row_ptrs[2] + di, &row_vecs[2]);                                                       \
                    load_vec_fn(row_ptrs[3] + di, &row_vecs[3]);                                                       \
                    load_vec_fn(col_ptrs[0] + di, &col_vecs[0]);                                                       \
                    load_vec_fn(col_ptrs[1] + di, &col_vecs[1]);                                                       \
                    load_vec_fn(col_ptrs[2] + di, &col_vecs[2]);                                                       \
                    load_vec_fn(col_ptrs[3] + di, &col_vecs[3]);                                                       \
                    nk_size_t vector_offset = di * dimensions_per_value;                                               \
                    if (tile_rows == 4 && tile_cols == 4) {                                                            \
                        inner_product_fn(&accumulators[0][0], row_vecs[0], col_vecs[0], vector_offset,                 \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[0][1], row_vecs[0], col_vecs[1], vector_offset,                 \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[0][2], row_vecs[0], col_vecs[2], vector_offset,                 \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[0][3], row_vecs[0], col_vecs[3], vector_offset,                 \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[1][0], row_vecs[1], col_vecs[0], vector_offset,                 \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[1][1], row_vecs[1], col_vecs[1], vector_offset,                 \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[1][2], row_vecs[1], col_vecs[2], vector_offset,                 \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[1][3], row_vecs[1], col_vecs[3], vector_offset,                 \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[2][0], row_vecs[2], col_vecs[0], vector_offset,                 \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[2][1], row_vecs[2], col_vecs[1], vector_offset,                 \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[2][2], row_vecs[2], col_vecs[2], vector_offset,                 \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[2][3], row_vecs[2], col_vecs[3], vector_offset,                 \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[3][0], row_vecs[3], col_vecs[0], vector_offset,                 \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[3][1], row_vecs[3], col_vecs[1], vector_offset,                 \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[3][2], row_vecs[3], col_vecs[2], vector_offset,                 \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[3][3], row_vecs[3], col_vecs[3], vector_offset,                 \
                                         depth_simd_dimensions);                                                       \
                    }                                                                                                  \
                    else {                                                                                             \
                        for (nk_size_t row = 0; row < tile_rows; row++)                                                \
                            for (nk_size_t col = 0; col < tile_cols; col++)                                            \
                                inner_product_fn(&accumulators[row][col], row_vecs[row], col_vecs[col], vector_offset, \
                                                 depth_simd_dimensions);                                               \
                    }                                                                                                  \
                    /* Progressive sum accumulation (SADs on port 5, parallel with DPBUSD on ports 0+1) */             \
                    if (compute_row_sums_flag) {                                                                       \
                        update_sum_fn(&rsum[0], row_vecs[0]);                                                          \
                        if (tile_rows > 1) update_sum_fn(&rsum[1], row_vecs[1]);                                       \
                        if (tile_rows > 2) update_sum_fn(&rsum[2], row_vecs[2]);                                       \
                        if (tile_rows > 3) update_sum_fn(&rsum[3], row_vecs[3]);                                       \
                    }                                                                                                  \
                    if (compute_col_sums_flag) {                                                                       \
                        update_sum_fn(&csum[0], col_vecs[0]);                                                          \
                        if (tile_cols > 1) update_sum_fn(&csum[1], col_vecs[1]);                                       \
                        if (tile_cols > 2) update_sum_fn(&csum[2], col_vecs[2]);                                       \
                        if (tile_cols > 3) update_sum_fn(&csum[3], col_vecs[3]);                                       \
                    }                                                                                                  \
                }                                                                                                      \
                /* Handle remainder depth */                                                                           \
                if (remainder_depth > 0) {                                                                             \
                    partial_load_vec_fn(row_ptrs[0] + aligned_depth, &row_vecs[0], remainder_dimensions);              \
                    partial_load_vec_fn(row_ptrs[1] + aligned_depth, &row_vecs[1], remainder_dimensions);              \
                    partial_load_vec_fn(row_ptrs[2] + aligned_depth, &row_vecs[2], remainder_dimensions);              \
                    partial_load_vec_fn(row_ptrs[3] + aligned_depth, &row_vecs[3], remainder_dimensions);              \
                    partial_load_vec_fn(col_ptrs[0] + aligned_depth, &col_vecs[0], remainder_dimensions);              \
                    partial_load_vec_fn(col_ptrs[1] + aligned_depth, &col_vecs[1], remainder_dimensions);              \
                    partial_load_vec_fn(col_ptrs[2] + aligned_depth, &col_vecs[2], remainder_dimensions);              \
                    partial_load_vec_fn(col_ptrs[3] + aligned_depth, &col_vecs[3], remainder_dimensions);              \
                    nk_size_t vector_offset = aligned_depth * dimensions_per_value;                                    \
                    for (nk_size_t row = 0; row < tile_rows; row++)                                                    \
                        for (nk_size_t col = 0; col < tile_cols; col++)                                                \
                            inner_product_fn(&accumulators[row][col], row_vecs[row], col_vecs[col], vector_offset,     \
                                             remainder_dimensions);                                                    \
                    if (compute_row_sums_flag) {                                                                       \
                        update_sum_fn(&rsum[0], row_vecs[0]);                                                          \
                        if (tile_rows > 1) update_sum_fn(&rsum[1], row_vecs[1]);                                       \
                        if (tile_rows > 2) update_sum_fn(&rsum[2], row_vecs[2]);                                       \
                        if (tile_rows > 3) update_sum_fn(&rsum[3], row_vecs[3]);                                       \
                    }                                                                                                  \
                    if (compute_col_sums_flag) {                                                                       \
                        update_sum_fn(&csum[0], col_vecs[0]);                                                          \
                        if (tile_cols > 1) update_sum_fn(&csum[1], col_vecs[1]);                                       \
                        if (tile_cols > 2) update_sum_fn(&csum[2], col_vecs[2]);                                       \
                        if (tile_cols > 3) update_sum_fn(&csum[3], col_vecs[3]);                                       \
                    }                                                                                                  \
                }                                                                                                      \
                /* Finalize and cache sums */                                                                          \
                if (compute_row_sums_flag)                                                                             \
                    for (nk_size_t r = 0; r < tile_rows; r++)                                                          \
                        row_sums[tile_row_start + r] = finalize_sum_fn(&rsum[r], padded_depth_dimensions);             \
                if (compute_col_sums_flag)                                                                             \
                    for (nk_size_t c = 0; c < tile_cols; c++)                                                          \
                        col_sums[tile_col_start + c] = finalize_sum_fn(&csum[c], padded_depth_dimensions);             \
                /* Build col_sum SIMD vector once (constant across rows) */                                            \
                nk_##sum_value_type##_t cs_arr[4] = {0};                                                               \
                for (nk_size_t c = 0; c < tile_cols; c++) cs_arr[c] = col_sums[tile_col_start + c];                    \
                result_vec_type cs_vec;                                                                                \
                partial_load_sum_fn(cs_arr, &cs_vec, tile_cols);                                                       \
                /* Compensated finalize + store */                                                                     \
                for (nk_size_t row = 0; row < tile_rows; row++) {                                                      \
                    result_vec_type rv;                                                                                \
                    compensated_finalize_fn(&accumulators[row][0], &accumulators[row][1], &accumulators[row][2],       \
                                            &accumulators[row][3], depth, row_sums[tile_row_start + row], &cs_vec,     \
                                            &rv);                                                                      \
                    nk_##result_value_type##_t *dest = result +                                                        \
                                                       (i_macro + tile_row_start + row) * result_stride_values +       \
                                                       (j_macro + tile_col_start);                                     \
                    partial_store_fn(&rv, dest, tile_cols);                                                            \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    NK_PUBLIC void nk_##api_name##_symmetric_##input_type_name##_##isa_suffix(                                         \
        nk_##input_value_type##_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_in_bytes, \
        nk_##result_value_type##_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start,                     \
        nk_size_t row_count) {                                                                                         \
        nk_size_t const macro_tile_size = 32;                                                                          \
        nk_size_t const row_block_size = 128;     /* L2 cache blocking */                                              \
        nk_size_t const column_block_size = 2048; /* L3 cache blocking */                                              \
        nk_size_t const depth_dimensions_aligned = (depth / depth_simd_dimensions) * depth_simd_dimensions;            \
        nk_size_t const aligned_depth = nk_size_divide_round_up_(depth_dimensions_aligned, dimensions_per_value);      \
        nk_size_t const depth_in_values = nk_size_divide_round_up_(depth, dimensions_per_value);                       \
        nk_size_t const remainder_depth = depth_in_values - aligned_depth;                                             \
        nk_size_t const remainder_dimensions = depth - depth_dimensions_aligned;                                       \
        nk_size_t const depth_step = nk_size_divide_round_up_(depth_simd_dimensions, dimensions_per_value);            \
        nk_size_t const result_stride_values = result_stride_in_bytes / sizeof(nk_##result_value_type##_t);            \
        nk_size_t const row_end = (row_start + row_count < vectors_count) ? (row_start + row_count) : vectors_count;   \
                                                                                                                       \
        /* Process upper triangle with L3/L2/L1 blocking (column blocks → row blocks → 32×32 macro-tiles) */           \
        for (nk_size_t j_block = 0; j_block < vectors_count; j_block += column_block_size) {                           \
            nk_size_t j_block_end = (j_block + column_block_size < vectors_count) ? j_block + column_block_size        \
                                                                                  : vectors_count;                     \
                                                                                                                       \
            for (nk_size_t i_block = row_start; i_block < row_end; i_block += row_block_size) {                        \
                nk_size_t i_block_end = (i_block + row_block_size < row_end) ? i_block + row_block_size : row_end;     \
                                                                                                                       \
                /* Skip blocks entirely below diagonal. Blocks fully above the diagonal are still part of the upper    \
                 * triangle and must be computed. */                                                                   \
                if (i_block >= j_block_end) continue;                                                                  \
                                                                                                                       \
                for (nk_size_t i_macro = i_block; i_macro < i_block_end; i_macro += macro_tile_size) {                 \
                    /* Upper triangle: j_macro starts at max(i_macro, j_block) */                                      \
                    nk_size_t j_start = (i_macro > j_block) ? i_macro : j_block;                                       \
                    for (nk_size_t j_macro = j_start; j_macro < j_block_end; j_macro += macro_tile_size) {             \
                        nk_size_t macro_i_size = (i_macro + macro_tile_size <= i_block_end) ? macro_tile_size          \
                                                                                            : (i_block_end - i_macro); \
                        nk_size_t macro_j_size = (j_macro + macro_tile_size <= j_block_end) ? macro_tile_size          \
                                                                                            : (j_block_end - j_macro); \
                                                                                                                       \
                        /* Build pointer arrays */                                                                     \
                        nk_##input_value_type##_t const *vec_ptrs_i[32];                                               \
                        nk_##input_value_type##_t const *vec_ptrs_j[32];                                               \
                        for (nk_size_t k = 0; k < macro_i_size; k++)                                                   \
                            vec_ptrs_i[k] = (nk_##input_value_type##_t const *)((char const *)vectors +                \
                                                                                (i_macro + k) * stride_in_bytes);      \
                        for (nk_size_t k = macro_i_size; k < 32; k++) vec_ptrs_i[k] = vec_ptrs_i[0];                   \
                                                                                                                       \
                        if (i_macro == j_macro && macro_i_size == macro_j_size) {                                      \
                            /* Diagonal macro-tile */                                                                  \
                            nk_##api_name##_symmetric_diagonal_##input_type_name##_##isa_suffix##_(                    \
                                vec_ptrs_i, i_macro, macro_i_size, aligned_depth, remainder_depth,                     \
                                remainder_dimensions, depth_step, dimensions_per_value, result, result_stride_values,  \
                                4, depth);                                                                             \
                        }                                                                                              \
                        else {                                                                                         \
                            /* Off-diagonal macro-tile */                                                              \
                            for (nk_size_t k = 0; k < macro_j_size; k++)                                               \
                                vec_ptrs_j[k] = (nk_##input_value_type##_t const *)((char const *)vectors +            \
                                                                                    (j_macro + k) * stride_in_bytes);  \
                            for (nk_size_t k = macro_j_size; k < 32; k++) vec_ptrs_j[k] = vec_ptrs_j[0];               \
                            nk_##api_name##_symmetric_offdiagonal_##input_type_name##_##isa_suffix##_(                 \
                                vec_ptrs_i, vec_ptrs_j, i_macro, j_macro, macro_i_size, macro_j_size, aligned_depth,   \
                                remainder_depth, remainder_dimensions, depth_step, dimensions_per_value, result,       \
                                result_stride_values, 4, depth);                                                       \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }

/**
 *  @brief Generates optimized symmetric Gram matrix computation: C = A × Aᵀ (upper triangle only).
 *
 *  This macro creates a complete symmetric cross-product implementation with two specialized
 *  internal helper functions (diagonal and off-diagonal) that are called by a public wrapper.
 *  Symmetric computation exploits the property that C[i,j] = C[j,i], computing only the upper
 *  triangle and avoiding redundant computation and storage.
 *
 *  @par Mathematical Operation For each pair (i,j) where i ≤ j:
 *    C[i,j] = operation(A[i,:], A[j,:])
 *  where operation can be dot product, Hamming distance, Jaccard similarity, etc.
 *
 *  @par Architecture - Three-Level Tiling Hierarchy
 *
 *  1. @b 32×32 @b macro-tiles (outermost): Divides the upper triangle into 32×32 blocks
 *     - Rationale: Fits well in L1 cache (32 vectors × depth × value_size)
 *     - Enables diagonal vs off-diagonal specialization
 *     - Amortizes vector loads across all depth iterations
 *     - Pre-loads and upcasts ALL 32 vectors ONCE per depth iteration (not per FMA)
 *
 *  2. @b 4×4 @b register @b tiles (middle): Within each macro-tile, process 4×4 sub-blocks
 *     - Rationale: Maximizes register reuse (4 A vectors × 4 A vectors = 16 accumulators)
 *     - Enables full FMA unrolling (16 FMAs for off-diagonal, 10 for diagonal)
 *     - Balances register pressure with instruction-level parallelism
 *
 *  3. @b Depth @b loop (innermost): For each depth chunk, accumulate outer products
 *     - Depth loop is INSIDE macro-tile, OUTSIDE register tiles
 *     - Type conversion (e.g., bf16→f32) happens at macro-tile level (once per vector)
 *
 *  @par Diagonal vs Off-Diagonal Optimization
 *
 *  - @b Diagonal @b macro-tiles (i_macro == j_macro): Computes C[i:i+32, i:i+32]
 *    - Loads 32 vectors ONCE (50% load reduction vs off-diagonal)
 *    - Computes upper triangle only within the tile (10 FMAs per 4×4 block)
 *    - Uses nk_##api_name##_symmetric_diagonal_##input_type_name##_##isa_suffix##_ helper
 *
 *  - @b Off-diagonal @b macro-tiles (i_macro < j_macro): Computes C[i:i+32, j:j+32]
 *    - Loads vec_i[32] + vec_j[32] (full 64 vectors for two sets)
 *    - Computes full 32×32 block (16 FMAs per 4×4 block)
 *    - Uses nk_##api_name##_symmetric_offdiagonal_##input_type_name##_##isa_suffix##_ helper
 *
 *  @par When to Use Symmetric vs Packed Variant
 *
 *  - Use symmetric (this macro) when: A is the SAME matrix for both sides (C = A × Aᵀ)
 *    - Saves 50% computation and storage (upper triangle only)
 *    - Automatic diagonal optimization (50% fewer loads on diagonal tiles)
 *    - Ideal for: distance matrices, correlation matrices, Gram matrices
 *
 *  - Use packed variant when: Computing C = A × Bᵀ where A ≠ B
 *    - Full matrix computation (no symmetry to exploit)
 *    - B can be pre-packed for cache efficiency
 *
 *  @par Generated Functions
 *
 *  This macro generates THREE functions:
 *  1. nk_##api_name##_symmetric_diagonal_##input_type_name##_##isa_suffix##_ (NK_INTERNAL)
 *  2. nk_##api_name##_symmetric_offdiagonal_##input_type_name##_##isa_suffix##_ (NK_INTERNAL)
 *  3. nk_##api_name##_symmetric_##input_type_name##_##isa_suffix (NK_PUBLIC wrapper)
 *
 *  @param api_name Operation family (dots, hammings, jaccards) for codegen namespace
 *  @param input_type_name Type identifier for codegen (f32, bf16, i8, u1, etc.)
 *  @param isa_suffix ISA backend identifier (serial, haswell, neon, sve, icelake, etc.)
 *  @param input_type C type of input matrix values (f32, bf16, i8, u1x8, etc.)
 *  @param output_type C type of output matrix values (f32, u32, f64, etc.)
 *  @param vec_type SIMD vector type for input vectors (e.g., __m256, nk_f32x8_t)
 *  @param state_type Accumulator state type (often vec_type or wider, e.g., __m256 or __m512)
 *  @param result_vec_type SIMD vector type for reduction results (e.g., __m128 for 4 f32 results)
 *  @param init_accumulator_fn Initialize accumulator: void fn(state_type*)
 *  @param load_vec_fn Full vector load: vec_type fn(input_type const*, nk_size_t offset)
 *  @param partial_load_vec_fn Partial vector load for remainder
 *  @param inner_product_fn Inner product accumulate
 *  @param reduce_accumulators_fn Reduce 4 accumulators
 *  @param partial_store_fn Partial store for results
 *  @param depth_simd_dimensions SIMD vector width in logical dimensions (e.g., 8 for f32 on AVX2, 128 for u1 on serial)
 *  @param dimensions_per_value Packing ratio: dimensions per storage value (1 for f32, 2 for i4x2, 8 for u1x8)
 *
 *  @sa nk_define_cross_packed_ for asymmetric C = A × Bᵀ computation
 *  @sa nk_define_cross_pack_size_ for calculating packed buffer size
 *  @sa nk_define_cross_pack_ for packing B matrix
 *  @sa include/numkong/set/serial.h for state type definitions
 *  @sa include/numkong/cast/serial.h for load/store function implementations
 */
#define nk_define_cross_symmetric_(api_name, input_type_name, isa_suffix, input_value_type, result_value_type,         \
                                   vec_type, state_type, result_vec_type, init_accumulator_fn, load_vec_fn,            \
                                   partial_load_vec_fn, inner_product_fn, reduce_accumulators_fn, store_fn,            \
                                   partial_store_fn, depth_simd_dimensions, dimensions_per_value)                      \
    NK_INTERNAL void nk_##api_name##_symmetric_diagonal_##input_type_name##_##isa_suffix##_(                           \
        nk_##input_value_type##_t const **vector_base_ptrs, nk_size_t i_macro, nk_size_t macro_size,                   \
        nk_size_t aligned_depth, nk_size_t remainder_depth, nk_size_t remainder_dimensions,                            \
        nk_size_t depth_step_values, nk_size_t dimensions_per_value_runtime, nk_##result_value_type##_t *result,       \
        nk_size_t result_stride_values, nk_size_t finalizer_batch_size, nk_size_t depth) {                             \
                                                                                                                       \
        nk_unused_(dimensions_per_value_runtime);                                                                      \
        nk_unused_(finalizer_batch_size);                                                                              \
        /* Tile-first architecture: Process 32×32 macro-tile as 4×4 register tiles (depth innermost) */                \
        for (nk_size_t tile_row_start = 0; tile_row_start < macro_size; tile_row_start += 4) {                         \
            for (nk_size_t tile_column_start = tile_row_start; tile_column_start < macro_size;                         \
                 tile_column_start += 4) {                                                                             \
                                                                                                                       \
                nk_size_t tile_rows = (tile_row_start + 4 <= macro_size) ? 4 : (macro_size - tile_row_start);          \
                nk_size_t tile_columns = (tile_column_start + 4 <= macro_size) ? 4 : (macro_size - tile_column_start); \
                int is_diagonal_tile = (tile_row_start == tile_column_start);                                          \
                                                                                                                       \
                /* Initialize register-resident accumulators — padded to [4][7] so that the reduce call  */            \
                /* (which always reads 4 consecutive entries starting at column_start) stays in bounds */              \
                NK_ALIGN64 state_type accumulators[4][7];                                                              \
                for (nk_size_t row = 0; row < tile_rows; row++) {                                                      \
                    nk_size_t init_start = is_diagonal_tile ? row : 0;                                                 \
                    nk_size_t init_end = is_diagonal_tile ? (row + 4) : tile_columns;                                  \
                    for (nk_size_t column = init_start; column < init_end; column++) {                                 \
                        init_accumulator_fn(&accumulators[row][column]);                                               \
                    }                                                                                                  \
                }                                                                                                      \
                                                                                                                       \
                /* Setup pointers (hoist outside depth loop) - always safe even for partial tiles */                   \
                nk_##input_value_type##_t const *row_ptrs[4];                                                          \
                nk_##input_value_type##_t const *column_ptrs[4];                                                       \
                row_ptrs[0] = vector_base_ptrs[tile_row_start + 0];                                                    \
                row_ptrs[1] = (tile_rows > 1) ? vector_base_ptrs[tile_row_start + 1] : row_ptrs[0];                    \
                row_ptrs[2] = (tile_rows > 2) ? vector_base_ptrs[tile_row_start + 2] : row_ptrs[0];                    \
                row_ptrs[3] = (tile_rows > 3) ? vector_base_ptrs[tile_row_start + 3] : row_ptrs[0];                    \
                                                                                                                       \
                if (is_diagonal_tile) {                                                                                \
                    column_ptrs[0] = row_ptrs[0];                                                                      \
                    column_ptrs[1] = row_ptrs[1];                                                                      \
                    column_ptrs[2] = row_ptrs[2];                                                                      \
                    column_ptrs[3] = row_ptrs[3];                                                                      \
                }                                                                                                      \
                else {                                                                                                 \
                    column_ptrs[0] = vector_base_ptrs[tile_column_start + 0];                                          \
                    column_ptrs[1] = (tile_columns > 1) ? vector_base_ptrs[tile_column_start + 1] : column_ptrs[0];    \
                    column_ptrs[2] = (tile_columns > 2) ? vector_base_ptrs[tile_column_start + 2] : column_ptrs[0];    \
                    column_ptrs[3] = (tile_columns > 3) ? vector_base_ptrs[tile_column_start + 3] : column_ptrs[0];    \
                }                                                                                                      \
                                                                                                                       \
                /* Depth loop is now innermost - key optimization */                                                   \
                vec_type row_vecs[4];                                                                                  \
                vec_type column_vecs[4];                                                                               \
                                                                                                                       \
                for (nk_size_t depth_offset = 0; depth_offset < aligned_depth; depth_offset += depth_step_values) {    \
                    /* Always load all 4 vectors - aliasing is cheaper than branches */                                \
                    load_vec_fn(row_ptrs[0] + depth_offset, &row_vecs[0]);                                             \
                    load_vec_fn(row_ptrs[1] + depth_offset, &row_vecs[1]);                                             \
                    load_vec_fn(row_ptrs[2] + depth_offset, &row_vecs[2]);                                             \
                    load_vec_fn(row_ptrs[3] + depth_offset, &row_vecs[3]);                                             \
                                                                                                                       \
                    /* For diagonal tiles, column vectors alias row vectors (same memory) */                           \
                    load_vec_fn(column_ptrs[0] + depth_offset, &column_vecs[0]);                                       \
                    load_vec_fn(column_ptrs[1] + depth_offset, &column_vecs[1]);                                       \
                    load_vec_fn(column_ptrs[2] + depth_offset, &column_vecs[2]);                                       \
                    load_vec_fn(column_ptrs[3] + depth_offset, &column_vecs[3]);                                       \
                                                                                                                       \
                    nk_size_t vector_offset = depth_offset * dimensions_per_value;                                     \
                                                                                                                       \
                    /* Compute: always unroll for full 4×4, use loops only for partial tiles */                        \
                    if (tile_rows == 4 && tile_columns == 4) {                                                         \
                        if (is_diagonal_tile) {                                                                        \
                            /* Full 4×4 diagonal tile - upper triangle only (10 FMAs) */                               \
                            inner_product_fn(&accumulators[0][0], row_vecs[0], column_vecs[0], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[0][1], row_vecs[0], column_vecs[1], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[0][2], row_vecs[0], column_vecs[2], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[0][3], row_vecs[0], column_vecs[3], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[1][1], row_vecs[1], column_vecs[1], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[1][2], row_vecs[1], column_vecs[2], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[1][3], row_vecs[1], column_vecs[3], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[2][2], row_vecs[2], column_vecs[2], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[2][3], row_vecs[2], column_vecs[3], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[3][3], row_vecs[3], column_vecs[3], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                        }                                                                                              \
                        else {                                                                                         \
                            /* Full 4×4 off-diagonal tile (16 FMAs) */                                                 \
                            inner_product_fn(&accumulators[0][0], row_vecs[0], column_vecs[0], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[0][1], row_vecs[0], column_vecs[1], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[0][2], row_vecs[0], column_vecs[2], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[0][3], row_vecs[0], column_vecs[3], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[1][0], row_vecs[1], column_vecs[0], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[1][1], row_vecs[1], column_vecs[1], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[1][2], row_vecs[1], column_vecs[2], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[1][3], row_vecs[1], column_vecs[3], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[2][0], row_vecs[2], column_vecs[0], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[2][1], row_vecs[2], column_vecs[1], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[2][2], row_vecs[2], column_vecs[2], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[2][3], row_vecs[2], column_vecs[3], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[3][0], row_vecs[3], column_vecs[0], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[3][1], row_vecs[3], column_vecs[1], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[3][2], row_vecs[3], column_vecs[2], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                            inner_product_fn(&accumulators[3][3], row_vecs[3], column_vecs[3], vector_offset,          \
                                             depth_simd_dimensions);                                                   \
                        }                                                                                              \
                    }                                                                                                  \
                    else {                                                                                             \
                        /* Partial tile - use loops (rare edge case) */                                                \
                        for (nk_size_t row = 0; row < tile_rows; row++) {                                              \
                            nk_size_t column_start = is_diagonal_tile ? row : 0;                                       \
                            for (nk_size_t column = column_start; column < tile_columns; column++) {                   \
                                inner_product_fn(&accumulators[row][column], row_vecs[row], column_vecs[column],       \
                                                 vector_offset, depth_simd_dimensions);                                \
                            }                                                                                          \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
                                                                                                                       \
                /* Handle remainder depth (happens once per tile, not in hot loop) */                                  \
                if (remainder_depth > 0) {                                                                             \
                    partial_load_vec_fn(row_ptrs[0] + aligned_depth, &row_vecs[0], remainder_dimensions);              \
                    partial_load_vec_fn(row_ptrs[1] + aligned_depth, &row_vecs[1], remainder_dimensions);              \
                    partial_load_vec_fn(row_ptrs[2] + aligned_depth, &row_vecs[2], remainder_dimensions);              \
                    partial_load_vec_fn(row_ptrs[3] + aligned_depth, &row_vecs[3], remainder_dimensions);              \
                    partial_load_vec_fn(column_ptrs[0] + aligned_depth, &column_vecs[0], remainder_dimensions);        \
                    partial_load_vec_fn(column_ptrs[1] + aligned_depth, &column_vecs[1], remainder_dimensions);        \
                    partial_load_vec_fn(column_ptrs[2] + aligned_depth, &column_vecs[2], remainder_dimensions);        \
                    partial_load_vec_fn(column_ptrs[3] + aligned_depth, &column_vecs[3], remainder_dimensions);        \
                                                                                                                       \
                    nk_size_t vector_offset = aligned_depth * dimensions_per_value;                                    \
                    for (nk_size_t row = 0; row < tile_rows; row++) {                                                  \
                        nk_size_t column_start = is_diagonal_tile ? row : 0;                                           \
                        for (nk_size_t column = column_start; column < tile_columns; column++) {                       \
                            inner_product_fn(&accumulators[row][column], row_vecs[row], column_vecs[column],           \
                                             vector_offset, remainder_dimensions);                                     \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
                                                                                                                       \
                /* Direct finalization and store (no intermediate buffer) */                                           \
                for (nk_size_t row = 0; row < tile_rows; row++) {                                                      \
                    nk_size_t column_start = is_diagonal_tile ? row : 0;                                               \
                    nk_size_t columns_remaining = tile_columns - column_start;                                         \
                    result_vec_type result_vec;                                                                        \
                                                                                                                       \
                    /* Always reduce 4 accumulators (partial_store handles actual count) */                            \
                    reduce_accumulators_fn(&accumulators[row][column_start], &accumulators[row][column_start + 1],     \
                                           &accumulators[row][column_start + 2], &accumulators[row][column_start + 3], \
                                           depth, &result_vec);                                                        \
                                                                                                                       \
                    nk_##result_value_type##_t *output_ptr =                                                           \
                        &result[(i_macro + tile_row_start + row) * result_stride_values +                              \
                                (i_macro + tile_column_start + column_start)];                                         \
                    partial_store_fn(&result_vec, output_ptr, columns_remaining);                                      \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    NK_INTERNAL void nk_##api_name##_symmetric_##input_type_name##_##isa_suffix##_offdiagonal_(                        \
        nk_##input_value_type##_t const **vector_base_ptrs_i, nk_##input_value_type##_t const **vector_base_ptrs_j,    \
        nk_size_t i_macro, nk_size_t j_macro, nk_size_t macro_i_size, nk_size_t macro_j_size, nk_size_t aligned_depth, \
        nk_size_t remainder_depth, nk_size_t remainder_dimensions, nk_size_t depth_step_values,                        \
        nk_size_t dimensions_per_value_runtime, nk_##result_value_type##_t *result, nk_size_t result_stride_values,    \
        nk_size_t finalizer_batch_size, nk_size_t depth) {                                                             \
                                                                                                                       \
        nk_unused_(dimensions_per_value_runtime);                                                                      \
        nk_unused_(finalizer_batch_size);                                                                              \
        /* Tile-first architecture: Process 32×32 macro-tile as 4×4 register tiles (depth innermost) */                \
        for (nk_size_t tile_row_start = 0; tile_row_start < macro_i_size; tile_row_start += 4) {                       \
            for (nk_size_t tile_column_start = 0; tile_column_start < macro_j_size; tile_column_start += 4) {          \
                                                                                                                       \
                nk_size_t tile_rows = (tile_row_start + 4 <= macro_i_size) ? 4 : (macro_i_size - tile_row_start);      \
                nk_size_t tile_columns = (tile_column_start + 4 <= macro_j_size) ? 4                                   \
                                                                                 : (macro_j_size - tile_column_start); \
                                                                                                                       \
                /* Initialize 4×4 register-resident accumulators (full rectangle for off-diagonal) */                  \
                NK_ALIGN64 state_type accumulators[4][4];                                                              \
                for (nk_size_t row = 0; row < tile_rows; row++) {                                                      \
                    for (nk_size_t column = 0; column < tile_columns; column++) {                                      \
                        init_accumulator_fn(&accumulators[row][column]);                                               \
                    }                                                                                                  \
                }                                                                                                      \
                                                                                                                       \
                /* Setup pointers (hoist outside depth loop) - always safe even for partial tiles */                   \
                nk_##input_value_type##_t const *row_ptrs[4];                                                          \
                nk_##input_value_type##_t const *column_ptrs[4];                                                       \
                row_ptrs[0] = vector_base_ptrs_i[tile_row_start + 0];                                                  \
                row_ptrs[1] = (tile_rows > 1) ? vector_base_ptrs_i[tile_row_start + 1] : row_ptrs[0];                  \
                row_ptrs[2] = (tile_rows > 2) ? vector_base_ptrs_i[tile_row_start + 2] : row_ptrs[0];                  \
                row_ptrs[3] = (tile_rows > 3) ? vector_base_ptrs_i[tile_row_start + 3] : row_ptrs[0];                  \
                column_ptrs[0] = vector_base_ptrs_j[tile_column_start + 0];                                            \
                column_ptrs[1] = (tile_columns > 1) ? vector_base_ptrs_j[tile_column_start + 1] : column_ptrs[0];      \
                column_ptrs[2] = (tile_columns > 2) ? vector_base_ptrs_j[tile_column_start + 2] : column_ptrs[0];      \
                column_ptrs[3] = (tile_columns > 3) ? vector_base_ptrs_j[tile_column_start + 3] : column_ptrs[0];      \
                                                                                                                       \
                /* Depth loop is now innermost - key optimization */                                                   \
                vec_type row_vecs[4];                                                                                  \
                vec_type column_vecs[4];                                                                               \
                                                                                                                       \
                for (nk_size_t depth_offset = 0; depth_offset < aligned_depth; depth_offset += depth_step_values) {    \
                    /* Always load all 8 vectors - aliasing is cheaper than branches */                                \
                    load_vec_fn(row_ptrs[0] + depth_offset, &row_vecs[0]);                                             \
                    load_vec_fn(row_ptrs[1] + depth_offset, &row_vecs[1]);                                             \
                    load_vec_fn(row_ptrs[2] + depth_offset, &row_vecs[2]);                                             \
                    load_vec_fn(row_ptrs[3] + depth_offset, &row_vecs[3]);                                             \
                    load_vec_fn(column_ptrs[0] + depth_offset, &column_vecs[0]);                                       \
                    load_vec_fn(column_ptrs[1] + depth_offset, &column_vecs[1]);                                       \
                    load_vec_fn(column_ptrs[2] + depth_offset, &column_vecs[2]);                                       \
                    load_vec_fn(column_ptrs[3] + depth_offset, &column_vecs[3]);                                       \
                                                                                                                       \
                    nk_size_t vector_offset = depth_offset * dimensions_per_value;                                     \
                                                                                                                       \
                    /* Compute: always unroll for full 4×4, use loops only for partial tiles */                        \
                    if (tile_rows == 4 && tile_columns == 4) {                                                         \
                        /* Full 4×4 off-diagonal tile (16 FMAs) */                                                     \
                        inner_product_fn(&accumulators[0][0], row_vecs[0], column_vecs[0], vector_offset,              \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[0][1], row_vecs[0], column_vecs[1], vector_offset,              \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[0][2], row_vecs[0], column_vecs[2], vector_offset,              \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[0][3], row_vecs[0], column_vecs[3], vector_offset,              \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[1][0], row_vecs[1], column_vecs[0], vector_offset,              \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[1][1], row_vecs[1], column_vecs[1], vector_offset,              \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[1][2], row_vecs[1], column_vecs[2], vector_offset,              \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[1][3], row_vecs[1], column_vecs[3], vector_offset,              \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[2][0], row_vecs[2], column_vecs[0], vector_offset,              \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[2][1], row_vecs[2], column_vecs[1], vector_offset,              \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[2][2], row_vecs[2], column_vecs[2], vector_offset,              \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[2][3], row_vecs[2], column_vecs[3], vector_offset,              \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[3][0], row_vecs[3], column_vecs[0], vector_offset,              \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[3][1], row_vecs[3], column_vecs[1], vector_offset,              \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[3][2], row_vecs[3], column_vecs[2], vector_offset,              \
                                         depth_simd_dimensions);                                                       \
                        inner_product_fn(&accumulators[3][3], row_vecs[3], column_vecs[3], vector_offset,              \
                                         depth_simd_dimensions);                                                       \
                    }                                                                                                  \
                    else {                                                                                             \
                        /* Partial tile - use loops (rare edge case) */                                                \
                        for (nk_size_t row = 0; row < tile_rows; row++) {                                              \
                            for (nk_size_t column = 0; column < tile_columns; column++) {                              \
                                inner_product_fn(&accumulators[row][column], row_vecs[row], column_vecs[column],       \
                                                 vector_offset, depth_simd_dimensions);                                \
                            }                                                                                          \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
                                                                                                                       \
                /* Handle remainder depth (happens once per tile, not in hot loop) */                                  \
                if (remainder_depth > 0) {                                                                             \
                    partial_load_vec_fn(row_ptrs[0] + aligned_depth, &row_vecs[0], remainder_dimensions);              \
                    partial_load_vec_fn(row_ptrs[1] + aligned_depth, &row_vecs[1], remainder_dimensions);              \
                    partial_load_vec_fn(row_ptrs[2] + aligned_depth, &row_vecs[2], remainder_dimensions);              \
                    partial_load_vec_fn(row_ptrs[3] + aligned_depth, &row_vecs[3], remainder_dimensions);              \
                    partial_load_vec_fn(column_ptrs[0] + aligned_depth, &column_vecs[0], remainder_dimensions);        \
                    partial_load_vec_fn(column_ptrs[1] + aligned_depth, &column_vecs[1], remainder_dimensions);        \
                    partial_load_vec_fn(column_ptrs[2] + aligned_depth, &column_vecs[2], remainder_dimensions);        \
                    partial_load_vec_fn(column_ptrs[3] + aligned_depth, &column_vecs[3], remainder_dimensions);        \
                                                                                                                       \
                    nk_size_t vector_offset = aligned_depth * dimensions_per_value;                                    \
                    for (nk_size_t row = 0; row < tile_rows; row++) {                                                  \
                        for (nk_size_t column = 0; column < tile_columns; column++) {                                  \
                            inner_product_fn(&accumulators[row][column], row_vecs[row], column_vecs[column],           \
                                             vector_offset, remainder_dimensions);                                     \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
                                                                                                                       \
                /* Direct finalization and store (no intermediate buffer) */                                           \
                for (nk_size_t row = 0; row < tile_rows; row++) {                                                      \
                    result_vec_type result_vec;                                                                        \
                                                                                                                       \
                    /* Always reduce 4 accumulators (partial_store handles actual count) */                            \
                    reduce_accumulators_fn(&accumulators[row][0], &accumulators[row][1], &accumulators[row][2],        \
                                           &accumulators[row][3], depth, &result_vec);                                 \
                                                                                                                       \
                    nk_##result_value_type##_t *output_ptr =                                                           \
                        &result[(i_macro + tile_row_start + row) * result_stride_values +                              \
                                (j_macro + tile_column_start)];                                                        \
                    partial_store_fn(&result_vec, output_ptr, tile_columns);                                           \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    NK_PUBLIC void nk_##api_name##_symmetric_##input_type_name##_##isa_suffix(                                         \
        nk_##input_value_type##_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_in_bytes, \
        nk_##result_value_type##_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start,                     \
        nk_size_t row_count) {                                                                                         \
        nk_size_t const macro_tile_size = 32;                                                                          \
        nk_size_t const finalizer_batch_size = 4;                                                                      \
        nk_size_t const row_block_size = 128;     /* L2 cache blocking */                                              \
        nk_size_t const column_block_size = 2048; /* L3 cache blocking */                                              \
                                                                                                                       \
        /* Stride and depth calculations */                                                                            \
        nk_size_t const vectors_stride_values = stride_in_bytes / sizeof(nk_##input_value_type##_t);                   \
        nk_size_t const result_stride_values = result_stride_in_bytes / sizeof(nk_##result_value_type##_t);            \
        nk_size_t const depth_dimensions_aligned = (depth / depth_simd_dimensions) * depth_simd_dimensions;            \
        nk_size_t const aligned_depth = nk_size_divide_round_up_(depth_dimensions_aligned, dimensions_per_value);      \
        nk_size_t const depth_in_values = nk_size_divide_round_up_(depth, dimensions_per_value);                       \
        nk_size_t const remainder_depth = depth_in_values - aligned_depth;                                             \
        nk_size_t const remainder_dimensions = depth - depth_dimensions_aligned;                                       \
        nk_size_t const depth_step_values = nk_size_divide_round_up_(depth_simd_dimensions, dimensions_per_value);     \
        nk_size_t const row_end = (row_start + row_count < vectors_count) ? (row_start + row_count) : vectors_count;   \
                                                                                                                       \
        /* Process upper triangle with L3/L2/L1 blocking (column blocks → row blocks → 32×32 macro-tiles) */           \
        for (nk_size_t j_block = 0; j_block < vectors_count; j_block += column_block_size) {                           \
            nk_size_t j_block_end = (j_block + column_block_size < vectors_count) ? j_block + column_block_size        \
                                                                                  : vectors_count;                     \
                                                                                                                       \
            for (nk_size_t i_block = row_start; i_block < row_end; i_block += row_block_size) {                        \
                nk_size_t i_block_end = (i_block + row_block_size < row_end) ? i_block + row_block_size : row_end;     \
                                                                                                                       \
                /* Skip blocks entirely below diagonal. Blocks fully above the diagonal are still part of the upper    \
                 * triangle and must be computed. */                                                                   \
                if (i_block >= j_block_end) continue;                                                                  \
                                                                                                                       \
                for (nk_size_t i_macro = i_block; i_macro < i_block_end; i_macro += macro_tile_size) {                 \
                    /* Upper triangle: j_macro starts at max(i_macro, j_block) */                                      \
                    nk_size_t j_start = (i_macro > j_block) ? i_macro : j_block;                                       \
                    for (nk_size_t j_macro = j_start; j_macro < j_block_end; j_macro += macro_tile_size) {             \
                        nk_size_t macro_i_size = (i_macro + macro_tile_size <= i_block_end) ? macro_tile_size          \
                                                                                            : (i_block_end - i_macro); \
                        nk_size_t macro_j_size = (j_macro + macro_tile_size <= j_block_end) ? macro_tile_size          \
                                                                                            : (j_block_end - j_macro); \
                                                                                                                       \
                        /* Hoist pointer computation outside depth loop */                                             \
                        nk_##input_value_type##_t const *vector_base_ptrs_i[32];                                       \
                        nk_##input_value_type##_t const *vector_base_ptrs_j[32];                                       \
                        for (nk_size_t i = 0; i < macro_i_size; i++) {                                                 \
                            vector_base_ptrs_i[i] = vectors + (i_macro + i) * vectors_stride_values;                   \
                        }                                                                                              \
                        if (i_macro != j_macro || macro_i_size != macro_j_size) {                                      \
                            for (nk_size_t j = 0; j < macro_j_size; j++) {                                             \
                                vector_base_ptrs_j[j] = vectors + (j_macro + j) * vectors_stride_values;               \
                            }                                                                                          \
                        }                                                                                              \
                                                                                                                       \
                        if (i_macro == j_macro && macro_i_size == macro_j_size) {                                      \
                            /* Diagonal macro-tile: symmetric, upper triangle only */                                  \
                            nk_##api_name##_symmetric_diagonal_##input_type_name##_##isa_suffix##_(                    \
                                vector_base_ptrs_i, i_macro, macro_i_size, aligned_depth, remainder_depth,             \
                                remainder_dimensions, depth_step_values, dimensions_per_value, result,                 \
                                result_stride_values, finalizer_batch_size, depth);                                    \
                        }                                                                                              \
                        else {                                                                                         \
                            /* Off-diagonal macro-tile: full rectangle */                                              \
                            nk_##api_name##_symmetric_##input_type_name##_##isa_suffix##_offdiagonal##_(               \
                                vector_base_ptrs_i, vector_base_ptrs_j, i_macro, j_macro, macro_i_size, macro_j_size,  \
                                aligned_depth, remainder_depth, remainder_dimensions, depth_step_values,               \
                                dimensions_per_value, result, result_stride_values, finalizer_batch_size, depth);      \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }

/*  Keep the serial instantiations below actually scalar, regardless of build type.
 *  Without this, -O3 + LTO can vectorize or clone the serial kernels under AVX-512
 *  callers in dispatch_*.c, which wastes ~1 MB of binary and — more importantly —
 *  breaks the nk_*_serial-as-scalar-oracle contract that tests and the numerical-
 *  stability docs in this header rely on. */
#if defined(__clang__)
#pragma clang attribute push(__attribute__((noinline)), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-slp-vectorize", "no-ipa-cp-clone", "no-inline")
#endif

/*  Size bias for release. Gated on NDEBUG so Debug builds keep -O0 for stepping. */
#if defined(NDEBUG)
#if defined(_MSC_VER)
#pragma optimize("s", on)
#elif defined(__clang__)
#pragma clang attribute push(__attribute__((minsize)), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("Os")
#endif
#endif

/* F64 GEMM: depth_simd_dimensions=2 (2 f64s = 16 bytes) */
nk_define_cross_pack_size_(dots, f64, serial, f64, f64, /*norm_value_type=*/f64, /*depth_simd_dimensions=*/2,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, f64, serial, f64, f64, nk_b128_vec_t, nk_load_b128_serial_, nk_partial_load_b64x2_serial_,
                      nk_store_b128_serial_, nk_partial_store_b64x2_serial_, /*simd_width=*/2, /*norm_value_type=*/f64,
                      nk_dots_reduce_sumsq_f64_, /*depth_simd_dimensions=*/2, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, f64, serial, f64, f64, nk_b128_vec_t, nk_dot_f64x2_state_serial_t, nk_b256_vec_t,
                           nk_dot_f64x2_init_serial, nk_load_b128_serial_, nk_partial_load_b64x2_serial_,
                           nk_dot_f64x2_update_serial, nk_dot_f64x2_finalize_serial, nk_store_b256_serial_,
                           nk_partial_store_b64x4_serial_,
                           /*depth_simd_dimensions=*/2, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, f64, serial, f64, f64, f64, nk_b128_vec_t, nk_dot_f64x2_state_serial_t, nk_b256_vec_t,
                        nk_dot_f64x2_init_serial, nk_load_b128_serial_, nk_partial_load_b64x2_serial_,
                        nk_load_b128_serial_, nk_partial_load_b64x2_serial_, nk_dot_f64x2_update_serial,
                        nk_dot_f64x2_finalize_serial, nk_store_b256_serial_, nk_partial_store_b64x4_serial_,
                        /*depth_simd_dimensions=*/2, /*dimensions_per_value=*/1)

/* F32 GEMM: depth_simd_dimensions=4 (4 f32s = 16 bytes) */
nk_define_cross_pack_size_(dots, f32, serial, f32, f32, /*norm_value_type=*/f64, /*depth_simd_dimensions=*/4,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, f32, serial, f32, f32, nk_b128_vec_t, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                      nk_store_b128_serial_, nk_partial_store_b32x4_serial_, /*simd_width=*/4, /*norm_value_type=*/f64,
                      nk_dots_reduce_sumsq_f32_, /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, f32, serial, f32, f64, nk_b128_vec_t, nk_dot_f32x4_state_serial_t, nk_b256_vec_t,
                           nk_dot_f32x4_init_serial, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                           nk_dot_f32x4_update_serial, nk_dot_f32x4_finalize_serial, nk_store_b256_serial_,
                           nk_partial_store_b64x4_serial_,
                           /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, f32, serial, f32, f32, f64, nk_b128_vec_t, nk_dot_f32x4_state_serial_t, nk_b256_vec_t,
                        nk_dot_f32x4_init_serial, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                        nk_load_b128_serial_, nk_partial_load_b32x4_serial_, nk_dot_f32x4_update_serial,
                        nk_dot_f32x4_finalize_serial, nk_store_b256_serial_, nk_partial_store_b64x4_serial_,
                        /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)

/* F16 packed GEMM: pre-upcast B to f32 and process 4 logical dimensions per 128-bit step. */
nk_define_cross_pack_size_(dots, f16, serial, f16, f32, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/4,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, f16, serial, f16, f32, nk_b128_vec_t, nk_load_f16x4_to_f32x4_serial_,
                      nk_partial_load_f16x4_to_f32x4_serial_, nk_store_b128_serial_, nk_partial_store_b32x4_serial_,
                      /*simd_width=*/4, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_f16_,
                      /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, f16, serial, f16, f32, nk_b128_vec_t, nk_dot_f16x8_state_serial_t, nk_b128_vec_t,
                           nk_dot_f16x8_init_serial, nk_load_b128_serial_, nk_partial_load_b16x8_serial_,
                           nk_dot_f16x8_update_serial, nk_dot_f16x8_finalize_serial, nk_store_b128_serial_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, f16, serial, f16, f32, f32, nk_b128_vec_t, nk_dot_through_f32x4_state_serial_t,
                        nk_b128_vec_t, nk_dot_through_f32x4_init_serial, nk_load_f16x4_to_f32x4_serial_,
                        nk_partial_load_f16x4_to_f32x4_serial_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                        nk_dot_through_f32x4_update_serial, nk_dot_through_f32x4_finalize_serial, nk_store_b128_serial_,
                        nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)

/* BF16 GEMM: depth_simd_dimensions=8 (8 bf16s = 16 bytes), F32 accumulator */
nk_define_cross_pack_size_(dots, bf16, serial, bf16, bf16, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/8,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, bf16, serial, bf16, bf16, nk_b128_vec_t, nk_load_b128_serial_,
                      nk_partial_load_b16x8_serial_, nk_store_b128_serial_, nk_partial_store_b16x8_serial_,
                      /*simd_width=*/8, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_bf16_,
                      /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, bf16, serial, bf16, f32, nk_b128_vec_t, nk_dot_bf16x8_state_serial_t, nk_b128_vec_t,
                           nk_dot_bf16x8_init_serial, nk_load_b128_serial_, nk_partial_load_b16x8_serial_,
                           nk_dot_bf16x8_update_serial, nk_dot_bf16x8_finalize_serial, nk_store_b128_serial_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, bf16, serial, bf16, bf16, f32, nk_b128_vec_t, nk_dot_bf16x8_state_serial_t, nk_b128_vec_t,
                        nk_dot_bf16x8_init_serial, nk_load_b128_serial_, nk_partial_load_b16x8_serial_,
                        nk_load_b128_serial_, nk_partial_load_b16x8_serial_, nk_dot_bf16x8_update_serial,
                        nk_dot_bf16x8_finalize_serial, nk_store_b128_serial_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)

/* I8 GEMM: depth_simd_dimensions=16 (16 i8s = 16 bytes), I32 accumulator */
nk_define_cross_pack_size_(dots, i8, serial, i8, i8, /*norm_value_type=*/u32, /*depth_simd_dimensions=*/16,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, i8, serial, i8, i8, nk_b128_vec_t, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                      nk_store_b128_serial_, nk_partial_store_b8x16_serial_, /*simd_width=*/16,
                      /*norm_value_type=*/u32, nk_dots_reduce_sumsq_i8_, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, i8, serial, i8, i32, nk_b128_vec_t, nk_dot_i8x16_state_serial_t, nk_b128_vec_t,
                           nk_dot_i8x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                           nk_dot_i8x16_update_serial, nk_dot_i8x16_finalize_serial, nk_store_b128_serial_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, i8, serial, i8, i8, i32, nk_b128_vec_t, nk_dot_i8x16_state_serial_t, nk_b128_vec_t,
                        nk_dot_i8x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                        nk_load_b128_serial_, nk_partial_load_b8x16_serial_, nk_dot_i8x16_update_serial,
                        nk_dot_i8x16_finalize_serial, nk_store_b128_serial_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* U8 GEMM: depth_simd_dimensions=16 (16 u8s = 16 bytes), U32 accumulator */
nk_define_cross_pack_size_(dots, u8, serial, u8, u8, /*norm_value_type=*/u32, /*depth_simd_dimensions=*/16,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, u8, serial, u8, u8, nk_b128_vec_t, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                      nk_store_b128_serial_, nk_partial_store_b8x16_serial_, /*simd_width=*/16,
                      /*norm_value_type=*/u32, nk_dots_reduce_sumsq_u8_, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, u8, serial, u8, u32, nk_b128_vec_t, nk_dot_u8x16_state_serial_t, nk_b128_vec_t,
                           nk_dot_u8x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                           nk_dot_u8x16_update_serial, nk_dot_u8x16_finalize_serial, nk_store_b128_serial_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, u8, serial, u8, u8, u32, nk_b128_vec_t, nk_dot_u8x16_state_serial_t, nk_b128_vec_t,
                        nk_dot_u8x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                        nk_load_b128_serial_, nk_partial_load_b8x16_serial_, nk_dot_u8x16_update_serial,
                        nk_dot_u8x16_finalize_serial, nk_store_b128_serial_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* E4M3 GEMM: depth_simd_dimensions=16 (16 e4m3s = 16 bytes), F32 accumulator */
nk_define_cross_pack_size_(dots, e4m3, serial, e4m3, e4m3, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/16,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, e4m3, serial, e4m3, e4m3, nk_b128_vec_t, nk_load_b128_serial_,
                      nk_partial_load_b8x16_serial_, nk_store_b128_serial_, nk_partial_store_b8x16_serial_,
                      /*simd_width=*/16, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e4m3_,
                      /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e4m3, serial, e4m3, f32, nk_b128_vec_t, nk_dot_e4m3x16_state_serial_t, nk_b128_vec_t,
                           nk_dot_e4m3x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                           nk_dot_e4m3x16_update_serial, nk_dot_e4m3x16_finalize_serial, nk_store_b128_serial_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e4m3, serial, e4m3, e4m3, f32, nk_b128_vec_t, nk_dot_e4m3x16_state_serial_t,
                        nk_b128_vec_t, nk_dot_e4m3x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                        nk_load_b128_serial_, nk_partial_load_b8x16_serial_, nk_dot_e4m3x16_update_serial,
                        nk_dot_e4m3x16_finalize_serial, nk_store_b128_serial_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* E5M2 GEMM: depth_simd_dimensions=16 (16 e5m2s = 16 bytes), F32 accumulator */
nk_define_cross_pack_size_(dots, e5m2, serial, e5m2, e5m2, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/16,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, e5m2, serial, e5m2, e5m2, nk_b128_vec_t, nk_load_b128_serial_,
                      nk_partial_load_b8x16_serial_, nk_store_b128_serial_, nk_partial_store_b8x16_serial_,
                      /*simd_width=*/16, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e5m2_,
                      /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e5m2, serial, e5m2, f32, nk_b128_vec_t, nk_dot_e5m2x16_state_serial_t, nk_b128_vec_t,
                           nk_dot_e5m2x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                           nk_dot_e5m2x16_update_serial, nk_dot_e5m2x16_finalize_serial, nk_store_b128_serial_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e5m2, serial, e5m2, e5m2, f32, nk_b128_vec_t, nk_dot_e5m2x16_state_serial_t,
                        nk_b128_vec_t, nk_dot_e5m2x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                        nk_load_b128_serial_, nk_partial_load_b8x16_serial_, nk_dot_e5m2x16_update_serial,
                        nk_dot_e5m2x16_finalize_serial, nk_store_b128_serial_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* E2M3 GEMM: depth_simd_dimensions=16 (16 e2m3s = 16 bytes), F32 accumulator */
nk_define_cross_pack_size_(dots, e2m3, serial, e2m3, e2m3, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/16,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, e2m3, serial, e2m3, e2m3, nk_b128_vec_t, nk_load_b128_serial_,
                      nk_partial_load_b8x16_serial_, nk_store_b128_serial_, nk_partial_store_b8x16_serial_,
                      /*simd_width=*/16, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e2m3_,
                      /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e2m3, serial, e2m3, f32, nk_b128_vec_t, nk_dot_e2m3x16_state_serial_t, nk_b128_vec_t,
                           nk_dot_e2m3x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                           nk_dot_e2m3x16_update_serial, nk_dot_e2m3x16_finalize_serial, nk_store_b128_serial_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e2m3, serial, e2m3, e2m3, f32, nk_b128_vec_t, nk_dot_e2m3x16_state_serial_t,
                        nk_b128_vec_t, nk_dot_e2m3x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                        nk_load_b128_serial_, nk_partial_load_b8x16_serial_, nk_dot_e2m3x16_update_serial,
                        nk_dot_e2m3x16_finalize_serial, nk_store_b128_serial_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* E3M2 GEMM: depth_simd_dimensions=16 (16 e3m2s = 16 bytes), F32 accumulator */
nk_define_cross_pack_size_(dots, e3m2, serial, e3m2, e3m2, /*norm_value_type=*/f32, /*depth_simd_dimensions=*/16,
                           /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, e3m2, serial, e3m2, e3m2, nk_b128_vec_t, nk_load_b128_serial_,
                      nk_partial_load_b8x16_serial_, nk_store_b128_serial_, nk_partial_store_b8x16_serial_,
                      /*simd_width=*/16, /*norm_value_type=*/f32, nk_dots_reduce_sumsq_e3m2_,
                      /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e3m2, serial, e3m2, f32, nk_b128_vec_t, nk_dot_e3m2x16_state_serial_t, nk_b128_vec_t,
                           nk_dot_e3m2x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                           nk_dot_e3m2x16_update_serial, nk_dot_e3m2x16_finalize_serial, nk_store_b128_serial_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e3m2, serial, e3m2, e3m2, f32, nk_b128_vec_t, nk_dot_e3m2x16_state_serial_t,
                        nk_b128_vec_t, nk_dot_e3m2x16_init_serial, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                        nk_load_b128_serial_, nk_partial_load_b8x16_serial_, nk_dot_e3m2x16_update_serial,
                        nk_dot_e3m2x16_finalize_serial, nk_store_b128_serial_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* U4 GEMM: u4x2 for both A and B */
nk_define_cross_pack_size_(dots, u4, serial, u4x2, u4x2, /*norm_value_type=*/u32, /*depth_simd_dimensions=*/16,
                           /*dimensions_per_value=*/2)
nk_define_cross_pack_(dots, u4, serial, u4x2, u4x2, nk_b128_vec_t, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                      nk_store_b128_serial_, nk_partial_store_b8x16_serial_, /*simd_width=*/16,
                      /*norm_value_type=*/u32, nk_dots_reduce_sumsq_u4_, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/2)
nk_define_cross_symmetric_(dots, u4, serial, u4x2, u32, nk_b64_vec_t, nk_dot_u4x16_state_serial_t, nk_b128_vec_t,
                           nk_dot_u4x16_init_serial, nk_load_b64_serial_, nk_partial_load_b4x16_serial_,
                           nk_dot_u4x16_update_serial, nk_dot_u4x16_finalize_serial, nk_store_b128_serial_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/2)
nk_define_cross_packed_(dots, u4, serial, u4x2, u4x2, u32, nk_b64_vec_t, nk_dot_u4x16_state_serial_t, nk_b128_vec_t,
                        nk_dot_u4x16_init_serial, nk_load_b64_serial_, nk_partial_load_b4x16_serial_,
                        nk_load_b64_serial_, nk_partial_load_b4x16_serial_, nk_dot_u4x16_update_serial,
                        nk_dot_u4x16_finalize_serial, nk_store_b128_serial_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/2)

/* I4 GEMM: i4x2 for both A and B */
nk_define_cross_pack_size_(dots, i4, serial, i4x2, i4x2, /*norm_value_type=*/u32, /*depth_simd_dimensions=*/16,
                           /*dimensions_per_value=*/2)
nk_define_cross_pack_(dots, i4, serial, i4x2, i4x2, nk_b128_vec_t, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                      nk_store_b128_serial_, nk_partial_store_b8x16_serial_, /*simd_width=*/16,
                      /*norm_value_type=*/u32, nk_dots_reduce_sumsq_i4_, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/2)
nk_define_cross_symmetric_(dots, i4, serial, i4x2, i32, nk_b64_vec_t, nk_dot_i4x16_state_serial_t, nk_b128_vec_t,
                           nk_dot_i4x16_init_serial, nk_load_b64_serial_, nk_partial_load_b4x16_serial_,
                           nk_dot_i4x16_update_serial, nk_dot_i4x16_finalize_serial, nk_store_b128_serial_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/2)
nk_define_cross_packed_(dots, i4, serial, i4x2, i4x2, i32, nk_b64_vec_t, nk_dot_i4x16_state_serial_t, nk_b128_vec_t,
                        nk_dot_i4x16_init_serial, nk_load_b64_serial_, nk_partial_load_b4x16_serial_,
                        nk_load_b64_serial_, nk_partial_load_b4x16_serial_, nk_dot_i4x16_update_serial,
                        nk_dot_i4x16_finalize_serial, nk_store_b128_serial_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/2)

/* U1 GEMM: u1x8 for both A and B */
nk_define_cross_pack_size_(dots, u1, serial, u1x8, u1x8, /*norm_value_type=*/u32, /*depth_simd_dimensions=*/128,
                           /*dimensions_per_value=*/8)
nk_define_cross_pack_(dots, u1, serial, u1x8, u1x8, nk_b128_vec_t, nk_load_b128_serial_, nk_partial_load_b8x16_serial_,
                      nk_store_b128_serial_, nk_partial_store_b8x16_serial_, /*simd_width=*/16,
                      /*norm_value_type=*/u32, nk_dots_reduce_sum_u1_, /*depth_simd_dimensions=*/128,
                      /*dimensions_per_value=*/8)
nk_define_cross_symmetric_(dots, u1, serial, u1x8, u32, nk_b128_vec_t, nk_dot_u1x128_state_serial_t, nk_b128_vec_t,
                           nk_dot_u1x128_init_serial, nk_load_b128_serial_, nk_partial_load_b1x128_serial_,
                           nk_dot_u1x128_update_serial, nk_dot_u1x128_finalize_serial, nk_store_b128_serial_,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/8)
nk_define_cross_packed_(dots, u1, serial, u1x8, u1x8, u32, nk_b128_vec_t, nk_dot_u1x128_state_serial_t, nk_b128_vec_t,
                        nk_dot_u1x128_init_serial, nk_load_b128_serial_, nk_partial_load_b1x128_serial_,
                        nk_load_b128_serial_, nk_partial_load_b1x128_serial_, nk_dot_u1x128_update_serial,
                        nk_dot_u1x128_finalize_serial, nk_store_b128_serial_, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/8)

#if defined(NDEBUG)
#if defined(_MSC_VER)
#pragma optimize("", on)
#elif defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

/*  BF16 compact: truncate F32 → BF16 in-place.
 *  Reads F32 matrix with c_stride_in_bytes, writes BF16 tightly packed (stride_in_bytes = column_count × sizeof(bf16)).
 */
NK_PUBLIC void nk_dots_compact_bf16_serial(void *c, nk_size_t row_count, nk_size_t column_count,
                                           nk_size_t c_stride_in_bytes) {
    nk_size_t const c_stride_in_values = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_f32_t const *c_f32 = (nk_f32_t const *)c;
    nk_bf16_t *c_bf16 = (nk_bf16_t *)c;

    for (nk_size_t row_index = 0; row_index < row_count; row_index++) {
        nk_f32_t const *source_row = c_f32 + row_index * c_stride_in_values;
        nk_bf16_t *destination_row = c_bf16 + row_index * column_count;
        for (nk_size_t column_index = 0; column_index < column_count; column_index++) {
            nk_f32_to_bf16_serial(source_row + column_index, destination_row + column_index);
        }
    }
}

/*  I8 compact: re-normalize I32 → I8 using precomputed squared norms.
 *  Formula: c_i8[i][j] = c_i32[i][j] × 127 / sqrt(a_norm[i] × b_norm[j])
 *  Output is tightly packed (stride_in_bytes = column_count × sizeof(i8)).
 */
NK_PUBLIC void nk_dots_compact_i8_serial(void *c, nk_size_t row_count, nk_size_t column_count,
                                         nk_size_t c_stride_in_bytes, nk_i32_t const *a_squared_norms,
                                         nk_i32_t const *b_squared_norms) {
    nk_size_t const c_stride_in_values = c_stride_in_bytes / sizeof(nk_i32_t);
    nk_i32_t const *c_i32 = (nk_i32_t const *)c;
    nk_i8_t *c_i8 = (nk_i8_t *)c;

    for (nk_size_t row_index = 0; row_index < row_count; row_index++) {
        nk_i32_t const *source_row = c_i32 + row_index * c_stride_in_values;
        nk_i8_t *destination_row = c_i8 + row_index * column_count;

        nk_f32_t a_norm_f32_value = (nk_f32_t)a_squared_norms[row_index];
        nk_f32_t a_rsqrt_value = (a_norm_f32_value > 0) ? (1.0f / nk_f32_sqrt_serial(a_norm_f32_value)) : 0.0f;

        for (nk_size_t column_index = 0; column_index < column_count; column_index++) {
            nk_f32_t b_norm_f32_value = (nk_f32_t)b_squared_norms[column_index];
            nk_f32_t b_rsqrt_value = (b_norm_f32_value > 0) ? (1.0f / nk_f32_sqrt_serial(b_norm_f32_value)) : 0.0f;

            nk_f32_t normalized_value = (nk_f32_t)source_row[column_index] * 127.0f * a_rsqrt_value * b_rsqrt_value;
            nk_i32_t clamped_value = (nk_i32_t)normalized_value;
            if (clamped_value < -128) clamped_value = -128;
            if (clamped_value > 127) clamped_value = 127;
            destination_row[column_index] = (nk_i8_t)clamped_value;
        }
    }
}

#define nk_define_cross_normalized_packed_(metric_name, input_type_name, isa_suffix, input_value_type,                \
                                           packed_value_type, dot_result_type, norm_value_type, final_result_type,    \
                                           vec_type, dots_packed_fn, from_dot_fn, compute_norm_fn, load_fn,           \
                                           partial_load_fn, store_fn, partial_store_fn, dimensions_per_value)         \
    NK_PUBLIC void nk_##metric_name##s_packed_##input_type_name##_##isa_suffix(                                       \
        nk_##input_value_type##_t const *a_matrix, void const *b_packed_buffer, nk_##final_result_type##_t *c_matrix, \
        nk_size_t row_count, nk_size_t column_count, nk_size_t depth, nk_size_t a_stride_in_bytes,                    \
        nk_size_t c_stride_in_bytes) {                                                                                \
                                                                                                                      \
        dots_packed_fn(a_matrix, b_packed_buffer, (nk_##dot_result_type##_t *)c_matrix, row_count, column_count,      \
                       depth, a_stride_in_bytes, c_stride_in_bytes);                                                  \
                                                                                                                      \
        nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed_buffer;     \
        nk_size_t depth_padded = header->depth_padded_values;                                                         \
        nk_##norm_value_type##_t const *b_norms =                                                                     \
            (nk_##norm_value_type##_t const *)((char const *)b_packed_buffer +                                        \
                                               sizeof(nk_cross_packed_buffer_header_t) +                              \
                                               column_count * depth_padded * sizeof(nk_##packed_value_type##_t));     \
                                                                                                                      \
        for (nk_size_t row_index = 0; row_index < row_count; ++row_index) {                                           \
            nk_##input_value_type##_t const *a_row =                                                                  \
                (nk_##input_value_type##_t const *)((char const *)a_matrix + row_index * a_stride_in_bytes);          \
            nk_##dot_result_type##_t query_norm = compute_norm_fn(a_row, depth);                                      \
            nk_##dot_result_type##_t *r_row_dots = (nk_##dot_result_type##_t *)((char *)c_matrix +                    \
                                                                                row_index * c_stride_in_bytes);       \
            nk_##final_result_type##_t *r_row_out = (nk_##final_result_type##_t *)((char *)c_matrix +                 \
                                                                                   row_index * c_stride_in_bytes);    \
                                                                                                                      \
            nk_size_t column_index = 0;                                                                               \
            for (; column_index + 4 <= column_count; column_index += 4) {                                             \
                vec_type dots_vec, norms_vec, results_vec;                                                            \
                load_fn(r_row_dots + column_index, &dots_vec);                                                        \
                load_fn(b_norms + column_index, &norms_vec);                                                          \
                from_dot_fn(&dots_vec, query_norm, &norms_vec, &results_vec);                                         \
                store_fn(&results_vec, r_row_out + column_index);                                                     \
            }                                                                                                         \
            if (column_index < column_count) {                                                                        \
                vec_type dots_vec = {0}, norms_vec = {0}, results_vec;                                                \
                partial_load_fn(r_row_dots + column_index, &dots_vec, column_count - column_index);                   \
                partial_load_fn(b_norms + column_index, &norms_vec, column_count - column_index);                     \
                from_dot_fn(&dots_vec, query_norm, &norms_vec, &results_vec);                                         \
                partial_store_fn(&results_vec, r_row_out + column_index, column_count - column_index);                \
            }                                                                                                         \
        }                                                                                                             \
    }

#define nk_define_cross_normalized_symmetric_(metric_name, input_type_name, isa_suffix, input_value_type,              \
                                              dot_result_type, norm_value_type, final_result_type, vec_type,           \
                                              dots_symmetric_fn, from_dot_fn, compute_norm_fn, load_fn,                \
                                              partial_load_fn, store_fn, partial_store_fn, dimensions_per_value)       \
    NK_PUBLIC void nk_##metric_name##s_symmetric_##input_type_name##_##isa_suffix(                                     \
        nk_##input_value_type##_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_in_bytes, \
        nk_##final_result_type##_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start,                     \
        nk_size_t row_count) {                                                                                         \
                                                                                                                       \
        dots_symmetric_fn(vectors, vectors_count, depth, stride_in_bytes, (nk_##dot_result_type##_t *)result,          \
                          result_stride_in_bytes, row_start, row_count);                                               \
                                                                                                                       \
        /* Phase 1 — cache row norms in the result diagonal (O(row_count) calls) */                                    \
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {                        \
            nk_##input_value_type##_t const *row_vector =                                                              \
                (nk_##input_value_type##_t const *)((char const *)vectors + row_index * stride_in_bytes);              \
            nk_##norm_value_type##_t *row_diag = (nk_##norm_value_type##_t *)((char *)result +                         \
                                                                              row_index * result_stride_in_bytes);     \
            row_diag[row_index] = compute_norm_fn(row_vector, depth);                                                  \
        }                                                                                                              \
                                                                                                                       \
        /* Phase 2 — column-first post-processing with 256-element norm cache */                                       \
        nk_##norm_value_type##_t column_norms[256];                                                                    \
        for (nk_size_t column_chunk_start = 0; column_chunk_start < vectors_count; column_chunk_start += 256) {        \
            nk_size_t column_chunk_end = column_chunk_start + 256 < vectors_count ? column_chunk_start + 256           \
                                                                                  : vectors_count;                     \
                                                                                                                       \
            /* Pre-compute norms for this column chunk — each column visited exactly once */                           \
            for (nk_size_t col = column_chunk_start; col < column_chunk_end; ++col) {                                  \
                nk_##input_value_type##_t const *column_vector =                                                       \
                    (nk_##input_value_type##_t const *)((char const *)vectors + col * stride_in_bytes);                \
                column_norms[col - column_chunk_start] = compute_norm_fn(column_vector, depth);                        \
            }                                                                                                          \
                                                                                                                       \
            /* Sweep assigned rows against this column chunk */                                                        \
            for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {                    \
                nk_size_t j_start = row_index + 1 > column_chunk_start ? row_index + 1 : column_chunk_start;           \
                if (j_start >= column_chunk_end) continue;                                                             \
                char *row_ptr = (char *)result + row_index * result_stride_in_bytes;                                   \
                nk_##norm_value_type##_t sumsq_i = ((nk_##norm_value_type##_t *)row_ptr)[row_index];                   \
                nk_##dot_result_type##_t *r_dots = (nk_##dot_result_type##_t *)row_ptr;                                \
                nk_##final_result_type##_t *r_out = (nk_##final_result_type##_t *)row_ptr;                             \
                                                                                                                       \
                /* 4-wide vectorized loop */                                                                           \
                nk_size_t j = j_start;                                                                                 \
                for (; j + 4 <= column_chunk_end; j += 4) {                                                            \
                    vec_type target_norms_vec;                                                                         \
                    load_fn(&column_norms[j - column_chunk_start], &target_norms_vec);                                 \
                    vec_type dots_vec, results_vec;                                                                    \
                    load_fn(r_dots + j, &dots_vec);                                                                    \
                    from_dot_fn(&dots_vec, sumsq_i, &target_norms_vec, &results_vec);                                  \
                    store_fn(&results_vec, r_out + j);                                                                 \
                }                                                                                                      \
                /* Remainder */                                                                                        \
                if (j < column_chunk_end) {                                                                            \
                    vec_type dots_vec = {0}, norms_vec = {0}, results_vec;                                             \
                    partial_load_fn(r_dots + j, &dots_vec, column_chunk_end - j);                                      \
                    partial_load_fn(&column_norms[j - column_chunk_start], &norms_vec, column_chunk_end - j);          \
                    from_dot_fn(&dots_vec, sumsq_i, &norms_vec, &results_vec);                                         \
                    partial_store_fn(&results_vec, r_out + j, column_chunk_end - j);                                   \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
                                                                                                                       \
        /* Phase 3 — zero diagonals */                                                                                 \
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {                        \
            nk_##final_result_type##_t *r_out = (nk_##final_result_type##_t *)((char *)result +                        \
                                                                               row_index * result_stride_in_bytes);    \
            r_out[row_index] = 0;                                                                                      \
        }                                                                                                              \
    }

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#endif // NK_DOTS_SERIAL_H
