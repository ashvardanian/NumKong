/**
 *  @brief SIMD-accelerated Batched Dot Products.
 *  @file include/numkong/dots.h
 *  @author Ash Vardanian
 *  @date September 14, 2024
 *
 *  Implements batch dot-product kernels computing C[m × n] = A[m × k] × B[n × k]ᵀ
 *  with row-major A and arbitrary B, optimized for ML inference and similarity workloads.
 *
 *  Primary Use Cases (1-to-N focus):
 *
 *  - k-NN search: ‖a-b‖² = ‖a‖² + ‖b‖² - 2(a × b)
 *  - Cosine similarity: (a × b) / (‖a‖ × ‖b‖)
 *  - Sparse attention patterns
 *  - Embedding similarity matrices
 *  - k-means clustering, DBSCAN, hierarchical clustering
 *
 *  It implements several operations:
 *
 *  - "dots_packed" - computing dot-products where the B matrix is pre-packed into optimal form
 *  - "dots_packed_size" - which estimates the memory requirements for external `malloc`
 *  - "dots_pack" - to perform the pre-processing
 *  - "dots_compact" - optional helpers to normalize or downcast into original precision
 *  - "dots_symmetric" - for A × Aᵀ Gram matrix multiplication
 *
 *  If the original "dots_packed" is analogous to "GEMM" (General Matrix Multiplication) in BLAS,
 *  the "dots_symmetric" is similar to the "SYRK" (the Symmetric rank-k update of a matrix).
 *
 *  For dtypes:
 *
 *  - f64: 64-bit IEEE floating point numbers → 64-bit floats
 *  - f32: 32-bit IEEE floating point numbers → 64-bit floats
 *  - f16: 16-bit IEEE floating point numbers → 32-bit floats
 *  - bf16: 16-bit brain floating point numbers → 32-bit floats
 *  - e4m3: 8-bit e4m3 floating point numbers → 32-bit floats
 *  - e5m2: 8-bit e5m2 floating point numbers → 32-bit floats
 *  - e2m3: 8-bit e2m3 floating point numbers (MX) → 32-bit floats
 *  - e3m2: 8-bit e3m2 floating point numbers (MX) → 32-bit floats
 *  - i8: 8-bit signed integers → 32-bit signed integers
 *  - u8: 8-bit unsigned integers → 32-bit unsigned integers
 *  - i4: 4-bit signed integers (packed pairs) → 32-bit signed integers
 *  - u4: 4-bit unsigned integers (packed pairs) → 32-bit unsigned integers
 *  - u1: 1-bit binary (packed octets) → 32-bit unsigned integers
 *
 *  For hardware architectures:
 *
 *  - Arm: NEON, NEON+HALF, NEON+FHM, NEON+BF16, NEON+SDOT, SVE, SME, SME+F64, SME+BI32
 *  - x86: Haswell, Skylake, Ice Lake, Genoa, Sapphire Rapids (AMX), Sierra Forest
 *  - RISC-V: RVV
 *
 *  @section numerical_stability Numerical Stability
 *
 *  - f64: Dot2 (Ogita-Rump-Oishi) on the accurate backends, otherwise native f64 FMA accumulation.
 *  - f32: public outputs widen to f64. Packed and symmetric kernels keep payloads narrow but widen accumulation.
 *  - bf16/f16: f32 accumulation. VDPBF16PS on Genoa does bf16×bf16→f32 natively.
 *  - e2m3/e3m2: f16 intermediate with flush to f32 every 128 elements (Sapphire).
 *  - i8: i32 accumulation. AMX TDPBSSD gives i8×i8→i32 tiles. Overflows at k > ~131K.
 *  - u1: Popcount, exact.
 *
 *  @section memory_layout Memory Layout and Transpose Semantics
 *
 *  All matrices use row-major storage. Column-major is NOT supported.
 *  The kernel computes C = A × Bᵀ where:
 *
 *  - A is (m × k): m rows, k columns, stride = a_stride bytes between rows
 *  - B is (n × k): n rows, k columns, stride = b_stride bytes between rows
 *  - C is (m × n): m rows, n columns, stride = c_stride bytes between rows
 *
 *  This means C[i,j] = dot(row i of A, row j of B) = Σₗ A[i,l] × B[j,l].
 *
 *  All strides are in bytes.
 *
 *  To compute standard A × B (where B is k × n), pass Bᵀ to the packing function:
 *
 *  @code{.c}
 *  // Standard matmul: C[m × n] = A[m × k] × B[k × n]
 *  // B is stored row-major as k rows of n elements
 *  // Treat it as Bᵀ: n rows of k elements with stride = sizeof(element)
 *  nk_dots_pack_bf16(b, width, depth, sizeof(nk_bf16_t), b_packed);
 *  nk_dots_packed_bf16(a, b_packed, c, height, width, depth, a_stride, c_stride);
 *  // Result: C = A × (Bᵀ)ᵀ = A × B
 *  @endcode
 *
 *  @section two_phase_api Two-Phase API for Static Weights
 *
 *  Matrix multiplication hardware (AMX, SME) requires specific data layouts that differ
 *  from standard row-major ordering. Since one matrix (typically weights in neural networks)
 *  is often static, we provide a two-phase API: pack once, multiply many times.
 *
 *  @code{.c}
 *  // Similarity search: C[m × n] = queries[m × k] × database[n × k]ᵀ
 *  // Both matrices stored row-major, each row is one vector of dimension k
 *  nk_size_t packed_bytes = nk_dots_packed_size_bf16(width, depth);
 *  void *b_packed = malloc(packed_bytes);
 *  nk_dots_pack_bf16(database, width, depth, depth * sizeof(nk_bf16_t), b_packed);
 *  nk_dots_packed_bf16(queries, b_packed, c, height, width, depth, ...);
 *  // Result: C[i,j] = dot(query i, database vector j)
 *  @endcode
 *
 *  The packed format is opaque and backend-specific. AMX expects (16 × 32) tiles with interleaved
 *  pairs, while NEON/SVE use arrangements optimized for their vector lengths.
 *
 *  @section why_int8 Why INT8 and Not UINT8?
 *
 *  Unsigned 8-bit integers were considered but deprioritized. The industry has converged on
 *  signed INT8 as the standard for quantized inference:
 *
 *      Framework           Default     Notes
 *      PyTorch             qint8       New X86 backend uses INT8 via oneDNN
 *      TensorFlow Lite     int8        Actively removing UINT8 support
 *      ONNX Runtime        S8S8        "Should be the first choice"
 *      TensorRT            INT8        Symmetric [-128,127], no UINT8 option
 *      ARM CMSIS-NN        int8        Follows TFLite INT8 spec exactly
 *
 *  @section why_no_scaling Why No Alpha/Beta Scaling?
 *
 *  BLAS-style `C = α × A × B + β × C` scaling was considered but omitted. While useful for scientific
 *  computing (iterative solvers, matrix factorizations), it's rarely used in ML inference where
 *  frameworks handle such operations via graph fusion. More importantly, on chips with separate
 *  physical registers for vector and matrix operations (like AMX), moving scalars between register
 *  files adds transfer latency that negates any benefit.
 *
 *  @section why_no_pad Why Not Pad N Dimension to Eliminate Edge Handling?
 *
 *  Padding N to a tile-aligned boundary (multiple of 16) during packing was considered to eliminate
 *  the separate AVX-512 edge kernel for N remainder rows. While this sounds simpler ("pure AMX"),
 *  it actually increases code size by ~125 lines because:
 *
 *  - The AVX-512 edge fallback is compact (~40 lines) and handles both full-M × N-edge and
 *    M-edge × N-edge cases through a single reusable function
 *  - Replacing it with "AMX + masked stores" requires verbose tile handling code duplicated
 *    across all 4 multiply functions (aligned/misaligned × BF16/I8)
 *  - Each function needs a new "trailing N tile for full M blocks" section (~50 lines each)
 *
 *  The current hybrid layout (AMX for full tiles, AVX-512 for edges) is more maintainable despite
 *  being conceptually less uniform. Memory overhead of the edge region is negligible (<2% worst case).
 *
 *  @section x86_instructions Relevant x86 Instructions
 *
 *  Low-precision matmul relies on VPMADD* (AVX2), VNNI dot-products, and BF16 dot-products
 *  on AVX-512. Zen4 improves throughput by dual-issuing many integer ops on FP ports.
 *
 *      Intrinsic             Instruction                   Haswell   Genoa
 *      _mm256_maddubs_epi16  VPMADDUBSW (YMM, YMM, YMM)    5cy @ p0  3cy @ p01
 *      _mm256_madd_epi16     VPMADDWD (YMM, YMM, YMM)      5cy @ p0  3cy @ p01
 *      _mm256_dpbusd_epi32   VPDPBUSD (YMM, K, YMM, YMM)   n/a       4cy @ p01
 *      _mm256_dpwssds_epi32  VPDPWSSDS (YMM, K, YMM, YMM)  n/a       4cy @ p01
 *      _mm256_dpbf16_ps      VDPBF16PS (YMM, YMM, YMM)     n/a       6cy @ p01
 *
 *  AMX tile ops (TDPBF16PS/TDPBUSD/TDPBSSD) are not covered by the uops.info 2022 dataset.
 *
 *  @section references References
 *
 *  - x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
 *  - Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *  - uops.info: https://uops.info/
 *  - Matrix Multiplication in 40 lines: https://en.algorithmica.org/hpc/algorithms/matmul/
 *  - LLaMA CPU optimization: https://justine.lol/matmul/
 *  - SME outer-product notes: https://github.com/tzakharko/m4-sme-exploration
 *
 */
#ifndef NK_DOTS_H
#define NK_DOTS_H

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Returns packed buffer size in bytes for second multiplier matrix (B).
 *  @param[in] width The number of rows in B (output columns).
 *  @param[in] depth The number of columns in B.
 *  @note The packed layout is backend-specific and must be produced by the matching pack function.
 */
NK_DYNAMIC nk_size_t nk_dots_packed_size_bf16(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_packed_size_bf16 */
NK_DYNAMIC nk_size_t nk_dots_packed_size_f16(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_packed_size_bf16 */
NK_DYNAMIC nk_size_t nk_dots_packed_size_e4m3(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_packed_size_bf16 */
NK_DYNAMIC nk_size_t nk_dots_packed_size_e5m2(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_packed_size_bf16 */
NK_DYNAMIC nk_size_t nk_dots_packed_size_e2m3(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_packed_size_bf16 */
NK_DYNAMIC nk_size_t nk_dots_packed_size_e3m2(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_packed_size_bf16 */
NK_DYNAMIC nk_size_t nk_dots_packed_size_f32(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_packed_size_bf16 */
NK_DYNAMIC nk_size_t nk_dots_packed_size_f64(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_packed_size_bf16 */
NK_DYNAMIC nk_size_t nk_dots_packed_size_i8(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_packed_size_bf16 */
NK_DYNAMIC nk_size_t nk_dots_packed_size_u8(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_packed_size_bf16 */
NK_DYNAMIC nk_size_t nk_dots_packed_size_i4(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_packed_size_bf16 */
NK_DYNAMIC nk_size_t nk_dots_packed_size_u4(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_packed_size_bf16 */
NK_DYNAMIC nk_size_t nk_dots_packed_size_u1(nk_size_t width, nk_size_t depth);

/**
 *  @brief Packs the second multiplier (B) matrix into a backend-specific layout.
 *  @param[in] b The input B matrix in row-major order.
 *  @param[in] width The number of rows in B (output columns).
 *  @param[in] depth The number of columns in B.
 *  @param[in] b_stride The row stride in bytes for B.
 *  @param[out] b_packed The output packed buffer from nk_dots_packed_size_bf16.
 */
NK_DYNAMIC void nk_dots_pack_bf16(nk_bf16_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                  void *b_packed);
/** @copydoc nk_dots_pack_bf16 */
NK_DYNAMIC void nk_dots_pack_f16(nk_f16_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                 void *b_packed);
/** @copydoc nk_dots_pack_bf16 */
NK_DYNAMIC void nk_dots_pack_e4m3(nk_e4m3_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                  void *b_packed);
/** @copydoc nk_dots_pack_bf16 */
NK_DYNAMIC void nk_dots_pack_e5m2(nk_e5m2_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                  void *b_packed);
/** @copydoc nk_dots_pack_bf16 */
NK_DYNAMIC void nk_dots_pack_e2m3(nk_e2m3_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                  void *b_packed);
/** @copydoc nk_dots_pack_bf16 */
NK_DYNAMIC void nk_dots_pack_e3m2(nk_e3m2_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                  void *b_packed);
/** @copydoc nk_dots_pack_bf16 */
NK_DYNAMIC void nk_dots_pack_f32(nk_f32_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                 void *b_packed);
/** @copydoc nk_dots_pack_bf16 */
NK_DYNAMIC void nk_dots_pack_f64(nk_f64_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                 void *b_packed);
/** @copydoc nk_dots_pack_bf16 */
NK_DYNAMIC void nk_dots_pack_i8(nk_i8_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_pack_bf16 */
NK_DYNAMIC void nk_dots_pack_u8(nk_u8_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride, void *b_packed);
/** @copydoc nk_dots_pack_bf16 */
NK_DYNAMIC void nk_dots_pack_i4(nk_i4x2_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                void *b_packed);
/** @copydoc nk_dots_pack_bf16 */
NK_DYNAMIC void nk_dots_pack_u4(nk_u4x2_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                void *b_packed);
/** @copydoc nk_dots_pack_bf16 */
NK_DYNAMIC void nk_dots_pack_u1(nk_u1x8_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                void *b_packed);

/**
 *  @brief Computes C = A × Bᵀ using packed second multiplier matrix (B), accumulating into C.
 *  @param[in] a The input A matrix in row-major order.
 *  @param[in] b_packed The packed B matrix produced.
 *  @param[out] c The output C matrix in row-major order.
 *  @param[in] height The number of rows in A.
 *  @param[in] width The number of rows in B (output columns).
 *  @param[in] depth The shared inner dimension.
 *  @param[in] a_stride The row stride in bytes for A.
 *  @param[in] c_stride The row stride in bytes for C.
 */
NK_DYNAMIC void nk_dots_packed_bf16(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                    nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_packed_bf16 */
NK_DYNAMIC void nk_dots_packed_f16(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                   nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_packed_bf16 */
NK_DYNAMIC void nk_dots_packed_e4m3(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                    nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_packed_bf16 */
NK_DYNAMIC void nk_dots_packed_e5m2(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                    nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_packed_bf16 */
NK_DYNAMIC void nk_dots_packed_e2m3(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                    nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_packed_bf16 */
NK_DYNAMIC void nk_dots_packed_e3m2(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                    nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_packed_bf16 */
NK_DYNAMIC void nk_dots_packed_f32(nk_f32_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t height,
                                   nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_packed_bf16 */
NK_DYNAMIC void nk_dots_packed_f64(nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t height,
                                   nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_packed_bf16 */
NK_DYNAMIC void nk_dots_packed_i8(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t height,
                                  nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_packed_bf16 */
NK_DYNAMIC void nk_dots_packed_u8(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t height,
                                  nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_packed_bf16 */
NK_DYNAMIC void nk_dots_packed_i4(nk_i4x2_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t height,
                                  nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_packed_bf16 */
NK_DYNAMIC void nk_dots_packed_u4(nk_u4x2_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t height,
                                  nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_packed_bf16 */
NK_DYNAMIC void nk_dots_packed_u1(nk_u1x8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t height,
                                  nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);

/**
 *  @brief Computes C = A × Aᵀ symmetric Gram matrix.
 *  @param[in] vectors Input matrix of row vectors in row-major order.
 *  @param[in] vectors_count Number of vectors (rows) in the input matrix.
 *  @param[in] depth Dimension of each vector (columns).
 *  @param[in] stride Row stride in bytes for the input matrix.
 *  @param[out] result Output symmetric matrix (vectors_count × vectors_count).
 *  @param[in] result_stride Row stride in bytes for the result matrix.
 *  @param[in] row_start Starting row offset of results to compute (needed for parallelism).
 *  @param[in] row_count Number of rows of results to compute (needed for parallelism).
 */
NK_DYNAMIC void nk_dots_symmetric_bf16(nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                       nk_size_t stride, nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                       nk_size_t row_count);
/** @copydoc nk_dots_symmetric_bf16 */
NK_DYNAMIC void nk_dots_symmetric_f16(nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                      nk_size_t row_count);
/** @copydoc nk_dots_symmetric_bf16 */
NK_DYNAMIC void nk_dots_symmetric_e4m3(nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                       nk_size_t stride, nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                       nk_size_t row_count);
/** @copydoc nk_dots_symmetric_bf16 */
NK_DYNAMIC void nk_dots_symmetric_e5m2(nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                       nk_size_t stride, nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                       nk_size_t row_count);
/** @copydoc nk_dots_symmetric_bf16 */
NK_DYNAMIC void nk_dots_symmetric_e2m3(nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                       nk_size_t stride, nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                       nk_size_t row_count);
/** @copydoc nk_dots_symmetric_bf16 */
NK_DYNAMIC void nk_dots_symmetric_e3m2(nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                       nk_size_t stride, nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                       nk_size_t row_count);
/** @copydoc nk_dots_symmetric_bf16 */
NK_DYNAMIC void nk_dots_symmetric_f32(nk_f32_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                      nk_size_t stride, nk_f64_t *result, nk_size_t result_stride, nk_size_t row_start,
                                      nk_size_t row_count);
/** @copydoc nk_dots_symmetric_bf16 */
NK_DYNAMIC void nk_dots_symmetric_f64(nk_f64_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                      nk_size_t stride, nk_f64_t *result, nk_size_t result_stride, nk_size_t row_start,
                                      nk_size_t row_count);
/** @copydoc nk_dots_symmetric_bf16 */
NK_DYNAMIC void nk_dots_symmetric_i8(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride,
                                     nk_i32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                     nk_size_t row_count);
/** @copydoc nk_dots_symmetric_bf16 */
NK_DYNAMIC void nk_dots_symmetric_u8(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride,
                                     nk_u32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                     nk_size_t row_count);
/** @copydoc nk_dots_symmetric_bf16 */
NK_DYNAMIC void nk_dots_symmetric_i4(nk_i4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                     nk_size_t stride, nk_i32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                     nk_size_t row_count);
/** @copydoc nk_dots_symmetric_bf16 */
NK_DYNAMIC void nk_dots_symmetric_u4(nk_u4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                     nk_size_t stride, nk_u32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                     nk_size_t row_count);
/** @copydoc nk_dots_symmetric_bf16 */
NK_DYNAMIC void nk_dots_symmetric_u1(nk_u1x8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                     nk_size_t stride, nk_u32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                     nk_size_t row_count);

/** @copydoc nk_dots_packed_size_f32 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f32_serial(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_f32 */
NK_PUBLIC void nk_dots_pack_f32_serial(nk_f32_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                       void *b_packed);
/** @copydoc nk_dots_packed_f32 */
NK_PUBLIC void nk_dots_packed_f32_serial(nk_f32_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t height,
                                         nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f32 */
NK_PUBLIC void nk_dots_symmetric_f32_serial(nk_f32_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                            nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_f64 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f64_serial(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_f64 */
NK_PUBLIC void nk_dots_pack_f64_serial(nk_f64_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                       void *b_packed);
/** @copydoc nk_dots_packed_f64 */
NK_PUBLIC void nk_dots_packed_f64_serial(nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t height,
                                         nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f64 */
NK_PUBLIC void nk_dots_symmetric_f64_serial(nk_f64_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                            nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_f16 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f16_serial(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_f16 */
NK_PUBLIC void nk_dots_pack_f16_serial(nk_f16_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                       void *b_packed);
/** @copydoc nk_dots_packed_f16 */
NK_PUBLIC void nk_dots_packed_f16_serial(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                         nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f16 */
NK_PUBLIC void nk_dots_symmetric_f16_serial(nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                            nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_dots_packed_size_bf16_serial(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_bf16 */
NK_PUBLIC void nk_dots_pack_bf16_serial(nk_bf16_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                        void *b_packed);
/** @copydoc nk_dots_packed_bf16 */
NK_PUBLIC void nk_dots_packed_bf16_serial(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                          nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_bf16 */
NK_PUBLIC void nk_dots_symmetric_bf16_serial(nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                             nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_i8 */
NK_PUBLIC nk_size_t nk_dots_packed_size_i8_serial(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_i8 */
NK_PUBLIC void nk_dots_pack_i8_serial(nk_i8_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                      void *b_packed);
/** @copydoc nk_dots_packed_i8 */
NK_PUBLIC void nk_dots_packed_i8_serial(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t height,
                                        nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_i8 */
NK_PUBLIC void nk_dots_symmetric_i8_serial(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                           nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                           nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_u8 */
NK_PUBLIC nk_size_t nk_dots_packed_size_u8_serial(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_u8 */
NK_PUBLIC void nk_dots_pack_u8_serial(nk_u8_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                      void *b_packed);
/** @copydoc nk_dots_packed_u8 */
NK_PUBLIC void nk_dots_packed_u8_serial(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t height,
                                        nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_u8 */
NK_PUBLIC void nk_dots_symmetric_u8_serial(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                           nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                           nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_u4 */
NK_PUBLIC nk_size_t nk_dots_packed_size_u4_serial(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_u4 */
NK_PUBLIC void nk_dots_pack_u4_serial(nk_u4x2_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                      void *b_packed);
/** @copydoc nk_dots_packed_u4 */
NK_PUBLIC void nk_dots_packed_u4_serial(nk_u4x2_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t height,
                                        nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_u4 */
NK_PUBLIC void nk_dots_symmetric_u4_serial(nk_u4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                           nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                           nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_u1 */
NK_PUBLIC nk_size_t nk_dots_packed_size_u1_serial(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_u1 */
NK_PUBLIC void nk_dots_pack_u1_serial(nk_u1x8_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                      void *b_packed);
/** @copydoc nk_dots_packed_u1 */
NK_PUBLIC void nk_dots_packed_u1_serial(nk_u1x8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t height,
                                        nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_u1 */
NK_PUBLIC void nk_dots_symmetric_u1_serial(nk_u1x8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                           nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                           nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_i4 */
NK_PUBLIC nk_size_t nk_dots_packed_size_i4_serial(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_i4 */
NK_PUBLIC void nk_dots_pack_i4_serial(nk_i4x2_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                      void *b_packed);
/** @copydoc nk_dots_packed_i4 */
NK_PUBLIC void nk_dots_packed_i4_serial(nk_i4x2_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t height,
                                        nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_i4 */
NK_PUBLIC void nk_dots_symmetric_i4_serial(nk_i4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                           nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                           nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_symmetric_e4m3 */
NK_PUBLIC void nk_dots_symmetric_e4m3_serial(nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                             nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_symmetric_e5m2 */
NK_PUBLIC void nk_dots_symmetric_e5m2_serial(nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                             nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_symmetric_e2m3 */
NK_PUBLIC void nk_dots_symmetric_e2m3_serial(nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                             nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_symmetric_e3m2 */
NK_PUBLIC void nk_dots_symmetric_e3m2_serial(nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                             nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_e2m3 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e2m3_serial(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_e2m3 */
NK_PUBLIC void nk_dots_pack_e2m3_serial(nk_e2m3_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                        void *b_packed);
/** @copydoc nk_dots_packed_e2m3 */
NK_PUBLIC void nk_dots_packed_e2m3_serial(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                          nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_packed_size_e3m2 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e3m2_serial(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_e3m2 */
NK_PUBLIC void nk_dots_pack_e3m2_serial(nk_e3m2_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                        void *b_packed);
/** @copydoc nk_dots_packed_e3m2 */
NK_PUBLIC void nk_dots_packed_e3m2_serial(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                          nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);

/*  Genoa backends using AVX-512 with BF16 extensions.
 *  These use VDPBF16PS for BF16 dot products.
 *  Packing interleaves elements for SIMD broadcast patterns.
 */
#if NK_TARGET_GENOA
/** @copydoc nk_dots_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_dots_packed_size_bf16_genoa(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_bf16 */
NK_PUBLIC void nk_dots_pack_bf16_genoa(nk_bf16_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                       void *b_packed);
/** @copydoc nk_dots_packed_bf16 */
NK_PUBLIC void nk_dots_packed_bf16_genoa(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                         nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_bf16 */
NK_PUBLIC void nk_dots_symmetric_bf16_genoa(nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                            nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_e4m3 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e4m3_genoa(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_e4m3 */
NK_PUBLIC void nk_dots_pack_e4m3_genoa(nk_e4m3_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                       void *b_packed);
/** @copydoc nk_dots_packed_e4m3 */
NK_PUBLIC void nk_dots_packed_e4m3_genoa(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                         nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_packed_size_e5m2 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e5m2_genoa(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_e5m2 */
NK_PUBLIC void nk_dots_pack_e5m2_genoa(nk_e5m2_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                       void *b_packed);
/** @copydoc nk_dots_packed_e5m2 */
NK_PUBLIC void nk_dots_packed_e5m2_genoa(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                         nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e4m3 */
NK_PUBLIC void nk_dots_symmetric_e4m3_genoa(nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                            nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_symmetric_e5m2 */
NK_PUBLIC void nk_dots_symmetric_e5m2_genoa(nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                            nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_GENOA

#if NK_TARGET_DIAMOND
/** @copydoc nk_dots_packed_size_e4m3 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e4m3_diamond(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_e4m3 */
NK_PUBLIC void nk_dots_pack_e4m3_diamond(nk_e4m3_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                         void *b_packed);
/** @copydoc nk_dots_packed_e4m3 */
NK_PUBLIC void nk_dots_packed_e4m3_diamond(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                           nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_packed_size_e5m2 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e5m2_diamond(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_e5m2 */
NK_PUBLIC void nk_dots_pack_e5m2_diamond(nk_e5m2_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                         void *b_packed);
/** @copydoc nk_dots_packed_e5m2 */
NK_PUBLIC void nk_dots_packed_e5m2_diamond(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                           nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e4m3 */
NK_PUBLIC void nk_dots_symmetric_e4m3_diamond(nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_symmetric_e5m2 */
NK_PUBLIC void nk_dots_symmetric_e5m2_diamond(nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_DIAMOND

/*  Sapphire Rapids backends using Intel AMX (Advanced Matrix Extensions).
 *  AMX provides 8 tile registers (TMM0-TMM7), each holding up to 1KB of data.
 *  Tiles are configured as 16 rows × 64 bytes, enabling (16 × 32) BF16 or (16 × 64) INT8 tiles.
 *  Packing arranges data into AMX-native tile layout with pair interleaving for TDPBF16PS.
 */
#if NK_TARGET_SAPPHIREAMX
/** @copydoc nk_dots_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_dots_packed_size_bf16_sapphireamx(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_bf16 */
NK_PUBLIC void nk_dots_pack_bf16_sapphireamx(nk_bf16_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed);
/** @copydoc nk_dots_packed_bf16 */
NK_PUBLIC void nk_dots_packed_bf16_sapphireamx(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                               nk_size_t width, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_bf16 */
NK_PUBLIC void nk_dots_symmetric_bf16_sapphireamx(nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_i8 */
NK_PUBLIC nk_size_t nk_dots_packed_size_i8_sapphireamx(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_i8 */
NK_PUBLIC void nk_dots_pack_i8_sapphireamx(nk_i8_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                           void *b_packed);
/** @copydoc nk_dots_packed_i8 */
NK_PUBLIC void nk_dots_packed_i8_sapphireamx(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t height,
                                             nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_i8 */
NK_PUBLIC void nk_dots_symmetric_i8_sapphireamx(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_e4m3 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e4m3_sapphireamx(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_e4m3 */
NK_PUBLIC void nk_dots_pack_e4m3_sapphireamx(nk_e4m3_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed);
/** @copydoc nk_dots_packed_e4m3 */
NK_PUBLIC void nk_dots_packed_e4m3_sapphireamx(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                               nk_size_t width, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride);

/** @copydoc nk_dots_symmetric_e4m3 */
NK_PUBLIC void nk_dots_symmetric_e4m3_sapphireamx(nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_e5m2 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e5m2_sapphireamx(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_e5m2 */
NK_PUBLIC void nk_dots_pack_e5m2_sapphireamx(nk_e5m2_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed);
/** @copydoc nk_dots_packed_e5m2 */
NK_PUBLIC void nk_dots_packed_e5m2_sapphireamx(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                               nk_size_t width, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e5m2 */
NK_PUBLIC void nk_dots_symmetric_e5m2_sapphireamx(nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_e2m3 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e2m3_sapphireamx(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_e2m3 */
NK_PUBLIC void nk_dots_pack_e2m3_sapphireamx(nk_e2m3_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed);
/** @copydoc nk_dots_packed_e2m3 */
NK_PUBLIC void nk_dots_packed_e2m3_sapphireamx(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                               nk_size_t width, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e2m3 */
NK_PUBLIC void nk_dots_symmetric_e2m3_sapphireamx(nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_e3m2 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e3m2_sapphireamx(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_e3m2 */
NK_PUBLIC void nk_dots_pack_e3m2_sapphireamx(nk_e3m2_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed);
/** @copydoc nk_dots_packed_e3m2 */
NK_PUBLIC void nk_dots_packed_e3m2_sapphireamx(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                               nk_size_t width, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e3m2 */
NK_PUBLIC void nk_dots_symmetric_e3m2_sapphireamx(nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_u8 */
NK_PUBLIC nk_size_t nk_dots_packed_size_u8_sapphireamx(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_u8 */
NK_PUBLIC void nk_dots_pack_u8_sapphireamx(nk_u8_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                           void *b_packed);
/** @copydoc nk_dots_packed_u8 */
NK_PUBLIC void nk_dots_packed_u8_sapphireamx(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t height,
                                             nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_u8 */
NK_PUBLIC void nk_dots_symmetric_u8_sapphireamx(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_SAPPHIREAMX

/*  Granite Rapids backends using Intel AMX-FP16 (Advanced Matrix Extensions with FP16 support).
 *  AMX-FP16 adds TDPFP16PS (FP16×FP16→FP32 tile multiply-accumulate), same tile geometry as BF16.
 *  The F32 Ozaki kernel splits F32 inputs into 2 FP16 halves for ~35-40 bit effective precision.
 */
#if NK_TARGET_GRANITEAMX
/** @copydoc nk_dots_packed_size_f16 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f16_graniteamx(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_f16 */
NK_PUBLIC void nk_dots_pack_f16_graniteamx(nk_f16_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                           void *b_packed);
/** @copydoc nk_dots_packed_f16 */
NK_PUBLIC void nk_dots_packed_f16_graniteamx(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                             nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f16 */
NK_PUBLIC void nk_dots_symmetric_f16_graniteamx(nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_GRANITEAMX

/*  ARM SME backends using Scalable Matrix Extension.
 *  SME provides ZA tile registers for outer product operations.
 *  F16/BF16/I8/U8/E4M3 use ZA32 tiles, F32/F64 use ZA64 tiles (FEAT_SME_F64F64).
 */
#if NK_TARGET_SME
/** @copydoc nk_dots_packed_size_f16 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f16_sme(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_f16 */
NK_PUBLIC void nk_dots_pack_f16_sme(nk_f16_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                    void *b_packed);
/** @copydoc nk_dots_packed_f16 */
NK_PUBLIC void nk_dots_packed_f16_sme(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                      nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f16 */
NK_PUBLIC void nk_dots_symmetric_f16_sme(nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                         nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                         nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_dots_packed_size_bf16_sme(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_bf16 */
NK_PUBLIC void nk_dots_pack_bf16_sme(nk_bf16_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                     void *b_packed);
/** @copydoc nk_dots_packed_bf16 */
NK_PUBLIC void nk_dots_packed_bf16_sme(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                       nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_bf16 */
NK_PUBLIC void nk_dots_symmetric_bf16_sme(nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_i8 */
NK_PUBLIC nk_size_t nk_dots_packed_size_i8_sme(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_i8 */
NK_PUBLIC void nk_dots_pack_i8_sme(nk_i8_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                   void *b_packed);
/** @copydoc nk_dots_packed_i8 */
NK_PUBLIC void nk_dots_packed_i8_sme(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t height,
                                     nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_i8 */
NK_PUBLIC void nk_dots_symmetric_i8_sme(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                        nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                        nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_u8 */
NK_PUBLIC nk_size_t nk_dots_packed_size_u8_sme(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_u8 */
NK_PUBLIC void nk_dots_pack_u8_sme(nk_u8_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                   void *b_packed);
/** @copydoc nk_dots_packed_u8 */
NK_PUBLIC void nk_dots_packed_u8_sme(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t height,
                                     nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_u8 */
NK_PUBLIC void nk_dots_symmetric_u8_sme(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                        nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                        nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_e4m3 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e4m3_sme(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_e4m3 */
NK_PUBLIC void nk_dots_pack_e4m3_sme(nk_e4m3_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                     void *b_packed);
/** @copydoc nk_dots_packed_e4m3 */
NK_PUBLIC void nk_dots_packed_e4m3_sme(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                       nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e4m3 */
NK_PUBLIC void nk_dots_symmetric_e4m3_sme(nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_e5m2 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e5m2_sme(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_e5m2 */
NK_PUBLIC void nk_dots_pack_e5m2_sme(nk_e5m2_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                     void *b_packed);
/** @copydoc nk_dots_packed_e5m2 */
NK_PUBLIC void nk_dots_packed_e5m2_sme(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                       nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e5m2 */
NK_PUBLIC void nk_dots_symmetric_e5m2_sme(nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_u4 */
NK_PUBLIC nk_size_t nk_dots_packed_size_u4_sme(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_u4 */
NK_PUBLIC void nk_dots_pack_u4_sme(nk_u4x2_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                   void *b_packed);
/** @copydoc nk_dots_packed_u4 */
NK_PUBLIC void nk_dots_packed_u4_sme(nk_u4x2_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t height,
                                     nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_u4 */
NK_PUBLIC void nk_dots_symmetric_u4_sme(nk_u4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                        nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                        nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_i4 */
NK_PUBLIC nk_size_t nk_dots_packed_size_i4_sme(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_i4 */
NK_PUBLIC void nk_dots_pack_i4_sme(nk_i4x2_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                   void *b_packed);
/** @copydoc nk_dots_packed_i4 */
NK_PUBLIC void nk_dots_packed_i4_sme(nk_i4x2_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t height,
                                     nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_i4 */
NK_PUBLIC void nk_dots_symmetric_i4_sme(nk_i4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                        nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                        nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_e2m3 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e2m3_sme(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_e2m3 */
NK_PUBLIC void nk_dots_pack_e2m3_sme(nk_e2m3_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                     void *b_packed);
/** @copydoc nk_dots_packed_e2m3 */
NK_PUBLIC void nk_dots_packed_e2m3_sme(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                       nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e2m3 */
NK_PUBLIC void nk_dots_symmetric_e2m3_sme(nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_e3m2 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e3m2_sme(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_e3m2 */
NK_PUBLIC void nk_dots_pack_e3m2_sme(nk_e3m2_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                     void *b_packed);
/** @copydoc nk_dots_packed_e3m2 */
NK_PUBLIC void nk_dots_packed_e3m2_sme(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                       nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e3m2 */
NK_PUBLIC void nk_dots_symmetric_e3m2_sme(nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_SME

/*  ARM SME with integer-accumulating binary outer products.
 *  Used for packed 1-bit dot products backed by ZA32.
 */
#if NK_TARGET_SMEBI32
/** @copydoc nk_dots_packed_size_u1 */
NK_PUBLIC nk_size_t nk_dots_packed_size_u1_smebi32(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_u1 */
NK_PUBLIC void nk_dots_pack_u1_smebi32(nk_u1x8_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                       void *b_packed);
/** @copydoc nk_dots_packed_u1 */
NK_PUBLIC void nk_dots_packed_u1_smebi32(nk_u1x8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t height,
                                         nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_u1 */
NK_PUBLIC void nk_dots_symmetric_u1_smebi32(nk_u1x8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                            nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_SMEBI32

/*  ARM SME with FEAT_SME_F64F64 (F32/F64 with F64 accumulators).
 *  Requires Apple M4 or equivalent with F64 outer product support.
 */
#if NK_TARGET_SMEF64
/** @copydoc nk_dots_packed_size_f32 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f32_smef64(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_f32 */
NK_PUBLIC void nk_dots_pack_f32_smef64(nk_f32_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                       void *b_packed);
/** @copydoc nk_dots_packed_f32 */
NK_PUBLIC void nk_dots_packed_f32_smef64(nk_f32_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t height,
                                         nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f32 */
NK_PUBLIC void nk_dots_symmetric_f32_smef64(nk_f32_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                            nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_dots_packed_size_f64 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f64_smef64(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_f64 */
NK_PUBLIC void nk_dots_pack_f64_smef64(nk_f64_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                       void *b_packed);
/** @copydoc nk_dots_packed_f64 */
NK_PUBLIC void nk_dots_packed_f64_smef64(nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t height,
                                         nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f64 */
NK_PUBLIC void nk_dots_symmetric_f64_smef64(nk_f64_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                            nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_SMEF64

/*  Haswell backends using AVX2 (Intel Core 4th gen).
 *  Supports F32/F64 via FMA, F16/BF16/FP8 via software emulation, I8/U8 via VPMADDUBSW+VPADDD.
 */
#if NK_TARGET_HASWELL
/** @copydoc nk_dots_packed_size_f32 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f32_haswell(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_f32 */
NK_PUBLIC void nk_dots_pack_f32_haswell(nk_f32_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                        void *b_packed);
/** @copydoc nk_dots_packed_f32 */
NK_PUBLIC void nk_dots_packed_f32_haswell(nk_f32_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t height,
                                          nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f32 */
NK_PUBLIC void nk_dots_symmetric_f32_haswell(nk_f32_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                             nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_f64 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f64_haswell(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_f64 */
NK_PUBLIC void nk_dots_pack_f64_haswell(nk_f64_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                        void *b_packed);
/** @copydoc nk_dots_packed_f64 */
NK_PUBLIC void nk_dots_packed_f64_haswell(nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t height,
                                          nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f64 */
NK_PUBLIC void nk_dots_symmetric_f64_haswell(nk_f64_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                             nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_f16 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f16_haswell(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_f16 */
NK_PUBLIC void nk_dots_pack_f16_haswell(nk_f16_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                        void *b_packed);
/** @copydoc nk_dots_packed_f16 */
NK_PUBLIC void nk_dots_packed_f16_haswell(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                          nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f16 */
NK_PUBLIC void nk_dots_symmetric_f16_haswell(nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                             nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_dots_packed_size_bf16_haswell(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_bf16 */
NK_PUBLIC void nk_dots_pack_bf16_haswell(nk_bf16_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                         void *b_packed);
/** @copydoc nk_dots_packed_bf16 */
NK_PUBLIC void nk_dots_packed_bf16_haswell(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                           nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_bf16 */
NK_PUBLIC void nk_dots_symmetric_bf16_haswell(nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_e4m3 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e4m3_haswell(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_e4m3 */
NK_PUBLIC void nk_dots_pack_e4m3_haswell(nk_e4m3_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                         void *b_packed);
/** @copydoc nk_dots_packed_e4m3 */
NK_PUBLIC void nk_dots_packed_e4m3_haswell(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                           nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e4m3 */
NK_PUBLIC void nk_dots_symmetric_e4m3_haswell(nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_e5m2 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e5m2_haswell(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_e5m2 */
NK_PUBLIC void nk_dots_pack_e5m2_haswell(nk_e5m2_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                         void *b_packed);
/** @copydoc nk_dots_packed_e5m2 */
NK_PUBLIC void nk_dots_packed_e5m2_haswell(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                           nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e5m2 */
NK_PUBLIC void nk_dots_symmetric_e5m2_haswell(nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_e2m3 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e2m3_haswell(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_e2m3 */
NK_PUBLIC void nk_dots_pack_e2m3_haswell(nk_e2m3_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                         void *b_packed);
/** @copydoc nk_dots_packed_e2m3 */
NK_PUBLIC void nk_dots_packed_e2m3_haswell(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                           nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e2m3 */
NK_PUBLIC void nk_dots_symmetric_e2m3_haswell(nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_e3m2 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e3m2_haswell(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_e3m2 */
NK_PUBLIC void nk_dots_pack_e3m2_haswell(nk_e3m2_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                         void *b_packed);
/** @copydoc nk_dots_packed_e3m2 */
NK_PUBLIC void nk_dots_packed_e3m2_haswell(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                           nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e3m2 */
NK_PUBLIC void nk_dots_symmetric_e3m2_haswell(nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_i8 */
NK_PUBLIC nk_size_t nk_dots_packed_size_i8_haswell(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_i8 */
NK_PUBLIC void nk_dots_pack_i8_haswell(nk_i8_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                       void *b_packed);
/** @copydoc nk_dots_packed_i8 */
NK_PUBLIC void nk_dots_packed_i8_haswell(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t height,
                                         nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_i8 */
NK_PUBLIC void nk_dots_symmetric_i8_haswell(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                            nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_u8 */
NK_PUBLIC nk_size_t nk_dots_packed_size_u8_haswell(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_u8 */
NK_PUBLIC void nk_dots_pack_u8_haswell(nk_u8_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                       void *b_packed);
/** @copydoc nk_dots_packed_u8 */
NK_PUBLIC void nk_dots_packed_u8_haswell(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t height,
                                         nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_u8 */
NK_PUBLIC void nk_dots_symmetric_u8_haswell(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                            nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_u1 */
NK_PUBLIC nk_size_t nk_dots_packed_size_u1_haswell(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_u1 */
NK_PUBLIC void nk_dots_pack_u1_haswell(nk_u1x8_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                       void *b_packed);
/** @copydoc nk_dots_packed_u1 */
NK_PUBLIC void nk_dots_packed_u1_haswell(nk_u1x8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t height,
                                         nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_u1 */
NK_PUBLIC void nk_dots_symmetric_u1_haswell(nk_u1x8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                            nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_i4 */
NK_PUBLIC nk_size_t nk_dots_packed_size_i4_haswell(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_i4 */
NK_PUBLIC void nk_dots_pack_i4_haswell(nk_i4x2_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                       void *b_packed);
/** @copydoc nk_dots_packed_i4 */
NK_PUBLIC void nk_dots_packed_i4_haswell(nk_i4x2_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t height,
                                         nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_i4 */
NK_PUBLIC void nk_dots_symmetric_i4_haswell(nk_i4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                            nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_u4 */
NK_PUBLIC nk_size_t nk_dots_packed_size_u4_haswell(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_u4 */
NK_PUBLIC void nk_dots_pack_u4_haswell(nk_u4x2_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                       void *b_packed);
/** @copydoc nk_dots_packed_u4 */
NK_PUBLIC void nk_dots_packed_u4_haswell(nk_u4x2_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t height,
                                         nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_u4 */
NK_PUBLIC void nk_dots_symmetric_u4_haswell(nk_u4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                            nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_HASWELL

/*  Skylake backends using AVX-512 (Intel Core 6th gen+).
 *  Provides 512-bit vectors (16× f32, 8× f64), supporting F32/F64/F16/BF16/FP8 with FMA.
 */
#if NK_TARGET_SKYLAKE
/** @copydoc nk_dots_packed_size_f64 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f64_skylake(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_f64 */
NK_PUBLIC void nk_dots_pack_f64_skylake(nk_f64_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                        void *b_packed);
/** @copydoc nk_dots_packed_f64 */
NK_PUBLIC void nk_dots_packed_f64_skylake(nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t height,
                                          nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f64 */
NK_PUBLIC void nk_dots_symmetric_f64_skylake(nk_f64_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                             nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_f32 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f32_skylake(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_f32 */
NK_PUBLIC void nk_dots_pack_f32_skylake(nk_f32_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                        void *b_packed);
/** @copydoc nk_dots_packed_f32 */
NK_PUBLIC void nk_dots_packed_f32_skylake(nk_f32_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t height,
                                          nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f32 */
NK_PUBLIC void nk_dots_symmetric_f32_skylake(nk_f32_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                             nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_dots_packed_size_bf16_skylake(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_bf16 */
NK_PUBLIC void nk_dots_pack_bf16_skylake(nk_bf16_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                         void *b_packed);
/** @copydoc nk_dots_packed_bf16 */
NK_PUBLIC void nk_dots_packed_bf16_skylake(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                           nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_bf16 */
NK_PUBLIC void nk_dots_symmetric_bf16_skylake(nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_f16 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f16_skylake(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_f16 */
NK_PUBLIC void nk_dots_pack_f16_skylake(nk_f16_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                        void *b_packed);
/** @copydoc nk_dots_packed_f16 */
NK_PUBLIC void nk_dots_packed_f16_skylake(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                          nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f16 */
NK_PUBLIC void nk_dots_symmetric_f16_skylake(nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                             nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_e4m3 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e4m3_skylake(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_e4m3 */
NK_PUBLIC void nk_dots_pack_e4m3_skylake(nk_e4m3_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                         void *b_packed);
/** @copydoc nk_dots_packed_e4m3 */
NK_PUBLIC void nk_dots_packed_e4m3_skylake(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                           nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e4m3 */
NK_PUBLIC void nk_dots_symmetric_e4m3_skylake(nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_e5m2 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e5m2_skylake(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_e5m2 */
NK_PUBLIC void nk_dots_pack_e5m2_skylake(nk_e5m2_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                         void *b_packed);
/** @copydoc nk_dots_packed_e5m2 */
NK_PUBLIC void nk_dots_packed_e5m2_skylake(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                           nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e5m2 */
NK_PUBLIC void nk_dots_symmetric_e5m2_skylake(nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_e2m3 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e2m3_skylake(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_e2m3 */
NK_PUBLIC void nk_dots_pack_e2m3_skylake(nk_e2m3_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                         void *b_packed);
/** @copydoc nk_dots_packed_e2m3 */
NK_PUBLIC void nk_dots_packed_e2m3_skylake(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                           nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e2m3 */
NK_PUBLIC void nk_dots_symmetric_e2m3_skylake(nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_e3m2 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e3m2_skylake(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_e3m2 */
NK_PUBLIC void nk_dots_pack_e3m2_skylake(nk_e3m2_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                         void *b_packed);
/** @copydoc nk_dots_packed_e3m2 */
NK_PUBLIC void nk_dots_packed_e3m2_skylake(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                           nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e3m2 */
NK_PUBLIC void nk_dots_symmetric_e3m2_skylake(nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_SKYLAKE

/*  Ice Lake backends using AVX-512 with VNNI (Vector Neural Network Instructions).
 *  Adds VPDPBUSD for I8/U8, VPDPWSSD for I4/U4 with efficient dot products.
 */
#if NK_TARGET_ICELAKE
/** @copydoc nk_dots_packed_size_i8 */
NK_PUBLIC nk_size_t nk_dots_packed_size_i8_icelake(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_i8 */
NK_PUBLIC void nk_dots_pack_i8_icelake(nk_i8_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                       void *b_packed);
/** @copydoc nk_dots_packed_i8 */
NK_PUBLIC void nk_dots_packed_i8_icelake(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t height,
                                         nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_i8 */
NK_PUBLIC void nk_dots_symmetric_i8_icelake(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                            nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_u8 */
NK_PUBLIC nk_size_t nk_dots_packed_size_u8_icelake(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_u8 */
NK_PUBLIC void nk_dots_pack_u8_icelake(nk_u8_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                       void *b_packed);
/** @copydoc nk_dots_packed_u8 */
NK_PUBLIC void nk_dots_packed_u8_icelake(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t height,
                                         nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_u8 */
NK_PUBLIC void nk_dots_symmetric_u8_icelake(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                            nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_i4 */
NK_PUBLIC nk_size_t nk_dots_packed_size_i4_icelake(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_i4 */
NK_PUBLIC void nk_dots_pack_i4_icelake(nk_i4x2_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                       void *b_packed);
/** @copydoc nk_dots_packed_i4 */
NK_PUBLIC void nk_dots_packed_i4_icelake(nk_i4x2_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t height,
                                         nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_i4 */
NK_PUBLIC void nk_dots_symmetric_i4_icelake(nk_i4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                            nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_u4 */
NK_PUBLIC nk_size_t nk_dots_packed_size_u4_icelake(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_u4 */
NK_PUBLIC void nk_dots_pack_u4_icelake(nk_u4x2_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                       void *b_packed);
/** @copydoc nk_dots_packed_u4 */
NK_PUBLIC void nk_dots_packed_u4_icelake(nk_u4x2_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t height,
                                         nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_u4 */
NK_PUBLIC void nk_dots_symmetric_u4_icelake(nk_u4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                            nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_u1 */
NK_PUBLIC nk_size_t nk_dots_packed_size_u1_icelake(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_u1 */
NK_PUBLIC void nk_dots_pack_u1_icelake(nk_u1x8_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                       void *b_packed);
/** @copydoc nk_dots_packed_u1 */
NK_PUBLIC void nk_dots_packed_u1_icelake(nk_u1x8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t height,
                                         nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_u1 */
NK_PUBLIC void nk_dots_symmetric_u1_icelake(nk_u1x8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                            nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_ICELAKE

/*  Alder backends using AMX with TDPB[SU]SD / TDPBF16PS.
 *  Optimized for I8/U8 via AMX integer tiles, E2M3 via AMX BF16 tiles.
 */
#if NK_TARGET_ALDER
/** @copydoc nk_dots_packed_size_i8 */
NK_PUBLIC nk_size_t nk_dots_packed_size_i8_alder(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_i8 */
NK_PUBLIC void nk_dots_pack_i8_alder(nk_i8_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                     void *b_packed);
/** @copydoc nk_dots_packed_i8 */
NK_PUBLIC void nk_dots_packed_i8_alder(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t height,
                                       nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_i8 */
NK_PUBLIC void nk_dots_symmetric_i8_alder(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                          nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_u8 */
NK_PUBLIC nk_size_t nk_dots_packed_size_u8_alder(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_u8 */
NK_PUBLIC void nk_dots_pack_u8_alder(nk_u8_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                     void *b_packed);
/** @copydoc nk_dots_packed_u8 */
NK_PUBLIC void nk_dots_packed_u8_alder(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t height,
                                       nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_u8 */
NK_PUBLIC void nk_dots_symmetric_u8_alder(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                          nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_e2m3 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e2m3_alder(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_e2m3 */
NK_PUBLIC void nk_dots_pack_e2m3_alder(nk_e2m3_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                       void *b_packed);
/** @copydoc nk_dots_packed_e2m3 */
NK_PUBLIC void nk_dots_packed_e2m3_alder(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                         nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e2m3 */
NK_PUBLIC void nk_dots_symmetric_e2m3_alder(nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                            nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_ALDER

/*  Sierra backends using AVX10.2 with VMPSADBW.
 *  Optimized for I8/U8 via VMPSADBW (vector multiply-sum of absolute differences).
 */
#if NK_TARGET_SIERRA
/** @copydoc nk_dots_packed_size_i8 */
NK_PUBLIC nk_size_t nk_dots_packed_size_i8_sierra(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_i8 */
NK_PUBLIC void nk_dots_pack_i8_sierra(nk_i8_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                      void *b_packed);
/** @copydoc nk_dots_packed_i8 */
NK_PUBLIC void nk_dots_packed_i8_sierra(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t height,
                                        nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_i8 */
NK_PUBLIC void nk_dots_symmetric_i8_sierra(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                           nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                           nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_u8 */
NK_PUBLIC nk_size_t nk_dots_packed_size_u8_sierra(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_u8 */
NK_PUBLIC void nk_dots_pack_u8_sierra(nk_u8_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                      void *b_packed);
/** @copydoc nk_dots_packed_u8 */
NK_PUBLIC void nk_dots_packed_u8_sierra(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t height,
                                        nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_u8 */
NK_PUBLIC void nk_dots_symmetric_u8_sierra(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                           nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                           nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_e2m3 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e2m3_sierra(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_e2m3 */
NK_PUBLIC void nk_dots_pack_e2m3_sierra(nk_e2m3_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                        void *b_packed);
/** @copydoc nk_dots_packed_e2m3 */
NK_PUBLIC void nk_dots_packed_e2m3_sierra(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                          nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e2m3 */
NK_PUBLIC void nk_dots_symmetric_e2m3_sierra(nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                             nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_SIERRA

/*  WASM Relaxed SIMD backends using wasm_i32x4_relaxed_dot_i8x16_i7x16_add.
 *  Covers I8/U8/E2M3 (depth_simd_dimensions=16), BF16/F32 (4), F64 (2).
 */
#if NK_TARGET_V128RELAXED
/** @copydoc nk_dots_packed_size_i8 */
NK_PUBLIC nk_size_t nk_dots_packed_size_i8_v128relaxed(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_i8 */
NK_PUBLIC void nk_dots_pack_i8_v128relaxed(nk_i8_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                           void *b_packed);
/** @copydoc nk_dots_packed_i8 */
NK_PUBLIC void nk_dots_packed_i8_v128relaxed(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t height,
                                             nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_i8 */
NK_PUBLIC void nk_dots_symmetric_i8_v128relaxed(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_u8 */
NK_PUBLIC nk_size_t nk_dots_packed_size_u8_v128relaxed(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_u8 */
NK_PUBLIC void nk_dots_pack_u8_v128relaxed(nk_u8_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                           void *b_packed);
/** @copydoc nk_dots_packed_u8 */
NK_PUBLIC void nk_dots_packed_u8_v128relaxed(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t height,
                                             nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_u8 */
NK_PUBLIC void nk_dots_symmetric_u8_v128relaxed(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_e2m3 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e2m3_v128relaxed(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_e2m3 */
NK_PUBLIC void nk_dots_pack_e2m3_v128relaxed(nk_e2m3_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed);
/** @copydoc nk_dots_packed_e2m3 */
NK_PUBLIC void nk_dots_packed_e2m3_v128relaxed(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                               nk_size_t width, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e2m3 */
NK_PUBLIC void nk_dots_symmetric_e2m3_v128relaxed(nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_dots_packed_size_bf16_v128relaxed(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_bf16 */
NK_PUBLIC void nk_dots_pack_bf16_v128relaxed(nk_bf16_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed);
/** @copydoc nk_dots_packed_bf16 */
NK_PUBLIC void nk_dots_packed_bf16_v128relaxed(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                               nk_size_t width, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_bf16 */
NK_PUBLIC void nk_dots_symmetric_bf16_v128relaxed(nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_f32 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f32_v128relaxed(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_f32 */
NK_PUBLIC void nk_dots_pack_f32_v128relaxed(nk_f32_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                            void *b_packed);
/** @copydoc nk_dots_packed_f32 */
NK_PUBLIC void nk_dots_packed_f32_v128relaxed(nk_f32_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t height,
                                              nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f32 */
NK_PUBLIC void nk_dots_symmetric_f32_v128relaxed(nk_f32_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                 nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                 nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_f64 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f64_v128relaxed(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_f64 */
NK_PUBLIC void nk_dots_pack_f64_v128relaxed(nk_f64_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                            void *b_packed);
/** @copydoc nk_dots_packed_f64 */
NK_PUBLIC void nk_dots_packed_f64_v128relaxed(nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t height,
                                              nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f64 */
NK_PUBLIC void nk_dots_symmetric_f64_v128relaxed(nk_f64_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                 nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                 nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e4m3_v128relaxed(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_bf16 */
NK_PUBLIC void nk_dots_pack_e4m3_v128relaxed(nk_e4m3_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed);
/** @copydoc nk_dots_packed_bf16 */
NK_PUBLIC void nk_dots_packed_e4m3_v128relaxed(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                               nk_size_t width, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_bf16 */
NK_PUBLIC void nk_dots_symmetric_e4m3_v128relaxed(nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e5m2_v128relaxed(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_bf16 */
NK_PUBLIC void nk_dots_pack_e5m2_v128relaxed(nk_e5m2_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed);
/** @copydoc nk_dots_packed_bf16 */
NK_PUBLIC void nk_dots_packed_e5m2_v128relaxed(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                               nk_size_t width, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_bf16 */
NK_PUBLIC void nk_dots_symmetric_e5m2_v128relaxed(nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_dots_packed_size_u4_v128relaxed(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_bf16 */
NK_PUBLIC void nk_dots_pack_u4_v128relaxed(nk_u4x2_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                           void *b_packed);
/** @copydoc nk_dots_packed_bf16 */
NK_PUBLIC void nk_dots_packed_u4_v128relaxed(nk_u4x2_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t height,
                                             nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_bf16 */
NK_PUBLIC void nk_dots_symmetric_u4_v128relaxed(nk_u4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_dots_packed_size_i4_v128relaxed(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_bf16 */
NK_PUBLIC void nk_dots_pack_i4_v128relaxed(nk_i4x2_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                           void *b_packed);
/** @copydoc nk_dots_packed_bf16 */
NK_PUBLIC void nk_dots_packed_i4_v128relaxed(nk_i4x2_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t height,
                                             nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_bf16 */
NK_PUBLIC void nk_dots_symmetric_i4_v128relaxed(nk_i4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_u1 */
NK_PUBLIC nk_size_t nk_dots_packed_size_u1_v128relaxed(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_u1 */
NK_PUBLIC void nk_dots_pack_u1_v128relaxed(nk_u1x8_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                           void *b_packed);
/** @copydoc nk_dots_packed_u1 */
NK_PUBLIC void nk_dots_packed_u1_v128relaxed(nk_u1x8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t height,
                                             nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_u1 */
NK_PUBLIC void nk_dots_symmetric_u1_v128relaxed(nk_u1x8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_V128RELAXED

/*  ARM NEON backends (base NEON with F32/F64 support).
 *  Uses FMLA for F32 dots, FMLA (scalar) for F64.
 */
#if NK_TARGET_NEON
/** @copydoc nk_dots_packed_size_f32 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f32_neon(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_f32 */
NK_PUBLIC void nk_dots_pack_f32_neon(nk_f32_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                     void *b_packed);
/** @copydoc nk_dots_packed_f32 */
NK_PUBLIC void nk_dots_packed_f32_neon(nk_f32_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t height,
                                       nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f32 */
NK_PUBLIC void nk_dots_symmetric_f32_neon(nk_f32_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                          nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_f64 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f64_neon(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_f64 */
NK_PUBLIC void nk_dots_pack_f64_neon(nk_f64_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                     void *b_packed);
/** @copydoc nk_dots_packed_f64 */
NK_PUBLIC void nk_dots_packed_f64_neon(nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t height,
                                       nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f64 */
NK_PUBLIC void nk_dots_symmetric_f64_neon(nk_f64_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                          nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_u1 */
NK_PUBLIC nk_size_t nk_dots_packed_size_u1_neon(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_u1 */
NK_PUBLIC void nk_dots_pack_u1_neon(nk_u1x8_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                    void *b_packed);
/** @copydoc nk_dots_packed_u1 */
NK_PUBLIC void nk_dots_packed_u1_neon(nk_u1x8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t height,
                                      nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_u1 */
NK_PUBLIC void nk_dots_symmetric_u1_neon(nk_u1x8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                         nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                         nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_f16 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f16_neon(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_f16 */
NK_PUBLIC void nk_dots_pack_f16_neon(nk_f16_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                     void *b_packed);
/** @copydoc nk_dots_packed_f16 */
NK_PUBLIC void nk_dots_packed_f16_neon(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                       nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f16 */
NK_PUBLIC void nk_dots_symmetric_f16_neon(nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_dots_packed_size_bf16_neon(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_bf16 */
NK_PUBLIC void nk_dots_pack_bf16_neon(nk_bf16_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                      void *b_packed);
/** @copydoc nk_dots_packed_bf16 */
NK_PUBLIC void nk_dots_packed_bf16_neon(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                        nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_bf16 */
NK_PUBLIC void nk_dots_symmetric_bf16_neon(nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                           nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                           nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_NEON

/*  ARM NEON with BF16 dot product (ARMv8.6-A BF16).
 *  Uses BFDOT/BFMMLA for efficient BF16 matrix operations.
 */
#if NK_TARGET_NEONBFDOT
/** @copydoc nk_dots_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_dots_packed_size_bf16_neonbfdot(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_bf16 */
NK_PUBLIC void nk_dots_pack_bf16_neonbfdot(nk_bf16_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                           void *b_packed);
/** @copydoc nk_dots_packed_bf16 */
NK_PUBLIC void nk_dots_packed_bf16_neonbfdot(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                             nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_bf16 */
NK_PUBLIC void nk_dots_symmetric_bf16_neonbfdot(nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_NEONBFDOT

/*  ARM NEON with signed/unsigned dot product (ARMv8.2-A DotProd).
 *  Provides SDOT/UDOT for I8/U8 vector dot products.
 */
#if NK_TARGET_NEONSDOT
/** @copydoc nk_dots_packed_size_i8 */
NK_PUBLIC nk_size_t nk_dots_packed_size_i8_neonsdot(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_i8 */
NK_PUBLIC void nk_dots_pack_i8_neonsdot(nk_i8_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                        void *b_packed);
/** @copydoc nk_dots_packed_i8 */
NK_PUBLIC void nk_dots_packed_i8_neonsdot(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t height,
                                          nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_i8 */
NK_PUBLIC void nk_dots_symmetric_i8_neonsdot(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                             nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_u8 */
NK_PUBLIC nk_size_t nk_dots_packed_size_u8_neonsdot(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_u8 */
NK_PUBLIC void nk_dots_pack_u8_neonsdot(nk_u8_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                        void *b_packed);
/** @copydoc nk_dots_packed_u8 */
NK_PUBLIC void nk_dots_packed_u8_neonsdot(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t height,
                                          nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_u8 */
NK_PUBLIC void nk_dots_symmetric_u8_neonsdot(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                             nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_NEONSDOT

/*  ARM NEON with FP16 FML (fused multiply-long, ARMv8.2-A FP16FML).
 *  Uses FMLAL/FMLSL for F16 and custom FP8 (E2M3/E3M2) operations.
 */
#if NK_TARGET_NEONFHM
/** @copydoc nk_dots_packed_size_f16 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f16_neonfhm(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_f16 */
NK_PUBLIC void nk_dots_pack_f16_neonfhm(nk_f16_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                        void *b_packed);
/** @copydoc nk_dots_packed_f16 */
NK_PUBLIC void nk_dots_packed_f16_neonfhm(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                          nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f16 */
NK_PUBLIC void nk_dots_symmetric_f16_neonfhm(nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                             nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_e4m3 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e4m3_neonfhm(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_e4m3 */
NK_PUBLIC void nk_dots_pack_e4m3_neonfhm(nk_e4m3_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                         void *b_packed);
/** @copydoc nk_dots_packed_e4m3 */
NK_PUBLIC void nk_dots_packed_e4m3_neonfhm(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                           nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e4m3 */
NK_PUBLIC void nk_dots_symmetric_e4m3_neonfhm(nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_e5m2 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e5m2_neonfhm(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_e5m2 */
NK_PUBLIC void nk_dots_pack_e5m2_neonfhm(nk_e5m2_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                         void *b_packed);
/** @copydoc nk_dots_packed_e5m2 */
NK_PUBLIC void nk_dots_packed_e5m2_neonfhm(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                           nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e5m2 */
NK_PUBLIC void nk_dots_symmetric_e5m2_neonfhm(nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_NEONFHM

/*  ARM NEON with FP8 (ARMv9.2-A FP8).
 *  Uses native FP8 dot-product instructions for E4M3/E5M2/E2M3/E3M2 operations.
 */
#if NK_TARGET_NEONFP8
/** @copydoc nk_dots_packed_size_e4m3 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e4m3_neonfp8(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_e4m3 */
NK_PUBLIC void nk_dots_pack_e4m3_neonfp8(nk_e4m3_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                         void *b_packed);
/** @copydoc nk_dots_packed_e4m3 */
NK_PUBLIC void nk_dots_packed_e4m3_neonfp8(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                           nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e4m3 */
NK_PUBLIC void nk_dots_symmetric_e4m3_neonfp8(nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_e5m2 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e5m2_neonfp8(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_e5m2 */
NK_PUBLIC void nk_dots_pack_e5m2_neonfp8(nk_e5m2_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                         void *b_packed);
/** @copydoc nk_dots_packed_e5m2 */
NK_PUBLIC void nk_dots_packed_e5m2_neonfp8(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                           nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e5m2 */
NK_PUBLIC void nk_dots_symmetric_e5m2_neonfp8(nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_e2m3 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e2m3_neonfp8(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_e2m3 */
NK_PUBLIC void nk_dots_pack_e2m3_neonfp8(nk_e2m3_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                         void *b_packed);
/** @copydoc nk_dots_packed_e2m3 */
NK_PUBLIC void nk_dots_packed_e2m3_neonfp8(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                           nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e2m3 */
NK_PUBLIC void nk_dots_symmetric_e2m3_neonfp8(nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_e3m2 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e3m2_neonfp8(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_e3m2 */
NK_PUBLIC void nk_dots_pack_e3m2_neonfp8(nk_e3m2_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                         void *b_packed);
/** @copydoc nk_dots_packed_e3m2 */
NK_PUBLIC void nk_dots_packed_e3m2_neonfp8(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                           nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e3m2 */
NK_PUBLIC void nk_dots_symmetric_e3m2_neonfp8(nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_NEONFP8

#if NK_TARGET_RVV
/** @copydoc nk_dots_packed_size_e2m3 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e2m3_rvv(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_e2m3 */
NK_PUBLIC void nk_dots_pack_e2m3_rvv(nk_e2m3_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                     void *b_packed);
/** @copydoc nk_dots_packed_e2m3 */
NK_PUBLIC void nk_dots_packed_e2m3_rvv(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                       nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e2m3 */
NK_PUBLIC void nk_dots_symmetric_e2m3_rvv(nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_e3m2 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e3m2_rvv(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_e3m2 */
NK_PUBLIC void nk_dots_pack_e3m2_rvv(nk_e3m2_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                     void *b_packed);
/** @copydoc nk_dots_packed_e3m2 */
NK_PUBLIC void nk_dots_packed_e3m2_rvv(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                       nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e3m2 */
NK_PUBLIC void nk_dots_symmetric_e3m2_rvv(nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_f32 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f32_rvv(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_f32 */
NK_PUBLIC void nk_dots_pack_f32_rvv(nk_f32_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                    void *b_packed);
/** @copydoc nk_dots_packed_f32 */
NK_PUBLIC void nk_dots_packed_f32_rvv(nk_f32_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t height,
                                      nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f32 */
NK_PUBLIC void nk_dots_symmetric_f32_rvv(nk_f32_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                         nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                         nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_f64 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f64_rvv(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_f64 */
NK_PUBLIC void nk_dots_pack_f64_rvv(nk_f64_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                    void *b_packed);
/** @copydoc nk_dots_packed_f64 */
NK_PUBLIC void nk_dots_packed_f64_rvv(nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t height,
                                      nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f64 */
NK_PUBLIC void nk_dots_symmetric_f64_rvv(nk_f64_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                         nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                         nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_dots_packed_size_bf16_rvv(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_bf16 */
NK_PUBLIC void nk_dots_pack_bf16_rvv(nk_bf16_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                     void *b_packed);
/** @copydoc nk_dots_packed_bf16 */
NK_PUBLIC void nk_dots_packed_bf16_rvv(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                       nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_bf16 */
NK_PUBLIC void nk_dots_symmetric_bf16_rvv(nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_f16 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f16_rvv(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_f16 */
NK_PUBLIC void nk_dots_pack_f16_rvv(nk_f16_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                    void *b_packed);
/** @copydoc nk_dots_packed_f16 */
NK_PUBLIC void nk_dots_packed_f16_rvv(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                      nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f16 */
NK_PUBLIC void nk_dots_symmetric_f16_rvv(nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                         nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                         nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_i8 */
NK_PUBLIC nk_size_t nk_dots_packed_size_i8_rvv(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_i8 */
NK_PUBLIC void nk_dots_pack_i8_rvv(nk_i8_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                   void *b_packed);
/** @copydoc nk_dots_packed_i8 */
NK_PUBLIC void nk_dots_packed_i8_rvv(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t height,
                                     nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_i8 */
NK_PUBLIC void nk_dots_symmetric_i8_rvv(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                        nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                        nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_u8 */
NK_PUBLIC nk_size_t nk_dots_packed_size_u8_rvv(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_u8 */
NK_PUBLIC void nk_dots_pack_u8_rvv(nk_u8_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                   void *b_packed);
/** @copydoc nk_dots_packed_u8 */
NK_PUBLIC void nk_dots_packed_u8_rvv(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t height,
                                     nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_u8 */
NK_PUBLIC void nk_dots_symmetric_u8_rvv(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                        nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                        nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_e4m3 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e4m3_rvv(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_e4m3 */
NK_PUBLIC void nk_dots_pack_e4m3_rvv(nk_e4m3_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                     void *b_packed);
/** @copydoc nk_dots_packed_e4m3 */
NK_PUBLIC void nk_dots_packed_e4m3_rvv(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                       nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e4m3 */
NK_PUBLIC void nk_dots_symmetric_e4m3_rvv(nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_e5m2 */
NK_PUBLIC nk_size_t nk_dots_packed_size_e5m2_rvv(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_e5m2 */
NK_PUBLIC void nk_dots_pack_e5m2_rvv(nk_e5m2_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                     void *b_packed);
/** @copydoc nk_dots_packed_e5m2 */
NK_PUBLIC void nk_dots_packed_e5m2_rvv(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                       nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_e5m2 */
NK_PUBLIC void nk_dots_symmetric_e5m2_rvv(nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_RVV

/*  Loongson LASX backends using 256-bit SIMD (LoongArch).
 */
#if NK_TARGET_LOONGSONASX
/** @copydoc nk_dots_packed_size_f32 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f32_loongsonasx(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_f32 */
NK_PUBLIC void nk_dots_pack_f32_loongsonasx(nk_f32_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                            void *b_packed);
/** @copydoc nk_dots_packed_f32 */
NK_PUBLIC void nk_dots_packed_f32_loongsonasx(nk_f32_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t height,
                                              nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f32 */
NK_PUBLIC void nk_dots_symmetric_f32_loongsonasx(nk_f32_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                 nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                 nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_f64 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f64_loongsonasx(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_f64 */
NK_PUBLIC void nk_dots_pack_f64_loongsonasx(nk_f64_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                            void *b_packed);
/** @copydoc nk_dots_packed_f64 */
NK_PUBLIC void nk_dots_packed_f64_loongsonasx(nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t height,
                                              nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f64 */
NK_PUBLIC void nk_dots_symmetric_f64_loongsonasx(nk_f64_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                 nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                 nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_f16 */
NK_PUBLIC nk_size_t nk_dots_packed_size_f16_loongsonasx(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_f16 */
NK_PUBLIC void nk_dots_pack_f16_loongsonasx(nk_f16_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                            void *b_packed);
/** @copydoc nk_dots_packed_f16 */
NK_PUBLIC void nk_dots_packed_f16_loongsonasx(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                              nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_f16 */
NK_PUBLIC void nk_dots_symmetric_f16_loongsonasx(nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                 nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                 nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_dots_packed_size_bf16_loongsonasx(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_bf16 */
NK_PUBLIC void nk_dots_pack_bf16_loongsonasx(nk_bf16_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed);
/** @copydoc nk_dots_packed_bf16 */
NK_PUBLIC void nk_dots_packed_bf16_loongsonasx(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                               nk_size_t width, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_bf16 */
NK_PUBLIC void nk_dots_symmetric_bf16_loongsonasx(nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_i8 */
NK_PUBLIC nk_size_t nk_dots_packed_size_i8_loongsonasx(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_i8 */
NK_PUBLIC void nk_dots_pack_i8_loongsonasx(nk_i8_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                           void *b_packed);
/** @copydoc nk_dots_packed_i8 */
NK_PUBLIC void nk_dots_packed_i8_loongsonasx(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t height,
                                             nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_i8 */
NK_PUBLIC void nk_dots_symmetric_i8_loongsonasx(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_u8 */
NK_PUBLIC nk_size_t nk_dots_packed_size_u8_loongsonasx(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_u8 */
NK_PUBLIC void nk_dots_pack_u8_loongsonasx(nk_u8_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                           void *b_packed);
/** @copydoc nk_dots_packed_u8 */
NK_PUBLIC void nk_dots_packed_u8_loongsonasx(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t height,
                                             nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_u8 */
NK_PUBLIC void nk_dots_symmetric_u8_loongsonasx(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_dots_packed_size_u1 */
NK_PUBLIC nk_size_t nk_dots_packed_size_u1_loongsonasx(nk_size_t width, nk_size_t depth);
/** @copydoc nk_dots_pack_u1 */
NK_PUBLIC void nk_dots_pack_u1_loongsonasx(nk_u1x8_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                           void *b_packed);
/** @copydoc nk_dots_packed_u1 */
NK_PUBLIC void nk_dots_packed_u1_loongsonasx(nk_u1x8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t height,
                                             nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
/** @copydoc nk_dots_symmetric_u1 */
NK_PUBLIC void nk_dots_symmetric_u1_loongsonasx(nk_u1x8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_LOONGSONASX

#if defined(__cplusplus)
} // extern "C"
#endif

#include "numkong/dots/serial.h"
#include "numkong/dots/haswell.h"
#include "numkong/dots/skylake.h"
#include "numkong/dots/icelake.h"
#include "numkong/dots/alder.h"
#include "numkong/dots/sierra.h"
#include "numkong/dots/genoa.h"
#include "numkong/dots/diamond.h"
#include "numkong/dots/sapphireamx.h"
#include "numkong/dots/graniteamx.h"
#include "numkong/dots/neon.h"
#include "numkong/dots/neonsdot.h"
#include "numkong/dots/neonfhm.h"
#include "numkong/dots/neonfp8.h"
#include "numkong/dots/neonbfdot.h"
#include "numkong/dots/sme.h"
#include "numkong/dots/smef64.h"
#include "numkong/dots/smebi32.h"
#include "numkong/dots/rvv.h"
#include "numkong/dots/powervsx.h"
#include "numkong/dots/v128relaxed.h"
#include "numkong/dots/loongsonasx.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if !NK_DYNAMIC_DISPATCH

NK_PUBLIC nk_size_t nk_dots_packed_size_f32(nk_size_t width, nk_size_t depth) {
#if NK_TARGET_SMEF64
    return nk_dots_packed_size_f32_smef64(width, depth);
#elif NK_TARGET_SKYLAKE
    return nk_dots_packed_size_f32_skylake(width, depth);
#elif NK_TARGET_HASWELL
    return nk_dots_packed_size_f32_haswell(width, depth);
#elif NK_TARGET_NEON
    return nk_dots_packed_size_f32_neon(width, depth);
#elif NK_TARGET_POWERVSX
    return nk_dots_packed_size_f32_powervsx(width, depth);
#elif NK_TARGET_RVV
    return nk_dots_packed_size_f32_rvv(width, depth);
#elif NK_TARGET_V128RELAXED
    return nk_dots_packed_size_f32_v128relaxed(width, depth);
#else
    return nk_dots_packed_size_f32_serial(width, depth);
#endif
}

NK_PUBLIC void nk_dots_pack_f32(nk_f32_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                void *b_packed) {
#if NK_TARGET_SMEF64
    nk_dots_pack_f32_smef64(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_SKYLAKE
    nk_dots_pack_f32_skylake(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_HASWELL
    nk_dots_pack_f32_haswell(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_NEON
    nk_dots_pack_f32_neon(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_POWERVSX
    nk_dots_pack_f32_powervsx(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_RVV
    nk_dots_pack_f32_rvv(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_V128RELAXED
    nk_dots_pack_f32_v128relaxed(b, width, depth, b_stride, b_packed);
#else
    nk_dots_pack_f32_serial(b, width, depth, b_stride, b_packed);
#endif
}

NK_PUBLIC void nk_dots_packed_f32(nk_f32_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t height,
                                  nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride) {
#if NK_TARGET_SMEF64
    nk_dots_packed_f32_smef64(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_SKYLAKE
    nk_dots_packed_f32_skylake(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_HASWELL
    nk_dots_packed_f32_haswell(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_NEON
    nk_dots_packed_f32_neon(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_POWERVSX
    nk_dots_packed_f32_powervsx(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_RVV
    nk_dots_packed_f32_rvv(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_V128RELAXED
    nk_dots_packed_f32_v128relaxed(a, b_packed, c, height, width, depth, a_stride, c_stride);
#else
    nk_dots_packed_f32_serial(a, b_packed, c, height, width, depth, a_stride, c_stride);
#endif
}

NK_PUBLIC nk_size_t nk_dots_packed_size_f64(nk_size_t width, nk_size_t depth) {
#if NK_TARGET_SMEF64
    return nk_dots_packed_size_f64_smef64(width, depth);
#elif NK_TARGET_SKYLAKE
    return nk_dots_packed_size_f64_skylake(width, depth);
#elif NK_TARGET_HASWELL
    return nk_dots_packed_size_f64_haswell(width, depth);
#elif NK_TARGET_NEON
    return nk_dots_packed_size_f64_neon(width, depth);
#elif NK_TARGET_POWERVSX
    return nk_dots_packed_size_f64_powervsx(width, depth);
#elif NK_TARGET_RVV
    return nk_dots_packed_size_f64_rvv(width, depth);
#elif NK_TARGET_V128RELAXED
    return nk_dots_packed_size_f64_v128relaxed(width, depth);
#else
    return nk_dots_packed_size_f64_serial(width, depth);
#endif
}

NK_PUBLIC void nk_dots_pack_f64(nk_f64_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                void *b_packed) {
#if NK_TARGET_SMEF64
    nk_dots_pack_f64_smef64(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_SKYLAKE
    nk_dots_pack_f64_skylake(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_HASWELL
    nk_dots_pack_f64_haswell(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_NEON
    nk_dots_pack_f64_neon(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_POWERVSX
    nk_dots_pack_f64_powervsx(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_RVV
    nk_dots_pack_f64_rvv(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_V128RELAXED
    nk_dots_pack_f64_v128relaxed(b, width, depth, b_stride, b_packed);
#else
    nk_dots_pack_f64_serial(b, width, depth, b_stride, b_packed);
#endif
}

NK_PUBLIC void nk_dots_packed_f64(nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t height,
                                  nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride) {
#if NK_TARGET_SMEF64
    nk_dots_packed_f64_smef64(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_SKYLAKE
    nk_dots_packed_f64_skylake(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_HASWELL
    nk_dots_packed_f64_haswell(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_NEON
    nk_dots_packed_f64_neon(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_POWERVSX
    nk_dots_packed_f64_powervsx(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_RVV
    nk_dots_packed_f64_rvv(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_V128RELAXED
    nk_dots_packed_f64_v128relaxed(a, b_packed, c, height, width, depth, a_stride, c_stride);
#else
    nk_dots_packed_f64_serial(a, b_packed, c, height, width, depth, a_stride, c_stride);
#endif
}

NK_PUBLIC nk_size_t nk_dots_packed_size_f16(nk_size_t width, nk_size_t depth) {
#if NK_TARGET_GRANITEAMX
    return nk_dots_packed_size_f16_graniteamx(width, depth);
#elif NK_TARGET_SME
    return nk_dots_packed_size_f16_sme(width, depth);
#elif NK_TARGET_NEONFHM
    return nk_dots_packed_size_f16_neonfhm(width, depth);
#elif NK_TARGET_NEON
    return nk_dots_packed_size_f16_neon(width, depth);
#elif NK_TARGET_SKYLAKE
    return nk_dots_packed_size_f16_skylake(width, depth);
#elif NK_TARGET_HASWELL
    return nk_dots_packed_size_f16_haswell(width, depth);
#elif NK_TARGET_POWERVSX
    return nk_dots_packed_size_f16_powervsx(width, depth);
#elif NK_TARGET_RVV
    return nk_dots_packed_size_f16_rvv(width, depth);
#else
    return nk_dots_packed_size_f16_serial(width, depth);
#endif
}

NK_PUBLIC void nk_dots_pack_f16(nk_f16_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                void *b_packed) {
#if NK_TARGET_GRANITEAMX
    nk_dots_pack_f16_graniteamx(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_SME
    nk_dots_pack_f16_sme(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_NEONFHM
    nk_dots_pack_f16_neonfhm(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_NEON
    nk_dots_pack_f16_neon(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_SKYLAKE
    nk_dots_pack_f16_skylake(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_HASWELL
    nk_dots_pack_f16_haswell(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_POWERVSX
    nk_dots_pack_f16_powervsx(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_RVV
    nk_dots_pack_f16_rvv(b, width, depth, b_stride, b_packed);
#else
    nk_dots_pack_f16_serial(b, width, depth, b_stride, b_packed);
#endif
}

NK_PUBLIC void nk_dots_packed_f16(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                  nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride) {
#if NK_TARGET_GRANITEAMX
    nk_dots_packed_f16_graniteamx(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_SME
    nk_dots_packed_f16_sme(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_NEONFHM
    nk_dots_packed_f16_neonfhm(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_NEON
    nk_dots_packed_f16_neon(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_SKYLAKE
    nk_dots_packed_f16_skylake(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_HASWELL
    nk_dots_packed_f16_haswell(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_POWERVSX
    nk_dots_packed_f16_powervsx(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_RVV
    nk_dots_packed_f16_rvv(a, b_packed, c, height, width, depth, a_stride, c_stride);
#else
    nk_dots_packed_f16_serial(a, b_packed, c, height, width, depth, a_stride, c_stride);
#endif
}

NK_PUBLIC nk_size_t nk_dots_packed_size_bf16(nk_size_t width, nk_size_t depth) {
#if NK_TARGET_SME
    return nk_dots_packed_size_bf16_sme(width, depth);
#elif NK_TARGET_SAPPHIREAMX
    return nk_dots_packed_size_bf16_sapphireamx(width, depth);
#elif NK_TARGET_NEONBFDOT
    return nk_dots_packed_size_bf16_neonbfdot(width, depth);
#elif NK_TARGET_GENOA
    return nk_dots_packed_size_bf16_genoa(width, depth);
#elif NK_TARGET_SKYLAKE
    return nk_dots_packed_size_bf16_skylake(width, depth);
#elif NK_TARGET_HASWELL
    return nk_dots_packed_size_bf16_haswell(width, depth);
#elif NK_TARGET_POWERVSX
    return nk_dots_packed_size_bf16_powervsx(width, depth);
#elif NK_TARGET_RVV
    return nk_dots_packed_size_bf16_rvv(width, depth);
#elif NK_TARGET_V128RELAXED
    return nk_dots_packed_size_bf16_v128relaxed(width, depth);
#else
    return nk_dots_packed_size_bf16_serial(width, depth);
#endif
}

NK_PUBLIC void nk_dots_pack_bf16(nk_bf16_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                 void *b_packed) {
#if NK_TARGET_SME
    nk_dots_pack_bf16_sme(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_SAPPHIREAMX
    nk_dots_pack_bf16_sapphireamx(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_NEONBFDOT
    nk_dots_pack_bf16_neonbfdot(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_GENOA
    nk_dots_pack_bf16_genoa(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_SKYLAKE
    nk_dots_pack_bf16_skylake(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_HASWELL
    nk_dots_pack_bf16_haswell(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_POWERVSX
    nk_dots_pack_bf16_powervsx(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_RVV
    nk_dots_pack_bf16_rvv(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_V128RELAXED
    nk_dots_pack_bf16_v128relaxed(b, width, depth, b_stride, b_packed);
#else
    nk_dots_pack_bf16_serial(b, width, depth, b_stride, b_packed);
#endif
}

NK_PUBLIC void nk_dots_packed_bf16(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                   nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride) {
#if NK_TARGET_SME
    nk_dots_packed_bf16_sme(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_SAPPHIREAMX
    nk_dots_packed_bf16_sapphireamx(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_NEONBFDOT
    nk_dots_packed_bf16_neonbfdot(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_GENOA
    nk_dots_packed_bf16_genoa(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_SKYLAKE
    nk_dots_packed_bf16_skylake(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_HASWELL
    nk_dots_packed_bf16_haswell(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_POWERVSX
    nk_dots_packed_bf16_powervsx(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_RVV
    nk_dots_packed_bf16_rvv(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_V128RELAXED
    nk_dots_packed_bf16_v128relaxed(a, b_packed, c, height, width, depth, a_stride, c_stride);
#else
    nk_dots_packed_bf16_serial(a, b_packed, c, height, width, depth, a_stride, c_stride);
#endif
}

NK_PUBLIC nk_size_t nk_dots_packed_size_i8(nk_size_t width, nk_size_t depth) {
#if NK_TARGET_SME
    return nk_dots_packed_size_i8_sme(width, depth);
#elif NK_TARGET_SAPPHIREAMX
    return nk_dots_packed_size_i8_sapphireamx(width, depth);
#elif NK_TARGET_NEONSDOT
    return nk_dots_packed_size_i8_neonsdot(width, depth);
#elif NK_TARGET_ICELAKE
    return nk_dots_packed_size_i8_icelake(width, depth);
#elif NK_TARGET_SIERRA
    return nk_dots_packed_size_i8_sierra(width, depth);
#elif NK_TARGET_ALDER
    return nk_dots_packed_size_i8_alder(width, depth);
#elif NK_TARGET_HASWELL
    return nk_dots_packed_size_i8_haswell(width, depth);
#elif NK_TARGET_POWERVSX
    return nk_dots_packed_size_i8_powervsx(width, depth);
#elif NK_TARGET_RVV
    return nk_dots_packed_size_i8_rvv(width, depth);
#elif NK_TARGET_V128RELAXED
    return nk_dots_packed_size_i8_v128relaxed(width, depth);
#else
    return nk_dots_packed_size_i8_serial(width, depth);
#endif
}

NK_PUBLIC void nk_dots_pack_i8(nk_i8_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride, void *b_packed) {
#if NK_TARGET_SME
    nk_dots_pack_i8_sme(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_SAPPHIREAMX
    nk_dots_pack_i8_sapphireamx(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_NEONSDOT
    nk_dots_pack_i8_neonsdot(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_ICELAKE
    nk_dots_pack_i8_icelake(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_SIERRA
    nk_dots_pack_i8_sierra(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_ALDER
    nk_dots_pack_i8_alder(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_HASWELL
    nk_dots_pack_i8_haswell(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_POWERVSX
    nk_dots_pack_i8_powervsx(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_RVV
    nk_dots_pack_i8_rvv(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_V128RELAXED
    nk_dots_pack_i8_v128relaxed(b, width, depth, b_stride, b_packed);
#else
    nk_dots_pack_i8_serial(b, width, depth, b_stride, b_packed);
#endif
}

NK_PUBLIC void nk_dots_packed_i8(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t height, nk_size_t width,
                                 nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride) {
#if NK_TARGET_SME
    nk_dots_packed_i8_sme(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_SAPPHIREAMX
    nk_dots_packed_i8_sapphireamx(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_NEONSDOT
    nk_dots_packed_i8_neonsdot(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_ICELAKE
    nk_dots_packed_i8_icelake(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_SIERRA
    nk_dots_packed_i8_sierra(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_ALDER
    nk_dots_packed_i8_alder(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_HASWELL
    nk_dots_packed_i8_haswell(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_POWERVSX
    nk_dots_packed_i8_powervsx(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_RVV
    nk_dots_packed_i8_rvv(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_V128RELAXED
    nk_dots_packed_i8_v128relaxed(a, b_packed, c, height, width, depth, a_stride, c_stride);
#else
    nk_dots_packed_i8_serial(a, b_packed, c, height, width, depth, a_stride, c_stride);
#endif
}

NK_PUBLIC nk_size_t nk_dots_packed_size_u8(nk_size_t width, nk_size_t depth) {
#if NK_TARGET_SME
    return nk_dots_packed_size_u8_sme(width, depth);
#elif NK_TARGET_SAPPHIREAMX
    return nk_dots_packed_size_u8_sapphireamx(width, depth);
#elif NK_TARGET_NEONSDOT
    return nk_dots_packed_size_u8_neonsdot(width, depth);
#elif NK_TARGET_ICELAKE
    return nk_dots_packed_size_u8_icelake(width, depth);
#elif NK_TARGET_SIERRA
    return nk_dots_packed_size_u8_sierra(width, depth);
#elif NK_TARGET_ALDER
    return nk_dots_packed_size_u8_alder(width, depth);
#elif NK_TARGET_HASWELL
    return nk_dots_packed_size_u8_haswell(width, depth);
#elif NK_TARGET_POWERVSX
    return nk_dots_packed_size_u8_powervsx(width, depth);
#elif NK_TARGET_RVV
    return nk_dots_packed_size_u8_rvv(width, depth);
#elif NK_TARGET_V128RELAXED
    return nk_dots_packed_size_u8_v128relaxed(width, depth);
#else
    return nk_dots_packed_size_u8_serial(width, depth);
#endif
}

NK_PUBLIC void nk_dots_pack_u8(nk_u8_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride, void *b_packed) {
#if NK_TARGET_SME
    nk_dots_pack_u8_sme(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_SAPPHIREAMX
    nk_dots_pack_u8_sapphireamx(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_NEONSDOT
    nk_dots_pack_u8_neonsdot(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_ICELAKE
    nk_dots_pack_u8_icelake(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_SIERRA
    nk_dots_pack_u8_sierra(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_ALDER
    nk_dots_pack_u8_alder(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_HASWELL
    nk_dots_pack_u8_haswell(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_POWERVSX
    nk_dots_pack_u8_powervsx(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_RVV
    nk_dots_pack_u8_rvv(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_V128RELAXED
    nk_dots_pack_u8_v128relaxed(b, width, depth, b_stride, b_packed);
#else
    nk_dots_pack_u8_serial(b, width, depth, b_stride, b_packed);
#endif
}

NK_PUBLIC void nk_dots_packed_u8(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t height, nk_size_t width,
                                 nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride) {
#if NK_TARGET_SME
    nk_dots_packed_u8_sme(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_SAPPHIREAMX
    nk_dots_packed_u8_sapphireamx(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_NEONSDOT
    nk_dots_packed_u8_neonsdot(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_ICELAKE
    nk_dots_packed_u8_icelake(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_SIERRA
    nk_dots_packed_u8_sierra(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_ALDER
    nk_dots_packed_u8_alder(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_HASWELL
    nk_dots_packed_u8_haswell(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_POWERVSX
    nk_dots_packed_u8_powervsx(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_RVV
    nk_dots_packed_u8_rvv(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_V128RELAXED
    nk_dots_packed_u8_v128relaxed(a, b_packed, c, height, width, depth, a_stride, c_stride);
#else
    nk_dots_packed_u8_serial(a, b_packed, c, height, width, depth, a_stride, c_stride);
#endif
}

NK_PUBLIC nk_size_t nk_dots_packed_size_e4m3(nk_size_t width, nk_size_t depth) {
#if NK_TARGET_SME
    return nk_dots_packed_size_e4m3_sme(width, depth);
#elif NK_TARGET_SAPPHIREAMX
    return nk_dots_packed_size_e4m3_sapphireamx(width, depth);
#elif NK_TARGET_NEONFP8
    return nk_dots_packed_size_e4m3_neonfp8(width, depth);
#elif NK_TARGET_NEONFHM
    return nk_dots_packed_size_e4m3_neonfhm(width, depth);
#elif NK_TARGET_DIAMOND
    return nk_dots_packed_size_e4m3_diamond(width, depth);
#elif NK_TARGET_GENOA
    return nk_dots_packed_size_e4m3_genoa(width, depth);
#elif NK_TARGET_SKYLAKE
    return nk_dots_packed_size_e4m3_skylake(width, depth);
#elif NK_TARGET_HASWELL
    return nk_dots_packed_size_e4m3_haswell(width, depth);
#elif NK_TARGET_RVV
    return nk_dots_packed_size_e4m3_rvv(width, depth);
#elif NK_TARGET_V128RELAXED
    return nk_dots_packed_size_e4m3_v128relaxed(width, depth);
#else
    return nk_dots_packed_size_e4m3_serial(width, depth);
#endif
}

NK_PUBLIC void nk_dots_pack_e4m3(nk_e4m3_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                 void *b_packed) {
#if NK_TARGET_SME
    nk_dots_pack_e4m3_sme(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_SAPPHIREAMX
    nk_dots_pack_e4m3_sapphireamx(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_NEONFP8
    nk_dots_pack_e4m3_neonfp8(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_NEONFHM
    nk_dots_pack_e4m3_neonfhm(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_DIAMOND
    nk_dots_pack_e4m3_diamond(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_GENOA
    nk_dots_pack_e4m3_genoa(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_SKYLAKE
    nk_dots_pack_e4m3_skylake(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_HASWELL
    nk_dots_pack_e4m3_haswell(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_RVV
    nk_dots_pack_e4m3_rvv(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_V128RELAXED
    nk_dots_pack_e4m3_v128relaxed(b, width, depth, b_stride, b_packed);
#else
    nk_dots_pack_e4m3_serial(b, width, depth, b_stride, b_packed);
#endif
}

NK_PUBLIC void nk_dots_packed_e4m3(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                   nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride) {
#if NK_TARGET_SME
    nk_dots_packed_e4m3_sme(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_SAPPHIREAMX
    nk_dots_packed_e4m3_sapphireamx(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_NEONFP8
    nk_dots_packed_e4m3_neonfp8(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_NEONFHM
    nk_dots_packed_e4m3_neonfhm(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_DIAMOND
    nk_dots_packed_e4m3_diamond(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_GENOA
    nk_dots_packed_e4m3_genoa(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_SKYLAKE
    nk_dots_packed_e4m3_skylake(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_HASWELL
    nk_dots_packed_e4m3_haswell(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_RVV
    nk_dots_packed_e4m3_rvv(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_V128RELAXED
    nk_dots_packed_e4m3_v128relaxed(a, b_packed, c, height, width, depth, a_stride, c_stride);
#else
    nk_dots_packed_e4m3_serial(a, b_packed, c, height, width, depth, a_stride, c_stride);
#endif
}

NK_PUBLIC nk_size_t nk_dots_packed_size_e5m2(nk_size_t width, nk_size_t depth) {
#if NK_TARGET_SME
    return nk_dots_packed_size_e5m2_sme(width, depth);
#elif NK_TARGET_SAPPHIREAMX
    return nk_dots_packed_size_e5m2_sapphireamx(width, depth);
#elif NK_TARGET_NEONFP8
    return nk_dots_packed_size_e5m2_neonfp8(width, depth);
#elif NK_TARGET_NEONFHM
    return nk_dots_packed_size_e5m2_neonfhm(width, depth);
#elif NK_TARGET_DIAMOND
    return nk_dots_packed_size_e5m2_diamond(width, depth);
#elif NK_TARGET_GENOA
    return nk_dots_packed_size_e5m2_genoa(width, depth);
#elif NK_TARGET_SKYLAKE
    return nk_dots_packed_size_e5m2_skylake(width, depth);
#elif NK_TARGET_HASWELL
    return nk_dots_packed_size_e5m2_haswell(width, depth);
#elif NK_TARGET_RVV
    return nk_dots_packed_size_e5m2_rvv(width, depth);
#elif NK_TARGET_V128RELAXED
    return nk_dots_packed_size_e5m2_v128relaxed(width, depth);
#else
    return nk_dots_packed_size_e5m2_serial(width, depth);
#endif
}

NK_PUBLIC void nk_dots_pack_e5m2(nk_e5m2_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                 void *b_packed) {
#if NK_TARGET_SME
    nk_dots_pack_e5m2_sme(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_SAPPHIREAMX
    nk_dots_pack_e5m2_sapphireamx(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_NEONFP8
    nk_dots_pack_e5m2_neonfp8(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_NEONFHM
    nk_dots_pack_e5m2_neonfhm(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_DIAMOND
    nk_dots_pack_e5m2_diamond(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_GENOA
    nk_dots_pack_e5m2_genoa(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_SKYLAKE
    nk_dots_pack_e5m2_skylake(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_HASWELL
    nk_dots_pack_e5m2_haswell(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_RVV
    nk_dots_pack_e5m2_rvv(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_V128RELAXED
    nk_dots_pack_e5m2_v128relaxed(b, width, depth, b_stride, b_packed);
#else
    nk_dots_pack_e5m2_serial(b, width, depth, b_stride, b_packed);
#endif
}

NK_PUBLIC void nk_dots_packed_e5m2(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                   nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride) {
#if NK_TARGET_SME
    nk_dots_packed_e5m2_sme(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_SAPPHIREAMX
    nk_dots_packed_e5m2_sapphireamx(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_NEONFP8
    nk_dots_packed_e5m2_neonfp8(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_NEONFHM
    nk_dots_packed_e5m2_neonfhm(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_DIAMOND
    nk_dots_packed_e5m2_diamond(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_GENOA
    nk_dots_packed_e5m2_genoa(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_SKYLAKE
    nk_dots_packed_e5m2_skylake(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_HASWELL
    nk_dots_packed_e5m2_haswell(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_RVV
    nk_dots_packed_e5m2_rvv(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_V128RELAXED
    nk_dots_packed_e5m2_v128relaxed(a, b_packed, c, height, width, depth, a_stride, c_stride);
#else
    nk_dots_packed_e5m2_serial(a, b_packed, c, height, width, depth, a_stride, c_stride);
#endif
}

NK_PUBLIC nk_size_t nk_dots_packed_size_e2m3(nk_size_t width, nk_size_t depth) {
#if NK_TARGET_SME
    return nk_dots_packed_size_e2m3_sme(width, depth);
#elif NK_TARGET_SAPPHIREAMX
    return nk_dots_packed_size_e2m3_sapphireamx(width, depth);
#elif NK_TARGET_NEONFP8
    return nk_dots_packed_size_e2m3_neonfp8(width, depth);
#elif NK_TARGET_SKYLAKE
    return nk_dots_packed_size_e2m3_skylake(width, depth);
#elif NK_TARGET_SIERRA
    return nk_dots_packed_size_e2m3_sierra(width, depth);
#elif NK_TARGET_ALDER
    return nk_dots_packed_size_e2m3_alder(width, depth);
#elif NK_TARGET_HASWELL
    return nk_dots_packed_size_e2m3_haswell(width, depth);
#elif NK_TARGET_RVV
    return nk_dots_packed_size_e2m3_rvv(width, depth);
#elif NK_TARGET_V128RELAXED
    return nk_dots_packed_size_e2m3_v128relaxed(width, depth);
#else
    return nk_dots_packed_size_e2m3_serial(width, depth);
#endif
}

NK_PUBLIC void nk_dots_pack_e2m3(nk_e2m3_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                 void *b_packed) {
#if NK_TARGET_SME
    nk_dots_pack_e2m3_sme(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_SAPPHIREAMX
    nk_dots_pack_e2m3_sapphireamx(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_NEONFP8
    nk_dots_pack_e2m3_neonfp8(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_SKYLAKE
    nk_dots_pack_e2m3_skylake(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_SIERRA
    nk_dots_pack_e2m3_sierra(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_ALDER
    nk_dots_pack_e2m3_alder(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_HASWELL
    nk_dots_pack_e2m3_haswell(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_RVV
    nk_dots_pack_e2m3_rvv(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_V128RELAXED
    nk_dots_pack_e2m3_v128relaxed(b, width, depth, b_stride, b_packed);
#else
    nk_dots_pack_e2m3_serial(b, width, depth, b_stride, b_packed);
#endif
}

NK_PUBLIC void nk_dots_packed_e2m3(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                   nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride) {
#if NK_TARGET_SME
    nk_dots_packed_e2m3_sme(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_SAPPHIREAMX
    nk_dots_packed_e2m3_sapphireamx(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_NEONFP8
    nk_dots_packed_e2m3_neonfp8(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_SKYLAKE
    nk_dots_packed_e2m3_skylake(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_SIERRA
    nk_dots_packed_e2m3_sierra(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_ALDER
    nk_dots_packed_e2m3_alder(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_HASWELL
    nk_dots_packed_e2m3_haswell(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_RVV
    nk_dots_packed_e2m3_rvv(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_V128RELAXED
    nk_dots_packed_e2m3_v128relaxed(a, b_packed, c, height, width, depth, a_stride, c_stride);
#else
    nk_dots_packed_e2m3_serial(a, b_packed, c, height, width, depth, a_stride, c_stride);
#endif
}

NK_PUBLIC nk_size_t nk_dots_packed_size_e3m2(nk_size_t width, nk_size_t depth) {
#if NK_TARGET_SME
    return nk_dots_packed_size_e3m2_sme(width, depth);
#elif NK_TARGET_SAPPHIREAMX
    return nk_dots_packed_size_e3m2_sapphireamx(width, depth);
#elif NK_TARGET_NEONFP8
    return nk_dots_packed_size_e3m2_neonfp8(width, depth);
#elif NK_TARGET_SKYLAKE
    return nk_dots_packed_size_e3m2_skylake(width, depth);
#elif NK_TARGET_HASWELL
    return nk_dots_packed_size_e3m2_haswell(width, depth);
#elif NK_TARGET_RVV
    return nk_dots_packed_size_e3m2_rvv(width, depth);
#else
    return nk_dots_packed_size_e3m2_serial(width, depth);
#endif
}

NK_PUBLIC void nk_dots_pack_e3m2(nk_e3m2_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                 void *b_packed) {
#if NK_TARGET_SME
    nk_dots_pack_e3m2_sme(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_SAPPHIREAMX
    nk_dots_pack_e3m2_sapphireamx(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_NEONFP8
    nk_dots_pack_e3m2_neonfp8(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_SKYLAKE
    nk_dots_pack_e3m2_skylake(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_HASWELL
    nk_dots_pack_e3m2_haswell(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_RVV
    nk_dots_pack_e3m2_rvv(b, width, depth, b_stride, b_packed);
#else
    nk_dots_pack_e3m2_serial(b, width, depth, b_stride, b_packed);
#endif
}

NK_PUBLIC void nk_dots_packed_e3m2(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t height,
                                   nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride) {
#if NK_TARGET_SME
    nk_dots_packed_e3m2_sme(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_SAPPHIREAMX
    nk_dots_packed_e3m2_sapphireamx(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_NEONFP8
    nk_dots_packed_e3m2_neonfp8(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_SKYLAKE
    nk_dots_packed_e3m2_skylake(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_HASWELL
    nk_dots_packed_e3m2_haswell(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_RVV
    nk_dots_packed_e3m2_rvv(a, b_packed, c, height, width, depth, a_stride, c_stride);
#else
    nk_dots_packed_e3m2_serial(a, b_packed, c, height, width, depth, a_stride, c_stride);
#endif
}

NK_PUBLIC nk_size_t nk_dots_packed_size_u4(nk_size_t width, nk_size_t depth) {
#if NK_TARGET_SME
    return nk_dots_packed_size_u4_sme(width, depth);
#elif NK_TARGET_ICELAKE
    return nk_dots_packed_size_u4_icelake(width, depth);
#elif NK_TARGET_NEONSDOT
    return nk_dots_packed_size_u4_neonsdot(width, depth);
#elif NK_TARGET_HASWELL
    return nk_dots_packed_size_u4_haswell(width, depth);
#elif NK_TARGET_V128RELAXED
    return nk_dots_packed_size_u4_v128relaxed(width, depth);
#else
    return nk_dots_packed_size_u4_serial(width, depth);
#endif
}

NK_PUBLIC void nk_dots_pack_u4(nk_u4x2_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                               void *b_packed) {
#if NK_TARGET_SME
    nk_dots_pack_u4_sme(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_ICELAKE
    nk_dots_pack_u4_icelake(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_NEONSDOT
    nk_dots_pack_u4_neonsdot(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_HASWELL
    nk_dots_pack_u4_haswell(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_V128RELAXED
    nk_dots_pack_u4_v128relaxed(b, width, depth, b_stride, b_packed);
#else
    nk_dots_pack_u4_serial(b, width, depth, b_stride, b_packed);
#endif
}

NK_PUBLIC void nk_dots_packed_u4(nk_u4x2_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t height,
                                 nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride) {
#if NK_TARGET_SME
    nk_dots_packed_u4_sme(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_ICELAKE
    nk_dots_packed_u4_icelake(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_NEONSDOT
    nk_dots_packed_u4_neonsdot(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_HASWELL
    nk_dots_packed_u4_haswell(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_V128RELAXED
    nk_dots_packed_u4_v128relaxed(a, b_packed, c, height, width, depth, a_stride, c_stride);
#else
    nk_dots_packed_u4_serial(a, b_packed, c, height, width, depth, a_stride, c_stride);
#endif
}

NK_PUBLIC nk_size_t nk_dots_packed_size_u1(nk_size_t width, nk_size_t depth) {
#if NK_TARGET_SMEBI32
    return nk_dots_packed_size_u1_smebi32(width, depth);
#elif NK_TARGET_ICELAKE
    return nk_dots_packed_size_u1_icelake(width, depth);
#elif NK_TARGET_HASWELL
    return nk_dots_packed_size_u1_haswell(width, depth);
#elif NK_TARGET_NEON
    return nk_dots_packed_size_u1_neon(width, depth);
#elif NK_TARGET_POWERVSX
    return nk_dots_packed_size_u1_powervsx(width, depth);
#elif NK_TARGET_LOONGSONASX
    return nk_dots_packed_size_u1_loongsonasx(width, depth);
#elif NK_TARGET_V128RELAXED
    return nk_dots_packed_size_u1_v128relaxed(width, depth);
#else
    return nk_dots_packed_size_u1_serial(width, depth);
#endif
}

NK_PUBLIC void nk_dots_pack_u1(nk_u1x8_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                               void *b_packed) {
#if NK_TARGET_SMEBI32
    nk_dots_pack_u1_smebi32(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_ICELAKE
    nk_dots_pack_u1_icelake(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_HASWELL
    nk_dots_pack_u1_haswell(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_NEON
    nk_dots_pack_u1_neon(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_POWERVSX
    nk_dots_pack_u1_powervsx(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_LOONGSONASX
    nk_dots_pack_u1_loongsonasx(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_V128RELAXED
    nk_dots_pack_u1_v128relaxed(b, width, depth, b_stride, b_packed);
#else
    nk_dots_pack_u1_serial(b, width, depth, b_stride, b_packed);
#endif
}

NK_PUBLIC void nk_dots_packed_u1(nk_u1x8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t height,
                                 nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride) {
#if NK_TARGET_SMEBI32
    nk_dots_packed_u1_smebi32(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_ICELAKE
    nk_dots_packed_u1_icelake(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_HASWELL
    nk_dots_packed_u1_haswell(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_NEON
    nk_dots_packed_u1_neon(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_POWERVSX
    nk_dots_packed_u1_powervsx(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_LOONGSONASX
    nk_dots_packed_u1_loongsonasx(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_V128RELAXED
    nk_dots_packed_u1_v128relaxed(a, b_packed, c, height, width, depth, a_stride, c_stride);
#else
    nk_dots_packed_u1_serial(a, b_packed, c, height, width, depth, a_stride, c_stride);
#endif
}

NK_PUBLIC nk_size_t nk_dots_packed_size_i4(nk_size_t width, nk_size_t depth) {
#if NK_TARGET_SME
    return nk_dots_packed_size_i4_sme(width, depth);
#elif NK_TARGET_ICELAKE
    return nk_dots_packed_size_i4_icelake(width, depth);
#elif NK_TARGET_NEONSDOT
    return nk_dots_packed_size_i4_neonsdot(width, depth);
#elif NK_TARGET_HASWELL
    return nk_dots_packed_size_i4_haswell(width, depth);
#elif NK_TARGET_V128RELAXED
    return nk_dots_packed_size_i4_v128relaxed(width, depth);
#else
    return nk_dots_packed_size_i4_serial(width, depth);
#endif
}

NK_PUBLIC void nk_dots_pack_i4(nk_i4x2_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                               void *b_packed) {
#if NK_TARGET_SME
    nk_dots_pack_i4_sme(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_ICELAKE
    nk_dots_pack_i4_icelake(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_NEONSDOT
    nk_dots_pack_i4_neonsdot(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_HASWELL
    nk_dots_pack_i4_haswell(b, width, depth, b_stride, b_packed);
#elif NK_TARGET_V128RELAXED
    nk_dots_pack_i4_v128relaxed(b, width, depth, b_stride, b_packed);
#else
    nk_dots_pack_i4_serial(b, width, depth, b_stride, b_packed);
#endif
}

NK_PUBLIC void nk_dots_packed_i4(nk_i4x2_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t height,
                                 nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride) {
#if NK_TARGET_SME
    nk_dots_packed_i4_sme(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_ICELAKE
    nk_dots_packed_i4_icelake(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_NEONSDOT
    nk_dots_packed_i4_neonsdot(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_HASWELL
    nk_dots_packed_i4_haswell(a, b_packed, c, height, width, depth, a_stride, c_stride);
#elif NK_TARGET_V128RELAXED
    nk_dots_packed_i4_v128relaxed(a, b_packed, c, height, width, depth, a_stride, c_stride);
#else
    nk_dots_packed_i4_serial(a, b_packed, c, height, width, depth, a_stride, c_stride);
#endif
}

NK_PUBLIC void nk_dots_symmetric_f16(nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                     nk_size_t stride, nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                     nk_size_t row_count) {
#if NK_TARGET_SME
    nk_dots_symmetric_f16_sme(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_NEON
    nk_dots_symmetric_f16_neon(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_NEONFHM
    nk_dots_symmetric_f16_neonfhm(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_SKYLAKE
    nk_dots_symmetric_f16_skylake(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_HASWELL
    nk_dots_symmetric_f16_haswell(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_POWERVSX
    nk_dots_symmetric_f16_powervsx(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_RVV
    nk_dots_symmetric_f16_rvv(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#else
    nk_dots_symmetric_f16_serial(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#endif
}

NK_PUBLIC void nk_dots_symmetric_bf16(nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                      nk_size_t row_count) {
#if NK_TARGET_SME
    nk_dots_symmetric_bf16_sme(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_SAPPHIREAMX
    nk_dots_symmetric_bf16_sapphireamx(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#elif NK_TARGET_NEONBFDOT
    nk_dots_symmetric_bf16_neonbfdot(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#elif NK_TARGET_GENOA
    nk_dots_symmetric_bf16_genoa(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_SKYLAKE
    nk_dots_symmetric_bf16_skylake(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_HASWELL
    nk_dots_symmetric_bf16_haswell(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_POWERVSX
    nk_dots_symmetric_bf16_powervsx(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_RVV
    nk_dots_symmetric_bf16_rvv(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_V128RELAXED
    nk_dots_symmetric_bf16_v128relaxed(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#else
    nk_dots_symmetric_bf16_serial(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#endif
}

NK_PUBLIC void nk_dots_symmetric_i8(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride,
                                    nk_i32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                    nk_size_t row_count) {
#if NK_TARGET_SME
    nk_dots_symmetric_i8_sme(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_SAPPHIREAMX
    nk_dots_symmetric_i8_sapphireamx(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#elif NK_TARGET_NEONSDOT
    nk_dots_symmetric_i8_neonsdot(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_ICELAKE
    nk_dots_symmetric_i8_icelake(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_SIERRA
    nk_dots_symmetric_i8_sierra(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_ALDER
    nk_dots_symmetric_i8_alder(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_HASWELL
    nk_dots_symmetric_i8_haswell(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_POWERVSX
    nk_dots_symmetric_i8_powervsx(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_RVV
    nk_dots_symmetric_i8_rvv(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_V128RELAXED
    nk_dots_symmetric_i8_v128relaxed(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#else
    nk_dots_symmetric_i8_serial(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#endif
}

NK_PUBLIC void nk_dots_symmetric_u8(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride,
                                    nk_u32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                    nk_size_t row_count) {
#if NK_TARGET_SME
    nk_dots_symmetric_u8_sme(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_SAPPHIREAMX
    nk_dots_symmetric_u8_sapphireamx(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#elif NK_TARGET_ICELAKE
    nk_dots_symmetric_u8_icelake(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_SIERRA
    nk_dots_symmetric_u8_sierra(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_ALDER
    nk_dots_symmetric_u8_alder(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_NEONSDOT
    nk_dots_symmetric_u8_neonsdot(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_HASWELL
    nk_dots_symmetric_u8_haswell(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_POWERVSX
    nk_dots_symmetric_u8_powervsx(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_RVV
    nk_dots_symmetric_u8_rvv(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_V128RELAXED
    nk_dots_symmetric_u8_v128relaxed(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#else
    nk_dots_symmetric_u8_serial(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#endif
}

NK_PUBLIC void nk_dots_symmetric_e4m3(nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                      nk_size_t row_count) {
#if NK_TARGET_SME
    nk_dots_symmetric_e4m3_sme(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_NEONFP8
    nk_dots_symmetric_e4m3_neonfp8(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_NEONFHM
    nk_dots_symmetric_e4m3_neonfhm(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_SAPPHIREAMX
    nk_dots_symmetric_e4m3_sapphireamx(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#elif NK_TARGET_DIAMOND
    nk_dots_symmetric_e4m3_diamond(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_GENOA
    nk_dots_symmetric_e4m3_genoa(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_SKYLAKE
    nk_dots_symmetric_e4m3_skylake(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_HASWELL
    nk_dots_symmetric_e4m3_haswell(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_RVV
    nk_dots_symmetric_e4m3_rvv(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_V128RELAXED
    nk_dots_symmetric_e4m3_v128relaxed(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#else
    nk_dots_symmetric_e4m3_serial(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#endif
}

NK_PUBLIC void nk_dots_symmetric_e5m2(nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                      nk_size_t row_count) {
#if NK_TARGET_SME
    nk_dots_symmetric_e5m2_sme(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_NEONFP8
    nk_dots_symmetric_e5m2_neonfp8(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_NEONFHM
    nk_dots_symmetric_e5m2_neonfhm(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_SAPPHIREAMX
    nk_dots_symmetric_e5m2_sapphireamx(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#elif NK_TARGET_DIAMOND
    nk_dots_symmetric_e5m2_diamond(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_GENOA
    nk_dots_symmetric_e5m2_genoa(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_SKYLAKE
    nk_dots_symmetric_e5m2_skylake(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_HASWELL
    nk_dots_symmetric_e5m2_haswell(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_RVV
    nk_dots_symmetric_e5m2_rvv(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_V128RELAXED
    nk_dots_symmetric_e5m2_v128relaxed(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#else
    nk_dots_symmetric_e5m2_serial(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#endif
}

NK_PUBLIC void nk_dots_symmetric_e2m3(nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                      nk_size_t row_count) {
#if NK_TARGET_SME
    nk_dots_symmetric_e2m3_sme(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_SAPPHIREAMX
    nk_dots_symmetric_e2m3_sapphireamx(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#elif NK_TARGET_NEONFP8
    nk_dots_symmetric_e2m3_neonfp8(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_SKYLAKE
    nk_dots_symmetric_e2m3_skylake(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_SIERRA
    nk_dots_symmetric_e2m3_sierra(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_ALDER
    nk_dots_symmetric_e2m3_alder(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_HASWELL
    nk_dots_symmetric_e2m3_haswell(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_RVV
    nk_dots_symmetric_e2m3_rvv(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_V128RELAXED
    nk_dots_symmetric_e2m3_v128relaxed(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#else
    nk_dots_symmetric_e2m3_serial(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#endif
}

NK_PUBLIC void nk_dots_symmetric_e3m2(nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                      nk_size_t row_count) {
#if NK_TARGET_SME
    nk_dots_symmetric_e3m2_sme(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_SAPPHIREAMX
    nk_dots_symmetric_e3m2_sapphireamx(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#elif NK_TARGET_NEONFP8
    nk_dots_symmetric_e3m2_neonfp8(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_SKYLAKE
    nk_dots_symmetric_e3m2_skylake(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_HASWELL
    nk_dots_symmetric_e3m2_haswell(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_RVV
    nk_dots_symmetric_e3m2_rvv(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#else
    nk_dots_symmetric_e3m2_serial(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#endif
}

NK_PUBLIC void nk_dots_symmetric_u4(nk_u4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                    nk_size_t stride, nk_u32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                    nk_size_t row_count) {
#if NK_TARGET_SME
    nk_dots_symmetric_u4_sme(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_ICELAKE
    nk_dots_symmetric_u4_icelake(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_NEONSDOT
    nk_dots_symmetric_u4_neonsdot(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_HASWELL
    nk_dots_symmetric_u4_haswell(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_V128RELAXED
    nk_dots_symmetric_u4_v128relaxed(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#else
    nk_dots_symmetric_u4_serial(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#endif
}

NK_PUBLIC void nk_dots_symmetric_u1(nk_u1x8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                    nk_size_t stride, nk_u32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                    nk_size_t row_count) {
#if NK_TARGET_SMEBI32
    nk_dots_symmetric_u1_smebi32(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_ICELAKE
    nk_dots_symmetric_u1_icelake(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_HASWELL
    nk_dots_symmetric_u1_haswell(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_NEON
    nk_dots_symmetric_u1_neon(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_POWERVSX
    nk_dots_symmetric_u1_powervsx(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_LOONGSONASX
    nk_dots_symmetric_u1_loongsonasx(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#elif NK_TARGET_V128RELAXED
    nk_dots_symmetric_u1_v128relaxed(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#else
    nk_dots_symmetric_u1_serial(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#endif
}

NK_PUBLIC void nk_dots_symmetric_i4(nk_i4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                    nk_size_t stride, nk_i32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                    nk_size_t row_count) {
#if NK_TARGET_SME
    nk_dots_symmetric_i4_sme(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_ICELAKE
    nk_dots_symmetric_i4_icelake(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_NEONSDOT
    nk_dots_symmetric_i4_neonsdot(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_HASWELL
    nk_dots_symmetric_i4_haswell(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_V128RELAXED
    nk_dots_symmetric_i4_v128relaxed(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#else
    nk_dots_symmetric_i4_serial(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#endif
}

NK_PUBLIC void nk_dots_symmetric_f32(nk_f32_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                     nk_size_t stride, nk_f64_t *result, nk_size_t result_stride, nk_size_t row_start,
                                     nk_size_t row_count) {
#if NK_TARGET_SMEF64
    nk_dots_symmetric_f32_smef64(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_SKYLAKE
    nk_dots_symmetric_f32_skylake(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_HASWELL
    nk_dots_symmetric_f32_haswell(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_NEON
    nk_dots_symmetric_f32_neon(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_POWERVSX
    nk_dots_symmetric_f32_powervsx(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_RVV
    nk_dots_symmetric_f32_rvv(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_V128RELAXED
    nk_dots_symmetric_f32_v128relaxed(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
#else
    nk_dots_symmetric_f32_serial(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#endif
}

NK_PUBLIC void nk_dots_symmetric_f64(nk_f64_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                     nk_size_t stride, nk_f64_t *result, nk_size_t result_stride, nk_size_t row_start,
                                     nk_size_t row_count) {
#if NK_TARGET_SMEF64
    nk_dots_symmetric_f64_smef64(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_SKYLAKE
    nk_dots_symmetric_f64_skylake(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_HASWELL
    nk_dots_symmetric_f64_haswell(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_NEON
    nk_dots_symmetric_f64_neon(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_POWERVSX
    nk_dots_symmetric_f64_powervsx(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_RVV
    nk_dots_symmetric_f64_rvv(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_V128RELAXED
    nk_dots_symmetric_f64_v128relaxed(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
#else
    nk_dots_symmetric_f64_serial(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#endif
}

#endif // !NK_DYNAMIC_DISPATCH

#if defined(__cplusplus)
} // extern "C"
#endif

#endif
