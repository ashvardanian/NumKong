/**
 *  @file include/numkong/dots.h
 *  @author Ash Vardanian
 *  @date September 14, 2024
 *  @brief SIMD-accelerated Batched Dot Products.
 *
 *  Implements batch dot-product kernels computing output @b C of shape @b [rows,columns] as A × Bᵀ,
 *  from row-major @b A of shape @b [rows,depth] and arbitrary @b B of shape @b [columns,depth],
 *  optimized for ML inference and similarity workloads.
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
 *  - "dots_pack_size" - which estimates the memory requirements for external @c malloc
 *  - "dots_pack" - to perform the pre-processing
 *  - "dots_symmetric" - for A × Aᵀ Gram matrix multiplication
 *
 *  If the original "dots_packed" is analogous to "GEMM" (General Matrix Multiplication) in BLAS,
 *  the "dots_symmetric" is similar to the "SYRK" (the Symmetric rank-k update of a matrix). It
 *  visits the band (0, NUMKONG_SIZE_MAX) of @c nk_diagonal_band_t, the upper triangle, as the
 *  lower one of A × Aᵀ mirrors it. Symmetric kernels fill the result rows in
 *  `[rows_begin, rows_end)`, clamped to @c vector_count, so threads split the work by rows.
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
 *  - e2m1: 4-bit e2m1 floating point numbers (packed pairs) → 32-bit floats
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
 *  - NVIDIA: the CUDA baseline on every device, Ampere @c mma.sync from compute capability 8.0,
 *    Hopper @c wgmma, datacenter Blackwell @c tcgen05, Blackwell RTX Float8, Float6 and Float4 MMA
 *  - AMD: the ROCm baseline on every device, CDNA3, CDNA4 and CDNA5 matrix cores
 *  - Apple: the Metal baseline on every device, Apple9 @c simdgroup_matrix, Apple10 @c matmul2d
 *
 *  @section dots_numerical_stability Numerical Stability
 *
 *  - f64: Dot2 (Ogita-Rump-Oishi) on the accurate backends, otherwise native f64 FMA accumulation.
 *  - f32: public outputs widen to f64. Packed and symmetric kernels keep payloads narrow but widen
 *    accumulation.
 *  - bf16/f16: f32 accumulation. VDPBF16PS on Genoa does bf16 × bf16 → f32 natively.
 *  - e2m3/e3m2: f16 intermediate with flush to f32 every 128 elements (Sapphire).
 *  - i8: i32 accumulation. AMX TDPBSSD gives i8 × i8 → i32 tiles. Overflows at k > ~131K.
 *  - u1: Popcount, exact.
 *
 *  @section gpu_backends GPU Backends
 *
 *  Every GPU kernel, like @c nk_dots_packed_bf16_ampere, keeps its CPU twin's arguments, queues on
 *  the @c stream of one device, and returns an @ref nk_status_t without waiting on the device.
 *  Plain GPU A and @c vectors and their strides must be multiples of 16 bytes. Metal block-scaled
 *  sources allow byte-aligned codes and strides spanning whole blocks. Packed buffers remain
 *  16-byte-aligned; C and its stride must be multiples of the result's size, or the call returns
 *  @c nk_misaligned_k.
 *
 *  @section dots_pack_capabilities Packs Record Their Capability
 *
 *  Every pack writes its capability into its header, and every CPU consumer returns
 *  @c nk_pack_mismatch_k for a buffer another capability packed. So pack and multiply with the
 *  same mask. A GPU multiply can't read a device header without waiting on the stream, so there
 *  only the packed-shape reader, which waits anyway, checks it.
 *
 *  @section memory_layout Memory Layout and Transpose Semantics
 *
 *  All matrices use row-major storage. Column-major is not supported.
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
 *  Packs read B row by row and never fuse a transposition. A B stored as depth rows of columns
 *  values is transposed in a separate pass first, for example by copying a transposed
 *  @c nk::tensor_view into a row-major one:
 *
 *  @code{.c}
 *  // Standard matmul: C rows-by-columns from A rows-by-depth times B depth-by-columns
 *  nk_capability_t capabilities = nk_cap_serial_k;
 *  nk_cpu_capabilities_enabled(&capabilities);
 *  nk_dots_pack_bf16_best(b_transposed, columns, depth, depth * sizeof(nk_bf16_t), b_packed, 0, columns,
 *                         capabilities, NULL);
 *  nk_dots_packed_bf16_best(a, b_packed, c, rows, columns, depth, a_stride, c_stride, capabilities, NULL);
 *  @endcode
 *
 *  @section two_phase_api Two-Phase API for Static Weights
 *
 *  Matrix multiplication hardware, AMX and SME, requires specific data layouts that differ from
 *  standard row-major ordering. Since one matrix, typically weights in neural networks, is often
 *  static, we provide a two-phase API: pack once, multiply many times.
 *
 *  @code{.c}
 *  // Similarity search: C rows-by-columns from queries rows-by-depth, database columns-by-depth
 *  // Both matrices stored row-major, each row is one vector of dimension depth
 *  nk_size_t packed_bytes = 0;
 *  nk_dots_pack_size_bf16_best(columns, depth, capabilities, &packed_bytes);
 *  void *b_packed = malloc(packed_bytes);
 *  nk_dots_pack_bf16_best(database, columns, depth, depth * sizeof(nk_bf16_t), b_packed, 0, columns,
 *                         capabilities, NULL);
 *  nk_dots_packed_bf16_best(queries, b_packed, c, rows, columns, depth, depth * sizeof(nk_bf16_t),
 *                           columns * sizeof(nk_f32_t), capabilities, NULL);
 *  // Result: C[i,j] = dot(query i, database vector j)
 *  @endcode
 *
 *  Block-scaled operands arrive as references to their codes, their block scales and, for NVFP4,
 *  their tensor scale. One stride serves both: scale rows sit `stride / block_bytes` apart.
 *
 *  @code{.c}
 *  // NVFP4 rows of depth elements: depth / 2 bytes of E2M1 codes and depth / 16 UE4M3 scales each
 *  nk_nvfp4_cref_t const queries = {query_codes, query_scales, &query_tensor_scale};
 *  nk_nvfp4_cref_t const database = {database_codes, database_scales, &database_tensor_scale};
 *  nk_dots_pack_nvfp4_best(&database, columns, depth, depth / 2, b_packed, 0, columns, capabilities, NULL);
 *  nk_dots_packed_nvfp4_best(&queries, b_packed, c, rows, columns, depth, depth / 2, columns * sizeof(nk_f32_t),
 *                            capabilities, NULL);
 *  // Result: true dot products, both tensor scales included
 *  @endcode
 *
 *  The packed format is opaque and backend-specific. AMX expects (16 × 32) tiles with interleaved
 *  pairs, while NEON/SVE use arrangements optimized for their vector lengths.
 *
 *  @section why_int8 Why INT8 and Not UINT8?
 *
 *  Unsigned 8-bit integers were considered but deprioritized. The industry has converged on signed
 *  INT8 as the standard for quantized inference:
 *
 *  @verbatim
 *  Framework           Default     Notes
 *  PyTorch             qint8       New x86 backend uses INT8 via oneDNN
 *  TensorFlow Lite     int8        Actively removing UINT8 support
 *  ONNX Runtime        S8S8        "Should be the first choice"
 *  TensorRT            INT8        Symmetric [-128,127], no UINT8 option
 *  ARM CMSIS-NN        int8        Follows TFLite INT8 spec exactly
 *  @endverbatim
 *
 *  @section why_no_scaling Why No Alpha/Beta Scaling?
 *
 *  BLAS-style scaling, C = α × A × B + β × C, was considered but omitted. While useful for
 *  scientific computing — iterative solvers, matrix factorizations — it's rarely used in ML
 *  inference, where frameworks handle such operations via graph fusion. More importantly, on chips
 *  with separate physical registers for vector and matrix operations, like AMX, moving scalars
 *  between register files adds transfer latency that negates any benefit.
 *
 *  @section why_no_pad Why Not Pad N Dimension to Eliminate Edge Handling?
 *
 *  Padding N to a tile-aligned boundary (multiple of 16) during packing was considered to eliminate
 *  the separate AVX-512 edge kernel for N remainder rows. While this sounds simpler ("pure AMX"),
 *  it actually increases code size by ~125 lines because:
 *
 *  - The AVX-512 edge fallback is compact, ~40 lines, and handles both full-M × N-edge and M-edge ×
 *    N-edge cases through a single reusable function
 *  - Replacing it with "AMX + masked stores" requires verbose tile handling code duplicated across
 *    all 4 multiply functions — aligned or misaligned, BF16 or I8
 *  - Each function needs a new "trailing N tile for full M blocks" section (~50 lines each)
 *
 *  The current hybrid layout (AMX for full tiles, AVX-512 for edges) is more maintainable despite
 *  being conceptually less uniform. Memory overhead of the edge region is negligible, under 2% in
 *  the worst case.
 *
 *  @section dots_x86_instructions Relevant x86 Instructions
 *
 *  Low-precision matmul relies on VPMADD* from AVX2, VNNI dot-products, and BF16 dot-products on
 *  AVX-512. Zen4 improves throughput by dual-issuing many integer ops on FP ports.
 *
 *  @verbatim
 *  Intrinsic             Instruction                   Haswell   Genoa
 *  _mm256_maddubs_epi16  VPMADDUBSW (YMM, YMM, YMM)    5cy @ p0  3cy @ p01
 *  _mm256_madd_epi16     VPMADDWD (YMM, YMM, YMM)      5cy @ p0  3cy @ p01
 *  _mm256_dpbusd_epi32   VPDPBUSD (YMM, K, YMM, YMM)   n/a       4cy @ p01
 *  _mm256_dpwssds_epi32  VPDPWSSDS (YMM, K, YMM, YMM)  n/a       4cy @ p01
 *  _mm256_dpbf16_ps      VDPBF16PS (YMM, YMM, YMM)     n/a       6cy @ p01
 *  @endverbatim
 *
 *  AMX tile ops (TDPBF16PS/TDPBUSD/TDPBSSD) are not covered by the uops.info 2022 dataset.
 *
 *  @section dots_references References
 *
 *  @see x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
 *  @see Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *  @see uops.info: https://uops.info/
 *  @see Matrix Multiplication in 40 lines: https://en.algorithmica.org/hpc/algorithms/matmul/
 *  @see LLaMA CPU optimization: https://justine.lol/matmul/
 *  @see SME outer-product notes: https://github.com/tzakharko/m4-sme-exploration
 *
 */
#ifndef NUMKONG_DOTS_H
#define NUMKONG_DOTS_H

#include "numkong/types.h"
#include "numkong/capabilities.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Returns packed buffer size in bytes for second multiplier matrix (B).
 *  @param[in] columns The number of rows in B (output columns).
 *  @param[in] depth Columns in B, counting dimensions, a multiple of the values per byte.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[out] bytes The size of the packed buffer.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 *  @note The packed layout is backend-specific and must be produced by the matching pack function.
 */
NUMKONG_API nk_status_t nk_dots_pack_size_bf16_best(nk_size_t columns, nk_size_t depth, nk_capability_t capabilities,
                                                    nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f16_best(nk_size_t columns, nk_size_t depth, nk_capability_t capabilities,
                                                   nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e4m3_best(nk_size_t columns, nk_size_t depth, nk_capability_t capabilities,
                                                    nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e5m2_best(nk_size_t columns, nk_size_t depth, nk_capability_t capabilities,
                                                    nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m3_best(nk_size_t columns, nk_size_t depth, nk_capability_t capabilities,
                                                    nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m1_best(nk_size_t columns, nk_size_t depth, nk_capability_t capabilities,
                                                    nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_nvfp4_best(nk_size_t columns, nk_size_t depth, nk_capability_t capabilities,
                                                     nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp4_best(nk_size_t columns, nk_size_t depth, nk_capability_t capabilities,
                                                     nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp6e2m3_best(nk_size_t columns, nk_size_t depth,
                                                         nk_capability_t capabilities, nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp6e3m2_best(nk_size_t columns, nk_size_t depth,
                                                         nk_capability_t capabilities, nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp8e4m3_best(nk_size_t columns, nk_size_t depth,
                                                         nk_capability_t capabilities, nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp8e5m2_best(nk_size_t columns, nk_size_t depth,
                                                         nk_capability_t capabilities, nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e3m2_best(nk_size_t columns, nk_size_t depth, nk_capability_t capabilities,
                                                    nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f32_best(nk_size_t columns, nk_size_t depth, nk_capability_t capabilities,
                                                   nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f64_best(nk_size_t columns, nk_size_t depth, nk_capability_t capabilities,
                                                   nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i8_best(nk_size_t columns, nk_size_t depth, nk_capability_t capabilities,
                                                  nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u8_best(nk_size_t columns, nk_size_t depth, nk_capability_t capabilities,
                                                  nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i4_best(nk_size_t columns, nk_size_t depth, nk_capability_t capabilities,
                                                  nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u4_best(nk_size_t columns, nk_size_t depth, nk_capability_t capabilities,
                                                  nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u1_best(nk_size_t columns, nk_size_t depth, nk_capability_t capabilities,
                                                  nk_size_t *bytes);

/**
 *  @brief Reads a packed B matrix's shape from its header.
 *  @param[in] b_packed A buffer produced by the matching nk_dots_pack_bf16_best.
 *  @param[out] columns Receives the column count (B rows).
 *  @param[out] depth Receives the inner dimension.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_dots_packed_shape_bf16_best(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f16_best(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e4m3_best(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e5m2_best(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m3_best(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m1_best(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_nvfp4_best(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp4_best(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp6e2m3_best(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                            nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp6e3m2_best(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                            nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp8e4m3_best(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                            nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp8e5m2_best(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                            nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e3m2_best(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f32_best(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f64_best(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i8_best(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                     nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u8_best(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                     nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i4_best(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                     nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u4_best(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                     nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u1_best(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                     nk_capability_t capabilities, nk_stream_t stream);

/**
 *  @brief Packs the second multiplier (B) matrix into a backend-specific layout.
 *  @param[in] b The input B matrix in row-major order. Packing fuses no transposition, so a
 *      transposed B is transposed in a separate pass first.
 *  @param[in] columns The number of rows in B (output columns).
 *  @param[in] depth Columns in B, counting dimensions, a multiple of the values per byte.
 *  @param[in] b_stride The row stride in bytes for B.
 *  @param[out] b_packed The output packed buffer from nk_dots_pack_size_bf16_best.
 *  @param[in] columns_begin First output column to pack; 0 for a full pack.
 *  @param[in] columns_end One past the last output column to pack; @p columns for a full pack.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_dots_pack_bf16_best(nk_bf16_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_f16_best(nk_f16_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                              void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                              nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_e4m3_best(nk_e4m3_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_e5m2_best(nk_e5m2_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m3_best(nk_e2m3_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m1_best(nk_e2m1x2_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_capability_t capabilities, nk_stream_t stream);

/**
 *  @copydoc nk_dots_pack_bf16_best
 *  @note Block-scaled B arrives as a reference to its first codes and scales. @p b_stride counts
 *      bytes of codes and whole blocks, and rows of scales sit `b_stride / block_bytes` apart, as
 *      @c nk_block_scaled_format_t says. The pack keeps B's tensor scale and every product applies
 *      it. @p depth must be a multiple of the block size: the CPU backends assert it, and the GPU
 *      backends return @c nk_unexpected_dimensions_k.
 */
NUMKONG_API nk_status_t nk_dots_pack_nvfp4_best(nk_nvfp4_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_capability_t capabilities,
                                                nk_stream_t stream);
/** @copydoc nk_dots_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp4_best(nk_mxfp4_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_capability_t capabilities,
                                                nk_stream_t stream);
/** @copydoc nk_dots_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp6e2m3_best(nk_mxfp6e2m3_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                    nk_size_t columns_end, nk_capability_t capabilities,
                                                    nk_stream_t stream);
/** @copydoc nk_dots_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp6e3m2_best(nk_mxfp6e3m2_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                    nk_size_t columns_end, nk_capability_t capabilities,
                                                    nk_stream_t stream);
/** @copydoc nk_dots_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp8e4m3_best(nk_mxfp8e4m3_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                    nk_size_t columns_end, nk_capability_t capabilities,
                                                    nk_stream_t stream);
/** @copydoc nk_dots_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp8e5m2_best(nk_mxfp8e5m2_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                    nk_size_t columns_end, nk_capability_t capabilities,
                                                    nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_e3m2_best(nk_e3m2_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_f32_best(nk_f32_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                              void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                              nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_f64_best(nk_f64_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                              void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                              nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_i8_best(nk_i8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                             nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_u8_best(nk_u8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                             nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_i4_best(nk_i4x2_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                             nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_u4_best(nk_u4x2_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                             nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_u1_best(nk_u1x8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                             nk_capability_t capabilities, nk_stream_t stream);

/**
 *  @brief Computes C = A × Bᵀ using packed second multiplier matrix (B), accumulating into C.
 *  @param[in] a The input A matrix in row-major order.
 *  @param[in] b_packed The packed B matrix produced.
 *  @param[out] c The output C matrix in row-major order.
 *  @param[in] rows The number of rows in A.
 *  @param[in] columns The number of rows in B (output columns).
 *  @param[in] depth Columns in A and B, counting dimensions, a multiple of the values per byte.
 *  @param[in] a_stride The row stride in bytes for A.
 *  @param[in] c_stride The row stride in bytes for C.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_dots_packed_bf16_best(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_f16_best(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_e4m3_best(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_e5m2_best(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m3_best(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m1_best(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                 nk_size_t rows, nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_capability_t capabilities, nk_stream_t stream);

/**
 *  @copydoc nk_dots_packed_bf16_best
 *  @note Block-scaled A arrives as a reference, laid out as @c nk_dots_pack_nvfp4_best describes
 *      for B, and C holds true values, so every product takes the block scales and the tensor
 *      scales of both operands.
 */
NUMKONG_API nk_status_t nk_dots_packed_nvfp4_best(nk_nvfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                  nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t a_stride, nk_size_t c_stride, nk_capability_t capabilities,
                                                  nk_stream_t stream);
/** @copydoc nk_dots_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp4_best(nk_mxfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                  nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t a_stride, nk_size_t c_stride, nk_capability_t capabilities,
                                                  nk_stream_t stream);
/** @copydoc nk_dots_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp6e2m3_best(nk_mxfp6e2m3_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride,
                                                      nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp6e3m2_best(nk_mxfp6e3m2_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride,
                                                      nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp8e4m3_best(nk_mxfp8e4m3_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride,
                                                      nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp8e5m2_best(nk_mxfp8e5m2_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride,
                                                      nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_e3m2_best(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_f32_best(nk_f32_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_f64_best(nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_i8_best(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                               nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride, nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_u8_best(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                               nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride, nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_i4_best(nk_i4x2_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                               nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride, nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_u4_best(nk_u4x2_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                               nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride, nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_u1_best(nk_u1x8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                               nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride, nk_capability_t capabilities, nk_stream_t stream);

/**
 *  @brief Computes C = A × Aᵀ symmetric Gram matrix.
 *
 *  Visits the upper triangle, the band (0, NUMKONG_SIZE_MAX) of @c nk_diagonal_band_t.
 *
 *  @param[in] vectors Input matrix of row vectors in row-major order.
 *  @param[in] vector_count Number of vectors (rows) in the input matrix.
 *  @param[in] depth Counts dimensions, a multiple of the values per byte.
 *  @param[in] stride Row stride in bytes for the input matrix.
 *  @param[out] result Output symmetric matrix of @p vector_count × @p vector_count.
 *  @param[in] result_stride Row stride in bytes for the result matrix.
 *  @param[in] rows_begin First result row to compute, so threads split the work by rows.
 *  @param[in] rows_end Result row after the last to compute, clamped to @p vector_count.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_dots_symmetric_bf16_best(nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end,
                                                    nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f16_best(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end,
                                                   nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e4m3_best(nk_e4m3_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end,
                                                    nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e5m2_best(nk_e5m2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end,
                                                    nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m3_best(nk_e2m3_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end,
                                                    nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m1_best(nk_e2m1x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end,
                                                    nk_capability_t capabilities, nk_stream_t stream);

/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_nvfp4_best(nk_nvfp4_cref_t const *vectors, nk_size_t vector_count,
                                                     nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                     nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end,
                                                     nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp4_best(nk_mxfp4_cref_t const *vectors, nk_size_t vector_count,
                                                     nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                     nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end,
                                                     nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp6e2m3_best(nk_mxfp6e2m3_cref_t const *vectors, nk_size_t vector_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t rows_begin,
                                                         nk_size_t rows_end, nk_capability_t capabilities,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp6e3m2_best(nk_mxfp6e3m2_cref_t const *vectors, nk_size_t vector_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t rows_begin,
                                                         nk_size_t rows_end, nk_capability_t capabilities,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp8e4m3_best(nk_mxfp8e4m3_cref_t const *vectors, nk_size_t vector_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t rows_begin,
                                                         nk_size_t rows_end, nk_capability_t capabilities,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp8e5m2_best(nk_mxfp8e5m2_cref_t const *vectors, nk_size_t vector_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t rows_begin,
                                                         nk_size_t rows_end, nk_capability_t capabilities,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e3m2_best(nk_e3m2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end,
                                                    nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f32_best(nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end,
                                                   nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f64_best(nk_f64_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end,
                                                   nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i8_best(nk_i8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                  nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                  nk_size_t rows_begin, nk_size_t rows_end,
                                                  nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u8_best(nk_u8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                  nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                  nk_size_t rows_begin, nk_size_t rows_end,
                                                  nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i4_best(nk_i4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                  nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                  nk_size_t rows_begin, nk_size_t rows_end,
                                                  nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u4_best(nk_u4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                  nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                  nk_size_t rows_begin, nk_size_t rows_end,
                                                  nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u1_best(nk_u1x8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                  nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                  nk_size_t rows_begin, nk_size_t rows_end,
                                                  nk_capability_t capabilities, nk_stream_t stream);

/** @copydoc nk_dots_pack_size_f32_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f32_serial(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f32_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f32_serial(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_f32_best */
NUMKONG_API nk_status_t nk_dots_pack_f32_serial(nk_f32_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_f32_best */
NUMKONG_API nk_status_t nk_dots_packed_f32_serial(nk_f32_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f32_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f32_serial(nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);

/** @copydoc nk_dots_pack_size_f64_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f64_serial(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f64_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f64_serial(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_f64_best */
NUMKONG_API nk_status_t nk_dots_pack_f64_serial(nk_f64_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_f64_best */
NUMKONG_API nk_status_t nk_dots_packed_f64_serial(nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f64_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f64_serial(nk_f64_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);

/** @copydoc nk_dots_pack_size_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f16_serial(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f16_serial(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_f16_serial(nk_f16_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_f16_serial(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f16_serial(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);

/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_bf16_serial(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_bf16_serial(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_bf16_serial(nk_bf16_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_bf16_serial(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_bf16_serial(nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);

/** @copydoc nk_dots_pack_size_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i8_serial(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i8_serial(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_stream_t stream);
/** @copydoc nk_dots_pack_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_i8_serial(nk_i8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                               void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                               nk_stream_t stream);
/** @copydoc nk_dots_packed_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_i8_serial(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_i8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i8_serial(nk_i8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);

/** @copydoc nk_dots_pack_size_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u8_serial(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u8_serial(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_stream_t stream);
/** @copydoc nk_dots_pack_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_u8_serial(nk_u8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                               void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                               nk_stream_t stream);
/** @copydoc nk_dots_packed_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_u8_serial(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u8_serial(nk_u8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);

/** @copydoc nk_dots_pack_size_u4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u4_serial(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u4_serial(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_stream_t stream);
/** @copydoc nk_dots_pack_u4_best */
NUMKONG_API nk_status_t nk_dots_pack_u4_serial(nk_u4x2_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_u4_best */
NUMKONG_API nk_status_t nk_dots_packed_u4_serial(nk_u4x2_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u4_serial(nk_u4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);

/** @copydoc nk_dots_pack_size_u1_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u1_serial(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u1_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u1_serial(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_stream_t stream);
/** @copydoc nk_dots_pack_u1_best */
NUMKONG_API nk_status_t nk_dots_pack_u1_serial(nk_u1x8_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_u1_best */
NUMKONG_API nk_status_t nk_dots_packed_u1_serial(nk_u1x8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u1_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u1_serial(nk_u1x8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);

/** @copydoc nk_dots_pack_size_i4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i4_serial(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i4_serial(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_stream_t stream);
/** @copydoc nk_dots_pack_i4_best */
NUMKONG_API nk_status_t nk_dots_pack_i4_serial(nk_i4x2_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_i4_best */
NUMKONG_API nk_status_t nk_dots_packed_i4_serial(nk_i4x2_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_i4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i4_serial(nk_i4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e4m3_serial(nk_e4m3_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e5m2_serial(nk_e5m2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e4m3_serial(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e4m3_serial(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e4m3_serial(nk_e4m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e4m3_serial(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e5m2_serial(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e5m2_serial(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e5m2_serial(nk_e5m2_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e5m2_serial(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m3_serial(nk_e2m3_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m1_serial(nk_e2m1x2_t const *vectors, nk_size_t vector_count,
                                                      nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                      nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_nvfp4_serial(nk_nvfp4_cref_t const *vectors, nk_size_t vector_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t rows_begin,
                                                       nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp4_serial(nk_mxfp4_cref_t const *vectors, nk_size_t vector_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t rows_begin,
                                                       nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp6e2m3_serial(nk_mxfp6e2m3_cref_t const *vectors, nk_size_t vector_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t rows_begin,
                                                           nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp6e3m2_serial(nk_mxfp6e3m2_cref_t const *vectors, nk_size_t vector_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t rows_begin,
                                                           nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp8e4m3_serial(nk_mxfp8e4m3_cref_t const *vectors, nk_size_t vector_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t rows_begin,
                                                           nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp8e5m2_serial(nk_mxfp8e5m2_cref_t const *vectors, nk_size_t vector_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t rows_begin,
                                                           nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e3m2_serial(nk_e3m2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m3_serial(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m1_serial(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_nvfp4_serial(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp4_serial(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp6e2m3_serial(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp6e3m2_serial(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp8e4m3_serial(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp8e5m2_serial(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m3_serial(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m1_serial(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_nvfp4_serial(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp4_serial(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp6e2m3_serial(void const *b_packed, nk_size_t *columns,
                                                              nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp6e3m2_serial(void const *b_packed, nk_size_t *columns,
                                                              nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp8e4m3_serial(void const *b_packed, nk_size_t *columns,
                                                              nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp8e5m2_serial(void const *b_packed, nk_size_t *columns,
                                                              nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m3_serial(nk_e2m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m1_serial(nk_e2m1x2_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_nvfp4_serial(nk_nvfp4_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp4_serial(nk_mxfp4_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp6e2m3_serial(nk_mxfp6e2m3_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                      nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp6e3m2_serial(nk_mxfp6e3m2_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                      nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp8e4m3_serial(nk_mxfp8e4m3_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                      nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp8e5m2_serial(nk_mxfp8e5m2_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                      nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m3_serial(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m1_serial(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_nvfp4_serial(nk_nvfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp4_serial(nk_mxfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp6e2m3_serial(nk_mxfp6e2m3_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp6e3m2_serial(nk_mxfp6e3m2_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp8e4m3_serial(nk_mxfp8e4m3_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp8e5m2_serial(nk_mxfp8e5m2_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e3m2_serial(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e3m2_serial(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e3m2_serial(nk_e3m2_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e3m2_serial(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);

/*  Genoa backends using AVX-512 with BF16 extensions.
 *  These use VDPBF16PS for BF16 dot products.
 *  Packing interleaves elements for SIMD broadcast patterns. */
#if NUMKONG_TARGET_GENOA
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_bf16_genoa(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_bf16_genoa(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_bf16_genoa(nk_bf16_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_bf16_genoa(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_bf16_genoa(nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);

/** @copydoc nk_dots_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e4m3_genoa(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e4m3_genoa(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e4m3_genoa(nk_e4m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e4m3_genoa(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e5m2_genoa(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e5m2_genoa(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e5m2_genoa(nk_e5m2_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e5m2_genoa(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e4m3_genoa(nk_e4m3_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e5m2_genoa(nk_e5m2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_GENOA

#if NUMKONG_TARGET_DIAMOND
/** @copydoc nk_dots_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e4m3_diamond(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e4m3_diamond(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_pack_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e4m3_diamond(nk_e4m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e4m3_diamond(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e5m2_diamond(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e5m2_diamond(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_pack_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e5m2_diamond(nk_e5m2_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e5m2_diamond(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e4m3_diamond(nk_e4m3_t const *vectors, nk_size_t vector_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t rows_begin,
                                                       nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e5m2_diamond(nk_e5m2_t const *vectors, nk_size_t vector_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t rows_begin,
                                                       nk_size_t rows_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_DIAMOND

/*  Sapphire Rapids backends using Intel AMX (Advanced Matrix Extensions).
 *  AMX provides 8 tile registers (TMM0-TMM7), each holding up to 1KB of data.
 *  Tiles are configured as 16 rows × 64 bytes, enabling (16 × 32) BF16 or (16 × 64) INT8 tiles.
 *  Packing arranges data into AMX-native tile layout with pair interleaving for TDPBF16PS. */
#if NUMKONG_TARGET_SAPPHIREAMX
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_bf16_sapphireamx(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_bf16_sapphireamx(void const *b_packed, nk_size_t *columns,
                                                              nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_bf16_sapphireamx(nk_bf16_t const *b, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                      nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_bf16_sapphireamx(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_bf16_sapphireamx(nk_bf16_t const *vectors, nk_size_t vector_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t rows_begin,
                                                           nk_size_t rows_end, nk_stream_t stream);

/** @copydoc nk_dots_pack_size_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i8_sapphireamx(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i8_sapphireamx(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                            nk_stream_t stream);
/** @copydoc nk_dots_pack_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_i8_sapphireamx(nk_i8_t const *b, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                    nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_i8_sapphireamx(nk_i8_t const *a, void const *b_packed, nk_i32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_i8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i8_sapphireamx(nk_i8_t const *vectors, nk_size_t vector_count,
                                                         nk_size_t depth, nk_size_t stride, nk_i32_t *result,
                                                         nk_size_t result_stride, nk_size_t rows_begin,
                                                         nk_size_t rows_end, nk_stream_t stream);

/** @copydoc nk_dots_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e4m3_sapphireamx(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e4m3_sapphireamx(void const *b_packed, nk_size_t *columns,
                                                              nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_pack_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e4m3_sapphireamx(nk_e4m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                      nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e4m3_sapphireamx(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);

/** @copydoc nk_dots_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e4m3_sapphireamx(nk_e4m3_t const *vectors, nk_size_t vector_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t rows_begin,
                                                           nk_size_t rows_end, nk_stream_t stream);

/** @copydoc nk_dots_pack_size_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e5m2_sapphireamx(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e5m2_sapphireamx(void const *b_packed, nk_size_t *columns,
                                                              nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_pack_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e5m2_sapphireamx(nk_e5m2_t const *b, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                      nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e5m2_sapphireamx(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e5m2_sapphireamx(nk_e5m2_t const *vectors, nk_size_t vector_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t rows_begin,
                                                           nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m3_sapphireamx(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m1_sapphireamx(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_nvfp4_sapphireamx(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp4_sapphireamx(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp8e4m3_sapphireamx(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp8e5m2_sapphireamx(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m3_sapphireamx(void const *b_packed, nk_size_t *columns,
                                                              nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m1_sapphireamx(void const *b_packed, nk_size_t *columns,
                                                              nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_nvfp4_sapphireamx(void const *b_packed, nk_size_t *columns,
                                                               nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp4_sapphireamx(void const *b_packed, nk_size_t *columns,
                                                               nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp8e4m3_sapphireamx(void const *b_packed, nk_size_t *columns,
                                                                   nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp8e5m2_sapphireamx(void const *b_packed, nk_size_t *columns,
                                                                   nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m3_sapphireamx(nk_e2m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                      nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m1_sapphireamx(nk_e2m1x2_t const *b, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                      nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_nvfp4_sapphireamx(nk_nvfp4_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                       nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp4_sapphireamx(nk_mxfp4_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                       nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp8e4m3_sapphireamx(nk_mxfp8e4m3_cref_t const *b, nk_size_t columns,
                                                           nk_size_t depth, nk_size_t b_stride, void *b_packed,
                                                           nk_size_t columns_begin, nk_size_t columns_end,
                                                           nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp8e5m2_sapphireamx(nk_mxfp8e5m2_cref_t const *b, nk_size_t columns,
                                                           nk_size_t depth, nk_size_t b_stride, void *b_packed,
                                                           nk_size_t columns_begin, nk_size_t columns_end,
                                                           nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m3_sapphireamx(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m1_sapphireamx(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_nvfp4_sapphireamx(nk_nvfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp4_sapphireamx(nk_mxfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp8e4m3_sapphireamx(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                             nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                             nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                             nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp8e5m2_sapphireamx(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                             nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                             nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                             nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m3_sapphireamx(nk_e2m3_t const *vectors, nk_size_t vector_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t rows_begin,
                                                           nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m1_sapphireamx(nk_e2m1x2_t const *vectors, nk_size_t vector_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t rows_begin,
                                                           nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_nvfp4_sapphireamx(nk_nvfp4_cref_t const *vectors, nk_size_t vector_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t rows_begin,
                                                            nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp4_sapphireamx(nk_mxfp4_cref_t const *vectors, nk_size_t vector_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t rows_begin,
                                                            nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp8e4m3_sapphireamx(nk_mxfp8e4m3_cref_t const *vectors,
                                                                nk_size_t vector_count, nk_size_t depth,
                                                                nk_size_t stride, nk_f32_t *result,
                                                                nk_size_t result_stride, nk_size_t rows_begin,
                                                                nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp8e5m2_sapphireamx(nk_mxfp8e5m2_cref_t const *vectors,
                                                                nk_size_t vector_count, nk_size_t depth,
                                                                nk_size_t stride, nk_f32_t *result,
                                                                nk_size_t result_stride, nk_size_t rows_begin,
                                                                nk_size_t rows_end, nk_stream_t stream);

/** @copydoc nk_dots_pack_size_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e3m2_sapphireamx(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e3m2_sapphireamx(void const *b_packed, nk_size_t *columns,
                                                              nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_pack_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e3m2_sapphireamx(nk_e3m2_t const *b, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                      nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e3m2_sapphireamx(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e3m2_sapphireamx(nk_e3m2_t const *vectors, nk_size_t vector_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t rows_begin,
                                                           nk_size_t rows_end, nk_stream_t stream);

/** @copydoc nk_dots_pack_size_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u8_sapphireamx(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u8_sapphireamx(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                            nk_stream_t stream);
/** @copydoc nk_dots_pack_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_u8_sapphireamx(nk_u8_t const *b, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                    nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_u8_sapphireamx(nk_u8_t const *a, void const *b_packed, nk_u32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u8_sapphireamx(nk_u8_t const *vectors, nk_size_t vector_count,
                                                         nk_size_t depth, nk_size_t stride, nk_u32_t *result,
                                                         nk_size_t result_stride, nk_size_t rows_begin,
                                                         nk_size_t rows_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_SAPPHIREAMX

/*  Granite Rapids backends using Intel AMX-FP16 (Advanced Matrix Extensions with FP16 support).
 *  AMX-FP16 adds TDPFP16PS, an FP16 × FP16 → FP32 tile multiply-accumulate with the same tile
 *  geometry as BF16. The F32 Ozaki kernel splits F32 inputs into 2 FP16 halves for ~35-40 bit
 *  effective precision. */
#if NUMKONG_TARGET_GRANITEAMX
/** @copydoc nk_dots_pack_size_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f16_graniteamx(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f16_graniteamx(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                            nk_stream_t stream);
/** @copydoc nk_dots_pack_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_f16_graniteamx(nk_f16_t const *b, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                    nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_f16_graniteamx(nk_f16_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f16_graniteamx(nk_f16_t const *vectors, nk_size_t vector_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t rows_begin,
                                                         nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e5m2_graniteamx(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e5m2_graniteamx(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                             nk_stream_t stream);
/** @copydoc nk_dots_pack_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_e5m2_graniteamx(nk_e5m2_t const *b, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                     nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_e5m2_graniteamx(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e5m2_graniteamx(nk_e5m2_t const *vectors, nk_size_t vector_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t rows_begin,
                                                          nk_size_t rows_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_GRANITEAMX

/*  ARM SME backends using Scalable Matrix Extension.
 *  SME provides ZA tile registers for outer product operations.
 *  F16/BF16/I8/U8/E4M3 use ZA32 tiles, F32/F64 use ZA64 tiles (FEAT_SME_F64F64). */
#if NUMKONG_TARGET_SME
/** @copydoc nk_dots_pack_size_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f16_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f16_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                     nk_stream_t stream);
/** @copydoc nk_dots_pack_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_f16_sme(nk_f16_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                             nk_stream_t stream);
/** @copydoc nk_dots_packed_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_f16_sme(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                               nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f16_sme(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);

/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_bf16_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_bf16_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_bf16_sme(nk_bf16_t const *b, nk_size_t columns, nk_size_t depth,
                                              nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                              nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_bf16_sme(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_bf16_sme(nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);

/** @copydoc nk_dots_pack_size_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i8_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i8_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                    nk_stream_t stream);
/** @copydoc nk_dots_pack_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_i8_sme(nk_i8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                            void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                            nk_stream_t stream);
/** @copydoc nk_dots_packed_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_i8_sme(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                              nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                              nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_i8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i8_sme(nk_i8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                 nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                 nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);

/** @copydoc nk_dots_pack_size_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u8_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u8_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                    nk_stream_t stream);
/** @copydoc nk_dots_pack_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_u8_sme(nk_u8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                            void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                            nk_stream_t stream);
/** @copydoc nk_dots_packed_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_u8_sme(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                              nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                              nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u8_sme(nk_u8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                 nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                 nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);

/** @copydoc nk_dots_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e4m3_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e4m3_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_pack_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e4m3_sme(nk_e4m3_t const *b, nk_size_t columns, nk_size_t depth,
                                              nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                              nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e4m3_sme(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e4m3_sme(nk_e4m3_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);

/** @copydoc nk_dots_pack_size_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e5m2_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e5m2_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_pack_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e5m2_sme(nk_e5m2_t const *b, nk_size_t columns, nk_size_t depth,
                                              nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                              nk_size_t columns_end, nk_stream_t stream);

/**
 *  @copydoc nk_dots_packed_e5m2_best
 *  @note Handles edges with predicates, without scalar fallbacks.
 */
NUMKONG_API nk_status_t nk_dots_packed_e5m2_sme(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e5m2_sme(nk_e5m2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);

/** @copydoc nk_dots_pack_size_u4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u4_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u4_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                    nk_stream_t stream);
/** @copydoc nk_dots_pack_u4_best */
NUMKONG_API nk_status_t nk_dots_pack_u4_sme(nk_u4x2_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                            void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                            nk_stream_t stream);
/** @copydoc nk_dots_packed_u4_best */
NUMKONG_API nk_status_t nk_dots_packed_u4_sme(nk_u4x2_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                              nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                              nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u4_sme(nk_u4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                 nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                 nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);

/** @copydoc nk_dots_pack_size_i4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i4_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i4_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                    nk_stream_t stream);
/** @copydoc nk_dots_pack_i4_best */
NUMKONG_API nk_status_t nk_dots_pack_i4_sme(nk_i4x2_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                            void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                            nk_stream_t stream);
/** @copydoc nk_dots_packed_i4_best */
NUMKONG_API nk_status_t nk_dots_packed_i4_sme(nk_i4x2_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                              nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                              nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_i4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i4_sme(nk_i4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                 nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                 nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);

/** @copydoc nk_dots_pack_size_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m3_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m1_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m3_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m1_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m3_sme(nk_e2m3_t const *b, nk_size_t columns, nk_size_t depth,
                                              nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                              nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m1_sme(nk_e2m1x2_t const *b, nk_size_t columns, nk_size_t depth,
                                              nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                              nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m3_sme(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m1_sme(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m3_sme(nk_e2m3_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m1_sme(nk_e2m1x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);

/** @copydoc nk_dots_pack_size_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e3m2_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e3m2_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_pack_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e3m2_sme(nk_e3m2_t const *b, nk_size_t columns, nk_size_t depth,
                                              nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                              nk_size_t columns_end, nk_stream_t stream);

/**
 *  @copydoc nk_dots_packed_e3m2_best
 *  @note Handles edges with predicates, without scalar fallbacks.
 */
NUMKONG_API nk_status_t nk_dots_packed_e3m2_sme(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e3m2_sme(nk_e3m2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);

/** @copydoc nk_dots_pack_size_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_nvfp4_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_nvfp4_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_stream_t stream);
/** @copydoc nk_dots_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_nvfp4_sme(nk_nvfp4_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_nvfp4_sme(nk_nvfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                 nk_size_t rows, nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_nvfp4_sme(nk_nvfp4_cref_t const *vectors, nk_size_t vector_count,
                                                    nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end,
                                                    nk_stream_t stream);
/** @copydoc nk_dots_pack_size_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp4_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp4_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp4_sme(nk_mxfp4_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp4_sme(nk_mxfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                 nk_size_t rows, nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp4_sme(nk_mxfp4_cref_t const *vectors, nk_size_t vector_count,
                                                    nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end,
                                                    nk_stream_t stream);
/** @copydoc nk_dots_pack_size_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp6e2m3_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp6e2m3_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                           nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp6e2m3_sme(nk_mxfp6e2m3_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                   nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp6e2m3_sme(nk_mxfp6e2m3_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp6e2m3_sme(nk_mxfp6e2m3_cref_t const *vectors, nk_size_t vector_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t rows_begin,
                                                        nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp6e3m2_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp6e3m2_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                           nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp6e3m2_sme(nk_mxfp6e3m2_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                   nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp6e3m2_sme(nk_mxfp6e3m2_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp6e3m2_sme(nk_mxfp6e3m2_cref_t const *vectors, nk_size_t vector_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t rows_begin,
                                                        nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp8e4m3_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp8e4m3_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                           nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp8e4m3_sme(nk_mxfp8e4m3_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                   nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp8e4m3_sme(nk_mxfp8e4m3_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp8e4m3_sme(nk_mxfp8e4m3_cref_t const *vectors, nk_size_t vector_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t rows_begin,
                                                        nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp8e5m2_sme(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp8e5m2_sme(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                           nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp8e5m2_sme(nk_mxfp8e5m2_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                   nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp8e5m2_sme(nk_mxfp8e5m2_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp8e5m2_sme(nk_mxfp8e5m2_cref_t const *vectors, nk_size_t vector_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t rows_begin,
                                                        nk_size_t rows_end, nk_stream_t stream);

#endif // NUMKONG_TARGET_SME

/*  ARM SME with integer-accumulating binary outer products.
 *  Used for packed 1-bit dot products backed by ZA32. */
#if NUMKONG_TARGET_SMEBI32
/** @copydoc nk_dots_pack_size_u1_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u1_smebi32(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u1_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u1_smebi32(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_u1_best */
NUMKONG_API nk_status_t nk_dots_pack_u1_smebi32(nk_u1x8_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_u1_best */
NUMKONG_API nk_status_t nk_dots_packed_u1_smebi32(nk_u1x8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u1_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u1_smebi32(nk_u1x8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_SMEBI32

/*  ARM SME with FEAT_SME_F64F64 (F32/F64 with F64 accumulators).
 *  Requires Apple M4 or equivalent with F64 outer product support. */
#if NUMKONG_TARGET_SMEF64
/** @copydoc nk_dots_pack_size_f32_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f32_smef64(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f32_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f32_smef64(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_f32_best */
NUMKONG_API nk_status_t nk_dots_pack_f32_smef64(nk_f32_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_f32_best */
NUMKONG_API nk_status_t nk_dots_packed_f32_smef64(nk_f32_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f32_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f32_smef64(nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);

/** @copydoc nk_dots_pack_size_f64_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f64_smef64(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f64_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f64_smef64(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_f64_best */
NUMKONG_API nk_status_t nk_dots_pack_f64_smef64(nk_f64_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_f64_best */
NUMKONG_API nk_status_t nk_dots_packed_f64_smef64(nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f64_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f64_smef64(nk_f64_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_SMEF64

/*  Haswell backends using AVX2 (Intel Core 4th gen).
 *  Supports F32/F64 via FMA, F16/BF16/FP8 via software emulation, I8/U8 via VPMADDUBSW+VPADDD. */
#if NUMKONG_TARGET_HASWELL
/** @copydoc nk_dots_pack_size_f32_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f32_haswell(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f32_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f32_haswell(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_f32_best */
NUMKONG_API nk_status_t nk_dots_pack_f32_haswell(nk_f32_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_f32_best */
NUMKONG_API nk_status_t nk_dots_packed_f32_haswell(nk_f32_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows,
                                                   nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                   nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f32_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f32_haswell(nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                      nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_f64_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f64_haswell(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f64_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f64_haswell(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_f64_best */
NUMKONG_API nk_status_t nk_dots_pack_f64_haswell(nk_f64_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_f64_best */
NUMKONG_API nk_status_t nk_dots_packed_f64_haswell(nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows,
                                                   nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                   nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f64_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f64_haswell(nk_f64_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                      nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f16_haswell(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f16_haswell(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_f16_haswell(nk_f16_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_f16_haswell(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                   nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                   nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f16_haswell(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_bf16_haswell(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_bf16_haswell(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_bf16_haswell(nk_bf16_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_bf16_haswell(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_bf16_haswell(nk_bf16_t const *vectors, nk_size_t vector_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t rows_begin,
                                                       nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e4m3_haswell(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e4m3_haswell(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_pack_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e4m3_haswell(nk_e4m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e4m3_haswell(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e4m3_haswell(nk_e4m3_t const *vectors, nk_size_t vector_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t rows_begin,
                                                       nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e5m2_haswell(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e5m2_haswell(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_pack_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e5m2_haswell(nk_e5m2_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e5m2_haswell(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e5m2_haswell(nk_e5m2_t const *vectors, nk_size_t vector_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t rows_begin,
                                                       nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m3_haswell(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m1_haswell(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m3_haswell(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m1_haswell(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m3_haswell(nk_e2m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m1_haswell(nk_e2m1x2_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m3_haswell(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m1_haswell(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m3_haswell(nk_e2m3_t const *vectors, nk_size_t vector_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t rows_begin,
                                                       nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m1_haswell(nk_e2m1x2_t const *vectors, nk_size_t vector_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t rows_begin,
                                                       nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e3m2_haswell(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e3m2_haswell(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_pack_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e3m2_haswell(nk_e3m2_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e3m2_haswell(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e3m2_haswell(nk_e3m2_t const *vectors, nk_size_t vector_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t rows_begin,
                                                       nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i8_haswell(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i8_haswell(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_i8_haswell(nk_i8_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_i8_haswell(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_i8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i8_haswell(nk_i8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u8_haswell(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u8_haswell(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_u8_haswell(nk_u8_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_u8_haswell(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u8_haswell(nk_u8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u1_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u1_haswell(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u1_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u1_haswell(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_u1_best */
NUMKONG_API nk_status_t nk_dots_pack_u1_haswell(nk_u1x8_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_u1_best */
NUMKONG_API nk_status_t nk_dots_packed_u1_haswell(nk_u1x8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u1_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u1_haswell(nk_u1x8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_i4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i4_haswell(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i4_haswell(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_i4_best */
NUMKONG_API nk_status_t nk_dots_pack_i4_haswell(nk_i4x2_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_i4_best */
NUMKONG_API nk_status_t nk_dots_packed_i4_haswell(nk_i4x2_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_i4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i4_haswell(nk_i4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u4_haswell(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u4_haswell(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_u4_best */
NUMKONG_API nk_status_t nk_dots_pack_u4_haswell(nk_u4x2_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_u4_best */
NUMKONG_API nk_status_t nk_dots_packed_u4_haswell(nk_u4x2_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u4_haswell(nk_u4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_HASWELL

/*  Skylake backends using AVX-512 (Intel Core 6th gen+).
 *  Provides 512-bit vectors (16× f32, 8× f64), supporting F32/F64/F16/BF16/FP8 with FMA. */
#if NUMKONG_TARGET_SKYLAKE
/** @copydoc nk_dots_pack_size_f64_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f64_skylake(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f64_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f64_skylake(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_f64_best */
NUMKONG_API nk_status_t nk_dots_pack_f64_skylake(nk_f64_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_f64_best */
NUMKONG_API nk_status_t nk_dots_packed_f64_skylake(nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows,
                                                   nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                   nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f64_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f64_skylake(nk_f64_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                      nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_f32_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f32_skylake(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f32_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f32_skylake(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_f32_best */
NUMKONG_API nk_status_t nk_dots_pack_f32_skylake(nk_f32_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_f32_best */
NUMKONG_API nk_status_t nk_dots_packed_f32_skylake(nk_f32_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows,
                                                   nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                   nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f32_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f32_skylake(nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                      nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_bf16_skylake(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_bf16_skylake(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_bf16_skylake(nk_bf16_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_bf16_skylake(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_bf16_skylake(nk_bf16_t const *vectors, nk_size_t vector_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t rows_begin,
                                                       nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f16_skylake(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f16_skylake(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_f16_skylake(nk_f16_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_f16_skylake(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                   nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                   nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f16_skylake(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e4m3_skylake(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e4m3_skylake(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_pack_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e4m3_skylake(nk_e4m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e4m3_skylake(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e4m3_skylake(nk_e4m3_t const *vectors, nk_size_t vector_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t rows_begin,
                                                       nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e5m2_skylake(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e5m2_skylake(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_pack_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e5m2_skylake(nk_e5m2_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e5m2_skylake(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e5m2_skylake(nk_e5m2_t const *vectors, nk_size_t vector_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t rows_begin,
                                                       nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m3_skylake(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m1_skylake(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_nvfp4_skylake(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp4_skylake(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp8e4m3_skylake(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp8e5m2_skylake(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m3_skylake(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m1_skylake(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_nvfp4_skylake(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                           nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp4_skylake(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                           nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp8e4m3_skylake(void const *b_packed, nk_size_t *columns,
                                                               nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp8e5m2_skylake(void const *b_packed, nk_size_t *columns,
                                                               nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m3_skylake(nk_e2m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m1_skylake(nk_e2m1x2_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_nvfp4_skylake(nk_nvfp4_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                   nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp4_skylake(nk_mxfp4_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                   nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp8e4m3_skylake(nk_mxfp8e4m3_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                       nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp8e5m2_skylake(nk_mxfp8e5m2_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                       nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m3_skylake(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m1_skylake(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_nvfp4_skylake(nk_nvfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp4_skylake(nk_mxfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp8e4m3_skylake(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                         nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                         nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp8e5m2_skylake(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                         nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                         nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m3_skylake(nk_e2m3_t const *vectors, nk_size_t vector_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t rows_begin,
                                                       nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m1_skylake(nk_e2m1x2_t const *vectors, nk_size_t vector_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t rows_begin,
                                                       nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_nvfp4_skylake(nk_nvfp4_cref_t const *vectors, nk_size_t vector_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t rows_begin,
                                                        nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp4_skylake(nk_mxfp4_cref_t const *vectors, nk_size_t vector_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t rows_begin,
                                                        nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp8e4m3_skylake(nk_mxfp8e4m3_cref_t const *vectors, nk_size_t vector_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t rows_begin,
                                                            nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp8e5m2_skylake(nk_mxfp8e5m2_cref_t const *vectors, nk_size_t vector_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t rows_begin,
                                                            nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e3m2_skylake(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e3m2_skylake(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_pack_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e3m2_skylake(nk_e3m2_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e3m2_skylake(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e3m2_skylake(nk_e3m2_t const *vectors, nk_size_t vector_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t rows_begin,
                                                       nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp6e2m3_skylake(nk_mxfp6e2m3_cref_t const *vectors, nk_size_t vector_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t rows_begin,
                                                            nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp6e2m3_skylake(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp6e2m3_skylake(void const *b_packed, nk_size_t *columns,
                                                               nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp6e2m3_skylake(nk_mxfp6e2m3_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                       nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp6e2m3_skylake(nk_mxfp6e2m3_cref_t const *a, void const *b_packed,
                                                         nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                         nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp6e3m2_skylake(nk_mxfp6e3m2_cref_t const *vectors, nk_size_t vector_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t rows_begin,
                                                            nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp6e3m2_skylake(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp6e3m2_skylake(void const *b_packed, nk_size_t *columns,
                                                               nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp6e3m2_skylake(nk_mxfp6e3m2_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                       nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp6e3m2_skylake(nk_mxfp6e3m2_cref_t const *a, void const *b_packed,
                                                         nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                         nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                         nk_stream_t stream);
#endif // NUMKONG_TARGET_SKYLAKE

/*  Ice Lake backends using AVX-512 with VNNI (Vector Neural Network Instructions).
 *  Adds VPDPBUSD for I8/U8, VPDPWSSD for I4/U4 with efficient dot products. */
#if NUMKONG_TARGET_ICELAKE
/** @copydoc nk_dots_pack_size_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i8_icelake(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i8_icelake(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_i8_icelake(nk_i8_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_i8_icelake(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_i8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i8_icelake(nk_i8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u8_icelake(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u8_icelake(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_u8_icelake(nk_u8_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_u8_icelake(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u8_icelake(nk_u8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_i4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i4_icelake(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i4_icelake(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_i4_best */
NUMKONG_API nk_status_t nk_dots_pack_i4_icelake(nk_i4x2_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_i4_best */
NUMKONG_API nk_status_t nk_dots_packed_i4_icelake(nk_i4x2_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_i4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i4_icelake(nk_i4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u4_icelake(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u4_icelake(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_u4_best */
NUMKONG_API nk_status_t nk_dots_pack_u4_icelake(nk_u4x2_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_u4_best */
NUMKONG_API nk_status_t nk_dots_packed_u4_icelake(nk_u4x2_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u4_icelake(nk_u4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u1_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u1_icelake(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u1_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u1_icelake(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_u1_best */
NUMKONG_API nk_status_t nk_dots_pack_u1_icelake(nk_u1x8_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_u1_best */
NUMKONG_API nk_status_t nk_dots_packed_u1_icelake(nk_u1x8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u1_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u1_icelake(nk_u1x8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_ICELAKE

/*  Alder backends using AMX with TDPB[SU]SD / TDPBF16PS.
 *  Optimized for I8/U8 via AMX integer tiles, E2M3 via AMX BF16 tiles. */
#if NUMKONG_TARGET_ALDER
/** @copydoc nk_dots_pack_size_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i8_alder(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i8_alder(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_pack_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_i8_alder(nk_i8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                              void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                              nk_stream_t stream);
/** @copydoc nk_dots_packed_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_i8_alder(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_i8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i8_alder(nk_i8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u8_alder(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u8_alder(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_pack_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_u8_alder(nk_u8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                              void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                              nk_stream_t stream);
/** @copydoc nk_dots_packed_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_u8_alder(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u8_alder(nk_u8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m3_alder(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m1_alder(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m3_alder(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m1_alder(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m3_alder(nk_e2m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m1_alder(nk_e2m1x2_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m3_alder(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m1_alder(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                  nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m3_alder(nk_e2m3_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m1_alder(nk_e2m1x2_t const *vectors, nk_size_t vector_count,
                                                     nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                     nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end,
                                                     nk_stream_t stream);
#endif // NUMKONG_TARGET_ALDER

/*  Sierra backends using AVX10.2 with VMPSADBW.
 *  Optimized for I8/U8 via VMPSADBW (vector multiply-sum of absolute differences). */
#if NUMKONG_TARGET_SIERRA
/** @copydoc nk_dots_pack_size_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i8_sierra(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i8_sierra(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_stream_t stream);
/** @copydoc nk_dots_pack_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_i8_sierra(nk_i8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                               void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                               nk_stream_t stream);
/** @copydoc nk_dots_packed_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_i8_sierra(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_i8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i8_sierra(nk_i8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u8_sierra(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u8_sierra(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_stream_t stream);
/** @copydoc nk_dots_pack_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_u8_sierra(nk_u8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                               void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                               nk_stream_t stream);
/** @copydoc nk_dots_packed_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_u8_sierra(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u8_sierra(nk_u8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m3_sierra(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m1_sierra(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m3_sierra(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m1_sierra(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m3_sierra(nk_e2m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m1_sierra(nk_e2m1x2_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m3_sierra(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m1_sierra(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m3_sierra(nk_e2m3_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m1_sierra(nk_e2m1x2_t const *vectors, nk_size_t vector_count,
                                                      nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                      nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end,
                                                      nk_stream_t stream);
#endif // NUMKONG_TARGET_SIERRA

/*  WASM Relaxed SIMD backends using wasm_i32x4_relaxed_dot_i8x16_i7x16_add.
 *  Covers I8/U8/E2M3 (depth_simd_dimensions=16), BF16/F32 (4), F64 (2). */
#if NUMKONG_TARGET_V128RELAXED
/** @copydoc nk_dots_pack_size_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i8_v128relaxed(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i8_v128relaxed(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                            nk_stream_t stream);
/** @copydoc nk_dots_pack_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_i8_v128relaxed(nk_i8_t const *b, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                    nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_i8_v128relaxed(nk_i8_t const *a, void const *b_packed, nk_i32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_i8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i8_v128relaxed(nk_i8_t const *vectors, nk_size_t vector_count,
                                                         nk_size_t depth, nk_size_t stride, nk_i32_t *result,
                                                         nk_size_t result_stride, nk_size_t rows_begin,
                                                         nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u8_v128relaxed(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u8_v128relaxed(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                            nk_stream_t stream);
/** @copydoc nk_dots_pack_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_u8_v128relaxed(nk_u8_t const *b, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                    nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_u8_v128relaxed(nk_u8_t const *a, void const *b_packed, nk_u32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u8_v128relaxed(nk_u8_t const *vectors, nk_size_t vector_count,
                                                         nk_size_t depth, nk_size_t stride, nk_u32_t *result,
                                                         nk_size_t result_stride, nk_size_t rows_begin,
                                                         nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m3_v128relaxed(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m1_v128relaxed(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m3_v128relaxed(void const *b_packed, nk_size_t *columns,
                                                              nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m1_v128relaxed(void const *b_packed, nk_size_t *columns,
                                                              nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m3_v128relaxed(nk_e2m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                      nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m1_v128relaxed(nk_e2m1x2_t const *b, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                      nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m3_v128relaxed(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m1_v128relaxed(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m3_v128relaxed(nk_e2m3_t const *vectors, nk_size_t vector_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t rows_begin,
                                                           nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m1_v128relaxed(nk_e2m1x2_t const *vectors, nk_size_t vector_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t rows_begin,
                                                           nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_bf16_v128relaxed(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_bf16_v128relaxed(void const *b_packed, nk_size_t *columns,
                                                              nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_bf16_v128relaxed(nk_bf16_t const *b, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                      nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_bf16_v128relaxed(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_bf16_v128relaxed(nk_bf16_t const *vectors, nk_size_t vector_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t rows_begin,
                                                           nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f16_v128relaxed(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f16_v128relaxed(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                             nk_stream_t stream);
/** @copydoc nk_dots_pack_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_f16_v128relaxed(nk_f16_t const *b, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                     nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_f16_v128relaxed(nk_f16_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f16_v128relaxed(nk_f16_t const *vectors, nk_size_t vector_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t rows_begin,
                                                          nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e3m2_v128relaxed(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e3m2_v128relaxed(void const *b_packed, nk_size_t *columns,
                                                              nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_pack_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e3m2_v128relaxed(nk_e3m2_t const *b, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                      nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e3m2_v128relaxed(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e3m2_v128relaxed(nk_e3m2_t const *vectors, nk_size_t vector_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t rows_begin,
                                                           nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_f32_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f32_v128relaxed(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f32_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f32_v128relaxed(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                             nk_stream_t stream);
/** @copydoc nk_dots_pack_f32_best */
NUMKONG_API nk_status_t nk_dots_pack_f32_v128relaxed(nk_f32_t const *b, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                     nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_f32_best */
NUMKONG_API nk_status_t nk_dots_packed_f32_v128relaxed(nk_f32_t const *a, void const *b_packed, nk_f64_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f32_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f32_v128relaxed(nk_f32_t const *vectors, nk_size_t vector_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                          nk_size_t result_stride, nk_size_t rows_begin,
                                                          nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_f64_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f64_v128relaxed(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f64_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f64_v128relaxed(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                             nk_stream_t stream);
/** @copydoc nk_dots_pack_f64_best */
NUMKONG_API nk_status_t nk_dots_pack_f64_v128relaxed(nk_f64_t const *b, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                     nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_f64_best */
NUMKONG_API nk_status_t nk_dots_packed_f64_v128relaxed(nk_f64_t const *a, void const *b_packed, nk_f64_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f64_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f64_v128relaxed(nk_f64_t const *vectors, nk_size_t vector_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                          nk_size_t result_stride, nk_size_t rows_begin,
                                                          nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e4m3_v128relaxed(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e4m3_v128relaxed(void const *b_packed, nk_size_t *columns,
                                                              nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_e4m3_v128relaxed(nk_e4m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                      nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_e4m3_v128relaxed(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e4m3_v128relaxed(nk_e4m3_t const *vectors, nk_size_t vector_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t rows_begin,
                                                           nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e5m2_v128relaxed(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e5m2_v128relaxed(void const *b_packed, nk_size_t *columns,
                                                              nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_e5m2_v128relaxed(nk_e5m2_t const *b, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                      nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_e5m2_v128relaxed(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e5m2_v128relaxed(nk_e5m2_t const *vectors, nk_size_t vector_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t rows_begin,
                                                           nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u4_v128relaxed(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u4_v128relaxed(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                            nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_u4_v128relaxed(nk_u4x2_t const *b, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                    nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_u4_v128relaxed(nk_u4x2_t const *a, void const *b_packed, nk_u32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u4_v128relaxed(nk_u4x2_t const *vectors, nk_size_t vector_count,
                                                         nk_size_t depth, nk_size_t stride, nk_u32_t *result,
                                                         nk_size_t result_stride, nk_size_t rows_begin,
                                                         nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i4_v128relaxed(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i4_v128relaxed(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                            nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_i4_v128relaxed(nk_i4x2_t const *b, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                    nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_i4_v128relaxed(nk_i4x2_t const *a, void const *b_packed, nk_i32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i4_v128relaxed(nk_i4x2_t const *vectors, nk_size_t vector_count,
                                                         nk_size_t depth, nk_size_t stride, nk_i32_t *result,
                                                         nk_size_t result_stride, nk_size_t rows_begin,
                                                         nk_size_t rows_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_V128RELAXED

#if NUMKONG_TARGET_V128
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_bf16_v128(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_bf16_v128(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_bf16_v128(nk_bf16_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_bf16_v128(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_bf16_v128(nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i8_v128(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i8_v128(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                     nk_stream_t stream);
/** @copydoc nk_dots_pack_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_i8_v128(nk_i8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                             nk_stream_t stream);
/** @copydoc nk_dots_packed_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_i8_v128(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                               nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_i8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i8_v128(nk_i8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                  nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                  nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u8_v128(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u8_v128(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                     nk_stream_t stream);
/** @copydoc nk_dots_pack_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_u8_v128(nk_u8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                             nk_stream_t stream);
/** @copydoc nk_dots_packed_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_u8_v128(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                               nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u8_v128(nk_u8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                  nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                  nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u1_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u1_v128(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u1_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u1_v128(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                     nk_stream_t stream);
/** @copydoc nk_dots_pack_u1_best */
NUMKONG_API nk_status_t nk_dots_pack_u1_v128(nk_u1x8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                             nk_stream_t stream);
/** @copydoc nk_dots_packed_u1_best */
NUMKONG_API nk_status_t nk_dots_packed_u1_v128(nk_u1x8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                               nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u1_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u1_v128(nk_u1x8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                  nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                  nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_V128

/*  ARM NEON backends (base NEON with F32/F64 support).
 *  Uses FMLA for F32 dots, FMLA (scalar) for F64. */
#if NUMKONG_TARGET_NEON
/** @copydoc nk_dots_pack_size_f32_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f32_neon(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f32_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f32_neon(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_pack_f32_best */
NUMKONG_API nk_status_t nk_dots_pack_f32_neon(nk_f32_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                              void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                              nk_stream_t stream);
/** @copydoc nk_dots_packed_f32_best */
NUMKONG_API nk_status_t nk_dots_packed_f32_neon(nk_f32_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f32_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f32_neon(nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_f64_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f64_neon(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f64_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f64_neon(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_pack_f64_best */
NUMKONG_API nk_status_t nk_dots_pack_f64_neon(nk_f64_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                              void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                              nk_stream_t stream);
/** @copydoc nk_dots_packed_f64_best */
NUMKONG_API nk_status_t nk_dots_packed_f64_neon(nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f64_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f64_neon(nk_f64_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u1_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u1_neon(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u1_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u1_neon(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                     nk_stream_t stream);
/** @copydoc nk_dots_pack_u1_best */
NUMKONG_API nk_status_t nk_dots_pack_u1_neon(nk_u1x8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                             nk_stream_t stream);
/** @copydoc nk_dots_packed_u1_best */
NUMKONG_API nk_status_t nk_dots_packed_u1_neon(nk_u1x8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                               nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u1_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u1_neon(nk_u1x8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                  nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                  nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f16_neon(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f16_neon(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_pack_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_f16_neon(nk_f16_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                              void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                              nk_stream_t stream);
/** @copydoc nk_dots_packed_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_f16_neon(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f16_neon(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_bf16_neon(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_bf16_neon(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_bf16_neon(nk_bf16_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_bf16_neon(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_bf16_neon(nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);

/** @copydoc nk_dots_symmetric_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp6e2m3_neon(nk_mxfp6e2m3_cref_t const *vectors, nk_size_t vector_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t rows_begin,
                                                         nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp6e2m3_neon(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp6e2m3_neon(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                            nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp6e2m3_neon(nk_mxfp6e2m3_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                    nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp6e2m3_neon(nk_mxfp6e2m3_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp6e3m2_neon(nk_mxfp6e3m2_cref_t const *vectors, nk_size_t vector_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t rows_begin,
                                                         nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp6e3m2_neon(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp6e3m2_neon(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                            nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp6e3m2_neon(nk_mxfp6e3m2_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                    nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp6e3m2_neon(nk_mxfp6e3m2_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);

/** @copydoc nk_dots_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_nvfp4_neon(nk_nvfp4_cref_t const *vectors, nk_size_t vector_count,
                                                     nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                     nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end,
                                                     nk_stream_t stream);
/** @copydoc nk_dots_pack_size_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_nvfp4_neon(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_nvfp4_neon(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_nvfp4_neon(nk_nvfp4_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_nvfp4_neon(nk_nvfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                  nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp4_neon(nk_mxfp4_cref_t const *vectors, nk_size_t vector_count,
                                                     nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                     nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end,
                                                     nk_stream_t stream);
/** @copydoc nk_dots_pack_size_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp4_neon(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp4_neon(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp4_neon(nk_mxfp4_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp4_neon(nk_mxfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                  nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp8e4m3_neon(nk_mxfp8e4m3_cref_t const *vectors, nk_size_t vector_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t rows_begin,
                                                         nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp8e4m3_neon(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp8e4m3_neon(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                            nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp8e4m3_neon(nk_mxfp8e4m3_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                    nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp8e4m3_neon(nk_mxfp8e4m3_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp8e5m2_neon(nk_mxfp8e5m2_cref_t const *vectors, nk_size_t vector_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t rows_begin,
                                                         nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp8e5m2_neon(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp8e5m2_neon(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                            nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp8e5m2_neon(nk_mxfp8e5m2_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                    nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp8e5m2_neon(nk_mxfp8e5m2_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);

#endif // NUMKONG_TARGET_NEON

/*  ARM NEON with BF16 dot product (ARMv8.6-A BF16).
 *  Uses BFDOT/BFMMLA for efficient BF16 matrix operations. */
#if NUMKONG_TARGET_NEONBFDOT
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_bf16_neonbfdot(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_bf16_neonbfdot(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                            nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_bf16_neonbfdot(nk_bf16_t const *b, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                    nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_bf16_neonbfdot(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_bf16_neonbfdot(nk_bf16_t const *vectors, nk_size_t vector_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t rows_begin,
                                                         nk_size_t rows_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_NEONBFDOT

/*  ARM NEON with signed/unsigned dot product (ARMv8.2-A DotProd).
 *  Provides SDOT/UDOT for I8/U8 vector dot products. */
#if NUMKONG_TARGET_NEONSDOT
/** @copydoc nk_dots_pack_size_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i8_neonsdot(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i8_neonsdot(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_i8_neonsdot(nk_i8_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_i8_neonsdot(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                                   nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                   nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_i8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i8_neonsdot(nk_i8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                      nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                      nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u8_neonsdot(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u8_neonsdot(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_u8_neonsdot(nk_u8_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_u8_neonsdot(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                                   nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                   nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u8_neonsdot(nk_u8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                      nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                      nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m3_neonsdot(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m1_neonsdot(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m3_neonsdot(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                           nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m1_neonsdot(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                           nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m3_neonsdot(nk_e2m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                   nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m1_neonsdot(nk_e2m1x2_t const *b, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                   nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m3_neonsdot(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m1_neonsdot(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m3_neonsdot(nk_e2m3_t const *vectors, nk_size_t vector_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t rows_begin,
                                                        nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m1_neonsdot(nk_e2m1x2_t const *vectors, nk_size_t vector_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t rows_begin,
                                                        nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e3m2_neonsdot(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e3m2_neonsdot(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                           nk_stream_t stream);
/** @copydoc nk_dots_pack_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e3m2_neonsdot(nk_e3m2_t const *b, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                   nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e3m2_neonsdot(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e3m2_neonsdot(nk_e3m2_t const *vectors, nk_size_t vector_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t rows_begin,
                                                        nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_i4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i4_neonsdot(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i4_neonsdot(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_i4_best */
NUMKONG_API nk_status_t nk_dots_pack_i4_neonsdot(nk_i4x2_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_i4_best */
NUMKONG_API nk_status_t nk_dots_packed_i4_neonsdot(nk_i4x2_t const *a, void const *b_packed, nk_i32_t *c,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_i4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i4_neonsdot(nk_i4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                      nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                      nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u4_neonsdot(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u4_neonsdot(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_u4_best */
NUMKONG_API nk_status_t nk_dots_pack_u4_neonsdot(nk_u4x2_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_u4_best */
NUMKONG_API nk_status_t nk_dots_packed_u4_neonsdot(nk_u4x2_t const *a, void const *b_packed, nk_u32_t *c,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u4_neonsdot(nk_u4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                      nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                      nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_nvfp4_neonsdot(nk_nvfp4_cref_t const *vectors, nk_size_t vector_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t rows_begin,
                                                         nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_nvfp4_neonsdot(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_nvfp4_neonsdot(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                            nk_stream_t stream);
/** @copydoc nk_dots_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_nvfp4_neonsdot(nk_nvfp4_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                    nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_nvfp4_neonsdot(nk_nvfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp4_neonsdot(nk_mxfp4_cref_t const *vectors, nk_size_t vector_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t rows_begin,
                                                         nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp4_neonsdot(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp4_neonsdot(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                            nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp4_neonsdot(nk_mxfp4_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                    nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp4_neonsdot(nk_mxfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp6e2m3_neonsdot(nk_mxfp6e2m3_cref_t const *vectors, nk_size_t vector_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t rows_begin,
                                                             nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp6e2m3_neonsdot(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp6e2m3_neonsdot(void const *b_packed, nk_size_t *columns,
                                                                nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp6e2m3_neonsdot(nk_mxfp6e2m3_cref_t const *b, nk_size_t columns,
                                                        nk_size_t depth, nk_size_t b_stride, void *b_packed,
                                                        nk_size_t columns_begin, nk_size_t columns_end,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp6e2m3_neonsdot(nk_mxfp6e2m3_cref_t const *a, void const *b_packed,
                                                          nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                          nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                          nk_stream_t stream);
#endif // NUMKONG_TARGET_NEONSDOT

/*  ARM NEON with FP16 FML (fused multiply-long, ARMv8.2-A FP16FML).
 *  Uses FMLAL/FMLSL for F16 and custom FP8 (E2M3/E3M2) operations. */
#if NUMKONG_TARGET_NEONFHM
/** @copydoc nk_dots_pack_size_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f16_neonfhm(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f16_neonfhm(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_f16_neonfhm(nk_f16_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_f16_neonfhm(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                   nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                   nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f16_neonfhm(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e4m3_neonfhm(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e4m3_neonfhm(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_pack_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e4m3_neonfhm(nk_e4m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e4m3_neonfhm(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e4m3_neonfhm(nk_e4m3_t const *vectors, nk_size_t vector_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t rows_begin,
                                                       nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e5m2_neonfhm(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e5m2_neonfhm(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_pack_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e5m2_neonfhm(nk_e5m2_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e5m2_neonfhm(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e5m2_neonfhm(nk_e5m2_t const *vectors, nk_size_t vector_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t rows_begin,
                                                       nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp8e4m3_neonfhm(nk_mxfp8e4m3_cref_t const *vectors, nk_size_t vector_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t rows_begin,
                                                            nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp8e4m3_neonfhm(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp8e4m3_neonfhm(void const *b_packed, nk_size_t *columns,
                                                               nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp8e4m3_neonfhm(nk_mxfp8e4m3_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                       nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp8e4m3_neonfhm(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                         nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                         nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp8e5m2_neonfhm(nk_mxfp8e5m2_cref_t const *vectors, nk_size_t vector_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t rows_begin,
                                                            nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp8e5m2_neonfhm(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp8e5m2_neonfhm(void const *b_packed, nk_size_t *columns,
                                                               nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp8e5m2_neonfhm(nk_mxfp8e5m2_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                       nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp8e5m2_neonfhm(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                         nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                         nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                         nk_stream_t stream);
#endif // NUMKONG_TARGET_NEONFHM

/*  ARM NEON with FP8 (ARMv9.2-A FP8).
 *  Uses native FP8 dot-product instructions for E4M3/E5M2/E2M3/E3M2 operations. */
#if NUMKONG_TARGET_NEONFP8
/** @copydoc nk_dots_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e4m3_neonfp8(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e4m3_neonfp8(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_pack_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e4m3_neonfp8(nk_e4m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e4m3_neonfp8(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e4m3_neonfp8(nk_e4m3_t const *vectors, nk_size_t vector_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t rows_begin,
                                                       nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e5m2_neonfp8(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e5m2_neonfp8(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_pack_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e5m2_neonfp8(nk_e5m2_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e5m2_neonfp8(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e5m2_neonfp8(nk_e5m2_t const *vectors, nk_size_t vector_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t rows_begin,
                                                       nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m3_neonfp8(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m1_neonfp8(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m3_neonfp8(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m1_neonfp8(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m3_neonfp8(nk_e2m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m1_neonfp8(nk_e2m1x2_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m3_neonfp8(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m1_neonfp8(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m3_neonfp8(nk_e2m3_t const *vectors, nk_size_t vector_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t rows_begin,
                                                       nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m1_neonfp8(nk_e2m1x2_t const *vectors, nk_size_t vector_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t rows_begin,
                                                       nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e3m2_neonfp8(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e3m2_neonfp8(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_pack_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e3m2_neonfp8(nk_e3m2_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e3m2_neonfp8(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e3m2_neonfp8(nk_e3m2_t const *vectors, nk_size_t vector_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t rows_begin,
                                                       nk_size_t rows_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_NEONFP8

/*  RISC-V RVV backends. @c vsetvl handles partial vectors, so every packed kernel dispatches to its
 *  aligned kernel without a separate edge kernel. Packed B uses a column-panel layout with
 *  depth-contiguous storage, cache-line padding and zeroed padding values. */
#if NUMKONG_TARGET_RVV
/** @copydoc nk_dots_pack_size_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m3_rvv(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m1_rvv(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m3_rvv(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m1_rvv(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);

/**
 *  @copydoc nk_dots_pack_e2m3_best
 *  @note Converts each e2m3 byte through a scalar LUT to a signed i8 holding value × 16, for
 *      integer dot products.
 */
NUMKONG_API nk_status_t nk_dots_pack_e2m3_rvv(nk_e2m3_t const *b, nk_size_t columns, nk_size_t depth,
                                              nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                              nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m1_rvv(nk_e2m1x2_t const *b, nk_size_t columns, nk_size_t depth,
                                              nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                              nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m3_rvv(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m1_rvv(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);

/**
 *  @copydoc nk_dots_symmetric_e2m3_best
 *  @note Uses i8 LUT arithmetic with i32 accumulation, scaled by 1/256.
 */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m3_rvv(nk_e2m3_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m1_rvv(nk_e2m1x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e3m2_rvv(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e3m2_rvv(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);

/**
 *  @copydoc nk_dots_pack_e3m2_best
 *  @note Converts each e3m2 byte through a scalar LUT to a signed i16 holding value × 16, for
 *      integer dot products.
 */
NUMKONG_API nk_status_t nk_dots_pack_e3m2_rvv(nk_e3m2_t const *b, nk_size_t columns, nk_size_t depth,
                                              nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                              nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e3m2_rvv(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);

/**
 *  @copydoc nk_dots_symmetric_e3m2_best
 *  @note Uses i16 LUT arithmetic with an i32 widening multiply-accumulate, scaled by 1/256.
 */
NUMKONG_API nk_status_t nk_dots_symmetric_e3m2_rvv(nk_e3m2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_f32_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f32_rvv(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f32_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f32_rvv(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                     nk_stream_t stream);
/** @copydoc nk_dots_pack_f32_best */
NUMKONG_API nk_status_t nk_dots_pack_f32_rvv(nk_f32_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                             nk_stream_t stream);
/** @copydoc nk_dots_packed_f32_best */
NUMKONG_API nk_status_t nk_dots_packed_f32_rvv(nk_f32_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows,
                                               nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride, nk_stream_t stream);

/**
 *  @copydoc nk_dots_symmetric_f32_best
 *  @note Accumulates in f64 via @c vfwmacc_vv_f64m4.
 */
NUMKONG_API nk_status_t nk_dots_symmetric_f32_rvv(nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                  nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_f64_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f64_rvv(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f64_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f64_rvv(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                     nk_stream_t stream);
/** @copydoc nk_dots_pack_f64_best */
NUMKONG_API nk_status_t nk_dots_pack_f64_rvv(nk_f64_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                             nk_stream_t stream);
/** @copydoc nk_dots_packed_f64_best */
NUMKONG_API nk_status_t nk_dots_packed_f64_rvv(nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows,
                                               nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride, nk_stream_t stream);

/**
 *  @copydoc nk_dots_symmetric_f64_best
 *  @note Applies Kahan compensation over the full depth.
 */
NUMKONG_API nk_status_t nk_dots_symmetric_f64_rvv(nk_f64_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                  nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);

/**
 *  @copydoc nk_dots_pack_size_bf16_best
 *  @note B is stored as f32, so the vector length comes from `__riscv_vsetvlmax_e32m2()`.
 */
NUMKONG_API nk_status_t nk_dots_pack_size_bf16_rvv(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_bf16_rvv(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);

/**
 *  @copydoc nk_dots_pack_bf16_best
 *  @note Widens each bf16 to f32 with a bit shift, as bf16 is the upper 16 bits of f32.
 */
NUMKONG_API nk_status_t nk_dots_pack_bf16_rvv(nk_bf16_t const *b, nk_size_t columns, nk_size_t depth,
                                              nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                              nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_bf16_rvv(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);

/**
 *  @copydoc nk_dots_symmetric_bf16_best
 *  @note Loads both inputs as u16, widens them to f32 via @c nk_bf16m1_to_f32m2_rvv_, and
 *      accumulates in f64 via @c vfwmacc_vv_f64m4.
 */
NUMKONG_API nk_status_t nk_dots_symmetric_bf16_rvv(nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);

/**
 *  @copydoc nk_dots_pack_size_f16_best
 *  @note B is stored as f32, so the vector length comes from `__riscv_vsetvlmax_e32m2()`.
 */
NUMKONG_API nk_status_t nk_dots_pack_size_f16_rvv(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f16_rvv(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                     nk_stream_t stream);

/**
 *  @copydoc nk_dots_pack_f16_best
 *  @note Widens each f16 to f32 via @c nk_f16_to_f32_serial.
 */
NUMKONG_API nk_status_t nk_dots_pack_f16_rvv(nk_f16_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                             nk_stream_t stream);
/** @copydoc nk_dots_packed_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_f16_rvv(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                               nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride, nk_stream_t stream);

/**
 *  @copydoc nk_dots_symmetric_f16_best
 *  @note Loads both inputs as u16, widens them to f32 via @c nk_f16m1_to_f32m2_rvv_, and
 *      accumulates in f64 via @c vfwmacc_vv_f64m4.
 */
NUMKONG_API nk_status_t nk_dots_symmetric_f16_rvv(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);

/**
 *  @copydoc nk_dots_pack_size_i8_best
 *  @note B is stored as i8, so the vector length comes from `__riscv_vsetvlmax_e8m1()`.
 */
NUMKONG_API nk_status_t nk_dots_pack_size_i8_rvv(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i8_rvv(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                    nk_stream_t stream);

/**
 *  @copydoc nk_dots_pack_i8_best
 *  @note Copies the values without conversion.
 */
NUMKONG_API nk_status_t nk_dots_pack_i8_rvv(nk_i8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                            void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                            nk_stream_t stream);
/** @copydoc nk_dots_packed_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_i8_rvv(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                              nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                              nk_size_t c_stride, nk_stream_t stream);

/**
 *  @copydoc nk_dots_symmetric_i8_best
 *  @note Widens i8 × i8 → i16 → i32 for accumulation.
 */
NUMKONG_API nk_status_t nk_dots_symmetric_i8_rvv(nk_i8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                 nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                 nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);

/**
 *  @copydoc nk_dots_pack_size_u8_best
 *  @note B is stored as u8, so the vector length comes from `__riscv_vsetvlmax_e8m1()`.
 */
NUMKONG_API nk_status_t nk_dots_pack_size_u8_rvv(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u8_rvv(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                    nk_stream_t stream);

/**
 *  @copydoc nk_dots_pack_u8_best
 *  @note Copies the values without conversion.
 */
NUMKONG_API nk_status_t nk_dots_pack_u8_rvv(nk_u8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                            void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                            nk_stream_t stream);
/** @copydoc nk_dots_packed_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_u8_rvv(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                              nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                              nk_size_t c_stride, nk_stream_t stream);

/**
 *  @copydoc nk_dots_symmetric_u8_best
 *  @note Widens u8 × u8 → u16 → u32 for accumulation.
 */
NUMKONG_API nk_status_t nk_dots_symmetric_u8_rvv(nk_u8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                 nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                 nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e4m3_rvv(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e4m3_rvv(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);

/**
 *  @copydoc nk_dots_pack_e4m3_best
 *  @note Converts each e4m3 byte to f32 via @c nk_e4m3_to_f32_serial.
 */
NUMKONG_API nk_status_t nk_dots_pack_e4m3_rvv(nk_e4m3_t const *b, nk_size_t columns, nk_size_t depth,
                                              nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                              nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e4m3_rvv(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);

/**
 *  @copydoc nk_dots_symmetric_e4m3_best
 *  @note Decodes both operands from e4m3 on the fly through an f32 magnitude LUT gather, and
 *      accumulates in f64.
 */
NUMKONG_API nk_status_t nk_dots_symmetric_e4m3_rvv(nk_e4m3_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e5m2_rvv(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e5m2_rvv(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);

/**
 *  @copydoc nk_dots_pack_e5m2_best
 *  @note Converts each e5m2 byte to f32 via @c nk_e5m2_to_f32_serial.
 */
NUMKONG_API nk_status_t nk_dots_pack_e5m2_rvv(nk_e5m2_t const *b, nk_size_t columns, nk_size_t depth,
                                              nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                              nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e5m2_rvv(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);

/**
 *  @copydoc nk_dots_symmetric_e5m2_best
 *  @note Decodes both operands from e5m2 on the fly through an f32 magnitude LUT gather, and
 *      accumulates in f64.
 */
NUMKONG_API nk_status_t nk_dots_symmetric_e5m2_rvv(nk_e5m2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_RVV

/*  IBM Power VSX backends using 128-bit SIMD (Power9). */
#if NUMKONG_TARGET_POWERVSX
/** @copydoc nk_dots_pack_size_f32_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f32_powervsx(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f32_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f32_powervsx(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_pack_f32_best */
NUMKONG_API nk_status_t nk_dots_pack_f32_powervsx(nk_f32_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_f32_best */
NUMKONG_API nk_status_t nk_dots_packed_f32_powervsx(nk_f32_t const *a, void const *b_packed, nk_f64_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f32_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f32_powervsx(nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                       nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                       nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_f64_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f64_powervsx(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f64_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f64_powervsx(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_pack_f64_best */
NUMKONG_API nk_status_t nk_dots_pack_f64_powervsx(nk_f64_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_f64_best */
NUMKONG_API nk_status_t nk_dots_packed_f64_powervsx(nk_f64_t const *a, void const *b_packed, nk_f64_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f64_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f64_powervsx(nk_f64_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                       nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                       nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_bf16_powervsx(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_bf16_powervsx(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                           nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_bf16_powervsx(nk_bf16_t const *b, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                   nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_bf16_powervsx(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_bf16_powervsx(nk_bf16_t const *vectors, nk_size_t vector_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t rows_begin,
                                                        nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f16_powervsx(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f16_powervsx(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_pack_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_f16_powervsx(nk_f16_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_f16_powervsx(nk_f16_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f16_powervsx(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                       nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                       nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i8_powervsx(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i8_powervsx(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_i8_powervsx(nk_i8_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_i8_powervsx(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                                   nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                   nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_i8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i8_powervsx(nk_i8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                      nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                      nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u8_powervsx(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u8_powervsx(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_u8_powervsx(nk_u8_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_u8_powervsx(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                                   nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                   nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u8_powervsx(nk_u8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                      nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                      nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u1_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u1_powervsx(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u1_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u1_powervsx(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_u1_best */
NUMKONG_API nk_status_t nk_dots_pack_u1_powervsx(nk_u1x8_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_u1_best */
NUMKONG_API nk_status_t nk_dots_packed_u1_powervsx(nk_u1x8_t const *a, void const *b_packed, nk_u32_t *c,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u1_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u1_powervsx(nk_u1x8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                      nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                      nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_POWERVSX

/*  Loongson LASX backends using 256-bit SIMD (LoongArch). */
#if NUMKONG_TARGET_LOONGSONASX
/** @copydoc nk_dots_pack_size_f32_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f32_loongsonasx(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f32_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f32_loongsonasx(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                             nk_stream_t stream);
/** @copydoc nk_dots_pack_f32_best */
NUMKONG_API nk_status_t nk_dots_pack_f32_loongsonasx(nk_f32_t const *b, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                     nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_f32_best */
NUMKONG_API nk_status_t nk_dots_packed_f32_loongsonasx(nk_f32_t const *a, void const *b_packed, nk_f64_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f32_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f32_loongsonasx(nk_f32_t const *vectors, nk_size_t vector_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                          nk_size_t result_stride, nk_size_t rows_begin,
                                                          nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_f64_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f64_loongsonasx(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f64_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f64_loongsonasx(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                             nk_stream_t stream);
/** @copydoc nk_dots_pack_f64_best */
NUMKONG_API nk_status_t nk_dots_pack_f64_loongsonasx(nk_f64_t const *b, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                     nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_f64_best */
NUMKONG_API nk_status_t nk_dots_packed_f64_loongsonasx(nk_f64_t const *a, void const *b_packed, nk_f64_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f64_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f64_loongsonasx(nk_f64_t const *vectors, nk_size_t vector_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                          nk_size_t result_stride, nk_size_t rows_begin,
                                                          nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f16_loongsonasx(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f16_loongsonasx(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                             nk_stream_t stream);
/** @copydoc nk_dots_pack_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_f16_loongsonasx(nk_f16_t const *b, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                     nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_f16_loongsonasx(nk_f16_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f16_loongsonasx(nk_f16_t const *vectors, nk_size_t vector_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t rows_begin,
                                                          nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_bf16_loongsonasx(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_bf16_loongsonasx(void const *b_packed, nk_size_t *columns,
                                                              nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_bf16_loongsonasx(nk_bf16_t const *b, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                      nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_bf16_loongsonasx(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_bf16_loongsonasx(nk_bf16_t const *vectors, nk_size_t vector_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t rows_begin,
                                                           nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i8_loongsonasx(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i8_loongsonasx(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                            nk_stream_t stream);
/** @copydoc nk_dots_pack_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_i8_loongsonasx(nk_i8_t const *b, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                    nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_i8_loongsonasx(nk_i8_t const *a, void const *b_packed, nk_i32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_i8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i8_loongsonasx(nk_i8_t const *vectors, nk_size_t vector_count,
                                                         nk_size_t depth, nk_size_t stride, nk_i32_t *result,
                                                         nk_size_t result_stride, nk_size_t rows_begin,
                                                         nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u8_loongsonasx(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u8_loongsonasx(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                            nk_stream_t stream);
/** @copydoc nk_dots_pack_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_u8_loongsonasx(nk_u8_t const *b, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                    nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_u8_loongsonasx(nk_u8_t const *a, void const *b_packed, nk_u32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u8_loongsonasx(nk_u8_t const *vectors, nk_size_t vector_count,
                                                         nk_size_t depth, nk_size_t stride, nk_u32_t *result,
                                                         nk_size_t result_stride, nk_size_t rows_begin,
                                                         nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u1_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u1_loongsonasx(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u1_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u1_loongsonasx(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                            nk_stream_t stream);
/** @copydoc nk_dots_pack_u1_best */
NUMKONG_API nk_status_t nk_dots_pack_u1_loongsonasx(nk_u1x8_t const *b, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                    nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_u1_best */
NUMKONG_API nk_status_t nk_dots_packed_u1_loongsonasx(nk_u1x8_t const *a, void const *b_packed, nk_u32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u1_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u1_loongsonasx(nk_u1x8_t const *vectors, nk_size_t vector_count,
                                                         nk_size_t depth, nk_size_t stride, nk_u32_t *result,
                                                         nk_size_t result_stride, nk_size_t rows_begin,
                                                         nk_size_t rows_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_LOONGSONASX

/*  CUDA baseline on every NVIDIA device: every input widened exactly and multiplied with scalar
 *  FMAs, accumulating in F64, and F64 inputs in Dot2. */
#if NUMKONG_TARGET_CUDA
/** @copydoc nk_dots_pack_size_f64_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f64_cuda(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f64_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f64_cuda(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_pack_f64_best */
NUMKONG_API nk_status_t nk_dots_pack_f64_cuda(nk_f64_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                              void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                              nk_stream_t stream);
/** @copydoc nk_dots_packed_f64_best */
NUMKONG_API nk_status_t nk_dots_packed_f64_cuda(nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f64_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f64_cuda(nk_f64_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_f32_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f32_cuda(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f32_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f32_cuda(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_pack_f32_best */
NUMKONG_API nk_status_t nk_dots_pack_f32_cuda(nk_f32_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                              void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                              nk_stream_t stream);
/** @copydoc nk_dots_packed_f32_best */
NUMKONG_API nk_status_t nk_dots_packed_f32_cuda(nk_f32_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f32_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f32_cuda(nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_bf16_cuda(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_bf16_cuda(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_bf16_cuda(nk_bf16_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_bf16_cuda(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_bf16_cuda(nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f16_cuda(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f16_cuda(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_pack_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_f16_cuda(nk_f16_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                              void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                              nk_stream_t stream);
/** @copydoc nk_dots_packed_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_f16_cuda(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f16_cuda(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e5m2_cuda(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e5m2_cuda(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_stream_t stream);
/** @copydoc nk_dots_pack_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e5m2_cuda(nk_e5m2_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e5m2_cuda(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e5m2_cuda(nk_e5m2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e4m3_cuda(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e4m3_cuda(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_stream_t stream);
/** @copydoc nk_dots_pack_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e4m3_cuda(nk_e4m3_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e4m3_cuda(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e4m3_cuda(nk_e4m3_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e3m2_cuda(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e3m2_cuda(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_stream_t stream);
/** @copydoc nk_dots_pack_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e3m2_cuda(nk_e3m2_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e3m2_cuda(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e3m2_cuda(nk_e3m2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m3_cuda(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m3_cuda(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m3_cuda(nk_e2m3_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m3_cuda(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m3_cuda(nk_e2m3_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m1_cuda(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_nvfp4_cuda(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp4_cuda(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp8e4m3_cuda(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp8e5m2_cuda(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m1_cuda(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_nvfp4_cuda(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp4_cuda(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp8e4m3_cuda(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                            nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp8e5m2_cuda(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                            nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m1_cuda(nk_e2m1x2_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_nvfp4_cuda(nk_nvfp4_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp4_cuda(nk_mxfp4_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp8e4m3_cuda(nk_mxfp8e4m3_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                    nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp8e5m2_cuda(nk_mxfp8e5m2_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                    nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m1_cuda(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                 nk_size_t rows, nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_nvfp4_cuda(nk_nvfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                  nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp4_cuda(nk_mxfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                  nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp8e4m3_cuda(nk_mxfp8e4m3_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp8e5m2_cuda(nk_mxfp8e5m2_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m1_cuda(nk_e2m1x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_nvfp4_cuda(nk_nvfp4_cref_t const *vectors, nk_size_t vector_count,
                                                     nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                     nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end,
                                                     nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp4_cuda(nk_mxfp4_cref_t const *vectors, nk_size_t vector_count,
                                                     nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                     nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end,
                                                     nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp8e4m3_cuda(nk_mxfp8e4m3_cref_t const *vectors, nk_size_t vector_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t rows_begin,
                                                         nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp8e5m2_cuda(nk_mxfp8e5m2_cref_t const *vectors, nk_size_t vector_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t rows_begin,
                                                         nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i8_cuda(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i8_cuda(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                     nk_stream_t stream);
/** @copydoc nk_dots_pack_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_i8_cuda(nk_i8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                             nk_stream_t stream);
/** @copydoc nk_dots_packed_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_i8_cuda(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                               nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_i8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i8_cuda(nk_i8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                  nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                  nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u8_cuda(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u8_cuda(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                     nk_stream_t stream);
/** @copydoc nk_dots_pack_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_u8_cuda(nk_u8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                             nk_stream_t stream);
/** @copydoc nk_dots_packed_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_u8_cuda(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                               nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u8_cuda(nk_u8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                  nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                  nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_i4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i4_cuda(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i4_cuda(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                     nk_stream_t stream);
/** @copydoc nk_dots_pack_i4_best */
NUMKONG_API nk_status_t nk_dots_pack_i4_cuda(nk_i4x2_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                             nk_stream_t stream);
/** @copydoc nk_dots_packed_i4_best */
NUMKONG_API nk_status_t nk_dots_packed_i4_cuda(nk_i4x2_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                               nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_i4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i4_cuda(nk_i4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                  nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                  nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u4_cuda(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u4_cuda(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                     nk_stream_t stream);
/** @copydoc nk_dots_pack_u4_best */
NUMKONG_API nk_status_t nk_dots_pack_u4_cuda(nk_u4x2_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                             nk_stream_t stream);
/** @copydoc nk_dots_packed_u4_best */
NUMKONG_API nk_status_t nk_dots_packed_u4_cuda(nk_u4x2_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                               nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u4_cuda(nk_u4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                  nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                  nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_CUDA

/*  NVIDIA backends from Ampere on: BF16, F16 and the integers through warp-level `mma.sync`, E2M3
 *  and E2M1 scaled into exact integers, the other Float8 and Float6 widened into F16. */
#if NUMKONG_TARGET_AMPERE
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_bf16_ampere(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_bf16_ampere(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_bf16_ampere(nk_bf16_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_bf16_ampere(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_bf16_ampere(nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f16_ampere(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f16_ampere(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_f16_ampere(nk_f16_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_f16_ampere(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f16_ampere(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e5m2_ampere(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e5m2_ampere(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e5m2_ampere(nk_e5m2_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e5m2_ampere(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e5m2_ampere(nk_e5m2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e4m3_ampere(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e4m3_ampere(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e4m3_ampere(nk_e4m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e4m3_ampere(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e4m3_ampere(nk_e4m3_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e3m2_ampere(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e3m2_ampere(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e3m2_ampere(nk_e3m2_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e3m2_ampere(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e3m2_ampere(nk_e3m2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m3_ampere(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m3_ampere(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m3_ampere(nk_e2m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m3_ampere(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m3_ampere(nk_e2m3_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m1_ampere(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m1_ampere(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m1_ampere(nk_e2m1x2_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m1_ampere(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m1_ampere(nk_e2m1x2_t const *vectors, nk_size_t vector_count,
                                                      nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                      nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_pack_size_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i8_ampere(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i8_ampere(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_stream_t stream);
/** @copydoc nk_dots_pack_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_i8_ampere(nk_i8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                               void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                               nk_stream_t stream);
/** @copydoc nk_dots_packed_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_i8_ampere(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_i8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i8_ampere(nk_i8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u8_ampere(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u8_ampere(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_stream_t stream);
/** @copydoc nk_dots_pack_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_u8_ampere(nk_u8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                               void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                               nk_stream_t stream);
/** @copydoc nk_dots_packed_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_u8_ampere(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u8_ampere(nk_u8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_i4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i4_ampere(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i4_ampere(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_stream_t stream);
/** @copydoc nk_dots_pack_i4_best */
NUMKONG_API nk_status_t nk_dots_pack_i4_ampere(nk_i4x2_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_i4_best */
NUMKONG_API nk_status_t nk_dots_packed_i4_ampere(nk_i4x2_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_i4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i4_ampere(nk_i4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u4_ampere(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u4_ampere(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_stream_t stream);
/** @copydoc nk_dots_pack_u4_best */
NUMKONG_API nk_status_t nk_dots_pack_u4_ampere(nk_u4x2_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_u4_best */
NUMKONG_API nk_status_t nk_dots_packed_u4_ampere(nk_u4x2_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u4_ampere(nk_u4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_AMPERE

/*  NVIDIA Hopper backends, compute capability 9.0, through warpgroup @c wgmma over shared-memory
 *  descriptors. */
#if NUMKONG_TARGET_HOPPER
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_bf16_hopper(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_bf16_hopper(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_bf16_hopper(nk_bf16_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_bf16_hopper(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_bf16_hopper(nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f16_hopper(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f16_hopper(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_f16_hopper(nk_f16_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_f16_hopper(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f16_hopper(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m3_hopper(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m3_hopper(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m3_hopper(nk_e2m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m3_hopper(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m3_hopper(nk_e2m3_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m1_hopper(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m1_hopper(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m1_hopper(nk_e2m1x2_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m1_hopper(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m1_hopper(nk_e2m1x2_t const *vectors, nk_size_t vector_count,
                                                      nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                      nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_pack_size_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i8_hopper(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i8_hopper(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_stream_t stream);
/** @copydoc nk_dots_pack_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_i8_hopper(nk_i8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                               void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                               nk_stream_t stream);
/** @copydoc nk_dots_packed_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_i8_hopper(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_i8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i8_hopper(nk_i8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u8_hopper(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u8_hopper(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_stream_t stream);
/** @copydoc nk_dots_pack_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_u8_hopper(nk_u8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                               void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                               nk_stream_t stream);
/** @copydoc nk_dots_packed_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_u8_hopper(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u8_hopper(nk_u8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_i4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i4_hopper(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i4_hopper(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_stream_t stream);
/** @copydoc nk_dots_pack_i4_best */
NUMKONG_API nk_status_t nk_dots_pack_i4_hopper(nk_i4x2_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_i4_best */
NUMKONG_API nk_status_t nk_dots_packed_i4_hopper(nk_i4x2_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_i4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i4_hopper(nk_i4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u4_hopper(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u4_hopper(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_stream_t stream);
/** @copydoc nk_dots_pack_u4_best */
NUMKONG_API nk_status_t nk_dots_pack_u4_hopper(nk_u4x2_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_u4_best */
NUMKONG_API nk_status_t nk_dots_packed_u4_hopper(nk_u4x2_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u4_hopper(nk_u4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_HOPPER

/*  NVIDIA datacenter Blackwell backends, the compute capability 10.x family, through single-thread
 *  @c tcgen05 MMAs into tensor memory. */
#if NUMKONG_TARGET_BLACKWELL
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_bf16_blackwell(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_bf16_blackwell(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                            nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_bf16_blackwell(nk_bf16_t const *b, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                    nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_bf16_blackwell(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_bf16_blackwell(nk_bf16_t const *vectors, nk_size_t vector_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t rows_begin,
                                                         nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f16_blackwell(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f16_blackwell(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                           nk_stream_t stream);
/** @copydoc nk_dots_pack_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_f16_blackwell(nk_f16_t const *b, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                   nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_f16_blackwell(nk_f16_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f16_blackwell(nk_f16_t const *vectors, nk_size_t vector_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t rows_begin,
                                                        nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e5m2_blackwell(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e5m2_blackwell(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                            nk_stream_t stream);
/** @copydoc nk_dots_pack_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e5m2_blackwell(nk_e5m2_t const *b, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                    nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e5m2_blackwell(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e5m2_blackwell(nk_e5m2_t const *vectors, nk_size_t vector_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t rows_begin,
                                                         nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e4m3_blackwell(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e4m3_blackwell(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                            nk_stream_t stream);
/** @copydoc nk_dots_pack_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e4m3_blackwell(nk_e4m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                    nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e4m3_blackwell(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e4m3_blackwell(nk_e4m3_t const *vectors, nk_size_t vector_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t rows_begin,
                                                         nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e3m2_blackwell(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e3m2_blackwell(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                            nk_stream_t stream);
/** @copydoc nk_dots_pack_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e3m2_blackwell(nk_e3m2_t const *b, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                    nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e3m2_blackwell(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e3m2_blackwell(nk_e3m2_t const *vectors, nk_size_t vector_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t rows_begin,
                                                         nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m3_blackwell(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m3_blackwell(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                            nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m3_blackwell(nk_e2m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                    nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m3_blackwell(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m3_blackwell(nk_e2m3_t const *vectors, nk_size_t vector_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t rows_begin,
                                                         nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i8_blackwell(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i8_blackwell(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_pack_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_i8_blackwell(nk_i8_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_i8_blackwell(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                                    nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                    nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_i8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i8_blackwell(nk_i8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                       nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                       nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u8_blackwell(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u8_blackwell(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_pack_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_u8_blackwell(nk_u8_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_u8_blackwell(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                                    nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                    nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u8_blackwell(nk_u8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                       nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                       nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_i4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i4_blackwell(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i4_blackwell(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_pack_i4_best */
NUMKONG_API nk_status_t nk_dots_pack_i4_blackwell(nk_i4x2_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_i4_best */
NUMKONG_API nk_status_t nk_dots_packed_i4_blackwell(nk_i4x2_t const *a, void const *b_packed, nk_i32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_i4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i4_blackwell(nk_i4x2_t const *vectors, nk_size_t vector_count,
                                                       nk_size_t depth, nk_size_t stride, nk_i32_t *result,
                                                       nk_size_t result_stride, nk_size_t rows_begin,
                                                       nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u4_blackwell(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u4_blackwell(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_pack_u4_best */
NUMKONG_API nk_status_t nk_dots_pack_u4_blackwell(nk_u4x2_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_u4_best */
NUMKONG_API nk_status_t nk_dots_packed_u4_blackwell(nk_u4x2_t const *a, void const *b_packed, nk_u32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u4_blackwell(nk_u4x2_t const *vectors, nk_size_t vector_count,
                                                       nk_size_t depth, nk_size_t stride, nk_u32_t *result,
                                                       nk_size_t result_stride, nk_size_t rows_begin,
                                                       nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m1_blackwell(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_nvfp4_blackwell(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp4_blackwell(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp8e4m3_blackwell(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_pack_size_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp8e5m2_blackwell(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m1_blackwell(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                            nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_nvfp4_blackwell(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                             nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp4_blackwell(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                             nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp8e4m3_blackwell(void const *b_packed, nk_size_t *columns,
                                                                 nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_packed_shape_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp8e5m2_blackwell(void const *b_packed, nk_size_t *columns,
                                                                 nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m1_blackwell(nk_e2m1x2_t const *b, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                    nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_nvfp4_blackwell(nk_nvfp4_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                     nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp4_blackwell(nk_mxfp4_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                     nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp8e4m3_blackwell(nk_mxfp8e4m3_cref_t const *b, nk_size_t columns,
                                                         nk_size_t depth, nk_size_t b_stride, void *b_packed,
                                                         nk_size_t columns_begin, nk_size_t columns_end,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp8e5m2_blackwell(nk_mxfp8e5m2_cref_t const *b, nk_size_t columns,
                                                         nk_size_t depth, nk_size_t b_stride, void *b_packed,
                                                         nk_size_t columns_begin, nk_size_t columns_end,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m1_blackwell(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_nvfp4_blackwell(nk_nvfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp4_blackwell(nk_mxfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp8e4m3_blackwell(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                           nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                           nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                           nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp8e5m2_blackwell(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                           nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                           nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                           nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m1_blackwell(nk_e2m1x2_t const *vectors, nk_size_t vector_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t rows_begin,
                                                         nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_nvfp4_blackwell(nk_nvfp4_cref_t const *vectors, nk_size_t vector_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t rows_begin,
                                                          nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp4_blackwell(nk_mxfp4_cref_t const *vectors, nk_size_t vector_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t rows_begin,
                                                          nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp8e4m3_blackwell(nk_mxfp8e4m3_cref_t const *vectors,
                                                              nk_size_t vector_count, nk_size_t depth, nk_size_t stride,
                                                              nk_f32_t *result, nk_size_t result_stride,
                                                              nk_size_t rows_begin, nk_size_t rows_end,
                                                              nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp8e5m2_blackwell(nk_mxfp8e5m2_cref_t const *vectors,
                                                              nk_size_t vector_count, nk_size_t depth, nk_size_t stride,
                                                              nk_f32_t *result, nk_size_t result_stride,
                                                              nk_size_t rows_begin, nk_size_t rows_end,
                                                              nk_stream_t stream);
#endif // NUMKONG_TARGET_BLACKWELL

/*  NVIDIA backends for the compute capability 12.x family, feeding Float8, Float6 and Float4 to the
 *  tensor cores natively. BF16, F16 and the integers there use the Ampere kernels, which already
 *  run at the native rate. */
#if NUMKONG_TARGET_BLACKWELLRTX
/** @copydoc nk_dots_pack_size_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e5m2_blackwellrtx(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e5m2_blackwellrtx(void const *b_packed, nk_size_t *columns,
                                                               nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_pack_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e5m2_blackwellrtx(nk_e5m2_t const *b, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                       nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e5m2_blackwellrtx(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e5m2_blackwellrtx(nk_e5m2_t const *vectors, nk_size_t vector_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t rows_begin,
                                                            nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e4m3_blackwellrtx(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e4m3_blackwellrtx(void const *b_packed, nk_size_t *columns,
                                                               nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_pack_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e4m3_blackwellrtx(nk_e4m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                       nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e4m3_blackwellrtx(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e4m3_blackwellrtx(nk_e4m3_t const *vectors, nk_size_t vector_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t rows_begin,
                                                            nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e3m2_blackwellrtx(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e3m2_blackwellrtx(void const *b_packed, nk_size_t *columns,
                                                               nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_pack_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e3m2_blackwellrtx(nk_e3m2_t const *b, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                       nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e3m2_blackwellrtx(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e3m2_blackwellrtx(nk_e3m2_t const *vectors, nk_size_t vector_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t rows_begin,
                                                            nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m3_blackwellrtx(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m3_blackwellrtx(void const *b_packed, nk_size_t *columns,
                                                               nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m3_blackwellrtx(nk_e2m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                       nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m3_blackwellrtx(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m3_blackwellrtx(nk_e2m3_t const *vectors, nk_size_t vector_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t rows_begin,
                                                            nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m1_blackwellrtx(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m1_blackwellrtx(void const *b_packed, nk_size_t *columns,
                                                               nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m1_blackwellrtx(nk_e2m1x2_t const *b, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                       nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m1_blackwellrtx(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m1_blackwellrtx(nk_e2m1x2_t const *vectors, nk_size_t vector_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t rows_begin,
                                                            nk_size_t rows_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_BLACKWELLRTX

/*  ROCm baseline on every AMD device: the CUDA baseline's source compiled by HIP. */
#if NUMKONG_TARGET_ROCM
/** @copydoc nk_dots_pack_size_f64_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f64_rocm(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f64_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f64_rocm(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_pack_f64_best */
NUMKONG_API nk_status_t nk_dots_pack_f64_rocm(nk_f64_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                              void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                              nk_stream_t stream);
/** @copydoc nk_dots_packed_f64_best */
NUMKONG_API nk_status_t nk_dots_packed_f64_rocm(nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f64_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f64_rocm(nk_f64_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_f32_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f32_rocm(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f32_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f32_rocm(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_pack_f32_best */
NUMKONG_API nk_status_t nk_dots_pack_f32_rocm(nk_f32_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                              void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                              nk_stream_t stream);
/** @copydoc nk_dots_packed_f32_best */
NUMKONG_API nk_status_t nk_dots_packed_f32_rocm(nk_f32_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f32_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f32_rocm(nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_bf16_rocm(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_bf16_rocm(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_bf16_rocm(nk_bf16_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_bf16_rocm(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_bf16_rocm(nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f16_rocm(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f16_rocm(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_pack_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_f16_rocm(nk_f16_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                              void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                              nk_stream_t stream);
/** @copydoc nk_dots_packed_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_f16_rocm(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f16_rocm(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e5m2_rocm(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e5m2_rocm(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_stream_t stream);
/** @copydoc nk_dots_pack_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e5m2_rocm(nk_e5m2_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e5m2_rocm(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e5m2_rocm(nk_e5m2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e4m3_rocm(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e4m3_rocm(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_stream_t stream);
/** @copydoc nk_dots_pack_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e4m3_rocm(nk_e4m3_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e4m3_rocm(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e4m3_rocm(nk_e4m3_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e3m2_rocm(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e3m2_rocm(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_stream_t stream);
/** @copydoc nk_dots_pack_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e3m2_rocm(nk_e3m2_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e3m2_rocm(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e3m2_rocm(nk_e3m2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m3_rocm(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m3_rocm(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m3_rocm(nk_e2m3_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m3_rocm(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m3_rocm(nk_e2m3_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m1_rocm(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m1_rocm(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m1_rocm(nk_e2m1x2_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m1_rocm(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                 nk_size_t rows, nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m1_rocm(nk_e2m1x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i8_rocm(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i8_rocm(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                     nk_stream_t stream);
/** @copydoc nk_dots_pack_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_i8_rocm(nk_i8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                             nk_stream_t stream);
/** @copydoc nk_dots_packed_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_i8_rocm(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                               nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_i8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i8_rocm(nk_i8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                  nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                  nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u8_rocm(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u8_rocm(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                     nk_stream_t stream);
/** @copydoc nk_dots_pack_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_u8_rocm(nk_u8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                             nk_stream_t stream);
/** @copydoc nk_dots_packed_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_u8_rocm(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                               nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u8_rocm(nk_u8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                  nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                  nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_i4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i4_rocm(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i4_rocm(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                     nk_stream_t stream);
/** @copydoc nk_dots_pack_i4_best */
NUMKONG_API nk_status_t nk_dots_pack_i4_rocm(nk_i4x2_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                             nk_stream_t stream);
/** @copydoc nk_dots_packed_i4_best */
NUMKONG_API nk_status_t nk_dots_packed_i4_rocm(nk_i4x2_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                               nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_i4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i4_rocm(nk_i4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                  nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                  nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u4_rocm(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u4_rocm(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                     nk_stream_t stream);
/** @copydoc nk_dots_pack_u4_best */
NUMKONG_API nk_status_t nk_dots_pack_u4_rocm(nk_u4x2_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                             nk_stream_t stream);
/** @copydoc nk_dots_packed_u4_best */
NUMKONG_API nk_status_t nk_dots_packed_u4_rocm(nk_u4x2_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                               nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u4_rocm(nk_u4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                  nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                  nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_ROCM

/*  AMD Instinct MI300 backends, gfx942, through its matrix cores and SIMT dot instructions. */
#if NUMKONG_TARGET_CDNA3
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_bf16_cdna3(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_bf16_cdna3(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_bf16_cdna3(nk_bf16_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_bf16_cdna3(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_bf16_cdna3(nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f16_cdna3(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f16_cdna3(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_stream_t stream);
/** @copydoc nk_dots_pack_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_f16_cdna3(nk_f16_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_f16_cdna3(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f16_cdna3(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e5m2_cdna3(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e5m2_cdna3(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e5m2_cdna3(nk_e5m2_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e5m2_cdna3(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e5m2_cdna3(nk_e5m2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e4m3_cdna3(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e4m3_cdna3(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e4m3_cdna3(nk_e4m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e4m3_cdna3(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e4m3_cdna3(nk_e4m3_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e3m2_cdna3(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e3m2_cdna3(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e3m2_cdna3(nk_e3m2_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e3m2_cdna3(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e3m2_cdna3(nk_e3m2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m3_cdna3(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m3_cdna3(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m3_cdna3(nk_e2m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m3_cdna3(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m3_cdna3(nk_e2m3_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m1_cdna3(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m1_cdna3(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m1_cdna3(nk_e2m1x2_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m1_cdna3(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                  nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m1_cdna3(nk_e2m1x2_t const *vectors, nk_size_t vector_count,
                                                     nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                     nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end,
                                                     nk_stream_t stream);
/** @copydoc nk_dots_pack_size_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i8_cdna3(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i8_cdna3(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_pack_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_i8_cdna3(nk_i8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                              void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                              nk_stream_t stream);
/** @copydoc nk_dots_packed_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_i8_cdna3(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_i8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i8_cdna3(nk_i8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_i4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i4_cdna3(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i4_cdna3(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_pack_i4_best */
NUMKONG_API nk_status_t nk_dots_pack_i4_cdna3(nk_i4x2_t const *b, nk_size_t columns, nk_size_t depth,
                                              nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                              nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_i4_best */
NUMKONG_API nk_status_t nk_dots_packed_i4_cdna3(nk_i4x2_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_i4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i4_cdna3(nk_i4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u8_cdna3(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u8_cdna3(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_pack_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_u8_cdna3(nk_u8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                              void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                              nk_stream_t stream);
/** @copydoc nk_dots_packed_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_u8_cdna3(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u8_cdna3(nk_u8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u4_cdna3(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u4_cdna3(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_pack_u4_best */
NUMKONG_API nk_status_t nk_dots_pack_u4_cdna3(nk_u4x2_t const *b, nk_size_t columns, nk_size_t depth,
                                              nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                              nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_u4_best */
NUMKONG_API nk_status_t nk_dots_packed_u4_cdna3(nk_u4x2_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u4_cdna3(nk_u4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_CDNA3

/*  AMD Instinct MI350 backends, gfx950, through its matrix cores. */
#if NUMKONG_TARGET_CDNA4
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_bf16_cdna4(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_bf16_cdna4(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_bf16_cdna4(nk_bf16_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_bf16_cdna4(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_bf16_cdna4(nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f16_cdna4(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f16_cdna4(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_stream_t stream);
/** @copydoc nk_dots_pack_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_f16_cdna4(nk_f16_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_f16_cdna4(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f16_cdna4(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e5m2_cdna4(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e5m2_cdna4(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e5m2_cdna4(nk_e5m2_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e5m2_cdna4(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e5m2_cdna4(nk_e5m2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e4m3_cdna4(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e4m3_cdna4(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e4m3_cdna4(nk_e4m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e4m3_cdna4(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e4m3_cdna4(nk_e4m3_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e3m2_cdna4(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e3m2_cdna4(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e3m2_cdna4(nk_e3m2_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e3m2_cdna4(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e3m2_cdna4(nk_e3m2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m3_cdna4(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m3_cdna4(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m3_cdna4(nk_e2m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m3_cdna4(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m3_cdna4(nk_e2m3_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m1_cdna4(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m1_cdna4(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m1_cdna4(nk_e2m1x2_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m1_cdna4(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                  nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m1_cdna4(nk_e2m1x2_t const *vectors, nk_size_t vector_count,
                                                     nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                     nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end,
                                                     nk_stream_t stream);
/** @copydoc nk_dots_pack_size_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i8_cdna4(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i8_cdna4(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_pack_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_i8_cdna4(nk_i8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                              void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                              nk_stream_t stream);
/** @copydoc nk_dots_packed_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_i8_cdna4(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_i8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i8_cdna4(nk_i8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u8_cdna4(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u8_cdna4(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_pack_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_u8_cdna4(nk_u8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                              void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                              nk_stream_t stream);
/** @copydoc nk_dots_packed_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_u8_cdna4(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u8_cdna4(nk_u8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_i4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i4_cdna4(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i4_cdna4(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_pack_i4_best */
NUMKONG_API nk_status_t nk_dots_pack_i4_cdna4(nk_i4x2_t const *b, nk_size_t columns, nk_size_t depth,
                                              nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                              nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_i4_best */
NUMKONG_API nk_status_t nk_dots_packed_i4_cdna4(nk_i4x2_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_i4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i4_cdna4(nk_i4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u4_cdna4(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u4_cdna4(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_pack_u4_best */
NUMKONG_API nk_status_t nk_dots_pack_u4_cdna4(nk_u4x2_t const *b, nk_size_t columns, nk_size_t depth,
                                              nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                              nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_u4_best */
NUMKONG_API nk_status_t nk_dots_packed_u4_cdna4(nk_u4x2_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u4_cdna4(nk_u4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_CDNA4

/*  AMD Instinct MI400 backends, through its matrix cores. */
#if NUMKONG_TARGET_CDNA5
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_bf16_cdna5(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_bf16_cdna5(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_bf16_cdna5(nk_bf16_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_bf16_cdna5(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_bf16_cdna5(nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f16_cdna5(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f16_cdna5(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_stream_t stream);
/** @copydoc nk_dots_pack_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_f16_cdna5(nk_f16_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_f16_cdna5(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f16_cdna5(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e5m2_cdna5(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e5m2_cdna5(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e5m2_cdna5(nk_e5m2_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e5m2_cdna5(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e5m2_cdna5(nk_e5m2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e4m3_cdna5(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e4m3_cdna5(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e4m3_cdna5(nk_e4m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e4m3_cdna5(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e4m3_cdna5(nk_e4m3_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e3m2_cdna5(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e3m2_cdna5(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e3m2_cdna5(nk_e3m2_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e3m2_cdna5(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e3m2_cdna5(nk_e3m2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m3_cdna5(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m3_cdna5(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m3_cdna5(nk_e2m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m3_cdna5(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m3_cdna5(nk_e2m3_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m1_cdna5(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m1_cdna5(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m1_cdna5(nk_e2m1x2_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m1_cdna5(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                  nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m1_cdna5(nk_e2m1x2_t const *vectors, nk_size_t vector_count,
                                                     nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                     nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end,
                                                     nk_stream_t stream);
/** @copydoc nk_dots_pack_size_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i8_cdna5(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i8_cdna5(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_pack_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_i8_cdna5(nk_i8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                              void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                              nk_stream_t stream);
/** @copydoc nk_dots_packed_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_i8_cdna5(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_i8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i8_cdna5(nk_i8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u8_cdna5(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u8_cdna5(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_pack_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_u8_cdna5(nk_u8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                              void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                              nk_stream_t stream);
/** @copydoc nk_dots_packed_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_u8_cdna5(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u8_cdna5(nk_u8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_i4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i4_cdna5(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i4_cdna5(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_pack_i4_best */
NUMKONG_API nk_status_t nk_dots_pack_i4_cdna5(nk_i4x2_t const *b, nk_size_t columns, nk_size_t depth,
                                              nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                              nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_i4_best */
NUMKONG_API nk_status_t nk_dots_packed_i4_cdna5(nk_i4x2_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_i4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i4_cdna5(nk_i4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u4_cdna5(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u4_cdna5(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_pack_u4_best */
NUMKONG_API nk_status_t nk_dots_pack_u4_cdna5(nk_u4x2_t const *b, nk_size_t columns, nk_size_t depth,
                                              nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                              nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_u4_best */
NUMKONG_API nk_status_t nk_dots_packed_u4_cdna5(nk_u4x2_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u4_cdna5(nk_u4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_CDNA5

/*  Metal baseline on every Apple GPU, which has no F64: every input widened exactly and multiplied
 *  with scalar FMAs, accumulating in F32, or exactly for integer codes. */
#if NUMKONG_TARGET_METAL
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_bf16_metal(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_bf16_metal(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_bf16_metal(nk_bf16_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_bf16_metal(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_bf16_metal(nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f16_metal(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f16_metal(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_stream_t stream);
/** @copydoc nk_dots_pack_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_f16_metal(nk_f16_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_f16_metal(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f16_metal(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e5m2_metal(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e5m2_metal(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e5m2_metal(nk_e5m2_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e5m2_metal(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e5m2_metal(nk_e5m2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e4m3_metal(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e4m3_metal(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e4m3_metal(nk_e4m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e4m3_metal(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e4m3_metal(nk_e4m3_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e3m2_metal(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e3m2_metal(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e3m2_metal(nk_e3m2_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e3m2_metal(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e3m2_metal(nk_e3m2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m3_metal(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m3_metal(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m3_metal(nk_e2m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m3_metal(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m3_metal(nk_e2m3_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m1_metal(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m1_metal(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m1_metal(nk_e2m1x2_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m1_metal(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                  nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m1_metal(nk_e2m1x2_t const *vectors, nk_size_t vector_count,
                                                     nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                     nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end,
                                                     nk_stream_t stream);
/** @copydoc nk_dots_pack_size_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i8_metal(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i8_metal(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_pack_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_i8_metal(nk_i8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                              void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                              nk_stream_t stream);
/** @copydoc nk_dots_packed_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_i8_metal(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_i8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i8_metal(nk_i8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u8_metal(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u8_metal(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_pack_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_u8_metal(nk_u8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                              void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                              nk_stream_t stream);
/** @copydoc nk_dots_packed_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_u8_metal(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u8_metal(nk_u8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_i4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i4_metal(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i4_metal(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_pack_i4_best */
NUMKONG_API nk_status_t nk_dots_pack_i4_metal(nk_i4x2_t const *b, nk_size_t columns, nk_size_t depth,
                                              nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                              nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_i4_best */
NUMKONG_API nk_status_t nk_dots_packed_i4_metal(nk_i4x2_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_i4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i4_metal(nk_i4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u4_metal(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u4_metal(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_pack_u4_best */
NUMKONG_API nk_status_t nk_dots_pack_u4_metal(nk_u4x2_t const *b, nk_size_t columns, nk_size_t depth,
                                              nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                              nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_u4_best */
NUMKONG_API nk_status_t nk_dots_packed_u4_metal(nk_u4x2_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u4_metal(nk_u4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);

/** @copydoc nk_dots_pack_size_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_nvfp4_metal(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_nvfp4_metal(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_nvfp4_metal(nk_nvfp4_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_nvfp4_metal(nk_nvfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_nvfp4_metal(nk_nvfp4_cref_t const *vectors, nk_size_t vector_count,
                                                      nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                      nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_pack_size_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp4_metal(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp4_metal(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp4_metal(nk_mxfp4_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp4_metal(nk_mxfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp4_metal(nk_mxfp4_cref_t const *vectors, nk_size_t vector_count,
                                                      nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                      nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end,
                                                      nk_stream_t stream);
/** @copydoc nk_dots_pack_size_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp6e2m3_metal(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp6e2m3_metal(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                             nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp6e2m3_metal(nk_mxfp6e2m3_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                     nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp6e2m3_metal(nk_mxfp6e2m3_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp6e2m3_metal(nk_mxfp6e2m3_cref_t const *vectors, nk_size_t vector_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t rows_begin,
                                                          nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp6e3m2_metal(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp6e3m2_metal(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                             nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp6e3m2_metal(nk_mxfp6e3m2_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                     nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp6e3m2_metal(nk_mxfp6e3m2_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp6e3m2_metal(nk_mxfp6e3m2_cref_t const *vectors, nk_size_t vector_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t rows_begin,
                                                          nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp8e4m3_metal(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp8e4m3_metal(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                             nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp8e4m3_metal(nk_mxfp8e4m3_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                     nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp8e4m3_metal(nk_mxfp8e4m3_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp8e4m3_metal(nk_mxfp8e4m3_cref_t const *vectors, nk_size_t vector_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t rows_begin,
                                                          nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp8e5m2_metal(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp8e5m2_metal(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                             nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp8e5m2_metal(nk_mxfp8e5m2_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                     nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp8e5m2_metal(nk_mxfp8e5m2_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp8e5m2_metal(nk_mxfp8e5m2_cref_t const *vectors, nk_size_t vector_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t rows_begin,
                                                          nk_size_t rows_end, nk_stream_t stream);

#endif // NUMKONG_TARGET_METAL

/*  Apple backends of Metal family 9, M3 and newer, through @c simdgroup_matrix on the GPU cores,
 *  every float format widened to F16 or BF16 and accumulating in F32. */
#if NUMKONG_TARGET_APPLE9
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_bf16_apple9(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_bf16_apple9(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_bf16_apple9(nk_bf16_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_bf16_apple9(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_bf16_apple9(nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f16_apple9(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f16_apple9(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_f16_apple9(nk_f16_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_f16_apple9(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f16_apple9(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e5m2_apple9(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e5m2_apple9(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e5m2_apple9(nk_e5m2_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e5m2_apple9(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e5m2_apple9(nk_e5m2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e4m3_apple9(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e4m3_apple9(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e4m3_apple9(nk_e4m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e4m3_apple9(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e4m3_apple9(nk_e4m3_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e3m2_apple9(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e3m2_apple9(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e3m2_apple9(nk_e3m2_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e3m2_apple9(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e3m2_apple9(nk_e3m2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m3_apple9(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m3_apple9(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m3_apple9(nk_e2m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m3_apple9(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m3_apple9(nk_e2m3_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m1_apple9(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m1_apple9(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m1_apple9(nk_e2m1x2_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m1_apple9(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m1_apple9(nk_e2m1x2_t const *vectors, nk_size_t vector_count,
                                                      nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                      nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end,
                                                      nk_stream_t stream);

/** @copydoc nk_dots_pack_size_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_nvfp4_apple9(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_nvfp4_apple9(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_nvfp4_apple9(nk_nvfp4_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_nvfp4_apple9(nk_nvfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_nvfp4_apple9(nk_nvfp4_cref_t const *vectors, nk_size_t vector_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t rows_begin,
                                                       nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp4_apple9(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp4_apple9(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp4_apple9(nk_mxfp4_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp4_apple9(nk_mxfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp4_apple9(nk_mxfp4_cref_t const *vectors, nk_size_t vector_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t rows_begin,
                                                       nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp6e2m3_apple9(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp6e2m3_apple9(void const *b_packed, nk_size_t *columns,
                                                              nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp6e2m3_apple9(nk_mxfp6e2m3_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                      nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp6e2m3_apple9(nk_mxfp6e2m3_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp6e2m3_apple9(nk_mxfp6e2m3_cref_t const *vectors, nk_size_t vector_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t rows_begin,
                                                           nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp6e3m2_apple9(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp6e3m2_apple9(void const *b_packed, nk_size_t *columns,
                                                              nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp6e3m2_apple9(nk_mxfp6e3m2_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                      nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp6e3m2_apple9(nk_mxfp6e3m2_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp6e3m2_apple9(nk_mxfp6e3m2_cref_t const *vectors, nk_size_t vector_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t rows_begin,
                                                           nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp8e4m3_apple9(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp8e4m3_apple9(void const *b_packed, nk_size_t *columns,
                                                              nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp8e4m3_apple9(nk_mxfp8e4m3_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                      nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp8e4m3_apple9(nk_mxfp8e4m3_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp8e4m3_apple9(nk_mxfp8e4m3_cref_t const *vectors, nk_size_t vector_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t rows_begin,
                                                           nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp8e5m2_apple9(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp8e5m2_apple9(void const *b_packed, nk_size_t *columns,
                                                              nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp8e5m2_apple9(nk_mxfp8e5m2_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                      nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp8e5m2_apple9(nk_mxfp8e5m2_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp8e5m2_apple9(nk_mxfp8e5m2_cref_t const *vectors, nk_size_t vector_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t rows_begin,
                                                           nk_size_t rows_end, nk_stream_t stream);

#endif // NUMKONG_TARGET_APPLE9

/*  Apple backends of Metal family 10, M5 and newer, through @c matmul2d on the Neural Accelerators:
 *  the integers exactly, BF16 and F16 accumulating in F32, and the 8-, 6- and 4-bit floats widened
 *  to F16 on the way in. */
#if NUMKONG_TARGET_APPLE10
/** @copydoc nk_dots_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_bf16_apple10(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_bf16_apple10(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_pack_bf16_best */
NUMKONG_API nk_status_t nk_dots_pack_bf16_apple10(nk_bf16_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_bf16_best */
NUMKONG_API nk_status_t nk_dots_packed_bf16_apple10(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_bf16_apple10(nk_bf16_t const *vectors, nk_size_t vector_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t rows_begin,
                                                       nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_size_f16_apple10(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_f16_apple10(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_pack_f16_best */
NUMKONG_API nk_status_t nk_dots_pack_f16_apple10(nk_f16_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                 nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_f16_best */
NUMKONG_API nk_status_t nk_dots_packed_f16_apple10(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                   nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                   nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_f16_best */
NUMKONG_API nk_status_t nk_dots_symmetric_f16_apple10(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e5m2_apple10(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e5m2_apple10(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_pack_e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e5m2_apple10(nk_e5m2_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e5m2_apple10(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e5m2_apple10(nk_e5m2_t const *vectors, nk_size_t vector_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t rows_begin,
                                                       nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e4m3_apple10(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e4m3_apple10(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_pack_e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e4m3_apple10(nk_e4m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e4m3_apple10(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e4m3_apple10(nk_e4m3_t const *vectors, nk_size_t vector_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t rows_begin,
                                                       nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e3m2_apple10(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e3m2_apple10(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_pack_e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_e3m2_apple10(nk_e3m2_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_e3m2_apple10(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e3m2_apple10(nk_e3m2_t const *vectors, nk_size_t vector_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t rows_begin,
                                                       nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m3_apple10(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m3_apple10(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m3_apple10(nk_e2m3_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m3_apple10(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m3_apple10(nk_e2m3_t const *vectors, nk_size_t vector_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t rows_begin,
                                                       nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_size_e2m1_apple10(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_e2m1_apple10(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                          nk_stream_t stream);
/** @copydoc nk_dots_pack_e2m1_best */
NUMKONG_API nk_status_t nk_dots_pack_e2m1_apple10(nk_e2m1x2_t const *b, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                  nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_e2m1_best */
NUMKONG_API nk_status_t nk_dots_packed_e2m1_apple10(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_dots_symmetric_e2m1_apple10(nk_e2m1x2_t const *vectors, nk_size_t vector_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t rows_begin,
                                                       nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i8_apple10(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i8_apple10(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_i8_best */
NUMKONG_API nk_status_t nk_dots_pack_i8_apple10(nk_i8_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_i8_best */
NUMKONG_API nk_status_t nk_dots_packed_i8_apple10(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_i8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i8_apple10(nk_i8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u8_apple10(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u8_apple10(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_u8_best */
NUMKONG_API nk_status_t nk_dots_pack_u8_apple10(nk_u8_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_u8_best */
NUMKONG_API nk_status_t nk_dots_packed_u8_apple10(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u8_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u8_apple10(nk_u8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);

/** @copydoc nk_dots_pack_size_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_nvfp4_apple10(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_nvfp4_apple10(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                           nk_stream_t stream);
/** @copydoc nk_dots_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_nvfp4_apple10(nk_nvfp4_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                   nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_nvfp4_apple10(nk_nvfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_nvfp4_apple10(nk_nvfp4_cref_t const *vectors, nk_size_t vector_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t rows_begin,
                                                        nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp4_apple10(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp4_apple10(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                           nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp4_apple10(nk_mxfp4_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                   nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp4_apple10(nk_mxfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp4_apple10(nk_mxfp4_cref_t const *vectors, nk_size_t vector_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t rows_begin,
                                                        nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp6e2m3_apple10(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp6e2m3_apple10(void const *b_packed, nk_size_t *columns,
                                                               nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp6e2m3_apple10(nk_mxfp6e2m3_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                       nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp6e2m3_apple10(nk_mxfp6e2m3_cref_t const *a, void const *b_packed,
                                                         nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                         nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp6e2m3_apple10(nk_mxfp6e2m3_cref_t const *vectors, nk_size_t vector_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t rows_begin,
                                                            nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp6e3m2_apple10(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp6e3m2_apple10(void const *b_packed, nk_size_t *columns,
                                                               nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp6e3m2_apple10(nk_mxfp6e3m2_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                       nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp6e3m2_apple10(nk_mxfp6e3m2_cref_t const *a, void const *b_packed,
                                                         nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                         nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp6e3m2_apple10(nk_mxfp6e3m2_cref_t const *vectors, nk_size_t vector_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t rows_begin,
                                                            nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp8e4m3_apple10(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp8e4m3_apple10(void const *b_packed, nk_size_t *columns,
                                                               nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp8e4m3_apple10(nk_mxfp8e4m3_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                       nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp8e4m3_apple10(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                         nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                         nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp8e4m3_apple10(nk_mxfp8e4m3_cref_t const *vectors, nk_size_t vector_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t rows_begin,
                                                            nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp8e5m2_apple10(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp8e5m2_apple10(void const *b_packed, nk_size_t *columns,
                                                               nk_size_t *depth, nk_stream_t stream);
/** @copydoc nk_dots_pack_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_pack_mxfp8e5m2_apple10(nk_mxfp8e5m2_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                       nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_packed_mxfp8e5m2_apple10(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                         nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                         nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                         nk_stream_t stream);
/** @copydoc nk_dots_symmetric_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp8e5m2_apple10(nk_mxfp8e5m2_cref_t const *vectors, nk_size_t vector_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t rows_begin,
                                                            nk_size_t rows_end, nk_stream_t stream);

/** @copydoc nk_dots_pack_size_i4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_i4_apple10(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_i4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_i4_apple10(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_i4_best */
NUMKONG_API nk_status_t nk_dots_pack_i4_apple10(nk_i4x2_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_i4_best */
NUMKONG_API nk_status_t nk_dots_packed_i4_apple10(nk_i4x2_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_i4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_i4_apple10(nk_i4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);
/** @copydoc nk_dots_pack_size_u4_best */
NUMKONG_API nk_status_t nk_dots_pack_size_u4_apple10(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_dots_packed_shape_u4_best */
NUMKONG_API nk_status_t nk_dots_packed_shape_u4_apple10(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_stream_t stream);
/** @copydoc nk_dots_pack_u4_best */
NUMKONG_API nk_status_t nk_dots_pack_u4_apple10(nk_u4x2_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_stream_t stream);
/** @copydoc nk_dots_packed_u4_best */
NUMKONG_API nk_status_t nk_dots_packed_u4_apple10(nk_u4x2_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                                  nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                  nk_size_t c_stride, nk_stream_t stream);
/** @copydoc nk_dots_symmetric_u4_best */
NUMKONG_API nk_status_t nk_dots_symmetric_u4_apple10(nk_u4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                     nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream);

#endif // NUMKONG_TARGET_APPLE10

/**
 *  @brief Finds the dots kernel of @p kind for @p dtype, from the best of @p capabilities.
 *  @param[out] kernel The kernel, or null when none of @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *  @return @c nk_success_k, @c nk_missing_kernel_k, or @c nk_missing_library_k when header-only.
 */
NUMKONG_API nk_status_t nk_dots_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                            nk_kernel_punned_t *kernel, nk_capability_t *capability);

#if defined(__cplusplus)
} // extern "C"
#endif

#if NUMKONG_HEADER_ONLY
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
#include "numkong/dots/v128.h"
#include "numkong/dots/v128relaxed.h"
#include "numkong/dots/loongsonasx.h"
#include "numkong/dots/cuda.cuh"
#include "numkong/dots/rocm.cuh"
#include "numkong/dots/ampere.cuh"
#include "numkong/dots/hopper.cuh"
#include "numkong/dots/blackwell.cuh"
#include "numkong/dots/blackwellrtx.cuh"
#include "numkong/dots/cdna3.cuh"
#include "numkong/dots/cdna4.cuh"
#include "numkong/dots/cdna5.cuh"
#include "numkong/dots/metal.h"
#include "numkong/dots/apple9.h"
#include "numkong/dots/apple10.h"

#if defined(__cplusplus)
extern "C" {
#endif

NUMKONG_API nk_status_t nk_dots_pack_size_f64_best(nk_size_t columns, nk_size_t depth, nk_capability_t capabilities,
                                                   nk_size_t *bytes) {
    nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_packed_shape_f64_best(void const *packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(packed), nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_pack_f64_best(nk_f64_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                              void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                              nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(b), nk_unused_(columns), nk_unused_(depth), nk_unused_(b_stride), nk_unused_(b_packed),
        nk_unused_(columns_begin), nk_unused_(columns_end), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_packed_f64_best(nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_symmetric_f64_best(nk_f64_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end,
                                                   nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(vectors), nk_unused_(vector_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(rows_begin), nk_unused_(rows_end), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_pack_size_f32_best(nk_size_t columns, nk_size_t depth, nk_capability_t capabilities,
                                                   nk_size_t *bytes) {
    nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_packed_shape_f32_best(void const *packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(packed), nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_pack_f32_best(nk_f32_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                              void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                              nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(b), nk_unused_(columns), nk_unused_(depth), nk_unused_(b_stride), nk_unused_(b_packed),
        nk_unused_(columns_begin), nk_unused_(columns_end), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_packed_f32_best(nk_f32_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_symmetric_f32_best(nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end,
                                                   nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(vectors), nk_unused_(vector_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(rows_begin), nk_unused_(rows_end), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_pack_size_bf16_best(nk_size_t columns, nk_size_t depth, nk_capability_t capabilities,
                                                    nk_size_t *bytes) {
    nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_packed_shape_bf16_best(void const *packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(packed), nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_pack_bf16_best(nk_bf16_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_capability_t capabilities,
                                               nk_stream_t stream) {
    nk_unused_(b), nk_unused_(columns), nk_unused_(depth), nk_unused_(b_stride), nk_unused_(b_packed),
        nk_unused_(columns_begin), nk_unused_(columns_end), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_packed_bf16_best(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_symmetric_bf16_best(nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end,
                                                    nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(vectors), nk_unused_(vector_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(rows_begin), nk_unused_(rows_end), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_pack_size_f16_best(nk_size_t columns, nk_size_t depth, nk_capability_t capabilities,
                                                   nk_size_t *bytes) {
    nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_packed_shape_f16_best(void const *packed, nk_size_t *columns, nk_size_t *depth,
                                                      nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(packed), nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_pack_f16_best(nk_f16_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                              void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                              nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(b), nk_unused_(columns), nk_unused_(depth), nk_unused_(b_stride), nk_unused_(b_packed),
        nk_unused_(columns_begin), nk_unused_(columns_end), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_packed_f16_best(nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                nk_size_t c_stride, nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_symmetric_f16_best(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t rows_begin, nk_size_t rows_end,
                                                   nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(vectors), nk_unused_(vector_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(rows_begin), nk_unused_(rows_end), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_pack_size_e5m2_best(nk_size_t columns, nk_size_t depth, nk_capability_t capabilities,
                                                    nk_size_t *bytes) {
    nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_packed_shape_e5m2_best(void const *packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(packed), nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_pack_e5m2_best(nk_e5m2_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_capability_t capabilities,
                                               nk_stream_t stream) {
    nk_unused_(b), nk_unused_(columns), nk_unused_(depth), nk_unused_(b_stride), nk_unused_(b_packed),
        nk_unused_(columns_begin), nk_unused_(columns_end), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_packed_e5m2_best(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_symmetric_e5m2_best(nk_e5m2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end,
                                                    nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(vectors), nk_unused_(vector_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(rows_begin), nk_unused_(rows_end), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_pack_size_e4m3_best(nk_size_t columns, nk_size_t depth, nk_capability_t capabilities,
                                                    nk_size_t *bytes) {
    nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_packed_shape_e4m3_best(void const *packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(packed), nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_pack_e4m3_best(nk_e4m3_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_capability_t capabilities,
                                               nk_stream_t stream) {
    nk_unused_(b), nk_unused_(columns), nk_unused_(depth), nk_unused_(b_stride), nk_unused_(b_packed),
        nk_unused_(columns_begin), nk_unused_(columns_end), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_packed_e4m3_best(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_symmetric_e4m3_best(nk_e4m3_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end,
                                                    nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(vectors), nk_unused_(vector_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(rows_begin), nk_unused_(rows_end), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_pack_size_e3m2_best(nk_size_t columns, nk_size_t depth, nk_capability_t capabilities,
                                                    nk_size_t *bytes) {
    nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_packed_shape_e3m2_best(void const *packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(packed), nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_pack_e3m2_best(nk_e3m2_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_capability_t capabilities,
                                               nk_stream_t stream) {
    nk_unused_(b), nk_unused_(columns), nk_unused_(depth), nk_unused_(b_stride), nk_unused_(b_packed),
        nk_unused_(columns_begin), nk_unused_(columns_end), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_packed_e3m2_best(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_symmetric_e3m2_best(nk_e3m2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end,
                                                    nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(vectors), nk_unused_(vector_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(rows_begin), nk_unused_(rows_end), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_pack_size_e2m3_best(nk_size_t columns, nk_size_t depth, nk_capability_t capabilities,
                                                    nk_size_t *bytes) {
    nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_packed_shape_e2m3_best(void const *packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(packed), nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_pack_e2m3_best(nk_e2m3_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_capability_t capabilities,
                                               nk_stream_t stream) {
    nk_unused_(b), nk_unused_(columns), nk_unused_(depth), nk_unused_(b_stride), nk_unused_(b_packed),
        nk_unused_(columns_begin), nk_unused_(columns_end), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_packed_e2m3_best(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_symmetric_e2m3_best(nk_e2m3_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end,
                                                    nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(vectors), nk_unused_(vector_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(rows_begin), nk_unused_(rows_end), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_pack_size_e2m1_best(nk_size_t columns, nk_size_t depth, nk_capability_t capabilities,
                                                    nk_size_t *bytes) {
    nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_dots_pack_size_nvfp4_best(nk_size_t columns, nk_size_t depth, nk_capability_t capabilities,
                                                     nk_size_t *bytes) {
    nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp4_best(nk_size_t columns, nk_size_t depth, nk_capability_t capabilities,
                                                     nk_size_t *bytes) {
    nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp8e4m3_best(nk_size_t columns, nk_size_t depth,
                                                         nk_capability_t capabilities, nk_size_t *bytes) {
    nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp6e2m3_best(nk_size_t columns, nk_size_t depth,
                                                         nk_capability_t capabilities, nk_size_t *bytes) {
    nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp6e3m2_best(nk_size_t columns, nk_size_t depth,
                                                         nk_capability_t capabilities, nk_size_t *bytes) {
    nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_dots_pack_size_mxfp8e5m2_best(nk_size_t columns, nk_size_t depth,
                                                         nk_capability_t capabilities, nk_size_t *bytes) {
    nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_packed_shape_e2m1_best(void const *packed, nk_size_t *columns, nk_size_t *depth,
                                                       nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(packed), nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_nvfp4_best(void const *packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(packed), nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp4_best(void const *packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(packed), nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp8e4m3_best(void const *packed, nk_size_t *columns, nk_size_t *depth,
                                                            nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(packed), nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp6e2m3_best(void const *packed, nk_size_t *columns, nk_size_t *depth,
                                                            nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(packed), nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp6e3m2_best(void const *packed, nk_size_t *columns, nk_size_t *depth,
                                                            nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(packed), nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_dots_packed_shape_mxfp8e5m2_best(void const *packed, nk_size_t *columns, nk_size_t *depth,
                                                            nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(packed), nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_pack_e2m1_best(nk_e2m1x2_t const *b, nk_size_t columns, nk_size_t depth,
                                               nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                               nk_size_t columns_end, nk_capability_t capabilities,
                                               nk_stream_t stream) {
    nk_unused_(b), nk_unused_(columns), nk_unused_(depth), nk_unused_(b_stride), nk_unused_(b_packed),
        nk_unused_(columns_begin), nk_unused_(columns_end), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_dots_pack_nvfp4_best(nk_nvfp4_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_capability_t capabilities,
                                                nk_stream_t stream) {
    nk_unused_(b), nk_unused_(columns), nk_unused_(depth), nk_unused_(b_stride), nk_unused_(b_packed),
        nk_unused_(columns_begin), nk_unused_(columns_end), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_dots_pack_mxfp4_best(nk_mxfp4_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                nk_size_t columns_end, nk_capability_t capabilities,
                                                nk_stream_t stream) {
    nk_unused_(b), nk_unused_(columns), nk_unused_(depth), nk_unused_(b_stride), nk_unused_(b_packed),
        nk_unused_(columns_begin), nk_unused_(columns_end), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_dots_pack_mxfp8e4m3_best(nk_mxfp8e4m3_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                    nk_size_t columns_end, nk_capability_t capabilities,
                                                    nk_stream_t stream) {
    nk_unused_(b), nk_unused_(columns), nk_unused_(depth), nk_unused_(b_stride), nk_unused_(b_packed),
        nk_unused_(columns_begin), nk_unused_(columns_end), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_dots_pack_mxfp6e2m3_best(nk_mxfp6e2m3_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                    nk_size_t columns_end, nk_capability_t capabilities,
                                                    nk_stream_t stream) {
    nk_unused_(b), nk_unused_(columns), nk_unused_(depth), nk_unused_(b_stride), nk_unused_(b_packed),
        nk_unused_(columns_begin), nk_unused_(columns_end), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_dots_pack_mxfp6e3m2_best(nk_mxfp6e3m2_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                    nk_size_t columns_end, nk_capability_t capabilities,
                                                    nk_stream_t stream) {
    nk_unused_(b), nk_unused_(columns), nk_unused_(depth), nk_unused_(b_stride), nk_unused_(b_packed),
        nk_unused_(columns_begin), nk_unused_(columns_end), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_dots_pack_mxfp8e5m2_best(nk_mxfp8e5m2_cref_t const *b, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                    nk_size_t columns_end, nk_capability_t capabilities,
                                                    nk_stream_t stream) {
    nk_unused_(b), nk_unused_(columns), nk_unused_(depth), nk_unused_(b_stride), nk_unused_(b_packed),
        nk_unused_(columns_begin), nk_unused_(columns_end), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_packed_e2m1_best(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                 nk_size_t rows, nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                 nk_size_t c_stride, nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_dots_packed_nvfp4_best(nk_nvfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                  nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t a_stride, nk_size_t c_stride, nk_capability_t capabilities,
                                                  nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_dots_packed_mxfp4_best(nk_mxfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                  nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t a_stride, nk_size_t c_stride, nk_capability_t capabilities,
                                                  nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_dots_packed_mxfp8e4m3_best(nk_mxfp8e4m3_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride,
                                                      nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_dots_packed_mxfp6e2m3_best(nk_mxfp6e2m3_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride,
                                                      nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_dots_packed_mxfp6e3m2_best(nk_mxfp6e3m2_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride,
                                                      nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_dots_packed_mxfp8e5m2_best(nk_mxfp8e5m2_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride,
                                                      nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_symmetric_e2m1_best(nk_e2m1x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t rows_begin, nk_size_t rows_end,
                                                    nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(vectors), nk_unused_(vector_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(rows_begin), nk_unused_(rows_end), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_dots_symmetric_nvfp4_best(nk_nvfp4_cref_t const *vectors, nk_size_t vector_count,
                                                     nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                     nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end,
                                                     nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(vectors), nk_unused_(vector_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(rows_begin), nk_unused_(rows_end), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp4_best(nk_mxfp4_cref_t const *vectors, nk_size_t vector_count,
                                                     nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                     nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end,
                                                     nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(vectors), nk_unused_(vector_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(rows_begin), nk_unused_(rows_end), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp8e4m3_best(nk_mxfp8e4m3_cref_t const *vectors, nk_size_t vector_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t rows_begin,
                                                         nk_size_t rows_end, nk_capability_t capabilities,
                                                         nk_stream_t stream) {
    nk_unused_(vectors), nk_unused_(vector_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(rows_begin), nk_unused_(rows_end), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp6e2m3_best(nk_mxfp6e2m3_cref_t const *vectors, nk_size_t vector_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t rows_begin,
                                                         nk_size_t rows_end, nk_capability_t capabilities,
                                                         nk_stream_t stream) {
    nk_unused_(vectors), nk_unused_(vector_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(rows_begin), nk_unused_(rows_end), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp6e3m2_best(nk_mxfp6e3m2_cref_t const *vectors, nk_size_t vector_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t rows_begin,
                                                         nk_size_t rows_end, nk_capability_t capabilities,
                                                         nk_stream_t stream) {
    nk_unused_(vectors), nk_unused_(vector_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(rows_begin), nk_unused_(rows_end), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_dots_symmetric_mxfp8e5m2_best(nk_mxfp8e5m2_cref_t const *vectors, nk_size_t vector_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t rows_begin,
                                                         nk_size_t rows_end, nk_capability_t capabilities,
                                                         nk_stream_t stream) {
    nk_unused_(vectors), nk_unused_(vector_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(rows_begin), nk_unused_(rows_end), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_pack_size_i8_best(nk_size_t columns, nk_size_t depth, nk_capability_t capabilities,
                                                  nk_size_t *bytes) {
    nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_packed_shape_i8_best(void const *packed, nk_size_t *columns, nk_size_t *depth,
                                                     nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(packed), nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_pack_i8_best(nk_i8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                             nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(b), nk_unused_(columns), nk_unused_(depth), nk_unused_(b_stride), nk_unused_(b_packed),
        nk_unused_(columns_begin), nk_unused_(columns_end), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_packed_i8_best(nk_i8_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                               nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride, nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_symmetric_i8_best(nk_i8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                  nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                  nk_size_t rows_begin, nk_size_t rows_end,
                                                  nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(vectors), nk_unused_(vector_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(rows_begin), nk_unused_(rows_end), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_pack_size_i4_best(nk_size_t columns, nk_size_t depth, nk_capability_t capabilities,
                                                  nk_size_t *bytes) {
    nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_packed_shape_i4_best(void const *packed, nk_size_t *columns, nk_size_t *depth,
                                                     nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(packed), nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_pack_i4_best(nk_i4x2_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                             nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(b), nk_unused_(columns), nk_unused_(depth), nk_unused_(b_stride), nk_unused_(b_packed),
        nk_unused_(columns_begin), nk_unused_(columns_end), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_packed_i4_best(nk_i4x2_t const *a, void const *b_packed, nk_i32_t *c, nk_size_t rows,
                                               nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride, nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_symmetric_i4_best(nk_i4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                  nk_size_t stride, nk_i32_t *result, nk_size_t result_stride,
                                                  nk_size_t rows_begin, nk_size_t rows_end,
                                                  nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(vectors), nk_unused_(vector_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(rows_begin), nk_unused_(rows_end), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_pack_size_u8_best(nk_size_t columns, nk_size_t depth, nk_capability_t capabilities,
                                                  nk_size_t *bytes) {
    nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_packed_shape_u8_best(void const *packed, nk_size_t *columns, nk_size_t *depth,
                                                     nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(packed), nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_pack_u8_best(nk_u8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                             nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(b), nk_unused_(columns), nk_unused_(depth), nk_unused_(b_stride), nk_unused_(b_packed),
        nk_unused_(columns_begin), nk_unused_(columns_end), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_packed_u8_best(nk_u8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                               nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride, nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_symmetric_u8_best(nk_u8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                  nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                  nk_size_t rows_begin, nk_size_t rows_end,
                                                  nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(vectors), nk_unused_(vector_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(rows_begin), nk_unused_(rows_end), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_pack_size_u4_best(nk_size_t columns, nk_size_t depth, nk_capability_t capabilities,
                                                  nk_size_t *bytes) {
    nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_packed_shape_u4_best(void const *packed, nk_size_t *columns, nk_size_t *depth,
                                                     nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(packed), nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_pack_u4_best(nk_u4x2_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                             nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(b), nk_unused_(columns), nk_unused_(depth), nk_unused_(b_stride), nk_unused_(b_packed),
        nk_unused_(columns_begin), nk_unused_(columns_end), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_packed_u4_best(nk_u4x2_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                               nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride, nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_symmetric_u4_best(nk_u4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                  nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                  nk_size_t rows_begin, nk_size_t rows_end,
                                                  nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(vectors), nk_unused_(vector_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(rows_begin), nk_unused_(rows_end), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_pack_size_u1_best(nk_size_t columns, nk_size_t depth, nk_capability_t capabilities,
                                                  nk_size_t *bytes) {
    nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_packed_shape_u1_best(void const *packed, nk_size_t *columns, nk_size_t *depth,
                                                     nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(packed), nk_unused_(columns), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_pack_u1_best(nk_u1x8_t const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                             nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(b), nk_unused_(columns), nk_unused_(depth), nk_unused_(b_stride), nk_unused_(b_packed),
        nk_unused_(columns_begin), nk_unused_(columns_end), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_packed_u1_best(nk_u1x8_t const *a, void const *b_packed, nk_u32_t *c, nk_size_t rows,
                                               nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride, nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_symmetric_u1_best(nk_u1x8_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                  nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                  nk_size_t rows_begin, nk_size_t rows_end,
                                                  nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(vectors), nk_unused_(vector_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(rows_begin), nk_unused_(rows_end), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_dots_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                            nk_kernel_punned_t *kernel, nk_capability_t *capability) {
    nk_unused_(kind), nk_unused_(dtype), nk_unused_(capabilities);
    *kernel = NUMKONG_NULL, *capability = 0;
    return nk_missing_library_k;
}

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // NUMKONG_HEADER_ONLY

#endif
