/**
 *  @file include/numkong/cast.h
 *  @author Ash Vardanian
 *  @date October 8, 2023
 *  @brief SIMD-accelerated Type Conversions.
 *
 *  This file focuses on numeric types not uniformly supported across platforms, prioritizing:
 *
 *  - @c e5m2 & @c e4m3 ↔ @c f16 & @c bf16 - used for low-precision dot-products on modern CPUs,
 *  - @c e5m2 & @c e4m3 ↔ @c f32 - used for low-precision dot-products on older CPUs,
 *  - @c f16 & @c bf16 ↔ @c f32 - often used for half-precision dot-products on older CPUs,
 *
 *  Unlike most operation classes in NumKong, these are dependent on two input types: "from" & "to".
 *  It contains scalar helpers named like @c nk_f16_to_f32_serial as well as buffer-to-buffer
 *  vectorized operations akin to @c memcpy, such as @c nk_cast with @c nk_cast_serial,
 *  @c nk_cast_neon, @c nk_cast_skylake, and other platform-specific variants.
 *
 *  It also includes "partial load" and "partial store" type-punned helpers for IO between memory
 *  and registers, extensively reused in reductions, elementwise operations, and dot-products.
 *
 *  Float-format narrowing uses round-to-nearest, ties-to-even. Float-to-integer narrowing follows
 *  the same tie rule, saturates infinities, and maps NaNs to zero.
 *
 *  Given the breadth and sparsity of our type system, not all conversions carry equal weight: with
 *  ~16 numeric types, a dense matrix would need 21×21 = 441 conversions for:
 *
 *  @verbatim
 *          e4m3    e5m2    bf16    f16     f32     f64
 *                          bf16c   f16c    f32c    f64c
 *          i4      i8              i16     i32     i64
 *  u1      u4      u8              u16     u32     u64
 *  @endverbatim
 *
 *  To keep the design simple and broadly applicable to AI workloads, we route most conversions
 *  through a slower @b "hub-and-spoke" design, via an intermediate type such as @c f64 or @c i64.
 *
 */
#ifndef NK_CAST_H
#define NK_CAST_H

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Elementwise type-casting for arrays of entries.
 *
 *  @param[in] from The immutable input source array containing @p n elements of @p from_type type.
 *  @param[in] from_type The type of elements in the immutable source array.
 *  @param[in] n Counts dimensions, a multiple of the values per byte.
 *  @param[in] to The mutable output array containing @p n elements of @p to_type type.
 *  @param[in] to_type The type of elements in the mutable target array.
 */
NK_API_RUNTIME void nk_cast(void const *from, nk_dtype_t from_type, nk_size_t n, void *to, nk_dtype_t to_type);

/** @copydoc nk_cast */
NK_API_COMPTIME void nk_cast_serial(void const *from, nk_dtype_t from_type, nk_size_t n, void *to, nk_dtype_t to_type);

/**
 *  @brief Scalar conversion from @c f16 to @c f32, covering every IEEE 754 edge case.
 *
 *  @verbatim
 *      Input        F16 Hex   F32 Hex       Description
 *      +0           0x0000    0x00000000    Positive zero
 *      -0           0x8000    0x80000000    Negative zero
 *      +inf         0x7C00    0x7F800000    Positive infinity
 *      -inf         0xFC00    0xFF800000    Negative infinity
 *      NaN          0x7E00    0x7FC00000    Quiet NaN, payload preserved
 *      Min normal   0x0400    0x38800000    2⁻¹⁴
 *      Max normal   0x7BFF    0x477FE000    65504
 *      Min denorm   0x0001    0x33800000    2⁻²⁴
 *      Max denorm   0x03FF    0x387FC000    2⁻¹⁴ - 2⁻²⁴
 *  @endverbatim
 *
 *  @see Stack Overflow on half-precision conversion: https://stackoverflow.com/a/60047308
 *  @see Half-float conversion gist: https://gist.github.com/milhidaka/95863906fe828198f47991c813dbe233
 *  @see Libcanard float16 codec: https://github.com/OpenCyphal/libcanard/blob/636795f4bc395f56af8d2c61d3757b5e762bb9e5/canard.c#L811-L834
 */
NK_API_RUNTIME void nk_f16_to_f32(nk_f16_t const *src, nk_f32_t *dest);

/**
 *  @brief Scalar conversion from @c bf16 to @c f32.
 *
 *  @see Stack Overflow on f32 and bf16 conversion: https://stackoverflow.com/questions/55253233/convert-fp32-to-bfloat16-in-c/55254307#55254307
 *  @see Bfloat16 on Cloud TPUs: https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus
 */
NK_API_RUNTIME void nk_bf16_to_f32(nk_bf16_t const *src, nk_f32_t *dest);

/**
 *  @brief Scalar conversion from FP8 @c e4m3 to @c f32.
 *
 *  E4M3 has 1 sign bit, 4 exponent bits biased by 7, and 3 mantissa bits. It spans [-448, +448]
 *  with no infinities and only two NaN encodings, 0x7F and 0xFF. Subnormals decode to the value
 *  (-1)ˢ × mantissa × 2⁻⁹, which is mantissa / 512.
 *
 *  @verbatim
 *      Input        E4M3 Hex  F32 Hex       Description
 *      +0           0x00      0x00000000    Positive zero
 *      -0           0x80      0x80000000    Negative zero
 *      +NaN         0x7F      0x7FC00000    Quiet NaN, exp = 15, mant ≠ 0
 *      -NaN         0xFF      0xFFC00000    Quiet NaN, signed
 *      +448 (max)   0x7E      0x43E00000    Max normal = 448
 *      -448         0xFE      0xC3E00000    Min normal = -448
 *      1.0          0x38      0x3F800000    Normal, exp = 7, mant = 0
 *      Min denorm   0x01      0x3B000000    1/512 = 2⁻⁹
 *      Max denorm   0x07      0x3BE00000    7/512 = 7 × 2⁻⁹
 *  @endverbatim
 *
 *  @see FP8 Formats for Deep Learning: https://arxiv.org/pdf/2209.05433
 *  @see OCP 8-bit Floating Point Specification: https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1
 *  @see ONNX Float8 types: https://onnx.ai/onnx/technical/float8.html
 */
NK_API_RUNTIME void nk_e4m3_to_f32(nk_e4m3_t const *src, nk_f32_t *dest);

/** Scalar conversion from FP8 @c e5m2 to @c f32. */
NK_API_RUNTIME void nk_e5m2_to_f32(nk_e5m2_t const *src, nk_f32_t *dest);

/** Scalar conversion from FP6 @c e2m3 to @c f32. */
NK_API_RUNTIME void nk_e2m3_to_f32(nk_e2m3_t const *src, nk_f32_t *dest);

/** Scalar conversion from FP6 @c e3m2 to @c f32. */
NK_API_RUNTIME void nk_e3m2_to_f32(nk_e3m2_t const *src, nk_f32_t *dest);

/**
 *  @brief Scalar conversion from @c f32 to @c f16, rounding to nearest, for all IEEE 754 inputs.
 *
 *  @verbatim
 *      Input           F32 Hex       F16 Hex   Description
 *      +0              0x00000000    0x0000    Positive zero
 *      -0              0x80000000    0x8000    Negative zero
 *      +inf            0x7F800000    0x7C00    Positive infinity
 *      -inf            0xFF800000    0xFC00    Negative infinity
 *      NaN             0x7FC00000    0x7E00    Quiet NaN, payload truncated
 *      1.0             0x3F800000    0x3C00    Normal number
 *      65504           0x477FE000    0x7BFF    Max f16 normal
 *      65520+          >0x477FE000   0x7C00    Overflow → infinity
 *      2⁻¹⁴            0x38800000    0x0400    Min f16 normal
 *      2⁻²⁴            0x33800000    0x0001    Min f16 denormal
 *      <2⁻²⁵           <0x33000000   0x0000    Underflow → zero
 *  @endverbatim
 *
 *  @see Stack Overflow on half-precision conversion: https://stackoverflow.com/a/60047308
 *  @see Half-float conversion gist: https://gist.github.com/milhidaka/95863906fe828198f47991c813dbe233
 *  @see Libcanard float16 codec: https://github.com/OpenCyphal/libcanard/blob/636795f4bc395f56af8d2c61d3757b5e762bb9e5/canard.c#L811-L834
 */
NK_API_RUNTIME void nk_f32_to_f16(nk_f32_t const *src, nk_f16_t *dest);

/**
 *  @brief Scalar conversion from @c f32 to @c bf16.
 *
 *  @see Stack Overflow on f32 and bf16 conversion: https://stackoverflow.com/questions/55253233/convert-fp32-to-bfloat16-in-c/55254307#55254307
 *  @see Bfloat16 on Cloud TPUs: https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus
 */
NK_API_RUNTIME void nk_f32_to_bf16(nk_f32_t const *src, nk_bf16_t *dest);

/**
 *  @brief Scalar conversion from @c f32 to FP8 @c e4m3, rounding to nearest even per IEEE 754 and
 *      the OCP FP8 specification.
 *
 *  E4M3 has 1 sign bit, 4 exponent bits biased by 7, and 3 mantissa bits. It spans [-448, +448]
 *  with no infinities and only two NaN encodings. Values with |x| < 2⁻⁶ are encoded as subnormals.
 *
 *  @verbatim
 *      Input        F32 Hex       E4M3 Hex  Description
 *      +0           0x00000000    0x00      Positive zero
 *      -0           0x80000000    0x80      Negative zero
 *      +inf         0x7F800000    0x7E      Saturates to max, +448
 *      -inf         0xFF800000    0xFE      Saturates to min, -448
 *      NaN          0x7FC00000    0x7F      Quiet NaN
 *      1.0          0x3F800000    0x38      Normal, exp = 7, mant = 0
 *      448+         >0x43E00000   0x7E      Overflow → max
 *      2⁻⁶          0x3E800000    0x08      Min normal
 *      ≤2⁻¹⁰        ≤0x3A800000   0x00      Underflow → zero, RNE boundary
 *  @endverbatim
 *
 *  @see FP8 Formats for Deep Learning: https://arxiv.org/pdf/2209.05433
 *  @see OCP 8-bit Floating Point Specification: https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1
 *  @see ONNX Float8 types: https://onnx.ai/onnx/technical/float8.html
 */
NK_API_RUNTIME void nk_f32_to_e4m3(nk_f32_t const *src, nk_e4m3_t *dest);

/**
 *  @brief Scalar conversion from @c f32 to FP8 @c e5m2, rounding to nearest even per IEEE 754 and
 *      the OCP FP8 specification.
 *
 *  E5M2 has 1 sign bit, 5 exponent bits biased by 15, and 2 mantissa bits. Its range is ±57344, and
 *  it keeps IEEE 754 infinities and NaNs. Values with |x| < 2⁻¹⁴ are encoded as subnormals.
 *
 *  @verbatim
 *      Input        F32 Hex       E5M2 Hex  Description
 *      +0           0x00000000    0x00      Positive zero
 *      -0           0x80000000    0x80      Negative zero
 *      +inf         0x7F800000    0x7C      Positive infinity
 *      -inf         0xFF800000    0xFC      Negative infinity
 *      NaN          0x7FC00000    0x7D      Quiet NaN
 *      1.0          0x3F800000    0x3C      Normal, exp = 15, mant = 0
 *      57344+       >0x47600000   0x7C      Overflow → infinity
 *      2⁻¹⁴         0x38800000    0x04      Min normal
 *      ≤2⁻¹⁷        ≤0x37000000   0x00      Underflow → zero, RNE boundary
 *  @endverbatim
 *
 *  @see FP8 Formats for Deep Learning: https://arxiv.org/pdf/2209.05433
 *  @see OCP 8-bit Floating Point Specification: https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1
 *  @see ONNX Float8 types: https://onnx.ai/onnx/technical/float8.html
 */
NK_API_RUNTIME void nk_f32_to_e5m2(nk_f32_t const *src, nk_e5m2_t *dest);

/**
 *  @brief Scalar conversion from @c f32 to FP6 @c e2m3, rounding to nearest even per IEEE 754.
 *
 *  E2M3FN has 1 sign bit, 2 exponent bits biased by 1, and 3 mantissa bits. It spans [-7.5, +7.5]
 *  with no infinities or NaNs, and saturates to the maximum on overflow. Values with |x| < 0.5 are
 *  encoded as subnormals.
 */
NK_API_RUNTIME void nk_f32_to_e2m3(nk_f32_t const *src, nk_e2m3_t *dest);

/**
 *  @brief Scalar conversion from @c f32 to FP6 @c e3m2, rounding to nearest even per IEEE 754.
 *
 *  E3M2FN has 1 sign bit, 3 exponent bits biased by 3, and 2 mantissa bits. It spans [-28, +28]
 *  with no infinities or NaNs, and saturates to the maximum on overflow. Values with |x| < 0.25
 *  are encoded as subnormals.
 */
NK_API_RUNTIME void nk_f32_to_e3m2(nk_f32_t const *src, nk_e3m2_t *dest);

/** @copydoc nk_f16_to_f32 */
NK_API_COMPTIME void nk_f16_to_f32_serial(nk_f16_t const *src, nk_f32_t *dest);
/** @copydoc nk_f32_to_f16 */
NK_API_COMPTIME void nk_f32_to_f16_serial(nk_f32_t const *src, nk_f16_t *dest);

/** @copydoc nk_bf16_to_f32
 *
 *  Uses the compiler's native @c __bf16 type when present, or widens the bits by hand otherwise. */
NK_API_COMPTIME void nk_bf16_to_f32_serial(nk_bf16_t const *src, nk_f32_t *dest);
/** @copydoc nk_f32_to_bf16 */
NK_API_COMPTIME void nk_f32_to_bf16_serial(nk_f32_t const *src, nk_bf16_t *dest);
/** @copydoc nk_e4m3_to_f32 */
NK_API_COMPTIME void nk_e4m3_to_f32_serial(nk_e4m3_t const *src, nk_f32_t *dest);
/** @copydoc nk_f32_to_e4m3 */
NK_API_COMPTIME void nk_f32_to_e4m3_serial(nk_f32_t const *src, nk_e4m3_t *dest);
/** @copydoc nk_e5m2_to_f32 */
NK_API_COMPTIME void nk_e5m2_to_f32_serial(nk_e5m2_t const *src, nk_f32_t *dest);
/** @copydoc nk_f32_to_e5m2 */
NK_API_COMPTIME void nk_f32_to_e5m2_serial(nk_f32_t const *src, nk_e5m2_t *dest);
/** @copydoc nk_e2m3_to_f32 */
NK_API_COMPTIME void nk_e2m3_to_f32_serial(nk_e2m3_t const *src, nk_f32_t *dest);
/** @copydoc nk_f32_to_e2m3 */
NK_API_COMPTIME void nk_f32_to_e2m3_serial(nk_f32_t const *src, nk_e2m3_t *dest);
/** @copydoc nk_e3m2_to_f32 */
NK_API_COMPTIME void nk_e3m2_to_f32_serial(nk_e3m2_t const *src, nk_f32_t *dest);
/** @copydoc nk_f32_to_e3m2 */
NK_API_COMPTIME void nk_f32_to_e3m2_serial(nk_f32_t const *src, nk_e3m2_t *dest);

/** Unpacks a byte of two E2M1 nibbles into two f32 values: the high nibble lands in `dest[0]` and
 *  the low nibble in `dest[1]`, as in @c nk_i4x2_t and @c nk_u4x2_t. */
NK_API_COMPTIME void nk_e2m1x2_to_f32x2_serial(nk_e2m1x2_t const *src, nk_f32_t *dest);

/** Packs two f32 values into one byte of two E2M1 nibbles: `src[0]` becomes the high nibble and
 *  `src[1]` the low one, as in @c nk_i4x2_t and @c nk_u4x2_t. */
NK_API_COMPTIME void nk_f32x2_to_e2m1x2_serial(nk_f32_t const *src, nk_e2m1x2_t *dest);

/**
 *  @brief Converts a UE8M0 power-of-two scale byte, as used by OCP MX, to f32.
 *
 *  A zero byte decodes to 0, 0xFF to NaN as the block-NaN sentinel, and any other v to 2^(v - 127).
 */
NK_API_COMPTIME void nk_ue8m0_to_f32_serial(nk_ue8m0_t const *src, nk_f32_t *dest);

/**
 *  @brief Converts an f32 magnitude to a UE8M0 power-of-two scale byte.
 *
 *  Rounds to the nearest power of two, ties to even in log2 space, per the OCP MX, NVIDIA and AMD
 *  convention. Plain ceil or floor biases the block's dynamic range, as floor underestimates it, so
 *  the split sits at the geometric midpoint √2 × 2^e. NaN maps to 0xFF as the block-NaN sentinel,
 *  zero and subnormals map to 0x00, and overflow saturates to 0xFE.
 */
NK_API_COMPTIME void nk_f32_to_ue8m0_serial(nk_f32_t const *src, nk_ue8m0_t *dest);

/** Converts a UE4M3 byte, the NVFP4 scale that is an E4M3 with its sign bit forced to 0, to f32. */
NK_API_COMPTIME void nk_ue4m3_to_f32_serial(nk_ue4m3_t const *src, nk_f32_t *dest);

/**
 *  @brief Converts an f32 magnitude to a UE4M3 NVFP4 scale byte.
 *
 *  Takes the absolute value first, as the sign bit is not representable, then rounds to nearest
 *  even like the underlying E4M3 encoder, matching the NVFP4 and OCP scale convention. NaN maps to
 *  the E4M3 NaN code 0x7F.
 */
NK_API_COMPTIME void nk_f32_to_ue4m3_serial(nk_f32_t const *src, nk_ue4m3_t *dest);

/** Decode one NVFP4 block (16 elements) to f32. @p tensor_scale is the per-tensor multiplier. */
NK_API_COMPTIME void nk_nvfp4_to_f32x16_serial(nk_nvfp4_t const *src, nk_f32_t tensor_scale, nk_f32_t *dest);

/** Encode 16 f32 values into one NVFP4 block, deriving a UE4M3 scale via per-block amax. */
NK_API_COMPTIME void nk_f32x16_to_nvfp4_serial(nk_f32_t const *src, nk_f32_t tensor_scale, nk_nvfp4_t *dest);

/** Decode one MXFP4 block (32 elements) to f32. */
NK_API_COMPTIME void nk_mxfp4_to_f32x32_serial(nk_mxfp4_t const *src, nk_f32_t *dest);

/** Encode 32 f32 values into one MXFP4 block, deriving a UE8M0 scale. */
NK_API_COMPTIME void nk_f32x32_to_mxfp4_serial(nk_f32_t const *src, nk_mxfp4_t *dest);

/** Decode one MXFP6 E2M3 block (32 elements) to f32. */
NK_API_COMPTIME void nk_mxfp6_e2m3_to_f32x32_serial(nk_mxfp6_e2m3_t const *src, nk_f32_t *dest);

/** Encode 32 f32 values into one MXFP6 E2M3 block. */
NK_API_COMPTIME void nk_f32x32_to_mxfp6_e2m3_serial(nk_f32_t const *src, nk_mxfp6_e2m3_t *dest);

/** Decode one MXFP6 E3M2 block (32 elements) to f32. */
NK_API_COMPTIME void nk_mxfp6_e3m2_to_f32x32_serial(nk_mxfp6_e3m2_t const *src, nk_f32_t *dest);

/** Encode 32 f32 values into one MXFP6 E3M2 block. */
NK_API_COMPTIME void nk_f32x32_to_mxfp6_e3m2_serial(nk_f32_t const *src, nk_mxfp6_e3m2_t *dest);

/** Decode one MXFP8 E4M3 block (32 elements) to f32. */
NK_API_COMPTIME void nk_mxfp8_e4m3_to_f32x32_serial(nk_mxfp8_e4m3_t const *src, nk_f32_t *dest);

/** Encode 32 f32 values into one MXFP8 E4M3 block. */
NK_API_COMPTIME void nk_f32x32_to_mxfp8_e4m3_serial(nk_f32_t const *src, nk_mxfp8_e4m3_t *dest);

/** Decode one MXFP8 E5M2 block (32 elements) to f32. */
NK_API_COMPTIME void nk_mxfp8_e5m2_to_f32x32_serial(nk_mxfp8_e5m2_t const *src, nk_f32_t *dest);

/** Encode 32 f32 values into one MXFP8 E5M2 block. */
NK_API_COMPTIME void nk_f32x32_to_mxfp8_e5m2_serial(nk_f32_t const *src, nk_mxfp8_e5m2_t *dest);

/** Decode one MXINT8 block (32 elements) to f32. */
NK_API_COMPTIME void nk_mxint8_to_f32x32_serial(nk_mxint8_t const *src, nk_f32_t *dest);

/** Encode 32 f32 values into one MXINT8 block. */
NK_API_COMPTIME void nk_f32x32_to_mxint8_serial(nk_f32_t const *src, nk_mxint8_t *dest);

/**
 *  @brief Unified cast between plain and block-scaled layouts, or between two block-scaled ones.
 *
 *  The direction follows from the format descriptors:
 *
 *  @verbatim
 *      Source          Destination     Action
 *      plain           plain           delegates to nk_cast
 *      plain           block-scaled    encodes: per-block amax, derived scale, quantization
 *      block-scaled    plain           decodes: reads the scale, dequantizes to the plain dtype
 *      block-scaled    block-scaled    transcodes: decode → encode
 *  @endverbatim
 *
 *  @param[in] from Source element bytes.
 *  @param[in] from_scales One scale byte per block, NULL when @p from_format is plain.
 *  @param[in] from_tensor_scale Per-tensor multiplier, NULL when @p from_format has none.
 *  @param[in] from_format Source layout descriptor, `nk_plain(dtype)` for plain data.
 *  @param[out] to Destination element bytes.
 *  @param[out] to_scales One scale byte per block, NULL when @p to_format is plain.
 *  @param[inout] to_tensor_scale Per-tensor multiplier, NULL when @p to_format has none.
 *  @param[in] to_format Destination layout descriptor.
 *  @param[in] count Logical element count, a multiple of both block sizes.
 *
 *  A non-NULL @p to_tensor_scale holding a non-zero value is applied as is; holding zero on encode,
 *  it receives the tensor scale the kernel derives from the data.
 */
NK_API_RUNTIME void nk_cast_block_scaled(                                                                      //
    void const *from, void const *from_scales, nk_scalar_buffer_t const *from_tensor_scale,                    //
    nk_block_scaled_format_t const *from_format,                                                               //
    void *to, void *to_scales, nk_scalar_buffer_t *to_tensor_scale, nk_block_scaled_format_t const *to_format, //
    nk_size_t count);

/**
 *  @copydoc nk_cast_block_scaled
 *
 *  The serial reference runs one loop for encode, decode and transcode, and reuses the element
 *  codec of @c nk_cast_serial for every element dtype, packed E2M1 included:
 *
 *  @verbatim
 *      for each chunk of lcm(from_block, to_block) elements:
 *          decode the chunk into an f32 scratch buffer, applying source scales if block-scaled
 *          encode the scratch buffer into the destination, deriving amax and scales if block-scaled
 *  @endverbatim
 */
NK_API_COMPTIME void nk_cast_block_scaled_serial(                                                              //
    void const *from, void const *from_scales, nk_scalar_buffer_t const *from_tensor_scale,                    //
    nk_block_scaled_format_t const *from_format,                                                               //
    void *to, void *to_scales, nk_scalar_buffer_t *to_tensor_scale, nk_block_scaled_format_t const *to_format, //
    nk_size_t count);

/** Number of element storage bytes needed for @p count logical elements of @p format. */
NK_API_COMPTIME nk_size_t nk_block_scaled_elements_size(nk_size_t count, nk_block_scaled_format_t format);

/** Number of scale storage bytes needed for @p count logical elements of @p format. */
NK_API_COMPTIME nk_size_t nk_block_scaled_scales_size(nk_size_t count, nk_block_scaled_format_t format);

/** `{nk_e2m1_k, nk_ue4m3_k, nk_f32_k, 16}` — NVIDIA NVFP4 (Blackwell-native). */
NK_API_COMPTIME nk_block_scaled_format_t nk_nvfp4(void);

/** `{nk_e2m1_k, nk_ue8m0_k, unknown, 32}` — OCP MXFP4. */
NK_API_COMPTIME nk_block_scaled_format_t nk_mxfp4(void);

/** `{nk_e2m3_k, nk_ue8m0_k, unknown, 32}` — OCP MXFP6 (E2M3 variant). */
NK_API_COMPTIME nk_block_scaled_format_t nk_mxfp6_e2m3(void);

/** `{nk_e3m2_k, nk_ue8m0_k, unknown, 32}` — OCP MXFP6 (E3M2 variant). */
NK_API_COMPTIME nk_block_scaled_format_t nk_mxfp6_e3m2(void);

/** `{nk_e4m3_k, nk_ue8m0_k, unknown, 32}` — OCP MXFP8 (E4M3 variant). */
NK_API_COMPTIME nk_block_scaled_format_t nk_mxfp8_e4m3(void);

/** `{nk_e5m2_k, nk_ue8m0_k, unknown, 32}` — OCP MXFP8 (E5M2 variant). */
NK_API_COMPTIME nk_block_scaled_format_t nk_mxfp8_e5m2(void);

/** `{nk_i8_k, nk_ue8m0_k, unknown, 32}` — OCP MXINT8. */
NK_API_COMPTIME nk_block_scaled_format_t nk_mxint8(void);

/** `{element_dtype, unknown, unknown, 0}` — plain scalar buffer of @p element_dtype. */
NK_API_COMPTIME nk_block_scaled_format_t nk_plain(nk_dtype_t element_dtype);

/** Build a block-scaled format descriptor from a composite @p dtype enum value. Returns
 *  `nk_plain(dtype)` when @p dtype is not a composite. */
NK_API_COMPTIME nk_block_scaled_format_t nk_block_scaled_format_of_dtype(nk_dtype_t dtype);

#if NK_TARGET_NEON
/** @copydoc nk_f16_to_f32 */
NK_API_COMPTIME void nk_f16_to_f32_neon(nk_f16_t const *src, nk_f32_t *dest);
/** @copydoc nk_f32_to_f16 */
NK_API_COMPTIME void nk_f32_to_f16_neon(nk_f32_t const *src, nk_f16_t *dest);
/** @copydoc nk_cast */
NK_API_COMPTIME void nk_cast_neon(void const *from, nk_dtype_t from_type, nk_size_t n, void *to, nk_dtype_t to_type);

/** @copydoc nk_cast_block_scaled
 *
 *  Reduces amax over f32x4 and multiplies by a broadcast reciprocal around the NEON element
 *  codec hub, as @c nk_cast_neon already packs E2M1, E4M3 and the other element formats. */
NK_API_COMPTIME void nk_cast_block_scaled_neon(                                                                //
    void const *from, void const *from_scales, nk_scalar_buffer_t const *from_tensor_scale,                    //
    nk_block_scaled_format_t const *from_format,                                                               //
    void *to, void *to_scales, nk_scalar_buffer_t *to_tensor_scale, nk_block_scaled_format_t const *to_format, //
    nk_size_t count);
#endif // NK_TARGET_NEON

#if NK_TARGET_HASWELL
/** @copydoc nk_f16_to_f32 */
NK_API_COMPTIME void nk_f16_to_f32_haswell(nk_f16_t const *src, nk_f32_t *dest);
/** @copydoc nk_f32_to_f16 */
NK_API_COMPTIME void nk_f32_to_f16_haswell(nk_f32_t const *src, nk_f16_t *dest);
/** @copydoc nk_cast */
NK_API_COMPTIME void nk_cast_haswell(void const *from, nk_dtype_t from_type, nk_size_t n, void *to, nk_dtype_t to_type);

/** @copydoc nk_cast_block_scaled
 *
 *  Uses an AVX2 amax and a broadcast reciprocal multiply around the serial element codec hub,
 *  mirroring @c nk_cast_block_scaled_skylake at 8 lanes. */
NK_API_COMPTIME void nk_cast_block_scaled_haswell(                                                             //
    void const *from, void const *from_scales, nk_scalar_buffer_t const *from_tensor_scale,                    //
    nk_block_scaled_format_t const *from_format,                                                               //
    void *to, void *to_scales, nk_scalar_buffer_t *to_tensor_scale, nk_block_scaled_format_t const *to_format, //
    nk_size_t count);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
/** @copydoc nk_cast */
NK_API_COMPTIME void nk_cast_skylake(void const *from, nk_dtype_t from_type, nk_size_t n, void *to, nk_dtype_t to_type);

/** @copydoc nk_cast_block_scaled
 *
 *  Uses an AVX-512 amax and a broadcast reciprocal multiply around the serial element codec hub,
 *  which vectorizes the dominant scale-derivation cost for MXFP8, MXFP6, MXFP4, MXINT8 and NVFP4
 *  without duplicating per-format packing logic. */
NK_API_COMPTIME void nk_cast_block_scaled_skylake(                                                             //
    void const *from, void const *from_scales, nk_scalar_buffer_t const *from_tensor_scale,                    //
    nk_block_scaled_format_t const *from_format,                                                               //
    void *to, void *to_scales, nk_scalar_buffer_t *to_tensor_scale, nk_block_scaled_format_t const *to_format, //
    nk_size_t count);
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_ICELAKE
/** @copydoc nk_cast */
NK_API_COMPTIME void nk_cast_icelake(void const *from, nk_dtype_t from_type, nk_size_t n, void *to, nk_dtype_t to_type);

/** @copydoc nk_cast_block_scaled
 *
 *  Mirrors @c nk_cast_block_scaled_skylake but routes the element codec through
 *  @c nk_cast_icelake, whose 32-wide BF16-LUT decodes of FP4 and FP6 replace Skylake's per-32-bit
 *  permutes, while the f32 scale derivation reuses the Skylake amax and reciprocal multiply. */
NK_API_COMPTIME void nk_cast_block_scaled_icelake(                                                             //
    void const *from, void const *from_scales, nk_scalar_buffer_t const *from_tensor_scale,                    //
    nk_block_scaled_format_t const *from_format,                                                               //
    void *to, void *to_scales, nk_scalar_buffer_t *to_tensor_scale, nk_block_scaled_format_t const *to_format, //
    nk_size_t count);
#endif // NK_TARGET_ICELAKE

#if NK_TARGET_SAPPHIRE
/** @copydoc nk_cast */
NK_API_COMPTIME void nk_cast_sapphire(void const *from, nk_dtype_t from_type, nk_size_t n, void *to,
                                      nk_dtype_t to_type);
/** @copydoc nk_f16_to_f32 */
NK_API_COMPTIME void nk_f16_to_f32_sapphire(nk_f16_t const *src, nk_f32_t *dest);
/** @copydoc nk_f32_to_f16 */
NK_API_COMPTIME void nk_f32_to_f16_sapphire(nk_f32_t const *src, nk_f16_t *dest);
#endif // NK_TARGET_SAPPHIRE

#if NK_TARGET_RVV
/** @copydoc nk_cast */
NK_API_COMPTIME void nk_cast_rvv(void const *from, nk_dtype_t from_type, nk_size_t n, void *to, nk_dtype_t to_type);
#endif // NK_TARGET_RVV

#if NK_TARGET_POWERVSX
/** @copydoc nk_cast */
NK_API_COMPTIME void nk_cast_powervsx(void const *from, nk_dtype_t from_type, nk_size_t n, void *to,
                                      nk_dtype_t to_type);

/** @copydoc nk_f16_to_f32
 *
 *  Converts through the POWER9 vector unit with @c xvcvhpsp. */
NK_API_COMPTIME void nk_f16_to_f32_powervsx(nk_f16_t const *src, nk_f32_t *dest);

/** @copydoc nk_f32_to_f16
 *
 *  Converts through the POWER9 vector unit with @c xvcvsphp. */
NK_API_COMPTIME void nk_f32_to_f16_powervsx(nk_f32_t const *src, nk_f16_t *dest);
#endif // NK_TARGET_POWERVSX

#if NK_TARGET_V128RELAXED
/** @copydoc nk_cast */
NK_API_COMPTIME void nk_cast_v128relaxed(void const *from, nk_dtype_t from_type, nk_size_t n, void *to,
                                         nk_dtype_t to_type);
#endif // NK_TARGET_V128RELAXED

#if defined(__cplusplus)
} // extern "C"
#endif

#include "numkong/cast/serial.h"
#include "numkong/cast/neon.h"
#include "numkong/cast/haswell.h"
#include "numkong/cast/skylake.h"
#include "numkong/cast/icelake.h"
#include "numkong/cast/sapphire.h"
#include "numkong/cast/rvv.h"
#include "numkong/cast/v128.h"
#include "numkong/cast/v128relaxed.h"
#include "numkong/cast/powervsx.h"
#include "numkong/cast/loongsonasx.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if !NK_RUNTIME_DISPATCH

NK_API_COMPTIME void nk_cast(void const *from, nk_dtype_t from_type, nk_size_t n, void *to, nk_dtype_t to_type) {
#if NK_TARGET_SAPPHIRE
    nk_cast_sapphire(from, from_type, n, to, to_type);
#elif NK_TARGET_ICELAKE
    nk_cast_icelake(from, from_type, n, to, to_type);
#elif NK_TARGET_SKYLAKE
    nk_cast_skylake(from, from_type, n, to, to_type);
#elif NK_TARGET_HASWELL
    nk_cast_haswell(from, from_type, n, to, to_type);
#elif NK_TARGET_POWERVSX
    nk_cast_powervsx(from, from_type, n, to, to_type);
#elif NK_TARGET_RVV
    nk_cast_rvv(from, from_type, n, to, to_type);
#elif NK_TARGET_NEON
    nk_cast_neon(from, from_type, n, to, to_type);
#elif NK_TARGET_V128RELAXED
    nk_cast_v128relaxed(from, from_type, n, to, to_type);
#else
    nk_cast_serial(from, from_type, n, to, to_type);
#endif
}

NK_API_COMPTIME void nk_f16_to_f32(nk_f16_t const *src, nk_f32_t *dest) {
#if NK_TARGET_SAPPHIRE
    nk_f16_to_f32_sapphire(src, dest);
#elif NK_TARGET_HASWELL
    nk_f16_to_f32_haswell(src, dest);
#elif NK_TARGET_POWERVSX
    nk_f16_to_f32_powervsx(src, dest);
#elif NK_TARGET_NEON
    nk_f16_to_f32_neon(src, dest);
#else
    nk_f16_to_f32_serial(src, dest);
#endif
}

NK_API_COMPTIME void nk_f32_to_f16(nk_f32_t const *src, nk_f16_t *dest) {
#if NK_TARGET_SAPPHIRE
    nk_f32_to_f16_sapphire(src, dest);
#elif NK_TARGET_HASWELL
    nk_f32_to_f16_haswell(src, dest);
#elif NK_TARGET_POWERVSX
    nk_f32_to_f16_powervsx(src, dest);
#elif NK_TARGET_NEON
    nk_f32_to_f16_neon(src, dest);
#else
    nk_f32_to_f16_serial(src, dest);
#endif
}

NK_API_COMPTIME void nk_bf16_to_f32(nk_bf16_t const *src, nk_f32_t *dest) { nk_bf16_to_f32_serial(src, dest); }
NK_API_COMPTIME void nk_f32_to_bf16(nk_f32_t const *src, nk_bf16_t *dest) { nk_f32_to_bf16_serial(src, dest); }
NK_API_COMPTIME void nk_e4m3_to_f32(nk_e4m3_t const *src, nk_f32_t *dest) { nk_e4m3_to_f32_serial(src, dest); }
NK_API_COMPTIME void nk_f32_to_e4m3(nk_f32_t const *src, nk_e4m3_t *dest) { nk_f32_to_e4m3_serial(src, dest); }
NK_API_COMPTIME void nk_e5m2_to_f32(nk_e5m2_t const *src, nk_f32_t *dest) { nk_e5m2_to_f32_serial(src, dest); }
NK_API_COMPTIME void nk_f32_to_e5m2(nk_f32_t const *src, nk_e5m2_t *dest) { nk_f32_to_e5m2_serial(src, dest); }
NK_API_COMPTIME void nk_e2m3_to_f32(nk_e2m3_t const *src, nk_f32_t *dest) { nk_e2m3_to_f32_serial(src, dest); }
NK_API_COMPTIME void nk_f32_to_e2m3(nk_f32_t const *src, nk_e2m3_t *dest) { nk_f32_to_e2m3_serial(src, dest); }
NK_API_COMPTIME void nk_e3m2_to_f32(nk_e3m2_t const *src, nk_f32_t *dest) { nk_e3m2_to_f32_serial(src, dest); }
NK_API_COMPTIME void nk_f32_to_e3m2(nk_f32_t const *src, nk_e3m2_t *dest) { nk_f32_to_e3m2_serial(src, dest); }

NK_API_COMPTIME void nk_cast_block_scaled(                                                                     //
    void const *from, void const *from_scales, nk_scalar_buffer_t const *from_tensor_scale,                    //
    nk_block_scaled_format_t const *from_format,                                                               //
    void *to, void *to_scales, nk_scalar_buffer_t *to_tensor_scale, nk_block_scaled_format_t const *to_format, //
    nk_size_t count) {
#if NK_TARGET_ICELAKE
    nk_cast_block_scaled_icelake(from, from_scales, from_tensor_scale, from_format, to, to_scales, to_tensor_scale,
                                 to_format, count);
#elif NK_TARGET_SKYLAKE
    nk_cast_block_scaled_skylake(from, from_scales, from_tensor_scale, from_format, to, to_scales, to_tensor_scale,
                                 to_format, count);
#elif NK_TARGET_HASWELL
    nk_cast_block_scaled_haswell(from, from_scales, from_tensor_scale, from_format, to, to_scales, to_tensor_scale,
                                 to_format, count);
#elif NK_TARGET_NEON
    nk_cast_block_scaled_neon(from, from_scales, from_tensor_scale, from_format, to, to_scales, to_tensor_scale,
                              to_format, count);
#else
    nk_cast_block_scaled_serial(from, from_scales, from_tensor_scale, from_format, to, to_scales, to_tensor_scale,
                                to_format, count);
#endif
}

#endif // !NK_RUNTIME_DISPATCH

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_CAST_H
