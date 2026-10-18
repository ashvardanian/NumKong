/**
 *  @brief SIMD-accelerated Type Conversions.
 *  @file include/numkong/cast.h
 *  @author Ash Vardanian
 *  @date January 2, 2026
 *
 *  This file focuses on numeric types not uniformly supported across platforms, prioritizing:
 *
 *  - `e5m2` & `e4m3` ↔ `f16` & `bf16` - used for low-precision dot-products on modern CPUs,
 *  - `e5m2` & `e4m3` ↔ `f32` - used for low-precision dot-products on older CPUs,
 *  - `f16` & `bf16` ↔ `f32` - often used for half-precision dot-products on older CPUs,
 *
 *  Unlike most operation classes in NumKong, these are dependent on two input types: "from" & "to".
 *  It contains scalar helpers named like `nk_f16_to_f32_serial` as well as buffer-to-buffer
 *  `memcpy`-like vectorized operations, such as `nk_cast` with `nk_cast_serial`,
 *  `nk_cast_neon`, `nk_cast_skylake`, and other platform-specific variants.
 *
 *  It also includes "partial load" and "partial store" type-punned helper functions for handling
 *  IO between memory and registers, that are extensively reused in reductions, elementwise ops, and
 *  dot-products.
 *
 *  Float-format narrowing uses round-to-nearest, ties-to-even. Float-to-integer narrowing follows
 *  the same tie rule, saturates infinities, and maps NaNs to zero.
 *
 *  Assuming the overall breadth and sparsity of our type system, its clear, that not all type conversions
 *  have equivalent relevance. With ~16 numeric types we'd be looking at 21x21=441 conversions for:
 *
 *              e4m3    e5m2    bf16    f16     f32     f64
 *                              bf16c   f16c    f32c    f64c
 *              i4      i8              i16     i32     i64
 *      u1      u4      u8              u16     u32     u64
 *
 *  To simplify the design and make it more broadly applicable in AI workloads, we implement a slower
 *  @b "hub-and-spoke" design to guiding most conversions through an intermediate type, like `f64` or `i64`.
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
 *  @param[in] from The immutable input source array containing `n` elements of `from_type` type.
 *  @param[in] from_type The type of elements in the immutable source array.
 *  @param[in] n The number of elements in both input and output arrays.
 *  @param[in] to The mutable output array containing `n` elements of `to_type` type.
 *  @param[in] to_type The type of elements in the mutable target array.
 */
NK_DYNAMIC void nk_cast(void const *from, nk_dtype_t from_type, nk_size_t n, void *to, nk_dtype_t to_type);

/** @copydoc nk_cast */
NK_PUBLIC void nk_cast_serial(void const *from, nk_dtype_t from_type, nk_size_t n, void *to, nk_dtype_t to_type);

/** @brief Scalar conversion from f16 to f32. */
NK_DYNAMIC void nk_f16_to_f32(nk_f16_t const *src, nk_f32_t *dest);
/** @brief Scalar conversion from bf16 to f32. */
NK_DYNAMIC void nk_bf16_to_f32(nk_bf16_t const *src, nk_f32_t *dest);
/** @brief Scalar conversion from e4m3 to f32. */
NK_DYNAMIC void nk_e4m3_to_f32(nk_e4m3_t const *src, nk_f32_t *dest);
/** @brief Scalar conversion from e5m2 to f32. */
NK_DYNAMIC void nk_e5m2_to_f32(nk_e5m2_t const *src, nk_f32_t *dest);
/** @brief Scalar conversion from e2m3 to f32. */
NK_DYNAMIC void nk_e2m3_to_f32(nk_e2m3_t const *src, nk_f32_t *dest);
/** @brief Scalar conversion from e3m2 to f32. */
NK_DYNAMIC void nk_e3m2_to_f32(nk_e3m2_t const *src, nk_f32_t *dest);

/** @brief Scalar conversion from f32 to f16. */
NK_DYNAMIC void nk_f32_to_f16(nk_f32_t const *src, nk_f16_t *dest);
/** @brief Scalar conversion from f32 to bf16. */
NK_DYNAMIC void nk_f32_to_bf16(nk_f32_t const *src, nk_bf16_t *dest);
/** @brief Scalar conversion from f32 to e4m3. */
NK_DYNAMIC void nk_f32_to_e4m3(nk_f32_t const *src, nk_e4m3_t *dest);
/** @brief Scalar conversion from f32 to e5m2. */
NK_DYNAMIC void nk_f32_to_e5m2(nk_f32_t const *src, nk_e5m2_t *dest);
/** @brief Scalar conversion from f32 to e2m3. */
NK_DYNAMIC void nk_f32_to_e2m3(nk_f32_t const *src, nk_e2m3_t *dest);
/** @brief Scalar conversion from f32 to e3m2. */
NK_DYNAMIC void nk_f32_to_e3m2(nk_f32_t const *src, nk_e3m2_t *dest);

/** @copydoc nk_f16_to_f32 */
NK_PUBLIC void nk_f16_to_f32_serial(nk_f16_t const *src, nk_f32_t *dest);
/** @copydoc nk_f32_to_f16 */
NK_PUBLIC void nk_f32_to_f16_serial(nk_f32_t const *src, nk_f16_t *dest);
/** @copydoc nk_bf16_to_f32 */
NK_PUBLIC void nk_bf16_to_f32_serial(nk_bf16_t const *src, nk_f32_t *dest);
/** @copydoc nk_f32_to_bf16 */
NK_PUBLIC void nk_f32_to_bf16_serial(nk_f32_t const *src, nk_bf16_t *dest);
/** @copydoc nk_e4m3_to_f32 */
NK_PUBLIC void nk_e4m3_to_f32_serial(nk_e4m3_t const *src, nk_f32_t *dest);
/** @copydoc nk_f32_to_e4m3 */
NK_PUBLIC void nk_f32_to_e4m3_serial(nk_f32_t const *src, nk_e4m3_t *dest);
/** @copydoc nk_e5m2_to_f32 */
NK_PUBLIC void nk_e5m2_to_f32_serial(nk_e5m2_t const *src, nk_f32_t *dest);
/** @copydoc nk_f32_to_e5m2 */
NK_PUBLIC void nk_f32_to_e5m2_serial(nk_f32_t const *src, nk_e5m2_t *dest);
/** @copydoc nk_e2m3_to_f32 */
NK_PUBLIC void nk_e2m3_to_f32_serial(nk_e2m3_t const *src, nk_f32_t *dest);
/** @copydoc nk_f32_to_e2m3 */
NK_PUBLIC void nk_f32_to_e2m3_serial(nk_f32_t const *src, nk_e2m3_t *dest);
/** @copydoc nk_e3m2_to_f32 */
NK_PUBLIC void nk_e3m2_to_f32_serial(nk_e3m2_t const *src, nk_f32_t *dest);
/** @copydoc nk_f32_to_e3m2 */
NK_PUBLIC void nk_f32_to_e3m2_serial(nk_f32_t const *src, nk_e3m2_t *dest);

#if NK_TARGET_NEON
/** @copydoc nk_cast */
NK_PUBLIC void nk_cast_neon(void const *from, nk_dtype_t from_type, nk_size_t n, void *to, nk_dtype_t to_type);
/** @copydoc nk_f16_to_f32 */
NK_PUBLIC void nk_f16_to_f32_neon(nk_f16_t const *src, nk_f32_t *dest);
/** @copydoc nk_f32_to_f16 */
NK_PUBLIC void nk_f32_to_f16_neon(nk_f32_t const *src, nk_f16_t *dest);
#endif // NK_TARGET_NEON

#if NK_TARGET_HASWELL
/** @copydoc nk_cast */
NK_PUBLIC void nk_cast_haswell(void const *from, nk_dtype_t from_type, nk_size_t n, void *to, nk_dtype_t to_type);
/** @copydoc nk_f16_to_f32 */
NK_PUBLIC void nk_f16_to_f32_haswell(nk_f16_t const *src, nk_f32_t *dest);
/** @copydoc nk_f32_to_f16 */
NK_PUBLIC void nk_f32_to_f16_haswell(nk_f32_t const *src, nk_f16_t *dest);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
/** @copydoc nk_cast */
NK_PUBLIC void nk_cast_skylake(void const *from, nk_dtype_t from_type, nk_size_t n, void *to, nk_dtype_t to_type);
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_ICELAKE
/** @copydoc nk_cast */
NK_PUBLIC void nk_cast_icelake(void const *from, nk_dtype_t from_type, nk_size_t n, void *to, nk_dtype_t to_type);
#endif // NK_TARGET_ICELAKE

#if NK_TARGET_SAPPHIRE
/** @copydoc nk_cast */
NK_PUBLIC void nk_cast_sapphire(void const *from, nk_dtype_t from_type, nk_size_t n, void *to, nk_dtype_t to_type);
/** @copydoc nk_f16_to_f32 */
NK_PUBLIC void nk_f16_to_f32_sapphire(nk_f16_t const *src, nk_f32_t *dest);
/** @copydoc nk_f32_to_f16 */
NK_PUBLIC void nk_f32_to_f16_sapphire(nk_f32_t const *src, nk_f16_t *dest);
#endif // NK_TARGET_SAPPHIRE

#if NK_TARGET_RVV
/** @copydoc nk_cast */
NK_PUBLIC void nk_cast_rvv(void const *from, nk_dtype_t from_type, nk_size_t n, void *to, nk_dtype_t to_type);
#endif // NK_TARGET_RVV

#if NK_TARGET_POWERVSX
/** @copydoc nk_cast */
NK_PUBLIC void nk_cast_powervsx(void const *from, nk_dtype_t from_type, nk_size_t n, void *to, nk_dtype_t to_type);
/** @copydoc nk_f16_to_f32 */
NK_PUBLIC void nk_f16_to_f32_powervsx(nk_f16_t const *src, nk_f32_t *dest);
/** @copydoc nk_f32_to_f16 */
NK_PUBLIC void nk_f32_to_f16_powervsx(nk_f32_t const *src, nk_f16_t *dest);
#endif // NK_TARGET_POWERVSX

#if NK_TARGET_V128RELAXED
/** @copydoc nk_cast */
NK_PUBLIC void nk_cast_v128relaxed(void const *from, nk_dtype_t from_type, nk_size_t n, void *to, nk_dtype_t to_type);
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
#include "numkong/cast/v128relaxed.h"
#include "numkong/cast/powervsx.h"
#include "numkong/cast/loongsonasx.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if !NK_DYNAMIC_DISPATCH

NK_PUBLIC void nk_cast(void const *from, nk_dtype_t from_type, nk_size_t n, void *to, nk_dtype_t to_type) {
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

NK_PUBLIC void nk_f16_to_f32(nk_f16_t const *src, nk_f32_t *dest) {
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

NK_PUBLIC void nk_f32_to_f16(nk_f32_t const *src, nk_f16_t *dest) {
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

NK_PUBLIC void nk_bf16_to_f32(nk_bf16_t const *src, nk_f32_t *dest) { nk_bf16_to_f32_serial(src, dest); }
NK_PUBLIC void nk_f32_to_bf16(nk_f32_t const *src, nk_bf16_t *dest) { nk_f32_to_bf16_serial(src, dest); }
NK_PUBLIC void nk_e4m3_to_f32(nk_e4m3_t const *src, nk_f32_t *dest) { nk_e4m3_to_f32_serial(src, dest); }
NK_PUBLIC void nk_f32_to_e4m3(nk_f32_t const *src, nk_e4m3_t *dest) { nk_f32_to_e4m3_serial(src, dest); }
NK_PUBLIC void nk_e5m2_to_f32(nk_e5m2_t const *src, nk_f32_t *dest) { nk_e5m2_to_f32_serial(src, dest); }
NK_PUBLIC void nk_f32_to_e5m2(nk_f32_t const *src, nk_e5m2_t *dest) { nk_f32_to_e5m2_serial(src, dest); }
NK_PUBLIC void nk_e2m3_to_f32(nk_e2m3_t const *src, nk_f32_t *dest) { nk_e2m3_to_f32_serial(src, dest); }
NK_PUBLIC void nk_f32_to_e2m3(nk_f32_t const *src, nk_e2m3_t *dest) { nk_f32_to_e2m3_serial(src, dest); }
NK_PUBLIC void nk_e3m2_to_f32(nk_e3m2_t const *src, nk_f32_t *dest) { nk_e3m2_to_f32_serial(src, dest); }
NK_PUBLIC void nk_f32_to_e3m2(nk_f32_t const *src, nk_e3m2_t *dest) { nk_f32_to_e3m2_serial(src, dest); }

#endif // !NK_DYNAMIC_DISPATCH

/** @brief Pack byte-addressable strided leaves around an existing contiguous cast kernel.
 *  Returns zero for packed sub-byte types, which require layout-specific handling. */
NK_INTERNAL int nk_cast_strided_(void const *from, nk_dtype_t from_type, nk_i64_t from_stride_bytes, nk_size_t n,
                                 void *to, nk_dtype_t to_type, nk_i64_t to_stride_bytes,
                                 void (*convert)(void const *, nk_dtype_t, nk_size_t, void *, nk_dtype_t)) {
    nk_size_t const from_bytes = nk_dtype_bits(from_type) / NK_BITS_PER_BYTE;
    nk_size_t const to_bytes = nk_dtype_bits(to_type) / NK_BITS_PER_BYTE;
    if (!from_bytes || !to_bytes) return 0;
    if (from_stride_bytes == (nk_i64_t)from_bytes && to_stride_bytes == (nk_i64_t)to_bytes) {
        convert(from, from_type, n, to, to_type);
        return 1;
    }

    // One cache-line of padding also accommodates older MSVC masked-store lowering.
    enum { batch_bytes = 1024, padding_bytes = 64 };
    NK_ALIGN64 nk_u8_t gathered[batch_bytes + padding_bytes];
    NK_ALIGN64 nk_u8_t converted[batch_bytes + padding_bytes];
    nk_size_t const largest_element = from_bytes > to_bytes ? from_bytes : to_bytes;
    nk_size_t const batch_count = batch_bytes / largest_element;
    for (nk_size_t offset = 0; offset < n;) {
        nk_size_t const count = n - offset < batch_count ? n - offset : batch_count;
        char const *source = (char const *)from + (nk_i64_t)offset * from_stride_bytes;
        char *destination = (char *)to + (nk_i64_t)offset * to_stride_bytes;
        if (from_stride_bytes != (nk_i64_t)from_bytes) {
            for (nk_size_t i = 0; i < count; ++i)
                nk_copy_bytes_(gathered + i * from_bytes, source + (nk_i64_t)i * from_stride_bytes, from_bytes);
            source = (char const *)gathered;
        }
        void *output = to_stride_bytes == (nk_i64_t)to_bytes ? (void *)destination : (void *)converted;
        convert(source, from_type, count, output, to_type);
        if (to_stride_bytes != (nk_i64_t)to_bytes)
            for (nk_size_t i = 0; i < count; ++i)
                nk_copy_bytes_(destination + (nk_i64_t)i * to_stride_bytes, converted + i * to_bytes, to_bytes);
        offset += count;
    }
    return 1;
}

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_CAST_H
