/**
 *  @file include/numkong/cast/simt.cuh
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Conversions on the SIMT cores of every CUDA and ROCm device.
 *
 *  @sa include/numkong/cast/serial.h
 *  @sa include/numkong/cast.h
 *
 *  Device twins of the serial casts, with their signatures, that the GPU families load and store
 *  through. Most forward to the serial helpers, which C++20 makes constexpr and so callable from
 *  device code; the table decoders of E5M2, E2M3, E3M2 and E2M1 get arithmetic twins. Every
 *  conversion matches its serial original bit for bit, NaN payloads included.
 *
 *  A bulk cast gives each thread one unit: a value, or the 2 or 8 values sharing a packed byte. It
 *  walks the hub the serial loop picks, U64, I64 or an F64 pair, except that values F32 holds
 *  exactly stay in an F32 pair, quieted the way the CPUs' F64 round trip quiets NaNs. A
 *  block-scaled cast gives each thread one chunk of the larger block and runs the serial loop on
 *  it. A derived destination tensor scale first takes the abs-max of the whole source, which
 *  @c atomicMax gathers into the tensor scale itself, while the first destination scale byte
 *  remembers whether the call derives it, until the chunks overwrite it.
 */
#ifndef NUMKONG_CAST_SIMT_CUH
#define NUMKONG_CAST_SIMT_CUH

#if NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_

#include "numkong/types.h"
#include "numkong/cast/serial.h" // `nk_f32_to_e5m2_`, `nk_block_scaled_encode_scale_serial_`

#if defined(__cplusplus)
extern "C" {
#endif

/*  Every tile places staged bytes and moves them through these, so each formula and each cast is
 *  written once. Loads and stores take shared or global addresses aligned to their width. */
#pragma region Addressing

/** Byte @p byte of row @p row of an operand in the 32, 64 or 128-byte swizzle that @p swizzle_bytes
 *  names, rows that wide: each 16-byte chunk XORs with the row's 128-byte line, modulo the chunks
 *  of a row, as the NVIDIA tensor cores and copy engines lay operands out. */
NUMKONG_DEVICE unsigned nk_swizzled_offset_(unsigned row, unsigned byte, unsigned swizzle_bytes) {
    unsigned const line = (row / (128 / swizzle_bytes)) & (swizzle_bytes / 16 - 1);
    return row * swizzle_bytes + (((byte >> 4) ^ line) << 4) + (byte & 15);
}

/** Byte @p byte of row @p row of an operand deeper than one swizzle: blocks of @p panel_bytes, each
 *  holding @p swizzle_bytes of every row in the swizzle of that width. */
NUMKONG_DEVICE unsigned nk_swizzled_panel_offset_(unsigned row, unsigned byte, unsigned swizzle_bytes,
                                                  unsigned panel_bytes) {
    return byte / swizzle_bytes * panel_bytes + nk_swizzled_offset_(row, byte % swizzle_bytes, swizzle_bytes);
}

NUMKONG_DEVICE uint4 nk_load_b128_(void const *address) { return *(uint4 const *)address; }
NUMKONG_DEVICE uint2 nk_load_b64_(void const *address) { return *(uint2 const *)address; }
NUMKONG_DEVICE nk_u32_t nk_load_b32_(void const *address) { return *(nk_u32_t const *)address; }
NUMKONG_DEVICE nk_u16_t nk_load_b16_(void const *address) { return *(nk_u16_t const *)address; }
NUMKONG_DEVICE void nk_store_b128_(void *address, uint4 value) { *(uint4 *)address = value; }
NUMKONG_DEVICE void nk_store_b32_(void *address, nk_u32_t value) { *(nk_u32_t *)address = value; }
NUMKONG_DEVICE void nk_store_b16_(void *address, nk_u16_t value) { *(nk_u16_t *)address = value; }

/** Loads the 16 bytes at @p address as the view every element type reads them through. */
NUMKONG_DEVICE nk_b128_vec_t nk_load_b128_vec_(void const *address) {
    uint4 const bits = nk_load_b128_(address);
    nk_b128_vec_t vector;
    vector.u32s[0] = bits.x, vector.u32s[1] = bits.y, vector.u32s[2] = bits.z, vector.u32s[3] = bits.w;
    return vector;
}

/** Stores the 16 bytes of @p vector at @p address. */
NUMKONG_DEVICE void nk_store_b128_vec_(void *address, nk_b128_vec_t const *vector) {
    nk_store_b128_(address, make_uint4(vector->u32s[0], vector->u32s[1], vector->u32s[2], vector->u32s[3]));
}

#pragma endregion Addressing

#pragma region Conversions

/** One E4M3 code over 256, its NaN code read as 480. */
NUMKONG_DEVICE nk_f32_t nk_e4m3_to_scaled_f32_(nk_u32_t code) {
    return __half2float(__ushort_as_half((unsigned short)(((code & 0x7Fu) << 7) | ((code & 0x80u) << 8))));
}

/** Widens one BF16 value to F32, exactly. */
NUMKONG_DEVICE void nk_bf16_to_f32_simt_(nk_bf16_t const *src, nk_f32_t *dest) {
    *dest = __uint_as_float((nk_u32_t)(*(unsigned short const *)src) << 16);
}

/** Widens one F16 value to F32, exactly, keeping NaN payloads unquieted like the serial cast. */
NUMKONG_DEVICE void nk_f16_to_f32_simt_(nk_f16_t const *src, nk_f32_t *dest) {
    nk_u32_t const bits = *(unsigned short const *)src;
    *dest = (bits & 0x7FFFu) > 0x7C00u
                ? __uint_as_float(((bits & 0x8000u) << 16) | 0x7F800000u | ((bits & 0x3FFu) << 13))
                : __half2float(__ushort_as_half((unsigned short)bits));
}

/** Widens one E5M2 value to F32, exactly, a NaN to the quiet NaN of its sign. */
NUMKONG_DEVICE void nk_e5m2_to_f32_simt_(nk_e5m2_t const *src, nk_f32_t *dest) {
    nk_u32_t const code = *src;
    *dest = (code & 0x7Fu) > 0x7Cu ? __uint_as_float(((code & 0x80u) << 24) | 0x7FC00000u)
                                   : __half2float(__ushort_as_half((unsigned short)(code << 8)));
}

/** Widens one E4M3FN value to F32, exactly, a NaN to the quiet NaN of its sign. */
NUMKONG_DEVICE void nk_e4m3_to_f32_simt_(nk_e4m3_t const *src, nk_f32_t *dest) {
    nk_u32_t const code = *src;
    *dest = (code & 0x7Fu) == 0x7Fu ? __uint_as_float(((code & 0x80u) << 24) | 0x7FC00000u)
                                    : nk_e4m3_to_scaled_f32_(code) * 256.0f;
}

/** Widens one E2M3FN value to F32, exactly, reading its low 6 bits. */
NUMKONG_DEVICE void nk_e2m3_to_f32_simt_(nk_e2m3_t const *src, nk_f32_t *dest) {
    nk_u32_t const code = *src, exponent = (code >> 3) & 3u, mantissa = code & 7u;
    nk_u32_t const magnitude = exponent ? ((exponent + 126u) << 23) | (mantissa << 20)
                                        : __float_as_uint((nk_f32_t)mantissa * 0.125f);
    *dest = __uint_as_float(((code & 0x20u) << 26) | magnitude);
}

/** Widens one E3M2FN value to F32, exactly, reading its low 6 bits. */
NUMKONG_DEVICE void nk_e3m2_to_f32_simt_(nk_e3m2_t const *src, nk_f32_t *dest) {
    nk_u32_t const code = *src, exponent = (code >> 2) & 7u, mantissa = code & 3u;
    nk_u32_t const magnitude = exponent ? ((exponent + 124u) << 23) | (mantissa << 21)
                                        : __float_as_uint((nk_f32_t)mantissa * 0.0625f);
    *dest = __uint_as_float(((code & 0x20u) << 26) | magnitude);
}

/** Twice the E2M1 value in the low 4 bits of @p nibble, an exact integer in [-12, +12]. */
NUMKONG_DEVICE nk_i32_t nk_e2m1_nibble_to_i8x2_simt_(nk_u32_t nibble) {
    nk_u32_t const exponent = (nibble >> 1) & 3u, mantissa = nibble & 1u;
    nk_i32_t const doubled = (nk_i32_t)(exponent ? (2u + mantissa) << (exponent - 1u) : mantissa);
    return nibble & 8u ? -doubled : doubled;
}

/** Widens the E2M1 value in the low 4 bits of @p nibble to F32, exactly, keeping a signed zero. */
NUMKONG_DEVICE nk_f32_t nk_e2m1_nibble_to_f32_simt_(nk_u32_t nibble) {
    nk_f32_t const magnitude = (nk_f32_t)nk_e2m1_nibble_to_i8x2_simt_(nibble & 7u) * 0.5f;
    return __uint_as_float(__float_as_uint(magnitude) | ((nibble & 8u) << 28));
}

/** Widens the two E2M1 nibbles of one byte to F32, the high nibble first. */
NUMKONG_DEVICE void nk_e2m1x2_to_f32x2_simt_(nk_e2m1x2_t const *src, nk_f32_t *dest) {
    dest[0] = nk_e2m1_nibble_to_f32_simt_((nk_u32_t)*src >> 4), dest[1] = nk_e2m1_nibble_to_f32_simt_(*src & 0x0Fu);
}

/** Narrows one F32 value to BF16, rounding to nearest even and keeping NaNs. */
NUMKONG_DEVICE void nk_f32_to_bf16_simt_(nk_f32_t const *src, nk_bf16_t *dest) {
    nk_u32_t const bits = __float_as_uint(*src);
    nk_store_b16_(dest, (bits & 0x7FFFFFFFu) > 0x7F800000u ? (nk_u16_t)((bits >> 16) | 0x0040u)
                                                           : (nk_u16_t)((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16));
}

/** Narrows one F32 value to F16, rounding to nearest even and keeping NaN payloads like the serial
 *  cast. */
NUMKONG_DEVICE void nk_f32_to_f16_simt_(nk_f32_t const *src, nk_f16_t *dest) {
    nk_u32_t const bits = __float_as_uint(*src), payload = (bits & 0x7FFFFFu) >> 13;
    *(unsigned short *)dest = (bits & 0x7FFFFFFFu) > 0x7F800000u
                                  ? (unsigned short)(((bits >> 16) & 0x8000u) | 0x7C00u | (payload ? payload : 1u))
                                  : __half_as_ushort(__float2half_rn(*src));
}

/** Narrows one F32 value to E4M3FN, rounding to nearest even and saturating at 448. */
NUMKONG_DEVICE void nk_f32_to_e4m3_simt_(nk_f32_t const *src, nk_e4m3_t *dest) {
    nk_u32_t const bits = __float_as_uint(*src), magnitude_bits = bits & 0x7FFFFFFFu;
    unsigned const sign = (bits >> 24) & 0x80u;
    nk_f32_t const magnitude = __uint_as_float(magnitude_bits);
    if (magnitude_bits > 0x7F800000u) { *dest = (nk_e4m3_t)(sign | 0x7Fu); }
    else if (magnitude_bits == 0x7F800000u) { *dest = (nk_e4m3_t)(sign | 0x7Eu); }
    else if (magnitude < 1.0f / 64.0f) {
        // Subnormals step by 1/512 below the smallest normal; rounding up to 8 is the first normal
        nk_f32_t const scaled = magnitude * 512.0f;
        unsigned mantissa = (unsigned)scaled;
        nk_f32_t const fraction = scaled - (nk_f32_t)mantissa;
        if (fraction > 0.5f || (fraction == 0.5f && (mantissa & 1u))) ++mantissa;
        *dest = (nk_e4m3_t)(sign | (mantissa > 7u ? 0x08u : mantissa));
    }
    else {
        int exponent = (int)((magnitude_bits >> 23) & 0xFFu) - 127;
        nk_u32_t const significand = (1u << 23) | (magnitude_bits & 0x7FFFFFu);
        nk_u32_t const remainder = significand & ((1u << 20) - 1u), halfway = 1u << 19;
        nk_u32_t rounded = significand >> 20;
        if (remainder > halfway || (remainder == halfway && (rounded & 1u))) ++rounded;
        if (rounded == 16u) rounded >>= 1, ++exponent;
        unsigned const exponent_field = (unsigned)(exponent + 7), mantissa_field = rounded & 0x07u;
        // The top exponent keeps mantissa 7 for NaN, so it saturates at 6
        if (exponent > 8) { *dest = (nk_e4m3_t)(sign | 0x7Eu); }
        else if (exponent_field == 15u && mantissa_field > 6u) { *dest = (nk_e4m3_t)(sign | 0x7Eu); }
        else { *dest = (nk_e4m3_t)(sign | (exponent_field << 3) | mantissa_field); }
    }
}

/** Narrows one F32 value to E5M2, rounding to nearest even and keeping infinities and NaNs. */
NUMKONG_DEVICE void nk_f32_to_e5m2_simt_(nk_f32_t const *src, nk_e5m2_t *dest) { nk_f32_to_e5m2_(src, dest); }

/** Narrows one F32 value to E2M3FN, rounding to nearest even and saturating at 7.5. */
NUMKONG_DEVICE void nk_f32_to_e2m3_simt_(nk_f32_t const *src, nk_e2m3_t *dest) { nk_f32_to_e2m3_(src, dest); }

/** Narrows one F32 value to E3M2FN, rounding to nearest even and saturating at 28. */
NUMKONG_DEVICE void nk_f32_to_e3m2_simt_(nk_f32_t const *src, nk_e3m2_t *dest) { nk_f32_to_e3m2_(src, dest); }

/** Narrows two F32 values to the E2M1 nibbles of one byte, the first into the high nibble. */
NUMKONG_DEVICE void nk_f32x2_to_e2m1x2_simt_(nk_f32_t const *src, nk_e2m1x2_t *dest) { nk_f32x2_to_e2m1x2_(src, dest); }

/** Widens one F32 value to F64, exactly, quieting a NaN and keeping its payload as CPUs do. */
NUMKONG_DEVICE void nk_f32_to_f64_simt_(nk_f32_t const *src, nk_f64_t *dest) {
    nk_u32_t const bits = __float_as_uint(*src);
    *dest = (bits & 0x7FFFFFFFu) > 0x7F800000u
                ? __longlong_as_double((long long)(((nk_u64_t)(bits & 0x80000000u) << 32) | 0x7FF8000000000000ull |
                                                   ((nk_u64_t)(bits & 0x7FFFFFu) << 29)))
                : (nk_f64_t)*src;
}

/** Narrows one F64 value to F32, rounding to nearest even, quieting a NaN and keeping the top of
 *  its payload as CPUs do. */
NUMKONG_DEVICE void nk_f64_to_f32_simt_(nk_f64_t const *src, nk_f32_t *dest) {
    nk_u64_t const bits = (nk_u64_t)__double_as_longlong(*src);
    *dest = (bits & 0x7FFFFFFFFFFFFFFFull) > 0x7FF0000000000000ull
                ? __uint_as_float((nk_u32_t)((bits >> 32) & 0x80000000u) | 0x7FC00000u |
                                  (nk_u32_t)((bits >> 29) & 0x7FFFFFu))
                : (nk_f32_t)*src;
}

/** Quiets a NaN like a CPU's round trip through F64, leaving every other value as it is. */
NUMKONG_DEVICE nk_f32_t nk_f32_quiet_simt_(nk_f32_t value) {
    nk_u32_t const bits = __float_as_uint(value);
    return (bits & 0x7FFFFFFFu) > 0x7F800000u ? __uint_as_float(bits | 0x00400000u) : value;
}

/** Rounds @p value to the nearest integer from @p smallest to @p largest, ties to even and a NaN
 *  to zero, with one F32 add instead of a conversion, for bounds within ±2²². */
NUMKONG_DEVICE nk_i32_t nk_cast_f32_rint_simt_(nk_f32_t value, nk_f32_t smallest, nk_f32_t largest) {
    if (value != value) return 0;
    nk_f32_t const clamped = value > largest ? largest : value < smallest ? smallest : value;
    return (nk_i32_t)(__float_as_uint(__fadd_rn(clamped, 0x1.8p23f)) - 0x4B400000u);
}

/*  Integer stores in the serial casts' rounding: F32 sources through one add of 1.5 × 2²³, F64
 *  sources through the conversions, which saturate and round to nearest even like the serial
 *  casts, while a test on the bits, off the F64 pipe, zeroes NaN as they do. */
NUMKONG_DEVICE void nk_f32_to_i8_simt_(nk_f32_t const *src, nk_i8_t *dest) {
    *dest = (nk_i8_t)nk_cast_f32_rint_simt_(*src, -128.0f, 127.0f);
}
NUMKONG_DEVICE void nk_f32_to_u8_simt_(nk_f32_t const *src, nk_u8_t *dest) {
    *dest = (nk_u8_t)nk_cast_f32_rint_simt_(*src, 0.0f, 255.0f);
}
NUMKONG_DEVICE void nk_f32_to_i16_simt_(nk_f32_t const *src, nk_i16_t *dest) {
    *dest = (nk_i16_t)nk_cast_f32_rint_simt_(*src, -32768.0f, 32767.0f);
}
NUMKONG_DEVICE void nk_f32_to_u16_simt_(nk_f32_t const *src, nk_u16_t *dest) {
    *dest = (nk_u16_t)nk_cast_f32_rint_simt_(*src, 0.0f, 65535.0f);
}
NUMKONG_DEVICE int nk_f64_is_nan_simt_(nk_f64_t value) {
    return ((nk_u64_t)__double_as_longlong(value) << 1) > 0xFFE0000000000000ull;
}
NUMKONG_DEVICE void nk_f64_to_i32_simt_(nk_f64_t const *src, nk_i32_t *dest) {
    *dest = nk_f64_is_nan_simt_(*src) ? 0 : __double2int_rn(*src);
}
NUMKONG_DEVICE void nk_f64_to_u32_simt_(nk_f64_t const *src, nk_u32_t *dest) {
    *dest = nk_f64_is_nan_simt_(*src) ? 0 : __double2uint_rn(*src);
}
NUMKONG_DEVICE void nk_f64_to_i64_simt_(nk_f64_t const *src, nk_i64_t *dest) {
    *dest = nk_f64_is_nan_simt_(*src) ? 0 : __double2ll_rn(*src);
}
NUMKONG_DEVICE void nk_f64_to_u64_simt_(nk_f64_t const *src, nk_u64_t *dest) {
    *dest = nk_f64_is_nan_simt_(*src) ? 0 : __double2ull_rn(*src);
}

#pragma endregion Conversions

#pragma region Bulk Casts

/** Whether F32 holds every value of @p dtype exactly, so a bulk cast between two such types never
 *  needs the F64 hub. */
NUMKONG_CONSTEXPR int nk_dtype_f32_exact_(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f32_k:
    case nk_f32c_k:
    case nk_f16_k:
    case nk_f16c_k:
    case nk_bf16_k:
    case nk_bf16c_k:
    case nk_e4m3_k:
    case nk_e5m2_k:
    case nk_e2m3_k:
    case nk_e3m2_k:
    case nk_e2m1_k:
    case nk_ue8m0_k:
    case nk_ue4m3_k:
    case nk_i8_k:
    case nk_u8_k:
    case nk_i16_k:
    case nk_u16_k:
    case nk_i4_k:
    case nk_u4_k:
    case nk_u1_k: return 1;
    default: return 0;
    }
}

/** The alignment a device load or store of one @p dtype value needs: its component's size. */
NUMKONG_CONSTEXPR nk_size_t nk_dtype_alignment_(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k:
    case nk_f64c_k:
    case nk_i64_k:
    case nk_u64_k: return 8;
    case nk_f32_k:
    case nk_f32c_k:
    case nk_i32_k:
    case nk_u32_k: return 4;
    case nk_f16_k:
    case nk_f16c_k:
    case nk_bf16_k:
    case nk_bf16c_k:
    case nk_i16_k:
    case nk_u16_k: return 2;
    default: return 1;
    }
}

/** The hub a bulk cast converts through, as @c nk_cast_elementwise_ picks it. */
typedef enum {

    /** The same type on both ends, copied as bytes. */
    nk_cast_hub_bytes_k = 0,

    /** F32 values or complex pairs narrowed straight to BF16, keeping NaN payloads. */
    nk_cast_hub_bf16_k,

    /** Unsigned integers on both ends, through U64 with saturation. */
    nk_cast_hub_u64_k,

    /** Integers on both ends, one of them signed, through I64 with saturation. */
    nk_cast_hub_i64_k,

    /** Values that F32 holds exactly into types it narrows to, through a quieted F32 pair. */
    nk_cast_hub_f32c_k,

    /** Everything else, through an F64 pair. */
    nk_cast_hub_f64c_k,
} nk_cast_hub_t;

/** Everything one bulk cast shares, passed by value as the kernel's only argument. */
typedef struct {
    unsigned char const *from;
    unsigned char *to;
    nk_size_t count;
    nk_size_t units;
    nk_dtype_t from_dtype;
    nk_dtype_t to_dtype;
    nk_cast_hub_t hub;
    unsigned unit_values;
} nk_cast_arguments_t;

/** Code @p index of a packed 4-bit type: the high nibble of its byte for even indices. */
NUMKONG_DEVICE nk_u32_t nk_cast_nibble_simt_(unsigned char const *bytes, nk_size_t index) {
    nk_u32_t const byte = bytes[index / 2];
    return index & 1 ? byte & 0x0Fu : byte >> 4;
}

/** Bit @p index of packed bits, the most significant bit of each byte first. */
NUMKONG_DEVICE nk_u32_t nk_cast_bit_simt_(unsigned char const *bytes, nk_size_t index) {
    return ((nk_u32_t)bytes[index / 8] >> (7 - index % 8)) & 1u;
}

/** Reads value @p index of @p dtype, which F32 holds exactly, as a pair quieted like the serial F64
 *  hub's. */
NUMKONG_DEVICE void nk_cast_load_f32c_simt_(unsigned char const *bytes, nk_dtype_t dtype, nk_size_t index,
                                            nk_f32c_t *value) {
    nk_f32_t real = 0, imag = 0;
    switch (dtype) {
    case nk_f32_k: real = ((nk_f32_t const *)bytes)[index]; break;
    case nk_f32c_k: real = ((nk_f32c_t const *)bytes)[index].real, imag = ((nk_f32c_t const *)bytes)[index].imag; break;
    case nk_f16_k: nk_f16_to_f32_simt_((nk_f16_t const *)bytes + index, &real); break;
    case nk_f16c_k:
        nk_f16_to_f32_simt_(&((nk_f16c_t const *)bytes)[index].real, &real);
        nk_f16_to_f32_simt_(&((nk_f16c_t const *)bytes)[index].imag, &imag);
        break;
    case nk_bf16_k: nk_bf16_to_f32_simt_((nk_bf16_t const *)bytes + index, &real); break;
    case nk_bf16c_k:
        nk_bf16_to_f32_simt_(&((nk_bf16c_t const *)bytes)[index].real, &real);
        nk_bf16_to_f32_simt_(&((nk_bf16c_t const *)bytes)[index].imag, &imag);
        break;
    case nk_e4m3_k: nk_e4m3_to_f32_simt_(bytes + index, &real); break;
    case nk_e5m2_k: nk_e5m2_to_f32_simt_(bytes + index, &real); break;
    case nk_e2m3_k: nk_e2m3_to_f32_simt_(bytes + index, &real); break;
    case nk_e3m2_k: nk_e3m2_to_f32_simt_(bytes + index, &real); break;
    case nk_e2m1_k: real = nk_e2m1_nibble_to_f32_simt_(nk_cast_nibble_simt_(bytes, index)); break;
    case nk_ue8m0_k: nk_ue8m0_to_f32_(bytes + index, &real); break;
    case nk_ue4m3_k: nk_ue4m3_to_f32_(bytes + index, &real); break;
    case nk_i8_k: real = (nk_f32_t)((nk_i8_t const *)bytes)[index]; break;
    case nk_u8_k: real = (nk_f32_t)bytes[index]; break;
    case nk_i16_k: real = (nk_f32_t)((nk_i16_t const *)bytes)[index]; break;
    case nk_u16_k: real = (nk_f32_t)((nk_u16_t const *)bytes)[index]; break;
    case nk_i4_k: real = (nk_f32_t)((nk_i32_t)(nk_cast_nibble_simt_(bytes, index) ^ 8u) - 8); break;
    case nk_u4_k: real = (nk_f32_t)nk_cast_nibble_simt_(bytes, index); break;
    case nk_u1_k: real = (nk_f32_t)nk_cast_bit_simt_(bytes, index); break;
    default: break;
    }
    value->real = nk_f32_quiet_simt_(real), value->imag = nk_f32_quiet_simt_(imag);
}

/** Writes @p value as value @p index of @p dtype, a byte or wider, like the serial F64 hub
 *  narrowing a value that F32 holds exactly. */
NUMKONG_DEVICE void nk_cast_store_f32c_simt_(nk_f32c_t const *value, unsigned char *bytes, nk_dtype_t dtype,
                                             nk_size_t index) {
    switch (dtype) {
    case nk_f32_k: ((nk_f32_t *)bytes)[index] = value->real; break;
    case nk_f32c_k: ((nk_f32c_t *)bytes)[index] = *value; break;
    case nk_f16_k: nk_f32_to_f16_simt_(&value->real, (nk_f16_t *)bytes + index); break;
    case nk_f16c_k:
        nk_f32_to_f16_simt_(&value->real, &((nk_f16c_t *)bytes)[index].real);
        nk_f32_to_f16_simt_(&value->imag, &((nk_f16c_t *)bytes)[index].imag);
        break;
    case nk_bf16_k: nk_f32_to_bf16_simt_(&value->real, (nk_bf16_t *)bytes + index); break;
    case nk_bf16c_k:
        nk_f32_to_bf16_simt_(&value->real, &((nk_bf16c_t *)bytes)[index].real);
        nk_f32_to_bf16_simt_(&value->imag, &((nk_bf16c_t *)bytes)[index].imag);
        break;
    case nk_e4m3_k: nk_f32_to_e4m3_simt_(&value->real, bytes + index); break;
    case nk_e5m2_k: nk_f32_to_e5m2_(&value->real, bytes + index); break;
    case nk_e2m3_k: nk_f32_to_e2m3_(&value->real, bytes + index); break;
    case nk_e3m2_k: nk_f32_to_e3m2_(&value->real, bytes + index); break;
    case nk_ue8m0_k: nk_f32_to_ue8m0_(&value->real, bytes + index); break;
    case nk_ue4m3_k: nk_f32_to_ue4m3_(&value->real, bytes + index); break;
    case nk_i8_k: ((nk_i8_t *)bytes)[index] = (nk_i8_t)nk_cast_f32_rint_simt_(value->real, -128.0f, 127.0f); break;
    case nk_u8_k: bytes[index] = (nk_u8_t)nk_cast_f32_rint_simt_(value->real, 0.0f, 255.0f); break;
    case nk_i16_k:
        ((nk_i16_t *)bytes)[index] = (nk_i16_t)nk_cast_f32_rint_simt_(value->real, -32768.0f, 32767.0f);
        break;
    case nk_u16_k: ((nk_u16_t *)bytes)[index] = (nk_u16_t)nk_cast_f32_rint_simt_(value->real, 0.0f, 65535.0f); break;
    default: break;
    }
}

/** The code of @p value in a packed @p dtype, as the serial F64 hub packs a value F32 holds
 *  exactly: integers truncated after clamping, bits set by any nonzero value. */
NUMKONG_DEVICE nk_u32_t nk_cast_code_f32c_simt_(nk_f32c_t const *value, nk_dtype_t dtype) {
    nk_f32_t const real = value->real;
    nk_u8_t code = 0;
    switch (dtype) {
    case nk_i4_k:
        return real != real ? 0u : (nk_u32_t)(nk_i32_t)(real > 7.0f ? 7.0f : real < -8.0f ? -8.0f : real) & 0x0Fu;
    case nk_u4_k: return real != real ? 0u : (nk_u32_t)(real > 15.0f ? 15.0f : real < 0.0f ? 0.0f : real);
    case nk_u1_k: return real != 0.0f;
    case nk_e2m1_k: nk_f32_to_e2m1_nibble_serial_(real, &code); return code;
    default: return 0;
    }
}

/** Reads value @p index of @p dtype as an F64 pair, like @c nk_scalar_buffer_to_f64c_. */
NUMKONG_DEVICE void nk_cast_load_f64c_simt_(unsigned char const *bytes, nk_dtype_t dtype, nk_size_t index,
                                            nk_f64c_t *value) {
    nk_f32c_t narrow;
    value->real = 0, value->imag = 0;
    switch (dtype) {
    case nk_f64_k: value->real = ((nk_f64_t const *)bytes)[index]; break;
    case nk_f64c_k: *value = ((nk_f64c_t const *)bytes)[index]; break;
    case nk_i32_k: value->real = (nk_f64_t)((nk_i32_t const *)bytes)[index]; break;
    case nk_u32_k: value->real = (nk_f64_t)((nk_u32_t const *)bytes)[index]; break;
    case nk_i64_k: value->real = (nk_f64_t)((nk_i64_t const *)bytes)[index]; break;
    case nk_u64_k: value->real = (nk_f64_t)((nk_u64_t const *)bytes)[index]; break;
    default:
        nk_cast_load_f32c_simt_(bytes, dtype, index, &narrow);
        nk_f32_to_f64_simt_(&narrow.real, &value->real), nk_f32_to_f64_simt_(&narrow.imag, &value->imag);
        break;
    }
}

/** Writes @p value as value @p index of @p dtype, a byte or wider, like
 *  @c nk_scalar_buffer_from_f64c_: integers round from F64, floats narrow through F32. */
NUMKONG_DEVICE void nk_cast_store_f64c_simt_(nk_f64c_t const *value, unsigned char *bytes, nk_dtype_t dtype,
                                             nk_size_t index) {
    nk_f32c_t narrow;
    switch (dtype) {
    case nk_f64_k: ((nk_f64_t *)bytes)[index] = value->real; break;
    case nk_f64c_k: ((nk_f64c_t *)bytes)[index] = *value; break;
    case nk_i64_k: nk_f64_to_i64_serial_(&value->real, (nk_i64_t *)bytes + index); break;
    case nk_u64_k: nk_f64_to_u64_serial_(&value->real, (nk_u64_t *)bytes + index); break;
    case nk_i32_k: nk_f64_to_i32_serial_(&value->real, (nk_i32_t *)bytes + index); break;
    case nk_u32_k: nk_f64_to_u32_serial_(&value->real, (nk_u32_t *)bytes + index); break;
    case nk_i16_k: nk_f64_to_i16_serial_(&value->real, (nk_i16_t *)bytes + index); break;
    case nk_u16_k: nk_f64_to_u16_serial_(&value->real, (nk_u16_t *)bytes + index); break;
    case nk_i8_k: nk_f64_to_i8_serial_(&value->real, (nk_i8_t *)bytes + index); break;
    case nk_u8_k: nk_f64_to_u8_serial_(&value->real, bytes + index); break;
    default:
        nk_f64_to_f32_simt_(&value->real, &narrow.real), narrow.imag = 0;
        if (nk_dtype_family(dtype) == nk_dtype_family_complex_float_k) nk_f64_to_f32_simt_(&value->imag, &narrow.imag);
        nk_cast_store_f32c_simt_(&narrow, bytes, dtype, index);
        break;
    }
}

/** The code of @p value in a packed @p dtype, like the serial F64 hub packs it. */
NUMKONG_DEVICE nk_u32_t nk_cast_code_f64c_simt_(nk_f64c_t const *value, nk_dtype_t dtype) {
    nk_f64_t const real = value->real;
    nk_f32c_t narrow;
    switch (dtype) {
    case nk_i4_k: return real != real ? 0u : (nk_u32_t)(nk_i32_t)(real > 7.0 ? 7.0 : real < -8.0 ? -8.0 : real) & 0x0Fu;
    case nk_u4_k: return real != real ? 0u : (nk_u32_t)(real > 15.0 ? 15.0 : real < 0.0 ? 0.0 : real);
    case nk_u1_k: return real != 0.0;
    default:
        nk_f64_to_f32_simt_(&value->real, &narrow.real), narrow.imag = 0;
        return nk_cast_code_f32c_simt_(&narrow, dtype);
    }
}

/** Reads value @p index of an integer @p dtype into I64, like @c nk_scalar_buffers_to_i64_. */
NUMKONG_DEVICE nk_i64_t nk_cast_load_i64_simt_(unsigned char const *bytes, nk_dtype_t dtype, nk_size_t index) {
    switch (dtype) {
    case nk_i64_k: return ((nk_i64_t const *)bytes)[index];
    case nk_i32_k: return ((nk_i32_t const *)bytes)[index];
    case nk_i16_k: return ((nk_i16_t const *)bytes)[index];
    case nk_i8_k: return ((nk_i8_t const *)bytes)[index];
    case nk_u64_k: {
        nk_i64_t value;
        nk_u64_to_i64_serial_((nk_u64_t const *)bytes + index, &value);
        return value;
    }
    case nk_u32_k: return ((nk_u32_t const *)bytes)[index];
    case nk_u16_k: return ((nk_u16_t const *)bytes)[index];
    case nk_u8_k: return bytes[index];
    case nk_i4_k: return (nk_i32_t)(nk_cast_nibble_simt_(bytes, index) ^ 8u) - 8;
    case nk_u4_k: return nk_cast_nibble_simt_(bytes, index);
    case nk_u1_k: return nk_cast_bit_simt_(bytes, index);
    default: return 0;
    }
}

/** Writes @p value as value @p index of an integer @p dtype, a byte or wider, saturating like
 *  @c nk_scalar_buffers_from_i64_. */
NUMKONG_DEVICE void nk_cast_store_i64_simt_(nk_i64_t const *value, unsigned char *bytes, nk_dtype_t dtype,
                                            nk_size_t index) {
    switch (dtype) {
    case nk_i64_k: ((nk_i64_t *)bytes)[index] = *value; break;
    case nk_i32_k: nk_i64_to_i32_serial_(value, (nk_i32_t *)bytes + index); break;
    case nk_i16_k: nk_i64_to_i16_serial_(value, (nk_i16_t *)bytes + index); break;
    case nk_i8_k: nk_i64_to_i8_serial_(value, (nk_i8_t *)bytes + index); break;
    case nk_u64_k: nk_i64_to_u64_serial_(value, (nk_u64_t *)bytes + index); break;
    case nk_u32_k: nk_i64_to_u32_serial_(value, (nk_u32_t *)bytes + index); break;
    case nk_u16_k: nk_i64_to_u16_serial_(value, (nk_u16_t *)bytes + index); break;
    case nk_u8_k: nk_i64_to_u8_serial_(value, bytes + index); break;
    default: break;
    }
}

/** Reads value @p index of an unsigned @p dtype into U64, like @c nk_scalar_buffers_to_u64_. */
NUMKONG_DEVICE nk_u64_t nk_cast_load_u64_simt_(unsigned char const *bytes, nk_dtype_t dtype, nk_size_t index) {
    switch (dtype) {
    case nk_u64_k: return ((nk_u64_t const *)bytes)[index];
    case nk_u32_k: return ((nk_u32_t const *)bytes)[index];
    case nk_u16_k: return ((nk_u16_t const *)bytes)[index];
    case nk_u8_k: return bytes[index];
    case nk_u4_k: return nk_cast_nibble_simt_(bytes, index);
    case nk_u1_k: return nk_cast_bit_simt_(bytes, index);
    default: return 0;
    }
}

/** Writes @p value as value @p index of an unsigned @p dtype, a byte or wider, saturating like
 *  @c nk_scalar_buffers_from_u64_. */
NUMKONG_DEVICE void nk_cast_store_u64_simt_(nk_u64_t const *value, unsigned char *bytes, nk_dtype_t dtype,
                                            nk_size_t index) {
    switch (dtype) {
    case nk_u64_k: ((nk_u64_t *)bytes)[index] = *value; break;
    case nk_u32_k: nk_u64_to_u32_serial_(value, (nk_u32_t *)bytes + index); break;
    case nk_u16_k: nk_u64_to_u16_serial_(value, (nk_u16_t *)bytes + index); break;
    case nk_u8_k: nk_u64_to_u8_serial_(value, bytes + index); break;
    default: break;
    }
}

/** Reads value @p index of @p dtype into the member of @p value that @p hub uses. */
NUMKONG_DEVICE void nk_cast_load_simt_(nk_cast_hub_t hub, unsigned char const *bytes, nk_dtype_t dtype, nk_size_t index,
                                       nk_scalar_buffer_t *value) {
    switch (hub) {
    case nk_cast_hub_u64_k: value->u64 = nk_cast_load_u64_simt_(bytes, dtype, index); break;
    case nk_cast_hub_i64_k: value->i64 = nk_cast_load_i64_simt_(bytes, dtype, index); break;
    case nk_cast_hub_f32c_k: nk_cast_load_f32c_simt_(bytes, dtype, index, &value->f32c); break;
    default: nk_cast_load_f64c_simt_(bytes, dtype, index, &value->f64c); break;
    }
}

/** Writes the member of @p value that @p hub uses as value @p index of a @p dtype a byte wide or
 *  wider. */
NUMKONG_DEVICE void nk_cast_store_simt_(nk_cast_hub_t hub, nk_scalar_buffer_t const *value, unsigned char *bytes,
                                        nk_dtype_t dtype, nk_size_t index) {
    switch (hub) {
    case nk_cast_hub_u64_k: nk_cast_store_u64_simt_(&value->u64, bytes, dtype, index); break;
    case nk_cast_hub_i64_k: nk_cast_store_i64_simt_(&value->i64, bytes, dtype, index); break;
    case nk_cast_hub_f32c_k: nk_cast_store_f32c_simt_(&value->f32c, bytes, dtype, index); break;
    default: nk_cast_store_f64c_simt_(&value->f64c, bytes, dtype, index); break;
    }
}

/** The code of the member of @p value that @p hub uses in a packed @p dtype: I4, U4, E2M1 or U1. */
NUMKONG_DEVICE nk_u32_t nk_cast_code_simt_(nk_cast_hub_t hub, nk_scalar_buffer_t const *value, nk_dtype_t dtype) {
    switch (hub) {
    case nk_cast_hub_u64_k: return dtype == nk_u1_k ? value->u64 != 0 : (nk_u32_t)(value->u64 > 15 ? 15 : value->u64);
    case nk_cast_hub_i64_k: return (nk_u32_t)(value->i64 > 7 ? 7 : value->i64 < -8 ? -8 : value->i64) & 0x0Fu;
    case nk_cast_hub_f32c_k: return nk_cast_code_f32c_simt_(&value->f32c, dtype);
    default: return nk_cast_code_f64c_simt_(&value->f64c, dtype);
    }
}

/** Converts unit @p unit of a bulk cast: one value, the 2 or 8 values sharing a packed byte, or one
 *  word of a copy. A packed destination gets whole bytes only, like the serial batches. */
NUMKONG_DEVICE void nk_cast_unit_simt_(nk_cast_arguments_t const *arguments, nk_size_t unit) {
    nk_size_t const first = unit * arguments->unit_values, left = arguments->count - first;
    unsigned const values = left < arguments->unit_values ? (unsigned)left : arguments->unit_values;
    nk_cast_hub_t const hub = arguments->hub;
    nk_scalar_buffer_t value, partner;
    nk_u32_t packed = 0;
    if (hub == nk_cast_hub_bytes_k) {
        switch (arguments->unit_values) {
        case 16: ((uint4 *)arguments->to)[unit] = ((uint4 const *)arguments->from)[unit]; break;
        case 8: ((nk_u64_t *)arguments->to)[unit] = ((nk_u64_t const *)arguments->from)[unit]; break;
        case 4: ((nk_u32_t *)arguments->to)[unit] = ((nk_u32_t const *)arguments->from)[unit]; break;
        case 2: ((nk_u16_t *)arguments->to)[unit] = ((nk_u16_t const *)arguments->from)[unit]; break;
        default: arguments->to[unit] = arguments->from[unit]; break;
        }
        return;
    }
    if (hub == nk_cast_hub_bf16_k) {
        nk_f32_to_bf16_simt_((nk_f32_t const *)arguments->from + unit, (nk_bf16_t *)arguments->to + unit);
        return;
    }
    switch (arguments->to_dtype) {
    case nk_u1_k:
        for (unsigned offset = 0; offset != values; ++offset) {
            nk_cast_load_simt_(hub, arguments->from, arguments->from_dtype, first + offset, &value);
            packed |= nk_cast_code_simt_(hub, &value, nk_u1_k) << (7 - offset);
        }
        arguments->to[first / 8] = (unsigned char)packed;
        break;
    case nk_i4_k:
    case nk_u4_k:
    case nk_e2m1_k:
        for (unsigned offset = 0; offset + 2 <= values; offset += 2) {
            nk_cast_load_simt_(hub, arguments->from, arguments->from_dtype, first + offset, &value);
            nk_cast_load_simt_(hub, arguments->from, arguments->from_dtype, first + offset + 1, &partner);
            arguments->to[(first + offset) / 2] =
                (unsigned char)((nk_cast_code_simt_(hub, &value, arguments->to_dtype) << 4) |
                                nk_cast_code_simt_(hub, &partner, arguments->to_dtype));
        }
        break;
    default:
        for (unsigned offset = 0; offset != values; ++offset) {
            nk_cast_load_simt_(hub, arguments->from, arguments->from_dtype, first + offset, &value);
            nk_cast_store_simt_(hub, &value, arguments->to, arguments->to_dtype, first + offset);
        }
        break;
    }
}

/** Plans the bulk cast of @p count values from @p from to @p to through the hub
 *  @c nk_cast_elementwise_ would take, copying same-type bytes in the widest words their
 *  alignment allows. Returns zero when the serial cast writes nothing. */
NUMKONG_INLINE int nk_cast_plan_simt_(void const *from, nk_dtype_t from_dtype, nk_size_t count, void *to,
                                      nk_dtype_t to_dtype, nk_cast_arguments_t *arguments) {
    nk_size_t const from_bits = nk_dtype_bits(from_dtype), to_bits = nk_dtype_bits(to_dtype);
    nk_dtype_family_t const from_family = nk_dtype_family(from_dtype), to_family = nk_dtype_family(to_dtype);
    int const from_integer = from_family == nk_dtype_family_int_k || from_family == nk_dtype_family_uint_k;
    int const to_integer = to_family == nk_dtype_family_int_k || to_family == nk_dtype_family_uint_k;
    nk_size_t const from_per_byte = from_bits && from_bits < NUMKONG_BITS_PER_BYTE ? NUMKONG_BITS_PER_BYTE / from_bits
                                                                                   : 1;
    nk_size_t const to_per_byte = to_bits && to_bits < NUMKONG_BITS_PER_BYTE ? NUMKONG_BITS_PER_BYTE / to_bits : 1;
    arguments->from = (unsigned char const *)from, arguments->to = (unsigned char *)to, arguments->count = count;
    arguments->from_dtype = from_dtype, arguments->to_dtype = to_dtype;
    arguments->unit_values = (unsigned)(from_per_byte > to_per_byte ? from_per_byte : to_per_byte);
    if (from_dtype == to_dtype) {
        arguments->count = count * from_bits / NUMKONG_BITS_PER_BYTE;
        arguments->hub = nk_cast_hub_bytes_k, arguments->unit_values = 16;
        while (((nk_size_t)from | (nk_size_t)to | arguments->count) & (arguments->unit_values - 1))
            arguments->unit_values /= 2;
    }
    else if (from_dtype == nk_f32_k && to_dtype == nk_bf16_k) { arguments->hub = nk_cast_hub_bf16_k; }
    else if (from_dtype == nk_f32c_k && to_dtype == nk_bf16c_k) {
        arguments->hub = nk_cast_hub_bf16_k, arguments->count *= 2;
    }
    else if (!from_bits || !to_bits) { return 0; }
    else if (from_family == nk_dtype_family_uint_k && to_family == nk_dtype_family_uint_k) {
        arguments->hub = nk_cast_hub_u64_k;
    }
    else if (from_integer && to_integer) {
        // The serial I64 hub stores no U4 and no U1
        if (to_dtype == nk_u4_k || to_dtype == nk_u1_k) return 0;
        arguments->hub = nk_cast_hub_i64_k;
    }
    else if (nk_dtype_f32_exact_(from_dtype) && nk_dtype_f32_exact_(to_dtype)) { arguments->hub = nk_cast_hub_f32c_k; }
    else { arguments->hub = nk_cast_hub_f64c_k; }
    arguments->units = nk_size_divide_round_up_(arguments->count, arguments->unit_values);
    return arguments->units != 0;
}

/** Whether a planned bulk cast takes a vector path instead of the per-unit type switch: F32 to BF16
 *  or E4M3, or BF16 or E4M3 to F32, with both ends on 16-byte boundaries. If so, its units turn
 *  into chunks of the values that fill 16 bytes of the narrower end. */
NUMKONG_INLINE int nk_cast_plan_vectors_simt_(nk_cast_arguments_t *arguments) {
    nk_dtype_t const from_dtype = arguments->from_dtype, to_dtype = arguments->to_dtype;
    int const vectorized = (from_dtype == nk_f32_k && (to_dtype == nk_bf16_k || to_dtype == nk_e4m3_k)) ||
                           ((from_dtype == nk_bf16_k || from_dtype == nk_e4m3_k) && to_dtype == nk_f32_k);
    if (!vectorized || (((nk_size_t)arguments->from | (nk_size_t)arguments->to) & 15)) return 0;
    nk_size_t const from_bits = nk_dtype_bits(from_dtype), to_bits = nk_dtype_bits(to_dtype);
    nk_size_t const chunk_values = 16 * NUMKONG_BITS_PER_BYTE / (from_bits < to_bits ? from_bits : to_bits);
    arguments->unit_values = (unsigned)chunk_values;
    arguments->units = nk_size_divide_round_up_(arguments->count, chunk_values);
    return 1;
}

/** Narrows the 8 F32 values at @p from into the 16 bytes of BF16 at @p to. */
NUMKONG_DEVICE void nk_cast_f32x8_to_bf16x8_simt_(unsigned char const *from, unsigned char *to) {
    nk_b128_vec_t loaded[2], narrowed;
    loaded[0] = nk_load_b128_vec_(from), loaded[1] = nk_load_b128_vec_(from + 16);
#pragma unroll
    for (unsigned offset = 0; offset != 8; ++offset)
        nk_f32_to_bf16_simt_(loaded[offset / 4].f32s + offset % 4, narrowed.bf16s + offset);
    nk_store_b128_vec_(to, &narrowed);
}

/** Narrows the 16 F32 values at @p from into the 16 bytes of E4M3FN at @p to. */
NUMKONG_DEVICE void nk_cast_f32x16_to_e4m3x16_simt_(unsigned char const *from, unsigned char *to) {
    nk_b128_vec_t loaded[4], narrowed;
#pragma unroll
    for (unsigned quad = 0; quad != 4; ++quad) loaded[quad] = nk_load_b128_vec_(from + quad * 16);
#pragma unroll
    for (unsigned offset = 0; offset != 16; ++offset)
        nk_f32_to_e4m3_simt_(loaded[offset / 4].f32s + offset % 4, narrowed.e4m3s + offset);
    nk_store_b128_vec_(to, &narrowed);
}

/** Widens the 16 bytes of E4M3FN at @p from into 16 F32 values at @p to. */
NUMKONG_DEVICE void nk_cast_e4m3x16_to_f32x16_simt_(unsigned char const *from, unsigned char *to) {
    nk_b128_vec_t const loaded = nk_load_b128_vec_(from);
    nk_b128_vec_t widened[4];
#pragma unroll
    for (unsigned offset = 0; offset != 16; ++offset)
        nk_e4m3_to_f32_simt_(loaded.e4m3s + offset, widened[offset / 4].f32s + offset % 4);
#pragma unroll
    for (unsigned quad = 0; quad != 4; ++quad) nk_store_b128_vec_(to + quad * 16, widened + quad);
}

/** Widens the 16 bytes of BF16 at @p from into 8 F32 values at @p to, quieting NaNs like the F32
 *  hub. */
NUMKONG_DEVICE void nk_cast_bf16x8_to_f32x8_simt_(unsigned char const *from, unsigned char *to) {
    nk_b128_vec_t const loaded = nk_load_b128_vec_(from);
    nk_b128_vec_t widened[2];
    nk_f32_t value;
#pragma unroll
    for (unsigned offset = 0; offset != 8; ++offset) {
        nk_bf16_to_f32_simt_(loaded.bf16s + offset, &value);
        widened[offset / 4].f32s[offset % 4] = nk_f32_quiet_simt_(value);
    }
    nk_store_b128_vec_(to, widened), nk_store_b128_vec_(to + 16, widened + 1);
}

/** Converts the values of a vector cast past its last whole chunk one by one, through the per-unit
 *  path. Every thread calls it after converting its own chunks a grid apart. */
NUMKONG_DEVICE void nk_cast_vectors_tail_simt_(nk_cast_arguments_t const *arguments) {
    nk_size_t const threads = (nk_size_t)gridDim.x * blockDim.x;
    nk_size_t const first = (nk_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    nk_cast_arguments_t per_unit = *arguments;
    // Every hub these casts plan takes units of one value
    per_unit.unit_values = 1;
    for (nk_size_t index = arguments->count / arguments->unit_values * arguments->unit_values + first;
         index < arguments->count; index += threads)
        nk_cast_unit_simt_(&per_unit, index);
}

/** Runs a vector cast that @c nk_cast_plan_vectors_simt_ accepted with the portable converters:
 *  each thread converts chunks a grid apart, then the tail. */
NUMKONG_DEVICE void nk_cast_vectors_simt_(nk_cast_arguments_t const *arguments) {
    nk_size_t const threads = (nk_size_t)gridDim.x * blockDim.x;
    nk_size_t const first = (nk_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    nk_size_t const chunks = arguments->count / arguments->unit_values;
    unsigned char const *from = arguments->from;
    unsigned char *to = arguments->to;
    if (arguments->from_dtype == nk_bf16_k)
        for (nk_size_t chunk = first; chunk < chunks; chunk += threads)
            nk_cast_bf16x8_to_f32x8_simt_(from + chunk * 16, to + chunk * 32);
    else if (arguments->from_dtype == nk_e4m3_k)
        for (nk_size_t chunk = first; chunk < chunks; chunk += threads)
            nk_cast_e4m3x16_to_f32x16_simt_(from + chunk * 16, to + chunk * 64);
    else if (arguments->to_dtype == nk_bf16_k)
        for (nk_size_t chunk = first; chunk < chunks; chunk += threads)
            nk_cast_f32x8_to_bf16x8_simt_(from + chunk * 32, to + chunk * 16);
    else
        for (nk_size_t chunk = first; chunk < chunks; chunk += threads)
            nk_cast_f32x16_to_e4m3x16_simt_(from + chunk * 64, to + chunk * 16);
    nk_cast_vectors_tail_simt_(arguments);
}

#pragma endregion Bulk Casts

#pragma region Block Scaled Casts

/** Everything one block-scaled cast shares, passed by value as its kernel's only argument. A plain
 *  side has an unknown scale type and blocks of one value; a tensor scale not applied is NULL. */
typedef struct {
    unsigned char const *from;
    unsigned char const *from_scales;
    nk_f32_t const *from_tensor_scale;
    unsigned char *to;
    unsigned char *to_scales;
    nk_f32_t *to_tensor_scale;
    nk_size_t count;
    nk_size_t chunks;
    nk_dtype_t from_dtype;
    nk_dtype_t from_scale_dtype;
    nk_dtype_t to_dtype;
    nk_dtype_t to_scale_dtype;
    unsigned from_block;
    unsigned to_block;
    unsigned phase;
} nk_cast_block_scaled_arguments_t;

/** The launches of one block-scaled cast, in stream order. */
typedef enum {

    /** A grid gathers the source's abs-max into a tensor scale left to derive. */
    nk_cast_block_scaled_amax_k = 0,

    /** A grid converts the chunks, deriving the tensor scale from that abs-max where it must. */
    nk_cast_block_scaled_chunks_k,
} nk_cast_block_scaled_phase_t;

/*  A tensor scale left to derive is zero, and the abs-max gathers into its bits with the sign set,
 *  so either state still reads as "derive" until the chunks write the positive scale. */

/** The destination's tensor scale: the caller's, or derived from the abs-max its bits gathered. */
NUMKONG_DEVICE nk_f32_t nk_cast_block_scaled_to_tensor_scale_simt_(nk_cast_block_scaled_arguments_t const *arguments) {
    if (!arguments->to_tensor_scale) return 1.0f;
    nk_u32_t const bits = *(nk_u32_t volatile *)arguments->to_tensor_scale;
    if (bits && !(bits >> 31)) return __uint_as_float(bits);
    // ptxas 13.2 folded `bits & 0x7FFFFFFF` into an |abs| operand of a register it reused for 1.0
    nk_f32_t const amax = fabsf(__uint_as_float(bits));
    nk_f32_t const scale_max = arguments->to_scale_dtype == nk_ue4m3_k ? 448.0f : 1.0f;
    return amax > 0 ? amax / (nk_element_max_representable_(arguments->to_dtype) * scale_max) : 1.0f;
}

/** Reads values @p first onward of @p dtype as @p count F32 values, like @c nk_cast_elementwise_
 *  into F32; @p first is a multiple of the values per byte. */
NUMKONG_DEVICE void nk_cast_decode_f32s_simt_(unsigned char const *bytes, nk_dtype_t dtype, nk_size_t first,
                                              nk_size_t count, nk_f32_t *values) {
    nk_size_t const bits = nk_dtype_bits(dtype);
    nk_cast_arguments_t arguments;
    if (dtype == nk_f32_k) {
        for (nk_size_t offset = 0; offset != count; ++offset)
            values[offset] = ((nk_f32_t const *)bytes)[first + offset];
        return;
    }
    if (!bits) return;
    arguments.from = bytes + first * bits / NUMKONG_BITS_PER_BYTE, arguments.to = (unsigned char *)values;
    arguments.count = count, arguments.from_dtype = dtype, arguments.to_dtype = nk_f32_k;
    arguments.unit_values = bits < NUMKONG_BITS_PER_BYTE ? (unsigned)(NUMKONG_BITS_PER_BYTE / bits) : 1;
    arguments.hub = nk_dtype_f32_exact_(dtype) ? nk_cast_hub_f32c_k : nk_cast_hub_f64c_k;
    for (nk_size_t unit = 0; unit * arguments.unit_values < count; ++unit) nk_cast_unit_simt_(&arguments, unit);
}

/** Writes @p count F32 values as values @p first onward of @p dtype, like @c nk_cast_elementwise_
 *  from F32; @p first is a multiple of the values per byte. */
NUMKONG_DEVICE void nk_cast_encode_f32s_simt_(nk_f32_t const *values, nk_size_t count, unsigned char *bytes,
                                              nk_dtype_t dtype, nk_size_t first) {
    nk_size_t const bits = nk_dtype_bits(dtype);
    nk_cast_arguments_t arguments;
    if (dtype == nk_f32_k) {
        for (nk_size_t offset = 0; offset != count; ++offset) ((nk_f32_t *)bytes)[first + offset] = values[offset];
        return;
    }
    if (!bits) return;
    arguments.from = (unsigned char const *)values, arguments.to = bytes + first * bits / NUMKONG_BITS_PER_BYTE;
    arguments.count = count, arguments.from_dtype = nk_f32_k, arguments.to_dtype = dtype;
    arguments.unit_values = bits < NUMKONG_BITS_PER_BYTE ? (unsigned)(NUMKONG_BITS_PER_BYTE / bits) : 1;
    arguments.hub = dtype == nk_bf16_k           ? nk_cast_hub_bf16_k
                    : nk_dtype_f32_exact_(dtype) ? nk_cast_hub_f32c_k
                                                 : nk_cast_hub_f64c_k;
    for (nk_size_t unit = 0; unit * arguments.unit_values < count; ++unit) nk_cast_unit_simt_(&arguments, unit);
}

/** The scale of source block @p block as F32: its decoded scale byte times the tensor scale, or one
 *  for a plain source. */
NUMKONG_DEVICE nk_f32_t nk_cast_block_scaled_from_scale_simt_(nk_cast_block_scaled_arguments_t const *arguments,
                                                              nk_size_t block) {
    if (arguments->from_scale_dtype == nk_dtype_unknown_k) return 1.0f;
    return nk_block_scaled_decode_scale_serial_(arguments->from_scales[block], arguments->from_scale_dtype) *
           (arguments->from_tensor_scale ? *arguments->from_tensor_scale : 1.0f);
}

/** Converts the chunk at @p chunk_start like one step of @c nk_cast_block_scaled_through_f32_. */
NUMKONG_DEVICE void nk_cast_block_scaled_chunk_simt_(nk_cast_block_scaled_arguments_t const *arguments,
                                                     nk_size_t chunk_start) {
    nk_f32_t scratch[32], encoded[32];
    unsigned const chunk = arguments->from_block > arguments->to_block ? arguments->from_block : arguments->to_block;
    unsigned const chunk_count = arguments->count - chunk_start < chunk ? (unsigned)(arguments->count - chunk_start)
                                                                        : chunk;
    int const from_plain = arguments->from_scale_dtype == nk_dtype_unknown_k;
    int const to_plain = arguments->to_scale_dtype == nk_dtype_unknown_k;
    unsigned const from_step = from_plain ? chunk_count : arguments->from_block;
    unsigned const to_step = to_plain ? chunk_count : arguments->to_block;
    nk_f32_t const to_tensor_scale = nk_cast_block_scaled_to_tensor_scale_simt_(arguments);
    nk_f32_t const element_max = nk_element_max_representable_(arguments->to_dtype);
    for (unsigned begin = 0; begin < chunk_count; begin += from_step) {
        unsigned const valid = chunk_count - begin < from_step ? chunk_count - begin : from_step;
        nk_cast_decode_f32s_simt_(arguments->from, arguments->from_dtype, chunk_start + begin, valid, scratch + begin);
        if (from_plain) continue;
        nk_f32_t const scale = nk_cast_block_scaled_from_scale_simt_(arguments,
                                                                     (chunk_start + begin) / arguments->from_block);
        for (unsigned offset = 0; offset != valid; ++offset) scratch[begin + offset] *= scale;
    }
    for (unsigned begin = 0; begin < chunk_count; begin += to_step) {
        unsigned const valid = chunk_count - begin < to_step ? chunk_count - begin : to_step;
        if (to_plain) {
            nk_cast_encode_f32s_simt_(scratch + begin, valid, arguments->to, arguments->to_dtype, chunk_start + begin);
            continue;
        }
        nk_f32_t const block_amax = nk_block_amax_f32_serial_(scratch + begin, valid);
        nk_u8_t const raw = nk_block_scaled_encode_scale_serial_(block_amax, element_max, to_tensor_scale,
                                                                 arguments->to_scale_dtype);
        arguments->to_scales[(chunk_start + begin) / arguments->to_block] = raw;
        nk_f32_t const effective_scale = nk_block_scaled_decode_scale_serial_(raw, arguments->to_scale_dtype) *
                                         to_tensor_scale;
        nk_f32_t const reciprocal = effective_scale > 0 ? 1.0f / effective_scale : 0.0f;
        for (unsigned offset = 0; offset != valid; ++offset) {
            nk_f32_t const scaled = scratch[begin + offset] * reciprocal;
            encoded[offset] = scaled > element_max ? element_max : scaled < -element_max ? -element_max : scaled;
        }
        nk_cast_encode_f32s_simt_(encoded, valid, arguments->to, arguments->to_dtype, chunk_start + begin);
    }
}

/** Gathers the abs-max of the scaled source, skipping NaNs, into the tensor scale's bits, which
 *  order like the magnitudes they encode. Every thread of the block must call it. */
NUMKONG_DEVICE void nk_cast_block_scaled_amax_simt_(nk_cast_block_scaled_arguments_t const *arguments,
                                                    nk_u32_t *partials) {
    nk_size_t const bits = nk_dtype_bits(arguments->from_dtype);
    nk_size_t const per_byte = bits < NUMKONG_BITS_PER_BYTE ? NUMKONG_BITS_PER_BYTE / bits : 1;
    nk_size_t const units = nk_size_divide_round_up_(arguments->count, per_byte);
    nk_size_t const stride = (nk_size_t)gridDim.x * blockDim.x;
    nk_f32_t values[8], amax = 0;
    for (nk_size_t unit = (nk_size_t)blockIdx.x * blockDim.x + threadIdx.x; unit < units; unit += stride) {
        nk_size_t const first = unit * per_byte, left = arguments->count - first;
        nk_size_t const valid = left < per_byte ? left : per_byte;
        nk_cast_decode_f32s_simt_(arguments->from, arguments->from_dtype, first, valid, values);
        for (nk_size_t offset = 0; offset != valid; ++offset) {
            nk_f32_t const magnitude = fabsf(values[offset] * nk_cast_block_scaled_from_scale_simt_(
                                                                  arguments, (first + offset) / arguments->from_block));
            if (magnitude > amax) amax = magnitude;
        }
    }
    partials[threadIdx.x] = __float_as_uint(amax);
    __syncthreads();
    for (unsigned half = blockDim.x / 2; half; half >>= 1) {
        if (threadIdx.x < half && partials[threadIdx.x + half] > partials[threadIdx.x])
            partials[threadIdx.x] = partials[threadIdx.x + half];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicMax((unsigned int *)arguments->to_tensor_scale, partials[0] | 0x80000000u);
}

/** Runs one phase of a block-scaled cast, the same kernel launched once per phase. */
NUMKONG_DEVICE void nk_cast_block_scaled_phase_simt_(nk_cast_block_scaled_arguments_t const *arguments,
                                                     nk_u32_t *partials) {
    nk_size_t const stride = (nk_size_t)gridDim.x * blockDim.x;
    nk_f32_t *tensor_scale = arguments->to_tensor_scale;
    switch (arguments->phase) {
    case nk_cast_block_scaled_amax_k: {
        nk_u32_t const bits = *(nk_u32_t volatile *)tensor_scale;
        if (!bits || bits >> 31) nk_cast_block_scaled_amax_simt_(arguments, partials);
        break;
    }
    default:
        for (nk_size_t chunk = (nk_size_t)blockIdx.x * blockDim.x + threadIdx.x; chunk < arguments->chunks;
             chunk += stride)
            nk_cast_block_scaled_chunk_simt_(
                arguments,
                chunk * (arguments->from_block > arguments->to_block ? arguments->from_block : arguments->to_block));
        // Every thread derives this same scale, even one still reading the gathered bits
        if (tensor_scale && blockIdx.x == 0 && threadIdx.x == 0)
            *tensor_scale = nk_cast_block_scaled_to_tensor_scale_simt_(arguments);
        break;
    }
}

#pragma endregion Block Scaled Casts

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_
#endif // NUMKONG_CAST_SIMT_CUH
