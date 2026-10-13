/**
 *  @file include/numkong/cast/metal.metal
 *  @author Ash Vardanian
 *  @date October 12, 2026
 *  @brief Bulk and block-scaled casts on the SIMT cores of every Apple GPU.
 *
 *  @sa include/numkong/cast/metal.h, which embeds and launches this source
 *  @sa include/numkong/cast/simt.cuh, the CUDA and ROCm sibling
 *
 *  A bulk cast gives each thread units a grid apart: a value, or the 2 or 8 values sharing a packed
 *  byte, through the hub the serial loop picks. Metal has no F64 and compiles with fast math, which
 *  may flush subnormals and fold NaN tests, so every float stays in its bits: F64 values travel as
 *  @c ulong and F32 ones as @c uint, rounded to nearest even by integer arithmetic exactly where
 *  the serial casts round. A block-scaled cast gives each thread one chunk of the larger block and
 *  multiplies and divides its scales on F32 bits too, with the NaN choices of Arm cores. A derived
 *  destination tensor scale first gathers the abs-max of the source into its bits with the sign
 *  set, through @c atomic_fetch_max_explicit, so it reads as "derive" until the chunk pass writes
 *  the final scale over it.
 */

/** The bits of @c nk_dtype_t that the launcher passes. */
enum nk_dtype_t : uint {
    nk_dtype_unknown_k = 0,
    nk_u1_k = 1u << 1,
    nk_i8_k = 1u << 2,
    nk_i16_k = 1u << 3,
    nk_i32_k = 1u << 4,
    nk_i64_k = 1u << 5,
    nk_u8_k = 1u << 6,
    nk_u16_k = 1u << 7,
    nk_u32_k = 1u << 8,
    nk_u64_k = 1u << 9,
    nk_f64_k = 1u << 10,
    nk_f32_k = 1u << 11,
    nk_f16_k = 1u << 12,
    nk_bf16_k = 1u << 13,
    nk_e4m3_k = 1u << 14,
    nk_e5m2_k = 1u << 15,
    nk_i4_k = 1u << 16,
    nk_u4_k = 1u << 17,
    nk_e2m3_k = 1u << 18,
    nk_e3m2_k = 1u << 19,
    nk_f64c_k = 1u << 20,
    nk_f32c_k = 1u << 21,
    nk_f16c_k = 1u << 22,
    nk_bf16c_k = 1u << 23,
    nk_e2m1_k = 1u << 24,
    nk_ue8m0_k = 1u << 25,
    nk_ue4m3_k = 1u << 26,
};

/** The hub a bulk cast converts through, laid out as the C launcher's enumeration of that name. */
enum nk_cast_hub_metal_t : uint {
    nk_cast_hub_bytes_metal_k = 0,
    nk_cast_hub_bf16_metal_k,
    nk_cast_hub_u64_metal_k,
    nk_cast_hub_i64_metal_k,
    nk_cast_hub_f32c_metal_k,
    nk_cast_hub_f64c_metal_k,
};

/** The launch record of a bulk cast, laid out as the C launcher's record of the same name. */
struct nk_cast_arguments_metal_t {
    ulong count, units;
    uint from_dtype, to_dtype, hub, unit_values;
};

/** The launch record of a block-scaled cast, laid out as the C launcher's record of that name. */
struct nk_cast_block_scaled_arguments_metal_t {
    ulong count, chunks;
    uint from_dtype, from_scale_dtype, from_tensor_scale_dtype, from_block;
    uint to_dtype, to_scale_dtype, to_tensor_scale_dtype, to_block;
};

static_assert(sizeof(nk_cast_arguments_metal_t) == 32, "mirrors the C record in cast/metal.h");
static_assert(sizeof(nk_cast_block_scaled_arguments_metal_t) == 48, "mirrors the C record in cast/metal.h");

/** The F32 bits of one and of the default NaN of Arm cores. */
constant constexpr uint nk_f32_one_metal_k = 0x3F800000u, nk_f32_nan_metal_k = 0x7FC00000u;

/** Whether the F32 of @p bits is a NaN. */
inline bool nk_f32_is_nan_metal_(uint bits) { return (bits & 0x7FFFFFFFu) > 0x7F800000u; }

/** Sets the quiet bit of an F32 NaN, as the round trip of a CPU through F64 does. */
inline uint nk_f32_quiet_metal_(uint bits) { return nk_f32_is_nan_metal_(bits) ? bits | 0x00400000u : bits; }

/** The F32 bits of @p integer times 2 to the power of @p exponent, a normal number or zero. */
inline uint nk_f32_from_scaled_integer_metal_(uint integer, int exponent) {
    if (!integer) return 0;
    uint const top = 31 - clz(integer);
    return (uint(int(top) + exponent + 127) << 23) | ((integer << (23 - top)) & 0x7FFFFFu);
}

/** The F64 bits nearest @p magnitude, ties to even, with the sign bit @p sign. */
inline ulong nk_f64_from_magnitude_metal_(ulong magnitude, ulong sign) {
    if (!magnitude) return sign;
    uint top = 63 - uint(clz(magnitude));
    ulong significand = magnitude;
    if (top > 52) {
        uint const shift = top - 52;
        ulong const remainder = magnitude & ((1ul << shift) - 1), halfway = 1ul << (shift - 1);
        significand = magnitude >> shift;
        if (remainder > halfway || (remainder == halfway && (significand & 1))) ++significand;
        if (significand >> 53) significand >>= 1, ++top;
    }
    else significand <<= 52 - top;
    return sign | (ulong(top + 1023) << 52) | (significand & 0xFFFFFFFFFFFFFul);
}

/** Widens the F32 of @p bits to F64 bits, exactly, quieting a NaN and keeping its payload. */
inline ulong nk_f64_from_f32_metal_(uint bits) {
    ulong const sign = ulong(bits & 0x80000000u) << 32;
    uint const exponent = (bits >> 23) & 0xFFu, mantissa = bits & 0x7FFFFFu;
    if (exponent == 0xFF)
        return sign | (mantissa ? 0x7FF8000000000000ul : 0x7FF0000000000000ul) | (ulong(mantissa) << 29);
    if (exponent) return sign | (ulong(exponent + 896) << 52) | (ulong(mantissa) << 29);
    return mantissa ? nk_f64_from_magnitude_metal_(mantissa, sign) - (149ul << 52) : sign;
}

/** Narrows the F64 of @p bits to F32 bits, ties to even, quieting a NaN and keeping the top of its
 *  payload. */
inline uint nk_f32_from_f64_metal_(ulong bits) {
    uint const sign = uint(bits >> 32) & 0x80000000u, exponent = uint(bits >> 52) & 0x7FFu;
    ulong const mantissa = bits & 0xFFFFFFFFFFFFFul;
    if (exponent == 0x7FF) return sign | (mantissa ? 0x7FC00000u | (uint(bits >> 29) & 0x7FFFFFu) : 0x7F800000u);
    int const biased = int(exponent) - 896;
    if (biased >= 255) return sign | 0x7F800000u;
    // Normal results drop 29 bits; subnormal ones drop more, down to nothing past 2^-150
    uint const shift = biased > 0 ? 29 : uint(30 - biased);
    if (!exponent || shift > 54) return sign;
    ulong const significand = mantissa | (1ul << 52);
    ulong const remainder = significand & ((1ul << shift) - 1), halfway = 1ul << (shift - 1);
    ulong rounded = significand >> shift;
    if (remainder > halfway || (remainder == halfway && (rounded & 1))) ++rounded;
    // A normal significand keeps its leading bit, which the exponent field absorbs as a carry
    return sign | (biased > 0 ? (uint(biased - 1) << 23) + uint(rounded) : uint(rounded));
}

/** The two's complement bits of @p magnitude with the sign bit @p sign, saturated into the
 *  negative and positive magnitudes of @p limits. */
inline ulong nk_cast_saturate_metal_(ulong magnitude, ulong sign, ulong2 limits) {
    return sign ? 0ul - min(magnitude, limits.x) : min(magnitude, limits.y);
}

/** The largest negative and positive magnitudes of integer @p dtype. */
inline ulong2 nk_cast_integer_limits_metal_(uint dtype) {
    switch (dtype) {
    case nk_i8_k: return ulong2(128, 127);
    case nk_u8_k: return ulong2(0, 255);
    case nk_i16_k: return ulong2(32768, 32767);
    case nk_u16_k: return ulong2(0, 65535);
    case nk_i32_k: return ulong2(2147483648ul, 2147483647ul);
    case nk_u32_k: return ulong2(0, 4294967295ul);
    case nk_i64_k: return ulong2(9223372036854775808ul, 9223372036854775807ul);
    default: return ulong2(0, ~0ul);
    }
}

/** The integer nearest the F64 of @p bits, ties to even, saturated into @p limits and NaN to zero,
 *  as two's complement bits. */
inline ulong nk_f64_to_integer_metal_(ulong bits, ulong2 limits) {
    uint const exponent = uint(bits >> 52) & 0x7FFu;
    ulong const mantissa = bits & 0xFFFFFFFFFFFFFul, significand = mantissa | (1ul << 52);
    if (exponent == 0x7FF && mantissa) return 0;
    ulong magnitude = 0;
    if (exponent >= 1023 + 64) magnitude = ~0ul;
    else if (exponent >= 1023 + 52) magnitude = significand << (exponent - 1023 - 52);
    else if (exponent >= 1022) {
        uint const shift = 1023 + 52 - exponent;
        ulong const remainder = significand & ((1ul << shift) - 1), halfway = 1ul << (shift - 1);
        magnitude = significand >> shift;
        if (remainder > halfway || (remainder == halfway && (magnitude & 1))) ++magnitude;
    }
    return nk_cast_saturate_metal_(magnitude, bits >> 63, limits);
}

/** Widens one F16 code to F32 bits, exactly, keeping NaN payloads. */
inline uint nk_f32_from_f16_metal_(uint code) {
    uint const sign = (code & 0x8000u) << 16, exponent = (code >> 10) & 0x1Fu, mantissa = code & 0x3FFu;
    if (exponent == 0x1F) return sign | 0x7F800000u | (mantissa << 13);
    if (exponent) return sign | ((exponent + 112) << 23) | (mantissa << 13);
    return sign | nk_f32_from_scaled_integer_metal_(mantissa, -24);
}

/** Widens one E4M3FN code to F32 bits, its NaN to the quiet NaN of its sign. */
inline uint nk_f32_from_e4m3_metal_(uint code) {
    uint const sign = (code & 0x80u) << 24, exponent = (code >> 3) & 0xFu, mantissa = code & 7u;
    if (exponent == 0xF && mantissa == 7) return sign | 0x7FC00000u;
    if (exponent) return sign | ((exponent + 120) << 23) | (mantissa << 20);
    return sign | nk_f32_from_scaled_integer_metal_(mantissa, -9);
}

/** Widens one E5M2 code to F32 bits, a NaN to the quiet NaN of its sign. */
inline uint nk_f32_from_e5m2_metal_(uint code) {
    return (code & 0x7Fu) > 0x7Cu ? ((code & 0x80u) << 24) | 0x7FC00000u : nk_f32_from_f16_metal_(code << 8);
}

/** Widens one E2M3FN code, its low 6 bits, to F32 bits. */
inline uint nk_f32_from_e2m3_metal_(uint code) {
    uint const exponent = (code >> 3) & 3u, mantissa = code & 7u;
    uint const magnitude = exponent ? ((exponent + 126) << 23) | (mantissa << 20)
                                    : nk_f32_from_scaled_integer_metal_(mantissa, -3);
    return ((code & 0x20u) << 26) | magnitude;
}

/** Widens one E3M2FN code, its low 6 bits, to F32 bits. */
inline uint nk_f32_from_e3m2_metal_(uint code) {
    uint const exponent = (code >> 2) & 7u, mantissa = code & 3u;
    uint const magnitude = exponent ? ((exponent + 124) << 23) | (mantissa << 21)
                                    : nk_f32_from_scaled_integer_metal_(mantissa, -4);
    return ((code & 0x20u) << 26) | magnitude;
}

/** Widens the E2M1 code in the low 4 bits of @p nibble to F32 bits, keeping a signed zero. */
inline uint nk_f32_from_e2m1_metal_(uint nibble) {
    uint const exponent = (nibble >> 1) & 3u, mantissa = nibble & 1u;
    uint const magnitude = exponent ? ((exponent + 126) << 23) | (mantissa << 22) : mantissa * 0x3F000000u;
    return ((nibble & 8u) << 28) | magnitude;
}

/** Widens one UE8M0 scale to F32 bits: 0x00 to zero and 0xFF to a NaN. */
inline uint nk_f32_from_ue8m0_metal_(uint code) { return (code << 23) | (uint(code == 0xFF) << 22); }

/** Narrows the F32 of @p bits to F16, ties to even, keeping NaN payloads like the serial cast. */
inline ushort nk_f32_to_f16_metal_(uint bits) {
    uint const sign = (bits >> 16) & 0x8000u, exponent = (bits >> 23) & 0xFFu, mantissa = bits & 0x7FFFFFu;
    if (exponent == 0xFF) return ushort(sign | 0x7C00u | (mantissa ? max(mantissa >> 13, 1u) : 0u));
    if (exponent <= 102) return ushort(sign | uint(exponent == 102 && mantissa));
    if (exponent >= 143) return ushort(sign | 0x7C00u);
    // Subnormal halves drop 14 to 23 bits, normal ones 13, carrying into the exponent field
    uint const shift = exponent < 113 ? 126 - exponent : 13;
    uint const significand = mantissa | 0x800000u, halfway = 1u << (shift - 1),
               remainder = significand & ((halfway << 1) - 1);
    uint rounded = significand >> shift;
    if (remainder > halfway || (remainder == halfway && (rounded & 1))) ++rounded;
    return ushort(sign | (exponent < 113 ? rounded : min(((exponent - 113) << 10) + rounded, 0x7C00u)));
}

/** Narrows the F32 of @p bits to BF16, ties to even, keeping a NaN's sign and top payload bits. */
inline ushort nk_f32_to_bf16_metal_(uint bits) {
    return ushort(nk_f32_is_nan_metal_(bits) ? (bits >> 16) | 0x40u : (bits + 0x7FFFu + ((bits >> 16) & 1)) >> 16);
}

/** The code of the F32 of @p bits in a float of @p mantissa_bits_ and exponent bias @p bias_,
 *  signed at bit @p sign_bit_, ties to even and clamped at @p largest_code_, or @p nan_code_ for
 *  NaN. Subnormal codes count steps of the smallest subnormal, so rounding up past them lands on
 *  the first normal code, and normal codes round on the F32 bits, carrying into the exponent. */
template <uint mantissa_bits_, uint bias_, uint sign_bit_, uint largest_code_, uint nan_code_>
inline uint nk_f32_to_minifloat_metal_(uint bits) {
    uint const magnitude = bits & 0x7FFFFFFFu, sign = (bits >> 31) << sign_bit_;
    if (magnitude > 0x7F800000u) return sign | nan_code_;
    uint code = 0;
    if (magnitude < (128 - bias_) << 23) {
        // Below 2^-25 of the subnormal step every value rounds to zero, F32 subnormals included
        uint const shift = 151 - bias_ - mantissa_bits_ - (magnitude >> 23);
        if (shift <= 25) {
            uint const significand = (magnitude & 0x7FFFFFu) | 0x800000u, halfway = 1u << (shift - 1);
            uint const remainder = significand & ((halfway << 1) - 1);
            code = significand >> shift;
            if (remainder > halfway || (remainder == halfway && (code & 1))) ++code;
        }
    }
    else {
        uint const shift = 23 - mantissa_bits_;
        code = ((magnitude + (1u << (shift - 1)) - 1 + ((magnitude >> shift) & 1)) >> shift) -
               ((127 - bias_) << mantissa_bits_);
    }
    return sign | min(code, largest_code_);
}

/** Narrows the F32 of @p bits to the E2M1 code of a nibble: ties to even, saturating at 6, and a
 *  NaN to 6 of its sign. */
inline uint nk_f32_to_e2m1_metal_(uint bits) {
    uint const sign = (bits >> 28) & 8u, magnitude = bits & 0x7FFFFFFFu;
    uint const code = magnitude <= 0x3E800000u   ? 0
                      : magnitude < 0x3F400000u  ? 1
                      : magnitude <= 0x3FA00000u ? 2
                      : magnitude < 0x3FE00000u  ? 3
                      : magnitude <= 0x40200000u ? 4
                      : magnitude < 0x40600000u  ? 5
                      : magnitude <= 0x40A00000u ? 6
                                                 : 7;
    return sign | code;
}

/** Encodes the magnitude of the F32 of @p bits as the nearest UE8M0 power of two, saturating at
 *  0xFE, with a NaN at 0xFF and zeros and subnormals at 0x00. */
inline uint nk_f32_to_ue8m0_metal_(uint bits) {
    uint const magnitude = bits & 0x7FFFFFFFu;
    if (magnitude > 0x7F800000u) return 0xFF;
    if (magnitude < 0x00800000u) return 0;
    if (magnitude == 0x7F800000u) return 0xFE;
    // The rounding midpoint between powers of two is the mantissa fraction √2 - 1
    return min((magnitude >> 23) + uint((magnitude & 0x7FFFFFu) >= 0x3504F4u), 0xFEu);
}

/** Encodes the magnitude of the F32 of @p bits as UE4M3, ties to even, a NaN to 0x7F. */
inline uint nk_f32_to_ue4m3_metal_(uint bits) {
    return nk_f32_to_minifloat_metal_<3, 7, 7, 0x7E, 0x7F>(bits & 0x7FFFFFFFu);
}

/** Code @p index of a packed 4-bit type: the high nibble of its byte for even indices. */
inline uint nk_cast_nibble_metal_(device uchar const *bytes, ulong index) {
    return index & 1 ? bytes[index / 2] & 0xFu : bytes[index / 2] >> 4;
}

/** Bit @p index of packed bits, the most significant bit of each byte first. */
inline uint nk_cast_bit_metal_(device uchar const *bytes, ulong index) {
    return (bytes[index / 8] >> (7 - index % 8)) & 1u;
}

/** Reads value @p index of @p dtype, which F32 holds exactly, as an F32 pair quieted like the
 *  serial F64 hub. */
inline uint2 nk_cast_load_f32c_metal_(device uchar const *bytes, uint dtype, ulong index) {
    device uint const *u32s = (device uint const *)bytes;
    device ushort const *u16s = (device ushort const *)bytes;
    uint real = 0, imaginary = 0;
    switch (dtype) {
    case nk_f32_k: real = u32s[index]; break;
    case nk_f32c_k: real = u32s[index * 2], imaginary = u32s[index * 2 + 1]; break;
    case nk_f16_k: real = nk_f32_from_f16_metal_(u16s[index]); break;
    case nk_f16c_k:
        real = nk_f32_from_f16_metal_(u16s[index * 2]), imaginary = nk_f32_from_f16_metal_(u16s[index * 2 + 1]);
        break;
    case nk_bf16_k: real = uint(u16s[index]) << 16; break;
    case nk_bf16c_k: real = uint(u16s[index * 2]) << 16, imaginary = uint(u16s[index * 2 + 1]) << 16; break;
    case nk_e4m3_k: real = nk_f32_from_e4m3_metal_(bytes[index]); break;
    case nk_e5m2_k: real = nk_f32_from_e5m2_metal_(bytes[index]); break;
    case nk_e2m3_k: real = nk_f32_from_e2m3_metal_(bytes[index]); break;
    case nk_e3m2_k: real = nk_f32_from_e3m2_metal_(bytes[index]); break;
    case nk_e2m1_k: real = nk_f32_from_e2m1_metal_(nk_cast_nibble_metal_(bytes, index)); break;
    case nk_ue8m0_k: real = nk_f32_from_ue8m0_metal_(bytes[index]); break;
    case nk_ue4m3_k: real = nk_f32_from_e4m3_metal_(bytes[index] & 0x7Fu); break;
    case nk_i8_k: {
        int const value = (char)bytes[index];
        real = (value < 0 ? 0x80000000u : 0) | nk_f32_from_scaled_integer_metal_(uint(abs(value)), 0);
    } break;
    case nk_u8_k: real = nk_f32_from_scaled_integer_metal_(bytes[index], 0); break;
    case nk_i16_k: {
        int const value = (short)u16s[index];
        real = (value < 0 ? 0x80000000u : 0) | nk_f32_from_scaled_integer_metal_(uint(abs(value)), 0);
    } break;
    case nk_u16_k: real = nk_f32_from_scaled_integer_metal_(u16s[index], 0); break;
    case nk_i4_k: {
        int const value = int(nk_cast_nibble_metal_(bytes, index) ^ 8u) - 8;
        real = (value < 0 ? 0x80000000u : 0) | nk_f32_from_scaled_integer_metal_(uint(abs(value)), 0);
    } break;
    case nk_u4_k: real = nk_f32_from_scaled_integer_metal_(nk_cast_nibble_metal_(bytes, index), 0); break;
    case nk_u1_k: real = nk_cast_bit_metal_(bytes, index) * nk_f32_one_metal_k; break;
    default: break;
    }
    return uint2(nk_f32_quiet_metal_(real), nk_f32_quiet_metal_(imaginary));
}

/** Reads value @p index of @p dtype as an F64 pair, like @c nk_scalar_buffer_to_f64c_serial_. */
inline ulong2 nk_cast_load_f64c_metal_(device uchar const *bytes, uint dtype, ulong index) {
    device ulong const *u64s = (device ulong const *)bytes;
    device uint const *u32s = (device uint const *)bytes;
    switch (dtype) {
    case nk_f64_k: return ulong2(u64s[index], 0);
    case nk_f64c_k: return ulong2(u64s[index * 2], u64s[index * 2 + 1]);
    case nk_i32_k: {
        int const value = (int)u32s[index];
        return ulong2(nk_f64_from_magnitude_metal_(value < 0 ? 0u - uint(value) : uint(value),
                                                   value < 0 ? 0x8000000000000000ul : 0),
                      0);
    }
    case nk_u32_k: return ulong2(nk_f64_from_magnitude_metal_(u32s[index], 0), 0);
    case nk_i64_k: {
        long const value = (long)u64s[index];
        return ulong2(nk_f64_from_magnitude_metal_(value < 0 ? 0ul - ulong(value) : ulong(value),
                                                   value < 0 ? 0x8000000000000000ul : 0),
                      0);
    }
    case nk_u64_k: return ulong2(nk_f64_from_magnitude_metal_(u64s[index], 0), 0);
    default: {
        uint2 const pair = nk_cast_load_f32c_metal_(bytes, dtype, index);
        return ulong2(nk_f64_from_f32_metal_(pair.x), nk_f64_from_f32_metal_(pair.y));
    }
    }
}

/** Reads integer @p dtype value @p index as I64, like @c nk_scalar_buffers_to_i64_serial_. */
inline long nk_cast_load_i64_metal_(device uchar const *bytes, uint dtype, ulong index) {
    device long const *i64s = (device long const *)bytes;
    device int const *i32s = (device int const *)bytes;
    device short const *i16s = (device short const *)bytes;
    device char const *i8s = (device char const *)bytes;
    device ulong const *u64s = (device ulong const *)bytes;
    device uint const *u32s = (device uint const *)bytes;
    device ushort const *u16s = (device ushort const *)bytes;
    switch (dtype) {
    case nk_i64_k: return i64s[index];
    case nk_i32_k: return i32s[index];
    case nk_i16_k: return i16s[index];
    case nk_i8_k: return i8s[index];
    case nk_u64_k: return long(min(u64s[index], 9223372036854775807ul));
    case nk_u32_k: return u32s[index];
    case nk_u16_k: return u16s[index];
    case nk_u8_k: return bytes[index];
    case nk_i4_k: return int(nk_cast_nibble_metal_(bytes, index) ^ 8u) - 8;
    case nk_u4_k: return nk_cast_nibble_metal_(bytes, index);
    case nk_u1_k: return nk_cast_bit_metal_(bytes, index);
    default: return 0;
    }
}

/** Reads unsigned @p dtype value @p index as U64, like @c nk_scalar_buffers_to_u64_serial_. */
inline ulong nk_cast_load_u64_metal_(device uchar const *bytes, uint dtype, ulong index) {
    device ulong const *u64s = (device ulong const *)bytes;
    device uint const *u32s = (device uint const *)bytes;
    device ushort const *u16s = (device ushort const *)bytes;
    switch (dtype) {
    case nk_u64_k: return u64s[index];
    case nk_u32_k: return u32s[index];
    case nk_u16_k: return u16s[index];
    case nk_u8_k: return bytes[index];
    case nk_u4_k: return nk_cast_nibble_metal_(bytes, index);
    case nk_u1_k: return nk_cast_bit_metal_(bytes, index);
    default: return 0;
    }
}

/** Writes the low bits of @p bits as value @p index of an integer @p dtype a byte or wider. */
inline void nk_cast_store_integer_metal_(ulong bits, device uchar *bytes, uint dtype, ulong index) {
    device ushort *u16s = (device ushort *)bytes;
    device uint *u32s = (device uint *)bytes;
    device ulong *u64s = (device ulong *)bytes;
    switch (dtype) {
    case nk_i8_k:
    case nk_u8_k: bytes[index] = uchar(bits); break;
    case nk_i16_k:
    case nk_u16_k: u16s[index] = ushort(bits); break;
    case nk_i32_k:
    case nk_u32_k: u32s[index] = uint(bits); break;
    default: u64s[index] = bits; break;
    }
}

/** Writes an F32 pair as value @p index of @p dtype, a byte or wider, which F32 rounds into: the
 *  floats through their encoders and the integers to nearest even with saturation. */
inline void nk_cast_store_f32c_metal_(uint2 value, device uchar *bytes, uint dtype, ulong index) {
    device uint *u32s = (device uint *)bytes;
    device ushort *u16s = (device ushort *)bytes;
    switch (dtype) {
    case nk_f32_k: u32s[index] = value.x; break;
    case nk_f32c_k: u32s[index * 2] = value.x, u32s[index * 2 + 1] = value.y; break;
    case nk_f16_k: u16s[index] = nk_f32_to_f16_metal_(value.x); break;
    case nk_f16c_k:
        u16s[index * 2] = nk_f32_to_f16_metal_(value.x), u16s[index * 2 + 1] = nk_f32_to_f16_metal_(value.y);
        break;
    case nk_bf16_k: u16s[index] = nk_f32_to_bf16_metal_(value.x); break;
    case nk_bf16c_k:
        u16s[index * 2] = nk_f32_to_bf16_metal_(value.x), u16s[index * 2 + 1] = nk_f32_to_bf16_metal_(value.y);
        break;
    case nk_e4m3_k: bytes[index] = uchar(nk_f32_to_minifloat_metal_<3, 7, 7, 0x7E, 0x7F>(value.x)); break;
    case nk_e5m2_k: bytes[index] = uchar(nk_f32_to_minifloat_metal_<2, 15, 7, 0x7C, 0x7D>(value.x)); break;
    case nk_e2m3_k: bytes[index] = uchar(nk_f32_to_minifloat_metal_<3, 1, 5, 0x1F, 0x1F>(value.x)); break;
    case nk_e3m2_k: bytes[index] = uchar(nk_f32_to_minifloat_metal_<2, 3, 5, 0x1F, 0x1F>(value.x)); break;
    case nk_ue8m0_k: bytes[index] = uchar(nk_f32_to_ue8m0_metal_(value.x)); break;
    case nk_ue4m3_k: bytes[index] = uchar(nk_f32_to_ue4m3_metal_(value.x)); break;
    default:
        nk_cast_store_integer_metal_(
            nk_f64_to_integer_metal_(nk_f64_from_f32_metal_(value.x), nk_cast_integer_limits_metal_(dtype)), bytes,
            dtype, index);
        break;
    }
}

/** Writes an F64 pair as value @p index of @p dtype, a byte or wider, like
 *  @c nk_scalar_buffer_from_f64c_serial_: narrower floats round to F32 first. */
inline void nk_cast_store_f64c_metal_(ulong2 value, device uchar *bytes, uint dtype, ulong index) {
    device ulong *u64s = (device ulong *)bytes;
    switch (dtype) {
    case nk_f64_k: u64s[index] = value.x; break;
    case nk_f64c_k: u64s[index * 2] = value.x, u64s[index * 2 + 1] = value.y; break;
    case nk_i8_k:
    case nk_u8_k:
    case nk_i16_k:
    case nk_u16_k:
    case nk_i32_k:
    case nk_u32_k:
    case nk_i64_k:
    case nk_u64_k:
        nk_cast_store_integer_metal_(nk_f64_to_integer_metal_(value.x, nk_cast_integer_limits_metal_(dtype)), bytes,
                                     dtype, index);
        break;
    default:
        nk_cast_store_f32c_metal_(uint2(nk_f32_from_f64_metal_(value.x), nk_f32_from_f64_metal_(value.y)), bytes, dtype,
                                  index);
        break;
    }
}

/** The code of the F64 of @p bits in a packed @p dtype, like the serial F64 hub packs it: I4 and
 *  U4 saturated and rounded to nearest even, U1 for any nonzero, and E2M1 through F32. */
inline uint nk_cast_code_f64_metal_(ulong bits, uint dtype) {
    switch (dtype) {
    case nk_i4_k: return uint(nk_f64_to_integer_metal_(bits, ulong2(8, 7))) & 0xFu;
    case nk_u4_k: return uint(nk_f64_to_integer_metal_(bits, ulong2(0, 15)));
    case nk_u1_k: return (bits << 1) != 0;
    default: return nk_f32_to_e2m1_metal_(nk_f32_from_f64_metal_(bits));
    }
}

/** Reads value @p index of @p dtype into the components of a pair that @p hub uses. */
inline ulong2 nk_cast_load_metal_(uint hub, device uchar const *bytes, uint dtype, ulong index) {
    switch (hub) {
    case nk_cast_hub_u64_metal_k: return ulong2(nk_cast_load_u64_metal_(bytes, dtype, index), 0);
    case nk_cast_hub_i64_metal_k: return ulong2(ulong(nk_cast_load_i64_metal_(bytes, dtype, index)), 0);
    case nk_cast_hub_f32c_metal_k: {
        uint2 const pair = nk_cast_load_f32c_metal_(bytes, dtype, index);
        return ulong2(pair.x, pair.y);
    }
    default: return nk_cast_load_f64c_metal_(bytes, dtype, index);
    }
}

/** Writes the pair @p value of @p hub as value @p index of a @p dtype a byte wide or wider. */
inline void nk_cast_store_metal_(uint hub, ulong2 value, device uchar *bytes, uint dtype, ulong index) {
    switch (hub) {
    case nk_cast_hub_u64_metal_k:
        nk_cast_store_integer_metal_(nk_cast_saturate_metal_(value.x, 0, nk_cast_integer_limits_metal_(dtype)), bytes,
                                     dtype, index);
        break;
    case nk_cast_hub_i64_metal_k: {
        ulong const sign = value.x >> 63;
        nk_cast_store_integer_metal_(
            nk_cast_saturate_metal_(sign ? 0ul - value.x : value.x, sign, nk_cast_integer_limits_metal_(dtype)), bytes,
            dtype, index);
    } break;
    case nk_cast_hub_f32c_metal_k:
        nk_cast_store_f32c_metal_(uint2(uint(value.x), uint(value.y)), bytes, dtype, index);
        break;
    default: nk_cast_store_f64c_metal_(value, bytes, dtype, index); break;
    }
}

/** The code of the pair @p value of @p hub in a packed @p dtype: I4, U4, E2M1 or U1. */
inline uint nk_cast_code_metal_(uint hub, ulong2 value, uint dtype) {
    switch (hub) {
    case nk_cast_hub_u64_metal_k: return dtype == nk_u1_k ? uint(value.x != 0) : uint(min(value.x, 15ul));
    case nk_cast_hub_i64_metal_k: return uint(clamp(long(value.x), -8l, 7l)) & 0xFu;
    case nk_cast_hub_f32c_metal_k: return nk_cast_code_f64_metal_(nk_f64_from_f32_metal_(uint(value.x)), dtype);
    default: return nk_cast_code_f64_metal_(value.x, dtype);
    }
}

/** Converts unit @p unit of a bulk cast: one value, the 2 or 8 values sharing a packed byte, or
 *  one word of a copy. A packed destination gets whole bytes only, like the serial batches. */
inline void nk_cast_unit_metal_(device uchar const *from, device uchar *to,
                                constant nk_cast_arguments_metal_t &arguments, ulong unit) {
    uint const hub = arguments.hub, to_dtype = arguments.to_dtype, from_dtype = arguments.from_dtype;
    device uint4 const *from_u128s = (device uint4 const *)from;
    device ulong const *from_u64s = (device ulong const *)from;
    device uint const *from_u32s = (device uint const *)from;
    device ushort const *from_u16s = (device ushort const *)from;
    device uint4 *to_u128s = (device uint4 *)to;
    device ulong *to_u64s = (device ulong *)to;
    device uint *to_u32s = (device uint *)to;
    device ushort *to_u16s = (device ushort *)to;
    if (hub == nk_cast_hub_bytes_metal_k) {
        switch (arguments.unit_values) {
        case 16: to_u128s[unit] = from_u128s[unit]; break;
        case 8: to_u64s[unit] = from_u64s[unit]; break;
        case 4: to_u32s[unit] = from_u32s[unit]; break;
        case 2: to_u16s[unit] = from_u16s[unit]; break;
        default: to[unit] = from[unit]; break;
        }
        return;
    }
    if (hub == nk_cast_hub_bf16_metal_k) {
        to_u16s[unit] = nk_f32_to_bf16_metal_(from_u32s[unit]);
        return;
    }
    ulong const first = unit * arguments.unit_values, left = arguments.count - first;
    uint const values = left < arguments.unit_values ? uint(left) : arguments.unit_values;
    if (to_dtype == nk_u1_k) {
        uint packed = 0;
        for (uint offset = 0; offset != values; ++offset)
            packed |= nk_cast_code_metal_(hub, nk_cast_load_metal_(hub, from, from_dtype, first + offset), nk_u1_k)
                      << (7 - offset);
        to[first / 8] = uchar(packed);
    }
    else if (to_dtype == nk_i4_k || to_dtype == nk_u4_k || to_dtype == nk_e2m1_k) {
        for (uint offset = 0; offset + 2 <= values; offset += 2) {
            ulong2 const high = nk_cast_load_metal_(hub, from, from_dtype, first + offset);
            ulong2 const low = nk_cast_load_metal_(hub, from, from_dtype, first + offset + 1);
            to[(first + offset) / 2] = uchar((nk_cast_code_metal_(hub, high, to_dtype) << 4) |
                                             nk_cast_code_metal_(hub, low, to_dtype));
        }
    }
    else
        for (uint offset = 0; offset != values; ++offset)
            nk_cast_store_metal_(hub, nk_cast_load_metal_(hub, from, from_dtype, first + offset), to, to_dtype,
                                 first + offset);
}

kernel void nk_cast_metal_kernel_(device uchar const *from [[buffer(0)]], device uchar *to [[buffer(1)]],
                                  constant nk_cast_arguments_metal_t &arguments [[buffer(2)]],
                                  uint position [[thread_position_in_grid]], uint threads [[threads_per_grid]]) {
    for (ulong unit = position; unit < arguments.units; unit += threads) nk_cast_unit_metal_(from, to, arguments, unit);
}

/** The F32 NaN that an Arm core returns for an operation with a NaN among @p a and @p b: the first
 *  signaling one, else the first quiet one, quieted. */
inline uint nk_f32_propagate_nan_metal_(uint a, uint b) {
    bool const a_nan = nk_f32_is_nan_metal_(a), b_nan = nk_f32_is_nan_metal_(b);
    bool const a_signaling = a_nan && !(a & 0x00400000u), b_signaling = b_nan && !(b & 0x00400000u);
    return (a_signaling ? a : b_signaling ? b : a_nan ? a : b) | 0x00400000u;
}

/** The significand of the finite nonzero F32 magnitude @p magnitude scaled into [2^23, 2^24),
 *  with the power of two it multiplies in @p exponent. */
inline uint nk_f32_unpack_metal_(uint magnitude, thread int &exponent) {
    uint const biased = magnitude >> 23;
    if (biased) {
        exponent = int(biased) - 150;
        return (magnitude & 0x7FFFFFu) | 0x800000u;
    }
    uint const shift = clz(magnitude) - 8;
    exponent = -149 - int(shift);
    return magnitude << shift;
}

/** The F32 bits nearest @p significand times 2 to the power of @p exponent, ties to even, with
 *  @p sign, where a nonzero @p tail holds bits cut below the significand, kept below 2^62. */
inline uint nk_f32_round_metal_(uint sign, ulong significand, int exponent, ulong tail) {
    int const top = 63 - int(clz(significand)), biased = top + exponent + 127;
    if (biased >= 255) return sign | 0x7F800000u;
    int const shift = top - 23 + (biased < 1 ? 1 - biased : 0);
    if (shift >= 64) return sign;
    ulong rounded = significand;
    if (shift > 0) {
        ulong const halfway = 1ul << (shift - 1), remainder = significand & ((halfway << 1) - 1);
        rounded = significand >> shift;
        if (remainder > halfway || (remainder == halfway && (tail || (rounded & 1)))) ++rounded;
    }
    else rounded <<= -shift;
    // A normal significand keeps its leading bit, which the exponent field absorbs as a carry
    return sign | (biased < 1 ? uint(rounded) : (uint(biased - 1) << 23) + uint(rounded));
}

/** The F32 product of the F32s of @p a and @p b, rounded to nearest even, subnormals included. */
inline uint nk_f32_multiply_metal_(uint a, uint b) {
    uint const sign = (a ^ b) & 0x80000000u, a_magnitude = a & 0x7FFFFFFFu, b_magnitude = b & 0x7FFFFFFFu;
    if (a_magnitude > 0x7F800000u || b_magnitude > 0x7F800000u) return nk_f32_propagate_nan_metal_(a, b);
    if (a_magnitude == 0x7F800000u || b_magnitude == 0x7F800000u)
        return !a_magnitude || !b_magnitude ? nk_f32_nan_metal_k : sign | 0x7F800000u;
    if (!a_magnitude || !b_magnitude) return sign;
    int a_exponent, b_exponent;
    ulong const product = ulong(nk_f32_unpack_metal_(a_magnitude, a_exponent)) *
                          nk_f32_unpack_metal_(b_magnitude, b_exponent);
    return nk_f32_round_metal_(sign, product, a_exponent + b_exponent, 0);
}

/** The F32 quotient of the F32s of @p a and @p b, rounded to nearest even, subnormals included. */
inline uint nk_f32_divide_metal_(uint a, uint b) {
    uint const sign = (a ^ b) & 0x80000000u, a_magnitude = a & 0x7FFFFFFFu, b_magnitude = b & 0x7FFFFFFFu;
    if (a_magnitude > 0x7F800000u || b_magnitude > 0x7F800000u) return nk_f32_propagate_nan_metal_(a, b);
    if (a_magnitude == 0x7F800000u) return b_magnitude == 0x7F800000u ? nk_f32_nan_metal_k : sign | 0x7F800000u;
    if (b_magnitude == 0x7F800000u) return sign;
    if (!b_magnitude) return a_magnitude ? sign | 0x7F800000u : nk_f32_nan_metal_k;
    if (!a_magnitude) return sign;
    int a_exponent, b_exponent;
    ulong const numerator = ulong(nk_f32_unpack_metal_(a_magnitude, a_exponent)) << 38;
    ulong const denominator = nk_f32_unpack_metal_(b_magnitude, b_exponent);
    return nk_f32_round_metal_(sign, numerator / denominator, a_exponent - 38 - b_exponent, numerator % denominator);
}

/** The largest magnitude of each block-scaled element dtype, as F32 bits. */
inline uint nk_cast_element_max_metal_(uint dtype) {
    switch (dtype) {
    case nk_e5m2_k: return 0x47600000u;
    case nk_e4m3_k: return 0x43E00000u;
    case nk_e2m3_k: return 0x40F00000u;
    case nk_e3m2_k: return 0x41E00000u;
    case nk_e2m1_k: return 0x40C00000u;
    case nk_i8_k: return 0x42FE0000u;
    default: return nk_f32_one_metal_k;
    }
}

/** Decodes one scale byte of @p scale_dtype to F32 bits. */
inline uint nk_cast_decode_scale_metal_(uint code, uint scale_dtype) {
    return scale_dtype == nk_ue8m0_k ? nk_f32_from_ue8m0_metal_(code) : nk_f32_from_e4m3_metal_(code & 0x7Fu);
}

/** Reads value @p index of @p dtype as F32 bits, like @c nk_cast_elementwise_serial_ into F32. */
inline uint nk_cast_decode_f32_metal_(device uchar const *bytes, uint dtype, ulong index) {
    device uint const *u32s = (device uint const *)bytes;
    switch (dtype) {
    case nk_f32_k: return u32s[index];
    case nk_f64_k:
    case nk_f64c_k:
    case nk_i32_k:
    case nk_u32_k:
    case nk_i64_k:
    case nk_u64_k: return nk_f32_from_f64_metal_(nk_cast_load_f64c_metal_(bytes, dtype, index).x);
    default: return nk_cast_load_f32c_metal_(bytes, dtype, index).x;
    }
}

/** Writes @p count F32 values from @p values as @p dtype from value @p first on, like
 *  @c nk_cast_elementwise_serial_ from F32, with @p first a multiple of the values per byte. */
inline void nk_cast_encode_f32s_metal_(thread uint const *values, uint count, device uchar *bytes, uint dtype,
                                       ulong first) {
    if (dtype == nk_u1_k)
        for (uint begin = 0; begin < count; begin += 8) {
            uint packed = 0;
            for (uint offset = 0; offset != 8 && begin + offset < count; ++offset)
                packed |= uint((values[begin + offset] << 1) != 0) << (7 - offset);
            bytes[(first + begin) / 8] = uchar(packed);
        }
    else if (dtype == nk_i4_k || dtype == nk_u4_k || dtype == nk_e2m1_k)
        for (uint offset = 0; offset + 2 <= count; offset += 2)
            bytes[(first + offset) / 2] = uchar(
                (nk_cast_code_f64_metal_(nk_f64_from_f32_metal_(values[offset]), dtype) << 4) |
                nk_cast_code_f64_metal_(nk_f64_from_f32_metal_(values[offset + 1]), dtype));
    else {
        device uint *u32s = (device uint *)bytes;
        device ushort *u16s = (device ushort *)bytes;
        for (uint offset = 0; offset != count; ++offset) {
            uint const value = values[offset];
            if (dtype == nk_f32_k) u32s[first + offset] = value;
            else if (dtype == nk_bf16_k) u16s[first + offset] = nk_f32_to_bf16_metal_(value);
            else nk_cast_store_f64c_metal_(ulong2(nk_f64_from_f32_metal_(value), 0), bytes, dtype, first + offset);
        }
    }
}

/** The scale of source block @p block as F32 bits: its decoded scale byte times the tensor scale,
 *  or one for a plain source. */
inline uint nk_cast_from_scale_metal_(device uchar const *scales, device uint const *tensor_scale,
                                      constant nk_cast_block_scaled_arguments_metal_t &arguments, ulong block) {
    if (arguments.from_scale_dtype == nk_dtype_unknown_k) return nk_f32_one_metal_k;
    uint const tensor = arguments.from_tensor_scale_dtype == nk_f32_k ? *tensor_scale : nk_f32_one_metal_k;
    return nk_f32_multiply_metal_(nk_cast_decode_scale_metal_(scales[block], arguments.from_scale_dtype), tensor);
}

/** The destination's tensor scale as F32 bits: the caller's, or derived from the abs-max its bits
 *  gathered, or one without a tensor scale. */
inline uint nk_cast_to_tensor_scale_metal_(device atomic_uint *tensor_scale,
                                           constant nk_cast_block_scaled_arguments_metal_t &arguments) {
    if (arguments.to_tensor_scale_dtype != nk_f32_k) return nk_f32_one_metal_k;
    uint const bits = atomic_load_explicit(tensor_scale, memory_order_relaxed);
    if (bits && !(bits >> 31)) return bits;
    uint const amax = bits & 0x7FFFFFFFu;
    uint const scale_max = arguments.to_scale_dtype == nk_ue4m3_k ? 0x43E00000u : nk_f32_one_metal_k;
    return amax ? nk_f32_divide_metal_(
                      amax, nk_f32_multiply_metal_(nk_cast_element_max_metal_(arguments.to_dtype), scale_max))
                : nk_f32_one_metal_k;
}

/** Converts the chunk at @p chunk_start, a step of @c nk_cast_block_scaled_through_f32_serial_. */
inline void nk_cast_block_scaled_chunk_metal_(device uchar const *from, device uchar const *from_scales,
                                              device uint const *from_tensor_scale, device uchar *to,
                                              device uchar *to_scales, device atomic_uint *to_tensor_scale,
                                              constant nk_cast_block_scaled_arguments_metal_t &arguments,
                                              ulong chunk_start) {
    uint scratch[32], encoded[32];
    uint const chunk = max(arguments.from_block, arguments.to_block);
    uint const chunk_count = arguments.count - chunk_start < chunk ? uint(arguments.count - chunk_start) : chunk;
    bool const from_plain = arguments.from_scale_dtype == nk_dtype_unknown_k;
    bool const to_plain = arguments.to_scale_dtype == nk_dtype_unknown_k;
    uint const from_step = from_plain ? chunk_count : arguments.from_block;
    uint const to_step = to_plain ? chunk_count : arguments.to_block;
    uint const to_tensor_scale_bits = nk_cast_to_tensor_scale_metal_(to_tensor_scale, arguments);
    uint const element_max = nk_cast_element_max_metal_(arguments.to_dtype);
    for (uint begin = 0; begin < chunk_count; begin += from_step) {
        uint const valid = min(chunk_count - begin, from_step);
        for (uint offset = 0; offset != valid; ++offset)
            scratch[begin + offset] = nk_cast_decode_f32_metal_(from, arguments.from_dtype,
                                                                chunk_start + begin + offset);
        if (from_plain) continue;
        uint const scale = nk_cast_from_scale_metal_(from_scales, from_tensor_scale, arguments,
                                                     (chunk_start + begin) / arguments.from_block);
        for (uint offset = 0; offset != valid; ++offset)
            scratch[begin + offset] = nk_f32_multiply_metal_(scratch[begin + offset], scale);
    }
    for (uint begin = 0; begin < chunk_count; begin += to_step) {
        uint const valid = min(chunk_count - begin, to_step);
        if (to_plain) {
            nk_cast_encode_f32s_metal_(scratch + begin, valid, to, arguments.to_dtype, chunk_start + begin);
            continue;
        }
        // The abs-max of the block, or its first NaN, as the serial loop finds it
        uint block_amax = 0;
        for (uint offset = 0; offset != valid && !nk_f32_is_nan_metal_(block_amax); ++offset) {
            uint const value = scratch[begin + offset];
            block_amax = nk_f32_is_nan_metal_(value) ? value : max(block_amax, value & 0x7FFFFFFFu);
        }
        uint raw;
        if (arguments.to_scale_dtype == nk_ue8m0_k) {
            // OCP MX: the floor of log2 amax minus the floor of log2 of the element's largest value
            uint const amax_magnitude = block_amax & 0x7FFFFFFFu;
            raw = amax_magnitude > 0x7F800000u    ? 0xFFu
                  : amax_magnitude == 0x7F800000u ? 0xFEu
                  : amax_magnitude < 0x00800000u
                      ? 0u
                      : uint(clamp(int(amax_magnitude >> 23) - int(element_max >> 23) + 127, 0, 0xFE));
        }
        else
            raw = nk_f32_to_ue4m3_metal_(
                nk_f32_divide_metal_(nk_f32_divide_metal_(block_amax, element_max), to_tensor_scale_bits));
        to_scales[(chunk_start + begin) / arguments.to_block] = uchar(raw);
        uint const effective_scale = nk_f32_multiply_metal_(nk_cast_decode_scale_metal_(raw, arguments.to_scale_dtype),
                                                            to_tensor_scale_bits);
        uint const reciprocal = effective_scale && effective_scale <= 0x7F800000u
                                    ? nk_f32_divide_metal_(nk_f32_one_metal_k, effective_scale)
                                    : 0u;
        for (uint offset = 0; offset != valid; ++offset) {
            uint const scaled = nk_f32_multiply_metal_(scratch[begin + offset], reciprocal);
            bool const clipped = !nk_f32_is_nan_metal_(scaled) && (scaled & 0x7FFFFFFFu) > element_max;
            encoded[offset] = clipped ? (scaled & 0x80000000u) | element_max : scaled;
        }
        nk_cast_encode_f32s_metal_(encoded, valid, to, arguments.to_dtype, chunk_start + begin);
    }
}

kernel void nk_cast_block_scaled_chunks_metal_kernel_(
    device uchar const *from [[buffer(0)]], device uchar const *from_scales [[buffer(1)]],
    device uint const *from_tensor_scale [[buffer(2)]], device uchar *to [[buffer(3)]],
    device uchar *to_scales [[buffer(4)]], device atomic_uint *to_tensor_scale [[buffer(5)]],
    constant nk_cast_block_scaled_arguments_metal_t &arguments [[buffer(6)]], uint position [[thread_position_in_grid]],
    uint threads [[threads_per_grid]]) {
    uint const chunk = max(arguments.from_block, arguments.to_block);
    for (ulong index = position; index < arguments.chunks; index += threads)
        nk_cast_block_scaled_chunk_metal_(from, from_scales, from_tensor_scale, to, to_scales, to_tensor_scale,
                                          arguments, index * chunk);
    // Every thread derives this same scale, even one still reading the gathered bits
    if (arguments.to_tensor_scale_dtype == nk_f32_k && position == 0)
        atomic_store_explicit(to_tensor_scale, nk_cast_to_tensor_scale_metal_(to_tensor_scale, arguments),
                              memory_order_relaxed);
}

/** Gathers the abs-max of the scaled source, skipping NaNs, into the bits of a tensor scale left
 *  to derive, which order like the magnitudes they encode, one atomic per SIMD-group. */
kernel void nk_cast_block_scaled_amax_metal_kernel_(
    device uchar const *from [[buffer(0)]], device uchar const *from_scales [[buffer(1)]],
    device uint const *from_tensor_scale [[buffer(2)]], device uchar *to [[buffer(3)]],
    device uchar *to_scales [[buffer(4)]], device atomic_uint *to_tensor_scale [[buffer(5)]],
    constant nk_cast_block_scaled_arguments_metal_t &arguments [[buffer(6)]], uint position [[thread_position_in_grid]],
    uint threads [[threads_per_grid]], uint simd_lane [[thread_index_in_simdgroup]]) {
    uint const bits = atomic_load_explicit(to_tensor_scale, memory_order_relaxed);
    if (bits && !(bits >> 31)) return;
    uint amax = 0;
    for (ulong index = position; index < arguments.count; index += threads) {
        uint const value = nk_cast_decode_f32_metal_(from, arguments.from_dtype, index);
        uint const scaled = arguments.from_scale_dtype == nk_dtype_unknown_k
                                ? value
                                : nk_f32_multiply_metal_(
                                      value, nk_cast_from_scale_metal_(from_scales, from_tensor_scale, arguments,
                                                                       index / arguments.from_block));
        if (!nk_f32_is_nan_metal_(scaled)) amax = max(amax, scaled & 0x7FFFFFFFu);
    }
    amax = simd_max(amax);
    if (!simd_lane) atomic_fetch_max_explicit(to_tensor_scale, amax | 0x80000000u, memory_order_relaxed);
}
