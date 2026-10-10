/**
 *  @file include/numkong/types.metal
 *  @author Ash Vardanian
 *  @date October 5, 2026
 *  @brief Element types and arithmetic helpers shared by every Metal kernel.
 *
 *  @sa include/numkong/types.h
 *  @sa include/numkong/types.hpp, which names the element types the same way
 *
 *  Every Metal source is compiled with this one concatenated ahead of it. The element types read
 *  one value of a row as @c float, or exactly as an integer, and write one back the way the serial
 *  casts round: to nearest even, saturating where the format has no infinity, and keeping NaN
 *  payloads where it has room for them.
 */
#include <metal_stdlib>

using namespace metal;

namespace nk {

/** Divides rounding up without overflowing the numerator. */
template <typename number_type_, typename divisor_type_>
constexpr number_type_ divide_round_up(number_type_ number, divisor_type_ divisor) {
    return number / divisor + (number % divisor != 0);
}

/** Rounds up to a multiple of @p divisor; the rounded value must fit the result type. */
template <typename number_type_, typename divisor_type_>
constexpr number_type_ round_up_to_multiple(number_type_ number, divisor_type_ divisor) {
    return divide_round_up(number, divisor) * divisor;
}

} // namespace nk

/** 2 raised to @p x, clamped into [-125, 127], through a polynomial of its rounded-off fraction. */
inline float nk_exp2_metal_(float x) {
#pragma clang fp contract(off) reassociate(off)
    x = x > 127.0f ? 127.0f : x;
    x = x < -125.0f ? -125.0f : x;
    int const whole = int(x >= 0 ? x + 0.5f : x - 0.5f);
    float const reduced = x - float(whole);
    float poly = 1.52527338e-5f;
    poly = poly * reduced + 1.54035304e-4f;
    poly = poly * reduced + 1.33335581e-3f;
    poly = poly * reduced + 9.61812910e-3f;
    poly = poly * reduced + 5.55041087e-2f;
    poly = poly * reduced + 2.40226507e-1f;
    poly = poly * reduced + 6.93147181e-1f;
    poly = poly * reduced + 1.0f;
    return poly * as_type<float>(uint(whole + 127) << 23);
}

/** Rounds @p value to an integer from @p low to @p high, ties to even and NaN to zero. */
inline int nk_rint_saturated_metal_(float value, float low, float high) {
    // Fast math may fold a self-comparison away, so NaN is told apart by its bits
    return (as_type<uint>(value) & 0x7FFFFFFFu) > 0x7F800000u ? 0 : int(rint(clamp(value, low, high)));
}

/** The code of @p value in a float of @p mantissa_bits_ and exponent @p bias_, signed at bit
 *  @p sign_bit_, rounded to nearest even and clamped at @p largest_code_, or @p nan_code_ for NaN.
 *  Subnormal codes count steps of the smallest subnormal, so rounding up past them lands on the
 *  first normal code, and normal codes round on the F32 bits, carrying into the exponent. */
template <uint mantissa_bits_, uint bias_, uint sign_bit_, uint largest_code_, uint nan_code_>
inline uchar nk_minifloat_encode_metal_(float value) {
    uint const bits = as_type<uint>(value), magnitude = bits & 0x7FFFFFFFu, sign = (bits >> 31) << sign_bit_;
    uint const shift = 23 - mantissa_bits_;
    uint const code = magnitude < (128 - bias_) << 23
                          ? uint(rint(as_type<float>(magnitude) * float(1u << (bias_ - 1 + mantissa_bits_))))
                          : ((magnitude + (1u << (shift - 1)) - 1 + ((magnitude >> shift) & 1)) >> shift) -
                                ((127 - bias_) << mantissa_bits_);
    return uchar(sign | (magnitude > 0x7F800000u ? nan_code_ : min(code, largest_code_)));
}

/** One dimension's code of a row of nibble pairs, the high nibble first as in @c nk_i4x2_t, so
 *  dimensions 2k and 2k + 1 share byte k. */
inline ushort nk_b4_load_(device uchar const *row, uint index) {
    return (ushort)((row[index >> 1] >> (~index & 1) * 4) & 15);
}

/** The square of one stored value of a widened format, both nibbles of a packed pair when two
 *  dimensions share a byte. Every widened format places its sign, exponent and mantissa bits where
 *  @c half keeps them and rescales by the difference of the biases, a power of two; subnormal codes
 *  land on @c half subnormals, which Apple GPUs keep, so every code widens exactly. */
template <typename dtype_>
inline float nk_widened_squares_(uint bits) {
    float4 const values = dtype_::dimensions_per_value == 2
                              ? float4(dtype_::widen(ushort4(bits & 0xF, (bits >> 4) & 0xF, 0, 0)))
                              : float4(dtype_::widen(ushort4(bits & 0xFF, 0, 0, 0)));
    return values.x * values.x + values.y * values.y;
}

/** One dimension of a widened format's row as @c float. */
template <typename dtype_>
inline float nk_widened_load_(device uchar const *row, uint index) {
    ushort const code = dtype_::dimensions_per_value == 2 ? nk_b4_load_(row, index) : (ushort)row[index];
    return (float)dtype_::widen(ushort4(code, 0, 0, 0)).x;
}

/*  How the kernels read each dtype, named as in `types.hpp`: its stored @c raw_t, the dimensions
 *  one value holds, the @c dot_result_t the @c metal tile sums it in, and the @c norm_t of its
 *  packed squares. A float also names the @c stage_t that four codes widen into for multiplying.
 *  Every dtype counts the dimensions of a 16-byte chunk in @c chunk_values, and every dtype a
 *  kernel writes stores one dimension with @c store. */
namespace nk {

struct f32_t {
    using raw_t = float;
    static constant constexpr uint chunk_values = 4;
    static float load(device uchar const *row, uint index) { return ((device float const *)row)[index]; }
    static void store(device uchar *row, uint index, float value) { ((device float *)row)[index] = value; }
};

struct i32_t {
    using raw_t = int;
    static constant constexpr uint chunk_values = 4;
    static int load(device uchar const *row, uint index) { return ((device int const *)row)[index]; }
};

struct u32_t {
    using raw_t = uint;
    static constant constexpr uint chunk_values = 4;
    static uint load(device uchar const *row, uint index) { return ((device uint const *)row)[index]; }
};

struct i8_t {
    using raw_t = int8_t;
    using dot_result_t = int;
    using norm_t = uint;
    static constant constexpr uint dimensions_per_value = 1;
    static constant constexpr uint chunk_values = 16;
    static int load(device uchar const *row, uint index) { return (int)(char)row[index]; }
    static void store(device uchar *row, uint index, float value) {
        row[index] = (uchar)nk_rint_saturated_metal_(value, -128.0f, 127.0f);
    }
    static uint squares(uint bits) { return (uint)((int)(char)bits * (int)(char)bits); }
};

struct u8_t {
    using raw_t = uchar;
    using dot_result_t = uint;
    using norm_t = uint;
    static constant constexpr uint dimensions_per_value = 1;
    static constant constexpr uint chunk_values = 16;
    static uint load(device uchar const *row, uint index) { return row[index]; }
    static void store(device uchar *row, uint index, float value) {
        row[index] = (uchar)nk_rint_saturated_metal_(value, 0.0f, 255.0f);
    }
    static uint squares(uint bits) { return bits * bits; }
};

struct i4x2_t {
    using raw_t = uchar;
    using dot_result_t = int;
    using norm_t = uint;
    static constant constexpr uint dimensions_per_value = 2;
    static constant constexpr uint chunk_values = 32;
    static int load(device uchar const *row, uint index) { return ((int)nk_b4_load_(row, index) ^ 8) - 8; }
    static uint squares(uint bits) {
        int const low = ((int)(bits & 15) ^ 8) - 8, high = ((int)((bits >> 4) & 15) ^ 8) - 8;
        return (uint)(low * low + high * high);
    }
};

struct u4x2_t {
    using raw_t = uchar;
    using dot_result_t = uint;
    using norm_t = uint;
    static constant constexpr uint dimensions_per_value = 2;
    static constant constexpr uint chunk_values = 32;
    static uint load(device uchar const *row, uint index) { return nk_b4_load_(row, index); }
    static uint squares(uint bits) { return (bits & 15) * (bits & 15) + ((bits >> 4) & 15) * ((bits >> 4) & 15); }
};

struct f16_t {
    using raw_t = half;
    using stage_t = half;
    using dot_result_t = float;
    using norm_t = float;
    static constant constexpr uint dimensions_per_value = 1;
    static constant constexpr uint chunk_values = 8;
    static half4 widen(ushort4 codes) { return as_type<half4>(codes); }
    static float load(device uchar const *row, uint index) {
        device half const *elements = (device half const *)row;
        return (float)elements[index];
    }
    static void store(device uchar *row, uint index, float value) {
        uint const bits = as_type<uint>(value), payload = (bits & 0x7FFFFFu) >> 13;
        ((device ushort *)row)[index] = (bits & 0x7FFFFFFFu) > 0x7F800000u
                                            ? (ushort)(((bits >> 16) & 0x8000u) | 0x7C00u | max(payload, 1u))
                                            : as_type<ushort>((half)value);
    }
    static float squares(uint bits) {
        float const value = (float)as_type<half>((ushort)bits);
        return value * value;
    }
};

struct bf16_t {
    using raw_t = bfloat;
    using stage_t = bfloat;
    using dot_result_t = float;
    using norm_t = float;
    static constant constexpr uint dimensions_per_value = 1;
    static constant constexpr uint chunk_values = 8;
    static bfloat4 widen(ushort4 codes) { return as_type<bfloat4>(codes); }
    static float load(device uchar const *row, uint index) {
        device bfloat const *elements = (device bfloat const *)row;
        return (float)elements[index];
    }
    static void store(device uchar *row, uint index, float value) {
        uint const bits = as_type<uint>(value);
        // A NaN keeps its sign and top payload bits, as rounding could carry it into infinity
        ((device ushort *)row)[index] = (ushort)((bits & 0x7FFFFFFFu) > 0x7F800000u
                                                     ? (bits >> 16) | 0x40u
                                                     : (bits + 0x7FFFu + ((bits >> 16) & 1)) >> 16);
    }
    static float squares(uint bits) {
        float const value = as_type<float>(bits << 16);
        return value * value;
    }
};

/** E4M3FN: bias 7, and the two all-ones codes are NaN rather than infinities. */
struct e4m3_t {
    using raw_t = uchar;
    using stage_t = half;
    using dot_result_t = float;
    using norm_t = float;
    static constant constexpr uint dimensions_per_value = 1;
    static constant constexpr uint chunk_values = 16;
    static half4 widen(ushort4 codes) {
        ushort4 const bits = ((codes & ushort(0x7F)) << ushort(7)) | ((codes & ushort(0x80)) << ushort(8));
        return select(as_type<half4>(bits) * half(256), half4(NAN), (codes & ushort(0x7F)) == ushort(0x7F));
    }
    static float load(device uchar const *row, uint index) { return nk_widened_load_<e4m3_t>(row, index); }
    static void store(device uchar *row, uint index, float value) {
        row[index] = nk_minifloat_encode_metal_<3, 7, 7, 0x7E, 0x7F>(value);
    }
    static float squares(uint bits) { return nk_widened_squares_<e4m3_t>(bits); }
};

/** E5M2 is the top byte of an IEEE half, infinities and NaNs included. */
struct e5m2_t {
    using raw_t = uchar;
    using stage_t = half;
    using dot_result_t = float;
    using norm_t = float;
    static constant constexpr uint dimensions_per_value = 1;
    static constant constexpr uint chunk_values = 16;
    static half4 widen(ushort4 codes) { return as_type<half4>(codes << ushort(8)); }
    static float load(device uchar const *row, uint index) { return nk_widened_load_<e5m2_t>(row, index); }
    static void store(device uchar *row, uint index, float value) {
        row[index] = nk_minifloat_encode_metal_<2, 15, 7, 0x7C, 0x7D>(value);
    }
    static float squares(uint bits) { return nk_widened_squares_<e5m2_t>(bits); }
};

/** E3M2, OCP FP6 in the low six bits of a byte: bias 3, no infinities or NaNs. */
struct e3m2_t {
    using raw_t = uchar;
    using stage_t = half;
    using dot_result_t = float;
    using norm_t = float;
    static constant constexpr uint dimensions_per_value = 1;
    static constant constexpr uint chunk_values = 16;
    static half4 widen(ushort4 codes) {
        ushort4 const bits = ((codes & ushort(0x1F)) << ushort(8)) | ((codes & ushort(0x20)) << ushort(10));
        return as_type<half4>(bits) * half(4096);
    }
    static float load(device uchar const *row, uint index) { return nk_widened_load_<e3m2_t>(row, index); }
    static void store(device uchar *row, uint index, float value) {
        row[index] = nk_minifloat_encode_metal_<2, 3, 5, 0x1F, 0x1F>(value);
    }
    static float squares(uint bits) { return nk_widened_squares_<e3m2_t>(bits); }
};

/** E2M3, OCP FP6 in the low six bits of a byte: bias 1, no infinities or NaNs. */
struct e2m3_t {
    using raw_t = uchar;
    using stage_t = half;
    using dot_result_t = float;
    using norm_t = float;
    static constant constexpr uint dimensions_per_value = 1;
    static constant constexpr uint chunk_values = 16;
    static half4 widen(ushort4 codes) {
        ushort4 const bits = ((codes & ushort(0x1F)) << ushort(7)) | ((codes & ushort(0x20)) << ushort(10));
        return as_type<half4>(bits) * half(16384);
    }
    static float load(device uchar const *row, uint index) { return nk_widened_load_<e2m3_t>(row, index); }
    static void store(device uchar *row, uint index, float value) {
        row[index] = nk_minifloat_encode_metal_<3, 1, 5, 0x1F, 0x1F>(value);
    }
    static float squares(uint bits) { return nk_widened_squares_<e2m3_t>(bits); }
};

/** E2M1, OCP FP4, two to a byte with the high nibble first: bias 1, eight magnitudes up to 6. Its
 *  @c store rewrites the whole byte, so one thread writes both nibbles of a byte. */
struct e2m1x2_t {
    using raw_t = uchar;
    using stage_t = half;
    using dot_result_t = float;
    using norm_t = float;
    static constant constexpr uint dimensions_per_value = 2;
    static constant constexpr uint chunk_values = 32;
    static half4 widen(ushort4 codes) {
        ushort4 const bits = ((codes & ushort(0x7)) << ushort(9)) | ((codes & ushort(0x8)) << ushort(12));
        return as_type<half4>(bits) * half(16384);
    }
    static float load(device uchar const *row, uint index) { return nk_widened_load_<e2m1x2_t>(row, index); }
    static void store(device uchar *row, uint index, float value) {
        uint const code = nk_minifloat_encode_metal_<1, 1, 3, 7, 7>(value), shift = (~index & 1) * 4;
        row[index >> 1] = (uchar)((row[index >> 1] & ~(15u << shift)) | (code << shift));
    }
    static float squares(uint bits) { return nk_widened_squares_<e2m1x2_t>(bits); }
};

} // namespace nk

/** A compensated F32 pair scaled by 2 to the power of @c exponent, which block products of any
 *  scales add into, rounding once at the end. */
struct nk_cross_scaled_sum_metal_t {
    float2 sum;
    int exponent;
};

/** Splits @p value into a mantissa and its power of two, subnormals included; zeros, infinities and
 *  NaNs return themselves at exponent zero. */
float nk_cross_float_mantissa_metal_(float value, thread int &exponent) {
    uint const bits = as_type<uint>(value), fraction = bits & 0x7fffff, biased = (bits >> 23) & 255;
    if (biased == 255 || (biased == 0 && fraction == 0)) {
        exponent = 0;
        return value;
    }
    if (biased) {
        exponent = int(biased) - 127;
        return as_type<float>((bits & 0x807fffff) | 0x3f800000);
    }
    int shift;
    float const mantissa = frexp(float(fraction), shift);
    exponent = shift - 149;
    return bits & 0x80000000 ? -mantissa : mantissa;
}

/** Adds @p value times 2 to the power of @p exponent into @p state, keeping what the sum sheds in
 *  its low half; a non-finite term or sum leaves a plain F32 sum. */
void nk_cross_scaled_add_metal_(thread nk_cross_scaled_sum_metal_t &state, float value, int exponent) {
#pragma clang fp contract(off) reassociate(off)
    if (!isfinite(value) || !isfinite(state.sum.x)) {
        state.sum = float2(state.sum.x + value, 0);
        state.exponent = 0;
        return;
    }
    if (value == 0) return;
    int shift;
    value = frexp(value, shift);
    exponent += shift;
    if (all(state.sum == float2(0))) {
        state.sum = float2(value, 0);
        state.exponent = exponent;
        return;
    }
    int const common = max(state.exponent, exponent);
    float const high = ldexp(state.sum.x, state.exponent - common);
    float const low = ldexp(state.sum.y, state.exponent - common);
    value = ldexp(value, exponent - common);
    float const sum = high + value, split = sum - high;
    float const error = (high - (sum - split)) + (value - split) + low;
    float const rounded = sum + error;
    state.sum = float2(rounded, error - (rounded - sum));
    state.exponent = common;
}

/** The bits of the F32 nearest @p sum times 2 to the power of @p exponent, rounded once, subnormals
 *  included. */
uint nk_cross_scaled_bits_metal_(float2 sum, int exponent) {
#pragma clang fp contract(off) reassociate(off)
    if (!isfinite(sum.x) || all(sum == float2(0))) return as_type<uint>(sum.x);
    uint const sign = as_type<uint>(sum.x) & 0x80000000;
    if (sign) sum = -sum;
    int shift;
    float const high = frexp(sum.x, shift);
    sum = float2(high, ldexp(sum.y, -shift));
    exponent += shift;
    if (exponent >= -125) return as_type<uint>(ldexp(sum.x + sum.y, exponent)) | sign;
    if (exponent < -149) return sign;
    float const scaled = ldexp(sum.x, exponent + 149), error = ldexp(sum.y, exponent + 149);
    float rounded = rint(scaled);
    if (abs(scaled - rounded) == 0.5f && error != 0) rounded = error > 0 ? ceil(scaled) : floor(scaled);
    return sign | uint(rounded);
}

/** @p state rounded to F32, times the F32 whose bits are @p tensor_product, rounded again. */
float nk_cross_scaled_dot_metal_(nk_cross_scaled_sum_metal_t state, uint tensor_product) {
#pragma clang fp contract(off) reassociate(off)
    uint const dot_bits = nk_cross_scaled_bits_metal_(state.sum, state.exponent);
    if (tensor_product == 0x3f800000) return as_type<float>(dot_bits);
    int dot_exponent, tensor_exponent;
    float const dot = nk_cross_float_mantissa_metal_(as_type<float>(dot_bits), dot_exponent);
    float const tensor = nk_cross_float_mantissa_metal_(as_type<float>(tensor_product), tensor_exponent);
    float const product = dot * tensor;
    return as_type<float>(nk_cross_scaled_bits_metal_(
        float2(product, isfinite(product) ? fma(dot, tensor, -product) : 0), dot_exponent + tensor_exponent));
}

/** The block scales of the block-scaled formats: UE4M3 for NVFP4, UE8M0 for the MX formats. */
enum nk_cross_scale_metal_t { nk_cross_scale_e4m3_metal_k, nk_cross_scale_e8m0_metal_k };

/** The product of the block scale codes @p a and @p b as a mantissa and a power of two in
 *  @p exponent: UE4M3 values multiply, UE8M0 exponents add, 0xFF giving a NaN and 0x00 a zero. */
template <nk_cross_scale_metal_t scale_>
float nk_cross_block_scale_metal_(uchar a, uchar b, thread int &exponent) {
#pragma clang fp contract(off) reassociate(off)
    exponent = 0;
    // UE4M3 is unsigned, so the top bit is ignored as the serial decoders do
    if (scale_ == nk_cross_scale_e4m3_metal_k)
        return float(nk::e4m3_t::widen(ushort4(a & 0x7F))[0]) * float(nk::e4m3_t::widen(ushort4(b & 0x7F))[0]);
    exponent = int(a) + int(b) - 254;
    return a == 255 || b == 255 ? NAN : a == 0 || b == 0 ? 0.0f : 1.0f;
}

/** Adds a block's F32 @p dot times its two scales into @p state. */
template <nk_cross_scale_metal_t scale_>
void nk_cross_scaled_block_add_metal_(thread nk_cross_scaled_sum_metal_t &state, float dot, uchar a_scale,
                                      uchar b_scale) {
#pragma clang fp contract(off) reassociate(off)
    int exponent;
    float const scale = nk_cross_block_scale_metal_<scale_>(a_scale, b_scale, exponent);
    float const product = dot * scale;
    if (scale_ == nk_cross_scale_e4m3_metal_k) {
        // NVFP4 block products fit in 20 significant bits and cannot overflow F32.
        float const sum = state.sum.x + product, split = sum - state.sum.x;
        float const error = (state.sum.x - (sum - split)) + (product - split) + state.sum.y;
        float const rounded = sum + error;
        state.sum = isfinite(product) && isfinite(state.sum.x) ? float2(rounded, error - (rounded - sum))
                                                               : float2(state.sum.x + product, 0);
    }
    else nk_cross_scaled_add_metal_(state, product, exponent);
}
