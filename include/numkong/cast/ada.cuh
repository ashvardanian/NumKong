/**
 *  @file include/numkong/cast/ada.cuh
 *  @author Ash Vardanian
 *  @date October 6, 2026
 *  @brief Float8 conversions and bulk casts for NVIDIA Ada, compute capability 8.9,
 *      and every GPU since.
 *
 *  @sa include/numkong/cast/ampere.cuh
 *  @sa include/numkong/dots/ada.cuh
 *
 *  The lowest NVIDIA capability with `cvt.*e4m3x2` and `cvt.*e5m2x2`, whose helpers the later
 *  capabilities reuse. Its E4M3 conversions and E5M2 narrowing take one instruction for two values,
 *  NaNs and E5M2 overflows patched to the serial casts' codes, and the @c ada bulk cast runs the
 *  E4M3 ones, and Ampere's BF16 narrowing, in 16-byte chunks.
 */
#ifndef NUMKONG_CAST_ADA_CUH
#define NUMKONG_CAST_ADA_CUH

#if NUMKONG_ARCH_CUDA_
#if NUMKONG_ARCH_CUDA_ADA_

#include "numkong/cast/ampere.cuh" // `nk_cast_f32x8_to_bf16x8_ampere_`, `nk_cast_launch_cuda_`

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Instructions

/** Rounds two F32 into an E4M3 pair, saturating at 448, @p low in the low byte. */
NUMKONG_DEVICE unsigned short nk_f32x2_to_e4m3x2_ada_(nk_f32_t low, nk_f32_t high) {
    unsigned short pair;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(pair) : "f"(high), "f"(low));
    return pair;
}

/** Widens an E4M3 pair into an F16 pair, the low byte into the low half. */
NUMKONG_DEVICE nk_u32_t nk_e4m3x2_to_f16x2_ada_(unsigned short pair) {
    nk_u32_t halves;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(halves) : "h"(pair));
    return halves;
}

/** Rounds two F32 into an E5M2 pair, saturating at 57344, @p low in the low byte. */
NUMKONG_DEVICE unsigned short nk_f32x2_to_e5m2x2_ada_(nk_f32_t low, nk_f32_t high) {
    unsigned short pair;
    asm("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;\n" : "=h"(pair) : "f"(high), "f"(low));
    return pair;
}

#pragma endregion Instructions

#pragma region Conversions

/** Widens one E4M3FN value to F32 like @c nk_e4m3_to_f32_simt_, through one instruction. */
NUMKONG_DEVICE void nk_e4m3_to_f32_ada_(nk_e4m3_t const *src, nk_f32_t *dest) {
    nk_u32_t const code = *src;
    nk_u32_t const halves = nk_e4m3x2_to_f16x2_ada_((unsigned short)code);
    // The instruction widens both NaN codes to one canonical NaN; the serial cast keeps the sign
    *dest = (code & 0x7Fu) == 0x7Fu ? __uint_as_float(((code & 0x80u) << 24) | 0x7FC00000u)
                                    : __half2float(__ushort_as_half((unsigned short)halves));
}

/** Narrows one F32 value to E4M3FN like @c nk_f32_to_e4m3_simt_, in one instruction. */
NUMKONG_DEVICE void nk_f32_to_e4m3_ada_(nk_f32_t const *src, nk_e4m3_t *dest) {
    // The instruction drops only the sign of a NaN, which every other result carries already
    *dest = (nk_e4m3_t)(nk_f32x2_to_e4m3x2_ada_(*src, 0.0f) | ((__float_as_uint(*src) >> 24) & 0x80u));
}

/** Narrows one F32 value to E5M2 like @c nk_f32_to_e5m2_simt_, in one instruction. */
NUMKONG_DEVICE void nk_f32_to_e5m2_ada_(nk_f32_t const *src, nk_e5m2_t *dest) {
    nk_u32_t const bits = __float_as_uint(*src), magnitude = bits & 0x7FFFFFFFu, sign = (bits >> 24) & 0x80u;
    // Saturates from 61440, where the serial cast rounds to infinity, and drops the signs of NaNs
    *dest = (nk_e5m2_t)(magnitude > 0x7F800000u    ? sign | 0x7Du
                        : magnitude >= 0x47700000u ? sign | 0x7Cu
                                                   : nk_f32x2_to_e5m2x2_ada_(*src, 0.0f));
}

/** Narrows the 16 F32 values at @p from into the 16 bytes of E4M3FN at @p to, like
 *  @c nk_cast_f32x16_to_e4m3x16_simt_. */
NUMKONG_DEVICE void nk_cast_f32x16_to_e4m3x16_ada_(uint4 const *from, uint4 *to) {
    uint4 const loaded[4] = {from[0], from[1], from[2], from[3]};
    uint4 narrowed;
    nk_b128_vec_t const *const values = (nk_b128_vec_t const *)loaded;
    nk_e4m3_t *const codes = ((nk_b128_vec_t *)&narrowed)->e4m3s;
#pragma unroll
    for (unsigned offset = 0; offset != 16; ++offset)
        nk_f32_to_e4m3_ada_(values[offset / 4].f32s + offset % 4, codes + offset);
    *to = narrowed;
}

/** Widens the 16 bytes of E4M3FN at @p from into 16 F32 values at @p to, like
 *  @c nk_cast_e4m3x16_to_f32x16_simt_. */
NUMKONG_DEVICE void nk_cast_e4m3x16_to_f32x16_ada_(uint4 const *from, uint4 *to) {
    uint4 const loaded = *from;
    uint4 widened[4];
    nk_e4m3_t const *const codes = ((nk_b128_vec_t const *)&loaded)->e4m3s;
    nk_b128_vec_t *const values = (nk_b128_vec_t *)widened;
#pragma unroll
    for (unsigned offset = 0; offset != 16; ++offset)
        nk_e4m3_to_f32_ada_(codes + offset, values[offset / 4].f32s + offset % 4);
#pragma unroll
    for (unsigned quad = 0; quad != 4; ++quad) to[quad] = widened[quad];
}

#pragma endregion Conversions

#if NUMKONG_TARGET_ADA

static __global__ void nk_cast_ada_kernel_(nk_cast_arguments_t arguments) {
    nk_size_t const stride = (nk_size_t)gridDim.x * blockDim.x;
    for (nk_size_t unit = (nk_size_t)blockIdx.x * blockDim.x + threadIdx.x; unit < arguments.units; unit += stride)
        nk_cast_unit_simt_(&arguments, unit);
}

static __global__ void nk_cast_vectors_ada_kernel_(nk_cast_arguments_t arguments) {
    nk_size_t const threads = (nk_size_t)gridDim.x * blockDim.x;
    nk_size_t const first = (nk_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    nk_size_t const chunks = arguments.count / arguments.unit_values;
    uint4 const *from = (uint4 const *)arguments.from;
    uint4 *to = (uint4 *)arguments.to;
    if (arguments.from_dtype == nk_bf16_k) {
        nk_cast_vectors_simt_(&arguments);
        return;
    }
    if (arguments.from_dtype == nk_e4m3_k)
        for (nk_size_t chunk = first; chunk < chunks; chunk += threads)
            nk_cast_e4m3x16_to_f32x16_ada_(from + chunk, to + chunk * 4);
    else if (arguments.to_dtype == nk_bf16_k)
        for (nk_size_t chunk = first; chunk < chunks; chunk += threads)
            nk_cast_f32x8_to_bf16x8_ampere_(from + chunk * 2, to + chunk);
    else
        for (nk_size_t chunk = first; chunk < chunks; chunk += threads)
            nk_cast_f32x16_to_e4m3x16_ada_(from + chunk * 4, to + chunk);
    nk_cast_vectors_tail_simt_(&arguments);
}

static __global__ void nk_cast_block_scaled_ada_kernel_(nk_cast_block_scaled_arguments_t arguments) {
    __shared__ nk_u32_t partials[256];
    nk_cast_block_scaled_phase_simt_(&arguments, partials);
}

NUMKONG_API nk_status_t nk_cast_ada(void const *from, nk_dtype_t from_dtype, void *to, nk_dtype_t to_dtype,
                                    nk_size_t count, void *stream) {
    return nk_cast_launch_cuda_((void const *)&nk_cast_ada_kernel_, (void const *)&nk_cast_vectors_ada_kernel_,
                                (void const *)&nk_cast_block_scaled_ada_kernel_, from, from_dtype, to, to_dtype, count,
                                stream);
}

#endif // NUMKONG_TARGET_ADA

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_ADA_
#endif // NUMKONG_ARCH_CUDA_
#endif // NUMKONG_CAST_ADA_CUH
