/**
 *  @file include/numkong/cast/ampere.cuh
 *  @author Ash Vardanian
 *  @date October 6, 2026
 *  @brief Conversions and bulk casts for NVIDIA Ampere, compute capability 8.0,
 *      and every GPU since.
 *
 *  @sa include/numkong/cast/cuda.cuh
 *  @sa include/numkong/cast/ada.cuh
 *
 *  The lowest NVIDIA capability with `cvt.*x2.f32`, whose helpers the later capabilities reuse. Its
 *  F32 to BF16 narrowing takes one instruction, with NaNs keeping the serial cast's sign and top
 *  payload bits, and the @c ampere bulk cast runs it in 16-byte chunks; every other cast takes the
 *  portable code of `cast/simt.cuh`.
 */
#ifndef NUMKONG_CAST_AMPERE_CUH
#define NUMKONG_CAST_AMPERE_CUH

#include "numkong/cast/cuda.cuh" // `nk_cast_launch_cuda_`, `nk_cast_vectors_simt_`

#if NUMKONG_ARCH_CUDA_AMPERE_

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Instructions

/** Rounds two F32 into a BF16 pair, @p low in the low half. */
NUMKONG_DEVICE nk_u32_t nk_f32x2_to_bf16x2_ampere_(nk_f32_t low, nk_f32_t high) {
    nk_u32_t pair;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(pair) : "f"(high), "f"(low));
    return pair;
}

/** Rounds two F32 into an F16 pair, @p low in the low half. */
NUMKONG_DEVICE nk_u32_t nk_f32x2_to_f16x2_ampere_(nk_f32_t low, nk_f32_t high) {
    nk_u32_t pair;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(pair) : "f"(high), "f"(low));
    return pair;
}

#pragma endregion Instructions

#pragma region Conversions

/** Narrows one F32 value to BF16 like @c nk_f32_to_bf16_simt_, in one instruction. */
NUMKONG_DEVICE void nk_f32_to_bf16_ampere_(nk_f32_t const *src, nk_bf16_t *dest) {
    nk_u32_t const bits = __float_as_uint(*src);
    // The instruction writes one canonical NaN; the serial cast keeps the sign and top payload bits
    *(unsigned short *)dest = (bits & 0x7FFFFFFFu) > 0x7F800000u
                                  ? (unsigned short)((bits >> 16) | 0x0040u)
                                  : (unsigned short)nk_f32x2_to_bf16x2_ampere_(*src, 0.0f);
}

/** Narrows the 8 F32 values at @p from into the 16 bytes of BF16 at @p to, like
 *  @c nk_cast_f32x8_to_bf16x8_simt_. */
NUMKONG_DEVICE void nk_cast_f32x8_to_bf16x8_ampere_(uint4 const *from, uint4 *to) {
    uint4 const loaded[2] = {from[0], from[1]};
    uint4 narrowed;
    nk_b128_vec_t const *const values = (nk_b128_vec_t const *)loaded;
    nk_bf16_t *const halves = ((nk_b128_vec_t *)&narrowed)->bf16s;
#pragma unroll
    for (unsigned offset = 0; offset != 8; ++offset)
        nk_f32_to_bf16_ampere_(values[offset / 4].f32s + offset % 4, halves + offset);
    *to = narrowed;
}

#pragma endregion Conversions

#if NUMKONG_TARGET_AMPERE

static __global__ void nk_cast_ampere_kernel_(nk_cast_arguments_t arguments) {
    nk_size_t const stride = (nk_size_t)gridDim.x * blockDim.x;
    for (nk_size_t unit = (nk_size_t)blockIdx.x * blockDim.x + threadIdx.x; unit < arguments.units; unit += stride)
        nk_cast_unit_simt_(&arguments, unit);
}

static __global__ void nk_cast_vectors_ampere_kernel_(nk_cast_arguments_t arguments) {
    nk_size_t const threads = (nk_size_t)gridDim.x * blockDim.x;
    nk_size_t const first = (nk_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    nk_size_t const chunks = arguments.count / arguments.unit_values;
    if (arguments.from_dtype != nk_f32_k || arguments.to_dtype != nk_bf16_k) {
        nk_cast_vectors_simt_(&arguments);
        return;
    }
    for (nk_size_t chunk = first; chunk < chunks; chunk += threads)
        nk_cast_f32x8_to_bf16x8_ampere_((uint4 const *)arguments.from + chunk * 2, (uint4 *)arguments.to + chunk);
    nk_cast_vectors_tail_simt_(&arguments);
}

static __global__ void nk_cast_block_scaled_ampere_kernel_(nk_cast_block_scaled_arguments_t arguments) {
    __shared__ nk_u32_t partials[256];
    nk_cast_block_scaled_phase_simt_(&arguments, partials);
}

NUMKONG_API nk_status_t nk_cast_ampere(void const *from, nk_dtype_t from_dtype, void *to, nk_dtype_t to_dtype,
                                       nk_size_t count, void *stream) {
    return nk_cast_launch_cuda_((void const *)&nk_cast_ampere_kernel_, (void const *)&nk_cast_vectors_ampere_kernel_,
                                (void const *)&nk_cast_block_scaled_ampere_kernel_, from, from_dtype, to, to_dtype,
                                count, stream);
}

#endif // NUMKONG_TARGET_AMPERE

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_AMPERE_
#endif // NUMKONG_CAST_AMPERE_CUH
