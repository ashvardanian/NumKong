/**
 *  @brief Ragged attention for the NVIDIA compute capability 12.x family.
 *  @file include/numkong/attention/blackwellrtx.cuh
 *  @author Ash Vardanian
 *  @date September 22, 2026
 *
 *  @sa include/numkong/attention.h
 *  @sa include/numkong/attention/ampere.cuh
 *
 *  The Ampere tile with E4M3 going to the tensor cores as it is: S takes one `mma.m16n8k32.kind::f8f6f4` per 16 × 8
 *  scores, at twice the F16 rate, and P is quantized to `e4m3(256 · p)` for the same instruction against the transposed
 *  V codes. The ×256 keeps every weight down to 2⁻¹⁴ of the row maximum in E4M3's normal range, and the row sum adds
 *  the dequantized weights, so the 256 cancels in the normalization. BF16 and I8 use the Ampere kernels.
 */
#ifndef NK_ATTENTION_BLACKWELLRTX_CUH
#define NK_ATTENTION_BLACKWELLRTX_CUH

#if NK_TARGET_BLACKWELLRTX

#include "numkong/attention/ampere.cuh"
#include "numkong/dots/blackwellrtx.cuh" // `nk_mma_e4m3_blackwellrtx_`

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Instructions

#if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1200 && \
    __CUDA_ARCH_FAMILY_SPECIFIC__ < 1300

/* Rounds two F32 into an E4M3 pair, saturating at 448, @p low in the low byte. */
NK_HELPER_DEVICE_INLINE unsigned short nk_f32x2_to_e4m3x2_blackwellrtx_(nk_f32_t low, nk_f32_t high) {
    unsigned short pair;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(pair) : "f"(high), "f"(low));
    return pair;
}

/* Widens an E4M3 pair into an F16 pair, the low byte into the low half. */
NK_HELPER_DEVICE_INLINE nk_u32_t nk_e4m3x2_to_f16x2_blackwellrtx_(unsigned short pair) {
    nk_u32_t halves;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(halves) : "h"(pair));
    return halves;
}

#else

NK_HELPER_DEVICE_INLINE unsigned short nk_f32x2_to_e4m3x2_blackwellrtx_(nk_f32_t low, nk_f32_t high) {
    __trap();
    return 0;
}
NK_HELPER_DEVICE_INLINE nk_u32_t nk_e4m3x2_to_f16x2_blackwellrtx_(unsigned short pair) {
    __trap();
    return 0;
}

#endif // __CUDA_ARCH_FAMILY_SPECIFIC__ in the 12.x family

#pragma endregion Instructions

#pragma region Fragments

NK_HELPER_DEVICE_INLINE void nk_attention_scores_e4m3_blackwellrtx_(nk_fui32_t scores[2][4], nk_u32_t const query[8],
                                                                    nk_u32_t const keys[4]) {
    nk_mma_e4m3_blackwellrtx_(scores[0], query, keys[0], keys[1]);
    nk_mma_e4m3_blackwellrtx_(scores[1], query, keys[2], keys[3]);
}

/** E4M3 weights `e4m3(256 · p)`, summed as the tensor cores will read them. */
NK_HELPER_DEVICE_INLINE void nk_attention_weights_e4m3_blackwellrtx_(nk_f32_t const probabilities[4],
                                                                     nk_u32_t packed[2], nk_f32_t *sum) {
    unsigned short const low = nk_f32x2_to_e4m3x2_blackwellrtx_(probabilities[0] * 256.0f, probabilities[1] * 256.0f);
    unsigned short const high = nk_f32x2_to_e4m3x2_blackwellrtx_(probabilities[2] * 256.0f, probabilities[3] * 256.0f);
    packed[0] = (nk_u32_t)low | ((nk_u32_t)high << 16), packed[1] = 0;
    nk_u32_t const low_halves = nk_e4m3x2_to_f16x2_blackwellrtx_(low);
    nk_u32_t const high_halves = nk_e4m3x2_to_f16x2_blackwellrtx_(high);
    *sum += (__half2float(__ushort_as_half((unsigned short)(low_halves & 0xFFFFu))) +
             __half2float(__ushort_as_half((unsigned short)(low_halves >> 16)))) +
            (__half2float(__ushort_as_half((unsigned short)(high_halves & 0xFFFFu))) +
             __half2float(__ushort_as_half((unsigned short)(high_halves >> 16))));
}

#pragma endregion Fragments

#pragma region Instantiations

nk_define_attention_cuda_pack_size_(e4m3, blackwellrtx, 1)
nk_define_attention_cuda_packed_shape_(e4m3, blackwellrtx)
nk_define_attention_cuda_pack_(e4m3, blackwellrtx, e4m3, nk_attention_kind_bytes_k)
nk_define_attention_cuda_packed_(e4m3, blackwellrtx, e4m3, nk_attention_kind_bytes_k, nk_cross_epilogue_f32_k,
                                 nk_attention_scores_e4m3_blackwellrtx_, nk_mma_e4m3_blackwellrtx_,
                                 nk_attention_weights_e4m3_blackwellrtx_, 1.0f, 1.0f, nk_e4m3_k)

#pragma endregion Instantiations

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_BLACKWELLRTX
#endif // NK_ATTENTION_BLACKWELLRTX_CUH
