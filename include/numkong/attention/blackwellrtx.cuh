/**
 *  @file include/numkong/attention/blackwellrtx.cuh
 *  @author Ash Vardanian
 *  @date September 22, 2026
 *  @brief Ragged attention for the NVIDIA compute capability 12.x family.
 *
 *  @sa include/numkong/attention.h
 *  @sa include/numkong/attention/ampere.cuh
 *
 *  The Ampere tile with E4M3 going to the tensor cores as it is: S takes one
 *  `mma.m16n8k32.kind::f8f6f4` per 16 × 8 scores, at twice the F16 rate, and P is quantized to
 *  e4m3(256 · p) for the same instruction against the transposed V codes. The × 256 keeps every
 *  weight down to 2⁻¹⁴ of the row maximum in E4M3's normal range, and the row sum adds the
 *  dequantized weights, so 256 cancels in the normalization. BF16 and I8 reuse Ampere's kernels.
 */
#ifndef NUMKONG_ATTENTION_BLACKWELLRTX_CUH
#define NUMKONG_ATTENTION_BLACKWELLRTX_CUH

#if NUMKONG_TARGET_BLACKWELLRTX

#include "numkong/attention/ampere.cuh"
#include "numkong/attention/ada.cuh"     // `nk_attention_weights_e4m3_ada_`
#include "numkong/dots/blackwellrtx.cuh" // `nk_mma_e4m3_blackwellrtx_`

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Fragments

NUMKONG_DEVICE void nk_attention_scores_e4m3_blackwellrtx_(nk_fui32_t scores[2][4], nk_u32_t const query[8],
                                                           nk_u32_t const keys[4]) {
    nk_mma_e4m3_blackwellrtx_(scores[0], query, keys[0], keys[1]);
    nk_mma_e4m3_blackwellrtx_(scores[1], query, keys[2], keys[3]);
}

#pragma endregion Fragments

#pragma region Instantiations

nk_define_attention_pack_size_simt_(e4m3, blackwellrtx, 1)
nk_define_attention_packed_shape_cuda_(e4m3, blackwellrtx)
nk_define_attention_pack_cuda_(e4m3, blackwellrtx, e4m3)
nk_define_attention_packed_cuda_(e4m3, blackwellrtx, ampere, nk_attention_launch_ampere_, e4m3, nk_cross_epilogue_f32_k,
                                 nk_attention_scores_e4m3_blackwellrtx_, nk_mma_e4m3_blackwellrtx_,
                                 nk_attention_weights_e4m3_ada_, 1.0f, 1.0f)

#pragma endregion Instantiations

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_BLACKWELLRTX
#endif // NUMKONG_ATTENTION_BLACKWELLRTX_CUH
