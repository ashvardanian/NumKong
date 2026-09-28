/**
 *  @file include/numkong/attention/ada.cuh
 *  @author Ash Vardanian
 *  @date September 27, 2026
 *  @brief E4M3 attention weights for NVIDIA Ada, compute capability 8.9, and every GPU since.
 *
 *  @sa include/numkong/attention.h
 *  @sa include/numkong/dots/ada.cuh
 *
 *  The lowest NVIDIA capability with `cvt.*e4m3x2`, whose weight quantization the Hopper, Blackwell
 *  and BlackwellRTX capabilities share. Ada runs no attention kernels of its own.
 */
#ifndef NUMKONG_ATTENTION_ADA_CUH
#define NUMKONG_ATTENTION_ADA_CUH

#if NUMKONG_ARCH_CUDA_ADA_

#include "numkong/dots/ada.cuh" // `nk_f32x2_to_e4m3x2_ada_`, `nk_e4m3x2_to_f16x2_ada_`

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Fragments

/** E4M3 weights e4m3(256 · p), summed as the tensor cores will read them. */
NUMKONG_DEVICE void nk_attention_weights_e4m3_ada_(nk_f32_t const probabilities[4], nk_u32_t packed[2], nk_f32_t *sum) {
    unsigned short const low = nk_f32x2_to_e4m3x2_ada_(probabilities[0] * 256.0f, probabilities[1] * 256.0f);
    unsigned short const high = nk_f32x2_to_e4m3x2_ada_(probabilities[2] * 256.0f, probabilities[3] * 256.0f);
    packed[0] = (nk_u32_t)low | ((nk_u32_t)high << 16), packed[1] = 0;
    nk_u32_t const low_halves = nk_e4m3x2_to_f16x2_ada_(low);
    nk_u32_t const high_halves = nk_e4m3x2_to_f16x2_ada_(high);
    *sum += (__half2float(__ushort_as_half((unsigned short)(low_halves & 0xFFFFu))) +
             __half2float(__ushort_as_half((unsigned short)(low_halves >> 16)))) +
            (__half2float(__ushort_as_half((unsigned short)(high_halves & 0xFFFFu))) +
             __half2float(__ushort_as_half((unsigned short)(high_halves >> 16))));
}

#pragma endregion Fragments

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_ADA_
#endif // NUMKONG_ATTENTION_ADA_CUH
