/**
 *  @file include/numkong/dots/ada.cuh
 *  @author Ash Vardanian
 *  @date September 26, 2026
 *  @brief Float6 relabelling for NVIDIA Ada, compute capability 8.9, and every GPU since.
 *
 *  The Float6 relabelling into Float8 that the Hopper, Blackwell and BlackwellRTX capabilities
 *  reuse, next to the `cvt.*e4m3x2` helpers of `cast/ada.cuh`. Ada runs no matrix kernels of its
 *  own, as its FP8 `mma.sync` keeps only 13 fraction bits of each 16-deep step; its element-wise
 *  kernels and bulk casts live in `each/ada.cuh` and `cast/ada.cuh`.
 */
#ifndef NUMKONG_DOTS_ADA_CUH
#define NUMKONG_DOTS_ADA_CUH

#if NUMKONG_ARCH_CUDA_
#if NUMKONG_ARCH_CUDA_ADA_

#include "numkong/cast/ada.cuh" // `nk_f32x2_to_e4m3x2_ada_`, `nk_e4m3x2_to_f16x2_ada_`

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Conversions

/** Four Float6 codes as Float8 ones: the five magnitude bits stay and the sign moves from bit 5 to
 *  bit 7, so E2M3 reads as E4M3 times 2⁻⁶ and E3M2 as E5M2 times 2⁻¹². */
NUMKONG_DEVICE nk_u32_t nk_f6x4_to_f8x4_ada_(nk_u32_t codes) {
    return (codes & 0x1F1F1F1Fu) | ((codes << 2) & 0x80808080u);
}

/** Packs one Float6 code as the Float8 code the tensor cores read. */
NUMKONG_DEVICE unsigned char nk_load_f6_to_f8_ada_(unsigned char code) {
    return (unsigned char)nk_f6x4_to_f8x4_ada_(code);
}

#pragma endregion Conversions

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_ADA_
#endif // NUMKONG_ARCH_CUDA_
#endif // NUMKONG_DOTS_ADA_CUH
