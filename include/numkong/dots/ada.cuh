/**
 *  @file include/numkong/dots/ada.cuh
 *  @author Ash Vardanian
 *  @date September 26, 2026
 *  @brief Float8 conversions for NVIDIA Ada, compute capability 8.9, and every GPU since.
 *
 *  The lowest NVIDIA capability with `cvt.*e4m3x2`, whose helpers the Hopper, Blackwell and
 *  BlackwellRTX capabilities reuse, next to the Float6 relabelling into Float8. Ada runs no kernels
 *  of its own, as its FP8 `mma.sync` keeps only 13 fraction bits of each 16-deep step.
 */
#ifndef NUMKONG_DOTS_ADA_CUH
#define NUMKONG_DOTS_ADA_CUH

#if NUMKONG_ARCH_CUDA_ADA_

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Instructions

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890

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

#else

/*  Device passes older than 8.9 and the host pass get trapping bodies, so a cubin picked for the
 *  wrong device fails loudly rather than returning zeros. */
NUMKONG_DEVICE unsigned short nk_f32x2_to_e4m3x2_ada_(nk_f32_t low, nk_f32_t high) {
    __trap();
    return 0;
}
NUMKONG_DEVICE nk_u32_t nk_e4m3x2_to_f16x2_ada_(unsigned short pair) {
    __trap();
    return 0;
}

#endif // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890

#pragma endregion Instructions

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
#endif // NUMKONG_DOTS_ADA_CUH
