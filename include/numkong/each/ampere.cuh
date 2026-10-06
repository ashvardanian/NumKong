/**
 *  @file include/numkong/each/ampere.cuh
 *  @author Ash Vardanian
 *  @date October 6, 2026
 *  @brief Element-wise BF16 scales, sums, blends, FMAs, SwiGLU and RMSNorm for NVIDIA Ampere,
 *      compute capability 8.0, and every GPU since.
 *
 *  @sa include/numkong/each/cuda.cuh
 *  @sa include/numkong/cast/ampere.cuh
 *
 *  The kernels of `each/simt.cuh` through the generators of `each/cuda.cuh`, narrowing every result
 *  to BF16 in one instruction rather than with integer rounding, RMSNorm reading BF16 or the F32 of
 *  the dot products.
 */
#ifndef NUMKONG_EACH_AMPERE_CUH
#define NUMKONG_EACH_AMPERE_CUH

#include "numkong/each/cuda.cuh"   // `nk_define_each_sum_cuda_`, `nk_define_each_rmsnorm_cuda_`
#include "numkong/cast/ampere.cuh" // `nk_f32_to_bf16_ampere_`

#if NUMKONG_ARCH_CUDA_AMPERE_

#if defined(__cplusplus)
extern "C" {
#endif

#if NUMKONG_TARGET_AMPERE
nk_define_each_sum_cuda_(bf16, f32, ampere, nk_bf16_to_f32_simt_, nk_f32_to_bf16_ampere_)
nk_define_each_scale_cuda_(bf16, f32, ampere, nk_bf16_to_f32_simt_, nk_f32_to_bf16_ampere_)
nk_define_each_blend_cuda_(bf16, f32, ampere, nk_bf16_to_f32_simt_, nk_f32_to_bf16_ampere_)
nk_define_each_fma_cuda_(bf16, f32, ampere, nk_bf16_to_f32_simt_, nk_f32_to_bf16_ampere_)
nk_define_each_swiglu_cuda_(bf16, ampere, nk_bf16_to_f32_simt_, nk_f32_to_bf16_ampere_)
nk_define_each_rmsnorm_cuda_(rmsnorm, bf16, bf16, ampere, nk_bf16_to_f32_simt_, nk_f32_to_bf16_ampere_)
nk_define_each_rmsnorm_cuda_(rmscast, f32, bf16, ampere, nk_assign_from_to_, nk_f32_to_bf16_ampere_)
#endif // NUMKONG_TARGET_AMPERE

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_AMPERE_
#endif // NUMKONG_EACH_AMPERE_CUH
