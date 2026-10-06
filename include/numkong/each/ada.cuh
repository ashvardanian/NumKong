/**
 *  @file include/numkong/each/ada.cuh
 *  @author Ash Vardanian
 *  @date October 6, 2026
 *  @brief Element-wise E4M3 scales, sums, blends, FMAs, SwiGLU and RMSNorm for NVIDIA Ada, compute
 *      capability 8.9, and every GPU since.
 *
 *  @sa include/numkong/each/cuda.cuh
 *  @sa include/numkong/cast/ada.cuh
 *
 *  The kernels of `each/simt.cuh` through the generators of `each/cuda.cuh`, widening E4M3 inputs
 *  and narrowing results to E4M3 in one instruction rather than in branches, RMSNorm reading E4M3
 *  or the F32 of the dot products, which it also narrows to E5M2.
 */
#ifndef NUMKONG_EACH_ADA_CUH
#define NUMKONG_EACH_ADA_CUH

#include "numkong/each/cuda.cuh" // `nk_define_each_sum_cuda_`, `nk_define_each_rmsnorm_cuda_`
#include "numkong/cast/ada.cuh"  // `nk_e4m3_to_f32_ada_`, `nk_f32_to_e4m3_ada_`, `nk_f32_to_e5m2_ada_`

#if NUMKONG_ARCH_CUDA_ADA_

#if defined(__cplusplus)
extern "C" {
#endif

#if NUMKONG_TARGET_ADA
nk_define_each_sum_cuda_(e4m3, f32, ada, nk_e4m3_to_f32_ada_, nk_f32_to_e4m3_ada_)
nk_define_each_scale_cuda_(e4m3, f32, ada, nk_e4m3_to_f32_ada_, nk_f32_to_e4m3_ada_)
nk_define_each_blend_cuda_(e4m3, f32, ada, nk_e4m3_to_f32_ada_, nk_f32_to_e4m3_ada_)
nk_define_each_fma_cuda_(e4m3, f32, ada, nk_e4m3_to_f32_ada_, nk_f32_to_e4m3_ada_)
nk_define_each_swiglu_cuda_(e4m3, ada, nk_e4m3_to_f32_ada_, nk_f32_to_e4m3_ada_)
nk_define_each_rmsnorm_cuda_(rmsnorm, e4m3, e4m3, ada, nk_e4m3_to_f32_ada_, nk_f32_to_e4m3_ada_)
nk_define_each_rmsnorm_cuda_(rmscast, f32, e4m3, ada, nk_assign_from_to_, nk_f32_to_e4m3_ada_)
nk_define_each_rmsnorm_cuda_(rmscast, f32, e5m2, ada, nk_assign_from_to_, nk_f32_to_e5m2_ada_)
#endif // NUMKONG_TARGET_ADA

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_ADA_
#endif // NUMKONG_EACH_ADA_CUH
