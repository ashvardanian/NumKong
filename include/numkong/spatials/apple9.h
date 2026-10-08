/**
 *  @file include/numkong/spatials/apple9.h
 *  @author Ash Vardanian
 *  @date October 5, 2026
 *  @brief Batched spatial distances on Apple family 9 GPUs, launched from C.
 *
 *  @sa include/numkong/spatials.h
 */
#ifndef NUMKONG_SPATIALS_APPLE9_H
#define NUMKONG_SPATIALS_APPLE9_H

#if NUMKONG_ARCH_METAL_
#if NUMKONG_TARGET_APPLE9
#include "numkong/dots/apple9.h"
#include "numkong/spatials/metal.h"

#if defined(__cplusplus)
extern "C" {
#endif

nk_define_spatials_metal_(angular, f16, apple9, f16, 16, 1, NUMKONG_METAL_LANGUAGE_3_1_, 16)
nk_define_spatials_metal_(angular, bf16, apple9, bf16, 16, 1, NUMKONG_METAL_LANGUAGE_3_1_, 16)
nk_define_spatials_metal_(angular, e4m3, apple9, e4m3, 16, 1, NUMKONG_METAL_LANGUAGE_3_1_, 16)
nk_define_spatials_metal_(angular, e5m2, apple9, e5m2, 16, 1, NUMKONG_METAL_LANGUAGE_3_1_, 16)
nk_define_spatials_metal_(angular, e3m2, apple9, e3m2, 16, 1, NUMKONG_METAL_LANGUAGE_3_1_, 16)
nk_define_spatials_metal_(angular, e2m3, apple9, e2m3, 16, 1, NUMKONG_METAL_LANGUAGE_3_1_, 16)
nk_define_spatials_metal_(angular, e2m1, apple9, e2m1x2, 32, 2, NUMKONG_METAL_LANGUAGE_3_1_, 16)
nk_define_spatials_metal_(euclidean, f16, apple9, f16, 16, 1, NUMKONG_METAL_LANGUAGE_3_1_, 16)
nk_define_spatials_metal_(euclidean, bf16, apple9, bf16, 16, 1, NUMKONG_METAL_LANGUAGE_3_1_, 16)
nk_define_spatials_metal_(euclidean, e4m3, apple9, e4m3, 16, 1, NUMKONG_METAL_LANGUAGE_3_1_, 16)
nk_define_spatials_metal_(euclidean, e5m2, apple9, e5m2, 16, 1, NUMKONG_METAL_LANGUAGE_3_1_, 16)
nk_define_spatials_metal_(euclidean, e3m2, apple9, e3m2, 16, 1, NUMKONG_METAL_LANGUAGE_3_1_, 16)
nk_define_spatials_metal_(euclidean, e2m3, apple9, e2m3, 16, 1, NUMKONG_METAL_LANGUAGE_3_1_, 16)
nk_define_spatials_metal_(euclidean, e2m1, apple9, e2m1x2, 32, 2, NUMKONG_METAL_LANGUAGE_3_1_, 16)
nk_define_spatials_metal_(angular, mxfp8e4m3, apple9, e4m3, 32, 1, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(euclidean, mxfp8e4m3, apple9, e4m3, 32, 1, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(angular, mxfp8e5m2, apple9, e5m2, 32, 1, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(euclidean, mxfp8e5m2, apple9, e5m2, 32, 1, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(angular, mxfp6e2m3, apple9, e2m3, 32, 1, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(euclidean, mxfp6e2m3, apple9, e2m3, 32, 1, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(angular, mxfp6e3m2, apple9, e3m2, 32, 1, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(euclidean, mxfp6e3m2, apple9, e3m2, 32, 1, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(angular, mxfp4, apple9, e2m1x2, 32, 2, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(euclidean, mxfp4, apple9, e2m1x2, 32, 2, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(angular, nvfp4, apple9, e2m1x2, 32, 2, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(euclidean, nvfp4, apple9, e2m1x2, 32, 2, NUMKONG_METAL_LANGUAGE_3_1_, 1)

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // NUMKONG_TARGET_APPLE9
#endif // NUMKONG_ARCH_METAL_
#endif // NUMKONG_SPATIALS_APPLE9_H
