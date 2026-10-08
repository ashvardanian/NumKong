/**
 *  @file include/numkong/spatials/apple10.h
 *  @author Ash Vardanian
 *  @date October 5, 2026
 *  @brief Batched spatial distances on Apple family 10 GPUs, launched from C.
 *
 *  @sa include/numkong/spatials.h
 */
#ifndef NUMKONG_SPATIALS_APPLE10_H
#define NUMKONG_SPATIALS_APPLE10_H

#if NUMKONG_ARCH_METAL_
#if NUMKONG_TARGET_APPLE10
#include "numkong/dots/apple10.h"
#include "numkong/spatials/metal.h"

#if defined(__cplusplus)
extern "C" {
#endif

nk_define_spatials_metal_(angular, i8, apple10, i8, 64, 1, NUMKONG_METAL_LANGUAGE_4_0_, 1)
nk_define_spatials_metal_(angular, i4, apple10, i4x2, 64, 2, NUMKONG_METAL_LANGUAGE_4_0_, 1)
nk_define_spatials_metal_(angular, u4, apple10, u4x2, 64, 2, NUMKONG_METAL_LANGUAGE_4_0_, 1)
nk_define_spatials_metal_(angular, u8, apple10, u8, 64, 1, NUMKONG_METAL_LANGUAGE_4_0_, 1)
nk_define_spatials_metal_(angular, f16, apple10, f16, 32, 1, NUMKONG_METAL_LANGUAGE_4_0_, 1)
nk_define_spatials_metal_(angular, bf16, apple10, bf16, 32, 1, NUMKONG_METAL_LANGUAGE_4_0_, 1)
nk_define_spatials_metal_(angular, e4m3, apple10, e4m3, 64, 1, NUMKONG_METAL_LANGUAGE_4_0_, 1)
nk_define_spatials_metal_(angular, e5m2, apple10, e5m2, 64, 1, NUMKONG_METAL_LANGUAGE_4_0_, 1)
nk_define_spatials_metal_(angular, e3m2, apple10, e3m2, 64, 1, NUMKONG_METAL_LANGUAGE_4_0_, 1)
nk_define_spatials_metal_(angular, e2m3, apple10, e2m3, 64, 1, NUMKONG_METAL_LANGUAGE_4_0_, 1)
nk_define_spatials_metal_(angular, e2m1, apple10, e2m1x2, 128, 2, NUMKONG_METAL_LANGUAGE_4_0_, 1)
nk_define_spatials_metal_(euclidean, i8, apple10, i8, 64, 1, NUMKONG_METAL_LANGUAGE_4_0_, 1)
nk_define_spatials_metal_(euclidean, i4, apple10, i4x2, 64, 2, NUMKONG_METAL_LANGUAGE_4_0_, 1)
nk_define_spatials_metal_(euclidean, u4, apple10, u4x2, 64, 2, NUMKONG_METAL_LANGUAGE_4_0_, 1)
nk_define_spatials_metal_(euclidean, u8, apple10, u8, 64, 1, NUMKONG_METAL_LANGUAGE_4_0_, 1)
nk_define_spatials_metal_(euclidean, f16, apple10, f16, 32, 1, NUMKONG_METAL_LANGUAGE_4_0_, 1)
nk_define_spatials_metal_(euclidean, bf16, apple10, bf16, 32, 1, NUMKONG_METAL_LANGUAGE_4_0_, 1)
nk_define_spatials_metal_(euclidean, e4m3, apple10, e4m3, 64, 1, NUMKONG_METAL_LANGUAGE_4_0_, 1)
nk_define_spatials_metal_(euclidean, e5m2, apple10, e5m2, 64, 1, NUMKONG_METAL_LANGUAGE_4_0_, 1)
nk_define_spatials_metal_(euclidean, e3m2, apple10, e3m2, 64, 1, NUMKONG_METAL_LANGUAGE_4_0_, 1)
nk_define_spatials_metal_(euclidean, e2m3, apple10, e2m3, 64, 1, NUMKONG_METAL_LANGUAGE_4_0_, 1)
nk_define_spatials_metal_(euclidean, e2m1, apple10, e2m1x2, 128, 2, NUMKONG_METAL_LANGUAGE_4_0_, 1)
nk_define_spatials_metal_(angular, mxfp8e4m3, apple10, e4m3, 64, 1, NUMKONG_METAL_LANGUAGE_4_0_, 1)
nk_define_spatials_metal_(euclidean, mxfp8e4m3, apple10, e4m3, 64, 1, NUMKONG_METAL_LANGUAGE_4_0_, 1)
nk_define_spatials_metal_(angular, mxfp8e5m2, apple10, e5m2, 64, 1, NUMKONG_METAL_LANGUAGE_4_0_, 1)
nk_define_spatials_metal_(euclidean, mxfp8e5m2, apple10, e5m2, 64, 1, NUMKONG_METAL_LANGUAGE_4_0_, 1)
nk_define_spatials_metal_(angular, mxfp6e2m3, apple10, e2m3, 64, 1, NUMKONG_METAL_LANGUAGE_4_0_, 1)
nk_define_spatials_metal_(euclidean, mxfp6e2m3, apple10, e2m3, 64, 1, NUMKONG_METAL_LANGUAGE_4_0_, 1)
nk_define_spatials_metal_(angular, mxfp6e3m2, apple10, e3m2, 64, 1, NUMKONG_METAL_LANGUAGE_4_0_, 1)
nk_define_spatials_metal_(euclidean, mxfp6e3m2, apple10, e3m2, 64, 1, NUMKONG_METAL_LANGUAGE_4_0_, 1)
nk_define_spatials_metal_(angular, mxfp4, apple10, e2m1x2, 64, 2, NUMKONG_METAL_LANGUAGE_4_0_, 1)
nk_define_spatials_metal_(euclidean, mxfp4, apple10, e2m1x2, 64, 2, NUMKONG_METAL_LANGUAGE_4_0_, 1)
nk_define_spatials_metal_(angular, nvfp4, apple10, e2m1x2, 64, 2, NUMKONG_METAL_LANGUAGE_4_0_, 1)
nk_define_spatials_metal_(euclidean, nvfp4, apple10, e2m1x2, 64, 2, NUMKONG_METAL_LANGUAGE_4_0_, 1)

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // NUMKONG_TARGET_APPLE10
#endif // NUMKONG_ARCH_METAL_
#endif // NUMKONG_SPATIALS_APPLE10_H
