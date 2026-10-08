/**
 *  @file c/target/ada.cu
 *  @author Ash Vardanian
 *  @date October 6, 2026
 *  @brief Every family's @c ada kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_CUDA
#define NUMKONG_TARGET_CUDA 0
#undef NUMKONG_TARGET_AMPERE
#define NUMKONG_TARGET_AMPERE 0
#include "numkong/numkong.h"

#include "numkong/each/ada.cuh"
#include "numkong/cast/ada.cuh"
