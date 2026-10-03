/**
 *  @file c/cuda/blackwell.cu
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c blackwell kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_CUDA
#define NUMKONG_TARGET_CUDA 0
#undef NUMKONG_TARGET_AMPERE
#define NUMKONG_TARGET_AMPERE 0
#undef NUMKONG_TARGET_ADA
#define NUMKONG_TARGET_ADA 0
#include "numkong/numkong.h"

#include "numkong/dots/blackwell.cuh"
#include "numkong/spatials/blackwell.cuh"
#include "numkong/attention/blackwell.cuh"
