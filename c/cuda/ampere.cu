/**
 *  @file c/cuda/ampere.cu
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c ampere kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_CUDA
#define NUMKONG_TARGET_CUDA 0
#include "numkong/numkong.h"

#include "numkong/dots/ampere.cuh"
#include "numkong/spatials/ampere.cuh"
#include "numkong/attention/ampere.cuh"
