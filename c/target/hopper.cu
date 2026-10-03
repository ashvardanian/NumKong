/**
 *  @file c/target/hopper.cu
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c hopper kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_CUDA
#define NUMKONG_TARGET_CUDA 0
#undef NUMKONG_TARGET_AMPERE
#define NUMKONG_TARGET_AMPERE 0
#undef NUMKONG_TARGET_ADA
#define NUMKONG_TARGET_ADA 0
#include "numkong/numkong.h"

#include "numkong/dots/hopper.cuh"
#include "numkong/spatials/hopper.cuh"
#include "numkong/attention/hopper.cuh"
