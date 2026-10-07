/**
 *  @file c/target/blackwellultra.cu
 *  @author Ash Vardanian
 *  @date October 7, 2026
 *  @brief Every family's @c blackwellultra kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_CUDA
#define NUMKONG_TARGET_CUDA 0
#undef NUMKONG_TARGET_AMPERE
#define NUMKONG_TARGET_AMPERE 0
#undef NUMKONG_TARGET_ADA
#define NUMKONG_TARGET_ADA 0
#undef NUMKONG_TARGET_HOPPER
#define NUMKONG_TARGET_HOPPER 0
#undef NUMKONG_TARGET_BLACKWELL
#define NUMKONG_TARGET_BLACKWELL 0
#undef NUMKONG_TARGET_BLACKWELLRTX
#define NUMKONG_TARGET_BLACKWELLRTX 0
#include "numkong/numkong.h"

#include "numkong/attention/blackwellultra.cuh"
