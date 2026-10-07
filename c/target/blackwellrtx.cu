/**
 *  @file c/target/blackwellrtx.cu
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c blackwellrtx kernels, defined once for the NumKong library.
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
#undef NUMKONG_TARGET_BLACKWELLULTRA
#define NUMKONG_TARGET_BLACKWELLULTRA 0
#include "numkong/numkong.h"

#include "numkong/dots/blackwellrtx.cuh"
#include "numkong/spatials/blackwellrtx.cuh"
#include "numkong/attention/blackwellrtx.cuh"
