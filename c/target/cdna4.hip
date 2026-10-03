/**
 *  @file c/target/cdna4.hip
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c cdna4 kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_ROCM
#define NUMKONG_TARGET_ROCM 0
#include "numkong/numkong.h"

#include "numkong/dots/cdna4.cuh"
#include "numkong/spatials/cdna4.cuh"
#include "numkong/attention/cdna4.cuh"
