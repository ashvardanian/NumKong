/**
 *  @file c/target/cdna5.hip
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c cdna5 kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_ROCM
#define NUMKONG_TARGET_ROCM 0
#undef NUMKONG_TARGET_CDNA4
#define NUMKONG_TARGET_CDNA4 0
#include "numkong/numkong.h"

#include "numkong/dots/cdna5.cuh"
#include "numkong/spatials/cdna5.cuh"
#include "numkong/attention/cdna5.cuh"
