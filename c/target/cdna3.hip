/**
 *  @file c/target/cdna3.hip
 *  @author Ash Vardanian
 *  @date October 7, 2026
 *  @brief Every family's @c cdna3 kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_ROCM
#define NUMKONG_TARGET_ROCM 0
#undef NUMKONG_TARGET_CDNA4
#define NUMKONG_TARGET_CDNA4 0
#undef NUMKONG_TARGET_CDNA5
#define NUMKONG_TARGET_CDNA5 0
#include "numkong/numkong.h"

#include "numkong/dots/cdna3.cuh"
#include "numkong/spatials/cdna3.cuh"
