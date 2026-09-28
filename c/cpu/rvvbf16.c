/**
 *  @file c/cpu/rvvbf16.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c rvvbf16 kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_RVV
#define NUMKONG_TARGET_RVV 0
#include "numkong/numkong.h"

#include "numkong/dot/rvvbf16.h"
#include "numkong/spatial/rvvbf16.h"
