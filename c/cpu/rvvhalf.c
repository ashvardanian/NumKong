/**
 *  @file c/cpu/rvvhalf.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c rvvhalf kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_RVV
#define NUMKONG_TARGET_RVV 0
#include "numkong/numkong.h"

#include "numkong/dot/rvvhalf.h"
#include "numkong/spatial/rvvhalf.h"
