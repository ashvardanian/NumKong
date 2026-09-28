/**
 *  @file c/cpu/diamond.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c diamond kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_HASWELL
#define NUMKONG_TARGET_HASWELL 0
#undef NUMKONG_TARGET_SKYLAKE
#define NUMKONG_TARGET_SKYLAKE 0
#include "numkong/numkong.h"

#include "numkong/cast/diamond.h"
#include "numkong/dot/diamond.h"
#include "numkong/spatial/diamond.h"
#include "numkong/dots/diamond.h"
#include "numkong/spatials/diamond.h"
