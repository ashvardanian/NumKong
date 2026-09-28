/**
 *  @file c/cpu/svesdot.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c svesdot kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_NEON
#define NUMKONG_TARGET_NEON 0
#undef NUMKONG_TARGET_SVE
#define NUMKONG_TARGET_SVE 0
#include "numkong/numkong.h"

#include "numkong/dot/svesdot.h"
#include "numkong/spatial/svesdot.h"
