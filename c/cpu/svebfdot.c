/**
 *  @file c/cpu/svebfdot.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c svebfdot kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_NEON
#define NUMKONG_TARGET_NEON 0
#undef NUMKONG_TARGET_SVE
#define NUMKONG_TARGET_SVE 0
#include "numkong/numkong.h"

#include "numkong/dot/svebfdot.h"
#include "numkong/spatial/svebfdot.h"
