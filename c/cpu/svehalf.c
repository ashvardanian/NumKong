/**
 *  @file c/cpu/svehalf.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c svehalf kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_NEON
#define NUMKONG_TARGET_NEON 0
#undef NUMKONG_TARGET_SVE
#define NUMKONG_TARGET_SVE 0
#include "numkong/numkong.h"

#include "numkong/dot/svehalf.h"
#include "numkong/spatial/svehalf.h"
#include "numkong/trigonometry/svehalf.h"
