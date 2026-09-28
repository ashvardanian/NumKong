/**
 *  @file c/cpu/sve.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c sve kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_NEON
#define NUMKONG_TARGET_NEON 0
#include "numkong/numkong.h"

#include "numkong/reduce/sve.h"
#include "numkong/dot/sve.h"
#include "numkong/set/sve.h"
#include "numkong/spatial/sve.h"
