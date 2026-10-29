/**
 *  @file c/target/sierra.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c sierra kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_HASWELL
#define NUMKONG_TARGET_HASWELL 0
#include "numkong/numkong.h"

#include "numkong/reduce/sierra.h"
#include "numkong/dot/sierra.h"
#include "numkong/spatial/sierra.h"
#include "numkong/dots/sierra.h"
#include "numkong/spatials/sierra.h"
