/**
 *  @file c/cpu/neonfhm.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c neonfhm kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_NEON
#define NUMKONG_TARGET_NEON 0
#include "numkong/numkong.h"

#include "numkong/reduce/neonfhm.h"
#include "numkong/dot/neonfhm.h"
#include "numkong/mesh/neonfhm.h"
#include "numkong/dots/neonfhm.h"
#include "numkong/spatials/neonfhm.h"
#include "numkong/attention/neonfhm.h"
