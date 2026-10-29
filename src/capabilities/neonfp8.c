/**
 *  @file c/target/neonfp8.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c neonfp8 kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_NEON
#define NUMKONG_TARGET_NEON 0
#include "numkong/numkong.h"

#include "numkong/dot/neonfp8.h"
#include "numkong/spatial/neonfp8.h"
#include "numkong/dots/neonfp8.h"
#include "numkong/spatials/neonfp8.h"
