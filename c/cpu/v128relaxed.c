/**
 *  @file c/cpu/v128relaxed.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c v128relaxed kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_V128
#define NUMKONG_TARGET_V128 0
#include "numkong/numkong.h"

#include "numkong/scalar/v128relaxed.h"
#include "numkong/cast/v128relaxed.h"
#include "numkong/reduce/v128relaxed.h"
#include "numkong/dot/v128relaxed.h"
#include "numkong/spatial/v128relaxed.h"
#include "numkong/geospatial/v128relaxed.h"
#include "numkong/mesh/v128relaxed.h"
#include "numkong/each/v128relaxed.h"
#include "numkong/trigonometry/v128relaxed.h"
#include "numkong/dots/v128relaxed.h"
#include "numkong/spatials/v128relaxed.h"
#include "numkong/maxsim/v128relaxed.h"
#include "numkong/attention/v128relaxed.h"
