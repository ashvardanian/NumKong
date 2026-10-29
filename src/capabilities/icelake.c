/**
 *  @file c/target/icelake.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c icelake kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_HASWELL
#define NUMKONG_TARGET_HASWELL 0
#undef NUMKONG_TARGET_SKYLAKE
#define NUMKONG_TARGET_SKYLAKE 0
#include "numkong/numkong.h"

#include "numkong/cast/icelake.h"
#include "numkong/reduce/icelake.h"
#include "numkong/dot/icelake.h"
#include "numkong/set/icelake.h"
#include "numkong/spatial/icelake.h"
#include "numkong/sparse/icelake.h"
#include "numkong/each/icelake.h"
#include "numkong/dots/icelake.h"
#include "numkong/sets/icelake.h"
#include "numkong/spatials/icelake.h"
#include "numkong/maxsim/icelake.h"
#include "numkong/attention/icelake.h"
