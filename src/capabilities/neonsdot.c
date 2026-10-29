/**
 *  @file c/target/neonsdot.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c neonsdot kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_NEON
#define NUMKONG_TARGET_NEON 0
#include "numkong/numkong.h"

#include "numkong/reduce/neonsdot.h"
#include "numkong/dot/neonsdot.h"
#include "numkong/spatial/neonsdot.h"
#include "numkong/dots/neonsdot.h"
#include "numkong/spatials/neonsdot.h"
#include "numkong/maxsim/neonsdot.h"
#include "numkong/attention/neonsdot.h"
