/**
 *  @file c/cpu/neonbfdot.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c neonbfdot kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_NEON
#define NUMKONG_TARGET_NEON 0
#include "numkong/numkong.h"

#include "numkong/reduce/neonbfdot.h"
#include "numkong/dot/neonbfdot.h"
#include "numkong/spatial/neonbfdot.h"
#include "numkong/curved/neonbfdot.h"
#include "numkong/mesh/neonbfdot.h"
#include "numkong/each/neonbfdot.h"
#include "numkong/dots/neonbfdot.h"
#include "numkong/spatials/neonbfdot.h"
#include "numkong/attention/neonbfdot.h"
