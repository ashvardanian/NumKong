/**
 *  @file c/cpu/alder.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c alder kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_HASWELL
#define NUMKONG_TARGET_HASWELL 0
#include "numkong/numkong.h"

#include "numkong/reduce/alder.h"
#include "numkong/dot/alder.h"
#include "numkong/spatial/alder.h"
#include "numkong/dots/alder.h"
#include "numkong/spatials/alder.h"
#include "numkong/maxsim/alder.h"
