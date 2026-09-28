/**
 *  @file c/cpu/genoa.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c genoa kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_HASWELL
#define NUMKONG_TARGET_HASWELL 0
#undef NUMKONG_TARGET_SKYLAKE
#define NUMKONG_TARGET_SKYLAKE 0
#undef NUMKONG_TARGET_ICELAKE
#define NUMKONG_TARGET_ICELAKE 0
#include "numkong/numkong.h"

#include "numkong/reduce/genoa.h"
#include "numkong/dot/genoa.h"
#include "numkong/spatial/genoa.h"
#include "numkong/curved/genoa.h"
#include "numkong/mesh/genoa.h"
#include "numkong/dots/genoa.h"
#include "numkong/spatials/genoa.h"
#include "numkong/maxsim/genoa.h"
#include "numkong/attention/genoa.h"
