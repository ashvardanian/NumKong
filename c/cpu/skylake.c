/**
 *  @file c/cpu/skylake.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c skylake kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_HASWELL
#define NUMKONG_TARGET_HASWELL 0
#include "numkong/numkong.h"

#include "numkong/cast/skylake.h"
#include "numkong/reduce/skylake.h"
#include "numkong/dot/skylake.h"
#include "numkong/spatial/skylake.h"
#include "numkong/curved/skylake.h"
#include "numkong/probability/skylake.h"
#include "numkong/geospatial/skylake.h"
#include "numkong/mesh/skylake.h"
#include "numkong/each/skylake.h"
#include "numkong/trigonometry/skylake.h"
#include "numkong/dots/skylake.h"
#include "numkong/spatials/skylake.h"
#include "numkong/attention/skylake.h"
