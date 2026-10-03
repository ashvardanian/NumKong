/**
 *  @file c/target/diamondamx.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c diamondamx kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_HASWELL
#define NUMKONG_TARGET_HASWELL 0
#undef NUMKONG_TARGET_SKYLAKE
#define NUMKONG_TARGET_SKYLAKE 0
#undef NUMKONG_TARGET_ICELAKE
#define NUMKONG_TARGET_ICELAKE 0
#undef NUMKONG_TARGET_SAPPHIREAMX
#define NUMKONG_TARGET_SAPPHIREAMX 0
#include "numkong/numkong.h"

#include "numkong/attention/diamondamx.h"
