/**
 *  @file c/cpu/sapphireamx.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c sapphireamx kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_HASWELL
#define NUMKONG_TARGET_HASWELL 0
#undef NUMKONG_TARGET_SKYLAKE
#define NUMKONG_TARGET_SKYLAKE 0
#undef NUMKONG_TARGET_ICELAKE
#define NUMKONG_TARGET_ICELAKE 0
#include "numkong/numkong.h"

#include "numkong/dots/sapphireamx.h"
#include "numkong/spatials/sapphireamx.h"
#include "numkong/maxsim/sapphireamx.h"
#include "numkong/attention/sapphireamx.h"
