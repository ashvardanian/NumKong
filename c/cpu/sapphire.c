/**
 *  @file c/cpu/sapphire.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c sapphire kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_SKYLAKE
#define NUMKONG_TARGET_SKYLAKE 0
#undef NUMKONG_TARGET_ICELAKE
#define NUMKONG_TARGET_ICELAKE 0
#include "numkong/numkong.h"

#include "numkong/scalar/sapphire.h"
#include "numkong/cast/sapphire.h"
#include "numkong/each/sapphire.h"
#include "numkong/trigonometry/sapphire.h"
