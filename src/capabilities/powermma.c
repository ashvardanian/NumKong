/**
 *  @file src/capabilities/powermma.c
 *  @author Ash Vardanian
 *  @date November 12, 2026
 *  @brief Every family's @c powermma kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_POWERVSX
#define NUMKONG_TARGET_POWERVSX 0
#include "numkong/numkong.h"

#include "numkong/dots/powermma.h"
#include "numkong/spatials/powermma.h"
