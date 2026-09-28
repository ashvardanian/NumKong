/**
 *  @file c/cpu/sve2.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c sve2 kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_SVE
#define NUMKONG_TARGET_SVE 0
#include "numkong/numkong.h"

#include "numkong/sparse/sve2.h"
