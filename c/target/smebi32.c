/**
 *  @file c/target/smebi32.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c smebi32 kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_NEON
#define NUMKONG_TARGET_NEON 0
#undef NUMKONG_TARGET_SVE
#define NUMKONG_TARGET_SVE 0
#undef NUMKONG_TARGET_SME
#define NUMKONG_TARGET_SME 0
#include "numkong/numkong.h"

#include "numkong/dots/smebi32.h"
#include "numkong/sets/smebi32.h"
