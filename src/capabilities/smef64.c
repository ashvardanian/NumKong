/**
 *  @file c/target/smef64.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c smef64 kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_SVE
#define NUMKONG_TARGET_SVE 0
#undef NUMKONG_TARGET_SME
#define NUMKONG_TARGET_SME 0
#include "numkong/numkong.h"

#include "numkong/curved/smef64.h"
#include "numkong/dots/smef64.h"
#include "numkong/spatials/smef64.h"
