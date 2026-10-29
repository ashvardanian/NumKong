/**
 *  @file c/target/sme.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c sme kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_NEON
#define NUMKONG_TARGET_NEON 0
#undef NUMKONG_TARGET_SVE
#define NUMKONG_TARGET_SVE 0
#include "numkong/numkong.h"

#include "numkong/each/sme.h"
#include "numkong/dots/sme.h"
#include "numkong/spatials/sme.h"
#include "numkong/maxsim/sme.h"
#include "numkong/attention/sme.h"
