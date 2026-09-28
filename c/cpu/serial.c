/**
 *  @file c/cpu/serial.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c serial kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_SERIAL
#define NUMKONG_TARGET_SERIAL 1
#include "numkong/numkong.h"

#include "numkong/scalar/serial.h"
#include "numkong/cast/serial.h"
#include "numkong/reduce/serial.h"
#include "numkong/dot/serial.h"
#include "numkong/set/serial.h"
#include "numkong/spatial/serial.h"
#include "numkong/curved/serial.h"
#include "numkong/probability/serial.h"
#include "numkong/geospatial/serial.h"
#include "numkong/mesh/serial.h"
#include "numkong/sparse/serial.h"
#include "numkong/each/serial.h"
#include "numkong/trigonometry/serial.h"
#include "numkong/dots/serial.h"
#include "numkong/sets/serial.h"
#include "numkong/spatials/serial.h"
#include "numkong/maxsim/serial.h"
#include "numkong/attention/serial.h"
