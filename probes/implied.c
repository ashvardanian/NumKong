/**
 *  @file probes/implied.c
 *  @author Ash Vardanian
 *  @date October 28, 2026
 *  @brief Whether the floor's predefined macros imply the capability @c NUMKONG_PROBE_MACRO names.
 */
#include "numkong/types.h"
#if !NUMKONG_PROBE_MACRO
#error
#endif
