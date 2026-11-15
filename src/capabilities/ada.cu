/**
 *  @file src/capabilities/ada.cu
 *  @author Ash Vardanian
 *  @date October 6, 2026
 *  @brief Every family's @c ada kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_CUDA
#define NUMKONG_TARGET_CUDA 0
#undef NUMKONG_TARGET_AMPERE
#define NUMKONG_TARGET_AMPERE 0
#include "numkong/numkong.h"

#include "numkong/each/ada.cuh"
#include "numkong/cast/ada.cuh"

/** One of this unit's kernels, which @c nk_capabilities_compiled_cuda asks the runtime to load. */
extern "C" void const *nk_probe_ada_(void) { return (void const *)&nk_cast_ada_kernel_; }
