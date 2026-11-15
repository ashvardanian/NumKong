/**
 *  @file src/capabilities/blackwellrtx.cu
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c blackwellrtx kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_CUDA
#define NUMKONG_TARGET_CUDA 0
#undef NUMKONG_TARGET_AMPERE
#define NUMKONG_TARGET_AMPERE 0
#undef NUMKONG_TARGET_ADA
#define NUMKONG_TARGET_ADA 0
#include "numkong/numkong.h"

#include "numkong/dots/blackwellrtx.cuh"
#include "numkong/spatials/blackwellrtx.cuh"
#include "numkong/attention/blackwellrtx.cuh"

/** One of this unit's kernels, which @c nk_capabilities_compiled_cuda asks the runtime to load. */
extern "C" void const *nk_probe_blackwellrtx_(void) { return (void const *)&nk_dots_pack_e4m3_blackwellrtx_kernel_; }
