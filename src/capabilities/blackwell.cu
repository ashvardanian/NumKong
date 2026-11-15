/**
 *  @file src/capabilities/blackwell.cu
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c blackwell kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_CUDA
#define NUMKONG_TARGET_CUDA 0
#undef NUMKONG_TARGET_AMPERE
#define NUMKONG_TARGET_AMPERE 0
#undef NUMKONG_TARGET_ADA
#define NUMKONG_TARGET_ADA 0
#include "numkong/numkong.h"

#include "numkong/dots/blackwell.cuh"
#include "numkong/spatials/blackwell.cuh"
#include "numkong/attention/blackwell.cuh"
#include "numkong/shapes/blackwell.cuh"

/** One of this unit's kernels, which @c nk_capabilities_compiled_cuda asks the runtime to load. */
extern "C" void const *nk_probe_blackwell_(void) { return (void const *)&nk_dots_pack_bf16_blackwell_kernel_; }
