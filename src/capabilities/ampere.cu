/**
 *  @file src/capabilities/ampere.cu
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c ampere kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_CUDA
#define NUMKONG_TARGET_CUDA 0
#include "numkong/numkong.h"

#include "numkong/dots/ampere.cuh"
#include "numkong/spatials/ampere.cuh"
#include "numkong/attention/ampere.cuh"
#include "numkong/each/ampere.cuh"
#include "numkong/cast/ampere.cuh"

/** One of this unit's kernels, which @c nk_capabilities_compiled_cuda asks the runtime to load. */
extern "C" void const *nk_probe_ampere_(void) { return (void const *)&nk_dots_pack_f64_ampere_kernel_; }
