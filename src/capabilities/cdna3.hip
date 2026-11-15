/**
 *  @file src/capabilities/cdna3.hip
 *  @author Ash Vardanian
 *  @date October 7, 2026
 *  @brief Every family's @c cdna3 kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_ROCM
#define NUMKONG_TARGET_ROCM 0
#include "numkong/numkong.h"

#include "numkong/dots/cdna3.cuh"
#include "numkong/spatials/cdna3.cuh"
#include "numkong/attention/cdna3.cuh"

/** One of this unit's kernels, which @c nk_capabilities_compiled_rocm asks the runtime to load. */
extern "C" void const *nk_probe_cdna3_(void) { return (void const *)&nk_dots_pack_bf16_cdna3_kernel_; }
