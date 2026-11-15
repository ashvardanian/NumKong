/**
 *  @file src/capabilities/blackwellultra.cu
 *  @author Ash Vardanian
 *  @date October 7, 2026
 *  @brief Every family's @c blackwellultra kernels, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_CUDA
#define NUMKONG_TARGET_CUDA 0
#undef NUMKONG_TARGET_AMPERE
#define NUMKONG_TARGET_AMPERE 0
#undef NUMKONG_TARGET_ADA
#define NUMKONG_TARGET_ADA 0
#undef NUMKONG_TARGET_BLACKWELL
#define NUMKONG_TARGET_BLACKWELL 0
#include "numkong/numkong.h"

#include "numkong/attention/blackwellultra.cuh"

/** One of this unit's kernels, which @c nk_capabilities_compiled_cuda asks the runtime to load. */
extern "C" void const *nk_probe_blackwellultra_(void) {
    return (void const *)&nk_attention_pack_bf16_blackwellultra_kernel_;
}
