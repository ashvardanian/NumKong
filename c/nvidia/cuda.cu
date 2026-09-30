/**
 *  @file c/nvidia/cuda.cu
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c cuda kernels, defined once for the NumKong library.
 */
#include "numkong/numkong.h"

#include "numkong/dots/simt.cuh"
#include "numkong/spatials/simt.cuh"
#include "numkong/attention/simt.cuh"
#include "numkong/each/simt.cuh"

extern "C" NUMKONG_API nk_status_t nk_cuda_count_devices(nk_size_t *count) {
    *count = nk_cuda_count_devices_();
    return *count ? nk_success_k : nk_missing_gpu_k;
}
extern "C" NUMKONG_API nk_status_t nk_cuda_capabilities_detected(nk_size_t device, nk_capability_t *capabilities) {
    return nk_cuda_capabilities_detected_(device, capabilities);
}
