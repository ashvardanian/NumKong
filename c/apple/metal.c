/**
 *  @file c/apple/metal.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief The @c metal kernels and the Metal runtime helpers, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_METAL
#define NUMKONG_TARGET_METAL 1
#include "numkong/numkong.h"

#include "numkong/metal.h"
#include "numkong/dots/simt.h"

NUMKONG_API nk_status_t nk_metal_count_devices(nk_size_t *count) {
    *count = nk_metal_count_devices_();
    return *count ? nk_success_k : nk_missing_gpu_k;
}

NUMKONG_API nk_status_t nk_metal_capabilities_detected(nk_size_t device, nk_capability_t *capabilities) {
    return nk_metal_capabilities_detected_(device, capabilities);
}
