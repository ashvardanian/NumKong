/**
 *  @file c/metal/metal.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief The @c metal kernels and the Metal runtime helpers, defined once for the NumKong library.
 */
#undef NUMKONG_TARGET_METAL
#define NUMKONG_TARGET_METAL 1
#include "numkong/numkong.h"

#include "numkong/metal.h"
#include "numkong/dots/metal.h"

NUMKONG_API nk_status_t nk_metal_count_devices(nk_size_t *count) {
    *count = nk_metal_count_devices_();
    return *count ? nk_success_k : nk_missing_gpu_k;
}

NUMKONG_API nk_status_t nk_metal_capabilities_detected(nk_size_t ordinal, nk_capability_t *capabilities) {
    return nk_metal_capabilities_detected_(ordinal, capabilities);
}

NUMKONG_API nk_status_t nk_metal_stream_init(nk_size_t ordinal, void **stream) {
    return nk_metal_stream_init_(ordinal, stream);
}

NUMKONG_API nk_status_t nk_metal_stream_free(void *stream) { return nk_metal_stream_free_(stream); }
