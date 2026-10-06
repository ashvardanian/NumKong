/**
 *  @file c/target/rocm.hip
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c rocm kernels and the HIP runtime calls, defined once for the library.
 */
#undef NUMKONG_TARGET_CDNA4
#define NUMKONG_TARGET_CDNA4 0
#undef NUMKONG_TARGET_CDNA5
#define NUMKONG_TARGET_CDNA5 0
#include "numkong/numkong.h"

#include "numkong/dots/rocm.cuh"
#include "numkong/spatials/rocm.cuh"
#include "numkong/attention/rocm.cuh"
#include "numkong/each/rocm.cuh"

extern "C" NUMKONG_API nk_status_t nk_rocm_count_devices(nk_size_t *count) {
    *count = nk_rocm_count_devices_();
    return *count ? nk_success_k : nk_missing_gpu_k;
}
extern "C" NUMKONG_API nk_status_t nk_rocm_capabilities_detected(nk_size_t ordinal, nk_capability_t *capabilities) {
    return nk_rocm_capabilities_detected_(ordinal, capabilities);
}
extern "C" NUMKONG_API nk_status_t nk_rocm_stream_init(nk_size_t ordinal, void **stream) {
    return nk_rocm_stream_init_(ordinal, stream);
}
extern "C" NUMKONG_API nk_status_t nk_rocm_stream_free(void *stream) { return nk_rocm_stream_free_(stream); }

extern "C" NUMKONG_API nk_status_t nk_memory_allocate_unified_rocm(nk_size_t bytes, void **pointer, void *stream) {
    return nk_memory_allocate_unified_rocm_(bytes, pointer, stream);
}
extern "C" NUMKONG_API nk_status_t nk_memory_free_unified_rocm(void *pointer, nk_size_t bytes, void *stream) {
    nk_unused_(bytes);
    return nk_memory_free_unified_rocm_(pointer, stream);
}
extern "C" NUMKONG_API nk_status_t nk_stream_synchronize_rocm(void *stream) {
    return nk_stream_synchronize_rocm_(stream);
}

NUMKONG_API nk_status_t nk_allocator_init_unified_rocm(nk_allocator_t *allocator) {
    return nk_allocator_init_unified_rocm_(allocator);
}
NUMKONG_API nk_status_t nk_allocator_init_device_rocm(nk_allocator_t *allocator) {
    return nk_allocator_init_device_rocm_(allocator);
}
NUMKONG_API nk_status_t nk_allocator_init_pinned_rocm(nk_allocator_t *allocator) {
    return nk_allocator_init_pinned_rocm_(allocator);
}
