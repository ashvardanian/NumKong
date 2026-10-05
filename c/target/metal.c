/**
 *  @file c/target/metal.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief The @c metal kernels and the Metal runtime helpers, defined once for the NumKong library.
 */
#include "numkong/numkong.h"

#include "numkong/metal.h"
#include "numkong/dots/metal.h"

NUMKONG_API nk_metal_context_t *nk_metal_contexts_(os_unfair_lock_t *contexts_lock) {
    static nk_metal_context_t contexts[nk_metal_contexts_max_k];
    static os_unfair_lock lock;
    *contexts_lock = &lock;
    return contexts;
}

NUMKONG_API nk_status_t nk_memory_allocate_unified_metal(nk_size_t bytes, void **pointer, void *stream) {
    return nk_memory_allocate_unified_metal_(bytes, pointer, stream);
}

NUMKONG_API nk_status_t nk_memory_free_unified_metal(void *pointer, nk_size_t bytes, void *stream) {
    return nk_memory_free_unified_metal_(pointer, bytes, stream);
}

NUMKONG_API nk_status_t nk_allocator_init_unified_metal(nk_allocator_t *allocator) {
    return nk_allocator_init_unified_metal_(allocator);
}

NUMKONG_API nk_status_t nk_stream_synchronize_metal(void *stream) { return nk_stream_synchronize_metal_(stream); }

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
