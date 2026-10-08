/**
 *  @file c/target/metal.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief The @c metal kernels and the Metal runtime helpers, defined once for the NumKong library.
 */
#include "numkong/numkong.h"

#include "numkong/metal.h"
#include "numkong/dots/metal.h"
#include "numkong/spatials/metal.h"

NUMKONG_API nk_metal_context_t *nk_contexts_metal_(os_unfair_lock_t *contexts_lock) {
    static nk_metal_context_t contexts[nk_metal_contexts_max_k];
    static os_unfair_lock lock;
    *contexts_lock = &lock;
    return contexts;
}

NUMKONG_API nk_status_t nk_memory_allocate_unified_metal(nk_size_t bytes, void **pointer, nk_stream_t stream) {
    return nk_memory_allocate_unified_metal_(bytes, pointer, stream);
}

NUMKONG_API nk_status_t nk_memory_free_unified_metal(void *pointer, nk_size_t bytes, nk_stream_t stream) {
    return nk_memory_free_unified_metal_(pointer, bytes, stream);
}

NUMKONG_API nk_status_t nk_allocator_init_unified_metal(nk_allocator_t *allocator) {
    return nk_allocator_init_unified_metal_(allocator);
}

NUMKONG_API nk_status_t nk_stream_synchronize_metal(nk_stream_t stream) { return nk_stream_synchronize_metal_(stream); }

NUMKONG_API nk_status_t nk_metal_count_devices(nk_size_t *count) {
    *count = nk_count_devices_metal_();
    return *count ? nk_success_k : nk_missing_gpu_k;
}

NUMKONG_API nk_status_t nk_metal_capabilities_detected(nk_size_t ordinal, nk_capability_t *capabilities) {
    return nk_capabilities_detected_metal_(ordinal, capabilities);
}

NUMKONG_API nk_status_t nk_stream_init_metal(nk_size_t ordinal, nk_stream_t *stream) {
    return nk_stream_init_metal_(ordinal, stream);
}

NUMKONG_API nk_status_t nk_stream_free_metal(nk_stream_t stream) { return nk_stream_free_metal_(stream); }
