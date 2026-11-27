/**
 *  @file src/missing/metal.c
 *  @author Ash Vardanian
 *  @date November 27, 2026
 *  @brief Every Metal producer and memory twin, answering @c nk_missing_gpu_k, for a library built
 *      without Metal kernels, so its exports match the builds that have them.
 */
#include "numkong/memory.h" // `nk_allocator_t`, `nk_capability_t`

NUMKONG_API nk_status_t nk_count_devices_metal(nk_size_t *count) {
    *count = 0;
    return nk_missing_gpu_k;
}
NUMKONG_API nk_status_t nk_capabilities_detected_metal(nk_size_t ordinal, nk_capability_t *capabilities) {
    nk_unused_(ordinal);
    *capabilities = 0;
    return nk_missing_gpu_k;
}
NUMKONG_API nk_status_t nk_capabilities_compiled_metal(nk_size_t ordinal, nk_capability_t *capabilities) {
    nk_unused_(ordinal);
    *capabilities = 0;
    return nk_missing_gpu_k;
}
NUMKONG_API nk_status_t nk_stream_init_metal(nk_size_t ordinal, nk_stream_t *stream) {
    nk_unused_(ordinal);
    *stream = NUMKONG_NULL;
    return nk_missing_gpu_k;
}
NUMKONG_API nk_status_t nk_stream_free_metal(nk_stream_t stream) {
    nk_unused_(stream);
    return nk_missing_gpu_k;
}
NUMKONG_API nk_status_t nk_stream_synchronize_metal(nk_stream_t stream) {
    nk_unused_(stream);
    return nk_missing_gpu_k;
}

NUMKONG_API nk_status_t nk_allocator_init_unified_metal(nk_allocator_t *allocator) {
    nk_unused_(allocator);
    return nk_missing_gpu_k;
}
NUMKONG_API nk_status_t nk_memory_allocate_unified_metal(nk_size_t bytes, void **pointer, nk_stream_t stream) {
    nk_unused_(bytes), nk_unused_(stream);
    *pointer = NUMKONG_NULL;
    return nk_missing_gpu_k;
}
NUMKONG_API nk_status_t nk_memory_free_unified_metal(void *pointer, nk_size_t bytes, nk_stream_t stream) {
    nk_unused_(pointer), nk_unused_(bytes), nk_unused_(stream);
    return nk_missing_gpu_k;
}
