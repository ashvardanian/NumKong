/**
 *  @file src/missing/rocm.c
 *  @author Ash Vardanian
 *  @date November 27, 2026
 *  @brief Every ROCm producer and memory twin, answering @c nk_missing_gpu_k, for a library built
 *      without ROCm kernels, so its exports match the builds that have them.
 */
#include "numkong/memory.h" // `nk_allocator_t`, `nk_capability_t`

NUMKONG_API nk_status_t nk_count_devices_rocm(nk_size_t *count) {
    *count = 0;
    return nk_missing_gpu_k;
}
NUMKONG_API nk_status_t nk_capabilities_detected_rocm(nk_size_t ordinal, nk_capability_t *capabilities) {
    nk_unused_(ordinal);
    *capabilities = 0;
    return nk_missing_gpu_k;
}
NUMKONG_API nk_status_t nk_capabilities_compiled_rocm(nk_size_t ordinal, nk_capability_t *capabilities) {
    nk_unused_(ordinal);
    *capabilities = 0;
    return nk_missing_gpu_k;
}
NUMKONG_API nk_status_t nk_stream_init_rocm(nk_size_t ordinal, nk_stream_t *stream) {
    nk_unused_(ordinal);
    *stream = NUMKONG_NULL;
    return nk_missing_gpu_k;
}
NUMKONG_API nk_status_t nk_stream_free_rocm(nk_stream_t stream) {
    nk_unused_(stream);
    return nk_missing_gpu_k;
}
NUMKONG_API nk_status_t nk_stream_synchronize_rocm(nk_stream_t stream) {
    nk_unused_(stream);
    return nk_missing_gpu_k;
}

NUMKONG_API nk_status_t nk_allocator_init_unified_rocm(nk_allocator_t *allocator) {
    nk_unused_(allocator);
    return nk_missing_gpu_k;
}
NUMKONG_API nk_status_t nk_allocator_init_device_rocm(nk_allocator_t *allocator) {
    nk_unused_(allocator);
    return nk_missing_gpu_k;
}
NUMKONG_API nk_status_t nk_allocator_init_pinned_rocm(nk_allocator_t *allocator) {
    nk_unused_(allocator);
    return nk_missing_gpu_k;
}
NUMKONG_API nk_status_t nk_memory_allocate_unified_rocm(nk_size_t bytes, void **pointer, nk_stream_t stream) {
    nk_unused_(bytes), nk_unused_(stream);
    *pointer = NUMKONG_NULL;
    return nk_missing_gpu_k;
}
NUMKONG_API nk_status_t nk_memory_free_unified_rocm(void *pointer, nk_size_t bytes, nk_stream_t stream) {
    nk_unused_(pointer), nk_unused_(bytes), nk_unused_(stream);
    return nk_missing_gpu_k;
}
