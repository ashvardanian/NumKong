/**
 *  @file c/target/cuda.cu
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c cuda kernels and the CUDA runtime calls, defined once for the library.
 */
#undef NUMKONG_TARGET_AMPERE
#define NUMKONG_TARGET_AMPERE 0
#undef NUMKONG_TARGET_ADA
#define NUMKONG_TARGET_ADA 0
#undef NUMKONG_TARGET_HOPPER
#define NUMKONG_TARGET_HOPPER 0
#undef NUMKONG_TARGET_BLACKWELL
#define NUMKONG_TARGET_BLACKWELL 0
#undef NUMKONG_TARGET_BLACKWELLRTX
#define NUMKONG_TARGET_BLACKWELLRTX 0
#undef NUMKONG_TARGET_BLACKWELLULTRA
#define NUMKONG_TARGET_BLACKWELLULTRA 0
#include "numkong/numkong.h"

#include "numkong/dots/cuda.cuh"
#include "numkong/spatials/cuda.cuh"
#include "numkong/attention/cuda.cuh"
#include "numkong/each/cuda.cuh"
#include "numkong/cast/cuda.cuh"
#include "numkong/reduce/cuda.cuh"

extern "C" NUMKONG_API nk_status_t nk_cuda_count_devices(nk_size_t *count) {
    *count = nk_cuda_count_devices_();
    return *count ? nk_success_k : nk_missing_gpu_k;
}
extern "C" NUMKONG_API nk_status_t nk_cuda_capabilities_detected(nk_size_t ordinal, nk_capability_t *capabilities) {
    return nk_cuda_capabilities_detected_(ordinal, capabilities);
}
extern "C" NUMKONG_API nk_status_t nk_cuda_stream_init(nk_size_t ordinal, void **stream) {
    return nk_cuda_stream_init_(ordinal, stream);
}
extern "C" NUMKONG_API nk_status_t nk_cuda_stream_free(void *stream) { return nk_cuda_stream_free_(stream); }

extern "C" NUMKONG_API nk_status_t nk_memory_allocate_unified_cuda(nk_size_t bytes, void **pointer, void *stream) {
    return nk_memory_allocate_unified_cuda_(bytes, pointer, stream);
}
extern "C" NUMKONG_API nk_status_t nk_memory_free_unified_cuda(void *pointer, nk_size_t bytes, void *stream) {
    nk_unused_(bytes);
    return nk_memory_free_unified_cuda_(pointer, stream);
}
extern "C" NUMKONG_API nk_status_t nk_stream_synchronize_cuda(void *stream) {
    return nk_stream_synchronize_cuda_(stream);
}

NUMKONG_API nk_status_t nk_allocator_init_unified_cuda(nk_allocator_t *allocator) {
    return nk_allocator_init_unified_cuda_(allocator);
}
NUMKONG_API nk_status_t nk_allocator_init_device_cuda(nk_allocator_t *allocator) {
    return nk_allocator_init_device_cuda_(allocator);
}
NUMKONG_API nk_status_t nk_allocator_init_pinned_cuda(nk_allocator_t *allocator) {
    return nk_allocator_init_pinned_cuda_(allocator);
}
