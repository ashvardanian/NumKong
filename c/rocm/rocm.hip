/**
 *  @file c/rocm/rocm.hip
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c rocm kernels and the HIP runtime calls, defined once for the library.
 */
#include "numkong/numkong.h"

#include "numkong/dots/simt.cuh"
#include "numkong/spatials/simt.cuh"
#include "numkong/attention/simt.cuh"
#include "numkong/each/simt.cuh"

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
    *pointer = NUMKONG_NULL;
    if (!bytes) return nk_success_k;
    int current = 0, device = 0;
    if (hipGetDevice(&current) != hipSuccess) return nk_missing_gpu_k;
    if (!stream) device = current;
    else if (hipStreamGetDevice((hipStream_t)stream, &device) != hipSuccess) return nk_device_code_mismatch_k;
    // Managed memory belongs to the current device's context, so switch to the stream's device
    if (device != current && hipSetDevice(device) != hipSuccess) return nk_device_code_mismatch_k;
    hipError_t const allocated = hipMallocManaged(pointer, bytes, hipMemAttachGlobal);
    [[maybe_unused]] hipError_t const restored = device != current ? hipSetDevice(current) : hipSuccess;
    if (allocated == hipSuccess) return nk_success_k;
    *pointer = NUMKONG_NULL;
    return nk_bad_alloc_k;
}
extern "C" NUMKONG_API nk_status_t nk_memory_free_unified_rocm(void *pointer, nk_size_t bytes, void *stream) {
    nk_unused_(bytes), nk_unused_(stream);
    return hipFree(pointer) == hipSuccess ? nk_success_k : nk_device_memory_mismatch_k;
}
extern "C" NUMKONG_API nk_status_t nk_stream_synchronize_rocm(void *stream) {
    return hipStreamSynchronize((hipStream_t)stream) == hipSuccess ? nk_success_k : nk_device_code_mismatch_k;
}
