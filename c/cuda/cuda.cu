/**
 *  @file c/cuda/cuda.cu
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c cuda kernels and the CUDA runtime calls, defined once for the library.
 */
#include "numkong/numkong.h"

#include "numkong/dots/simt.cuh"
#include "numkong/spatials/simt.cuh"
#include "numkong/attention/simt.cuh"
#include "numkong/each/simt.cuh"
#include "numkong/cast/simt.cuh"
#include "numkong/reduce/simt.cuh"

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
    *pointer = NUMKONG_NULL;
    if (!bytes) return nk_success_k;
    int current = 0, device = 0;
    if (cudaGetDevice(&current) != cudaSuccess) return nk_missing_gpu_k;
    if (!stream) device = current;
    else if (cudaStreamGetDevice((cudaStream_t)stream, &device) != cudaSuccess) return nk_device_code_mismatch_k;
    // Managed memory belongs to the current device's context, so switch to the stream's device
    if (device != current && cudaSetDevice(device) != cudaSuccess) return nk_device_code_mismatch_k;
    cudaError_t const allocated = cudaMallocManaged(pointer, bytes, cudaMemAttachGlobal);
    if (device != current) cudaSetDevice(current);
    if (allocated == cudaSuccess) return nk_success_k;
    *pointer = NUMKONG_NULL;
    return nk_bad_alloc_k;
}
extern "C" NUMKONG_API nk_status_t nk_memory_free_unified_cuda(void *pointer, nk_size_t bytes, void *stream) {
    nk_unused_(bytes), nk_unused_(stream);
    return cudaFree(pointer) == cudaSuccess ? nk_success_k : nk_device_memory_mismatch_k;
}
extern "C" NUMKONG_API nk_status_t nk_stream_synchronize_cuda(void *stream) {
    return cudaStreamSynchronize((cudaStream_t)stream) == cudaSuccess ? nk_success_k : nk_device_code_mismatch_k;
}
