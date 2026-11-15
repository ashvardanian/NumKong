/**
 *  @file src/capabilities/cuda.cu
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c cuda kernels and the CUDA runtime calls, defined once for the library.
 */
#include "numkong/numkong.h"

#include "numkong/dots/cuda.cuh"
#include "numkong/spatials/cuda.cuh"
#include "numkong/attention/cuda.cuh"
#include "numkong/each/cuda.cuh"
#include "numkong/cast/cuda.cuh"
#include "numkong/reduce/cuda.cuh"
#include "numkong/geospatial/cuda.cuh"
#include "numkong/shapes/cuda.cuh"

extern "C" NUMKONG_API nk_status_t nk_count_devices_cuda(nk_size_t *count) {
    *count = nk_count_devices_cuda_();
    return *count ? nk_success_k : nk_missing_gpu_k;
}
extern "C" NUMKONG_API nk_status_t nk_capabilities_detected_cuda(nk_size_t ordinal, nk_capability_t *capabilities) {
    return nk_capabilities_detected_cuda_(ordinal, capabilities);
}
extern "C" NUMKONG_API nk_status_t nk_stream_init_cuda(nk_size_t ordinal, nk_stream_t *stream) {
    return nk_stream_init_cuda_(ordinal, stream);
}
extern "C" NUMKONG_API nk_status_t nk_stream_free_cuda(nk_stream_t stream) { return nk_stream_free_cuda_(stream); }

extern "C" NUMKONG_API nk_status_t nk_memory_allocate_unified_cuda(nk_size_t bytes, void **pointer,
                                                                   nk_stream_t stream) {
    return nk_memory_allocate_unified_cuda_(bytes, pointer, stream);
}
extern "C" NUMKONG_API nk_status_t nk_memory_free_unified_cuda(void *pointer, nk_size_t bytes, nk_stream_t stream) {
    nk_unused_(bytes);
    return nk_memory_free_unified_cuda_(pointer, stream);
}
extern "C" NUMKONG_API nk_status_t nk_stream_synchronize_cuda(nk_stream_t stream) {
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

extern "C" void const *nk_probe_ampere_(void);
extern "C" void const *nk_probe_ada_(void);
extern "C" void const *nk_probe_hopper_(void);
extern "C" void const *nk_probe_blackwell_(void);
extern "C" void const *nk_probe_blackwellrtx_(void);
extern "C" void const *nk_probe_blackwellultra_(void);

/** Whether the driver loads @p kernel on the current device, which fails where no code runs. */
static bool nk_loads_cuda_(void const *kernel) {
    cudaFuncAttributes attributes;
    bool const loads = cudaFuncGetAttributes(&attributes, kernel) == cudaSuccess;
    nk_unused_(cudaGetLastError());
    return loads;
}

extern "C" NUMKONG_API nk_status_t nk_capabilities_compiled_cuda(nk_size_t ordinal, nk_capability_t *capabilities) {
    int previous = 0;
    *capabilities = 0;
    if (ordinal >= nk_count_devices_cuda_()) return nk_missing_gpu_k;
    if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice((int)ordinal) != cudaSuccess) {
        nk_unused_(cudaGetLastError());
        return nk_device_code_mismatch_k;
    }
    nk_capability_t compiled = nk_cap_cuda_k * nk_loads_cuda_((void const *)&nk_dots_pack_f64_cuda_kernel_);
#if NUMKONG_TARGET_AMPERE
    compiled |= nk_cap_ampere_k * nk_loads_cuda_(nk_probe_ampere_());
#endif
#if NUMKONG_TARGET_ADA
    compiled |= nk_cap_ada_k * nk_loads_cuda_(nk_probe_ada_());
#endif
#if NUMKONG_TARGET_HOPPER
    compiled |= nk_cap_hopper_k * nk_loads_cuda_(nk_probe_hopper_());
#endif
#if NUMKONG_TARGET_BLACKWELL
    compiled |= nk_cap_blackwell_k * nk_loads_cuda_(nk_probe_blackwell_());
#endif
#if NUMKONG_TARGET_BLACKWELLRTX
    compiled |= nk_cap_blackwellrtx_k * nk_loads_cuda_(nk_probe_blackwellrtx_());
#endif
#if NUMKONG_TARGET_BLACKWELLULTRA
    compiled |= nk_cap_blackwellultra_k * nk_loads_cuda_(nk_probe_blackwellultra_());
#endif
    nk_unused_(cudaSetDevice(previous));
    *capabilities = compiled;
    return nk_success_k;
}
