/**
 *  @file src/capabilities/rocm.hip
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Every family's @c rocm kernels and the HIP runtime calls, defined once for the library.
 */
#include "numkong/numkong.h"

#include "numkong/dots/rocm.cuh"
#include "numkong/spatials/rocm.cuh"
#include "numkong/attention/rocm.cuh"
#include "numkong/each/rocm.cuh"
#include "numkong/cast/rocm.cuh"
#include "numkong/reduce/rocm.cuh"
#include "numkong/geospatial/rocm.cuh"
#include "numkong/shapes/rocm.cuh"

extern "C" NUMKONG_API nk_status_t nk_count_devices_rocm(nk_size_t *count) {
    *count = nk_count_devices_rocm_();
    return *count ? nk_success_k : nk_missing_gpu_k;
}
extern "C" NUMKONG_API nk_status_t nk_capabilities_detected_rocm(nk_size_t ordinal, nk_capability_t *capabilities) {
    return nk_capabilities_detected_rocm_(ordinal, capabilities);
}
extern "C" NUMKONG_API nk_status_t nk_stream_init_rocm(nk_size_t ordinal, nk_stream_t *stream) {
    return nk_stream_init_rocm_(ordinal, stream);
}
extern "C" NUMKONG_API nk_status_t nk_stream_free_rocm(nk_stream_t stream) { return nk_stream_free_rocm_(stream); }

extern "C" NUMKONG_API nk_status_t nk_memory_allocate_unified_rocm(nk_size_t bytes, void **pointer,
                                                                   nk_stream_t stream) {
    return nk_memory_allocate_unified_rocm_(bytes, pointer, stream);
}
extern "C" NUMKONG_API nk_status_t nk_memory_free_unified_rocm(void *pointer, nk_size_t bytes, nk_stream_t stream) {
    nk_unused_(bytes);
    return nk_memory_free_unified_rocm_(pointer, stream);
}
extern "C" NUMKONG_API nk_status_t nk_stream_synchronize_rocm(nk_stream_t stream) {
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

extern "C" void const *nk_probe_cdna3_(void);
extern "C" void const *nk_probe_cdna4_(void);
extern "C" void const *nk_probe_cdna5_(void);

/** Whether the runtime loads @p kernel on the current device, which fails where no code runs. */
static bool nk_loads_rocm_(void const *kernel) {
    hipFuncAttributes attributes;
    bool const loads = hipFuncGetAttributes(&attributes, kernel) == hipSuccess;
    nk_unused_(hipGetLastError());
    return loads;
}

extern "C" NUMKONG_API nk_status_t nk_capabilities_compiled_rocm(nk_size_t ordinal, nk_capability_t *capabilities) {
    int previous = 0;
    *capabilities = 0;
    if (ordinal >= nk_count_devices_rocm_()) return nk_missing_gpu_k;
    if (hipGetDevice(&previous) != hipSuccess || hipSetDevice((int)ordinal) != hipSuccess) {
        nk_unused_(hipGetLastError());
        return nk_device_code_mismatch_k;
    }
    nk_capability_t compiled = nk_cap_rocm_k * nk_loads_rocm_((void const *)&nk_dots_pack_f64_rocm_kernel_);
#if NUMKONG_TARGET_CDNA3
    compiled |= nk_cap_cdna3_k * nk_loads_rocm_(nk_probe_cdna3_());
#endif
#if NUMKONG_TARGET_CDNA4
    compiled |= nk_cap_cdna4_k * nk_loads_rocm_(nk_probe_cdna4_());
#endif
#if NUMKONG_TARGET_CDNA5
    compiled |= nk_cap_cdna5_k * nk_loads_rocm_(nk_probe_cdna5_());
#endif
    nk_unused_(hipSetDevice(previous));
    *capabilities = compiled;
    return nk_success_k;
}
