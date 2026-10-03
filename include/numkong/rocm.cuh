/**
 *  @file include/numkong/rocm.cuh
 *  @author Ash Vardanian
 *  @date October 3, 2026
 *  @brief The HIP runtime as the library's ROCm host code drives it: the device a call runs on,
 *      managed memory, the launches, the streams and the device producers.
 *
 *  Only hipcc sees past the guard, and nothing here is shared with CUDA, whose twins live in
 *  `cuda.cuh` under their own names, so a library holding both vendors links one body per name.
 *
 *  @sa include/numkong/cuda.cuh
 */
#ifndef NUMKONG_ROCM_CUH
#define NUMKONG_ROCM_CUH

#include "numkong/capabilities.h" // `nk_cap_rocm_k`

#if NUMKONG_ARCH_ROCM_ && defined(__HIP__)

#include <string.h> // `strncmp`

#if defined(__cplusplus)
extern "C" {
#endif

/** The runtime's current device ordinal. */
NUMKONG_INLINE nk_status_t nk_device_current_rocm_(int *device) {
    return hipGetDevice(device) == hipSuccess ? nk_success_k : nk_device_code_mismatch_k;
}

/** Launches @p blocks blocks of @p threads running @p kernel on @p stream, its arguments passed by
 *  address. The runtime's own error stays readable through @c hipGetLastError. */
NUMKONG_INLINE nk_status_t nk_launch_rocm_(void const *kernel, nk_size_t blocks, unsigned threads, void **arguments,
                                           nk_size_t shared_bytes, void *stream) {
    dim3 grid, block;
    grid.x = (unsigned)blocks, grid.y = 1, grid.z = 1;
    block.x = threads, block.y = 1, block.z = 1;
    hipError_t const status = hipLaunchKernel(kernel, grid, block, arguments, shared_bytes, (hipStream_t)stream);
    return status == hipSuccess ? nk_success_k : nk_device_code_mismatch_k;
}

/**
 *  @brief Launches as many blocks of @p kernel as stay resident across the current device, at most
 *      @p blocks_wanted, passing the one argument struct at @p arguments by value.
 *  @param[in] shared_bytes Dynamic shared memory of this launch.
 *  @param[in] shared_ceiling Dynamic shared memory the kernel may take at any depth, or zero for
 *      the runtime's default ceiling.
 */
NUMKONG_INLINE nk_status_t nk_launch_resident_rocm_(void const *kernel, unsigned threads, nk_size_t shared_bytes,
                                                    nk_size_t shared_ceiling, nk_size_t blocks_wanted, void *arguments,
                                                    void *stream) {
    int device = 0, multiprocessors = 0, per_multiprocessor = 0;
    nk_status_t const status = nk_device_current_rocm_(&device);
    if (status != nk_success_k) return status;
    if ((shared_ceiling &&
         hipFuncSetAttribute(kernel, hipFuncAttributeMaxDynamicSharedMemorySize, (int)shared_ceiling) != hipSuccess) ||
        hipDeviceGetAttribute(&multiprocessors, hipDeviceAttributeMultiprocessorCount, device) != hipSuccess ||
        hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_multiprocessor, kernel, (int)threads, shared_bytes) !=
            hipSuccess)
        return nk_device_code_mismatch_k;
    nk_size_t const blocks = (nk_size_t)multiprocessors * (nk_size_t)per_multiprocessor;
    if (blocks == 0) return nk_device_code_mismatch_k;
    void *launch_arguments[1];
    launch_arguments[0] = arguments;
    return nk_launch_rocm_(kernel, blocks < blocks_wanted ? blocks : blocks_wanted, threads, launch_arguments,
                           shared_bytes, stream);
}

/** Reads @p attribute of the current device into @p value. */
NUMKONG_INLINE nk_status_t nk_device_attribute_rocm_(hipDeviceAttribute_t attribute, int *value) {
    int device = 0;
    nk_status_t const status = nk_device_current_rocm_(&device);
    if (status != nk_success_k) return status;
    return hipDeviceGetAttribute(value, attribute, device) == hipSuccess ? nk_success_k : nk_device_code_mismatch_k;
}

/** Copies @p bytes from the device back to @p host once everything queued on @p stream is done. */
NUMKONG_INLINE nk_status_t nk_read_rocm_(void *host, void const *device, nk_size_t bytes, void *stream) {
    hipError_t status = hipMemcpyAsync(host, device, bytes, hipMemcpyDeviceToHost, (hipStream_t)stream);
    if (status == hipSuccess) status = hipStreamSynchronize((hipStream_t)stream);
    return status == hipSuccess ? nk_success_k : nk_device_code_mismatch_k;
}

/** Allocates @p bytes of managed memory on the device of @p stream, the current one for a null
 *  stream, leaving the caller's current device as it was. */
NUMKONG_INLINE nk_status_t nk_memory_allocate_unified_rocm_(nk_size_t bytes, void **pointer, void *stream) {
    *pointer = NUMKONG_NULL;
    if (!bytes) return nk_success_k;
    int current = 0;
    if (nk_device_current_rocm_(&current) != nk_success_k) return nk_missing_gpu_k;
    int device = current;
    // Managed memory belongs to the current device's context, so switch to the stream's device
    if (stream && hipStreamGetDevice((hipStream_t)stream, &device) != hipSuccess) return nk_device_code_mismatch_k;
    if (device != current && hipSetDevice(device) != hipSuccess) return nk_device_code_mismatch_k;
    hipError_t const allocated = hipMallocManaged(pointer, bytes, hipMemAttachGlobal);
    if (device != current) nk_unused_(hipSetDevice(current));
    if (allocated == hipSuccess) return nk_success_k;
    *pointer = NUMKONG_NULL;
    return nk_bad_alloc_k;
}

/** Frees a block of @ref nk_memory_allocate_unified_rocm_ once the device is done with it. */
NUMKONG_INLINE nk_status_t nk_memory_free_unified_rocm_(void *pointer) {
    return hipFree(pointer) == hipSuccess ? nk_success_k : nk_device_memory_mismatch_k;
}

/** Waits for everything queued on @p stream. */
NUMKONG_INLINE nk_status_t nk_stream_synchronize_rocm_(void *stream) {
    return hipStreamSynchronize((hipStream_t)stream) == hipSuccess ? nk_success_k : nk_device_code_mismatch_k;
}

/** How many ROCm devices the runtime sees, or zero. */
NUMKONG_INLINE nk_size_t nk_rocm_count_devices_(void) {
    int count = 0;
    return hipGetDeviceCount(&count) == hipSuccess ? (nk_size_t)count : 0;
}

/** The capabilities ROCm device @p ordinal runs, by the runtime's own ordinal. */
NUMKONG_INLINE nk_status_t nk_rocm_capabilities_detected_(nk_size_t ordinal, nk_capability_t *capabilities) {
    *capabilities = 0;
    if (ordinal >= nk_rocm_count_devices_()) return nk_missing_gpu_k;
    hipDeviceProp_t properties;
    if (hipGetDeviceProperties(&properties, (int)ordinal) != hipSuccess) return nk_device_code_mismatch_k;
    char const *const name = properties.gcnArchName;
    nk_capability_t detected = nk_cap_rocm_k;
    if (strncmp(name, "gfx950", 6) == 0) detected |= nk_cap_cdna4_k;
    if (strncmp(name, "gfx1250", 7) == 0 || strncmp(name, "gfx1251", 7) == 0) detected |= nk_cap_cdna5_k;
    *capabilities = detected;
    return nk_success_k;
}

/** Creates a stream on ROCm device @p ordinal, leaving the caller's current device as it was. */
NUMKONG_INLINE nk_status_t nk_rocm_stream_init_(nk_size_t ordinal, void **stream) {
    *stream = NUMKONG_NULL;
    if (ordinal >= nk_rocm_count_devices_()) return nk_missing_gpu_k;
    int caller = 0;
    hipStream_t created = 0;
    if (hipGetDevice(&caller) != hipSuccess || hipSetDevice((int)ordinal) != hipSuccess) return nk_missing_gpu_k;
    hipError_t const error = hipStreamCreate(&created);
    nk_unused_(hipSetDevice(caller));
    if (error != hipSuccess) return nk_bad_alloc_k;
    *stream = (void *)created;
    return nk_success_k;
}

/** Destroys @p stream once the work queued on it completes; a null stream is the default one. */
NUMKONG_INLINE nk_status_t nk_rocm_stream_free_(void *stream) {
    if (!stream) return nk_success_k;
    return hipStreamDestroy((hipStream_t)stream) == hipSuccess ? nk_success_k : nk_device_code_mismatch_k;
}

/*  The library defines these once, in `c/target/rocm.hip`; header-only builds define them here. */
#if NUMKONG_HEADER_ONLY && NUMKONG_TARGET_ROCM

NUMKONG_API nk_status_t nk_rocm_count_devices(nk_size_t *count) {
    *count = nk_rocm_count_devices_();
    return *count ? nk_success_k : nk_missing_gpu_k;
}
NUMKONG_API nk_status_t nk_rocm_capabilities_detected(nk_size_t ordinal, nk_capability_t *capabilities) {
    return nk_rocm_capabilities_detected_(ordinal, capabilities);
}
NUMKONG_API nk_status_t nk_rocm_stream_init(nk_size_t ordinal, void **stream) {
    return nk_rocm_stream_init_(ordinal, stream);
}
NUMKONG_API nk_status_t nk_rocm_stream_free(void *stream) { return nk_rocm_stream_free_(stream); }
NUMKONG_API nk_status_t nk_memory_allocate_unified_rocm(nk_size_t bytes, void **pointer, void *stream) {
    return nk_memory_allocate_unified_rocm_(bytes, pointer, stream);
}
NUMKONG_API nk_status_t nk_memory_free_unified_rocm(void *pointer, nk_size_t bytes, void *stream) {
    nk_unused_(bytes), nk_unused_(stream);
    return nk_memory_free_unified_rocm_(pointer);
}
NUMKONG_API nk_status_t nk_stream_synchronize_rocm(void *stream) { return nk_stream_synchronize_rocm_(stream); }

#endif // NUMKONG_HEADER_ONLY && NUMKONG_TARGET_ROCM

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_ROCM_ && defined(__HIP__)
#endif // NUMKONG_ROCM_CUH
