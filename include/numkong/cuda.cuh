/**
 *  @file include/numkong/cuda.cuh
 *  @author Ash Vardanian
 *  @date October 3, 2026
 *  @brief The CUDA runtime as the library's CUDA host code drives it: the device a call runs on,
 *      managed memory, the launches, the streams and the device producers.
 *
 *  Only nvcc sees past the guard, and nothing here is shared with ROCm, whose twins live in
 *  `rocm.cuh` under their own names, so a library holding both vendors links one body per name.
 *
 *  @sa include/numkong/rocm.cuh
 */
#ifndef NUMKONG_CUDA_CUH
#define NUMKONG_CUDA_CUH

#include "numkong/capabilities.h" // `nk_cap_cuda_k`

#if NUMKONG_ARCH_CUDA_ && defined(__CUDACC__) && !defined(__HIP__)

#if defined(__cplusplus)
extern "C" {
#endif

/** Makes the stream's device current, retaining the caller's device for restoration. */
NUMKONG_INLINE nk_status_t nk_device_enter_cuda_(void *stream, int *caller) {
    int device = 0;
    if (cudaGetDevice(caller) != cudaSuccess) return nk_missing_gpu_k;
    if (!stream) return nk_success_k;
    if (cudaStreamGetDevice((cudaStream_t)stream, &device) != cudaSuccess) return nk_device_memory_mismatch_k;
    if (device == *caller) return nk_success_k;
    return cudaSetDevice(device) == cudaSuccess ? nk_success_k : nk_missing_gpu_k;
}

NUMKONG_INLINE void nk_device_leave_cuda_(int caller) { nk_unused_(cudaSetDevice(caller)); }

/** Launches @p blocks blocks of @p threads running @p kernel on @p stream, its arguments passed by
 *  address. The runtime's own error stays readable through @c cudaGetLastError. */
NUMKONG_INLINE nk_status_t nk_launch_cuda_(void const *kernel, nk_size_t blocks, unsigned threads, void **arguments,
                                           nk_size_t shared_bytes, void *stream) {
    int caller = 0;
    nk_status_t const entered = nk_device_enter_cuda_(stream, &caller);
    if (entered != nk_success_k) return entered;
    dim3 grid, block;
    grid.x = (unsigned)blocks, grid.y = 1, grid.z = 1;
    block.x = threads, block.y = 1, block.z = 1;
    cudaError_t const status = cudaLaunchKernel(kernel, grid, block, arguments, shared_bytes, (cudaStream_t)stream);
    nk_device_leave_cuda_(caller);
    return status == cudaSuccess ? nk_success_k : nk_device_code_mismatch_k;
}

/**
 *  @brief Launches as many blocks of @p kernel as stay resident across the current device, at most
 *      @p blocks_wanted, passing the one argument struct at @p arguments by value.
 *  @param[in] shared_bytes Dynamic shared memory of this launch.
 *  @param[in] shared_ceiling Dynamic shared memory the kernel may take at any depth, or zero for
 *      the runtime's default ceiling.
 */
NUMKONG_INLINE nk_status_t nk_launch_resident_cuda_(void const *kernel, unsigned threads, nk_size_t shared_bytes,
                                                    nk_size_t shared_ceiling, nk_size_t blocks_wanted, void *arguments,
                                                    void *stream) {
    int caller = 0;
    nk_status_t const entered = nk_device_enter_cuda_(stream, &caller);
    if (entered != nk_success_k) return entered;
    int device = 0, multiprocessors = 0, per_multiprocessor = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        (shared_ceiling && cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                (int)shared_ceiling) != cudaSuccess) ||
        cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device) != cudaSuccess ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_multiprocessor, kernel, (int)threads, shared_bytes) !=
            cudaSuccess) {
        nk_device_leave_cuda_(caller);
        return nk_device_code_mismatch_k;
    }
    nk_size_t const blocks = (nk_size_t)multiprocessors * (nk_size_t)per_multiprocessor;
    if (blocks == 0) {
        nk_device_leave_cuda_(caller);
        return nk_device_code_mismatch_k;
    }
    void *launch_arguments[1];
    launch_arguments[0] = arguments;
    nk_status_t const launched = nk_launch_cuda_(kernel, blocks < blocks_wanted ? blocks : blocks_wanted, threads,
                                                 launch_arguments, shared_bytes, stream);
    nk_device_leave_cuda_(caller);
    return launched;
}

/** Reads @p attribute of the stream's device into @p value. */
NUMKONG_INLINE nk_status_t nk_device_attribute_cuda_(enum cudaDeviceAttr attribute, int *value, void *stream) {
    int device = 0;
    cudaError_t const status = stream ? cudaStreamGetDevice((cudaStream_t)stream, &device) : cudaGetDevice(&device);
    if (status != cudaSuccess) return nk_device_code_mismatch_k;
    return cudaDeviceGetAttribute(value, attribute, device) == cudaSuccess ? nk_success_k : nk_device_code_mismatch_k;
}

/** Copies @p bytes from the device back to @p host once everything queued on @p stream is done. */
NUMKONG_INLINE nk_status_t nk_read_cuda_(void *host, void const *device, nk_size_t bytes, void *stream) {
    int caller = 0;
    nk_status_t const entered = nk_device_enter_cuda_(stream, &caller);
    if (entered != nk_success_k) return entered;
    cudaError_t status = cudaMemcpyAsync(host, device, bytes, cudaMemcpyDeviceToHost, (cudaStream_t)stream);
    if (status == cudaSuccess) status = cudaStreamSynchronize((cudaStream_t)stream);
    nk_device_leave_cuda_(caller);
    return status == cudaSuccess ? nk_success_k : nk_device_code_mismatch_k;
}

/** Allocates @p bytes of managed memory on the device of @p stream, the current one for a null
 *  stream, leaving the caller's current device as it was. */
NUMKONG_INLINE nk_status_t nk_memory_allocate_unified_cuda_(nk_size_t bytes, void **pointer, void *stream) {
    *pointer = NUMKONG_NULL;
    if (!bytes) return nk_success_k;
    int caller = 0;
    nk_status_t const entered = nk_device_enter_cuda_(stream, &caller);
    if (entered != nk_success_k) return entered;
    cudaError_t const allocated = cudaMallocManaged(pointer, bytes, cudaMemAttachGlobal);
    nk_device_leave_cuda_(caller);
    if (allocated == cudaSuccess) return nk_success_k;
    *pointer = NUMKONG_NULL;
    return nk_bad_alloc_k;
}

/** Frees a block of @ref nk_memory_allocate_unified_cuda_ once the device is done with it. */
NUMKONG_INLINE nk_status_t nk_memory_free_unified_cuda_(void *pointer, void *stream) {
    if (!pointer) return nk_success_k;
    int caller = 0;
    nk_status_t const entered = nk_device_enter_cuda_(stream, &caller);
    if (entered != nk_success_k) return entered;
    cudaError_t const status = cudaFree(pointer);
    nk_device_leave_cuda_(caller);
    return status == cudaSuccess ? nk_success_k : nk_device_memory_mismatch_k;
}

/** Waits for everything queued on @p stream. */
NUMKONG_INLINE nk_status_t nk_stream_synchronize_cuda_(void *stream) {
    int caller = 0;
    nk_status_t const entered = nk_device_enter_cuda_(stream, &caller);
    if (entered != nk_success_k) return entered;
    cudaError_t const status = cudaStreamSynchronize((cudaStream_t)stream);
    nk_device_leave_cuda_(caller);
    return status == cudaSuccess ? nk_success_k : nk_device_code_mismatch_k;
}

/** How many CUDA devices the runtime sees, or zero. */
NUMKONG_INLINE nk_size_t nk_cuda_count_devices_(void) {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess ? (nk_size_t)count : 0;
}

/** The capabilities CUDA device @p ordinal runs, by the runtime's own ordinal. */
NUMKONG_INLINE nk_status_t nk_cuda_capabilities_detected_(nk_size_t ordinal, nk_capability_t *capabilities) {
    *capabilities = 0;
    if (ordinal >= nk_cuda_count_devices_()) return nk_missing_gpu_k;
    int major = 0, minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, (int)ordinal) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, (int)ordinal) != cudaSuccess)
        return nk_device_code_mismatch_k;
    nk_capability_t detected = nk_cap_cuda_k;
    if (major * 10 + minor >= 80) detected |= nk_cap_ampere_k;
    if (major * 10 + minor >= 89) detected |= nk_cap_ada_k;
    if (major == 9) detected |= nk_cap_hopper_k;
    if (major == 10) detected |= nk_cap_blackwell_k;
    if (major == 12) detected |= nk_cap_blackwellrtx_k;
    *capabilities = detected;
    return nk_success_k;
}

/** Creates a stream on CUDA device @p ordinal, leaving the caller's current device as it was. */
NUMKONG_INLINE nk_status_t nk_cuda_stream_init_(nk_size_t ordinal, void **stream) {
    *stream = NUMKONG_NULL;
    if (ordinal >= nk_cuda_count_devices_()) return nk_missing_gpu_k;
    int caller = 0;
    cudaStream_t created = 0;
    if (cudaGetDevice(&caller) != cudaSuccess || cudaSetDevice((int)ordinal) != cudaSuccess) return nk_missing_gpu_k;
    cudaError_t const error = cudaStreamCreate(&created);
    nk_unused_(cudaSetDevice(caller));
    if (error != cudaSuccess) return nk_bad_alloc_k;
    *stream = (void *)created;
    return nk_success_k;
}

/** Destroys @p stream once the work queued on it completes; a null stream is the default one. */
NUMKONG_INLINE nk_status_t nk_cuda_stream_free_(void *stream) {
    if (!stream) return nk_success_k;
    int caller = 0;
    nk_status_t const entered = nk_device_enter_cuda_(stream, &caller);
    if (entered != nk_success_k) return entered;
    cudaError_t const status = cudaStreamDestroy((cudaStream_t)stream);
    nk_device_leave_cuda_(caller);
    return status == cudaSuccess ? nk_success_k : nk_device_code_mismatch_k;
}

NUMKONG_INLINE void *nk_allocate_unified_cuda_(nk_size_t bytes, void *handle, void *stream) {
    nk_unused_(handle);
    void *pointer = NUMKONG_NULL;
    nk_unused_(nk_memory_allocate_unified_cuda_(bytes, &pointer, stream));
    return pointer;
}

NUMKONG_INLINE void *nk_allocate_device_cuda_(nk_size_t bytes, void *handle, void *stream) {
    nk_unused_(handle);
    void *pointer = NUMKONG_NULL;
    if (!bytes) return NUMKONG_NULL;
    int caller = 0;
    if (nk_device_enter_cuda_(stream, &caller) != nk_success_k) return NUMKONG_NULL;
    if (cudaMalloc(&pointer, bytes) != cudaSuccess) pointer = NUMKONG_NULL;
    nk_device_leave_cuda_(caller);
    return pointer;
}

NUMKONG_INLINE void *nk_allocate_pinned_cuda_(nk_size_t bytes, void *handle, void *stream) {
    nk_unused_(handle);
    void *pointer = NUMKONG_NULL;
    if (!bytes) return NUMKONG_NULL;
    int caller = 0;
    if (nk_device_enter_cuda_(stream, &caller) != nk_success_k) return NUMKONG_NULL;
    if (cudaHostAlloc(&pointer, bytes, cudaHostAllocDefault) != cudaSuccess) pointer = NUMKONG_NULL;
    nk_device_leave_cuda_(caller);
    return pointer;
}

NUMKONG_INLINE void nk_free_device_cuda_(void *pointer, nk_size_t bytes, void *handle, void *stream) {
    nk_unused_(bytes), nk_unused_(handle);
    nk_unused_(nk_memory_free_unified_cuda_(pointer, stream));
}

NUMKONG_INLINE void nk_free_pinned_cuda_(void *pointer, nk_size_t bytes, void *handle, void *stream) {
    nk_unused_(bytes), nk_unused_(handle);
    int caller = 0;
    if (!pointer || nk_device_enter_cuda_(stream, &caller) != nk_success_k) return;
    nk_unused_(cudaFreeHost(pointer));
    nk_device_leave_cuda_(caller);
}

NUMKONG_INLINE nk_status_t nk_allocator_init_unified_cuda_(nk_allocator_t *allocator) {
    allocator->allocate = nk_allocate_unified_cuda_;
    allocator->free = nk_free_device_cuda_;
    allocator->handle = NUMKONG_NULL;
    return nk_success_k;
}

NUMKONG_INLINE nk_status_t nk_allocator_init_device_cuda_(nk_allocator_t *allocator) {
    allocator->allocate = nk_allocate_device_cuda_;
    allocator->free = nk_free_device_cuda_;
    allocator->handle = NUMKONG_NULL;
    return nk_success_k;
}

NUMKONG_INLINE nk_status_t nk_allocator_init_pinned_cuda_(nk_allocator_t *allocator) {
    allocator->allocate = nk_allocate_pinned_cuda_;
    allocator->free = nk_free_pinned_cuda_;
    allocator->handle = NUMKONG_NULL;
    return nk_success_k;
}

/*  The library defines these once, in `c/target/cuda.cu`; header-only builds define them here. */
#if NUMKONG_HEADER_ONLY && NUMKONG_TARGET_CUDA

NUMKONG_API nk_status_t nk_allocator_init_unified_cuda(nk_allocator_t *allocator) {
    return nk_allocator_init_unified_cuda_(allocator);
}
NUMKONG_API nk_status_t nk_allocator_init_device_cuda(nk_allocator_t *allocator) {
    return nk_allocator_init_device_cuda_(allocator);
}
NUMKONG_API nk_status_t nk_allocator_init_pinned_cuda(nk_allocator_t *allocator) {
    return nk_allocator_init_pinned_cuda_(allocator);
}

NUMKONG_API nk_status_t nk_cuda_count_devices(nk_size_t *count) {
    *count = nk_cuda_count_devices_();
    return *count ? nk_success_k : nk_missing_gpu_k;
}
NUMKONG_API nk_status_t nk_cuda_capabilities_detected(nk_size_t ordinal, nk_capability_t *capabilities) {
    return nk_cuda_capabilities_detected_(ordinal, capabilities);
}
NUMKONG_API nk_status_t nk_cuda_stream_init(nk_size_t ordinal, void **stream) {
    return nk_cuda_stream_init_(ordinal, stream);
}
NUMKONG_API nk_status_t nk_cuda_stream_free(void *stream) { return nk_cuda_stream_free_(stream); }
NUMKONG_API nk_status_t nk_memory_allocate_unified_cuda(nk_size_t bytes, void **pointer, void *stream) {
    return nk_memory_allocate_unified_cuda_(bytes, pointer, stream);
}
NUMKONG_API nk_status_t nk_memory_free_unified_cuda(void *pointer, nk_size_t bytes, void *stream) {
    nk_unused_(bytes);
    return nk_memory_free_unified_cuda_(pointer, stream);
}
NUMKONG_API nk_status_t nk_stream_synchronize_cuda(void *stream) { return nk_stream_synchronize_cuda_(stream); }

#endif // NUMKONG_HEADER_ONLY && NUMKONG_TARGET_CUDA

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_ && defined(__CUDACC__) && !defined(__HIP__)
#endif // NUMKONG_CUDA_CUH
