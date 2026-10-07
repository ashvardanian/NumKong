/**
 *  @file include/numkong/rocm.cuh
 *  @author Ash Vardanian
 *  @date October 3, 2026
 *  @brief The HIP runtime as the library's ROCm host code drives it: the device a call runs on,
 *      managed memory, the launches, the streams and the device producers, then the wavefront and
 *      rounding primitives every ROCm kernel shares.
 *
 *  Only hipcc sees past the guard, and nothing here is shared with CUDA, whose twins live in
 *  `cuda.cuh` under their own names, so a library holding both vendors links one body per name.
 *
 *  @sa include/numkong/cuda.cuh
 */
#ifndef NUMKONG_ROCM_CUH
#define NUMKONG_ROCM_CUH

#include "numkong/capabilities.h" // `nk_cap_rocm_k`

#if NUMKONG_ARCH_ROCM_

#include <string.h> // `strncmp`

#if defined(__cplusplus)
extern "C" {
#endif

/** Makes the stream's device current, retaining the caller's device for restoration. */
NUMKONG_INLINE nk_status_t nk_device_enter_rocm_(void *stream, int *caller) {
    int current = 0, device = 0;
    *caller = -1;
    if (hipGetDevice(&current) != hipSuccess) return nk_missing_gpu_k;
    if (!stream) return nk_success_k;
    if (hipStreamGetDevice((hipStream_t)stream, &device) != hipSuccess) return nk_device_memory_mismatch_k;
    if (device == current) return nk_success_k;
    if (hipSetDevice(device) != hipSuccess) return nk_missing_gpu_k;
    *caller = current;
    return nk_success_k;
}

NUMKONG_INLINE void nk_device_leave_rocm_(int caller) {
    if (caller >= 0) nk_unused_(hipSetDevice(caller));
}

/** Launches @p blocks blocks of @p threads running @p kernel on @p stream, its arguments passed by
 *  address. The runtime's own error stays readable through @c hipGetLastError. */
NUMKONG_INLINE nk_status_t nk_launch_rocm_(void const *kernel, nk_size_t blocks, unsigned threads, void **arguments,
                                           nk_size_t shared_bytes, void *stream) {
    int caller = 0;
    nk_status_t const entered = nk_device_enter_rocm_(stream, &caller);
    if (entered != nk_success_k) return entered;
    dim3 grid, block;
    grid.x = (unsigned)blocks, grid.y = 1, grid.z = 1;
    block.x = threads, block.y = 1, block.z = 1;
    hipError_t const status = hipLaunchKernel(kernel, grid, block, arguments, shared_bytes, (hipStream_t)stream);
    nk_device_leave_rocm_(caller);
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
    int caller = 0;
    nk_status_t const entered = nk_device_enter_rocm_(stream, &caller);
    if (entered != nk_success_k) return entered;
    int device = 0, multiprocessors = 0, per_multiprocessor = 0;
    if (hipGetDevice(&device) != hipSuccess ||
        (shared_ceiling &&
         hipFuncSetAttribute(kernel, hipFuncAttributeMaxDynamicSharedMemorySize, (int)shared_ceiling) != hipSuccess) ||
        hipDeviceGetAttribute(&multiprocessors, hipDeviceAttributeMultiprocessorCount, device) != hipSuccess ||
        hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_multiprocessor, kernel, (int)threads, shared_bytes) !=
            hipSuccess) {
        nk_device_leave_rocm_(caller);
        return nk_device_code_mismatch_k;
    }
    nk_size_t const resident = (nk_size_t)multiprocessors * (nk_size_t)per_multiprocessor;
    nk_size_t const blocks = resident < blocks_wanted ? resident : blocks_wanted;
    void *launch_arguments[1];
    launch_arguments[0] = arguments;
    dim3 grid, block;
    grid.x = (unsigned)blocks, grid.y = 1, grid.z = 1;
    block.x = threads, block.y = 1, block.z = 1;
    hipError_t const status = blocks ? hipLaunchKernel(kernel, grid, block, launch_arguments, shared_bytes,
                                                       (hipStream_t)stream)
                                     : hipErrorInvalidConfiguration;
    nk_device_leave_rocm_(caller);
    return status == hipSuccess ? nk_success_k : nk_device_code_mismatch_k;
}

/** Reads @p attribute of the stream's device into @p value. */
NUMKONG_INLINE nk_status_t nk_device_attribute_rocm_(hipDeviceAttribute_t attribute, int *value, void *stream) {
    int device = 0;
    hipError_t const status = stream ? hipStreamGetDevice((hipStream_t)stream, &device) : hipGetDevice(&device);
    if (status != hipSuccess) return nk_device_code_mismatch_k;
    return hipDeviceGetAttribute(value, attribute, device) == hipSuccess ? nk_success_k : nk_device_code_mismatch_k;
}

/** Copies @p bytes from the device back to @p host once everything queued on @p stream is done. */
NUMKONG_INLINE nk_status_t nk_read_rocm_(void *host, void const *device, nk_size_t bytes, void *stream) {
    int caller = 0;
    nk_status_t const entered = nk_device_enter_rocm_(stream, &caller);
    if (entered != nk_success_k) return entered;
    hipError_t status = hipMemcpyAsync(host, device, bytes, hipMemcpyDeviceToHost, (hipStream_t)stream);
    if (status == hipSuccess) status = hipStreamSynchronize((hipStream_t)stream);
    nk_device_leave_rocm_(caller);
    return status == hipSuccess ? nk_success_k : nk_device_code_mismatch_k;
}

/** Allocates @p bytes of managed memory on the device of @p stream, the current one for a null
 *  stream, leaving the caller's current device as it was. */
NUMKONG_INLINE nk_status_t nk_memory_allocate_unified_rocm_(nk_size_t bytes, void **pointer, void *stream) {
    *pointer = NUMKONG_NULL;
    if (!bytes) return nk_success_k;
    int caller = 0;
    nk_status_t const entered = nk_device_enter_rocm_(stream, &caller);
    if (entered != nk_success_k) return entered;
    hipError_t const allocated = hipMallocManaged(pointer, bytes, hipMemAttachGlobal);
    nk_device_leave_rocm_(caller);
    if (allocated == hipSuccess) return nk_success_k;
    *pointer = NUMKONG_NULL;
    return nk_bad_alloc_k;
}

/** Frees a block of @ref nk_memory_allocate_unified_rocm_ once the device is done with it. */
NUMKONG_INLINE nk_status_t nk_memory_free_unified_rocm_(void *pointer, void *stream) {
    if (!pointer) return nk_success_k;
    int caller = 0;
    nk_status_t const entered = nk_device_enter_rocm_(stream, &caller);
    if (entered != nk_success_k) return entered;
    hipError_t const status = hipFree(pointer);
    nk_device_leave_rocm_(caller);
    return status == hipSuccess ? nk_success_k : nk_device_memory_mismatch_k;
}

/** Waits for everything queued on @p stream. */
NUMKONG_INLINE nk_status_t nk_stream_synchronize_rocm_(void *stream) {
    int caller = 0;
    nk_status_t const entered = nk_device_enter_rocm_(stream, &caller);
    if (entered != nk_success_k) return entered;
    hipError_t const status = hipStreamSynchronize((hipStream_t)stream);
    nk_device_leave_rocm_(caller);
    return status == hipSuccess ? nk_success_k : nk_device_code_mismatch_k;
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
    if (strncmp(name, "gfx942", 6) == 0) detected |= nk_cap_cdna3_k;
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
    int caller = 0;
    nk_status_t const entered = nk_device_enter_rocm_(stream, &caller);
    if (entered != nk_success_k) return entered;
    hipError_t const status = hipStreamDestroy((hipStream_t)stream);
    nk_device_leave_rocm_(caller);
    return status == hipSuccess ? nk_success_k : nk_device_code_mismatch_k;
}

NUMKONG_INLINE void *nk_allocate_unified_rocm_(nk_size_t bytes, void *handle, void *stream) {
    nk_unused_(handle);
    void *pointer = NUMKONG_NULL;
    nk_unused_(nk_memory_allocate_unified_rocm_(bytes, &pointer, stream));
    return pointer;
}

NUMKONG_INLINE void *nk_allocate_device_rocm_(nk_size_t bytes, void *handle, void *stream) {
    nk_unused_(handle);
    void *pointer = NUMKONG_NULL;
    if (!bytes) return NUMKONG_NULL;
    int caller = 0;
    if (nk_device_enter_rocm_(stream, &caller) != nk_success_k) return NUMKONG_NULL;
    if (hipMalloc(&pointer, bytes) != hipSuccess) pointer = NUMKONG_NULL;
    nk_device_leave_rocm_(caller);
    return pointer;
}

NUMKONG_INLINE void *nk_allocate_pinned_rocm_(nk_size_t bytes, void *handle, void *stream) {
    nk_unused_(handle);
    void *pointer = NUMKONG_NULL;
    if (!bytes) return NUMKONG_NULL;
    int caller = 0;
    if (nk_device_enter_rocm_(stream, &caller) != nk_success_k) return NUMKONG_NULL;
    if (hipHostMalloc(&pointer, bytes, hipHostMallocDefault) != hipSuccess) pointer = NUMKONG_NULL;
    nk_device_leave_rocm_(caller);
    return pointer;
}

NUMKONG_INLINE void nk_free_device_rocm_(void *pointer, nk_size_t bytes, void *handle, void *stream) {
    nk_unused_(bytes), nk_unused_(handle);
    nk_unused_(nk_memory_free_unified_rocm_(pointer, stream));
}

NUMKONG_INLINE void nk_free_pinned_rocm_(void *pointer, nk_size_t bytes, void *handle, void *stream) {
    nk_unused_(bytes), nk_unused_(handle);
    int caller = 0;
    if (!pointer || nk_device_enter_rocm_(stream, &caller) != nk_success_k) return;
    nk_unused_(hipHostFree(pointer));
    nk_device_leave_rocm_(caller);
}

NUMKONG_INLINE nk_status_t nk_allocator_init_unified_rocm_(nk_allocator_t *allocator) {
    allocator->allocate = nk_allocate_unified_rocm_;
    allocator->free = nk_free_device_rocm_;
    allocator->handle = NUMKONG_NULL;
    return nk_success_k;
}

NUMKONG_INLINE nk_status_t nk_allocator_init_device_rocm_(nk_allocator_t *allocator) {
    allocator->allocate = nk_allocate_device_rocm_;
    allocator->free = nk_free_device_rocm_;
    allocator->handle = NUMKONG_NULL;
    return nk_success_k;
}

NUMKONG_INLINE nk_status_t nk_allocator_init_pinned_rocm_(nk_allocator_t *allocator) {
    allocator->allocate = nk_allocate_pinned_rocm_;
    allocator->free = nk_free_pinned_rocm_;
    allocator->handle = NUMKONG_NULL;
    return nk_success_k;
}

#pragma region Device Primitives

/** Lanes of a wavefront, 64 on CDNA and 32 on RDNA and MI400, which a build spanning both only
 *  learns per device pass. */
NUMKONG_DEVICE unsigned nk_warp_lanes_rocm_(void) { return __builtin_amdgcn_wavefrontsize(); }

/** Lane `lane ^ offset`'s @p value, from across the whole wavefront, where offsets below 32 stay
 *  inside each half of a 64-lane one. */
NUMKONG_DEVICE nk_u32_t nk_shuffle_xor_u32_rocm_(nk_u32_t value, unsigned offset) { return __shfl_xor(value, offset); }
NUMKONG_DEVICE nk_i32_t nk_shuffle_xor_i32_rocm_(nk_i32_t value, unsigned offset) { return __shfl_xor(value, offset); }
NUMKONG_DEVICE nk_u64_t nk_shuffle_xor_u64_rocm_(nk_u64_t value, unsigned offset) {
    return __shfl_xor((unsigned long long)value, offset);
}
NUMKONG_DEVICE nk_f32_t nk_shuffle_xor_f32_rocm_(nk_f32_t value, unsigned offset) { return __shfl_xor(value, offset); }
NUMKONG_DEVICE nk_f64_t nk_shuffle_xor_f64_rocm_(nk_f64_t value, unsigned offset) { return __shfl_xor(value, offset); }

/** Lane `lane - offset`'s @p value within each group of 32 lanes, a lane's own below @p offset. */
NUMKONG_DEVICE nk_u64_t nk_shuffle_up_u64_rocm_(nk_u64_t value, unsigned offset) {
    return __shfl_up((unsigned long long)value, offset, 32);
}

/*  Arithmetic rounded one operation at a time, which HIP-Clang never contracts into an FMA that
 *  would break compensated sums' error terms, as its own @c __dmul_rn may be. */
NUMKONG_DEVICE nk_f64_t nk_f64_add_rn_rocm_(nk_f64_t a, nk_f64_t b) {
#pragma clang fp contract(off)
    return a + b;
}
NUMKONG_DEVICE nk_f64_t nk_f64_sub_rn_rocm_(nk_f64_t a, nk_f64_t b) {
#pragma clang fp contract(off)
    return a - b;
}
NUMKONG_DEVICE nk_f64_t nk_f64_mul_rn_rocm_(nk_f64_t a, nk_f64_t b) {
#pragma clang fp contract(off)
    return a * b;
}
NUMKONG_DEVICE nk_f32_t nk_f32_mul_rn_rocm_(nk_f32_t a, nk_f32_t b) {
#pragma clang fp contract(off)
    return a * b;
}

/** 2^x through @c exp2f , and 0 for −∞. */
NUMKONG_DEVICE nk_f32_t nk_f32_exp2_rocm_(nk_f32_t exponent) { return exp2f(exponent); }

#pragma endregion Device Primitives

/*  The library defines these once, in `c/target/rocm.hip`; header-only builds define them here. */
#if NUMKONG_HEADER_ONLY && NUMKONG_TARGET_ROCM

NUMKONG_API nk_status_t nk_allocator_init_unified_rocm(nk_allocator_t *allocator) {
    return nk_allocator_init_unified_rocm_(allocator);
}
NUMKONG_API nk_status_t nk_allocator_init_device_rocm(nk_allocator_t *allocator) {
    return nk_allocator_init_device_rocm_(allocator);
}
NUMKONG_API nk_status_t nk_allocator_init_pinned_rocm(nk_allocator_t *allocator) {
    return nk_allocator_init_pinned_rocm_(allocator);
}

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
    nk_unused_(bytes);
    return nk_memory_free_unified_rocm_(pointer, stream);
}
NUMKONG_API nk_status_t nk_stream_synchronize_rocm(void *stream) { return nk_stream_synchronize_rocm_(stream); }

#endif // NUMKONG_HEADER_ONLY && NUMKONG_TARGET_ROCM

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_ROCM_
#endif // NUMKONG_ROCM_CUH
