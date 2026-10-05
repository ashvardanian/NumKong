/**
 *  @file include/numkong/memory.h
 *  @author Ash Vardanian
 *  @date October 3, 2026
 *  @brief Host and device memory allocation.
 */
#ifndef NUMKONG_MEMORY_H
#define NUMKONG_MEMORY_H

#include "numkong/capabilities.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Heap allocations with power-of-two @p alignment; invalid alignment preserves the policy. */
NUMKONG_INLINE nk_status_t nk_allocator_init_heap(nk_allocator_t *allocator, nk_size_t alignment);

/** A fixed-capacity arena; invalid storage or alignment preserves both the policy and buffer. */
NUMKONG_INLINE nk_status_t nk_allocator_init_arena(nk_allocator_t *allocator, void *buffer, nk_size_t bytes,
                                                   nk_size_t alignment);

/** Selects host/device shared storage; failures preserve the allocator. */
NUMKONG_API nk_status_t nk_allocator_init_unified_best(nk_allocator_t *allocator, nk_capability_t capabilities);

/** Selects device-local storage, or shared storage on Metal. CPU masks have no device allocator. */
NUMKONG_API nk_status_t nk_allocator_init_device_best(nk_allocator_t *allocator, nk_capability_t capabilities);

/** Selects pinned host storage on CUDA/ROCm or shared storage on Metal; unsupported on CPU. */
NUMKONG_API nk_status_t nk_allocator_init_pinned_best(nk_allocator_t *allocator, nk_capability_t capabilities);

/** @copydoc nk_allocator_init_unified_best */
NUMKONG_API nk_status_t nk_allocator_init_unified_serial(nk_allocator_t *allocator);

#if NUMKONG_TARGET_CUDA
/** @copydoc nk_allocator_init_unified_best */
NUMKONG_API nk_status_t nk_allocator_init_unified_cuda(nk_allocator_t *allocator);
/** @copydoc nk_allocator_init_device_best */
NUMKONG_API nk_status_t nk_allocator_init_device_cuda(nk_allocator_t *allocator);
/** @copydoc nk_allocator_init_pinned_best */
NUMKONG_API nk_status_t nk_allocator_init_pinned_cuda(nk_allocator_t *allocator);
#endif

#if NUMKONG_TARGET_ROCM
/** @copydoc nk_allocator_init_unified_best */
NUMKONG_API nk_status_t nk_allocator_init_unified_rocm(nk_allocator_t *allocator);
/** @copydoc nk_allocator_init_device_best */
NUMKONG_API nk_status_t nk_allocator_init_device_rocm(nk_allocator_t *allocator);
/** @copydoc nk_allocator_init_pinned_best */
NUMKONG_API nk_status_t nk_allocator_init_pinned_rocm(nk_allocator_t *allocator);
#endif

#if NUMKONG_WITH_METAL
/** @copydoc nk_allocator_init_unified_best */
NUMKONG_API nk_status_t nk_allocator_init_unified_metal(nk_allocator_t *allocator);
#endif

/**
 *  @brief Allocates @p bytes that the host and the device of @p stream both address.
 *  @param[in] bytes The size of the block, where zero hands out a null one.
 *  @param[out] pointer The block, or null on failure.
 *      CPU storage uses @c nk_default_alignment_k alignment.
 *  @param[in] capabilities The group to allocate for, like @c nk_cuda_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or a stream of that group, naming the device.
 *  @return @c nk_success_k, @c nk_bad_alloc_k, or @c nk_missing_gpu_k for a group without a device
 *      or missing from this build.
 *
 *  CUDA and ROCm hand out managed memory, and Metal a shared buffer every kernel on that device
 *  binds, so the host reads what a kernel wrote once the stream is synchronized.
 */
NUMKONG_API nk_status_t nk_memory_allocate_unified_best(nk_size_t bytes, void **pointer, nk_capability_t capabilities,
                                                        void *stream);

/**
 *  @brief Returns a block of @ref nk_memory_allocate_unified_best once the work queued on
 *      @p stream is done with it.
 *  @param[in] pointer The block, or null for none.
 *  @param[in] bytes The size it was allocated with.
 *  @return @c nk_success_k, @c nk_device_memory_mismatch_k for a block the group never handed out,
 *      or @c nk_missing_gpu_k.
 *
 *  CUDA and ROCm wait for the device first, and Metal releases the buffer once the work committed
 *  to @p stream so far completes.
 */
NUMKONG_API nk_status_t nk_memory_free_unified_best(void *pointer, nk_size_t bytes, nk_capability_t capabilities,
                                                    void *stream);

/** @copydoc nk_memory_allocate_unified_best */
NUMKONG_API nk_status_t nk_memory_allocate_unified_serial(nk_size_t bytes, void **pointer, void *stream);
/** @copydoc nk_memory_free_unified_best */
NUMKONG_API nk_status_t nk_memory_free_unified_serial(void *pointer, nk_size_t bytes, void *stream);

#if NUMKONG_TARGET_CUDA
/** @copydoc nk_memory_allocate_unified_best */
NUMKONG_API nk_status_t nk_memory_allocate_unified_cuda(nk_size_t bytes, void **pointer, void *stream);
/** @copydoc nk_memory_free_unified_best */
NUMKONG_API nk_status_t nk_memory_free_unified_cuda(void *pointer, nk_size_t bytes, void *stream);
#endif

#if NUMKONG_TARGET_ROCM
/** @copydoc nk_memory_allocate_unified_best */
NUMKONG_API nk_status_t nk_memory_allocate_unified_rocm(nk_size_t bytes, void **pointer, void *stream);
/** @copydoc nk_memory_free_unified_best */
NUMKONG_API nk_status_t nk_memory_free_unified_rocm(void *pointer, nk_size_t bytes, void *stream);
#endif

#if NUMKONG_WITH_METAL
/** @copydoc nk_memory_allocate_unified_best */
NUMKONG_API nk_status_t nk_memory_allocate_unified_metal(nk_size_t bytes, void **pointer, void *stream);
/** @copydoc nk_memory_free_unified_best */
NUMKONG_API nk_status_t nk_memory_free_unified_metal(void *pointer, nk_size_t bytes, void *stream);
#endif

#if NUMKONG_HEADER_ONLY

NUMKONG_API nk_status_t nk_allocator_init_unified_best(nk_allocator_t *allocator, nk_capability_t capabilities) {
    nk_unused_(allocator), nk_unused_(capabilities);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_allocator_init_device_best(nk_allocator_t *allocator, nk_capability_t capabilities) {
    nk_unused_(allocator), nk_unused_(capabilities);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_allocator_init_pinned_best(nk_allocator_t *allocator, nk_capability_t capabilities) {
    nk_unused_(allocator), nk_unused_(capabilities);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_memory_allocate_unified_best(nk_size_t bytes, void **pointer, nk_capability_t capabilities,
                                                        void *stream) {
    nk_unused_(bytes), nk_unused_(capabilities), nk_unused_(stream);
    *pointer = NUMKONG_NULL;
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_memory_free_unified_best(void *pointer, nk_size_t bytes, nk_capability_t capabilities,
                                                    void *stream) {
    nk_unused_(pointer), nk_unused_(bytes), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

#endif

#ifdef __cplusplus
}
#endif

#include "numkong/memory/serial.h"

#endif // NUMKONG_MEMORY_H
