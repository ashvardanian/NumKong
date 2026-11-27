/**
 *  @file c/numkong/memory.h
 *  @author Ash Vardanian
 *  @date October 3, 2026
 *  @brief Host and device memory allocation.
 */
#ifndef NUMKONG_MEMORY_H
#define NUMKONG_MEMORY_H

#include <stdlib.h> // `malloc`, `free`

#include "numkong/capabilities.h"

#ifdef __cplusplus
extern "C" {
#endif

NUMKONG_INLINE void *nk_allocate_heap_(nk_size_t bytes, void *handle, nk_stream_t stream) {
    nk_unused_(stream);
    nk_size_t const alignment = (nk_size_t)handle;
    if (!bytes || bytes > NUMKONG_SIZE_MAX - alignment - sizeof(void *)) return NUMKONG_NULL;
    void *allocation = malloc(bytes + alignment + sizeof(void *));
    if (!allocation) return NUMKONG_NULL;
    nk_size_t const address = ((nk_size_t)allocation + sizeof(void *) + alignment - 1) & ~(alignment - 1);
    void **allocations = (void **)address;
    allocations[-1] = allocation;
    return (void *)address;
}

NUMKONG_INLINE void nk_free_heap_(void *pointer, nk_size_t bytes, void *handle, nk_stream_t stream) {
    nk_unused_(bytes), nk_unused_(handle), nk_unused_(stream);
    void **allocations = (void **)pointer;
    if (pointer) free(allocations[-1]);
}

NUMKONG_INLINE nk_status_t nk_allocator_init_heap(nk_allocator_t *allocator, nk_size_t alignment) {
    if (!alignment || (alignment & (alignment - 1))) return nk_unexpected_dimensions_k;
    allocator->allocate = nk_allocate_heap_;
    allocator->free = nk_free_heap_;
    allocator->handle = (void *)(alignment < sizeof(void *) ? sizeof(void *) : alignment);
    return nk_success_k;
}

typedef struct nk_arena_t_ {
    nk_size_t capacity, consumed, alignment;
} nk_arena_t_;

NUMKONG_INLINE void *nk_allocate_arena_(nk_size_t bytes, void *handle, nk_stream_t stream) {
    nk_unused_(stream);
    nk_arena_t_ *arena = (nk_arena_t_ *)handle;
    nk_size_t const padding = (-((nk_size_t)handle + arena->consumed)) & (arena->alignment - 1);
    nk_size_t const remaining = arena->capacity - arena->consumed;
    if (!bytes || padding > remaining || bytes > remaining - padding) return NUMKONG_NULL;
    void *pointer = (unsigned char *)handle + arena->consumed + padding;
    arena->consumed += padding + bytes;
    return pointer;
}

NUMKONG_INLINE void nk_free_arena_(void *pointer, nk_size_t bytes, void *handle, nk_stream_t stream) {
    nk_unused_(pointer), nk_unused_(bytes), nk_unused_(handle), nk_unused_(stream);
}

NUMKONG_INLINE nk_status_t nk_allocator_init_arena(nk_allocator_t *allocator, void *buffer, nk_size_t bytes,
                                                   nk_size_t alignment) {
    if (!alignment || (alignment & (alignment - 1))) return nk_unexpected_dimensions_k;
    if (!buffer || bytes > NUMKONG_SIZE_MAX - (nk_size_t)buffer) return nk_bad_alloc_k;
    nk_size_t const padding = (-(nk_size_t)buffer) & (sizeof(nk_size_t) - 1);
    if (padding > bytes || bytes - padding < sizeof(nk_arena_t_)) return nk_bad_alloc_k;
    nk_arena_t_ *arena = (nk_arena_t_ *)((unsigned char *)buffer + padding);
    arena->capacity = bytes - padding;
    arena->consumed = sizeof(nk_arena_t_);
    arena->alignment = alignment;
    allocator->allocate = nk_allocate_arena_;
    allocator->free = nk_free_arena_;
    allocator->handle = arena;
    return nk_success_k;
}

/** Selects host/device shared storage; failures preserve the allocator. */
NUMKONG_API nk_status_t nk_allocator_init_unified_best(nk_allocator_t *allocator, nk_capability_t capabilities);

/** Selects device-local storage, or shared storage on Metal. CPU masks have no device allocator. */
NUMKONG_API nk_status_t nk_allocator_init_device_best(nk_allocator_t *allocator, nk_capability_t capabilities);

/** Selects pinned host storage on CUDA/ROCm or shared storage on Metal; unsupported on CPU. */
NUMKONG_API nk_status_t nk_allocator_init_pinned_best(nk_allocator_t *allocator, nk_capability_t capabilities);

/** @copydoc nk_allocator_init_unified_best */
NUMKONG_API nk_status_t nk_allocator_init_unified_serial(nk_allocator_t *allocator);

/** @copydoc nk_allocator_init_unified_best */
NUMKONG_API nk_status_t nk_allocator_init_unified_cuda(nk_allocator_t *allocator);
/** @copydoc nk_allocator_init_device_best */
NUMKONG_API nk_status_t nk_allocator_init_device_cuda(nk_allocator_t *allocator);
/** @copydoc nk_allocator_init_pinned_best */
NUMKONG_API nk_status_t nk_allocator_init_pinned_cuda(nk_allocator_t *allocator);

/** @copydoc nk_allocator_init_unified_best */
NUMKONG_API nk_status_t nk_allocator_init_unified_rocm(nk_allocator_t *allocator);
/** @copydoc nk_allocator_init_device_best */
NUMKONG_API nk_status_t nk_allocator_init_device_rocm(nk_allocator_t *allocator);
/** @copydoc nk_allocator_init_pinned_best */
NUMKONG_API nk_status_t nk_allocator_init_pinned_rocm(nk_allocator_t *allocator);

/** @copydoc nk_allocator_init_unified_best */
NUMKONG_API nk_status_t nk_allocator_init_unified_metal(nk_allocator_t *allocator);

/**
 *  @brief Allocates @p bytes that the host and the device of @p stream both address.
 *  @param[in] bytes The size of the block, where zero hands out a null one.
 *  @param[out] pointer The block, or null on failure.
 *      CPU storage uses @c nk_default_alignment_k alignment.
 *  @param[in] capabilities The group to allocate for, like @c nk_capabilities_enabled_cuda reports.
 *  @param[in] stream Null on the CPU, or a stream of that group, naming the device.
 *
 *  CUDA and ROCm hand out managed memory, CUDA from the device's managed pool in the order of
 *  @p stream, and Metal a shared buffer every kernel on that device binds, so the host reads what a
 *  kernel wrote once the stream is synchronized.
 *
 *  @par Statuses:
 *  - @c nk_missing_gpu_k A group without a device or missing from this build.
 */
NUMKONG_API nk_status_t nk_memory_allocate_unified_best(nk_size_t bytes, void **pointer, nk_capability_t capabilities,
                                                        nk_stream_t stream);

/**
 *  @brief Returns a block of @ref nk_memory_allocate_unified_best once the work queued on
 *      @p stream is done with it.
 *  @param[in] pointer The block, or null for none.
 *  @param[in] bytes The size it was allocated with.
 *  @param[in] capabilities One device's capabilities, like @c nk_capabilities_enabled_cpu reports.
 *  @param[in] stream The stream the block was allocated on, or null for the default.
 *
 *  CUDA frees blocks of its managed pool once the work queued on @p stream completes, without
 *  waiting; devices without concurrent managed access and ROCm wait for the device first; Metal
 *  releases the buffer once the work committed to @p stream so far completes.
 *
 *  @par Statuses:
 *  - @c nk_device_memory_mismatch_k A block the group never handed out.
 *  - @c nk_missing_gpu_k
 */
NUMKONG_API nk_status_t nk_memory_free_unified_best(void *pointer, nk_size_t bytes, nk_capability_t capabilities,
                                                    nk_stream_t stream);

/** @copydoc nk_memory_allocate_unified_best */
NUMKONG_API nk_status_t nk_memory_allocate_unified_serial(nk_size_t bytes, void **pointer, nk_stream_t stream);
/** @copydoc nk_memory_free_unified_best */
NUMKONG_API nk_status_t nk_memory_free_unified_serial(void *pointer, nk_size_t bytes, nk_stream_t stream);

/** @copydoc nk_memory_allocate_unified_best */
NUMKONG_API nk_status_t nk_memory_allocate_unified_cuda(nk_size_t bytes, void **pointer, nk_stream_t stream);
/** @copydoc nk_memory_free_unified_best */
NUMKONG_API nk_status_t nk_memory_free_unified_cuda(void *pointer, nk_size_t bytes, nk_stream_t stream);

/** @copydoc nk_memory_allocate_unified_best */
NUMKONG_API nk_status_t nk_memory_allocate_unified_rocm(nk_size_t bytes, void **pointer, nk_stream_t stream);
/** @copydoc nk_memory_free_unified_best */
NUMKONG_API nk_status_t nk_memory_free_unified_rocm(void *pointer, nk_size_t bytes, nk_stream_t stream);

/** @copydoc nk_memory_allocate_unified_best */
NUMKONG_API nk_status_t nk_memory_allocate_unified_metal(nk_size_t bytes, void **pointer, nk_stream_t stream);
/** @copydoc nk_memory_free_unified_best */
NUMKONG_API nk_status_t nk_memory_free_unified_metal(void *pointer, nk_size_t bytes, nk_stream_t stream);

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
                                                        nk_stream_t stream) {
    nk_unused_(bytes), nk_unused_(capabilities), nk_unused_(stream);
    *pointer = NUMKONG_NULL;
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_memory_free_unified_best(void *pointer, nk_size_t bytes, nk_capability_t capabilities,
                                                    nk_stream_t stream) {
    nk_unused_(pointer), nk_unused_(bytes), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_allocator_init_unified_cuda(nk_allocator_t *allocator) {
    nk_unused_(allocator);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_allocator_init_device_cuda(nk_allocator_t *allocator) {
    nk_unused_(allocator);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_allocator_init_pinned_cuda(nk_allocator_t *allocator) {
    nk_unused_(allocator);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_allocator_init_unified_rocm(nk_allocator_t *allocator) {
    nk_unused_(allocator);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_allocator_init_device_rocm(nk_allocator_t *allocator) {
    nk_unused_(allocator);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_allocator_init_pinned_rocm(nk_allocator_t *allocator) {
    nk_unused_(allocator);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_allocator_init_unified_metal(nk_allocator_t *allocator) {
    nk_unused_(allocator);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_memory_allocate_unified_cuda(nk_size_t bytes, void **pointer, nk_stream_t stream) {
    nk_unused_(bytes), nk_unused_(stream);
    *pointer = NUMKONG_NULL;
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_memory_free_unified_cuda(void *pointer, nk_size_t bytes, nk_stream_t stream) {
    nk_unused_(pointer), nk_unused_(bytes), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_memory_allocate_unified_rocm(nk_size_t bytes, void **pointer, nk_stream_t stream) {
    nk_unused_(bytes), nk_unused_(stream);
    *pointer = NUMKONG_NULL;
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_memory_free_unified_rocm(void *pointer, nk_size_t bytes, nk_stream_t stream) {
    nk_unused_(pointer), nk_unused_(bytes), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_memory_allocate_unified_metal(nk_size_t bytes, void **pointer, nk_stream_t stream) {
    nk_unused_(bytes), nk_unused_(stream);
    *pointer = NUMKONG_NULL;
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_memory_free_unified_metal(void *pointer, nk_size_t bytes, nk_stream_t stream) {
    nk_unused_(pointer), nk_unused_(bytes), nk_unused_(stream);
    return nk_missing_library_k;
}

#endif

#ifdef __cplusplus
}
#endif

#if NUMKONG_HEADER_ONLY
#include "numkong/memory/serial.h"
#endif

#endif // NUMKONG_MEMORY_H
