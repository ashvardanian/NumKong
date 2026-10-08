/**
 *  @file include/numkong/memory/serial.h
 *  @author Ash Vardanian
 *  @date October 5, 2026
 *  @brief Aligned heap and arena allocation for host memory.
 */
#ifndef NUMKONG_MEMORY_SERIAL_H
#define NUMKONG_MEMORY_SERIAL_H

#include <stdlib.h> // `malloc`, `free`

#include "numkong/types.h"

#ifdef __cplusplus
extern "C" {
#endif

NUMKONG_INLINE void *nk_allocate_heap_serial_(nk_size_t bytes, void *handle, nk_stream_t stream) {
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

NUMKONG_INLINE void nk_free_heap_serial_(void *pointer, nk_size_t bytes, void *handle, nk_stream_t stream) {
    nk_unused_(bytes), nk_unused_(handle), nk_unused_(stream);
    void **allocations = (void **)pointer;
    if (pointer) free(allocations[-1]);
}

NUMKONG_INLINE nk_status_t nk_allocator_init_heap(nk_allocator_t *allocator, nk_size_t alignment) {
    if (!alignment || (alignment & (alignment - 1))) return nk_unexpected_dimensions_k;
    allocator->allocate = nk_allocate_heap_serial_;
    allocator->free = nk_free_heap_serial_;
    allocator->handle = (void *)(alignment < sizeof(void *) ? sizeof(void *) : alignment);
    return nk_success_k;
}

typedef struct nk_arena_t_ {
    nk_size_t capacity, consumed, alignment;
} nk_arena_t_;

NUMKONG_INLINE void *nk_allocate_arena_serial_(nk_size_t bytes, void *handle, nk_stream_t stream) {
    nk_unused_(stream);
    nk_arena_t_ *arena = (nk_arena_t_ *)handle;
    nk_size_t const padding = (-((nk_size_t)handle + arena->consumed)) & (arena->alignment - 1);
    nk_size_t const remaining = arena->capacity - arena->consumed;
    if (!bytes || padding > remaining || bytes > remaining - padding) return NUMKONG_NULL;
    void *pointer = (unsigned char *)handle + arena->consumed + padding;
    arena->consumed += padding + bytes;
    return pointer;
}

NUMKONG_INLINE void nk_free_arena_serial_(void *pointer, nk_size_t bytes, void *handle, nk_stream_t stream) {
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
    allocator->allocate = nk_allocate_arena_serial_;
    allocator->free = nk_free_arena_serial_;
    allocator->handle = arena;
    return nk_success_k;
}

#if NUMKONG_TARGET_SERIAL

NUMKONG_API nk_status_t nk_allocator_init_unified_serial(nk_allocator_t *allocator) {
    return nk_allocator_init_heap(allocator, nk_default_alignment_k);
}

NUMKONG_API nk_status_t nk_memory_allocate_unified_serial(nk_size_t bytes, void **pointer, nk_stream_t stream) {
    nk_allocator_t allocator;
    nk_unused_(nk_allocator_init_heap(&allocator, nk_default_alignment_k));
    *pointer = allocator.allocate(bytes, allocator.handle, stream);
    return *pointer || !bytes ? nk_success_k : nk_bad_alloc_k;
}

NUMKONG_API nk_status_t nk_memory_free_unified_serial(void *pointer, nk_size_t bytes, nk_stream_t stream) {
    nk_allocator_t allocator;
    nk_unused_(nk_allocator_init_heap(&allocator, nk_default_alignment_k));
    allocator.free(pointer, bytes, allocator.handle, stream);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_SERIAL

#ifdef __cplusplus
}
#endif

#endif // NUMKONG_MEMORY_SERIAL_H
