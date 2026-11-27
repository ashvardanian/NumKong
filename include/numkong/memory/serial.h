/**
 *  @file include/numkong/memory/serial.h
 *  @author Ash Vardanian
 *  @date October 5, 2026
 *  @brief Aligned heap and arena allocation for host memory.
 */
#ifndef NUMKONG_MEMORY_SERIAL_H
#define NUMKONG_MEMORY_SERIAL_H

#include "numkong/memory.h"

#ifdef __cplusplus
extern "C" {
#endif

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
