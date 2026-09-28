/**
 *  @file include/numkong/metal.h
 *  @author Ash Vardanian
 *  @date September 24, 2026
 *  @brief The Metal queue every Apple GPU kernel runs on: device, command stream, and the memory
 *      both sides address.
 *
 *  CUDA kernels take a @c cudaStream_t, and Metal has no such thing in C, so NumKong owns one here.
 *  An @ref nk_metal_queue_t is the @c stream every dispatch point takes on Apple GPUs: each call
 *  encodes into its command buffer and returns, and @ref nk_metal_synchronize waits for them.
 *
 *  A shared Metal buffer's host address is never its GPU address, so a kernel cannot be handed a
 *  host pointer the way a CUDA kernel is handed a managed one. Memory from @ref nk_metal_allocate
 *  is the device-reachable kind: the queue records every block it hands out, and a launch resolves
 *  each pointer to its block and an offset inside it.
 *
 *  Written in C, as every GPU header in this library is. Metal's API is Objective-C, reached here
 *  through @c objc_msgSend cast to each call's exact prototype, so no Objective-C translation unit
 *  is needed and a C or C++ caller links against Metal and Foundation alone.
 */
#ifndef NUMKONG_METAL_H
#define NUMKONG_METAL_H

#include "numkong/types.h" // `nk_size_t`, `NUMKONG_WITH_METAL`

#if NUMKONG_WITH_METAL
#include <TargetConditionals.h> // `TARGET_OS_OSX`
#include <objc/message.h>       // `objc_msgSend`
#include <objc/runtime.h>       // `sel_registerName`, `objc_getClass`
#include <stdlib.h>             // `malloc`, `realloc`, `free`
#include <string.h>             // `strcmp`, `memmove`

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Runtime

/** A launch extent, laid out as @c MTLSize so it passes to Metal by value. */
typedef struct {
    nk_size_t width, height, depth;
} nk_metal_size_t;

/** @c MTLCreateSystemDefaultDevice, renamed so Objective-C units may include `<Metal/Metal.h>`. */
typedef void *nk_metal_device_factory_t(void);
nk_metal_device_factory_t nk_metal_default_device_ __asm__("_MTLCreateSystemDefaultDevice");
#if TARGET_OS_OSX

/** @c MTLCopyAllDevices, renamed the same way; only macOS lists more than the default device. */
nk_metal_device_factory_t nk_metal_all_devices_ __asm__("_MTLCopyAllDevices");
#endif
void *objc_autoreleasePoolPush(void);
void objc_autoreleasePoolPop(void *pool);

NUMKONG_INLINE void *nk_metal_class_(char const *name) { return (void *)objc_getClass(name); }
NUMKONG_INLINE void *nk_metal_get_(void *object, char const *selector) {
    return ((void *(*)(void *, SEL))objc_msgSend)(object, sel_registerName(selector));
}
NUMKONG_INLINE void nk_metal_do_(void *object, char const *selector) {
    ((void (*)(void *, SEL))objc_msgSend)(object, sel_registerName(selector));
}
NUMKONG_INLINE nk_size_t nk_metal_count_(void *object, char const *selector) {
    return ((nk_size_t (*)(void *, SEL))objc_msgSend)(object, sel_registerName(selector));
}
NUMKONG_INLINE void *nk_metal_string_(char const *text) {
    return ((void *(*)(void *, SEL, char const *))objc_msgSend)(nk_metal_class_("NSString"),
                                                                sel_registerName("stringWithUTF8String:"), text);
}

/** @c MTLLanguageVersion3_1, the first with @c bfloat. */
#define NUMKONG_METAL_LANGUAGE_3_1_ ((3u << 16) | 1u)

/** @c MTLLanguageVersion4_0, the first with tensor operations. */
#define NUMKONG_METAL_LANGUAGE_4_0_ ((4u << 16) | 0u)

/** How many Metal devices the system lists: every GPU on macOS, the default one elsewhere. */
NUMKONG_INLINE nk_size_t nk_metal_list_devices_(void) {
#if TARGET_OS_OSX
    void *const devices = nk_metal_all_devices_();
    if (!devices) return 0;
    nk_size_t const count = nk_metal_count_(devices, "count");
    nk_metal_do_(devices, "release");
    return count;
#else
    void *const device = nk_metal_default_device_();
    if (!device) return 0;
    nk_metal_do_(device, "release");
    return 1;
#endif
}

/** The Metal device at @p index of the system's list, retained, or null past its end. */
NUMKONG_INLINE void *nk_metal_device_(nk_size_t index) {
#if TARGET_OS_OSX
    void *const devices = nk_metal_all_devices_();
    if (!devices) return NULL;
    void *device = NULL;
    if (index < nk_metal_count_(devices, "count")) {
        device = ((void *(*)(void *, SEL, nk_size_t))objc_msgSend)(devices, sel_registerName("objectAtIndex:"), index);
        nk_metal_do_(device, "retain");
    }
    nk_metal_do_(devices, "release");
    return device;
#else
    return index == 0 ? nk_metal_default_device_() : NULL;
#endif
}

#pragma endregion Runtime

#pragma region Queue

/** One block @ref nk_metal_allocate handed out: its buffer and the host bytes it spans. */
typedef struct {
    void *buffer;
    char *host;
    nk_size_t bytes;
} nk_metal_allocation_t;

/** One compiled kernel, keyed by its function name. */
typedef struct {
    char const *name;
    void *pipeline;
} nk_metal_pipeline_t;

/** Pipelines one queue keeps; a family registers a handful, so the bound is never approached. */
enum { nk_metal_pipelines_max_k = 64 };

/**
 *  @brief The stream every Metal call encodes into, and the memory those calls may address.
 *
 *  Owned by the caller like a @c cudaStream_t, initialized by @ref nk_metal_queue_init and released
 *  by @ref nk_metal_queue_free. Not thread-safe: one queue serves one host thread at a time.
 */
typedef struct {

    /** The @c id<MTLDevice> the queue drives. */
    void *device;

    /** The @c id<MTLCommandQueue> command buffers are taken from. */
    void *command_queue;

    /** The command buffer and compute encoder of the call being encoded. */
    void *command_buffer, *encoder;

    /** Command buffers committed since the last @ref nk_metal_synchronize, in commit order. */
    void **pending;
    nk_size_t pending_count, pending_capacity;

    /** The first failure since the last @ref nk_metal_synchronize. */
    nk_status_t status;

    /** Every block handed out and not yet freed, ascending by host address. */
    nk_metal_allocation_t *allocations;
    nk_size_t allocations_count, allocations_capacity;

    /** The one @c id<MTLLibrary> per family source, keyed by the whole source text. */
    void *libraries[8];
    char const *library_sources[8];

    /** Every pipeline built so far. */
    nk_metal_pipeline_t pipelines[nk_metal_pipelines_max_k];
    nk_size_t pipelines_count;
} nk_metal_queue_t;

/**
 *  @brief Opens a queue on one GPU.
 *  @param[out] queue The queue to open.
 *  @param[in] device The device's index, as @c nk_metal_capabilities_detected takes it.
 *  @return @c nk_success_k, or @c nk_missing_gpu_k when there is no such device.
 */
NUMKONG_API nk_status_t nk_metal_queue_init(nk_metal_queue_t *queue, nk_size_t device);

/**
 *  @brief Waits for every call committed since the last synchronization.
 *  @return The first failure since the last call: an encoding refusal, or
 *      @c nk_device_code_mismatch_k when a command buffer finished in error.
 */
NUMKONG_API nk_status_t nk_metal_synchronize(nk_metal_queue_t *queue);

/** Drains the queue, then releases every pipeline, library, block, and the queue itself. */
NUMKONG_API void nk_metal_queue_free(nk_metal_queue_t *queue);

/**
 *  @brief Hands out @p bytes both the host and the queue's kernels address, or null.
 *
 *  The block is a shared-storage buffer, so the host reads what a kernel wrote once the queue is
 *  synchronized, as with CUDA managed memory. Every pointer inside it is device-reachable.
 */
NUMKONG_API void *nk_metal_allocate(nk_metal_queue_t *queue, nk_size_t bytes);

/** Returns a block from @ref nk_metal_allocate; the queue must not still be using it. */
NUMKONG_API void nk_metal_free(nk_metal_queue_t *queue, void *pointer);

#if NUMKONG_TARGET_METAL

NUMKONG_API nk_status_t nk_metal_queue_init(nk_metal_queue_t *queue, nk_size_t device) {
    memset(queue, 0, sizeof(*queue));
    queue->device = nk_metal_device_(device);
    if (!queue->device) return nk_missing_gpu_k;
    queue->command_queue = nk_metal_get_(queue->device, "newCommandQueue");
    return queue->command_queue ? nk_success_k : nk_missing_gpu_k;
}

NUMKONG_API nk_status_t nk_metal_synchronize(nk_metal_queue_t *queue) {
    nk_size_t const completed = 4; // `MTLCommandBufferStatusCompleted`
    for (nk_size_t index = 0; index != queue->pending_count; ++index) {
        nk_metal_do_(queue->pending[index], "waitUntilCompleted");
        if (nk_metal_count_(queue->pending[index], "status") != completed && queue->status == nk_success_k)
            queue->status = nk_device_code_mismatch_k;
        nk_metal_do_(queue->pending[index], "release");
    }
    queue->pending_count = 0;
    nk_status_t const status = queue->status;
    queue->status = nk_success_k;
    return status;
}

NUMKONG_API void nk_metal_queue_free(nk_metal_queue_t *queue) {
    if (!queue->device) return;
    nk_status_t const drained = nk_metal_synchronize(queue);
    nk_unused_(drained);
    for (nk_size_t index = 0; index != queue->pipelines_count; ++index)
        nk_metal_do_(queue->pipelines[index].pipeline, "release");
    for (nk_size_t index = 0; index != sizeof(queue->libraries) / sizeof(queue->libraries[0]); ++index)
        if (queue->libraries[index]) nk_metal_do_(queue->libraries[index], "release");
    for (nk_size_t index = 0; index != queue->allocations_count; ++index)
        nk_metal_do_(queue->allocations[index].buffer, "release");
    free(queue->allocations);
    free(queue->pending);
    nk_metal_do_(queue->command_queue, "release");
    nk_metal_do_(queue->device, "release");
    memset(queue, 0, sizeof(*queue));
}

NUMKONG_API void *nk_metal_allocate(nk_metal_queue_t *queue, nk_size_t bytes) {
    if (!bytes) return NULL;
    if (queue->allocations_count == queue->allocations_capacity) {
        nk_size_t const capacity = queue->allocations_capacity ? queue->allocations_capacity * 2 : 16;
        nk_metal_allocation_t *grown = (nk_metal_allocation_t *)realloc(queue->allocations,
                                                                        capacity * sizeof(nk_metal_allocation_t));
        if (!grown) return NULL;
        queue->allocations = grown, queue->allocations_capacity = capacity;
    }
    nk_size_t const shared_storage = 0; // `MTLResourceStorageModeShared`
    void *buffer = ((void *(*)(void *, SEL, nk_size_t, nk_size_t))objc_msgSend)(
        queue->device, sel_registerName("newBufferWithLength:options:"), bytes, shared_storage);
    if (!buffer) return NULL;
    char *host = (char *)nk_metal_get_(buffer, "contents");
    nk_size_t position = queue->allocations_count;
    while (position != 0 && queue->allocations[position - 1].host > host) --position;
    memmove(queue->allocations + position + 1, queue->allocations + position,
            (queue->allocations_count - position) * sizeof(nk_metal_allocation_t));
    queue->allocations[position].buffer = buffer;
    queue->allocations[position].host = host;
    queue->allocations[position].bytes = bytes;
    ++queue->allocations_count;
    return host;
}

NUMKONG_API void nk_metal_free(nk_metal_queue_t *queue, void *pointer) {
    for (nk_size_t index = 0; index != queue->allocations_count; ++index) {
        if (queue->allocations[index].host != (char *)pointer) continue;
        nk_metal_do_(queue->allocations[index].buffer, "release");
        memmove(queue->allocations + index, queue->allocations + index + 1,
                (queue->allocations_count - index - 1) * sizeof(nk_metal_allocation_t));
        --queue->allocations_count;
        return;
    }
}

#endif // NUMKONG_TARGET_METAL

/**
 *  @brief Finds the block holding @p pointer, and its offset there.
 *  @return The block, or null when @p pointer lies outside every block.
 */
NUMKONG_INLINE nk_metal_allocation_t const *nk_metal_resolve_(nk_metal_queue_t const *queue, void const *pointer,
                                                              nk_size_t *offset) {
    char const *const address = (char const *)pointer;
    nk_size_t low = 0, high = queue->allocations_count;
    while (low < high) {
        nk_size_t const middle = low + (high - low) / 2;
        if (queue->allocations[middle].host <= address) low = middle + 1;
        else high = middle;
    }
    if (low == 0) return NULL;
    nk_metal_allocation_t const *const allocation = queue->allocations + low - 1;
    if (address >= allocation->host + allocation->bytes) return NULL;
    *offset = (nk_size_t)(address - allocation->host);
    return allocation;
}

/**
 *  @brief Kernel @p name's pipeline from the library built out of @p source, built on first use.
 *  @param[in] language_version An @c MTLLanguageVersion, like `(3 << 16) | 1` for Metal 3.1.
 *  @return The pipeline, or null after recording @c nk_device_code_mismatch_k.
 */
NUMKONG_INLINE void *nk_metal_pipeline_(nk_metal_queue_t *queue, char const *source, char const *name,
                                        nk_size_t language_version) {
    for (nk_size_t index = 0; index != queue->pipelines_count; ++index)
        if (strcmp(queue->pipelines[index].name, name) == 0) return queue->pipelines[index].pipeline;
    if (queue->pipelines_count == nk_metal_pipelines_max_k) {
        queue->status = nk_device_code_mismatch_k;
        return NULL;
    }

    // Each family's source is a distinct string per translation unit, so libraries key on the text.
    nk_size_t const slots = sizeof(queue->libraries) / sizeof(queue->libraries[0]);
    nk_size_t slot = 0;
    while (slot != slots && queue->library_sources[slot] && strcmp(queue->library_sources[slot], source) != 0) ++slot;
    if (slot == slots) {
        queue->status = nk_device_code_mismatch_k;
        return NULL;
    }

    void *const pool = objc_autoreleasePoolPush();
    void *pipeline = NULL;
    if (!queue->libraries[slot]) {
        void *const options = nk_metal_get_(nk_metal_get_(nk_metal_class_("MTLCompileOptions"), "alloc"), "init");
        ((void (*)(void *, SEL, nk_size_t))objc_msgSend)(options, sel_registerName("setLanguageVersion:"),
                                                         language_version);
        void *error = NULL;
        queue->libraries[slot] = ((void *(*)(void *, SEL, void *, void *, void **))objc_msgSend)(
            queue->device, sel_registerName("newLibraryWithSource:options:error:"), nk_metal_string_(source), options,
            &error);
        nk_metal_do_(options, "release");
        if (queue->libraries[slot]) queue->library_sources[slot] = source;
    }
    if (queue->libraries[slot]) {
        void *const function = ((void *(*)(void *, SEL, void *))objc_msgSend)(
            queue->libraries[slot], sel_registerName("newFunctionWithName:"), nk_metal_string_(name));
        void *error = NULL;
        if (function)
            pipeline = ((void *(*)(void *, SEL, void *, void **))objc_msgSend)(
                queue->device, sel_registerName("newComputePipelineStateWithFunction:error:"), function, &error);
        if (function) nk_metal_do_(function, "release");
    }
    objc_autoreleasePoolPop(pool);
    if (!pipeline) {
        queue->status = nk_device_code_mismatch_k;
        return NULL;
    }
    queue->pipelines[queue->pipelines_count].name = name;
    queue->pipelines[queue->pipelines_count].pipeline = pipeline;
    ++queue->pipelines_count;
    return pipeline;
}

/** Opens one call's command buffer and compute encoder; @ref nk_metal_dispatch_ commits them. */
NUMKONG_INLINE void *nk_metal_encoder_(nk_metal_queue_t *queue) {
    if (queue->encoder) return queue->encoder;
    if (queue->pending_count == queue->pending_capacity) {
        nk_size_t const capacity = queue->pending_capacity ? queue->pending_capacity * 2 : 16;
        void **grown = (void **)realloc(queue->pending, capacity * sizeof(void *));
        if (!grown) {
            queue->status = nk_bad_alloc_k;
            return NULL;
        }
        queue->pending = grown, queue->pending_capacity = capacity;
    }
    void *const pool = objc_autoreleasePoolPush();
    queue->command_buffer = nk_metal_get_(nk_metal_get_(queue->command_queue, "commandBuffer"), "retain");
    if (queue->command_buffer)
        queue->encoder = nk_metal_get_(nk_metal_get_(queue->command_buffer, "computeCommandEncoder"), "retain");
    objc_autoreleasePoolPop(pool);
    if (!queue->encoder) queue->status = nk_device_code_mismatch_k;
    return queue->encoder;
}

/** Binds @p allocation at @p offset to buffer slot @p index of @p encoder. */
NUMKONG_INLINE void nk_metal_bind_(void *encoder, nk_metal_allocation_t const *allocation, nk_size_t offset,
                                   nk_size_t index) {
    ((void (*)(void *, SEL, void *, nk_size_t, nk_size_t))objc_msgSend)(
        encoder, sel_registerName("setBuffer:offset:atIndex:"), allocation->buffer, offset, index);
}

/** Copies @p bytes of @p arguments into buffer slot @p index of @p encoder. */
NUMKONG_INLINE void nk_metal_bind_bytes_(void *encoder, void const *arguments, nk_size_t bytes, nk_size_t index) {
    ((void (*)(void *, SEL, void const *, nk_size_t, nk_size_t))objc_msgSend)(
        encoder, sel_registerName("setBytes:length:atIndex:"), arguments, bytes, index);
}

/** Dispatches @p pipeline over @p groups threadgroups of @p threads each, then commits the call,
 *  so it starts running while the host moves on, as a CUDA launch does. */
NUMKONG_INLINE void nk_metal_dispatch_(nk_metal_queue_t *queue, void *pipeline, nk_metal_size_t groups,
                                       nk_metal_size_t threads) {
    void *const encoder = queue->encoder;
    ((void (*)(void *, SEL, void *))objc_msgSend)(encoder, sel_registerName("setComputePipelineState:"), pipeline);
    ((void (*)(void *, SEL, nk_metal_size_t, nk_metal_size_t))objc_msgSend)(
        encoder, sel_registerName("dispatchThreadgroups:threadsPerThreadgroup:"), groups, threads);
    nk_metal_do_(encoder, "endEncoding");
    nk_metal_do_(encoder, "release");
    nk_metal_do_(queue->command_buffer, "commit");
    queue->pending[queue->pending_count++] = queue->command_buffer;
    queue->command_buffer = NULL, queue->encoder = NULL;
}

#pragma endregion Queue

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // NUMKONG_WITH_METAL
#endif // NUMKONG_METAL_H
