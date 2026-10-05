/**
 *  @file include/numkong/metal.h
 *  @author Ash Vardanian
 *  @date September 24, 2026
 *  @brief The Metal runtime every Apple GPU kernel runs on: one context per device, holding its
 *      default queue, the memory both sides address, and the work committed to each stream.
 *
 *  The @c stream every dispatch point takes on Apple GPUs is an @c id<MTLCommandQueue>, or null for
 *  the library's own queue on the system default device. Each call encodes one command buffer,
 *  commits it and returns, and @ref nk_stream_synchronize_metal waits for those of one stream.
 *
 *  A shared Metal buffer's host address is never its GPU address, so a kernel cannot be handed a
 *  host pointer the way a CUDA kernel is handed a managed one. Memory from
 *  @ref nk_memory_allocate_unified_metal is the device-reachable kind: the device's context records
 *  every block it hands out, and a launch resolves each pointer to its block and an offset in it.
 *  Encoders live on each caller's stack and the context sits behind a lock, so any number of
 *  threads may encode into any queues.
 *
 *  Written in C, as every GPU header in this library is. Metal's API is Objective-C, reached here
 *  through @c objc_msgSend cast to each call's exact prototype, so no Objective-C translation unit
 *  is needed and a C or C++ caller links against Metal and Foundation alone.
 */
#ifndef NUMKONG_METAL_H
#define NUMKONG_METAL_H

#include "numkong/types.h" // `nk_size_t`, `NUMKONG_ARCH_METAL_`

#if NUMKONG_ARCH_METAL_
#include <TargetConditionals.h> // `TARGET_OS_OSX`
#include <os/lock.h>            // `os_unfair_lock`
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

#pragma region Context

/** One block the unified allocator handed out: its shared buffer and the host bytes it spans. */
typedef struct {
    void *buffer;
    char *host;
    nk_size_t bytes;
} nk_metal_block_t;

/** One compiled kernel, keyed by its function name. */
typedef struct {
    char const *name;
    void *pipeline;
} nk_metal_pipeline_t;

/** Pipelines one device keeps; a family registers a handful, so the bound is never approached. */
enum { nk_metal_pipelines_max_k = 64 };

/** One stream with work committed since its last synchronization. */
typedef struct {

    /** The @c id<MTLCommandQueue>, retained while pending. */
    void *queue;

    /** The command buffers committed to it, in commit order. */
    void **commands;
    nk_size_t commands_count, commands_capacity;
    nk_size_t waiters;

    /** The @c id<MTLBuffer> of every block freed while those commands may still read it. */
    void **frees;
    nk_size_t frees_count, frees_capacity;
} nk_metal_pending_t;

/** The library's state for one @c id<MTLDevice>, built on first use and kept for the process, like
 *  a CUDA primary context. */
typedef struct {

    /** The @c id<MTLDevice>. */
    void *device;

    /** The @c id<MTLCommandQueue> a null stream names. */
    void *queue;

    /** Guards every field below. */
    os_unfair_lock lock;

    /** Every block handed out and not yet freed, ascending by host address. */
    nk_metal_block_t *blocks;
    nk_size_t blocks_count, blocks_capacity;

    /** Every stream with committed work. */
    nk_metal_pending_t *pending;
    nk_size_t pending_count, pending_capacity;

    /** The one @c id<MTLLibrary> per family source, keyed by the whole source text. */
    void *libraries[8];
    char const *library_sources[8];

    /** Every pipeline built so far. */
    nk_metal_pipeline_t pipelines[nk_metal_pipelines_max_k];
    nk_size_t pipelines_count;
} nk_metal_context_t;

/** One call being encoded, on its caller's stack: the context and queue of its stream, and its
 *  command buffer and compute encoder. */
typedef struct {
    nk_metal_context_t *context;
    void *queue, *commands, *encoder;
} nk_metal_call_t;

enum { nk_metal_contexts_max_k = 8 };

/** The device contexts and the lock guarding their initialization. */
NUMKONG_API nk_metal_context_t *nk_metal_contexts_(os_unfair_lock_t *contexts_lock);

#if NUMKONG_HEADER_ONLY
NUMKONG_API nk_metal_context_t *nk_metal_contexts_(os_unfair_lock_t *contexts_lock) {
    static nk_metal_context_t contexts[nk_metal_contexts_max_k];
    static os_unfair_lock lock;
    *contexts_lock = &lock;
    return contexts;
}
#endif

/** The context of the device @p stream belongs to, of the system default device for a null one, or
 *  null without a GPU or past the 8 devices it keeps. */
NUMKONG_INLINE nk_metal_context_t *nk_metal_context_(void *stream);

/** @p items with room for one past @p count of @p size bytes each, grown twofold from 16, or null
 *  leaving @p items as they were. */
NUMKONG_INLINE void *nk_metal_reserve_(void *items, nk_size_t count, nk_size_t *capacity, nk_size_t size) {
    if (count < *capacity) return items;
    nk_size_t const grown_capacity = *capacity ? *capacity * 2 : 16;
    void *const grown = realloc(items, grown_capacity * size);
    if (grown) *capacity = grown_capacity;
    return grown;
}

/** The committed work of @p queue in @p context, or null if it has none; called under the lock. */
NUMKONG_INLINE nk_metal_pending_t *nk_metal_pending_(nk_metal_context_t *context, void *queue) {
    for (nk_size_t index = 0; index != context->pending_count; ++index)
        if (context->pending[index].queue == queue) return &context->pending[index];
    return NULL;
}

/** Opens @p call on @p stream: its device's context and the queue it names. */
NUMKONG_INLINE nk_status_t nk_metal_enter_(void *stream, nk_metal_call_t *call) {
    call->context = nk_metal_context_(stream);
    call->queue = stream ? stream : call->context ? call->context->queue : NULL;
    call->commands = NULL, call->encoder = NULL;
    return call->context ? nk_success_k : nk_missing_gpu_k;
}

/**
 *  @brief Finds the block of @p context holding @p pointer, and its offset there.
 *  @return The block's @c id<MTLBuffer>, or null when @p pointer lies outside every block.
 */
NUMKONG_INLINE void *nk_metal_resolve_(nk_metal_context_t *context, void const *pointer, nk_size_t *offset) {
    char const *const address = (char const *)pointer;
    void *buffer = NULL;
    os_unfair_lock_lock(&context->lock);
    nk_size_t low = 0, high = context->blocks_count;
    while (low < high) {
        nk_size_t const middle = low + (high - low) / 2;
        if (context->blocks[middle].host <= address) low = middle + 1;
        else high = middle;
    }
    nk_metal_block_t const *const block = low ? context->blocks + low - 1 : NULL;
    if (block && address < block->host + block->bytes)
        buffer = block->buffer, *offset = (nk_size_t)(address - block->host);
    os_unfair_lock_unlock(&context->lock);
    return buffer;
}

/**
 *  @brief Kernel @p name's pipeline from the library built out of @p source, built on first use.
 *  @param[in] language_version An @c MTLLanguageVersion, like `(3 << 16) | 1` for Metal 3.1.
 *  @return The pipeline, or null when it fails to build.
 */
NUMKONG_INLINE void *nk_metal_pipeline_(nk_metal_context_t *context, char const *source, char const *name,
                                        nk_size_t language_version) {
    void *pipeline = NULL;
    os_unfair_lock_lock(&context->lock);
    for (nk_size_t index = 0; index != context->pipelines_count && !pipeline; ++index)
        if (strcmp(context->pipelines[index].name, name) == 0) pipeline = context->pipelines[index].pipeline;

    // Each family's source is a distinct string, so libraries key on the text.
    nk_size_t const slots = sizeof(context->libraries) / sizeof(context->libraries[0]);
    nk_size_t slot = 0;
    while (slot != slots && context->library_sources[slot] && strcmp(context->library_sources[slot], source) != 0)
        ++slot;
    if (!pipeline && slot != slots && context->pipelines_count != nk_metal_pipelines_max_k) {
        void *const pool = objc_autoreleasePoolPush();
        if (!context->libraries[slot]) {
            void *const options = nk_metal_get_(nk_metal_get_(nk_metal_class_("MTLCompileOptions"), "alloc"), "init");
            ((void (*)(void *, SEL, nk_size_t))objc_msgSend)(options, sel_registerName("setLanguageVersion:"),
                                                             language_version);
            void *error = NULL;
            context->libraries[slot] = ((void *(*)(void *, SEL, void *, void *, void **))objc_msgSend)(
                context->device, sel_registerName("newLibraryWithSource:options:error:"), nk_metal_string_(source),
                options, &error);
            nk_metal_do_(options, "release");
            if (context->libraries[slot]) context->library_sources[slot] = source;
        }
        if (context->libraries[slot]) {
            void *const function = ((void *(*)(void *, SEL, void *))objc_msgSend)(
                context->libraries[slot], sel_registerName("newFunctionWithName:"), nk_metal_string_(name));
            void *error = NULL;
            if (function)
                pipeline = ((void *(*)(void *, SEL, void *, void **))objc_msgSend)(
                    context->device, sel_registerName("newComputePipelineStateWithFunction:error:"), function, &error);
            if (function) nk_metal_do_(function, "release");
        }
        objc_autoreleasePoolPop(pool);
        if (pipeline) {
            context->pipelines[context->pipelines_count].name = name;
            context->pipelines[context->pipelines_count].pipeline = pipeline;
            ++context->pipelines_count;
        }
    }
    os_unfair_lock_unlock(&context->lock);
    return pipeline;
}

/** Opens @p call's command buffer and compute encoder on @p pipeline; @ref nk_metal_dispatch_
 *  commits them. */
NUMKONG_INLINE nk_status_t nk_metal_encoder_(nk_metal_call_t *call, void *pipeline) {
    void *const pool = objc_autoreleasePoolPush();
    call->commands = nk_metal_get_(nk_metal_get_(call->queue, "commandBuffer"), "retain");
    if (call->commands) call->encoder = nk_metal_get_(nk_metal_get_(call->commands, "computeCommandEncoder"), "retain");
    objc_autoreleasePoolPop(pool);
    if (!call->encoder) {
        if (call->commands) nk_metal_do_(call->commands, "release");
        return nk_device_code_mismatch_k;
    }
    ((void (*)(void *, SEL, void *))objc_msgSend)(call->encoder, sel_registerName("setComputePipelineState:"),
                                                  pipeline);
    return nk_success_k;
}

/** Binds @p buffer at @p offset to buffer slot @p index of @p encoder. */
NUMKONG_INLINE void nk_metal_bind_(void *encoder, void *buffer, nk_size_t offset, nk_size_t index) {
    ((void (*)(void *, SEL, void *, nk_size_t, nk_size_t))objc_msgSend)(
        encoder, sel_registerName("setBuffer:offset:atIndex:"), buffer, offset, index);
}

/** Copies @p bytes of @p arguments into buffer slot @p index of @p encoder. */
NUMKONG_INLINE void nk_metal_bind_bytes_(void *encoder, void const *arguments, nk_size_t bytes, nk_size_t index) {
    ((void (*)(void *, SEL, void const *, nk_size_t, nk_size_t))objc_msgSend)(
        encoder, sel_registerName("setBytes:length:atIndex:"), arguments, bytes, index);
}

/** Dispatches @p groups threadgroups of @p threads each, then commits the call, so it starts
 *  running while the host moves on, as a CUDA launch does. */
NUMKONG_INLINE nk_status_t nk_metal_dispatch_(nk_metal_call_t *call, nk_metal_size_t groups, nk_metal_size_t threads) {
    ((void (*)(void *, SEL, nk_metal_size_t, nk_metal_size_t))objc_msgSend)(
        call->encoder, sel_registerName("dispatchThreadgroups:threadsPerThreadgroup:"), groups, threads);
    nk_metal_do_(call->encoder, "endEncoding");
    nk_metal_do_(call->encoder, "release");

    // Committing under the lock keeps every stream's commands in their commit order
    nk_metal_context_t *const context = call->context;
    os_unfair_lock_lock(&context->lock);
    nk_metal_pending_t *pending = nk_metal_pending_(context, call->queue);
    if (!pending) {
        nk_metal_pending_t *const grown = (nk_metal_pending_t *)nk_metal_reserve_(
            context->pending, context->pending_count, &context->pending_capacity, sizeof(nk_metal_pending_t));
        if (grown) {
            context->pending = grown, pending = &grown[context->pending_count++];
            memset(pending, 0, sizeof(*pending));
            pending->queue = nk_metal_get_(call->queue, "retain");
        }
    }
    void **const commands = pending ? (void **)nk_metal_reserve_(pending->commands, pending->commands_count,
                                                                 &pending->commands_capacity, sizeof(void *))
                                    : NULL;
    if (commands) {
        pending->commands = commands, commands[pending->commands_count++] = call->commands;
        nk_metal_do_(call->commands, "commit");
    }
    os_unfair_lock_unlock(&context->lock);
    if (commands) return nk_success_k;
    nk_metal_do_(call->commands, "release");
    return nk_bad_alloc_k;
}

NUMKONG_INLINE nk_metal_context_t *nk_metal_context_(void *stream) {
    os_unfair_lock_t contexts_lock;
    nk_metal_context_t *const contexts = nk_metal_contexts_(&contexts_lock);
    void *const device = stream ? nk_metal_get_(stream, "device") : nk_metal_default_device_();
    if (!device) return NULL;
    nk_metal_context_t *context = NULL;
    os_unfair_lock_lock(contexts_lock);
    nk_size_t index = 0;
    for (; index != nk_metal_contexts_max_k && contexts[index].device && !context; ++index)
        if (contexts[index].device == device) context = &contexts[index];
    if (!context && index != nk_metal_contexts_max_k) {
        void *const queue = nk_metal_get_(device, "newCommandQueue");
        if (queue) {
            context = &contexts[index];
            context->device = nk_metal_get_(device, "retain");
            context->queue = queue;
        }
    }
    os_unfair_lock_unlock(contexts_lock);
    if (!stream) nk_metal_do_(device, "release");
    return context;
}

NUMKONG_INLINE nk_status_t nk_memory_allocate_unified_metal_(nk_size_t bytes, void **pointer, void *stream) {
    *pointer = NULL;
    if (!bytes) return nk_success_k;
    nk_metal_context_t *const context = nk_metal_context_(stream);
    if (!context) return nk_missing_gpu_k;
    nk_size_t const shared_storage = 0; // `MTLResourceStorageModeShared`
    void *const buffer = ((void *(*)(void *, SEL, nk_size_t, nk_size_t))objc_msgSend)(
        context->device, sel_registerName("newBufferWithLength:options:"), bytes, shared_storage);
    if (!buffer) return nk_bad_alloc_k;
    char *const host = (char *)nk_metal_get_(buffer, "contents");
    os_unfair_lock_lock(&context->lock);
    nk_metal_block_t *const blocks = (nk_metal_block_t *)nk_metal_reserve_(
        context->blocks, context->blocks_count, &context->blocks_capacity, sizeof(nk_metal_block_t));
    if (blocks) {
        nk_size_t position = context->blocks_count;
        while (position != 0 && blocks[position - 1].host > host) --position;
        memmove(blocks + position + 1, blocks + position,
                (context->blocks_count - position) * sizeof(nk_metal_block_t));
        blocks[position].buffer = buffer, blocks[position].host = host, blocks[position].bytes = bytes;
        context->blocks = blocks, ++context->blocks_count;
    }
    os_unfair_lock_unlock(&context->lock);
    if (!blocks) {
        nk_metal_do_(buffer, "release");
        return nk_bad_alloc_k;
    }
    *pointer = host;
    return nk_success_k;
}

NUMKONG_INLINE nk_status_t nk_memory_free_unified_metal_(void *pointer, nk_size_t bytes, void *stream) {
    nk_unused_(bytes);
    if (!pointer) return nk_success_k;
    nk_metal_context_t *const context = nk_metal_context_(stream);
    if (!context) return nk_missing_gpu_k;
    os_unfair_lock_lock(&context->lock);
    nk_size_t index = 0;
    while (index != context->blocks_count && context->blocks[index].host != (char *)pointer) ++index;
    nk_status_t status = index != context->blocks_count ? nk_success_k : nk_device_memory_mismatch_k;
    nk_metal_pending_t *const pending = status == nk_success_k
                                            ? nk_metal_pending_(context, stream ? stream : context->queue)
                                            : NULL;
    void **const frees = pending ? (void **)nk_metal_reserve_(pending->frees, pending->frees_count,
                                                              &pending->frees_capacity, sizeof(void *))
                                 : NULL;
    if (pending && !frees) status = nk_bad_alloc_k;
    void *const buffer = status == nk_success_k ? context->blocks[index].buffer : NULL;
    if (buffer) {
        memmove(context->blocks + index, context->blocks + index + 1,
                (context->blocks_count - index - 1) * sizeof(nk_metal_block_t));
        --context->blocks_count;
    }
    if (buffer && frees) pending->frees = frees, frees[pending->frees_count++] = buffer;
    os_unfair_lock_unlock(&context->lock);
    if (buffer && !frees) nk_metal_do_(buffer, "release");
    return status;
}

NUMKONG_INLINE void *nk_allocate_unified_metal_(nk_size_t bytes, void *handle, void *stream) {
    nk_unused_(handle);
    void *pointer = NUMKONG_NULL;
    nk_unused_(nk_memory_allocate_unified_metal_(bytes, &pointer, stream));
    return pointer;
}

NUMKONG_INLINE void nk_free_unified_metal_(void *pointer, nk_size_t bytes, void *handle, void *stream) {
    nk_unused_(handle);
    nk_unused_(nk_memory_free_unified_metal_(pointer, bytes, stream));
}

NUMKONG_INLINE nk_status_t nk_allocator_init_unified_metal_(nk_allocator_t *allocator) {
    allocator->allocate = nk_allocate_unified_metal_;
    allocator->free = nk_free_unified_metal_;
    allocator->handle = NUMKONG_NULL;
    return nk_success_k;
}

NUMKONG_INLINE nk_status_t nk_stream_synchronize_metal_(void *stream) {
    nk_metal_context_t *const context = nk_metal_context_(stream);
    if (!context) return nk_missing_gpu_k;
    void *const queue = stream ? stream : context->queue;
    void **commands = NULL;
    nk_size_t commands_count = 0;
    os_unfair_lock_lock(&context->lock);
    nk_metal_pending_t *pending = nk_metal_pending_(context, queue);
    if (pending) {
        commands = pending->commands, commands_count = pending->commands_count;
        pending->commands = NULL, pending->commands_count = 0, pending->commands_capacity = 0;
        ++pending->waiters;
    }
    os_unfair_lock_unlock(&context->lock);
    if (!pending) return nk_success_k;

    nk_status_t status = nk_success_k;
    nk_size_t const completed = 4; // `MTLCommandBufferStatusCompleted`
    for (nk_size_t index = 0; index != commands_count; ++index) {
        nk_metal_do_(commands[index], "waitUntilCompleted");
        if (nk_metal_count_(commands[index], "status") != completed) status = nk_device_code_mismatch_k;
        nk_metal_do_(commands[index], "release");
    }
    free(commands);

    nk_metal_pending_t drained;
    memset(&drained, 0, sizeof(drained));
    os_unfair_lock_lock(&context->lock);
    // Other launches may have moved the entry while commands completed.
    pending = nk_metal_pending_(context, queue);
    if (--pending->waiters == 0 && pending->commands_count == 0)
        drained = *pending, *pending = context->pending[--context->pending_count];
    os_unfair_lock_unlock(&context->lock);
    for (nk_size_t index = 0; index != drained.frees_count; ++index) nk_metal_do_(drained.frees[index], "release");
    if (drained.queue) nk_metal_do_(drained.queue, "release");
    free(drained.commands);
    free(drained.frees);
    return status;
}

#if NUMKONG_HEADER_ONLY
NUMKONG_API nk_status_t nk_memory_allocate_unified_metal(nk_size_t bytes, void **pointer, void *stream) {
    return nk_memory_allocate_unified_metal_(bytes, pointer, stream);
}

NUMKONG_API nk_status_t nk_memory_free_unified_metal(void *pointer, nk_size_t bytes, void *stream) {
    return nk_memory_free_unified_metal_(pointer, bytes, stream);
}

NUMKONG_API nk_status_t nk_allocator_init_unified_metal(nk_allocator_t *allocator) {
    return nk_allocator_init_unified_metal_(allocator);
}

NUMKONG_API nk_status_t nk_stream_synchronize_metal(void *stream) { return nk_stream_synchronize_metal_(stream); }
#endif // NUMKONG_HEADER_ONLY

#pragma endregion Context

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // NUMKONG_ARCH_METAL_
#endif // NUMKONG_METAL_H
