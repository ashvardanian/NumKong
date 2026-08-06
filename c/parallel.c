/**
 *  @brief Tile-parallel execution for NumKong language bindings.
 *  @file c/parallel.c
 *  @author Ash Vardanian
 *  @date August 6, 2026
 *
 *  One shared tile counter drives every backend, so dynamic scheduling and the
 *  thread-count cap behave identically no matter which pool runs the tiles.
 */
#include "parallel.h"

#if defined(__APPLE__)
#include <dispatch/dispatch.h> // `dispatch_apply_f`, part of libSystem
#include <unistd.h>            // `sysconf`
#elif defined(_WIN32)
#include <windows.h> // `CreateThreadpoolWork`, part of kernel32
#else
#include <unistd.h> // `sysconf`
#if defined(_OPENMP)
#include <omp.h>
#endif
#endif

/*  OpenMP schedules tiles itself, so the counter below is for the other pools. */
#if !defined(__APPLE__) && !defined(_WIN32) && defined(_OPENMP)
#define NK_PARALLEL_VIA_OPENMP 1
#else
#define NK_PARALLEL_VIA_OPENMP 0
#endif

#if !NK_PARALLEL_VIA_OPENMP

#pragma region Tile Queue

/** @brief Tiles remaining, plus the body that consumes them. */
typedef struct nk_tile_queue_t {
    nk_tile_body_t body;
    void *context;
    nk_size_t tile_count;
#if defined(_MSC_VER)
    volatile long long next_tile;
#else
    nk_size_t next_tile;
#endif
} nk_tile_queue_t;

/** @brief Claim and run tiles until the queue is empty. Safe to call from every worker at once. */
static void nk_tile_queue_drain_(nk_tile_queue_t *queue) {
    for (;;) {
#if defined(_MSC_VER)
        nk_size_t const tile_index = (nk_size_t)_InterlockedExchangeAdd64(&queue->next_tile, 1);
#else
        nk_size_t const tile_index = __atomic_fetch_add(&queue->next_tile, 1, __ATOMIC_RELAXED);
#endif
        if (tile_index >= queue->tile_count) return;
        queue->body(tile_index, queue->context);
    }
}

#pragma endregion Tile Queue

#endif // !NK_PARALLEL_VIA_OPENMP

#pragma region Platform Pools

#if defined(__APPLE__)

/** @brief libdispatch hands each worker its index; the queue decides what it runs. */
static void nk_tile_worker_dispatch_(void *context, size_t worker_index) {
    (void)worker_index;
    nk_tile_queue_drain_((nk_tile_queue_t *)context);
}

#elif defined(_WIN32)

static void CALLBACK nk_tile_worker_threadpool_(PTP_CALLBACK_INSTANCE instance, PVOID context, PTP_WORK work) {
    (void)instance;
    (void)work;
    nk_tile_queue_drain_((nk_tile_queue_t *)context);
}

#endif

nk_size_t nk_parallel_concurrency(void) {
#if defined(_WIN32)
    DWORD const count = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    return count ? (nk_size_t)count : 1;
#elif defined(_SC_NPROCESSORS_ONLN)
    long const count = sysconf(_SC_NPROCESSORS_ONLN);
    return count > 0 ? (nk_size_t)count : 1;
#else
    return 1;
#endif
}

void nk_parallel_for_tiles(nk_size_t tile_count, nk_size_t threads, nk_tile_body_t body, void *context) {
    if (!tile_count) return;
    if (threads == 0) threads = nk_parallel_concurrency();
    if (threads > tile_count) threads = tile_count;

    // A single worker needs no pool, no atomics, and no launch latency.
    if (threads <= 1) {
        for (nk_size_t tile_index = 0; tile_index < tile_count; ++tile_index) body(tile_index, context);
        return;
    }

#if NK_PARALLEL_VIA_OPENMP
    // OpenMP's own dynamic queue beats draining a shared counter.
    long long const tile_limit = (long long)tile_count;
#pragma omp parallel for schedule(dynamic, 1) num_threads((int)threads)
    for (long long tile_index = 0; tile_index < tile_limit; ++tile_index) body((nk_size_t)tile_index, context);
#else
    nk_tile_queue_t queue;
    queue.body = body;
    queue.context = context;
    queue.tile_count = tile_count;
    queue.next_tile = 0;

#if defined(__APPLE__)
    dispatch_apply_f((size_t)threads, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &queue,
                     nk_tile_worker_dispatch_);
#elif defined(_WIN32)
    PTP_WORK work = CreateThreadpoolWork(nk_tile_worker_threadpool_, &queue, NULL);
    if (!work) {
        nk_tile_queue_drain_(&queue); // Out of pool resources — the caller still needs the result
        return;
    }
    // The calling thread takes a share too, so only the extra workers are submitted.
    for (nk_size_t worker = 1; worker < threads; ++worker) SubmitThreadpoolWork(work);
    nk_tile_queue_drain_(&queue);
    WaitForThreadpoolWorkCallbacks(work, FALSE);
    CloseThreadpoolWork(work);
#else
    nk_tile_queue_drain_(&queue);
#endif
#endif // NK_PARALLEL_VIA_OPENMP
}

#pragma endregion Platform Pools
