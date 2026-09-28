/**
 *  @file c/parallel.c
 *  @author Ash Vardanian
 *  @date August 6, 2026
 *  @brief Tile-parallel execution for NumKong language bindings.
 *
 *  One shared tile counter drives every backend, so dynamic scheduling and the thread-count cap
 *  behave identically no matter which pool runs the tiles.
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

/** OpenMP schedules tiles itself, so the counter below is for the other pools. */
#if !defined(__APPLE__) && !defined(_WIN32) && defined(_OPENMP)
#define NUMKONG_PARALLEL_VIA_OPENMP 1
#else
#define NUMKONG_PARALLEL_VIA_OPENMP 0
#endif

#if !NUMKONG_PARALLEL_VIA_OPENMP

#pragma region Tile Queue

/** Tiles remaining, the body that consumes them, and one status slot per worker. */
typedef struct nk_tile_queue_t {
    nk_tile_body_t body;
    void *context;
    nk_size_t tile_count;
    nk_status_t *statuses;
#if defined(_MSC_VER)
    volatile long long next_tile;
#else
    nk_size_t next_tile;
#endif
#if defined(_WIN32)
    volatile long long next_worker;
#endif
} nk_tile_queue_t;

/** Claim and run tiles until the queue is empty or one fails, whose status only this worker's slot
 *  receives. Safe to call from every worker at once, each with its own @p worker. */
static void nk_tile_queue_drain_(nk_tile_queue_t *queue, nk_size_t worker) {
    for (;;) {
#if defined(_MSC_VER)
        nk_size_t const tile_index = (nk_size_t)_InterlockedExchangeAdd64(&queue->next_tile, 1);
#else
        nk_size_t const tile_index = __atomic_fetch_add(&queue->next_tile, 1, __ATOMIC_RELAXED);
#endif
        if (tile_index >= queue->tile_count) return;
        nk_status_t const status = queue->body(tile_index, queue->context);
        if (status != nk_success_k) {
            queue->statuses[worker] = status;
            return;
        }
    }
}

#pragma endregion Tile Queue

#endif // !NUMKONG_PARALLEL_VIA_OPENMP

#pragma region Platform Pools

#if defined(__APPLE__)

/** libdispatch hands each worker its index, also its status slot; the queue picks its tiles. */
static void nk_tile_worker_dispatch_(void *context, size_t worker_index) {
    nk_tile_queue_drain_((nk_tile_queue_t *)context, (nk_size_t)worker_index);
}

#elif defined(_WIN32)

/** The pool passes no index, so each submitted worker claims its status slot from a counter. */
static void CALLBACK nk_tile_worker_threadpool_(PTP_CALLBACK_INSTANCE instance, PVOID context, PTP_WORK work) {
    (void)instance;
    (void)work;
    nk_tile_queue_t *queue = (nk_tile_queue_t *)context;
#if defined(_MSC_VER)
    nk_size_t const worker = (nk_size_t)_InterlockedIncrement64(&queue->next_worker);
#else
    nk_size_t const worker = (nk_size_t)__atomic_add_fetch(&queue->next_worker, 1, __ATOMIC_RELAXED);
#endif
    nk_tile_queue_drain_(queue, worker);
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

nk_status_t nk_parallel_for_tiles(nk_size_t tile_count, nk_size_t threads, nk_tile_body_t body, void *context) {
    if (!tile_count) return nk_success_k;
    if (threads == 0) threads = nk_parallel_concurrency();
    if (threads > tile_count) threads = tile_count;
    if (threads > NUMKONG_PARALLEL_MAX_THREADS) threads = NUMKONG_PARALLEL_MAX_THREADS;

    // A single worker needs no pool, no atomics, and no launch latency.
    if (threads <= 1) {
        for (nk_size_t tile_index = 0; tile_index < tile_count; ++tile_index) {
            nk_status_t const status = body(tile_index, context);
            if (status != nk_success_k) return status;
        }
        return nk_success_k;
    }

    nk_status_t statuses[NUMKONG_PARALLEL_MAX_THREADS];
    for (nk_size_t worker = 0; worker < threads; ++worker) statuses[worker] = nk_success_k;

#if NUMKONG_PARALLEL_VIA_OPENMP
    // OpenMP's own dynamic queue beats draining a shared counter.
    long long const tile_limit = (long long)tile_count;
#pragma omp parallel for schedule(dynamic, 1) num_threads((int)threads)
    for (long long tile_index = 0; tile_index < tile_limit; ++tile_index) {
        nk_status_t *slot = &statuses[omp_get_thread_num()];
        if (*slot != nk_success_k) continue;
        nk_status_t const status = body((nk_size_t)tile_index, context);
        if (status != nk_success_k) *slot = status;
    }
#else
    nk_tile_queue_t queue;
    queue.body = body;
    queue.context = context;
    queue.tile_count = tile_count;
    queue.statuses = statuses;
    queue.next_tile = 0;

#if defined(__APPLE__)
    dispatch_apply_f((size_t)threads, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &queue,
                     nk_tile_worker_dispatch_);
#elif defined(_WIN32)
    queue.next_worker = 0;
    PTP_WORK work = CreateThreadpoolWork(nk_tile_worker_threadpool_, &queue, NULL);
    if (!work) {
        nk_tile_queue_drain_(&queue, 0); // Out of pool resources — the caller still needs the result
        return statuses[0];
    }
    // The calling thread takes a share too, as worker 0, so only the extra workers are submitted.
    for (nk_size_t worker = 1; worker < threads; ++worker) SubmitThreadpoolWork(work);
    nk_tile_queue_drain_(&queue, 0);
    WaitForThreadpoolWorkCallbacks(work, FALSE);
    CloseThreadpoolWork(work);
#else
    nk_tile_queue_drain_(&queue, 0);
#endif
#endif // NUMKONG_PARALLEL_VIA_OPENMP

    for (nk_size_t worker = 0; worker < threads; ++worker)
        if (statuses[worker] != nk_success_k) return statuses[worker];
    return nk_success_k;
}

#pragma endregion Platform Pools
