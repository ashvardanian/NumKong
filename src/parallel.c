/**
 *  @file src/parallel.c
 *  @author Ash Vardanian
 *  @date August 6, 2026
 *  @brief Tile-parallel execution for NumKong language bindings.
 *
 *  One shared tile counter drives every backend, so dynamic scheduling and the thread-count cap
 *  behave identically no matter which pool runs the tiles.
 */
#include "parallel.h"

#if NUMKONG_OS_APPLE_
#include <dispatch/dispatch.h> // `dispatch_apply_f`, part of libSystem
#include <unistd.h>            // `sysconf`
#elif NUMKONG_OS_WINDOWS_
#include <windows.h> // `CreateThreadpoolWork`, part of kernel32
#else
#include <unistd.h> // `sysconf`
#if defined(_OPENMP)
#include <omp.h>
#else
#include <pthread.h> // `pthread_create`, `pthread_join`
#include <stdlib.h>  // `malloc`, `free`
#endif
#endif

/** OpenMP schedules tiles itself, so the counter below is for the other pools. */
#if !NUMKONG_OS_APPLE_ && !NUMKONG_OS_WINDOWS_ && defined(_OPENMP)
#define NUMKONG_PARALLEL_VIA_OPENMP 1
#else
#define NUMKONG_PARALLEL_VIA_OPENMP 0
#endif

#if !NUMKONG_PARALLEL_VIA_OPENMP

#pragma region Tile Queue

/** Tiles remaining, the body that consumes them, and the first failure. */
typedef struct nk_tile_queue_t {
    nk_tile_body_t body;
    void *context;
    nk_size_t tile_count;
#if defined(_MSC_VER)
    volatile long long next_tile;
    volatile long failure;
#else
    nk_size_t next_tile;
    nk_status_t failure;
#endif
} nk_tile_queue_t;

/** Claim and run tiles until the queue is empty or some tile failed, keeping the first failure.
 *  Safe to call from every worker at once. */
static void nk_tile_queue_drain_(nk_tile_queue_t *queue) {
    for (;;) {
#if defined(_MSC_VER)
        if (queue->failure != nk_success_k) return;
        nk_size_t const tile_index = (nk_size_t)_InterlockedExchangeAdd64(&queue->next_tile, 1);
#else
        if (__atomic_load_n(&queue->failure, __ATOMIC_RELAXED) != nk_success_k) return;
        nk_size_t const tile_index = __atomic_fetch_add(&queue->next_tile, 1, __ATOMIC_RELAXED);
#endif
        if (tile_index >= queue->tile_count) return;
        nk_status_t const status = queue->body(tile_index, queue->context);
        if (status == nk_success_k) continue;
#if defined(_MSC_VER)
        _InterlockedCompareExchange(&queue->failure, (long)status, (long)nk_success_k);
#else
        nk_status_t expected = nk_success_k;
        __atomic_compare_exchange_n(&queue->failure, &expected, status, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
#endif
        return;
    }
}

#pragma endregion Tile Queue

#endif // !NUMKONG_PARALLEL_VIA_OPENMP

#pragma region Platform Pools

#if NUMKONG_OS_APPLE_

/** libdispatch hands each worker an index it does not need; the queue picks its tiles. */
static void nk_tile_worker_dispatch_(void *context, size_t worker_index) {
    nk_unused_(worker_index);
    nk_tile_queue_drain_((nk_tile_queue_t *)context);
}

#elif NUMKONG_OS_WINDOWS_

static void CALLBACK nk_tile_worker_threadpool_(PTP_CALLBACK_INSTANCE instance, PVOID context, PTP_WORK work) {
    (void)instance;
    (void)work;
    nk_tile_queue_drain_((nk_tile_queue_t *)context);
}

#elif !NUMKONG_PARALLEL_VIA_OPENMP

static void *nk_tile_worker_posix_(void *queue) {
    nk_tile_queue_drain_((nk_tile_queue_t *)queue);
    return NULL;
}

#endif

nk_size_t nk_parallel_concurrency(void) {
#if NUMKONG_OS_WINDOWS_
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

    // A single worker needs no pool, no atomics, and no launch latency.
    if (threads <= 1) {
        for (nk_size_t tile_index = 0; tile_index < tile_count; ++tile_index) {
            nk_status_t const status = body(tile_index, context);
            if (status != nk_success_k) return status;
        }
        return nk_success_k;
    }

#if NUMKONG_PARALLEL_VIA_OPENMP
    // OpenMP's own dynamic queue beats draining a shared counter.
    nk_status_t failure = nk_success_k;
    long long const tile_limit = (long long)tile_count;
#pragma omp parallel for schedule(dynamic, 1) num_threads((int)threads)
    for (long long tile_index = 0; tile_index < tile_limit; ++tile_index) {
        if (__atomic_load_n(&failure, __ATOMIC_RELAXED) != nk_success_k) continue;
        nk_status_t const status = body((nk_size_t)tile_index, context);
        nk_status_t expected = nk_success_k;
        if (status != nk_success_k)
            __atomic_compare_exchange_n(&failure, &expected, status, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    }
    return failure;
#else
    nk_tile_queue_t queue;
    queue.body = body;
    queue.context = context;
    queue.tile_count = tile_count;
    queue.next_tile = 0;
    queue.failure = nk_success_k;

#if NUMKONG_OS_APPLE_
    dispatch_apply_f((size_t)threads, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &queue,
                     nk_tile_worker_dispatch_);
#elif NUMKONG_OS_WINDOWS_
    PTP_WORK work = CreateThreadpoolWork(nk_tile_worker_threadpool_, &queue, NULL);
    if (work) {
        // The calling thread takes a share too, so only the extra workers are submitted.
        for (nk_size_t worker = 1; worker < threads; ++worker) SubmitThreadpoolWork(work);
        nk_tile_queue_drain_(&queue);
        WaitForThreadpoolWorkCallbacks(work, FALSE);
        CloseThreadpoolWork(work);
    }
    else nk_tile_queue_drain_(&queue); // Out of pool resources — the caller still needs the result
#else
    // The calling thread takes a share too, so only the extra workers get threads of their own.
    // Any that fail to start, or all when the handles get no memory, leave their tiles to the rest.
    pthread_t *workers = (pthread_t *)malloc((threads - 1) * sizeof(pthread_t));
    nk_size_t started = 0;
    if (workers)
        while (started + 1 < threads && pthread_create(&workers[started], NULL, nk_tile_worker_posix_, &queue) == 0)
            ++started;
    nk_tile_queue_drain_(&queue);
    for (nk_size_t worker = 0; worker != started; ++worker) pthread_join(workers[worker], NULL);
    free(workers);
#endif
    return (nk_status_t)queue.failure;
#endif // NUMKONG_PARALLEL_VIA_OPENMP
}

#pragma endregion Platform Pools

#pragma region Dots Tiles

static nk_status_t nk_dots_pack_tile_(nk_size_t tile_index, void *context) {
    nk_dots_pack_task_t const *task = (nk_dots_pack_task_t const *)context;
    nk_size_t const columns_begin = tile_index * NUMKONG_PARALLEL_PACKED_TILE;
    nk_size_t const columns_end = nk_min_of_two(columns_begin + NUMKONG_PARALLEL_PACKED_TILE, task->columns);
    return task->kernel(task->b, task->columns, task->depth, task->b_stride, task->b_packed, columns_begin, columns_end,
                        task->stream);
}

nk_status_t nk_parallel_dots_pack(nk_dots_pack_task_t const *task, nk_size_t threads) {
    // The tile starting at column 0 writes the header, so an empty B still gets that one tile.
    nk_size_t const tile_count = nk_size_divide_round_up_(task->columns, NUMKONG_PARALLEL_PACKED_TILE);
    return nk_parallel_for_tiles(tile_count ? tile_count : 1, threads, nk_dots_pack_tile_, (void *)task);
}

static nk_status_t nk_dots_packed_tile_(nk_size_t tile_index, void *context) {
    nk_dots_packed_task_t const *task = (nk_dots_packed_task_t const *)context;
    nk_size_t const first_row = tile_index * NUMKONG_PARALLEL_PACKED_TILE;
    nk_size_t const rows = nk_min_of_two((nk_size_t)NUMKONG_PARALLEL_PACKED_TILE, task->rows - first_row);
    return task->kernel(task->a + first_row * task->a_stride, task->b_packed, task->c + first_row * task->c_stride,
                        rows, task->columns, task->depth, task->a_stride, task->c_stride, NUMKONG_NULL, 0,
                        task->stream);
}

nk_status_t nk_parallel_dots_packed(nk_dots_packed_task_t const *task, nk_size_t threads) {
    if (task->workspace_bytes)
        return task->kernel(task->a, task->b_packed, task->c, task->rows, task->columns, task->depth, task->a_stride,
                            task->c_stride, task->workspace, task->workspace_bytes, task->stream);
    return nk_parallel_for_tiles(nk_size_divide_round_up_(task->rows, NUMKONG_PARALLEL_PACKED_TILE), threads,
                                 nk_dots_packed_tile_, (void *)task);
}

static nk_status_t nk_dots_symmetric_tile_(nk_size_t tile_index, void *context) {
    nk_dots_symmetric_task_t const *task = (nk_dots_symmetric_task_t const *)context;
    nk_size_t const tile_begin = task->rows_begin + tile_index * NUMKONG_PARALLEL_SYMMETRIC_TILE;
    nk_size_t const tile_end = nk_min_of_two(tile_begin + NUMKONG_PARALLEL_SYMMETRIC_TILE, task->rows_end);
    return task->kernel(task->vectors, task->vector_count, task->depth, task->vectors_stride, task->result,
                        task->result_stride, tile_begin, tile_end, NUMKONG_NULL, 0, task->stream);
}

nk_status_t nk_parallel_dots_symmetric(nk_dots_symmetric_task_t const *task, nk_size_t threads) {
    if (task->workspace_bytes)
        return task->kernel(task->vectors, task->vector_count, task->depth, task->vectors_stride, task->result,
                            task->result_stride, task->rows_begin, task->rows_end, task->workspace,
                            task->workspace_bytes, task->stream);
    nk_size_t const rows = task->rows_end - task->rows_begin;
    return nk_parallel_for_tiles(nk_size_divide_round_up_(rows, NUMKONG_PARALLEL_SYMMETRIC_TILE), threads,
                                 nk_dots_symmetric_tile_, (void *)task);
}

#pragma endregion Dots Tiles
