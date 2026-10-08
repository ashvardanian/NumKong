/**
 *  @file c/parallel.h
 *  @author Ash Vardanian
 *  @date August 6, 2026
 *  @brief Tile-parallel execution for NumKong language bindings.
 *
 *  Compiled into the Python and Node extensions, never into the libraries.
 *  Each platform's own pool runs the tiles, so no binding ships a @c libomp.
 */
#ifndef NUMKONG_PARALLEL_H
#define NUMKONG_PARALLEL_H

#include "numkong/capabilities.h" // `nk_dots_packed_punned_t`, `nk_dots_symmetric_punned_t`

#ifdef __cplusplus
extern "C" {
#endif

/** Row-tile sizes: 2 kernel tile blocks per chunk, packed blocking 2×2 rows of 16. */
#define NUMKONG_PARALLEL_PACKED_TILE    64
#define NUMKONG_PARALLEL_SYMMETRIC_TILE 32

/** Most workers one call runs, as each keeps its first failure in a slot on the caller's stack. */
#define NUMKONG_PARALLEL_MAX_THREADS 1024

/**
 *  @brief Work for one tile. Must be reentrant and confine writes to its own tile.
 *  @param[in] tile_index Zero-based index below the @c tile_count passed to the scheduler.
 *  @param[in] context Caller-owned state, shared unsynchronized across all tiles.
 *  @return @c nk_success_k, or a failure that stops this worker from claiming more tiles.
 */
typedef nk_status_t (*nk_tile_body_t)(nk_size_t tile_index, void *context);

/** Logical processors available to this process, or 1 when undetectable. */
nk_size_t nk_parallel_concurrency(void);

/**
 *  @brief Run @p tile_count independent tiles, at most @p threads of them at a time.
 *
 *  Tiles are claimed from a shared counter, giving the dynamic scheduling uneven workloads need.
 *  @p threads of 0 means every logical processor, up to @c NUMKONG_PARALLEL_MAX_THREADS, and 1 runs
 *  the tiles inline on the calling thread with no pool involved. Returns once every worker stopped,
 *  at its first failure or when no tile is left. The caller must release the GIL if the bodies do
 *  not need it.
 *
 *  @return @c nk_success_k, or the failure of the lowest-numbered worker that had one.
 */
nk_status_t nk_parallel_for_tiles(nk_size_t tile_count, nk_size_t threads, nk_tile_body_t body, void *context);

/** One C = A × Bᵀ against a packed B, for plain dtypes, named as the kernel names its arguments. */
typedef struct {
    nk_dots_packed_punned_t kernel;
    char const *a;
    void const *b_packed;
    char *c;
    nk_size_t rows;
    nk_size_t columns;
    nk_size_t depth;
    nk_size_t a_stride;
    nk_size_t c_stride;
    nk_stream_t stream;
} nk_dots_packed_task_t;

/** One Gram matrix C = A × Aᵀ over rows [rows_begin, rows_end), for plain dtypes. */
typedef struct {
    nk_dots_symmetric_punned_t kernel;
    char const *vectors;
    nk_size_t vector_count;
    nk_size_t depth;
    nk_size_t vectors_stride;
    char *result;
    nk_size_t result_stride;
    nk_size_t rows_begin;
    nk_size_t rows_end;
    nk_stream_t stream;
} nk_dots_symmetric_task_t;

/** Runs @p task in tiles of @c NUMKONG_PARALLEL_PACKED_TILE rows of A on @p threads workers. */
nk_status_t nk_parallel_dots_packed(nk_dots_packed_task_t const *task, nk_size_t threads);

/** Runs @p task in tiles of @c NUMKONG_PARALLEL_SYMMETRIC_TILE rows on @p threads workers. */
nk_status_t nk_parallel_dots_symmetric(nk_dots_symmetric_task_t const *task, nk_size_t threads);

#ifdef __cplusplus
}
#endif

#endif // NUMKONG_PARALLEL_H
