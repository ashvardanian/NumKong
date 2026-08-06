/**
 *  @brief Tile-parallel execution for NumKong language bindings.
 *  @file c/parallel.h
 *  @author Ash Vardanian
 *  @date August 6, 2026
 *
 *  Compiled only by `setup.py` and `binding.gyp`, never by CMake or Cargo.
 *  Each platform's own pool runs the tiles, so no binding ships a `libomp`.
 */
#ifndef NUMKONG_PARALLEL_H
#define NUMKONG_PARALLEL_H

#include <numkong/types.h> // `nk_size_t`

#ifdef __cplusplus
extern "C" {
#endif

/*  Row-tile sizes: 2 kernel tile blocks per chunk, packed blocking 2×2 rows of 16. */
#define NK_PARALLEL_PACKED_TILE    64
#define NK_PARALLEL_SYMMETRIC_TILE 32

/**
 *  @brief Work for one tile. Must be reentrant and confine writes to its own tile.
 *  @param tile_index Zero-based index below the `tile_count` passed to the scheduler.
 *  @param context Caller-owned state, shared unsynchronized across all tiles.
 */
typedef void (*nk_tile_body_t)(nk_size_t tile_index, void *context);

/** @brief Logical processors available to this process, or 1 when undetectable. */
nk_size_t nk_parallel_concurrency(void);

/**
 *  @brief Run @p tile_count independent tiles, at most @p threads of them at a time.
 *
 *  Tiles are claimed from a shared counter, giving the dynamic scheduling uneven
 *  workloads need. @p threads of 0 means every logical processor, and 1 runs the
 *  tiles inline on the calling thread with no pool involved. Returns once every
 *  tile has completed. The caller must release the GIL if the bodies do not need it.
 */
void nk_parallel_for_tiles(nk_size_t tile_count, nk_size_t threads, nk_tile_body_t body, void *context);

#ifdef __cplusplus
}
#endif

#endif // NUMKONG_PARALLEL_H
