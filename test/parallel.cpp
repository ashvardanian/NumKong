/**
 *  @file test/parallel.cpp
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief Checks that @c nk_parallel_for_tiles runs every tile once at any thread count,
 *      and that the status of a failing tile comes back.
 */
#include <cstdio>  // `std::fprintf`
#include <cstdlib> // `EXIT_SUCCESS`, `EXIT_FAILURE`

#include <atomic> // `std::atomic_ref`
#include <vector> // `std::vector`

#include "parallel.h" // `nk_parallel_for_tiles`

namespace {

struct tiles_t {
    std::vector<unsigned> runs;
    nk_size_t failing_tile = static_cast<nk_size_t>(-1);
};

nk_status_t count_tile(nk_size_t tile_index, void *context) {
    tiles_t &tiles = *static_cast<tiles_t *>(context);
    std::atomic_ref<unsigned>(tiles.runs[tile_index]).fetch_add(1, std::memory_order_relaxed);
    return tile_index == tiles.failing_tile ? nk_bad_alloc_k : nk_success_k;
}

} // namespace

int main() {
    int failures = 0;
    // 4096 threads exceeds any core count here, so the scheduler multiplexes the threads.
    for (nk_size_t threads : {nk_size_t {1}, nk_size_t {4}, nk_size_t {64}, nk_size_t {4096}}) {
        tiles_t tiles;
        tiles.runs.assign(10000, 0);
        nk_status_t const status = nk_parallel_for_tiles(tiles.runs.size(), threads, count_tile, &tiles);
        nk_size_t once = 0;
        for (unsigned runs : tiles.runs) once += runs == 1;
        if (status != nk_success_k || once != tiles.runs.size()) {
            std::fprintf(stderr, "%zu threads: status %d, %zu of %zu tiles ran once\n", static_cast<size_t>(threads),
                         static_cast<int>(status), static_cast<size_t>(once), tiles.runs.size());
            ++failures;
        }

        tiles.runs.assign(10000, 0);
        tiles.failing_tile = 777;
        if (nk_parallel_for_tiles(tiles.runs.size(), threads, count_tile, &tiles) != nk_bad_alloc_k) {
            std::fprintf(stderr, "%zu threads: the failing tile's status was lost\n", static_cast<size_t>(threads));
            ++failures;
        }
    }
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
