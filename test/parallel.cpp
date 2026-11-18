/**
 *  @file test/parallel.cpp
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief Checks that @c nk_parallel_for_tiles runs every tile once at any thread count,
 *      and that the status of a failing tile comes back.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <atomic> // `std::atomic_ref`
#include <vector> // `std::vector`

#include "harness.hpp"
#include "parallel.h" // `nk_parallel_for_tiles`

namespace ashvardanian::numkong::test {

struct tiles_t {
    std::vector<unsigned> runs;
    nk_size_t failing_tile = static_cast<nk_size_t>(-1);
};

nk_status_t count_tile_(nk_size_t tile_index, void *context) {
    tiles_t &tiles = *static_cast<tiles_t *>(context);
    std::atomic_ref<unsigned>(tiles.runs[tile_index]).fetch_add(1, std::memory_order_relaxed);
    return tile_index == tiles.failing_tile ? nk_bad_alloc_k : nk_success_k;
}

/** Thread counts to run, the largest past any core count, so the scheduler multiplexes them. */
constexpr nk_size_t thread_counts_k[] = {1, 4, 64, 4096};
constexpr std::size_t tiles_count_k = 10000;

/** Every tile runs exactly once, and the call reports success, at every thread count. */
void test_parallel_every_tile_once() {
    for (nk_size_t threads : thread_counts_k) {
        tiles_t tiles;
        tiles.runs.assign(tiles_count_k, 0);
        nk_status_t const status = nk_parallel_for_tiles(tiles.runs.size(), threads, count_tile_, &tiles);
        std::size_t once = 0;
        for (unsigned runs : tiles.runs) once += runs == 1;
        expect_equal(status, nk_success_k, "{} threads", threads);
        expect_equal(once, tiles.runs.size(), "{} threads", threads);
    }
}

/** The status of a failing tile comes back from the call, at every thread count. */
void test_parallel_failing_tile() {
    for (nk_size_t threads : thread_counts_k) {
        tiles_t tiles;
        tiles.runs.assign(tiles_count_k, 0);
        tiles.failing_tile = 777;
        expect_equal(nk_parallel_for_tiles(tiles.runs.size(), threads, count_tile_, &tiles), nk_bad_alloc_k,
                     "{} threads", threads);
    }
}

} // namespace ashvardanian::numkong::test

int main(int argc, char **argv) {
    using namespace ashvardanian::numkong::test;
    start_test_run(argc, argv);
    run_test_case("test_parallel_every_tile_once", test_parallel_every_tile_once);
    run_test_case("test_parallel_failing_tile", test_parallel_failing_tile);
    return finish_test_run();
}
