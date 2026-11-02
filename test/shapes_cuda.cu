/**
 *  @file test/shapes_cuda.cu
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief Chamfer, Hausdorff and RMSD device bodies, per warp and per thread, against the serial
 *      kernels, on CUDA and, through an include CMake generates, on ROCm.
 *
 *  The bodies are device helpers, so this unit launches its own kernels over them and keeps the
 *  exports of their headers switched off, as the library already defines them.
 */
#undef NUMKONG_TARGET_CUDA
#define NUMKONG_TARGET_CUDA 0
#undef NUMKONG_TARGET_ROCM
#define NUMKONG_TARGET_ROCM 0
#include "cross_device.hpp" // `device_shapes_packed`

#if NUMKONG_ARCH_ROCM_
#include "numkong/shapes/rocm.cuh" // `nk_chamfer_warp_packed_f32_rocm_`, `nk_chamfer_packed_f32_simt_`
#elif NUMKONG_ARCH_CUDA_
#include "numkong/shapes/cuda.cuh" // `nk_chamfer_warp_packed_f32_cuda_`, `nk_chamfer_packed_f32_simt_`
#endif

namespace ashvardanian::numkong::test {

#if NUMKONG_ARCH_ROCM_

using shapes_backend_t = rocm_backend_t;
using shapes_stream_t = hipStream_t;
inline constexpr nk_capability_t shapes_capability_k = nk_cap_rocm_k;

inline bool shapes_launched() { return hipGetLastError() == hipSuccess; }

__device__ inline nk_f64_t chamfer_warp(void const *a, void const *b, nk_f64_t limit, unsigned lane) {
    return nk_chamfer_warp_packed_f32_rocm_(a, b, limit, lane);
}
__device__ inline nk_f32_t chamfer_warp(void const *a, void const *b, nk_f32_t limit, unsigned lane) {
    return nk_chamfer_warp_packed_f16_rocm_(a, b, limit, lane);
}
__device__ inline nk_f64_t hausdorff_warp(void const *a, void const *b, nk_f64_t limit, unsigned lane) {
    return nk_hausdorff_warp_packed_f32_rocm_(a, b, limit, lane);
}
__device__ inline nk_f32_t hausdorff_warp(void const *a, void const *b, nk_f32_t limit, unsigned lane) {
    return nk_hausdorff_warp_packed_f16_rocm_(a, b, limit, lane);
}

#elif NUMKONG_ARCH_CUDA_

using shapes_backend_t = cuda_backend_t;
using shapes_stream_t = cudaStream_t;
inline constexpr nk_capability_t shapes_capability_k = nk_cap_cuda_k;

inline bool shapes_launched() { return cudaGetLastError() == cudaSuccess; }

__device__ inline nk_f64_t chamfer_warp(void const *a, void const *b, nk_f64_t limit, unsigned lane) {
    return nk_chamfer_warp_packed_f32_cuda_(a, b, limit, lane);
}
__device__ inline nk_f32_t chamfer_warp(void const *a, void const *b, nk_f32_t limit, unsigned lane) {
    return nk_chamfer_warp_packed_f16_cuda_(a, b, limit, lane);
}
__device__ inline nk_f64_t hausdorff_warp(void const *a, void const *b, nk_f64_t limit, unsigned lane) {
    return nk_hausdorff_warp_packed_f32_cuda_(a, b, limit, lane);
}
__device__ inline nk_f32_t hausdorff_warp(void const *a, void const *b, nk_f32_t limit, unsigned lane) {
    return nk_hausdorff_warp_packed_f16_cuda_(a, b, limit, lane);
}

#endif // NUMKONG_ARCH_ROCM_

#if NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_

/** The dispatch point @p best_ in the shape of its capability kernels, over this vendor's baseline
 *  capability alone, as the unit keeps the exports' declarations switched off. */
template <auto best_>
inline constexpr auto shapes_best =
    [](auto... arguments) noexcept { return call_best<best_>(shapes_capability_k, arguments...); };

/** Scores each pair of the two pack tables on its own warp, each lane keeping its own result, or
 *  with @p warp_ off on its own thread. */
template <typename scalar_type_, nk_kernel_kind_t kind_, bool warp_>
__global__ void shapes_pairs_kernel_(void const *const *a_packed, void const *const *b_packed, std::size_t pairs,
                                     typename scalar_type_::shapes_distance_t::raw_t const *limits,
                                     typename scalar_type_::shapes_distance_t::raw_t *results) {
    std::size_t const thread = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t const pair = warp_ ? thread / 32 : thread;
    unsigned const lane = threadIdx.x % 32;
    if (pair >= pairs) return;
    void const *a = a_packed[pair], *b = b_packed[pair];
    constexpr bool chamfer_k = kind_ == nk_kernel_chamfers_packed_k;
    if constexpr (warp_)
        results[thread] = chamfer_k ? chamfer_warp(a, b, limits[pair], lane) : hausdorff_warp(a, b, limits[pair], lane);
    else if constexpr (std::is_same_v<scalar_type_, f32_t>)
        results[thread] = chamfer_k ? nk_chamfer_packed_f32_simt_(a, b, limits[pair])
                                    : nk_hausdorff_packed_f32_simt_(a, b, limits[pair]);
    else
        results[thread] = chamfer_k ? nk_chamfer_packed_f16_simt_(a, b, limits[pair])
                                    : nk_hausdorff_packed_f16_simt_(a, b, limits[pair]);
}

/**
 *  @brief Every pair of the clouds above on the device against the serial kernel.
 *
 *  Unbounded calls must match within the CPU capabilities' bounds: n + m + 8 roundings of a Chamfer
 *  distance, whose sums the warp reduces in another order, and 8 of a Hausdorff distance, which
 *  differs only by the hardware square root. Bounded calls draw each pair's limit up to twice its
 *  distance, and a result past the limit must still be a lower bound of the distance, up to the 3
 *  roundings of each pack radius the Hausdorff header bound subtracts. Every lane of a warp must
 *  return the same value.
 */
template <typename scalar_type_, nk_kernel_kind_t kind_, bool warp_>
error_stats_t test_shapes_device(settings_t const &settings) {
    using distance_t = typename scalar_type_::shapes_distance_t;
    using raw_distance_t = typename distance_t::raw_t;
    constexpr bool is_f32 = std::is_same_v<scalar_type_, f32_t>;
    error_stats_t stats(comparison_family_t::bounded_k);
    auto backend = make_backend<shapes_backend_t>(settings);
    std::mt19937 generator(settings.seed.value);
    auto const packs = device_shapes_packed<scalar_type_>(settings, backend, generator, stats);
    std::size_t const count = packs.size(), pairs = count * count, lanes = warp_ ? 32 : 1;

    auto pair_packs = make_vector<void const *>(backend, 2 * pairs);
    void const **a_packed = pair_packs.values_data();
    void const **b_packed = a_packed + pairs;
    for (std::size_t pair = 0; pair != pairs; ++pair)
        a_packed[pair] = packs[pair / count], b_packed[pair] = packs[pair % count];

    std::vector<raw_distance_t> references(pairs);
    raw_distance_t const infinity = distance_t::positive_infinity().raw_;
    auto const serial = [&](auto chamfers, auto hausdorffs) {
        if constexpr (kind_ == nk_kernel_chamfers_packed_k)
            return chamfers(packs.data(), count, packs.data(), count, references.data(), count * sizeof(raw_distance_t),
                            nullptr, &infinity, nullptr, 0, nullptr);
        else
            return hausdorffs(packs.data(), count, packs.data(), count, references.data(),
                              count * sizeof(raw_distance_t), nullptr, &infinity, nullptr);
    };
    if constexpr (is_f32) stats.expect(serial(nk_chamfers_packed_f32_serial, nk_hausdorffs_packed_f32_serial));
    else stats.expect(serial(nk_chamfers_packed_f16_serial, nk_hausdorffs_packed_f16_serial));

    auto limits = make_vector<distance_t>(backend, pairs);
    auto results = make_vector<distance_t>(backend, pairs * lanes);
    std::uniform_real_distribution<nk_f64_t> fraction(0.0, 2.0);
    nk_f64_t const roundoff = half_ulp<distance_t>(1.0);
    auto const radius = [](void const *packed) {
        if constexpr (is_f32) return static_cast<nk_f64_t>(((nk_shapes_packed_header_f64_t const *)packed)->max_radius);
        else return static_cast<nk_f64_t>(((nk_shapes_packed_header_f32_t const *)packed)->max_radius);
    };
    for (int bounded = 0; bounded != 2; ++bounded) {
        for (std::size_t pair = 0; pair != pairs; ++pair)
            limits.raw_values_data()[pair] = bounded
                                                 ? static_cast<raw_distance_t>(references[pair] * fraction(generator))
                                                 : std::numeric_limits<raw_distance_t>::infinity();
        unsigned const threads = 128;
        unsigned const blocks = static_cast<unsigned>(nk_size_divide_round_up_(pairs * lanes, threads));
        shapes_pairs_kernel_<scalar_type_, kind_, warp_><<<blocks, threads, 0, (shapes_stream_t)backend.stream>>>(
            a_packed, b_packed, pairs, limits.raw_values_data(), results.raw_values_data());
        bool const launched = shapes_launched();
        stats.expect(launched, "the kernel failed to launch");
        if (!launched || !stats.expect(backend.synchronize())) return stats;
        for (std::size_t pair = 0; pair != pairs; ++pair) {
            raw_distance_t const *lane_results = results.raw_values_data() + pair * lanes;
            for (std::size_t lane = 1; lane != lanes; ++lane)
                stats.expect(std::memcmp(lane_results + lane, lane_results, sizeof(raw_distance_t)) == 0,
                             "the lanes of a warp disagree");
            std::size_t const a_points = packs.points(pair / count);
            std::size_t const b_points = packs.points(pair % count);
            nk_f64_t const reference = references[pair], result = lane_results[0];
            nk_f64_t const roundings = kind_ == nk_kernel_chamfers_packed_k ? a_points + b_points + 8 : 8;
            nk_f64_t const bound = roundings * roundoff * reference;
            if (result <= limits.raw_values_data()[pair] || !std::isfinite(reference))
                stats.accumulate_bounded(distance_t::from_raw(lane_results[0]), f64_t(reference), bound);
            else
                stats.expect(result <= reference + bound + half_ulp<distance_t>(reference) +
                                           3 * roundoff * (radius(a_packed[pair]) + radius(b_packed[pair])),
                             "a result past its limit exceeds the distance");
        }
    }
    return stats;
}

/** Scores each pair of the two pack tables on its own thread. */
template <typename scalar_type_>
__global__ void rmsds_pairs_kernel_(void const *const *a_packed, void const *const *b_packed, std::size_t pairs,
                                    typename scalar_type_::shapes_distance_t::raw_t *results) {
    std::size_t const pair = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair >= pairs) return;
    if constexpr (std::is_same_v<scalar_type_, f16_t>)
        results[pair] = nk_rmsds_pair_f16_simt_(a_packed[pair], b_packed[pair]);
    else if constexpr (std::is_same_v<scalar_type_, f32_t>)
        results[pair] = nk_rmsds_pair_f32_simt_(a_packed[pair], b_packed[pair]);
    else results[pair] = nk_rmsds_pair_f64_simt_(a_packed[pair], b_packed[pair]);
}

/**
 *  @brief Every pair of equally large clouds above on the device against the serial kernel.
 *
 *  F16 folds in F32, moving a result by the bound of @ref test_rmsds_packed_f16, and wider dtypes
 *  sum the residuals in F64, n + 16 roundings of each frame's spread. A pair of unequal sizes,
 *  which the serial kernel refuses and a launch cannot, must report NaN.
 */
template <typename scalar_type_>
error_stats_t test_rmsds_pairs(settings_t const &settings) {
    using distance_t = typename scalar_type_::shapes_distance_t;
    using raw_distance_t = typename distance_t::raw_t;
    constexpr bool is_f16 = std::is_same_v<scalar_type_, f16_t>;
    error_stats_t stats(comparison_family_t::bounded_k);
    auto backend = make_backend<shapes_backend_t>(settings);
    std::mt19937 generator(settings.seed.value);
    auto const packs = device_shapes_packed<scalar_type_>(settings, backend, generator, stats);
    std::size_t const count = packs.size();

    std::vector<void const *> rows, columns;
    for (std::size_t row = 0; row != count; ++row)
        for (std::size_t column = 0; column != count; ++column)
            if (packs.points(row) == packs.points(column)) rows.push_back(packs[row]), columns.push_back(packs[column]);
    std::size_t const equal_pairs = rows.size(), pairs = equal_pairs + 1;
    auto pair_packs = make_vector<void const *>(backend, 2 * pairs);
    void const **a_packed = pair_packs.values_data();
    void const **b_packed = a_packed + pairs;
    std::copy(rows.begin(), rows.end(), a_packed);
    std::copy(columns.begin(), columns.end(), b_packed);
    a_packed[equal_pairs] = packs[0], b_packed[equal_pairs] = packs[count - 1];

    auto results = make_vector<distance_t>(backend, pairs);
    unsigned const threads = 128;
    unsigned const blocks = static_cast<unsigned>(nk_size_divide_round_up_(pairs, threads));
    rmsds_pairs_kernel_<scalar_type_>
        <<<blocks, threads, 0, (shapes_stream_t)backend.stream>>>(a_packed, b_packed, pairs, results.raw_values_data());
    bool const launched = shapes_launched();
    stats.expect(launched, "the kernel failed to launch");
    if (!launched || !stats.expect(backend.synchronize())) return stats;

    auto const radius = [](void const *packed) {
        if constexpr (is_f16) {
            nk_f32_t const *spreads = ((nk_shapes_packed_header_f32_t const *)packed)->spreads;
            return std::sqrt(static_cast<nk_f64_t>(spreads[0]) + spreads[1] + spreads[2]);
        }
        else {
            nk_f64_t const *spreads = ((nk_shapes_packed_header_f64_t const *)packed)->spreads;
            return std::sqrt(spreads[0] + spreads[1] + spreads[2]);
        }
    };
    for (std::size_t pair = 0; pair != equal_pairs; ++pair) {
        raw_distance_t reference = 0;
        if constexpr (is_f16)
            stats.expect(nk_rmsds_packed_f16_serial(a_packed + pair, 1, b_packed + pair, 1, &reference,
                                                    sizeof(raw_distance_t), nullptr));
        else if constexpr (std::is_same_v<scalar_type_, f32_t>)
            stats.expect(nk_rmsds_packed_f32_serial(a_packed + pair, 1, b_packed + pair, 1, &reference,
                                                    sizeof(raw_distance_t), nullptr));
        else
            stats.expect(nk_rmsds_packed_f64_serial(a_packed + pair, 1, b_packed + pair, 1, &reference,
                                                    sizeof(raw_distance_t), nullptr));
        std::size_t const points = is_f16 ? ((nk_shapes_packed_header_f32_t const *)a_packed[pair])->points
                                          : ((nk_shapes_packed_header_f64_t const *)a_packed[pair])->points;
        nk_f64_t const spread = radius(a_packed[pair]) + radius(b_packed[pair]);
        nk_f64_t bound = 2 * (points + 16) * std::ldexp(1.0, -53) * spread;
        if constexpr (is_f16) {
            nk_f64_t const squares = points * nk_accumulation_error_bound(nk_f32_k) * spread * spread;
            bound = reference > 0 ? std::min(std::sqrt(squares), squares / reference) : std::sqrt(squares);
        }
        stats.accumulate_bounded(distance_t::from_raw(results.raw_values_data()[pair]), f64_t(reference), bound);
    }
    raw_distance_t const unequal = results.raw_values_data()[equal_pairs];
    stats.expect(unequal != unequal, "a pair of unequal sizes reported a distance");
    return stats;
}

#endif // NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_

#if NUMKONG_ARCH_ROCM_

void test_shapes_rocm(error_stats_section_t &check) {
    check.section("Shapes ROCm", nk_cap_rocm_k);
    check("rmsds_pair_f64_simt", test_rmsds_pairs<f64_t>);
    check("rmsds_pair_f32_simt", test_rmsds_pairs<f32_t>);
    check("rmsds_pair_f16_simt", test_rmsds_pairs<f16_t>);
    check("rmsds_packed_f16_rocm", test_rmsds_packed_f16<rocm_backend_t>, shapes_best<nk_rmsds_packed_f16_best>);
    check("hausdorff_warp_packed_f32_rocm", test_shapes_device<f32_t, nk_kernel_hausdorffs_packed_k, true>);
    check("hausdorff_warp_packed_f16_rocm", test_shapes_device<f16_t, nk_kernel_hausdorffs_packed_k, true>);
    check("hausdorff_packed_f32_simt", test_shapes_device<f32_t, nk_kernel_hausdorffs_packed_k, false>);
    check("hausdorff_packed_f16_simt", test_shapes_device<f16_t, nk_kernel_hausdorffs_packed_k, false>);
    check("hausdorffs_packed_f16_rocm", test_shapes_flips_f16<rocm_backend_t, nk_kernel_hausdorffs_packed_k>,
          shapes_best<nk_hausdorffs_packed_f16_best>);
    check("chamfer_warp_packed_f32_rocm", test_shapes_device<f32_t, nk_kernel_chamfers_packed_k, true>);
    check("chamfer_warp_packed_f16_rocm", test_shapes_device<f16_t, nk_kernel_chamfers_packed_k, true>);
    check("chamfer_packed_f32_simt", test_shapes_device<f32_t, nk_kernel_chamfers_packed_k, false>);
    check("chamfer_packed_f16_simt", test_shapes_device<f16_t, nk_kernel_chamfers_packed_k, false>);
    check("chamfers_packed_f16_rocm", test_shapes_flips_f16<rocm_backend_t, nk_kernel_chamfers_packed_k>,
          shapes_best<nk_chamfers_packed_f16_best>);
}

#elif NUMKONG_ARCH_CUDA_

void test_shapes_cuda(error_stats_section_t &check) {
    check.section("Shapes CUDA", nk_cap_cuda_k);
    check("rmsds_pair_f64_simt", test_rmsds_pairs<f64_t>);
    check("rmsds_pair_f32_simt", test_rmsds_pairs<f32_t>);
    check("rmsds_pair_f16_simt", test_rmsds_pairs<f16_t>);
    check("rmsds_packed_f16_cuda", test_rmsds_packed_f16<cuda_backend_t>, shapes_best<nk_rmsds_packed_f16_best>);
    check("hausdorff_warp_packed_f32_cuda", test_shapes_device<f32_t, nk_kernel_hausdorffs_packed_k, true>);
    check("hausdorff_warp_packed_f16_cuda", test_shapes_device<f16_t, nk_kernel_hausdorffs_packed_k, true>);
    check("hausdorff_packed_f32_simt", test_shapes_device<f32_t, nk_kernel_hausdorffs_packed_k, false>);
    check("hausdorff_packed_f16_simt", test_shapes_device<f16_t, nk_kernel_hausdorffs_packed_k, false>);
    check("hausdorffs_packed_f16_cuda", test_shapes_flips_f16<cuda_backend_t, nk_kernel_hausdorffs_packed_k>,
          shapes_best<nk_hausdorffs_packed_f16_best>);
    check("chamfer_warp_packed_f32_cuda", test_shapes_device<f32_t, nk_kernel_chamfers_packed_k, true>);
    check("chamfer_warp_packed_f16_cuda", test_shapes_device<f16_t, nk_kernel_chamfers_packed_k, true>);
    check("chamfer_packed_f32_simt", test_shapes_device<f32_t, nk_kernel_chamfers_packed_k, false>);
    check("chamfer_packed_f16_simt", test_shapes_device<f16_t, nk_kernel_chamfers_packed_k, false>);
    check("chamfers_packed_f16_cuda", test_shapes_flips_f16<cuda_backend_t, nk_kernel_chamfers_packed_k>,
          shapes_best<nk_chamfers_packed_f16_best>);
}

#else  // !NUMKONG_ARCH_ROCM_ && !NUMKONG_ARCH_CUDA_
void test_shapes_cuda(error_stats_section_t &) {}
#endif // NUMKONG_ARCH_ROCM_

} // namespace ashvardanian::numkong::test
