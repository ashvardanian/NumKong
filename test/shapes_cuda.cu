/**
 *  @file test/shapes_cuda.cu
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief Chamfer and Hausdorff device bodies, per warp and per thread, against the serial kernels.
 *
 *  The bodies are device helpers, so this unit launches its own kernels over them and keeps the
 *  @c cuda exports of their header switched off, as the library already defines them.
 */
#undef NUMKONG_TARGET_CUDA
#define NUMKONG_TARGET_CUDA 0
#include "harness.hpp"

#if NUMKONG_ARCH_CUDA_
#include "numkong/shapes/cuda.cuh" // `nk_chamfer_warp_packed_f32_cuda_`, `nk_chamfer_packed_f32_simt_`
#endif

namespace ashvardanian::numkong::test {

#if NUMKONG_ARCH_CUDA_

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
    if constexpr (std::is_same_v<scalar_type_, f32_t> && warp_)
        results[thread] = chamfer_k ? nk_chamfer_warp_packed_f32_cuda_(a, b, limits[pair], lane)
                                    : nk_hausdorff_warp_packed_f32_cuda_(a, b, limits[pair], lane);
    else if constexpr (std::is_same_v<scalar_type_, f32_t>)
        results[thread] = chamfer_k ? nk_chamfer_packed_f32_simt_(a, b, limits[pair])
                                    : nk_hausdorff_packed_f32_simt_(a, b, limits[pair]);
    else if constexpr (warp_)
        results[thread] = chamfer_k ? nk_chamfer_warp_packed_f16_cuda_(a, b, limits[pair], lane)
                                    : nk_hausdorff_warp_packed_f16_cuda_(a, b, limits[pair], lane);
    else
        results[thread] = chamfer_k ? nk_chamfer_packed_f16_simt_(a, b, limits[pair])
                                    : nk_hausdorff_packed_f16_simt_(a, b, limits[pair]);
}

/** Clouds over every branch of the device bodies: no points, one, fewer and more than a warp of
 *  them, each random, coincident, collinear, coplanar or with every point doubled, plus a mirrored
 *  copy of every random cloud, which lies a rounding away from it under one flip. Coordinates
 *  follow the settings' distribution, clamped to a quarter of the range for finite packs. */
template <typename scalar_type_>
std::vector<nk::vector<scalar_type_>> device_shapes_clouds(settings_t const &settings, std::mt19937 &generator) {
    std::size_t const warp = 32, points = settings.shape_points;
    std::size_t const sizes[] = {0, 1, 2, 3, warp - 1, warp, warp + 1, points, 2 * points + 3};
    nk_f64_t const limit = static_cast<nk_f64_t>(scalar_type_::finite_max()) / 4;
    std::vector<nk::vector<scalar_type_>> clouds;
    for (std::size_t size : sizes)
        for (int shape = 0; shape != 5; ++shape) {
            auto cloud = make_vector<scalar_type_>(size * 3);
            fill_random(settings, generator, cloud);
            scalar_type_ *values = cloud.values_data();
            for (std::size_t index = 0; index != size * 3; ++index)
                values[index] = scalar_type_(std::clamp(static_cast<nk_f64_t>(values[index]), -limit, limit));
            for (std::size_t point = 0; point != size; ++point) {
                scalar_type_ *coordinates = values + point * 3;
                if (shape == 1) coordinates[0] = values[0], coordinates[1] = values[1], coordinates[2] = values[2];
                if (shape == 2)
                    coordinates[1] = scalar_type_(static_cast<nk_f64_t>(coordinates[0]) / 2),
                    coordinates[2] = scalar_type_(static_cast<nk_f64_t>(coordinates[0]) / -4);
                if (shape == 3) coordinates[2] = scalar_type_(0.0);
                if (shape == 4 && point % 2)
                    coordinates[0] = coordinates[-3], coordinates[1] = coordinates[-2],
                    coordinates[2] = coordinates[-1];
            }
            if (shape == 0) {
                auto mirrored = make_vector<scalar_type_>(size * 3);
                for (std::size_t point = 0; point != size; ++point) {
                    scalar_type_ const *coordinates = values + point * 3;
                    scalar_type_ *image = mirrored.values_data() + point * 3;
                    image[0] = coordinates[1], image[1] = coordinates[0], image[2] = coordinates[2];
                }
                clouds.push_back(std::move(mirrored));
            }
            clouds.push_back(std::move(cloud));
        }
    return clouds;
}

/**
 *  @brief Every pair of the clouds above on the device against the serial kernel.
 *
 *  Unbounded calls must match within the CPU capabilities' bounds: n + m + 8 roundings of a Chamfer
 *  distance, whose sums the warp reduces in another order, and 8 of a Hausdorff distance, which
 *  differs only by the hardware square root. Bounded calls draw each pair's limit up to twice its
 *  distance, and a result past the limit must still be a lower bound of the distance. Every lane of
 *  a warp must return the same value.
 */
template <typename scalar_type_, nk_kernel_kind_t kind_, bool warp_>
error_stats_t test_shapes_device(settings_t const &settings) {
    using distance_t = typename scalar_type_::shapes_distance_t;
    using raw_distance_t = typename distance_t::raw_t;
    constexpr bool is_f32 = std::is_same_v<scalar_type_, f32_t>;
    error_stats_t stats(comparison_family_t::bounded_k);
    auto backend = make_backend<cuda_backend_t>(settings);
    std::mt19937 generator(settings.seed.value);
    auto const clouds = device_shapes_clouds<scalar_type_>(settings, generator);
    std::size_t const count = clouds.size(), pairs = count * count, lanes = warp_ ? 32 : 1;

    std::vector<std::size_t> offsets(count + 1);
    for (std::size_t cloud = 0; cloud != count; ++cloud) {
        nk_size_t bytes = 0;
        if constexpr (is_f32) stats.expect(nk_shapes_pack_size_f32_serial(clouds[cloud].size_values() / 3, &bytes));
        else stats.expect(nk_shapes_pack_size_f16_serial(clouds[cloud].size_values() / 3, &bytes));
        offsets[cloud + 1] = offsets[cloud] + nk_size_round_up_to_multiple_(bytes, nk_default_alignment_k);
    }
    auto packs = make_vector<char>(backend, offsets[count] + 2 * pairs * sizeof(void const *));
    void const **a_packed = reinterpret_cast<void const **>(packs.raw_values_data() + offsets[count]);
    void const **b_packed = a_packed + pairs;
    std::vector<void const *> pointers(count);
    for (std::size_t cloud = 0; cloud != count; ++cloud) {
        void *pack = packs.raw_values_data() + offsets[cloud];
        std::size_t const points = clouds[cloud].size_values() / 3;
        if constexpr (is_f32)
            stats.expect(nk_shapes_pack_f32_serial(clouds[cloud].raw_values_data(), points, pack, nullptr));
        else stats.expect(nk_shapes_pack_f16_serial(clouds[cloud].raw_values_data(), points, pack, nullptr));
        pointers[cloud] = pack;
    }
    for (std::size_t pair = 0; pair != pairs; ++pair)
        a_packed[pair] = pointers[pair / count], b_packed[pair] = pointers[pair % count];

    std::vector<raw_distance_t> references(pairs);
    auto const serial = [&](auto chamfers, auto hausdorffs) {
        return (kind_ == nk_kernel_chamfers_packed_k ? chamfers : hausdorffs)(
            pointers.data(), count, pointers.data(), count, references.data(), count * sizeof(raw_distance_t), nullptr,
            nullptr);
    };
    if constexpr (is_f32) stats.expect(serial(nk_chamfers_packed_f32_serial, nk_hausdorffs_packed_f32_serial));
    else stats.expect(serial(nk_chamfers_packed_f16_serial, nk_hausdorffs_packed_f16_serial));

    auto limits = make_vector<distance_t>(backend, pairs);
    auto results = make_vector<distance_t>(backend, pairs * lanes);
    std::uniform_real_distribution<nk_f64_t> fraction(0.0, 2.0);
    nk_f64_t const roundoff = half_ulp<distance_t>(1.0);
    for (int bounded = 0; bounded != 2; ++bounded) {
        for (std::size_t pair = 0; pair != pairs; ++pair)
            limits.raw_values_data()[pair] = bounded
                                                 ? static_cast<raw_distance_t>(references[pair] * fraction(generator))
                                                 : std::numeric_limits<raw_distance_t>::infinity();
        unsigned const threads = 128;
        unsigned const blocks = static_cast<unsigned>(nk_size_divide_round_up_(pairs * lanes, threads));
        shapes_pairs_kernel_<scalar_type_, kind_, warp_><<<blocks, threads, 0, (cudaStream_t)backend.stream>>>(
            a_packed, b_packed, pairs, limits.raw_values_data(), results.raw_values_data());
        bool const launched = cudaGetLastError() == cudaSuccess;
        stats.expect(launched, "the kernel failed to launch");
        if (!launched || !stats.expect(backend.synchronize())) return stats;
        for (std::size_t pair = 0; pair != pairs; ++pair) {
            raw_distance_t const *lane_results = results.raw_values_data() + pair * lanes;
            for (std::size_t lane = 1; lane != lanes; ++lane)
                stats.expect(std::memcmp(lane_results + lane, lane_results, sizeof(raw_distance_t)) == 0,
                             "the lanes of a warp disagree");
            std::size_t const a_points = clouds[pair / count].size_values() / 3;
            std::size_t const b_points = clouds[pair % count].size_values() / 3;
            nk_f64_t const reference = references[pair], result = lane_results[0];
            nk_f64_t const roundings = kind_ == nk_kernel_chamfers_packed_k ? a_points + b_points + 8 : 8;
            nk_f64_t const bound = roundings * roundoff * reference;
            if (result <= limits.raw_values_data()[pair] || !std::isfinite(reference))
                stats.accumulate_bounded(distance_t::from_raw(lane_results[0]), f64_t(reference), bound);
            else
                stats.expect(result <= reference + bound + half_ulp<distance_t>(reference),
                             "a result past its limit exceeds the distance");
        }
    }
    return stats;
}

void test_shapes_cuda(error_stats_section_t &check) {
    check.section("Shapes CUDA", nk_cap_cuda_k);
    check("chamfer_warp_packed_f32_cuda", test_shapes_device<f32_t, nk_kernel_chamfers_packed_k, true>);
    check("chamfer_warp_packed_f16_cuda", test_shapes_device<f16_t, nk_kernel_chamfers_packed_k, true>);
    check("hausdorff_warp_packed_f32_cuda", test_shapes_device<f32_t, nk_kernel_hausdorffs_packed_k, true>);
    check("hausdorff_warp_packed_f16_cuda", test_shapes_device<f16_t, nk_kernel_hausdorffs_packed_k, true>);
    check("chamfer_packed_f32_simt", test_shapes_device<f32_t, nk_kernel_chamfers_packed_k, false>);
    check("chamfer_packed_f16_simt", test_shapes_device<f16_t, nk_kernel_chamfers_packed_k, false>);
    check("hausdorff_packed_f32_simt", test_shapes_device<f32_t, nk_kernel_hausdorffs_packed_k, false>);
    check("hausdorff_packed_f16_simt", test_shapes_device<f16_t, nk_kernel_hausdorffs_packed_k, false>);
}

#else  // !NUMKONG_ARCH_CUDA_
void test_shapes_cuda(error_stats_section_t &) {}
#endif // NUMKONG_ARCH_CUDA_

} // namespace ashvardanian::numkong::test
