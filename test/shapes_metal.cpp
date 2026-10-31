/**
 *  @file test/shapes_metal.cpp
 *  @author Ash Vardanian
 *  @date October 24, 2026
 *  @brief Chamfer and Hausdorff simdgroup bodies against the serial kernels.
 *
 *  The bodies are device helpers, so this unit compiles its own kernel over them, after the source
 *  the library embeds, and keeps the @c metal exports of their header switched off, as the library
 *  already defines them.
 */
#undef NUMKONG_TARGET_METAL
#define NUMKONG_TARGET_METAL 0
#include "cross_device.hpp" // `device_shapes_packed`

#if NUMKONG_ARCH_METAL_
#include "numkong/shapes/metal.h" // `nk_shapes_source_metal_`, `nk_shapes_resolve_packs_metal_`
#endif

namespace ashvardanian::numkong::test {

#if NUMKONG_ARCH_METAL_

/** The library's shapes source and a kernel scoring each pair of the two pack tables on its own
 *  simdgroup, each lane keeping its own result. Kept for the program's life, as compiled pipelines
 *  remember their source by address. */
static char const *shapes_pairs_source_metal() {
    static std::string const source = std::string(nk_shapes_source_metal_) + R"(
kernel void shapes_pairs_kernel_(device ulong const *a_table [[buffer(0)]], device ulong const *b_table [[buffer(1)]],
                                 device float const *limits [[buffer(2)]], device float *results [[buffer(3)]],
                                 constant nk_shapes_arguments_metal_t &arguments [[buffer(4)]],
                                 constant uint &hausdorff [[buffer(5)]], uint thread_index [[thread_position_in_grid]]) {
    ulong const pair = thread_index / 32;
    if (pair >= arguments.rows) return;
    device uchar const *a = nk_shapes_pack_metal_(a_table[pair], arguments);
    device uchar const *b = nk_shapes_pack_metal_(b_table[pair], arguments);
    results[thread_index] = hausdorff ? nk_hausdorff_simdgroup_packed_f16_metal_(a, b, limits[pair], thread_index % 32)
                                      : nk_chamfer_simdgroup_packed_f16_metal_(a, b, limits[pair], thread_index % 32);
}
)";
    return source.c_str();
}

/** Encodes the pairs kernel over @p pairs pairs of the two tables. */
static nk_status_t shapes_pairs_metal(void const *const *a_packed, void const *const *b_packed, std::size_t pairs,
                                      nk_f32_t const *limits, nk_f32_t *results, nk_kernel_kind_t kind,
                                      nk_stream_t stream) {
    nk_metal_call_t call;
    nk_status_t status = nk_enter_metal_(stream, &call);
    if (status != nk_success_k) return status;
    nk_shapes_arguments_metal_t arguments {};
    arguments.rows = pairs;
    status = nk_shapes_resolve_packs_metal_(&call, a_packed, pairs, NUMKONG_SIZE_MAX, &arguments);
    if (status == nk_success_k)
        status = nk_shapes_resolve_packs_metal_(&call, b_packed, pairs, NUMKONG_SIZE_MAX, &arguments);
    if (status != nk_success_k) return nk_abort_metal_(&call, status);
    nk_size_t offsets[4] = {0, 0, 0, 0};
    void *const buffers[4] = {nk_resolve_metal_(&call, a_packed, pairs * sizeof(void const *), offsets),
                              nk_resolve_metal_(&call, b_packed, pairs * sizeof(void const *), offsets + 1),
                              nk_resolve_metal_(&call, limits, pairs * sizeof(nk_f32_t), offsets + 2),
                              nk_resolve_metal_(&call, results, pairs * 32 * sizeof(nk_f32_t), offsets + 3)};
    for (void *buffer : buffers)
        if (!buffer) return nk_abort_metal_(&call, nk_device_memory_mismatch_k);
    void *const pipeline = nk_pipeline_metal_(call.context, shapes_pairs_source_metal(), "shapes_pairs_kernel_",
                                              NUMKONG_METAL_LANGUAGE_3_1_);
    if (!pipeline) return nk_abort_metal_(&call, nk_device_code_mismatch_k);
    nk_encoder_metal_(&call, pipeline);
    for (std::size_t slot = 0; slot != 4; ++slot) nk_bind_metal_(call.encoder, buffers[slot], offsets[slot], slot);
    nk_bind_bytes_metal_(call.encoder, &arguments, sizeof(arguments), 4);
    nk_u32_t const hausdorff = kind == nk_kernel_hausdorffs_packed_k;
    nk_bind_bytes_metal_(call.encoder, &hausdorff, sizeof(hausdorff), 5);
    nk_metal_size_t const grid = {nk_size_divide_round_up_(pairs * 32, nk_shapes_threads_metal_k), 1, 1};
    nk_metal_size_t const threads = {nk_shapes_threads_metal_k, 1, 1};
    return nk_dispatch_metal_(&call, grid, threads);
}

/**
 *  @brief Every pair of the device clouds on a simdgroup against the serial kernel.
 *
 *  Unbounded calls must match within the CPU capabilities' bounds: n + m + 8 roundings of a Chamfer
 *  distance, whose sums the simdgroup reduces in another order, and 8 of a Hausdorff distance.
 *  Bounded calls draw each pair's limit up to twice its distance, and a result past the limit must
 *  still be a lower bound of the distance, up to the 3 roundings of each pack radius the Hausdorff
 *  header bound subtracts. Every lane of a simdgroup must return the same value.
 */
template <nk_kernel_kind_t kind_>
error_stats_t test_shapes_simdgroup_metal(settings_t const &settings) {
    error_stats_t stats(comparison_family_t::bounded_k);
    auto backend = make_backend<metal_backend_t>(settings);
    std::mt19937 generator(settings.seed.value);
    auto const packs = device_shapes_packed<f16_t>(settings, backend, generator, stats);
    std::size_t const count = packs.size(), pairs = count * count, lanes = 32;

    auto pair_packs = make_vector<void const *>(backend, 2 * pairs);
    void const **a_packed = pair_packs.values_data();
    void const **b_packed = a_packed + pairs;
    for (std::size_t pair = 0; pair != pairs; ++pair)
        a_packed[pair] = packs[pair / count], b_packed[pair] = packs[pair % count];

    std::vector<nk_f32_t> references(pairs);
    auto const serial = kind_ == nk_kernel_chamfers_packed_k ? nk_chamfers_packed_f16_serial
                                                             : nk_hausdorffs_packed_f16_serial;
    stats.expect(serial(packs.data(), count, packs.data(), count, references.data(), count * sizeof(nk_f32_t), nullptr,
                        nullptr));

    auto limits = make_vector<f32_t>(backend, pairs);
    auto results = make_vector<f32_t>(backend, pairs * lanes);
    std::uniform_real_distribution<nk_f64_t> fraction(0.0, 2.0);
    nk_f64_t const roundoff = half_ulp<f32_t>(1.0);
    auto const radius = [](void const *packed) {
        return static_cast<nk_f64_t>(((nk_shapes_packed_header_f32_t const *)packed)->max_radius);
    };
    for (int bounded = 0; bounded != 2; ++bounded) {
        for (std::size_t pair = 0; pair != pairs; ++pair)
            limits.raw_values_data()[pair] = bounded ? static_cast<nk_f32_t>(references[pair] * fraction(generator))
                                                     : std::numeric_limits<nk_f32_t>::infinity();
        backend.call(shapes_pairs_metal, a_packed, b_packed, pairs, limits.raw_values_data(), results.raw_values_data(),
                     kind_);
        if (!stats.expect(backend.synchronize())) return stats;
        for (std::size_t pair = 0; pair != pairs; ++pair) {
            nk_f32_t const *lane_results = results.raw_values_data() + pair * lanes;
            for (std::size_t lane = 1; lane != lanes; ++lane)
                stats.expect(std::memcmp(lane_results + lane, lane_results, sizeof(nk_f32_t)) == 0,
                             "the lanes of a simdgroup disagree");
            std::size_t const a_points = packs.points(pair / count);
            std::size_t const b_points = packs.points(pair % count);
            nk_f64_t const reference = references[pair], result = lane_results[0];
            nk_f64_t const roundings = kind_ == nk_kernel_chamfers_packed_k ? a_points + b_points + 8 : 8;
            nk_f64_t const bound = roundings * roundoff * reference;
            if (result <= limits.raw_values_data()[pair] || !std::isfinite(reference))
                stats.accumulate_bounded(f32_t::from_raw(lane_results[0]), f64_t(reference), bound);
            else
                stats.expect(result <= reference + bound + half_ulp<f32_t>(reference) +
                                           3 * roundoff * (radius(a_packed[pair]) + radius(b_packed[pair])),
                             "a result past its limit exceeds the distance");
        }
    }
    return stats;
}

void test_shapes_metal(error_stats_section_t &check) {
    check.section("Shapes Metal", nk_cap_metal_k);
    check("chamfer_simdgroup_packed_f16_metal", test_shapes_simdgroup_metal<nk_kernel_chamfers_packed_k>);
    check("hausdorff_simdgroup_packed_f16_metal", test_shapes_simdgroup_metal<nk_kernel_hausdorffs_packed_k>);
}

#else  // !NUMKONG_ARCH_METAL_
void test_shapes_metal(error_stats_section_t &) {}
#endif // NUMKONG_ARCH_METAL_

} // namespace ashvardanian::numkong::test
