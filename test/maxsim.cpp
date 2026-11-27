/**
 *  @file test/maxsim.cpp
 *  @author Ash Vardanian
 *  @date February 28, 2026
 *  @brief MaxSim precision tests, ColBERT late-interaction.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include "harness.hpp"

#include "numkong/maxsim.h"
#include "numkong/maxsim.hpp"

namespace ashvardanian::numkong::test {

/**
 *  @brief Exhaustive MaxSim, Σᵢ minⱼ angular(qᵢ, dⱼ), summed in @p result_type_.
 *  @param[in] queries Query vectors in row-major order.
 *  @param[in] query_count Number of query vectors.
 *  @param[in] query_stride Row stride in bytes for query vectors.
 *  @param[in] documents Document vectors in row-major order.
 *  @param[in] document_count Number of document vectors.
 *  @param[in] document_stride Row stride in bytes for document vectors.
 *  @param[in] depth Number of dimensions per vector.
 *  @return The sum of per-query minimum angular distances.
 */
template <nk::a_numeric_dtype in_type_, nk::a_numeric_dtype result_type_>
result_type_ maxsim_reference(in_type_ const *queries, std::size_t query_count, std::size_t query_stride,
                              in_type_ const *documents, std::size_t document_count, std::size_t document_stride,
                              std::size_t depth) noexcept {
    result_type_ const zero(0), one(1);
    result_type_ total_angular_distance {};
    for (std::size_t query_index = 0; query_index < query_count; query_index++) {
        in_type_ const *query_row = reinterpret_cast<in_type_ const *>(reinterpret_cast<char const *>(queries) +
                                                                       query_index * query_stride);
        result_type_ min_angular = result_type_::finite_max();
        for (std::size_t document_index = 0; document_index < document_count; document_index++) {
            in_type_ const *document_row = reinterpret_cast<in_type_ const *>(
                reinterpret_cast<char const *>(documents) + document_index * document_stride);
            result_type_ ab {}, aa {}, bb {};
            for (std::size_t i = 0; i < depth; i++) {
                ab = fma_reference(query_row[i], document_row[i], ab);
                aa = fma_reference(query_row[i], query_row[i], aa);
                bb = fma_reference(document_row[i], document_row[i], bb);
            }
            result_type_ angular_distance = one - ab / (square_root_reference(aa) * square_root_reference(bb));
            // As in `spatial.h`: two zero norms give 0, one zero norm or a zero dot 1, NaN stays
            if (aa == zero && bb == zero) angular_distance = zero;
            else if (ab == zero || aa == zero || bb == zero) angular_distance = one;
            if (angular_distance < zero) angular_distance = zero;
            if (angular_distance < min_angular) min_angular = angular_distance;
        }
        total_angular_distance = total_angular_distance + min_angular;
    }
    return total_angular_distance;
}

template <typename scalar_type_>
error_stats_t test_maxsim_packed(typename scalar_type_::dots_pack_size_kernel_t packed_size_fn,
                                 typename scalar_type_::maxsim_pack_kernel_t pack_fn,
                                 typename scalar_type_::maxsim_packed_kernel_t maxsim_fn) {
    using scalar_t = scalar_type_;
    using result_t = typename scalar_t::maxsim_result_t;
    using reference_t = bounded_reference_for<scalar_t, result_t>;

    error_stats_t stats(nk_maxsim_error_bound(scalar_t::dtype()));
    split_mix_generator_t generator(case_stream_key());

    std::size_t query_count = test_run.settings.matrix_height;
    std::size_t document_count = test_run.settings.matrix_width;
    std::size_t depth = test_run.settings.matrix_depth;
    std::size_t stride = depth * sizeof(typename scalar_t::raw_t);

    auto queries = make_vector<scalar_t>(query_count * depth);
    auto documents = make_vector<scalar_t>(document_count * depth);

    nk_size_t query_pack_size = 0, document_pack_size = 0;
    stats.expect_status(packed_size_fn(query_count, depth, &query_pack_size));
    stats.expect_status(packed_size_fn(document_count, depth, &document_pack_size));
    auto query_packed = make_vector<std::byte>(query_pack_size);
    auto document_packed = make_vector<std::byte>(document_pack_size);

    for (deadline_t const deadline = test_run.settings.deadline(); deadline.ahead();) {
        fill_random(generator, queries);
        fill_random(generator, documents);
        // A zero query beside a zero document scores 0, as two zero vectors do in `spatials.h`
        std::memset(queries.raw_values_data(), 0, stride), std::memset(documents.raw_values_data(), 0, stride);

        // Pack and compute with kernel under test
        stats.expect_status(
            pack_fn(queries.raw_values_data(), query_count, depth, stride, query_packed.raw_values_data(), nullptr));
        stats.expect_status(pack_fn(documents.raw_values_data(), document_count, depth, stride,
                                    document_packed.raw_values_data(), nullptr));
        result_t result;
        stats.expect_status(maxsim_fn(query_packed.raw_values_data(), document_packed.raw_values_data(), query_count,
                                      document_count, depth, &result.raw_, nullptr));

        reference_t const reference = maxsim_reference<scalar_t, reference_t>(
            queries.values_data(), query_count, stride, documents.values_data(), document_count, stride, depth);

        stats.accumulate(result, reference);
    }
    return stats;
}

void run_maxsim_cases() {
    auto const serial = nk_cap_serial_k;

    run_kernel_case("nk_maxsim_packed_bf16_serial", test_maxsim_packed<bf16_t>, serial, nk_maxsim_pack_size_bf16_serial,
                    nk_maxsim_pack_bf16_serial, nk_maxsim_packed_bf16_serial);
    run_kernel_case("nk_maxsim_packed_f32_serial", test_maxsim_packed<f32_t>, serial, nk_maxsim_pack_size_f32_serial,
                    nk_maxsim_pack_f32_serial, nk_maxsim_packed_f32_serial);
    run_kernel_case("nk_maxsim_packed_f16_serial", test_maxsim_packed<f16_t>, serial, nk_maxsim_pack_size_f16_serial,
                    nk_maxsim_pack_f16_serial, nk_maxsim_packed_f16_serial);

    run_kernel_case("nk_maxsim_packed_bf16_best", test_maxsim_packed<bf16_t>, serial,
                    cpu_best<nk_maxsim_pack_size_bf16_best>, cpu_best<nk_maxsim_pack_bf16_best>,
                    cpu_best<nk_maxsim_packed_bf16_best>);
    run_kernel_case("nk_maxsim_packed_f32_best", test_maxsim_packed<f32_t>, serial,
                    cpu_best<nk_maxsim_pack_size_f32_best>, cpu_best<nk_maxsim_pack_f32_best>,
                    cpu_best<nk_maxsim_packed_f32_best>);
    run_kernel_case("nk_maxsim_packed_f16_best", test_maxsim_packed<f16_t>, serial,
                    cpu_best<nk_maxsim_pack_size_f16_best>, cpu_best<nk_maxsim_pack_f16_best>,
                    cpu_best<nk_maxsim_packed_f16_best>);

#if NUMKONG_TARGET_HASWELL
    auto const haswell = nk_cap_haswell_k;
    run_kernel_case("nk_maxsim_packed_bf16_haswell", test_maxsim_packed<bf16_t>, haswell,
                    nk_maxsim_pack_size_bf16_haswell, nk_maxsim_pack_bf16_haswell, nk_maxsim_packed_bf16_haswell);
    run_kernel_case("nk_maxsim_packed_f32_haswell", test_maxsim_packed<f32_t>, haswell, nk_maxsim_pack_size_f32_haswell,
                    nk_maxsim_pack_f32_haswell, nk_maxsim_packed_f32_haswell);
    run_kernel_case("nk_maxsim_packed_f16_haswell", test_maxsim_packed<f16_t>, haswell, nk_maxsim_pack_size_f16_haswell,
                    nk_maxsim_pack_f16_haswell, nk_maxsim_packed_f16_haswell);
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_ALDER
    auto const alder = nk_cap_alder_k;
    run_kernel_case("nk_maxsim_packed_bf16_alder", test_maxsim_packed<bf16_t>, alder, nk_maxsim_pack_size_bf16_alder,
                    nk_maxsim_pack_bf16_alder, nk_maxsim_packed_bf16_alder);
    run_kernel_case("nk_maxsim_packed_f32_alder", test_maxsim_packed<f32_t>, alder, nk_maxsim_pack_size_f32_alder,
                    nk_maxsim_pack_f32_alder, nk_maxsim_packed_f32_alder);
    run_kernel_case("nk_maxsim_packed_f16_alder", test_maxsim_packed<f16_t>, alder, nk_maxsim_pack_size_f16_alder,
                    nk_maxsim_pack_f16_alder, nk_maxsim_packed_f16_alder);
#endif // NUMKONG_TARGET_ALDER

#if NUMKONG_TARGET_ICELAKE
    auto const icelake = nk_cap_icelake_k;
    run_kernel_case("nk_maxsim_packed_f32_icelake", test_maxsim_packed<f32_t>, icelake, nk_maxsim_pack_size_f32_icelake,
                    nk_maxsim_pack_f32_icelake, nk_maxsim_packed_f32_icelake);
    run_kernel_case("nk_maxsim_packed_f16_icelake", test_maxsim_packed<f16_t>, icelake, nk_maxsim_pack_size_f16_icelake,
                    nk_maxsim_pack_f16_icelake, nk_maxsim_packed_f16_icelake);
#endif // NUMKONG_TARGET_ICELAKE

#if NUMKONG_TARGET_GENOA
    auto const genoa = nk_cap_genoa_k;
    run_kernel_case("nk_maxsim_packed_bf16_genoa", test_maxsim_packed<bf16_t>, genoa, nk_maxsim_pack_size_bf16_genoa,
                    nk_maxsim_pack_bf16_genoa, nk_maxsim_packed_bf16_genoa);
#endif // NUMKONG_TARGET_GENOA

#if NUMKONG_TARGET_SAPPHIREAMX
    auto const sapphireamx = nk_cap_sapphireamx_k;
    run_kernel_case("nk_maxsim_packed_bf16_sapphireamx", test_maxsim_packed<bf16_t>, sapphireamx,
                    nk_maxsim_pack_size_bf16_sapphireamx, nk_maxsim_pack_bf16_sapphireamx,
                    nk_maxsim_packed_bf16_sapphireamx);
    run_kernel_case("nk_maxsim_packed_f32_sapphireamx", test_maxsim_packed<f32_t>, sapphireamx,
                    nk_maxsim_pack_size_f32_sapphireamx, nk_maxsim_pack_f32_sapphireamx,
                    nk_maxsim_packed_f32_sapphireamx);
    run_kernel_case("nk_maxsim_packed_f16_sapphireamx", test_maxsim_packed<f16_t>, sapphireamx,
                    nk_maxsim_pack_size_f16_sapphireamx, nk_maxsim_pack_f16_sapphireamx,
                    nk_maxsim_packed_f16_sapphireamx);
#endif // NUMKONG_TARGET_SAPPHIREAMX

#if NUMKONG_TARGET_NEONSDOT
    auto const neonsdot = nk_cap_neonsdot_k;
    run_kernel_case("nk_maxsim_packed_bf16_neonsdot", test_maxsim_packed<bf16_t>, neonsdot,
                    nk_maxsim_pack_size_bf16_neonsdot, nk_maxsim_pack_bf16_neonsdot, nk_maxsim_packed_bf16_neonsdot);
    run_kernel_case("nk_maxsim_packed_f32_neonsdot", test_maxsim_packed<f32_t>, neonsdot,
                    nk_maxsim_pack_size_f32_neonsdot, nk_maxsim_pack_f32_neonsdot, nk_maxsim_packed_f32_neonsdot);
    run_kernel_case("nk_maxsim_packed_f16_neonsdot", test_maxsim_packed<f16_t>, neonsdot,
                    nk_maxsim_pack_size_f16_neonsdot, nk_maxsim_pack_f16_neonsdot, nk_maxsim_packed_f16_neonsdot);
#endif // NUMKONG_TARGET_NEONSDOT

#if NUMKONG_TARGET_V128RELAXED
    auto const v128relaxed = nk_cap_v128relaxed_k;
    run_kernel_case("nk_maxsim_packed_bf16_v128relaxed", test_maxsim_packed<bf16_t>, v128relaxed,
                    nk_maxsim_pack_size_bf16_v128relaxed, nk_maxsim_pack_bf16_v128relaxed,
                    nk_maxsim_packed_bf16_v128relaxed);
    run_kernel_case("nk_maxsim_packed_f32_v128relaxed", test_maxsim_packed<f32_t>, v128relaxed,
                    nk_maxsim_pack_size_f32_v128relaxed, nk_maxsim_pack_f32_v128relaxed,
                    nk_maxsim_packed_f32_v128relaxed);
    run_kernel_case("nk_maxsim_packed_f16_v128relaxed", test_maxsim_packed<f16_t>, v128relaxed,
                    nk_maxsim_pack_size_f16_v128relaxed, nk_maxsim_pack_f16_v128relaxed,
                    nk_maxsim_packed_f16_v128relaxed);
#endif // NUMKONG_TARGET_V128RELAXED

#if NUMKONG_TARGET_SME
    auto const sme = nk_cap_sme_k;
    run_kernel_case("nk_maxsim_packed_bf16_sme", test_maxsim_packed<bf16_t>, sme, nk_maxsim_pack_size_bf16_sme,
                    nk_maxsim_pack_bf16_sme, nk_maxsim_packed_bf16_sme);
    run_kernel_case("nk_maxsim_packed_f32_sme", test_maxsim_packed<f32_t>, sme, nk_maxsim_pack_size_f32_sme,
                    nk_maxsim_pack_f32_sme, nk_maxsim_packed_f32_sme);
    run_kernel_case("nk_maxsim_packed_f16_sme", test_maxsim_packed<f16_t>, sme, nk_maxsim_pack_size_f16_sme,
                    nk_maxsim_pack_f16_sme, nk_maxsim_packed_f16_sme);
#endif // NUMKONG_TARGET_SME
}

} // namespace ashvardanian::numkong::test
