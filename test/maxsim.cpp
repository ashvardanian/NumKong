/**
 *  @file test/maxsim.cpp
 *  @author Ash Vardanian
 *  @date February 28, 2026
 *  @brief MaxSim precision tests, ColBERT late-interaction.
 */

#include "harness.hpp"

#include "numkong/maxsim.h"
#include "numkong/maxsim.hpp"

namespace ashvardanian::numkong::test {

template <typename scalar_type_>
error_stats_t test_maxsim_packed(settings_t const &settings,
                                 typename scalar_type_::dots_pack_size_kernel_t packed_size_fn,
                                 typename scalar_type_::maxsim_pack_kernel_t pack_fn,
                                 typename scalar_type_::maxsim_packed_kernel_t maxsim_fn) {
    using scalar_t = scalar_type_;
    using result_t = typename scalar_t::maxsim_result_t;
    using reference_t = bounded_reference_for<scalar_t, result_t>;

    error_stats_t stats(nk_maxsim_error_bound(scalar_t::dtype()));
    std::mt19937 generator(settings.seed.value);

    std::size_t query_count = settings.matrix_height;
    std::size_t document_count = settings.matrix_width;
    std::size_t depth = settings.matrix_depth;
    std::size_t stride = depth * sizeof(typename scalar_t::raw_t);

    auto queries = make_vector<scalar_t>(query_count * depth);
    auto documents = make_vector<scalar_t>(document_count * depth);

    nk_size_t query_pack_size = 0, document_pack_size = 0;
    stats.expect(packed_size_fn(query_count, depth, &query_pack_size));
    stats.expect(packed_size_fn(document_count, depth, &document_pack_size));
    auto query_packed = make_vector<char>(query_pack_size);
    auto document_packed = make_vector<char>(document_pack_size);

    for (time_point_t const deadline = steady_clock_t::now() + settings.time_limit_per_kernel;
         steady_clock_t::now() < deadline;) {
        fill_random(settings, generator, queries);
        fill_random(settings, generator, documents);

        // Pack and compute with kernel under test
        stats.expect(
            pack_fn(queries.raw_values_data(), query_count, depth, stride, query_packed.raw_values_data(), nullptr));
        stats.expect(pack_fn(documents.raw_values_data(), document_count, depth, stride,
                             document_packed.raw_values_data(), nullptr));
        result_t result;
        stats.expect(maxsim_fn(query_packed.raw_values_data(), document_packed.raw_values_data(), query_count,
                               document_count, depth, &result.raw_, nullptr));

        // Exhaustive scalar reference
        reference_t reference;
        stats.expect(nk::maxsim_reference<scalar_t, reference_t>(queries.raw_values_data(), query_count, stride,
                                                                 documents.raw_values_data(), document_count, stride,
                                                                 depth, &reference, no_tiers_k));

        stats.accumulate(result, reference);
    }
    return stats;
}

void test_maxsim(error_stats_section_t &check) {

    check.section("MaxSim Serial", nk_cap_serial_k);
    check("maxsim_packed_bf16_serial", test_maxsim_packed<bf16_t>, nk_maxsim_pack_size_bf16_serial,
          nk_maxsim_pack_bf16_serial, nk_maxsim_packed_bf16_serial);
    check("maxsim_packed_f32_serial", test_maxsim_packed<f32_t>, nk_maxsim_pack_size_f32_serial,
          nk_maxsim_pack_f32_serial, nk_maxsim_packed_f32_serial);
    check("maxsim_packed_f16_serial", test_maxsim_packed<f16_t>, nk_maxsim_pack_size_f16_serial,
          nk_maxsim_pack_f16_serial, nk_maxsim_packed_f16_serial);

#if !NUMKONG_HEADER_ONLY
    check.section("MaxSim Runtime Dispatch", nk_cap_serial_k);
    check("maxsim_packed_bf16", test_maxsim_packed<bf16_t>, cpu_best<nk_maxsim_pack_size_bf16_best>,
          cpu_best<nk_maxsim_pack_bf16_best>, cpu_best<nk_maxsim_packed_bf16_best>);
    check("maxsim_packed_f32", test_maxsim_packed<f32_t>, cpu_best<nk_maxsim_pack_size_f32_best>,
          cpu_best<nk_maxsim_pack_f32_best>, cpu_best<nk_maxsim_packed_f32_best>);
    check("maxsim_packed_f16", test_maxsim_packed<f16_t>, cpu_best<nk_maxsim_pack_size_f16_best>,
          cpu_best<nk_maxsim_pack_f16_best>, cpu_best<nk_maxsim_packed_f16_best>);
#endif

#if NUMKONG_TARGET_HASWELL
    check.section("MaxSim Haswell", nk_cap_haswell_k);
    check("maxsim_packed_bf16_haswell", test_maxsim_packed<bf16_t>, nk_maxsim_pack_size_bf16_haswell,
          nk_maxsim_pack_bf16_haswell, nk_maxsim_packed_bf16_haswell);
    check("maxsim_packed_f32_haswell", test_maxsim_packed<f32_t>, nk_maxsim_pack_size_f32_haswell,
          nk_maxsim_pack_f32_haswell, nk_maxsim_packed_f32_haswell);
    check("maxsim_packed_f16_haswell", test_maxsim_packed<f16_t>, nk_maxsim_pack_size_f16_haswell,
          nk_maxsim_pack_f16_haswell, nk_maxsim_packed_f16_haswell);
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_ALDER
    check.section("MaxSim Alder", nk_cap_alder_k);
    check("maxsim_packed_bf16_alder", test_maxsim_packed<bf16_t>, nk_maxsim_pack_size_bf16_alder,
          nk_maxsim_pack_bf16_alder, nk_maxsim_packed_bf16_alder);
    check("maxsim_packed_f32_alder", test_maxsim_packed<f32_t>, nk_maxsim_pack_size_f32_alder, nk_maxsim_pack_f32_alder,
          nk_maxsim_packed_f32_alder);
    check("maxsim_packed_f16_alder", test_maxsim_packed<f16_t>, nk_maxsim_pack_size_f16_alder, nk_maxsim_pack_f16_alder,
          nk_maxsim_packed_f16_alder);
#endif // NUMKONG_TARGET_ALDER

#if NUMKONG_TARGET_ICELAKE
    check.section("MaxSim Ice Lake", nk_cap_icelake_k);
    check("maxsim_packed_f32_icelake", test_maxsim_packed<f32_t>, nk_maxsim_pack_size_f32_icelake,
          nk_maxsim_pack_f32_icelake, nk_maxsim_packed_f32_icelake);
    check("maxsim_packed_f16_icelake", test_maxsim_packed<f16_t>, nk_maxsim_pack_size_f16_icelake,
          nk_maxsim_pack_f16_icelake, nk_maxsim_packed_f16_icelake);
#endif // NUMKONG_TARGET_ICELAKE

#if NUMKONG_TARGET_GENOA
    check.section("MaxSim Genoa", nk_cap_genoa_k);
    check("maxsim_packed_bf16_genoa", test_maxsim_packed<bf16_t>, nk_maxsim_pack_size_bf16_genoa,
          nk_maxsim_pack_bf16_genoa, nk_maxsim_packed_bf16_genoa);
#endif // NUMKONG_TARGET_GENOA

#if NUMKONG_TARGET_SAPPHIREAMX
    check.section("MaxSim Sapphire AMX", nk_cap_sapphireamx_k);
    check("maxsim_packed_bf16_sapphireamx", test_maxsim_packed<bf16_t>, nk_maxsim_pack_size_bf16_sapphireamx,
          nk_maxsim_pack_bf16_sapphireamx, nk_maxsim_packed_bf16_sapphireamx);
    check("maxsim_packed_f32_sapphireamx", test_maxsim_packed<f32_t>, nk_maxsim_pack_size_f32_sapphireamx,
          nk_maxsim_pack_f32_sapphireamx, nk_maxsim_packed_f32_sapphireamx);
    check("maxsim_packed_f16_sapphireamx", test_maxsim_packed<f16_t>, nk_maxsim_pack_size_f16_sapphireamx,
          nk_maxsim_pack_f16_sapphireamx, nk_maxsim_packed_f16_sapphireamx);
#endif // NUMKONG_TARGET_SAPPHIREAMX

#if NUMKONG_TARGET_NEONSDOT
    check.section("MaxSim NEON I8", nk_cap_neonsdot_k);
    check("maxsim_packed_bf16_neonsdot", test_maxsim_packed<bf16_t>, nk_maxsim_pack_size_bf16_neonsdot,
          nk_maxsim_pack_bf16_neonsdot, nk_maxsim_packed_bf16_neonsdot);
    check("maxsim_packed_f32_neonsdot", test_maxsim_packed<f32_t>, nk_maxsim_pack_size_f32_neonsdot,
          nk_maxsim_pack_f32_neonsdot, nk_maxsim_packed_f32_neonsdot);
    check("maxsim_packed_f16_neonsdot", test_maxsim_packed<f16_t>, nk_maxsim_pack_size_f16_neonsdot,
          nk_maxsim_pack_f16_neonsdot, nk_maxsim_packed_f16_neonsdot);
#endif // NUMKONG_TARGET_NEONSDOT

#if NUMKONG_TARGET_V128RELAXED
    check.section("MaxSim V128 Relaxed", nk_cap_v128relaxed_k);
    check("maxsim_packed_bf16_v128relaxed", test_maxsim_packed<bf16_t>, nk_maxsim_pack_size_bf16_v128relaxed,
          nk_maxsim_pack_bf16_v128relaxed, nk_maxsim_packed_bf16_v128relaxed);
    check("maxsim_packed_f32_v128relaxed", test_maxsim_packed<f32_t>, nk_maxsim_pack_size_f32_v128relaxed,
          nk_maxsim_pack_f32_v128relaxed, nk_maxsim_packed_f32_v128relaxed);
    check("maxsim_packed_f16_v128relaxed", test_maxsim_packed<f16_t>, nk_maxsim_pack_size_f16_v128relaxed,
          nk_maxsim_pack_f16_v128relaxed, nk_maxsim_packed_f16_v128relaxed);
#endif // NUMKONG_TARGET_V128RELAXED

#if NUMKONG_TARGET_SME
    check.section("MaxSim SME", nk_cap_sme_k);
    check("maxsim_packed_bf16_sme", test_maxsim_packed<bf16_t>, nk_maxsim_pack_size_bf16_sme, nk_maxsim_pack_bf16_sme,
          nk_maxsim_packed_bf16_sme);
    check("maxsim_packed_f32_sme", test_maxsim_packed<f32_t>, nk_maxsim_pack_size_f32_sme, nk_maxsim_pack_f32_sme,
          nk_maxsim_packed_f32_sme);
    check("maxsim_packed_f16_sme", test_maxsim_packed<f16_t>, nk_maxsim_pack_size_f16_sme, nk_maxsim_pack_f16_sme,
          nk_maxsim_packed_f16_sme);
#endif // NUMKONG_TARGET_SME
}

} // namespace ashvardanian::numkong::test
