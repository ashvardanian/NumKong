/**
 *  @file bench/maxsim.cpp
 *  @author Ash Vardanian
 *  @date February 28, 2026
 *  @brief MaxSim, ColBERT late-interaction, benchmarks.
 */
#include "numkong/maxsim.h"

#include "harness.hpp"

namespace nk = ashvardanian::numkong;

namespace ashvardanian::numkong::bench {

template <nk_dtype_t input_dtype_>
void measure_maxsim_packed(                                                            //
    loop_t &loop, environment_t const &env,                                            //
    typename nk::type_for<input_dtype_>::type::dots_pack_size_kernel_t packed_size_fn, //
    typename nk::type_for<input_dtype_>::type::maxsim_pack_kernel_t pack_fn,           //
    typename nk::type_for<input_dtype_>::type::maxsim_packed_kernel_t maxsim_fn,       //
    std::size_t query_count, std::size_t document_count, std::size_t depth) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using raw_input_t = typename input_t::raw_t;
    using result_t = typename input_t::maxsim_result_t;

    nk_size_t stride = depth * sizeof(raw_input_t);
    nk_size_t query_packed_bytes = 0, document_packed_bytes = 0;
    if (!succeeded(loop, packed_size_fn(query_count, depth, &query_packed_bytes)) ||
        !succeeded(loop, packed_size_fn(document_count, depth, &document_packed_bytes)))
        return;

    bytes_t const per_set {query_count * stride + document_count * stride + query_packed_bytes + document_packed_bytes};
    std::size_t const sets_count = input_sets_count(per_set);

    struct maxsim_set_t {
        nk::vector<input_t> queries, documents;
        std::vector<char> query_packed, document_packed;
    };
    std::vector<maxsim_set_t> sets(sets_count);
    std::mt19937 generator(env.settings.seed.value);
    for (auto &s : sets) {
        s.queries = make_vector<input_t>(query_count * depth);
        s.documents = make_vector<input_t>(document_count * depth);
        s.query_packed.resize(query_packed_bytes);
        s.document_packed.resize(document_packed_bytes);
        nk::fill_uniform(generator, s.queries.values_data(), s.queries.size_values());
        nk::fill_uniform(generator, s.documents.values_data(), s.documents.size_values());
        if (!succeeded(loop, pack_fn(s.queries.raw_values_data(), query_count, depth, stride, s.query_packed.data(),
                                     nullptr)) ||
            !succeeded(loop, pack_fn(s.documents.raw_values_data(), document_count, depth, stride,
                                     s.document_packed.data(), nullptr)))
            return;
    }

    for (std::size_t call : loop) {
        auto &s = sets[call & (sets_count - 1)];
        result_t result;
        if (!succeeded(loop, maxsim_fn(s.query_packed.data(), s.document_packed.data(), query_count, document_count,
                                       depth, &result.raw_, nullptr)))
            break;
        do_not_optimize(result);
    }

    loop.rate("scalar-ops", 2.0 * query_count * document_count * depth);
}

template <nk_dtype_t input_dtype_>
void run_maxsim_packed(environment_t const &env, std::string name, //
                       typename nk::type_for<input_dtype_>::type::dots_pack_size_kernel_t packed_size_fn,
                       typename nk::type_for<input_dtype_>::type::maxsim_pack_kernel_t pack_fn,
                       typename nk::type_for<input_dtype_>::type::maxsim_packed_kernel_t maxsim_fn) {
    std::size_t query_count = env.settings.matrix_height;
    std::size_t document_count = env.settings.matrix_width;
    std::size_t depth = env.settings.matrix_depth;
    std::string bench_name = name + "<" + std::to_string(query_count) + "x" + std::to_string(document_count) + "x" +
                             std::to_string(depth) + ">";
    run_benchmark(env, bench_name, measure_maxsim_packed<input_dtype_>, packed_size_fn, pack_fn, maxsim_fn, query_count,
                  document_count, depth);
}

void bench_maxsim(environment_t const &env) {
    constexpr nk_dtype_t bf16_k = nk_bf16_k, f32_k = nk_f32_k, f16_k = nk_f16_k;

    section(env, "MaxSim Serial", nk_cap_serial_k);
    run_maxsim_packed<bf16_k>(env, "maxsim_bf16_serial", nk_maxsim_pack_size_bf16_serial, nk_maxsim_pack_bf16_serial,
                              nk_maxsim_packed_bf16_serial);
    run_maxsim_packed<f32_k>(env, "maxsim_f32_serial", nk_maxsim_pack_size_f32_serial, nk_maxsim_pack_f32_serial,
                             nk_maxsim_packed_f32_serial);
    run_maxsim_packed<f16_k>(env, "maxsim_f16_serial", nk_maxsim_pack_size_f16_serial, nk_maxsim_pack_f16_serial,
                             nk_maxsim_packed_f16_serial);

#if NUMKONG_TARGET_HASWELL
    if (section(env, "MaxSim Haswell", nk_cap_haswell_k)) {
        run_maxsim_packed<bf16_k>(env, "maxsim_bf16_haswell", nk_maxsim_pack_size_bf16_haswell,
                                  nk_maxsim_pack_bf16_haswell, nk_maxsim_packed_bf16_haswell);
        run_maxsim_packed<f32_k>(env, "maxsim_f32_haswell", nk_maxsim_pack_size_f32_haswell, nk_maxsim_pack_f32_haswell,
                                 nk_maxsim_packed_f32_haswell);
        run_maxsim_packed<f16_k>(env, "maxsim_f16_haswell", nk_maxsim_pack_size_f16_haswell, nk_maxsim_pack_f16_haswell,
                                 nk_maxsim_packed_f16_haswell);
    }
#endif

#if NUMKONG_TARGET_ALDER
    if (section(env, "MaxSim Alder", nk_cap_alder_k)) {
        run_maxsim_packed<bf16_k>(env, "maxsim_bf16_alder", nk_maxsim_pack_size_bf16_alder, nk_maxsim_pack_bf16_alder,
                                  nk_maxsim_packed_bf16_alder);
        run_maxsim_packed<f32_k>(env, "maxsim_f32_alder", nk_maxsim_pack_size_f32_alder, nk_maxsim_pack_f32_alder,
                                 nk_maxsim_packed_f32_alder);
        run_maxsim_packed<f16_k>(env, "maxsim_f16_alder", nk_maxsim_pack_size_f16_alder, nk_maxsim_pack_f16_alder,
                                 nk_maxsim_packed_f16_alder);
    }
#endif

#if NUMKONG_TARGET_ICELAKE
    if (section(env, "MaxSim Ice Lake", nk_cap_icelake_k)) {
        run_maxsim_packed<f32_k>(env, "maxsim_f32_icelake", nk_maxsim_pack_size_f32_icelake, nk_maxsim_pack_f32_icelake,
                                 nk_maxsim_packed_f32_icelake);
        run_maxsim_packed<f16_k>(env, "maxsim_f16_icelake", nk_maxsim_pack_size_f16_icelake, nk_maxsim_pack_f16_icelake,
                                 nk_maxsim_packed_f16_icelake);
    }
#endif

#if NUMKONG_TARGET_GENOA
    if (section(env, "MaxSim Genoa", nk_cap_genoa_k)) {
        run_maxsim_packed<bf16_k>(env, "maxsim_bf16_genoa", nk_maxsim_pack_size_bf16_genoa, nk_maxsim_pack_bf16_genoa,
                                  nk_maxsim_packed_bf16_genoa);
    }
#endif

#if NUMKONG_TARGET_SAPPHIREAMX
    if (section(env, "MaxSim Sapphire AMX", nk_cap_sapphireamx_k)) {
        run_maxsim_packed<bf16_k>(env, "maxsim_bf16_sapphireamx", nk_maxsim_pack_size_bf16_sapphireamx,
                                  nk_maxsim_pack_bf16_sapphireamx, nk_maxsim_packed_bf16_sapphireamx);
        run_maxsim_packed<f32_k>(env, "maxsim_f32_sapphireamx", nk_maxsim_pack_size_f32_sapphireamx,
                                 nk_maxsim_pack_f32_sapphireamx, nk_maxsim_packed_f32_sapphireamx);
        run_maxsim_packed<f16_k>(env, "maxsim_f16_sapphireamx", nk_maxsim_pack_size_f16_sapphireamx,
                                 nk_maxsim_pack_f16_sapphireamx, nk_maxsim_packed_f16_sapphireamx);
    }
#endif

#if NUMKONG_TARGET_NEONSDOT
    if (section(env, "MaxSim NEON I8", nk_cap_neonsdot_k)) {
        run_maxsim_packed<bf16_k>(env, "maxsim_bf16_neonsdot", nk_maxsim_pack_size_bf16_neonsdot,
                                  nk_maxsim_pack_bf16_neonsdot, nk_maxsim_packed_bf16_neonsdot);
        run_maxsim_packed<f32_k>(env, "maxsim_f32_neonsdot", nk_maxsim_pack_size_f32_neonsdot,
                                 nk_maxsim_pack_f32_neonsdot, nk_maxsim_packed_f32_neonsdot);
        run_maxsim_packed<f16_k>(env, "maxsim_f16_neonsdot", nk_maxsim_pack_size_f16_neonsdot,
                                 nk_maxsim_pack_f16_neonsdot, nk_maxsim_packed_f16_neonsdot);
    }
#endif

#if NUMKONG_TARGET_V128RELAXED
    if (section(env, "MaxSim V128 Relaxed", nk_cap_v128relaxed_k)) {
        run_maxsim_packed<bf16_k>(env, "maxsim_bf16_v128relaxed", nk_maxsim_pack_size_bf16_v128relaxed,
                                  nk_maxsim_pack_bf16_v128relaxed, nk_maxsim_packed_bf16_v128relaxed);
        run_maxsim_packed<f32_k>(env, "maxsim_f32_v128relaxed", nk_maxsim_pack_size_f32_v128relaxed,
                                 nk_maxsim_pack_f32_v128relaxed, nk_maxsim_packed_f32_v128relaxed);
        run_maxsim_packed<f16_k>(env, "maxsim_f16_v128relaxed", nk_maxsim_pack_size_f16_v128relaxed,
                                 nk_maxsim_pack_f16_v128relaxed, nk_maxsim_packed_f16_v128relaxed);
    }
#endif

#if NUMKONG_TARGET_SME
    if (section(env, "MaxSim SME", nk_cap_sme_k)) {
        run_maxsim_packed<bf16_k>(env, "maxsim_bf16_sme", nk_maxsim_pack_size_bf16_sme, nk_maxsim_pack_bf16_sme,
                                  nk_maxsim_packed_bf16_sme);
        run_maxsim_packed<f32_k>(env, "maxsim_f32_sme", nk_maxsim_pack_size_f32_sme, nk_maxsim_pack_f32_sme,
                                 nk_maxsim_packed_f32_sme);
        run_maxsim_packed<f16_k>(env, "maxsim_f16_sme", nk_maxsim_pack_size_f16_sme, nk_maxsim_pack_f16_sme,
                                 nk_maxsim_packed_f16_sme);
    }
#endif
}

} // namespace ashvardanian::numkong::bench
