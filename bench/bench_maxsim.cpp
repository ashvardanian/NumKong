/**
 *  @brief MaxSim (ColBERT late-interaction) benchmarks.
 *  @file bench/bench_maxsim.cpp
 *  @author Ash Vardanian
 *  @date February 28, 2026
 */
#include "numkong/maxsim.h"

#include "bench.hpp"

namespace nk = ashvardanian::numkong;
namespace bm = benchmark;

template <nk_dtype_t input_dtype_>
void measure_maxsim_packed(                                                            //
    bm::State &state,                                                                  //
    typename nk::type_for<input_dtype_>::type::dots_pack_size_kernel_t packed_size_fn, //
    typename nk::type_for<input_dtype_>::type::maxsim_pack_kernel_t pack_fn,           //
    typename nk::type_for<input_dtype_>::type::maxsim_packed_kernel_t maxsim_fn,       //
    std::size_t query_count, std::size_t document_count, std::size_t depth) {

    using input_t = typename nk::type_for<input_dtype_>::type;
    using raw_input_t = typename input_t::raw_t;
    using result_t = typename input_t::maxsim_result_t;

    nk_size_t stride = depth * sizeof(raw_input_t);
    nk_size_t query_packed_bytes = packed_size_fn(query_count, depth);
    nk_size_t document_packed_bytes = packed_size_fn(document_count, depth);

    std::size_t bytes_per_set = query_count * stride + document_count * stride + query_packed_bytes +
                                document_packed_bytes;
    std::size_t const sets_count = bench_input_count(bytes_per_set);

    struct maxsim_set_t {
        nk::vector<input_t> queries, documents;
        std::vector<char> query_packed, document_packed;
    };
    std::vector<maxsim_set_t> sets(sets_count);
    auto generator = make_random_engine();
    for (auto &s : sets) {
        s.queries = make_vector<input_t>(query_count * depth);
        s.documents = make_vector<input_t>(document_count * depth);
        s.query_packed.resize(query_packed_bytes);
        s.document_packed.resize(document_packed_bytes);
        nk::fill_uniform(generator, s.queries.values_data(), s.queries.size_values());
        nk::fill_uniform(generator, s.documents.values_data(), s.documents.size_values());
        pack_fn(s.queries.raw_values_data(), query_count, depth, stride, s.query_packed.data());
        pack_fn(s.documents.raw_values_data(), document_count, depth, stride, s.document_packed.data());
    }

    std::size_t iterations = 0;
    for (auto _ : state) {
        auto &s = sets[iterations & (sets_count - 1)];
        result_t result;
        maxsim_fn(s.query_packed.data(), s.document_packed.data(), query_count, document_count, depth, &result.raw_);
        bm::DoNotOptimize(result);
        ++iterations;
    }

    state.counters["scalar-ops"] = bm::Counter(2.0 * iterations * query_count * document_count * depth,
                                               bm::Counter::kIsRate);
}

template <nk_dtype_t input_dtype_>
void run_maxsim_packed(std::string name, //
                       typename nk::type_for<input_dtype_>::type::dots_pack_size_kernel_t packed_size_fn,
                       typename nk::type_for<input_dtype_>::type::maxsim_pack_kernel_t pack_fn,
                       typename nk::type_for<input_dtype_>::type::maxsim_packed_kernel_t maxsim_fn) {
    std::size_t query_count = bench_config.matrix_height;
    std::size_t document_count = bench_config.matrix_width;
    std::size_t depth = bench_config.matrix_depth;
    std::string bench_name = name + "<" + std::to_string(query_count) + "x" + std::to_string(document_count) + "x" +
                             std::to_string(depth) + ">";
    bm::RegisterBenchmark(bench_name.c_str(), measure_maxsim_packed<input_dtype_>, packed_size_fn, pack_fn, maxsim_fn,
                          query_count, document_count, depth);
}

void bench_maxsim() {
    constexpr nk_dtype_t bf16_k = nk_bf16_k, f32_k = nk_f32_k, f16_k = nk_f16_k;

    // Serial maxsim benchmarks
    run_maxsim_packed<bf16_k>("maxsim_bf16_serial", nk_maxsim_pack_size_bf16_serial, nk_maxsim_pack_bf16_serial,
                              nk_maxsim_packed_bf16_serial);
    run_maxsim_packed<f32_k>("maxsim_f32_serial", nk_maxsim_pack_size_f32_serial, nk_maxsim_pack_f32_serial,
                             nk_maxsim_packed_f32_serial);
    run_maxsim_packed<f16_k>("maxsim_f16_serial", nk_maxsim_pack_size_f16_serial, nk_maxsim_pack_f16_serial,
                             nk_maxsim_packed_f16_serial);

#if NK_TARGET_HASWELL
    run_maxsim_packed<bf16_k>("maxsim_bf16_haswell", nk_maxsim_pack_size_bf16_haswell, nk_maxsim_pack_bf16_haswell,
                              nk_maxsim_packed_bf16_haswell);
    run_maxsim_packed<f32_k>("maxsim_f32_haswell", nk_maxsim_pack_size_f32_haswell, nk_maxsim_pack_f32_haswell,
                             nk_maxsim_packed_f32_haswell);
    run_maxsim_packed<f16_k>("maxsim_f16_haswell", nk_maxsim_pack_size_f16_haswell, nk_maxsim_pack_f16_haswell,
                             nk_maxsim_packed_f16_haswell);
#endif

#if NK_TARGET_ALDER
    run_maxsim_packed<bf16_k>("maxsim_bf16_alder", nk_maxsim_pack_size_bf16_alder, nk_maxsim_pack_bf16_alder,
                              nk_maxsim_packed_bf16_alder);
    run_maxsim_packed<f32_k>("maxsim_f32_alder", nk_maxsim_pack_size_f32_alder, nk_maxsim_pack_f32_alder,
                             nk_maxsim_packed_f32_alder);
    run_maxsim_packed<f16_k>("maxsim_f16_alder", nk_maxsim_pack_size_f16_alder, nk_maxsim_pack_f16_alder,
                             nk_maxsim_packed_f16_alder);
#endif

#if NK_TARGET_ICELAKE
    run_maxsim_packed<f32_k>("maxsim_f32_icelake", nk_maxsim_pack_size_f32_icelake, nk_maxsim_pack_f32_icelake,
                             nk_maxsim_packed_f32_icelake);
    run_maxsim_packed<f16_k>("maxsim_f16_icelake", nk_maxsim_pack_size_f16_icelake, nk_maxsim_pack_f16_icelake,
                             nk_maxsim_packed_f16_icelake);
#endif

#if NK_TARGET_GENOA
    run_maxsim_packed<bf16_k>("maxsim_bf16_genoa", nk_maxsim_pack_size_bf16_genoa, nk_maxsim_pack_bf16_genoa,
                              nk_maxsim_packed_bf16_genoa);
#endif

#if NK_TARGET_SAPPHIREAMX
    run_maxsim_packed<bf16_k>("maxsim_bf16_sapphireamx", nk_maxsim_pack_size_bf16_sapphireamx,
                              nk_maxsim_pack_bf16_sapphireamx, nk_maxsim_packed_bf16_sapphireamx);
    run_maxsim_packed<f32_k>("maxsim_f32_sapphireamx", nk_maxsim_pack_size_f32_sapphireamx,
                             nk_maxsim_pack_f32_sapphireamx, nk_maxsim_packed_f32_sapphireamx);
    run_maxsim_packed<f16_k>("maxsim_f16_sapphireamx", nk_maxsim_pack_size_f16_sapphireamx,
                             nk_maxsim_pack_f16_sapphireamx, nk_maxsim_packed_f16_sapphireamx);
#endif

#if NK_TARGET_NEONSDOT
    run_maxsim_packed<bf16_k>("maxsim_bf16_neonsdot", nk_maxsim_pack_size_bf16_neonsdot, nk_maxsim_pack_bf16_neonsdot,
                              nk_maxsim_packed_bf16_neonsdot);
    run_maxsim_packed<f32_k>("maxsim_f32_neonsdot", nk_maxsim_pack_size_f32_neonsdot, nk_maxsim_pack_f32_neonsdot,
                             nk_maxsim_packed_f32_neonsdot);
    run_maxsim_packed<f16_k>("maxsim_f16_neonsdot", nk_maxsim_pack_size_f16_neonsdot, nk_maxsim_pack_f16_neonsdot,
                             nk_maxsim_packed_f16_neonsdot);
#endif

#if NK_TARGET_V128RELAXED
    run_maxsim_packed<bf16_k>("maxsim_bf16_v128relaxed", nk_maxsim_pack_size_bf16_v128relaxed,
                              nk_maxsim_pack_bf16_v128relaxed, nk_maxsim_packed_bf16_v128relaxed);
    run_maxsim_packed<f32_k>("maxsim_f32_v128relaxed", nk_maxsim_pack_size_f32_v128relaxed,
                             nk_maxsim_pack_f32_v128relaxed, nk_maxsim_packed_f32_v128relaxed);
    run_maxsim_packed<f16_k>("maxsim_f16_v128relaxed", nk_maxsim_pack_size_f16_v128relaxed,
                             nk_maxsim_pack_f16_v128relaxed, nk_maxsim_packed_f16_v128relaxed);
#endif

#if NK_TARGET_SME
    run_maxsim_packed<bf16_k>("maxsim_bf16_sme", nk_maxsim_pack_size_bf16_sme, nk_maxsim_pack_bf16_sme,
                              nk_maxsim_packed_bf16_sme);
    run_maxsim_packed<f32_k>("maxsim_f32_sme", nk_maxsim_pack_size_f32_sme, nk_maxsim_pack_f32_sme,
                             nk_maxsim_packed_f32_sme);
    run_maxsim_packed<f16_k>("maxsim_f16_sme", nk_maxsim_pack_size_f16_sme, nk_maxsim_pack_f16_sme,
                             nk_maxsim_packed_f16_sme);
#endif
}
