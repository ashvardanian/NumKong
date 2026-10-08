/**
 *  @file test/cross_cuda.cu
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Batch operation tests - CUDA capabilities.
 *
 *  Compiled with every capability some listed architecture runs: the CUDA baseline, Ampere, Hopper,
 *  Blackwell and Blackwell RTX.
 */
#include "cross_device.hpp"

namespace ashvardanian::numkong::test {

#if NUMKONG_ARCH_CUDA_

/** Packed angular distance against serial where the B column holds the E4M3 NaN code 0x7F, which
 *  must give NaN as serial does rather than decode to 480. */
template <auto packed_size_fn_, auto pack_fn_, auto angulars_fn_>
static error_stats_t test_angulars_packed_nan_e4m3(settings_t const &settings) {
    using bytes_t = nk::vector<char, cuda_backend_t::allocator<char>>;
    cuda_backend_t backend = make_backend<cuda_backend_t>(settings);
    error_stats_t stats(comparison_family_t::exact_k);
    std::size_t const depth = 64;
    auto a = bytes_t::zeros(depth, allocator_of<char>(backend)).value,
         b = bytes_t::zeros(depth, allocator_of<char>(backend)).value,
         distance = bytes_t::zeros(4, allocator_of<char>(backend)).value;
    auto packed = bytes_t::zeros(pack_size_bytes(stats, packed_size_fn_, 1, depth), allocator_of<char>(backend)).value;
    auto serial_packed = make_vector<char>(pack_size_bytes(stats, nk_dots_pack_size_e4m3_serial, 1, depth));
    std::memset(a.raw_values_data(), 0x38, depth), std::memset(b.raw_values_data(), 0x38, depth); // 1.0
    b[depth / 2] = 0x7F;
    auto const *a_codes = reinterpret_cast<nk_e4m3_t const *>(a.raw_values_data());
    auto const *b_codes = reinterpret_cast<nk_e4m3_t const *>(b.raw_values_data());
    auto *gpu = reinterpret_cast<nk_f32_t *>(distance.raw_values_data());
    nk_status_t submission_status = backend.call(pack_fn_, b_codes, 1, depth, depth, packed.raw_values_data(), 0, 1);
    if (submission_status == nk_success_k)
        submission_status = backend.call(angulars_fn_, a_codes, packed.raw_values_data(), gpu, 1, 1, depth, depth,
                                         sizeof(nk_f32_t));
    nk_status_t const completion_status = backend.synchronize();
    stats.expect(submission_status);
    stats.expect(completion_status);
    if (submission_status != nk_success_k || completion_status != nk_success_k) return stats;
    nk_f32_t serial = 0;
    stats.expect(nk_dots_pack_e4m3_serial(b_codes, 1, depth, depth, serial_packed.raw_values_data(), 0, 1, nullptr));
    stats.expect(nk_angulars_packed_e4m3_serial(a_codes, serial_packed.raw_values_data(), &serial, 1, 1, depth, depth,
                                                sizeof(nk_f32_t), nullptr));
    stats.expect(std::isnan(*gpu) && std::isnan(serial), "a NaN code in B gives no NaN angle");
    return stats;
}

void test_cross_cuda(error_stats_section_t &check) {
    check.section("Cross CUDA", nk_cap_cuda_k);
    check("dots_packed_f64_cuda", test_dots_packed<f64_t, cuda_backend_t>, nk_dots_pack_size_f64_cuda,
          nk_dots_pack_f64_cuda, nk_dots_packed_f64_cuda);
    check("dots_pack_f64_cuda", test_dots_pack_layout<f64_t, cuda_backend_t, nk_dots_pack_size_f64_cuda,
                                                      nk_dots_packed_shape_f64_cuda, nk_dots_pack_f64_cuda>);
    check("dots_contract_f64_cuda", test_dots_launch_contract<f64_t, cuda_backend_t, nk_dots_pack_size_f64_cuda,
                                                              nk_dots_packed_f64_cuda, nk_dots_symmetric_f64_cuda>);
    check("dots_symmetric_f64_cuda", test_dots_symmetric<f64_t, cuda_backend_t>, nk_dots_symmetric_f64_cuda);
    check("dots_packed_f32_cuda", test_dots_packed<f32_t, cuda_backend_t>, nk_dots_pack_size_f32_cuda,
          nk_dots_pack_f32_cuda, nk_dots_packed_f32_cuda);
    check("dots_pack_f32_cuda", test_dots_pack_layout<f32_t, cuda_backend_t, nk_dots_pack_size_f32_cuda,
                                                      nk_dots_packed_shape_f32_cuda, nk_dots_pack_f32_cuda>);
    check("dots_contract_f32_cuda", test_dots_launch_contract<f32_t, cuda_backend_t, nk_dots_pack_size_f32_cuda,
                                                              nk_dots_packed_f32_cuda, nk_dots_symmetric_f32_cuda>);
    check("dots_symmetric_f32_cuda", test_dots_symmetric<f32_t, cuda_backend_t>, nk_dots_symmetric_f32_cuda);
    check("dots_packed_bf16_cuda", test_dots_packed<bf16_t, cuda_backend_t>, nk_dots_pack_size_bf16_cuda,
          nk_dots_pack_bf16_cuda, nk_dots_packed_bf16_cuda);
    check("dots_pack_bf16_cuda", test_dots_pack_layout<bf16_t, cuda_backend_t, nk_dots_pack_size_bf16_cuda,
                                                       nk_dots_packed_shape_bf16_cuda, nk_dots_pack_bf16_cuda>);
    check("dots_contract_bf16_cuda", test_dots_launch_contract<bf16_t, cuda_backend_t, nk_dots_pack_size_bf16_cuda,
                                                               nk_dots_packed_bf16_cuda, nk_dots_symmetric_bf16_cuda>);
    check("dots_symmetric_bf16_cuda", test_dots_symmetric<bf16_t, cuda_backend_t>, nk_dots_symmetric_bf16_cuda);
    check("dots_packed_f16_cuda", test_dots_packed<f16_t, cuda_backend_t>, nk_dots_pack_size_f16_cuda,
          nk_dots_pack_f16_cuda, nk_dots_packed_f16_cuda);
    check("dots_pack_f16_cuda", test_dots_pack_layout<f16_t, cuda_backend_t, nk_dots_pack_size_f16_cuda,
                                                      nk_dots_packed_shape_f16_cuda, nk_dots_pack_f16_cuda>);
    check("dots_contract_f16_cuda", test_dots_launch_contract<f16_t, cuda_backend_t, nk_dots_pack_size_f16_cuda,
                                                              nk_dots_packed_f16_cuda, nk_dots_symmetric_f16_cuda>);
    check("dots_symmetric_f16_cuda", test_dots_symmetric<f16_t, cuda_backend_t>, nk_dots_symmetric_f16_cuda);
    check("dots_packed_e5m2_cuda", test_dots_packed<e5m2_t, cuda_backend_t>, nk_dots_pack_size_e5m2_cuda,
          nk_dots_pack_e5m2_cuda, nk_dots_packed_e5m2_cuda);
    check("dots_pack_e5m2_cuda", test_dots_pack_layout<e5m2_t, cuda_backend_t, nk_dots_pack_size_e5m2_cuda,
                                                       nk_dots_packed_shape_e5m2_cuda, nk_dots_pack_e5m2_cuda>);
    check("dots_contract_e5m2_cuda", test_dots_launch_contract<e5m2_t, cuda_backend_t, nk_dots_pack_size_e5m2_cuda,
                                                               nk_dots_packed_e5m2_cuda, nk_dots_symmetric_e5m2_cuda>);
    check("dots_symmetric_e5m2_cuda", test_dots_symmetric<e5m2_t, cuda_backend_t>, nk_dots_symmetric_e5m2_cuda);
    check("dots_packed_e4m3_cuda", test_dots_packed<e4m3_t, cuda_backend_t>, nk_dots_pack_size_e4m3_cuda,
          nk_dots_pack_e4m3_cuda, nk_dots_packed_e4m3_cuda);
    check("dots_pack_e4m3_cuda", test_dots_pack_layout<e4m3_t, cuda_backend_t, nk_dots_pack_size_e4m3_cuda,
                                                       nk_dots_packed_shape_e4m3_cuda, nk_dots_pack_e4m3_cuda>);
    check("dots_contract_e4m3_cuda", test_dots_launch_contract<e4m3_t, cuda_backend_t, nk_dots_pack_size_e4m3_cuda,
                                                               nk_dots_packed_e4m3_cuda, nk_dots_symmetric_e4m3_cuda>);
    check("dots_symmetric_e4m3_cuda", test_dots_symmetric<e4m3_t, cuda_backend_t>, nk_dots_symmetric_e4m3_cuda);
    check("dots_packed_e3m2_cuda", test_dots_packed<e3m2_t, cuda_backend_t>, nk_dots_pack_size_e3m2_cuda,
          nk_dots_pack_e3m2_cuda, nk_dots_packed_e3m2_cuda);
    check("dots_pack_e3m2_cuda", test_dots_pack_layout<e3m2_t, cuda_backend_t, nk_dots_pack_size_e3m2_cuda,
                                                       nk_dots_packed_shape_e3m2_cuda, nk_dots_pack_e3m2_cuda>);
    check("dots_contract_e3m2_cuda", test_dots_launch_contract<e3m2_t, cuda_backend_t, nk_dots_pack_size_e3m2_cuda,
                                                               nk_dots_packed_e3m2_cuda, nk_dots_symmetric_e3m2_cuda>);
    check("dots_symmetric_e3m2_cuda", test_dots_symmetric<e3m2_t, cuda_backend_t>, nk_dots_symmetric_e3m2_cuda);
    check("dots_packed_e2m3_cuda", test_dots_packed<e2m3_t, cuda_backend_t>, nk_dots_pack_size_e2m3_cuda,
          nk_dots_pack_e2m3_cuda, nk_dots_packed_e2m3_cuda);
    check("dots_pack_e2m3_cuda", test_dots_pack_layout<e2m3_t, cuda_backend_t, nk_dots_pack_size_e2m3_cuda,
                                                       nk_dots_packed_shape_e2m3_cuda, nk_dots_pack_e2m3_cuda>);
    check("dots_contract_e2m3_cuda", test_dots_launch_contract<e2m3_t, cuda_backend_t, nk_dots_pack_size_e2m3_cuda,
                                                               nk_dots_packed_e2m3_cuda, nk_dots_symmetric_e2m3_cuda>);
    check("dots_symmetric_e2m3_cuda", test_dots_symmetric<e2m3_t, cuda_backend_t>, nk_dots_symmetric_e2m3_cuda);
    check("dots_packed_e2m1_cuda", test_dots_packed<e2m1x2_t, cuda_backend_t>, nk_dots_pack_size_e2m1_cuda,
          nk_dots_pack_e2m1_cuda, nk_dots_packed_e2m1_cuda);
    check("dots_pack_e2m1_cuda", test_dots_pack_layout<e2m1x2_t, cuda_backend_t, nk_dots_pack_size_e2m1_cuda,
                                                       nk_dots_packed_shape_e2m1_cuda, nk_dots_pack_e2m1_cuda>);
    check("dots_contract_e2m1_cuda", test_dots_launch_contract<e2m1x2_t, cuda_backend_t, nk_dots_pack_size_e2m1_cuda,
                                                               nk_dots_packed_e2m1_cuda, nk_dots_symmetric_e2m1_cuda>);
    check("dots_symmetric_e2m1_cuda", test_dots_symmetric<e2m1x2_t, cuda_backend_t>, nk_dots_symmetric_e2m1_cuda);
    check("dots_packed_nvfp4_cuda", test_dots_packed<nvfp4_t, cuda_backend_t>, nk_dots_pack_size_nvfp4_cuda,
          nk_dots_pack_nvfp4_cuda, nk_dots_packed_nvfp4_cuda);
    check("dots_pack_nvfp4_cuda", test_dots_pack_layout<nvfp4_t, cuda_backend_t, nk_dots_pack_size_nvfp4_cuda,
                                                        nk_dots_packed_shape_nvfp4_cuda, nk_dots_pack_nvfp4_cuda>);
    check("dots_contract_nvfp4_cuda",
          test_dots_launch_contract<nvfp4_t, cuda_backend_t, nk_dots_pack_size_nvfp4_cuda, nk_dots_packed_nvfp4_cuda,
                                    nk_dots_symmetric_nvfp4_cuda>);
    check("dots_symmetric_nvfp4_cuda", test_dots_symmetric<nvfp4_t, cuda_backend_t>, nk_dots_symmetric_nvfp4_cuda);
    check("dots_packed_mxfp4_cuda", test_dots_packed<mxfp4_t, cuda_backend_t>, nk_dots_pack_size_mxfp4_cuda,
          nk_dots_pack_mxfp4_cuda, nk_dots_packed_mxfp4_cuda);
    check("dots_pack_mxfp4_cuda", test_dots_pack_layout<mxfp4_t, cuda_backend_t, nk_dots_pack_size_mxfp4_cuda,
                                                        nk_dots_packed_shape_mxfp4_cuda, nk_dots_pack_mxfp4_cuda>);
    check("dots_contract_mxfp4_cuda",
          test_dots_launch_contract<mxfp4_t, cuda_backend_t, nk_dots_pack_size_mxfp4_cuda, nk_dots_packed_mxfp4_cuda,
                                    nk_dots_symmetric_mxfp4_cuda>);
    check("dots_symmetric_mxfp4_cuda", test_dots_symmetric<mxfp4_t, cuda_backend_t>, nk_dots_symmetric_mxfp4_cuda);
    check("dots_packed_mxfp8e4m3_cuda", test_dots_packed<mxfp8e4m3_t, cuda_backend_t>, nk_dots_pack_size_mxfp8e4m3_cuda,
          nk_dots_pack_mxfp8e4m3_cuda, nk_dots_packed_mxfp8e4m3_cuda);
    check("dots_pack_mxfp8e4m3_cuda",
          test_dots_pack_layout<mxfp8e4m3_t, cuda_backend_t, nk_dots_pack_size_mxfp8e4m3_cuda,
                                nk_dots_packed_shape_mxfp8e4m3_cuda, nk_dots_pack_mxfp8e4m3_cuda>);
    check("dots_contract_mxfp8e4m3_cuda",
          test_dots_launch_contract<mxfp8e4m3_t, cuda_backend_t, nk_dots_pack_size_mxfp8e4m3_cuda,
                                    nk_dots_packed_mxfp8e4m3_cuda, nk_dots_symmetric_mxfp8e4m3_cuda>);
    check("dots_symmetric_mxfp8e4m3_cuda", test_dots_symmetric<mxfp8e4m3_t, cuda_backend_t>,
          nk_dots_symmetric_mxfp8e4m3_cuda);
    check("dots_packed_mxfp8e5m2_cuda", test_dots_packed<mxfp8e5m2_t, cuda_backend_t>, nk_dots_pack_size_mxfp8e5m2_cuda,
          nk_dots_pack_mxfp8e5m2_cuda, nk_dots_packed_mxfp8e5m2_cuda);
    check("dots_pack_mxfp8e5m2_cuda",
          test_dots_pack_layout<mxfp8e5m2_t, cuda_backend_t, nk_dots_pack_size_mxfp8e5m2_cuda,
                                nk_dots_packed_shape_mxfp8e5m2_cuda, nk_dots_pack_mxfp8e5m2_cuda>);
    check("dots_contract_mxfp8e5m2_cuda",
          test_dots_launch_contract<mxfp8e5m2_t, cuda_backend_t, nk_dots_pack_size_mxfp8e5m2_cuda,
                                    nk_dots_packed_mxfp8e5m2_cuda, nk_dots_symmetric_mxfp8e5m2_cuda>);
    check("dots_symmetric_mxfp8e5m2_cuda", test_dots_symmetric<mxfp8e5m2_t, cuda_backend_t>,
          nk_dots_symmetric_mxfp8e5m2_cuda);
    check("dots_packed_i8_cuda", test_dots_packed<i8_t, cuda_backend_t>, nk_dots_pack_size_i8_cuda,
          nk_dots_pack_i8_cuda, nk_dots_packed_i8_cuda);
    check("dots_pack_i8_cuda", test_dots_pack_layout<i8_t, cuda_backend_t, nk_dots_pack_size_i8_cuda,
                                                     nk_dots_packed_shape_i8_cuda, nk_dots_pack_i8_cuda>);
    check("dots_contract_i8_cuda", test_dots_launch_contract<i8_t, cuda_backend_t, nk_dots_pack_size_i8_cuda,
                                                             nk_dots_packed_i8_cuda, nk_dots_symmetric_i8_cuda>);
    check("dots_symmetric_i8_cuda", test_dots_symmetric<i8_t, cuda_backend_t>, nk_dots_symmetric_i8_cuda);
    check("dots_packed_u8_cuda", test_dots_packed<u8_t, cuda_backend_t>, nk_dots_pack_size_u8_cuda,
          nk_dots_pack_u8_cuda, nk_dots_packed_u8_cuda);
    check("dots_pack_u8_cuda", test_dots_pack_layout<u8_t, cuda_backend_t, nk_dots_pack_size_u8_cuda,
                                                     nk_dots_packed_shape_u8_cuda, nk_dots_pack_u8_cuda>);
    check("dots_contract_u8_cuda", test_dots_launch_contract<u8_t, cuda_backend_t, nk_dots_pack_size_u8_cuda,
                                                             nk_dots_packed_u8_cuda, nk_dots_symmetric_u8_cuda>);
    check("dots_symmetric_u8_cuda", test_dots_symmetric<u8_t, cuda_backend_t>, nk_dots_symmetric_u8_cuda);
    check("dots_packed_i4_cuda", test_dots_packed<i4x2_t, cuda_backend_t>, nk_dots_pack_size_i4_cuda,
          nk_dots_pack_i4_cuda, nk_dots_packed_i4_cuda);
    check("dots_pack_i4_cuda", test_dots_pack_layout<i4x2_t, cuda_backend_t, nk_dots_pack_size_i4_cuda,
                                                     nk_dots_packed_shape_i4_cuda, nk_dots_pack_i4_cuda>);
    check("dots_contract_i4_cuda", test_dots_launch_contract<i4x2_t, cuda_backend_t, nk_dots_pack_size_i4_cuda,
                                                             nk_dots_packed_i4_cuda, nk_dots_symmetric_i4_cuda>);
    check("dots_symmetric_i4_cuda", test_dots_symmetric<i4x2_t, cuda_backend_t>, nk_dots_symmetric_i4_cuda);
    check("dots_packed_u4_cuda", test_dots_packed<u4x2_t, cuda_backend_t>, nk_dots_pack_size_u4_cuda,
          nk_dots_pack_u4_cuda, nk_dots_packed_u4_cuda);
    check("dots_pack_u4_cuda", test_dots_pack_layout<u4x2_t, cuda_backend_t, nk_dots_pack_size_u4_cuda,
                                                     nk_dots_packed_shape_u4_cuda, nk_dots_pack_u4_cuda>);
    check("dots_contract_u4_cuda", test_dots_launch_contract<u4x2_t, cuda_backend_t, nk_dots_pack_size_u4_cuda,
                                                             nk_dots_packed_u4_cuda, nk_dots_symmetric_u4_cuda>);
    check("dots_symmetric_u4_cuda", test_dots_symmetric<u4x2_t, cuda_backend_t>, nk_dots_symmetric_u4_cuda);
    check("angulars_packed_f64_cuda", test_angulars_packed<f64_t, cuda_backend_t>, nk_dots_pack_size_f64_cuda,
          nk_dots_pack_f64_cuda, nk_angulars_packed_f64_cuda);
    check("angulars_symmetric_f64_cuda", test_angulars_symmetric<f64_t, cuda_backend_t>,
          nk_angulars_symmetric_f64_cuda);
    check("euclideans_packed_f64_cuda", test_euclideans_packed<f64_t, cuda_backend_t>, nk_dots_pack_size_f64_cuda,
          nk_dots_pack_f64_cuda, nk_euclideans_packed_f64_cuda);
    check("euclideans_symmetric_f64_cuda", test_euclideans_symmetric<f64_t, cuda_backend_t>,
          nk_euclideans_symmetric_f64_cuda);
    check("angulars_packed_f32_cuda", test_angulars_packed<f32_t, cuda_backend_t>, nk_dots_pack_size_f32_cuda,
          nk_dots_pack_f32_cuda, nk_angulars_packed_f32_cuda);
    check("angulars_symmetric_f32_cuda", test_angulars_symmetric<f32_t, cuda_backend_t>,
          nk_angulars_symmetric_f32_cuda);
    check("euclideans_packed_f32_cuda", test_euclideans_packed<f32_t, cuda_backend_t>, nk_dots_pack_size_f32_cuda,
          nk_dots_pack_f32_cuda, nk_euclideans_packed_f32_cuda);
    check("euclideans_symmetric_f32_cuda", test_euclideans_symmetric<f32_t, cuda_backend_t>,
          nk_euclideans_symmetric_f32_cuda);
    check("angulars_packed_bf16_cuda", test_angulars_packed<bf16_t, cuda_backend_t>, nk_dots_pack_size_bf16_cuda,
          nk_dots_pack_bf16_cuda, nk_angulars_packed_bf16_cuda);
    check("angulars_symmetric_bf16_cuda", test_angulars_symmetric<bf16_t, cuda_backend_t>,
          nk_angulars_symmetric_bf16_cuda);
    check("euclideans_packed_bf16_cuda", test_euclideans_packed<bf16_t, cuda_backend_t>, nk_dots_pack_size_bf16_cuda,
          nk_dots_pack_bf16_cuda, nk_euclideans_packed_bf16_cuda);
    check("euclideans_symmetric_bf16_cuda", test_euclideans_symmetric<bf16_t, cuda_backend_t>,
          nk_euclideans_symmetric_bf16_cuda);
    check("angulars_packed_f16_cuda", test_angulars_packed<f16_t, cuda_backend_t>, nk_dots_pack_size_f16_cuda,
          nk_dots_pack_f16_cuda, nk_angulars_packed_f16_cuda);
    check("angulars_symmetric_f16_cuda", test_angulars_symmetric<f16_t, cuda_backend_t>,
          nk_angulars_symmetric_f16_cuda);
    check("euclideans_packed_f16_cuda", test_euclideans_packed<f16_t, cuda_backend_t>, nk_dots_pack_size_f16_cuda,
          nk_dots_pack_f16_cuda, nk_euclideans_packed_f16_cuda);
    check("euclideans_symmetric_f16_cuda", test_euclideans_symmetric<f16_t, cuda_backend_t>,
          nk_euclideans_symmetric_f16_cuda);
    check("angulars_packed_e5m2_cuda", test_angulars_packed<e5m2_t, cuda_backend_t>, nk_dots_pack_size_e5m2_cuda,
          nk_dots_pack_e5m2_cuda, nk_angulars_packed_e5m2_cuda);
    check("angulars_symmetric_e5m2_cuda", test_angulars_symmetric<e5m2_t, cuda_backend_t>,
          nk_angulars_symmetric_e5m2_cuda);
    check("euclideans_packed_e5m2_cuda", test_euclideans_packed<e5m2_t, cuda_backend_t>, nk_dots_pack_size_e5m2_cuda,
          nk_dots_pack_e5m2_cuda, nk_euclideans_packed_e5m2_cuda);
    check("euclideans_symmetric_e5m2_cuda", test_euclideans_symmetric<e5m2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e5m2_cuda);
    check("angulars_packed_e4m3_cuda", test_angulars_packed<e4m3_t, cuda_backend_t>, nk_dots_pack_size_e4m3_cuda,
          nk_dots_pack_e4m3_cuda, nk_angulars_packed_e4m3_cuda);
    check("angulars_symmetric_e4m3_cuda", test_angulars_symmetric<e4m3_t, cuda_backend_t>,
          nk_angulars_symmetric_e4m3_cuda);
    check("euclideans_packed_e4m3_cuda", test_euclideans_packed<e4m3_t, cuda_backend_t>, nk_dots_pack_size_e4m3_cuda,
          nk_dots_pack_e4m3_cuda, nk_euclideans_packed_e4m3_cuda);
    check("euclideans_symmetric_e4m3_cuda", test_euclideans_symmetric<e4m3_t, cuda_backend_t>,
          nk_euclideans_symmetric_e4m3_cuda);
    check("angulars_packed_e3m2_cuda", test_angulars_packed<e3m2_t, cuda_backend_t>, nk_dots_pack_size_e3m2_cuda,
          nk_dots_pack_e3m2_cuda, nk_angulars_packed_e3m2_cuda);
    check("angulars_symmetric_e3m2_cuda", test_angulars_symmetric<e3m2_t, cuda_backend_t>,
          nk_angulars_symmetric_e3m2_cuda);
    check("euclideans_packed_e3m2_cuda", test_euclideans_packed<e3m2_t, cuda_backend_t>, nk_dots_pack_size_e3m2_cuda,
          nk_dots_pack_e3m2_cuda, nk_euclideans_packed_e3m2_cuda);
    check("euclideans_symmetric_e3m2_cuda", test_euclideans_symmetric<e3m2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e3m2_cuda);
    check("angulars_packed_e2m3_cuda", test_angulars_packed<e2m3_t, cuda_backend_t>, nk_dots_pack_size_e2m3_cuda,
          nk_dots_pack_e2m3_cuda, nk_angulars_packed_e2m3_cuda);
    check("angulars_symmetric_e2m3_cuda", test_angulars_symmetric<e2m3_t, cuda_backend_t>,
          nk_angulars_symmetric_e2m3_cuda);
    check("euclideans_packed_e2m3_cuda", test_euclideans_packed<e2m3_t, cuda_backend_t>, nk_dots_pack_size_e2m3_cuda,
          nk_dots_pack_e2m3_cuda, nk_euclideans_packed_e2m3_cuda);
    check("euclideans_symmetric_e2m3_cuda", test_euclideans_symmetric<e2m3_t, cuda_backend_t>,
          nk_euclideans_symmetric_e2m3_cuda);
    check("angulars_packed_e2m1_cuda", test_angulars_packed<e2m1x2_t, cuda_backend_t>, nk_dots_pack_size_e2m1_cuda,
          nk_dots_pack_e2m1_cuda, nk_angulars_packed_e2m1_cuda);
    check("angulars_symmetric_e2m1_cuda", test_angulars_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_angulars_symmetric_e2m1_cuda);
    check("euclideans_packed_e2m1_cuda", test_euclideans_packed<e2m1x2_t, cuda_backend_t>, nk_dots_pack_size_e2m1_cuda,
          nk_dots_pack_e2m1_cuda, nk_euclideans_packed_e2m1_cuda);
    check("euclideans_symmetric_e2m1_cuda", test_euclideans_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e2m1_cuda);
    check("angulars_packed_nvfp4_cuda", test_angulars_packed<nvfp4_t, cuda_backend_t>, nk_dots_pack_size_nvfp4_cuda,
          nk_dots_pack_nvfp4_cuda, nk_angulars_packed_nvfp4_cuda);
    check("angulars_symmetric_nvfp4_cuda", test_angulars_symmetric<nvfp4_t, cuda_backend_t>,
          nk_angulars_symmetric_nvfp4_cuda);
    check("euclideans_packed_nvfp4_cuda", test_euclideans_packed<nvfp4_t, cuda_backend_t>, nk_dots_pack_size_nvfp4_cuda,
          nk_dots_pack_nvfp4_cuda, nk_euclideans_packed_nvfp4_cuda);
    check("euclideans_symmetric_nvfp4_cuda", test_euclideans_symmetric<nvfp4_t, cuda_backend_t>,
          nk_euclideans_symmetric_nvfp4_cuda);
    check("angulars_packed_mxfp4_cuda", test_angulars_packed<mxfp4_t, cuda_backend_t>, nk_dots_pack_size_mxfp4_cuda,
          nk_dots_pack_mxfp4_cuda, nk_angulars_packed_mxfp4_cuda);
    check("angulars_symmetric_mxfp4_cuda", test_angulars_symmetric<mxfp4_t, cuda_backend_t>,
          nk_angulars_symmetric_mxfp4_cuda);
    check("euclideans_packed_mxfp4_cuda", test_euclideans_packed<mxfp4_t, cuda_backend_t>, nk_dots_pack_size_mxfp4_cuda,
          nk_dots_pack_mxfp4_cuda, nk_euclideans_packed_mxfp4_cuda);
    check("euclideans_symmetric_mxfp4_cuda", test_euclideans_symmetric<mxfp4_t, cuda_backend_t>,
          nk_euclideans_symmetric_mxfp4_cuda);
    check("angulars_packed_mxfp8e4m3_cuda", test_angulars_packed<mxfp8e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e4m3_cuda, nk_dots_pack_mxfp8e4m3_cuda, nk_angulars_packed_mxfp8e4m3_cuda);
    check("angulars_symmetric_mxfp8e4m3_cuda", test_angulars_symmetric<mxfp8e4m3_t, cuda_backend_t>,
          nk_angulars_symmetric_mxfp8e4m3_cuda);
    check("euclideans_packed_mxfp8e4m3_cuda", test_euclideans_packed<mxfp8e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e4m3_cuda, nk_dots_pack_mxfp8e4m3_cuda, nk_euclideans_packed_mxfp8e4m3_cuda);
    check("euclideans_symmetric_mxfp8e4m3_cuda", test_euclideans_symmetric<mxfp8e4m3_t, cuda_backend_t>,
          nk_euclideans_symmetric_mxfp8e4m3_cuda);
    check("angulars_packed_mxfp8e5m2_cuda", test_angulars_packed<mxfp8e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e5m2_cuda, nk_dots_pack_mxfp8e5m2_cuda, nk_angulars_packed_mxfp8e5m2_cuda);
    check("angulars_symmetric_mxfp8e5m2_cuda", test_angulars_symmetric<mxfp8e5m2_t, cuda_backend_t>,
          nk_angulars_symmetric_mxfp8e5m2_cuda);
    check("euclideans_packed_mxfp8e5m2_cuda", test_euclideans_packed<mxfp8e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e5m2_cuda, nk_dots_pack_mxfp8e5m2_cuda, nk_euclideans_packed_mxfp8e5m2_cuda);
    check("euclideans_symmetric_mxfp8e5m2_cuda", test_euclideans_symmetric<mxfp8e5m2_t, cuda_backend_t>,
          nk_euclideans_symmetric_mxfp8e5m2_cuda);
    check("angulars_packed_i8_cuda", test_angulars_packed<i8_t, cuda_backend_t>, nk_dots_pack_size_i8_cuda,
          nk_dots_pack_i8_cuda, nk_angulars_packed_i8_cuda);
    check("angulars_symmetric_i8_cuda", test_angulars_symmetric<i8_t, cuda_backend_t>, nk_angulars_symmetric_i8_cuda);
    check("euclideans_packed_i8_cuda", test_euclideans_packed<i8_t, cuda_backend_t>, nk_dots_pack_size_i8_cuda,
          nk_dots_pack_i8_cuda, nk_euclideans_packed_i8_cuda);
    check("euclideans_symmetric_i8_cuda", test_euclideans_symmetric<i8_t, cuda_backend_t>,
          nk_euclideans_symmetric_i8_cuda);
    check("angulars_packed_u8_cuda", test_angulars_packed<u8_t, cuda_backend_t>, nk_dots_pack_size_u8_cuda,
          nk_dots_pack_u8_cuda, nk_angulars_packed_u8_cuda);
    check("angulars_symmetric_u8_cuda", test_angulars_symmetric<u8_t, cuda_backend_t>, nk_angulars_symmetric_u8_cuda);
    check("euclideans_packed_u8_cuda", test_euclideans_packed<u8_t, cuda_backend_t>, nk_dots_pack_size_u8_cuda,
          nk_dots_pack_u8_cuda, nk_euclideans_packed_u8_cuda);
    check("euclideans_symmetric_u8_cuda", test_euclideans_symmetric<u8_t, cuda_backend_t>,
          nk_euclideans_symmetric_u8_cuda);
    check("angulars_packed_i4_cuda", test_angulars_packed<i4x2_t, cuda_backend_t>, nk_dots_pack_size_i4_cuda,
          nk_dots_pack_i4_cuda, nk_angulars_packed_i4_cuda);
    check("angulars_symmetric_i4_cuda", test_angulars_symmetric<i4x2_t, cuda_backend_t>, nk_angulars_symmetric_i4_cuda);
    check("euclideans_packed_i4_cuda", test_euclideans_packed<i4x2_t, cuda_backend_t>, nk_dots_pack_size_i4_cuda,
          nk_dots_pack_i4_cuda, nk_euclideans_packed_i4_cuda);
    check("euclideans_symmetric_i4_cuda", test_euclideans_symmetric<i4x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_i4_cuda);
    check("angulars_packed_u4_cuda", test_angulars_packed<u4x2_t, cuda_backend_t>, nk_dots_pack_size_u4_cuda,
          nk_dots_pack_u4_cuda, nk_angulars_packed_u4_cuda);
    check("angulars_symmetric_u4_cuda", test_angulars_symmetric<u4x2_t, cuda_backend_t>, nk_angulars_symmetric_u4_cuda);
    check("euclideans_packed_u4_cuda", test_euclideans_packed<u4x2_t, cuda_backend_t>, nk_dots_pack_size_u4_cuda,
          nk_dots_pack_u4_cuda, nk_euclideans_packed_u4_cuda);
    check("euclideans_symmetric_u4_cuda", test_euclideans_symmetric<u4x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_u4_cuda);
    check("attention_packed_bf16_cuda", test_attention_packed<bf16_t, cuda_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_bf16_cuda, nk_attention_pack_bf16_cuda, nk_attention_packed_bf16_cuda);
    check("attention_packed_f16_cuda", test_attention_packed<f16_t, cuda_backend_t, attention_weights_t::bits_11_k>,
          nk_attention_pack_size_f16_cuda, nk_attention_pack_f16_cuda, nk_attention_packed_f16_cuda);
    check("attention_packed_gradients_bf16_cuda", test_attention_packed_gradients<bf16_t, cuda_backend_t>,
          nk_attention_pack_size_bf16_cuda, nk_attention_pack_bf16_cuda, nk_attention_packed_gradients_bf16_cuda);
    check("attention_packed_e4m3_cuda", test_attention_packed<e4m3_t, cuda_backend_t, attention_weights_t::bits_11_k>,
          nk_attention_pack_size_e4m3_cuda, nk_attention_pack_e4m3_cuda, nk_attention_packed_e4m3_cuda);
    check("attention_packed_i8_cuda", test_attention_packed<i8_t, cuda_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_i8_cuda, nk_attention_pack_i8_cuda, nk_attention_packed_i8_cuda);
    check("attention_rope_f32_cuda", test_attention_rope<f32_t, cuda_backend_t>, nk_attention_rope_f32_cuda);
    check("attention_rope_bf16_cuda", test_attention_rope<bf16_t, cuda_backend_t>, nk_attention_rope_bf16_cuda);
    check("attention_rope_e4m3_cuda", test_attention_rope<e4m3_t, cuda_backend_t>, nk_attention_rope_e4m3_cuda);

#if NUMKONG_TARGET_AMPERE
    check.section("Cross Ampere", nk_cap_ampere_k);
    check("dots_packed_bf16_ampere", test_dots_packed<bf16_t, cuda_backend_t>, nk_dots_pack_size_bf16_ampere,
          nk_dots_pack_bf16_ampere, nk_dots_packed_bf16_ampere);
    check("dots_pack_bf16_ampere", test_dots_pack_layout<bf16_t, cuda_backend_t, nk_dots_pack_size_bf16_ampere,
                                                         nk_dots_packed_shape_bf16_ampere, nk_dots_pack_bf16_ampere>);
    check("dots_contract_bf16_ampere",
          test_dots_launch_contract<bf16_t, cuda_backend_t, nk_dots_pack_size_bf16_ampere, nk_dots_packed_bf16_ampere,
                                    nk_dots_symmetric_bf16_ampere>);
    check("dots_symmetric_bf16_ampere", test_dots_symmetric<bf16_t, cuda_backend_t>, nk_dots_symmetric_bf16_ampere);
    check("dots_packed_f16_ampere", test_dots_packed<f16_t, cuda_backend_t>, nk_dots_pack_size_f16_ampere,
          nk_dots_pack_f16_ampere, nk_dots_packed_f16_ampere);
    check("dots_pack_f16_ampere", test_dots_pack_layout<f16_t, cuda_backend_t, nk_dots_pack_size_f16_ampere,
                                                        nk_dots_packed_shape_f16_ampere, nk_dots_pack_f16_ampere>);
    check("dots_contract_f16_ampere",
          test_dots_launch_contract<f16_t, cuda_backend_t, nk_dots_pack_size_f16_ampere, nk_dots_packed_f16_ampere,
                                    nk_dots_symmetric_f16_ampere>);
    check("dots_symmetric_f16_ampere", test_dots_symmetric<f16_t, cuda_backend_t>, nk_dots_symmetric_f16_ampere);
    check("dots_packed_e5m2_ampere", test_dots_packed<e5m2_t, cuda_backend_t>, nk_dots_pack_size_e5m2_ampere,
          nk_dots_pack_e5m2_ampere, nk_dots_packed_e5m2_ampere);
    check("dots_pack_e5m2_ampere", test_dots_pack_layout<e5m2_t, cuda_backend_t, nk_dots_pack_size_e5m2_ampere,
                                                         nk_dots_packed_shape_e5m2_ampere, nk_dots_pack_e5m2_ampere>);
    check("dots_contract_e5m2_ampere",
          test_dots_launch_contract<e5m2_t, cuda_backend_t, nk_dots_pack_size_e5m2_ampere, nk_dots_packed_e5m2_ampere,
                                    nk_dots_symmetric_e5m2_ampere>);
    check("dots_symmetric_e5m2_ampere", test_dots_symmetric<e5m2_t, cuda_backend_t>, nk_dots_symmetric_e5m2_ampere);
    check("dots_packed_e4m3_ampere", test_dots_packed<e4m3_t, cuda_backend_t>, nk_dots_pack_size_e4m3_ampere,
          nk_dots_pack_e4m3_ampere, nk_dots_packed_e4m3_ampere);
    check("dots_pack_e4m3_ampere", test_dots_pack_layout<e4m3_t, cuda_backend_t, nk_dots_pack_size_e4m3_ampere,
                                                         nk_dots_packed_shape_e4m3_ampere, nk_dots_pack_e4m3_ampere>);
    check("dots_contract_e4m3_ampere",
          test_dots_launch_contract<e4m3_t, cuda_backend_t, nk_dots_pack_size_e4m3_ampere, nk_dots_packed_e4m3_ampere,
                                    nk_dots_symmetric_e4m3_ampere>);
    check("dots_symmetric_e4m3_ampere", test_dots_symmetric<e4m3_t, cuda_backend_t>, nk_dots_symmetric_e4m3_ampere);
    check("dots_packed_e3m2_ampere", test_dots_packed<e3m2_t, cuda_backend_t>, nk_dots_pack_size_e3m2_ampere,
          nk_dots_pack_e3m2_ampere, nk_dots_packed_e3m2_ampere);
    check("dots_pack_e3m2_ampere", test_dots_pack_layout<e3m2_t, cuda_backend_t, nk_dots_pack_size_e3m2_ampere,
                                                         nk_dots_packed_shape_e3m2_ampere, nk_dots_pack_e3m2_ampere>);
    check("dots_contract_e3m2_ampere",
          test_dots_launch_contract<e3m2_t, cuda_backend_t, nk_dots_pack_size_e3m2_ampere, nk_dots_packed_e3m2_ampere,
                                    nk_dots_symmetric_e3m2_ampere>);
    check("dots_symmetric_e3m2_ampere", test_dots_symmetric<e3m2_t, cuda_backend_t>, nk_dots_symmetric_e3m2_ampere);
    check("dots_packed_e2m3_ampere", test_dots_packed<e2m3_t, cuda_backend_t>, nk_dots_pack_size_e2m3_ampere,
          nk_dots_pack_e2m3_ampere, nk_dots_packed_e2m3_ampere);
    check("dots_pack_e2m3_ampere", test_dots_pack_layout<e2m3_t, cuda_backend_t, nk_dots_pack_size_e2m3_ampere,
                                                         nk_dots_packed_shape_e2m3_ampere, nk_dots_pack_e2m3_ampere>);
    check("dots_contract_e2m3_ampere",
          test_dots_launch_contract<e2m3_t, cuda_backend_t, nk_dots_pack_size_e2m3_ampere, nk_dots_packed_e2m3_ampere,
                                    nk_dots_symmetric_e2m3_ampere>);
    check("dots_symmetric_e2m3_ampere", test_dots_symmetric<e2m3_t, cuda_backend_t>, nk_dots_symmetric_e2m3_ampere);
    check("dots_packed_e2m1_ampere", test_dots_packed<e2m1x2_t, cuda_backend_t>, nk_dots_pack_size_e2m1_ampere,
          nk_dots_pack_e2m1_ampere, nk_dots_packed_e2m1_ampere);
    check("dots_pack_e2m1_ampere", test_dots_pack_layout<e2m1x2_t, cuda_backend_t, nk_dots_pack_size_e2m1_ampere,
                                                         nk_dots_packed_shape_e2m1_ampere, nk_dots_pack_e2m1_ampere>);
    check("dots_contract_e2m1_ampere",
          test_dots_launch_contract<e2m1x2_t, cuda_backend_t, nk_dots_pack_size_e2m1_ampere, nk_dots_packed_e2m1_ampere,
                                    nk_dots_symmetric_e2m1_ampere>);
    check("dots_symmetric_e2m1_ampere", test_dots_symmetric<e2m1x2_t, cuda_backend_t>, nk_dots_symmetric_e2m1_ampere);
    check("dots_packed_i8_ampere", test_dots_packed<i8_t, cuda_backend_t>, nk_dots_pack_size_i8_ampere,
          nk_dots_pack_i8_ampere, nk_dots_packed_i8_ampere);
    check("dots_pack_i8_ampere", test_dots_pack_layout<i8_t, cuda_backend_t, nk_dots_pack_size_i8_ampere,
                                                       nk_dots_packed_shape_i8_ampere, nk_dots_pack_i8_ampere>);
    check("dots_contract_i8_ampere", test_dots_launch_contract<i8_t, cuda_backend_t, nk_dots_pack_size_i8_ampere,
                                                               nk_dots_packed_i8_ampere, nk_dots_symmetric_i8_ampere>);
    check("dots_symmetric_i8_ampere", test_dots_symmetric<i8_t, cuda_backend_t>, nk_dots_symmetric_i8_ampere);
    check("dots_packed_u8_ampere", test_dots_packed<u8_t, cuda_backend_t>, nk_dots_pack_size_u8_ampere,
          nk_dots_pack_u8_ampere, nk_dots_packed_u8_ampere);
    check("dots_pack_u8_ampere", test_dots_pack_layout<u8_t, cuda_backend_t, nk_dots_pack_size_u8_ampere,
                                                       nk_dots_packed_shape_u8_ampere, nk_dots_pack_u8_ampere>);
    check("dots_contract_u8_ampere", test_dots_launch_contract<u8_t, cuda_backend_t, nk_dots_pack_size_u8_ampere,
                                                               nk_dots_packed_u8_ampere, nk_dots_symmetric_u8_ampere>);
    check("dots_symmetric_u8_ampere", test_dots_symmetric<u8_t, cuda_backend_t>, nk_dots_symmetric_u8_ampere);
    check("dots_packed_i4_ampere", test_dots_packed<i4x2_t, cuda_backend_t>, nk_dots_pack_size_i4_ampere,
          nk_dots_pack_i4_ampere, nk_dots_packed_i4_ampere);
    check("dots_pack_i4_ampere", test_dots_pack_layout<i4x2_t, cuda_backend_t, nk_dots_pack_size_i4_ampere,
                                                       nk_dots_packed_shape_i4_ampere, nk_dots_pack_i4_ampere>);
    check("dots_contract_i4_ampere", test_dots_launch_contract<i4x2_t, cuda_backend_t, nk_dots_pack_size_i4_ampere,
                                                               nk_dots_packed_i4_ampere, nk_dots_symmetric_i4_ampere>);
    check("dots_symmetric_i4_ampere", test_dots_symmetric<i4x2_t, cuda_backend_t>, nk_dots_symmetric_i4_ampere);
    check("dots_packed_u4_ampere", test_dots_packed<u4x2_t, cuda_backend_t>, nk_dots_pack_size_u4_ampere,
          nk_dots_pack_u4_ampere, nk_dots_packed_u4_ampere);
    check("dots_pack_u4_ampere", test_dots_pack_layout<u4x2_t, cuda_backend_t, nk_dots_pack_size_u4_ampere,
                                                       nk_dots_packed_shape_u4_ampere, nk_dots_pack_u4_ampere>);
    check("dots_contract_u4_ampere", test_dots_launch_contract<u4x2_t, cuda_backend_t, nk_dots_pack_size_u4_ampere,
                                                               nk_dots_packed_u4_ampere, nk_dots_symmetric_u4_ampere>);
    check("dots_symmetric_u4_ampere", test_dots_symmetric<u4x2_t, cuda_backend_t>, nk_dots_symmetric_u4_ampere);
    check("angulars_packed_bf16_ampere", test_angulars_packed<bf16_t, cuda_backend_t>, nk_dots_pack_size_bf16_ampere,
          nk_dots_pack_bf16_ampere, nk_angulars_packed_bf16_ampere);
    check("angulars_symmetric_bf16_ampere", test_angulars_symmetric<bf16_t, cuda_backend_t>,
          nk_angulars_symmetric_bf16_ampere);
    check("euclideans_packed_bf16_ampere", test_euclideans_packed<bf16_t, cuda_backend_t>,
          nk_dots_pack_size_bf16_ampere, nk_dots_pack_bf16_ampere, nk_euclideans_packed_bf16_ampere);
    check("euclideans_symmetric_bf16_ampere", test_euclideans_symmetric<bf16_t, cuda_backend_t>,
          nk_euclideans_symmetric_bf16_ampere);
    check("angulars_packed_f16_ampere", test_angulars_packed<f16_t, cuda_backend_t>, nk_dots_pack_size_f16_ampere,
          nk_dots_pack_f16_ampere, nk_angulars_packed_f16_ampere);
    check("angulars_symmetric_f16_ampere", test_angulars_symmetric<f16_t, cuda_backend_t>,
          nk_angulars_symmetric_f16_ampere);
    check("euclideans_packed_f16_ampere", test_euclideans_packed<f16_t, cuda_backend_t>, nk_dots_pack_size_f16_ampere,
          nk_dots_pack_f16_ampere, nk_euclideans_packed_f16_ampere);
    check("euclideans_symmetric_f16_ampere", test_euclideans_symmetric<f16_t, cuda_backend_t>,
          nk_euclideans_symmetric_f16_ampere);
    check("angulars_packed_e5m2_ampere", test_angulars_packed<e5m2_t, cuda_backend_t>, nk_dots_pack_size_e5m2_ampere,
          nk_dots_pack_e5m2_ampere, nk_angulars_packed_e5m2_ampere);
    check("angulars_symmetric_e5m2_ampere", test_angulars_symmetric<e5m2_t, cuda_backend_t>,
          nk_angulars_symmetric_e5m2_ampere);
    check("euclideans_packed_e5m2_ampere", test_euclideans_packed<e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_e5m2_ampere, nk_dots_pack_e5m2_ampere, nk_euclideans_packed_e5m2_ampere);
    check("euclideans_symmetric_e5m2_ampere", test_euclideans_symmetric<e5m2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e5m2_ampere);
    check("angulars_packed_e4m3_ampere", test_angulars_packed<e4m3_t, cuda_backend_t>, nk_dots_pack_size_e4m3_ampere,
          nk_dots_pack_e4m3_ampere, nk_angulars_packed_e4m3_ampere);
    check("angulars_packed_nan_e4m3_ampere",
          test_angulars_packed_nan_e4m3<nk_dots_pack_size_e4m3_ampere, nk_dots_pack_e4m3_ampere,
                                        nk_angulars_packed_e4m3_ampere>);
    check("angulars_symmetric_e4m3_ampere", test_angulars_symmetric<e4m3_t, cuda_backend_t>,
          nk_angulars_symmetric_e4m3_ampere);
    check("euclideans_packed_e4m3_ampere", test_euclideans_packed<e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_e4m3_ampere, nk_dots_pack_e4m3_ampere, nk_euclideans_packed_e4m3_ampere);
    check("euclideans_symmetric_e4m3_ampere", test_euclideans_symmetric<e4m3_t, cuda_backend_t>,
          nk_euclideans_symmetric_e4m3_ampere);
    check("angulars_packed_e3m2_ampere", test_angulars_packed<e3m2_t, cuda_backend_t>, nk_dots_pack_size_e3m2_ampere,
          nk_dots_pack_e3m2_ampere, nk_angulars_packed_e3m2_ampere);
    check("angulars_symmetric_e3m2_ampere", test_angulars_symmetric<e3m2_t, cuda_backend_t>,
          nk_angulars_symmetric_e3m2_ampere);
    check("euclideans_packed_e3m2_ampere", test_euclideans_packed<e3m2_t, cuda_backend_t>,
          nk_dots_pack_size_e3m2_ampere, nk_dots_pack_e3m2_ampere, nk_euclideans_packed_e3m2_ampere);
    check("euclideans_symmetric_e3m2_ampere", test_euclideans_symmetric<e3m2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e3m2_ampere);
    check("angulars_packed_e2m3_ampere", test_angulars_packed<e2m3_t, cuda_backend_t>, nk_dots_pack_size_e2m3_ampere,
          nk_dots_pack_e2m3_ampere, nk_angulars_packed_e2m3_ampere);
    check("angulars_symmetric_e2m3_ampere", test_angulars_symmetric<e2m3_t, cuda_backend_t>,
          nk_angulars_symmetric_e2m3_ampere);
    check("euclideans_packed_e2m3_ampere", test_euclideans_packed<e2m3_t, cuda_backend_t>,
          nk_dots_pack_size_e2m3_ampere, nk_dots_pack_e2m3_ampere, nk_euclideans_packed_e2m3_ampere);
    check("euclideans_symmetric_e2m3_ampere", test_euclideans_symmetric<e2m3_t, cuda_backend_t>,
          nk_euclideans_symmetric_e2m3_ampere);
    check("angulars_packed_e2m1_ampere", test_angulars_packed<e2m1x2_t, cuda_backend_t>, nk_dots_pack_size_e2m1_ampere,
          nk_dots_pack_e2m1_ampere, nk_angulars_packed_e2m1_ampere);
    check("angulars_symmetric_e2m1_ampere", test_angulars_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_angulars_symmetric_e2m1_ampere);
    check("euclideans_packed_e2m1_ampere", test_euclideans_packed<e2m1x2_t, cuda_backend_t>,
          nk_dots_pack_size_e2m1_ampere, nk_dots_pack_e2m1_ampere, nk_euclideans_packed_e2m1_ampere);
    check("euclideans_symmetric_e2m1_ampere", test_euclideans_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e2m1_ampere);
    check("angulars_packed_i8_ampere", test_angulars_packed<i8_t, cuda_backend_t>, nk_dots_pack_size_i8_ampere,
          nk_dots_pack_i8_ampere, nk_angulars_packed_i8_ampere);
    check("angulars_symmetric_i8_ampere", test_angulars_symmetric<i8_t, cuda_backend_t>,
          nk_angulars_symmetric_i8_ampere);
    check("euclideans_packed_i8_ampere", test_euclideans_packed<i8_t, cuda_backend_t>, nk_dots_pack_size_i8_ampere,
          nk_dots_pack_i8_ampere, nk_euclideans_packed_i8_ampere);
    check("euclideans_symmetric_i8_ampere", test_euclideans_symmetric<i8_t, cuda_backend_t>,
          nk_euclideans_symmetric_i8_ampere);
    check("angulars_packed_u8_ampere", test_angulars_packed<u8_t, cuda_backend_t>, nk_dots_pack_size_u8_ampere,
          nk_dots_pack_u8_ampere, nk_angulars_packed_u8_ampere);
    check("angulars_symmetric_u8_ampere", test_angulars_symmetric<u8_t, cuda_backend_t>,
          nk_angulars_symmetric_u8_ampere);
    check("euclideans_packed_u8_ampere", test_euclideans_packed<u8_t, cuda_backend_t>, nk_dots_pack_size_u8_ampere,
          nk_dots_pack_u8_ampere, nk_euclideans_packed_u8_ampere);
    check("euclideans_symmetric_u8_ampere", test_euclideans_symmetric<u8_t, cuda_backend_t>,
          nk_euclideans_symmetric_u8_ampere);
    check("angulars_packed_i4_ampere", test_angulars_packed<i4x2_t, cuda_backend_t>, nk_dots_pack_size_i4_ampere,
          nk_dots_pack_i4_ampere, nk_angulars_packed_i4_ampere);
    check("angulars_symmetric_i4_ampere", test_angulars_symmetric<i4x2_t, cuda_backend_t>,
          nk_angulars_symmetric_i4_ampere);
    check("euclideans_packed_i4_ampere", test_euclideans_packed<i4x2_t, cuda_backend_t>, nk_dots_pack_size_i4_ampere,
          nk_dots_pack_i4_ampere, nk_euclideans_packed_i4_ampere);
    check("euclideans_symmetric_i4_ampere", test_euclideans_symmetric<i4x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_i4_ampere);
    check("angulars_packed_u4_ampere", test_angulars_packed<u4x2_t, cuda_backend_t>, nk_dots_pack_size_u4_ampere,
          nk_dots_pack_u4_ampere, nk_angulars_packed_u4_ampere);
    check("angulars_symmetric_u4_ampere", test_angulars_symmetric<u4x2_t, cuda_backend_t>,
          nk_angulars_symmetric_u4_ampere);
    check("euclideans_packed_u4_ampere", test_euclideans_packed<u4x2_t, cuda_backend_t>, nk_dots_pack_size_u4_ampere,
          nk_dots_pack_u4_ampere, nk_euclideans_packed_u4_ampere);
    check("euclideans_symmetric_u4_ampere", test_euclideans_symmetric<u4x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_u4_ampere);
    check("attention_packed_bf16_ampere", test_attention_packed<bf16_t, cuda_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_bf16_ampere, nk_attention_pack_bf16_ampere, nk_attention_packed_bf16_ampere);
    check("attention_packed_f16_ampere", test_attention_packed<f16_t, cuda_backend_t, attention_weights_t::bits_11_k>,
          nk_attention_pack_size_f16_ampere, nk_attention_pack_f16_ampere, nk_attention_packed_f16_ampere);
    check("attention_packed_gradients_bf16_ampere", test_attention_packed_gradients<bf16_t, cuda_backend_t>,
          nk_attention_pack_size_bf16_ampere, nk_attention_pack_bf16_ampere, nk_attention_packed_gradients_bf16_ampere);
    check("attention_packed_e4m3_ampere", test_attention_packed<e4m3_t, cuda_backend_t, attention_weights_t::bits_11_k>,
          nk_attention_pack_size_e4m3_ampere, nk_attention_pack_e4m3_ampere, nk_attention_packed_e4m3_ampere);
    check("attention_packed_i8_ampere", test_attention_packed<i8_t, cuda_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_i8_ampere, nk_attention_pack_i8_ampere, nk_attention_packed_i8_ampere);
#endif // NUMKONG_TARGET_AMPERE

#if NUMKONG_TARGET_HOPPER
    check.section("Cross Hopper", nk_cap_hopper_k);
    check("dots_packed_bf16_hopper", test_dots_packed<bf16_t, cuda_backend_t>, nk_dots_pack_size_bf16_hopper,
          nk_dots_pack_bf16_hopper, nk_dots_packed_bf16_hopper);
    check("dots_pack_bf16_hopper", test_dots_pack_layout<bf16_t, cuda_backend_t, nk_dots_pack_size_bf16_hopper,
                                                         nk_dots_packed_shape_bf16_hopper, nk_dots_pack_bf16_hopper>);
    check("dots_contract_bf16_hopper",
          test_dots_launch_contract<bf16_t, cuda_backend_t, nk_dots_pack_size_bf16_hopper, nk_dots_packed_bf16_hopper,
                                    nk_dots_symmetric_bf16_hopper>);
    check("dots_symmetric_bf16_hopper", test_dots_symmetric<bf16_t, cuda_backend_t>, nk_dots_symmetric_bf16_hopper);
    check("dots_packed_f16_hopper", test_dots_packed<f16_t, cuda_backend_t>, nk_dots_pack_size_f16_hopper,
          nk_dots_pack_f16_hopper, nk_dots_packed_f16_hopper);
    check("dots_pack_f16_hopper", test_dots_pack_layout<f16_t, cuda_backend_t, nk_dots_pack_size_f16_hopper,
                                                        nk_dots_packed_shape_f16_hopper, nk_dots_pack_f16_hopper>);
    check("dots_contract_f16_hopper",
          test_dots_launch_contract<f16_t, cuda_backend_t, nk_dots_pack_size_f16_hopper, nk_dots_packed_f16_hopper,
                                    nk_dots_symmetric_f16_hopper>);
    check("dots_symmetric_f16_hopper", test_dots_symmetric<f16_t, cuda_backend_t>, nk_dots_symmetric_f16_hopper);
    check("dots_packed_e2m3_hopper", test_dots_packed<e2m3_t, cuda_backend_t>, nk_dots_pack_size_e2m3_hopper,
          nk_dots_pack_e2m3_hopper, nk_dots_packed_e2m3_hopper);
    check("dots_pack_e2m3_hopper", test_dots_pack_layout<e2m3_t, cuda_backend_t, nk_dots_pack_size_e2m3_hopper,
                                                         nk_dots_packed_shape_e2m3_hopper, nk_dots_pack_e2m3_hopper>);
    check("dots_contract_e2m3_hopper",
          test_dots_launch_contract<e2m3_t, cuda_backend_t, nk_dots_pack_size_e2m3_hopper, nk_dots_packed_e2m3_hopper,
                                    nk_dots_symmetric_e2m3_hopper>);
    check("dots_symmetric_e2m3_hopper", test_dots_symmetric<e2m3_t, cuda_backend_t>, nk_dots_symmetric_e2m3_hopper);
    check("dots_packed_e2m1_hopper", test_dots_packed<e2m1x2_t, cuda_backend_t>, nk_dots_pack_size_e2m1_hopper,
          nk_dots_pack_e2m1_hopper, nk_dots_packed_e2m1_hopper);
    check("dots_pack_e2m1_hopper", test_dots_pack_layout<e2m1x2_t, cuda_backend_t, nk_dots_pack_size_e2m1_hopper,
                                                         nk_dots_packed_shape_e2m1_hopper, nk_dots_pack_e2m1_hopper>);
    check("dots_contract_e2m1_hopper",
          test_dots_launch_contract<e2m1x2_t, cuda_backend_t, nk_dots_pack_size_e2m1_hopper, nk_dots_packed_e2m1_hopper,
                                    nk_dots_symmetric_e2m1_hopper>);
    check("dots_symmetric_e2m1_hopper", test_dots_symmetric<e2m1x2_t, cuda_backend_t>, nk_dots_symmetric_e2m1_hopper);
    check("dots_packed_i8_hopper", test_dots_packed<i8_t, cuda_backend_t>, nk_dots_pack_size_i8_hopper,
          nk_dots_pack_i8_hopper, nk_dots_packed_i8_hopper);
    check("dots_pack_i8_hopper", test_dots_pack_layout<i8_t, cuda_backend_t, nk_dots_pack_size_i8_hopper,
                                                       nk_dots_packed_shape_i8_hopper, nk_dots_pack_i8_hopper>);
    check("dots_contract_i8_hopper", test_dots_launch_contract<i8_t, cuda_backend_t, nk_dots_pack_size_i8_hopper,
                                                               nk_dots_packed_i8_hopper, nk_dots_symmetric_i8_hopper>);
    check("dots_symmetric_i8_hopper", test_dots_symmetric<i8_t, cuda_backend_t>, nk_dots_symmetric_i8_hopper);
    check("dots_packed_u8_hopper", test_dots_packed<u8_t, cuda_backend_t>, nk_dots_pack_size_u8_hopper,
          nk_dots_pack_u8_hopper, nk_dots_packed_u8_hopper);
    check("dots_pack_u8_hopper", test_dots_pack_layout<u8_t, cuda_backend_t, nk_dots_pack_size_u8_hopper,
                                                       nk_dots_packed_shape_u8_hopper, nk_dots_pack_u8_hopper>);
    check("dots_contract_u8_hopper", test_dots_launch_contract<u8_t, cuda_backend_t, nk_dots_pack_size_u8_hopper,
                                                               nk_dots_packed_u8_hopper, nk_dots_symmetric_u8_hopper>);
    check("dots_symmetric_u8_hopper", test_dots_symmetric<u8_t, cuda_backend_t>, nk_dots_symmetric_u8_hopper);
    check("dots_packed_i4_hopper", test_dots_packed<i4x2_t, cuda_backend_t>, nk_dots_pack_size_i4_hopper,
          nk_dots_pack_i4_hopper, nk_dots_packed_i4_hopper);
    check("dots_pack_i4_hopper", test_dots_pack_layout<i4x2_t, cuda_backend_t, nk_dots_pack_size_i4_hopper,
                                                       nk_dots_packed_shape_i4_hopper, nk_dots_pack_i4_hopper>);
    check("dots_contract_i4_hopper", test_dots_launch_contract<i4x2_t, cuda_backend_t, nk_dots_pack_size_i4_hopper,
                                                               nk_dots_packed_i4_hopper, nk_dots_symmetric_i4_hopper>);
    check("dots_symmetric_i4_hopper", test_dots_symmetric<i4x2_t, cuda_backend_t>, nk_dots_symmetric_i4_hopper);
    check("dots_packed_u4_hopper", test_dots_packed<u4x2_t, cuda_backend_t>, nk_dots_pack_size_u4_hopper,
          nk_dots_pack_u4_hopper, nk_dots_packed_u4_hopper);
    check("dots_pack_u4_hopper", test_dots_pack_layout<u4x2_t, cuda_backend_t, nk_dots_pack_size_u4_hopper,
                                                       nk_dots_packed_shape_u4_hopper, nk_dots_pack_u4_hopper>);
    check("dots_contract_u4_hopper", test_dots_launch_contract<u4x2_t, cuda_backend_t, nk_dots_pack_size_u4_hopper,
                                                               nk_dots_packed_u4_hopper, nk_dots_symmetric_u4_hopper>);
    check("dots_symmetric_u4_hopper", test_dots_symmetric<u4x2_t, cuda_backend_t>, nk_dots_symmetric_u4_hopper);
    check("angulars_packed_bf16_hopper", test_angulars_packed<bf16_t, cuda_backend_t>, nk_dots_pack_size_bf16_hopper,
          nk_dots_pack_bf16_hopper, nk_angulars_packed_bf16_hopper);
    check("angulars_symmetric_bf16_hopper", test_angulars_symmetric<bf16_t, cuda_backend_t>,
          nk_angulars_symmetric_bf16_hopper);
    check("euclideans_packed_bf16_hopper", test_euclideans_packed<bf16_t, cuda_backend_t>,
          nk_dots_pack_size_bf16_hopper, nk_dots_pack_bf16_hopper, nk_euclideans_packed_bf16_hopper);
    check("euclideans_symmetric_bf16_hopper", test_euclideans_symmetric<bf16_t, cuda_backend_t>,
          nk_euclideans_symmetric_bf16_hopper);
    check("angulars_packed_f16_hopper", test_angulars_packed<f16_t, cuda_backend_t>, nk_dots_pack_size_f16_hopper,
          nk_dots_pack_f16_hopper, nk_angulars_packed_f16_hopper);
    check("angulars_symmetric_f16_hopper", test_angulars_symmetric<f16_t, cuda_backend_t>,
          nk_angulars_symmetric_f16_hopper);
    check("euclideans_packed_f16_hopper", test_euclideans_packed<f16_t, cuda_backend_t>, nk_dots_pack_size_f16_hopper,
          nk_dots_pack_f16_hopper, nk_euclideans_packed_f16_hopper);
    check("euclideans_symmetric_f16_hopper", test_euclideans_symmetric<f16_t, cuda_backend_t>,
          nk_euclideans_symmetric_f16_hopper);
    check("angulars_packed_e2m3_hopper", test_angulars_packed<e2m3_t, cuda_backend_t>, nk_dots_pack_size_e2m3_hopper,
          nk_dots_pack_e2m3_hopper, nk_angulars_packed_e2m3_hopper);
    check("angulars_symmetric_e2m3_hopper", test_angulars_symmetric<e2m3_t, cuda_backend_t>,
          nk_angulars_symmetric_e2m3_hopper);
    check("euclideans_packed_e2m3_hopper", test_euclideans_packed<e2m3_t, cuda_backend_t>,
          nk_dots_pack_size_e2m3_hopper, nk_dots_pack_e2m3_hopper, nk_euclideans_packed_e2m3_hopper);
    check("euclideans_symmetric_e2m3_hopper", test_euclideans_symmetric<e2m3_t, cuda_backend_t>,
          nk_euclideans_symmetric_e2m3_hopper);
    check("angulars_packed_e2m1_hopper", test_angulars_packed<e2m1x2_t, cuda_backend_t>, nk_dots_pack_size_e2m1_hopper,
          nk_dots_pack_e2m1_hopper, nk_angulars_packed_e2m1_hopper);
    check("angulars_symmetric_e2m1_hopper", test_angulars_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_angulars_symmetric_e2m1_hopper);
    check("euclideans_packed_e2m1_hopper", test_euclideans_packed<e2m1x2_t, cuda_backend_t>,
          nk_dots_pack_size_e2m1_hopper, nk_dots_pack_e2m1_hopper, nk_euclideans_packed_e2m1_hopper);
    check("euclideans_symmetric_e2m1_hopper", test_euclideans_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e2m1_hopper);
    check("angulars_packed_i8_hopper", test_angulars_packed<i8_t, cuda_backend_t>, nk_dots_pack_size_i8_hopper,
          nk_dots_pack_i8_hopper, nk_angulars_packed_i8_hopper);
    check("angulars_symmetric_i8_hopper", test_angulars_symmetric<i8_t, cuda_backend_t>,
          nk_angulars_symmetric_i8_hopper);
    check("euclideans_packed_i8_hopper", test_euclideans_packed<i8_t, cuda_backend_t>, nk_dots_pack_size_i8_hopper,
          nk_dots_pack_i8_hopper, nk_euclideans_packed_i8_hopper);
    check("euclideans_symmetric_i8_hopper", test_euclideans_symmetric<i8_t, cuda_backend_t>,
          nk_euclideans_symmetric_i8_hopper);
    check("angulars_packed_u8_hopper", test_angulars_packed<u8_t, cuda_backend_t>, nk_dots_pack_size_u8_hopper,
          nk_dots_pack_u8_hopper, nk_angulars_packed_u8_hopper);
    check("angulars_symmetric_u8_hopper", test_angulars_symmetric<u8_t, cuda_backend_t>,
          nk_angulars_symmetric_u8_hopper);
    check("euclideans_packed_u8_hopper", test_euclideans_packed<u8_t, cuda_backend_t>, nk_dots_pack_size_u8_hopper,
          nk_dots_pack_u8_hopper, nk_euclideans_packed_u8_hopper);
    check("euclideans_symmetric_u8_hopper", test_euclideans_symmetric<u8_t, cuda_backend_t>,
          nk_euclideans_symmetric_u8_hopper);
    check("angulars_packed_i4_hopper", test_angulars_packed<i4x2_t, cuda_backend_t>, nk_dots_pack_size_i4_hopper,
          nk_dots_pack_i4_hopper, nk_angulars_packed_i4_hopper);
    check("angulars_symmetric_i4_hopper", test_angulars_symmetric<i4x2_t, cuda_backend_t>,
          nk_angulars_symmetric_i4_hopper);
    check("euclideans_packed_i4_hopper", test_euclideans_packed<i4x2_t, cuda_backend_t>, nk_dots_pack_size_i4_hopper,
          nk_dots_pack_i4_hopper, nk_euclideans_packed_i4_hopper);
    check("euclideans_symmetric_i4_hopper", test_euclideans_symmetric<i4x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_i4_hopper);
    check("angulars_packed_u4_hopper", test_angulars_packed<u4x2_t, cuda_backend_t>, nk_dots_pack_size_u4_hopper,
          nk_dots_pack_u4_hopper, nk_angulars_packed_u4_hopper);
    check("angulars_symmetric_u4_hopper", test_angulars_symmetric<u4x2_t, cuda_backend_t>,
          nk_angulars_symmetric_u4_hopper);
    check("euclideans_packed_u4_hopper", test_euclideans_packed<u4x2_t, cuda_backend_t>, nk_dots_pack_size_u4_hopper,
          nk_dots_pack_u4_hopper, nk_euclideans_packed_u4_hopper);
    check("euclideans_symmetric_u4_hopper", test_euclideans_symmetric<u4x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_u4_hopper);
    check("attention_packed_bf16_hopper", test_attention_packed<bf16_t, cuda_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_bf16_hopper, nk_attention_pack_bf16_hopper, nk_attention_packed_bf16_hopper);
    check("attention_packed_gradients_bf16_hopper", test_attention_packed_gradients<bf16_t, cuda_backend_t>,
          nk_attention_pack_size_bf16_hopper, nk_attention_pack_bf16_hopper, nk_attention_packed_gradients_bf16_hopper);
    check("attention_packed_e4m3_hopper", test_attention_packed<e4m3_t, cuda_backend_t, attention_weights_t::bits_4_k>,
          nk_attention_pack_size_e4m3_hopper, nk_attention_pack_e4m3_hopper, nk_attention_packed_e4m3_hopper);
    check("attention_packed_i8_hopper", test_attention_packed<i8_t, cuda_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_i8_hopper, nk_attention_pack_i8_hopper, nk_attention_packed_i8_hopper);
#endif // NUMKONG_TARGET_HOPPER

#if NUMKONG_TARGET_BLACKWELL
    check.section("Cross Blackwell", nk_cap_blackwell_k);
    check("dots_packed_bf16_blackwell", test_dots_packed<bf16_t, cuda_backend_t>, nk_dots_pack_size_bf16_blackwell,
          nk_dots_pack_bf16_blackwell, nk_dots_packed_bf16_blackwell);
    check("dots_pack_bf16_blackwell",
          test_dots_pack_layout<bf16_t, cuda_backend_t, nk_dots_pack_size_bf16_blackwell,
                                nk_dots_packed_shape_bf16_blackwell, nk_dots_pack_bf16_blackwell>);
    check("dots_contract_bf16_blackwell",
          test_dots_launch_contract<bf16_t, cuda_backend_t, nk_dots_pack_size_bf16_blackwell,
                                    nk_dots_packed_bf16_blackwell, nk_dots_symmetric_bf16_blackwell>);
    check("dots_symmetric_bf16_blackwell", test_dots_symmetric<bf16_t, cuda_backend_t>,
          nk_dots_symmetric_bf16_blackwell);
    check("dots_packed_f16_blackwell", test_dots_packed<f16_t, cuda_backend_t>, nk_dots_pack_size_f16_blackwell,
          nk_dots_pack_f16_blackwell, nk_dots_packed_f16_blackwell);
    check("dots_pack_f16_blackwell",
          test_dots_pack_layout<f16_t, cuda_backend_t, nk_dots_pack_size_f16_blackwell,
                                nk_dots_packed_shape_f16_blackwell, nk_dots_pack_f16_blackwell>);
    check("dots_contract_f16_blackwell",
          test_dots_launch_contract<f16_t, cuda_backend_t, nk_dots_pack_size_f16_blackwell,
                                    nk_dots_packed_f16_blackwell, nk_dots_symmetric_f16_blackwell>);
    check("dots_symmetric_f16_blackwell", test_dots_symmetric<f16_t, cuda_backend_t>, nk_dots_symmetric_f16_blackwell);
    check("dots_packed_e5m2_blackwell", test_dots_packed<e5m2_t, cuda_backend_t>, nk_dots_pack_size_e5m2_blackwell,
          nk_dots_pack_e5m2_blackwell, nk_dots_packed_e5m2_blackwell);
    check("dots_pack_e5m2_blackwell",
          test_dots_pack_layout<e5m2_t, cuda_backend_t, nk_dots_pack_size_e5m2_blackwell,
                                nk_dots_packed_shape_e5m2_blackwell, nk_dots_pack_e5m2_blackwell>);
    check("dots_contract_e5m2_blackwell",
          test_dots_launch_contract<e5m2_t, cuda_backend_t, nk_dots_pack_size_e5m2_blackwell,
                                    nk_dots_packed_e5m2_blackwell, nk_dots_symmetric_e5m2_blackwell>);
    check("dots_symmetric_e5m2_blackwell", test_dots_symmetric<e5m2_t, cuda_backend_t>,
          nk_dots_symmetric_e5m2_blackwell);
    check("dots_packed_e4m3_blackwell", test_dots_packed<e4m3_t, cuda_backend_t>, nk_dots_pack_size_e4m3_blackwell,
          nk_dots_pack_e4m3_blackwell, nk_dots_packed_e4m3_blackwell);
    check("dots_pack_e4m3_blackwell",
          test_dots_pack_layout<e4m3_t, cuda_backend_t, nk_dots_pack_size_e4m3_blackwell,
                                nk_dots_packed_shape_e4m3_blackwell, nk_dots_pack_e4m3_blackwell>);
    check("dots_contract_e4m3_blackwell",
          test_dots_launch_contract<e4m3_t, cuda_backend_t, nk_dots_pack_size_e4m3_blackwell,
                                    nk_dots_packed_e4m3_blackwell, nk_dots_symmetric_e4m3_blackwell>);
    check("dots_symmetric_e4m3_blackwell", test_dots_symmetric<e4m3_t, cuda_backend_t>,
          nk_dots_symmetric_e4m3_blackwell);
    check("dots_packed_e3m2_blackwell", test_dots_packed<e3m2_t, cuda_backend_t>, nk_dots_pack_size_e3m2_blackwell,
          nk_dots_pack_e3m2_blackwell, nk_dots_packed_e3m2_blackwell);
    check("dots_pack_e3m2_blackwell",
          test_dots_pack_layout<e3m2_t, cuda_backend_t, nk_dots_pack_size_e3m2_blackwell,
                                nk_dots_packed_shape_e3m2_blackwell, nk_dots_pack_e3m2_blackwell>);
    check("dots_contract_e3m2_blackwell",
          test_dots_launch_contract<e3m2_t, cuda_backend_t, nk_dots_pack_size_e3m2_blackwell,
                                    nk_dots_packed_e3m2_blackwell, nk_dots_symmetric_e3m2_blackwell>);
    check("dots_symmetric_e3m2_blackwell", test_dots_symmetric<e3m2_t, cuda_backend_t>,
          nk_dots_symmetric_e3m2_blackwell);
    check("dots_packed_e2m3_blackwell", test_dots_packed<e2m3_t, cuda_backend_t>, nk_dots_pack_size_e2m3_blackwell,
          nk_dots_pack_e2m3_blackwell, nk_dots_packed_e2m3_blackwell);
    check("dots_pack_e2m3_blackwell",
          test_dots_pack_layout<e2m3_t, cuda_backend_t, nk_dots_pack_size_e2m3_blackwell,
                                nk_dots_packed_shape_e2m3_blackwell, nk_dots_pack_e2m3_blackwell>);
    check("dots_contract_e2m3_blackwell",
          test_dots_launch_contract<e2m3_t, cuda_backend_t, nk_dots_pack_size_e2m3_blackwell,
                                    nk_dots_packed_e2m3_blackwell, nk_dots_symmetric_e2m3_blackwell>);
    check("dots_symmetric_e2m3_blackwell", test_dots_symmetric<e2m3_t, cuda_backend_t>,
          nk_dots_symmetric_e2m3_blackwell);
    check("dots_packed_e2m1_blackwell", test_dots_packed<e2m1x2_t, cuda_backend_t>, nk_dots_pack_size_e2m1_blackwell,
          nk_dots_pack_e2m1_blackwell, nk_dots_packed_e2m1_blackwell);
    check("dots_pack_e2m1_blackwell",
          test_dots_pack_layout<e2m1x2_t, cuda_backend_t, nk_dots_pack_size_e2m1_blackwell,
                                nk_dots_packed_shape_e2m1_blackwell, nk_dots_pack_e2m1_blackwell>);
    check("dots_contract_e2m1_blackwell",
          test_dots_launch_contract<e2m1x2_t, cuda_backend_t, nk_dots_pack_size_e2m1_blackwell,
                                    nk_dots_packed_e2m1_blackwell, nk_dots_symmetric_e2m1_blackwell>);
    check("dots_symmetric_e2m1_blackwell", test_dots_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_dots_symmetric_e2m1_blackwell);
    check("dots_packed_nvfp4_blackwell", test_dots_packed<nvfp4_t, cuda_backend_t>, nk_dots_pack_size_nvfp4_blackwell,
          nk_dots_pack_nvfp4_blackwell, nk_dots_packed_nvfp4_blackwell);
    check("dots_pack_nvfp4_blackwell",
          test_dots_pack_layout<nvfp4_t, cuda_backend_t, nk_dots_pack_size_nvfp4_blackwell,
                                nk_dots_packed_shape_nvfp4_blackwell, nk_dots_pack_nvfp4_blackwell>);
    check("dots_contract_nvfp4_blackwell",
          test_dots_launch_contract<nvfp4_t, cuda_backend_t, nk_dots_pack_size_nvfp4_blackwell,
                                    nk_dots_packed_nvfp4_blackwell, nk_dots_symmetric_nvfp4_blackwell>);
    check("dots_symmetric_nvfp4_blackwell", test_dots_symmetric<nvfp4_t, cuda_backend_t>,
          nk_dots_symmetric_nvfp4_blackwell);
    check("dots_packed_mxfp4_blackwell", test_dots_packed<mxfp4_t, cuda_backend_t>, nk_dots_pack_size_mxfp4_blackwell,
          nk_dots_pack_mxfp4_blackwell, nk_dots_packed_mxfp4_blackwell);
    check("dots_pack_mxfp4_blackwell",
          test_dots_pack_layout<mxfp4_t, cuda_backend_t, nk_dots_pack_size_mxfp4_blackwell,
                                nk_dots_packed_shape_mxfp4_blackwell, nk_dots_pack_mxfp4_blackwell>);
    check("dots_contract_mxfp4_blackwell",
          test_dots_launch_contract<mxfp4_t, cuda_backend_t, nk_dots_pack_size_mxfp4_blackwell,
                                    nk_dots_packed_mxfp4_blackwell, nk_dots_symmetric_mxfp4_blackwell>);
    check("dots_symmetric_mxfp4_blackwell", test_dots_symmetric<mxfp4_t, cuda_backend_t>,
          nk_dots_symmetric_mxfp4_blackwell);
    check("dots_packed_mxfp8e4m3_blackwell", test_dots_packed<mxfp8e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e4m3_blackwell, nk_dots_pack_mxfp8e4m3_blackwell, nk_dots_packed_mxfp8e4m3_blackwell);
    check("dots_pack_mxfp8e4m3_blackwell",
          test_dots_pack_layout<mxfp8e4m3_t, cuda_backend_t, nk_dots_pack_size_mxfp8e4m3_blackwell,
                                nk_dots_packed_shape_mxfp8e4m3_blackwell, nk_dots_pack_mxfp8e4m3_blackwell>);
    check("dots_contract_mxfp8e4m3_blackwell",
          test_dots_launch_contract<mxfp8e4m3_t, cuda_backend_t, nk_dots_pack_size_mxfp8e4m3_blackwell,
                                    nk_dots_packed_mxfp8e4m3_blackwell, nk_dots_symmetric_mxfp8e4m3_blackwell>);
    check("dots_symmetric_mxfp8e4m3_blackwell", test_dots_symmetric<mxfp8e4m3_t, cuda_backend_t>,
          nk_dots_symmetric_mxfp8e4m3_blackwell);
    check("dots_packed_mxfp8e5m2_blackwell", test_dots_packed<mxfp8e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e5m2_blackwell, nk_dots_pack_mxfp8e5m2_blackwell, nk_dots_packed_mxfp8e5m2_blackwell);
    check("dots_pack_mxfp8e5m2_blackwell",
          test_dots_pack_layout<mxfp8e5m2_t, cuda_backend_t, nk_dots_pack_size_mxfp8e5m2_blackwell,
                                nk_dots_packed_shape_mxfp8e5m2_blackwell, nk_dots_pack_mxfp8e5m2_blackwell>);
    check("dots_contract_mxfp8e5m2_blackwell",
          test_dots_launch_contract<mxfp8e5m2_t, cuda_backend_t, nk_dots_pack_size_mxfp8e5m2_blackwell,
                                    nk_dots_packed_mxfp8e5m2_blackwell, nk_dots_symmetric_mxfp8e5m2_blackwell>);
    check("dots_symmetric_mxfp8e5m2_blackwell", test_dots_symmetric<mxfp8e5m2_t, cuda_backend_t>,
          nk_dots_symmetric_mxfp8e5m2_blackwell);
    check("angulars_packed_bf16_blackwell", test_angulars_packed<bf16_t, cuda_backend_t>,
          nk_dots_pack_size_bf16_blackwell, nk_dots_pack_bf16_blackwell, nk_angulars_packed_bf16_blackwell);
    check("angulars_symmetric_bf16_blackwell", test_angulars_symmetric<bf16_t, cuda_backend_t>,
          nk_angulars_symmetric_bf16_blackwell);
    check("euclideans_packed_bf16_blackwell", test_euclideans_packed<bf16_t, cuda_backend_t>,
          nk_dots_pack_size_bf16_blackwell, nk_dots_pack_bf16_blackwell, nk_euclideans_packed_bf16_blackwell);
    check("euclideans_symmetric_bf16_blackwell", test_euclideans_symmetric<bf16_t, cuda_backend_t>,
          nk_euclideans_symmetric_bf16_blackwell);
    check("angulars_packed_f16_blackwell", test_angulars_packed<f16_t, cuda_backend_t>, nk_dots_pack_size_f16_blackwell,
          nk_dots_pack_f16_blackwell, nk_angulars_packed_f16_blackwell);
    check("angulars_symmetric_f16_blackwell", test_angulars_symmetric<f16_t, cuda_backend_t>,
          nk_angulars_symmetric_f16_blackwell);
    check("euclideans_packed_f16_blackwell", test_euclideans_packed<f16_t, cuda_backend_t>,
          nk_dots_pack_size_f16_blackwell, nk_dots_pack_f16_blackwell, nk_euclideans_packed_f16_blackwell);
    check("euclideans_symmetric_f16_blackwell", test_euclideans_symmetric<f16_t, cuda_backend_t>,
          nk_euclideans_symmetric_f16_blackwell);
    check("angulars_packed_e5m2_blackwell", test_angulars_packed<e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_e5m2_blackwell, nk_dots_pack_e5m2_blackwell, nk_angulars_packed_e5m2_blackwell);
    check("angulars_symmetric_e5m2_blackwell", test_angulars_symmetric<e5m2_t, cuda_backend_t>,
          nk_angulars_symmetric_e5m2_blackwell);
    check("euclideans_packed_e5m2_blackwell", test_euclideans_packed<e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_e5m2_blackwell, nk_dots_pack_e5m2_blackwell, nk_euclideans_packed_e5m2_blackwell);
    check("euclideans_symmetric_e5m2_blackwell", test_euclideans_symmetric<e5m2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e5m2_blackwell);
    check("angulars_packed_e4m3_blackwell", test_angulars_packed<e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_e4m3_blackwell, nk_dots_pack_e4m3_blackwell, nk_angulars_packed_e4m3_blackwell);
    check("angulars_packed_nan_e4m3_blackwell",
          test_angulars_packed_nan_e4m3<nk_dots_pack_size_e4m3_blackwell, nk_dots_pack_e4m3_blackwell,
                                        nk_angulars_packed_e4m3_blackwell>);
    check("angulars_symmetric_e4m3_blackwell", test_angulars_symmetric<e4m3_t, cuda_backend_t>,
          nk_angulars_symmetric_e4m3_blackwell);
    check("euclideans_packed_e4m3_blackwell", test_euclideans_packed<e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_e4m3_blackwell, nk_dots_pack_e4m3_blackwell, nk_euclideans_packed_e4m3_blackwell);
    check("euclideans_symmetric_e4m3_blackwell", test_euclideans_symmetric<e4m3_t, cuda_backend_t>,
          nk_euclideans_symmetric_e4m3_blackwell);
    check("angulars_packed_e3m2_blackwell", test_angulars_packed<e3m2_t, cuda_backend_t>,
          nk_dots_pack_size_e3m2_blackwell, nk_dots_pack_e3m2_blackwell, nk_angulars_packed_e3m2_blackwell);
    check("angulars_symmetric_e3m2_blackwell", test_angulars_symmetric<e3m2_t, cuda_backend_t>,
          nk_angulars_symmetric_e3m2_blackwell);
    check("euclideans_packed_e3m2_blackwell", test_euclideans_packed<e3m2_t, cuda_backend_t>,
          nk_dots_pack_size_e3m2_blackwell, nk_dots_pack_e3m2_blackwell, nk_euclideans_packed_e3m2_blackwell);
    check("euclideans_symmetric_e3m2_blackwell", test_euclideans_symmetric<e3m2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e3m2_blackwell);
    check("angulars_packed_e2m3_blackwell", test_angulars_packed<e2m3_t, cuda_backend_t>,
          nk_dots_pack_size_e2m3_blackwell, nk_dots_pack_e2m3_blackwell, nk_angulars_packed_e2m3_blackwell);
    check("angulars_symmetric_e2m3_blackwell", test_angulars_symmetric<e2m3_t, cuda_backend_t>,
          nk_angulars_symmetric_e2m3_blackwell);
    check("euclideans_packed_e2m3_blackwell", test_euclideans_packed<e2m3_t, cuda_backend_t>,
          nk_dots_pack_size_e2m3_blackwell, nk_dots_pack_e2m3_blackwell, nk_euclideans_packed_e2m3_blackwell);
    check("euclideans_symmetric_e2m3_blackwell", test_euclideans_symmetric<e2m3_t, cuda_backend_t>,
          nk_euclideans_symmetric_e2m3_blackwell);
    check("angulars_packed_e2m1_blackwell", test_angulars_packed<e2m1x2_t, cuda_backend_t>,
          nk_dots_pack_size_e2m1_blackwell, nk_dots_pack_e2m1_blackwell, nk_angulars_packed_e2m1_blackwell);
    check("angulars_symmetric_e2m1_blackwell", test_angulars_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_angulars_symmetric_e2m1_blackwell);
    check("euclideans_packed_e2m1_blackwell", test_euclideans_packed<e2m1x2_t, cuda_backend_t>,
          nk_dots_pack_size_e2m1_blackwell, nk_dots_pack_e2m1_blackwell, nk_euclideans_packed_e2m1_blackwell);
    check("euclideans_symmetric_e2m1_blackwell", test_euclideans_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e2m1_blackwell);
    check("angulars_packed_nvfp4_blackwell", test_angulars_packed<nvfp4_t, cuda_backend_t>,
          nk_dots_pack_size_nvfp4_blackwell, nk_dots_pack_nvfp4_blackwell, nk_angulars_packed_nvfp4_blackwell);
    check("angulars_symmetric_nvfp4_blackwell", test_angulars_symmetric<nvfp4_t, cuda_backend_t>,
          nk_angulars_symmetric_nvfp4_blackwell);
    check("euclideans_packed_nvfp4_blackwell", test_euclideans_packed<nvfp4_t, cuda_backend_t>,
          nk_dots_pack_size_nvfp4_blackwell, nk_dots_pack_nvfp4_blackwell, nk_euclideans_packed_nvfp4_blackwell);
    check("euclideans_symmetric_nvfp4_blackwell", test_euclideans_symmetric<nvfp4_t, cuda_backend_t>,
          nk_euclideans_symmetric_nvfp4_blackwell);
    check("angulars_packed_mxfp4_blackwell", test_angulars_packed<mxfp4_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp4_blackwell, nk_dots_pack_mxfp4_blackwell, nk_angulars_packed_mxfp4_blackwell);
    check("angulars_symmetric_mxfp4_blackwell", test_angulars_symmetric<mxfp4_t, cuda_backend_t>,
          nk_angulars_symmetric_mxfp4_blackwell);
    check("euclideans_packed_mxfp4_blackwell", test_euclideans_packed<mxfp4_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp4_blackwell, nk_dots_pack_mxfp4_blackwell, nk_euclideans_packed_mxfp4_blackwell);
    check("euclideans_symmetric_mxfp4_blackwell", test_euclideans_symmetric<mxfp4_t, cuda_backend_t>,
          nk_euclideans_symmetric_mxfp4_blackwell);
    check("angulars_packed_mxfp8e4m3_blackwell", test_angulars_packed<mxfp8e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e4m3_blackwell, nk_dots_pack_mxfp8e4m3_blackwell,
          nk_angulars_packed_mxfp8e4m3_blackwell);
    check("angulars_symmetric_mxfp8e4m3_blackwell", test_angulars_symmetric<mxfp8e4m3_t, cuda_backend_t>,
          nk_angulars_symmetric_mxfp8e4m3_blackwell);
    check("euclideans_packed_mxfp8e4m3_blackwell", test_euclideans_packed<mxfp8e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e4m3_blackwell, nk_dots_pack_mxfp8e4m3_blackwell,
          nk_euclideans_packed_mxfp8e4m3_blackwell);
    check("euclideans_symmetric_mxfp8e4m3_blackwell", test_euclideans_symmetric<mxfp8e4m3_t, cuda_backend_t>,
          nk_euclideans_symmetric_mxfp8e4m3_blackwell);
    check("angulars_packed_mxfp8e5m2_blackwell", test_angulars_packed<mxfp8e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e5m2_blackwell, nk_dots_pack_mxfp8e5m2_blackwell,
          nk_angulars_packed_mxfp8e5m2_blackwell);
    check("angulars_symmetric_mxfp8e5m2_blackwell", test_angulars_symmetric<mxfp8e5m2_t, cuda_backend_t>,
          nk_angulars_symmetric_mxfp8e5m2_blackwell);
    check("euclideans_packed_mxfp8e5m2_blackwell", test_euclideans_packed<mxfp8e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_mxfp8e5m2_blackwell, nk_dots_pack_mxfp8e5m2_blackwell,
          nk_euclideans_packed_mxfp8e5m2_blackwell);
    check("euclideans_symmetric_mxfp8e5m2_blackwell", test_euclideans_symmetric<mxfp8e5m2_t, cuda_backend_t>,
          nk_euclideans_symmetric_mxfp8e5m2_blackwell);
    check("dots_packed_i8_blackwell", test_dots_packed<i8_t, cuda_backend_t>, nk_dots_pack_size_i8_blackwell,
          nk_dots_pack_i8_blackwell, nk_dots_packed_i8_blackwell);
    check("dots_pack_i8_blackwell",
          test_dots_pack_layout<i8_t, cuda_backend_t, nk_dots_pack_size_i8_blackwell, nk_dots_packed_shape_i8_blackwell,
                                nk_dots_pack_i8_blackwell>);
    check("dots_contract_i8_blackwell",
          test_dots_launch_contract<i8_t, cuda_backend_t, nk_dots_pack_size_i8_blackwell, nk_dots_packed_i8_blackwell,
                                    nk_dots_symmetric_i8_blackwell>);
    check("dots_symmetric_i8_blackwell", test_dots_symmetric<i8_t, cuda_backend_t>, nk_dots_symmetric_i8_blackwell);
    check("dots_packed_u8_blackwell", test_dots_packed<u8_t, cuda_backend_t>, nk_dots_pack_size_u8_blackwell,
          nk_dots_pack_u8_blackwell, nk_dots_packed_u8_blackwell);
    check("dots_pack_u8_blackwell",
          test_dots_pack_layout<u8_t, cuda_backend_t, nk_dots_pack_size_u8_blackwell, nk_dots_packed_shape_u8_blackwell,
                                nk_dots_pack_u8_blackwell>);
    check("dots_contract_u8_blackwell",
          test_dots_launch_contract<u8_t, cuda_backend_t, nk_dots_pack_size_u8_blackwell, nk_dots_packed_u8_blackwell,
                                    nk_dots_symmetric_u8_blackwell>);
    check("dots_symmetric_u8_blackwell", test_dots_symmetric<u8_t, cuda_backend_t>, nk_dots_symmetric_u8_blackwell);
    check("dots_packed_i4_blackwell", test_dots_packed<i4x2_t, cuda_backend_t>, nk_dots_pack_size_i4_blackwell,
          nk_dots_pack_i4_blackwell, nk_dots_packed_i4_blackwell);
    check("dots_pack_i4_blackwell",
          test_dots_pack_layout<i4x2_t, cuda_backend_t, nk_dots_pack_size_i4_blackwell,
                                nk_dots_packed_shape_i4_blackwell, nk_dots_pack_i4_blackwell>);
    check("dots_contract_i4_blackwell",
          test_dots_launch_contract<i4x2_t, cuda_backend_t, nk_dots_pack_size_i4_blackwell, nk_dots_packed_i4_blackwell,
                                    nk_dots_symmetric_i4_blackwell>);
    check("dots_symmetric_i4_blackwell", test_dots_symmetric<i4x2_t, cuda_backend_t>, nk_dots_symmetric_i4_blackwell);
    check("dots_packed_u4_blackwell", test_dots_packed<u4x2_t, cuda_backend_t>, nk_dots_pack_size_u4_blackwell,
          nk_dots_pack_u4_blackwell, nk_dots_packed_u4_blackwell);
    check("dots_pack_u4_blackwell",
          test_dots_pack_layout<u4x2_t, cuda_backend_t, nk_dots_pack_size_u4_blackwell,
                                nk_dots_packed_shape_u4_blackwell, nk_dots_pack_u4_blackwell>);
    check("dots_contract_u4_blackwell",
          test_dots_launch_contract<u4x2_t, cuda_backend_t, nk_dots_pack_size_u4_blackwell, nk_dots_packed_u4_blackwell,
                                    nk_dots_symmetric_u4_blackwell>);
    check("dots_symmetric_u4_blackwell", test_dots_symmetric<u4x2_t, cuda_backend_t>, nk_dots_symmetric_u4_blackwell);
    check("angulars_packed_i8_blackwell", test_angulars_packed<i8_t, cuda_backend_t>, nk_dots_pack_size_i8_blackwell,
          nk_dots_pack_i8_blackwell, nk_angulars_packed_i8_blackwell);
    check("angulars_symmetric_i8_blackwell", test_angulars_symmetric<i8_t, cuda_backend_t>,
          nk_angulars_symmetric_i8_blackwell);
    check("euclideans_packed_i8_blackwell", test_euclideans_packed<i8_t, cuda_backend_t>,
          nk_dots_pack_size_i8_blackwell, nk_dots_pack_i8_blackwell, nk_euclideans_packed_i8_blackwell);
    check("euclideans_symmetric_i8_blackwell", test_euclideans_symmetric<i8_t, cuda_backend_t>,
          nk_euclideans_symmetric_i8_blackwell);
    check("angulars_packed_u8_blackwell", test_angulars_packed<u8_t, cuda_backend_t>, nk_dots_pack_size_u8_blackwell,
          nk_dots_pack_u8_blackwell, nk_angulars_packed_u8_blackwell);
    check("angulars_symmetric_u8_blackwell", test_angulars_symmetric<u8_t, cuda_backend_t>,
          nk_angulars_symmetric_u8_blackwell);
    check("euclideans_packed_u8_blackwell", test_euclideans_packed<u8_t, cuda_backend_t>,
          nk_dots_pack_size_u8_blackwell, nk_dots_pack_u8_blackwell, nk_euclideans_packed_u8_blackwell);
    check("euclideans_symmetric_u8_blackwell", test_euclideans_symmetric<u8_t, cuda_backend_t>,
          nk_euclideans_symmetric_u8_blackwell);
    check("angulars_packed_i4_blackwell", test_angulars_packed<i4x2_t, cuda_backend_t>, nk_dots_pack_size_i4_blackwell,
          nk_dots_pack_i4_blackwell, nk_angulars_packed_i4_blackwell);
    check("angulars_symmetric_i4_blackwell", test_angulars_symmetric<i4x2_t, cuda_backend_t>,
          nk_angulars_symmetric_i4_blackwell);
    check("euclideans_packed_i4_blackwell", test_euclideans_packed<i4x2_t, cuda_backend_t>,
          nk_dots_pack_size_i4_blackwell, nk_dots_pack_i4_blackwell, nk_euclideans_packed_i4_blackwell);
    check("euclideans_symmetric_i4_blackwell", test_euclideans_symmetric<i4x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_i4_blackwell);
    check("angulars_packed_u4_blackwell", test_angulars_packed<u4x2_t, cuda_backend_t>, nk_dots_pack_size_u4_blackwell,
          nk_dots_pack_u4_blackwell, nk_angulars_packed_u4_blackwell);
    check("angulars_symmetric_u4_blackwell", test_angulars_symmetric<u4x2_t, cuda_backend_t>,
          nk_angulars_symmetric_u4_blackwell);
    check("euclideans_packed_u4_blackwell", test_euclideans_packed<u4x2_t, cuda_backend_t>,
          nk_dots_pack_size_u4_blackwell, nk_dots_pack_u4_blackwell, nk_euclideans_packed_u4_blackwell);
    check("euclideans_symmetric_u4_blackwell", test_euclideans_symmetric<u4x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_u4_blackwell);
    check("attention_packed_bf16_blackwell",
          test_attention_packed<bf16_t, cuda_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_bf16_blackwell, nk_attention_pack_bf16_blackwell, nk_attention_packed_bf16_blackwell);
    check("attention_packed_f16_blackwell",
          test_attention_packed<f16_t, cuda_backend_t, attention_weights_t::bits_11_k>,
          nk_attention_pack_size_f16_blackwell, nk_attention_pack_f16_blackwell, nk_attention_packed_f16_blackwell);
    check("attention_packed_gradients_bf16_blackwell", test_attention_packed_gradients<bf16_t, cuda_backend_t>,
          nk_attention_pack_size_bf16_blackwell, nk_attention_pack_bf16_blackwell,
          nk_attention_packed_gradients_bf16_blackwell);
    check("attention_packed_i8_blackwell", test_attention_packed<i8_t, cuda_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_i8_blackwell, nk_attention_pack_i8_blackwell, nk_attention_packed_i8_blackwell);
    check("attention_packed_e4m3_blackwell",
          test_attention_packed<e4m3_t, cuda_backend_t, attention_weights_t::bits_4_k>,
          nk_attention_pack_size_e4m3_blackwell, nk_attention_pack_e4m3_blackwell, nk_attention_packed_e4m3_blackwell);
#endif // NUMKONG_TARGET_BLACKWELL

#if NUMKONG_TARGET_BLACKWELLRTX
    check.section("Cross Blackwell RTX", nk_cap_blackwellrtx_k);
    check("dots_packed_e5m2_blackwellrtx", test_dots_packed<e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_e5m2_blackwellrtx, nk_dots_pack_e5m2_blackwellrtx, nk_dots_packed_e5m2_blackwellrtx);
    check("dots_pack_e5m2_blackwellrtx",
          test_dots_pack_layout<e5m2_t, cuda_backend_t, nk_dots_pack_size_e5m2_blackwellrtx,
                                nk_dots_packed_shape_e5m2_blackwellrtx, nk_dots_pack_e5m2_blackwellrtx>);
    check("dots_contract_e5m2_blackwellrtx",
          test_dots_launch_contract<e5m2_t, cuda_backend_t, nk_dots_pack_size_e5m2_blackwellrtx,
                                    nk_dots_packed_e5m2_blackwellrtx, nk_dots_symmetric_e5m2_blackwellrtx>);
    check("dots_symmetric_e5m2_blackwellrtx", test_dots_symmetric<e5m2_t, cuda_backend_t>,
          nk_dots_symmetric_e5m2_blackwellrtx);
    check("dots_packed_e4m3_blackwellrtx", test_dots_packed<e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_e4m3_blackwellrtx, nk_dots_pack_e4m3_blackwellrtx, nk_dots_packed_e4m3_blackwellrtx);
    check("dots_pack_e4m3_blackwellrtx",
          test_dots_pack_layout<e4m3_t, cuda_backend_t, nk_dots_pack_size_e4m3_blackwellrtx,
                                nk_dots_packed_shape_e4m3_blackwellrtx, nk_dots_pack_e4m3_blackwellrtx>);
    check("dots_contract_e4m3_blackwellrtx",
          test_dots_launch_contract<e4m3_t, cuda_backend_t, nk_dots_pack_size_e4m3_blackwellrtx,
                                    nk_dots_packed_e4m3_blackwellrtx, nk_dots_symmetric_e4m3_blackwellrtx>);
    check("dots_symmetric_e4m3_blackwellrtx", test_dots_symmetric<e4m3_t, cuda_backend_t>,
          nk_dots_symmetric_e4m3_blackwellrtx);
    check("dots_packed_e3m2_blackwellrtx", test_dots_packed<e3m2_t, cuda_backend_t>,
          nk_dots_pack_size_e3m2_blackwellrtx, nk_dots_pack_e3m2_blackwellrtx, nk_dots_packed_e3m2_blackwellrtx);
    check("dots_pack_e3m2_blackwellrtx",
          test_dots_pack_layout<e3m2_t, cuda_backend_t, nk_dots_pack_size_e3m2_blackwellrtx,
                                nk_dots_packed_shape_e3m2_blackwellrtx, nk_dots_pack_e3m2_blackwellrtx>);
    check("dots_contract_e3m2_blackwellrtx",
          test_dots_launch_contract<e3m2_t, cuda_backend_t, nk_dots_pack_size_e3m2_blackwellrtx,
                                    nk_dots_packed_e3m2_blackwellrtx, nk_dots_symmetric_e3m2_blackwellrtx>);
    check("dots_symmetric_e3m2_blackwellrtx", test_dots_symmetric<e3m2_t, cuda_backend_t>,
          nk_dots_symmetric_e3m2_blackwellrtx);
    check("dots_packed_e2m3_blackwellrtx", test_dots_packed<e2m3_t, cuda_backend_t>,
          nk_dots_pack_size_e2m3_blackwellrtx, nk_dots_pack_e2m3_blackwellrtx, nk_dots_packed_e2m3_blackwellrtx);
    check("dots_pack_e2m3_blackwellrtx",
          test_dots_pack_layout<e2m3_t, cuda_backend_t, nk_dots_pack_size_e2m3_blackwellrtx,
                                nk_dots_packed_shape_e2m3_blackwellrtx, nk_dots_pack_e2m3_blackwellrtx>);
    check("dots_contract_e2m3_blackwellrtx",
          test_dots_launch_contract<e2m3_t, cuda_backend_t, nk_dots_pack_size_e2m3_blackwellrtx,
                                    nk_dots_packed_e2m3_blackwellrtx, nk_dots_symmetric_e2m3_blackwellrtx>);
    check("dots_symmetric_e2m3_blackwellrtx", test_dots_symmetric<e2m3_t, cuda_backend_t>,
          nk_dots_symmetric_e2m3_blackwellrtx);
    check("dots_packed_e2m1_blackwellrtx", test_dots_packed<e2m1x2_t, cuda_backend_t>,
          nk_dots_pack_size_e2m1_blackwellrtx, nk_dots_pack_e2m1_blackwellrtx, nk_dots_packed_e2m1_blackwellrtx);
    check("dots_pack_e2m1_blackwellrtx",
          test_dots_pack_layout<e2m1x2_t, cuda_backend_t, nk_dots_pack_size_e2m1_blackwellrtx,
                                nk_dots_packed_shape_e2m1_blackwellrtx, nk_dots_pack_e2m1_blackwellrtx>);
    check("dots_contract_e2m1_blackwellrtx",
          test_dots_launch_contract<e2m1x2_t, cuda_backend_t, nk_dots_pack_size_e2m1_blackwellrtx,
                                    nk_dots_packed_e2m1_blackwellrtx, nk_dots_symmetric_e2m1_blackwellrtx>);
    check("dots_symmetric_e2m1_blackwellrtx", test_dots_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_dots_symmetric_e2m1_blackwellrtx);
    check("angulars_packed_e5m2_blackwellrtx", test_angulars_packed<e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_e5m2_blackwellrtx, nk_dots_pack_e5m2_blackwellrtx, nk_angulars_packed_e5m2_blackwellrtx);
    check("angulars_symmetric_e5m2_blackwellrtx", test_angulars_symmetric<e5m2_t, cuda_backend_t>,
          nk_angulars_symmetric_e5m2_blackwellrtx);
    check("euclideans_packed_e5m2_blackwellrtx", test_euclideans_packed<e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_e5m2_blackwellrtx, nk_dots_pack_e5m2_blackwellrtx, nk_euclideans_packed_e5m2_blackwellrtx);
    check("euclideans_symmetric_e5m2_blackwellrtx", test_euclideans_symmetric<e5m2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e5m2_blackwellrtx);
    check("angulars_packed_e4m3_blackwellrtx", test_angulars_packed<e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_e4m3_blackwellrtx, nk_dots_pack_e4m3_blackwellrtx, nk_angulars_packed_e4m3_blackwellrtx);
    check("angulars_packed_nan_e4m3_blackwellrtx",
          test_angulars_packed_nan_e4m3<nk_dots_pack_size_e4m3_blackwellrtx, nk_dots_pack_e4m3_blackwellrtx,
                                        nk_angulars_packed_e4m3_blackwellrtx>);
    check("angulars_symmetric_e4m3_blackwellrtx", test_angulars_symmetric<e4m3_t, cuda_backend_t>,
          nk_angulars_symmetric_e4m3_blackwellrtx);
    check("euclideans_packed_e4m3_blackwellrtx", test_euclideans_packed<e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_e4m3_blackwellrtx, nk_dots_pack_e4m3_blackwellrtx, nk_euclideans_packed_e4m3_blackwellrtx);
    check("euclideans_symmetric_e4m3_blackwellrtx", test_euclideans_symmetric<e4m3_t, cuda_backend_t>,
          nk_euclideans_symmetric_e4m3_blackwellrtx);
    check("angulars_packed_e3m2_blackwellrtx", test_angulars_packed<e3m2_t, cuda_backend_t>,
          nk_dots_pack_size_e3m2_blackwellrtx, nk_dots_pack_e3m2_blackwellrtx, nk_angulars_packed_e3m2_blackwellrtx);
    check("angulars_symmetric_e3m2_blackwellrtx", test_angulars_symmetric<e3m2_t, cuda_backend_t>,
          nk_angulars_symmetric_e3m2_blackwellrtx);
    check("euclideans_packed_e3m2_blackwellrtx", test_euclideans_packed<e3m2_t, cuda_backend_t>,
          nk_dots_pack_size_e3m2_blackwellrtx, nk_dots_pack_e3m2_blackwellrtx, nk_euclideans_packed_e3m2_blackwellrtx);
    check("euclideans_symmetric_e3m2_blackwellrtx", test_euclideans_symmetric<e3m2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e3m2_blackwellrtx);
    check("angulars_packed_e2m3_blackwellrtx", test_angulars_packed<e2m3_t, cuda_backend_t>,
          nk_dots_pack_size_e2m3_blackwellrtx, nk_dots_pack_e2m3_blackwellrtx, nk_angulars_packed_e2m3_blackwellrtx);
    check("angulars_symmetric_e2m3_blackwellrtx", test_angulars_symmetric<e2m3_t, cuda_backend_t>,
          nk_angulars_symmetric_e2m3_blackwellrtx);
    check("euclideans_packed_e2m3_blackwellrtx", test_euclideans_packed<e2m3_t, cuda_backend_t>,
          nk_dots_pack_size_e2m3_blackwellrtx, nk_dots_pack_e2m3_blackwellrtx, nk_euclideans_packed_e2m3_blackwellrtx);
    check("euclideans_symmetric_e2m3_blackwellrtx", test_euclideans_symmetric<e2m3_t, cuda_backend_t>,
          nk_euclideans_symmetric_e2m3_blackwellrtx);
    check("angulars_packed_e2m1_blackwellrtx", test_angulars_packed<e2m1x2_t, cuda_backend_t>,
          nk_dots_pack_size_e2m1_blackwellrtx, nk_dots_pack_e2m1_blackwellrtx, nk_angulars_packed_e2m1_blackwellrtx);
    check("angulars_symmetric_e2m1_blackwellrtx", test_angulars_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_angulars_symmetric_e2m1_blackwellrtx);
    check("euclideans_packed_e2m1_blackwellrtx", test_euclideans_packed<e2m1x2_t, cuda_backend_t>,
          nk_dots_pack_size_e2m1_blackwellrtx, nk_dots_pack_e2m1_blackwellrtx, nk_euclideans_packed_e2m1_blackwellrtx);
    check("euclideans_symmetric_e2m1_blackwellrtx", test_euclideans_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e2m1_blackwellrtx);
    check("attention_packed_e4m3_blackwellrtx",
          test_attention_packed<e4m3_t, cuda_backend_t, attention_weights_t::bits_4_k>,
          nk_attention_pack_size_e4m3_blackwellrtx, nk_attention_pack_e4m3_blackwellrtx,
          nk_attention_packed_e4m3_blackwellrtx);
#endif // NUMKONG_TARGET_BLACKWELLRTX

#if NUMKONG_TARGET_BLACKWELLULTRA
    check.section("Cross Blackwell Ultra", nk_cap_blackwellultra_k);
    check("attention_packed_bf16_blackwellultra",
          test_attention_packed<bf16_t, cuda_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_bf16_blackwellultra, nk_attention_pack_bf16_blackwellultra,
          nk_attention_packed_bf16_blackwellultra);
    check("attention_packed_f16_blackwellultra",
          test_attention_packed<f16_t, cuda_backend_t, attention_weights_t::bits_11_k>,
          nk_attention_pack_size_f16_blackwellultra, nk_attention_pack_f16_blackwellultra,
          nk_attention_packed_f16_blackwellultra);
    check("attention_packed_gradients_bf16_blackwellultra", test_attention_packed_gradients<bf16_t, cuda_backend_t>,
          nk_attention_pack_size_bf16_blackwellultra, nk_attention_pack_bf16_blackwellultra,
          nk_attention_packed_gradients_bf16_blackwellultra);
    check("attention_packed_i8_blackwellultra",
          test_attention_packed<i8_t, cuda_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_i8_blackwellultra, nk_attention_pack_i8_blackwellultra,
          nk_attention_packed_i8_blackwellultra);
    check("attention_packed_e4m3_blackwellultra",
          test_attention_packed<e4m3_t, cuda_backend_t, attention_weights_t::bits_4_k>,
          nk_attention_pack_size_e4m3_blackwellultra, nk_attention_pack_e4m3_blackwellultra,
          nk_attention_packed_e4m3_blackwellultra);
#endif // NUMKONG_TARGET_BLACKWELLULTRA

    test_cross_dispatch<cuda_backend_t>(check);
}

#else  // !NUMKONG_ARCH_CUDA_
void test_cross_cuda(error_stats_section_t &) {}
#endif // NUMKONG_ARCH_CUDA_

} // namespace ashvardanian::numkong::test
