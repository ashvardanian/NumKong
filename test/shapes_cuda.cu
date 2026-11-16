/**
 *  @file test/shapes_cuda.cu
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief Chamfer, Hausdorff and RMSD exports over each vendor's baseline capability, on CUDA and,
 *      through an include CMake generates, on ROCm.
 */
#include "cross_device.hpp" // `test_rmsds_packed`, `test_shapes_flips`

namespace ashvardanian::numkong::test {

#if NUMKONG_ARCH_ROCM_
inline constexpr nk_capability_t shapes_capability_k = nk_cap_rocm_k;
#elif NUMKONG_ARCH_CUDA_
inline constexpr nk_capability_t shapes_capability_k = nk_cap_cuda_k;
#endif // NUMKONG_ARCH_ROCM_

#if NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_

/** The dispatch point @p best_ in the shape of its capability kernels, over this vendor's baseline
 *  capability alone. */
template <auto best_>
inline constexpr auto shapes_best =
    [](auto... arguments) noexcept { return call_best<best_>(shapes_capability_k, arguments...); };

#endif // NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_

#if NUMKONG_ARCH_ROCM_

void test_shapes_rocm(error_stats_section_t &check) {
    check.section("Shapes ROCm", nk_cap_rocm_k);
    check("rmsds_packed_f16_rocm", test_rmsds_packed<f16_t, rocm_backend_t>, shapes_best<nk_rmsds_packed_f16_best>);
    check("hausdorffs_packed_f16_rocm", test_shapes_flips<f16_t, nk_kernel_hausdorffs_packed_k, rocm_backend_t>,
          shapes_best<nk_hausdorffs_packed_f16_best>);
    check("chamfers_packed_f16_rocm", test_shapes_flips<f16_t, nk_kernel_chamfers_packed_k, rocm_backend_t>,
          shapes_best<nk_chamfers_packed_f16_best>);
}

#elif NUMKONG_ARCH_CUDA_

void test_shapes_cuda(error_stats_section_t &check) {
    check.section("Shapes CUDA", nk_cap_cuda_k);
    check("rmsds_packed_f16_cuda", test_rmsds_packed<f16_t, cuda_backend_t>, shapes_best<nk_rmsds_packed_f16_best>);
    check("hausdorffs_packed_f16_cuda", test_shapes_flips<f16_t, nk_kernel_hausdorffs_packed_k, cuda_backend_t>,
          shapes_best<nk_hausdorffs_packed_f16_best>);
    check("chamfers_packed_f16_cuda", test_shapes_flips<f16_t, nk_kernel_chamfers_packed_k, cuda_backend_t>,
          shapes_best<nk_chamfers_packed_f16_best>);
}

#else  // !NUMKONG_ARCH_ROCM_ && !NUMKONG_ARCH_CUDA_
void test_shapes_cuda(error_stats_section_t &) {}
#endif // NUMKONG_ARCH_ROCM_

} // namespace ashvardanian::numkong::test
