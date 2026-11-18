/**
 *  @file test/shapes_cuda.cu
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief Chamfer, Hausdorff and RMSD exports over each vendor's baseline capability, on CUDA and,
 *      through an include CMake generates, on ROCm.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include "cross_device.hpp" // `test_rmsds_packed`, `test_shapes_flips`

namespace ashvardanian::numkong::test {

#if NUMKONG_ARCH_CUDA_
static constexpr nk_capability_t shapes_capability_k = nk_cap_cuda_k;
#endif
#if NUMKONG_ARCH_ROCM_
static constexpr nk_capability_t shapes_capability_k = nk_cap_rocm_k;
#endif

#if NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_

/** The dispatch point @p best_ in the shape of its capability kernels, over this vendor's baseline
 *  capability alone. */
template <auto best_>
static constexpr auto shapes_best =
    [](auto... arguments) noexcept { return call_best<best_>(shapes_capability_k, arguments...); };

#endif // NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_

#if NUMKONG_ARCH_ROCM_

void run_shapes_rocm_cases() {
    auto const rocm = nk_cap_rocm_k;
    run_kernel_case("test_rmsds_packed<f16_t, rocm_backend_t>(shapes_best<nk_rmsds_packed_f16_best>)",
                    test_rmsds_packed<f16_t, rocm_backend_t>, rocm, shapes_best<nk_rmsds_packed_f16_best>);
    run_kernel_case("test_shapes_flips<f16_t, nk_kernel_hausdorffs_packed_k, rocm_backend_t>",
                    test_shapes_flips<f16_t, nk_kernel_hausdorffs_packed_k, rocm_backend_t>, rocm,
                    shapes_best<nk_hausdorffs_packed_f16_best>);
    run_kernel_case("test_shapes_flips<f16_t, nk_kernel_chamfers_packed_k, rocm_backend_t>",
                    test_shapes_flips<f16_t, nk_kernel_chamfers_packed_k, rocm_backend_t>, rocm,
                    shapes_best<nk_chamfers_packed_f16_best>);
}

#endif // NUMKONG_ARCH_ROCM_
#if NUMKONG_ARCH_CUDA_

void run_shapes_cuda_cases() {
    auto const cuda = nk_cap_cuda_k;
    run_kernel_case("test_rmsds_packed<f16_t, cuda_backend_t>(shapes_best<nk_rmsds_packed_f16_best>)",
                    test_rmsds_packed<f16_t, cuda_backend_t>, cuda, shapes_best<nk_rmsds_packed_f16_best>);
    run_kernel_case("test_shapes_flips<f16_t, nk_kernel_hausdorffs_packed_k, cuda_backend_t>",
                    test_shapes_flips<f16_t, nk_kernel_hausdorffs_packed_k, cuda_backend_t>, cuda,
                    shapes_best<nk_hausdorffs_packed_f16_best>);
    run_kernel_case("test_shapes_flips<f16_t, nk_kernel_chamfers_packed_k, cuda_backend_t>",
                    test_shapes_flips<f16_t, nk_kernel_chamfers_packed_k, cuda_backend_t>, cuda,
                    shapes_best<nk_chamfers_packed_f16_best>);
}

#endif // NUMKONG_ARCH_CUDA_
#if !(NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_)
void run_shapes_cuda_cases() {}
#endif // !(NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_)

} // namespace ashvardanian::numkong::test
