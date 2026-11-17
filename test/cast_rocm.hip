/**
 *  @file test/cast_rocm.hip
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief Conversion tests on ROCm.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include "harness.hpp"

#if NUMKONG_ARCH_ROCM_
#include "cast.hpp"
#endif // NUMKONG_ARCH_ROCM_

namespace ashvardanian::numkong::test {

#if NUMKONG_ARCH_ROCM_
void run_cast_rocm_cases() {
    auto const rocm = nk_cap_rocm_k;
    run_kernel_case("test_cast_pairs<rocm_backend_t>(nk_cast_rocm)", test_cast_pairs<rocm_backend_t>, rocm,
                    nk_cast_rocm);
    run_kernel_case("test_cast_block_scaled<rocm_backend_t>(nk_cast_rocm)", test_cast_block_scaled<rocm_backend_t>,
                    rocm, nk_cast_rocm);
}

#else  // !NUMKONG_ARCH_ROCM_
void run_cast_rocm_cases() {}
#endif // NUMKONG_ARCH_ROCM_

} // namespace ashvardanian::numkong::test
