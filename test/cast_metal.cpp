/**
 *  @file test/cast_metal.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief Conversion tests on Metal.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include "harness.hpp"

#if NUMKONG_ARCH_METAL_
#include "cast.hpp"
#endif // NUMKONG_ARCH_METAL_

namespace ashvardanian::numkong::test {

#if NUMKONG_ARCH_METAL_
void run_cast_metal_cases() {
#if NUMKONG_TARGET_METAL
    auto const metal = nk_cap_metal_k;
    run_kernel_case("test_cast_pairs<metal_backend_t>(nk_cast_metal)", test_cast_pairs<metal_backend_t>, metal,
                    nk_cast_metal);
    run_kernel_case("test_cast_block_scaled<metal_backend_t>(nk_cast_metal)", test_cast_block_scaled<metal_backend_t>,
                    metal, nk_cast_metal);
#endif // NUMKONG_TARGET_METAL
}

#else  // !NUMKONG_ARCH_METAL_
void run_cast_metal_cases() {}
#endif // NUMKONG_ARCH_METAL_

} // namespace ashvardanian::numkong::test
