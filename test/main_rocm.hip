/**
 *  @file test/main_rocm.hip
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief ROCm test: the capability report, the ROCm cross-kernel sections of `cross_rocm.hip`, and
 *      the dispatching entry points.
 *
 *  NUMKONG_ROCM_ARCHITECTURES picks the codes compiled. `NUMKONG_FILTER=<regex>` keeps only the
 *  matching sections and kernels. Without a device the test exits zero, so a build host with no GPU
 *  still passes, having compiled every kernel for each architecture.
 */
#include <cstring> // `std::strncmp`

#include "harness.hpp" // `error_stats_section_t`
#include "harness.cuh" // `rocm_backend_t`, `device_capabilities`
#include "each.hpp"    // `test_sum`, `test_rmsnorm`, `test_swiglu`

using namespace ashvardanian::numkong::test;

char const *volatile nk::test::nk_test_current_kernel_ = nullptr;

namespace ashvardanian::numkong::test {

/** Every ROCm capability this build compiled, from @c test/cross_rocm.hip. */
void test_cross_rocm(error_stats_section_t &check);

} // namespace ashvardanian::numkong::test

/** Element-wise operations on the ROCm baseline, over @c test/each.hpp. */
static void test_each_rocm(error_stats_section_t &check) {
    check.section("Elementwise Operations ROCm", nk_cap_rocm_k);
    check("each_swiglu_f32_rocm", test_swiglu<f32_t, rocm_backend_t>, nk_each_swiglu_f32_rocm);
    check("each_swiglu_bf16_rocm", test_swiglu<bf16_t, rocm_backend_t>, nk_each_swiglu_bf16_rocm);
    check("each_swiglu_e4m3_rocm", test_swiglu<e4m3_t, rocm_backend_t>, nk_each_swiglu_e4m3_rocm);
    check("each_sum_f32_rocm", test_sum<f32_t, rocm_backend_t>, nk_each_sum_f32_rocm);
    check("each_sum_f16_rocm", test_sum<f16_t, rocm_backend_t>, nk_each_sum_f16_rocm);
    check("each_sum_bf16_rocm", test_sum<bf16_t, rocm_backend_t>, nk_each_sum_bf16_rocm);
    check("each_rmsnorm_f32_rocm", test_rmsnorm<f32_t, rocm_backend_t>, nk_each_rmsnorm_f32_rocm);
    check("each_rmsnorm_bf16_rocm", test_rmsnorm<bf16_t, rocm_backend_t>, nk_each_rmsnorm_bf16_rocm);
    check("each_rmsnorm_e4m3_rocm", test_rmsnorm<e4m3_t, rocm_backend_t>, nk_each_rmsnorm_e4m3_rocm);
}

/** @c nk_rocm_capabilities_detected reports the ROCm baseline on every device, and the matrix-core
 *  capabilities the device's architecture runs. */
static error_stats_t test_rocm_capabilities(settings_t const &, int device) {
    error_stats_t stats(comparison_family_t::exact_k);
    hipDeviceProp_t properties {};
    stats.expect(hipGetDeviceProperties(&properties, device) == hipSuccess, "the device properties were unreadable");
    char const *const architecture = properties.gcnArchName;
    nk_capability_t expected = nk_cap_rocm_k;
    if (std::strncmp(architecture, "gfx950", 6) == 0) expected |= nk_cap_cdna4_k;
    if (std::strncmp(architecture, "gfx1250", 7) == 0 || std::strncmp(architecture, "gfx1251", 7) == 0)
        expected |= nk_cap_cdna5_k;
    nk_capability_t reported = 0;
    stats.expect(nk_rocm_capabilities_detected(static_cast<nk_size_t>(device), &reported));
    stats.expect(reported == expected, "the capabilities differ from the architecture's");
    return stats;
}

int main(int, char **argv) {
    environment_t const env {read_settings(argv[0]), probe_machine()};
    print(env.machine);
    print(env.settings);
    auto const devices = nk::device_t::count(nk::device_kind_t::rocm_k);
    if (!devices) {
        fmt::println("- ROCm: no device");
        return 0;
    }
    nk_capability_t const capabilities = device_capabilities<rocm_runtime_t>();
    char names[NUMKONG_CAPABILITIES_NAME_CAPACITY];
    nk_capabilities_name(capabilities, names, sizeof(names));
    fmt::println("- ROCm: {} devices, the first running {}", devices.value, names);

    error_stats_section_t check(env.settings, capabilities | nk_cap_serial_k);
    check.section("ROCm capabilities", nk_cap_serial_k); // the report itself is under test, so never gate on it
    check("gpu_capabilities_rocm", test_rocm_capabilities, 0);

    test_each_rocm(check);
    test_cross_rocm(check);

    int const passed = static_cast<int>(check.kernel_count - check.failure_count);
    int const failed = static_cast<int>(check.failure_count);
    fmt::println("\n{} passed, {} failed", passed, failed);
    return failed != 0 && env.settings.on_failure == on_failure_t::exit_k ? 1 : 0;
}
