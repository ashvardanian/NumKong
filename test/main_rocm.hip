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

using namespace ashvardanian::numkong::test;

test_config_t nk::test::global_config;
char const *volatile nk::test::nk_test_current_kernel_ = nullptr;

/** Every ROCm capability this build compiled, from @c test/cross_rocm.hip. */
void test_cross_rocm();

/** The dispatching entry points, from @c test/cross_simt.cuh. */
void test_cross_dispatch();

/** @c nk_rocm_capabilities_detected reports the ROCm baseline on every device, and the matrix-core
 *  capabilities the device's architecture runs. */
static error_stats_t test_rocm_capabilities(int device) {
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
    global_config.load_environment();
    global_config.program = argv[0];
    log_environment();
    fmt::println("- Seed: {}", global_config.seed);
    fmt::println("- Rerun one test: NUMKONG_SEED={} NUMKONG_FILTER='^<name>$' {}", global_config.seed, argv[0]);
    auto const devices = nk::device_t::count(nk::device_kind_t::rocm_k);
    if (!devices) {
        fmt::println("- ROCm: no device");
        return 0;
    }
    nk_capability_t const capabilities =
        nk::device_t::make(nk::device_kind_t::rocm_k, 0).value.capabilities_enabled().value;
    char names[NUMKONG_CAPABILITIES_NAME_CAPACITY];
    nk_capabilities_name(capabilities, names, sizeof(names));
    fmt::println("- ROCm: {} devices, the first running {}", devices.value, names);

    error_stats_section_t check(nk_cap_any_k);
    check.section("ROCm capabilities", nk_cap_rocm_k);
    check("gpu_capabilities_rocm", test_rocm_capabilities, 0);

    test_cross_rocm();
    test_cross_dispatch();

    int const passed = static_cast<int>(global_config.kernel_count - global_config.failure_count);
    int const failed = static_cast<int>(global_config.failure_count);
    fmt::println("\n{} passed, {} failed", passed, failed);
    return failed == 0 ? 0 : 1;
}
