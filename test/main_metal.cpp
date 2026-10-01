/**
 *  @file test/main_metal.cpp
 *  @author Ash Vardanian
 *  @date September 24, 2026
 *  @brief Metal test: the capability report, the cross-kernel sections of `cross_metal.cpp`, and
 *      the dispatching entry points.
 *
 *  Every section runs on the queue @c main opens. `NUMKONG_FILTER=<regex>` keeps only the matching
 *  sections and kernels.
 */
#include "numkong/metal.h" // `nk_metal_queue_t`

#include "harness.hpp" // `error_stats_section_t`

using namespace ashvardanian::numkong::test;

char const *volatile nk::test::nk_test_current_kernel_ = nullptr;

namespace ashvardanian::numkong::test {

/** Every Metal capability this build compiled, from @c test/cross_metal.cpp. */
void test_cross_metal(error_stats_section_t &check, nk_metal_queue_t &queue);

/** The dispatching entry points, from @c test/cross_metal.cpp. */
void test_cross_dispatch(error_stats_section_t &check, nk_metal_queue_t &queue);

} // namespace ashvardanian::numkong::test

/** @c nk_metal_capabilities_detected reports the Metal baseline, and Apple GPU families only as a
 *  cumulative ladder, since every Apple10 device also runs Apple9. */
static error_stats_t test_metal_capabilities(settings_t const &, nk_size_t device) {
    error_stats_t stats(comparison_family_t::exact_k);
    nk_capability_t reported = 0;
    stats.expect(nk_metal_capabilities_detected(device, &reported));
    stats.expect((reported & nk_cap_metal_k) != 0, "the Metal baseline is missing");
    stats.expect((reported & ~(nk_cap_metal_k | nk_cap_apple9_k | nk_cap_apple10_k)) == 0,
                 "a capability of another vendor is reported");
    stats.expect(!(reported & nk_cap_apple10_k) || (reported & nk_cap_apple9_k), "Apple10 is reported without Apple9");
    return stats;
}

int main(int, char **argv) {
    environment_t const env {read_settings(argv[0]), probe_machine()};
    print(env.machine);
    print(env.settings);
    auto const devices = nk::device_t::count(nk::device_kind_t::metal_k);
    if (!devices) {
        fmt::println("- Metal: no device");
        return 0;
    }
    auto const device = nk::device_t::make(nk::device_kind_t::metal_k, 0);
    nk_capability_t const capabilities = device ? device.value.capabilities_enabled().value : 0;
    char names[NUMKONG_CAPABILITIES_NAME_CAPACITY];
    nk_capabilities_name(capabilities, names, sizeof(names));
    fmt::println("- Metal: {} devices, the first running {}", devices.value, names);
    nk_metal_queue_t queue;
    if (nk_metal_queue_init(&queue, 0) != nk_success_k) {
        fmt::println("- Metal: the first device refused a queue");
        return 1;
    }

    error_stats_section_t check(env.settings, capabilities | nk_cap_serial_k);
    check.section("Metal capabilities", nk_cap_serial_k); // the report itself is under test, so never gate on it
    check("gpu_capabilities_metal", test_metal_capabilities, 0);

    test_cross_metal(check, queue);
    test_cross_dispatch(check, queue);
    nk_metal_queue_free(&queue);

    int const passed = static_cast<int>(check.kernel_count - check.failure_count);
    int const failed = static_cast<int>(check.failure_count);
    fmt::println("\n{} passed, {} failed", passed, failed);
    return failed != 0 && env.settings.on_failure == on_failure_t::exit_k ? 1 : 0;
}
