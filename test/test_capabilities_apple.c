/**
 *  @brief Apple ARM64 capability probing when sysctl fails or reports an unavailable feature.
 *  @file test/test_capabilities_apple.c
 */
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/sysctl.h>

static int baseline_error_ = 0;
static unsigned baseline_support_ = 0;
static unsigned half_support_ = 0;

static int mock_sysctlbyname_(char const *name, void *output, size_t *output_size, void *new_value, size_t new_size) {
    (void)new_value;
    (void)new_size;
    if (strcmp(name, "hw.optional.neon") == 0) {
        if (baseline_error_) {
            errno = baseline_error_;
            return -1;
        }
        *(unsigned *)output = baseline_support_;
        *output_size = sizeof(unsigned);
        return 0;
    }
    if (strcmp(name, "hw.optional.arm.FEAT_FP16") == 0 && half_support_) {
        *(unsigned *)output = half_support_;
        *output_size = sizeof(unsigned);
        return 0;
    }
    errno = ENOENT;
    return -1;
}

#define sysctlbyname mock_sysctlbyname_
#include "numkong/capabilities.h"
#undef sysctlbyname

int main(void) {
    struct probe_case_t {
        int error;
        unsigned baseline_support;
        unsigned half_support;
        nk_capability_t expected;
    } const cases[] = {
        {EPERM, 0, 0, nk_cap_serial_k | nk_cap_neon_k},
        {ENOENT, 0, 0, nk_cap_serial_k | nk_cap_neon_k},
        {0, 0, 0, nk_cap_serial_k},
        {0, 1, 0, nk_cap_serial_k | nk_cap_neon_k},
        {EPERM, 0, 1, nk_cap_serial_k | nk_cap_neon_k | nk_cap_neonhalf_k},
        {0, 0, 1, nk_cap_serial_k},
    };
    for (size_t case_index = 0; case_index < sizeof(cases) / sizeof(cases[0]); ++case_index) {
        baseline_error_ = cases[case_index].error;
        baseline_support_ = cases[case_index].baseline_support;
        half_support_ = cases[case_index].half_support;
        nk_capability_t actual = nk_capabilities_arm64_();
        if (actual != cases[case_index].expected) {
            fprintf(stderr, "Case %zu: expected capabilities %llx, got %llx\n", case_index,
                    (unsigned long long)cases[case_index].expected, (unsigned long long)actual);
            return 1;
        }
    }
    return 0;
}
