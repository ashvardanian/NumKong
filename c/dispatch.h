/**
 *  @file c/dispatch.h
 *  @author Ash Vardanian
 *  @date March 13, 2024
 *  @brief The capability lists' shape and the kernel selection every dispatch unit shares.
 *
 *  @c NUMKONG_NATIVE_F16 and @c NUMKONG_NATIVE_BF16 are deliberately not pinned here. These are the
 *  only translation units that define @c nk_f16_sqrt_best and friends, so pinning them would make
 *  the library disagree with the bindings that call it — the sources under `python/` do not include
 *  this header and honour the build system instead. `types.h` defaults both to 0 for everyone.
 *
 *  The @c NUMKONG_TARGET_* verdicts come from the build: the probes in `CMakeLists.txt`, `setup.py`,
 *  `build.rs` and `binding.gyp`, or the fixed table in `Package.swift`. Builds that pass none, like
 *  Go's, get them from the compiler flags in `types.h`.
 */
#ifndef NUMKONG_DISPATCH_H
#define NUMKONG_DISPATCH_H

#include <numkong/numkong.h>

#ifdef __cplusplus
extern "C" {
#endif

/** One capability group's kernels of one dispatch point: a kernel per bit of @c capabilities,
 *  ascending by bit. */
typedef struct {
    nk_capability_t capabilities;
    nk_kernel_punned_t const *kernels;
} nk_capability_kernels_t;

/** The capability groups a binary may hold: its CPU's, and one per GPU vendor it was built for. */
typedef enum {
    nk_capability_group_cpu_k,
    nk_capability_group_nvidia_k,
    nk_capability_group_amd_k,
    nk_capability_group_apple_k,
    nk_capability_groups_k,
} nk_capability_group_t;

/** The highest set bit of @p x as a mask, or zero; compilers lower it to one @c clz. */
NUMKONG_CONSTEXPR nk_u64_t nk_u64_highest_bit_(nk_u64_t x) {
    x |= x >> 1, x |= x >> 2, x |= x >> 4, x |= x >> 8, x |= x >> 16, x |= x >> 32;
    return x ^ (x >> 1);
}

/** The capability group @p capabilities describes, by its highest bit; zero is the CPU's. */
NUMKONG_CONSTEXPR nk_capability_group_t nk_capability_group_of_(nk_capability_t capabilities) {
    nk_u64_t const top = nk_u64_highest_bit_(capabilities);
    return (nk_capability_group_t)((top >= nk_cap_cuda_k) + (top >= nk_cap_rocm_k) + (top >= nk_cap_metal_k));
}

/** The best capability @p capabilities shares with its group's kernels, or zero if none. */
NUMKONG_CONSTEXPR nk_capability_t nk_capability_pick_(nk_capability_t capabilities,
                                                      nk_capability_kernels_t const groups[nk_capability_groups_k]) {
    return nk_u64_highest_bit_(capabilities & groups[nk_capability_group_of_(capabilities)].capabilities);
}

/** The kernel of the best capability @p capabilities shares with its group's kernels, or null. */
NUMKONG_CONSTEXPR nk_kernel_punned_t nk_kernel_pick_(nk_capability_t capabilities,
                                                     nk_capability_kernels_t const groups[nk_capability_groups_k]) {
    nk_capability_kernels_t const *group = &groups[nk_capability_group_of_(capabilities)];
    nk_capability_t const capability = nk_u64_highest_bit_(capabilities & group->capabilities);
    return capability ? group->kernels[nk_u64_popcount_(group->capabilities & (capability - 1))]
                      : (nk_kernel_punned_t)NUMKONG_NULL;
}

#ifdef __cplusplus
}
#endif

#endif // NUMKONG_DISPATCH_H
