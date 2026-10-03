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
 *  The @c NUMKONG_TARGET_* verdicts come from the probes in `CMakeLists.txt`, which every binding
 *  builds through or links the output of. Builds that pass none get them from `types.h`, which
 *  reads the compiler flags.
 */
#ifndef NUMKONG_DISPATCH_H
#define NUMKONG_DISPATCH_H

#include <numkong/numkong.h>

#ifdef __cplusplus
extern "C" {
#endif

/** One capability group's kernels of one dispatch point: a null slot 0, then a kernel per bit of
 *  @c capabilities, ascending by bit. */
typedef struct {
    nk_capability_t capabilities;
    nk_kernel_punned_t const *kernels;
} nk_capability_kernels_t;

/** The kernels of a group a dispatch point has none in: only the null slot 0. */
static nk_kernel_punned_t const nk_no_kernels_[1] = {NUMKONG_NULL};

/** The capability groups a binary may hold: its CPU's, and one per GPU vendor it was built for. */
typedef enum {
    nk_capability_group_cpu_k,
    nk_capability_group_cuda_k,
    nk_capability_group_rocm_k,
    nk_capability_group_metal_k,
    nk_capability_groups_k,
} nk_capability_group_t;

/** Every bit of @p x at or below its highest set one, or zero. */
NUMKONG_CONSTEXPR nk_u64_t nk_u64_smear_down_(nk_u64_t x) {
    x |= x >> 1, x |= x >> 2, x |= x >> 4, x |= x >> 8, x |= x >> 16, x |= x >> 32;
    return x;
}

/** The capability group @p capabilities describes: each GPU vendor's bits sit above the CPU's. */
NUMKONG_CONSTEXPR nk_capability_group_t nk_capability_group_of_(nk_capability_t capabilities) {
    return (nk_capability_group_t)((capabilities >= nk_cap_cuda_k) + (capabilities >= nk_cap_rocm_k) +
                                   (capabilities >= nk_cap_metal_k));
}

/** What each group runs in this process: the CPU's starts at serial and widens as the library
 *  loads, and the GPU groups keep every bit, as their masks come from per-device producers. */
extern nk_capability_t nk_capabilities_runnable_[nk_capability_groups_k];

/** The best capability @p capabilities shares with its group's kernels and runnable ones, or zero. */
NUMKONG_INLINE nk_capability_t nk_capability_pick_(nk_capability_t capabilities,
                                                   nk_capability_kernels_t const groups[nk_capability_groups_k]) {
    nk_capability_group_t const group_index = nk_capability_group_of_(capabilities);
    nk_u64_t const at_or_below = nk_u64_smear_down_(capabilities & groups[group_index].capabilities &
                                                    nk_capabilities_runnable_[group_index]);
    return at_or_below ^ (at_or_below >> 1);
}

/** The kernel of the best capability @p capabilities shares with its group's kernels and runnable
 *  ones, or null. */
NUMKONG_INLINE nk_kernel_punned_t nk_kernel_pick_(nk_capability_t capabilities,
                                                  nk_capability_kernels_t const groups[nk_capability_groups_k]) {
    nk_capability_group_t const group_index = nk_capability_group_of_(capabilities);
    nk_capability_kernels_t const *group = &groups[group_index];
    nk_u64_t const at_or_below = nk_u64_smear_down_(capabilities & group->capabilities &
                                                    nk_capabilities_runnable_[group_index]);
    return group->kernels[nk_u64_popcount_(group->capabilities & at_or_below)];
}

#ifdef __cplusplus
}
#endif

#endif // NUMKONG_DISPATCH_H
