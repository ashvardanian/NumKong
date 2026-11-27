/**
 *  @file src/dispatch/cast.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Dispatch points of the type conversions: bulk, block-scaled and scalar casts.
 */
#include "dispatch.h"

#include "numkong/cast/serial.h"

static nk_capability_kernels_t const *nk_cast_capabilities(void) {
    enum {
        best_k = NUMKONG_MULTIVERSIONED       ? 0
                 : NUMKONG_TARGET_POWERVSX    ? nk_cap_powervsx_index_k
                 : NUMKONG_TARGET_V128RELAXED ? nk_cap_v128relaxed_index_k
                 : NUMKONG_TARGET_RVV         ? nk_cap_rvv_index_k
                 : NUMKONG_TARGET_SAPPHIRE    ? nk_cap_sapphire_index_k
                 : NUMKONG_TARGET_ICELAKE     ? nk_cap_icelake_index_k
                 : NUMKONG_TARGET_SKYLAKE     ? nk_cap_skylake_index_k
                 : NUMKONG_TARGET_HASWELL     ? nk_cap_haswell_index_k
                 : NUMKONG_TARGET_NEON        ? nk_cap_neon_index_k
                                              : nk_cap_serial_index_k,
    };
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        best_k > nk_cap_serial_index_k ? NUMKONG_NULL : (nk_kernel_punned_t)&nk_cast_serial,
#if NUMKONG_TARGET_NEON
        best_k > nk_cap_neon_index_k ? NUMKONG_NULL : (nk_kernel_punned_t)&nk_cast_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        best_k > nk_cap_haswell_index_k ? NUMKONG_NULL : (nk_kernel_punned_t)&nk_cast_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        best_k > nk_cap_skylake_index_k ? NUMKONG_NULL : (nk_kernel_punned_t)&nk_cast_skylake,
#endif
#if NUMKONG_TARGET_ICELAKE
        best_k > nk_cap_icelake_index_k ? NUMKONG_NULL : (nk_kernel_punned_t)&nk_cast_icelake,
#endif
#if NUMKONG_TARGET_SAPPHIRE
        best_k > nk_cap_sapphire_index_k ? NUMKONG_NULL : (nk_kernel_punned_t)&nk_cast_sapphire,
#endif
#if NUMKONG_TARGET_RVV
        best_k > nk_cap_rvv_index_k ? NUMKONG_NULL : (nk_kernel_punned_t)&nk_cast_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        best_k > nk_cap_v128relaxed_index_k ? NUMKONG_NULL : (nk_kernel_punned_t)&nk_cast_v128relaxed,
#endif
#if NUMKONG_TARGET_POWERVSX
        best_k > nk_cap_powervsx_index_k ? NUMKONG_NULL : (nk_kernel_punned_t)&nk_cast_powervsx,
#endif
    };
    enum {
        cuda_best_k = NUMKONG_MULTIVERSIONED  ? 0
                      : NUMKONG_TARGET_ADA    ? nk_cap_ada_index_k
                      : NUMKONG_TARGET_AMPERE ? nk_cap_ampere_index_k
                      : NUMKONG_TARGET_CUDA   ? nk_cap_cuda_index_k
                                              : 0,
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        cuda_best_k > nk_cap_cuda_index_k ? NUMKONG_NULL : (nk_kernel_punned_t)&nk_cast_cuda,
#endif
#if NUMKONG_TARGET_AMPERE
        cuda_best_k > nk_cap_ampere_index_k ? NUMKONG_NULL : (nk_kernel_punned_t)&nk_cast_ampere,
#endif
#if NUMKONG_TARGET_ADA
        cuda_best_k > nk_cap_ada_index_k ? NUMKONG_NULL : (nk_kernel_punned_t)&nk_cast_ada,
#endif
    };
    static nk_kernel_punned_t const rocm[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_ROCM
        (nk_kernel_punned_t)&nk_cast_rocm,
#endif
    };
    static nk_kernel_punned_t const metal[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_METAL
        (nk_kernel_punned_t)&nk_cast_metal,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_sapphire_k * NUMKONG_TARGET_SAPPHIRE | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED | nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE |
             nk_cap_ada_k * NUMKONG_TARGET_ADA,
         cuda},
        {nk_cap_rocm_k * NUMKONG_TARGET_ROCM, rocm},
        {nk_cap_metal_k * NUMKONG_TARGET_METAL, metal},
    };
    return lists;
}

static nk_capability_kernels_t const *nk_f16_to_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_f16_to_f32_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_f16_to_f32_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_f16_to_f32_haswell,
#endif
#if NUMKONG_TARGET_SAPPHIRE
        (nk_kernel_punned_t)&nk_f16_to_f32_sapphire,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_f16_to_f32_powervsx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_sapphire_k * NUMKONG_TARGET_SAPPHIRE | nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_f16_to_f32_best(nk_f16_t const *source, nk_f32_t *destination, nk_capability_t capabilities) {
    ((nk_f16_to_f32_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                             nk_f16_to_f32_capabilities()))(source, destination);
}

static nk_capability_kernels_t const *nk_bf16_to_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_bf16_to_f32_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_bf16_to_f32_best(nk_bf16_t const *source, nk_f32_t *destination, nk_capability_t capabilities) {
    ((nk_bf16_to_f32_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                              nk_bf16_to_f32_capabilities()))(source, destination);
}

static nk_capability_kernels_t const *nk_e4m3_to_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_e4m3_to_f32_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_e4m3_to_f32_best(nk_e4m3_t const *source, nk_f32_t *destination, nk_capability_t capabilities) {
    ((nk_u8_to_f32_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                            nk_e4m3_to_f32_capabilities()))(source, destination);
}

static nk_capability_kernels_t const *nk_e5m2_to_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_e5m2_to_f32_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_e5m2_to_f32_best(nk_e5m2_t const *source, nk_f32_t *destination, nk_capability_t capabilities) {
    ((nk_u8_to_f32_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                            nk_e5m2_to_f32_capabilities()))(source, destination);
}

static nk_capability_kernels_t const *nk_e2m3_to_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_e2m3_to_f32_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_e2m3_to_f32_best(nk_e2m3_t const *source, nk_f32_t *destination, nk_capability_t capabilities) {
    ((nk_u8_to_f32_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                            nk_e2m3_to_f32_capabilities()))(source, destination);
}

static nk_capability_kernels_t const *nk_e3m2_to_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_e3m2_to_f32_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_e3m2_to_f32_best(nk_e3m2_t const *source, nk_f32_t *destination, nk_capability_t capabilities) {
    ((nk_u8_to_f32_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                            nk_e3m2_to_f32_capabilities()))(source, destination);
}

static nk_capability_kernels_t const *nk_f32_to_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_f32_to_f16_serial,
#if NUMKONG_TARGET_NEON
        (nk_kernel_punned_t)&nk_f32_to_f16_neon,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_f32_to_f16_haswell,
#endif
#if NUMKONG_TARGET_SAPPHIRE
        (nk_kernel_punned_t)&nk_f32_to_f16_sapphire,
#endif
#if NUMKONG_TARGET_POWERVSX
        (nk_kernel_punned_t)&nk_f32_to_f16_powervsx,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neon_k * NUMKONG_TARGET_NEON | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL |
             nk_cap_sapphire_k * NUMKONG_TARGET_SAPPHIRE | nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_f32_to_f16_best(nk_f32_t const *source, nk_f16_t *destination, nk_capability_t capabilities) {
    ((nk_f32_to_f16_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                             nk_f32_to_f16_capabilities()))(source, destination);
}

static nk_capability_kernels_t const *nk_f32_to_bf16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_f32_to_bf16_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_f32_to_bf16_best(nk_f32_t const *source, nk_bf16_t *destination, nk_capability_t capabilities) {
    ((nk_f32_to_bf16_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                              nk_f32_to_bf16_capabilities()))(source, destination);
}

static nk_capability_kernels_t const *nk_f32_to_e4m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_f32_to_e4m3_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_f32_to_e4m3_best(nk_f32_t const *source, nk_e4m3_t *destination, nk_capability_t capabilities) {
    ((nk_f32_to_u8_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                            nk_f32_to_e4m3_capabilities()))(source, destination);
}

static nk_capability_kernels_t const *nk_f32_to_e5m2_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_f32_to_e5m2_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_f32_to_e5m2_best(nk_f32_t const *source, nk_e5m2_t *destination, nk_capability_t capabilities) {
    ((nk_f32_to_u8_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                            nk_f32_to_e5m2_capabilities()))(source, destination);
}

static nk_capability_kernels_t const *nk_f32_to_e2m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_f32_to_e2m3_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_f32_to_e2m3_best(nk_f32_t const *source, nk_e2m3_t *destination, nk_capability_t capabilities) {
    ((nk_f32_to_u8_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                            nk_f32_to_e2m3_capabilities()))(source, destination);
}

static nk_capability_kernels_t const *nk_f32_to_e3m2_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_f32_to_e3m2_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_f32_to_e3m2_best(nk_f32_t const *source, nk_e3m2_t *destination, nk_capability_t capabilities) {
    ((nk_f32_to_u8_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                            nk_f32_to_e3m2_capabilities()))(source, destination);
}

static nk_capability_kernels_t const *nk_ue8m0_to_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_ue8m0_to_f32_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_ue8m0_to_f32_best(nk_ue8m0_t const *src, nk_f32_t *dest, nk_capability_t capabilities) {
    ((nk_u8_to_f32_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                            nk_ue8m0_to_f32_capabilities()))(src, dest);
}

static nk_capability_kernels_t const *nk_f32_to_ue8m0_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_f32_to_ue8m0_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_f32_to_ue8m0_best(nk_f32_t const *src, nk_ue8m0_t *dest, nk_capability_t capabilities) {
    ((nk_f32_to_u8_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                            nk_f32_to_ue8m0_capabilities()))(src, dest);
}

static nk_capability_kernels_t const *nk_ue4m3_to_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_ue4m3_to_f32_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_ue4m3_to_f32_best(nk_ue4m3_t const *src, nk_f32_t *dest, nk_capability_t capabilities) {
    ((nk_u8_to_f32_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                            nk_ue4m3_to_f32_capabilities()))(src, dest);
}

static nk_capability_kernels_t const *nk_f32_to_ue4m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_f32_to_ue4m3_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_f32_to_ue4m3_best(nk_f32_t const *src, nk_ue4m3_t *dest, nk_capability_t capabilities) {
    ((nk_f32_to_u8_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                            nk_f32_to_ue4m3_capabilities()))(src, dest);
}

static nk_capability_kernels_t const *nk_nvfp4_to_f32x16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_nvfp4_to_f32x16_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_nvfp4_to_f32x16_best(nk_nvfp4_t const *src, nk_f32_t tensor_scale, nk_f32_t *dest,
                                         nk_capability_t capabilities) {
    ((nk_nvfp4_to_f32x16_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                  nk_nvfp4_to_f32x16_capabilities()))(src, tensor_scale, dest);
}

static nk_capability_kernels_t const *nk_f32x16_to_nvfp4_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_f32x16_to_nvfp4_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_f32x16_to_nvfp4_best(nk_f32_t const *src, nk_f32_t tensor_scale, nk_nvfp4_t *dest,
                                         nk_capability_t capabilities) {
    ((nk_f32x16_to_nvfp4_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                  nk_f32x16_to_nvfp4_capabilities()))(src, tensor_scale, dest);
}

static nk_capability_kernels_t const *nk_mxfp4_to_f32x32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_mxfp4_to_f32x32_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_mxfp4_to_f32x32_best(nk_mxfp4_t const *src, nk_f32_t *dest, nk_capability_t capabilities) {
    ((nk_mxfp4_to_f32x32_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                  nk_mxfp4_to_f32x32_capabilities()))(src, dest);
}

static nk_capability_kernels_t const *nk_f32x32_to_mxfp4_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_f32x32_to_mxfp4_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_f32x32_to_mxfp4_best(nk_f32_t const *src, nk_mxfp4_t *dest, nk_capability_t capabilities) {
    ((nk_f32x32_to_mxfp4_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                  nk_f32x32_to_mxfp4_capabilities()))(src, dest);
}

static nk_capability_kernels_t const *nk_mxfp6e2m3_to_f32x32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_mxfp6e2m3_to_f32x32_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_mxfp6e2m3_to_f32x32_best(nk_mxfp6e2m3_t const *src, nk_f32_t *dest, nk_capability_t capabilities) {
    ((nk_mxfp6e2m3_to_f32x32_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                      nk_mxfp6e2m3_to_f32x32_capabilities()))(src, dest);
}

static nk_capability_kernels_t const *nk_f32x32_to_mxfp6e2m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_f32x32_to_mxfp6e2m3_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_f32x32_to_mxfp6e2m3_best(nk_f32_t const *src, nk_mxfp6e2m3_t *dest, nk_capability_t capabilities) {
    ((nk_f32x32_to_mxfp6e2m3_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                      nk_f32x32_to_mxfp6e2m3_capabilities()))(src, dest);
}

static nk_capability_kernels_t const *nk_mxfp6e3m2_to_f32x32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_mxfp6e3m2_to_f32x32_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_mxfp6e3m2_to_f32x32_best(nk_mxfp6e3m2_t const *src, nk_f32_t *dest, nk_capability_t capabilities) {
    ((nk_mxfp6e3m2_to_f32x32_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                      nk_mxfp6e3m2_to_f32x32_capabilities()))(src, dest);
}

static nk_capability_kernels_t const *nk_f32x32_to_mxfp6e3m2_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_f32x32_to_mxfp6e3m2_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_f32x32_to_mxfp6e3m2_best(nk_f32_t const *src, nk_mxfp6e3m2_t *dest, nk_capability_t capabilities) {
    ((nk_f32x32_to_mxfp6e3m2_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                      nk_f32x32_to_mxfp6e3m2_capabilities()))(src, dest);
}

static nk_capability_kernels_t const *nk_mxfp8e4m3_to_f32x32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_mxfp8e4m3_to_f32x32_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_mxfp8e4m3_to_f32x32_best(nk_mxfp8e4m3_t const *src, nk_f32_t *dest, nk_capability_t capabilities) {
    ((nk_mxfp8e4m3_to_f32x32_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                      nk_mxfp8e4m3_to_f32x32_capabilities()))(src, dest);
}

static nk_capability_kernels_t const *nk_f32x32_to_mxfp8e4m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_f32x32_to_mxfp8e4m3_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_f32x32_to_mxfp8e4m3_best(nk_f32_t const *src, nk_mxfp8e4m3_t *dest, nk_capability_t capabilities) {
    ((nk_f32x32_to_mxfp8e4m3_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                      nk_f32x32_to_mxfp8e4m3_capabilities()))(src, dest);
}

static nk_capability_kernels_t const *nk_mxfp8e5m2_to_f32x32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_mxfp8e5m2_to_f32x32_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_mxfp8e5m2_to_f32x32_best(nk_mxfp8e5m2_t const *src, nk_f32_t *dest, nk_capability_t capabilities) {
    ((nk_mxfp8e5m2_to_f32x32_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                      nk_mxfp8e5m2_to_f32x32_capabilities()))(src, dest);
}

static nk_capability_kernels_t const *nk_f32x32_to_mxfp8e5m2_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_f32x32_to_mxfp8e5m2_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_f32x32_to_mxfp8e5m2_best(nk_f32_t const *src, nk_mxfp8e5m2_t *dest, nk_capability_t capabilities) {
    ((nk_f32x32_to_mxfp8e5m2_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                      nk_f32x32_to_mxfp8e5m2_capabilities()))(src, dest);
}

static nk_capability_kernels_t const *nk_mxint8_to_f32x32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_mxint8_to_f32x32_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_mxint8_to_f32x32_best(nk_mxint8_t const *src, nk_f32_t *dest, nk_capability_t capabilities) {
    ((nk_mxint8_to_f32x32_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                   nk_mxint8_to_f32x32_capabilities()))(src, dest);
}

static nk_capability_kernels_t const *nk_f32x32_to_mxint8_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_f32x32_to_mxint8_serial,
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API void nk_f32x32_to_mxint8_best(nk_f32_t const *src, nk_mxint8_t *dest, nk_capability_t capabilities) {
    ((nk_f32x32_to_mxint8_punned_t)nk_kernel_pick_((capabilities & nk_cap_cpus_k) | nk_cap_serial_k,
                                                   nk_f32x32_to_mxint8_capabilities()))(src, dest);
}

NUMKONG_API nk_status_t nk_scalar_buffer_cast(nk_scalar_buffer_t const *source, nk_dtype_t source_dtype,
                                              nk_scalar_buffer_t *target, nk_dtype_t target_dtype) {
    nk_scalar_buffer_t local = *source;
    nk_f64c_t hub;
    if (!nk_scalar_buffer_to_f64c_serial_(&local, source_dtype, &hub)) return nk_missing_kernel_k;
    return nk_scalar_buffer_from_f64c_serial_(&hub, target, target_dtype) ? nk_success_k : nk_missing_kernel_k;
}

NUMKONG_API nk_status_t nk_cast_best(void const *from, nk_dtype_t from_dtype, void *to, nk_dtype_t to_dtype,
                                     nk_size_t count, nk_capability_t capabilities, nk_stream_t stream) {
    nk_kernel_cast_punned_t const kernel = (nk_kernel_cast_punned_t)nk_kernel_pick_(capabilities,
                                                                                    nk_cast_capabilities());
    return kernel ? kernel(from, from_dtype, to, to_dtype, count, stream) : nk_missing_kernel_k;
}

NUMKONG_API nk_status_t nk_cast_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                            nk_kernel_punned_t *kernel, nk_capability_t *capability) {
    nk_capability_kernels_t const *lists = NUMKONG_NULL;
    switch (dtype) {
    case nk_dtype_unknown_k:
        switch (kind) {
        case nk_kernel_cast_k: lists = nk_cast_capabilities(); break;
        default: break;
        }
        break;
    default: break;
    }
    *kernel = lists ? nk_kernel_pick_(capabilities, lists) : (nk_kernel_punned_t)NUMKONG_NULL;
    *capability = *kernel ? nk_capability_pick_(capabilities, lists) : 0;
    return *kernel ? nk_success_k : nk_missing_kernel_k;
}
