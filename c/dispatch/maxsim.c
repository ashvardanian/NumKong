/**
 *  @file c/dispatch/maxsim.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief The MaxSim capability lists, @c _best dispatch points and @c nk_maxsim_find_kernel.
 */
#include "dispatch.h"
#include "numkong/maxsim.h"

static nk_capability_kernels_t const *nk_maxsim_pack_size_bf16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_maxsim_pack_size_bf16_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_maxsim_pack_size_bf16_neonsdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_maxsim_pack_size_bf16_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_maxsim_pack_size_bf16_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_maxsim_pack_size_bf16_alder,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_maxsim_pack_size_bf16_genoa,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_maxsim_pack_size_bf16_sapphireamx,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_maxsim_pack_size_bf16_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_alder_k * NUMKONG_TARGET_ALDER |
             nk_cap_genoa_k * NUMKONG_TARGET_GENOA | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_maxsim_pack_size_bf16_best(nk_size_t columns, nk_size_t depth, nk_capability_t capabilities,
                                                      nk_size_t *bytes) {
    nk_dots_pack_size_punned_t const kernel = (nk_dots_pack_size_punned_t)nk_kernel_pick_(
        capabilities, nk_maxsim_pack_size_bf16_capabilities());
    return kernel ? kernel(columns, depth, bytes) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_maxsim_pack_size_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_maxsim_pack_size_f32_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_maxsim_pack_size_f32_neonsdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_maxsim_pack_size_f32_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_maxsim_pack_size_f32_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_maxsim_pack_size_f32_alder,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_maxsim_pack_size_f32_icelake,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_maxsim_pack_size_f32_sapphireamx,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_maxsim_pack_size_f32_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_alder_k * NUMKONG_TARGET_ALDER |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_maxsim_pack_size_f32_best(nk_size_t columns, nk_size_t depth, nk_capability_t capabilities,
                                                     nk_size_t *bytes) {
    nk_dots_pack_size_punned_t const kernel = (nk_dots_pack_size_punned_t)nk_kernel_pick_(
        capabilities, nk_maxsim_pack_size_f32_capabilities());
    return kernel ? kernel(columns, depth, bytes) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_maxsim_pack_size_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_maxsim_pack_size_f16_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_maxsim_pack_size_f16_neonsdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_maxsim_pack_size_f16_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_maxsim_pack_size_f16_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_maxsim_pack_size_f16_alder,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_maxsim_pack_size_f16_icelake,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_maxsim_pack_size_f16_sapphireamx,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_maxsim_pack_size_f16_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_alder_k * NUMKONG_TARGET_ALDER |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_maxsim_pack_size_f16_best(nk_size_t columns, nk_size_t depth, nk_capability_t capabilities,
                                                     nk_size_t *bytes) {
    nk_dots_pack_size_punned_t const kernel = (nk_dots_pack_size_punned_t)nk_kernel_pick_(
        capabilities, nk_maxsim_pack_size_f16_capabilities());
    return kernel ? kernel(columns, depth, bytes) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_maxsim_packed_shape_bf16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_maxsim_packed_shape_bf16_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_maxsim_packed_shape_bf16_neonsdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_maxsim_packed_shape_bf16_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_maxsim_packed_shape_bf16_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_maxsim_packed_shape_bf16_alder,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_maxsim_packed_shape_bf16_genoa,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_maxsim_packed_shape_bf16_sapphireamx,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_maxsim_packed_shape_bf16_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_alder_k * NUMKONG_TARGET_ALDER |
             nk_cap_genoa_k * NUMKONG_TARGET_GENOA | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_maxsim_packed_shape_bf16_best(void const *packed, nk_size_t *columns, nk_size_t *depth,
                                                         nk_capability_t capabilities, void *stream) {
    nk_dots_packed_shape_punned_t const kernel = (nk_dots_packed_shape_punned_t)nk_kernel_pick_(
        capabilities, nk_maxsim_packed_shape_bf16_capabilities());
    return kernel ? kernel(packed, columns, depth, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_maxsim_packed_shape_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_maxsim_packed_shape_f32_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_maxsim_packed_shape_f32_neonsdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_maxsim_packed_shape_f32_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_maxsim_packed_shape_f32_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_maxsim_packed_shape_f32_alder,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_maxsim_packed_shape_f32_icelake,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_maxsim_packed_shape_f32_sapphireamx,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_maxsim_packed_shape_f32_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_alder_k * NUMKONG_TARGET_ALDER |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_maxsim_packed_shape_f32_best(void const *packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_capability_t capabilities, void *stream) {
    nk_dots_packed_shape_punned_t const kernel = (nk_dots_packed_shape_punned_t)nk_kernel_pick_(
        capabilities, nk_maxsim_packed_shape_f32_capabilities());
    return kernel ? kernel(packed, columns, depth, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_maxsim_packed_shape_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_maxsim_packed_shape_f16_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_maxsim_packed_shape_f16_neonsdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_maxsim_packed_shape_f16_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_maxsim_packed_shape_f16_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_maxsim_packed_shape_f16_alder,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_maxsim_packed_shape_f16_icelake,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_maxsim_packed_shape_f16_sapphireamx,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_maxsim_packed_shape_f16_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_alder_k * NUMKONG_TARGET_ALDER |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_maxsim_packed_shape_f16_best(void const *packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_capability_t capabilities, void *stream) {
    nk_dots_packed_shape_punned_t const kernel = (nk_dots_packed_shape_punned_t)nk_kernel_pick_(
        capabilities, nk_maxsim_packed_shape_f16_capabilities());
    return kernel ? kernel(packed, columns, depth, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_maxsim_pack_bf16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_maxsim_pack_bf16_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_maxsim_pack_bf16_neonsdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_maxsim_pack_bf16_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_maxsim_pack_bf16_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_maxsim_pack_bf16_alder,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_maxsim_pack_bf16_genoa,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_maxsim_pack_bf16_sapphireamx,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_maxsim_pack_bf16_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_alder_k * NUMKONG_TARGET_ALDER |
             nk_cap_genoa_k * NUMKONG_TARGET_GENOA | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_maxsim_pack_bf16_best(nk_bf16_t const *b, nk_size_t columns, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_capability_t capabilities,
                                                 void *stream) {
    nk_maxsim_pack_punned_t const kernel = (nk_maxsim_pack_punned_t)nk_kernel_pick_(capabilities,
                                                                                    nk_maxsim_pack_bf16_capabilities());
    return kernel ? kernel(b, columns, depth, b_stride, b_packed, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_maxsim_pack_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_maxsim_pack_f32_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_maxsim_pack_f32_neonsdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_maxsim_pack_f32_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_maxsim_pack_f32_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_maxsim_pack_f32_alder,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_maxsim_pack_f32_icelake,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_maxsim_pack_f32_sapphireamx,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_maxsim_pack_f32_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_alder_k * NUMKONG_TARGET_ALDER |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_maxsim_pack_f32_best(nk_f32_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_capability_t capabilities,
                                                void *stream) {
    nk_maxsim_pack_punned_t const kernel = (nk_maxsim_pack_punned_t)nk_kernel_pick_(capabilities,
                                                                                    nk_maxsim_pack_f32_capabilities());
    return kernel ? kernel(b, columns, depth, b_stride, b_packed, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_maxsim_pack_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_maxsim_pack_f16_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_maxsim_pack_f16_neonsdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_maxsim_pack_f16_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_maxsim_pack_f16_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_maxsim_pack_f16_alder,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_maxsim_pack_f16_icelake,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_maxsim_pack_f16_sapphireamx,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_maxsim_pack_f16_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_alder_k * NUMKONG_TARGET_ALDER |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_maxsim_pack_f16_best(nk_f16_t const *b, nk_size_t columns, nk_size_t depth,
                                                nk_size_t b_stride, void *b_packed, nk_capability_t capabilities,
                                                void *stream) {
    nk_maxsim_pack_punned_t const kernel = (nk_maxsim_pack_punned_t)nk_kernel_pick_(capabilities,
                                                                                    nk_maxsim_pack_f16_capabilities());
    return kernel ? kernel(b, columns, depth, b_stride, b_packed, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_maxsim_packed_bf16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_maxsim_packed_bf16_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_maxsim_packed_bf16_neonsdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_maxsim_packed_bf16_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_maxsim_packed_bf16_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_maxsim_packed_bf16_alder,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_maxsim_packed_bf16_genoa,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_maxsim_packed_bf16_sapphireamx,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_maxsim_packed_bf16_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_alder_k * NUMKONG_TARGET_ALDER |
             nk_cap_genoa_k * NUMKONG_TARGET_GENOA | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_maxsim_packed_bf16_best(void const *query_packed, void const *document_packed,
                                                   nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                                   nk_f32_t *result, nk_capability_t capabilities, void *stream) {
    nk_maxsim_packed_punned_t const kernel = (nk_maxsim_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_maxsim_packed_bf16_capabilities());
    return kernel ? kernel(query_packed, document_packed, query_count, document_count, depth, result, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_maxsim_packed_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_maxsim_packed_f32_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_maxsim_packed_f32_neonsdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_maxsim_packed_f32_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_maxsim_packed_f32_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_maxsim_packed_f32_alder,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_maxsim_packed_f32_icelake,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_maxsim_packed_f32_sapphireamx,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_maxsim_packed_f32_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_alder_k * NUMKONG_TARGET_ALDER |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_maxsim_packed_f32_best(void const *query_packed, void const *document_packed,
                                                  nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                                  nk_f64_t *result, nk_capability_t capabilities, void *stream) {
    nk_maxsim_packed_punned_t const kernel = (nk_maxsim_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_maxsim_packed_f32_capabilities());
    return kernel ? kernel(query_packed, document_packed, query_count, document_count, depth, result, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_maxsim_packed_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_maxsim_packed_f16_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_maxsim_packed_f16_neonsdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_maxsim_packed_f16_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_maxsim_packed_f16_haswell,
#endif
#if NUMKONG_TARGET_ALDER
        (nk_kernel_punned_t)&nk_maxsim_packed_f16_alder,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_maxsim_packed_f16_icelake,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_maxsim_packed_f16_sapphireamx,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_maxsim_packed_f16_v128relaxed,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_alder_k * NUMKONG_TARGET_ALDER |
             nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_maxsim_packed_f16_best(void const *query_packed, void const *document_packed,
                                                  nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                                  nk_f32_t *result, nk_capability_t capabilities, void *stream) {
    nk_maxsim_packed_punned_t const kernel = (nk_maxsim_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_maxsim_packed_f16_capabilities());
    return kernel ? kernel(query_packed, document_packed, query_count, document_count, depth, result, stream)
                  : nk_missing_kernel_k;
}

NUMKONG_API nk_status_t nk_maxsim_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                              nk_kernel_punned_t *kernel, nk_capability_t *capability) {
    nk_capability_kernels_t const *lists = NUMKONG_NULL;
    switch (dtype) {
    case nk_f32_k:
        switch (kind) {
        case nk_kernel_maxsim_pack_size_k: lists = nk_maxsim_pack_size_f32_capabilities(); break;
        case nk_kernel_maxsim_packed_shape_k: lists = nk_maxsim_packed_shape_f32_capabilities(); break;
        case nk_kernel_maxsim_pack_k: lists = nk_maxsim_pack_f32_capabilities(); break;
        case nk_kernel_maxsim_packed_k: lists = nk_maxsim_packed_f32_capabilities(); break;
        default: break;
        }
        break;
    case nk_bf16_k:
        switch (kind) {
        case nk_kernel_maxsim_pack_size_k: lists = nk_maxsim_pack_size_bf16_capabilities(); break;
        case nk_kernel_maxsim_packed_shape_k: lists = nk_maxsim_packed_shape_bf16_capabilities(); break;
        case nk_kernel_maxsim_pack_k: lists = nk_maxsim_pack_bf16_capabilities(); break;
        case nk_kernel_maxsim_packed_k: lists = nk_maxsim_packed_bf16_capabilities(); break;
        default: break;
        }
        break;
    case nk_f16_k:
        switch (kind) {
        case nk_kernel_maxsim_pack_size_k: lists = nk_maxsim_pack_size_f16_capabilities(); break;
        case nk_kernel_maxsim_packed_shape_k: lists = nk_maxsim_packed_shape_f16_capabilities(); break;
        case nk_kernel_maxsim_pack_k: lists = nk_maxsim_pack_f16_capabilities(); break;
        case nk_kernel_maxsim_packed_k: lists = nk_maxsim_packed_f16_capabilities(); break;
        default: break;
        }
        break;
    default: break;
    }
    *kernel = lists ? nk_kernel_pick_(capabilities, lists) : (nk_kernel_punned_t)NUMKONG_NULL;
    *capability = *kernel ? nk_capability_pick_(capabilities, lists) : 0;
    return *kernel ? nk_success_k : nk_missing_kernel_k;
}
