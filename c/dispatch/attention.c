/**
 *  @file c/dispatch/attention.c
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief The attention capability lists, @c _best dispatch points and @c nk_attention_find_kernel.
 */
#include "dispatch.h"
#include "numkong/attention.h"

static nk_capability_kernels_t const *nk_attention_pack_size_bf16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_attention_pack_size_bf16_serial,
#if NUMKONG_TARGET_NEONBFDOT
        (nk_kernel_punned_t)&nk_attention_pack_size_bf16_neonbfdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_attention_pack_size_bf16_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_attention_pack_size_bf16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_attention_pack_size_bf16_skylake,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_attention_pack_size_bf16_genoa,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_attention_pack_size_bf16_sapphireamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_attention_pack_size_bf16_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_attention_pack_size_bf16_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_attention_pack_size_bf16_cuda,
#endif
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_attention_pack_size_bf16_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_attention_pack_size_bf16_hopper,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_attention_pack_size_bf16_blackwell,
#endif
#if NUMKONG_TARGET_BLACKWELLULTRA
        (nk_kernel_punned_t)&nk_attention_pack_size_bf16_blackwellultra,
#endif
    };
    static nk_kernel_punned_t const rocm[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_ROCM
        (nk_kernel_punned_t)&nk_attention_pack_size_bf16_rocm,
#endif
#if NUMKONG_TARGET_CDNA3
        (nk_kernel_punned_t)&nk_attention_pack_size_bf16_cdna3,
#endif
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_attention_pack_size_bf16_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_attention_pack_size_bf16_cdna5,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonbfdot_k * NUMKONG_TARGET_NEONBFDOT | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_genoa_k * NUMKONG_TARGET_GENOA | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE |
             nk_cap_hopper_k * NUMKONG_TARGET_HOPPER | nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL |
             nk_cap_blackwellultra_k * NUMKONG_TARGET_BLACKWELLULTRA,
         cuda},
        {nk_cap_rocm_k * NUMKONG_TARGET_ROCM | nk_cap_cdna3_k * NUMKONG_TARGET_CDNA3 |
             nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5,
         rocm},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_attention_pack_size_bf16_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_size_t token_count, nk_size_t segment_count,
                                                         nk_capability_t capabilities, nk_size_t *bytes) {
    nk_attention_pack_size_punned_t const kernel = (nk_attention_pack_size_punned_t)nk_kernel_pick_(
        capabilities, nk_attention_pack_size_bf16_capabilities());
    return kernel ? kernel(key_value_head_count, depth, token_count, segment_count, bytes) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_attention_pack_size_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_attention_pack_size_f16_serial,
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_attention_pack_size_f16_cuda,
#endif
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_attention_pack_size_f16_ampere,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_attention_pack_size_f16_blackwell,
#endif
#if NUMKONG_TARGET_BLACKWELLULTRA
        (nk_kernel_punned_t)&nk_attention_pack_size_f16_blackwellultra,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE |
             nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL | nk_cap_blackwellultra_k * NUMKONG_TARGET_BLACKWELLULTRA,
         cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_attention_pack_size_f16_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_size_t token_count, nk_size_t segment_count,
                                                        nk_capability_t capabilities, nk_size_t *bytes) {
    nk_attention_pack_size_punned_t const kernel = (nk_attention_pack_size_punned_t)nk_kernel_pick_(
        capabilities, nk_attention_pack_size_f16_capabilities());
    return kernel ? kernel(key_value_head_count, depth, token_count, segment_count, bytes) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_attention_pack_size_e4m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_attention_pack_size_e4m3_serial,
#if NUMKONG_TARGET_NEONFHM
        (nk_kernel_punned_t)&nk_attention_pack_size_e4m3_neonfhm,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_attention_pack_size_e4m3_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_attention_pack_size_e4m3_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_attention_pack_size_e4m3_skylake,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_attention_pack_size_e4m3_genoa,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_attention_pack_size_e4m3_sapphireamx,
#endif
#if NUMKONG_TARGET_DIAMONDAMX
        (nk_kernel_punned_t)&nk_attention_pack_size_e4m3_diamondamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_attention_pack_size_e4m3_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_attention_pack_size_e4m3_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_attention_pack_size_e4m3_cuda,
#endif
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_attention_pack_size_e4m3_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_attention_pack_size_e4m3_hopper,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_attention_pack_size_e4m3_blackwell,
#endif
#if NUMKONG_TARGET_BLACKWELLRTX
        (nk_kernel_punned_t)&nk_attention_pack_size_e4m3_blackwellrtx,
#endif
#if NUMKONG_TARGET_BLACKWELLULTRA
        (nk_kernel_punned_t)&nk_attention_pack_size_e4m3_blackwellultra,
#endif
    };
    static nk_kernel_punned_t const rocm[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_ROCM
        (nk_kernel_punned_t)&nk_attention_pack_size_e4m3_rocm,
#endif
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_attention_pack_size_e4m3_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_attention_pack_size_e4m3_cdna5,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonfhm_k * NUMKONG_TARGET_NEONFHM | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_genoa_k * NUMKONG_TARGET_GENOA | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_diamondamx_k * NUMKONG_TARGET_DIAMONDAMX | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE |
             nk_cap_hopper_k * NUMKONG_TARGET_HOPPER | nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL |
             nk_cap_blackwellrtx_k * NUMKONG_TARGET_BLACKWELLRTX |
             nk_cap_blackwellultra_k * NUMKONG_TARGET_BLACKWELLULTRA,
         cuda},
        {nk_cap_rocm_k * NUMKONG_TARGET_ROCM | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 |
             nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5,
         rocm},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_size_t token_count, nk_size_t segment_count,
                                                         nk_capability_t capabilities, nk_size_t *bytes) {
    nk_attention_pack_size_punned_t const kernel = (nk_attention_pack_size_punned_t)nk_kernel_pick_(
        capabilities, nk_attention_pack_size_e4m3_capabilities());
    return kernel ? kernel(key_value_head_count, depth, token_count, segment_count, bytes) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_attention_pack_size_i8_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_attention_pack_size_i8_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_attention_pack_size_i8_neonsdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_attention_pack_size_i8_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_attention_pack_size_i8_haswell,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_attention_pack_size_i8_icelake,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_attention_pack_size_i8_sapphireamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_attention_pack_size_i8_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_attention_pack_size_i8_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_attention_pack_size_i8_cuda,
#endif
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_attention_pack_size_i8_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_attention_pack_size_i8_hopper,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_attention_pack_size_i8_blackwell,
#endif
#if NUMKONG_TARGET_BLACKWELLULTRA
        (nk_kernel_punned_t)&nk_attention_pack_size_i8_blackwellultra,
#endif
    };
    static nk_kernel_punned_t const rocm[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_ROCM
        (nk_kernel_punned_t)&nk_attention_pack_size_i8_rocm,
#endif
#if NUMKONG_TARGET_CDNA3
        (nk_kernel_punned_t)&nk_attention_pack_size_i8_cdna3,
#endif
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_attention_pack_size_i8_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_attention_pack_size_i8_cdna5,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE |
             nk_cap_hopper_k * NUMKONG_TARGET_HOPPER | nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL |
             nk_cap_blackwellultra_k * NUMKONG_TARGET_BLACKWELLULTRA,
         cuda},
        {nk_cap_rocm_k * NUMKONG_TARGET_ROCM | nk_cap_cdna3_k * NUMKONG_TARGET_CDNA3 |
             nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5,
         rocm},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_attention_pack_size_i8_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_size_t token_count, nk_size_t segment_count,
                                                       nk_capability_t capabilities, nk_size_t *bytes) {
    nk_attention_pack_size_punned_t const kernel = (nk_attention_pack_size_punned_t)nk_kernel_pick_(
        capabilities, nk_attention_pack_size_i8_capabilities());
    return kernel ? kernel(key_value_head_count, depth, token_count, segment_count, bytes) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_attention_packed_shape_bf16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_attention_packed_shape_bf16_serial,
#if NUMKONG_TARGET_NEONBFDOT
        (nk_kernel_punned_t)&nk_attention_packed_shape_bf16_neonbfdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_attention_packed_shape_bf16_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_attention_packed_shape_bf16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_attention_packed_shape_bf16_skylake,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_attention_packed_shape_bf16_genoa,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_attention_packed_shape_bf16_sapphireamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_attention_packed_shape_bf16_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_attention_packed_shape_bf16_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_attention_packed_shape_bf16_cuda,
#endif
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_attention_packed_shape_bf16_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_attention_packed_shape_bf16_hopper,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_attention_packed_shape_bf16_blackwell,
#endif
#if NUMKONG_TARGET_BLACKWELLULTRA
        (nk_kernel_punned_t)&nk_attention_packed_shape_bf16_blackwellultra,
#endif
    };
    static nk_kernel_punned_t const rocm[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_ROCM
        (nk_kernel_punned_t)&nk_attention_packed_shape_bf16_rocm,
#endif
#if NUMKONG_TARGET_CDNA3
        (nk_kernel_punned_t)&nk_attention_packed_shape_bf16_cdna3,
#endif
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_attention_packed_shape_bf16_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_attention_packed_shape_bf16_cdna5,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonbfdot_k * NUMKONG_TARGET_NEONBFDOT | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_genoa_k * NUMKONG_TARGET_GENOA | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE |
             nk_cap_hopper_k * NUMKONG_TARGET_HOPPER | nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL |
             nk_cap_blackwellultra_k * NUMKONG_TARGET_BLACKWELLULTRA,
         cuda},
        {nk_cap_rocm_k * NUMKONG_TARGET_ROCM | nk_cap_cdna3_k * NUMKONG_TARGET_CDNA3 |
             nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5,
         rocm},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_best(void const *key_value_packed, nk_size_t *heads,
                                                            nk_size_t *depth, nk_size_t *segments,
                                                            nk_capability_t capabilities, nk_stream_t stream) {
    nk_attention_packed_shape_punned_t const kernel = (nk_attention_packed_shape_punned_t)nk_kernel_pick_(
        capabilities, nk_attention_packed_shape_bf16_capabilities());
    return kernel ? kernel(key_value_packed, heads, depth, segments, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_attention_packed_shape_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_attention_packed_shape_f16_serial,
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_attention_packed_shape_f16_cuda,
#endif
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_attention_packed_shape_f16_ampere,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_attention_packed_shape_f16_blackwell,
#endif
#if NUMKONG_TARGET_BLACKWELLULTRA
        (nk_kernel_punned_t)&nk_attention_packed_shape_f16_blackwellultra,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE |
             nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL | nk_cap_blackwellultra_k * NUMKONG_TARGET_BLACKWELLULTRA,
         cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_f16_best(void const *key_value_packed, nk_size_t *heads,
                                                           nk_size_t *depth, nk_size_t *segments,
                                                           nk_capability_t capabilities, nk_stream_t stream) {
    nk_attention_packed_shape_punned_t const kernel = (nk_attention_packed_shape_punned_t)nk_kernel_pick_(
        capabilities, nk_attention_packed_shape_f16_capabilities());
    return kernel ? kernel(key_value_packed, heads, depth, segments, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_attention_packed_shape_e4m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_attention_packed_shape_e4m3_serial,
#if NUMKONG_TARGET_NEONFHM
        (nk_kernel_punned_t)&nk_attention_packed_shape_e4m3_neonfhm,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_attention_packed_shape_e4m3_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_attention_packed_shape_e4m3_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_attention_packed_shape_e4m3_skylake,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_attention_packed_shape_e4m3_genoa,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_attention_packed_shape_e4m3_sapphireamx,
#endif
#if NUMKONG_TARGET_DIAMONDAMX
        (nk_kernel_punned_t)&nk_attention_packed_shape_e4m3_diamondamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_attention_packed_shape_e4m3_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_attention_packed_shape_e4m3_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_attention_packed_shape_e4m3_cuda,
#endif
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_attention_packed_shape_e4m3_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_attention_packed_shape_e4m3_hopper,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_attention_packed_shape_e4m3_blackwell,
#endif
#if NUMKONG_TARGET_BLACKWELLRTX
        (nk_kernel_punned_t)&nk_attention_packed_shape_e4m3_blackwellrtx,
#endif
#if NUMKONG_TARGET_BLACKWELLULTRA
        (nk_kernel_punned_t)&nk_attention_packed_shape_e4m3_blackwellultra,
#endif
    };
    static nk_kernel_punned_t const rocm[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_ROCM
        (nk_kernel_punned_t)&nk_attention_packed_shape_e4m3_rocm,
#endif
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_attention_packed_shape_e4m3_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_attention_packed_shape_e4m3_cdna5,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonfhm_k * NUMKONG_TARGET_NEONFHM | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_genoa_k * NUMKONG_TARGET_GENOA | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_diamondamx_k * NUMKONG_TARGET_DIAMONDAMX | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE |
             nk_cap_hopper_k * NUMKONG_TARGET_HOPPER | nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL |
             nk_cap_blackwellrtx_k * NUMKONG_TARGET_BLACKWELLRTX |
             nk_cap_blackwellultra_k * NUMKONG_TARGET_BLACKWELLULTRA,
         cuda},
        {nk_cap_rocm_k * NUMKONG_TARGET_ROCM | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 |
             nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5,
         rocm},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_best(void const *key_value_packed, nk_size_t *heads,
                                                            nk_size_t *depth, nk_size_t *segments,
                                                            nk_capability_t capabilities, nk_stream_t stream) {
    nk_attention_packed_shape_punned_t const kernel = (nk_attention_packed_shape_punned_t)nk_kernel_pick_(
        capabilities, nk_attention_packed_shape_e4m3_capabilities());
    return kernel ? kernel(key_value_packed, heads, depth, segments, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_attention_packed_shape_i8_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_attention_packed_shape_i8_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_attention_packed_shape_i8_neonsdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_attention_packed_shape_i8_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_attention_packed_shape_i8_haswell,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_attention_packed_shape_i8_icelake,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_attention_packed_shape_i8_sapphireamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_attention_packed_shape_i8_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_attention_packed_shape_i8_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_attention_packed_shape_i8_cuda,
#endif
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_attention_packed_shape_i8_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_attention_packed_shape_i8_hopper,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_attention_packed_shape_i8_blackwell,
#endif
#if NUMKONG_TARGET_BLACKWELLULTRA
        (nk_kernel_punned_t)&nk_attention_packed_shape_i8_blackwellultra,
#endif
    };
    static nk_kernel_punned_t const rocm[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_ROCM
        (nk_kernel_punned_t)&nk_attention_packed_shape_i8_rocm,
#endif
#if NUMKONG_TARGET_CDNA3
        (nk_kernel_punned_t)&nk_attention_packed_shape_i8_cdna3,
#endif
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_attention_packed_shape_i8_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_attention_packed_shape_i8_cdna5,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE |
             nk_cap_hopper_k * NUMKONG_TARGET_HOPPER | nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL |
             nk_cap_blackwellultra_k * NUMKONG_TARGET_BLACKWELLULTRA,
         cuda},
        {nk_cap_rocm_k * NUMKONG_TARGET_ROCM | nk_cap_cdna3_k * NUMKONG_TARGET_CDNA3 |
             nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5,
         rocm},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_i8_best(void const *key_value_packed, nk_size_t *heads,
                                                          nk_size_t *depth, nk_size_t *segments,
                                                          nk_capability_t capabilities, nk_stream_t stream) {
    nk_attention_packed_shape_punned_t const kernel = (nk_attention_packed_shape_punned_t)nk_kernel_pick_(
        capabilities, nk_attention_packed_shape_i8_capabilities());
    return kernel ? kernel(key_value_packed, heads, depth, segments, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_attention_pack_bf16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_attention_pack_bf16_serial,
#if NUMKONG_TARGET_NEONBFDOT
        (nk_kernel_punned_t)&nk_attention_pack_bf16_neonbfdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_attention_pack_bf16_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_attention_pack_bf16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_attention_pack_bf16_skylake,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_attention_pack_bf16_genoa,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_attention_pack_bf16_sapphireamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_attention_pack_bf16_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_attention_pack_bf16_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_attention_pack_bf16_cuda,
#endif
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_attention_pack_bf16_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_attention_pack_bf16_hopper,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_attention_pack_bf16_blackwell,
#endif
#if NUMKONG_TARGET_BLACKWELLULTRA
        (nk_kernel_punned_t)&nk_attention_pack_bf16_blackwellultra,
#endif
    };
    static nk_kernel_punned_t const rocm[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_ROCM
        (nk_kernel_punned_t)&nk_attention_pack_bf16_rocm,
#endif
#if NUMKONG_TARGET_CDNA3
        (nk_kernel_punned_t)&nk_attention_pack_bf16_cdna3,
#endif
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_attention_pack_bf16_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_attention_pack_bf16_cdna5,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonbfdot_k * NUMKONG_TARGET_NEONBFDOT | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_genoa_k * NUMKONG_TARGET_GENOA | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE |
             nk_cap_hopper_k * NUMKONG_TARGET_HOPPER | nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL |
             nk_cap_blackwellultra_k * NUMKONG_TARGET_BLACKWELLULTRA,
         cuda},
        {nk_cap_rocm_k * NUMKONG_TARGET_ROCM | nk_cap_cdna3_k * NUMKONG_TARGET_CDNA3 |
             nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5,
         rocm},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_attention_pack_bf16_best(nk_bf16_t const *keys, nk_bf16_t const *values,
                                                    nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                    nk_size_t segment_count, nk_size_t key_stride,
                                                    nk_size_t value_stride, void *key_value_packed,
                                                    nk_size_t tasks_begin, nk_size_t tasks_end,
                                                    nk_capability_t capabilities, nk_stream_t stream) {
    nk_attention_pack_punned_t const kernel = (nk_attention_pack_punned_t)nk_kernel_pick_(
        capabilities, nk_attention_pack_bf16_capabilities());
    return kernel ? kernel(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths, segment_count,
                           key_stride, value_stride, key_value_packed, tasks_begin, tasks_end, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_attention_pack_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_attention_pack_f16_serial,
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_attention_pack_f16_cuda,
#endif
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_attention_pack_f16_ampere,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_attention_pack_f16_blackwell,
#endif
#if NUMKONG_TARGET_BLACKWELLULTRA
        (nk_kernel_punned_t)&nk_attention_pack_f16_blackwellultra,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE |
             nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL | nk_cap_blackwellultra_k * NUMKONG_TARGET_BLACKWELLULTRA,
         cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_attention_pack_f16_best(nk_f16_t const *keys, nk_f16_t const *values,
                                                   nk_size_t key_value_head_count, nk_size_t depth,
                                                   nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                   nk_size_t segment_count, nk_size_t key_stride,
                                                   nk_size_t value_stride, void *key_value_packed,
                                                   nk_size_t tasks_begin, nk_size_t tasks_end,
                                                   nk_capability_t capabilities, nk_stream_t stream) {
    nk_attention_pack_punned_t const kernel = (nk_attention_pack_punned_t)nk_kernel_pick_(
        capabilities, nk_attention_pack_f16_capabilities());
    return kernel ? kernel(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths, segment_count,
                           key_stride, value_stride, key_value_packed, tasks_begin, tasks_end, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_attention_pack_e4m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_attention_pack_e4m3_serial,
#if NUMKONG_TARGET_NEONFHM
        (nk_kernel_punned_t)&nk_attention_pack_e4m3_neonfhm,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_attention_pack_e4m3_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_attention_pack_e4m3_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_attention_pack_e4m3_skylake,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_attention_pack_e4m3_genoa,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_attention_pack_e4m3_sapphireamx,
#endif
#if NUMKONG_TARGET_DIAMONDAMX
        (nk_kernel_punned_t)&nk_attention_pack_e4m3_diamondamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_attention_pack_e4m3_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_attention_pack_e4m3_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_attention_pack_e4m3_cuda,
#endif
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_attention_pack_e4m3_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_attention_pack_e4m3_hopper,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_attention_pack_e4m3_blackwell,
#endif
#if NUMKONG_TARGET_BLACKWELLRTX
        (nk_kernel_punned_t)&nk_attention_pack_e4m3_blackwellrtx,
#endif
#if NUMKONG_TARGET_BLACKWELLULTRA
        (nk_kernel_punned_t)&nk_attention_pack_e4m3_blackwellultra,
#endif
    };
    static nk_kernel_punned_t const rocm[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_ROCM
        (nk_kernel_punned_t)&nk_attention_pack_e4m3_rocm,
#endif
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_attention_pack_e4m3_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_attention_pack_e4m3_cdna5,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonfhm_k * NUMKONG_TARGET_NEONFHM | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_genoa_k * NUMKONG_TARGET_GENOA | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_diamondamx_k * NUMKONG_TARGET_DIAMONDAMX | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE |
             nk_cap_hopper_k * NUMKONG_TARGET_HOPPER | nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL |
             nk_cap_blackwellrtx_k * NUMKONG_TARGET_BLACKWELLRTX |
             nk_cap_blackwellultra_k * NUMKONG_TARGET_BLACKWELLULTRA,
         cuda},
        {nk_cap_rocm_k * NUMKONG_TARGET_ROCM | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 |
             nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5,
         rocm},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_attention_pack_e4m3_best(nk_e4m3_t const *keys, nk_e4m3_t const *values,
                                                    nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                    nk_size_t segment_count, nk_size_t key_stride,
                                                    nk_size_t value_stride, void *key_value_packed,
                                                    nk_size_t tasks_begin, nk_size_t tasks_end,
                                                    nk_capability_t capabilities, nk_stream_t stream) {
    nk_attention_pack_punned_t const kernel = (nk_attention_pack_punned_t)nk_kernel_pick_(
        capabilities, nk_attention_pack_e4m3_capabilities());
    return kernel ? kernel(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths, segment_count,
                           key_stride, value_stride, key_value_packed, tasks_begin, tasks_end, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_attention_pack_i8_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_attention_pack_i8_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_attention_pack_i8_neonsdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_attention_pack_i8_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_attention_pack_i8_haswell,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_attention_pack_i8_icelake,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_attention_pack_i8_sapphireamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_attention_pack_i8_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_attention_pack_i8_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_attention_pack_i8_cuda,
#endif
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_attention_pack_i8_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_attention_pack_i8_hopper,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_attention_pack_i8_blackwell,
#endif
#if NUMKONG_TARGET_BLACKWELLULTRA
        (nk_kernel_punned_t)&nk_attention_pack_i8_blackwellultra,
#endif
    };
    static nk_kernel_punned_t const rocm[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_ROCM
        (nk_kernel_punned_t)&nk_attention_pack_i8_rocm,
#endif
#if NUMKONG_TARGET_CDNA3
        (nk_kernel_punned_t)&nk_attention_pack_i8_cdna3,
#endif
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_attention_pack_i8_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_attention_pack_i8_cdna5,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE |
             nk_cap_hopper_k * NUMKONG_TARGET_HOPPER | nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL |
             nk_cap_blackwellultra_k * NUMKONG_TARGET_BLACKWELLULTRA,
         cuda},
        {nk_cap_rocm_k * NUMKONG_TARGET_ROCM | nk_cap_cdna3_k * NUMKONG_TARGET_CDNA3 |
             nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5,
         rocm},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_attention_pack_i8_best(nk_i8_t const *keys, nk_i8_t const *values,
                                                  nk_size_t key_value_head_count, nk_size_t depth,
                                                  nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                  nk_size_t segment_count, nk_size_t key_stride, nk_size_t value_stride,
                                                  void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end,
                                                  nk_capability_t capabilities, nk_stream_t stream) {
    nk_attention_pack_punned_t const kernel = (nk_attention_pack_punned_t)nk_kernel_pick_(
        capabilities, nk_attention_pack_i8_capabilities());
    return kernel ? kernel(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths, segment_count,
                           key_stride, value_stride, key_value_packed, tasks_begin, tasks_end, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_attention_packed_bf16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_attention_packed_bf16_serial,
#if NUMKONG_TARGET_NEONBFDOT
        (nk_kernel_punned_t)&nk_attention_packed_bf16_neonbfdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_attention_packed_bf16_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_attention_packed_bf16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_attention_packed_bf16_skylake,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_attention_packed_bf16_genoa,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_attention_packed_bf16_sapphireamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_attention_packed_bf16_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_attention_packed_bf16_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_attention_packed_bf16_cuda,
#endif
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_attention_packed_bf16_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_attention_packed_bf16_hopper,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_attention_packed_bf16_blackwell,
#endif
#if NUMKONG_TARGET_BLACKWELLULTRA
        (nk_kernel_punned_t)&nk_attention_packed_bf16_blackwellultra,
#endif
    };
    static nk_kernel_punned_t const rocm[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_ROCM
        (nk_kernel_punned_t)&nk_attention_packed_bf16_rocm,
#endif
#if NUMKONG_TARGET_CDNA3
        (nk_kernel_punned_t)&nk_attention_packed_bf16_cdna3,
#endif
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_attention_packed_bf16_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_attention_packed_bf16_cdna5,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonbfdot_k * NUMKONG_TARGET_NEONBFDOT | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_genoa_k * NUMKONG_TARGET_GENOA | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_rvv_k * NUMKONG_TARGET_RVV | nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE |
             nk_cap_hopper_k * NUMKONG_TARGET_HOPPER | nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL |
             nk_cap_blackwellultra_k * NUMKONG_TARGET_BLACKWELLULTRA,
         cuda},
        {nk_cap_rocm_k * NUMKONG_TARGET_ROCM | nk_cap_cdna3_k * NUMKONG_TARGET_CDNA3 |
             nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5,
         rocm},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_attention_packed_bf16_best(nk_bf16_t const *queries, void const *key_value_packed,
                                                      nk_f32_t *output, nk_f32_t *log_sum_exp, nk_size_t head_count,
                                                      nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_u32_t const *query_offsets, nk_size_t query_stride,
                                                      nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before,
                                                      nk_size_t keys_after, nk_size_t tasks_begin, nk_size_t tasks_end,
                                                      nk_capability_t capabilities, nk_stream_t stream) {
    nk_attention_packed_punned_t const kernel = (nk_attention_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_attention_packed_bf16_capabilities());
    return kernel ? kernel(queries, key_value_packed, output, log_sum_exp, head_count, key_value_head_count, depth,
                           query_offsets, query_stride, output_stride, scale, keys_before, keys_after, tasks_begin,
                           tasks_end, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_attention_packed_f16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_attention_packed_f16_serial,
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_attention_packed_f16_cuda,
#endif
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_attention_packed_f16_ampere,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_attention_packed_f16_blackwell,
#endif
#if NUMKONG_TARGET_BLACKWELLULTRA
        (nk_kernel_punned_t)&nk_attention_packed_f16_blackwellultra,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE |
             nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL | nk_cap_blackwellultra_k * NUMKONG_TARGET_BLACKWELLULTRA,
         cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_attention_packed_f16_best(nk_f16_t const *queries, void const *key_value_packed,
                                                     nk_f32_t *output, nk_f32_t *log_sum_exp, nk_size_t head_count,
                                                     nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *query_offsets, nk_size_t query_stride,
                                                     nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before,
                                                     nk_size_t keys_after, nk_size_t tasks_begin, nk_size_t tasks_end,
                                                     nk_capability_t capabilities, nk_stream_t stream) {
    nk_attention_packed_punned_t const kernel = (nk_attention_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_attention_packed_f16_capabilities());
    return kernel ? kernel(queries, key_value_packed, output, log_sum_exp, head_count, key_value_head_count, depth,
                           query_offsets, query_stride, output_stride, scale, keys_before, keys_after, tasks_begin,
                           tasks_end, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_attention_packed_gradients_bf16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_attention_packed_gradients_bf16_serial,
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_attention_packed_gradients_bf16_cuda,
#endif
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_attention_packed_gradients_bf16_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_attention_packed_gradients_bf16_hopper,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_attention_packed_gradients_bf16_blackwell,
#endif
#if NUMKONG_TARGET_BLACKWELLULTRA
        (nk_kernel_punned_t)&nk_attention_packed_gradients_bf16_blackwellultra,
#endif
    };
    static nk_kernel_punned_t const rocm[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_ROCM
        (nk_kernel_punned_t)&nk_attention_packed_gradients_bf16_rocm,
#endif
#if NUMKONG_TARGET_CDNA3
        (nk_kernel_punned_t)&nk_attention_packed_gradients_bf16_cdna3,
#endif
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_attention_packed_gradients_bf16_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_attention_packed_gradients_bf16_cdna5,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k, cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE |
             nk_cap_hopper_k * NUMKONG_TARGET_HOPPER | nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL |
             nk_cap_blackwellultra_k * NUMKONG_TARGET_BLACKWELLULTRA,
         cuda},
        {nk_cap_rocm_k * NUMKONG_TARGET_ROCM | nk_cap_cdna3_k * NUMKONG_TARGET_CDNA3 |
             nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5,
         rocm},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_attention_packed_gradients_bf16_best(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t const *output, nk_f32_t const *output_gradient,
    nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient, nk_f32_t *key_gradient, nk_f32_t *value_gradient,
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_u32_t const *key_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_size_t query_gradient_stride,
    nk_size_t key_value_gradient_stride, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_size_t tasks_begin, nk_size_t tasks_end, nk_capability_t capabilities, nk_stream_t stream) {
    nk_attention_packed_gradients_punned_t const kernel = (nk_attention_packed_gradients_punned_t)nk_kernel_pick_(
        capabilities, nk_attention_packed_gradients_bf16_capabilities());
    return kernel ? kernel(queries, key_value_packed, output, output_gradient, log_sum_exp, query_gradient,
                           key_gradient, value_gradient, head_count, key_value_head_count, depth, query_offsets,
                           key_offsets, query_stride, output_stride, query_gradient_stride, key_value_gradient_stride,
                           scale, keys_before, keys_after, tasks_begin, tasks_end, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_attention_packed_e4m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_attention_packed_e4m3_serial,
#if NUMKONG_TARGET_NEONFHM
        (nk_kernel_punned_t)&nk_attention_packed_e4m3_neonfhm,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_attention_packed_e4m3_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_attention_packed_e4m3_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_attention_packed_e4m3_skylake,
#endif
#if NUMKONG_TARGET_GENOA
        (nk_kernel_punned_t)&nk_attention_packed_e4m3_genoa,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_attention_packed_e4m3_sapphireamx,
#endif
#if NUMKONG_TARGET_DIAMONDAMX
        (nk_kernel_punned_t)&nk_attention_packed_e4m3_diamondamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_attention_packed_e4m3_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_attention_packed_e4m3_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_attention_packed_e4m3_cuda,
#endif
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_attention_packed_e4m3_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_attention_packed_e4m3_hopper,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_attention_packed_e4m3_blackwell,
#endif
#if NUMKONG_TARGET_BLACKWELLRTX
        (nk_kernel_punned_t)&nk_attention_packed_e4m3_blackwellrtx,
#endif
#if NUMKONG_TARGET_BLACKWELLULTRA
        (nk_kernel_punned_t)&nk_attention_packed_e4m3_blackwellultra,
#endif
    };
    static nk_kernel_punned_t const rocm[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_ROCM
        (nk_kernel_punned_t)&nk_attention_packed_e4m3_rocm,
#endif
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_attention_packed_e4m3_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_attention_packed_e4m3_cdna5,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonfhm_k * NUMKONG_TARGET_NEONFHM | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE |
             nk_cap_genoa_k * NUMKONG_TARGET_GENOA | nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX |
             nk_cap_diamondamx_k * NUMKONG_TARGET_DIAMONDAMX | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE |
             nk_cap_hopper_k * NUMKONG_TARGET_HOPPER | nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL |
             nk_cap_blackwellrtx_k * NUMKONG_TARGET_BLACKWELLRTX |
             nk_cap_blackwellultra_k * NUMKONG_TARGET_BLACKWELLULTRA,
         cuda},
        {nk_cap_rocm_k * NUMKONG_TARGET_ROCM | nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 |
             nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5,
         rocm},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_attention_packed_e4m3_best(nk_e4m3_t const *queries, void const *key_value_packed,
                                                      nk_f32_t *output, nk_f32_t *log_sum_exp, nk_size_t head_count,
                                                      nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_u32_t const *query_offsets, nk_size_t query_stride,
                                                      nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before,
                                                      nk_size_t keys_after, nk_size_t tasks_begin, nk_size_t tasks_end,
                                                      nk_capability_t capabilities, nk_stream_t stream) {
    nk_attention_packed_punned_t const kernel = (nk_attention_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_attention_packed_e4m3_capabilities());
    return kernel ? kernel(queries, key_value_packed, output, log_sum_exp, head_count, key_value_head_count, depth,
                           query_offsets, query_stride, output_stride, scale, keys_before, keys_after, tasks_begin,
                           tasks_end, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_attention_packed_i8_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_attention_packed_i8_serial,
#if NUMKONG_TARGET_NEONSDOT
        (nk_kernel_punned_t)&nk_attention_packed_i8_neonsdot,
#endif
#if NUMKONG_TARGET_SME
        (nk_kernel_punned_t)&nk_attention_packed_i8_sme,
#endif
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_attention_packed_i8_haswell,
#endif
#if NUMKONG_TARGET_ICELAKE
        (nk_kernel_punned_t)&nk_attention_packed_i8_icelake,
#endif
#if NUMKONG_TARGET_SAPPHIREAMX
        (nk_kernel_punned_t)&nk_attention_packed_i8_sapphireamx,
#endif
#if NUMKONG_TARGET_RVV
        (nk_kernel_punned_t)&nk_attention_packed_i8_rvv,
#endif
#if NUMKONG_TARGET_V128RELAXED
        (nk_kernel_punned_t)&nk_attention_packed_i8_v128relaxed,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_attention_packed_i8_cuda,
#endif
#if NUMKONG_TARGET_AMPERE
        (nk_kernel_punned_t)&nk_attention_packed_i8_ampere,
#endif
#if NUMKONG_TARGET_HOPPER
        (nk_kernel_punned_t)&nk_attention_packed_i8_hopper,
#endif
#if NUMKONG_TARGET_BLACKWELL
        (nk_kernel_punned_t)&nk_attention_packed_i8_blackwell,
#endif
#if NUMKONG_TARGET_BLACKWELLULTRA
        (nk_kernel_punned_t)&nk_attention_packed_i8_blackwellultra,
#endif
    };
    static nk_kernel_punned_t const rocm[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_ROCM
        (nk_kernel_punned_t)&nk_attention_packed_i8_rocm,
#endif
#if NUMKONG_TARGET_CDNA3
        (nk_kernel_punned_t)&nk_attention_packed_i8_cdna3,
#endif
#if NUMKONG_TARGET_CDNA4
        (nk_kernel_punned_t)&nk_attention_packed_i8_cdna4,
#endif
#if NUMKONG_TARGET_CDNA5
        (nk_kernel_punned_t)&nk_attention_packed_i8_cdna5,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT | nk_cap_sme_k * NUMKONG_TARGET_SME |
             nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE |
             nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX | nk_cap_rvv_k * NUMKONG_TARGET_RVV |
             nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED,
         cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA | nk_cap_ampere_k * NUMKONG_TARGET_AMPERE |
             nk_cap_hopper_k * NUMKONG_TARGET_HOPPER | nk_cap_blackwell_k * NUMKONG_TARGET_BLACKWELL |
             nk_cap_blackwellultra_k * NUMKONG_TARGET_BLACKWELLULTRA,
         cuda},
        {nk_cap_rocm_k * NUMKONG_TARGET_ROCM | nk_cap_cdna3_k * NUMKONG_TARGET_CDNA3 |
             nk_cap_cdna4_k * NUMKONG_TARGET_CDNA4 | nk_cap_cdna5_k * NUMKONG_TARGET_CDNA5,
         rocm},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_attention_packed_i8_best(nk_i8_t const *queries, void const *key_value_packed,
                                                    nk_f32_t *output, nk_f32_t *log_sum_exp, nk_size_t head_count,
                                                    nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *query_offsets, nk_size_t query_stride,
                                                    nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before,
                                                    nk_size_t keys_after, nk_size_t tasks_begin, nk_size_t tasks_end,
                                                    nk_capability_t capabilities, nk_stream_t stream) {
    nk_attention_packed_punned_t const kernel = (nk_attention_packed_punned_t)nk_kernel_pick_(
        capabilities, nk_attention_packed_i8_capabilities());
    return kernel ? kernel(queries, key_value_packed, output, log_sum_exp, head_count, key_value_head_count, depth,
                           query_offsets, query_stride, output_stride, scale, keys_before, keys_after, tasks_begin,
                           tasks_end, stream)
                  : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_attention_rope_f32_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_attention_rope_f32_serial,
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_attention_rope_f32_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_attention_rope_f32_skylake,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_attention_rope_f32_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE, cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_attention_rope_f32_best(nk_f32_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                   nk_f32_t *y, nk_size_t rows, nk_size_t head_count, nk_size_t depth,
                                                   nk_size_t x_stride, nk_size_t y_stride, nk_capability_t capabilities,
                                                   nk_stream_t stream) {
    nk_attention_rope_punned_t const kernel = (nk_attention_rope_punned_t)nk_kernel_pick_(
        capabilities, nk_attention_rope_f32_capabilities());
    return kernel ? kernel(x, cos, sin, y, rows, head_count, depth, x_stride, y_stride, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_attention_rope_bf16_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_attention_rope_bf16_serial,
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_attention_rope_bf16_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_attention_rope_bf16_skylake,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_attention_rope_bf16_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE, cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_attention_rope_bf16_best(nk_bf16_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                    nk_bf16_t *y, nk_size_t rows, nk_size_t head_count, nk_size_t depth,
                                                    nk_size_t x_stride, nk_size_t y_stride,
                                                    nk_capability_t capabilities, nk_stream_t stream) {
    nk_attention_rope_punned_t const kernel = (nk_attention_rope_punned_t)nk_kernel_pick_(
        capabilities, nk_attention_rope_bf16_capabilities());
    return kernel ? kernel(x, cos, sin, y, rows, head_count, depth, x_stride, y_stride, stream) : nk_missing_kernel_k;
}

static nk_capability_kernels_t const *nk_attention_rope_e4m3_capabilities(void) {
    static nk_kernel_punned_t const cpu[] = {
        NUMKONG_NULL,
        (nk_kernel_punned_t)&nk_attention_rope_e4m3_serial,
#if NUMKONG_TARGET_HASWELL
        (nk_kernel_punned_t)&nk_attention_rope_e4m3_haswell,
#endif
#if NUMKONG_TARGET_SKYLAKE
        (nk_kernel_punned_t)&nk_attention_rope_e4m3_skylake,
#endif
    };
    static nk_kernel_punned_t const cuda[] = {
        NUMKONG_NULL,
#if NUMKONG_TARGET_CUDA
        (nk_kernel_punned_t)&nk_attention_rope_e4m3_cuda,
#endif
    };
    static nk_capability_kernels_t const lists[nk_capability_groups_k] = {
        {nk_cap_serial_k | nk_cap_haswell_k * NUMKONG_TARGET_HASWELL | nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE, cpu},
        {nk_cap_cuda_k * NUMKONG_TARGET_CUDA, cuda},
        {0, nk_no_kernels_},
        {0, nk_no_kernels_},
    };
    return lists;
}

NUMKONG_API nk_status_t nk_attention_rope_e4m3_best(nk_e4m3_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                    nk_e4m3_t *y, nk_size_t rows, nk_size_t head_count, nk_size_t depth,
                                                    nk_size_t x_stride, nk_size_t y_stride,
                                                    nk_capability_t capabilities, nk_stream_t stream) {
    nk_attention_rope_punned_t const kernel = (nk_attention_rope_punned_t)nk_kernel_pick_(
        capabilities, nk_attention_rope_e4m3_capabilities());
    return kernel ? kernel(x, cos, sin, y, rows, head_count, depth, x_stride, y_stride, stream) : nk_missing_kernel_k;
}

NUMKONG_API nk_status_t nk_attention_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                                 nk_kernel_punned_t *kernel, nk_capability_t *capability) {
    nk_capability_kernels_t const *lists = NUMKONG_NULL;
    switch (dtype) {
    case nk_f32_k:
        switch (kind) {
        case nk_kernel_attention_rope_k: lists = nk_attention_rope_f32_capabilities(); break;
        default: break;
        }
        break;
    case nk_bf16_k:
        switch (kind) {
        case nk_kernel_attention_pack_size_k: lists = nk_attention_pack_size_bf16_capabilities(); break;
        case nk_kernel_attention_packed_shape_k: lists = nk_attention_packed_shape_bf16_capabilities(); break;
        case nk_kernel_attention_pack_k: lists = nk_attention_pack_bf16_capabilities(); break;
        case nk_kernel_attention_packed_k: lists = nk_attention_packed_bf16_capabilities(); break;
        case nk_kernel_attention_packed_gradients_k: lists = nk_attention_packed_gradients_bf16_capabilities(); break;
        case nk_kernel_attention_rope_k: lists = nk_attention_rope_bf16_capabilities(); break;
        default: break;
        }
        break;
    case nk_f16_k:
        switch (kind) {
        case nk_kernel_attention_pack_size_k: lists = nk_attention_pack_size_f16_capabilities(); break;
        case nk_kernel_attention_packed_shape_k: lists = nk_attention_packed_shape_f16_capabilities(); break;
        case nk_kernel_attention_pack_k: lists = nk_attention_pack_f16_capabilities(); break;
        case nk_kernel_attention_packed_k: lists = nk_attention_packed_f16_capabilities(); break;
        default: break;
        }
        break;
    case nk_e4m3_k:
        switch (kind) {
        case nk_kernel_attention_pack_size_k: lists = nk_attention_pack_size_e4m3_capabilities(); break;
        case nk_kernel_attention_packed_shape_k: lists = nk_attention_packed_shape_e4m3_capabilities(); break;
        case nk_kernel_attention_pack_k: lists = nk_attention_pack_e4m3_capabilities(); break;
        case nk_kernel_attention_packed_k: lists = nk_attention_packed_e4m3_capabilities(); break;
        case nk_kernel_attention_rope_k: lists = nk_attention_rope_e4m3_capabilities(); break;
        default: break;
        }
        break;
    case nk_i8_k:
        switch (kind) {
        case nk_kernel_attention_pack_size_k: lists = nk_attention_pack_size_i8_capabilities(); break;
        case nk_kernel_attention_packed_shape_k: lists = nk_attention_packed_shape_i8_capabilities(); break;
        case nk_kernel_attention_pack_k: lists = nk_attention_pack_i8_capabilities(); break;
        case nk_kernel_attention_packed_k: lists = nk_attention_packed_i8_capabilities(); break;
        default: break;
        }
        break;
    default: break;
    }
    *kernel = lists ? nk_kernel_pick_(capabilities, lists) : (nk_kernel_punned_t)NUMKONG_NULL;
    *capability = *kernel ? nk_capability_pick_(capabilities, lists) : 0;
    return *kernel ? nk_success_k : nk_missing_kernel_k;
}
