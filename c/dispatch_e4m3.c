/**
 *  @brief Dispatch Initialization for E4M3 Data Types.
 *  @file c/dispatch_e4m3.c
 *  @author Ash Vardanian
 *  @date February 3, 2026
 */
#include "dispatch.h"

void nk_dispatch_e4m3_find_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m, nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_V128RELAXED
    if (v & nk_cap_v128relaxed_k) switch (k) {
        case nk_kernel_reduce_moments_k:
            *m = (m_t)&nk_reduce_moments_e4m3_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_e4m3_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e4m3_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e4m3_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e4m3_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e4m3_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_dots_packed_size_k:
            *m = (m_t)&nk_dots_packed_size_e4m3_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e4m3_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e4m3_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_dots_symmetric_k:
            *m = (m_t)&nk_dots_symmetric_e4m3_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_angulars_packed_k:
            *m = (m_t)&nk_angulars_packed_e4m3_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_e4m3_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_euclideans_packed_k:
            *m = (m_t)&nk_euclideans_packed_e4m3_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_e4m3_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_attention_packed_size_k:
            *m = (m_t)&nk_attention_packed_size_e4m3_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_attention_pack_k:
            *m = (m_t)&nk_attention_pack_e4m3_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_attention_packed_k:
            *m = (m_t)&nk_attention_packed_e4m3_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_SME
    if (v & nk_cap_sme_k) switch (k) {
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e4m3_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e4m3_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e4m3_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e4m3_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_e4m3_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_e4m3_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_e4m3_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_euclideans_symmetric_k: *m = (m_t)&nk_euclideans_symmetric_e4m3_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_attention_packed_size_k: *m = (m_t)&nk_attention_packed_size_e4m3_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_attention_pack_k: *m = (m_t)&nk_attention_pack_e4m3_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_attention_packed_k: *m = (m_t)&nk_attention_packed_e4m3_sme, *c = nk_cap_sme_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEONFP8
    if (v & nk_cap_neonfp8_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e4m3_neonfp8, *c = nk_cap_neonfp8_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e4m3_neonfp8, *c = nk_cap_neonfp8_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e4m3_neonfp8, *c = nk_cap_neonfp8_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e4m3_neonfp8, *c = nk_cap_neonfp8_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e4m3_neonfp8, *c = nk_cap_neonfp8_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e4m3_neonfp8, *c = nk_cap_neonfp8_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e4m3_neonfp8, *c = nk_cap_neonfp8_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e4m3_neonfp8, *c = nk_cap_neonfp8_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_e4m3_neonfp8, *c = nk_cap_neonfp8_k; return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_e4m3_neonfp8, *c = nk_cap_neonfp8_k;
            return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_e4m3_neonfp8, *c = nk_cap_neonfp8_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_e4m3_neonfp8, *c = nk_cap_neonfp8_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_NEONFHM
    if (v & nk_cap_neonfhm_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e4m3_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_e4m3_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e4m3_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e4m3_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e4m3_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e4m3_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_e4m3_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_e4m3_neonfhm, *c = nk_cap_neonfhm_k;
            return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_e4m3_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_e4m3_neonfhm, *c = nk_cap_neonfhm_k;
            return;
        case nk_kernel_attention_packed_size_k:
            *m = (m_t)&nk_attention_packed_size_e4m3_neonfhm, *c = nk_cap_neonfhm_k;
            return;
        case nk_kernel_attention_pack_k: *m = (m_t)&nk_attention_pack_e4m3_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_attention_packed_k: *m = (m_t)&nk_attention_packed_e4m3_neonfhm, *c = nk_cap_neonfhm_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEONBFDOT
    if (v & nk_cap_neonbfdot_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e4m3_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e4m3_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e4m3_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e4m3_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e4m3_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_e4m3_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_e4m3_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_e4m3_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_e4m3_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_DIAMONDAMX
    // Diamond Rapids AMX-FP8 (`_tile_dphf8ps`) runs E4M3 attention natively — preferred over the
    // Sapphire AMX path below, which widens FP8→BF16. Its I8/BF16 variants are Sapphire clones, so
    // only E4M3 is provided here.
    if (v & nk_cap_diamondamx_k) switch (k) {
        case nk_kernel_attention_packed_size_k:
            *m = (m_t)&nk_attention_packed_size_e4m3_diamondamx, *c = nk_cap_diamondamx_k;
            return;
        case nk_kernel_attention_pack_k: *m = (m_t)&nk_attention_pack_e4m3_diamondamx, *c = nk_cap_diamondamx_k; return;
        case nk_kernel_attention_packed_k:
            *m = (m_t)&nk_attention_packed_e4m3_diamondamx, *c = nk_cap_diamondamx_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_SAPPHIREAMX
    if (v & nk_cap_sapphireamx_k) switch (k) {
        case nk_kernel_dots_packed_size_k:
            *m = (m_t)&nk_dots_packed_size_e4m3_sapphireamx, *c = nk_cap_sapphireamx_k;
            return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e4m3_sapphireamx, *c = nk_cap_sapphireamx_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e4m3_sapphireamx, *c = nk_cap_sapphireamx_k; return;
        case nk_kernel_angulars_packed_k:
            *m = (m_t)&nk_angulars_packed_e4m3_sapphireamx, *c = nk_cap_sapphireamx_k;
            return;
        case nk_kernel_euclideans_packed_k:
            *m = (m_t)&nk_euclideans_packed_e4m3_sapphireamx, *c = nk_cap_sapphireamx_k;
            return;
        case nk_kernel_dots_symmetric_k:
            *m = (m_t)&nk_dots_symmetric_e4m3_sapphireamx, *c = nk_cap_sapphireamx_k;
            return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_e4m3_sapphireamx, *c = nk_cap_sapphireamx_k;
            return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_e4m3_sapphireamx, *c = nk_cap_sapphireamx_k;
            return;
        case nk_kernel_attention_packed_size_k:
            *m = (m_t)&nk_attention_packed_size_e4m3_sapphireamx, *c = nk_cap_sapphireamx_k;
            return;
        case nk_kernel_attention_pack_k:
            *m = (m_t)&nk_attention_pack_e4m3_sapphireamx, *c = nk_cap_sapphireamx_k;
            return;
        case nk_kernel_attention_packed_k:
            *m = (m_t)&nk_attention_packed_e4m3_sapphireamx, *c = nk_cap_sapphireamx_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_DIAMOND
    if (v & nk_cap_diamond_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e4m3_diamond, *c = nk_cap_diamond_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e4m3_diamond, *c = nk_cap_diamond_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e4m3_diamond, *c = nk_cap_diamond_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e4m3_diamond, *c = nk_cap_diamond_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e4m3_diamond, *c = nk_cap_diamond_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e4m3_diamond, *c = nk_cap_diamond_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e4m3_diamond, *c = nk_cap_diamond_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e4m3_diamond, *c = nk_cap_diamond_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_e4m3_diamond, *c = nk_cap_diamond_k; return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_e4m3_diamond, *c = nk_cap_diamond_k;
            return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_e4m3_diamond, *c = nk_cap_diamond_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_e4m3_diamond, *c = nk_cap_diamond_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_ICELAKE
    if (v & nk_cap_icelake_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e4m3_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e4m3_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e4m3_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e4m3_icelake, *c = nk_cap_icelake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_GENOA
    if (v & nk_cap_genoa_k) switch (k) {
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e4m3_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e4m3_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e4m3_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e4m3_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_e4m3_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_e4m3_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_e4m3_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_e4m3_genoa, *c = nk_cap_genoa_k;
            return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_e4m3_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_reduce_rmsnorm_k: *m = (m_t)&nk_reduce_rmsnorm_e4m3_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_attention_packed_size_k:
            *m = (m_t)&nk_attention_packed_size_e4m3_genoa, *c = nk_cap_genoa_k;
            return;
        case nk_kernel_attention_pack_k: *m = (m_t)&nk_attention_pack_e4m3_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_attention_packed_k: *m = (m_t)&nk_attention_packed_e4m3_genoa, *c = nk_cap_genoa_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e4m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e4m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e4m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e4m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e4m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e4m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e4m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e4m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_e4m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_e4m3_skylake, *c = nk_cap_skylake_k;
            return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_e4m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_e4m3_skylake, *c = nk_cap_skylake_k;
            return;
        case nk_kernel_reduce_rmsnorm_k: *m = (m_t)&nk_reduce_rmsnorm_e4m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_swiglu_k: *m = (m_t)&nk_each_swiglu_e4m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_trig_rope_k: *m = (m_t)&nk_trig_rope_e4m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_e4m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_e4m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_attention_packed_size_k:
            *m = (m_t)&nk_attention_packed_size_e4m3_skylake, *c = nk_cap_skylake_k;
            return;
        case nk_kernel_attention_pack_k: *m = (m_t)&nk_attention_pack_e4m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_attention_packed_k: *m = (m_t)&nk_attention_packed_e4m3_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e4m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_e4m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_e4m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_rmsnorm_k: *m = (m_t)&nk_reduce_rmsnorm_e4m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_swiglu_k: *m = (m_t)&nk_each_swiglu_e4m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_trig_rope_k: *m = (m_t)&nk_trig_rope_e4m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e4m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e4m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e4m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e4m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_e4m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_e4m3_haswell, *c = nk_cap_haswell_k;
            return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_e4m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_e4m3_haswell, *c = nk_cap_haswell_k;
            return;
        case nk_kernel_attention_packed_size_k:
            *m = (m_t)&nk_attention_packed_size_e4m3_haswell, *c = nk_cap_haswell_k;
            return;
        case nk_kernel_attention_pack_k: *m = (m_t)&nk_attention_pack_e4m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_attention_packed_k: *m = (m_t)&nk_attention_packed_e4m3_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
#if NK_TARGET_RVVHALF
    if (v & nk_cap_rvvhalf_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e4m3_rvvhalf, *c = nk_cap_rvvhalf_k; return;
        default: break;
        }
#endif
#if NK_TARGET_RVVBF16
    if (v & nk_cap_rvvbf16_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e4m3_rvvbf16, *c = nk_cap_rvvbf16_k; return;
        default: break;
        }
#endif
#if NK_TARGET_RVV
    if (v & nk_cap_rvv_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e4m3_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e4m3_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e4m3_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e4m3_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_e4m3_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_e4m3_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e4m3_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e4m3_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e4m3_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e4m3_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_e4m3_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_e4m3_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_e4m3_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_euclideans_symmetric_k: *m = (m_t)&nk_euclideans_symmetric_e4m3_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_e4m3_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_e4m3_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_e4m3_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_e4m3_rvv, *c = nk_cap_rvv_k; return;
        default: break;
        }
#endif
#if NK_TARGET_RVV
    if (v & nk_cap_rvv_k) switch (k) {
        case nk_kernel_attention_packed_size_k: *m = (m_t)&nk_attention_packed_size_e4m3_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_attention_pack_k: *m = (m_t)&nk_attention_pack_e4m3_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_attention_packed_k: *m = (m_t)&nk_attention_packed_e4m3_rvv, *c = nk_cap_rvv_k; return;
        default: break;
        }
#endif

    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e4m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e4m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e4m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e4m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_e4m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_e4m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_rmsnorm_k: *m = (m_t)&nk_reduce_rmsnorm_e4m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_e4m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_swiglu_k: *m = (m_t)&nk_each_swiglu_e4m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_trig_rope_k: *m = (m_t)&nk_trig_rope_e4m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_e4m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_e4m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_e4m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e4m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e4m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e4m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e4m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_e4m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_e4m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_e4m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_e4m3_serial, *c = nk_cap_serial_k;
            return;
        case nk_kernel_attention_packed_size_k:
            *m = (m_t)&nk_attention_packed_size_e4m3_serial, *c = nk_cap_serial_k;
            return;
        case nk_kernel_attention_pack_k: *m = (m_t)&nk_attention_pack_e4m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_attention_packed_k: *m = (m_t)&nk_attention_packed_e4m3_serial, *c = nk_cap_serial_k; return;
        default: break;
        }

    // Error fallback - zero capability signals lookup failure
    *m = (m_t)nk_error_dense_, *c = 0;
}

void nk_dispatch_e4m3_init_(nk_capability_t caps) {
    nk_implementations_t *t = &nk_dispatch_table;
    nk_capability_t used;

    nk_dispatch_e4m3_find_(caps, nk_kernel_dot_k, (nk_kernel_punned_t *)&t->dot_e4m3, &used);
    nk_dispatch_e4m3_find_(caps, nk_kernel_angular_k, (nk_kernel_punned_t *)&t->angular_e4m3, &used);
    nk_dispatch_e4m3_find_(caps, nk_kernel_euclidean_k, (nk_kernel_punned_t *)&t->euclidean_e4m3, &used);
    nk_dispatch_e4m3_find_(caps, nk_kernel_sqeuclidean_k, (nk_kernel_punned_t *)&t->sqeuclidean_e4m3, &used);
    nk_dispatch_e4m3_find_(caps, nk_kernel_each_scale_k, (nk_kernel_punned_t *)&t->each_scale_e4m3, &used);
    nk_dispatch_e4m3_find_(caps, nk_kernel_each_sum_k, (nk_kernel_punned_t *)&t->each_sum_e4m3, &used);
    nk_dispatch_e4m3_find_(caps, nk_kernel_each_blend_k, (nk_kernel_punned_t *)&t->each_blend_e4m3, &used);
    nk_dispatch_e4m3_find_(caps, nk_kernel_each_fma_k, (nk_kernel_punned_t *)&t->each_fma_e4m3, &used);
    nk_dispatch_e4m3_find_(caps, nk_kernel_each_swiglu_k, (nk_kernel_punned_t *)&t->each_swiglu_e4m3, &used);
    nk_dispatch_e4m3_find_(caps, nk_kernel_trig_rope_k, (nk_kernel_punned_t *)&t->trig_rope_e4m3, &used);
    nk_dispatch_e4m3_find_(caps, nk_kernel_reduce_moments_k, (nk_kernel_punned_t *)&t->reduce_moments_e4m3, &used);
    nk_dispatch_e4m3_find_(caps, nk_kernel_reduce_minmax_k, (nk_kernel_punned_t *)&t->reduce_minmax_e4m3, &used);
    nk_dispatch_e4m3_find_(caps, nk_kernel_reduce_rmsnorm_k, (nk_kernel_punned_t *)&t->reduce_rmsnorm_e4m3, &used);
    nk_dispatch_e4m3_find_(caps, nk_kernel_dots_packed_size_k, (nk_kernel_punned_t *)&t->dots_packed_size_e4m3, &used);
    nk_dispatch_e4m3_find_(caps, nk_kernel_dots_pack_k, (nk_kernel_punned_t *)&t->dots_pack_e4m3, &used);
    nk_dispatch_e4m3_find_(caps, nk_kernel_dots_packed_k, (nk_kernel_punned_t *)&t->dots_packed_e4m3, &used);
    nk_dispatch_e4m3_find_(caps, nk_kernel_dots_symmetric_k, (nk_kernel_punned_t *)&t->dots_symmetric_e4m3, &used);
    nk_dispatch_e4m3_find_(caps, nk_kernel_angulars_packed_k, (nk_kernel_punned_t *)&t->angulars_packed_e4m3, &used);
    nk_dispatch_e4m3_find_(caps, nk_kernel_angulars_symmetric_k, (nk_kernel_punned_t *)&t->angulars_symmetric_e4m3,
                           &used);
    nk_dispatch_e4m3_find_(caps, nk_kernel_euclideans_packed_k, (nk_kernel_punned_t *)&t->euclideans_packed_e4m3,
                           &used);
    nk_dispatch_e4m3_find_(caps, nk_kernel_euclideans_symmetric_k, (nk_kernel_punned_t *)&t->euclideans_symmetric_e4m3,
                           &used);
    nk_dispatch_e4m3_find_(caps, nk_kernel_attention_packed_size_k,
                           (nk_kernel_punned_t *)&t->attention_packed_size_e4m3, &used);
    nk_dispatch_e4m3_find_(caps, nk_kernel_attention_pack_k, (nk_kernel_punned_t *)&t->attention_pack_e4m3, &used);
    nk_dispatch_e4m3_find_(caps, nk_kernel_attention_packed_k, (nk_kernel_punned_t *)&t->attention_packed_e4m3, &used);
}
