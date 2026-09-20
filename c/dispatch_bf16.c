/**
 *  @brief Dispatch Initialization for BF16 Data Types.
 *  @file c/dispatch_bf16.c
 *  @author Ash Vardanian
 *  @date February 3, 2026
 */
#include "dispatch.h"

void nk_dispatch_bf16_find_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m, nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;

#if NK_TARGET_V128RELAXED
    if (v & nk_cap_v128relaxed_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_bf16_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_bf16_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_bf16_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_bf16_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_reduce_moments_k:
            *m = (m_t)&nk_reduce_moments_bf16_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_bf16_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_dots_packed_size_k:
            *m = (m_t)&nk_dots_packed_size_bf16_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_bf16_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_bf16_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_dots_symmetric_k:
            *m = (m_t)&nk_dots_symmetric_bf16_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_angulars_packed_k:
            *m = (m_t)&nk_angulars_packed_bf16_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_bf16_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_euclideans_packed_k:
            *m = (m_t)&nk_euclideans_packed_bf16_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_bf16_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_maxsim_packed_size_k:
            *m = (m_t)&nk_maxsim_packed_size_bf16_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_maxsim_pack_k: *m = (m_t)&nk_maxsim_pack_bf16_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_maxsim_packed_k: *m = (m_t)&nk_maxsim_packed_bf16_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_bf16_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_bf16_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_bf16_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_bf16_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SME
    if (v & nk_cap_sme_k) switch (k) {
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_bf16_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_bf16_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_bf16_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_bf16_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_bf16_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_bf16_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_bf16_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_euclideans_symmetric_k: *m = (m_t)&nk_euclideans_symmetric_bf16_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_maxsim_packed_size_k: *m = (m_t)&nk_maxsim_packed_size_bf16_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_maxsim_pack_k: *m = (m_t)&nk_maxsim_pack_bf16_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_maxsim_packed_k: *m = (m_t)&nk_maxsim_packed_bf16_sme, *c = nk_cap_sme_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SVE2
    if (v & nk_cap_sve2_k) switch (k) {
        case nk_kernel_sparse_dot_k: *m = (m_t)&nk_sparse_dot_u16bf16_sve2, *c = nk_cap_sve2_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SVEBFDOT
    if (v & nk_cap_svebfdot_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_bf16_svebfdot, *c = nk_cap_svebfdot_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_bf16_svebfdot, *c = nk_cap_svebfdot_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_bf16_svebfdot, *c = nk_cap_svebfdot_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_bf16_svebfdot, *c = nk_cap_svebfdot_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEONBFDOT
    if (v & nk_cap_neonbfdot_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_bf16_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_bf16_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_bf16_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_bf16_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_bf16_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        case nk_kernel_mahalanobis_k: *m = (m_t)&nk_mahalanobis_bf16_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        case nk_kernel_rmsd_k: *m = (m_t)&nk_rmsd_bf16_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        case nk_kernel_kabsch_k: *m = (m_t)&nk_kabsch_bf16_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        case nk_kernel_umeyama_k: *m = (m_t)&nk_umeyama_bf16_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_bf16_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_bf16_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_bf16_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_bf16_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_bf16_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        case nk_kernel_dots_packed_size_k:
            *m = (m_t)&nk_dots_packed_size_bf16_neonbfdot, *c = nk_cap_neonbfdot_k;
            return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_bf16_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_bf16_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_bf16_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_bf16_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_bf16_neonbfdot, *c = nk_cap_neonbfdot_k;
            return;
        case nk_kernel_euclideans_packed_k:
            *m = (m_t)&nk_euclideans_packed_bf16_neonbfdot, *c = nk_cap_neonbfdot_k;
            return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_bf16_neonbfdot, *c = nk_cap_neonbfdot_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_bf16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_bf16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_bf16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_bf16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_bf16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_bf16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_bf16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_bf16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_bf16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_bf16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_bf16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_euclideans_symmetric_k: *m = (m_t)&nk_euclideans_symmetric_bf16_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SAPPHIREAMX
    if (v & nk_cap_sapphireamx_k) switch (k) {
        case nk_kernel_dots_packed_size_k:
            *m = (m_t)&nk_dots_packed_size_bf16_sapphireamx, *c = nk_cap_sapphireamx_k;
            return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_bf16_sapphireamx, *c = nk_cap_sapphireamx_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_bf16_sapphireamx, *c = nk_cap_sapphireamx_k; return;
        case nk_kernel_dots_symmetric_k:
            *m = (m_t)&nk_dots_symmetric_bf16_sapphireamx, *c = nk_cap_sapphireamx_k;
            return;
        case nk_kernel_angulars_packed_k:
            *m = (m_t)&nk_angulars_packed_bf16_sapphireamx, *c = nk_cap_sapphireamx_k;
            return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_bf16_sapphireamx, *c = nk_cap_sapphireamx_k;
            return;
        case nk_kernel_euclideans_packed_k:
            *m = (m_t)&nk_euclideans_packed_bf16_sapphireamx, *c = nk_cap_sapphireamx_k;
            return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_bf16_sapphireamx, *c = nk_cap_sapphireamx_k;
            return;
        case nk_kernel_maxsim_packed_size_k:
            *m = (m_t)&nk_maxsim_packed_size_bf16_sapphireamx, *c = nk_cap_sapphireamx_k;
            return;
        case nk_kernel_maxsim_pack_k: *m = (m_t)&nk_maxsim_pack_bf16_sapphireamx, *c = nk_cap_sapphireamx_k; return;
        case nk_kernel_maxsim_packed_k: *m = (m_t)&nk_maxsim_packed_bf16_sapphireamx, *c = nk_cap_sapphireamx_k; return;
        default: break;
        }
#endif
#if NK_TARGET_TURIN
    if (v & nk_cap_turin_k) switch (k) {
        case nk_kernel_sparse_dot_k: *m = (m_t)&nk_sparse_dot_u16bf16_turin, *c = nk_cap_turin_k; return;
        default: break;
        }
#endif
#if NK_TARGET_GENOA
    if (v & nk_cap_genoa_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_bf16_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_bf16_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_bf16_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_bf16_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_bf16_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_mahalanobis_k: *m = (m_t)&nk_mahalanobis_bf16_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_bf16_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_bf16_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_bf16_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_bf16_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_bf16_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_bf16_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_bf16_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_bf16_genoa, *c = nk_cap_genoa_k;
            return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_bf16_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_maxsim_packed_size_k: *m = (m_t)&nk_maxsim_packed_size_bf16_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_maxsim_pack_k: *m = (m_t)&nk_maxsim_pack_bf16_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_maxsim_packed_k: *m = (m_t)&nk_maxsim_packed_bf16_genoa, *c = nk_cap_genoa_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_bf16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_bf16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_bf16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_bf16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_bf16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_bf16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_bf16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_bf16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_bf16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_bf16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_bf16_skylake, *c = nk_cap_skylake_k;
            return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_bf16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_bf16_skylake, *c = nk_cap_skylake_k;
            return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_bf16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_bf16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_rmsd_k: *m = (m_t)&nk_rmsd_bf16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_kabsch_k: *m = (m_t)&nk_kabsch_bf16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_umeyama_k: *m = (m_t)&nk_umeyama_bf16_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_ALDER
    if (v & nk_cap_alder_k) switch (k) {
        case nk_kernel_maxsim_packed_size_k: *m = (m_t)&nk_maxsim_packed_size_bf16_alder, *c = nk_cap_alder_k; return;
        case nk_kernel_maxsim_pack_k: *m = (m_t)&nk_maxsim_pack_bf16_alder, *c = nk_cap_alder_k; return;
        case nk_kernel_maxsim_packed_k: *m = (m_t)&nk_maxsim_packed_bf16_alder, *c = nk_cap_alder_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_mahalanobis_k: *m = (m_t)&nk_mahalanobis_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_rmsd_k: *m = (m_t)&nk_rmsd_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_kabsch_k: *m = (m_t)&nk_kabsch_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_umeyama_k: *m = (m_t)&nk_umeyama_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_bf16_haswell, *c = nk_cap_haswell_k;
            return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_bf16_haswell, *c = nk_cap_haswell_k;
            return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_maxsim_packed_size_k:
            *m = (m_t)&nk_maxsim_packed_size_bf16_haswell, *c = nk_cap_haswell_k;
            return;
        case nk_kernel_maxsim_pack_k: *m = (m_t)&nk_maxsim_pack_bf16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_maxsim_packed_k: *m = (m_t)&nk_maxsim_packed_bf16_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEONSDOT
    if (v & nk_cap_neonsdot_k) switch (k) {
        case nk_kernel_maxsim_packed_size_k:
            *m = (m_t)&nk_maxsim_packed_size_bf16_neonsdot, *c = nk_cap_neonsdot_k;
            return;
        case nk_kernel_maxsim_pack_k: *m = (m_t)&nk_maxsim_pack_bf16_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_maxsim_packed_k: *m = (m_t)&nk_maxsim_packed_bf16_neonsdot, *c = nk_cap_neonsdot_k; return;
        default: break;
        }
#endif
#if NK_TARGET_RVVBF16
    if (v & nk_cap_rvvbf16_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_bf16_rvvbf16, *c = nk_cap_rvvbf16_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_bf16_rvvbf16, *c = nk_cap_rvvbf16_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_bf16_rvvbf16, *c = nk_cap_rvvbf16_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_bf16_rvvbf16, *c = nk_cap_rvvbf16_k; return;
        default: break;
        }
#endif
#if NK_TARGET_RVV
    if (v & nk_cap_rvv_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_bf16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_bf16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_bf16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_bf16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_bf16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_bf16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_kld_k: *m = (m_t)&nk_kld_bf16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_jsd_k: *m = (m_t)&nk_jsd_bf16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_bf16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_bf16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_bf16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_bf16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_bf16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_bf16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_bf16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_euclideans_symmetric_k: *m = (m_t)&nk_euclideans_symmetric_bf16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_bf16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_bf16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_bf16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_bf16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_bf16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_mahalanobis_k: *m = (m_t)&nk_mahalanobis_bf16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_rmsd_k: *m = (m_t)&nk_rmsd_bf16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_kabsch_k: *m = (m_t)&nk_kabsch_bf16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_umeyama_k: *m = (m_t)&nk_umeyama_bf16_rvv, *c = nk_cap_rvv_k; return;
        default: break;
        }
#endif
#if NK_TARGET_POWERVSX
    if (v & nk_cap_powervsx_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_bf16_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_bf16_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_bf16_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_bf16_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_bf16_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_bf16_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_bf16_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_bf16_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_bf16_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_bf16_powervsx, *c = nk_cap_powervsx_k;
            return;
        case nk_kernel_euclideans_packed_k:
            *m = (m_t)&nk_euclideans_packed_bf16_powervsx, *c = nk_cap_powervsx_k;
            return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_bf16_powervsx, *c = nk_cap_powervsx_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_LOONGSONASX
    if (v & nk_cap_loongsonasx_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_bf16_loongsonasx, *c = nk_cap_loongsonasx_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_bf16_loongsonasx, *c = nk_cap_loongsonasx_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_bf16_loongsonasx, *c = nk_cap_loongsonasx_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_bf16_loongsonasx, *c = nk_cap_loongsonasx_k; return;
        case nk_kernel_dots_packed_size_k:
            *m = (m_t)&nk_dots_packed_size_bf16_loongsonasx, *c = nk_cap_loongsonasx_k;
            return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_bf16_loongsonasx, *c = nk_cap_loongsonasx_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_bf16_loongsonasx, *c = nk_cap_loongsonasx_k; return;
        case nk_kernel_dots_symmetric_k:
            *m = (m_t)&nk_dots_symmetric_bf16_loongsonasx, *c = nk_cap_loongsonasx_k;
            return;
        case nk_kernel_angulars_packed_k:
            *m = (m_t)&nk_angulars_packed_bf16_loongsonasx, *c = nk_cap_loongsonasx_k;
            return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_bf16_loongsonasx, *c = nk_cap_loongsonasx_k;
            return;
        case nk_kernel_euclideans_packed_k:
            *m = (m_t)&nk_euclideans_packed_bf16_loongsonasx, *c = nk_cap_loongsonasx_k;
            return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_bf16_loongsonasx, *c = nk_cap_loongsonasx_k;
            return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_jsd_k: *m = (m_t)&nk_jsd_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_kld_k: *m = (m_t)&nk_kld_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_mahalanobis_k: *m = (m_t)&nk_mahalanobis_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_rmsd_k: *m = (m_t)&nk_rmsd_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_kabsch_k: *m = (m_t)&nk_kabsch_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_umeyama_k: *m = (m_t)&nk_umeyama_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_sparse_dot_k: *m = (m_t)&nk_sparse_dot_u16bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_bf16_serial, *c = nk_cap_serial_k;
            return;
        case nk_kernel_maxsim_packed_size_k: *m = (m_t)&nk_maxsim_packed_size_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_maxsim_pack_k: *m = (m_t)&nk_maxsim_pack_bf16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_maxsim_packed_k: *m = (m_t)&nk_maxsim_packed_bf16_serial, *c = nk_cap_serial_k; return;
        default: break;
        }

    // Error fallback - zero capability signals lookup failure
    *m = (m_t)nk_error_dense_, *c = 0;
}

void nk_dispatch_bf16_init_(nk_capability_t caps) {
    nk_implementations_t *t = &nk_dispatch_table;
    nk_capability_t used;

    nk_dispatch_bf16_find_(caps, nk_kernel_dot_k, (nk_kernel_punned_t *)&t->dot_bf16, &used);
    nk_dispatch_bf16_find_(caps, nk_kernel_angular_k, (nk_kernel_punned_t *)&t->angular_bf16, &used);
    nk_dispatch_bf16_find_(caps, nk_kernel_euclidean_k, (nk_kernel_punned_t *)&t->euclidean_bf16, &used);
    nk_dispatch_bf16_find_(caps, nk_kernel_sqeuclidean_k, (nk_kernel_punned_t *)&t->sqeuclidean_bf16, &used);
    nk_dispatch_bf16_find_(caps, nk_kernel_bilinear_k, (nk_kernel_punned_t *)&t->bilinear_bf16, &used);
    nk_dispatch_bf16_find_(caps, nk_kernel_mahalanobis_k, (nk_kernel_punned_t *)&t->mahalanobis_bf16, &used);
    nk_dispatch_bf16_find_(caps, nk_kernel_kld_k, (nk_kernel_punned_t *)&t->kld_bf16, &used);
    nk_dispatch_bf16_find_(caps, nk_kernel_jsd_k, (nk_kernel_punned_t *)&t->jsd_bf16, &used);
    nk_dispatch_bf16_find_(caps, nk_kernel_rmsd_k, (nk_kernel_punned_t *)&t->rmsd_bf16, &used);
    nk_dispatch_bf16_find_(caps, nk_kernel_kabsch_k, (nk_kernel_punned_t *)&t->kabsch_bf16, &used);
    nk_dispatch_bf16_find_(caps, nk_kernel_umeyama_k, (nk_kernel_punned_t *)&t->umeyama_bf16, &used);
    nk_dispatch_bf16_find_(caps, nk_kernel_sparse_dot_k, (nk_kernel_punned_t *)&t->sparse_dot_u16bf16, &used);
    nk_dispatch_bf16_find_(caps, nk_kernel_each_scale_k, (nk_kernel_punned_t *)&t->each_scale_bf16, &used);
    nk_dispatch_bf16_find_(caps, nk_kernel_each_sum_k, (nk_kernel_punned_t *)&t->each_sum_bf16, &used);
    nk_dispatch_bf16_find_(caps, nk_kernel_each_blend_k, (nk_kernel_punned_t *)&t->each_blend_bf16, &used);
    nk_dispatch_bf16_find_(caps, nk_kernel_each_fma_k, (nk_kernel_punned_t *)&t->each_fma_bf16, &used);
    nk_dispatch_bf16_find_(caps, nk_kernel_reduce_moments_k, (nk_kernel_punned_t *)&t->reduce_moments_bf16, &used);
    nk_dispatch_bf16_find_(caps, nk_kernel_reduce_minmax_k, (nk_kernel_punned_t *)&t->reduce_minmax_bf16, &used);
    nk_dispatch_bf16_find_(caps, nk_kernel_dots_packed_size_k, (nk_kernel_punned_t *)&t->dots_packed_size_bf16, &used);
    nk_dispatch_bf16_find_(caps, nk_kernel_dots_pack_k, (nk_kernel_punned_t *)&t->dots_pack_bf16, &used);
    nk_dispatch_bf16_find_(caps, nk_kernel_dots_packed_k, (nk_kernel_punned_t *)&t->dots_packed_bf16, &used);
    nk_dispatch_bf16_find_(caps, nk_kernel_dots_symmetric_k, (nk_kernel_punned_t *)&t->dots_symmetric_bf16, &used);
    nk_dispatch_bf16_find_(caps, nk_kernel_angulars_packed_k, (nk_kernel_punned_t *)&t->angulars_packed_bf16, &used);
    nk_dispatch_bf16_find_(caps, nk_kernel_angulars_symmetric_k, (nk_kernel_punned_t *)&t->angulars_symmetric_bf16,
                           &used);
    nk_dispatch_bf16_find_(caps, nk_kernel_euclideans_packed_k, (nk_kernel_punned_t *)&t->euclideans_packed_bf16,
                           &used);
    nk_dispatch_bf16_find_(caps, nk_kernel_euclideans_symmetric_k, (nk_kernel_punned_t *)&t->euclideans_symmetric_bf16,
                           &used);
    nk_dispatch_bf16_find_(caps, nk_kernel_maxsim_packed_size_k, (nk_kernel_punned_t *)&t->maxsim_packed_size_bf16,
                           &used);
    nk_dispatch_bf16_find_(caps, nk_kernel_maxsim_pack_k, (nk_kernel_punned_t *)&t->maxsim_pack_bf16, &used);
    nk_dispatch_bf16_find_(caps, nk_kernel_maxsim_packed_k, (nk_kernel_punned_t *)&t->maxsim_packed_bf16, &used);
}
