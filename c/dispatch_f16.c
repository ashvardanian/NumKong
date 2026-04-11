/**
 *  @brief Dispatch Initialization for F16 Data Types.
 *  @file c/dispatch_f16.c
 *  @author Ash Vardanian
 *  @date February 3, 2026
 */
#include "dispatch.h"

void nk_dispatch_f16_find_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m, nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_V128RELAXED
    if (v & nk_cap_v128relaxed_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f16_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f16_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f16_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f16_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_reduce_moments_k:
            *m = (m_t)&nk_reduce_moments_f16_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_f16_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_maxsim_packed_size_k:
            *m = (m_t)&nk_maxsim_packed_size_f16_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_maxsim_pack_k: *m = (m_t)&nk_maxsim_pack_f16_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_maxsim_packed_k: *m = (m_t)&nk_maxsim_packed_f16_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_dots_packed_size_k:
            *m = (m_t)&nk_dots_packed_size_f16_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f16_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_f16_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_dots_symmetric_k:
            *m = (m_t)&nk_dots_symmetric_f16_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_angulars_packed_k:
            *m = (m_t)&nk_angulars_packed_f16_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_f16_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_euclideans_packed_k:
            *m = (m_t)&nk_euclideans_packed_f16_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_f16_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_f16_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_f16_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_f16_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_f16_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SME
    if (v & nk_cap_sme_k) switch (k) {
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_f16_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f16_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_f16_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f16_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_f16_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_f16_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_f16_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_euclideans_symmetric_k: *m = (m_t)&nk_euclideans_symmetric_f16_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_maxsim_packed_size_k: *m = (m_t)&nk_maxsim_packed_size_f16_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_maxsim_pack_k: *m = (m_t)&nk_maxsim_pack_f16_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_maxsim_packed_k: *m = (m_t)&nk_maxsim_packed_f16_sme, *c = nk_cap_sme_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SVEHALF
    if (v & nk_cap_svehalf_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f16_svehalf, *c = nk_cap_svehalf_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f16_svehalf, *c = nk_cap_svehalf_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f16_svehalf, *c = nk_cap_svehalf_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f16_svehalf, *c = nk_cap_svehalf_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEONFHM
    if (v & nk_cap_neonfhm_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f16_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_f16_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f16_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_f16_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f16_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_f16_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_f16_neonfhm, *c = nk_cap_neonfhm_k;
            return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_f16_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_f16_neonfhm, *c = nk_cap_neonfhm_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_NEONHALF
    if (v & nk_cap_neonhalf_k) switch (k) {
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_f16_neonhalf, *c = nk_cap_neonhalf_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_f16_neonhalf, *c = nk_cap_neonhalf_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_f16_neonhalf, *c = nk_cap_neonhalf_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_f16_neonhalf, *c = nk_cap_neonhalf_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_jsd_k: *m = (m_t)&nk_jsd_f16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_kld_k: *m = (m_t)&nk_kld_f16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_mahalanobis_k: *m = (m_t)&nk_mahalanobis_f16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_kabsch_k: *m = (m_t)&nk_kabsch_f16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_rmsd_k: *m = (m_t)&nk_rmsd_f16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_umeyama_k: *m = (m_t)&nk_umeyama_f16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_f16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_f16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_f16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_f16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_f16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_f16_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_euclideans_symmetric_k: *m = (m_t)&nk_euclideans_symmetric_f16_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEONSDOT
    if (v & nk_cap_neonsdot_k) switch (k) {
        case nk_kernel_maxsim_packed_size_k:
            *m = (m_t)&nk_maxsim_packed_size_f16_neonsdot, *c = nk_cap_neonsdot_k;
            return;
        case nk_kernel_maxsim_pack_k: *m = (m_t)&nk_maxsim_pack_f16_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_maxsim_packed_k: *m = (m_t)&nk_maxsim_packed_f16_neonsdot, *c = nk_cap_neonsdot_k; return;
        default: break;
        }
#endif
#if NK_TARGET_GRANITEAMX
    if (v & nk_cap_graniteamx_k) switch (k) {
        case nk_kernel_dots_packed_size_k:
            *m = (m_t)&nk_dots_packed_size_f16_graniteamx, *c = nk_cap_graniteamx_k;
            return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f16_graniteamx, *c = nk_cap_graniteamx_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_f16_graniteamx, *c = nk_cap_graniteamx_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f16_graniteamx, *c = nk_cap_graniteamx_k; return;
        case nk_kernel_angulars_packed_k:
            *m = (m_t)&nk_angulars_packed_f16_graniteamx, *c = nk_cap_graniteamx_k;
            return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_f16_graniteamx, *c = nk_cap_graniteamx_k;
            return;
        case nk_kernel_euclideans_packed_k:
            *m = (m_t)&nk_euclideans_packed_f16_graniteamx, *c = nk_cap_graniteamx_k;
            return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_f16_graniteamx, *c = nk_cap_graniteamx_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_SAPPHIREAMX
    if (v & nk_cap_sapphireamx_k) switch (k) {
        case nk_kernel_maxsim_packed_size_k:
            *m = (m_t)&nk_maxsim_packed_size_f16_sapphireamx, *c = nk_cap_sapphireamx_k;
            return;
        case nk_kernel_maxsim_pack_k: *m = (m_t)&nk_maxsim_pack_f16_sapphireamx, *c = nk_cap_sapphireamx_k; return;
        case nk_kernel_maxsim_packed_k: *m = (m_t)&nk_maxsim_packed_f16_sapphireamx, *c = nk_cap_sapphireamx_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SAPPHIRE
    if (v & nk_cap_sapphire_k) switch (k) {
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_f16_sapphire, *c = nk_cap_sapphire_k; return;
        default: break;
        }
#endif
#if NK_TARGET_ICELAKE
    if (v & nk_cap_icelake_k) switch (k) {
        case nk_kernel_maxsim_packed_size_k:
            *m = (m_t)&nk_maxsim_packed_size_f16_icelake, *c = nk_cap_icelake_k;
            return;
        case nk_kernel_maxsim_pack_k: *m = (m_t)&nk_maxsim_pack_f16_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_maxsim_packed_k: *m = (m_t)&nk_maxsim_packed_f16_icelake, *c = nk_cap_icelake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_DIAMOND
    if (v & nk_cap_diamond_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f16_diamond, *c = nk_cap_diamond_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f16_diamond, *c = nk_cap_diamond_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f16_diamond, *c = nk_cap_diamond_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f16_diamond, *c = nk_cap_diamond_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_f16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_f16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_f16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_f16_skylake, *c = nk_cap_skylake_k;
            return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_f16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_f16_skylake, *c = nk_cap_skylake_k;
            return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_f16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_f16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_kld_k: *m = (m_t)&nk_kld_f16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_jsd_k: *m = (m_t)&nk_jsd_f16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_sin_k: *m = (m_t)&nk_each_sin_f16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_cos_k: *m = (m_t)&nk_each_cos_f16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_atan_k: *m = (m_t)&nk_each_atan_f16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_f16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_f16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_f16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_rmsd_k: *m = (m_t)&nk_rmsd_f16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_kabsch_k: *m = (m_t)&nk_kabsch_f16_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_umeyama_k: *m = (m_t)&nk_umeyama_f16_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_ALDER
    if (v & nk_cap_alder_k) switch (k) {
        case nk_kernel_maxsim_packed_size_k: *m = (m_t)&nk_maxsim_packed_size_f16_alder, *c = nk_cap_alder_k; return;
        case nk_kernel_maxsim_pack_k: *m = (m_t)&nk_maxsim_pack_f16_alder, *c = nk_cap_alder_k; return;
        case nk_kernel_maxsim_packed_k: *m = (m_t)&nk_maxsim_packed_f16_alder, *c = nk_cap_alder_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_jsd_k: *m = (m_t)&nk_jsd_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_kld_k: *m = (m_t)&nk_kld_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_mahalanobis_k: *m = (m_t)&nk_mahalanobis_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_f16_haswell, *c = nk_cap_haswell_k;
            return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_f16_haswell, *c = nk_cap_haswell_k;
            return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_maxsim_packed_size_k:
            *m = (m_t)&nk_maxsim_packed_size_f16_haswell, *c = nk_cap_haswell_k;
            return;
        case nk_kernel_maxsim_pack_k: *m = (m_t)&nk_maxsim_pack_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_maxsim_packed_k: *m = (m_t)&nk_maxsim_packed_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_kabsch_k: *m = (m_t)&nk_kabsch_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_rmsd_k: *m = (m_t)&nk_rmsd_f16_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_umeyama_k: *m = (m_t)&nk_umeyama_f16_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
#if NK_TARGET_POWERVSX
    if (v & nk_cap_powervsx_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f16_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f16_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f16_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f16_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_f16_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f16_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_f16_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f16_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_f16_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_f16_powervsx, *c = nk_cap_powervsx_k;
            return;
        case nk_kernel_euclideans_packed_k:
            *m = (m_t)&nk_euclideans_packed_f16_powervsx, *c = nk_cap_powervsx_k;
            return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_f16_powervsx, *c = nk_cap_powervsx_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_LOONGSONASX
    if (v & nk_cap_loongsonasx_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f16_loongsonasx, *c = nk_cap_loongsonasx_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f16_loongsonasx, *c = nk_cap_loongsonasx_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f16_loongsonasx, *c = nk_cap_loongsonasx_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f16_loongsonasx, *c = nk_cap_loongsonasx_k; return;
        case nk_kernel_dots_packed_size_k:
            *m = (m_t)&nk_dots_packed_size_f16_loongsonasx, *c = nk_cap_loongsonasx_k;
            return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f16_loongsonasx, *c = nk_cap_loongsonasx_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_f16_loongsonasx, *c = nk_cap_loongsonasx_k; return;
        case nk_kernel_dots_symmetric_k:
            *m = (m_t)&nk_dots_symmetric_f16_loongsonasx, *c = nk_cap_loongsonasx_k;
            return;
        case nk_kernel_angulars_packed_k:
            *m = (m_t)&nk_angulars_packed_f16_loongsonasx, *c = nk_cap_loongsonasx_k;
            return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_f16_loongsonasx, *c = nk_cap_loongsonasx_k;
            return;
        case nk_kernel_euclideans_packed_k:
            *m = (m_t)&nk_euclideans_packed_f16_loongsonasx, *c = nk_cap_loongsonasx_k;
            return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_f16_loongsonasx, *c = nk_cap_loongsonasx_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_RVVHALF
    if (v & nk_cap_rvvhalf_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f16_rvvhalf, *c = nk_cap_rvvhalf_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f16_rvvhalf, *c = nk_cap_rvvhalf_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f16_rvvhalf, *c = nk_cap_rvvhalf_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f16_rvvhalf, *c = nk_cap_rvvhalf_k; return;
        default: break;
        }
#endif
#if NK_TARGET_RVV
    if (v & nk_cap_rvv_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_f16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_f16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_kld_k: *m = (m_t)&nk_kld_f16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_jsd_k: *m = (m_t)&nk_jsd_f16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_f16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_f16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_f16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_f16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_f16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_euclideans_symmetric_k: *m = (m_t)&nk_euclideans_symmetric_f16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_each_sin_k: *m = (m_t)&nk_each_sin_f16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_each_cos_k: *m = (m_t)&nk_each_cos_f16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_each_atan_k: *m = (m_t)&nk_each_atan_f16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_f16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_f16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_f16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_f16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_mahalanobis_k: *m = (m_t)&nk_mahalanobis_f16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_kabsch_k: *m = (m_t)&nk_kabsch_f16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_rmsd_k: *m = (m_t)&nk_rmsd_f16_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_umeyama_k: *m = (m_t)&nk_umeyama_f16_rvv, *c = nk_cap_rvv_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_jsd_k: *m = (m_t)&nk_jsd_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_kld_k: *m = (m_t)&nk_kld_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_mahalanobis_k: *m = (m_t)&nk_mahalanobis_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_rmsd_k: *m = (m_t)&nk_rmsd_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_kabsch_k: *m = (m_t)&nk_kabsch_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_umeyama_k: *m = (m_t)&nk_umeyama_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_f16_serial, *c = nk_cap_serial_k;
            return;
        case nk_kernel_each_sin_k: *m = (m_t)&nk_each_sin_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_cos_k: *m = (m_t)&nk_each_cos_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_atan_k: *m = (m_t)&nk_each_atan_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_maxsim_packed_size_k: *m = (m_t)&nk_maxsim_packed_size_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_maxsim_pack_k: *m = (m_t)&nk_maxsim_pack_f16_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_maxsim_packed_k: *m = (m_t)&nk_maxsim_packed_f16_serial, *c = nk_cap_serial_k; return;
        default: break;
        }

    // Error fallback - zero capability signals lookup failure
    *m = (m_t)nk_error_dense_, *c = 0;
}

void nk_dispatch_f16_init_(nk_capability_t caps) {
    nk_implementations_t *t = &nk_dispatch_table;
    nk_capability_t used;

    nk_dispatch_f16_find_(caps, nk_kernel_dot_k, (nk_kernel_punned_t *)&t->dot_f16, &used);
    nk_dispatch_f16_find_(caps, nk_kernel_angular_k, (nk_kernel_punned_t *)&t->angular_f16, &used);
    nk_dispatch_f16_find_(caps, nk_kernel_euclidean_k, (nk_kernel_punned_t *)&t->euclidean_f16, &used);
    nk_dispatch_f16_find_(caps, nk_kernel_sqeuclidean_k, (nk_kernel_punned_t *)&t->sqeuclidean_f16, &used);
    nk_dispatch_f16_find_(caps, nk_kernel_bilinear_k, (nk_kernel_punned_t *)&t->bilinear_f16, &used);
    nk_dispatch_f16_find_(caps, nk_kernel_mahalanobis_k, (nk_kernel_punned_t *)&t->mahalanobis_f16, &used);
    nk_dispatch_f16_find_(caps, nk_kernel_kld_k, (nk_kernel_punned_t *)&t->kld_f16, &used);
    nk_dispatch_f16_find_(caps, nk_kernel_jsd_k, (nk_kernel_punned_t *)&t->jsd_f16, &used);
    nk_dispatch_f16_find_(caps, nk_kernel_rmsd_k, (nk_kernel_punned_t *)&t->rmsd_f16, &used);
    nk_dispatch_f16_find_(caps, nk_kernel_kabsch_k, (nk_kernel_punned_t *)&t->kabsch_f16, &used);
    nk_dispatch_f16_find_(caps, nk_kernel_umeyama_k, (nk_kernel_punned_t *)&t->umeyama_f16, &used);
    nk_dispatch_f16_find_(caps, nk_kernel_each_scale_k, (nk_kernel_punned_t *)&t->each_scale_f16, &used);
    nk_dispatch_f16_find_(caps, nk_kernel_each_sum_k, (nk_kernel_punned_t *)&t->each_sum_f16, &used);
    nk_dispatch_f16_find_(caps, nk_kernel_each_blend_k, (nk_kernel_punned_t *)&t->each_blend_f16, &used);
    nk_dispatch_f16_find_(caps, nk_kernel_each_fma_k, (nk_kernel_punned_t *)&t->each_fma_f16, &used);
    nk_dispatch_f16_find_(caps, nk_kernel_each_sin_k, (nk_kernel_punned_t *)&t->each_sin_f16, &used);
    nk_dispatch_f16_find_(caps, nk_kernel_each_cos_k, (nk_kernel_punned_t *)&t->each_cos_f16, &used);
    nk_dispatch_f16_find_(caps, nk_kernel_each_atan_k, (nk_kernel_punned_t *)&t->each_atan_f16, &used);
    nk_dispatch_f16_find_(caps, nk_kernel_reduce_moments_k, (nk_kernel_punned_t *)&t->reduce_moments_f16, &used);
    nk_dispatch_f16_find_(caps, nk_kernel_reduce_minmax_k, (nk_kernel_punned_t *)&t->reduce_minmax_f16, &used);
    nk_dispatch_f16_find_(caps, nk_kernel_dots_packed_size_k, (nk_kernel_punned_t *)&t->dots_packed_size_f16, &used);
    nk_dispatch_f16_find_(caps, nk_kernel_dots_pack_k, (nk_kernel_punned_t *)&t->dots_pack_f16, &used);
    nk_dispatch_f16_find_(caps, nk_kernel_dots_packed_k, (nk_kernel_punned_t *)&t->dots_packed_f16, &used);
    nk_dispatch_f16_find_(caps, nk_kernel_dots_symmetric_k, (nk_kernel_punned_t *)&t->dots_symmetric_f16, &used);
    nk_dispatch_f16_find_(caps, nk_kernel_angulars_packed_k, (nk_kernel_punned_t *)&t->angulars_packed_f16, &used);
    nk_dispatch_f16_find_(caps, nk_kernel_angulars_symmetric_k, (nk_kernel_punned_t *)&t->angulars_symmetric_f16,
                          &used);
    nk_dispatch_f16_find_(caps, nk_kernel_euclideans_packed_k, (nk_kernel_punned_t *)&t->euclideans_packed_f16, &used);
    nk_dispatch_f16_find_(caps, nk_kernel_euclideans_symmetric_k, (nk_kernel_punned_t *)&t->euclideans_symmetric_f16,
                          &used);
    nk_dispatch_f16_find_(caps, nk_kernel_maxsim_packed_size_k, (nk_kernel_punned_t *)&t->maxsim_packed_size_f16,
                          &used);
    nk_dispatch_f16_find_(caps, nk_kernel_maxsim_pack_k, (nk_kernel_punned_t *)&t->maxsim_pack_f16, &used);
    nk_dispatch_f16_find_(caps, nk_kernel_maxsim_packed_k, (nk_kernel_punned_t *)&t->maxsim_packed_f16, &used);
}
