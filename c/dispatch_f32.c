/**
 *  @brief Dispatch Initialization for F32 Data Types.
 *  @file c/dispatch_f32.c
 *  @author Ash Vardanian
 *  @date February 3, 2026
 */
#include "dispatch.h"

void nk_dispatch_f32_find_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m, nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_V128RELAXED
    if (v & nk_cap_v128relaxed_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f32_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f32_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f32_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f32_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_haversine_k: *m = (m_t)&nk_haversine_f32_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_vincenty_k: *m = (m_t)&nk_vincenty_f32_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_trig_sin_k: *m = (m_t)&nk_trig_sin_f32_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_trig_cos_k: *m = (m_t)&nk_trig_cos_f32_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_trig_atan_k: *m = (m_t)&nk_trig_atan_f32_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_reduce_moments_k:
            *m = (m_t)&nk_reduce_moments_f32_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_f32_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_dots_pack_size_k:
            *m = (m_t)&nk_dots_pack_size_f32_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f32_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_f32_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_dots_symmetric_k:
            *m = (m_t)&nk_dots_symmetric_f32_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_angulars_packed_k:
            *m = (m_t)&nk_angulars_packed_f32_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_f32_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_euclideans_packed_k:
            *m = (m_t)&nk_euclideans_packed_f32_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_f32_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_maxsim_pack_size_k:
            *m = (m_t)&nk_maxsim_pack_size_f32_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_maxsim_pack_k: *m = (m_t)&nk_maxsim_pack_f32_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_maxsim_packed_k: *m = (m_t)&nk_maxsim_packed_f32_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_rmsd_k: *m = (m_t)&nk_rmsd_f32_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_kabsch_k: *m = (m_t)&nk_kabsch_f32_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_umeyama_k: *m = (m_t)&nk_umeyama_f32_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_f32_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_f32_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_f32_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_f32_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SME
    if (v & nk_cap_sme_k) switch (k) {
        case nk_kernel_maxsim_pack_size_k: *m = (m_t)&nk_maxsim_pack_size_f32_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_maxsim_pack_k: *m = (m_t)&nk_maxsim_pack_f32_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_maxsim_packed_k: *m = (m_t)&nk_maxsim_packed_f32_sme, *c = nk_cap_sme_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SMEF64
    if (v & nk_cap_smef64_k) switch (k) {
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f32_smef64, *c = nk_cap_smef64_k; return;
        case nk_kernel_mahalanobis_k: *m = (m_t)&nk_mahalanobis_f32_smef64, *c = nk_cap_smef64_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_f32_smef64, *c = nk_cap_smef64_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f32_smef64, *c = nk_cap_smef64_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_f32_smef64, *c = nk_cap_smef64_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f32_smef64, *c = nk_cap_smef64_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_f32_smef64, *c = nk_cap_smef64_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_f32_smef64, *c = nk_cap_smef64_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_f32_smef64, *c = nk_cap_smef64_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_f32_smef64, *c = nk_cap_smef64_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_SVE2
    if (v & nk_cap_sve2_k) switch (k) {
        case nk_kernel_sparse_dot_k: *m = (m_t)&nk_sparse_dot_u32f32_sve2, *c = nk_cap_sve2_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SVE
    if (v & nk_cap_sve_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f32_sve, *c = nk_cap_sve_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f32_sve, *c = nk_cap_sve_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f32_sve, *c = nk_cap_sve_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f32_sve, *c = nk_cap_sve_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_jsd_k: *m = (m_t)&nk_jsd_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_kld_k: *m = (m_t)&nk_kld_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_haversine_k: *m = (m_t)&nk_haversine_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_vincenty_k: *m = (m_t)&nk_vincenty_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_trig_sin_k: *m = (m_t)&nk_trig_sin_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_trig_cos_k: *m = (m_t)&nk_trig_cos_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_trig_atan_k: *m = (m_t)&nk_trig_atan_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_mahalanobis_k: *m = (m_t)&nk_mahalanobis_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_rmsd_k: *m = (m_t)&nk_rmsd_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_kabsch_k: *m = (m_t)&nk_kabsch_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_umeyama_k: *m = (m_t)&nk_umeyama_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_f32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_euclideans_symmetric_k: *m = (m_t)&nk_euclideans_symmetric_f32_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_TURIN
    if (v & nk_cap_turin_k) switch (k) {
        case nk_kernel_sparse_dot_k: *m = (m_t)&nk_sparse_dot_u32f32_turin, *c = nk_cap_turin_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SAPPHIREAMX
    if (v & nk_cap_sapphireamx_k) switch (k) {
        case nk_kernel_maxsim_pack_size_k:
            *m = (m_t)&nk_maxsim_pack_size_f32_sapphireamx, *c = nk_cap_sapphireamx_k;
            return;
        case nk_kernel_maxsim_pack_k: *m = (m_t)&nk_maxsim_pack_f32_sapphireamx, *c = nk_cap_sapphireamx_k; return;
        case nk_kernel_maxsim_packed_k: *m = (m_t)&nk_maxsim_packed_f32_sapphireamx, *c = nk_cap_sapphireamx_k; return;
        default: break;
        }
#endif
#if NK_TARGET_ICELAKE
    if (v & nk_cap_icelake_k) switch (k) {
        case nk_kernel_sparse_dot_k: *m = (m_t)&nk_sparse_dot_u32f32_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_maxsim_pack_size_k: *m = (m_t)&nk_maxsim_pack_size_f32_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_maxsim_pack_k: *m = (m_t)&nk_maxsim_pack_f32_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_maxsim_packed_k: *m = (m_t)&nk_maxsim_packed_f32_icelake, *c = nk_cap_icelake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_jsd_k: *m = (m_t)&nk_jsd_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_kld_k: *m = (m_t)&nk_kld_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_mahalanobis_k: *m = (m_t)&nk_mahalanobis_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_haversine_k: *m = (m_t)&nk_haversine_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_vincenty_k: *m = (m_t)&nk_vincenty_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_swiglu_k: *m = (m_t)&nk_each_swiglu_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_trig_sin_k: *m = (m_t)&nk_trig_sin_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_trig_cos_k: *m = (m_t)&nk_trig_cos_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_trig_atan_k: *m = (m_t)&nk_trig_atan_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_trig_rope_k: *m = (m_t)&nk_trig_rope_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_rmsd_k: *m = (m_t)&nk_rmsd_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_kabsch_k: *m = (m_t)&nk_kabsch_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_umeyama_k: *m = (m_t)&nk_umeyama_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_rmsnorm_k: *m = (m_t)&nk_reduce_rmsnorm_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_f32_skylake, *c = nk_cap_skylake_k;
            return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_f32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_f32_skylake, *c = nk_cap_skylake_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_ALDER
    if (v & nk_cap_alder_k) switch (k) {
        case nk_kernel_maxsim_pack_size_k: *m = (m_t)&nk_maxsim_pack_size_f32_alder, *c = nk_cap_alder_k; return;
        case nk_kernel_maxsim_pack_k: *m = (m_t)&nk_maxsim_pack_f32_alder, *c = nk_cap_alder_k; return;
        case nk_kernel_maxsim_packed_k: *m = (m_t)&nk_maxsim_packed_f32_alder, *c = nk_cap_alder_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_haversine_k: *m = (m_t)&nk_haversine_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_vincenty_k: *m = (m_t)&nk_vincenty_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_swiglu_k: *m = (m_t)&nk_each_swiglu_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_rmsd_k: *m = (m_t)&nk_rmsd_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_kabsch_k: *m = (m_t)&nk_kabsch_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_umeyama_k: *m = (m_t)&nk_umeyama_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_rmsnorm_k: *m = (m_t)&nk_reduce_rmsnorm_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_trig_sin_k: *m = (m_t)&nk_trig_sin_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_trig_cos_k: *m = (m_t)&nk_trig_cos_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_trig_atan_k: *m = (m_t)&nk_trig_atan_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_trig_rope_k: *m = (m_t)&nk_trig_rope_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_f32_haswell, *c = nk_cap_haswell_k;
            return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_f32_haswell, *c = nk_cap_haswell_k;
            return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_mahalanobis_k: *m = (m_t)&nk_mahalanobis_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_maxsim_pack_size_k: *m = (m_t)&nk_maxsim_pack_size_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_maxsim_pack_k: *m = (m_t)&nk_maxsim_pack_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_maxsim_packed_k: *m = (m_t)&nk_maxsim_packed_f32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_sparse_dot_k: *m = (m_t)&nk_sparse_dot_u32f32_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEONSDOT
    if (v & nk_cap_neonsdot_k) switch (k) {
        case nk_kernel_maxsim_pack_size_k: *m = (m_t)&nk_maxsim_pack_size_f32_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_maxsim_pack_k: *m = (m_t)&nk_maxsim_pack_f32_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_maxsim_packed_k: *m = (m_t)&nk_maxsim_packed_f32_neonsdot, *c = nk_cap_neonsdot_k; return;
        default: break;
        }
#endif
#if NK_TARGET_POWERVSX
    if (v & nk_cap_powervsx_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f32_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f32_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f32_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f32_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_f32_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f32_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_f32_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f32_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_f32_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_f32_powervsx, *c = nk_cap_powervsx_k;
            return;
        case nk_kernel_euclideans_packed_k:
            *m = (m_t)&nk_euclideans_packed_f32_powervsx, *c = nk_cap_powervsx_k;
            return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_f32_powervsx, *c = nk_cap_powervsx_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_LOONGSONASX
    if (v & nk_cap_loongsonasx_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f32_loongsonasx, *c = nk_cap_loongsonasx_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f32_loongsonasx, *c = nk_cap_loongsonasx_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f32_loongsonasx, *c = nk_cap_loongsonasx_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f32_loongsonasx, *c = nk_cap_loongsonasx_k; return;
        case nk_kernel_dots_pack_size_k:
            *m = (m_t)&nk_dots_pack_size_f32_loongsonasx, *c = nk_cap_loongsonasx_k;
            return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f32_loongsonasx, *c = nk_cap_loongsonasx_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_f32_loongsonasx, *c = nk_cap_loongsonasx_k; return;
        case nk_kernel_dots_symmetric_k:
            *m = (m_t)&nk_dots_symmetric_f32_loongsonasx, *c = nk_cap_loongsonasx_k;
            return;
        case nk_kernel_angulars_packed_k:
            *m = (m_t)&nk_angulars_packed_f32_loongsonasx, *c = nk_cap_loongsonasx_k;
            return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_f32_loongsonasx, *c = nk_cap_loongsonasx_k;
            return;
        case nk_kernel_euclideans_packed_k:
            *m = (m_t)&nk_euclideans_packed_f32_loongsonasx, *c = nk_cap_loongsonasx_k;
            return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_f32_loongsonasx, *c = nk_cap_loongsonasx_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_RVV
    if (v & nk_cap_rvv_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f32_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f32_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f32_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f32_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_haversine_k: *m = (m_t)&nk_haversine_f32_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_vincenty_k: *m = (m_t)&nk_vincenty_f32_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_f32_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_f32_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_kld_k: *m = (m_t)&nk_kld_f32_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_jsd_k: *m = (m_t)&nk_jsd_f32_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_f32_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f32_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_f32_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f32_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_f32_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_f32_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_f32_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_euclideans_symmetric_k: *m = (m_t)&nk_euclideans_symmetric_f32_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_f32_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_f32_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_f32_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_f32_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_trig_sin_k: *m = (m_t)&nk_trig_sin_f32_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_trig_cos_k: *m = (m_t)&nk_trig_cos_f32_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_trig_atan_k: *m = (m_t)&nk_trig_atan_f32_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f32_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_mahalanobis_k: *m = (m_t)&nk_mahalanobis_f32_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_rmsd_k: *m = (m_t)&nk_rmsd_f32_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_kabsch_k: *m = (m_t)&nk_kabsch_f32_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_umeyama_k: *m = (m_t)&nk_umeyama_f32_rvv, *c = nk_cap_rvv_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_jsd_k: *m = (m_t)&nk_jsd_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_kld_k: *m = (m_t)&nk_kld_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_mahalanobis_k: *m = (m_t)&nk_mahalanobis_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_haversine_k: *m = (m_t)&nk_haversine_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_vincenty_k: *m = (m_t)&nk_vincenty_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_swiglu_k: *m = (m_t)&nk_each_swiglu_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_trig_sin_k: *m = (m_t)&nk_trig_sin_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_trig_cos_k: *m = (m_t)&nk_trig_cos_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_trig_atan_k: *m = (m_t)&nk_trig_atan_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_trig_rope_k: *m = (m_t)&nk_trig_rope_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_rmsd_k: *m = (m_t)&nk_rmsd_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_kabsch_k: *m = (m_t)&nk_kabsch_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_umeyama_k: *m = (m_t)&nk_umeyama_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_sparse_dot_k: *m = (m_t)&nk_sparse_dot_u32f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_rmsnorm_k: *m = (m_t)&nk_reduce_rmsnorm_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_f32_serial, *c = nk_cap_serial_k;
            return;
        case nk_kernel_maxsim_pack_size_k: *m = (m_t)&nk_maxsim_pack_size_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_maxsim_pack_k: *m = (m_t)&nk_maxsim_pack_f32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_maxsim_packed_k: *m = (m_t)&nk_maxsim_packed_f32_serial, *c = nk_cap_serial_k; return;
        default: break;
        }

    // Error fallback - zero capability signals lookup failure
    *m = (m_t)nk_error_dense_, *c = 0;
}

void nk_dispatch_f32_init_(nk_capability_t caps) {
    nk_implementations_t *t = &nk_dispatch_table;
    nk_capability_t used;

    nk_dispatch_f32_find_(caps, nk_kernel_dot_k, (nk_kernel_punned_t *)&t->dot_f32, &used);
    nk_dispatch_f32_find_(caps, nk_kernel_angular_k, (nk_kernel_punned_t *)&t->angular_f32, &used);
    nk_dispatch_f32_find_(caps, nk_kernel_euclidean_k, (nk_kernel_punned_t *)&t->euclidean_f32, &used);
    nk_dispatch_f32_find_(caps, nk_kernel_sqeuclidean_k, (nk_kernel_punned_t *)&t->sqeuclidean_f32, &used);
    nk_dispatch_f32_find_(caps, nk_kernel_bilinear_k, (nk_kernel_punned_t *)&t->bilinear_f32, &used);
    nk_dispatch_f32_find_(caps, nk_kernel_mahalanobis_k, (nk_kernel_punned_t *)&t->mahalanobis_f32, &used);
    nk_dispatch_f32_find_(caps, nk_kernel_haversine_k, (nk_kernel_punned_t *)&t->haversine_f32, &used);
    nk_dispatch_f32_find_(caps, nk_kernel_vincenty_k, (nk_kernel_punned_t *)&t->vincenty_f32, &used);
    nk_dispatch_f32_find_(caps, nk_kernel_kld_k, (nk_kernel_punned_t *)&t->kld_f32, &used);
    nk_dispatch_f32_find_(caps, nk_kernel_jsd_k, (nk_kernel_punned_t *)&t->jsd_f32, &used);
    nk_dispatch_f32_find_(caps, nk_kernel_rmsd_k, (nk_kernel_punned_t *)&t->rmsd_f32, &used);
    nk_dispatch_f32_find_(caps, nk_kernel_kabsch_k, (nk_kernel_punned_t *)&t->kabsch_f32, &used);
    nk_dispatch_f32_find_(caps, nk_kernel_umeyama_k, (nk_kernel_punned_t *)&t->umeyama_f32, &used);
    nk_dispatch_f32_find_(caps, nk_kernel_sparse_dot_k, (nk_kernel_punned_t *)&t->sparse_dot_u32f32, &used);
    nk_dispatch_f32_find_(caps, nk_kernel_each_scale_k, (nk_kernel_punned_t *)&t->each_scale_f32, &used);
    nk_dispatch_f32_find_(caps, nk_kernel_each_sum_k, (nk_kernel_punned_t *)&t->each_sum_f32, &used);
    nk_dispatch_f32_find_(caps, nk_kernel_each_blend_k, (nk_kernel_punned_t *)&t->each_blend_f32, &used);
    nk_dispatch_f32_find_(caps, nk_kernel_each_fma_k, (nk_kernel_punned_t *)&t->each_fma_f32, &used);
    nk_dispatch_f32_find_(caps, nk_kernel_each_swiglu_k, (nk_kernel_punned_t *)&t->each_swiglu_f32, &used);
    nk_dispatch_f32_find_(caps, nk_kernel_trig_sin_k, (nk_kernel_punned_t *)&t->trig_sin_f32, &used);
    nk_dispatch_f32_find_(caps, nk_kernel_trig_cos_k, (nk_kernel_punned_t *)&t->trig_cos_f32, &used);
    nk_dispatch_f32_find_(caps, nk_kernel_trig_atan_k, (nk_kernel_punned_t *)&t->trig_atan_f32, &used);
    nk_dispatch_f32_find_(caps, nk_kernel_trig_rope_k, (nk_kernel_punned_t *)&t->trig_rope_f32, &used);
    nk_dispatch_f32_find_(caps, nk_kernel_reduce_moments_k, (nk_kernel_punned_t *)&t->reduce_moments_f32, &used);
    nk_dispatch_f32_find_(caps, nk_kernel_reduce_minmax_k, (nk_kernel_punned_t *)&t->reduce_minmax_f32, &used);
    nk_dispatch_f32_find_(caps, nk_kernel_reduce_rmsnorm_k, (nk_kernel_punned_t *)&t->reduce_rmsnorm_f32, &used);
    nk_dispatch_f32_find_(caps, nk_kernel_dots_pack_size_k, (nk_kernel_punned_t *)&t->dots_pack_size_f32, &used);
    nk_dispatch_f32_find_(caps, nk_kernel_dots_pack_k, (nk_kernel_punned_t *)&t->dots_pack_f32, &used);
    nk_dispatch_f32_find_(caps, nk_kernel_dots_packed_k, (nk_kernel_punned_t *)&t->dots_packed_f32, &used);
    nk_dispatch_f32_find_(caps, nk_kernel_dots_symmetric_k, (nk_kernel_punned_t *)&t->dots_symmetric_f32, &used);
    nk_dispatch_f32_find_(caps, nk_kernel_angulars_packed_k, (nk_kernel_punned_t *)&t->angulars_packed_f32, &used);
    nk_dispatch_f32_find_(caps, nk_kernel_angulars_symmetric_k, (nk_kernel_punned_t *)&t->angulars_symmetric_f32,
                          &used);
    nk_dispatch_f32_find_(caps, nk_kernel_euclideans_packed_k, (nk_kernel_punned_t *)&t->euclideans_packed_f32, &used);
    nk_dispatch_f32_find_(caps, nk_kernel_euclideans_symmetric_k, (nk_kernel_punned_t *)&t->euclideans_symmetric_f32,
                          &used);
    nk_dispatch_f32_find_(caps, nk_kernel_maxsim_pack_size_k, (nk_kernel_punned_t *)&t->maxsim_pack_size_f32, &used);
    nk_dispatch_f32_find_(caps, nk_kernel_maxsim_pack_k, (nk_kernel_punned_t *)&t->maxsim_pack_f32, &used);
    nk_dispatch_f32_find_(caps, nk_kernel_maxsim_packed_k, (nk_kernel_punned_t *)&t->maxsim_packed_f32, &used);
}
