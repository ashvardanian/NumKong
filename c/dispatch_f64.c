/**
 *  @brief Dispatch Initialization for F64 Data Types.
 *  @file c/dispatch_f64.c
 *  @author Ash Vardanian
 *  @date February 3, 2026
 */
#include "dispatch.h"

void nk_dispatch_f64_find_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m, nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_V128RELAXED
    if (v & nk_cap_v128relaxed_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f64_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f64_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f64_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f64_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_haversine_k: *m = (m_t)&nk_haversine_f64_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_vincenty_k: *m = (m_t)&nk_vincenty_f64_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_trig_sin_k: *m = (m_t)&nk_trig_sin_f64_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_trig_cos_k: *m = (m_t)&nk_trig_cos_f64_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_trig_atan_k: *m = (m_t)&nk_trig_atan_f64_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_reduce_moments_k:
            *m = (m_t)&nk_reduce_moments_f64_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_f64_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_dots_pack_size_k:
            *m = (m_t)&nk_dots_pack_size_f64_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_dots_packed_shape_k:
            *m = (m_t)&nk_dots_packed_shape_f64_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f64_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_f64_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_dots_symmetric_k:
            *m = (m_t)&nk_dots_symmetric_f64_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_angulars_packed_k:
            *m = (m_t)&nk_angulars_packed_f64_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_f64_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_euclideans_packed_k:
            *m = (m_t)&nk_euclideans_packed_f64_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_f64_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_rmsd_k: *m = (m_t)&nk_rmsd_f64_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_kabsch_k: *m = (m_t)&nk_kabsch_f64_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_umeyama_k: *m = (m_t)&nk_umeyama_f64_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SMEF64
    if (v & nk_cap_smef64_k) switch (k) {
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f64_smef64, *c = nk_cap_smef64_k; return;
        case nk_kernel_mahalanobis_k: *m = (m_t)&nk_mahalanobis_f64_smef64, *c = nk_cap_smef64_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_f64_smef64, *c = nk_cap_smef64_k; return;
        case nk_kernel_dots_packed_shape_k: *m = (m_t)&nk_dots_packed_shape_f64_smef64, *c = nk_cap_smef64_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f64_smef64, *c = nk_cap_smef64_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_f64_smef64, *c = nk_cap_smef64_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f64_smef64, *c = nk_cap_smef64_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_f64_smef64, *c = nk_cap_smef64_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_f64_smef64, *c = nk_cap_smef64_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_f64_smef64, *c = nk_cap_smef64_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_f64_smef64, *c = nk_cap_smef64_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_SVE
    if (v & nk_cap_sve_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f64_sve, *c = nk_cap_sve_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f64_sve, *c = nk_cap_sve_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f64_sve, *c = nk_cap_sve_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f64_sve, *c = nk_cap_sve_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_haversine_k: *m = (m_t)&nk_haversine_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_vincenty_k: *m = (m_t)&nk_vincenty_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_trig_sin_k: *m = (m_t)&nk_trig_sin_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_trig_cos_k: *m = (m_t)&nk_trig_cos_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_trig_atan_k: *m = (m_t)&nk_trig_atan_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_rmsd_k: *m = (m_t)&nk_rmsd_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_kabsch_k: *m = (m_t)&nk_kabsch_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_umeyama_k: *m = (m_t)&nk_umeyama_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_dots_packed_shape_k: *m = (m_t)&nk_dots_packed_shape_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_f64_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_euclideans_symmetric_k: *m = (m_t)&nk_euclideans_symmetric_f64_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_haversine_k: *m = (m_t)&nk_haversine_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_vincenty_k: *m = (m_t)&nk_vincenty_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_trig_sin_k: *m = (m_t)&nk_trig_sin_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_trig_cos_k: *m = (m_t)&nk_trig_cos_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_trig_atan_k: *m = (m_t)&nk_trig_atan_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_rmsd_k: *m = (m_t)&nk_rmsd_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_kabsch_k: *m = (m_t)&nk_kabsch_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_umeyama_k: *m = (m_t)&nk_umeyama_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_mahalanobis_k: *m = (m_t)&nk_mahalanobis_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_kld_k: *m = (m_t)&nk_kld_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_jsd_k: *m = (m_t)&nk_jsd_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_packed_shape_k: *m = (m_t)&nk_dots_packed_shape_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_f64_skylake, *c = nk_cap_skylake_k;
            return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_f64_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_f64_skylake, *c = nk_cap_skylake_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_haversine_k: *m = (m_t)&nk_haversine_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_vincenty_k: *m = (m_t)&nk_vincenty_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_rmsd_k: *m = (m_t)&nk_rmsd_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_kabsch_k: *m = (m_t)&nk_kabsch_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_umeyama_k: *m = (m_t)&nk_umeyama_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_trig_sin_k: *m = (m_t)&nk_trig_sin_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_trig_cos_k: *m = (m_t)&nk_trig_cos_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_trig_atan_k: *m = (m_t)&nk_trig_atan_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_kld_k: *m = (m_t)&nk_kld_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_jsd_k: *m = (m_t)&nk_jsd_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_packed_shape_k: *m = (m_t)&nk_dots_packed_shape_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_f64_haswell, *c = nk_cap_haswell_k;
            return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_f64_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_f64_haswell, *c = nk_cap_haswell_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_POWERVSX
    if (v & nk_cap_powervsx_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f64_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f64_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f64_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f64_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_f64_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_dots_packed_shape_k:
            *m = (m_t)&nk_dots_packed_shape_f64_powervsx, *c = nk_cap_powervsx_k;
            return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f64_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_f64_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f64_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_f64_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_f64_powervsx, *c = nk_cap_powervsx_k;
            return;
        case nk_kernel_euclideans_packed_k:
            *m = (m_t)&nk_euclideans_packed_f64_powervsx, *c = nk_cap_powervsx_k;
            return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_f64_powervsx, *c = nk_cap_powervsx_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_LOONGSONASX
    if (v & nk_cap_loongsonasx_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f64_loongsonasx, *c = nk_cap_loongsonasx_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f64_loongsonasx, *c = nk_cap_loongsonasx_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f64_loongsonasx, *c = nk_cap_loongsonasx_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f64_loongsonasx, *c = nk_cap_loongsonasx_k; return;
        case nk_kernel_dots_pack_size_k:
            *m = (m_t)&nk_dots_pack_size_f64_loongsonasx, *c = nk_cap_loongsonasx_k;
            return;
        case nk_kernel_dots_packed_shape_k:
            *m = (m_t)&nk_dots_packed_shape_f64_loongsonasx, *c = nk_cap_loongsonasx_k;
            return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f64_loongsonasx, *c = nk_cap_loongsonasx_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_f64_loongsonasx, *c = nk_cap_loongsonasx_k; return;
        case nk_kernel_dots_symmetric_k:
            *m = (m_t)&nk_dots_symmetric_f64_loongsonasx, *c = nk_cap_loongsonasx_k;
            return;
        case nk_kernel_angulars_packed_k:
            *m = (m_t)&nk_angulars_packed_f64_loongsonasx, *c = nk_cap_loongsonasx_k;
            return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_f64_loongsonasx, *c = nk_cap_loongsonasx_k;
            return;
        case nk_kernel_euclideans_packed_k:
            *m = (m_t)&nk_euclideans_packed_f64_loongsonasx, *c = nk_cap_loongsonasx_k;
            return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_f64_loongsonasx, *c = nk_cap_loongsonasx_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_RVV
    if (v & nk_cap_rvv_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f64_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f64_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f64_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f64_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_haversine_k: *m = (m_t)&nk_haversine_f64_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_vincenty_k: *m = (m_t)&nk_vincenty_f64_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_f64_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_f64_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_kld_k: *m = (m_t)&nk_kld_f64_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_jsd_k: *m = (m_t)&nk_jsd_f64_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_f64_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_dots_packed_shape_k: *m = (m_t)&nk_dots_packed_shape_f64_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f64_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_f64_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f64_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_f64_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_f64_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_f64_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_euclideans_symmetric_k: *m = (m_t)&nk_euclideans_symmetric_f64_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_f64_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_f64_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_f64_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_f64_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_trig_sin_k: *m = (m_t)&nk_trig_sin_f64_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_trig_cos_k: *m = (m_t)&nk_trig_cos_f64_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_trig_atan_k: *m = (m_t)&nk_trig_atan_f64_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f64_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_mahalanobis_k: *m = (m_t)&nk_mahalanobis_f64_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_rmsd_k: *m = (m_t)&nk_rmsd_f64_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_kabsch_k: *m = (m_t)&nk_kabsch_f64_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_umeyama_k: *m = (m_t)&nk_umeyama_f64_rvv, *c = nk_cap_rvv_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_jsd_k: *m = (m_t)&nk_jsd_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_kld_k: *m = (m_t)&nk_kld_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_mahalanobis_k: *m = (m_t)&nk_mahalanobis_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_haversine_k: *m = (m_t)&nk_haversine_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_vincenty_k: *m = (m_t)&nk_vincenty_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_trig_sin_k: *m = (m_t)&nk_trig_sin_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_trig_cos_k: *m = (m_t)&nk_trig_cos_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_trig_atan_k: *m = (m_t)&nk_trig_atan_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_rmsd_k: *m = (m_t)&nk_rmsd_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_kabsch_k: *m = (m_t)&nk_kabsch_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_umeyama_k: *m = (m_t)&nk_umeyama_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_packed_shape_k: *m = (m_t)&nk_dots_packed_shape_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_f64_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_f64_serial, *c = nk_cap_serial_k;
            return;
        default: break;
        }

    // Error fallback - zero capability signals lookup failure
    *m = (m_t)nk_error_dense_, *c = 0;
}

void nk_dispatch_f64_init_(nk_capability_t caps) {
    nk_implementations_t *t = &nk_dispatch_table;
    nk_capability_t used;

    nk_dispatch_f64_find_(caps, nk_kernel_dot_k, (nk_kernel_punned_t *)&t->dot_f64, &used);
    nk_dispatch_f64_find_(caps, nk_kernel_angular_k, (nk_kernel_punned_t *)&t->angular_f64, &used);
    nk_dispatch_f64_find_(caps, nk_kernel_euclidean_k, (nk_kernel_punned_t *)&t->euclidean_f64, &used);
    nk_dispatch_f64_find_(caps, nk_kernel_sqeuclidean_k, (nk_kernel_punned_t *)&t->sqeuclidean_f64, &used);
    nk_dispatch_f64_find_(caps, nk_kernel_bilinear_k, (nk_kernel_punned_t *)&t->bilinear_f64, &used);
    nk_dispatch_f64_find_(caps, nk_kernel_mahalanobis_k, (nk_kernel_punned_t *)&t->mahalanobis_f64, &used);
    nk_dispatch_f64_find_(caps, nk_kernel_haversine_k, (nk_kernel_punned_t *)&t->haversine_f64, &used);
    nk_dispatch_f64_find_(caps, nk_kernel_vincenty_k, (nk_kernel_punned_t *)&t->vincenty_f64, &used);
    nk_dispatch_f64_find_(caps, nk_kernel_kld_k, (nk_kernel_punned_t *)&t->kld_f64, &used);
    nk_dispatch_f64_find_(caps, nk_kernel_jsd_k, (nk_kernel_punned_t *)&t->jsd_f64, &used);
    nk_dispatch_f64_find_(caps, nk_kernel_rmsd_k, (nk_kernel_punned_t *)&t->rmsd_f64, &used);
    nk_dispatch_f64_find_(caps, nk_kernel_kabsch_k, (nk_kernel_punned_t *)&t->kabsch_f64, &used);
    nk_dispatch_f64_find_(caps, nk_kernel_umeyama_k, (nk_kernel_punned_t *)&t->umeyama_f64, &used);
    nk_dispatch_f64_find_(caps, nk_kernel_each_scale_k, (nk_kernel_punned_t *)&t->each_scale_f64, &used);
    nk_dispatch_f64_find_(caps, nk_kernel_each_sum_k, (nk_kernel_punned_t *)&t->each_sum_f64, &used);
    nk_dispatch_f64_find_(caps, nk_kernel_each_blend_k, (nk_kernel_punned_t *)&t->each_blend_f64, &used);
    nk_dispatch_f64_find_(caps, nk_kernel_each_fma_k, (nk_kernel_punned_t *)&t->each_fma_f64, &used);
    nk_dispatch_f64_find_(caps, nk_kernel_trig_sin_k, (nk_kernel_punned_t *)&t->trig_sin_f64, &used);
    nk_dispatch_f64_find_(caps, nk_kernel_trig_cos_k, (nk_kernel_punned_t *)&t->trig_cos_f64, &used);
    nk_dispatch_f64_find_(caps, nk_kernel_trig_atan_k, (nk_kernel_punned_t *)&t->trig_atan_f64, &used);
    nk_dispatch_f64_find_(caps, nk_kernel_reduce_moments_k, (nk_kernel_punned_t *)&t->reduce_moments_f64, &used);
    nk_dispatch_f64_find_(caps, nk_kernel_reduce_minmax_k, (nk_kernel_punned_t *)&t->reduce_minmax_f64, &used);
    nk_dispatch_f64_find_(caps, nk_kernel_dots_pack_size_k, (nk_kernel_punned_t *)&t->dots_pack_size_f64, &used);
    nk_dispatch_f64_find_(caps, nk_kernel_dots_packed_shape_k, (nk_kernel_punned_t *)&t->dots_packed_shape_f64, &used);
    nk_dispatch_f64_find_(caps, nk_kernel_dots_pack_k, (nk_kernel_punned_t *)&t->dots_pack_f64, &used);
    nk_dispatch_f64_find_(caps, nk_kernel_dots_packed_k, (nk_kernel_punned_t *)&t->dots_packed_f64, &used);
    nk_dispatch_f64_find_(caps, nk_kernel_dots_symmetric_k, (nk_kernel_punned_t *)&t->dots_symmetric_f64, &used);
    nk_dispatch_f64_find_(caps, nk_kernel_angulars_packed_k, (nk_kernel_punned_t *)&t->angulars_packed_f64, &used);
    nk_dispatch_f64_find_(caps, nk_kernel_angulars_symmetric_k, (nk_kernel_punned_t *)&t->angulars_symmetric_f64,
                          &used);
    nk_dispatch_f64_find_(caps, nk_kernel_euclideans_packed_k, (nk_kernel_punned_t *)&t->euclideans_packed_f64, &used);
    nk_dispatch_f64_find_(caps, nk_kernel_euclideans_symmetric_k, (nk_kernel_punned_t *)&t->euclideans_symmetric_f64,
                          &used);
}
