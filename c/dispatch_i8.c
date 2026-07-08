/**
 *  @brief Dispatch Initialization for I8 Data Types.
 *  @file c/dispatch_i8.c
 *  @author Ash Vardanian
 *  @date February 3, 2026
 */
#include "dispatch.h"

void nk_dispatch_i8_find_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m, nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_V128RELAXED
    if (v & nk_cap_v128relaxed_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_i8_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_i8_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_i8_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_i8_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_i8_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_i8_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_dots_packed_size_k:
            *m = (m_t)&nk_dots_packed_size_i8_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_i8_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_i8_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_i8_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_angulars_packed_k:
            *m = (m_t)&nk_angulars_packed_i8_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_i8_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_euclideans_packed_k:
            *m = (m_t)&nk_euclideans_packed_i8_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_i8_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_i8_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_i8_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_i8_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_i8_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SME
    if (v & nk_cap_sme_k) switch (k) {
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_i8_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_i8_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_i8_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_i8_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_i8_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_i8_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_i8_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_euclideans_symmetric_k: *m = (m_t)&nk_euclideans_symmetric_i8_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_attention_packed_size_k: *m = (m_t)&nk_attention_packed_size_i8_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_attention_pack_k: *m = (m_t)&nk_attention_pack_i8_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_attention_packed_k: *m = (m_t)&nk_attention_packed_i8_sme, *c = nk_cap_sme_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SVESDOT
    if (v & nk_cap_svesdot_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_i8_svesdot, *c = nk_cap_svesdot_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_i8_svesdot, *c = nk_cap_svesdot_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_i8_svesdot, *c = nk_cap_svesdot_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_i8_svesdot, *c = nk_cap_svesdot_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEONSDOT
    if (v & nk_cap_neonsdot_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_i8_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_i8_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_i8_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_i8_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_i8_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_i8_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_i8_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_i8_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_i8_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_i8_neonsdot, *c = nk_cap_neonsdot_k;
            return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_i8_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_i8_neonsdot, *c = nk_cap_neonsdot_k;
            return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_i8_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_attention_packed_size_k:
            *m = (m_t)&nk_attention_packed_size_i8_neonsdot, *c = nk_cap_neonsdot_k;
            return;
        case nk_kernel_attention_pack_k: *m = (m_t)&nk_attention_pack_i8_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_attention_packed_k: *m = (m_t)&nk_attention_packed_i8_neonsdot, *c = nk_cap_neonsdot_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEONHALF
    if (v & nk_cap_neonhalf_k) switch (k) {
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_i8_neonhalf, *c = nk_cap_neonhalf_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_i8_neonhalf, *c = nk_cap_neonhalf_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_i8_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_i8_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_i8_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SAPPHIREAMX
    if (v & nk_cap_sapphireamx_k) switch (k) {
        case nk_kernel_dots_packed_size_k:
            *m = (m_t)&nk_dots_packed_size_i8_sapphireamx, *c = nk_cap_sapphireamx_k;
            return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_i8_sapphireamx, *c = nk_cap_sapphireamx_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_i8_sapphireamx, *c = nk_cap_sapphireamx_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_i8_sapphireamx, *c = nk_cap_sapphireamx_k; return;
        case nk_kernel_angulars_packed_k:
            *m = (m_t)&nk_angulars_packed_i8_sapphireamx, *c = nk_cap_sapphireamx_k;
            return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_i8_sapphireamx, *c = nk_cap_sapphireamx_k;
            return;
        case nk_kernel_euclideans_packed_k:
            *m = (m_t)&nk_euclideans_packed_i8_sapphireamx, *c = nk_cap_sapphireamx_k;
            return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_i8_sapphireamx, *c = nk_cap_sapphireamx_k;
            return;
        case nk_kernel_attention_packed_size_k:
            *m = (m_t)&nk_attention_packed_size_i8_sapphireamx, *c = nk_cap_sapphireamx_k;
            return;
        case nk_kernel_attention_pack_k: *m = (m_t)&nk_attention_pack_i8_sapphireamx, *c = nk_cap_sapphireamx_k; return;
        case nk_kernel_attention_packed_k:
            *m = (m_t)&nk_attention_packed_i8_sapphireamx, *c = nk_cap_sapphireamx_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_SAPPHIRE
    if (v & nk_cap_sapphire_k) switch (k) {
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_i8_sapphire, *c = nk_cap_sapphire_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_i8_sapphire, *c = nk_cap_sapphire_k; return;
        default: break;
        }
#endif
#if NK_TARGET_ICELAKE
    if (v & nk_cap_icelake_k) switch (k) {
        case nk_kernel_attention_packed_size_k:
            *m = (m_t)&nk_attention_packed_size_i8_icelake, *c = nk_cap_icelake_k;
            return;
        case nk_kernel_attention_pack_k: *m = (m_t)&nk_attention_pack_i8_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_attention_packed_k: *m = (m_t)&nk_attention_packed_i8_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_i8_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_i8_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_i8_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_i8_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_i8_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_i8_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_i8_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_i8_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_i8_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_i8_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_i8_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_i8_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_i8_icelake, *c = nk_cap_icelake_k;
            return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_i8_icelake, *c = nk_cap_icelake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_i8_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_i8_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_i8_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_i8_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_i8_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SIERRA
    if (v & nk_cap_sierra_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_i8_sierra, *c = nk_cap_sierra_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_i8_sierra, *c = nk_cap_sierra_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_i8_sierra, *c = nk_cap_sierra_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_i8_sierra, *c = nk_cap_sierra_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_i8_sierra, *c = nk_cap_sierra_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_i8_sierra, *c = nk_cap_sierra_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_i8_sierra, *c = nk_cap_sierra_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_i8_sierra, *c = nk_cap_sierra_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_i8_sierra, *c = nk_cap_sierra_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_i8_sierra, *c = nk_cap_sierra_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_i8_sierra, *c = nk_cap_sierra_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_i8_sierra, *c = nk_cap_sierra_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_i8_sierra, *c = nk_cap_sierra_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_ALDER
    if (v & nk_cap_alder_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_i8_alder, *c = nk_cap_alder_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_i8_alder, *c = nk_cap_alder_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_i8_alder, *c = nk_cap_alder_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_i8_alder, *c = nk_cap_alder_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_i8_alder, *c = nk_cap_alder_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_i8_alder, *c = nk_cap_alder_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_i8_alder, *c = nk_cap_alder_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_i8_alder, *c = nk_cap_alder_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_i8_alder, *c = nk_cap_alder_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_i8_alder, *c = nk_cap_alder_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_i8_alder, *c = nk_cap_alder_k; return;
        case nk_kernel_euclideans_symmetric_k: *m = (m_t)&nk_euclideans_symmetric_i8_alder, *c = nk_cap_alder_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_attention_packed_size_k:
            *m = (m_t)&nk_attention_packed_size_i8_haswell, *c = nk_cap_haswell_k;
            return;
        case nk_kernel_attention_pack_k: *m = (m_t)&nk_attention_pack_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_attention_packed_k: *m = (m_t)&nk_attention_packed_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_i8_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_i8_haswell, *c = nk_cap_haswell_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_POWERVSX
    if (v & nk_cap_powervsx_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_i8_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_i8_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_i8_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_i8_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_i8_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_i8_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_i8_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_i8_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_i8_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_i8_powervsx, *c = nk_cap_powervsx_k;
            return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_i8_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_i8_powervsx, *c = nk_cap_powervsx_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_LOONGSONASX
    if (v & nk_cap_loongsonasx_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_i8_loongsonasx, *c = nk_cap_loongsonasx_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_i8_loongsonasx, *c = nk_cap_loongsonasx_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_i8_loongsonasx, *c = nk_cap_loongsonasx_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_i8_loongsonasx, *c = nk_cap_loongsonasx_k; return;
        case nk_kernel_dots_packed_size_k:
            *m = (m_t)&nk_dots_packed_size_i8_loongsonasx, *c = nk_cap_loongsonasx_k;
            return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_i8_loongsonasx, *c = nk_cap_loongsonasx_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_i8_loongsonasx, *c = nk_cap_loongsonasx_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_i8_loongsonasx, *c = nk_cap_loongsonasx_k; return;
        case nk_kernel_angulars_packed_k:
            *m = (m_t)&nk_angulars_packed_i8_loongsonasx, *c = nk_cap_loongsonasx_k;
            return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_i8_loongsonasx, *c = nk_cap_loongsonasx_k;
            return;
        case nk_kernel_euclideans_packed_k:
            *m = (m_t)&nk_euclideans_packed_i8_loongsonasx, *c = nk_cap_loongsonasx_k;
            return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_i8_loongsonasx, *c = nk_cap_loongsonasx_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_RVV
    if (v & nk_cap_rvv_k) switch (k) {
        case nk_kernel_attention_packed_size_k: *m = (m_t)&nk_attention_packed_size_i8_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_attention_pack_k: *m = (m_t)&nk_attention_pack_i8_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_attention_packed_k: *m = (m_t)&nk_attention_packed_i8_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_i8_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_i8_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_i8_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_i8_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_i8_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_i8_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_i8_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_i8_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_i8_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_i8_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_i8_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_i8_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_i8_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_euclideans_symmetric_k: *m = (m_t)&nk_euclideans_symmetric_i8_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_i8_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_i8_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_i8_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_i8_rvv, *c = nk_cap_rvv_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_i8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_i8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_i8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_i8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_i8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_i8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_i8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_i8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_i8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_i8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_i8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_i8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_i8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_i8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_i8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_i8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_i8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_i8_serial, *c = nk_cap_serial_k;
            return;
        case nk_kernel_attention_packed_size_k:
            *m = (m_t)&nk_attention_packed_size_i8_serial, *c = nk_cap_serial_k;
            return;
        case nk_kernel_attention_pack_k: *m = (m_t)&nk_attention_pack_i8_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_attention_packed_k: *m = (m_t)&nk_attention_packed_i8_serial, *c = nk_cap_serial_k; return;
        default: break;
        }

    // Error fallback - zero capability signals lookup failure
    *m = (m_t)nk_error_dense_, *c = 0;
}

void nk_dispatch_i8_init_(nk_capability_t caps) {
    nk_implementations_t *t = &nk_dispatch_table;
    nk_capability_t used;

    nk_dispatch_i8_find_(caps, nk_kernel_attention_packed_size_k, (nk_kernel_punned_t *)&t->attention_packed_size_i8,
                         &used);
    nk_dispatch_i8_find_(caps, nk_kernel_attention_pack_k, (nk_kernel_punned_t *)&t->attention_pack_i8, &used);
    nk_dispatch_i8_find_(caps, nk_kernel_attention_packed_k, (nk_kernel_punned_t *)&t->attention_packed_i8, &used);

    nk_dispatch_i8_find_(caps, nk_kernel_dot_k, (nk_kernel_punned_t *)&t->dot_i8, &used);
    nk_dispatch_i8_find_(caps, nk_kernel_angular_k, (nk_kernel_punned_t *)&t->angular_i8, &used);
    nk_dispatch_i8_find_(caps, nk_kernel_euclidean_k, (nk_kernel_punned_t *)&t->euclidean_i8, &used);
    nk_dispatch_i8_find_(caps, nk_kernel_sqeuclidean_k, (nk_kernel_punned_t *)&t->sqeuclidean_i8, &used);
    nk_dispatch_i8_find_(caps, nk_kernel_each_scale_k, (nk_kernel_punned_t *)&t->each_scale_i8, &used);
    nk_dispatch_i8_find_(caps, nk_kernel_each_sum_k, (nk_kernel_punned_t *)&t->each_sum_i8, &used);
    nk_dispatch_i8_find_(caps, nk_kernel_each_blend_k, (nk_kernel_punned_t *)&t->each_blend_i8, &used);
    nk_dispatch_i8_find_(caps, nk_kernel_each_fma_k, (nk_kernel_punned_t *)&t->each_fma_i8, &used);
    nk_dispatch_i8_find_(caps, nk_kernel_reduce_moments_k, (nk_kernel_punned_t *)&t->reduce_moments_i8, &used);
    nk_dispatch_i8_find_(caps, nk_kernel_reduce_minmax_k, (nk_kernel_punned_t *)&t->reduce_minmax_i8, &used);
    nk_dispatch_i8_find_(caps, nk_kernel_dots_packed_size_k, (nk_kernel_punned_t *)&t->dots_packed_size_i8, &used);
    nk_dispatch_i8_find_(caps, nk_kernel_dots_pack_k, (nk_kernel_punned_t *)&t->dots_pack_i8, &used);
    nk_dispatch_i8_find_(caps, nk_kernel_dots_packed_k, (nk_kernel_punned_t *)&t->dots_packed_i8, &used);
    nk_dispatch_i8_find_(caps, nk_kernel_dots_symmetric_k, (nk_kernel_punned_t *)&t->dots_symmetric_i8, &used);
    nk_dispatch_i8_find_(caps, nk_kernel_angulars_packed_k, (nk_kernel_punned_t *)&t->angulars_packed_i8, &used);
    nk_dispatch_i8_find_(caps, nk_kernel_angulars_symmetric_k, (nk_kernel_punned_t *)&t->angulars_symmetric_i8, &used);
    nk_dispatch_i8_find_(caps, nk_kernel_euclideans_packed_k, (nk_kernel_punned_t *)&t->euclideans_packed_i8, &used);
    nk_dispatch_i8_find_(caps, nk_kernel_euclideans_symmetric_k, (nk_kernel_punned_t *)&t->euclideans_symmetric_i8,
                         &used);
}
