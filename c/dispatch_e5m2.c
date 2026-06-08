/**
 *  @brief Dispatch Initialization for E5M2 Data Types.
 *  @file c/dispatch_e5m2.c
 *  @author Ash Vardanian
 *  @date February 3, 2026
 */
#include "dispatch.h"

void nk_dispatch_e5m2_find_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m, nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_V128RELAXED
    if (v & nk_cap_v128relaxed_k) switch (k) {
        case nk_kernel_reduce_moments_k:
            *m = (m_t)&nk_reduce_moments_e5m2_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_e5m2_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e5m2_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e5m2_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e5m2_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e5m2_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_dots_packed_size_k:
            *m = (m_t)&nk_dots_packed_size_e5m2_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e5m2_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e5m2_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_dots_symmetric_k:
            *m = (m_t)&nk_dots_symmetric_e5m2_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_angulars_packed_k:
            *m = (m_t)&nk_angulars_packed_e5m2_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_e5m2_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_euclideans_packed_k:
            *m = (m_t)&nk_euclideans_packed_e5m2_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_e5m2_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_SME
    if (v & nk_cap_sme_k) switch (k) {
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e5m2_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e5m2_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e5m2_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e5m2_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_e5m2_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_e5m2_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_e5m2_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_euclideans_symmetric_k: *m = (m_t)&nk_euclideans_symmetric_e5m2_sme, *c = nk_cap_sme_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEONFP8
    if (v & nk_cap_neonfp8_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e5m2_neonfp8, *c = nk_cap_neonfp8_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e5m2_neonfp8, *c = nk_cap_neonfp8_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e5m2_neonfp8, *c = nk_cap_neonfp8_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e5m2_neonfp8, *c = nk_cap_neonfp8_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e5m2_neonfp8, *c = nk_cap_neonfp8_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e5m2_neonfp8, *c = nk_cap_neonfp8_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e5m2_neonfp8, *c = nk_cap_neonfp8_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e5m2_neonfp8, *c = nk_cap_neonfp8_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_e5m2_neonfp8, *c = nk_cap_neonfp8_k; return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_e5m2_neonfp8, *c = nk_cap_neonfp8_k;
            return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_e5m2_neonfp8, *c = nk_cap_neonfp8_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_e5m2_neonfp8, *c = nk_cap_neonfp8_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_NEONFHM
    if (v & nk_cap_neonfhm_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e5m2_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_e5m2_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e5m2_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e5m2_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e5m2_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e5m2_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_e5m2_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_e5m2_neonfhm, *c = nk_cap_neonfhm_k;
            return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_e5m2_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_e5m2_neonfhm, *c = nk_cap_neonfhm_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_NEONBFDOT
    if (v & nk_cap_neonbfdot_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e5m2_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e5m2_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e5m2_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e5m2_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e5m2_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_e5m2_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_e5m2_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_e5m2_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_e5m2_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_GRANITEAMX
    if (v & nk_cap_graniteamx_k) switch (k) {
        case nk_kernel_dots_packed_size_k:
            *m = (m_t)&nk_dots_packed_size_e5m2_graniteamx, *c = nk_cap_graniteamx_k;
            return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e5m2_graniteamx, *c = nk_cap_graniteamx_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e5m2_graniteamx, *c = nk_cap_graniteamx_k; return;
        case nk_kernel_angulars_packed_k:
            *m = (m_t)&nk_angulars_packed_e5m2_graniteamx, *c = nk_cap_graniteamx_k;
            return;
        case nk_kernel_euclideans_packed_k:
            *m = (m_t)&nk_euclideans_packed_e5m2_graniteamx, *c = nk_cap_graniteamx_k;
            return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e5m2_graniteamx, *c = nk_cap_graniteamx_k; return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_e5m2_graniteamx, *c = nk_cap_graniteamx_k;
            return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_e5m2_graniteamx, *c = nk_cap_graniteamx_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_SAPPHIREAMX
    if (v & nk_cap_sapphireamx_k) switch (k) {
        case nk_kernel_dots_packed_size_k:
            *m = (m_t)&nk_dots_packed_size_e5m2_sapphireamx, *c = nk_cap_sapphireamx_k;
            return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e5m2_sapphireamx, *c = nk_cap_sapphireamx_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e5m2_sapphireamx, *c = nk_cap_sapphireamx_k; return;
        case nk_kernel_angulars_packed_k:
            *m = (m_t)&nk_angulars_packed_e5m2_sapphireamx, *c = nk_cap_sapphireamx_k;
            return;
        case nk_kernel_euclideans_packed_k:
            *m = (m_t)&nk_euclideans_packed_e5m2_sapphireamx, *c = nk_cap_sapphireamx_k;
            return;
        case nk_kernel_dots_symmetric_k:
            *m = (m_t)&nk_dots_symmetric_e5m2_sapphireamx, *c = nk_cap_sapphireamx_k;
            return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_e5m2_sapphireamx, *c = nk_cap_sapphireamx_k;
            return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_e5m2_sapphireamx, *c = nk_cap_sapphireamx_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_DIAMOND
    if (v & nk_cap_diamond_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e5m2_diamond, *c = nk_cap_diamond_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e5m2_diamond, *c = nk_cap_diamond_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e5m2_diamond, *c = nk_cap_diamond_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e5m2_diamond, *c = nk_cap_diamond_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e5m2_diamond, *c = nk_cap_diamond_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e5m2_diamond, *c = nk_cap_diamond_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e5m2_diamond, *c = nk_cap_diamond_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e5m2_diamond, *c = nk_cap_diamond_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_e5m2_diamond, *c = nk_cap_diamond_k; return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_e5m2_diamond, *c = nk_cap_diamond_k;
            return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_e5m2_diamond, *c = nk_cap_diamond_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_e5m2_diamond, *c = nk_cap_diamond_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_GENOA
    if (v & nk_cap_genoa_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e5m2_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e5m2_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e5m2_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e5m2_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e5m2_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_e5m2_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_e5m2_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_e5m2_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_e5m2_genoa, *c = nk_cap_genoa_k;
            return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_e5m2_genoa, *c = nk_cap_genoa_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e5m2_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e5m2_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e5m2_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e5m2_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e5m2_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e5m2_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e5m2_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e5m2_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_e5m2_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_e5m2_skylake, *c = nk_cap_skylake_k;
            return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_e5m2_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_e5m2_skylake, *c = nk_cap_skylake_k;
            return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_e5m2_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_e5m2_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e5m2_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e5m2_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e5m2_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e5m2_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_e5m2_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_e5m2_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e5m2_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e5m2_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e5m2_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e5m2_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_e5m2_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_e5m2_haswell, *c = nk_cap_haswell_k;
            return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_e5m2_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_e5m2_haswell, *c = nk_cap_haswell_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_RVVHALF
    if (v & nk_cap_rvvhalf_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e5m2_rvvhalf, *c = nk_cap_rvvhalf_k; return;
        default: break;
        }
#endif
#if NK_TARGET_RVVBF16
    if (v & nk_cap_rvvbf16_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e5m2_rvvbf16, *c = nk_cap_rvvbf16_k; return;
        default: break;
        }
#endif
#if NK_TARGET_RVV
    if (v & nk_cap_rvv_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e5m2_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e5m2_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e5m2_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e5m2_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_e5m2_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_e5m2_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e5m2_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e5m2_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e5m2_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e5m2_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_e5m2_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_e5m2_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_e5m2_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_euclideans_symmetric_k: *m = (m_t)&nk_euclideans_symmetric_e5m2_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_e5m2_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_e5m2_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_e5m2_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_e5m2_rvv, *c = nk_cap_rvv_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e5m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e5m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e5m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e5m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_e5m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_e5m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_e5m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_e5m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_e5m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_e5m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_packed_size_k: *m = (m_t)&nk_dots_packed_size_e5m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e5m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e5m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e5m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_e5m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_e5m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_e5m2_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_e5m2_serial, *c = nk_cap_serial_k;
            return;
        default: break;
        }

    // Error fallback - zero capability signals lookup failure
    *m = (m_t)nk_error_dense_, *c = 0;
}

void nk_dispatch_e5m2_init_(nk_capability_t caps) {
    nk_implementations_t *t = &nk_dispatch_table;
    nk_capability_t used;

    nk_dispatch_e5m2_find_(caps, nk_kernel_dot_k, (nk_kernel_punned_t *)&t->dot_e5m2, &used);
    nk_dispatch_e5m2_find_(caps, nk_kernel_angular_k, (nk_kernel_punned_t *)&t->angular_e5m2, &used);
    nk_dispatch_e5m2_find_(caps, nk_kernel_euclidean_k, (nk_kernel_punned_t *)&t->euclidean_e5m2, &used);
    nk_dispatch_e5m2_find_(caps, nk_kernel_sqeuclidean_k, (nk_kernel_punned_t *)&t->sqeuclidean_e5m2, &used);
    nk_dispatch_e5m2_find_(caps, nk_kernel_each_scale_k, (nk_kernel_punned_t *)&t->each_scale_e5m2, &used);
    nk_dispatch_e5m2_find_(caps, nk_kernel_each_sum_k, (nk_kernel_punned_t *)&t->each_sum_e5m2, &used);
    nk_dispatch_e5m2_find_(caps, nk_kernel_each_blend_k, (nk_kernel_punned_t *)&t->each_blend_e5m2, &used);
    nk_dispatch_e5m2_find_(caps, nk_kernel_each_fma_k, (nk_kernel_punned_t *)&t->each_fma_e5m2, &used);
    nk_dispatch_e5m2_find_(caps, nk_kernel_reduce_moments_k, (nk_kernel_punned_t *)&t->reduce_moments_e5m2, &used);
    nk_dispatch_e5m2_find_(caps, nk_kernel_reduce_minmax_k, (nk_kernel_punned_t *)&t->reduce_minmax_e5m2, &used);
    nk_dispatch_e5m2_find_(caps, nk_kernel_dots_packed_size_k, (nk_kernel_punned_t *)&t->dots_packed_size_e5m2, &used);
    nk_dispatch_e5m2_find_(caps, nk_kernel_dots_pack_k, (nk_kernel_punned_t *)&t->dots_pack_e5m2, &used);
    nk_dispatch_e5m2_find_(caps, nk_kernel_dots_packed_k, (nk_kernel_punned_t *)&t->dots_packed_e5m2, &used);
    nk_dispatch_e5m2_find_(caps, nk_kernel_dots_symmetric_k, (nk_kernel_punned_t *)&t->dots_symmetric_e5m2, &used);
    nk_dispatch_e5m2_find_(caps, nk_kernel_angulars_packed_k, (nk_kernel_punned_t *)&t->angulars_packed_e5m2, &used);
    nk_dispatch_e5m2_find_(caps, nk_kernel_angulars_symmetric_k, (nk_kernel_punned_t *)&t->angulars_symmetric_e5m2,
                           &used);
    nk_dispatch_e5m2_find_(caps, nk_kernel_euclideans_packed_k, (nk_kernel_punned_t *)&t->euclideans_packed_e5m2,
                           &used);
    nk_dispatch_e5m2_find_(caps, nk_kernel_euclideans_symmetric_k, (nk_kernel_punned_t *)&t->euclideans_symmetric_e5m2,
                           &used);
}
