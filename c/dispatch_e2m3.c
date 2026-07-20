/**
 *  @brief Dispatch Initialization for E2M3 Data Types.
 *  @file c/dispatch_e2m3.c
 *  @author Ash Vardanian
 *  @date February 3, 2026
 */
#include "dispatch.h"

#ifdef __cplusplus
extern "C" {
#endif

void nk_dispatch_e2m3_find_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m, nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_V128RELAXED
    if (v & nk_cap_v128relaxed_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e2m3_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e2m3_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e2m3_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e2m3_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_reduce_moments_k:
            *m = (m_t)&nk_reduce_moments_e2m3_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_e2m3_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_dots_pack_size_k:
            *m = (m_t)&nk_dots_pack_size_e2m3_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e2m3_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e2m3_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_dots_symmetric_k:
            *m = (m_t)&nk_dots_symmetric_e2m3_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_angulars_packed_k:
            *m = (m_t)&nk_angulars_packed_e2m3_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_e2m3_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_euclideans_packed_k:
            *m = (m_t)&nk_euclideans_packed_e2m3_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_e2m3_v128relaxed, *c = nk_cap_v128relaxed_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_SME
    if (v & nk_cap_sme_k) switch (k) {
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_e2m3_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e2m3_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e2m3_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e2m3_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_e2m3_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_e2m3_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_e2m3_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_euclideans_symmetric_k: *m = (m_t)&nk_euclideans_symmetric_e2m3_sme, *c = nk_cap_sme_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEONFP8
    if (v & nk_cap_neonfp8_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e2m3_neonfp8, *c = nk_cap_neonfp8_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e2m3_neonfp8, *c = nk_cap_neonfp8_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e2m3_neonfp8, *c = nk_cap_neonfp8_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e2m3_neonfp8, *c = nk_cap_neonfp8_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_e2m3_neonfp8, *c = nk_cap_neonfp8_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e2m3_neonfp8, *c = nk_cap_neonfp8_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e2m3_neonfp8, *c = nk_cap_neonfp8_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e2m3_neonfp8, *c = nk_cap_neonfp8_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_e2m3_neonfp8, *c = nk_cap_neonfp8_k; return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_e2m3_neonfp8, *c = nk_cap_neonfp8_k;
            return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_e2m3_neonfp8, *c = nk_cap_neonfp8_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_e2m3_neonfp8, *c = nk_cap_neonfp8_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_NEONSDOT
    if (v & nk_cap_neonsdot_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e2m3_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_e2m3_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_e2m3_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e2m3_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e2m3_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e2m3_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_e2m3_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_e2m3_neonsdot, *c = nk_cap_neonsdot_k;
            return;
        case nk_kernel_euclideans_packed_k:
            *m = (m_t)&nk_euclideans_packed_e2m3_neonsdot, *c = nk_cap_neonsdot_k;
            return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_e2m3_neonsdot, *c = nk_cap_neonsdot_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e2m3_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e2m3_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e2m3_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e2m3_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_e2m3_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_e2m3_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SAPPHIREAMX
    if (v & nk_cap_sapphireamx_k) switch (k) {
        case nk_kernel_dots_pack_size_k:
            *m = (m_t)&nk_dots_pack_size_e2m3_sapphireamx, *c = nk_cap_sapphireamx_k;
            return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e2m3_sapphireamx, *c = nk_cap_sapphireamx_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e2m3_sapphireamx, *c = nk_cap_sapphireamx_k; return;
        case nk_kernel_dots_symmetric_k:
            *m = (m_t)&nk_dots_symmetric_e2m3_sapphireamx, *c = nk_cap_sapphireamx_k;
            return;
        case nk_kernel_angulars_packed_k:
            *m = (m_t)&nk_angulars_packed_e2m3_sapphireamx, *c = nk_cap_sapphireamx_k;
            return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_e2m3_sapphireamx, *c = nk_cap_sapphireamx_k;
            return;
        case nk_kernel_euclideans_packed_k:
            *m = (m_t)&nk_euclideans_packed_e2m3_sapphireamx, *c = nk_cap_sapphireamx_k;
            return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_e2m3_sapphireamx, *c = nk_cap_sapphireamx_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_ICELAKE
    if (v & nk_cap_icelake_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e2m3_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e2m3_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e2m3_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e2m3_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_e2m3_icelake, *c = nk_cap_icelake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e2m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e2m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e2m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e2m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_e2m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e2m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e2m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e2m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_e2m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_e2m3_skylake, *c = nk_cap_skylake_k;
            return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_e2m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_e2m3_skylake, *c = nk_cap_skylake_k;
            return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_e2m3_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_e2m3_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SIERRA
    if (v & nk_cap_sierra_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e2m3_sierra, *c = nk_cap_sierra_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e2m3_sierra, *c = nk_cap_sierra_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e2m3_sierra, *c = nk_cap_sierra_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e2m3_sierra, *c = nk_cap_sierra_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_e2m3_sierra, *c = nk_cap_sierra_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e2m3_sierra, *c = nk_cap_sierra_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e2m3_sierra, *c = nk_cap_sierra_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e2m3_sierra, *c = nk_cap_sierra_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_e2m3_sierra, *c = nk_cap_sierra_k; return;
        default: break;
        }
#endif
#if NK_TARGET_ALDER
    if (v & nk_cap_alder_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e2m3_alder, *c = nk_cap_alder_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e2m3_alder, *c = nk_cap_alder_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e2m3_alder, *c = nk_cap_alder_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e2m3_alder, *c = nk_cap_alder_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_e2m3_alder, *c = nk_cap_alder_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_e2m3_alder, *c = nk_cap_alder_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e2m3_alder, *c = nk_cap_alder_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e2m3_alder, *c = nk_cap_alder_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e2m3_alder, *c = nk_cap_alder_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e2m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e2m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e2m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e2m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_e2m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e2m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e2m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e2m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_e2m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_e2m3_haswell, *c = nk_cap_haswell_k;
            return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_e2m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_e2m3_haswell, *c = nk_cap_haswell_k;
            return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_e2m3_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_e2m3_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
#if NK_TARGET_RVV
    if (v & nk_cap_rvv_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e2m3_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_e2m3_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e2m3_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e2m3_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e2m3_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_e2m3_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_e2m3_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_e2m3_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_euclideans_symmetric_k: *m = (m_t)&nk_euclideans_symmetric_e2m3_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_e2m3_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_e2m3_rvv, *c = nk_cap_rvv_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_e2m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_e2m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_e2m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_e2m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_e2m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_e2m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_e2m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_e2m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_e2m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_e2m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_e2m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_e2m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_e2m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_e2m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_e2m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_e2m3_serial, *c = nk_cap_serial_k;
            return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_e2m3_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_e2m3_serial, *c = nk_cap_serial_k; return;
        default: break;
        }

    // Error fallback - zero capability signals lookup failure
    *m = (m_t)nk_error_dense_, *c = 0;
}

void nk_dispatch_e2m3_init_(nk_capability_t caps) {
    nk_implementations_t *t = &nk_dispatch_table;
    nk_capability_t used;

    nk_dispatch_e2m3_find_(caps, nk_kernel_dot_k, (nk_kernel_punned_t *)&t->dot_e2m3, &used);
    nk_dispatch_e2m3_find_(caps, nk_kernel_angular_k, (nk_kernel_punned_t *)&t->angular_e2m3, &used);
    nk_dispatch_e2m3_find_(caps, nk_kernel_euclidean_k, (nk_kernel_punned_t *)&t->euclidean_e2m3, &used);
    nk_dispatch_e2m3_find_(caps, nk_kernel_sqeuclidean_k, (nk_kernel_punned_t *)&t->sqeuclidean_e2m3, &used);
    nk_dispatch_e2m3_find_(caps, nk_kernel_each_scale_k, (nk_kernel_punned_t *)&t->each_scale_e2m3, &used);
    nk_dispatch_e2m3_find_(caps, nk_kernel_each_sum_k, (nk_kernel_punned_t *)&t->each_sum_e2m3, &used);
    nk_dispatch_e2m3_find_(caps, nk_kernel_each_blend_k, (nk_kernel_punned_t *)&t->each_blend_e2m3, &used);
    nk_dispatch_e2m3_find_(caps, nk_kernel_each_fma_k, (nk_kernel_punned_t *)&t->each_fma_e2m3, &used);
    nk_dispatch_e2m3_find_(caps, nk_kernel_reduce_moments_k, (nk_kernel_punned_t *)&t->reduce_moments_e2m3, &used);
    nk_dispatch_e2m3_find_(caps, nk_kernel_reduce_minmax_k, (nk_kernel_punned_t *)&t->reduce_minmax_e2m3, &used);
    nk_dispatch_e2m3_find_(caps, nk_kernel_dots_pack_size_k, (nk_kernel_punned_t *)&t->dots_pack_size_e2m3, &used);
    nk_dispatch_e2m3_find_(caps, nk_kernel_dots_pack_k, (nk_kernel_punned_t *)&t->dots_pack_e2m3, &used);
    nk_dispatch_e2m3_find_(caps, nk_kernel_dots_packed_k, (nk_kernel_punned_t *)&t->dots_packed_e2m3, &used);
    nk_dispatch_e2m3_find_(caps, nk_kernel_dots_symmetric_k, (nk_kernel_punned_t *)&t->dots_symmetric_e2m3, &used);
    nk_dispatch_e2m3_find_(caps, nk_kernel_angulars_packed_k, (nk_kernel_punned_t *)&t->angulars_packed_e2m3, &used);
    nk_dispatch_e2m3_find_(caps, nk_kernel_angulars_symmetric_k, (nk_kernel_punned_t *)&t->angulars_symmetric_e2m3,
                           &used);
    nk_dispatch_e2m3_find_(caps, nk_kernel_euclideans_packed_k, (nk_kernel_punned_t *)&t->euclideans_packed_e2m3,
                           &used);
    nk_dispatch_e2m3_find_(caps, nk_kernel_euclideans_symmetric_k, (nk_kernel_punned_t *)&t->euclideans_symmetric_e2m3,
                           &used);
}

#ifdef __cplusplus
}
#endif
