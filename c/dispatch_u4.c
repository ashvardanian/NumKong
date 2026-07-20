/**
 *  @brief Dispatch Initialization for U4 Data Types.
 *  @file c/dispatch_u4.c
 *  @author Ash Vardanian
 *  @date February 3, 2026
 */
#include "dispatch.h"

#ifdef __cplusplus
extern "C" {
#endif

void nk_dispatch_u4_find_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m, nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_V128RELAXED
    if (v & nk_cap_v128relaxed_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_u4_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_u4_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_u4_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_u4_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_u4_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SME
    if (v & nk_cap_sme_k) switch (k) {
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_u4_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_u4_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_u4_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_u4_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_u4_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_u4_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_u4_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_euclideans_symmetric_k: *m = (m_t)&nk_euclideans_symmetric_u4_sme, *c = nk_cap_sme_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEONSDOT
    if (v & nk_cap_neonsdot_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_u4_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_u4_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_u4_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_u4_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_u4_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_u4_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_u4_neonsdot, *c = nk_cap_neonsdot_k;
            return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_u4_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_u4_neonsdot, *c = nk_cap_neonsdot_k;
            return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_u4_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_u4_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_u4_neonsdot, *c = nk_cap_neonsdot_k; return;
        default: break;
        }
#endif
#if NK_TARGET_ICELAKE
    if (v & nk_cap_icelake_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_u4_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_u4_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_u4_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_u4_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_u4_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_u4_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_u4_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_u4_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_u4_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_u4_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_u4_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_u4_icelake, *c = nk_cap_icelake_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_u4_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_u4_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_u4_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_u4_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_u4_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_u4_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_u4_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
#if NK_TARGET_RVV
    if (v & nk_cap_rvv_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_u4_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_u4_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_u4_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_u4_rvv, *c = nk_cap_rvv_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_u4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_u4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_u4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_u4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_u4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_u4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_u4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_u4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_u4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_u4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_u4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_u4_serial, *c = nk_cap_serial_k;
            return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_u4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_u4_serial, *c = nk_cap_serial_k; return;
        default: break;
        }

    // Error fallback - zero capability signals lookup failure
    *m = (m_t)nk_error_dense_, *c = 0;
}

void nk_dispatch_u4_init_(nk_capability_t caps) {
    nk_implementations_t *t = &nk_dispatch_table;
    nk_capability_t used;

    nk_dispatch_u4_find_(caps, nk_kernel_dot_k, (nk_kernel_punned_t *)&t->dot_u4, &used);
    nk_dispatch_u4_find_(caps, nk_kernel_angular_k, (nk_kernel_punned_t *)&t->angular_u4, &used);
    nk_dispatch_u4_find_(caps, nk_kernel_euclidean_k, (nk_kernel_punned_t *)&t->euclidean_u4, &used);
    nk_dispatch_u4_find_(caps, nk_kernel_sqeuclidean_k, (nk_kernel_punned_t *)&t->sqeuclidean_u4, &used);
    nk_dispatch_u4_find_(caps, nk_kernel_reduce_moments_k, (nk_kernel_punned_t *)&t->reduce_moments_u4, &used);
    nk_dispatch_u4_find_(caps, nk_kernel_reduce_minmax_k, (nk_kernel_punned_t *)&t->reduce_minmax_u4, &used);
    nk_dispatch_u4_find_(caps, nk_kernel_dots_pack_size_k, (nk_kernel_punned_t *)&t->dots_pack_size_u4, &used);
    nk_dispatch_u4_find_(caps, nk_kernel_dots_pack_k, (nk_kernel_punned_t *)&t->dots_pack_u4, &used);
    nk_dispatch_u4_find_(caps, nk_kernel_dots_packed_k, (nk_kernel_punned_t *)&t->dots_packed_u4, &used);
    nk_dispatch_u4_find_(caps, nk_kernel_dots_symmetric_k, (nk_kernel_punned_t *)&t->dots_symmetric_u4, &used);
    nk_dispatch_u4_find_(caps, nk_kernel_angulars_packed_k, (nk_kernel_punned_t *)&t->angulars_packed_u4, &used);
    nk_dispatch_u4_find_(caps, nk_kernel_angulars_symmetric_k, (nk_kernel_punned_t *)&t->angulars_symmetric_u4, &used);
    nk_dispatch_u4_find_(caps, nk_kernel_euclideans_packed_k, (nk_kernel_punned_t *)&t->euclideans_packed_u4, &used);
    nk_dispatch_u4_find_(caps, nk_kernel_euclideans_symmetric_k, (nk_kernel_punned_t *)&t->euclideans_symmetric_u4,
                         &used);
}

#ifdef __cplusplus
}
#endif
