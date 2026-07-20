/**
 *  @brief Dispatch Initialization for I4 Data Types.
 *  @file c/dispatch_i4.c
 *  @author Ash Vardanian
 *  @date February 3, 2026
 */
#include "dispatch.h"

#ifdef __cplusplus
extern "C" {
#endif

void nk_dispatch_i4_find_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m, nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_V128RELAXED
    if (v & nk_cap_v128relaxed_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_i4_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_i4_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_i4_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_i4_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_i4_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SME
    if (v & nk_cap_sme_k) switch (k) {
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_i4_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_i4_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_i4_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_i4_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_i4_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_i4_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_i4_sme, *c = nk_cap_sme_k; return;
        case nk_kernel_euclideans_symmetric_k: *m = (m_t)&nk_euclideans_symmetric_i4_sme, *c = nk_cap_sme_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEONSDOT
    if (v & nk_cap_neonsdot_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_i4_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_i4_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_i4_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_i4_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_i4_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_i4_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_angulars_symmetric_k:
            *m = (m_t)&nk_angulars_symmetric_i4_neonsdot, *c = nk_cap_neonsdot_k;
            return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_i4_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_i4_neonsdot, *c = nk_cap_neonsdot_k;
            return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_i4_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_i4_neonsdot, *c = nk_cap_neonsdot_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_i4_neonsdot, *c = nk_cap_neonsdot_k; return;
        default: break;
        }
#endif
#if NK_TARGET_ICELAKE
    if (v & nk_cap_icelake_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_i4_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_i4_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_i4_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_i4_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_i4_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_i4_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_i4_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_i4_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_i4_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_i4_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_i4_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_i4_icelake, *c = nk_cap_icelake_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_i4_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_i4_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_i4_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_i4_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_i4_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_i4_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_i4_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
#if NK_TARGET_RVV
    if (v & nk_cap_rvv_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_i4_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_i4_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_i4_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_i4_rvv, *c = nk_cap_rvv_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_i4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angular_k: *m = (m_t)&nk_angular_i4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_sqeuclidean_k: *m = (m_t)&nk_sqeuclidean_i4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclidean_k: *m = (m_t)&nk_euclidean_i4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_i4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_i4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_i4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_i4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angulars_packed_k: *m = (m_t)&nk_angulars_packed_i4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_angulars_symmetric_k: *m = (m_t)&nk_angulars_symmetric_i4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclideans_packed_k: *m = (m_t)&nk_euclideans_packed_i4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_euclideans_symmetric_k:
            *m = (m_t)&nk_euclideans_symmetric_i4_serial, *c = nk_cap_serial_k;
            return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_i4_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_i4_serial, *c = nk_cap_serial_k; return;
        default: break;
        }

    // Error fallback - zero capability signals lookup failure
    *m = (m_t)nk_error_dense_, *c = 0;
}

void nk_dispatch_i4_init_(nk_capability_t caps) {
    nk_implementations_t *t = &nk_dispatch_table;
    nk_capability_t used;

    nk_dispatch_i4_find_(caps, nk_kernel_dot_k, (nk_kernel_punned_t *)&t->dot_i4, &used);
    nk_dispatch_i4_find_(caps, nk_kernel_angular_k, (nk_kernel_punned_t *)&t->angular_i4, &used);
    nk_dispatch_i4_find_(caps, nk_kernel_euclidean_k, (nk_kernel_punned_t *)&t->euclidean_i4, &used);
    nk_dispatch_i4_find_(caps, nk_kernel_sqeuclidean_k, (nk_kernel_punned_t *)&t->sqeuclidean_i4, &used);
    nk_dispatch_i4_find_(caps, nk_kernel_reduce_moments_k, (nk_kernel_punned_t *)&t->reduce_moments_i4, &used);
    nk_dispatch_i4_find_(caps, nk_kernel_reduce_minmax_k, (nk_kernel_punned_t *)&t->reduce_minmax_i4, &used);
    nk_dispatch_i4_find_(caps, nk_kernel_dots_pack_size_k, (nk_kernel_punned_t *)&t->dots_pack_size_i4, &used);
    nk_dispatch_i4_find_(caps, nk_kernel_dots_pack_k, (nk_kernel_punned_t *)&t->dots_pack_i4, &used);
    nk_dispatch_i4_find_(caps, nk_kernel_dots_packed_k, (nk_kernel_punned_t *)&t->dots_packed_i4, &used);
    nk_dispatch_i4_find_(caps, nk_kernel_dots_symmetric_k, (nk_kernel_punned_t *)&t->dots_symmetric_i4, &used);
    nk_dispatch_i4_find_(caps, nk_kernel_angulars_packed_k, (nk_kernel_punned_t *)&t->angulars_packed_i4, &used);
    nk_dispatch_i4_find_(caps, nk_kernel_angulars_symmetric_k, (nk_kernel_punned_t *)&t->angulars_symmetric_i4, &used);
    nk_dispatch_i4_find_(caps, nk_kernel_euclideans_packed_k, (nk_kernel_punned_t *)&t->euclideans_packed_i4, &used);
    nk_dispatch_i4_find_(caps, nk_kernel_euclideans_symmetric_k, (nk_kernel_punned_t *)&t->euclideans_symmetric_i4,
                         &used);
}

#ifdef __cplusplus
}
#endif
