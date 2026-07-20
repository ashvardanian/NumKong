/**
 *  @brief Dispatch Initialization for U1 Data Types.
 *  @file c/dispatch_u1.c
 *  @author Ash Vardanian
 *  @date February 3, 2026
 */
#include "dispatch.h"

void nk_dispatch_u1_find_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m, nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_V128RELAXED
    if (v & nk_cap_v128relaxed_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_u1_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_hamming_k: *m = (m_t)&nk_hamming_u1_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u1_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SMEBI32
    if (v & nk_cap_smebi32_k) switch (k) {
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_u1_smebi32, *c = nk_cap_smebi32_k; return;
        case nk_kernel_dots_packed_shape_k: *m = (m_t)&nk_dots_packed_shape_u1_smebi32, *c = nk_cap_smebi32_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_u1_smebi32, *c = nk_cap_smebi32_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_u1_smebi32, *c = nk_cap_smebi32_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_u1_smebi32, *c = nk_cap_smebi32_k; return;
        case nk_kernel_hammings_packed_k: *m = (m_t)&nk_hammings_packed_u1_smebi32, *c = nk_cap_smebi32_k; return;
        case nk_kernel_hammings_symmetric_k: *m = (m_t)&nk_hammings_symmetric_u1_smebi32, *c = nk_cap_smebi32_k; return;
        case nk_kernel_jaccards_packed_k: *m = (m_t)&nk_jaccards_packed_u1_smebi32, *c = nk_cap_smebi32_k; return;
        case nk_kernel_jaccards_symmetric_k: *m = (m_t)&nk_jaccards_symmetric_u1_smebi32, *c = nk_cap_smebi32_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SVE
    if (v & nk_cap_sve_k) switch (k) {
        case nk_kernel_hamming_k: *m = (m_t)&nk_hamming_u1_sve, *c = nk_cap_sve_k; return;
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u1_sve, *c = nk_cap_sve_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_u1_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_hamming_k: *m = (m_t)&nk_hamming_u1_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u1_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_u1_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_dots_packed_shape_k: *m = (m_t)&nk_dots_packed_shape_u1_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_u1_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_u1_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_u1_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_hammings_packed_k: *m = (m_t)&nk_hammings_packed_u1_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_hammings_symmetric_k: *m = (m_t)&nk_hammings_symmetric_u1_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_jaccards_packed_k: *m = (m_t)&nk_jaccards_packed_u1_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_jaccards_symmetric_k: *m = (m_t)&nk_jaccards_symmetric_u1_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_ICELAKE
    if (v & nk_cap_icelake_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_u1_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_hamming_k: *m = (m_t)&nk_hamming_u1_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u1_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_u1_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_dots_packed_shape_k: *m = (m_t)&nk_dots_packed_shape_u1_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_u1_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_u1_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_u1_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_hammings_packed_k: *m = (m_t)&nk_hammings_packed_u1_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_hammings_symmetric_k: *m = (m_t)&nk_hammings_symmetric_u1_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_jaccards_packed_k: *m = (m_t)&nk_jaccards_packed_u1_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_jaccards_symmetric_k: *m = (m_t)&nk_jaccards_symmetric_u1_icelake, *c = nk_cap_icelake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_u1_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_u1_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_hamming_k: *m = (m_t)&nk_hamming_u1_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u1_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_u1_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_packed_shape_k: *m = (m_t)&nk_dots_packed_shape_u1_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_u1_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_u1_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_u1_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_hammings_packed_k: *m = (m_t)&nk_hammings_packed_u1_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_hammings_symmetric_k: *m = (m_t)&nk_hammings_symmetric_u1_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_jaccards_packed_k: *m = (m_t)&nk_jaccards_packed_u1_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_jaccards_symmetric_k: *m = (m_t)&nk_jaccards_symmetric_u1_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_u1_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
#if NK_TARGET_POWERVSX
    if (v & nk_cap_powervsx_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_u1_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_hamming_k: *m = (m_t)&nk_hamming_u1_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u1_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_u1_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_dots_packed_shape_k: *m = (m_t)&nk_dots_packed_shape_u1_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_u1_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_u1_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_u1_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_hammings_packed_k: *m = (m_t)&nk_hammings_packed_u1_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_hammings_symmetric_k:
            *m = (m_t)&nk_hammings_symmetric_u1_powervsx, *c = nk_cap_powervsx_k;
            return;
        case nk_kernel_jaccards_packed_k: *m = (m_t)&nk_jaccards_packed_u1_powervsx, *c = nk_cap_powervsx_k; return;
        case nk_kernel_jaccards_symmetric_k:
            *m = (m_t)&nk_jaccards_symmetric_u1_powervsx, *c = nk_cap_powervsx_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_LOONGSONASX
    if (v & nk_cap_loongsonasx_k) switch (k) {
        case nk_kernel_hamming_k: *m = (m_t)&nk_hamming_u1_loongsonasx, *c = nk_cap_loongsonasx_k; return;
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u1_loongsonasx, *c = nk_cap_loongsonasx_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_u1_loongsonasx, *c = nk_cap_loongsonasx_k; return;
        case nk_kernel_dots_packed_shape_k:
            *m = (m_t)&nk_dots_packed_shape_u1_loongsonasx, *c = nk_cap_loongsonasx_k;
            return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_u1_loongsonasx, *c = nk_cap_loongsonasx_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_u1_loongsonasx, *c = nk_cap_loongsonasx_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_u1_loongsonasx, *c = nk_cap_loongsonasx_k; return;
        case nk_kernel_hammings_packed_k:
            *m = (m_t)&nk_hammings_packed_u1_loongsonasx, *c = nk_cap_loongsonasx_k;
            return;
        case nk_kernel_hammings_symmetric_k:
            *m = (m_t)&nk_hammings_symmetric_u1_loongsonasx, *c = nk_cap_loongsonasx_k;
            return;
        case nk_kernel_jaccards_packed_k:
            *m = (m_t)&nk_jaccards_packed_u1_loongsonasx, *c = nk_cap_loongsonasx_k;
            return;
        case nk_kernel_jaccards_symmetric_k:
            *m = (m_t)&nk_jaccards_symmetric_u1_loongsonasx, *c = nk_cap_loongsonasx_k;
            return;
        default: break;
        }
#endif
#if NK_TARGET_RVVBB
    if (v & nk_cap_rvvbb_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_u1_rvvbb, *c = nk_cap_rvvbb_k; return;
        case nk_kernel_hamming_k: *m = (m_t)&nk_hamming_u1_rvvbb, *c = nk_cap_rvvbb_k; return;
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u1_rvvbb, *c = nk_cap_rvvbb_k; return;
        default: break;
        }
#endif
#if NK_TARGET_RVV
    if (v & nk_cap_rvv_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_u1_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_hamming_k: *m = (m_t)&nk_hamming_u1_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u1_rvv, *c = nk_cap_rvv_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_u1_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_hamming_k: *m = (m_t)&nk_hamming_u1_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u1_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_pack_size_k: *m = (m_t)&nk_dots_pack_size_u1_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_packed_shape_k: *m = (m_t)&nk_dots_packed_shape_u1_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_pack_k: *m = (m_t)&nk_dots_pack_u1_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_packed_k: *m = (m_t)&nk_dots_packed_u1_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_dots_symmetric_k: *m = (m_t)&nk_dots_symmetric_u1_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_hammings_packed_k: *m = (m_t)&nk_hammings_packed_u1_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_hammings_symmetric_k: *m = (m_t)&nk_hammings_symmetric_u1_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_jaccards_packed_k: *m = (m_t)&nk_jaccards_packed_u1_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_jaccards_symmetric_k: *m = (m_t)&nk_jaccards_symmetric_u1_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_u1_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_u1_serial, *c = nk_cap_serial_k; return;
        default: break;
        }

    // Error fallback - zero capability signals lookup failure
    *m = (m_t)nk_error_dense_, *c = 0;
}

void nk_dispatch_u1_init_(nk_capability_t caps) {
    nk_implementations_t *t = &nk_dispatch_table;
    nk_capability_t used;

    nk_dispatch_u1_find_(caps, nk_kernel_dot_k, (nk_kernel_punned_t *)&t->dot_u1, &used);
    nk_dispatch_u1_find_(caps, nk_kernel_hamming_k, (nk_kernel_punned_t *)&t->hamming_u1, &used);
    nk_dispatch_u1_find_(caps, nk_kernel_jaccard_k, (nk_kernel_punned_t *)&t->jaccard_u1, &used);
    nk_dispatch_u1_find_(caps, nk_kernel_reduce_moments_k, (nk_kernel_punned_t *)&t->reduce_moments_u1, &used);
    nk_dispatch_u1_find_(caps, nk_kernel_reduce_minmax_k, (nk_kernel_punned_t *)&t->reduce_minmax_u1, &used);
    nk_dispatch_u1_find_(caps, nk_kernel_dots_pack_size_k, (nk_kernel_punned_t *)&t->dots_pack_size_u1, &used);
    nk_dispatch_u1_find_(caps, nk_kernel_dots_packed_shape_k, (nk_kernel_punned_t *)&t->dots_packed_shape_u1, &used);
    nk_dispatch_u1_find_(caps, nk_kernel_dots_pack_k, (nk_kernel_punned_t *)&t->dots_pack_u1, &used);
    nk_dispatch_u1_find_(caps, nk_kernel_dots_packed_k, (nk_kernel_punned_t *)&t->dots_packed_u1, &used);
    nk_dispatch_u1_find_(caps, nk_kernel_dots_symmetric_k, (nk_kernel_punned_t *)&t->dots_symmetric_u1, &used);
    nk_dispatch_u1_find_(caps, nk_kernel_hammings_packed_k, (nk_kernel_punned_t *)&t->hammings_packed_u1, &used);
    nk_dispatch_u1_find_(caps, nk_kernel_hammings_symmetric_k, (nk_kernel_punned_t *)&t->hammings_symmetric_u1, &used);
    nk_dispatch_u1_find_(caps, nk_kernel_jaccards_packed_k, (nk_kernel_punned_t *)&t->jaccards_packed_u1, &used);
    nk_dispatch_u1_find_(caps, nk_kernel_jaccards_symmetric_k, (nk_kernel_punned_t *)&t->jaccards_symmetric_u1, &used);
}
