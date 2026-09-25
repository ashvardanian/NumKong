/**
 *  @file c/dispatch_bf16c.c
 *  @author Ash Vardanian
 *  @date June 3, 2024
 *  @brief Dispatch initialization for BF16C data types.
 */
#include "dispatch.h"

void nk_dispatch_bf16c_find_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m, nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NUMKONG_TARGET_NEONBFDOT
    if (v & nk_cap_neonbfdot_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_bf16c_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_bf16c_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_bf16c_neonbfdot, *c = nk_cap_neonbfdot_k; return;
        default: break;
        }
#endif
#if NUMKONG_TARGET_GENOA
    if (v & nk_cap_genoa_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_bf16c_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_bf16c_genoa, *c = nk_cap_genoa_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_bf16c_genoa, *c = nk_cap_genoa_k; return;
        default: break;
        }
#endif
#if NUMKONG_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_bf16c_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_bf16c_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_bf16c_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_bf16c_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_bf16c_serial, *c = nk_cap_serial_k; return;
        default: break;
        }

    // Error fallback - zero capability signals lookup failure
    *m = (m_t)nk_error_dense_, *c = 0;
}

void nk_dispatch_bf16c_init_(nk_capability_t caps) {
    nk_implementations_t *t = &nk_dispatch_table;
    nk_capability_t used;

    nk_dispatch_bf16c_find_(caps, nk_kernel_dot_k, (nk_kernel_punned_t *)&t->dot_bf16c, &used);
    nk_dispatch_bf16c_find_(caps, nk_kernel_vdot_k, (nk_kernel_punned_t *)&t->vdot_bf16c, &used);
    nk_dispatch_bf16c_find_(caps, nk_kernel_bilinear_k, (nk_kernel_punned_t *)&t->bilinear_bf16c, &used);
}
