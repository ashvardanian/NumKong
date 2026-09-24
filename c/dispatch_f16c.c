/**
 *  @file c/dispatch_f16c.c
 *  @author Ash Vardanian
 *  @date March 3, 2024
 *  @brief Dispatch initialization for F16C data types.
 */
#include "dispatch.h"

void nk_dispatch_f16c_find_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m, nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_SVEHALF
    if (v & nk_cap_svehalf_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f16c_svehalf, *c = nk_cap_svehalf_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_f16c_svehalf, *c = nk_cap_svehalf_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEONFHM
    if (v & nk_cap_neonfhm_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f16c_neonfhm, *c = nk_cap_neonfhm_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_f16c_neonfhm, *c = nk_cap_neonfhm_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f16c_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_f16c_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f16c_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f16c_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_f16c_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f16c_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_f16c_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f16c_serial, *c = nk_cap_serial_k; return;
        default: break;
        }

    // Error fallback - zero capability signals lookup failure
    *m = (m_t)nk_error_dense_, *c = 0;
}

void nk_dispatch_f16c_init_(nk_capability_t caps) {
    nk_implementations_t *t = &nk_dispatch_table;
    nk_capability_t used;

    nk_dispatch_f16c_find_(caps, nk_kernel_dot_k, (nk_kernel_punned_t *)&t->dot_f16c, &used);
    nk_dispatch_f16c_find_(caps, nk_kernel_vdot_k, (nk_kernel_punned_t *)&t->vdot_f16c, &used);
    nk_dispatch_f16c_find_(caps, nk_kernel_bilinear_k, (nk_kernel_punned_t *)&t->bilinear_f16c, &used);
}
