/**
 *  @file c/dispatch_f64c.c
 *  @author Ash Vardanian
 *  @date March 3, 2024
 *  @brief Dispatch initialization for F64C data types.
 */
#include "dispatch.h"

void nk_dispatch_f64c_find_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m, nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_V128RELAXED
    if (v & nk_cap_v128relaxed_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f64c_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_f64c_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SMEF64
    if (v & nk_cap_smef64_k) switch (k) {
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f64c_smef64, *c = nk_cap_smef64_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SVE
    if (v & nk_cap_sve_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f64c_sve, *c = nk_cap_sve_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_f64c_sve, *c = nk_cap_sve_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f64c_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_f64c_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_f64c_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_f64c_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_f64c_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_RVV
    if (v & nk_cap_rvv_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f64c_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_f64c_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_f64c_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_f64c_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_f64c_rvv, *c = nk_cap_rvv_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f64c_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_f64c_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f64c_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_f64c_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_f64c_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_f64c_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f64c_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_f64c_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_f64c_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_f64c_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_f64c_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_dot_k: *m = (m_t)&nk_dot_f64c_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_vdot_k: *m = (m_t)&nk_vdot_f64c_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_bilinear_k: *m = (m_t)&nk_bilinear_f64c_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_f64c_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_f64c_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_f64c_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_f64c_serial, *c = nk_cap_serial_k; return;
        default: break;
        }

    // Error fallback - zero capability signals lookup failure
    *m = (m_t)nk_error_dense_, *c = 0;
}

void nk_dispatch_f64c_init_(nk_capability_t caps) {
    nk_implementations_t *t = &nk_dispatch_table;
    nk_capability_t used;

    nk_dispatch_f64c_find_(caps, nk_kernel_dot_k, (nk_kernel_punned_t *)&t->dot_f64c, &used);
    nk_dispatch_f64c_find_(caps, nk_kernel_vdot_k, (nk_kernel_punned_t *)&t->vdot_f64c, &used);
    nk_dispatch_f64c_find_(caps, nk_kernel_bilinear_k, (nk_kernel_punned_t *)&t->bilinear_f64c, &used);
    nk_dispatch_f64c_find_(caps, nk_kernel_each_scale_k, (nk_kernel_punned_t *)&t->each_scale_f64c, &used);
    nk_dispatch_f64c_find_(caps, nk_kernel_each_sum_k, (nk_kernel_punned_t *)&t->each_sum_f64c, &used);
    nk_dispatch_f64c_find_(caps, nk_kernel_each_blend_k, (nk_kernel_punned_t *)&t->each_blend_f64c, &used);
    nk_dispatch_f64c_find_(caps, nk_kernel_each_fma_k, (nk_kernel_punned_t *)&t->each_fma_f64c, &used);
}
