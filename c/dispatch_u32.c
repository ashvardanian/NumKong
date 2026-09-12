/**
 *  @brief Dispatch Initialization for U32 Data Types.
 *  @file c/dispatch_u32.c
 *  @author Ash Vardanian
 *  @date February 3, 2026
 */
#include "dispatch.h"

void nk_dispatch_u32_find_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m, nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_V128RELAXED
    if (v & nk_cap_v128relaxed_k) switch (k) {
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_u32_v128relaxed, *c = nk_cap_v128relaxed_k; return;
        default: break;
        }
#endif
#if NK_TARGET_V128
    if (v & nk_cap_v128_k) switch (k) {
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u32_v128, *c = nk_cap_v128_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_u32_v128, *c = nk_cap_v128_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SVE2
    if (v & nk_cap_sve2_k) switch (k) {
        case nk_kernel_sparse_intersect_k: *m = (m_t)&nk_sparse_intersect_u32_sve2, *c = nk_cap_sve2_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SVE
    if (v & nk_cap_sve_k) switch (k) {
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u32_sve, *c = nk_cap_sve_k; return;
        default: break;
        }
#endif
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_kernel_sparse_intersect_k: *m = (m_t)&nk_sparse_intersect_u32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_u32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_u32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_u32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_u32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_u32_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u32_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_TURIN
    if (v & nk_cap_turin_k) switch (k) {
        case nk_kernel_sparse_intersect_k: *m = (m_t)&nk_sparse_intersect_u32_turin, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_ICELAKE
    if (v & nk_cap_icelake_k) switch (k) {
        case nk_kernel_sparse_intersect_k: *m = (m_t)&nk_sparse_intersect_u32_icelake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_u32_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u32_icelake, *c = nk_cap_icelake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_u32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_u32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_u32_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_u32_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_u32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_u32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_u32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_u32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_u32_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u32_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
#if NK_TARGET_RVV
    if (v & nk_cap_rvv_k) switch (k) {
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u32_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_u32_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_u32_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_u32_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_u32_rvv, *c = nk_cap_rvv_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_u32_rvv, *c = nk_cap_rvv_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_sparse_intersect_k: *m = (m_t)&nk_sparse_intersect_u32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_jaccard_k: *m = (m_t)&nk_jaccard_u32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_fma_k: *m = (m_t)&nk_each_fma_u32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_scale_k: *m = (m_t)&nk_each_scale_u32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_sum_k: *m = (m_t)&nk_each_sum_u32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_each_blend_k: *m = (m_t)&nk_each_blend_u32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_moments_k: *m = (m_t)&nk_reduce_moments_u32_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_reduce_minmax_k: *m = (m_t)&nk_reduce_minmax_u32_serial, *c = nk_cap_serial_k; return;
        default: break;
        }

    // Error fallback - zero capability signals lookup failure
    *m = (m_t)nk_error_dense_, *c = 0;
}

void nk_dispatch_u32_init_(nk_capability_t caps) {
    nk_implementations_t *t = &nk_dispatch_table;
    nk_capability_t used;

    nk_dispatch_u32_find_(caps, nk_kernel_jaccard_k, (nk_kernel_punned_t *)&t->jaccard_u32, &used);
    nk_dispatch_u32_find_(caps, nk_kernel_sparse_intersect_k, (nk_kernel_punned_t *)&t->sparse_intersect_u32, &used);
    nk_dispatch_u32_find_(caps, nk_kernel_each_scale_k, (nk_kernel_punned_t *)&t->each_scale_u32, &used);
    nk_dispatch_u32_find_(caps, nk_kernel_each_sum_k, (nk_kernel_punned_t *)&t->each_sum_u32, &used);
    nk_dispatch_u32_find_(caps, nk_kernel_each_blend_k, (nk_kernel_punned_t *)&t->each_blend_u32, &used);
    nk_dispatch_u32_find_(caps, nk_kernel_each_fma_k, (nk_kernel_punned_t *)&t->each_fma_u32, &used);
    nk_dispatch_u32_find_(caps, nk_kernel_reduce_moments_k, (nk_kernel_punned_t *)&t->reduce_moments_u32, &used);
    nk_dispatch_u32_find_(caps, nk_kernel_reduce_minmax_k, (nk_kernel_punned_t *)&t->reduce_minmax_u32, &used);
}
