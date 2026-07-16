/**
 *  @brief Dispatch Initialization for Type Conversions and Scalar Math.
 *  @file c/dispatch_other.c
 *  @author Ash Vardanian
 *  @date February 3, 2026
 */
#include "dispatch.h"

void nk_dispatch_cast_find_(nk_capability_t v, nk_kernel_kind_t k, nk_kernel_punned_t *m, nk_capability_t *c) {
    typedef nk_kernel_punned_t m_t;
#if NK_TARGET_NEON
    if (v & nk_cap_neon_k) switch (k) {
        case nk_kernel_cast_k: *m = (m_t)&nk_cast_neon, *c = nk_cap_neon_k; return;
        case nk_kernel_cast_block_scaled_k: *m = (m_t)&nk_cast_block_scaled_neon, *c = nk_cap_neon_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SAPPHIRE
    if (v & nk_cap_sapphire_k) switch (k) {
        case nk_kernel_cast_k: *m = (m_t)&nk_cast_sapphire, *c = nk_cap_sapphire_k; return;
        default: break;
        }
#endif
#if NK_TARGET_ICELAKE
    if (v & nk_cap_icelake_k) switch (k) {
        case nk_kernel_cast_k: *m = (m_t)&nk_cast_icelake, *c = nk_cap_icelake_k; return;
        case nk_kernel_cast_block_scaled_k: *m = (m_t)&nk_cast_block_scaled_icelake, *c = nk_cap_icelake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_SKYLAKE
    if (v & nk_cap_skylake_k) switch (k) {
        case nk_kernel_cast_k: *m = (m_t)&nk_cast_skylake, *c = nk_cap_skylake_k; return;
        case nk_kernel_cast_block_scaled_k: *m = (m_t)&nk_cast_block_scaled_skylake, *c = nk_cap_skylake_k; return;
        default: break;
        }
#endif
#if NK_TARGET_HASWELL
    if (v & nk_cap_haswell_k) switch (k) {
        case nk_kernel_cast_k: *m = (m_t)&nk_cast_haswell, *c = nk_cap_haswell_k; return;
        case nk_kernel_cast_block_scaled_k: *m = (m_t)&nk_cast_block_scaled_haswell, *c = nk_cap_haswell_k; return;
        default: break;
        }
#endif
#if NK_TARGET_RVV
    if (v & nk_cap_rvv_k) switch (k) {
        case nk_kernel_cast_k: *m = (m_t)&nk_cast_rvv, *c = nk_cap_rvv_k; return;
        default: break;
        }
#endif
    if (v & nk_cap_serial_k) switch (k) {
        case nk_kernel_cast_k: *m = (m_t)&nk_cast_serial, *c = nk_cap_serial_k; return;
        case nk_kernel_cast_block_scaled_k: *m = (m_t)&nk_cast_block_scaled_serial, *c = nk_cap_serial_k; return;
        default: break;
        }

    // Error fallback - zero capability signals lookup failure
    *m = (m_t)nk_error_dense_, *c = 0;
}

void nk_dispatch_cast_init_(nk_capability_t caps) {
    nk_implementations_t *t = &nk_dispatch_table;
    nk_capability_t used;

    // Type casting (buffer-to-buffer)
    nk_dispatch_cast_find_(caps, nk_kernel_cast_k, (nk_kernel_punned_t *)&t->cast, &used);

    // Block-scaled cast (MX / NVFP4)
    nk_dispatch_cast_find_(caps, nk_kernel_cast_block_scaled_k, (nk_kernel_punned_t *)&t->cast_block_scaled, &used);

    // Scalar conversions: bf16 ↔ f32
    t->bf16_to_f32 = &nk_bf16_to_f32_serial;
    t->f32_to_bf16 = &nk_f32_to_bf16_serial;

    // Scalar conversions: f16 ↔ f32
    t->f16_to_f32 = &nk_f16_to_f32_serial;
    t->f32_to_f16 = &nk_f32_to_f16_serial;

#if NK_TARGET_HASWELL
    if (caps & nk_cap_haswell_k) {
        t->f16_to_f32 = &nk_f16_to_f32_haswell;
        t->f32_to_f16 = &nk_f32_to_f16_haswell;
    }
#endif

#if NK_TARGET_SAPPHIRE
    if (caps & nk_cap_sapphire_k) {
        t->f16_to_f32 = &nk_f16_to_f32_sapphire;
        t->f32_to_f16 = &nk_f32_to_f16_sapphire;
    }
#endif

#if NK_TARGET_NEON
    if (caps & nk_cap_neon_k) {
        t->f16_to_f32 = &nk_f16_to_f32_neon;
        t->f32_to_f16 = &nk_f32_to_f16_neon;
    }
#endif

#if NK_TARGET_POWERVSX
    if (caps & nk_cap_powervsx_k) {
        t->f16_to_f32 = &nk_f16_to_f32_powervsx;
        t->f32_to_f16 = &nk_f32_to_f16_powervsx;
    }
#endif

    // Scalar conversions: e5m2, e4m3, e3m2, e2m3 (serial only)
    t->e5m2_to_f32 = &nk_e5m2_to_f32_serial;
    t->f32_to_e5m2 = &nk_f32_to_e5m2_serial;
    t->e4m3_to_f32 = &nk_e4m3_to_f32_serial;
    t->f32_to_e4m3 = &nk_f32_to_e4m3_serial;
    t->e3m2_to_f32 = &nk_e3m2_to_f32_serial;
    t->f32_to_e3m2 = &nk_f32_to_e3m2_serial;
    t->e2m3_to_f32 = &nk_e2m3_to_f32_serial;
    t->f32_to_e2m3 = &nk_f32_to_e2m3_serial;
}

void nk_dispatch_math_init_(nk_capability_t caps) {
    nk_implementations_t *t = &nk_dispatch_table;

    // Scalar math: f64
    t->f64_sqrt = &nk_f64_sqrt_serial;
    t->f64_rsqrt = &nk_f64_rsqrt_serial;
    t->f64_fma = &nk_f64_fma_serial;

    // Scalar math: f32
    t->f32_sqrt = &nk_f32_sqrt_serial;
    t->f32_rsqrt = &nk_f32_rsqrt_serial;
    t->f32_fma = &nk_f32_fma_serial;

#if NK_TARGET_V128RELAXED
    if (caps & nk_cap_v128relaxed_k) {
        t->f64_rsqrt = &nk_f64_rsqrt_v128relaxed;
        t->f64_fma = &nk_f64_fma_v128relaxed;
        t->f32_rsqrt = &nk_f32_rsqrt_v128relaxed;
        t->f32_fma = &nk_f32_fma_v128relaxed;
    }
#endif

#if NK_TARGET_HASWELL
    if (caps & nk_cap_haswell_k) {
        t->f64_sqrt = &nk_f64_sqrt_haswell;
        t->f64_rsqrt = &nk_f64_rsqrt_haswell;
        t->f64_fma = &nk_f64_fma_haswell;
        t->f32_sqrt = &nk_f32_sqrt_haswell;
        t->f32_rsqrt = &nk_f32_rsqrt_haswell;
        t->f32_fma = &nk_f32_fma_haswell;
    }
#endif

#if NK_TARGET_NEON
    if (caps & nk_cap_neon_k) {
        t->f64_sqrt = &nk_f64_sqrt_neon;
        t->f64_rsqrt = &nk_f64_rsqrt_neon;
        t->f64_fma = &nk_f64_fma_neon;
        t->f32_sqrt = &nk_f32_sqrt_neon;
        t->f32_rsqrt = &nk_f32_rsqrt_neon;
        t->f32_fma = &nk_f32_fma_neon;
    }
#endif

#if NK_TARGET_POWERVSX
    if (caps & nk_cap_powervsx_k) {
        t->f64_sqrt = &nk_f64_sqrt_powervsx;
        t->f64_rsqrt = &nk_f64_rsqrt_powervsx;
        t->f64_fma = &nk_f64_fma_powervsx;
        t->f32_sqrt = &nk_f32_sqrt_powervsx;
        t->f32_rsqrt = &nk_f32_rsqrt_powervsx;
        t->f32_fma = &nk_f32_fma_powervsx;
    }
#endif

    // Scalar math: f16
    t->f16_sqrt = &nk_f16_sqrt_serial;
    t->f16_rsqrt = &nk_f16_rsqrt_serial;
    t->f16_fma = &nk_f16_fma_serial;

#if NK_TARGET_HASWELL
    if (caps & nk_cap_haswell_k) {
        t->f16_sqrt = &nk_f16_sqrt_haswell;
        t->f16_rsqrt = &nk_f16_rsqrt_haswell;
        t->f16_fma = &nk_f16_fma_haswell;
    }
#endif

#if NK_TARGET_NEONHALF
    if (caps & nk_cap_neonhalf_k) {
        t->f16_sqrt = &nk_f16_sqrt_neonhalf;
        t->f16_rsqrt = &nk_f16_rsqrt_neonhalf;
        t->f16_fma = &nk_f16_fma_neonhalf;
    }
#endif

#if NK_TARGET_SAPPHIRE
    if (caps & nk_cap_sapphire_k) {
        t->f16_sqrt = &nk_f16_sqrt_sapphire;
        t->f16_rsqrt = &nk_f16_rsqrt_sapphire;
        t->f16_fma = &nk_f16_fma_sapphire;
    }
#endif

#if NK_TARGET_RVV
    if (caps & nk_cap_rvv_k) {
        t->f64_fma = &nk_f64_fma_rvv;
        t->f32_fma = &nk_f32_fma_rvv;
    }
#endif

    // Saturating arithmetic
    t->i64_saturating_add = &nk_i64_saturating_add_serial;
    t->i64_saturating_mul = &nk_i64_saturating_mul_serial;
    t->i32_saturating_add = &nk_i32_saturating_add_serial;
    t->i32_saturating_mul = &nk_i32_saturating_mul_serial;
    t->i16_saturating_add = &nk_i16_saturating_add_serial;
    t->i16_saturating_mul = &nk_i16_saturating_mul_serial;
    t->i8_saturating_add = &nk_i8_saturating_add_serial;
    t->i8_saturating_mul = &nk_i8_saturating_mul_serial;
    t->i4x2_saturating_add = &nk_i4x2_saturating_add_serial;
    t->i4x2_saturating_mul = &nk_i4x2_saturating_mul_serial;
    t->u64_saturating_add = &nk_u64_saturating_add_serial;
    t->u64_saturating_mul = &nk_u64_saturating_mul_serial;
    t->u32_saturating_add = &nk_u32_saturating_add_serial;
    t->u32_saturating_mul = &nk_u32_saturating_mul_serial;
    t->u16_saturating_add = &nk_u16_saturating_add_serial;
    t->u16_saturating_mul = &nk_u16_saturating_mul_serial;
    t->u8_saturating_add = &nk_u8_saturating_add_serial;
    t->u8_saturating_mul = &nk_u8_saturating_mul_serial;
    t->u4x2_saturating_add = &nk_u4x2_saturating_add_serial;
    t->u4x2_saturating_mul = &nk_u4x2_saturating_mul_serial;

#if NK_TARGET_RVV
    if (caps & nk_cap_rvv_k) {
        t->i64_saturating_add = &nk_i64_saturating_add_rvv;
        t->i64_saturating_mul = &nk_i64_saturating_mul_rvv;
        t->i32_saturating_add = &nk_i32_saturating_add_rvv;
        t->i32_saturating_mul = &nk_i32_saturating_mul_rvv;
        t->i16_saturating_add = &nk_i16_saturating_add_rvv;
        t->i16_saturating_mul = &nk_i16_saturating_mul_rvv;
        t->i8_saturating_add = &nk_i8_saturating_add_rvv;
        t->i8_saturating_mul = &nk_i8_saturating_mul_rvv;
        t->u64_saturating_add = &nk_u64_saturating_add_rvv;
        t->u64_saturating_mul = &nk_u64_saturating_mul_rvv;
        t->u32_saturating_add = &nk_u32_saturating_add_rvv;
        t->u32_saturating_mul = &nk_u32_saturating_mul_rvv;
        t->u16_saturating_add = &nk_u16_saturating_add_rvv;
        t->u16_saturating_mul = &nk_u16_saturating_mul_rvv;
        t->u8_saturating_add = &nk_u8_saturating_add_rvv;
        t->u8_saturating_mul = &nk_u8_saturating_mul_rvv;
    }
#endif

#if NK_TARGET_NEON
    if (caps & nk_cap_neon_k) {
        t->i64_saturating_add = &nk_i64_saturating_add_neon;
        t->i64_saturating_mul = &nk_i64_saturating_mul_neon;
        t->i32_saturating_add = &nk_i32_saturating_add_neon;
        t->i16_saturating_add = &nk_i16_saturating_add_neon;
        t->i8_saturating_add = &nk_i8_saturating_add_neon;
        t->u64_saturating_add = &nk_u64_saturating_add_neon;
        t->u64_saturating_mul = &nk_u64_saturating_mul_neon;
        t->u32_saturating_add = &nk_u32_saturating_add_neon;
        t->u16_saturating_add = &nk_u16_saturating_add_neon;
        t->u8_saturating_add = &nk_u8_saturating_add_neon;
    }
#endif

#if NK_TARGET_HASWELL
    if (caps & nk_cap_haswell_k) {
        t->i64_saturating_mul = &nk_i64_saturating_mul_haswell;
        t->i16_saturating_add = &nk_i16_saturating_add_haswell;
        t->i8_saturating_add = &nk_i8_saturating_add_haswell;
        t->u64_saturating_mul = &nk_u64_saturating_mul_haswell;
        t->u16_saturating_add = &nk_u16_saturating_add_haswell;
        t->u8_saturating_add = &nk_u8_saturating_add_haswell;
    }
#endif

    // Conversion-free ordering for mini-floats
    t->bf16_order = &nk_bf16_order_serial;
    t->f16_order = &nk_f16_order_serial;
    t->e5m2_order = &nk_e5m2_order_serial;
    t->e4m3_order = &nk_e4m3_order_serial;
    t->e3m2_order = &nk_e3m2_order_serial;
    t->e2m3_order = &nk_e2m3_order_serial;

#if NK_TARGET_SAPPHIRE
    if (caps & nk_cap_sapphire_k) { t->f16_order = &nk_f16_order_sapphire; }
#endif
}

// Scalar conversion dispatch functions

NK_API_RUNTIME void nk_bf16_to_f32(nk_bf16_t const *src, nk_f32_t *dest) { nk_dispatch_table.bf16_to_f32(src, dest); }
NK_API_RUNTIME void nk_f32_to_bf16(nk_f32_t const *src, nk_bf16_t *dest) { nk_dispatch_table.f32_to_bf16(src, dest); }
NK_API_RUNTIME void nk_f16_to_f32(nk_f16_t const *src, nk_f32_t *dest) { nk_dispatch_table.f16_to_f32(src, dest); }
NK_API_RUNTIME void nk_f32_to_f16(nk_f32_t const *src, nk_f16_t *dest) { nk_dispatch_table.f32_to_f16(src, dest); }
NK_API_RUNTIME void nk_e5m2_to_f32(nk_e5m2_t const *src, nk_f32_t *dest) { nk_dispatch_table.e5m2_to_f32(src, dest); }
NK_API_RUNTIME void nk_f32_to_e5m2(nk_f32_t const *src, nk_e5m2_t *dest) { nk_dispatch_table.f32_to_e5m2(src, dest); }
NK_API_RUNTIME void nk_e4m3_to_f32(nk_e4m3_t const *src, nk_f32_t *dest) { nk_dispatch_table.e4m3_to_f32(src, dest); }
NK_API_RUNTIME void nk_f32_to_e4m3(nk_f32_t const *src, nk_e4m3_t *dest) { nk_dispatch_table.f32_to_e4m3(src, dest); }
NK_API_RUNTIME void nk_e3m2_to_f32(nk_e3m2_t const *src, nk_f32_t *dest) { nk_dispatch_table.e3m2_to_f32(src, dest); }
NK_API_RUNTIME void nk_f32_to_e3m2(nk_f32_t const *src, nk_e3m2_t *dest) { nk_dispatch_table.f32_to_e3m2(src, dest); }
NK_API_RUNTIME void nk_e2m3_to_f32(nk_e2m3_t const *src, nk_f32_t *dest) { nk_dispatch_table.e2m3_to_f32(src, dest); }
NK_API_RUNTIME void nk_f32_to_e2m3(nk_f32_t const *src, nk_e2m3_t *dest) { nk_dispatch_table.f32_to_e2m3(src, dest); }

// Scalar math dispatch functions

NK_API_RUNTIME nk_f64_t nk_f64_sqrt(nk_f64_t x) { return nk_dispatch_table.f64_sqrt(x); }
NK_API_RUNTIME nk_f64_t nk_f64_rsqrt(nk_f64_t x) { return nk_dispatch_table.f64_rsqrt(x); }
NK_API_RUNTIME nk_f64_t nk_f64_fma(nk_f64_t a, nk_f64_t b, nk_f64_t c) { return nk_dispatch_table.f64_fma(a, b, c); }
NK_API_RUNTIME nk_f32_t nk_f32_sqrt(nk_f32_t x) { return nk_dispatch_table.f32_sqrt(x); }
NK_API_RUNTIME nk_f32_t nk_f32_rsqrt(nk_f32_t x) { return nk_dispatch_table.f32_rsqrt(x); }
NK_API_RUNTIME nk_f32_t nk_f32_fma(nk_f32_t a, nk_f32_t b, nk_f32_t c) { return nk_dispatch_table.f32_fma(a, b, c); }
NK_API_RUNTIME nk_f16_t nk_f16_sqrt(nk_f16_t x) { return nk_dispatch_table.f16_sqrt(x); }
NK_API_RUNTIME nk_f16_t nk_f16_rsqrt(nk_f16_t x) { return nk_dispatch_table.f16_rsqrt(x); }
NK_API_RUNTIME nk_f16_t nk_f16_fma(nk_f16_t a, nk_f16_t b, nk_f16_t c) { return nk_dispatch_table.f16_fma(a, b, c); }

// Saturating arithmetic dispatch functions

NK_API_RUNTIME nk_i64_t nk_i64_saturating_add(nk_i64_t a, nk_i64_t b) {
    return nk_dispatch_table.i64_saturating_add(a, b);
}
NK_API_RUNTIME nk_i64_t nk_i64_saturating_mul(nk_i64_t a, nk_i64_t b) {
    return nk_dispatch_table.i64_saturating_mul(a, b);
}
NK_API_RUNTIME nk_i32_t nk_i32_saturating_add(nk_i32_t a, nk_i32_t b) {
    return nk_dispatch_table.i32_saturating_add(a, b);
}
NK_API_RUNTIME nk_i32_t nk_i32_saturating_mul(nk_i32_t a, nk_i32_t b) {
    return nk_dispatch_table.i32_saturating_mul(a, b);
}
NK_API_RUNTIME nk_i16_t nk_i16_saturating_add(nk_i16_t a, nk_i16_t b) {
    return nk_dispatch_table.i16_saturating_add(a, b);
}
NK_API_RUNTIME nk_i16_t nk_i16_saturating_mul(nk_i16_t a, nk_i16_t b) {
    return nk_dispatch_table.i16_saturating_mul(a, b);
}
NK_API_RUNTIME nk_i8_t nk_i8_saturating_add(nk_i8_t a, nk_i8_t b) { return nk_dispatch_table.i8_saturating_add(a, b); }
NK_API_RUNTIME nk_i8_t nk_i8_saturating_mul(nk_i8_t a, nk_i8_t b) { return nk_dispatch_table.i8_saturating_mul(a, b); }
NK_API_RUNTIME nk_i4x2_t nk_i4x2_saturating_add(nk_i4x2_t a, nk_i4x2_t b) {
    return nk_dispatch_table.i4x2_saturating_add(a, b);
}
NK_API_RUNTIME nk_i4x2_t nk_i4x2_saturating_mul(nk_i4x2_t a, nk_i4x2_t b) {
    return nk_dispatch_table.i4x2_saturating_mul(a, b);
}
NK_API_RUNTIME nk_u64_t nk_u64_saturating_add(nk_u64_t a, nk_u64_t b) {
    return nk_dispatch_table.u64_saturating_add(a, b);
}
NK_API_RUNTIME nk_u64_t nk_u64_saturating_mul(nk_u64_t a, nk_u64_t b) {
    return nk_dispatch_table.u64_saturating_mul(a, b);
}
NK_API_RUNTIME nk_u32_t nk_u32_saturating_add(nk_u32_t a, nk_u32_t b) {
    return nk_dispatch_table.u32_saturating_add(a, b);
}
NK_API_RUNTIME nk_u32_t nk_u32_saturating_mul(nk_u32_t a, nk_u32_t b) {
    return nk_dispatch_table.u32_saturating_mul(a, b);
}
NK_API_RUNTIME nk_u16_t nk_u16_saturating_add(nk_u16_t a, nk_u16_t b) {
    return nk_dispatch_table.u16_saturating_add(a, b);
}
NK_API_RUNTIME nk_u16_t nk_u16_saturating_mul(nk_u16_t a, nk_u16_t b) {
    return nk_dispatch_table.u16_saturating_mul(a, b);
}
NK_API_RUNTIME nk_u8_t nk_u8_saturating_add(nk_u8_t a, nk_u8_t b) { return nk_dispatch_table.u8_saturating_add(a, b); }
NK_API_RUNTIME nk_u8_t nk_u8_saturating_mul(nk_u8_t a, nk_u8_t b) { return nk_dispatch_table.u8_saturating_mul(a, b); }
NK_API_RUNTIME nk_u4x2_t nk_u4x2_saturating_add(nk_u4x2_t a, nk_u4x2_t b) {
    return nk_dispatch_table.u4x2_saturating_add(a, b);
}
NK_API_RUNTIME nk_u4x2_t nk_u4x2_saturating_mul(nk_u4x2_t a, nk_u4x2_t b) {
    return nk_dispatch_table.u4x2_saturating_mul(a, b);
}

// Ordering dispatch functions

NK_API_RUNTIME int nk_bf16_order(nk_bf16_t a, nk_bf16_t b) { return nk_dispatch_table.bf16_order(a, b); }
NK_API_RUNTIME int nk_f16_order(nk_f16_t a, nk_f16_t b) { return nk_dispatch_table.f16_order(a, b); }
NK_API_RUNTIME int nk_e5m2_order(nk_e5m2_t a, nk_e5m2_t b) { return nk_dispatch_table.e5m2_order(a, b); }
NK_API_RUNTIME int nk_e4m3_order(nk_e4m3_t a, nk_e4m3_t b) { return nk_dispatch_table.e4m3_order(a, b); }
NK_API_RUNTIME int nk_e3m2_order(nk_e3m2_t a, nk_e3m2_t b) { return nk_dispatch_table.e3m2_order(a, b); }
NK_API_RUNTIME int nk_e2m3_order(nk_e2m3_t a, nk_e2m3_t b) { return nk_dispatch_table.e2m3_order(a, b); }
