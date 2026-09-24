/**
 *  @brief SWAR-accelerated Elementwise Arithmetic for SIMD-free CPUs.
 *  @file include/numkong/each/serial.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/each.h
 */
#ifndef NK_EACH_SERIAL_H
#define NK_EACH_SERIAL_H

#include "numkong/types.h"
#include "numkong/cast/serial.h" // `nk_f16_to_f32_serial`

#if defined(__cplusplus)
extern "C" {
#endif

#define nk_define_each_scale_(input_type, accumulator_type, load_and_convert, convert_and_store) \
    NK_PUBLIC void nk_each_scale_##input_type##_serial(                                          \
        nk_##input_type##_t const *a, nk_size_t n, nk_##accumulator_type##_t const *alpha,       \
        nk_##accumulator_type##_t const *beta, nk_##input_type##_t *result) {                    \
        nk_##accumulator_type##_t alpha_val = *alpha;                                            \
        nk_##accumulator_type##_t beta_val = *beta;                                              \
        nk_##accumulator_type##_t ai, sum;                                                       \
        for (nk_size_t i = 0; i != n; ++i) {                                                     \
            load_and_convert(a + i, &ai);                                                        \
            sum = (nk_##accumulator_type##_t)(alpha_val * ai + beta_val);                        \
            convert_and_store(&sum, result + i);                                                 \
        }                                                                                        \
    }
#define nk_define_each_sum_(input_type, accumulator_type, load_and_convert, convert_and_store)                   \
    NK_PUBLIC void nk_each_sum_##input_type##_serial(nk_##input_type##_t const *a, nk_##input_type##_t const *b, \
                                                     nk_size_t n, nk_##input_type##_t *result) {                 \
        nk_##accumulator_type##_t ai, bi, sum;                                                                   \
        for (nk_size_t i = 0; i != n; ++i) {                                                                     \
            load_and_convert(a + i, &ai);                                                                        \
            load_and_convert(b + i, &bi);                                                                        \
            sum = ai + bi;                                                                                       \
            convert_and_store(&sum, result + i);                                                                 \
        }                                                                                                        \
    }

#define nk_define_each_blend_(input_type, accumulator_type, load_and_convert, convert_and_store)                      \
    NK_PUBLIC void nk_each_blend_##input_type##_serial(                                                               \
        nk_##input_type##_t const *a, nk_##input_type##_t const *b, nk_size_t n,                                      \
        nk_##accumulator_type##_t const *alpha, nk_##accumulator_type##_t const *beta, nk_##input_type##_t *result) { \
        nk_##accumulator_type##_t alpha_val = *alpha;                                                                 \
        nk_##accumulator_type##_t beta_val = *beta;                                                                   \
        nk_##accumulator_type##_t ai, bi, ai_scaled, bi_scaled, sum;                                                  \
        for (nk_size_t i = 0; i != n; ++i) {                                                                          \
            load_and_convert(a + i, &ai);                                                                             \
            load_and_convert(b + i, &bi);                                                                             \
            ai_scaled = ai * alpha_val;                                                                               \
            bi_scaled = bi * beta_val;                                                                                \
            sum = ai_scaled + bi_scaled;                                                                              \
            convert_and_store(&sum, result + i);                                                                      \
        }                                                                                                             \
    }

#define nk_define_each_fma_(input_type, accumulator_type, load_and_convert, convert_and_store)                        \
    NK_PUBLIC void nk_each_fma_##input_type##_serial(                                                                 \
        nk_##input_type##_t const *a, nk_##input_type##_t const *b, nk_##input_type##_t const *c, nk_size_t n,        \
        nk_##accumulator_type##_t const *alpha, nk_##accumulator_type##_t const *beta, nk_##input_type##_t *result) { \
        nk_##accumulator_type##_t alpha_val = *alpha;                                                                 \
        nk_##accumulator_type##_t beta_val = *beta;                                                                   \
        nk_##accumulator_type##_t ai, bi, ci, abi_scaled, ci_scaled, sum;                                             \
        for (nk_size_t i = 0; i != n; ++i) {                                                                          \
            load_and_convert(a + i, &ai);                                                                             \
            load_and_convert(b + i, &bi);                                                                             \
            load_and_convert(c + i, &ci);                                                                             \
            abi_scaled = ai * bi * alpha_val;                                                                         \
            ci_scaled = ci * beta_val;                                                                                \
            sum = abi_scaled + ci_scaled;                                                                             \
            convert_and_store(&sum, result + i);                                                                      \
        }                                                                                                             \
    }

nk_define_each_sum_(f64, f64, nk_assign_from_to_, nk_assign_from_to_)        // nk_each_sum_f64_serial
nk_define_each_sum_(f32, f32, nk_assign_from_to_, nk_assign_from_to_)        // nk_each_sum_f32_serial
nk_define_each_sum_(f16, f32, nk_f16_to_f32_serial, nk_f32_to_f16_serial)    // nk_each_sum_f16_serial
nk_define_each_sum_(bf16, f32, nk_bf16_to_f32_serial, nk_f32_to_bf16_serial) // nk_each_sum_bf16_serial
nk_define_each_sum_(e4m3, f32, nk_e4m3_to_f32_serial, nk_f32_to_e4m3_serial) // nk_each_sum_e4m3_serial
nk_define_each_sum_(e5m2, f32, nk_e5m2_to_f32_serial, nk_f32_to_e5m2_serial) // nk_each_sum_e5m2_serial
nk_define_each_sum_(e2m3, f32, nk_e2m3_to_f32_serial, nk_f32_to_e2m3_serial) // nk_each_sum_e2m3_serial
nk_define_each_sum_(e3m2, f32, nk_e3m2_to_f32_serial, nk_f32_to_e3m2_serial) // nk_each_sum_e3m2_serial
nk_define_each_sum_(i8, i64, nk_assign_from_to_, nk_i64_to_i8_serial)        // nk_each_sum_i8_serial
nk_define_each_sum_(u8, i64, nk_assign_from_to_, nk_i64_to_u8_serial)        // nk_each_sum_u8_serial
nk_define_each_sum_(i16, i64, nk_assign_from_to_, nk_i64_to_i16_serial)      // nk_each_sum_i16_serial
nk_define_each_sum_(u16, i64, nk_assign_from_to_, nk_i64_to_u16_serial)      // nk_each_sum_u16_serial
nk_define_each_sum_(i32, i64, nk_assign_from_to_, nk_i64_to_i32_serial)      // nk_each_sum_i32_serial
nk_define_each_sum_(u32, i64, nk_assign_from_to_, nk_i64_to_u32_serial)      // nk_each_sum_u32_serial
nk_define_each_sum_(i64, i64, nk_assign_from_to_, nk_assign_from_to_)        // nk_each_sum_i64_serial
nk_define_each_sum_(u64, u64, nk_assign_from_to_, nk_assign_from_to_)        // nk_each_sum_u64_serial

nk_define_each_scale_(f64, f64, nk_assign_from_to_, nk_assign_from_to_)        // nk_each_scale_f64_serial
nk_define_each_scale_(f32, f32, nk_assign_from_to_, nk_assign_from_to_)        // nk_each_scale_f32_serial
nk_define_each_scale_(f16, f32, nk_f16_to_f32_serial, nk_f32_to_f16_serial)    // nk_each_scale_f16_serial
nk_define_each_scale_(bf16, f32, nk_bf16_to_f32_serial, nk_f32_to_bf16_serial) // nk_each_scale_bf16_serial
nk_define_each_scale_(e4m3, f32, nk_e4m3_to_f32_serial, nk_f32_to_e4m3_serial) // nk_each_scale_e4m3_serial
nk_define_each_scale_(e5m2, f32, nk_e5m2_to_f32_serial, nk_f32_to_e5m2_serial) // nk_each_scale_e5m2_serial
nk_define_each_scale_(e2m3, f32, nk_e2m3_to_f32_serial, nk_f32_to_e2m3_serial) // nk_each_scale_e2m3_serial
nk_define_each_scale_(e3m2, f32, nk_e3m2_to_f32_serial, nk_f32_to_e3m2_serial) // nk_each_scale_e3m2_serial
nk_define_each_scale_(i8, f32, nk_assign_from_to_, nk_f32_to_i8_serial)        // nk_each_scale_i8_serial
nk_define_each_scale_(u8, f32, nk_assign_from_to_, nk_f32_to_u8_serial)        // nk_each_scale_u8_serial
nk_define_each_scale_(i16, f32, nk_assign_from_to_, nk_f32_to_i16_serial)      // nk_each_scale_i16_serial
nk_define_each_scale_(u16, f32, nk_assign_from_to_, nk_f32_to_u16_serial)      // nk_each_scale_u16_serial
nk_define_each_scale_(i32, f64, nk_assign_from_to_, nk_f64_to_i32_serial)      // nk_each_scale_i32_serial
nk_define_each_scale_(u32, f64, nk_assign_from_to_, nk_f64_to_u32_serial)      // nk_each_scale_u32_serial
nk_define_each_scale_(i64, f64, nk_assign_from_to_, nk_f64_to_i64_serial)      // nk_each_scale_i64_serial
nk_define_each_scale_(u64, f64, nk_assign_from_to_, nk_f64_to_u64_serial)      // nk_each_scale_u64_serial

nk_define_each_blend_(f64, f64, nk_assign_from_to_, nk_assign_from_to_)        // nk_each_blend_f64_serial
nk_define_each_blend_(f32, f32, nk_assign_from_to_, nk_assign_from_to_)        // nk_each_blend_f32_serial
nk_define_each_blend_(f16, f32, nk_f16_to_f32_serial, nk_f32_to_f16_serial)    // nk_each_blend_f16_serial
nk_define_each_blend_(bf16, f32, nk_bf16_to_f32_serial, nk_f32_to_bf16_serial) // nk_each_blend_bf16_serial
nk_define_each_blend_(e4m3, f32, nk_e4m3_to_f32_serial, nk_f32_to_e4m3_serial) // nk_each_blend_e4m3_serial
nk_define_each_blend_(e5m2, f32, nk_e5m2_to_f32_serial, nk_f32_to_e5m2_serial) // nk_each_blend_e5m2_serial
nk_define_each_blend_(e2m3, f32, nk_e2m3_to_f32_serial, nk_f32_to_e2m3_serial) // nk_each_blend_e2m3_serial
nk_define_each_blend_(e3m2, f32, nk_e3m2_to_f32_serial, nk_f32_to_e3m2_serial) // nk_each_blend_e3m2_serial
nk_define_each_blend_(i8, f32, nk_assign_from_to_, nk_f32_to_i8_serial)        // nk_each_blend_i8_serial
nk_define_each_blend_(u8, f32, nk_assign_from_to_, nk_f32_to_u8_serial)        // nk_each_blend_u8_serial
nk_define_each_blend_(i16, f32, nk_assign_from_to_, nk_f32_to_i16_serial)      // nk_each_blend_i16_serial
nk_define_each_blend_(u16, f32, nk_assign_from_to_, nk_f32_to_u16_serial)      // nk_each_blend_u16_serial
nk_define_each_blend_(i32, f64, nk_assign_from_to_, nk_f64_to_i32_serial)      // nk_each_blend_i32_serial
nk_define_each_blend_(u32, f64, nk_assign_from_to_, nk_f64_to_u32_serial)      // nk_each_blend_u32_serial
nk_define_each_blend_(i64, f64, nk_assign_from_to_, nk_f64_to_i64_serial)      // nk_each_blend_i64_serial
nk_define_each_blend_(u64, f64, nk_assign_from_to_, nk_f64_to_u64_serial)      // nk_each_blend_u64_serial

nk_define_each_fma_(f64, f64, nk_assign_from_to_, nk_assign_from_to_)        // nk_each_fma_f64_serial
nk_define_each_fma_(f32, f32, nk_assign_from_to_, nk_assign_from_to_)        // nk_each_fma_f32_serial
nk_define_each_fma_(f16, f32, nk_f16_to_f32_serial, nk_f32_to_f16_serial)    // nk_each_fma_f16_serial
nk_define_each_fma_(bf16, f32, nk_bf16_to_f32_serial, nk_f32_to_bf16_serial) // nk_each_fma_bf16_serial
nk_define_each_fma_(e4m3, f32, nk_e4m3_to_f32_serial, nk_f32_to_e4m3_serial) // nk_each_fma_e4m3_serial
nk_define_each_fma_(e5m2, f32, nk_e5m2_to_f32_serial, nk_f32_to_e5m2_serial) // nk_each_fma_e5m2_serial
nk_define_each_fma_(e2m3, f32, nk_e2m3_to_f32_serial, nk_f32_to_e2m3_serial) // nk_each_fma_e2m3_serial
nk_define_each_fma_(e3m2, f32, nk_e3m2_to_f32_serial, nk_f32_to_e3m2_serial) // nk_each_fma_e3m2_serial
nk_define_each_fma_(i8, f32, nk_assign_from_to_, nk_f32_to_i8_serial)        // nk_each_fma_i8_serial
nk_define_each_fma_(u8, f32, nk_assign_from_to_, nk_f32_to_u8_serial)        // nk_each_fma_u8_serial
nk_define_each_fma_(i16, f32, nk_assign_from_to_, nk_f32_to_i16_serial)      // nk_each_fma_i16_serial
nk_define_each_fma_(u16, f32, nk_assign_from_to_, nk_f32_to_u16_serial)      // nk_each_fma_u16_serial
nk_define_each_fma_(i32, f64, nk_assign_from_to_, nk_f64_to_i32_serial)      // nk_each_fma_i32_serial
nk_define_each_fma_(u32, f64, nk_assign_from_to_, nk_f64_to_u32_serial)      // nk_each_fma_u32_serial
nk_define_each_fma_(i64, f64, nk_assign_from_to_, nk_f64_to_i64_serial)      // nk_each_fma_i64_serial
nk_define_each_fma_(u64, f64, nk_assign_from_to_, nk_f64_to_u64_serial)      // nk_each_fma_u64_serial

#undef nk_define_each_scale_
#undef nk_define_each_sum_
#undef nk_define_each_blend_
#undef nk_define_each_fma_

NK_DYNAMIC void nk_each_sum_f32(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);

NK_PUBLIC void nk_each_sum_f32c_serial(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f32c_t *result) {
    nk_each_sum_f32((nk_f32_t const *)a, (nk_f32_t const *)b, 2 * n, (nk_f32_t *)result);
}

NK_PUBLIC void nk_each_sum_f64c_serial(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t *result) {
    nk_each_sum_f64((nk_f64_t const *)a, (nk_f64_t const *)b, 2 * n, (nk_f64_t *)result);
}

NK_DYNAMIC void nk_each_sum_f64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);

NK_PUBLIC void nk_each_scale_f32c_serial(nk_f32c_t const *a, nk_size_t n, nk_f32c_t const *alpha, nk_f32c_t const *beta,
                                         nk_f32c_t *result) {
    nk_f32_t alpha_real = alpha->real, alpha_imag = alpha->imag;
    nk_f32_t beta_real = beta->real, beta_imag = beta->imag;
    for (nk_size_t i = 0; i != n; ++i) {
        nk_f32_t a_real = a[i].real, a_imag = a[i].imag;
        result[i].real = alpha_real * a_real - alpha_imag * a_imag + beta_real;
        result[i].imag = alpha_real * a_imag + alpha_imag * a_real + beta_imag;
    }
}

NK_PUBLIC void nk_each_scale_f64c_serial(nk_f64c_t const *a, nk_size_t n, nk_f64c_t const *alpha, nk_f64c_t const *beta,
                                         nk_f64c_t *result) {
    nk_f64_t alpha_real = alpha->real, alpha_imag = alpha->imag;
    nk_f64_t beta_real = beta->real, beta_imag = beta->imag;
    for (nk_size_t i = 0; i != n; ++i) {
        nk_f64_t a_real = a[i].real, a_imag = a[i].imag;
        result[i].real = alpha_real * a_real - alpha_imag * a_imag + beta_real;
        result[i].imag = alpha_real * a_imag + alpha_imag * a_real + beta_imag;
    }
}

NK_PUBLIC void nk_each_blend_f32c_serial(nk_f32c_t const *a, nk_f32c_t const *b, nk_size_t n, nk_f32c_t const *alpha,
                                         nk_f32c_t const *beta, nk_f32c_t *result) {
    nk_f32_t alpha_real = alpha->real, alpha_imag = alpha->imag;
    nk_f32_t beta_real = beta->real, beta_imag = beta->imag;
    for (nk_size_t i = 0; i != n; ++i) {
        nk_f32_t a_real = a[i].real, a_imag = a[i].imag;
        nk_f32_t b_real = b[i].real, b_imag = b[i].imag;
        nk_f32_t alpha_a_real = alpha_real * a_real - alpha_imag * a_imag;
        nk_f32_t alpha_a_imag = alpha_real * a_imag + alpha_imag * a_real;
        nk_f32_t beta_b_real = beta_real * b_real - beta_imag * b_imag;
        nk_f32_t beta_b_imag = beta_real * b_imag + beta_imag * b_real;
        result[i].real = alpha_a_real + beta_b_real;
        result[i].imag = alpha_a_imag + beta_b_imag;
    }
}

NK_PUBLIC void nk_each_blend_f64c_serial(nk_f64c_t const *a, nk_f64c_t const *b, nk_size_t n, nk_f64c_t const *alpha,
                                         nk_f64c_t const *beta, nk_f64c_t *result) {
    nk_f64_t alpha_real = alpha->real, alpha_imag = alpha->imag;
    nk_f64_t beta_real = beta->real, beta_imag = beta->imag;
    for (nk_size_t i = 0; i != n; ++i) {
        nk_f64_t a_real = a[i].real, a_imag = a[i].imag;
        nk_f64_t b_real = b[i].real, b_imag = b[i].imag;
        nk_f64_t alpha_a_real = alpha_real * a_real - alpha_imag * a_imag;
        nk_f64_t alpha_a_imag = alpha_real * a_imag + alpha_imag * a_real;
        nk_f64_t beta_b_real = beta_real * b_real - beta_imag * b_imag;
        nk_f64_t beta_b_imag = beta_real * b_imag + beta_imag * b_real;
        result[i].real = alpha_a_real + beta_b_real;
        result[i].imag = alpha_a_imag + beta_b_imag;
    }
}

NK_PUBLIC void nk_each_fma_f32c_serial(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c, nk_size_t n,
                                       nk_f32c_t const *alpha, nk_f32c_t const *beta, nk_f32c_t *result) {
    nk_f32_t alpha_real = alpha->real, alpha_imag = alpha->imag;
    nk_f32_t beta_real = beta->real, beta_imag = beta->imag;
    for (nk_size_t i = 0; i != n; ++i) {
        nk_f32_t a_real = a[i].real, a_imag = a[i].imag;
        nk_f32_t b_real = b[i].real, b_imag = b[i].imag;
        nk_f32_t c_real = c[i].real, c_imag = c[i].imag;
        nk_f32_t product_real = a_real * b_real - a_imag * b_imag;
        nk_f32_t product_imag = a_real * b_imag + a_imag * b_real;
        nk_f32_t alpha_product_real = alpha_real * product_real - alpha_imag * product_imag;
        nk_f32_t alpha_product_imag = alpha_real * product_imag + alpha_imag * product_real;
        nk_f32_t beta_c_real = beta_real * c_real - beta_imag * c_imag;
        nk_f32_t beta_c_imag = beta_real * c_imag + beta_imag * c_real;
        result[i].real = alpha_product_real + beta_c_real;
        result[i].imag = alpha_product_imag + beta_c_imag;
    }
}

NK_PUBLIC void nk_each_fma_f64c_serial(nk_f64c_t const *a, nk_f64c_t const *b, nk_f64c_t const *c, nk_size_t n,
                                       nk_f64c_t const *alpha, nk_f64c_t const *beta, nk_f64c_t *result) {
    nk_f64_t alpha_real = alpha->real, alpha_imag = alpha->imag;
    nk_f64_t beta_real = beta->real, beta_imag = beta->imag;
    for (nk_size_t i = 0; i != n; ++i) {
        nk_f64_t a_real = a[i].real, a_imag = a[i].imag;
        nk_f64_t b_real = b[i].real, b_imag = b[i].imag;
        nk_f64_t c_real = c[i].real, c_imag = c[i].imag;
        nk_f64_t product_real = a_real * b_real - a_imag * b_imag;
        nk_f64_t product_imag = a_real * b_imag + a_imag * b_real;
        nk_f64_t alpha_product_real = alpha_real * product_real - alpha_imag * product_imag;
        nk_f64_t alpha_product_imag = alpha_real * product_imag + alpha_imag * product_real;
        nk_f64_t beta_c_real = beta_real * c_real - beta_imag * c_imag;
        nk_f64_t beta_c_imag = beta_real * c_imag + beta_imag * c_real;
        result[i].real = alpha_product_real + beta_c_real;
        result[i].imag = alpha_product_imag + beta_c_imag;
    }
}

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_EACH_SERIAL_H
