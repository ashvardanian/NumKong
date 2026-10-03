/**
 *  @brief SWAR-accelerated Curved Space Similarity for SIMD-free CPUs.
 *  @file include/numkong/curved/serial.h
 *  @author Ash Vardanian
 *  @date January 14, 2026
 *
 *  @sa include/numkong/curved.h
 *
 *  Implements bilinear forms and Mahalanobis distance with precision-appropriate strategies:
 *  - f64 inputs use Dot2 (Ogita-Rump-Oishi 2005) for error-free transformations
 *  - f32/f16/bf16 inputs upcast to wider accumulators (f64/f32), providing sufficient
 *    precision headroom without compensation overhead
 *
 *  Bilinear form: aᵀ × C × b = Σᵢ aᵢ × (Σⱼ cᵢⱼ × bⱼ)
 *
 *  The nested loop structure has two accumulation levels:
 *  - Inner: Σⱼ cᵢⱼ × bⱼ (O(n) terms per row)
 *  - Outer: Σᵢ aᵢ × inner_result (O(n) terms total)
 *
 *  For f64→f64 (no upcast headroom): Dot2 uses TwoProd and TwoSum error-free
 *  transformations at both levels, capturing rounding errors in compensation terms.
 *
 *  For upcasted types (f32→f64, f16→f32, bf16→f32): the wider accumulator provides
 *  enough extra mantissa bits that simple accumulation suffices.
 *
 *  @see Ogita, T., Rump, S.M., Oishi, S. (2005). "Accurate Sum and Dot Product"
 */
#ifndef NK_CURVED_SERIAL_H
#define NK_CURVED_SERIAL_H

#include "numkong/types.h"
#include "numkong/spatial/serial.h" // `nk_f64_sqrt_serial`

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Macro for bilinear form aᵀ × C × b with simple accumulation.
 *
 *  Suitable for upcasted types where the wider accumulator provides sufficient
 *  precision headroom (f32→f64, f16→f32, bf16→f32).
 */
#define nk_define_bilinear_(input_type, accumulator_type, output_type, load_and_convert)                         \
    NK_PUBLIC void nk_bilinear_##input_type##_serial(nk_##input_type##_t const *a, nk_##input_type##_t const *b, \
                                                     nk_##input_type##_t const *c, nk_size_t n,                  \
                                                     nk_##output_type##_t *result) {                             \
        nk_##accumulator_type##_t outer_sum = 0;                                                                 \
        nk_##accumulator_type##_t vector_a_value, vector_b_value, tensor_value;                                  \
        for (nk_size_t row = 0; row != n; ++row) {                                                               \
            nk_##accumulator_type##_t inner_sum = 0;                                                             \
            load_and_convert(a + row, &vector_a_value);                                                          \
            for (nk_size_t column = 0; column != n; ++column) {                                                  \
                load_and_convert(b + column, &vector_b_value);                                                   \
                load_and_convert(c + row * n + column, &tensor_value);                                           \
                inner_sum += tensor_value * vector_b_value;                                                      \
            }                                                                                                    \
            outer_sum += vector_a_value * inner_sum;                                                             \
        }                                                                                                        \
        *result = (nk_##output_type##_t)(outer_sum);                                                             \
    }

/**
 *  @brief Macro for complex bilinear form aᵀ × C × b with simple accumulation.
 *
 *  Suitable for upcasted complex types where the wider accumulator provides
 *  sufficient precision headroom.
 */
#define nk_define_bilinear_complex_(input_type, accumulator_type, output_type, load_and_convert)                    \
    NK_PUBLIC void nk_bilinear_##input_type##_serial(                                                               \
        nk_##input_type##_t const *a_pairs, nk_##input_type##_t const *b_pairs, nk_##input_type##_t const *c_pairs, \
        nk_size_t n, nk_##output_type##c_t *results) {                                                              \
        nk_##accumulator_type##_t outer_sum_real = 0, outer_sum_imag = 0;                                           \
        nk_##accumulator_type##_t a_real, a_imag, b_real, b_imag, c_real, c_imag;                                   \
        for (nk_size_t row = 0; row != n; ++row) {                                                                  \
            nk_##accumulator_type##_t inner_sum_real = 0, inner_sum_imag = 0;                                       \
            load_and_convert(&(a_pairs + row)->real, &a_real);                                                      \
            load_and_convert(&(a_pairs + row)->imag, &a_imag);                                                      \
            for (nk_size_t column = 0; column != n; ++column) {                                                     \
                load_and_convert(&(b_pairs + column)->real, &b_real);                                               \
                load_and_convert(&(b_pairs + column)->imag, &b_imag);                                               \
                load_and_convert(&(c_pairs + row * n + column)->real, &c_real);                                     \
                load_and_convert(&(c_pairs + row * n + column)->imag, &c_imag);                                     \
                inner_sum_real += c_real * b_real - c_imag * b_imag;                                                \
                inner_sum_imag += c_real * b_imag + c_imag * b_real;                                                \
            }                                                                                                       \
            /* Complex multiply: a_i * inner_result */                                                              \
            outer_sum_real += a_real * inner_sum_real - a_imag * inner_sum_imag;                                    \
            outer_sum_imag += a_real * inner_sum_imag + a_imag * inner_sum_real;                                    \
        }                                                                                                           \
        results->real = outer_sum_real;                                                                             \
        results->imag = outer_sum_imag;                                                                             \
    }

/**
 *  @brief Macro for Mahalanobis distance √((a−b)ᵀ × C × (a−b)) with simple accumulation.
 *
 *  Suitable for upcasted types where the wider accumulator provides sufficient
 *  precision headroom. Differences are computed in the accumulator precision.
 */
#define nk_define_mahalanobis_(input_type, accumulator_type, output_type, load_and_convert)                         \
    NK_PUBLIC void nk_mahalanobis_##input_type##_serial(nk_##input_type##_t const *a, nk_##input_type##_t const *b, \
                                                        nk_##input_type##_t const *c, nk_size_t n,                  \
                                                        nk_##output_type##_t *result) {                             \
        nk_##accumulator_type##_t outer_sum = 0;                                                                    \
        nk_##accumulator_type##_t a_row_value, b_row_value, a_column_value, b_column_value, tensor_value;           \
        for (nk_size_t row = 0; row != n; ++row) {                                                                  \
            nk_##accumulator_type##_t inner_sum = 0;                                                                \
            load_and_convert(a + row, &a_row_value);                                                                \
            load_and_convert(b + row, &b_row_value);                                                                \
            nk_##accumulator_type##_t difference_row = a_row_value - b_row_value;                                   \
            for (nk_size_t column = 0; column != n; ++column) {                                                     \
                load_and_convert(a + column, &a_column_value);                                                      \
                load_and_convert(b + column, &b_column_value);                                                      \
                load_and_convert(c + row * n + column, &tensor_value);                                              \
                nk_##accumulator_type##_t difference_column = a_column_value - b_column_value;                      \
                inner_sum += tensor_value * difference_column;                                                      \
            }                                                                                                       \
            outer_sum += difference_row * inner_sum;                                                                \
        }                                                                                                           \
        nk_##accumulator_type##_t quadratic = outer_sum;                                                            \
        *result = nk_##accumulator_type##_sqrt_serial(quadratic > 0 ? quadratic : 0);                               \
    }

// f32 → f64 accumulator → f64 output
nk_define_bilinear_(f32, f64, f64, nk_assign_from_to_)          // nk_bilinear_f32_serial
nk_define_bilinear_complex_(f32c, f64, f64, nk_assign_from_to_) // nk_bilinear_f32c_serial
nk_define_mahalanobis_(f32, f64, f64, nk_assign_from_to_)       // nk_mahalanobis_f32_serial

// f16 → f32 accumulator → f32 output: f32 provides ample headroom for f16 (~3 vs ~7 decimal digits)
nk_define_bilinear_(f16, f32, f32, nk_f16_to_f32_serial)          // nk_bilinear_f16_serial
nk_define_bilinear_complex_(f16c, f32, f32, nk_f16_to_f32_serial) // nk_bilinear_f16c_serial
nk_define_mahalanobis_(f16, f32, f32, nk_f16_to_f32_serial)       // nk_mahalanobis_f16_serial

// bf16 → f32 accumulator → f32 output: f32 provides ample headroom for bf16 (~2 vs ~7 decimal digits)
nk_define_bilinear_(bf16, f32, f32, nk_bf16_to_f32_serial)          // nk_bilinear_bf16_serial
nk_define_bilinear_complex_(bf16c, f32, f32, nk_bf16_to_f32_serial) // nk_bilinear_bf16c_serial
nk_define_mahalanobis_(bf16, f32, f32, nk_bf16_to_f32_serial)       // nk_mahalanobis_bf16_serial

#undef nk_define_bilinear_
#undef nk_define_bilinear_complex_
#undef nk_define_mahalanobis_

NK_PUBLIC void nk_bilinear_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                      nk_f64_t *result) {
    nk_f64_t outer_sum = 0, outer_comp = 0;
    for (nk_size_t row = 0; row != n; ++row) {
        nk_f64_t inner_sum = 0, inner_comp = 0;
        for (nk_size_t col = 0; col != n; ++col) nk_f64_dot2_(&inner_sum, &inner_comp, c[row * n + col], b[col]);
        nk_f64_t cb_j = inner_sum + inner_comp;
        nk_f64_dot2_(&outer_sum, &outer_comp, a[row], cb_j);
    }
    *result = outer_sum + outer_comp;
}

NK_PUBLIC void nk_bilinear_f64c_serial(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs, nk_f64c_t const *c_pairs,
                                       nk_size_t n, nk_f64c_t *results) {
    nk_f64_t outer_sum_real = 0, outer_comp_real = 0;
    nk_f64_t outer_sum_imag = 0, outer_comp_imag = 0;
    for (nk_size_t row = 0; row != n; ++row) {
        nk_f64_t a_real = a_pairs[row].real;
        nk_f64_t a_imag = a_pairs[row].imag;
        // 4 Dot2 accumulators for inner cross-terms
        nk_f64_t sum_rr = 0, comp_rr = 0;
        nk_f64_t sum_ii = 0, comp_ii = 0;
        nk_f64_t sum_ri = 0, comp_ri = 0;
        nk_f64_t sum_ir = 0, comp_ir = 0;
        for (nk_size_t col = 0; col != n; ++col) {
            nk_f64_t b_real = b_pairs[col].real, b_imag = b_pairs[col].imag;
            nk_f64_t c_real = c_pairs[row * n + col].real, c_imag = c_pairs[row * n + col].imag;
            nk_f64_dot2_(&sum_rr, &comp_rr, c_real, b_real);
            nk_f64_dot2_(&sum_ii, &comp_ii, c_imag, b_imag);
            nk_f64_dot2_(&sum_ri, &comp_ri, c_real, b_imag);
            nk_f64_dot2_(&sum_ir, &comp_ir, c_imag, b_real);
        }
        nk_f64_t inner_real = (sum_rr + comp_rr) - (sum_ii + comp_ii);
        nk_f64_t inner_imag = (sum_ri + comp_ri) + (sum_ir + comp_ir);
        // Outer Dot2 complex multiply: a × inner
        nk_f64_dot2_(&outer_sum_real, &outer_comp_real, a_real, inner_real);
        nk_f64_dot2_(&outer_sum_real, &outer_comp_real, -a_imag, inner_imag);
        nk_f64_dot2_(&outer_sum_imag, &outer_comp_imag, a_real, inner_imag);
        nk_f64_dot2_(&outer_sum_imag, &outer_comp_imag, a_imag, inner_real);
    }
    results->real = outer_sum_real + outer_comp_real;
    results->imag = outer_sum_imag + outer_comp_imag;
}

NK_PUBLIC void nk_mahalanobis_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                         nk_f64_t *result) {
    nk_f64_t outer_sum = 0, outer_comp = 0;
    for (nk_size_t row = 0; row != n; ++row) {
        nk_f64_t diff_row = a[row] - b[row];
        nk_f64_t inner_sum = 0, inner_comp = 0;
        for (nk_size_t col = 0; col != n; ++col)
            nk_f64_dot2_(&inner_sum, &inner_comp, c[row * n + col], a[col] - b[col]);
        nk_f64_t cb_j = inner_sum + inner_comp;
        nk_f64_dot2_(&outer_sum, &outer_comp, diff_row, cb_j);
    }
    nk_f64_t quadratic = outer_sum + outer_comp;
    *result = nk_f64_sqrt_serial(quadratic > 0 ? quadratic : 0);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_CURVED_SERIAL_H
