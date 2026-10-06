/**
 *  @brief Serial Sparse Vector Operations.
 *  @file include/numkong/sparse/serial.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *
 *  @sa include/numkong/sparse.h
 */
#ifndef NK_SPARSE_SERIAL_H
#define NK_SPARSE_SERIAL_H

#include "numkong/types.h"
#include "numkong/cast/serial.h" // `nk_bf16_to_f32_serial`, `nk_assign_from_to_`

#if defined(__cplusplus)
extern "C" {
#endif

#define nk_define_sparse_intersect_(input_type)                                                                      \
    NK_PUBLIC nk_size_t nk_sparse_intersect_##input_type##_galloping_search_(                                        \
        nk_##input_type##_t const *array, nk_size_t start, nk_size_t length, nk_##input_type##_t val) {              \
        nk_size_t low = start;                                                                                       \
        nk_size_t high = start + 1;                                                                                  \
        while (high < length && array[high] < val) {                                                                 \
            low = high;                                                                                              \
            high = (2 * high < length) ? 2 * high : length;                                                          \
        }                                                                                                            \
        while (low < high) {                                                                                         \
            nk_size_t mid = low + (high - low) / 2;                                                                  \
            if (array[mid] < val) { low = mid + 1; }                                                                 \
            else { high = mid; }                                                                                     \
        }                                                                                                            \
        return low;                                                                                                  \
    }                                                                                                                \
    NK_PUBLIC nk_size_t nk_sparse_intersect_##input_type##_linear_scan_(                                             \
        nk_##input_type##_t const *a, nk_##input_type##_t const *b, nk_size_t a_length, nk_size_t b_length,          \
        nk_##input_type##_t *result) {                                                                               \
        nk_size_t intersection_size = 0;                                                                             \
        nk_size_t i = 0, j = 0;                                                                                      \
        while (i != a_length && j != b_length) {                                                                     \
            nk_##input_type##_t ai = a[i];                                                                           \
            nk_##input_type##_t bj = b[j];                                                                           \
            if (ai == bj) {                                                                                          \
                if (result) result[intersection_size] = ai;                                                          \
                intersection_size++;                                                                                 \
            }                                                                                                        \
            i += ai <= bj;                                                                                           \
            j += ai >= bj;                                                                                           \
        }                                                                                                            \
        return intersection_size;                                                                                    \
    }                                                                                                                \
    NK_PUBLIC void nk_sparse_intersect_##input_type##_serial(                                                        \
        nk_##input_type##_t const *shorter, nk_##input_type##_t const *longer, nk_size_t shorter_length,             \
        nk_size_t longer_length, nk_##input_type##_t *result, nk_size_t *count) {                                    \
        /* Swap arrays if necessary, as we want "longer" to be larger than "shorter" */                              \
        if (longer_length < shorter_length) {                                                                        \
            nk_##input_type##_t const *temp = shorter;                                                               \
            shorter = longer;                                                                                        \
            longer = temp;                                                                                           \
            nk_size_t temp_length = shorter_length;                                                                  \
            shorter_length = longer_length;                                                                          \
            longer_length = temp_length;                                                                             \
        }                                                                                                            \
                                                                                                                     \
        /* Use the accurate implementation if galloping is not beneficial */                                         \
        if (longer_length < 64 * shorter_length) {                                                                   \
            *count = nk_sparse_intersect_##input_type##_linear_scan_(shorter, longer, shorter_length, longer_length, \
                                                                     result);                                        \
            return;                                                                                                  \
        }                                                                                                            \
                                                                                                                     \
        /* Perform galloping, shrinking the target range */                                                          \
        nk_size_t intersection_size = 0;                                                                             \
        nk_size_t j = 0;                                                                                             \
        for (nk_size_t i = 0; i < shorter_length; ++i) {                                                             \
            nk_##input_type##_t shorter_i = shorter[i];                                                              \
            j = nk_sparse_intersect_##input_type##_galloping_search_(longer, j, longer_length, shorter_i);           \
            if (j < longer_length && longer[j] == shorter_i) {                                                       \
                if (result) result[intersection_size] = shorter_i;                                                   \
                intersection_size++;                                                                                 \
            }                                                                                                        \
        }                                                                                                            \
        *count = intersection_size;                                                                                  \
    }

#define nk_define_sparse_dot_(input_type, weight_type, accumulator_type, load_and_convert)                 \
    NK_PUBLIC void nk_sparse_dot_##input_type##weight_type##_serial(                                       \
        nk_##input_type##_t const *a, nk_##input_type##_t const *b, nk_##weight_type##_t const *a_weights, \
        nk_##weight_type##_t const *b_weights, nk_size_t a_length, nk_size_t b_length,                     \
        nk_##accumulator_type##_t *product) {                                                              \
        nk_##accumulator_type##_t weights_product = 0, awi, bwi;                                           \
        nk_size_t i = 0, j = 0;                                                                            \
        while (i != a_length && j != b_length) {                                                           \
            nk_##input_type##_t ai = a[i];                                                                 \
            nk_##input_type##_t bj = b[j];                                                                 \
            int matches = ai == bj;                                                                        \
            load_and_convert(a_weights + i, &awi);                                                         \
            load_and_convert(b_weights + j, &bwi);                                                         \
            weights_product += matches * awi * bwi;                                                        \
            i += ai < bj;                                                                                  \
            j += ai >= bj;                                                                                 \
        }                                                                                                  \
        *product = weights_product;                                                                        \
    }

nk_define_sparse_intersect_(u16) // nk_sparse_intersect_u16_serial
nk_define_sparse_intersect_(u32) // nk_sparse_intersect_u32_serial
nk_define_sparse_intersect_(u64) // nk_sparse_intersect_u64_serial

nk_define_sparse_dot_(u16, bf16, f32, nk_bf16_to_f32_serial) // nk_sparse_dot_u16bf16_serial
nk_define_sparse_dot_(u32, f32, f64, nk_assign_from_to_)     // nk_sparse_dot_u32f32_serial

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_SPARSE_SERIAL_H
