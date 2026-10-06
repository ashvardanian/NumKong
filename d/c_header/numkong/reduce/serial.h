/**
 *  @brief Serial fallbacks for the redesigned reduction API (moments + minmax).
 *  @file include/numkong/reduce/serial.h
 *  @author Ash Vardanian
 *  @date February 11, 2026
 *
 *  @sa include/numkong/reduce.h
 *
 *  Provides serial (non-SIMD) implementations of:
 *  - `nk_reduce_moments_*_serial` — sum + sum-of-squares in one pass
 *  - `nk_reduce_minmax_*_serial` — min + max with indices in one pass
 */
#ifndef NK_REDUCE_SERIAL_H
#define NK_REDUCE_SERIAL_H

#include "numkong/types.h"
#include "numkong/scalar/serial.h"
#include "numkong/cast/serial.h"
#include "numkong/scalar/serial.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_INTERNAL nk_f64_t nk_reduce_sum_f64_serial_(nk_f64_t const *values, nk_f64_t const *compensations, int count) {
    nk_f64_t running_sum = 0, accumulated_error = 0;
    for (int i = 0; i < count; i++) {
        // TwoSum: fold in values[i]
        nk_f64_t tentative_sum = running_sum + values[i];
        nk_f64_t virtual_addend = tentative_sum - running_sum;
        accumulated_error += (running_sum - (tentative_sum - virtual_addend)) + (values[i] - virtual_addend);
        running_sum = tentative_sum;
        // TwoSum: fold in compensations[i]
        tentative_sum = running_sum + compensations[i];
        virtual_addend = tentative_sum - running_sum;
        accumulated_error += (running_sum - (tentative_sum - virtual_addend)) + (compensations[i] - virtual_addend);
        running_sum = tentative_sum;
    }
    return running_sum + accumulated_error;
}

NK_PUBLIC void nk_reduce_moments_f32_serial(                       //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *sum_ptr, nk_f64_t *sumsq_ptr) {
    nk_f64_t running_sum = 0, sum_compensation = 0;
    nk_f64_t running_sumsq = 0, sumsq_compensation = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f64_t val = (nk_f64_t)(*(nk_f32_t const *)ptr);
        nk_f64_t tentative_sum = running_sum + val;
        if (nk_f64_abs_(running_sum) >= nk_f64_abs_(val)) sum_compensation += (running_sum - tentative_sum) + val;
        else sum_compensation += (val - tentative_sum) + running_sum;
        running_sum = tentative_sum;

        nk_f64_t squared_value = val * val;
        nk_f64_t tentative_sumsq = running_sumsq + squared_value;
        if (running_sumsq >= squared_value) sumsq_compensation += (running_sumsq - tentative_sumsq) + squared_value;
        else sumsq_compensation += (squared_value - tentative_sumsq) + running_sumsq;
        running_sumsq = tentative_sumsq;
    }
    *sum_ptr = running_sum + sum_compensation, *sumsq_ptr = running_sumsq + sumsq_compensation;
}

NK_PUBLIC void nk_reduce_moments_f64_serial(                       //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *sum_ptr, nk_f64_t *sumsq_ptr) {
    nk_f64_t running_sum = 0, sum_compensation = 0;
    nk_f64_t running_sumsq = 0, sumsq_compensation = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f64_t val = *(nk_f64_t const *)ptr;
        nk_f64_t tentative_sum = running_sum + val;
        if (nk_f64_abs_(running_sum) >= nk_f64_abs_(val)) sum_compensation += (running_sum - tentative_sum) + val;
        else sum_compensation += (val - tentative_sum) + running_sum;
        running_sum = tentative_sum;

        nk_f64_t squared_value = val * val;
        nk_f64_t tentative_sumsq = running_sumsq + squared_value;
        if (running_sumsq >= squared_value) sumsq_compensation += (running_sumsq - tentative_sumsq) + squared_value;
        else sumsq_compensation += (squared_value - tentative_sumsq) + running_sumsq;
        running_sumsq = tentative_sumsq;
    }
    *sum_ptr = running_sum + sum_compensation, *sumsq_ptr = running_sumsq + sumsq_compensation;
}

NK_PUBLIC void nk_reduce_moments_i8_serial(                       //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_i64_t sum = 0;
    nk_u64_t sumsq = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_i64_t val = (nk_i64_t)(*(nk_i8_t const *)ptr);
        sum += val;
        sumsq += (nk_u64_t)(val * val);
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_u8_serial(                       //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_u64_t sum = 0, sumsq = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_u64_t val = (nk_u64_t)(*(nk_u8_t const *)ptr);
        sum += val;
        sumsq += val * val;
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_i16_serial(                       //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_i64_t sum = 0;
    nk_u64_t sumsq = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_i64_t val = (nk_i64_t)(*(nk_i16_t const *)ptr);
        sum += val;
        sumsq += (nk_u64_t)(val * val);
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_u16_serial(                       //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_u64_t sum = 0, sumsq = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_u64_t val = (nk_u64_t)(*(nk_u16_t const *)ptr);
        sum += val;
        sumsq += val * val;
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_i32_serial(                       //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_u64_t sum_lower = 0;
    nk_i64_t sum_upper = 0;
    nk_u64_t sumsq = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_i64_t val = (nk_i64_t)(*(nk_i32_t const *)ptr);
        nk_u64_t product = (nk_u64_t)(val * val);
        nk_u64_t sum_before = sum_lower;
        sum_lower += (nk_u64_t)val;
        if (sum_lower < sum_before) sum_upper++;
        sum_upper += (val >> 63);
        sumsq = nk_u64_saturating_add_serial(sumsq, product);
    }
    nk_i64_t sum_lower_signed = (nk_i64_t)sum_lower;
    if (sum_upper == (sum_lower_signed >> 63)) *sum_ptr = sum_lower_signed;
    else if (sum_upper >= 0) *sum_ptr = NK_I64_MAX;
    else *sum_ptr = NK_I64_MIN;
    *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_u32_serial(                       //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_u64_t sum = 0, sumsq = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_u64_t val = (nk_u64_t)(*(nk_u32_t const *)ptr);
        nk_u64_t product = val * val;
        sum = nk_u64_saturating_add_serial(sum, val);
        sumsq = nk_u64_saturating_add_serial(sumsq, product);
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_i64_serial(                       //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_u64_t sum_lower = 0;
    nk_i64_t sum_upper = 0;
    nk_u64_t sumsq = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_i64_t val = *(nk_i64_t const *)ptr;
        nk_i64_t product = nk_i64_saturating_mul_serial(val, val);
        nk_u64_t unsigned_product = (nk_u64_t)product;
        sumsq = nk_u64_saturating_add_serial(sumsq, unsigned_product);
        nk_u64_t sum_before = sum_lower;
        sum_lower += (nk_u64_t)val;
        if (sum_lower < sum_before) sum_upper++;
        sum_upper += (val >> 63);
    }
    nk_i64_t sum_lower_signed = (nk_i64_t)sum_lower;
    if (sum_upper == (sum_lower_signed >> 63)) *sum_ptr = sum_lower_signed;
    else if (sum_upper >= 0) *sum_ptr = NK_I64_MAX;
    else *sum_ptr = NK_I64_MIN;
    *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_u64_serial(                       //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_u64_t sum = 0, sumsq = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_u64_t val = *(nk_u64_t const *)ptr;
        nk_u64_t product = nk_u64_saturating_mul_serial(val, val);
        sum = nk_u64_saturating_add_serial(sum, val);
        sumsq = nk_u64_saturating_add_serial(sumsq, product);
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_f16_serial(                       //
    nk_f16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_f32_t running_sum = 0, sum_compensation = 0;
    nk_f32_t running_sumsq = 0, sumsq_compensation = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_f16_to_f32_serial((nk_f16_t const *)ptr, &val);
        nk_f32_t tentative_sum = running_sum + val;
        nk_f32_t abs_running_sum = nk_f32_abs_(running_sum);
        nk_f32_t abs_val = nk_f32_abs_(val);
        if (abs_running_sum >= abs_val) sum_compensation += (running_sum - tentative_sum) + val;
        else sum_compensation += (val - tentative_sum) + running_sum;
        running_sum = tentative_sum;

        nk_f32_t squared_value = val * val;
        nk_f32_t tentative_sumsq = running_sumsq + squared_value;
        if (running_sumsq >= squared_value) sumsq_compensation += (running_sumsq - tentative_sumsq) + squared_value;
        else sumsq_compensation += (squared_value - tentative_sumsq) + running_sumsq;
        running_sumsq = tentative_sumsq;
    }
    *sum_ptr = running_sum + sum_compensation, *sumsq_ptr = running_sumsq + sumsq_compensation;
}

NK_PUBLIC void nk_reduce_moments_bf16_serial(                       //
    nk_bf16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_f32_t running_sum = 0, sum_compensation = 0;
    nk_f32_t running_sumsq = 0, sumsq_compensation = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_bf16_to_f32_serial((nk_bf16_t const *)ptr, &val);
        nk_f32_t tentative_sum = running_sum + val;
        nk_f32_t abs_running_sum = nk_f32_abs_(running_sum);
        nk_f32_t abs_val = nk_f32_abs_(val);
        if (abs_running_sum >= abs_val) sum_compensation += (running_sum - tentative_sum) + val;
        else sum_compensation += (val - tentative_sum) + running_sum;
        running_sum = tentative_sum;

        nk_f32_t squared_value = val * val;
        nk_f32_t tentative_sumsq = running_sumsq + squared_value;
        if (running_sumsq >= squared_value) sumsq_compensation += (running_sumsq - tentative_sumsq) + squared_value;
        else sumsq_compensation += (squared_value - tentative_sumsq) + running_sumsq;
        running_sumsq = tentative_sumsq;
    }
    *sum_ptr = running_sum + sum_compensation, *sumsq_ptr = running_sumsq + sumsq_compensation;
}

NK_PUBLIC void nk_reduce_moments_e4m3_serial(                       //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_f32_t running_sum = 0, sum_compensation = 0;
    nk_f32_t running_sumsq = 0, sumsq_compensation = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_e4m3_to_f32_serial((nk_e4m3_t const *)ptr, &val);
        nk_f32_t tentative_sum = running_sum + val;
        nk_f32_t abs_running_sum = nk_f32_abs_(running_sum);
        nk_f32_t abs_val = nk_f32_abs_(val);
        if (abs_running_sum >= abs_val) sum_compensation += (running_sum - tentative_sum) + val;
        else sum_compensation += (val - tentative_sum) + running_sum;
        running_sum = tentative_sum;

        nk_f32_t squared_value = val * val;
        nk_f32_t tentative_sumsq = running_sumsq + squared_value;
        if (running_sumsq >= squared_value) sumsq_compensation += (running_sumsq - tentative_sumsq) + squared_value;
        else sumsq_compensation += (squared_value - tentative_sumsq) + running_sumsq;
        running_sumsq = tentative_sumsq;
    }
    *sum_ptr = running_sum + sum_compensation, *sumsq_ptr = running_sumsq + sumsq_compensation;
}

NK_PUBLIC void nk_reduce_moments_e5m2_serial(                       //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_f32_t running_sum = 0, sum_compensation = 0;
    nk_f32_t running_sumsq = 0, sumsq_compensation = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_e5m2_to_f32_serial((nk_e5m2_t const *)ptr, &val);
        nk_f32_t tentative_sum = running_sum + val;
        nk_f32_t abs_running_sum = nk_f32_abs_(running_sum);
        nk_f32_t abs_val = nk_f32_abs_(val);
        if (abs_running_sum >= abs_val) sum_compensation += (running_sum - tentative_sum) + val;
        else sum_compensation += (val - tentative_sum) + running_sum;
        running_sum = tentative_sum;

        nk_f32_t squared_value = val * val;
        nk_f32_t tentative_sumsq = running_sumsq + squared_value;
        if (running_sumsq >= squared_value) sumsq_compensation += (running_sumsq - tentative_sumsq) + squared_value;
        else sumsq_compensation += (squared_value - tentative_sumsq) + running_sumsq;
        running_sumsq = tentative_sumsq;
    }
    *sum_ptr = running_sum + sum_compensation, *sumsq_ptr = running_sumsq + sumsq_compensation;
}

NK_PUBLIC void nk_reduce_moments_e2m3_serial(                       //
    nk_e2m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_f32_t running_sum = 0, sum_compensation = 0;
    nk_f32_t running_sumsq = 0, sumsq_compensation = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_e2m3_to_f32_serial((nk_e2m3_t const *)ptr, &val);
        nk_f32_t tentative_sum = running_sum + val;
        nk_f32_t abs_running_sum = nk_f32_abs_(running_sum);
        nk_f32_t abs_val = nk_f32_abs_(val);
        if (abs_running_sum >= abs_val) sum_compensation += (running_sum - tentative_sum) + val;
        else sum_compensation += (val - tentative_sum) + running_sum;
        running_sum = tentative_sum;

        nk_f32_t squared_value = val * val;
        nk_f32_t tentative_sumsq = running_sumsq + squared_value;
        if (running_sumsq >= squared_value) sumsq_compensation += (running_sumsq - tentative_sumsq) + squared_value;
        else sumsq_compensation += (squared_value - tentative_sumsq) + running_sumsq;
        running_sumsq = tentative_sumsq;
    }
    *sum_ptr = running_sum + sum_compensation, *sumsq_ptr = running_sumsq + sumsq_compensation;
}

NK_PUBLIC void nk_reduce_moments_e3m2_serial(                       //
    nk_e3m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_f32_t running_sum = 0, sum_compensation = 0;
    nk_f32_t running_sumsq = 0, sumsq_compensation = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val;
        nk_e3m2_to_f32_serial((nk_e3m2_t const *)ptr, &val);
        nk_f32_t tentative_sum = running_sum + val;
        nk_f32_t abs_running_sum = nk_f32_abs_(running_sum);
        nk_f32_t abs_val = nk_f32_abs_(val);
        if (abs_running_sum >= abs_val) sum_compensation += (running_sum - tentative_sum) + val;
        else sum_compensation += (val - tentative_sum) + running_sum;
        running_sum = tentative_sum;

        nk_f32_t squared_value = val * val;
        nk_f32_t tentative_sumsq = running_sumsq + squared_value;
        if (running_sumsq >= squared_value) sumsq_compensation += (running_sumsq - tentative_sumsq) + squared_value;
        else sumsq_compensation += (squared_value - tentative_sumsq) + running_sumsq;
        running_sumsq = tentative_sumsq;
    }
    *sum_ptr = running_sum + sum_compensation, *sumsq_ptr = running_sumsq + sumsq_compensation;
}

NK_PUBLIC void nk_reduce_moments_i4_serial(                         //
    nk_i4x2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    count = nk_size_round_up_to_multiple_(count, 2);
    nk_i64_t sum = 0;
    nk_u64_t sumsq = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; i += 2) {
        unsigned char byte_val = ptr[(i / 2) * stride_bytes];
        nk_i64_t low = (nk_i64_t)nk_i4x2_low_(byte_val);
        nk_i64_t high = (nk_i64_t)nk_i4x2_high_(byte_val);
        sum += low + high;
        sumsq += (nk_u64_t)(low * low) + (nk_u64_t)(high * high);
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_u4_serial(                         //
    nk_u4x2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    count = nk_size_round_up_to_multiple_(count, 2);
    nk_u64_t sum = 0, sumsq = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; i += 2) {
        unsigned char byte_val = ptr[(i / 2) * stride_bytes];
        nk_u64_t low = nk_u4x2_low_(byte_val);
        nk_u64_t high = nk_u4x2_high_(byte_val);
        sum += low + high, sumsq += low * low + high * high;
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_u1_serial(                         //
    nk_u1x8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    count = nk_size_round_up_to_multiple_(count, 8);
    nk_u64_t sum = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; i += 8) {
        unsigned char byte_val = ptr[(i / 8) * stride_bytes];
        sum += nk_u64_popcount_(byte_val);
    }
    *sum_ptr = sum, *sumsq_ptr = sum; // 0^2 = 0, 1^2 = 1, so sumsq == sum
}

NK_PUBLIC void nk_reduce_minmax_f32_serial(                        //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value_ptr, nk_size_t *min_index_ptr,             //
    nk_f32_t *max_value_ptr, nk_size_t *max_index_ptr) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f32_t min_value = NK_F32_MAX, max_value = NK_F32_MIN;
    nk_size_t min_idx = NK_SIZE_MAX, max_idx = NK_SIZE_MAX;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val = *(nk_f32_t const *)ptr;
        if (val < min_value) min_value = val, min_idx = i;
        if (val > max_value) max_value = val, max_idx = i;
    }
    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_f64_serial(                        //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *min_value_ptr, nk_size_t *min_index_ptr,             //
    nk_f64_t *max_value_ptr, nk_size_t *max_index_ptr) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f64_t min_value = NK_F64_MAX, max_value = NK_F64_MIN;
    nk_size_t min_idx = NK_SIZE_MAX, max_idx = NK_SIZE_MAX;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f64_t val = *(nk_f64_t const *)ptr;
        if (val < min_value) min_value = val, min_idx = i;
        if (val > max_value) max_value = val, max_idx = i;
    }
    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_i8_serial(                        //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i8_t *min_value_ptr, nk_size_t *min_index_ptr,             //
    nk_i8_t *max_value_ptr, nk_size_t *max_index_ptr) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_i8_t min_value = NK_I8_MAX, max_value = NK_I8_MIN;
    nk_size_t min_idx = NK_SIZE_MAX, max_idx = NK_SIZE_MAX;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_i8_t val = *(nk_i8_t const *)ptr;
        if (val < min_value) min_value = val, min_idx = i;
        if (val > max_value) max_value = val, max_idx = i;
    }
    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_u8_serial(                        //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *min_value_ptr, nk_size_t *min_index_ptr,             //
    nk_u8_t *max_value_ptr, nk_size_t *max_index_ptr) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_u8_t min_value = NK_U8_MAX, max_value = NK_U8_MIN;
    nk_size_t min_idx = NK_SIZE_MAX, max_idx = NK_SIZE_MAX;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_u8_t val = *(nk_u8_t const *)ptr;
        if (val < min_value) min_value = val, min_idx = i;
        if (val > max_value) max_value = val, max_idx = i;
    }
    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_i16_serial(                        //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i16_t *min_value_ptr, nk_size_t *min_index_ptr,             //
    nk_i16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_i16_t min_value = NK_I16_MAX, max_value = NK_I16_MIN;
    nk_size_t min_idx = NK_SIZE_MAX, max_idx = NK_SIZE_MAX;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_i16_t val = *(nk_i16_t const *)ptr;
        if (val < min_value) min_value = val, min_idx = i;
        if (val > max_value) max_value = val, max_idx = i;
    }
    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_u16_serial(                        //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u16_t *min_value_ptr, nk_size_t *min_index_ptr,             //
    nk_u16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_u16_t min_value = NK_U16_MAX, max_value = NK_U16_MIN;
    nk_size_t min_idx = NK_SIZE_MAX, max_idx = NK_SIZE_MAX;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_u16_t val = *(nk_u16_t const *)ptr;
        if (val < min_value) min_value = val, min_idx = i;
        if (val > max_value) max_value = val, max_idx = i;
    }
    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_i32_serial(                        //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i32_t *min_value_ptr, nk_size_t *min_index_ptr,             //
    nk_i32_t *max_value_ptr, nk_size_t *max_index_ptr) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_i32_t min_value = NK_I32_MAX, max_value = NK_I32_MIN;
    nk_size_t min_idx = NK_SIZE_MAX, max_idx = NK_SIZE_MAX;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_i32_t val = *(nk_i32_t const *)ptr;
        if (val < min_value) min_value = val, min_idx = i;
        if (val > max_value) max_value = val, max_idx = i;
    }
    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_u32_serial(                        //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u32_t *min_value_ptr, nk_size_t *min_index_ptr,             //
    nk_u32_t *max_value_ptr, nk_size_t *max_index_ptr) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_u32_t min_value = NK_U32_MAX, max_value = NK_U32_MIN;
    nk_size_t min_idx = NK_SIZE_MAX, max_idx = NK_SIZE_MAX;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_u32_t val = *(nk_u32_t const *)ptr;
        if (val < min_value) min_value = val, min_idx = i;
        if (val > max_value) max_value = val, max_idx = i;
    }
    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_i64_serial(                        //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *min_value_ptr, nk_size_t *min_index_ptr,             //
    nk_i64_t *max_value_ptr, nk_size_t *max_index_ptr) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_i64_t min_value = NK_I64_MAX, max_value = NK_I64_MIN;
    nk_size_t min_idx = NK_SIZE_MAX, max_idx = NK_SIZE_MAX;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_i64_t val = *(nk_i64_t const *)ptr;
        if (val < min_value) min_value = val, min_idx = i;
        if (val > max_value) max_value = val, max_idx = i;
    }
    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_u64_serial(                        //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *min_value_ptr, nk_size_t *min_index_ptr,             //
    nk_u64_t *max_value_ptr, nk_size_t *max_index_ptr) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_u64_t min_value = NK_U64_MAX, max_value = NK_U64_MIN;
    nk_size_t min_idx = NK_SIZE_MAX, max_idx = NK_SIZE_MAX;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_u64_t val = *(nk_u64_t const *)ptr;
        if (val < min_value) min_value = val, min_idx = i;
        if (val > max_value) max_value = val, max_idx = i;
    }
    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_f16_serial(                        //
    nk_f16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f16_t *min_value_ptr, nk_size_t *min_index_ptr,             //
    nk_f16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f16_t min_value = nk_f16_from_u16_(NK_F16_MAX), max_value = nk_f16_from_u16_(NK_F16_MIN);
    nk_size_t min_idx = NK_SIZE_MAX, max_idx = NK_SIZE_MAX;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_f16_t raw_value = *(nk_f16_t const *)ptr;
        nk_fui16_t raw_fui;
        raw_fui.f = raw_value;
        if (nk_f16_is_nan_(raw_fui.u)) continue;
        if (min_idx == NK_SIZE_MAX || nk_f16_order_serial(raw_value, min_value) < 0) min_value = raw_value, min_idx = i;
        if (max_idx == NK_SIZE_MAX || nk_f16_order_serial(raw_value, max_value) > 0) max_value = raw_value, max_idx = i;
    }
    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_bf16_serial(                        //
    nk_bf16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_bf16_t *min_value_ptr, nk_size_t *min_index_ptr,             //
    nk_bf16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_bf16_t min_value = nk_bf16_from_u16_(NK_BF16_MAX), max_value = nk_bf16_from_u16_(NK_BF16_MIN);
    nk_size_t min_idx = NK_SIZE_MAX, max_idx = NK_SIZE_MAX;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_bf16_t raw_value = *(nk_bf16_t const *)ptr;
        nk_fui16_t raw_fui;
        raw_fui.bf = raw_value;
        if (nk_bf16_is_nan_(raw_fui.u)) continue;
        if (min_idx == NK_SIZE_MAX || nk_bf16_order_serial(raw_value, min_value) < 0)
            min_value = raw_value, min_idx = i;
        if (max_idx == NK_SIZE_MAX || nk_bf16_order_serial(raw_value, max_value) > 0)
            max_value = raw_value, max_idx = i;
    }
    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_e4m3_serial(                        //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_e4m3_t *min_value_ptr, nk_size_t *min_index_ptr,             //
    nk_e4m3_t *max_value_ptr, nk_size_t *max_index_ptr) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_e4m3_t min_value = NK_E4M3_MAX, max_value = NK_E4M3_MIN;
    nk_size_t min_idx = NK_SIZE_MAX, max_idx = NK_SIZE_MAX;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_e4m3_t raw_value = *(nk_e4m3_t const *)ptr;
        if (nk_e4m3_is_nan_(raw_value)) continue;
        if (min_idx == NK_SIZE_MAX || nk_e4m3_order_serial(raw_value, min_value) < 0)
            min_value = raw_value, min_idx = i;
        if (max_idx == NK_SIZE_MAX || nk_e4m3_order_serial(raw_value, max_value) > 0)
            max_value = raw_value, max_idx = i;
    }
    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_e5m2_serial(                        //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_e5m2_t *min_value_ptr, nk_size_t *min_index_ptr,             //
    nk_e5m2_t *max_value_ptr, nk_size_t *max_index_ptr) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_e5m2_t min_value = NK_E5M2_MAX, max_value = NK_E5M2_MIN;
    nk_size_t min_idx = NK_SIZE_MAX, max_idx = NK_SIZE_MAX;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_e5m2_t raw_value = *(nk_e5m2_t const *)ptr;
        if (nk_e5m2_is_nan_(raw_value)) continue;
        if (min_idx == NK_SIZE_MAX || nk_e5m2_order_serial(raw_value, min_value) < 0)
            min_value = raw_value, min_idx = i;
        if (max_idx == NK_SIZE_MAX || nk_e5m2_order_serial(raw_value, max_value) > 0)
            max_value = raw_value, max_idx = i;
    }
    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_e2m3_serial(                        //
    nk_e2m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_e2m3_t *min_value_ptr, nk_size_t *min_index_ptr,             //
    nk_e2m3_t *max_value_ptr, nk_size_t *max_index_ptr) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_e2m3_t min_value = NK_E2M3_MAX, max_value = NK_E2M3_MIN;
    nk_size_t min_idx = NK_SIZE_MAX, max_idx = NK_SIZE_MAX;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_e2m3_t raw_value = *(nk_e2m3_t const *)ptr;
        if (nk_e2m3_order_serial(raw_value, min_value) < 0) min_value = raw_value, min_idx = i;
        if (nk_e2m3_order_serial(raw_value, max_value) > 0) max_value = raw_value, max_idx = i;
    }
    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_e3m2_serial(                        //
    nk_e3m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_e3m2_t *min_value_ptr, nk_size_t *min_index_ptr,             //
    nk_e3m2_t *max_value_ptr, nk_size_t *max_index_ptr) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_e3m2_t min_value = NK_E3M2_MAX, max_value = NK_E3M2_MIN;
    nk_size_t min_idx = NK_SIZE_MAX, max_idx = NK_SIZE_MAX;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) {
        nk_e3m2_t raw_value = *(nk_e3m2_t const *)ptr;
        if (nk_e3m2_order_serial(raw_value, min_value) < 0) min_value = raw_value, min_idx = i;
        if (nk_e3m2_order_serial(raw_value, max_value) > 0) max_value = raw_value, max_idx = i;
    }
    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_i4_serial(                          //
    nk_i4x2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i8_t *min_value_ptr, nk_size_t *min_index_ptr,               //
    nk_i8_t *max_value_ptr, nk_size_t *max_index_ptr) {
    count = nk_size_round_up_to_multiple_(count, 2);
    unsigned char const *ptr = (unsigned char const *)data;
    nk_i8_t min_value = 7, max_value = -8; // i4 range: -8 to 7
    nk_size_t min_idx = NK_SIZE_MAX, max_idx = NK_SIZE_MAX;
    for (nk_size_t i = 0; i < count; ++i) {
        nk_size_t byte_idx = i / 2;
        unsigned char byte_val = ptr[byte_idx * stride_bytes];
        nk_i8_t val = nk_i4x2_get_(byte_val, (int)(i & 1));
        if (val < min_value) min_value = val, min_idx = i;
        if (val > max_value) max_value = val, max_idx = i;
    }
    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_u4_serial(                          //
    nk_u4x2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *min_value_ptr, nk_size_t *min_index_ptr,               //
    nk_u8_t *max_value_ptr, nk_size_t *max_index_ptr) {
    count = nk_size_round_up_to_multiple_(count, 2);
    unsigned char const *ptr = (unsigned char const *)data;
    nk_u8_t min_value = 15, max_value = 0; // u4 range: 0 to 15
    nk_size_t min_idx = NK_SIZE_MAX, max_idx = NK_SIZE_MAX;
    for (nk_size_t i = 0; i < count; ++i) {
        nk_size_t byte_idx = i / 2;
        unsigned char byte_val = ptr[byte_idx * stride_bytes];
        nk_u8_t nibble = nk_u4x2_get_(byte_val, (int)(i & 1));
        if (nibble < min_value) min_value = nibble, min_idx = i;
        if (nibble > max_value) max_value = nibble, max_idx = i;
    }
    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_u1_serial(                          //
    nk_u1x8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *min_value_ptr, nk_size_t *min_index_ptr,               //
    nk_u8_t *max_value_ptr, nk_size_t *max_index_ptr) {
    count = nk_size_round_up_to_multiple_(count, 8);
    unsigned char const *ptr = (unsigned char const *)data;
    nk_u8_t min_value = 1, max_value = 0;
    nk_size_t min_idx = NK_SIZE_MAX, max_idx = NK_SIZE_MAX;
    for (nk_size_t i = 0; i < count; ++i) {
        nk_size_t byte_idx = i / 8;
        unsigned char byte_val = ptr[byte_idx * stride_bytes];
        nk_u8_t bit = (byte_val >> (i % 8)) & 1;
        if (bit < min_value) {
            min_value = bit;
            min_idx = i;
            if (min_value == 0 && max_value == 1) {
                *min_value_ptr = min_value, *min_index_ptr = min_idx;
                *max_value_ptr = max_value, *max_index_ptr = max_idx;
                return;
            }
        }
        if (bit > max_value) {
            max_value = bit;
            max_idx = i;
            if (min_value == 0 && max_value == 1) {
                *min_value_ptr = min_value, *min_index_ptr = min_idx;
                *max_value_ptr = max_value, *max_index_ptr = max_idx;
                return;
            }
        }
    }
    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
}

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_REDUCE_SERIAL_H
