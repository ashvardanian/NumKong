/**
 *  @file include/numkong/reduce/simt.cuh
 *  @author Ash Vardanian
 *  @date October 1, 2026
 *  @brief Moments and min/max with indices on the SIMT cores of every CUDA and ROCm device.
 *
 *  @sa include/numkong/reduce.h
 *  @sa include/numkong/reduce/serial.h
 *
 *  Each call launches as many 1024-thread blocks as stay resident: every thread walks values a grid
 *  apart, 8 of them loaded at once to overlap their latencies, each block merges its threads'
 *  partials in shared memory, and its thread 0 folds the block's partial into the outputs with
 *  atomics, which a memset zeroes first, so no call needs scratch memory. Min/max keeps only
 *  indices there, swapped in while their values win, and a one-thread launch then writes the
 *  values. Values sit any number of bytes apart; misaligned ones load byte by byte.
 *
 *  Integers sum exactly in 128 bits per block and saturate into the outputs, so unsigned sums and
 *  all squares match the serial ones, and signed sums do unless a block's partial overflows I64.
 *  F64 inputs sum in F64 with Neumaier compensation like their serial kernel. F32 and BF16 inputs
 *  sum as unevaluated F32 pairs and the other narrow floats in F32 with Neumaier compensation,
 *  sparing the F64 units, which run at 2 results per clock per SM on the B300. Both loops square
 *  through the vendor's rounded product, so they live in `cuda.cuh` beside this file. E2M1 sums
 *  twice its values as exact integers.
 *
 *  Min/max orders every value by a signed key: the narrow floats by the sign-magnitude order of
 *  `nk_*_order_`, -0 below +0, and F32 and F64 by value, ±0 tied, both skipping NaNs. Ties go to
 *  the first index, and a side with no value keeps the serial sentinels and @c NUMKONG_SIZE_MAX.
 */
#ifndef NUMKONG_REDUCE_SIMT_CUH
#define NUMKONG_REDUCE_SIMT_CUH

#if NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_

#include "numkong/reduce/serial.h" // `nk_i4x2_get_`, `nk_u64_saturating_add_`
#include "numkong/cast/simt.cuh"   // `nk_e5m2_to_f32_simt_`, `nk_e2m1_nibble_to_i8x2_simt_`

#if defined(__cplusplus)
extern "C" {
#endif

enum { nk_reduce_threads_simt_k = 1024, nk_reduce_batch_simt_k = 8 };

/** Everything one reduction shares, passed by value as its kernel's only argument: @c first and
 *  @c second are the sum and sum of squares, or the minimum and maximum with their indices. */
typedef struct {
    unsigned char const *data;
    nk_size_t count;
    nk_size_t stride;
    void *first;
    nk_size_t *first_index;
    void *second;
    nk_size_t *second_index;
    int aligned;
} nk_reduce_arguments_t;

/** The arguments of a reduction over @p data, flagging whether every value is aligned to its
 *  @p value_bytes. */
NUMKONG_INLINE nk_reduce_arguments_t nk_reduce_arguments_simt_(nk_size_t value_bytes, void const *data, nk_size_t count,
                                                               nk_size_t stride, void *first, nk_size_t *first_index,
                                                               void *second, nk_size_t *second_index) {
    nk_reduce_arguments_t arguments;
    arguments.data = (unsigned char const *)data, arguments.count = count, arguments.stride = stride;
    arguments.first = first, arguments.first_index = first_index;
    arguments.second = second, arguments.second_index = second_index;
    arguments.aligned = !(((nk_size_t)data | stride) & (value_bytes - 1));
    return arguments;
}

/** This thread's place among every thread of the grid, and their count. */
NUMKONG_DEVICE nk_size_t nk_reduce_lane_simt_(void) { return (nk_size_t)blockIdx.x * blockDim.x + threadIdx.x; }
NUMKONG_DEVICE nk_size_t nk_reduce_lanes_simt_(void) { return (nk_size_t)gridDim.x * blockDim.x; }

/** Adds @p value to the U64 at @p slot, saturating, which no order of non-negative terms alters. */
NUMKONG_DEVICE void nk_reduce_add_u64_simt_(nk_u64_t *slot, nk_u64_t value) {
    unsigned long long seen = *(unsigned long long volatile *)slot, prior;
    while ((prior = atomicCAS((unsigned long long *)slot, seen, nk_u64_saturating_add_(seen, value))) != seen)
        seen = prior;
}

/** Adds @p value to the I64 at @p slot, saturating. */
NUMKONG_DEVICE void nk_reduce_add_i64_simt_(nk_i64_t *slot, nk_i64_t value) {
    unsigned long long seen = *(unsigned long long volatile *)slot, prior;
    while ((prior = atomicCAS((unsigned long long *)slot, seen,
                              (unsigned long long)nk_i64_saturating_add_((nk_i64_t)seen, value))) != seen)
        seen = prior;
}

/** The @p bytes wide value at @p address as little-endian bits, assembled byte by byte unless
 *  @p aligned. */
NUMKONG_DEVICE nk_u64_t nk_reduce_load_simt_(unsigned char const *address, unsigned bytes, int aligned) {
    nk_u64_t bits = 0;
    if (aligned) switch (bytes) {
        case 8: return *(nk_u64_t const *)address;
        case 4: return *(nk_u32_t const *)address;
        case 2: return *(nk_u16_t const *)address;
        default: return *address;
        }
    for (unsigned byte = 0; byte != bytes; ++byte) bits |= (nk_u64_t)address[byte] << (8 * byte);
    return bits;
}

/** Value @p index of @p dtype as the bits its min/max output stores: packed values unpacked, I4
 *  sign-extended, and U1 bits counted from the most significant, like the serial kernels. */
NUMKONG_DEVICE nk_u64_t nk_reduce_raw_simt_(nk_dtype_t dtype, nk_reduce_arguments_t const *arguments, nk_size_t index) {
    unsigned char const *data = arguments->data;
    nk_size_t const stride = arguments->stride;
    switch (dtype) {
    case nk_i4_k: return (nk_u8_t)nk_i4x2_get_(data[index / 2 * stride], (int)(index & 1));
    case nk_u4_k:
    case nk_e2m1_k: return nk_u4x2_get_(data[index / 2 * stride], (int)(index & 1));
    case nk_u1_k: return (data[index / 8 * stride] >> (7 - index % 8)) & 1u;
    default:
        return nk_reduce_load_simt_(data + index * stride, (unsigned)(nk_dtype_bits(dtype) / NUMKONG_BITS_PER_BYTE),
                                    arguments->aligned);
    }
}

/** Loads the raw bits of the values @p step apart from @p first and below @p limit, up to
 *  @c nk_reduce_batch_simt_k of them, before any is used, keeping them all in flight. */
NUMKONG_DEVICE void nk_reduce_batch_simt_(nk_dtype_t dtype, nk_reduce_arguments_t const *arguments, nk_size_t first,
                                          nk_size_t step, nk_size_t limit, nk_u64_t *raws) {
#pragma unroll
    for (unsigned slot = 0; slot != nk_reduce_batch_simt_k; ++slot)
        raws[slot] = first + slot * step < limit ? nk_reduce_raw_simt_(dtype, arguments, first + slot * step) : 0;
}

/** Sets @p key to the order of @p raw among values of @p dtype, returning whether it takes part in
 *  min/max at all, as NaNs don't. */
NUMKONG_DEVICE int nk_reduce_order_simt_(nk_dtype_t dtype, nk_u64_t raw, nk_i64_t *key) {
    nk_u64_t const f32_magnitude = raw & 0x7FFFFFFFu, f64_magnitude = raw & 0x7FFFFFFFFFFFFFFFull;
    nk_i32_t const narrow = (nk_i32_t)raw, six_bits = (nk_i32_t)(raw & 0x3Fu);
    switch (dtype) {
    case nk_i8_k:
    case nk_i4_k: *key = (nk_i8_t)raw; return 1;
    case nk_i16_k: *key = (nk_i16_t)raw; return 1;
    case nk_i32_k: *key = (nk_i32_t)raw; return 1;
    case nk_i64_k: *key = (nk_i64_t)raw; return 1;
    case nk_u64_k: *key = (nk_i64_t)(raw ^ 0x8000000000000000ull); return 1;
    case nk_f32_k:
        *key = raw & 0x80000000u ? -(nk_i64_t)f32_magnitude : (nk_i64_t)f32_magnitude;
        return f32_magnitude <= 0x7F800000u;
    case nk_f64_k:
        *key = raw >> 63 ? -(nk_i64_t)f64_magnitude : (nk_i64_t)f64_magnitude;
        return f64_magnitude <= 0x7FF0000000000000ull;
    case nk_f16_k: *key = narrow ^ -(narrow >> 15); return (raw & 0x7FFFu) <= 0x7C00u;
    case nk_bf16_k: *key = narrow ^ -(narrow >> 15); return (raw & 0x7FFFu) <= 0x7F80u;
    case nk_e4m3_k: *key = narrow ^ -(narrow >> 7); return (raw & 0x7Fu) != 0x7Fu;
    case nk_e5m2_k: *key = narrow ^ -(narrow >> 7); return (raw & 0x7Fu) <= 0x7Cu;
    case nk_e2m3_k:
    case nk_e3m2_k: *key = six_bits ^ -(six_bits >> 5); return 1;
    default: *key = (nk_i64_t)raw; return 1;
    }
}

/** One thread's or one block's candidates for the minimum and the maximum. */
typedef struct {
    nk_i64_t min_key;
    nk_size_t min_index;
    nk_i64_t max_key;
    nk_size_t max_index;
} nk_reduce_minmax_state_t;

/** Folds @p other into @p state, keeping the smaller key, and the first index among equal ones. */
NUMKONG_DEVICE void nk_reduce_minmax_merge_simt_(nk_reduce_minmax_state_t *state,
                                                 nk_reduce_minmax_state_t const *other) {
    if (other->min_index != NUMKONG_SIZE_MAX &&
        (state->min_index == NUMKONG_SIZE_MAX || other->min_key < state->min_key ||
         (other->min_key == state->min_key && other->min_index < state->min_index)))
        state->min_key = other->min_key, state->min_index = other->min_index;
    if (other->max_index != NUMKONG_SIZE_MAX &&
        (state->max_index == NUMKONG_SIZE_MAX || other->max_key > state->max_key ||
         (other->max_key == state->max_key && other->max_index < state->max_index)))
        state->max_key = other->max_key, state->max_index = other->max_index;
}

/** Stores @p raw as a @p dtype min/max output, sub-byte types widened to a byte. */
NUMKONG_DEVICE void nk_reduce_store_raw_simt_(nk_dtype_t dtype, nk_u64_t raw, void *output) {
    switch (nk_dtype_bits(dtype)) {
    case 64: *(nk_u64_t *)output = raw; break;
    case 32: *(nk_u32_t *)output = (nk_u32_t)raw; break;
    case 16: *(nk_u16_t *)output = (nk_u16_t)raw; break;
    default: *(nk_u8_t *)output = (nk_u8_t)raw; break;
    }
}

/** Swaps @p index, with order @p key, into the index at @p slot while it beats the one there:
 *  smaller keys when @p smallest, larger ones otherwise, and the first index among equal keys. */
NUMKONG_DEVICE void nk_reduce_minmax_offer_simt_(nk_dtype_t dtype, nk_reduce_arguments_t const *arguments, nk_i64_t key,
                                                 nk_size_t index, nk_size_t *slot, int smallest) {
    unsigned long long seen = *(unsigned long long volatile *)slot, prior;
    while (index != NUMKONG_SIZE_MAX) {
        if (seen != NUMKONG_SIZE_MAX) {
            nk_i64_t held;
            nk_reduce_order_simt_(dtype, nk_reduce_raw_simt_(dtype, arguments, seen), &held);
            if (smallest ? key > held : key < held) return;
            if (key == held && index > seen) return;
        }
        if ((prior = atomicCAS((unsigned long long *)slot, seen, index)) == seen) return;
        seen = prior;
    }
}

/** Writes the values of the indices the grid settled on, a side whose key never beat the
 *  @p min_sentinel or @p max_sentinel reporting the sentinel itself, as the serial kernels do. */
NUMKONG_DEVICE void nk_reduce_minmax_finish_simt_(nk_dtype_t dtype, nk_u64_t min_sentinel, nk_u64_t max_sentinel,
                                                  nk_reduce_arguments_t const *arguments) {
    nk_size_t const min_index = *arguments->first_index, max_index = *arguments->second_index;
    nk_i64_t min_key = 0, max_key = 0, min_sentinel_key, max_sentinel_key;
    nk_reduce_order_simt_(dtype, min_sentinel, &min_sentinel_key);
    nk_reduce_order_simt_(dtype, max_sentinel, &max_sentinel_key);
    nk_u64_t const min_raw = min_index == NUMKONG_SIZE_MAX ? min_sentinel
                                                           : nk_reduce_raw_simt_(dtype, arguments, min_index);
    nk_u64_t const max_raw = max_index == NUMKONG_SIZE_MAX ? max_sentinel
                                                           : nk_reduce_raw_simt_(dtype, arguments, max_index);
    nk_reduce_order_simt_(dtype, min_raw, &min_key), nk_reduce_order_simt_(dtype, max_raw, &max_key);
    nk_reduce_store_raw_simt_(dtype, min_key == min_sentinel_key ? min_sentinel : min_raw, arguments->first);
    nk_reduce_store_raw_simt_(dtype, max_key == max_sentinel_key ? max_sentinel : max_raw, arguments->second);
}

/** Offers the block's first minimum and maximum of @p dtype values to the output indices. */
NUMKONG_DEVICE void nk_reduce_minmax_simt_(nk_dtype_t dtype, nk_reduce_arguments_t const *arguments) {
    __shared__ nk_reduce_minmax_state_t states[nk_reduce_threads_simt_k];
    nk_reduce_minmax_state_t state;
    nk_i64_t key;
    state.min_key = 0, state.min_index = NUMKONG_SIZE_MAX, state.max_key = 0, state.max_index = NUMKONG_SIZE_MAX;
    for (nk_size_t first = nk_reduce_lane_simt_(); first < arguments->count;
         first += nk_reduce_lanes_simt_() * nk_reduce_batch_simt_k) {
        nk_u64_t raws[nk_reduce_batch_simt_k];
        nk_reduce_batch_simt_(dtype, arguments, first, nk_reduce_lanes_simt_(), arguments->count, raws);
#pragma unroll
        for (unsigned slot = 0; slot != nk_reduce_batch_simt_k; ++slot) {
            nk_size_t const index = first + slot * nk_reduce_lanes_simt_();
            if (index >= arguments->count) break;
            if (!nk_reduce_order_simt_(dtype, raws[slot], &key)) continue;
            if (state.min_index == NUMKONG_SIZE_MAX || key < state.min_key)
                state.min_key = key, state.min_index = index;
            if (state.max_index == NUMKONG_SIZE_MAX || key > state.max_key)
                state.max_key = key, state.max_index = index;
        }
    }
    states[threadIdx.x] = state;
    __syncthreads();
    for (unsigned half = blockDim.x / 2; half; half >>= 1) {
        if (threadIdx.x < half) nk_reduce_minmax_merge_simt_(&states[threadIdx.x], &states[threadIdx.x + half]);
        __syncthreads();
    }
    if (threadIdx.x) return;
    nk_reduce_minmax_offer_simt_(dtype, arguments, states[0].min_key, states[0].min_index, arguments->first_index, 1);
    nk_reduce_minmax_offer_simt_(dtype, arguments, states[0].max_key, states[0].max_index, arguments->second_index, 0);
}

/** One thread's or one block's integer sums: a 128-bit sum and a saturating sum of squares. */
typedef struct {
    nk_u64_t sum_low;
    nk_i64_t sum_high;
    nk_u64_t sumsq;
} nk_reduce_integer_moments_t;

/** Adds the 128-bit value @p high : @p low to the sum and @p square to the sum of squares. */
NUMKONG_DEVICE void nk_reduce_integer_moments_add_simt_(nk_reduce_integer_moments_t *state, nk_u64_t low, nk_i64_t high,
                                                        nk_u64_t square) {
    nk_u64_t const sum_low = state->sum_low + low;
    state->sum_high += high + (sum_low < low);
    state->sum_low = sum_low, state->sumsq = nk_u64_saturating_add_(state->sumsq, square);
}

/** Adds the integer @p dtype value with @p raw bits, E2M1 as twice its value, an integer. */
NUMKONG_DEVICE void nk_reduce_integer_moments_raw_simt_(nk_dtype_t dtype, nk_u64_t raw,
                                                        nk_reduce_integer_moments_t *state) {
    nk_i64_t value;
    switch (dtype) {
    case nk_u4_k:
    case nk_u1_k:
    case nk_u8_k:
    case nk_u16_k:
    case nk_u32_k: nk_reduce_integer_moments_add_simt_(state, raw, 0, raw * raw); return;
    case nk_u64_k: nk_reduce_integer_moments_add_simt_(state, raw, 0, nk_u64_saturating_mul_(raw, raw)); return;
    case nk_i64_k:
        value = (nk_i64_t)raw;
        nk_reduce_integer_moments_add_simt_(state, raw, value >> 63, (nk_u64_t)nk_i64_saturating_mul_(value, value));
        return;
    default:
        value = dtype == nk_i16_k    ? (nk_i16_t)raw
                : dtype == nk_i32_k  ? (nk_i32_t)raw
                : dtype == nk_e2m1_k ? nk_e2m1_nibble_to_i8x2_simt_((nk_u32_t)raw)
                                     : (nk_i8_t)raw;
        nk_reduce_integer_moments_add_simt_(state, (nk_u64_t)value, value >> 63, (nk_u64_t)(value * value));
        return;
    }
}

/** Sums integer @p dtype values and their squares: signed sums saturate from 128 bits into I64,
 *  unsigned ones into U64, and E2M1's doubled sums halve into F32 like the serial kernels. */
NUMKONG_DEVICE void nk_reduce_integer_moments_simt_(nk_dtype_t dtype, nk_reduce_arguments_t const *arguments) {
    __shared__ nk_reduce_integer_moments_t states[nk_reduce_threads_simt_k];
    nk_size_t const limit = dtype == nk_i4_k || dtype == nk_u4_k ? nk_size_round_up_to_multiple_(arguments->count, 2)
                            : dtype == nk_u1_k                   ? nk_size_round_up_to_multiple_(arguments->count, 8)
                                                                 : arguments->count;
    nk_reduce_integer_moments_t state;
    state.sum_low = 0, state.sum_high = 0, state.sumsq = 0;
    for (nk_size_t first = nk_reduce_lane_simt_(); first < limit;
         first += nk_reduce_lanes_simt_() * nk_reduce_batch_simt_k) {
        nk_u64_t raws[nk_reduce_batch_simt_k];
        nk_reduce_batch_simt_(dtype, arguments, first, nk_reduce_lanes_simt_(), limit, raws);
#pragma unroll
        for (unsigned slot = 0; slot != nk_reduce_batch_simt_k; ++slot) {
            if (first + slot * nk_reduce_lanes_simt_() >= limit) break;
            nk_reduce_integer_moments_raw_simt_(dtype, raws[slot], &state);
        }
    }
    states[threadIdx.x] = state;
    __syncthreads();
    for (unsigned half = blockDim.x / 2; half; half >>= 1) {
        if (threadIdx.x < half)
            nk_reduce_integer_moments_add_simt_(&states[threadIdx.x], states[threadIdx.x + half].sum_low,
                                                states[threadIdx.x + half].sum_high, states[threadIdx.x + half].sumsq);
        __syncthreads();
    }
    if (threadIdx.x) return;
    state = states[0];
    nk_i64_t const sum_low_signed = (nk_i64_t)state.sum_low;
    switch (dtype) {
    case nk_e2m1_k:
        atomicAdd((nk_f32_t *)arguments->first, (nk_f32_t)sum_low_signed * 0.5f);
        atomicAdd((nk_f32_t *)arguments->second, (nk_f32_t)(nk_i64_t)state.sumsq * 0.25f);
        return;
    case nk_i8_k:
    case nk_i16_k:
    case nk_i32_k:
    case nk_i64_k:
    case nk_i4_k:
        nk_reduce_add_i64_simt_((nk_i64_t *)arguments->first, state.sum_high == (sum_low_signed >> 63) ? sum_low_signed
                                                              : state.sum_high >= 0 ? NUMKONG_I64_MAX
                                                                                    : NUMKONG_I64_MIN);
        break;
    default:
        nk_reduce_add_u64_simt_((nk_u64_t *)arguments->first, state.sum_high ? NUMKONG_U64_MAX : state.sum_low);
        break;
    }
    nk_reduce_add_u64_simt_((nk_u64_t *)arguments->second, state.sumsq);
}

/** Widens the float @p dtype value with @p raw bits to F32, exactly. */
NUMKONG_DEVICE nk_f32_t nk_reduce_f32_simt_(nk_dtype_t dtype, nk_u64_t raw) {
    nk_u8_t const code = (nk_u8_t)raw;
    nk_u16_t const half = (nk_u16_t)raw;
    nk_f32_t value = 0;
    switch (dtype) {
    case nk_f32_k: value = __uint_as_float((nk_u32_t)raw); break;
    case nk_f16_k: nk_f16_to_f32_simt_((nk_f16_t const *)&half, &value); break;
    case nk_bf16_k: nk_bf16_to_f32_simt_((nk_bf16_t const *)&half, &value); break;
    case nk_e4m3_k: nk_e4m3_to_f32_simt_(&code, &value); break;
    case nk_e5m2_k: nk_e5m2_to_f32_simt_(&code, &value); break;
    case nk_e2m3_k: nk_e2m3_to_f32_simt_(&code, &value); break;
    case nk_e3m2_k: nk_e3m2_to_f32_simt_(&code, &value); break;
    default: break;
    }
    return value;
}

/** Adds @p addend to @p sum through TwoSum, gathering the rounding error into @p compensation. */
NUMKONG_DEVICE void nk_f32_two_sum_simt_(nk_f32_t addend, nk_f32_t *sum, nk_f32_t *compensation) {
    nk_f32_t const total = *sum + addend, virtual_addend = total - *sum;
    *compensation += (*sum - (total - virtual_addend)) + (addend - virtual_addend);
    *sum = total;
}

/** One thread's or one block's compensated F32 sums. */
typedef struct {
    nk_f32_t sum;
    nk_f32_t sum_compensation;
    nk_f32_t sumsq;
    nk_f32_t sumsq_compensation;
} nk_reduce_f32_moments_t;

/** Adds @p value to the Neumaier sum @p sum with its running @p compensation in F64, as the serial
 *  F64 moments do. */
NUMKONG_DEVICE void nk_reduce_neumaier_f64_simt_(nk_f64_t *sum, nk_f64_t *compensation, nk_f64_t value) {
    nk_f64_t const tentative = *sum + value;
    *compensation += fabs(*sum) >= fabs(value) ? (*sum - tentative) + value : (value - tentative) + *sum;
    *sum = tentative;
}

/** Adds @p value to the unevaluated F32 sum @p high + @p low, keeping each rounding error with
 *  TwoSum, so the pair carries twice the F32 precision. */
NUMKONG_DEVICE void nk_reduce_two_sum_f32_simt_(nk_f32_t *high, nk_f32_t *low, nk_f32_t value) {
    nk_f32_t const sum = *high + value, value_part = sum - *high;
    nk_f32_t const error = (*high - (sum - value_part)) + (value - value_part) + *low;
    *high = sum + error, *low = error - (*high - sum);
}

/** One thread's or one block's F64 sums with their Neumaier compensations. */
typedef struct {
    nk_f64_t sum;
    nk_f64_t sum_compensation;
    nk_f64_t sumsq;
    nk_f64_t sumsq_compensation;
} nk_reduce_f64_moments_t;

/** Merges every thread's @p state across the block with Neumaier compensation, leaving the totals
 *  in thread 0's @p state. */
NUMKONG_DEVICE void nk_reduce_f64_moments_merge_simt_(nk_reduce_f64_moments_t *state) {
    __shared__ nk_reduce_f64_moments_t states[nk_reduce_threads_simt_k];
    states[threadIdx.x] = *state;
    __syncthreads();
    for (unsigned half = blockDim.x / 2; half; half >>= 1) {
        if (threadIdx.x < half) {
            nk_reduce_f64_moments_t *into = &states[threadIdx.x];
            nk_reduce_f64_moments_t const *other = &states[threadIdx.x + half];
            nk_reduce_neumaier_f64_simt_(&into->sum, &into->sum_compensation, other->sum);
            nk_reduce_neumaier_f64_simt_(&into->sumsq, &into->sumsq_compensation, other->sumsq);
            into->sum_compensation += other->sum_compensation, into->sumsq_compensation += other->sumsq_compensation;
        }
        __syncthreads();
    }
    *state = states[0];
}

/** Whether F32 squares @p value exactly and sums a thread's squares without overflow: zero, or a
 *  magnitude from 2⁻⁵⁰ up to 2⁵¹. */
NUMKONG_DEVICE int nk_reduce_pairs_hold_simt_(nk_f32_t value) {
    return ((__float_as_uint(value) >> 23) & 0xFFu) - 77u <= 100u || value == 0;
}

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_
#endif // NUMKONG_REDUCE_SIMT_CUH
