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
 *  One generator emits every moments kernel from typed pieces: the input's value type, the
 *  converter that widens it, and the accumulator its values add into. Floats sum one tier above
 *  their input, F32 ones in plain F64 and the narrower ones in plain F32, and only F64 inputs,
 *  whose sums stay F64, sum with Neumaier compensation like their serial kernel. Signed integers
 *  sum exactly in 128 bits per block and saturate into the outputs, so their sums match the serial
 *  ones unless a block's partial overflows I64, unsigned sums and all squares saturate in U64 like
 *  the serial ones, and E2M1 sums twice its values as exact integers.
 *
 *  Min/max orders every value by a signed key: the narrow floats by the sign-magnitude order of
 *  `nk_*_order_`, -0 below +0, and F32 and F64 by value, ±0 tied, both skipping NaNs. Ties go to
 *  the first index, and a side with no value keeps the serial sentinels and @c NUMKONG_SIZE_MAX.
 */
#ifndef NUMKONG_REDUCE_SIMT_CUH
#define NUMKONG_REDUCE_SIMT_CUH

#if NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_

#include "numkong/reduce/serial.h" // `nk_get_i4x2_serial_`, `nk_saturating_add_u64_serial_`
#include "numkong/cast/simt.cuh"   // `nk_e5m2_to_f32_simt_`, `nk_e2m1_nibble_to_i8x2_simt_`

#if defined(__cplusplus)
extern "C" {
#endif

enum { nk_reduce_threads_simt_k = 1024, nk_reduce_finish_threads_simt_k = 1, nk_reduce_batch_simt_k = 8 };

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
    while ((prior = atomicCAS((unsigned long long *)slot, seen, nk_saturating_add_u64_serial_(seen, value))) != seen)
        seen = prior;
}

/** Adds @p value to the I64 at @p slot, saturating. */
NUMKONG_DEVICE void nk_reduce_add_i64_simt_(nk_i64_t *slot, nk_i64_t value) {
    unsigned long long seen = *(unsigned long long volatile *)slot, prior;
    while ((prior = atomicCAS((unsigned long long *)slot, seen,
                              (unsigned long long)nk_saturating_add_i64_serial_((nk_i64_t)seen, value))) != seen)
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
    case nk_i4_k: return (nk_u8_t)nk_get_i4x2_serial_(data[index / 2 * stride], (int)(index & 1));
    case nk_u4_k:
    case nk_e2m1_k: return nk_get_u4x2_serial_(data[index / 2 * stride], (int)(index & 1));
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

/** Copies the @p bytes of a misaligned value from @p source to @p target, one at a time. */
NUMKONG_DEVICE void nk_reduce_copy_simt_(void *target, unsigned char const *source, nk_size_t bytes) {
    unsigned char *target_bytes = (unsigned char *)target;
    for (nk_size_t byte = 0; byte != bytes; ++byte) target_bytes[byte] = source[byte];
}

/** Widens the two I4 nibbles of one byte to I64, the high nibble first. */
NUMKONG_DEVICE void nk_i4x2_to_i64x2_simt_(nk_i4x2_t const *src, nk_i64_t *dest) {
    dest[0] = nk_i4x2_high_(*src), dest[1] = nk_i4x2_low_(*src);
}

/** Widens the two U4 nibbles of one byte to U64, the high nibble first. */
NUMKONG_DEVICE void nk_u4x2_to_u64x2_simt_(nk_u4x2_t const *src, nk_u64_t *dest) {
    dest[0] = nk_u4x2_high_(*src), dest[1] = nk_u4x2_low_(*src);
}

/** Widens the eight bits of one byte to U64, the most significant first. */
NUMKONG_DEVICE void nk_u1x8_to_u64x8_simt_(nk_u1x8_t const *src, nk_u64_t *dest) {
    for (unsigned bit = 0; bit != NUMKONG_BITS_PER_BYTE; ++bit) dest[bit] = (*src >> (7 - bit)) & 1u;
}

/** One thread's or one block's plain F32 sums, which narrow floats widen into exactly. */
typedef struct {
    nk_f32_t sum;
    nk_f32_t sumsq;
} nk_reduce_f32_moments_t;

/** Adds @p value and its square to the sums in @p state. */
NUMKONG_DEVICE void nk_reduce_f32_moments_add_simt_(nk_reduce_f32_moments_t *state, nk_f32_t value) {
    state->sum += value, state->sumsq += value * value;
}

/** Adds the sums in @p other to those in @p state. */
NUMKONG_DEVICE void nk_reduce_f32_moments_merge_simt_(nk_reduce_f32_moments_t *state,
                                                      nk_reduce_f32_moments_t const *other) {
    state->sum += other->sum, state->sumsq += other->sumsq;
}

/** Adds the block's sums in @p state to the outputs. */
NUMKONG_DEVICE void nk_reduce_f32_moments_store_simt_(nk_reduce_f32_moments_t const *state,
                                                      nk_reduce_arguments_t const *arguments) {
    atomicAdd((nk_f32_t *)arguments->first, state->sum), atomicAdd((nk_f32_t *)arguments->second, state->sumsq);
}

/** One thread's or one block's plain F64 sums, which hold the squares of F32 values exactly. */
typedef struct {
    nk_f64_t sum;
    nk_f64_t sumsq;
} nk_reduce_f64_moments_t;

/** Adds @p value and its square to the sums in @p state. */
NUMKONG_DEVICE void nk_reduce_f64_moments_add_simt_(nk_reduce_f64_moments_t *state, nk_f64_t value) {
    state->sum += value, state->sumsq += value * value;
}

/** Adds the sums in @p other to those in @p state. */
NUMKONG_DEVICE void nk_reduce_f64_moments_merge_simt_(nk_reduce_f64_moments_t *state,
                                                      nk_reduce_f64_moments_t const *other) {
    state->sum += other->sum, state->sumsq += other->sumsq;
}

/** Adds the block's sums in @p state to the outputs. */
NUMKONG_DEVICE void nk_reduce_f64_moments_store_simt_(nk_reduce_f64_moments_t const *state,
                                                      nk_reduce_arguments_t const *arguments) {
    atomicAdd((nk_f64_t *)arguments->first, state->sum), atomicAdd((nk_f64_t *)arguments->second, state->sumsq);
}

/** Adds @p value to the Neumaier sum @p sum with its running @p compensation in F64, as the serial
 *  F64 moments do. */
NUMKONG_DEVICE void nk_reduce_neumaier_f64_simt_(nk_f64_t *sum, nk_f64_t *compensation, nk_f64_t value) {
    nk_f64_t const tentative = *sum + value;
    *compensation += fabs(*sum) >= fabs(value) ? (*sum - tentative) + value : (value - tentative) + *sum;
    *sum = tentative;
}

/** One thread's or one block's F64 sums with their Neumaier compensations. */
typedef struct {
    nk_f64_t sum;
    nk_f64_t sum_compensation;
    nk_f64_t sumsq;
    nk_f64_t sumsq_compensation;
} nk_reduce_compensated_moments_t;

/** Adds @p value and its square to the compensated sums in @p state. */
NUMKONG_DEVICE void nk_reduce_compensated_moments_add_simt_(nk_reduce_compensated_moments_t *state, nk_f64_t value) {
    nk_reduce_neumaier_f64_simt_(&state->sum, &state->sum_compensation, value);
    nk_reduce_neumaier_f64_simt_(&state->sumsq, &state->sumsq_compensation, value * value);
}

/** Adds the compensated sums in @p other to those in @p state. */
NUMKONG_DEVICE void nk_reduce_compensated_moments_merge_simt_(nk_reduce_compensated_moments_t *state,
                                                              nk_reduce_compensated_moments_t const *other) {
    nk_reduce_neumaier_f64_simt_(&state->sum, &state->sum_compensation, other->sum);
    nk_reduce_neumaier_f64_simt_(&state->sumsq, &state->sumsq_compensation, other->sumsq);
    state->sum_compensation += other->sum_compensation, state->sumsq_compensation += other->sumsq_compensation;
}

/** Adds the block's compensated sums in @p state to the outputs. */
NUMKONG_DEVICE void nk_reduce_compensated_moments_store_simt_(nk_reduce_compensated_moments_t const *state,
                                                              nk_reduce_arguments_t const *arguments) {
    atomicAdd((nk_f64_t *)arguments->first, state->sum + state->sum_compensation);
    atomicAdd((nk_f64_t *)arguments->second, state->sumsq + state->sumsq_compensation);
}

/** One thread's or one block's signed sums: a 128-bit sum and a saturating sum of squares. */
typedef struct {
    nk_u64_t sum_low;
    nk_i64_t sum_high;
    nk_u64_t sumsq;
} nk_reduce_i64_moments_t;

/** Adds the 128-bit sum and the sum of squares in @p other to those in @p state. */
NUMKONG_DEVICE void nk_reduce_i64_moments_merge_simt_(nk_reduce_i64_moments_t *state,
                                                      nk_reduce_i64_moments_t const *other) {
    nk_u64_t const sum_low = state->sum_low + other->sum_low;
    state->sum_high += other->sum_high + (sum_low < other->sum_low);
    state->sum_low = sum_low, state->sumsq = nk_saturating_add_u64_serial_(state->sumsq, other->sumsq);
}

/** Adds @p value and its square, saturating at I64 like the serial kernels, to @p state. */
NUMKONG_DEVICE void nk_reduce_i64_moments_add_simt_(nk_reduce_i64_moments_t *state, nk_i64_t value) {
    nk_reduce_i64_moments_t const single = {(nk_u64_t)value, value >> 63,
                                            (nk_u64_t)nk_saturating_mul_i64_serial_(value, value)};
    nk_reduce_i64_moments_merge_simt_(state, &single);
}

/** Adds the block's sums in @p state to the outputs, its 128-bit sum saturated into I64. */
NUMKONG_DEVICE void nk_reduce_i64_moments_store_simt_(nk_reduce_i64_moments_t const *state,
                                                      nk_reduce_arguments_t const *arguments) {
    nk_i64_t const sum_low = (nk_i64_t)state->sum_low;
    nk_reduce_add_i64_simt_((nk_i64_t *)arguments->first, state->sum_high == (sum_low >> 63) ? sum_low
                                                          : state->sum_high >= 0             ? NUMKONG_I64_MAX
                                                                                             : NUMKONG_I64_MIN);
    nk_reduce_add_u64_simt_((nk_u64_t *)arguments->second, state->sumsq);
}

/** One thread's or one block's unsigned integer sums, saturating in U64, which no order of
 *  non-negative terms alters. */
typedef struct {
    nk_u64_t sum;
    nk_u64_t sumsq;
} nk_reduce_u64_moments_t;

/** Adds @p value and its square to the saturating sums in @p state. */
NUMKONG_DEVICE void nk_reduce_u64_moments_add_simt_(nk_reduce_u64_moments_t *state, nk_u64_t value) {
    state->sum = nk_saturating_add_u64_serial_(state->sum, value);
    state->sumsq = nk_saturating_add_u64_serial_(state->sumsq, nk_saturating_mul_u64_serial_(value, value));
}

/** Adds the saturating sums in @p other to those in @p state. */
NUMKONG_DEVICE void nk_reduce_u64_moments_merge_simt_(nk_reduce_u64_moments_t *state,
                                                      nk_reduce_u64_moments_t const *other) {
    state->sum = nk_saturating_add_u64_serial_(state->sum, other->sum);
    state->sumsq = nk_saturating_add_u64_serial_(state->sumsq, other->sumsq);
}

/** Adds the block's saturating sums in @p state to the outputs. */
NUMKONG_DEVICE void nk_reduce_u64_moments_store_simt_(nk_reduce_u64_moments_t const *state,
                                                      nk_reduce_arguments_t const *arguments) {
    nk_reduce_add_u64_simt_((nk_u64_t *)arguments->first, state->sum);
    nk_reduce_add_u64_simt_((nk_u64_t *)arguments->second, state->sumsq);
}

/** One thread's or one block's E2M1 sums as the exact integers that twice every value and four
 *  times every square are. */
typedef struct {
    nk_i64_t doubled_sum;
    nk_i64_t quadrupled_sumsq;
} nk_reduce_e2m1_moments_t;

/** Adds the E2M1 @p value, widened to F32, and its square to the integer sums in @p state. */
NUMKONG_DEVICE void nk_reduce_e2m1_moments_add_simt_(nk_reduce_e2m1_moments_t *state, nk_f32_t value) {
    nk_i64_t const doubled = (nk_i64_t)(value * 2);
    state->doubled_sum += doubled, state->quadrupled_sumsq += doubled * doubled;
}

/** Adds the integer sums in @p other to those in @p state. */
NUMKONG_DEVICE void nk_reduce_e2m1_moments_merge_simt_(nk_reduce_e2m1_moments_t *state,
                                                       nk_reduce_e2m1_moments_t const *other) {
    state->doubled_sum += other->doubled_sum, state->quadrupled_sumsq += other->quadrupled_sumsq;
}

/** Adds the block's integer sums in @p state, halved and quartered into F32, to the outputs. */
NUMKONG_DEVICE void nk_reduce_e2m1_moments_store_simt_(nk_reduce_e2m1_moments_t const *state,
                                                       nk_reduce_arguments_t const *arguments) {
    atomicAdd((nk_f32_t *)arguments->first, (nk_f32_t)state->doubled_sum * 0.5f);
    atomicAdd((nk_f32_t *)arguments->second, (nk_f32_t)state->quadrupled_sumsq * 0.25f);
}

/** Generates the moments kernel of @p input_type for @p isa_suffix and its entry point, launched
 *  through the helpers of @p vendor, @c cuda or @c rocm . Each thread loads @p input_value_type
 *  values a grid apart, widens each to @p dimensions_per_value values of @p sum_type through
 *  @p load_and_convert, and adds them to its @c nk_reduce_<accumulator>_moments_t , which the
 *  block merges in shared memory before thread 0 adds it to the outputs. */
#define nk_define_reduce_moments_simt_(input_type, input_value_type, dimensions_per_value, accumulator, sum_type, \
                                       sumsq_type, isa_suffix, vendor, load_and_convert)                          \
    static __global__ void __launch_bounds__(nk_reduce_threads_simt_k)                                            \
        nk_reduce_moments_##input_type##_##isa_suffix##_kernel_(nk_reduce_arguments_t arguments) {                \
        __shared__ nk_reduce_##accumulator##_moments_t states[nk_reduce_threads_simt_k];                          \
        nk_reduce_##accumulator##_moments_t state = {0};                                                          \
        nk_size_t const lanes = nk_reduce_lanes_simt_();                                                          \
        nk_size_t const values = nk_size_divide_round_up_(arguments.count, dimensions_per_value);                 \
        for (nk_size_t first = nk_reduce_lane_simt_(); first < values; first += lanes * nk_reduce_batch_simt_k) { \
            nk_##input_value_type##_t batch[nk_reduce_batch_simt_k];                                              \
            _Pragma("unroll") for (unsigned slot = 0; slot != nk_reduce_batch_simt_k; ++slot) {                   \
                if (first + slot * lanes >= values) break;                                                        \
                unsigned char const *address = arguments.data + (first + slot * lanes) * arguments.stride;        \
                if (arguments.aligned) batch[slot] = *(nk_##input_value_type##_t const *)address;                 \
                else nk_reduce_copy_simt_(batch + slot, address, sizeof(*batch));                                 \
            }                                                                                                     \
            _Pragma("unroll") for (unsigned slot = 0; slot != nk_reduce_batch_simt_k; ++slot) {                   \
                nk_size_t const index = first + slot * lanes;                                                     \
                nk_##sum_type##_t dimensions[dimensions_per_value];                                               \
                if (index >= values) break;                                                                       \
                load_and_convert(batch + slot, dimensions);                                                       \
                _Pragma("unroll") for (unsigned dimension = 0; dimension != dimensions_per_value; ++dimension) {  \
                    if (index * dimensions_per_value + dimension < arguments.count)                               \
                        nk_reduce_##accumulator##_moments_add_simt_(&state, dimensions[dimension]);               \
                }                                                                                                 \
            }                                                                                                     \
        }                                                                                                         \
        states[threadIdx.x] = state;                                                                              \
        __syncthreads();                                                                                          \
        for (unsigned half = blockDim.x / 2; half; half >>= 1) {                                                  \
            if (threadIdx.x < half)                                                                               \
                nk_reduce_##accumulator##_moments_merge_simt_(&states[threadIdx.x], &states[threadIdx.x + half]); \
            __syncthreads();                                                                                      \
        }                                                                                                         \
        if (threadIdx.x) return;                                                                                  \
        nk_reduce_##accumulator##_moments_store_simt_(&states[0], &arguments);                                    \
    }                                                                                                             \
    NUMKONG_API nk_status_t nk_reduce_moments_##input_type##_##isa_suffix(                                        \
        nk_##input_value_type##_t const *data, nk_size_t count, nk_size_t stride, nk_##sum_type##_t *sum,         \
        nk_##sumsq_type##_t *sumsq, nk_stream_t stream) {                                                         \
        return nk_reduce_moments_launch_##vendor##_(                                                              \
            (void const *)&nk_reduce_moments_##input_type##_##isa_suffix##_kernel_,                               \
            nk_reduce_arguments_simt_(sizeof(nk_##input_value_type##_t), data, count, stride, sum, NUMKONG_NULL,  \
                                      sumsq, NUMKONG_NULL),                                                       \
            sizeof(nk_##sum_type##_t), sizeof(nk_##sumsq_type##_t), stream);                                      \
    }

/** Generates the entry point and the min/max and finishing kernels of @p input_type for
 *  @p isa_suffix, their sides starting from the serial kernels' @p min_sentinel and
 *  @p max_sentinel bits, launched through the helpers of @p vendor. */
#define nk_define_reduce_minmax_simt_(input_type, input_value_type, output_type, min_sentinel, max_sentinel,        \
                                      isa_suffix, vendor)                                                           \
    static __global__ void __launch_bounds__(nk_reduce_threads_simt_k)                                              \
        nk_reduce_minmax_##input_type##_##isa_suffix##_kernel_(nk_reduce_arguments_t arguments) {                   \
        nk_reduce_minmax_simt_(nk_##input_type##_k, &arguments);                                                    \
    }                                                                                                               \
    static __global__ void __launch_bounds__(nk_reduce_finish_threads_simt_k)                                       \
        nk_reduce_minmax_##input_type##_finish_##isa_suffix##_kernel_(nk_reduce_arguments_t arguments) {            \
        nk_reduce_minmax_finish_simt_(nk_##input_type##_k, min_sentinel, max_sentinel, &arguments);                 \
    }                                                                                                               \
    NUMKONG_API nk_status_t nk_reduce_minmax_##input_type##_##isa_suffix(                                           \
        nk_##input_value_type##_t const *data, nk_size_t count, nk_size_t stride, nk_##output_type##_t *min_value,  \
        nk_size_t *min_index, nk_##output_type##_t *max_value, nk_size_t *max_index, nk_stream_t stream) {          \
        return nk_reduce_minmax_launch_##vendor##_(                                                                 \
            (void const *)&nk_reduce_minmax_##input_type##_##isa_suffix##_kernel_,                                  \
            (void const *)&nk_reduce_minmax_##input_type##_finish_##isa_suffix##_kernel_,                           \
            nk_reduce_arguments_simt_(sizeof(nk_##input_value_type##_t), data, count, stride, min_value, min_index, \
                                      max_value, max_index),                                                        \
            stream);                                                                                                \
    }

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_
#endif // NUMKONG_REDUCE_SIMT_CUH
