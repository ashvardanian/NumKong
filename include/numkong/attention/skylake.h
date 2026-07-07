/**
 *  @brief Ragged attention for AVX-512 Skylake-X generation CPUs.
 *  @file include/numkong/attention/skylake.h
 *  @author Ash Vardanian
 *  @date July 6, 2026
 *
 *  @sa include/numkong/attention.h
 *
 *  Compatibility backend for AVX-512F machines without BF16 or AMX ISA extensions.
 *  Storage follows the `dots/skylake.h` conventions exactly: BF16 inputs stay BF16 at
 *  rest and widen to F32 inside the compute loops (shift-based, two ops per 16 lanes);
 *  E4M3 converts once to F16 during packing, so the in-loop widening is a single
 *  hardware `VCVTPH2PS` — the same asymmetry the GEMM family chose, trading one cheap
 *  pack-time pass for halved KV streaming traffic against F32 planes.
 *
 *  The panel structure matches the family design — per query row, KV is swept in panels
 *  with an exact online correction, a base-2 streaming softmax sharing the family's
 *  degree-4 polynomial, and a score core with four KV rows in flight on the dual FMA
 *  ports. Packed payload per segment: K planes then V planes, `[key_value_head][position][channel]`
 *  in 16-bit scalars with channels zero-padded to a multiple of 16. `depth > 256`
 *  routes to the width-agnostic serial tier from every entry point.
 */
#ifndef NK_ATTENTION_SKYLAKE_H
#define NK_ATTENTION_SKYLAKE_H

#if NK_TARGET_X8664_
#if NK_TARGET_SKYLAKE

#include "numkong/attention/serial.h" // shared packed-KV header/offsets, width-agnostic fallback
#include "numkong/cast/skylake.h"     // widening loaders like `nk_load_bf16x16_to_f32x16_skylake_`
#include "numkong/reduce/skylake.h"   // `nk_reduce_add_f32x16_skylake_`, `nk_reduce_max_f32x16_skylake_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,f16c,fma,bmi,bmi2"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "f16c", "fma", "bmi", "bmi2")
#endif

enum {
    /** KV panel width in positions; the F32 score row (2 KB) stays L1-resident. */
    nk_attention_panel_skylake_k_ = 512,
    /** Widest head this backend handles in registers; larger heads route to the serial tier. */
    nk_attention_max_depth_skylake_k_ = 256,
};

/** @brief Fast vectorized 2^x: exact range reduction + the family's shared degree-4 polynomial. */
NK_INTERNAL __m512 nk_attention_exp2_f32x16_skylake_(__m512 x_f32x16) {
    x_f32x16 = _mm512_max_ps(_mm512_min_ps(x_f32x16, _mm512_set1_ps(127.0f)), _mm512_set1_ps(-125.0f));
    __m512 n_f32x16 = _mm512_roundscale_ps(x_f32x16, _MM_FROUND_TO_NEAREST_INT);
    __m512 r_f32x16 = _mm512_sub_ps(x_f32x16, n_f32x16);
    __m512 p_f32x16 = _mm512_set1_ps(9.61812910e-3f);
    p_f32x16 = _mm512_fmadd_ps(p_f32x16, r_f32x16, _mm512_set1_ps(5.55041087e-2f));
    p_f32x16 = _mm512_fmadd_ps(p_f32x16, r_f32x16, _mm512_set1_ps(2.40226507e-1f));
    p_f32x16 = _mm512_fmadd_ps(p_f32x16, r_f32x16, _mm512_set1_ps(6.93147181e-1f));
    p_f32x16 = _mm512_fmadd_ps(p_f32x16, r_f32x16, _mm512_set1_ps(1.0f));
    __m512i n_i32x16 = _mm512_cvtps_epi32(n_f32x16);
    n_i32x16 = _mm512_slli_epi32(_mm512_add_epi32(n_i32x16, _mm512_set1_epi32(127)), 23);
    return _mm512_mul_ps(p_f32x16, _mm512_castsi512_ps(n_i32x16));
}

/**
 *  @brief One panel of the streaming base-2 softmax: merges the panel maximum into the
 *         running one, exponentiates the scores in place into weights, and folds the panel
 *         weight-sum into the running sum. Returns the correction `2^(m_old − m_new)` the
 *         caller applies to its output accumulators.
 */
NK_INTERNAL nk_f32_t nk_attention_softmax_panel_skylake_(nk_f32_t *scores, nk_size_t panel_length, nk_f32_t scale2,
                                                         nk_f32_t *running_max2, nk_f32_t *running_sum) {
    __m512 max_f32x16 = _mm512_set1_ps(NK_F32_MIN);
    nk_size_t position_idx = 0;
    for (; position_idx + 16 <= panel_length; position_idx += 16)
        max_f32x16 = _mm512_max_ps(max_f32x16, _mm512_loadu_ps(scores + position_idx));
    nk_f32_t panel_max2 = nk_reduce_max_f32x16_skylake_(max_f32x16) * scale2;
    for (; position_idx < panel_length; position_idx++) {
        nk_f32_t const scaled2 = scores[position_idx] * scale2;
        if (scaled2 > panel_max2) panel_max2 = scaled2;
    }
    nk_f32_t const new_max2 = *running_max2 > panel_max2 ? *running_max2 : panel_max2;
    nk_f32_t const correction = _mm512_cvtss_f32(
        nk_attention_exp2_f32x16_skylake_(_mm512_set1_ps(*running_max2 - new_max2)));
    *running_max2 = new_max2;

    __m512 const scale2_f32x16 = _mm512_set1_ps(scale2);
    __m512 const max2_f32x16 = _mm512_set1_ps(new_max2);
    __m512 sum_f32x16 = _mm512_setzero_ps();
    nk_size_t const panel_full = panel_length & ~(nk_size_t)15;
    __mmask16 const panel_tail_mask = (__mmask16)((1u << (panel_length - panel_full)) - 1);
    for (position_idx = 0; position_idx < panel_full; position_idx += 16) {
        __m512 weights_f32x16 = nk_attention_exp2_f32x16_skylake_(
            _mm512_fmsub_ps(_mm512_loadu_ps(scores + position_idx), scale2_f32x16, max2_f32x16));
        sum_f32x16 = _mm512_add_ps(sum_f32x16, weights_f32x16);
        _mm512_storeu_ps(scores + position_idx, weights_f32x16);
    }
    if (position_idx < panel_length) {
        __m512 weights_f32x16 = _mm512_maskz_mov_ps(
            panel_tail_mask, nk_attention_exp2_f32x16_skylake_(
                                 _mm512_fmsub_ps(_mm512_loadu_ps(scores + position_idx), scale2_f32x16, max2_f32x16)));
        sum_f32x16 = _mm512_add_ps(sum_f32x16, weights_f32x16);
        _mm512_storeu_ps(scores + position_idx, weights_f32x16);
    }
    *running_sum = *running_sum * correction + nk_reduce_add_f32x16_skylake_(sum_f32x16);
    return correction;
}

/** @brief Widens 16 packed-plane scalars (BF16 or F16 at rest) to F32 inside the hot loops. */
typedef __m512 (*nk_attention_load_skylake_t_)(void const *plane_chunk);

NK_INTERNAL __m512 nk_attention_load_bf16x16_skylake_(void const *plane_chunk) {
    nk_b512_vec_t widened;
    nk_load_bf16x16_to_f32x16_skylake_(plane_chunk, &widened);
    return widened.zmm_ps;
}

NK_INTERNAL __m512 nk_attention_load_f16x16_skylake_(void const *plane_chunk) {
    nk_b512_vec_t widened;
    nk_load_f16x16_to_f32x16_skylake_(plane_chunk, &widened);
    return widened.zmm_ps;
}

/** @brief Narrows `count` input elements into 16-bit plane scalars, zero-filling to `padded`. */
typedef void (*nk_attention_narrow_skylake_t_)(void const *source, void *destination, nk_size_t count,
                                               nk_size_t padded);

NK_INTERNAL void nk_attention_narrow_bf16_skylake_(void const *source, void *destination, nk_size_t count,
                                                   nk_size_t padded) {
    nk_size_t channel_idx = 0; // BF16 stays BF16 at rest, like the dots family
    for (; channel_idx + 16 <= count; channel_idx += 16)
        _mm256_storeu_si256((__m256i *)((nk_bf16_t *)destination + channel_idx),
                            _mm256_loadu_si256((__m256i const *)((nk_bf16_t const *)source + channel_idx)));
    if (channel_idx < count) {
        __mmask16 const tail_mask = (__mmask16)((1u << (count - channel_idx)) - 1);
        _mm256_storeu_si256((__m256i *)((nk_bf16_t *)destination + channel_idx),
                            _mm256_maskz_loadu_epi16(tail_mask, (nk_bf16_t const *)source + channel_idx));
        channel_idx += 16;
    }
    for (; channel_idx < padded; channel_idx += 16)
        _mm256_storeu_si256((__m256i *)((nk_bf16_t *)destination + channel_idx), _mm256_setzero_si256());
}

NK_INTERNAL void nk_attention_narrow_e4m3_skylake_(void const *source, void *destination, nk_size_t count,
                                                   nk_size_t padded) {
    nk_size_t channel_idx = 0; // E4M3 converts once to F16, so the hot loops widen with one VCVTPH2PS
    nk_b256_vec_t converted;
    for (; channel_idx + 16 <= count; channel_idx += 16) {
        nk_load_e4m3x16_to_f16x16_skylake_((nk_e4m3_t const *)source + channel_idx, &converted);
        _mm256_storeu_si256((__m256i *)((nk_f16_t *)destination + channel_idx), converted.ymm);
    }
    if (channel_idx < count) {
        nk_partial_load_e4m3x16_to_f16x16_skylake_((nk_e4m3_t const *)source + channel_idx, &converted,
                                                   count - channel_idx);
        _mm256_storeu_si256((__m256i *)((nk_f16_t *)destination + channel_idx), converted.ymm);
        channel_idx += 16;
    }
    for (; channel_idx < padded; channel_idx += 16)
        _mm256_storeu_si256((__m256i *)((nk_f16_t *)destination + channel_idx), _mm256_setzero_si256());
}

/** @brief Widens `count` raw query elements to F32 into `destination`, zero-filling to `padded`. */
typedef void (*nk_attention_widen_skylake_t_)(void const *source, nk_f32_t *destination, nk_size_t count,
                                              nk_size_t padded);

NK_INTERNAL void nk_attention_widen_bf16_skylake_(void const *source, nk_f32_t *destination, nk_size_t count,
                                                  nk_size_t padded) {
    nk_size_t channel_idx = 0;
    nk_b512_vec_t widened;
    for (; channel_idx + 16 <= count; channel_idx += 16) {
        nk_load_bf16x16_to_f32x16_skylake_((nk_bf16_t const *)source + channel_idx, &widened);
        _mm512_storeu_ps(destination + channel_idx, widened.zmm_ps);
    }
    if (channel_idx < count) {
        nk_partial_load_bf16x16_to_f32x16_skylake_((nk_bf16_t const *)source + channel_idx, &widened,
                                                   count - channel_idx);
        _mm512_storeu_ps(destination + channel_idx, widened.zmm_ps);
        channel_idx += 16;
    }
    for (; channel_idx < padded; channel_idx += 16) _mm512_storeu_ps(destination + channel_idx, _mm512_setzero_ps());
}

NK_INTERNAL void nk_attention_widen_e4m3_skylake_(void const *source, nk_f32_t *destination, nk_size_t count,
                                                  nk_size_t padded) {
    nk_size_t channel_idx = 0;
    nk_b512_vec_t widened;
    for (; channel_idx + 16 <= count; channel_idx += 16) {
        nk_load_e4m3x16_to_f32x16_skylake_((nk_e4m3_t const *)source + channel_idx, &widened);
        _mm512_storeu_ps(destination + channel_idx, widened.zmm_ps);
    }
    if (channel_idx < count) {
        nk_partial_load_e4m3x16_to_f32x16_skylake_((nk_e4m3_t const *)source + channel_idx, &widened,
                                                   count - channel_idx);
        _mm512_storeu_ps(destination + channel_idx, widened.zmm_ps);
        channel_idx += 16;
    }
    for (; channel_idx < padded; channel_idx += 16) _mm512_storeu_ps(destination + channel_idx, _mm512_setzero_ps());
}

NK_INTERNAL nk_size_t nk_attention_packed_size_skylake_(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_u32_t const *segment_lengths, nk_size_t segment_count) {
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 16);
    nk_size_t payload_bytes = 0; // planes hold 16-bit scalars for both dtypes (BF16 or F16)
    for (nk_size_t segment_idx = 0; segment_idx < segment_count; segment_idx++)
        payload_bytes += 2 * key_value_head_count * (nk_size_t)segment_lengths[segment_idx] * depth_padded *
                         sizeof(nk_bf16_t);
    return sizeof(nk_attention_packed_header_t) + nk_attention_pack_directory_size_(segment_count) + payload_bytes;
}

NK_PUBLIC nk_size_t nk_attention_packed_size_bf16_skylake(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_u32_t const *segment_lengths, nk_size_t segment_count) {
    if (depth > nk_attention_max_depth_skylake_k_)
        return nk_attention_packed_size_bf16_serial(key_value_head_count, depth, segment_lengths, segment_count);
    return nk_attention_packed_size_skylake_(key_value_head_count, depth, segment_lengths, segment_count);
}

NK_PUBLIC nk_size_t nk_attention_packed_size_e4m3_skylake(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_u32_t const *segment_lengths, nk_size_t segment_count) {
    if (depth > nk_attention_max_depth_skylake_k_)
        return nk_attention_packed_size_e4m3_serial(key_value_head_count, depth, segment_lengths, segment_count);
    return nk_attention_packed_size_skylake_(key_value_head_count, depth, segment_lengths, segment_count);
}

NK_INTERNAL void nk_attention_pack_skylake_(                                                                   //
    void const *keys, void const *values, nk_size_t element_bytes,                                             //
    nk_attention_narrow_skylake_t_ narrow,                                                                     //
    nk_size_t key_value_head_count, nk_size_t depth,                                                           //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,                                          //
    nk_size_t segment_count, nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, void *key_value_packed, //
    nk_size_t first_task, nk_size_t task_count) {

    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 16);
    nk_attention_pack_directory_(key_value_packed, key_value_head_count, depth, segment_lengths, segment_count,
                                 first_task, 1, depth_padded * sizeof(nk_bf16_t));
    nk_attention_packed_header_t *header = (nk_attention_packed_header_t *)key_value_packed;
    nk_u64_t const *payload_offsets_ro = (nk_u64_t const *)((char *)key_value_packed + sizeof(*header));
    char *payload_base = (char *)key_value_packed + sizeof(*header) + nk_attention_pack_directory_size_(segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    if (first_task >= total_tasks) return;
    if (task_count == 0 || first_task + task_count > total_tasks) task_count = total_tasks - first_task;

    for (nk_size_t task_idx = first_task; task_idx < first_task + task_count; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count,
                        key_value_head_idx = task_idx % key_value_head_count;
        nk_size_t const position_count = segment_lengths[segment_idx];
        if (position_count == 0) continue;
        nk_size_t const position_first = segment_offsets[segment_idx];
        nk_size_t const plane_bytes = position_count * depth_padded * sizeof(nk_bf16_t);
        char *keys_plane = payload_base + payload_offsets_ro[segment_idx] + key_value_head_idx * plane_bytes;
        char *values_plane = keys_plane + key_value_head_count * plane_bytes;
        for (nk_size_t position_idx = 0; position_idx < position_count; position_idx++) {
            narrow((char const *)keys + (position_first + position_idx) * key_stride_bytes +
                       key_value_head_idx * depth * element_bytes,
                   keys_plane + position_idx * depth_padded * sizeof(nk_bf16_t), depth, depth_padded);
            narrow((char const *)values + (position_first + position_idx) * value_stride_bytes +
                       key_value_head_idx * depth * element_bytes,
                   values_plane + position_idx * depth_padded * sizeof(nk_bf16_t), depth, depth_padded);
        }
    }
}

NK_PUBLIC void nk_attention_pack_bf16_skylake(                                                       //
    nk_bf16_t const *keys, nk_bf16_t const *values, nk_size_t key_value_head_count, nk_size_t depth, //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, nk_size_t segment_count,
    nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, void *key_value_packed, nk_size_t first_task,
    nk_size_t task_count) {
    if (depth > nk_attention_max_depth_skylake_k_) {
        nk_attention_pack_bf16_serial(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                      segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, first_task,
                                      task_count);
        return;
    }
    nk_attention_pack_skylake_(keys, values, sizeof(nk_bf16_t), &nk_attention_narrow_bf16_skylake_,
                               key_value_head_count, depth, segment_offsets, segment_lengths, segment_count,
                               key_stride_bytes, value_stride_bytes, key_value_packed, first_task, task_count);
}

NK_PUBLIC void nk_attention_pack_e4m3_skylake(                                                       //
    nk_e4m3_t const *keys, nk_e4m3_t const *values, nk_size_t key_value_head_count, nk_size_t depth, //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, nk_size_t segment_count,
    nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, void *key_value_packed, nk_size_t first_task,
    nk_size_t task_count) {
    if (depth > nk_attention_max_depth_skylake_k_) {
        nk_attention_pack_e4m3_serial(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                      segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, first_task,
                                      task_count);
        return;
    }
    nk_attention_pack_skylake_(keys, values, sizeof(nk_e4m3_t), &nk_attention_narrow_e4m3_skylake_,
                               key_value_head_count, depth, segment_offsets, segment_lengths, segment_count,
                               key_stride_bytes, value_stride_bytes, key_value_packed, first_task, task_count);
}

/**
 *  @brief Shared attention core over 16-bit planes: per query row, panel-flash with an
 *         exact online correction; scores keep four KV rows in flight, widening in-loop.
 */
NK_INTERNAL void nk_attention_packed_skylake_(                                                                  //
    void const *queries, nk_size_t element_bytes, nk_attention_widen_skylake_t_ widen,                          //
    nk_attention_load_skylake_t_ load,                                                                          //
    void const *key_value_packed, nk_f32_t *output,                                                             //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,                                      //
    nk_u32_t const *query_offsets, nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_size_t first_task, nk_size_t task_count) {

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    if (header->depth != depth || header->key_value_head_count != key_value_head_count) return;
    nk_size_t const segment_count = header->segment_count;
    nk_u64_t const *payload_offsets = (nk_u64_t const *)((char const *)key_value_packed + sizeof(*header));
    nk_u32_t const *segment_lengths = (nk_u32_t const *)(payload_offsets + segment_count + 1);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_(segment_count);
    nk_size_t const output_stride_floats = output_stride_bytes / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 16);
    nk_size_t const plane_row_bytes = depth_padded * sizeof(nk_bf16_t);
    nk_f32_t const scale2 = scale * NK_F32_LOG2E_; // softmax(x) = softmax₂(x·log₂e)
    nk_size_t const panel_width = nk_attention_panel_skylake_k_;

    nk_size_t const total_tasks = segment_count * head_count;
    if (first_task >= total_tasks) return;
    if (task_count == 0 || first_task + task_count > total_tasks) task_count = total_tasks - first_task;

    NK_ALIGN64 nk_f32_t query_row[nk_attention_max_depth_skylake_k_];
    NK_ALIGN64 nk_f32_t output_row[nk_attention_max_depth_skylake_k_];
    NK_ALIGN64 nk_f32_t scores[nk_attention_panel_skylake_k_];
    nk_size_t const depth_full = depth & ~(nk_size_t)15;
    __mmask16 const depth_tail_mask = (__mmask16)((1u << (depth - depth_full)) - 1);

    for (nk_size_t task_idx = first_task; task_idx < first_task + task_count; task_idx++) {
        nk_size_t const segment_idx = task_idx / head_count, head_idx = task_idx % head_count;
        nk_size_t const position_count = segment_lengths[segment_idx];
        nk_size_t const row_count = query_offsets[segment_idx + 1] - query_offsets[segment_idx];
        if (position_count == 0 || row_count == 0) continue;
        nk_size_t const plane_bytes = position_count * plane_row_bytes;
        char const *keys_plane = payload_base + payload_offsets[segment_idx] +
                                 (head_idx / head_group_size) * plane_bytes;
        char const *values_plane = keys_plane + key_value_head_count * plane_bytes;

        for (nk_size_t row_idx = 0; row_idx < row_count; row_idx++) {
            widen((char const *)queries + (query_offsets[segment_idx] + row_idx) * query_stride_bytes +
                      head_idx * depth * element_bytes,
                  query_row, depth, depth_padded);
            for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 16)
                _mm512_store_ps(output_row + channel_idx, _mm512_setzero_ps());
            nk_f32_t running_max2 = NK_F32_MIN, running_sum = 0;

            for (nk_size_t panel_start = 0; panel_start < position_count; panel_start += panel_width) {
                nk_size_t const panel_length = (panel_start + panel_width <= position_count)
                                                   ? panel_width
                                                   : (position_count - panel_start);
                // Scores: four KV rows in flight, plane scalars widened in-loop.

                nk_size_t position_idx = 0;
                for (; position_idx + 4 <= panel_length; position_idx += 4) {
                    char const *keys_row = keys_plane + (panel_start + position_idx) * plane_row_bytes;
                    __m512 acc0_f32x16 = _mm512_setzero_ps(), acc1_f32x16 = _mm512_setzero_ps();
                    __m512 acc2_f32x16 = _mm512_setzero_ps(), acc3_f32x16 = _mm512_setzero_ps();
                    for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 16) {
                        __m512 const query_f32x16 = _mm512_load_ps(query_row + channel_idx);
                        nk_size_t const chunk_bytes = channel_idx * sizeof(nk_bf16_t);
                        acc0_f32x16 = _mm512_fmadd_ps(query_f32x16, load(keys_row + chunk_bytes), acc0_f32x16);
                        acc1_f32x16 = _mm512_fmadd_ps(query_f32x16, load(keys_row + plane_row_bytes + chunk_bytes),
                                                      acc1_f32x16);
                        acc2_f32x16 = _mm512_fmadd_ps(query_f32x16, load(keys_row + 2 * plane_row_bytes + chunk_bytes),
                                                      acc2_f32x16);
                        acc3_f32x16 = _mm512_fmadd_ps(query_f32x16, load(keys_row + 3 * plane_row_bytes + chunk_bytes),
                                                      acc3_f32x16);
                    }
                    scores[position_idx + 0] = nk_reduce_add_f32x16_skylake_(acc0_f32x16);
                    scores[position_idx + 1] = nk_reduce_add_f32x16_skylake_(acc1_f32x16);
                    scores[position_idx + 2] = nk_reduce_add_f32x16_skylake_(acc2_f32x16);
                    scores[position_idx + 3] = nk_reduce_add_f32x16_skylake_(acc3_f32x16);
                }
                for (; position_idx < panel_length; position_idx++) {
                    char const *keys_row = keys_plane + (panel_start + position_idx) * plane_row_bytes;
                    __m512 acc_f32x16 = _mm512_setzero_ps();
                    for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 16)
                        acc_f32x16 = _mm512_fmadd_ps(_mm512_load_ps(query_row + channel_idx),
                                                     load(keys_row + channel_idx * sizeof(nk_bf16_t)), acc_f32x16);
                    scores[position_idx] = nk_reduce_add_f32x16_skylake_(acc_f32x16);
                }

                nk_f32_t const correction = nk_attention_softmax_panel_skylake_(scores, panel_length, scale2,
                                                                                &running_max2, &running_sum);

                // O = O · correction + Σ weight · widened V-row over the panel.

                __m512 const correction_f32x16 = _mm512_set1_ps(correction);
                for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 16)
                    _mm512_store_ps(output_row + channel_idx,
                                    _mm512_mul_ps(_mm512_load_ps(output_row + channel_idx), correction_f32x16));
                for (position_idx = 0; position_idx < panel_length; position_idx++) {
                    __m512 const weight_f32x16 = _mm512_set1_ps(scores[position_idx]);
                    char const *values_row = values_plane + (panel_start + position_idx) * plane_row_bytes;
                    for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 16)
                        _mm512_store_ps(
                            output_row + channel_idx,
                            _mm512_fmadd_ps(weight_f32x16, load(values_row + channel_idx * sizeof(nk_bf16_t)),
                                            _mm512_load_ps(output_row + channel_idx)));
                }
            }

            __m512 const inverse_sum_f32x16 = _mm512_set1_ps(1.0f / running_sum);
            nk_f32_t *destination = output + (query_offsets[segment_idx] + row_idx) * output_stride_floats +
                                    head_idx * depth;
            nk_size_t channel_idx = 0;
            for (; channel_idx < depth_full; channel_idx += 16)
                _mm512_storeu_ps(destination + channel_idx,
                                 _mm512_mul_ps(_mm512_load_ps(output_row + channel_idx), inverse_sum_f32x16));
            if (channel_idx < depth)
                _mm512_mask_storeu_ps(destination + channel_idx, depth_tail_mask,
                                      _mm512_mul_ps(_mm512_load_ps(output_row + channel_idx), inverse_sum_f32x16));
        }
    }
}

NK_PUBLIC void nk_attention_packed_bf16_skylake(                                 //
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output,    //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,       //
    nk_u32_t const *query_offsets,                                               //
    nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_size_t first_task, nk_size_t task_count) {
    if (depth > nk_attention_max_depth_skylake_k_) {
        nk_attention_packed_bf16_serial(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                        query_offsets, query_stride_bytes, output_stride_bytes, scale, first_task,
                                        task_count);
        return;
    }
    nk_attention_packed_skylake_(queries, sizeof(nk_bf16_t), &nk_attention_widen_bf16_skylake_,
                                 &nk_attention_load_bf16x16_skylake_, key_value_packed, output, head_count,
                                 key_value_head_count, depth, query_offsets, query_stride_bytes, output_stride_bytes,
                                 scale, first_task, task_count);
}

NK_PUBLIC void nk_attention_packed_e4m3_skylake(                                 //
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output,    //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,       //
    nk_u32_t const *query_offsets,                                               //
    nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_size_t first_task, nk_size_t task_count) {
    if (depth > nk_attention_max_depth_skylake_k_) {
        nk_attention_packed_e4m3_serial(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                        query_offsets, query_stride_bytes, output_stride_bytes, scale, first_task,
                                        task_count);
        return;
    }
    nk_attention_packed_skylake_(queries, sizeof(nk_e4m3_t), &nk_attention_widen_e4m3_skylake_,
                                 &nk_attention_load_f16x16_skylake_, key_value_packed, output, head_count,
                                 key_value_head_count, depth, query_offsets, query_stride_bytes, output_stride_bytes,
                                 scale, first_task, task_count);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SKYLAKE
#endif // NK_TARGET_X8664_
#endif // NK_ATTENTION_SKYLAKE_H
