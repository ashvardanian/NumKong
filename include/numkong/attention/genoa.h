/**
 *  @brief Ragged attention for AVX-512 BF16-capable Genoa generation CPUs.
 *  @file include/numkong/attention/genoa.h
 *  @author Ash Vardanian
 *  @date July 6, 2026
 *
 *  @sa include/numkong/attention.h
 *
 *  Backend for AVX512_BF16 machines without AMX: scores accumulate with `vdpbf16ps`
 *  straight from BF16 planes, doubling the per-instruction throughput over the widened-F32
 *  Skylake tier while halving the packed-KV footprint. The panel structure, online
 *  correction, and the base-2 softmax polynomial are shared with the rest of the family;
 *  the softmax and weighted-sum stages reuse the Skylake helpers (Genoa always implies the
 *  Skylake feature set, matching `dots/genoa.h`).
 *
 *  Packed payload per segment: K planes then V planes, `[key_value_head][position][channel]` in BF16
 *  with channels zero-padded to a multiple of 32 — `vdpbf16ps` consumes value pairs, so
 *  full-width loops need no masks. E4M3 widens to BF16 during packing and Q staging via
 *  the Ice Lake converters, exactly like `dots/genoa.h`. `depth > 256` routes to the
 *  width-agnostic serial tier from every entry point.
 */
#ifndef NK_ATTENTION_GENOA_H
#define NK_ATTENTION_GENOA_H

#if NK_TARGET_X8664_
#if NK_TARGET_GENOA

#include "numkong/attention/serial.h" // shared packed-KV header/offsets, width-agnostic fallback
#include "numkong/attention/skylake.h" // `nk_attention_exp2_f32x16_skylake_`; Genoa implies Skylake
#include "numkong/cast/icelake.h"      // `nk_load_e4m3x32_to_bf16x32_icelake_`
#include "numkong/cast/skylake.h"      // `nk_bf16x16_to_f32x16_skylake_`
#include "numkong/reduce/skylake.h"    // `nk_reduce_add_f32x16_skylake_`, `nk_reduce_max_f32x16_skylake_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(                                                                        \
    __attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512bf16,f16c,fma,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512bf16", "f16c", "fma", "bmi", "bmi2")
#endif

enum {
    /** KV panel width in positions; the F32 score row (2 KB) stays L1-resident. */
    nk_attention_panel_genoa_k_ = 512,
    /** Widest head this backend handles in registers; larger heads route to the serial tier. */
    nk_attention_max_depth_genoa_k_ = 256,
};

/** @brief Narrows or converts `count` elements to BF16 into `destination`, zero-filling to `padded`. */
typedef void (*nk_attention_narrow_genoa_t_)(void const *source, nk_bf16_t *destination, nk_size_t count,
                                             nk_size_t padded);

NK_HELPER_INLINE void nk_attention_narrow_bf16_genoa_(void const *source, nk_bf16_t *destination, nk_size_t count,
                                                      nk_size_t padded) {
    nk_size_t channel_idx = 0;
    for (; channel_idx + 32 <= count; channel_idx += 32)
        _mm512_storeu_si512(destination + channel_idx,
                            _mm512_loadu_si512((char const *)source + channel_idx * sizeof(nk_bf16_t)));
    if (channel_idx < count) {
        __mmask32 const tail_m32 = (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned int)(count - channel_idx));
        _mm512_storeu_si512(destination + channel_idx,
                            _mm512_maskz_loadu_epi16(tail_m32, (char const *)source + channel_idx * sizeof(nk_bf16_t)));
        channel_idx += 32;
    }
    for (; channel_idx < padded; channel_idx += 32)
        _mm512_storeu_si512(destination + channel_idx, _mm512_setzero_si512());
}

NK_HELPER_INLINE void nk_attention_narrow_e4m3_genoa_(void const *source, nk_bf16_t *destination, nk_size_t count,
                                                      nk_size_t padded) {
    nk_size_t channel_idx = 0;
    nk_b512_vec_t converted;
    for (; channel_idx + 32 <= count; channel_idx += 32) {
        nk_load_e4m3x32_to_bf16x32_icelake_((nk_e4m3_t const *)source + channel_idx, &converted);
        _mm512_storeu_si512(destination + channel_idx, converted.zmm);
    }
    if (channel_idx < count) {
        nk_partial_load_e4m3x32_to_bf16x32_icelake_((nk_e4m3_t const *)source + channel_idx, &converted,
                                                    count - channel_idx);
        _mm512_storeu_si512(destination + channel_idx, converted.zmm);
        channel_idx += 32;
    }
    for (; channel_idx < padded; channel_idx += 32)
        _mm512_storeu_si512(destination + channel_idx, _mm512_setzero_si512());
}

NK_HELPER_INLINE nk_size_t nk_attention_packed_size_genoa_(nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_u32_t const *segment_lengths, nk_size_t segment_count) {
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 32);
    nk_size_t payload_bytes = 0;
    for (nk_size_t segment_idx = 0; segment_idx < segment_count; segment_idx++)
        payload_bytes += 2 * key_value_head_count * (nk_size_t)segment_lengths[segment_idx] * depth_padded *
                         sizeof(nk_bf16_t); // K + V
    return sizeof(nk_attention_packed_header_t) + nk_attention_pack_directory_size_(segment_count) + payload_bytes;
}

NK_API_COMPTIME nk_size_t nk_attention_packed_size_bf16_genoa(nk_size_t key_value_head_count, nk_size_t depth,
                                                              nk_u32_t const *segment_lengths,
                                                              nk_size_t segment_count) {
    if (depth > nk_attention_max_depth_genoa_k_)
        return nk_attention_packed_size_bf16_serial(key_value_head_count, depth, segment_lengths, segment_count);
    return nk_attention_packed_size_genoa_(key_value_head_count, depth, segment_lengths, segment_count);
}

NK_API_COMPTIME nk_size_t nk_attention_packed_size_e4m3_genoa(nk_size_t key_value_head_count, nk_size_t depth,
                                                              nk_u32_t const *segment_lengths,
                                                              nk_size_t segment_count) {
    if (depth > nk_attention_max_depth_genoa_k_)
        return nk_attention_packed_size_e4m3_serial(key_value_head_count, depth, segment_lengths, segment_count);
    return nk_attention_packed_size_genoa_(key_value_head_count, depth, segment_lengths, segment_count);
}

NK_HELPER_INLINE void nk_attention_pack_genoa_(                                                                //
    void const *keys, void const *values, nk_size_t element_bytes,                                             //
    nk_attention_narrow_genoa_t_ narrow,                                                                       //
    nk_size_t key_value_head_count, nk_size_t depth,                                                           //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,                                          //
    nk_size_t segment_count, nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, void *key_value_packed, //
    nk_size_t first_task, nk_size_t task_count) {

    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 32);
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
        nk_size_t const plane_values = position_count * depth_padded;
        nk_bf16_t *keys_plane = (nk_bf16_t *)(payload_base + payload_offsets_ro[segment_idx]) +
                                key_value_head_idx * plane_values;
        nk_bf16_t *values_plane = keys_plane + key_value_head_count * plane_values;
        for (nk_size_t position_idx = 0; position_idx < position_count; position_idx++) {
            narrow((char const *)keys + (position_first + position_idx) * key_stride_bytes +
                       key_value_head_idx * depth * element_bytes,
                   keys_plane + position_idx * depth_padded, depth, depth_padded);
            narrow((char const *)values + (position_first + position_idx) * value_stride_bytes +
                       key_value_head_idx * depth * element_bytes,
                   values_plane + position_idx * depth_padded, depth, depth_padded);
        }
    }
}

NK_API_COMPTIME void nk_attention_pack_bf16_genoa(                                                   //
    nk_bf16_t const *keys, nk_bf16_t const *values, nk_size_t key_value_head_count, nk_size_t depth, //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, nk_size_t segment_count,
    nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, void *key_value_packed, nk_size_t first_task,
    nk_size_t task_count) {
    if (depth > nk_attention_max_depth_genoa_k_) {
        nk_attention_pack_bf16_serial(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                      segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, first_task,
                                      task_count);
        return;
    }
    nk_attention_pack_genoa_(keys, values, sizeof(nk_bf16_t), &nk_attention_narrow_bf16_genoa_, key_value_head_count,
                             depth, segment_offsets, segment_lengths, segment_count, key_stride_bytes,
                             value_stride_bytes, key_value_packed, first_task, task_count);
}

NK_API_COMPTIME void nk_attention_pack_e4m3_genoa(                                                   //
    nk_e4m3_t const *keys, nk_e4m3_t const *values, nk_size_t key_value_head_count, nk_size_t depth, //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, nk_size_t segment_count,
    nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, void *key_value_packed, nk_size_t first_task,
    nk_size_t task_count) {
    if (depth > nk_attention_max_depth_genoa_k_) {
        nk_attention_pack_e4m3_serial(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                      segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, first_task,
                                      task_count);
        return;
    }
    nk_attention_pack_genoa_(keys, values, sizeof(nk_e4m3_t), &nk_attention_narrow_e4m3_genoa_, key_value_head_count,
                             depth, segment_offsets, segment_lengths, segment_count, key_stride_bytes,
                             value_stride_bytes, key_value_packed, first_task, task_count);
}

/**
 *  @brief Shared attention core over BF16 planes: per query row, panel-flash with an exact
 *         online correction; scores use `vdpbf16ps` with four KV rows in flight.
 */
NK_HELPER_INLINE void nk_attention_packed_genoa_(                                                               //
    void const *queries, nk_size_t element_bytes, nk_attention_narrow_genoa_t_ narrow,                          //
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
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 32);
    nk_f32_t const scale2 = scale * NK_F32_LOG2E_; // softmax(x) = softmax₂(x·log₂e)
    nk_size_t const panel_width = nk_attention_panel_genoa_k_;

    nk_size_t const total_tasks = segment_count * head_count;
    if (first_task >= total_tasks) return;
    if (task_count == 0 || first_task + task_count > total_tasks) task_count = total_tasks - first_task;

    NK_ALIGN64 nk_bf16_t query_row[nk_attention_max_depth_genoa_k_];
    NK_ALIGN64 nk_f32_t output_row[nk_attention_max_depth_genoa_k_];
    NK_ALIGN64 nk_f32_t scores[nk_attention_panel_genoa_k_];
    nk_size_t const depth_full = depth & ~(nk_size_t)15;
    __mmask16 const depth_tail_m16 = (__mmask16)((1u << (depth - depth_full)) - 1);

    for (nk_size_t task_idx = first_task; task_idx < first_task + task_count; task_idx++) {
        nk_size_t const segment_idx = task_idx / head_count, head_idx = task_idx % head_count;
        nk_size_t const position_count = segment_lengths[segment_idx];
        nk_size_t const row_count = query_offsets[segment_idx + 1] - query_offsets[segment_idx];
        if (position_count == 0 || row_count == 0) continue;
        nk_size_t const plane_values = position_count * depth_padded;
        nk_bf16_t const *keys_plane = (nk_bf16_t const *)(payload_base + payload_offsets[segment_idx]) +
                                      (head_idx / head_group_size) * plane_values;
        nk_bf16_t const *values_plane = keys_plane + key_value_head_count * plane_values;

        for (nk_size_t row_idx = 0; row_idx < row_count; row_idx++) {
            narrow((char const *)queries + (query_offsets[segment_idx] + row_idx) * query_stride_bytes +
                       head_idx * depth * element_bytes,
                   query_row, depth, depth_padded);
            for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 16)
                _mm512_store_ps(output_row + channel_idx, _mm512_setzero_ps());
            nk_f32_t running_max2 = NK_F32_MIN, running_sum = 0;

            for (nk_size_t panel_start = 0; panel_start < position_count; panel_start += panel_width) {
                nk_size_t const panel_length = (panel_start + panel_width <= position_count)
                                                   ? panel_width
                                                   : (position_count - panel_start);
                // Scores: `vdpbf16ps` accumulation, four KV rows in flight.

                nk_size_t position_idx = 0;
                for (; position_idx + 4 <= panel_length; position_idx += 4) {
                    nk_bf16_t const *keys_row = keys_plane + (panel_start + position_idx) * depth_padded;
                    __m512 accumulator0_f32x16 = _mm512_setzero_ps(), accumulator1_f32x16 = _mm512_setzero_ps();
                    __m512 accumulator2_f32x16 = _mm512_setzero_ps(), accumulator3_f32x16 = _mm512_setzero_ps();
                    for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 32) {
                        __m512bh const query_bf16x32 = (__m512bh)_mm512_load_si512(query_row + channel_idx);
                        accumulator0_f32x16 = _mm512_dpbf16_ps(accumulator0_f32x16, query_bf16x32,
                                                               (__m512bh)_mm512_loadu_si512(keys_row + channel_idx));
                        accumulator1_f32x16 = _mm512_dpbf16_ps(
                            accumulator1_f32x16, query_bf16x32,
                            (__m512bh)_mm512_loadu_si512(keys_row + depth_padded + channel_idx));
                        accumulator2_f32x16 = _mm512_dpbf16_ps(
                            accumulator2_f32x16, query_bf16x32,
                            (__m512bh)_mm512_loadu_si512(keys_row + 2 * depth_padded + channel_idx));
                        accumulator3_f32x16 = _mm512_dpbf16_ps(
                            accumulator3_f32x16, query_bf16x32,
                            (__m512bh)_mm512_loadu_si512(keys_row + 3 * depth_padded + channel_idx));
                    }
                    scores[position_idx + 0] = nk_reduce_add_f32x16_skylake_(accumulator0_f32x16);
                    scores[position_idx + 1] = nk_reduce_add_f32x16_skylake_(accumulator1_f32x16);
                    scores[position_idx + 2] = nk_reduce_add_f32x16_skylake_(accumulator2_f32x16);
                    scores[position_idx + 3] = nk_reduce_add_f32x16_skylake_(accumulator3_f32x16);
                }
                for (; position_idx < panel_length; position_idx++) {
                    nk_bf16_t const *keys_row = keys_plane + (panel_start + position_idx) * depth_padded;
                    __m512 accumulator_f32x16 = _mm512_setzero_ps();
                    for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 32)
                        accumulator_f32x16 = _mm512_dpbf16_ps(accumulator_f32x16,
                                                              (__m512bh)_mm512_load_si512(query_row + channel_idx),
                                                              (__m512bh)_mm512_loadu_si512(keys_row + channel_idx));
                    scores[position_idx] = nk_reduce_add_f32x16_skylake_(accumulator_f32x16);
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
                    nk_bf16_t const *values_row = values_plane + (panel_start + position_idx) * depth_padded;
                    for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 16) {
                        __m512 const v_f32x16 = nk_bf16x16_to_f32x16_skylake_(
                            _mm256_loadu_si256((__m256i const *)(values_row + channel_idx)));
                        _mm512_store_ps(
                            output_row + channel_idx,
                            _mm512_fmadd_ps(weight_f32x16, v_f32x16, _mm512_load_ps(output_row + channel_idx)));
                    }
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
                _mm512_mask_storeu_ps(destination + channel_idx, depth_tail_m16,
                                      _mm512_mul_ps(_mm512_load_ps(output_row + channel_idx), inverse_sum_f32x16));
        }
    }
}

NK_API_COMPTIME void nk_attention_packed_bf16_genoa(                             //
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output,    //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,       //
    nk_u32_t const *query_offsets,                                               //
    nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_size_t first_task, nk_size_t task_count) {
    if (depth > nk_attention_max_depth_genoa_k_) {
        nk_attention_packed_bf16_serial(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                        query_offsets, query_stride_bytes, output_stride_bytes, scale, first_task,
                                        task_count);
        return;
    }
    nk_attention_packed_genoa_(queries, sizeof(nk_bf16_t), &nk_attention_narrow_bf16_genoa_, key_value_packed, output,
                               head_count, key_value_head_count, depth, query_offsets, query_stride_bytes,
                               output_stride_bytes, scale, first_task, task_count);
}

NK_API_COMPTIME void nk_attention_packed_e4m3_genoa(                             //
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output,    //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,       //
    nk_u32_t const *query_offsets,                                               //
    nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_size_t first_task, nk_size_t task_count) {
    if (depth > nk_attention_max_depth_genoa_k_) {
        nk_attention_packed_e4m3_serial(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                        query_offsets, query_stride_bytes, output_stride_bytes, scale, first_task,
                                        task_count);
        return;
    }
    nk_attention_packed_genoa_(queries, sizeof(nk_e4m3_t), &nk_attention_narrow_e4m3_genoa_, key_value_packed, output,
                               head_count, key_value_head_count, depth, query_offsets, query_stride_bytes,
                               output_stride_bytes, scale, first_task, task_count);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_GENOA
#endif // NK_TARGET_X8664_
#endif // NK_ATTENTION_GENOA_H
