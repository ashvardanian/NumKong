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
 *  Skylake tier while halving the packed-KV footprint.  The panel structure, online
 *  correction, and the base-2 softmax polynomial are shared with the rest of the family;
 *  the softmax and weighted-sum stages reuse the Skylake helpers (Genoa always implies the
 *  Skylake feature set, matching `dots/genoa.h`).
 *
 *  Packed payload per segment: K planes then V planes, `[kv_head][token][channel]` in BF16
 *  with channels zero-padded to a multiple of 32 — `vdpbf16ps` consumes value pairs, so
 *  full-width loops need no masks.  E4M3 widens to BF16 during packing and Q staging via
 *  the Ice Lake converters, exactly like `dots/genoa.h`.  `head_dim > 256` routes to the
 *  width-agnostic serial tier from every entry point.
 */
#ifndef NK_ATTENTION_GENOA_H
#define NK_ATTENTION_GENOA_H

#if NK_TARGET_X8664_
#if NK_TARGET_GENOA

#include "numkong/attention/serial.h"  // shared packed-KV header/directory, width-agnostic fallback
#include "numkong/attention/skylake.h" // `nk_attention_exp2_ps_skylake_`; Genoa implies Skylake
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
    /// KV panel width in positions; the F32 score row (2 KB) stays L1-resident.
    nk_attention_panel_genoa_k_ = 512,
    /// Widest head this backend handles in registers; larger heads route to the serial tier.
    nk_attention_max_head_dim_genoa_k_ = 256,
};

/** @brief Narrows or converts `count` elements to BF16 into `destination`, zero-filling to `padded`. */
typedef void (*nk_attention_narrow_genoa_t_)(void const *source, nk_bf16_t *destination, nk_size_t count,
                                             nk_size_t padded);

NK_INTERNAL void nk_attention_narrow_bf16_genoa_(void const *source, nk_bf16_t *destination, nk_size_t count,
                                                 nk_size_t padded) {
    nk_size_t channel = 0;
    for (; channel + 32 <= count; channel += 32)
        _mm512_storeu_si512(destination + channel,
                            _mm512_loadu_si512((char const *)source + channel * sizeof(nk_bf16_t)));
    if (channel < count) {
        __mmask32 const mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned int)(count - channel));
        _mm512_storeu_si512(destination + channel,
                            _mm512_maskz_loadu_epi16(mask, (char const *)source + channel * sizeof(nk_bf16_t)));
        channel += 32;
    }
    for (; channel < padded; channel += 32) _mm512_storeu_si512(destination + channel, _mm512_setzero_si512());
}

NK_INTERNAL void nk_attention_narrow_e4m3_genoa_(void const *source, nk_bf16_t *destination, nk_size_t count,
                                                 nk_size_t padded) {
    nk_size_t channel = 0;
    nk_b512_vec_t converted;
    for (; channel + 32 <= count; channel += 32) {
        nk_load_e4m3x32_to_bf16x32_icelake_((nk_e4m3_t const *)source + channel, &converted);
        _mm512_storeu_si512(destination + channel, converted.zmm);
    }
    if (channel < count) {
        nk_partial_load_e4m3x32_to_bf16x32_icelake_((nk_e4m3_t const *)source + channel, &converted, count - channel);
        _mm512_storeu_si512(destination + channel, converted.zmm);
        channel += 32;
    }
    for (; channel < padded; channel += 32) _mm512_storeu_si512(destination + channel, _mm512_setzero_si512());
}

NK_INTERNAL nk_size_t nk_attention_packed_size_genoa_(nk_size_t num_kv_heads, nk_size_t head_dim,
                                                      nk_u32_t const *segment_lengths, nk_size_t segment_count) {
    nk_size_t const dim_padded = nk_size_round_up_to_multiple_(head_dim, 32);
    nk_size_t payload_bytes = 0;
    for (nk_size_t s = 0; s < segment_count; s++)
        payload_bytes += 2 * num_kv_heads * (nk_size_t)segment_lengths[s] * dim_padded * sizeof(nk_bf16_t); // K + V
    return sizeof(nk_attention_packed_header_t) + nk_attention_packed_directory_size_(segment_count) + payload_bytes;
}

NK_PUBLIC nk_size_t nk_attention_packed_size_bf16_genoa(nk_size_t num_kv_heads, nk_size_t head_dim,
                                                        nk_u32_t const *segment_lengths, nk_size_t segment_count) {
    if (head_dim > nk_attention_max_head_dim_genoa_k_)
        return nk_attention_packed_size_bf16_serial(num_kv_heads, head_dim, segment_lengths, segment_count);
    return nk_attention_packed_size_genoa_(num_kv_heads, head_dim, segment_lengths, segment_count);
}

NK_PUBLIC nk_size_t nk_attention_packed_size_e4m3_genoa(nk_size_t num_kv_heads, nk_size_t head_dim,
                                                        nk_u32_t const *segment_lengths, nk_size_t segment_count) {
    if (head_dim > nk_attention_max_head_dim_genoa_k_)
        return nk_attention_packed_size_e4m3_serial(num_kv_heads, head_dim, segment_lengths, segment_count);
    return nk_attention_packed_size_genoa_(num_kv_heads, head_dim, segment_lengths, segment_count);
}

NK_INTERNAL void nk_attention_pack_genoa_(                                            //
    void const *k, void const *v, nk_size_t element_bytes,                            //
    nk_attention_narrow_genoa_t_ narrow,                                              //
    nk_size_t num_kv_heads, nk_size_t head_dim,                                       //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,                 //
    nk_size_t segment_count, nk_size_t k_stride, nk_size_t v_stride, void *kv_packed, //
    nk_size_t first_task, nk_size_t task_count) {

    nk_size_t const dim_padded = nk_size_round_up_to_multiple_(head_dim, 32);
    nk_attention_pack_directory_(kv_packed, num_kv_heads, head_dim, segment_lengths, segment_count, first_task, 1,
                                 dim_padded * sizeof(nk_bf16_t));
    nk_attention_packed_header_t *header = (nk_attention_packed_header_t *)kv_packed;
    nk_u64_t const *payload_offsets_ro = (nk_u64_t const *)((char *)kv_packed + sizeof(*header));
    char *payload_base = (char *)kv_packed + sizeof(*header) + nk_attention_packed_directory_size_(segment_count);

    nk_size_t const total_tasks = segment_count * num_kv_heads;
    if (first_task >= total_tasks) return;
    if (task_count == 0 || first_task + task_count > total_tasks) task_count = total_tasks - first_task;

    for (nk_size_t task = first_task; task < first_task + task_count; task++) {
        nk_size_t const segment = task / num_kv_heads, kv_head = task % num_kv_heads;
        nk_size_t const seq_len = segment_lengths[segment];
        if (seq_len == 0) continue;
        nk_size_t const token_first = segment_offsets[segment];
        nk_size_t const plane_values = seq_len * dim_padded;
        nk_bf16_t *k_plane = (nk_bf16_t *)(payload_base + payload_offsets_ro[segment]) + kv_head * plane_values;
        nk_bf16_t *v_plane = k_plane + num_kv_heads * plane_values;
        for (nk_size_t token = 0; token < seq_len; token++) {
            narrow((char const *)k + (token_first + token) * k_stride + kv_head * head_dim * element_bytes,
                   k_plane + token * dim_padded, head_dim, dim_padded);
            narrow((char const *)v + (token_first + token) * v_stride + kv_head * head_dim * element_bytes,
                   v_plane + token * dim_padded, head_dim, dim_padded);
        }
    }
}

NK_PUBLIC void nk_attention_pack_bf16_genoa(                                            //
    nk_bf16_t const *k, nk_bf16_t const *v, nk_size_t num_kv_heads, nk_size_t head_dim, //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, nk_size_t segment_count, nk_size_t k_stride,
    nk_size_t v_stride, void *kv_packed, nk_size_t first_task, nk_size_t task_count) {
    if (head_dim > nk_attention_max_head_dim_genoa_k_) {
        nk_attention_pack_bf16_serial(k, v, num_kv_heads, head_dim, segment_offsets, segment_lengths, segment_count,
                                      k_stride, v_stride, kv_packed, first_task, task_count);
        return;
    }
    nk_attention_pack_genoa_(k, v, sizeof(nk_bf16_t), &nk_attention_narrow_bf16_genoa_, num_kv_heads, head_dim,
                             segment_offsets, segment_lengths, segment_count, k_stride, v_stride, kv_packed, first_task,
                             task_count);
}

NK_PUBLIC void nk_attention_pack_e4m3_genoa(                                            //
    nk_e4m3_t const *k, nk_e4m3_t const *v, nk_size_t num_kv_heads, nk_size_t head_dim, //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, nk_size_t segment_count, nk_size_t k_stride,
    nk_size_t v_stride, void *kv_packed, nk_size_t first_task, nk_size_t task_count) {
    if (head_dim > nk_attention_max_head_dim_genoa_k_) {
        nk_attention_pack_e4m3_serial(k, v, num_kv_heads, head_dim, segment_offsets, segment_lengths, segment_count,
                                      k_stride, v_stride, kv_packed, first_task, task_count);
        return;
    }
    nk_attention_pack_genoa_(k, v, sizeof(nk_e4m3_t), &nk_attention_narrow_e4m3_genoa_, num_kv_heads, head_dim,
                             segment_offsets, segment_lengths, segment_count, k_stride, v_stride, kv_packed, first_task,
                             task_count);
}

/**
 *  @brief Shared attention core over BF16 planes: per query row, panel-flash with an exact
 *         online correction; scores use `vdpbf16ps` with four KV rows in flight.
 */
NK_INTERNAL void nk_attention_packed_genoa_(                                               //
    void const *queries, nk_size_t element_bytes, nk_attention_narrow_genoa_t_ narrow,     //
    void const *kv_packed, nk_f32_t *output,                                               //
    nk_size_t num_heads, nk_size_t num_kv_heads, nk_size_t head_dim,                       //
    nk_u32_t const *query_offsets, nk_size_t q_stride, nk_size_t o_stride, nk_f32_t scale, //
    nk_size_t first_task, nk_size_t task_count) {

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)kv_packed;
    if (header->head_dim != head_dim || header->num_kv_heads != num_kv_heads) return;
    nk_size_t const segment_count = header->segment_count;
    nk_u64_t const *payload_offsets = (nk_u64_t const *)((char const *)kv_packed + sizeof(*header));
    nk_u32_t const *segment_lengths = (nk_u32_t const *)(payload_offsets + segment_count + 1);
    char const *payload_base = (char const *)kv_packed + sizeof(*header) +
                               nk_attention_packed_directory_size_(segment_count);
    nk_size_t const o_stride_floats = o_stride / sizeof(nk_f32_t);
    nk_size_t const gqa_ratio = num_heads / num_kv_heads;
    nk_size_t const dim_padded = nk_size_round_up_to_multiple_(head_dim, 32);
    nk_f32_t const scale2 = scale * NK_LOG2E_; // softmax(x) = softmax₂(x·log₂e)
    nk_size_t const panel_width = nk_attention_panel_genoa_k_;

    nk_size_t const total_tasks = segment_count * num_heads;
    if (first_task >= total_tasks) return;
    if (task_count == 0 || first_task + task_count > total_tasks) task_count = total_tasks - first_task;

    NK_ALIGN64 nk_bf16_t query_row[nk_attention_max_head_dim_genoa_k_];
    NK_ALIGN64 nk_f32_t output_row[nk_attention_max_head_dim_genoa_k_];
    NK_ALIGN64 nk_f32_t scores[nk_attention_panel_genoa_k_];
    nk_size_t const dim_full = head_dim & ~(nk_size_t)15;
    __mmask16 const dim_tail_mask = (__mmask16)((1u << (head_dim - dim_full)) - 1);

    for (nk_size_t task = first_task; task < first_task + task_count; task++) {
        nk_size_t const segment = task / num_heads, head = task % num_heads;
        nk_size_t const kv_len = segment_lengths[segment];
        nk_size_t const query_count = query_offsets[segment + 1] - query_offsets[segment];
        if (kv_len == 0 || query_count == 0) continue;
        nk_size_t const plane_values = kv_len * dim_padded;
        nk_bf16_t const *k_plane = (nk_bf16_t const *)(payload_base + payload_offsets[segment]) +
                                   (head / gqa_ratio) * plane_values;
        nk_bf16_t const *v_plane = k_plane + num_kv_heads * plane_values;

        for (nk_size_t row = 0; row < query_count; row++) {
            narrow((char const *)queries + (query_offsets[segment] + row) * q_stride + head * head_dim * element_bytes,
                   query_row, head_dim, dim_padded);
            for (nk_size_t channel = 0; channel < dim_padded; channel += 16)
                _mm512_store_ps(output_row + channel, _mm512_setzero_ps());
            nk_f32_t running_max2 = NK_F32_MIN, running_sum = 0;

            for (nk_size_t panel_start = 0; panel_start < kv_len; panel_start += panel_width) {
                nk_size_t const panel_len = (panel_start + panel_width <= kv_len) ? panel_width
                                                                                  : (kv_len - panel_start);
                // Scores: `vdpbf16ps` accumulation, four KV rows in flight.

                nk_size_t position = 0;
                for (; position + 4 <= panel_len; position += 4) {
                    nk_bf16_t const *k_row = k_plane + (panel_start + position) * dim_padded;
                    __m512 acc0_f32x16 = _mm512_setzero_ps(), acc1_f32x16 = _mm512_setzero_ps();
                    __m512 acc2_f32x16 = _mm512_setzero_ps(), acc3_f32x16 = _mm512_setzero_ps();
                    for (nk_size_t channel = 0; channel < dim_padded; channel += 32) {
                        __m512bh const q_bf16x32 = (__m512bh)_mm512_load_si512(query_row + channel);
                        acc0_f32x16 = _mm512_dpbf16_ps(acc0_f32x16, q_bf16x32,
                                                       (__m512bh)_mm512_loadu_si512(k_row + channel));
                        acc1_f32x16 = _mm512_dpbf16_ps(acc1_f32x16, q_bf16x32,
                                                       (__m512bh)_mm512_loadu_si512(k_row + dim_padded + channel));
                        acc2_f32x16 = _mm512_dpbf16_ps(acc2_f32x16, q_bf16x32,
                                                       (__m512bh)_mm512_loadu_si512(k_row + 2 * dim_padded + channel));
                        acc3_f32x16 = _mm512_dpbf16_ps(acc3_f32x16, q_bf16x32,
                                                       (__m512bh)_mm512_loadu_si512(k_row + 3 * dim_padded + channel));
                    }
                    scores[position + 0] = nk_reduce_add_f32x16_skylake_(acc0_f32x16);
                    scores[position + 1] = nk_reduce_add_f32x16_skylake_(acc1_f32x16);
                    scores[position + 2] = nk_reduce_add_f32x16_skylake_(acc2_f32x16);
                    scores[position + 3] = nk_reduce_add_f32x16_skylake_(acc3_f32x16);
                }
                for (; position < panel_len; position++) {
                    nk_bf16_t const *k_row = k_plane + (panel_start + position) * dim_padded;
                    __m512 acc_f32x16 = _mm512_setzero_ps();
                    for (nk_size_t channel = 0; channel < dim_padded; channel += 32)
                        acc_f32x16 = _mm512_dpbf16_ps(acc_f32x16, (__m512bh)_mm512_load_si512(query_row + channel),
                                                      (__m512bh)_mm512_loadu_si512(k_row + channel));
                    scores[position] = nk_reduce_add_f32x16_skylake_(acc_f32x16);
                }

                nk_f32_t const correction = nk_attention_softmax_panel_skylake_(scores, panel_len, scale2,
                                                                                &running_max2, &running_sum);

                // O = O · correction + Σ weight · widened V-row over the panel.

                __m512 const correction_f32x16 = _mm512_set1_ps(correction);
                for (nk_size_t channel = 0; channel < dim_padded; channel += 16)
                    _mm512_store_ps(output_row + channel,
                                    _mm512_mul_ps(_mm512_load_ps(output_row + channel), correction_f32x16));
                for (position = 0; position < panel_len; position++) {
                    __m512 const weight_f32x16 = _mm512_set1_ps(scores[position]);
                    nk_bf16_t const *v_row = v_plane + (panel_start + position) * dim_padded;
                    for (nk_size_t channel = 0; channel < dim_padded; channel += 16) {
                        __m512 const v_f32x16 = nk_bf16x16_to_f32x16_skylake_(
                            _mm256_loadu_si256((__m256i const *)(v_row + channel)));
                        _mm512_store_ps(output_row + channel,
                                        _mm512_fmadd_ps(weight_f32x16, v_f32x16, _mm512_load_ps(output_row + channel)));
                    }
                }
            }

            __m512 const inverse_sum_f32x16 = _mm512_set1_ps(1.0f / running_sum);
            nk_f32_t *destination = output + (query_offsets[segment] + row) * o_stride_floats + head * head_dim;
            nk_size_t channel = 0;
            for (; channel < dim_full; channel += 16)
                _mm512_storeu_ps(destination + channel,
                                 _mm512_mul_ps(_mm512_load_ps(output_row + channel), inverse_sum_f32x16));
            if (channel < head_dim)
                _mm512_mask_storeu_ps(destination + channel, dim_tail_mask,
                                      _mm512_mul_ps(_mm512_load_ps(output_row + channel), inverse_sum_f32x16));
        }
    }
}

NK_PUBLIC void nk_attention_packed_bf16_genoa(                         //
    nk_bf16_t const *queries, void const *kv_packed, nk_f32_t *output, //
    nk_size_t num_heads, nk_size_t num_kv_heads, nk_size_t head_dim,   //
    nk_u32_t const *query_offsets,                                     //
    nk_size_t q_stride, nk_size_t o_stride, nk_f32_t scale,            //
    nk_size_t first_task, nk_size_t task_count) {
    if (head_dim > nk_attention_max_head_dim_genoa_k_) {
        nk_attention_packed_bf16_serial(queries, kv_packed, output, num_heads, num_kv_heads, head_dim, query_offsets,
                                        q_stride, o_stride, scale, first_task, task_count);
        return;
    }
    nk_attention_packed_genoa_(queries, sizeof(nk_bf16_t), &nk_attention_narrow_bf16_genoa_, kv_packed, output,
                               num_heads, num_kv_heads, head_dim, query_offsets, q_stride, o_stride, scale, first_task,
                               task_count);
}

NK_PUBLIC void nk_attention_packed_e4m3_genoa(                         //
    nk_e4m3_t const *queries, void const *kv_packed, nk_f32_t *output, //
    nk_size_t num_heads, nk_size_t num_kv_heads, nk_size_t head_dim,   //
    nk_u32_t const *query_offsets,                                     //
    nk_size_t q_stride, nk_size_t o_stride, nk_f32_t scale,            //
    nk_size_t first_task, nk_size_t task_count) {
    if (head_dim > nk_attention_max_head_dim_genoa_k_) {
        nk_attention_packed_e4m3_serial(queries, kv_packed, output, num_heads, num_kv_heads, head_dim, query_offsets,
                                        q_stride, o_stride, scale, first_task, task_count);
        return;
    }
    nk_attention_packed_genoa_(queries, sizeof(nk_e4m3_t), &nk_attention_narrow_e4m3_genoa_, kv_packed, output,
                               num_heads, num_kv_heads, head_dim, query_offsets, q_stride, o_stride, scale, first_task,
                               task_count);
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
