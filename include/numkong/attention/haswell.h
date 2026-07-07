/**
 *  @brief Ragged attention for AVX2 Haswell generation CPUs.
 *  @file include/numkong/attention/haswell.h
 *  @author Ash Vardanian
 *  @date July 6, 2026
 *
 *  @sa include/numkong/attention.h
 *
 *  Compatibility backend for AVX2/FMA machines.  Storage follows the `dots/haswell.h`
 *  conventions exactly: both BF16 and E4M3 stay in their source encoding at rest —
 *  packing is a raw strided-row copy — and every value widens to F32 on the fly inside
 *  the compute loops, 8-wide on the FMA ports.  That keeps the packed KV cache at 2 or
 *  even 1 byte per scalar, trading a couple of unpacking ops per FMA for halved-to-
 *  quartered streaming traffic, the same call the GEMM family made for this ISA.
 *
 *  The panel structure matches the family design — per query row, KV is swept in panels
 *  with an exact online correction and the family's base-2 degree-4 softmax polynomial;
 *  scores keep four KV rows in flight.  Packed payload per segment: K planes then V
 *  planes, `[kv_head][token][channel]` in the source dtype with channels zero-padded to
 *  a multiple of 8.  `head_dim > 256` routes to the width-agnostic serial tier.
 */
#ifndef NK_ATTENTION_HASWELL_H
#define NK_ATTENTION_HASWELL_H

#if NK_TARGET_X8664_
#if NK_TARGET_HASWELL

#include "numkong/attention/serial.h" // shared packed-KV header/directory, width-agnostic fallback
#include "numkong/cast/haswell.h"     // widening helpers like `nk_bf16x8_to_f32x8_haswell_`
#include "numkong/reduce/haswell.h"   // `nk_reduce_add_f32x8_haswell_`, `nk_reduce_max_f32x8_haswell_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2")
#endif

enum {
    /// KV panel width in positions; the F32 score row (2 KB) stays L1-resident.
    nk_attention_panel_haswell_k_ = 512,
    /// Widest head this backend handles in registers; larger heads route to the serial tier.
    nk_attention_max_head_dim_haswell_k_ = 256,
};

/** @brief Fast vectorized 2^x: exact range reduction + the family's shared degree-4 polynomial. */
NK_INTERNAL __m256 nk_attention_exp2_ps_haswell_(__m256 x_f32x8) {
    x_f32x8 = _mm256_max_ps(_mm256_min_ps(x_f32x8, _mm256_set1_ps(127.0f)), _mm256_set1_ps(-125.0f));
    __m256 n_f32x8 = _mm256_round_ps(x_f32x8, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r_f32x8 = _mm256_sub_ps(x_f32x8, n_f32x8);
    __m256 p_f32x8 = _mm256_set1_ps(9.61812910e-3f);
    p_f32x8 = _mm256_fmadd_ps(p_f32x8, r_f32x8, _mm256_set1_ps(5.55041087e-2f));
    p_f32x8 = _mm256_fmadd_ps(p_f32x8, r_f32x8, _mm256_set1_ps(2.40226507e-1f));
    p_f32x8 = _mm256_fmadd_ps(p_f32x8, r_f32x8, _mm256_set1_ps(6.93147181e-1f));
    p_f32x8 = _mm256_fmadd_ps(p_f32x8, r_f32x8, _mm256_set1_ps(1.0f));
    __m256i n_i32x8 = _mm256_cvtps_epi32(n_f32x8);
    n_i32x8 = _mm256_slli_epi32(_mm256_add_epi32(n_i32x8, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p_f32x8, _mm256_castsi256_ps(n_i32x8));
}

/** @brief Widens 8 raw plane scalars (BF16 or E4M3 at rest) to F32 inside the hot loops. */
typedef __m256 (*nk_attention_plane_widen_haswell_t_)(void const *plane_chunk);

NK_INTERNAL __m256 nk_attention_plane_widen_bf16_haswell_(void const *plane_chunk) {
    return nk_bf16x8_to_f32x8_haswell_(_mm_loadu_si128((__m128i const *)plane_chunk));
}

NK_INTERNAL __m256 nk_attention_plane_widen_e4m3_haswell_(void const *plane_chunk) {
    return nk_e4m3x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)plane_chunk));
}

/** @brief Widens `count` raw query elements to F32 into `destination`, zero-filling to `padded`. */
typedef void (*nk_attention_widen_haswell_t_)(void const *source, nk_f32_t *destination, nk_size_t count,
                                              nk_size_t padded);

NK_INTERNAL void nk_attention_widen_bf16_haswell_(void const *source, nk_f32_t *destination, nk_size_t count,
                                                  nk_size_t padded) {
    nk_size_t channel = 0;
    nk_b256_vec_t widened;
    for (; channel + 8 <= count; channel += 8) {
        nk_load_bf16x8_to_f32x8_haswell_((nk_bf16_t const *)source + channel, &widened);
        _mm256_storeu_ps(destination + channel, widened.ymm_ps);
    }
    if (channel < count) {
        nk_partial_load_bf16x8_to_f32x8_haswell_((nk_bf16_t const *)source + channel, &widened, count - channel);
        _mm256_storeu_ps(destination + channel, widened.ymm_ps);
        channel += 8;
    }
    for (; channel < padded; channel += 8) _mm256_storeu_ps(destination + channel, _mm256_setzero_ps());
}

NK_INTERNAL void nk_attention_widen_e4m3_haswell_(void const *source, nk_f32_t *destination, nk_size_t count,
                                                  nk_size_t padded) {
    nk_size_t channel = 0;
    nk_b256_vec_t widened;
    for (; channel + 8 <= count; channel += 8) {
        nk_load_e4m3x8_to_f32x8_haswell_((nk_e4m3_t const *)source + channel, &widened);
        _mm256_storeu_ps(destination + channel, widened.ymm_ps);
    }
    if (channel < count) {
        nk_partial_load_e4m3x8_to_f32x8_haswell_((nk_e4m3_t const *)source + channel, &widened, count - channel);
        _mm256_storeu_ps(destination + channel, widened.ymm_ps);
        channel += 8;
    }
    for (; channel < padded; channel += 8) _mm256_storeu_ps(destination + channel, _mm256_setzero_ps());
}

NK_INTERNAL nk_size_t nk_attention_packed_size_haswell_(nk_size_t num_kv_heads, nk_size_t head_dim,
                                                        nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                        nk_size_t element_bytes) {
    nk_size_t const dim_padded = nk_size_round_up_to_multiple_(head_dim, 8);
    nk_size_t payload_bytes = 0; // planes keep the source encoding, like the dots family
    for (nk_size_t s = 0; s < segment_count; s++)
        payload_bytes += 2 * num_kv_heads * (nk_size_t)segment_lengths[s] * dim_padded * element_bytes;
    return sizeof(nk_attention_packed_header_t) + nk_attention_packed_directory_size_(segment_count) + payload_bytes;
}

NK_PUBLIC nk_size_t nk_attention_packed_size_bf16_haswell(nk_size_t num_kv_heads, nk_size_t head_dim,
                                                          nk_u32_t const *segment_lengths, nk_size_t segment_count) {
    if (head_dim > nk_attention_max_head_dim_haswell_k_)
        return nk_attention_packed_size_bf16_serial(num_kv_heads, head_dim, segment_lengths, segment_count);
    return nk_attention_packed_size_haswell_(num_kv_heads, head_dim, segment_lengths, segment_count, sizeof(nk_bf16_t));
}

NK_PUBLIC nk_size_t nk_attention_packed_size_e4m3_haswell(nk_size_t num_kv_heads, nk_size_t head_dim,
                                                          nk_u32_t const *segment_lengths, nk_size_t segment_count) {
    if (head_dim > nk_attention_max_head_dim_haswell_k_)
        return nk_attention_packed_size_e4m3_serial(num_kv_heads, head_dim, segment_lengths, segment_count);
    return nk_attention_packed_size_haswell_(num_kv_heads, head_dim, segment_lengths, segment_count, sizeof(nk_e4m3_t));
}

/** @brief Raw strided-row repack: source encoding is preserved, tails zero-padded. */
NK_INTERNAL void nk_attention_pack_haswell_(                                          //
    void const *k, void const *v, nk_size_t element_bytes,                            //
    nk_size_t num_kv_heads, nk_size_t head_dim,                                       //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,                 //
    nk_size_t segment_count, nk_size_t k_stride, nk_size_t v_stride, void *kv_packed, //
    nk_size_t first_task, nk_size_t task_count) {

    nk_size_t const dim_padded = nk_size_round_up_to_multiple_(head_dim, 8);
    nk_size_t const row_bytes = head_dim * element_bytes;
    nk_size_t const padded_row_bytes = dim_padded * element_bytes;
    nk_attention_pack_directory_(kv_packed, num_kv_heads, head_dim, segment_lengths, segment_count, first_task, 1,
                                 padded_row_bytes);
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
        nk_size_t const plane_bytes = seq_len * padded_row_bytes;
        char *k_plane = payload_base + payload_offsets_ro[segment] + kv_head * plane_bytes;
        char *v_plane = k_plane + num_kv_heads * plane_bytes;
        for (nk_size_t token = 0; token < seq_len; token++) {
            char const *k_row = (char const *)k + (token_first + token) * k_stride + kv_head * row_bytes;
            char const *v_row = (char const *)v + (token_first + token) * v_stride + kv_head * row_bytes;
            char *k_destination = k_plane + token * padded_row_bytes;
            char *v_destination = v_plane + token * padded_row_bytes;
            for (nk_size_t byte = 0; byte < row_bytes; byte++) k_destination[byte] = k_row[byte];
            for (nk_size_t byte = row_bytes; byte < padded_row_bytes; byte++) k_destination[byte] = 0;
            for (nk_size_t byte = 0; byte < row_bytes; byte++) v_destination[byte] = v_row[byte];
            for (nk_size_t byte = row_bytes; byte < padded_row_bytes; byte++) v_destination[byte] = 0;
        }
    }
}

NK_PUBLIC void nk_attention_pack_bf16_haswell(                                          //
    nk_bf16_t const *k, nk_bf16_t const *v, nk_size_t num_kv_heads, nk_size_t head_dim, //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, nk_size_t segment_count, nk_size_t k_stride,
    nk_size_t v_stride, void *kv_packed, nk_size_t first_task, nk_size_t task_count) {
    if (head_dim > nk_attention_max_head_dim_haswell_k_) {
        nk_attention_pack_bf16_serial(k, v, num_kv_heads, head_dim, segment_offsets, segment_lengths, segment_count,
                                      k_stride, v_stride, kv_packed, first_task, task_count);
        return;
    }
    nk_attention_pack_haswell_(k, v, sizeof(nk_bf16_t), num_kv_heads, head_dim, segment_offsets, segment_lengths,
                               segment_count, k_stride, v_stride, kv_packed, first_task, task_count);
}

NK_PUBLIC void nk_attention_pack_e4m3_haswell(                                          //
    nk_e4m3_t const *k, nk_e4m3_t const *v, nk_size_t num_kv_heads, nk_size_t head_dim, //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, nk_size_t segment_count, nk_size_t k_stride,
    nk_size_t v_stride, void *kv_packed, nk_size_t first_task, nk_size_t task_count) {
    if (head_dim > nk_attention_max_head_dim_haswell_k_) {
        nk_attention_pack_e4m3_serial(k, v, num_kv_heads, head_dim, segment_offsets, segment_lengths, segment_count,
                                      k_stride, v_stride, kv_packed, first_task, task_count);
        return;
    }
    nk_attention_pack_haswell_(k, v, sizeof(nk_e4m3_t), num_kv_heads, head_dim, segment_offsets, segment_lengths,
                               segment_count, k_stride, v_stride, kv_packed, first_task, task_count);
}

/**
 *  @brief Shared attention core over raw-encoded planes: per query row, panel-flash with
 *         an exact online correction; scores keep four KV rows in flight, widening in-loop.
 */
NK_INTERNAL void nk_attention_packed_haswell_(                                             //
    void const *queries, nk_size_t element_bytes, nk_attention_widen_haswell_t_ widen,     //
    nk_attention_plane_widen_haswell_t_ plane_widen,                                       //
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
    nk_size_t const dim_padded = nk_size_round_up_to_multiple_(head_dim, 8);
    nk_size_t const plane_row_bytes = dim_padded * element_bytes;
    nk_f32_t const scale2 = scale * NK_LOG2E_; // softmax(x) = softmax₂(x·log₂e)
    nk_size_t const panel_width = nk_attention_panel_haswell_k_;

    nk_size_t const total_tasks = segment_count * num_heads;
    if (first_task >= total_tasks) return;
    if (task_count == 0 || first_task + task_count > total_tasks) task_count = total_tasks - first_task;

    NK_ALIGN64 nk_f32_t query_row[nk_attention_max_head_dim_haswell_k_];
    NK_ALIGN64 nk_f32_t output_row[nk_attention_max_head_dim_haswell_k_];
    NK_ALIGN64 nk_f32_t scores[nk_attention_panel_haswell_k_];

    for (nk_size_t task = first_task; task < first_task + task_count; task++) {
        nk_size_t const segment = task / num_heads, head = task % num_heads;
        nk_size_t const kv_len = segment_lengths[segment];
        nk_size_t const query_count = query_offsets[segment + 1] - query_offsets[segment];
        if (kv_len == 0 || query_count == 0) continue;
        nk_size_t const plane_bytes = kv_len * plane_row_bytes;
        char const *k_plane = payload_base + payload_offsets[segment] + (head / gqa_ratio) * plane_bytes;
        char const *v_plane = k_plane + num_kv_heads * plane_bytes;

        for (nk_size_t row = 0; row < query_count; row++) {
            widen((char const *)queries + (query_offsets[segment] + row) * q_stride + head * head_dim * element_bytes,
                  query_row, head_dim, dim_padded);
            for (nk_size_t channel = 0; channel < dim_padded; channel += 8)
                _mm256_store_ps(output_row + channel, _mm256_setzero_ps());
            nk_f32_t running_max2 = NK_F32_MIN, running_sum = 0;

            for (nk_size_t panel_start = 0; panel_start < kv_len; panel_start += panel_width) {
                nk_size_t const panel_len = (panel_start + panel_width <= kv_len) ? panel_width
                                                                                  : (kv_len - panel_start);
                // Scores: four KV rows in flight, raw plane scalars widened in-loop.

                nk_size_t position = 0;
                for (; position + 4 <= panel_len; position += 4) {
                    char const *k_row = k_plane + (panel_start + position) * plane_row_bytes;
                    __m256 acc0_f32x8 = _mm256_setzero_ps(), acc1_f32x8 = _mm256_setzero_ps();
                    __m256 acc2_f32x8 = _mm256_setzero_ps(), acc3_f32x8 = _mm256_setzero_ps();
                    for (nk_size_t channel = 0; channel < dim_padded; channel += 8) {
                        __m256 const q_f32x8 = _mm256_load_ps(query_row + channel);
                        nk_size_t const chunk_bytes = channel * element_bytes;
                        acc0_f32x8 = _mm256_fmadd_ps(q_f32x8, plane_widen(k_row + chunk_bytes), acc0_f32x8);
                        acc1_f32x8 = _mm256_fmadd_ps(q_f32x8, plane_widen(k_row + plane_row_bytes + chunk_bytes),
                                                     acc1_f32x8);
                        acc2_f32x8 = _mm256_fmadd_ps(q_f32x8, plane_widen(k_row + 2 * plane_row_bytes + chunk_bytes),
                                                     acc2_f32x8);
                        acc3_f32x8 = _mm256_fmadd_ps(q_f32x8, plane_widen(k_row + 3 * plane_row_bytes + chunk_bytes),
                                                     acc3_f32x8);
                    }
                    scores[position + 0] = nk_reduce_add_f32x8_haswell_(acc0_f32x8);
                    scores[position + 1] = nk_reduce_add_f32x8_haswell_(acc1_f32x8);
                    scores[position + 2] = nk_reduce_add_f32x8_haswell_(acc2_f32x8);
                    scores[position + 3] = nk_reduce_add_f32x8_haswell_(acc3_f32x8);
                }
                for (; position < panel_len; position++) {
                    char const *k_row = k_plane + (panel_start + position) * plane_row_bytes;
                    __m256 acc_f32x8 = _mm256_setzero_ps();
                    for (nk_size_t channel = 0; channel < dim_padded; channel += 8)
                        acc_f32x8 = _mm256_fmadd_ps(_mm256_load_ps(query_row + channel),
                                                    plane_widen(k_row + channel * element_bytes), acc_f32x8);
                    scores[position] = nk_reduce_add_f32x8_haswell_(acc_f32x8);
                }

                // Panel max, online correction, exp2 in place (scalar tail via the serial exp2).

                __m256 max_f32x8 = _mm256_set1_ps(NK_F32_MIN);
                nk_size_t col = 0;
                for (; col + 8 <= panel_len; col += 8)
                    max_f32x8 = _mm256_max_ps(max_f32x8, _mm256_loadu_ps(scores + col));
                nk_f32_t panel_max2 = nk_reduce_max_f32x8_haswell_(max_f32x8) * scale2;
                for (; col < panel_len; col++) {
                    nk_f32_t const scaled2 = scores[col] * scale2;
                    if (scaled2 > panel_max2) panel_max2 = scaled2;
                }
                nk_f32_t const new_max2 = running_max2 > panel_max2 ? running_max2 : panel_max2;
                nk_f32_t const correction = _mm256_cvtss_f32(
                    nk_attention_exp2_ps_haswell_(_mm256_set1_ps(running_max2 - new_max2)));
                running_max2 = new_max2;

                __m256 const scale2_f32x8 = _mm256_set1_ps(scale2);
                __m256 const max2_f32x8 = _mm256_set1_ps(new_max2);
                __m256 sum_f32x8 = _mm256_setzero_ps();
                for (col = 0; col + 8 <= panel_len; col += 8) {
                    __m256 weights_f32x8 = nk_attention_exp2_ps_haswell_(
                        _mm256_fmsub_ps(_mm256_loadu_ps(scores + col), scale2_f32x8, max2_f32x8));
                    sum_f32x8 = _mm256_add_ps(sum_f32x8, weights_f32x8);
                    _mm256_storeu_ps(scores + col, weights_f32x8);
                }
                if (col < panel_len) { // masked tail keeps the vector rounding mode end-to-end
                    __m256i const lane_idx_i32x8 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                    __m256i const tail_mask_i32x8 = _mm256_cmpgt_epi32(_mm256_set1_epi32((int)(panel_len - col)),
                                                                       lane_idx_i32x8);
                    __m256 weights_f32x8 = nk_attention_exp2_ps_haswell_(
                        _mm256_fmsub_ps(_mm256_maskload_ps(scores + col, tail_mask_i32x8), scale2_f32x8, max2_f32x8));
                    weights_f32x8 = _mm256_and_ps(weights_f32x8, _mm256_castsi256_ps(tail_mask_i32x8));
                    sum_f32x8 = _mm256_add_ps(sum_f32x8, weights_f32x8);
                    _mm256_maskstore_ps(scores + col, tail_mask_i32x8, weights_f32x8);
                }
                running_sum = running_sum * correction + nk_reduce_add_f32x8_haswell_(sum_f32x8);

                // O = O · correction + Σ weight · widened V-row over the panel.

                __m256 const correction_f32x8 = _mm256_set1_ps(correction);
                for (nk_size_t channel = 0; channel < dim_padded; channel += 8)
                    _mm256_store_ps(output_row + channel,
                                    _mm256_mul_ps(_mm256_load_ps(output_row + channel), correction_f32x8));
                for (position = 0; position < panel_len; position++) {
                    __m256 const weight_f32x8 = _mm256_set1_ps(scores[position]);
                    char const *v_row = v_plane + (panel_start + position) * plane_row_bytes;
                    for (nk_size_t channel = 0; channel < dim_padded; channel += 8)
                        _mm256_store_ps(output_row + channel,
                                        _mm256_fmadd_ps(weight_f32x8, plane_widen(v_row + channel * element_bytes),
                                                        _mm256_load_ps(output_row + channel)));
                }
            }

            nk_f32_t const inverse_sum = 1.0f / running_sum;
            __m256 const inverse_sum_f32x8 = _mm256_set1_ps(inverse_sum);
            nk_f32_t *destination = output + (query_offsets[segment] + row) * o_stride_floats + head * head_dim;
            nk_size_t channel = 0;
            for (; channel + 8 <= head_dim; channel += 8)
                _mm256_storeu_ps(destination + channel,
                                 _mm256_mul_ps(_mm256_load_ps(output_row + channel), inverse_sum_f32x8));
            for (; channel < head_dim; channel++) destination[channel] = output_row[channel] * inverse_sum;
        }
    }
}

NK_PUBLIC void nk_attention_packed_bf16_haswell(                       //
    nk_bf16_t const *queries, void const *kv_packed, nk_f32_t *output, //
    nk_size_t num_heads, nk_size_t num_kv_heads, nk_size_t head_dim,   //
    nk_u32_t const *query_offsets,                                     //
    nk_size_t q_stride, nk_size_t o_stride, nk_f32_t scale,            //
    nk_size_t first_task, nk_size_t task_count) {
    if (head_dim > nk_attention_max_head_dim_haswell_k_) {
        nk_attention_packed_bf16_serial(queries, kv_packed, output, num_heads, num_kv_heads, head_dim, query_offsets,
                                        q_stride, o_stride, scale, first_task, task_count);
        return;
    }
    nk_attention_packed_haswell_(queries, sizeof(nk_bf16_t), &nk_attention_widen_bf16_haswell_,
                                 &nk_attention_plane_widen_bf16_haswell_, kv_packed, output, num_heads, num_kv_heads,
                                 head_dim, query_offsets, q_stride, o_stride, scale, first_task, task_count);
}

NK_PUBLIC void nk_attention_packed_e4m3_haswell(                       //
    nk_e4m3_t const *queries, void const *kv_packed, nk_f32_t *output, //
    nk_size_t num_heads, nk_size_t num_kv_heads, nk_size_t head_dim,   //
    nk_u32_t const *query_offsets,                                     //
    nk_size_t q_stride, nk_size_t o_stride, nk_f32_t scale,            //
    nk_size_t first_task, nk_size_t task_count) {
    if (head_dim > nk_attention_max_head_dim_haswell_k_) {
        nk_attention_packed_e4m3_serial(queries, kv_packed, output, num_heads, num_kv_heads, head_dim, query_offsets,
                                        q_stride, o_stride, scale, first_task, task_count);
        return;
    }
    nk_attention_packed_haswell_(queries, sizeof(nk_e4m3_t), &nk_attention_widen_e4m3_haswell_,
                                 &nk_attention_plane_widen_e4m3_haswell_, kv_packed, output, num_heads, num_kv_heads,
                                 head_dim, query_offsets, q_stride, o_stride, scale, first_task, task_count);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_HASWELL
#endif // NK_TARGET_X8664_
#endif // NK_ATTENTION_HASWELL_H
