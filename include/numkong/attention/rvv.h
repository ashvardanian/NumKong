/**
 *  @file include/numkong/attention/rvv.h
 *  @author Ash Vardanian
 *  @date July 6, 2026
 *  @brief Ragged attention for RISC-V CPUs with the RVV 1.0 vector extension.
 *
 *  @sa include/numkong/attention.h
 *
 *  Vector-length-agnostic backend: every loop strip-mines with @c __riscv_vsetvl, so there are no
 *  masks, no scalar tails, and no channel padding anywhere in the file. That is a deliberate
 *  divergence from the x86 backends, which zero-pad packed planes to multiples of 16 channels —
 *  here the packed K/V planes are raw and unpadded, and VLA strip-mining absorbs arbitrary depths
 *  and position counts, including odd ones, at full speed.
 *
 *  The panel structure matches the family design — per query row, KV is swept in panels with an
 *  exact online correction and a base-2 streaming softmax sharing the family's degree-7 polynomial
 *  from nk_f32_exp2_serial_. Score and accumulation loops keep operands at LMUL m1/m2 to leave
 *  register-group headroom; the softmax elementwise passes over the score row run wider at e32m4.
 *
 *  Storage mirrors `dots/rvv.h` economics: BF16 stays BF16 at rest and widens to F32 with a
 *  zero-extend + shift, through nk_bf16m1_to_f32m2_rvv_. E4M3 K/V converts once to BF16 at pack
 *  time — the conversion is lossless, 3 mantissa bits fit into 7, the packed plane stays 2 bytes
 *  per value, and the hot loops reuse the cheap BF16 shift loader instead of an in-loop gather.
 *  Only the unpacked query side keeps a 128-entry magnitude-LUT gather loader for E4M3.
 *
 *  The I8 path is exact: scores are true I32 integer dot-products — widening multiply and
 *  accumulate — the row maximum is taken over live columns only, and softmax weights quantize to U8
 *  as round(255 · 2^(s₂ − m₂)) with the vector exponent of each/rvv.h, which tracks the serial
 *  kernel's polynomial to within a unit in the last place. Outputs accumulate in place, so any
 *  depth runs on RVV.
 */
#ifndef NUMKONG_ATTENTION_RVV_H
#define NUMKONG_ATTENTION_RVV_H

#if NUMKONG_ARCH_RISCV64_
#if NUMKONG_ARCH_RISCV64_RVV_

#include "numkong/attention/serial.h" // shared packed-KV header/directory, width-agnostic fallback
#include "numkong/cast/rvv.h"         // `nk_bf16m1_to_f32m2_rvv_`, `nk_f32m2_to_bf16m1_rvv_`
#include "numkong/each/rvv.h"         // `nk_exp2_f32m4_rvv_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

enum {

    /** KV panel width in positions; the F32 score row (2 KB) stays L1-resident. */
    nk_attention_panel_rvv_k_ = 512,
};

/** Widens @p vector_length raw scalars (BF16 or E4M3) to an F32 (m2) register group. */
NUMKONG_INLINE vfloat32m2_t nk_attention_load_bf16_f32m2_rvv_(void const *source, nk_size_t vector_length) {
    return nk_bf16m1_to_f32m2_rvv_(__riscv_vle16_v_u16m1((nk_u16_t const *)source, vector_length), vector_length);
}

NUMKONG_INLINE vfloat32m2_t nk_attention_load_e4m3_f32m2_rvv_(void const *source, nk_size_t vector_length) {
    // Sign-symmetric magnitude LUT, `cast/rvv.h`-style: 128 F32 bit patterns for bits 6:0,
    // the sign bit re-attached separately. Entry 0x7F is the E4M3FN NaN.
    static nk_u32_t const nk_e4m3_mag_to_f32_lut_[128] = {
        0x00000000, 0x3B000000, 0x3B800000, 0x3BC00000, 0x3C000000, 0x3C200000, 0x3C400000, 0x3C600000, /* [  0..  7] */
        0x3C800000, 0x3C900000, 0x3CA00000, 0x3CB00000, 0x3CC00000, 0x3CD00000, 0x3CE00000, 0x3CF00000, /* [  8.. 15] */
        0x3D000000, 0x3D100000, 0x3D200000, 0x3D300000, 0x3D400000, 0x3D500000, 0x3D600000, 0x3D700000, /* [ 16.. 23] */
        0x3D800000, 0x3D900000, 0x3DA00000, 0x3DB00000, 0x3DC00000, 0x3DD00000, 0x3DE00000, 0x3DF00000, /* [ 24.. 31] */
        0x3E000000, 0x3E100000, 0x3E200000, 0x3E300000, 0x3E400000, 0x3E500000, 0x3E600000, 0x3E700000, /* [ 32.. 39] */
        0x3E800000, 0x3E900000, 0x3EA00000, 0x3EB00000, 0x3EC00000, 0x3ED00000, 0x3EE00000, 0x3EF00000, /* [ 40.. 47] */
        0x3F000000, 0x3F100000, 0x3F200000, 0x3F300000, 0x3F400000, 0x3F500000, 0x3F600000, 0x3F700000, /* [ 48.. 55] */
        0x3F800000, 0x3F900000, 0x3FA00000, 0x3FB00000, 0x3FC00000, 0x3FD00000, 0x3FE00000, 0x3FF00000, /* [ 56.. 63] */
        0x40000000, 0x40100000, 0x40200000, 0x40300000, 0x40400000, 0x40500000, 0x40600000, 0x40700000, /* [ 64.. 71] */
        0x40800000, 0x40900000, 0x40A00000, 0x40B00000, 0x40C00000, 0x40D00000, 0x40E00000, 0x40F00000, /* [ 72.. 79] */
        0x41000000, 0x41100000, 0x41200000, 0x41300000, 0x41400000, 0x41500000, 0x41600000, 0x41700000, /* [ 80.. 87] */
        0x41800000, 0x41900000, 0x41A00000, 0x41B00000, 0x41C00000, 0x41D00000, 0x41E00000, 0x41F00000, /* [ 88.. 95] */
        0x42000000, 0x42100000, 0x42200000, 0x42300000, 0x42400000, 0x42500000, 0x42600000, 0x42700000, /* [ 96..103] */
        0x42800000, 0x42900000, 0x42A00000, 0x42B00000, 0x42C00000, 0x42D00000, 0x42E00000, 0x42F00000, /* [104..111] */
        0x43000000, 0x43100000, 0x43200000, 0x43300000, 0x43400000, 0x43500000, 0x43600000, 0x43700000, /* [112..119] */
        0x43800000, 0x43900000, 0x43A00000, 0x43B00000, 0x43C00000, 0x43D00000, 0x43E00000, 0x7FC00000  /* [120..127] */
    };
    vuint8mf2_t raw_u8mf2 = __riscv_vle8_v_u8mf2((nk_u8_t const *)source, vector_length);
    vuint16m1_t offsets_u16m1 = __riscv_vsll_vx_u16m1(
        __riscv_vzext_vf2_u16m1(__riscv_vand_vx_u8mf2(raw_u8mf2, 0x7F, vector_length), vector_length), 2,
        vector_length);
    vuint32m2_t magnitude_u32m2 = __riscv_vluxei16_v_u32m2(nk_e4m3_mag_to_f32_lut_, offsets_u16m1, vector_length);
    vuint32m2_t sign_u32m2 = __riscv_vsll_vx_u32m2(
        __riscv_vzext_vf4_u32m2(__riscv_vand_vx_u8mf2(raw_u8mf2, 0x80, vector_length), vector_length), 24,
        vector_length);
    return __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vor_vv_u32m2(magnitude_u32m2, sign_u32m2, vector_length));
}

/** Repacks one row of @p count input scalars into the packed-plane representation. */
NUMKONG_INLINE void nk_attention_pack_row_bf16_rvv_(void const *source, void *destination, nk_size_t count) {
    nk_u16_t const *source_u16 = (nk_u16_t const *)source;
    nk_u16_t *destination_u16 = (nk_u16_t *)destination;
    for (nk_size_t channel_idx = 0, remaining = count, vector_length = 0; remaining > 0;
         remaining -= vector_length, channel_idx += vector_length) {
        vector_length = __riscv_vsetvl_e16m2(remaining);
        __riscv_vse16_v_u16m2(destination_u16 + channel_idx,
                              __riscv_vle16_v_u16m2(source_u16 + channel_idx, vector_length), vector_length);
    }
}

NUMKONG_INLINE void nk_attention_pack_row_e4m3_rvv_(void const *source, void *destination, nk_size_t count) {
    // E4M3 → BF16 is lossless (3 mantissa bits into 7), so the RNE narrowing is an identity;
    // paying one LUT gather per value here keeps every hot-loop K/V load a two-op shift.
    nk_e4m3_t const *source_e4m3 = (nk_e4m3_t const *)source;
    nk_u16_t *destination_u16 = (nk_u16_t *)destination;
    for (nk_size_t channel_idx = 0, remaining = count, vector_length = 0; remaining > 0;
         remaining -= vector_length, channel_idx += vector_length) {
        vector_length = __riscv_vsetvl_e32m2(remaining);
        vfloat32m2_t values_f32m2 = nk_attention_load_e4m3_f32m2_rvv_(source_e4m3 + channel_idx, vector_length);
        __riscv_vse16_v_u16m1(destination_u16 + channel_idx, nk_f32m2_to_bf16m1_rvv_(values_f32m2, vector_length),
                              vector_length);
    }
}

NUMKONG_INLINE void nk_attention_pack_row_i8_rvv_(void const *source, void *destination, nk_size_t count) {
    nk_i8_t const *source_i8 = (nk_i8_t const *)source;
    nk_i8_t *destination_i8 = (nk_i8_t *)destination;
    for (nk_size_t channel_idx = 0, remaining = count, vector_length = 0; remaining > 0;
         remaining -= vector_length, channel_idx += vector_length) {
        vector_length = __riscv_vsetvl_e8m2(remaining);
        __riscv_vse8_v_i8m2(destination_i8 + channel_idx, __riscv_vle8_v_i8m2(source_i8 + channel_idx, vector_length),
                            vector_length);
    }
}

/** Bytes of a pack of raw unpadded planes, as VLA strip-mining needs no channel padding, of
 *  @p packed_element_bytes each. */
NUMKONG_INLINE nk_size_t nk_attention_pack_size_rvv_(nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_size_t token_count, nk_size_t segment_count,
                                                     nk_size_t packed_element_bytes) {
    return nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1,
                                           depth * packed_element_bytes);
}

/** BF16 packing core: repacks K and V rows into raw `[key_value_head][position][channel]` planes
 *  per segment, with no padding in either axis. Each window derives its plane offsets from
 *  @p key_lengths, so windows run in any order. */
NUMKONG_INLINE void nk_attention_pack_bf16_rvv_(nk_bf16_t const *keys, nk_bf16_t const *values,
                                                nk_size_t key_value_head_count, nk_size_t depth,
                                                nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                nk_size_t segment_count, nk_size_t key_stride, nk_size_t value_stride,
                                                void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end) {
    nk_attention_pack_directory_serial_(key_value_packed, key_value_head_count, depth, key_offsets, key_lengths,
                                        segment_count, tasks_begin, 1, depth * sizeof(nk_bf16_t), nk_cap_rvv_k);
    char *payload_base = (char *)key_value_packed + sizeof(nk_attention_packed_header_t) +
                         nk_attention_pack_directory_size_serial_(segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    if (tasks_begin >= total_tasks) return;
    if (tasks_end > total_tasks) tasks_end = total_tasks;

    nk_size_t payload_segment = 0;
    nk_u64_t payload_offset = 0;
    for (nk_size_t task_idx = tasks_begin; task_idx < tasks_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count;
        nk_size_t const key_value_head_idx = task_idx % key_value_head_count;
        for (; payload_segment < segment_idx; payload_segment++)
            payload_offset += nk_attention_pack_segment_bytes_serial_(
                nk_attention_pack_key_count_serial_(key_offsets, key_lengths, payload_segment), key_value_head_count, 1,
                depth * sizeof(nk_bf16_t));
        nk_size_t const position_count = nk_attention_pack_key_count_serial_(key_offsets, key_lengths, segment_idx);
        if (position_count == 0) continue;
        nk_size_t const position_first = key_offsets[segment_idx];
        nk_size_t const plane_bytes = position_count * depth * sizeof(nk_bf16_t);
        char *keys_plane = payload_base + payload_offset + key_value_head_idx * plane_bytes;
        char *values_plane = keys_plane + key_value_head_count * plane_bytes;
        for (nk_size_t position_idx = 0; position_idx < position_count; position_idx++) {
            nk_attention_pack_row_bf16_rvv_((char const *)keys + (position_first + position_idx) * key_stride +
                                                key_value_head_idx * depth * sizeof(nk_bf16_t),
                                            keys_plane + position_idx * depth * sizeof(nk_bf16_t), depth);
            nk_attention_pack_row_bf16_rvv_((char const *)values + (position_first + position_idx) * value_stride +
                                                key_value_head_idx * depth * sizeof(nk_bf16_t),
                                            values_plane + position_idx * depth * sizeof(nk_bf16_t), depth);
        }
    }
}

/** E4M3 packing core: repacks K and V rows into raw `[key_value_head][position][channel]` planes
 *  per segment, with no padding in either axis. Each window derives its plane offsets from
 *  @p key_lengths, so windows run in any order. */
NUMKONG_INLINE void nk_attention_pack_e4m3_rvv_(nk_e4m3_t const *keys, nk_e4m3_t const *values,
                                                nk_size_t key_value_head_count, nk_size_t depth,
                                                nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                nk_size_t segment_count, nk_size_t key_stride, nk_size_t value_stride,
                                                void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end) {
    nk_attention_pack_directory_serial_(key_value_packed, key_value_head_count, depth, key_offsets, key_lengths,
                                        segment_count, tasks_begin, 1, depth * sizeof(nk_bf16_t), nk_cap_rvv_k);
    char *payload_base = (char *)key_value_packed + sizeof(nk_attention_packed_header_t) +
                         nk_attention_pack_directory_size_serial_(segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    if (tasks_begin >= total_tasks) return;
    if (tasks_end > total_tasks) tasks_end = total_tasks;

    nk_size_t payload_segment = 0;
    nk_u64_t payload_offset = 0;
    for (nk_size_t task_idx = tasks_begin; task_idx < tasks_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count;
        nk_size_t const key_value_head_idx = task_idx % key_value_head_count;
        for (; payload_segment < segment_idx; payload_segment++)
            payload_offset += nk_attention_pack_segment_bytes_serial_(
                nk_attention_pack_key_count_serial_(key_offsets, key_lengths, payload_segment), key_value_head_count, 1,
                depth * sizeof(nk_bf16_t));
        nk_size_t const position_count = nk_attention_pack_key_count_serial_(key_offsets, key_lengths, segment_idx);
        if (position_count == 0) continue;
        nk_size_t const position_first = key_offsets[segment_idx];
        nk_size_t const plane_bytes = position_count * depth * sizeof(nk_bf16_t);
        char *keys_plane = payload_base + payload_offset + key_value_head_idx * plane_bytes;
        char *values_plane = keys_plane + key_value_head_count * plane_bytes;
        for (nk_size_t position_idx = 0; position_idx < position_count; position_idx++) {
            nk_attention_pack_row_e4m3_rvv_((char const *)keys + (position_first + position_idx) * key_stride +
                                                key_value_head_idx * depth * sizeof(nk_e4m3_t),
                                            keys_plane + position_idx * depth * sizeof(nk_bf16_t), depth);
            nk_attention_pack_row_e4m3_rvv_((char const *)values + (position_first + position_idx) * value_stride +
                                                key_value_head_idx * depth * sizeof(nk_e4m3_t),
                                            values_plane + position_idx * depth * sizeof(nk_bf16_t), depth);
        }
    }
}

/** I8 packing core: repacks K and V rows into raw `[key_value_head][position][channel]` planes per
 *  segment, with no padding in either axis. Each window derives its plane offsets from
 *  @p key_lengths, so windows run in any order. */
NUMKONG_INLINE void nk_attention_pack_i8_rvv_(nk_i8_t const *keys, nk_i8_t const *values,
                                              nk_size_t key_value_head_count, nk_size_t depth,
                                              nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                              nk_size_t segment_count, nk_size_t key_stride, nk_size_t value_stride,
                                              void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end) {
    nk_attention_pack_directory_serial_(key_value_packed, key_value_head_count, depth, key_offsets, key_lengths,
                                        segment_count, tasks_begin, 1, depth * sizeof(nk_i8_t), nk_cap_rvv_k);
    char *payload_base = (char *)key_value_packed + sizeof(nk_attention_packed_header_t) +
                         nk_attention_pack_directory_size_serial_(segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    if (tasks_begin >= total_tasks) return;
    if (tasks_end > total_tasks) tasks_end = total_tasks;

    nk_size_t payload_segment = 0;
    nk_u64_t payload_offset = 0;
    for (nk_size_t task_idx = tasks_begin; task_idx < tasks_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count;
        nk_size_t const key_value_head_idx = task_idx % key_value_head_count;
        for (; payload_segment < segment_idx; payload_segment++)
            payload_offset += nk_attention_pack_segment_bytes_serial_(
                nk_attention_pack_key_count_serial_(key_offsets, key_lengths, payload_segment), key_value_head_count, 1,
                depth * sizeof(nk_i8_t));
        nk_size_t const position_count = nk_attention_pack_key_count_serial_(key_offsets, key_lengths, segment_idx);
        if (position_count == 0) continue;
        nk_size_t const position_first = key_offsets[segment_idx];
        nk_size_t const plane_bytes = position_count * depth * sizeof(nk_i8_t);
        char *keys_plane = payload_base + payload_offset + key_value_head_idx * plane_bytes;
        char *values_plane = keys_plane + key_value_head_count * plane_bytes;
        for (nk_size_t position_idx = 0; position_idx < position_count; position_idx++) {
            nk_attention_pack_row_i8_rvv_((char const *)keys + (position_first + position_idx) * key_stride +
                                              key_value_head_idx * depth * sizeof(nk_i8_t),
                                          keys_plane + position_idx * depth * sizeof(nk_i8_t), depth);
            nk_attention_pack_row_i8_rvv_((char const *)values + (position_first + position_idx) * value_stride +
                                              key_value_head_idx * depth * sizeof(nk_i8_t),
                                          values_plane + position_idx * depth * sizeof(nk_i8_t), depth);
        }
    }
}

NUMKONG_INLINE void nk_attention_zero_row_rvv_(nk_f32_t *output_row, nk_size_t depth) {
    for (nk_size_t channel_idx = 0, remaining = depth, vector_length = 0; remaining > 0;
         remaining -= vector_length, channel_idx += vector_length) {
        vector_length = __riscv_vsetvl_e32m2(remaining);
        __riscv_vse32_v_f32m2(output_row + channel_idx, __riscv_vfmv_v_f_f32m2(0.0f, vector_length), vector_length);
    }
}

NUMKONG_INLINE void nk_attention_scale_row_rvv_(nk_f32_t *output_row, nk_size_t depth, nk_f32_t factor) {
    for (nk_size_t channel_idx = 0, remaining = depth, vector_length = 0; remaining > 0;
         remaining -= vector_length, channel_idx += vector_length) {
        vector_length = __riscv_vsetvl_e32m2(remaining);
        __riscv_vse32_v_f32m2(output_row + channel_idx,
                              __riscv_vfmul_vf_f32m2(__riscv_vle32_v_f32m2(output_row + channel_idx, vector_length),
                                                     factor, vector_length),
                              vector_length);
    }
}

/** Panel max, online correction, output rescale and exp2 in place for one query row. */
NUMKONG_INLINE void nk_attention_softmax_panel_rvv_(nk_f32_t *scores, nk_size_t panel_length, nk_f32_t scale2,
                                                    nk_f32_t *output_row, nk_size_t depth, nk_f32_t *running_max2,
                                                    nk_f32_t *running_sum) {
    nk_size_t const vlmax4 = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t max_f32m4 = __riscv_vfmv_v_f_f32m4(NUMKONG_F32_MIN, vlmax4);
    for (nk_size_t position_idx = 0, remaining = panel_length, vector_length = 0; remaining > 0;
         remaining -= vector_length, position_idx += vector_length) {
        vector_length = __riscv_vsetvl_e32m4(remaining);
        max_f32m4 = __riscv_vfmax_vv_f32m4_tu(
            max_f32m4, max_f32m4, __riscv_vle32_v_f32m4(scores + position_idx, vector_length), vector_length);
    }
    nk_f32_t const panel_max2 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1(
                                    max_f32m4, __riscv_vfmv_v_f_f32m1(NUMKONG_F32_MIN, 1), vlmax4)) *
                                scale2;
    nk_f32_t const new_max2 = *running_max2 > panel_max2 ? *running_max2 : panel_max2;
    nk_f32_t const correction = nk_f32_exp2_serial_(*running_max2 - new_max2);
    *running_max2 = new_max2;
    nk_attention_scale_row_rvv_(output_row, depth, correction);

    vfloat32m4_t sum_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax4);
    for (nk_size_t position_idx = 0, remaining = panel_length, vector_length = 0; remaining > 0;
         remaining -= vector_length, position_idx += vector_length) {
        vector_length = __riscv_vsetvl_e32m4(remaining);
        vfloat32m4_t weights_f32m4 = nk_exp2_f32m4_rvv_(
            __riscv_vfsub_vf_f32m4(__riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(scores + position_idx, vector_length),
                                                          scale2, vector_length),
                                   new_max2, vector_length),
            vector_length);
        sum_f32m4 = __riscv_vfadd_vv_f32m4_tu(sum_f32m4, sum_f32m4, weights_f32m4, vector_length);
        __riscv_vse32_v_f32m4(scores + position_idx, weights_f32m4, vector_length);
    }
    nk_f32_t const panel_sum = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m4_f32m1(sum_f32m4, __riscv_vfmv_v_f_f32m1(0.0f, 1), vlmax4));
    *running_sum = *running_sum * correction + panel_sum;
}

/** Adds the panel's weights times its BF16 V rows into @p output_row. */
NUMKONG_INLINE void nk_attention_weighted_sum_rvv_(nk_f32_t *output_row, nk_f32_t const *scores,
                                                   char const *values_rows, nk_size_t panel_length, nk_size_t depth,
                                                   nk_size_t plane_row_bytes) {
    for (nk_size_t position_idx = 0; position_idx < panel_length; position_idx++) {
        nk_f32_t const weight = scores[position_idx];
        char const *values_row = values_rows + position_idx * plane_row_bytes;
        for (nk_size_t channel_idx = 0, remaining = depth, vector_length = 0; remaining > 0;
             remaining -= vector_length, channel_idx += vector_length) {
            vector_length = __riscv_vsetvl_e32m2(remaining);
            __riscv_vse32_v_f32m2(
                output_row + channel_idx,
                __riscv_vfmacc_vf_f32m2(
                    __riscv_vle32_v_f32m2(output_row + channel_idx, vector_length), weight,
                    nk_attention_load_bf16_f32m2_rvv_(values_row + channel_idx * sizeof(nk_bf16_t), vector_length),
                    vector_length),
                vector_length);
        }
    }
}

/** Scores of one panel of BF16 keys against the BF16 query, one KV row at a time. */
NUMKONG_INLINE void nk_attention_scores_panel_bf16_rvv_(char const *query_source, char const *keys_rows,
                                                        nk_f32_t *scores, nk_size_t panel_length, nk_size_t depth,
                                                        nk_size_t plane_row_bytes) {
    nk_size_t const vlmax2 = __riscv_vsetvlmax_e32m2();
    for (nk_size_t position_idx = 0; position_idx < panel_length; position_idx++) {
        char const *keys_row = keys_rows + position_idx * plane_row_bytes;
        vfloat32m2_t accumulator_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax2);
        for (nk_size_t channel_idx = 0, remaining = depth, vector_length = 0; remaining > 0;
             remaining -= vector_length, channel_idx += vector_length) {
            vector_length = __riscv_vsetvl_e32m2(remaining);
            accumulator_f32m2 = __riscv_vfmacc_vv_f32m2_tu(
                accumulator_f32m2,
                nk_attention_load_bf16_f32m2_rvv_(query_source + channel_idx * sizeof(nk_bf16_t), vector_length),
                nk_attention_load_bf16_f32m2_rvv_(keys_row + channel_idx * sizeof(nk_bf16_t), vector_length),
                vector_length);
        }
        scores[position_idx] = __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m2_f32m1(accumulator_f32m2, __riscv_vfmv_v_f_f32m1(0.0f, 1), vlmax2));
    }
}

/** Scores of one panel of BF16 keys against the E4M3 query, one KV row at a time. */
NUMKONG_INLINE void nk_attention_scores_panel_e4m3_rvv_(char const *query_source, char const *keys_rows,
                                                        nk_f32_t *scores, nk_size_t panel_length, nk_size_t depth,
                                                        nk_size_t plane_row_bytes) {
    nk_size_t const vlmax2 = __riscv_vsetvlmax_e32m2();
    for (nk_size_t position_idx = 0; position_idx < panel_length; position_idx++) {
        char const *keys_row = keys_rows + position_idx * plane_row_bytes;
        vfloat32m2_t accumulator_f32m2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax2);
        for (nk_size_t channel_idx = 0, remaining = depth, vector_length = 0; remaining > 0;
             remaining -= vector_length, channel_idx += vector_length) {
            vector_length = __riscv_vsetvl_e32m2(remaining);
            accumulator_f32m2 = __riscv_vfmacc_vv_f32m2_tu(
                accumulator_f32m2,
                nk_attention_load_e4m3_f32m2_rvv_(query_source + channel_idx * sizeof(nk_e4m3_t), vector_length),
                nk_attention_load_bf16_f32m2_rvv_(keys_row + channel_idx * sizeof(nk_bf16_t), vector_length),
                vector_length);
        }
        scores[position_idx] = __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m2_f32m1(accumulator_f32m2, __riscv_vfmv_v_f_f32m1(0.0f, 1), vlmax2));
    }
}

/** BF16 attention core over BF16 planes: per query row, panel-flash with an exact online
 *  correction; queries and K/V widen in-loop, softmax elementwise passes run at e32m4. */
NUMKONG_INLINE void nk_attention_packed_bf16_rvv_(                                                   //
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_f32_t *log_sum_exp, //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,                           //
    nk_u32_t const *query_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale,  //
    nk_diagonal_band_t band, nk_size_t tasks_begin, nk_size_t tasks_end) {

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->key_value_head_count == key_value_head_count &&
               key_value_head_count != 0 && head_count % key_value_head_count == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_size_t const plane_row_bytes = depth * sizeof(nk_bf16_t);
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_; // softmax(x) = softmax₂(x · log₂e)
    nk_size_t const panel_width = nk_attention_panel_rvv_k_;
    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;

    nk_align_(64) nk_f32_t scores[nk_attention_panel_rvv_k_];

    for (nk_size_t head_idx = 0; head_idx < head_count && tasks_begin < tasks_end; head_idx++) {
        nk_size_t const token_first = (tasks_begin + head_count - 1 - head_idx) / head_count;
        nk_size_t const token_end = (tasks_end + head_count - 1 - head_idx) / head_count;
        for (nk_size_t segment_idx = nk_attention_segment_of_serial_(query_offsets, segment_count, token_first);
             segment_idx < segment_count && query_offsets[segment_idx] < token_end; segment_idx++) {
            nk_size_t const query_first = query_offsets[segment_idx], query_end = query_offsets[segment_idx + 1];
            nk_size_t const row_begin = token_first > query_first ? token_first - query_first : 0;
            nk_size_t const row_end = (token_end < query_end ? token_end : query_end) - query_first;
            if (row_begin >= row_end) continue;
            nk_size_t const position_count = key_lengths[segment_idx];
            nk_i64_t const first_position = nk_attention_first_position_serial_(query_end - query_first,
                                                                                position_count);
            nk_size_t const plane_bytes = position_count * plane_row_bytes;
            char const *keys_plane = payload_base + payload_offsets[segment_idx] +
                                     (head_idx / head_group_size) * plane_bytes;
            char const *values_plane = keys_plane + key_value_head_count * plane_bytes;

            for (nk_size_t row_idx = row_begin; row_idx < row_end; row_idx++) {
                nk_size_t key_begin, key_end;
                nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row_idx, position_count, &key_begin,
                                            &key_end);
                char const *query_source = (char const *)queries + (query_first + row_idx) * query_stride +
                                           head_idx * depth * sizeof(nk_bf16_t);
                nk_size_t const token = query_first + row_idx;
                nk_f32_t *output_row = output + token * output_stride_floats + head_idx * depth;
                nk_attention_zero_row_rvv_(output_row, depth);
                nk_f32_t running_max2 = NUMKONG_F32_MIN, running_sum = 0;

                for (nk_size_t panel_start = key_begin; panel_start < key_end; panel_start += panel_width) {
                    nk_size_t const panel_length = (panel_start + panel_width <= key_end) ? panel_width
                                                                                          : (key_end - panel_start);
                    nk_attention_scores_panel_bf16_rvv_(query_source, keys_plane + panel_start * plane_row_bytes,
                                                        scores, panel_length, depth, plane_row_bytes);
                    nk_attention_softmax_panel_rvv_(scores, panel_length, scale2, output_row, depth, &running_max2,
                                                    &running_sum);
                    nk_attention_weighted_sum_rvv_(output_row, scores, values_plane + panel_start * plane_row_bytes,
                                                   panel_length, depth, plane_row_bytes);
                }

                nk_attention_scale_row_rvv_(output_row, depth, running_sum > 0 ? 1 / running_sum : 0.0f);
                if (log_sum_exp)
                    log_sum_exp[token * head_count + head_idx] = nk_attention_log_sum_exp_serial_(running_max2,
                                                                                                  running_sum);
            }
        }
    }
}

/** E4M3 attention core over BF16 planes: per query row, panel-flash with an exact online
 *  correction; queries and K/V widen in-loop, softmax elementwise passes run at e32m4. */
NUMKONG_INLINE void nk_attention_packed_e4m3_rvv_(                                                   //
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_f32_t *log_sum_exp, //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,                           //
    nk_u32_t const *query_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale,  //
    nk_diagonal_band_t band, nk_size_t tasks_begin, nk_size_t tasks_end) {

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->key_value_head_count == key_value_head_count &&
               key_value_head_count != 0 && head_count % key_value_head_count == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_size_t const plane_row_bytes = depth * sizeof(nk_bf16_t);
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_; // softmax(x) = softmax₂(x · log₂e)
    nk_size_t const panel_width = nk_attention_panel_rvv_k_;
    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;

    nk_align_(64) nk_f32_t scores[nk_attention_panel_rvv_k_];

    for (nk_size_t head_idx = 0; head_idx < head_count && tasks_begin < tasks_end; head_idx++) {
        nk_size_t const token_first = (tasks_begin + head_count - 1 - head_idx) / head_count;
        nk_size_t const token_end = (tasks_end + head_count - 1 - head_idx) / head_count;
        for (nk_size_t segment_idx = nk_attention_segment_of_serial_(query_offsets, segment_count, token_first);
             segment_idx < segment_count && query_offsets[segment_idx] < token_end; segment_idx++) {
            nk_size_t const query_first = query_offsets[segment_idx], query_end = query_offsets[segment_idx + 1];
            nk_size_t const row_begin = token_first > query_first ? token_first - query_first : 0;
            nk_size_t const row_end = (token_end < query_end ? token_end : query_end) - query_first;
            if (row_begin >= row_end) continue;
            nk_size_t const position_count = key_lengths[segment_idx];
            nk_i64_t const first_position = nk_attention_first_position_serial_(query_end - query_first,
                                                                                position_count);
            nk_size_t const plane_bytes = position_count * plane_row_bytes;
            char const *keys_plane = payload_base + payload_offsets[segment_idx] +
                                     (head_idx / head_group_size) * plane_bytes;
            char const *values_plane = keys_plane + key_value_head_count * plane_bytes;

            for (nk_size_t row_idx = row_begin; row_idx < row_end; row_idx++) {
                nk_size_t key_begin, key_end;
                nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row_idx, position_count, &key_begin,
                                            &key_end);
                char const *query_source = (char const *)queries + (query_first + row_idx) * query_stride +
                                           head_idx * depth * sizeof(nk_e4m3_t);
                nk_size_t const token = query_first + row_idx;
                nk_f32_t *output_row = output + token * output_stride_floats + head_idx * depth;
                nk_attention_zero_row_rvv_(output_row, depth);
                nk_f32_t running_max2 = NUMKONG_F32_MIN, running_sum = 0;

                for (nk_size_t panel_start = key_begin; panel_start < key_end; panel_start += panel_width) {
                    nk_size_t const panel_length = (panel_start + panel_width <= key_end) ? panel_width
                                                                                          : (key_end - panel_start);
                    nk_attention_scores_panel_e4m3_rvv_(query_source, keys_plane + panel_start * plane_row_bytes,
                                                        scores, panel_length, depth, plane_row_bytes);
                    nk_attention_softmax_panel_rvv_(scores, panel_length, scale2, output_row, depth, &running_max2,
                                                    &running_sum);
                    nk_attention_weighted_sum_rvv_(output_row, scores, values_plane + panel_start * plane_row_bytes,
                                                   panel_length, depth, plane_row_bytes);
                }

                nk_attention_scale_row_rvv_(output_row, depth, running_sum > 0 ? 1 / running_sum : 0.0f);
                if (log_sum_exp)
                    log_sum_exp[token * head_count + head_idx] = nk_attention_log_sum_exp_serial_(running_max2,
                                                                                                  running_sum);
            }
        }
    }
}

NUMKONG_INLINE void nk_attention_packed_i8_rvv_(                                                    //
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_f32_t *log_sum_exp,  //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,                          //
    nk_u32_t const *query_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale, //
    nk_diagonal_band_t band, nk_size_t tasks_begin, nk_size_t tasks_end) {
    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->key_value_head_count == key_value_head_count &&
               key_value_head_count != 0 && head_count % key_value_head_count == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_; // softmax(x) = softmax₂(x · log₂e)
    nk_size_t const panel_width = nk_attention_panel_rvv_k_;
    nk_size_t const vlmax4 = __riscv_vsetvlmax_e32m4();
    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;

    nk_align_(64) nk_i32_t scores[nk_attention_panel_rvv_k_];

    for (nk_size_t head_idx = 0; head_idx < head_count && tasks_begin < tasks_end; head_idx++) {
        nk_size_t const token_first = (tasks_begin + head_count - 1 - head_idx) / head_count;
        nk_size_t const token_end = (tasks_end + head_count - 1 - head_idx) / head_count;
        for (nk_size_t segment_idx = nk_attention_segment_of_serial_(query_offsets, segment_count, token_first);
             segment_idx < segment_count && query_offsets[segment_idx] < token_end; segment_idx++) {
            nk_size_t const query_first = query_offsets[segment_idx], query_end = query_offsets[segment_idx + 1];
            nk_size_t const row_begin = token_first > query_first ? token_first - query_first : 0;
            nk_size_t const row_end = (token_end < query_end ? token_end : query_end) - query_first;
            if (row_begin >= row_end) continue;
            nk_size_t const position_count = key_lengths[segment_idx];
            nk_i64_t const first_position = nk_attention_first_position_serial_(query_end - query_first,
                                                                                position_count);
            nk_size_t const plane_bytes = position_count * depth;
            nk_i8_t const *keys_plane = (nk_i8_t const *)(payload_base + payload_offsets[segment_idx]) +
                                        (head_idx / head_group_size) * plane_bytes;
            nk_i8_t const *values_plane = keys_plane + key_value_head_count * plane_bytes;

            for (nk_size_t row_idx = row_begin; row_idx < row_end; row_idx++) {
                nk_size_t key_begin, key_end;
                nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row_idx, position_count, &key_begin,
                                            &key_end);
                nk_i8_t const *query_row = (nk_i8_t const *)((char const *)queries +
                                                             (query_first + row_idx) * query_stride) +
                                           head_idx * depth;
                nk_size_t const token = query_first + row_idx;
                nk_f32_t *output_row = output + token * output_stride_floats + head_idx * depth;
                for (nk_size_t channel_idx = 0, remaining = depth, vector_length = 0; remaining > 0;
                     remaining -= vector_length, channel_idx += vector_length) {
                    vector_length = __riscv_vsetvl_e32m2(remaining);
                    __riscv_vse32_v_f32m2(output_row + channel_idx, __riscv_vfmv_v_f_f32m2(0.0f, vector_length),
                                          vector_length);
                }
                nk_f32_t running_max2 = NUMKONG_F32_MIN, running_sum = 0;

                for (nk_size_t panel_start = key_begin; panel_start < key_end; panel_start += panel_width) {
                    nk_size_t const panel_length = (panel_start + panel_width <= key_end) ? panel_width
                                                                                          : (key_end - panel_start);
                    for (nk_size_t position_idx = 0; position_idx < panel_length; position_idx++) {
                        nk_i8_t const *keys_row = keys_plane + (panel_start + position_idx) * depth;
                        vint32m4_t accumulator_i32m4 = __riscv_vmv_v_x_i32m4(0, vlmax4);
                        for (nk_size_t channel_idx = 0, remaining = depth, vector_length = 0; remaining > 0;
                             remaining -= vector_length, channel_idx += vector_length) {
                            vector_length = __riscv_vsetvl_e8m1(remaining);
                            vint16m2_t product_i16m2 = __riscv_vwmul_vv_i16m2(
                                __riscv_vle8_v_i8m1(query_row + channel_idx, vector_length),
                                __riscv_vle8_v_i8m1(keys_row + channel_idx, vector_length), vector_length);
                            accumulator_i32m4 = __riscv_vwadd_wv_i32m4_tu(accumulator_i32m4, accumulator_i32m4,
                                                                          product_i16m2, vector_length);
                        }
                        scores[position_idx] = __riscv_vmv_x_s_i32m1_i32(
                            __riscv_vredsum_vs_i32m4_i32m1(accumulator_i32m4, __riscv_vmv_v_x_i32m1(0, 1), vlmax4));
                    }

                    vfloat32m4_t max_f32m4 = __riscv_vfmv_v_f_f32m4(NUMKONG_F32_MIN, vlmax4);
                    for (nk_size_t position_idx = 0, remaining = panel_length, vector_length = 0; remaining > 0;
                         remaining -= vector_length, position_idx += vector_length) {
                        vector_length = __riscv_vsetvl_e32m4(remaining);
                        vfloat32m4_t scaled_f32m4 = __riscv_vfmul_vf_f32m4(
                            __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4(scores + position_idx, vector_length),
                                                      vector_length),
                            scale2, vector_length);
                        max_f32m4 = __riscv_vfmax_vv_f32m4_tu(max_f32m4, max_f32m4, scaled_f32m4, vector_length);
                    }
                    nk_f32_t const panel_max2 = __riscv_vfmv_f_s_f32m1_f32(
                        __riscv_vfredmax_vs_f32m4_f32m1(max_f32m4, __riscv_vfmv_v_f_f32m1(NUMKONG_F32_MIN, 1), vlmax4));
                    nk_f32_t const new_max2 = running_max2 > panel_max2 ? running_max2 : panel_max2;
                    nk_f32_t const correction = nk_f32_exp2_serial_(running_max2 - new_max2);
                    running_max2 = new_max2;
                    for (nk_size_t channel_idx = 0, remaining = depth, vector_length = 0; remaining > 0;
                         remaining -= vector_length, channel_idx += vector_length) {
                        vector_length = __riscv_vsetvl_e32m2(remaining);
                        __riscv_vse32_v_f32m2(
                            output_row + channel_idx,
                            __riscv_vfmul_vf_f32m2(__riscv_vle32_v_f32m2(output_row + channel_idx, vector_length),
                                                   correction, vector_length),
                            vector_length);
                    }

                    vuint32m4_t sum_u32m4 = __riscv_vmv_v_x_u32m4(0, vlmax4);
                    for (nk_size_t position_idx = 0, remaining = panel_length, vector_length = 0; remaining > 0;
                         remaining -= vector_length, position_idx += vector_length) {
                        vector_length = __riscv_vsetvl_e32m4(remaining);
                        vfloat32m4_t weights_f32m4 = nk_exp2_f32m4_rvv_(
                            __riscv_vfsub_vf_f32m4(
                                __riscv_vfmul_vf_f32m4(
                                    __riscv_vfcvt_f_x_v_f32m4(
                                        __riscv_vle32_v_i32m4(scores + position_idx, vector_length), vector_length),
                                    scale2, vector_length),
                                new_max2, vector_length),
                            vector_length);
                        vuint32m4_t weights_u32m4 = __riscv_vfcvt_rtz_xu_f_v_u32m4(
                            __riscv_vfmadd_vf_f32m4(weights_f32m4, 255.0f, __riscv_vfmv_v_f_f32m4(0.5f, vector_length),
                                                    vector_length),
                            vector_length);
                        sum_u32m4 = __riscv_vadd_vv_u32m4_tu(sum_u32m4, sum_u32m4, weights_u32m4, vector_length);
                        __riscv_vse32_v_i32m4(scores + position_idx, __riscv_vreinterpret_v_u32m4_i32m4(weights_u32m4),
                                              vector_length);
                    }
                    nk_f32_t const panel_sum = (nk_f32_t)__riscv_vmv_x_s_u32m1_u32(
                        __riscv_vredsum_vs_u32m4_u32m1(sum_u32m4, __riscv_vmv_v_x_u32m1(0, 1), vlmax4));

                    for (nk_size_t position_idx = 0; position_idx < panel_length; position_idx++) {
                        if (scores[position_idx] == 0) continue;
                        nk_f32_t const weight = (nk_f32_t)scores[position_idx];
                        nk_i8_t const *values_row = values_plane + (panel_start + position_idx) * depth;
                        for (nk_size_t channel_idx = 0, remaining = depth, vector_length = 0; remaining > 0;
                             remaining -= vector_length, channel_idx += vector_length) {
                            vector_length = __riscv_vsetvl_e32m2(remaining);
                            vfloat32m2_t value_f32m2 = __riscv_vfcvt_f_x_v_f32m2(
                                __riscv_vsext_vf4_i32m2(__riscv_vle8_v_i8mf2(values_row + channel_idx, vector_length),
                                                        vector_length),
                                vector_length);
                            __riscv_vse32_v_f32m2(
                                output_row + channel_idx,
                                __riscv_vfmacc_vf_f32m2(__riscv_vle32_v_f32m2(output_row + channel_idx, vector_length),
                                                        weight, value_f32m2, vector_length),
                                vector_length);
                        }
                    }
                    running_sum = running_sum * correction + panel_sum;
                }

                nk_f32_t const inverse_sum = running_sum > 0 ? 1 / running_sum : 0.0f;
                for (nk_size_t channel_idx = 0, remaining = depth, vector_length = 0; remaining > 0;
                     remaining -= vector_length, channel_idx += vector_length) {
                    vector_length = __riscv_vsetvl_e32m2(remaining);
                    __riscv_vse32_v_f32m2(
                        output_row + channel_idx,
                        __riscv_vfmul_vf_f32m2(__riscv_vle32_v_f32m2(output_row + channel_idx, vector_length),
                                               inverse_sum, vector_length),
                        vector_length);
                }
                if (log_sum_exp)
                    log_sum_exp[token * head_count + head_idx] = nk_attention_log_sum_exp_serial_(running_max2,
                                                                                                  running_sum / 255.0f);
            }
        }
    }
}

#if NUMKONG_TARGET_RVV

NUMKONG_API nk_status_t nk_attention_pack_size_bf16_rvv(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_size_t token_count, nk_size_t segment_count,
                                                        nk_size_t *bytes) {
    *bytes = nk_attention_pack_size_rvv_(key_value_head_count, depth, token_count, segment_count, sizeof(nk_bf16_t));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_rvv(void const *key_value_packed,
                                                           nk_size_t *key_value_head_count, nk_size_t *depth,
                                                           nk_size_t *segments, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_rvv_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_serial_(key_value_packed, key_value_head_count, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_rvv(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_size_t token_count, nk_size_t segment_count,
                                                        nk_size_t *bytes) {
    // E4M3 K/V packs as BF16
    *bytes = nk_attention_pack_size_rvv_(key_value_head_count, depth, token_count, segment_count, sizeof(nk_bf16_t));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_rvv(void const *key_value_packed,
                                                           nk_size_t *key_value_head_count, nk_size_t *depth,
                                                           nk_size_t *segments, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_rvv_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_serial_(key_value_packed, key_value_head_count, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_i8_rvv(nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_size_t token_count, nk_size_t segment_count,
                                                      nk_size_t *bytes) {
    *bytes = nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_i8_rvv(void const *key_value_packed, nk_size_t *key_value_head_count,
                                                         nk_size_t *depth, nk_size_t *segments, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_rvv_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_serial_(key_value_packed, key_value_head_count, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_bf16_rvv(                                                  //
    nk_bf16_t const *keys, nk_bf16_t const *values, nk_size_t key_value_head_count, nk_size_t depth, //
    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths, nk_size_t segment_count, nk_size_t key_stride,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_status_t const validation = nk_attention_pack_validate_serial_(key_value_head_count, depth, key_offsets,
                                                                      key_lengths, segment_count);
    if (validation != nk_success_k) return validation;
    nk_attention_pack_bf16_rvv_(keys, values, key_value_head_count, depth, key_offsets, key_lengths, segment_count,
                                key_stride, value_stride, key_value_packed, tasks_begin, tasks_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_e4m3_rvv(                                                  //
    nk_e4m3_t const *keys, nk_e4m3_t const *values, nk_size_t key_value_head_count, nk_size_t depth, //
    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths, nk_size_t segment_count, nk_size_t key_stride,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_status_t const validation = nk_attention_pack_validate_serial_(key_value_head_count, depth, key_offsets,
                                                                      key_lengths, segment_count);
    if (validation != nk_success_k) return validation;
    nk_attention_pack_e4m3_rvv_(keys, values, key_value_head_count, depth, key_offsets, key_lengths, segment_count,
                                key_stride, value_stride, key_value_packed, tasks_begin, tasks_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_i8_rvv(                                                //
    nk_i8_t const *keys, nk_i8_t const *values, nk_size_t key_value_head_count, nk_size_t depth, //
    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths, nk_size_t segment_count, nk_size_t key_stride,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_status_t const validation = nk_attention_pack_validate_serial_(key_value_head_count, depth, key_offsets,
                                                                      key_lengths, segment_count);
    if (validation != nk_success_k) return validation;
    nk_attention_pack_i8_rvv_(keys, values, key_value_head_count, depth, key_offsets, key_lengths, segment_count,
                              key_stride, value_stride, key_value_packed, tasks_begin, tasks_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_bf16_rvv(                            //
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output,    //
    nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count, //
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride,      //
    nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before,              //
    nk_size_t keys_after, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_rvv_k)) return nk_pack_mismatch_k;
    nk_diagonal_band_t const band = {keys_before, keys_after};
    nk_attention_packed_bf16_rvv_(queries, key_value_packed, output, log_sum_exp, head_count, key_value_head_count,
                                  depth, query_offsets, query_stride, output_stride, scale, band, tasks_begin,
                                  tasks_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_e4m3_rvv(                            //
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output,    //
    nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count, //
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride,      //
    nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before,              //
    nk_size_t keys_after, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_rvv_k)) return nk_pack_mismatch_k;
    nk_diagonal_band_t const band = {keys_before, keys_after};
    // queries load as E4M3 through a LUT gather, the packed K/V planes as BF16 through shifts
    nk_attention_packed_e4m3_rvv_(queries, key_value_packed, output, log_sum_exp, head_count, key_value_head_count,
                                  depth, query_offsets, query_stride, output_stride, scale, band, tasks_begin,
                                  tasks_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_i8_rvv(                              //
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output,      //
    nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count, //
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride,      //
    nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before,              //
    nk_size_t keys_after, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_rvv_k)) return nk_pack_mismatch_k;
    nk_diagonal_band_t const band = {keys_before, keys_after};
    nk_attention_packed_i8_rvv_(queries, key_value_packed, output, log_sum_exp, head_count, key_value_head_count, depth,
                                query_offsets, query_stride, output_stride, scale, band, tasks_begin, tasks_end);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_RVV

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_RISCV64_RVV_
#endif // NUMKONG_ARCH_RISCV64_
#endif // NUMKONG_ATTENTION_RVV_H
