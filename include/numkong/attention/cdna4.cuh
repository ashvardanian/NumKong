/**
 *  @file include/numkong/attention/cdna4.cuh
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Ragged attention for AMD Instinct MI350, gfx950.
 *
 *  @sa include/numkong/attention.h
 *  @sa include/numkong/attention/cdna3.cuh
 *  @sa include/numkong/dots/cdna4.cuh
 *
 *  The CDNA3 kernel with MI350's wider MFMAs, one per 64-byte depth step and per P · V step: BF16
 *  multiplies BF16, E4M3 scores take the native Float8 MFMA and its P · V runs on F16 with V cast
 *  exactly, and I8 runs exact integer MFMA with U8 probabilities offset into I8. Depths above 256
 *  fall back to the @c rocm kernel. The pack layout and the work scheduler are the @c rocm
 *  capability's; the F16 conversions serve the CDNA5 kernel too.
 */
#ifndef NUMKONG_ATTENTION_CDNA4_CUH
#define NUMKONG_ATTENTION_CDNA4_CUH

#if NUMKONG_ARCH_ROCM_
#if NUMKONG_ARCH_ROCM_CDNA4_

#include "numkong/attention/cdna3.cuh" // `nk_attention_tile_cdna3_`, `nk_attention_launch_cdna3_`
#include "numkong/dots/cdna4.cuh"      // `nk_mfma_bf16_cdna4_`, `nk_byte_permute_rocm_`

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Conversions

NUMKONG_DEVICE nk_u32_t nk_f32_to_f16_cdna4_(nk_f32_t value) { return __half_as_ushort(__float2half_rn(value)); }

/** Four E4M3 codes as two pairs of F16 patterns scaled by 2⁻⁸: each code lands in the high byte
 *  of a half and keeps only its fields. */
NUMKONG_DEVICE void nk_e4m3x4_to_f16x4_cdna4_(nk_u32_t codes, nk_u32_t *low, nk_u32_t *high) {
    nk_u32_t const low_halves = nk_byte_permute_rocm_(0, codes, 0x010C000Cu);
    nk_u32_t const high_halves = nk_byte_permute_rocm_(0, codes, 0x030C020Cu);
    // The NaN magnitude 0x7F shifts to a finite 0x3F80, and setting 0x4000 fills its exponent.
    nk_u32_t const nan_bytes = ((codes & 0x7F7F7F7Fu) + 0x01010101u) & 0x80808080u;
    *low = ((low_halves >> 1) & 0x3F803F80u) | (low_halves & 0x80008000u) |
           (nk_byte_permute_rocm_(0, nan_bytes, 0x010C000Cu) >> 1);
    *high = ((high_halves >> 1) & 0x3F803F80u) | (high_halves & 0x80008000u) |
            (nk_byte_permute_rocm_(0, nan_bytes, 0x030C020Cu) >> 1);
}

#pragma endregion Conversions

#if NUMKONG_TARGET_CDNA4

#pragma region Steps

NUMKONG_DEVICE void nk_attention_scores_bf16_cdna4_(nk_fui32_t scores[4], nk_u32_t const keys[8],
                                                    nk_u32_t const queries[8]) {
    nk_mfma_bf16_cdna4_(scores, keys, queries);
}

NUMKONG_DEVICE void nk_attention_scores_e4m3_cdna4_(nk_fui32_t scores[4], nk_u32_t const keys[8],
                                                    nk_u32_t const queries[8]) {
    nk_mfma_e4m3_cdna4_(scores, keys, queries);
}

NUMKONG_DEVICE void nk_attention_scores_i8_cdna4_(nk_fui32_t scores[4], nk_u32_t const keys[8],
                                                  nk_u32_t const queries[8]) {
    nk_mfma_i8_cdna4_(scores, keys, queries);
}

NUMKONG_DEVICE void nk_attention_values_bf16_cdna4_(nk_fui32_t output[4], nk_u32_t const probabilities[4],
                                                    nk_u32_t const values[4]) {
    nk_mfma_bf16_cdna4_(output, probabilities, values);
}

/** F16 weights against 8 E4M3 codes of V converted to F16 over 256, which the output scale
 *  undoes. */
NUMKONG_DEVICE void nk_attention_values_e4m3_cdna4_(nk_fui32_t output[4], nk_u32_t const probabilities[4],
                                                    nk_u32_t const values[4]) {
    nk_u32_t halves[4];
    nk_e4m3x4_to_f16x4_cdna4_(values[0], &halves[0], &halves[1]);
    nk_e4m3x4_to_f16x4_cdna4_(values[1], &halves[2], &halves[3]);
    nk_mfma_f16_cdna4_(output, probabilities, halves);
}

NUMKONG_DEVICE void nk_attention_values_i8_cdna4_(nk_fui32_t output[4], nk_u32_t const probabilities[4],
                                                  nk_u32_t const values[4]) {
    nk_mfma_i8_cdna4_(output, probabilities, values);
}

NUMKONG_DEVICE void nk_attention_weights_f16_cdna4_(nk_f32_t const probabilities[16], nk_u32_t packed[4],
                                                    nk_f32_t *sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) {
        nk_u32_t const low = nk_f32_to_f16_cdna4_(probabilities[word * 2]);
        nk_u32_t const high = nk_f32_to_f16_cdna4_(probabilities[word * 2 + 1]);
        packed[word] = low | (high << 16);
        *sum += __half2float(__ushort_as_half((unsigned short)low)) +
                __half2float(__ushort_as_half((unsigned short)high));
    }
}

#pragma endregion Steps

#pragma region Instantiations

nk_define_attention_pack_size_simt_(bf16, cdna4, 2)
nk_define_attention_packed_shape_rocm_(bf16, cdna4)
nk_define_attention_pack_rocm_(bf16, cdna4, bf16)
nk_define_attention_packed_rocm_(bf16, cdna4, cdna3, nk_attention_launch_cdna3_, bf16, nk_cross_epilogue_f32_k,
                                 nk_attention_scores_bf16_cdna4_, nk_attention_values_bf16_cdna4_,
                                 nk_attention_weights_bf16_cdna3_, 1.0f, 1.0f)
nk_define_attention_backward_rocm_(bf16, cdna4, bf16)

nk_define_attention_pack_size_simt_(e4m3, cdna4, 1)
nk_define_attention_packed_shape_rocm_(e4m3, cdna4)
nk_define_attention_pack_rocm_(e4m3, cdna4, e4m3)
nk_define_attention_packed_rocm_(e4m3, cdna4, cdna3, nk_attention_launch_cdna3_, e4m3, nk_cross_epilogue_f32_k,
                                 nk_attention_scores_e4m3_cdna4_, nk_attention_values_e4m3_cdna4_,
                                 nk_attention_weights_f16_cdna4_, 1.0f, 256.0f)

nk_define_attention_pack_size_simt_(i8, cdna4, 1)
nk_define_attention_packed_shape_rocm_(i8, cdna4)
nk_define_attention_pack_rocm_(i8, cdna4, i8)
nk_define_attention_packed_rocm_(i8, cdna4, cdna3, nk_attention_launch_cdna3_, i8, nk_cross_epilogue_i32_to_f32_k,
                                 nk_attention_scores_i8_cdna4_, nk_attention_values_i8_cdna4_,
                                 nk_attention_weights_u8_cdna3_, 1.0f, 1.0f)

#pragma endregion Instantiations

#endif // NUMKONG_TARGET_CDNA4

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_ROCM_CDNA4_
#endif // NUMKONG_ARCH_ROCM_
#endif // NUMKONG_ATTENTION_CDNA4_CUH
