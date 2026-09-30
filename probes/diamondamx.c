/**
 *  @file probes/diamondamx.c
 *  @author Ash Vardanian
 *  @date July 7, 2026
 *  @brief Whether the toolchain builds the Diamond Rapids AMX kernels, AMX-FP8 plus AMX-AVX512.
 */
#define NUMKONG_HEADER_ONLY       1
#define NUMKONG_TARGET_DIAMONDAMX 1
#include <numkong/types.h>
#include <numkong/attention/diamondamx.h> // `nk_attention_bidirectional_packed_e4m3_diamondamx`

int main(void) {
    static nk_e4m3_t tokens[16 * 64];
    static nk_b512_vec_t packed[1024];
    static nk_f32_t output[16 * 64];
    nk_u32_t offsets[2] = {0, 16}, lengths[1] = {16};
    nk_size_t bytes = 0;
    nk_attention_pack_size_e4m3_diamondamx(1, 64, lengths, 1, &bytes);
    if (bytes > sizeof(packed)) return 1;
    nk_attention_pack_e4m3_diamondamx(tokens, tokens, 1, 64, offsets, lengths, 1, 64, 64, packed, 0, 1, 0);
    return nk_attention_bidirectional_packed_e4m3_diamondamx(tokens, packed, output, 1, 1, 64, offsets, 64,
                                                             64 * sizeof(nk_f32_t), 0.125f, 0, 1, 0) != nk_success_k;
}
