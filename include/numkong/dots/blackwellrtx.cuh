/**
 *  @brief SIMD-accelerated Batched Dot Products for the NVIDIA compute capability 12.x family.
 *  @file include/numkong/dots/blackwellrtx.cuh
 *  @author Ash Vardanian
 *  @date September 22, 2026
 *
 *  @sa include/numkong/dots.h
 *  @sa include/numkong/dots/ampere.cuh
 *
 *  The Ampere tile, with each 32-byte sub-slab going to the tensor cores as it is. Float8 and Float6 take one
 *  `mma.m16n8k32.kind::f8f6f4` per 16 × 8 output, at twice the 16-bit rate; the MMA reads a Float6 code from the low 6
 *  bits of each byte, which is how `nk_e3m2_t` and `nk_e2m3_t` already store it. Float4 takes one block-scaled
 *  `mma.m16n8k64.kind::mxf4` with every scale at 2⁰, at four times the 16-bit rate, straight from packed nibble pairs.
 *  Only the 12.x family carries these instructions, so every other device pass traps.
 */
#ifndef NK_DOTS_BLACKWELLRTX_CUH
#define NK_DOTS_BLACKWELLRTX_CUH

#if NK_TARGET_BLACKWELLRTX

#include "numkong/dots/ampere.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Instructions

#if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1200 && \
    __CUDA_ARCH_FAMILY_SPECIFIC__ < 1300

NK_HELPER_DEVICE_INLINE void nk_mma_e5m2_blackwellrtx_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t b_first,
                                                       nk_u32_t b_second) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e5m2.e5m2.f32 " //
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(accumulator[0].f), "+f"(accumulator[1].f), "+f"(accumulator[2].f), "+f"(accumulator[3].f)
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b_first), "r"(b_second));
}

NK_HELPER_DEVICE_INLINE void nk_mma_e4m3_blackwellrtx_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t b_first,
                                                       nk_u32_t b_second) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 " //
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(accumulator[0].f), "+f"(accumulator[1].f), "+f"(accumulator[2].f), "+f"(accumulator[3].f)
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b_first), "r"(b_second));
}

NK_HELPER_DEVICE_INLINE void nk_mma_e3m2_blackwellrtx_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t b_first,
                                                       nk_u32_t b_second) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e3m2.e3m2.f32 " //
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(accumulator[0].f), "+f"(accumulator[1].f), "+f"(accumulator[2].f), "+f"(accumulator[3].f)
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b_first), "r"(b_second));
}

NK_HELPER_DEVICE_INLINE void nk_mma_e2m3_blackwellrtx_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t b_first,
                                                       nk_u32_t b_second) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m3.e2m3.f32 " //
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(accumulator[0].f), "+f"(accumulator[1].f), "+f"(accumulator[2].f), "+f"(accumulator[3].f)
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b_first), "r"(b_second));
}

/* One 16 × 8 × 64 step from nibble pairs, both block scales at 2⁰ so the products are the codes' own. */
NK_HELPER_DEVICE_INLINE void nk_mma_e2m1_blackwellrtx_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t b_first,
                                                       nk_u32_t b_second) {
    asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 " //
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, %10, {0, 0}, %10, {0, 0};\n"
                 : "+f"(accumulator[0].f), "+f"(accumulator[1].f), "+f"(accumulator[2].f), "+f"(accumulator[3].f)
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b_first), "r"(b_second), "r"(0x7F7F7F7Fu));
}

#else

NK_HELPER_DEVICE_INLINE void nk_mma_e5m2_blackwellrtx_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t b_first,
                                                       nk_u32_t b_second) {
    __trap();
}
NK_HELPER_DEVICE_INLINE void nk_mma_e4m3_blackwellrtx_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t b_first,
                                                       nk_u32_t b_second) {
    __trap();
}
NK_HELPER_DEVICE_INLINE void nk_mma_e3m2_blackwellrtx_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t b_first,
                                                       nk_u32_t b_second) {
    __trap();
}
NK_HELPER_DEVICE_INLINE void nk_mma_e2m3_blackwellrtx_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t b_first,
                                                       nk_u32_t b_second) {
    __trap();
}
NK_HELPER_DEVICE_INLINE void nk_mma_e2m1_blackwellrtx_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t b_first,
                                                       nk_u32_t b_second) {
    __trap();
}

#endif // __CUDA_ARCH_FAMILY_SPECIFIC__ in the 12.x family

#pragma endregion Instructions

#pragma region Multiplies

NK_HELPER_DEVICE_INLINE void nk_dots_e5m2_multiply_blackwellrtx_(nk_fui32_t accumulators[4][8][4],
                                                                 nk_u32_t const a[4][4], nk_u32_t const b[8][2]) {
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
            nk_mma_e5m2_blackwellrtx_(accumulators[row_tile][column_tile], a[row_tile], b[column_tile][0],
                                      b[column_tile][1]);
}

NK_HELPER_DEVICE_INLINE void nk_dots_e4m3_multiply_blackwellrtx_(nk_fui32_t accumulators[4][8][4],
                                                                 nk_u32_t const a[4][4], nk_u32_t const b[8][2]) {
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
            nk_mma_e4m3_blackwellrtx_(accumulators[row_tile][column_tile], a[row_tile], b[column_tile][0],
                                      b[column_tile][1]);
}

NK_HELPER_DEVICE_INLINE void nk_dots_e3m2_multiply_blackwellrtx_(nk_fui32_t accumulators[4][8][4],
                                                                 nk_u32_t const a[4][4], nk_u32_t const b[8][2]) {
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
            nk_mma_e3m2_blackwellrtx_(accumulators[row_tile][column_tile], a[row_tile], b[column_tile][0],
                                      b[column_tile][1]);
}

NK_HELPER_DEVICE_INLINE void nk_dots_e2m3_multiply_blackwellrtx_(nk_fui32_t accumulators[4][8][4],
                                                                 nk_u32_t const a[4][4], nk_u32_t const b[8][2]) {
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
            nk_mma_e2m3_blackwellrtx_(accumulators[row_tile][column_tile], a[row_tile], b[column_tile][0],
                                      b[column_tile][1]);
}

NK_HELPER_DEVICE_INLINE void nk_dots_e2m1_multiply_blackwellrtx_(nk_fui32_t accumulators[4][8][4],
                                                                 nk_u32_t const a[4][4], nk_u32_t const b[8][2]) {
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 8; ++column_tile)
            nk_mma_e2m1_blackwellrtx_(accumulators[row_tile][column_tile], a[row_tile], b[column_tile][0],
                                      b[column_tile][1]);
}

#pragma endregion Multiplies

#pragma region E5M2

nk_define_cross_pack_size_(dots, e5m2, blackwellrtx, e5m2, e5m2, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_cuda_packed_shape_(dots, e5m2, blackwellrtx)
nk_define_cross_cuda_pack_(dots, e5m2, blackwellrtx, e5m2, e5m2, nk_load_b8_ampere_, /*norm_value_type=*/f32,
                           nk_dots_reduce_sumsq_e5m2_ampere_, /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_cuda_symmetric_(dots, e5m2, blackwellrtx, e5m2, f32, nk_dots_e5m2_multiply_blackwellrtx_,
                                nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, /*dimensions_per_value=*/1)
nk_define_cross_cuda_packed_(dots, e5m2, blackwellrtx, e5m2, e5m2, f32, nk_dots_e5m2_multiply_blackwellrtx_,
                             nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, /*depth_simd_dimensions=*/64,
                             /*dimensions_per_value=*/1)

#pragma endregion E5M2

#pragma region E4M3

nk_define_cross_pack_size_(dots, e4m3, blackwellrtx, e4m3, e4m3, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_cuda_packed_shape_(dots, e4m3, blackwellrtx)
nk_define_cross_cuda_pack_(dots, e4m3, blackwellrtx, e4m3, e4m3, nk_load_b8_ampere_, /*norm_value_type=*/f32,
                           nk_dots_reduce_sumsq_e4m3_ampere_, /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_cuda_symmetric_(dots, e4m3, blackwellrtx, e4m3, f32, nk_dots_e4m3_multiply_blackwellrtx_,
                                nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, /*dimensions_per_value=*/1)
nk_define_cross_cuda_packed_(dots, e4m3, blackwellrtx, e4m3, e4m3, f32, nk_dots_e4m3_multiply_blackwellrtx_,
                             nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, /*depth_simd_dimensions=*/64,
                             /*dimensions_per_value=*/1)

#pragma endregion E4M3

#pragma region E3M2

nk_define_cross_pack_size_(dots, e3m2, blackwellrtx, e3m2, e3m2, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_cuda_packed_shape_(dots, e3m2, blackwellrtx)
nk_define_cross_cuda_pack_(dots, e3m2, blackwellrtx, e3m2, e3m2, nk_load_b8_ampere_, /*norm_value_type=*/f32,
                           nk_dots_reduce_sumsq_e3m2_ampere_, /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_cuda_symmetric_(dots, e3m2, blackwellrtx, e3m2, f32, nk_dots_e3m2_multiply_blackwellrtx_,
                                nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, /*dimensions_per_value=*/1)
nk_define_cross_cuda_packed_(dots, e3m2, blackwellrtx, e3m2, e3m2, f32, nk_dots_e3m2_multiply_blackwellrtx_,
                             nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, /*depth_simd_dimensions=*/64,
                             /*dimensions_per_value=*/1)

#pragma endregion E3M2

#pragma region E2M3

nk_define_cross_pack_size_(dots, e2m3, blackwellrtx, e2m3, e2m3, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_cuda_packed_shape_(dots, e2m3, blackwellrtx)
nk_define_cross_cuda_pack_(dots, e2m3, blackwellrtx, e2m3, e2m3, nk_load_b8_ampere_, /*norm_value_type=*/f32,
                           nk_dots_reduce_sumsq_e2m3_ampere_, /*depth_simd_dimensions=*/64, /*dimensions_per_value=*/1)
nk_define_cross_cuda_symmetric_(dots, e2m3, blackwellrtx, e2m3, f32, nk_dots_e2m3_multiply_blackwellrtx_,
                                nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, /*dimensions_per_value=*/1)
nk_define_cross_cuda_packed_(dots, e2m3, blackwellrtx, e2m3, e2m3, f32, nk_dots_e2m3_multiply_blackwellrtx_,
                             nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, /*depth_simd_dimensions=*/64,
                             /*dimensions_per_value=*/1)

#pragma endregion E2M3

#pragma region E2M1

nk_define_cross_pack_size_(dots, e2m1, blackwellrtx, e2m1x2, e2m1x2, /*norm_value_type=*/f32,
                           /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/2)
nk_define_cross_cuda_packed_shape_(dots, e2m1, blackwellrtx)
nk_define_cross_cuda_pack_(dots, e2m1, blackwellrtx, e2m1x2, e2m1x2, nk_load_b8_ampere_, /*norm_value_type=*/f32,
                           nk_dots_reduce_sumsq_e2m1_ampere_, /*depth_simd_dimensions=*/128,
                           /*dimensions_per_value=*/2)
nk_define_cross_cuda_symmetric_(dots, e2m1, blackwellrtx, e2m1x2, f32, nk_dots_e2m1_multiply_blackwellrtx_,
                                nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, /*dimensions_per_value=*/2)
nk_define_cross_cuda_packed_(dots, e2m1, blackwellrtx, e2m1x2, e2m1x2, f32, nk_dots_e2m1_multiply_blackwellrtx_,
                             nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, /*depth_simd_dimensions=*/128,
                             /*dimensions_per_value=*/2)

#pragma endregion E2M1

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_BLACKWELLRTX
#endif // NK_DOTS_BLACKWELLRTX_CUH
