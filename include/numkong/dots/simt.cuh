/**
 *  @file include/numkong/dots/simt.cuh
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Batched Dot Products on the SIMT cores of every CUDA and ROCm device.
 *
 *  @sa include/numkong/dots.h
 *
 *  The baseline every other GPU capability stands on, as serial is on the CPU: no matrix unit, only
 *  each vendor's scalar and dot-product instructions. The same source compiles under NVCC into the
 *  @c cuda capability and under HIP-Clang into the @c rocm capability. F64 inputs sum in Dot2 and
 *  F32 inputs in F64 on the F64 tile. Every narrower type folds 32-bit words on the B32 tile:
 *  integers into wrapping I32 or U32 like the serial backends, through @c dp4a on NVIDIA and
 *  @c v_dot4 or @c v_dot8 on AMD, and the 16-bit and narrower floats into F32, through FMA on
 *  NVIDIA and @c v_dot2_f32_f16 pairs on AMD. Each of 256 threads owns a 4 × 4 grid of one 64 × 64
 *  output tile, over 16-word slabs staged in shared memory. The pack stores rows as they are,
 *  padded to 16 bytes, with the serial backends' norm types.
 *
 *  Every tensor tile of the other capabilities builds on the contract, runtime and generators here,
 *  so the file runs from host code to device code, then to the baseline tiles and their kernels.
 */
#ifndef NUMKONG_DOTS_SIMT_CUH
#define NUMKONG_DOTS_SIMT_CUH

#if NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_

#include "numkong/dots/serial.h"
#include "numkong/cast/simt.cuh" // `nk_e4m3_to_f32_simt_`, `nk_f32_to_f16_simt_`

/*  AMD dot instructions per device pass: `dot1-insts` multiply signed codes, `dot7-insts` unsigned
 *  ones, `dot8-insts` either, and `dot10-insts` F16 pairs into F32; other targets fold portably. */
#if defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__) || \
    defined(__gfx1011__) || defined(__gfx1012__) || defined(__gfx1030__) || defined(__gfx1031__) ||                    \
    defined(__gfx1032__) || defined(__gfx1033__) || defined(__gfx1034__) || defined(__gfx1035__) ||                    \
    defined(__gfx1036__)
#define NUMKONG_HAS_ROCM_SDOT_ 1
#else
#define NUMKONG_HAS_ROCM_SDOT_ 0
#endif
#if defined(__GFX11__) || defined(__GFX12__)
#define NUMKONG_HAS_ROCM_SUDOT_ 1
#else
#define NUMKONG_HAS_ROCM_SUDOT_ 0
#endif
#if NUMKONG_HAS_ROCM_SDOT_ || NUMKONG_HAS_ROCM_SUDOT_
#define NUMKONG_HAS_ROCM_UDOT_ 1
#else
#define NUMKONG_HAS_ROCM_UDOT_ 0
#endif
#if NUMKONG_HAS_ROCM_SDOT_ || defined(__GFX11__) || defined(__gfx1200__) || defined(__gfx1201__)
#define NUMKONG_HAS_ROCM_FDOT2_ 1
#else
#define NUMKONG_HAS_ROCM_FDOT2_ 0
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/*  The baseline tiles' launch shape, the same on every vendor. AMD blocks stay whole multiples of
 *  64 threads, so they hold whole wavefronts of either width. The grid side counts the threads
 *  along each side of a tile, and each block of a device pack holds 8 groups of 32 lanes. */
#pragma region Configuration

enum {
    nk_cross_threads_simt_k = 256,
    nk_cross_tile_simt_k = 64,
    nk_cross_thread_tile_simt_k = 4,
    nk_cross_slab_simt_k = 16,
    nk_cross_loads_simt_k = nk_cross_tile_simt_k * nk_cross_slab_simt_k / nk_cross_threads_simt_k,
    nk_cross_grid_side_simt_k = nk_cross_tile_simt_k / nk_cross_thread_tile_simt_k,
    // Both baseline tiles launch that shape, named after each tile for the generators' pastes.
    nk_cross_threads_simt_f64_k = nk_cross_threads_simt_k,
    nk_cross_tile_simt_f64_k = nk_cross_tile_simt_k,
    nk_cross_threads_simt_b32_k = nk_cross_threads_simt_k,
    nk_cross_tile_simt_b32_k = nk_cross_tile_simt_k,
    nk_cross_pack_groups_k = 8,
};

nk_static_assert_(nk_cross_grid_side_simt_k *nk_cross_grid_side_simt_k == nk_cross_threads_simt_k,
                  nk_cross_grid_is_square);
nk_static_assert_(nk_cross_threads_simt_k % 64 == 0, nk_cross_blocks_hold_whole_wavefronts);
nk_static_assert_(nk_cross_threads_simt_k % nk_cross_slab_simt_k == 0 && nk_cross_slab_simt_k <= 32,
                  nk_cross_slab_fits_a_warp);

/** How a tile accumulates, matching the serial backends, and how a B32 tile's word holds depth. */
typedef enum {

    /** One F64 FMA per product, for F32 inputs. */
    nk_cross_accumulation_f64_k,

    /** Ogita-Rump-Oishi Dot2: TwoProd and TwoSum, for F64 inputs. */
    nk_cross_accumulation_dot2_k,

    /** One element per word as F32, folded by one F32 FMA. */
    nk_cross_accumulation_f32_k,

    /** Two elements per word as F16, folded by one dot2 into F32. */
    nk_cross_accumulation_f16x2_k,

    /** Four signed bytes per word, folded by one dot4 into wrapping I32. */
    nk_cross_accumulation_i8x4_k,

    /** Four unsigned bytes per word, folded by one dot4 into wrapping U32. */
    nk_cross_accumulation_u8x4_k,

    /** Four signed nibbles widened to bytes, folded as @c i8x4. */
    nk_cross_accumulation_i4x4_k,

    /** Four unsigned nibbles widened to bytes, folded as @c u8x4. */
    nk_cross_accumulation_u4x4_k,

    /** Eight signed nibbles as packed, folded by one dot8 into wrapping I32. */
    nk_cross_accumulation_i4x8_k,

    /** Eight unsigned nibbles as packed, folded by one dot8 into wrapping U32. */
    nk_cross_accumulation_u4x8_k,
} nk_cross_accumulation_t;

/** Which outputs a tile writes. */
typedef enum {

    /** Every output, for @c packed. */
    nk_cross_triangle_full_k,

    /** The upper triangle with its diagonal, for @c symmetric. */
    nk_cross_triangle_upper_k,
} nk_cross_triangle_t;

/** What a tile turns each dot product into. */
typedef enum {

    /** The dot product itself. */
    nk_cross_metric_dot_k,

    /** 1 − dot / (‖a‖ ‖b‖), clamped at 0. */
    nk_cross_metric_angular_k,

    /** √(‖a‖² + ‖b‖² − 2 · dot), clamped at 0. */
    nk_cross_metric_euclidean_k,
} nk_cross_metric_t;

/** Everything one launch shares, passed by value as the kernels' only argument. */
typedef struct {

    /** Row-major A, or the vectors for @c symmetric. */
    unsigned char const *a;

    /** Packed B rows past the header, or the vectors again for @c symmetric. */
    unsigned char const *b;

    /** Row-major output, indexed by absolute row. */
    void *c;

    /** First output row. */
    nk_size_t row_start;

    /** One past the last output row. */
    nk_size_t row_end;

    /** Output columns, the rows of B. */
    nk_size_t column_count;

    /** Bytes of depth per row, past which A and B read as zeros. */
    nk_size_t depth_bytes;

    /** Elements of depth per row, which the baseline tiles count in. */
    nk_size_t depth;

    /** Bytes between rows of A. */
    nk_size_t a_stride;

    /** Bytes between rows of B. */
    nk_size_t b_stride;

    /** Bytes between rows of C. */
    nk_size_t c_stride;

    /** 64-byte slabs of depth, which the tensor tiles count in. */
    nk_size_t depth_slabs;

    /** Output tiles per row of tiles. */
    nk_size_t column_tiles;

    /** Output tiles in all, which the blocks walk with a stride of the grid. */
    nk_size_t tiles;

    /** Column norms past the packed rows, which only a @c packed metric reads. */
    void const *b_norms;
} nk_cross_tile_arguments_t;

/** What the accumulators hold and how they reach the output. */
typedef enum {

    /** F32 sums, stored times the output scale. */
    nk_cross_epilogue_f32_k,

    /** Integer sums, stored as they are. */
    nk_cross_epilogue_i32_k,

    /** Integer sums of scaled codes, converted and stored times the output scale. */
    nk_cross_epilogue_i32_to_f32_k,

    /** Integer sums of U8 codes offset by −128, restored from the byte sums. */
    nk_cross_epilogue_offset_u32_k,
} nk_cross_epilogue_t;

/** How squared norms are stored, which also fixes the precision a metric is computed in. */
typedef enum {

    /** F32 in true units. */
    nk_cross_norm_f32_k,

    /** F64. */
    nk_cross_norm_f64_k,

    /** Wrapping U32 sums read as I32. */
    nk_cross_norm_i32_k,

    /** Wrapping U32 sums. */
    nk_cross_norm_u32_k,
} nk_cross_norm_t;

/** Adds the squares of 16 staged bytes: exact integer codes into @p integer_sum, the others
 *  into @p real_sum. */
typedef void (*nk_cross_norm_update_t)(nk_u32_t const words[4], nk_u32_t *integer_sum, nk_f32_t *real_sum);

/** Splits one fragment register into two narrower-typed ones: 4 codes into F16 pairs, or 8 nibbles
 *  into I8 quads. */
typedef void (*nk_cross_widen_t)(nk_u32_t codes, nk_u32_t *low, nk_u32_t *high);

#pragma endregion Configuration

#pragma region Launchers

/** Storage values in one packed GPU row: @c nk_cross_padded_values_ without its power-of-two
 *  break, since the GPU loaders never read the padding and only lose bandwidth to it. */
NUMKONG_INLINE nk_size_t nk_device_cross_padded_values_(nk_size_t depth, nk_size_t depth_simd_dimensions,
                                                        nk_size_t dimensions_per_value, nk_size_t packed_value_bytes) {
    nk_unused_(packed_value_bytes);
    return nk_size_round_up_to_multiple_(depth, depth_simd_dimensions) / dimensions_per_value;
}

/** Validates the contract and launches as many blocks of @p kernel as stay resident, each walking
 *  @p tile × @p tile output tiles with a stride of the grid. @p b_norms holds the packed column
 *  norms a @c packed metric reads, or is null. */
NUMKONG_INLINE nk_status_t nk_cross_launch_(void const *kernel, unsigned tile, unsigned threads, void const *a,
                                            void const *b, void const *b_norms, void *c, nk_size_t result_bytes,
                                            nk_size_t row_start, nk_size_t row_end, nk_size_t column_count,
                                            nk_size_t depth, nk_size_t depth_bytes, nk_size_t a_stride,
                                            nk_size_t b_stride, nk_size_t c_stride, void *stream) {
    if ((((nk_size_t)a) | a_stride | ((nk_size_t)b) | b_stride) & 15 ||
        (((nk_size_t)c) | c_stride) & (result_bytes - 1))
        return nk_misaligned_k;
    if (row_end <= row_start || column_count == 0) return nk_success_k;
    nk_size_t const column_tiles = nk_size_divide_round_up_(column_count, tile);
    nk_size_t const tiles = nk_size_divide_round_up_(row_end - row_start, tile) * column_tiles;
    nk_cross_tile_arguments_t arguments;
    arguments.a = (unsigned char const *)a, arguments.b = (unsigned char const *)b, arguments.c = c;
    arguments.row_start = row_start, arguments.row_end = row_end, arguments.column_count = column_count;
    arguments.depth = depth, arguments.depth_bytes = depth_bytes, arguments.a_stride = a_stride;
    arguments.b_stride = b_stride;
    arguments.c_stride = c_stride, arguments.column_tiles = column_tiles, arguments.tiles = tiles;
    arguments.depth_slabs = nk_size_divide_round_up_(depth_bytes, 64);
    arguments.b_norms = b_norms;
    return nk_device_launch_resident_(kernel, threads, 0, 0, tiles, &arguments, stream);
}

/** Launches @p kernel with one 32-lane group per packed column, walked with a grid stride, which
 *  records the packing @p capability. */
NUMKONG_INLINE nk_status_t nk_cross_pack_launch_(void const *kernel, void const *b, nk_size_t column_count,
                                                 nk_size_t depth, nk_size_t depth_bytes, nk_size_t b_stride,
                                                 void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                                 nk_size_t depth_values_padded, nk_capability_t capability,
                                                 void *stream) {
    nk_size_t const columns = columns_end > columns_begin ? columns_end - columns_begin : 0;
    nk_size_t const needed = nk_size_divide_round_up_(columns, nk_cross_pack_groups_k);
    nk_size_t const blocks = needed == 0 ? 1 : needed < 65535 ? needed : 65535;
    void *arguments[10];
    arguments[0] = &b, arguments[1] = &column_count, arguments[2] = &depth, arguments[3] = &depth_bytes;
    arguments[4] = &b_stride, arguments[5] = &b_packed, arguments[6] = &columns_begin, arguments[7] = &columns_end;
    arguments[8] = &depth_values_padded, arguments[9] = &capability;
    return nk_device_launch_(kernel, blocks, nk_cross_pack_groups_k * 32, arguments, 0, stream);
}

#pragma endregion Launchers

#pragma region Primitives

/** Lanes of a warp or wavefront: 32 on NVIDIA, and on AMD 64 on CDNA and 32 on RDNA and MI400,
 *  which a build spanning both only learns per device pass. */
NUMKONG_DEVICE unsigned nk_warp_lanes_(void) {
#if NUMKONG_ARCH_ROCM_
    return __builtin_amdgcn_wavefrontsize();
#else
    return 32;
#endif
}

/*  The xor shuffles span the whole warp or wavefront, where offsets below 32 stay inside each half
 *  of a 64-lane one, and the up shuffle scans groups of 32 lanes. HIP takes no lane mask. */
#if NUMKONG_ARCH_ROCM_
NUMKONG_DEVICE nk_u32_t nk_shuffle_xor_u32_(nk_u32_t value, unsigned offset) { return __shfl_xor(value, offset); }
NUMKONG_DEVICE nk_i32_t nk_shuffle_xor_i32_(nk_i32_t value, unsigned offset) { return __shfl_xor(value, offset); }
NUMKONG_DEVICE nk_u64_t nk_shuffle_xor_u64_(nk_u64_t value, unsigned offset) {
    return __shfl_xor((unsigned long long)value, offset);
}
NUMKONG_DEVICE nk_f32_t nk_shuffle_xor_f32_(nk_f32_t value, unsigned offset) { return __shfl_xor(value, offset); }
NUMKONG_DEVICE nk_f64_t nk_shuffle_xor_f64_(nk_f64_t value, unsigned offset) { return __shfl_xor(value, offset); }
NUMKONG_DEVICE nk_u64_t nk_shuffle_up_u64_(nk_u64_t value, unsigned offset) {
    return __shfl_up((unsigned long long)value, offset, 32);
}
#else
NUMKONG_DEVICE nk_u32_t nk_shuffle_xor_u32_(nk_u32_t value, unsigned offset) {
    return __shfl_xor_sync(0xFFFFFFFFu, value, offset);
}
NUMKONG_DEVICE nk_i32_t nk_shuffle_xor_i32_(nk_i32_t value, unsigned offset) {
    return __shfl_xor_sync(0xFFFFFFFFu, value, offset);
}
NUMKONG_DEVICE nk_u64_t nk_shuffle_xor_u64_(nk_u64_t value, unsigned offset) {
    return __shfl_xor_sync(0xFFFFFFFFu, (unsigned long long)value, offset);
}
NUMKONG_DEVICE nk_f32_t nk_shuffle_xor_f32_(nk_f32_t value, unsigned offset) {
    return __shfl_xor_sync(0xFFFFFFFFu, value, offset);
}
NUMKONG_DEVICE nk_f64_t nk_shuffle_xor_f64_(nk_f64_t value, unsigned offset) {
    return __shfl_xor_sync(0xFFFFFFFFu, value, offset);
}
NUMKONG_DEVICE nk_u64_t nk_shuffle_up_u64_(nk_u64_t value, unsigned offset) {
    return __shfl_up_sync(0xFFFFFFFFu, (unsigned long long)value, offset);
}
#endif

/*  F64 arithmetic rounded one operation at a time, never contracted into an FMA, which would break
 *  Dot2's error terms. NVCC honors its rounding intrinsics, and HIP-Clang the pragma. */
#if NUMKONG_ARCH_ROCM_
NUMKONG_DEVICE nk_f64_t nk_f64_add_rn_(nk_f64_t a, nk_f64_t b) {
#pragma clang fp contract(off)
    return a + b;
}
NUMKONG_DEVICE nk_f64_t nk_f64_sub_rn_(nk_f64_t a, nk_f64_t b) {
#pragma clang fp contract(off)
    return a - b;
}
NUMKONG_DEVICE nk_f64_t nk_f64_mul_rn_(nk_f64_t a, nk_f64_t b) {
#pragma clang fp contract(off)
    return a * b;
}
#else
NUMKONG_DEVICE nk_f64_t nk_f64_add_rn_(nk_f64_t a, nk_f64_t b) { return __dadd_rn(a, b); }
NUMKONG_DEVICE nk_f64_t nk_f64_sub_rn_(nk_f64_t a, nk_f64_t b) { return __dsub_rn(a, b); }
NUMKONG_DEVICE nk_f64_t nk_f64_mul_rn_(nk_f64_t a, nk_f64_t b) { return __dmul_rn(a, b); }
#endif

/** Adds a × b into a running sum: one F64 FMA, or Dot2's TwoProd and TwoSum with both
 *  errors kept apart. */
NUMKONG_DEVICE void nk_cross_step_f64_(nk_cross_accumulation_t accumulation, nk_f64_t a, nk_f64_t b, nk_f64_t *sum,
                                       nk_f64_t *compensation) {
    if (accumulation == nk_cross_accumulation_f64_k) {
        *sum = __fma_rn(a, b, *sum);
        return;
    }
    nk_f64_t const product = nk_f64_mul_rn_(a, b), product_error = __fma_rn(a, b, -product);
    nk_f64_t const total = nk_f64_add_rn_(*sum, product), virtual_addend = nk_f64_sub_rn_(total, *sum);
    nk_f64_t const sum_error = nk_f64_add_rn_(nk_f64_sub_rn_(*sum, nk_f64_sub_rn_(total, virtual_addend)),
                                              nk_f64_sub_rn_(product, virtual_addend));
    *sum = total;
    *compensation = nk_f64_add_rn_(*compensation, nk_f64_add_rn_(sum_error, product_error));
}

/** Merges the running sum of lane `lane ^ offset` into this one, through TwoSum under Dot2. */
NUMKONG_DEVICE void nk_cross_merge_lanes_f64_(nk_cross_accumulation_t accumulation, unsigned offset, nk_f64_t *sum,
                                              nk_f64_t *compensation) {
    nk_f64_t const other_sum = nk_shuffle_xor_f64_(*sum, offset);
    if (accumulation == nk_cross_accumulation_f64_k) {
        *sum = nk_f64_add_rn_(*sum, other_sum);
        return;
    }
    nk_f64_t const other_compensation = nk_shuffle_xor_f64_(*compensation, offset);
    nk_f64_t const total = nk_f64_add_rn_(*sum, other_sum), virtual_addend = nk_f64_sub_rn_(total, *sum);
    nk_f64_t const sum_error = nk_f64_add_rn_(nk_f64_sub_rn_(*sum, nk_f64_sub_rn_(total, virtual_addend)),
                                              nk_f64_sub_rn_(other_sum, virtual_addend));
    *sum = total;
    *compensation = nk_f64_add_rn_(nk_f64_add_rn_(*compensation, other_compensation), sum_error);
}

/** Adds the four signed byte products of @p a and @p b to @p sum, wrapping like serial I32. */
NUMKONG_DEVICE nk_i32_t nk_dot_i8x4_(nk_u32_t a, nk_u32_t b, nk_i32_t sum) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    return __dp4a((int)a, (int)b, sum);
#elif NUMKONG_HAS_ROCM_SDOT_
    return __builtin_amdgcn_sdot4((int)a, (int)b, sum, 0);
#elif NUMKONG_HAS_ROCM_SUDOT_
    return __builtin_amdgcn_sudot4(1, (int)a, 1, (int)b, sum, 0);
#else
    nk_i32_t products = 0;
    for (unsigned shift = 0; shift < 32; shift += 8)
        products += (nk_i32_t)(signed char)(a >> shift) * (nk_i32_t)(signed char)(b >> shift);
    return (nk_i32_t)((nk_u32_t)sum + (nk_u32_t)products);
#endif
}

/** Adds the four unsigned byte products of @p a and @p b to @p sum, wrapping like serial U32. */
NUMKONG_DEVICE nk_u32_t nk_dot_u8x4_(nk_u32_t a, nk_u32_t b, nk_u32_t sum) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    return __dp4a(a, b, sum);
#elif NUMKONG_HAS_ROCM_UDOT_
    return __builtin_amdgcn_udot4(a, b, sum, 0);
#else
    for (unsigned shift = 0; shift < 32; shift += 8) sum += ((a >> shift) & 0xFFu) * ((b >> shift) & 0xFFu);
    return sum;
#endif
}

/** Adds the eight signed nibble products of @p a and @p b, paired by position, to @p sum. */
NUMKONG_DEVICE nk_i32_t nk_dot_i4x8_(nk_u32_t a, nk_u32_t b, nk_i32_t sum) {
#if NUMKONG_HAS_ROCM_SDOT_
    return __builtin_amdgcn_sdot8((int)a, (int)b, sum, 0);
#elif NUMKONG_HAS_ROCM_SUDOT_
    return __builtin_amdgcn_sudot8(1, (int)a, 1, (int)b, sum, 0);
#else
    nk_i32_t products = 0;
    for (unsigned shift = 0; shift < 32; shift += 4)
        products += ((nk_i32_t)(((a >> shift) & 0xFu) ^ 8u) - 8) * ((nk_i32_t)(((b >> shift) & 0xFu) ^ 8u) - 8);
    return (nk_i32_t)((nk_u32_t)sum + (nk_u32_t)products);
#endif
}

/** Adds the eight unsigned nibble products of @p a and @p b, paired by position, to @p sum. */
NUMKONG_DEVICE nk_u32_t nk_dot_u4x8_(nk_u32_t a, nk_u32_t b, nk_u32_t sum) {
#if NUMKONG_HAS_ROCM_UDOT_
    return __builtin_amdgcn_udot8(a, b, sum, 0);
#else
    for (unsigned shift = 0; shift < 32; shift += 4) sum += ((a >> shift) & 0xFu) * ((b >> shift) & 0xFu);
    return sum;
#endif
}

/** Adds both F16 products of @p a and @p b to @p sum, each exact in F32. */
NUMKONG_DEVICE nk_f32_t nk_dot_f16x2_(nk_u32_t a, nk_u32_t b, nk_f32_t sum) {
#if NUMKONG_HAS_ROCM_FDOT2_
    union {
        nk_u32_t bits;
        _Float16_2 halves;
    } const a_pair = {a}, b_pair = {b};
    return __builtin_amdgcn_fdot2(a_pair.halves, b_pair.halves, sum, 0);
#else
    nk_f32_t const a_low = __half2float(__ushort_as_half((unsigned short)(a & 0xFFFFu)));
    nk_f32_t const b_low = __half2float(__ushort_as_half((unsigned short)(b & 0xFFFFu)));
    nk_f32_t const a_high = __half2float(__ushort_as_half((unsigned short)(a >> 16)));
    nk_f32_t const b_high = __half2float(__ushort_as_half((unsigned short)(b >> 16)));
    return __fmaf_rn(a_high, b_high, __fmaf_rn(a_low, b_low, sum));
#endif
}

#pragma endregion Primitives

/*  Widenings every vendor's tiles share, then element decoders that widen one element exactly: F64
 *  and F32 to F64 for the F64 tile, and every narrower float to F32. The Float8 and Float6 ones go
 *  through F16, which holds E5M2 exactly, E4M3 over 256 and E3M2 over 4096. Nibble types hold
 *  element 2 × i in the high nibble of byte i and element 2 × i + 1 in the low one. 16-bit types
 *  read bytewise, since a pack's source row stride need not be a whole number of elements. */
#pragma region Conversions

/** Eight I4 nibbles become two registers of four sign-extended I8, each (v ^ 8) − 8 without
 *  cross-byte borrows. */
NUMKONG_DEVICE void nk_i4x8_to_i8x8_(nk_u32_t codes, nk_u32_t *low, nk_u32_t *high) {
    nk_u32_t const low_nibbles = codes & 0x0F0F0F0Fu, high_nibbles = (codes >> 4) & 0x0F0F0F0Fu;
    *low = (((low_nibbles ^ 0x08080808u) | 0x80808080u) - 0x08080808u) ^ 0x80808080u;
    *high = (((high_nibbles ^ 0x08080808u) | 0x80808080u) - 0x08080808u) ^ 0x80808080u;
}

/** Eight U4 nibbles become two registers of four U8. */
NUMKONG_DEVICE void nk_u4x8_to_u8x8_(nk_u32_t codes, nk_u32_t *low, nk_u32_t *high) {
    *low = codes & 0x0F0F0F0Fu, *high = (codes >> 4) & 0x0F0F0F0Fu;
}

/** Four E2M3 magnitudes times 8 as U8: @c m where e = 0, otherwise (8 + m) << (e − 1), at most
 *  60. */
NUMKONG_DEVICE nk_u32_t nk_e2m3x4_to_u8x4_magnitudes_(nk_u32_t codes) {
    nk_u32_t const exponent_low = (codes >> 3) & 0x01010101u, exponent_high = (codes >> 4) & 0x01010101u;
    nk_u32_t const significand = (codes & 0x07070707u) | ((exponent_low | exponent_high) << 3);
    nk_u32_t const doubled_mask = exponent_high * 0xFFu, quadrupled_mask = (exponent_high & exponent_low) * 0xFFu;
    return significand + (significand & doubled_mask) + ((significand << 1) & quadrupled_mask);
}

/** One E3M2 code over 4096. */
NUMKONG_DEVICE nk_f32_t nk_e3m2_to_scaled_f32_(nk_u32_t code) {
    return __half2float(__ushort_as_half((unsigned short)(((code & 0x1Fu) << 8) | ((code & 0x20u) << 10))));
}

/** Keeps a packed byte as it is. */
NUMKONG_DEVICE unsigned char nk_load_b8_(unsigned char value) { return value; }

NUMKONG_DEVICE nk_f64_t nk_f64_load_f64_(unsigned char const *row, nk_size_t index) {
    return ((nk_f64_t const *)row)[index];
}

NUMKONG_DEVICE nk_f64_t nk_f32_load_f64_(unsigned char const *row, nk_size_t index) {
    return (nk_f64_t)((nk_f32_t const *)row)[index];
}

NUMKONG_DEVICE nk_f32_t nk_bf16_load_f32_(unsigned char const *row, nk_size_t index) {
    return __uint_as_float(((nk_u32_t)row[index * 2] | ((nk_u32_t)row[index * 2 + 1] << 8)) << 16);
}

NUMKONG_DEVICE nk_f32_t nk_f16_load_f32_(unsigned char const *row, nk_size_t index) {
    return __half2float(__ushort_as_half((unsigned short)(row[index * 2] | (row[index * 2 + 1] << 8))));
}

NUMKONG_DEVICE nk_f32_t nk_e5m2_load_f32_(unsigned char const *row, nk_size_t index) {
    nk_f32_t value;
    nk_e5m2_to_f32_simt_(row + index, &value);
    return value;
}

NUMKONG_DEVICE nk_f32_t nk_e4m3_load_f32_(unsigned char const *row, nk_size_t index) {
    nk_f32_t value;
    nk_e4m3_to_f32_simt_(row + index, &value);
    return value;
}

NUMKONG_DEVICE nk_f32_t nk_e3m2_load_f32_(unsigned char const *row, nk_size_t index) {
    return nk_e3m2_to_scaled_f32_(row[index]) * 4096.0f;
}

NUMKONG_DEVICE nk_f32_t nk_e2m3_load_f32_(unsigned char const *row, nk_size_t index) {
    unsigned const code = row[index];
    nk_f32_t const magnitude = (nk_f32_t)nk_e2m3x4_to_u8x4_magnitudes_(code) * 0.125f;
    return code & 0x20u ? -magnitude : magnitude;
}

NUMKONG_DEVICE unsigned nk_b4_load_(unsigned char const *row, nk_size_t index) {
    return (index & 1) ? (row[index / 2] & 0x0Fu) : (row[index / 2] >> 4);
}

NUMKONG_DEVICE nk_f32_t nk_e2m1_load_f32_(unsigned char const *row, nk_size_t index) {
    unsigned const code = nk_b4_load_(row, index), exponent = (code >> 1) & 3u;
    nk_f32_t const magnitude = exponent ? (nk_f32_t)((2u + (code & 1u)) << (exponent - 1)) * 0.5f
                                        : (nk_f32_t)(code & 1u) * 0.5f;
    return code & 8u ? -magnitude : magnitude;
}

/** Element @p index of an F64 or F32 row, as the F64 tile reads it. */
NUMKONG_DEVICE nk_f64_t nk_cross_load_f64_(nk_dtype_t dtype, unsigned char const *row, nk_size_t index) {
    return dtype == nk_f64_k ? nk_f64_load_f64_(row, index) : nk_f32_load_f64_(row, index);
}

/** Element @p index of a 16-bit or narrower float row, exactly as F32. */
NUMKONG_DEVICE nk_f32_t nk_cross_load_f32_(nk_dtype_t dtype, unsigned char const *row, nk_size_t index) {
    switch (dtype) {
    case nk_bf16_k: return nk_bf16_load_f32_(row, index);
    case nk_f16_k: return nk_f16_load_f32_(row, index);
    case nk_e5m2_k: return nk_e5m2_load_f32_(row, index);
    case nk_e4m3_k: return nk_e4m3_load_f32_(row, index);
    case nk_e3m2_k: return nk_e3m2_load_f32_(row, index);
    case nk_e2m3_k: return nk_e2m3_load_f32_(row, index);
    default: return nk_e2m1_load_f32_(row, index);
    }
}

#pragma endregion Conversions

/*  Each lane share returns a lane's part of a column's sum of squares, over indices `lane + 32 × k`
 *  below @c depth, which the pack merges across the 32 lanes: F64 for floats, Dot2-compensated for
 *  F64 and F32 inputs, and exact 64-bit sums for integers. Rows read bytewise, since the source's
 *  stride need not be element-aligned. */
#pragma region Norms

NUMKONG_DEVICE nk_f64_t nk_f64_lane_sumsq_(unsigned char const *row, nk_size_t depth, unsigned lane) {
    nk_f64_t sum = 0, compensation = 0;
    for (nk_size_t index = lane; index < depth; index += 32) {
        unsigned long long bits = 0;
        for (unsigned byte = 0; byte < 8; ++byte) bits |= (unsigned long long)row[index * 8 + byte] << (byte * 8);
        nk_f64_t const value = __longlong_as_double((long long)bits);
        nk_cross_step_f64_(nk_cross_accumulation_dot2_k, value, value, &sum, &compensation);
    }
    return nk_f64_add_rn_(sum, compensation);
}

NUMKONG_DEVICE nk_f64_t nk_f32_lane_sumsq_(unsigned char const *row, nk_size_t depth, unsigned lane) {
    nk_f64_t sum = 0, compensation = 0;
    for (nk_size_t index = lane; index < depth; index += 32) {
        nk_u32_t bits = 0;
        for (unsigned byte = 0; byte < 4; ++byte) bits |= (nk_u32_t)row[index * 4 + byte] << (byte * 8);
        nk_f64_t const value = (nk_f64_t)__uint_as_float(bits);
        nk_cross_step_f64_(nk_cross_accumulation_dot2_k, value, value, &sum, &compensation);
    }
    return nk_f64_add_rn_(sum, compensation);
}

#define nk_define_device_lane_sumsq_(input_type_name)                                                     \
    NUMKONG_DEVICE nk_f64_t nk_##input_type_name##_lane_sumsq_(unsigned char const *row, nk_size_t depth, \
                                                               unsigned lane) {                           \
        nk_f64_t sum = 0;                                                                                 \
        for (nk_size_t index = lane; index < depth; index += 32) {                                        \
            nk_f64_t const value = nk_##input_type_name##_load_f32_(row, index);                          \
            sum = __fma_rn(value, value, sum);                                                            \
        }                                                                                                 \
        return sum;                                                                                       \
    }

nk_define_device_lane_sumsq_(bf16)
nk_define_device_lane_sumsq_(f16)
nk_define_device_lane_sumsq_(e5m2)
nk_define_device_lane_sumsq_(e4m3)
nk_define_device_lane_sumsq_(e3m2)
nk_define_device_lane_sumsq_(e2m3)
nk_define_device_lane_sumsq_(e2m1)

#undef nk_define_device_lane_sumsq_

NUMKONG_DEVICE nk_u64_t nk_i8_lane_sumsq_(unsigned char const *row, nk_size_t depth, unsigned lane) {
    nk_u64_t sum = 0;
    for (nk_size_t index = lane; index < depth; index += 32) {
        nk_i32_t const value = (signed char)row[index];
        sum += (nk_u64_t)(value * value);
    }
    return sum;
}

NUMKONG_DEVICE nk_u64_t nk_u8_lane_sumsq_(unsigned char const *row, nk_size_t depth, unsigned lane) {
    nk_u64_t sum = 0;
    for (nk_size_t index = lane; index < depth; index += 32) sum += (nk_u64_t)row[index] * row[index];
    return sum;
}

NUMKONG_DEVICE nk_u64_t nk_i4_lane_sumsq_(unsigned char const *row, nk_size_t depth, unsigned lane) {
    nk_u64_t sum = 0;
    for (nk_size_t index = lane; index < depth; index += 32) {
        nk_i32_t const value = (nk_i32_t)(nk_b4_load_(row, index) ^ 8u) - 8;
        sum += (nk_u64_t)(value * value);
    }
    return sum;
}

NUMKONG_DEVICE nk_u64_t nk_u4_lane_sumsq_(unsigned char const *row, nk_size_t depth, unsigned lane) {
    nk_u64_t sum = 0;
    for (nk_size_t index = lane; index < depth; index += 32) {
        nk_u64_t const value = nk_b4_load_(row, index);
        sum += value * value;
    }
    return sum;
}

/** A column's norm from each of 32 lanes' F64 shares, merged through TwoSum and rounded once. */
NUMKONG_DEVICE nk_f64_t nk_cross_pack_norm_f64_(nk_f64_t share) {
    nk_f64_t compensation = 0;
    for (unsigned offset = 16; offset != 0; offset >>= 1)
        nk_cross_merge_lanes_f64_(nk_cross_accumulation_dot2_k, offset, &share, &compensation);
    return nk_f64_add_rn_(share, compensation);
}

/** A column's F32 norm from each of 32 lanes' F64 shares, summed in F64 and rounded once. */
NUMKONG_DEVICE nk_f32_t nk_cross_pack_norm_f32_(nk_f64_t share) { return (nk_f32_t)nk_cross_pack_norm_f64_(share); }

/** A column's U32 norm from each of 32 lanes' exact shares, truncated like the serial backends'. */
NUMKONG_DEVICE nk_u32_t nk_cross_pack_norm_u32_(nk_u64_t share) {
    for (unsigned offset = 16; offset != 0; offset >>= 1) share += nk_shuffle_xor_u64_(share, offset);
    return (nk_u32_t)share;
}

/*  The tensor tiles' row norms: each update adds the squares of one 16-byte chunk of a staged row,
 *  in any order, since a sum of squares has none, and the finish stores them in 32 bits. */

/** Adds the squares of both F16 halves of @p halves to @p real_sum. */
NUMKONG_DEVICE void nk_f16x2_norm_update_(nk_u32_t halves, nk_f32_t *real_sum) {
    nk_f32_t const low = __half2float(__ushort_as_half((unsigned short)(halves & 0xFFFFu)));
    nk_f32_t const high = __half2float(__ushort_as_half((unsigned short)(halves >> 16)));
    *real_sum = __fmaf_rn(high, high, __fmaf_rn(low, low, *real_sum));
}

NUMKONG_DEVICE void nk_bf16_norm_update_(nk_u32_t const words[4], nk_u32_t *integer_sum, nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) {
        nk_f32_t const low = __uint_as_float(words[word] << 16), high = __uint_as_float(words[word] & 0xFFFF0000u);
        *real_sum = __fmaf_rn(high, high, __fmaf_rn(low, low, *real_sum));
    }
}

NUMKONG_DEVICE void nk_f16_norm_update_(nk_u32_t const words[4], nk_u32_t *integer_sum, nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) nk_f16x2_norm_update_(words[word], real_sum);
}

/** A thread's accumulated squared norm in the 32 bits the epilogue reads: integers as summed,
 *  floats in true units. */
NUMKONG_DEVICE nk_fui32_t nk_cross_norm_finalize_(nk_cross_norm_t norm, nk_u32_t integer_sum, nk_f32_t real_sum,
                                                  nk_f32_t norm_scale) {
    nk_fui32_t result;
    if (norm == nk_cross_norm_f32_k) result.f = ((nk_f32_t)integer_sum + real_sum) * norm_scale;
    else result.u = integer_sum;
    return result;
}

/** A norm's 32 bits as F32, integers read signed or unsigned as @p norm says. */
NUMKONG_DEVICE nk_f32_t nk_cross_norm_to_f32_(nk_fui32_t bits, nk_cross_norm_t norm) {
    if (norm == nk_cross_norm_i32_k) return (nk_f32_t)bits.i;
    if (norm == nk_cross_norm_u32_k) return (nk_f32_t)bits.u;
    return bits.f;
}

#pragma endregion Norms

#pragma region Metrics

/** 1 − dot / (‖a‖ ‖b‖) clamped at 0; with a zero norm, 0 when the dot is 0 and 1 otherwise. */
NUMKONG_DEVICE nk_f64_t nk_f64_angular_(nk_f64_t dot, nk_f64_t row_norm, nk_f64_t column_norm) {
    if (!(row_norm > 0 && column_norm > 0)) return dot == 0 ? 0.0 : 1.0;
    nk_f64_t const unclipped = 1.0 - dot * (rsqrt(row_norm) * rsqrt(column_norm));
    return unclipped > 0 ? unclipped : 0.0;
}

/** √(‖a‖² + ‖b‖² − 2 · dot), with a negative radicand from rounding clamped to 0. */
NUMKONG_DEVICE nk_f64_t nk_f64_euclidean_(nk_f64_t dot, nk_f64_t row_norm, nk_f64_t column_norm) {
    nk_f64_t const squared = row_norm + column_norm - 2.0 * dot;
    return squared > 0 ? sqrt(squared) : 0.0;
}

/** 1 − dot / (‖a‖ ‖b‖) clamped at 0; with a zero norm, 0 when the dot is 0 and 1 otherwise. */
NUMKONG_DEVICE nk_f32_t nk_f32_angular_(nk_f32_t dot, nk_f32_t row_norm, nk_f32_t column_norm) {
    if (!(row_norm > 0 && column_norm > 0)) return dot == 0 ? 0.0f : 1.0f;
    nk_f32_t const unclipped = 1.0f - dot * (rsqrtf(row_norm) * rsqrtf(column_norm));
    return unclipped > 0 ? unclipped : 0.0f;
}

/** √(‖a‖² + ‖b‖² − 2 · dot), with a negative radicand from rounding clamped to 0. */
NUMKONG_DEVICE nk_f32_t nk_f32_euclidean_(nk_f32_t dot, nk_f32_t row_norm, nk_f32_t column_norm) {
    nk_f32_t const squared = row_norm + column_norm - 2.0f * dot;
    return squared > 0 ? sqrtf(squared) : 0.0f;
}

/** A tensor-core dot product as F32, the way the serial metrics read it: F32 sums and scaled
 *  integer sums times @p output_scale, other integers signed unless their norms are unsigned. */
NUMKONG_DEVICE nk_f32_t nk_cross_dot_to_f32_(nk_fui32_t sum, nk_cross_epilogue_t epilogue, nk_cross_norm_t norm,
                                             nk_f32_t output_scale) {
    if (epilogue == nk_cross_epilogue_f32_k) return sum.f * output_scale;
    if (epilogue == nk_cross_epilogue_i32_to_f32_k) return (nk_f32_t)sum.i * output_scale;
    return norm == nk_cross_norm_u32_k ? (nk_f32_t)sum.u : (nk_f32_t)sum.i;
}

#pragma endregion Metrics

/*  The two baseline tiles share one shape: each of 256 threads owns a 4 × 4 grid of one 64 × 64
 *  output tile, strided by 16, so shared-memory reads broadcast along one axis and sweep the banks
 *  along the other. The F64 tile stages F64 values and sums F64 or Dot2; the B32 tile stages 32-bit
 *  words and folds each pair with one instruction. For a metric, each thread also squares the A
 *  and, for @c symmetric, B elements it stages, and the threads staging one row merge them. */
#pragma region Baseline Tile

/** Folds one 32-bit word of each operand into @p sum, by the instruction @p accumulation names. */
NUMKONG_DEVICE void nk_cross_fold_b32_(nk_cross_accumulation_t accumulation, nk_fui32_t *sum, nk_u32_t a, nk_u32_t b) {
    switch (accumulation) {
    case nk_cross_accumulation_f32_k: sum->f = __fmaf_rn(__uint_as_float(a), __uint_as_float(b), sum->f); break;
    case nk_cross_accumulation_f16x2_k: sum->f = nk_dot_f16x2_(a, b, sum->f); break;
    case nk_cross_accumulation_i8x4_k:
    case nk_cross_accumulation_i4x4_k: sum->i = nk_dot_i8x4_(a, b, sum->i); break;
    case nk_cross_accumulation_u8x4_k:
    case nk_cross_accumulation_u4x4_k: sum->u = nk_dot_u8x4_(a, b, sum->u); break;
    case nk_cross_accumulation_i4x8_k: sum->i = nk_dot_i4x8_(a, b, sum->i); break;
    default: sum->u = nk_dot_u4x8_(a, b, sum->u); break;
    }
}

/** Adds lane `lane ^ offset`'s B32 sum into @p sum: in F32 for floats, wrapping for integers. */
NUMKONG_DEVICE nk_fui32_t nk_cross_merge_lanes_b32_(nk_cross_accumulation_t accumulation, unsigned offset,
                                                    nk_fui32_t sum) {
    nk_fui32_t other;
    other.u = nk_shuffle_xor_u32_(sum.u, offset);
    if (accumulation == nk_cross_accumulation_f32_k || accumulation == nk_cross_accumulation_f16x2_k) sum.f += other.f;
    else sum.u += other.u;
    return sum;
}

/** A B32 sum as F32: floats as they are, integers read signed or unsigned as serial backends do. */
NUMKONG_DEVICE nk_f32_t nk_cross_b32_to_f32_(nk_cross_accumulation_t accumulation, nk_fui32_t sum) {
    switch (accumulation) {
    case nk_cross_accumulation_f32_k:
    case nk_cross_accumulation_f16x2_k: return sum.f;
    case nk_cross_accumulation_i8x4_k:
    case nk_cross_accumulation_i4x4_k:
    case nk_cross_accumulation_i4x8_k: return (nk_f32_t)sum.i;
    default: return (nk_f32_t)sum.u;
    }
}

/** Elements of depth one staged word holds under @p accumulation. */
NUMKONG_DEVICE unsigned nk_cross_b32_dimensions_(nk_cross_accumulation_t accumulation) {
    switch (accumulation) {
    case nk_cross_accumulation_f32_k: return 1;
    case nk_cross_accumulation_f16x2_k: return 2;
    case nk_cross_accumulation_i4x8_k:
    case nk_cross_accumulation_u4x8_k: return 8;
    default: return 4;
    }
}

/** Word @p word of a row as the B32 tile stages it for @p accumulation, with every element at or
 *  past @p depth zeroed. Rows start on 16 bytes, so whole words load aligned. */
NUMKONG_DEVICE nk_u32_t nk_cross_stage_b32_(nk_cross_accumulation_t accumulation, nk_dtype_t dtype,
                                            unsigned char const *row, nk_size_t word, nk_size_t depth) {
    nk_size_t const first = word * nk_cross_b32_dimensions_(accumulation);
    nk_u32_t bits = 0;
    switch (accumulation) {
    case nk_cross_accumulation_f32_k: return __float_as_uint(nk_cross_load_f32_(dtype, row, first));
    case nk_cross_accumulation_f16x2_k: {
        unsigned short halves[2] = {0, 0};
        nk_f32_t value = nk_cross_load_f32_(dtype, row, first);
        nk_f32_to_f16_simt_(&value, (nk_f16_t *)&halves[0]);
        if (first + 1 < depth) {
            value = nk_cross_load_f32_(dtype, row, first + 1);
            nk_f32_to_f16_simt_(&value, (nk_f16_t *)&halves[1]);
        }
        return (nk_u32_t)halves[0] | ((nk_u32_t)halves[1] << 16);
    }
    case nk_cross_accumulation_i8x4_k:
    case nk_cross_accumulation_u8x4_k:
        if (first + 4 <= depth) return *(nk_u32_t const *)(row + first);
        for (unsigned byte = 0; byte < 4; ++byte)
            if (first + byte < depth) bits |= (nk_u32_t)row[first + byte] << (byte * 8);
        return bits;
    case nk_cross_accumulation_i4x4_k:
    case nk_cross_accumulation_u4x4_k:
        // Sign- or zero-extends each nibble into its own byte, for NVIDIA's byte-wise `dp4a`.
        for (unsigned nibble = 0; nibble < 4; ++nibble) {
            if (first + nibble >= depth) continue;
            nk_u32_t const code = nk_b4_load_(row, first + nibble);
            nk_u32_t const byte = accumulation == nk_cross_accumulation_i4x4_k ? ((0u - (code & 8u)) | code) & 0xFFu
                                                                               : code;
            bits |= byte << (nibble * 8);
        }
        return bits;
    default:
        if (first + 8 <= depth) return *(nk_u32_t const *)(row + first / 2);
        for (unsigned nibble = 0; nibble < 8; ++nibble)
            if (first + nibble < depth)
                bits |= nk_b4_load_(row, first + nibble) << (nibble / 2 * 8 + (nibble & 1 ? 0 : 4));
        return bits;
    }
}

/** Stages one slab of F64 values for the tile at @p first_row and @p first_column, squaring each
 *  into its row's compensated norm when @p metric needs norms. */
NUMKONG_DEVICE void nk_cross_stage_slab_simt_f64_(nk_dtype_t dtype, nk_cross_triangle_t triangle,
                                                  nk_cross_metric_t metric, nk_cross_tile_arguments_t const *arguments,
                                                  nk_size_t first_row, nk_size_t first_column, nk_size_t slab,
                                                  nk_f64_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
                                                  nk_f64_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
                                                  nk_f64_t a_norms[nk_cross_loads_simt_k][2],
                                                  nk_f64_t b_norms[nk_cross_loads_simt_k][2]) {
#pragma unroll
    for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step) {
        unsigned const element = threadIdx.x + step * nk_cross_threads_simt_k;
        unsigned const tile_row = element / nk_cross_slab_simt_k, offset = element % nk_cross_slab_simt_k;
        nk_size_t const index = slab + offset, row = first_row + tile_row, column = first_column + tile_row;
        nk_f64_t const a_value = row < arguments->row_end && index < arguments->depth
                                     ? nk_cross_load_f64_(dtype, arguments->a + row * arguments->a_stride, index)
                                     : 0;
        nk_f64_t const b_value = column < arguments->column_count && index < arguments->depth
                                     ? nk_cross_load_f64_(dtype, arguments->b + column * arguments->b_stride, index)
                                     : 0;
        a_slab[offset][tile_row] = a_value, b_slab[offset][tile_row] = b_value;
        if (metric == nk_cross_metric_dot_k) continue;
        nk_cross_step_f64_(nk_cross_accumulation_dot2_k, a_value, a_value, &a_norms[step][0], &a_norms[step][1]);
        if (triangle == nk_cross_triangle_upper_k)
            nk_cross_step_f64_(nk_cross_accumulation_dot2_k, b_value, b_value, &b_norms[step][0], &b_norms[step][1]);
    }
}

/** Folds one staged slab into this thread's grid of F64 sums and their compensations. */
NUMKONG_DEVICE void nk_cross_fold_slab_simt_f64_(
    nk_cross_accumulation_t accumulation, nk_f64_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
    nk_f64_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
    nk_f64_t sums[nk_cross_thread_tile_simt_k][nk_cross_thread_tile_simt_k],
    nk_f64_t compensations[nk_cross_thread_tile_simt_k][nk_cross_thread_tile_simt_k]) {
    unsigned const thread_column = threadIdx.x % nk_cross_grid_side_simt_k;
    unsigned const thread_row = threadIdx.x / nk_cross_grid_side_simt_k;
#pragma unroll
    for (unsigned offset = 0; offset < nk_cross_slab_simt_k; ++offset) {
        nk_f64_t a_values[nk_cross_thread_tile_simt_k], b_values[nk_cross_thread_tile_simt_k];
#pragma unroll
        for (unsigned step = 0; step < nk_cross_thread_tile_simt_k; ++step)
            a_values[step] = a_slab[offset][thread_row + nk_cross_grid_side_simt_k * step],
            b_values[step] = b_slab[offset][thread_column + nk_cross_grid_side_simt_k * step];
#pragma unroll
        for (unsigned row_step = 0; row_step < nk_cross_thread_tile_simt_k; ++row_step)
#pragma unroll
            for (unsigned column_step = 0; column_step < nk_cross_thread_tile_simt_k; ++column_step)
                nk_cross_step_f64_(accumulation, a_values[row_step], b_values[column_step],
                                   &sums[row_step][column_step], &compensations[row_step][column_step]);
    }
}

/** Merges each staged row's compensated norm across the threads that staged it, then publishes the
 *  tile's row norms and its column norms, read from @c b_norms for @c packed. */
NUMKONG_DEVICE void nk_cross_finish_norms_simt_f64_(nk_cross_triangle_t triangle,
                                                    nk_cross_tile_arguments_t const *arguments, nk_size_t first_column,
                                                    nk_f64_t a_norms[nk_cross_loads_simt_k][2],
                                                    nk_f64_t b_norms[nk_cross_loads_simt_k][2],
                                                    nk_f64_t norms[2][nk_cross_tile_simt_k]) {
#pragma unroll
    for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step)
#pragma unroll
        for (unsigned offset = nk_cross_slab_simt_k / 2; offset != 0; offset >>= 1) {
            nk_cross_merge_lanes_f64_(nk_cross_accumulation_dot2_k, offset, &a_norms[step][0], &a_norms[step][1]);
            if (triangle == nk_cross_triangle_upper_k)
                nk_cross_merge_lanes_f64_(nk_cross_accumulation_dot2_k, offset, &b_norms[step][0], &b_norms[step][1]);
        }
    // Zero-depth tiles skip the slab loop's barriers, so this one retires the last epilogue reads.
    __syncthreads();
    if (threadIdx.x % nk_cross_slab_simt_k == 0)
#pragma unroll
        for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step) {
            unsigned const tile_row = threadIdx.x / nk_cross_slab_simt_k +
                                      step * (nk_cross_threads_simt_k / nk_cross_slab_simt_k);
            norms[0][tile_row] = nk_f64_add_rn_(a_norms[step][0], a_norms[step][1]);
            if (triangle == nk_cross_triangle_upper_k)
                norms[1][tile_row] = nk_f64_add_rn_(b_norms[step][0], b_norms[step][1]);
        }
    if (triangle == nk_cross_triangle_full_k && threadIdx.x < nk_cross_tile_simt_k) {
        nk_size_t const column = first_column + threadIdx.x;
        norms[1][threadIdx.x] = column < arguments->column_count ? ((nk_f64_t const *)arguments->b_norms)[column] : 0;
    }
    __syncthreads();
}

/** Writes this thread's outputs of the tile as F64: dots, or metrics with zeros on the diagonal of
 *  @c symmetric. */
NUMKONG_DEVICE void nk_cross_store_tile_simt_f64_(
    nk_cross_triangle_t triangle, nk_cross_metric_t metric, nk_cross_tile_arguments_t const *arguments,
    nk_size_t first_row, nk_size_t first_column,
    nk_f64_t sums[nk_cross_thread_tile_simt_k][nk_cross_thread_tile_simt_k],
    nk_f64_t compensations[nk_cross_thread_tile_simt_k][nk_cross_thread_tile_simt_k],
    nk_f64_t norms[2][nk_cross_tile_simt_k]) {
    unsigned const thread_column = threadIdx.x % nk_cross_grid_side_simt_k;
    unsigned const thread_row = threadIdx.x / nk_cross_grid_side_simt_k;
#pragma unroll
    for (unsigned row_step = 0; row_step < nk_cross_thread_tile_simt_k; ++row_step) {
        unsigned const tile_row = thread_row + nk_cross_grid_side_simt_k * row_step;
        nk_size_t const row = first_row + tile_row;
        if (row >= arguments->row_end) continue;
        nk_f64_t *output = (nk_f64_t *)((unsigned char *)arguments->c + row * arguments->c_stride);
#pragma unroll
        for (unsigned column_step = 0; column_step < nk_cross_thread_tile_simt_k; ++column_step) {
            unsigned const tile_column = thread_column + nk_cross_grid_side_simt_k * column_step;
            nk_size_t const column = first_column + tile_column;
            if (column >= arguments->column_count || (triangle == nk_cross_triangle_upper_k && column < row)) continue;
            nk_f64_t const dot = nk_f64_add_rn_(sums[row_step][column_step], compensations[row_step][column_step]);
            if (metric == nk_cross_metric_dot_k) output[column] = dot;
            else if (triangle == nk_cross_triangle_upper_k && column == row) output[column] = 0;
            else if (metric == nk_cross_metric_angular_k)
                output[column] = nk_f64_angular_(dot, norms[0][tile_row], norms[1][tile_column]);
            else output[column] = nk_f64_euclidean_(dot, norms[0][tile_row], norms[1][tile_column]);
        }
    }
}

/**
 *  @brief The GEMM of one 64 × 64 output tile of F64 or F32 inputs, in F64 or Dot2.
 *  @param[in] dtype @c nk_f64_k or @c nk_f32_k, the rows' element type.
 *  @param[in] accumulation @c nk_cross_accumulation_dot2_k for F64 inputs and
 *      @c nk_cross_accumulation_f64_k for F32 ones; the norms always sum in Dot2.
 *
 *  Tensor-core F64 MMA never exposes a product's rounding error, which Dot2 needs, and outside the
 *  8.0 and 9.0 datacenter parts it runs no faster than these FMAs.
 */
NUMKONG_DEVICE void nk_cross_tile_simt_f64_(nk_dtype_t dtype, nk_cross_accumulation_t accumulation,
                                            nk_cross_triangle_t triangle, nk_cross_metric_t metric,
                                            nk_cross_tile_arguments_t const *arguments) {
    // One extra column breaks the 64-element stride that would put a slab's stores on one bank.
    __shared__ nk_f64_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1];
    __shared__ nk_f64_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1];
    __shared__ nk_f64_t norms[2][nk_cross_tile_simt_k];

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        nk_size_t const first_row = arguments->row_start + tile / arguments->column_tiles * nk_cross_tile_simt_k;
        nk_size_t const first_column = tile % arguments->column_tiles * nk_cross_tile_simt_k;
        if (triangle == nk_cross_triangle_upper_k && first_column + nk_cross_tile_simt_k <= first_row) continue;
        nk_f64_t sums[nk_cross_thread_tile_simt_k][nk_cross_thread_tile_simt_k] = {{0}};
        nk_f64_t compensations[nk_cross_thread_tile_simt_k][nk_cross_thread_tile_simt_k] = {{0}};
        nk_f64_t a_norms[nk_cross_loads_simt_k][2] = {{0}}, b_norms[nk_cross_loads_simt_k][2] = {{0}};
        for (nk_size_t slab = 0; slab < arguments->depth; slab += nk_cross_slab_simt_k) {
            nk_cross_stage_slab_simt_f64_(dtype, triangle, metric, arguments, first_row, first_column, slab, a_slab,
                                          b_slab, a_norms, b_norms);
            __syncthreads();
            nk_cross_fold_slab_simt_f64_(accumulation, a_slab, b_slab, sums, compensations);
            __syncthreads();
        }
        if (metric != nk_cross_metric_dot_k)
            nk_cross_finish_norms_simt_f64_(triangle, arguments, first_column, a_norms, b_norms, norms);
        nk_cross_store_tile_simt_f64_(triangle, metric, arguments, first_row, first_column, sums, compensations, norms);
    }
}

/** Stages one slab of 32-bit words for the tile at @p first_row and @p first_column, folding each
 *  word with itself into its row's norm when @p metric needs norms. */
NUMKONG_DEVICE void nk_cross_stage_slab_simt_b32_(nk_dtype_t dtype, nk_cross_accumulation_t accumulation,
                                                  nk_cross_triangle_t triangle, nk_cross_metric_t metric,
                                                  nk_cross_tile_arguments_t const *arguments, nk_size_t first_row,
                                                  nk_size_t first_column, nk_size_t slab, nk_size_t words,
                                                  nk_u32_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
                                                  nk_u32_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
                                                  nk_fui32_t a_norms[nk_cross_loads_simt_k],
                                                  nk_fui32_t b_norms[nk_cross_loads_simt_k]) {
#pragma unroll
    for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step) {
        unsigned const element = threadIdx.x + step * nk_cross_threads_simt_k;
        unsigned const tile_row = element / nk_cross_slab_simt_k, offset = element % nk_cross_slab_simt_k;
        nk_size_t const word = slab + offset, row = first_row + tile_row, column = first_column + tile_row;
        nk_u32_t const a_word = row < arguments->row_end && word < words
                                    ? nk_cross_stage_b32_(accumulation, dtype, arguments->a + row * arguments->a_stride,
                                                          word, arguments->depth)
                                    : 0;
        nk_u32_t const b_word = column < arguments->column_count && word < words
                                    ? nk_cross_stage_b32_(accumulation, dtype,
                                                          arguments->b + column * arguments->b_stride, word,
                                                          arguments->depth)
                                    : 0;
        a_slab[offset][tile_row] = a_word, b_slab[offset][tile_row] = b_word;
        if (metric == nk_cross_metric_dot_k) continue;
        nk_cross_fold_b32_(accumulation, &a_norms[step], a_word, a_word);
        if (triangle == nk_cross_triangle_upper_k) nk_cross_fold_b32_(accumulation, &b_norms[step], b_word, b_word);
    }
}

/** Folds one staged slab into this thread's grid of 32-bit sums. */
NUMKONG_DEVICE void nk_cross_fold_slab_simt_b32_(
    nk_cross_accumulation_t accumulation, nk_u32_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
    nk_u32_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1],
    nk_fui32_t sums[nk_cross_thread_tile_simt_k][nk_cross_thread_tile_simt_k]) {
    unsigned const thread_column = threadIdx.x % nk_cross_grid_side_simt_k;
    unsigned const thread_row = threadIdx.x / nk_cross_grid_side_simt_k;
#pragma unroll
    for (unsigned offset = 0; offset < nk_cross_slab_simt_k; ++offset) {
        nk_u32_t a_words[nk_cross_thread_tile_simt_k], b_words[nk_cross_thread_tile_simt_k];
#pragma unroll
        for (unsigned step = 0; step < nk_cross_thread_tile_simt_k; ++step)
            a_words[step] = a_slab[offset][thread_row + nk_cross_grid_side_simt_k * step],
            b_words[step] = b_slab[offset][thread_column + nk_cross_grid_side_simt_k * step];
#pragma unroll
        for (unsigned row_step = 0; row_step < nk_cross_thread_tile_simt_k; ++row_step)
#pragma unroll
            for (unsigned column_step = 0; column_step < nk_cross_thread_tile_simt_k; ++column_step)
                nk_cross_fold_b32_(accumulation, &sums[row_step][column_step], a_words[row_step], b_words[column_step]);
    }
}

/** Merges each staged row's norm across the threads that staged it, then publishes the tile's row
 *  norms and its column norms, read from the pack for @c packed. */
NUMKONG_DEVICE void nk_cross_finish_norms_simt_b32_(nk_cross_accumulation_t accumulation, nk_cross_triangle_t triangle,
                                                    nk_cross_tile_arguments_t const *arguments, nk_size_t first_column,
                                                    nk_fui32_t a_norms[nk_cross_loads_simt_k],
                                                    nk_fui32_t b_norms[nk_cross_loads_simt_k],
                                                    nk_fui32_t norms[2][nk_cross_tile_simt_k]) {
#pragma unroll
    for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step)
#pragma unroll
        for (unsigned offset = nk_cross_slab_simt_k / 2; offset != 0; offset >>= 1) {
            a_norms[step] = nk_cross_merge_lanes_b32_(accumulation, offset, a_norms[step]);
            if (triangle == nk_cross_triangle_upper_k)
                b_norms[step] = nk_cross_merge_lanes_b32_(accumulation, offset, b_norms[step]);
        }
    // Zero-depth tiles skip the slab loop's barriers, so this one retires the last epilogue reads.
    __syncthreads();
    if (threadIdx.x % nk_cross_slab_simt_k == 0)
#pragma unroll
        for (unsigned step = 0; step < nk_cross_loads_simt_k; ++step) {
            unsigned const tile_row = threadIdx.x / nk_cross_slab_simt_k +
                                      step * (nk_cross_threads_simt_k / nk_cross_slab_simt_k);
            norms[0][tile_row] = a_norms[step];
            if (triangle == nk_cross_triangle_upper_k) norms[1][tile_row] = b_norms[step];
        }
    if (triangle == nk_cross_triangle_full_k && threadIdx.x < nk_cross_tile_simt_k) {
        nk_size_t const column = first_column + threadIdx.x;
        norms[1][threadIdx.x].u = column < arguments->column_count ? ((nk_u32_t const *)arguments->b_norms)[column] : 0;
    }
    __syncthreads();
}

/** Writes this thread's outputs of the tile: dots as their 32 bits, or metrics in F32 with zeros
 *  on the diagonal of @c symmetric. */
NUMKONG_DEVICE void nk_cross_store_tile_simt_b32_(
    nk_cross_accumulation_t accumulation, nk_cross_triangle_t triangle, nk_cross_metric_t metric,
    nk_cross_tile_arguments_t const *arguments, nk_size_t first_row, nk_size_t first_column,
    nk_fui32_t sums[nk_cross_thread_tile_simt_k][nk_cross_thread_tile_simt_k],
    nk_fui32_t norms[2][nk_cross_tile_simt_k]) {
    unsigned const thread_column = threadIdx.x % nk_cross_grid_side_simt_k;
    unsigned const thread_row = threadIdx.x / nk_cross_grid_side_simt_k;
#pragma unroll
    for (unsigned row_step = 0; row_step < nk_cross_thread_tile_simt_k; ++row_step) {
        unsigned const tile_row = thread_row + nk_cross_grid_side_simt_k * row_step;
        nk_size_t const row = first_row + tile_row;
        if (row >= arguments->row_end) continue;
        nk_fui32_t *output = (nk_fui32_t *)((unsigned char *)arguments->c + row * arguments->c_stride);
#pragma unroll
        for (unsigned column_step = 0; column_step < nk_cross_thread_tile_simt_k; ++column_step) {
            unsigned const tile_column = thread_column + nk_cross_grid_side_simt_k * column_step;
            nk_size_t const column = first_column + tile_column;
            if (column >= arguments->column_count || (triangle == nk_cross_triangle_upper_k && column < row)) continue;
            if (metric == nk_cross_metric_dot_k) {
                output[column] = sums[row_step][column_step];
                continue;
            }
            nk_f32_t const dot = nk_cross_b32_to_f32_(accumulation, sums[row_step][column_step]);
            nk_f32_t const row_norm = nk_cross_b32_to_f32_(accumulation, norms[0][tile_row]);
            nk_f32_t const column_norm = nk_cross_b32_to_f32_(accumulation, norms[1][tile_column]);
            if (triangle == nk_cross_triangle_upper_k && column == row) output[column].f = 0;
            else if (metric == nk_cross_metric_angular_k)
                output[column].f = nk_f32_angular_(dot, row_norm, column_norm);
            else output[column].f = nk_f32_euclidean_(dot, row_norm, column_norm);
        }
    }
}

/**
 *  @brief The GEMM of one 64 × 64 output tile of 16-bit or narrower inputs, folding 32-bit words.
 *  @param[in] dtype The rows' element type, which a float word decodes from.
 *  @param[in] accumulation How each word holds depth and which instruction folds it; the norms fold
 *      the same way.
 */
NUMKONG_DEVICE void nk_cross_tile_simt_b32_(nk_dtype_t dtype, nk_cross_accumulation_t accumulation,
                                            nk_cross_triangle_t triangle, nk_cross_metric_t metric,
                                            nk_cross_tile_arguments_t const *arguments) {
    // One extra column breaks the 64-word stride that would put a slab's stores on one bank.
    __shared__ nk_u32_t a_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1];
    __shared__ nk_u32_t b_slab[nk_cross_slab_simt_k][nk_cross_tile_simt_k + 1];
    __shared__ nk_fui32_t norms[2][nk_cross_tile_simt_k];
    nk_size_t const words = nk_size_divide_round_up_(arguments->depth, nk_cross_b32_dimensions_(accumulation));

    for (nk_size_t tile = blockIdx.x; tile < arguments->tiles; tile += gridDim.x) {
        nk_size_t const first_row = arguments->row_start + tile / arguments->column_tiles * nk_cross_tile_simt_k;
        nk_size_t const first_column = tile % arguments->column_tiles * nk_cross_tile_simt_k;
        if (triangle == nk_cross_triangle_upper_k && first_column + nk_cross_tile_simt_k <= first_row) continue;
        nk_fui32_t sums[nk_cross_thread_tile_simt_k][nk_cross_thread_tile_simt_k] = {{{0}}};
        nk_fui32_t a_norms[nk_cross_loads_simt_k] = {{0}}, b_norms[nk_cross_loads_simt_k] = {{0}};
        for (nk_size_t slab = 0; slab < words; slab += nk_cross_slab_simt_k) {
            nk_cross_stage_slab_simt_b32_(dtype, accumulation, triangle, metric, arguments, first_row, first_column,
                                          slab, words, a_slab, b_slab, a_norms, b_norms);
            __syncthreads();
            nk_cross_fold_slab_simt_b32_(accumulation, a_slab, b_slab, sums);
            __syncthreads();
        }
        if (metric != nk_cross_metric_dot_k)
            nk_cross_finish_norms_simt_b32_(accumulation, triangle, arguments, first_column, a_norms, b_norms, norms);
        nk_cross_store_tile_simt_b32_(accumulation, triangle, metric, arguments, first_row, first_column, sums, norms);
    }
}

#pragma endregion Baseline Tile

/*  Every tile, baseline or tensor, takes its own leading arguments, then the triangle, the metric
 *  and the launch arguments, so one generator per shape serves them all: the site passes the tile's
 *  own arguments last, through the variadic tail. */
#pragma region Cross Macros

/**
 *  @brief Generates the bytes of a device pack: the header, then rows padded by
 *      @c nk_device_cross_padded_values_, then one norm per column.
 *  @param[in] depth_simd_dimensions The dimensions each row rounds up to, 16 bytes' worth.
 *  @sa nk_define_cross_pack_size_ for the host original.
 */
#define nk_define_device_cross_pack_size_(input_type_name, isa_suffix, packed_value_type, norm_value_type,             \
                                          depth_simd_dimensions, dimensions_per_value)                                 \
    NUMKONG_API nk_status_t nk_dots_pack_size_##input_type_name##_##isa_suffix(nk_size_t column_count,                 \
                                                                               nk_size_t depth, nk_size_t *bytes) {    \
        nk_assert_(depth % dimensions_per_value == 0);                                                                 \
        nk_size_t const row_bytes = nk_device_cross_padded_values_(depth, depth_simd_dimensions, dimensions_per_value, \
                                                                   sizeof(nk_##packed_value_type##_t)) *               \
                                    sizeof(nk_##packed_value_type##_t);                                                \
        *bytes = sizeof(nk_cross_packed_buffer_header_t) +                                                             \
                 column_count * (row_bytes + sizeof(nk_##norm_value_type##_t));                                        \
        return nk_success_k;                                                                                           \
    }

/**
 *  @brief Generates a packed-shape accessor copying a device-resident packed buffer's header back.
 *  @sa nk_define_cross_packed_shape_ for the host-resident original.
 */
#define nk_define_device_cross_packed_shape_(input_type_name, isa_suffix)                      \
    NUMKONG_API nk_status_t nk_dots_packed_shape_##input_type_name##_##isa_suffix(             \
        void const *b_packed, nk_size_t *width, nk_size_t *depth, void *stream) {              \
        if ((nk_size_t)b_packed & 15) return nk_misaligned_k;                                  \
        nk_cross_packed_buffer_header_t header;                                                \
        nk_status_t const status = nk_device_read_(&header, b_packed, sizeof(header), stream); \
        if (status != nk_success_k) return status;                                             \
        if (header.capability != nk_cap_##isa_suffix##_k) return nk_pack_mismatch_k;           \
        *width = header.column_count, *depth = header.depth_dimensions;                        \
        return nk_success_k;                                                                   \
    }

/**
 *  @brief Generates a pack into the serial layout on the device, one 32-lane group per column.
 *
 *  The header is written only when @p columns_begin is zero, and each packed column gets its row
 *  and its norm, so disjoint column ranges may be packed by separate calls, exactly as with
 *  @c nk_define_cross_pack_.
 *
 *  @param[in] load_fn Device map from an input byte to its packed byte, the identity unless the
 *      multiply wants a remap.
 *  @param[in] norm_value_type The serial backends' norm type: @c f64, @c f32 or @c u32.
 *  @param[in] compute_norm_fn Device share of a column's sum of squares for one lane of 32, which
 *      @c nk_cross_pack_norm_f64_, @c nk_cross_pack_norm_f32_ or @c nk_cross_pack_norm_u32_ merges.
 *  @sa nk_define_cross_pack_ for the host original.
 */
#define nk_define_device_cross_pack_rows_(input_type_name, isa_suffix, input_value_type, packed_value_type, load_fn, \
                                          norm_value_type, compute_norm_fn, depth_simd_dimensions,                   \
                                          dimensions_per_value)                                                      \
    static __global__ void nk_dots_pack_##input_type_name##_##isa_suffix##_kernel_(                                  \
        unsigned char const *b, nk_size_t column_count, nk_size_t depth, nk_size_t depth_bytes,                      \
        nk_size_t b_stride_in_bytes, unsigned char *b_packed, nk_size_t columns_begin, nk_size_t columns_end,        \
        nk_size_t depth_values_padded, nk_capability_t capability) {                                                 \
        nk_size_t const row_bytes = depth_values_padded * sizeof(nk_##packed_value_type##_t);                        \
        if (columns_begin == 0 && blockIdx.x == 0 && threadIdx.x == 0) {                                             \
            nk_cross_packed_buffer_header_t *header = (nk_cross_packed_buffer_header_t *)b_packed;                   \
            header->column_count = (nk_u32_t)column_count;                                                           \
            header->depth_dimensions = (nk_u32_t)depth;                                                              \
            header->depth_padded_values = (nk_u32_t)depth_values_padded;                                             \
            header->capability = capability;                                                                         \
            for (unsigned reserved_index = 0; reserved_index < 11; ++reserved_index)                                 \
                header->reserved[reserved_index] = 0;                                                                \
        }                                                                                                            \
        unsigned char *rows = b_packed + sizeof(nk_cross_packed_buffer_header_t);                                    \
        nk_##norm_value_type##_t *norms = (nk_##norm_value_type##_t *)(rows + column_count * row_bytes);             \
        unsigned const lane = threadIdx.x & 31;                                                                      \
        nk_size_t const groups = (nk_size_t)gridDim.x * (blockDim.x >> 5);                                           \
        for (nk_size_t column = columns_begin + (nk_size_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);      \
             column < columns_end; column += groups) {                                                               \
            unsigned char const *source = b + column * b_stride_in_bytes;                                            \
            unsigned char *destination = rows + column * row_bytes;                                                  \
            for (nk_size_t byte = lane; byte < row_bytes; byte += 32)                                                \
                destination[byte] = byte < depth_bytes ? load_fn(source[byte]) : 0;                                  \
            nk_##norm_value_type##_t const norm = nk_cross_pack_norm_##norm_value_type##_(                           \
                compute_norm_fn(source, depth, lane));                                                               \
            if (lane == 0) norms[column] = norm;                                                                     \
        }                                                                                                            \
    }                                                                                                                \
    NUMKONG_API nk_status_t nk_dots_pack_##input_type_name##_##isa_suffix(                                           \
        nk_##input_value_type##_t const *b, void const *b_scales, nk_size_t column_count, nk_size_t depth,           \
        nk_size_t b_stride_in_bytes, nk_size_t b_scales_stride, void *b_packed, nk_size_t columns_begin,             \
        nk_size_t columns_end, void *stream) {                                                                       \
        nk_size_t const depth_values_padded = nk_device_cross_padded_values_(                                        \
            depth, depth_simd_dimensions, dimensions_per_value, sizeof(nk_##packed_value_type##_t));                 \
        nk_size_t const depth_bytes = depth / dimensions_per_value * sizeof(nk_##input_value_type##_t);              \
        return nk_cross_pack_launch_((void const *)nk_dots_pack_##input_type_name##_##isa_suffix##_kernel_, b,       \
                                     column_count, depth, depth_bytes, b_stride_in_bytes, b_packed, columns_begin,   \
                                     columns_end, depth_values_padded, nk_cap_##isa_suffix##_k, stream);             \
    }

/** The pack's size, shape reader and kernel, which always ship together, with the norm share of
 *  the input type, @c nk_<input_type_name>_lane_sumsq_. */
#define nk_define_device_cross_pack_(input_type_name, isa_suffix, input_value_type, packed_value_type, load_fn,   \
                                     norm_value_type, depth_simd_dimensions, dimensions_per_value)                \
    nk_define_device_cross_pack_size_(input_type_name, isa_suffix, packed_value_type, norm_value_type,            \
                                      depth_simd_dimensions, dimensions_per_value)                                \
    nk_define_device_cross_packed_shape_(input_type_name, isa_suffix)                                             \
    nk_define_device_cross_pack_rows_(input_type_name, isa_suffix, input_value_type, packed_value_type, load_fn,  \
                                      norm_value_type, nk_##input_type_name##_lane_sumsq_, depth_simd_dimensions, \
                                      dimensions_per_value)

/**
 *  @brief Generates C = A × Bᵀ, or its angular or euclidean distances, on @p tile over a B packed
 *      by @c nk_define_device_cross_pack_, each block walking tiles with a stride of the grid.
 *  @param[in] metric @c dot, @c angular or @c euclidean, naming both the entry and the epilogue.
 *  @param[in] tile The tile stem, like @c simt_b32 or @c ampere, naming its
 *      function and launch shape.
 *  @param[in] ... The tile's own leading arguments, which precede the triangle, the metric
 *      and the launch arguments.
 *  @sa nk_define_cross_packed_ for the host original.
 */
#define nk_define_device_cross_packed_(metric, input_type_name, isa_suffix, tile, input_value_type, packed_value_type, \
                                       result_value_type, depth_simd_dimensions, dimensions_per_value, ...)            \
    static __global__ void __launch_bounds__(nk_cross_threads_##tile##_k)                                              \
        nk_##metric##s_packed_##input_type_name##_##isa_suffix##_kernel_(nk_cross_tile_arguments_t arguments) {        \
        nk_cross_tile_##tile##_(__VA_ARGS__, nk_cross_triangle_full_k, nk_cross_metric_##metric##_k, &arguments);      \
    }                                                                                                                  \
    NUMKONG_API nk_status_t nk_##metric##s_packed_##input_type_name##_##isa_suffix(                                    \
        nk_##input_value_type##_t const *a_matrix, void const *a_scales, void const *b_packed_buffer,                  \
        nk_##result_value_type##_t *c_matrix, nk_size_t row_count, nk_size_t column_count, nk_size_t depth,            \
        nk_size_t a_stride_in_bytes, nk_size_t a_scales_stride, nk_size_t c_stride_in_bytes, void *stream) {           \
        nk_size_t const row_bytes = nk_device_cross_padded_values_(depth, depth_simd_dimensions, dimensions_per_value, \
                                                                   sizeof(nk_##packed_value_type##_t)) *               \
                                    sizeof(nk_##packed_value_type##_t);                                                \
        unsigned char const *b_rows = (unsigned char const *)b_packed_buffer +                                         \
                                      sizeof(nk_cross_packed_buffer_header_t);                                         \
        return nk_cross_launch_((void const *)nk_##metric##s_packed_##input_type_name##_##isa_suffix##_kernel_,        \
                                nk_cross_tile_##tile##_k, nk_cross_threads_##tile##_k, a_matrix, b_rows,               \
                                b_rows + column_count * row_bytes, c_matrix, sizeof(nk_##result_value_type##_t), 0,    \
                                row_count, column_count, depth,                                                        \
                                depth / dimensions_per_value * sizeof(nk_##input_value_type##_t), a_stride_in_bytes,   \
                                row_bytes, c_stride_in_bytes, stream);                                                 \
    }

/**
 *  @brief Generates the Gram matrix C = A × Aᵀ, or its angular or euclidean distances, on @p tile
 *      over rows [row_start, row_start + row_count): the upper triangle, with the diagonal for dots
 *      and zeros on it for distances, skipping tiles wholly below it.
 *
 *  Takes the parameters of @c nk_define_device_cross_packed_, so one bundle feeds both.
 *
 *  @sa nk_define_cross_symmetric_ for the host original.
 */
#define nk_define_device_cross_symmetric_(metric, input_type_name, isa_suffix, tile, input_value_type,                 \
                                          packed_value_type, result_value_type, depth_simd_dimensions,                 \
                                          dimensions_per_value, ...)                                                   \
    static __global__ void __launch_bounds__(nk_cross_threads_##tile##_k)                                              \
        nk_##metric##s_symmetric_##input_type_name##_##isa_suffix##_kernel_(nk_cross_tile_arguments_t arguments) {     \
        nk_cross_tile_##tile##_(__VA_ARGS__, nk_cross_triangle_upper_k, nk_cross_metric_##metric##_k, &arguments);     \
    }                                                                                                                  \
    NUMKONG_API nk_status_t nk_##metric##s_symmetric_##input_type_name##_##isa_suffix(                                 \
        nk_##input_value_type##_t const *vectors, void const *vector_scales, nk_size_t vectors_count, nk_size_t depth, \
        nk_size_t stride_in_bytes, nk_size_t scales_stride, nk_##result_value_type##_t *result,                        \
        nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count, void *stream) {                    \
        nk_size_t const row_end = row_start + row_count < vectors_count ? row_start + row_count : vectors_count;       \
        return nk_cross_launch_((void const *)nk_##metric##s_symmetric_##input_type_name##_##isa_suffix##_kernel_,     \
                                nk_cross_tile_##tile##_k, nk_cross_threads_##tile##_k, vectors, vectors, 0, result,    \
                                sizeof(nk_##result_value_type##_t), row_start, row_end, vectors_count, depth,          \
                                depth / dimensions_per_value * sizeof(nk_##input_value_type##_t), stride_in_bytes,     \
                                stride_in_bytes, result_stride_in_bytes, stream);                                      \
    }

/** Both shapes of one metric on @p tile, packed and symmetric. */
#define nk_define_device_cross_(metric, input_type_name, isa_suffix, tile, input_value_type, packed_value_type,       \
                                result_value_type, depth_simd_dimensions, dimensions_per_value, ...)                  \
    nk_define_device_cross_packed_(metric, input_type_name, isa_suffix, tile, input_value_type, packed_value_type,    \
                                   result_value_type, depth_simd_dimensions, dimensions_per_value, __VA_ARGS__)       \
    nk_define_device_cross_symmetric_(metric, input_type_name, isa_suffix, tile, input_value_type, packed_value_type, \
                                      result_value_type, depth_simd_dimensions, dimensions_per_value, __VA_ARGS__)

#pragma endregion Cross Macros

#pragma region Baseline Kernels

/*  Each vendor stages depth in the words its own instructions fold: F32 for NVIDIA's FMA, F16 pairs
 *  for AMD's @c v_dot2_f32_f16, and nibbles as packed for AMD's @c v_dot8. */
#if NUMKONG_TARGET_CUDA
nk_define_device_cross_pack_(f64, cuda, f64, f64, nk_load_b8_, f64, 2, 1)
nk_define_device_cross_(dot, f64, cuda, simt_f64, f64, f64, f64, 2, 1, nk_f64_k, nk_cross_accumulation_dot2_k)
nk_define_device_cross_pack_(f32, cuda, f32, f32, nk_load_b8_, f64, 4, 1)
nk_define_device_cross_(dot, f32, cuda, simt_f64, f32, f32, f64, 4, 1, nk_f32_k, nk_cross_accumulation_f64_k)
nk_define_device_cross_pack_(bf16, cuda, bf16, bf16, nk_load_b8_, f32, 8, 1)
nk_define_device_cross_(dot, bf16, cuda, simt_b32, bf16, bf16, f32, 8, 1, nk_bf16_k, nk_cross_accumulation_f32_k)
nk_define_device_cross_pack_(f16, cuda, f16, f16, nk_load_b8_, f32, 8, 1)
nk_define_device_cross_(dot, f16, cuda, simt_b32, f16, f16, f32, 8, 1, nk_f16_k, nk_cross_accumulation_f32_k)
nk_define_device_cross_pack_(e5m2, cuda, e5m2, e5m2, nk_load_b8_, f32, 16, 1)
nk_define_device_cross_(dot, e5m2, cuda, simt_b32, e5m2, e5m2, f32, 16, 1, nk_e5m2_k, nk_cross_accumulation_f32_k)
nk_define_device_cross_pack_(e4m3, cuda, e4m3, e4m3, nk_load_b8_, f32, 16, 1)
nk_define_device_cross_(dot, e4m3, cuda, simt_b32, e4m3, e4m3, f32, 16, 1, nk_e4m3_k, nk_cross_accumulation_f32_k)
nk_define_device_cross_pack_(e3m2, cuda, e3m2, e3m2, nk_load_b8_, f32, 16, 1)
nk_define_device_cross_(dot, e3m2, cuda, simt_b32, e3m2, e3m2, f32, 16, 1, nk_e3m2_k, nk_cross_accumulation_f32_k)
nk_define_device_cross_pack_(e2m3, cuda, e2m3, e2m3, nk_load_b8_, f32, 16, 1)
nk_define_device_cross_(dot, e2m3, cuda, simt_b32, e2m3, e2m3, f32, 16, 1, nk_e2m3_k, nk_cross_accumulation_f32_k)
nk_define_device_cross_pack_(e2m1, cuda, e2m1x2, e2m1x2, nk_load_b8_, f32, 32, 2)
nk_define_device_cross_(dot, e2m1, cuda, simt_b32, e2m1x2, e2m1x2, f32, 32, 2, nk_e2m1_k, nk_cross_accumulation_f32_k)
nk_define_device_cross_pack_(i8, cuda, i8, i8, nk_load_b8_, u32, 16, 1)
nk_define_device_cross_(dot, i8, cuda, simt_b32, i8, i8, i32, 16, 1, nk_i8_k, nk_cross_accumulation_i8x4_k)
nk_define_device_cross_pack_(u8, cuda, u8, u8, nk_load_b8_, u32, 16, 1)
nk_define_device_cross_(dot, u8, cuda, simt_b32, u8, u8, u32, 16, 1, nk_u8_k, nk_cross_accumulation_u8x4_k)
nk_define_device_cross_pack_(i4, cuda, i4x2, i4x2, nk_load_b8_, u32, 32, 2)
nk_define_device_cross_(dot, i4, cuda, simt_b32, i4x2, i4x2, i32, 32, 2, nk_i4_k, nk_cross_accumulation_i4x4_k)
nk_define_device_cross_pack_(u4, cuda, u4x2, u4x2, nk_load_b8_, u32, 32, 2)
nk_define_device_cross_(dot, u4, cuda, simt_b32, u4x2, u4x2, u32, 32, 2, nk_u4_k, nk_cross_accumulation_u4x4_k)
#elif NUMKONG_TARGET_ROCM
nk_define_device_cross_pack_(f64, rocm, f64, f64, nk_load_b8_, f64, 2, 1)
nk_define_device_cross_(dot, f64, rocm, simt_f64, f64, f64, f64, 2, 1, nk_f64_k, nk_cross_accumulation_dot2_k)
nk_define_device_cross_pack_(f32, rocm, f32, f32, nk_load_b8_, f64, 4, 1)
nk_define_device_cross_(dot, f32, rocm, simt_f64, f32, f32, f64, 4, 1, nk_f32_k, nk_cross_accumulation_f64_k)
nk_define_device_cross_pack_(bf16, rocm, bf16, bf16, nk_load_b8_, f32, 8, 1)
nk_define_device_cross_(dot, bf16, rocm, simt_b32, bf16, bf16, f32, 8, 1, nk_bf16_k, nk_cross_accumulation_f32_k)
nk_define_device_cross_pack_(f16, rocm, f16, f16, nk_load_b8_, f32, 8, 1)
nk_define_device_cross_(dot, f16, rocm, simt_b32, f16, f16, f32, 8, 1, nk_f16_k, nk_cross_accumulation_f16x2_k)
nk_define_device_cross_pack_(e5m2, rocm, e5m2, e5m2, nk_load_b8_, f32, 16, 1)
nk_define_device_cross_(dot, e5m2, rocm, simt_b32, e5m2, e5m2, f32, 16, 1, nk_e5m2_k, nk_cross_accumulation_f16x2_k)
nk_define_device_cross_pack_(e4m3, rocm, e4m3, e4m3, nk_load_b8_, f32, 16, 1)
nk_define_device_cross_(dot, e4m3, rocm, simt_b32, e4m3, e4m3, f32, 16, 1, nk_e4m3_k, nk_cross_accumulation_f16x2_k)
nk_define_device_cross_pack_(e3m2, rocm, e3m2, e3m2, nk_load_b8_, f32, 16, 1)
nk_define_device_cross_(dot, e3m2, rocm, simt_b32, e3m2, e3m2, f32, 16, 1, nk_e3m2_k, nk_cross_accumulation_f16x2_k)
nk_define_device_cross_pack_(e2m3, rocm, e2m3, e2m3, nk_load_b8_, f32, 16, 1)
nk_define_device_cross_(dot, e2m3, rocm, simt_b32, e2m3, e2m3, f32, 16, 1, nk_e2m3_k, nk_cross_accumulation_f16x2_k)
nk_define_device_cross_pack_(e2m1, rocm, e2m1x2, e2m1x2, nk_load_b8_, f32, 32, 2)
nk_define_device_cross_(dot, e2m1, rocm, simt_b32, e2m1x2, e2m1x2, f32, 32, 2, nk_e2m1_k, nk_cross_accumulation_f16x2_k)
nk_define_device_cross_pack_(i8, rocm, i8, i8, nk_load_b8_, u32, 16, 1)
nk_define_device_cross_(dot, i8, rocm, simt_b32, i8, i8, i32, 16, 1, nk_i8_k, nk_cross_accumulation_i8x4_k)
nk_define_device_cross_pack_(u8, rocm, u8, u8, nk_load_b8_, u32, 16, 1)
nk_define_device_cross_(dot, u8, rocm, simt_b32, u8, u8, u32, 16, 1, nk_u8_k, nk_cross_accumulation_u8x4_k)
nk_define_device_cross_pack_(i4, rocm, i4x2, i4x2, nk_load_b8_, u32, 32, 2)
nk_define_device_cross_(dot, i4, rocm, simt_b32, i4x2, i4x2, i32, 32, 2, nk_i4_k, nk_cross_accumulation_i4x8_k)
nk_define_device_cross_pack_(u4, rocm, u4x2, u4x2, nk_load_b8_, u32, 32, 2)
nk_define_device_cross_(dot, u4, rocm, simt_b32, u4x2, u4x2, u32, 32, 2, nk_u4_k, nk_cross_accumulation_u4x8_k)
#endif

#pragma endregion Baseline Kernels

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_ || NUMKONG_ARCH_ROCM_
#endif // NUMKONG_DOTS_SIMT_CUH
